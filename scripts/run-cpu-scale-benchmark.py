#!/usr/bin/env python3
"""Record matched CPU builder replays, retaining raw ticks and binary identities."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import os
import math
import subprocess
import shutil
import time


def digest(path):
    with path.open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def run_until_idle(command, output, idle_seconds):
    """Bound a stalled tick, while allowing long runs that keep completing ticks."""
    with subprocess.Popen(command, stdout=output) as process:
        last_size = os.fstat(output.fileno()).st_size
        last_progress = time.monotonic()
        while True:
            try:
                code = process.wait(timeout=min(10.0, idle_seconds))
                return subprocess.CompletedProcess(command, code), False
            except subprocess.TimeoutExpired:
                size = os.fstat(output.fileno()).st_size
                now = time.monotonic()
                if size != last_size:
                    last_size, last_progress = size, now
                elif now - last_progress >= idle_seconds:
                    process.kill()
                    return subprocess.CompletedProcess(command, process.wait()), True


def run(args):
    args.output.mkdir(parents=True, exist_ok=False)
    binaries = {name: path.resolve() for name, path in [('baseline', args.baseline), ('candidate', args.candidate)]}
    manifest = {'machine': platform.platform(), 'architecture': platform.machine(),
                'warmup_ticks': args.warmup, 'measured_ticks': args.ticks, 'repeats': args.repeats, 'assemblies': args.assemblies,
                'binaries': {name: {'path': str(path), 'sha256': digest(path)} for name, path in binaries.items()},
                'runs': [], 'no_progress_timeout_seconds': args.no_progress_seconds}
    manifest_path = args.output / 'manifest.json'
    reused = None
    if args.reuse_baseline:
        reused = json.loads((args.reuse_baseline / 'manifest.json').read_text())
        if reused['binaries']['baseline']['sha256'] != manifest['binaries']['baseline']['sha256']:
            raise ValueError('reused baseline binary differs')
        if (reused['warmup_ticks'], reused['measured_ticks']) != (args.warmup, args.ticks):
            raise ValueError('reused baseline tick protocol differs')
    failed = False
    for copies in args.copies:
        for assembly in args.assemblies:
            connected = assembly == "connected"
            for repeat in range(args.repeats):
                # Alternate order to expose drift rather than favor one binary.
                order = list(binaries) if repeat % 2 == 0 else list(reversed(binaries))
                for name in order:
                    label = f'{copies}x-{"connected" if connected else "separated"}-{name}-{repeat+1}'
                    command = [str(binaries[name]), '--scenario', 'builder-scale', '--copies', str(copies),
                               '--warmup', str(args.warmup), '--ticks', str(args.ticks)]
                    if connected:
                        command.append('--connected')
                    if args.floor:
                        command.append('--floor')
                    previous = args.reuse_baseline / f'{label}.jsonl' if args.reuse_baseline and name == 'baseline' else None
                    if previous and previous.exists():
                        records = [json.loads(line) for line in previous.read_text().splitlines()]
                        expected_terrain = 'diagnostic-floor' if args.floor else 'saved-seed-leaf-mesh'
                        if (records[0]['copies'], records[0]['connected'], records[0]['terrain'], records[0]['hold'], records[0]['route']) != (copies, connected, expected_terrain, False, 'cpu'):
                            raise ValueError('reused baseline fixture protocol differs')
                        if records[-1].get('kind') != 'summary' or sum(r.get('kind') == 'tick' for r in records) != args.warmup + args.ticks:
                            raise ValueError('reused baseline is incomplete')
                        original = next(r for r in reused['runs'] if r['label'] == label)
                        if original['exit_code']:
                            raise ValueError('reused baseline failed')
                        shutil.copyfile(previous, args.output / previous.name)
                        manifest['runs'].append({**original, 'copied_from': str(previous.resolve()),
                                                 'note': 'Historical baseline; compilation may separate this run from candidate.'})
                        manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
                        print(label + ' (reused)', flush=True)
                        continue
                    started = time.time()
                    print(label, flush=True)
                    with (args.output / f'{label}.jsonl').open('w') as output:
                        result, stalled = run_until_idle(command, output, args.no_progress_seconds)
                    manifest['runs'].append({'label': label, 'command': command, 'started_unix': started,
                                             'elapsed_seconds': time.time()-started, 'exit_code': result.returncode,
                                             'no_progress_timeout': stalled})
                    manifest_path.write_text(json.dumps(manifest, indent=2)+'\n')
                    if result.returncode:
                        failed = True
                        print(f'{label} failed; partial evidence retained, continuing', flush=True)
    summary = {}
    for path in sorted(args.output.glob('*.jsonl')):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        summary[path.stem] = records[-1] if records and records[-1].get('kind') == 'summary' else {
            'kind': 'incomplete', 'completed_ticks': sum(r.get('kind') == 'tick' for r in records),
            'last_record': records[-1] if records else None}
    def replay_states(label):
        records = [json.loads(line) for line in (args.output / f'{label}.jsonl').read_text().splitlines()]
        return [(r['tick'], r.get('state_hash')) for r in records if r.get('kind') == 'tick']

    comparisons = {}
    for label, baseline in summary.items():
        if '-baseline-' not in label:
            continue
        candidate_label = label.replace('-baseline-', '-candidate-')
        candidate = summary[candidate_label]
        if baseline.get('kind') != 'summary' or candidate.get('kind') != 'summary':
            comparisons[candidate_label] = {'quality_within_baseline_tolerances': None,
                                            'headless_gate_passed': False, 'reason': 'incomplete replay'}
            continue
        if any(r.get('degraded_ticks', 0) or r.get('outside_terrain_region', False) for r in (baseline, candidate)):
            comparisons[candidate_label] = {'quality_within_baseline_tolerances': None,
                                            'headless_gate_passed': False, 'reason': 'degraded or escaped replay'}
            continue
        baseline_states = replay_states(label)
        candidate_states = replay_states(candidate_label)
        exact_states = (len(baseline_states) == args.warmup + args.ticks
                        and all(state is not None for _, state in baseline_states)
                        and baseline_states == candidate_states)
        quality = (candidate['maximum_penetration_m'] <= baseline['maximum_penetration_m'] + 0.001
                   and candidate['closure_gap_m'] <= baseline['closure_gap_m'] + 0.001
                   and candidate['closure_angle_rad'] <= baseline['closure_angle_rad'] + 0.01)
        comparisons[candidate_label] = {'matching_replay_state_hashes': exact_states,
                                        'quality_within_baseline_tolerances': quality,
                                        'headless_gate_passed': candidate['timing_gate'] and quality and exact_states,
                                        'app_scheduler_and_rendering_verified': False}
    (args.output / 'summary.json').write_text(json.dumps({'runs': summary, 'comparisons': comparisons}, indent=2)+'\n')
    if failed:
        raise RuntimeError('One or more replays failed; summaries and partial evidence retained')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--copies', type=int, nargs='+', choices=[1,2,5,10], default=[1,2,5,10])
    parser.add_argument('--assemblies', nargs='+', choices=['separated', 'connected'], default=['separated', 'connected'])
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--warmup', type=int, default=600)
    parser.add_argument('--ticks', type=int, default=3600)
    parser.add_argument('--no-progress-seconds', type=float, default=300, help='Stop a replay only if it produces no completed-tick output for this long')
    parser.add_argument('--reuse-baseline', type=Path, help='Reuse complete baseline runs after validating binary hash and fixture protocol')
    parser.add_argument('--floor', action='store_true', help='Use diagnostic finite floor instead of saved terrain')
    options = parser.parse_args()
    if options.repeats < 1 or options.ticks < 1 or options.warmup < 0 or not math.isfinite(options.no_progress_seconds) or options.no_progress_seconds <= 0:
        parser.error('repeats/ticks must be positive; warmup must be nonnegative')
    run(options)
