"""Audit final matched replays, including warm-up and repeat determinism."""
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
identities = {name: json.loads((root / f'{name}-identity.json').read_text())
              for name in ('baseline', 'candidate')}
protocols = [('cold', [1, 2, 5, 10], range(1, 4), 0, 120),
             ('connected', [1, 2, 5, 10], range(1, 4), 600, 3600),
             ('packed-separated-check', [10], range(1, 2), 600, 3600),
             ('separated', [10], range(1, 3), 600, 3600)]
observable = ['state_hash', 'contacts', 'degraded', 'degraded_reason', 'penetration_m',
              'closure_gap_m', 'closure_angle_rad', 'triangle_candidates',
              'collider_pair_candidates', 'continuous_triangle_candidates',
              'continuous_collider_pair_candidates', 'continuous_separation_evaluations']
repeated = {}
paired_ticks = 0
pairs = []
for folder, copies, repeats, warmup, measured in protocols:
    directory = root / folder
    manifest = json.loads((directory / 'manifest.json').read_text())
    assert (manifest['warmup_ticks'], manifest['measured_ticks']) == (warmup, measured)
    for name, identity in identities.items():
        binary = manifest['binaries'][name]
        assert binary['sha256'] == identity['binary_sha256']
        assert hashlib.sha256(Path(binary['path']).read_bytes()).hexdigest() == binary['sha256']
    assembly = 'separated' if 'separated' in folder else 'connected'
    for count in copies:
        for repeat in repeats:
            traces = []
            for name in ('baseline', 'candidate'):
                label = f'{count}x-{assembly}-{name}-{repeat}'
                run = next(r for r in manifest['runs'] if r['label'] == label)
                assert run['exit_code'] == 0 and not run['no_progress_timeout']
                records = [json.loads(line) for line in (directory / f'{label}.jsonl').read_text().splitlines()]
                metadata, summary = records[0], records[-1]
                assert metadata['copies'] == count and metadata['connected'] == (assembly == 'connected')
                assert not metadata['hold'] and metadata['route'] == 'cpu'
                assert metadata['terrain'] == 'saved-seed-leaf-mesh'
                assert summary['kind'] == 'summary' and not summary['degraded_ticks']
                assert not summary['outside_terrain_region']
                ticks = [r for r in records if r.get('kind') == 'tick']
                assert [t['tick'] for t in ticks] == list(range(1, warmup + measured + 1))
                assert all(t['warmup'] == (t['tick'] <= warmup) and not t['degraded'] for t in ticks)
                trace = [[t[key] for key in observable] for t in ticks]
                key = (assembly, count, name, warmup, measured)
                assert trace == repeated.setdefault(key, trace), (folder, label, 'repeat differs')
                traces.append(trace)
            assert traces[0] == traces[1], (folder, count, repeat, 'paired observable differs')
            paired_ticks += len(traces[0])
            pairs.append({'folder': folder, 'copies': count, 'repeat': repeat, 'paired_ticks': len(traces[0])})
result = {'passed': True, 'paired_ticks': paired_ticks, 'pairs': pairs,
          'matched_observables': observable, 'repeat_determinism': True,
          'includes_warmup': True, 'degraded_ticks': 0, 'incomplete_runs': 0}
(root / 'evidence-audit.json').write_text(json.dumps(result, indent=2) + '\n')
print(f'PASS: {paired_ticks:,} paired ticks; {len(pairs)} comparisons; all observables and repeats match')
