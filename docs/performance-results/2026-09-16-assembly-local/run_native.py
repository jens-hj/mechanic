"""Fixed-camera CPU captures of disposable restored saved-world copies.

Run after building both source-matched app binaries, with no compilation,
profiling, or headless benchmark running. Raw captures remain in the ignored
.physics-reference directory; compact summaries and screenshots accompany this
report. Capture-start hammer delivery is logged, including any stability clamp.
"""
import json
import os
import hashlib
from pathlib import Path
import shutil
import subprocess
import sys

report = Path(__file__).resolve().parent
reference = Path('.physics-reference')
fast = os.environ.get('MECHANIC_NATIVE_MODE') == 'fast'
prefix = 'fast-' if fast else ''
failed = []
binaries = {
    'baseline': reference / 'assembly-local-baseline/mechanic-app-hammer',
    'candidate': Path('target/release/mechanic-app'),
}
identities = {
    'baseline': reference / 'assembly-local-baseline/app-identity.json',
    'candidate': report / 'candidate-app-identity.json',
}
assert hashlib.sha256(binaries['baseline'].read_bytes()).digest() != hashlib.sha256(binaries['candidate'].read_bytes()).digest(), 'baseline and candidate must be different executables'
selected = os.environ.get('MECHANIC_NATIVE_CASES', '').split(',')
for world, body, order in [('new', 1, ['baseline', 'candidate']),
                           ('builder', 18, ['candidate', 'baseline'])]:
    for name in order:
        if selected != [''] and f'{world}:{name}' not in selected:
            continue
        output = reference / f'assembly-local-native-{prefix}{world}-{name}'
        hammer = {'body_index': body, 'local_point': [-0.25, 0, -0.375],
                  'impulse': [0, 200, 0]}
        if fast:
            # Tangential pulses overcome the pipe's gravitational restoring
            # torque while each pulse still uses the normal delivery calculation.
            hammer.update(repeat=128, impulse=[0, 3, 0], body_local_impulse=True)
        command = [sys.executable, 'scripts/run-background-capture.py',
                   '--binary', str(binaries[name]), '--identity', str(identities[name]),
                   '--world', str(reference / 'assembly-local-worlds' / world),
                   '--output', str(output), '--assets', 'crates/mechanic-app',
                   '--physics', 'cpu', '--foreground', '--replay-ticks', '600',
                   '--hammer', json.dumps(hammer)]
        result = subprocess.run(command, check=False)
        if result.returncode:
            failed.append((world, name))
        for filename in ['summary.json', 'foreground-protocol.json', 'failure.json']:
            if (output / filename).is_file():
                shutil.copy2(output / filename, report / f'native-{prefix}{world}-{name}-{filename}')
        for capture in output.glob('capture-*.jsonl'):
            events = [json.loads(line) for line in capture.read_text().splitlines()]
            strikes = [row for row in events if row.get('kind') == 'scripted_hammer']
            if fast and not sum(row['data']['continuous_sweeps'] for row in events
                                if row['kind'] == 'physics_cpu_tick'):
                failed.append((world, name, 'no continuous sweeps'))
            (report / f'native-{prefix}{world}-{name}-hammer.json').write_text(
                json.dumps(strikes, indent=2) + '\n')
            if capture.with_suffix('.png').is_file():
                shutil.copy2(capture.with_suffix('.png'), report / f'native-{prefix}{world}-{name}.png')
        print(world, name, result.returncode, flush=True)

if failed:
    raise SystemExit(f"Failed captures: {failed}")
