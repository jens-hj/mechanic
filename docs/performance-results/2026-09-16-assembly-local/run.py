"""Three alternating release pairs using immutable saved instance snapshots."""
import hashlib
import json
import os
from pathlib import Path
import statistics
import subprocess

root = Path(os.environ.get('MECHANIC_ASSEMBLY_OUTPUT', Path(__file__).resolve().parent))
root.mkdir(parents=True, exist_ok=True)
binary = {'baseline': Path(os.environ.get('MECHANIC_ASSEMBLY_BASELINE', '.physics-reference/assembly-local-baseline/cpu-physics')).resolve(),
          'candidate': Path(os.environ.get('MECHANIC_ASSEMBLY_CANDIDATE', 'target/release/cpu-physics')).resolve()}
fixtures = Path('docs/performance-results/2026-09-16-wishbone')
scenes = {'alone': None, 'wishbone': fixtures / 'wishbone-instance.ron',
          'builder-with': fixtures / 'builder-instance-171.ron',
          'builder-without': fixtures / 'builder-instance-172.ron'}
pipe = Path('docs/performance-results/2026-09-16-distant-scene/new-instance-8.ron')
hash_file = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
manifest = {'binaries': {name: {'path':str(path), 'sha256': hash_file(path)} for name,path in binary.items()},
            'fixtures': {str(path):hash_file(path) for path in [pipe, *filter(None, scenes.values())]},
            'speeds':os.environ.get('MECHANIC_PIPE_SPEEDS', '0,1,10,40'),
            'pairs':3, 'warmup':120, 'ticks':600, 'order':[['baseline','candidate'],['candidate','baseline'],['baseline','candidate']]}
assert manifest['binaries']['baseline']['sha256'] != manifest['binaries']['candidate']['sha256'], 'baseline and candidate must be different executables'
(root / 'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
records = {}
for pair, order in enumerate(manifest['order'],1):
    for scene, background in scenes.items():
        for name in order:
            command = [str(binary[name]), '--scenario', 'pipe-scene', '--instance', str(pipe), '--warmup','120','--ticks','600']
            if background: command.extend(['--background',str(background)])
            result = subprocess.run(command,check=True,text=True,capture_output=True)
            (root / f'{scene}-{name}-{pair}.jsonl').write_text(result.stdout)
            records.setdefault(scene,{}).setdefault(name,[]).append([json.loads(line) for line in result.stdout.splitlines() if '"kind":"metadata"' not in line])
            print(pair,scene,name,flush=True)
summary=[]
for scene, values in records.items():
    for index, row in enumerate(values['baseline'][0]):
        before=[run[index] for run in values['baseline']]
        after=[run[index] for run in values['candidate']]
        summary.append({'scene':scene,'speed':row['speed'],
            'baseline_p50_ms':statistics.median(r['p50_ms'] for r in before),
            'candidate_p50_ms':statistics.median(r['p50_ms'] for r in after),
            'baseline_p95_ms':statistics.median(r['p95_ms'] for r in before),
            'candidate_p95_ms':statistics.median(r['p95_ms'] for r in after),
            'ratios':[b['p50_ms']/a['p50_ms'] for a,b in zip(before,after)],
            'baseline_degraded':sum(r['degraded_ticks'] for r in before),
            'candidate_degraded':sum(r['degraded_ticks'] for r in after),
            'pipe_speed_difference':max(abs(a['final_speed']-b['final_speed']) for a,b in zip(before,after)),
            'state_hashes_match':all(a['state_hash']==b['state_hash'] for a,b in zip(before,after))})
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
