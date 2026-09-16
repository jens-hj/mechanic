from pathlib import Path
import subprocess,json,hashlib
out=Path('docs/performance-results/2026-09-16-wishbone')
base=['target/release/cpu-physics','--scenario','pipe-scene','--instance','docs/performance-results/2026-09-16-distant-scene/new-instance-8.ron','--warmup','120','--ticks','600']
scenes={'alone':None,'wishbone':out/'wishbone-instance.ron','builder-with':out/'builder-instance-171.ron','builder-without':out/'builder-instance-172.ron'}
(out/'manifest.json').write_text(json.dumps({'binary_sha256':hashlib.sha256(Path(base[0]).read_bytes()).hexdigest(),'command':base,'scenes':{k:str(v) for k,v in scenes.items()},'runs':3},indent=2))
for run in range(1,4):
 for name in (list(scenes) if run%2 else list(reversed(scenes))):
  cmd=base+(['--background',str(scenes[name])] if scenes[name] else [])
  with (out/f'{name}-{run}.jsonl').open('w') as f:subprocess.run(cmd,stdout=f,check=True)
  print(run,name,flush=True)
