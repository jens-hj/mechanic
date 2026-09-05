import os,pathlib,subprocess,shutil,json,uuid,hashlib
root=pathlib.Path('/Users/jens/repos/mechanic')
world=pathlib.Path('/Users/jens/Library/Application Support/Mechanic/worlds/test4')
output=pathlib.Path('/tmp/mechanic-attribution-pair-01')
assert subprocess.run(['pgrep','-x','mechanic-app'],capture_output=True).returncode==1,'game already running'
output.mkdir()
frozen=world.parent/('mechanic-attribution-source-'+uuid.uuid4().hex)
try:
 shutil.copytree(world,frozen)
 source_hashes={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*') if p.is_file()}
 (output/'world-fingerprints.json').write_text(json.dumps(source_hashes,indent=2))
 fingerprints={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['crates','vendor','scripts'] for p in (root/folder).rglob('*') if p.is_file() and '__pycache__' not in p.parts}
 for p in [root/'Cargo.toml',root/'Cargo.lock',root/'target/release/mechanic-app']:fingerprints[str(p.relative_to(root))]=hashlib.sha256(p.read_bytes()).hexdigest()
 (output/'source-fingerprints.json').write_text(json.dumps(fingerprints,indent=2))
 for name,flag in [('normal','0'),('partitioned','1')]:
  env=os.environ.copy();env['MECHANIC_PERF_TERRAIN_PASSES']=flag
  subprocess.run(['python3','scripts/run-background-capture.py','--world',str(frozen),'--output',str(output/name)],cwd=root,env=env,check=True,timeout=390)
  result=json.loads((output/name/'summary.json').read_text())[0]
  assert result['focused_frame_counts']['true']==0
  assert result['metadata']['terrain_pass_partition']==(flag=='1')
  assert result['submission_to_callbacks_ms']['count']>0
  assert result['callback_to_publication_ms']['count']>0
  if flag=='1':assert result['render_gpu_ms']['terrain_ms']['count']>0
  print(name,'verified',flush=True)
finally:
 shutil.rmtree(frozen)
print(output)
