import os,pathlib,subprocess,shutil,json,uuid,hashlib
root=pathlib.Path('/Users/jens/repos/mechanic');world=pathlib.Path('/Users/jens/Library/Application Support/Mechanic/worlds/test4');output=pathlib.Path('/tmp/mechanic-attribution-pair-01')
assert subprocess.run(['pgrep','-x','mechanic-app'],capture_output=True).returncode==1
frozen=world.parent/('mechanic-attribution-source-'+uuid.uuid4().hex)
try:
 shutil.copytree(world,frozen)
 hashes={str(p.relative_to(frozen)):hashlib.sha256(p.read_bytes()).hexdigest() for p in frozen.rglob('*') if p.is_file()}
 assert hashes==json.loads((output/'world-fingerprints.json').read_text()),'source changed; cannot match first run'
 env=os.environ.copy();env['MECHANIC_PERF_TERRAIN_PASSES']='1'
 subprocess.run(['python3','scripts/run-background-capture.py','--world',str(frozen),'--output',str(output/'partitioned')],cwd=root,env=env,check=True,timeout=390)
 r=json.loads((output/'partitioned/summary.json').read_text())[0]
 assert r['metadata']['terrain_pass_partition']
 assert r['render_gpu_ms']['terrain_ms']['count']>0
 print('partitioned verified',r['focused_frame_counts'])
finally:shutil.rmtree(frozen)
