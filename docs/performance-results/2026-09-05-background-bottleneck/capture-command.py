import subprocess,time,pathlib,json,hashlib
root=pathlib.Path('/Users/jens/repos/mechanic')
out=pathlib.Path('/tmp/mechanic-background-bottleneck-01')
assert subprocess.run(['pgrep','-x','mechanic-app'],capture_output=True).returncode==1,'game already running'
with open('/tmp/mechanic-background-bottleneck-launch.log','w') as log:
 p=subprocess.Popen(['python3','scripts/run-background-capture.py','--world','/Users/jens/Library/Application Support/Mechanic/worlds/test4','--output',str(out)],cwd=root,stdout=log,stderr=subprocess.STDOUT)
 deadline=time.monotonic()+355
 try:
  while p.poll() is None and time.monotonic()<deadline:
   f=out/'app.log'
   if f.exists() and 'Performance capture started (60 seconds)' in f.read_text():
    pid=subprocess.check_output(['pgrep','-P',str(p.pid),'-x','mechanic-app'],text=True).strip()
    for n in (1,2):
     subprocess.run(['sample',pid,'5','1','-file',str(out/f'cpu-{n}.txt')],check=True,timeout=20)
     time.sleep(5)
    break
   time.sleep(1)
  p.wait(timeout=max(1,deadline-time.monotonic()))
  if p.returncode: raise RuntimeError(f'runner exited {p.returncode}')
  fingerprints={str(f.relative_to(root)):hashlib.sha256(f.read_bytes()).hexdigest() for directory in ['crates','vendor','scripts'] for f in (root/directory).rglob('*') if f.is_file() and '.git' not in f.parts}
  for f in [root/'Cargo.toml',root/'Cargo.lock',root/'target/release/mechanic-app']:
   fingerprints[str(f.relative_to(root))]=hashlib.sha256(f.read_bytes()).hexdigest()
  (out/'fingerprints.json').write_text(json.dumps(fingerprints,indent=2))
 finally:
  if p.poll() is None:
   p.terminate()
   p.wait(timeout=10)
print(out)
