import os,pathlib,subprocess,time,shutil,json,hashlib
root=pathlib.Path('/tmp/mechanic-fence-release')
worlds=pathlib.Path('/Users/jens/Library/Application Support/Mechanic/worlds')
captures=pathlib.Path('/tmp/mechanic-fence-captures')
def osa(*lines):
    args=['osascript']
    for line in lines: args += ['-e',line]
    return subprocess.run(args,check=True,capture_output=True,text=True).stdout

def close(name):
    osa(f'tell application "System Events" to tell process "{name}"', 'click (first button of window 1 whose subrole is "AXCloseButton")','end tell')

def wait_for(log,needle,timeout):
    deadline=time.monotonic()+timeout
    while time.monotonic()<deadline:
        if needle in log.read_text(): return
        time.sleep(1)
    raise RuntimeError(f'timed out waiting for {needle} in {log}')

# The first baseline was set up and visually verified interactively.
log=pathlib.Path('/tmp/mechanic-fence-A1.log')
wait_for(log,'Performance capture: /',150)
subprocess.run(['sample','mechanic-app-baseline','5','1','-file','/tmp/mechanic-fence-A1-profile.txt'],check=True)
close('mechanic-app-baseline')
time.sleep(4)
for variant,label in [('candidate','B1'),('candidate','B2'),('baseline','A2')]:
    dst=worlds/'test4-perf'
    assert dst.name=='test4-perf' and dst.parent==worlds
    shutil.rmtree(dst)
    shutil.copytree(worlds/'test4',dst)
    manifest=dst/'world.ron'
    manifest.write_text(manifest.read_text().replace('name: "TEST4",','name: "TEST4 PERF",',1))
    env=os.environ.copy()
    env.update(BEVY_ASSET_ROOT=str(root/variant/'crates/mechanic-app'),MECHANIC_RENDER_EXPERIMENT='baseline',MECHANIC_PERF_CAPTURE_DIR=str(captures),MECHANIC_PERF_LABEL=f'TEST4-offscreen-{label}')
    name=f'mechanic-app-{variant}'
    log=pathlib.Path(f'/tmp/mechanic-fence-{label}.log')
    with log.open('w') as output:
        process=subprocess.Popen([str(root/name)],env=env,stdout=output,stderr=subprocess.STDOUT)
        try:
            wait_for(log,'Creating new window',40)
            time.sleep(4)
            osa(f'tell application "System Events" to tell process "{name}"','set frontmost to true','set position of window 1 to {0, 39}','set size of window 1 to {2056, 1290}','end tell')
            time.sleep(2)
            subprocess.run(['/tmp/mechanic-click','600','407'],check=True)
            time.sleep(30)
            osa(f'tell application "System Events" to tell process "{name}"','key code 99','delay 0.3','key code 101','end tell')
            print(f'{label}: armed',flush=True)
            wait_for(log,'Performance capture started',180)
            print(f'{label}: capture started',flush=True)
            wait_for(log,'Performance capture: /',90)
            print(f'{label}: capture finished',flush=True)
            subprocess.run(['screencapture','-x',f'/tmp/mechanic-fence-{label}.png'],check=True)
            subprocess.run(['sample',str(process.pid),'5','1','-file',f'/tmp/mechanic-fence-{label}-profile.txt'],check=True)
        finally:
            close(name)
            process.wait(timeout=30)
    time.sleep(4)
print('A/B/B/A completed',flush=True)
