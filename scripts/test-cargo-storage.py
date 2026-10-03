#!/usr/bin/env python3
"""Exercise real processes and OS locks without compiling the workspace."""
import importlib.util
import json
import os
from pathlib import Path
import signal
import shutil
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

sys.dont_write_bytecode = True
from cargo_storage import Store, require_binary_lease, size
from cargo_process import is_msvc_service

LAUNCHER = Path(__file__).with_name('cargo-storage.py').resolve()
TOOLCHAIN = (subprocess.check_output(['rustup', 'show', 'active-toolchain'],
                                   cwd=LAUNCHER.parent.parent, text=True).split()[0]
             if shutil.which('rustup') else None)

spec = importlib.util.spec_from_file_location('launcher', LAUNCHER)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


class StorageTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve() / 'store'
        self.env = {k: v for k, v in os.environ.items()
                    if k not in ('CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET_DIR', 'CARGO_BUILD_BUILD_DIR',
                                 'MECHANIC_CARGO_LEASE', 'MECHANIC_CARGO_STORAGE')}
        self.env['PYTHONFAULTHANDLER'] = '1'
        if TOOLCHAIN:
            self.env['RUSTUP_TOOLCHAIN'] = TOOLCHAIN
        self.children = []
        self.addCleanup(self.stop_children)

    def stop_children(self):
        for child in self.children:
            if child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            child.stdout.close()
            child.stderr.close()

    def invoke(self, *args):
        try:
            return subprocess.run([sys.executable, str(LAUNCHER), '--root', str(self.root), *args],
                                  env=self.env, text=True, capture_output=True,
                                  timeout=60 if args[:1] == ('cargo',) else 10)
        except subprocess.TimeoutExpired as error:
            self.fail(f'{error}\nstdout: {error.stdout!r}\nstderr: {error.stderr!r}')

    def start(self, source, cwd=None):
        child = subprocess.Popen([sys.executable, str(LAUNCHER), '--root', str(self.root),
                                  'run', '--', sys.executable, '-c', source],
                                 env=self.env, cwd=cwd, text=True, stdout=subprocess.PIPE,
                                 stderr=subprocess.PIPE)
        self.children.append(child)
        return child

    def wait_for(self, predicate):
        deadline = time.monotonic() + 8
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(.03)
        self.fail('condition did not become true')

    def hold(self, name):
        marker = self.root.parent / name
        release = self.root.parent / (name + '-release')
        source = (f'from pathlib import Path; import time; Path({str(marker)!r}).touch(); '
                  f'\nwhile not Path({str(release)!r}).exists(): time.sleep(.03)')
        return self.start(source), marker, release

    def test_two_workers_run_and_third_queues_until_release(self):
        first, a, release = self.hold('first')
        second, b, release_b = self.hold('second')
        self.wait_for(lambda: a.exists() and b.exists())
        third, c, release_c = self.hold('third')
        time.sleep(.3)
        self.assertFalse(c.exists())
        self.assertIsNone(third.poll())
        release.touch()
        self.assertEqual(first.wait(timeout=5), 0)
        self.wait_for(c.exists)
        release_b.touch()
        release_c.touch()
        self.assertEqual(second.wait(timeout=5), 0)
        self.assertEqual(third.wait(timeout=5), 0)

    def test_nested_launcher_reuses_the_only_slot_and_preserves_exit(self):
        Store(self.root, 1)
        source = (f'import subprocess,sys; sys.exit(subprocess.call([sys.executable, '
                  f'{str(LAUNCHER)!r}, "run", "--", sys.executable, "-c", "raise SystemExit(23)"]))')
        child = self.start(source)
        self.assertEqual(child.wait(timeout=8), 23)
        self.assertEqual(Store(self.root).state(0)['state'], 'idle')

    def test_cleanup_skips_active_lease_and_preserves_reports(self):
        store = Store(self.root, 1)
        target = store.target(0)
        cache = target / 'debug/incremental'
        cache.mkdir(parents=True)
        (cache / 'generated').write_bytes(b'x' * 4096)
        report = target / 'debug/report.json'
        report.write_text('retain me')
        store.save(0, {'state': 'idle', 'finished': 0})
        with store.lock(0) as lock:
            self.assertTrue(lock.acquire())
            result = self.invoke('clean', '--idle-hours', '0', '--apply')
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(cache.exists())
            self.assertIn('active lease', result.stdout)
        result = self.invoke('clean', '--idle-hours', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(cache.exists())
        result = self.invoke('clean', '--idle-hours', '0', '--apply')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(cache.exists())
        self.assertEqual(report.read_text(), 'retain me')

    def test_cleanup_estimate_counts_hardlinks_once_and_excludes_retained_links(self):
        store = Store(self.root, 1)
        target = store.target(0)
        deps = target / 'debug/deps'
        incremental = target / 'debug/incremental'
        deps.mkdir(parents=True)
        incremental.mkdir()
        original = deps / 'object.o'
        original.write_bytes(b'x' * 8192)
        os.link(original, incremental / 'object.o')
        file_size = size(original)
        self.assertEqual(size(target), file_size)
        result = self.invoke('clean', '--idle-hours', '0')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['reclaimable_file_bytes_estimate'], file_size)
        protected = target / 'debug/retained-binary'
        os.link(original, protected)
        result = self.invoke('clean', '--idle-hours', '0', '--apply')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)['reclaimable_file_bytes_estimate'], 0)
        self.assertEqual(protected.read_bytes(), b'x' * 8192)
        self.assertEqual(size(target), file_size)

    def test_application_named_like_compiler_service_is_not_reaped(self):
        self.assertFalse(is_msvc_service(r'C:\build\vctip.exe'))
        self.assertFalse(is_msvc_service(r'C:\build\mspdbsrv.exe'))
        self.assertFalse(is_msvc_service(r'C:\VS\VC\Tools\MSVC\14.51\bin\cl.exe'))
        self.assertTrue(is_msvc_service(r'C:\VS\VC\Tools\MSVC\14.51\bin\vctip.exe'))
        self.assertTrue(is_msvc_service(r'C:\VS\VC\Tools\MSVC\14.51\bin\mspdbsrv.exe'))

    def test_conflicting_count_and_target_override_are_refused(self):
        Store(self.root, 1)
        with self.assertRaises(ValueError):
            Store(self.root, 2)
        with self.assertRaisesRegex(ValueError, 'slot count must be'):
            Store(self.root.parent / 'invalid-count', 0)
        result = self.invoke('cargo', 'build', '--target-dir=/tmp/foreign')
        self.assertNotEqual(result.returncode, 0)

    def test_unclean_state_is_not_reclaimed_by_missing_pid(self):
        store = Store(self.root, 1)
        store.save(0, {'state': 'running', 'pid': 99999999, 'boot': None})
        result = self.invoke('recover')
        self.assertEqual(result.returncode, 1, result.stderr)
        self.assertEqual(store.state(0)['state'], 'running')
        self.assertIn('quarantined', self.invoke('clean', '--idle-hours', '0', '--apply').stdout)

    @unittest.skipIf(os.name == 'nt', 'POSIX signal semantics')
    def test_cancellation_waits_for_child_exit_and_releases_slot(self):
        Store(self.root, 1)
        child, marker, _ = self.hold('cancel')
        self.wait_for(marker.exists)
        child.send_signal(signal.SIGTERM)
        self.assertEqual(child.wait(timeout=5), 143)
        self.assertEqual(Store(self.root).state(0)['state'], 'idle')

    @unittest.skipIf(os.name == 'nt', 'POSIX SIGKILL; Windows job kills children on supervisor exit')
    def test_killed_supervisor_quarantines_live_orphan(self):
        Store(self.root, 1)
        source = self.root.parent / 'crash-fixture'
        (source / 'src').mkdir(parents=True)
        (source / 'Cargo.toml').write_text(
            '[package]\nname="crash-fixture"\nversion="0.1.0"\nedition="2021"\n[workspace]\n')
        (source / 'src/main.rs').write_text('fn main() {}')
        build = self.invoke('cargo', 'build', '--quiet', '--offline',
                            '--manifest-path', str(source / 'Cargo.toml'))
        self.assertEqual(build.returncode, 0, build.stderr)
        child, marker, release = self.hold('crash')
        self.wait_for(marker.exists)
        stale = Store(self.root).target(0) / 'debug/stale-artifact'
        stale.parent.mkdir(parents=True, exist_ok=True)
        stale.write_text('uncertain checkout output')
        child.kill()
        child.wait(timeout=5)
        try:
            result = self.invoke('status')
            self.assertTrue(json.loads(result.stdout)['slots'][0]['quarantined'])
            result = self.invoke('clean', '--idle-hours', '0', '--apply')
            self.assertIn('quarantined', result.stdout)
            self.assertEqual(self.invoke('recover').returncode, 1)
        finally:
            release.touch()
        self.wait_for(lambda: self.invoke('recover').returncode == 0)
        self.assertEqual(Store(self.root).state(0)['state'], 'idle')
        replacement = self.invoke('run', '--', sys.executable, '-c', 'print("reused safely")')
        self.assertEqual(replacement.returncode, 0, replacement.stderr)
        self.assertEqual(replacement.stdout.strip(), 'reused safely')
        self.assertFalse(stale.exists())

    @unittest.skipUnless(os.name == 'nt', 'Windows Job Object lifecycle')
    def test_killed_windows_supervisor_terminates_descendant(self):
        import ctypes
        from ctypes import wintypes
        Store(self.root, 1)
        marker = self.root.parent / 'windows-descendant-pid'
        release = self.root.parent / 'windows-release'
        child = self.start(
            f'import os,time; from pathlib import Path; Path({str(marker)!r}).write_text(str(os.getpid())); '
            f'\nwhile not Path({str(release)!r}).exists(): time.sleep(.03)')
        self.wait_for(marker.exists)
        api = ctypes.WinDLL('kernel32', use_last_error=True)
        api.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        api.OpenProcess.restype = wintypes.HANDLE
        api.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
        api.WaitForSingleObject.restype = wintypes.DWORD
        api.CloseHandle.argtypes = [wintypes.HANDLE]
        # Open a handle while alive: this assertion never relies on PID reuse.
        handle = api.OpenProcess(0x100000, False, int(marker.read_text()))
        self.assertTrue(handle)
        try:
            child.kill()
            child.wait(timeout=5)
            self.assertEqual(api.WaitForSingleObject(handle, 5000), 0)
            result = self.invoke('status')
            self.assertTrue(json.loads(result.stdout)['slots'][0]['quarantined'])
        finally:
            release.touch()
            api.CloseHandle(handle)

    def test_lease_outlives_direct_child_until_descendant_finishes(self):
        Store(self.root, 1)
        marker = self.root.parent / 'descendant'
        release = self.root.parent / 'descendant-release'
        descendant = (f'from pathlib import Path; import time; Path({str(marker)!r}).touch(); '
                      f'\nwhile not Path({str(release)!r}).exists(): time.sleep(.03)')
        parent = f'import subprocess,sys; subprocess.Popen([sys.executable, "-c", {descendant!r}])'
        child = self.start(parent)
        self.wait_for(marker.exists)
        time.sleep(.15)
        self.assertIsNone(child.poll())
        result = self.invoke('clean', '--idle-hours', '0', '--apply')
        self.assertIn('active lease', result.stdout)
        release.touch()
        self.assertEqual(child.wait(timeout=5), 0)
        self.assertEqual(Store(self.root).state(0)['state'], 'idle')

    def test_slot_binary_cannot_run_outside_its_lease(self):
        store = Store(self.root, 1)
        binary = store.target(0) / 'debug/app'
        binary.parent.mkdir(parents=True)
        binary.touch()
        with self.assertRaisesRegex(ValueError, 'active lease'):
            require_binary_lease(binary)
        copied = self.root.parent / 'durable-app'
        copied.touch()
        require_binary_lease(copied)

    def test_recovery_requires_a_proven_later_boot(self):
        store = Store(self.root, 1)
        store.save(0, {'state': 'running', 'boot': 'boot-a'})
        with patch.object(launcher, 'boot_identity', return_value='boot-a'):
            launcher.recover(store)
        self.assertEqual(store.state(0)['state'], 'running')
        with patch.object(launcher, 'boot_identity', return_value='boot-b'):
            launcher.recover(store)
        self.assertEqual(store.state(0)['state'], 'idle')

    def test_budget_can_evict_recent_idle_cache_but_retention_preserves_it(self):
        store = Store(self.root, 1)
        cache = store.target(0) / 'debug/deps'
        cache.mkdir(parents=True)
        (cache / 'generated').write_bytes(b'x' * 8192)
        store.save(0, {'state': 'idle', 'finished': time.time()})
        result = self.invoke('clean', '--idle-hours', '24', '--apply')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertTrue(cache.exists())
        result = self.invoke('clean', '--idle-hours', '24', '--budget-gib', '0', '--apply')
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(cache.exists())

    def test_real_cargo_rebuilds_distinct_checkouts_in_reused_slot(self):
        import shutil
        if not shutil.which('cargo'):
            self.skipTest('Cargo unavailable')
        Store(self.root, 1)
        for name in ('branch_a', 'branch_b', 'branch_a'):
            source = self.root.parent / name
            source.mkdir(exist_ok=True)
            (source / 'src').mkdir(exist_ok=True)
            (source / 'Cargo.toml').write_text(
                '[package]\nname="slot-smoke"\nversion="0.1.0"\nedition="2021"\n[workspace]\n')
            (source / 'src/main.rs').write_text('fn main() { println!("' + name + '"); }')
            result = self.invoke('cargo', 'run', '--quiet', '--offline',
                                 '--manifest-path', str(source / 'Cargo.toml'))
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(result.stdout.strip(), name)
        self.assertEqual(len(list(self.root.glob('slot-*'))), 1)

    def test_checkout_cannot_replace_binary_between_build_and_execution(self):
        import shutil
        if not shutil.which('cargo'):
            self.skipTest('Cargo unavailable')
        Store(self.root, 1)
        checkouts = []
        for name in ('first', 'second'):
            source = self.root.parent / name
            (source / 'src').mkdir(parents=True)
            (source / 'Cargo.toml').write_text(
                '[package]\nname="collision-smoke"\nversion="0.1.0"\nedition="2021"\n[workspace]\n')
            (source / 'src/lib.rs').write_text(
                'pub fn value() -> &' + "'static str { \"first\" }" if name == 'first'
                else 'pub fn value() -> usize { 2 }')
            (source / 'src/main.rs').write_text(
                'fn main() { let value: ' + ('&str' if name == 'first' else 'usize') +
                '=collision_smoke::value(); assert_eq!(value, ' +
                ('"first"' if name == 'first' else '2') + '); println!("' + name + '"); }')
            (source / 'src/bin').mkdir()
            (source / 'src/bin/xtask.rs').write_text(
                'fn main() { assert!(std::process::Command::new("cargo")'
                '.args(["build", "--quiet", "--offline", "--bin", "collision-smoke"])'
                '.status().unwrap().success()); }')
            (source / '.cargo').mkdir()
            (source / '.cargo/config.toml').write_text(
                '[alias]\nxtask="run --quiet --offline --bin xtask --"\n')
            checkouts.append(source)
        built = self.root.parent / 'built'
        release = self.root.parent / 'release'
        entered = self.root.parent / 'second-entered'
        pipeline = (
            'import os,subprocess,time; from pathlib import Path; '
            f'subprocess.run([{sys.executable!r}, {str(LAUNCHER)!r}, "cargo", "xtask"], check=True); '
            'binary=Path(os.environ["CARGO_TARGET_DIR"])/"debug"/'
            '("collision-smoke.exe" if os.name == "nt" else "collision-smoke"); ')
        first = self.start(pipeline +
                           f'Path({str(built)!r}).touch(); '
                           f'\nwhile not Path({str(release)!r}).exists(): time.sleep(.03)'
                           '\nsubprocess.run([str(binary)], check=True)', cwd=checkouts[0])
        self.wait_for(built.exists)
        second = self.start(f'from pathlib import Path; Path({str(entered)!r}).touch(); ' +
                            pipeline + 'subprocess.run([str(binary)], check=True)',
                            cwd=checkouts[1])
        try:
            time.sleep(.4)
            self.assertIsNone(second.poll())
            self.assertFalse(entered.exists())
        finally:
            release.touch()
        first_out, first_err = first.communicate(timeout=30)
        second_out, second_err = second.communicate(timeout=30)
        self.assertEqual(first.returncode, 0, first_err)
        self.assertEqual(second.returncode, 0, second_err)
        self.assertEqual(first_out.strip(), 'first')
        self.assertEqual(second_out.strip(), 'second')
        report = self.root / 'slot-0/target/report.json'
        report.write_text('preserve')
        cleaned = self.invoke('clean', '--idle-hours', '0', '--apply')
        self.assertEqual(cleaned.returncode, 0, cleaned.stderr)
        for expected_clean in (True, False):
            last = self.start(pipeline + 'subprocess.run([str(binary)], check=True)',
                              cwd=checkouts[0])
            output, errors = last.communicate(timeout=30)
            self.assertEqual(last.returncode, 0, errors)
            self.assertEqual(output.strip(), 'first')
            self.assertEqual('checkout reassignment' in errors, expected_clean)
            self.assertEqual(report.read_text(), 'preserve')

    def test_failed_reassignment_never_launches_or_blesses_new_checkout(self):
        store = Store(self.root, 1)
        target = store.target(0)
        target.mkdir(parents=True)
        (target / 'unrecognized').write_text('not a Cargo target')
        store.save(0, {'state': 'idle', 'checkout': 'another-checkout'})
        with patch.object(launcher, 'run', return_value=101) as run:
            for _ in range(2):
                self.assertEqual(launcher.execute(store, [sys.executable, '-c', 'pass']), 101)
                self.assertIn('_reassign', run.call_args.args[0])
                self.assertIsNone(store.state(0)['checkout'])

    @unittest.skipIf(os.name == 'nt', 'symlink creation requires privileges on Windows')
    def test_cleanup_refuses_symlinked_profile(self):
        store = Store(self.root, 1)
        source = self.root.parent / 'source'
        source.mkdir()
        (source / 'incremental').mkdir()
        target = store.target(0)
        target.mkdir(parents=True)
        (target / 'debug').symlink_to(source, target_is_directory=True)
        result = self.invoke('clean', '--idle-hours', '0', '--apply')
        self.assertNotEqual(result.returncode, 0)
        self.assertTrue((source / 'incremental').exists())


if __name__ == '__main__':
    unittest.main()
