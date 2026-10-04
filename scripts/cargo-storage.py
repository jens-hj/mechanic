#!/usr/bin/env python3
"""Lease reusable Cargo storage before Cargo (including xtask) starts.

python3 scripts/cargo-storage.py cargo xtask test
python3 scripts/cargo-storage.py run -- sh -c 'cargo build; "$CARGO_TARGET_DIR/debug/app"'
python3 scripts/cargo-storage.py status
python3 scripts/cargo-storage.py clean --idle-hours 24 --budget-gib 80 [--apply]
"""
import argparse
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from cargo_storage import Lock, Store, boot_identity, default_root, reclaimable, size
from cargo_process import group_is_empty, run

TOKEN = 'MECHANIC_CARGO_LEASE'
ROOT = 'MECHANIC_CARGO_STORAGE'
SELF = Path(__file__).resolve()
TARGET_VARIABLES = ('CARGO_TARGET_DIR', 'CARGO_BUILD_TARGET_DIR', 'CARGO_BUILD_BUILD_DIR')


def emit(message):
    print(f'cargo-storage: {message}', file=sys.stderr, flush=True)


def inherited(store):
    token = os.environ.get(TOKEN)
    if not token:
        return None
    for index in range(store.count):
        state = store.state(index)
        if state.get('token') == token and state['state'] == 'running':
            with store.lock(index) as lock:
                if lock.acquire():
                    raise ValueError('inherited lease has lost its supervisor; slot quarantined')
            target = str(store.target(index))
            if any(os.environ.get(name) != target for name in TARGET_VARIABLES):
                raise ValueError('nested command changed CARGO_TARGET_DIR')
            return index
    raise ValueError('stale or foreign inherited lease')


def validate_command(command):
    if not command:
        raise ValueError('a command is required')
    if Path(command[0]).stem in ('cargo', 'cargo.exe'):
        for arg in command[1:]:
            if arg == '--':
                break
            if arg.startswith(('--target-dir', '--build-dir', '--artifact-dir', '--config')):
                raise ValueError('target/config overrides require explicit --unmanaged mode')


def execute(store, command, min_free_gib=20):
    if not math.isfinite(min_free_gib) or min_free_gib < 0:
        raise ValueError('minimum free GiB must be finite and nonnegative')
    validate_command(command)
    if inherited(store) is not None:
        code = subprocess.call(command)
        return 128 - code if code < 0 else code
    if any(os.environ.get(name) for name in TARGET_VARIABLES):
        raise ValueError('unset Cargo target/build directory variables or use explicit --unmanaged mode')
    affinity = str(Path.cwd().resolve())
    waited = time.monotonic()
    last_report = 0
    while True:
        indices = sorted(range(store.count), key=lambda i: store.state(i).get('checkout') != affinity)
        for index in indices:
            with store.lock(index) as lock:
                if not lock.acquire():
                    continue
                previous = store.state(index)
                if previous['state'] != 'idle':
                    continue
                slot = store.slot(index)
                target = store.target(index)
                if target.is_symlink() or target.parent.is_symlink():
                    raise ValueError('target must not be a symlink')
                target.parent.mkdir(parents=True, exist_ok=True)
                token = uuid.uuid4().hex
                state = {'state': 'running', 'pid': os.getpid(), 'token': token,
                         'boot': boot_identity(), 'checkout': affinity,
                         'started': time.time(), 'command': command}
                # Serialize admission snapshots and owner publication across slots.
                # This is not a reservation: running or unmanaged writers can grow.
                with Lock(store.root / 'admission.lock') as admission:
                    while not admission.acquire():
                        time.sleep(.05)
                    volume_path = target if target.exists() else target.parent
                    free = shutil.disk_usage(volume_path).free
                    required = math.ceil(min_free_gib * 1024**3)
                    if free < required:
                        raise ValueError(
                            f'low storage space at {volume_path}: {free} bytes '
                            f'({free / 1024**3:.2f} GiB) free; required {required} bytes '
                            f'({min_free_gib:g} GiB). No command started. '
                            'Use this root with status and clean (preview); '
                            'review idle candidates before clean --apply. '
                            'Active storage is never automatically deleted.')
                    state['admission_free_bytes'] = free
                    store.save(index, state)
                env = {**os.environ, ROOT: str(store.root), TOKEN: token,
                       **{name: str(target) for name in TARGET_VARIABLES}}
                if os.name == 'nt':
                    env['_MSPDBSRV_ENDPOINT_'] = f'mechanic-{token}'
                emit(f'slot {index}, waited {time.monotonic()-waited:.2f}s, target {target}')
                def record_containment(containment):
                    state.update(containment)
                    store.save(index, state)

                # Reassignment must invalidate both final and intermediate artifacts.
                # Run Cargo's supported clean boundary inside the same containment
                # as the command: a crash cannot expose a still-running cleaner.
                prepared = command
                if (previous.get('checkout') != affinity and target.exists()
                        and any(target.iterdir())):
                    manifest_dir = slot / 'clean-manifest'
                    if any(p.is_symlink() for p in (manifest_dir,
                           manifest_dir / 'Cargo.toml', manifest_dir / 'lib.rs')):
                        raise ValueError('cleanup manifest must not contain symlinks')
                    manifest_dir.mkdir(exist_ok=True)
                    (manifest_dir / 'Cargo.toml').write_text(
                        '[package]\nname="slot-clean"\nversion="0.0.0"\nedition="2021"\n'
                        '[lib]\npath="lib.rs"\n[workspace]\n')
                    (manifest_dir / 'lib.rs').write_text('')
                    emit(f'slot {index}: checkout reassignment; invalidating Cargo artifacts')
                    prepared = [sys.executable, str(SELF), '_reassign',
                                str(manifest_dir / 'Cargo.toml'), *command]
                code = run(prepared, env, slot / f'gate-{token}', SELF, record_containment)
                store.save(index, {'state': 'idle', 'checkout': affinity if code == 0 else None,
                                   'finished': time.time(), 'exit_code': code})
                return code
        now = time.monotonic()
        if now - last_report >= 5:
            emit(f'queued {now-waited:.1f}s: ' + '; '.join(
                f'slot {i}: {store.state(i)}' for i in range(store.count)))
            last_report = now
        time.sleep(.1)


def status(store):
    rows = []
    for index in range(store.count):
        with store.lock(index) as lock:
            available = lock.acquire()
            state = store.state(index)
            rows.append({'slot': index, 'locked': not available,
                         'quarantined': available and state['state'] != 'idle',
                         'bytes': size(store.target(index)), 'owner': state})
    print(json.dumps({'root': str(store.root), 'slots': rows}, indent=2))


def candidates(target):
    """Only Cargo-owned names; custom reports/captures at target root survive."""
    # Cargo requires its ownership tag for later full reassignment cleanup.
    names = ['.rustc_info.json']
    for profile in ('debug', 'release', 'profiling'):
        names.extend(f'{profile}/{name}' for name in
                     ('deps', '.fingerprint', 'build', 'incremental', 'examples'))
    result = []
    if target.is_symlink():
        raise ValueError('target must not be a symlink')
    for name in names:
        path = target / name
        if path.is_symlink() or path.parent.is_symlink():
            raise ValueError(f'refusing symlink: {path}')
        if path.exists():
            result.append(path)
    return result


def cleanup(store, args):
    total = sum(size(store.target(i)) for i in range(store.count))
    rows = []
    reclaimable_bytes = 0
    indices = sorted(range(store.count), key=lambda i: store.state(i).get('finished', 0))
    for index in indices:
        with store.lock(index) as lock:
            if not lock.acquire():
                rows.append({'slot': index, 'skip': 'active lease'})
                continue
            state = store.state(index)
            if state['state'] != 'idle':
                rows.append({'slot': index, 'skip': 'quarantined'})
                continue
            age = max(0, time.time() - state.get('finished', time.time())) / 3600
            if age < args.idle_hours and total <= args.budget_gib * 1024**3:
                rows.append({'slot': index, 'skip': 'within retention and budget'})
                continue
            paths = candidates(store.target(index))
            estimate = reclaimable(paths)
            seen = set()
            for path in paths:
                count = size(path, seen)
                rows.append({'slot': index, 'path': str(path), 'bytes': count, 'deleted': args.apply})
                if args.apply:
                    if path.is_dir():
                        shutil.rmtree(path)
                    else:
                        path.unlink()
            total -= estimate
            reclaimable_bytes += estimate
    print(json.dumps({'apply': args.apply, 'remaining_bytes_estimate': total,
                      'reclaimable_file_bytes_estimate': reclaimable_bytes, 'actions': rows}, indent=2))


def recover(store):
    current = boot_identity()
    unresolved = False
    for index in range(store.count):
        with store.lock(index) as lock:
            if not lock.acquire():
                emit(f'slot {index}: active, skipped')
                continue
            state = store.state(index)
            if state['state'] == 'idle':
                continue
            if current and state.get('boot') and state['boot'] != current:
                reason = 'reboot'
            elif group_is_empty(state.get('process_group')):
                reason = 'recorded process group drained'
            else:
                emit(f'slot {index}: quarantined; process group not proven empty and no proven reboot')
                unresolved = True
                continue
            store.save(index, {'state': 'idle', 'finished': time.time()})
            emit(f'slot {index}: recovered after {reason}')
    return 1 if unresolved else 0


def main():
    # This child cannot spawn before the supervisor has assigned containment.
    if len(sys.argv) > 1 and sys.argv[1] == '_child':
        gate = Path(sys.argv[2])
        while not gate.exists():
            time.sleep(.01)
        if os.name == 'nt':
            # Windows CRT execvpe crashes on the supported CI Python and does
            # not provide POSIX exec semantics. The assigned Job Object already
            # contains every subprocess; CreateProcess also preserves quoting.
            return subprocess.call(sys.argv[4:])
        os.execvpe(sys.argv[4], sys.argv[4:], os.environ)
    if len(sys.argv) > 1 and sys.argv[1] == '_reassign':
        # The dedicated cargo subtree contains compiler artifacts only. Reports
        # and durable captures belong alongside it, never within it.
        result = subprocess.call(['cargo', 'clean', '--manifest-path', sys.argv[2],
                                  '--target-dir', os.environ['CARGO_TARGET_DIR']])
        if result:
            return result
        code = subprocess.call(sys.argv[3:])
        return 128 - code if code < 0 else code
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--root', type=Path, default=os.environ.get(ROOT))
    parser.add_argument('--slots', type=int)
    parser.add_argument('--min-free-gib', type=float, default=20,
                        help='minimum free space on storage volume before admission (default: 20 GiB)')
    parser.add_argument('--unmanaged', action='store_true', help='explicit isolated/CI opt-out; command only')
    sub = parser.add_subparsers(dest='action', required=True)
    for action in ('cargo', 'run'):
        sub.add_parser(action).add_argument('command', nargs=argparse.REMAINDER)
    sub.add_parser('status')
    sub.add_parser('recover')
    clean = sub.add_parser('clean')
    clean.add_argument('--idle-hours', type=float, default=24)
    clean.add_argument('--budget-gib', type=float, default=80)
    clean.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    if args.action in ('cargo', 'run'):
        command = args.command
        if command[:1] == ['--']:
            command = command[1:]
        if args.action == 'cargo':
            command = ['cargo', *command]
        if args.unmanaged:
            if os.environ.get(TOKEN):
                raise ValueError('cannot opt out from inside an active lease')
            code = subprocess.call(command)
            return 128 - code if code < 0 else code
        return execute(Store(args.root or default_root(), args.slots), command, args.min_free_gib)
    if args.unmanaged:
        raise ValueError('--unmanaged applies only to commands')
    store = Store(args.root or default_root(), args.slots)
    if args.action == 'status':
        status(store)
    elif args.action == 'recover':
        return recover(store)
    else:
        if args.idle_hours < 0 or args.budget_gib < 0:
            raise ValueError('retention and budget must be nonnegative')
        cleanup(store, args)
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
    except (OSError, ValueError) as error:
        emit(str(error))
        sys.exit(1)
