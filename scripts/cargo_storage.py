"""Shared Cargo slot metadata and OS locks. No third-party dependencies."""
import contextlib
import json
import os
from pathlib import Path
import subprocess
import time

FORMAT = 'mechanic-cargo-slots-v2'


class Lock:
    """A stable lock inode: never unlink it, including during cleanup."""
    def __init__(self, path):
        if path.is_symlink():
            raise ValueError(f'refusing symlink: {path}')
        self.file = path.open('a+b')
        self.held = False

    def acquire(self):
        try:
            if os.name == 'nt':
                import msvcrt
                self.file.seek(0)
                if not self.file.read(1):
                    self.file.write(b'\0')
                    self.file.flush()
                self.file.seek(0)
                msvcrt.locking(self.file.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            self.held = True
            return True
        except (BlockingIOError, PermissionError):
            return False
        except OSError as error:
            if os.name == 'nt' and error.errno in (13, 36):
                return False
            raise

    def close(self):
        if self.held and os.name == 'nt':
            import msvcrt
            self.file.seek(0)
            msvcrt.locking(self.file.fileno(), msvcrt.LK_UNLCK, 1)
        self.file.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def read(path):
    if path.is_symlink():
        raise ValueError(f'refusing symlink: {path}')
    return json.loads(path.read_text()) if path.exists() else None


def write(path, value):
    temporary = path.with_suffix('.tmp')
    if temporary.is_symlink() or path.is_symlink():
        raise ValueError(f'refusing symlink: {path}')
    with temporary.open('w') as stream:
        json.dump(value, stream, indent=2)
        stream.write('\n')
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(50):
        try:
            temporary.replace(path)
            break
        except PermissionError:
            # Windows readers can briefly hold a handle without delete sharing.
            if os.name != 'nt' or attempt == 49:
                raise
            time.sleep(.01)


def boot_identity():
    """Return an OS boot identity; unknown means recovery stays blocked."""
    try:
        if os.name == 'nt':
            return subprocess.check_output([
                'powershell', '-NoProfile', '-Command',
                '(Get-CimInstance Win32_OperatingSystem).LastBootUpTime.ToFileTimeUtc()'
            ], text=True, stderr=subprocess.DEVNULL).strip()
        linux = Path('/proc/sys/kernel/random/boot_id')
        if linux.exists():
            return linux.read_text().strip()
        return subprocess.check_output(['sysctl', '-n', 'kern.boottime'], text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def default_root():
    common = subprocess.check_output(
        ['git', 'rev-parse', '--path-format=absolute', '--git-common-dir'], text=True
    ).strip()
    return Path(common).resolve() / 'mechanic-cargo-storage'


class Store:
    def __init__(self, root, count=None):
        self.root = Path(root).absolute()
        # A local filesystem is required. Refuse symlink components so cleanup
        # cannot traverse an alias to a source or user-owned directory.
        if any(p.is_symlink() for p in (self.root, *self.root.parents)):
            raise ValueError('storage root must not contain symlink components')
        self.root.mkdir(parents=True, exist_ok=True)
        with Lock(self.root / 'registry.lock') as lock:
            while not lock.acquire():
                time.sleep(.05)
            manifest = read(self.root / 'store.json')
            if manifest is None:
                unexpected = set(p.name for p in self.root.iterdir()) - {'registry.lock'}
                if unexpected:
                    raise ValueError('storage root is not empty or recognized')
                manifest = {'format': FORMAT, 'slots': 2 if count is None else count}
                if not 1 <= manifest['slots'] <= 32:
                    raise ValueError('slot count must be 1..32')
                write(self.root / 'store.json', manifest)
            if manifest.get('format') != FORMAT:
                raise ValueError('unrecognized storage format')
            if count is not None and manifest['slots'] != count:
                raise ValueError('slot count differs from initialized store')
            self.count = manifest['slots']
            for index in range(self.count):
                slot = self.slot(index)
                if slot.is_symlink():
                    raise ValueError('slot must not be a symlink')
                slot.mkdir(exist_ok=True)

    def slot(self, index):
        return self.root / f'slot-{index}'

    def target(self, index):
        return self.slot(index) / 'target' / 'cargo'

    def lock(self, index):
        return Lock(self.slot(index) / 'lease.lock')

    def state(self, index):
        return read(self.slot(index) / 'owner.json') or {'state': 'idle'}

    def save(self, index, state):
        write(self.slot(index) / 'owner.json', state)


def file_stats(path):
    """Yield regular file metadata without following symlinks."""
    if path.is_symlink() or not path.exists():
        return
    if path.is_file():
        yield path.stat()
        return
    for directory, _, files in os.walk(path, followlinks=False):
        for name in files:
            file = Path(directory) / name
            with contextlib.suppress(FileNotFoundError):
                if not file.is_symlink() and file.is_file():
                    yield file.stat()


def allocated_bytes(stat):
    return stat.st_blocks * 512 if hasattr(stat, 'st_blocks') else stat.st_size


def size(path, seen=None):
    """Allocated file bytes (logical on Windows), counting each inode once."""
    seen = set() if seen is None else seen
    total = 0
    for stat in file_stats(path):
        key = (stat.st_dev, stat.st_ino)
        if key not in seen:
            seen.add(key)
            total += allocated_bytes(stat)
    return total


def reclaimable(paths):
    """Exclude file blocks whose hardlinks survive outside the deletion set."""
    entries = {}
    for path in paths:
        for stat in file_stats(path):
            key = (stat.st_dev, stat.st_ino)
            if key not in entries:
                entries[key] = [stat, 0]
            entries[key][1] += 1
    return sum(allocated_bytes(stat) for stat, count in entries.values()
               if count == stat.st_nlink)


def require_binary_lease(binary):
    """Refuse a managed slot executable unless its matching lease is inherited."""
    for target in Path(binary).resolve().parents:
        slot = target.parent
        if target.name != 'target' or not slot.name.startswith('slot-'):
            continue
        manifest = read(slot.parent / 'store.json')
        if not manifest or manifest.get('format') != FORMAT:
            continue
        owner = read(slot / 'owner.json') or {}
        if (not os.environ.get('MECHANIC_CARGO_LEASE')
                or owner.get('token') != os.environ['MECHANIC_CARGO_LEASE']
                or owner.get('state') != 'running'):
            raise ValueError('slot binary requires its active lease; use a leased pipeline or a durable copy')
        with Lock(slot / 'lease.lock') as lock:
            if lock.acquire():
                raise ValueError('slot binary supervisor is gone; refusing orphaned lease')
        return
