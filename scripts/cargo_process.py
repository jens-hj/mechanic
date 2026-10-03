"""Keep a command's descendants inside the lifetime of its storage lease.

Commands must not daemonize or escape their process group/job. An interrupted
supervisor leaves persistent quarantine; parent-PID checks never clear it.
"""
import ctypes
import os
from pathlib import Path
import signal
import subprocess
import time


class WindowsJob:
    """Assign the gated child before it can spawn; kill descendants on close."""
    def __init__(self):
        from ctypes import wintypes as w

        class Basic(ctypes.Structure):
            _fields_ = [('process_time', ctypes.c_int64), ('job_time', ctypes.c_int64),
                        ('flags', w.DWORD), ('min_ws', ctypes.c_size_t),
                        ('max_ws', ctypes.c_size_t), ('active_limit', w.DWORD),
                        ('affinity', ctypes.c_size_t), ('priority', w.DWORD),
                        ('scheduling', w.DWORD)]

        class IO(ctypes.Structure):
            _fields_ = [(name, ctypes.c_uint64) for name in
                        ('read_ops', 'write_ops', 'other_ops', 'read_bytes', 'write_bytes', 'other_bytes')]

        class Extended(ctypes.Structure):
            _fields_ = [('basic', Basic), ('io', IO), ('process_memory', ctypes.c_size_t),
                        ('job_memory', ctypes.c_size_t), ('peak_process', ctypes.c_size_t),
                        ('peak_job', ctypes.c_size_t)]

        self.api = ctypes.WinDLL('kernel32', use_last_error=True)
        self.api.CreateJobObjectW.argtypes = [ctypes.c_void_p, w.LPCWSTR]
        self.api.CreateJobObjectW.restype = w.HANDLE
        self.api.SetInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD]
        self.api.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        self.api.QueryInformationJobObject.argtypes = [w.HANDLE, ctypes.c_int, ctypes.c_void_p, w.DWORD, ctypes.c_void_p]
        self.api.TerminateJobObject.argtypes = [w.HANDLE, w.UINT]
        self.api.CloseHandle.argtypes = [w.HANDLE]
        self.api.OpenProcess.argtypes = [w.DWORD, w.BOOL, w.DWORD]
        self.api.OpenProcess.restype = w.HANDLE
        self.api.QueryFullProcessImageNameW.argtypes = [w.HANDLE, w.DWORD, w.LPWSTR, ctypes.POINTER(w.DWORD)]
        self.api.IsProcessInJob.argtypes = [w.HANDLE, w.HANDLE, ctypes.POINTER(w.BOOL)]
        self.handle = self.api.CreateJobObjectW(None, None)
        if not self.handle:
            raise ctypes.WinError(ctypes.get_last_error())
        limits = Extended()
        limits.basic.flags = 0x2000  # JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE
        if not self.api.SetInformationJobObject(self.handle, 9, ctypes.byref(limits), ctypes.sizeof(limits)):
            self.close()
            raise ctypes.WinError(ctypes.get_last_error())

    def assign(self, child):
        if not self.api.AssignProcessToJobObject(self.handle, int(child._handle)):
            raise ctypes.WinError(ctypes.get_last_error())

    def active(self):
        # JOBOBJECT_BASIC_ACCOUNTING_INFORMATION: four LARGE_INTEGERs,
        # then page faults, total, active and terminated process counts.
        buffer = ctypes.create_string_buffer(48)
        if not self.api.QueryInformationJobObject(self.handle, 1, buffer, len(buffer), None):
            raise ctypes.WinError(ctypes.get_last_error())
        return int.from_bytes(buffer.raw[40:44], 'little') > 0

    def process_images(self):
        """Resolve current members through handles; uncertainty stays active."""
        from ctypes import wintypes as w
        buffer = ctypes.create_string_buffer(8 + 1024 * ctypes.sizeof(ctypes.c_size_t))
        if not self.api.QueryInformationJobObject(self.handle, 3, buffer, len(buffer), None):
            return None
        count = int.from_bytes(buffer.raw[4:8], 'little')
        images = []
        for index in range(count):
            pid = ctypes.c_size_t.from_buffer(buffer, 8 + index * ctypes.sizeof(ctypes.c_size_t)).value
            handle = self.api.OpenProcess(0x1000, False, pid)
            if not handle:
                return None
            try:
                belongs = w.BOOL()
                if not self.api.IsProcessInJob(handle, self.handle, ctypes.byref(belongs)) or not belongs.value:
                    return None
                capacity = w.DWORD(32768)
                image = ctypes.create_unicode_buffer(capacity.value)
                if not self.api.QueryFullProcessImageNameW(handle, 0, image, ctypes.byref(capacity)):
                    return None
                images.append(image.value)
            finally:
                self.api.CloseHandle(handle)
        return images

    def terminate(self, code):
        if not self.api.TerminateJobObject(self.handle, code):
            raise ctypes.WinError(ctypes.get_last_error())

    def close(self):
        self.api.CloseHandle(self.handle)


def group_is_empty(group):
    """Only absence of the entire recorded Unix group proves it has drained."""
    if os.name == 'nt' or not isinstance(group, int) or group <= 1:
        return False
    try:
        os.killpg(group, 0)
    except ProcessLookupError:
        return True
    except PermissionError:
        pass
    return False


def run(command, env, gate, launcher, record_containment):
    """Return only after the process group/job empties; forward cancellation."""
    import sys
    job = WindowsJob() if os.name == 'nt' else None
    child = None
    cancelled = []
    handlers = {}

    def forward(signum, _):
        cancelled.append(signum)
        if child is not None:
            try:
                if job:
                    job.terminate(128 + signum)
                else:
                    os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        signals = [signal.SIGINT, signal.SIGTERM]
        if hasattr(signal, 'SIGHUP'):
            signals.append(signal.SIGHUP)
        for sig in signals:
            handlers[sig] = signal.signal(sig, forward)
        child = subprocess.Popen(
            [sys.executable, str(launcher), '_child', str(gate), '--', *command],
            env=env, start_new_session=os.name != 'nt',
        )
        if job:
            job.assign(child)
        # Persist containment before permitting any build to start. A crash
        # before this record remains conservatively quarantined until reboot.
        record_containment({} if job else {'process_group': child.pid})
        gate.write_text('go')
        if cancelled:
            forward(cancelled[-1], None)
        code = child.wait()
        last_report = time.monotonic()
        while True:
            if job:
                active = job.active()
                images = job.process_images() if active else []
                # MSVC's PDB server intentionally outlives link.exe. Its unique
                # per-lease endpoint prevents other builds from using it. Only
                # terminate once every ordinary member has exited, and the only
                # remaining images are that known compiler service.
                if images and all(Path(path).name.casefold() == 'mspdbsrv.exe' for path in images):
                    job.terminate(0)
                elif active and time.monotonic() - last_report >= 5:
                    print(f'cargo-storage: retaining lease for Job Object members: {images}',
                          file=sys.stderr, flush=True)
                    last_report = time.monotonic()
            else:
                active = not group_is_empty(child.pid)
            if not active:
                break
            time.sleep(.1)
        return 128 + cancelled[-1] if cancelled else (128 - code if code < 0 else code)
    finally:
        for sig, handler in handlers.items():
            signal.signal(sig, handler)
        if job:
            job.close()
        if child is not None and child.poll() is None:
            # Failure here must leave owner.json quarantined, even if termination
            # fails. The launcher clears the record only after run returns.
            if os.name != 'nt':
                try:
                    os.killpg(child.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            else:
                child.kill()
            child.wait()
        gate.unlink(missing_ok=True)
