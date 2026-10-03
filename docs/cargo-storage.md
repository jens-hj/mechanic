# Reusable Cargo storage

`python3 scripts/cargo-storage.py cargo xtask test` leases storage **before**
Cargo builds xtask. Use `python` on Windows. Python's standard library is the only
launcher dependency. Rust/Cargo settings and user-global configuration are unchanged.

The default root is `<git-common-dir>/mechanic-cargo-storage`, shared by linked
worktrees and outside their working directories. `--root /absolute/local/path`
or `MECHANIC_CARGO_STORAGE` selects another shared root. Initialize with `--slots N`
to change the default of two slots; subsequent callers cannot change its count.
All workers must use the same root. Local filesystems only: network filesystem
lock semantics are not supported. Do not move or delete the root or lock files
while a launcher exists. Symlink roots/slots/targets are refused.

## Running and nesting

```sh
python3 scripts/cargo-storage.py cargo xtask lint
python3 scripts/cargo-storage.py cargo xtask test -p mechanic-core
python3 scripts/cargo-storage.py cargo run -p mechanic-app
python3 scripts/cargo-storage.py cargo run -p mechanic-bench --release -- --scenario four_bar
```

The first idle slot with checkout affinity is preferred; otherwise any idle slot
is reused. A third worker waits and prints owners every five seconds. Scheduling
is opportunistic, not FIFO; long contention can starve a waiter. Waiting does not
hold a slot. Each slot has one stable OS lock (flock on Unix, byte-range locking
on Windows). Metadata names the command, checkout, PID, start time, and token.
`status` shows both lock state and quarantine; a PID alone is never proof of safety.
Nested launchers validate the token and lock and inherit the target without
acquiring another slot. Cargo's own locks remain in effect inside the lease.

Both final and intermediate Cargo artifacts stay in the slot: the launcher sets
`CARGO_TARGET_DIR`, `CARGO_BUILD_TARGET_DIR`, and `CARGO_BUILD_BUILD_DIR` for its
children. This explicitly covers Cargo's separate [intermediate build directory](https://doc.rust-lang.org/cargo/reference/config.html#buildbuild-dir)
without changing global configuration. Existing directory environment overrides
require unsetting them or using `--unmanaged`.

The lease covers the command's complete process tree. Build and direct-run or
capture must be **one command**; shell example for macOS/Linux:

```sh
python3 scripts/cargo-storage.py run -- sh -c '
  cargo build --release -p mechanic-app &&
  python3 scripts/run-background-capture.py --world /path/to/world --output /path/to/new-capture
'
```

The capture default follows `CARGO_TARGET_DIR`. On Windows use a Python pipeline
or PowerShell under `run --`; argument lists are forwarded without shell quoting.
Always build for the current source before invoking a reused slot's binary.
Do not override the target within the pipeline. Never launch slot binaries after
the lease ends. To compare two revisions, copy each binary and its identity to a
durable measurement directory **while still leased**, then run those copies.
Existing isolated reference/fence builders keep their separate targets and
immutable result binaries; they are explicit measurement exceptions and outside
the slot budget. CPU comparison runners should consume those stable copies.

Direct Cargo continues to work for ordinary ephemeral CI. For explicit isolated
experiments, `--unmanaged cargo ...` bypasses slots and respects the caller's
target. It is forbidden inside a lease. Target/config overrides to the launcher’s
`cargo` command require this opt-out. The arbitrary `run` command is a cooperative
interface, not a sandbox: scripts must honor the target and lifecycle contract.

## Lifecycle and crash recovery

Unix commands run in a fresh process group; cancellation is forwarded to the group
and ownership remains until it empties. Windows assigns a gated child to a Job
Object before allowing it to launch the command. The job retains descendants and
kills them if the supervisor disappears. MSVC receives a unique
`_MSPDBSRV_ENDPOINT_` for each lease, following [Microsoft's per-invocation
isolation](https://github.com/microsoft/BuildXL/blob/main/Public/Sdk/Experimental/Msvc/Native/Tools/Link/Link.dsc).
After the main command and all ordinary descendants exit, a job containing only
the MSVC compiler-tree services `mspdbsrv.exe` or `vctip.exe` is terminated so
idle compiler/telemetry services cannot pin the slot. Other images keep ownership.
Process images and job membership are checked through handles; uncertain members
keep the lease. Do not override the endpoint inside a leased command.
Assignment failure fails closed. Child
exit codes are preserved; Unix signal termination maps to `128 + signal`.
A child that ignores cancellation keeps its lease until it exits.

Commands must not daemonize, change sessions/groups, or escape containment.
This launcher is for Cargo, tests, foreground apps, and capture pipelines, not
arbitrary background services. Outputs must remain under the leased process tree.

Before spawning anything the supervisor persists a running record. If it crashes,
gets SIGKILL, or encounters an uncertain lifecycle error, a released OS lock does
**not** make the slot reusable: it is quarantined. This also covers orphaned Cargo
children. `clean` skips quarantine. Explicit
`python3 scripts/cargo-storage.py recover` reacquires the slot lock and checks the
Unix process group that was persisted **before** the command's startup gate opened.
It clears quarantine only when the entire recorded group is empty. A live orphan,
permission failure, or reused group ID blocks recovery; an absent parent PID alone
never clears it. This works even when boot identity is unavailable.

If containment was not yet recorded when the supervisor crashed, or on Windows,
recovery requires recorded/current boot identities proving a subsequent OS reboot.
If those identities are unavailable, the uncertain slot remains quarantined.
`recover` returns nonzero while any unlocked slot remains quarantined. There is no
force-unlock, automatic eviction, or missing-parent-PID heuristic. Never delete
owner metadata to bypass recovery. The no-daemonizing containment contract above
also applies to same-boot recovery; escaping children are unsupported.


## Inventory and explicit cleanup

```sh
python3 scripts/cargo-storage.py status
python3 scripts/cargo-storage.py clean --idle-hours 24 --budget-gib 80
python3 scripts/cargo-storage.py clean --idle-hours 24 --budget-gib 80 --apply
```

Cleanup rechecks exclusive ownership slot by slot and deletes only recognized
Cargo cache subdirectories (`deps`, `.fingerprint`, `build`, `incremental`,
`examples`) in `debug`, `release`, and `profiling`, plus Cargo root cache markers.
It preserves other files, top-level binaries, reports/captures alongside cache
directories, unknown/custom profiles, cross-compilation targets, and source.
Do not put authored data inside Cargo-owned cache subdirectories. Symlinked
profiles are refused and recursive deletion does not follow internal symlinks.
Lock files and ownership metadata are never cleanup candidates.

Idle slots qualify when older than retention **or** the total recognized store
exceeds the configured budget, oldest first. Busy and quarantined slots always
survive, so the budget is a cleanup target, not a hard allocation cap. Preview is
advisory: apply reacquires locks and recomputes eligibility. No automatic eviction
runs. Status sizes count each inode once (allocated file bytes on Unix, logical bytes
on Windows). Cleanup reports `reclaimable_file_bytes_estimate` separately from
per-path sizes: it excludes any file whose hardlinks survive outside the deletion
set. Filesystem metadata, shared clone extents, and concurrent activity can still
make the observed free-space delta differ from this estimate.

Two slots limit duplicated working caches, not all historical artifacts or peak
build size. Branches, features, profiles, compiler changes and dependency path
fingerprints still grow a slot. Reusing worktrees does not promise complete Cargo
deduplication. Cleanup sacrifices warm-cache speed. A third build trades queue time
for lower concurrent disk demand. Durable captures and isolated builds need their
own inventory and retention policy.

## Rollout and evidence

1. Inventory current targets and live users with the orchestrator. Let active
   builds finish in place. Never repoint a running build or remove its artifacts.
2. Exercise launcher tests and small Cargo builds in a test-owned root, then have
   idle workers adopt the same shared root on their next invocation.
3. Preview recognized slot cleanup. Legacy targets are never auto-discovered or
   deleted by this tool; reclaim them only after independent proof that all
   builders, tests, apps, and captures using them have ended.
4. Keep automatic idle eviction off until operational safety is demonstrated.

Initial macOS inventory on 2026-10-03: main target 129,867,080 KiB, coupler target
27,417,904 KiB, about 20 GiB filesystem space available. These sizes are **not**
proven reclaimable. This session's sandbox blocks process inspection and boot
identity reads. AO coordination is established; shared-worker rollout remains
pending while ordinary shared-target builds/captures are active. The urgent recovery below later removed only the independently verified-idle
retired incremental cache. Shared-slot adoption remains separate from recovery.

`python3 scripts/test-cargo-storage.py` exercises actual process locks, two occupied
slots plus a queued third, nested reuse, exit status, cancellation, crash quarantine,
cleanup contention, symlink refusal, and a tiny real Cargo binary rebuilt across
checkout paths. Full workspace builds are unnecessary for scheduler regression.
Platform CI must verify the Windows Job Object path before adoption there.

Incremental-off and reduced-debug experiments remain opt-in. No optimizer, debug,
assertion, or profile defaults change. Measure cold and warm builds separately in
an isolated target (including disk size and elapsed time); a tiny launcher smoke
fixture cannot establish the Bevy workspace's build-speed/storage tradeoff.

### Local validation, 2026-10-03

macOS arm64, Rust/Cargo 1.97.1, Python 3.14.3:

- 13 focused storage tests passed, including real Cargo checkout/binary isolation.
- Outer leased `cargo xtask consistency`, `scripts-test`, and `fmt` passed.
  The script suite contains 44 tests including the 13 storage tests.
- Real xtask cache cleanup was exercised in a fresh test-owned temporary root.
  The initial per-path totals overstated physical reclaim where Cargo hardlinked
  cache files. The later inode-aware accounting regression and recovery audit
  below supersede those initial byte claims.
- `cargo metadata --no-deps --locked --offline` validates the 0.4.2 manifest/lock bump.
- Added a dedicated three-platform lifecycle CI job with tiny Cargo fixtures and
  no private workspace dependencies. Linux/Windows results are pending CI.

No full app build, runtime performance claim, profile-default change, legacy
reclamation, or shared-worker rollout is included in this local evidence.

Follow-up lifecycle validation: the killed-supervisor regression now proves that
recovery refuses a live orphan, then safely reuses the slot after its recorded
Unix process group drains. This removes the reboot requirement for ordinary Unix
supervisor crashes while retaining fail-closed behavior for uncertain containment.

### Urgent legacy-cache recovery, 2026-10-03

After a reported disk-full event, fresh inspection already showed about 56.7 GiB
available before this session deleted anything. Main and mechanic-5 targets still
had live compiler users and were excluded. The retired mechanic-4 target had no
open users. Recovery acquired its existing `.cargo-lock`, `.cargo-build-lock`, and
`.cargo-artifact-lock` exclusively, then repeated the open-user check while holding
those locks. Only `target/debug/incremental` was removed: 25,384 compiler files
(`.bc`, `.o`, `.bin`, `.lock`, `.rmeta`), with no symlinks or unrecognized file types.
Source, reports/captures, binaries, dependency artifacts, and lock files survived.

- Candidate: 19,765,223,424 unique allocated file bytes.
- Expected reclaim after retained hardlinks: 14,601,969,664 bytes.
- Free immediately before/after: 60,303,097,856 / 74,897,154,048 bytes.
- Observed free-space gain: 14,594,056,192 bytes (13.59 GiB).
- Immediate remaining space: 69.75 GiB; subsequent reading about 68.7 GiB as other
  work continued. This is current capacity, not a guarantee for additional builds.

This was explicit, narrowly scoped legacy recovery under independent Cargo locks,
not managed-slot cleanup or evidence of completed shared-worker rollout.

The expanded lifecycle suite defines 16 tests, including hardlink accounting and a
Windows-specific descendant-handle test. Lightweight Linux/macOS CI passed before
the Windows fix. Windows CI exposed a native Python `os.execvpe` crash; its gated
child now uses `subprocess` within the Job Object. Verification of that fix is pending.

The subsequent Windows smoke trace confirmed Cargo had run its expected binary,
but MSVC's `vctip.exe` remained alive. Completion now also drains this known
compiler-tree telemetry helper after every ordinary job member exits; binaries
outside the compiler tree are not treated as disposable services.
