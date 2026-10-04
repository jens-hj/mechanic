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

Checkout affinity is an absolute worktree path, not Cargo's package hash. On
reassignment to another checkout (or after uncertain/crashed ownership), the
launcher runs `cargo clean` before the new command, under the same exclusive lease
and process containment. Failed cleanup never blesses the new identity. The clean
uses a tiny slot-owned manifest and explicit target directory, with the same
intermediate build-directory environment. It invalidates all profiles and local
path dependencies, including incompatible APIs, without relying on timestamps.
Same-checkout successful invocations retain their cache. Two slots retain at most
two checkout caches; switching more checkouts sacrifices dependency reuse and may
require a full rebuild. There is no per-checkout cache directory accumulation.

Both final and intermediate Cargo artifacts stay in `slot-N/target/cargo`: the launcher sets
`CARGO_TARGET_DIR`, `CARGO_BUILD_TARGET_DIR`, and `CARGO_BUILD_BUILD_DIR` for its
children. This explicitly covers Cargo's separate [intermediate build directory](https://doc.rust-lang.org/cargo/reference/config.html#buildbuild-dir)
without changing global configuration. That subtree is exclusively disposable
Cargo output: keep reports, captures and authored files outside it (for example,
`slot-N/target/report.json` survives reassignment). Store format v2 refuses old v1
roots instead of migrating or deleting their contents; no active root is upgraded.
Existing directory environment overrides
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
force-unlock, automatic budget eviction, or missing-parent-PID heuristic. Never delete
owner metadata to bypass recovery. The no-daemonizing containment contract above
also applies to same-boot recovery; escaping children are unsupported.


## Inventory and explicit cleanup

Managed admission requires **20 GiB free** by default, formalizing the repository's
earlier 20 GB low-space policy with an explicit binary-unit threshold. Configure
it before the subcommand, for example `--min-free-gib 30 cargo xtask test`; zero
explicitly disables the threshold for small isolated fixtures. Negative and
non-finite values are rejected. After acquiring an idle slot, the launcher checks
`disk_usage` on its actual Cargo target (or its existing parent before the first
build), not the checkout or temporary filesystem. Admission checks and owner
publication are serialized across slots. Below the threshold, no build,
reassignment cleanup, or command starts; the error prints current/required bytes
and suggests status plus explicit cleanup preview. Nested commands retain their
existing admission and lease rather than independently admitting another build.

This is **not a disk reservation or runtime quota**. Serialized checks remeasure
current space but do not reserve future growth: both slots can pass before either
allocates much. An active build or unmanaged writer can still fill the volume.
The guard never deletes active storage automatically. Initial rollout remains one
needed invocation at a time with before/after capacity measurements, not two cold
workspace builds. The explicit cleanup budget remains a soft idle-cache target.

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
advisory: apply reacquires locks and recomputes eligibility. No automatic budget eviction
runs. Correctness invalidation on checkout reassignment is separate from this
optional retention cleanup and cleans the entire dedicated Cargo subtree. Status sizes count each inode once (allocated file bytes on Unix, logical bytes
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

### Transition policy while legacy targets remain

The managed admission floor protects only launcher invocations. Existing direct
Cargo commands, legacy shared targets and measurement exceptions can still grow
without that guard; the recurrence reported below demonstrates this adoption gap.
Until an explicitly coordinated integration/adoption step, workers must apply the
same 20 GiB pre-build floor manually to the filesystem holding their actual target.
Below it, defer new substantial builds and report current free bytes, target path
and known users. Do not kill existing work or independently delete shared caches
to make the next build fit. A low-space instruction is a recovery trigger, not
cleanup authorization. No worker needs to wait merely because future rollout is
queued when current capacity and normal task coordination permit its work.

For any proposed legacy cleanup, assign one cleaner and coordinate an explicit
no-new-users window for the exact target. Obtain release from all builders,
tests, apps and captures, including retained binaries and orphaned descendants;
one worker's compilation finishing does not release everyone else's outputs.
Inventory exact generated candidates and retained hardlinks, then acquire all
actual Cargo locks used by that toolchain/profile exclusively. Recheck live users
while holding those locks, and keep both locks and the no-new-users coordination
in effect through deletion. Cargo locks alone do not protect later execution by
unmanaged consumers. If any user, lock or ownership is uncertain, do not delete.
Never remove lock files, source, reports, captures or other retained user outputs.
Record before/after filesystem free bytes separately from logical/allocated file
estimates and concurrent activity. Release the cleanup window explicitly afterward.

After integration is activated, replace legacy low-space deletion advice in the
consumer instructions with this policy, then have an idle worker adopt the common
store for its next genuinely needed command. Include the entire build/execution
pipeline, measure growth, and keep initial adoption to one invocation at a time.
Do not create another cold target merely to demonstrate adoption. Retire legacy
caches only through the independent review above; do not migrate live storage.
As of 2026-10-04, integration refresh remains queued behind PR #71 and draft PR
publication approval is pending. These are prepared instructions, not evidence
that existing workers already use the managed workflow.

### Validation status

Latest validation (revision `f91f3ff`, 2026-10-03): the full lightweight lifecycle
matrix passed on Windows, macOS and Ubuntu, including admission and checkout
invalidation. The formerly proposed worker world-suite pilot was dropped after
that worker completed its 323 tests; no redundant world rebuild was run.

One needed `cargo xtask consistency` invocation then passed through the common
store with the 20 GiB admission floor: 1.922s elapsed, 0.01s queue wait, store
growth from zero to 13,365,248 allocated bytes. Filesystem free space changed from
47,132,459,008 to 47,120,547,840 bytes; concurrent filesystem activity means this
delta is not an isolated allocation measurement. Only slot 0 compiled xtask.
This validates the real outer bootstrap and a repository workflow, not broad
worker adoption or a full workspace footprint. The common v2 root is now
initialized; future adopters still need a genuinely needed invocation and capacity
coordination. No second cold slot, legacy-target cleanup or GPU work was started.

The historical validation notes below describe earlier checkpoints; their pending
platform status is superseded by this result. Full repository CI and PR approval
remain separate from the lightweight lifecycle and limited workflow evidence.

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
proven reclaimable. At that initial checkpoint, process inspection required elevated access and
shared-worker rollout was pending coordination with active builds/captures. The urgent recovery below later removed only the independently verified-idle
retired incremental cache. Shared-slot adoption remains separate from recovery.

`python3 scripts/test-cargo-storage.py` exercises actual process locks, two occupied
slots plus a queued third, nested reuse, exit status, cancellation, crash quarantine,
cleanup contention, symlink refusal, and a tiny real Cargo binary rebuilt across
checkout paths. Full workspace builds are unnecessary for scheduler regression.
The latest lifecycle matrix now verifies the Windows Job Object path as well.

Incremental-off and reduced-debug experiments remain opt-in. No optimizer, debug,
assertion, or profile defaults change. Measure cold and warm builds separately in
an isolated target (including disk size and elapsed time); a tiny launcher smoke
fixture cannot establish the Bevy workspace's build-speed/storage tradeoff.

Operational cost evidence (2026-10-03): mechanic-7 reported an 8m29s sky-capture
compile after incremental caches were cleared during the disk-full event; its
earlier test execution took about 20s. These are worker-reported observations,
not a controlled before/after compilation measurement or independently verified
cleanup causality. Compilation and test execution are different phases, so these
times do not establish a slowdown ratio. Evaluate retained incremental caches,
cache eviction, incremental-off, and reduced debug information with separate cold
and warm rebuild measurements before changing defaults. Reclamation can impose a
substantial rebuild cost; it is not free. Active sky-test and capture storage must
remain untouched during this evaluation.

Rollout blocker found in subsequent validation: two pre-created checkouts with
the same package name/version can reuse the wrong top-level executable after a
successful Cargo build. The lease correctly prevents replacement during the
first checkout's build-and-execution pipeline, but does not establish checkout
freshness after handoff. The failing regression
`test_checkout_cannot_replace_binary_between_build_and_execution` supersedes the
earlier sequential checkout-isolation claim below. That blocker is fixed and lifecycle CI is now green; broader adoption remains
opportunistic and existing active targets stay in place.
Before the fix, the original regression failed with `AssertionError: 'first' !=
'second'` after B's successful build. Reassignment invalidation now passes locally:
pre-created unchanged A/B/A sources with incompatible library return types, nested
`cargo xtask` bootstrap/build, queued build-and-execution pipelines, and a fourth
warm A invocation. The test does not touch source files between invocations or
use delays to establish build correctness; its short delay checks only queuing.

Reassignment measurement on Cargo 1.97.1/macOS arm64 (single dependency-free
binary, not representative of Bevy): cold A 0.942s, reassigned B 0.616s,
reassigned A 0.660s, warm A 0.225s. Each completed invocation retained 1,097,728
allocated file bytes in its one Cargo subtree; A/B caches did not accumulate.
These single observations include launcher overhead and do not establish a
general slowdown ratio. Eighteen local lifecycle tests pass (one Windows-only
case skipped); removing only reassignment invalidation reproduces
`AssertionError: 'first' != 'second'` on pinned Cargo 1.97.1. Subsequent all-platform verification passed as recorded above; this does not
constitute broad worker adoption.

Mechanic-5 separately reported a shared-target `cargo check -p mechanic-physics
--all-targets` success followed by two `cargo test -p mechanic-physics treads`
failures resolving `mechanic_core::CompiledTreads` and `LocalCollider.treads`.
Verbose test output referenced `libmechanic_core-ea127a26adde3ca3.rlib`; a local
target resolved the symptom. Another checkout overwriting that rlib is the
worker's hypothesis: interleaving and hash causality were not observed. The tiny
isolation regression now exercises nested xtask running check-all-targets, an
integration test requiring checkout-specific struct fields, then build/direct
execution across concurrently submitted and alternating A/B/A pipelines.

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
  no private workspace dependencies. Linux/Windows subsequently passed in the latest lifecycle matrix.

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

Separately, mechanic-5 reported deleting only its own `target/debug/incremental`
after compilation while CPU tests still ran, estimating about 15 GB via `du -sh`
and reporting about 65 GB free afterward. This was not our audited recovery and
is not added to the measured 13.59 GiB free-space gain. Mechanic-6 reported no
deletion. The original ENOSPC path/cause remains unresolved; these observations
do not prove what caused the earlier recovery. Future cleanup requires exclusive
protection through dependent execution as well as compilation.

Mechanic-7 later attributed an earlier cleanup during the first reported ENOSPC
(approximately 17:5x) to deleting shared
`/Users/jens/repos/mechanic/target/*/incremental` (about 18 GB reported), plus
three of its temporary capture directories (about 11 MB reported). These are
worker-reported amounts, not audited unique physical-byte deltas or evidence of
exclusive locking. They are not summed with the measured recovery above and do
not establish the original ENOSPC cause or which action restored capacity.

Mechanic-6 subsequently reported deleting only
`/tmp/mechanic-62-capture-app` (217,896,784 logical bytes) and
`/tmp/mechanic-62-readback-app` (219,209,264 logical bytes): 437,106,048 logical
bytes total, with APFS physical recovery unmeasured. It reported preserving final
binaries, logs and visual evidence and leaving the shared target untouched.
This later deletion cannot explain the earlier 58 GiB available-space reading
and is not added to the audited physical recovery. Its retained logs contain no
ENOSPC/no-space/disk-full message, and it cannot verify an original failed path.
The original incident's path and cause are therefore recorded as unknown; no
further request for that unavailable evidence is pending.

New incident reported on 2026-10-04: mechanic-9 reported removing
`/Users/jens/repos/mechanic/target/debug/incremental`, estimating 52 GB, after free
space reached a reported 9.5 GB and citing the older repository disk guidance.
The deletion time, exclusive ownership/locking and physical free-space delta are
not confirmed. This is a separate worker report, not part of the audited 13.59 GiB
recovery and not a measured total to add to it. No cleanup was performed by this
task in response. The report shows that unmanaged legacy growth and independent
cleanup remain operational gaps despite the launcher admission checks and limited
pilot; broad adoption is not complete.

The expanded lifecycle suite defines 16 tests, including hardlink accounting and a
Windows-specific descendant-handle test. Lightweight Linux/macOS CI passed before
the Windows fix. Windows CI exposed a native Python `os.execvpe` crash; its gated
child now uses `subprocess` within the Job Object. The latest Windows lifecycle job verifies that fix.

The subsequent Windows smoke trace confirmed Cargo had run its expected binary,
but MSVC's `vctip.exe` remained alive. Completion now also drains this known
compiler-tree telemetry helper after every ordinary job member exits; binaries
outside the compiler tree are not treated as disposable services.
