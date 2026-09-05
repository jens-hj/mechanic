# wgpu 29.0.4 fence synchronization backport

`wgpu-core/` and `wgpu-hal/` are crates.io 29.0.4 package sources, originally
from gfx-rs/wgpu revision `e99f5305ded96ff7006f0714d043a7f735bd45c2` (retained
`.cargo_vcs_info.json`). Each retains its MIT and Apache-2.0 licenses.

The source patch is upstream commit
[a1fceed0710fafed9abe0f7fb16c0a19bb390bf2](https://github.com/gfx-rs/wgpu/commit/a1fceed0710fafed9abe0f7fb16c0a19bb390bf2),
“Move fence synchronisation into wgpu-hal” (#9475). Its original patch is
[`wgpu-fence-upstream.patch`](wgpu-fence-upstream.patch).

Both dependencies are patched together in the root Cargo manifest. Public wgpu,
Naga, Bevy and Mosaic versions are unchanged. HAL submit accepts a shared fence;
Metal pending buffers, GLES syncs, and Vulkan fence pools synchronize internally.
Metal clones the command buffer before releasing the pending-list lock and
waiting. Vulkan/GLES retain shared ownership while a waiter uses a native fence,
preventing its reset/destruction. Acquisition no longer owns a core fence lock.

## Adaptations to 29.0.4

- Keep 29's monolithic `Queue::submit`; do not import newer `PendingSubmission`,
  `SubmissionResult`, or command encoder refactors. Hold command indices through
  HAL submit, successful-index publication and lifetime registration; release
  pending writes and command indices before maintenance and callbacks.
- 29 additionally reads the fence in `Queue::drop`. Use the shared fence there,
  retaining every original timeout, device-loss branch, and destruction check.
- `Device::maintain` takes no fence guard; its direct (unnamed in 29) guard type
  is removed. The upstream `FenceReadGuard`/`FenceWriteGuard` aliases/re-export
  did not yet exist. Keep upstream's synchronized validity snapshot: once invalid
  under command indices, subsequent submitters are rejected, and every previously
  accepted submission is already lifetime-tracked before resource release.
- 29 presents in `Surface::present`, not `Queue::present`. Serialize its HAL
  present with submits/other presents using the same command-index write lock.
- Adapt GLES wait to 29's existing timeout clamp and exact-value assertion;
  clone the native sync and drop the pending-list guard before the client wait.
  Preserve 29's `wait_value` completion publication (equal to its asserted value).
- HAL trait context and Vulkan imports differ; apply the same shared reference
  and `RwLock` changes without other interface/refactor changes.
- 29's optional lock validator was incomplete: `Mutex::into_inner` was missing,
  and several existing acquisition/lifetime/presentation edges were undeclared.
  Add only edges exercised by this regression plus the command-index edges
  required by this patch. Keep rank assertions and stack-release checks active.
  This validates the tested paths, not all historical 29 lock paths.
- Omit the upstream root changelog hunk; retain provenance here. Dependency-local
  regression and CI commands are additions to the upstream patch.

No app unsafe code, ABI, solver, scheduling, queue depth, or file-format changes.
To remove the backport, remove both crates.io patches and vendor directories and
update Cargo.lock together. The capture recorder is independent.

## Regression

From the repository root:

```sh
cargo test --manifest-path vendor/wgpu-core/Cargo.toml --features noop,wgsl \
  --config 'patch.crates-io.wgpu-hal.path="vendor/wgpu-hal"' \
  acquisition_does_not_block -- --nocapture
RUSTFLAGS='--cfg wgpu_validate_locks' cargo test \
  --manifest-path vendor/wgpu-core/Cargo.toml --features noop,wgsl \
  --config 'patch.crates-io.wgpu-hal.path="vendor/wgpu-hal"' \
  acquisition_does_not_block -- --nocapture
```

The test pauses inside `DynSurface::acquire_texture` and calls actual core submit
on another thread. Arrival/release channels provide bounded rendezvous; release
runs before any assertion, with a five-second HAL watchdog. A barrier starts
concurrent producers, polling and presentation. Completion callbacks re-enter
submit and all 100 callbacks must run. A process watchdog fails a deadlocked test
rather than leaving CI hung. NOOP tests prove core synchronization, not Metal WSI.

On 2026-09-05 the same test injected into pristine 29.0.4 failed at the one-second
submission watchdog (`submission stalled behind HAL acquisition`), then cleaned
up. The candidate passed normally and with lock validation. Raw local logs are
listed in `docs/performance-todo.md`. macOS/Linux/Windows CI now runs both modes;
remote CI requires a separate publication request and has not been run here.
