# Repository Guidelines

## Project Structure & Module Organization

Mechanic is a Rust 2024 Cargo workspace. Dependencies point one way:
`core ← world ← physics`, `core + world ← gpu`, and everything `← bench, app`.

| Crate | Owns |
|---|---|
| `crates/mechanic-core` | Construction graph, geometry, compilation, compiled dynamics schedules, creation documents, and the shared units and constants every other crate imports. No filesystem access and no GPU or engine types: authored bearing, suspension, and piston assets are defined here as plain geometry with their material-slot legend, and the app turns them into meshes and materials. |
| `crates/mechanic-world` | Terrain generation, edits, meshing, streaming, queries, material clumps, and world documents with their on-disk store. |
| `crates/mechanic-physics` | CPU solvers. They run the app by default. |
| `crates/mechanic-gpu` | GPU runtime, ABI structs, and compute shaders; WGSL kernels live under `src/kernels`. Selected with `MECHANIC_PHYSICS=gpu`. |
| `crates/mechanic-bench` | Reproducible headless performance scenarios; every binary emits JSONL. |
| `crates/mechanic-app` | The interactive Bevy prototype. |
| `crates/bevy_mosaic` | Runs the Mosaic GUI framework inside a Bevy app. |
| `crates/xtask` | The task runner behind `cargo xtask`. Dependency-free. |

`vendor/` holds patched upstream crates; each carries a `MECHANIC-PATCH.md`
naming its source revision and the change. `scripts/` holds Python capture and
benchmark tooling with `test-*.py` regression tests. Architectural decisions
and milestone status are documented in `docs/`, indexed by `docs/README.md`.

## Domain Terminology

- A construction "block" is one grid unit: 25 cm per side. An 8 × 8 block square is therefore 2 × 2 m.
- Construction material textures allocate 512 × 512 pixels to each block. The current 3072 × 3072 maps span six blocks, or 1.5 m, per repeat.

## Build, Test, and Development Commands

`cargo xtask` is the single entry point for checks; CI runs `cargo xtask ci`.

- `cargo xtask ci` runs every task below except `wgsl`, `budgets`, and `bench-smoke`.
- `cargo xtask fmt` verifies formatting; `cargo xtask fmt --fix` applies it.
- `cargo xtask consistency` verifies the repository conventions below that no compiler checks.
- `cargo xtask lint` runs Clippy on every target with warnings denied.
- `cargo xtask test` runs core, world, physics, GPU, app, benchmark, and WGSL validation tests without stopping at the first failure.
- `cargo xtask doc` builds the API docs with rustdoc warnings denied.
- `cargo xtask wgsl` validates the compute kernels without a GPU.
- `cargo xtask scripts-test` runs the Python regression tests under `scripts/`.
- `cargo xtask budgets` enforces the app tests' wall-clock budgets, alone and serially. Ordinary test runs check those tests' behaviour only, because a parallel run on a busy machine says nothing about a frame budget.
- `cargo xtask bench-smoke` runs the quick headless benchmark. Use `cargo run -p mechanic-bench --release -- --scenario <name>` for performance measurements such as `four_bar` or `dense_100k`.
- `cargo run -p mechanic-app` launches the construction and simulation prototype.

Tasks forward extra arguments to the underlying command, for example `cargo xtask test -p mechanic-core`.

### Disk Space

Worker builds use `python3 scripts/cargo-storage.py cargo <arguments>` (Windows:
`python scripts/cargo-storage.py`). This acquires one of two reusable slots
before the outer Cargo invocation, including `cargo xtask` bootstrap. For example:

- `python3 scripts/cargo-storage.py cargo xtask test -p mechanic-core`
- `python3 scripts/cargo-storage.py cargo run -p mechanic-app`
- `python3 scripts/cargo-storage.py cargo xtask lint`

Unset a previously configured `CARGO_TARGET_DIR` when adopting the launcher.
Nested commands inherit the lease. For a build followed by direct binary runs,
wrap the **entire** pipeline with `cargo-storage.py run -- <command>`, and resolve
binaries beneath the inherited `CARGO_TARGET_DIR`. Keep captures/reports outside
that directory. Never run a slot binary after its lease ends; rebuild under a
lease or copy a durable measurement binary while holding the lease.

- Existing builds finish on their existing targets before adoption. Never clean
  another worker's target or change global Cargo configuration.
- Use `cargo-storage.py status` for owners and sizes, and `cargo-storage.py clean`
  for a preview. Add `--apply` only after reviewing it. Cleanup takes the same
  exclusive locks as builds and skips active or quarantined slots.
- Do not daemonize or detach child processes from a leased command. A crashed
  supervisor quarantines its slot; `recover` requires proof of a later OS boot.
- Keep release/profiling builds for tasks that need them. No profile defaults
  change. Check `df -h .` before a large build; two slots are not a byte quota.
- Explicit isolated measurements and ordinary ephemeral CI can use
  `cargo-storage.py --unmanaged cargo ...`. Isolated reference-builder scripts
  already create and preserve their own source-matched binaries; their storage
  is outside the slot budget and must be inventoried separately.

See [Cargo storage](docs/cargo-storage.md) for retention/budget controls, platform
limits, crash recovery, and staged rollout. Never blindly delete legacy targets.

## Coding Style & Naming Conventions

Follow standard `rustfmt` output with four-space indentation. Use `snake_case` for modules, functions, variables, and test names; `UpperCamelCase` for types and traits; and `SCREAMING_SNAKE_CASE` for constants. Keep CPU/GPU layouts synchronized when editing ABI structs or WGSL bindings. Unsafe Rust is forbidden, public APIs should be documented, and Clippy `all` plus `pedantic` warnings are enabled workspace-wide.

## Structural Conventions

These hold everywhere; `cargo xtask consistency` enforces the mechanical ones.

- **Modules.** A module with children is `foo.rs` beside `foo/`. Never `mod.rs`.
- **Tests.** One `#[cfg(test)] mod tests` per file, as its last item. Keep it inline below roughly 300 lines; above that it becomes `foo/tests.rs`, split by theme under `foo/tests/` when it grows further. No `*_tests.rs` files. Fixtures shared across modules live in the crate's `#[cfg(test)] mod testing`.
- **Crate roots.** `lib.rs` is crate docs, an alphabetical `mod` list, then alphabetical `pub use` blocks. It defines no items. Re-exports are flat; the one exception is a module written to be glob-imported, such as a `prelude`, which is its own file.
- **Type suffixes.** `*Error` is an `Err` payload and derives `thiserror::Error`. `*Outcome` is a successful return with several variants; `*Result` is reserved for `std::result::Result` aliases and otherwise unused. `*Config` is parameters supplied by code; `*Settings` is preferences the player persists. `*Doc` is a serialized file row. `*Spec` is validated authoring input. Use `Cpu*` and `Gpu*` prefixes whenever both forms of a concept exist.
- **Constants.** A physical quantity has one owner, `mechanic_core::units` when more than one crate needs it. Do not define alias constants or repeat a literal; import the owner.
- **Lint suppressions.** Use `#[expect(lint)]`, so a suppression that stops applying fails the build, and give it a `reason` unless the item makes the cause evident. Where a lint fires only under some `cfg`, or Clippy never marks the expectation fulfilled (module-level `clippy::wildcard_imports`), use `#[allow(lint, reason = "…")]`. A bare `#[allow]` is never acceptable.
- **Configuration.** The app names every environment variable it reads in `env.rs` and reads them through that module's accessors; no other file spells a `MECHANIC_*` name. `docs/environment.md` lists every variable any crate reads.
- **Manifests.** Every dependency, internal crates included, is declared in `[workspace.dependencies]` and consumed with `workspace = true`.
- **Model and view (app).** A root feature module owns the ECS state and the systems that change the world. Its `ui/<feature>` counterpart owns the Mosaic view, the plain-data snapshot that view renders, the intents it sends back, and the `push_*` system that builds the snapshot from the ECS. A view never touches a resource, and nothing in the world reads a view.
- **Frame order (app).** Systems are ordered through the sets in `schedule.rs`, never by naming another module's system function.

## Dependency Quality

Treat Mosaic as a first-party UI dependency, not a fixed limitation to work around. Do not add Mechanic-specific hacks, duplicate rendering paths, or brittle layout tricks to compensate for missing or incorrect Mosaic behavior. Identify the capability or fix that belongs in Mosaic, propose it explicitly, and prefer implementing and consuming that upstream change before continuing the Mechanic feature.

## Pre-production Compatibility

Mechanic is pre-production. Do not add backward-compatibility readers, migrations, legacy fallbacks, or backup formats unless explicitly requested. Replace formats directly and update fixtures and tests.

## Versioning

Mechanic is in alpha. Keep the major version at `0`; never increment it while this policy applies.

- Bump the **minor** version for breaking changes, including incompatible public API, persisted format, or protocol changes. Reset patch to `0`, for example `0.1.4` → `0.2.0`.
- Bump the **patch** version for everything else, including compatible features, fixes, refactors, documentation, and tooling changes, for example `0.1.4` → `0.1.5`.
- Bump once per completed, coherent task that changes repository files, after its implementation and verification are complete and before committing or handing the work back. Do not wait for a release or an explicit request to bump.
- A task may span several commits or review iterations; follow-up fixes within that task share its bump. A new, independent task gets a new bump. Read-only investigation and questions do not trigger a bump.
- Choose the highest applicable level across the task: any breaking change requires a minor bump; otherwise use patch. If a task becomes breaking after a patch bump, replace that bump with a minor bump from the task's starting version.
- Update `[workspace.package].version` in the root `Cargo.toml` and keep workspace package versions in `Cargo.lock` synchronized.

## Testing Guidelines

Add focused regression tests beside the code being changed. Name tests after observable behavior, for example `off_centre_external_impulse_changes_linear_and_angular_motion`. Exercise both graph compilation and GPU behavior when a change crosses that boundary. Hardware-specific GPU tests may require a real adapter; report the adapter and command used. Mechanic is experimental: tests assert observable behaviour with tolerances suited to a game, not solver internals. Captured solver states, iteration counts, and storage bounds belong in bench reports, not `cargo test`. Do not claim scale-gate completion unless the exact body count, kernel coverage, failure flags, throughput, and p95 requirements in `README.md` are satisfied.

## Commit & Pull Request Guidelines

Use short, imperative Conventional Commit subjects, matching history (for example, `feat: add GPU mechanism simulator`). Keep commits scoped to one coherent change. Pull requests should explain the behavior and rationale, list verification commands, link relevant issues, and include screenshots or benchmark JSONL when UI or performance behavior changes. Call out GPU/ABI compatibility changes and any tests that require specific hardware.
