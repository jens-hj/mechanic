# Repository Guidelines

## Project Structure & Module Organization

Mechanic is a Rust 2024 Cargo workspace. Dependencies point one way:
`core ← world ← physics`, `core + world ← gpu`, and everything `← bench, app`.

| Crate | Owns |
|---|---|
| `crates/mechanic-core` | Construction graph, geometry, compilation, compiled dynamics schedules, creation documents, and the shared units and constants every other crate imports. No filesystem access and no GPU or engine types: authored bearing and suspension assets are defined here as plain geometry with their material-slot legend, and the app turns them into meshes and materials. |
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
- **Model and view (app).** A root feature module owns ECS state and systems; its `ui/<feature>.rs` counterpart is a Mosaic view with no model types. Systems are ordered through the sets in `schedule.rs`, not by naming another module's system function.

## Dependency Quality

Treat Mosaic as a first-party UI dependency, not a fixed limitation to work around. Do not add Mechanic-specific hacks, duplicate rendering paths, or brittle layout tricks to compensate for missing or incorrect Mosaic behavior. Identify the capability or fix that belongs in Mosaic, propose it explicitly, and prefer implementing and consuming that upstream change before continuing the Mechanic feature.

## Pre-production Compatibility

Mechanic is pre-production. Do not add backward-compatibility readers, migrations, legacy fallbacks, or backup formats unless explicitly requested. Replace formats directly and update fixtures and tests.

## Testing Guidelines

Add focused regression tests beside the code being changed. Name tests after observable behavior, for example `off_centre_external_impulse_changes_linear_and_angular_motion`. Exercise both graph compilation and GPU behavior when a change crosses that boundary. Hardware-specific GPU tests may require a real adapter; report the adapter and command used. Mechanic is experimental: tests assert observable behaviour with tolerances suited to a game, not solver internals. Captured solver states, iteration counts, and storage bounds belong in bench reports, not `cargo test`. Do not claim scale-gate completion unless the exact body count, kernel coverage, failure flags, throughput, and p95 requirements in `README.md` are satisfied.

## Commit & Pull Request Guidelines

Use short, imperative Conventional Commit subjects, matching history (for example, `feat: add GPU mechanism simulator`). Keep commits scoped to one coherent change. Pull requests should explain the behavior and rationale, list verification commands, link relevant issues, and include screenshots or benchmark JSONL when UI or performance behavior changes. Call out GPU/ABI compatibility changes and any tests that require specific hardware.
