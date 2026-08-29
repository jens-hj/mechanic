# Repository Guidelines

## Project Structure & Module Organization

Mechanic is a Rust 2024 Cargo workspace. Keep shared construction data structures, geometry, and compilation logic in `crates/mechanic-core`. GPU runtime code, ABI definitions, and compute shaders belong in `crates/mechanic-gpu`; WGSL kernels live under `crates/mechanic-gpu/src/kernels`. The interactive Bevy prototype is in `crates/mechanic-app`, while reproducible performance scenarios are in `crates/mechanic-bench`. Architectural decisions and milestone status are documented in `docs/`. Tests are generally colocated with their Rust modules in `#[cfg(test)]` blocks.

## Domain Terminology

- A construction "block" is one grid unit: 25 cm per side. An 8 × 8 block square is therefore 2 × 2 m.
- Construction material textures allocate 512 × 512 pixels to each block. The current 3072 × 3072 maps span six blocks, or 1.5 m, per repeat.

## Build, Test, and Development Commands

- `cargo build --workspace` builds every crate with the pinned toolchain.
- `cargo run -p mechanic-app` launches the construction and simulation prototype.
- `cargo test --workspace` runs core, GPU, app, benchmark, and WGSL validation tests.
- `cargo clippy --workspace --all-targets -- -D warnings` enforces all configured lints without warnings.
- `cargo fmt --all -- --check` verifies formatting; use `cargo fmt --all` to apply it.
- `cargo run -p mechanic-bench -- --scenario smoke` runs the quick headless benchmark. Use `--release` for performance measurements such as `four_bar` or `dense_100k`.

## Coding Style & Naming Conventions

Follow standard `rustfmt` output with four-space indentation. Use `snake_case` for modules, functions, variables, and test names; `UpperCamelCase` for types and traits; and `SCREAMING_SNAKE_CASE` for constants. Keep CPU/GPU layouts synchronized when editing ABI structs or WGSL bindings. Unsafe Rust is forbidden, public APIs should be documented, and Clippy `all` plus `pedantic` warnings are enabled workspace-wide.

## Dependency Quality

Treat Mosaic as a first-party UI dependency, not a fixed limitation to work around. Do not add Mechanic-specific hacks, duplicate rendering paths, or brittle layout tricks to compensate for missing or incorrect Mosaic behavior. Identify the capability or fix that belongs in Mosaic, propose it explicitly, and prefer implementing and consuming that upstream change before continuing the Mechanic feature.

## Pre-production Compatibility

Mechanic is pre-production. Do not add backward-compatibility readers, migrations, legacy fallbacks, or backup formats unless explicitly requested. Replace formats directly and update fixtures and tests.

## Testing Guidelines

Add focused regression tests beside the code being changed. Name tests after observable behavior, for example `off_centre_external_impulse_changes_linear_and_angular_motion`. Exercise both graph compilation and GPU behavior when a change crosses that boundary. Hardware-specific GPU tests may require a real adapter; report the adapter and command used. Do not claim scale-gate completion unless the exact body count, kernel coverage, failure flags, throughput, and p95 requirements in `README.md` are satisfied.

## Commit & Pull Request Guidelines

Use short, imperative Conventional Commit subjects, matching history (for example, `feat: add GPU mechanism simulator`). Keep commits scoped to one coherent change. Pull requests should explain the behavior and rationale, list verification commands, link relevant issues, and include screenshots or benchmark JSONL when UI or performance behavior changes. Call out GPU/ABI compatibility changes and any tests that require specific hardware.
