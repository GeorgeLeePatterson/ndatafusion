# AGENTS.md

## Project Intent

`ndatafusion` exists to expose `nabled`'s linear algebra, machine learning, and `ndarrow`-based Arrow interfaces as DataFusion functions.

The guiding principle for all design and implementation work is:

> algebraic, homomorphic, compositional built from denotational semantics

Keep that principle visible in API design, function composition, trait boundaries, and execution semantics.

## Core Workflow Rules

1. Treat `.justfile` as the canonical task runner.
2. Run `just checks` after every legitimate checkpoint.
3. `just checks` is not optional. It runs:
   - `cargo +nightly fmt -- --check`
   - `cargo +nightly clippy --all-features --all-targets`
   - `cargo +stable clippy --all-features --all-targets -- -D warnings`
   - all unit and integration tests via `just test`
4. Keep the `e2e` integration test target present, because `.justfile` expects `cargo test --test e2e`.
5. Before any commit is pushed, line coverage must be greater than 90%. Use `cargo llvm-cov` via the coverage commands in `.justfile` to verify this.

## Code Standards

1. Use the new module layout only. Do not create `mod.rs`.
2. Clippy pedantic lints are enabled in `Cargo.toml` and must be addressed.
3. Do not silence pedantic lints with `allow` unless there is no viable alternative.
4. Prefer fixing the underlying code. If a conscious assertion is needed, `expect` is preferable to `allow`.
5. Any unavoidable exception must be narrowly scoped and justified in code review terms.

## Dependency And Feature Rules

1. `ndatafusion` must expose and forward all `nabled` features cleanly so downstream users can enable them from this crate.
2. As of this scaffold, the relevant `nabled` features are:
   - `test-utils`
   - `arrow`
   - `blas`
   - `lapack-provider`
   - `openblas-system`
   - `openblas-static`
   - `netlib-system`
   - `netlib-static`
   - `magma-system`
   - `accelerator-rayon`
   - `accelerator-wgpu`
3. When dependency work touches Arrow compatibility, verify the `datafusion` and `nabled` Arrow versions explicitly before changing versions. The local DataFusion source at `../../packages/datafusion` is available for reference, and `../nabled` is the source of truth for the upstream feature surface.

## Local References

1. Use `../nabled` as the primary implementation reference for the functionality being exposed.
2. Use `../../packages/datafusion` to validate DataFusion interfaces and compatibility details when needed.
3. Use `../../packages/datafusion-functions-json` as a reference for how a standalone crate exposes functions to DataFusion.
