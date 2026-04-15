# Publish Checklist

Last updated: 2026-04-15

This document is the authoritative release checklist for `ndatafusion`.

## Current Publication Posture

`ndatafusion` is crates.io-publishable today.

Validated on the current tree:

1. `datafusion = "53"` resolves from crates.io and stays aligned with Arrow 58, `ndarrow 0.0.3`,
   and `nabled 0.0.7`.
2. `just checks` passes.
3. line coverage is `90.01%`.
4. `cargo doc --no-default-features --no-deps` passes.
5. `cargo package --allow-dirty --no-default-features` passes.
6. `cargo publish --dry-run --allow-dirty --no-default-features` passes.
7. the tag-triggered release workflow can publish to crates.io when `CARGO_REGISTRY_TOKEN` is
   configured in GitHub repo secrets.

The `--allow-dirty` flag above was only needed because dependency-source updates in `Cargo.toml`
and `Cargo.lock` were still uncommitted during validation. Use a clean tree for the real publish.

## Pre-Release Gates

1. `just checks`
2. `cargo llvm-cov clean --workspace && cargo llvm-cov --no-default-features --lib --tests --no-report && cargo llvm-cov report --summary-only --ignore-filename-regex '.*/examples/.*|.*/src/udf/docs.rs' --fail-under-lines 90`
3. `cargo doc --no-default-features --no-deps`
4. Verify README install snippet and quick-start examples against the current release and SQL
   constructor surface.
5. Verify `docs/STATUS.md`, `docs/EXECUTION_TRACKER.md`, and `docs/CAPABILITY_MATRIX.md` reflect
   the current state truthfully.
6. Verify `Cargo.toml` metadata:
   - `version`
   - `description`
   - `repository`
   - `documentation`
   - `keywords`
   - `categories`
   - forwarded feature list
7. Confirm the dependency alignment still holds:
   - `datafusion` resolves from crates.io on the intended release line
   - Arrow remains aligned across `datafusion`, `ndarrow`, and `nabled`
8. If the GitHub release workflow will own the real publish, confirm `CARGO_REGISTRY_TOKEN` is
   configured in GitHub repo secrets.

## Publish Validation

Run these from a clean tree immediately before the real publish, or before pushing the release tag
that will trigger automated publish:

1. `cargo package --no-default-features`
2. `cargo publish --dry-run --no-default-features`
3. Confirm docs.rs posture still matches:
   - `package.metadata.docs.rs.no-default-features = true`
   - crate-level docs still describe the constructor-backed `Float32` / `Float64` real-valued
     contract accurately
4. If the GitHub release workflow will publish, confirm `CARGO_REGISTRY_TOKEN` is set.

## Release Notes Minimum

Every release should call out:

1. new UDFs or result contracts
2. new or changed SQL constructors
3. feature-forwarding changes
4. dependency changes affecting Arrow/DataFusion compatibility
5. any remaining known limitations or intentional non-goals

## First Publish Exit Criteria

`ndatafusion` is ready for its first crates.io publish only when all are true:

1. `datafusion`, Arrow, `ndarrow`, and `nabled` remain aligned on the intended crates.io release line
2. the admitted v1 catalog is complete for the chosen scope
3. release hardening gates are green
4. README and crate docs reflect the actual published surface, not a planned one
