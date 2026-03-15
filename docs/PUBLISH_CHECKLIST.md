# Publish Checklist

Last updated: 2026-03-13

This document is the authoritative release checklist for `ndatafusion`.

## Current Publication Posture

`ndatafusion` is not crates.io-publishable today.

Reason:

1. `datafusion` is still pinned to a git revision on `main` so `ndatafusion` can align with Arrow
   58 and the released `nabled` / `ndarrow` contract.
2. Until an Arrow-58-compatible DataFusion release exists on crates.io, `cargo publish` for
   `ndatafusion` remains intentionally blocked.
3. Git tag and GitHub releases are still valid today; the blocker is specifically crates.io
   publication.

Use this checklist for release hardening now, and as the publish gate once the dependency blocker is
removed.

## Pre-Release Gates

1. `just checks`
2. `cargo llvm-cov clean --workspace && cargo llvm-cov --no-default-features --lib --tests --no-report && cargo llvm-cov report --summary-only --ignore-filename-regex '.*/examples/.*|.*/src/udf/docs.rs' --fail-under-lines 90`
3. `cargo doc --no-default-features --no-deps`
4. Verify README quick-start examples against the current SQL constructor surface.
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
7. Confirm the DataFusion dependency source:
   - if it is still a git dependency, do not attempt crates.io publication
   - if it has moved to a compatible crates.io release, continue with publish validation

## Publish Validation

Run these only once the DataFusion dependency blocker is removed:

1. `cargo package --allow-dirty --no-default-features`
2. `cargo publish --dry-run --no-default-features`
3. Confirm docs.rs posture still matches:
   - `package.metadata.docs.rs.no-default-features = true`
   - crate-level docs still describe the constructor-backed `Float32` / `Float64` real-valued
     contract accurately

## Release Notes Minimum

Every release should call out:

1. new UDFs or result contracts
2. new or changed SQL constructors
3. feature-forwarding changes
4. dependency changes affecting Arrow/DataFusion compatibility
5. any remaining known limitations or intentional non-goals

## First Publish Exit Criteria

`ndatafusion` is ready for its first crates.io publish only when all are true:

1. the dependency blocker above is gone
2. the admitted v1 catalog is complete for the chosen scope
3. release hardening gates are green
4. README and crate docs reflect the actual published surface, not a planned one
