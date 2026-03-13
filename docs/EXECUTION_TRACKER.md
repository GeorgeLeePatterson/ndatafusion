# Execution Tracker

Last updated: 2026-03-13

## Purpose

This is the operational companion to `docs/CAPABILITY_MATRIX.md`.

Use this file to resume work quickly after context compaction without re-auditing the full
codebase.

## Usage Rules

1. Treat this file as the canonical `Done / Next / Needed` tracker.
2. Update it in the same change set when non-trivial implementation work lands.
3. Only do a full repository re-assessment if:
   - this file is stale,
   - statuses conflict with observed code, or
   - architectural direction changed.

## Current Baseline

1. Repository scaffold exists as a single crate with a thin registration/catalog entry surface.
2. DataFusion is pinned to an Arrow-58-compatible git revision on `main`.
3. `.justfile` quality gates are present and passing on the scaffold baseline.
4. The repository now has authoritative planning docs and AGENTS bootstrap instructions.
5. `ndatafusion` now depends on published `nabled 0.0.7` and `ndarrow 0.0.3`.
6. A substantial `f64`-first numerical DataFusion catalog now exists across vector, matrix,
   decomposition, sparse, tensor, and ML/stat slices.
7. The next local work is SQL constructors plus publish hardening over the expanded catalog.

## V1 Publish Gate (Ordered, Required)

`ndatafusion` is considered ready for a v1 publish only when all items below are complete, in
order:

1. Lower-layer Arrow contracts are batch-native, null-correct, and performance-appropriate for the
   admitted numerical domains.
2. Dependency and feature contract is explicit and stable (`nabled`, `ndarrow`, Arrow, DataFusion,
   feature forwarding), and published upstream releases exist for the required lower-layer fixes.
3. `register_all`, exported UDF constructors, and common error/metadata utilities are stable.
4. Dense vector and matrix function slices are implemented and integration-tested.
5. Struct-returning decomposition and solver contracts are implemented and documented.
6. Sparse, tensor, and ML/stat capability parity is implemented for the admitted v1 surface.
7. SQL constructors or normalizers exist for the canonical numerical value contracts.
8. Docs, examples, tests, feature-matrix checks, and coverage are publish-ready.

## Done

1. `D-001`: Project scaffold exists and `just checks` passes on the baseline crate.
2. `D-002`: DataFusion is pinned to Arrow 58 via git revision `8d9b080882179b618a2057e042fc32865f6484b4`.
3. `D-003`: Governance baseline now exists via `docs/README.md`, `docs/DECISIONS.md`,
   `docs/CAPABILITY_MATRIX.md`, `docs/EXECUTION_TRACKER.md`, `docs/ARCHITECTURE.md`,
   `docs/STATUS.md`, and aligned root `AGENTS.md`.
4. `D-004`: Cross-layer gap analysis is now recorded: `ndarrow` needs explicit contract hardening,
   `nabled::arrow` needs batch-native/null-aware/performance-aligned expansion, and `ndatafusion`
   should prefer direct lower-layer delegation with codec lifting only as a fallback.
5. `D-005`: The canonical ingress model is now concept-first: mathematical object families own one
   canonical `rows-of-X` batch carrier, and standalone batching should prefer the same carriers
   that `ndatafusion` will use.
6. `D-006`: Upstream prerequisite checkpoints are now complete and published:
   - `ndarrow 0.0.3`
   - `nabled 0.0.7`
7. `D-007`: `ndatafusion` now depends on published `nabled 0.0.7`, enables `nabled/arrow`
   unconditionally, depends on `ndarrow 0.0.3` directly, and mirrors the remaining `nabled`
   feature flags one-for-one.
8. `D-008`: The initial local extension-crate scaffold now exists via `register_all`, `functions`,
   and `udfs`.
9. `D-009`: `N-004` through `N-007` are now covered by the first local catalog:
    - shared metadata, signature, and error helpers
    - dense vector row ops
    - batched dense matrix matmul and LU solve
   - struct-valued LU decomposition
   - sparse batch matvec
   - fixed-shape tensor last-axis reduction
   - matrix column means and linear regression
10. `D-010`: The `N-004` through `N-007` catalog is now checkpoint-safe:
    - contract-edge unit coverage exists for helpers and admitted UDF slices
    - `just checks` passes on the current tree
    - `cargo llvm-cov --no-default-features --summary-only --lib --test e2e`
      reports line coverage above 90%
11. `D-011`: `N-010` has now advanced substantially on the admitted `f64`-first surface:
    - matrix inverse, determinant, and log-determinant
    - Cholesky decomposition, solve, and inverse
    - QR and SVD struct-return contracts
    - matrix centering, covariance, correlation, and PCA
    - sparse batch dense matmat, transpose, and sparse matmat
    - fixed-shape and variable-shape tensor norm/normalize/dot/matmul expansions
    - the catalog now exposes 35 registered scalar UDFs with coverage still above 90%

## Next

1. `N-008` (`Layer 3`, `ndatafusion`): Add SQL constructors and normalizers for the canonical
   numerical value contracts.
2. `N-009` (`Layer 3`, `ndatafusion`): Harden the new catalog with examples, richer integration
   coverage, and publish-ready docs/release metadata.
3. `N-010` (`Layer 3`, `ndatafusion`): Finish the residual admitted type/domain parity work that
   still fits the `f64`-first, SQL-natural v1 contract while preserving direct batch delegation on
   hot paths.

## Needed

1. Constructor and normalization functions that let SQL callers build canonical numerical values
   from nested literals or arrays without bespoke Rust setup.
2. Explicit publish hardening: README examples, docs.rs viability, release notes, and coverage
   proof over the admitted v1 surface.
3. A post-v1 decision on whether UDAFs, window functions, planners, or rewrites are worth
   introducing for dataset-level or ergonomic workflows.
4. A post-v1 performance pass to reduce fallback lift/assembly overhead where direct batch
   delegation is still impossible.

## Round Scope Lock

1. This round starts local `ndatafusion` implementation after the upstream prerequisite releases.
2. The next execution round should start at `N-008`: constructors and normalizers.
3. Preserve the concept-first contract while the current catalog expands.
