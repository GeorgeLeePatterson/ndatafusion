# Execution Tracker

Last updated: 2026-04-15

## Purpose

This is the canonical `Done / Next / Needed` tracker for `ndatafusion`.

Use it to resume work without replaying the full implementation history.

## Current State

1. The crate now exposes a broad validated SQL catalog over `nabled` and `ndarrow`.
2. The real-valued admitted surface is implemented across the constructor, scalar, aggregate, and
   tensor-decomposition slices.
3. The first complex surface is implemented across vectors, matrices, PCA, tensors, and complex
   spectral / matrix-function helpers.
4. The first non-scalar expansion is implemented:
   - typed sufficient-statistics aggregate UDFs
   - ordered window usage through retractable aggregate state
   - the generic `unpack_struct` table function through `register_all_session`
5. The first planner pass is implemented via per-UDF simplify hooks.
6. The repository quality gates are green, line coverage is above `90%`, and publish validation
   now passes on the crates.io `datafusion 53.0.0` line.

## Done

1. Governance baseline: AGENTS, docs bootstrap, tracker discipline, and repository quality gates.
2. Upstream dependency alignment:
   - `ndarrow 0.0.3`
   - `nabled 0.0.7`
   - crates.io `datafusion 53.0.0`
3. Base crate shape:
   - `register_all`
   - `register_all_session`
   - public SQL-expression helpers
   - shared metadata, signature, and error layers
4. Constructor surface:
   - `make_vector`
   - `make_matrix`
   - `make_tensor`
   - `make_variable_tensor`
   - `make_csr_matrix_batch`
5. Real-valued scalar catalog:
   - dense vector, matrix, sparse, tensor, matrix-function, decomposition, matrix-equation, and
     ML/stat slices
6. Complex scalar catalog:
   - complex vectors
   - complex matrices
   - complex PCA
   - complex tensors
   - complex spectral and matrix-function slices
7. Additional scalar expansions:
   - named-function differentiation
   - named-function complex optimization
   - sparse factorization and preconditioners
   - tensor decomposition and tensor-train workflows
8. Aggregate catalog:
   - `vector_covariance_agg`
   - `vector_correlation_agg`
   - `vector_pca_fit`
   - `linear_regression_fit`
9. Aggregate design cleanup:
   - typed Arrow-native state fields
   - sufficient-statistics state
   - retractable window support
10. Table-function surface:
   - `unpack_struct`
11. Planner integration:
   - per-UDF simplify hooks for the admitted obvious rewrite cases
12. Documentation and ergonomics:
   - crate-level rustdoc
   - README quick start
   - catalog and exercises
   - named arguments
   - aliases
   - programmatic `documentation()`
   - custom scalar coercion
13. Publish validation:
   - `cargo package --allow-dirty --no-default-features`
   - `cargo publish --dry-run --allow-dirty --no-default-features`

## Next

Planning-only work remains:

1. cut the first crates.io release from the current validated surface
2. decide whether broader planner hooks are worthwhile beyond `simplify`
3. decide whether custom expression planning is justified for future SQL forms
4. decide whether any richer table-function catalog is actually better than struct-valued scalar
   results plus `unpack_struct`
5. decide whether any dedicated `WindowUDF` surfaces are needed beyond retractable aggregates

## Needed

When the next implementation round starts:

1. update this file in the same change set as any non-trivial surface-area change
2. keep `CATALOG.md`, `docs/CAPABILITY_MATRIX.md`, and `docs/STATUS.md` aligned with the real
   implemented catalog
3. keep the aggregate design constraints intact:
   - typed state
   - sufficient statistics when exact
   - Arrow output materialization only at `evaluate`
4. keep `docs/PUBLISH_CHECKLIST.md` and release automation text aligned with the real dependency
   source and publish posture
