# Execution Tracker

Last updated: 2026-03-14

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

1. The repository is still a single crate, but it now has a real registration/catalog surface and
   domain UDF modules.
2. DataFusion is pinned to an Arrow-58-compatible git revision on `main`.
3. `.justfile` quality gates are present and passing on the current tree.
4. The repository now has authoritative planning docs and AGENTS bootstrap instructions.
5. `ndatafusion` now depends on published `nabled 0.0.7` and `ndarrow 0.0.3`.
6. A substantial real-valued numerical DataFusion catalog now exists across vector, matrix,
   decomposition, sparse, tensor, and ML/stat slices.
7. The catalog now includes SQL-native constructors for the canonical real-valued vector, matrix,
   tensor, variable-tensor, and CSR sparse-batch contracts.
8. README examples and end-to-end SQL integration coverage now exist for the constructor-backed
   catalog.
9. Publish hardening is now in place for the current git-consumed release posture, and the next
   local work is residual admitted parity over the expanded catalog.

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
11. `D-011`: `N-010` has now advanced substantially on the admitted real-valued surface:
    - matrix inverse, determinant, and log-determinant
    - Cholesky decomposition, solve, and inverse
    - QR and SVD struct-return contracts
    - matrix centering, covariance, correlation, and PCA
    - sparse batch dense matmat, transpose, and sparse matmat
    - fixed-shape and variable-shape tensor norm/normalize/dot/matmul expansions
    - the catalog now exposes 35 registered scalar UDFs with coverage still above 90%
12. `D-012`: `N-008` is now complete on the current real-valued contract:
    - `make_vector`
    - `make_matrix`
    - `make_tensor`
    - `make_variable_tensor`
    - `make_csr_matrix_batch`
    - constructor success/failure coverage now exists for scalar-literal-style and array-column
      inputs
13. `D-013`: `N-009` is now underway with real publish-facing usage coverage:
    - README quick-start examples now use the constructor-backed SQL surface
    - integration tests now exercise `SessionContext` registration plus SQL execution for
      literal-backed constructor pipelines
    - integration tests now exercise list-column-backed vector/matrix flows
    - integration tests now exercise sparse-plus-variable-tensor and fixed-shape tensor pipelines
14. `D-014`: `N-009` publish hardening is now complete for the current local release posture:
    - README install guidance now reflects git consumption instead of implying crates.io
      publication
    - crate-level rustdoc now documents the constructor-backed real-valued contract and quick-start
      usage
    - docs.rs metadata is configured for `--no-default-features`
    - `docs/PUBLISH_CHECKLIST.md` now records the release gate, release-note minimums, and the
      current DataFusion git-dependency publication blocker
15. `D-015`: `N-010` has advanced again on dense matrix/decomposition parity:
    - row-wise `matrix_matvec` now exists over canonical matrix/vector batch carriers
    - QR least-squares and QR condition-number helpers now exist on the direct real-valued matrix
      batch surface
    - SVD pseudo-inverse, condition-number, and rank helpers now exist on the direct real-valued
      matrix batch surface
    - SQL integration coverage now exercises the new matrix helper slice through constructor-backed
      queries
16. `D-016`: `N-010` has advanced again on matrix parity:
    - lower/upper triangular vector solves now exist over canonical matrix/vector batch carriers
    - lower/upper triangular matrix solves now exist over canonical matrix/matrix batch carriers
    - zero-config matrix functions (`matrix_exp_eigen`, `matrix_log_eigen`, `matrix_log_svd`,
      `matrix_sign`) now exist over square matrix batches
    - the current catalog now exposes 54 registered scalar UDFs
    - unit and SQL integration coverage now exercise the new triangular and matrix-function slice
17. `D-017`: `N-010` has advanced again on configurable matrix-function parity:
    - `matrix_exp` now exists over square matrix batches with explicit `max_terms` and
      `tolerance` scalar arguments
    - `matrix_log_taylor` now exists over square matrix batches with explicit `max_terms` and
      `tolerance` scalar arguments
    - `matrix_power` now exists over square matrix batches with an explicit scalar exponent
    - shared scalar parsing helpers now cover integer and real scalar contracts directly
    - the current catalog now exposes 57 registered scalar UDFs
    - unit and SQL integration coverage now exercise the new parameterized matrix-function slice
18. `D-018`: `N-010` has advanced again on decomposition variant parity:
    - `matrix_qr_reduced` now exists over canonical matrix batches with the reduced QR struct
      result contract
    - `matrix_qr_pivoted` now exists over canonical matrix batches with an explicit permutation
      matrix in the struct result
    - `matrix_svd_truncated` now exists with an explicit scalar `k` argument
    - `matrix_svd_with_tolerance` now exists with an explicit scalar tolerance argument
    - `matrix_svd_null_space` now exists with a variable-shape tensor batch result contract
    - the current catalog now exposes 62 registered scalar UDFs
    - unit and SQL integration coverage now exercise the new decomposition-variant slice
19. `D-019`: `N-010` has advanced again on spectral and orthogonalization parity:
    - `matrix_eigen_symmetric` now exists over canonical square matrix batches with a struct
      result contract for eigenvalues plus eigenvectors
    - `matrix_eigen_generalized` now exists over canonical square matrix-pair batches with the
      same struct result contract
    - `matrix_schur` and `matrix_polar` now exist over canonical square matrix batches with
      paired matrix struct results
    - `matrix_gram_schmidt` and `matrix_gram_schmidt_classic` now exist over canonical matrix
      batches as direct orthogonalization helpers
    - the current catalog now exposes 68 registered scalar UDFs
    - direct unit coverage, float32 branch coverage, contract-edge validation, and constructor-
      backed SQL integration coverage now exercise the new spectral slice
20. `D-020`: `N-010` has advanced again on iterative-solver parity:
    - `matrix_conjugate_gradient` now exists over canonical square matrix/vector batch carriers
      with explicit scalar `tolerance` and `max_iterations` arguments
    - `matrix_gmres` now exists over canonical square matrix/vector batch carriers with the same
      explicit scalar configuration
    - the current catalog now exposes 70 registered scalar UDFs
    - direct unit coverage, float32 branch coverage, contract-edge validation, and constructor-
      backed SQL integration coverage now exercise the iterative solver slice
21. `D-021`: `N-010` has advanced again on fixed-shape tensor parity:
    - `tensor_permute_axes` now exists over canonical fixed-shape tensor batches with variadic
      integer-axis SQL arguments and row-preserving batch semantics
    - `tensor_contract_axes` now exists over canonical fixed-shape tensor batches with variadic
      left/right integer-axis pairs and row-preserving batch semantics
    - the current catalog now exposes 72 registered scalar UDFs
    - direct unit coverage, float32 branch coverage, contract-edge validation, and constructor-
      backed SQL integration coverage now exercise the new tensor-axis slice

## Next

1. `N-010` (`Layer 3`, `ndatafusion`): Finish the residual admitted type/domain parity work that
   still fits the current real-valued, SQL-natural v1 contract while preserving direct batch
   delegation on hot paths.

## Needed

1. Replace the temporary DataFusion git dependency once a published Arrow-58-compatible release
   exists on crates.io.
2. A post-v1 decision on whether UDAFs, window functions, planners, or rewrites are worth
   introducing for dataset-level or ergonomic workflows.
3. A post-v1 performance pass to reduce fallback lift/assembly overhead where direct batch
   delegation is still impossible.

## Round Scope Lock

1. This round starts local `ndatafusion` implementation after the upstream prerequisite releases.
2. The next execution round should continue `N-010`: residual admitted parity on the current
   real-valued catalog.
3. Preserve the concept-first contract while the current catalog expands.
