# Status Snapshot

Last updated: 2026-03-14

## Summary

`ndatafusion` is now past the planning-only baseline, and local implementation work has started on
top of the released upstream contracts.

1. The repository is currently a single crate with a thin but real module surface.
2. DataFusion is pinned to an Arrow-58-compatible git revision on `main`.
3. Local quality gates pass on the current scaffold.
4. Planning docs and AGENTS tracking instructions now exist.
5. Published `nabled 0.0.7` and `ndarrow 0.0.3` are now wired in.
6. `ndatafusion` mirrors `nabled`'s feature flags one-for-one.
7. A real local registration/catalog surface now exists via `register_all`, `functions`, `udfs`,
   shared metadata/error helpers, and domain modules.
8. A substantial real-valued numerical DataFusion UDF catalog is now implemented directly as the
   crate surface.
9. The current catalog now includes SQL constructors for canonical dense vector, dense matrix,
   fixed-shape tensor, variable-shape tensor, and CSR sparse-batch contracts.
10. The current catalog now has constructor/helper tests, contract-edge validation, successful-path
    domain coverage, and line coverage above 90%.
11. The active ingress model remains concept-first: each mathematical object family needs one
    canonical standalone ingress and one canonical `rows-of-X` batch carrier.
12. End-to-end SQL integration coverage now exists for literal-backed constructors, list-column
    constructor pipelines, triangular solve plus matrix-function flows,
    parameterized matrix-function flows, decomposition-variant flows,
    sparse-plus-variable-tensor composition, and fixed-shape tensor constructor/reduction flows.
13. Crate-level rustdoc, docs.rs metadata, and an explicit publish checklist now exist for the
    current constructor-backed surface.
14. crates.io publication is still intentionally blocked while `ndatafusion` depends on a git
    revision of DataFusion for Arrow 58 compatibility.
15. The dense matrix/decomposition slice now also covers row-wise `matrix_matvec`, QR
    least-squares, QR condition number, and SVD pseudo-inverse / condition-number / rank helpers,
    with SQL integration coverage for the new helper family.
16. The dense matrix slice now also covers lower/upper triangular solves, lower/upper triangular
    matrix solves, and zero-config matrix functions (`matrix_exp_eigen`, `matrix_log_eigen`,
    `matrix_log_svd`, `matrix_sign`) across the admitted real-valued surface.
17. The matrix-function slice now also covers configurable matrix exponential / logarithm / power
    helpers (`matrix_exp`, `matrix_log_taylor`, `matrix_power`) with scalar-parameter validation,
    direct unit coverage, and SQL integration coverage.
18. The decomposition slice now also covers reduced and pivoted QR plus truncated, tolerance-based,
    and null-space SVD helpers, with direct unit coverage and constructor-backed SQL integration
    coverage.
19. The decomposition slice now also covers symmetric/generalized eigen, Schur, polar, and both
    Gram-Schmidt variants, with direct unit coverage, float32 type-propagation coverage, and
    constructor-backed SQL integration coverage.
20. The ML/stat slice now also covers dense iterative solvers (`matrix_conjugate_gradient` and
    `matrix_gmres`) with explicit scalar configuration, direct unit coverage, float32
    type-propagation coverage, and constructor-backed SQL integration coverage.
21. The tensor slice now also covers row-wise fixed-shape axis permutation and contraction
    (`tensor_permute_axes` and `tensor_contract_axes`) with variadic integer-axis SQL contracts,
    direct unit coverage, float32 type-propagation coverage, and constructor-backed SQL
    integration coverage.
22. The decomposition-helper slice now also covers direct real-valued QR and SVD reconstruction
    (`matrix_qr_reconstruct` and `matrix_svd_reconstruct`) with direct unit coverage, float32
    type-propagation coverage, and constructor-backed SQL integration coverage.
23. The spectral-helper slice now also covers direct real-valued non-symmetric balancing
    (`matrix_balance_nonsymmetric`) with a struct result contract for the balanced matrix plus
    balancing diagonal, direct unit coverage, float32 type-propagation coverage, square-contract
    validation, and constructor-backed SQL integration coverage.
24. The PCA application slice now also covers direct transform and inverse-transform helpers
    (`matrix_pca_transform` and `matrix_pca_inverse_transform`) against the existing PCA struct
    contract, with direct unit coverage, float32 type-propagation coverage, and constructor-backed
    SQL integration coverage.
25. The sparse direct-solve slice now also covers `sparse_lu_solve` over canonical
    `ndarrow.csr_matrix_batch` plus rank-1 variable-shape tensor batches, with direct unit
    coverage, float32 type-propagation coverage, and constructor-backed SQL integration coverage.

## Current Repository Reality

1. Root files now cover crate metadata, linting, formatting, docs.rs posture, and constructor-aware
   integration coverage.
2. There is no `docs/` implementation history before this planning baseline.
3. A larger internal module tree now exists for registration, shared metadata/signature/error
   helpers, and domain UDFs.
4. Direct batch-native delegation now exists across the admitted vector, matrix, sparse, and
   tensor workflows whenever `nabled::arrow` exposes the needed batch carrier directly.
5. The fallback generic cell-codec layer still does not exist; residual unsupported workflows are
   currently handled case-by-case or remain unimplemented.
6. SQL-native constructor ingress now exists from ordinary `List` values into the canonical
   real-valued vector, matrix, tensor, variable-tensor, and CSR sparse-batch contracts.
7. README-level quick-start examples now match real constructor-backed SQL flows.
8. Current validation covers both successful batch-native paths and representative type, shape,
   scalar-argument, and batch-length failure contracts.

## Constraints In Force

1. `ndatafusion` is a facade over `nabled`, not a numerical fork.
2. Arrow/DataFusion value contracts must remain explicit and aligned with `ndarrow`.
3. Capability parity beats overload parity.
4. Quality gates remain strict: `just checks`, pedantic clippy compliance, and coverage greater
   than 90% before push.
5. `nabled` feature forwarding is a hard requirement for the published crate surface.
6. Cross-repo prerequisite work must preserve correctness and performance in already admitted
   lower-layer behavior.
7. Existing broader lower-layer public APIs do not need to be removed just because `ndatafusion`
   will depend on a narrower SQL-facing contract surface.

## Current Code Ownership

Today:

1. `src/lib.rs`, `src/register.rs`, `src/functions.rs`, `src/udfs.rs`
   - crate-level docs, registration, public expression helpers, and UDF catalog surface
2. `src/error.rs`, `src/metadata.rs`, `src/signatures.rs`
   - shared contract, signature, and error helpers
3. `src/udf/`
   - constructor plus domain UDF implementations across vector, matrix, decomposition, sparse,
     tensor, and ML/stat slices
4. `tests/e2e.rs`
   - end-to-end SQL coverage for registration, constructors, and representative numerical flows

Target ownership after the first real implementation rounds:

1. registration and public crate surface
2. shared metadata and any narrow fallback cell codecs still justified by missing lower-layer
   batch contracts
3. domain UDF modules grouped by numerical domain
4. optional planner or rewrite integration only if admitted later

## Next Required Milestone

Local implementation round:

1. Hold the current non-controversial real-valued catalog stable for release.
2. Move the next planning pass to controversial or post-v1 work only:
   - complex-valued result contracts,
   - callback-driven differentiation / optimization,
   - stateful sparse factorization reuse,
   - richer planner, UDAF, window, or table-function surfaces.
3. Revisit crates.io publication only once DataFusion exposes a compatible published release line.

## V1 Publish Readiness

Not ready for crates.io publication.

Current state:

1. the non-controversial, SQL-natural, real-valued catalog is now implemented and release-worthy
   for git consumption
2. the remaining implementation work is intentionally controversial or post-v1
3. the current DataFusion git dependency still blocks crates.io publication until a compatible
   published release exists
