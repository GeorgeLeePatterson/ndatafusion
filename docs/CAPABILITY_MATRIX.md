# Capability Matrix

Last updated: 2026-03-15

## Purpose

This document is the canonical scope and gap map for `ndatafusion`.

It serves two goals:

1. Track what exists today.
2. Define what must exist for `ndatafusion` to be complete and ready for a v1 publish.

Operational sequencing (`Done / Next / Needed`) lives in `docs/EXECUTION_TRACKER.md`.

## Status Legend

- `Implemented`: present in repository and validated by local quality gates.
- `Partial`: direction is locked and some scaffolding exists, but meaningful implementation is
  still missing.
- `Missing`: not currently provided by `ndatafusion`.

## Current Capability Inventory

| Area | Capability | Status | Verification | Notes |
|---|---|---|---|---|
| Governance | docs bootstrap, tracker set, and AGENTS alignment | Implemented | docs + AGENTS | Repository now has authoritative planning docs and resume order. |
| Build baseline | Arrow-58-compatible DataFusion dependency | Implemented | `just checks` | DataFusion is pinned to a git revision on `main` that resolves Arrow 58. |
| Build baseline | mandatory checkpoint workflow via `.justfile` | Implemented | `just checks` | Nightly fmt, nightly/stable clippy, and unit/integration tests now run over the supported forwarded-feature matrix instead of brittle `--all-features` expansion. |
| Upstream prerequisite | checkpoint 1: `ndarrow` release-ready concept-first bridge contract | Implemented | published `0.0.3` | Stable and published for local `ndatafusion` adoption. |
| Upstream prerequisite | checkpoint 2: `nabled` release-ready concept-first Arrow façade | Implemented | published `0.0.7` | Stable and published for local `ndatafusion` adoption. |
| Dependency surface | `nabled` dependency and feature forwarding | Implemented | `Cargo.toml` + `just checks` | `ndatafusion` now depends on published `nabled 0.0.7`, enables `nabled/arrow` unconditionally, and mirrors the remaining feature flags one-for-one. |
| Publication channel | crates.io-ready dependency posture | Partial | docs + `Cargo.toml` | `ndatafusion` still depends on DataFusion from git because the published release line does not yet satisfy the Arrow 58 contract; crates.io publication remains blocked until that changes. |
| Extension crate shape | `register_all`, `functions`, `udfs`, and `udafs` public surface | Implemented | `src/` + tests | Public registration/catalog surface now exists and registers non-empty scalar and aggregate numerical catalogs directly. |
| Data contract layer | shared DataType/Field builders for vector, matrix, tensor, sparse, and complex values | Implemented | `src/metadata.rs` + tests | Shared field builders and validation helpers now own the first SQL-facing numerical contracts. |
| Direct batch delegation | whole-array delegation into stabilized `nabled::arrow` contracts | Partial | unit tests | Present for the first vector, matrix, sparse, and tensor slices; residual unsupported workflows still fall back to direct ndarray view iteration or remain unimplemented. |
| Cell codec layer | row extraction and result assembly for lifted `nabled` contracts | Missing | No | Fallback-only layer for workflows that still lack a direct batch-native lower-layer path. |
| Dense vector surface | row-wise vector kernels (`dot`, norms, cosine, pairwise/batched where natural) | Partial | unit tests + SQL e2e | Real-valued `l2_norm`, `dot`, `cosine_similarity`, `cosine_distance`, and `normalize` now exist for `rows-of-vectors` over `FixedSizeList<Float32|Float64>(D)`, and the current complex-vector subset now covers Hermitian dot, norm, cosine similarity, and normalization over canonical `ndarrow.complex64` vectors. |
| Dense matrix surface | row-wise matrix kernels and helpers | Partial | unit tests + SQL e2e | Row-wise `matrix_matvec`, batched matrix-matrix product, lower/upper triangular solves, lower/upper triangular matrix solves, zero-config matrix functions, and configurable matrix exponential / logarithm / power helpers now exist over square fixed-shape tensor matrix batches plus fixed-size-list vector batches. The current complex matrix slice now covers `matrix_matvec_complex`, `matrix_matmat_complex`, complex column means / centering / covariance / correlation, complex dense iterative solves, and the current complex matrix-function family over canonical `arrow.fixed_shape_tensor<ndarrow.complex64>` batches. |
| Decomposition surface | struct-returning factorization and solver contracts | Partial | unit tests + SQL e2e | LU, Cholesky, QR, reduced QR, pivoted QR, SVD, truncated SVD, tolerance-thresholded SVD, SVD null-space, symmetric/generalized eigen, non-symmetric balancing, Schur, and polar helpers now exist, along with direct QR condition-number / reconstruction, SVD pseudo-inverse / rank / condition-number / reconstruction, and Gram-Schmidt helpers. The current complex decomposition slice now covers complex Schur and polar over canonical complex square matrix batches; residual nonsymmetric-complex and other config-heavy workflows still remain. |
| Sparse surface | CSR-aware DataFusion contracts and wrappers | Partial | unit tests | Sparse batch matvec, direct solve, dense matmat, transpose, and sparse matmat now exist over `ndarrow.csr_matrix_batch`. |
| Tensor surface | fixed-shape tensor contracts and wrappers | Partial | unit tests + SQL e2e | Fixed-shape last-axis reductions, normalization, batched products, and row-wise axis permutation / contraction now exist alongside the admitted variable-shape last-axis workflows on the real-valued surface. The first complex tensor slice now covers fixed-shape and variable-shape last-axis norm / normalization over canonical `ndarrow.complex64` tensor batches. |
| ML/stat surface | DataFusion wrappers for iterative, jacobian, optimization, PCA, regression, stats | Partial | unit tests + SQL e2e | Column means, centering, covariance, correlation, real and complex PCA fit / transform / inverse-transform, dense iterative solvers, linear regression, and the first grouped aggregate wave (`vector_covariance_agg`, `vector_correlation_agg`, `vector_pca_fit`, `linear_regression_fit`) now exist; the grouped aggregate implementation now uses typed sufficient-statistics state instead of raw-row accumulation, the current complex slice now covers complex matrix statistics, complex PCA, and complex dense iterative solvers, and callback/config-heavy workflows still remain. |
| SQL usability | constructors and normalizers from SQL-friendly nested values into canonical contracts | Implemented | unit tests | `make_vector`, `make_matrix`, `make_tensor`, `make_variable_tensor`, and `make_csr_matrix_batch` now cover the admitted real-valued canonical contracts from SQL-style `List` values plus scalar dimensions. |
| Planner layer | function rewrites or expression planners | Missing | No | Optional for v1 unless required by constructors or ergonomics. |
| Hardening | examples, integration coverage, docs, and publish checklist readiness | Implemented | `just checks` + `cargo doc --no-default-features --no-deps` | Contract-edge unit coverage, README quick-start examples, crate-level docs, docs.rs metadata, and an explicit publish checklist now exist for the current constructor-backed catalog. |

## Target Scope Matrix

### P0: Required Foundation

| Capability Group | Current Status | Gap |
|---|---|---|
| Governance and tracker discipline | Implemented | Keep docs authoritative as implementation lands. |
| Upstream Arrow bridge hardening | Implemented | Published `ndarrow 0.0.3` now satisfies the checkpoint-1 contract. |
| Upstream Arrow façade hardening | Implemented | Published `nabled 0.0.7` now satisfies the checkpoint-2 contract. |
| Dependency alignment and feature forwarding | Implemented | `nabled` and `ndarrow` are wired in and `ndatafusion` mirrors `nabled` features one-for-one. |
| Shared DataFusion contract layer | Implemented | Keep extending the shared builders as new admitted result contracts land. |
| Direct batch delegation path | Partial | Present for the initial admitted hot paths; still expand direct delegation before adding generic lifting. |
| Lifted cell codec layer | Missing | Implement only the residual zero-copy row extraction and result assembly that still lacks a direct batch-native lower-layer path. |
| Extension crate registration surface | Implemented | Grow the catalog without regressing the public crate shape. |

### P1: Required For V1 Publish

| Capability Group | Current Status | Gap |
|---|---|---|
| Dense vector and matrix kernels | Partial | The non-controversial SQL-natural real-valued slice is now complete, including row-wise matvec, triangular solves, zero-config matrix functions, and configurable matrix exponential / logarithm / power helpers; residual work is now complex-valued or otherwise post-v1. |
| Decomposition and solver workflows | Partial | The non-controversial real-valued slice is now complete across LU, Cholesky, QR, reduced QR, pivoted QR, SVD, symmetric/generalized eigen, non-symmetric balancing, Schur, polar, QR least-squares / condition-number / reconstruction, SVD truncated/tolerance/null-space/pseudo-inverse/rank/condition-number/reconstruction, and direct Gram-Schmidt variants; residual work is now complex-valued, nonsymmetric-complex, or otherwise post-v1. |
| Sparse and tensor workflows | Partial | The non-controversial real-valued slice is now complete across sparse matvec / direct solve / dense matmat / transpose / sparse matmat plus fixed-shape tensor last-axis / axis-structure workflows and variable-shape tensor last-axis workflows; residual work is now richer stateful sparse reuse or other post-v1 expansions. |
| ML/stat workflows | Partial | The non-controversial real-valued slice is now complete across stats, PCA fit / transform / inverse-transform, dense iterative solvers, and linear regression; callback-driven workflows remain explicitly post-v1. |
| Constructor and normalization surface | Implemented | The admitted real-valued constructor set now exists via the `make_*` UDF family over SQL `List` values and explicit shape scalars. |
| Quality and publish hardening | Implemented | Feature-matrix checks, line coverage > 90%, crate docs, docs.rs metadata, README examples, and the publish checklist now exist. |

### P2: Post-V1 Expansion

| Capability Group | Current Status | Gap |
|---|---|---|
| UDAFs and window functions | Partial | The first grouped aggregate wave is now implemented on typed sufficient-statistics state instead of raw-row accumulation or opaque binary state; ordered windowed workflows remain deferred. |
| Table functions and planner rewrites | Missing | Add only when they provide clear ergonomic or semantic value. |
| Residual fallback optimization | Missing | After v1, reduce the cost of any remaining lift/assembly fallback paths that could not be eliminated. |
| Richer SQL sugar and alternate result contracts | Missing | Explicitly admit only after core contracts stabilize. |

## Sufficiency Verdict

`ndatafusion` is sufficient for a git-consumed v1 release of the current non-controversial,
SQL-natural, real-valued surface, but it is not yet sufficient for a crates.io v1 publish.

What exists now is the governance baseline, released upstream dependency alignment, a complete
non-controversial constructor-backed real-valued catalog, and publish-hardening docs for the
current git-consumed release posture.

Primary remaining work:

1. Replace the temporary DataFusion git dependency with a published compatible release once the
   upstream crates.io line satisfies the Arrow 58 contract.
2. Decide which controversial or post-v1 capabilities are actually worth admitting after the
   current release checkpoint.

## Execution Order Driven By This Matrix

1. Shared metadata and direct delegation plus residual lift/codec support.
2. Dense vector/matrix registration slice.
3. Decomposition result contracts and solver helpers.
4. Sparse, tensor, and ML/stat expansion.
5. Post-v1 planning for controversial surfaces and any narrowly justified fallback-only support.
