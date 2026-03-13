# Capability Matrix

Last updated: 2026-03-13

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
| Extension crate shape | `register_all`, `functions`, and `udfs` public surface | Implemented | `src/` + tests | Public registration/catalog surface now exists and registers a non-empty numerical catalog directly. |
| Data contract layer | shared DataType/Field builders for vector, matrix, tensor, sparse, and complex values | Implemented | `src/metadata.rs` + tests | Shared field builders and validation helpers now own the first SQL-facing numerical contracts. |
| Direct batch delegation | whole-array delegation into stabilized `nabled::arrow` contracts | Partial | unit tests | Present for the first vector, matrix, sparse, and tensor slices; residual unsupported workflows still fall back to direct ndarray view iteration or remain unimplemented. |
| Cell codec layer | row extraction and result assembly for lifted `nabled` contracts | Missing | No | Fallback-only layer for workflows that still lack a direct batch-native lower-layer path. |
| Dense vector surface | row-wise vector kernels (`dot`, norms, cosine, pairwise/batched where natural) | Partial | unit tests | `l2_norm`, `dot`, `cosine_similarity`, `cosine_distance`, and `normalize` now exist for `rows-of-vectors` over `FixedSizeList<Float64>(D)`. |
| Dense matrix surface | row-wise matrix kernels and helpers | Partial | unit tests | Batched matrix-matrix product plus LU/Cholesky solves now exist over fixed-shape tensor matrix batches. |
| Decomposition surface | struct-returning factorization and solver contracts | Partial | unit tests | LU, Cholesky, QR, and SVD struct-valued contracts now exist; additional config-heavy decomposition variants still remain. |
| Sparse surface | CSR-aware DataFusion contracts and wrappers | Partial | unit tests | Sparse batch matvec, dense matmat, transpose, and sparse matmat now exist over `ndarrow.csr_matrix_batch`. |
| Tensor surface | fixed-shape tensor contracts and wrappers | Partial | unit tests | Fixed-shape and variable-shape last-axis reductions, normalization, and batched products now exist on the admitted `f64` surface. |
| ML/stat surface | DataFusion wrappers for iterative, jacobian, optimization, PCA, regression, stats | Partial | unit tests | Column means, centering, covariance, correlation, PCA, and linear regression now exist; callback/config-heavy workflows still remain. |
| SQL usability | constructors and normalizers from SQL-friendly nested values into canonical contracts | Missing | No | Important for real SQL use, not optional polish. |
| Planner layer | function rewrites or expression planners | Missing | No | Optional for v1 unless required by constructors or ergonomics. |
| Hardening | examples, integration coverage, docs, and publish checklist readiness | Partial | `just checks` + coverage | Contract-edge unit coverage and the current coverage gate now pass, but publish-ready examples and release hardening are still missing. |

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
| Dense vector and matrix kernels | Partial | Core row-preserving `f64` slices are in place; residual type coverage and a few higher-level matrix workflows remain. |
| Decomposition and solver workflows | Partial | LU, Cholesky, QR, and SVD contracts now exist; residual config-heavy and alternate decomposition variants remain. |
| Sparse and tensor workflows | Partial | Core sparse batch and tensor last-axis workflows now exist on both fixed-shape and variable-shape carriers. |
| ML/stat workflows | Partial | Stats, PCA, and linear regression now exist on the admitted `f64` surface; iterative and callback-driven workflows remain open. |
| Constructor and normalization surface | Missing | Allow SQL callers to construct canonical numerical values without bespoke Rust setup. |
| Quality and publish hardening | Partial | Feature-matrix checks and line coverage > 90% now pass; examples, publish docs, and release metadata are still missing. |

### P2: Post-V1 Expansion

| Capability Group | Current Status | Gap |
|---|---|---|
| UDAFs and window functions | Missing | Defer until scalar-UDF-first surface is stable and justified. |
| Table functions and planner rewrites | Missing | Add only when they provide clear ergonomic or semantic value. |
| Residual fallback optimization | Missing | After v1, reduce the cost of any remaining lift/assembly fallback paths that could not be eliminated. |
| Richer SQL sugar and alternate result contracts | Missing | Explicitly admit only after core contracts stabilize. |

## Sufficiency Verdict

`ndatafusion` is not yet sufficient for a v1 publish.

What exists now is the governance baseline, released upstream dependency alignment, and a minimal
extension-crate scaffold.

Primary remaining work:

1. Add constructors and normalizers for SQL-facing canonical numerical values.
2. Finish the residual admitted catalog while preferring direct batch delegation over bespoke lifting.
3. Harden for publish with examples, broader integration coverage, and release-ready docs.

## Execution Order Driven By This Matrix

1. Shared metadata and direct delegation plus residual lift/codec support.
2. Dense vector/matrix registration slice.
3. Decomposition result contracts and solver helpers.
4. Sparse, tensor, and ML/stat expansion.
5. Publish hardening.
