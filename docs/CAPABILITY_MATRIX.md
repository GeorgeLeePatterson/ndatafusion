# Capability Matrix

Last updated: 2026-03-15

## Purpose

This file is the canonical scope and sufficiency map for `ndatafusion`.

## Status Legend

- `Implemented`: present in the repository and covered by the current quality gates.
- `Partial`: admitted in some form, but future expansion is still possible or likely.
- `Missing`: intentionally not yet admitted in `ndatafusion`.

## Current Inventory

| Area | Capability | Status | Notes |
|---|---|---|---|
| Build baseline | Arrow-58-compatible DataFusion dependency | Implemented | Pinned to a git revision on `main`. |
| Dependency surface | published `ndarrow` / `nabled` alignment | Implemented | Uses `ndarrow 0.0.3` and `nabled 0.0.7`. |
| Dependency surface | `nabled` feature forwarding | Implemented | `nabled/arrow` is always enabled; remaining feature flags are forwarded directly. |
| Publication channel | crates.io-ready dependency posture | Partial | Still blocked by the DataFusion published-release mismatch for Arrow 58. |
| Registration surface | scalar UDF registration | Implemented | `register_all` installs the full scalar catalog. |
| Registration surface | aggregate UDF registration | Implemented | `register_all` installs the full aggregate catalog. |
| Registration surface | table-function registration | Implemented | `register_all_session` adds `unpack_struct`. |
| Contract layer | shared metadata and signature helpers | Implemented | Covers real, complex, sparse, tensor, aggregate, and table-function contracts. |
| SQL ergonomics | constructor UDFs | Implemented | Canonical vector, matrix, fixed-shape tensor, variable-shape tensor, and CSR constructors are present. |
| SQL ergonomics | named arguments | Implemented | Present across selected constructors, controls, differentiation, optimization, sparse-factorization, PCA, and decomposition surfaces. |
| SQL ergonomics | aliases | Implemented | Canonical names stay explicit; shorter aliases exist for predictable repetitive suffixes. |
| SQL ergonomics | programmatic `documentation()` | Implemented | Present across the admitted scalar and aggregate surface. |
| Planner integration | simplify hooks | Implemented | Present for the current obvious rewrite cases. |
| Planner integration | broader planner hooks | Partial | `simplify` exists; other hooks are intentionally selective. |
| Planner integration | custom expression planning | Missing | Reserved for future SQL forms that need it. |
| Dense vectors | real-valued vector batch surface | Implemented | Row-wise dot, norm, cosine, distance, and normalization. |
| Dense vectors | complex-valued vector batch surface | Implemented | Hermitian dot, norm, cosine similarity, and normalization. |
| Dense matrices | real-valued matrix/direct-solver surface | Implemented | Matvec, matmul, triangular solves, LU/Cholesky solves, and matrix functions. |
| Dense matrices | complex-valued matrix surface | Implemented | Matvec, matmat, matrix statistics, iterative solves, spectral helpers, and matrix functions. |
| Decompositions | real-valued decomposition surface | Implemented | LU, Cholesky, QR, SVD, eigen, Schur, polar, balancing, Gram-Schmidt, and reconstruction helpers. |
| Decompositions | complex-valued decomposition surface | Implemented | Complex nonsymmetric eigen, Schur, and polar. |
| Matrix equations | Sylvester surface | Implemented | Real, mixed-precision real, complex, and mixed-precision complex variants. |
| Sparse | CSR batch surface | Implemented | Matvec, direct solve, dense matmat, transpose, sparse matmat. |
| Sparse | sparse factorization / preconditioner surface | Implemented | LU factors, LU solve reuse, Jacobi, ILUT, ILUK, and apply helpers. |
| Tensor | fixed-shape tensor surface | Implemented | Last-axis reductions, normalization, batched products, and axis transforms. |
| Tensor | variable-shape tensor surface | Implemented | Last-axis reductions, normalization, and batched dot. |
| Tensor | complex tensor surface | Implemented | Fixed-shape and variable-shape norm / normalization helpers. |
| Tensor | tensor decomposition surface | Implemented | CP, Tucker/HOSVD/HOOI, TT SVD, TT algebra, and reconstruction helpers. |
| ML/stat | real-valued stats / PCA / regression | Implemented | Means, centering, covariance, correlation, PCA, iterative solves, and regression. |
| ML/stat | complex-valued stats / PCA | Implemented | Complex matrix statistics and complex PCA fit / transform / inverse-transform. |
| Differentiation | named-function differential operators | Implemented | `jacobian`, `jacobian_central`, `gradient`, `hessian`. |
| Optimization | named-function complex optimization surface | Implemented | Backtracking line search, gradient descent, momentum descent, and Adam. |
| Aggregates | grouped aggregate UDF wave | Implemented | Covariance, correlation, PCA fit, and linear regression fit. |
| Windows | ordered window use of aggregates | Implemented | Supported through retractable aggregate state. |
| Table functions | generic struct unpacking | Implemented | `unpack_struct` exists today. |
| Table functions | richer relation-shaped catalogs | Partial | No broader UDTF wave yet. |

## Sufficiency Verdict

`ndatafusion` is sufficient for a broad git-consumed release of the current surface.

It is not yet sufficient for crates.io publication because of the external DataFusion release
constraint, not because of a local implementation gap.

## Remaining Strategic Work

Only planning-grade expansion work remains:

1. broader planner hooks where they are demonstrably valuable
2. custom expression planning for SQL forms that cannot be represented cleanly otherwise
3. optional richer table-function surfaces
4. optional dedicated `WindowUDF` surfaces beyond retractable aggregates
5. crates.io publication once the DataFusion published line is Arrow-58-compatible
