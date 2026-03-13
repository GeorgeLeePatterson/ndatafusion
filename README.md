# ndatafusion

`ndatafusion` is a DataFusion-facing facade over `nabled`.

Current baseline:

1. DataFusion is pinned to an Arrow-58-compatible git revision.
2. Published `nabled 0.0.7` and `ndarrow 0.0.3` are now wired in.
3. `ndatafusion` mirrors `nabled`'s public feature flags one-for-one.
4. The initial extension-crate surface exists via `register_all`, `functions`, and `udfs`.
5. The first numerical UDF catalog now exists.

## Install

```toml
[dependencies]
ndatafusion = { version = "0.0.1", features = ["openblas-system"] }
```

`nabled/arrow` is part of the base `ndatafusion` contract and is enabled unconditionally.

Feature forwarding follows `nabled` directly:

1. `test-utils`
2. `blas`
3. `lapack-provider`
4. `openblas-system`
5. `openblas-static`
6. `netlib-system`
7. `netlib-static`
8. `magma-system`
9. `accelerator-rayon`
10. `accelerator-wgpu`

## Status

The current implementation is no longer scaffold-only. `ndatafusion` now registers the first
substantial direct batch-native catalog across:

1. canonical SQL constructors for dense vector, dense matrix, fixed-shape tensor,
   variable-shape tensor, and CSR sparse-matrix batches
2. dense vector row ops
3. dense matrix matmul plus LU and Cholesky solves
4. struct-valued LU, Cholesky, QR, SVD, and PCA workflows
5. matrix inverse, determinant, log-determinant, centering, covariance, and correlation
6. sparse batch matvec, sparse-dense matmat, sparse transpose, and sparse-sparse matmat
7. fixed-shape and variable-shape tensor last-axis reductions, normalization, and batched products
8. matrix column means and linear regression

The next milestone is publish hardening, examples, and residual admitted parity on top of the
current `f64`-first constructor-backed catalog.
