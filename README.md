# ndatafusion

`ndatafusion` is a DataFusion-facing facade over `nabled`.

Current baseline:

1. DataFusion is pinned to an Arrow-58-compatible git revision.
2. Published `nabled 0.0.7` and `ndarrow 0.0.3` are now wired in.
3. `ndatafusion` mirrors `nabled`'s public feature flags one-for-one.
4. The initial extension-crate surface exists via `register_all`, `functions`, and `udfs`.
5. The first numerical UDF catalog now exists.

## Use Today

```toml
[dependencies]
ndatafusion = { git = "https://github.com/georgeleepatterson/ndatafusion", features = ["openblas-system"] }
```

`ndatafusion` is not published yet. The crate still depends on DataFusion from a pinned git
revision because the latest published DataFusion release does not yet match the Arrow 58 contract
required by `nabled`.

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

## Quick Start

```rust
use datafusion::prelude::SessionContext;

#[tokio::main]
async fn main() -> datafusion::common::Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batches = ctx
        .sql(
            "SELECT
                vector_dot(make_vector(left_values, 2), make_vector(right_values, 2)) AS dot,
                matrix_determinant(make_matrix(matrix_values, 2, 2)) AS det
             FROM (
                SELECT
                    [3.0, 4.0] AS left_values,
                    [4.0, 0.0] AS right_values,
                    [9.0, 0.0, 0.0, 4.0] AS matrix_values
             )",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches[0].num_rows(), 1);
    Ok(())
}
```

The `make_*` constructors are the SQL ingress boundary from ordinary nested `List` values into the
canonical numerical contracts used by the linalg/ml UDFs:

1. `make_vector`
2. `make_matrix`
3. `make_tensor`
4. `make_variable_tensor`
5. `make_csr_matrix_batch`

## Status

The current implementation is no longer scaffold-only. `ndatafusion` now registers a substantial
direct batch-native catalog across 72 scalar UDFs:

1. canonical SQL constructors for dense vector, dense matrix, fixed-shape tensor,
   variable-shape tensor, and CSR sparse-matrix batches
2. dense vector row ops
3. dense matrix matvec, batched matmul, lower/upper triangular solves, and
   LU/Cholesky/QR least-squares solves
4. struct-valued LU, Cholesky, QR, reduced QR, pivoted QR, SVD, truncated SVD,
   tolerance-thresholded SVD, symmetric/generalized eigen, Schur, polar, and PCA workflows
5. matrix inverse, determinant, log-determinant, QR condition number, SVD null-space,
   pseudo-inverse / condition number / rank, Gram-Schmidt variants, zero-config matrix functions
   (`matrix_exp_eigen`, `matrix_log_eigen`, `matrix_log_svd`, `matrix_sign`), and configurable
   matrix functions (`matrix_exp`, `matrix_log_taylor`, `matrix_power`)
6. sparse batch matvec, sparse-dense matmat, sparse transpose, and sparse-sparse matmat
7. fixed-shape tensor last-axis reductions, normalization, batched products, row-wise
   permutation/contraction, and variable-shape tensor last-axis workflows
8. matrix column means, PCA, dense iterative solvers, and linear regression

The current admitted real-valued surface supports `Float32` and `Float64` across the implemented
catalog. The crate also has end-to-end SQL integration coverage for:

1. literal-backed constructor pipelines
2. list-column-backed vector and matrix queries
3. triangular solve plus matrix-function queries
4. parameterized matrix-function queries
5. decomposition-variant queries
6. spectral / orthogonalization queries
7. iterative solver queries
8. sparse plus variable-shape tensor composition
9. fixed-shape tensor constructor plus reduction pipelines
10. fixed-shape tensor axis permutation / contraction queries

The next milestone is residual admitted parity on top of the current constructor-backed
real-valued catalog. Actual crates.io publication remains blocked until `ndatafusion` can depend
on a published Arrow-58-compatible DataFusion release instead of the current git revision.
