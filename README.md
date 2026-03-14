# 🪐 ndatafusion

`ndatafusion` is a DataFusion extension providing `nabled` powered linear algebra and ML UDFs.

# Install

```toml
[dependencies]
ndatafusion = { version = "0.0.1", features = ["openblas-system"] }
```

`nabled/arrow` is part of the base `ndatafusion` contract and is enabled unconditionally.

Feature forwarding follows `nabled` directly:

* `blas`
* `lapack-provider`
* `openblas-system`
* `openblas-static`
* `netlib-system`
* `netlib-static`
* `magma-system`
* `accelerator-rayon`
* `accelerator-wgpu`

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

The `make_*` constructors convert ordinary nested `List` values into the canonical numerical
contracts used by the linalg/ml UDFs. They are not required when input columns already use the
expected Arrow contract, such as `FixedSizeList<Float32|Float64>(D)` for dense vectors or the
extension-backed matrix, tensor, and sparse batch layouts emitted by `ndarrow`:

1. `make_vector`
2. `make_matrix`
3. `make_tensor`
4. `make_variable_tensor`
5. `make_csr_matrix_batch`

## Status

For the user-facing SQL catalog, see [CATALOG.md](https://github.com/GeorgeLeePatterson/ndatafusion/blob/master/CATALOG.md).
For quick copy-paste queries, see [EXERCISES.md](https://github.com/GeorgeLeePatterson/ndatafusion/blob/master/EXERCISES.md).
For runnable examples, see `cargo run --example hello_sql`, `cargo run --example direct_arrow_vectors`, and `cargo run --example pca_pipeline`.

`ndatafusion` registers a direct batch-native catalog across 78 scalar UDFs:

1. canonical SQL constructors for dense vector, dense matrix, fixed-shape tensor,
   variable-shape tensor, and CSR sparse-matrix batches
2. dense vector row ops
3. dense matrix matvec, batched matmul, lower/upper triangular solves, and
   LU/Cholesky/QR least-squares solves
4. struct-valued LU, Cholesky, QR, reduced QR, pivoted QR, SVD, truncated SVD,
   tolerance-thresholded SVD, symmetric/generalized eigen, Schur, polar, and PCA workflows
5. matrix inverse, determinant, log-determinant, QR condition number / reconstruction,
   SVD null-space / pseudo-inverse / condition number / rank / reconstruction,
   non-symmetric matrix balancing, Gram-Schmidt variants, zero-config matrix functions
   (`matrix_exp_eigen`, `matrix_log_eigen`, `matrix_log_svd`, `matrix_sign`), and configurable
   matrix functions (`matrix_exp`, `matrix_log_taylor`, `matrix_power`)
6. sparse batch matvec, sparse-dense matmat, sparse transpose, and sparse-sparse matmat
7. fixed-shape tensor last-axis reductions, normalization, batched products, row-wise
   permutation/contraction, and variable-shape tensor last-axis workflows
8. matrix column means, PCA fit plus PCA transform/inverse-transform, dense iterative solvers,
   and linear regression
9. sparse direct solve via `sparse_lu_solve`

The current surface supports `Float32` and `Float64` across the implemented catalog. The crate also has end-to-end SQL integration coverage for:

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

## License

Apache License, Version 2.0
