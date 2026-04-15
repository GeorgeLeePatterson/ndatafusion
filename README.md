# 🪐 ndatafusion

`ndatafusion` is a DataFusion extension providing `nabled` powered linear algebra and ML UDFs.

## Installation

```toml
[dependencies]
ndatafusion = { version = "0.1.0", features = ["0.1.0"] }
```

This crate currently targets `DataFusion 53` on the Arrow 58 line.

`nabled/arrow` is part of the base `ndatafusion` contract and is enabled unconditionally.

`openblas-system` above is only an example provider selection. Choose the forwarded provider and
accelerator features that match your environment and deployment target.

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

Selected constructor and control-parameter UDFs also support named arguments in SQL. For
numerical UDFs, prefer positional data arguments first and named trailing control arguments after.
For example:

```sql
SELECT matrix_exp(
  make_matrix(values => matrix_values, rows => 2, cols => 2),
  max_terms => 32,
  tolerance => 1e-6
)
```

## Status

For the user-facing SQL catalog, see [CATALOG.md](https://github.com/GeorgeLeePatterson/ndatafusion/blob/master/CATALOG.md).
For quick copy-paste queries, see [EXERCISES.md](https://github.com/GeorgeLeePatterson/ndatafusion/blob/master/EXERCISES.md).
For runnable examples, see `cargo run --example hello_sql`, `cargo run --example direct_arrow_vectors`, and `cargo run --example pca_pipeline`.

`ndatafusion` currently registers:

1. `147` scalar UDFs through `register_all`
2. `4` aggregate UDFs through `register_all`
3. `1` table function through `register_all_session`

The current surface includes:

1. canonical SQL constructors for dense vector, dense matrix, fixed-shape tensor,
   variable-shape tensor, and CSR sparse-matrix batches
2. dense vector row ops plus complex-vector Hermitian dot / norm / cosine-similarity /
   normalization helpers over canonical `ndarrow.complex64` vectors, plus named-function
   differentiation helpers (`jacobian`, `jacobian_central`, `gradient`, `hessian`)
3. complex dense matrix matvec / matmat, complex matrix statistics, and complex dense iterative
   solvers over canonical `arrow.fixed_shape_tensor<ndarrow.complex64>` matrix batches
4. dense matrix matvec, batched matmul, lower/upper triangular solves, and
   LU/Cholesky/QR least-squares solves
5. struct-valued LU, Cholesky, QR, reduced QR, pivoted QR, SVD, truncated SVD,
   tolerance-thresholded SVD, symmetric/generalized eigen, real-input complex nonsymmetric
   eigen plus bi-eigen, complex nonsymmetric eigen, Schur, polar, their current complex
   Schur/polar counterparts, and PCA workflows
6. matrix inverse, determinant, log-determinant, QR condition number / reconstruction,
   SVD null-space / pseudo-inverse / condition number / rank / reconstruction,
   non-symmetric matrix balancing, Gram-Schmidt variants, zero-config matrix functions
   (`matrix_exp_eigen`, `matrix_log_eigen`, `matrix_log_svd`, `matrix_sign`), and configurable
   matrix functions (`matrix_exp`, `matrix_log_taylor`, `matrix_power`) plus the current complex
   matrix-function slice (`matrix_exp_complex`, `matrix_exp_eigen_complex`,
   `matrix_log_eigen_complex`, `matrix_log_svd_complex`, `matrix_power_complex`,
   `matrix_sign_complex`)
7. sparse batch matvec, sparse-dense matmat, sparse transpose, and sparse-sparse matmat
8. fixed-shape tensor last-axis reductions, normalization, batched products, row-wise
   permutation/contraction, variable-shape tensor last-axis workflows, and the first complex
   fixed-shape / variable-shape tensor norm / normalization surface
9. matrix column means, real and complex PCA fit plus PCA transform/inverse-transform, dense
   iterative solvers, linear regression, and grouped aggregate fits for vector
   covariance/correlation/PCA and linear regression
10. sparse direct solve via `sparse_lu_solve`
11. Sylvester matrix-equation solvers for real and complex batches, including mixed-precision
    result contracts that report refinement iterations when `magma-system` is enabled
12. complex optimization helpers, sparse factorization/preconditioner UDFs, and tensor
    decomposition / tensor-train / Tucker UDFs over canonical batch contracts
13. retractable grouped aggregates that can also be used in ordered window frames, and the
    generic `unpack_struct` table function for turning struct-valued results into one-row tables

The current real-valued surface supports `Float32` and `Float64` across the implemented catalog.
The current complex-valued surface covers canonical `ndarrow.complex64` dense vector, dense
matrix, complex PCA, fixed-shape tensor, and variable-shape tensor inputs. The
crate also has end-to-end SQL integration coverage for:

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
11. direct canonical complex vector, matrix, and tensor queries without `make_*`
12. direct sparse factorization, tensor decomposition, differentiation, optimization, grouped
    aggregate, and ordered window queries

If you want the table-function surface in SQL, register with `ndatafusion::register_all_session`
instead of `register_all`. The current table-function catalog exposes:

1. `unpack_struct`, which expands a scalar struct-valued expression into a one-row relation

## License

Apache License, Version 2.0
