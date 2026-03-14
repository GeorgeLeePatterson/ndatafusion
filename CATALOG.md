# UDF Catalog

Last updated: 2026-03-14

This document is the current inventory of the `ndatafusion` SQL catalog.

Constructor UDFs are only needed when SQL starts from ordinary nested `List` values. If a table
already stores the canonical Arrow contract expected by a UDF, such as
`FixedSizeList<Float32|Float64>(D)` for dense vectors or the extension-backed matrix, tensor, and
sparse layouts produced by `ndarrow`, the numerical UDF can be called directly without `make_*`.

## Status Legend

- `Implemented`: available today through `register_all`.
- `Partial`: lower-layer support or direction exists, but `ndatafusion` does not yet expose a full,
  settled SQL-facing UDF surface.
- `Missing`: not currently exposed by `ndatafusion`.

## Constructors

| UDF | Status | Notes |
|---|---|---|
| `make_vector` | Implemented | Builds canonical dense vector batches from SQL `List` values plus a scalar dimension. |
| `make_matrix` | Implemented | Builds canonical dense matrix batches from flat row-major or nested SQL `List` values plus scalar `rows` and `cols`. |
| `make_tensor` | Implemented | Builds canonical fixed-shape tensor batches from SQL `List` values plus scalar dimensions. |
| `make_variable_tensor` | Implemented | Builds canonical variable-shape tensor batches from SQL `List` data plus explicit per-row shapes and scalar rank. |
| `make_csr_matrix_batch` | Implemented | Builds canonical `ndarrow.csr_matrix_batch` values from CSR component lists. |

## Dense Vectors

| UDF | Status | Notes |
|---|---|---|
| `vector_l2_norm` | Implemented | Row-wise norm over `rows-of-vectors`. |
| `vector_dot` | Implemented | Row-wise dot product over paired vector batches. |
| `vector_cosine_similarity` | Implemented | Row-wise cosine similarity. |
| `vector_cosine_distance` | Implemented | Row-wise cosine distance. |
| `vector_normalize` | Implemented | Row-wise normalization. |

## Dense Matrix Ops And Direct Solvers

| UDF | Status | Notes |
|---|---|---|
| `matrix_matvec` | Implemented | Row-wise matrix-vector product. |
| `matrix_matmul` | Implemented | Row-wise matrix-matrix product. |
| `matrix_solve_lower` | Implemented | Row-wise lower-triangular solve with vector RHS. |
| `matrix_solve_upper` | Implemented | Row-wise upper-triangular solve with vector RHS. |
| `matrix_solve_lower_matrix` | Implemented | Row-wise lower-triangular solve with matrix RHS. |
| `matrix_solve_upper_matrix` | Implemented | Row-wise upper-triangular solve with matrix RHS. |
| `matrix_lu_solve` | Implemented | Row-wise dense LU solve with vector RHS. |
| `matrix_cholesky_solve` | Implemented | Row-wise Cholesky solve with vector RHS. |
| `matrix_inverse` | Implemented | Row-wise dense inverse. |
| `matrix_determinant` | Implemented | Row-wise determinant. |
| `matrix_log_determinant` | Implemented | Row-wise sign plus log-abs-determinant struct result. |
| `matrix_exp` | Implemented | Configurable matrix exponential with explicit `max_terms` and `tolerance`. |
| `matrix_exp_eigen` | Implemented | Zero-config matrix exponential path. |
| `matrix_log_taylor` | Implemented | Configurable matrix logarithm with explicit `max_terms` and `tolerance`. |
| `matrix_log_eigen` | Implemented | Zero-config eigen-based matrix logarithm path. |
| `matrix_log_svd` | Implemented | Zero-config SVD-based matrix logarithm path. |
| `matrix_power` | Implemented | Matrix power with explicit scalar exponent. |
| `matrix_sign` | Implemented | Matrix sign function. |

## Decompositions And Spectral Helpers

| UDF | Status | Notes |
|---|---|---|
| `matrix_lu` | Implemented | Returns a struct containing `lower` and `upper`. |
| `matrix_cholesky` | Implemented | Returns a struct containing `lower`. |
| `matrix_cholesky_inverse` | Implemented | Row-wise inverse via Cholesky factorization. |
| `matrix_qr` | Implemented | Returns a struct containing `q`, `r`, and `rank`. |
| `matrix_qr_reduced` | Implemented | Returns reduced QR factors plus `rank`. |
| `matrix_qr_pivoted` | Implemented | Returns pivoted QR factors plus permutation and `rank`. |
| `matrix_qr_solve_least_squares` | Implemented | Row-wise least-squares solve. |
| `matrix_qr_condition_number` | Implemented | Row-wise scalar condition number helper. |
| `matrix_qr_reconstruct` | Implemented | Reconstructs the original matrix from QR output semantics. |
| `matrix_svd` | Implemented | Returns a struct containing `u`, `singular_values`, and `vt`. |
| `matrix_svd_truncated` | Implemented | Truncated SVD with explicit scalar `k`. |
| `matrix_svd_with_tolerance` | Implemented | Tolerance-thresholded SVD. |
| `matrix_svd_null_space` | Implemented | Returns a variable-shape tensor batch because null-space width varies by row. |
| `matrix_svd_pseudo_inverse` | Implemented | Row-wise pseudo-inverse helper. |
| `matrix_svd_condition_number` | Implemented | Row-wise scalar condition number helper. |
| `matrix_svd_rank` | Implemented | Row-wise scalar rank helper. |
| `matrix_svd_reconstruct` | Implemented | Reconstructs the original matrix from SVD output semantics. |
| `matrix_eigen_symmetric` | Implemented | Returns a struct containing eigenvalues and eigenvectors for symmetric inputs. |
| `matrix_eigen_generalized` | Implemented | Returns a struct containing eigenvalues and eigenvectors for generalized symmetric inputs. |
| `matrix_balance_nonsymmetric` | Implemented | Returns a struct containing the balanced matrix plus balancing diagonal. |
| `matrix_schur` | Implemented | Returns a paired-matrix Schur struct result. |
| `matrix_polar` | Implemented | Returns a paired-matrix polar-decomposition struct result. |
| `matrix_gram_schmidt` | Implemented | Modified Gram-Schmidt orthogonalization helper. |
| `matrix_gram_schmidt_classic` | Implemented | Classical Gram-Schmidt orthogonalization helper. |

## Sparse

| UDF | Status | Notes |
|---|---|---|
| `sparse_matvec` | Implemented | Row-wise CSR matrix times dense rank-1 variable-shape tensor batch. |
| `sparse_lu_solve` | Implemented | Row-wise sparse direct solve over square CSR matrices and dense rank-1 RHS batches. |
| `sparse_matmat_dense` | Implemented | Row-wise CSR matrix times dense rank-2 variable-shape tensor batch. |
| `sparse_transpose` | Implemented | CSR batch transpose. |
| `sparse_matmat_sparse` | Implemented | Row-wise CSR matrix times CSR matrix batch. |

## Tensor

| UDF | Status | Notes |
|---|---|---|
| `tensor_sum_last_axis` | Implemented | Fixed-shape last-axis reduction. |
| `tensor_l2_norm_last_axis` | Implemented | Fixed-shape last-axis norm. |
| `tensor_normalize_last_axis` | Implemented | Fixed-shape last-axis normalization. |
| `tensor_batched_dot_last_axis` | Implemented | Fixed-shape row-wise batched dot over the last axis. |
| `tensor_batched_matmul_last_two` | Implemented | Fixed-shape row-wise batched matmul over the last two axes. |
| `tensor_permute_axes` | Implemented | Fixed-shape row-wise axis permutation with variadic integer axes. |
| `tensor_contract_axes` | Implemented | Fixed-shape row-wise tensor contraction with variadic left/right axis pairs. |
| `tensor_variable_sum_last_axis` | Implemented | Variable-shape last-axis reduction. |
| `tensor_variable_l2_norm_last_axis` | Implemented | Variable-shape last-axis norm. |
| `tensor_variable_normalize_last_axis` | Implemented | Variable-shape last-axis normalization. |
| `tensor_variable_batched_dot_last_axis` | Implemented | Variable-shape row-wise batched dot over the last axis. |

## ML, Statistics, And Iterative Solvers

| UDF | Status | Notes |
|---|---|---|
| `matrix_column_means` | Implemented | Row-wise column means. |
| `matrix_center_columns` | Implemented | Row-wise column centering. |
| `matrix_covariance` | Implemented | Row-wise covariance matrix. |
| `matrix_correlation` | Implemented | Row-wise correlation matrix. |
| `matrix_pca` | Implemented | Returns a PCA struct containing `components`, explained-variance fields, `mean`, and `scores`. |
| `matrix_pca_transform` | Implemented | Applies an existing PCA struct to a matrix batch. |
| `matrix_pca_inverse_transform` | Implemented | Reconstructs feature space from score batches plus an existing PCA struct. |
| `matrix_conjugate_gradient` | Implemented | Dense iterative solve with explicit `tolerance` and `max_iterations`. |
| `matrix_gmres` | Implemented | Dense GMRES solve with explicit `tolerance` and `max_iterations`. |
| `linear_regression` | Implemented | Returns a struct containing `coefficients`, `fitted_values`, `residuals`, and `r_squared`. |

## Roadmap Items 

| Capability Family | Status | Notes |
|---|---|---|
| Complex-valued vector / matrix / tensor UDFs | Partial | Lower-layer Arrow contracts exist in `ndarrow` and `nabled`, but `ndatafusion` does not yet expose a stable SQL-facing complex catalog. |
| Non-symmetric eigendecomposition with complex outputs | Missing | Blocked on choosing complex result contracts for SQL. |
| Jacobian / gradient / Hessian | Missing | Generic `nabled` APIs are callback-driven; `ndatafusion` needs named-function registry or specialized built-ins instead of a direct closure-based surface. |
| Generic optimization workflows | Missing | No SQL-natural contract is admitted yet. |
| Stateful sparse factorization reuse and preconditioners | Missing | Requires object-carrying or stateful contracts beyond the current scalar-UDF-first design. |
| Tensor decompositions (`CP`, `Tucker`, `TT`) | Missing | No admitted SQL-facing contract yet. |
| UDAFs, window functions, table functions, and planner rewrites | Missing | Explicitly deferred; current implementation is scalar-UDF-first. |
