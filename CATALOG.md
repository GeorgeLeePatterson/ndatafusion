# UDF Catalog

Last updated: 2026-03-15

This document is the current inventory of the `ndatafusion` SQL catalog.
For small copy-paste queries, see [EXERCISES.md](https://github.com/GeorgeLeePatterson/ndatafusion/blob/master/EXERCISES.md).

Constructor UDFs are only needed when SQL starts from ordinary nested `List` values. If a table
already stores the canonical Arrow contract expected by a UDF, such as
`FixedSizeList<Float32|Float64>(D)` for dense vectors or the extension-backed matrix, tensor, and
sparse layouts produced by `ndarrow`, the numerical UDF can be called directly without `make_*`.

## Status Legend

- `Implemented`: available today through `register_all`.
- `Partial`: lower-layer support or direction exists, but `ndatafusion` does not yet expose a full,
  settled SQL-facing UDF surface.
- `Missing`: not currently exposed by `ndatafusion`.

## Alias Conventions

`ndatafusion` keeps the explicit canonical names in the catalog, and also registers a small set of
shorter SQL aliases for repetitive suffixes:

- `l2_norm` -> `norm`
- `last_axis` -> `last`
- `variable` -> `var`
- `least_squares` -> `ls`

Examples:

- `vector_norm` aliases `vector_l2_norm`
- `tensor_norm_last` aliases `tensor_l2_norm_last_axis`
- `tensor_var_sum_last` aliases `tensor_variable_sum_last_axis`
- `matrix_qr_solve_ls` aliases `matrix_qr_solve_least_squares`

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

| Method Or Surface | `nabled` | `ndatafusion` | Notes |
|---|---|---|---|
| `vector_dot_hermitian` | Implemented | Missing | `nabled::arrow::vector::dot_hermitian` exists, but `ndatafusion` does not yet expose a complex vector SQL surface. |
| `vector_batched_dot_hermitian` | Implemented | Missing | Row-wise Hermitian dot exists in `nabled`, but there is no complex batch vector catalog in `ndatafusion`. |
| `vector_batched_l2_norm_complex` | Implemented | Missing | Complex row-wise vector norms exist in `nabled`; `ndatafusion` only exposes real-valued vector norms today. |
| `vector_batched_cosine_similarity_complex` | Implemented | Missing | Complex cosine similarity exists in `nabled`; no SQL-facing complex vector contract is admitted yet. |
| `vector_batched_normalize_complex` | Implemented | Missing | Complex batch normalization exists in `nabled`; `ndatafusion` has no complex vector UDFs yet. |
| `matrix_matvec_complex`, `matrix_matmat_complex` | Implemented | Missing | Complex dense matrix products exist in `nabled::arrow::matrix`, but `ndatafusion` only exposes real-valued matrix products today. |
| `matrix_column_means_complex`, `matrix_center_columns_complex` | Implemented | Missing | Complex statistics exist in `nabled::arrow::stats`, but `ndatafusion` does not yet expose a complex statistics surface. |
| `matrix_covariance_complex`, `matrix_correlation_complex` | Implemented | Missing | Complex covariance and correlation are implemented in `nabled`, but `ndatafusion` currently stops at real-valued covariance/correlation. |
| `matrix_pca_complex` | Implemented | Missing | `nabled::arrow::pca::compute_complex` exists, but `ndatafusion` does not yet define a SQL-facing complex PCA contract. |
| `matrix_conjugate_gradient_complex`, `matrix_gmres_complex` | Implemented | Missing | Complex dense iterative solvers exist in `nabled::arrow::iterative`, but `ndatafusion` only exposes real-valued iterative solves today. |
| `matrix_eigen_nonsymmetric_f32`, `matrix_eigen_nonsymmetric_f64` | Implemented | Missing | `nabled::arrow::eigen::nonsymmetric_f32` and `nonsymmetric_f64` exist and return complex outputs; `ndatafusion` has not yet settled the SQL result contract for those complex results. |
| `matrix_eigen_nonsymmetric_bi_f32`, `matrix_eigen_nonsymmetric_bi_f64` | Implemented | Missing | Bi-eigen variants with left and right eigenvectors exist in `nabled`, but require a richer complex struct contract in `ndatafusion`. |
| `matrix_eigen_nonsymmetric_complex` | Implemented | Missing | Complex nonsymmetric eigendecomposition exists in `nabled::arrow::eigen`, but is not yet exposed in `ndatafusion`. |
| `matrix_schur_complex`, `matrix_polar_complex` | Implemented | Missing | Complex Schur and polar exist in `nabled`, but `ndatafusion` only exposes the current real-valued SQL contract. |
| `matrix_exp_complex`, `matrix_exp_eigen_complex` | Implemented | Missing | Complex matrix exponentials exist in `nabled::arrow::matrix_functions`; `ndatafusion` has not yet admitted complex matrix outputs. |
| `matrix_log_eigen_complex`, `matrix_log_svd_complex` | Implemented | Missing | Complex matrix logarithms exist in `nabled`, but `ndatafusion` does not yet expose them. |
| `matrix_power_complex`, `matrix_sign_complex` | Implemented | Missing | Complex matrix power and sign exist in `nabled`; `ndatafusion` currently exposes only real-valued variants. |
| `tensor_l2_norm_last_axis_complex`, `tensor_normalize_last_axis_complex` | Implemented | Missing | Complex fixed-shape tensor norm and normalization exist in `nabled::arrow::tensor`, but `ndatafusion` does not yet expose a complex tensor surface. |
| `tensor_variable_l2_norm_last_axis_complex`, `tensor_variable_normalize_last_axis_complex` | Implemented | Missing | Complex variable-shape tensor norm and normalization exist in `nabled`, but `ndatafusion` does not yet expose them. |
| `tensor_cp_als3`, `tensor_cp_als_nd` | Implemented | Missing | CP decomposition and related reporting/reconstruction helpers exist in `nabled::arrow::tensor`, but `ndatafusion` has not yet admitted a SQL-facing decomposition contract. |
| `tensor_hosvd_nd`, `tensor_hooi_nd` | Implemented | Missing | Higher-order SVD and HOOI exist in `nabled`, but there is no settled `ndatafusion` SQL contract yet. |
| `tensor_tucker_project`, `tensor_tucker_expand` | Implemented | Missing | Tucker projection and expansion exist in `nabled::arrow::tensor`, but are not yet exposed in `ndatafusion`. |
| `tensor_tt_svd`, `tensor_tt_orthogonalize_left`, `tensor_tt_orthogonalize_right`, `tensor_tt_round` | Implemented | Missing | Tensor-train factorization and orthogonalization exist in `nabled`, but `ndatafusion` has no SQL-facing TT contract yet. |
| `tensor_tt_inner`, `tensor_tt_norm`, `tensor_tt_add`, `tensor_tt_hadamard`, `tensor_tt_hadamard_round`, `tensor_tt_svd_reconstruct` | Implemented | Missing | Tensor-train algebra and reconstruction helpers exist in `nabled`; `ndatafusion` does not yet expose them. |
| `solve_sylvester`, `solve_sylvester_mixed_f64`, `solve_sylvester_complex` | Implemented | Missing | Sylvester solvers exist in `nabled-linalg`, but `ndatafusion` does not yet expose a matrix-equation SQL surface. |
| `gradient_descent_complex`, `adam_complex`, `momentum_descent_complex`, `backtracking_line_search_complex` | Implemented | Missing | Complex optimization helpers exist in `nabled`, but they are not yet shaped into a SQL-natural `ndatafusion` contract. |
| `sparse_lu_factor_csr_extension`, `sparse_lu_solve_with_factorization_csr_extension`, `sparse_lu_solve_multiple_with_factorization_csr_extension` | Implemented | Missing | Stateful sparse LU factorization and reuse exist in `nabled::arrow::sparse`, but `ndatafusion` currently avoids object-carrying sparse state contracts. |
| `jacobi_preconditioner_csr_extension`, `apply_jacobi_preconditioner` | Implemented | Missing | Jacobi preconditioner construction and application exist in `nabled`, but there is no SQL-facing stateful contract in `ndatafusion` yet. |
| `ilut_factor_csr_extension`, `iluk_factor_csr_extension` | Implemented | Missing | ILUT and ILUK factorization builders exist in `nabled::arrow::sparse`, but `ndatafusion` does not yet expose sparse factorization objects. |
| `jacobian`, `gradient`, `hessian` | Implemented | Missing | Generic `nabled` APIs are callback-driven; `ndatafusion` needs named-function registry or specialized built-ins instead of a direct closure-based SQL surface. |
| UDAFs, window functions, table functions, and planner rewrites | N/A | Missing | Explicitly deferred; the current `ndatafusion` implementation is scalar-UDF-first. |
