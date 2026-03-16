# UDF Catalog

Last updated: 2026-03-15

This document is the current inventory of the `ndatafusion` SQL catalog.
For small copy-paste queries, see [EXERCISES.md](https://github.com/GeorgeLeePatterson/ndatafusion/blob/master/EXERCISES.md).

Constructor UDFs are only needed when SQL starts from ordinary nested `List` values. If a table
already stores the canonical Arrow contract expected by a UDF, such as
`FixedSizeList<Float32|Float64>(D)` for dense vectors or the extension-backed matrix, tensor, and
sparse layouts produced by `ndarrow`, the numerical UDF can be called directly without `make_*`.

## Status Legend

- `Implemented`: available today through `register_all` or `register_all_session`.
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

## Named Argument Support

`ndatafusion` also supports named SQL arguments on selected fixed-arity constructors and
control-parameter UDFs where argument order is easy to confuse. Current named-argument coverage
includes:

- `make_vector`
- `make_matrix`
- `make_tensor`
- `make_variable_tensor`
- `make_csr_matrix_batch`
- `matrix_exp`
- `matrix_exp_complex`
- `matrix_log_taylor`
- `matrix_power`
- `matrix_power_complex`
- `matrix_eigen_nonsymmetric`
- `matrix_eigen_nonsymmetric_bi`
- `matrix_eigen_nonsymmetric_complex`
- `matrix_conjugate_gradient`
- `matrix_gmres`
- `matrix_conjugate_gradient_complex`
- `matrix_gmres_complex`
- `matrix_svd_truncated`
- `matrix_svd_with_tolerance`
- `linear_regression`
- `linear_regression_fit`
- `jacobian`
- `jacobian_central`
- `gradient`
- `hessian`
- `backtracking_line_search_complex`
- `gradient_descent_complex`
- `adam_complex`
- `momentum_descent_complex`
- `sparse_ilut_factor`
- `sparse_iluk_factor`
- `matrix_pca_complex`
- `matrix_pca_transform_complex`
- `matrix_pca_inverse_transform_complex`
- `tensor_cp_als3`
- `tensor_cp_als_nd`
- `tensor_hosvd_nd`
- `tensor_hooi_nd`
- `tensor_tt_svd`
- `tensor_tt_hadamard_round`

For numerical UDFs with data operands followed by control scalars, prefer positional data
arguments first and named trailing controls after. For example:

- `matrix_exp(matrix_batch, max_terms => 32, tolerance => 1e-6)`
- `matrix_svd_truncated(matrix_batch, k => 8)`
- `matrix_conjugate_gradient(matrix_batch, rhs_batch, tolerance => 1e-6, max_iterations => 64)`

Constructors remain a good fit for fully named calls because their arguments are structural and
order-sensitive.

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

## Complex Vectors

| UDF | Status | Notes |
|---|---|---|
| `vector_dot_hermitian` | Implemented | Row-wise Hermitian dot product over canonical `FixedSizeList<ndarrow.complex64>(D)` batches. |
| `vector_l2_norm_complex` | Implemented | Row-wise complex L2 norm with real-valued output. |
| `vector_cosine_similarity_complex` | Implemented | Row-wise complex cosine similarity with complex-valued output. |
| `vector_normalize_complex` | Implemented | Row-wise complex vector normalization. |

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

## Complex Matrix Ops, Statistics, And Iterative Solvers

| UDF | Status | Notes |
|---|---|---|
| `matrix_matvec_complex` | Implemented | Row-wise complex matrix-vector product over canonical `arrow.fixed_shape_tensor<ndarrow.complex64>` plus `FixedSizeList<ndarrow.complex64>(D)` inputs. |
| `matrix_matmat_complex` | Implemented | Row-wise complex matrix-matrix product over canonical complex matrix batches. |
| `matrix_column_means_complex` | Implemented | Row-wise complex column means. |
| `matrix_center_columns_complex` | Implemented | Row-wise complex column centering. |
| `matrix_covariance_complex` | Implemented | Row-wise complex covariance matrix. |
| `matrix_correlation_complex` | Implemented | Row-wise complex correlation matrix. |
| `matrix_conjugate_gradient_complex` | Implemented | Complex dense iterative solve with explicit `tolerance` and `max_iterations`. |
| `matrix_gmres_complex` | Implemented | Complex dense GMRES solve with explicit `tolerance` and `max_iterations`. |

## Complex Matrix Functions

| UDF | Status | Notes |
|---|---|---|
| `matrix_exp_complex` | Implemented | Configurable complex matrix exponential with explicit `max_terms` and `tolerance`. |
| `matrix_exp_eigen_complex` | Implemented | Zero-config eigen-based complex matrix exponential path. |
| `matrix_log_eigen_complex` | Implemented | Zero-config eigen-based complex matrix logarithm path. |
| `matrix_log_svd_complex` | Implemented | Zero-config SVD-based complex matrix logarithm path. |
| `matrix_power_complex` | Implemented | Complex matrix power with explicit scalar exponent. |
| `matrix_sign_complex` | Implemented | Complex matrix sign function. |

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
| `matrix_eigen_nonsymmetric` | Implemented | Returns a struct containing complex `eigenvalues` and complex `schur_vectors` for each real square matrix batch row. |
| `matrix_eigen_nonsymmetric_bi` | Implemented | Returns complex `eigenvalues`, complex left/right eigenvectors, real `balancing_diagonal`, and real `balanced_matrix` for each real square matrix batch row. |
| `matrix_balance_nonsymmetric` | Implemented | Returns a struct containing the balanced matrix plus balancing diagonal. |
| `matrix_schur` | Implemented | Returns a paired-matrix Schur struct result. |
| `matrix_polar` | Implemented | Returns a paired-matrix polar-decomposition struct result. |
| `matrix_gram_schmidt` | Implemented | Modified Gram-Schmidt orthogonalization helper. |
| `matrix_gram_schmidt_classic` | Implemented | Classical Gram-Schmidt orthogonalization helper. |

## Complex Decompositions

| UDF | Status | Notes |
|---|---|---|
| `matrix_eigen_nonsymmetric_complex` | Implemented | Returns a struct containing complex `eigenvalues` and complex `schur_vectors` for each complex square matrix batch row. |
| `matrix_schur_complex` | Implemented | Returns a complex paired-matrix Schur struct result with `q` and `t`. |
| `matrix_polar_complex` | Implemented | Returns a complex paired-matrix polar-decomposition struct result with `u` and `p`. |

## Sparse

| UDF | Status | Notes |
|---|---|---|
| `sparse_matvec` | Implemented | Row-wise CSR matrix times dense rank-1 variable-shape tensor batch. |
| `sparse_lu_solve` | Implemented | Row-wise sparse direct solve over square CSR matrices and dense rank-1 RHS batches. |
| `sparse_matmat_dense` | Implemented | Row-wise CSR matrix times dense rank-2 variable-shape tensor batch. |
| `sparse_transpose` | Implemented | CSR batch transpose. |
| `sparse_matmat_sparse` | Implemented | Row-wise CSR matrix times CSR matrix batch. |

## Matrix Equations

| UDF | Status | Notes |
|---|---|---|
| `matrix_solve_sylvester` | Implemented | Solves `A X + X B = C` for each real batch row over canonical fixed-shape matrix batches. |
| `matrix_solve_sylvester_mixed_f64` | Implemented | Mixed-precision Sylvester solve over `Float64` matrix batches with a struct result containing `solution` and `refinement_iterations`; requires `magma-system` at runtime. |
| `matrix_solve_sylvester_complex` | Implemented | Solves `A X + X B = C` for each complex batch row over canonical `ndarrow.complex64` matrix batches. |
| `matrix_solve_sylvester_mixed_complex` | Implemented | Mixed-precision complex Sylvester solve with a struct result containing `solution` and `refinement_iterations`; requires `magma-system` at runtime. |

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

## Complex Tensor

| UDF | Status | Notes |
|---|---|---|
| `tensor_l2_norm_last_axis_complex` | Implemented | Fixed-shape complex last-axis norm with real-valued output. |
| `tensor_normalize_last_axis_complex` | Implemented | Fixed-shape complex last-axis normalization. |
| `tensor_variable_l2_norm_last_axis_complex` | Implemented | Variable-shape complex last-axis norm with real-valued output. |
| `tensor_variable_normalize_last_axis_complex` | Implemented | Variable-shape complex last-axis normalization. |

## ML, Statistics, And Iterative Solvers

| UDF | Status | Notes |
|---|---|---|
| `matrix_column_means` | Implemented | Row-wise column means. |
| `matrix_center_columns` | Implemented | Row-wise column centering. |
| `matrix_covariance` | Implemented | Row-wise covariance matrix. |
| `matrix_correlation` | Implemented | Row-wise correlation matrix. |
| `matrix_pca` | Implemented | Returns a PCA struct containing `components`, explained-variance fields, `mean`, and `scores`. |
| `matrix_pca_complex` | Implemented | Returns a complex PCA struct with complex `components`, `mean`, `scores`, and `Float64` explained-variance fields. |
| `matrix_pca_transform` | Implemented | Applies an existing PCA struct to a matrix batch. |
| `matrix_pca_transform_complex` | Implemented | Applies an existing complex PCA struct to a dense complex matrix batch. |
| `matrix_pca_inverse_transform` | Implemented | Reconstructs feature space from score batches plus an existing PCA struct. |
| `matrix_pca_inverse_transform_complex` | Implemented | Reconstructs complex feature space from complex score batches plus an existing complex PCA struct. |
| `matrix_conjugate_gradient` | Implemented | Dense iterative solve with explicit `tolerance` and `max_iterations`. |
| `matrix_gmres` | Implemented | Dense GMRES solve with explicit `tolerance` and `max_iterations`. |
| `linear_regression` | Implemented | Returns a struct containing `coefficients`, `fitted_values`, `residuals`, and `r_squared`. |

## Aggregate UDFs

| UDF | Status | Notes |
|---|---|---|
| `vector_covariance_agg` | Implemented | Grouped covariance matrix over canonical dense vector rows. |
| `vector_correlation_agg` | Implemented | Grouped correlation matrix over canonical dense vector rows. |
| `vector_pca_fit` | Implemented | Grouped PCA fit returning `components`, explained-variance fields, and `mean`. |
| `linear_regression_fit` | Implemented | Grouped linear-regression fit returning `coefficients` and `r_squared`. |

## Differentiation

| UDF | Status | Notes |
|---|---|---|
| `jacobian` | Implemented | Named-function Jacobian with explicit `step_size`, `tolerance`, and `max_iterations`. |
| `jacobian_central` | Implemented | Named-function central-difference Jacobian with the same control contract. |
| `gradient` | Implemented | Named-function gradient over canonical dense vector batches. |
| `hessian` | Implemented | Named-function Hessian over canonical dense vector batches. |

## Optimization

| UDF | Status | Notes |
|---|---|---|
| `backtracking_line_search_complex` | Implemented | Named-function complex line-search helper with explicit control arguments. |
| `gradient_descent_complex` | Implemented | Named-function complex gradient-descent helper. |
| `momentum_descent_complex` | Implemented | Named-function complex momentum-descent helper. |
| `adam_complex` | Implemented | Named-function complex Adam helper. |

## Sparse Factorization And Preconditioners

| UDF | Status | Notes |
|---|---|---|
| `sparse_lu_factor` | Implemented | Returns a sparse LU factorization struct over canonical CSR batches. |
| `sparse_lu_solve_with_factorization` | Implemented | Solves dense rank-1 RHS batches from a sparse LU factorization struct. |
| `sparse_lu_solve_multiple_with_factorization` | Implemented | Solves dense rank-2 RHS batches from a sparse LU factorization struct. |
| `sparse_jacobi_preconditioner` | Implemented | Returns a Jacobi preconditioner struct. |
| `sparse_apply_jacobi_preconditioner` | Implemented | Applies a Jacobi preconditioner struct to dense rank-1 RHS batches. |
| `sparse_ilut_factor` | Implemented | Returns an ILUT factorization struct with explicit `drop_tolerance` and `max_fill`. |
| `sparse_apply_ilut_preconditioner` | Implemented | Applies an ILUT factorization struct to dense rank-1 RHS batches. |
| `sparse_iluk_factor` | Implemented | Returns an ILU(k) factorization struct with explicit `level_of_fill`. |
| `sparse_apply_iluk_preconditioner` | Implemented | Applies an ILU(k) factorization struct to dense rank-1 RHS batches. |

## Tensor Decomposition

| UDF | Status | Notes |
|---|---|---|
| `tensor_cp_als3` | Implemented | Rank-3 CP-ALS decomposition over fixed-shape tensor batches. |
| `tensor_cp_als3_reconstruct` | Implemented | Reconstructs from the CP-ALS3 struct contract. |
| `tensor_cp_als_nd` | Implemented | N-dimensional CP-ALS decomposition over fixed-shape tensor batches. |
| `tensor_cp_als_nd_reconstruct` | Implemented | Reconstructs from the CP-ALS-ND struct contract. |
| `tensor_hosvd_nd` | Implemented | Higher-order SVD over fixed-shape tensor batches. |
| `tensor_hooi_nd` | Implemented | Higher-order orthogonal iteration over fixed-shape tensor batches. |
| `tensor_tucker_project` | Implemented | Projects a tensor batch into Tucker core space. |
| `tensor_tucker_expand` | Implemented | Expands a Tucker core back into tensor space. |
| `tensor_tt_svd` | Implemented | Tensor-train SVD with explicit `max_rank` and `tolerance`. |
| `tensor_tt_svd_reconstruct` | Implemented | Reconstructs from the tensor-train SVD struct contract. |
| `tensor_tt_orthogonalize_left` | Implemented | Left-orthogonalizes a tensor-train batch. |
| `tensor_tt_orthogonalize_right` | Implemented | Right-orthogonalizes a tensor-train batch. |
| `tensor_tt_round` | Implemented | Rounds a tensor-train batch to a target rank/tolerance. |
| `tensor_tt_inner` | Implemented | Tensor-train inner-product helper. |
| `tensor_tt_norm` | Implemented | Tensor-train norm helper. |
| `tensor_tt_add` | Implemented | Tensor-train addition helper. |
| `tensor_tt_hadamard` | Implemented | Tensor-train Hadamard-product helper. |
| `tensor_tt_hadamard_round` | Implemented | Tensor-train Hadamard-product helper with explicit rounding controls. |

## Table Functions And Window Usage

| Surface | Status | Notes |
|---|---|---|
| `unpack_struct` | Implemented | Registered through `register_all_session`; expands a scalar struct-valued expression into a one-row relation. |
| Ordered window use of aggregates | Implemented | The aggregate UDF wave supports retractable ordered windows through `OVER (...)` frames. |

## Future Directions

| Surface | Status | Notes |
|---|---|---|
| Additional planner hooks beyond `simplify` | Partial | Per-UDF simplify hooks are implemented today; other hooks should be added only when they improve optimization materially. |
| Custom expression planning | Missing | Reserved for SQL shapes that cannot be represented cleanly as scalar, aggregate, or table functions. |
| Richer table-function catalog | Partial | `unpack_struct` exists now; additional relation-shaped functions should be added only where they are clearer than struct-valued scalar outputs. |
| Dedicated `WindowUDF` catalog | Partial | Rolling use cases already work through retractable aggregates; standalone window functions are optional, not required. |
| crates.io publication | Missing | Still blocked by the upstream Arrow 58 / DataFusion published-release mismatch. |
