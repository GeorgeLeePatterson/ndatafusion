//! Expression builders for the `ndatafusion` scalar and aggregate UDF catalog.
//!
//! Each function in this module returns a `datafusion::logical_expr::Expr` that calls a registered
//! `ndatafusion` scalar or aggregate UDF. The helper names match the SQL function names.
//!
//! For the complete catalog and notes on input and output contracts, see `CATALOG.md` in the
//! repository root.

use datafusion::logical_expr::Expr;

use crate::{udafs, udfs};

#[must_use]
pub fn make_vector(values: Expr, dim: Expr) -> Expr {
    udfs::make_vector_udf().call(vec![values, dim])
}

#[must_use]
pub fn make_matrix(values: Expr, rows: Expr, cols: Expr) -> Expr {
    udfs::make_matrix_udf().call(vec![values, rows, cols])
}

#[must_use]
pub fn make_tensor(values: Expr, dims: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(dims.len() + 1);
    args.push(values);
    args.extend(dims);
    udfs::make_tensor_udf().call(args)
}

#[must_use]
pub fn make_variable_tensor(data: Expr, shape: Expr, rank: Expr) -> Expr {
    udfs::make_variable_tensor_udf().call(vec![data, shape, rank])
}

#[must_use]
pub fn make_csr_matrix_batch(shape: Expr, row_ptrs: Expr, col_indices: Expr, values: Expr) -> Expr {
    udfs::make_csr_matrix_batch_udf().call(vec![shape, row_ptrs, col_indices, values])
}

#[must_use]
pub fn vector_l2_norm(vector: Expr) -> Expr { udfs::vector_l2_norm_udf().call(vec![vector]) }

#[must_use]
pub fn vector_dot(left: Expr, right: Expr) -> Expr {
    udfs::vector_dot_udf().call(vec![left, right])
}

#[must_use]
pub fn vector_cosine_similarity(left: Expr, right: Expr) -> Expr {
    udfs::vector_cosine_similarity_udf().call(vec![left, right])
}

#[must_use]
pub fn vector_cosine_distance(left: Expr, right: Expr) -> Expr {
    udfs::vector_cosine_distance_udf().call(vec![left, right])
}

#[must_use]
pub fn vector_normalize(vector: Expr) -> Expr { udfs::vector_normalize_udf().call(vec![vector]) }

#[must_use]
pub fn jacobian(function: Expr, vector: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(function);
    args.push(vector);
    args.extend(config);
    udfs::jacobian_udf().call(args)
}

#[must_use]
pub fn jacobian_central(function: Expr, vector: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(function);
    args.push(vector);
    args.extend(config);
    udfs::jacobian_central_udf().call(args)
}

#[must_use]
pub fn gradient(function: Expr, vector: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(function);
    args.push(vector);
    args.extend(config);
    udfs::gradient_udf().call(args)
}

#[must_use]
pub fn hessian(function: Expr, vector: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(function);
    args.push(vector);
    args.extend(config);
    udfs::hessian_udf().call(args)
}

#[must_use]
pub fn backtracking_line_search_complex(
    function: Expr,
    point: Expr,
    direction: Expr,
    config: Vec<Expr>,
) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 3);
    args.push(function);
    args.push(point);
    args.push(direction);
    args.extend(config);
    udfs::backtracking_line_search_complex_udf().call(args)
}

#[must_use]
pub fn gradient_descent_complex(function: Expr, initial: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(function);
    args.push(initial);
    args.extend(config);
    udfs::gradient_descent_complex_udf().call(args)
}

#[must_use]
pub fn adam_complex(function: Expr, initial: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(function);
    args.push(initial);
    args.extend(config);
    udfs::adam_complex_udf().call(args)
}

#[must_use]
pub fn momentum_descent_complex(function: Expr, initial: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(function);
    args.push(initial);
    args.extend(config);
    udfs::momentum_descent_complex_udf().call(args)
}

#[must_use]
pub fn vector_dot_hermitian(left: Expr, right: Expr) -> Expr {
    udfs::vector_dot_hermitian_udf().call(vec![left, right])
}

#[must_use]
pub fn vector_l2_norm_complex(vector: Expr) -> Expr {
    udfs::vector_l2_norm_complex_udf().call(vec![vector])
}

#[must_use]
pub fn vector_cosine_similarity_complex(left: Expr, right: Expr) -> Expr {
    udfs::vector_cosine_similarity_complex_udf().call(vec![left, right])
}

#[must_use]
pub fn vector_normalize_complex(vector: Expr) -> Expr {
    udfs::vector_normalize_complex_udf().call(vec![vector])
}

#[must_use]
pub fn matrix_matvec_complex(matrix: Expr, vector: Expr) -> Expr {
    udfs::matrix_matvec_complex_udf().call(vec![matrix, vector])
}

#[must_use]
pub fn matrix_matvec(matrix: Expr, vector: Expr) -> Expr {
    udfs::matrix_matvec_udf().call(vec![matrix, vector])
}

#[must_use]
pub fn matrix_matmat_complex(left: Expr, right: Expr) -> Expr {
    udfs::matrix_matmat_complex_udf().call(vec![left, right])
}

#[must_use]
pub fn matrix_matmul(left: Expr, right: Expr) -> Expr {
    udfs::matrix_matmul_udf().call(vec![left, right])
}

#[must_use]
pub fn matrix_exp(matrix: Expr, max_terms: Expr, tolerance: Expr) -> Expr {
    udfs::matrix_exp_udf().call(vec![matrix, max_terms, tolerance])
}

#[must_use]
pub fn matrix_exp_complex(matrix: Expr, max_terms: Expr, tolerance: Expr) -> Expr {
    udfs::matrix_exp_complex_udf().call(vec![matrix, max_terms, tolerance])
}

#[must_use]
pub fn matrix_solve_lower(matrix: Expr, rhs: Expr) -> Expr {
    udfs::matrix_solve_lower_udf().call(vec![matrix, rhs])
}

#[must_use]
pub fn matrix_solve_upper(matrix: Expr, rhs: Expr) -> Expr {
    udfs::matrix_solve_upper_udf().call(vec![matrix, rhs])
}

#[must_use]
pub fn matrix_solve_lower_matrix(matrix: Expr, rhs: Expr) -> Expr {
    udfs::matrix_solve_lower_matrix_udf().call(vec![matrix, rhs])
}

#[must_use]
pub fn matrix_solve_upper_matrix(matrix: Expr, rhs: Expr) -> Expr {
    udfs::matrix_solve_upper_matrix_udf().call(vec![matrix, rhs])
}

#[must_use]
pub fn matrix_solve_sylvester(matrix_a: Expr, matrix_b: Expr, matrix_c: Expr) -> Expr {
    udfs::matrix_solve_sylvester_udf().call(vec![matrix_a, matrix_b, matrix_c])
}

#[must_use]
pub fn matrix_solve_sylvester_mixed_f64(matrix_a: Expr, matrix_b: Expr, matrix_c: Expr) -> Expr {
    udfs::matrix_solve_sylvester_mixed_f64_udf().call(vec![matrix_a, matrix_b, matrix_c])
}

#[must_use]
pub fn matrix_solve_sylvester_complex(matrix_a: Expr, matrix_b: Expr, matrix_c: Expr) -> Expr {
    udfs::matrix_solve_sylvester_complex_udf().call(vec![matrix_a, matrix_b, matrix_c])
}

#[must_use]
pub fn matrix_solve_sylvester_mixed_complex(
    matrix_a: Expr,
    matrix_b: Expr,
    matrix_c: Expr,
) -> Expr {
    udfs::matrix_solve_sylvester_mixed_complex_udf().call(vec![matrix_a, matrix_b, matrix_c])
}

#[must_use]
pub fn matrix_lu(matrix: Expr) -> Expr { udfs::matrix_lu_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_lu_solve(matrix: Expr, rhs: Expr) -> Expr {
    udfs::matrix_lu_solve_udf().call(vec![matrix, rhs])
}

#[must_use]
pub fn matrix_cholesky_solve(matrix: Expr, rhs: Expr) -> Expr {
    udfs::matrix_cholesky_solve_udf().call(vec![matrix, rhs])
}

#[must_use]
pub fn matrix_inverse(matrix: Expr) -> Expr { udfs::matrix_inverse_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_determinant(matrix: Expr) -> Expr {
    udfs::matrix_determinant_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_log_determinant(matrix: Expr) -> Expr {
    udfs::matrix_log_determinant_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_cholesky(matrix: Expr) -> Expr { udfs::matrix_cholesky_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_cholesky_inverse(matrix: Expr) -> Expr {
    udfs::matrix_cholesky_inverse_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_qr(matrix: Expr) -> Expr { udfs::matrix_qr_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_qr_reduced(matrix: Expr) -> Expr { udfs::matrix_qr_reduced_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_qr_pivoted(matrix: Expr) -> Expr { udfs::matrix_qr_pivoted_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_qr_solve_least_squares(matrix: Expr, rhs: Expr) -> Expr {
    udfs::matrix_qr_solve_least_squares_udf().call(vec![matrix, rhs])
}

#[must_use]
pub fn matrix_qr_condition_number(matrix: Expr) -> Expr {
    udfs::matrix_qr_condition_number_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_qr_reconstruct(matrix: Expr) -> Expr {
    udfs::matrix_qr_reconstruct_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_svd(matrix: Expr) -> Expr { udfs::matrix_svd_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_svd_truncated(matrix: Expr, k: Expr) -> Expr {
    udfs::matrix_svd_truncated_udf().call(vec![matrix, k])
}

#[must_use]
pub fn matrix_svd_with_tolerance(matrix: Expr, tolerance: Expr) -> Expr {
    udfs::matrix_svd_with_tolerance_udf().call(vec![matrix, tolerance])
}

#[must_use]
pub fn matrix_svd_null_space(matrix: Expr) -> Expr {
    udfs::matrix_svd_null_space_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_svd_pseudo_inverse(matrix: Expr) -> Expr {
    udfs::matrix_svd_pseudo_inverse_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_svd_condition_number(matrix: Expr) -> Expr {
    udfs::matrix_svd_condition_number_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_svd_rank(matrix: Expr) -> Expr { udfs::matrix_svd_rank_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_svd_reconstruct(matrix: Expr) -> Expr {
    udfs::matrix_svd_reconstruct_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_eigen_symmetric(matrix: Expr) -> Expr {
    udfs::matrix_eigen_symmetric_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_eigen_generalized(left: Expr, right: Expr) -> Expr {
    udfs::matrix_eigen_generalized_udf().call(vec![left, right])
}

#[must_use]
pub fn matrix_eigen_nonsymmetric(matrix: Expr) -> Expr {
    udfs::matrix_eigen_nonsymmetric_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_eigen_nonsymmetric_bi(matrix: Expr) -> Expr {
    udfs::matrix_eigen_nonsymmetric_bi_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_eigen_nonsymmetric_complex(matrix: Expr) -> Expr {
    udfs::matrix_eigen_nonsymmetric_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_balance_nonsymmetric(matrix: Expr) -> Expr {
    udfs::matrix_balance_nonsymmetric_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_schur(matrix: Expr) -> Expr { udfs::matrix_schur_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_schur_complex(matrix: Expr) -> Expr {
    udfs::matrix_schur_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_polar(matrix: Expr) -> Expr { udfs::matrix_polar_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_polar_complex(matrix: Expr) -> Expr {
    udfs::matrix_polar_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_gram_schmidt(matrix: Expr) -> Expr {
    udfs::matrix_gram_schmidt_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_gram_schmidt_classic(matrix: Expr) -> Expr {
    udfs::matrix_gram_schmidt_classic_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_exp_eigen(matrix: Expr) -> Expr { udfs::matrix_exp_eigen_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_exp_eigen_complex(matrix: Expr) -> Expr {
    udfs::matrix_exp_eigen_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_log_taylor(matrix: Expr, max_terms: Expr, tolerance: Expr) -> Expr {
    udfs::matrix_log_taylor_udf().call(vec![matrix, max_terms, tolerance])
}

#[must_use]
pub fn matrix_log_eigen(matrix: Expr) -> Expr { udfs::matrix_log_eigen_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_log_eigen_complex(matrix: Expr) -> Expr {
    udfs::matrix_log_eigen_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_log_svd(matrix: Expr) -> Expr { udfs::matrix_log_svd_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_log_svd_complex(matrix: Expr) -> Expr {
    udfs::matrix_log_svd_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_power(matrix: Expr, power: Expr) -> Expr {
    udfs::matrix_power_udf().call(vec![matrix, power])
}

#[must_use]
pub fn matrix_power_complex(matrix: Expr, power: Expr) -> Expr {
    udfs::matrix_power_complex_udf().call(vec![matrix, power])
}

#[must_use]
pub fn matrix_sign(matrix: Expr) -> Expr { udfs::matrix_sign_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_sign_complex(matrix: Expr) -> Expr {
    udfs::matrix_sign_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn sparse_matvec(matrices: Expr, vectors: Expr) -> Expr {
    udfs::sparse_matvec_udf().call(vec![matrices, vectors])
}

#[must_use]
pub fn sparse_lu_solve(matrices: Expr, rhs: Expr) -> Expr {
    udfs::sparse_lu_solve_udf().call(vec![matrices, rhs])
}

#[must_use]
pub fn sparse_lu_factor(matrices: Expr) -> Expr {
    udfs::sparse_lu_factor_udf().call(vec![matrices])
}

#[must_use]
pub fn sparse_lu_solve_with_factorization(matrices: Expr, rhs: Expr, factorization: Expr) -> Expr {
    udfs::sparse_lu_solve_with_factorization_udf().call(vec![matrices, rhs, factorization])
}

#[must_use]
pub fn sparse_lu_solve_multiple_with_factorization(
    matrices: Expr,
    rhs: Expr,
    factorization: Expr,
) -> Expr {
    udfs::sparse_lu_solve_multiple_with_factorization_udf().call(vec![matrices, rhs, factorization])
}

#[must_use]
pub fn sparse_jacobi_preconditioner(matrices: Expr) -> Expr {
    udfs::sparse_jacobi_preconditioner_udf().call(vec![matrices])
}

#[must_use]
pub fn sparse_apply_jacobi_preconditioner(preconditioner: Expr, rhs: Expr) -> Expr {
    udfs::sparse_apply_jacobi_preconditioner_udf().call(vec![preconditioner, rhs])
}

#[must_use]
pub fn sparse_ilut_factor(matrices: Expr, drop_tolerance: Expr, max_fill: Expr) -> Expr {
    udfs::sparse_ilut_factor_udf().call(vec![matrices, drop_tolerance, max_fill])
}

#[must_use]
pub fn sparse_iluk_factor(matrices: Expr, level_of_fill: Expr) -> Expr {
    udfs::sparse_iluk_factor_udf().call(vec![matrices, level_of_fill])
}

#[must_use]
pub fn sparse_apply_ilut_preconditioner(factorization: Expr, rhs: Expr) -> Expr {
    udfs::sparse_apply_ilut_preconditioner_udf().call(vec![factorization, rhs])
}

#[must_use]
pub fn sparse_apply_iluk_preconditioner(factorization: Expr, rhs: Expr) -> Expr {
    udfs::sparse_apply_iluk_preconditioner_udf().call(vec![factorization, rhs])
}

#[must_use]
pub fn sparse_matmat_dense(matrices: Expr, dense: Expr) -> Expr {
    udfs::sparse_matmat_dense_udf().call(vec![matrices, dense])
}

#[must_use]
pub fn sparse_transpose(matrices: Expr) -> Expr {
    udfs::sparse_transpose_udf().call(vec![matrices])
}

#[must_use]
pub fn sparse_matmat_sparse(left: Expr, right: Expr) -> Expr {
    udfs::sparse_matmat_sparse_udf().call(vec![left, right])
}

#[must_use]
pub fn tensor_sum_last_axis(tensor: Expr) -> Expr {
    udfs::tensor_sum_last_axis_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_l2_norm_last_axis(tensor: Expr) -> Expr {
    udfs::tensor_l2_norm_last_axis_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_l2_norm_last_axis_complex(tensor: Expr) -> Expr {
    udfs::tensor_l2_norm_last_axis_complex_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_normalize_last_axis(tensor: Expr) -> Expr {
    udfs::tensor_normalize_last_axis_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_normalize_last_axis_complex(tensor: Expr) -> Expr {
    udfs::tensor_normalize_last_axis_complex_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_batched_dot_last_axis(left: Expr, right: Expr) -> Expr {
    udfs::tensor_batched_dot_last_axis_udf().call(vec![left, right])
}

#[must_use]
pub fn tensor_batched_matmul_last_two(left: Expr, right: Expr) -> Expr {
    udfs::tensor_batched_matmul_last_two_udf().call(vec![left, right])
}

#[must_use]
pub fn tensor_permute_axes(tensor: Expr, axes: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(axes.len() + 1);
    args.push(tensor);
    args.extend(axes);
    udfs::tensor_permute_axes_udf().call(args)
}

#[must_use]
pub fn tensor_contract_axes(left: Expr, right: Expr, axes: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(axes.len() + 2);
    args.push(left);
    args.push(right);
    args.extend(axes);
    udfs::tensor_contract_axes_udf().call(args)
}

#[must_use]
pub fn tensor_variable_sum_last_axis(tensor: Expr) -> Expr {
    udfs::tensor_variable_sum_last_axis_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_variable_l2_norm_last_axis(tensor: Expr) -> Expr {
    udfs::tensor_variable_l2_norm_last_axis_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_variable_l2_norm_last_axis_complex(tensor: Expr) -> Expr {
    udfs::tensor_variable_l2_norm_last_axis_complex_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_variable_normalize_last_axis(tensor: Expr) -> Expr {
    udfs::tensor_variable_normalize_last_axis_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_variable_normalize_last_axis_complex(tensor: Expr) -> Expr {
    udfs::tensor_variable_normalize_last_axis_complex_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_variable_batched_dot_last_axis(left: Expr, right: Expr) -> Expr {
    udfs::tensor_variable_batched_dot_last_axis_udf().call(vec![left, right])
}

#[must_use]
pub fn tensor_cp_als3(tensor: Expr, rank: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(tensor);
    args.push(rank);
    args.extend(config);
    udfs::tensor_cp_als3_udf().call(args)
}

#[must_use]
pub fn tensor_cp_als3_reconstruct(cp: Expr) -> Expr {
    udfs::tensor_cp_als3_reconstruct_udf().call(vec![cp])
}

#[must_use]
pub fn tensor_cp_als_nd(tensor: Expr, rank: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(tensor);
    args.push(rank);
    args.extend(config);
    udfs::tensor_cp_als_nd_udf().call(args)
}

#[must_use]
pub fn tensor_cp_als_nd_reconstruct(cp: Expr) -> Expr {
    udfs::tensor_cp_als_nd_reconstruct_udf().call(vec![cp])
}

#[must_use]
pub fn tensor_hosvd_nd(tensor: Expr, ranks: Expr) -> Expr {
    udfs::tensor_hosvd_nd_udf().call(vec![tensor, ranks])
}

#[must_use]
pub fn tensor_hooi_nd(tensor: Expr, ranks: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(tensor);
    args.push(ranks);
    args.extend(config);
    udfs::tensor_hooi_nd_udf().call(args)
}

#[must_use]
pub fn tensor_tucker_project(tensor: Expr, decomposition: Expr) -> Expr {
    udfs::tensor_tucker_project_udf().call(vec![tensor, decomposition])
}

#[must_use]
pub fn tensor_tucker_expand(decomposition: Expr) -> Expr {
    udfs::tensor_tucker_expand_udf().call(vec![decomposition])
}

#[must_use]
pub fn tensor_tt_svd(tensor: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 1);
    args.push(tensor);
    args.extend(config);
    udfs::tensor_tt_svd_udf().call(args)
}

#[must_use]
pub fn tensor_tt_orthogonalize_left(tt: Expr) -> Expr {
    udfs::tensor_tt_orthogonalize_left_udf().call(vec![tt])
}

#[must_use]
pub fn tensor_tt_orthogonalize_right(tt: Expr) -> Expr {
    udfs::tensor_tt_orthogonalize_right_udf().call(vec![tt])
}

#[must_use]
pub fn tensor_tt_round(tt: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 1);
    args.push(tt);
    args.extend(config);
    udfs::tensor_tt_round_udf().call(args)
}

#[must_use]
pub fn tensor_tt_inner(left: Expr, right: Expr) -> Expr {
    udfs::tensor_tt_inner_udf().call(vec![left, right])
}

#[must_use]
pub fn tensor_tt_norm(tt: Expr) -> Expr { udfs::tensor_tt_norm_udf().call(vec![tt]) }

#[must_use]
pub fn tensor_tt_add(left: Expr, right: Expr) -> Expr {
    udfs::tensor_tt_add_udf().call(vec![left, right])
}

#[must_use]
pub fn tensor_tt_hadamard(left: Expr, right: Expr) -> Expr {
    udfs::tensor_tt_hadamard_udf().call(vec![left, right])
}

#[must_use]
pub fn tensor_tt_hadamard_round(left: Expr, right: Expr, config: Vec<Expr>) -> Expr {
    let mut args = Vec::with_capacity(config.len() + 2);
    args.push(left);
    args.push(right);
    args.extend(config);
    udfs::tensor_tt_hadamard_round_udf().call(args)
}

#[must_use]
pub fn tensor_tt_svd_reconstruct(tt: Expr) -> Expr {
    udfs::tensor_tt_svd_reconstruct_udf().call(vec![tt])
}

#[must_use]
pub fn matrix_column_means(matrix: Expr) -> Expr {
    udfs::matrix_column_means_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_column_means_complex(matrix: Expr) -> Expr {
    udfs::matrix_column_means_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_center_columns(matrix: Expr) -> Expr {
    udfs::matrix_center_columns_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_center_columns_complex(matrix: Expr) -> Expr {
    udfs::matrix_center_columns_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_covariance(matrix: Expr) -> Expr { udfs::matrix_covariance_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_covariance_complex(matrix: Expr) -> Expr {
    udfs::matrix_covariance_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_correlation(matrix: Expr) -> Expr {
    udfs::matrix_correlation_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_correlation_complex(matrix: Expr) -> Expr {
    udfs::matrix_correlation_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_pca(matrix: Expr) -> Expr { udfs::matrix_pca_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_pca_complex(matrix: Expr) -> Expr {
    udfs::matrix_pca_complex_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_pca_transform(matrix: Expr, pca: Expr) -> Expr {
    udfs::matrix_pca_transform_udf().call(vec![matrix, pca])
}

#[must_use]
pub fn matrix_pca_transform_complex(matrix: Expr, pca: Expr) -> Expr {
    udfs::matrix_pca_transform_complex_udf().call(vec![matrix, pca])
}

#[must_use]
pub fn matrix_pca_inverse_transform(scores: Expr, pca: Expr) -> Expr {
    udfs::matrix_pca_inverse_transform_udf().call(vec![scores, pca])
}

#[must_use]
pub fn matrix_pca_inverse_transform_complex(scores: Expr, pca: Expr) -> Expr {
    udfs::matrix_pca_inverse_transform_complex_udf().call(vec![scores, pca])
}

#[must_use]
pub fn matrix_conjugate_gradient(
    matrix: Expr,
    rhs: Expr,
    tolerance: Expr,
    max_iterations: Expr,
) -> Expr {
    udfs::matrix_conjugate_gradient_udf().call(vec![matrix, rhs, tolerance, max_iterations])
}

#[must_use]
pub fn matrix_conjugate_gradient_complex(
    matrix: Expr,
    rhs: Expr,
    tolerance: Expr,
    max_iterations: Expr,
) -> Expr {
    udfs::matrix_conjugate_gradient_complex_udf().call(vec![matrix, rhs, tolerance, max_iterations])
}

#[must_use]
pub fn matrix_gmres(matrix: Expr, rhs: Expr, tolerance: Expr, max_iterations: Expr) -> Expr {
    udfs::matrix_gmres_udf().call(vec![matrix, rhs, tolerance, max_iterations])
}

#[must_use]
pub fn matrix_gmres_complex(
    matrix: Expr,
    rhs: Expr,
    tolerance: Expr,
    max_iterations: Expr,
) -> Expr {
    udfs::matrix_gmres_complex_udf().call(vec![matrix, rhs, tolerance, max_iterations])
}

#[must_use]
pub fn linear_regression(design: Expr, response: Expr, add_intercept: Expr) -> Expr {
    udfs::linear_regression_udf().call(vec![design, response, add_intercept])
}

#[must_use]
pub fn vector_covariance_agg(vectors: Expr) -> Expr {
    udafs::vector_covariance_agg_udaf().call(vec![vectors])
}

#[must_use]
pub fn vector_correlation_agg(vectors: Expr) -> Expr {
    udafs::vector_correlation_agg_udaf().call(vec![vectors])
}

#[must_use]
pub fn vector_pca_fit(vectors: Expr) -> Expr { udafs::vector_pca_fit_udaf().call(vec![vectors]) }

#[must_use]
pub fn linear_regression_fit(design: Expr, response: Expr, add_intercept: Expr) -> Expr {
    udafs::linear_regression_fit_udaf().call(vec![design, response, add_intercept])
}

#[cfg(test)]
mod tests {
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Expr;

    use super::{
        adam_complex, backtracking_line_search_complex, gradient, gradient_descent_complex,
        hessian, jacobian, jacobian_central, linear_regression, linear_regression_fit,
        make_csr_matrix_batch, make_matrix, make_tensor, make_variable_tensor, make_vector,
        matrix_balance_nonsymmetric, matrix_center_columns, matrix_center_columns_complex,
        matrix_cholesky, matrix_cholesky_inverse, matrix_cholesky_solve, matrix_column_means,
        matrix_column_means_complex, matrix_conjugate_gradient, matrix_conjugate_gradient_complex,
        matrix_correlation, matrix_correlation_complex, matrix_covariance,
        matrix_covariance_complex, matrix_determinant, matrix_eigen_generalized,
        matrix_eigen_nonsymmetric, matrix_eigen_nonsymmetric_bi, matrix_eigen_nonsymmetric_complex,
        matrix_eigen_symmetric, matrix_exp, matrix_exp_complex, matrix_exp_eigen,
        matrix_exp_eigen_complex, matrix_gmres, matrix_gmres_complex, matrix_gram_schmidt,
        matrix_gram_schmidt_classic, matrix_inverse, matrix_log_determinant, matrix_log_eigen,
        matrix_log_eigen_complex, matrix_log_svd, matrix_log_svd_complex, matrix_log_taylor,
        matrix_lu, matrix_lu_solve, matrix_matmat_complex, matrix_matmul, matrix_matvec,
        matrix_matvec_complex, matrix_pca, matrix_pca_complex, matrix_pca_inverse_transform,
        matrix_pca_inverse_transform_complex, matrix_pca_transform, matrix_pca_transform_complex,
        matrix_polar, matrix_polar_complex, matrix_power, matrix_power_complex, matrix_qr,
        matrix_qr_condition_number, matrix_qr_pivoted, matrix_qr_reconstruct, matrix_qr_reduced,
        matrix_qr_solve_least_squares, matrix_schur, matrix_schur_complex, matrix_sign,
        matrix_sign_complex, matrix_solve_lower, matrix_solve_lower_matrix, matrix_solve_sylvester,
        matrix_solve_sylvester_complex, matrix_solve_sylvester_mixed_complex,
        matrix_solve_sylvester_mixed_f64, matrix_solve_upper, matrix_solve_upper_matrix,
        matrix_svd, matrix_svd_condition_number, matrix_svd_null_space, matrix_svd_pseudo_inverse,
        matrix_svd_rank, matrix_svd_reconstruct, matrix_svd_truncated, matrix_svd_with_tolerance,
        momentum_descent_complex, sparse_apply_iluk_preconditioner,
        sparse_apply_ilut_preconditioner, sparse_apply_jacobi_preconditioner, sparse_iluk_factor,
        sparse_ilut_factor, sparse_jacobi_preconditioner, sparse_lu_factor, sparse_lu_solve,
        sparse_lu_solve_multiple_with_factorization, sparse_lu_solve_with_factorization,
        sparse_matmat_dense, sparse_matmat_sparse, sparse_matvec, sparse_transpose,
        tensor_batched_dot_last_axis, tensor_batched_matmul_last_two, tensor_contract_axes,
        tensor_cp_als_nd, tensor_cp_als_nd_reconstruct, tensor_cp_als3, tensor_cp_als3_reconstruct,
        tensor_hooi_nd, tensor_hosvd_nd, tensor_l2_norm_last_axis,
        tensor_l2_norm_last_axis_complex, tensor_normalize_last_axis,
        tensor_normalize_last_axis_complex, tensor_permute_axes, tensor_sum_last_axis,
        tensor_tt_add, tensor_tt_hadamard, tensor_tt_hadamard_round, tensor_tt_inner,
        tensor_tt_norm, tensor_tt_orthogonalize_left, tensor_tt_orthogonalize_right,
        tensor_tt_round, tensor_tt_svd, tensor_tt_svd_reconstruct, tensor_tucker_expand,
        tensor_tucker_project, tensor_variable_batched_dot_last_axis,
        tensor_variable_l2_norm_last_axis, tensor_variable_l2_norm_last_axis_complex,
        tensor_variable_normalize_last_axis, tensor_variable_normalize_last_axis_complex,
        tensor_variable_sum_last_axis, vector_correlation_agg, vector_cosine_distance,
        vector_cosine_similarity, vector_cosine_similarity_complex, vector_covariance_agg,
        vector_dot, vector_dot_hermitian, vector_l2_norm, vector_l2_norm_complex, vector_normalize,
        vector_normalize_complex, vector_pca_fit,
    };

    fn literal_i64(value: i64) -> Expr { Expr::Literal(ScalarValue::Int64(Some(value)), None) }

    fn assert_scalar_function(expr: Expr, expected_name: &str, expected_args: usize) {
        let Expr::ScalarFunction(function) = expr else {
            panic!("expected scalar function expression");
        };
        assert_eq!(function.name(), expected_name);
        assert_eq!(function.args.len(), expected_args);
    }

    fn assert_aggregate_function(expr: Expr, expected_name: &str, expected_args: usize) {
        let Expr::AggregateFunction(function) = expr else {
            panic!("expected aggregate function expression");
        };
        assert_eq!(function.func.name(), expected_name);
        assert_eq!(function.params.args.len(), expected_args);
    }

    #[test]
    fn constructor_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);
        let four = literal_i64(4);

        assert_scalar_function(make_vector(one.clone(), two.clone()), "make_vector", 2);
        assert_scalar_function(
            make_matrix(one.clone(), two.clone(), three.clone()),
            "make_matrix",
            3,
        );
        assert_scalar_function(
            make_tensor(one.clone(), vec![two.clone(), three.clone()]),
            "make_tensor",
            3,
        );
        assert_scalar_function(
            make_variable_tensor(one.clone(), two.clone(), three.clone()),
            "make_variable_tensor",
            3,
        );
        assert_scalar_function(
            make_csr_matrix_batch(one.clone(), two.clone(), three.clone(), four.clone()),
            "make_csr_matrix_batch",
            4,
        );
    }

    #[test]
    fn vector_and_matrix_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);

        assert_scalar_function(vector_l2_norm(one.clone()), "vector_l2_norm", 1);
        assert_scalar_function(vector_dot(one.clone(), two.clone()), "vector_dot", 2);
        assert_scalar_function(
            vector_cosine_similarity(one.clone(), two.clone()),
            "vector_cosine_similarity",
            2,
        );
        assert_scalar_function(
            vector_cosine_distance(one.clone(), two.clone()),
            "vector_cosine_distance",
            2,
        );
        assert_scalar_function(vector_normalize(one.clone()), "vector_normalize", 1);
        assert_scalar_function(
            vector_dot_hermitian(one.clone(), two.clone()),
            "vector_dot_hermitian",
            2,
        );
        assert_scalar_function(vector_l2_norm_complex(one.clone()), "vector_l2_norm_complex", 1);
        assert_scalar_function(
            vector_cosine_similarity_complex(one.clone(), two.clone()),
            "vector_cosine_similarity_complex",
            2,
        );
        assert_scalar_function(
            vector_normalize_complex(one.clone()),
            "vector_normalize_complex",
            1,
        );
        assert_scalar_function(
            matrix_matvec_complex(one.clone(), two.clone()),
            "matrix_matvec_complex",
            2,
        );
        assert_scalar_function(matrix_matvec(one.clone(), two.clone()), "matrix_matvec", 2);
        assert_scalar_function(
            matrix_matmat_complex(one.clone(), two.clone()),
            "matrix_matmat_complex",
            2,
        );
        assert_scalar_function(matrix_matmul(one.clone(), two.clone()), "matrix_matmul", 2);
        assert_scalar_function(
            matrix_solve_lower(one.clone(), two.clone()),
            "matrix_solve_lower",
            2,
        );
        assert_scalar_function(
            matrix_solve_upper(one.clone(), two.clone()),
            "matrix_solve_upper",
            2,
        );
        assert_scalar_function(
            matrix_solve_lower_matrix(one.clone(), two.clone()),
            "matrix_solve_lower_matrix",
            2,
        );
        assert_scalar_function(
            matrix_solve_upper_matrix(one.clone(), two.clone()),
            "matrix_solve_upper_matrix",
            2,
        );
        assert_scalar_function(
            matrix_solve_sylvester(one.clone(), two.clone(), one.clone()),
            "matrix_solve_sylvester",
            3,
        );
        assert_scalar_function(
            matrix_solve_sylvester_mixed_f64(one.clone(), two.clone(), one.clone()),
            "matrix_solve_sylvester_mixed_f64",
            3,
        );
        assert_scalar_function(
            matrix_solve_sylvester_complex(one.clone(), two.clone(), one.clone()),
            "matrix_solve_sylvester_complex",
            3,
        );
        assert_scalar_function(
            matrix_solve_sylvester_mixed_complex(one.clone(), two.clone(), one.clone()),
            "matrix_solve_sylvester_mixed_complex",
            3,
        );
        assert_scalar_function(
            jacobian(one.clone(), two.clone(), vec![three.clone()]),
            "jacobian",
            3,
        );
        assert_scalar_function(
            jacobian_central(one.clone(), two.clone(), vec![three.clone()]),
            "jacobian_central",
            3,
        );
        assert_scalar_function(
            gradient(one.clone(), two.clone(), vec![three.clone()]),
            "gradient",
            3,
        );
        assert_scalar_function(hessian(one, two, vec![three]), "hessian", 3);
    }

    #[test]
    fn aggregate_helpers_wrap_the_expected_udafs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);

        assert_aggregate_function(vector_covariance_agg(one.clone()), "vector_covariance_agg", 1);
        assert_aggregate_function(vector_correlation_agg(one.clone()), "vector_correlation_agg", 1);
        assert_aggregate_function(vector_pca_fit(one.clone()), "vector_pca_fit", 1);
        assert_aggregate_function(
            linear_regression_fit(one, two, three),
            "linear_regression_fit",
            3,
        );
    }

    #[test]
    fn decomposition_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);

        assert_scalar_function(matrix_lu(one.clone()), "matrix_lu", 1);
        assert_scalar_function(matrix_lu_solve(one.clone(), two.clone()), "matrix_lu_solve", 2);
        assert_scalar_function(
            matrix_cholesky_solve(one.clone(), two.clone()),
            "matrix_cholesky_solve",
            2,
        );
        assert_scalar_function(matrix_inverse(one.clone()), "matrix_inverse", 1);
        assert_scalar_function(matrix_determinant(one.clone()), "matrix_determinant", 1);
        assert_scalar_function(matrix_log_determinant(one.clone()), "matrix_log_determinant", 1);
        assert_scalar_function(matrix_cholesky(one.clone()), "matrix_cholesky", 1);
        assert_scalar_function(matrix_cholesky_inverse(one.clone()), "matrix_cholesky_inverse", 1);
        assert_scalar_function(matrix_qr(one.clone()), "matrix_qr", 1);
        assert_scalar_function(matrix_qr_reduced(one.clone()), "matrix_qr_reduced", 1);
        assert_scalar_function(matrix_qr_pivoted(one.clone()), "matrix_qr_pivoted", 1);
        assert_scalar_function(
            matrix_qr_solve_least_squares(one.clone(), two.clone()),
            "matrix_qr_solve_least_squares",
            2,
        );
        assert_scalar_function(
            matrix_qr_condition_number(one.clone()),
            "matrix_qr_condition_number",
            1,
        );
        assert_scalar_function(matrix_qr_reconstruct(one.clone()), "matrix_qr_reconstruct", 1);
        assert_scalar_function(matrix_svd(one.clone()), "matrix_svd", 1);
        assert_scalar_function(
            matrix_svd_truncated(one.clone(), two.clone()),
            "matrix_svd_truncated",
            2,
        );
        assert_scalar_function(
            matrix_svd_with_tolerance(one.clone(), two.clone()),
            "matrix_svd_with_tolerance",
            2,
        );
        assert_scalar_function(matrix_svd_null_space(one.clone()), "matrix_svd_null_space", 1);
        assert_scalar_function(
            matrix_svd_pseudo_inverse(one.clone()),
            "matrix_svd_pseudo_inverse",
            1,
        );
        assert_scalar_function(
            matrix_svd_condition_number(one.clone()),
            "matrix_svd_condition_number",
            1,
        );
        assert_scalar_function(matrix_svd_rank(one.clone()), "matrix_svd_rank", 1);
        assert_scalar_function(matrix_svd_reconstruct(one.clone()), "matrix_svd_reconstruct", 1);
        assert_scalar_function(matrix_eigen_symmetric(one.clone()), "matrix_eigen_symmetric", 1);
        assert_scalar_function(
            matrix_eigen_generalized(one.clone(), two.clone()),
            "matrix_eigen_generalized",
            2,
        );
        assert_scalar_function(
            matrix_eigen_nonsymmetric(one.clone()),
            "matrix_eigen_nonsymmetric",
            1,
        );
        assert_scalar_function(
            matrix_eigen_nonsymmetric_bi(one.clone()),
            "matrix_eigen_nonsymmetric_bi",
            1,
        );
        assert_scalar_function(
            matrix_eigen_nonsymmetric_complex(one.clone()),
            "matrix_eigen_nonsymmetric_complex",
            1,
        );
        assert_scalar_function(
            matrix_balance_nonsymmetric(one.clone()),
            "matrix_balance_nonsymmetric",
            1,
        );
        assert_scalar_function(matrix_schur(one.clone()), "matrix_schur", 1);
        assert_scalar_function(matrix_schur_complex(one.clone()), "matrix_schur_complex", 1);
        assert_scalar_function(matrix_polar(one.clone()), "matrix_polar", 1);
        assert_scalar_function(matrix_polar_complex(one.clone()), "matrix_polar_complex", 1);
        assert_scalar_function(matrix_gram_schmidt(one.clone()), "matrix_gram_schmidt", 1);
        assert_scalar_function(
            matrix_gram_schmidt_classic(one.clone()),
            "matrix_gram_schmidt_classic",
            1,
        );
    }

    #[test]
    fn matrix_function_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);

        assert_scalar_function(
            matrix_exp(one.clone(), two.clone(), three.clone()),
            "matrix_exp",
            3,
        );
        assert_scalar_function(
            matrix_exp_complex(one.clone(), two.clone(), three.clone()),
            "matrix_exp_complex",
            3,
        );
        assert_scalar_function(matrix_exp_eigen(one.clone()), "matrix_exp_eigen", 1);
        assert_scalar_function(
            matrix_exp_eigen_complex(one.clone()),
            "matrix_exp_eigen_complex",
            1,
        );
        assert_scalar_function(
            matrix_log_taylor(one.clone(), two.clone(), three.clone()),
            "matrix_log_taylor",
            3,
        );
        assert_scalar_function(matrix_log_eigen(one.clone()), "matrix_log_eigen", 1);
        assert_scalar_function(
            matrix_log_eigen_complex(one.clone()),
            "matrix_log_eigen_complex",
            1,
        );
        assert_scalar_function(matrix_log_svd(one.clone()), "matrix_log_svd", 1);
        assert_scalar_function(matrix_log_svd_complex(one.clone()), "matrix_log_svd_complex", 1);
        assert_scalar_function(matrix_power(one.clone(), two.clone()), "matrix_power", 2);
        assert_scalar_function(
            matrix_power_complex(one.clone(), two.clone()),
            "matrix_power_complex",
            2,
        );
        assert_scalar_function(matrix_sign(one.clone()), "matrix_sign", 1);
        assert_scalar_function(matrix_sign_complex(one.clone()), "matrix_sign_complex", 1);
    }

    #[test]
    fn sparse_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);

        assert_scalar_function(sparse_matvec(one.clone(), two.clone()), "sparse_matvec", 2);
        assert_scalar_function(sparse_lu_solve(one.clone(), two.clone()), "sparse_lu_solve", 2);
        assert_scalar_function(sparse_lu_factor(one.clone()), "sparse_lu_factor", 1);
        assert_scalar_function(
            sparse_lu_solve_with_factorization(one.clone(), two.clone(), three.clone()),
            "sparse_lu_solve_with_factorization",
            3,
        );
        assert_scalar_function(
            sparse_lu_solve_multiple_with_factorization(one.clone(), two.clone(), three.clone()),
            "sparse_lu_solve_multiple_with_factorization",
            3,
        );
        assert_scalar_function(
            sparse_jacobi_preconditioner(one.clone()),
            "sparse_jacobi_preconditioner",
            1,
        );
        assert_scalar_function(
            sparse_apply_jacobi_preconditioner(one.clone(), two.clone()),
            "sparse_apply_jacobi_preconditioner",
            2,
        );
        assert_scalar_function(
            sparse_ilut_factor(one.clone(), two.clone(), three.clone()),
            "sparse_ilut_factor",
            3,
        );
        assert_scalar_function(
            sparse_iluk_factor(one.clone(), two.clone()),
            "sparse_iluk_factor",
            2,
        );
        assert_scalar_function(
            sparse_apply_ilut_preconditioner(one.clone(), two.clone()),
            "sparse_apply_ilut_preconditioner",
            2,
        );
        assert_scalar_function(
            sparse_apply_iluk_preconditioner(one.clone(), two.clone()),
            "sparse_apply_iluk_preconditioner",
            2,
        );
        assert_scalar_function(
            sparse_matmat_dense(one.clone(), two.clone()),
            "sparse_matmat_dense",
            2,
        );
        assert_scalar_function(sparse_transpose(one.clone()), "sparse_transpose", 1);
        assert_scalar_function(
            sparse_matmat_sparse(one.clone(), two.clone()),
            "sparse_matmat_sparse",
            2,
        );
    }

    #[test]
    fn tensor_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);

        assert_scalar_function(tensor_sum_last_axis(one.clone()), "tensor_sum_last_axis", 1);
        assert_scalar_function(
            tensor_l2_norm_last_axis(one.clone()),
            "tensor_l2_norm_last_axis",
            1,
        );
        assert_scalar_function(
            tensor_l2_norm_last_axis_complex(one.clone()),
            "tensor_l2_norm_last_axis_complex",
            1,
        );
        assert_scalar_function(
            tensor_normalize_last_axis(one.clone()),
            "tensor_normalize_last_axis",
            1,
        );
        assert_scalar_function(
            tensor_normalize_last_axis_complex(one.clone()),
            "tensor_normalize_last_axis_complex",
            1,
        );
        assert_scalar_function(
            tensor_batched_dot_last_axis(one.clone(), two.clone()),
            "tensor_batched_dot_last_axis",
            2,
        );
        assert_scalar_function(
            tensor_batched_matmul_last_two(one.clone(), two.clone()),
            "tensor_batched_matmul_last_two",
            2,
        );
        assert_scalar_function(
            tensor_permute_axes(one.clone(), vec![two.clone(), one.clone()]),
            "tensor_permute_axes",
            3,
        );
        assert_scalar_function(
            tensor_contract_axes(one.clone(), two.clone(), vec![three.clone(), one.clone()]),
            "tensor_contract_axes",
            4,
        );
        assert_scalar_function(
            tensor_variable_sum_last_axis(one.clone()),
            "tensor_variable_sum_last_axis",
            1,
        );
        assert_scalar_function(
            tensor_variable_l2_norm_last_axis(one.clone()),
            "tensor_variable_l2_norm_last_axis",
            1,
        );
        assert_scalar_function(
            tensor_variable_l2_norm_last_axis_complex(one.clone()),
            "tensor_variable_l2_norm_last_axis_complex",
            1,
        );
        assert_scalar_function(
            tensor_variable_normalize_last_axis(one.clone()),
            "tensor_variable_normalize_last_axis",
            1,
        );
        assert_scalar_function(
            tensor_variable_normalize_last_axis_complex(one.clone()),
            "tensor_variable_normalize_last_axis_complex",
            1,
        );
        assert_scalar_function(
            tensor_variable_batched_dot_last_axis(one.clone(), two.clone()),
            "tensor_variable_batched_dot_last_axis",
            2,
        );
    }

    #[test]
    fn tensor_decomposition_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);
        let four = literal_i64(4);

        assert_scalar_function(
            tensor_cp_als3(one.clone(), two.clone(), vec![three.clone(), four.clone()]),
            "tensor_cp_als3",
            4,
        );
        assert_scalar_function(
            tensor_cp_als3_reconstruct(one.clone()),
            "tensor_cp_als3_reconstruct",
            1,
        );
        assert_scalar_function(
            tensor_cp_als_nd(one.clone(), two.clone(), vec![three.clone(), four.clone()]),
            "tensor_cp_als_nd",
            4,
        );
        assert_scalar_function(
            tensor_cp_als_nd_reconstruct(one.clone()),
            "tensor_cp_als_nd_reconstruct",
            1,
        );
        assert_scalar_function(tensor_hosvd_nd(one.clone(), two.clone()), "tensor_hosvd_nd", 2);
        assert_scalar_function(
            tensor_hooi_nd(one.clone(), two.clone(), vec![three.clone(), four.clone()]),
            "tensor_hooi_nd",
            4,
        );
        assert_scalar_function(
            tensor_tucker_project(one.clone(), two.clone()),
            "tensor_tucker_project",
            2,
        );
        assert_scalar_function(tensor_tucker_expand(one.clone()), "tensor_tucker_expand", 1);
        assert_scalar_function(
            tensor_tt_svd(one.clone(), vec![two.clone(), three.clone()]),
            "tensor_tt_svd",
            3,
        );
        assert_scalar_function(
            tensor_tt_orthogonalize_left(one.clone()),
            "tensor_tt_orthogonalize_left",
            1,
        );
        assert_scalar_function(
            tensor_tt_orthogonalize_right(one.clone()),
            "tensor_tt_orthogonalize_right",
            1,
        );
        assert_scalar_function(
            tensor_tt_round(one.clone(), vec![two.clone(), three.clone()]),
            "tensor_tt_round",
            3,
        );
        assert_scalar_function(tensor_tt_inner(one.clone(), two.clone()), "tensor_tt_inner", 2);
        assert_scalar_function(tensor_tt_norm(one.clone()), "tensor_tt_norm", 1);
        assert_scalar_function(tensor_tt_add(one.clone(), two.clone()), "tensor_tt_add", 2);
        assert_scalar_function(
            tensor_tt_hadamard(one.clone(), two.clone()),
            "tensor_tt_hadamard",
            2,
        );
        assert_scalar_function(
            tensor_tt_hadamard_round(one.clone(), two.clone(), vec![three.clone(), four]),
            "tensor_tt_hadamard_round",
            4,
        );
        assert_scalar_function(tensor_tt_svd_reconstruct(one), "tensor_tt_svd_reconstruct", 1);
    }

    #[test]
    fn matrix_stats_and_pca_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);

        assert_scalar_function(matrix_column_means(one.clone()), "matrix_column_means", 1);
        assert_scalar_function(
            matrix_column_means_complex(one.clone()),
            "matrix_column_means_complex",
            1,
        );
        assert_scalar_function(matrix_center_columns(one.clone()), "matrix_center_columns", 1);
        assert_scalar_function(
            matrix_center_columns_complex(one.clone()),
            "matrix_center_columns_complex",
            1,
        );
        assert_scalar_function(matrix_covariance(one.clone()), "matrix_covariance", 1);
        assert_scalar_function(
            matrix_covariance_complex(one.clone()),
            "matrix_covariance_complex",
            1,
        );
        assert_scalar_function(matrix_correlation(one.clone()), "matrix_correlation", 1);
        assert_scalar_function(
            matrix_correlation_complex(one.clone()),
            "matrix_correlation_complex",
            1,
        );
        assert_scalar_function(matrix_pca(one.clone()), "matrix_pca", 1);
        assert_scalar_function(matrix_pca_complex(one.clone()), "matrix_pca_complex", 1);
        assert_scalar_function(
            matrix_pca_transform(one.clone(), two.clone()),
            "matrix_pca_transform",
            2,
        );
        assert_scalar_function(
            matrix_pca_transform_complex(one.clone(), two.clone()),
            "matrix_pca_transform_complex",
            2,
        );
        assert_scalar_function(
            matrix_pca_inverse_transform(one.clone(), two.clone()),
            "matrix_pca_inverse_transform",
            2,
        );
        assert_scalar_function(
            matrix_pca_inverse_transform_complex(one.clone(), two.clone()),
            "matrix_pca_inverse_transform_complex",
            2,
        );
    }

    #[test]
    fn iterative_and_optimization_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);
        let four = literal_i64(4);

        assert_scalar_function(
            matrix_conjugate_gradient(one.clone(), two.clone(), three.clone(), four.clone()),
            "matrix_conjugate_gradient",
            4,
        );
        assert_scalar_function(
            matrix_conjugate_gradient_complex(
                one.clone(),
                two.clone(),
                three.clone(),
                four.clone(),
            ),
            "matrix_conjugate_gradient_complex",
            4,
        );
        assert_scalar_function(
            matrix_gmres(one.clone(), two.clone(), three.clone(), four.clone()),
            "matrix_gmres",
            4,
        );
        assert_scalar_function(
            matrix_gmres_complex(one.clone(), two.clone(), three.clone(), literal_i64(4)),
            "matrix_gmres_complex",
            4,
        );
        assert_scalar_function(
            backtracking_line_search_complex(one.clone(), two.clone(), three.clone(), vec![
                four.clone(),
            ]),
            "backtracking_line_search_complex",
            4,
        );
        assert_scalar_function(
            gradient_descent_complex(one.clone(), two.clone(), vec![three.clone(), four.clone()]),
            "gradient_descent_complex",
            4,
        );
        assert_scalar_function(
            adam_complex(one.clone(), two.clone(), vec![
                three.clone(),
                four.clone(),
                literal_i64(5),
            ]),
            "adam_complex",
            5,
        );
        assert_scalar_function(
            momentum_descent_complex(one.clone(), two.clone(), vec![three.clone(), four]),
            "momentum_descent_complex",
            4,
        );
    }

    #[test]
    fn regression_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);

        assert_scalar_function(linear_regression(one, two, three), "linear_regression", 3);
    }
}
