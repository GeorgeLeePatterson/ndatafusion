//! Expression builders for the `ndatafusion` UDF catalog.
//!
//! Each function in this module returns a `datafusion::logical_expr::Expr` that calls a registered
//! `ndatafusion` scalar UDF. The helper names match the SQL function names.
//!
//! For the complete catalog and notes on input and output contracts, see `CATALOG.md` in the
//! repository root.

use datafusion::logical_expr::Expr;

use crate::udfs;

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
pub fn matrix_matvec(matrix: Expr, vector: Expr) -> Expr {
    udfs::matrix_matvec_udf().call(vec![matrix, vector])
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
pub fn matrix_balance_nonsymmetric(matrix: Expr) -> Expr {
    udfs::matrix_balance_nonsymmetric_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_schur(matrix: Expr) -> Expr { udfs::matrix_schur_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_polar(matrix: Expr) -> Expr { udfs::matrix_polar_udf().call(vec![matrix]) }

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
pub fn matrix_log_taylor(matrix: Expr, max_terms: Expr, tolerance: Expr) -> Expr {
    udfs::matrix_log_taylor_udf().call(vec![matrix, max_terms, tolerance])
}

#[must_use]
pub fn matrix_log_eigen(matrix: Expr) -> Expr { udfs::matrix_log_eigen_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_log_svd(matrix: Expr) -> Expr { udfs::matrix_log_svd_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_power(matrix: Expr, power: Expr) -> Expr {
    udfs::matrix_power_udf().call(vec![matrix, power])
}

#[must_use]
pub fn matrix_sign(matrix: Expr) -> Expr { udfs::matrix_sign_udf().call(vec![matrix]) }

#[must_use]
pub fn sparse_matvec(matrices: Expr, vectors: Expr) -> Expr {
    udfs::sparse_matvec_udf().call(vec![matrices, vectors])
}

#[must_use]
pub fn sparse_lu_solve(matrices: Expr, rhs: Expr) -> Expr {
    udfs::sparse_lu_solve_udf().call(vec![matrices, rhs])
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
pub fn tensor_normalize_last_axis(tensor: Expr) -> Expr {
    udfs::tensor_normalize_last_axis_udf().call(vec![tensor])
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
pub fn tensor_variable_normalize_last_axis(tensor: Expr) -> Expr {
    udfs::tensor_variable_normalize_last_axis_udf().call(vec![tensor])
}

#[must_use]
pub fn tensor_variable_batched_dot_last_axis(left: Expr, right: Expr) -> Expr {
    udfs::tensor_variable_batched_dot_last_axis_udf().call(vec![left, right])
}

#[must_use]
pub fn matrix_column_means(matrix: Expr) -> Expr {
    udfs::matrix_column_means_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_center_columns(matrix: Expr) -> Expr {
    udfs::matrix_center_columns_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_covariance(matrix: Expr) -> Expr { udfs::matrix_covariance_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_correlation(matrix: Expr) -> Expr {
    udfs::matrix_correlation_udf().call(vec![matrix])
}

#[must_use]
pub fn matrix_pca(matrix: Expr) -> Expr { udfs::matrix_pca_udf().call(vec![matrix]) }

#[must_use]
pub fn matrix_pca_transform(matrix: Expr, pca: Expr) -> Expr {
    udfs::matrix_pca_transform_udf().call(vec![matrix, pca])
}

#[must_use]
pub fn matrix_pca_inverse_transform(scores: Expr, pca: Expr) -> Expr {
    udfs::matrix_pca_inverse_transform_udf().call(vec![scores, pca])
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
pub fn matrix_gmres(matrix: Expr, rhs: Expr, tolerance: Expr, max_iterations: Expr) -> Expr {
    udfs::matrix_gmres_udf().call(vec![matrix, rhs, tolerance, max_iterations])
}

#[must_use]
pub fn linear_regression(design: Expr, response: Expr, add_intercept: Expr) -> Expr {
    udfs::linear_regression_udf().call(vec![design, response, add_intercept])
}

#[cfg(test)]
mod tests {
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Expr;

    use super::{
        linear_regression, make_csr_matrix_batch, make_matrix, make_tensor, make_variable_tensor,
        make_vector, matrix_balance_nonsymmetric, matrix_center_columns, matrix_cholesky,
        matrix_cholesky_inverse, matrix_cholesky_solve, matrix_column_means,
        matrix_conjugate_gradient, matrix_correlation, matrix_covariance, matrix_determinant,
        matrix_eigen_generalized, matrix_eigen_symmetric, matrix_exp, matrix_exp_eigen,
        matrix_gmres, matrix_gram_schmidt, matrix_gram_schmidt_classic, matrix_inverse,
        matrix_log_determinant, matrix_log_eigen, matrix_log_svd, matrix_log_taylor, matrix_lu,
        matrix_lu_solve, matrix_matmul, matrix_matvec, matrix_pca, matrix_pca_inverse_transform,
        matrix_pca_transform, matrix_polar, matrix_power, matrix_qr, matrix_qr_condition_number,
        matrix_qr_pivoted, matrix_qr_reconstruct, matrix_qr_reduced, matrix_qr_solve_least_squares,
        matrix_schur, matrix_sign, matrix_solve_lower, matrix_solve_lower_matrix,
        matrix_solve_upper, matrix_solve_upper_matrix, matrix_svd, matrix_svd_condition_number,
        matrix_svd_null_space, matrix_svd_pseudo_inverse, matrix_svd_rank, matrix_svd_reconstruct,
        matrix_svd_truncated, matrix_svd_with_tolerance, sparse_lu_solve, sparse_matmat_dense,
        sparse_matmat_sparse, sparse_matvec, sparse_transpose, tensor_batched_dot_last_axis,
        tensor_batched_matmul_last_two, tensor_contract_axes, tensor_l2_norm_last_axis,
        tensor_normalize_last_axis, tensor_permute_axes, tensor_sum_last_axis,
        tensor_variable_batched_dot_last_axis, tensor_variable_l2_norm_last_axis,
        tensor_variable_normalize_last_axis, tensor_variable_sum_last_axis, vector_cosine_distance,
        vector_cosine_similarity, vector_dot, vector_l2_norm, vector_normalize,
    };

    fn literal_i64(value: i64) -> Expr { Expr::Literal(ScalarValue::Int64(Some(value)), None) }

    fn assert_scalar_function(expr: Expr, expected_name: &str, expected_args: usize) {
        let Expr::ScalarFunction(function) = expr else {
            panic!("expected scalar function expression");
        };
        assert_eq!(function.name(), expected_name);
        assert_eq!(function.args.len(), expected_args);
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
        assert_scalar_function(matrix_matvec(one.clone(), two.clone()), "matrix_matvec", 2);
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
    }

    #[test]
    fn decomposition_and_matrix_function_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);

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
            matrix_balance_nonsymmetric(one.clone()),
            "matrix_balance_nonsymmetric",
            1,
        );
        assert_scalar_function(matrix_schur(one.clone()), "matrix_schur", 1);
        assert_scalar_function(matrix_polar(one.clone()), "matrix_polar", 1);
        assert_scalar_function(matrix_gram_schmidt(one.clone()), "matrix_gram_schmidt", 1);
        assert_scalar_function(
            matrix_gram_schmidt_classic(one.clone()),
            "matrix_gram_schmidt_classic",
            1,
        );
        assert_scalar_function(
            matrix_exp(one.clone(), two.clone(), three.clone()),
            "matrix_exp",
            3,
        );
        assert_scalar_function(matrix_exp_eigen(one.clone()), "matrix_exp_eigen", 1);
        assert_scalar_function(
            matrix_log_taylor(one.clone(), two.clone(), three.clone()),
            "matrix_log_taylor",
            3,
        );
        assert_scalar_function(matrix_log_eigen(one.clone()), "matrix_log_eigen", 1);
        assert_scalar_function(matrix_log_svd(one.clone()), "matrix_log_svd", 1);
        assert_scalar_function(matrix_power(one.clone(), two.clone()), "matrix_power", 2);
        assert_scalar_function(matrix_sign(one.clone()), "matrix_sign", 1);
    }

    #[test]
    fn sparse_and_tensor_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);

        assert_scalar_function(sparse_matvec(one.clone(), two.clone()), "sparse_matvec", 2);
        assert_scalar_function(sparse_lu_solve(one.clone(), two.clone()), "sparse_lu_solve", 2);
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
        assert_scalar_function(tensor_sum_last_axis(one.clone()), "tensor_sum_last_axis", 1);
        assert_scalar_function(
            tensor_l2_norm_last_axis(one.clone()),
            "tensor_l2_norm_last_axis",
            1,
        );
        assert_scalar_function(
            tensor_normalize_last_axis(one.clone()),
            "tensor_normalize_last_axis",
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
            tensor_variable_normalize_last_axis(one.clone()),
            "tensor_variable_normalize_last_axis",
            1,
        );
        assert_scalar_function(
            tensor_variable_batched_dot_last_axis(one.clone(), two.clone()),
            "tensor_variable_batched_dot_last_axis",
            2,
        );
    }

    #[test]
    fn ml_helpers_wrap_the_expected_udfs() {
        let one = literal_i64(1);
        let two = literal_i64(2);
        let three = literal_i64(3);
        let four = literal_i64(4);

        assert_scalar_function(matrix_column_means(one.clone()), "matrix_column_means", 1);
        assert_scalar_function(matrix_center_columns(one.clone()), "matrix_center_columns", 1);
        assert_scalar_function(matrix_covariance(one.clone()), "matrix_covariance", 1);
        assert_scalar_function(matrix_correlation(one.clone()), "matrix_correlation", 1);
        assert_scalar_function(matrix_pca(one.clone()), "matrix_pca", 1);
        assert_scalar_function(
            matrix_pca_transform(one.clone(), two.clone()),
            "matrix_pca_transform",
            2,
        );
        assert_scalar_function(
            matrix_pca_inverse_transform(one.clone(), two.clone()),
            "matrix_pca_inverse_transform",
            2,
        );
        assert_scalar_function(
            matrix_conjugate_gradient(one.clone(), two.clone(), three.clone(), four.clone()),
            "matrix_conjugate_gradient",
            4,
        );
        assert_scalar_function(
            matrix_gmres(one.clone(), two.clone(), three.clone(), four),
            "matrix_gmres",
            4,
        );
        assert_scalar_function(linear_regression(one, two, three), "linear_regression", 3);
    }
}
