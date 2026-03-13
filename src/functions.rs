//! Direct Rust call sites layered over the registered UDF catalog.
//!
//! Domain-specific helpers land here as the first concrete UDF slices are implemented.

use datafusion::logical_expr::Expr;

use crate::udfs;

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
pub fn matrix_matmul(left: Expr, right: Expr) -> Expr {
    udfs::matrix_matmul_udf().call(vec![left, right])
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
pub fn matrix_svd(matrix: Expr) -> Expr { udfs::matrix_svd_udf().call(vec![matrix]) }

#[must_use]
pub fn sparse_matvec(matrices: Expr, vectors: Expr) -> Expr {
    udfs::sparse_matvec_udf().call(vec![matrices, vectors])
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
pub fn linear_regression(design: Expr, response: Expr, add_intercept: Expr) -> Expr {
    udfs::linear_regression_udf().call(vec![design, response, add_intercept])
}

#[cfg(test)]
mod tests {
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Expr;

    use super::{
        linear_regression, matrix_center_columns, matrix_cholesky, matrix_cholesky_inverse,
        matrix_cholesky_solve, matrix_column_means, matrix_correlation, matrix_covariance,
        matrix_determinant, matrix_inverse, matrix_log_determinant, matrix_lu, matrix_lu_solve,
        matrix_matmul, matrix_pca, matrix_qr, matrix_svd, sparse_matmat_dense,
        sparse_matmat_sparse, sparse_matvec, sparse_transpose, tensor_batched_dot_last_axis,
        tensor_batched_matmul_last_two, tensor_l2_norm_last_axis, tensor_normalize_last_axis,
        tensor_sum_last_axis, tensor_variable_batched_dot_last_axis,
        tensor_variable_l2_norm_last_axis, tensor_variable_normalize_last_axis,
        tensor_variable_sum_last_axis, vector_cosine_distance, vector_cosine_similarity,
        vector_dot, vector_l2_norm, vector_normalize,
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
        assert_scalar_function(matrix_matmul(one.clone(), two.clone()), "matrix_matmul", 2);
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
        assert_scalar_function(matrix_svd(one.clone()), "matrix_svd", 1);
        assert_scalar_function(sparse_matvec(one.clone(), two.clone()), "sparse_matvec", 2);
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
        assert_scalar_function(matrix_column_means(one.clone()), "matrix_column_means", 1);
        assert_scalar_function(matrix_center_columns(one.clone()), "matrix_center_columns", 1);
        assert_scalar_function(matrix_covariance(one.clone()), "matrix_covariance", 1);
        assert_scalar_function(matrix_correlation(one.clone()), "matrix_correlation", 1);
        assert_scalar_function(matrix_pca(one.clone()), "matrix_pca", 1);
        assert_scalar_function(linear_regression(one, two, three), "linear_regression", 3);
    }
}
