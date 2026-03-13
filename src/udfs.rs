use std::sync::Arc;

use datafusion::logical_expr::ScalarUDF;

pub use crate::udf::constructors::{
    make_csr_matrix_batch_udf, make_matrix_udf, make_tensor_udf, make_variable_tensor_udf,
    make_vector_udf,
};
pub use crate::udf::decomposition::{
    matrix_cholesky_inverse_udf, matrix_cholesky_udf, matrix_determinant_udf, matrix_inverse_udf,
    matrix_log_determinant_udf, matrix_lu_udf, matrix_qr_udf, matrix_svd_udf,
};
pub use crate::udf::matrix::{matrix_cholesky_solve_udf, matrix_lu_solve_udf, matrix_matmul_udf};
pub use crate::udf::ml::{
    linear_regression_udf, matrix_center_columns_udf, matrix_column_means_udf,
    matrix_correlation_udf, matrix_covariance_udf, matrix_pca_udf,
};
pub use crate::udf::sparse::{
    sparse_matmat_dense_udf, sparse_matmat_sparse_udf, sparse_matvec_udf, sparse_transpose_udf,
};
pub use crate::udf::tensor::{
    tensor_batched_dot_last_axis_udf, tensor_batched_matmul_last_two_udf,
    tensor_l2_norm_last_axis_udf, tensor_normalize_last_axis_udf, tensor_sum_last_axis_udf,
    tensor_variable_batched_dot_last_axis_udf, tensor_variable_l2_norm_last_axis_udf,
    tensor_variable_normalize_last_axis_udf, tensor_variable_sum_last_axis_udf,
};
pub use crate::udf::vector::{
    vector_cosine_distance_udf, vector_cosine_similarity_udf, vector_dot_udf, vector_l2_norm_udf,
    vector_normalize_udf,
};

/// Return all currently implemented `ndatafusion` scalar UDFs.
#[must_use]
pub fn all_default_functions() -> Vec<Arc<ScalarUDF>> {
    vec![
        make_vector_udf(),
        make_matrix_udf(),
        make_tensor_udf(),
        make_variable_tensor_udf(),
        make_csr_matrix_batch_udf(),
        vector_l2_norm_udf(),
        vector_dot_udf(),
        vector_cosine_similarity_udf(),
        vector_cosine_distance_udf(),
        vector_normalize_udf(),
        matrix_matmul_udf(),
        matrix_lu_udf(),
        matrix_lu_solve_udf(),
        matrix_cholesky_solve_udf(),
        matrix_inverse_udf(),
        matrix_determinant_udf(),
        matrix_log_determinant_udf(),
        matrix_cholesky_udf(),
        matrix_cholesky_inverse_udf(),
        matrix_qr_udf(),
        matrix_svd_udf(),
        sparse_matvec_udf(),
        sparse_matmat_dense_udf(),
        sparse_transpose_udf(),
        sparse_matmat_sparse_udf(),
        tensor_sum_last_axis_udf(),
        tensor_l2_norm_last_axis_udf(),
        tensor_normalize_last_axis_udf(),
        tensor_batched_dot_last_axis_udf(),
        tensor_batched_matmul_last_two_udf(),
        tensor_variable_sum_last_axis_udf(),
        tensor_variable_l2_norm_last_axis_udf(),
        tensor_variable_normalize_last_axis_udf(),
        tensor_variable_batched_dot_last_axis_udf(),
        matrix_column_means_udf(),
        matrix_center_columns_udf(),
        matrix_covariance_udf(),
        matrix_correlation_udf(),
        matrix_pca_udf(),
        linear_regression_udf(),
    ]
}

#[cfg(test)]
mod tests {
    use super::all_default_functions;

    #[test]
    fn default_udf_catalog_matches_current_surface() {
        assert_eq!(all_default_functions().len(), 40);
    }
}
