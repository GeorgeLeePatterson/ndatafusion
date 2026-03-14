use std::sync::Arc;

use datafusion::logical_expr::ScalarUDF;

pub use crate::udf::constructors::{
    make_csr_matrix_batch_udf, make_matrix_udf, make_tensor_udf, make_variable_tensor_udf,
    make_vector_udf,
};
pub use crate::udf::decomposition::{
    matrix_cholesky_inverse_udf, matrix_cholesky_udf, matrix_determinant_udf,
    matrix_eigen_generalized_udf, matrix_eigen_symmetric_udf, matrix_gram_schmidt_classic_udf,
    matrix_gram_schmidt_udf, matrix_inverse_udf, matrix_log_determinant_udf, matrix_lu_udf,
    matrix_polar_udf, matrix_qr_condition_number_udf, matrix_qr_pivoted_udf, matrix_qr_reduced_udf,
    matrix_qr_solve_least_squares_udf, matrix_qr_udf, matrix_schur_udf,
    matrix_svd_condition_number_udf, matrix_svd_null_space_udf, matrix_svd_pseudo_inverse_udf,
    matrix_svd_rank_udf, matrix_svd_truncated_udf, matrix_svd_udf, matrix_svd_with_tolerance_udf,
};
pub use crate::udf::iterative::{matrix_conjugate_gradient_udf, matrix_gmres_udf};
pub use crate::udf::matrix::{
    matrix_cholesky_solve_udf, matrix_lu_solve_udf, matrix_matmul_udf, matrix_matvec_udf,
};
pub use crate::udf::matrix_functions::{
    matrix_exp_eigen_udf, matrix_exp_udf, matrix_log_eigen_udf, matrix_log_svd_udf,
    matrix_log_taylor_udf, matrix_power_udf, matrix_sign_udf,
};
pub use crate::udf::ml::{
    linear_regression_udf, matrix_center_columns_udf, matrix_column_means_udf,
    matrix_correlation_udf, matrix_covariance_udf, matrix_pca_udf,
};
pub use crate::udf::sparse::{
    sparse_matmat_dense_udf, sparse_matmat_sparse_udf, sparse_matvec_udf, sparse_transpose_udf,
};
pub use crate::udf::tensor::{
    tensor_batched_dot_last_axis_udf, tensor_batched_matmul_last_two_udf, tensor_contract_axes_udf,
    tensor_l2_norm_last_axis_udf, tensor_normalize_last_axis_udf, tensor_permute_axes_udf,
    tensor_sum_last_axis_udf, tensor_variable_batched_dot_last_axis_udf,
    tensor_variable_l2_norm_last_axis_udf, tensor_variable_normalize_last_axis_udf,
    tensor_variable_sum_last_axis_udf,
};
pub use crate::udf::triangular::{
    matrix_solve_lower_matrix_udf, matrix_solve_lower_udf, matrix_solve_upper_matrix_udf,
    matrix_solve_upper_udf,
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
        matrix_matvec_udf(),
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
        matrix_qr_reduced_udf(),
        matrix_qr_pivoted_udf(),
        matrix_qr_solve_least_squares_udf(),
        matrix_qr_condition_number_udf(),
        matrix_svd_udf(),
        matrix_svd_truncated_udf(),
        matrix_svd_with_tolerance_udf(),
        matrix_svd_null_space_udf(),
        matrix_svd_pseudo_inverse_udf(),
        matrix_svd_condition_number_udf(),
        matrix_svd_rank_udf(),
        matrix_eigen_symmetric_udf(),
        matrix_eigen_generalized_udf(),
        matrix_schur_udf(),
        matrix_polar_udf(),
        matrix_gram_schmidt_udf(),
        matrix_gram_schmidt_classic_udf(),
        matrix_conjugate_gradient_udf(),
        matrix_gmres_udf(),
        matrix_solve_lower_udf(),
        matrix_solve_upper_udf(),
        matrix_solve_lower_matrix_udf(),
        matrix_solve_upper_matrix_udf(),
        matrix_exp_udf(),
        matrix_exp_eigen_udf(),
        matrix_log_taylor_udf(),
        matrix_log_eigen_udf(),
        matrix_log_svd_udf(),
        matrix_power_udf(),
        matrix_sign_udf(),
        sparse_matvec_udf(),
        sparse_matmat_dense_udf(),
        sparse_transpose_udf(),
        sparse_matmat_sparse_udf(),
        tensor_sum_last_axis_udf(),
        tensor_l2_norm_last_axis_udf(),
        tensor_normalize_last_axis_udf(),
        tensor_batched_dot_last_axis_udf(),
        tensor_batched_matmul_last_two_udf(),
        tensor_permute_axes_udf(),
        tensor_contract_axes_udf(),
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
        assert_eq!(all_default_functions().len(), 72);
    }
}
