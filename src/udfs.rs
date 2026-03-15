use std::sync::Arc;

use datafusion::logical_expr::ScalarUDF;

pub use crate::udf::constructors::{
    make_csr_matrix_batch_udf, make_matrix_udf, make_tensor_udf, make_variable_tensor_udf,
    make_vector_udf,
};
pub use crate::udf::decomposition::{
    matrix_balance_nonsymmetric_udf, matrix_cholesky_inverse_udf, matrix_cholesky_udf,
    matrix_determinant_udf, matrix_eigen_generalized_udf, matrix_eigen_symmetric_udf,
    matrix_gram_schmidt_classic_udf, matrix_gram_schmidt_udf, matrix_inverse_udf,
    matrix_log_determinant_udf, matrix_lu_udf, matrix_polar_complex_udf, matrix_polar_udf,
    matrix_qr_condition_number_udf, matrix_qr_pivoted_udf, matrix_qr_reconstruct_udf,
    matrix_qr_reduced_udf, matrix_qr_solve_least_squares_udf, matrix_qr_udf,
    matrix_schur_complex_udf, matrix_schur_udf, matrix_svd_condition_number_udf,
    matrix_svd_null_space_udf, matrix_svd_pseudo_inverse_udf, matrix_svd_rank_udf,
    matrix_svd_reconstruct_udf, matrix_svd_truncated_udf, matrix_svd_udf,
    matrix_svd_with_tolerance_udf,
};
pub use crate::udf::iterative::{
    matrix_conjugate_gradient_complex_udf, matrix_conjugate_gradient_udf, matrix_gmres_complex_udf,
    matrix_gmres_udf,
};
pub use crate::udf::matrix::{
    matrix_cholesky_solve_udf, matrix_lu_solve_udf, matrix_matmat_complex_udf, matrix_matmul_udf,
    matrix_matvec_complex_udf, matrix_matvec_udf,
};
pub use crate::udf::matrix_functions::{
    matrix_exp_complex_udf, matrix_exp_eigen_complex_udf, matrix_exp_eigen_udf, matrix_exp_udf,
    matrix_log_eigen_complex_udf, matrix_log_eigen_udf, matrix_log_svd_complex_udf,
    matrix_log_svd_udf, matrix_log_taylor_udf, matrix_power_complex_udf, matrix_power_udf,
    matrix_sign_complex_udf, matrix_sign_udf,
};
pub use crate::udf::ml::{
    linear_regression_udf, matrix_center_columns_complex_udf, matrix_center_columns_udf,
    matrix_column_means_complex_udf, matrix_column_means_udf, matrix_correlation_complex_udf,
    matrix_correlation_udf, matrix_covariance_complex_udf, matrix_covariance_udf,
    matrix_pca_inverse_transform_udf, matrix_pca_transform_udf, matrix_pca_udf,
};
pub use crate::udf::sparse::{
    sparse_lu_solve_udf, sparse_matmat_dense_udf, sparse_matmat_sparse_udf, sparse_matvec_udf,
    sparse_transpose_udf,
};
pub use crate::udf::tensor::{
    tensor_batched_dot_last_axis_udf, tensor_batched_matmul_last_two_udf, tensor_contract_axes_udf,
    tensor_l2_norm_last_axis_complex_udf, tensor_l2_norm_last_axis_udf,
    tensor_normalize_last_axis_complex_udf, tensor_normalize_last_axis_udf,
    tensor_permute_axes_udf, tensor_sum_last_axis_udf, tensor_variable_batched_dot_last_axis_udf,
    tensor_variable_l2_norm_last_axis_complex_udf, tensor_variable_l2_norm_last_axis_udf,
    tensor_variable_normalize_last_axis_complex_udf, tensor_variable_normalize_last_axis_udf,
    tensor_variable_sum_last_axis_udf,
};
pub use crate::udf::triangular::{
    matrix_solve_lower_matrix_udf, matrix_solve_lower_udf, matrix_solve_upper_matrix_udf,
    matrix_solve_upper_udf,
};
pub use crate::udf::vector::{
    vector_cosine_distance_udf, vector_cosine_similarity_complex_udf, vector_cosine_similarity_udf,
    vector_dot_hermitian_udf, vector_dot_udf, vector_l2_norm_complex_udf, vector_l2_norm_udf,
    vector_normalize_complex_udf, vector_normalize_udf,
};

/// Return all currently implemented `ndatafusion` scalar UDFs.
fn constructor_functions() -> Vec<Arc<ScalarUDF>> {
    vec![
        make_vector_udf(),
        make_matrix_udf(),
        make_tensor_udf(),
        make_variable_tensor_udf(),
        make_csr_matrix_batch_udf(),
    ]
}

fn vector_functions() -> Vec<Arc<ScalarUDF>> {
    vec![
        vector_l2_norm_udf(),
        vector_dot_udf(),
        vector_cosine_similarity_udf(),
        vector_cosine_distance_udf(),
        vector_normalize_udf(),
        vector_dot_hermitian_udf(),
        vector_l2_norm_complex_udf(),
        vector_cosine_similarity_complex_udf(),
        vector_normalize_complex_udf(),
    ]
}

fn matrix_and_decomposition_functions() -> Vec<Arc<ScalarUDF>> {
    vec![
        matrix_matvec_udf(),
        matrix_matvec_complex_udf(),
        matrix_matmul_udf(),
        matrix_matmat_complex_udf(),
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
        matrix_qr_reconstruct_udf(),
        matrix_svd_udf(),
        matrix_svd_truncated_udf(),
        matrix_svd_with_tolerance_udf(),
        matrix_svd_null_space_udf(),
        matrix_svd_pseudo_inverse_udf(),
        matrix_svd_condition_number_udf(),
        matrix_svd_rank_udf(),
        matrix_svd_reconstruct_udf(),
        matrix_eigen_symmetric_udf(),
        matrix_eigen_generalized_udf(),
        matrix_balance_nonsymmetric_udf(),
        matrix_schur_udf(),
        matrix_schur_complex_udf(),
        matrix_polar_udf(),
        matrix_polar_complex_udf(),
        matrix_gram_schmidt_udf(),
        matrix_gram_schmidt_classic_udf(),
        matrix_conjugate_gradient_udf(),
        matrix_gmres_udf(),
        matrix_conjugate_gradient_complex_udf(),
        matrix_gmres_complex_udf(),
        matrix_solve_lower_udf(),
        matrix_solve_upper_udf(),
        matrix_solve_lower_matrix_udf(),
        matrix_solve_upper_matrix_udf(),
        matrix_exp_udf(),
        matrix_exp_eigen_udf(),
        matrix_exp_complex_udf(),
        matrix_exp_eigen_complex_udf(),
        matrix_log_taylor_udf(),
        matrix_log_eigen_udf(),
        matrix_log_eigen_complex_udf(),
        matrix_log_svd_udf(),
        matrix_log_svd_complex_udf(),
        matrix_power_udf(),
        matrix_power_complex_udf(),
        matrix_sign_udf(),
        matrix_sign_complex_udf(),
    ]
}

fn sparse_tensor_and_ml_functions() -> Vec<Arc<ScalarUDF>> {
    vec![
        sparse_matvec_udf(),
        sparse_lu_solve_udf(),
        sparse_matmat_dense_udf(),
        sparse_transpose_udf(),
        sparse_matmat_sparse_udf(),
        tensor_sum_last_axis_udf(),
        tensor_l2_norm_last_axis_udf(),
        tensor_l2_norm_last_axis_complex_udf(),
        tensor_normalize_last_axis_udf(),
        tensor_normalize_last_axis_complex_udf(),
        tensor_batched_dot_last_axis_udf(),
        tensor_batched_matmul_last_two_udf(),
        tensor_permute_axes_udf(),
        tensor_contract_axes_udf(),
        tensor_variable_sum_last_axis_udf(),
        tensor_variable_l2_norm_last_axis_udf(),
        tensor_variable_l2_norm_last_axis_complex_udf(),
        tensor_variable_normalize_last_axis_udf(),
        tensor_variable_normalize_last_axis_complex_udf(),
        tensor_variable_batched_dot_last_axis_udf(),
        matrix_column_means_udf(),
        matrix_column_means_complex_udf(),
        matrix_center_columns_udf(),
        matrix_center_columns_complex_udf(),
        matrix_covariance_udf(),
        matrix_covariance_complex_udf(),
        matrix_correlation_udf(),
        matrix_correlation_complex_udf(),
        matrix_pca_udf(),
        matrix_pca_transform_udf(),
        matrix_pca_inverse_transform_udf(),
        linear_regression_udf(),
    ]
}

#[must_use]
pub fn all_default_functions() -> Vec<Arc<ScalarUDF>> {
    let mut functions = constructor_functions();
    functions.extend(vector_functions());
    functions.extend(matrix_and_decomposition_functions());
    functions.extend(sparse_tensor_and_ml_functions());
    functions
}

#[cfg(test)]
mod tests {
    use super::{
        all_default_functions, linear_regression_udf, make_csr_matrix_batch_udf, make_matrix_udf,
        make_variable_tensor_udf, make_vector_udf, matrix_center_columns_complex_udf,
        matrix_cholesky_solve_udf, matrix_column_means_complex_udf,
        matrix_conjugate_gradient_complex_udf, matrix_conjugate_gradient_udf,
        matrix_correlation_complex_udf, matrix_covariance_complex_udf, matrix_exp_complex_udf,
        matrix_exp_eigen_complex_udf, matrix_exp_udf, matrix_gmres_complex_udf, matrix_gmres_udf,
        matrix_log_eigen_complex_udf, matrix_log_svd_complex_udf, matrix_log_taylor_udf,
        matrix_lu_solve_udf, matrix_matmat_complex_udf, matrix_matmul_udf,
        matrix_matvec_complex_udf, matrix_matvec_udf, matrix_polar_complex_udf,
        matrix_power_complex_udf, matrix_power_udf, matrix_qr_solve_least_squares_udf,
        matrix_schur_complex_udf, matrix_sign_complex_udf, matrix_solve_lower_matrix_udf,
        matrix_solve_lower_udf, matrix_solve_upper_matrix_udf, matrix_solve_upper_udf,
        matrix_svd_truncated_udf, matrix_svd_with_tolerance_udf, sparse_lu_solve_udf,
        sparse_matmat_dense_udf, sparse_matmat_sparse_udf, sparse_matvec_udf, sparse_transpose_udf,
        tensor_batched_dot_last_axis_udf, tensor_batched_matmul_last_two_udf,
        tensor_contract_axes_udf, tensor_l2_norm_last_axis_complex_udf,
        tensor_l2_norm_last_axis_udf, tensor_normalize_last_axis_complex_udf,
        tensor_normalize_last_axis_udf, tensor_permute_axes_udf, tensor_sum_last_axis_udf,
        tensor_variable_batched_dot_last_axis_udf, tensor_variable_l2_norm_last_axis_complex_udf,
        tensor_variable_l2_norm_last_axis_udf, tensor_variable_normalize_last_axis_complex_udf,
        tensor_variable_normalize_last_axis_udf, tensor_variable_sum_last_axis_udf,
        vector_cosine_distance_udf, vector_cosine_similarity_complex_udf,
        vector_cosine_similarity_udf, vector_dot_hermitian_udf, vector_dot_udf,
        vector_l2_norm_complex_udf, vector_l2_norm_udf, vector_normalize_complex_udf,
        vector_normalize_udf,
    };

    #[test]
    fn default_udf_catalog_matches_current_surface() {
        assert_eq!(all_default_functions().len(), 102);
    }

    #[test]
    fn representative_udfs_expose_expected_aliases() {
        assert_eq!(vector_l2_norm_udf().aliases(), ["vector_norm"]);
        assert_eq!(vector_l2_norm_complex_udf().aliases(), ["vector_norm_complex"]);
        assert_eq!(make_variable_tensor_udf().aliases(), ["make_var_tensor"]);
        assert_eq!(tensor_l2_norm_last_axis_udf().aliases(), ["tensor_norm_last"]);
        assert_eq!(tensor_l2_norm_last_axis_complex_udf().aliases(), ["tensor_norm_last_complex"]);
        assert_eq!(tensor_variable_sum_last_axis_udf().aliases(), ["tensor_var_sum_last"]);
        assert_eq!(tensor_variable_l2_norm_last_axis_complex_udf().aliases(), [
            "tensor_var_norm_last_complex"
        ]);
        assert_eq!(matrix_qr_solve_least_squares_udf().aliases(), ["matrix_qr_solve_ls"],);
    }

    #[test]
    fn representative_udfs_expose_expected_parameter_names() {
        assert_eq!(
            make_matrix_udf().signature().parameter_names.as_deref(),
            Some(["values".to_string(), "rows".to_string(), "cols".to_string()].as_slice())
        );
        assert_eq!(
            make_csr_matrix_batch_udf().signature().parameter_names.as_deref(),
            Some(
                [
                    "shape".to_string(),
                    "row_ptrs".to_string(),
                    "col_indices".to_string(),
                    "values".to_string(),
                ]
                .as_slice()
            )
        );
        assert_eq!(
            matrix_exp_udf().signature().parameter_names.as_deref(),
            Some(
                ["matrix".to_string(), "max_terms".to_string(), "tolerance".to_string()].as_slice()
            )
        );
        assert_eq!(
            matrix_exp_complex_udf().signature().parameter_names.as_deref(),
            Some(
                ["matrix".to_string(), "max_terms".to_string(), "tolerance".to_string()].as_slice()
            )
        );
        assert_eq!(
            matrix_conjugate_gradient_udf().signature().parameter_names.as_deref(),
            Some(
                [
                    "matrix".to_string(),
                    "rhs".to_string(),
                    "tolerance".to_string(),
                    "max_iterations".to_string(),
                ]
                .as_slice()
            )
        );
        assert_eq!(
            matrix_gmres_udf().signature().parameter_names.as_deref(),
            Some(
                [
                    "matrix".to_string(),
                    "rhs".to_string(),
                    "tolerance".to_string(),
                    "max_iterations".to_string(),
                ]
                .as_slice()
            )
        );
        assert_eq!(
            matrix_svd_truncated_udf().signature().parameter_names.as_deref(),
            Some(["matrix".to_string(), "k".to_string()].as_slice())
        );
        assert_eq!(
            matrix_svd_with_tolerance_udf().signature().parameter_names.as_deref(),
            Some(["matrix".to_string(), "tolerance".to_string()].as_slice())
        );
        assert_eq!(
            matrix_power_complex_udf().signature().parameter_names.as_deref(),
            Some(["matrix".to_string(), "power".to_string()].as_slice())
        );
        assert_eq!(
            linear_regression_udf().signature().parameter_names.as_deref(),
            Some(
                ["design".to_string(), "response".to_string(), "add_intercept".to_string(),]
                    .as_slice()
            )
        );
    }

    #[test]
    fn documented_udfs_expose_documentation() {
        for udf in [
            make_vector_udf(),
            make_matrix_udf(),
            make_variable_tensor_udf(),
            make_csr_matrix_batch_udf(),
            vector_l2_norm_udf(),
            vector_dot_udf(),
            vector_cosine_similarity_udf(),
            vector_cosine_distance_udf(),
            vector_normalize_udf(),
            vector_dot_hermitian_udf(),
            vector_l2_norm_complex_udf(),
            vector_cosine_similarity_complex_udf(),
            vector_normalize_complex_udf(),
            matrix_matmul_udf(),
            matrix_matvec_udf(),
            matrix_lu_solve_udf(),
            matrix_cholesky_solve_udf(),
            matrix_exp_udf(),
            matrix_log_taylor_udf(),
            matrix_power_udf(),
            matrix_conjugate_gradient_udf(),
            matrix_gmres_udf(),
            matrix_conjugate_gradient_complex_udf(),
            matrix_gmres_complex_udf(),
            matrix_solve_lower_udf(),
            matrix_solve_upper_udf(),
            matrix_solve_lower_matrix_udf(),
            matrix_solve_upper_matrix_udf(),
            matrix_schur_complex_udf(),
            matrix_polar_complex_udf(),
            sparse_matvec_udf(),
            sparse_matmat_dense_udf(),
            sparse_transpose_udf(),
            sparse_matmat_sparse_udf(),
            sparse_lu_solve_udf(),
            tensor_sum_last_axis_udf(),
            tensor_l2_norm_last_axis_udf(),
            tensor_l2_norm_last_axis_complex_udf(),
            tensor_normalize_last_axis_udf(),
            tensor_normalize_last_axis_complex_udf(),
            tensor_batched_dot_last_axis_udf(),
            tensor_batched_matmul_last_two_udf(),
            tensor_permute_axes_udf(),
            tensor_contract_axes_udf(),
            tensor_variable_sum_last_axis_udf(),
            tensor_variable_l2_norm_last_axis_udf(),
            tensor_variable_l2_norm_last_axis_complex_udf(),
            tensor_variable_normalize_last_axis_udf(),
            tensor_variable_normalize_last_axis_complex_udf(),
            tensor_variable_batched_dot_last_axis_udf(),
            matrix_matvec_complex_udf(),
            matrix_matmat_complex_udf(),
            matrix_column_means_complex_udf(),
            matrix_center_columns_complex_udf(),
            matrix_covariance_complex_udf(),
            matrix_correlation_complex_udf(),
            matrix_exp_complex_udf(),
            matrix_exp_eigen_complex_udf(),
            matrix_log_eigen_complex_udf(),
            matrix_log_svd_complex_udf(),
            matrix_power_complex_udf(),
            matrix_sign_complex_udf(),
            matrix_svd_truncated_udf(),
            matrix_svd_with_tolerance_udf(),
            linear_regression_udf(),
        ] {
            assert!(udf.documentation().is_some(), "missing docs for {}", udf.name());
        }
    }
}
