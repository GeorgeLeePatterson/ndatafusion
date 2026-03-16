use std::sync::Arc;

use datafusion::logical_expr::ScalarUDF;

pub use crate::udf::constructors::{
    make_csr_matrix_batch_udf, make_matrix_udf, make_tensor_udf, make_variable_tensor_udf,
    make_vector_udf,
};
pub use crate::udf::decomposition::{
    matrix_balance_nonsymmetric_udf, matrix_cholesky_inverse_udf, matrix_cholesky_udf,
    matrix_determinant_udf, matrix_eigen_generalized_udf, matrix_eigen_nonsymmetric_bi_udf,
    matrix_eigen_nonsymmetric_complex_udf, matrix_eigen_nonsymmetric_udf,
    matrix_eigen_symmetric_udf, matrix_gram_schmidt_classic_udf, matrix_gram_schmidt_udf,
    matrix_inverse_udf, matrix_log_determinant_udf, matrix_lu_udf, matrix_polar_complex_udf,
    matrix_polar_udf, matrix_qr_condition_number_udf, matrix_qr_pivoted_udf,
    matrix_qr_reconstruct_udf, matrix_qr_reduced_udf, matrix_qr_solve_least_squares_udf,
    matrix_qr_udf, matrix_schur_complex_udf, matrix_schur_udf, matrix_svd_condition_number_udf,
    matrix_svd_null_space_udf, matrix_svd_pseudo_inverse_udf, matrix_svd_rank_udf,
    matrix_svd_reconstruct_udf, matrix_svd_truncated_udf, matrix_svd_udf,
    matrix_svd_with_tolerance_udf,
};
pub use crate::udf::differentiation::{
    gradient_udf, hessian_udf, jacobian_central_udf, jacobian_udf,
};
pub use crate::udf::iterative::{
    matrix_conjugate_gradient_complex_udf, matrix_conjugate_gradient_udf, matrix_gmres_complex_udf,
    matrix_gmres_udf,
};
pub use crate::udf::matrix::{
    matrix_cholesky_solve_udf, matrix_lu_solve_udf, matrix_matmat_complex_udf, matrix_matmul_udf,
    matrix_matvec_complex_udf, matrix_matvec_udf,
};
pub use crate::udf::matrix_equations::{
    matrix_solve_sylvester_complex_udf, matrix_solve_sylvester_mixed_complex_udf,
    matrix_solve_sylvester_mixed_f64_udf, matrix_solve_sylvester_udf,
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
    matrix_pca_complex_udf, matrix_pca_inverse_transform_complex_udf,
    matrix_pca_inverse_transform_udf, matrix_pca_transform_complex_udf, matrix_pca_transform_udf,
    matrix_pca_udf,
};
pub use crate::udf::optimization::{
    adam_complex_udf, backtracking_line_search_complex_udf, gradient_descent_complex_udf,
    momentum_descent_complex_udf,
};
pub use crate::udf::sparse::{
    sparse_lu_solve_udf, sparse_matmat_dense_udf, sparse_matmat_sparse_udf, sparse_matvec_udf,
    sparse_transpose_udf,
};
pub use crate::udf::sparse_factorization::{
    sparse_apply_iluk_preconditioner_udf, sparse_apply_ilut_preconditioner_udf,
    sparse_apply_jacobi_preconditioner_udf, sparse_iluk_factor_udf, sparse_ilut_factor_udf,
    sparse_jacobi_preconditioner_udf, sparse_lu_factor_udf,
    sparse_lu_solve_multiple_with_factorization_udf, sparse_lu_solve_with_factorization_udf,
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
pub use crate::udf::tensor_decomposition::{
    tensor_cp_als_nd_reconstruct_udf, tensor_cp_als_nd_udf, tensor_cp_als3_reconstruct_udf,
    tensor_cp_als3_udf, tensor_hooi_nd_udf, tensor_hosvd_nd_udf, tensor_tt_add_udf,
    tensor_tt_hadamard_round_udf, tensor_tt_hadamard_udf, tensor_tt_inner_udf, tensor_tt_norm_udf,
    tensor_tt_orthogonalize_left_udf, tensor_tt_orthogonalize_right_udf, tensor_tt_round_udf,
    tensor_tt_svd_reconstruct_udf, tensor_tt_svd_udf, tensor_tucker_expand_udf,
    tensor_tucker_project_udf,
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
        jacobian_udf(),
        jacobian_central_udf(),
        gradient_udf(),
        hessian_udf(),
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
        matrix_eigen_nonsymmetric_udf(),
        matrix_eigen_nonsymmetric_bi_udf(),
        matrix_eigen_nonsymmetric_complex_udf(),
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
        matrix_solve_sylvester_udf(),
        matrix_solve_sylvester_mixed_f64_udf(),
        matrix_solve_sylvester_complex_udf(),
        matrix_solve_sylvester_mixed_complex_udf(),
    ]
}

fn sparse_tensor_and_ml_functions() -> Vec<Arc<ScalarUDF>> {
    vec![
        sparse_matvec_udf(),
        sparse_lu_solve_udf(),
        sparse_lu_factor_udf(),
        sparse_lu_solve_with_factorization_udf(),
        sparse_lu_solve_multiple_with_factorization_udf(),
        sparse_jacobi_preconditioner_udf(),
        sparse_apply_jacobi_preconditioner_udf(),
        sparse_ilut_factor_udf(),
        sparse_iluk_factor_udf(),
        sparse_apply_ilut_preconditioner_udf(),
        sparse_apply_iluk_preconditioner_udf(),
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
        matrix_pca_complex_udf(),
        matrix_pca_transform_udf(),
        matrix_pca_transform_complex_udf(),
        matrix_pca_inverse_transform_udf(),
        matrix_pca_inverse_transform_complex_udf(),
        linear_regression_udf(),
        backtracking_line_search_complex_udf(),
        gradient_descent_complex_udf(),
        adam_complex_udf(),
        momentum_descent_complex_udf(),
    ]
}

fn tensor_decomposition_functions() -> Vec<Arc<ScalarUDF>> {
    vec![
        tensor_cp_als3_udf(),
        tensor_cp_als3_reconstruct_udf(),
        tensor_cp_als_nd_udf(),
        tensor_cp_als_nd_reconstruct_udf(),
        tensor_hosvd_nd_udf(),
        tensor_hooi_nd_udf(),
        tensor_tucker_project_udf(),
        tensor_tucker_expand_udf(),
        tensor_tt_svd_udf(),
        tensor_tt_orthogonalize_left_udf(),
        tensor_tt_orthogonalize_right_udf(),
        tensor_tt_round_udf(),
        tensor_tt_inner_udf(),
        tensor_tt_norm_udf(),
        tensor_tt_add_udf(),
        tensor_tt_hadamard_udf(),
        tensor_tt_hadamard_round_udf(),
        tensor_tt_svd_reconstruct_udf(),
    ]
}

#[must_use]
pub fn all_default_functions() -> Vec<Arc<ScalarUDF>> {
    let mut functions = constructor_functions();
    functions.extend(vector_functions());
    functions.extend(matrix_and_decomposition_functions());
    functions.extend(sparse_tensor_and_ml_functions());
    functions.extend(tensor_decomposition_functions());
    functions
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::simplify::{ExprSimplifyResult, SimplifyContext};
    use datafusion::logical_expr::{Expr, ScalarUDF};

    use super::{
        adam_complex_udf, all_default_functions, backtracking_line_search_complex_udf,
        gradient_descent_complex_udf, gradient_udf, hessian_udf, jacobian_central_udf,
        jacobian_udf, linear_regression_udf, make_csr_matrix_batch_udf, make_matrix_udf,
        make_variable_tensor_udf, make_vector_udf, matrix_center_columns_complex_udf,
        matrix_cholesky_solve_udf, matrix_column_means_complex_udf,
        matrix_conjugate_gradient_complex_udf, matrix_conjugate_gradient_udf,
        matrix_correlation_complex_udf, matrix_covariance_complex_udf,
        matrix_eigen_nonsymmetric_bi_udf, matrix_eigen_nonsymmetric_complex_udf,
        matrix_eigen_nonsymmetric_udf, matrix_exp_complex_udf, matrix_exp_udf,
        matrix_gmres_complex_udf, matrix_gmres_udf, matrix_log_taylor_udf, matrix_lu_solve_udf,
        matrix_matmat_complex_udf, matrix_matmul_udf, matrix_matvec_complex_udf, matrix_matvec_udf,
        matrix_pca_complex_udf, matrix_pca_inverse_transform_complex_udf,
        matrix_pca_transform_complex_udf, matrix_polar_complex_udf, matrix_power_complex_udf,
        matrix_qr_solve_least_squares_udf, matrix_schur_complex_udf, matrix_solve_lower_matrix_udf,
        matrix_solve_lower_udf, matrix_solve_sylvester_complex_udf,
        matrix_solve_sylvester_mixed_complex_udf, matrix_solve_sylvester_mixed_f64_udf,
        matrix_solve_sylvester_udf, matrix_solve_upper_matrix_udf, matrix_solve_upper_udf,
        matrix_svd_truncated_udf, matrix_svd_with_tolerance_udf, momentum_descent_complex_udf,
        sparse_apply_iluk_preconditioner_udf, sparse_apply_ilut_preconditioner_udf,
        sparse_apply_jacobi_preconditioner_udf, sparse_iluk_factor_udf, sparse_ilut_factor_udf,
        sparse_jacobi_preconditioner_udf, sparse_lu_factor_udf,
        sparse_lu_solve_multiple_with_factorization_udf, sparse_lu_solve_udf,
        sparse_lu_solve_with_factorization_udf, sparse_matmat_dense_udf, sparse_matmat_sparse_udf,
        sparse_matvec_udf, sparse_transpose_udf, tensor_batched_dot_last_axis_udf,
        tensor_batched_matmul_last_two_udf, tensor_contract_axes_udf,
        tensor_cp_als_nd_reconstruct_udf, tensor_cp_als_nd_udf, tensor_cp_als3_reconstruct_udf,
        tensor_cp_als3_udf, tensor_hooi_nd_udf, tensor_hosvd_nd_udf,
        tensor_l2_norm_last_axis_complex_udf, tensor_l2_norm_last_axis_udf,
        tensor_normalize_last_axis_complex_udf, tensor_normalize_last_axis_udf,
        tensor_permute_axes_udf, tensor_sum_last_axis_udf, tensor_tt_add_udf,
        tensor_tt_hadamard_round_udf, tensor_tt_hadamard_udf, tensor_tt_inner_udf,
        tensor_tt_norm_udf, tensor_tt_orthogonalize_left_udf, tensor_tt_orthogonalize_right_udf,
        tensor_tt_round_udf, tensor_tt_svd_reconstruct_udf, tensor_tt_svd_udf,
        tensor_tucker_expand_udf, tensor_tucker_project_udf,
        tensor_variable_batched_dot_last_axis_udf, tensor_variable_l2_norm_last_axis_complex_udf,
        tensor_variable_l2_norm_last_axis_udf, tensor_variable_normalize_last_axis_complex_udf,
        tensor_variable_normalize_last_axis_udf, tensor_variable_sum_last_axis_udf,
        vector_cosine_distance_udf, vector_cosine_similarity_complex_udf,
        vector_cosine_similarity_udf, vector_dot_hermitian_udf, vector_dot_udf,
        vector_l2_norm_complex_udf, vector_l2_norm_udf, vector_normalize_complex_udf,
        vector_normalize_udf,
    };

    fn assert_parameter_names(udf: &Arc<ScalarUDF>, expected: &[&str]) {
        let expected = expected.iter().map(ToString::to_string).collect::<Vec<_>>();
        assert_eq!(udf.signature().parameter_names.as_deref(), Some(expected.as_slice()));
    }

    fn assert_documented(udfs: &[Arc<ScalarUDF>]) {
        for udf in udfs {
            assert!(udf.documentation().is_some(), "missing docs for {}", udf.name());
        }
    }

    #[test]
    fn default_udf_catalog_matches_current_surface() {
        assert_eq!(all_default_functions().len(), 147);
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
        assert_eq!(matrix_qr_solve_least_squares_udf().aliases(), ["matrix_qr_solve_ls"]);
    }

    #[test]
    fn constructor_udfs_expose_expected_parameter_names() {
        assert_parameter_names(&make_matrix_udf(), &["values", "rows", "cols"]);
        assert_parameter_names(&make_csr_matrix_batch_udf(), &[
            "shape",
            "row_ptrs",
            "col_indices",
            "values",
        ]);
    }

    #[test]
    fn representative_udfs_expose_expected_simplify_rules() {
        let context = SimplifyContext::default();
        let matrix = Expr::Column(datafusion::common::Column::from_name("matrix"));
        let tensor = Expr::Column(datafusion::common::Column::from_name("tensor"));

        let power = matrix_power_complex_udf()
            .simplify(
                vec![matrix.clone(), Expr::Literal(ScalarValue::Float64(Some(1.0)), None)],
                &context,
            )
            .expect("matrix power simplify");
        assert!(matches!(power, ExprSimplifyResult::Simplified(expr) if expr == matrix));

        let permute = tensor_permute_axes_udf()
            .simplify(
                vec![
                    tensor.clone(),
                    Expr::Literal(ScalarValue::Int64(Some(0)), None),
                    Expr::Literal(ScalarValue::Int64(Some(1)), None),
                    Expr::Literal(ScalarValue::Int64(Some(2)), None),
                ],
                &context,
            )
            .expect("tensor permute simplify");
        assert!(matches!(permute, ExprSimplifyResult::Simplified(expr) if expr == tensor));
    }

    #[test]
    fn matrix_and_sparse_udfs_expose_expected_parameter_names() {
        assert_parameter_names(&matrix_exp_udf(), &["matrix", "max_terms", "tolerance"]);
        assert_parameter_names(&matrix_exp_complex_udf(), &["matrix", "max_terms", "tolerance"]);
        assert_parameter_names(&matrix_conjugate_gradient_udf(), &[
            "matrix",
            "rhs",
            "tolerance",
            "max_iterations",
        ]);
        assert_parameter_names(&matrix_gmres_udf(), &[
            "matrix",
            "rhs",
            "tolerance",
            "max_iterations",
        ]);
        assert_parameter_names(&matrix_svd_truncated_udf(), &["matrix", "k"]);
        assert_parameter_names(&matrix_svd_with_tolerance_udf(), &["matrix", "tolerance"]);
        assert_parameter_names(&matrix_power_complex_udf(), &["matrix", "power"]);
        assert_parameter_names(&sparse_ilut_factor_udf(), &[
            "matrix",
            "drop_tolerance",
            "max_fill",
        ]);
        assert_parameter_names(&sparse_iluk_factor_udf(), &["matrix", "level_of_fill"]);
        assert_parameter_names(&linear_regression_udf(), &["design", "response", "add_intercept"]);
    }

    #[test]
    fn differentiation_and_optimization_udfs_expose_expected_parameter_names() {
        let named_function_args =
            &["function", "vector", "step_size", "tolerance", "max_iterations"];
        assert_parameter_names(&jacobian_udf(), named_function_args);
        assert_parameter_names(&gradient_udf(), named_function_args);
        assert_parameter_names(&backtracking_line_search_complex_udf(), &[
            "function",
            "point",
            "direction",
            "initial_step",
            "contraction",
            "sufficient_decrease",
            "max_iterations",
        ]);
        assert_parameter_names(&adam_complex_udf(), &[
            "function",
            "initial",
            "learning_rate",
            "beta1",
            "beta2",
            "epsilon",
            "max_iterations",
            "tolerance",
        ]);
    }

    #[test]
    fn tensor_decomposition_udfs_expose_expected_parameter_names() {
        assert_parameter_names(&tensor_cp_als3_udf(), &[
            "tensor",
            "rank",
            "max_iterations",
            "tolerance",
        ]);
        assert_parameter_names(&tensor_hosvd_nd_udf(), &["tensor", "ranks"]);
        assert_parameter_names(&tensor_hooi_nd_udf(), &[
            "tensor",
            "ranks",
            "max_iterations",
            "tolerance",
        ]);
        assert_parameter_names(&tensor_tt_svd_udf(), &["tensor", "max_rank", "tolerance"]);
        assert_parameter_names(&tensor_tt_hadamard_round_udf(), &[
            "left",
            "right",
            "max_rank",
            "tolerance",
        ]);
    }

    #[test]
    fn constructor_and_vector_udfs_expose_documentation() {
        assert_documented(&[
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
            jacobian_udf(),
            jacobian_central_udf(),
            gradient_udf(),
            hessian_udf(),
        ]);
    }

    #[test]
    fn matrix_and_spectral_udfs_expose_documentation() {
        assert_documented(&[
            matrix_matmul_udf(),
            matrix_matvec_udf(),
            matrix_lu_solve_udf(),
            matrix_cholesky_solve_udf(),
            matrix_exp_udf(),
            matrix_log_taylor_udf(),
            matrix_conjugate_gradient_udf(),
            matrix_gmres_udf(),
            matrix_conjugate_gradient_complex_udf(),
            matrix_gmres_complex_udf(),
            matrix_solve_lower_udf(),
            matrix_solve_upper_udf(),
            matrix_solve_lower_matrix_udf(),
            matrix_solve_upper_matrix_udf(),
            matrix_solve_sylvester_udf(),
            matrix_solve_sylvester_mixed_f64_udf(),
            matrix_solve_sylvester_complex_udf(),
            matrix_solve_sylvester_mixed_complex_udf(),
            matrix_eigen_nonsymmetric_udf(),
            matrix_eigen_nonsymmetric_bi_udf(),
            matrix_eigen_nonsymmetric_complex_udf(),
            matrix_schur_complex_udf(),
            matrix_polar_complex_udf(),
            matrix_power_complex_udf(),
        ]);
    }

    #[test]
    fn sparse_and_tensor_udfs_expose_documentation() {
        assert_documented(&[
            sparse_matvec_udf(),
            sparse_matmat_dense_udf(),
            sparse_transpose_udf(),
            sparse_matmat_sparse_udf(),
            sparse_lu_solve_udf(),
            sparse_lu_factor_udf(),
            sparse_lu_solve_with_factorization_udf(),
            sparse_lu_solve_multiple_with_factorization_udf(),
            sparse_jacobi_preconditioner_udf(),
            sparse_apply_jacobi_preconditioner_udf(),
            sparse_ilut_factor_udf(),
            sparse_iluk_factor_udf(),
            sparse_apply_ilut_preconditioner_udf(),
            sparse_apply_iluk_preconditioner_udf(),
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
        ]);
    }

    #[test]
    fn tensor_decomposition_and_ml_udfs_expose_documentation() {
        assert_documented(&[
            tensor_cp_als3_udf(),
            tensor_cp_als3_reconstruct_udf(),
            tensor_cp_als_nd_udf(),
            tensor_cp_als_nd_reconstruct_udf(),
            tensor_hosvd_nd_udf(),
            tensor_hooi_nd_udf(),
            tensor_tucker_project_udf(),
            tensor_tucker_expand_udf(),
            tensor_tt_svd_udf(),
            tensor_tt_orthogonalize_left_udf(),
            tensor_tt_orthogonalize_right_udf(),
            tensor_tt_round_udf(),
            tensor_tt_inner_udf(),
            tensor_tt_norm_udf(),
            tensor_tt_add_udf(),
            tensor_tt_hadamard_udf(),
            tensor_tt_hadamard_round_udf(),
            tensor_tt_svd_reconstruct_udf(),
            matrix_matvec_complex_udf(),
            matrix_matmat_complex_udf(),
            matrix_column_means_complex_udf(),
            matrix_center_columns_complex_udf(),
            matrix_covariance_complex_udf(),
            matrix_correlation_complex_udf(),
            matrix_pca_complex_udf(),
            matrix_pca_transform_complex_udf(),
            matrix_pca_inverse_transform_complex_udf(),
            matrix_exp_complex_udf(),
            matrix_power_complex_udf(),
            matrix_svd_truncated_udf(),
            matrix_svd_with_tolerance_udf(),
            backtracking_line_search_complex_udf(),
            gradient_descent_complex_udf(),
            adam_complex_udf(),
            momentum_descent_complex_udf(),
            linear_regression_udf(),
        ]);
    }
}
