use std::sync::Arc;

use datafusion::logical_expr::AggregateUDF;

pub use crate::udaf::{
    linear_regression_fit_udaf, vector_correlation_agg_udaf, vector_covariance_agg_udaf,
    vector_pca_fit_udaf,
};

/// Return all currently implemented `ndatafusion` aggregate UDFs.
#[must_use]
pub fn all_default_aggregates() -> Vec<Arc<AggregateUDF>> {
    vec![
        vector_covariance_agg_udaf(),
        vector_correlation_agg_udaf(),
        vector_pca_fit_udaf(),
        linear_regression_fit_udaf(),
    ]
}

#[cfg(test)]
mod tests {
    use super::{
        all_default_aggregates, linear_regression_fit_udaf, vector_correlation_agg_udaf,
        vector_covariance_agg_udaf, vector_pca_fit_udaf,
    };

    #[test]
    fn default_udaf_catalog_matches_current_surface() {
        assert_eq!(all_default_aggregates().len(), 4);
    }

    #[test]
    fn representative_udafs_expose_expected_parameter_names() {
        assert_eq!(
            linear_regression_fit_udaf().signature().parameter_names.as_deref(),
            Some(
                ["design".to_string(), "response".to_string(), "add_intercept".to_string(),]
                    .as_slice()
            )
        );
    }

    #[test]
    fn documented_udafs_expose_documentation() {
        for udaf in [
            vector_covariance_agg_udaf(),
            vector_correlation_agg_udaf(),
            vector_pca_fit_udaf(),
            linear_regression_fit_udaf(),
        ] {
            assert!(udaf.documentation().is_some(), "missing docs for {}", udaf.name());
        }
    }
}
