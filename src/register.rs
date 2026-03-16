use std::sync::Arc;

use datafusion::common::Result;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::{AggregateUDF, ScalarUDF};
use datafusion::prelude::SessionContext;

use crate::{udafs, udfs};

fn register_udfs(
    registry: &mut dyn FunctionRegistry,
    udfs: impl IntoIterator<Item = Arc<ScalarUDF>>,
) -> Result<()> {
    udfs.into_iter().try_for_each(|udf| {
        let existing_udf = registry.register_udf(udf)?;
        drop(existing_udf);
        Ok(())
    })
}

fn register_udafs(
    registry: &mut dyn FunctionRegistry,
    udafs: impl IntoIterator<Item = Arc<AggregateUDF>>,
) -> Result<()> {
    udafs.into_iter().try_for_each(|udaf| {
        let existing_udaf = registry.register_udaf(udaf)?;
        drop(existing_udaf);
        Ok(())
    })
}

/// Register the `ndatafusion` scalar and aggregate UDF catalog in a `FunctionRegistry`.
///
/// This is the entry point for making the `ndatafusion` SQL functions available to a
/// `SessionContext` or any other `DataFusion` function registry.
///
/// # Errors
///
/// Returns an error when the provided registry rejects UDF registration.
pub fn register_all(registry: &mut dyn FunctionRegistry) -> Result<()> {
    register_udfs(registry, udfs::all_default_functions())?;
    register_udafs(registry, udafs::all_default_aggregates())
}

/// Register the full `ndatafusion` catalog, including table functions, in a
/// [`SessionContext`].
///
/// This helper extends [`register_all`] with SQL-only surfaces that are not
/// part of the generic [`FunctionRegistry`] trait, such as table functions.
///
/// # Errors
///
/// Returns an error when the provided context rejects UDF or UDAF registration.
pub fn register_all_session(ctx: &mut SessionContext) -> Result<()> {
    register_all(ctx)?;
    ctx.register_udtf("unpack_struct", Arc::new(crate::udtf::UnpackStructTableFunction));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::Float64Array;
    use datafusion::arrow::datatypes::DataType;
    use datafusion::common::ScalarValue;
    use datafusion::execution::FunctionRegistry;
    use datafusion::execution::registry::MemoryFunctionRegistry;
    use datafusion::logical_expr::{ColumnarValue, Volatility, create_udf};
    use datafusion::prelude::SessionContext;

    use super::{register_all, register_all_session, register_udafs, register_udfs};

    fn stub_udf(name: &str) -> Arc<datafusion::logical_expr::ScalarUDF> {
        Arc::new(create_udf(
            name,
            Vec::new(),
            DataType::Int32,
            Volatility::Immutable,
            Arc::new(|_| Ok(ColumnarValue::Scalar(ScalarValue::Int32(Some(1))))),
        ))
    }

    #[test]
    fn register_all_accepts_the_current_catalog() {
        let mut registry = MemoryFunctionRegistry::new();

        register_all(&mut registry).expect("empty udf catalog should still register cleanly");

        assert!(registry.udfs().len() >= crate::udfs::all_default_functions().len());
        assert!(registry.udafs().len() >= crate::udafs::all_default_aggregates().len());
        assert!(registry.udfs().contains("vector_l2_norm"));
        assert!(registry.udafs().contains("vector_covariance_agg"));
        assert!(registry.udfs().contains("matrix_qr_solve_least_squares"));
    }

    #[tokio::test]
    async fn register_all_session_adds_table_functions() {
        let mut ctx = SessionContext::new();
        register_all_session(&mut ctx).expect("session registration should succeed");
        assert!(ctx.state().table_functions().contains_key("unpack_struct"));

        let batches = ctx
            .sql("SELECT * FROM unpack_struct(named_struct('sign', 1.0, 'log_abs', 3.5))")
            .await
            .expect("query should plan")
            .collect()
            .await
            .expect("query should execute");
        let sign =
            batches[0].column(0).as_any().downcast_ref::<Float64Array>().expect("sign column");
        assert!((sign.value(0) - 1.0).abs() < f64::EPSILON);
        let log_abs =
            batches[0].column(1).as_any().downcast_ref::<Float64Array>().expect("log_abs column");
        assert!((log_abs.value(0) - 3.5).abs() < f64::EPSILON);
    }

    #[test]
    fn register_udfs_adds_new_udfs_to_the_registry() {
        let mut registry = MemoryFunctionRegistry::new();

        register_udfs(&mut registry, vec![stub_udf("vector_norm"), stub_udf("vector_dot")])
            .expect("udf registration should succeed");

        assert!(registry.udfs().contains("vector_norm"));
        assert!(registry.udfs().contains("vector_dot"));
    }

    #[test]
    fn register_udfs_overwrites_existing_names_cleanly() {
        let mut registry = MemoryFunctionRegistry::new();

        register_udfs(&mut registry, vec![stub_udf("vector_norm")])
            .expect("initial registration should succeed");
        register_udfs(&mut registry, vec![stub_udf("vector_norm")])
            .expect("re-registration should succeed");

        assert_eq!(registry.udfs().len(), 1);
        assert!(registry.udfs().contains("vector_norm"));
    }

    #[test]
    fn register_udafs_adds_new_udafs_to_the_registry() {
        let mut registry = MemoryFunctionRegistry::new();

        register_udafs(&mut registry, crate::udafs::all_default_aggregates())
            .expect("udaf registration should succeed");

        assert!(registry.udafs().contains("vector_covariance_agg"));
        assert!(registry.udafs().contains("linear_regression_fit"));
    }
}
