use std::sync::Arc;

use datafusion::common::Result;
use datafusion::execution::FunctionRegistry;
use datafusion::logical_expr::ScalarUDF;

use crate::udfs;

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

/// Register the `ndatafusion` scalar UDF catalog in a `FunctionRegistry`.
///
/// This is the entry point for making the `ndatafusion` SQL functions available to a
/// `SessionContext` or any other `DataFusion` function registry.
///
/// # Errors
///
/// Returns an error when the provided registry rejects UDF registration.
pub fn register_all(registry: &mut dyn FunctionRegistry) -> Result<()> {
    register_udfs(registry, udfs::all_default_functions())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::datatypes::DataType;
    use datafusion::common::ScalarValue;
    use datafusion::execution::FunctionRegistry;
    use datafusion::execution::registry::MemoryFunctionRegistry;
    use datafusion::logical_expr::{ColumnarValue, Volatility, create_udf};

    use super::{register_all, register_udfs};

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

        assert_eq!(registry.udfs().len(), crate::udfs::all_default_functions().len());
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
}
