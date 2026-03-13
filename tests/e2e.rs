//! Minimal integration test target required by `.justfile`.

use datafusion::execution::FunctionRegistry;
use datafusion::execution::registry::MemoryFunctionRegistry;

#[test]
fn register_all_accepts_the_current_catalog() {
    let mut registry = MemoryFunctionRegistry::new();

    ndatafusion::register_all(&mut registry)
        .expect("the current scaffold should register successfully");

    assert_eq!(registry.udfs().len(), ndatafusion::udfs::all_default_functions().len());
}
