use std::fmt::Display;

use datafusion::arrow::datatypes::DataType;
use datafusion::common::DataFusionError;

pub(crate) fn plan_error(function_name: &str, message: impl Display) -> DataFusionError {
    DataFusionError::Plan(format!("{function_name}: {message}"))
}

pub(crate) fn exec_error(function_name: &str, message: impl Display) -> DataFusionError {
    DataFusionError::Execution(format!("{function_name}: {message}"))
}

pub(crate) fn array_argument_required(function_name: &str, position: usize) -> DataFusionError {
    exec_error(function_name, format!("argument {position} must be an array column"))
}

pub(crate) fn scalar_argument_required(function_name: &str, position: usize) -> DataFusionError {
    plan_error(function_name, format!("argument {position} must be a non-null scalar"))
}

pub(crate) fn type_mismatch(
    function_name: &str,
    position: usize,
    expected: &str,
    actual: &DataType,
) -> DataFusionError {
    plan_error(function_name, format!("argument {position} expected {expected}, found {actual}"))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::DataType;
    use datafusion::common::DataFusionError;

    use super::{
        array_argument_required, exec_error, plan_error, scalar_argument_required, type_mismatch,
    };

    #[test]
    fn plan_and_exec_error_helpers_prefix_function_name() {
        let plan = plan_error("vector_dot", "bad plan");
        let exec = exec_error("vector_dot", "bad exec");

        assert!(
            matches!(plan, DataFusionError::Plan(message) if message == "vector_dot: bad plan")
        );
        assert!(
            matches!(exec, DataFusionError::Execution(message) if message == "vector_dot: bad exec")
        );
    }

    #[test]
    fn argument_error_helpers_produce_stable_messages() {
        let array = array_argument_required("vector_dot", 1);
        let scalar = scalar_argument_required("linear_regression", 3);

        assert!(matches!(
            array,
            DataFusionError::Execution(message)
                if message == "vector_dot: argument 1 must be an array column"
        ));
        assert!(matches!(
            scalar,
            DataFusionError::Plan(message)
                if message == "linear_regression: argument 3 must be a non-null scalar"
        ));
    }

    #[test]
    fn type_mismatch_reports_expected_and_actual_types() {
        let error = type_mismatch(
            "tensor_sum_last_axis",
            1,
            "arrow.fixed_shape_tensor<Float64>",
            &DataType::Float64,
        );

        assert!(matches!(
            error,
            DataFusionError::Plan(message)
                if message == "tensor_sum_last_axis: argument 1 expected arrow.fixed_shape_tensor<Float64>, found Float64"
        ));
    }
}
