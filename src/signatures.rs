use datafusion::arrow::datatypes::DataType;
use datafusion::common::Result;
use datafusion::logical_expr::{Signature, Volatility};

use crate::error::plan_error;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ScalarCoercion {
    Boolean,
    Integer,
    Real,
}

pub(crate) fn any_signature(arg_count: usize) -> Signature {
    Signature::any(arg_count, Volatility::Immutable)
}

pub(crate) fn named_any_signature(arg_count: usize, parameter_names: &[&str]) -> Signature {
    Signature::any(arg_count, Volatility::Immutable)
        .with_parameter_names(parameter_names.to_vec())
        .expect("parameter names must match the fixed-arity signature")
}

pub(crate) fn user_defined_signature() -> Signature {
    Signature::user_defined(Volatility::Immutable)
}

pub(crate) fn named_user_defined_signature(parameter_names: &[&str]) -> Signature {
    Signature::user_defined(Volatility::Immutable)
        .with_parameter_names(parameter_names.to_vec())
        .expect("user-defined signatures accept named parameters")
}

fn coerce_scalar_type(
    function_name: &str,
    position: usize,
    data_type: &DataType,
    coercion: ScalarCoercion,
) -> Result<DataType> {
    match coercion {
        ScalarCoercion::Boolean => match data_type {
            DataType::Boolean | DataType::Null => Ok(DataType::Boolean),
            actual => Err(plan_error(
                function_name,
                format!("argument {position} must be a Boolean scalar, found {actual}"),
            )),
        },
        ScalarCoercion::Integer => match data_type {
            DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::Null => Ok(DataType::Int64),
            DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
                Ok(DataType::UInt64)
            }
            actual => Err(plan_error(
                function_name,
                format!("argument {position} must be an integer scalar, found {actual}"),
            )),
        },
        ScalarCoercion::Real => match data_type {
            DataType::Float32
            | DataType::Float64
            | DataType::Int8
            | DataType::Int16
            | DataType::Int32
            | DataType::Int64
            | DataType::UInt8
            | DataType::UInt16
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Null => Ok(DataType::Float64),
            actual => Err(plan_error(
                function_name,
                format!("argument {position} must be a numeric scalar, found {actual}"),
            )),
        },
    }
}

pub(crate) fn coerce_scalar_arguments(
    function_name: &str,
    arg_types: &[DataType],
    scalar_positions: &[(usize, ScalarCoercion)],
) -> Result<Vec<DataType>> {
    let mut coerced = arg_types.to_vec();
    for (position, coercion) in scalar_positions {
        let index =
            position.checked_sub(1).expect("argument positions are 1-based and must be positive");
        let Some(data_type) = arg_types.get(index) else {
            return Err(plan_error(function_name, format!("argument {position} is missing")));
        };
        coerced[index] = coerce_scalar_type(function_name, *position, data_type, *coercion)?;
    }
    Ok(coerced)
}

pub(crate) fn coerce_trailing_scalar_arguments(
    function_name: &str,
    arg_types: &[DataType],
    start_position: usize,
    coercion: ScalarCoercion,
) -> Result<Vec<DataType>> {
    let mut coerced = arg_types.to_vec();
    for (index, data_type) in arg_types.iter().enumerate().skip(start_position - 1) {
        let position = index + 1;
        coerced[index] = coerce_scalar_type(function_name, position, data_type, coercion)?;
    }
    Ok(coerced)
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::DataType;

    use super::{
        ScalarCoercion, coerce_scalar_arguments, coerce_trailing_scalar_arguments,
        named_user_defined_signature,
    };

    #[test]
    fn named_user_defined_signature_preserves_parameter_names() {
        let signature = named_user_defined_signature(&["matrix", "k"]);

        assert_eq!(
            signature.parameter_names.as_deref(),
            Some(["matrix".to_string(), "k".to_string()].as_slice())
        );
    }

    #[test]
    fn scalar_argument_coercion_normalizes_numeric_controls() {
        let arg_types = vec![DataType::Float32, DataType::UInt16, DataType::Int32];
        let coerced = coerce_scalar_arguments("matrix_exp", &arg_types, &[
            (2, ScalarCoercion::Integer),
            (3, ScalarCoercion::Real),
        ])
        .expect("valid scalar controls should coerce");

        assert_eq!(coerced, vec![DataType::Float32, DataType::UInt64, DataType::Float64]);
    }

    #[test]
    fn trailing_scalar_coercion_normalizes_variadic_integer_axes() {
        let arg_types = vec![DataType::Float64, DataType::Int16, DataType::UInt32];
        let coerced = coerce_trailing_scalar_arguments(
            "tensor_permute_axes",
            &arg_types,
            2,
            ScalarCoercion::Integer,
        )
        .expect("variadic integer axes should coerce");

        assert_eq!(coerced, vec![DataType::Float64, DataType::Int64, DataType::UInt64]);
    }

    #[test]
    fn scalar_argument_coercion_rejects_invalid_types() {
        let error = coerce_scalar_arguments(
            "matrix_power",
            &[DataType::Float64, DataType::Utf8],
            &[(2, ScalarCoercion::Real)],
        )
        .expect_err("non-numeric scalar controls must be rejected");

        let message = error.to_string();
        assert!(message.contains("matrix_power"));
        assert!(message.contains("argument 2 must be a numeric scalar"));
    }

    #[test]
    fn boolean_scalar_coercion_accepts_boolean_and_null() {
        let coerced = coerce_scalar_arguments(
            "linear_regression",
            &[DataType::Float32, DataType::Float32, DataType::Null],
            &[(3, ScalarCoercion::Boolean)],
        )
        .expect("null booleans should be castable for later validation");

        assert_eq!(coerced, vec![DataType::Float32, DataType::Float32, DataType::Boolean]);
    }
}
