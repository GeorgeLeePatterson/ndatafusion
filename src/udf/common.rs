use std::sync::Arc;

use datafusion::arrow::array::types::ArrowPrimitiveType;
use datafusion::arrow::array::{ArrayRef, FixedSizeListArray, PrimitiveArray, StructArray};
use datafusion::arrow::datatypes::{Field, FieldRef};
#[cfg(test)]
use datafusion::common::config::ConfigOptions;
use datafusion::common::{DataFusionError, Result, ScalarValue};
#[cfg(test)]
use datafusion::logical_expr::ScalarUDF;
use datafusion::logical_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs};
use nabled::arrow::ArrowInteropError;
use ndarray::{Array2, ArrayD, ArrayView2, ArrayView3, ArrayViewD, Ix3, IxDyn};
use ndarrow::NdarrowElement;
use num_complex::Complex64;

use crate::error::{array_argument_required, exec_error, scalar_argument_required};

pub(crate) fn nullable_or(fields: &[FieldRef]) -> bool {
    fields.iter().any(|field| field.is_nullable())
}

pub(crate) fn expect_array_arg<'a>(
    args: &'a ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<&'a ArrayRef> {
    let index = position - 1;
    match &args.args[index] {
        ColumnarValue::Array(array) => Ok(array),
        ColumnarValue::Scalar(_) => Err(array_argument_required(function_name, position)),
    }
}

pub(crate) fn expect_fixed_size_list_arg<'a>(
    args: &'a ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<&'a FixedSizeListArray> {
    let array = expect_array_arg(args, position, function_name)?;
    array.as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
        exec_error(
            function_name,
            format!(
                "argument {position} expected FixedSizeListArray storage, found {}",
                array.data_type()
            ),
        )
    })
}

pub(crate) fn expect_struct_arg<'a>(
    args: &'a ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<&'a StructArray> {
    let array = expect_array_arg(args, position, function_name)?;
    array.as_any().downcast_ref::<StructArray>().ok_or_else(|| {
        exec_error(
            function_name,
            format!(
                "argument {position} expected StructArray storage, found {}",
                array.data_type()
            ),
        )
    })
}

pub(crate) fn expect_bool_scalar_arg(
    args: &ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<bool> {
    let index = position - 1;
    match &args.args[index] {
        ColumnarValue::Scalar(ScalarValue::Boolean(Some(value))) => Ok(*value),
        ColumnarValue::Scalar(ScalarValue::Boolean(None) | ScalarValue::Null) => {
            Err(scalar_argument_required(function_name, position))
        }
        ColumnarValue::Scalar(value) => Err(exec_error(
            function_name,
            format!("argument {position} expected Boolean scalar, found {value:?}"),
        )),
        ColumnarValue::Array(_) => {
            Err(exec_error(function_name, format!("argument {position} must be a scalar Boolean")))
        }
    }
}

fn scalar_usize(value: &ScalarValue, function_name: &str, position: usize) -> Result<usize> {
    match value {
        ScalarValue::Int64(Some(value)) => usize::try_from(*value).map_err(|_| {
            exec_error(
                function_name,
                format!("argument {position} must be a non-negative integer, found {value}"),
            )
        }),
        ScalarValue::Int32(Some(value)) => usize::try_from(*value).map_err(|_| {
            exec_error(
                function_name,
                format!("argument {position} must be a non-negative integer, found {value}"),
            )
        }),
        ScalarValue::UInt64(Some(value)) => usize::try_from(*value).map_err(|_| {
            exec_error(
                function_name,
                format!("argument {position} exceeds usize limits, found {value}"),
            )
        }),
        ScalarValue::UInt32(Some(value)) => usize::try_from(*value).map_err(|_| {
            exec_error(
                function_name,
                format!("argument {position} exceeds usize limits, found {value}"),
            )
        }),
        ScalarValue::Int64(None)
        | ScalarValue::Int32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::Null => Err(scalar_argument_required(function_name, position)),
        value => Err(exec_error(
            function_name,
            format!("argument {position} must be an integer scalar, found {value:?}"),
        )),
    }
}

pub(crate) fn expect_usize_scalar_arg(
    args: &ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<usize> {
    match &args.args[position - 1] {
        ColumnarValue::Scalar(value) => scalar_usize(value, function_name, position),
        ColumnarValue::Array(_) => {
            Err(exec_error(function_name, format!("argument {position} must be an integer scalar")))
        }
    }
}

pub(crate) fn expect_usize_scalar_argument(
    args: &ReturnFieldArgs<'_>,
    position: usize,
    function_name: &str,
) -> Result<usize> {
    match args.scalar_arguments.get(position - 1).copied().flatten() {
        Some(value) => scalar_usize(value, function_name, position),
        None => Err(scalar_argument_required(function_name, position)),
    }
}

fn scalar_real(value: &ScalarValue, function_name: &str, position: usize) -> Result<f64> {
    match value {
        ScalarValue::Float64(Some(value)) => Ok(*value),
        ScalarValue::Float32(Some(value)) => Ok(f64::from(*value)),
        ScalarValue::Int64(Some(value)) => value.to_string().parse::<f64>().map_err(|error| {
            exec_error(
                function_name,
                format!("argument {position} could not be represented as f64: {error}"),
            )
        }),
        ScalarValue::Int32(Some(value)) => Ok(f64::from(*value)),
        ScalarValue::UInt64(Some(value)) => value.to_string().parse::<f64>().map_err(|error| {
            exec_error(
                function_name,
                format!("argument {position} could not be represented as f64: {error}"),
            )
        }),
        ScalarValue::UInt32(Some(value)) => Ok(f64::from(*value)),
        ScalarValue::Float64(None)
        | ScalarValue::Float32(None)
        | ScalarValue::Int64(None)
        | ScalarValue::Int32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::Null => Err(scalar_argument_required(function_name, position)),
        value => Err(exec_error(
            function_name,
            format!("argument {position} must be a numeric scalar, found {value:?}"),
        )),
    }
}

pub(crate) fn expect_real_scalar_arg(
    args: &ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<f64> {
    match &args.args[position - 1] {
        ColumnarValue::Scalar(value) => scalar_real(value, function_name, position),
        ColumnarValue::Array(_) => {
            Err(exec_error(function_name, format!("argument {position} must be a numeric scalar")))
        }
    }
}

pub(crate) fn expect_real_scalar_argument(
    args: &ReturnFieldArgs<'_>,
    position: usize,
    function_name: &str,
) -> Result<f64> {
    match args.scalar_arguments.get(position - 1).copied().flatten() {
        Some(value) => scalar_real(value, function_name, position),
        None => Err(scalar_argument_required(function_name, position)),
    }
}

pub(crate) fn expect_bool_scalar_argument(
    args: &ReturnFieldArgs<'_>,
    position: usize,
    function_name: &str,
) -> Result<bool> {
    let index = position - 1;
    match args.scalar_arguments.get(index).copied().flatten() {
        Some(ScalarValue::Boolean(Some(value))) => Ok(*value),
        Some(ScalarValue::Boolean(None) | ScalarValue::Null) | None => {
            Err(scalar_argument_required(function_name, position))
        }
        Some(value) => Err(exec_error(
            function_name,
            format!("argument {position} expected Boolean scalar, found {value:?}"),
        )),
    }
}

pub(crate) fn fixed_size_list_view2<'a, T>(
    array: &'a FixedSizeListArray,
    function_name: &str,
) -> Result<ArrayView2<'a, T::Native>>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    ndarrow::fixed_size_list_as_array2::<T>(array).map_err(|error| exec_error(function_name, error))
}

pub(crate) fn fixed_shape_tensor_viewd<'a, T>(
    field: &'a FieldRef,
    array: &'a FixedSizeListArray,
    function_name: &str,
) -> Result<ArrayViewD<'a, T::Native>>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    ndarrow::fixed_shape_tensor_as_array_viewd::<T>(field.as_ref(), array)
        .map_err(|error| exec_error(function_name, error))
}

pub(crate) fn fixed_shape_tensor_view3<'a, T>(
    field: &'a FieldRef,
    array: &'a FixedSizeListArray,
    function_name: &str,
) -> Result<ArrayView3<'a, T::Native>>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    fixed_shape_tensor_viewd::<T>(field, array, function_name)?
        .into_dimensionality::<Ix3>()
        .map_err(|error| exec_error(function_name, error))
}

pub(crate) fn complex_fixed_shape_tensor_viewd<'a>(
    field: &'a FieldRef,
    array: &'a FixedSizeListArray,
    function_name: &str,
) -> Result<ArrayViewD<'a, Complex64>> {
    ndarrow::complex64_fixed_shape_tensor_as_array_viewd(field.as_ref(), array)
        .map_err(|error| exec_error(function_name, error))
}

pub(crate) fn complex_fixed_shape_tensor_view3<'a>(
    field: &'a FieldRef,
    array: &'a FixedSizeListArray,
    function_name: &str,
) -> Result<ArrayView3<'a, Complex64>> {
    complex_fixed_shape_tensor_viewd(field, array, function_name)?
        .into_dimensionality::<Ix3>()
        .map_err(|error| exec_error(function_name, error))
}

pub(crate) fn complex_fixed_size_list_array_from_flat_rows(
    function_name: &str,
    row_count: usize,
    row_width: usize,
    values: Vec<Complex64>,
) -> Result<FixedSizeListArray> {
    let expected_len = row_count
        .checked_mul(row_width)
        .ok_or_else(|| exec_error(function_name, "row count overflow"))?;
    if values.len() != expected_len {
        return Err(exec_error(
            function_name,
            format!(
                "expected {expected_len} complex values for ({row_count}, {row_width}) rows, \
                 found {}",
                values.len()
            ),
        ));
    }
    let output = Array2::from_shape_vec((row_count, row_width), values)
        .map_err(|error| exec_error(function_name, error))?;
    ndarrow::array2_complex64_to_fixed_size_list(output)
        .map_err(|error| exec_error(function_name, error))
}

pub(crate) fn complex_fixed_shape_tensor_array_from_flat_rows(
    function_name: &str,
    batch: usize,
    shape: &[usize],
    values: Vec<Complex64>,
) -> Result<(Field, FixedSizeListArray)> {
    let mut full_shape = Vec::with_capacity(shape.len() + 1);
    full_shape.push(batch);
    full_shape.extend_from_slice(shape);
    let output = ArrayD::from_shape_vec(IxDyn(&full_shape), values)
        .map_err(|error| exec_error(function_name, error))?;
    ndarrow::arrayd_complex64_to_fixed_shape_tensor(function_name, output)
        .map_err(|error| exec_error(function_name, error))
}

pub(crate) fn primitive_array_from_values<T>(values: Vec<T::Native>) -> PrimitiveArray<T>
where
    T: ArrowPrimitiveType,
{
    PrimitiveArray::<T>::from_iter_values(values)
}

pub(crate) fn fixed_size_list_array_from_flat_rows<T>(
    function_name: &str,
    row_count: usize,
    row_width: usize,
    values: &[T::Native],
) -> Result<FixedSizeListArray>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    let expected_len = row_count
        .checked_mul(row_width)
        .ok_or_else(|| exec_error(function_name, "row count overflow"))?;
    if values.len() != expected_len {
        return Err(exec_error(
            function_name,
            format!(
                "expected {expected_len} values for ({row_count}, {row_width}) rows, found {}",
                values.len()
            ),
        ));
    }
    let row_width = i32::try_from(row_width)
        .map_err(|_| exec_error(function_name, "row width exceeds Arrow i32 limits"))?;
    let values = PrimitiveArray::<T>::from_iter_values(values.iter().copied());
    Ok(FixedSizeListArray::new(
        Arc::new(Field::new("item", T::DATA_TYPE, false)),
        row_width,
        Arc::new(values),
        None,
    ))
}

pub(crate) fn map_arrow_error(function_name: &str, error: ArrowInteropError) -> DataFusionError {
    exec_error(function_name, error)
}

#[cfg(test)]
pub(crate) fn invoke_udf(
    udf: &Arc<ScalarUDF>,
    args: Vec<ColumnarValue>,
    arg_fields: Vec<FieldRef>,
    scalar_arguments: &[Option<ScalarValue>],
    number_rows: usize,
) -> Result<(FieldRef, ColumnarValue)> {
    let scalar_refs =
        scalar_arguments.iter().map(Option::as_ref).collect::<Vec<Option<&ScalarValue>>>();
    let return_field = udf.return_field_from_args(ReturnFieldArgs {
        arg_fields:       &arg_fields,
        scalar_arguments: &scalar_refs,
    })?;
    let output = udf.invoke_with_args(ScalarFunctionArgs {
        args,
        arg_fields,
        number_rows,
        return_field: Arc::clone(&return_field),
        config_options: Arc::new(ConfigOptions::new()),
    })?;
    Ok((return_field, output))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::types::Float32Type;
    use datafusion::arrow::array::{FixedSizeListArray, Float32Array, Int64Array};
    use datafusion::arrow::datatypes::{DataType, Field};
    use ndarray::Array2;

    use super::*;

    fn dummy_field(name: &str) -> FieldRef { Arc::new(Field::new(name, DataType::Float32, false)) }

    #[test]
    fn expect_array_and_storage_helpers_validate_inputs() {
        let fixed = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(vec![Some(1.0_f32), Some(2.0)])],
            2,
        );
        let array = Float32Array::from(vec![1.0_f32, 2.0]);
        let args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Array(Arc::new(fixed.clone()))],
            arg_fields:     vec![dummy_field("vector")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        assert!(expect_array_arg(&args, 1, "vector_l2_norm").is_ok());
        assert!(expect_fixed_size_list_arg(&args, 1, "vector_l2_norm").is_ok());

        let scalar_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Scalar(ScalarValue::Float32(Some(1.0)))],
            arg_fields:     vec![dummy_field("vector")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let scalar_error =
            expect_array_arg(&scalar_args, 1, "vector_l2_norm").expect_err("scalar should fail");
        assert!(scalar_error.to_string().contains("argument 1 must be an array column"));

        let wrong_storage_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Array(Arc::new(array))],
            arg_fields:     vec![dummy_field("vector")],
            number_rows:    2,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let fixed_error = expect_fixed_size_list_arg(&wrong_storage_args, 1, "vector_l2_norm")
            .expect_err("plain array should fail");
        assert!(fixed_error.to_string().contains("expected FixedSizeListArray storage"));

        let struct_error =
            expect_struct_arg(&args, 1, "sparse_matvec").expect_err("fixed-size list should fail");
        assert!(struct_error.to_string().contains("expected StructArray storage"));
    }

    #[test]
    fn bool_scalar_helpers_validate_null_type_and_array_cases() {
        let bool_value = ScalarValue::Boolean(Some(true));
        let wrong_value = ScalarValue::Int64(Some(1));
        let arg_fields = vec![dummy_field("flag")];
        let scalar_refs = vec![Some(&bool_value)];
        let return_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &scalar_refs };
        assert!(
            expect_bool_scalar_argument(&return_args, 1, "linear_regression")
                .expect("boolean scalar")
        );

        let null_refs = vec![Some(&ScalarValue::Null)];
        let null_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &null_refs };
        let null_error = expect_bool_scalar_argument(&null_args, 1, "linear_regression")
            .expect_err("null boolean should fail");
        assert!(null_error.to_string().contains("argument 1 must be a non-null scalar"));

        let wrong_refs = vec![Some(&wrong_value)];
        let wrong_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &wrong_refs };
        let wrong_error = expect_bool_scalar_argument(&wrong_args, 1, "linear_regression")
            .expect_err("int scalar should fail");
        assert!(wrong_error.to_string().contains("expected Boolean scalar"));

        let exec_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Scalar(ScalarValue::Boolean(Some(false)))],
            arg_fields:     vec![dummy_field("flag")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        assert!(
            !expect_bool_scalar_arg(&exec_args, 1, "linear_regression").expect("boolean scalar")
        );

        let wrong_exec_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Scalar(ScalarValue::Int64(Some(1)))],
            arg_fields:     vec![dummy_field("flag")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let wrong_exec_error = expect_bool_scalar_arg(&wrong_exec_args, 1, "linear_regression")
            .expect_err("int scalar should fail");
        assert!(wrong_exec_error.to_string().contains("expected Boolean scalar"));

        let array_exec_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Array(Arc::new(Int64Array::from(vec![1_i64])))],
            arg_fields:     vec![dummy_field("flag")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let array_error = expect_bool_scalar_arg(&array_exec_args, 1, "linear_regression")
            .expect_err("array should fail");
        assert!(array_error.to_string().contains("must be a scalar Boolean"));
    }

    #[test]
    fn integer_scalar_helpers_validate_type_range_and_array_cases() {
        let value = ScalarValue::Int64(Some(3));
        let arg_fields = vec![dummy_field("count")];
        let scalar_refs = vec![Some(&value)];
        let return_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &scalar_refs };
        assert_eq!(
            expect_usize_scalar_argument(&return_args, 1, "matrix_exp").expect("integer scalar"),
            3
        );

        let negative_refs = vec![Some(&ScalarValue::Int64(Some(-1)))];
        let negative_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &negative_refs };
        let negative_error = expect_usize_scalar_argument(&negative_args, 1, "matrix_exp")
            .expect_err("negative integer should fail");
        assert!(negative_error.to_string().contains("must be a non-negative integer"));

        let wrong_refs = vec![Some(&ScalarValue::Float64(Some(1.0)))];
        let wrong_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &wrong_refs };
        let wrong_error = expect_usize_scalar_argument(&wrong_args, 1, "matrix_exp")
            .expect_err("float should fail");
        assert!(wrong_error.to_string().contains("must be an integer scalar"));

        let exec_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Scalar(ScalarValue::UInt32(Some(4)))],
            arg_fields:     vec![dummy_field("count")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        assert_eq!(expect_usize_scalar_arg(&exec_args, 1, "matrix_exp").expect("u32 scalar"), 4);

        let array_exec_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Array(Arc::new(Int64Array::from(vec![1_i64])))],
            arg_fields:     vec![dummy_field("count")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let array_error = expect_usize_scalar_arg(&array_exec_args, 1, "matrix_exp")
            .expect_err("array should fail");
        assert!(array_error.to_string().contains("must be an integer scalar"));
    }

    #[test]
    fn real_scalar_helpers_validate_type_null_and_array_cases() {
        let value = ScalarValue::Float32(Some(1.5));
        let arg_fields = vec![dummy_field("power")];
        let scalar_refs = vec![Some(&value)];
        let return_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &scalar_refs };
        let parsed =
            expect_real_scalar_argument(&return_args, 1, "matrix_power").expect("float scalar");
        assert!((parsed - 1.5_f64).abs() < f64::EPSILON);

        let int_value = ScalarValue::Int64(Some(2));
        let int_refs = vec![Some(&int_value)];
        let int_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &int_refs };
        let parsed_int =
            expect_real_scalar_argument(&int_args, 1, "matrix_power").expect("int scalar");
        assert!((parsed_int - 2.0_f64).abs() < f64::EPSILON);

        let wrong_refs = vec![Some(&ScalarValue::Boolean(Some(true)))];
        let wrong_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &wrong_refs };
        let wrong_error = expect_real_scalar_argument(&wrong_args, 1, "matrix_power")
            .expect_err("bool should fail");
        assert!(wrong_error.to_string().contains("must be a numeric scalar"));

        let exec_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Scalar(ScalarValue::Float64(Some(0.5)))],
            arg_fields:     vec![dummy_field("power")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let parsed_exec =
            expect_real_scalar_arg(&exec_args, 1, "matrix_power").expect("float64 scalar");
        assert!((parsed_exec - 0.5_f64).abs() < f64::EPSILON);

        let array_exec_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Array(Arc::new(Float32Array::from(vec![1.0_f32])))],
            arg_fields:     vec![dummy_field("power")],
            number_rows:    1,
            return_field:   dummy_field("return"),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let array_error = expect_real_scalar_arg(&array_exec_args, 1, "matrix_power")
            .expect_err("array should fail");
        assert!(array_error.to_string().contains("must be a numeric scalar"));
    }

    #[test]
    fn tensor_view_and_flat_row_helpers_validate_shapes() {
        let batch =
            Array2::from_shape_vec((1, 4), vec![1.0_f32, 2.0, 3.0, 4.0]).expect("rank-2 batch");
        let (field, tensor) =
            ndarrow::arrayd_to_fixed_shape_tensor("tensor", batch.into_dyn()).expect("tensor");
        let field = Arc::new(field);

        let rank_error = fixed_shape_tensor_view3::<Float32Type>(&field, &tensor, "tensor")
            .expect_err("rank-2 tensor batch should fail view3 conversion");
        assert!(rank_error.to_string().contains("tensor"));

        let mismatch =
            fixed_size_list_array_from_flat_rows::<Float32Type>("vector_normalize", 2, 2, &[
                1.0_f32, 2.0, 3.0,
            ])
            .expect_err("row mismatch should fail");
        assert!(mismatch.to_string().contains("expected 4 values"));

        let fixed =
            fixed_size_list_array_from_flat_rows::<Float32Type>("vector_normalize", 2, 2, &[
                1.0_f32, 2.0, 3.0, 4.0,
            ])
            .expect("fixed-size list");
        assert_eq!(fixed.value_length(), 2);
        assert_eq!(fixed.values().data_type(), &DataType::Float32);
    }
}
