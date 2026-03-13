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
use ndarray::{ArrayView2, ArrayView3, ArrayViewD, Ix3};
use ndarrow::NdarrowElement;

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
