#[cfg(test)]
use std::sync::Arc;

use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::array::{ArrayRef, FixedSizeListArray, StructArray};
use datafusion::arrow::datatypes::FieldRef;
#[cfg(test)]
use datafusion::common::config::ConfigOptions;
use datafusion::common::{DataFusionError, Result, ScalarValue};
#[cfg(test)]
use datafusion::logical_expr::ScalarUDF;
use datafusion::logical_expr::{ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs};
use nabled::arrow::ArrowInteropError;
use ndarray::{ArrayView2, ArrayView3, ArrayViewD, Ix3};

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

pub(crate) fn fixed_size_list_view2_f64<'a>(
    array: &'a FixedSizeListArray,
    function_name: &str,
) -> Result<ArrayView2<'a, f64>> {
    ndarrow::fixed_size_list_as_array2::<Float64Type>(array)
        .map_err(|error| exec_error(function_name, error))
}

pub(crate) fn fixed_shape_tensor_viewd_f64<'a>(
    field: &'a FieldRef,
    array: &'a FixedSizeListArray,
    function_name: &str,
) -> Result<ArrayViewD<'a, f64>> {
    ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(field.as_ref(), array)
        .map_err(|error| exec_error(function_name, error))
}

pub(crate) fn fixed_shape_tensor_view3_f64<'a>(
    field: &'a FieldRef,
    array: &'a FixedSizeListArray,
    function_name: &str,
) -> Result<ArrayView3<'a, f64>> {
    fixed_shape_tensor_viewd_f64(field, array, function_name)?
        .into_dimensionality::<Ix3>()
        .map_err(|error| exec_error(function_name, error))
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
