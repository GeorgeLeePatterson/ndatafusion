use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float64Array, Int32Array, ListArray, StructArray,
    UInt32Array,
};
use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use ndarray::{Array2, ArrayD, IxDyn};
use ndarrow::IntoArrow;

use super::common::nullable_or;
use crate::error::{exec_error, plan_error, scalar_argument_required, type_mismatch};
use crate::metadata::{
    fixed_shape_tensor_field, float64_csr_matrix_batch_field, variable_shape_tensor_field,
    vector_field,
};
use crate::signatures::any_signature;

fn scalar_usize(value: &ScalarValue, function_name: &str, position: usize) -> Result<usize> {
    match value {
        ScalarValue::Int64(Some(value)) => usize::try_from(*value).map_err(|_| {
            plan_error(
                function_name,
                format!("argument {position} must be a non-negative integer, found {value}"),
            )
        }),
        ScalarValue::Int32(Some(value)) => usize::try_from(*value).map_err(|_| {
            plan_error(
                function_name,
                format!("argument {position} must be a non-negative integer, found {value}"),
            )
        }),
        ScalarValue::UInt64(Some(value)) => usize::try_from(*value).map_err(|_| {
            plan_error(
                function_name,
                format!("argument {position} exceeds usize limits, found {value}"),
            )
        }),
        ScalarValue::UInt32(Some(value)) => usize::try_from(*value).map_err(|_| {
            plan_error(
                function_name,
                format!("argument {position} exceeds usize limits, found {value}"),
            )
        }),
        ScalarValue::Int64(None)
        | ScalarValue::Int32(None)
        | ScalarValue::UInt64(None)
        | ScalarValue::UInt32(None)
        | ScalarValue::Null => Err(scalar_argument_required(function_name, position)),
        value => Err(plan_error(
            function_name,
            format!("argument {position} must be an integer scalar, found {value:?}"),
        )),
    }
}

fn plan_usize_arg(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
    position: usize,
) -> Result<usize> {
    match args.scalar_arguments.get(position - 1).copied().flatten() {
        Some(value) => scalar_usize(value, function_name, position),
        None => Err(scalar_argument_required(function_name, position)),
    }
}

fn exec_usize_arg(
    args: &ScalarFunctionArgs,
    function_name: &str,
    position: usize,
) -> Result<usize> {
    match &args.args[position - 1] {
        ColumnarValue::Scalar(value) => scalar_usize(value, function_name, position),
        ColumnarValue::Array(_) => {
            Err(exec_error(function_name, format!("argument {position} must be an integer scalar")))
        }
    }
}

fn plan_dims(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
    start_position: usize,
    minimum_dims: usize,
) -> Result<Vec<usize>> {
    let dims = (start_position..=args.arg_fields.len())
        .map(|position| plan_usize_arg(args, function_name, position))
        .collect::<Result<Vec<_>>>()?;
    if dims.len() < minimum_dims {
        return Err(plan_error(
            function_name,
            format!("{function_name} requires at least {minimum_dims} dimension arguments"),
        ));
    }
    Ok(dims)
}

fn exec_dims(
    args: &ScalarFunctionArgs,
    function_name: &str,
    start_position: usize,
    minimum_dims: usize,
) -> Result<Vec<usize>> {
    let dims = (start_position..=args.args.len())
        .map(|position| exec_usize_arg(args, function_name, position))
        .collect::<Result<Vec<_>>>()?;
    if dims.len() < minimum_dims {
        return Err(exec_error(
            function_name,
            format!("{function_name} requires at least {minimum_dims} dimension arguments"),
        ));
    }
    Ok(dims)
}

fn array_arg(args: &ScalarFunctionArgs, function_name: &str, position: usize) -> Result<ArrayRef> {
    match &args.args[position - 1] {
        ColumnarValue::Array(array) => Ok(Arc::clone(array)),
        ColumnarValue::Scalar(value) => value
            .to_array_of_size(args.number_rows)
            .map_err(|error| exec_error(function_name, error)),
    }
}

fn expect_list_arg(
    args: &ScalarFunctionArgs,
    function_name: &str,
    position: usize,
) -> Result<ArrayRef> {
    let array = array_arg(args, function_name, position)?;
    if array.as_any().downcast_ref::<ListArray>().is_none() {
        return Err(exec_error(
            function_name,
            format!("argument {position} expected ListArray storage, found {}", array.data_type()),
        ));
    }
    Ok(array)
}

fn is_float64_list(data_type: &DataType) -> bool {
    matches!(data_type, DataType::List(item) if item.data_type() == &DataType::Float64)
}

fn is_nested_float64_list(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::List(item)
            if matches!(item.data_type(), DataType::List(inner) if inner.data_type() == &DataType::Float64)
    )
}

fn validate_float64_list_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<()> {
    if is_float64_list(field.data_type()) {
        Ok(())
    } else {
        Err(type_mismatch(function_name, position, "List<Float64>", field.data_type()))
    }
}

fn validate_int32_list_field(field: &FieldRef, function_name: &str, position: usize) -> Result<()> {
    match field.data_type() {
        DataType::List(item) if item.data_type() == &DataType::Int32 => Ok(()),
        actual => Err(type_mismatch(function_name, position, "List<Int32>", actual)),
    }
}

fn validate_u32_list_field(field: &FieldRef, function_name: &str, position: usize) -> Result<()> {
    match field.data_type() {
        DataType::List(item) if item.data_type() == &DataType::UInt32 => Ok(()),
        actual => Err(type_mismatch(function_name, position, "List<UInt32>", actual)),
    }
}

fn list_row_bounds(list: &ListArray, row: usize) -> (usize, usize) {
    let offsets = list.value_offsets();
    let start = usize::try_from(offsets[row]).expect("Arrow list offsets must be non-negative");
    let end = usize::try_from(offsets[row + 1]).expect("Arrow list offsets must be non-negative");
    (start, end)
}

fn reject_list_nulls(list: &ListArray, function_name: &str, position: usize) -> Result<()> {
    if list.null_count() > 0 {
        return Err(exec_error(function_name, format!("argument {position} contains null rows")));
    }
    Ok(())
}

fn float64_rows(list: &ListArray, function_name: &str, position: usize) -> Result<Vec<Vec<f64>>> {
    reject_list_nulls(list, function_name, position)?;
    let values = list.values().as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
        exec_error(
            function_name,
            format!(
                "argument {position} expected List<Float64> storage, found {}",
                list.values().data_type()
            ),
        )
    })?;
    if values.null_count() > 0 {
        return Err(exec_error(function_name, format!("argument {position} contains inner nulls")));
    }

    Ok((0..list.len())
        .map(|row| {
            let (start, end) = list_row_bounds(list, row);
            values.values().as_ref()[start..end].to_vec()
        })
        .collect())
}

fn int32_rows(list: &ListArray, function_name: &str, position: usize) -> Result<Vec<Vec<i32>>> {
    reject_list_nulls(list, function_name, position)?;
    let values = list.values().as_any().downcast_ref::<Int32Array>().ok_or_else(|| {
        exec_error(
            function_name,
            format!(
                "argument {position} expected List<Int32> storage, found {}",
                list.values().data_type()
            ),
        )
    })?;
    if values.null_count() > 0 {
        return Err(exec_error(function_name, format!("argument {position} contains inner nulls")));
    }

    Ok((0..list.len())
        .map(|row| {
            let (start, end) = list_row_bounds(list, row);
            values.values().as_ref()[start..end].to_vec()
        })
        .collect())
}

fn u32_rows(list: &ListArray, function_name: &str, position: usize) -> Result<Vec<Vec<u32>>> {
    reject_list_nulls(list, function_name, position)?;
    let values = list.values().as_any().downcast_ref::<UInt32Array>().ok_or_else(|| {
        exec_error(
            function_name,
            format!(
                "argument {position} expected List<UInt32> storage, found {}",
                list.values().data_type()
            ),
        )
    })?;
    if values.null_count() > 0 {
        return Err(exec_error(function_name, format!("argument {position} contains inner nulls")));
    }

    Ok((0..list.len())
        .map(|row| {
            let (start, end) = list_row_bounds(list, row);
            values.values().as_ref()[start..end].to_vec()
        })
        .collect())
}

fn nested_matrix_rows(
    list: &ListArray,
    function_name: &str,
    position: usize,
    row_count: usize,
    col_count: usize,
) -> Result<Vec<Vec<f64>>> {
    reject_list_nulls(list, function_name, position)?;

    let mut matrices = Vec::with_capacity(list.len());
    for outer_row in 0..list.len() {
        let nested = list.value(outer_row);
        let nested = nested.as_any().downcast_ref::<ListArray>().ok_or_else(|| {
            exec_error(
                function_name,
                format!("argument {position} expected List<List<Float64>> storage"),
            )
        })?;
        reject_list_nulls(nested, function_name, position)?;
        if nested.len() != row_count {
            return Err(exec_error(
                function_name,
                format!(
                    "argument {position} row {outer_row} expected {row_count} nested rows, found \
                     {}",
                    nested.len()
                ),
            ));
        }
        let values = nested.values().as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
            exec_error(function_name, format!("argument {position} expected nested Float64 values"))
        })?;
        if values.null_count() > 0 {
            return Err(exec_error(
                function_name,
                format!("argument {position} contains inner nulls"),
            ));
        }

        let mut flattened = Vec::with_capacity(row_count.saturating_mul(col_count));
        for inner_row in 0..nested.len() {
            let (start, end) = list_row_bounds(nested, inner_row);
            let width = end - start;
            if width != col_count {
                return Err(exec_error(
                    function_name,
                    format!(
                        "argument {position} row {outer_row} nested row {inner_row} expected \
                         width {col_count}, found {width}"
                    ),
                ));
            }
            flattened.extend_from_slice(&values.values().as_ref()[start..end]);
        }
        matrices.push(flattened);
    }

    Ok(matrices)
}

fn expected_elements(function_name: &str, dims: &[usize]) -> Result<usize> {
    dims.iter().try_fold(1_usize, |acc, dim| {
        acc.checked_mul(*dim).ok_or_else(|| {
            plan_error(function_name, format!("dimension product overflow for {dims:?}"))
        })
    })
}

fn build_empty_variable_tensor(function_name: &str, rank: usize) -> Result<StructArray> {
    let rank_i32 = i32::try_from(rank).map_err(|_| {
        exec_error(function_name, format!("tensor rank {rank} exceeds Arrow i32 limits"))
    })?;
    let data = ListArray::new(
        Arc::new(Field::new_list_field(DataType::Float64, false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32])),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
        None,
    );
    let shape = FixedSizeListArray::new(
        Arc::new(Field::new_list_field(DataType::Int32, false)),
        rank_i32,
        Arc::new(Int32Array::from(Vec::<i32>::new())),
        None,
    );
    Ok(StructArray::new(
        vec![
            Field::new("data", data.data_type().clone(), false),
            Field::new("shape", shape.data_type().clone(), false),
        ]
        .into(),
        vec![Arc::new(data), Arc::new(shape)],
        None,
    ))
}

fn build_empty_csr_matrix_batch(function_name: &str) -> Result<StructArray> {
    let shapes = FixedSizeListArray::new(
        Arc::new(Field::new_list_field(DataType::Int32, false)),
        2,
        Arc::new(Int32Array::from(Vec::<i32>::new())),
        None,
    );
    let row_ptrs = ListArray::new(
        Arc::new(Field::new_list_field(DataType::Int32, false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32])),
        Arc::new(Int32Array::from(Vec::<i32>::new())),
        None,
    );
    let col_indices = ListArray::new(
        Arc::new(Field::new_list_field(DataType::UInt32, false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32])),
        Arc::new(UInt32Array::from(Vec::<u32>::new())),
        None,
    );
    let values = ListArray::new(
        Arc::new(Field::new_list_field(DataType::Float64, false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32])),
        Arc::new(Float64Array::from(Vec::<f64>::new())),
        None,
    );

    let data_type = float64_csr_matrix_batch_field(function_name, false)?.data_type().clone();
    let DataType::Struct(fields) = data_type else {
        return Err(exec_error(function_name, "csr field builder returned non-struct storage"));
    };
    Ok(StructArray::new(
        fields,
        vec![Arc::new(shapes), Arc::new(row_ptrs), Arc::new(col_indices), Arc::new(values)],
        None,
    ))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MakeVector {
    signature: Signature,
}

impl MakeVector {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MakeVector {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "make_vector" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_float64_list_field(&args.arg_fields[0], self.name(), 1)?;
        let dim = plan_usize_arg(&args, self.name(), 2)?;
        vector_field(self.name(), dim, args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let dim = exec_usize_arg(&args, self.name(), 2)?;
        let values = expect_list_arg(&args, self.name(), 1)?;
        let values = values
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 1 expected ListArray storage"))?;
        let rows = float64_rows(values, self.name(), 1)?;
        for (row, values) in rows.iter().enumerate() {
            if values.len() != dim {
                return Err(exec_error(
                    self.name(),
                    format!("argument 1 row {row} expected length {dim}, found {}", values.len()),
                ));
            }
        }
        let data = rows.into_iter().flatten().collect::<Vec<_>>();
        let array = Array2::from_shape_vec((values.len(), dim), data)
            .map_err(|error| exec_error(self.name(), error))?;
        let output = array.into_arrow().map_err(|error| exec_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MakeMatrix {
    signature: Signature,
}

impl MakeMatrix {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for MakeMatrix {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "make_matrix" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        if !is_float64_list(args.arg_fields[0].data_type())
            && !is_nested_float64_list(args.arg_fields[0].data_type())
        {
            return Err(type_mismatch(
                self.name(),
                1,
                "List<Float64> or List<List<Float64>>",
                args.arg_fields[0].data_type(),
            ));
        }
        let rows = plan_usize_arg(&args, self.name(), 2)?;
        let cols = plan_usize_arg(&args, self.name(), 3)?;
        fixed_shape_tensor_field(self.name(), &[rows, cols], args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let row_count = exec_usize_arg(&args, self.name(), 2)?;
        let col_count = exec_usize_arg(&args, self.name(), 3)?;
        let values = expect_list_arg(&args, self.name(), 1)?;
        let values = values
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 1 expected ListArray storage"))?;

        let matrices = match args.arg_fields[0].data_type() {
            data_type if is_float64_list(data_type) => {
                let rows = float64_rows(values, self.name(), 1)?;
                let expected_len = expected_elements(self.name(), &[row_count, col_count])?;
                for (row, values) in rows.iter().enumerate() {
                    if values.len() != expected_len {
                        return Err(exec_error(
                            self.name(),
                            format!(
                                "argument 1 row {row} expected {expected_len} row-major values, \
                                 found {}",
                                values.len()
                            ),
                        ));
                    }
                }
                rows
            }
            data_type if is_nested_float64_list(data_type) => {
                nested_matrix_rows(values, self.name(), 1, row_count, col_count)?
            }
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!(
                        "argument 1 expected List<Float64> or List<List<Float64>>, found {actual}"
                    ),
                ));
            }
        };

        let data = matrices.into_iter().flatten().collect::<Vec<_>>();
        let batch_shape = IxDyn(&[values.len(), row_count, col_count]);
        let array = ArrayD::from_shape_vec(batch_shape, data)
            .map_err(|error| exec_error(self.name(), error))?;
        let (_field, output) = ndarrow::arrayd_to_fixed_shape_tensor(self.name(), array)
            .map_err(|error| exec_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MakeTensor {
    signature: Signature,
}

impl MakeTensor {
    fn new() -> Self {
        Self { signature: Signature::variadic_any(datafusion::logical_expr::Volatility::Immutable) }
    }
}

impl ScalarUDFImpl for MakeTensor {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "make_tensor" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_float64_list_field(&args.arg_fields[0], self.name(), 1)?;
        let dims = plan_dims(&args, self.name(), 2, 2)?;
        fixed_shape_tensor_field(self.name(), &dims, args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let dims = exec_dims(&args, self.name(), 2, 2)?;
        let expected_len = expected_elements(self.name(), &dims)?;
        let values = expect_list_arg(&args, self.name(), 1)?;
        let values = values
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 1 expected ListArray storage"))?;
        let rows = float64_rows(values, self.name(), 1)?;
        for (row, values) in rows.iter().enumerate() {
            if values.len() != expected_len {
                return Err(exec_error(
                    self.name(),
                    format!(
                        "argument 1 row {row} expected {expected_len} row-major values, found {}",
                        values.len()
                    ),
                ));
            }
        }

        let data = rows.into_iter().flatten().collect::<Vec<_>>();
        let mut shape = Vec::with_capacity(dims.len() + 1);
        shape.push(values.len());
        shape.extend(dims);
        let array = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|error| exec_error(self.name(), error))?;
        let (_field, output) = ndarrow::arrayd_to_fixed_shape_tensor(self.name(), array)
            .map_err(|error| exec_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MakeVariableTensor {
    signature: Signature,
}

impl MakeVariableTensor {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for MakeVariableTensor {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "make_variable_tensor" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_float64_list_field(&args.arg_fields[0], self.name(), 1)?;
        validate_int32_list_field(&args.arg_fields[1], self.name(), 2)?;
        let rank = plan_usize_arg(&args, self.name(), 3)?;
        variable_shape_tensor_field(self.name(), rank, None, nullable_or(&args.arg_fields[..2]))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let rank = exec_usize_arg(&args, self.name(), 3)?;
        let data = expect_list_arg(&args, self.name(), 1)?;
        let shape = expect_list_arg(&args, self.name(), 2)?;
        let data = data
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 1 expected ListArray storage"))?;
        let shape = shape
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 2 expected ListArray storage"))?;
        let data_rows = float64_rows(data, self.name(), 1)?;
        let shape_rows = int32_rows(shape, self.name(), 2)?;
        if data_rows.len() != shape_rows.len() {
            return Err(exec_error(
                self.name(),
                format!(
                    "argument length mismatch: data has {} rows, shape has {} rows",
                    data_rows.len(),
                    shape_rows.len()
                ),
            ));
        }
        if data_rows.is_empty() {
            return Ok(ColumnarValue::Array(Arc::new(build_empty_variable_tensor(
                self.name(),
                rank,
            )?)));
        }

        let mut tensors = Vec::with_capacity(data_rows.len());
        for (row, (values, shape)) in data_rows.into_iter().zip(shape_rows).enumerate() {
            if shape.len() != rank {
                return Err(exec_error(
                    self.name(),
                    format!("argument 2 row {row} expected rank {rank}, found {}", shape.len()),
                ));
            }
            let dims = shape
                .into_iter()
                .enumerate()
                .map(|(dim_idx, dim)| {
                    usize::try_from(dim).map_err(|_| {
                        exec_error(
                            self.name(),
                            format!(
                                "argument 2 row {row} contains negative dimension at index \
                                 {dim_idx}: {dim}"
                            ),
                        )
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            let expected_len = expected_elements(self.name(), &dims)
                .map_err(|error| exec_error(self.name(), error))?;
            if values.len() != expected_len {
                return Err(exec_error(
                    self.name(),
                    format!(
                        "argument 1 row {row} expected {expected_len} values for shape {dims:?}, \
                         found {}",
                        values.len()
                    ),
                ));
            }
            let tensor = ArrayD::from_shape_vec(IxDyn(&dims), values)
                .map_err(|error| exec_error(self.name(), error))?;
            tensors.push(tensor);
        }

        let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(self.name(), tensors, None)
            .map_err(|error| exec_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MakeCsrMatrixBatch {
    signature: Signature,
}

impl MakeCsrMatrixBatch {
    fn new() -> Self { Self { signature: any_signature(4) } }
}

impl ScalarUDFImpl for MakeCsrMatrixBatch {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "make_csr_matrix_batch" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_int32_list_field(&args.arg_fields[0], self.name(), 1)?;
        validate_int32_list_field(&args.arg_fields[1], self.name(), 2)?;
        validate_u32_list_field(&args.arg_fields[2], self.name(), 3)?;
        validate_float64_list_field(&args.arg_fields[3], self.name(), 4)?;
        float64_csr_matrix_batch_field(self.name(), nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let shape = expect_list_arg(&args, self.name(), 1)?;
        let row_ptrs = expect_list_arg(&args, self.name(), 2)?;
        let col_indices = expect_list_arg(&args, self.name(), 3)?;
        let values = expect_list_arg(&args, self.name(), 4)?;
        let shape = shape
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 1 expected ListArray storage"))?;
        let row_ptrs = row_ptrs
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 2 expected ListArray storage"))?;
        let col_indices = col_indices
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 3 expected ListArray storage"))?;
        let values = values
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 4 expected ListArray storage"))?;

        let shapes = int32_rows(shape, self.name(), 1)?;
        let row_ptrs = int32_rows(row_ptrs, self.name(), 2)?;
        let col_indices = u32_rows(col_indices, self.name(), 3)?;
        let values = float64_rows(values, self.name(), 4)?;
        let len = shapes.len();
        if row_ptrs.len() != len || col_indices.len() != len || values.len() != len {
            return Err(exec_error(
                self.name(),
                format!(
                    "argument length mismatch: shape={len}, row_ptrs={}, col_indices={}, values={}",
                    row_ptrs.len(),
                    col_indices.len(),
                    values.len()
                ),
            ));
        }
        if len == 0 {
            return Ok(ColumnarValue::Array(Arc::new(build_empty_csr_matrix_batch(self.name())?)));
        }

        let shapes = shapes
            .into_iter()
            .enumerate()
            .map(|(row, dims)| {
                if dims.len() != 2 {
                    return Err(exec_error(
                        self.name(),
                        format!(
                            "argument 1 row {row} expected shape length 2, found {}",
                            dims.len()
                        ),
                    ));
                }
                let rows = usize::try_from(dims[0]).map_err(|_| {
                    exec_error(
                        self.name(),
                        format!("argument 1 row {row} has negative row count {}", dims[0]),
                    )
                })?;
                let cols = usize::try_from(dims[1]).map_err(|_| {
                    exec_error(
                        self.name(),
                        format!("argument 1 row {row} has negative column count {}", dims[1]),
                    )
                })?;
                Ok([rows, cols])
            })
            .collect::<Result<Vec<_>>>()?;

        let (_field, output) = ndarrow::csr_batch_to_extension_array(
            self.name(),
            shapes,
            row_ptrs,
            col_indices,
            values,
        )
        .map_err(|error| exec_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
    }
}

#[must_use]
pub fn make_vector_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MakeVector::new())) }

#[must_use]
pub fn make_matrix_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MakeMatrix::new())) }

#[must_use]
pub fn make_tensor_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MakeTensor::new())) }

#[must_use]
pub fn make_variable_tensor_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MakeVariableTensor::new()))
}

#[must_use]
pub fn make_csr_matrix_batch_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MakeCsrMatrixBatch::new()))
}
