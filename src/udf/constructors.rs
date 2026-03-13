use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int32Array, ListArray,
    PrimitiveArray, StructArray, UInt32Array,
};
use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use ndarray::{Array2, ArrayD, IxDyn};
use ndarrow::{IntoArrow, NdarrowElement};

use super::common::nullable_or;
use crate::error::{exec_error, plan_error, scalar_argument_required, type_mismatch};
use crate::metadata::{
    csr_matrix_batch_field, fixed_shape_tensor_field, variable_shape_tensor_field, vector_field,
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

fn numeric_value_type(data_type: &DataType) -> Option<DataType> {
    match data_type {
        DataType::Float32 => Some(DataType::Float32),
        DataType::Float64 => Some(DataType::Float64),
        _ => None,
    }
}

fn list_numeric_value_type(data_type: &DataType) -> Option<DataType> {
    match data_type {
        DataType::List(item) => numeric_value_type(item.data_type()),
        _ => None,
    }
}

fn nested_list_numeric_value_type(data_type: &DataType) -> Option<DataType> {
    match data_type {
        DataType::List(item) => list_numeric_value_type(item.data_type()),
        _ => None,
    }
}

fn validate_numeric_list_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<DataType> {
    list_numeric_value_type(field.data_type()).ok_or_else(|| {
        type_mismatch(function_name, position, "List<Float32|Float64>", field.data_type())
    })
}

fn validate_numeric_or_nested_list_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<DataType> {
    if let Some(value_type) = list_numeric_value_type(field.data_type()) {
        return Ok(value_type);
    }
    if let Some(value_type) = nested_list_numeric_value_type(field.data_type()) {
        return Ok(value_type);
    }
    Err(type_mismatch(
        function_name,
        position,
        "List<Float32|Float64> or List<List<Float32|Float64>>",
        field.data_type(),
    ))
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

fn numeric_rows<T>(
    list: &ListArray,
    function_name: &str,
    position: usize,
) -> Result<Vec<Vec<T::Native>>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
    reject_list_nulls(list, function_name, position)?;
    let values = list.values().as_any().downcast_ref::<PrimitiveArray<T>>().ok_or_else(|| {
        exec_error(
            function_name,
            format!(
                "argument {position} expected numeric list storage, found {}",
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

fn nested_matrix_rows<T>(
    list: &ListArray,
    function_name: &str,
    position: usize,
    row_count: usize,
    col_count: usize,
) -> Result<Vec<Vec<T::Native>>>
where
    T: ArrowPrimitiveType,
    T::Native: Copy,
{
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
        let values =
            nested.values().as_any().downcast_ref::<PrimitiveArray<T>>().ok_or_else(|| {
                exec_error(
                    function_name,
                    format!("argument {position} expected nested numeric values"),
                )
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

fn build_empty_variable_tensor_typed(
    function_name: &str,
    value_type: &DataType,
    rank: usize,
) -> Result<StructArray> {
    let rank_i32 = i32::try_from(rank).map_err(|_| {
        exec_error(function_name, format!("tensor rank {rank} exceeds Arrow i32 limits"))
    })?;
    let data = ListArray::new(
        Arc::new(Field::new_list_field(value_type.clone(), false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32])),
        match value_type {
            DataType::Float32 => Arc::new(Float32Array::from(Vec::<f32>::new())) as ArrayRef,
            DataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
            _ => {
                return Err(exec_error(
                    function_name,
                    format!("unsupported constructor value type {value_type}"),
                ));
            }
        },
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

fn build_empty_csr_matrix_batch(function_name: &str, value_type: &DataType) -> Result<StructArray> {
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
        Arc::new(Field::new_list_field(value_type.clone(), false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32])),
        match value_type {
            DataType::Float32 => Arc::new(Float32Array::from(Vec::<f32>::new())) as ArrayRef,
            DataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
            _ => {
                return Err(exec_error(
                    function_name,
                    format!("unsupported constructor value type {value_type}"),
                ));
            }
        },
        None,
    );

    let data_type = csr_matrix_batch_field(function_name, value_type, false)?.data_type().clone();
    let DataType::Struct(fields) = data_type else {
        return Err(exec_error(function_name, "csr field builder returned non-struct storage"));
    };
    Ok(StructArray::new(
        fields,
        vec![Arc::new(shapes), Arc::new(row_ptrs), Arc::new(col_indices), Arc::new(values)],
        None,
    ))
}

fn variable_tensor_dims(
    function_name: &str,
    row: usize,
    shape: Vec<i32>,
    rank: usize,
) -> Result<Vec<usize>> {
    if shape.len() != rank {
        return Err(exec_error(
            function_name,
            format!("argument 2 row {row} expected rank {rank}, found {}", shape.len()),
        ));
    }

    shape
        .into_iter()
        .enumerate()
        .map(|(dim_idx, dim)| {
            usize::try_from(dim).map_err(|_| {
                exec_error(
                    function_name,
                    format!(
                        "argument 2 row {row} contains negative dimension at index {dim_idx}: \
                         {dim}"
                    ),
                )
            })
        })
        .collect()
}

fn invoke_make_variable_tensor_typed<T>(
    function_name: &str,
    value_type: &DataType,
    rank: usize,
    data: &ListArray,
    shape_rows: Vec<Vec<i32>>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let data_rows = numeric_rows::<T>(data, function_name, 1)?;
    if data_rows.len() != shape_rows.len() {
        return Err(exec_error(
            function_name,
            format!(
                "argument length mismatch: data has {} rows, shape has {} rows",
                data_rows.len(),
                shape_rows.len()
            ),
        ));
    }
    if data_rows.is_empty() {
        return Ok(ColumnarValue::Array(Arc::new(build_empty_variable_tensor_typed(
            function_name,
            value_type,
            rank,
        )?)));
    }

    let tensors = data_rows
        .into_iter()
        .zip(shape_rows)
        .enumerate()
        .map(|(row, (values, shape))| {
            let dims = variable_tensor_dims(function_name, row, shape, rank)?;
            let expected_len = expected_elements(function_name, &dims)
                .map_err(|error| exec_error(function_name, error))?;
            if values.len() != expected_len {
                return Err(exec_error(
                    function_name,
                    format!(
                        "argument 1 row {row} expected {expected_len} values for shape {dims:?}, \
                         found {}",
                        values.len()
                    ),
                ));
            }
            ArrayD::from_shape_vec(IxDyn(&dims), values)
                .map_err(|error| exec_error(function_name, error))
        })
        .collect::<Result<Vec<_>>>()?;
    let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(function_name, tensors, None)
        .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn parse_csr_batch_shapes(function_name: &str, shapes: Vec<Vec<i32>>) -> Result<Vec<[usize; 2]>> {
    shapes
        .into_iter()
        .enumerate()
        .map(|(row, dims)| {
            if dims.len() != 2 {
                return Err(exec_error(
                    function_name,
                    format!("argument 1 row {row} expected shape length 2, found {}", dims.len()),
                ));
            }
            let rows = usize::try_from(dims[0]).map_err(|_| {
                exec_error(
                    function_name,
                    format!("argument 1 row {row} has negative row count {}", dims[0]),
                )
            })?;
            let cols = usize::try_from(dims[1]).map_err(|_| {
                exec_error(
                    function_name,
                    format!("argument 1 row {row} has negative column count {}", dims[1]),
                )
            })?;
            Ok([rows, cols])
        })
        .collect()
}

fn invoke_make_csr_matrix_batch_typed<T>(
    function_name: &str,
    value_type: &DataType,
    shapes: Vec<Vec<i32>>,
    row_ptrs: Vec<Vec<i32>>,
    col_indices: Vec<Vec<u32>>,
    values: &ListArray,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let values = numeric_rows::<T>(values, function_name, 4)?;
    let len = shapes.len();
    if row_ptrs.len() != len || col_indices.len() != len || values.len() != len {
        return Err(exec_error(
            function_name,
            format!(
                "argument length mismatch: shape={len}, row_ptrs={}, col_indices={}, values={}",
                row_ptrs.len(),
                col_indices.len(),
                values.len()
            ),
        ));
    }
    if len == 0 {
        return Ok(ColumnarValue::Array(Arc::new(build_empty_csr_matrix_batch(
            function_name,
            value_type,
        )?)));
    }

    let shapes = parse_csr_batch_shapes(function_name, shapes)?;
    let (_field, output) =
        ndarrow::csr_batch_to_extension_array(function_name, shapes, row_ptrs, col_indices, values)
            .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
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
        let value_type = validate_numeric_list_field(&args.arg_fields[0], self.name(), 1)?;
        let dim = plan_usize_arg(&args, self.name(), 2)?;
        vector_field(self.name(), &value_type, dim, args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let dim = exec_usize_arg(&args, self.name(), 2)?;
        let values = expect_list_arg(&args, self.name(), 1)?;
        let values = values
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 1 expected ListArray storage"))?;
        match validate_numeric_list_field(&args.arg_fields[0], self.name(), 1)? {
            DataType::Float32 => {
                let rows = numeric_rows::<Float32Type>(values, self.name(), 1)?;
                for (row, values) in rows.iter().enumerate() {
                    if values.len() != dim {
                        return Err(exec_error(
                            self.name(),
                            format!(
                                "argument 1 row {row} expected length {dim}, found {}",
                                values.len()
                            ),
                        ));
                    }
                }
                let data = rows.into_iter().flatten().collect::<Vec<_>>();
                let output = Array2::from_shape_vec((values.len(), dim), data)
                    .map_err(|error| exec_error(self.name(), error))?
                    .into_arrow()
                    .map_err(|error| exec_error(self.name(), error))?;
                Ok(ColumnarValue::Array(Arc::new(output)))
            }
            DataType::Float64 => {
                let rows = numeric_rows::<Float64Type>(values, self.name(), 1)?;
                for (row, values) in rows.iter().enumerate() {
                    if values.len() != dim {
                        return Err(exec_error(
                            self.name(),
                            format!(
                                "argument 1 row {row} expected length {dim}, found {}",
                                values.len()
                            ),
                        ));
                    }
                }
                let data = rows.into_iter().flatten().collect::<Vec<_>>();
                let output = Array2::from_shape_vec((values.len(), dim), data)
                    .map_err(|error| exec_error(self.name(), error))?
                    .into_arrow()
                    .map_err(|error| exec_error(self.name(), error))?;
                Ok(ColumnarValue::Array(Arc::new(output)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported constructor value type {actual}")))
            }
        }
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
        let value_type =
            validate_numeric_or_nested_list_field(&args.arg_fields[0], self.name(), 1)?;
        let rows = plan_usize_arg(&args, self.name(), 2)?;
        let cols = plan_usize_arg(&args, self.name(), 3)?;
        fixed_shape_tensor_field(
            self.name(),
            &value_type,
            &[rows, cols],
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let row_count = exec_usize_arg(&args, self.name(), 2)?;
        let col_count = exec_usize_arg(&args, self.name(), 3)?;
        let values = expect_list_arg(&args, self.name(), 1)?;
        let values = values
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 1 expected ListArray storage"))?;

        match validate_numeric_or_nested_list_field(&args.arg_fields[0], self.name(), 1)? {
            DataType::Float32 => {
                let matrices = if list_numeric_value_type(args.arg_fields[0].data_type()).is_some()
                {
                    let rows = numeric_rows::<Float32Type>(values, self.name(), 1)?;
                    let expected_len = expected_elements(self.name(), &[row_count, col_count])?;
                    for (row, values) in rows.iter().enumerate() {
                        if values.len() != expected_len {
                            return Err(exec_error(
                                self.name(),
                                format!(
                                    "argument 1 row {row} expected {expected_len} row-major \
                                     values, found {}",
                                    values.len()
                                ),
                            ));
                        }
                    }
                    rows
                } else {
                    nested_matrix_rows::<Float32Type>(values, self.name(), 1, row_count, col_count)?
                };
                let data = matrices.into_iter().flatten().collect::<Vec<_>>();
                let array =
                    ArrayD::from_shape_vec(IxDyn(&[values.len(), row_count, col_count]), data)
                        .map_err(|error| exec_error(self.name(), error))?;
                let (_field, output) = ndarrow::arrayd_to_fixed_shape_tensor(self.name(), array)
                    .map_err(|error| exec_error(self.name(), error))?;
                Ok(ColumnarValue::Array(Arc::new(output)))
            }
            DataType::Float64 => {
                let matrices = if list_numeric_value_type(args.arg_fields[0].data_type()).is_some()
                {
                    let rows = numeric_rows::<Float64Type>(values, self.name(), 1)?;
                    let expected_len = expected_elements(self.name(), &[row_count, col_count])?;
                    for (row, values) in rows.iter().enumerate() {
                        if values.len() != expected_len {
                            return Err(exec_error(
                                self.name(),
                                format!(
                                    "argument 1 row {row} expected {expected_len} row-major \
                                     values, found {}",
                                    values.len()
                                ),
                            ));
                        }
                    }
                    rows
                } else {
                    nested_matrix_rows::<Float64Type>(values, self.name(), 1, row_count, col_count)?
                };
                let data = matrices.into_iter().flatten().collect::<Vec<_>>();
                let array =
                    ArrayD::from_shape_vec(IxDyn(&[values.len(), row_count, col_count]), data)
                        .map_err(|error| exec_error(self.name(), error))?;
                let (_field, output) = ndarrow::arrayd_to_fixed_shape_tensor(self.name(), array)
                    .map_err(|error| exec_error(self.name(), error))?;
                Ok(ColumnarValue::Array(Arc::new(output)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported constructor value type {actual}")))
            }
        }
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
        let value_type = validate_numeric_list_field(&args.arg_fields[0], self.name(), 1)?;
        let dims = plan_dims(&args, self.name(), 2, 2)?;
        fixed_shape_tensor_field(self.name(), &value_type, &dims, args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let dims = exec_dims(&args, self.name(), 2, 2)?;
        let expected_len = expected_elements(self.name(), &dims)?;
        let values = expect_list_arg(&args, self.name(), 1)?;
        let values = values
            .as_any()
            .downcast_ref::<ListArray>()
            .ok_or_else(|| exec_error(self.name(), "argument 1 expected ListArray storage"))?;
        match validate_numeric_list_field(&args.arg_fields[0], self.name(), 1)? {
            DataType::Float32 => {
                let rows = numeric_rows::<Float32Type>(values, self.name(), 1)?;
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
            DataType::Float64 => {
                let rows = numeric_rows::<Float64Type>(values, self.name(), 1)?;
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
            actual => {
                Err(exec_error(self.name(), format!("unsupported constructor value type {actual}")))
            }
        }
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
        let value_type = validate_numeric_list_field(&args.arg_fields[0], self.name(), 1)?;
        validate_int32_list_field(&args.arg_fields[1], self.name(), 2)?;
        let rank = plan_usize_arg(&args, self.name(), 3)?;
        variable_shape_tensor_field(
            self.name(),
            &value_type,
            rank,
            None,
            nullable_or(&args.arg_fields[..2]),
        )
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
        let shape_rows = int32_rows(shape, self.name(), 2)?;
        let value_type = validate_numeric_list_field(&args.arg_fields[0], self.name(), 1)?;
        match value_type {
            DataType::Float32 => invoke_make_variable_tensor_typed::<Float32Type>(
                self.name(),
                &value_type,
                rank,
                data,
                shape_rows,
            ),
            DataType::Float64 => invoke_make_variable_tensor_typed::<Float64Type>(
                self.name(),
                &value_type,
                rank,
                data,
                shape_rows,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported constructor value type {actual}")))
            }
        }
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
        let value_type = validate_numeric_list_field(&args.arg_fields[3], self.name(), 4)?;
        csr_matrix_batch_field(self.name(), &value_type, nullable_or(args.arg_fields))
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
        let value_type = validate_numeric_list_field(&args.arg_fields[3], self.name(), 4)?;
        match value_type {
            DataType::Float32 => invoke_make_csr_matrix_batch_typed::<Float32Type>(
                self.name(),
                &value_type,
                shapes,
                row_ptrs,
                col_indices,
                values,
            ),
            DataType::Float64 => invoke_make_csr_matrix_batch_typed::<Float64Type>(
                self.name(),
                &value_type,
                shapes,
                row_ptrs,
                col_indices,
                values,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported constructor value type {actual}")))
            }
        }
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

#[cfg(test)]
mod tests {
    use datafusion::common::config::ConfigOptions;
    use datafusion::common::utils::arrays_into_list_array;

    use super::*;

    fn dummy_field(name: &str, data_type: DataType) -> FieldRef {
        Arc::new(Field::new(name, data_type, false))
    }

    fn float32_list(rows: Vec<Vec<f32>>) -> ListArray {
        ListArray::from_iter_primitive::<Float32Type, _, _>(
            rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
        )
    }

    fn int32_list(rows: Vec<Vec<i32>>) -> ListArray {
        ListArray::from_iter_primitive::<datafusion::arrow::array::types::Int32Type, _, _>(
            rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
        )
    }

    fn u32_list(rows: Vec<Vec<u32>>) -> ListArray {
        ListArray::from_iter_primitive::<datafusion::arrow::array::types::UInt32Type, _, _>(
            rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
        )
    }

    fn nested_float32_list(rows: Vec<Vec<Vec<f32>>>) -> ListArray {
        let arrays = rows
            .into_iter()
            .map(|matrix| Arc::new(float32_list(matrix)) as ArrayRef)
            .collect::<Vec<_>>();
        arrays_into_list_array(arrays).expect("nested list")
    }

    #[test]
    fn scalar_and_dimension_helpers_validate_inputs() {
        assert_eq!(scalar_usize(&ScalarValue::Int64(Some(4)), "make_tensor", 2).expect("int64"), 4);
        assert_eq!(scalar_usize(&ScalarValue::Int32(Some(3)), "make_tensor", 2).expect("int32"), 3);
        assert_eq!(
            scalar_usize(&ScalarValue::UInt32(Some(2)), "make_tensor", 2).expect("uint32"),
            2
        );

        let negative = scalar_usize(&ScalarValue::Int64(Some(-1)), "make_tensor", 2)
            .expect_err("negative integer should fail");
        assert!(negative.to_string().contains("must be a non-negative integer"));

        let null_error = scalar_usize(&ScalarValue::Null, "make_tensor", 2)
            .expect_err("null scalar should fail");
        assert!(null_error.to_string().contains("argument 2 must be a non-null scalar"));

        let wrong_type = scalar_usize(&ScalarValue::Float64(Some(1.0)), "make_tensor", 2)
            .expect_err("float scalar should fail");
        assert!(wrong_type.to_string().contains("must be an integer scalar"));

        let arg_fields = vec![dummy_field("values", DataType::Float32)];
        let dim_scalar = ScalarValue::Int64(Some(5));
        let scalar_refs = vec![Some(&dim_scalar)];
        let plan_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &scalar_refs };
        assert_eq!(plan_usize_arg(&plan_args, "make_vector", 1).expect("planned dim"), 5);

        let missing_refs = Vec::<Option<&ScalarValue>>::new();
        let missing_args =
            ReturnFieldArgs { arg_fields: &arg_fields, scalar_arguments: &missing_refs };
        let missing = plan_usize_arg(&missing_args, "make_vector", 1)
            .expect_err("missing scalar should fail");
        assert!(missing.to_string().contains("argument 1 must be a non-null scalar"));

        let dims_fields = vec![
            dummy_field("values", DataType::Float32),
            dummy_field("d0", DataType::Int64),
            dummy_field("d1", DataType::Int64),
        ];
        let d0 = ScalarValue::Int64(Some(2));
        let d1 = ScalarValue::Int64(Some(3));
        let dims_refs = vec![None, Some(&d0), Some(&d1)];
        let dims_args =
            ReturnFieldArgs { arg_fields: &dims_fields, scalar_arguments: &dims_refs };
        assert_eq!(plan_dims(&dims_args, "make_tensor", 2, 2).expect("dims"), vec![2, 3]);

        let too_few_dims =
            plan_dims(&plan_args, "make_tensor", 2, 2).expect_err("too few dimensions should fail");
        assert!(too_few_dims.to_string().contains("requires at least 2 dimension arguments"));

        let exec_args = ScalarFunctionArgs {
            args:           vec![
                ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(3))),
            ],
            arg_fields:     vec![
                dummy_field("d0", DataType::Int64),
                dummy_field("d1", DataType::Int64),
            ],
            number_rows:    1,
            return_field:   dummy_field("return", DataType::Null),
            config_options: Arc::new(ConfigOptions::new()),
        };
        assert_eq!(exec_dims(&exec_args, "make_tensor", 1, 2).expect("exec dims"), vec![2, 3]);

        let array_exec_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Array(Arc::new(Int32Array::from(vec![1_i32])))],
            arg_fields:     vec![dummy_field("d0", DataType::Int32)],
            number_rows:    1,
            return_field:   dummy_field("return", DataType::Null),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let exec_error = exec_usize_arg(&array_exec_args, "make_tensor", 1)
            .expect_err("array dimension should fail");
        assert!(exec_error.to_string().contains("must be an integer scalar"));
    }

    #[test]
    fn list_and_row_helpers_validate_storage_and_nulls() {
        let scalar_list = ScalarValue::List(Arc::new(float32_list(vec![vec![1.0, 2.0]])));
        let scalar_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Scalar(scalar_list)],
            arg_fields:     vec![dummy_field(
                "values",
                DataType::new_list(DataType::Float32, false),
            )],
            number_rows:    2,
            return_field:   dummy_field("return", DataType::Null),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let scalar_array = array_arg(&scalar_args, "make_vector", 1).expect("scalar list expands");
        assert_eq!(scalar_array.len(), 2);
        assert!(expect_list_arg(&scalar_args, "make_vector", 1).is_ok());

        let wrong_storage_args = ScalarFunctionArgs {
            args:           vec![ColumnarValue::Array(Arc::new(Float32Array::from(vec![
                1.0_f32, 2.0,
            ])))],
            arg_fields:     vec![dummy_field("values", DataType::Float32)],
            number_rows:    2,
            return_field:   dummy_field("return", DataType::Null),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let storage_error = expect_list_arg(&wrong_storage_args, "make_vector", 1)
            .expect_err("plain array should fail");
        assert!(storage_error.to_string().contains("expected ListArray storage"));

        let list_field = dummy_field("values", DataType::new_list(DataType::Float32, false));
        let nested_field = dummy_field(
            "nested",
            DataType::new_list(DataType::new_list(DataType::Float32, false), false),
        );
        let int_list_field = dummy_field("ints", DataType::new_list(DataType::Int32, false));
        let u32_list_field = dummy_field("indices", DataType::new_list(DataType::UInt32, false));

        assert_eq!(
            validate_numeric_list_field(&list_field, "make_vector", 1).expect("numeric list"),
            DataType::Float32
        );
        assert_eq!(
            validate_numeric_or_nested_list_field(&nested_field, "make_matrix", 1)
                .expect("nested numeric list"),
            DataType::Float32
        );
        assert!(validate_int32_list_field(&int_list_field, "make_csr_matrix_batch", 1).is_ok());
        assert!(validate_u32_list_field(&u32_list_field, "make_csr_matrix_batch", 3).is_ok());

        let bad_numeric =
            validate_numeric_list_field(&dummy_field("bad", DataType::Int32), "make_vector", 1)
                .expect_err("bad numeric field should fail");
        assert!(bad_numeric.to_string().contains("List<Float32|Float64>"));

        let bad_nested = validate_numeric_or_nested_list_field(
            &dummy_field("bad_nested", DataType::new_list(DataType::Int32, false)),
            "make_matrix",
            1,
        )
        .expect_err("bad nested field should fail");
        assert!(
            bad_nested.to_string().contains("List<Float32|Float64> or List<List<Float32|Float64>>")
        );

        let int32_error = validate_int32_list_field(&list_field, "make_csr_matrix_batch", 1)
            .expect_err("float list should fail int32 validation");
        assert!(int32_error.to_string().contains("List<Int32>"));

        let u32_error = validate_u32_list_field(&list_field, "make_csr_matrix_batch", 3)
            .expect_err("float list should fail u32 validation");
        assert!(u32_error.to_string().contains("List<UInt32>"));

        let values = float32_list(vec![vec![1.0, 2.0], vec![3.0]]);
        assert_eq!(list_row_bounds(&values, 1), (2, 3));
        assert_eq!(
            numeric_rows::<Float32Type>(&values, "make_vector", 1).expect("numeric rows"),
            vec![vec![1.0_f32, 2.0], vec![3.0]]
        );

        let null_rows = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![
            None,
            Some(vec![Some(1.0_f32)]),
        ]);
        let null_row_error =
            reject_list_nulls(&null_rows, "make_vector", 1).expect_err("null rows should fail");
        assert!(null_row_error.to_string().contains("contains null rows"));

        let inner_nulls = ListArray::from_iter_primitive::<Float32Type, _, _>(vec![Some(vec![
            Some(1.0_f32),
            None,
        ])]);
        let numeric_error = numeric_rows::<Float32Type>(&inner_nulls, "make_vector", 1)
            .expect_err("inner nulls should fail");
        assert!(numeric_error.to_string().contains("contains inner nulls"));

        let int_rows = int32_rows(&int32_list(vec![vec![0, 1, 2]]), "make_csr_matrix_batch", 2)
            .expect("int rows");
        assert_eq!(int_rows, vec![vec![0, 1, 2]]);

        let bad_int_rows =
            int32_rows(&values, "make_csr_matrix_batch", 2).expect_err("float storage should fail");
        assert!(bad_int_rows.to_string().contains("expected List<Int32> storage"));

        let u32_rows_out =
            u32_rows(&u32_list(vec![vec![0, 1]]), "make_csr_matrix_batch", 3).expect("u32 rows");
        assert_eq!(u32_rows_out, vec![vec![0_u32, 1]]);

        let bad_u32_rows =
            u32_rows(&values, "make_csr_matrix_batch", 3).expect_err("float storage should fail");
        assert!(bad_u32_rows.to_string().contains("expected List<UInt32> storage"));
    }

    #[test]
    fn nested_matrix_and_shape_helpers_validate_contracts() {
        let nested = nested_float32_list(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);
        let parsed = nested_matrix_rows::<Float32Type>(&nested, "make_matrix", 1, 2, 2)
            .expect("matrix rows");
        assert_eq!(parsed, vec![vec![1.0_f32, 2.0, 3.0, 4.0]]);

        let wrong_row_count = nested_matrix_rows::<Float32Type>(
            &nested_float32_list(vec![vec![vec![1.0, 2.0]]]),
            "make_matrix",
            1,
            2,
            2,
        )
        .expect_err("row count mismatch should fail");
        assert!(wrong_row_count.to_string().contains("expected 2 nested rows"));

        let wrong_width = nested_matrix_rows::<Float32Type>(
            &nested_float32_list(vec![vec![vec![1.0, 2.0], vec![3.0]]]),
            "make_matrix",
            1,
            2,
            2,
        )
        .expect_err("width mismatch should fail");
        assert!(wrong_width.to_string().contains("expected width 2"));

        let outer_nulls =
            arrays_into_list_array([Arc::new(ListArray::from_iter_primitive::<Float32Type, _, _>(
                vec![Some(vec![Some(1.0_f32), Some(2.0)])],
            )) as ArrayRef])
            .expect("nested list");
        let nested_null = nested_matrix_rows::<Float32Type>(&outer_nulls, "make_matrix", 1, 2, 2)
            .expect_err("nested row mismatch should fail");
        assert!(nested_null.to_string().contains("expected 2 nested rows"));

        assert_eq!(expected_elements("make_tensor", &[2, 3, 4]).expect("expected count"), 24);
        let overflow =
            expected_elements("make_tensor", &[usize::MAX, 2]).expect_err("overflow should fail");
        assert!(overflow.to_string().contains("dimension product overflow"));
    }

    #[test]
    fn empty_builders_cover_supported_and_error_types() {
        let variable_f32 =
            build_empty_variable_tensor_typed("make_variable_tensor", &DataType::Float32, 2)
                .expect("empty variable tensor");
        assert_eq!(variable_f32.len(), 0);

        let variable_f64 =
            build_empty_variable_tensor_typed("make_variable_tensor", &DataType::Float64, 1)
                .expect("empty variable tensor");
        assert_eq!(variable_f64.len(), 0);

        let bad_variable =
            build_empty_variable_tensor_typed("make_variable_tensor", &DataType::Int32, 1)
                .expect_err("unsupported type should fail");
        assert!(bad_variable.to_string().contains("unsupported constructor value type"));

        let rank_overflow = build_empty_variable_tensor_typed(
            "make_variable_tensor",
            &DataType::Float32,
            usize::MAX,
        )
        .expect_err("rank overflow should fail");
        assert!(rank_overflow.to_string().contains("tensor rank"));

        let csr_f32 = build_empty_csr_matrix_batch("make_csr_matrix_batch", &DataType::Float32)
            .expect("empty csr batch");
        assert_eq!(csr_f32.len(), 0);

        let csr_f64 = build_empty_csr_matrix_batch("make_csr_matrix_batch", &DataType::Float64)
            .expect("empty csr batch");
        assert_eq!(csr_f64.len(), 0);

        let bad_csr = build_empty_csr_matrix_batch("make_csr_matrix_batch", &DataType::Int32)
            .expect_err("unsupported type should fail");
        assert!(bad_csr.to_string().contains("unsupported constructor value type"));

        assert_eq!(
            variable_tensor_dims("make_variable_tensor", 0, vec![2, 3], 2).expect("dims"),
            vec![2, 3]
        );
        let rank_error = variable_tensor_dims("make_variable_tensor", 0, vec![2], 2)
            .expect_err("rank mismatch should fail");
        assert!(rank_error.to_string().contains("expected rank 2"));

        let negative_dim = variable_tensor_dims("make_variable_tensor", 0, vec![2, -1], 2)
            .expect_err("negative dim should fail");
        assert!(negative_dim.to_string().contains("negative dimension"));
    }

    #[test]
    fn typed_variable_tensor_and_csr_helpers_cover_empty_and_error_paths() {
        let values = float32_list(vec![vec![1.0, 2.0]]);
        let mismatch = invoke_make_variable_tensor_typed::<Float32Type>(
            "make_variable_tensor",
            &DataType::Float32,
            1,
            &values,
            vec![vec![2], vec![1]],
        )
        .expect_err("length mismatch should fail");
        assert!(mismatch.to_string().contains("argument length mismatch"));

        let wrong_values = invoke_make_variable_tensor_typed::<Float32Type>(
            "make_variable_tensor",
            &DataType::Float32,
            2,
            &values,
            vec![vec![2, 2]],
        )
        .expect_err("shape/value mismatch should fail");
        assert!(wrong_values.to_string().contains("expected 4 values"));

        let empty_variable = invoke_make_variable_tensor_typed::<Float32Type>(
            "make_variable_tensor",
            &DataType::Float32,
            1,
            &float32_list(vec![]),
            vec![],
        )
        .expect("empty variable tensor");
        let ColumnarValue::Array(empty_variable) = empty_variable else {
            panic!("expected array output");
        };
        assert_eq!(empty_variable.len(), 0);

        let shape_error = parse_csr_batch_shapes("make_csr_matrix_batch", vec![vec![2]])
            .expect_err("shape length should fail");
        assert!(shape_error.to_string().contains("expected shape length 2"));

        let negative_shape = parse_csr_batch_shapes("make_csr_matrix_batch", vec![vec![-1, 2]])
            .expect_err("negative shape should fail");
        assert!(negative_shape.to_string().contains("negative row count"));

        let csr_mismatch = invoke_make_csr_matrix_batch_typed::<Float32Type>(
            "make_csr_matrix_batch",
            &DataType::Float32,
            vec![vec![1, 1]],
            vec![vec![0, 1], vec![0, 1]],
            vec![vec![0]],
            &float32_list(vec![vec![1.0]]),
        )
        .expect_err("batch mismatch should fail");
        assert!(csr_mismatch.to_string().contains("argument length mismatch"));

        let empty_csr = invoke_make_csr_matrix_batch_typed::<Float32Type>(
            "make_csr_matrix_batch",
            &DataType::Float32,
            vec![],
            vec![],
            vec![],
            &float32_list(vec![]),
        )
        .expect("empty csr batch");
        let ColumnarValue::Array(empty_csr) = empty_csr else {
            panic!("expected array output");
        };
        assert_eq!(empty_csr.len(), 0);
    }
}
