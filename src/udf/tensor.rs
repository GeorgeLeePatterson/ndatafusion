use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::simplify::{ExprSimplifyResult, SimplifyContext};
use datafusion::logical_expr::{
    ColumnarValue, Documentation, Expr, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF,
    ScalarUDFImpl, Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Axis, IxDyn};
use ndarrow::NdarrowElement;

use super::common::{
    expect_fixed_size_list_arg, expect_struct_arg, expect_usize_scalar_arg,
    expect_usize_scalar_argument, fixed_shape_tensor_viewd, map_arrow_error, nullable_or,
};
use super::docs::tensor_doc;
use crate::error::{exec_error, plan_error};
use crate::metadata::{
    complex_fixed_shape_tensor_field, complex_variable_shape_tensor_field,
    fixed_shape_tensor_field, parse_complex_tensor_batch_field,
    parse_complex_variable_shape_tensor_field, parse_tensor_batch_field,
    parse_variable_shape_tensor_field, variable_shape_tensor_field,
};
use crate::signatures::{
    ScalarCoercion, any_signature, coerce_trailing_scalar_arguments, user_defined_signature,
};

fn reduced_shape(function_name: &str, shape: &[usize]) -> Result<Vec<usize>> {
    if shape.len() < 2 {
        return Err(plan_error(
            function_name,
            format!("{function_name} requires tensors with rank >= 2, found shape {shape:?}"),
        ));
    }
    let mut reduced = shape.to_vec();
    let _ = reduced.pop();
    Ok(reduced)
}

fn reduced_uniform_shape(
    function_name: &str,
    rank: usize,
    uniform_shape: Option<Vec<Option<i32>>>,
) -> Result<Option<Vec<Option<i32>>>> {
    if rank < 2 {
        return Err(plan_error(
            function_name,
            format!("{function_name} requires tensors with rank >= 2, found rank {rank}"),
        ));
    }
    Ok(uniform_shape.map(|mut shape| {
        let _ = shape.pop();
        shape
    }))
}

fn validate_variable_pair_ranks(
    function_name: &str,
    left_rank: usize,
    right_rank: usize,
) -> Result<()> {
    if left_rank != right_rank {
        return Err(plan_error(
            function_name,
            format!("tensor rank mismatch: left rank {left_rank}, right rank {right_rank}"),
        ));
    }
    if left_rank < 2 {
        return Err(plan_error(
            function_name,
            format!("{function_name} requires tensors with rank >= 2, found rank {left_rank}"),
        ));
    }
    Ok(())
}

fn permutation_shape(
    function_name: &str,
    shape: &[usize],
    permutation: &[usize],
) -> Result<Vec<usize>> {
    if permutation.len() != shape.len() {
        return Err(plan_error(
            function_name,
            format!(
                "{function_name} requires permutation length {} to match tensor rank {}, found {}",
                shape.len(),
                shape.len(),
                permutation.len()
            ),
        ));
    }
    let mut seen = vec![false; shape.len()];
    let mut output = Vec::with_capacity(shape.len());
    for &axis in permutation {
        if axis >= shape.len() {
            return Err(plan_error(
                function_name,
                format!(
                    "{function_name} axis {axis} is out of bounds for tensor rank {}",
                    shape.len()
                ),
            ));
        }
        if std::mem::replace(&mut seen[axis], true) {
            return Err(plan_error(
                function_name,
                format!("{function_name} permutation contains duplicate axis {axis}"),
            ));
        }
        output.push(shape[axis]);
    }
    Ok(output)
}

fn axis_mask(function_name: &str, label: &str, rank: usize, axes: &[usize]) -> Result<Vec<bool>> {
    let mut mask = vec![false; rank];
    for &axis in axes {
        if axis >= rank {
            return Err(plan_error(
                function_name,
                format!(
                    "{function_name} {label} axis {axis} is out of bounds for tensor rank {rank}"
                ),
            ));
        }
        if std::mem::replace(&mut mask[axis], true) {
            return Err(plan_error(
                function_name,
                format!("{function_name} {label} axes contain duplicate axis {axis}"),
            ));
        }
    }
    Ok(mask)
}

fn contracted_shape(
    function_name: &str,
    left_shape: &[usize],
    right_shape: &[usize],
    left_axes: &[usize],
    right_axes: &[usize],
) -> Result<Vec<usize>> {
    if left_axes.len() != right_axes.len() {
        return Err(plan_error(
            function_name,
            format!(
                "{function_name} requires matching axis counts, found {} left axes and {} right \
                 axes",
                left_axes.len(),
                right_axes.len()
            ),
        ));
    }
    let left_mask = axis_mask(function_name, "left", left_shape.len(), left_axes)?;
    let right_mask = axis_mask(function_name, "right", right_shape.len(), right_axes)?;
    for (&left_axis, &right_axis) in left_axes.iter().zip(right_axes.iter()) {
        if left_shape[left_axis] != right_shape[right_axis] {
            return Err(plan_error(
                function_name,
                format!(
                    "{function_name} axis mismatch: left axis {left_axis} has size {}, right axis \
                     {right_axis} has size {}",
                    left_shape[left_axis], right_shape[right_axis]
                ),
            ));
        }
    }
    let mut output = Vec::with_capacity(
        left_shape.len() + right_shape.len() - left_axes.len() - right_axes.len(),
    );
    for (axis, &dim) in left_shape.iter().enumerate() {
        if !left_mask[axis] {
            output.push(dim);
        }
    }
    for (axis, &dim) in right_shape.iter().enumerate() {
        if !right_mask[axis] {
            output.push(dim);
        }
    }
    Ok(output)
}

fn build_fixed_shape_tensor_output<T>(
    function_name: &str,
    batch: usize,
    shape: &[usize],
    values: Vec<T::Native>,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let mut full_shape = Vec::with_capacity(shape.len() + 1);
    full_shape.push(batch);
    full_shape.extend_from_slice(shape);
    let output = ndarray::ArrayD::from_shape_vec(IxDyn(&full_shape), values)
        .map_err(|error| exec_error(function_name, error))?;
    let (_field, output) = ndarrow::arrayd_to_fixed_shape_tensor(function_name, output)
        .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_tensor_permute_axes_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    tensor: &datafusion::arrow::array::FixedSizeListArray,
    permutation: &[usize],
    output_shape: &[usize],
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let tensor_view = fixed_shape_tensor_viewd::<T>(&args.arg_fields[0], tensor, function_name)?;
    let batch = tensor_view.len_of(Axis(0));
    let mut values = Vec::with_capacity(batch * output_shape.iter().product::<usize>());
    for row in 0..batch {
        let tensor_row = tensor_view.index_axis(Axis(0), row).into_dyn();
        let output = nabled::linalg::tensor::permute_axes_view(&tensor_row, permutation)
            .map_err(|error| exec_error(function_name, error))?;
        values.extend(output.iter().copied());
    }
    build_fixed_shape_tensor_output::<T>(function_name, batch, output_shape, values)
}

fn invoke_tensor_contract_axes_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    left: &datafusion::arrow::array::FixedSizeListArray,
    right: &datafusion::arrow::array::FixedSizeListArray,
    left_axes: &[usize],
    right_axes: &[usize],
    output_shape: &[usize],
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Default,
{
    let left_view = fixed_shape_tensor_viewd::<T>(&args.arg_fields[0], left, function_name)?;
    let right_view = fixed_shape_tensor_viewd::<T>(&args.arg_fields[1], right, function_name)?;
    if left_view.len_of(Axis(0)) != right_view.len_of(Axis(0)) {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} left tensors vs {} right tensors",
                left_view.len_of(Axis(0)),
                right_view.len_of(Axis(0))
            ),
        ));
    }

    let batch = left_view.len_of(Axis(0));
    let mut values = Vec::with_capacity(batch * output_shape.iter().product::<usize>());
    for row in 0..batch {
        let left_row = left_view.index_axis(Axis(0), row).into_dyn();
        let right_row = right_view.index_axis(Axis(0), row).into_dyn();
        let output = nabled::linalg::tensor::contract_axes_view(
            &left_row, &right_row, left_axes, right_axes,
        )
        .map_err(|error| exec_error(function_name, error))?;
        values.extend(output.iter().copied());
    }
    build_fixed_shape_tensor_output::<T>(function_name, batch, output_shape, values)
}

fn permutation_from_return_args(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<Vec<usize>> {
    let axis_count = args.arg_fields.len().saturating_sub(1);
    let mut axes = Vec::with_capacity(axis_count);
    for position in 2..=(axis_count + 1) {
        axes.push(expect_usize_scalar_argument(args, position, function_name)?);
    }
    Ok(axes)
}

fn permutation_from_scalar_args(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<Vec<usize>> {
    let axis_count = args.arg_fields.len().saturating_sub(1);
    let mut axes = Vec::with_capacity(axis_count);
    for position in 2..=(axis_count + 1) {
        axes.push(expect_usize_scalar_arg(args, position, function_name)?);
    }
    Ok(axes)
}

fn contract_axes_from_return_args(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let axis_arg_count = args.arg_fields.len().saturating_sub(2);
    if axis_arg_count == 0 || !axis_arg_count.is_multiple_of(2) {
        return Err(plan_error(
            function_name,
            format!(
                "{function_name} requires one or more left/right axis pairs after the tensor \
                 arguments"
            ),
        ));
    }
    let pair_count = axis_arg_count / 2;
    let mut left_axes = Vec::with_capacity(pair_count);
    let mut right_axes = Vec::with_capacity(pair_count);
    for pair in 0..pair_count {
        left_axes.push(expect_usize_scalar_argument(args, 3 + pair * 2, function_name)?);
        right_axes.push(expect_usize_scalar_argument(args, 4 + pair * 2, function_name)?);
    }
    Ok((left_axes, right_axes))
}

fn contract_axes_from_scalar_args(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<(Vec<usize>, Vec<usize>)> {
    let axis_arg_count = args.arg_fields.len().saturating_sub(2);
    if axis_arg_count == 0 || !axis_arg_count.is_multiple_of(2) {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires one or more left/right axis pairs after the tensor \
                 arguments"
            ),
        ));
    }
    let pair_count = axis_arg_count / 2;
    let mut left_axes = Vec::with_capacity(pair_count);
    let mut right_axes = Vec::with_capacity(pair_count);
    for pair in 0..pair_count {
        left_axes.push(expect_usize_scalar_arg(args, 3 + pair * 2, function_name)?);
        right_axes.push(expect_usize_scalar_arg(args, 4 + pair * 2, function_name)?);
    }
    Ok((left_axes, right_axes))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorSumLastAxis {
    signature: Signature,
}

impl TensorSumLastAxis {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorSumLastAxis {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_sum_last_axis" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let reduced = reduced_shape(self.name(), &contract.shape)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &reduced,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let output = match contract.value_type {
            DataType::Float32 => nabled::arrow::tensor::sum_last_axis::<Float32Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            DataType::Float64 => nabled::arrow::tensor::sum_last_axis::<Float64Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported tensor value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reduce each fixed-shape tensor row by summing over the last axis.",
                "tensor_sum_last_axis(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_sum_last(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorL2NormLastAxis {
    signature: Signature,
}

impl TensorL2NormLastAxis {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorL2NormLastAxis {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_l2_norm_last_axis" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let reduced = reduced_shape(self.name(), &contract.shape)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &reduced,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let output = match contract.value_type {
            DataType::Float32 => nabled::arrow::tensor::l2_norm_last_axis::<Float32Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            DataType::Float64 => nabled::arrow::tensor::l2_norm_last_axis::<Float64Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported tensor value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reduce each fixed-shape tensor row by computing the L2 norm over the last axis.",
                "tensor_l2_norm_last_axis(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_norm_last(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorNormalizeLastAxis {
    signature: Signature,
}

impl TensorNormalizeLastAxis {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorNormalizeLastAxis {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_normalize_last_axis" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &contract.shape,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let output = match contract.value_type {
            DataType::Float32 => nabled::arrow::tensor::normalize_last_axis::<Float32Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            DataType::Float64 => nabled::arrow::tensor::normalize_last_axis::<Float64Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported tensor value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Normalize each fixed-shape tensor row over the last axis.",
                "tensor_normalize_last_axis(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_normalize_last(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorL2NormLastAxisComplex {
    signature: Signature,
}

impl TensorL2NormLastAxisComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorL2NormLastAxisComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_l2_norm_last_axis_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_complex_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let reduced = reduced_shape(self.name(), &contract.shape)?;
        fixed_shape_tensor_field(
            self.name(),
            &DataType::Float64,
            &reduced,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let _contract = parse_complex_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let output =
            nabled::arrow::tensor::l2_norm_last_axis_complex(args.arg_fields[0].as_ref(), tensor)
                .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reduce each fixed-shape complex tensor row by computing the L2 norm over the \
                 last axis.",
                "tensor_l2_norm_last_axis_complex(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Fixed-shape complex tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_norm_last_complex(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorNormalizeLastAxisComplex {
    signature: Signature,
}

impl TensorNormalizeLastAxisComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorNormalizeLastAxisComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_normalize_last_axis_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_complex_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        complex_fixed_shape_tensor_field(
            self.name(),
            &contract.shape,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let _contract = parse_complex_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let output =
            nabled::arrow::tensor::normalize_last_axis_complex(args.arg_fields[0].as_ref(), tensor)
                .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Normalize each fixed-shape complex tensor row over the last axis.",
                "tensor_normalize_last_axis_complex(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Fixed-shape complex tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_normalize_last_complex(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorBatchedDotLastAxis {
    signature: Signature,
}

impl TensorBatchedDotLastAxis {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for TensorBatchedDotLastAxis {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_batched_dot_last_axis" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_tensor_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "tensor value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        if left.shape != right.shape {
            return Err(plan_error(
                self.name(),
                format!("tensor shape mismatch: left {:?}, right {:?}", left.shape, right.shape),
            ));
        }
        let reduced = reduced_shape(self.name(), &left.shape)?;
        fixed_shape_tensor_field(
            self.name(),
            &left.value_type,
            &reduced,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left_contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right_contract = parse_tensor_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left_contract.value_type != right_contract.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "tensor value type mismatch: left {}, right {}",
                    left_contract.value_type, right_contract.value_type
                ),
            ));
        }
        let left = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let right = expect_fixed_size_list_arg(&args, 2, self.name())?;
        let output = match left_contract.value_type {
            DataType::Float32 => nabled::arrow::tensor::batched_dot_last_axis::<Float32Type>(
                args.arg_fields[0].as_ref(),
                left,
                args.arg_fields[1].as_ref(),
                right,
            ),
            DataType::Float64 => nabled::arrow::tensor::batched_dot_last_axis::<Float64Type>(
                args.arg_fields[0].as_ref(),
                left,
                args.arg_fields[1].as_ref(),
                right,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported tensor value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute the row-wise batched dot product for paired fixed-shape tensor batches \
                 over the last axis.",
                "tensor_batched_dot_last_axis(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_argument(
                "right_batch",
                "Right fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_batched_dot_last(left_batch, right_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorBatchedMatmulLastTwo {
    signature: Signature,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorPermuteAxes {
    signature: Signature,
}

impl TensorPermuteAxes {
    fn new() -> Self { Self { signature: user_defined_signature() } }
}

impl ScalarUDFImpl for TensorPermuteAxes {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_permute_axes" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_trailing_scalar_arguments(self.name(), arg_types, 2, ScalarCoercion::Integer)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let permutation = permutation_from_return_args(&args, self.name())?;
        let output_shape = permutation_shape(self.name(), &contract.shape, &permutation)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &output_shape,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let permutation = permutation_from_scalar_args(&args, self.name())?;
        let output_shape = permutation_shape(self.name(), &contract.shape, &permutation)?;
        match contract.value_type {
            DataType::Float32 => invoke_tensor_permute_axes_typed::<Float32Type>(
                &args,
                self.name(),
                tensor,
                &permutation,
                &output_shape,
            ),
            DataType::Float64 => invoke_tensor_permute_axes_typed::<Float64Type>(
                &args,
                self.name(),
                tensor,
                &permutation,
                &output_shape,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn simplify(&self, args: Vec<Expr>, _info: &SimplifyContext) -> Result<ExprSimplifyResult> {
        fn matches_permutation_index(index: usize, expr: &Expr) -> bool {
            match expr {
                Expr::Literal(ScalarValue::Int64(Some(value)), _) => {
                    i64::try_from(index).ok() == Some(*value)
                }
                Expr::Literal(ScalarValue::Int32(Some(value)), _) => {
                    i32::try_from(index).ok() == Some(*value)
                }
                Expr::Literal(ScalarValue::UInt64(Some(value)), _) => {
                    u64::try_from(index).ok() == Some(*value)
                }
                Expr::Literal(ScalarValue::UInt32(Some(value)), _) => {
                    u32::try_from(index).ok() == Some(*value)
                }
                Expr::Literal(ScalarValue::UInt16(Some(value)), _) => {
                    u16::try_from(index).ok() == Some(*value)
                }
                Expr::Literal(ScalarValue::UInt8(Some(value)), _) => {
                    u8::try_from(index).ok() == Some(*value)
                }
                Expr::Literal(ScalarValue::Int16(Some(value)), _) => {
                    i16::try_from(index).ok() == Some(*value)
                }
                Expr::Literal(ScalarValue::Int8(Some(value)), _) => {
                    i8::try_from(index).ok() == Some(*value)
                }
                _ => false,
            }
        }

        let Some((tensor, permutation)) = args.split_first() else {
            return Ok(ExprSimplifyResult::Original(args));
        };
        if permutation
            .iter()
            .enumerate()
            .all(|(index, expr)| matches_permutation_index(index, expr))
        {
            return Ok(ExprSimplifyResult::Simplified(tensor.clone()));
        }
        Ok(ExprSimplifyResult::Original(args))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Permute the axes of each fixed-shape tensor row using the supplied axis order.",
                "tensor_permute_axes(tensor_batch, 1, 0, 2)",
            )
            .with_argument(
                "tensor_batch",
                "Fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_argument(
                "axis_i",
                "Zero-based permutation axes supplied positionally after the tensor batch.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorContractAxes {
    signature: Signature,
}

impl TensorContractAxes {
    fn new() -> Self { Self { signature: user_defined_signature() } }
}

impl ScalarUDFImpl for TensorContractAxes {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_contract_axes" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_trailing_scalar_arguments(self.name(), arg_types, 3, ScalarCoercion::Integer)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_tensor_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "tensor value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        let (left_axes, right_axes) = contract_axes_from_return_args(&args, self.name())?;
        let output_shape =
            contracted_shape(self.name(), &left.shape, &right.shape, &left_axes, &right_axes)?;
        fixed_shape_tensor_field(
            self.name(),
            &left.value_type,
            &output_shape,
            nullable_or(&args.arg_fields[..2]),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left_contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right_contract = parse_tensor_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left_contract.value_type != right_contract.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "tensor value type mismatch: left {}, right {}",
                    left_contract.value_type, right_contract.value_type
                ),
            ));
        }
        let left = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let right = expect_fixed_size_list_arg(&args, 2, self.name())?;
        let (left_axes, right_axes) = contract_axes_from_scalar_args(&args, self.name())?;
        let output_shape = contracted_shape(
            self.name(),
            &left_contract.shape,
            &right_contract.shape,
            &left_axes,
            &right_axes,
        )?;
        match left_contract.value_type {
            DataType::Float32 => invoke_tensor_contract_axes_typed::<Float32Type>(
                &args,
                self.name(),
                left,
                right,
                &left_axes,
                &right_axes,
                &output_shape,
            ),
            DataType::Float64 => invoke_tensor_contract_axes_typed::<Float64Type>(
                &args,
                self.name(),
                left,
                right,
                &left_axes,
                &right_axes,
                &output_shape,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Contract paired fixed-shape tensor rows over the supplied left and right axis \
                 pairs.",
                "tensor_contract_axes(left_batch, right_batch, 1, 0)",
            )
            .with_argument(
                "left_batch",
                "Left fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_argument(
                "right_batch",
                "Right fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_argument(
                "left_axis, right_axis",
                "Zero-based axis pairs supplied positionally after the two tensor operands.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

impl TensorBatchedMatmulLastTwo {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for TensorBatchedMatmulLastTwo {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_batched_matmul_last_two" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_tensor_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "tensor value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        if left.shape.len() < 3 || right.shape.len() < 3 {
            return Err(plan_error(
                self.name(),
                format!(
                    "{} requires tensors with rank >= 3, found left {:?} and right {:?}",
                    self.name(),
                    left.shape,
                    right.shape
                ),
            ));
        }
        if left.shape[..left.shape.len() - 2] != right.shape[..right.shape.len() - 2] {
            return Err(plan_error(
                self.name(),
                format!(
                    "tensor batch-prefix mismatch: left {:?}, right {:?}",
                    &left.shape[..left.shape.len() - 2],
                    &right.shape[..right.shape.len() - 2]
                ),
            ));
        }
        if left.shape[left.shape.len() - 1] != right.shape[right.shape.len() - 2] {
            return Err(plan_error(
                self.name(),
                format!(
                    "incompatible tensor matmul shapes: left {:?}, right {:?}",
                    left.shape, right.shape
                ),
            ));
        }
        let mut output_shape = left.shape[..left.shape.len() - 2].to_vec();
        output_shape.push(left.shape[left.shape.len() - 2]);
        output_shape.push(right.shape[right.shape.len() - 1]);
        fixed_shape_tensor_field(
            self.name(),
            &left.value_type,
            &output_shape,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left_contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right_contract = parse_tensor_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left_contract.value_type != right_contract.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "tensor value type mismatch: left {}, right {}",
                    left_contract.value_type, right_contract.value_type
                ),
            ));
        }
        let left = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let right = expect_fixed_size_list_arg(&args, 2, self.name())?;
        let output = match left_contract.value_type {
            DataType::Float32 => nabled::arrow::tensor::batched_matmul_last_two::<Float32Type>(
                args.arg_fields[0].as_ref(),
                left,
                args.arg_fields[1].as_ref(),
                right,
            ),
            DataType::Float64 => nabled::arrow::tensor::batched_matmul_last_two::<Float64Type>(
                args.arg_fields[0].as_ref(),
                left,
                args.arg_fields[1].as_ref(),
                right,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported tensor value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute the row-wise batched matrix product over the last two axes of paired \
                 fixed-shape tensor batches.",
                "tensor_batched_matmul_last_two(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_argument(
                "right_batch",
                "Right fixed-shape tensor batch in canonical arrow.fixed_shape_tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorVariableSumLastAxis {
    signature: Signature,
}

impl TensorVariableSumLastAxis {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorVariableSumLastAxis {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_variable_sum_last_axis" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let reduced =
            reduced_uniform_shape(self.name(), contract.dimensions, contract.uniform_shape)?;
        variable_shape_tensor_field(
            self.name(),
            &contract.value_type,
            contract.dimensions - 1,
            reduced.as_deref(),
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_struct_arg(&args, 1, self.name())?;
        let output = match contract.value_type {
            DataType::Float32 => nabled::arrow::tensor::sum_last_axis_variable::<Float32Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            DataType::Float64 => nabled::arrow::tensor::sum_last_axis_variable::<Float64Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported variable tensor value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reduce each variable-shape tensor row by summing over the last axis.",
                "tensor_variable_sum_last_axis(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Variable-shape tensor batch in canonical arrow.variable_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_var_sum_last(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorVariableL2NormLastAxis {
    signature: Signature,
}

impl TensorVariableL2NormLastAxis {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorVariableL2NormLastAxis {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_variable_l2_norm_last_axis" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let reduced =
            reduced_uniform_shape(self.name(), contract.dimensions, contract.uniform_shape)?;
        variable_shape_tensor_field(
            self.name(),
            &contract.value_type,
            contract.dimensions - 1,
            reduced.as_deref(),
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_struct_arg(&args, 1, self.name())?;
        let output = match contract.value_type {
            DataType::Float32 => nabled::arrow::tensor::l2_norm_last_axis_variable::<Float32Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            DataType::Float64 => nabled::arrow::tensor::l2_norm_last_axis_variable::<Float64Type>(
                args.arg_fields[0].as_ref(),
                tensor,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported variable tensor value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reduce each variable-shape tensor row by computing the L2 norm over the last \
                 axis.",
                "tensor_variable_l2_norm_last_axis(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Variable-shape tensor batch in canonical arrow.variable_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_var_norm_last(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorVariableNormalizeLastAxis {
    signature: Signature,
}

impl TensorVariableNormalizeLastAxis {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorVariableNormalizeLastAxis {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_variable_normalize_last_axis" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        variable_shape_tensor_field(
            self.name(),
            &contract.value_type,
            contract.dimensions,
            contract.uniform_shape.as_deref(),
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_struct_arg(&args, 1, self.name())?;
        let output =
            match contract.value_type {
                DataType::Float32 => nabled::arrow::tensor::normalize_last_axis_variable::<
                    Float32Type,
                >(args.arg_fields[0].as_ref(), tensor),
                DataType::Float64 => nabled::arrow::tensor::normalize_last_axis_variable::<
                    Float64Type,
                >(args.arg_fields[0].as_ref(), tensor),
                actual => {
                    return Err(exec_error(
                        self.name(),
                        format!("unsupported variable tensor value type {actual}"),
                    ));
                }
            }
            .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Normalize each variable-shape tensor row over the last axis.",
                "tensor_variable_normalize_last_axis(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Variable-shape tensor batch in canonical arrow.variable_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_var_normalize_last(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorVariableL2NormLastAxisComplex {
    signature: Signature,
}

impl TensorVariableL2NormLastAxisComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorVariableL2NormLastAxisComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_variable_l2_norm_last_axis_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract =
            parse_complex_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let reduced =
            reduced_uniform_shape(self.name(), contract.dimensions, contract.uniform_shape)?;
        variable_shape_tensor_field(
            self.name(),
            &DataType::Float64,
            contract.dimensions - 1,
            reduced.as_deref(),
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let _contract =
            parse_complex_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_struct_arg(&args, 1, self.name())?;
        let output = nabled::arrow::tensor::l2_norm_last_axis_variable_complex(
            args.arg_fields[0].as_ref(),
            tensor,
        )
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reduce each variable-shape complex tensor row by computing the L2 norm over the \
                 last axis.",
                "tensor_variable_l2_norm_last_axis_complex(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Variable-shape complex tensor batch in canonical arrow.variable_shape_tensor \
                 form.",
            )
            .with_alternative_syntax("tensor_var_norm_last_complex(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorVariableNormalizeLastAxisComplex {
    signature: Signature,
}

impl TensorVariableNormalizeLastAxisComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for TensorVariableNormalizeLastAxisComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_variable_normalize_last_axis_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract =
            parse_complex_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        complex_variable_shape_tensor_field(
            self.name(),
            contract.dimensions,
            contract.uniform_shape.as_deref(),
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let _contract =
            parse_complex_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let tensor = expect_struct_arg(&args, 1, self.name())?;
        let output = nabled::arrow::tensor::normalize_last_axis_variable_complex(
            args.arg_fields[0].as_ref(),
            tensor,
        )
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Normalize each variable-shape complex tensor row over the last axis.",
                "tensor_variable_normalize_last_axis_complex(tensor_batch)",
            )
            .with_argument(
                "tensor_batch",
                "Variable-shape complex tensor batch in canonical arrow.variable_shape_tensor \
                 form.",
            )
            .with_alternative_syntax("tensor_var_normalize_last_complex(tensor_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorVariableBatchedDotLastAxis {
    signature: Signature,
}

impl TensorVariableBatchedDotLastAxis {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for TensorVariableBatchedDotLastAxis {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_variable_batched_dot_last_axis" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_variable_shape_tensor_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "variable tensor value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        validate_variable_pair_ranks(self.name(), left.dimensions, right.dimensions)?;
        let reduced = reduced_uniform_shape(self.name(), left.dimensions, left.uniform_shape)?;
        variable_shape_tensor_field(
            self.name(),
            &left.value_type,
            left.dimensions - 1,
            reduced.as_deref(),
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left_contract = parse_variable_shape_tensor_field(&args.arg_fields[0], self.name(), 1)?;
        let right_contract =
            parse_variable_shape_tensor_field(&args.arg_fields[1], self.name(), 2)?;
        if left_contract.value_type != right_contract.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "variable tensor value type mismatch: left {}, right {}",
                    left_contract.value_type, right_contract.value_type
                ),
            ));
        }
        let left = expect_struct_arg(&args, 1, self.name())?;
        let right = expect_struct_arg(&args, 2, self.name())?;
        let output = match left_contract.value_type {
            DataType::Float32 => {
                nabled::arrow::tensor::batched_dot_last_axis_variable::<Float32Type>(
                    args.arg_fields[0].as_ref(),
                    left,
                    args.arg_fields[1].as_ref(),
                    right,
                )
            }
            DataType::Float64 => {
                nabled::arrow::tensor::batched_dot_last_axis_variable::<Float64Type>(
                    args.arg_fields[0].as_ref(),
                    left,
                    args.arg_fields[1].as_ref(),
                    right,
                )
            }
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported variable tensor value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute the row-wise batched dot product for paired variable-shape tensor \
                 batches over the last axis.",
                "tensor_variable_batched_dot_last_axis(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left variable-shape tensor batch in canonical arrow.variable_shape_tensor form.",
            )
            .with_argument(
                "right_batch",
                "Right variable-shape tensor batch in canonical arrow.variable_shape_tensor form.",
            )
            .with_alternative_syntax("tensor_var_batched_dot_last(left_batch, right_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[must_use]
pub fn tensor_sum_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorSumLastAxis::new()).with_aliases(["tensor_sum_last"]))
}

#[must_use]
pub fn tensor_l2_norm_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorL2NormLastAxis::new()).with_aliases(["tensor_norm_last"]),
    )
}

#[must_use]
pub fn tensor_l2_norm_last_axis_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorL2NormLastAxisComplex::new())
            .with_aliases(["tensor_norm_last_complex"]),
    )
}

#[must_use]
pub fn tensor_normalize_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorNormalizeLastAxis::new())
            .with_aliases(["tensor_normalize_last"]),
    )
}

#[must_use]
pub fn tensor_normalize_last_axis_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorNormalizeLastAxisComplex::new())
            .with_aliases(["tensor_normalize_last_complex"]),
    )
}

#[must_use]
pub fn tensor_batched_dot_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorBatchedDotLastAxis::new())
            .with_aliases(["tensor_batched_dot_last"]),
    )
}

#[must_use]
pub fn tensor_batched_matmul_last_two_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorBatchedMatmulLastTwo::new()))
}

#[must_use]
pub fn tensor_permute_axes_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorPermuteAxes::new()))
}

#[must_use]
pub fn tensor_contract_axes_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorContractAxes::new()))
}

#[must_use]
pub fn tensor_variable_sum_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorVariableSumLastAxis::new())
            .with_aliases(["tensor_var_sum_last"]),
    )
}

#[must_use]
pub fn tensor_variable_l2_norm_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorVariableL2NormLastAxis::new())
            .with_aliases(["tensor_var_norm_last"]),
    )
}

#[must_use]
pub fn tensor_variable_l2_norm_last_axis_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorVariableL2NormLastAxisComplex::new())
            .with_aliases(["tensor_var_norm_last_complex"]),
    )
}

#[must_use]
pub fn tensor_variable_normalize_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorVariableNormalizeLastAxis::new())
            .with_aliases(["tensor_var_normalize_last"]),
    )
}

#[must_use]
pub fn tensor_variable_normalize_last_axis_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorVariableNormalizeLastAxisComplex::new())
            .with_aliases(["tensor_var_normalize_last_complex"]),
    )
}

#[must_use]
pub fn tensor_variable_batched_dot_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(TensorVariableBatchedDotLastAxis::new())
            .with_aliases(["tensor_var_batched_dot_last"]),
    )
}
