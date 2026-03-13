use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};

use super::common::{expect_fixed_size_list_arg, expect_struct_arg, map_arrow_error};
use crate::error::{exec_error, plan_error};
use crate::metadata::{
    fixed_shape_tensor_field, parse_tensor_batch_field, parse_variable_shape_tensor_field,
    variable_shape_tensor_field,
};
use crate::signatures::any_signature;

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
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorBatchedMatmulLastTwo {
    signature: Signature,
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
}

#[must_use]
pub fn tensor_sum_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorSumLastAxis::new()))
}

#[must_use]
pub fn tensor_l2_norm_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorL2NormLastAxis::new()))
}

#[must_use]
pub fn tensor_normalize_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorNormalizeLastAxis::new()))
}

#[must_use]
pub fn tensor_batched_dot_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorBatchedDotLastAxis::new()))
}

#[must_use]
pub fn tensor_batched_matmul_last_two_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorBatchedMatmulLastTwo::new()))
}

#[must_use]
pub fn tensor_variable_sum_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorVariableSumLastAxis::new()))
}

#[must_use]
pub fn tensor_variable_l2_norm_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorVariableL2NormLastAxis::new()))
}

#[must_use]
pub fn tensor_variable_normalize_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorVariableNormalizeLastAxis::new()))
}

#[must_use]
pub fn tensor_variable_batched_dot_last_axis_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorVariableBatchedDotLastAxis::new()))
}
