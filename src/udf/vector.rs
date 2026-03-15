use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::array::{FixedSizeListArray, PrimitiveArray};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use nabled::core::prelude::NabledReal;
use ndarrow::NdarrowElement;

use super::common::{expect_fixed_size_list_arg, map_arrow_error, nullable_or};
use crate::error::exec_error;
use crate::metadata::{parse_vector_field, scalar_field, vector_field};
use crate::signatures::any_signature;

fn invoke_unary_scalar<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl FnOnce(&FixedSizeListArray) -> Result<PrimitiveArray<T>, nabled::arrow::ArrowInteropError>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let rows = expect_fixed_size_list_arg(args, 1, function_name)?;
    let output = op(rows).map_err(|error| map_arrow_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_binary_scalar<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl FnOnce(
        &FixedSizeListArray,
        &FixedSizeListArray,
    ) -> Result<PrimitiveArray<T>, nabled::arrow::ArrowInteropError>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let output = op(left, right).map_err(|error| map_arrow_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_unary_vector<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl FnOnce(&FixedSizeListArray) -> Result<FixedSizeListArray, nabled::arrow::ArrowInteropError>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let rows = expect_fixed_size_list_arg(args, 1, function_name)?;
    let output = op(rows).map_err(|error| map_arrow_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorL2Norm {
    signature: Signature,
}

impl VectorL2Norm {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for VectorL2Norm {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_l2_norm" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_vector_field(&args.arg_fields[0], self.name(), 1)?;
        Ok(scalar_field(self.name(), &contract.value_type, args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_vector_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => invoke_unary_scalar::<Float32Type>(&args, self.name(), |rows| {
                nabled::arrow::vector::batched_l2_norm::<Float32Type>(rows)
            }),
            DataType::Float64 => invoke_unary_scalar::<Float64Type>(&args, self.name(), |rows| {
                nabled::arrow::vector::batched_l2_norm::<Float64Type>(rows)
            }),
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorDot {
    signature: Signature,
}

impl VectorDot {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for VectorDot {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_dot" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "vector value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        Ok(scalar_field(self.name(), &left.value_type, nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = parse_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "vector value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        match left.value_type {
            DataType::Float32 => {
                invoke_binary_scalar::<Float32Type>(&args, self.name(), |lhs, rhs| {
                    nabled::arrow::vector::batched_dot::<Float32Type>(lhs, rhs)
                })
            }
            DataType::Float64 => {
                invoke_binary_scalar::<Float64Type>(&args, self.name(), |lhs, rhs| {
                    nabled::arrow::vector::batched_dot::<Float64Type>(lhs, rhs)
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorCosineSimilarity {
    signature: Signature,
}

impl VectorCosineSimilarity {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for VectorCosineSimilarity {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_cosine_similarity" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "vector value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        Ok(scalar_field(self.name(), &left.value_type, nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = parse_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "vector value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        match left.value_type {
            DataType::Float32 => {
                invoke_binary_scalar::<Float32Type>(&args, self.name(), |lhs, rhs| {
                    nabled::arrow::vector::batched_cosine_similarity::<Float32Type>(lhs, rhs)
                })
            }
            DataType::Float64 => {
                invoke_binary_scalar::<Float64Type>(&args, self.name(), |lhs, rhs| {
                    nabled::arrow::vector::batched_cosine_similarity::<Float64Type>(lhs, rhs)
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorCosineDistance {
    signature: Signature,
}

impl VectorCosineDistance {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for VectorCosineDistance {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_cosine_distance" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "vector value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        Ok(scalar_field(self.name(), &left.value_type, nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = parse_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "vector value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        match left.value_type {
            DataType::Float32 => {
                invoke_binary_scalar::<Float32Type>(&args, self.name(), |lhs, rhs| {
                    nabled::arrow::vector::batched_cosine_distance::<Float32Type>(lhs, rhs)
                })
            }
            DataType::Float64 => {
                invoke_binary_scalar::<Float64Type>(&args, self.name(), |lhs, rhs| {
                    nabled::arrow::vector::batched_cosine_distance::<Float64Type>(lhs, rhs)
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorNormalize {
    signature: Signature,
}

impl VectorNormalize {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for VectorNormalize {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_normalize" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_vector_field(&args.arg_fields[0], self.name(), 1)?;
        vector_field(
            self.name(),
            &contract.value_type,
            contract.len,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_vector_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => invoke_unary_vector::<Float32Type>(&args, self.name(), |rows| {
                nabled::arrow::vector::batched_normalize::<Float32Type>(rows)
            }),
            DataType::Float64 => invoke_unary_vector::<Float64Type>(&args, self.name(), |rows| {
                nabled::arrow::vector::batched_normalize::<Float64Type>(rows)
            }),
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }
}

#[must_use]
pub fn vector_l2_norm_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(VectorL2Norm::new()).with_aliases(["vector_norm"]))
}

#[must_use]
pub fn vector_dot_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(VectorDot::new())) }

#[must_use]
pub fn vector_cosine_similarity_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(VectorCosineSimilarity::new()))
}

#[must_use]
pub fn vector_cosine_distance_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(VectorCosineDistance::new()))
}

#[must_use]
pub fn vector_normalize_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(VectorNormalize::new()))
}
