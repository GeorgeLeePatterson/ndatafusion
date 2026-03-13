use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::array::{FixedSizeListArray, PrimitiveArray};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};

use super::common::{expect_fixed_size_list_arg, map_arrow_error, nullable_or};
use crate::metadata::{float64_scalar_field, parse_float64_vector_field, vector_field};
use crate::signatures::float64_vector_signature;

fn invoke_unary_scalar(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl FnOnce(
        &FixedSizeListArray,
    ) -> Result<PrimitiveArray<Float64Type>, nabled::arrow::ArrowInteropError>,
) -> Result<ColumnarValue> {
    let rows = expect_fixed_size_list_arg(args, 1, function_name)?;
    let output = op(rows).map_err(|error| map_arrow_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_binary_scalar(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl FnOnce(
        &FixedSizeListArray,
        &FixedSizeListArray,
    ) -> Result<PrimitiveArray<Float64Type>, nabled::arrow::ArrowInteropError>,
) -> Result<ColumnarValue> {
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let output = op(left, right).map_err(|error| map_arrow_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_unary_vector(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl FnOnce(&FixedSizeListArray) -> Result<FixedSizeListArray, nabled::arrow::ArrowInteropError>,
) -> Result<ColumnarValue> {
    let rows = expect_fixed_size_list_arg(args, 1, function_name)?;
    let output = op(rows).map_err(|error| map_arrow_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorL2Norm {
    signature: Signature,
}

impl VectorL2Norm {
    fn new() -> Self { Self { signature: float64_vector_signature(1) } }
}

impl ScalarUDFImpl for VectorL2Norm {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_l2_norm" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let _ = parse_float64_vector_field(&args.arg_fields[0], self.name(), 1)?;
        Ok(float64_scalar_field(self.name(), args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_unary_scalar(&args, self.name(), |rows| {
            nabled::arrow::vector::batched_l2_norm::<Float64Type>(rows)
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorDot {
    signature: Signature,
}

impl VectorDot {
    fn new() -> Self { Self { signature: float64_vector_signature(2) } }
}

impl ScalarUDFImpl for VectorDot {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_dot" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let _ = parse_float64_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let _ = parse_float64_vector_field(&args.arg_fields[1], self.name(), 2)?;
        Ok(float64_scalar_field(self.name(), nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_binary_scalar(&args, self.name(), |left, right| {
            nabled::arrow::vector::batched_dot::<Float64Type>(left, right)
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorCosineSimilarity {
    signature: Signature,
}

impl VectorCosineSimilarity {
    fn new() -> Self { Self { signature: float64_vector_signature(2) } }
}

impl ScalarUDFImpl for VectorCosineSimilarity {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_cosine_similarity" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let _ = parse_float64_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let _ = parse_float64_vector_field(&args.arg_fields[1], self.name(), 2)?;
        Ok(float64_scalar_field(self.name(), nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_binary_scalar(&args, self.name(), |left, right| {
            nabled::arrow::vector::batched_cosine_similarity::<Float64Type>(left, right)
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorCosineDistance {
    signature: Signature,
}

impl VectorCosineDistance {
    fn new() -> Self { Self { signature: float64_vector_signature(2) } }
}

impl ScalarUDFImpl for VectorCosineDistance {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_cosine_distance" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let _ = parse_float64_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let _ = parse_float64_vector_field(&args.arg_fields[1], self.name(), 2)?;
        Ok(float64_scalar_field(self.name(), nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_binary_scalar(&args, self.name(), |left, right| {
            nabled::arrow::vector::batched_cosine_distance::<Float64Type>(left, right)
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorNormalize {
    signature: Signature,
}

impl VectorNormalize {
    fn new() -> Self { Self { signature: float64_vector_signature(1) } }
}

impl ScalarUDFImpl for VectorNormalize {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_normalize" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let len = parse_float64_vector_field(&args.arg_fields[0], self.name(), 1)?;
        vector_field(self.name(), len, args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_unary_vector(&args, self.name(), |rows| {
            nabled::arrow::vector::batched_normalize::<Float64Type>(rows)
        })
    }
}

#[must_use]
pub fn vector_l2_norm_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(VectorL2Norm::new()))
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
