use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::array::{FixedSizeListArray, PrimitiveArray};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use nabled::core::prelude::NabledReal;
use ndarrow::NdarrowElement;

use super::common::{expect_fixed_size_list_arg, map_arrow_error, nullable_or};
use super::docs::vector_doc;
use crate::error::exec_error;
use crate::metadata::{
    complex_scalar_field, complex_vector_field, parse_complex_vector_field, parse_vector_field,
    scalar_field, vector_field,
};
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

fn invoke_complex_binary_scalar(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl FnOnce(
        &FixedSizeListArray,
        &FixedSizeListArray,
    ) -> Result<
        (datafusion::arrow::datatypes::Field, FixedSizeListArray),
        nabled::arrow::ArrowInteropError,
    >,
) -> Result<ColumnarValue> {
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let (_field, output) =
        op(left, right).map_err(|error| map_arrow_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_complex_unary_scalar(
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

fn invoke_complex_unary_vector(
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Compute the row-wise L2 norm for a batch of dense vectors.",
                "vector_l2_norm(vector_batch)",
            )
            .with_argument(
                "vector_batch",
                "Dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_alternative_syntax("vector_norm(vector_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Compute the row-wise dot product for paired dense vector batches.",
                "vector_dot(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_argument(
                "right_batch",
                "Right dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Compute the row-wise cosine similarity for paired dense vector batches.",
                "vector_cosine_similarity(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_argument(
                "right_batch",
                "Right dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Compute the row-wise cosine distance for paired dense vector batches.",
                "vector_cosine_distance(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_argument(
                "right_batch",
                "Right dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorNormalize {
    signature: Signature,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorDotHermitian {
    signature: Signature,
}

impl VectorDotHermitian {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for VectorDotHermitian {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_dot_hermitian" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (_left_item, left) = parse_complex_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let (_right_item, right) = parse_complex_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.len != right.len {
            return Err(exec_error(
                self.name(),
                format!("complex vector width mismatch: left {}, right {}", left.len, right.len),
            ));
        }
        complex_scalar_field(self.name(), nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (_left_item, left) = parse_complex_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let (_right_item, right) = parse_complex_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.len != right.len {
            return Err(exec_error(
                self.name(),
                format!("complex vector width mismatch: left {}, right {}", left.len, right.len),
            ));
        }
        invoke_complex_binary_scalar(
            &args,
            self.name(),
            nabled::arrow::vector::batched_dot_hermitian,
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Compute the row-wise Hermitian dot product for paired complex vector batches.",
                "vector_dot_hermitian(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left complex vector batch in canonical FixedSizeList<ndarrow.complex64>(D) form.",
            )
            .with_argument(
                "right_batch",
                "Right complex vector batch in canonical FixedSizeList<ndarrow.complex64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorL2NormComplex {
    signature: Signature,
}

impl VectorL2NormComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for VectorL2NormComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_l2_norm_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let _unused = parse_complex_vector_field(&args.arg_fields[0], self.name(), 1)?;
        Ok(scalar_field(self.name(), &DataType::Float64, args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let _unused = parse_complex_vector_field(&args.arg_fields[0], self.name(), 1)?;
        invoke_complex_unary_scalar(
            &args,
            self.name(),
            nabled::arrow::vector::batched_l2_norm_complex,
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Compute the row-wise L2 norm for a batch of complex vectors.",
                "vector_l2_norm_complex(vector_batch)",
            )
            .with_argument(
                "vector_batch",
                "Complex vector batch in canonical FixedSizeList<ndarrow.complex64>(D) form.",
            )
            .with_alternative_syntax("vector_norm_complex(vector_batch)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorCosineSimilarityComplex {
    signature: Signature,
}

impl VectorCosineSimilarityComplex {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for VectorCosineSimilarityComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_cosine_similarity_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (_left_item, left) = parse_complex_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let (_right_item, right) = parse_complex_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.len != right.len {
            return Err(exec_error(
                self.name(),
                format!("complex vector width mismatch: left {}, right {}", left.len, right.len),
            ));
        }
        complex_scalar_field(self.name(), nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (_left_item, left) = parse_complex_vector_field(&args.arg_fields[0], self.name(), 1)?;
        let (_right_item, right) = parse_complex_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if left.len != right.len {
            return Err(exec_error(
                self.name(),
                format!("complex vector width mismatch: left {}, right {}", left.len, right.len),
            ));
        }
        invoke_complex_binary_scalar(
            &args,
            self.name(),
            nabled::arrow::vector::batched_cosine_similarity_complex,
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Compute the row-wise complex cosine similarity for paired complex vector batches.",
                "vector_cosine_similarity_complex(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left complex vector batch in canonical FixedSizeList<ndarrow.complex64>(D) form.",
            )
            .with_argument(
                "right_batch",
                "Right complex vector batch in canonical FixedSizeList<ndarrow.complex64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorNormalizeComplex {
    signature: Signature,
}

impl VectorNormalizeComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for VectorNormalizeComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_normalize_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (_item, contract) = parse_complex_vector_field(&args.arg_fields[0], self.name(), 1)?;
        complex_vector_field(self.name(), contract.len, args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let _unused = parse_complex_vector_field(&args.arg_fields[0], self.name(), 1)?;
        invoke_complex_unary_vector(
            &args,
            self.name(),
            nabled::arrow::vector::batched_normalize_complex,
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Normalize each complex vector in the batch to unit L2 norm.",
                "vector_normalize_complex(vector_batch)",
            )
            .with_argument(
                "vector_batch",
                "Complex vector batch in canonical FixedSizeList<ndarrow.complex64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            vector_doc(
                "Normalize each dense vector in the batch to unit L2 norm.",
                "vector_normalize(vector_batch)",
            )
            .with_argument(
                "vector_batch",
                "Dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
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

#[must_use]
pub fn vector_dot_hermitian_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(VectorDotHermitian::new()))
}

#[must_use]
pub fn vector_l2_norm_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(VectorL2NormComplex::new()).with_aliases(["vector_norm_complex"]),
    )
}

#[must_use]
pub fn vector_cosine_similarity_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(VectorCosineSimilarityComplex::new()))
}

#[must_use]
pub fn vector_normalize_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(VectorNormalizeComplex::new()))
}
