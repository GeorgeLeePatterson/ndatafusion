use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};

use super::common::{expect_struct_arg, map_arrow_error, nullable_or};
use crate::error::{exec_error, plan_error};
use crate::metadata::{
    field_like, parse_csr_matrix_batch_field, parse_variable_shape_tensor_field,
    variable_shape_tensor_field,
};
use crate::signatures::any_signature;

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseMatvec {
    signature: Signature,
}

impl SparseMatvec {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for SparseMatvec {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_matvec" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let vector_contract =
            parse_variable_shape_tensor_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix_type != vector_contract.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {}, vector {}",
                    matrix_type, vector_contract.value_type
                ),
            ));
        }
        if vector_contract.dimensions != 1 {
            return Err(plan_error(
                self.name(),
                format!(
                    "argument 2 must be a batch of rank-1 dense vectors, found rank {}",
                    vector_contract.dimensions
                ),
            ));
        }
        let uniform_shape = [None];
        variable_shape_tensor_field(
            self.name(),
            &vector_contract.value_type,
            1,
            Some(&uniform_shape),
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let value_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let matrices = expect_struct_arg(&args, 1, self.name())?;
        let vectors = expect_struct_arg(&args, 2, self.name())?;
        let output = match value_type {
            DataType::Float32 => nabled::arrow::sparse::matvec_csr_batch_extension::<Float32Type>(
                args.arg_fields[0].as_ref(),
                matrices,
                args.arg_fields[1].as_ref(),
                vectors,
            ),
            DataType::Float64 => nabled::arrow::sparse::matvec_csr_batch_extension::<Float64Type>(
                args.arg_fields[0].as_ref(),
                matrices,
                args.arg_fields[1].as_ref(),
                vectors,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported sparse matrix value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseMatmatDense {
    signature: Signature,
}

impl SparseMatmatDense {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for SparseMatmatDense {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_matmat_dense" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let dense_contract =
            parse_variable_shape_tensor_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix_type != dense_contract.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "value type mismatch: sparse {}, dense {}",
                    matrix_type, dense_contract.value_type
                ),
            ));
        }
        if dense_contract.dimensions != 2 {
            return Err(plan_error(
                self.name(),
                format!(
                    "argument 2 must be a batch of rank-2 dense matrices, found rank {}",
                    dense_contract.dimensions
                ),
            ));
        }
        let uniform_shape = [None, None];
        variable_shape_tensor_field(
            self.name(),
            &dense_contract.value_type,
            2,
            Some(&uniform_shape),
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let value_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let matrices = expect_struct_arg(&args, 1, self.name())?;
        let right = expect_struct_arg(&args, 2, self.name())?;
        let output = match value_type {
            DataType::Float32 => {
                nabled::arrow::sparse::matmat_dense_csr_batch_extension::<Float32Type>(
                    args.arg_fields[0].as_ref(),
                    matrices,
                    args.arg_fields[1].as_ref(),
                    right,
                )
            }
            DataType::Float64 => {
                nabled::arrow::sparse::matmat_dense_csr_batch_extension::<Float64Type>(
                    args.arg_fields[0].as_ref(),
                    matrices,
                    args.arg_fields[1].as_ref(),
                    right,
                )
            }
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported sparse matrix value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseTranspose {
    signature: Signature,
}

impl SparseTranspose {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for SparseTranspose {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_transpose" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        drop(parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?);
        Ok(field_like(self.name(), &args.arg_fields[0], args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let value_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let matrices = expect_struct_arg(&args, 1, self.name())?;
        let output =
            match value_type {
                DataType::Float32 => nabled::arrow::sparse::transpose_csr_batch_extension::<
                    Float32Type,
                >(args.arg_fields[0].as_ref(), matrices),
                DataType::Float64 => nabled::arrow::sparse::transpose_csr_batch_extension::<
                    Float64Type,
                >(args.arg_fields[0].as_ref(), matrices),
                actual => {
                    return Err(exec_error(
                        self.name(),
                        format!("unsupported sparse matrix value type {actual}"),
                    ));
                }
            }
            .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseMatmatSparse {
    signature: Signature,
}

impl SparseMatmatSparse {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for SparseMatmatSparse {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_matmat_sparse" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right_type = parse_csr_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left_type != right_type {
            return Err(plan_error(
                self.name(),
                format!("sparse matrix value type mismatch: left {left_type}, right {right_type}"),
            ));
        }
        Ok(field_like(self.name(), &args.arg_fields[0], nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right_type = parse_csr_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left_type != right_type {
            return Err(exec_error(
                self.name(),
                format!("sparse matrix value type mismatch: left {left_type}, right {right_type}"),
            ));
        }
        let left = expect_struct_arg(&args, 1, self.name())?;
        let right = expect_struct_arg(&args, 2, self.name())?;
        let output = match left_type {
            DataType::Float32 => {
                nabled::arrow::sparse::matmat_sparse_csr_batch_extension::<Float32Type>(
                    args.arg_fields[0].as_ref(),
                    left,
                    args.arg_fields[1].as_ref(),
                    right,
                )
            }
            DataType::Float64 => {
                nabled::arrow::sparse::matmat_sparse_csr_batch_extension::<Float64Type>(
                    args.arg_fields[0].as_ref(),
                    left,
                    args.arg_fields[1].as_ref(),
                    right,
                )
            }
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported sparse matrix value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }
}

#[must_use]
pub fn sparse_matvec_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseMatvec::new()))
}

#[must_use]
pub fn sparse_matmat_dense_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseMatmatDense::new()))
}

#[must_use]
pub fn sparse_transpose_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseTranspose::new()))
}

#[must_use]
pub fn sparse_matmat_sparse_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseMatmatSparse::new()))
}
