use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};

use super::common::{expect_struct_arg, map_arrow_error, nullable_or};
use crate::metadata::{
    field_like, parse_float64_variable_shape_tensor_field, require_float64_csr_matrix_batch_field,
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
        require_float64_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let vector_contract =
            parse_float64_variable_shape_tensor_field(&args.arg_fields[1], self.name(), 2)?;
        if vector_contract.dimensions != 1 {
            return Err(crate::error::plan_error(
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
            1,
            Some(&uniform_shape),
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_struct_arg(&args, 1, self.name())?;
        let vectors = expect_struct_arg(&args, 2, self.name())?;
        let (_field, output) = nabled::arrow::sparse::matvec_csr_batch_extension::<Float64Type>(
            args.arg_fields[0].as_ref(),
            matrices,
            args.arg_fields[1].as_ref(),
            vectors,
        )
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
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
        require_float64_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let matrix_contract =
            parse_float64_variable_shape_tensor_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix_contract.dimensions != 2 {
            return Err(crate::error::plan_error(
                self.name(),
                format!(
                    "argument 2 must be a batch of rank-2 dense matrices, found rank {}",
                    matrix_contract.dimensions
                ),
            ));
        }
        let uniform_shape = [None, None];
        variable_shape_tensor_field(
            self.name(),
            2,
            Some(&uniform_shape),
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_struct_arg(&args, 1, self.name())?;
        let right = expect_struct_arg(&args, 2, self.name())?;
        let (_field, output) =
            nabled::arrow::sparse::matmat_dense_csr_batch_extension::<Float64Type>(
                args.arg_fields[0].as_ref(),
                matrices,
                args.arg_fields[1].as_ref(),
                right,
            )
            .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
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
        require_float64_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        Ok(field_like(self.name(), &args.arg_fields[0], args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_struct_arg(&args, 1, self.name())?;
        let (_field, output) = nabled::arrow::sparse::transpose_csr_batch_extension::<Float64Type>(
            args.arg_fields[0].as_ref(),
            matrices,
        )
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
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
        require_float64_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        require_float64_csr_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        Ok(field_like(self.name(), &args.arg_fields[0], nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = expect_struct_arg(&args, 1, self.name())?;
        let right = expect_struct_arg(&args, 2, self.name())?;
        let (_field, output) =
            nabled::arrow::sparse::matmat_sparse_csr_batch_extension::<Float64Type>(
                args.arg_fields[0].as_ref(),
                left,
                args.arg_fields[1].as_ref(),
                right,
            )
            .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
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
