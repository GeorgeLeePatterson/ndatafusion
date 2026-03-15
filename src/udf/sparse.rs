use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::Array;
use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use ndarray::Ix1;

use super::common::{expect_struct_arg, map_arrow_error, nullable_or};
use super::docs::sparse_doc;
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Multiply each CSR sparse matrix in the batch by a dense vector in the matching \
                 row.",
                "sparse_matvec(sparse_batch, vector_batch)",
            )
            .with_argument(
                "sparse_batch",
                "Canonical ndarrow.csr_matrix_batch column containing one sparse matrix per row.",
            )
            .with_argument(
                "vector_batch",
                "Canonical variable-shape tensor batch containing rank-1 dense vectors.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Multiply each CSR sparse matrix in the batch by a dense matrix in the matching \
                 row.",
                "sparse_matmat_dense(sparse_batch, dense_batch)",
            )
            .with_argument(
                "sparse_batch",
                "Canonical ndarrow.csr_matrix_batch column containing one sparse matrix per row.",
            )
            .with_argument(
                "dense_batch",
                "Canonical variable-shape tensor batch containing rank-2 dense matrices.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

fn return_sparse_vector_output_field(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<FieldRef> {
    let matrix_type = parse_csr_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    let vector_contract = parse_variable_shape_tensor_field(&args.arg_fields[1], function_name, 2)?;
    if matrix_type != vector_contract.value_type {
        return Err(plan_error(
            function_name,
            format!(
                "value type mismatch: matrix {}, vector {}",
                matrix_type, vector_contract.value_type
            ),
        ));
    }
    if vector_contract.dimensions != 1 {
        return Err(plan_error(
            function_name,
            format!(
                "argument 2 must be a batch of rank-1 dense vectors, found rank {}",
                vector_contract.dimensions
            ),
        ));
    }
    variable_shape_tensor_field(
        function_name,
        &vector_contract.value_type,
        1,
        Some(&[None]),
        nullable_or(args.arg_fields),
    )
}

fn invoke_sparse_lu_solve_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: nabled::core::prelude::NabledReal + ndarrow::NdarrowElement,
{
    let matrices = expect_struct_arg(args, 1, function_name)?;
    let vectors = expect_struct_arg(args, 2, function_name)?;
    if matrices.len() != vectors.len() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} sparse matrices vs {} dense vectors",
                matrices.len(),
                vectors.len()
            ),
        ));
    }

    let mut outputs = Vec::with_capacity(matrices.len());
    let mut vector_iter =
        ndarrow::variable_shape_tensor_iter::<T>(args.arg_fields[1].as_ref(), vectors)
            .map_err(|error| exec_error(function_name, error))?;
    for matrix_row in ndarrow::csr_matrix_batch_iter::<T>(args.arg_fields[0].as_ref(), matrices)
        .map_err(|error| exec_error(function_name, error))?
    {
        let (_, matrix_view) = matrix_row.map_err(|error| exec_error(function_name, error))?;
        let (_, vector_view) = vector_iter
            .next()
            .ok_or_else(|| exec_error(function_name, "dense vector batch iterator ended early"))?
            .map_err(|error| exec_error(function_name, error))?;
        let vector_view = vector_view
            .into_dimensionality::<Ix1>()
            .map_err(|error| exec_error(function_name, error))?;
        let matrix_view = nabled::linalg::sparse::CsrMatrixView::new(
            matrix_view.nrows,
            matrix_view.ncols,
            matrix_view.row_ptrs,
            matrix_view.col_indices,
            matrix_view.values,
        )
        .map_err(|error| exec_error(function_name, error))?;
        let solved = nabled::linalg::sparse::sparse_lu_solve_view(&matrix_view, &vector_view)
            .map_err(|error| exec_error(function_name, error))?;
        outputs.push(solved.into_dyn());
    }
    let (_field, output) =
        ndarrow::arrays_to_variable_shape_tensor(function_name, outputs, Some(vec![None]))
            .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Transpose each CSR sparse matrix in the batch.",
                "sparse_transpose(sparse_batch)",
            )
            .with_argument(
                "sparse_batch",
                "Canonical ndarrow.csr_matrix_batch column containing one sparse matrix per row.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
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

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Multiply paired CSR sparse matrices row by row.",
                "sparse_matmat_sparse(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left canonical ndarrow.csr_matrix_batch column containing one sparse matrix per \
                 row.",
            )
            .with_argument(
                "right_batch",
                "Right canonical ndarrow.csr_matrix_batch column containing one sparse matrix per \
                 row.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct SparseLuSolve {
    signature: Signature,
}

impl SparseLuSolve {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for SparseLuSolve {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "sparse_lu_solve" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        return_sparse_vector_output_field(&args, self.name())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let value_type = parse_csr_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let vector_contract =
            parse_variable_shape_tensor_field(&args.arg_fields[1], self.name(), 2)?;
        if value_type != vector_contract.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {}, vector {}",
                    value_type, vector_contract.value_type
                ),
            ));
        }
        if vector_contract.dimensions != 1 {
            return Err(exec_error(
                self.name(),
                format!(
                    "argument 2 must be a batch of rank-1 dense vectors, found rank {}",
                    vector_contract.dimensions
                ),
            ));
        }
        match value_type {
            DataType::Float32 => invoke_sparse_lu_solve_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_sparse_lu_solve_typed::<Float64Type>(&args, self.name()),
            actual => Err(exec_error(
                self.name(),
                format!("unsupported sparse matrix value type {actual}"),
            )),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            sparse_doc(
                "Solve each square CSR sparse system in the batch with a dense vector right-hand \
                 side.",
                "sparse_lu_solve(sparse_batch, vector_batch)",
            )
            .with_argument(
                "sparse_batch",
                "Canonical ndarrow.csr_matrix_batch column containing one sparse matrix per row.",
            )
            .with_argument(
                "vector_batch",
                "Canonical variable-shape tensor batch containing rank-1 dense vectors.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
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

#[must_use]
pub fn sparse_lu_solve_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(SparseLuSolve::new()))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::datatypes::DataType;
    use datafusion::logical_expr::ReturnFieldArgs;

    use super::*;
    use crate::metadata::{parse_variable_shape_tensor_field, variable_shape_tensor_field};

    #[test]
    fn sparse_vector_output_field_validates_value_type_and_rank() {
        let (sparse_field, _) = ndarrow::csr_batch_to_extension_array(
            "sparse",
            vec![[2, 2]],
            vec![vec![0, 1, 2]],
            vec![vec![0, 1]],
            vec![vec![1.0, 2.0]],
        )
        .expect("sparse batch");
        let sparse_field = Arc::new(sparse_field);
        let vectors =
            variable_shape_tensor_field("rhs", &DataType::Float64, 1, Some(&[None]), false)
                .expect("vector batch field");
        let scalar_arguments = [None, None];
        let args = ReturnFieldArgs {
            arg_fields:       &[Arc::clone(&sparse_field), Arc::clone(&vectors)],
            scalar_arguments: &scalar_arguments,
        };
        let output =
            return_sparse_vector_output_field(&args, "sparse_lu_solve").expect("return field");
        let contract =
            parse_variable_shape_tensor_field(&output, "sparse_lu_solve", 1).expect("contract");
        assert_eq!(contract.value_type, DataType::Float64);
        assert_eq!(contract.dimensions, 1);

        let rank_two =
            variable_shape_tensor_field("rhs", &DataType::Float64, 2, Some(&[None, None]), false)
                .expect("matrix batch field");
        let rank_two_args = ReturnFieldArgs {
            arg_fields:       &[Arc::clone(&sparse_field), rank_two],
            scalar_arguments: &scalar_arguments,
        };
        let error = return_sparse_vector_output_field(&rank_two_args, "sparse_lu_solve")
            .expect_err("rank mismatch should fail");
        assert!(error.to_string().contains("rank-1 dense vectors"), "unexpected error: {error}");

        let float32_vectors =
            variable_shape_tensor_field("rhs", &DataType::Float32, 1, Some(&[None]), false)
                .expect("vector batch field");
        let mismatch_args = ReturnFieldArgs {
            arg_fields:       &[sparse_field, float32_vectors],
            scalar_arguments: &scalar_arguments,
        };
        let error = return_sparse_vector_output_field(&mismatch_args, "sparse_lu_solve")
            .expect_err("value type mismatch should fail");
        assert!(error.to_string().contains("value type mismatch"), "unexpected error: {error}");
    }
}
