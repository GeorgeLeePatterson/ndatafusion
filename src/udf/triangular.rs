use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::FixedSizeListArray;
use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Array3, Axis};
use ndarrow::NdarrowElement;

use super::common::{
    expect_fixed_size_list_arg, fixed_shape_tensor_view3, fixed_size_list_array_from_flat_rows,
    fixed_size_list_view2, nullable_or,
};
use crate::error::exec_error;
use crate::metadata::{
    fixed_shape_tensor_field, parse_matrix_batch_field, parse_vector_field, vector_field,
};
use crate::signatures::any_signature;

fn validate_square_matrix_vector_contract(
    function_name: &str,
    matrix: &crate::metadata::MatrixBatchContract,
    rhs: &crate::metadata::VectorContract,
) -> Result<()> {
    if matrix.value_type != rhs.value_type {
        return Err(exec_error(
            function_name,
            format!("value type mismatch: matrix {}, rhs {}", matrix.value_type, rhs.value_type),
        ));
    }
    if matrix.rows != matrix.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square matrices, found ({}, {})",
                matrix.rows, matrix.cols
            ),
        ));
    }
    if rhs.len != matrix.cols {
        return Err(exec_error(
            function_name,
            format!("rhs vector length mismatch: expected {}, found {}", matrix.cols, rhs.len),
        ));
    }
    Ok(())
}

fn validate_square_matrix_rhs_contract(
    function_name: &str,
    matrix: &crate::metadata::MatrixBatchContract,
    rhs: &crate::metadata::MatrixBatchContract,
) -> Result<()> {
    if matrix.value_type != rhs.value_type {
        return Err(exec_error(
            function_name,
            format!("value type mismatch: matrix {}, rhs {}", matrix.value_type, rhs.value_type),
        ));
    }
    if matrix.rows != matrix.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square matrices, found ({}, {})",
                matrix.rows, matrix.cols
            ),
        ));
    }
    if rhs.rows != matrix.rows {
        return Err(exec_error(
            function_name,
            format!("rhs matrix row mismatch: expected {}, found {}", matrix.rows, rhs.rows),
        ));
    }
    Ok(())
}

fn tensor_batch_from_flat_values<T>(
    function_name: &str,
    batch: usize,
    rows: usize,
    cols: usize,
    values: Vec<T::Native>,
) -> Result<FixedSizeListArray>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let tensor = Array3::from_shape_vec((batch, rows, cols), values)
        .map_err(|error| exec_error(function_name, error))?;
    let (_field, array) = ndarrow::arrayd_to_fixed_shape_tensor(function_name, tensor.into_dyn())
        .map_err(|error| exec_error(function_name, error))?;
    Ok(array)
}

fn invoke_triangular_vector_solver<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    solve: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
        &ndarray::ArrayView1<'_, T::Native>,
    ) -> std::result::Result<ndarray::Array1<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let rhs = expect_fixed_size_list_arg(args, 2, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let rhs_view = fixed_size_list_view2::<T>(rhs, function_name)?;
    if matrix_view.len_of(Axis(0)) != rhs_view.nrows() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} matrices vs {} rhs vectors",
                matrix_view.len_of(Axis(0)),
                rhs_view.nrows()
            ),
        ));
    }

    let mut output = Vec::with_capacity(rhs_view.len());
    for row in 0..matrix_view.len_of(Axis(0)) {
        let solution =
            solve(&matrix_view.index_axis(Axis(0), row), &rhs_view.index_axis(Axis(0), row))
                .map_err(|error| exec_error(function_name, error))?;
        output.extend(solution.iter().copied());
    }
    let output = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        rhs_view.nrows(),
        rhs_view.ncols(),
        &output,
    )?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_triangular_matrix_solver<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    solve: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<ndarray::Array2<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let rhs = expect_fixed_size_list_arg(args, 2, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let rhs_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[1], rhs, function_name)?;
    if matrix_view.len_of(Axis(0)) != rhs_view.len_of(Axis(0)) {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} matrices vs {} rhs matrices",
                matrix_view.len_of(Axis(0)),
                rhs_view.len_of(Axis(0))
            ),
        ));
    }
    if matrix_view.len_of(Axis(1)) != rhs_view.len_of(Axis(1)) {
        return Err(exec_error(
            function_name,
            format!(
                "rhs matrix row mismatch: expected {}, found {}",
                matrix_view.len_of(Axis(1)),
                rhs_view.len_of(Axis(1))
            ),
        ));
    }

    let batch = matrix_view.len_of(Axis(0));
    let rhs_cols = rhs_view.len_of(Axis(2));
    let mut output = Vec::with_capacity(batch * matrix_view.len_of(Axis(2)) * rhs_cols);
    for row in 0..batch {
        let solution =
            solve(&matrix_view.index_axis(Axis(0), row), &rhs_view.index_axis(Axis(0), row))
                .map_err(|error| exec_error(function_name, error))?;
        output.extend(solution.iter().copied());
    }
    let output = tensor_batch_from_flat_values::<T>(
        function_name,
        batch,
        matrix_view.len_of(Axis(2)),
        rhs_cols,
        output,
    )?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSolveLower {
    signature: Signature,
}

impl MatrixSolveLower {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixSolveLower {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_solve_lower" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        validate_square_matrix_vector_contract(self.name(), &matrix, &rhs)?;
        vector_field(self.name(), &matrix.value_type, matrix.cols, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        validate_square_matrix_vector_contract(self.name(), &matrix, &rhs)?;
        match matrix.value_type {
            DataType::Float32 => invoke_triangular_vector_solver::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::triangular::solve_lower_view,
            ),
            DataType::Float64 => invoke_triangular_vector_solver::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::triangular::solve_lower_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSolveUpper {
    signature: Signature,
}

impl MatrixSolveUpper {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixSolveUpper {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_solve_upper" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        validate_square_matrix_vector_contract(self.name(), &matrix, &rhs)?;
        vector_field(self.name(), &matrix.value_type, matrix.cols, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        validate_square_matrix_vector_contract(self.name(), &matrix, &rhs)?;
        match matrix.value_type {
            DataType::Float32 => invoke_triangular_vector_solver::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::triangular::solve_upper_view,
            ),
            DataType::Float64 => invoke_triangular_vector_solver::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::triangular::solve_upper_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSolveLowerMatrix {
    signature: Signature,
}

impl MatrixSolveLowerMatrix {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixSolveLowerMatrix {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_solve_lower_matrix" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        validate_square_matrix_rhs_contract(self.name(), &matrix, &rhs)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.cols, rhs.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        validate_square_matrix_rhs_contract(self.name(), &matrix, &rhs)?;
        match matrix.value_type {
            DataType::Float32 => invoke_triangular_matrix_solver::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::triangular::solve_lower_matrix_view,
            ),
            DataType::Float64 => invoke_triangular_matrix_solver::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::triangular::solve_lower_matrix_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSolveUpperMatrix {
    signature: Signature,
}

impl MatrixSolveUpperMatrix {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixSolveUpperMatrix {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_solve_upper_matrix" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        validate_square_matrix_rhs_contract(self.name(), &matrix, &rhs)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.cols, rhs.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        validate_square_matrix_rhs_contract(self.name(), &matrix, &rhs)?;
        match matrix.value_type {
            DataType::Float32 => invoke_triangular_matrix_solver::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::triangular::solve_upper_matrix_view,
            ),
            DataType::Float64 => invoke_triangular_matrix_solver::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::triangular::solve_upper_matrix_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[must_use]
pub fn matrix_solve_lower_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSolveLower::new()))
}

#[must_use]
pub fn matrix_solve_upper_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSolveUpper::new()))
}

#[must_use]
pub fn matrix_solve_lower_matrix_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSolveLowerMatrix::new()))
}

#[must_use]
pub fn matrix_solve_upper_matrix_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSolveUpperMatrix::new()))
}
