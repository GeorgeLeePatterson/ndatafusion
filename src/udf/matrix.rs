use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use ndarray::{Array2, Axis};
use ndarrow::IntoArrow;

use super::common::{
    expect_fixed_size_list_arg, fixed_shape_tensor_view3_f64, fixed_size_list_view2_f64,
    map_arrow_error, nullable_or,
};
use crate::error::exec_error;
use crate::metadata::{
    fixed_shape_tensor_field, parse_float64_matrix_batch_field, parse_float64_vector_field,
    vector_field,
};
use crate::signatures::any_signature;

fn return_square_matrix_vector(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(usize, usize)> {
    let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    let vector_len = parse_float64_vector_field(&args.arg_fields[1], function_name, 2)?;
    if rows != cols {
        return Err(exec_error(
            function_name,
            format!("{function_name} requires square matrices, found ({rows}, {cols})"),
        ));
    }
    if vector_len != cols {
        return Err(exec_error(
            function_name,
            format!("rhs vector length mismatch: expected {cols}, found {vector_len}"),
        ));
    }
    Ok((rows, cols))
}

fn invoke_matrix_solver<E: std::fmt::Display>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    solve: impl Fn(
        &ndarray::ArrayView2<'_, f64>,
        &ndarray::ArrayView1<'_, f64>,
    ) -> std::result::Result<ndarray::Array1<f64>, E>,
) -> Result<ColumnarValue> {
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let rhs = expect_fixed_size_list_arg(args, 2, function_name)?;
    let matrix_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], matrices, function_name)?;
    let rhs_view = fixed_size_list_view2_f64(rhs, function_name)?;
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
    let output = Array2::from_shape_vec((rhs_view.nrows(), rhs_view.ncols()), output)
        .map_err(|error| exec_error(function_name, error))?;
    let output = output.into_arrow().map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixMatmul {
    signature: Signature,
}

impl MatrixMatmul {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixMatmul {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_matmul" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let [left_rows, left_cols] =
            parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let [right_rows, right_cols] =
            parse_float64_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left_cols != right_rows {
            return Err(exec_error(
                self.name(),
                format!(
                    "incompatible matrix shapes for batched matmul: ({left_rows}, {left_cols}) x \
                     ({right_rows}, {right_cols})"
                ),
            ));
        }
        fixed_shape_tensor_field(
            self.name(),
            &[left_rows, right_cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let right = expect_fixed_size_list_arg(&args, 2, self.name())?;
        let (_field, output) = nabled::arrow::matrix::batched_matmat::<Float64Type>(
            args.arg_fields[0].as_ref(),
            left,
            args.arg_fields[1].as_ref(),
            right,
        )
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixLuSolve {
    signature: Signature,
}

impl MatrixLuSolve {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixLuSolve {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_lu_solve" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (_rows, cols) = return_square_matrix_vector(&args, self.name())?;
        vector_field(self.name(), cols, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_matrix_solver(&args, self.name(), nabled::linalg::lu::solve_view)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixCholeskySolve {
    signature: Signature,
}

impl MatrixCholeskySolve {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixCholeskySolve {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_cholesky_solve" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (_rows, cols) = return_square_matrix_vector(&args, self.name())?;
        vector_field(self.name(), cols, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_matrix_solver(&args, self.name(), nabled::linalg::cholesky::solve_view)
    }
}

#[must_use]
pub fn matrix_matmul_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixMatmul::new()))
}

#[must_use]
pub fn matrix_lu_solve_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixLuSolve::new()))
}

#[must_use]
pub fn matrix_cholesky_solve_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixCholeskySolve::new()))
}
