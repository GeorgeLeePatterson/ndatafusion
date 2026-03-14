use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Array1, ArrayView1, ArrayView2, Axis};
use ndarrow::NdarrowElement;

use super::common::{
    expect_fixed_size_list_arg, expect_real_scalar_arg, expect_real_scalar_argument,
    expect_usize_scalar_arg, expect_usize_scalar_argument, fixed_shape_tensor_view3,
    fixed_size_list_array_from_flat_rows, fixed_size_list_view2, nullable_or,
};
use crate::error::exec_error;
use crate::metadata::{parse_matrix_batch_field, parse_vector_field, vector_field};
use crate::signatures::any_signature;

fn return_square_system(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(DataType, usize)> {
    let matrix = parse_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    let vector = parse_vector_field(&args.arg_fields[1], function_name, 2)?;
    if matrix.value_type != vector.value_type {
        return Err(exec_error(
            function_name,
            format!("value type mismatch: matrix {}, rhs {}", matrix.value_type, vector.value_type),
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
    if vector.len != matrix.cols {
        return Err(exec_error(
            function_name,
            format!("rhs vector length mismatch: expected {}, found {}", matrix.cols, vector.len),
        ));
    }
    Ok((matrix.value_type, matrix.rows))
}

fn validate_tolerance(function_name: &str, tolerance: f64) -> Result<f64> {
    if !tolerance.is_finite() {
        return Err(exec_error(function_name, "tolerance must be finite"));
    }
    if tolerance <= 0.0 {
        return Err(exec_error(function_name, "tolerance must be positive"));
    }
    Ok(tolerance)
}

fn validate_max_iterations(function_name: &str, max_iterations: usize) -> Result<usize> {
    if max_iterations == 0 {
        return Err(exec_error(function_name, "max_iterations must be greater than 0"));
    }
    Ok(max_iterations)
}

fn iterative_config_f32(
    function_name: &str,
    tolerance: f64,
    max_iterations: usize,
) -> Result<nabled::ml::iterative::IterativeConfig<f32>> {
    let tolerance = validate_tolerance(function_name, tolerance)?;
    let max_iterations = validate_max_iterations(function_name, max_iterations)?;
    let tolerance = tolerance.to_string().parse::<f32>().map_err(|error| {
        exec_error(
            function_name,
            format!("tolerance could not be represented in matrix value type: {error}"),
        )
    })?;
    Ok(nabled::ml::iterative::IterativeConfig { tolerance, max_iterations })
}

fn iterative_config_f64(
    function_name: &str,
    tolerance: f64,
    max_iterations: usize,
) -> Result<nabled::ml::iterative::IterativeConfig<f64>> {
    let tolerance = validate_tolerance(function_name, tolerance)?;
    let max_iterations = validate_max_iterations(function_name, max_iterations)?;
    Ok(nabled::ml::iterative::IterativeConfig { tolerance, max_iterations })
}

fn invoke_iterative_solver<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    config: &nabled::ml::iterative::IterativeConfig<T::Native>,
    op: impl Fn(
        &ArrayView2<'_, T::Native>,
        &ArrayView1<'_, T::Native>,
        &nabled::ml::iterative::IterativeConfig<T::Native>,
    ) -> std::result::Result<Array1<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native:
        NdarrowElement + NabledReal + std::ops::SubAssign + nabled::linalg::lu::LuProviderScalar,
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
    if matrix_view.len_of(Axis(1)) != matrix_view.len_of(Axis(2)) {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square matrices, found ({}, {})",
                matrix_view.len_of(Axis(1)),
                matrix_view.len_of(Axis(2))
            ),
        ));
    }
    if rhs_view.ncols() != matrix_view.len_of(Axis(2)) {
        return Err(exec_error(
            function_name,
            format!(
                "rhs vector length mismatch: expected {}, found {}",
                matrix_view.len_of(Axis(2)),
                rhs_view.ncols()
            ),
        ));
    }

    let mut output = Vec::with_capacity(rhs_view.len());
    for row in 0..matrix_view.len_of(Axis(0)) {
        let solution =
            op(&matrix_view.index_axis(Axis(0), row), &rhs_view.index_axis(Axis(0), row), config)
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

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixConjugateGradient {
    signature: Signature,
}

impl MatrixConjugateGradient {
    fn new() -> Self { Self { signature: any_signature(4) } }
}

impl ScalarUDFImpl for MatrixConjugateGradient {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_conjugate_gradient" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, len) = return_square_system(&args, self.name())?;
        let tolerance = expect_real_scalar_argument(&args, 3, self.name())?;
        let max_iterations = expect_usize_scalar_argument(&args, 4, self.name())?;
        let _ = validate_tolerance(self.name(), tolerance)?;
        let _ = validate_max_iterations(self.name(), max_iterations)?;
        vector_field(self.name(), &value_type, len, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let vector = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix.value_type != vector.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {}, rhs {}",
                    matrix.value_type, vector.value_type
                ),
            ));
        }
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        if vector.len != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "rhs vector length mismatch: expected {}, found {}",
                    matrix.cols, vector.len
                ),
            ));
        }
        let tolerance = expect_real_scalar_arg(&args, 3, self.name())?;
        let max_iterations = expect_usize_scalar_arg(&args, 4, self.name())?;
        match matrix.value_type {
            DataType::Float32 => {
                let config = iterative_config_f32(self.name(), tolerance, max_iterations)?;
                invoke_iterative_solver::<Float32Type, _>(
                    &args,
                    self.name(),
                    &config,
                    nabled::ml::iterative::conjugate_gradient_view,
                )
            }
            DataType::Float64 => {
                let config = iterative_config_f64(self.name(), tolerance, max_iterations)?;
                invoke_iterative_solver::<Float64Type, _>(
                    &args,
                    self.name(),
                    &config,
                    nabled::ml::iterative::conjugate_gradient_view,
                )
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixGmres {
    signature: Signature,
}

impl MatrixGmres {
    fn new() -> Self { Self { signature: any_signature(4) } }
}

impl ScalarUDFImpl for MatrixGmres {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_gmres" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, len) = return_square_system(&args, self.name())?;
        let tolerance = expect_real_scalar_argument(&args, 3, self.name())?;
        let max_iterations = expect_usize_scalar_argument(&args, 4, self.name())?;
        let _ = validate_tolerance(self.name(), tolerance)?;
        let _ = validate_max_iterations(self.name(), max_iterations)?;
        vector_field(self.name(), &value_type, len, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let vector = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix.value_type != vector.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {}, rhs {}",
                    matrix.value_type, vector.value_type
                ),
            ));
        }
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        if vector.len != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "rhs vector length mismatch: expected {}, found {}",
                    matrix.cols, vector.len
                ),
            ));
        }
        let tolerance = expect_real_scalar_arg(&args, 3, self.name())?;
        let max_iterations = expect_usize_scalar_arg(&args, 4, self.name())?;
        match matrix.value_type {
            DataType::Float32 => {
                let config = iterative_config_f32(self.name(), tolerance, max_iterations)?;
                invoke_iterative_solver::<Float32Type, _>(
                    &args,
                    self.name(),
                    &config,
                    nabled::ml::iterative::gmres_view,
                )
            }
            DataType::Float64 => {
                let config = iterative_config_f64(self.name(), tolerance, max_iterations)?;
                invoke_iterative_solver::<Float64Type, _>(
                    &args,
                    self.name(),
                    &config,
                    nabled::ml::iterative::gmres_view,
                )
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[must_use]
pub fn matrix_conjugate_gradient_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixConjugateGradient::new()))
}

#[must_use]
pub fn matrix_gmres_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixGmres::new()))
}
