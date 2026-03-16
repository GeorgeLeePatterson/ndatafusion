use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::array::{Int64Array, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Array3, Axis};
use ndarrow::NdarrowElement;
use num_complex::Complex64;

use super::common::{
    complex_fixed_shape_tensor_view3, expect_fixed_size_list_arg, fixed_shape_tensor_view3,
    nullable_or,
};
use super::docs::matrix_doc;
use crate::error::exec_error;
use crate::metadata::{
    complex_fixed_shape_tensor_field, fixed_shape_tensor_field, parse_complex_matrix_batch_field,
    parse_matrix_batch_field, scalar_field, struct_field,
};
use crate::signatures::any_signature;

fn real_matrix_equation_shape(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(DataType, usize, usize, bool)> {
    let left = parse_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    let right = parse_matrix_batch_field(&args.arg_fields[1], function_name, 2)?;
    let constant = parse_matrix_batch_field(&args.arg_fields[2], function_name, 3)?;
    if left.value_type != right.value_type || left.value_type != constant.value_type {
        return Err(exec_error(
            function_name,
            format!(
                "matrix value type mismatch: left {}, right {}, constant {}",
                left.value_type, right.value_type, constant.value_type
            ),
        ));
    }
    if left.rows != left.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square left matrices, found ({}, {})",
                left.rows, left.cols
            ),
        ));
    }
    if right.rows != right.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square right matrices, found ({}, {})",
                right.rows, right.cols
            ),
        ));
    }
    if constant.rows != left.rows || constant.cols != right.rows {
        return Err(exec_error(
            function_name,
            format!(
                "constant matrix shape mismatch: expected ({}, {}), found ({}, {})",
                left.rows, right.rows, constant.rows, constant.cols
            ),
        ));
    }
    Ok((left.value_type, constant.rows, constant.cols, nullable_or(args.arg_fields)))
}

fn complex_matrix_equation_shape(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(usize, usize, bool)> {
    let left = parse_complex_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    let right = parse_complex_matrix_batch_field(&args.arg_fields[1], function_name, 2)?;
    let constant = parse_complex_matrix_batch_field(&args.arg_fields[2], function_name, 3)?;
    if left.rows != left.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square left matrices, found ({}, {})",
                left.rows, left.cols
            ),
        ));
    }
    if right.rows != right.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square right matrices, found ({}, {})",
                right.rows, right.cols
            ),
        ));
    }
    if constant.rows != left.rows || constant.cols != right.rows {
        return Err(exec_error(
            function_name,
            format!(
                "constant matrix shape mismatch: expected ({}, {}), found ({}, {})",
                left.rows, right.rows, constant.rows, constant.cols
            ),
        ));
    }
    Ok((constant.rows, constant.cols, nullable_or(args.arg_fields)))
}

fn fixed_shape_tensor_batch_from_values<T>(
    function_name: &str,
    batch: usize,
    rows: usize,
    cols: usize,
    values: Vec<T>,
) -> Result<datafusion::arrow::array::FixedSizeListArray>
where
    T: NdarrowElement,
{
    let array = Array3::from_shape_vec((batch, rows, cols), values)
        .map_err(|error| exec_error(function_name, error))?;
    let (_field, array) = ndarrow::arrayd_to_fixed_shape_tensor(function_name, array.into_dyn())
        .map_err(|error| exec_error(function_name, error))?;
    Ok(array)
}

fn complex_fixed_shape_tensor_batch_from_values(
    function_name: &str,
    batch: usize,
    rows: usize,
    cols: usize,
    values: Vec<Complex64>,
) -> Result<datafusion::arrow::array::FixedSizeListArray> {
    let array = Array3::from_shape_vec((batch, rows, cols), values)
        .map_err(|error| exec_error(function_name, error))?;
    let (_field, array) =
        ndarrow::arrayd_complex64_to_fixed_shape_tensor(function_name, array.into_dyn())
            .map_err(|error| exec_error(function_name, error))?;
    Ok(array)
}

fn mixed_result_field(
    name: &str,
    value_type: &DataType,
    rows: usize,
    cols: usize,
    nullable: bool,
) -> Result<FieldRef> {
    let solution = fixed_shape_tensor_field("solution", value_type, &[rows, cols], false)?;
    let refinement_iterations = scalar_field("refinement_iterations", &DataType::Int64, false);
    Ok(struct_field(
        name,
        vec![solution.as_ref().clone(), refinement_iterations.as_ref().clone()],
        nullable,
    ))
}

fn complex_mixed_result_field(
    name: &str,
    rows: usize,
    cols: usize,
    nullable: bool,
) -> Result<FieldRef> {
    let solution = complex_fixed_shape_tensor_field("solution", &[rows, cols], false)?;
    let refinement_iterations = scalar_field("refinement_iterations", &DataType::Int64, false);
    Ok(struct_field(
        name,
        vec![solution.as_ref().clone(), refinement_iterations.as_ref().clone()],
        nullable,
    ))
}

fn invoke_real_matrix_equation_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    rows: usize,
    cols: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
        &ndarray::ArrayView2<'_, T::Native>,
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<ndarray::Array2<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let constant = expect_fixed_size_list_arg(args, 3, function_name)?;
    let left_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], left, function_name)?;
    let right_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[1], right, function_name)?;
    let constant_view =
        fixed_shape_tensor_view3::<T>(&args.arg_fields[2], constant, function_name)?;
    let batch = left_view.len_of(Axis(0));
    if right_view.len_of(Axis(0)) != batch || constant_view.len_of(Axis(0)) != batch {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: left {}, right {}, constant {}",
                batch,
                right_view.len_of(Axis(0)),
                constant_view.len_of(Axis(0))
            ),
        ));
    }
    let mut output = Vec::with_capacity(batch * rows * cols);
    for row in 0..batch {
        let solution = op(
            &left_view.index_axis(Axis(0), row),
            &right_view.index_axis(Axis(0), row),
            &constant_view.index_axis(Axis(0), row),
        )
        .map_err(|error| exec_error(function_name, error))?;
        output.extend(solution.iter().copied());
    }
    let output = fixed_shape_tensor_batch_from_values(function_name, batch, rows, cols, output)?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_real_matrix_equation_mixed_output<E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    rows: usize,
    cols: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, f64>,
        &ndarray::ArrayView2<'_, f64>,
        &ndarray::ArrayView2<'_, f64>,
    ) -> std::result::Result<nabled::linalg::sylvester::MixedSylvesterResult<f64>, E>,
) -> Result<ColumnarValue>
where
    E: std::fmt::Display,
{
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let constant = expect_fixed_size_list_arg(args, 3, function_name)?;
    let left_view =
        fixed_shape_tensor_view3::<Float64Type>(&args.arg_fields[0], left, function_name)?;
    let right_view =
        fixed_shape_tensor_view3::<Float64Type>(&args.arg_fields[1], right, function_name)?;
    let constant_view =
        fixed_shape_tensor_view3::<Float64Type>(&args.arg_fields[2], constant, function_name)?;
    let batch = left_view.len_of(Axis(0));
    if right_view.len_of(Axis(0)) != batch || constant_view.len_of(Axis(0)) != batch {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: left {}, right {}, constant {}",
                batch,
                right_view.len_of(Axis(0)),
                constant_view.len_of(Axis(0))
            ),
        ));
    }

    let mut solutions = Vec::with_capacity(batch * rows * cols);
    let mut refinement_iterations = Vec::with_capacity(batch);
    for row in 0..batch {
        let result = op(
            &left_view.index_axis(Axis(0), row),
            &right_view.index_axis(Axis(0), row),
            &constant_view.index_axis(Axis(0), row),
        )
        .map_err(|error| exec_error(function_name, error))?;
        solutions.extend(result.solution.iter().copied());
        refinement_iterations.push(
            i64::try_from(result.refinement_iterations).map_err(|_| {
                exec_error(function_name, "refinement_iterations exceeds i64 limits")
            })?,
        );
    }
    let solution =
        fixed_shape_tensor_batch_from_values(function_name, batch, rows, cols, solutions)?;
    let struct_array = StructArray::new(
        vec![
            fixed_shape_tensor_field("solution", &DataType::Float64, &[rows, cols], false)?,
            Arc::new(Field::new("refinement_iterations", DataType::Int64, false)),
        ]
        .into(),
        vec![Arc::new(solution), Arc::new(Int64Array::from(refinement_iterations))],
        None,
    );
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn invoke_complex_matrix_equation_output<E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    rows: usize,
    cols: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, Complex64>,
        &ndarray::ArrayView2<'_, Complex64>,
        &ndarray::ArrayView2<'_, Complex64>,
    ) -> std::result::Result<ndarray::Array2<Complex64>, E>,
) -> Result<ColumnarValue>
where
    E: std::fmt::Display,
{
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let constant = expect_fixed_size_list_arg(args, 3, function_name)?;
    let left_view = complex_fixed_shape_tensor_view3(&args.arg_fields[0], left, function_name)?;
    let right_view = complex_fixed_shape_tensor_view3(&args.arg_fields[1], right, function_name)?;
    let constant_view =
        complex_fixed_shape_tensor_view3(&args.arg_fields[2], constant, function_name)?;
    let batch = left_view.len_of(Axis(0));
    if right_view.len_of(Axis(0)) != batch || constant_view.len_of(Axis(0)) != batch {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: left {}, right {}, constant {}",
                batch,
                right_view.len_of(Axis(0)),
                constant_view.len_of(Axis(0))
            ),
        ));
    }
    let mut output = Vec::with_capacity(batch * rows * cols);
    for row in 0..batch {
        let solution = op(
            &left_view.index_axis(Axis(0), row),
            &right_view.index_axis(Axis(0), row),
            &constant_view.index_axis(Axis(0), row),
        )
        .map_err(|error| exec_error(function_name, error))?;
        output.extend(solution.iter().copied());
    }
    let output =
        complex_fixed_shape_tensor_batch_from_values(function_name, batch, rows, cols, output)?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_complex_matrix_equation_mixed_output<E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    rows: usize,
    cols: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, Complex64>,
        &ndarray::ArrayView2<'_, Complex64>,
        &ndarray::ArrayView2<'_, Complex64>,
    )
        -> std::result::Result<nabled::linalg::sylvester::MixedSylvesterResult<Complex64>, E>,
) -> Result<ColumnarValue>
where
    E: std::fmt::Display,
{
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let constant = expect_fixed_size_list_arg(args, 3, function_name)?;
    let left_view = complex_fixed_shape_tensor_view3(&args.arg_fields[0], left, function_name)?;
    let right_view = complex_fixed_shape_tensor_view3(&args.arg_fields[1], right, function_name)?;
    let constant_view =
        complex_fixed_shape_tensor_view3(&args.arg_fields[2], constant, function_name)?;
    let batch = left_view.len_of(Axis(0));
    if right_view.len_of(Axis(0)) != batch || constant_view.len_of(Axis(0)) != batch {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: left {}, right {}, constant {}",
                batch,
                right_view.len_of(Axis(0)),
                constant_view.len_of(Axis(0))
            ),
        ));
    }

    let mut solutions = Vec::with_capacity(batch * rows * cols);
    let mut refinement_iterations = Vec::with_capacity(batch);
    for row in 0..batch {
        let result = op(
            &left_view.index_axis(Axis(0), row),
            &right_view.index_axis(Axis(0), row),
            &constant_view.index_axis(Axis(0), row),
        )
        .map_err(|error| exec_error(function_name, error))?;
        solutions.extend(result.solution.iter().copied());
        refinement_iterations.push(
            i64::try_from(result.refinement_iterations).map_err(|_| {
                exec_error(function_name, "refinement_iterations exceeds i64 limits")
            })?,
        );
    }
    let solution =
        complex_fixed_shape_tensor_batch_from_values(function_name, batch, rows, cols, solutions)?;
    let struct_array = StructArray::new(
        vec![
            complex_fixed_shape_tensor_field("solution", &[rows, cols], false)?,
            Arc::new(Field::new("refinement_iterations", DataType::Int64, false)),
        ]
        .into(),
        vec![Arc::new(solution), Arc::new(Int64Array::from(refinement_iterations))],
        None,
    );
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSolveSylvester {
    signature: Signature,
}

impl MatrixSolveSylvester {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for MatrixSolveSylvester {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_solve_sylvester" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, rows, cols, nullable) = real_matrix_equation_shape(&args, self.name())?;
        fixed_shape_tensor_field(self.name(), &value_type, &[rows, cols], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (value_type, rows, cols, _nullable) = real_matrix_equation_shape(
            &ReturnFieldArgs { arg_fields: &args.arg_fields, scalar_arguments: &[] },
            self.name(),
        )?;
        match value_type {
            DataType::Float32 => invoke_real_matrix_equation_output::<Float32Type, _>(
                &args,
                self.name(),
                rows,
                cols,
                nabled::linalg::sylvester::solve_sylvester_view,
            ),
            DataType::Float64 => invoke_real_matrix_equation_output::<Float64Type, _>(
                &args,
                self.name(),
                rows,
                cols,
                nabled::linalg::sylvester::solve_sylvester_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Solve the Sylvester equation A X + X B = C for each batch row of real matrices.",
                "matrix_solve_sylvester(matrix_a, matrix_b, matrix_c)",
            )
            .with_argument(
                "matrix_a",
                "Square dense left matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument(
                "matrix_b",
                "Square dense right matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument("matrix_c", "Dense constant matrix batch with shape (rows(A), rows(B)).")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSolveSylvesterMixedF64 {
    signature: Signature,
}

impl MatrixSolveSylvesterMixedF64 {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for MatrixSolveSylvesterMixedF64 {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_solve_sylvester_mixed_f64" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, rows, cols, nullable) = real_matrix_equation_shape(&args, self.name())?;
        if value_type != DataType::Float64 {
            return Err(exec_error(
                self.name(),
                format!("{} requires Float64 matrix inputs, found {value_type}", self.name()),
            ));
        }
        mixed_result_field(self.name(), &DataType::Float64, rows, cols, nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (value_type, rows, cols, _nullable) = real_matrix_equation_shape(
            &ReturnFieldArgs { arg_fields: &args.arg_fields, scalar_arguments: &[] },
            self.name(),
        )?;
        if value_type != DataType::Float64 {
            return Err(exec_error(
                self.name(),
                format!("{} requires Float64 matrix inputs, found {value_type}", self.name()),
            ));
        }
        invoke_real_matrix_equation_mixed_output(
            &args,
            self.name(),
            rows,
            cols,
            nabled::linalg::sylvester::solve_sylvester_mixed_f64_view,
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Solve the Sylvester equation A X + X B = C with mixed-precision refinement for \
                 each batch row of Float64 matrices.",
                "matrix_solve_sylvester_mixed_f64(matrix_a, matrix_b, matrix_c)",
            )
            .with_argument(
                "matrix_a",
                "Square Float64 left matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument(
                "matrix_b",
                "Square Float64 right matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument(
                "matrix_c",
                "Float64 constant matrix batch with shape (rows(A), rows(B)).",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSolveSylvesterComplex {
    signature: Signature,
}

impl MatrixSolveSylvesterComplex {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for MatrixSolveSylvesterComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_solve_sylvester_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (rows, cols, nullable) = complex_matrix_equation_shape(&args, self.name())?;
        complex_fixed_shape_tensor_field(self.name(), &[rows, cols], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (rows, cols, _nullable) = complex_matrix_equation_shape(
            &ReturnFieldArgs { arg_fields: &args.arg_fields, scalar_arguments: &[] },
            self.name(),
        )?;
        invoke_complex_matrix_equation_output(
            &args,
            self.name(),
            rows,
            cols,
            nabled::linalg::sylvester::solve_sylvester_complex_view,
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Solve the Sylvester equation A X + X B = C for each batch row of complex \
                 matrices.",
                "matrix_solve_sylvester_complex(matrix_a, matrix_b, matrix_c)",
            )
            .with_argument(
                "matrix_a",
                "Square complex left matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument(
                "matrix_b",
                "Square complex right matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument(
                "matrix_c",
                "Complex constant matrix batch with shape (rows(A), rows(B)).",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSolveSylvesterMixedComplex {
    signature: Signature,
}

impl MatrixSolveSylvesterMixedComplex {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for MatrixSolveSylvesterMixedComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_solve_sylvester_mixed_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (rows, cols, nullable) = complex_matrix_equation_shape(&args, self.name())?;
        complex_mixed_result_field(self.name(), rows, cols, nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let (rows, cols, _nullable) = complex_matrix_equation_shape(
            &ReturnFieldArgs { arg_fields: &args.arg_fields, scalar_arguments: &[] },
            self.name(),
        )?;
        invoke_complex_matrix_equation_mixed_output(
            &args,
            self.name(),
            rows,
            cols,
            nabled::linalg::sylvester::solve_sylvester_mixed_complex_view,
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Solve the Sylvester equation A X + X B = C with mixed-precision refinement for \
                 each batch row of complex matrices.",
                "matrix_solve_sylvester_mixed_complex(matrix_a, matrix_b, matrix_c)",
            )
            .with_argument(
                "matrix_a",
                "Square complex left matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument(
                "matrix_b",
                "Square complex right matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument(
                "matrix_c",
                "Complex constant matrix batch with shape (rows(A), rows(B)).",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[must_use]
pub fn matrix_solve_sylvester_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSolveSylvester::new()))
}

#[must_use]
pub fn matrix_solve_sylvester_mixed_f64_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSolveSylvesterMixedF64::new()))
}

#[must_use]
pub fn matrix_solve_sylvester_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSolveSylvesterComplex::new()))
}

#[must_use]
pub fn matrix_solve_sylvester_mixed_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSolveSylvesterMixedComplex::new()))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::Array;
    use datafusion::common::config::ConfigOptions;
    use datafusion::logical_expr::ReturnFieldArgs;

    use super::*;

    fn real_matrix_equation_args() -> ScalarFunctionArgs {
        let (left_field, left) = ndarrow::arrayd_to_fixed_shape_tensor(
            "left",
            Array3::from_shape_vec((1, 2, 2), vec![1.0_f64, 0.0, 0.0, 2.0])
                .expect("left")
                .into_dyn(),
        )
        .expect("left");
        let (right_field, right) = ndarrow::arrayd_to_fixed_shape_tensor(
            "right",
            Array3::from_shape_vec((1, 2, 2), vec![3.0_f64, 0.0, 0.0, 4.0])
                .expect("right")
                .into_dyn(),
        )
        .expect("right");
        let (constant_field, constant) = ndarrow::arrayd_to_fixed_shape_tensor(
            "constant",
            Array3::from_shape_vec((1, 2, 2), vec![5.0_f64, 6.0, 7.0, 8.0])
                .expect("constant")
                .into_dyn(),
        )
        .expect("constant");
        ScalarFunctionArgs {
            args:           vec![
                ColumnarValue::Array(Arc::new(left)),
                ColumnarValue::Array(Arc::new(right)),
                ColumnarValue::Array(Arc::new(constant)),
            ],
            arg_fields:     vec![
                Arc::new(left_field),
                Arc::new(right_field),
                Arc::new(constant_field),
            ],
            number_rows:    1,
            return_field:   scalar_field("out", &DataType::Float64, false),
            config_options: Arc::new(ConfigOptions::new()),
        }
    }

    fn complex_matrix_equation_args() -> ScalarFunctionArgs {
        let (left_field, left) = ndarrow::arrayd_complex64_to_fixed_shape_tensor(
            "left_complex",
            Array3::from_shape_vec((1, 2, 2), vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(2.0, 0.0),
            ])
            .expect("left complex")
            .into_dyn(),
        )
        .expect("left complex");
        let (right_field, right) = ndarrow::arrayd_complex64_to_fixed_shape_tensor(
            "right_complex",
            Array3::from_shape_vec((1, 2, 2), vec![
                Complex64::new(3.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(0.0, 0.0),
                Complex64::new(4.0, 0.0),
            ])
            .expect("right complex")
            .into_dyn(),
        )
        .expect("right complex");
        let (constant_field, constant) = ndarrow::arrayd_complex64_to_fixed_shape_tensor(
            "constant_complex",
            Array3::from_shape_vec((1, 2, 2), vec![
                Complex64::new(5.0, 0.0),
                Complex64::new(6.0, 0.0),
                Complex64::new(7.0, 0.0),
                Complex64::new(8.0, 0.0),
            ])
            .expect("constant complex")
            .into_dyn(),
        )
        .expect("constant complex");
        ScalarFunctionArgs {
            args:           vec![
                ColumnarValue::Array(Arc::new(left)),
                ColumnarValue::Array(Arc::new(right)),
                ColumnarValue::Array(Arc::new(constant)),
            ],
            arg_fields:     vec![
                Arc::new(left_field),
                Arc::new(right_field),
                Arc::new(constant_field),
            ],
            number_rows:    1,
            return_field:   scalar_field("out", &DataType::Float64, false),
            config_options: Arc::new(ConfigOptions::new()),
        }
    }

    #[test]
    fn matrix_equation_shape_helpers_validate_real_and_complex_contracts() {
        let left =
            fixed_shape_tensor_field("left", &DataType::Float64, &[2, 2], false).expect("left");
        let right =
            fixed_shape_tensor_field("right", &DataType::Float64, &[2, 2], false).expect("right");
        let constant = fixed_shape_tensor_field("constant", &DataType::Float64, &[2, 2], false)
            .expect("constant");
        let scalar_arguments: [Option<&datafusion::common::ScalarValue>; 0] = [];
        let fields = vec![left, right, constant];
        let args =
            ReturnFieldArgs { arg_fields: &fields, scalar_arguments: &scalar_arguments };
        let (value_type, rows, cols, nullable) =
            real_matrix_equation_shape(&args, "matrix_solve_sylvester").expect("real shape");
        assert_eq!(value_type, DataType::Float64);
        assert_eq!((rows, cols, nullable), (2, 2, false));

        let complex_fields = vec![
            complex_fixed_shape_tensor_field("left", &[2, 2], false).expect("left"),
            complex_fixed_shape_tensor_field("right", &[2, 2], false).expect("right"),
            complex_fixed_shape_tensor_field("constant", &[2, 2], true).expect("constant"),
        ];
        let args = ReturnFieldArgs {
            arg_fields:       &complex_fields,
            scalar_arguments: &scalar_arguments,
        };
        let (rows, cols, nullable) =
            complex_matrix_equation_shape(&args, "matrix_solve_sylvester_complex")
                .expect("complex shape");
        assert_eq!((rows, cols, nullable), (2, 2, true));
    }

    #[test]
    fn matrix_equation_shape_helpers_reject_mismatches() {
        let scalar_arguments: [Option<&datafusion::common::ScalarValue>; 0] = [];
        let mismatched = vec![
            fixed_shape_tensor_field("left", &DataType::Float64, &[2, 2], false).expect("left"),
            fixed_shape_tensor_field("right", &DataType::Float32, &[2, 2], false).expect("right"),
            fixed_shape_tensor_field("constant", &DataType::Float64, &[2, 2], false)
                .expect("constant"),
        ];
        let args =
            ReturnFieldArgs { arg_fields: &mismatched, scalar_arguments: &scalar_arguments };
        assert!(real_matrix_equation_shape(&args, "matrix_solve_sylvester").is_err());

        let nonsquare = vec![
            complex_fixed_shape_tensor_field("left", &[2, 3], false).expect("left"),
            complex_fixed_shape_tensor_field("right", &[2, 2], false).expect("right"),
            complex_fixed_shape_tensor_field("constant", &[2, 2], false).expect("constant"),
        ];
        let args =
            ReturnFieldArgs { arg_fields: &nonsquare, scalar_arguments: &scalar_arguments };
        assert!(complex_matrix_equation_shape(&args, "matrix_solve_sylvester_complex").is_err());

        let right_nonsquare = vec![
            fixed_shape_tensor_field("left", &DataType::Float64, &[2, 2], false).expect("left"),
            fixed_shape_tensor_field("right", &DataType::Float64, &[2, 3], false).expect("right"),
            fixed_shape_tensor_field("constant", &DataType::Float64, &[2, 2], false)
                .expect("constant"),
        ];
        let args = ReturnFieldArgs {
            arg_fields:       &right_nonsquare,
            scalar_arguments: &scalar_arguments,
        };
        assert!(real_matrix_equation_shape(&args, "matrix_solve_sylvester").is_err());

        let complex_right_nonsquare = vec![
            complex_fixed_shape_tensor_field("left", &[2, 2], false).expect("left"),
            complex_fixed_shape_tensor_field("right", &[2, 3], false).expect("right"),
            complex_fixed_shape_tensor_field("constant", &[2, 2], false).expect("constant"),
        ];
        let args = ReturnFieldArgs {
            arg_fields:       &complex_right_nonsquare,
            scalar_arguments: &scalar_arguments,
        };
        assert!(complex_matrix_equation_shape(&args, "matrix_solve_sylvester_complex").is_err());
    }

    #[test]
    fn matrix_equation_helpers_build_expected_output_contracts() {
        let output = fixed_shape_tensor_batch_from_values("matrix_equation", 1, 2, 2, vec![
            1.0_f64, 2.0, 3.0, 4.0,
        ])
        .expect("real tensor batch");
        assert_eq!(output.len(), 1);

        let output =
            complex_fixed_shape_tensor_batch_from_values("matrix_equation_complex", 1, 2, 2, vec![
                Complex64::new(1.0, 0.0),
                Complex64::new(2.0, 0.0),
                Complex64::new(3.0, 0.0),
                Complex64::new(4.0, 0.0),
            ])
            .expect("complex tensor batch");
        assert_eq!(output.len(), 1);

        let mixed = mixed_result_field("mixed", &DataType::Float64, 2, 3, true).expect("mixed");
        let DataType::Struct(fields) = mixed.data_type() else {
            panic!("expected struct field");
        };
        assert_eq!(fields[0].name(), "solution");
        assert_eq!(fields[1].name(), "refinement_iterations");

        let complex =
            complex_mixed_result_field("mixed_complex", 2, 3, false).expect("complex mixed");
        let DataType::Struct(fields) = complex.data_type() else {
            panic!("expected complex struct field");
        };
        assert_eq!(fields[0].name(), "solution");
        assert_eq!(fields[1].name(), "refinement_iterations");
    }

    #[test]
    fn matrix_equation_helpers_reject_batch_mismatches() {
        let (left_field, left) = ndarrow::arrayd_to_fixed_shape_tensor(
            "left",
            Array3::from_shape_vec((1, 2, 2), vec![1.0_f64, 0.0, 0.0, 2.0])
                .expect("left")
                .into_dyn(),
        )
        .expect("left");
        let (right_field, right) = ndarrow::arrayd_to_fixed_shape_tensor(
            "right",
            Array3::from_shape_vec((2, 2, 2), vec![3.0_f64, 0.0, 0.0, 4.0, 3.0, 0.0, 0.0, 4.0])
                .expect("right")
                .into_dyn(),
        )
        .expect("right");
        let (constant_field, constant) = ndarrow::arrayd_to_fixed_shape_tensor(
            "constant",
            Array3::from_shape_vec((1, 2, 2), vec![5.0_f64, 6.0, 7.0, 8.0])
                .expect("constant")
                .into_dyn(),
        )
        .expect("constant");
        let args = ScalarFunctionArgs {
            args:           vec![
                ColumnarValue::Array(Arc::new(left)),
                ColumnarValue::Array(Arc::new(right)),
                ColumnarValue::Array(Arc::new(constant)),
            ],
            arg_fields:     vec![
                Arc::new(left_field),
                Arc::new(right_field),
                Arc::new(constant_field),
            ],
            number_rows:    1,
            return_field:   scalar_field("out", &DataType::Float64, false),
            config_options: Arc::new(ConfigOptions::new()),
        };
        assert!(
            invoke_real_matrix_equation_output::<Float64Type, _>(
                &args,
                "matrix_solve_sylvester",
                2,
                2,
                |_left, _right, _constant| {
                    Ok::<ndarray::Array2<f64>, &'static str>(ndarray::Array2::<f64>::zeros((2, 2)))
                },
            )
            .is_err()
        );
    }

    #[test]
    fn matrix_equation_output_helpers_cover_real_execution_paths() {
        let args = real_matrix_equation_args();
        let output = invoke_real_matrix_equation_output::<Float64Type, _>(
            &args,
            "matrix_solve_sylvester",
            2,
            2,
            |_left, _right, constant| Ok::<ndarray::Array2<f64>, &'static str>(constant.to_owned()),
        )
        .expect("real output");
        let ColumnarValue::Array(output) = output else {
            panic!("expected array output");
        };
        assert_eq!(output.len(), 1);

        let mixed = invoke_real_matrix_equation_mixed_output(
            &args,
            "matrix_solve_sylvester_mixed_f64",
            2,
            2,
            |_left, _right, constant| {
                Ok::<nabled::linalg::sylvester::MixedSylvesterResult<f64>, &'static str>(
                    nabled::linalg::sylvester::MixedSylvesterResult {
                        solution:              constant.to_owned(),
                        refinement_iterations: 2,
                    },
                )
            },
        )
        .expect("mixed output");
        let ColumnarValue::Array(mixed) = mixed else {
            panic!("expected mixed array output");
        };
        let mixed = mixed.as_any().downcast_ref::<StructArray>().expect("mixed struct array");
        let iterations =
            mixed.column(1).as_any().downcast_ref::<Int64Array>().expect("refinement iterations");
        assert_eq!(iterations.value(0), 2);
    }

    #[test]
    fn matrix_equation_output_helpers_cover_complex_execution_paths() {
        let complex_args = complex_matrix_equation_args();
        let output = invoke_complex_matrix_equation_output(
            &complex_args,
            "matrix_solve_sylvester_complex",
            2,
            2,
            |_left, _right, constant| {
                Ok::<ndarray::Array2<Complex64>, &'static str>(constant.to_owned())
            },
        )
        .expect("complex output");
        let ColumnarValue::Array(output) = output else {
            panic!("expected complex array output");
        };
        assert_eq!(output.len(), 1);

        let mixed = invoke_complex_matrix_equation_mixed_output(
            &complex_args,
            "matrix_solve_sylvester_mixed_complex",
            2,
            2,
            |_left, _right, constant| {
                Ok::<nabled::linalg::sylvester::MixedSylvesterResult<Complex64>, &'static str>(
                    nabled::linalg::sylvester::MixedSylvesterResult {
                        solution:              constant.to_owned(),
                        refinement_iterations: 3,
                    },
                )
            },
        )
        .expect("complex mixed output");
        let ColumnarValue::Array(mixed) = mixed else {
            panic!("expected complex mixed array output");
        };
        let mixed = mixed.as_any().downcast_ref::<StructArray>().expect("complex mixed struct");
        let iterations = mixed
            .column(1)
            .as_any()
            .downcast_ref::<Int64Array>()
            .expect("complex refinement iterations");
        assert_eq!(iterations.value(0), 3);
    }
}
