use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::Axis;
use ndarrow::NdarrowElement;

use super::common::{
    expect_fixed_size_list_arg, fixed_shape_tensor_view3, fixed_size_list_array_from_flat_rows,
    fixed_size_list_view2, map_arrow_error, nullable_or,
};
use super::docs::matrix_doc;
use crate::error::exec_error;
use crate::metadata::{
    complex_fixed_shape_tensor_field, complex_vector_field, fixed_shape_tensor_field,
    parse_complex_matrix_batch_field, parse_complex_vector_field, parse_matrix_batch_field,
    parse_vector_field, vector_field,
};
use crate::signatures::any_signature;

fn return_square_matrix_vector(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(DataType, usize, usize)> {
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
    Ok((matrix.value_type, matrix.rows, matrix.cols))
}

fn return_matrix_vector(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(DataType, usize, usize)> {
    let matrix = parse_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    let vector = parse_vector_field(&args.arg_fields[1], function_name, 2)?;
    if matrix.value_type != vector.value_type {
        return Err(exec_error(
            function_name,
            format!("value type mismatch: matrix {}, rhs {}", matrix.value_type, vector.value_type),
        ));
    }
    if vector.len != matrix.cols {
        return Err(exec_error(
            function_name,
            format!("rhs vector length mismatch: expected {}, found {}", matrix.cols, vector.len),
        ));
    }
    Ok((matrix.value_type, matrix.rows, matrix.cols))
}

fn return_complex_matrix_vector(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(usize, usize)> {
    let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    let (_vector_field, vector) =
        parse_complex_vector_field(&args.arg_fields[1], function_name, 2)?;
    if vector.len != matrix.cols {
        return Err(exec_error(
            function_name,
            format!("rhs vector length mismatch: expected {}, found {}", matrix.cols, vector.len),
        ));
    }
    Ok((matrix.rows, matrix.cols))
}

fn invoke_matrix_solver<T, E>(
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

fn invoke_matrix_matvec<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(
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
    if matrix_view.len_of(Axis(2)) != rhs_view.ncols() {
        return Err(exec_error(
            function_name,
            format!(
                "rhs vector length mismatch: expected {}, found {}",
                matrix_view.len_of(Axis(2)),
                rhs_view.ncols()
            ),
        ));
    }

    let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)) * matrix_view.len_of(Axis(1)));
    for row in 0..matrix_view.len_of(Axis(0)) {
        let result = op(&matrix_view.index_axis(Axis(0), row), &rhs_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        output.extend(result.iter().copied());
    }
    let output = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        rhs_view.nrows(),
        matrix_view.len_of(Axis(1)),
        &output,
    )?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_complex_matrix_matvec(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue> {
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let rhs = expect_fixed_size_list_arg(args, 2, function_name)?;
    let output =
        nabled::arrow::tensor::cube_matvec_complex(args.arg_fields[0].as_ref(), matrices, rhs)
            .map_err(|error| map_arrow_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_complex_matrix_matmul(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue> {
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let (_field, output) = nabled::arrow::tensor::cube_matmat_complex(
        args.arg_fields[0].as_ref(),
        left,
        args.arg_fields[1].as_ref(),
        right,
    )
    .map_err(|error| map_arrow_error(function_name, error))?;
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
        let left = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "matrix value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        if left.cols != right.rows {
            return Err(exec_error(
                self.name(),
                format!(
                    "incompatible matrix shapes for batched matmul: ({}, {}) x ({}, {})",
                    left.rows, left.cols, right.rows, right.cols
                ),
            ));
        }
        fixed_shape_tensor_field(
            self.name(),
            &left.value_type,
            &[left.rows, right.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left_contract = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right_contract = parse_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left_contract.value_type != right_contract.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "matrix value type mismatch: left {}, right {}",
                    left_contract.value_type, right_contract.value_type
                ),
            ));
        }
        let left = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let right = expect_fixed_size_list_arg(&args, 2, self.name())?;
        let output = match left_contract.value_type {
            DataType::Float32 => nabled::arrow::matrix::batched_matmat::<Float32Type>(
                args.arg_fields[0].as_ref(),
                left,
                args.arg_fields[1].as_ref(),
                right,
            ),
            DataType::Float64 => nabled::arrow::matrix::batched_matmat::<Float64Type>(
                args.arg_fields[0].as_ref(),
                left,
                args.arg_fields[1].as_ref(),
                right,
            ),
            actual => {
                return Err(exec_error(
                    self.name(),
                    format!("unsupported matrix value type {actual}"),
                ));
            }
        }
        .map_err(|error| map_arrow_error(self.name(), error))?;
        Ok(ColumnarValue::Array(Arc::new(output.1)))
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute the row-wise dense matrix-matrix product for paired matrix batches.",
                "matrix_matmul(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left dense matrix batch in canonical fixed-shape tensor rank-2 form.",
            )
            .with_argument(
                "right_batch",
                "Right dense matrix batch in canonical fixed-shape tensor rank-2 form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixMatvec {
    signature: Signature,
}

impl MatrixMatvec {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixMatvec {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_matvec" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, rows, _cols) = return_matrix_vector(&args, self.name())?;
        vector_field(self.name(), &value_type, rows, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => invoke_matrix_matvec::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix::matvec_view,
            ),
            DataType::Float64 => invoke_matrix_matvec::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix::matvec_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute the row-wise dense matrix-vector product for paired matrix and vector \
                 batches.",
                "matrix_matvec(matrix_batch, rhs_batch)",
            )
            .with_argument(
                "matrix_batch",
                "Dense matrix batch in canonical fixed-shape tensor rank-2 form.",
            )
            .with_argument(
                "rhs_batch",
                "Dense vector batch in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixMatvecComplex {
    signature: Signature,
}

impl MatrixMatvecComplex {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixMatvecComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_matvec_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (rows, _cols) = return_complex_matrix_vector(&args, self.name())?;
        complex_vector_field(self.name(), rows, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let (_rhs_field, vector) = parse_complex_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if vector.len != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "rhs vector length mismatch: expected {}, found {}",
                    matrix.cols, vector.len
                ),
            ));
        }
        invoke_complex_matrix_matvec(&args, self.name())
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute the row-wise dense complex matrix-vector product for paired matrix and \
                 vector batches.",
                "matrix_matvec_complex(matrix_batch, rhs_batch)",
            )
            .with_argument(
                "matrix_batch",
                "Dense complex matrix batch in canonical fixed-shape tensor rank-2 form.",
            )
            .with_argument(
                "rhs_batch",
                "Dense complex vector batch containing one right-hand side per matrix row.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixMatmulComplex {
    signature: Signature,
}

impl MatrixMatmulComplex {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixMatmulComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_matmat_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_complex_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left.cols != right.rows {
            return Err(exec_error(
                self.name(),
                format!(
                    "incompatible matrix shapes for batched matmul: ({}, {}) x ({}, {})",
                    left.rows, left.cols, right.rows, right.cols
                ),
            ));
        }
        complex_fixed_shape_tensor_field(
            self.name(),
            &[left.rows, right.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_complex_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left.cols != right.rows {
            return Err(exec_error(
                self.name(),
                format!(
                    "incompatible matrix shapes for batched matmul: ({}, {}) x ({}, {})",
                    left.rows, left.cols, right.rows, right.cols
                ),
            ));
        }
        invoke_complex_matrix_matmul(&args, self.name())
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute the row-wise dense complex matrix-matrix product for paired matrix \
                 batches.",
                "matrix_matmat_complex(left_batch, right_batch)",
            )
            .with_argument(
                "left_batch",
                "Left dense complex matrix batch in canonical fixed-shape tensor rank-2 form.",
            )
            .with_argument(
                "right_batch",
                "Right dense complex matrix batch in canonical fixed-shape tensor rank-2 form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
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
        let (value_type, _rows, cols) = return_square_matrix_vector(&args, self.name())?;
        vector_field(self.name(), &value_type, cols, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => invoke_matrix_solver::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::lu::solve_view,
            ),
            DataType::Float64 => invoke_matrix_solver::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::lu::solve_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Solve each square dense system in the batch with LU factorization and a vector \
                 right-hand side.",
                "matrix_lu_solve(matrix_batch, rhs_batch)",
            )
            .with_argument(
                "matrix_batch",
                "Square dense matrix batch in canonical fixed-shape tensor rank-2 form.",
            )
            .with_argument(
                "rhs_batch",
                "Dense vector batch containing one right-hand side per matrix row.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
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
        let (value_type, _rows, cols) = return_square_matrix_vector(&args, self.name())?;
        vector_field(self.name(), &value_type, cols, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => invoke_matrix_solver::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::cholesky::solve_view,
            ),
            DataType::Float64 => invoke_matrix_solver::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::cholesky::solve_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Solve each square dense system in the batch with Cholesky factorization and a \
                 vector right-hand side.",
                "matrix_cholesky_solve(matrix_batch, rhs_batch)",
            )
            .with_argument(
                "matrix_batch",
                "Square dense matrix batch in canonical fixed-shape tensor rank-2 form.",
            )
            .with_argument(
                "rhs_batch",
                "Dense vector batch containing one right-hand side per matrix row.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[must_use]
pub fn matrix_matmul_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixMatmul::new()))
}

#[must_use]
pub fn matrix_matmat_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixMatmulComplex::new()))
}

#[must_use]
pub fn matrix_matvec_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixMatvec::new()))
}

#[must_use]
pub fn matrix_matvec_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixMatvecComplex::new()))
}

#[must_use]
pub fn matrix_lu_solve_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixLuSolve::new()))
}

#[must_use]
pub fn matrix_cholesky_solve_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixCholeskySolve::new()))
}
