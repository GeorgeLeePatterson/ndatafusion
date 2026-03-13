use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::array::{Int8Array, Int64Array, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Array2, Array3, Axis};
use ndarrow::NdarrowElement;

use super::common::{
    expect_fixed_size_list_arg, fixed_shape_tensor_view3, fixed_size_list_array_from_flat_rows,
    fixed_size_list_view2, nullable_or, primitive_array_from_values,
};
use crate::error::exec_error;
use crate::metadata::{
    fixed_shape_tensor_field, parse_matrix_batch_field, parse_vector_field, scalar_field,
    struct_field, vector_field,
};
use crate::signatures::any_signature;

fn square_matrix_shape(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(DataType, usize, usize, bool)> {
    let matrix = parse_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    if matrix.rows != matrix.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square matrices, found ({}, {})",
                matrix.rows, matrix.cols
            ),
        ));
    }
    Ok((matrix.value_type, matrix.rows, matrix.cols, nullable_or(args.arg_fields)))
}

fn tensor_column<T>(
    name: &str,
    shape: [usize; 3],
    values: Vec<T>,
) -> Result<(FieldRef, Arc<datafusion::arrow::array::FixedSizeListArray>)>
where
    T: NdarrowElement,
{
    let tensor = Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
        .map_err(|error| exec_error(name, error))?;
    let (field, array) = ndarrow::arrayd_to_fixed_shape_tensor(name, tensor.into_dyn())
        .map_err(|error| exec_error(name, error))?;
    Ok((Arc::new(field), Arc::new(array)))
}

fn int64_scalar_field(name: &str, nullable: bool) -> FieldRef {
    Arc::new(Field::new(name, DataType::Int64, nullable))
}

fn invoke_matrix_scalar<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(&ndarray::ArrayView2<'_, T::Native>) -> std::result::Result<T::Native, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)));
    for row in 0..matrix_view.len_of(Axis(0)) {
        output.push(
            op(&matrix_view.index_axis(Axis(0), row))
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    let output = primitive_array_from_values::<T>(output);
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_matrix_scalar_i64<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(&ndarray::ArrayView2<'_, T::Native>) -> std::result::Result<i64, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)));
    for row in 0..matrix_view.len_of(Axis(0)) {
        output.push(
            op(&matrix_view.index_axis(Axis(0), row))
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    Ok(ColumnarValue::Array(Arc::new(Int64Array::from(output))))
}

fn invoke_matrix_tensor_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    output_rows: usize,
    output_cols: usize,
    op: impl Fn(&ndarray::ArrayView2<'_, T::Native>) -> std::result::Result<Array2<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut output = Vec::with_capacity(batch * output_rows * output_cols);
    for row in 0..batch {
        let values = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        output.extend(values.iter().copied());
    }
    let (_field, output) = tensor_column(function_name, [batch, output_rows, output_cols], output)?;
    Ok(ColumnarValue::Array(output))
}

fn invoke_matrix_qr_vector_output<T, E>(
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
    if matrix_view.len_of(Axis(1)) != rhs_view.ncols() {
        return Err(exec_error(
            function_name,
            format!(
                "rhs vector length mismatch: expected {}, found {}",
                matrix_view.len_of(Axis(1)),
                rhs_view.ncols()
            ),
        ));
    }

    let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)) * matrix_view.len_of(Axis(2)));
    for row in 0..matrix_view.len_of(Axis(0)) {
        let solution =
            op(&matrix_view.index_axis(Axis(0), row), &rhs_view.index_axis(Axis(0), row))
                .map_err(|error| exec_error(function_name, error))?;
        output.extend(solution.iter().copied());
    }
    let output = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        rhs_view.nrows(),
        matrix_view.len_of(Axis(2)),
        &output,
    )?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixLu {
    signature: Signature,
}

impl MatrixLu {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixLu {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_lu" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        let l_field = fixed_shape_tensor_field("l", &value_type, &[rows, cols], false)?;
        let u_field = fixed_shape_tensor_field("u", &value_type, &[rows, cols], false)?;
        Ok(struct_field(
            self.name(),
            vec![l_field.as_ref().clone(), u_field.as_ref().clone()],
            nullable,
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                let output = nabled::arrow::batched::lu_f32(args.arg_fields[0].as_ref(), matrices)
                    .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut lower_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut upper_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                for result in output {
                    lower_values.extend(result.l.iter().copied());
                    upper_values.extend(result.u.iter().copied());
                }
                let (lower_field, lower_array) =
                    tensor_column("l", [batch, matrix.rows, matrix.cols], lower_values)?;
                let (upper_field, upper_array) =
                    tensor_column("u", [batch, matrix.rows, matrix.cols], upper_values)?;
                let struct_array = StructArray::new(
                    vec![lower_field, upper_field].into(),
                    vec![lower_array, upper_array],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let output = nabled::arrow::batched::lu_f64(args.arg_fields[0].as_ref(), matrices)
                    .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut lower_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut upper_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                for result in output {
                    lower_values.extend(result.l.iter().copied());
                    upper_values.extend(result.u.iter().copied());
                }
                let (lower_field, lower_array) =
                    tensor_column("l", [batch, matrix.rows, matrix.cols], lower_values)?;
                let (upper_field, upper_array) =
                    tensor_column("u", [batch, matrix.rows, matrix.cols], upper_values)?;
                let struct_array = StructArray::new(
                    vec![lower_field, upper_field].into(),
                    vec![lower_array, upper_array],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixInverse {
    signature: Signature,
}

impl MatrixInverse {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixInverse {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_inverse" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        fixed_shape_tensor_field(self.name(), &value_type, &[rows, cols], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| nabled::linalg::lu::inverse_view(view),
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| nabled::linalg::lu::inverse_view(view),
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixDeterminant {
    signature: Signature,
}

impl MatrixDeterminant {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixDeterminant {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_determinant" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, _rows, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        Ok(scalar_field(self.name(), &value_type, nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => invoke_matrix_scalar::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::lu::determinant_view,
            ),
            DataType::Float64 => invoke_matrix_scalar::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::lu::determinant_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixLogDeterminant {
    signature: Signature,
}

impl MatrixLogDeterminant {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixLogDeterminant {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_log_determinant" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, _rows, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        let sign_field = Field::new("sign", DataType::Int8, false);
        let log_abs_field = scalar_field("ln_abs_det", &value_type, false);
        Ok(struct_field(self.name(), vec![sign_field, log_abs_field.as_ref().clone()], nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        match matrix.value_type {
            DataType::Float32 => {
                let matrix_view = fixed_shape_tensor_view3::<Float32Type>(
                    &args.arg_fields[0],
                    matrices,
                    self.name(),
                )?;
                let mut signs = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                let mut log_abs = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                for row in 0..matrix_view.len_of(Axis(0)) {
                    let result = nabled::linalg::lu::log_determinant_view(
                        &matrix_view.index_axis(Axis(0), row),
                    )
                    .map_err(|error| exec_error(self.name(), error))?;
                    signs.push(result.sign);
                    log_abs.push(result.ln_abs_det);
                }
                let log_abs = primitive_array_from_values::<Float32Type>(log_abs);
                let struct_array = StructArray::new(
                    vec![
                        Arc::new(Field::new("sign", DataType::Int8, false)),
                        scalar_field("ln_abs_det", &matrix.value_type, false),
                    ]
                    .into(),
                    vec![Arc::new(Int8Array::from(signs)), Arc::new(log_abs)],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let matrix_view = fixed_shape_tensor_view3::<Float64Type>(
                    &args.arg_fields[0],
                    matrices,
                    self.name(),
                )?;
                let mut signs = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                let mut log_abs = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                for row in 0..matrix_view.len_of(Axis(0)) {
                    let result = nabled::linalg::lu::log_determinant_view(
                        &matrix_view.index_axis(Axis(0), row),
                    )
                    .map_err(|error| exec_error(self.name(), error))?;
                    signs.push(result.sign);
                    log_abs.push(result.ln_abs_det);
                }
                let log_abs = primitive_array_from_values::<Float64Type>(log_abs);
                let struct_array = StructArray::new(
                    vec![
                        Arc::new(Field::new("sign", DataType::Int8, false)),
                        scalar_field("ln_abs_det", &matrix.value_type, false),
                    ]
                    .into(),
                    vec![Arc::new(Int8Array::from(signs)), Arc::new(log_abs)],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixCholesky {
    signature: Signature,
}

impl MatrixCholesky {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixCholesky {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_cholesky" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        let l_field = fixed_shape_tensor_field("l", &value_type, &[rows, cols], false)?;
        Ok(struct_field(self.name(), vec![l_field.as_ref().clone()], nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                let output =
                    nabled::arrow::batched::cholesky_f32(args.arg_fields[0].as_ref(), matrices)
                        .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut lower_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                for result in output {
                    lower_values.extend(result.l.iter().copied());
                }
                let (lower_field, lower_array) =
                    tensor_column("l", [batch, matrix.rows, matrix.cols], lower_values)?;
                let struct_array =
                    StructArray::new(vec![lower_field].into(), vec![lower_array], None);
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let output =
                    nabled::arrow::batched::cholesky_f64(args.arg_fields[0].as_ref(), matrices)
                        .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut lower_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                for result in output {
                    lower_values.extend(result.l.iter().copied());
                }
                let (lower_field, lower_array) =
                    tensor_column("l", [batch, matrix.rows, matrix.cols], lower_values)?;
                let struct_array =
                    StructArray::new(vec![lower_field].into(), vec![lower_array], None);
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixCholeskyInverse {
    signature: Signature,
}

impl MatrixCholeskyInverse {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixCholeskyInverse {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_cholesky_inverse" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        fixed_shape_tensor_field(self.name(), &value_type, &[rows, cols], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                nabled::linalg::cholesky::inverse_view,
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                nabled::linalg::cholesky::inverse_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixQr {
    signature: Signature,
}

impl MatrixQr {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixQr {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_qr" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let q_field =
            fixed_shape_tensor_field("q", &matrix.value_type, &[matrix.rows, matrix.cols], false)?;
        let r_field =
            fixed_shape_tensor_field("r", &matrix.value_type, &[matrix.cols, matrix.cols], false)?;
        let rank_field = Field::new("rank", DataType::Int64, false);
        Ok(struct_field(
            self.name(),
            vec![q_field.as_ref().clone(), r_field.as_ref().clone(), rank_field],
            args.arg_fields[0].is_nullable(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                let output = nabled::arrow::batched::qr_f32(
                    args.arg_fields[0].as_ref(),
                    matrices,
                    &nabled::linalg::qr::QRConfig::<f32>::default(),
                )
                .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut q_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut r_values = Vec::with_capacity(batch * matrix.cols * matrix.cols);
                let mut rank_values = Vec::with_capacity(batch);
                for result in output {
                    q_values.extend(result.q.iter().copied());
                    r_values.extend(result.r.iter().copied());
                    rank_values.push(
                        i64::try_from(result.rank)
                            .map_err(|_| exec_error(self.name(), "rank exceeds i64 limits"))?,
                    );
                }
                let (q_field, q_array) =
                    tensor_column("q", [batch, matrix.rows, matrix.cols], q_values)?;
                let (r_field, r_array) =
                    tensor_column("r", [batch, matrix.cols, matrix.cols], r_values)?;
                let struct_array = StructArray::new(
                    vec![q_field, r_field, Arc::new(Field::new("rank", DataType::Int64, false))]
                        .into(),
                    vec![q_array, r_array, Arc::new(Int64Array::from(rank_values))],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let output = nabled::arrow::batched::qr_f64(
                    args.arg_fields[0].as_ref(),
                    matrices,
                    &nabled::linalg::qr::QRConfig::<f64>::default(),
                )
                .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut q_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut r_values = Vec::with_capacity(batch * matrix.cols * matrix.cols);
                let mut rank_values = Vec::with_capacity(batch);
                for result in output {
                    q_values.extend(result.q.iter().copied());
                    r_values.extend(result.r.iter().copied());
                    rank_values.push(
                        i64::try_from(result.rank)
                            .map_err(|_| exec_error(self.name(), "rank exceeds i64 limits"))?,
                    );
                }
                let (q_field, q_array) =
                    tensor_column("q", [batch, matrix.rows, matrix.cols], q_values)?;
                let (r_field, r_array) =
                    tensor_column("r", [batch, matrix.cols, matrix.cols], r_values)?;
                let struct_array = StructArray::new(
                    vec![q_field, r_field, Arc::new(Field::new("rank", DataType::Int64, false))]
                        .into(),
                    vec![q_array, r_array, Arc::new(Int64Array::from(rank_values))],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvd {
    signature: Signature,
}

impl MatrixSvd {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvd {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = matrix.rows.min(matrix.cols);
        let u_field =
            fixed_shape_tensor_field("u", &matrix.value_type, &[matrix.rows, keep], false)?;
        let singular_values = vector_field("singular_values", &matrix.value_type, keep, false)?;
        let vt_field =
            fixed_shape_tensor_field("vt", &matrix.value_type, &[keep, matrix.cols], false)?;
        Ok(struct_field(
            self.name(),
            vec![
                u_field.as_ref().clone(),
                singular_values.as_ref().clone(),
                vt_field.as_ref().clone(),
            ],
            args.arg_fields[0].is_nullable(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = matrix.rows.min(matrix.cols);
        match matrix.value_type {
            DataType::Float32 => {
                let output = nabled::arrow::batched::svd_f32(args.arg_fields[0].as_ref(), matrices)
                    .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut u_values = Vec::with_capacity(batch * matrix.rows * keep);
                let mut singular_values = Vec::with_capacity(batch * keep);
                let mut vt_values = Vec::with_capacity(batch * keep * matrix.cols);
                for result in output {
                    u_values.extend(result.u.iter().copied());
                    singular_values.extend(result.singular_values.iter().copied());
                    vt_values.extend(result.vt.iter().copied());
                }
                let (u_field, u_array) = tensor_column("u", [batch, matrix.rows, keep], u_values)?;
                let singular_values = fixed_size_list_array_from_flat_rows::<Float32Type>(
                    self.name(),
                    batch,
                    keep,
                    &singular_values,
                )?;
                let singular_field =
                    vector_field("singular_values", &matrix.value_type, keep, false)?;
                let (vt_field, vt_array) =
                    tensor_column("vt", [batch, keep, matrix.cols], vt_values)?;
                let struct_array = StructArray::new(
                    vec![u_field, singular_field, vt_field].into(),
                    vec![u_array, Arc::new(singular_values), vt_array],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let output = nabled::arrow::batched::svd_f64(args.arg_fields[0].as_ref(), matrices)
                    .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut u_values = Vec::with_capacity(batch * matrix.rows * keep);
                let mut singular_values = Vec::with_capacity(batch * keep);
                let mut vt_values = Vec::with_capacity(batch * keep * matrix.cols);
                for result in output {
                    u_values.extend(result.u.iter().copied());
                    singular_values.extend(result.singular_values.iter().copied());
                    vt_values.extend(result.vt.iter().copied());
                }
                let (u_field, u_array) = tensor_column("u", [batch, matrix.rows, keep], u_values)?;
                let singular_values = fixed_size_list_array_from_flat_rows::<Float64Type>(
                    self.name(),
                    batch,
                    keep,
                    &singular_values,
                )?;
                let singular_field =
                    vector_field("singular_values", &matrix.value_type, keep, false)?;
                let (vt_field, vt_array) =
                    tensor_column("vt", [batch, keep, matrix.cols], vt_values)?;
                let struct_array = StructArray::new(
                    vec![u_field, singular_field, vt_field].into(),
                    vec![u_array, Arc::new(singular_values), vt_array],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixQrSolveLeastSquares {
    signature: Signature,
}

impl MatrixQrSolveLeastSquares {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixQrSolveLeastSquares {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_qr_solve_least_squares" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix.value_type != rhs.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {}, rhs {}",
                    matrix.value_type, rhs.value_type
                ),
            ));
        }
        if rhs.len != matrix.rows {
            return Err(exec_error(
                self.name(),
                format!("rhs vector length mismatch: expected {}, found {}", matrix.rows, rhs.len),
            ));
        }
        vector_field(self.name(), &matrix.value_type, matrix.cols, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                invoke_matrix_qr_vector_output::<Float32Type, _>(&args, self.name(), |m, rhs| {
                    nabled::linalg::qr::solve_least_squares_view(
                        m,
                        rhs,
                        &nabled::linalg::qr::QRConfig::<f32>::default(),
                    )
                })
            }
            DataType::Float64 => {
                invoke_matrix_qr_vector_output::<Float64Type, _>(&args, self.name(), |m, rhs| {
                    nabled::linalg::qr::solve_least_squares_view(
                        m,
                        rhs,
                        &nabled::linalg::qr::QRConfig::<f64>::default(),
                    )
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixQrConditionNumber {
    signature: Signature,
}

impl MatrixQrConditionNumber {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixQrConditionNumber {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_qr_condition_number" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        Ok(scalar_field(self.name(), &matrix.value_type, args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => {
                invoke_matrix_scalar::<Float32Type, _>(&args, self.name(), |matrix| {
                    let qr = nabled::linalg::qr::decompose_view(
                        matrix,
                        &nabled::linalg::qr::QRConfig::<f32>::default(),
                    )
                    .map_err(|error| exec_error("matrix_qr_condition_number", error))?;
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::linalg::qr::condition_number(&qr),
                    )
                })
            }
            DataType::Float64 => {
                invoke_matrix_scalar::<Float64Type, _>(&args, self.name(), |matrix| {
                    let qr = nabled::linalg::qr::decompose_view(
                        matrix,
                        &nabled::linalg::qr::QRConfig::<f64>::default(),
                    )
                    .map_err(|error| exec_error("matrix_qr_condition_number", error))?;
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::linalg::qr::condition_number(&qr),
                    )
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdPseudoInverse {
    signature: Signature,
}

impl MatrixSvdPseudoInverse {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvdPseudoInverse {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_pseudo_inverse" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.cols, matrix.rows],
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.cols,
                matrix.rows,
                |m| {
                    nabled::linalg::svd::pseudo_inverse_view(
                        m,
                        &nabled::linalg::svd::PseudoInverseConfig::<f32>::default(),
                    )
                },
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.cols,
                matrix.rows,
                |m| {
                    nabled::linalg::svd::pseudo_inverse_view(
                        m,
                        &nabled::linalg::svd::PseudoInverseConfig::<f64>::default(),
                    )
                },
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdConditionNumber {
    signature: Signature,
}

impl MatrixSvdConditionNumber {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvdConditionNumber {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_condition_number" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        Ok(scalar_field(self.name(), &matrix.value_type, args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => {
                invoke_matrix_scalar::<Float32Type, _>(&args, self.name(), |matrix| {
                    let svd = nabled::linalg::svd::decompose_view(matrix)
                        .map_err(|error| exec_error("matrix_svd_condition_number", error))?;
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::linalg::svd::condition_number(&svd),
                    )
                })
            }
            DataType::Float64 => {
                invoke_matrix_scalar::<Float64Type, _>(&args, self.name(), |matrix| {
                    let svd = nabled::linalg::svd::decompose_view(matrix)
                        .map_err(|error| exec_error("matrix_svd_condition_number", error))?;
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::linalg::svd::condition_number(&svd),
                    )
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdRank {
    signature: Signature,
}

impl MatrixSvdRank {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvdRank {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_rank" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        Ok(int64_scalar_field(self.name(), args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => {
                invoke_matrix_scalar_i64::<Float32Type, _>(&args, self.name(), |matrix| {
                    let svd = nabled::linalg::svd::decompose_view(matrix)
                        .map_err(|error| exec_error("matrix_svd_rank", error))?;
                    i64::try_from(nabled::linalg::svd::rank(&svd, None))
                        .map_err(|_| exec_error("matrix_svd_rank", "rank exceeds i64 limits"))
                })
            }
            DataType::Float64 => {
                invoke_matrix_scalar_i64::<Float64Type, _>(&args, self.name(), |matrix| {
                    let svd = nabled::linalg::svd::decompose_view(matrix)
                        .map_err(|error| exec_error("matrix_svd_rank", error))?;
                    i64::try_from(nabled::linalg::svd::rank(&svd, None))
                        .map_err(|_| exec_error("matrix_svd_rank", "rank exceeds i64 limits"))
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[must_use]
pub fn matrix_lu_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MatrixLu::new())) }

#[must_use]
pub fn matrix_inverse_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixInverse::new()))
}

#[must_use]
pub fn matrix_determinant_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixDeterminant::new()))
}

#[must_use]
pub fn matrix_log_determinant_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixLogDeterminant::new()))
}

#[must_use]
pub fn matrix_cholesky_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixCholesky::new()))
}

#[must_use]
pub fn matrix_cholesky_inverse_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixCholeskyInverse::new()))
}

#[must_use]
pub fn matrix_qr_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MatrixQr::new())) }

#[must_use]
pub fn matrix_svd_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MatrixSvd::new())) }

#[must_use]
pub fn matrix_qr_solve_least_squares_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixQrSolveLeastSquares::new()))
}

#[must_use]
pub fn matrix_qr_condition_number_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixQrConditionNumber::new()))
}

#[must_use]
pub fn matrix_svd_pseudo_inverse_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdPseudoInverse::new()))
}

#[must_use]
pub fn matrix_svd_condition_number_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdConditionNumber::new()))
}

#[must_use]
pub fn matrix_svd_rank_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdRank::new()))
}
