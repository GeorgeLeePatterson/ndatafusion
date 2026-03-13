use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::{Float64Array, Int8Array, Int64Array, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use ndarray::Array3;
use ndarrow::IntoArrow;

use super::common::{expect_fixed_size_list_arg, fixed_shape_tensor_view3_f64, nullable_or};
use crate::error::exec_error;
use crate::metadata::{
    fixed_shape_tensor_field, float64_scalar_field, parse_float64_matrix_batch_field, struct_field,
    vector_field,
};
use crate::signatures::any_signature;

fn square_matrix_shape(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(usize, usize, bool)> {
    let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    if rows != cols {
        return Err(exec_error(
            function_name,
            format!("{function_name} requires square matrices, found ({rows}, {cols})"),
        ));
    }
    Ok((rows, cols, nullable_or(args.arg_fields)))
}

fn tensor_column(
    name: &str,
    shape: &[usize],
    values: Vec<f64>,
) -> Result<(FieldRef, Arc<datafusion::arrow::array::FixedSizeListArray>)> {
    let tensor = Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
        .map_err(|error| exec_error(name, error))?;
    let (field, array) = ndarrow::arrayd_to_fixed_shape_tensor(name, tensor.into_dyn())
        .map_err(|error| exec_error(name, error))?;
    Ok((Arc::new(field), Arc::new(array)))
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
        let (rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        let l_field = fixed_shape_tensor_field("l", &[rows, cols], false)?;
        let u_field = fixed_shape_tensor_field("u", &[rows, cols], false)?;
        Ok(struct_field(
            self.name(),
            vec![l_field.as_ref().clone(), u_field.as_ref().clone()],
            nullable,
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let output = nabled::arrow::batched::lu_f64(args.arg_fields[0].as_ref(), matrices)
            .map_err(|error| exec_error(self.name(), error))?;
        let batch = output.len();
        let mut lower_values = Vec::with_capacity(batch * rows * cols);
        let mut upper_values = Vec::with_capacity(batch * rows * cols);
        for result in output {
            lower_values.extend(result.l.iter().copied());
            upper_values.extend(result.u.iter().copied());
        }

        let (lower_field, lower_array) = tensor_column("l", &[batch, rows, cols], lower_values)?;
        let (upper_field, upper_array) = tensor_column("u", &[batch, rows, cols], upper_values)?;
        let struct_array = StructArray::new(
            vec![lower_field, upper_field].into(),
            vec![lower_array, upper_array],
            None,
        );
        Ok(ColumnarValue::Array(Arc::new(struct_array)))
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
        let (rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        fixed_shape_tensor_field(self.name(), &[rows, cols], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], matrices, self.name())?;
        let shape = matrix_view.raw_dim();
        let mut output = Vec::with_capacity(matrix_view.len());
        for row in 0..shape[0] {
            let inverse =
                nabled::linalg::lu::inverse_view(&matrix_view.index_axis(ndarray::Axis(0), row))
                    .map_err(|error| exec_error(self.name(), error))?;
            output.extend(inverse.iter().copied());
        }
        let (_field, array) = tensor_column(self.name(), &[shape[0], shape[1], shape[2]], output)?;
        Ok(ColumnarValue::Array(array))
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
        let (_rows, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        Ok(float64_scalar_field(self.name(), nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], matrices, self.name())?;
        let mut values = Vec::with_capacity(matrix_view.len_of(ndarray::Axis(0)));
        for row in 0..matrix_view.len_of(ndarray::Axis(0)) {
            values.push(
                nabled::linalg::lu::determinant_view(
                    &matrix_view.index_axis(ndarray::Axis(0), row),
                )
                .map_err(|error| exec_error(self.name(), error))?,
            );
        }
        Ok(ColumnarValue::Array(Arc::new(Float64Array::from(values))))
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
        let (_rows, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        let sign_field = Field::new("sign", DataType::Int8, false);
        let log_abs_field = float64_scalar_field("ln_abs_det", false);
        Ok(struct_field(self.name(), vec![sign_field, log_abs_field.as_ref().clone()], nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], matrices, self.name())?;
        let mut signs = Vec::with_capacity(matrix_view.len_of(ndarray::Axis(0)));
        let mut log_abs = Vec::with_capacity(matrix_view.len_of(ndarray::Axis(0)));
        for row in 0..matrix_view.len_of(ndarray::Axis(0)) {
            let result = nabled::linalg::lu::log_determinant_view(
                &matrix_view.index_axis(ndarray::Axis(0), row),
            )
            .map_err(|error| exec_error(self.name(), error))?;
            signs.push(result.sign);
            log_abs.push(result.ln_abs_det);
        }
        let struct_array = StructArray::new(
            vec![
                Arc::new(Field::new("sign", DataType::Int8, false)),
                float64_scalar_field("ln_abs_det", false),
            ]
            .into(),
            vec![Arc::new(Int8Array::from(signs)), Arc::new(Float64Array::from(log_abs))],
            None,
        );
        Ok(ColumnarValue::Array(Arc::new(struct_array)))
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
        let (rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        let l_field = fixed_shape_tensor_field("l", &[rows, cols], false)?;
        Ok(struct_field(self.name(), vec![l_field.as_ref().clone()], nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let output = nabled::arrow::batched::cholesky_f64(args.arg_fields[0].as_ref(), matrices)
            .map_err(|error| exec_error(self.name(), error))?;
        let batch = output.len();
        let mut lower_values = Vec::with_capacity(batch * rows * cols);
        for result in output {
            lower_values.extend(result.l.iter().copied());
        }
        let (lower_field, lower_array) = tensor_column("l", &[batch, rows, cols], lower_values)?;
        let struct_array = StructArray::new(vec![lower_field].into(), vec![lower_array], None);
        Ok(ColumnarValue::Array(Arc::new(struct_array)))
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
        let (rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        fixed_shape_tensor_field(self.name(), &[rows, cols], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], matrices, self.name())?;
        let shape = matrix_view.raw_dim();
        let mut output = Vec::with_capacity(matrix_view.len());
        for row in 0..shape[0] {
            let inverse = nabled::linalg::cholesky::inverse_view(
                &matrix_view.index_axis(ndarray::Axis(0), row),
            )
            .map_err(|error| exec_error(self.name(), error))?;
            output.extend(inverse.iter().copied());
        }
        let (_field, array) = tensor_column(self.name(), &[shape[0], shape[1], shape[2]], output)?;
        Ok(ColumnarValue::Array(array))
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
        let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let nullable = args.arg_fields[0].is_nullable();
        let q_field = fixed_shape_tensor_field("q", &[rows, cols], false)?;
        let r_field = fixed_shape_tensor_field("r", &[cols, cols], false)?;
        let rank_field = Field::new("rank", DataType::Int64, false);
        Ok(struct_field(
            self.name(),
            vec![q_field.as_ref().clone(), r_field.as_ref().clone(), rank_field],
            nullable,
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let output = nabled::arrow::batched::qr_f64(
            args.arg_fields[0].as_ref(),
            matrices,
            &nabled::linalg::qr::QRConfig::default(),
        )
        .map_err(|error| exec_error(self.name(), error))?;
        let batch = output.len();
        let mut q_values = Vec::with_capacity(batch * rows * cols);
        let mut r_values = Vec::with_capacity(batch * cols * cols);
        let mut rank_values = Vec::with_capacity(batch);
        for result in output {
            q_values.extend(result.q.iter().copied());
            r_values.extend(result.r.iter().copied());
            rank_values.push(
                i64::try_from(result.rank)
                    .map_err(|_| exec_error(self.name(), "rank exceeds i64 limits"))?,
            );
        }
        let (q_field, q_array) = tensor_column("q", &[batch, rows, cols], q_values)?;
        let (r_field, r_array) = tensor_column("r", &[batch, cols, cols], r_values)?;
        let struct_array = StructArray::new(
            vec![q_field, r_field, Arc::new(Field::new("rank", DataType::Int64, false))].into(),
            vec![q_array, r_array, Arc::new(Int64Array::from(rank_values))],
            None,
        );
        Ok(ColumnarValue::Array(Arc::new(struct_array)))
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
        let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = rows.min(cols);
        let nullable = args.arg_fields[0].is_nullable();
        let u_field = fixed_shape_tensor_field("u", &[rows, keep], false)?;
        let singular_values = vector_field("singular_values", keep, false)?;
        let vt_field = fixed_shape_tensor_field("vt", &[keep, cols], false)?;
        Ok(struct_field(
            self.name(),
            vec![
                u_field.as_ref().clone(),
                singular_values.as_ref().clone(),
                vt_field.as_ref().clone(),
            ],
            nullable,
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = rows.min(cols);
        let output = nabled::arrow::batched::svd_f64(args.arg_fields[0].as_ref(), matrices)
            .map_err(|error| exec_error(self.name(), error))?;
        let batch = output.len();
        let mut u_values = Vec::with_capacity(batch * rows * keep);
        let mut singular_values = Vec::with_capacity(batch * keep);
        let mut vt_values = Vec::with_capacity(batch * keep * cols);
        for result in output {
            u_values.extend(result.u.iter().copied());
            singular_values.extend(result.singular_values.iter().copied());
            vt_values.extend(result.vt.iter().copied());
        }
        let (u_field, u_array) = tensor_column("u", &[batch, rows, keep], u_values)?;
        let singular_values = ndarray::Array2::from_shape_vec((batch, keep), singular_values)
            .map_err(|error| exec_error(self.name(), error))?
            .into_arrow()
            .map_err(|error| exec_error(self.name(), error))?;
        let singular_field = vector_field("singular_values", keep, false)?;
        let (vt_field, vt_array) = tensor_column("vt", &[batch, keep, cols], vt_values)?;
        let struct_array = StructArray::new(
            vec![u_field, singular_field, vt_field].into(),
            vec![u_array, Arc::new(singular_values), vt_array],
            None,
        );
        Ok(ColumnarValue::Array(Arc::new(struct_array)))
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
