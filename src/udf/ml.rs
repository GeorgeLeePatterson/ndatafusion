use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::StructArray;
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use ndarray::{Array1, Array2, Array3, Axis};
use ndarrow::IntoArrow;

use super::common::{
    expect_bool_scalar_arg, expect_bool_scalar_argument, expect_fixed_size_list_arg,
    fixed_shape_tensor_view3_f64, fixed_size_list_view2_f64, nullable_or,
};
use crate::error::exec_error;
use crate::metadata::{
    fixed_shape_tensor_field, float64_scalar_field, parse_float64_matrix_batch_field,
    parse_float64_vector_field, struct_field, vector_field,
};
use crate::signatures::any_signature;

fn invoke_matrix_batch_to_vector_output(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(
        &ndarray::ArrayView2<'_, f64>,
    ) -> std::result::Result<Array1<f64>, datafusion::common::DataFusionError>,
) -> Result<ColumnarValue> {
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let cols = matrix_view.len_of(Axis(2));
    let mut output = Vec::with_capacity(batch * cols);
    for row in 0..batch {
        let values = op(&matrix_view.index_axis(Axis(0), row))?;
        output.extend(values.iter().copied());
    }
    let output = Array2::from_shape_vec((batch, cols), output)
        .map_err(|error| exec_error(function_name, error))?;
    let output = output.into_arrow().map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_matrix_batch_to_tensor_output(
    args: &ScalarFunctionArgs,
    function_name: &str,
    output_rows: usize,
    output_cols: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, f64>,
    ) -> std::result::Result<Array2<f64>, datafusion::common::DataFusionError>,
) -> Result<ColumnarValue> {
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut output = Vec::with_capacity(batch * output_rows * output_cols);
    for row in 0..batch {
        let values = op(&matrix_view.index_axis(Axis(0), row))?;
        output.extend(values.iter().copied());
    }
    let output = Array3::from_shape_vec((batch, output_rows, output_cols), output)
        .map_err(|error| exec_error(function_name, error))?;
    let (_field, output) = ndarrow::arrayd_to_fixed_shape_tensor(function_name, output.into_dyn())
        .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixColumnMeans {
    signature: Signature,
}

impl MatrixColumnMeans {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixColumnMeans {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_column_means" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let [_rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        vector_field(self.name(), cols, args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        invoke_matrix_batch_to_vector_output(&args, self.name(), |matrix| {
            Ok(nabled::ml::stats::column_means_view(matrix))
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixCenterColumns {
    signature: Signature,
}

impl MatrixCenterColumns {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixCenterColumns {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_center_columns" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(self.name(), &[rows, cols], args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        invoke_matrix_batch_to_tensor_output(&args, self.name(), rows, cols, |matrix| {
            Ok(nabled::ml::stats::center_columns_view(matrix))
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixCovariance {
    signature: Signature,
}

impl MatrixCovariance {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixCovariance {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_covariance" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let [_rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(self.name(), &[cols, cols], args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [_rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        invoke_matrix_batch_to_tensor_output(&args, self.name(), cols, cols, |matrix| {
            nabled::ml::stats::covariance_matrix_view(matrix)
                .map_err(|error| exec_error("matrix_covariance", error))
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixCorrelation {
    signature: Signature,
}

impl MatrixCorrelation {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixCorrelation {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_correlation" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let [_rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(self.name(), &[cols, cols], args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let [_rows, cols] = parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        invoke_matrix_batch_to_tensor_output(&args, self.name(), cols, cols, |matrix| {
            nabled::ml::stats::correlation_matrix_view(matrix)
                .map_err(|error| exec_error("matrix_correlation", error))
        })
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixPca {
    signature: Signature,
}

impl MatrixPca {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixPca {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_pca" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let [observations, features] =
            parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = observations.min(features);
        let components = fixed_shape_tensor_field("components", &[keep, features], false)?;
        let explained_variance = vector_field("explained_variance", keep, false)?;
        let explained_variance_ratio = vector_field("explained_variance_ratio", keep, false)?;
        let mean = vector_field("mean", features, false)?;
        let scores = fixed_shape_tensor_field("scores", &[observations, keep], false)?;
        Ok(struct_field(
            self.name(),
            vec![
                components.as_ref().clone(),
                explained_variance.as_ref().clone(),
                explained_variance_ratio.as_ref().clone(),
                mean.as_ref().clone(),
                scores.as_ref().clone(),
            ],
            args.arg_fields[0].is_nullable(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let [observations, features] =
            parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = observations.min(features);
        let matrix_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], matrices, self.name())?;
        let batch = matrix_view.len_of(Axis(0));

        let mut components_values = Vec::with_capacity(batch * keep * features);
        let mut explained_variance_values = Vec::with_capacity(batch * keep);
        let mut explained_variance_ratio_values = Vec::with_capacity(batch * keep);
        let mut mean_values = Vec::with_capacity(batch * features);
        let mut scores_values = Vec::with_capacity(batch * observations * keep);

        for row in 0..batch {
            let result =
                nabled::ml::pca::compute_pca_view(&matrix_view.index_axis(Axis(0), row), None)
                    .map_err(|error| exec_error(self.name(), error))?;
            components_values.extend(result.components.iter().copied());
            explained_variance_values.extend(result.explained_variance.iter().copied());
            explained_variance_ratio_values.extend(result.explained_variance_ratio.iter().copied());
            mean_values.extend(result.mean.iter().copied());
            scores_values.extend(result.scores.iter().copied());
        }

        let components = Array3::from_shape_vec((batch, keep, features), components_values)
            .map_err(|error| exec_error(self.name(), error))?;
        let (_components_field, components) =
            ndarrow::arrayd_to_fixed_shape_tensor("components", components.into_dyn())
                .map_err(|error| exec_error(self.name(), error))?;
        let explained_variance = Array2::from_shape_vec((batch, keep), explained_variance_values)
            .map_err(|error| exec_error(self.name(), error))?
            .into_arrow()
            .map_err(|error| exec_error(self.name(), error))?;
        let explained_variance_ratio =
            Array2::from_shape_vec((batch, keep), explained_variance_ratio_values)
                .map_err(|error| exec_error(self.name(), error))?
                .into_arrow()
                .map_err(|error| exec_error(self.name(), error))?;
        let mean = Array2::from_shape_vec((batch, features), mean_values)
            .map_err(|error| exec_error(self.name(), error))?
            .into_arrow()
            .map_err(|error| exec_error(self.name(), error))?;
        let scores = Array3::from_shape_vec((batch, observations, keep), scores_values)
            .map_err(|error| exec_error(self.name(), error))?;
        let (_scores_field, scores) =
            ndarrow::arrayd_to_fixed_shape_tensor("scores", scores.into_dyn())
                .map_err(|error| exec_error(self.name(), error))?;

        let struct_array = StructArray::new(
            vec![
                fixed_shape_tensor_field("components", &[keep, features], false)?,
                vector_field("explained_variance", keep, false)?,
                vector_field("explained_variance_ratio", keep, false)?,
                vector_field("mean", features, false)?,
                fixed_shape_tensor_field("scores", &[observations, keep], false)?,
            ]
            .into(),
            vec![
                Arc::new(components),
                Arc::new(explained_variance),
                Arc::new(explained_variance_ratio),
                Arc::new(mean),
                Arc::new(scores),
            ],
            None,
        );
        Ok(ColumnarValue::Array(Arc::new(struct_array)))
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct LinearRegression {
    signature: Signature,
}

impl LinearRegression {
    fn new() -> Self { Self { signature: any_signature(3) } }
}

impl ScalarUDFImpl for LinearRegression {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "linear_regression" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let [observations, features] =
            parse_float64_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let vector_len = parse_float64_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if vector_len != observations {
            return Err(exec_error(
                self.name(),
                format!(
                    "response vector length mismatch: expected {observations}, found {vector_len}"
                ),
            ));
        }
        let add_intercept = expect_bool_scalar_argument(&args, 3, self.name())?;
        let coefficient_len = features + usize::from(add_intercept);
        let coefficients = vector_field("coefficients", coefficient_len, false)?;
        let fitted_values = vector_field("fitted_values", observations, false)?;
        let residuals = vector_field("residuals", observations, false)?;
        let r_squared = float64_scalar_field("r_squared", false);
        Ok(struct_field(
            self.name(),
            vec![
                coefficients.as_ref().clone(),
                fitted_values.as_ref().clone(),
                residuals.as_ref().clone(),
                r_squared.as_ref().clone(),
            ],
            nullable_or(args.arg_fields),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let design = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let response = expect_fixed_size_list_arg(&args, 2, self.name())?;
        let add_intercept = expect_bool_scalar_arg(&args, 3, self.name())?;
        let design_view = fixed_shape_tensor_view3_f64(&args.arg_fields[0], design, self.name())?;
        let response_view = fixed_size_list_view2_f64(response, self.name())?;
        if design_view.len_of(Axis(0)) != response_view.nrows() {
            return Err(exec_error(
                self.name(),
                format!(
                    "batch length mismatch: {} design matrices vs {} response vectors",
                    design_view.len_of(Axis(0)),
                    response_view.nrows()
                ),
            ));
        }

        let batch = design_view.len_of(Axis(0));
        let fitted_len = response_view.ncols();
        let coefficient_len = design_view.len_of(Axis(2)) + usize::from(add_intercept);
        let mut coefficients = Vec::with_capacity(batch * coefficient_len);
        let mut fitted_values = Vec::with_capacity(batch * fitted_len);
        let mut residuals = Vec::with_capacity(batch * fitted_len);
        let mut r_squared = Vec::with_capacity(batch);

        for row in 0..batch {
            let result = nabled::ml::regression::linear_regression_view(
                &design_view.index_axis(Axis(0), row),
                &response_view.index_axis(Axis(0), row),
                add_intercept,
            )
            .map_err(|error| exec_error(self.name(), error))?;
            coefficients.extend(result.coefficients.iter().copied());
            fitted_values.extend(result.fitted_values.iter().copied());
            residuals.extend(result.residuals.iter().copied());
            r_squared.push(result.r_squared);
        }

        let coefficients = Array2::from_shape_vec((batch, coefficient_len), coefficients)
            .map_err(|error| exec_error(self.name(), error))?
            .into_arrow()
            .map_err(|error| exec_error(self.name(), error))?;
        let fitted_values = Array2::from_shape_vec((batch, fitted_len), fitted_values)
            .map_err(|error| exec_error(self.name(), error))?
            .into_arrow()
            .map_err(|error| exec_error(self.name(), error))?;
        let residuals = Array2::from_shape_vec((batch, fitted_len), residuals)
            .map_err(|error| exec_error(self.name(), error))?
            .into_arrow()
            .map_err(|error| exec_error(self.name(), error))?;
        let r_squared = Array1::from_vec(r_squared)
            .into_arrow()
            .map_err(|error| exec_error(self.name(), error))?;

        let coefficients_field = vector_field("coefficients", coefficient_len, false)?;
        let fitted_values_field = vector_field("fitted_values", fitted_len, false)?;
        let residuals_field = vector_field("residuals", fitted_len, false)?;
        let r_squared_field = float64_scalar_field("r_squared", false);
        let struct_array = StructArray::new(
            vec![coefficients_field, fitted_values_field, residuals_field, r_squared_field].into(),
            vec![
                Arc::new(coefficients),
                Arc::new(fitted_values),
                Arc::new(residuals),
                Arc::new(r_squared),
            ],
            None,
        );
        Ok(ColumnarValue::Array(Arc::new(struct_array)))
    }
}

#[must_use]
pub fn matrix_column_means_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixColumnMeans::new()))
}

#[must_use]
pub fn matrix_center_columns_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixCenterColumns::new()))
}

#[must_use]
pub fn matrix_covariance_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixCovariance::new()))
}

#[must_use]
pub fn matrix_correlation_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixCorrelation::new()))
}

#[must_use]
pub fn matrix_pca_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MatrixPca::new())) }

#[must_use]
pub fn linear_regression_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(LinearRegression::new()))
}
