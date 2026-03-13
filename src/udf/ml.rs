use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::StructArray;
use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Array1, Array2, Array3, ArrayView2, ArrayView3, Axis};
use ndarrow::NdarrowElement;

use super::common::{
    expect_bool_scalar_arg, expect_bool_scalar_argument, expect_fixed_size_list_arg,
    fixed_shape_tensor_view3, fixed_size_list_array_from_flat_rows, fixed_size_list_view2,
    nullable_or, primitive_array_from_values,
};
use crate::error::exec_error;
use crate::metadata::{
    MatrixBatchContract, fixed_shape_tensor_field, parse_matrix_batch_field, parse_vector_field,
    scalar_field, struct_field, vector_field,
};
use crate::signatures::any_signature;

fn invoke_matrix_batch_to_vector_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(&ArrayView2<'_, T::Native>) -> std::result::Result<Array1<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let cols = matrix_view.len_of(Axis(2));
    let mut output = Vec::with_capacity(batch * cols);
    for row in 0..batch {
        let values = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        output.extend(values.iter().copied());
    }
    let output = fixed_size_list_array_from_flat_rows::<T>(function_name, batch, cols, &output)?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_matrix_batch_to_tensor_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    output_rows: usize,
    output_cols: usize,
    op: impl Fn(&ArrayView2<'_, T::Native>) -> std::result::Result<Array2<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
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
    let output = Array3::from_shape_vec((batch, output_rows, output_cols), output)
        .map_err(|error| exec_error(function_name, error))?;
    let (_field, output) = ndarrow::arrayd_to_fixed_shape_tensor(function_name, output.into_dyn())
        .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

struct PcaOutputs<T> {
    batch:                           usize,
    components_values:               Vec<T>,
    explained_variance_values:       Vec<T>,
    explained_variance_ratio_values: Vec<T>,
    mean_values:                     Vec<T>,
    scores_values:                   Vec<T>,
}

fn collect_pca_outputs<T>(
    function_name: &str,
    matrix_view: &ArrayView3<'_, T::Native>,
) -> Result<PcaOutputs<T::Native>>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
{
    let batch = matrix_view.len_of(Axis(0));
    let rows = matrix_view.len_of(Axis(1));
    let cols = matrix_view.len_of(Axis(2));
    let keep = rows.min(cols);
    let mut outputs = PcaOutputs {
        batch,
        components_values: Vec::with_capacity(batch * keep * cols),
        explained_variance_values: Vec::with_capacity(batch * keep),
        explained_variance_ratio_values: Vec::with_capacity(batch * keep),
        mean_values: Vec::with_capacity(batch * cols),
        scores_values: Vec::with_capacity(batch * rows * keep),
    };
    for row in 0..batch {
        let result = nabled::ml::pca::compute_pca_view(&matrix_view.index_axis(Axis(0), row), None)
            .map_err(|error| exec_error(function_name, error))?;
        outputs.components_values.extend(result.components.iter().copied());
        outputs.explained_variance_values.extend(result.explained_variance.iter().copied());
        outputs
            .explained_variance_ratio_values
            .extend(result.explained_variance_ratio.iter().copied());
        outputs.mean_values.extend(result.mean.iter().copied());
        outputs.scores_values.extend(result.scores.iter().copied());
    }
    Ok(outputs)
}

fn build_pca_struct_array<T>(
    function_name: &str,
    matrix: &MatrixBatchContract,
    outputs: PcaOutputs<T::Native>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let keep = matrix.rows.min(matrix.cols);
    let components =
        Array3::from_shape_vec((outputs.batch, keep, matrix.cols), outputs.components_values)
            .map_err(|error| exec_error(function_name, error))?;
    let (_components_field, components) =
        ndarrow::arrayd_to_fixed_shape_tensor("components", components.into_dyn())
            .map_err(|error| exec_error(function_name, error))?;
    let explained_variance = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        outputs.batch,
        keep,
        &outputs.explained_variance_values,
    )?;
    let explained_variance_ratio = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        outputs.batch,
        keep,
        &outputs.explained_variance_ratio_values,
    )?;
    let mean = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        outputs.batch,
        matrix.cols,
        &outputs.mean_values,
    )?;
    let scores = Array3::from_shape_vec((outputs.batch, matrix.rows, keep), outputs.scores_values)
        .map_err(|error| exec_error(function_name, error))?;
    let (_scores_field, scores) =
        ndarrow::arrayd_to_fixed_shape_tensor("scores", scores.into_dyn())
            .map_err(|error| exec_error(function_name, error))?;
    let struct_array = StructArray::new(
        vec![
            fixed_shape_tensor_field(
                "components",
                &matrix.value_type,
                &[keep, matrix.cols],
                false,
            )?,
            vector_field("explained_variance", &matrix.value_type, keep, false)?,
            vector_field("explained_variance_ratio", &matrix.value_type, keep, false)?,
            vector_field("mean", &matrix.value_type, matrix.cols, false)?,
            fixed_shape_tensor_field("scores", &matrix.value_type, &[matrix.rows, keep], false)?,
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

fn invoke_matrix_pca_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    matrix: &MatrixBatchContract,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let outputs = collect_pca_outputs::<T>(function_name, &matrix_view)?;
    build_pca_struct_array::<T>(function_name, matrix, outputs)
}

struct LinearRegressionOutputs<T> {
    batch:           usize,
    coefficient_len: usize,
    fitted_len:      usize,
    coefficients:    Vec<T>,
    fitted_values:   Vec<T>,
    residuals:       Vec<T>,
    r_squared:       Vec<T>,
}

fn collect_linear_regression_outputs<T>(
    function_name: &str,
    design_view: &ArrayView3<'_, T::Native>,
    response_view: &ArrayView2<'_, T::Native>,
    add_intercept: bool,
) -> Result<LinearRegressionOutputs<T::Native>>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
{
    if design_view.len_of(Axis(0)) != response_view.nrows() {
        return Err(exec_error(
            function_name,
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
    let mut outputs = LinearRegressionOutputs {
        batch,
        coefficient_len,
        fitted_len,
        coefficients: Vec::with_capacity(batch * coefficient_len),
        fitted_values: Vec::with_capacity(batch * fitted_len),
        residuals: Vec::with_capacity(batch * fitted_len),
        r_squared: Vec::with_capacity(batch),
    };
    for row in 0..batch {
        let result = nabled::ml::regression::linear_regression_view(
            &design_view.index_axis(Axis(0), row),
            &response_view.index_axis(Axis(0), row),
            add_intercept,
        )
        .map_err(|error| exec_error(function_name, error))?;
        outputs.coefficients.extend(result.coefficients.iter().copied());
        outputs.fitted_values.extend(result.fitted_values.iter().copied());
        outputs.residuals.extend(result.residuals.iter().copied());
        outputs.r_squared.push(result.r_squared);
    }
    Ok(outputs)
}

fn build_linear_regression_struct_array<T>(
    function_name: &str,
    value_type: &DataType,
    outputs: LinearRegressionOutputs<T::Native>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
{
    let coefficients = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        outputs.batch,
        outputs.coefficient_len,
        &outputs.coefficients,
    )?;
    let fitted_values = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        outputs.batch,
        outputs.fitted_len,
        &outputs.fitted_values,
    )?;
    let residuals = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        outputs.batch,
        outputs.fitted_len,
        &outputs.residuals,
    )?;
    let r_squared = primitive_array_from_values::<T>(outputs.r_squared);
    let struct_array = StructArray::new(
        vec![
            vector_field("coefficients", value_type, outputs.coefficient_len, false)?,
            vector_field("fitted_values", value_type, outputs.fitted_len, false)?,
            vector_field("residuals", value_type, outputs.fitted_len, false)?,
            scalar_field("r_squared", value_type, false),
        ]
        .into(),
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

fn invoke_linear_regression_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    design: &MatrixBatchContract,
    add_intercept: bool,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
{
    let design_array = expect_fixed_size_list_arg(args, 1, function_name)?;
    let response_array = expect_fixed_size_list_arg(args, 2, function_name)?;
    let design_view =
        fixed_shape_tensor_view3::<T>(&args.arg_fields[0], design_array, function_name)?;
    let response_view = fixed_size_list_view2::<T>(response_array, function_name)?;
    let outputs = collect_linear_regression_outputs::<T>(
        function_name,
        &design_view,
        &response_view,
        add_intercept,
    )?;
    build_linear_regression_struct_array::<T>(function_name, &design.value_type, outputs)
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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        vector_field(self.name(), &matrix.value_type, matrix.cols, args.arg_fields[0].is_nullable())
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => invoke_matrix_batch_to_vector_output::<Float32Type, _>(
                &args,
                self.name(),
                |matrix| {
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::ml::stats::column_means_view(matrix),
                    )
                },
            ),
            DataType::Float64 => invoke_matrix_batch_to_vector_output::<Float64Type, _>(
                &args,
                self.name(),
                |matrix| {
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::ml::stats::column_means_view(matrix),
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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, matrix.cols],
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_batch_to_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| {
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::ml::stats::center_columns_view(view),
                    )
                },
            ),
            DataType::Float64 => invoke_matrix_batch_to_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| {
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::ml::stats::center_columns_view(view),
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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.cols, matrix.cols],
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_batch_to_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.cols,
                matrix.cols,
                nabled::ml::stats::covariance_matrix_view,
            ),
            DataType::Float64 => invoke_matrix_batch_to_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.cols,
                matrix.cols,
                nabled::ml::stats::covariance_matrix_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.cols, matrix.cols],
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_batch_to_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.cols,
                matrix.cols,
                nabled::ml::stats::correlation_matrix_view,
            ),
            DataType::Float64 => invoke_matrix_batch_to_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.cols,
                matrix.cols,
                nabled::ml::stats::correlation_matrix_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = matrix.rows.min(matrix.cols);
        let components = fixed_shape_tensor_field(
            "components",
            &matrix.value_type,
            &[keep, matrix.cols],
            false,
        )?;
        let explained_variance =
            vector_field("explained_variance", &matrix.value_type, keep, false)?;
        let explained_variance_ratio =
            vector_field("explained_variance_ratio", &matrix.value_type, keep, false)?;
        let mean = vector_field("mean", &matrix.value_type, matrix.cols, false)?;
        let scores =
            fixed_shape_tensor_field("scores", &matrix.value_type, &[matrix.rows, keep], false)?;
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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                invoke_matrix_pca_typed::<Float32Type>(&args, self.name(), &matrix)
            }
            DataType::Float64 => {
                invoke_matrix_pca_typed::<Float64Type>(&args, self.name(), &matrix)
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let design = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let response = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if design.value_type != response.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: design {}, response {}",
                    design.value_type, response.value_type
                ),
            ));
        }
        if response.len != design.rows {
            return Err(exec_error(
                self.name(),
                format!(
                    "response vector length mismatch: expected {}, found {}",
                    design.rows, response.len
                ),
            ));
        }
        let add_intercept = expect_bool_scalar_argument(&args, 3, self.name())?;
        let coefficient_len = design.cols + usize::from(add_intercept);
        let coefficients =
            vector_field("coefficients", &design.value_type, coefficient_len, false)?;
        let fitted_values = vector_field("fitted_values", &design.value_type, design.rows, false)?;
        let residuals = vector_field("residuals", &design.value_type, design.rows, false)?;
        let r_squared = scalar_field("r_squared", &design.value_type, false);
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
        let add_intercept = expect_bool_scalar_arg(&args, 3, self.name())?;
        let design_contract = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match design_contract.value_type {
            DataType::Float32 => invoke_linear_regression_typed::<Float32Type>(
                &args,
                self.name(),
                &design_contract,
                add_intercept,
            ),
            DataType::Float64 => invoke_linear_regression_typed::<Float64Type>(
                &args,
                self.name(),
                &design_contract,
                add_intercept,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
