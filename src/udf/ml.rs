use std::any::Any;
use std::sync::Arc;

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::array::{Array, FixedSizeListArray, StructArray};
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
    expect_struct_arg, fixed_shape_tensor_view3, fixed_size_list_array_from_flat_rows,
    fixed_size_list_view2, nullable_or, primitive_array_from_values,
};
use crate::error::{exec_error, plan_error};
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

#[derive(Debug, Clone, PartialEq, Eq)]
struct PcaContract {
    value_type: DataType,
    rows:       usize,
    cols:       usize,
    keep:       usize,
}

fn parse_pca_field(field: &FieldRef, function_name: &str, position: usize) -> Result<PcaContract> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} must be a PCA struct result"),
        ));
    };
    if fields.len() != 5 {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} must be a PCA struct with 5 fields, found {}",
                fields.len()
            ),
        ));
    }
    for (index, expected) in
        ["components", "explained_variance", "explained_variance_ratio", "mean", "scores"]
            .into_iter()
            .enumerate()
    {
        if fields[index].name() != expected {
            return Err(plan_error(
                function_name,
                format!(
                    "argument {position} field {} must be named {expected}, found {}",
                    index + 1,
                    fields[index].name()
                ),
            ));
        }
    }

    let components = parse_matrix_batch_field(&Arc::clone(&fields[0]), function_name, position)?;
    let explained = parse_vector_field(&Arc::clone(&fields[1]), function_name, position)?;
    let explained_ratio = parse_vector_field(&Arc::clone(&fields[2]), function_name, position)?;
    let mean = parse_vector_field(&Arc::clone(&fields[3]), function_name, position)?;
    let scores = parse_matrix_batch_field(&Arc::clone(&fields[4]), function_name, position)?;
    if explained.value_type != components.value_type
        || explained_ratio.value_type != components.value_type
        || mean.value_type != components.value_type
        || scores.value_type != components.value_type
    {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} PCA fields must share one value type, found components {}, \
                 explained {}, ratio {}, mean {}, scores {}",
                components.value_type,
                explained.value_type,
                explained_ratio.value_type,
                mean.value_type,
                scores.value_type,
            ),
        ));
    }
    if components.rows != explained.len
        || components.rows != explained_ratio.len
        || components.rows != scores.cols
    {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} PCA component width mismatch: components rows {}, explained \
                 {}, ratio {}, scores cols {}",
                components.rows, explained.len, explained_ratio.len, scores.cols,
            ),
        ));
    }
    if components.cols != mean.len {
        return Err(plan_error(
            function_name,
            format!(
                "argument {position} PCA feature width mismatch: components cols {}, mean len {}",
                components.cols, mean.len
            ),
        ));
    }
    Ok(PcaContract {
        value_type: components.value_type,
        rows:       scores.rows,
        cols:       components.cols,
        keep:       components.rows,
    })
}

fn expect_pca_fixed_size_list<'a>(
    pca: &'a StructArray,
    index: usize,
    function_name: &str,
    label: &str,
) -> Result<&'a FixedSizeListArray> {
    pca.column(index).as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
        exec_error(function_name, format!("PCA field {label} expected FixedSizeListArray storage"))
    })
}

fn pca_child_field(
    args: &ScalarFunctionArgs,
    index: usize,
    function_name: &str,
) -> Result<FieldRef> {
    let DataType::Struct(fields) = args.arg_fields[1].data_type() else {
        return Err(exec_error(function_name, "argument 2 expected PCA struct field metadata"));
    };
    Ok(Arc::clone(&fields[index]))
}

fn invoke_matrix_pca_transform_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    matrix: &MatrixBatchContract,
    pca: &PcaContract,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let pca_struct = expect_struct_arg(args, 2, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let components_field = pca_child_field(args, 0, function_name)?;
    let components_array = expect_pca_fixed_size_list(pca_struct, 0, function_name, "components")?;
    let mean_array = expect_pca_fixed_size_list(pca_struct, 3, function_name, "mean")?;
    let components_view =
        fixed_shape_tensor_view3::<T>(&components_field, components_array, function_name)?;
    let mean_view = fixed_size_list_view2::<T>(mean_array, function_name)?;
    if matrix_view.len_of(Axis(0)) != components_view.len_of(Axis(0))
        || matrix_view.len_of(Axis(0)) != mean_view.nrows()
    {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} matrices vs {} PCA rows",
                matrix_view.len_of(Axis(0)),
                pca_struct.len()
            ),
        ));
    }

    let batch = matrix_view.len_of(Axis(0));
    let mut values = Vec::with_capacity(batch * matrix.rows * pca.keep);
    for row in 0..batch {
        let matrix_row = matrix_view.index_axis(Axis(0), row);
        let components_row = components_view.index_axis(Axis(0), row);
        let mean_row = mean_view.index_axis(Axis(0), row);
        let mut centered = matrix_row.to_owned();
        for mut sample in centered.rows_mut() {
            sample -= &mean_row;
        }
        let scores = centered.dot(&components_row.t());
        values.extend(scores.iter().copied());
    }
    let output = Array3::from_shape_vec((batch, matrix.rows, pca.keep), values)
        .map_err(|error| exec_error(function_name, error))?;
    let (_field, output) = ndarrow::arrayd_to_fixed_shape_tensor(function_name, output.into_dyn())
        .map_err(|error| exec_error(function_name, error))?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_matrix_pca_inverse_transform_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    scores: &MatrixBatchContract,
    pca: &PcaContract,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + ndarray_linalg::Lapack<Real = T::Native>,
{
    let score_array = expect_fixed_size_list_arg(args, 1, function_name)?;
    let pca_struct = expect_struct_arg(args, 2, function_name)?;
    let score_view =
        fixed_shape_tensor_view3::<T>(&args.arg_fields[0], score_array, function_name)?;
    let components_field = pca_child_field(args, 0, function_name)?;
    let components_array = expect_pca_fixed_size_list(pca_struct, 0, function_name, "components")?;
    let mean_array = expect_pca_fixed_size_list(pca_struct, 3, function_name, "mean")?;
    let components_view =
        fixed_shape_tensor_view3::<T>(&components_field, components_array, function_name)?;
    let mean_view = fixed_size_list_view2::<T>(mean_array, function_name)?;
    if score_view.len_of(Axis(0)) != components_view.len_of(Axis(0))
        || score_view.len_of(Axis(0)) != mean_view.nrows()
    {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} score matrices vs {} PCA rows",
                score_view.len_of(Axis(0)),
                pca_struct.len()
            ),
        ));
    }

    let batch = score_view.len_of(Axis(0));
    let mut values = Vec::with_capacity(batch * scores.rows * pca.cols);
    for row in 0..batch {
        let score_row = score_view.index_axis(Axis(0), row);
        let components_row = components_view.index_axis(Axis(0), row);
        let mean_row = mean_view.index_axis(Axis(0), row);
        let mut reconstructed = score_row.dot(&components_row);
        for mut sample in reconstructed.rows_mut() {
            sample += &mean_row;
        }
        values.extend(reconstructed.iter().copied());
    }
    let output = Array3::from_shape_vec((batch, scores.rows, pca.cols), values)
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
struct MatrixPcaTransform {
    signature: Signature,
}

impl MatrixPcaTransform {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixPcaTransform {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_pca_transform" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let pca = parse_pca_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix.value_type != pca.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {}, PCA {}",
                    matrix.value_type, pca.value_type
                ),
            ));
        }
        if matrix.cols != pca.cols {
            return Err(plan_error(
                self.name(),
                format!(
                    "matrix feature width mismatch: expected {}, found {}",
                    pca.cols, matrix.cols
                ),
            ));
        }
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, pca.keep],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let pca = parse_pca_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix.value_type != pca.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {}, PCA {}",
                    matrix.value_type, pca.value_type
                ),
            ));
        }
        if matrix.cols != pca.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "matrix feature width mismatch: expected {}, found {}",
                    pca.cols, matrix.cols
                ),
            ));
        }
        match matrix.value_type {
            DataType::Float32 => {
                invoke_matrix_pca_transform_typed::<Float32Type>(&args, self.name(), &matrix, &pca)
            }
            DataType::Float64 => {
                invoke_matrix_pca_transform_typed::<Float64Type>(&args, self.name(), &matrix, &pca)
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixPcaInverseTransform {
    signature: Signature,
}

impl MatrixPcaInverseTransform {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixPcaInverseTransform {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_pca_inverse_transform" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let scores = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let pca = parse_pca_field(&args.arg_fields[1], self.name(), 2)?;
        if scores.value_type != pca.value_type {
            return Err(plan_error(
                self.name(),
                format!(
                    "value type mismatch: scores {}, PCA {}",
                    scores.value_type, pca.value_type
                ),
            ));
        }
        if scores.cols != pca.keep {
            return Err(plan_error(
                self.name(),
                format!("score width mismatch: expected {}, found {}", pca.keep, scores.cols),
            ));
        }
        fixed_shape_tensor_field(
            self.name(),
            &scores.value_type,
            &[scores.rows, pca.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let scores = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let pca = parse_pca_field(&args.arg_fields[1], self.name(), 2)?;
        if scores.value_type != pca.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: scores {}, PCA {}",
                    scores.value_type, pca.value_type
                ),
            ));
        }
        if scores.cols != pca.keep {
            return Err(exec_error(
                self.name(),
                format!("score width mismatch: expected {}, found {}", pca.keep, scores.cols),
            ));
        }
        match scores.value_type {
            DataType::Float32 => invoke_matrix_pca_inverse_transform_typed::<Float32Type>(
                &args,
                self.name(),
                &scores,
                &pca,
            ),
            DataType::Float64 => invoke_matrix_pca_inverse_transform_typed::<Float64Type>(
                &args,
                self.name(),
                &scores,
                &pca,
            ),
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
pub fn matrix_pca_transform_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixPcaTransform::new()))
}

#[must_use]
pub fn matrix_pca_inverse_transform_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixPcaInverseTransform::new()))
}

#[must_use]
pub fn linear_regression_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(LinearRegression::new()))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::Float64Array;
    use datafusion::arrow::datatypes::Field;
    use datafusion::common::ScalarValue;
    use datafusion::config::ConfigOptions;
    use datafusion::logical_expr::ScalarFunctionArgs;

    use super::*;

    fn pca_field(value_type: &DataType, rows: usize, cols: usize, keep: usize) -> Result<FieldRef> {
        Ok(struct_field(
            "pca",
            vec![
                fixed_shape_tensor_field("components", value_type, &[keep, cols], false)?
                    .as_ref()
                    .clone(),
                vector_field("explained_variance", value_type, keep, false)?.as_ref().clone(),
                vector_field("explained_variance_ratio", value_type, keep, false)?.as_ref().clone(),
                vector_field("mean", value_type, cols, false)?.as_ref().clone(),
                fixed_shape_tensor_field("scores", value_type, &[rows, keep], false)?
                    .as_ref()
                    .clone(),
            ],
            false,
        ))
    }

    #[test]
    fn parse_pca_field_validates_shape_and_name_contracts() {
        let field = pca_field(&DataType::Float64, 3, 2, 2).expect("pca field");
        let contract = parse_pca_field(&field, "matrix_pca_transform", 2).expect("pca contract");
        assert_eq!(contract, PcaContract {
            value_type: DataType::Float64,
            rows:       3,
            cols:       2,
            keep:       2,
        });

        let renamed = struct_field(
            "pca",
            vec![
                Field::new("basis", field.data_type().clone(), false),
                vector_field("explained_variance", &DataType::Float64, 2, false)
                    .expect("variance")
                    .as_ref()
                    .clone(),
                vector_field("explained_variance_ratio", &DataType::Float64, 2, false)
                    .expect("ratio")
                    .as_ref()
                    .clone(),
                vector_field("mean", &DataType::Float64, 2, false).expect("mean").as_ref().clone(),
                fixed_shape_tensor_field("scores", &DataType::Float64, &[3, 2], false)
                    .expect("scores")
                    .as_ref()
                    .clone(),
            ],
            false,
        );
        let error = parse_pca_field(&renamed, "matrix_pca_transform", 2)
            .expect_err("renamed PCA field should fail");
        assert!(
            error.to_string().contains("must be named components"),
            "unexpected error: {error}"
        );

        let mismatched = struct_field(
            "pca",
            vec![
                fixed_shape_tensor_field("components", &DataType::Float64, &[2, 2], false)
                    .expect("components")
                    .as_ref()
                    .clone(),
                vector_field("explained_variance", &DataType::Float64, 2, false)
                    .expect("variance")
                    .as_ref()
                    .clone(),
                vector_field("explained_variance_ratio", &DataType::Float64, 2, false)
                    .expect("ratio")
                    .as_ref()
                    .clone(),
                vector_field("mean", &DataType::Float64, 3, false).expect("mean").as_ref().clone(),
                fixed_shape_tensor_field("scores", &DataType::Float64, &[3, 2], false)
                    .expect("scores")
                    .as_ref()
                    .clone(),
            ],
            false,
        );
        let error = parse_pca_field(&mismatched, "matrix_pca_transform", 2)
            .expect_err("mismatched PCA field should fail");
        assert!(error.to_string().contains("feature width mismatch"), "unexpected error: {error}");
    }

    #[test]
    fn pca_exec_helpers_validate_storage_and_metadata() {
        let not_a_struct = scalar_field("pca", &DataType::Float64, false);
        let error = parse_pca_field(&not_a_struct, "matrix_pca_transform", 2)
            .expect_err("non-struct PCA argument should fail");
        assert!(
            error.to_string().contains("must be a PCA struct result"),
            "unexpected error: {error}"
        );

        let struct_array = StructArray::new(
            vec![scalar_field("mean", &DataType::Float64, false)].into(),
            vec![Arc::new(Float64Array::from(vec![1.0]))],
            None,
        );
        let error = expect_pca_fixed_size_list(&struct_array, 0, "matrix_pca_transform", "mean")
            .expect_err("scalar storage should fail");
        assert!(
            error.to_string().contains("expected FixedSizeListArray storage"),
            "unexpected error: {error}"
        );

        let args = ScalarFunctionArgs {
            args:           vec![
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(2.0))),
            ],
            arg_fields:     vec![
                scalar_field("matrix", &DataType::Float64, false),
                scalar_field("pca", &DataType::Float64, false),
            ],
            number_rows:    1,
            return_field:   scalar_field("out", &DataType::Float64, false),
            config_options: Arc::new(ConfigOptions::new()),
        };
        let error = pca_child_field(&args, 0, "matrix_pca_transform")
            .expect_err("non-struct PCA metadata should fail");
        assert!(
            error.to_string().contains("expected PCA struct field metadata"),
            "unexpected error: {error}"
        );
    }
}
