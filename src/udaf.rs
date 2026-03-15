use std::any::Any;
use std::mem::{size_of, size_of_val};
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::array::{
    Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, PrimitiveArray,
    StructArray, UInt64Array,
};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{AggregateUDF, AggregateUDFImpl, Documentation, Signature};
use datafusion::physical_plan::Accumulator;
use nabled::core::prelude::NabledReal;
use ndarray::{Array1, Array2, ArrayView1, ArrayView2, Axis, s};
use ndarrow::NdarrowElement;

use crate::error::{exec_error, plan_error};
use crate::metadata::{
    VectorContract, fixed_shape_tensor_field, parse_vector_field, scalar_field, struct_field,
    variable_shape_tensor_field, vector_field,
};
use crate::signatures::{
    ScalarCoercion, any_signature, coerce_scalar_arguments, named_user_defined_signature,
};
use crate::udf::common::{fixed_size_list_array_from_flat_rows, fixed_size_list_view2};
use crate::udf::docs::ml_doc;

fn state_count_field(name: &str) -> FieldRef { Arc::new(Field::new(name, DataType::UInt64, true)) }

fn flat_matrix_state_field(
    name: &str,
    value_type: &DataType,
    dimension: usize,
) -> Result<FieldRef> {
    let width = dimension
        .checked_mul(dimension)
        .ok_or_else(|| plan_error(name, "matrix state width overflow"))?;
    vector_field(name, value_type, width, true)
}

fn expect_state_array<'a, A: Array + 'static>(
    states: &'a [ArrayRef],
    index: usize,
    function_name: &str,
    expected: &str,
) -> Result<&'a A> {
    let Some(state) = states.get(index) else {
        return Err(exec_error(function_name, format!("missing aggregate state column {index}")));
    };
    state.as_any().downcast_ref::<A>().ok_or_else(|| {
        exec_error(function_name, format!("aggregate state column {index} must be {expected}"))
    })
}

fn to_scalar<T: NabledReal>(value: u64) -> T { T::from_u64(value).unwrap_or_else(T::max_value) }

fn collect_bool_argument(function_name: &str, values: &BooleanArray) -> Result<Option<bool>> {
    if values.is_empty() {
        return Ok(None);
    }
    if values.null_count() > 0 {
        return Err(exec_error(
            function_name,
            "Boolean aggregate arguments must not contain nulls",
        ));
    }
    let expected = values.value(0);
    for index in 1..values.len() {
        if values.value(index) != expected {
            return Err(exec_error(
                function_name,
                "Boolean aggregate argument must be constant within each group",
            ));
        }
    }
    Ok(Some(expected))
}

fn infer_vector_value_type(function_name: &str, arg_types: &[DataType]) -> Result<DataType> {
    let Some(data_type) = arg_types.first() else {
        return Err(plan_error(function_name, "missing vector argument"));
    };
    match data_type {
        DataType::FixedSizeList(item, _) => match item.data_type() {
            DataType::Float32 | DataType::Float64 => Ok(item.data_type().clone()),
            actual => Err(plan_error(
                function_name,
                format!("expected vector argument with Float32 or Float64 items, found {actual}"),
            )),
        },
        actual => Err(plan_error(
            function_name,
            format!("expected fixed-size-list vector argument, found {actual}"),
        )),
    }
}

fn coerce_regression_fit_arguments(
    function_name: &str,
    arg_types: &[DataType],
) -> Result<Vec<DataType>> {
    let value_type = infer_vector_value_type(function_name, arg_types)?;
    let mut coerced =
        coerce_scalar_arguments(function_name, arg_types, &[(3, ScalarCoercion::Boolean)])?;
    let response =
        arg_types.get(1).ok_or_else(|| plan_error(function_name, "missing response argument"))?;
    coerced[1] = match response {
        DataType::Float32
        | DataType::Float64
        | DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Null => value_type,
        actual => {
            return Err(plan_error(
                function_name,
                format!("response argument must be numeric, found {actual}"),
            ));
        }
    };
    Ok(coerced)
}

#[derive(Debug, Clone, PartialEq)]
struct VectorMoments<T> {
    len:     usize,
    count:   u64,
    mean:    Vec<T>,
    scatter: Vec<T>,
}

impl<T: NabledReal> VectorMoments<T> {
    fn new(len: usize) -> Self {
        Self { len, count: 0, mean: vec![T::zero(); len], scatter: vec![T::zero(); len * len] }
    }

    fn is_empty(&self) -> bool { self.count == 0 }

    fn update_batch(&mut self, function_name: &str, rows: &ArrayView2<'_, T>) -> Result<()> {
        if rows.ncols() != self.len {
            return Err(exec_error(
                function_name,
                format!(
                    "vector width mismatch while updating state: expected {}, found {}",
                    self.len,
                    rows.ncols()
                ),
            ));
        }
        if rows.nrows() == 0 {
            return Ok(());
        }

        let batch_count = u64::try_from(rows.nrows())
            .map_err(|_| exec_error(function_name, "row count exceeds u64 limits"))?;
        let batch_mean = rows
            .mean_axis(Axis(0))
            .ok_or_else(|| exec_error(function_name, "failed to compute batch mean"))?;
        let centered = rows.to_owned() - &batch_mean;
        let batch_scatter = centered.t().dot(&centered);

        self.merge_batch_stats(
            function_name,
            batch_count,
            batch_mean.as_slice().ok_or_else(|| {
                exec_error(function_name, "batch mean was not stored contiguously")
            })?,
            batch_scatter.as_slice().ok_or_else(|| {
                exec_error(function_name, "batch scatter matrix was not stored contiguously")
            })?,
        )
    }

    fn merge_batch_stats(
        &mut self,
        function_name: &str,
        batch_count: u64,
        batch_mean: &[T],
        batch_scatter: &[T],
    ) -> Result<()> {
        if batch_count == 0 {
            return Ok(());
        }
        if batch_mean.len() != self.len {
            return Err(exec_error(
                function_name,
                format!(
                    "vector mean width mismatch while merging states: expected {}, found {}",
                    self.len,
                    batch_mean.len()
                ),
            ));
        }
        if batch_scatter.len() != self.len * self.len {
            return Err(exec_error(
                function_name,
                format!(
                    "vector scatter width mismatch while merging states: expected {}, found {}",
                    self.len * self.len,
                    batch_scatter.len()
                ),
            ));
        }
        if self.count == 0 {
            self.count = batch_count;
            self.mean.clone_from_slice(batch_mean);
            self.scatter.clone_from_slice(batch_scatter);
            return Ok(());
        }

        let left_count = self.count;
        let total_count = left_count
            .checked_add(batch_count)
            .ok_or_else(|| exec_error(function_name, "aggregate count overflow"))?;
        let left_scalar = to_scalar::<T>(left_count);
        let batch_scalar = to_scalar::<T>(batch_count);
        let total_scalar = to_scalar::<T>(total_count);
        let correction = (left_scalar * batch_scalar) / total_scalar;

        let mut delta = vec![T::zero(); self.len];
        for (index, mean) in self.mean.iter_mut().enumerate() {
            delta[index] = batch_mean[index] - *mean;
            *mean += delta[index] * (batch_scalar / total_scalar);
        }
        for row in 0..self.len {
            for col in 0..self.len {
                let index = (row * self.len) + col;
                self.scatter[index] +=
                    batch_scatter[index] + (delta[row] * delta[col] * correction);
            }
        }
        self.count = total_count;
        Ok(())
    }

    fn covariance_matrix(&self, function_name: &str) -> Result<Array2<T>> {
        if self.count < 2 {
            return Err(exec_error(function_name, "at least two observations are required"));
        }
        let denominator = to_scalar::<T>(self.count - 1);
        let mut covariance = Array2::<T>::zeros((self.len, self.len));
        for row in 0..self.len {
            for col in 0..self.len {
                covariance[[row, col]] = self.scatter[(row * self.len) + col] / denominator;
            }
        }
        Ok(covariance)
    }
}

#[derive(Debug)]
struct RegressionBatchStats<T> {
    count:         u64,
    sum_x:         Vec<T>,
    gram_x:        Vec<T>,
    cross_xy:      Vec<T>,
    sum_y:         T,
    sum_y2:        T,
    add_intercept: Option<bool>,
}

#[derive(Clone, Copy)]
struct RegressionStateArrays<'a, T: datafusion::arrow::array::types::ArrowPrimitiveType> {
    counts:        &'a UInt64Array,
    sum_x:         &'a FixedSizeListArray,
    xtx:           &'a FixedSizeListArray,
    xty:           &'a FixedSizeListArray,
    sum_y:         &'a PrimitiveArray<T>,
    sum_y2:        &'a PrimitiveArray<T>,
    add_intercept: &'a BooleanArray,
}

#[derive(Debug, Clone, PartialEq)]
enum VectorMomentsState {
    F32(VectorMoments<f32>),
    F64(VectorMoments<f64>),
}

impl VectorMomentsState {
    fn from_contract(contract: &VectorContract) -> Result<Self> {
        match contract.value_type {
            DataType::Float32 => Ok(Self::F32(VectorMoments::new(contract.len))),
            DataType::Float64 => Ok(Self::F64(VectorMoments::new(contract.len))),
            ref actual => Err(exec_error(
                "vector aggregate",
                format!("unsupported vector value type {actual}"),
            )),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::F32(state) => state.is_empty(),
            Self::F64(state) => state.is_empty(),
        }
    }

    fn append_batch(&mut self, function_name: &str, rows: &FixedSizeListArray) -> Result<()> {
        match self {
            Self::F32(state) => {
                let view = fixed_size_list_view2::<Float32Type>(rows, function_name)?;
                state.update_batch(function_name, &view)?;
            }
            Self::F64(state) => {
                let view = fixed_size_list_view2::<Float64Type>(rows, function_name)?;
                state.update_batch(function_name, &view)?;
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, function_name: &str, states: &[ArrayRef]) -> Result<()> {
        let counts = expect_state_array::<UInt64Array>(states, 0, function_name, "UInt64Array")?;
        let means =
            expect_state_array::<FixedSizeListArray>(states, 1, function_name, "FixedSizeList")?;
        let scatters =
            expect_state_array::<FixedSizeListArray>(states, 2, function_name, "FixedSizeList")?;

        match self {
            Self::F32(state) => {
                let mean_view = fixed_size_list_view2::<Float32Type>(means, function_name)?;
                let scatter_view = fixed_size_list_view2::<Float32Type>(scatters, function_name)?;
                for index in 0..counts.len() {
                    let count = counts.value(index);
                    if count == 0 {
                        continue;
                    }
                    state.merge_batch_stats(
                        function_name,
                        count,
                        mean_view.row(index).as_slice().ok_or_else(|| {
                            exec_error(function_name, "state mean row was not contiguous")
                        })?,
                        scatter_view.row(index).as_slice().ok_or_else(|| {
                            exec_error(function_name, "state scatter row was not contiguous")
                        })?,
                    )?;
                }
            }
            Self::F64(state) => {
                let mean_view = fixed_size_list_view2::<Float64Type>(means, function_name)?;
                let scatter_view = fixed_size_list_view2::<Float64Type>(scatters, function_name)?;
                for index in 0..counts.len() {
                    let count = counts.value(index);
                    if count == 0 {
                        continue;
                    }
                    state.merge_batch_stats(
                        function_name,
                        count,
                        mean_view.row(index).as_slice().ok_or_else(|| {
                            exec_error(function_name, "state mean row was not contiguous")
                        })?,
                        scatter_view.row(index).as_slice().ok_or_else(|| {
                            exec_error(function_name, "state scatter row was not contiguous")
                        })?,
                    )?;
                }
            }
        }
        Ok(())
    }

    fn state_values(&self, function_name: &str) -> Result<Vec<ScalarValue>> {
        match self {
            Self::F32(state) => Ok(vec![
                ScalarValue::UInt64(Some(state.count)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float32Type,
                >(
                    function_name, 1, state.len, &state.mean
                )?)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float32Type,
                >(
                    function_name,
                    1,
                    state.len * state.len,
                    &state.scatter,
                )?)),
            ]),
            Self::F64(state) => Ok(vec![
                ScalarValue::UInt64(Some(state.count)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float64Type,
                >(
                    function_name, 1, state.len, &state.mean
                )?)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float64Type,
                >(
                    function_name,
                    1,
                    state.len * state.len,
                    &state.scatter,
                )?)),
            ]),
        }
    }

    fn evaluate_covariance(&self, function_name: &str) -> Result<ScalarValue> {
        match self {
            Self::F32(state) => {
                let covariance = state.covariance_matrix(function_name)?;
                let covariance = covariance.insert_axis(Axis(0)).into_dyn();
                let (_field, array) =
                    ndarrow::arrayd_to_fixed_shape_tensor(function_name, covariance)
                        .map_err(|error| exec_error(function_name, error))?;
                Ok(ScalarValue::FixedSizeList(Arc::new(array)))
            }
            Self::F64(state) => {
                let covariance = state.covariance_matrix(function_name)?;
                let covariance = covariance.insert_axis(Axis(0)).into_dyn();
                let (_field, array) =
                    ndarrow::arrayd_to_fixed_shape_tensor(function_name, covariance)
                        .map_err(|error| exec_error(function_name, error))?;
                Ok(ScalarValue::FixedSizeList(Arc::new(array)))
            }
        }
    }

    fn evaluate_correlation(&self, function_name: &str) -> Result<ScalarValue> {
        match self {
            Self::F32(state) => {
                let covariance = state.covariance_matrix(function_name)?;
                let dimension = covariance.nrows();
                let mut correlation = Array2::<f32>::zeros((dimension, dimension));
                for row in 0..dimension {
                    let sigma_row = covariance[[row, row]].sqrt();
                    for col in 0..dimension {
                        let sigma_col = covariance[[col, col]].sqrt();
                        let denominator = (sigma_row * sigma_col).max(f32::EPSILON);
                        correlation[[row, col]] = covariance[[row, col]] / denominator;
                    }
                }
                let correlation = correlation.insert_axis(Axis(0)).into_dyn();
                let (_field, array) =
                    ndarrow::arrayd_to_fixed_shape_tensor(function_name, correlation)
                        .map_err(|error| exec_error(function_name, error))?;
                Ok(ScalarValue::FixedSizeList(Arc::new(array)))
            }
            Self::F64(state) => {
                let covariance = state.covariance_matrix(function_name)?;
                let dimension = covariance.nrows();
                let mut correlation = Array2::<f64>::zeros((dimension, dimension));
                for row in 0..dimension {
                    let sigma_row = covariance[[row, row]].sqrt();
                    for col in 0..dimension {
                        let sigma_col = covariance[[col, col]].sqrt();
                        let denominator = (sigma_row * sigma_col).max(f64::EPSILON);
                        correlation[[row, col]] = covariance[[row, col]] / denominator;
                    }
                }
                let correlation = correlation.insert_axis(Axis(0)).into_dyn();
                let (_field, array) =
                    ndarrow::arrayd_to_fixed_shape_tensor(function_name, correlation)
                        .map_err(|error| exec_error(function_name, error))?;
                Ok(ScalarValue::FixedSizeList(Arc::new(array)))
            }
        }
    }

    fn evaluate_pca(&self, function_name: &str) -> Result<ScalarValue> {
        match self {
            Self::F32(state) => {
                let covariance = state.covariance_matrix(function_name)?;
                let eigen = nabled::linalg::eigen::symmetric_view(&covariance.view())
                    .map_err(|error| exec_error(function_name, error))?;
                let keep = usize::try_from(state.count).unwrap_or(usize::MAX).min(state.len).max(1);
                let explained = eigen.eigenvalues.slice(s![..keep]).to_owned();
                let components = eigen.eigenvectors.t().slice(s![..keep, ..]).to_owned();
                let total_variance = explained
                    .iter()
                    .copied()
                    .fold(0.0_f32, |acc, value| acc + value)
                    .max(f32::EPSILON);
                let explained_variance_ratio = explained.map(|value| *value / total_variance);
                let pca = nabled::ml::pca::NdarrayPCAResult {
                    components,
                    explained_variance: explained,
                    explained_variance_ratio,
                    mean: Array1::from_vec(state.mean.clone()),
                    scores: Array2::zeros((0, keep)),
                };
                build_pca_scalar::<Float32Type>(function_name, state.len, pca)
            }
            Self::F64(state) => {
                let covariance = state.covariance_matrix(function_name)?;
                let eigen = nabled::linalg::eigen::symmetric_view(&covariance.view())
                    .map_err(|error| exec_error(function_name, error))?;
                let keep = usize::try_from(state.count).unwrap_or(usize::MAX).min(state.len).max(1);
                let explained = eigen.eigenvalues.slice(s![..keep]).to_owned();
                let components = eigen.eigenvectors.t().slice(s![..keep, ..]).to_owned();
                let total_variance = explained
                    .iter()
                    .copied()
                    .fold(0.0_f64, |acc, value| acc + value)
                    .max(f64::EPSILON);
                let explained_variance_ratio = explained.map(|value| *value / total_variance);
                let pca = nabled::ml::pca::NdarrayPCAResult {
                    components,
                    explained_variance: explained,
                    explained_variance_ratio,
                    mean: Array1::from_vec(state.mean.clone()),
                    scores: Array2::zeros((0, keep)),
                };
                build_pca_scalar::<Float64Type>(function_name, state.len, pca)
            }
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::F32(state) => {
                size_of_val(self)
                    + (state.mean.capacity() + state.scatter.capacity()) * size_of::<f32>()
            }
            Self::F64(state) => {
                size_of_val(self)
                    + (state.mean.capacity() + state.scatter.capacity()) * size_of::<f64>()
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
struct RegressionMoments<T> {
    cols:          usize,
    count:         u64,
    sum_x:         Vec<T>,
    xtx:           Vec<T>,
    xty:           Vec<T>,
    sum_y:         T,
    sum_y2:        T,
    add_intercept: Option<bool>,
}

impl<T: NabledReal> RegressionMoments<T> {
    fn new(cols: usize) -> Self {
        Self {
            cols,
            count: 0,
            sum_x: vec![T::zero(); cols],
            xtx: vec![T::zero(); cols * cols],
            xty: vec![T::zero(); cols],
            sum_y: T::zero(),
            sum_y2: T::zero(),
            add_intercept: None,
        }
    }

    fn is_empty(&self) -> bool { self.count == 0 }

    fn merge_batch_stats(
        &mut self,
        function_name: &str,
        batch: &RegressionBatchStats<T>,
    ) -> Result<()> {
        if batch.count == 0 {
            return Ok(());
        }
        if batch.sum_x.len() != self.cols {
            return Err(exec_error(
                function_name,
                format!(
                    "design width mismatch while merging states: expected {}, found {}",
                    self.cols,
                    batch.sum_x.len()
                ),
            ));
        }
        if batch.gram_x.len() != self.cols * self.cols {
            return Err(exec_error(
                function_name,
                format!(
                    "design scatter width mismatch while merging states: expected {}, found {}",
                    self.cols * self.cols,
                    batch.gram_x.len()
                ),
            ));
        }
        if batch.cross_xy.len() != self.cols {
            return Err(exec_error(
                function_name,
                format!(
                    "design/response width mismatch while merging states: expected {}, found {}",
                    self.cols,
                    batch.cross_xy.len()
                ),
            ));
        }
        match (self.add_intercept, batch.add_intercept) {
            (Some(existing), Some(incoming)) if existing != incoming => {
                return Err(exec_error(
                    function_name,
                    "add_intercept must be constant within each group",
                ));
            }
            (None, Some(incoming)) => self.add_intercept = Some(incoming),
            _ => {}
        }

        self.count = self
            .count
            .checked_add(batch.count)
            .ok_or_else(|| exec_error(function_name, "aggregate count overflow"))?;
        for (left, right) in self.sum_x.iter_mut().zip(batch.sum_x.iter().copied()) {
            *left += right;
        }
        for (left, right) in self.xtx.iter_mut().zip(batch.gram_x.iter().copied()) {
            *left += right;
        }
        for (left, right) in self.xty.iter_mut().zip(batch.cross_xy.iter().copied()) {
            *left += right;
        }
        self.sum_y += batch.sum_y;
        self.sum_y2 += batch.sum_y2;
        Ok(())
    }

    fn append_batch(
        &mut self,
        function_name: &str,
        design: &ArrayView2<'_, T>,
        response: &ArrayView1<'_, T>,
        add_intercept: &BooleanArray,
    ) -> Result<()> {
        let add_intercept = collect_bool_argument(function_name, add_intercept)?;
        if design.ncols() != self.cols || design.nrows() != response.len() {
            return Err(exec_error(function_name, "design/response batch shape mismatch"));
        }
        if design.nrows() == 0 {
            return Ok(());
        }

        let batch_count = u64::try_from(design.nrows())
            .map_err(|_| exec_error(function_name, "row count exceeds u64 limits"))?;
        let batch_sum_x = design.sum_axis(Axis(0));
        let batch_gram_x = design.t().dot(design);
        let batch_cross_xy = design.t().dot(response);
        let batch_stats = RegressionBatchStats {
            count: batch_count,
            sum_x: batch_sum_x
                .as_slice()
                .ok_or_else(|| exec_error(function_name, "design sum was not stored contiguously"))?
                .to_vec(),
            gram_x: batch_gram_x
                .as_slice()
                .ok_or_else(|| {
                    exec_error(function_name, "design scatter matrix was not stored contiguously")
                })?
                .to_vec(),
            cross_xy: batch_cross_xy
                .as_slice()
                .ok_or_else(|| {
                    exec_error(function_name, "design/response product was not contiguous")
                })?
                .to_vec(),
            sum_y: response.iter().copied().fold(T::zero(), |acc, value| acc + value),
            sum_y2: response.dot(response),
            add_intercept,
        };

        self.merge_batch_stats(function_name, &batch_stats)
    }
}

#[derive(Debug, Clone, PartialEq)]
enum RegressionMomentsState {
    F32(RegressionMoments<f32>),
    F64(RegressionMoments<f64>),
}

impl RegressionMomentsState {
    fn from_contract(contract: &VectorContract) -> Result<Self> {
        match contract.value_type {
            DataType::Float32 => Ok(Self::F32(RegressionMoments::new(contract.len))),
            DataType::Float64 => Ok(Self::F64(RegressionMoments::new(contract.len))),
            ref actual => Err(exec_error(
                "linear_regression_fit",
                format!("unsupported design value type {actual}"),
            )),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::F32(state) => state.is_empty(),
            Self::F64(state) => state.is_empty(),
        }
    }

    fn append_batch(
        &mut self,
        function_name: &str,
        design: &FixedSizeListArray,
        response: &ArrayRef,
        add_intercept: &BooleanArray,
    ) -> Result<()> {
        match self {
            Self::F32(state) => {
                let design_view = fixed_size_list_view2::<Float32Type>(design, function_name)?;
                let response = response
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| exec_error(function_name, "response column must be Float32"))?;
                if response.null_count() > 0 {
                    return Err(exec_error(
                        function_name,
                        "response column must not contain nulls",
                    ));
                }
                let response = ArrayView1::from(response.values().as_ref());
                state.append_batch(function_name, &design_view, &response, add_intercept)?;
            }
            Self::F64(state) => {
                let design_view = fixed_size_list_view2::<Float64Type>(design, function_name)?;
                let response = response
                    .as_any()
                    .downcast_ref::<Float64Array>()
                    .ok_or_else(|| exec_error(function_name, "response column must be Float64"))?;
                if response.null_count() > 0 {
                    return Err(exec_error(
                        function_name,
                        "response column must not contain nulls",
                    ));
                }
                let response = ArrayView1::from(response.values().as_ref());
                state.append_batch(function_name, &design_view, &response, add_intercept)?;
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, function_name: &str, states: &[ArrayRef]) -> Result<()> {
        let counts = expect_state_array::<UInt64Array>(states, 0, function_name, "UInt64Array")?;
        let sum_x =
            expect_state_array::<FixedSizeListArray>(states, 1, function_name, "FixedSizeList")?;
        let xtx =
            expect_state_array::<FixedSizeListArray>(states, 2, function_name, "FixedSizeList")?;
        let xty =
            expect_state_array::<FixedSizeListArray>(states, 3, function_name, "FixedSizeList")?;
        let add_intercept =
            expect_state_array::<BooleanArray>(states, 6, function_name, "BooleanArray")?;

        match self {
            Self::F32(state) => {
                let sum_y =
                    expect_state_array::<Float32Array>(states, 4, function_name, "Float32Array")?;
                let sum_y2 =
                    expect_state_array::<Float32Array>(states, 5, function_name, "Float32Array")?;
                merge_regression_state_batch::<Float32Type>(
                    state,
                    function_name,
                    &RegressionStateArrays {
                        counts,
                        sum_x,
                        xtx,
                        xty,
                        sum_y,
                        sum_y2,
                        add_intercept,
                    },
                )?;
            }
            Self::F64(state) => {
                let sum_y =
                    expect_state_array::<Float64Array>(states, 4, function_name, "Float64Array")?;
                let sum_y2 =
                    expect_state_array::<Float64Array>(states, 5, function_name, "Float64Array")?;
                merge_regression_state_batch::<Float64Type>(
                    state,
                    function_name,
                    &RegressionStateArrays {
                        counts,
                        sum_x,
                        xtx,
                        xty,
                        sum_y,
                        sum_y2,
                        add_intercept,
                    },
                )?;
            }
        }
        Ok(())
    }

    fn state_values(&self, function_name: &str) -> Result<Vec<ScalarValue>> {
        match self {
            Self::F32(state) => Ok(vec![
                ScalarValue::UInt64(Some(state.count)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float32Type,
                >(
                    function_name, 1, state.cols, &state.sum_x
                )?)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float32Type,
                >(
                    function_name,
                    1,
                    state.cols * state.cols,
                    &state.xtx,
                )?)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float32Type,
                >(
                    function_name, 1, state.cols, &state.xty
                )?)),
                ScalarValue::Float32(Some(state.sum_y)),
                ScalarValue::Float32(Some(state.sum_y2)),
                ScalarValue::Boolean(state.add_intercept),
            ]),
            Self::F64(state) => Ok(vec![
                ScalarValue::UInt64(Some(state.count)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float64Type,
                >(
                    function_name, 1, state.cols, &state.sum_x
                )?)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float64Type,
                >(
                    function_name,
                    1,
                    state.cols * state.cols,
                    &state.xtx,
                )?)),
                ScalarValue::FixedSizeList(Arc::new(fixed_size_list_array_from_flat_rows::<
                    Float64Type,
                >(
                    function_name, 1, state.cols, &state.xty
                )?)),
                ScalarValue::Float64(Some(state.sum_y)),
                ScalarValue::Float64(Some(state.sum_y2)),
                ScalarValue::Boolean(state.add_intercept),
            ]),
        }
    }

    fn evaluate(&self, function_name: &str) -> Result<ScalarValue> {
        match self {
            Self::F32(state) => {
                let add_intercept = state.add_intercept.unwrap_or(true);
                let coefficient_count = state.cols + usize::from(add_intercept);
                let mut normal_matrix =
                    Array2::<f32>::zeros((coefficient_count, coefficient_count));
                let mut normal_rhs = Array1::<f32>::zeros(coefficient_count);

                if add_intercept {
                    normal_matrix[[0, 0]] = to_scalar::<f32>(state.count);
                    normal_rhs[0] = state.sum_y;
                    for col in 0..state.cols {
                        normal_matrix[[0, col + 1]] = state.sum_x[col];
                        normal_matrix[[col + 1, 0]] = state.sum_x[col];
                        normal_rhs[col + 1] = state.xty[col];
                    }
                    for row in 0..state.cols {
                        for col in 0..state.cols {
                            normal_matrix[[row + 1, col + 1]] = state.xtx[(row * state.cols) + col];
                        }
                    }
                } else {
                    for row in 0..state.cols {
                        normal_rhs[row] = state.xty[row];
                        for col in 0..state.cols {
                            normal_matrix[[row, col]] = state.xtx[(row * state.cols) + col];
                        }
                    }
                }

                let coefficients =
                    nabled::linalg::lu::solve_view(&normal_matrix.view(), &normal_rhs.view())
                        .map_err(|error| exec_error(function_name, error))?;
                let normal_beta = normal_matrix.dot(&coefficients);
                let rss = state.sum_y2 - (coefficients.dot(&normal_rhs) * 2.0)
                    + coefficients.dot(&normal_beta);
                let count_scalar = to_scalar::<f32>(state.count);
                let ss_total = state.sum_y2 - ((state.sum_y * state.sum_y) / count_scalar);
                let r_squared = if ss_total <= f32::EPSILON { 1.0 } else { 1.0 - (rss / ss_total) };
                let result = nabled::ml::regression::NdarrayRegressionResult {
                    coefficients,
                    fitted_values: Array1::zeros(0),
                    residuals: Array1::zeros(0),
                    r_squared,
                };
                build_regression_fit_scalar_f32(function_name, result)
            }
            Self::F64(state) => {
                let add_intercept = state.add_intercept.unwrap_or(true);
                let coefficient_count = state.cols + usize::from(add_intercept);
                let mut normal_matrix =
                    Array2::<f64>::zeros((coefficient_count, coefficient_count));
                let mut normal_rhs = Array1::<f64>::zeros(coefficient_count);

                if add_intercept {
                    normal_matrix[[0, 0]] = to_scalar::<f64>(state.count);
                    normal_rhs[0] = state.sum_y;
                    for col in 0..state.cols {
                        normal_matrix[[0, col + 1]] = state.sum_x[col];
                        normal_matrix[[col + 1, 0]] = state.sum_x[col];
                        normal_rhs[col + 1] = state.xty[col];
                    }
                    for row in 0..state.cols {
                        for col in 0..state.cols {
                            normal_matrix[[row + 1, col + 1]] = state.xtx[(row * state.cols) + col];
                        }
                    }
                } else {
                    for row in 0..state.cols {
                        normal_rhs[row] = state.xty[row];
                        for col in 0..state.cols {
                            normal_matrix[[row, col]] = state.xtx[(row * state.cols) + col];
                        }
                    }
                }

                let coefficients =
                    nabled::linalg::lu::solve_view(&normal_matrix.view(), &normal_rhs.view())
                        .map_err(|error| exec_error(function_name, error))?;
                let normal_beta = normal_matrix.dot(&coefficients);
                let rss = state.sum_y2 - (coefficients.dot(&normal_rhs) * 2.0)
                    + coefficients.dot(&normal_beta);
                let count_scalar = to_scalar::<f64>(state.count);
                let ss_total = state.sum_y2 - ((state.sum_y * state.sum_y) / count_scalar);
                let r_squared = if ss_total <= f64::EPSILON { 1.0 } else { 1.0 - (rss / ss_total) };
                let result = nabled::ml::regression::NdarrayRegressionResult {
                    coefficients,
                    fitted_values: Array1::zeros(0),
                    residuals: Array1::zeros(0),
                    r_squared,
                };
                build_regression_fit_scalar_f64(function_name, result)
            }
        }
    }

    fn size(&self) -> usize {
        match self {
            Self::F32(state) => {
                size_of_val(self)
                    + (state.sum_x.capacity() + state.xtx.capacity() + state.xty.capacity())
                        * size_of::<f32>()
            }
            Self::F64(state) => {
                size_of_val(self)
                    + (state.sum_x.capacity() + state.xtx.capacity() + state.xty.capacity())
                        * size_of::<f64>()
            }
        }
    }
}

fn merge_regression_state_batch<T>(
    state: &mut RegressionMoments<T::Native>,
    function_name: &str,
    arrays: &RegressionStateArrays<'_, T>,
) -> Result<()>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let sum_x = fixed_size_list_view2::<T>(arrays.sum_x, function_name)?;
    let xtx = fixed_size_list_view2::<T>(arrays.xtx, function_name)?;
    let xty = fixed_size_list_view2::<T>(arrays.xty, function_name)?;
    for index in 0..arrays.counts.len() {
        let count = arrays.counts.value(index);
        if count == 0 {
            continue;
        }
        let batch_stats = RegressionBatchStats {
            count,
            sum_x: sum_x
                .row(index)
                .as_slice()
                .ok_or_else(|| exec_error(function_name, "state sum_x row was not contiguous"))?
                .to_vec(),
            gram_x: xtx
                .row(index)
                .as_slice()
                .ok_or_else(|| exec_error(function_name, "state xtx row was not contiguous"))?
                .to_vec(),
            cross_xy: xty
                .row(index)
                .as_slice()
                .ok_or_else(|| exec_error(function_name, "state xty row was not contiguous"))?
                .to_vec(),
            sum_y: arrays.sum_y.value(index),
            sum_y2: arrays.sum_y2.value(index),
            add_intercept: if arrays.add_intercept.is_null(index) {
                None
            } else {
                Some(arrays.add_intercept.value(index))
            },
        };
        state.merge_batch_stats(function_name, &batch_stats)?;
    }
    Ok(())
}

fn build_pca_scalar<T>(
    function_name: &str,
    feature_count: usize,
    pca: nabled::ml::pca::NdarrayPCAResult<T::Native>,
) -> Result<ScalarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: Copy + NabledReal + NdarrowElement,
{
    let feature_count_i32 = i32::try_from(feature_count)
        .map_err(|_| exec_error(function_name, "feature count exceeds Arrow i32 limits"))?;
    let (components_field, components) = ndarrow::arrays_to_variable_shape_tensor(
        "components",
        vec![pca.components.into_dyn()],
        Some(vec![None, Some(feature_count_i32)]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    let (explained_variance_field, explained_variance) = ndarrow::arrays_to_variable_shape_tensor(
        "explained_variance",
        vec![pca.explained_variance.into_dyn()],
        Some(vec![None]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    let (explained_variance_ratio_field, explained_variance_ratio) =
        ndarrow::arrays_to_variable_shape_tensor(
            "explained_variance_ratio",
            vec![pca.explained_variance_ratio.into_dyn()],
            Some(vec![None]),
        )
        .map_err(|error| exec_error(function_name, error))?;
    let mean_values = pca.mean.iter().copied().collect::<Vec<_>>();
    let mean =
        fixed_size_list_array_from_flat_rows::<T>(function_name, 1, feature_count, &mean_values)?;
    let struct_array = StructArray::new(
        vec![
            Arc::new(components_field),
            Arc::new(explained_variance_field),
            Arc::new(explained_variance_ratio_field),
            vector_field("mean", &T::DATA_TYPE, feature_count, false)?,
        ]
        .into(),
        vec![
            Arc::new(components) as ArrayRef,
            Arc::new(explained_variance) as ArrayRef,
            Arc::new(explained_variance_ratio) as ArrayRef,
            Arc::new(mean) as ArrayRef,
        ],
        None,
    );
    Ok(ScalarValue::Struct(Arc::new(struct_array)))
}

fn build_regression_fit_scalar_f32(
    function_name: &str,
    result: nabled::ml::regression::NdarrayRegressionResult<f32>,
) -> Result<ScalarValue> {
    let (coefficients_field, coefficients) = ndarrow::arrays_to_variable_shape_tensor(
        "coefficients",
        vec![result.coefficients.into_dyn()],
        Some(vec![None]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    let struct_array = StructArray::new(
        vec![Arc::new(coefficients_field), scalar_field("r_squared", &DataType::Float32, false)]
            .into(),
        vec![
            Arc::new(coefficients) as ArrayRef,
            ScalarValue::Float32(Some(result.r_squared)).to_array_of_size(1)?,
        ],
        None,
    );
    Ok(ScalarValue::Struct(Arc::new(struct_array)))
}

fn build_regression_fit_scalar_f64(
    function_name: &str,
    result: nabled::ml::regression::NdarrayRegressionResult<f64>,
) -> Result<ScalarValue> {
    let (coefficients_field, coefficients) = ndarrow::arrays_to_variable_shape_tensor(
        "coefficients",
        vec![result.coefficients.into_dyn()],
        Some(vec![None]),
    )
    .map_err(|error| exec_error(function_name, error))?;
    let struct_array = StructArray::new(
        vec![Arc::new(coefficients_field), scalar_field("r_squared", &DataType::Float64, false)]
            .into(),
        vec![
            Arc::new(coefficients) as ArrayRef,
            ScalarValue::Float64(Some(result.r_squared)).to_array_of_size(1)?,
        ],
        None,
    );
    Ok(ScalarValue::Struct(Arc::new(struct_array)))
}

#[derive(Debug)]
struct VectorMomentsAccumulator {
    function_name: &'static str,
    return_field:  FieldRef,
    state:         VectorMomentsState,
}

impl Accumulator for VectorMomentsAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> { self.state.state_values(self.function_name) }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let Some(value) = values.first() else {
            return Err(exec_error(self.function_name, "missing vector argument"));
        };
        let rows = value.as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
            exec_error(self.function_name, "vector argument must be FixedSizeList")
        })?;
        self.state.append_batch(self.function_name, rows)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.state.merge_batch(self.function_name, states)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.state.is_empty() {
            return ScalarValue::try_from(self.return_field.data_type());
        }
        match self.function_name {
            "vector_covariance_agg" => self.state.evaluate_covariance(self.function_name),
            "vector_correlation_agg" => self.state.evaluate_correlation(self.function_name),
            "vector_pca_fit" => self.state.evaluate_pca(self.function_name),
            _ => Err(exec_error(self.function_name, "unsupported vector aggregate")),
        }
    }

    fn size(&self) -> usize { size_of_val(self) + self.state.size() }
}

#[derive(Debug)]
struct RegressionMomentsAccumulator {
    return_field: FieldRef,
    state:        RegressionMomentsState,
}

impl Accumulator for RegressionMomentsAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        self.state.state_values("linear_regression_fit")
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        if values.len() != 3 {
            return Err(exec_error(
                "linear_regression_fit",
                "expected design, response, and add_intercept arguments",
            ));
        }
        let design = values[0].as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
            exec_error("linear_regression_fit", "design argument must be FixedSizeList")
        })?;
        let add_intercept = values[2].as_any().downcast_ref::<BooleanArray>().ok_or_else(|| {
            exec_error("linear_regression_fit", "add_intercept argument must be Boolean")
        })?;
        self.state.append_batch("linear_regression_fit", design, &values[1], add_intercept)
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.state.merge_batch("linear_regression_fit", states)
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.state.is_empty() {
            return ScalarValue::try_from(self.return_field.data_type());
        }
        self.state.evaluate("linear_regression_fit")
    }

    fn size(&self) -> usize { size_of_val(self) + self.state.size() }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorCovarianceAgg {
    signature: Signature,
}

impl VectorCovarianceAgg {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl AggregateUDFImpl for VectorCovarianceAgg {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_covariance_agg" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field should be used instead")
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let contract = parse_vector_field(&arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &[contract.len, contract.len],
            true,
        )
    }

    fn accumulator(&self, acc_args: AccumulatorArgs<'_>) -> Result<Box<dyn Accumulator>> {
        let contract = parse_vector_field(&acc_args.expr_fields[0], self.name(), 1)?;
        Ok(Box::new(VectorMomentsAccumulator {
            function_name: "vector_covariance_agg",
            return_field:  Arc::clone(&acc_args.return_field),
            state:         VectorMomentsState::from_contract(&contract)?,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs<'_>) -> Result<Vec<FieldRef>> {
        let contract = parse_vector_field(&args.input_fields[0], self.name(), 1)?;
        Ok(vec![
            state_count_field(&format!("{}_count", args.name)),
            vector_field(&format!("{}_mean", args.name), &contract.value_type, contract.len, true)?,
            flat_matrix_state_field(
                &format!("{}_scatter", args.name),
                &contract.value_type,
                contract.len,
            )?,
        ])
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Aggregate rows of dense vectors into a sample covariance matrix.",
                "vector_covariance_agg(vector_rows)",
            )
            .with_argument(
                "vector_rows",
                "Dense vector observations in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorCorrelationAgg {
    signature: Signature,
}

impl VectorCorrelationAgg {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl AggregateUDFImpl for VectorCorrelationAgg {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_correlation_agg" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field should be used instead")
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let contract = parse_vector_field(&arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &[contract.len, contract.len],
            true,
        )
    }

    fn accumulator(&self, acc_args: AccumulatorArgs<'_>) -> Result<Box<dyn Accumulator>> {
        let contract = parse_vector_field(&acc_args.expr_fields[0], self.name(), 1)?;
        Ok(Box::new(VectorMomentsAccumulator {
            function_name: "vector_correlation_agg",
            return_field:  Arc::clone(&acc_args.return_field),
            state:         VectorMomentsState::from_contract(&contract)?,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs<'_>) -> Result<Vec<FieldRef>> {
        let contract = parse_vector_field(&args.input_fields[0], self.name(), 1)?;
        Ok(vec![
            state_count_field(&format!("{}_count", args.name)),
            vector_field(&format!("{}_mean", args.name), &contract.value_type, contract.len, true)?,
            flat_matrix_state_field(
                &format!("{}_scatter", args.name),
                &contract.value_type,
                contract.len,
            )?,
        ])
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Aggregate rows of dense vectors into a correlation matrix.",
                "vector_correlation_agg(vector_rows)",
            )
            .with_argument(
                "vector_rows",
                "Dense vector observations in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct VectorPcaFit {
    signature: Signature,
}

impl VectorPcaFit {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl AggregateUDFImpl for VectorPcaFit {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "vector_pca_fit" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field should be used instead")
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let contract = parse_vector_field(&arg_fields[0], self.name(), 1)?;
        let feature_count_i32 = i32::try_from(contract.len)
            .map_err(|_| exec_error(self.name(), "feature count exceeds Arrow i32 limits"))?;
        Ok(struct_field(
            self.name(),
            vec![
                variable_shape_tensor_field(
                    "components",
                    &contract.value_type,
                    2,
                    Some(&[None, Some(feature_count_i32)]),
                    false,
                )?
                .as_ref()
                .clone(),
                variable_shape_tensor_field(
                    "explained_variance",
                    &contract.value_type,
                    1,
                    Some(&[None]),
                    false,
                )?
                .as_ref()
                .clone(),
                variable_shape_tensor_field(
                    "explained_variance_ratio",
                    &contract.value_type,
                    1,
                    Some(&[None]),
                    false,
                )?
                .as_ref()
                .clone(),
                vector_field("mean", &contract.value_type, contract.len, false)?.as_ref().clone(),
            ],
            true,
        ))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs<'_>) -> Result<Box<dyn Accumulator>> {
        let contract = parse_vector_field(&acc_args.expr_fields[0], self.name(), 1)?;
        Ok(Box::new(VectorMomentsAccumulator {
            function_name: "vector_pca_fit",
            return_field:  Arc::clone(&acc_args.return_field),
            state:         VectorMomentsState::from_contract(&contract)?,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs<'_>) -> Result<Vec<FieldRef>> {
        let contract = parse_vector_field(&args.input_fields[0], self.name(), 1)?;
        Ok(vec![
            state_count_field(&format!("{}_count", args.name)),
            vector_field(&format!("{}_mean", args.name), &contract.value_type, contract.len, true)?,
            flat_matrix_state_field(
                &format!("{}_scatter", args.name),
                &contract.value_type,
                contract.len,
            )?,
        ])
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Fit PCA over grouped dense vector observations and return the fitted components \
                 and summary fields.",
                "vector_pca_fit(vector_rows)",
            )
            .with_argument(
                "vector_rows",
                "Dense vector observations in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct LinearRegressionFit {
    signature: Signature,
}

impl LinearRegressionFit {
    fn new() -> Self {
        Self { signature: named_user_defined_signature(&["design", "response", "add_intercept"]) }
    }
}

impl AggregateUDFImpl for LinearRegressionFit {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "linear_regression_fit" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_regression_fit_arguments(self.name(), arg_types)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field should be used instead")
    }

    fn return_field(&self, arg_fields: &[FieldRef]) -> Result<FieldRef> {
        let design = parse_vector_field(&arg_fields[0], self.name(), 1)?;
        let response = &arg_fields[1];
        if !response.data_type().equals_datatype(&design.value_type) {
            return Err(exec_error(
                self.name(),
                format!(
                    "response value type mismatch: expected {}, found {}",
                    design.value_type,
                    response.data_type()
                ),
            ));
        }
        Ok(struct_field(
            self.name(),
            vec![
                variable_shape_tensor_field(
                    "coefficients",
                    &design.value_type,
                    1,
                    Some(&[None]),
                    false,
                )?
                .as_ref()
                .clone(),
                scalar_field("r_squared", &design.value_type, false).as_ref().clone(),
            ],
            true,
        ))
    }

    fn accumulator(&self, acc_args: AccumulatorArgs<'_>) -> Result<Box<dyn Accumulator>> {
        let design = parse_vector_field(&acc_args.expr_fields[0], self.name(), 1)?;
        Ok(Box::new(RegressionMomentsAccumulator {
            return_field: Arc::clone(&acc_args.return_field),
            state:        RegressionMomentsState::from_contract(&design)?,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs<'_>) -> Result<Vec<FieldRef>> {
        let design = parse_vector_field(&args.input_fields[0], self.name(), 1)?;
        Ok(vec![
            state_count_field(&format!("{}_count", args.name)),
            vector_field(&format!("{}_sum_x", args.name), &design.value_type, design.len, true)?,
            flat_matrix_state_field(&format!("{}_xtx", args.name), &design.value_type, design.len)?,
            vector_field(&format!("{}_xty", args.name), &design.value_type, design.len, true)?,
            scalar_field(&format!("{}_sum_y", args.name), &design.value_type, true),
            scalar_field(&format!("{}_sum_y2", args.name), &design.value_type, true),
            scalar_field(&format!("{}_add_intercept", args.name), &DataType::Boolean, true),
        ])
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Fit linear regression over grouped design rows and scalar responses.",
                "linear_regression_fit(design_rows, response_values, add_intercept => true)",
            )
            .with_argument(
                "design_rows",
                "Dense design observations in canonical FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_argument("response_values", "Scalar Float32 or Float64 response column.")
            .with_argument(
                "add_intercept",
                "Boolean flag controlling whether an intercept column is added to the design.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[must_use]
pub fn vector_covariance_agg_udaf() -> Arc<AggregateUDF> {
    Arc::new(AggregateUDF::new_from_impl(VectorCovarianceAgg::new()))
}

#[must_use]
pub fn vector_correlation_agg_udaf() -> Arc<AggregateUDF> {
    Arc::new(AggregateUDF::new_from_impl(VectorCorrelationAgg::new()))
}

#[must_use]
pub fn vector_pca_fit_udaf() -> Arc<AggregateUDF> {
    Arc::new(AggregateUDF::new_from_impl(VectorPcaFit::new()))
}

#[must_use]
pub fn linear_regression_fit_udaf() -> Arc<AggregateUDF> {
    Arc::new(AggregateUDF::new_from_impl(LinearRegressionFit::new()))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use datafusion::arrow::array::types::{Float32Type, Float64Type};
    use datafusion::arrow::array::{
        Array, ArrayRef, BooleanArray, FixedSizeListArray, Float32Array, Float64Array, Int64Array,
        UInt64Array,
    };
    use datafusion::arrow::datatypes::{DataType, Field};
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::function::StateFieldsArgs;
    use datafusion::logical_expr::{Accumulator, AggregateUDFImpl};
    use ndarray::{Array1, Array2, Ix1, Ix2, Ix3};

    use super::{
        LinearRegressionFit, RegressionBatchStats, RegressionMoments, RegressionMomentsAccumulator,
        RegressionMomentsState, VectorContract, VectorCorrelationAgg, VectorCovarianceAgg,
        VectorMoments, VectorMomentsAccumulator, VectorMomentsState, VectorPcaFit,
        coerce_regression_fit_arguments, collect_bool_argument, expect_state_array,
        flat_matrix_state_field, infer_vector_value_type,
    };

    fn float32_vector_rows() -> FixedSizeListArray {
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![
                Some(vec![Some(1.0), Some(2.0)]),
                Some(vec![Some(3.0), Some(4.0)]),
                Some(vec![Some(5.0), Some(6.0)]),
            ],
            2,
        )
    }

    fn float64_vector_rows() -> FixedSizeListArray {
        FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
            vec![
                Some(vec![Some(1.0), Some(2.0)]),
                Some(vec![Some(3.0), Some(4.0)]),
                Some(vec![Some(5.0), Some(6.0)]),
            ],
            2,
        )
    }

    fn float32_design_rows() -> FixedSizeListArray {
        FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
            vec![Some(vec![Some(1.0)]), Some(vec![Some(2.0)]), Some(vec![Some(3.0)])],
            1,
        )
    }

    fn float64_design_rows() -> FixedSizeListArray {
        FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
            vec![Some(vec![Some(1.0)]), Some(vec![Some(2.0)]), Some(vec![Some(3.0)])],
            1,
        )
    }

    fn bools(value: bool, len: usize) -> BooleanArray { BooleanArray::from(vec![Some(value); len]) }

    fn state_arrays(values: Vec<ScalarValue>) -> Vec<ArrayRef> {
        values.into_iter().map(|value| value.to_array_of_size(1).expect("state array")).collect()
    }

    #[test]
    fn aggregate_helper_functions_cover_validation_and_coercion() {
        let vectors = float32_vector_rows();
        let vector_type = vectors.data_type().clone();

        assert_eq!(
            infer_vector_value_type("vector_covariance_agg", std::slice::from_ref(&vector_type))
                .expect("vector type"),
            DataType::Float32
        );

        let coerced = coerce_regression_fit_arguments("linear_regression_fit", &[
            vector_type.clone(),
            DataType::Int64,
            DataType::Null,
        ])
        .expect("regression coercion");
        assert_eq!(coerced[0], vector_type);
        assert_eq!(coerced[1], DataType::Float32);
        assert_eq!(coerced[2], DataType::Boolean);

        let regression = LinearRegressionFit::new();
        let coerced = regression
            .coerce_types(&[
                float64_design_rows().data_type().clone(),
                DataType::Int32,
                DataType::Boolean,
            ])
            .expect("aggregate impl coercion");
        assert_eq!(coerced[1], DataType::Float64);
        assert_eq!(coerced[2], DataType::Boolean);

        assert!(
            regression
                .coerce_types(&[
                    float32_design_rows().data_type().clone(),
                    DataType::Utf8,
                    DataType::Boolean,
                ])
                .is_err()
        );
        assert!(
            infer_vector_value_type("vector_covariance_agg", &[DataType::Utf8]).is_err(),
            "non-vector input should be rejected"
        );

        assert_eq!(
            collect_bool_argument("linear_regression_fit", &bools(true, 3)).expect("bool argument"),
            Some(true)
        );
        assert!(
            collect_bool_argument(
                "linear_regression_fit",
                &BooleanArray::from(vec![Some(true), Some(false)])
            )
            .is_err()
        );
        assert!(
            collect_bool_argument(
                "linear_regression_fit",
                &BooleanArray::from(vec![Some(true), None])
            )
            .is_err()
        );

        assert!(
            expect_state_array::<UInt64Array>(&[], 0, "vector_covariance_agg", "UInt64Array")
                .is_err()
        );
        let wrong_type = Arc::new(Float32Array::from(vec![1.0_f32])) as ArrayRef;
        assert!(
            expect_state_array::<UInt64Array>(
                &[wrong_type],
                0,
                "vector_covariance_agg",
                "UInt64Array",
            )
            .is_err()
        );
        assert!(infer_vector_value_type("vector_covariance_agg", &[]).is_err());
        assert!(
            infer_vector_value_type("vector_covariance_agg", &[DataType::new_fixed_size_list(
                DataType::Int32,
                2,
                false
            )],)
            .is_err()
        );
        assert!(flat_matrix_state_field("state", &DataType::Float32, usize::MAX).is_err());
        assert!(
            VectorMomentsState::from_contract(&VectorContract {
                value_type: DataType::Int32,
                len:        2,
            })
            .is_err()
        );
        assert!(
            RegressionMomentsState::from_contract(&VectorContract {
                value_type: DataType::Int32,
                len:        1,
            })
            .is_err()
        );
    }

    #[test]
    fn aggregate_udaf_impls_expose_typed_state_fields() {
        let vector_rows = float32_vector_rows();
        let vector_field =
            Arc::new(Field::new("vector_rows", vector_rows.data_type().clone(), false));
        let covariance = VectorCovarianceAgg::new();
        let covariance_return =
            covariance.return_field(&[Arc::clone(&vector_field)]).expect("return");
        let covariance_state = covariance
            .state_fields(StateFieldsArgs {
                name:            covariance.name(),
                input_fields:    &[Arc::clone(&vector_field)],
                return_field:    Arc::clone(&covariance_return),
                ordering_fields: &[],
                is_distinct:     false,
            })
            .expect("state fields");
        assert_eq!(covariance_state.len(), 3);
        assert!(covariance.return_type(&[]).is_err());

        let correlation = VectorCorrelationAgg::new();
        let correlation_return =
            correlation.return_field(&[Arc::clone(&vector_field)]).expect("return");
        let correlation_state = correlation
            .state_fields(StateFieldsArgs {
                name:            correlation.name(),
                input_fields:    &[Arc::clone(&vector_field)],
                return_field:    Arc::clone(&correlation_return),
                ordering_fields: &[],
                is_distinct:     false,
            })
            .expect("state fields");
        assert_eq!(correlation_state.len(), 3);
        assert!(correlation.return_type(&[]).is_err());

        let pca = VectorPcaFit::new();
        let pca_return = pca.return_field(&[Arc::clone(&vector_field)]).expect("return");
        let pca_state = pca
            .state_fields(StateFieldsArgs {
                name:            pca.name(),
                input_fields:    &[Arc::clone(&vector_field)],
                return_field:    Arc::clone(&pca_return),
                ordering_fields: &[],
                is_distinct:     false,
            })
            .expect("state fields");
        assert_eq!(pca_state.len(), 3);
        assert!(pca.return_type(&[]).is_err());

        let design_rows = float32_design_rows();
        let design_field = Arc::new(Field::new("design", design_rows.data_type().clone(), false));
        let response_field = Arc::new(Field::new("response", DataType::Float32, false));
        let add_intercept_field = Arc::new(Field::new("add_intercept", DataType::Boolean, false));
        let regression = LinearRegressionFit::new();
        let regression_return = regression
            .return_field(&[
                Arc::clone(&design_field),
                Arc::clone(&response_field),
                Arc::clone(&add_intercept_field),
            ])
            .expect("return");
        let regression_state = regression
            .state_fields(StateFieldsArgs {
                name:            regression.name(),
                input_fields:    &[
                    Arc::clone(&design_field),
                    Arc::clone(&response_field),
                    Arc::clone(&add_intercept_field),
                ],
                return_field:    Arc::clone(&regression_return),
                ordering_fields: &[],
                is_distinct:     false,
            })
            .expect("state fields");
        assert_eq!(regression_state.len(), 7);
        assert!(regression.return_type(&[]).is_err());
        let wrong_response = Arc::new(Field::new("response", DataType::Float64, false));
        assert!(
            regression
                .return_field(&[
                    Arc::clone(&design_field),
                    wrong_response,
                    Arc::clone(&add_intercept_field),
                ])
                .is_err()
        );
    }

    #[test]
    fn vector_state_and_accumulator_cover_covariance_merge_and_typed_state_paths() {
        let rows = float32_vector_rows();
        let field = Arc::new(Field::new("vector_rows", rows.data_type().clone(), false));
        let contract = VectorContract { value_type: DataType::Float32, len: 2 };

        let covariance = VectorCovarianceAgg::new()
            .return_field(&[Arc::clone(&field)])
            .expect("covariance return field");
        let correlation = VectorCorrelationAgg::new()
            .return_field(&[Arc::clone(&field)])
            .expect("correlation return field");
        assert_eq!(covariance.data_type(), correlation.data_type());

        let mut accumulator = VectorMomentsAccumulator {
            function_name: "vector_covariance_agg",
            return_field:  Arc::clone(&covariance),
            state:         VectorMomentsState::from_contract(&contract).expect("vector state"),
        };
        assert!(accumulator.evaluate().expect("empty scalar").is_null());

        accumulator.update_batch(&[Arc::new(rows.clone()) as ArrayRef]).expect("update batch");
        let state_values = accumulator.state().expect("state values");
        assert_eq!(state_values.len(), 3);
        assert_eq!(state_values[0], ScalarValue::UInt64(Some(3)));
        let state_arrays = state_arrays(state_values);

        let mut merged = VectorMomentsAccumulator {
            function_name: "vector_covariance_agg",
            return_field:  Arc::clone(&covariance),
            state:         VectorMomentsState::from_contract(&contract).expect("vector state"),
        };
        merged.merge_batch(&state_arrays).expect("merge batch");
        let ScalarValue::FixedSizeList(covariance_array) = merged.evaluate().expect("covariance")
        else {
            panic!("expected covariance fixed-size-list scalar");
        };
        let covariance = ndarrow::fixed_shape_tensor_as_array_viewd::<Float32Type>(
            &covariance,
            &covariance_array,
        )
        .expect("covariance tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 covariance tensor");
        assert_eq!(covariance.shape(), &[1, 2, 2]);
        assert!((covariance[[0, 0, 0]] - 4.0).abs() < 1.0e-5);
        assert!((covariance[[0, 1, 1]] - 4.0).abs() < 1.0e-5);
    }

    #[test]
    fn vector_moments_cover_private_error_and_f32_execution_paths() {
        let mut moments = VectorMoments::<f32>::new(2);
        assert!(moments.covariance_matrix("vector_covariance_agg").is_err());
        let wrong_rows = Array2::<f32>::zeros((1, 1));
        assert!(moments.update_batch("vector_covariance_agg", &wrong_rows.view()).is_err());
        assert!(
            moments
                .merge_batch_stats("vector_covariance_agg", 1, &[1.0], &[0.0, 0.0, 0.0, 0.0],)
                .is_err()
        );
        assert!(
            moments
                .merge_batch_stats("vector_covariance_agg", 1, &[1.0, 2.0], &[0.0, 0.0, 0.0],)
                .is_err()
        );

        let mut state = VectorMomentsState::from_contract(&VectorContract {
            value_type: DataType::Float32,
            len:        2,
        })
        .expect("vector state");
        state
            .append_batch("vector_correlation_agg", &float32_vector_rows())
            .expect("append batch");
        assert!(state.size() > 0);
        let state_values = state.state_values("vector_covariance_agg").expect("state values");
        assert_eq!(state_values.len(), 3);

        let correlation_field = VectorCorrelationAgg::new()
            .return_field(&[Arc::new(Field::new(
                "vector_rows",
                float32_vector_rows().data_type().clone(),
                false,
            ))])
            .expect("correlation return field");
        let ScalarValue::FixedSizeList(correlation_array) =
            state.evaluate_correlation("vector_correlation_agg").expect("correlation")
        else {
            panic!("expected correlation fixed-size-list scalar");
        };
        let correlation = ndarrow::fixed_shape_tensor_as_array_viewd::<Float32Type>(
            &correlation_field,
            &correlation_array,
        )
        .expect("correlation tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 correlation tensor");
        assert!((correlation[[0, 0, 1]] - 1.0).abs() < 1.0e-5);

        let ScalarValue::Struct(pca) = state.evaluate_pca("vector_pca_fit").expect("pca") else {
            panic!("expected PCA struct scalar");
        };
        let components = pca
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StructArray>()
            .expect("components tensor");
        assert_eq!(components.len(), 1);
    }

    #[test]
    fn vector_state_covers_correlation_and_pca_outputs() {
        let rows = float64_vector_rows();
        let field = Arc::new(Field::new("vector_rows", rows.data_type().clone(), false));
        let contract = VectorContract { value_type: DataType::Float64, len: 2 };

        let mut state = VectorMomentsState::from_contract(&contract).expect("vector state");
        state.append_batch("vector_correlation_agg", &rows).expect("append batch");

        let correlation_field = VectorCorrelationAgg::new()
            .return_field(&[Arc::clone(&field)])
            .expect("correlation return field");
        let ScalarValue::FixedSizeList(correlation_array) =
            state.evaluate_correlation("vector_correlation_agg").expect("correlation")
        else {
            panic!("expected correlation fixed-size-list scalar");
        };
        let correlation = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(
            &correlation_field,
            &correlation_array,
        )
        .expect("correlation tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 correlation tensor");
        assert_eq!(correlation.shape(), &[1, 2, 2]);
        assert!((correlation[[0, 0, 1]] - 1.0).abs() < 1.0e-9);

        let pca_field = VectorPcaFit::new().return_field(&[field]).expect("pca return field");
        let ScalarValue::Struct(pca) = state.evaluate_pca("vector_pca_fit").expect("pca") else {
            panic!("expected PCA struct scalar");
        };
        let DataType::Struct(fields) = pca_field.data_type() else {
            panic!("expected struct field");
        };
        let components = pca
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StructArray>()
            .expect("components tensor");
        let mut components =
            ndarrow::variable_shape_tensor_iter::<Float64Type>(&fields[0], components)
                .expect("components iterator");
        let components = components
            .next()
            .expect("first component batch")
            .expect("component tensor")
            .1
            .into_dimensionality::<Ix2>()
            .expect("component matrix");
        assert_eq!(components.shape(), &[2, 2]);

        let mean =
            pca.column(3).as_any().downcast_ref::<FixedSizeListArray>().expect("mean vector");
        let mean = ndarrow::fixed_size_list_as_array2::<Float64Type>(mean).expect("mean view");
        assert!((mean[[0, 0]] - 3.0).abs() < 1.0e-9);
        assert!((mean[[0, 1]] - 4.0).abs() < 1.0e-9);
    }

    #[test]
    fn vector_state_covers_f64_covariance_and_f64_merge_paths() {
        let rows = float64_vector_rows();
        let field = Arc::new(Field::new("vector_rows", rows.data_type().clone(), false));
        let contract = VectorContract { value_type: DataType::Float64, len: 2 };
        let covariance_field = VectorCovarianceAgg::new()
            .return_field(&[Arc::clone(&field)])
            .expect("covariance return field");

        let mut state = VectorMomentsState::from_contract(&contract).expect("vector state");
        state.append_batch("vector_covariance_agg", &rows).expect("append batch");
        let values = state.state_values("vector_covariance_agg").expect("state values");
        let arrays = state_arrays(values);

        let mut merged = VectorMomentsState::from_contract(&contract).expect("vector state");
        merged.merge_batch("vector_covariance_agg", &arrays).expect("merge batch");
        let ScalarValue::FixedSizeList(covariance_array) =
            merged.evaluate_covariance("vector_covariance_agg").expect("covariance")
        else {
            panic!("expected covariance fixed-size-list scalar");
        };
        let covariance = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(
            &covariance_field,
            &covariance_array,
        )
        .expect("covariance tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 covariance tensor");
        assert!((covariance[[0, 0, 0]] - 4.0).abs() < 1.0e-9);
        assert!(merged.size() > 0);
    }

    fn assert_regression_coefficients_f32(output: &datafusion::arrow::array::StructArray) {
        let DataType::Struct(fields) = output.data_type() else {
            panic!("expected struct output");
        };
        let coefficients = output
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StructArray>()
            .expect("coefficient tensor");
        let mut coefficients =
            ndarrow::variable_shape_tensor_iter::<Float32Type>(&fields[0], coefficients)
                .expect("coefficient iterator");
        let coefficients = coefficients
            .next()
            .expect("first coefficients batch")
            .expect("coefficient tensor")
            .1
            .into_dimensionality::<Ix1>()
            .expect("coefficient vector");
        assert_eq!(coefficients.len(), 1);
        assert!((coefficients[0] - 2.0).abs() < 1.0e-5);

        let r_squared =
            output.column(1).as_any().downcast_ref::<Float32Array>().expect("r_squared");
        assert!((r_squared.value(0) - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn regression_accumulator_covers_fit_and_merge_paths() {
        let design = float32_design_rows();
        let design_field = Arc::new(Field::new("design", design.data_type().clone(), false));
        let response_field = Arc::new(Field::new("response", DataType::Float32, false));
        let add_intercept_field = Arc::new(Field::new("add_intercept", DataType::Boolean, false));
        let contract = VectorContract { value_type: DataType::Float32, len: 1 };

        let return_field = LinearRegressionFit::new()
            .return_field(&[
                Arc::clone(&design_field),
                Arc::clone(&response_field),
                Arc::clone(&add_intercept_field),
            ])
            .expect("regression return field");

        let mut accumulator = RegressionMomentsAccumulator {
            return_field: Arc::clone(&return_field),
            state:        RegressionMomentsState::from_contract(&contract)
                .expect("regression state"),
        };
        assert!(accumulator.evaluate().expect("empty scalar").is_null());

        let response = Arc::new(Float32Array::from(vec![2.0_f32, 4.0, 6.0])) as ArrayRef;
        let add_intercept = Arc::new(bools(false, 3)) as ArrayRef;
        accumulator
            .update_batch(&[
                Arc::new(design.clone()) as ArrayRef,
                Arc::clone(&response),
                add_intercept,
            ])
            .expect("update regression batch");
        let state_values = accumulator.state().expect("regression state");
        assert_eq!(state_values.len(), 7);
        assert_eq!(state_values[0], ScalarValue::UInt64(Some(3)));
        let state_arrays = state_arrays(state_values);

        let mut merged = RegressionMomentsAccumulator {
            return_field,
            state: RegressionMomentsState::from_contract(&contract).expect("regression state"),
        };
        merged.merge_batch(&state_arrays).expect("merge regression state");
        let ScalarValue::Struct(output) = merged.evaluate().expect("regression output") else {
            panic!("expected regression struct scalar");
        };
        assert_regression_coefficients_f32(&output);
    }

    #[test]
    fn regression_state_covers_validation_and_f64_path() {
        let design = float32_design_rows();
        let response = Arc::new(Float32Array::from(vec![2.0_f32, 4.0, 6.0])) as ArrayRef;
        let contract = VectorContract { value_type: DataType::Float32, len: 1 };
        let mut left =
            RegressionMomentsState::from_contract(&contract).expect("left regression state");
        left.append_batch("linear_regression_fit", &design, &response, &bools(false, 3))
            .expect("append left batch");
        let response_values = Arc::new(Float32Array::from(vec![2.0_f32, 4.0, 6.0])) as ArrayRef;
        let mut right =
            RegressionMomentsState::from_contract(&contract).expect("right regression state");
        right
            .append_batch("linear_regression_fit", &design, &response_values, &bools(true, 3))
            .expect("append right batch");
        let right_state = state_arrays(right.state_values("linear_regression_fit").expect("state"));
        assert!(
            left.merge_batch("linear_regression_fit", &right_state).is_err(),
            "conflicting add_intercept settings should be rejected"
        );

        let mut f64_state = RegressionMomentsState::from_contract(&VectorContract {
            value_type: DataType::Float64,
            len:        1,
        })
        .expect("f64 regression state");
        let f64_response = Arc::new(Float64Array::from(vec![3.0_f64, 6.0, 9.0])) as ArrayRef;
        f64_state
            .append_batch(
                "linear_regression_fit",
                &float64_design_rows(),
                &f64_response,
                &bools(false, 3),
            )
            .expect("append f64 batch");
        let ScalarValue::Struct(f64_output) =
            f64_state.evaluate("linear_regression_fit").expect("f64 regression")
        else {
            panic!("expected f64 regression struct scalar");
        };
        let r_squared =
            f64_output.column(1).as_any().downcast_ref::<Float64Array>().expect("f64 r_squared");
        assert!((r_squared.value(0) - 1.0).abs() < 1.0e-9);

        let null_response =
            Arc::new(Float32Array::from(vec![Some(2.0_f32), None, Some(6.0)])) as ArrayRef;
        let mut invalid =
            RegressionMomentsState::from_contract(&contract).expect("invalid regression state");
        assert!(
            invalid
                .append_batch("linear_regression_fit", &design, &null_response, &bools(false, 3),)
                .is_err()
        );
        let invalid_response = Arc::new(Int64Array::from(vec![2_i64, 4, 6])) as ArrayRef;
        assert!(
            invalid
                .append_batch(
                    "linear_regression_fit",
                    &design,
                    &invalid_response,
                    &bools(false, 3),
                )
                .is_err()
        );
    }

    #[test]
    fn regression_private_paths_cover_validation_errors() {
        let mut moments = RegressionMoments::<f32>::new(1);
        let empty_design = Array2::<f32>::zeros((0, 1));
        let empty_response = Array1::<f32>::zeros(0);
        moments
            .append_batch(
                "linear_regression_fit",
                &empty_design.view(),
                &empty_response.view(),
                &BooleanArray::from(Vec::<Option<bool>>::new()),
            )
            .expect("empty batch");
        let wrong_design = Array2::<f32>::zeros((1, 2));
        let wrong_response = Array1::<f32>::zeros(1);
        assert!(
            moments
                .append_batch(
                    "linear_regression_fit",
                    &wrong_design.view(),
                    &wrong_response.view(),
                    &bools(false, 1),
                )
                .is_err()
        );
        assert!(
            moments
                .merge_batch_stats("linear_regression_fit", &RegressionBatchStats {
                    count:         1,
                    sum_x:         vec![],
                    gram_x:        vec![0.0],
                    cross_xy:      vec![0.0],
                    sum_y:         0.0,
                    sum_y2:        0.0,
                    add_intercept: Some(false),
                },)
                .is_err()
        );
        assert!(
            moments
                .merge_batch_stats("linear_regression_fit", &RegressionBatchStats {
                    count:         1,
                    sum_x:         vec![1.0],
                    gram_x:        vec![],
                    cross_xy:      vec![0.0],
                    sum_y:         0.0,
                    sum_y2:        0.0,
                    add_intercept: Some(false),
                },)
                .is_err()
        );
        assert!(
            moments
                .merge_batch_stats("linear_regression_fit", &RegressionBatchStats {
                    count:         1,
                    sum_x:         vec![1.0],
                    gram_x:        vec![1.0],
                    cross_xy:      vec![],
                    sum_y:         0.0,
                    sum_y2:        0.0,
                    add_intercept: Some(false),
                },)
                .is_err()
        );
    }

    #[test]
    fn regression_private_paths_cover_f64_intercept_state() {
        let contract = VectorContract { value_type: DataType::Float64, len: 1 };
        let mut f64_state = RegressionMomentsState::from_contract(&contract).expect("state");
        let f64_response = Arc::new(Float64Array::from(vec![3.0_f64, 6.0, 9.0])) as ArrayRef;
        f64_state
            .append_batch(
                "linear_regression_fit",
                &float64_design_rows(),
                &f64_response,
                &bools(true, 3),
            )
            .expect("append f64 batch");
        let values = f64_state.state_values("linear_regression_fit").expect("state values");
        let arrays = state_arrays(values);
        let mut merged = RegressionMomentsState::from_contract(&contract).expect("state");
        merged.merge_batch("linear_regression_fit", &arrays).expect("merge state");
        let ScalarValue::Struct(output) = merged.evaluate("linear_regression_fit").expect("fit")
        else {
            panic!("expected regression struct scalar");
        };
        let DataType::Struct(fields) = output.data_type() else {
            panic!("expected struct output");
        };
        let coefficients = output
            .column(0)
            .as_any()
            .downcast_ref::<datafusion::arrow::array::StructArray>()
            .expect("coefficient tensor");
        let mut coefficients =
            ndarrow::variable_shape_tensor_iter::<Float64Type>(&fields[0], coefficients)
                .expect("coefficient iterator");
        let coefficients = coefficients
            .next()
            .expect("first coefficients batch")
            .expect("coefficient tensor")
            .1
            .into_dimensionality::<Ix1>()
            .expect("coefficient vector");
        assert_eq!(coefficients.len(), 2);
        assert!(merged.size() > 0);
    }

    #[test]
    fn aggregate_accumulators_cover_argument_validation_errors() {
        let vector_rows = float32_vector_rows();
        let vector_field =
            Arc::new(Field::new("vector_rows", vector_rows.data_type().clone(), false));
        let return_field = VectorCovarianceAgg::new()
            .return_field(&[Arc::clone(&vector_field)])
            .expect("return field");
        let contract = VectorContract { value_type: DataType::Float32, len: 2 };
        let mut vector_accumulator = VectorMomentsAccumulator {
            function_name: "unsupported",
            return_field,
            state: VectorMomentsState::from_contract(&contract).expect("state"),
        };
        assert!(vector_accumulator.update_batch(&[]).is_err());
        assert!(
            vector_accumulator.update_batch(&[Arc::new(Float32Array::from(vec![1.0]))]).is_err()
        );
        vector_accumulator
            .update_batch(&[Arc::new(float32_vector_rows()) as ArrayRef])
            .expect("valid batch");
        assert!(vector_accumulator.evaluate().is_err());

        let regression_return = LinearRegressionFit::new()
            .return_field(&[
                Arc::new(Field::new("design", float32_design_rows().data_type().clone(), false)),
                Arc::new(Field::new("response", DataType::Float32, false)),
                Arc::new(Field::new("add_intercept", DataType::Boolean, false)),
            ])
            .expect("return field");
        let mut regression_accumulator = RegressionMomentsAccumulator {
            return_field: regression_return,
            state:        RegressionMomentsState::from_contract(&VectorContract {
                value_type: DataType::Float32,
                len:        1,
            })
            .expect("state"),
        };
        assert!(regression_accumulator.update_batch(&[]).is_err());
        assert!(
            regression_accumulator
                .update_batch(&[
                    Arc::new(Float32Array::from(vec![1.0_f32])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![1.0_f32])) as ArrayRef,
                    Arc::new(bools(true, 1)) as ArrayRef,
                ])
                .is_err()
        );
        assert!(
            regression_accumulator
                .update_batch(&[
                    Arc::new(float32_design_rows()) as ArrayRef,
                    Arc::new(Float32Array::from(vec![1.0_f32, 2.0, 3.0])) as ArrayRef,
                    Arc::new(Float32Array::from(vec![1.0_f32])) as ArrayRef,
                ])
                .is_err()
        );
    }
}
