use std::any::Any;
use std::mem::size_of_val;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::array::{
    Array, ArrayRef, BinaryArray, BooleanArray, FixedSizeListArray, Float32Array, Float64Array,
    StructArray,
};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::{Result, ScalarValue};
use datafusion::logical_expr::function::{AccumulatorArgs, StateFieldsArgs};
use datafusion::logical_expr::{AggregateUDF, AggregateUDFImpl, Documentation, Signature};
use datafusion::physical_plan::Accumulator;
use nabled::core::prelude::NabledReal;
use ndarray::{ArrayView1, ArrayView2, Axis};
use ndarrow::NdarrowElement;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

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

fn binary_state_field(name: &str) -> FieldRef { Arc::new(Field::new(name, DataType::Binary, true)) }

fn serialize_state<T: Serialize>(function_name: &str, state: &T) -> Result<Vec<u8>> {
    serde_json::to_vec(state).map_err(|error| exec_error(function_name, error))
}

fn deserialize_state<T: DeserializeOwned>(function_name: &str, bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).map_err(|error| exec_error(function_name, error))
}

fn expect_binary_array<'a>(states: &'a [ArrayRef], function_name: &str) -> Result<&'a BinaryArray> {
    let Some(state) = states.first() else {
        return Err(exec_error(function_name, "missing aggregate state column"));
    };
    state
        .as_any()
        .downcast_ref::<BinaryArray>()
        .ok_or_else(|| exec_error(function_name, "aggregate state column must be Binary"))
}

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

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum VectorRowsState {
    F32 { len: usize, values: Vec<f32> },
    F64 { len: usize, values: Vec<f64> },
}

impl VectorRowsState {
    fn from_contract(contract: &VectorContract) -> Result<Self> {
        match contract.value_type {
            DataType::Float32 => Ok(Self::F32 { len: contract.len, values: Vec::new() }),
            DataType::Float64 => Ok(Self::F64 { len: contract.len, values: Vec::new() }),
            ref actual => Err(exec_error(
                "vector aggregate",
                format!("unsupported vector value type {actual}"),
            )),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::F32 { values, .. } => values.is_empty(),
            Self::F64 { values, .. } => values.is_empty(),
        }
    }

    fn append_batch(&mut self, function_name: &str, rows: &FixedSizeListArray) -> Result<()> {
        match self {
            Self::F32 { values, .. } => {
                let view = fixed_size_list_view2::<Float32Type>(rows, function_name)?;
                values.extend(view.iter().copied());
            }
            Self::F64 { values, .. } => {
                let view = fixed_size_list_view2::<Float64Type>(rows, function_name)?;
                values.extend(view.iter().copied());
            }
        }
        Ok(())
    }

    fn merge(&mut self, function_name: &str, other: Self) -> Result<()> {
        match (self, other) {
            (
                Self::F32 { len: left_len, values: left_values },
                Self::F32 { len: right_len, values: right_values },
            ) => {
                if *left_len != right_len {
                    return Err(exec_error(
                        function_name,
                        format!(
                            "vector width mismatch while merging states: {left_len} vs {right_len}"
                        ),
                    ));
                }
                left_values.extend(right_values);
            }
            (
                Self::F64 { len: left_len, values: left_values },
                Self::F64 { len: right_len, values: right_values },
            ) => {
                if *left_len != right_len {
                    return Err(exec_error(
                        function_name,
                        format!(
                            "vector width mismatch while merging states: {left_len} vs {right_len}"
                        ),
                    ));
                }
                left_values.extend(right_values);
            }
            _ => {
                return Err(exec_error(
                    function_name,
                    "vector aggregate state type mismatch while merging",
                ));
            }
        }
        Ok(())
    }

    fn evaluate_covariance(&self, function_name: &str) -> Result<ScalarValue> {
        match self {
            Self::F32 { len, values } => {
                let rows = values.len() / *len;
                let matrix = ArrayView2::from_shape((rows, *len), values.as_slice())
                    .map_err(|error| exec_error(function_name, error))?;
                let covariance = nabled::ml::stats::covariance_matrix_view(&matrix)
                    .map_err(|error| exec_error(function_name, error))?;
                let covariance = covariance.insert_axis(Axis(0)).into_dyn();
                let (_field, array) =
                    ndarrow::arrayd_to_fixed_shape_tensor(function_name, covariance)
                        .map_err(|error| exec_error(function_name, error))?;
                Ok(ScalarValue::FixedSizeList(Arc::new(array)))
            }
            Self::F64 { len, values } => {
                let rows = values.len() / *len;
                let matrix = ArrayView2::from_shape((rows, *len), values.as_slice())
                    .map_err(|error| exec_error(function_name, error))?;
                let covariance = nabled::ml::stats::covariance_matrix_view(&matrix)
                    .map_err(|error| exec_error(function_name, error))?;
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
            Self::F32 { len, values } => {
                let rows = values.len() / *len;
                let matrix = ArrayView2::from_shape((rows, *len), values.as_slice())
                    .map_err(|error| exec_error(function_name, error))?;
                let correlation = nabled::ml::stats::correlation_matrix_view(&matrix)
                    .map_err(|error| exec_error(function_name, error))?;
                let correlation = correlation.insert_axis(Axis(0)).into_dyn();
                let (_field, array) =
                    ndarrow::arrayd_to_fixed_shape_tensor(function_name, correlation)
                        .map_err(|error| exec_error(function_name, error))?;
                Ok(ScalarValue::FixedSizeList(Arc::new(array)))
            }
            Self::F64 { len, values } => {
                let rows = values.len() / *len;
                let matrix = ArrayView2::from_shape((rows, *len), values.as_slice())
                    .map_err(|error| exec_error(function_name, error))?;
                let correlation = nabled::ml::stats::correlation_matrix_view(&matrix)
                    .map_err(|error| exec_error(function_name, error))?;
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
            Self::F32 { len, values } => {
                let rows = values.len() / *len;
                let matrix = ArrayView2::from_shape((rows, *len), values.as_slice())
                    .map_err(|error| exec_error(function_name, error))?;
                let pca = nabled::ml::pca::compute_pca_view(&matrix, None)
                    .map_err(|error| exec_error(function_name, error))?;
                build_pca_scalar::<Float32Type>(function_name, *len, pca)
            }
            Self::F64 { len, values } => {
                let rows = values.len() / *len;
                let matrix = ArrayView2::from_shape((rows, *len), values.as_slice())
                    .map_err(|error| exec_error(function_name, error))?;
                let pca = nabled::ml::pca::compute_pca_view(&matrix, None)
                    .map_err(|error| exec_error(function_name, error))?;
                build_pca_scalar::<Float64Type>(function_name, *len, pca)
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
enum RegressionRowsState {
    F32 {
        cols:          usize,
        design_values: Vec<f32>,
        response:      Vec<f32>,
        add_intercept: Option<bool>,
    },
    F64 {
        cols:          usize,
        design_values: Vec<f64>,
        response:      Vec<f64>,
        add_intercept: Option<bool>,
    },
}

impl RegressionRowsState {
    fn from_contract(contract: &VectorContract) -> Result<Self> {
        match contract.value_type {
            DataType::Float32 => Ok(Self::F32 {
                cols:          contract.len,
                design_values: Vec::new(),
                response:      Vec::new(),
                add_intercept: None,
            }),
            DataType::Float64 => Ok(Self::F64 {
                cols:          contract.len,
                design_values: Vec::new(),
                response:      Vec::new(),
                add_intercept: None,
            }),
            ref actual => Err(exec_error(
                "linear_regression_fit",
                format!("unsupported design value type {actual}"),
            )),
        }
    }

    fn is_empty(&self) -> bool {
        match self {
            Self::F32 { response, .. } => response.is_empty(),
            Self::F64 { response, .. } => response.is_empty(),
        }
    }

    fn append_batch(
        &mut self,
        function_name: &str,
        design: &FixedSizeListArray,
        response: &ArrayRef,
        add_intercept: &BooleanArray,
    ) -> Result<()> {
        let add_intercept = collect_bool_argument(function_name, add_intercept)?;
        match self {
            Self::F32 {
                cols,
                design_values,
                response: response_values,
                add_intercept: state_add_intercept,
            } => {
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
                if design_view.ncols() != *cols || design_view.nrows() != response.len() {
                    return Err(exec_error(function_name, "design/response batch shape mismatch"));
                }
                match (*state_add_intercept, add_intercept) {
                    (Some(existing), Some(incoming)) if existing != incoming => {
                        return Err(exec_error(
                            function_name,
                            "add_intercept must be constant within each group",
                        ));
                    }
                    (None, Some(incoming)) => *state_add_intercept = Some(incoming),
                    _ => {}
                }
                design_values.extend(design_view.iter().copied());
                response_values.extend(response.iter().map(Option::unwrap_or_default));
            }
            Self::F64 {
                cols,
                design_values,
                response: response_values,
                add_intercept: state_add_intercept,
            } => {
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
                if design_view.ncols() != *cols || design_view.nrows() != response.len() {
                    return Err(exec_error(function_name, "design/response batch shape mismatch"));
                }
                match (*state_add_intercept, add_intercept) {
                    (Some(existing), Some(incoming)) if existing != incoming => {
                        return Err(exec_error(
                            function_name,
                            "add_intercept must be constant within each group",
                        ));
                    }
                    (None, Some(incoming)) => *state_add_intercept = Some(incoming),
                    _ => {}
                }
                design_values.extend(design_view.iter().copied());
                response_values.extend(response.iter().map(Option::unwrap_or_default));
            }
        }
        Ok(())
    }

    fn merge(&mut self, function_name: &str, other: Self) -> Result<()> {
        match (self, other) {
            (
                Self::F32 {
                    cols: left_cols,
                    design_values: left_design,
                    response: left_response,
                    add_intercept: left_add_intercept,
                },
                Self::F32 {
                    cols: right_cols,
                    design_values: right_design,
                    response: right_response,
                    add_intercept: right_add_intercept,
                },
            ) => {
                if *left_cols != right_cols {
                    return Err(exec_error(
                        function_name,
                        format!(
                            "design width mismatch while merging states: {left_cols} vs \
                             {right_cols}"
                        ),
                    ));
                }
                match (*left_add_intercept, right_add_intercept) {
                    (Some(existing), Some(incoming)) if existing != incoming => {
                        return Err(exec_error(
                            function_name,
                            "add_intercept must be constant within each group",
                        ));
                    }
                    (None, Some(incoming)) => *left_add_intercept = Some(incoming),
                    _ => {}
                }
                left_design.extend(right_design);
                left_response.extend(right_response);
            }
            (
                Self::F64 {
                    cols: left_cols,
                    design_values: left_design,
                    response: left_response,
                    add_intercept: left_add_intercept,
                },
                Self::F64 {
                    cols: right_cols,
                    design_values: right_design,
                    response: right_response,
                    add_intercept: right_add_intercept,
                },
            ) => {
                if *left_cols != right_cols {
                    return Err(exec_error(
                        function_name,
                        format!(
                            "design width mismatch while merging states: {left_cols} vs \
                             {right_cols}"
                        ),
                    ));
                }
                match (*left_add_intercept, right_add_intercept) {
                    (Some(existing), Some(incoming)) if existing != incoming => {
                        return Err(exec_error(
                            function_name,
                            "add_intercept must be constant within each group",
                        ));
                    }
                    (None, Some(incoming)) => *left_add_intercept = Some(incoming),
                    _ => {}
                }
                left_design.extend(right_design);
                left_response.extend(right_response);
            }
            _ => {
                return Err(exec_error(
                    function_name,
                    "linear_regression_fit state type mismatch while merging",
                ));
            }
        }
        Ok(())
    }

    fn evaluate(&self, function_name: &str) -> Result<ScalarValue> {
        match self {
            Self::F32 { cols, design_values, response, add_intercept } => {
                let rows = response.len();
                let design = ArrayView2::from_shape((rows, *cols), design_values.as_slice())
                    .map_err(|error| exec_error(function_name, error))?;
                let response = ArrayView1::from(response.as_slice());
                let result = nabled::ml::regression::linear_regression_view(
                    &design,
                    &response,
                    add_intercept.unwrap_or(true),
                )
                .map_err(|error| exec_error(function_name, error))?;
                build_regression_fit_scalar_f32(function_name, result)
            }
            Self::F64 { cols, design_values, response, add_intercept } => {
                let rows = response.len();
                let design = ArrayView2::from_shape((rows, *cols), design_values.as_slice())
                    .map_err(|error| exec_error(function_name, error))?;
                let response = ArrayView1::from(response.as_slice());
                let result = nabled::ml::regression::linear_regression_view(
                    &design,
                    &response,
                    add_intercept.unwrap_or(true),
                )
                .map_err(|error| exec_error(function_name, error))?;
                build_regression_fit_scalar_f64(function_name, result)
            }
        }
    }
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
struct VectorRowsAccumulator {
    function_name: &'static str,
    return_field:  FieldRef,
    state:         VectorRowsState,
}

impl Accumulator for VectorRowsAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Binary(Some(serialize_state(self.function_name, &self.state)?))])
    }

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
        let states = expect_binary_array(states, self.function_name)?;
        for index in 0..states.len() {
            if states.is_null(index) {
                continue;
            }
            let state: VectorRowsState =
                deserialize_state(self.function_name, states.value(index))?;
            self.state.merge(self.function_name, state)?;
        }
        Ok(())
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

    fn size(&self) -> usize { size_of_val(self) }
}

#[derive(Debug)]
struct RegressionAccumulator {
    return_field: FieldRef,
    state:        RegressionRowsState,
}

impl Accumulator for RegressionAccumulator {
    fn state(&mut self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::Binary(Some(serialize_state("linear_regression_fit", &self.state)?))])
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
        let states = expect_binary_array(states, "linear_regression_fit")?;
        for index in 0..states.len() {
            if states.is_null(index) {
                continue;
            }
            let state: RegressionRowsState =
                deserialize_state("linear_regression_fit", states.value(index))?;
            self.state.merge("linear_regression_fit", state)?;
        }
        Ok(())
    }

    fn evaluate(&mut self) -> Result<ScalarValue> {
        if self.state.is_empty() {
            return ScalarValue::try_from(self.return_field.data_type());
        }
        self.state.evaluate("linear_regression_fit")
    }

    fn size(&self) -> usize { size_of_val(self) }
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
        Ok(Box::new(VectorRowsAccumulator {
            function_name: "vector_covariance_agg",
            return_field:  Arc::clone(&acc_args.return_field),
            state:         VectorRowsState::from_contract(&contract)?,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs<'_>) -> Result<Vec<FieldRef>> {
        Ok(vec![binary_state_field(&format!("{}_state", args.name))])
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
        Ok(Box::new(VectorRowsAccumulator {
            function_name: "vector_correlation_agg",
            return_field:  Arc::clone(&acc_args.return_field),
            state:         VectorRowsState::from_contract(&contract)?,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs<'_>) -> Result<Vec<FieldRef>> {
        Ok(vec![binary_state_field(&format!("{}_state", args.name))])
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
        Ok(Box::new(VectorRowsAccumulator {
            function_name: "vector_pca_fit",
            return_field:  Arc::clone(&acc_args.return_field),
            state:         VectorRowsState::from_contract(&contract)?,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs<'_>) -> Result<Vec<FieldRef>> {
        Ok(vec![binary_state_field(&format!("{}_state", args.name))])
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
        Ok(Box::new(RegressionAccumulator {
            return_field: Arc::clone(&acc_args.return_field),
            state:        RegressionRowsState::from_contract(&design)?,
        }))
    }

    fn state_fields(&self, args: StateFieldsArgs<'_>) -> Result<Vec<FieldRef>> {
        Ok(vec![binary_state_field(&format!("{}_state", args.name))])
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
        Array, ArrayRef, BinaryArray, BooleanArray, FixedSizeListArray, Float32Array, Float64Array,
        Int64Array,
    };
    use datafusion::arrow::datatypes::{DataType, Field};
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::{Accumulator, AggregateUDFImpl};
    use ndarray::{Ix1, Ix2, Ix3};

    use super::{
        LinearRegressionFit, RegressionAccumulator, RegressionRowsState, VectorContract,
        VectorCorrelationAgg, VectorCovarianceAgg, VectorPcaFit, VectorRowsAccumulator,
        VectorRowsState, coerce_regression_fit_arguments, collect_bool_argument, deserialize_state,
        expect_binary_array, infer_vector_value_type, serialize_state,
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

        assert!(expect_binary_array(&[], "vector_covariance_agg").is_err());
        let wrong_type = Arc::new(Float32Array::from(vec![1.0_f32])) as ArrayRef;
        assert!(expect_binary_array(&[wrong_type], "vector_covariance_agg").is_err());
        let binary = Arc::new(BinaryArray::from(vec![Some("state".as_bytes())])) as ArrayRef;
        assert!(expect_binary_array(&[binary], "vector_covariance_agg").is_ok());
    }

    #[test]
    fn vector_state_and_accumulator_cover_covariance_merge_and_empty_paths() {
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

        let mut accumulator = VectorRowsAccumulator {
            function_name: "vector_covariance_agg",
            return_field:  Arc::clone(&covariance),
            state:         VectorRowsState::from_contract(&contract).expect("vector state"),
        };
        assert!(accumulator.evaluate().expect("empty scalar").is_null());

        accumulator.update_batch(&[Arc::new(rows.clone()) as ArrayRef]).expect("update batch");
        let serialized = accumulator.state().expect("serialized state");
        let state_array = vec![serialized[0].to_array_of_size(1).expect("state array for merge")];

        let mut merged = VectorRowsAccumulator {
            function_name: "vector_covariance_agg",
            return_field:  Arc::clone(&covariance),
            state:         VectorRowsState::from_contract(&contract).expect("vector state"),
        };
        merged.merge_batch(&state_array).expect("merge batch");
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

        let state = VectorRowsState::from_contract(&contract).expect("state");
        let encoded = serialize_state("vector_covariance_agg", &state).expect("serialize state");
        let decoded: VectorRowsState =
            deserialize_state("vector_covariance_agg", &encoded).expect("deserialize state");
        assert_eq!(state, decoded);

        let mut left = VectorRowsState::from_contract(&contract).expect("left state");
        left.append_batch("vector_covariance_agg", &rows).expect("left append");
        let right = VectorRowsState::F64 { len: 2, values: vec![1.0, 2.0] };
        assert!(
            left.merge("vector_covariance_agg", right).is_err(),
            "mismatched state types should be rejected"
        );
    }

    #[test]
    fn vector_state_covers_correlation_and_pca_outputs() {
        let rows = float64_vector_rows();
        let field = Arc::new(Field::new("vector_rows", rows.data_type().clone(), false));
        let contract = VectorContract { value_type: DataType::Float64, len: 2 };

        let mut state = VectorRowsState::from_contract(&contract).expect("vector state");
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

        let mut accumulator = RegressionAccumulator {
            return_field: Arc::clone(&return_field),
            state:        RegressionRowsState::from_contract(&contract).expect("regression state"),
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
        let state_array =
            vec![state_values[0].to_array_of_size(1).expect("regression state array")];

        let mut merged = RegressionAccumulator {
            return_field,
            state: RegressionRowsState::from_contract(&contract).expect("regression state"),
        };
        merged.merge_batch(&state_array).expect("merge regression state");
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
            RegressionRowsState::from_contract(&contract).expect("left regression state");
        left.append_batch("linear_regression_fit", &design, &response, &bools(false, 3))
            .expect("append left batch");
        let mut right =
            RegressionRowsState::from_contract(&contract).expect("right regression state");
        right
            .append_batch("linear_regression_fit", &design, &response, &bools(true, 3))
            .expect("append right batch");
        assert!(
            left.merge("linear_regression_fit", right).is_err(),
            "conflicting add_intercept settings should be rejected"
        );

        let mut f64_state = RegressionRowsState::from_contract(&VectorContract {
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
            RegressionRowsState::from_contract(&contract).expect("invalid regression state");
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
}
