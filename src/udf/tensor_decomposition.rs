use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, PrimitiveArray, StructArray,
};
use datafusion::arrow::buffer::{OffsetBuffer, ScalarBuffer};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Axis, Ix2, Ix3};
use ndarrow::NdarrowElement;

use super::common::{
    expect_fixed_size_list_arg, expect_real_scalar_arg, expect_struct_arg,
    expect_usize_list_scalar_arg, expect_usize_list_scalar_argument, expect_usize_scalar_arg,
    expect_usize_scalar_argument, fixed_shape_tensor_array_from_flat_rows,
    fixed_shape_tensor_viewd, fixed_size_list_array_from_flat_rows, fixed_size_list_view2,
    nullable_or,
};
use super::docs::tensor_doc;
use crate::error::{exec_error, plan_error};
use crate::metadata::{
    fixed_shape_tensor_field, parse_matrix_batch_field, parse_tensor_batch_field,
    parse_variable_shape_tensor_field, parse_vector_field, scalar_field, struct_field,
    variable_shape_tensor_field, vector_field,
};
use crate::signatures::{ScalarCoercion, coerce_scalar_arguments, named_user_defined_signature};

#[derive(Debug, Clone)]
struct CpContract {
    value_type: DataType,
    shape:      Vec<usize>,
}

#[derive(Debug, Clone)]
struct TuckerContract {
    value_type: DataType,
    shape:      Vec<usize>,
    ranks:      Vec<usize>,
}

#[derive(Debug, Clone)]
struct TtContract {
    value_type: DataType,
    shape:      Vec<usize>,
}

fn coerce_tensor_rank_arguments(
    function_name: &str,
    arg_types: &[DataType],
    scalar_positions: &[(usize, ScalarCoercion)],
) -> Result<Vec<DataType>> {
    let coerced = coerce_scalar_arguments(function_name, arg_types, scalar_positions)?;
    let Some(ranks_type) = coerced.get(1) else {
        return Err(plan_error(function_name, "argument 2 is missing"));
    };
    let rank_value_type = match ranks_type {
        DataType::List(field) | DataType::LargeList(field) => field.data_type(),
        actual => {
            return Err(plan_error(
                function_name,
                format!("argument 2 must be a list of integer scalars, found {actual}"),
            ));
        }
    };
    match rank_value_type {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64 => Ok(coerced),
        actual => Err(plan_error(
            function_name,
            format!("argument 2 must be a list of integer scalars, found list<{actual}>"),
        )),
    }
}

fn coerce_struct_arguments(
    function_name: &str,
    arg_types: &[DataType],
    struct_positions: &[usize],
    scalar_positions: &[(usize, ScalarCoercion)],
) -> Result<Vec<DataType>> {
    let coerced = coerce_scalar_arguments(function_name, arg_types, scalar_positions)?;
    for position in struct_positions {
        let index =
            position.checked_sub(1).expect("argument positions are 1-based and must be positive");
        let Some(data_type) = coerced.get(index) else {
            return Err(plan_error(function_name, format!("argument {position} is missing")));
        };
        if !matches!(data_type, DataType::Struct(_)) {
            return Err(plan_error(
                function_name,
                format!("argument {position} must be a struct, found {data_type}"),
            ));
        }
    }
    Ok(coerced)
}

fn optional_real_scalar_arg(
    args: &ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<Option<f64>> {
    if args.args.len() < position {
        Ok(None)
    } else {
        expect_real_scalar_arg(args, position, function_name).map(Some)
    }
}

fn optional_usize_scalar_arg(
    args: &ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<Option<usize>> {
    if args.args.len() < position {
        Ok(None)
    } else {
        expect_usize_scalar_arg(args, position, function_name).map(Some)
    }
}

fn cp_config<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::linalg::tensor::CpAlsConfig<T>>
where
    T: NabledReal,
    nabled::linalg::tensor::CpAlsConfig<T>: Default,
{
    let defaults = nabled::linalg::tensor::CpAlsConfig::<T>::default();
    let max_iterations =
        optional_usize_scalar_arg(args, 3, function_name)?.unwrap_or(defaults.max_iterations);
    let tolerance = optional_real_scalar_arg(args, 4, function_name)?
        .map(|value| {
            T::from_f64(value)
                .ok_or_else(|| exec_error(function_name, "tolerance could not be represented"))
        })
        .transpose()?
        .unwrap_or(defaults.tolerance);
    if max_iterations == 0 || tolerance <= T::zero() {
        return Err(exec_error(function_name, "invalid CP-ALS configuration"));
    }
    Ok(nabled::linalg::tensor::CpAlsConfig { max_iterations, tolerance })
}

fn hooi_config<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::linalg::tensor::HooiConfig<T>>
where
    T: NabledReal,
    nabled::linalg::tensor::HooiConfig<T>: Default,
{
    let defaults = nabled::linalg::tensor::HooiConfig::<T>::default();
    let max_iterations =
        optional_usize_scalar_arg(args, 3, function_name)?.unwrap_or(defaults.max_iterations);
    let tolerance = optional_real_scalar_arg(args, 4, function_name)?
        .map(|value| {
            T::from_f64(value)
                .ok_or_else(|| exec_error(function_name, "tolerance could not be represented"))
        })
        .transpose()?
        .unwrap_or(defaults.tolerance);
    if max_iterations == 0 || tolerance <= T::zero() {
        return Err(exec_error(function_name, "invalid HOOI configuration"));
    }
    Ok(nabled::linalg::tensor::HooiConfig { max_iterations, tolerance })
}

fn tt_svd_config<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::linalg::tensor::TtSvdConfig<T>>
where
    T: NabledReal,
    nabled::linalg::tensor::TtSvdConfig<T>: Default,
{
    let defaults = nabled::linalg::tensor::TtSvdConfig::<T>::default();
    let max_rank = optional_usize_scalar_arg(args, 2, function_name)?.or(defaults.max_rank);
    let tolerance = optional_real_scalar_arg(args, 3, function_name)?
        .map(|value| {
            T::from_f64(value)
                .ok_or_else(|| exec_error(function_name, "tolerance could not be represented"))
        })
        .transpose()?
        .unwrap_or(defaults.tolerance);
    if matches!(max_rank, Some(0)) || tolerance <= T::zero() {
        return Err(exec_error(function_name, "invalid TT-SVD configuration"));
    }
    Ok(nabled::linalg::tensor::TtSvdConfig { max_rank, tolerance })
}

fn tt_round_config_at<T>(
    args: &ScalarFunctionArgs,
    max_rank_position: usize,
    tolerance_position: usize,
    function_name: &str,
) -> Result<nabled::linalg::tensor::TtRoundConfig<T>>
where
    T: NabledReal,
    nabled::linalg::tensor::TtRoundConfig<T>: Default,
{
    let defaults = nabled::linalg::tensor::TtRoundConfig::<T>::default();
    let max_rank =
        optional_usize_scalar_arg(args, max_rank_position, function_name)?.or(defaults.max_rank);
    let tolerance = optional_real_scalar_arg(args, tolerance_position, function_name)?
        .map(|value| {
            T::from_f64(value)
                .ok_or_else(|| exec_error(function_name, "tolerance could not be represented"))
        })
        .transpose()?
        .unwrap_or(defaults.tolerance);
    if matches!(max_rank, Some(0)) || tolerance <= T::zero() {
        return Err(exec_error(function_name, "invalid TT-round configuration"));
    }
    Ok(nabled::linalg::tensor::TtRoundConfig { max_rank, tolerance })
}

fn tt_round_config<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::linalg::tensor::TtRoundConfig<T>>
where
    T: NabledReal,
    nabled::linalg::tensor::TtRoundConfig<T>: Default,
{
    tt_round_config_at(args, 2, 3, function_name)
}

fn fixed_tensor_batch<'a>(
    args: &'a ScalarFunctionArgs,
    position: usize,
    function_name: &str,
) -> Result<(&'a FixedSizeListArray, crate::metadata::TensorBatchContract)> {
    let array = expect_fixed_size_list_arg(args, position, function_name)?;
    let contract =
        parse_tensor_batch_field(&args.arg_fields[position - 1], function_name, position)?;
    Ok((array, contract))
}

fn metadata_dim(function_name: &str, value: usize) -> Result<i32> {
    i32::try_from(value).map_err(|_| {
        exec_error(function_name, format!("dimension {value} exceeds Arrow i32 limits"))
    })
}

fn cp_struct_field(
    name: &str,
    value_type: &DataType,
    shape: &[usize],
    rank: usize,
    nullable: bool,
) -> Result<FieldRef> {
    let mut fields = vec![vector_field("weights", value_type, rank, false)?.as_ref().clone()];
    for (mode, &dimension) in shape.iter().enumerate() {
        let factor = fixed_shape_tensor_field(
            &format!("factor_{mode}"),
            value_type,
            &[dimension, rank],
            false,
        )?;
        fields.push(factor.as_ref().clone());
    }
    Ok(struct_field(name, fields, nullable))
}

fn tucker_struct_field(
    name: &str,
    value_type: &DataType,
    shape: &[usize],
    ranks: &[usize],
    nullable: bool,
) -> Result<FieldRef> {
    let mut fields =
        vec![fixed_shape_tensor_field("core", value_type, ranks, false)?.as_ref().clone()];
    for (mode, (&dimension, &rank)) in shape.iter().zip(ranks).enumerate() {
        let factor = fixed_shape_tensor_field(
            &format!("factor_{mode}"),
            value_type,
            &[dimension, rank],
            false,
        )?;
        fields.push(factor.as_ref().clone());
    }
    Ok(struct_field(name, fields, nullable))
}

fn tt_struct_field(
    name: &str,
    value_type: &DataType,
    shape: &[usize],
    nullable: bool,
) -> Result<FieldRef> {
    let mut fields = Vec::with_capacity(shape.len());
    for (mode, &dimension) in shape.iter().enumerate() {
        let core = variable_shape_tensor_field(
            &format!("core_{mode}"),
            value_type,
            3,
            Some(&[None, Some(metadata_dim(name, dimension)?), None]),
            false,
        )?;
        fields.push(core.as_ref().clone());
    }
    Ok(struct_field(name, fields, nullable))
}

fn empty_variable_tensor_storage(value_type: &DataType, rank: usize) -> Result<StructArray> {
    let rank_i32 = i32::try_from(rank)
        .map_err(|_| exec_error("tensor_decomposition", "tensor rank overflow"))?;
    let data = datafusion::arrow::array::ListArray::new(
        Arc::new(Field::new_list_field(value_type.clone(), false)),
        OffsetBuffer::new(ScalarBuffer::from(vec![0_i32])),
        match value_type {
            DataType::Float32 => Arc::new(Float32Array::from(Vec::<f32>::new())) as ArrayRef,
            DataType::Float64 => Arc::new(Float64Array::from(Vec::<f64>::new())) as ArrayRef,
            actual => {
                return Err(exec_error(
                    "tensor_decomposition",
                    format!("unsupported tensor decomposition value type {actual}"),
                ));
            }
        },
        None,
    );
    let shape = FixedSizeListArray::new(
        Arc::new(Field::new_list_field(DataType::Int32, false)),
        rank_i32,
        Arc::new(datafusion::arrow::array::Int32Array::from(Vec::<i32>::new())),
        None,
    );
    Ok(StructArray::new(
        vec![
            Field::new("data", data.data_type().clone(), false),
            Field::new("shape", shape.data_type().clone(), false),
        ]
        .into(),
        vec![Arc::new(data), Arc::new(shape)],
        None,
    ))
}

fn expect_fixed_shape_tensor_column<'a>(
    array: &'a StructArray,
    index: usize,
    function_name: &str,
    name: &str,
) -> Result<&'a FixedSizeListArray> {
    array.column(index).as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
        exec_error(function_name, format!("expected FixedSizeListArray storage for {name}"))
    })
}

fn expect_variable_shape_tensor_column<'a>(
    array: &'a StructArray,
    index: usize,
    function_name: &str,
    name: &str,
) -> Result<&'a StructArray> {
    array.column(index).as_any().downcast_ref::<StructArray>().ok_or_else(|| {
        exec_error(function_name, format!("expected StructArray storage for {name}"))
    })
}

fn parse_cp_result_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<CpContract> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} must be a CP decomposition struct"),
        ));
    };
    if fields.len() < 2 {
        return Err(plan_error(
            function_name,
            "CP decomposition struct requires weights and factors",
        ));
    }
    let weights = parse_vector_field(&fields[0], function_name, position)?;
    let mut shape = Vec::with_capacity(fields.len() - 1);
    for (mode, factor) in fields.iter().enumerate().skip(1) {
        let factor = parse_matrix_batch_field(factor, function_name, position)?;
        if factor.value_type != weights.value_type {
            return Err(plan_error(
                function_name,
                format!(
                    "factor {mode} value type {} does not match weights {}",
                    factor.value_type, weights.value_type
                ),
            ));
        }
        if factor.cols != weights.len {
            return Err(plan_error(
                function_name,
                format!(
                    "factor {mode} expected {} columns to match CP rank, found {}",
                    weights.len, factor.cols
                ),
            ));
        }
        shape.push(factor.rows);
    }
    Ok(CpContract { value_type: weights.value_type, shape })
}

fn parse_tucker_result_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<TuckerContract> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} must be a Tucker decomposition struct"),
        ));
    };
    if fields.len() < 2 {
        return Err(plan_error(
            function_name,
            "Tucker decomposition struct requires core and factors",
        ));
    }
    let core = parse_tensor_batch_field(&fields[0], function_name, position)?;
    let mut shape = Vec::with_capacity(fields.len() - 1);
    let mut ranks = Vec::with_capacity(fields.len() - 1);
    for (mode, factor) in fields.iter().enumerate().skip(1) {
        let factor = parse_matrix_batch_field(factor, function_name, position)?;
        if factor.value_type != core.value_type {
            return Err(plan_error(
                function_name,
                format!(
                    "factor {mode} value type {} does not match core {}",
                    factor.value_type, core.value_type
                ),
            ));
        }
        let rank = *core.shape.get(mode - 1).ok_or_else(|| {
            plan_error(function_name, "Tucker factor count must match the core tensor rank")
        })?;
        if factor.cols != rank {
            return Err(plan_error(
                function_name,
                format!("factor {mode} expected {rank} columns, found {}", factor.cols),
            ));
        }
        shape.push(factor.rows);
        ranks.push(rank);
    }
    if ranks.len() != core.shape.len() {
        return Err(plan_error(
            function_name,
            "Tucker factor count must match the core tensor rank",
        ));
    }
    Ok(TuckerContract { value_type: core.value_type, shape, ranks })
}

fn parse_tt_result_field(
    field: &FieldRef,
    function_name: &str,
    position: usize,
) -> Result<TtContract> {
    let DataType::Struct(fields) = field.data_type() else {
        return Err(plan_error(
            function_name,
            format!("argument {position} must be a tensor-train struct"),
        ));
    };
    if fields.is_empty() {
        return Err(plan_error(function_name, "tensor-train struct requires at least one core"));
    }
    let first = parse_variable_shape_tensor_field(&fields[0], function_name, position)?;
    let mut shape = Vec::with_capacity(fields.len());
    let first_middle = first
        .uniform_shape
        .as_ref()
        .and_then(|shape| shape.get(1))
        .copied()
        .flatten()
        .ok_or_else(|| {
            plan_error(function_name, "tensor-train core metadata must pin the mode size")
        })?;
    shape.push(usize::try_from(first_middle).map_err(|_| {
        plan_error(function_name, "tensor-train mode metadata must be non-negative")
    })?);
    for core in fields.iter().skip(1) {
        let core = parse_variable_shape_tensor_field(core, function_name, position)?;
        if core.value_type != first.value_type {
            return Err(plan_error(
                function_name,
                format!(
                    "tensor-train core value type {} does not match {}",
                    core.value_type, first.value_type
                ),
            ));
        }
        if core.dimensions != 3 {
            return Err(plan_error(
                function_name,
                format!("tensor-train cores must be rank-3, found rank {}", core.dimensions),
            ));
        }
        let middle = core
            .uniform_shape
            .as_ref()
            .and_then(|shape| shape.get(1))
            .copied()
            .flatten()
            .ok_or_else(|| {
                plan_error(function_name, "tensor-train core metadata must pin the mode size")
            })?;
        shape.push(usize::try_from(middle).map_err(|_| {
            plan_error(function_name, "tensor-train mode metadata must be non-negative")
        })?);
    }
    Ok(TtContract { value_type: first.value_type, shape })
}

fn build_cp_result_array<T>(
    function_name: &str,
    value_type: &DataType,
    shape: &[usize],
    results: &[nabled::linalg::tensor::CpAlsNdResult<T::Native>],
) -> Result<StructArray>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let batch = results.len();
    let rank = results.first().map_or(0, |result| result.weights.len());
    let weights: Vec<T::Native> =
        results.iter().flat_map(|result| result.weights.iter().copied()).collect();
    let weights =
        Arc::new(fixed_size_list_array_from_flat_rows::<T>(function_name, batch, rank, &weights)?)
            as ArrayRef;
    let mut arrays = vec![weights];
    let mut fields = vec![vector_field("weights", value_type, rank, false)?.as_ref().clone()];
    for (mode, &dimension) in shape.iter().enumerate() {
        let factor_values: Vec<T::Native> =
            results.iter().flat_map(|result| result.factors[mode].iter().copied()).collect();
        let (_field, factor_array) = fixed_shape_tensor_array_from_flat_rows::<T>(
            function_name,
            batch,
            &[dimension, rank],
            factor_values,
        )?;
        arrays.push(Arc::new(factor_array));
        fields.push(
            fixed_shape_tensor_field(
                &format!("factor_{mode}"),
                value_type,
                &[dimension, rank],
                false,
            )?
            .as_ref()
            .clone(),
        );
    }
    Ok(StructArray::new(fields.into(), arrays, None))
}

fn build_tucker_result_array<T>(
    function_name: &str,
    value_type: &DataType,
    shape: &[usize],
    results: &[nabled::linalg::tensor::HosvdNdResult<T::Native>],
) -> Result<StructArray>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let batch = results.len();
    let ranks = results.first().map_or_else(Vec::new, |result| result.core.shape().to_vec());
    let core_values: Vec<T::Native> =
        results.iter().flat_map(|result| result.core.iter().copied()).collect();
    let (_core_field, core_array) =
        fixed_shape_tensor_array_from_flat_rows::<T>(function_name, batch, &ranks, core_values)?;
    let mut arrays = vec![Arc::new(core_array) as ArrayRef];
    let mut fields =
        vec![fixed_shape_tensor_field("core", value_type, &ranks, false)?.as_ref().clone()];
    for (mode, (&dimension, &rank)) in shape.iter().zip(&ranks).enumerate() {
        let factor_values: Vec<T::Native> =
            results.iter().flat_map(|result| result.factors[mode].iter().copied()).collect();
        let (_field, factor_array) = fixed_shape_tensor_array_from_flat_rows::<T>(
            function_name,
            batch,
            &[dimension, rank],
            factor_values,
        )?;
        arrays.push(Arc::new(factor_array));
        fields.push(
            fixed_shape_tensor_field(
                &format!("factor_{mode}"),
                value_type,
                &[dimension, rank],
                false,
            )?
            .as_ref()
            .clone(),
        );
    }
    Ok(StructArray::new(fields.into(), arrays, None))
}

fn build_tt_result_array<T>(
    function_name: &str,
    value_type: &DataType,
    shape: &[usize],
    results: &[nabled::linalg::tensor::TensorTrainResult<T::Native>],
) -> Result<StructArray>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let batch = results.len();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(shape.len());
    let mut fields = Vec::with_capacity(shape.len());
    for (mode, &dimension) in shape.iter().enumerate() {
        let field = variable_shape_tensor_field(
            &format!("core_{mode}"),
            value_type,
            3,
            Some(&[None, Some(metadata_dim(function_name, dimension)?), None]),
            false,
        )?;
        fields.push(field.as_ref().clone());
        if batch == 0 {
            arrays.push(Arc::new(empty_variable_tensor_storage(value_type, 3)?) as ArrayRef);
            continue;
        }
        let cores =
            results.iter().map(|result| result.cores[mode].clone().into_dyn()).collect::<Vec<_>>();
        let (_field, array) = ndarrow::arrays_to_variable_shape_tensor(
            &format!("core_{mode}"),
            cores,
            Some(vec![None, Some(metadata_dim(function_name, dimension)?), None]),
        )
        .map_err(|error| exec_error(function_name, error))?;
        arrays.push(Arc::new(array) as ArrayRef);
    }
    Ok(StructArray::new(fields.into(), arrays, None))
}

fn owned_cp_results<T>(
    field: &FieldRef,
    array: &StructArray,
    function_name: &str,
) -> Result<Vec<nabled::linalg::tensor::CpAlsNdResult<T::Native>>>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let contract = parse_cp_result_field(field, function_name, 1)?;
    let DataType::Struct(fields) = field.data_type() else { unreachable!() };
    let weights =
        array.column(0).as_any().downcast_ref::<FixedSizeListArray>().ok_or_else(|| {
            exec_error(function_name, "expected FixedSizeListArray storage for weights")
        })?;
    let weights_view = fixed_size_list_view2::<T>(weights, function_name)?;
    let factor_views = fields
        .iter()
        .enumerate()
        .skip(1)
        .map(|(index, factor_field)| {
            let factor =
                expect_fixed_shape_tensor_column(array, index, function_name, factor_field.name())?;
            fixed_shape_tensor_viewd::<T>(factor_field, factor, function_name)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        let weights = weights_view.index_axis(Axis(0), row).to_owned();
        let factors = factor_views
            .iter()
            .map(|view| {
                view.index_axis(Axis(0), row)
                    .into_dimensionality::<Ix2>()
                    .map(|matrix| matrix.to_owned())
                    .map_err(|error| exec_error(function_name, error))
            })
            .collect::<Result<Vec<_>>>()?;
        outputs.push(nabled::linalg::tensor::CpAlsNdResult {
            weights,
            factors,
            shape: contract.shape.clone(),
        });
    }
    Ok(outputs)
}

fn owned_tucker_results<T>(
    field: &FieldRef,
    array: &StructArray,
    function_name: &str,
) -> Result<Vec<nabled::linalg::tensor::HosvdNdResult<T::Native>>>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let _contract = parse_tucker_result_field(field, function_name, 1)?;
    let DataType::Struct(fields) = field.data_type() else { unreachable!() };
    let core = expect_fixed_shape_tensor_column(array, 0, function_name, "core")?;
    let core_view = fixed_shape_tensor_viewd::<T>(&fields[0], core, function_name)?;
    let factor_views = fields
        .iter()
        .enumerate()
        .skip(1)
        .map(|(index, factor_field)| {
            let factor =
                expect_fixed_shape_tensor_column(array, index, function_name, factor_field.name())?;
            fixed_shape_tensor_viewd::<T>(factor_field, factor, function_name)
        })
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = Vec::with_capacity(array.len());
    for row in 0..array.len() {
        let core = core_view.index_axis(Axis(0), row).to_owned().into_dyn();
        let factors = factor_views
            .iter()
            .map(|view| {
                view.index_axis(Axis(0), row)
                    .into_dimensionality::<Ix2>()
                    .map(|matrix| matrix.to_owned())
                    .map_err(|error| exec_error(function_name, error))
            })
            .collect::<Result<Vec<_>>>()?;
        outputs.push(nabled::linalg::tensor::HosvdNdResult { core, factors });
    }
    Ok(outputs)
}

fn owned_tt_results<T>(
    field: &FieldRef,
    array: &StructArray,
    function_name: &str,
) -> Result<Vec<nabled::linalg::tensor::TensorTrainResult<T::Native>>>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let contract = parse_tt_result_field(field, function_name, 1)?;
    let DataType::Struct(fields) = field.data_type() else { unreachable!() };
    let mut core_iters = fields
        .iter()
        .enumerate()
        .map(|(index, core_field)| {
            let core = expect_variable_shape_tensor_column(
                array,
                index,
                function_name,
                core_field.name(),
            )?;
            ndarrow::variable_shape_tensor_iter::<T>(core_field.as_ref(), core)
                .map_err(|error| exec_error(function_name, error))
        })
        .collect::<Result<Vec<_>>>()?;
    let mut outputs = Vec::with_capacity(array.len());
    for _ in 0..array.len() {
        let mut cores = Vec::with_capacity(core_iters.len());
        for iter in &mut core_iters {
            let (_, core) = iter
                .next()
                .ok_or_else(|| exec_error(function_name, "tensor-train core iterator ended early"))?
                .map_err(|error| exec_error(function_name, error))?;
            cores.push(
                core.into_dimensionality::<Ix3>()
                    .map_err(|error| exec_error(function_name, error))?
                    .to_owned(),
            );
        }
        outputs.push(nabled::linalg::tensor::TensorTrainResult {
            cores,
            shape: contract.shape.clone(),
        });
    }
    Ok(outputs)
}

fn invoke_cp_als3_typed<T>(args: &ScalarFunctionArgs, function_name: &str) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: nabled::linalg::tensor::CpAlsScalar + NdarrowElement,
    nabled::linalg::tensor::CpAlsConfig<T::Native>: Default,
{
    let (tensor, contract) = fixed_tensor_batch(args, 1, function_name)?;
    if contract.shape.len() != 3 {
        return Err(exec_error(
            function_name,
            "tensor_cp_als3 requires a rank-3 fixed-shape tensor batch",
        ));
    }
    let rank = expect_usize_scalar_arg(args, 2, function_name)?;
    let config = cp_config::<T::Native>(args, function_name)?;
    let tensor_view = fixed_shape_tensor_viewd::<T>(&args.arg_fields[0], tensor, function_name)?;
    let results = (0..tensor.len())
        .map(|row| {
            let row_tensor = tensor_view
                .index_axis(Axis(0), row)
                .into_dimensionality::<Ix3>()
                .map_err(|error| exec_error(function_name, error))?;
            nabled::linalg::tensor::cp_als3_view(&row_tensor, rank, &config)
                .map(|result| nabled::linalg::tensor::CpAlsNdResult {
                    weights: result.weights,
                    factors: vec![result.factor_0, result.factor_1, result.factor_2],
                    shape:   contract.shape.clone(),
                })
                .map_err(|error| exec_error(function_name, error))
        })
        .collect::<Result<Vec<_>>>()?;
    let struct_array =
        build_cp_result_array::<T>(function_name, &contract.value_type, &contract.shape, &results)?;
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn invoke_cp_als_nd_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: nabled::linalg::tensor::CpAlsScalar + NdarrowElement,
    nabled::linalg::tensor::CpAlsConfig<T::Native>: Default,
{
    let (tensor, contract) = fixed_tensor_batch(args, 1, function_name)?;
    let rank = expect_usize_scalar_arg(args, 2, function_name)?;
    let config = cp_config::<T::Native>(args, function_name)?;
    let tensor_view = fixed_shape_tensor_viewd::<T>(&args.arg_fields[0], tensor, function_name)?;
    let results = (0..tensor.len())
        .map(|row| {
            let row_tensor = tensor_view.index_axis(Axis(0), row);
            nabled::linalg::tensor::cp_als_nd_view(&row_tensor, rank, &config)
                .map_err(|error| exec_error(function_name, error))
        })
        .collect::<Result<Vec<_>>>()?;
    let struct_array =
        build_cp_result_array::<T>(function_name, &contract.value_type, &contract.shape, &results)?;
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn invoke_cp_reconstruct_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    require_rank3: bool,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let field = &args.arg_fields[0];
    let array = expect_struct_arg(args, 1, function_name)?;
    let results = owned_cp_results::<T>(field, array, function_name)?;
    if require_rank3 && results.iter().any(|result| result.factors.len() != 3) {
        return Err(exec_error(
            function_name,
            "tensor_cp_als3_reconstruct requires exactly three factor matrices",
        ));
    }
    let shape = parse_cp_result_field(field, function_name, 1)?.shape;
    let flattened = results
        .iter()
        .map(|result| {
            if require_rank3 {
                let result = nabled::linalg::tensor::CpAls3Result {
                    weights:  result.weights.clone(),
                    factor_0: result.factors[0].clone(),
                    factor_1: result.factors[1].clone(),
                    factor_2: result.factors[2].clone(),
                };
                nabled::linalg::tensor::cp_als3_reconstruct(&result)
                    .map(ndarray::ArrayBase::into_dyn)
                    .map_err(|error| exec_error(function_name, error))
            } else {
                nabled::linalg::tensor::cp_als_nd_reconstruct(result)
                    .map_err(|error| exec_error(function_name, error))
            }
            .map(|tensor| tensor.iter().copied().collect::<Vec<_>>())
        })
        .collect::<Result<Vec<_>>>()?
        .into_iter()
        .flatten()
        .collect::<Vec<_>>();
    let (_field, array) = fixed_shape_tensor_array_from_flat_rows::<T>(
        function_name,
        results.len(),
        &shape,
        flattened,
    )?;
    Ok(ColumnarValue::Array(Arc::new(array)))
}

fn invoke_hosvd_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    use_hooi: bool,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: nabled::linalg::tensor::HosvdNdScalar
        + nabled::linalg::tensor::HooiNdScalar
        + NdarrowElement,
    nabled::linalg::tensor::HooiConfig<T::Native>: Default,
{
    let (tensor, contract) = fixed_tensor_batch(args, 1, function_name)?;
    let ranks = expect_usize_list_scalar_arg(args, 2, function_name)?;
    if ranks.len() != contract.shape.len() {
        return Err(exec_error(
            function_name,
            format!("expected {} Tucker ranks, found {}", contract.shape.len(), ranks.len()),
        ));
    }
    let tensor_view = fixed_shape_tensor_viewd::<T>(&args.arg_fields[0], tensor, function_name)?;
    let results = (0..tensor.len())
        .map(|row| {
            let row_tensor = tensor_view.index_axis(Axis(0), row);
            if use_hooi {
                let config = hooi_config::<T::Native>(args, function_name)?;
                nabled::linalg::tensor::hooi_nd_view(&row_tensor, &ranks, &config)
            } else {
                nabled::linalg::tensor::hosvd_nd_view(&row_tensor, &ranks)
            }
            .map_err(|error| exec_error(function_name, error))
        })
        .collect::<Result<Vec<_>>>()?;
    let struct_array = build_tucker_result_array::<T>(
        function_name,
        &contract.value_type,
        &contract.shape,
        &results,
    )?;
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn invoke_tucker_project_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let (tensor, _tensor_contract) = fixed_tensor_batch(args, 1, function_name)?;
    let factors_field = &args.arg_fields[1];
    let factors_array = expect_struct_arg(args, 2, function_name)?;
    let factors = owned_tucker_results::<T>(factors_field, factors_array, function_name)?;
    if tensor.len() != factors.len() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} tensors vs {} factor rows",
                tensor.len(),
                factors.len()
            ),
        ));
    }
    let ranks = parse_tucker_result_field(factors_field, function_name, 2)?.ranks;
    let tensor_view = fixed_shape_tensor_viewd::<T>(&args.arg_fields[0], tensor, function_name)?;
    let mut outputs = Vec::with_capacity(tensor.len());
    for (row, factors) in factors.iter().enumerate() {
        let row_tensor = tensor_view.index_axis(Axis(0), row);
        let projected = nabled::linalg::tensor::tucker_project_view(&row_tensor, &factors.factors)
            .map_err(|error| exec_error(function_name, error))?;
        outputs.extend(projected.iter().copied());
    }
    let (_field, array) =
        fixed_shape_tensor_array_from_flat_rows::<T>(function_name, tensor.len(), &ranks, outputs)?;
    Ok(ColumnarValue::Array(Arc::new(array)))
}

fn invoke_tucker_expand_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let field = &args.arg_fields[0];
    let array = expect_struct_arg(args, 1, function_name)?;
    let contract = parse_tucker_result_field(field, function_name, 1)?;
    let results = owned_tucker_results::<T>(field, array, function_name)?;
    let mut outputs = Vec::new();
    for result in &results {
        let output = nabled::linalg::tensor::hosvd_nd_reconstruct(result)
            .map_err(|error| exec_error(function_name, error))?;
        outputs.extend(output.iter().copied());
    }
    let (_field, array) = fixed_shape_tensor_array_from_flat_rows::<T>(
        function_name,
        results.len(),
        &contract.shape,
        outputs,
    )?;
    Ok(ColumnarValue::Array(Arc::new(array)))
}

fn invoke_tt_svd_typed<T>(args: &ScalarFunctionArgs, function_name: &str) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: nabled::linalg::tensor::TtSvdScalar + NdarrowElement,
    nabled::linalg::tensor::TtSvdConfig<T::Native>: Default,
{
    let (tensor, contract) = fixed_tensor_batch(args, 1, function_name)?;
    let config = tt_svd_config::<T::Native>(args, function_name)?;
    let tensor_view = fixed_shape_tensor_viewd::<T>(&args.arg_fields[0], tensor, function_name)?;
    let results = (0..tensor.len())
        .map(|row| {
            let row_tensor = tensor_view.index_axis(Axis(0), row);
            nabled::linalg::tensor::tt_svd_view(&row_tensor, &config)
                .map_err(|error| exec_error(function_name, error))
        })
        .collect::<Result<Vec<_>>>()?;
    let struct_array =
        build_tt_result_array::<T>(function_name, &contract.value_type, &contract.shape, &results)?;
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn invoke_tt_unary_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(
        &nabled::linalg::tensor::TensorTrainResult<T::Native>,
    ) -> std::result::Result<
        nabled::linalg::tensor::TensorTrainResult<T::Native>,
        nabled::arrow::ArrowInteropError,
    >,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: nabled::linalg::tensor::TtSvdScalar + NdarrowElement,
{
    let field = &args.arg_fields[0];
    let array = expect_struct_arg(args, 1, function_name)?;
    let contract = parse_tt_result_field(field, function_name, 1)?;
    let results = owned_tt_results::<T>(field, array, function_name)?
        .into_iter()
        .map(|result| op(&result).map_err(|error| exec_error(function_name, error)))
        .collect::<Result<Vec<_>>>()?;
    let struct_array =
        build_tt_result_array::<T>(function_name, &contract.value_type, &contract.shape, &results)?;
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn invoke_tt_round_typed<T>(args: &ScalarFunctionArgs, function_name: &str) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: nabled::linalg::tensor::TtSvdScalar + NdarrowElement,
    nabled::linalg::tensor::TtRoundConfig<T::Native>: Default,
{
    let field = &args.arg_fields[0];
    let array = expect_struct_arg(args, 1, function_name)?;
    let contract = parse_tt_result_field(field, function_name, 1)?;
    let config = tt_round_config::<T::Native>(args, function_name)?;
    let results = owned_tt_results::<T>(field, array, function_name)?
        .into_iter()
        .map(|result| {
            nabled::arrow::tensor::tt_round(&result, &config)
                .map_err(|error| exec_error(function_name, error))
        })
        .collect::<Result<Vec<_>>>()?;
    let struct_array =
        build_tt_result_array::<T>(function_name, &contract.value_type, &contract.shape, &results)?;
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn invoke_tt_binary_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(
        &nabled::linalg::tensor::TensorTrainResult<T::Native>,
        &nabled::linalg::tensor::TensorTrainResult<T::Native>,
    ) -> std::result::Result<
        nabled::linalg::tensor::TensorTrainResult<T::Native>,
        nabled::arrow::ArrowInteropError,
    >,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_field = &args.arg_fields[0];
    let right_field = &args.arg_fields[1];
    let left_array = expect_struct_arg(args, 1, function_name)?;
    let right_array = expect_struct_arg(args, 2, function_name)?;
    let left_contract = parse_tt_result_field(left_field, function_name, 1)?;
    let right_contract = parse_tt_result_field(right_field, function_name, 2)?;
    if left_contract.value_type != right_contract.value_type {
        return Err(exec_error(
            function_name,
            format!(
                "tensor-train value type mismatch: {} vs {}",
                left_contract.value_type, right_contract.value_type
            ),
        ));
    }
    if left_contract.shape != right_contract.shape {
        return Err(exec_error(
            function_name,
            format!(
                "tensor-train shape mismatch: {:?} vs {:?}",
                left_contract.shape, right_contract.shape
            ),
        ));
    }
    let left = owned_tt_results::<T>(left_field, left_array, function_name)?;
    let right = owned_tt_results::<T>(right_field, right_array, function_name)?;
    if left.len() != right.len() {
        return Err(exec_error(
            function_name,
            format!("batch length mismatch: {} vs {}", left.len(), right.len()),
        ));
    }
    let results = left
        .iter()
        .zip(&right)
        .map(|(left, right)| op(left, right).map_err(|error| exec_error(function_name, error)))
        .collect::<Result<Vec<_>>>()?;
    let struct_array = build_tt_result_array::<T>(
        function_name,
        &left_contract.value_type,
        &left_contract.shape,
        &results,
    )?;
    Ok(ColumnarValue::Array(Arc::new(struct_array)))
}

fn invoke_tt_scalar_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(
        &nabled::linalg::tensor::TensorTrainResult<T::Native>,
    ) -> std::result::Result<T::Native, nabled::arrow::ArrowInteropError>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let field = &args.arg_fields[0];
    let array = expect_struct_arg(args, 1, function_name)?;
    let results = owned_tt_results::<T>(field, array, function_name)?;
    let outputs = results
        .iter()
        .map(|result| op(result).map_err(|error| exec_error(function_name, error)))
        .collect::<Result<Vec<_>>>()?;
    Ok(ColumnarValue::Array(Arc::new(PrimitiveArray::<T>::from_iter_values(outputs))))
}

fn invoke_tt_inner_typed<T>(args: &ScalarFunctionArgs, function_name: &str) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let left_field = &args.arg_fields[0];
    let right_field = &args.arg_fields[1];
    let left_array = expect_struct_arg(args, 1, function_name)?;
    let right_array = expect_struct_arg(args, 2, function_name)?;
    let left_contract = parse_tt_result_field(left_field, function_name, 1)?;
    let right_contract = parse_tt_result_field(right_field, function_name, 2)?;
    if left_contract.value_type != right_contract.value_type
        || left_contract.shape != right_contract.shape
    {
        return Err(exec_error(
            function_name,
            "tensor-train inputs must share value type and shape",
        ));
    }
    let left = owned_tt_results::<T>(left_field, left_array, function_name)?;
    let right = owned_tt_results::<T>(right_field, right_array, function_name)?;
    if left.len() != right.len() {
        return Err(exec_error(
            function_name,
            "tensor-train input batches must have matching lengths",
        ));
    }
    let outputs = left
        .iter()
        .zip(&right)
        .map(|(left, right)| {
            nabled::arrow::tensor::tt_inner(left, right)
                .map_err(|error| exec_error(function_name, error))
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(ColumnarValue::Array(Arc::new(PrimitiveArray::<T>::from_iter_values(outputs))))
}

fn invoke_tt_reconstruct_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let field = &args.arg_fields[0];
    let array = expect_struct_arg(args, 1, function_name)?;
    let contract = parse_tt_result_field(field, function_name, 1)?;
    let results = owned_tt_results::<T>(field, array, function_name)?;
    let mut outputs = Vec::new();
    for result in &results {
        let output = nabled::linalg::tensor::tt_svd_reconstruct(result)
            .map_err(|error| exec_error(function_name, error))?;
        outputs.extend(output.iter().copied());
    }
    let (_field, array) = fixed_shape_tensor_array_from_flat_rows::<T>(
        function_name,
        results.len(),
        &contract.shape,
        outputs,
    )?;
    Ok(ColumnarValue::Array(Arc::new(array)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorCpAls3 {
    signature: Signature,
}

impl TensorCpAls3 {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&[
                "tensor",
                "rank",
                "max_iterations",
                "tolerance",
            ]),
        }
    }
}

impl ScalarUDFImpl for TensorCpAls3 {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_cp_als3" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[
            (2, ScalarCoercion::Integer),
            (3, ScalarCoercion::Integer),
            (4, ScalarCoercion::Real),
        ])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if tensor.shape.len() != 3 {
            return Err(plan_error(
                self.name(),
                "tensor_cp_als3 requires a rank-3 fixed-shape tensor batch",
            ));
        }
        let rank = expect_usize_scalar_argument(&args, 2, self.name())?;
        cp_struct_field(
            self.name(),
            &tensor.value_type,
            &tensor.shape,
            rank,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match contract.value_type {
            DataType::Float32 => invoke_cp_als3_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_cp_als3_typed::<Float64Type>(&args, self.name()),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute a rank-R CP decomposition for each rank-3 fixed-shape tensor row.",
                "tensor_cp_als3(tensor_batch, 4, max_iterations => 100, tolerance => 1e-6)",
            )
            .with_argument("tensor_batch", "Canonical fixed-shape rank-3 tensor batch.")
            .with_argument("rank", "Requested CP rank.")
            .with_argument("max_iterations", "Optional ALS iteration budget.")
            .with_argument("tolerance", "Optional ALS convergence tolerance.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorCpAls3Reconstruct;

impl ScalarUDFImpl for TensorCpAls3Reconstruct {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_cp_als3_reconstruct" }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| named_user_defined_signature(&["cp"]));
        &SIGNATURE
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_struct_arguments(self.name(), arg_types, &[1], &[])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_cp_result_field(&args.arg_fields[0], self.name(), 1)?;
        if contract.shape.len() != 3 {
            return Err(plan_error(
                self.name(),
                "tensor_cp_als3_reconstruct requires exactly three CP factors",
            ));
        }
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &contract.shape,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_cp_result_field(&args.arg_fields[0], self.name(), 1)?;
        match contract.value_type {
            DataType::Float32 => {
                invoke_cp_reconstruct_typed::<Float32Type>(&args, self.name(), true)
            }
            DataType::Float64 => {
                invoke_cp_reconstruct_typed::<Float64Type>(&args, self.name(), true)
            }
            actual => Err(exec_error(self.name(), format!("unsupported CP value type {actual}"))),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reconstruct a rank-3 tensor batch from CP factors returned by tensor_cp_als3.",
                "tensor_cp_als3_reconstruct(cp_result)",
            )
            .with_argument("cp_result", "Struct returned by tensor_cp_als3.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorCpAlsNd {
    signature: Signature,
}

impl TensorCpAlsNd {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&[
                "tensor",
                "rank",
                "max_iterations",
                "tolerance",
            ]),
        }
    }
}

impl ScalarUDFImpl for TensorCpAlsNd {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_cp_als_nd" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[
            (2, ScalarCoercion::Integer),
            (3, ScalarCoercion::Integer),
            (4, ScalarCoercion::Real),
        ])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rank = expect_usize_scalar_argument(&args, 2, self.name())?;
        cp_struct_field(
            self.name(),
            &tensor.value_type,
            &tensor.shape,
            rank,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match contract.value_type {
            DataType::Float32 => invoke_cp_als_nd_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_cp_als_nd_typed::<Float64Type>(&args, self.name()),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute a rank-R CP decomposition for each fixed-shape tensor row.",
                "tensor_cp_als_nd(tensor_batch, 4, max_iterations => 100, tolerance => 1e-6)",
            )
            .with_argument("tensor_batch", "Canonical fixed-shape tensor batch.")
            .with_argument("rank", "Requested CP rank.")
            .with_argument("max_iterations", "Optional ALS iteration budget.")
            .with_argument("tolerance", "Optional ALS convergence tolerance.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorCpAlsNdReconstruct;

impl ScalarUDFImpl for TensorCpAlsNdReconstruct {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_cp_als_nd_reconstruct" }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| named_user_defined_signature(&["cp"]));
        &SIGNATURE
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_struct_arguments(self.name(), arg_types, &[1], &[])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_cp_result_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &contract.shape,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_cp_result_field(&args.arg_fields[0], self.name(), 1)?;
        match contract.value_type {
            DataType::Float32 => {
                invoke_cp_reconstruct_typed::<Float32Type>(&args, self.name(), false)
            }
            DataType::Float64 => {
                invoke_cp_reconstruct_typed::<Float64Type>(&args, self.name(), false)
            }
            actual => Err(exec_error(self.name(), format!("unsupported CP value type {actual}"))),
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reconstruct a fixed-shape tensor batch from CP factors returned by \
                 tensor_cp_als_nd.",
                "tensor_cp_als_nd_reconstruct(cp_result)",
            )
            .with_argument("cp_result", "Struct returned by tensor_cp_als_nd.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorHosvdNd {
    signature: Signature,
}

impl TensorHosvdNd {
    fn new() -> Self { Self { signature: named_user_defined_signature(&["tensor", "ranks"]) } }
}

impl ScalarUDFImpl for TensorHosvdNd {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_hosvd_nd" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_tensor_rank_arguments(self.name(), arg_types, &[])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let ranks = expect_usize_list_scalar_argument(&args, 2, self.name())?;
        if ranks.len() != tensor.shape.len() {
            return Err(plan_error(self.name(), "rank list length must match the tensor rank"));
        }
        tucker_struct_field(
            self.name(),
            &tensor.value_type,
            &tensor.shape,
            &ranks,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match tensor.value_type {
            DataType::Float32 => invoke_hosvd_typed::<Float32Type>(&args, self.name(), false),
            DataType::Float64 => invoke_hosvd_typed::<Float64Type>(&args, self.name(), false),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute a Tucker decomposition with HOSVD for each fixed-shape tensor row.",
                "tensor_hosvd_nd(tensor_batch, [2, 2, 2])",
            )
            .with_argument("tensor_batch", "Canonical fixed-shape tensor batch.")
            .with_argument("ranks", "List scalar giving the per-mode Tucker ranks.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorHooiNd {
    signature: Signature,
}

impl TensorHooiNd {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&[
                "tensor",
                "ranks",
                "max_iterations",
                "tolerance",
            ]),
        }
    }
}

impl ScalarUDFImpl for TensorHooiNd {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_hooi_nd" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_tensor_rank_arguments(self.name(), arg_types, &[
            (3, ScalarCoercion::Integer),
            (4, ScalarCoercion::Real),
        ])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let ranks = expect_usize_list_scalar_argument(&args, 2, self.name())?;
        if ranks.len() != tensor.shape.len() {
            return Err(plan_error(self.name(), "rank list length must match the tensor rank"));
        }
        tucker_struct_field(
            self.name(),
            &tensor.value_type,
            &tensor.shape,
            &ranks,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match tensor.value_type {
            DataType::Float32 => invoke_hosvd_typed::<Float32Type>(&args, self.name(), true),
            DataType::Float64 => invoke_hosvd_typed::<Float64Type>(&args, self.name(), true),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute a Tucker decomposition with HOOI refinement for each fixed-shape tensor \
                 row.",
                "tensor_hooi_nd(tensor_batch, [2, 2, 2], max_iterations => 25, tolerance => 1e-6)",
            )
            .with_argument("tensor_batch", "Canonical fixed-shape tensor batch.")
            .with_argument("ranks", "List scalar giving the per-mode Tucker ranks.")
            .with_argument("max_iterations", "Optional HOOI iteration budget.")
            .with_argument("tolerance", "Optional HOOI convergence tolerance.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorTuckerProject;

impl ScalarUDFImpl for TensorTuckerProject {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_tucker_project" }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| named_user_defined_signature(&["tensor", "decomposition"]));
        &SIGNATURE
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_struct_arguments(self.name(), arg_types, &[2], &[])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let decomposition = parse_tucker_result_field(&args.arg_fields[1], self.name(), 2)?;
        if decomposition.value_type != tensor.value_type || decomposition.shape != tensor.shape {
            return Err(plan_error(
                self.name(),
                "Tucker decomposition must match the tensor input value type and shape",
            ));
        }
        fixed_shape_tensor_field(
            self.name(),
            &tensor.value_type,
            &decomposition.ranks,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match tensor.value_type {
            DataType::Float32 => invoke_tucker_project_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_tucker_project_typed::<Float64Type>(&args, self.name()),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Project a fixed-shape tensor batch into the Tucker core defined by a HOSVD/HOOI \
                 decomposition.",
                "tensor_tucker_project(tensor_batch, decomposition)",
            )
            .with_argument("tensor_batch", "Canonical fixed-shape tensor batch.")
            .with_argument("decomposition", "Struct returned by tensor_hosvd_nd or tensor_hooi_nd.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorTuckerExpand;

impl ScalarUDFImpl for TensorTuckerExpand {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_tucker_expand" }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| named_user_defined_signature(&["decomposition"]));
        &SIGNATURE
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_struct_arguments(self.name(), arg_types, &[1], &[])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_tucker_result_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &contract.shape,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tucker_result_field(&args.arg_fields[0], self.name(), 1)?;
        match contract.value_type {
            DataType::Float32 => invoke_tucker_expand_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_tucker_expand_typed::<Float64Type>(&args, self.name()),
            actual => {
                Err(exec_error(self.name(), format!("unsupported Tucker value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Expand a Tucker decomposition back into the original tensor space.",
                "tensor_tucker_expand(decomposition)",
            )
            .with_argument("decomposition", "Struct returned by tensor_hosvd_nd or tensor_hooi_nd.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorTtSvd {
    signature: Signature,
}

impl TensorTtSvd {
    fn new() -> Self {
        Self { signature: named_user_defined_signature(&["tensor", "max_rank", "tolerance"]) }
    }
}

impl ScalarUDFImpl for TensorTtSvd {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_tt_svd" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[
            (2, ScalarCoercion::Integer),
            (3, ScalarCoercion::Real),
        ])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        tt_struct_field(
            self.name(),
            &tensor.value_type,
            &tensor.shape,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let tensor = parse_tensor_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match tensor.value_type {
            DataType::Float32 => invoke_tt_svd_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_tt_svd_typed::<Float64Type>(&args, self.name()),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute a tensor-train decomposition for each fixed-shape tensor row.",
                "tensor_tt_svd(tensor_batch, max_rank => 16, tolerance => 1e-6)",
            )
            .with_argument("tensor_batch", "Canonical fixed-shape tensor batch.")
            .with_argument("max_rank", "Optional global TT rank cap.")
            .with_argument("tolerance", "Optional truncation tolerance.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

macro_rules! tt_unary_udf {
    (
        $struct_name:ident,
        $udf_name:literal,
        $doc_description:literal,
        $doc_syntax:literal,
        $op:path
    ) => {
        #[derive(Debug, PartialEq, Eq, Hash)]
        struct $struct_name;

        impl ScalarUDFImpl for $struct_name {
            fn as_any(&self) -> &dyn Any { self }

            fn name(&self) -> &'static str { $udf_name }

            fn signature(&self) -> &Signature {
                static SIGNATURE: LazyLock<Signature> =
                    LazyLock::new(|| named_user_defined_signature(&["tt"]));
                &SIGNATURE
            }

            fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
                coerce_struct_arguments(self.name(), arg_types, &[1], &[])
            }

            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                datafusion::common::internal_err!("return_field_from_args should be used instead")
            }

            fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
                let contract = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
                tt_struct_field(
                    self.name(),
                    &contract.value_type,
                    &contract.shape,
                    nullable_or(args.arg_fields),
                )
            }

            fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
                let contract = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
                match contract.value_type {
                    DataType::Float32 => {
                        invoke_tt_unary_typed::<Float32Type>(&args, self.name(), $op)
                    }
                    DataType::Float64 => {
                        invoke_tt_unary_typed::<Float64Type>(&args, self.name(), $op)
                    }
                    actual => Err(exec_error(
                        self.name(),
                        format!("unsupported tensor value type {actual}"),
                    )),
                }
            }

            fn documentation(&self) -> Option<&Documentation> {
                static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
                    tensor_doc($doc_description, $doc_syntax)
                        .with_argument("tt", "Tensor-train struct returned by tensor_tt_svd.")
                        .build()
                });
                Some(&DOCUMENTATION)
            }
        }
    };
}

tt_unary_udf!(
    TensorTtOrthogonalizeLeft,
    "tensor_tt_orthogonalize_left",
    "Left-orthogonalize each tensor-train row.",
    "tensor_tt_orthogonalize_left(tt_result)",
    nabled::arrow::tensor::tt_orthogonalize_left
);

tt_unary_udf!(
    TensorTtOrthogonalizeRight,
    "tensor_tt_orthogonalize_right",
    "Right-orthogonalize each tensor-train row.",
    "tensor_tt_orthogonalize_right(tt_result)",
    nabled::arrow::tensor::tt_orthogonalize_right
);

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorTtRound {
    signature: Signature,
}

impl TensorTtRound {
    fn new() -> Self {
        Self { signature: named_user_defined_signature(&["tt", "max_rank", "tolerance"]) }
    }
}

impl ScalarUDFImpl for TensorTtRound {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_tt_round" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_struct_arguments(self.name(), arg_types, &[1], &[
            (2, ScalarCoercion::Integer),
            (3, ScalarCoercion::Real),
        ])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
        tt_struct_field(
            self.name(),
            &contract.value_type,
            &contract.shape,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
        match contract.value_type {
            DataType::Float32 => invoke_tt_round_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_tt_round_typed::<Float64Type>(&args, self.name()),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Round or compress each tensor-train row.",
                "tensor_tt_round(tt_result, max_rank => 16, tolerance => 1e-6)",
            )
            .with_argument("tt", "Tensor-train struct returned by tensor_tt_svd.")
            .with_argument("max_rank", "Optional global TT rank cap.")
            .with_argument("tolerance", "Optional truncation tolerance.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

macro_rules! tt_binary_struct_udf {
    (
        $struct_name:ident,
        $udf_name:literal,
        $doc_description:literal,
        $doc_syntax:literal,
        $op:path
    ) => {
        #[derive(Debug, PartialEq, Eq, Hash)]
        struct $struct_name;

        impl ScalarUDFImpl for $struct_name {
            fn as_any(&self) -> &dyn Any { self }

            fn name(&self) -> &'static str { $udf_name }

            fn signature(&self) -> &Signature {
                static SIGNATURE: LazyLock<Signature> =
                    LazyLock::new(|| named_user_defined_signature(&["left", "right"]));
                &SIGNATURE
            }

            fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
                coerce_struct_arguments(self.name(), arg_types, &[1, 2], &[])
            }

            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                datafusion::common::internal_err!("return_field_from_args should be used instead")
            }

            fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
                let left = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
                let right = parse_tt_result_field(&args.arg_fields[1], self.name(), 2)?;
                if left.value_type != right.value_type || left.shape != right.shape {
                    return Err(plan_error(
                        self.name(),
                        "tensor-train inputs must share value type and shape",
                    ));
                }
                tt_struct_field(
                    self.name(),
                    &left.value_type,
                    &left.shape,
                    nullable_or(args.arg_fields),
                )
            }

            fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
                let left = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
                match left.value_type {
                    DataType::Float32 => {
                        invoke_tt_binary_typed::<Float32Type>(&args, self.name(), $op)
                    }
                    DataType::Float64 => {
                        invoke_tt_binary_typed::<Float64Type>(&args, self.name(), $op)
                    }
                    actual => Err(exec_error(
                        self.name(),
                        format!("unsupported tensor value type {actual}"),
                    )),
                }
            }

            fn documentation(&self) -> Option<&Documentation> {
                static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
                    tensor_doc($doc_description, $doc_syntax)
                        .with_argument("left", "Left tensor-train struct.")
                        .with_argument("right", "Right tensor-train struct.")
                        .build()
                });
                Some(&DOCUMENTATION)
            }
        }
    };
}

tt_binary_struct_udf!(
    TensorTtAdd,
    "tensor_tt_add",
    "Add two tensor-train batches row-wise.",
    "tensor_tt_add(left_tt, right_tt)",
    nabled::arrow::tensor::tt_add
);

tt_binary_struct_udf!(
    TensorTtHadamard,
    "tensor_tt_hadamard",
    "Compute the row-wise Hadamard product of two tensor-train batches.",
    "tensor_tt_hadamard(left_tt, right_tt)",
    nabled::arrow::tensor::tt_hadamard
);

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorTtHadamardRound {
    signature: Signature,
}

impl TensorTtHadamardRound {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&["left", "right", "max_rank", "tolerance"]),
        }
    }
}

impl ScalarUDFImpl for TensorTtHadamardRound {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_tt_hadamard_round" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_struct_arguments(self.name(), arg_types, &[1, 2], &[
            (3, ScalarCoercion::Integer),
            (4, ScalarCoercion::Real),
        ])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_tt_result_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type || left.shape != right.shape {
            return Err(plan_error(
                self.name(),
                "tensor-train inputs must share value type and shape",
            ));
        }
        tt_struct_field(self.name(), &left.value_type, &left.shape, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
        match left.value_type {
            DataType::Float32 => {
                let config = tt_round_config_at::<f32>(&args, 3, 4, self.name())?;
                invoke_tt_binary_typed::<Float32Type>(&args, self.name(), |left, right| {
                    nabled::arrow::tensor::tt_hadamard_round(left, right, &config)
                })
            }
            DataType::Float64 => {
                let config = tt_round_config_at::<f64>(&args, 3, 4, self.name())?;
                invoke_tt_binary_typed::<Float64Type>(&args, self.name(), |left, right| {
                    nabled::arrow::tensor::tt_hadamard_round(left, right, &config)
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute row-wise Hadamard products followed by TT rounding.",
                "tensor_tt_hadamard_round(left_tt, right_tt, max_rank => 16, tolerance => 1e-6)",
            )
            .with_argument("left", "Left tensor-train struct.")
            .with_argument("right", "Right tensor-train struct.")
            .with_argument("max_rank", "Optional global TT rank cap.")
            .with_argument("tolerance", "Optional truncation tolerance.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

macro_rules! tt_scalar_udf {
    (
        $struct_name:ident,
        $udf_name:literal,
        $doc_description:literal,
        $doc_syntax:literal,
        $op:path
    ) => {
        #[derive(Debug, PartialEq, Eq, Hash)]
        struct $struct_name;

        impl ScalarUDFImpl for $struct_name {
            fn as_any(&self) -> &dyn Any { self }

            fn name(&self) -> &'static str { $udf_name }

            fn signature(&self) -> &Signature {
                static SIGNATURE: LazyLock<Signature> =
                    LazyLock::new(|| named_user_defined_signature(&["tt"]));
                &SIGNATURE
            }

            fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
                coerce_struct_arguments(self.name(), arg_types, &[1], &[])
            }

            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                datafusion::common::internal_err!("return_field_from_args should be used instead")
            }

            fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
                let contract = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
                Ok(scalar_field(self.name(), &contract.value_type, nullable_or(args.arg_fields)))
            }

            fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
                let contract = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
                match contract.value_type {
                    DataType::Float32 => {
                        invoke_tt_scalar_typed::<Float32Type>(&args, self.name(), $op)
                    }
                    DataType::Float64 => {
                        invoke_tt_scalar_typed::<Float64Type>(&args, self.name(), $op)
                    }
                    actual => Err(exec_error(
                        self.name(),
                        format!("unsupported tensor value type {actual}"),
                    )),
                }
            }

            fn documentation(&self) -> Option<&Documentation> {
                static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
                    tensor_doc($doc_description, $doc_syntax)
                        .with_argument("tt", "Tensor-train struct returned by tensor_tt_svd.")
                        .build()
                });
                Some(&DOCUMENTATION)
            }
        }
    };
}

tt_scalar_udf!(
    TensorTtNorm,
    "tensor_tt_norm",
    "Compute the Frobenius norm of each tensor-train row.",
    "tensor_tt_norm(tt_result)",
    nabled::arrow::tensor::tt_norm
);

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorTtInner;

impl ScalarUDFImpl for TensorTtInner {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_tt_inner" }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| named_user_defined_signature(&["left", "right"]));
        &SIGNATURE
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_struct_arguments(self.name(), arg_types, &[1, 2], &[])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let left = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_tt_result_field(&args.arg_fields[1], self.name(), 2)?;
        if left.value_type != right.value_type || left.shape != right.shape {
            return Err(plan_error(
                self.name(),
                "tensor-train inputs must share value type and shape",
            ));
        }
        Ok(scalar_field(self.name(), &left.value_type, nullable_or(args.arg_fields)))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
        match left.value_type {
            DataType::Float32 => invoke_tt_inner_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_tt_inner_typed::<Float64Type>(&args, self.name()),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Compute the row-wise inner product of two tensor-train batches.",
                "tensor_tt_inner(left_tt, right_tt)",
            )
            .with_argument("left", "Left tensor-train struct.")
            .with_argument("right", "Right tensor-train struct.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct TensorTtSvdReconstruct;

impl ScalarUDFImpl for TensorTtSvdReconstruct {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "tensor_tt_svd_reconstruct" }

    fn signature(&self) -> &Signature {
        static SIGNATURE: LazyLock<Signature> =
            LazyLock::new(|| named_user_defined_signature(&["tt"]));
        &SIGNATURE
    }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_struct_arguments(self.name(), arg_types, &[1], &[])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let contract = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &contract.shape,
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let contract = parse_tt_result_field(&args.arg_fields[0], self.name(), 1)?;
        match contract.value_type {
            DataType::Float32 => invoke_tt_reconstruct_typed::<Float32Type>(&args, self.name()),
            DataType::Float64 => invoke_tt_reconstruct_typed::<Float64Type>(&args, self.name()),
            actual => {
                Err(exec_error(self.name(), format!("unsupported tensor value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            tensor_doc(
                "Reconstruct a fixed-shape tensor batch from tensor-train cores.",
                "tensor_tt_svd_reconstruct(tt_result)",
            )
            .with_argument("tt", "Tensor-train struct returned by tensor_tt_svd.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[must_use]
pub fn tensor_cp_als3_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorCpAls3::new()))
}

#[must_use]
pub fn tensor_cp_als3_reconstruct_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorCpAls3Reconstruct))
}

#[must_use]
pub fn tensor_cp_als_nd_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorCpAlsNd::new()))
}

#[must_use]
pub fn tensor_cp_als_nd_reconstruct_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorCpAlsNdReconstruct))
}

#[must_use]
pub fn tensor_hosvd_nd_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorHosvdNd::new()))
}

#[must_use]
pub fn tensor_hooi_nd_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorHooiNd::new()))
}

#[must_use]
pub fn tensor_tucker_project_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTuckerProject))
}

#[must_use]
pub fn tensor_tucker_expand_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTuckerExpand))
}

#[must_use]
pub fn tensor_tt_svd_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTtSvd::new()))
}

#[must_use]
pub fn tensor_tt_orthogonalize_left_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTtOrthogonalizeLeft))
}

#[must_use]
pub fn tensor_tt_orthogonalize_right_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTtOrthogonalizeRight))
}

#[must_use]
pub fn tensor_tt_round_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTtRound::new()))
}

#[must_use]
pub fn tensor_tt_inner_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(TensorTtInner)) }

#[must_use]
pub fn tensor_tt_norm_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(TensorTtNorm)) }

#[must_use]
pub fn tensor_tt_add_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(TensorTtAdd)) }

#[must_use]
pub fn tensor_tt_hadamard_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTtHadamard))
}

#[must_use]
pub fn tensor_tt_hadamard_round_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTtHadamardRound::new()))
}

#[must_use]
pub fn tensor_tt_svd_reconstruct_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(TensorTtSvdReconstruct))
}

#[cfg(test)]
mod tests {
    use datafusion::arrow::array::Array;
    use datafusion::common::ScalarValue;
    use datafusion::common::config::ConfigOptions;

    use super::*;

    fn scalar_args(args: Vec<ColumnarValue>, arg_fields: Vec<FieldRef>) -> ScalarFunctionArgs {
        ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: 1,
            return_field: scalar_field("out", &DataType::Float64, false),
            config_options: Arc::new(ConfigOptions::new()),
        }
    }

    #[test]
    fn tensor_decomposition_config_helpers_validate_values() {
        let args = scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(8))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0e-6))),
            ],
            vec![
                fixed_shape_tensor_field("tensor", &DataType::Float64, &[2, 2, 2], false)
                    .expect("tensor"),
                scalar_field("rank", &DataType::Int64, false),
                scalar_field("max_iterations", &DataType::Int64, false),
                scalar_field("tolerance", &DataType::Float64, false),
            ],
        );
        assert!(cp_config::<f64>(&args, "tensor_cp_als3").is_ok());
        assert!(hooi_config::<f64>(&args, "tensor_hooi_nd").is_ok());

        let invalid = scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(0))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(-1.0))),
            ],
            vec![
                fixed_shape_tensor_field("tensor", &DataType::Float64, &[2, 2, 2], false)
                    .expect("tensor"),
                scalar_field("max_rank", &DataType::Int64, false),
                scalar_field("tolerance", &DataType::Float64, false),
            ],
        );
        assert!(tt_svd_config::<f64>(&invalid, "tensor_tt_svd").is_err());
        assert!(tt_round_config::<f64>(&invalid, "tensor_tt_round").is_err());

        let defaults = scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
            ],
            vec![
                fixed_shape_tensor_field("tensor", &DataType::Float64, &[2, 2, 2], false)
                    .expect("tensor"),
                scalar_field("max_rank", &DataType::Int64, false),
            ],
        );
        assert!(tt_svd_config::<f64>(&defaults, "tensor_tt_svd").is_ok());
        assert!(cp_config::<f32>(&args, "tensor_cp_als3").is_ok());
    }

    #[test]
    fn tensor_decomposition_field_builders_and_parsers_cover_contracts() {
        let cp_field =
            cp_struct_field("cp", &DataType::Float64, &[2, 3], 2, false).expect("cp field");
        let cp_contract =
            parse_cp_result_field(&cp_field, "tensor_cp_als3_reconstruct", 1).expect("cp contract");
        assert_eq!(cp_contract.shape, vec![2, 3]);

        let tucker_field =
            tucker_struct_field("tucker", &DataType::Float64, &[2, 3], &[1, 2], false)
                .expect("tucker field");
        let tucker_contract = parse_tucker_result_field(&tucker_field, "tensor_tucker_expand", 1)
            .expect("tucker contract");
        assert_eq!(tucker_contract.shape, vec![2, 3]);
        assert_eq!(tucker_contract.ranks, vec![1, 2]);

        let tt_field =
            tt_struct_field("tt", &DataType::Float64, &[2, 3, 4], false).expect("tt field");
        let tt_contract =
            parse_tt_result_field(&tt_field, "tensor_tt_norm", 1).expect("tt contract");
        assert_eq!(tt_contract.shape, vec![2, 3, 4]);
        assert_eq!(metadata_dim("tensor_tt_svd", 4).expect("dim"), 4);

        let wrong_tucker = struct_field(
            "tucker",
            vec![
                fixed_shape_tensor_field("core", &DataType::Float64, &[1, 2], false)
                    .expect("core")
                    .as_ref()
                    .clone(),
                fixed_shape_tensor_field("factor_0", &DataType::Float64, &[2, 3], false)
                    .expect("factor")
                    .as_ref()
                    .clone(),
            ],
            false,
        );
        assert!(parse_tucker_result_field(&wrong_tucker, "tensor_tucker_expand", 1).is_err());

        let tt_empty = struct_field("tt", vec![], false);
        assert!(parse_tt_result_field(&tt_empty, "tensor_tt_norm", 1).is_err());
    }

    #[test]
    fn tensor_decomposition_storage_helpers_cover_empty_and_error_paths() {
        let empty = empty_variable_tensor_storage(&DataType::Float64, 3).expect("empty storage");
        assert_eq!(empty.len(), 0);
        assert_eq!(empty.num_columns(), 2);
        assert!(empty_variable_tensor_storage(&DataType::Utf8, 1).is_err());

        let invalid_cp = struct_field(
            "cp",
            vec![
                vector_field("weights", &DataType::Float64, 2, false)
                    .expect("weights")
                    .as_ref()
                    .clone(),
                vector_field("factor_0", &DataType::Float64, 2, false)
                    .expect("factor")
                    .as_ref()
                    .clone(),
            ],
            false,
        );
        assert!(parse_cp_result_field(&invalid_cp, "tensor_cp_als3_reconstruct", 1).is_err());
        assert!(metadata_dim("tensor_tt_svd", usize::MAX).is_err());
    }
}
