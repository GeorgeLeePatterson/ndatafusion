use std::any::Any;
use std::mem::size_of;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Array1, Axis};
use ndarrow::NdarrowElement;

use super::common::{
    expect_fixed_size_list_arg, expect_real_scalar_arg, expect_string_scalar_arg,
    expect_string_scalar_argument, expect_usize_scalar_arg,
    fixed_shape_tensor_array_from_flat_rows, fixed_size_list_array_from_flat_rows,
    fixed_size_list_view2, nullable_or,
};
use super::docs::ml_doc;
use crate::error::{exec_error, plan_error};
use crate::metadata::{fixed_shape_tensor_field, parse_vector_field, vector_field};
use crate::signatures::{ScalarCoercion, coerce_scalar_arguments, named_user_defined_signature};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NamedVectorMap {
    Identity,
    Square,
    Sigmoid,
    Softmax,
}

impl NamedVectorMap {
    fn parse(function_name: &str, value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "identity" => Ok(Self::Identity),
            "square" => Ok(Self::Square),
            "sigmoid" => Ok(Self::Sigmoid),
            "softmax" => Ok(Self::Softmax),
            actual => Err(plan_error(
                function_name,
                format!(
                    "unsupported named vector function {actual}; supported values are identity, \
                     square, sigmoid, softmax",
                ),
            )),
        }
    }

    fn apply<T: NabledReal>(self, input: &Array1<T>) -> Result<Array1<T>> {
        match self {
            Self::Identity => Ok(input.clone()),
            Self::Square => Ok(input.mapv(|value| value * value)),
            Self::Sigmoid => Ok(input.mapv(|value| T::one() / (T::one() + (-value).exp()))),
            Self::Softmax => {
                if input.is_empty() {
                    return Err(exec_error("softmax", "input cannot be empty"));
                }
                let max_value = input
                    .iter()
                    .copied()
                    .reduce(|left, right| if left >= right { left } else { right })
                    .ok_or_else(|| exec_error("softmax", "input cannot be empty"))?;
                let exponentiated = input.mapv(|value| (value - max_value).exp());
                let sum = exponentiated.iter().copied().fold(T::zero(), |acc, value| acc + value);
                if !sum.is_finite() || sum <= T::zero() {
                    return Err(exec_error("softmax", "softmax normalization failed"));
                }
                Ok(exponentiated / sum)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NamedScalarObjective {
    SumSquares,
    Rosenbrock,
}

impl NamedScalarObjective {
    fn parse(function_name: &str, value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "sum_squares" => Ok(Self::SumSquares),
            "rosenbrock" => Ok(Self::Rosenbrock),
            actual => Err(plan_error(
                function_name,
                format!(
                    "unsupported named scalar objective {actual}; supported values are \
                     sum_squares, rosenbrock",
                ),
            )),
        }
    }

    fn apply<T: NabledReal>(self, input: &Array1<T>) -> Result<T> {
        match self {
            Self::SumSquares => Ok(input
                .iter()
                .copied()
                .fold(T::zero(), |accumulator, value| accumulator + (value * value))),
            Self::Rosenbrock => {
                if input.len() < 2 {
                    return Err(exec_error(
                        "rosenbrock",
                        "rosenbrock requires vectors with length >= 2",
                    ));
                }
                let hundred = T::from_f64(100.0).unwrap_or(T::one() + T::one());
                let one = T::one();
                let mut total = T::zero();
                for index in 0..(input.len() - 1) {
                    let xi = input[index];
                    let x_next = input[index + 1];
                    total += hundred * (x_next - (xi * xi)).powi(2) + (one - xi).powi(2);
                }
                Ok(total)
            }
        }
    }
}

fn default_jacobian_config<T: NabledReal>() -> nabled::ml::jacobian::JacobianConfig<T> {
    let (step_size, tolerance) =
        if size_of::<T>() == size_of::<f32>() { (1.0e-4, 1.0e-5) } else { (1.0e-6, 1.0e-8) };
    nabled::ml::jacobian::JacobianConfig {
        step_size:      T::from_f64(step_size).unwrap_or(T::epsilon()),
        tolerance:      T::from_f64(tolerance).unwrap_or(T::epsilon()),
        max_iterations: 100,
    }
}

fn parse_named_vector_map(function_name: &str, value: &str) -> Result<NamedVectorMap> {
    NamedVectorMap::parse(function_name, value)
}

fn parse_named_scalar_objective(function_name: &str, value: &str) -> Result<NamedScalarObjective> {
    NamedScalarObjective::parse(function_name, value)
}

fn validate_argument_count(function_name: &str, count: usize) -> Result<()> {
    if !(2..=5).contains(&count) {
        return Err(plan_error(
            function_name,
            format!(
                "{function_name} requires function name, vector batch, and up to three trailing \
                 scalar config arguments; found {count} arguments",
            ),
        ));
    }
    Ok(())
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

fn typed_jacobian_config<T: NabledReal>(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::ml::jacobian::JacobianConfig<T>> {
    let defaults = default_jacobian_config::<T>();
    let step_size = optional_real_scalar_arg(args, 3, function_name)?
        .map(|value| {
            T::from_f64(value)
                .ok_or_else(|| exec_error(function_name, "step_size could not be represented"))
        })
        .transpose()?
        .unwrap_or(defaults.step_size);
    let tolerance = optional_real_scalar_arg(args, 4, function_name)?
        .map(|value| {
            T::from_f64(value)
                .ok_or_else(|| exec_error(function_name, "tolerance could not be represented"))
        })
        .transpose()?
        .unwrap_or(defaults.tolerance);
    let max_iterations =
        optional_usize_scalar_arg(args, 5, function_name)?.unwrap_or(defaults.max_iterations);
    nabled::ml::jacobian::JacobianConfig::new(step_size, tolerance, max_iterations)
        .map_err(|error| exec_error(function_name, error))
}

fn invoke_jacobian_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    named_function: NamedVectorMap,
    use_central: bool,
    len: usize,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let vectors = expect_fixed_size_list_arg(args, 2, function_name)?;
    let vector_view = fixed_size_list_view2::<T>(vectors, function_name)?;
    let config = typed_jacobian_config::<T::Native>(args, function_name)?;
    let batch = vector_view.len_of(Axis(0));
    let mut output = Vec::with_capacity(batch * len * len);
    for row in 0..batch {
        let vector = vector_view.index_axis(Axis(0), row);
        let jacobian = if use_central {
            nabled::ml::jacobian::numerical_jacobian_central(
                &|input: &Array1<T::Native>| {
                    named_function.apply(input).map_err(|error| {
                        nabled::ml::jacobian::JacobianError::FunctionError(error.to_string())
                    })
                },
                &vector,
                &config,
            )
        } else {
            nabled::ml::jacobian::numerical_jacobian(
                &|input: &Array1<T::Native>| {
                    named_function.apply(input).map_err(|error| {
                        nabled::ml::jacobian::JacobianError::FunctionError(error.to_string())
                    })
                },
                &vector,
                &config,
            )
        }
        .map_err(|error| exec_error(function_name, error))?;
        output.extend(jacobian.iter().copied());
    }
    let (_field, output) =
        fixed_shape_tensor_array_from_flat_rows::<T>(function_name, batch, &[len, len], output)?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_gradient_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    objective: NamedScalarObjective,
    len: usize,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let vectors = expect_fixed_size_list_arg(args, 2, function_name)?;
    let vector_view = fixed_size_list_view2::<T>(vectors, function_name)?;
    let config = typed_jacobian_config::<T::Native>(args, function_name)?;
    let batch = vector_view.len_of(Axis(0));
    let mut output = Vec::with_capacity(batch * len);
    for row in 0..batch {
        let vector = vector_view.index_axis(Axis(0), row);
        let gradient = nabled::ml::jacobian::numerical_gradient(
            &|input: &Array1<T::Native>| {
                objective.apply(input).map_err(|error| {
                    nabled::ml::jacobian::JacobianError::FunctionError(error.to_string())
                })
            },
            &vector,
            &config,
        )
        .map_err(|error| exec_error(function_name, error))?;
        output.extend(gradient.iter().copied());
    }
    let output = fixed_size_list_array_from_flat_rows::<T>(function_name, batch, len, &output)?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_hessian_typed<T>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    objective: NamedScalarObjective,
    len: usize,
) -> Result<ColumnarValue>
where
    T: datafusion::arrow::array::types::ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
{
    let vectors = expect_fixed_size_list_arg(args, 2, function_name)?;
    let vector_view = fixed_size_list_view2::<T>(vectors, function_name)?;
    let config = typed_jacobian_config::<T::Native>(args, function_name)?;
    let batch = vector_view.len_of(Axis(0));
    let mut output = Vec::with_capacity(batch * len * len);
    for row in 0..batch {
        let vector = vector_view.index_axis(Axis(0), row);
        let hessian = nabled::ml::jacobian::numerical_hessian(
            &|input: &Array1<T::Native>| {
                objective.apply(input).map_err(|error| {
                    nabled::ml::jacobian::JacobianError::FunctionError(error.to_string())
                })
            },
            &vector,
            &config,
        )
        .map_err(|error| exec_error(function_name, error))?;
        output.extend(hessian.iter().copied());
    }
    let (_field, output) =
        fixed_shape_tensor_array_from_flat_rows::<T>(function_name, batch, &[len, len], output)?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct JacobianUdf {
    signature: Signature,
}

impl JacobianUdf {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&[
                "function",
                "vector",
                "step_size",
                "tolerance",
                "max_iterations",
            ]),
        }
    }
}

impl ScalarUDFImpl for JacobianUdf {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "jacobian" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        validate_argument_count(self.name(), arg_types.len())?;
        let mut scalars = Vec::new();
        if arg_types.len() >= 3 {
            scalars.push((3, ScalarCoercion::Real));
        }
        if arg_types.len() >= 4 {
            scalars.push((4, ScalarCoercion::Real));
        }
        if arg_types.len() >= 5 {
            scalars.push((5, ScalarCoercion::Integer));
        }
        coerce_scalar_arguments(self.name(), arg_types, &scalars)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_argument_count(self.name(), args.arg_fields.len())?;
        let named_function = expect_string_scalar_argument(&args, 1, self.name())?;
        let _named_function = parse_named_vector_map(self.name(), &named_function)?;
        let contract = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &[contract.len, contract.len],
            nullable_or(&args.arg_fields[..2]),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        validate_argument_count(self.name(), args.arg_fields.len())?;
        let named_function =
            parse_named_vector_map(self.name(), &expect_string_scalar_arg(&args, 1, self.name())?)?;
        let contract = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        match contract.value_type {
            DataType::Float32 => invoke_jacobian_typed::<Float32Type>(
                &args,
                self.name(),
                named_function,
                false,
                contract.len,
            ),
            DataType::Float64 => invoke_jacobian_typed::<Float64Type>(
                &args,
                self.name(),
                named_function,
                false,
                contract.len,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Compute a numerical Jacobian over each dense vector row using a built-in named \
                 vector function.",
                "jacobian('softmax', vector_batch, step_size => 1e-6)",
            )
            .with_argument(
                "function",
                "Built-in vector function name: identity, square, sigmoid, or softmax.",
            )
            .with_argument(
                "vector_batch",
                "Canonical dense vector batch in FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_argument("step_size", "Optional finite-difference step size.")
            .with_argument("tolerance", "Optional derivative-config tolerance.")
            .with_argument("max_iterations", "Optional derivative-config iteration budget.")
            .with_alternative_syntax(
                "numerical_jacobian('softmax', vector_batch, step_size => 1e-6)",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct JacobianCentralUdf {
    signature: Signature,
}

impl JacobianCentralUdf {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&[
                "function",
                "vector",
                "step_size",
                "tolerance",
                "max_iterations",
            ]),
        }
    }
}

impl ScalarUDFImpl for JacobianCentralUdf {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "jacobian_central" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        JacobianUdf::new().coerce_types(arg_types)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_argument_count(self.name(), args.arg_fields.len())?;
        let named_function = expect_string_scalar_argument(&args, 1, self.name())?;
        let _named_function = parse_named_vector_map(self.name(), &named_function)?;
        let contract = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &[contract.len, contract.len],
            nullable_or(&args.arg_fields[..2]),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        validate_argument_count(self.name(), args.arg_fields.len())?;
        let named_function =
            parse_named_vector_map(self.name(), &expect_string_scalar_arg(&args, 1, self.name())?)?;
        let contract = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        match contract.value_type {
            DataType::Float32 => invoke_jacobian_typed::<Float32Type>(
                &args,
                self.name(),
                named_function,
                true,
                contract.len,
            ),
            DataType::Float64 => invoke_jacobian_typed::<Float64Type>(
                &args,
                self.name(),
                named_function,
                true,
                contract.len,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Compute a central-difference numerical Jacobian over each dense vector row using \
                 a built-in named vector function.",
                "jacobian_central('softmax', vector_batch, step_size => 1e-6)",
            )
            .with_argument(
                "function",
                "Built-in vector function name: identity, square, sigmoid, or softmax.",
            )
            .with_argument(
                "vector_batch",
                "Canonical dense vector batch in FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_argument("step_size", "Optional finite-difference step size.")
            .with_argument("tolerance", "Optional derivative-config tolerance.")
            .with_argument("max_iterations", "Optional derivative-config iteration budget.")
            .with_alternative_syntax(
                "numerical_jacobian_central('softmax', vector_batch, step_size => 1e-6)",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct GradientUdf {
    signature: Signature,
}

impl GradientUdf {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&[
                "function",
                "vector",
                "step_size",
                "tolerance",
                "max_iterations",
            ]),
        }
    }
}

impl ScalarUDFImpl for GradientUdf {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "gradient" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        JacobianUdf::new().coerce_types(arg_types)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_argument_count(self.name(), args.arg_fields.len())?;
        let objective = expect_string_scalar_argument(&args, 1, self.name())?;
        let _objective = parse_named_scalar_objective(self.name(), &objective)?;
        let contract = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        vector_field(
            self.name(),
            &contract.value_type,
            contract.len,
            nullable_or(&args.arg_fields[..2]),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        validate_argument_count(self.name(), args.arg_fields.len())?;
        let objective = parse_named_scalar_objective(
            self.name(),
            &expect_string_scalar_arg(&args, 1, self.name())?,
        )?;
        let contract = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        match contract.value_type {
            DataType::Float32 => {
                invoke_gradient_typed::<Float32Type>(&args, self.name(), objective, contract.len)
            }
            DataType::Float64 => {
                invoke_gradient_typed::<Float64Type>(&args, self.name(), objective, contract.len)
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Compute a numerical gradient over each dense vector row using a built-in scalar \
                 objective.",
                "gradient('rosenbrock', vector_batch, step_size => 1e-6)",
            )
            .with_argument("function", "Built-in scalar objective name: sum_squares or rosenbrock.")
            .with_argument(
                "vector_batch",
                "Canonical dense vector batch in FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_argument("step_size", "Optional finite-difference step size.")
            .with_argument("tolerance", "Optional derivative-config tolerance.")
            .with_argument("max_iterations", "Optional derivative-config iteration budget.")
            .with_alternative_syntax(
                "numerical_gradient('rosenbrock', vector_batch, step_size => 1e-6)",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct HessianUdf {
    signature: Signature,
}

impl HessianUdf {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&[
                "function",
                "vector",
                "step_size",
                "tolerance",
                "max_iterations",
            ]),
        }
    }
}

impl ScalarUDFImpl for HessianUdf {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "hessian" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        JacobianUdf::new().coerce_types(arg_types)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_argument_count(self.name(), args.arg_fields.len())?;
        let objective = expect_string_scalar_argument(&args, 1, self.name())?;
        let _objective = parse_named_scalar_objective(self.name(), &objective)?;
        let contract = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        fixed_shape_tensor_field(
            self.name(),
            &contract.value_type,
            &[contract.len, contract.len],
            nullable_or(&args.arg_fields[..2]),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        validate_argument_count(self.name(), args.arg_fields.len())?;
        let objective = parse_named_scalar_objective(
            self.name(),
            &expect_string_scalar_arg(&args, 1, self.name())?,
        )?;
        let contract = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        match contract.value_type {
            DataType::Float32 => {
                invoke_hessian_typed::<Float32Type>(&args, self.name(), objective, contract.len)
            }
            DataType::Float64 => {
                invoke_hessian_typed::<Float64Type>(&args, self.name(), objective, contract.len)
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported vector value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Compute a numerical Hessian over each dense vector row using a built-in scalar \
                 objective.",
                "hessian('rosenbrock', vector_batch, step_size => 1e-6)",
            )
            .with_argument("function", "Built-in scalar objective name: sum_squares or rosenbrock.")
            .with_argument(
                "vector_batch",
                "Canonical dense vector batch in FixedSizeList<Float32|Float64>(D) form.",
            )
            .with_argument("step_size", "Optional finite-difference step size.")
            .with_argument("tolerance", "Optional derivative-config tolerance.")
            .with_argument("max_iterations", "Optional derivative-config iteration budget.")
            .with_alternative_syntax(
                "numerical_hessian('rosenbrock', vector_batch, step_size => 1e-6)",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[must_use]
pub fn jacobian_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(JacobianUdf::new()).with_aliases(["numerical_jacobian"]))
}

#[must_use]
pub fn jacobian_central_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(JacobianCentralUdf::new())
            .with_aliases(["numerical_jacobian_central"]),
    )
}

#[must_use]
pub fn gradient_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(GradientUdf::new()).with_aliases(["numerical_gradient"]))
}

#[must_use]
pub fn hessian_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(HessianUdf::new()).with_aliases(["numerical_hessian"]))
}

#[cfg(test)]
mod tests {
    use datafusion::common::ScalarValue;
    use datafusion::common::config::ConfigOptions;
    use ndarray::array;

    use super::*;

    fn scalar_args(args: Vec<ColumnarValue>, arg_fields: Vec<FieldRef>) -> ScalarFunctionArgs {
        ScalarFunctionArgs {
            args,
            arg_fields,
            number_rows: 1,
            return_field: Arc::new(datafusion::arrow::datatypes::Field::new(
                "out",
                DataType::Float64,
                false,
            )),
            config_options: Arc::new(ConfigOptions::new()),
        }
    }

    #[test]
    fn named_function_parsers_and_validation_cover_supported_values() {
        assert_eq!(
            parse_named_vector_map("jacobian", "softmax").expect("softmax"),
            NamedVectorMap::Softmax
        );
        assert_eq!(
            parse_named_scalar_objective("gradient", "rosenbrock").expect("rosenbrock"),
            NamedScalarObjective::Rosenbrock
        );
        assert!(parse_named_vector_map("jacobian", "missing").is_err());
        assert!(parse_named_scalar_objective("gradient", "missing").is_err());
        assert!(validate_argument_count("jacobian", 1).is_err());
        assert!(validate_argument_count("jacobian", 6).is_err());
        assert!(validate_argument_count("jacobian", 3).is_ok());
    }

    #[test]
    fn named_function_apply_helpers_cover_expected_outputs() {
        let input = array![1.0_f64, 2.0];
        assert_eq!(NamedVectorMap::Identity.apply(&input).expect("identity"), input);
        assert_eq!(NamedVectorMap::Square.apply(&input).expect("square"), array![1.0_f64, 4.0]);
        let sigmoid = NamedVectorMap::Sigmoid.apply(&input).expect("sigmoid");
        assert!(sigmoid[0] > 0.73 && sigmoid[0] < 0.74);
        let softmax = NamedVectorMap::Softmax.apply(&input).expect("softmax");
        assert!((softmax.sum() - 1.0).abs() < 1.0e-9);
        assert!(softmax[1] > softmax[0]);
        let sum_squares = NamedScalarObjective::SumSquares.apply(&input).expect("sum squares");
        assert!((sum_squares - 5.0).abs() < 1.0e-12);
        let rosenbrock =
            NamedScalarObjective::Rosenbrock.apply(&array![1.0_f64, 1.0]).expect("rosenbrock");
        assert!(rosenbrock.abs() < 1.0e-12);
    }

    #[test]
    fn named_function_apply_helpers_cover_error_paths_and_float32_execution() {
        let empty = Array1::<f64>::zeros(0);
        assert!(NamedVectorMap::Softmax.apply(&empty).is_err());
        assert!(NamedScalarObjective::Rosenbrock.apply(&array![1.0_f64]).is_err());

        let input_f32 = array![1.0_f32, 2.0];
        let sigmoid = NamedVectorMap::Sigmoid.apply(&input_f32).expect("sigmoid f32");
        assert!(sigmoid[0] > 0.73 && sigmoid[0] < 0.74);
        let softmax = NamedVectorMap::Softmax.apply(&input_f32).expect("softmax f32");
        assert!((softmax.sum() - 1.0).abs() < 1.0e-5);
    }

    #[test]
    fn jacobian_config_helpers_cover_defaults_and_invalid_values() {
        let vector = vector_field("vector", &DataType::Float64, 2, false).expect("vector field");
        let args = scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("square".to_string()))),
                ColumnarValue::Array(Arc::new(
                    datafusion::arrow::array::FixedSizeListArray::from_iter_primitive::<
                        Float64Type,
                        _,
                        _,
                    >(vec![Some(vec![Some(1.0_f64), Some(2.0)])], 2),
                )),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0e-5))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0e-7))),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(8))),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                vector,
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "step_size",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "tolerance",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "max_iterations",
                    DataType::Int64,
                    false,
                )),
            ],
        );
        let config = typed_jacobian_config::<f64>(&args, "jacobian").expect("config");
        assert_eq!(config.max_iterations, 8);
        assert!((config.step_size - 1.0e-5).abs() < 1.0e-12);
        assert!((config.tolerance - 1.0e-7).abs() < 1.0e-12);

        let invalid = scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("square".to_string()))),
                ColumnarValue::Array(Arc::new(
                    datafusion::arrow::array::FixedSizeListArray::from_iter_primitive::<
                        Float64Type,
                        _,
                        _,
                    >(vec![Some(vec![Some(1.0_f64), Some(2.0)])], 2),
                )),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(-1.0))),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                vector_field("vector", &DataType::Float64, 2, false).expect("vector field"),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "step_size",
                    DataType::Float64,
                    false,
                )),
            ],
        );
        assert!(typed_jacobian_config::<f64>(&invalid, "jacobian").is_err());
    }

    #[test]
    fn jacobian_helpers_cover_default_and_float32_paths() {
        let vector_f32 =
            vector_field("vector", &DataType::Float32, 2, false).expect("vector field");
        let args = scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("square".to_string()))),
                ColumnarValue::Array(Arc::new(
                    datafusion::arrow::array::FixedSizeListArray::from_iter_primitive::<
                        Float32Type,
                        _,
                        _,
                    >(vec![Some(vec![Some(1.0_f32), Some(2.0)])], 2),
                )),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                Arc::clone(&vector_f32),
            ],
        );
        let defaults = typed_jacobian_config::<f32>(&args, "jacobian").expect("defaults");
        assert_eq!(defaults.max_iterations, default_jacobian_config::<f32>().max_iterations);
        let jacobian = invoke_jacobian_typed::<Float32Type>(
            &args,
            "jacobian",
            NamedVectorMap::Square,
            false,
            2,
        )
        .expect("jacobian f32");
        let gradient = invoke_gradient_typed::<Float32Type>(
            &args,
            "gradient",
            NamedScalarObjective::SumSquares,
            2,
        )
        .expect("gradient f32");
        let hessian = invoke_hessian_typed::<Float32Type>(
            &args,
            "hessian",
            NamedScalarObjective::SumSquares,
            2,
        )
        .expect("hessian f32");
        assert!(matches!(jacobian, ColumnarValue::Array(_)));
        assert!(matches!(gradient, ColumnarValue::Array(_)));
        assert!(matches!(hessian, ColumnarValue::Array(_)));
    }
}
