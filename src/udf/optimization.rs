use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use ndarray::{Array1, Axis};
use num_complex::Complex64;

use super::common::{
    complex_fixed_size_list_array_from_flat_rows, complex_fixed_size_list_view2,
    expect_fixed_size_list_arg, expect_real_scalar_arg, expect_string_scalar_arg, nullable_or,
    primitive_array_from_values,
};
use super::docs::ml_doc;
use crate::error::{exec_error, plan_error};
use crate::metadata::{complex_vector_field, parse_complex_vector_field, scalar_field};
use crate::signatures::{ScalarCoercion, coerce_scalar_arguments, named_user_defined_signature};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum NamedComplexObjective {
    NormSquared,
    QuarticNorm,
}

impl NamedComplexObjective {
    fn parse(function_name: &str, value: &str) -> Result<Self> {
        match value.to_ascii_lowercase().as_str() {
            "norm_squared" => Ok(Self::NormSquared),
            "quartic_norm" => Ok(Self::QuarticNorm),
            actual => Err(plan_error(
                function_name,
                format!(
                    "unsupported complex objective {actual}; supported values are norm_squared, \
                     quartic_norm",
                ),
            )),
        }
    }

    fn objective(self, input: &Array1<Complex64>) -> f64 {
        match self {
            Self::NormSquared => input.iter().map(Complex64::norm_sqr).sum(),
            Self::QuarticNorm => input.iter().map(|value| value.norm_sqr().powi(2)).sum(),
        }
    }

    fn gradient(self, input: &Array1<Complex64>) -> Array1<Complex64> {
        match self {
            Self::NormSquared => input.mapv(|value| value * 2.0),
            Self::QuarticNorm => input.mapv(|value| value * (4.0 * value.norm_sqr())),
        }
    }
}

fn validate_arg_count(function_name: &str, count: usize, min: usize, max: usize) -> Result<()> {
    if !(min..=max).contains(&count) {
        return Err(plan_error(
            function_name,
            format!("{function_name} requires between {min} and {max} arguments, found {count}"),
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
        super::common::expect_usize_scalar_arg(args, position, function_name).map(Some)
    }
}

fn line_search_config(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::ml::optimization::LineSearchConfig<f64>> {
    let defaults = nabled::ml::optimization::LineSearchConfig::<f64>::default();
    let initial_step =
        optional_real_scalar_arg(args, 4, function_name)?.unwrap_or(defaults.initial_step);
    let contraction =
        optional_real_scalar_arg(args, 5, function_name)?.unwrap_or(defaults.contraction);
    let sufficient_decrease =
        optional_real_scalar_arg(args, 6, function_name)?.unwrap_or(defaults.sufficient_decrease);
    let max_iterations =
        optional_usize_scalar_arg(args, 7, function_name)?.unwrap_or(defaults.max_iterations);
    let config = nabled::ml::optimization::LineSearchConfig {
        initial_step,
        contraction,
        sufficient_decrease,
        max_iterations,
    };
    if config.initial_step <= 0.0
        || config.contraction <= 0.0
        || config.contraction >= 1.0
        || config.sufficient_decrease <= 0.0
        || config.sufficient_decrease >= 1.0
        || config.max_iterations == 0
    {
        return Err(exec_error(function_name, "invalid line search configuration"));
    }
    Ok(config)
}

fn sgd_config(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::ml::optimization::SGDConfig<f64>> {
    let defaults = nabled::ml::optimization::SGDConfig::<f64>::default();
    let learning_rate =
        optional_real_scalar_arg(args, 3, function_name)?.unwrap_or(defaults.learning_rate);
    let max_iterations =
        optional_usize_scalar_arg(args, 4, function_name)?.unwrap_or(defaults.max_iterations);
    let tolerance = optional_real_scalar_arg(args, 5, function_name)?.unwrap_or(defaults.tolerance);
    let config = nabled::ml::optimization::SGDConfig { learning_rate, max_iterations, tolerance };
    if config.learning_rate <= 0.0 || config.max_iterations == 0 || config.tolerance < 0.0 {
        return Err(exec_error(function_name, "invalid gradient-descent configuration"));
    }
    Ok(config)
}

fn adam_config(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::ml::optimization::AdamConfig<f64>> {
    let defaults = nabled::ml::optimization::AdamConfig::<f64>::default();
    let learning_rate =
        optional_real_scalar_arg(args, 3, function_name)?.unwrap_or(defaults.learning_rate);
    let beta1 = optional_real_scalar_arg(args, 4, function_name)?.unwrap_or(defaults.beta1);
    let beta2 = optional_real_scalar_arg(args, 5, function_name)?.unwrap_or(defaults.beta2);
    let epsilon = optional_real_scalar_arg(args, 6, function_name)?.unwrap_or(defaults.epsilon);
    let max_iterations =
        optional_usize_scalar_arg(args, 7, function_name)?.unwrap_or(defaults.max_iterations);
    let tolerance = optional_real_scalar_arg(args, 8, function_name)?.unwrap_or(defaults.tolerance);
    let config = nabled::ml::optimization::AdamConfig {
        learning_rate,
        beta1,
        beta2,
        epsilon,
        max_iterations,
        tolerance,
    };
    if config.learning_rate <= 0.0
        || config.beta1 <= 0.0
        || config.beta1 >= 1.0
        || config.beta2 <= 0.0
        || config.beta2 >= 1.0
        || config.epsilon <= 0.0
        || config.max_iterations == 0
        || config.tolerance < 0.0
    {
        return Err(exec_error(function_name, "invalid Adam configuration"));
    }
    Ok(config)
}

fn momentum_config(
    args: &ScalarFunctionArgs,
    function_name: &str,
) -> Result<nabled::ml::optimization::MomentumConfig<f64>> {
    let defaults = nabled::ml::optimization::MomentumConfig::<f64>::default();
    let learning_rate =
        optional_real_scalar_arg(args, 3, function_name)?.unwrap_or(defaults.learning_rate);
    let momentum = optional_real_scalar_arg(args, 4, function_name)?.unwrap_or(defaults.momentum);
    let max_iterations =
        optional_usize_scalar_arg(args, 5, function_name)?.unwrap_or(defaults.max_iterations);
    let tolerance = optional_real_scalar_arg(args, 6, function_name)?.unwrap_or(defaults.tolerance);
    let config = nabled::ml::optimization::MomentumConfig {
        learning_rate,
        momentum,
        max_iterations,
        tolerance,
    };
    if config.learning_rate <= 0.0
        || config.momentum < 0.0
        || config.momentum >= 1.0
        || config.max_iterations == 0
        || config.tolerance < 0.0
    {
        return Err(exec_error(function_name, "invalid momentum configuration"));
    }
    Ok(config)
}

fn invoke_line_search(args: &ScalarFunctionArgs, function_name: &str) -> Result<ColumnarValue> {
    let objective = NamedComplexObjective::parse(
        function_name,
        &expect_string_scalar_arg(args, 1, function_name)?,
    )?;
    let (_point_item, point_contract) =
        parse_complex_vector_field(&args.arg_fields[1], function_name, 2)?;
    let (_direction_item, direction_contract) =
        parse_complex_vector_field(&args.arg_fields[2], function_name, 3)?;
    if point_contract.len != direction_contract.len {
        return Err(exec_error(
            function_name,
            format!(
                "complex vector width mismatch: point {}, direction {}",
                point_contract.len, direction_contract.len
            ),
        ));
    }
    let point = expect_fixed_size_list_arg(args, 2, function_name)?;
    let direction = expect_fixed_size_list_arg(args, 3, function_name)?;
    let point_view = complex_fixed_size_list_view2(point, function_name)?;
    let direction_view = complex_fixed_size_list_view2(direction, function_name)?;
    if point_view.len_of(Axis(0)) != direction_view.len_of(Axis(0)) {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} points vs {} directions",
                point_view.len_of(Axis(0)),
                direction_view.len_of(Axis(0))
            ),
        ));
    }
    let config = line_search_config(args, function_name)?;
    let mut outputs = Vec::with_capacity(point_view.len_of(Axis(0)));
    for row in 0..point_view.len_of(Axis(0)) {
        let point_row = point_view.index_axis(Axis(0), row).to_owned();
        let direction_row = direction_view.index_axis(Axis(0), row).to_owned();
        let alpha = nabled::ml::optimization::backtracking_line_search_complex(
            &point_row,
            &direction_row,
            |input| objective.objective(input),
            |input| objective.gradient(input),
            &config,
        )
        .map_err(|error| exec_error(function_name, error))?;
        outputs.push(alpha);
    }
    Ok(ColumnarValue::Array(Arc::new(primitive_array_from_values::<Float64Type>(outputs))))
}

fn invoke_complex_optimizer(
    args: &ScalarFunctionArgs,
    function_name: &str,
    run: impl Fn(
        &Array1<Complex64>,
        &NamedComplexObjective,
        &ScalarFunctionArgs,
        &str,
    ) -> Result<Array1<Complex64>>,
) -> Result<ColumnarValue> {
    let objective = NamedComplexObjective::parse(
        function_name,
        &expect_string_scalar_arg(args, 1, function_name)?,
    )?;
    let (_item, contract) = parse_complex_vector_field(&args.arg_fields[1], function_name, 2)?;
    let initial = expect_fixed_size_list_arg(args, 2, function_name)?;
    let initial_view = complex_fixed_size_list_view2(initial, function_name)?;
    let batch = initial_view.len_of(Axis(0));
    let mut outputs = Vec::with_capacity(batch * contract.len);
    for row in 0..batch {
        let initial_row = initial_view.index_axis(Axis(0), row).to_owned();
        let optimized = run(&initial_row, &objective, args, function_name)?;
        outputs.extend(optimized.iter().copied());
    }
    let output =
        complex_fixed_size_list_array_from_flat_rows(function_name, batch, contract.len, outputs)?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct BacktrackingLineSearchComplex {
    signature: Signature,
}

impl BacktrackingLineSearchComplex {
    fn new() -> Self {
        Self {
            signature: named_user_defined_signature(&[
                "function",
                "point",
                "direction",
                "initial_step",
                "contraction",
                "sufficient_decrease",
                "max_iterations",
            ]),
        }
    }
}

impl ScalarUDFImpl for BacktrackingLineSearchComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "backtracking_line_search_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        validate_arg_count(self.name(), arg_types.len(), 3, 7)?;
        let mut scalars = Vec::new();
        for position in 4..=6 {
            if arg_types.len() >= position {
                scalars.push((position, ScalarCoercion::Real));
            }
        }
        if arg_types.len() >= 7 {
            scalars.push((7, ScalarCoercion::Integer));
        }
        coerce_scalar_arguments(self.name(), arg_types, &scalars)
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        validate_arg_count(self.name(), args.arg_fields.len(), 3, 7)?;
        let _objective = NamedComplexObjective::parse(
            self.name(),
            &super::common::expect_string_scalar_argument(&args, 1, self.name())?,
        )?;
        let _point = parse_complex_vector_field(&args.arg_fields[1], self.name(), 2)?;
        let _direction = parse_complex_vector_field(&args.arg_fields[2], self.name(), 3)?;
        Ok(scalar_field(self.name(), &DataType::Float64, nullable_or(&args.arg_fields[..3])))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        validate_arg_count(self.name(), args.arg_fields.len(), 3, 7)?;
        invoke_line_search(&args, self.name())
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            ml_doc(
                "Compute an Armijo backtracking step for each complex vector row using a built-in \
                 named objective.",
                "backtracking_line_search_complex('norm_squared', point_batch, direction_batch, \
                 initial_step => 1.0)",
            )
            .with_argument(
                "function",
                "Built-in complex objective name: norm_squared or quartic_norm.",
            )
            .with_argument("point_batch", "Canonical complex dense vector batch.")
            .with_argument("direction_batch", "Canonical complex search-direction batch.")
            .with_argument("initial_step", "Optional initial step size.")
            .with_argument("contraction", "Optional backtracking contraction in (0, 1).")
            .with_argument("sufficient_decrease", "Optional Armijo coefficient in (0, 1).")
            .with_argument("max_iterations", "Optional maximum backtracking iterations.")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

macro_rules! complex_optimizer_udf {
    (
        $struct_name:ident,
        $udf_name:literal,
        [$($param:literal),+ $(,)?],
        $min_args:expr,
        $max_args:expr,
        [$($coerce_position:expr => $coerce_kind:expr),* $(,)?],
        $config_expr:expr,
        $call:path,
        $doc_syntax:literal,
        $doc_description:literal
    ) => {
        #[derive(Debug, PartialEq, Eq, Hash)]
        struct $struct_name {
            signature: Signature,
        }

        impl $struct_name {
            fn new() -> Self {
                Self { signature: named_user_defined_signature(&[$($param),+]) }
            }
        }

        impl ScalarUDFImpl for $struct_name {
            fn as_any(&self) -> &dyn Any { self }
            fn name(&self) -> &'static str { $udf_name }
            fn signature(&self) -> &Signature { &self.signature }

            fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
                validate_arg_count(self.name(), arg_types.len(), $min_args, $max_args)?;
                let mut scalars = Vec::new();
                $(
                    if arg_types.len() >= $coerce_position {
                        scalars.push(($coerce_position, $coerce_kind));
                    }
                )*
                coerce_scalar_arguments(self.name(), arg_types, &scalars)
            }

            fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
                datafusion::common::internal_err!("return_field_from_args should be used instead")
            }

            fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
                validate_arg_count(self.name(), args.arg_fields.len(), $min_args, $max_args)?;
                let _objective = NamedComplexObjective::parse(
                    self.name(),
                    &super::common::expect_string_scalar_argument(&args, 1, self.name())?,
                )?;
                let (_item, contract) = parse_complex_vector_field(&args.arg_fields[1], self.name(), 2)?;
                complex_vector_field(self.name(), contract.len, nullable_or(&args.arg_fields[..2]))
            }

            fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
                validate_arg_count(self.name(), args.arg_fields.len(), $min_args, $max_args)?;
                invoke_complex_optimizer(&args, self.name(), |initial, objective, args, function_name| {
                    let config = $config_expr(args, function_name)?;
                    $call(
                        initial,
                        |input| objective.objective(input),
                        |input| objective.gradient(input),
                        &config,
                    )
                    .map_err(|error| exec_error(function_name, error))
                })
            }

            fn documentation(&self) -> Option<&Documentation> {
                static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
                    ml_doc($doc_description, $doc_syntax)
                        .with_argument("function", "Built-in complex objective name: norm_squared or quartic_norm.")
                        .with_argument("initial_batch", "Canonical complex dense vector batch.")
                        .build()
                });
                Some(&DOCUMENTATION)
            }
        }
    };
}

complex_optimizer_udf!(
    GradientDescentComplex,
    "gradient_descent_complex",
    ["function", "initial", "learning_rate", "max_iterations", "tolerance"],
    2,
    5,
    [3 => ScalarCoercion::Real, 4 => ScalarCoercion::Integer, 5 => ScalarCoercion::Real],
    sgd_config,
    nabled::ml::optimization::gradient_descent_complex,
    "gradient_descent_complex('norm_squared', initial_batch, learning_rate => 1e-2)",
    "Run fixed-step complex gradient descent over each initial vector row using a built-in named objective."
);

complex_optimizer_udf!(
    AdamComplex,
    "adam_complex",
    ["function", "initial", "learning_rate", "beta1", "beta2", "epsilon", "max_iterations", "tolerance"],
    2,
    8,
    [
        3 => ScalarCoercion::Real,
        4 => ScalarCoercion::Real,
        5 => ScalarCoercion::Real,
        6 => ScalarCoercion::Real,
        7 => ScalarCoercion::Integer,
        8 => ScalarCoercion::Real
    ],
    adam_config,
    nabled::ml::optimization::adam_complex,
    "adam_complex('norm_squared', initial_batch, learning_rate => 1e-2, beta1 => 0.9, beta2 => 0.999)",
    "Run Adam optimization over each complex initial vector row using a built-in named objective."
);

complex_optimizer_udf!(
    MomentumDescentComplex,
    "momentum_descent_complex",
    ["function", "initial", "learning_rate", "momentum", "max_iterations", "tolerance"],
    2,
    6,
    [
        3 => ScalarCoercion::Real,
        4 => ScalarCoercion::Real,
        5 => ScalarCoercion::Integer,
        6 => ScalarCoercion::Real
    ],
    momentum_config,
    nabled::ml::optimization::momentum_descent_complex,
    "momentum_descent_complex('norm_squared', initial_batch, learning_rate => 1e-2, momentum => 0.9)",
    "Run momentum gradient descent over each complex initial vector row using a built-in named objective."
);

#[must_use]
pub fn backtracking_line_search_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(BacktrackingLineSearchComplex::new()))
}

#[must_use]
pub fn gradient_descent_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(GradientDescentComplex::new()))
}

#[must_use]
pub fn adam_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(AdamComplex::new()))
}

#[must_use]
pub fn momentum_descent_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MomentumDescentComplex::new()))
}

#[cfg(test)]
mod tests {
    use datafusion::common::ScalarValue;
    use datafusion::common::config::ConfigOptions;

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
    fn named_complex_objective_helpers_cover_parse_and_evaluation() {
        assert_eq!(
            NamedComplexObjective::parse("adam_complex", "quartic_norm").expect("quartic"),
            NamedComplexObjective::QuarticNorm
        );
        assert!(NamedComplexObjective::parse("adam_complex", "unknown").is_err());

        let input = ndarray::array![Complex64::new(1.0, 1.0), Complex64::new(-1.0, 0.0)];
        let norm_squared = NamedComplexObjective::NormSquared.objective(&input);
        assert!((norm_squared - 3.0).abs() < 1.0e-12);
        let gradient = NamedComplexObjective::NormSquared.gradient(&input);
        assert_eq!(gradient[0], Complex64::new(2.0, 2.0));
        assert_eq!(gradient[1], Complex64::new(-2.0, 0.0));
        let quartic = NamedComplexObjective::QuarticNorm.objective(&input);
        assert!((quartic - 5.0).abs() < 1.0e-12);
        let quartic_gradient = NamedComplexObjective::QuarticNorm.gradient(&input);
        assert_eq!(quartic_gradient[0], Complex64::new(8.0, 8.0));
        assert_eq!(quartic_gradient[1], Complex64::new(-4.0, 0.0));
        assert!(validate_arg_count("adam_complex", 1, 2, 8).is_err());
        assert!(validate_arg_count("adam_complex", 2, 2, 8).is_ok());
    }

    fn valid_sgd_args() -> ScalarFunctionArgs {
        scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("norm_squared".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.25))),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(8))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0e-4))),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                complex_vector_field("initial", 2, false).expect("initial"),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "learning_rate",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "max_iterations",
                    DataType::Int64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "tolerance",
                    DataType::Float64,
                    false,
                )),
            ],
        )
    }

    fn valid_momentum_args() -> ScalarFunctionArgs {
        scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("norm_squared".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.25))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.5))),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(8))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0e-4))),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                complex_vector_field("initial", 2, false).expect("initial"),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "learning_rate",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "momentum",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "max_iterations",
                    DataType::Int64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "tolerance",
                    DataType::Float64,
                    false,
                )),
            ],
        )
    }

    fn valid_line_search_defaults() -> ScalarFunctionArgs {
        scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("norm_squared".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Null),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                complex_vector_field("point", 2, false).expect("point"),
                complex_vector_field("direction", 2, false).expect("direction"),
            ],
        )
    }

    fn valid_adam_args() -> ScalarFunctionArgs {
        scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("norm_squared".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.1))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.9))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.999))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0e-8))),
                ColumnarValue::Scalar(ScalarValue::Int64(Some(16))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0e-6))),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                complex_vector_field("initial", 2, false).expect("initial"),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "learning_rate",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "beta1",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "beta2",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "epsilon",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "max_iterations",
                    DataType::Int64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "tolerance",
                    DataType::Float64,
                    false,
                )),
            ],
        )
    }

    #[test]
    fn optimizer_config_helpers_accept_valid_descent_values() {
        assert!(sgd_config(&valid_sgd_args(), "gradient_descent_complex").is_ok());
        assert!(momentum_config(&valid_momentum_args(), "momentum_descent_complex").is_ok());
    }

    #[test]
    fn optimizer_config_helpers_accept_valid_line_search_and_adam_values() {
        assert!(
            line_search_config(&valid_line_search_defaults(), "backtracking_line_search_complex")
                .is_ok()
        );
        assert!(adam_config(&valid_adam_args(), "adam_complex").is_ok());
    }

    #[test]
    fn optimizer_config_helpers_reject_invalid_values() {
        let invalid_line_search = scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("norm_squared".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(-1.0))),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                complex_vector_field("point", 2, false).expect("point"),
                complex_vector_field("direction", 2, false).expect("direction"),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "initial_step",
                    DataType::Float64,
                    false,
                )),
            ],
        );
        assert!(
            line_search_config(&invalid_line_search, "backtracking_line_search_complex").is_err()
        );

        let invalid_adam = scalar_args(
            vec![
                ColumnarValue::Scalar(ScalarValue::Utf8(Some("norm_squared".to_string()))),
                ColumnarValue::Scalar(ScalarValue::Null),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(0.1))),
                ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0))),
            ],
            vec![
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "function",
                    DataType::Utf8,
                    false,
                )),
                complex_vector_field("initial", 2, false).expect("initial"),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "learning_rate",
                    DataType::Float64,
                    false,
                )),
                Arc::new(datafusion::arrow::datatypes::Field::new(
                    "beta1",
                    DataType::Float64,
                    false,
                )),
            ],
        );
        assert!(adam_config(&invalid_adam, "adam_complex").is_err());
    }
}
