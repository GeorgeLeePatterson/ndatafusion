use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::FixedSizeListArray;
use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::datatypes::{DataType, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use ndarray::{Array3, Axis};
use ndarrow::NdarrowElement;

use super::common::{
    complex_fixed_shape_tensor_array_from_flat_rows, complex_fixed_shape_tensor_view3,
    expect_fixed_size_list_arg, expect_real_scalar_arg, expect_real_scalar_argument,
    expect_usize_scalar_arg, expect_usize_scalar_argument, fixed_shape_tensor_view3, nullable_or,
};
use super::docs::matrix_doc;
use crate::error::exec_error;
use crate::metadata::{
    complex_fixed_shape_tensor_field, fixed_shape_tensor_field, parse_complex_matrix_batch_field,
    parse_matrix_batch_field,
};
use crate::signatures::{
    ScalarCoercion, any_signature, coerce_scalar_arguments, named_user_defined_signature,
};

fn validate_square_matrix_contract(
    function_name: &str,
    matrix: &crate::metadata::MatrixBatchContract,
) -> Result<()> {
    validate_square_dimensions(function_name, matrix.rows, matrix.cols)
}

fn validate_square_dimensions(function_name: &str, rows: usize, cols: usize) -> Result<()> {
    if rows != cols {
        return Err(exec_error(
            function_name,
            format!("{function_name} requires square matrices, found ({rows}, {cols})"),
        ));
    }
    Ok(())
}

fn validate_positive_finite_scalar(function_name: &str, label: &str, value: f64) -> Result<()> {
    if !value.is_finite() {
        return Err(exec_error(function_name, format!("{label} must be finite, found {value}")));
    }
    if value <= 0.0 {
        return Err(exec_error(function_name, format!("{label} must be positive, found {value}")));
    }
    Ok(())
}

fn validate_finite_scalar(function_name: &str, label: &str, value: f64) -> Result<()> {
    if value.is_finite() {
        Ok(())
    } else {
        Err(exec_error(function_name, format!("{label} must be finite, found {value}")))
    }
}

fn native_scalar<T>(function_name: &str, label: &str, value: f64) -> Result<T>
where
    T: nabled::core::prelude::NabledReal,
{
    T::from_f64(value).ok_or_else(|| {
        exec_error(
            function_name,
            format!("{label} could not be represented in the target value type"),
        )
    })
}

fn tensor_batch_from_flat_values<T>(
    function_name: &str,
    batch: usize,
    rows: usize,
    cols: usize,
    values: Vec<T::Native>,
) -> Result<FixedSizeListArray>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
{
    let tensor = Array3::from_shape_vec((batch, rows, cols), values)
        .map_err(|error| exec_error(function_name, error))?;
    let (_field, array) = ndarrow::arrayd_to_fixed_shape_tensor(function_name, tensor.into_dyn())
        .map_err(|error| exec_error(function_name, error))?;
    Ok(array)
}

fn invoke_square_matrix_tensor_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<ndarray::Array2<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: nabled::linalg::matrix_functions::MatrixFunctionScalar + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let rows = matrix_view.len_of(Axis(1));
    let cols = matrix_view.len_of(Axis(2));
    let mut output = Vec::with_capacity(batch * rows * cols);
    for row in 0..batch {
        let result = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        output.extend(result.iter().copied());
    }
    let output = tensor_batch_from_flat_values::<T>(function_name, batch, rows, cols, output)?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_complex_square_matrix_tensor_output<E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(
        &ndarray::ArrayView2<'_, num_complex::Complex64>,
    ) -> std::result::Result<ndarray::Array2<num_complex::Complex64>, E>,
) -> Result<ColumnarValue>
where
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view =
        complex_fixed_shape_tensor_view3(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let rows = matrix_view.len_of(Axis(1));
    let cols = matrix_view.len_of(Axis(2));
    let mut output = Vec::with_capacity(batch * rows * cols);
    for row in 0..batch {
        let result = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        output.extend(result.iter().copied());
    }
    let (_field, output) = complex_fixed_shape_tensor_array_from_flat_rows(
        function_name,
        batch,
        &[rows, cols],
        output,
    )?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixExp {
    signature: Signature,
}

impl MatrixExp {
    fn new() -> Self {
        Self { signature: named_user_defined_signature(&["matrix", "max_terms", "tolerance"]) }
    }
}

impl ScalarUDFImpl for MatrixExp {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_exp" }

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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        let _ = expect_usize_scalar_argument(&args, 2, self.name())?;
        let tolerance = expect_real_scalar_argument(&args, 3, self.name())?;
        validate_positive_finite_scalar(self.name(), "tolerance", tolerance)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        let max_terms = expect_usize_scalar_arg(&args, 2, self.name())?;
        let tolerance = expect_real_scalar_arg(&args, 3, self.name())?;
        validate_positive_finite_scalar(self.name(), "tolerance", tolerance)?;
        match matrix.value_type {
            DataType::Float32 => {
                let tolerance = native_scalar::<f32>(self.name(), "tolerance", tolerance)?;
                invoke_square_matrix_tensor_output::<Float32Type, _>(&args, self.name(), |view| {
                    nabled::linalg::matrix_functions::matrix_exp_view(view, max_terms, tolerance)
                })
            }
            DataType::Float64 => {
                let tolerance = native_scalar::<f64>(self.name(), "tolerance", tolerance)?;
                invoke_square_matrix_tensor_output::<Float64Type, _>(&args, self.name(), |view| {
                    nabled::linalg::matrix_functions::matrix_exp_view(view, max_terms, tolerance)
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute a configurable matrix exponential over a batch of square matrices.",
                "matrix_exp(matrix_batch, max_terms => 32, tolerance => 1e-6)",
            )
            .with_argument(
                "matrix",
                "Square dense matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument("max_terms", "Positive integer maximum number of series terms.")
            .with_argument("tolerance", "Positive finite convergence tolerance.")
            .with_alternative_syntax(
                "matrix_exp(matrix => matrix_batch, max_terms => 32, tolerance => 1e-6)",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixExpEigen {
    signature: Signature,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixExpComplex {
    signature: Signature,
}

impl MatrixExpComplex {
    fn new() -> Self {
        Self { signature: named_user_defined_signature(&["matrix", "max_terms", "tolerance"]) }
    }
}

impl ScalarUDFImpl for MatrixExpComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_exp_complex" }

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
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &crate::metadata::MatrixBatchContract {
            value_type: DataType::Float64,
            rows:       matrix.rows,
            cols:       matrix.cols,
        })?;
        let _ = expect_usize_scalar_argument(&args, 2, self.name())?;
        let tolerance = expect_real_scalar_argument(&args, 3, self.name())?;
        validate_positive_finite_scalar(self.name(), "tolerance", tolerance)?;
        complex_fixed_shape_tensor_field(
            self.name(),
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &crate::metadata::MatrixBatchContract {
            value_type: DataType::Float64,
            rows:       matrix.rows,
            cols:       matrix.cols,
        })?;
        let max_terms = expect_usize_scalar_arg(&args, 2, self.name())?;
        let tolerance = expect_real_scalar_arg(&args, 3, self.name())?;
        validate_positive_finite_scalar(self.name(), "tolerance", tolerance)?;
        invoke_complex_square_matrix_tensor_output(&args, self.name(), |view| {
            nabled::linalg::matrix_functions::matrix_exp_complex_view(view, max_terms, tolerance)
        })
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute a configurable complex matrix exponential over a batch of square \
                 matrices.",
                "matrix_exp_complex(matrix_batch, max_terms => 32, tolerance => 1e-6)",
            )
            .with_argument(
                "matrix",
                "Square complex matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_argument("max_terms", "Positive integer maximum number of series terms.")
            .with_argument("tolerance", "Positive finite convergence tolerance.")
            .with_alternative_syntax(
                "matrix_exp_complex(matrix => matrix_batch, max_terms => 32, tolerance => 1e-6)",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixExpEigenComplex {
    signature: Signature,
}

impl MatrixExpEigenComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixExpEigenComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_exp_eigen_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &crate::metadata::MatrixBatchContract {
            value_type: DataType::Float64,
            rows:       matrix.rows,
            cols:       matrix.cols,
        })?;
        complex_fixed_shape_tensor_field(
            self.name(),
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &crate::metadata::MatrixBatchContract {
            value_type: DataType::Float64,
            rows:       matrix.rows,
            cols:       matrix.cols,
        })?;
        invoke_complex_square_matrix_tensor_output(&args, self.name(), |view| {
            nabled::linalg::matrix_functions::matrix_exp_eigen_complex_view(view)
        })
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute a complex matrix exponential via eigen decomposition over a batch of \
                 square matrices.",
                "matrix_exp_eigen_complex(matrix_batch)",
            )
            .with_argument(
                "matrix",
                "Square complex matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

impl MatrixExpEigen {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixExpEigen {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_exp_eigen" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        match matrix.value_type {
            DataType::Float32 => invoke_square_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix_functions::matrix_exp_eigen_view,
            ),
            DataType::Float64 => invoke_square_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix_functions::matrix_exp_eigen_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixLogTaylor {
    signature: Signature,
}

impl MatrixLogTaylor {
    fn new() -> Self {
        Self { signature: named_user_defined_signature(&["matrix", "max_terms", "tolerance"]) }
    }
}

impl ScalarUDFImpl for MatrixLogTaylor {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_log_taylor" }

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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        let _ = expect_usize_scalar_argument(&args, 2, self.name())?;
        let tolerance = expect_real_scalar_argument(&args, 3, self.name())?;
        validate_positive_finite_scalar(self.name(), "tolerance", tolerance)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        let max_terms = expect_usize_scalar_arg(&args, 2, self.name())?;
        let tolerance = expect_real_scalar_arg(&args, 3, self.name())?;
        validate_positive_finite_scalar(self.name(), "tolerance", tolerance)?;
        match matrix.value_type {
            DataType::Float32 => {
                let tolerance = native_scalar::<f32>(self.name(), "tolerance", tolerance)?;
                invoke_square_matrix_tensor_output::<Float32Type, _>(&args, self.name(), |view| {
                    nabled::linalg::matrix_functions::matrix_log_taylor_view(
                        view, max_terms, tolerance,
                    )
                })
            }
            DataType::Float64 => {
                let tolerance = native_scalar::<f64>(self.name(), "tolerance", tolerance)?;
                invoke_square_matrix_tensor_output::<Float64Type, _>(&args, self.name(), |view| {
                    nabled::linalg::matrix_functions::matrix_log_taylor_view(
                        view, max_terms, tolerance,
                    )
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute a configurable matrix logarithm with a Taylor-series path over a batch \
                 of square matrices.",
                "matrix_log_taylor(matrix_batch, max_terms => 32, tolerance => 1e-6)",
            )
            .with_argument(
                "matrix",
                "Square dense matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument("max_terms", "Positive integer maximum number of series terms.")
            .with_argument("tolerance", "Positive finite convergence tolerance.")
            .with_alternative_syntax(
                "matrix_log_taylor(matrix => matrix_batch, max_terms => 32, tolerance => 1e-6)",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixLogEigen {
    signature: Signature,
}

impl MatrixLogEigen {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixLogEigen {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_log_eigen" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        match matrix.value_type {
            DataType::Float32 => invoke_square_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix_functions::matrix_log_eigen_view,
            ),
            DataType::Float64 => invoke_square_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix_functions::matrix_log_eigen_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixLogSvd {
    signature: Signature,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixLogEigenComplex {
    signature: Signature,
}

impl MatrixLogEigenComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixLogEigenComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_log_eigen_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &crate::metadata::MatrixBatchContract {
            value_type: DataType::Float64,
            rows:       matrix.rows,
            cols:       matrix.cols,
        })?;
        complex_fixed_shape_tensor_field(
            self.name(),
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &crate::metadata::MatrixBatchContract {
            value_type: DataType::Float64,
            rows:       matrix.rows,
            cols:       matrix.cols,
        })?;
        invoke_complex_square_matrix_tensor_output(&args, self.name(), |view| {
            nabled::linalg::matrix_functions::matrix_log_eigen_complex_view(view)
        })
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute a complex matrix logarithm via eigen decomposition over a batch of \
                 square matrices.",
                "matrix_log_eigen_complex(matrix_batch)",
            )
            .with_argument(
                "matrix",
                "Square complex matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixLogSvdComplex {
    signature: Signature,
}

impl MatrixLogSvdComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixLogSvdComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_log_svd_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &crate::metadata::MatrixBatchContract {
            value_type: DataType::Float64,
            rows:       matrix.rows,
            cols:       matrix.cols,
        })?;
        complex_fixed_shape_tensor_field(
            self.name(),
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &crate::metadata::MatrixBatchContract {
            value_type: DataType::Float64,
            rows:       matrix.rows,
            cols:       matrix.cols,
        })?;
        invoke_complex_square_matrix_tensor_output(&args, self.name(), |view| {
            nabled::linalg::matrix_functions::matrix_log_svd_complex_view(view)
        })
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute a complex matrix logarithm via SVD over a batch of square matrices.",
                "matrix_log_svd_complex(matrix_batch)",
            )
            .with_argument(
                "matrix",
                "Square complex matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

impl MatrixLogSvd {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixLogSvd {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_log_svd" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        match matrix.value_type {
            DataType::Float32 => invoke_square_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix_functions::matrix_log_svd_view,
            ),
            DataType::Float64 => invoke_square_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix_functions::matrix_log_svd_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixPower {
    signature: Signature,
}

impl MatrixPower {
    fn new() -> Self { Self { signature: named_user_defined_signature(&["matrix", "power"]) } }
}

impl ScalarUDFImpl for MatrixPower {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_power" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[(2, ScalarCoercion::Real)])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        let power = expect_real_scalar_argument(&args, 2, self.name())?;
        validate_finite_scalar(self.name(), "power", power)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        let power = expect_real_scalar_arg(&args, 2, self.name())?;
        validate_finite_scalar(self.name(), "power", power)?;
        match matrix.value_type {
            DataType::Float32 => {
                let power = native_scalar::<f32>(self.name(), "power", power)?;
                invoke_square_matrix_tensor_output::<Float32Type, _>(&args, self.name(), |view| {
                    nabled::linalg::matrix_functions::matrix_power_view(view, power)
                })
            }
            DataType::Float64 => {
                let power = native_scalar::<f64>(self.name(), "power", power)?;
                invoke_square_matrix_tensor_output::<Float64Type, _>(&args, self.name(), |view| {
                    nabled::linalg::matrix_functions::matrix_power_view(view, power)
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Raise each square matrix in the batch to a scalar power.",
                "matrix_power(matrix_batch, power => 2.0)",
            )
            .with_argument(
                "matrix",
                "Square dense matrix batch in canonical fixed-shape tensor form.",
            )
            .with_argument("power", "Finite scalar exponent applied to each matrix.")
            .with_alternative_syntax("matrix_power(matrix => matrix_batch, power => 2.0)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSign {
    signature: Signature,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixPowerComplex {
    signature: Signature,
}

impl MatrixPowerComplex {
    fn new() -> Self { Self { signature: named_user_defined_signature(&["matrix", "power"]) } }
}

impl ScalarUDFImpl for MatrixPowerComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_power_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[(2, ScalarCoercion::Real)])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_dimensions(self.name(), matrix.rows, matrix.cols)?;
        let power = expect_real_scalar_argument(&args, 2, self.name())?;
        validate_finite_scalar(self.name(), "power", power)?;
        complex_fixed_shape_tensor_field(
            self.name(),
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_dimensions(self.name(), matrix.rows, matrix.cols)?;
        let power = expect_real_scalar_arg(&args, 2, self.name())?;
        validate_finite_scalar(self.name(), "power", power)?;
        invoke_complex_square_matrix_tensor_output(&args, self.name(), |view| {
            nabled::linalg::matrix_functions::matrix_power_complex_view(view, power)
        })
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Raise each complex square matrix in the batch to a scalar power.",
                "matrix_power_complex(matrix_batch, power => 2.0)",
            )
            .with_argument(
                "matrix",
                "Square complex matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .with_argument("power", "Finite scalar exponent applied to each matrix.")
            .with_alternative_syntax("matrix_power_complex(matrix => matrix_batch, power => 2.0)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSignComplex {
    signature: Signature,
}

impl MatrixSignComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSignComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_sign_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_dimensions(self.name(), matrix.rows, matrix.cols)?;
        complex_fixed_shape_tensor_field(
            self.name(),
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_dimensions(self.name(), matrix.rows, matrix.cols)?;
        invoke_complex_square_matrix_tensor_output(&args, self.name(), |view| {
            nabled::linalg::matrix_functions::matrix_sign_complex_view(view)
        })
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            matrix_doc(
                "Compute the complex matrix sign function over a batch of square matrices.",
                "matrix_sign_complex(matrix_batch)",
            )
            .with_argument(
                "matrix",
                "Square complex matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

impl MatrixSign {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSign {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_sign" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.rows, matrix.cols],
            nullable_or(args.arg_fields),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        validate_square_matrix_contract(self.name(), &matrix)?;
        match matrix.value_type {
            DataType::Float32 => invoke_square_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix_functions::matrix_sign_view,
            ),
            DataType::Float64 => invoke_square_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::matrix_functions::matrix_sign_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[must_use]
pub fn matrix_exp_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MatrixExp::new())) }

#[must_use]
pub fn matrix_exp_eigen_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixExpEigen::new()))
}

#[must_use]
pub fn matrix_exp_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixExpComplex::new()))
}

#[must_use]
pub fn matrix_exp_eigen_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixExpEigenComplex::new()))
}

#[must_use]
pub fn matrix_log_taylor_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixLogTaylor::new()))
}

#[must_use]
pub fn matrix_log_eigen_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixLogEigen::new()))
}

#[must_use]
pub fn matrix_log_eigen_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixLogEigenComplex::new()))
}

#[must_use]
pub fn matrix_log_svd_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixLogSvd::new()))
}

#[must_use]
pub fn matrix_log_svd_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixLogSvdComplex::new()))
}

#[must_use]
pub fn matrix_power_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixPower::new()))
}

#[must_use]
pub fn matrix_power_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixPowerComplex::new()))
}

#[must_use]
pub fn matrix_sign_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MatrixSign::new())) }

#[must_use]
pub fn matrix_sign_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSignComplex::new()))
}
