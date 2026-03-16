use std::any::Any;
use std::sync::{Arc, LazyLock};

use datafusion::arrow::array::types::{ArrowPrimitiveType, Float32Type, Float64Type};
use datafusion::arrow::array::{ArrayRef, Int8Array, Int64Array, StructArray};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::Result;
use datafusion::logical_expr::{
    ColumnarValue, Documentation, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl,
    Signature,
};
use nabled::core::prelude::NabledReal;
use ndarray::{Array2, Array3, Axis};
use ndarrow::NdarrowElement;
use num_complex::{Complex, Complex64};

use super::common::{
    complex_fixed_shape_tensor_array_from_flat_rows, complex_fixed_shape_tensor_view3,
    complex_fixed_size_list_array_from_flat_rows, expect_fixed_size_list_arg,
    expect_real_scalar_arg, expect_real_scalar_argument, expect_usize_scalar_arg,
    expect_usize_scalar_argument, fixed_shape_tensor_view3, fixed_size_list_array_from_flat_rows,
    fixed_size_list_view2, nullable_or, primitive_array_from_values,
};
use super::docs::decomposition_doc;
use crate::error::exec_error;
use crate::metadata::{
    complex_fixed_shape_tensor_field, complex_vector_field, fixed_shape_tensor_field,
    parse_complex_matrix_batch_field, parse_matrix_batch_field, parse_vector_field, scalar_field,
    struct_field, variable_shape_tensor_field, vector_field,
};
use crate::signatures::{
    ScalarCoercion, any_signature, coerce_scalar_arguments, named_user_defined_signature,
};

fn square_matrix_shape(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(DataType, usize, usize, bool)> {
    let matrix = parse_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    if matrix.rows != matrix.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square matrices, found ({}, {})",
                matrix.rows, matrix.cols
            ),
        ));
    }
    Ok((matrix.value_type, matrix.rows, matrix.cols, nullable_or(args.arg_fields)))
}

fn complex_square_matrix_shape(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(usize, bool)> {
    let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    if matrix.rows != matrix.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square matrices, found ({}, {})",
                matrix.rows, matrix.cols
            ),
        ));
    }
    Ok((matrix.rows, nullable_or(args.arg_fields)))
}

fn tensor_column<T>(
    name: &str,
    shape: [usize; 3],
    values: Vec<T>,
) -> Result<(FieldRef, Arc<datafusion::arrow::array::FixedSizeListArray>)>
where
    T: NdarrowElement,
{
    let tensor = Array3::from_shape_vec((shape[0], shape[1], shape[2]), values)
        .map_err(|error| exec_error(name, error))?;
    let (field, array) = ndarrow::arrayd_to_fixed_shape_tensor(name, tensor.into_dyn())
        .map_err(|error| exec_error(name, error))?;
    Ok((Arc::new(field), Arc::new(array)))
}

fn int64_scalar_field(name: &str, nullable: bool) -> FieldRef {
    Arc::new(Field::new(name, DataType::Int64, nullable))
}

fn square_matrix_pair_shape(
    args: &ReturnFieldArgs<'_>,
    function_name: &str,
) -> Result<(DataType, usize, bool)> {
    let left = parse_matrix_batch_field(&args.arg_fields[0], function_name, 1)?;
    let right = parse_matrix_batch_field(&args.arg_fields[1], function_name, 2)?;
    if left.rows != left.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square left matrices, found ({}, {})",
                left.rows, left.cols
            ),
        ));
    }
    if right.rows != right.cols {
        return Err(exec_error(
            function_name,
            format!(
                "{function_name} requires square right matrices, found ({}, {})",
                right.rows, right.cols
            ),
        ));
    }
    if left.value_type != right.value_type {
        return Err(exec_error(
            function_name,
            format!(
                "matrix value type mismatch: left {}, right {}",
                left.value_type, right.value_type
            ),
        ));
    }
    if left.rows != right.rows || left.cols != right.cols {
        return Err(exec_error(
            function_name,
            format!(
                "matrix shape mismatch: left ({}, {}), right ({}, {})",
                left.rows, left.cols, right.rows, right.cols
            ),
        ));
    }
    Ok((left.value_type, left.rows, nullable_or(args.arg_fields)))
}

fn eigen_struct_field(
    name: &str,
    value_type: &DataType,
    dimension: usize,
    nullable: bool,
) -> Result<FieldRef> {
    let eigenvalues = vector_field("eigenvalues", value_type, dimension, false)?;
    let eigenvectors =
        fixed_shape_tensor_field("eigenvectors", value_type, &[dimension, dimension], false)?;
    Ok(struct_field(
        name,
        vec![eigenvalues.as_ref().clone(), eigenvectors.as_ref().clone()],
        nullable,
    ))
}

fn qr_struct_field(
    name: &str,
    value_type: &DataType,
    q_shape: [usize; 2],
    r_shape: [usize; 2],
    p_shape: Option<[usize; 2]>,
    nullable: bool,
) -> Result<FieldRef> {
    let q_field = fixed_shape_tensor_field("q", value_type, &q_shape, false)?;
    let r_field = fixed_shape_tensor_field("r", value_type, &r_shape, false)?;
    let mut fields = vec![
        q_field.as_ref().clone(),
        r_field.as_ref().clone(),
        Field::new("rank", DataType::Int64, false),
    ];
    if let Some(permutation_shape) = p_shape {
        let permutation = fixed_shape_tensor_field("p", value_type, &permutation_shape, false)?;
        fields.insert(2, permutation.as_ref().clone());
    }
    Ok(struct_field(name, fields, nullable))
}

fn svd_struct_field(
    name: &str,
    value_type: &DataType,
    u_shape: [usize; 2],
    singular_len: usize,
    vt_shape: [usize; 2],
    nullable: bool,
) -> Result<FieldRef> {
    let u_field = fixed_shape_tensor_field("u", value_type, &u_shape, false)?;
    let singular_values = vector_field("singular_values", value_type, singular_len, false)?;
    let vt_field = fixed_shape_tensor_field("vt", value_type, &vt_shape, false)?;
    Ok(struct_field(
        name,
        vec![u_field.as_ref().clone(), singular_values.as_ref().clone(), vt_field.as_ref().clone()],
        nullable,
    ))
}

fn double_tensor_struct_field(
    name: &str,
    value_type: &DataType,
    first_name: &str,
    second_name: &str,
    shape: [usize; 2],
    nullable: bool,
) -> Result<FieldRef> {
    let first = fixed_shape_tensor_field(first_name, value_type, &shape, false)?;
    let second = fixed_shape_tensor_field(second_name, value_type, &shape, false)?;
    Ok(struct_field(name, vec![first.as_ref().clone(), second.as_ref().clone()], nullable))
}

fn complex_double_tensor_struct_field(
    name: &str,
    first_name: &str,
    second_name: &str,
    shape: [usize; 2],
    nullable: bool,
) -> Result<FieldRef> {
    let first = complex_fixed_shape_tensor_field(first_name, &shape, false)?;
    let second = complex_fixed_shape_tensor_field(second_name, &shape, false)?;
    Ok(struct_field(name, vec![first.as_ref().clone(), second.as_ref().clone()], nullable))
}

fn complex_eigen_struct_field(name: &str, dimension: usize, nullable: bool) -> Result<FieldRef> {
    let eigenvalues = complex_vector_field("eigenvalues", dimension, false)?;
    let schur_vectors =
        complex_fixed_shape_tensor_field("schur_vectors", &[dimension, dimension], false)?;
    Ok(struct_field(
        name,
        vec![eigenvalues.as_ref().clone(), schur_vectors.as_ref().clone()],
        nullable,
    ))
}

fn real_input_complex_bi_eigen_struct_field(
    name: &str,
    value_type: &DataType,
    dimension: usize,
    nullable: bool,
) -> Result<FieldRef> {
    let eigenvalues = complex_vector_field("eigenvalues", dimension, false)?;
    let right_eigenvectors =
        complex_fixed_shape_tensor_field("right_eigenvectors", &[dimension, dimension], false)?;
    let left_eigenvectors =
        complex_fixed_shape_tensor_field("left_eigenvectors", &[dimension, dimension], false)?;
    let balancing_diagonal = vector_field("balancing_diagonal", value_type, dimension, false)?;
    let balanced_matrix =
        fixed_shape_tensor_field("balanced_matrix", value_type, &[dimension, dimension], false)?;
    Ok(struct_field(
        name,
        vec![
            eigenvalues.as_ref().clone(),
            right_eigenvectors.as_ref().clone(),
            left_eigenvectors.as_ref().clone(),
            balancing_diagonal.as_ref().clone(),
            balanced_matrix.as_ref().clone(),
        ],
        nullable,
    ))
}

fn tensor_vector_struct_field(
    name: &str,
    value_type: &DataType,
    tensor_name: &str,
    tensor_shape: [usize; 2],
    vector_name: &str,
    vector_len: usize,
    nullable: bool,
) -> Result<FieldRef> {
    let tensor = fixed_shape_tensor_field(tensor_name, value_type, &tensor_shape, false)?;
    let vector = vector_field(vector_name, value_type, vector_len, false)?;
    Ok(struct_field(name, vec![tensor.as_ref().clone(), vector.as_ref().clone()], nullable))
}

fn validate_positive_usize(function_name: &str, label: &str, value: usize) -> Result<usize> {
    if value == 0 {
        return Err(exec_error(function_name, format!("{label} must be greater than 0")));
    }
    Ok(value)
}

fn validate_non_negative_scalar(function_name: &str, label: &str, value: f64) -> Result<f64> {
    if !value.is_finite() {
        return Err(exec_error(function_name, format!("{label} must be finite")));
    }
    if value < 0.0 {
        return Err(exec_error(function_name, format!("{label} must be non-negative")));
    }
    Ok(value)
}

fn native_scalar<T>(function_name: &str, label: &str, value: f64) -> Result<T>
where
    T: NabledReal,
{
    T::from_f64(value).ok_or_else(|| {
        exec_error(function_name, format!("{label} could not be represented in matrix value type"))
    })
}

fn invoke_matrix_scalar<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(&ndarray::ArrayView2<'_, T::Native>) -> std::result::Result<T::Native, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)));
    for row in 0..matrix_view.len_of(Axis(0)) {
        output.push(
            op(&matrix_view.index_axis(Axis(0), row))
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    let output = primitive_array_from_values::<T>(output);
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_matrix_scalar_i64<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(&ndarray::ArrayView2<'_, T::Native>) -> std::result::Result<i64, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)));
    for row in 0..matrix_view.len_of(Axis(0)) {
        output.push(
            op(&matrix_view.index_axis(Axis(0), row))
                .map_err(|error| exec_error(function_name, error))?,
        );
    }
    Ok(ColumnarValue::Array(Arc::new(Int64Array::from(output))))
}

fn invoke_matrix_tensor_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    output_rows: usize,
    output_cols: usize,
    op: impl Fn(&ndarray::ArrayView2<'_, T::Native>) -> std::result::Result<Array2<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
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
    let (_field, output) = tensor_column(function_name, [batch, output_rows, output_cols], output)?;
    Ok(ColumnarValue::Array(output))
}

fn invoke_matrix_qr_vector_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
        &ndarray::ArrayView1<'_, T::Native>,
    ) -> std::result::Result<ndarray::Array1<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let rhs = expect_fixed_size_list_arg(args, 2, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let rhs_view = fixed_size_list_view2::<T>(rhs, function_name)?;
    if matrix_view.len_of(Axis(0)) != rhs_view.nrows() {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} matrices vs {} rhs vectors",
                matrix_view.len_of(Axis(0)),
                rhs_view.nrows()
            ),
        ));
    }
    if matrix_view.len_of(Axis(1)) != rhs_view.ncols() {
        return Err(exec_error(
            function_name,
            format!(
                "rhs vector length mismatch: expected {}, found {}",
                matrix_view.len_of(Axis(1)),
                rhs_view.ncols()
            ),
        ));
    }

    let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)) * matrix_view.len_of(Axis(2)));
    for row in 0..matrix_view.len_of(Axis(0)) {
        let solution =
            op(&matrix_view.index_axis(Axis(0), row), &rhs_view.index_axis(Axis(0), row))
                .map_err(|error| exec_error(function_name, error))?;
        output.extend(solution.iter().copied());
    }
    let output = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        rhs_view.nrows(),
        matrix_view.len_of(Axis(2)),
        &output,
    )?;
    Ok(ColumnarValue::Array(Arc::new(output)))
}

fn invoke_matrix_qr_struct_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    q_shape: [usize; 2],
    r_shape: [usize; 2],
    p_shape: Option<[usize; 2]>,
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<nabled::linalg::qr::QRResult<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut q_values = Vec::with_capacity(batch * q_shape[0] * q_shape[1]);
    let mut r_values = Vec::with_capacity(batch * r_shape[0] * r_shape[1]);
    let mut p_values = p_shape.map(|shape| Vec::with_capacity(batch * shape[0] * shape[1]));
    let mut rank_values = Vec::with_capacity(batch);

    for row in 0..batch {
        let result = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        q_values.extend(result.q.iter().copied());
        r_values.extend(result.r.iter().copied());
        if let Some(values) = &mut p_values {
            let permutation = result.p.ok_or_else(|| {
                exec_error(function_name, "expected permutation matrix in pivoted QR result")
            })?;
            values.extend(permutation.iter().copied());
        }
        rank_values.push(
            i64::try_from(result.rank)
                .map_err(|_| exec_error(function_name, "rank exceeds i64 limits"))?,
        );
    }

    let (q_field, q_array) = tensor_column("q", [batch, q_shape[0], q_shape[1]], q_values)?;
    let (r_field, r_array) = tensor_column("r", [batch, r_shape[0], r_shape[1]], r_values)?;
    let mut fields = vec![q_field, r_field];
    let mut arrays: Vec<ArrayRef> = vec![q_array, r_array];
    if let (Some(shape), Some(values)) = (p_shape, p_values) {
        let (p_field, p_array) = tensor_column("p", [batch, shape[0], shape[1]], values)?;
        fields.push(p_field);
        arrays.push(p_array);
    }
    fields.push(int64_scalar_field("rank", false));
    arrays.push(Arc::new(Int64Array::from(rank_values)));

    Ok(ColumnarValue::Array(Arc::new(StructArray::new(fields.into(), arrays, None))))
}

fn invoke_matrix_svd_struct_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    u_shape: [usize; 2],
    singular_len: usize,
    vt_shape: [usize; 2],
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<nabled::linalg::svd::NdarraySVD<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut u_values = Vec::with_capacity(batch * u_shape[0] * u_shape[1]);
    let mut singular_values = Vec::with_capacity(batch * singular_len);
    let mut vt_values = Vec::with_capacity(batch * vt_shape[0] * vt_shape[1]);

    for row in 0..batch {
        let result = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        u_values.extend(result.u.iter().copied());
        singular_values.extend(result.singular_values.iter().copied());
        vt_values.extend(result.vt.iter().copied());
    }

    let (u_field, u_array) = tensor_column("u", [batch, u_shape[0], u_shape[1]], u_values)?;
    let singular_values = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        batch,
        singular_len,
        &singular_values,
    )?;
    let singular_field = vector_field("singular_values", &T::DATA_TYPE, singular_len, false)?;
    let (vt_field, vt_array) = tensor_column("vt", [batch, vt_shape[0], vt_shape[1]], vt_values)?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![u_field, singular_field, vt_field].into(),
        vec![u_array, Arc::new(singular_values), vt_array],
        None,
    ))))
}

fn invoke_matrix_eigen_struct_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    dimension: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<nabled::linalg::eigen::NdarrayEigenResult<T::Native>, E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut eigenvalue_values = Vec::with_capacity(batch * dimension);
    let mut eigenvector_values = Vec::with_capacity(batch * dimension * dimension);
    for row in 0..batch {
        let result = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        eigenvalue_values.extend(result.eigenvalues.iter().copied());
        eigenvector_values.extend(result.eigenvectors.iter().copied());
    }
    let eigenvalues = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        batch,
        dimension,
        &eigenvalue_values,
    )?;
    let eigenvalue_field = vector_field("eigenvalues", &T::DATA_TYPE, dimension, false)?;
    let (eigenvector_field, eigenvector_array) =
        tensor_column("eigenvectors", [batch, dimension, dimension], eigenvector_values)?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![eigenvalue_field, eigenvector_field].into(),
        vec![Arc::new(eigenvalues), eigenvector_array],
        None,
    ))))
}

fn invoke_matrix_pair_eigen_struct_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    dimension: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<
        nabled::linalg::eigen::NdarrayGeneralizedEigenResult<T::Native>,
        E,
    >,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let left = expect_fixed_size_list_arg(args, 1, function_name)?;
    let right = expect_fixed_size_list_arg(args, 2, function_name)?;
    let left_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], left, function_name)?;
    let right_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[1], right, function_name)?;
    if left_view.len_of(Axis(0)) != right_view.len_of(Axis(0)) {
        return Err(exec_error(
            function_name,
            format!(
                "batch length mismatch: {} left matrices vs {} right matrices",
                left_view.len_of(Axis(0)),
                right_view.len_of(Axis(0))
            ),
        ));
    }

    let batch = left_view.len_of(Axis(0));
    let mut eigenvalue_values = Vec::with_capacity(batch * dimension);
    let mut eigenvector_values = Vec::with_capacity(batch * dimension * dimension);
    for row in 0..batch {
        let result = op(&left_view.index_axis(Axis(0), row), &right_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        eigenvalue_values.extend(result.eigenvalues.iter().copied());
        eigenvector_values.extend(result.eigenvectors.iter().copied());
    }
    let eigenvalues = fixed_size_list_array_from_flat_rows::<T>(
        function_name,
        batch,
        dimension,
        &eigenvalue_values,
    )?;
    let eigenvalue_field = vector_field("eigenvalues", &T::DATA_TYPE, dimension, false)?;
    let (eigenvector_field, eigenvector_array) =
        tensor_column("eigenvectors", [batch, dimension, dimension], eigenvector_values)?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![eigenvalue_field, eigenvector_field].into(),
        vec![Arc::new(eigenvalues), eigenvector_array],
        None,
    ))))
}

fn invoke_matrix_double_tensor_struct_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    first_name: &str,
    second_name: &str,
    shape: [usize; 2],
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<(Array2<T::Native>, Array2<T::Native>), E>,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut first_values = Vec::with_capacity(batch * shape[0] * shape[1]);
    let mut second_values = Vec::with_capacity(batch * shape[0] * shape[1]);
    for row in 0..batch {
        let (first, second) = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        first_values.extend(first.iter().copied());
        second_values.extend(second.iter().copied());
    }
    let (first_field, first_array) =
        tensor_column(first_name, [batch, shape[0], shape[1]], first_values)?;
    let (second_field, second_array) =
        tensor_column(second_name, [batch, shape[0], shape[1]], second_values)?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![first_field, second_field].into(),
        vec![first_array, second_array],
        None,
    ))))
}

fn invoke_complex_matrix_double_tensor_struct_output<E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    first_name: &str,
    second_name: &str,
    shape: [usize; 2],
    op: impl Fn(
        &ndarray::ArrayView2<'_, Complex64>,
    ) -> std::result::Result<(Array2<Complex64>, Array2<Complex64>), E>,
) -> Result<ColumnarValue>
where
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view =
        complex_fixed_shape_tensor_view3(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut first_values = Vec::with_capacity(batch * shape[0] * shape[1]);
    let mut second_values = Vec::with_capacity(batch * shape[0] * shape[1]);
    for row in 0..batch {
        let (first, second) = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        first_values.extend(first.iter().copied());
        second_values.extend(second.iter().copied());
    }
    let (first_field, first_array) =
        complex_fixed_shape_tensor_array_from_flat_rows(first_name, batch, &shape, first_values)?;
    let (second_field, second_array) =
        complex_fixed_shape_tensor_array_from_flat_rows(second_name, batch, &shape, second_values)?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![first_field, second_field].into(),
        vec![Arc::new(first_array), Arc::new(second_array)],
        None,
    ))))
}

fn invoke_complex_matrix_eigen_struct_output<E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    dimension: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, Complex64>,
    )
        -> std::result::Result<nabled::linalg::eigen::NdarrayNonsymmetricEigenResult<f64>, E>,
) -> Result<ColumnarValue>
where
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view =
        complex_fixed_shape_tensor_view3(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut eigenvalue_values = Vec::with_capacity(batch * dimension);
    let mut schur_vector_values = Vec::with_capacity(batch * dimension * dimension);
    for row in 0..batch {
        let result = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        eigenvalue_values.extend(result.eigenvalues.iter().copied());
        schur_vector_values.extend(result.schur_vectors.iter().copied());
    }
    let eigenvalues = complex_fixed_size_list_array_from_flat_rows(
        function_name,
        batch,
        dimension,
        eigenvalue_values,
    )?;
    let eigenvalue_field = complex_vector_field("eigenvalues", dimension, false)?;
    let (schur_vector_field, schur_vector_array) = complex_fixed_shape_tensor_array_from_flat_rows(
        "schur_vectors",
        batch,
        &[dimension, dimension],
        schur_vector_values,
    )?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![eigenvalue_field, Arc::new(schur_vector_field)].into(),
        vec![Arc::new(eigenvalues), Arc::new(schur_vector_array)],
        None,
    ))))
}

fn to_complex64<T>(value: Complex<T>) -> Complex64
where
    T: Into<f64> + Copy,
{
    Complex64::new(value.re.into(), value.im.into())
}

fn invoke_real_matrix_complex_eigen_struct_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    dimension: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<
        nabled::linalg::eigen::NdarrayNonsymmetricEigenResult<T::Native>,
        E,
    >,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Into<f64>,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut eigenvalue_values = Vec::with_capacity(batch * dimension);
    let mut schur_vector_values = Vec::with_capacity(batch * dimension * dimension);
    for row in 0..batch {
        let result = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        eigenvalue_values.extend(result.eigenvalues.iter().copied().map(to_complex64));
        schur_vector_values.extend(result.schur_vectors.iter().copied().map(to_complex64));
    }
    let eigenvalues = complex_fixed_size_list_array_from_flat_rows(
        function_name,
        batch,
        dimension,
        eigenvalue_values,
    )?;
    let eigenvalue_field = complex_vector_field("eigenvalues", dimension, false)?;
    let (schur_vector_field, schur_vector_array) = complex_fixed_shape_tensor_array_from_flat_rows(
        "schur_vectors",
        batch,
        &[dimension, dimension],
        schur_vector_values,
    )?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![eigenvalue_field, Arc::new(schur_vector_field)].into(),
        vec![Arc::new(eigenvalues), Arc::new(schur_vector_array)],
        None,
    ))))
}

fn invoke_real_matrix_complex_bi_eigen_struct_output<T, E>(
    args: &ScalarFunctionArgs,
    function_name: &str,
    rows: usize,
    cols: usize,
    op: impl Fn(
        &ndarray::ArrayView2<'_, T::Native>,
    ) -> std::result::Result<
        nabled::linalg::eigen::NdarrayNonsymmetricBiEigenResult<T::Native>,
        E,
    >,
) -> Result<ColumnarValue>
where
    T: ArrowPrimitiveType,
    T::Native: NabledReal + NdarrowElement + Into<f64>,
    E: std::fmt::Display,
{
    let matrices = expect_fixed_size_list_arg(args, 1, function_name)?;
    let matrix_view = fixed_shape_tensor_view3::<T>(&args.arg_fields[0], matrices, function_name)?;
    let batch = matrix_view.len_of(Axis(0));
    let mut eigenvalue_values = Vec::with_capacity(batch * rows);
    let mut right_values = Vec::with_capacity(batch * rows * cols);
    let mut left_values = Vec::with_capacity(batch * rows * cols);
    let mut diagonal_values = Vec::with_capacity(batch * rows);
    let mut balanced_values = Vec::with_capacity(batch * rows * cols);
    for row in 0..batch {
        let result = op(&matrix_view.index_axis(Axis(0), row))
            .map_err(|error| exec_error(function_name, error))?;
        eigenvalue_values.extend(result.eigenvalues.iter().copied().map(to_complex64));
        right_values.extend(result.right_eigenvectors.iter().copied().map(to_complex64));
        left_values.extend(result.left_eigenvectors.iter().copied().map(to_complex64));
        diagonal_values.extend(result.balancing_diagonal.iter().copied());
        balanced_values.extend(result.balanced_matrix.iter().copied());
    }
    let eigenvalues = complex_fixed_size_list_array_from_flat_rows(
        function_name,
        batch,
        rows,
        eigenvalue_values,
    )?;
    let right = complex_fixed_shape_tensor_array_from_flat_rows(
        "right_eigenvectors",
        batch,
        &[rows, cols],
        right_values,
    )?;
    let left = complex_fixed_shape_tensor_array_from_flat_rows(
        "left_eigenvectors",
        batch,
        &[rows, cols],
        left_values,
    )?;
    let diagonal =
        fixed_size_list_array_from_flat_rows::<T>(function_name, batch, rows, &diagonal_values)?;
    let (balanced_field, balanced_array) =
        tensor_column("balanced_matrix", [batch, rows, cols], balanced_values)?;
    Ok(ColumnarValue::Array(Arc::new(StructArray::new(
        vec![
            complex_vector_field("eigenvalues", rows, false)?,
            Arc::new(right.0),
            Arc::new(left.0),
            vector_field("balancing_diagonal", &T::DATA_TYPE, rows, false)?,
            balanced_field,
        ]
        .into(),
        vec![
            Arc::new(eigenvalues),
            Arc::new(right.1),
            Arc::new(left.1),
            Arc::new(diagonal),
            balanced_array,
        ],
        None,
    ))))
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
        let (value_type, rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        let l_field = fixed_shape_tensor_field("l", &value_type, &[rows, cols], false)?;
        let u_field = fixed_shape_tensor_field("u", &value_type, &[rows, cols], false)?;
        Ok(struct_field(
            self.name(),
            vec![l_field.as_ref().clone(), u_field.as_ref().clone()],
            nullable,
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                let output = nabled::arrow::batched::lu_f32(args.arg_fields[0].as_ref(), matrices)
                    .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut lower_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut upper_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                for result in output {
                    lower_values.extend(result.l.iter().copied());
                    upper_values.extend(result.u.iter().copied());
                }
                let (lower_field, lower_array) =
                    tensor_column("l", [batch, matrix.rows, matrix.cols], lower_values)?;
                let (upper_field, upper_array) =
                    tensor_column("u", [batch, matrix.rows, matrix.cols], upper_values)?;
                let struct_array = StructArray::new(
                    vec![lower_field, upper_field].into(),
                    vec![lower_array, upper_array],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let output = nabled::arrow::batched::lu_f64(args.arg_fields[0].as_ref(), matrices)
                    .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut lower_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut upper_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                for result in output {
                    lower_values.extend(result.l.iter().copied());
                    upper_values.extend(result.u.iter().copied());
                }
                let (lower_field, lower_array) =
                    tensor_column("l", [batch, matrix.rows, matrix.cols], lower_values)?;
                let (upper_field, upper_array) =
                    tensor_column("u", [batch, matrix.rows, matrix.cols], upper_values)?;
                let struct_array = StructArray::new(
                    vec![lower_field, upper_field].into(),
                    vec![lower_array, upper_array],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let (value_type, rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        fixed_shape_tensor_field(self.name(), &value_type, &[rows, cols], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| nabled::linalg::lu::inverse_view(view),
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| nabled::linalg::lu::inverse_view(view),
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let (value_type, _rows, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        Ok(scalar_field(self.name(), &value_type, nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => invoke_matrix_scalar::<Float32Type, _>(
                &args,
                self.name(),
                nabled::linalg::lu::determinant_view,
            ),
            DataType::Float64 => invoke_matrix_scalar::<Float64Type, _>(
                &args,
                self.name(),
                nabled::linalg::lu::determinant_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let (value_type, _rows, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        let sign_field = Field::new("sign", DataType::Int8, false);
        let log_abs_field = scalar_field("ln_abs_det", &value_type, false);
        Ok(struct_field(self.name(), vec![sign_field, log_abs_field.as_ref().clone()], nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        match matrix.value_type {
            DataType::Float32 => {
                let matrix_view = fixed_shape_tensor_view3::<Float32Type>(
                    &args.arg_fields[0],
                    matrices,
                    self.name(),
                )?;
                let mut signs = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                let mut log_abs = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                for row in 0..matrix_view.len_of(Axis(0)) {
                    let result = nabled::linalg::lu::log_determinant_view(
                        &matrix_view.index_axis(Axis(0), row),
                    )
                    .map_err(|error| exec_error(self.name(), error))?;
                    signs.push(result.sign);
                    log_abs.push(result.ln_abs_det);
                }
                let log_abs = primitive_array_from_values::<Float32Type>(log_abs);
                let struct_array = StructArray::new(
                    vec![
                        Arc::new(Field::new("sign", DataType::Int8, false)),
                        scalar_field("ln_abs_det", &matrix.value_type, false),
                    ]
                    .into(),
                    vec![Arc::new(Int8Array::from(signs)), Arc::new(log_abs)],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let matrix_view = fixed_shape_tensor_view3::<Float64Type>(
                    &args.arg_fields[0],
                    matrices,
                    self.name(),
                )?;
                let mut signs = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                let mut log_abs = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                for row in 0..matrix_view.len_of(Axis(0)) {
                    let result = nabled::linalg::lu::log_determinant_view(
                        &matrix_view.index_axis(Axis(0), row),
                    )
                    .map_err(|error| exec_error(self.name(), error))?;
                    signs.push(result.sign);
                    log_abs.push(result.ln_abs_det);
                }
                let log_abs = primitive_array_from_values::<Float64Type>(log_abs);
                let struct_array = StructArray::new(
                    vec![
                        Arc::new(Field::new("sign", DataType::Int8, false)),
                        scalar_field("ln_abs_det", &matrix.value_type, false),
                    ]
                    .into(),
                    vec![Arc::new(Int8Array::from(signs)), Arc::new(log_abs)],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let (value_type, rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        let l_field = fixed_shape_tensor_field("l", &value_type, &[rows, cols], false)?;
        Ok(struct_field(self.name(), vec![l_field.as_ref().clone()], nullable))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                let output =
                    nabled::arrow::batched::cholesky_f32(args.arg_fields[0].as_ref(), matrices)
                        .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut lower_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                for result in output {
                    lower_values.extend(result.l.iter().copied());
                }
                let (lower_field, lower_array) =
                    tensor_column("l", [batch, matrix.rows, matrix.cols], lower_values)?;
                let struct_array =
                    StructArray::new(vec![lower_field].into(), vec![lower_array], None);
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let output =
                    nabled::arrow::batched::cholesky_f64(args.arg_fields[0].as_ref(), matrices)
                        .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut lower_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                for result in output {
                    lower_values.extend(result.l.iter().copied());
                }
                let (lower_field, lower_array) =
                    tensor_column("l", [batch, matrix.rows, matrix.cols], lower_values)?;
                let struct_array =
                    StructArray::new(vec![lower_field].into(), vec![lower_array], None);
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let (value_type, rows, cols, nullable) = square_matrix_shape(&args, self.name())?;
        fixed_shape_tensor_field(self.name(), &value_type, &[rows, cols], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                nabled::linalg::cholesky::inverse_view,
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                nabled::linalg::cholesky::inverse_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let q_field =
            fixed_shape_tensor_field("q", &matrix.value_type, &[matrix.rows, matrix.cols], false)?;
        let r_field =
            fixed_shape_tensor_field("r", &matrix.value_type, &[matrix.cols, matrix.cols], false)?;
        let rank_field = Field::new("rank", DataType::Int64, false);
        Ok(struct_field(
            self.name(),
            vec![q_field.as_ref().clone(), r_field.as_ref().clone(), rank_field],
            args.arg_fields[0].is_nullable(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                let output = nabled::arrow::batched::qr_f32(
                    args.arg_fields[0].as_ref(),
                    matrices,
                    &nabled::linalg::qr::QRConfig::<f32>::default(),
                )
                .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut q_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut r_values = Vec::with_capacity(batch * matrix.cols * matrix.cols);
                let mut rank_values = Vec::with_capacity(batch);
                for result in output {
                    q_values.extend(result.q.iter().copied());
                    r_values.extend(result.r.iter().copied());
                    rank_values.push(
                        i64::try_from(result.rank)
                            .map_err(|_| exec_error(self.name(), "rank exceeds i64 limits"))?,
                    );
                }
                let (q_field, q_array) =
                    tensor_column("q", [batch, matrix.rows, matrix.cols], q_values)?;
                let (r_field, r_array) =
                    tensor_column("r", [batch, matrix.cols, matrix.cols], r_values)?;
                let struct_array = StructArray::new(
                    vec![q_field, r_field, Arc::new(Field::new("rank", DataType::Int64, false))]
                        .into(),
                    vec![q_array, r_array, Arc::new(Int64Array::from(rank_values))],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let output = nabled::arrow::batched::qr_f64(
                    args.arg_fields[0].as_ref(),
                    matrices,
                    &nabled::linalg::qr::QRConfig::<f64>::default(),
                )
                .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut q_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut r_values = Vec::with_capacity(batch * matrix.cols * matrix.cols);
                let mut rank_values = Vec::with_capacity(batch);
                for result in output {
                    q_values.extend(result.q.iter().copied());
                    r_values.extend(result.r.iter().copied());
                    rank_values.push(
                        i64::try_from(result.rank)
                            .map_err(|_| exec_error(self.name(), "rank exceeds i64 limits"))?,
                    );
                }
                let (q_field, q_array) =
                    tensor_column("q", [batch, matrix.rows, matrix.cols], q_values)?;
                let (r_field, r_array) =
                    tensor_column("r", [batch, matrix.cols, matrix.cols], r_values)?;
                let struct_array = StructArray::new(
                    vec![q_field, r_field, Arc::new(Field::new("rank", DataType::Int64, false))]
                        .into(),
                    vec![q_array, r_array, Arc::new(Int64Array::from(rank_values))],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixQrReduced {
    signature: Signature,
}

impl MatrixQrReduced {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixQrReduced {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_qr_reduced" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = matrix.rows.min(matrix.cols);
        qr_struct_field(
            self.name(),
            &matrix.value_type,
            [matrix.rows, keep],
            [keep, matrix.cols],
            None,
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = matrix.rows.min(matrix.cols);
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_qr_struct_output::<Float32Type, _>(
                &args,
                self.name(),
                [matrix.rows, keep],
                [keep, matrix.cols],
                None,
                |view| {
                    nabled::linalg::qr::decompose_reduced_view(
                        view,
                        &nabled::linalg::qr::QRConfig::<f32>::default(),
                    )
                },
            ),
            DataType::Float64 => invoke_matrix_qr_struct_output::<Float64Type, _>(
                &args,
                self.name(),
                [matrix.rows, keep],
                [keep, matrix.cols],
                None,
                |view| {
                    nabled::linalg::qr::decompose_reduced_view(
                        view,
                        &nabled::linalg::qr::QRConfig::<f64>::default(),
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
struct MatrixQrPivoted {
    signature: Signature,
}

impl MatrixQrPivoted {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixQrPivoted {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_qr_pivoted" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        qr_struct_field(
            self.name(),
            &matrix.value_type,
            [matrix.rows, matrix.cols],
            [matrix.cols, matrix.cols],
            Some([matrix.cols, matrix.cols]),
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_qr_struct_output::<Float32Type, _>(
                &args,
                self.name(),
                [matrix.rows, matrix.cols],
                [matrix.cols, matrix.cols],
                Some([matrix.cols, matrix.cols]),
                |view| {
                    nabled::linalg::qr::decompose_with_pivoting_view(
                        view,
                        &nabled::linalg::qr::QRConfig::<f32>::default(),
                    )
                },
            ),
            DataType::Float64 => invoke_matrix_qr_struct_output::<Float64Type, _>(
                &args,
                self.name(),
                [matrix.rows, matrix.cols],
                [matrix.cols, matrix.cols],
                Some([matrix.cols, matrix.cols]),
                |view| {
                    nabled::linalg::qr::decompose_with_pivoting_view(
                        view,
                        &nabled::linalg::qr::QRConfig::<f64>::default(),
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
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = matrix.rows.min(matrix.cols);
        let u_field =
            fixed_shape_tensor_field("u", &matrix.value_type, &[matrix.rows, keep], false)?;
        let singular_values = vector_field("singular_values", &matrix.value_type, keep, false)?;
        let vt_field =
            fixed_shape_tensor_field("vt", &matrix.value_type, &[keep, matrix.cols], false)?;
        Ok(struct_field(
            self.name(),
            vec![
                u_field.as_ref().clone(),
                singular_values.as_ref().clone(),
                vt_field.as_ref().clone(),
            ],
            args.arg_fields[0].is_nullable(),
        ))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let keep = matrix.rows.min(matrix.cols);
        match matrix.value_type {
            DataType::Float32 => {
                let output = nabled::arrow::batched::svd_f32(args.arg_fields[0].as_ref(), matrices)
                    .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut u_values = Vec::with_capacity(batch * matrix.rows * keep);
                let mut singular_values = Vec::with_capacity(batch * keep);
                let mut vt_values = Vec::with_capacity(batch * keep * matrix.cols);
                for result in output {
                    u_values.extend(result.u.iter().copied());
                    singular_values.extend(result.singular_values.iter().copied());
                    vt_values.extend(result.vt.iter().copied());
                }
                let (u_field, u_array) = tensor_column("u", [batch, matrix.rows, keep], u_values)?;
                let singular_values = fixed_size_list_array_from_flat_rows::<Float32Type>(
                    self.name(),
                    batch,
                    keep,
                    &singular_values,
                )?;
                let singular_field =
                    vector_field("singular_values", &matrix.value_type, keep, false)?;
                let (vt_field, vt_array) =
                    tensor_column("vt", [batch, keep, matrix.cols], vt_values)?;
                let struct_array = StructArray::new(
                    vec![u_field, singular_field, vt_field].into(),
                    vec![u_array, Arc::new(singular_values), vt_array],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            DataType::Float64 => {
                let output = nabled::arrow::batched::svd_f64(args.arg_fields[0].as_ref(), matrices)
                    .map_err(|error| exec_error(self.name(), error))?;
                let batch = output.len();
                let mut u_values = Vec::with_capacity(batch * matrix.rows * keep);
                let mut singular_values = Vec::with_capacity(batch * keep);
                let mut vt_values = Vec::with_capacity(batch * keep * matrix.cols);
                for result in output {
                    u_values.extend(result.u.iter().copied());
                    singular_values.extend(result.singular_values.iter().copied());
                    vt_values.extend(result.vt.iter().copied());
                }
                let (u_field, u_array) = tensor_column("u", [batch, matrix.rows, keep], u_values)?;
                let singular_values = fixed_size_list_array_from_flat_rows::<Float64Type>(
                    self.name(),
                    batch,
                    keep,
                    &singular_values,
                )?;
                let singular_field =
                    vector_field("singular_values", &matrix.value_type, keep, false)?;
                let (vt_field, vt_array) =
                    tensor_column("vt", [batch, keep, matrix.cols], vt_values)?;
                let struct_array = StructArray::new(
                    vec![u_field, singular_field, vt_field].into(),
                    vec![u_array, Arc::new(singular_values), vt_array],
                    None,
                );
                Ok(ColumnarValue::Array(Arc::new(struct_array)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdTruncated {
    signature: Signature,
}

impl MatrixSvdTruncated {
    fn new() -> Self { Self { signature: named_user_defined_signature(&["matrix", "k"]) } }
}

impl ScalarUDFImpl for MatrixSvdTruncated {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_truncated" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[(2, ScalarCoercion::Integer)])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let k = validate_positive_usize(
            self.name(),
            "k",
            expect_usize_scalar_argument(&args, 2, self.name())?,
        )?;
        let keep = k.min(matrix.rows.min(matrix.cols));
        svd_struct_field(
            self.name(),
            &matrix.value_type,
            [matrix.rows, keep],
            keep,
            [keep, matrix.cols],
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let k = validate_positive_usize(
            self.name(),
            "k",
            expect_usize_scalar_arg(&args, 2, self.name())?,
        )?;
        let keep = k.min(matrix.rows.min(matrix.cols));
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_svd_struct_output::<Float32Type, _>(
                &args,
                self.name(),
                [matrix.rows, keep],
                keep,
                [keep, matrix.cols],
                |view| nabled::linalg::svd::decompose_truncated_view(view, k),
            ),
            DataType::Float64 => invoke_matrix_svd_struct_output::<Float64Type, _>(
                &args,
                self.name(),
                [matrix.rows, keep],
                keep,
                [keep, matrix.cols],
                |view| nabled::linalg::svd::decompose_truncated_view(view, k),
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            decomposition_doc(
                "Compute a truncated singular value decomposition for each matrix in the batch.",
                "matrix_svd_truncated(matrix_batch, k => 8)",
            )
            .with_argument("matrix", "Dense matrix batch in canonical fixed-shape tensor form.")
            .with_argument("k", "Positive integer target rank for the truncated decomposition.")
            .with_alternative_syntax("matrix_svd_truncated(matrix => matrix_batch, k => 8)")
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdWithTolerance {
    signature: Signature,
}

impl MatrixSvdWithTolerance {
    fn new() -> Self { Self { signature: named_user_defined_signature(&["matrix", "tolerance"]) } }
}

impl ScalarUDFImpl for MatrixSvdWithTolerance {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_with_tolerance" }

    fn signature(&self) -> &Signature { &self.signature }

    fn coerce_types(&self, arg_types: &[DataType]) -> Result<Vec<DataType>> {
        coerce_scalar_arguments(self.name(), arg_types, &[(2, ScalarCoercion::Real)])
    }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let _ = validate_non_negative_scalar(
            self.name(),
            "tolerance",
            expect_real_scalar_argument(&args, 2, self.name())?,
        )?;
        let keep = matrix.rows.min(matrix.cols);
        svd_struct_field(
            self.name(),
            &matrix.value_type,
            [matrix.rows, keep],
            keep,
            [keep, matrix.cols],
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let tolerance = validate_non_negative_scalar(
            self.name(),
            "tolerance",
            expect_real_scalar_arg(&args, 2, self.name())?,
        )?;
        let keep = matrix.rows.min(matrix.cols);
        match matrix.value_type {
            DataType::Float32 => {
                let tolerance = native_scalar::<f32>(self.name(), "tolerance", tolerance)?;
                invoke_matrix_svd_struct_output::<Float32Type, _>(
                    &args,
                    self.name(),
                    [matrix.rows, keep],
                    keep,
                    [keep, matrix.cols],
                    |view| nabled::linalg::svd::decompose_with_tolerance_view(view, tolerance),
                )
            }
            DataType::Float64 => {
                let tolerance = native_scalar::<f64>(self.name(), "tolerance", tolerance)?;
                invoke_matrix_svd_struct_output::<Float64Type, _>(
                    &args,
                    self.name(),
                    [matrix.rows, keep],
                    keep,
                    [keep, matrix.cols],
                    |view| nabled::linalg::svd::decompose_with_tolerance_view(view, tolerance),
                )
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            decomposition_doc(
                "Compute a tolerance-thresholded singular value decomposition for each matrix in \
                 the batch.",
                "matrix_svd_with_tolerance(matrix_batch, tolerance => 1e-6)",
            )
            .with_argument("matrix", "Dense matrix batch in canonical fixed-shape tensor form.")
            .with_argument("tolerance", "Finite non-negative singular-value threshold.")
            .with_alternative_syntax(
                "matrix_svd_with_tolerance(matrix => matrix_batch, tolerance => 1e-6)",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdNullSpace {
    signature: Signature,
}

impl MatrixSvdNullSpace {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvdNullSpace {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_null_space" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let cols = i32::try_from(matrix.cols)
            .map_err(|_| exec_error(self.name(), "matrix column count exceeds Arrow i32 limits"))?;
        let uniform_shape = [Some(cols), None];
        variable_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            2,
            Some(&uniform_shape),
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let cols = i32::try_from(matrix.cols)
            .map_err(|_| exec_error(self.name(), "matrix column count exceeds Arrow i32 limits"))?;
        match matrix.value_type {
            DataType::Float32 => {
                let matrix_view = fixed_shape_tensor_view3::<Float32Type>(
                    &args.arg_fields[0],
                    matrices,
                    self.name(),
                )?;
                let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                for row in 0..matrix_view.len_of(Axis(0)) {
                    output.push(
                        nabled::linalg::svd::null_space_view(
                            &matrix_view.index_axis(Axis(0), row),
                            None,
                        )
                        .map_err(|error| exec_error(self.name(), error))?
                        .into_dyn(),
                    );
                }
                let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(
                    self.name(),
                    output,
                    Some(vec![Some(cols), None]),
                )
                .map_err(|error| exec_error(self.name(), error))?;
                Ok(ColumnarValue::Array(Arc::new(output)))
            }
            DataType::Float64 => {
                let matrix_view = fixed_shape_tensor_view3::<Float64Type>(
                    &args.arg_fields[0],
                    matrices,
                    self.name(),
                )?;
                let mut output = Vec::with_capacity(matrix_view.len_of(Axis(0)));
                for row in 0..matrix_view.len_of(Axis(0)) {
                    output.push(
                        nabled::linalg::svd::null_space_view(
                            &matrix_view.index_axis(Axis(0), row),
                            None,
                        )
                        .map_err(|error| exec_error(self.name(), error))?
                        .into_dyn(),
                    );
                }
                let (_field, output) = ndarrow::arrays_to_variable_shape_tensor(
                    self.name(),
                    output,
                    Some(vec![Some(cols), None]),
                )
                .map_err(|error| exec_error(self.name(), error))?;
                Ok(ColumnarValue::Array(Arc::new(output)))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixQrSolveLeastSquares {
    signature: Signature,
}

impl MatrixQrSolveLeastSquares {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixQrSolveLeastSquares {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_qr_solve_least_squares" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let rhs = parse_vector_field(&args.arg_fields[1], self.name(), 2)?;
        if matrix.value_type != rhs.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "value type mismatch: matrix {}, rhs {}",
                    matrix.value_type, rhs.value_type
                ),
            ));
        }
        if rhs.len != matrix.rows {
            return Err(exec_error(
                self.name(),
                format!("rhs vector length mismatch: expected {}, found {}", matrix.rows, rhs.len),
            ));
        }
        vector_field(self.name(), &matrix.value_type, matrix.cols, nullable_or(args.arg_fields))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => {
                invoke_matrix_qr_vector_output::<Float32Type, _>(&args, self.name(), |m, rhs| {
                    nabled::linalg::qr::solve_least_squares_view(
                        m,
                        rhs,
                        &nabled::linalg::qr::QRConfig::<f32>::default(),
                    )
                })
            }
            DataType::Float64 => {
                invoke_matrix_qr_vector_output::<Float64Type, _>(&args, self.name(), |m, rhs| {
                    nabled::linalg::qr::solve_least_squares_view(
                        m,
                        rhs,
                        &nabled::linalg::qr::QRConfig::<f64>::default(),
                    )
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixQrConditionNumber {
    signature: Signature,
}

impl MatrixQrConditionNumber {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixQrConditionNumber {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_qr_condition_number" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        Ok(scalar_field(self.name(), &matrix.value_type, args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => {
                invoke_matrix_scalar::<Float32Type, _>(&args, self.name(), |matrix| {
                    let qr = nabled::linalg::qr::decompose_view(
                        matrix,
                        &nabled::linalg::qr::QRConfig::<f32>::default(),
                    )
                    .map_err(|error| exec_error("matrix_qr_condition_number", error))?;
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::linalg::qr::condition_number(&qr),
                    )
                })
            }
            DataType::Float64 => {
                invoke_matrix_scalar::<Float64Type, _>(&args, self.name(), |matrix| {
                    let qr = nabled::linalg::qr::decompose_view(
                        matrix,
                        &nabled::linalg::qr::QRConfig::<f64>::default(),
                    )
                    .map_err(|error| exec_error("matrix_qr_condition_number", error))?;
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::linalg::qr::condition_number(&qr),
                    )
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixQrReconstruct {
    signature: Signature,
}

impl MatrixQrReconstruct {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixQrReconstruct {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_qr_reconstruct" }

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
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| {
                    let qr = nabled::linalg::qr::decompose_view(
                        view,
                        &nabled::linalg::qr::QRConfig::<f32>::default(),
                    )?;
                    Ok::<_, nabled::linalg::qr::QRError>(nabled::linalg::qr::reconstruct_matrix(
                        &qr,
                    ))
                },
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| {
                    let qr = nabled::linalg::qr::decompose_view(
                        view,
                        &nabled::linalg::qr::QRConfig::<f64>::default(),
                    )?;
                    Ok::<_, nabled::linalg::qr::QRError>(nabled::linalg::qr::reconstruct_matrix(
                        &qr,
                    ))
                },
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdPseudoInverse {
    signature: Signature,
}

impl MatrixSvdPseudoInverse {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvdPseudoInverse {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_pseudo_inverse" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        fixed_shape_tensor_field(
            self.name(),
            &matrix.value_type,
            &[matrix.cols, matrix.rows],
            args.arg_fields[0].is_nullable(),
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.cols,
                matrix.rows,
                |m| {
                    nabled::linalg::svd::pseudo_inverse_view(
                        m,
                        &nabled::linalg::svd::PseudoInverseConfig::<f32>::default(),
                    )
                },
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.cols,
                matrix.rows,
                |m| {
                    nabled::linalg::svd::pseudo_inverse_view(
                        m,
                        &nabled::linalg::svd::PseudoInverseConfig::<f64>::default(),
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
struct MatrixSvdReconstruct {
    signature: Signature,
}

impl MatrixSvdReconstruct {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvdReconstruct {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_reconstruct" }

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
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| {
                    let svd = nabled::linalg::svd::decompose_view(view)?;
                    Ok::<_, nabled::linalg::svd::SVDError>(nabled::linalg::svd::reconstruct_matrix(
                        &svd,
                    ))
                },
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                |view| {
                    let svd = nabled::linalg::svd::decompose_view(view)?;
                    Ok::<_, nabled::linalg::svd::SVDError>(nabled::linalg::svd::reconstruct_matrix(
                        &svd,
                    ))
                },
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdConditionNumber {
    signature: Signature,
}

impl MatrixSvdConditionNumber {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvdConditionNumber {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_condition_number" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        Ok(scalar_field(self.name(), &matrix.value_type, args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => {
                invoke_matrix_scalar::<Float32Type, _>(&args, self.name(), |matrix| {
                    let svd = nabled::linalg::svd::decompose_view(matrix)
                        .map_err(|error| exec_error("matrix_svd_condition_number", error))?;
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::linalg::svd::condition_number(&svd),
                    )
                })
            }
            DataType::Float64 => {
                invoke_matrix_scalar::<Float64Type, _>(&args, self.name(), |matrix| {
                    let svd = nabled::linalg::svd::decompose_view(matrix)
                        .map_err(|error| exec_error("matrix_svd_condition_number", error))?;
                    Ok::<_, datafusion::common::DataFusionError>(
                        nabled::linalg::svd::condition_number(&svd),
                    )
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSvdRank {
    signature: Signature,
}

impl MatrixSvdRank {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSvdRank {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_svd_rank" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        Ok(int64_scalar_field(self.name(), args.arg_fields[0].is_nullable()))
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        match parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?.value_type {
            DataType::Float32 => {
                invoke_matrix_scalar_i64::<Float32Type, _>(&args, self.name(), |matrix| {
                    let svd = nabled::linalg::svd::decompose_view(matrix)
                        .map_err(|error| exec_error("matrix_svd_rank", error))?;
                    i64::try_from(nabled::linalg::svd::rank(&svd, None))
                        .map_err(|_| exec_error("matrix_svd_rank", "rank exceeds i64 limits"))
                })
            }
            DataType::Float64 => {
                invoke_matrix_scalar_i64::<Float64Type, _>(&args, self.name(), |matrix| {
                    let svd = nabled::linalg::svd::decompose_view(matrix)
                        .map_err(|error| exec_error("matrix_svd_rank", error))?;
                    i64::try_from(nabled::linalg::svd::rank(&svd, None))
                        .map_err(|_| exec_error("matrix_svd_rank", "rank exceeds i64 limits"))
                })
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixEigenSymmetric {
    signature: Signature,
}

impl MatrixEigenSymmetric {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixEigenSymmetric {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_eigen_symmetric" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, dimension, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        eigen_struct_field(self.name(), &value_type, dimension, nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_eigen_struct_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                nabled::linalg::eigen::symmetric_view,
            ),
            DataType::Float64 => invoke_matrix_eigen_struct_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                nabled::linalg::eigen::symmetric_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixEigenGeneralized {
    signature: Signature,
}

impl MatrixEigenGeneralized {
    fn new() -> Self { Self { signature: any_signature(2) } }
}

impl ScalarUDFImpl for MatrixEigenGeneralized {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_eigen_generalized" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, dimension, nullable) = square_matrix_pair_shape(&args, self.name())?;
        eigen_struct_field(self.name(), &value_type, dimension, nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let left = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        let right = parse_matrix_batch_field(&args.arg_fields[1], self.name(), 2)?;
        if left.rows != left.cols || right.rows != right.cols {
            return Err(exec_error(
                self.name(),
                "matrix_eigen_generalized requires square matrices",
            ));
        }
        if left.value_type != right.value_type {
            return Err(exec_error(
                self.name(),
                format!(
                    "matrix value type mismatch: left {}, right {}",
                    left.value_type, right.value_type
                ),
            ));
        }
        if left.rows != right.rows || left.cols != right.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "matrix shape mismatch: left ({}, {}), right ({}, {})",
                    left.rows, left.cols, right.rows, right.cols
                ),
            ));
        }
        match left.value_type {
            DataType::Float32 => invoke_matrix_pair_eigen_struct_output::<Float32Type, _>(
                &args,
                self.name(),
                left.rows,
                nabled::linalg::eigen::generalized_view,
            ),
            DataType::Float64 => invoke_matrix_pair_eigen_struct_output::<Float64Type, _>(
                &args,
                self.name(),
                left.rows,
                nabled::linalg::eigen::generalized_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixBalanceNonsymmetric {
    signature: Signature,
}

impl MatrixBalanceNonsymmetric {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixBalanceNonsymmetric {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_balance_nonsymmetric" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, dimension, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        tensor_vector_struct_field(
            self.name(),
            &value_type,
            "balanced",
            [dimension, dimension],
            "diagonal",
            dimension,
            nullable,
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        let matrices = expect_fixed_size_list_arg(&args, 1, self.name())?;
        match matrix.value_type {
            DataType::Float32 => {
                let matrix_view = fixed_shape_tensor_view3::<Float32Type>(
                    &args.arg_fields[0],
                    matrices,
                    self.name(),
                )?;
                let batch = matrix_view.len_of(Axis(0));
                let mut balanced_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut diagonal_values = Vec::with_capacity(batch * matrix.rows);
                let config = nabled::linalg::eigen::NonsymmetricEigenConfig::<f32>::default();
                for row in 0..batch {
                    let (balanced, diagonal) = nabled::linalg::eigen::balance_nonsymmetric_view(
                        &matrix_view.index_axis(Axis(0), row),
                        &config,
                    )
                    .map_err(|error| exec_error(self.name(), error))?;
                    balanced_values.extend(balanced.iter().copied());
                    diagonal_values.extend(diagonal.iter().copied());
                }
                let (balanced_field, balanced_array) =
                    tensor_column("balanced", [batch, matrix.rows, matrix.cols], balanced_values)?;
                let diagonal_array = fixed_size_list_array_from_flat_rows::<Float32Type>(
                    self.name(),
                    batch,
                    matrix.rows,
                    &diagonal_values,
                )?;
                let diagonal_field =
                    vector_field("diagonal", &matrix.value_type, matrix.rows, false)?;
                Ok(ColumnarValue::Array(Arc::new(StructArray::new(
                    vec![balanced_field, diagonal_field].into(),
                    vec![balanced_array, Arc::new(diagonal_array)],
                    None,
                ))))
            }
            DataType::Float64 => {
                let matrix_view = fixed_shape_tensor_view3::<Float64Type>(
                    &args.arg_fields[0],
                    matrices,
                    self.name(),
                )?;
                let batch = matrix_view.len_of(Axis(0));
                let mut balanced_values = Vec::with_capacity(batch * matrix.rows * matrix.cols);
                let mut diagonal_values = Vec::with_capacity(batch * matrix.rows);
                let config = nabled::linalg::eigen::NonsymmetricEigenConfig::<f64>::default();
                for row in 0..batch {
                    let (balanced, diagonal) = nabled::linalg::eigen::balance_nonsymmetric_view(
                        &matrix_view.index_axis(Axis(0), row),
                        &config,
                    )
                    .map_err(|error| exec_error(self.name(), error))?;
                    balanced_values.extend(balanced.iter().copied());
                    diagonal_values.extend(diagonal.iter().copied());
                }
                let (balanced_field, balanced_array) =
                    tensor_column("balanced", [batch, matrix.rows, matrix.cols], balanced_values)?;
                let diagonal_array = fixed_size_list_array_from_flat_rows::<Float64Type>(
                    self.name(),
                    batch,
                    matrix.rows,
                    &diagonal_values,
                )?;
                let diagonal_field =
                    vector_field("diagonal", &matrix.value_type, matrix.rows, false)?;
                Ok(ColumnarValue::Array(Arc::new(StructArray::new(
                    vec![balanced_field, diagonal_field].into(),
                    vec![balanced_array, Arc::new(diagonal_array)],
                    None,
                ))))
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixEigenNonsymmetric {
    signature: Signature,
}

impl MatrixEigenNonsymmetric {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixEigenNonsymmetric {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_eigen_nonsymmetric" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (_value_type, dimension, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        complex_eigen_struct_field(self.name(), dimension, nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        match matrix.value_type {
            DataType::Float32 => invoke_real_matrix_complex_eigen_struct_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                nabled::linalg::eigen::nonsymmetric_view,
            ),
            DataType::Float64 => invoke_real_matrix_complex_eigen_struct_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                nabled::linalg::eigen::nonsymmetric_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            decomposition_doc(
                "Compute a non-symmetric eigen decomposition for each real square matrix in the \
                 batch.",
                "matrix_eigen_nonsymmetric(matrix_batch)",
            )
            .with_argument(
                "matrix_batch",
                "Square dense matrix batch in canonical fixed-shape tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixEigenNonsymmetricBi {
    signature: Signature,
}

impl MatrixEigenNonsymmetricBi {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixEigenNonsymmetricBi {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_eigen_nonsymmetric_bi" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, dimension, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        real_input_complex_bi_eigen_struct_field(self.name(), &value_type, dimension, nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        match matrix.value_type {
            DataType::Float32 => {
                invoke_real_matrix_complex_bi_eigen_struct_output::<Float32Type, _>(
                    &args,
                    self.name(),
                    matrix.rows,
                    matrix.cols,
                    |view| {
                        let config =
                            nabled::linalg::eigen::NonsymmetricEigenConfig::<f32>::default();
                        nabled::linalg::eigen::nonsymmetric_bi_view(view, &config)
                    },
                )
            }
            DataType::Float64 => {
                invoke_real_matrix_complex_bi_eigen_struct_output::<Float64Type, _>(
                    &args,
                    self.name(),
                    matrix.rows,
                    matrix.cols,
                    |view| {
                        let config =
                            nabled::linalg::eigen::NonsymmetricEigenConfig::<f64>::default();
                        nabled::linalg::eigen::nonsymmetric_bi_view(view, &config)
                    },
                )
            }
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            decomposition_doc(
                "Compute matched left and right non-symmetric eigenvectors for each real square \
                 matrix in the batch.",
                "matrix_eigen_nonsymmetric_bi(matrix_batch)",
            )
            .with_argument(
                "matrix_batch",
                "Square dense matrix batch in canonical fixed-shape tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixEigenNonsymmetricComplex {
    signature: Signature,
}

impl MatrixEigenNonsymmetricComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixEigenNonsymmetricComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_eigen_nonsymmetric_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (dimension, nullable) = complex_square_matrix_shape(&args, self.name())?;
        complex_eigen_struct_field(self.name(), dimension, nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        invoke_complex_matrix_eigen_struct_output(&args, self.name(), matrix.rows, |view| {
            nabled::linalg::eigen::nonsymmetric_complex_view(view)
        })
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            decomposition_doc(
                "Compute a complex non-symmetric eigen decomposition for each complex square \
                 matrix in the batch.",
                "matrix_eigen_nonsymmetric_complex(matrix_batch)",
            )
            .with_argument(
                "matrix",
                "Complex square matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSchur {
    signature: Signature,
}

impl MatrixSchur {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSchur {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_schur" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, dimension, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        double_tensor_struct_field(
            self.name(),
            &value_type,
            "q",
            "t",
            [dimension, dimension],
            nullable,
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_double_tensor_struct_output::<Float32Type, _>(
                &args,
                self.name(),
                "q",
                "t",
                [matrix.rows, matrix.cols],
                |view| {
                    nabled::linalg::schur::compute_schur_view(view)
                        .map(|result| (result.q, result.t))
                },
            ),
            DataType::Float64 => invoke_matrix_double_tensor_struct_output::<Float64Type, _>(
                &args,
                self.name(),
                "q",
                "t",
                [matrix.rows, matrix.cols],
                |view| {
                    nabled::linalg::schur::compute_schur_view(view)
                        .map(|result| (result.q, result.t))
                },
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixPolar {
    signature: Signature,
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixSchurComplex {
    signature: Signature,
}

impl MatrixSchurComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixSchurComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_schur_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (dimension, nullable) = complex_square_matrix_shape(&args, self.name())?;
        complex_double_tensor_struct_field(self.name(), "q", "t", [dimension, dimension], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        invoke_complex_matrix_double_tensor_struct_output(
            &args,
            self.name(),
            "q",
            "t",
            [matrix.rows, matrix.cols],
            |view| {
                nabled::linalg::schur::compute_schur_complex_view(view)
                    .map(|result| (result.q, result.t))
            },
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            decomposition_doc(
                "Compute a complex Schur decomposition for each complex square matrix in the \
                 batch.",
                "matrix_schur_complex(matrix_batch)",
            )
            .with_argument(
                "matrix",
                "Complex square matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

impl MatrixPolar {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixPolar {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_polar" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (value_type, dimension, _cols, nullable) = square_matrix_shape(&args, self.name())?;
        double_tensor_struct_field(
            self.name(),
            &value_type,
            "u",
            "p",
            [dimension, dimension],
            nullable,
        )
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        match matrix.value_type {
            DataType::Float32 => invoke_matrix_double_tensor_struct_output::<Float32Type, _>(
                &args,
                self.name(),
                "u",
                "p",
                [matrix.rows, matrix.cols],
                |view| {
                    nabled::linalg::polar::compute_polar_view(view)
                        .map(|result| (result.u, result.p))
                },
            ),
            DataType::Float64 => invoke_matrix_double_tensor_struct_output::<Float64Type, _>(
                &args,
                self.name(),
                "u",
                "p",
                [matrix.rows, matrix.cols],
                |view| {
                    nabled::linalg::polar::compute_polar_view(view)
                        .map(|result| (result.u, result.p))
                },
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixPolarComplex {
    signature: Signature,
}

impl MatrixPolarComplex {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixPolarComplex {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_polar_complex" }

    fn signature(&self) -> &Signature { &self.signature }

    fn return_type(&self, _arg_types: &[DataType]) -> Result<DataType> {
        datafusion::common::internal_err!("return_field_from_args should be used instead")
    }

    fn return_field_from_args(&self, args: ReturnFieldArgs<'_>) -> Result<FieldRef> {
        let (dimension, nullable) = complex_square_matrix_shape(&args, self.name())?;
        complex_double_tensor_struct_field(self.name(), "u", "p", [dimension, dimension], nullable)
    }

    fn invoke_with_args(&self, args: ScalarFunctionArgs) -> Result<ColumnarValue> {
        let matrix = parse_complex_matrix_batch_field(&args.arg_fields[0], self.name(), 1)?;
        if matrix.rows != matrix.cols {
            return Err(exec_error(
                self.name(),
                format!(
                    "{} requires square matrices, found ({}, {})",
                    self.name(),
                    matrix.rows,
                    matrix.cols
                ),
            ));
        }
        invoke_complex_matrix_double_tensor_struct_output(
            &args,
            self.name(),
            "u",
            "p",
            [matrix.rows, matrix.cols],
            |view| {
                nabled::linalg::polar::compute_polar_complex_view(view)
                    .map(|result| (result.u, result.p))
            },
        )
    }

    fn documentation(&self) -> Option<&Documentation> {
        static DOCUMENTATION: LazyLock<Documentation> = LazyLock::new(|| {
            decomposition_doc(
                "Compute a complex polar decomposition for each complex square matrix in the \
                 batch.",
                "matrix_polar_complex(matrix_batch)",
            )
            .with_argument(
                "matrix",
                "Complex square matrix batch in canonical arrow.fixed_shape_tensor form.",
            )
            .build()
        });
        Some(&DOCUMENTATION)
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixGramSchmidt {
    signature: Signature,
}

impl MatrixGramSchmidt {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixGramSchmidt {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_gram_schmidt" }

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
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                nabled::linalg::orthogonalization::gram_schmidt_view,
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                nabled::linalg::orthogonalization::gram_schmidt_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
    }
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct MatrixGramSchmidtClassic {
    signature: Signature,
}

impl MatrixGramSchmidtClassic {
    fn new() -> Self { Self { signature: any_signature(1) } }
}

impl ScalarUDFImpl for MatrixGramSchmidtClassic {
    fn as_any(&self) -> &dyn Any { self }

    fn name(&self) -> &'static str { "matrix_gram_schmidt_classic" }

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
            DataType::Float32 => invoke_matrix_tensor_output::<Float32Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                nabled::linalg::orthogonalization::gram_schmidt_classic_view,
            ),
            DataType::Float64 => invoke_matrix_tensor_output::<Float64Type, _>(
                &args,
                self.name(),
                matrix.rows,
                matrix.cols,
                nabled::linalg::orthogonalization::gram_schmidt_classic_view,
            ),
            actual => {
                Err(exec_error(self.name(), format!("unsupported matrix value type {actual}")))
            }
        }
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
pub fn matrix_qr_reduced_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixQrReduced::new()))
}

#[must_use]
pub fn matrix_qr_pivoted_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixQrPivoted::new()))
}

#[must_use]
pub fn matrix_svd_udf() -> Arc<ScalarUDF> { Arc::new(ScalarUDF::new_from_impl(MatrixSvd::new())) }

#[must_use]
pub fn matrix_svd_truncated_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdTruncated::new()))
}

#[must_use]
pub fn matrix_svd_with_tolerance_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdWithTolerance::new()))
}

#[must_use]
pub fn matrix_svd_null_space_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdNullSpace::new()))
}

#[must_use]
pub fn matrix_qr_solve_least_squares_udf() -> Arc<ScalarUDF> {
    Arc::new(
        ScalarUDF::new_from_impl(MatrixQrSolveLeastSquares::new())
            .with_aliases(["matrix_qr_solve_ls"]),
    )
}

#[must_use]
pub fn matrix_qr_condition_number_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixQrConditionNumber::new()))
}

#[must_use]
pub fn matrix_qr_reconstruct_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixQrReconstruct::new()))
}

#[must_use]
pub fn matrix_svd_pseudo_inverse_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdPseudoInverse::new()))
}

#[must_use]
pub fn matrix_svd_reconstruct_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdReconstruct::new()))
}

#[must_use]
pub fn matrix_svd_condition_number_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdConditionNumber::new()))
}

#[must_use]
pub fn matrix_svd_rank_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSvdRank::new()))
}

#[must_use]
pub fn matrix_eigen_symmetric_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixEigenSymmetric::new()))
}

#[must_use]
pub fn matrix_eigen_generalized_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixEigenGeneralized::new()))
}

#[must_use]
pub fn matrix_balance_nonsymmetric_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixBalanceNonsymmetric::new()))
}

#[must_use]
pub fn matrix_eigen_nonsymmetric_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixEigenNonsymmetric::new()))
}

#[must_use]
pub fn matrix_eigen_nonsymmetric_bi_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixEigenNonsymmetricBi::new()))
}

#[must_use]
pub fn matrix_eigen_nonsymmetric_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixEigenNonsymmetricComplex::new()))
}

#[must_use]
pub fn matrix_schur_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSchur::new()))
}

#[must_use]
pub fn matrix_schur_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixSchurComplex::new()))
}

#[must_use]
pub fn matrix_polar_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixPolar::new()))
}

#[must_use]
pub fn matrix_polar_complex_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixPolarComplex::new()))
}

#[must_use]
pub fn matrix_gram_schmidt_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixGramSchmidt::new()))
}

#[must_use]
pub fn matrix_gram_schmidt_classic_udf() -> Arc<ScalarUDF> {
    Arc::new(ScalarUDF::new_from_impl(MatrixGramSchmidtClassic::new()))
}
