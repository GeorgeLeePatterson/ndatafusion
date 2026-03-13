use std::sync::Arc;

use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::array::{
    Array, FixedSizeListArray, Float64Array, Int8Array, Int64Array, StructArray,
};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::ScalarValue;
use datafusion::logical_expr::ColumnarValue;
use ndarray::{Array1, Array2, Array3, Array4, Ix1, Ix2, Ix3, Ix4};

use crate::metadata::vector_field;
use crate::udf::common::invoke_udf;
use crate::udfs;

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(delta < 1.0e-9, "expected {expected}, got {actual}, delta {delta}");
}

fn fixed_size_list<const R: usize, const C: usize>(rows: [[f64; C]; R]) -> FixedSizeListArray {
    FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
        i32::try_from(C).expect("fixed-size-list width should fit i32"),
    )
}

fn matrix_batch<const B: usize, const R: usize, const C: usize>(
    name: &str,
    values: [[[f64; C]; R]; B],
) -> (FieldRef, FixedSizeListArray) {
    let values = values.into_iter().flatten().flatten().collect::<Vec<f64>>();
    let array = Array3::from_shape_vec((B, R, C), values).expect("matrix batch shape");
    let (field, array) =
        ndarrow::arrayd_to_fixed_shape_tensor(name, array.into_dyn()).expect("matrix batch");
    (Arc::new(field), array)
}

fn ragged_vectors(name: &str, rows: Vec<Vec<f64>>) -> (FieldRef, StructArray) {
    let rows = rows
        .into_iter()
        .map(Array1::from_vec)
        .map(ndarray::ArrayBase::into_dyn)
        .collect::<Vec<_>>();
    let (field, array) =
        ndarrow::arrays_to_variable_shape_tensor(name, rows, Some(vec![None])).expect("ragged");
    (Arc::new(field), array)
}

fn ragged_matrices(name: &str, rows: Vec<Vec<Vec<f64>>>) -> (FieldRef, StructArray) {
    let rows = rows
        .into_iter()
        .map(|matrix| {
            let row_count = matrix.len();
            let col_count = matrix.first().map_or(0, Vec::len);
            let values = matrix.into_iter().flatten().collect::<Vec<_>>();
            Array2::from_shape_vec((row_count, col_count), values)
                .expect("matrix batch shape")
                .into_dyn()
        })
        .collect::<Vec<_>>();
    let (field, array) =
        ndarrow::arrays_to_variable_shape_tensor(name, rows, Some(vec![None, None]))
            .expect("ragged matrices");
    (Arc::new(field), array)
}

fn tensor_batch4<const B: usize, const D0: usize, const D1: usize, const D2: usize>(
    name: &str,
    values: [[[[f64; D2]; D1]; D0]; B],
) -> (FieldRef, FixedSizeListArray) {
    let values = values.into_iter().flatten().flatten().flatten().collect::<Vec<f64>>();
    let array = Array4::from_shape_vec((B, D0, D1, D2), values).expect("tensor batch shape");
    let (field, array) =
        ndarrow::arrayd_to_fixed_shape_tensor(name, array.into_dyn()).expect("tensor batch");
    (Arc::new(field), array)
}

fn sparse_batch(name: &str) -> (FieldRef, StructArray) {
    let (field, array) = ndarrow::csr_batch_to_extension_array(
        name,
        vec![[2, 3], [2, 2]],
        vec![vec![0, 2, 3], vec![0, 1, 3]],
        vec![vec![0, 2, 1], vec![0, 0, 1]],
        vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    )
    .expect("sparse batch");
    (Arc::new(field), array)
}

fn sparse_batch_rhs(name: &str) -> (FieldRef, StructArray) {
    let (field, array) = ndarrow::csr_batch_to_extension_array(
        name,
        vec![[3, 2], [2, 2]],
        vec![vec![0, 1, 2, 4], vec![0, 1, 2]],
        vec![vec![0, 1, 0, 1], vec![0, 1]],
        vec![vec![1.0, 1.0, 1.0, 1.0], vec![1.0, 1.0]],
    )
    .expect("sparse rhs batch");
    (Arc::new(field), array)
}

fn f64_array(values: &ColumnarValue) -> &Float64Array {
    let ColumnarValue::Array(array) = values else {
        panic!("expected array output");
    };
    array.as_any().downcast_ref::<Float64Array>().expect("expected Float64Array")
}

fn fixed_size_list_array(values: &ColumnarValue) -> &FixedSizeListArray {
    let ColumnarValue::Array(array) = values else {
        panic!("expected array output");
    };
    array.as_any().downcast_ref::<FixedSizeListArray>().expect("expected FixedSizeListArray")
}

fn struct_array(values: &ColumnarValue) -> &StructArray {
    let ColumnarValue::Array(array) = values else {
        panic!("expected array output");
    };
    array.as_any().downcast_ref::<StructArray>().expect("expected StructArray")
}

fn fixed_shape_view3<'a>(
    field: &'a FieldRef,
    values: &'a ColumnarValue,
) -> ndarray::ArrayView3<'a, f64> {
    fixed_shape_viewd(field, values).into_dimensionality::<Ix3>().expect("rank-3 tensor")
}

fn fixed_shape_viewd<'a>(
    field: &'a FieldRef,
    values: &'a ColumnarValue,
) -> ndarray::ArrayViewD<'a, f64> {
    ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(
        field.as_ref(),
        fixed_size_list_array(values),
    )
    .expect("fixed-shape tensor")
}

fn fixed_shape_view4<'a>(
    field: &'a FieldRef,
    values: &'a ColumnarValue,
) -> ndarray::ArrayView4<'a, f64> {
    fixed_shape_viewd(field, values).into_dimensionality::<Ix4>().expect("rank-4 tensor")
}

fn variable_shape_rows<'a>(
    field: &'a FieldRef,
    values: &'a ColumnarValue,
) -> ndarrow::VariableShapeTensorIter<'a, Float64Type> {
    ndarrow::variable_shape_tensor_iter::<Float64Type>(field.as_ref(), struct_array(values))
        .expect("variable-shape tensor")
}

fn csr_to_dense(view: &ndarrow::CsrView<'_, f64>) -> Array2<f64> {
    let mut dense = Array2::<f64>::zeros((view.nrows, view.ncols));
    for row in 0..view.nrows {
        let start = usize::try_from(view.row_ptrs[row]).expect("row ptr start");
        let end = usize::try_from(view.row_ptrs[row + 1]).expect("row ptr end");
        for offset in start..end {
            let col = usize::try_from(view.col_indices[offset]).expect("column index");
            dense[[row, col]] = view.values[offset];
        }
    }
    dense
}

fn invoke_udf_error(
    udf: &Arc<datafusion::logical_expr::ScalarUDF>,
    args: Vec<ColumnarValue>,
    arg_fields: Vec<FieldRef>,
    scalar_arguments: &[Option<ScalarValue>],
    number_rows: usize,
) -> String {
    invoke_udf(udf, args, arg_fields, scalar_arguments, number_rows)
        .expect_err("expected UDF invocation to fail")
        .to_string()
}

#[test]
fn vector_udfs_cover_real_batch_ops() {
    let left = fixed_size_list([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]]);
    let right = fixed_size_list([[4.0, 0.0, 3.0], [2.0, 2.0, 1.0]]);
    let left_field = vector_field("left", 3, false).expect("vector field");
    let right_field = vector_field("right", 3, false).expect("vector field");
    let l2_norm_udf = udfs::vector_l2_norm_udf();
    let dot_udf = udfs::vector_dot_udf();
    let cosine_similarity_udf = udfs::vector_cosine_similarity_udf();
    let cosine_distance_udf = udfs::vector_cosine_distance_udf();
    let normalize_udf = udfs::vector_normalize_udf();

    let (_, norms) = invoke_udf(
        &l2_norm_udf,
        vec![ColumnarValue::Array(Arc::new(left.clone()))],
        vec![Arc::clone(&left_field)],
        &[None],
        2,
    )
    .expect("vector_l2_norm");
    assert_close(f64_array(&norms).value(0), 5.0);
    assert_close(f64_array(&norms).value(1), 3.0);

    let (_, dots) = invoke_udf(
        &dot_udf,
        vec![
            ColumnarValue::Array(Arc::new(left.clone())),
            ColumnarValue::Array(Arc::new(right.clone())),
        ],
        vec![Arc::clone(&left_field), Arc::clone(&right_field)],
        &[None, None],
        2,
    )
    .expect("vector_dot");
    assert_close(f64_array(&dots).value(0), 12.0);
    assert_close(f64_array(&dots).value(1), 8.0);

    let (_, similarities) = invoke_udf(
        &cosine_similarity_udf,
        vec![
            ColumnarValue::Array(Arc::new(left.clone())),
            ColumnarValue::Array(Arc::new(right.clone())),
        ],
        vec![Arc::clone(&left_field), Arc::clone(&right_field)],
        &[None, None],
        2,
    )
    .expect("vector_cosine_similarity");
    assert_close(f64_array(&similarities).value(0), 0.48);
    assert_close(f64_array(&similarities).value(1), 8.0 / 9.0);

    let (_, distances) = invoke_udf(
        &cosine_distance_udf,
        vec![
            ColumnarValue::Array(Arc::new(left.clone())),
            ColumnarValue::Array(Arc::new(right.clone())),
        ],
        vec![Arc::clone(&left_field), Arc::clone(&right_field)],
        &[None, None],
        2,
    )
    .expect("vector_cosine_distance");
    assert_close(f64_array(&distances).value(0), 0.52);
    assert_close(f64_array(&distances).value(1), 1.0 / 9.0);

    let (_, normalized) = invoke_udf(
        &normalize_udf,
        vec![ColumnarValue::Array(Arc::new(left))],
        vec![left_field],
        &[None],
        2,
    )
    .expect("vector_normalize");
    let normalized = fixed_size_list_array(&normalized);
    let normalized =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(normalized).expect("normalized vectors");
    assert_close(normalized[[0, 0]], 0.6);
    assert_close(normalized[[0, 1]], 0.8);
    assert_close(normalized[[1, 0]], 1.0 / 3.0);
    assert_close(normalized[[1, 1]], 2.0 / 3.0);
    assert_close(normalized[[1, 2]], 2.0 / 3.0);
}

#[test]
fn matrix_udfs_cover_matmul_and_lu_solve() {
    let (left_field, left) =
        matrix_batch("left_matrix", [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [1.0, 2.0]]]);
    let (right_field, right) =
        matrix_batch("right_matrix", [[[2.0, 0.0], [1.0, 2.0]], [[1.0, 1.0], [0.0, 1.0]]]);
    let rhs = fixed_size_list([[5.0, 11.0], [4.0, 5.0]]);
    let rhs_field = vector_field("rhs", 2, false).expect("rhs field");
    let matmul_udf = udfs::matrix_matmul_udf();
    let lu_solve_udf = udfs::matrix_lu_solve_udf();

    let (product_field, product) = invoke_udf(
        &matmul_udf,
        vec![ColumnarValue::Array(Arc::new(left.clone())), ColumnarValue::Array(Arc::new(right))],
        vec![Arc::clone(&left_field), right_field],
        &[None, None],
        2,
    )
    .expect("matrix_matmul");
    let product = fixed_size_list_array(&product);
    let product =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(product_field.as_ref(), product)
            .expect("product tensor")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 tensor");
    assert_close(product[[0, 0, 0]], 4.0);
    assert_close(product[[0, 0, 1]], 4.0);
    assert_close(product[[0, 1, 0]], 10.0);
    assert_close(product[[0, 1, 1]], 8.0);
    assert_close(product[[1, 0, 0]], 2.0);
    assert_close(product[[1, 0, 1]], 2.0);
    assert_close(product[[1, 1, 0]], 1.0);
    assert_close(product[[1, 1, 1]], 3.0);

    let (_, solved) = invoke_udf(
        &lu_solve_udf,
        vec![ColumnarValue::Array(Arc::new(left)), ColumnarValue::Array(Arc::new(rhs))],
        vec![left_field, rhs_field],
        &[None, None],
        2,
    )
    .expect("matrix_lu_solve");
    let solved = fixed_size_list_array(&solved);
    let solved = ndarrow::fixed_size_list_as_array2::<Float64Type>(solved).expect("solutions");
    assert_close(solved[[0, 0]], 1.0);
    assert_close(solved[[0, 1]], 2.0);
    assert_close(solved[[1, 0]], 2.0);
    assert_close(solved[[1, 1]], 1.5);
}

#[test]
fn matrix_lu_returns_struct_of_tensor_batches() {
    let (field, matrices) =
        matrix_batch("lu_matrix", [[[4.0, 2.0], [1.0, 3.0]], [[5.0, 1.0], [2.0, 4.0]]]);
    let lu_udf = udfs::matrix_lu_udf();
    let (return_field, lu) = invoke_udf(
        &lu_udf,
        vec![ColumnarValue::Array(Arc::new(matrices))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_lu");
    let DataType::Struct(fields) = return_field.data_type() else {
        panic!("expected struct return field");
    };
    let lu = struct_array(&lu);
    let lower =
        lu.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("lower tensor array");
    let upper =
        lu.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("upper tensor array");
    let lower = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&fields[0], lower)
        .expect("lower tensor");
    let upper = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&fields[1], upper)
        .expect("upper tensor");
    assert_eq!(lower.shape(), &[2, 2, 2]);
    assert_eq!(upper.shape(), &[2, 2, 2]);
}

#[test]
fn sparse_matvec_returns_variable_shape_vectors() {
    let (sparse_field, sparse) = sparse_batch("sparse");
    let (ragged_field, vectors) =
        ragged_vectors("vectors", vec![vec![1.0, 2.0, 3.0], vec![2.0, 1.0]]);
    let sparse_matvec_udf = udfs::sparse_matvec_udf();
    let (sparse_return_field, sparse_result) = invoke_udf(
        &sparse_matvec_udf,
        vec![ColumnarValue::Array(Arc::new(sparse)), ColumnarValue::Array(Arc::new(vectors))],
        vec![sparse_field, ragged_field],
        &[None, None],
        2,
    )
    .expect("sparse_matvec");
    let sparse_result = struct_array(&sparse_result);
    let mut sparse_iter = ndarrow::variable_shape_tensor_iter::<Float64Type>(
        sparse_return_field.as_ref(),
        sparse_result,
    )
    .expect("ragged tensor output");
    let (_, row0) = sparse_iter.next().expect("row 0").expect("row 0 view");
    let (_, row1) = sparse_iter.next().expect("row 1").expect("row 1 view");
    assert_eq!(row0.shape(), &[2]);
    assert_eq!(row1.shape(), &[2]);
    assert_close(row0[[0]], 7.0);
    assert_close(row0[[1]], 6.0);
    assert_close(row1[[0]], 8.0);
    assert_close(row1[[1]], 16.0);
}

#[test]
fn tensor_sum_last_axis_and_matrix_column_means_work() {
    let (tensor_field, tensor) =
        matrix_batch("tensor", [[[1.0, 2.0], [3.0, 4.0]], [[5.0, 6.0], [7.0, 8.0]]]);
    let tensor_sum_udf = udfs::tensor_sum_last_axis_udf();
    let column_means_udf = udfs::matrix_column_means_udf();
    let (tensor_return_field, tensor_result) = invoke_udf(
        &tensor_sum_udf,
        vec![ColumnarValue::Array(Arc::new(tensor.clone()))],
        vec![Arc::clone(&tensor_field)],
        &[None],
        2,
    )
    .expect("tensor_sum_last_axis");
    let tensor_result = fixed_size_list_array(&tensor_result);
    let tensor_result = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(
        tensor_return_field.as_ref(),
        tensor_result,
    )
    .expect("tensor reduction")
    .into_dimensionality::<Ix2>()
    .expect("rank-2 tensor");
    assert_close(tensor_result[[0, 0]], 3.0);
    assert_close(tensor_result[[0, 1]], 7.0);
    assert_close(tensor_result[[1, 0]], 11.0);
    assert_close(tensor_result[[1, 1]], 15.0);

    let (_, means) = invoke_udf(
        &column_means_udf,
        vec![ColumnarValue::Array(Arc::new(tensor))],
        vec![Arc::clone(&tensor_field)],
        &[None],
        2,
    )
    .expect("matrix_column_means");
    let means = fixed_size_list_array(&means);
    let means = ndarrow::fixed_size_list_as_array2::<Float64Type>(means).expect("means");
    assert_close(means[[0, 0]], 2.0);
    assert_close(means[[0, 1]], 3.0);
    assert_close(means[[1, 0]], 6.0);
    assert_close(means[[1, 1]], 7.0);
}

#[test]
fn linear_regression_returns_struct_result() {
    let (design_field, design) =
        matrix_batch("design", [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [0.0, 1.0], [
            1.0, 1.0,
        ]]]);
    let response = fixed_size_list([[1.0, 2.0, 3.0], [2.0, 1.0, 3.0]]);
    let response_field = vector_field("response", 3, false).expect("response field");
    let linear_regression_udf = udfs::linear_regression_udf();
    let (_, regression) = invoke_udf(
        &linear_regression_udf,
        vec![
            ColumnarValue::Array(Arc::new(design)),
            ColumnarValue::Array(Arc::new(response)),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))),
        ],
        vec![
            design_field,
            response_field,
            Arc::new(Field::new("add_intercept", DataType::Boolean, false)),
        ],
        &[None, None, Some(ScalarValue::Boolean(Some(false)))],
        2,
    )
    .expect("linear_regression");
    let regression = struct_array(&regression);
    let coefficients =
        regression.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("coefficients");
    let coefficients =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(coefficients).expect("coefficients");
    let fitted =
        regression.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("fitted");
    let fitted = ndarrow::fixed_size_list_as_array2::<Float64Type>(fitted).expect("fitted");
    let residuals =
        regression.column(2).as_any().downcast_ref::<FixedSizeListArray>().expect("residuals");
    let residuals =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(residuals).expect("residuals");
    let r_squared =
        regression.column(3).as_any().downcast_ref::<Float64Array>().expect("r_squared");
    assert_close(coefficients[[0, 0]], 1.0);
    assert_close(coefficients[[0, 1]], 2.0);
    assert_close(coefficients[[1, 0]], 2.0);
    assert_close(coefficients[[1, 1]], 1.0);
    assert_close(fitted[[0, 0]], 1.0);
    assert_close(fitted[[0, 1]], 2.0);
    assert_close(fitted[[0, 2]], 3.0);
    assert_close(fitted[[1, 0]], 2.0);
    assert_close(fitted[[1, 1]], 1.0);
    assert_close(fitted[[1, 2]], 3.0);
    assert_close(residuals[[0, 0]], 0.0);
    assert_close(residuals[[0, 1]], 0.0);
    assert_close(residuals[[0, 2]], 0.0);
    assert_close(residuals[[1, 0]], 0.0);
    assert_close(residuals[[1, 1]], 0.0);
    assert_close(residuals[[1, 2]], 0.0);
    assert_close(r_squared.value(0), 1.0);
    assert_close(r_squared.value(1), 1.0);
}

#[test]
fn matrix_inverse_and_determinant_udfs_cover_scalar_outputs() {
    let (field, matrices) =
        matrix_batch("matrix", [[[9.0, 0.0], [0.0, 4.0]], [[16.0, 0.0], [0.0, 1.0]]]);

    let inverse_udf = udfs::matrix_inverse_udf();
    let determinant_udf = udfs::matrix_determinant_udf();
    let log_determinant_udf = udfs::matrix_log_determinant_udf();

    let (inverse_field, inverse) = invoke_udf(
        &inverse_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_inverse");
    let inverse = fixed_shape_view3(&inverse_field, &inverse);
    assert_close(inverse[[0, 0, 0]], 1.0 / 9.0);
    assert_close(inverse[[0, 1, 1]], 0.25);
    assert_close(inverse[[1, 0, 0]], 1.0 / 16.0);
    assert_close(inverse[[1, 1, 1]], 1.0);

    let (_, determinant) = invoke_udf(
        &determinant_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_determinant");
    assert_close(f64_array(&determinant).value(0), 36.0);
    assert_close(f64_array(&determinant).value(1), 16.0);

    let (_, log_determinant) = invoke_udf(
        &log_determinant_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_log_determinant");
    let log_determinant = struct_array(&log_determinant);
    let signs = log_determinant.column(0).as_any().downcast_ref::<Int8Array>().expect("signs");
    let log_abs =
        log_determinant.column(1).as_any().downcast_ref::<Float64Array>().expect("log abs");
    assert_eq!(signs.value(0), 1);
    assert_eq!(signs.value(1), 1);
    assert_close(log_abs.value(0), 36.0_f64.ln());
    assert_close(log_abs.value(1), 16.0_f64.ln());
}

#[test]
fn matrix_cholesky_udfs_cover_struct_solver_and_inverse_outputs() {
    let (field, matrices) =
        matrix_batch("matrix", [[[9.0, 0.0], [0.0, 4.0]], [[16.0, 0.0], [0.0, 1.0]]]);
    let rhs = fixed_size_list([[9.0, 8.0], [32.0, 3.0]]);
    let rhs_field = vector_field("rhs", 2, false).expect("rhs field");

    let cholesky_udf = udfs::matrix_cholesky_udf();
    let cholesky_solve_udf = udfs::matrix_cholesky_solve_udf();
    let cholesky_inverse_udf = udfs::matrix_cholesky_inverse_udf();

    let (cholesky_return_field, cholesky) = invoke_udf(
        &cholesky_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_cholesky");
    let DataType::Struct(cholesky_fields) = cholesky_return_field.data_type() else {
        panic!("expected struct return type");
    };
    let cholesky = struct_array(&cholesky);
    let lower =
        cholesky.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("lower tensor");
    let lower =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&cholesky_fields[0], lower)
            .expect("lower tensor")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 tensor");
    assert_close(lower[[0, 0, 0]], 3.0);
    assert_close(lower[[0, 1, 1]], 2.0);
    assert_close(lower[[1, 0, 0]], 4.0);
    assert_close(lower[[1, 1, 1]], 1.0);

    let (_, cholesky_solution) = invoke_udf(
        &cholesky_solve_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone())), ColumnarValue::Array(Arc::new(rhs))],
        vec![Arc::clone(&field), rhs_field],
        &[None, None],
        2,
    )
    .expect("matrix_cholesky_solve");
    let cholesky_solution = ndarrow::fixed_size_list_as_array2::<Float64Type>(
        fixed_size_list_array(&cholesky_solution),
    )
    .expect("cholesky solution");
    assert_close(cholesky_solution[[0, 0]], 1.0);
    assert_close(cholesky_solution[[0, 1]], 2.0);
    assert_close(cholesky_solution[[1, 0]], 2.0);
    assert_close(cholesky_solution[[1, 1]], 3.0);

    let (cholesky_inverse_field, cholesky_inverse) = invoke_udf(
        &cholesky_inverse_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_cholesky_inverse");
    let cholesky_inverse = fixed_shape_view3(&cholesky_inverse_field, &cholesky_inverse);
    assert_close(cholesky_inverse[[0, 0, 0]], 1.0 / 9.0);
    assert_close(cholesky_inverse[[0, 1, 1]], 0.25);
    assert_close(cholesky_inverse[[1, 0, 0]], 1.0 / 16.0);
    assert_close(cholesky_inverse[[1, 1, 1]], 1.0);
}

#[test]
fn matrix_qr_and_svd_udfs_cover_struct_outputs() {
    let (field, matrices) =
        matrix_batch("matrix", [[[9.0, 0.0], [0.0, 4.0]], [[16.0, 0.0], [0.0, 1.0]]]);
    let qr_udf = udfs::matrix_qr_udf();
    let svd_udf = udfs::matrix_svd_udf();

    let (qr_return_field, qr) = invoke_udf(
        &qr_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_qr");
    let DataType::Struct(qr_fields) = qr_return_field.data_type() else {
        panic!("expected QR struct return");
    };
    let qr = struct_array(&qr);
    let q = qr.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("q tensor");
    let q = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&qr_fields[0], q)
        .expect("q tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 tensor");
    let r = qr.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("r tensor");
    let r = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&qr_fields[1], r)
        .expect("r tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 tensor");
    let rank = qr.column(2).as_any().downcast_ref::<Int64Array>().expect("rank");
    assert_close(q[[0, 0, 0]], 1.0);
    assert_close(q[[0, 1, 1]], 1.0);
    assert_close(r[[0, 0, 0]], 9.0);
    assert_close(r[[0, 1, 1]], 4.0);
    assert_eq!(rank.value(0), 2);
    assert_eq!(rank.value(1), 2);

    let (svd_return_field, svd) = invoke_udf(
        &svd_udf,
        vec![ColumnarValue::Array(Arc::new(matrices))],
        vec![field],
        &[None],
        2,
    )
    .expect("matrix_svd");
    let DataType::Struct(svd_fields) = svd_return_field.data_type() else {
        panic!("expected SVD struct return");
    };
    let svd = struct_array(&svd);
    let u = svd.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("u tensor");
    let u = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&svd_fields[0], u)
        .expect("u tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 tensor");
    let singular_values =
        svd.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("singular values");
    let singular_values = ndarrow::fixed_size_list_as_array2::<Float64Type>(singular_values)
        .expect("singular values");
    let vt = svd.column(2).as_any().downcast_ref::<FixedSizeListArray>().expect("vt tensor");
    let vt = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&svd_fields[2], vt)
        .expect("vt tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 tensor");
    assert_close(u[[0, 0, 0]], 1.0);
    assert_close(u[[0, 1, 1]], 1.0);
    assert_close(singular_values[[0, 0]], 9.0);
    assert_close(singular_values[[0, 1]], 4.0);
    assert_close(singular_values[[1, 0]], 16.0);
    assert_close(singular_values[[1, 1]], 1.0);
    assert_close(vt[[0, 0, 0]], 1.0);
    assert_close(vt[[0, 1, 1]], 1.0);
}

#[test]
fn matrix_stats_and_pca_udfs_cover_batch_outputs() {
    let (field, matrices) =
        matrix_batch("stats", [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], [[1.0, 0.0], [0.0, 1.0], [
            1.0, 1.0,
        ]]]);
    let center_udf = udfs::matrix_center_columns_udf();
    let covariance_udf = udfs::matrix_covariance_udf();
    let correlation_udf = udfs::matrix_correlation_udf();
    let pca_udf = udfs::matrix_pca_udf();

    let (center_field, centered) = invoke_udf(
        &center_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_center_columns");
    let centered = fixed_shape_view3(&center_field, &centered);
    assert_close(centered[[0, 0, 0]], -2.0);
    assert_close(centered[[0, 2, 1]], 2.0);
    assert_close(centered[[1, 0, 0]], 1.0 / 3.0);
    assert_close(centered[[1, 1, 1]], 1.0 / 3.0);

    let (cov_field, covariance) = invoke_udf(
        &covariance_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_covariance");
    let covariance = fixed_shape_view3(&cov_field, &covariance);
    assert_close(covariance[[0, 0, 0]], 4.0);
    assert_close(covariance[[0, 0, 1]], 4.0);
    assert_close(covariance[[1, 0, 0]], 1.0 / 3.0);
    assert_close(covariance[[1, 0, 1]], -1.0 / 6.0);

    let (corr_field, correlation) = invoke_udf(
        &correlation_udf,
        vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
        vec![Arc::clone(&field)],
        &[None],
        2,
    )
    .expect("matrix_correlation");
    let correlation = fixed_shape_view3(&corr_field, &correlation);
    assert_close(correlation[[0, 0, 0]], 1.0);
    assert_close(correlation[[0, 0, 1]], 1.0);
    assert_close(correlation[[1, 0, 1]], -0.5);

    let (pca_return_field, pca) = invoke_udf(
        &pca_udf,
        vec![ColumnarValue::Array(Arc::new(matrices))],
        vec![field],
        &[None],
        2,
    )
    .expect("matrix_pca");
    let DataType::Struct(pca_fields) = pca_return_field.data_type() else {
        panic!("expected PCA struct return");
    };
    let pca = struct_array(&pca);
    let components =
        pca.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("components");
    let components =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&pca_fields[0], components)
            .expect("components")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 tensor");
    let explained =
        pca.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("explained variance");
    let explained =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(explained).expect("explained variance");
    let ratio = pca
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("explained variance ratio");
    let ratio = ndarrow::fixed_size_list_as_array2::<Float64Type>(ratio).expect("ratio");
    let mean = pca.column(3).as_any().downcast_ref::<FixedSizeListArray>().expect("mean");
    let mean = ndarrow::fixed_size_list_as_array2::<Float64Type>(mean).expect("mean");
    let scores = pca.column(4).as_any().downcast_ref::<FixedSizeListArray>().expect("scores");
    let scores = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&pca_fields[4], scores)
        .expect("scores")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 tensor");
    assert_close(mean[[0, 0]], 3.0);
    assert_close(mean[[0, 1]], 4.0);
    assert_close(explained[[0, 0]], 8.0);
    assert_close(explained[[0, 1]], 0.0);
    assert_close(ratio[[0, 0]], 1.0);
    assert_close(ratio[[0, 1]], 0.0);
    assert_close(components[[0, 0, 0]].abs(), 1.0 / 2.0_f64.sqrt());
    assert_close(components[[0, 0, 1]].abs(), 1.0 / 2.0_f64.sqrt());
    assert_close(scores[[0, 1, 0]], 0.0);
    assert_close(scores[[0, 1, 1]], 0.0);
}

#[test]
fn sparse_batch_udfs_cover_dense_sparse_products_and_transpose() {
    let (left_field, left) = sparse_batch("left");
    let (right_dense_field, right_dense) = ragged_matrices("right_dense", vec![
        vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        vec![vec![1.0, 0.0], vec![0.0, 1.0]],
    ]);
    let (right_sparse_field, right_sparse) = sparse_batch_rhs("right_sparse");

    let matmat_dense_udf = udfs::sparse_matmat_dense_udf();
    let transpose_udf = udfs::sparse_transpose_udf();
    let matmat_sparse_udf = udfs::sparse_matmat_sparse_udf();

    let (dense_return_field, dense_result) = invoke_udf(
        &matmat_dense_udf,
        vec![
            ColumnarValue::Array(Arc::new(left.clone())),
            ColumnarValue::Array(Arc::new(right_dense)),
        ],
        vec![Arc::clone(&left_field), right_dense_field],
        &[None, None],
        2,
    )
    .expect("sparse_matmat_dense");
    let mut dense_iter = variable_shape_rows(&dense_return_field, &dense_result);
    let (_, row0) = dense_iter.next().expect("row0").expect("row0 view");
    let row0 = row0.into_dimensionality::<Ix2>().expect("rank-2 output");
    let (_, row1) = dense_iter.next().expect("row1").expect("row1 view");
    let row1 = row1.into_dimensionality::<Ix2>().expect("rank-2 output");
    assert_close(row0[[0, 0]], 3.0);
    assert_close(row0[[0, 1]], 2.0);
    assert_close(row0[[1, 1]], 3.0);
    assert_close(row1[[0, 0]], 4.0);
    assert_close(row1[[1, 1]], 6.0);

    let (transpose_field, transpose_result) = invoke_udf(
        &transpose_udf,
        vec![ColumnarValue::Array(Arc::new(left.clone()))],
        vec![Arc::clone(&left_field)],
        &[None],
        2,
    )
    .expect("sparse_transpose");
    let transpose_result = struct_array(&transpose_result);
    let mut transpose_iter =
        ndarrow::csr_matrix_batch_iter::<Float64Type>(transpose_field.as_ref(), transpose_result)
            .expect("transpose batch");
    let (_, transposed0) = transpose_iter.next().expect("row0").expect("row0 csr");
    let (_, transposed1) = transpose_iter.next().expect("row1").expect("row1 csr");
    let transposed0 = csr_to_dense(&transposed0);
    let transposed1 = csr_to_dense(&transposed1);
    assert_eq!(transposed0.shape(), &[3, 2]);
    assert_eq!(transposed1.shape(), &[2, 2]);
    assert_close(transposed0[[0, 0]], 1.0);
    assert_close(transposed0[[1, 1]], 3.0);
    assert_close(transposed0[[2, 0]], 2.0);
    assert_close(transposed1[[0, 0]], 4.0);
    assert_close(transposed1[[0, 1]], 5.0);
    assert_close(transposed1[[1, 1]], 6.0);

    let (sparse_return_field, sparse_result) = invoke_udf(
        &matmat_sparse_udf,
        vec![ColumnarValue::Array(Arc::new(left)), ColumnarValue::Array(Arc::new(right_sparse))],
        vec![left_field, right_sparse_field],
        &[None, None],
        2,
    )
    .expect("sparse_matmat_sparse");
    let sparse_result = struct_array(&sparse_result);
    let mut sparse_iter =
        ndarrow::csr_matrix_batch_iter::<Float64Type>(sparse_return_field.as_ref(), sparse_result)
            .expect("sparse product batch");
    let (_, product0) = sparse_iter.next().expect("row0").expect("row0 csr");
    let (_, product1) = sparse_iter.next().expect("row1").expect("row1 csr");
    let product0 = csr_to_dense(&product0);
    let product1 = csr_to_dense(&product1);
    assert_close(product0[[0, 0]], 3.0);
    assert_close(product0[[0, 1]], 2.0);
    assert_close(product0[[1, 1]], 3.0);
    assert_close(product1[[0, 0]], 4.0);
    assert_close(product1[[1, 1]], 6.0);
}

#[test]
fn tensor_fixed_shape_last_axis_udfs_cover_outputs() {
    let (tensor_field, tensor) =
        matrix_batch("tensor", [[[3.0, 4.0], [0.0, 5.0]], [[8.0, 15.0], [7.0, 24.0]]]);
    let (other_tensor_field, other_tensor) =
        matrix_batch("other_tensor", [[[4.0, 0.0], [1.0, 2.0]], [[15.0, 8.0], [24.0, 7.0]]]);

    let l2_udf = udfs::tensor_l2_norm_last_axis_udf();
    let normalize_udf = udfs::tensor_normalize_last_axis_udf();
    let dot_udf = udfs::tensor_batched_dot_last_axis_udf();

    let (l2_field, l2_norms) = invoke_udf(
        &l2_udf,
        vec![ColumnarValue::Array(Arc::new(tensor.clone()))],
        vec![Arc::clone(&tensor_field)],
        &[None],
        2,
    )
    .expect("tensor_l2_norm_last_axis");
    let l2_norms = fixed_shape_viewd(&l2_field, &l2_norms)
        .into_dimensionality::<Ix2>()
        .expect("rank-2 tensor");
    assert_close(l2_norms[[0, 0]], 5.0);
    assert_close(l2_norms[[0, 1]], 5.0);
    assert_close(l2_norms[[1, 0]], 17.0);
    assert_close(l2_norms[[1, 1]], 25.0);

    let (normalize_field, normalized) = invoke_udf(
        &normalize_udf,
        vec![ColumnarValue::Array(Arc::new(tensor.clone()))],
        vec![Arc::clone(&tensor_field)],
        &[None],
        2,
    )
    .expect("tensor_normalize_last_axis");
    let normalized = fixed_shape_view3(&normalize_field, &normalized);
    assert_close(normalized[[0, 0, 0]], 0.6);
    assert_close(normalized[[0, 0, 1]], 0.8);
    assert_close(normalized[[0, 1, 1]], 1.0);

    let (dot_field, dots) = invoke_udf(
        &dot_udf,
        vec![
            ColumnarValue::Array(Arc::new(tensor.clone())),
            ColumnarValue::Array(Arc::new(other_tensor)),
        ],
        vec![Arc::clone(&tensor_field), other_tensor_field],
        &[None, None],
        2,
    )
    .expect("tensor_batched_dot_last_axis");
    let dots =
        fixed_shape_viewd(&dot_field, &dots).into_dimensionality::<Ix2>().expect("rank-2 tensor");
    assert_close(dots[[0, 0]], 12.0);
    assert_close(dots[[0, 1]], 10.0);
    assert_close(dots[[1, 0]], 240.0);
    assert_close(dots[[1, 1]], 336.0);
}

#[test]
fn tensor_fixed_shape_matmul_udf_covers_rank3plus_batch() {
    let (cube_left_field, cube_left) =
        tensor_batch4("cube_left", [[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [1.0, 2.0]]]]);
    let (cube_right_field, cube_right) =
        tensor_batch4("cube_right", [[[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 1.0]]]]);
    let matmul_udf = udfs::tensor_batched_matmul_last_two_udf();

    let (matmul_field, matmul) = invoke_udf(
        &matmul_udf,
        vec![ColumnarValue::Array(Arc::new(cube_left)), ColumnarValue::Array(Arc::new(cube_right))],
        vec![cube_left_field, cube_right_field],
        &[None, None],
        1,
    )
    .expect("tensor_batched_matmul_last_two");
    let matmul = fixed_shape_view4(&matmul_field, &matmul);
    assert_close(matmul[[0, 0, 0, 0]], 1.0);
    assert_close(matmul[[0, 0, 1, 1]], 4.0);
    assert_close(matmul[[0, 1, 0, 0]], 2.0);
    assert_close(matmul[[0, 1, 1, 1]], 3.0);
}

#[test]
fn tensor_variable_shape_udfs_cover_outputs() {
    let (variable_field, variable_batch) = ragged_matrices("variable", vec![
        vec![vec![3.0, 4.0], vec![0.0, 5.0]],
        vec![vec![6.0, 8.0, 0.0]],
    ]);
    let (other_variable_field, other_variable_batch) = ragged_matrices("other_variable", vec![
        vec![vec![4.0, 0.0], vec![1.0, 2.0]],
        vec![vec![1.0, 0.0, 2.0]],
    ]);
    let variable_sum_udf = udfs::tensor_variable_sum_last_axis_udf();
    let variable_l2_udf = udfs::tensor_variable_l2_norm_last_axis_udf();
    let variable_normalize_udf = udfs::tensor_variable_normalize_last_axis_udf();
    let variable_dot_udf = udfs::tensor_variable_batched_dot_last_axis_udf();

    let (variable_sum_field, variable_sums) = invoke_udf(
        &variable_sum_udf,
        vec![ColumnarValue::Array(Arc::new(variable_batch.clone()))],
        vec![Arc::clone(&variable_field)],
        &[None],
        2,
    )
    .expect("tensor_variable_sum_last_axis");
    let mut variable_sum_iter = variable_shape_rows(&variable_sum_field, &variable_sums);
    let (_, sum_row0) = variable_sum_iter.next().expect("row0").expect("row0 view");
    let sum_row0 = sum_row0.into_dimensionality::<Ix1>().expect("rank-1 output");
    let (_, sum_row1) = variable_sum_iter.next().expect("row1").expect("row1 view");
    let sum_row1 = sum_row1.into_dimensionality::<Ix1>().expect("rank-1 output");
    assert_close(sum_row0[[0]], 7.0);
    assert_close(sum_row0[[1]], 5.0);
    assert_close(sum_row1[[0]], 14.0);

    let (variable_l2_field, variable_l2) = invoke_udf(
        &variable_l2_udf,
        vec![ColumnarValue::Array(Arc::new(variable_batch.clone()))],
        vec![Arc::clone(&variable_field)],
        &[None],
        2,
    )
    .expect("tensor_variable_l2_norm_last_axis");
    let mut variable_l2_iter = variable_shape_rows(&variable_l2_field, &variable_l2);
    let (_, l2_row0) = variable_l2_iter.next().expect("row0").expect("row0 view");
    let l2_row0 = l2_row0.into_dimensionality::<Ix1>().expect("rank-1 output");
    let (_, l2_row1) = variable_l2_iter.next().expect("row1").expect("row1 view");
    let l2_row1 = l2_row1.into_dimensionality::<Ix1>().expect("rank-1 output");
    assert_close(l2_row0[[0]], 5.0);
    assert_close(l2_row0[[1]], 5.0);
    assert_close(l2_row1[[0]], 10.0);

    let (variable_normalize_field, variable_normalized) = invoke_udf(
        &variable_normalize_udf,
        vec![ColumnarValue::Array(Arc::new(variable_batch.clone()))],
        vec![Arc::clone(&variable_field)],
        &[None],
        2,
    )
    .expect("tensor_variable_normalize_last_axis");
    let mut variable_normalize_iter =
        variable_shape_rows(&variable_normalize_field, &variable_normalized);
    let (_, variable_normalized0) =
        variable_normalize_iter.next().expect("row0").expect("row0 view");
    let variable_normalized0 =
        variable_normalized0.into_dimensionality::<Ix2>().expect("rank-2 output");
    assert_close(variable_normalized0[[0, 0]], 0.6);
    assert_close(variable_normalized0[[0, 1]], 0.8);
    assert_close(variable_normalized0[[1, 1]], 1.0);

    let (variable_dot_field, variable_dots) = invoke_udf(
        &variable_dot_udf,
        vec![
            ColumnarValue::Array(Arc::new(variable_batch)),
            ColumnarValue::Array(Arc::new(other_variable_batch)),
        ],
        vec![variable_field, other_variable_field],
        &[None, None],
        2,
    )
    .expect("tensor_variable_batched_dot_last_axis");
    let mut variable_dot_iter = variable_shape_rows(&variable_dot_field, &variable_dots);
    let (_, dot_row0) = variable_dot_iter.next().expect("row0").expect("row0 view");
    let dot_row0 = dot_row0.into_dimensionality::<Ix1>().expect("rank-1 output");
    let (_, dot_row1) = variable_dot_iter.next().expect("row1").expect("row1 view");
    let dot_row1 = dot_row1.into_dimensionality::<Ix1>().expect("rank-1 output");
    assert_close(dot_row0[[0]], 12.0);
    assert_close(dot_row0[[1]], 10.0);
    assert_close(dot_row1[[0]], 6.0);
}

#[test]
fn udf_catalog_covers_core_trait_entrypoints() {
    for udf in udfs::all_default_functions() {
        assert!(!udf.name().is_empty());
        let _ = udf.signature();
        let _ = udf.inner().as_any();
        let error =
            udf.return_type(&[]).expect_err("return_type should be delegated through fields");
        assert!(error.to_string().contains("return_field_from_args should be used instead"));
    }
}

#[test]
fn vector_udfs_validate_array_and_storage_contracts() {
    let l2_norm_udf = udfs::vector_l2_norm_udf();
    let vector_input = fixed_size_list([[1.0, 2.0, 3.0]]);
    let vector_field = vector_field("vector", 3, false).expect("vector field");

    let scalar_error = invoke_udf_error(
        &l2_norm_udf,
        vec![ColumnarValue::Scalar(ScalarValue::Float64(Some(1.0)))],
        vec![Arc::clone(&vector_field)],
        &[None],
        1,
    );
    let storage_error = invoke_udf_error(
        &l2_norm_udf,
        vec![ColumnarValue::Array(Arc::new(Float64Array::from(vec![1.0, 2.0, 3.0])))],
        vec![Arc::clone(&vector_field)],
        &[None],
        3,
    );
    let field_error = invoke_udf_error(
        &l2_norm_udf,
        vec![ColumnarValue::Array(Arc::new(vector_input))],
        vec![Arc::new(Field::new("vector", DataType::Float64, false))],
        &[None],
        1,
    );

    assert!(scalar_error.contains("argument 1 must be an array column"));
    assert!(storage_error.contains("expected FixedSizeListArray storage"));
    assert!(field_error.contains("expected FixedSizeList<Float64>(D)"));
}

#[test]
fn matrix_udfs_validate_shape_contracts() {
    let matmul_udf = udfs::matrix_matmul_udf();
    let lu_udf = udfs::matrix_lu_udf();
    let lu_solve_udf = udfs::matrix_lu_solve_udf();
    let (left_field, left) = matrix_batch("left", [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
    let (right_field, right) = matrix_batch("right", [[[1.0, 0.0], [0.0, 1.0]]]);
    let (nonsquare_field, nonsquare) =
        matrix_batch("nonsquare", [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
    let rhs = fixed_size_list([[1.0, 2.0]]);
    let rhs_field = vector_field("rhs", 2, false).expect("rhs field");

    let matmul_error = invoke_udf_error(
        &matmul_udf,
        vec![ColumnarValue::Array(Arc::new(left)), ColumnarValue::Array(Arc::new(right))],
        vec![left_field, right_field],
        &[None, None],
        1,
    );
    let lu_error = invoke_udf_error(
        &lu_udf,
        vec![ColumnarValue::Array(Arc::new(nonsquare.clone()))],
        vec![Arc::clone(&nonsquare_field)],
        &[None],
        1,
    );
    let rhs_error = invoke_udf_error(
        &lu_solve_udf,
        vec![ColumnarValue::Array(Arc::new(nonsquare)), ColumnarValue::Array(Arc::new(rhs))],
        vec![nonsquare_field, rhs_field],
        &[None, None],
        1,
    );

    assert!(matmul_error.contains("incompatible matrix shapes"));
    assert!(lu_error.contains("matrix_lu requires square matrices"));
    assert!(rhs_error.contains("matrix_lu_solve requires square matrices"));
}

#[test]
fn matrix_lu_solve_and_linear_regression_validate_runtime_batch_edges() {
    let lu_solve_udf = udfs::matrix_lu_solve_udf();
    let linear_regression_udf = udfs::linear_regression_udf();
    let (matrices_field, matrices) =
        matrix_batch("matrices", [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [1.0, 2.0]]]);
    let short_rhs = fixed_size_list([[1.0, 2.0]]);
    let short_rhs_field = vector_field("rhs", 2, false).expect("rhs field");
    let (design_field, design) =
        matrix_batch("design", [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 0.0]]]);
    let response = fixed_size_list([[1.0, 2.0]]);
    let response_field = vector_field("response", 2, false).expect("response field");
    let bool_field = Arc::new(Field::new("add_intercept", DataType::Boolean, false));

    let lu_solve_error = invoke_udf_error(
        &lu_solve_udf,
        vec![
            ColumnarValue::Array(Arc::new(matrices)),
            ColumnarValue::Array(Arc::new(short_rhs.clone())),
        ],
        vec![Arc::clone(&matrices_field), Arc::clone(&short_rhs_field)],
        &[None, None],
        2,
    );
    let missing_scalar_error = invoke_udf_error(
        &linear_regression_udf,
        vec![
            ColumnarValue::Array(Arc::new(design.clone())),
            ColumnarValue::Array(Arc::new(response.clone())),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))),
        ],
        vec![Arc::clone(&design_field), Arc::clone(&response_field), Arc::clone(&bool_field)],
        &[None, None, None],
        2,
    );
    let scalar_runtime_error = invoke_udf_error(
        &linear_regression_udf,
        vec![
            ColumnarValue::Array(Arc::new(design.clone())),
            ColumnarValue::Array(Arc::new(response.clone())),
            ColumnarValue::Array(Arc::new(Float64Array::from(vec![1.0, 0.0]))),
        ],
        vec![Arc::clone(&design_field), Arc::clone(&response_field), Arc::clone(&bool_field)],
        &[None, None, Some(ScalarValue::Boolean(Some(false)))],
        2,
    );
    let regression_batch_error = invoke_udf_error(
        &linear_regression_udf,
        vec![
            ColumnarValue::Array(Arc::new(design)),
            ColumnarValue::Array(Arc::new(response)),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(false))),
        ],
        vec![design_field, response_field, bool_field],
        &[None, None, Some(ScalarValue::Boolean(Some(false)))],
        2,
    );

    assert!(lu_solve_error.contains("batch length mismatch"));
    assert!(missing_scalar_error.contains("argument 3 must be a non-null scalar"));
    assert!(scalar_runtime_error.contains("argument 3 must be a scalar Boolean"));
    assert!(regression_batch_error.contains("batch length mismatch"));
}

#[test]
fn sparse_and_tensor_udfs_validate_contract_edges() {
    let sparse_matvec_udf = udfs::sparse_matvec_udf();
    let tensor_sum_udf = udfs::tensor_sum_last_axis_udf();
    let (sparse_field, sparse) = sparse_batch("sparse");
    let (ragged_field, ragged_vectors) = ragged_vectors("vectors", vec![vec![1.0, 2.0], vec![3.0]]);
    let rank_two_vectors =
        crate::metadata::variable_shape_tensor_field("vectors", 2, Some(&[None, None]), false)
            .expect("rank-2 ragged field");
    let rank_one_tensor = crate::metadata::fixed_shape_tensor_field("tensor", &[4], false)
        .expect("rank-1 tensor field");
    let rank_one_tensor_values = FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
        vec![Some(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)])],
        4,
    );

    let sparse_rank_error = invoke_udf_error(
        &sparse_matvec_udf,
        vec![
            ColumnarValue::Array(Arc::new(sparse.clone())),
            ColumnarValue::Array(Arc::new(ragged_vectors.clone())),
        ],
        vec![Arc::clone(&sparse_field), rank_two_vectors],
        &[None, None],
        2,
    );
    let sparse_storage_error = invoke_udf_error(
        &sparse_matvec_udf,
        vec![
            ColumnarValue::Array(Arc::new(fixed_size_list([[1.0, 2.0]]))),
            ColumnarValue::Array(Arc::new(ragged_vectors)),
        ],
        vec![sparse_field, ragged_field],
        &[None, None],
        1,
    );
    let tensor_rank_error = invoke_udf_error(
        &tensor_sum_udf,
        vec![ColumnarValue::Array(Arc::new(rank_one_tensor_values))],
        vec![rank_one_tensor],
        &[None],
        1,
    );

    assert!(sparse_rank_error.contains("batch of rank-1 dense vectors"));
    assert!(sparse_storage_error.contains("expected StructArray storage"));
    assert!(tensor_rank_error.contains("requires tensors with rank >= 2"));
}
