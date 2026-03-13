use std::sync::Arc;

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int8Array, Int64Array,
    ListArray, StructArray,
};
use datafusion::arrow::datatypes::{DataType, Field, FieldRef};
use datafusion::common::ScalarValue;
use datafusion::common::utils::arrays_into_list_array;
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

fn fixed_size_list_f32<const R: usize, const C: usize>(rows: [[f32; C]; R]) -> FixedSizeListArray {
    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
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

fn matrix_batch_f32<const B: usize, const R: usize, const C: usize>(
    name: &str,
    values: [[[f32; C]; R]; B],
) -> (FieldRef, FixedSizeListArray) {
    let values = values.into_iter().flatten().flatten().collect::<Vec<f32>>();
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

fn ragged_vectors_f32(name: &str, rows: Vec<Vec<f32>>) -> (FieldRef, StructArray) {
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

fn ragged_matrices_f32(name: &str, rows: Vec<Vec<Vec<f32>>>) -> (FieldRef, StructArray) {
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

fn tensor_batch4_f32<const B: usize, const D0: usize, const D1: usize, const D2: usize>(
    name: &str,
    values: [[[[f32; D2]; D1]; D0]; B],
) -> (FieldRef, FixedSizeListArray) {
    let values = values.into_iter().flatten().flatten().flatten().collect::<Vec<f32>>();
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

fn sparse_batch_f32(name: &str) -> (FieldRef, StructArray) {
    let (field, array) = ndarrow::csr_batch_to_extension_array(
        name,
        vec![[2, 3], [2, 2]],
        vec![vec![0, 2, 3], vec![0, 1, 3]],
        vec![vec![0, 2, 1], vec![0, 0, 1]],
        vec![vec![1.0_f32, 2.0, 3.0], vec![4.0_f32, 5.0, 6.0]],
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

fn sparse_batch_rhs_f32(name: &str) -> (FieldRef, StructArray) {
    let (field, array) = ndarrow::csr_batch_to_extension_array(
        name,
        vec![[3, 2], [2, 2]],
        vec![vec![0, 1, 2, 4], vec![0, 1, 2]],
        vec![vec![0, 1, 0, 1], vec![0, 1]],
        vec![vec![1.0_f32, 1.0, 1.0, 1.0], vec![1.0_f32, 1.0]],
    )
    .expect("sparse rhs batch");
    (Arc::new(field), array)
}

fn float64_list_array(rows: Vec<Vec<f64>>) -> ListArray {
    ListArray::from_iter_primitive::<Float64Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
    )
}

fn float32_list_array(rows: Vec<Vec<f32>>) -> ListArray {
    ListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
    )
}

fn int32_list_array(rows: Vec<Vec<i32>>) -> ListArray {
    ListArray::from_iter_primitive::<datafusion::arrow::array::types::Int32Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
    )
}

fn u32_list_array(rows: Vec<Vec<u32>>) -> ListArray {
    ListArray::from_iter_primitive::<datafusion::arrow::array::types::UInt32Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
    )
}

fn scalar_float64_list(values: Vec<f64>) -> ScalarValue {
    ScalarValue::List(Arc::new(float64_list_array(vec![values])))
}

fn scalar_float32_list(values: Vec<f32>) -> ScalarValue {
    ScalarValue::List(Arc::new(float32_list_array(vec![values])))
}

fn scalar_int32_list(values: Vec<i32>) -> ScalarValue {
    ScalarValue::List(Arc::new(int32_list_array(vec![values])))
}

fn scalar_nested_float64_list(rows: Vec<Vec<f64>>) -> ScalarValue {
    let nested = float64_list_array(rows);
    let wrapped =
        arrays_into_list_array([Arc::new(nested) as ArrayRef]).expect("single nested list scalar");
    ScalarValue::List(Arc::new(wrapped))
}

fn scalar_nested_float32_list(rows: Vec<Vec<f32>>) -> ScalarValue {
    let nested = float32_list_array(rows);
    let wrapped =
        arrays_into_list_array([Arc::new(nested) as ArrayRef]).expect("single nested list scalar");
    ScalarValue::List(Arc::new(wrapped))
}

fn f64_array(values: &ColumnarValue) -> &Float64Array {
    let ColumnarValue::Array(array) = values else {
        panic!("expected array output");
    };
    array.as_any().downcast_ref::<Float64Array>().expect("expected Float64Array")
}

fn f32_array(values: &ColumnarValue) -> &Float32Array {
    let ColumnarValue::Array(array) = values else {
        panic!("expected array output");
    };
    array.as_any().downcast_ref::<Float32Array>().expect("expected Float32Array")
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

fn array_data_type(values: &ColumnarValue) -> &DataType {
    let ColumnarValue::Array(array) = values else {
        panic!("expected array output");
    };
    array.data_type()
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
fn constructor_udfs_build_fixed_shape_contracts() {
    let make_vector_udf = udfs::make_vector_udf();
    let make_matrix_udf = udfs::make_matrix_udf();
    let make_tensor_udf = udfs::make_tensor_udf();

    let vector_values = scalar_float64_list(vec![3.0, 4.0]);
    let vector_field =
        Arc::new(Field::new("values", DataType::new_list(DataType::Float64, false), false));
    let (vector_return_field, vector_output) = invoke_udf(
        &make_vector_udf,
        vec![
            ColumnarValue::Scalar(vector_values.clone()),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![Arc::clone(&vector_field), Arc::new(Field::new("dim", DataType::Int64, false))],
        &[Some(vector_values), Some(ScalarValue::Int64(Some(2)))],
        1,
    )
    .expect("make_vector");
    assert_eq!(
        vector_return_field.data_type(),
        &DataType::new_fixed_size_list(DataType::Float64, 2, false)
    );
    let vector_output =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(fixed_size_list_array(&vector_output))
            .expect("vector output");
    assert_close(vector_output[[0, 0]], 3.0);
    assert_close(vector_output[[0, 1]], 4.0);

    let matrix_values = scalar_nested_float64_list(vec![vec![1.0, 2.0], vec![3.0, 4.0]]);
    let matrix_field = Arc::new(Field::new(
        "matrix_values",
        DataType::new_list(DataType::new_list(DataType::Float64, false), false),
        false,
    ));
    let (matrix_return_field, matrix_output) = invoke_udf(
        &make_matrix_udf,
        vec![
            ColumnarValue::Scalar(matrix_values.clone()),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::clone(&matrix_field),
            Arc::new(Field::new("rows", DataType::Int64, false)),
            Arc::new(Field::new("cols", DataType::Int64, false)),
        ],
        &[
            Some(matrix_values),
            Some(ScalarValue::Int64(Some(2))),
            Some(ScalarValue::Int64(Some(2))),
        ],
        1,
    )
    .expect("make_matrix");
    let matrix_output = fixed_shape_view3(&matrix_return_field, &matrix_output);
    assert_close(matrix_output[[0, 0, 0]], 1.0);
    assert_close(matrix_output[[0, 0, 1]], 2.0);
    assert_close(matrix_output[[0, 1, 0]], 3.0);
    assert_close(matrix_output[[0, 1, 1]], 4.0);

    let tensor_values =
        float64_list_array(vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]);
    let (tensor_return_field, tensor_output) = invoke_udf(
        &make_tensor_udf,
        vec![
            ColumnarValue::Array(Arc::new(tensor_values)),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::new(Field::new(
                "tensor_values",
                DataType::new_list(DataType::Float64, false),
                false,
            )),
            Arc::new(Field::new("dim0", DataType::Int64, false)),
            Arc::new(Field::new("dim1", DataType::Int64, false)),
        ],
        &[None, Some(ScalarValue::Int64(Some(2))), Some(ScalarValue::Int64(Some(2)))],
        2,
    )
    .expect("make_tensor");
    let tensor_output = fixed_shape_view3(&tensor_return_field, &tensor_output);
    assert_close(tensor_output[[0, 0, 0]], 1.0);
    assert_close(tensor_output[[0, 1, 1]], 4.0);
    assert_close(tensor_output[[1, 0, 0]], 5.0);
    assert_close(tensor_output[[1, 1, 1]], 8.0);
}

#[test]
fn constructor_udfs_build_variable_and_sparse_contracts() {
    let make_variable_tensor_udf = udfs::make_variable_tensor_udf();
    let make_csr_matrix_batch_udf = udfs::make_csr_matrix_batch_udf();

    let variable_data = float64_list_array(vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0]]);
    let variable_shape = int32_list_array(vec![vec![2, 2], vec![1, 3]]);
    let (variable_return_field, variable_output) = invoke_udf(
        &make_variable_tensor_udf,
        vec![
            ColumnarValue::Array(Arc::new(variable_data)),
            ColumnarValue::Array(Arc::new(variable_shape)),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::new(Field::new("data", DataType::new_list(DataType::Float64, false), false)),
            Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("rank", DataType::Int64, false)),
        ],
        &[None, None, Some(ScalarValue::Int64(Some(2)))],
        2,
    )
    .expect("make_variable_tensor");
    let mut variable_rows = variable_shape_rows(&variable_return_field, &variable_output);
    let (_, tensor0) = variable_rows.next().expect("row0").expect("row0 tensor");
    let tensor0 = tensor0.into_dimensionality::<Ix2>().expect("rank-2 tensor");
    let (_, tensor1) = variable_rows.next().expect("row1").expect("row1 tensor");
    let tensor1 = tensor1.into_dimensionality::<Ix2>().expect("rank-2 tensor");
    assert_close(tensor0[[0, 0]], 1.0);
    assert_close(tensor0[[1, 1]], 4.0);
    assert_close(tensor1[[0, 0]], 5.0);
    assert_close(tensor1[[0, 2]], 7.0);

    let csr_shape = scalar_int32_list(vec![2, 3]);
    let csr_row_ptrs = scalar_int32_list(vec![0, 2, 3]);
    let csr_col_indices = ScalarValue::List(Arc::new(u32_list_array(vec![vec![0, 2, 1]])));
    let csr_values = scalar_float64_list(vec![1.0, 2.0, 3.0]);
    let (csr_return_field, csr_output) = invoke_udf(
        &make_csr_matrix_batch_udf,
        vec![
            ColumnarValue::Scalar(csr_shape.clone()),
            ColumnarValue::Scalar(csr_row_ptrs.clone()),
            ColumnarValue::Scalar(csr_col_indices.clone()),
            ColumnarValue::Scalar(csr_values.clone()),
        ],
        vec![
            Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false)),
            Arc::new(Field::new("values", DataType::new_list(DataType::Float64, false), false)),
        ],
        &[Some(csr_shape), Some(csr_row_ptrs), Some(csr_col_indices), Some(csr_values)],
        1,
    )
    .expect("make_csr_matrix_batch");
    let mut csr_rows = ndarrow::csr_matrix_batch_iter::<Float64Type>(
        csr_return_field.as_ref(),
        struct_array(&csr_output),
    )
    .expect("csr batch output");
    let (_, csr0) = csr_rows.next().expect("row0").expect("row0 csr");
    let csr0 = csr_to_dense(&csr0);
    assert_eq!(csr0.shape(), &[2, 3]);
    assert_close(csr0[[0, 0]], 1.0);
    assert_close(csr0[[0, 2]], 2.0);
    assert_close(csr0[[1, 1]], 3.0);
}

#[test]
fn constructor_udfs_reject_invalid_shapes() {
    let make_vector_udf = udfs::make_vector_udf();
    let make_matrix_udf = udfs::make_matrix_udf();
    let make_variable_tensor_udf = udfs::make_variable_tensor_udf();
    let make_csr_matrix_batch_udf = udfs::make_csr_matrix_batch_udf();

    let vector_error = invoke_udf_error(
        &make_vector_udf,
        vec![
            ColumnarValue::Array(Arc::new(float64_list_array(vec![vec![1.0, 2.0, 3.0]]))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::new(Field::new("values", DataType::new_list(DataType::Float64, false), false)),
            Arc::new(Field::new("dim", DataType::Int64, false)),
        ],
        &[None, Some(ScalarValue::Int64(Some(2)))],
        1,
    );
    assert!(vector_error.contains("expected length 2"));

    let matrix_error = invoke_udf_error(
        &make_matrix_udf,
        vec![
            ColumnarValue::Scalar(scalar_nested_float64_list(vec![vec![1.0], vec![2.0, 3.0]])),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::new(Field::new(
                "matrix_values",
                DataType::new_list(DataType::new_list(DataType::Float64, false), false),
                false,
            )),
            Arc::new(Field::new("rows", DataType::Int64, false)),
            Arc::new(Field::new("cols", DataType::Int64, false)),
        ],
        &[
            Some(scalar_nested_float64_list(vec![vec![1.0], vec![2.0, 3.0]])),
            Some(ScalarValue::Int64(Some(2))),
            Some(ScalarValue::Int64(Some(2))),
        ],
        1,
    );
    assert!(matrix_error.contains("nested row 0 expected width 2"));

    let variable_error = invoke_udf_error(
        &make_variable_tensor_udf,
        vec![
            ColumnarValue::Array(Arc::new(float64_list_array(vec![vec![1.0, 2.0]]))),
            ColumnarValue::Array(Arc::new(int32_list_array(vec![vec![-1, 2]]))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::new(Field::new("data", DataType::new_list(DataType::Float64, false), false)),
            Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("rank", DataType::Int64, false)),
        ],
        &[None, None, Some(ScalarValue::Int64(Some(2)))],
        1,
    );
    assert!(variable_error.contains("negative dimension"));

    let csr_error = invoke_udf_error(
        &make_csr_matrix_batch_udf,
        vec![
            ColumnarValue::Array(Arc::new(int32_list_array(vec![vec![2, 3]]))),
            ColumnarValue::Array(Arc::new(int32_list_array(vec![vec![0, 1]]))),
            ColumnarValue::Array(Arc::new(u32_list_array(vec![vec![0, 1, 2]]))),
            ColumnarValue::Array(Arc::new(float64_list_array(vec![vec![1.0, 2.0, 3.0]]))),
        ],
        vec![
            Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false)),
            Arc::new(Field::new("values", DataType::new_list(DataType::Float64, false), false)),
        ],
        &[None, None, None, None],
        1,
    );
    assert!(csr_error.contains("row_ptrs"));
}

#[test]
fn constructor_udfs_support_empty_batches() {
    let make_variable_tensor_udf = udfs::make_variable_tensor_udf();
    let make_csr_matrix_batch_udf = udfs::make_csr_matrix_batch_udf();

    let (variable_return_field, variable_output) = invoke_udf(
        &make_variable_tensor_udf,
        vec![
            ColumnarValue::Array(Arc::new(float64_list_array(Vec::new()))),
            ColumnarValue::Array(Arc::new(int32_list_array(Vec::new()))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::new(Field::new("data", DataType::new_list(DataType::Float64, false), false)),
            Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("rank", DataType::Int64, false)),
        ],
        &[None, None, Some(ScalarValue::Int64(Some(2)))],
        0,
    )
    .expect("empty variable tensor batch");
    assert_eq!(variable_return_field.extension_type_name(), Some("arrow.variable_shape_tensor"));
    assert_eq!(struct_array(&variable_output).len(), 0);

    let (csr_return_field, csr_output) = invoke_udf(
        &make_csr_matrix_batch_udf,
        vec![
            ColumnarValue::Array(Arc::new(int32_list_array(Vec::new()))),
            ColumnarValue::Array(Arc::new(int32_list_array(Vec::new()))),
            ColumnarValue::Array(Arc::new(u32_list_array(Vec::new()))),
            ColumnarValue::Array(Arc::new(float64_list_array(Vec::new()))),
        ],
        vec![
            Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false)),
            Arc::new(Field::new("col_indices", DataType::new_list(DataType::UInt32, false), false)),
            Arc::new(Field::new("values", DataType::new_list(DataType::Float64, false), false)),
        ],
        &[None, None, None, None],
        0,
    )
    .expect("empty csr batch");
    assert_eq!(csr_return_field.extension_type_name(), Some("ndarrow.csr_matrix_batch"));
    assert_eq!(struct_array(&csr_output).len(), 0);
}

#[test]
fn vector_udfs_cover_real_batch_ops() {
    let left = fixed_size_list([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]]);
    let right = fixed_size_list([[4.0, 0.0, 3.0], [2.0, 2.0, 1.0]]);
    let left_field = vector_field("left", &DataType::Float64, 3, false).expect("vector field");
    let right_field = vector_field("right", &DataType::Float64, 3, false).expect("vector field");
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
    let rhs_field = vector_field("rhs", &DataType::Float64, 2, false).expect("rhs field");
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
fn matrix_matvec_udf_covers_rowwise_matrix_vector_products() {
    let (matrix_field, matrices) =
        matrix_batch("matrices", [[[1.0, 2.0], [3.0, 4.0]], [[2.0, 0.0], [1.0, 2.0]]]);
    let vectors = fixed_size_list([[2.0, 1.0], [3.0, 4.0]]);
    let vector_field = vector_field("vectors", &DataType::Float64, 2, false).expect("vector field");
    let matvec_udf = udfs::matrix_matvec_udf();

    let (_, product) = invoke_udf(
        &matvec_udf,
        vec![ColumnarValue::Array(Arc::new(matrices)), ColumnarValue::Array(Arc::new(vectors))],
        vec![matrix_field, vector_field],
        &[None, None],
        2,
    )
    .expect("matrix_matvec");
    let product =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(fixed_size_list_array(&product))
            .expect("matrix-vector product");
    assert_close(product[[0, 0]], 4.0);
    assert_close(product[[0, 1]], 10.0);
    assert_close(product[[1, 0]], 6.0);
    assert_close(product[[1, 1]], 11.0);
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
    let response_field =
        vector_field("response", &DataType::Float64, 3, false).expect("response field");
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
    let rhs_field = vector_field("rhs", &DataType::Float64, 2, false).expect("rhs field");

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
fn matrix_qr_and_svd_helper_udfs_cover_solver_scalar_and_inverse_outputs() {
    let qr_solve_udf = udfs::matrix_qr_solve_least_squares_udf();
    let qr_condition_udf = udfs::matrix_qr_condition_number_udf();
    let svd_pseudo_inverse_udf = udfs::matrix_svd_pseudo_inverse_udf();
    let svd_condition_udf = udfs::matrix_svd_condition_number_udf();
    let svd_rank_udf = udfs::matrix_svd_rank_udf();

    let (qr_field, qr_matrices) =
        matrix_batch("qr", [[[2.0, 0.0], [0.0, 1.0]], [[3.0, 0.0], [0.0, 4.0]]]);
    let rhs = fixed_size_list([[4.0, 3.0], [9.0, 8.0]]);
    let rhs_field = vector_field("rhs", &DataType::Float64, 2, false).expect("rhs field");
    let (_, qr_solution) = invoke_udf(
        &qr_solve_udf,
        vec![
            ColumnarValue::Array(Arc::new(qr_matrices.clone())),
            ColumnarValue::Array(Arc::new(rhs)),
        ],
        vec![Arc::clone(&qr_field), rhs_field],
        &[None, None],
        2,
    )
    .expect("matrix_qr_solve_least_squares");
    let qr_solution =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(fixed_size_list_array(&qr_solution))
            .expect("least-squares solution");
    assert_close(qr_solution[[0, 0]], 2.0);
    assert_close(qr_solution[[0, 1]], 3.0);
    assert_close(qr_solution[[1, 0]], 3.0);
    assert_close(qr_solution[[1, 1]], 2.0);

    let (_, qr_condition) = invoke_udf(
        &qr_condition_udf,
        vec![ColumnarValue::Array(Arc::new(qr_matrices))],
        vec![qr_field],
        &[None],
        2,
    )
    .expect("matrix_qr_condition_number");
    assert_close(f64_array(&qr_condition).value(0), 2.0);
    assert_close(f64_array(&qr_condition).value(1), 4.0 / 3.0);

    let (svd_field, svd_matrices) =
        matrix_batch("svd", [[[4.0, 0.0], [0.0, 2.0]], [[9.0, 0.0], [0.0, 3.0]]]);
    let (pseudo_inverse_field, pseudo_inverse) = invoke_udf(
        &svd_pseudo_inverse_udf,
        vec![ColumnarValue::Array(Arc::new(svd_matrices.clone()))],
        vec![Arc::clone(&svd_field)],
        &[None],
        2,
    )
    .expect("matrix_svd_pseudo_inverse");
    let pseudo_inverse = fixed_shape_view3(&pseudo_inverse_field, &pseudo_inverse);
    assert_close(pseudo_inverse[[0, 0, 0]], 0.25);
    assert_close(pseudo_inverse[[0, 1, 1]], 0.5);
    assert_close(pseudo_inverse[[1, 0, 0]], 1.0 / 9.0);
    assert_close(pseudo_inverse[[1, 1, 1]], 1.0 / 3.0);

    let (_, svd_condition) = invoke_udf(
        &svd_condition_udf,
        vec![ColumnarValue::Array(Arc::new(svd_matrices.clone()))],
        vec![Arc::clone(&svd_field)],
        &[None],
        2,
    )
    .expect("matrix_svd_condition_number");
    assert_close(f64_array(&svd_condition).value(0), 2.0);
    assert_close(f64_array(&svd_condition).value(1), 3.0);

    let (rank_field, rank_matrices) =
        matrix_batch("rank", [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 1.0]]]);
    let (_, rank) = invoke_udf(
        &svd_rank_udf,
        vec![ColumnarValue::Array(Arc::new(rank_matrices))],
        vec![rank_field],
        &[None],
        2,
    )
    .expect("matrix_svd_rank");
    let ColumnarValue::Array(rank) = rank else {
        panic!("expected rank array output");
    };
    let rank = rank.as_any().downcast_ref::<Int64Array>().expect("rank int64");
    assert_eq!(rank.value(0), 2);
    assert_eq!(rank.value(1), 1);
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
    let vector_field = vector_field("vector", &DataType::Float64, 3, false).expect("vector field");

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
    assert!(field_error.contains("expected FixedSizeList<Float32|Float64>(D)"));
}

#[test]
fn matrix_udfs_validate_shape_contracts() {
    let matvec_udf = udfs::matrix_matvec_udf();
    let matmul_udf = udfs::matrix_matmul_udf();
    let lu_udf = udfs::matrix_lu_udf();
    let lu_solve_udf = udfs::matrix_lu_solve_udf();
    let qr_solve_udf = udfs::matrix_qr_solve_least_squares_udf();
    let (left_field, left) = matrix_batch("left", [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
    let (right_field, right) = matrix_batch("right", [[[1.0, 0.0], [0.0, 1.0]]]);
    let (nonsquare_field, nonsquare) =
        matrix_batch("nonsquare", [[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]]);
    let rhs = fixed_size_list([[1.0, 2.0]]);
    let rhs_field = vector_field("rhs", &DataType::Float64, 2, false).expect("rhs field");

    let matmul_error = invoke_udf_error(
        &matmul_udf,
        vec![ColumnarValue::Array(Arc::new(left)), ColumnarValue::Array(Arc::new(right))],
        vec![left_field, right_field],
        &[None, None],
        1,
    );
    let matvec_error = invoke_udf_error(
        &matvec_udf,
        vec![
            ColumnarValue::Array(Arc::new(nonsquare.clone())),
            ColumnarValue::Array(Arc::new(rhs.clone())),
        ],
        vec![Arc::clone(&nonsquare_field), Arc::clone(&rhs_field)],
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
    let qr_rhs_error = invoke_udf_error(
        &qr_solve_udf,
        vec![
            ColumnarValue::Array(Arc::new(matrix_batch("qr", [[[1.0, 2.0], [3.0, 4.0]]]).1)),
            ColumnarValue::Array(Arc::new(fixed_size_list([[1.0, 2.0, 3.0]]))),
        ],
        vec![
            matrix_batch("qr_field", [[[1.0, 2.0], [3.0, 4.0]]]).0,
            vector_field("rhs_long", &DataType::Float64, 3, false).expect("rhs field"),
        ],
        &[None, None],
        1,
    );

    assert!(matmul_error.contains("incompatible matrix shapes"));
    assert!(matvec_error.contains("rhs vector length mismatch"));
    assert!(lu_error.contains("matrix_lu requires square matrices"));
    assert!(rhs_error.contains("matrix_lu_solve requires square matrices"));
    assert!(qr_rhs_error.contains("rhs vector length mismatch"));
}

#[test]
fn matrix_lu_solve_and_linear_regression_validate_runtime_batch_edges() {
    let lu_solve_udf = udfs::matrix_lu_solve_udf();
    let linear_regression_udf = udfs::linear_regression_udf();
    let (matrices_field, matrices) =
        matrix_batch("matrices", [[[1.0, 0.0], [0.0, 1.0]], [[2.0, 1.0], [1.0, 2.0]]]);
    let short_rhs = fixed_size_list([[1.0, 2.0]]);
    let short_rhs_field = vector_field("rhs", &DataType::Float64, 2, false).expect("rhs field");
    let (design_field, design) =
        matrix_batch("design", [[[1.0, 0.0], [0.0, 1.0]], [[1.0, 1.0], [1.0, 0.0]]]);
    let response = fixed_size_list([[1.0, 2.0]]);
    let response_field =
        vector_field("response", &DataType::Float64, 2, false).expect("response field");
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
    let rank_two_vectors = crate::metadata::variable_shape_tensor_field(
        "vectors",
        &DataType::Float64,
        2,
        Some(&[None, None] as &[Option<i32>]),
        false,
    )
    .expect("rank-2 ragged field");
    let rank_one_tensor =
        crate::metadata::fixed_shape_tensor_field("tensor", &DataType::Float64, &[4], false)
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

#[test]
fn float32_constructor_udfs_reject_invalid_shapes() {
    let values_field =
        Arc::new(Field::new("values", DataType::new_list(DataType::Float32, false), false));
    let int_list_field =
        Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false));
    let rows_field = Arc::new(Field::new("rows", DataType::Int64, false));
    let cols_field = Arc::new(Field::new("cols", DataType::Int64, false));
    let rank_field = Arc::new(Field::new("rank", DataType::Int64, false));
    let row_ptrs_field =
        Arc::new(Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false));
    let indices_field =
        Arc::new(Field::new("indices", DataType::new_list(DataType::UInt32, false), false));

    let matrix_error = invoke_udf_error(
        &udfs::make_matrix_udf(),
        vec![
            ColumnarValue::Scalar(scalar_float32_list(vec![1.0, 2.0, 3.0])),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![Arc::clone(&values_field), Arc::clone(&rows_field), Arc::clone(&cols_field)],
        &[
            Some(scalar_float32_list(vec![1.0, 2.0, 3.0])),
            Some(ScalarValue::Int64(Some(2))),
            Some(ScalarValue::Int64(Some(2))),
        ],
        1,
    );
    let variable_error = invoke_udf_error(
        &udfs::make_variable_tensor_udf(),
        vec![
            ColumnarValue::Scalar(scalar_float32_list(vec![1.0, 2.0])),
            ColumnarValue::Scalar(scalar_int32_list(vec![1, 2])),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(1))),
        ],
        vec![Arc::clone(&values_field), Arc::clone(&int_list_field), Arc::clone(&rank_field)],
        &[
            Some(scalar_float32_list(vec![1.0, 2.0])),
            Some(scalar_int32_list(vec![1, 2])),
            Some(ScalarValue::Int64(Some(1))),
        ],
        1,
    );
    let csr_error = invoke_udf_error(
        &udfs::make_csr_matrix_batch_udf(),
        vec![
            ColumnarValue::Array(Arc::new(int32_list_array(vec![vec![2, 2]]))),
            ColumnarValue::Array(Arc::new(int32_list_array(vec![vec![0, 1, 2], vec![0, 1]]))),
            ColumnarValue::Array(Arc::new(u32_list_array(vec![vec![0, 1]]))),
            ColumnarValue::Array(Arc::new(float32_list_array(vec![vec![1.0, 2.0]]))),
        ],
        vec![int_list_field, row_ptrs_field, indices_field, values_field],
        &[None, None, None, None],
        2,
    );

    assert!(matrix_error.contains("expected 4 row-major values"));
    assert!(variable_error.contains("expected rank 1"));
    assert!(csr_error.contains("argument length mismatch"));
}

#[test]
fn float32_tensor_and_sparse_udfs_validate_contract_edges() {
    let sparse_matvec_udf = udfs::sparse_matvec_udf();
    let tensor_sum_udf = udfs::tensor_sum_last_axis_udf();
    let variable_sum_udf = udfs::tensor_variable_sum_last_axis_udf();
    let (sparse_field, sparse) = sparse_batch_f32("sparse");
    let (ragged_vectors_field, ragged_vectors) =
        ragged_vectors_f32("vectors", vec![vec![1.0, 2.0], vec![3.0]]);
    let rank_two_vectors = crate::metadata::variable_shape_tensor_field(
        "vectors",
        &DataType::Float32,
        2,
        Some(&[None, None] as &[Option<i32>]),
        false,
    )
    .expect("rank-2 ragged field");
    let rank_one_tensor =
        crate::metadata::fixed_shape_tensor_field("tensor", &DataType::Float32, &[4], false)
            .expect("rank-1 tensor field");
    let rank_one_tensor_values = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        vec![Some(vec![Some(1.0), Some(2.0), Some(3.0), Some(4.0)])],
        4,
    );

    let sparse_rank_error = invoke_udf_error(
        &sparse_matvec_udf,
        vec![
            ColumnarValue::Array(Arc::new(sparse)),
            ColumnarValue::Array(Arc::new(ragged_vectors.clone())),
        ],
        vec![sparse_field, rank_two_vectors],
        &[None, None],
        2,
    );
    let tensor_rank_error = invoke_udf_error(
        &tensor_sum_udf,
        vec![ColumnarValue::Array(Arc::new(rank_one_tensor_values))],
        vec![rank_one_tensor],
        &[None],
        1,
    );
    let variable_rank_error = invoke_udf_error(
        &variable_sum_udf,
        vec![ColumnarValue::Array(Arc::new(ragged_vectors))],
        vec![ragged_vectors_field],
        &[None],
        2,
    );

    assert!(sparse_rank_error.contains("batch of rank-1 dense vectors"));
    assert!(tensor_rank_error.contains("requires tensors with rank >= 2"));
    assert!(variable_rank_error.contains("requires tensors with rank >= 2"));
}

#[test]
fn dense_constructor_udfs_cover_float32_branches() {
    let values_field =
        Arc::new(Field::new("values", DataType::new_list(DataType::Float32, false), false));
    let nested_values_field = Arc::new(Field::new(
        "nested_values",
        DataType::new_list(DataType::new_list(DataType::Float32, false), false),
        false,
    ));

    let (vector_field_out, vector_output) = invoke_udf(
        &udfs::make_vector_udf(),
        vec![
            ColumnarValue::Scalar(scalar_float32_list(vec![3.0, 4.0])),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![Arc::clone(&values_field), Arc::new(Field::new("dim", DataType::Int64, false))],
        &[Some(scalar_float32_list(vec![3.0, 4.0])), Some(ScalarValue::Int64(Some(2)))],
        1,
    )
    .expect("make_vector f32");
    assert_eq!(
        vector_field_out.data_type(),
        &DataType::new_fixed_size_list(DataType::Float32, 2, false)
    );
    assert_eq!(array_data_type(&vector_output), vector_field_out.data_type());

    let (matrix_field_out, matrix_output) = invoke_udf(
        &udfs::make_matrix_udf(),
        vec![
            ColumnarValue::Scalar(scalar_nested_float32_list(vec![vec![1.0, 2.0], vec![3.0, 4.0]])),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::clone(&nested_values_field),
            Arc::new(Field::new("rows", DataType::Int64, false)),
            Arc::new(Field::new("cols", DataType::Int64, false)),
        ],
        &[
            Some(scalar_nested_float32_list(vec![vec![1.0, 2.0], vec![3.0, 4.0]])),
            Some(ScalarValue::Int64(Some(2))),
            Some(ScalarValue::Int64(Some(2))),
        ],
        1,
    )
    .expect("make_matrix f32");
    assert_eq!(array_data_type(&matrix_output), matrix_field_out.data_type());

    let (tensor_field_out, tensor_output) = invoke_udf(
        &udfs::make_tensor_udf(),
        vec![
            ColumnarValue::Scalar(scalar_float32_list(vec![1.0, 2.0, 3.0, 4.0])),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::clone(&values_field),
            Arc::new(Field::new("d0", DataType::Int64, false)),
            Arc::new(Field::new("d1", DataType::Int64, false)),
        ],
        &[
            Some(scalar_float32_list(vec![1.0, 2.0, 3.0, 4.0])),
            Some(ScalarValue::Int64(Some(2))),
            Some(ScalarValue::Int64(Some(2))),
        ],
        1,
    )
    .expect("make_tensor f32");
    assert_eq!(array_data_type(&tensor_output), tensor_field_out.data_type());
}

#[test]
fn variable_and_sparse_constructor_udfs_cover_float32_branches() {
    let values_field =
        Arc::new(Field::new("values", DataType::new_list(DataType::Float32, false), false));
    let int_list_field =
        Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false));
    let u32_list_field =
        Arc::new(Field::new("indices", DataType::new_list(DataType::UInt32, false), false));

    let (variable_field_out, variable_output) = invoke_udf(
        &udfs::make_variable_tensor_udf(),
        vec![
            ColumnarValue::Scalar(scalar_float32_list(vec![1.0, 2.0, 3.0])),
            ColumnarValue::Scalar(scalar_int32_list(vec![3])),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(1))),
        ],
        vec![
            Arc::clone(&values_field),
            Arc::clone(&int_list_field),
            Arc::new(Field::new("rank", DataType::Int64, false)),
        ],
        &[
            Some(scalar_float32_list(vec![1.0, 2.0, 3.0])),
            Some(scalar_int32_list(vec![3])),
            Some(ScalarValue::Int64(Some(1))),
        ],
        1,
    )
    .expect("make_variable_tensor f32");
    assert_eq!(array_data_type(&variable_output), variable_field_out.data_type());

    let (csr_field_out, csr_output) = invoke_udf(
        &udfs::make_csr_matrix_batch_udf(),
        vec![
            ColumnarValue::Array(Arc::new(int32_list_array(vec![vec![2, 2]]))),
            ColumnarValue::Array(Arc::new(int32_list_array(vec![vec![0, 1, 2]]))),
            ColumnarValue::Array(Arc::new(u32_list_array(vec![vec![0, 1]]))),
            ColumnarValue::Array(Arc::new(float32_list_array(vec![vec![5.0, 6.0]]))),
        ],
        vec![
            Arc::clone(&int_list_field),
            Arc::new(Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false)),
            u32_list_field,
            values_field,
        ],
        &[None, None, None, None],
        1,
    )
    .expect("make_csr_matrix_batch f32");
    assert_eq!(array_data_type(&csr_output), csr_field_out.data_type());
}

#[test]
fn float32_constructor_udfs_cover_flat_and_empty_paths() {
    let values_field =
        Arc::new(Field::new("values", DataType::new_list(DataType::Float32, false), false));
    let int_list_field =
        Arc::new(Field::new("shape", DataType::new_list(DataType::Int32, false), false));
    let u32_list_field =
        Arc::new(Field::new("indices", DataType::new_list(DataType::UInt32, false), false));

    let (matrix_field_out, matrix_output) = invoke_udf(
        &udfs::make_matrix_udf(),
        vec![
            ColumnarValue::Scalar(scalar_float32_list(vec![1.0, 2.0, 3.0, 4.0])),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(2))),
        ],
        vec![
            Arc::clone(&values_field),
            Arc::new(Field::new("rows", DataType::Int64, false)),
            Arc::new(Field::new("cols", DataType::Int64, false)),
        ],
        &[
            Some(scalar_float32_list(vec![1.0, 2.0, 3.0, 4.0])),
            Some(ScalarValue::Int64(Some(2))),
            Some(ScalarValue::Int64(Some(2))),
        ],
        1,
    )
    .expect("make_matrix flat f32");
    assert_eq!(array_data_type(&matrix_output), matrix_field_out.data_type());

    let (variable_field_out, variable_output) = invoke_udf(
        &udfs::make_variable_tensor_udf(),
        vec![
            ColumnarValue::Array(Arc::new(float32_list_array(vec![]))),
            ColumnarValue::Array(Arc::new(int32_list_array(vec![]))),
            ColumnarValue::Scalar(ScalarValue::Int64(Some(1))),
        ],
        vec![
            Arc::clone(&values_field),
            Arc::clone(&int_list_field),
            Arc::new(Field::new("rank", DataType::Int64, false)),
        ],
        &[None, None, Some(ScalarValue::Int64(Some(1)))],
        0,
    )
    .expect("make_variable_tensor empty f32");
    assert_eq!(array_data_type(&variable_output), variable_field_out.data_type());

    let (csr_field_out, csr_output) = invoke_udf(
        &udfs::make_csr_matrix_batch_udf(),
        vec![
            ColumnarValue::Array(Arc::new(int32_list_array(vec![]))),
            ColumnarValue::Array(Arc::new(int32_list_array(vec![]))),
            ColumnarValue::Array(Arc::new(u32_list_array(vec![]))),
            ColumnarValue::Array(Arc::new(float32_list_array(vec![]))),
        ],
        vec![
            int_list_field,
            Arc::new(Field::new("row_ptrs", DataType::new_list(DataType::Int32, false), false)),
            u32_list_field,
            values_field,
        ],
        &[None, None, None, None],
        0,
    )
    .expect("make_csr_matrix_batch empty f32");
    assert_eq!(array_data_type(&csr_output), csr_field_out.data_type());
}

#[test]
fn vector_udfs_cover_float32_branches() {
    let vectors = fixed_size_list_f32([[3.0, 4.0], [5.0, 12.0]]);
    let other = fixed_size_list_f32([[1.0, 2.0], [2.0, 1.0]]);
    let vector_field = vector_field("vector", &DataType::Float32, 2, false).expect("vector field");

    let (_norm_field, norm_output) = invoke_udf(
        &udfs::vector_l2_norm_udf(),
        vec![ColumnarValue::Array(Arc::new(vectors.clone()))],
        vec![Arc::clone(&vector_field)],
        &[None],
        2,
    )
    .expect("vector_l2_norm f32");
    assert!((f32_array(&norm_output).value(0) - 5.0_f32).abs() < 1.0e-6_f32);

    let (_dot_field, dot_output) = invoke_udf(
        &udfs::vector_dot_udf(),
        vec![
            ColumnarValue::Array(Arc::new(vectors.clone())),
            ColumnarValue::Array(Arc::new(other.clone())),
        ],
        vec![Arc::clone(&vector_field), Arc::clone(&vector_field)],
        &[None, None],
        2,
    )
    .expect("vector_dot f32");
    assert!((f32_array(&dot_output).value(0) - 11.0_f32).abs() < 1.0e-6_f32);

    let (_cos_field, cos_output) = invoke_udf(
        &udfs::vector_cosine_similarity_udf(),
        vec![
            ColumnarValue::Array(Arc::new(vectors.clone())),
            ColumnarValue::Array(Arc::new(other.clone())),
        ],
        vec![Arc::clone(&vector_field), Arc::clone(&vector_field)],
        &[None, None],
        2,
    )
    .expect("vector_cosine_similarity f32");
    assert!(f32_array(&cos_output).value(0) > 0.0);

    let (_distance_field, distance_output) = invoke_udf(
        &udfs::vector_cosine_distance_udf(),
        vec![
            ColumnarValue::Array(Arc::new(vectors.clone())),
            ColumnarValue::Array(Arc::new(other)),
        ],
        vec![Arc::clone(&vector_field), Arc::clone(&vector_field)],
        &[None, None],
        2,
    )
    .expect("vector_cosine_distance f32");
    assert!(f32_array(&distance_output).value(0) >= 0.0);

    let (normalize_field, normalize_output) = invoke_udf(
        &udfs::vector_normalize_udf(),
        vec![ColumnarValue::Array(Arc::new(vectors))],
        vec![vector_field],
        &[None],
        2,
    )
    .expect("vector_normalize f32");
    assert_eq!(array_data_type(&normalize_output), normalize_field.data_type());
}

#[test]
fn matrix_and_decomposition_udfs_cover_float32_branches() {
    let (matrix_field, matrices) = matrix_batch_f32("matrix", [[[4.0, 1.0], [1.0, 3.0]]]);
    let rhs_field = vector_field("rhs", &DataType::Float32, 2, false).expect("rhs field");
    let rhs = fixed_size_list_f32([[1.0, 2.0]]);

    let (matmul_field, matmul_output) = invoke_udf(
        &udfs::matrix_matmul_udf(),
        vec![
            ColumnarValue::Array(Arc::new(matrices.clone())),
            ColumnarValue::Array(Arc::new(matrices.clone())),
        ],
        vec![Arc::clone(&matrix_field), Arc::clone(&matrix_field)],
        &[None, None],
        1,
    )
    .expect("matrix_matmul f32");
    assert_eq!(array_data_type(&matmul_output), matmul_field.data_type());

    let (matvec_field, matvec_output) = invoke_udf(
        &udfs::matrix_matvec_udf(),
        vec![
            ColumnarValue::Array(Arc::new(matrices.clone())),
            ColumnarValue::Array(Arc::new(rhs.clone())),
        ],
        vec![Arc::clone(&matrix_field), Arc::clone(&rhs_field)],
        &[None, None],
        1,
    )
    .expect("matrix_matvec f32");
    assert_eq!(array_data_type(&matvec_output), matvec_field.data_type());

    let (solve_field, solve_output) = invoke_udf(
        &udfs::matrix_lu_solve_udf(),
        vec![
            ColumnarValue::Array(Arc::new(matrices.clone())),
            ColumnarValue::Array(Arc::new(rhs.clone())),
        ],
        vec![Arc::clone(&matrix_field), Arc::clone(&rhs_field)],
        &[None, None],
        1,
    )
    .expect("matrix_lu_solve f32");
    assert_eq!(array_data_type(&solve_output), solve_field.data_type());

    let (cholesky_solve_field, cholesky_solve_output) = invoke_udf(
        &udfs::matrix_cholesky_solve_udf(),
        vec![
            ColumnarValue::Array(Arc::new(matrices.clone())),
            ColumnarValue::Array(Arc::new(rhs.clone())),
        ],
        vec![Arc::clone(&matrix_field), Arc::clone(&rhs_field)],
        &[None, None],
        1,
    )
    .expect("matrix_cholesky_solve f32");
    assert_eq!(array_data_type(&cholesky_solve_output), cholesky_solve_field.data_type());

    for udf in [
        udfs::matrix_lu_udf(),
        udfs::matrix_inverse_udf(),
        udfs::matrix_determinant_udf(),
        udfs::matrix_log_determinant_udf(),
        udfs::matrix_cholesky_udf(),
        udfs::matrix_cholesky_inverse_udf(),
        udfs::matrix_qr_udf(),
        udfs::matrix_svd_udf(),
        udfs::matrix_qr_condition_number_udf(),
        udfs::matrix_svd_pseudo_inverse_udf(),
        udfs::matrix_svd_condition_number_udf(),
        udfs::matrix_svd_rank_udf(),
    ] {
        let (_field, output) = invoke_udf(
            &udf,
            vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
            vec![Arc::clone(&matrix_field)],
            &[None],
            1,
        )
        .unwrap_or_else(|error| panic!("{} f32: {error}", udf.name()));
        match output {
            ColumnarValue::Array(_) => {}
            ColumnarValue::Scalar(_) => panic!("expected array output"),
        }
    }
    let (_field, output) = invoke_udf(
        &udfs::matrix_qr_solve_least_squares_udf(),
        vec![ColumnarValue::Array(Arc::new(matrices.clone())), ColumnarValue::Array(Arc::new(rhs))],
        vec![matrix_field, rhs_field],
        &[None, None],
        1,
    )
    .expect("matrix_qr_solve_least_squares f32");
    match output {
        ColumnarValue::Array(_) => {}
        ColumnarValue::Scalar(_) => panic!("expected array output"),
    }
}

#[test]
fn tensor_udfs_cover_float32_branches() {
    let (tensor_field, tensors) =
        tensor_batch4_f32("tensor", [[[[1.0, 2.0], [3.0, 4.0]], [[2.0, 1.0], [0.0, 1.0]]]]);
    let (ragged_matrix_field, ragged_matrices) =
        ragged_matrices_f32("matrices", vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]]]);

    for udf in [
        udfs::tensor_sum_last_axis_udf(),
        udfs::tensor_l2_norm_last_axis_udf(),
        udfs::tensor_normalize_last_axis_udf(),
    ] {
        let (_field, output) = invoke_udf(
            &udf,
            vec![ColumnarValue::Array(Arc::new(tensors.clone()))],
            vec![Arc::clone(&tensor_field)],
            &[None],
            1,
        )
        .unwrap_or_else(|error| panic!("{} f32: {error}", udf.name()));
        match output {
            ColumnarValue::Array(_) => {}
            ColumnarValue::Scalar(_) => panic!("expected array output"),
        }
    }

    let (_field, _output) = invoke_udf(
        &udfs::tensor_batched_dot_last_axis_udf(),
        vec![
            ColumnarValue::Array(Arc::new(tensors.clone())),
            ColumnarValue::Array(Arc::new(tensors.clone())),
        ],
        vec![Arc::clone(&tensor_field), Arc::clone(&tensor_field)],
        &[None, None],
        1,
    )
    .expect("tensor batched dot f32");
    let (_field, _output) = invoke_udf(
        &udfs::tensor_batched_matmul_last_two_udf(),
        vec![
            ColumnarValue::Array(Arc::new(tensors.clone())),
            ColumnarValue::Array(Arc::new(tensors)),
        ],
        vec![Arc::clone(&tensor_field), Arc::clone(&tensor_field)],
        &[None, None],
        1,
    )
    .expect("tensor batched matmul f32");

    for udf in [
        udfs::tensor_variable_sum_last_axis_udf(),
        udfs::tensor_variable_l2_norm_last_axis_udf(),
        udfs::tensor_variable_normalize_last_axis_udf(),
    ] {
        let (_field, _output) = invoke_udf(
            &udf,
            vec![ColumnarValue::Array(Arc::new(ragged_matrices.clone()))],
            vec![Arc::clone(&ragged_matrix_field)],
            &[None],
            1,
        )
        .unwrap_or_else(|error| panic!("{} f32: {error}", udf.name()));
    }
    let (_field, _output) = invoke_udf(
        &udfs::tensor_variable_batched_dot_last_axis_udf(),
        vec![
            ColumnarValue::Array(Arc::new(ragged_matrices.clone())),
            ColumnarValue::Array(Arc::new(ragged_matrices)),
        ],
        vec![Arc::clone(&ragged_matrix_field), Arc::clone(&ragged_matrix_field)],
        &[None, None],
        1,
    )
    .expect("variable tensor batched dot f32");
}

#[test]
fn sparse_udfs_cover_float32_branches() {
    let (ragged_vector_field, ragged_vectors) =
        ragged_vectors_f32("vectors", vec![vec![1.0, 2.0, 3.0], vec![3.0, 4.0]]);
    let (dense_field, dense_matrices) = ragged_matrices_f32("dense", vec![
        vec![vec![1.0, 0.0], vec![0.0, 1.0], vec![1.0, 1.0]],
        vec![vec![1.0, 2.0], vec![3.0, 4.0]],
    ]);
    let (sparse_field, sparse) = sparse_batch_f32("sparse");
    let (sparse_rhs_field, sparse_rhs) = sparse_batch_rhs_f32("sparse_rhs");
    for udf in [
        udfs::sparse_matvec_udf(),
        udfs::sparse_matmat_dense_udf(),
        udfs::sparse_transpose_udf(),
        udfs::sparse_matmat_sparse_udf(),
    ] {
        let args = if udf.name() == "sparse_matvec" {
            vec![
                ColumnarValue::Array(Arc::new(sparse.clone())),
                ColumnarValue::Array(Arc::new(ragged_vectors.clone())),
            ]
        } else if udf.name() == "sparse_matmat_dense" {
            vec![
                ColumnarValue::Array(Arc::new(sparse.clone())),
                ColumnarValue::Array(Arc::new(dense_matrices.clone())),
            ]
        } else if udf.name() == "sparse_transpose" {
            vec![ColumnarValue::Array(Arc::new(sparse.clone()))]
        } else {
            vec![
                ColumnarValue::Array(Arc::new(sparse.clone())),
                ColumnarValue::Array(Arc::new(sparse_rhs.clone())),
            ]
        };
        let fields = if udf.name() == "sparse_matvec" {
            vec![Arc::clone(&sparse_field), Arc::clone(&ragged_vector_field)]
        } else if udf.name() == "sparse_matmat_dense" {
            vec![Arc::clone(&sparse_field), Arc::clone(&dense_field)]
        } else if udf.name() == "sparse_transpose" {
            vec![Arc::clone(&sparse_field)]
        } else {
            vec![Arc::clone(&sparse_field), Arc::clone(&sparse_rhs_field)]
        };
        let (_field, _output) = invoke_udf(&udf, args, fields, &[None, None], 2)
            .unwrap_or_else(|error| panic!("{} f32: {error}", udf.name()));
    }
}

#[test]
fn ml_udfs_cover_float32_branches() {
    let (matrix_field, matrices) =
        matrix_batch_f32("design", [[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]]);
    let response_field =
        vector_field("response", &DataType::Float32, 3, false).expect("response field");
    let responses = fixed_size_list_f32([[1.0, 2.0, 3.0]]);
    let bool_field = Arc::new(Field::new("add_intercept", DataType::Boolean, false));

    for udf in [
        udfs::matrix_column_means_udf(),
        udfs::matrix_center_columns_udf(),
        udfs::matrix_covariance_udf(),
        udfs::matrix_correlation_udf(),
        udfs::matrix_pca_udf(),
    ] {
        let (_field, output) = invoke_udf(
            &udf,
            vec![ColumnarValue::Array(Arc::new(matrices.clone()))],
            vec![Arc::clone(&matrix_field)],
            &[None],
            1,
        )
        .unwrap_or_else(|error| panic!("{} f32: {error}", udf.name()));
        match output {
            ColumnarValue::Array(_) => {}
            ColumnarValue::Scalar(_) => panic!("expected array output"),
        }
    }

    let (regression_field, regression_output) = invoke_udf(
        &udfs::linear_regression_udf(),
        vec![
            ColumnarValue::Array(Arc::new(matrices)),
            ColumnarValue::Array(Arc::new(responses)),
            ColumnarValue::Scalar(ScalarValue::Boolean(Some(true))),
        ],
        vec![matrix_field, response_field, bool_field],
        &[None, None, Some(ScalarValue::Boolean(Some(true)))],
        1,
    )
    .expect("linear_regression f32");
    assert_eq!(array_data_type(&regression_output), regression_field.data_type());
}
