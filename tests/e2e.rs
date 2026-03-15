use std::sync::Arc;

use datafusion::arrow::array::types::{Float32Type, Float64Type};
use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float32Array, Float64Array, Int64Array, ListArray,
    StructArray,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::common::utils::arrays_into_list_array;
use datafusion::execution::FunctionRegistry;
use datafusion::execution::registry::MemoryFunctionRegistry;
use datafusion::prelude::SessionContext;
use ndarray::{Array1, Array2, Array3, Array4, Ix1, Ix2, Ix3, Ix4};
use num_complex::Complex64;

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(delta < 1.0e-9, "expected {expected}, got {actual}, delta {delta}");
}

fn assert_close32(actual: f32, expected: f32) {
    let delta = (actual - expected).abs();
    assert!(delta < 1.0e-5, "expected {expected}, got {actual}, delta {delta}");
}

fn float64_list_array(rows: Vec<Vec<f64>>) -> ListArray {
    ListArray::from_iter_primitive::<Float64Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
    )
}

fn float32_fixed_size_list_array(rows: Vec<Vec<f32>>) -> FixedSizeListArray {
    let width = rows.first().map_or(0, Vec::len);
    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
        i32::try_from(width).expect("fixed-size-list width should fit i32"),
    )
}

fn complex64_fixed_size_list_array(rows: Vec<Vec<Complex64>>) -> FixedSizeListArray {
    let row_count = rows.len();
    let width = rows.first().map_or(0, Vec::len);
    let values = rows.into_iter().flatten().collect::<Vec<_>>();
    let array = Array2::from_shape_vec((row_count, width), values).expect("complex batch shape");
    ndarrow::array2_complex64_to_fixed_size_list(array).expect("complex batch")
}

fn complex64_matrix_batch<const B: usize, const R: usize, const C: usize>(
    name: &str,
    values: [[[Complex64; C]; R]; B],
) -> (Field, FixedSizeListArray) {
    let values = values.into_iter().flatten().flatten().collect::<Vec<_>>();
    let array = Array3::from_shape_vec((B, R, C), values).expect("complex matrix batch shape");
    let (field, array) = ndarrow::arrayd_complex64_to_fixed_shape_tensor(name, array.into_dyn())
        .expect("complex matrix batch");
    (field, array)
}

fn complex64_tensor_batch4<const B: usize, const D0: usize, const D1: usize, const D2: usize>(
    name: &str,
    values: [[[[Complex64; D2]; D1]; D0]; B],
) -> (Field, FixedSizeListArray) {
    let values = values.into_iter().flatten().flatten().flatten().collect::<Vec<_>>();
    let array =
        Array4::from_shape_vec((B, D0, D1, D2), values).expect("complex tensor batch shape");
    let (field, array) = ndarrow::arrayd_complex64_to_fixed_shape_tensor(name, array.into_dyn())
        .expect("complex tensor batch");
    (field, array)
}

fn complex64_ragged_matrices(name: &str, rows: Vec<Vec<Vec<Complex64>>>) -> (Field, StructArray) {
    let rows = rows
        .into_iter()
        .map(|matrix| {
            let row_count = matrix.len();
            let col_count = matrix.first().map_or(0, Vec::len);
            let values = matrix.into_iter().flatten().collect::<Vec<_>>();
            Array2::from_shape_vec((row_count, col_count), values)
                .expect("complex ragged matrix batch shape")
                .into_dyn()
        })
        .collect::<Vec<_>>();
    let (field, array) = ndarrow::arrays_complex64_to_variable_shape_tensor(name, rows, None)
        .expect("complex ragged tensors");
    (field, array)
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

fn nested_float64_list_column(rows: Vec<Vec<Vec<f64>>>) -> Result<ListArray> {
    let arrays = rows
        .into_iter()
        .map(float64_list_array)
        .map(|array| Arc::new(array) as ArrayRef)
        .collect::<Vec<_>>();
    arrays_into_list_array(arrays)
}

fn float64_column(batch: &RecordBatch, index: usize) -> &Float64Array {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("expected Float64Array column")
}

fn float32_column(batch: &RecordBatch, index: usize) -> &Float32Array {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Float32Array>()
        .expect("expected Float32Array column")
}

fn int64_column(batch: &RecordBatch, index: usize) -> &Int64Array {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<Int64Array>()
        .expect("expected Int64Array column")
}

fn struct_column(batch: &RecordBatch, index: usize) -> &StructArray {
    batch
        .column(index)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("expected StructArray column")
}

fn assert_eigen_struct_column(batch: &RecordBatch, index: usize, min: f64, max: f64) {
    let field = batch.schema().field(index).clone();
    let DataType::Struct(fields) = field.data_type() else {
        panic!("expected eigen struct output");
    };
    let output = struct_column(batch, index);
    let eigenvalues =
        output.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("eigenvalues");
    let eigenvalues =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(eigenvalues).expect("eigenvalues");
    let eigenvectors =
        output.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("eigenvectors");
    let eigenvectors =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&fields[1], eigenvectors)
            .expect("eigenvectors")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 eigenvectors");
    assert_close(eigenvalues.row(0).iter().copied().fold(f64::INFINITY, f64::min), min);
    assert_close(eigenvalues.row(0).iter().copied().fold(f64::NEG_INFINITY, f64::max), max);
    assert_close(eigenvectors.iter().map(|value| value.abs()).sum::<f64>(), 2.0);
}

fn assert_two_tensor_struct_column(batch: &RecordBatch, index: usize, min: f64, max: f64) {
    let field = batch.schema().field(index).clone();
    let DataType::Struct(fields) = field.data_type() else {
        panic!("expected two-tensor struct output");
    };
    let output = struct_column(batch, index);
    let first =
        output.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("first tensor");
    let first = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&fields[0], first)
        .expect("first tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 first tensor");
    let second =
        output.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("second tensor");
    let second = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&fields[1], second)
        .expect("second tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 second tensor");
    let diagonal = [second[[0, 0, 0]], second[[0, 1, 1]]];
    assert_close(first.iter().map(|value| value.abs()).sum::<f64>(), 2.0);
    assert_close(second[[0, 0, 1]], 0.0);
    assert_close(second[[0, 1, 0]], 0.0);
    assert_close(diagonal.into_iter().fold(f64::INFINITY, f64::min), min);
    assert_close(diagonal.into_iter().fold(f64::NEG_INFINITY, f64::max), max);
}

fn assert_orthogonal_matrix_column(batch: &RecordBatch, index: usize) {
    let schema = batch.schema();
    let field = schema.field(index);
    let output = batch
        .column(index)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("orthogonal output");
    let output = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(field, output)
        .expect("orthogonal output")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 orthogonal output");
    assert_close(output[[0, 0, 0]], 1.0);
    assert_close(output[[0, 1, 1]], 1.0);
}

fn assert_tensor_vector_struct_column(batch: &RecordBatch, index: usize, diagonal: [f64; 2]) {
    let field = batch.schema().field(index).clone();
    let DataType::Struct(fields) = field.data_type() else {
        panic!("expected tensor-vector struct output");
    };
    let output = struct_column(batch, index);
    let balanced =
        output.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("balanced tensor");
    let balanced = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&fields[0], balanced)
        .expect("balanced tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 balanced tensor");
    let scaling =
        output.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("diagonal vector");
    let scaling =
        ndarrow::fixed_size_list_as_array2::<Float64Type>(scaling).expect("diagonal vector");
    assert_close(balanced[[0, 0, 0]], diagonal[0]);
    assert_close(balanced[[0, 1, 1]], diagonal[1]);
    assert_close(scaling[[0, 0]], 1.0);
    assert_close(scaling[[0, 1]], 1.0);
}

fn direct_complex_matrix_batch() -> Result<RecordBatch> {
    let (stats_field, stats_matrix) = complex64_matrix_batch("stats_matrix", [
        [[Complex64::new(1.0, 1.0), Complex64::new(0.0, 0.0)], [
            Complex64::new(0.0, 0.0),
            Complex64::new(2.0, 0.0),
        ]],
        [[Complex64::new(2.0, 0.0), Complex64::new(1.0, -1.0)], [
            Complex64::new(1.0, 1.0),
            Complex64::new(3.0, 0.0),
        ]],
    ]);
    let (system_field, system_matrix) = complex64_matrix_batch("system_matrix", [
        [[Complex64::new(4.0, 0.0), Complex64::new(0.0, 0.0)], [
            Complex64::new(0.0, 0.0),
            Complex64::new(9.0, 0.0),
        ]],
        [[Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)], [
            Complex64::new(0.0, 0.0),
            Complex64::new(5.0, 0.0),
        ]],
    ]);
    let rhs = complex64_fixed_size_list_array(vec![
        vec![Complex64::new(8.0, 0.0), Complex64::new(18.0, 0.0)],
        vec![Complex64::new(4.0, 0.0), Complex64::new(10.0, 0.0)],
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        stats_field,
        system_field,
        Field::new("rhs_vector", rhs.data_type().clone(), false),
    ]));
    Ok(RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
        Arc::new(stats_matrix) as ArrayRef,
        Arc::new(system_matrix) as ArrayRef,
        Arc::new(rhs) as ArrayRef,
    ])?)
}

fn assert_direct_complex_matrix_results(batch: &RecordBatch) {
    assert_eq!(batch.num_rows(), 2);
    assert_eq!(int64_column(batch, 0).values().as_ref(), &[1, 2]);

    let matvec = batch
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("complex matvec output");
    let matvec = ndarrow::complex64_as_array_view2(matvec).expect("complex matvec output view");
    assert_close(matvec[[0, 0]].re, 32.0);
    assert_close(matvec[[0, 1]].re, 162.0);
    assert_close(matvec[[1, 0]].re, 8.0);
    assert_close(matvec[[1, 1]].re, 50.0);

    let means = batch
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("complex means output");
    let means = ndarrow::complex64_as_array_view2(means).expect("complex means output view");
    assert_close(means[[0, 0]].re, 0.5);
    assert_close(means[[0, 0]].im, 0.5);
    assert_close(means[[0, 1]].re, 1.0);
    assert_close(means[[0, 1]].im, 0.0);
    assert_close(means[[1, 0]].re, 1.5);
    assert_close(means[[1, 0]].im, 0.5);
    assert_close(means[[1, 1]].re, 2.0);
    assert_close(means[[1, 1]].im, -0.5);

    let cg =
        batch.column(3).as_any().downcast_ref::<FixedSizeListArray>().expect("complex cg output");
    let cg = ndarrow::complex64_as_array_view2(cg).expect("complex cg output view");
    assert_close(cg[[0, 0]].re, 2.0);
    assert_close(cg[[0, 1]].re, 2.0);
    assert_close(cg[[1, 0]].re, 2.0);
    assert_close(cg[[1, 1]].re, 2.0);

    let gmres = batch
        .column(4)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("complex gmres output");
    let gmres = ndarrow::complex64_as_array_view2(gmres).expect("complex gmres output view");
    assert_close(gmres[[0, 0]].re, 2.0);
    assert_close(gmres[[0, 1]].re, 2.0);
    assert_close(gmres[[1, 0]].re, 2.0);
    assert_close(gmres[[1, 1]].re, 2.0);
}

fn direct_complex_tensor_batch() -> Result<RecordBatch> {
    let (tensor_field, tensor) = complex64_tensor_batch4("tensor_batch", [[
        [[Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)], [
            Complex64::new(0.0, 0.0),
            Complex64::new(5.0, 12.0),
        ]],
        [[Complex64::new(8.0, 15.0), Complex64::new(0.0, 0.0)], [
            Complex64::new(7.0, 24.0),
            Complex64::new(0.0, 0.0),
        ]],
    ]]);
    let (ragged_field, ragged) = complex64_ragged_matrices("ragged_tensor", vec![vec![
        vec![Complex64::new(3.0, 4.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(5.0, 12.0), Complex64::new(0.0, 0.0)],
    ]]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        tensor_field,
        ragged_field,
    ]));
    Ok(RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef,
        Arc::new(tensor) as ArrayRef,
        Arc::new(ragged) as ArrayRef,
    ])?)
}

fn assert_direct_complex_tensor_results(batch: &RecordBatch) {
    assert_eq!(batch.num_rows(), 1);
    assert_eq!(int64_column(batch, 0).value(0), 1);

    let norm_field = batch.schema().field(1).clone();
    let norm = batch
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("complex tensor norm output");
    let norm = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&norm_field, norm)
        .expect("complex tensor norm view")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 norm output");
    assert_close(norm[[0, 0, 0]], 5.0);
    assert_close(norm[[0, 0, 1]], 13.0);
    assert_close(norm[[0, 1, 0]], 17.0);
    assert_close(norm[[0, 1, 1]], 25.0);

    let normalized_field = batch.schema().field(2).clone();
    let normalized = batch
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("complex tensor normalize output");
    let normalized =
        ndarrow::complex64_fixed_shape_tensor_as_array_viewd(&normalized_field, normalized)
            .expect("complex tensor normalize view")
            .into_dimensionality::<Ix4>()
            .expect("rank-4 complex tensor output");
    assert_close(normalized[[0, 0, 0, 0]].re, 0.6);
    assert_close(normalized[[0, 0, 0, 0]].im, 0.8);
    assert_close(normalized[[0, 0, 1, 0]].re, 0.0);
    assert_close(normalized[[0, 0, 1, 0]].im, 0.0);
    assert_close(normalized[[0, 1, 0, 0]].re, 8.0 / 17.0);
    assert_close(normalized[[0, 1, 0, 0]].im, 15.0 / 17.0);

    let ragged_norm_field = batch.schema().field(3).clone();
    let ragged_norm = batch
        .column(3)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("complex ragged norm output");
    let mut ragged_norm =
        ndarrow::variable_shape_tensor_iter::<Float64Type>(&ragged_norm_field, ragged_norm)
            .expect("complex ragged norm iterator");
    let (_, first_norm) = ragged_norm.next().expect("first ragged norm row").expect("first norm");
    assert_close(first_norm[[0]], 5.0);
    assert_close(first_norm[[1]], 13.0);
    assert!(ragged_norm.next().is_none());

    let ragged_normalized_field = batch.schema().field(4).clone();
    let ragged_normalized = batch
        .column(4)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("complex ragged normalize output");
    let mut ragged_normalized =
        ndarrow::complex64_variable_shape_tensor_iter(&ragged_normalized_field, ragged_normalized)
            .expect("complex ragged normalize iterator");
    let (_, first) = ragged_normalized
        .next()
        .expect("first ragged normalized row")
        .expect("first ragged normalized tensor");
    assert_close(first[[0, 0]].re, 0.6);
    assert_close(first[[0, 0]].im, 0.8);
    assert_close(first[[1, 0]].re, 5.0 / 13.0);
    assert_close(first[[1, 0]].im, 12.0 / 13.0);
    assert!(ragged_normalized.next().is_none());
}

#[test]
fn register_all_accepts_the_current_catalog() {
    let mut registry = MemoryFunctionRegistry::new();

    ndatafusion::register_all(&mut registry)
        .expect("the current scaffold should register successfully");

    assert!(registry.udfs().len() >= ndatafusion::udfs::all_default_functions().len());
    assert!(registry.udfs().contains("vector_l2_norm"));
    assert!(registry.udfs().contains("matrix_qr_solve_least_squares"));
}

#[tokio::test]
async fn sql_literal_constructor_pipeline_executes() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batches = ctx
        .sql(
            "SELECT
                vector_dot(make_vector(left_values, 2), make_vector(right_values, 2)) AS dot,
                vector_l2_norm(vector_normalize(make_vector(left_values, 2))) AS unit_norm,
                matrix_determinant(make_matrix(matrix_values, 2, 2)) AS det
             FROM (
                SELECT
                    [3.0, 4.0] AS left_values,
                    [4.0, 0.0] AS right_values,
                    [9.0, 0.0, 0.0, 4.0] AS matrix_values
             )",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_close(float64_column(&batches[0], 0).value(0), 12.0);
    assert_close(float64_column(&batches[0], 1).value(0), 1.0);
    assert_close(float64_column(&batches[0], 2).value(0), 36.0);
    Ok(())
}

#[tokio::test]
async fn sql_named_argument_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batches = ctx
        .sql(
            "SELECT
                vector_l2_norm(make_vector(values => left_values, dimension => 2)) AS norm,
                matrix_determinant(make_matrix(values => matrix_values, rows => 2, cols => 2)) AS \
             det,
                matrix_power(
                    make_matrix(values => matrix_values, rows => 2, cols => 2),
                    power => 2.0
                ) AS squared,
                matrix_exp(
                    make_matrix(values => identity_values, rows => 2, cols => 2),
                    tolerance => 1e-6,
                    max_terms => 12
                ) AS expm
             FROM (
                SELECT
                    [3.0, 4.0] AS left_values,
                    [2.0, 0.0, 0.0, 3.0] AS matrix_values,
                    [1.0, 0.0, 0.0, 1.0] AS identity_values
             )",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_close(float64_column(&batches[0], 0).value(0), 5.0);
    assert_close(float64_column(&batches[0], 1).value(0), 6.0);
    assert_eq!(batches[0].column(2).len(), 1);
    assert_eq!(batches[0].column(3).len(), 1);
    Ok(())
}

#[tokio::test]
async fn sql_alias_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let embedding = float32_fixed_size_list_array(vec![vec![3.0_f32, 4.0]]);
    let rhs = float32_fixed_size_list_array(vec![vec![5.0_f32, 6.0]]);
    let (matrix_field, matrix_storage) = ndarrow::arrayd_to_fixed_shape_tensor(
        "matrix",
        Array3::from_shape_vec((1, 2, 2), vec![1.0_f32, 0.0, 0.0, 1.0])
            .expect("matrix batch")
            .into_dyn(),
    )
    .expect("matrix tensor batch");
    let (tensor_field, tensor_storage) = ndarrow::arrayd_to_fixed_shape_tensor(
        "tensor_fixed",
        Array3::from_shape_vec((1, 2, 2), vec![3.0_f32, 4.0, 5.0, 12.0])
            .expect("fixed tensor batch")
            .into_dyn(),
    )
    .expect("fixed tensor batch");
    let (variable_field, variable_storage) = ndarrow::arrays_to_variable_shape_tensor(
        "tensor_var",
        vec![
            Array2::from_shape_vec((2, 2), vec![1.0_f32, 2.0, 3.0, 4.0])
                .expect("variable tensor row")
                .into_dyn(),
        ],
        Some(vec![Some(2), Some(2)]),
    )
    .expect("variable tensor batch");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("embedding", embedding.data_type().clone(), false),
        matrix_field.as_ref().clone(),
        Field::new("rhs", rhs.data_type().clone(), false),
        tensor_field.as_ref().clone(),
        variable_field.as_ref().clone(),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef,
        Arc::new(embedding) as ArrayRef,
        Arc::new(matrix_storage) as ArrayRef,
        Arc::new(rhs) as ArrayRef,
        Arc::new(tensor_storage) as ArrayRef,
        Arc::new(variable_storage) as ArrayRef,
    ])?;
    drop(ctx.register_batch("alias_inputs", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                vector_norm(embedding) AS vector_norm,
                matrix_qr_solve_ls(matrix, rhs) AS qr_ls,
                tensor_norm_last(tensor_fixed) AS tensor_norms,
                tensor_var_sum_last(tensor_var) AS tensor_var_sum
             FROM alias_inputs",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_close32(float32_column(&batches[0], 0).value(0), 5.0);

    let qr_solution =
        batches[0].column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("qr solution");
    let qr_solution =
        ndarrow::fixed_size_list_as_array2::<Float32Type>(qr_solution).expect("qr solution");
    assert_close32(qr_solution[[0, 0]], 5.0);
    assert_close32(qr_solution[[0, 1]], 6.0);

    let tensor_norm_field = batches[0].schema().field(2).clone();
    let tensor_norms =
        batches[0].column(2).as_any().downcast_ref::<FixedSizeListArray>().expect("tensor norms");
    let tensor_norms =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float32Type>(&tensor_norm_field, tensor_norms)
            .expect("tensor norms")
            .into_dimensionality::<Ix2>()
            .expect("rank-2 tensor norms");
    assert_close32(tensor_norms[[0, 0]], 5.0);
    assert_close32(tensor_norms[[0, 1]], 13.0);

    let tensor_var_field = batches[0].schema().field(3).clone();
    let mut tensor_rows = ndarrow::variable_shape_tensor_iter::<Float32Type>(
        &tensor_var_field,
        struct_column(&batches[0], 3),
    )
    .expect("variable-shape tensor output");
    let (_, tensor_var_sum) = tensor_rows.next().expect("row0").expect("row0 tensor");
    let tensor_var_sum = tensor_var_sum.into_dimensionality::<Ix1>().expect("rank-1 tensor");
    assert_close32(tensor_var_sum[[0]], 3.0);
    assert_close32(tensor_var_sum[[1]], 7.0);
    Ok(())
}

#[tokio::test]
async fn sql_list_columns_drive_vector_and_matrix_queries() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        (
            "vector_values",
            Arc::new(float64_list_array(vec![vec![3.0, 4.0], vec![6.0, 8.0]])) as ArrayRef,
        ),
        (
            "matrix_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0]], vec![
                vec![2.0, 0.0],
                vec![0.0, 2.0],
            ]])?) as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("inputs", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                vector_l2_norm(make_vector(vector_values, 2)) AS norm,
                matrix_determinant(make_matrix(matrix_values, 2, 2)) AS det
             FROM inputs
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);
    assert_close(float64_column(&batches[0], 1).value(0), 5.0);
    assert_close(float64_column(&batches[0], 1).value(1), 10.0);
    assert_close(float64_column(&batches[0], 2).value(0), -2.0);
    assert_close(float64_column(&batches[0], 2).value(1), 4.0);
    Ok(())
}

#[tokio::test]
async fn sql_direct_fixed_size_list_vector_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        (
            "left_vector",
            Arc::new(float32_fixed_size_list_array(vec![vec![3.0, 4.0], vec![6.0, 8.0]]))
                as ArrayRef,
        ),
        (
            "right_vector",
            Arc::new(float32_fixed_size_list_array(vec![vec![4.0, 0.0], vec![0.0, 6.0]]))
                as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("direct_vectors", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                vector_l2_norm(left_vector) AS norm,
                vector_dot(left_vector, right_vector) AS dot,
                vector_cosine_similarity(left_vector, right_vector) AS similarity
             FROM direct_vectors
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);
    assert_close32(float32_column(&batches[0], 1).value(0), 5.0);
    assert_close32(float32_column(&batches[0], 1).value(1), 10.0);
    assert_close32(float32_column(&batches[0], 2).value(0), 12.0);
    assert_close32(float32_column(&batches[0], 2).value(1), 48.0);
    assert_close32(float32_column(&batches[0], 3).value(0), 0.6);
    assert_close32(float32_column(&batches[0], 3).value(1), 0.8);
    Ok(())
}

#[tokio::test]
async fn sql_direct_complex_vector_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let left = complex64_fixed_size_list_array(vec![
        vec![Complex64::new(1.0, 1.0), Complex64::new(0.0, 0.0)],
        vec![Complex64::new(0.0, 0.0), Complex64::new(2.0, 0.0)],
    ]);
    let right = complex64_fixed_size_list_array(vec![
        vec![Complex64::new(1.0, 0.0), Complex64::new(0.0, 1.0)],
        vec![Complex64::new(2.0, 0.0), Complex64::new(0.0, 0.0)],
    ]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        Field::new("left_vector", left.data_type().clone(), false),
        Field::new("right_vector", right.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
        Arc::new(left) as ArrayRef,
        Arc::new(right) as ArrayRef,
    ])?;
    drop(ctx.register_batch("direct_complex_vectors", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                vector_dot_hermitian(left_vector, right_vector) AS dot,
                vector_l2_norm_complex(left_vector) AS norm,
                vector_cosine_similarity_complex(left_vector, right_vector) AS similarity,
                vector_normalize_complex(left_vector) AS normalized
             FROM direct_complex_vectors
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let dot_field = batches[0].schema().field(1).clone();
    let dots = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("complex scalar column");
    let dots = ndarrow::complex64_as_array_view1(dot_field.as_ref(), dots).expect("complex dots");
    assert_close(dots[0].re, 1.0);
    assert_close(dots[0].im, -1.0);
    assert_close(dots[1].re, 0.0);
    assert_close(dots[1].im, 0.0);

    assert_close(float64_column(&batches[0], 2).value(0), 2.0_f64.sqrt());
    assert_close(float64_column(&batches[0], 2).value(1), 2.0);

    let similarity_field = batches[0].schema().field(3).clone();
    let similarities = batches[0]
        .column(3)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("complex similarity column");
    let similarities = ndarrow::complex64_as_array_view1(similarity_field.as_ref(), similarities)
        .expect("complex similarities");
    assert_close(similarities[0].re, 0.5);
    assert_close(similarities[0].im, -0.5);
    assert_close(similarities[1].re, 0.0);
    assert_close(similarities[1].im, 0.0);

    let normalized = batches[0]
        .column(4)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("normalized complex vectors");
    let normalized =
        ndarrow::complex64_as_array_view2(normalized).expect("normalized complex vector view");
    assert_close(normalized[[0, 0]].re, 1.0 / 2.0_f64.sqrt());
    assert_close(normalized[[0, 0]].im, 1.0 / 2.0_f64.sqrt());
    assert_close(normalized[[0, 1]].re, 0.0);
    assert_close(normalized[[0, 1]].im, 0.0);
    assert_close(normalized[[1, 0]].re, 0.0);
    assert_close(normalized[[1, 0]].im, 0.0);
    assert_close(normalized[[1, 1]].re, 1.0);
    assert_close(normalized[[1, 1]].im, 0.0);
    Ok(())
}

#[tokio::test]
async fn sql_direct_complex_matrix_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;
    drop(ctx.register_batch("direct_complex_matrices", direct_complex_matrix_batch()?)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                matrix_matvec_complex(system_matrix, rhs_vector) AS matvec,
                matrix_column_means_complex(stats_matrix) AS means,
                matrix_conjugate_gradient_complex(
                    system_matrix,
                    rhs_vector,
                    tolerance => 1e-8,
                    max_iterations => 32
                ) AS cg,
                matrix_gmres_complex(
                    system_matrix,
                    rhs_vector,
                    tolerance => 1e-8,
                    max_iterations => 32
                ) AS gmres
             FROM direct_complex_matrices
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_direct_complex_matrix_results(&batches[0]);
    Ok(())
}

#[tokio::test]
async fn sql_direct_complex_tensor_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;
    drop(ctx.register_batch("direct_complex_tensors", direct_complex_tensor_batch()?)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                tensor_l2_norm_last_axis_complex(tensor_batch) AS norm,
                tensor_normalize_last_axis_complex(tensor_batch) AS normalized,
                tensor_variable_l2_norm_last_axis_complex(ragged_tensor) AS ragged_norm,
                tensor_variable_normalize_last_axis_complex(ragged_tensor) AS ragged_normalized
             FROM direct_complex_tensors",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_direct_complex_tensor_results(&batches[0]);
    Ok(())
}

#[tokio::test]
async fn sql_vector_aggregate_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let observation = FixedSizeListArray::from_iter_primitive::<Float64Type, _, _>(
        vec![
            Some(vec![Some(1.0), Some(2.0)]),
            Some(vec![Some(3.0), Some(4.0)]),
            Some(vec![Some(5.0), Some(6.0)]),
        ],
        2,
    );
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("observation", observation.data_type().clone(), false),
        ])),
        vec![
            Arc::new(Int64Array::from(vec![1_i64, 2, 3])) as ArrayRef,
            Arc::new(observation) as ArrayRef,
        ],
    )?;
    drop(ctx.register_batch("agg_vectors", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                vector_covariance_agg(observation) AS covariance,
                vector_correlation_agg(observation) AS correlation,
                vector_pca_fit(observation) AS pca
             FROM agg_vectors",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let covariance_field = batches[0].schema().field(0).clone();
    let covariance = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("covariance output");
    let covariance =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&covariance_field, covariance)
            .expect("covariance tensor")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 covariance tensor");
    assert_close(covariance[[0, 0, 0]], 4.0);
    assert_close(covariance[[0, 0, 1]], 4.0);
    assert_close(covariance[[0, 1, 0]], 4.0);
    assert_close(covariance[[0, 1, 1]], 4.0);

    let correlation_field = batches[0].schema().field(1).clone();
    let correlation = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("correlation output");
    let correlation =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&correlation_field, correlation)
            .expect("correlation tensor")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 correlation tensor");
    assert_close(correlation[[0, 0, 0]], 1.0);
    assert_close(correlation[[0, 0, 1]], 1.0);
    assert_close(correlation[[0, 1, 0]], 1.0);
    assert_close(correlation[[0, 1, 1]], 1.0);

    let pca_field = batches[0].schema().field(2).clone();
    let DataType::Struct(fields) = pca_field.data_type() else {
        panic!("expected PCA struct output");
    };
    let pca = struct_column(&batches[0], 2);
    let components =
        pca.column(0).as_any().downcast_ref::<StructArray>().expect("components variable tensor");
    let mut components = ndarrow::variable_shape_tensor_iter::<Float64Type>(&fields[0], components)
        .expect("components iterator");
    let components = components.next().expect("first component batch").expect("component tensor");
    let components = components.1.into_dimensionality::<Ix2>().expect("component matrix");
    assert_eq!(components.shape(), &[2, 2]);
    let mean = pca.column(3).as_any().downcast_ref::<FixedSizeListArray>().expect("mean vector");
    let mean = ndarrow::fixed_size_list_as_array2::<Float64Type>(mean).expect("mean vector view");
    assert_close(mean[[0, 0]], 3.0);
    assert_close(mean[[0, 1]], 4.0);
    Ok(())
}

#[tokio::test]
async fn sql_linear_regression_fit_aggregate_query_executes() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let design = float32_fixed_size_list_array(vec![vec![1.0], vec![2.0], vec![3.0]]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("design", design.data_type().clone(), false),
        Field::new("response", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(design) as ArrayRef,
        Arc::new(Int64Array::from(vec![2_i64, 4, 6])) as ArrayRef,
    ])?;
    drop(ctx.register_batch("agg_regression", batch)?);

    let batches = ctx
        .sql(
            "SELECT linear_regression_fit(design, response, add_intercept => false) AS fit
             FROM agg_regression",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    let fit_field = batches[0].schema().field(0).clone();
    let DataType::Struct(fields) = fit_field.data_type() else {
        panic!("expected regression struct output");
    };
    let fit = struct_column(&batches[0], 0);
    let coefficients = fit
        .column(0)
        .as_any()
        .downcast_ref::<StructArray>()
        .expect("coefficients variable tensor");
    let mut coefficients =
        ndarrow::variable_shape_tensor_iter::<Float32Type>(&fields[0], coefficients)
            .expect("coefficients iterator");
    let coefficients = coefficients
        .next()
        .expect("first coefficients batch")
        .expect("coefficient tensor")
        .1
        .into_dimensionality::<Ix1>()
        .expect("coefficient vector");
    assert_eq!(coefficients.len(), 1);
    assert_close32(coefficients[0], 2.0);
    let r_squared =
        fit.column(1).as_any().downcast_ref::<Float32Array>().expect("r_squared scalar");
    assert_close32(r_squared.value(0), 1.0);
    Ok(())
}

#[tokio::test]
async fn sql_direct_fixed_shape_tensor_matrix_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let matrices =
        Array3::from_shape_vec((2, 2, 2), vec![1.0_f32, 2.0, 3.0, 4.0, 2.0, 0.0, 0.0, 2.0])
            .expect("matrix batch");
    let (matrix_field, matrix_storage) =
        ndarrow::arrayd_to_fixed_shape_tensor("matrix", matrices.into_dyn())
            .expect("fixed-shape matrix batch");
    let rhs = float32_fixed_size_list_array(vec![vec![4.0, 3.0], vec![9.0, 8.0]]);
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        matrix_field,
        Field::new("rhs", rhs.data_type().clone(), false),
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
        Arc::new(matrix_storage) as ArrayRef,
        Arc::new(rhs) as ArrayRef,
    ])?;
    drop(ctx.register_batch("direct_matrices", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                matrix_determinant(matrix) AS det,
                matrix_matvec(matrix, rhs) AS matvec
             FROM direct_matrices
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);
    assert_close32(float32_column(&batches[0], 1).value(0), -2.0);
    assert_close32(float32_column(&batches[0], 1).value(1), 4.0);
    let matvec = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("matrix matvec output");
    let matvec = ndarrow::fixed_size_list_as_array2::<Float32Type>(matvec).expect("matrix matvec");
    assert_close32(matvec[[0, 0]], 10.0);
    assert_close32(matvec[[0, 1]], 24.0);
    assert_close32(matvec[[1, 0]], 18.0);
    assert_close32(matvec[[1, 1]], 16.0);
    Ok(())
}

#[tokio::test]
async fn sql_direct_fixed_shape_tensor_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let tensors = Array3::from_shape_vec((1, 2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
        .expect("tensor batch");
    let (tensor_field, tensor_storage) =
        ndarrow::arrayd_to_fixed_shape_tensor("tensor", tensors.into_dyn())
            .expect("fixed-shape tensor batch");
    let schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), tensor_field]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef,
        Arc::new(tensor_storage) as ArrayRef,
    ])?;
    drop(ctx.register_batch("direct_tensors", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                tensor_sum_last_axis(tensor) AS reduced,
                tensor_permute_axes(tensor, 1, 0) AS permuted
             FROM direct_tensors",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(int64_column(&batches[0], 0).value(0), 1);

    let schema = batches[0].schema();
    let reduced_storage = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("reduced tensor output");
    let reduced =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float32Type>(schema.field(1), reduced_storage)
            .expect("reduced tensor")
            .into_dimensionality::<Ix2>()
            .expect("rank-2 reduced tensor");
    assert_close32(reduced[[0, 0]], 6.0);
    assert_close32(reduced[[0, 1]], 15.0);

    let permuted_storage = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("permuted tensor output");
    let permuted = ndarrow::fixed_shape_tensor_as_array_viewd::<Float32Type>(
        schema.field(2),
        permuted_storage,
    )
    .expect("permuted tensor")
    .into_dimensionality::<Ix3>()
    .expect("rank-3 permuted tensor");
    assert_close32(permuted[[0, 0, 0]], 1.0);
    assert_close32(permuted[[0, 0, 1]], 4.0);
    assert_close32(permuted[[0, 2, 0]], 3.0);
    assert_close32(permuted[[0, 2, 1]], 6.0);
    Ok(())
}

#[tokio::test]
async fn sql_matrix_helper_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        (
            "matrix_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![2.0, 0.0], vec![0.0, 1.0]], vec![
                vec![3.0, 0.0],
                vec![0.0, 4.0],
            ]])?) as ArrayRef,
        ),
        (
            "vector_values",
            Arc::new(float64_list_array(vec![vec![4.0, 3.0], vec![9.0, 8.0]])) as ArrayRef,
        ),
        (
            "rank_matrix_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]], vec![
                vec![1.0, 1.0],
                vec![1.0, 1.0],
            ]])?) as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("matrix_helpers", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                matrix_matvec(make_matrix(matrix_values, 2, 2), make_vector(vector_values, 2)) AS \
             product,
                matrix_qr_condition_number(make_matrix(matrix_values, 2, 2)) AS qr_cond,
                matrix_qr_reconstruct(make_matrix(matrix_values, 2, 2)) AS qr_reconstructed,
                matrix_svd_rank(make_matrix(rank_matrix_values, 2, 2)) AS svd_rank,
                matrix_svd_reconstruct(make_matrix(matrix_values, 2, 2)) AS svd_reconstructed
             FROM matrix_helpers
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let product = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("vector batch output");
    let product = ndarrow::fixed_size_list_as_array2::<Float64Type>(product).expect("product");
    assert_close(product[[0, 0]], 8.0);
    assert_close(product[[0, 1]], 3.0);
    assert_close(product[[1, 0]], 27.0);
    assert_close(product[[1, 1]], 32.0);
    assert_close(float64_column(&batches[0], 2).value(0), 2.0);
    assert_close(float64_column(&batches[0], 2).value(1), 4.0 / 3.0);

    let schema = batches[0].schema();

    let qr_field = schema.field(3);
    let qr_reconstructed = batches[0]
        .column(3)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("qr reconstructed tensor output");
    let qr_reconstructed =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(qr_field, qr_reconstructed)
            .expect("qr reconstructed tensor")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 qr reconstructed tensor");
    assert_close(qr_reconstructed[[0, 0, 0]], 2.0);
    assert_close(qr_reconstructed[[0, 1, 1]], 1.0);
    assert_close(qr_reconstructed[[1, 0, 0]], 3.0);
    assert_close(qr_reconstructed[[1, 1, 1]], 4.0);

    assert_eq!(int64_column(&batches[0], 4).values().as_ref(), &[2, 1]);

    let svd_field = schema.field(5);
    let svd_reconstructed = batches[0]
        .column(5)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("svd reconstructed tensor output");
    let svd_reconstructed =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(svd_field, svd_reconstructed)
            .expect("svd reconstructed tensor")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 svd reconstructed tensor");
    assert_close(svd_reconstructed[[0, 0, 0]], 2.0);
    assert_close(svd_reconstructed[[0, 1, 1]], 1.0);
    assert_close(svd_reconstructed[[1, 0, 0]], 3.0);
    assert_close(svd_reconstructed[[1, 1, 1]], 4.0);
    Ok(())
}

#[tokio::test]
async fn sql_iterative_solver_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        (
            "matrix_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![4.0, 1.0], vec![1.0, 3.0]], vec![
                vec![10.0, 2.0],
                vec![2.0, 5.0],
            ]])?) as ArrayRef,
        ),
        (
            "rhs_values",
            Arc::new(float64_list_array(vec![vec![1.0, 2.0], vec![12.0, 7.0]])) as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("iterative_helpers", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                matrix_conjugate_gradient(
                    make_matrix(matrix_values, 2, 2),
                    make_vector(rhs_values, 2),
                    1e-12,
                    64
                ) AS cg,
                matrix_gmres(
                    make_matrix(matrix_values, 2, 2),
                    make_vector(rhs_values, 2),
                    1e-12,
                    64
                ) AS gmres
             FROM iterative_helpers
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    for index in [1_usize, 2] {
        let output = batches[0]
            .column(index)
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .expect("iterative output");
        let output =
            ndarrow::fixed_size_list_as_array2::<Float64Type>(output).expect("iterative output");
        assert_close(output[[0, 0]], 1.0 / 11.0);
        assert_close(output[[0, 1]], 7.0 / 11.0);
        assert_close(output[[1, 0]], 1.0);
        assert_close(output[[1, 1]], 1.0);
    }
    Ok(())
}

#[tokio::test]
async fn sql_decomposition_variant_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef),
        (
            "reduced_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![1.0, 0.0], vec![0.0, 2.0], vec![
                0.0, 0.0,
            ]]])?) as ArrayRef,
        ),
        (
            "pivoted_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![1.0, 0.0], vec![0.0, 3.0]]])?)
                as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("decomposition_variants", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                matrix_qr_reduced(make_matrix(reduced_values, 3, 2)) AS qr_reduced,
                matrix_qr_pivoted(make_matrix(pivoted_values, 2, 2)) AS qr_pivoted
             FROM decomposition_variants",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let qr_reduced_field = batches[0].schema().field(0).clone();
    let DataType::Struct(qr_reduced_fields) = qr_reduced_field.data_type() else {
        panic!("expected qr_reduced struct output");
    };
    let qr_reduced = struct_column(&batches[0], 0);
    let q = qr_reduced.column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("q tensor");
    let q = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(&qr_reduced_fields[0], q)
        .expect("q tensor")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 tensor");
    let rank = qr_reduced.column(2).as_any().downcast_ref::<Int64Array>().expect("rank");
    assert_eq!(q.dim(), (1, 3, 2));
    assert_eq!(rank.value(0), 2);

    let qr_pivoted_field = batches[0].schema().field(1).clone();
    let DataType::Struct(qr_pivoted_fields) = qr_pivoted_field.data_type() else {
        panic!("expected qr_pivoted struct output");
    };
    let qr_pivoted = struct_column(&batches[0], 1);
    let permutation =
        qr_pivoted.column(2).as_any().downcast_ref::<FixedSizeListArray>().expect("p tensor");
    let permutation = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(
        &qr_pivoted_fields[2],
        permutation,
    )
    .expect("p tensor")
    .into_dimensionality::<Ix3>()
    .expect("rank-3 tensor");
    assert_close(permutation[[0, 0, 1]], 1.0);
    assert_close(permutation[[0, 1, 0]], 1.0);
    Ok(())
}

#[tokio::test]
async fn sql_svd_variant_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef),
        (
            "truncated_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![9.0, 0.0], vec![0.0, 4.0]]])?)
                as ArrayRef,
        ),
        (
            "tolerance_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![5.0, 0.0], vec![0.0, 1.0]]])?)
                as ArrayRef,
        ),
        (
            "null_space_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![1.0, 1.0], vec![1.0, 1.0]]])?)
                as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("svd_variants", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                matrix_svd_truncated(make_matrix(truncated_values, 2, 2), 1) AS svd_truncated,
                matrix_svd_with_tolerance(make_matrix(tolerance_values, 2, 2), 2.0) AS svd_tol,
                matrix_svd_null_space(make_matrix(null_space_values, 2, 2)) AS null_space
             FROM svd_variants",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let svd_truncated = struct_column(&batches[0], 0);
    let singular_values = svd_truncated
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("singular values");
    let singular_values = ndarrow::fixed_size_list_as_array2::<Float64Type>(singular_values)
        .expect("singular values");
    assert_close(singular_values[[0, 0]], 9.0);

    let svd_tol = struct_column(&batches[0], 1);
    let singular_values =
        svd_tol.column(1).as_any().downcast_ref::<FixedSizeListArray>().expect("singular values");
    let singular_values = ndarrow::fixed_size_list_as_array2::<Float64Type>(singular_values)
        .expect("singular values");
    assert_close(singular_values[[0, 0]], 5.0);
    assert_close(singular_values[[0, 1]], 0.0);

    let null_space_field = batches[0].schema().field(2).clone();
    let mut rows = ndarrow::variable_shape_tensor_iter::<Float64Type>(
        null_space_field.as_ref(),
        struct_column(&batches[0], 2),
    )
    .expect("null-space variable tensor");
    let (_, basis) = rows.next().expect("basis").expect("basis tensor");
    let basis = basis.into_dimensionality::<Ix2>().expect("rank-2 basis");
    assert_eq!(basis.dim(), (2, 1));
    Ok(())
}

#[tokio::test]
async fn sql_spectral_and_orthogonalization_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef),
        (
            "spectral_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![4.0, 0.0], vec![0.0, 9.0]]])?)
                as ArrayRef,
        ),
        (
            "identity_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![1.0, 0.0], vec![0.0, 1.0]]])?)
                as ArrayRef,
        ),
        (
            "basis_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![3.0, 0.0], vec![0.0, 4.0]]])?)
                as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("spectral_helpers", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                matrix_eigen_symmetric(make_matrix(spectral_values, 2, 2)) AS eigen,
                matrix_eigen_generalized(
                    make_matrix(spectral_values, 2, 2),
                    make_matrix(identity_values, 2, 2)
                ) AS generalized,
                matrix_balance_nonsymmetric(make_matrix(spectral_values, 2, 2)) AS balanced,
                matrix_schur(make_matrix(spectral_values, 2, 2)) AS schur,
                matrix_polar(make_matrix(spectral_values, 2, 2)) AS polar,
                matrix_gram_schmidt(make_matrix(basis_values, 2, 2)) AS gram,
                matrix_gram_schmidt_classic(make_matrix(basis_values, 2, 2)) AS gram_classic
             FROM spectral_helpers",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(int64_column(&batches[0], 0).value(0), 1);
    assert_eigen_struct_column(&batches[0], 1, 4.0, 9.0);
    assert_eigen_struct_column(&batches[0], 2, 4.0, 9.0);
    assert_tensor_vector_struct_column(&batches[0], 3, [4.0, 9.0]);
    assert_two_tensor_struct_column(&batches[0], 4, 4.0, 9.0);
    assert_two_tensor_struct_column(&batches[0], 5, 4.0, 9.0);
    assert_orthogonal_matrix_column(&batches[0], 6);
    assert_orthogonal_matrix_column(&batches[0], 7);

    Ok(())
}

#[tokio::test]
async fn sql_pca_application_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![(
        "matrix_values",
        Arc::new(nested_float64_list_column(vec![vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![
            5.0, 6.0,
        ]]])?) as ArrayRef,
    )])?;
    drop(ctx.register_batch("pca_helpers", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                pca.scores AS scores,
                matrix_pca_transform(matrix, pca) AS projected,
                matrix_pca_inverse_transform(pca.scores, pca) AS reconstructed
             FROM (
                SELECT
                    make_matrix(matrix_values, 3, 2) AS matrix,
                    matrix_pca(make_matrix(matrix_values, 3, 2)) AS pca
                FROM pca_helpers
             )",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);

    let schema = batches[0].schema();
    let scores_field = schema.field(0);
    let scores =
        batches[0].column(0).as_any().downcast_ref::<FixedSizeListArray>().expect("PCA scores");
    let scores = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(scores_field, scores)
        .expect("scores")
        .into_dimensionality::<Ix3>()
        .expect("rank-3 scores");

    let projected_field = schema.field(1);
    let projected = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("projected scores");
    let projected =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(projected_field, projected)
            .expect("projected scores")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 projected scores");
    assert_close(projected[[0, 0, 0]], scores[[0, 0, 0]]);
    assert_close(projected[[0, 2, 0]], scores[[0, 2, 0]]);
    assert_close(projected[[0, 1, 1]], scores[[0, 1, 1]]);

    let reconstructed_field = schema.field(2);
    let reconstructed = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("reconstructed matrix");
    let reconstructed = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(
        reconstructed_field,
        reconstructed,
    )
    .expect("reconstructed matrix")
    .into_dimensionality::<Ix3>()
    .expect("rank-3 reconstructed matrix");
    assert_close(reconstructed[[0, 0, 0]], 1.0);
    assert_close(reconstructed[[0, 0, 1]], 2.0);
    assert_close(reconstructed[[0, 2, 0]], 5.0);
    assert_close(reconstructed[[0, 2, 1]], 6.0);

    Ok(())
}

#[tokio::test]
async fn sql_triangular_and_matrix_function_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        (
            "lower_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![2.0, 0.0], vec![3.0, 1.0]], vec![
                vec![4.0, 0.0],
                vec![1.0, 2.0],
            ]])?) as ArrayRef,
        ),
        (
            "rhs_values",
            Arc::new(float64_list_array(vec![vec![4.0, 5.0], vec![8.0, 6.0]])) as ArrayRef,
        ),
        (
            "sign_values",
            Arc::new(nested_float64_list_column(vec![
                vec![vec![4.0, 0.0], vec![0.0, -9.0]],
                vec![vec![-2.0, 0.0], vec![0.0, 3.0]],
            ])?) as ArrayRef,
        ),
        (
            "exp_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![0.0, 0.0], vec![0.0, 1.0]], vec![
                vec![1.0, 0.0],
                vec![0.0, 2.0],
            ]])?) as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("triangular_helpers", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                matrix_solve_lower(make_matrix(lower_values, 2, 2), make_vector(rhs_values, 2)) AS \
             solved,
                matrix_sign(make_matrix(sign_values, 2, 2)) AS sign_matrix,
                matrix_exp_eigen(make_matrix(exp_values, 2, 2)) AS exp_matrix
             FROM triangular_helpers
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let solved = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("vector batch output");
    let solved = ndarrow::fixed_size_list_as_array2::<Float64Type>(solved).expect("solved");
    assert_close(solved[[0, 0]], 2.0);
    assert_close(solved[[0, 1]], -1.0);
    assert_close(solved[[1, 0]], 2.0);
    assert_close(solved[[1, 1]], 2.0);

    let schema = batches[0].schema();
    let sign_field = schema.field(2);
    let sign_matrix = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("sign tensor output");
    let sign_matrix =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(sign_field, sign_matrix)
            .expect("sign matrix")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 sign tensor");
    assert_close(sign_matrix[[0, 0, 0]], 1.0);
    assert_close(sign_matrix[[0, 1, 1]], -1.0);
    assert_close(sign_matrix[[1, 0, 0]], -1.0);
    assert_close(sign_matrix[[1, 1, 1]], 1.0);

    let exp_field = schema.field(3);
    let exp_matrix = batches[0]
        .column(3)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("exp tensor output");
    let exp_matrix =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(exp_field, exp_matrix)
            .expect("exp matrix")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 exp tensor");
    assert_close(exp_matrix[[0, 0, 0]], 1.0);
    assert_close(exp_matrix[[0, 1, 1]], 1.0_f64.exp());
    assert_close(exp_matrix[[1, 0, 0]], 1.0_f64.exp());
    assert_close(exp_matrix[[1, 1, 1]], 2.0_f64.exp());
    Ok(())
}

#[tokio::test]
async fn sql_parameterized_matrix_function_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        (
            "exp_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![0.0, 0.0], vec![0.0, 1.0]], vec![
                vec![1.0, 0.0],
                vec![0.0, 2.0],
            ]])?) as ArrayRef,
        ),
        (
            "power_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![4.0, 0.0], vec![0.0, 9.0]], vec![
                vec![16.0, 0.0],
                vec![0.0, 25.0],
            ]])?) as ArrayRef,
        ),
        (
            "log_values",
            Arc::new(nested_float64_list_column(vec![vec![vec![1.0, 0.0], vec![0.0, 1.1]], vec![
                vec![1.0, 0.0],
                vec![0.0, 1.2],
            ]])?) as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("parameterized_matrix_functions", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                matrix_exp(make_matrix(exp_values, 2, 2), 32, 1e-12) AS exp_matrix,
                matrix_power(make_matrix(power_values, 2, 2), 0.5) AS root_matrix,
                matrix_log_taylor(make_matrix(log_values, 2, 2), 64, 1e-12) AS log_matrix
             FROM parameterized_matrix_functions
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let schema = batches[0].schema();

    let exp_field = schema.field(1);
    let exp_matrix = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("exp tensor output");
    let exp_matrix =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(exp_field, exp_matrix)
            .expect("exp matrix")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 exp tensor");
    assert_close(exp_matrix[[0, 0, 0]], 1.0);
    assert_close(exp_matrix[[0, 1, 1]], 1.0_f64.exp());
    assert_close(exp_matrix[[1, 0, 0]], 1.0_f64.exp());
    assert_close(exp_matrix[[1, 1, 1]], 2.0_f64.exp());

    let root_field = schema.field(2);
    let root_matrix = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("root tensor output");
    let root_matrix =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(root_field, root_matrix)
            .expect("root matrix")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 root tensor");
    assert_close(root_matrix[[0, 0, 0]], 2.0);
    assert_close(root_matrix[[0, 1, 1]], 3.0);
    assert_close(root_matrix[[1, 0, 0]], 4.0);
    assert_close(root_matrix[[1, 1, 1]], 5.0);

    let log_field = schema.field(3);
    let log_matrix = batches[0]
        .column(3)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("log tensor output");
    let log_matrix =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(log_field, log_matrix)
            .expect("log matrix")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 log tensor");
    assert_close(log_matrix[[0, 0, 0]], 0.0);
    assert_close(log_matrix[[0, 1, 1]], 1.1_f64.ln());
    assert_close(log_matrix[[1, 0, 0]], 0.0);
    assert_close(log_matrix[[1, 1, 1]], 1.2_f64.ln());
    Ok(())
}

#[tokio::test]
async fn sql_sparse_and_variable_tensor_pipeline_executes() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        ("shape", Arc::new(int32_list_array(vec![vec![2, 3], vec![2, 2]])) as ArrayRef),
        ("row_ptrs", Arc::new(int32_list_array(vec![vec![0, 2, 3], vec![0, 1, 3]])) as ArrayRef),
        ("col_indices", Arc::new(u32_list_array(vec![vec![0, 2, 1], vec![0, 0, 1]])) as ArrayRef),
        (
            "values",
            Arc::new(float64_list_array(vec![vec![1.0, 2.0, 3.0], vec![4.0, 5.0, 6.0]]))
                as ArrayRef,
        ),
        (
            "vector_data",
            Arc::new(float64_list_array(vec![vec![1.0, 2.0, 3.0], vec![2.0, 1.0]])) as ArrayRef,
        ),
        ("vector_shape", Arc::new(int32_list_array(vec![vec![3], vec![2]])) as ArrayRef),
    ])?;
    drop(ctx.register_batch("sparse_inputs", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                sparse_matvec(
                    make_csr_matrix_batch(shape, row_ptrs, col_indices, values),
                    make_variable_tensor(vector_data, vector_shape, 1)
                ) AS result
             FROM sparse_inputs
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let result_schema = batches[0].schema();
    let result_field = result_schema.field(1);
    assert_eq!(result_field.extension_type_name(), Some("arrow.variable_shape_tensor"));
    let mut tensor_rows = ndarrow::variable_shape_tensor_iter::<Float64Type>(
        result_field,
        struct_column(&batches[0], 1),
    )
    .expect("variable-shape output");
    let (_, first_vector) = tensor_rows.next().expect("row0").expect("row0 tensor");
    let first_vector = first_vector.into_dimensionality::<Ix1>().expect("rank-1 vector");
    let (_, second_vector) = tensor_rows.next().expect("row1").expect("row1 tensor");
    let second_vector = second_vector.into_dimensionality::<Ix1>().expect("rank-1 vector");
    assert_close(first_vector[[0]], 7.0);
    assert_close(first_vector[[1]], 6.0);
    assert_close(second_vector[[0]], 8.0);
    assert_close(second_vector[[1]], 16.0);
    Ok(())
}

#[tokio::test]
async fn sql_direct_variable_shape_tensor_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let (tensor_field, tensor_storage) = ndarrow::arrays_to_variable_shape_tensor(
        "tensor",
        vec![
            Array2::from_shape_vec((2, 3), vec![1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0])
                .expect("row 0 tensor")
                .into_dyn(),
            Array2::from_shape_vec((1, 3), vec![7.0_f32, 8.0, 9.0])
                .expect("row 1 tensor")
                .into_dyn(),
        ],
        Some(vec![None, Some(3)]),
    )
    .expect("variable tensor batch");
    let schema =
        Arc::new(Schema::new(vec![Field::new("id", DataType::Int64, false), tensor_field]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
        Arc::new(tensor_storage) as ArrayRef,
    ])?;
    drop(ctx.register_batch("direct_variable_tensors", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                tensor_variable_sum_last_axis(tensor) AS reduced,
                tensor_variable_l2_norm_last_axis(tensor) AS norms
             FROM direct_variable_tensors
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let schema = batches[0].schema();
    let mut reduced_rows = ndarrow::variable_shape_tensor_iter::<Float32Type>(
        schema.field(1),
        struct_column(&batches[0], 1),
    )
    .expect("reduced variable tensor");
    let (_, first_reduced) = reduced_rows.next().expect("row 0 reduced").expect("row 0 reduced");
    let first_reduced = first_reduced.into_dimensionality::<Ix1>().expect("rank-1 reduced");
    let (_, second_reduced) = reduced_rows.next().expect("row 1 reduced").expect("row 1 reduced");
    let second_reduced = second_reduced.into_dimensionality::<Ix1>().expect("rank-1 reduced");
    assert_close32(first_reduced[[0]], 6.0);
    assert_close32(first_reduced[[1]], 15.0);
    assert_close32(second_reduced[[0]], 24.0);

    let mut norm_rows = ndarrow::variable_shape_tensor_iter::<Float32Type>(
        schema.field(2),
        struct_column(&batches[0], 2),
    )
    .expect("norm variable tensor");
    let (_, first_norms) = norm_rows.next().expect("row 0 norms").expect("row 0 norms");
    let first_norms = first_norms.into_dimensionality::<Ix1>().expect("rank-1 norms");
    let (_, second_norms) = norm_rows.next().expect("row 1 norms").expect("row 1 norms");
    let second_norms = second_norms.into_dimensionality::<Ix1>().expect("rank-1 norms");
    assert_close32(first_norms[[0]], 14.0_f32.sqrt());
    assert_close32(first_norms[[1]], 77.0_f32.sqrt());
    assert_close32(second_norms[[0]], 194.0_f32.sqrt());
    Ok(())
}

#[tokio::test]
async fn sql_direct_sparse_extension_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let (matrix_field, matrices) = ndarrow::csr_batch_to_extension_array(
        "matrix",
        vec![[2, 3], [2, 2]],
        vec![vec![0, 2, 3], vec![0, 1, 3]],
        vec![vec![0, 2, 1], vec![0, 0, 1]],
        vec![vec![1.0_f32, 2.0, 3.0], vec![4.0, 5.0, 6.0]],
    )
    .expect("sparse matrix batch");
    let (vector_field, vectors) = ndarrow::arrays_to_variable_shape_tensor(
        "vector",
        vec![
            Array1::from_vec(vec![1.0_f32, 2.0, 3.0]).into_dyn(),
            Array1::from_vec(vec![2.0_f32, 1.0]).into_dyn(),
        ],
        Some(vec![None]),
    )
    .expect("sparse rhs batch");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        matrix_field,
        vector_field,
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
        Arc::new(matrices) as ArrayRef,
        Arc::new(vectors) as ArrayRef,
    ])?;
    drop(ctx.register_batch("direct_sparse_inputs", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                sparse_matvec(matrix, vector) AS result
             FROM direct_sparse_inputs
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let schema = batches[0].schema();
    let mut result_rows = ndarrow::variable_shape_tensor_iter::<Float32Type>(
        schema.field(1),
        struct_column(&batches[0], 1),
    )
    .expect("sparse matvec output");
    let (_, first) = result_rows.next().expect("row 0 result").expect("row 0 result");
    let first = first.into_dimensionality::<Ix1>().expect("rank-1 result");
    let (_, second) = result_rows.next().expect("row 1 result").expect("row 1 result");
    let second = second.into_dimensionality::<Ix1>().expect("rank-1 result");
    assert_close32(first[[0]], 7.0);
    assert_close32(first[[1]], 6.0);
    assert_close32(second[[0]], 8.0);
    assert_close32(second[[1]], 16.0);
    Ok(())
}

#[tokio::test]
async fn sql_direct_sparse_lu_solve_executes() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let (matrix_field, matrices) = ndarrow::csr_batch_to_extension_array(
        "matrix",
        vec![[2, 2], [2, 2]],
        vec![vec![0, 2, 4], vec![0, 1, 2]],
        vec![vec![0, 1, 0, 1], vec![0, 1]],
        vec![vec![4.0_f32, 1.0, 1.0, 3.0], vec![2.0, 5.0]],
    )
    .expect("sparse lu matrix batch");
    let (rhs_field, rhs) = ndarrow::arrays_to_variable_shape_tensor(
        "rhs",
        vec![
            Array1::from_vec(vec![1.0_f32, 2.0]).into_dyn(),
            Array1::from_vec(vec![4.0_f32, 10.0]).into_dyn(),
        ],
        Some(vec![None]),
    )
    .expect("sparse lu rhs batch");
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int64, false),
        matrix_field,
        rhs_field,
    ]));
    let batch = RecordBatch::try_new(schema, vec![
        Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef,
        Arc::new(matrices) as ArrayRef,
        Arc::new(rhs) as ArrayRef,
    ])?;
    drop(ctx.register_batch("direct_sparse_solve_inputs", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                sparse_lu_solve(matrix, rhs) AS solved
             FROM direct_sparse_solve_inputs
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let schema = batches[0].schema();
    let mut solved_rows = ndarrow::variable_shape_tensor_iter::<Float32Type>(
        schema.field(1),
        struct_column(&batches[0], 1),
    )
    .expect("sparse lu output");
    let (_, first) = solved_rows.next().expect("row 0 solved").expect("row 0 solved");
    let first = first.into_dimensionality::<Ix1>().expect("rank-1 solved");
    let (_, second) = solved_rows.next().expect("row 1 solved").expect("row 1 solved");
    let second = second.into_dimensionality::<Ix1>().expect("rank-1 solved");
    assert_close32(first[[0]], 1.0 / 11.0);
    assert_close32(first[[1]], 7.0 / 11.0);
    assert_close32(second[[0]], 2.0);
    assert_close32(second[[1]], 2.0);
    Ok(())
}

#[tokio::test]
async fn sql_sparse_lu_solve_executes() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        ("shape", Arc::new(int32_list_array(vec![vec![2, 2], vec![2, 2]])) as ArrayRef),
        ("row_ptrs", Arc::new(int32_list_array(vec![vec![0, 2, 4], vec![0, 1, 2]])) as ArrayRef),
        ("col_indices", Arc::new(u32_list_array(vec![vec![0, 1, 0, 1], vec![0, 1]])) as ArrayRef),
        (
            "values",
            Arc::new(float64_list_array(vec![vec![4.0, 1.0, 1.0, 3.0], vec![2.0, 5.0]]))
                as ArrayRef,
        ),
        (
            "rhs_data",
            Arc::new(float64_list_array(vec![vec![1.0, 2.0], vec![4.0, 10.0]])) as ArrayRef,
        ),
        ("rhs_shape", Arc::new(int32_list_array(vec![vec![2], vec![2]])) as ArrayRef),
    ])?;
    drop(ctx.register_batch("sparse_solve_inputs", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                sparse_lu_solve(
                    make_csr_matrix_batch(shape, row_ptrs, col_indices, values),
                    make_variable_tensor(rhs_data, rhs_shape, 1)
                ) AS solved
             FROM sparse_solve_inputs
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    assert_eq!(int64_column(&batches[0], 0).values().as_ref(), &[1, 2]);

    let schema = batches[0].schema();
    let solved_field = schema.field(1);
    let mut solved_rows = ndarrow::variable_shape_tensor_iter::<Float64Type>(
        solved_field,
        struct_column(&batches[0], 1),
    )
    .expect("sparse solve output");
    let (_, first) = solved_rows.next().expect("row 0").expect("row 0 tensor");
    let first = first.into_dimensionality::<Ix1>().expect("rank-1 vector");
    let (_, second) = solved_rows.next().expect("row 1").expect("row 1 tensor");
    let second = second.into_dimensionality::<Ix1>().expect("rank-1 vector");
    assert_close(first[[0]], 1.0 / 11.0);
    assert_close(first[[1]], 7.0 / 11.0);
    assert_close(second[[0]], 2.0);
    assert_close(second[[1]], 2.0);

    Ok(())
}

#[tokio::test]
async fn sql_tensor_constructor_pipeline_executes() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64, 2])) as ArrayRef),
        (
            "tensor_values",
            Arc::new(float64_list_array(vec![vec![1.0, 2.0, 3.0, 4.0], vec![5.0, 6.0, 7.0, 8.0]]))
                as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("tensor_inputs", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                tensor_sum_last_axis(make_tensor(tensor_values, 2, 2)) AS reduced
             FROM tensor_inputs
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 2);
    let reduced_schema = batches[0].schema();
    let reduced_field = reduced_schema.field(1);
    let reduced_storage = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("fixed-shape tensor output");
    let reduced =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(reduced_field, reduced_storage)
            .expect("reduced tensor")
            .into_dimensionality::<Ix2>()
            .expect("rank-2 reduction");
    assert_close(reduced[[0, 0]], 3.0);
    assert_close(reduced[[0, 1]], 7.0);
    assert_close(reduced[[1, 0]], 11.0);
    assert_close(reduced[[1, 1]], 15.0);
    Ok(())
}

#[tokio::test]
async fn sql_tensor_axis_queries_execute() -> Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batch = RecordBatch::try_from_iter(vec![
        ("id", Arc::new(Int64Array::from(vec![1_i64])) as ArrayRef),
        (
            "tensor_values",
            Arc::new(float64_list_array(vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])) as ArrayRef,
        ),
        (
            "left_values",
            Arc::new(float64_list_array(vec![vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]])) as ArrayRef,
        ),
        (
            "right_values",
            Arc::new(float64_list_array(vec![vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0]])) as ArrayRef,
        ),
    ])?;
    drop(ctx.register_batch("tensor_axis_inputs", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                tensor_permute_axes(make_tensor(tensor_values, 2, 3), 1, 0) AS permuted,
                tensor_contract_axes(
                    make_tensor(left_values, 2, 3),
                    make_tensor(right_values, 3, 2),
                    1,
                    0
                ) AS contracted
             FROM tensor_axis_inputs",
        )
        .await?
        .collect()
        .await?;

    assert_eq!(batches.len(), 1);
    assert_eq!(batches[0].num_rows(), 1);
    assert_eq!(int64_column(&batches[0], 0).value(0), 1);

    let schema = batches[0].schema();
    let permuted_field = schema.field(1);
    let permuted_storage = batches[0]
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("permuted tensor output");
    let permuted =
        ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(permuted_field, permuted_storage)
            .expect("permuted tensor")
            .into_dimensionality::<Ix3>()
            .expect("rank-3 permuted tensor");
    assert_close(permuted[[0, 0, 0]], 1.0);
    assert_close(permuted[[0, 0, 1]], 4.0);
    assert_close(permuted[[0, 2, 0]], 3.0);
    assert_close(permuted[[0, 2, 1]], 6.0);

    let contracted_field = schema.field(2);
    let contracted_storage = batches[0]
        .column(2)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("contracted tensor output");
    let contracted = ndarrow::fixed_shape_tensor_as_array_viewd::<Float64Type>(
        contracted_field,
        contracted_storage,
    )
    .expect("contracted tensor")
    .into_dimensionality::<Ix3>()
    .expect("rank-3 contracted tensor");
    assert_close(contracted[[0, 0, 0]], 58.0);
    assert_close(contracted[[0, 0, 1]], 64.0);
    assert_close(contracted[[0, 1, 0]], 139.0);
    assert_close(contracted[[0, 1, 1]], 154.0);
    Ok(())
}
