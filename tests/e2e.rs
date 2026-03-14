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
use ndarray::{Array1, Array2, Array3, Ix1, Ix2, Ix3};

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

#[test]
fn register_all_accepts_the_current_catalog() {
    let mut registry = MemoryFunctionRegistry::new();

    ndatafusion::register_all(&mut registry)
        .expect("the current scaffold should register successfully");

    assert_eq!(registry.udfs().len(), ndatafusion::udfs::all_default_functions().len());
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
