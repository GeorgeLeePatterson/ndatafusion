use std::sync::Arc;

use datafusion::arrow::array::types::Float64Type;
use datafusion::arrow::array::{
    Array, ArrayRef, FixedSizeListArray, Float64Array, Int64Array, ListArray, StructArray,
};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::common::Result;
use datafusion::common::utils::arrays_into_list_array;
use datafusion::execution::FunctionRegistry;
use datafusion::execution::registry::MemoryFunctionRegistry;
use datafusion::prelude::SessionContext;
use ndarray::Ix1;

fn assert_close(actual: f64, expected: f64) {
    let delta = (actual - expected).abs();
    assert!(delta < 1.0e-9, "expected {expected}, got {actual}, delta {delta}");
}

fn float64_list_array(rows: Vec<Vec<f64>>) -> ListArray {
    ListArray::from_iter_primitive::<Float64Type, _, _>(
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
                matrix_svd_rank(make_matrix(rank_matrix_values, 2, 2)) AS svd_rank
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
    assert_eq!(int64_column(&batches[0], 3).values().as_ref(), &[2, 1]);
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
            .into_dimensionality::<ndarray::Ix3>()
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
            .into_dimensionality::<ndarray::Ix3>()
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
            .into_dimensionality::<ndarray::Ix3>()
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
            .into_dimensionality::<ndarray::Ix3>()
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
            .into_dimensionality::<ndarray::Ix3>()
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
            .into_dimensionality::<ndarray::Ix2>()
            .expect("rank-2 reduction");
    assert_close(reduced[[0, 0]], 3.0);
    assert_close(reduced[[0, 1]], 7.0);
    assert_close(reduced[[1, 0]], 11.0);
    assert_close(reduced[[1, 1]], 15.0);
    Ok(())
}
