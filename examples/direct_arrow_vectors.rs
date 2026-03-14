use std::sync::Arc;

use datafusion::arrow::array::types::Float32Type;
use datafusion::arrow::array::{ArrayRef, FixedSizeListArray, Int64Array};
use datafusion::arrow::record_batch::RecordBatch;
use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;

fn float32_fixed_size_list_array(rows: Vec<Vec<f32>>) -> FixedSizeListArray {
    let width = rows.first().map_or(0, Vec::len);
    FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
        rows.into_iter().map(|row| Some(row.into_iter().map(Some).collect::<Vec<_>>())),
        i32::try_from(width).expect("fixed-size-list width should fit i32"),
    )
}

#[tokio::main]
async fn main() -> datafusion::common::Result<()> {
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
    drop(ctx.register_batch("embeddings", batch)?);

    let batches = ctx
        .sql(
            "SELECT
                id,
                vector_l2_norm(left_vector) AS norm,
                vector_dot(left_vector, right_vector) AS dot,
                vector_cosine_similarity(left_vector, right_vector) AS similarity
             FROM embeddings
             ORDER BY id",
        )
        .await?
        .collect()
        .await?;

    println!("{}", pretty_format_batches(&batches)?);
    Ok(())
}
