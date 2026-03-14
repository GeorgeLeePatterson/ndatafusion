use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;

#[tokio::main]
async fn main() -> datafusion::common::Result<()> {
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
             ) AS t",
        )
        .await?
        .collect()
        .await?;

    println!("{}", pretty_format_batches(&batches)?);
    Ok(())
}
