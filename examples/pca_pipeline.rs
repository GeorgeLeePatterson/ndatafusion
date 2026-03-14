use datafusion::arrow::util::pretty::pretty_format_batches;
use datafusion::prelude::SessionContext;

#[tokio::main]
async fn main() -> datafusion::common::Result<()> {
    let mut ctx = SessionContext::new();
    ndatafusion::register_all(&mut ctx)?;

    let batches = ctx
        .sql(
            "SELECT
                pca.explained_variance_ratio AS explained,
                pca.scores AS scores,
                matrix_pca_transform(matrix, pca) AS projected,
                matrix_pca_inverse_transform(pca.scores, pca) AS reconstructed
             FROM (
                SELECT
                    make_matrix(matrix_values, 3, 2) AS matrix,
                    matrix_pca(make_matrix(matrix_values, 3, 2)) AS pca
                FROM (
                    SELECT [1.0, 2.0, 3.0, 4.0, 5.0, 6.0] AS matrix_values
                ) AS input
             ) AS t",
        )
        .await?
        .collect()
        .await?;

    println!("{}", pretty_format_batches(&batches)?);
    Ok(())
}
