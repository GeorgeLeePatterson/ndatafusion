#![cfg_attr(docsrs, feature(doc_auto_cfg))]
//! `ndatafusion` provides linear algebra and machine learning scalar and aggregate UDFs for
//! `DataFusion`.
//!
//! Register the scalar and aggregate catalog with [`register_all`], or register the full SQL
//! surface including table functions with [`register_all_session`]. Call the functions from SQL or
//! by constructing expressions with helpers from [`functions`].
//!
//! The current catalog supports `Float32` and `Float64` across dense vector, dense matrix, sparse
//! CSR, fixed-shape tensor, variable-shape tensor, grouped statistics/model fits, and selected
//! solver routines. The current complex-valued slice covers dense vector, dense matrix,
//! complex PCA, fixed-shape tensor, and variable-shape tensor operations over canonical
//! `ndarrow.complex64` columns.
//!
//! Use the `make_*` constructor family when SQL starts from ordinary `List` values. If a table
//! already stores canonical `FixedSizeList` or extension-backed Arrow values, call the numerical
//! UDFs directly. Selected constructor, aggregate, and control-parameter UDFs also support named
//! arguments in SQL. For numerical UDFs, prefer positional data arguments first and named trailing
//! control arguments after.
//!
//! # Quick Start
//!
//! ```no_run
//! use datafusion::prelude::SessionContext;
//!
//! #[tokio::main]
//! async fn main() -> datafusion::common::Result<()> {
//!     let mut ctx = SessionContext::new();
//!     ndatafusion::register_all(&mut ctx)?;
//!
//!     let batches = ctx
//!         .sql(
//!             "SELECT
//!                 vector_dot(make_vector(left_values, 2), make_vector(right_values, 2)) AS dot,
//!                 matrix_determinant(make_matrix(matrix_values, 2, 2)) AS det
//!              FROM (
//!                 SELECT
//!                     [3.0, 4.0] AS left_values,
//!                     [4.0, 0.0] AS right_values,
//!                     [9.0, 0.0, 0.0, 4.0] AS matrix_values
//!              )",
//!         )
//!         .await?
//!         .collect()
//!         .await?;
//!
//!     assert_eq!(batches[0].num_rows(), 1);
//!     Ok(())
//! }
//! ```
//!
//! # Constructors
//!
//! The constructor UDFs convert ordinary nested `List` values into the canonical Arrow contracts
//! used by the numerical catalog. They are not required when input columns already use those
//! canonical contracts:
//!
//! - `make_vector`
//! - `make_matrix`
//! - `make_tensor`
//! - `make_variable_tensor`
//! - `make_csr_matrix_batch`
//!
//! # Included UDF Groups
//!
//! The registered catalog includes:
//!
//! - constructors for canonical numerical values
//! - dense vector operations, including the current complex-vector subset
//! - complex dense matrix products, statistics, iterative solvers, matrix functions, and the
//!   current complex eigen / Schur / polar subset
//! - dense matrix operations, decompositions, direct solvers, and Sylvester matrix equations
//! - sparse CSR operations
//! - fixed-shape and variable-shape tensor operations, including the current complex tensor subset
//! - differentiation, optimization, and matrix-equation helpers
//! - statistics, real and complex PCA, iterative solvers, and linear regression
//! - grouped aggregate fits for covariance, correlation, PCA, and linear regression
//! - sparse factorization, tensor decomposition, and the `unpack_struct` table function via
//!   [`register_all_session`]
//!
//! For the complete SQL function inventory and notes on result contracts, see `CATALOG.md` in the
//! repository root. For small copy-paste query examples, see `EXERCISES.md`.
//!
//! ## Features
//!
//! Feature forwarding follows `nabled` directly:
//!
//! * `blas`
//! * `lapack-provider`
//! * `openblas-system`
//! * `openblas-static`
//! * `netlib-system`
//! * `netlib-static`
//! * `magma-system`
//! * `accelerator-rayon`
//! * `accelerator-wgpu`

pub mod error;
pub mod functions;
pub(crate) mod metadata;
pub mod register;
pub(crate) mod signatures;
pub(crate) mod udaf;
pub mod udafs;
pub mod udf;
pub mod udfs;
pub(crate) mod udtf;

pub use register::{register_all, register_all_session};

#[cfg(test)]
mod udf_tests;
