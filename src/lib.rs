#![cfg_attr(docsrs, feature(doc_auto_cfg))]
//! `ndatafusion` is a DataFusion-facing facade over `nabled`.
//!
//! It owns `DataFusion` registration, SQL-facing numerical contracts, and constructor UDFs while
//! delegating numerical semantics to `nabled`.
//!
//! The current public real-valued contract admits both `Float32` and `Float64` across the
//! implemented catalog. SQL callers should construct canonical numerical values with the `make_*`
//! UDF family and then pass those values into the linalg/ml catalog.
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
//! The constructor UDFs are the SQL ingress boundary from ordinary nested `List` values into the
//! canonical Arrow contracts used by the numerical catalog:
//!
//! - `make_vector`
//! - `make_matrix`
//! - `make_tensor`
//! - `make_variable_tensor`
//! - `make_csr_matrix_batch`
//!
//! # Current Domain Coverage
//!
//! The currently admitted real-valued catalog includes:
//!
//! - dense vector row operations
//! - dense matrix products, triangular solves, configurable and zero-config matrix functions,
//!   decompositions, and summary statistics
//! - sparse CSR batch products and transpose
//! - fixed-shape and variable-shape tensor last-axis operations
//! - linear regression and PCA-style ML/stat helpers
//!
//! For project state, scope, and release posture, see the repository docs under `docs/`.

pub mod error;
pub mod functions;
pub mod metadata;
pub mod register;
pub mod signatures;
pub mod udf;
pub mod udfs;

pub use register::register_all;

#[cfg(test)]
mod udf_tests;
