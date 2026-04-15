//! Internal UDF family modules grouped by SQL surface area.
//!
//! Most callers should use [`crate::udfs`] for selective registration or [`crate::register_all`]
//! for the full scalar catalog. This module exposes the lower-level family boundaries that back
//! those higher-level entry points.

pub(crate) mod common;
pub mod constructors;
pub mod decomposition;
pub mod differentiation;
pub(crate) mod docs;
pub mod iterative;
pub mod matrix;
pub mod matrix_equations;
pub mod matrix_functions;
pub mod ml;
pub mod optimization;
pub mod sparse;
pub mod sparse_factorization;
pub mod tensor;
pub mod tensor_decomposition;
pub mod triangular;
pub mod vector;
