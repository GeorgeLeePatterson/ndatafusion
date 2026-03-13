//! `ndatafusion` is a DataFusion-facing facade over `nabled`.
//!
//! It owns `DataFusion` registration and SQL-facing contracts while delegating numerical semantics
//! to `nabled`.

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
