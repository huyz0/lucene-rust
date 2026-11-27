//! Throwaway spike: SQL over lucene-rust segments with DataFusion plus a
//! fast-path router. Measures the cost of each layer against the direct API.

pub mod data;
pub mod direct;
pub mod index;
pub mod provider;
pub mod router;
pub mod rule;

pub const VEC_DIM: usize = 32;
