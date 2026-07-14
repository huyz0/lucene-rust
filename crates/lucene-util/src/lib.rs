//! lucene-util: low-level primitives shared across the port. See /PLAN.md.

pub mod base36;
pub mod fixed_bit_set;
pub mod small_float;
pub mod term_interner;
pub mod zigzag;

pub use fixed_bit_set::FixedBitSet;
pub use term_interner::{TermId, TermInterner};
