//! lucene-util: low-level primitives shared across the port. See /PLAN.md.

pub mod base36;
pub mod fixed_bit_set;
pub mod numeric_utils;
pub mod small_float;
pub mod splittable_random;
pub mod term_interner;
pub mod ternary_long_heap;
// Shared test scratch directories (see the module docs). Compiled only for this
// crate's own tests and for consumers that opt in via the `test-support`
// feature on a `[dev-dependencies]` edge -- never in a production build.
#[cfg(any(test, feature = "test-support"))]
pub mod test_support;
pub mod zigzag;

pub use fixed_bit_set::FixedBitSet;
pub use splittable_random::SplittableRandom;
pub use term_interner::{TermId, TermInterner};
pub use ternary_long_heap::TernaryLongHeap;
