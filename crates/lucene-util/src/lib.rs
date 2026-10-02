//! lucene-util: low-level primitives shared across the port. See /PLAN.md.

pub mod automaton;
pub mod base36;
pub mod bit_util;
pub mod bits;
pub mod bytes_ref_hash;
pub mod doc_id_sort;
pub mod fixed_bit_set;
pub mod float_heap;
pub mod geo;
pub mod java_random;
pub mod live_docs;
pub mod math_util;
pub mod numeric_utils;
pub mod packed;
pub mod packed_longs;
pub mod point_values_relation;
pub mod quantization;
pub mod simd;
pub mod sloppy_math;
pub mod small_float;
pub mod sorter;
pub mod sparse_fixed_bit_set;
pub mod spatial3d;
pub mod splittable_random;
pub mod strict_math;
pub mod string_helper;
pub mod term_interner;
pub mod ternary_long_heap;
pub mod vector_util;
pub mod version;
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
