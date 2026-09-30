//! The index-time half of Lucene's `Similarity`: `computeNorm`, and the
//! `FieldInvertState` it reads.
//!
//! Java's `Similarity` is one class with an index-time method (`computeNorm`,
//! which `IndexingChain.PerField.finish` calls for every document's field)
//! and a query-time one (`scorer`). This crate sits below `lucene-search` in
//! the dependency graph (`util <- store <- codecs <- index <- search`), so the
//! contract is split along that line: [`NormSimilarity`] here is what the
//! writer needs, and `lucene_search::similarities::Similarity` extends it with
//! the scorer. Every Lucene similarity `lucene-search` ports implements both,
//! so an `Arc<dyn Similarity>` upcasts to the `Arc<dyn NormSimilarity>`
//! [`crate::index_writer::IndexWriter::set_similarity`] takes -- the same
//! object configures the writer and the searcher, as it does in Java.

use std::fmt;

use lucene_util::small_float;

/// The part of `FieldInvertState` `Similarity.computeNorm` reads: one
/// document's value(s) of one field, as the inverter counted them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FieldInvertState {
    /// The field's `IndexOptions` is `DOCS`.
    pub docs_only: bool,
    /// Tokens, positions counted (`getLength`).
    pub length: i32,
    /// Tokens at a position increment of 0 (`getNumOverlap`).
    pub num_overlap: i32,
    /// Distinct terms (`getUniqueTermCount`).
    pub unique_term_count: i32,
}

/// `Similarity.computeNorm`: the norm the writer stores for a document's
/// field. Never called for a field without tokens (`IndexingChain` stores `0`
/// for it), and a `0` for one with tokens is an error, as in Java.
pub trait NormSimilarity: Send + Sync + fmt::Debug {
    /// `Similarity.computeNorm(state)`; `field` is `state.getName()`.
    fn compute_norm(&self, field: &str, state: &FieldInvertState) -> i64 {
        let _ = field;
        default_compute_norm(self.discount_overlaps(), state)
    }

    /// `Similarity.getDiscountOverlaps`.
    fn discount_overlaps(&self) -> bool {
        true
    }
}

/// `Similarity.computeNorm`'s default body: `SmallFloat.intToByte4` of the
/// unique term count for a `DOCS` field, of the length (less the overlaps,
/// when they are discounted) otherwise, widened from a `byte` to a `long`.
pub fn default_compute_norm(discount_overlaps: bool, state: &FieldInvertState) -> i64 {
    let num_terms = if state.docs_only {
        state.unique_term_count
    } else if discount_overlaps {
        state.length.wrapping_sub(state.num_overlap)
    } else {
        state.length
    };
    // A negative count (Java throws) cannot arise: overlaps are a subset of
    // the length.
    i64::from(small_float::int_to_byte4(num_terms.max(0) as u32) as i8)
}

/// The similarity a writer computes norms with when none is set:
/// `BM25Similarity`'s (Lucene's default), overlaps discounted.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct DefaultNormSimilarity;

impl NormSimilarity for DefaultNormSimilarity {}

#[cfg(test)]
mod tests {
    use super::*;

    fn state(docs_only: bool, length: i32, num_overlap: i32, unique: i32) -> FieldInvertState {
        FieldInvertState {
            docs_only,
            length,
            num_overlap,
            unique_term_count: unique,
        }
    }

    #[derive(Debug)]
    struct NoDiscount;
    impl NormSimilarity for NoDiscount {
        fn discount_overlaps(&self) -> bool {
            false
        }
    }

    /// The three counts the default body chooses between.
    #[test]
    fn the_default_norm_counts_what_java_counts() {
        let byte4 = |n: u32| i64::from(small_float::int_to_byte4(n) as i8);
        let d = DefaultNormSimilarity;
        assert_eq!(d.compute_norm("f", &state(false, 10, 4, 3)), byte4(6));
        assert_eq!(
            NoDiscount.compute_norm("f", &state(false, 10, 4, 3)),
            byte4(10)
        );
        // `DOCS`: distinct terms, whatever the overlaps.
        assert_eq!(d.compute_norm("f", &state(true, 10, 4, 3)), byte4(3));
        assert_eq!(
            NoDiscount.compute_norm("f", &state(true, 10, 4, 3)),
            byte4(3)
        );
        assert!(d.discount_overlaps());
        // A large length encodes to a negative byte, sign-extended.
        assert!(d.compute_norm("f", &state(false, 1_000_000, 0, 1)) < 0);
    }
}
