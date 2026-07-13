//! Opens one field's real BM25 norms for a segment and computes
//! `avgFieldLength` once, so [`crate::search_term_query_scored`]/
//! [`crate::search_boolean_query_scored`] can score every matched doc against
//! real per-doc field lengths instead of [`crate::similarity::UNNORMED_FIELD_LENGTH`].
//!
//! This is deliberately a thin bundle, not a general "norms reader" — the
//! actual byte-level decode already lives in `lucene_codecs::norms`
//! (`norm_value`) and `lucene_util::small_float`/`similarity::decode_norm`;
//! this module only adds the "compute `avgFieldLength` once per field per
//! query, not once per scored doc" caching [`crate::similarity`]'s doc
//! comment calls for (see the `rust-performance` skill: recomputing a
//! segment-wide average per doc would be correct but needlessly slow).

use lucene_codecs::norms::{self, NormsEntry};
use lucene_util::fixed_bit_set::FixedBitSet;

/// One field's opened norms data plus its precomputed `avgFieldLength` —
/// pass `Some(&FieldNorms)` to a `*_scored` search function for real BM25
/// length-normalization on that field; `None` falls back to
/// [`crate::similarity::UNNORMED_FIELD_LENGTH`] for both `fieldLength` and
/// `avgFieldLength`, a documented, deliberate approximation (not silently
/// wrong data) for a field with no opened `.nvd`/`.nvm` pair — e.g. norms
/// disabled for that field, or a caller that hasn't wired up norms opening
/// yet.
#[derive(Debug, Clone)]
pub struct FieldNorms<'a> {
    /// The segment's whole `.nvd` file, matching [`norms::norm_value`]'s
    /// `data` parameter.
    pub data: &'a [u8],
    pub entry: NormsEntry,
    /// `sum(decode_norm(doc)) / count(docs with a norm)` across every *live*
    /// doc in `0..max_doc`, computed once by [`FieldNorms::open`].
    pub avg_field_length: f32,
}

impl<'a> FieldNorms<'a> {
    /// Computes `avgFieldLength` once by scanning every live doc in
    /// `0..max_doc` and decoding its norm (skipping docs the field's norms
    /// entry legitimately has none for — a sparse field's absent docs, or an
    /// entirely empty field). Returns `avg_field_length ==
    /// UNNORMED_FIELD_LENGTH` (not an error) when no live doc has a norm for
    /// this field at all — an edge case (every doc deleted, or an empty
    /// field) real Lucene's own `avgdl = sumTotalTermFreq / docCount` would
    /// divide-by-zero on; this port's fallback keeps the length-
    /// normalization term at its "no-op" constant instead.
    pub fn open(
        data: &'a [u8],
        entry: NormsEntry,
        max_doc: i32,
        live_docs: Option<&FixedBitSet>,
    ) -> norms::Result<Self> {
        let mut sum = 0.0f64;
        let mut count = 0i64;
        for doc in 0..max_doc {
            if !live_docs.is_none_or(|bits| bits.get(doc as usize)) {
                continue;
            }
            if let Some(norm) = norms::norm_value(data, &entry, doc)? {
                sum += crate::similarity::decode_norm(norm) as f64;
                count += 1;
            }
        }
        let avg_field_length = if count == 0 {
            crate::similarity::UNNORMED_FIELD_LENGTH
        } else {
            (sum / count as f64) as f32
        };
        Ok(Self {
            data,
            entry,
            avg_field_length,
        })
    }

    /// This doc's real decoded field length, or
    /// [`crate::similarity::UNNORMED_FIELD_LENGTH`] when the doc legitimately
    /// has no norm (same fallback rationale as [`FieldNorms::open`]'s
    /// zero-live-docs case — a sparse field's absent doc, scored anyway
    /// because it matched the term some other way `norms` doesn't
    /// second-guess here).
    pub fn field_length(&self, doc: i32) -> norms::Result<f32> {
        Ok(match norms::norm_value(self.data, &self.entry, doc)? {
            Some(norm) => crate::similarity::decode_norm(norm),
            None => crate::similarity::UNNORMED_FIELD_LENGTH,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_codecs::norms::NormsEntry;

    fn dense_entry(bytes_per_norm: u8, num_docs: i32, norms_offset: i64) -> NormsEntry {
        NormsEntry {
            field_number: 0,
            docs_with_field_offset: -1, // dense
            docs_with_field_length: 0,
            jump_table_entry_count: 0,
            dense_rank_power: 0,
            num_docs_with_field: num_docs,
            bytes_per_norm,
            norms_offset,
        }
    }

    #[test]
    fn avg_field_length_averages_decoded_norms() {
        // Three docs, raw norm bytes 5, 15, 25 -- bytes < 24 decode exactly
        // (subnormal range, see `lucene_util::small_float`), byte 25 decodes
        // to something a bit larger than 25.
        let data = vec![5u8, 15u8, 25u8];
        let entry = dense_entry(1, 3, 0);
        let fn_ = FieldNorms::open(&data, entry, 3, None).unwrap();
        let expected = (5.0 + 15.0 + lucene_util::small_float::byte4_to_int(25) as f32) / 3.0;
        assert!((fn_.avg_field_length - expected).abs() < 1e-4);
    }

    #[test]
    fn avg_field_length_skips_dead_docs() {
        let data = vec![5u8, 15u8, 25u8];
        let entry = dense_entry(1, 3, 0);
        let mut live = FixedBitSet::new(3);
        live.set(0);
        live.set(2);
        // doc 1 (norm byte 15) is dead and must not affect the average.
        let fn_ = FieldNorms::open(&data, entry, 3, Some(&live)).unwrap();
        let expected = (5.0 + lucene_util::small_float::byte4_to_int(25) as f32) / 2.0;
        assert!((fn_.avg_field_length - expected).abs() < 1e-4);
    }

    #[test]
    fn avg_field_length_falls_back_to_unnormed_when_no_live_docs_have_a_norm() {
        let data = vec![5u8, 15u8];
        let entry = dense_entry(1, 2, 0);
        let live = FixedBitSet::new(2); // nothing set -- all dead
        let fn_ = FieldNorms::open(&data, entry, 2, Some(&live)).unwrap();
        assert_eq!(
            fn_.avg_field_length,
            crate::similarity::UNNORMED_FIELD_LENGTH
        );
    }

    #[test]
    fn field_length_decodes_one_doc() {
        let data = vec![5u8, 15u8, 25u8];
        let entry = dense_entry(1, 3, 0);
        let fn_ = FieldNorms::open(&data, entry, 3, None).unwrap();
        assert_eq!(fn_.field_length(0).unwrap(), 5.0);
        assert_eq!(fn_.field_length(1).unwrap(), 15.0);
        assert_eq!(
            fn_.field_length(2).unwrap(),
            lucene_util::small_float::byte4_to_int(25) as f32
        );
    }

    #[test]
    fn field_length_falls_back_for_a_doc_with_no_norm() {
        // An empty field: no doc has a norm value at all.
        let mut entry = dense_entry(1, 0, 0);
        entry.docs_with_field_offset = -2; // empty
        let fn_ = FieldNorms::open(&[], entry, 3, None).unwrap();
        assert_eq!(
            fn_.field_length(0).unwrap(),
            crate::similarity::UNNORMED_FIELD_LENGTH
        );
    }
}
