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
    /// `1 / (k1 * ((1 - b) + b * length(i) / avgdl))` for every one of the 256
    /// possible norm bytes, at the default BM25 constants.
    ///
    /// Norms are a single byte, so the whole domain fits in a table. Real
    /// Lucene's `BM25Similarity` builds exactly this (`cache[]`) and scores with
    /// `weight - weight / (1 + freq * normInverse)`, which is algebraically the
    /// same as `idf * freq / (freq + k1 * ((1-b) + b*len/avgdl))` but performs
    /// one division per document instead of two, and skips decoding the norm to
    /// a length at all.
    ///
    /// Only valid for [`crate::similarity::DEFAULT_K1`]/[`DEFAULT_B`]; a caller
    /// using custom parameters must take the arithmetic path.
    norm_inverse: [f32; 256],
    /// The flat one-byte-per-doc norm array, when this field's norms are dense
    /// and one byte wide -- which is the shape `Lucene90NormsConsumer` writes
    /// for any ordinary analyzed field, so it is the case that matters.
    ///
    /// Resolved once here so that scoring a document is `table[bytes[doc]]`.
    /// Going through [`norms::norm_value`] per document instead cost 9.5% of a
    /// term query's profile: it re-tests denseness and width, then builds a
    /// `SliceInput` and seeks it, to read one byte at a known offset.
    /// `None` for a sparse field, a wider norm, a constant-valued field or an
    /// empty one; those still take the general path.
    dense_norm_bytes: Option<&'a [u8]>,
}

/// The flat norm array for a dense, one-byte-per-norm field, or `None` when
/// this field is not that shape.
fn dense_norm_bytes<'a>(data: &'a [u8], entry: &NormsEntry) -> Option<&'a [u8]> {
    if entry.is_empty_field() || !entry.is_dense() || entry.bytes_per_norm != 1 {
        return None;
    }
    let start = usize::try_from(entry.norms_offset).ok()?;
    let len = usize::try_from(entry.num_docs_with_field).ok()?;
    data.get(start..start.checked_add(len)?)
}

/// Builds [`FieldNorms::norm_inverse`] for one average field length.
fn norm_inverse_table(avg_field_length: f32) -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for (i, slot) in t.iter_mut().enumerate() {
        let len = crate::similarity::decode_norm(i as i64);
        *slot = 1.0
            / (crate::similarity::DEFAULT_K1
                * ((1.0 - crate::similarity::DEFAULT_B)
                    + crate::similarity::DEFAULT_B * len / avg_field_length));
    }
    t
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
    /// `avgFieldLength` exactly as real Lucene's `BM25Similarity` computes it:
    /// `sumTotalTermFreq / docCount`, from the field's `.tmd` aggregate
    /// counters ([`lucene_codecs::blocktree::FieldTerms`]).
    ///
    /// **Prefer this over [`FieldNorms::open`].** `open` derives the average by
    /// decoding each doc's norm and averaging the results, and norms are lossy:
    /// `SmallFloat`-quantized into a single byte. The average of the lossy
    /// values is close to, but not equal to, the average of the true lengths,
    /// so BM25 scores came out systematically 0.1-0.6% away from Lucene's --
    /// enough to reorder documents at the top-k boundary. M1's benchmark
    /// cross-check caught it: 19 of 20 queries disagreed with Java on hit sets
    /// for this reason alone. See `docs/benchmarks/verdict.md`.
    ///
    /// It is also cheaper by an order of magnitude: `open` scans every doc in
    /// the segment, which for a per-query call is O(maxDoc) of pure overhead.
    /// This reads two integers.
    ///
    /// `doc_count == 0` (an empty field) would divide by zero in Lucene; this
    /// port falls back to [`crate::similarity::UNNORMED_FIELD_LENGTH`], keeping
    /// the length-normalization term at its no-op constant, matching
    /// [`FieldNorms::open`]'s treatment of the same edge case.
    pub fn from_field_stats(
        data: &'a [u8],
        entry: NormsEntry,
        sum_total_term_freq: i64,
        doc_count: i32,
    ) -> Self {
        let avg_field_length = if doc_count <= 0 {
            crate::similarity::UNNORMED_FIELD_LENGTH
        } else {
            (sum_total_term_freq as f64 / doc_count as f64) as f32
        };
        Self {
            dense_norm_bytes: dense_norm_bytes(data, &entry),
            data,
            entry,
            avg_field_length,
            norm_inverse: norm_inverse_table(avg_field_length),
        }
    }

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
            dense_norm_bytes: dense_norm_bytes(data, &entry),
            data,
            entry,
            avg_field_length,
            norm_inverse: norm_inverse_table(avg_field_length),
        })
    }

    /// This doc's real decoded field length, or
    /// [`crate::similarity::UNNORMED_FIELD_LENGTH`] when the doc legitimately
    /// has no norm (same fallback rationale as [`FieldNorms::open`]'s
    /// zero-live-docs case — a sparse field's absent doc, scored anyway
    /// because it matched the term some other way `norms` doesn't
    /// second-guess here).
    /// This doc's precomputed `1 / (k1 * ((1-b) + b*len/avgdl))`, for scoring
    /// via `weight - weight / (1 + freq * normInverse)` -- see
    /// [`FieldNorms::norm_inverse`]. Avoids decoding the norm to a length and
    /// one of the two divisions the arithmetic form needs.
    #[inline]
    pub fn norm_inverse(&self, doc: i32) -> norms::Result<f32> {
        // The whole point of the table: for the ordinary dense one-byte field
        // this is a load and an index, as `BM25Scorer` reading `cache[norm]`
        // is. Anything else falls through to the general decode below.
        if let Some(bytes) = self.dense_norm_bytes {
            if let Some(&b) = bytes.get(doc as usize) {
                return Ok(self.norm_inverse[b as usize]);
            }
        }
        Ok(match norms::norm_value(self.data, &self.entry, doc)? {
            Some(norm) => self.norm_inverse[(norm as u8) as usize],
            None => {
                // Same fallback as `field_length`: a doc the field legitimately
                // has no norm for is scored at the unnormed length.
                1.0 / (crate::similarity::DEFAULT_K1
                    * ((1.0 - crate::similarity::DEFAULT_B)
                        + crate::similarity::DEFAULT_B * crate::similarity::UNNORMED_FIELD_LENGTH
                            / self.avg_field_length))
            }
        })
    }

    pub fn field_length(&self, doc: i32) -> norms::Result<f32> {
        Ok(match norms::norm_value(self.data, &self.entry, doc)? {
            Some(norm) => crate::similarity::decode_norm(norm),
            None => crate::similarity::UNNORMED_FIELD_LENGTH,
        })
    }
}

#[cfg(test)]
mod tests {

    /// `avgdl = sumTotalTermFreq / docCount`, real Lucene's formula.
    #[test]
    fn from_field_stats_divides_sum_by_doc_count() {
        let n = FieldNorms::from_field_stats(&[], dense_entry(1, 1, 0), 1000, 40);
        assert_eq!(n.avg_field_length, 25.0);
    }

    /// An empty field would divide by zero in Lucene; this port falls back to
    /// the no-op constant, matching `open`'s treatment of the same edge case.
    #[test]
    fn from_field_stats_empty_field_falls_back_to_unnormed() {
        let n = FieldNorms::from_field_stats(&[], dense_entry(1, 1, 0), 0, 0);
        assert_eq!(n.avg_field_length, crate::similarity::UNNORMED_FIELD_LENGTH);
        let n = FieldNorms::from_field_stats(&[], dense_entry(1, 1, 0), 500, -1);
        assert_eq!(n.avg_field_length, crate::similarity::UNNORMED_FIELD_LENGTH);
    }

    /// Why the old averaging approach was wrong, pinned as a test rather than
    /// left as prose.
    ///
    /// Norms are `SmallFloat`-quantized to one byte. Below 25 the encoding is
    /// exact, so averaging decoded norms and dividing exact counters agree --
    /// which is precisely why every existing fixture (whose documents are 1-3
    /// terms long) passed with the wrong formula. Above that range the encoding
    /// is lossy and the two diverge, which is what M1's 5M-document corpus,
    /// with 40-160 term documents, exposed.
    #[test]
    fn small_float_encoding_is_exact_below_25_and_lossy_above() {
        for len in 1u32..25 {
            let decoded =
                crate::similarity::decode_norm(lucene_util::small_float::int_to_byte4(len) as i64);
            assert_eq!(
                decoded, len as f32,
                "length {len} should round-trip exactly"
            );
        }
        // A realistic document length: the round trip no longer returns the
        // input, so an average of decoded norms is not the average of lengths.
        let lossy: Vec<u32> = (100..=160)
            .filter(|&len| {
                crate::similarity::decode_norm(lucene_util::small_float::int_to_byte4(len) as i64)
                    != len as f32
            })
            .collect();
        assert!(
            !lossy.is_empty(),
            "expected SmallFloat to be lossy for realistic document lengths"
        );
    }

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

    /// The dense one-byte fast path and the general `norms::norm_value` path
    /// must agree for every document and every possible norm byte, including
    /// the ones that decode as negative `i8`. They are two implementations of
    /// the same lookup, and the fast one exists only because it is faster --
    /// so the only thing worth testing about it is that it is not also
    /// different.
    #[test]
    fn dense_fast_path_agrees_with_the_general_norm_lookup_for_every_byte() {
        let data: Vec<u8> = (0..=255u8).collect();
        let entry = dense_entry(1, data.len() as i32, 0);
        let n = FieldNorms::from_field_stats(&data, entry, 4000, 256);
        assert!(
            n.dense_norm_bytes.is_some(),
            "this entry is exactly the shape the fast path is for"
        );
        for doc in 0..data.len() as i32 {
            let general = match norms::norm_value(n.data, &n.entry, doc).unwrap() {
                Some(norm) => n.norm_inverse[(norm as u8) as usize],
                None => unreachable!("dense field, every doc has a norm"),
            };
            assert_eq!(
                n.norm_inverse(doc).unwrap(),
                general,
                "fast and general paths disagree at doc {doc} (norm byte {})",
                data[doc as usize]
            );
        }
    }

    /// A doc past the end of a dense field must still be the error the general
    /// path raises, not a silent read of whatever byte follows the array.
    #[test]
    fn dense_fast_path_defers_out_of_range_docs_to_the_general_path() {
        let data = vec![5u8, 15u8, 25u8];
        let entry = dense_entry(1, 3, 0);
        let n = FieldNorms::from_field_stats(&data, entry, 45, 3);
        assert!(n.norm_inverse(3).is_err());
        assert!(n.norm_inverse(-1).is_err());
    }

    /// Shapes the fast path must decline: a wider norm, and a constant-valued
    /// field (`bytes_per_norm == 0`, the value carried in `norms_offset`).
    /// Declining is what keeps them correct, so assert it rather than assume.
    #[test]
    fn fast_path_declines_shapes_it_cannot_serve() {
        let data = vec![0u8; 16];
        assert!(
            FieldNorms::from_field_stats(&data, dense_entry(2, 4, 0), 40, 4)
                .dense_norm_bytes
                .is_none()
        );
        let constant = FieldNorms::from_field_stats(&data, dense_entry(0, 4, 7), 40, 4);
        assert!(constant.dense_norm_bytes.is_none());
        // ...and still scores, through the general path.
        assert_eq!(constant.norm_inverse(0).unwrap(), constant.norm_inverse[7]);
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
