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
    /// `decode_norm(i)` for every one of the 256 possible norm bytes. Same idea
    /// as [`FieldNorms::norm_inverse`] and the same reason: the domain is one
    /// byte wide, so the decode is a table lookup, not a computation.
    norm_length: [f32; 256],
    /// A sparse field's `IndexedDISI` region, sliced (and therefore
    /// bounds-checked) once. `None` when the field is dense, empty, or the
    /// recorded region does not lie inside `data` -- in which case the general
    /// [`norms::norm_value`] path re-raises that error per lookup.
    ///
    /// This used to be an `Option<Vec<i32>>` holding *every* doc id with a
    /// norm, decoded eagerly by each constructor, because
    /// `norms::norm_value`'s sparse branch then re-decoded the whole region on
    /// every call. c2 turned `IndexedDISI` into a real incremental cursor, so
    /// neither is needed: [`FieldNorms::cursor`] hands a per-scorer
    /// [`FieldNormsCursor`] that walks the region forward once per *scan*
    /// instead of once per lookup, and the `Vec` -- 4 bytes per
    /// document-with-the-field, allocated per `FieldNorms` and therefore per
    /// query per leaf -- is gone.
    sparse_region: Option<&'a [u8]>,
}

/// One scorer's position in a sparse field's norms, the way Lucene's
/// `NumericDocValues` is one scorer's position in a `.nvd`.
///
/// `FieldNorms` is the shared, immutable per-segment entry (Lucene's
/// `SegmentCoreReaders` slot) and is `Sync`, so `multi_segment.rs` can hand the
/// same `&FieldNorms` to every `rayon` task. The *cursor* is the mutable part,
/// and each leaf's scan owns its own -- which is what lets the sparse lookup be
/// an incremental `IndexedDISI` walk (`advance_exact`, `&mut self`) without
/// making `FieldNorms` itself non-`Sync`. Creating one allocates nothing and
/// reads no bytes.
///
/// Forward-only, like the `DisiCursor` underneath it and like Lucene's
/// `advanceExact`, but tolerant rather than panicking: a target behind the last
/// one rewinds the cursor (one block-header walk, no allocation) instead of
/// asserting, so a caller whose doc order is not monotonic -- a fuzzy query
/// restarting per expanded term, say -- stays correct and merely pays what the
/// old per-call decode paid anyway.
#[derive(Debug)]
pub struct FieldNormsCursor<'n, 'a> {
    norms: &'n FieldNorms<'a>,
    /// `Some` only for a sparse field; a dense or empty one never needs it.
    disi: Option<lucene_codecs::indexed_disi::DisiCursor<'a>>,
}

/// A sparse field's `IndexedDISI` region, or `None` when the field is dense,
/// empty, or the region cannot be sliced out of `data` (in which case the
/// general path re-raises the error per lookup).
///
/// Only the *bounds* are resolved here -- nothing is decoded, and
/// `dense_rank_power` is not validated: an illegal one surfaces from the
/// cursor's own first `advance_exact`, as an error rather than a guessed norm.
fn sparse_region<'a>(data: &'a [u8], entry: &NormsEntry) -> Option<&'a [u8]> {
    if entry.is_empty_field() || entry.is_dense() {
        return None;
    }
    let start = usize::try_from(entry.docs_with_field_offset).ok()?;
    let len = usize::try_from(entry.docs_with_field_length).ok()?;
    data.get(start..start.checked_add(len)?)
}

/// `BM25Similarity.avgFieldLength(FieldStats)`: `sumTotalTermFreq / docCount`,
/// computed in `f64` and narrowed once, exactly as Java's
/// `(float) (fieldStats.sumTotalTermFreq() / (double) fieldStats.docCount())`.
///
/// `doc_count <= 0` (an empty field) divides by zero in Java; this port falls
/// back to [`crate::similarity::UNNORMED_FIELD_LENGTH`], keeping the
/// length-normalization term at its no-op constant. Java gets there by
/// `IndexSearcher.fieldStats` returning `null` for `docCount == 0` and the
/// caller then scoring without a `CollectionStatistics` at all.
pub fn avg_field_length(sum_total_term_freq: i64, doc_count: i64) -> f32 {
    if doc_count <= 0 {
        crate::similarity::UNNORMED_FIELD_LENGTH
    } else {
        (sum_total_term_freq as f64 / doc_count as f64) as f32
    }
}

/// Builds [`FieldNorms::norm_length`]. Independent of `avgFieldLength`, unlike
/// the `norm_inverse` table, but built alongside it to keep the two in step.
fn norm_length_table() -> [f32; 256] {
    let mut t = [0.0f32; 256];
    for (i, slot) in t.iter_mut().enumerate() {
        *slot = crate::similarity::decode_norm(i as i64);
    }
    t
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
        Self::with_avg_field_length(
            data,
            entry,
            avg_field_length(sum_total_term_freq, doc_count as i64),
        )
    }

    /// [`FieldNorms::from_field_stats`] with `avgFieldLength` supplied rather
    /// than derived from this segment's own counters.
    ///
    /// **This is the constructor a multi-segment search wants.**
    /// `IndexSearcher.fieldStats` sums `getSumTotalTermFreq()`/`getDocCount()`
    /// over *every leaf* and hands one `CollectionStatistics` to
    /// `BM25Similarity.scorer`, so Java's `avgdl` is reader-wide exactly the
    /// way its `docFreq`/`docCount` are. Deriving it per leaf gives each
    /// segment a different length-normalization curve for the same term --
    /// the same class of cross-segment ranking bug
    /// [`crate::CollectionStats`] documents for idf, and invisible on a
    /// single-segment index, which is why every fixture missed it.
    ///
    /// [`crate::multi_segment::global_avg_field_length`] computes the value;
    /// [`crate::directory_reader::DirectoryReader::field_norms`] applies it to
    /// every leaf in one call.
    pub fn with_avg_field_length(data: &'a [u8], entry: NormsEntry, avg_field_length: f32) -> Self {
        Self {
            dense_norm_bytes: dense_norm_bytes(data, &entry),
            data,
            entry,
            avg_field_length,
            norm_inverse: norm_inverse_table(avg_field_length),
            norm_length: norm_length_table(),
            sparse_region: sparse_region(data, &entry),
        }
    }

    /// **Not Java's `avgFieldLength`.** Kept for callers that hold a segment's
    /// `.nvd`/`.nvm` and nothing else; prefer
    /// [`FieldNorms::from_field_stats`], which reads the two counters
    /// `BM25Similarity` actually divides.
    ///
    /// Three divergences from `CollectionStatistics`, in decreasing order of
    /// how much they move a score:
    ///
    /// 1. **Quantization.** Norms are `SmallFloat`-encoded into one byte and
    ///    the encoding is lossy above length 24, so averaging *decoded* norms
    ///    is not averaging lengths. Unfixable from norms alone -- this is the
    ///    whole reason `from_field_stats` exists.
    /// 2. **Population.** Java's `docCount` counts every document that has the
    ///    field, deleted ones included. Passing `live_docs` here excludes the
    ///    deleted ones from both the sum and the count, which shifts `avgdl`.
    ///    Every caller in this workspace passes `None`, so this one is latent
    ///    rather than active.
    /// 3. **Empty field.** `doc_count == 0` divides by zero in Java; this
    ///    returns [`crate::similarity::UNNORMED_FIELD_LENGTH`], keeping the
    ///    length-normalization term at its no-op constant.
    ///
    /// Cost is `O(maxDoc)` norm lookups, against `from_field_stats`' two
    /// integer reads.
    pub fn open(
        data: &'a [u8],
        entry: NormsEntry,
        max_doc: i32,
        live_docs: Option<&FixedBitSet>,
    ) -> norms::Result<Self> {
        // The scan is monotonic in `doc`, so it runs off one `FieldNormsCursor`
        // -- for a sparse field that is a single forward `IndexedDISI` walk
        // instead of `norms::norm_value`'s fresh block walk per document, which
        // made this O(maxDoc x blocks).
        let mut scratch = Self {
            dense_norm_bytes: dense_norm_bytes(data, &entry),
            data,
            entry,
            // Only `norm_byte` is used below, and it reads neither; the real
            // values are filled in once the average is known.
            avg_field_length: crate::similarity::UNNORMED_FIELD_LENGTH,
            norm_inverse: [0.0; 256],
            norm_length: norm_length_table(),
            sparse_region: sparse_region(data, &entry),
        };
        let mut sum = 0.0f64;
        let mut count = 0i64;
        {
            let mut cursor = scratch.cursor();
            for doc in 0..max_doc {
                if !live_docs.is_none_or(|bits| bits.get(doc as usize)) {
                    continue;
                }
                if let Some(norm) = cursor.norm_byte(doc)? {
                    sum += crate::similarity::decode_norm(norm as i64) as f64;
                    count += 1;
                }
            }
        }
        let avg_field_length = if count == 0 {
            crate::similarity::UNNORMED_FIELD_LENGTH
        } else {
            (sum / count as f64) as f32
        };
        scratch.avg_field_length = avg_field_length;
        scratch.norm_inverse = norm_inverse_table(avg_field_length);
        Ok(scratch)
    }

    /// A fresh [`FieldNormsCursor`] over this field -- Lucene's
    /// `LeafReader.getNormValues(field)`, which likewise returns a new
    /// `NumericDocValues` per scorer rather than sharing one.
    ///
    /// Allocates nothing and reads no bytes, so a scan takes one per leaf and
    /// then never pays the sparse region's block walk again. **Every
    /// document-at-a-time loop should hold one**; the `&self` conveniences
    /// below are for single-document callers (`explain`) and tests.
    #[inline]
    pub fn cursor(&self) -> FieldNormsCursor<'_, 'a> {
        FieldNormsCursor {
            norms: self,
            disi: self.sparse_region.map(|region| {
                lucene_codecs::indexed_disi::DisiCursor::new(region, self.entry.dense_rank_power)
            }),
        }
    }

    /// Whether a [`FieldNormsCursor`] over this field is only cheap when it is
    /// asked for ascending documents.
    ///
    /// `true` exactly for a sparse field, where a lookup is an `IndexedDISI`
    /// walk and a backwards target costs a `reset()` plus a fresh walk from the
    /// region's first block header. `false` for the ordinary dense one-byte
    /// field, whose lookup is an array index and is therefore
    /// order-insensitive, and for every shape that falls through to
    /// [`norms::norm_value`].
    ///
    /// Exposed so a caller whose natural iteration order is *not* ascending --
    /// [`crate::fuzzy_doc_scores`], which walks each expanded term's postings in
    /// turn and so restarts at a low doc id per term -- can decide whether
    /// reordering its work is worth it, instead of either always paying for a
    /// sort or always paying for the walks.
    #[inline]
    pub fn prefers_ascending_lookups(&self) -> bool {
        self.sparse_region.is_some()
    }

    /// This doc's precomputed `1 / (k1 * ((1-b) + b*len/avgdl))`, for scoring
    /// via `weight - weight / (1 + freq * normInverse)` -- see
    /// [`FieldNorms::norm_inverse_table`](FieldNorms#structfield.norm_inverse).
    ///
    /// One-shot: it builds a [`FieldNormsCursor`] per call, which for a sparse
    /// field means walking that field's `IndexedDISI` block headers from the
    /// start. Correct for any doc in any order, and exactly what a
    /// single-document caller wants; a loop should call [`FieldNorms::cursor`]
    /// once instead.
    #[inline]
    pub fn norm_inverse(&self, doc: i32) -> norms::Result<f32> {
        self.cursor().norm_inverse(doc)
    }

    /// This doc's real decoded field length, or
    /// [`crate::similarity::UNNORMED_FIELD_LENGTH`] when the doc legitimately
    /// has no norm (same fallback rationale as [`FieldNorms::open`]'s
    /// zero-live-docs case -- a sparse field's absent doc, scored anyway
    /// because it matched the term some other way `norms` doesn't
    /// second-guess here).
    ///
    /// One-shot, exactly as [`FieldNorms::norm_inverse`] is.
    #[inline]
    pub fn field_length(&self, doc: i32) -> norms::Result<f32> {
        self.cursor().field_length(doc)
    }
}

impl FieldNormsCursor<'_, '_> {
    /// This doc's raw norm byte, or `None` when the field legitimately has no
    /// norm for it.
    ///
    /// Three shapes, in the order they are tried:
    ///
    /// 1. **Dense, one byte per norm** -- the shape `Lucene90NormsConsumer`
    ///    writes for any ordinary analyzed field, and therefore the one that
    ///    matters: a bounds-checked index into a borrowed slice.
    /// 2. **Sparse** -- the `IndexedDISI` cursor, advanced (or, for a target
    ///    behind the last one, rewound and re-advanced). Authoritative in both
    ///    directions: a `None` here means "this doc has no norm", not "ask
    ///    someone else", so an absent doc costs nothing extra. That is the
    ///    difference from the pre-c6 code, which fell through to
    ///    [`norms::norm_value`] and paid a second full region walk to be told
    ///    the same thing.
    /// 3. **Everything else** (a wider norm, a constant-valued field, an empty
    ///    one, or a sparse region whose bounds could not be resolved) --
    ///    [`norms::norm_value`], which owns the error cases.
    #[inline]
    fn norm_byte(&mut self, doc: i32) -> norms::Result<Option<u8>> {
        // The whole point of the table: for the ordinary dense one-byte field
        // this is a load and an index, as `BM25Scorer` reading `cache[norm]`
        // is. Anything else falls through.
        if let Some(bytes) = self.norms.dense_norm_bytes {
            if let Some(&b) = bytes.get(doc as usize) {
                return Ok(Some(b));
            }
        }
        if let Some(disi) = self.disi.as_mut() {
            // `advance_exact` asserts on a negative doc; `norm_value` owns that
            // error, and returning it here keeps the two paths' error identical.
            if doc < 0 {
                return Err(norms::Error::DocOutOfRange(
                    doc,
                    self.norms.entry.num_docs_with_field,
                ));
            }
            // Forward-only underneath; rewind rather than panic so a caller
            // whose doc order is not monotonic stays correct.
            if doc < disi.doc_id() {
                disi.reset();
            }
            return Ok(match disi.advance_exact(doc)? {
                Some(ordinal) => Some(norms::read_value_at_ordinal(
                    self.norms.data,
                    &self.norms.entry,
                    ordinal as i64,
                )? as u8),
                None => None,
            });
        }
        Ok(norms::norm_value(self.norms.data, &self.norms.entry, doc)?.map(|n| n as u8))
    }

    /// Whether `doc` has a norm for this field at all -- "would
    /// `LeafReader.getNormValues(field)`'s iterator land on this document".
    ///
    /// This is `FieldExistsQuery`'s norms source
    /// (`crate::doc_value_query::search_field_exists_norms`): Java answers it
    /// by advancing the very `NumericDocValues` this cursor *is*, so it is
    /// exactly [`Self::norm_byte`]'s `Option` with the value dropped, not a
    /// second decode path that could disagree with it. A dense field answers
    /// `true` for every doc in range; a sparse one consults its `IndexedDISI`.
    ///
    /// Distinct from [`Self::field_length`]/[`Self::norm_inverse`], which
    /// deliberately *substitute* `UNNORMED_FIELD_LENGTH` for an absent norm
    /// because a scorer must produce a number -- that substitution is what
    /// made "does this doc have a norm" unanswerable through this type before
    /// (c12 §5.3).
    #[inline]
    pub fn has_norm(&mut self, doc: i32) -> norms::Result<bool> {
        Ok(self.norm_byte(doc)?.is_some())
    }

    /// This doc's precomputed `1 / (k1 * ((1-b) + b*len/avgdl))`, for scoring
    /// via `weight - weight / (1 + freq * normInverse)` -- see
    /// [`FieldNorms`]'s `norm_inverse` table.
    #[inline]
    pub fn norm_inverse(&mut self, doc: i32) -> norms::Result<f32> {
        Ok(match self.norm_byte(doc)? {
            Some(b) => self.norms.norm_inverse[b as usize],
            // Same fallback as `field_length`: a doc the field legitimately
            // has no norm for is scored at the unnormed length.
            None => {
                1.0 / (crate::similarity::DEFAULT_K1
                    * ((1.0 - crate::similarity::DEFAULT_B)
                        + crate::similarity::DEFAULT_B * crate::similarity::UNNORMED_FIELD_LENGTH
                            / self.norms.avg_field_length))
            }
        })
    }

    /// This doc's real decoded field length, or
    /// [`crate::similarity::UNNORMED_FIELD_LENGTH`] when the doc legitimately
    /// has no norm.
    #[inline]
    pub fn field_length(&mut self, doc: i32) -> norms::Result<f32> {
        Ok(match self.norm_byte(doc)? {
            Some(b) => self.norms.norm_length[b as usize],
            None => crate::similarity::UNNORMED_FIELD_LENGTH,
        })
    }

    /// The field's `avgFieldLength`, so a scoring loop holding only a cursor
    /// does not also have to carry the `&FieldNorms` it came from.
    #[inline]
    pub fn avg_field_length(&self) -> f32 {
        self.norms.avg_field_length
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

    /// `IndexedDISI`'s "this entry carries no rank table", written as the byte
    /// `0xFF` (Java's `denseRankPower == -1`). Every sparse entry this port
    /// writes uses it, and it is the only value besides 7..=15 that
    /// `lucene_codecs::indexed_disi` accepts.
    ///
    /// These tests used to put a literal `0` here, which is *not* a legal
    /// `denseRankPower`: `dense_rank_bytes(0)` rejects it, so any DENSE
    /// `IndexedDISI` block reached with it fails to decode. Dense-norms entries
    /// never look at the field at all and a 4-document sparse block is SPARSE,
    /// so nothing in the old tests noticed -- but the fixtures were describing
    /// metadata no writer can produce, which is exactly the kind of
    /// almost-right test input that hides a real decode bug.
    const NO_RANK: u8 = 0xFF;

    fn dense_entry(bytes_per_norm: u8, num_docs: i32, norms_offset: i64) -> NormsEntry {
        NormsEntry {
            field_number: 0,
            docs_with_field_offset: -1, // dense
            docs_with_field_length: 0,
            jump_table_entry_count: 0,
            dense_rank_power: NO_RANK,
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

    /// The sparse fast path and the general `norms::norm_value` path must agree
    /// for every document, present and absent alike -- the same argument as the
    /// dense test above, for the branch that decodes the `IndexedDISI` list
    /// once instead of per lookup.
    #[test]
    fn sparse_fast_path_agrees_with_the_general_norm_lookup() {
        // Docs 0, 3, 6, 9 have a norm; 1, 2, 4, 5, 7, 8 do not.
        let present: Vec<i32> = (0..4).map(|i| i * 3).collect();
        let disi = lucene_codecs::indexed_disi::write(&present);
        let norms_bytes = [7u8, 19, 31, 43];
        let mut data = disi.clone();
        let norms_offset = data.len() as i64;
        data.extend_from_slice(&norms_bytes);

        let entry = NormsEntry {
            field_number: 0,
            docs_with_field_offset: 0,
            docs_with_field_length: disi.len() as i64,
            jump_table_entry_count: 0,
            dense_rank_power: NO_RANK,
            num_docs_with_field: present.len() as i32,
            bytes_per_norm: 1,
            norms_offset,
        };
        let n = FieldNorms::from_field_stats(&data, entry, 400, 10);
        assert!(
            n.sparse_region.is_some(),
            "this entry is exactly the shape the sparse path is for"
        );
        assert!(
            n.dense_norm_bytes.is_none(),
            "a sparse field must not take the dense path"
        );

        for doc in 0..10i32 {
            let general = match norms::norm_value(n.data, &n.entry, doc).unwrap() {
                Some(norm) => n.norm_inverse[(norm as u8) as usize],
                None => n.norm_inverse(doc).unwrap(), // absent: both fall back identically
            };
            assert_eq!(
                n.norm_inverse(doc).unwrap(),
                general,
                "fast and general paths disagree at doc {doc}"
            );
        }
        // And the values are the ones actually written, not a coincidence of
        // both paths being wrong the same way.
        assert_eq!(n.norm_inverse(0).unwrap(), n.norm_inverse[7]);
        assert_eq!(n.norm_inverse(9).unwrap(), n.norm_inverse[43]);
        assert_eq!(n.field_length(3).unwrap(), n.norm_length[19]);
    }

    /// The previous test's sparse field has 4 documents, so its `IndexedDISI`
    /// region is a single SPARSE block -- which never reads the rank table, and
    /// so never notices an illegal `dense_rank_power`. This one crosses the
    /// 4096-document threshold into a DENSE block, where the rank-power byte
    /// *is* consulted (`dense_rank_bytes`), so it pins that the entry's
    /// metadata is a value a real writer produces.
    ///
    /// This is also the shape `Lucene90NormsProducer` reads for a large, mostly
    /// -absent field: the norms values stay a flat ordinal-indexed array while
    /// the doc-id list goes dense.
    #[test]
    fn sparse_norms_over_a_dense_indexed_disi_block_agree_with_the_general_path() {
        // 5000 of the first 10000 docs have a norm: > 4095 in block 0, so
        // `indexed_disi::write` emits a DENSE block, not a SPARSE one.
        let present: Vec<i32> = (0..5000).map(|i| i * 2).collect();
        let disi = lucene_codecs::indexed_disi::write(&present);
        let norms_bytes: Vec<u8> = (0..present.len()).map(|i| (i % 251 + 1) as u8).collect();
        let mut data = disi.clone();
        let norms_offset = data.len() as i64;
        data.extend_from_slice(&norms_bytes);

        let entry = NormsEntry {
            field_number: 0,
            docs_with_field_offset: 0,
            docs_with_field_length: disi.len() as i64,
            jump_table_entry_count: 0,
            dense_rank_power: NO_RANK,
            num_docs_with_field: present.len() as i32,
            bytes_per_norm: 1,
            norms_offset,
        };
        let n = FieldNorms::from_field_stats(&data, entry, 250_000, 5000);
        assert!(
            n.sparse_region.is_some(),
            "a legal rank power must let the DENSE block decode"
        );
        assert_eq!(
            lucene_codecs::indexed_disi::decode_doc_ids(n.sparse_region.unwrap(), NO_RANK)
                .unwrap()
                .len(),
            present.len()
        );

        for doc in [0i32, 1, 2, 4321, 9998, 9999] {
            let general = norms::norm_value(n.data, &n.entry, doc).unwrap();
            match general {
                Some(norm) => {
                    assert_eq!(
                        n.norm_inverse(doc).unwrap(),
                        n.norm_inverse[norm as u8 as usize]
                    );
                    assert_eq!(
                        n.field_length(doc).unwrap(),
                        n.norm_length[norm as u8 as usize]
                    );
                }
                None => assert_eq!(
                    n.field_length(doc).unwrap(),
                    crate::similarity::UNNORMED_FIELD_LENGTH
                ),
            }
        }
    }

    /// The incremental cursor and the one-shot `&self` conveniences must agree
    /// for every document of a sparse field, present and absent alike, in
    /// ascending order (the cursor's fast path) *and* in descending order (its
    /// rewind path). Two implementations of one lookup; the only thing worth
    /// testing about the fast one is that it is not also different.
    ///
    /// This is the test that would have caught the c6 rewrite silently turning
    /// `FieldNorms`'s eager `Vec<i32>` of doc ids into a forward-only walk:
    /// a forward-only cursor that did *not* rewind would answer "absent" for
    /// every doc after the first backwards step.
    #[test]
    fn the_cursor_agrees_with_the_one_shot_lookup_forwards_and_backwards() {
        // 5000 of the first 10000 docs: a DENSE `IndexedDISI` block, so the
        // rank table and the word walk are both exercised.
        let present: Vec<i32> = (0..5000).map(|i| i * 2).collect();
        let disi = lucene_codecs::indexed_disi::write(&present);
        let norms_bytes: Vec<u8> = (0..present.len()).map(|i| (i % 251 + 1) as u8).collect();
        let mut data = disi.clone();
        let norms_offset = data.len() as i64;
        data.extend_from_slice(&norms_bytes);

        let entry = NormsEntry {
            field_number: 0,
            docs_with_field_offset: 0,
            docs_with_field_length: disi.len() as i64,
            jump_table_entry_count: 0,
            dense_rank_power: NO_RANK,
            num_docs_with_field: present.len() as i32,
            bytes_per_norm: 1,
            norms_offset,
        };
        let n = FieldNorms::from_field_stats(&data, entry, 250_000, 5000);

        let mut ascending = n.cursor();
        for doc in 0..10_000i32 {
            assert_eq!(
                ascending.norm_inverse(doc).unwrap(),
                n.norm_inverse(doc).unwrap(),
                "ascending cursor disagrees at doc {doc}"
            );
        }
        let mut descending = n.cursor();
        for doc in (0..10_000i32).rev() {
            assert_eq!(
                descending.field_length(doc).unwrap(),
                n.field_length(doc).unwrap(),
                "rewinding cursor disagrees at doc {doc}"
            );
        }
        // And an arbitrary order, with repeats -- a caller such as the
        // fuzzy-expansion loop restarts per expanded term, and asking the same
        // document twice must not consume it (the `Vec` this replaced was
        // trivially idempotent; a cursor is not obviously so).
        let mut jumping = n.cursor();
        for doc in [9998i32, 9998, 3, 3, 4321, 0, 9999, 9999, 2, 7777, 7777, 1] {
            assert_eq!(
                jumping.field_length(doc).unwrap(),
                n.field_length(doc).unwrap(),
                "jumping cursor disagrees at doc {doc}"
            );
        }
    }

    /// A dense field needs no `IndexedDISI` cursor at all, and a negative doc
    /// must be the same `DocOutOfRange` error on every path -- the sparse
    /// branch's `advance_exact` would otherwise assert (panic) on it.
    #[test]
    fn a_negative_doc_is_an_error_not_a_panic_on_either_shape() {
        let dense = FieldNorms::from_field_stats(&[5u8, 15, 25], dense_entry(1, 3, 0), 45, 3);
        assert!(dense.cursor().norm_inverse(-1).is_err());
        assert!(dense.cursor().field_length(-1).is_err());

        let present: Vec<i32> = (0..4).map(|i| i * 3).collect();
        let disi = lucene_codecs::indexed_disi::write(&present);
        let mut data = disi.clone();
        let norms_offset = data.len() as i64;
        data.extend_from_slice(&[7u8, 19, 31, 43]);
        let sparse = FieldNorms::from_field_stats(
            &data,
            NormsEntry {
                field_number: 0,
                docs_with_field_offset: 0,
                docs_with_field_length: disi.len() as i64,
                jump_table_entry_count: 0,
                dense_rank_power: NO_RANK,
                num_docs_with_field: present.len() as i32,
                bytes_per_norm: 1,
                norms_offset,
            },
            400,
            10,
        );
        assert!(sparse.sparse_region.is_some());
        assert!(sparse.cursor().norm_inverse(-1).is_err());
        assert!(sparse.cursor().field_length(-1).is_err());
    }

    /// `avg_field_length()` on the cursor is the field's, so a scoring loop
    /// holding only a cursor needs nothing else.
    #[test]
    fn the_cursor_carries_its_fields_avg_field_length() {
        let n = FieldNorms::from_field_stats(&[], dense_entry(1, 1, 0), 1000, 40);
        assert_eq!(n.cursor().avg_field_length(), 25.0);
    }

    /// The point of the cursor split: `FieldNorms` stays `Sync` shared state so
    /// `multi_segment.rs` can hand the same `&FieldNorms` to every `rayon`
    /// leaf, and the mutable `IndexedDISI` position lives in a per-task cursor.
    ///
    /// A `RefCell`-ed cursor inside `FieldNorms` -- the obvious smaller diff --
    /// would have made the type non-`Sync` and broken that fan-out, so this
    /// test proves the fan-out is still *actually concurrent*, not merely still
    /// compiling: two rayon tasks each take their own cursor over one shared
    /// `&FieldNorms` and neither may finish until both have started. If the
    /// work had been serialized, the first task would spin until the timeout
    /// and the assert would fire.
    #[test]
    fn a_shared_field_norms_still_fans_out_concurrently() {
        fn assert_sync<T: Sync>() {}
        fn assert_send<T: Send>() {}
        assert_sync::<FieldNorms<'_>>();
        assert_send::<FieldNormsCursor<'_, '_>>();

        if rayon::current_num_threads() < 2 {
            return; // single-threaded pool: nothing to prove
        }

        let present: Vec<i32> = (0..5000).map(|i| i * 2).collect();
        let disi = lucene_codecs::indexed_disi::write(&present);
        let mut data = disi.clone();
        let norms_offset = data.len() as i64;
        data.extend((0..present.len()).map(|i| (i % 251 + 1) as u8));
        let entry = NormsEntry {
            field_number: 0,
            docs_with_field_offset: 0,
            docs_with_field_length: disi.len() as i64,
            jump_table_entry_count: 0,
            dense_rank_power: NO_RANK,
            num_docs_with_field: present.len() as i32,
            bytes_per_norm: 1,
            norms_offset,
        };
        let norms = FieldNorms::from_field_stats(&data, entry, 250_000, 5000);

        use rayon::prelude::*;
        use std::sync::atomic::{AtomicUsize, Ordering};
        let started = AtomicUsize::new(0);
        let shared: &FieldNorms<'_> = &norms;

        let both_ran: Vec<bool> = (0..2)
            .into_par_iter()
            .map(|_| {
                // Each task takes its own cursor over the shared norms.
                let mut cursor = shared.cursor();
                let mut acc = 0.0f32;
                for &doc in &present {
                    acc += cursor.field_length(doc).unwrap();
                }
                assert!(acc > 0.0);

                started.fetch_add(1, Ordering::SeqCst);
                let deadline = std::time::Instant::now() + std::time::Duration::from_secs(5);
                while started.load(Ordering::SeqCst) < 2 {
                    if std::time::Instant::now() > deadline {
                        return false;
                    }
                    std::hint::spin_loop();
                }
                true
            })
            .collect();

        assert!(
            both_ran.iter().all(|&ok| ok),
            "the two leaves did not run concurrently -- sharing `&FieldNorms` \
             across rayon has been serialized"
        );
    }

    /// An illegal `dense_rank_power` (anything outside 7..=15 and `0xFF`) is
    /// metadata no Lucene writer emits, and must not be silently tolerated:
    /// the DISI decode rejects it, so the once-decoded fast path declines and
    /// the general path surfaces the error per lookup rather than inventing a
    /// norm.
    #[test]
    fn an_illegal_dense_rank_power_is_rejected_rather_than_guessed() {
        let present: Vec<i32> = (0..5000).map(|i| i * 2).collect();
        let disi = lucene_codecs::indexed_disi::write(&present);
        let mut data = disi.clone();
        let norms_offset = data.len() as i64;
        data.extend_from_slice(&vec![7u8; present.len()]);

        let entry = NormsEntry {
            field_number: 0,
            docs_with_field_offset: 0,
            docs_with_field_length: disi.len() as i64,
            jump_table_entry_count: 0,
            dense_rank_power: 0, // not 7..=15, not 0xFF
            num_docs_with_field: present.len() as i32,
            bytes_per_norm: 1,
            norms_offset,
        };
        let n = FieldNorms::from_field_stats(&data, entry, 250_000, 5000);
        assert!(
            n.sparse_region.is_some(),
            "the region's bounds are fine; it is the rank power that is not"
        );
        assert!(
            n.norm_inverse(0).is_err(),
            "an illegal rank power must surface as an error, not a fallback norm"
        );
        assert!(
            n.field_length(0).is_err(),
            "and on the length path too, not just the reciprocal one"
        );
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
