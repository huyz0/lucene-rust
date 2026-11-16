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
use lucene_codecs::simpletext::doc_values_reader::SimpleTextNumericDocValues;
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
    /// The default codec's whole `.nvd` file, matching [`norms::norm_value`]'s
    /// `data` parameter. Empty for SimpleText, whose cursor owns its source.
    pub data: &'a [u8],
    /// Default-codec metadata. SimpleText reads its private numeric cursor
    /// instead; this entry's binary offsets do not describe its text file.
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
    /// Only valid for [`crate::similarity::DEFAULT_K1`]/[`crate::similarity::DEFAULT_B`]; a caller
    /// using custom parameters must take the arithmetic path.
    norm_inverse: std::sync::Arc<[f32; 256]>,
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
    norm_length: &'static [f32; 256],
    /// A sparse field's `IndexedDISI` region, sliced (and therefore
    /// bounds-checked) once. `None` when the field is dense, empty, or the
    /// recorded region does not lie inside `data` -- in which case the general
    /// [`norms::norm_value`] path re-raises that error per lookup.
    ///
    /// No doc id list is decoded: [`FieldNorms::cursor`] hands a per-scorer
    /// [`FieldNormsCursor`] that walks the region forward once per *scan*
    /// instead of once per lookup.
    sparse_region: Option<&'a [u8]>,
    /// SimpleText's numeric norms, cloned into each scorer's cursor. Values
    /// stay in the text file and are decoded by its fallible positioning path.
    text: Option<SimpleTextNumericDocValues<'a>>,
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
    norms: std::borrow::Cow<'n, FieldNorms<'a>>,
    /// `Some` only for a sparse field; a dense or empty one never needs it.
    disi: Option<lucene_codecs::indexed_disi::DisiCursor<'a>>,
    text: Option<SimpleTextNumericDocValues<'a>>,
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

/// [`norm_length_table`], built once for the process: it depends on nothing.
static NORM_LENGTH: std::sync::LazyLock<[f32; 256]> = std::sync::LazyLock::new(norm_length_table);

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

/// [`FieldNorms::norm_inverse`] for one average field length, shareable: every
/// segment of a reader scores a field against the same reader-wide average, so
/// a reader builds the table once per average ([`crate::directory_reader::DirectoryReader`]
/// caches it) instead of once per segment per search.
pub fn inverse_table(avg_field_length: f32) -> std::sync::Arc<[f32; 256]> {
    std::sync::Arc::new(norm_inverse_table(avg_field_length))
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
    /// enough to reorder documents at the top-k boundary. the benchmark
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

    /// A field that is indexed but has **no norms** (`FieldInfo.omitNorms`),
    /// scored as Java scores it: `LeafSimScorer` reads a missing norm as `1`
    /// (`LENGTH_TABLE[1]`, a field length of one), but `BM25Similarity`
    /// still divides by the collection's real `avgFieldLength`
    /// (`sumTotalTermFreq / docCount`, with `sumTotalTermFreq` equal to
    /// `sumDocFreq` for a `DOCS` field). Substituting `1` for *both* lengths,
    /// which a `None` norms argument still does, agrees with Java only when
    /// the field averages one token per document -- `VerifyIndex` caught
    /// it scoring a freqs-only field 12% low.
    pub fn unnormed(max_doc: i32, avg_field_length: f32) -> FieldNorms<'static> {
        FieldNorms::unnormed_with_table(max_doc, avg_field_length, inverse_table(avg_field_length))
    }

    /// [`FieldNorms::unnormed`] with its `norm_inverse` table already built.
    pub fn unnormed_with_table(
        max_doc: i32,
        avg_field_length: f32,
        table: std::sync::Arc<[f32; 256]>,
    ) -> FieldNorms<'static> {
        FieldNorms::with_inverse_table(
            &[],
            NormsEntry::constant(-1, max_doc, 1),
            avg_field_length,
            table,
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
        Self::with_inverse_table(
            data,
            entry,
            avg_field_length,
            inverse_table(avg_field_length),
        )
    }

    /// [`FieldNorms::with_avg_field_length`] with its `norm_inverse` table
    /// already built -- [`inverse_table`] of the same `avg_field_length`.
    pub fn with_inverse_table(
        data: &'a [u8],
        entry: NormsEntry,
        avg_field_length: f32,
        table: std::sync::Arc<[f32; 256]>,
    ) -> Self {
        Self {
            dense_norm_bytes: dense_norm_bytes(data, &entry),
            data,
            entry,
            avg_field_length,
            norm_inverse: table,
            norm_length: &NORM_LENGTH,
            sparse_region: sparse_region(data, &entry),
            text: None,
        }
    }

    /// Java: `TermScorer.score` takes the codec's `NumericDocValues`, including
    /// `SimpleTextNormsFormat`'s. Keep its full long values for custom
    /// similarities; only BM25 narrows them to a byte for its lookup table.
    pub(crate) fn from_simpletext(
        values: SimpleTextNumericDocValues<'a>,
        avg_field_length: f32,
        table: std::sync::Arc<[f32; 256]>,
    ) -> Self {
        Self {
            data: &[],
            // The binary metadata is unused when `text` is present.
            entry: NormsEntry::constant(-1, values.cost() as i32, 0),
            avg_field_length,
            norm_inverse: table,
            dense_norm_bytes: None,
            norm_length: &NORM_LENGTH,
            sparse_region: None,
            text: Some(values),
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
            norm_inverse: std::sync::Arc::new([0.0; 256]),
            norm_length: &NORM_LENGTH,
            sparse_region: sparse_region(data, &entry),
            text: None,
        };
        let mut sum = 0.0f64;
        let mut count = 0i64;
        {
            let mut cursor = scratch.cursor();
            for doc in 0..max_doc {
                if !live_docs.is_none_or(|bits| bits.get_doc(doc)) {
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
        scratch.norm_inverse = inverse_table(avg_field_length);
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
            norms: std::borrow::Cow::Borrowed(self),
            disi: self.sparse_region.map(|region| {
                lucene_codecs::indexed_disi::DisiCursor::new(
                    region,
                    self.entry.dense_rank_power,
                    self.entry.jump_table_entry_count,
                )
            }),
            text: self.text.clone(),
        }
    }

    /// [`Self::cursor`] owning these norms, for a scorer that outlives the
    /// map they were built into.
    pub fn into_cursor<'n>(self) -> FieldNormsCursor<'n, 'a> {
        let disi = self.sparse_region.map(|region| {
            lucene_codecs::indexed_disi::DisiCursor::new(
                region,
                self.entry.dense_rank_power,
                self.entry.jump_table_entry_count,
            )
        });
        let text = self.text.clone();
        FieldNormsCursor {
            norms: std::borrow::Cow::Owned(self),
            disi,
            text,
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

impl<'a> FieldNormsCursor<'_, 'a> {
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
        if self.text.is_some() {
            return Ok(self.norm_long(doc)?.map(|value| value as u8));
        }
        // The whole point of the table: for the ordinary dense one-byte field
        // this is a load and an index, as `BM25Scorer` reading `cache[norm]`
        // is. Anything else falls through.
        if let Some(bytes) = self.norms.dense_norm_bytes {
            if let Some(&b) = bytes.get(doc as usize) {
                return Ok(Some(b));
            }
        }
        // A constant-valued dense field -- every field without norms
        // ([`FieldNorms::unnormed`]) -- needs no lookup either.
        let entry = &self.norms.entry;
        if entry.bytes_per_norm == 0
            && entry.is_dense()
            && (0..entry.num_docs_with_field).contains(&doc)
        {
            return Ok(Some(entry.norms_offset as u8));
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

    /// The field's norms as one byte per document, indexed by document, when
    /// it is the dense one-byte shape every ordinary analyzed field has --
    /// what [`Self::norm_long`] reads first (sign-extended). `None` for any
    /// other shape, which only [`Self::norm_long`] answers.
    #[inline]
    pub fn dense_bytes(&self) -> Option<&'a [u8]> {
        self.norms.dense_norm_bytes
    }

    /// This doc's norm as `NumericDocValues.longValue()` reads it -- a
    /// stored byte sign-extended, a wider norm as stored -- or `None` when the
    /// field has no norm for it: what a `SimScorer` other than the default
    /// BM25 fast path scores with ([`crate::similarities::SimScorer::score`]).
    ///
    /// [`Self::norm_byte`]'s three shapes, kept separate from it so the BM25
    /// path's lookup stays exactly what it was.
    pub fn norm_long(&mut self, doc: i32) -> norms::Result<Option<i64>> {
        if let Some(values) = &mut self.text {
            if doc < 0 || i64::from(doc) >= values.cost() {
                return Err(norms::Error::DocOutOfRange(doc, values.cost() as i32));
            }
            let value = (|| {
                if values.advance_exact(doc)? {
                    values.long_value().map(Some)
                } else {
                    Ok(None)
                }
            })();
            return value.map_err(|error| {
                norms::Error::Store(lucene_store::Error::Corrupted(format!(
                    "SimpleText norms: {error}"
                )))
            });
        }
        if let Some(bytes) = self.norms.dense_norm_bytes {
            if let Some(&b) = bytes.get(doc as usize) {
                return Ok(Some(i64::from(b as i8)));
            }
        }
        let entry = &self.norms.entry;
        if entry.bytes_per_norm == 0
            && entry.is_dense()
            && (0..entry.num_docs_with_field).contains(&doc)
        {
            return Ok(Some(entry.norms_offset));
        }
        if let Some(disi) = self.disi.as_mut() {
            if doc < 0 {
                return Err(norms::Error::DocOutOfRange(
                    doc,
                    self.norms.entry.num_docs_with_field,
                ));
            }
            if doc < disi.doc_id() {
                disi.reset();
            }
            return Ok(match disi.advance_exact(doc)? {
                Some(ordinal) => Some(norms::read_value_at_ordinal(
                    self.norms.data,
                    &self.norms.entry,
                    ordinal as i64,
                )?),
                None => None,
            });
        }
        norms::norm_value(self.norms.data, &self.norms.entry, doc)
    }

    /// [`Self::norm_long`] for an ascending batch, a missing norm read as `1`
    /// (`TermScorer`'s `norms.advanceExact(doc) == false` fallback, and the
    /// value a field without norms scores with).
    pub fn norm_long_batch(&mut self, docs: &[i32], out: &mut Vec<i64>) -> norms::Result<()> {
        out.clear();
        if let (Some(bytes), Some(&first), Some(&last)) =
            (self.norms.dense_norm_bytes, docs.first(), docs.last())
        {
            if first >= 0 && (last as usize) < bytes.len() {
                out.extend(docs.iter().map(|&d| i64::from(bytes[d as usize] as i8)));
                return Ok(());
            }
        }
        out.reserve(docs.len());
        for &d in docs {
            out.push(self.norm_long(d)?.unwrap_or(1));
        }
        Ok(())
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
    /// because a scorer must produce a number.
    #[inline]
    pub fn has_norm(&mut self, doc: i32) -> norms::Result<bool> {
        Ok(self.norm_byte(doc)?.is_some())
    }

    /// This doc's precomputed `1 / (k1 * ((1-b) + b*len/avgdl))`, for scoring
    /// via `weight - weight / (1 + freq * normInverse)` -- see
    /// [`FieldNorms`]'s `norm_inverse` table.
    #[inline]
    pub fn norm_inverse(&mut self, doc: i32) -> norms::Result<f32> {
        // The dense one-byte field, inline: a scorer calls this once per
        // scored document, and `norm_byte`'s other cases stay out of line.
        if let (None, Some(bytes)) = (&self.text, self.norms.dense_norm_bytes) {
            if let Some(&b) = bytes.get(doc as usize) {
                return Ok(self.norms.norm_inverse[b as usize]);
            }
        }
        self.norm_inverse_slow(doc)
    }

    #[inline(never)]
    fn norm_inverse_slow(&mut self, doc: i32) -> norms::Result<f32> {
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

    /// [`Self::norm_inverse`] for a whole ascending batch of documents --
    /// `NumericDocValues.longValues(size, docs, values, default)` feeding
    /// `BM25Scorer`'s bulk `score`.
    ///
    /// The dense one-byte field -- every ordinary analyzed field -- is a
    /// straight gather through the 256-entry table with the range established
    /// once for the whole batch (the documents are ascending, so the last one
    /// bounds them all); anything else takes the per-document path.
    pub fn norm_inverse_batch(&mut self, docs: &[i32], out: &mut Vec<f32>) -> norms::Result<()> {
        out.clear();
        if let (Some(bytes), Some(&first), Some(&last)) =
            (self.norms.dense_norm_bytes, docs.first(), docs.last())
        {
            if first >= 0 && (last as usize) < bytes.len() {
                let table = &self.norms.norm_inverse;
                out.extend(docs.iter().map(|&d| table[bytes[d as usize] as usize]));
                return Ok(());
            }
        }
        // A constant-valued dense field -- every field without norms, a
        // keyword's -- is one value for the whole batch, as `norm_byte`
        // answers each document of it.
        let entry = &self.norms.entry;
        if let (Some(&first), Some(&last)) = (docs.first(), docs.last()) {
            if self.text.is_none()
                && entry.bytes_per_norm == 0
                && entry.is_dense()
                && first >= 0
                && last < entry.num_docs_with_field
            {
                let v = self.norms.norm_inverse[entry.norms_offset as u8 as usize];
                out.resize(docs.len(), v);
                return Ok(());
            }
        }
        out.reserve(docs.len());
        for &d in docs {
            out.push(self.norm_inverse(d)?);
        }
        Ok(())
    }

    /// `normInverse` for each of the 256 norm bytes: the table every document
    /// of this field is scored with.
    pub fn norm_inverse_table(&self) -> &[f32; 256] {
        &self.norms.norm_inverse
    }

    /// The largest value [`Self::norm_inverse`] can return for this field --
    /// the shortest possible document, or a document with no norm at all --
    /// which bounds every score from above the way `MaxScoreCache`'s global
    /// maximum does.
    pub fn max_norm_inverse(&self) -> f32 {
        let unnormed = 1.0
            / (crate::similarity::DEFAULT_K1
                * ((1.0 - crate::similarity::DEFAULT_B)
                    + crate::similarity::DEFAULT_B * crate::similarity::UNNORMED_FIELD_LENGTH
                        / self.norms.avg_field_length));
        self.norms
            .norm_inverse
            .iter()
            .copied()
            .fold(unnormed, f32::max)
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
mod tests;
