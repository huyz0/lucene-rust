//! Write side of the term dictionary and postings -- `.doc`/`.pos`/`.pay`/
//! `.psm` (`Lucene104PostingsWriter`) and `.tim`/`.tip`/`.tmd`
//! (`Lucene103BlockTreeTermsWriter`) -- for any number of fields per call.
//!
//! Correctness rests on two independent checks, not on this port's own reader
//! agreeing with it: `scripts/verify-write-path.sh` has real Lucene 10.5.0 open
//! segments written through `IndexWriter` and walk them (`VerifyFullSegment`,
//! `VerifyMergedSegment`, `VerifySortedSegment`, `VerifyPositionsSegment`), and
//! `VerifyIndex` runs 57 queries over a 120 000-document, seven-segment
//! Rust-written index and requires Lucene's top 50 to match this port's, with
//! its own `CheckIndex` clean. The unit tests below round-trip through
//! `crate::blocktree`/`crate::postings`, which are themselves verified against
//! Lucene-written fixtures.
//!
//! # What is written
//!
//! - **Term dictionary**: real block-tree blocks -- a field's terms are split
//!   into `.tim` blocks of 25..48 entries, with floor blocks and non-leaf
//!   blocks pointing at sub-blocks, under a multi-level `.tip` trie; see
//!   `crate::blocktree_writer`, which ports the splitting and `TrieBuilder`.
//!   Per-term stats run-length encode singleton terms as `StatsWriter` does.
//! - **`.doc`**: every complete 256-document chunk as a full block behind a
//!   level-0 skip header with competitive impacts, in whichever of Java's three
//!   doc-delta encodings it picks (packed `ForUtil`, the `bitsPerValue == 0`
//!   all-consecutive marker, or a `bitsPerValue < 0` bit set), a level-1 skip
//!   entry ahead of every 32 blocks, and a group-varint tail block for the
//!   remainder. Impacts come from a port of `CompetitiveImpactAccumulator` when
//!   the caller supplies norms ([`write_fields_with_norms`]); without norms the
//!   frontier is Java's `fieldHasNorms == false` one, `(maxFreq, 1)`.
//! - **`.pos`/`.pay`**: full 256-occurrence `PForUtil` blocks (positions,
//!   offset start deltas and lengths, payload lengths plus payload bytes) and a
//!   vint tail, with the `.pos`/`.pay` pointers every `.doc` skip entry carries
//!   ([`PosSkipWriter`]).
//! - **`docFreq == 1` is pulsed into the term dictionary**: no `.doc` bytes for
//!   a singleton term.
//!
//! # Byte identity
//!
//! `.tim` suffixes are compressed (`LZ4`, `LOWERCASE_ASCII`) exactly when
//! Java compresses them, and term metadata takes `encodeTerm`'s zigzag
//! singleton-doc-delta branch exactly when Java does
//! ([`write_term_metadata`]); `tests/blocktree_byte_identity_fixture.rs`
//! holds both to Lucene's bytes.
//!
//! # Caller obligations (checked where cheap)
//!
//! `terms` sorted ascending by bytes with no duplicates; each term's `docs`
//! sorted ascending by doc ID, no duplicates, every `freq >= 1`; positions,
//! offsets and payloads shaped to match the field's `IndexOptions`. The
//! structural checks are in `validate_field`.
//!
//! # Wire format
//!
//! Mirrors `crate::blocktree`/`crate::postings`'s module docs, writer side:
//! each file is `IndexHeader`, body, `Footer`; `.tmd` embeds the postings
//! writer's own header (`Lucene104PostingsWriterTerms`, `indexBlockSize =
//! 256`), then `numFields` and per field `fieldNumber, numTerms,
//! [sumTotalTermFreq,] sumDocFreq, docCount, minTerm, maxTerm, indexStart,
//! rootFP, indexEnd`, then `indexLength` and `termsLength`.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_output::DataOutput;
use lucene_store::index_output::{FsIndexOutput, IndexOutput};

use crate::blocktree::{
    POSTINGS_BLOCK_SIZE, POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT, TERMS_CODEC_NAME,
    TERMS_INDEX_CODEC_NAME, TERMS_META_CODEC_NAME, VERSION_CURRENT as BLOCKTREE_VERSION_CURRENT,
    ZSTD_TERMS_CODEC_NAME, ZSTD_TERMS_INDEX_CODEC_NAME, ZSTD_TERMS_META_CODEC_NAME,
};
use crate::blocktree_writer;
use crate::field_infos::IndexOptions;
use crate::for_util;
use crate::postings::{
    BLOCK_SIZE, DOC_CODEC, LEVEL1_NUM_DOCS, META_CODEC, PAY_CODEC, POS_CODEC,
    VERSION_CURRENT as DOC_VERSION_CURRENT,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// Native installed consumer output acquisition, writing or close failure.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("write_single_field: terms must be non-empty")]
    EmptyTerms,
    /// A terms dictionary other than the block tree failed to write a field.
    #[error("terms dictionary: {0}")]
    Dictionary(String),
    /// A [`StreamingFieldsWriter`] call out of order (a field opened twice,
    /// terms or a close with no field open, a finish with one still open).
    #[error("streaming postings writer: {0}")]
    StreamingOrder(&'static str),
    /// `Lucene103BlockTreeTermsWriter.validateSettings`'s
    /// `IllegalArgumentException` ([`WriteOptions::validate`]).
    #[error("{0}")]
    InvalidBlockSizes(String),
    #[error("write_single_field: terms out of order or duplicated at index {0}")]
    TermsNotSorted(usize),
    #[error("write_single_field: term at index {0} has no postings (docFreq == 0)")]
    EmptyPostings(usize),
    #[error("write_single_field: term at index {index} has non-ascending/duplicate doc IDs")]
    DocIdsNotSorted { index: usize },
    /// `Lucene104PostingsWriter.startDoc`'s
    /// `if (docID < 0 || docDelta <= 0) throw new CorruptIndexException("docs
    /// out of order (...)")` -- a real production check, not an assertion.
    /// Without it a negative doc ID reaches the `docID - lastDocID` delta as
    /// an unbounded subtraction and lands in the `.doc` stream as a delta no
    /// reader can undo.
    #[error("write_single_field: term at index {index} has a negative doc ID")]
    NegativeDocId { index: usize },
    /// `Lucene104PostingsWriter.addPosition`'s
    /// `if (position < 0) throw new CorruptIndexException("position=... is <
    /// 0")` -- again a production check. Positions are written as deltas from
    /// a zero base, so a negative one both overflows that subtraction and
    /// decodes back as a negative position.
    #[error(
        "write_single_field: term at index {index}, doc index {doc_index} has a negative position"
    )]
    NegativePosition { index: usize, doc_index: usize },
    #[error("write_single_field: term at index {index} has freq < 1")]
    NonPositiveFreq { index: usize },
    #[error(
        "write_single_field: only IndexOptions::Docs/DocsAndFreqs/DocsAndFreqsAndPositions/\
         DocsAndFreqsAndPositionsAndOffsets/DocsAndCustomFreqs is supported, got {0:?}"
    )]
    UnsupportedIndexOptions(IndexOptions),
    #[error(
        "write_single_field: term at index {index}, doc index {doc_index} has {positions} \
         position(s) but freq {freq}; they must match when index_options indexes positions"
    )]
    PositionsFreqMismatch {
        index: usize,
        doc_index: usize,
        positions: usize,
        freq: i32,
    },
    #[error(
        "write_single_field: term at index {index}, doc index {doc_index} has no positions but \
         index_options indexes positions; every doc needs exactly `freq` positions"
    )]
    MissingPositions { index: usize, doc_index: usize },
    #[error(
        "write_single_field: term at index {index}, doc index {doc_index} has decreasing \
         positions -- positions must not decrease within a doc"
    )]
    PositionsNotAscending { index: usize, doc_index: usize },
    #[error(
        "write_single_field: term at index {index}, doc index {doc_index} has no offsets but \
         index_options indexes offsets; every doc needs exactly `freq` (start, end) offset pairs"
    )]
    MissingOffsets { index: usize, doc_index: usize },
    #[error(
        "write_single_field: term at index {index}, doc index {doc_index} has {offsets} offset \
         pair(s) but freq {freq}; they must match when index_options indexes offsets"
    )]
    OffsetsFreqMismatch {
        index: usize,
        doc_index: usize,
        offsets: usize,
        freq: i32,
    },
    #[error(
        "write_single_field: term at index {index}, doc index {doc_index}, occurrence \
         {occurrence} has an invalid offset pair (startOffset must be >= the previous \
         occurrence's startOffset in the same doc, or >= 0 for the first occurrence, and \
         endOffset must be >= startOffset)"
    )]
    InvalidOffsets {
        index: usize,
        doc_index: usize,
        occurrence: usize,
    },
    #[error(
        "write_single_field: term at index {index} has {payload_lengths} payload lengths but \
         {total_term_freq} occurrences; when has_payloads is set the flat payload run needs \
         exactly one length per occurrence (each possibly zero)"
    )]
    PayloadsFreqMismatch {
        index: usize,
        payload_lengths: usize,
        total_term_freq: usize,
    },
    #[error(
        "write_single_field: term at index {index} has {payload_bytes} payload bytes but its \
         lengths sum to {expected}; the flat payload run must be exactly the occurrences' \
         payloads concatenated"
    )]
    PayloadBytesMismatch {
        index: usize,
        payload_bytes: usize,
        expected: usize,
    },
    #[error(
        "write_single_field: term at index {index}, occurrence {occurrence} has payload length \
         {length}, which does not fit the `int` the wire format writes it as"
    )]
    PayloadLengthTooLarge {
        index: usize,
        occurrence: usize,
        length: u32,
    },
    /// `CompletionFieldsConsumer.write`: a completion field's payload that is
    /// not a `SuggestField` payload, mixed suggestion types in one field, or
    /// a weight out of range.
    #[error("completion postings: {0}")]
    Completion(#[source] lucene_store::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// One term's postings: `docs` is `(doc_id, freq)` pairs, ascending doc-ID
/// order, no duplicates, every `freq >= 1` (see the module doc's "Caller
/// obligations").
///
/// `positions` carries per-occurrence position data and is only consulted
/// when [`FieldPostingsInput::index_options`] is
/// `IndexOptions::DocsAndFreqsAndPositions`; leave it `Vec::new()` for
/// `Docs`/`DocsAndFreqs` fields. When positions are required, `positions`
/// must have exactly `docs.len()` entries in the same doc order, and
/// `positions[i].len()` must equal `docs[i].1` (that doc's `freq`) —
/// `write_single_field` validates both. Each `positions[i]` entry is a doc's
/// *absolute*, ascending (Lucene positions never repeat or go backwards
/// within a doc) per-occurrence position sequence, e.g. `[0, 3, 4]` for a
/// term occurring at token positions 0, 3, and 4 in that doc; the writer
/// derives the on-wire deltas itself (position deltas reset to the absolute
/// first position at each doc's first occurrence, exactly like real
/// Lucene's `Lucene104PostingsWriter.startDoc`/`addPosition`).
/// `offsets` mirrors `positions`: only consulted when
/// [`FieldPostingsInput::index_options`] is
/// `IndexOptions::DocsAndFreqsAndPositionsAndOffsets`, in which case it must
/// have exactly `docs.len()` entries (same doc order as `positions`) and
/// `offsets[i].len()` must equal `positions[i].len()` (== that doc's
/// `freq`). Each entry is an occurrence's absolute `(startOffset,
/// endOffset)` pair; per real Lucene's `addPosition` assertions
/// (`Lucene104PostingsWriter.java:332-333`), `endOffset >= startOffset` and,
/// within one doc, `startOffset` never decreases from one occurrence to the
/// next (it resets to comparing against `0` at each doc's first
/// occurrence) — the writer derives the on-wire
/// `startOffset - lastStartOffset` delta itself, exactly like `positions`.
/// `payload_bytes`/`payload_lengths` are this term's payloads, **flat**:
/// only consulted when [`FieldPostingsInput::has_payloads`] is set, in which
/// case `payload_lengths` must have exactly one entry per occurrence (i.e.
/// `docs.iter().map(|&(_, freq)| freq).sum()` of them, in doc order and then
/// occurrence order within each doc) and `payload_bytes` must be exactly
/// those occurrences' payloads concatenated in the same order. A zero length
/// means "no payload for this occurrence" (real Lucene's `addPosition` treats
/// `payload == null` and `payload.length == 0` identically,
/// `Lucene104PostingsWriter.java:316-319`), exactly as valid as a non-empty
/// payload; payload *presence* is a per-field property
/// (`FieldInfo.hasPayloads()`), never a per-occurrence one, so there is no
/// "absent" state to model beyond zero-length.
///
/// Flat rather than nested `Vec<Vec<Vec<u8>>>`, because a nested shape costs a
/// heap object per occurrence *and* a vector header per posting entry. This is
/// also the exact layout [`write_position_tail`] builds internally and
/// `Lucene104PostingsWriter` accumulates into its own `payloadBytes` /
/// `payloadLengthBuffer`, so the writer borrows the caller's run instead of
/// re-flattening it.
#[derive(Debug, Clone, Default)]
pub struct TermPostings {
    pub term: Vec<u8>,
    pub docs: Vec<(i32, i32)>,
    pub positions: Vec<Vec<i32>>,
    pub offsets: Vec<Vec<(i32, i32)>>,
    pub payload_bytes: Vec<u8>,
    pub payload_lengths: Vec<u32>,
}

/// One term's postings as the writer reads them: what a [`TermPostings`]
/// holds, borrowed from wherever the caller keeps it. A flush's buffered
/// postings keep each term's positions in one run across its documents, the
/// way `FreqProxTermsWriterPerField` keeps them in its byte slices; reading
/// that run in place spares the flush a copy of every term.
pub trait TermSource {
    /// The term's bytes.
    fn term(&self) -> &[u8];
    /// `(doc, freq)` per document, doc ascending.
    fn docs(&self) -> &[(i32, i32)];
    /// How many documents carry a positions list: `docs().len()` when the
    /// field indexes positions.
    fn position_lists(&self) -> usize;
    /// Document `doc`'s positions, `doc < position_lists()`.
    fn positions(&self, doc: usize) -> &[i32];
    /// How many documents carry an offsets list: `docs().len()` when the
    /// field indexes offsets.
    fn offset_lists(&self) -> usize;
    /// Document `doc`'s offsets, `doc < offset_lists()`.
    fn offsets(&self, doc: usize) -> &[(i32, i32)];
    /// See [`TermPostings::payload_bytes`].
    fn payload_bytes(&self) -> &[u8];
    /// See [`TermPostings::payload_bytes`].
    fn payload_lengths(&self) -> &[u32];
}

impl<T: TermSource + ?Sized> TermSource for &T {
    fn term(&self) -> &[u8] {
        (**self).term()
    }
    fn docs(&self) -> &[(i32, i32)] {
        (**self).docs()
    }
    fn position_lists(&self) -> usize {
        (**self).position_lists()
    }
    fn positions(&self, doc: usize) -> &[i32] {
        (**self).positions(doc)
    }
    fn offset_lists(&self) -> usize {
        (**self).offset_lists()
    }
    fn offsets(&self, doc: usize) -> &[(i32, i32)] {
        (**self).offsets(doc)
    }
    fn payload_bytes(&self) -> &[u8] {
        (**self).payload_bytes()
    }
    fn payload_lengths(&self) -> &[u32] {
        (**self).payload_lengths()
    }
}

impl TermSource for TermPostings {
    fn term(&self) -> &[u8] {
        &self.term
    }
    fn docs(&self) -> &[(i32, i32)] {
        &self.docs
    }
    fn position_lists(&self) -> usize {
        self.positions.len()
    }
    fn positions(&self, doc: usize) -> &[i32] {
        self.positions.get(doc).map_or(&[], Vec::as_slice)
    }
    fn offset_lists(&self) -> usize {
        self.offsets.len()
    }
    fn offsets(&self, doc: usize) -> &[(i32, i32)] {
        self.offsets.get(doc).map_or(&[], Vec::as_slice)
    }
    fn payload_bytes(&self) -> &[u8] {
        &self.payload_bytes
    }
    fn payload_lengths(&self) -> &[u32] {
        &self.payload_lengths
    }
}

/// Reorders a [`TermPostings`] flat payload run to follow a permutation of
/// its documents — the payload half of a caller that has just re-ordered
/// `docs`/`positions`/`offsets` (a sorted merge interleaving its sources, or
/// an invert pass enforcing doc-ID order).
///
/// The run is doc-major with no per-document index of its own — one length
/// per occurrence and every payload concatenated, exactly as
/// `Lucene104PostingsWriter` accumulates its own `payloadBytes` — so `counts`
/// (each document's occurrence count, **in the pre-permutation order**) is
/// what turns it back into per-document spans. `permutation[n]` is the index
/// of the document that becomes position `n`, the same direction
/// `crate::doc_values::…` and `lucene_index::merge`'s own list reorder use.
///
/// A run shorter than `counts` describes says the caller built the two out of
/// step; this saturates rather than panicking, because
/// [`Error::PayloadsFreqMismatch`] is where that is meant to be reported and
/// a slice panic here would come first.
pub fn permute_payload_run(
    payload_bytes: &[u8],
    payload_lengths: &[u32],
    counts: &[u32],
    permutation: &[usize],
) -> (Vec<u8>, Vec<u32>) {
    debug_assert_eq!(counts.len(), permutation.len());
    // (byte offset, length offset) of each document's run, before the sort.
    let mut spans: Vec<(usize, usize)> = Vec::with_capacity(counts.len());
    let (mut byte_at, mut len_at) = (0usize, 0usize);
    for &count in counts {
        spans.push((byte_at, len_at));
        let end = len_at.saturating_add(count as usize);
        for &length in payload_lengths.get(len_at..end).unwrap_or_default() {
            byte_at = byte_at.saturating_add(length as usize);
        }
        len_at = end;
    }
    let mut bytes = Vec::with_capacity(payload_bytes.len());
    let mut lengths = Vec::with_capacity(payload_lengths.len());
    for &i in permutation {
        let Some(&(byte_start, len_start)) = spans.get(i) else {
            continue;
        };
        let count = counts.get(i).copied().unwrap_or(0) as usize;
        let run = payload_lengths
            .get(len_start..len_start.saturating_add(count))
            .unwrap_or_default();
        let run_bytes: usize = run.iter().map(|&l| l as usize).sum();
        lengths.extend_from_slice(run);
        bytes.extend_from_slice(
            payload_bytes
                .get(byte_start..byte_start.saturating_add(run_bytes))
                .unwrap_or_default(),
        );
    }
    (bytes, lengths)
}

/// Port of `org.apache.lucene.codecs.CompetitiveImpactAccumulator`: the set
/// of `(freq, norm)` pairs from a block (or a level-1 span) of documents that
/// can possibly produce a competitive score.
///
/// The set is the **Pareto frontier** of the block's `(freq, norm)` pairs
/// under "higher freq scores higher, higher norm scores lower": an entry is
/// kept only if no other entry has both a higher-or-equal freq and a
/// lower-or-equal norm. `get_competitive_freq_norm_pairs` returns it ordered
/// by strictly increasing freq *and* strictly increasing (unsigned) norm,
/// which is the invariant both [`write_impacts`] and
/// `crate::postings::decode_impacts` rely on.
///
/// Why the frontier and not just `(maxFreq, minNorm)`: the pair
/// `(maxFreq, minNorm)` need not occur in any single document. Emitting it
/// would still be *sound* (it out-scores every real document), but it is
/// looser than what Lucene writes, and the point of this type is to make the
/// bound tight enough to prune with.
///
/// Structure follows Java's exactly: a 256-entry `max_freqs` table for norms
/// in `-128..=127` (every norm the default similarity produces, since
/// `Lucene90NormsConsumer` writes those as a single byte), plus a sorted
/// overflow list for anything wider. The overflow list is a sorted `Vec`
/// rather than Java's `TreeSet` because it holds at most a handful of
/// entries in the only case that reaches it -- a segment whose `.nvd` stores
/// more than one byte per norm -- and a `Vec` keeps the ordering invariant
/// visible at the one place that maintains it.
#[derive(Debug, Clone)]
pub(crate) struct CompetitiveImpactAccumulator {
    /// `maxFreqs[Byte.toUnsignedInt((byte) norm)]`: the highest freq seen for
    /// each byte-valued norm. Index `i` stands for norm `i as u8 as i8`.
    max_freqs: [i32; 256],
    /// `otherFreqNormPairs`, kept sorted by Java's comparator: increasing
    /// freq, and for equal freqs *decreasing* unsigned norm.
    other: Vec<Impact>,
}

/// One competitive `(freq, norm)` pair -- Java's
/// `org.apache.lucene.codecs.Impact`.
///
/// The read side's [`crate::postings::Impact`] *is* that type, so this is a
/// local name for it rather than a second definition: the write side has to
/// produce exactly what `decode_impacts` reads back, and two structs would be
/// two chances to disagree.
type Impact = crate::postings::Impact;

impl Default for CompetitiveImpactAccumulator {
    fn default() -> Self {
        Self {
            max_freqs: [0i32; 256],
            other: Vec::new(),
        }
    }
}

impl CompetitiveImpactAccumulator {
    /// `CompetitiveImpactAccumulator.clear()`.
    pub(crate) fn clear(&mut self) {
        self.max_freqs = [0i32; 256];
        self.other.clear();
    }

    /// `CompetitiveImpactAccumulator.addAll(acc)`: the frontier of the union.
    pub(crate) fn add_all(&mut self, acc: &CompetitiveImpactAccumulator) {
        for (mine, &theirs) in self.max_freqs.iter_mut().zip(acc.max_freqs.iter()) {
            *mine = (*mine).max(theirs);
        }
        for &entry in &acc.other {
            Self::add_to_set(entry, &mut self.other);
        }
    }

    /// `CompetitiveImpactAccumulator.add(int freq, long norm)`.
    pub(crate) fn add(&mut self, freq: i32, norm: i64) {
        if (i64::from(i8::MIN)..=i64::from(i8::MAX)).contains(&norm) {
            let index = (norm as i8) as u8 as usize;
            self.max_freqs[index] = self.max_freqs[index].max(freq);
        } else {
            let entry = Impact { freq, norm };
            let mut other = std::mem::take(&mut self.other);
            Self::add_to_set(entry, &mut other);
            self.other = other;
        }
    }

    /// Java's `getCompetitiveFreqNormPairs()`.
    pub(crate) fn get_competitive_freq_norm_pairs(&self) -> Vec<Impact> {
        let mut impacts: Vec<Impact> = Vec::new();
        let mut max_freq_for_lower_norms = 0i32;
        for (i, &max_freq) in self.max_freqs.iter().enumerate() {
            if max_freq > max_freq_for_lower_norms {
                impacts.push(Impact {
                    freq: max_freq,
                    norm: i64::from(i as u8 as i8),
                });
                max_freq_for_lower_norms = max_freq;
            }
        }
        if self.other.is_empty() {
            // Common case: every norm is a byte.
            return impacts;
        }
        let mut merged = self.other.clone();
        for impact in impacts {
            Self::add_to_set(impact, &mut merged);
        }
        merged
    }

    /// Java's private `add(Impact, TreeSet<Impact>)`: insert unless something
    /// already dominates `new_entry`, then drop everything `new_entry`
    /// dominates.
    ///
    /// `set` is kept in the order Java's comparator defines -- increasing
    /// freq, ties broken by *decreasing* unsigned norm -- so `ceiling` is a
    /// binary search and `headSet(newEntry, false).descendingIterator()` is a
    /// backwards walk from the insertion point.
    fn add_to_set(new_entry: Impact, set: &mut Vec<Impact>) {
        let key = |i: &Impact| (i.freq, std::cmp::Reverse(i.norm as u64));
        let at = set.partition_point(|i| key(i) < key(&new_entry));
        match set.get(at) {
            // `ceiling` found an entry with at least this freq and a norm at
            // least as competitive: nothing to do.
            Some(next) if (next.norm as u64) <= (new_entry.norm as u64) => return,
            _ => set.insert(at, new_entry),
        }
        // `headSet(newEntry, false).descendingIterator()`: walk back over the
        // strictly-lesser entries, dropping the ones this entry dominates and
        // stopping at the first that is not comparable.
        let mut i = at;
        while i > 0 {
            // ARITH: guarded by `i > 0`.
            #[allow(clippy::arithmetic_side_effects)]
            let prev = i - 1;
            if (set[prev].norm as u64) >= (new_entry.norm as u64) {
                set.remove(prev);
                i = prev;
            } else {
                break;
            }
        }
    }
}

/// `Lucene104PostingsWriter.writeImpacts(Collection<Impact>, DataOutput)`.
///
/// Each entry is written as a delta against the previous one, starting from
/// `(0, 0)`: a vint `freqDelta` whose low bit says whether an explicit
/// zig-zag `normDelta` vlong follows. Both deltas are stored one *less* than
/// the true delta, which is why `crate::postings::decode_impacts` adds one
/// back to each. `impacts` must be ordered by strictly increasing freq and
/// strictly increasing unsigned norm --
/// [`CompetitiveImpactAccumulator::get_competitive_freq_norm_pairs`]
/// guarantees exactly that.
fn write_impacts(impacts: &[Impact], out: &mut Vec<u8>) {
    let mut prev = Impact { freq: 0, norm: 0 };
    for impact in impacts {
        // Java's `impact.freq - previous.freq - 1` and
        // `impact.norm - previous.norm - 1` are `int`/`long` arithmetic over
        // a strictly increasing sequence, so both are non-negative; the
        // wrapping spelling keeps a hypothetical out-of-order list producing
        // Java's bytes rather than a panic, and the ordering itself is
        // asserted below in debug builds.
        debug_assert!(impact.freq > prev.freq);
        // Unconditional, as Java's `assert Long.compareUnsigned(impact.norm,
        // previous.norm) > 0` is. `prev` starts at `(0, 0)`, so on the first
        // entry this *is* `CheckIndex.checkImpacts`' "First impact had a norm
        // == 0" rule -- which is exactly the entry where the guard earns its
        // keep, so it must not be excused. (A norm of 0 out-scores every other
        // norm, so such an impact is a sound bound but an illegal byte; see
        // `competitive_impacts`.)
        debug_assert!((impact.norm as u64) > (prev.norm as u64));
        let freq_delta = impact.freq.wrapping_sub(prev.freq).wrapping_sub(1);
        let norm_delta = impact.norm.wrapping_sub(prev.norm).wrapping_sub(1);
        if norm_delta == 0 {
            // The common case -- norms increase by exactly one along the
            // frontier -- folds into a single byte.
            out.write_vint(freq_delta.wrapping_shl(1));
        } else {
            out.write_vint(freq_delta.wrapping_shl(1) | 1);
            out.write_zlong(norm_delta);
        }
        prev = *impact;
    }
}

/// The competitive impacts of one run of `(doc_id, freq)` pairs, against
/// `norms`.
///
/// `norms` is the field's per-document norm column (see
/// [`FieldNorms::values`]); `None`, or a doc id past its end, means norm `1`
/// -- respectively Java's `fieldHasNorms == false` and its
/// `norms.advanceExact(docID) == false` branch, both of which use `1L`.
///
/// Level-1 spans call this over the whole 8192-document span rather than
/// merging the 32 level-0 accumulators as Java does. The two are the same
/// set: the frontier of a union is the frontier of the union of the
/// frontiers, and `CompetitiveImpactAccumulator::add` is what computes it
/// either way.
fn competitive_impacts(docs: &[(i32, i32)], norms: Option<&[i64]>) -> Vec<Impact> {
    let mut acc = CompetitiveImpactAccumulator::default();
    for &(doc_id, freq) in docs {
        let norm = norm_of(norms, doc_id);
        // Java's `assert norm != 0 : docID`. A norm of 0 decodes to a field
        // length of 0, which out-scores every other norm -- so an impact
        // carrying it is still a sound *bound*, but `CheckIndex.checkImpacts`
        // rejects the segment ("First impact had a norm == 0"). It is
        // unreachable by construction: a norm of 0 is what a document that
        // carries the field with no tokens gets, and such a document has no
        // posting for the field to be bounded.
        debug_assert!(norm != 0, "doc {doc_id} has a posting but a norm of 0");
        acc.add(freq, norm);
    }
    acc.get_competitive_freq_norm_pairs()
}

/// One document's norm, with both of Java's fallbacks folded in -- see
/// [`competitive_impacts`].
fn norm_of(norms: Option<&[i64]>, doc_id: i32) -> i64 {
    let Some(values) = norms else { return 1 };
    if doc_id < 0 {
        return 1;
    }
    values.get(doc_id as usize).copied().unwrap_or(1)
}

/// One field's per-document norms, as `Lucene104PostingsWriter` receives them
/// from `NormsProducer.getNorms(fieldInfo)` -- the input
/// [`write_fields_with_norms`] needs to emit real impacts rather than
/// `(maxFreq, 1)` ones.
pub struct FieldNorms<'a> {
    /// The field these norms belong to; matched against
    /// [`FieldPostingsInput::field_number`].
    pub field_number: i32,
    /// The norm of document `d` at index `d` -- a **dense** column addressed
    /// by doc id, which is what `NumericDocValues.advanceExact(docID)` /
    /// `longValue()` amounts to for the writer's purposes.
    ///
    /// A doc id past the end reads as norm `1`, which is Java's
    /// `advanceExact == false` branch ("this can happen if indexing hits a
    /// problem after adding a doc to the postings but before buffering the
    /// norm"). A document that carries no norm because it carries no
    /// occurrence of the field never appears in that field's postings, so the
    /// value stored for it is never read -- callers may put anything there.
    pub values: &'a [i64],
}

/// Input to [`write_single_field`]: one field's whole term dictionary,
/// already fully materialized and sorted.
pub struct FieldPostingsInput<'a, T = TermPostings> {
    pub field_number: i32,
    pub index_options: IndexOptions,
    /// `docCount`: number of distinct docs this field occurs in at least
    /// once across the whole segment — the caller's responsibility to
    /// compute (usually `terms.iter().flat_map(|t| &t.docs).map(|(d,_)| d)`'s
    /// distinct count, but the real writer just tracks it incrementally).
    pub doc_count: i32,
    /// `FieldInfo.hasPayloads()`: a per-field property, independent of
    /// `index_options` (unlike offsets, which get their own
    /// `IndexOptions::DocsAndFreqsAndPositionsAndOffsets` variant — see
    /// `FieldInfo`/`IndexOptions` in the Java source: payloads are a plain
    /// boolean orthogonal to the `IndexOptions` enum). Only meaningful when
    /// `index_options` indexes positions; every term's `positions`/
    /// `payload_lengths` entries must line up when this is set (see
    /// [`TermPostings::payload_bytes`]).
    pub has_payloads: bool,
    pub terms: &'a [T],
}

impl<T> Clone for FieldPostingsInput<'_, T> {
    fn clone(&self) -> Self {
        *self
    }
}

impl<T> Copy for FieldPostingsInput<'_, T> {}

/// The files this writer produces for one field. `pos` is empty when
/// `index_options` doesn't index positions (`IndexOptions::Docs`/
/// `DocsAndFreqs`) — no `.pos` file is needed in that case, mirroring how a
/// real segment simply has no `.pos` file when no field in it indexes
/// positions.
#[derive(Debug, Clone, Default)]
pub struct Output {
    pub doc: Vec<u8>,
    /// `.psm`, `Lucene104PostingsWriter`'s metadata file: the four
    /// maximum-impact figures plus the final length of each postings file.
    /// Real Lucene's reader opens this before anything else and fails the
    /// whole segment if it is absent, even though this port's own reader
    /// never needs it (it derives the same lengths from the buffers it holds).
    pub psm: Vec<u8>,
    pub pos: Vec<u8>,
    /// Empty unless at least one field indexes offsets
    /// (`IndexOptions::DocsAndFreqsAndPositionsAndOffsets`) — same "no file
    /// needed" convention as `pos`.
    pub pay: Vec<u8>,
    pub tim: Vec<u8>,
    pub tip: Vec<u8>,
    pub tmd: Vec<u8>,
}

/// Writes `.doc`/`.tim`/`.tip`/`.tmd` bytes for `input`'s single field — a
/// thin one-element-slice wrapper over [`write_fields`], kept so existing
/// single-field callers/tests are unaffected.
pub fn write_single_field<T: TermSource>(
    input: &FieldPostingsInput<'_, T>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    write_fields(std::slice::from_ref(input), segment_id, segment_suffix)
}

/// [`write_single_field`] with the field's norms, so its impacts are real --
/// see [`write_fields_with_norms`].
pub fn write_single_field_with_norms<T: TermSource>(
    input: &FieldPostingsInput<'_, T>,
    norms: &[FieldNorms<'_>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    write_fields_with_norms(
        std::slice::from_ref(input),
        norms,
        segment_id,
        segment_suffix,
    )
}

/// Writes `.doc`/`.tim`/`.tip`/`.tmd` bytes for **one or more** fields in a
/// single segment — see the module doc for the exact per-field scope and
/// wire format, each of which applies independently to every field in
/// `inputs`. `numFields` in the resulting `.tmd` is `inputs.len()`; each
/// field gets its own blocks and its own `.tip` trie, and all fields'
/// blocks/nodes/records are interleaved into the *same*
/// physical `.doc`/`.pos`/`.tim`/`.tip`/`.tmd` byte buffers, exactly like a
/// real multi-field segment. `segment_id`/`segment_suffix` must match what
/// the caller will later open the files with (`blocktree::open`/
/// `postings::DocInput::open` both check them).
pub fn write_fields<T: TermSource>(
    inputs: &[FieldPostingsInput<'_, T>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    write_fields_with_norms(inputs, &[], segment_id, segment_suffix)
}

/// [`write_fields`] with each field's **per-document norms**, so the impacts
/// it writes are `CompetitiveImpactAccumulator`'s real `(freq, norm)`
/// frontier instead of the single `(maxFreq, 1)` pair a normless writer can
/// prove.
///
/// `norms` names the fields it covers by number; a field of `inputs` with no
/// entry here is written exactly as [`write_fields`] writes it, which is
/// `Lucene104PostingsWriter`'s own `fieldHasNorms == false` path (`norm =
/// 1L`) and is what a field with `FieldInfo.omitNorms` really is. Norm 1 is
/// the shortest possible field, hence the highest-scoring one, so a normless
/// bound is an over-estimate: **sound, and only ever loose**. Feeding real
/// norms makes it tight, which is the whole of the win -- and getting it
/// wrong in the other direction would make it an *under*-estimate, which
/// silently drops hits under MAXSCORE. That is why the accumulator is a
/// straight port of Java's rather than a shortcut, and why
/// `bm25_scoring_fixtures.rs` compares hit *sets* and not just scores.
pub fn write_fields_with_norms<T: TermSource>(
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    write_fields_with_options(
        inputs,
        norms,
        &WriteOptions::default(),
        segment_id,
        segment_suffix,
    )
}

/// What distinguishes one `Lucene104PostingsFormat` instance's files from
/// another's, beyond the fields it is handed: the block-tree block sizes of
/// its constructor, and which of `.pos`/`.pay` exist, which Java decides
/// from the whole segment's `FieldInfos` (`hasProx`, `hasPayloads ||
/// hasOffsets`) rather than from the fields a per-field format was given --
/// so the second format of a segment with positions anywhere has an empty
/// `.pos` of its own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteOptions {
    /// `minTermBlockSize`, `DEFAULT_MIN_BLOCK_SIZE` (25) by default.
    pub min_items_in_block: usize,
    /// `maxTermBlockSize`, `DEFAULT_MAX_BLOCK_SIZE` (48) by default.
    pub max_items_in_block: usize,
    /// Force a `.pos` (`Some(true)`) or none (`Some(false)`); `None` writes
    /// one when some input field indexes positions.
    pub has_prox: Option<bool>,
    /// The same for `.pay`: `None` writes one when some input field has
    /// offsets or payloads.
    pub has_payloads_or_offsets: Option<bool>,
    /// Write the terms dictionary with zstd blocks at this level: a block is
    /// zstd-compressed when that makes it at least an eighth smaller, and a
    /// field with about a MiB or more of blocks gets a dictionary trained on
    /// its own blocks. Only this port reads such a dictionary; `None` (the
    /// default) writes Lucene's.
    pub zstd_term_blocks: Option<i32>,
}

impl Default for WriteOptions {
    fn default() -> Self {
        WriteOptions {
            min_items_in_block: blocktree_writer::MIN_ITEMS_IN_BLOCK,
            max_items_in_block: blocktree_writer::MAX_ITEMS_IN_BLOCK,
            has_prox: None,
            has_payloads_or_offsets: None,
            zstd_term_blocks: None,
        }
    }
}

impl WriteOptions {
    /// `Lucene103BlockTreeTermsWriter.validateSettings`.
    // ARITH: `min_items_in_block >= 2` is checked before `- 1`, and a block
    // size is a constructor argument, far from `usize::MAX / 2`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn validate(&self) -> Result<()> {
        let (min, max) = (self.min_items_in_block, self.max_items_in_block);
        if min <= 1 {
            return Err(Error::InvalidBlockSizes(format!(
                "minItemsInBlock must be >= 2; got {min}"
            )));
        }
        if min > max {
            return Err(Error::InvalidBlockSizes(format!(
                "maxItemsInBlock must be >= minItemsInBlock; got maxItemsInBlock={max} minItemsInBlock={min}"
            )));
        }
        if 2 * (min - 1) > max {
            return Err(Error::InvalidBlockSizes(format!(
                "maxItemsInBlock must be at least 2*(minItemsInBlock-1); got maxItemsInBlock={max} minItemsInBlock={min}"
            )));
        }
        if let Some(level) = self.zstd_term_blocks {
            lucene_zstd::check_level(level).map_err(|e| Error::InvalidBlockSizes(e.to_string()))?;
        }
        Ok(())
    }
}

/// [`write_fields_with_norms`] for one `Lucene104PostingsFormat` instance
/// of a segment -- see [`WriteOptions`]: [`write_postings`] with the
/// block-tree terms dictionary (`Lucene103BlockTreeTermsWriter`) as its
/// terms writer.
pub fn write_fields_with_options<T: TermSource>(
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    options: &WriteOptions,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    if inputs.is_empty() {
        return Err(Error::EmptyTerms);
    }
    write_any_fields(inputs, norms, options, segment_id, segment_suffix)
}

/// A `Lucene104PostingsFormat` instance whose every field lost all its terms:
/// `Lucene103BlockTreeTermsWriter` skips a field with `numTerms == 0`, so a
/// merge whose only source terms belonged to deleted documents still writes
/// the format's five files, with no field in them. `SegmentMerger` opens the
/// consumer whenever the merged `FieldInfos.hasPostings()`, and
/// `PerFieldPostingsFormat` routes every field the sources' term
/// dictionaries list -- empty or not -- to it.
///
/// # Errors
/// [`Error::InvalidBlockSizes`] for `options`.
pub fn write_no_fields(
    options: &WriteOptions,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    write_any_fields::<TermPostings>(&[], &[], options, segment_id, segment_suffix)
}

/// [`write_fields_with_options`] fed one field at a time, and each field a
/// batch of terms at a time -- `Lucene103BlockTreeTermsWriter` over
/// `Lucene104PostingsWriter` as `FieldsConsumer.merge` drives them. A merge
/// hands it each run of merged terms as soon as the run is decoded and drops
/// the run, so it never holds the merged segment's decoded postings, only
/// their encoded bytes and each term's dictionary entry.
///
/// Writes exactly the bytes [`write_fields_with_options`] writes for the
/// same fields and terms (fields with no terms get no record, as there), with
/// `options.has_prox`/`has_payloads_or_offsets` read as `false` when unset:
/// a streaming caller cannot let the inputs decide, and a merge always knows.
pub struct StreamingFieldsWriter<'a> {
    postings: PostingsWriter<'a>,
    dict: BlockTreeDictionary,
    field: Option<StreamedField>,
}

/// The open field of a [`StreamingFieldsWriter`]: its terms' bytes,
/// statistics and file pointers so far.
struct StreamedField {
    field_number: i32,
    index_options: IndexOptions,
    has_payloads: bool,
    term_bytes: Vec<u8>,
    /// Term `i` is `term_bytes[term_ends[i - 1]..term_ends[i]]`.
    term_ends: Vec<usize>,
    doc_freqs: Vec<i32>,
    total_term_freqs: Vec<i64>,
    singletons: Vec<i32>,
    states: TermStateColumns,
}

impl StreamedField {
    fn last_term(&self) -> Option<&[u8]> {
        let end = *self.term_ends.last()?;
        let start = self
            .term_ends
            .len()
            .checked_sub(2)
            .map_or(0, |i| self.term_ends[i]);
        Some(&self.term_bytes[start..end])
    }
}

impl<'a> StreamingFieldsWriter<'a> {
    /// The postings and dictionary headers.
    ///
    /// # Errors
    /// [`Error::InvalidBlockSizes`] for `options`.
    pub fn new(
        options: &WriteOptions,
        segment_id: &'a [u8; ID_LENGTH],
        segment_suffix: &'a str,
    ) -> Result<Self> {
        options.validate()?;
        Ok(Self {
            postings: PostingsWriter::new(
                options.has_prox.unwrap_or(false),
                options.has_payloads_or_offsets.unwrap_or(false),
                segment_id,
                segment_suffix,
            ),
            dict: BlockTreeDictionary::new(options, segment_id, segment_suffix),
            field: None,
        })
    }

    /// Streams `.doc`/`.pos`/`.pay` to these outputs (see
    /// [`PostingsWriter::stream_to`]); [`Self::finish`] then closes them and
    /// returns those three buffers empty.
    pub fn stream_to(
        &mut self,
        doc: FsIndexOutput,
        pos: Option<FsIndexOutput>,
        pay: Option<FsIndexOutput>,
    ) {
        self.postings.stream_to(doc, pos, pay);
    }

    /// `TermsWriter` for one field; its terms follow through
    /// [`Self::add_terms`], then [`Self::finish_field`].
    ///
    /// # Errors
    /// [`Error::UnsupportedIndexOptions`]; a field already open.
    pub fn start_field(
        &mut self,
        field_number: i32,
        index_options: IndexOptions,
        has_payloads: bool,
    ) -> Result<()> {
        if !matches!(
            index_options,
            IndexOptions::Docs
                | IndexOptions::DocsAndFreqs
                | IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
                | IndexOptions::DocsAndCustomFreqs
        ) {
            return Err(Error::UnsupportedIndexOptions(index_options));
        }
        if self.field.is_some() {
            return Err(Error::StreamingOrder(
                "start_field: the previous field is still open",
            ));
        }
        self.field = Some(StreamedField {
            field_number,
            index_options,
            has_payloads,
            term_bytes: Vec::new(),
            term_ends: Vec::new(),
            doc_freqs: Vec::new(),
            total_term_freqs: Vec::new(),
            singletons: Vec::new(),
            states: TermStateColumns::default(),
        });
        Ok(())
    }

    /// The open field's next terms, ascending and after every term already
    /// added; `norms` as [`write_fields_with_norms`] takes them. An empty
    /// batch is nothing.
    ///
    /// # Errors
    /// What [`write_fields`] rejects in a field's terms (indices are within
    /// the batch, except [`Error::TermsNotSorted`], which counts from the
    /// field's first term); no field open.
    pub fn add_terms<T: TermSource>(
        &mut self,
        terms: &[T],
        norms: &[FieldNorms<'_>],
    ) -> Result<()> {
        let Some(field) = self.field.as_mut() else {
            return Err(Error::StreamingOrder("add_terms: no field is open"));
        };
        if terms.is_empty() {
            return Ok(());
        }
        if field
            .last_term()
            .is_some_and(|last| last >= terms[0].term())
        {
            return Err(Error::TermsNotSorted(field.term_ends.len()));
        }
        let input = FieldPostingsInput {
            field_number: field.field_number,
            index_options: field.index_options,
            doc_count: 0,
            has_payloads: field.has_payloads,
            terms,
        };
        let stats = validate_field(&input)?;
        let (block_terms, singletons) = stats.slices();
        self.postings
            .encode_terms(&input, singletons, norms, &mut field.states);
        self.postings.maybe_spill();
        for term in block_terms {
            field.term_bytes.extend_from_slice(term.bytes);
            field.term_ends.push(field.term_bytes.len());
            field.doc_freqs.push(term.doc_freq);
            field.total_term_freqs.push(term.total_term_freq);
        }
        field.singletons.extend_from_slice(singletons);
        Ok(())
    }

    /// Closes the open field: its dictionary blocks, trie and `.tmd` record,
    /// or nothing when it got no terms. `doc_count` is `docsSeen`.
    ///
    /// # Errors
    /// No field open.
    pub fn finish_field(&mut self, doc_count: i32) -> Result<()> {
        let Some(field) = self.field.take() else {
            return Err(Error::StreamingOrder("finish_field: no field is open"));
        };
        if field.term_ends.is_empty() {
            return Ok(());
        }
        let mut start = 0usize;
        let block_terms: Vec<blocktree_writer::BlockTerm<'_>> = field
            .term_ends
            .iter()
            .zip(&field.doc_freqs)
            .zip(&field.total_term_freqs)
            .map(|((&end, &doc_freq), &total_term_freq)| {
                let bytes = &field.term_bytes[start..end];
                start = end;
                blocktree_writer::BlockTerm {
                    bytes,
                    doc_freq,
                    total_term_freq,
                }
            })
            .collect();
        self.dict.write_field(&FieldTermStates {
            field_number: field.field_number,
            index_options: field.index_options,
            has_payloads: field.has_payloads,
            doc_count,
            block_terms: &block_terms,
            singletons: &field.singletons,
            doc_start_fp: &field.states.doc_start_fp,
            pos_start_fp: &field.states.pos_start_fp,
            pay_start_fp: &field.states.pay_start_fp,
            last_pos_block_offset: &field.states.last_pos_block_offset,
        });
        Ok(())
    }

    /// Every file, footers included.
    ///
    /// # Errors
    /// A field still open.
    pub fn finish(self) -> Result<Output> {
        if self.field.is_some() {
            return Err(Error::StreamingOrder("finish: a field is still open"));
        }
        let postings = self.postings.finish()?;
        let (tim, tip, tmd) = self.dict.finish();
        Ok(Output {
            doc: postings.doc,
            psm: postings.psm,
            pos: postings.pos,
            pay: postings.pay,
            tim,
            tip,
            tmd,
        })
    }
}

/// [`write_fields_with_options`] without its non-empty check.
fn write_any_fields<T: TermSource>(
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    options: &WriteOptions,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    options.validate()?;
    let mut dict = BlockTreeDictionary::new(options, segment_id, segment_suffix);
    let postings = write_any_postings(
        inputs,
        norms,
        options.has_prox,
        options.has_payloads_or_offsets,
        segment_id,
        segment_suffix,
        |field| -> Result<()> {
            dict.write_field(field);
            Ok(())
        },
    )?;
    let (tim, tip, tmd) = dict.finish();
    Ok(Output {
        doc: postings.doc,
        psm: postings.psm,
        pos: postings.pos,
        pay: postings.pay,
        tim,
        tip,
        tmd,
    })
}

/// `Lucene103BlockTreeTermsWriter` over [`FieldTermStates`]: the `.tim`,
/// `.tip` and `.tmd` of one postings format instance, one field at a time.
struct BlockTreeDictionary {
    tim: Vec<u8>,
    tip: Vec<u8>,
    tmd: Vec<u8>,
    /// The `.tmd` field records, after `numFields`: kept apart so that a
    /// streaming writer, which only learns at the end which fields had
    /// terms, can still write the count first.
    records: Vec<u8>,
    num_fields: usize,
    block_sizes: (usize, usize),
    /// [`WriteOptions::zstd_term_blocks`].
    zstd_level: Option<i32>,
}

/// Fields with fewer bytes of terms than this get no zstd dictionary: one
/// would cost more than it saves.
const ZSTD_DICTIONARY_MIN_TERM_BYTES: usize = 1 << 20;
/// The most bytes of a field's zstd dictionary.
const ZSTD_DICTIONARY_SIZE: usize = 16 << 10;
/// About how many bytes of blocks a dictionary is trained on.
const ZSTD_SAMPLE_BYTES: usize = 2 << 20;

/// A zstd dictionary for `terms`' blocks, trained on a dry run of the block
/// writer over them; empty when the field is too small for one or training
/// fails.
// ARITH: sums of in-memory term lengths.
#[allow(clippy::arithmetic_side_effects)]
fn train_term_block_dictionary(
    terms: &[blocktree_writer::BlockTerm<'_>],
    block_sizes: (usize, usize),
    has_freqs: bool,
    encode_meta: impl FnMut(&mut Vec<u8>, &[usize]),
) -> Vec<u8> {
    let term_bytes: usize = terms.iter().map(|t| t.bytes.len() + 4).sum();
    if term_bytes < ZSTD_DICTIONARY_MIN_TERM_BYTES {
        return Vec::new();
    }
    let (mut tim, mut tip) = (Vec::new(), Vec::new());
    let (mut samples, mut sizes) = (Vec::new(), Vec::new());
    blocktree_writer::write_field_terms(
        &mut tim,
        &mut tip,
        terms,
        block_sizes,
        has_freqs,
        encode_meta,
        Some(blocktree_writer::BlockZstd::Sample {
            samples: &mut samples,
            sizes: &mut sizes,
            stride: term_bytes.div_ceil(ZSTD_SAMPLE_BYTES),
        }),
    );
    lucene_zstd::train_dictionary_joined(&samples, &sizes, ZSTD_DICTIONARY_SIZE).unwrap_or_default()
}

impl BlockTreeDictionary {
    /// The three headers. `options` must be validated.
    fn new(options: &WriteOptions, segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Self {
        // Every file starts at a size a small segment fits: growing an empty
        // buffer through 8, 16, 32 ... bytes was a measurable share of a
        // one-document flush (`MemoryIndex`).
        let initial = 256usize;
        let header = |codec| {
            let mut out = Vec::with_capacity(initial);
            codec_util::write_index_header(
                &mut out,
                codec,
                BLOCKTREE_VERSION_CURRENT,
                segment_id,
                segment_suffix,
            );
            out
        };
        let (tim, tip, mut tmd) = if options.zstd_term_blocks.is_some() {
            (
                header(ZSTD_TERMS_CODEC_NAME),
                header(ZSTD_TERMS_INDEX_CODEC_NAME),
                header(ZSTD_TERMS_META_CODEC_NAME),
            )
        } else {
            (
                header(TERMS_CODEC_NAME),
                header(TERMS_INDEX_CODEC_NAME),
                header(TERMS_META_CODEC_NAME),
            )
        };
        write_postings_terms_header(&mut tmd, segment_id, segment_suffix);
        BlockTreeDictionary {
            tim,
            tip,
            tmd,
            records: Vec::with_capacity(initial),
            num_fields: 0,
            block_sizes: (options.min_items_in_block, options.max_items_in_block),
            zstd_level: options.zstd_term_blocks,
        }
    }

    /// One field's `.tim` blocks, `.tip` trie and `.tmd` record. `field`
    /// must have terms.
    fn write_field(&mut self, field: &FieldTermStates<'_>) {
        self.num_fields = self.num_fields.saturating_add(1);
        let Self {
            tim,
            tip,
            records: tmd,
            ..
        } = self;
        // ---- this field's .tim blocks + .tip trie ----
        // `Lucene103BlockTreeTermsWriter`'s block splitting and
        // `TrieBuilder`, in `crate::blocktree_writer`; each block's term
        // metadata is encoded by the postings writer, against this
        // field's per-term file pointers.
        let block_terms = field.block_terms;
        let has_freqs = field.index_options != IndexOptions::Docs;
        let encode = |meta: &mut Vec<u8>, indices: &[usize]| field.encode_terms(meta, indices);
        let mut dictionary = Vec::new();
        let trie = match self.zstd_level {
            None => blocktree_writer::write_field_terms(
                tim,
                tip,
                block_terms,
                self.block_sizes,
                has_freqs,
                encode,
                None,
            ),
            Some(level) => {
                dictionary =
                    train_term_block_dictionary(block_terms, self.block_sizes, has_freqs, encode);
                // The level was validated with the options, so neither fails.
                let prepared = (!dictionary.is_empty())
                    .then(|| lucene_zstd::EncoderDictionary::new(&dictionary, level).ok())
                    .flatten();
                let mut compressor =
                    lucene_zstd::BareCompressor::new(level, prepared.as_ref()).ok();
                // The `.tmd` must name exactly the dictionary the blocks were
                // compressed against: a reader decoding a dictionary-less
                // frame against one starts from the dictionary's repeat
                // offsets and entropy tables, not the frame's.
                if prepared.is_none() || compressor.is_none() {
                    dictionary.clear();
                }
                blocktree_writer::write_field_terms(
                    tim,
                    tip,
                    block_terms,
                    self.block_sizes,
                    has_freqs,
                    encode,
                    compressor
                        .as_mut()
                        .map(blocktree_writer::BlockZstd::Compress),
                )
            }
        };

        // ---- this field's .tmd record ----
        tmd.write_vint(field.field_number);
        tmd.write_vlong(block_terms.len() as i64);
        let sum_doc_freq: i64 = block_terms.iter().map(|t| i64::from(t.doc_freq)).sum();
        if field.index_options != IndexOptions::Docs {
            let sum_total_term_freq: i64 = block_terms.iter().map(|t| t.total_term_freq).sum();
            tmd.write_vlong(sum_total_term_freq);
        }
        tmd.write_vlong(sum_doc_freq);
        tmd.write_vint(field.doc_count);
        let min_term = block_terms[0].bytes;
        // ARITH: a written field has terms (`validate_field`,
        // `write_block_tree_dictionary`).
        #[allow(clippy::arithmetic_side_effects)]
        let max_term = block_terms[block_terms.len() - 1].bytes;
        tmd.write_vint(min_term.len() as i32);
        tmd.write_bytes(min_term);
        tmd.write_vint(max_term.len() as i32);
        tmd.write_bytes(max_term);
        tmd.write_vlong(trie.index_start as i64);
        tmd.write_vlong(trie.root_fp as i64);
        tmd.write_vlong(trie.index_end as i64);
        if self.zstd_level.is_some() {
            tmd.write_vint(dictionary.len() as i32);
            tmd.write_bytes(&dictionary);
        }
    }

    /// The footers, and the `.tmd`'s two lengths: both are the *whole* file,
    /// footer included -- Java writes each footer and only then records
    /// `getFilePointer()`, and its reader feeds these straight to
    /// `CodecUtil.retrieveChecksum`, which rejects any file whose real length
    /// disagrees.
    fn finish(self) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let Self {
            mut tim,
            mut tip,
            mut tmd,
            records,
            num_fields,
            ..
        } = self;
        tmd.write_vint(num_fields as i32); // numFields
        tmd.extend_from_slice(&records);
        codec_util::write_footer(&mut tim);
        codec_util::write_footer(&mut tip);
        tmd.write_i64(tip.len() as i64); // indexLength
        tmd.write_i64(tim.len() as i64); // termsLength
        codec_util::write_footer(&mut tmd);
        (tim, tip, tmd)
    }
}

/// One term of a dictionary some other terms dictionary wrote over
/// `Lucene104PostingsWriter`'s postings: its bytes, statistics and
/// `IntBlockTermState`, as that dictionary's reader decodes them.
#[derive(Debug, Clone)]
pub struct DictionaryTerm {
    pub bytes: Vec<u8>,
    pub doc_freq: i32,
    pub total_term_freq: i64,
    pub meta: crate::postings::TermMetadata,
}

/// One field of such a dictionary, its terms in ascending order.
#[derive(Debug, Clone)]
pub struct DictionaryField {
    pub field_number: i32,
    pub index_options: IndexOptions,
    pub has_payloads: bool,
    pub doc_count: i32,
    pub terms: Vec<DictionaryTerm>,
}

/// Rewrites another terms dictionary's fields as a `Lucene103` block-tree
/// `.tim`/`.tip`/`.tmd` under `segment_suffix`, every term pointing at the
/// same postings it pointed at before -- so a reader that only knows the
/// block tree serves a segment whose postings went to `BlockTreeOrds`,
/// `UniformSplit` or any other dictionary over `Lucene104PostingsWriter`.
/// Fields without terms are left out, as `Lucene103BlockTreeTermsWriter`
/// leaves them out.
///
/// # Errors
/// [`Error::InvalidBlockSizes`] for `options`; [`Error::TermsNotSorted`]
/// when a field's terms are not strictly ascending.
pub fn write_block_tree_dictionary(
    fields: &[DictionaryField],
    options: &WriteOptions,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<(Vec<u8>, Vec<u8>, Vec<u8>)> {
    options.validate()?;
    let written: Vec<&DictionaryField> = fields.iter().filter(|f| !f.terms.is_empty()).collect();
    let mut dict = BlockTreeDictionary::new(options, segment_id, segment_suffix);
    for field in written {
        if let Some(i) = field
            .terms
            .windows(2)
            .position(|pair| pair[0].bytes >= pair[1].bytes)
        {
            // ARITH: `i` indexes a window, so `i + 1` is a term index.
            #[allow(clippy::arithmetic_side_effects)]
            return Err(Error::TermsNotSorted(i + 1));
        }
        let block_terms: Vec<blocktree_writer::BlockTerm<'_>> = field
            .terms
            .iter()
            .map(|t| blocktree_writer::BlockTerm {
                bytes: &t.bytes,
                doc_freq: t.doc_freq,
                total_term_freq: t.total_term_freq,
            })
            .collect();
        let column = |f: fn(&DictionaryTerm) -> u64| field.terms.iter().map(f).collect::<Vec<_>>();
        let singletons: Vec<i32> = field
            .terms
            .iter()
            .map(|t| t.meta.singleton_doc_id)
            .collect();
        let doc_start_fp = column(|t| t.meta.doc_start_fp);
        let pos_start_fp = column(|t| t.meta.pos_start_fp);
        let pay_start_fp = column(|t| t.meta.pay_start_fp);
        let last_pos_block_offset: Vec<i64> = field
            .terms
            .iter()
            .map(|t| t.meta.last_pos_block_offset)
            .collect();
        dict.write_field(&FieldTermStates {
            field_number: field.field_number,
            index_options: field.index_options,
            has_payloads: field.has_payloads,
            doc_count: field.doc_count,
            block_terms: &block_terms,
            singletons: &singletons,
            doc_start_fp: &doc_start_fp,
            pos_start_fp: &pos_start_fp,
            pay_start_fp: &pay_start_fp,
            last_pos_block_offset: &last_pos_block_offset,
        });
    }
    Ok(dict.finish())
}

/// `Lucene104PostingsWriter.init(termsOut, state)`: the postings writer's
/// own header (`Lucene104PostingsWriterTerms`) and `BLOCK_SIZE`, which every
/// terms dictionary over these postings embeds in its own file.
pub fn write_postings_terms_header(
    out: &mut Vec<u8>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) {
    codec_util::write_index_header(
        out,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    out.write_vint(POSTINGS_BLOCK_SIZE);
}

/// The postings files [`write_postings`] produces: everything of
/// [`Output`] but the terms dictionary's.
#[derive(Debug, Clone, Default)]
pub struct PostingsFiles {
    /// `.doc`.
    pub doc: Vec<u8>,
    /// `.psm`.
    pub psm: Vec<u8>,
    /// `.pos`, empty when the segment has no positions.
    pub pos: Vec<u8>,
    /// `.pay`, empty when the segment has no offsets or payloads.
    pub pay: Vec<u8>,
}

/// One field's terms as `PostingsWriterBase.writeTerm` left them: each
/// term's bytes, statistics and `BlockTermState`, which a terms dictionary
/// writes with [`Self::encode_term`] (`encodeTerm`).
#[derive(Debug, Clone, Copy)]
pub struct FieldTermStates<'s> {
    /// The field's number.
    pub field_number: i32,
    /// The field's `IndexOptions`.
    pub index_options: IndexOptions,
    /// `FieldInfo.hasPayloads()`.
    pub has_payloads: bool,
    /// `docsSeen.cardinality()`: the documents with the field.
    pub doc_count: i32,
    block_terms: &'s [blocktree_writer::BlockTerm<'s>],
    singletons: &'s [i32],
    doc_start_fp: &'s [u64],
    pos_start_fp: &'s [u64],
    pay_start_fp: &'s [u64],
    last_pos_block_offset: &'s [i64],
}

impl FieldTermStates<'_> {
    /// The number of terms.
    pub fn len(&self) -> usize {
        self.block_terms.len()
    }

    /// Whether the field has no terms (never, for a written field).
    pub fn is_empty(&self) -> bool {
        self.block_terms.is_empty()
    }

    /// Term `i`'s bytes.
    pub fn term(&self, i: usize) -> &[u8] {
        self.block_terms[i].bytes
    }

    /// Term `i`'s `docFreq`.
    pub fn doc_freq(&self, i: usize) -> i32 {
        self.block_terms[i].doc_freq
    }

    /// Term `i`'s `BlockTermState.totalTermFreq`: `-1` for a field without
    /// frequencies, as `Lucene104PostingsWriter.finishTerm` leaves it.
    // SENTINEL: `-1`, no frequencies (a terms dictionary writes it as such).
    pub fn total_term_freq(&self, i: usize) -> i64 {
        if self.index_options == IndexOptions::Docs {
            -1
        } else {
            self.block_terms[i].total_term_freq
        }
    }

    /// Term `i`'s `IntBlockTermState` pointers as
    /// `Lucene104PostingsWriter.finishTerm` leaves them: `lastPosBlockOffset`
    /// is `-1` unless the field has positions and `totalTermFreq >
    /// BLOCK_SIZE`; `.pos`/`.pay` pointers of a field without them are `0`.
    /// For a terms dictionary with its own term-state encoding
    /// (`uniformsplit`'s `DeltaBaseTermStateSerializer`).
    pub(crate) fn term_metadata(&self, i: usize) -> crate::postings::TermMetadata {
        let has_positions = self.index_options.subsumes_positions();
        let ttf = self.block_terms[i].total_term_freq;
        crate::postings::TermMetadata {
            doc_start_fp: self.doc_start_fp.get(i).copied().unwrap_or(0),
            singleton_doc_id: self.singletons.get(i).copied().unwrap_or(-1),
            pos_start_fp: self.pos_start_fp.get(i).copied().unwrap_or(0),
            pay_start_fp: self.pay_start_fp.get(i).copied().unwrap_or(0),
            last_pos_block_offset: if has_positions && ttf > i64::from(BLOCK_SIZE) {
                self.last_pos_block_offset.get(i).copied().unwrap_or(-1)
            } else {
                -1
            },
            skip_offset: -1,
        }
    }

    /// `encodeTerm(out, fieldInfo, state, absolute=true)` for term `i`.
    pub fn encode_term(&self, out: &mut Vec<u8>, i: usize) {
        self.encode_terms(out, std::slice::from_ref(&i));
    }

    /// `encodeTerm` for `indices` in order, the first `absolute` and each
    /// later one against the one before -- a block-tree block's metadata.
    pub(crate) fn encode_terms(&self, out: &mut Vec<u8>, indices: &[usize]) {
        let has_positions = self.index_options.subsumes_positions();
        write_term_metadata(
            out,
            self.block_terms,
            self.singletons,
            indices,
            self.doc_start_fp,
            self.pos_start_fp,
            self.pay_start_fp,
            self.last_pos_block_offset,
            has_positions,
            has_positions && (self.index_options.subsumes_offsets() || self.has_payloads),
        );
    }
}

/// `Lucene104PostingsWriter` for one postings format instance of a segment:
/// writes every field's `.doc`/`.pos`/`.pay` and the `.psm`, handing each
/// field's [`FieldTermStates`] to `terms_dict` -- the terms dictionary
/// (`Lucene103BlockTreeTermsWriter`, `FSTTermsWriter`, ...) -- once the
/// field's postings are laid out. `has_prox`/`has_payloads_or_offsets` are
/// [`WriteOptions`]'.
///
/// # Errors
/// What [`write_fields`] rejects in each field, and what `terms_dict` fails
/// with. An empty field group is valid, as in Java: it writes the postings
/// headers, footers and zero maxima without calling `terms_dict`.
pub fn write_postings<T, E, F>(
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    has_prox: Option<bool>,
    has_payloads_or_offsets: Option<bool>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    terms_dict: F,
) -> std::result::Result<PostingsFiles, E>
where
    T: TermSource,
    E: From<Error>,
    F: FnMut(&FieldTermStates<'_>) -> std::result::Result<(), E>,
{
    write_any_postings(
        inputs,
        norms,
        has_prox,
        has_payloads_or_offsets,
        segment_id,
        segment_suffix,
        terms_dict,
    )
}

/// Shared implementation for [`write_postings`] and the block-tree writer:
/// no fields writes the files' headers, footers and an all-zero `.psm`.
fn write_any_postings<T, E, F>(
    inputs: &[FieldPostingsInput<'_, T>],
    norms: &[FieldNorms<'_>],
    has_prox: Option<bool>,
    has_payloads_or_offsets: Option<bool>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    mut terms_dict: F,
) -> std::result::Result<PostingsFiles, E>
where
    T: TermSource,
    E: From<Error>,
    F: FnMut(&FieldTermStates<'_>) -> std::result::Result<(), E>,
{
    let mut field_stats = Vec::with_capacity(inputs.len());
    for input in inputs {
        field_stats.push(validate_field(input)?);
    }
    let any_positions = has_prox.unwrap_or_else(|| {
        inputs
            .iter()
            .any(|input| input.index_options.subsumes_positions())
    });
    let any_offsets = has_payloads_or_offsets.unwrap_or_else(|| {
        inputs
            .iter()
            .any(|input| input.index_options.subsumes_offsets())
    });
    let any_payloads =
        has_payloads_or_offsets.unwrap_or_else(|| inputs.iter().any(|input| input.has_payloads));
    let mut writer = PostingsWriter::new(
        any_positions,
        any_offsets || any_payloads,
        segment_id,
        segment_suffix,
    );
    for (input, stats) in inputs.iter().zip(field_stats) {
        writer.write_validated_field(input, stats, norms, &mut terms_dict)?;
    }
    Ok(writer.finish()?)
}

/// Per term, in term order, the `IntBlockTermState` file pointers
/// `finishTerm` leaves: `.doc` always, `.pos` and `lastPosBlockOffset` only
/// for a field with positions, `.pay` only for one that also has offsets or
/// payloads (empty otherwise, as [`FieldTermStates`] expects).
#[derive(Debug, Default)]
struct TermStateColumns {
    doc_start_fp: Vec<u64>,
    pos_start_fp: Vec<u64>,
    pay_start_fp: Vec<u64>,
    last_pos_block_offset: Vec<i64>,
}

impl TermStateColumns {
    fn clear(&mut self) {
        self.doc_start_fp.clear();
        self.pos_start_fp.clear();
        self.pay_start_fp.clear();
        self.last_pos_block_offset.clear();
    }
}

/// The postings streams kept open across `setField`/`writeTerm` calls.
/// Shared-terms merges supply one term/field at a time instead of buffering
/// an entire merged segment before the existing writer can consume it.
pub(crate) struct PostingsWriter<'a> {
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
    /// Where `.doc`/`.pos`/`.pay` go when [`Self::stream_to`] set outputs,
    /// with how many bytes of each already went there.
    spill: Option<PostingsSpill>,
    maxima: PostingsMaxima,
    /// The current call's term pointers and position layouts, kept for
    /// their capacity.
    states: TermStateColumns,
    layouts: Vec<PositionLayout>,
    any_positions: bool,
    any_offsets_or_payloads: bool,
    segment_id: &'a [u8; ID_LENGTH],
    segment_suffix: &'a str,
}

/// Buffered postings bytes a [`PostingsWriter::stream_to`] writer keeps
/// before handing them to its outputs (a few KB in unit tests, so that they
/// spill many times).
pub const SPILL_BYTES: usize = if cfg!(test) { 1 << 12 } else { 1 << 20 };

/// [`PostingsWriter::stream_to`]'s outputs and how much each has taken.
struct PostingsSpill {
    doc: FsIndexOutput,
    pos: Option<FsIndexOutput>,
    pay: Option<FsIndexOutput>,
    doc_bytes: u64,
    pos_bytes: u64,
    pay_bytes: u64,
}

impl PostingsSpill {
    /// Moves the buffers' bytes to the outputs, keeping the buffers' capacity.
    fn write(&mut self, doc: &mut Vec<u8>, pos: &mut Vec<u8>, pay: &mut Vec<u8>) {
        let send = |out: Option<&mut FsIndexOutput>, buf: &mut Vec<u8>, total: &mut u64| {
            if let Some(out) = out {
                out.write_bytes(buf);
                *total = total.saturating_add(buf.len() as u64);
                buf.clear();
            }
        };
        send(Some(&mut self.doc), doc, &mut self.doc_bytes);
        send(self.pos.as_mut(), pos, &mut self.pos_bytes);
        send(self.pay.as_mut(), pay, &mut self.pay_bytes);
    }
}

impl<'a> PostingsWriter<'a> {
    /// `Lucene104PostingsWriter`'s constructor: headers for the files the
    /// whole segment's FieldInfos require, including an empty field group.
    pub(crate) fn new(
        any_positions: bool,
        any_offsets_or_payloads: bool,
        segment_id: &'a [u8; ID_LENGTH],
        segment_suffix: &'a str,
    ) -> Self {
        let header = |codec| {
            let mut out = Vec::with_capacity(256);
            codec_util::write_index_header(
                &mut out,
                codec,
                DOC_VERSION_CURRENT,
                segment_id,
                segment_suffix,
            );
            out
        };
        Self {
            doc: header(DOC_CODEC),
            pos: if any_positions {
                header(POS_CODEC)
            } else {
                Vec::new()
            },
            pay: if any_offsets_or_payloads {
                header(PAY_CODEC)
            } else {
                Vec::new()
            },
            maxima: PostingsMaxima::default(),
            spill: None,
            states: TermStateColumns::default(),
            layouts: Vec::new(),
            any_positions,
            any_offsets_or_payloads,
            segment_id,
            segment_suffix,
        }
    }

    /// `setField` and `writeTerm`, handing their states to the dictionary.
    pub(crate) fn write_field<T, E, F>(
        &mut self,
        input: &FieldPostingsInput<'_, T>,
        norms: &[FieldNorms<'_>],
        terms_dict: F,
    ) -> std::result::Result<(), E>
    where
        T: TermSource,
        E: From<Error>,
        F: FnMut(&FieldTermStates<'_>) -> std::result::Result<(), E>,
    {
        let stats = validate_field(input)?;
        self.write_validated_field(input, stats, norms, terms_dict)
    }

    fn write_validated_field<'s, T, E, F>(
        &mut self,
        input: &FieldPostingsInput<'s, T>,
        stats: FieldStats<'s>,
        norms: &[FieldNorms<'_>],
        mut terms_dict: F,
    ) -> std::result::Result<(), E>
    where
        T: TermSource,
        E: From<Error>,
        F: FnMut(&FieldTermStates<'_>) -> std::result::Result<(), E>,
    {
        let (block_terms, singletons) = stats.slices();
        // Reused from field to field (and from call to call: the
        // incremental shared-terms merge supplies one term per call).
        let mut states = std::mem::take(&mut self.states);
        states.clear();
        self.encode_terms(input, singletons, norms, &mut states);
        let result = terms_dict(&FieldTermStates {
            field_number: input.field_number,
            index_options: input.index_options,
            has_payloads: input.has_payloads,
            doc_count: input.doc_count,
            block_terms,
            singletons,
            doc_start_fp: &states.doc_start_fp,
            pos_start_fp: &states.pos_start_fp,
            pay_start_fp: &states.pay_start_fp,
            last_pos_block_offset: &states.last_pos_block_offset,
        });
        self.states = states;
        result
    }

    /// `startTerm`/`startDoc`/`addPosition`/`finishTerm` for `input`'s terms
    /// (validated; `singletons` from their [`FieldStats`]), appending each
    /// term's file pointers to `states`.
    fn encode_terms<T: TermSource>(
        &mut self,
        input: &FieldPostingsInput<'_, T>,
        singletons: &[i32],
        norms: &[FieldNorms<'_>],
        states: &mut TermStateColumns,
    ) {
        let (doc_base, pos_base, pay_base) = self
            .spill
            .as_ref()
            .map_or((0, 0, 0), |s| (s.doc_bytes, s.pos_bytes, s.pay_bytes));
        let Self {
            doc,
            pos,
            pay,
            maxima,
            layouts,
            ..
        } = self;
        let index_has_positions = input.index_options.subsumes_positions();
        let index_has_freq = input.index_options != IndexOptions::Docs;
        // `Lucene104PostingsWriter.setField`'s `fieldHasNorms`: this field's
        // norms if the caller supplied them, `None` (norm 1 everywhere) if
        // not. Resolved once per field, not once per document.
        let field_norms: Option<&[i64]> = norms
            .iter()
            .find(|n| n.field_number == input.field_number)
            .map(|n| n.values);

        // ---- `.pos`/`.pay` first ----
        //
        // `.doc`'s level-0/level-1 skip records carry the `.pos`/`.pay` file
        // pointers each block's documents start at, so the position streams
        // have to be laid out before the `.doc` stream that points into them.
        // Real Lucene never faces this ordering question because it
        // interleaves all three files as documents arrive; this writer builds
        // each file whole, so it lays `.pos`/`.pay` down first and hands
        // `.doc` the resulting [`PositionLayout`].
        //
        // `pos_start_fp[i]` is term `i`'s absolute byte offset into the
        // shared `.pos` buffer, same convention as `doc_start_fp` below.
        // Absent when this field doesn't index positions: term_metadata
        // supplies zero, and write_term_metadata never reads the array.
        let index_has_offsets = input.index_options.subsumes_offsets();
        let index_has_payloads = input.has_payloads;
        let index_has_offsets_or_payloads = index_has_offsets || index_has_payloads;
        let TermStateColumns {
            doc_start_fp,
            pos_start_fp,
            pay_start_fp,
            last_pos_block_offset,
        } = states;
        // `lastPosBlockOffset` per term: where this term's vint position tail
        // starts, relative to its own `posStartFP`. Only written to the term
        // metadata when `totalTermFreq > BLOCK_SIZE` (see
        // [`write_term_metadata`]), but computed for every term because
        // [`write_position_tail`] is the only place that knows it.
        // One layout per term when the field indexes positions, none at all
        // otherwise: a `Vec<Option<_>>` of `None`s would cost a 80-byte slot
        // per term (80 MB for a million-term field) for nothing.
        layouts.clear();
        if index_has_positions {
            layouts.reserve(input.terms.len());
            let mut scratch = PositionScratch::default();
            for t in input.terms.iter() {
                pos_start_fp.push(pos_base.saturating_add(pos.len() as u64));
                if index_has_offsets_or_payloads {
                    pay_start_fp.push(pay_base.saturating_add(pay.len() as u64));
                }
                let layout = write_position_tail(
                    pos,
                    pay,
                    (pos_base, pay_base),
                    t,
                    index_has_offsets,
                    index_has_payloads,
                    &mut scratch,
                );
                last_pos_block_offset.push(layout.last_pos_block_offset as i64);
                layouts.push(layout);
            }
        }

        // `doc_start_fp[i]` is term `i`'s byte offset into the *shared* `.doc`
        // buffer (relative to the whole file including its header — the same
        // absolute convention `postings::TermMetadata::doc_start_fp` decodes
        // into) where its tail block begins, or `0` for a singleton term
        // (never read for singletons, see `postings::singleton_postings`).
        let mut block_scratch = BlockScratch::default();
        for (i, t) in input.terms.iter().enumerate() {
            // `startTerm` samples `docOut.getFilePointer()` for every term,
            // singletons included: a singleton writes no `.doc` bytes, so its
            // `docStartFP` equals the next term's, and `encodeTerm` keys the
            // zigzag singleton branch off that equality.
            doc_start_fp.push(doc_base.saturating_add(doc.len() as u64));
            if singletons[i] != -1 {
                continue;
            }
            let docs = t.docs();

            // Zero or more full 256-doc `ForUtil`/`PForUtil` blocks
            // (`write_full_block`) followed by at most one group-varint tail
            // block for the `docFreq % BLOCK_SIZE` remainder -- the exact
            // write-side inverse of `DocInput::read_postings`'s own
            // full-blocks-then-tail dispatch.
            let mut prev_doc_id = -1i32;
            let mut level1_last_doc_id = -1i32;
            let mut start = 0usize;
            // The running `.pos`/`.pay` pointers this term's skip records
            // carry (`Lucene104PostingsWriter`'s `level0LastPosFP` and
            // friends, reset per term at `startTerm`).
            let payload_lengths = if index_has_payloads {
                t.payload_lengths()
            } else {
                &[]
            };
            let mut skip = PosSkipWriter::new(
                layouts.get(i),
                payload_lengths,
                index_has_offsets_or_payloads,
            );
            // `docFreq >= LEVEL1_NUM_DOCS` (8192): emit a level-1 skip entry
            // before every complete span of `LEVEL1_FACTOR` (32) full
            // level-0 blocks, mirroring `DocInput::read_postings`'s own
            // `doc_count_left >= LEVEL1_NUM_DOCS` loop exactly.
            // ARITH: `start` only ever advances by the very amount the loop
            // condition just proved is left (`len - start >= N` before
            // `start += N`), so `start <= docs.len()` holds at every test
            // and `start + N <= len` at every slice.
            #[allow(clippy::arithmetic_side_effects)]
            while docs.len() - start >= LEVEL1_NUM_DOCS as usize {
                let span = &docs[start..start + LEVEL1_NUM_DOCS as usize];
                prev_doc_id = write_level1_span(
                    doc,
                    span,
                    prev_doc_id,
                    &mut level1_last_doc_id,
                    &mut BlockWriteCtx {
                        index_has_freq,
                        norms: field_norms,
                        maxima,
                        skip: &mut skip,
                        scratch: &mut block_scratch,
                    },
                );
                start += LEVEL1_NUM_DOCS as usize;
            }
            // ARITH: same invariant as the level-1 loop above.
            #[allow(clippy::arithmetic_side_effects)]
            while docs.len() - start >= BLOCK_SIZE as usize {
                let block = &docs[start..start + BLOCK_SIZE as usize];
                prev_doc_id = write_full_block(
                    doc,
                    block,
                    prev_doc_id,
                    &mut BlockWriteCtx {
                        index_has_freq,
                        norms: field_norms,
                        maxima,
                        skip: &mut skip,
                        scratch: &mut block_scratch,
                    },
                );
                start += BLOCK_SIZE as usize;
            }
            if start < docs.len() {
                write_tail_block(doc, &docs[start..], prev_doc_id, index_has_freq);
            }
        }
    }

    /// Streams `.doc` (and `.pos`/`.pay`, when the segment has them) to
    /// these outputs from here on: whatever is buffered goes now, and again
    /// whenever a batch of terms leaves more than [`SPILL_BYTES`] buffered,
    /// so a long write holds a batch's postings instead of the files.
    /// [`Self::finish`] then writes the footers, closes the outputs and
    /// returns empty `.doc`/`.pos`/`.pay` buffers.
    pub(crate) fn stream_to(
        &mut self,
        doc: FsIndexOutput,
        pos: Option<FsIndexOutput>,
        pay: Option<FsIndexOutput>,
    ) {
        let mut spill = PostingsSpill {
            doc,
            pos,
            pay,
            doc_bytes: 0,
            pos_bytes: 0,
            pay_bytes: 0,
        };
        spill.write(&mut self.doc, &mut self.pos, &mut self.pay);
        self.spill = Some(spill);
    }

    /// Hands the buffered bytes to the outputs once there are enough of them.
    fn maybe_spill(&mut self) {
        if let Some(spill) = self.spill.as_mut() {
            let buffered = self
                .doc
                .len()
                .saturating_add(self.pos.len())
                .saturating_add(self.pay.len());
            if buffered >= SPILL_BYTES {
                spill.write(&mut self.doc, &mut self.pos, &mut self.pay);
            }
        }
    }

    /// `close`: footers and the `.psm` impact maxima and lengths.
    ///
    /// # Errors
    /// Closing a [`Self::stream_to`] output.
    pub(crate) fn finish(self) -> Result<PostingsFiles> {
        let Self {
            mut doc,
            mut pos,
            mut pay,
            maxima,
            any_positions,
            any_offsets_or_payloads,
            segment_id,
            segment_suffix,
            spill,
            ..
        } = self;
        let initial = 256;
        // The files' lengths, footers included, for `.psm`.
        let (doc_len, pos_len, pay_len) = if let Some(mut spill) = spill {
            spill.write(&mut doc, &mut pos, &mut pay);
            // Empty, and without the capacity a large batch grew them to.
            (doc, pos, pay) = (Vec::new(), Vec::new(), Vec::new());
            let footer = |out: &mut FsIndexOutput| {
                codec_util::write_footer_to(out);
                out.file_pointer()
            };
            let doc_len = footer(&mut spill.doc);
            let pos_len = spill.pos.as_mut().map_or(0, footer);
            let pay_len = spill.pay.as_mut().map_or(0, footer);
            let mut closed = Vec::with_capacity(3);
            closed.push(spill.doc);
            closed.extend(spill.pos);
            closed.extend(spill.pay);
            for out in closed {
                out.close()?;
            }
            (doc_len, pos_len, pay_len)
        } else {
            codec_util::write_footer(&mut doc);
            if any_positions {
                codec_util::write_footer(&mut pos);
            }
            if any_offsets_or_payloads {
                codec_util::write_footer(&mut pay);
            }
            (doc.len() as u64, pos.len() as u64, pay.len() as u64)
        };
        // `.psm`, mirroring `Lucene104PostingsWriter.close()`: the impact maxima
        // first, then each postings file's length *including* its footer (Java
        // records `getFilePointer()` after writing the footer). Lucene sizes its
        // impact-decoding buffers from these maxima before reading any block, so
        // they must cover what was actually written.
        let mut psm = Vec::with_capacity(initial);
        codec_util::write_index_header(
            &mut psm,
            META_CODEC,
            DOC_VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
        psm.write_i32(maxima.num_impacts_level0);
        psm.write_i32(maxima.impact_bytes_level0);
        psm.write_i32(maxima.num_impacts_level1);
        psm.write_i32(maxima.impact_bytes_level1);
        psm.write_i64(doc_len as i64);
        if any_positions {
            psm.write_i64(pos_len as i64);
            if any_offsets_or_payloads {
                psm.write_i64(pay_len as i64);
            }
        }
        codec_util::write_footer(&mut psm);

        Ok(PostingsFiles { doc, psm, pos, pay })
    }
}

/// Validates one field's structural invariants (sortedness, `docFreq`/
/// `totalTermFreq` bounds, positions shape) — the exact same checks
/// `write_single_field` ran inline before this became a per-field helper
/// shared by [`write_fields`]'s loop.
fn validate_field<'a, T: TermSource>(input: &FieldPostingsInput<'a, T>) -> Result<FieldStats<'a>> {
    if !matches!(
        input.index_options,
        IndexOptions::Docs
            | IndexOptions::DocsAndFreqs
            | IndexOptions::DocsAndFreqsAndPositions
            | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
            | IndexOptions::DocsAndCustomFreqs
    ) {
        return Err(Error::UnsupportedIndexOptions(input.index_options));
    }
    if input.terms.is_empty() {
        return Err(Error::EmptyTerms);
    }
    for (i, w) in input.terms.windows(2).enumerate() {
        if w[0].term() >= w[1].term() {
            // ARITH: `i` indexes `terms.windows(2)`, so `i + 1 < terms.len()`.
            #[allow(clippy::arithmetic_side_effects)]
            let at = i + 1;
            return Err(Error::TermsNotSorted(at));
        }
    }
    let index_has_positions = input.index_options.subsumes_positions();
    let index_has_offsets = input.index_options.subsumes_offsets();
    let mut stats = FieldStats::new(input.terms.len());
    for (i, t) in input.terms.iter().enumerate() {
        let docs = t.docs();
        if docs.is_empty() {
            return Err(Error::EmptyPostings(i));
        }
        stats.push(
            blocktree_writer::BlockTerm {
                bytes: t.term(),
                doc_freq: docs.len() as i32,
                total_term_freq: docs.iter().map(|&(_, f)| i64::from(f)).sum(),
            },
            match docs {
                [(only, _)] => *only,
                _ => -1,
            },
        );
        // Checked on the first doc only: the ascending check below carries it
        // to the rest. This is what bounds every `docID - lastDocID` delta
        // below to `1..=i32::MAX`, and it is the same rejection
        // `Lucene104PostingsWriter.startDoc` makes.
        if docs[0].0 < 0 {
            return Err(Error::NegativeDocId { index: i });
        }
        for (j, &(_, freq)) in docs.iter().enumerate() {
            if freq < 1 {
                return Err(Error::NonPositiveFreq { index: i });
            }
            // ARITH: guarded by `j > 0`.
            #[allow(clippy::arithmetic_side_effects)]
            let out_of_order = j > 0 && docs[j - 1].0 >= docs[j].0;
            if out_of_order {
                return Err(Error::DocIdsNotSorted { index: i });
            }
        }
        if index_has_positions {
            if t.position_lists() != docs.len() {
                return Err(Error::MissingPositions {
                    index: i,
                    doc_index: t.position_lists(),
                });
            }
            for (j, &(_, freq)) in docs.iter().enumerate() {
                let positions = t.positions(j);
                if positions.len() != freq as usize {
                    return Err(Error::PositionsFreqMismatch {
                        index: i,
                        doc_index: j,
                        positions: positions.len(),
                        freq,
                    });
                }
                // A repeated position is legal: a token with a position
                // increment of 0 that repeats the term (Java's indexing chain
                // allows it, and `addPosition` writes a delta of 0).
                if positions.windows(2).any(|w| w[0] > w[1]) {
                    return Err(Error::PositionsNotAscending {
                        index: i,
                        doc_index: j,
                    });
                }
                // As with doc IDs: the first occurrence plus ascendingness
                // bounds every position to `0..=i32::MAX`, which is what makes
                // the `p - prev` deltas in `write_position_tail` provably
                // in-range. `Lucene104PostingsWriter.addPosition` rejects the
                // same input.
                if positions.first().is_some_and(|&p| p < 0) {
                    return Err(Error::NegativePosition {
                        index: i,
                        doc_index: j,
                    });
                }
            }
            if index_has_offsets {
                if t.offset_lists() != docs.len() {
                    return Err(Error::MissingOffsets {
                        index: i,
                        doc_index: t.offset_lists(),
                    });
                }
                for (j, &(_, freq)) in docs.iter().enumerate() {
                    let doc_offsets = t.offsets(j);
                    if doc_offsets.len() != freq as usize {
                        return Err(Error::OffsetsFreqMismatch {
                            index: i,
                            doc_index: j,
                            offsets: doc_offsets.len(),
                            freq,
                        });
                    }
                    let mut last_start_offset = 0i32;
                    for (k, &(start_offset, end_offset)) in doc_offsets.iter().enumerate() {
                        if start_offset < last_start_offset || end_offset < start_offset {
                            return Err(Error::InvalidOffsets {
                                index: i,
                                doc_index: j,
                                occurrence: k,
                            });
                        }
                        last_start_offset = start_offset;
                    }
                }
            }
            if input.has_payloads {
                // ARITH: `freq >= 1` and `docs.len()` is bounded by the
                // segment, so the sum is bounded by the total occurrence
                // count of one term -- a `usize` cannot overflow summing
                // counts over a structure that is already in memory.
                #[allow(clippy::arithmetic_side_effects)]
                let total_term_freq: usize = docs.iter().map(|&(_, freq)| freq as usize).sum();
                if t.payload_lengths().len() != total_term_freq {
                    return Err(Error::PayloadsFreqMismatch {
                        index: i,
                        payload_lengths: t.payload_lengths().len(),
                        total_term_freq,
                    });
                }
                // ARITH: a `u32` widened to `usize` cannot overflow the sum
                // unless the run has more occurrences than `usize::MAX / 2^32`
                // -- more than the address space holds -- and the sum is
                // checked against `payload_bytes.len()` immediately below.
                #[allow(clippy::arithmetic_side_effects)]
                let expected: usize = t.payload_lengths().iter().map(|&l| l as usize).sum();
                if let Some((k, &length)) = t
                    .payload_lengths()
                    .iter()
                    .enumerate()
                    .find(|&(_, &l)| l > i32::MAX as u32)
                {
                    return Err(Error::PayloadLengthTooLarge {
                        index: i,
                        occurrence: k,
                        length,
                    });
                }
                if t.payload_bytes().len() != expected {
                    return Err(Error::PayloadBytesMismatch {
                        index: i,
                        payload_bytes: t.payload_bytes().len(),
                        expected,
                    });
                }
            }
        }
    }
    Ok(stats)
}

/// One field's per-term statistics, gathered by [`validate_field`] in the
/// pass that already reads every term's `docs`: the block tree, `encodeTerm`
/// and the `.tmd` sums read these compact arrays instead of walking each
/// term's `docs` again -- for a million-term field every such walk is a
/// million cache misses, where Java streams each term once.
enum FieldStats<'a> {
    /// Incremental merge: no heap allocation for the current term's stats.
    One {
        term: blocktree_writer::BlockTerm<'a>,
        singleton: i32,
    },
    /// Flush: compact arrays over the entire field.
    Many {
        block_terms: Vec<blocktree_writer::BlockTerm<'a>>,
        singletons: Vec<i32>,
    },
}

impl<'a> FieldStats<'a> {
    fn new(terms: usize) -> Self {
        if terms == 1 {
            Self::One {
                term: blocktree_writer::BlockTerm {
                    bytes: &[],
                    doc_freq: 0,
                    total_term_freq: 0,
                },
                singleton: -1,
            }
        } else {
            Self::Many {
                block_terms: Vec::with_capacity(terms),
                singletons: Vec::with_capacity(terms),
            }
        }
    }

    /// Called once per input term by validate_field; the One variant is
    /// selected only when that loop has exactly one iteration.
    fn push(&mut self, term: blocktree_writer::BlockTerm<'a>, singleton: i32) {
        match self {
            Self::One {
                term: value,
                singleton: doc,
            } => {
                *value = term;
                *doc = singleton;
            }
            Self::Many {
                block_terms,
                singletons,
            } => {
                block_terms.push(term);
                singletons.push(singleton);
            }
        }
    }

    fn slices(&self) -> (&[blocktree_writer::BlockTerm<'a>], &[i32]) {
        match self {
            Self::One { term, singleton } => {
                (std::slice::from_ref(term), std::slice::from_ref(singleton))
            }
            Self::Many {
                block_terms,
                singletons,
            } => (block_terms, singletons),
        }
    }
}

/// `Lucene104PostingsWriter.writeVInt15`'s write-side companion to
/// `crate::postings::read_vint15` (the 2-byte fast path for `0..=0x7FFF`,
/// else a negative `i16` flag carrying the low 15 bits plus a following vint
/// for the high bits).
fn write_vint15(out: &mut Vec<u8>, value: i32) {
    if (0..=0x7FFF).contains(&value) {
        out.write_i16(value as i16);
    } else {
        out.write_i16((0x8000 | (value & 0x7FFF)) as i16);
        out.write_vint(value >> 15);
    }
}

/// `Lucene104PostingsWriter.writeVLong15`'s write-side companion to
/// `crate::postings::read_vlong15`, the `long`-widening sibling of
/// [`write_vint15`].
fn write_vlong15(out: &mut Vec<u8>, value: i64) {
    if (0..=0x7FFF).contains(&value) {
        out.write_i16(value as i16);
    } else {
        out.write_i16((0x8000 | (value & 0x7FFF)) as i16);
        out.write_vlong(value >> 15);
    }
}

/// The four figures `.psm` records, accumulated as blocks are written.
///
/// Lucene's reader sizes its impact-decoding buffers from these before it
/// reads a single block, so they are not bookkeeping: understating
/// `impact_bytes_level0` gives `readBytes` a buffer shorter than the region it
/// is asked to fill.
#[derive(Default)]
struct PostingsMaxima {
    num_impacts_level0: i32,
    impact_bytes_level0: i32,
    num_impacts_level1: i32,
    impact_bytes_level1: i32,
}

impl PostingsMaxima {
    fn observe_level0(&mut self, num_impacts: i32, num_bytes: usize) {
        self.num_impacts_level0 = self.num_impacts_level0.max(num_impacts);
        self.impact_bytes_level0 = self.impact_bytes_level0.max(num_bytes as i32);
    }

    fn observe_level1(&mut self, num_impacts: i32, num_bytes: usize) {
        self.num_impacts_level1 = self.num_impacts_level1.max(num_impacts);
        self.impact_bytes_level1 = self.impact_bytes_level1.max(num_bytes as i32);
    }
}

/// The per-term state both `.doc` block writers thread through: what the
/// field indexes, its norms, the running `.psm` maxima, and the `.pos`/`.pay`
/// skip writer.
///
/// `Lucene104PostingsWriter` holds all four as fields of itself and mutates
/// them in place across `flushDocBlock`/`writeLevel1SkipData`. This writer
/// builds each file whole rather than interleaving them, so the same state
/// travels as one borrow instead of four parameters.
struct BlockWriteCtx<'n, 'p> {
    index_has_freq: bool,
    norms: Option<&'n [i64]>,
    maxima: &'n mut PostingsMaxima,
    skip: &'n mut PosSkipWriter<'p>,
    scratch: &'n mut BlockScratch,
}

/// [`write_full_block`]'s buffers, cleared and refilled per block rather
/// than allocated per block.
#[derive(Default)]
struct BlockScratch {
    rest: Vec<u8>,
    impacts: Vec<u8>,
    header: Vec<u8>,
}

/// Writes one full 256-doc `.doc` block — a level-0 skip header
/// (`level0NumBytes` skip pointer, `docDelta`, `blockLength`, an always-empty
/// impacts region) followed by the doc-delta/freq body — the exact
/// write-side inverse of `crate::postings::read_full_block_header`/
/// `decode_full_block_body`. The header's pos/pay skip fields — present on
/// the wire exactly when the field indexes positions *and* has freqs — are
/// written by `skip` ([`PosSkipWriter::write_level0`]), which is a no-op for
/// a field without positions. `block` must be exactly `BLOCK_SIZE` (256)
/// `(doc_id, freq)` pairs, ascending. Returns `block`'s last doc ID, which
/// the caller threads through as `prev_doc_id` for the next full block or
/// the trailing tail block (`Lucene104PostingsReader.prefixSum`'s running
/// per-term base).
///
/// Doc deltas pick one of the three shapes `decode_full_block_body` can
/// parse, using the exact same heuristic as
/// `Lucene104PostingsWriter.flushDocBlock`:
///
/// - `docRange == BLOCK_SIZE` (every delta is 1, i.e. all 256 docs in the
///   block are consecutive): the `bitsPerValue == 0` marker, no body bytes.
/// - Otherwise, compare the packed-`ForUtil` cost at the *next* bits-per-value
///   step (`min(32, bitsPerValue + 1) * BLOCK_SIZE` bits) against the dense
///   bit-set cost (`bits2words(docRange) * 64` bits, one `i64` word per 64
///   possible doc IDs spanned). If the *next-tier* packed cost is no smaller
///   than the bit-set cost, use the bit set (`bitsPerValue < 0`, `numLongs =
///   -bitsPerValue` words follow) -- comparing against the next tier rather
///   than the current one (and taking the plain packed array on an exact
///   tie) is what slightly biases this toward the bit set, matching Java
///   exactly. Otherwise fall back to the plain positive-`bitsPerValue`
///   packed array.
///
/// Freqs (when `index_has_freq`) go through
/// `for_util::pfor_encode` directly — its on-wire token/body shape is byte-
/// identical to what `for_util::pfor_decode` (called from
/// `decode_full_block_body`) expects, so no re-derivation of that format
/// happens here.
fn write_full_block(
    out: &mut Vec<u8>,
    block: &[(i32, i32)],
    prev_doc_id: i32,
    ctx: &mut BlockWriteCtx<'_, '_>,
) -> i32 {
    let BlockWriteCtx {
        index_has_freq,
        norms,
        maxima,
        skip,
        scratch,
    } = ctx;
    let BlockScratch {
        rest,
        impacts,
        header,
    } = &mut **scratch;
    rest.clear();
    impacts.clear();
    header.clear();
    let (index_has_freq, norms) = (*index_has_freq, *norms);
    debug_assert_eq!(block.len(), BLOCK_SIZE as usize);
    // `addPosition` has run for every occurrence of this block's documents by
    // the time `flushDocBlock` samples the `.pos`/`.pay` pointers.
    skip.add_block_docs(block);

    // Everything from here down is what `blockLength` measures (i.e. what
    // `read_full_block_header` reads as `body_end - r.position()`
    // immediately after `blockLength` itself) -- build it in a scratch
    // buffer first so `blockLength`'s value is known before the header is
    // written.
    if index_has_freq {
        // `flushDocBlock`'s level-0 impacts: the block's competitive
        // `(freq, norm)` frontier, from `CompetitiveImpactAccumulator`. With
        // no norms supplied every document's norm is `1`, and the frontier
        // collapses to the single `(maxFreq, 1)` pair this writer emitted
        // before item 18 -- so the normless path is byte-identical to what it
        // always wrote.
        //
        // Impacts bound scores for dynamic pruning: a bound that is too high
        // only costs pruning, while one that is too low **drops real hits**.
        // Norm 1 is the shortest field and therefore the highest-scoring one,
        // which is why the normless bound is safe; the accumulator is what
        // keeps the with-norms bound safe, by taking the frontier rather than
        // the (possibly unrealised) `(maxFreq, minNorm)` corner.
        //
        // An *empty* region is not the conservative choice it looks like:
        // Lucene rejects the segment outright with "Got empty list of impacts
        // on level 0".
        let block_impacts = competitive_impacts(block, norms);
        // `validate_field` rejects `freq < 1`, so every block has at least one
        // document and its frontier at least one pair. Asserted here rather
        // than inferred from a validator several hundred lines up: an empty
        // impacts region is what Lucene rejects outright.
        debug_assert!(!block_impacts.is_empty());
        write_impacts(&block_impacts, impacts);
        maxima.observe_level0(block_impacts.len() as i32, impacts.len());
        rest.write_vint(impacts.len() as i32);
        rest.write_bytes(impacts);
        // `.pos`/`.pay` skip pointers, immediately after the impacts run and
        // still inside `writeFreqs` -- `Lucene104PostingsWriter.java:413-422`.
        skip.write_level0(rest);
    }
    // Java's `numSkipBytes = level0Output.size()` is sampled exactly here --
    // after the impacts region *and* the pos/pay skip fields, before the doc
    // deltas and freqs are appended.
    let impacts_region_len = rest.len() as i64;

    let last_doc_id = encode_doc_block(block, prev_doc_id, index_has_freq, rest);

    // `numSkipBytes` spans the two header fields below plus the impacts
    // region -- landing a reader that skips this block on the first doc-delta
    // byte. This port's own reader parses the field and ignores it, deriving
    // the same position from `blockLength`; real Lucene's `skipLevel0To`
    // seeks by it, and a zero here sends it *backwards* to re-read the header
    // as block data and then off the end of the file. Measure the fields by
    // building them first, as Java does with its `scratchOutput`.
    // Java writes `docBuffer[BLOCK_SIZE - 1] - level0LastDocID` as an `int`;
    // see the per-doc delta in `encode_doc_block`.
    write_vint15(header, last_doc_id.wrapping_sub(prev_doc_id));
    write_vlong15(header, rest.len() as i64);
    // ARITH: both terms are lengths of in-memory scratch buffers this call
    // just built -- a few KiB each, and both bounded by `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    let num_skip_bytes = impacts_region_len + header.len() as i64;
    out.write_vlong(num_skip_bytes);
    out.write_bytes(header);
    out.write_bytes(rest);

    last_doc_id
}

/// A full block's documents and frequencies as its body holds them after
/// the impacts: the `bitsPerValue` token, the doc deltas from
/// `prev_doc_id` (packed, a bit set, or nothing when consecutive), then the
/// frequencies when `index_has_freq`. Returns the block's last document.
/// `block` is `BLOCK_SIZE` ascending `(doc, freq)` pairs, each document
/// after `prev_doc_id`.
///
/// # Panics
/// When `block` is not `BLOCK_SIZE` pairs long.
pub fn encode_doc_block(
    block: &[(i32, i32)],
    prev_doc_id: i32,
    index_has_freq: bool,
    rest: &mut Vec<u8>,
) -> i32 {
    assert_eq!(block.len(), BLOCK_SIZE as usize, "a full block's documents");
    let mut deltas = [0u32; for_util::BLOCK_SIZE];
    let mut prev = prev_doc_id;
    let mut max_delta = 0u32;
    for (i, &(doc_id, _)) in block.iter().enumerate() {
        // `validate_field` bounds every doc ID to `0..=i32::MAX` and makes
        // them strictly ascending, and `prev` starts at -1, so the true delta
        // is in `1..=i32::MAX + 1`. Only the very last of those overflows an
        // `i32`, and Java's `docID - lastDocID` is an `int` subtraction that
        // wraps to the same bit pattern this `as u32` then reads.
        let delta = doc_id.wrapping_sub(prev) as u32;
        deltas[i] = delta;
        max_delta = max_delta.max(delta);
        prev = doc_id;
    }
    // `bits_required` returns 0 only for an all-zero input; every delta here
    // is `>= 1` (ascending, no duplicates), so `max_delta >= 1` and this is
    // always `>= 1` in practice -- `.max(1)` just keeps the invariant
    // explicit rather than relying on that fact silently.
    let bits_per_value = for_util::bits_required(max_delta).max(1);
    // ARITH: `write_full_block` is only ever called with a `BLOCK_SIZE`-long
    // slice, so `block` is non-empty.
    #[allow(clippy::arithmetic_side_effects)]
    let last_doc_id = block[block.len() - 1].0;
    // See the per-doc delta above for why this is a `wrapping_sub`.
    let doc_range = last_doc_id.wrapping_sub(prev_doc_id) as u32;
    // `FixedBitSet.bits2words`: ceil(doc_range / 64), doc_range >= 1 here.
    let num_bit_set_longs = doc_range.div_ceil(64);
    // ARITH: the left factor is capped at 32 by the `.min(32)`, so the
    // product is at most `32 * 256 = 8192`.
    #[allow(clippy::arithmetic_side_effects)]
    let num_bits_next_bits_per_value = bits_per_value.saturating_add(1).min(32) * BLOCK_SIZE as u32;
    if doc_range == BLOCK_SIZE as u32 {
        // Every delta is 1: all 256 docs in the block are consecutive.
        rest.write_byte(0);
    // `Lucene104PostingsWriter.flushDocBlock` in **Lucene 10.5.0**, verbatim:
    //
    //   } else if (numBitsNextBitsPerValue <= docRange) {
    //
    // i.e. the packed shape wins only when the *next* bits-per-value tier
    // would cost no more than the bit set's *unrounded* `docRange` bits.
    // Comparing against the next tier (rather than the current one) is the
    // bias Lucene documents:
    //
    //   we make the decision based on storage requirements, picking the bit
    //   set approach whenever it's more storage-efficient than the next number
    //   of bits per value (which effectively slightly biases towards the bit
    //   set approach)
    //
    //   FOR makes #nextDoc() a bit faster while the bit set approach makes
    //   #advance() usually faster and #intoBitSet() much faster
    //
    // **Version note.** Lucene `main` (post-10.5.0) loosened the right-hand
    // side to `numBitSetLongs * Long.SIZE`, i.e. the bit set's real
    // rounded-up-to-whole-words size, which is `>= docRange` and so picks
    // packed FOR for every block in the band
    // `doc_range < num_bits_next <= ceil(doc_range/64)*64`. This port pins
    // **10.5.0** (see `AGENTS.md` and `scripts/lib-lucene-jars.sh`), so it
    // must use `doc_range`. Both shapes are legal and this port's reader
    // takes either, so no round-trip test can tell them apart -- only
    // `full_block_encoding_choice_matches_lucene_in_the_disputed_band`,
    // which asserts the chosen token, can.
    } else if num_bits_next_bits_per_value <= doc_range {
        rest.write_byte(bits_per_value as u8);
        for_util::for_encode(&mut deltas, bits_per_value, rest);
    } else {
        // Dense unary bit-set encoding: doc IDs are the set-bit positions
        // (ascending) in a `num_bit_set_longs`-word bitset based at
        // `prev_doc_id + 1`, matching `FixedBitSet`'s word/bit layout
        // (word = bit_index / 64, bit = bit_index % 64).
        // At most 128 words: this branch needs `doc_range < 32 * 256`.
        let mut all_words = [0u64; 128];
        let words = &mut all_words[..num_bit_set_longs as usize];
        let mut s: i64 = -1;
        // ARITH: `s` accumulates 256 deltas, each below 2^32, so it stays
        // under 2^41. It ends at `doc_range - 1` (the deltas telescope), and
        // `num_bit_set_longs = ceil(doc_range / 64)`, so `s / 64` indexes
        // `words` in range; every delta is `>= 1` (`validate_field` makes doc
        // IDs strictly ascending), so `s >= 0` from the first iteration and
        // `s % 64` is a non-negative shift amount.
        #[allow(clippy::arithmetic_side_effects)]
        for &delta in deltas.iter() {
            s += delta as i64;
            words[(s / 64) as usize] |= 1u64 << (s % 64);
        }
        // ARITH: this branch runs only when `num_bits_next <= doc_range` is
        // false, i.e. `doc_range < 32 * 256`, so `num_bit_set_longs =
        // ceil(doc_range / 64) <= 128` and the negation is in `-128..=-1` --
        // exactly the range `read_full_block_header` decodes as a bit set.
        #[allow(clippy::arithmetic_side_effects)]
        let token = (-(num_bit_set_longs as i32)) as u8;
        rest.write_byte(token);
        for word in words.iter() {
            rest.write_i64(*word as i64);
        }
    }

    if index_has_freq {
        let mut freqs = [0u32; for_util::BLOCK_SIZE];
        for (i, &(_, freq)) in block.iter().enumerate() {
            freqs[i] = freq as u32;
        }
        for_util::pfor_encode(&mut freqs, rest);
    }

    last_doc_id
}

/// Writes one level-1 skip entry followed by the `LEVEL1_FACTOR` (32) full
/// level-0 blocks it covers — the exact write-side inverse of
/// `crate::postings::read_level1_entry` (shared by `DocInput::read_postings`
/// and `LazyDocsCursor::skip_level1_to`). `span` must be exactly
/// `LEVEL1_NUM_DOCS` (8192) `(doc_id, freq)` pairs, ascending. `prev_doc_id`
/// is the running per-term doc-ID base threaded in from whatever preceded
/// this span (`-1` for the first span, or the previous span's last doc ID).
/// `level1_last_doc_id` is the running level-1 accumulator the read side
/// also keeps (`LazyDocsCursor::level1_last_doc_id`, starts at `-1`, `+=
/// doc_delta` per entry) — passed by `&mut` so the caller can thread it
/// across multiple spans for the same term. Returns this span's last doc ID,
/// for the caller to thread as `prev_doc_id` into the next span or the
/// trailing full-block/tail-block loop.
///
/// The level-1 entry's own fields, in wire order: `doc_delta` (vint, `this
/// span's last doc ID - *level1_last_doc_id` before update), the span's
/// byte length (vlong, needed by the reader to compute `level1DocEndFP`
/// without decoding the span), then — only when `index_has_freq` — a
/// `skip1EndFP` `i16` (byte length from right after it to the end of this
/// entry's freq-gated metadata) and a `numImpactBytes` `i16`. `skip1EndFP`
/// spans `numImpactBytes`'s own two bytes, the impact bytes, and the
/// `indexHasPos`-gated pos/pay sub-fields (written by `skip`,
/// [`PosSkipWriter::write_level1`], and absent for a field without
/// positions) -- exactly the region `crate::postings::read_level1_entry`
/// reads before checking it landed on `skip1EndFP`.
///
/// The span's `.pos`/`.pay` pointers are sampled *after* its 32 blocks have
/// been built, because `Lucene104PostingsWriter` calls `writeLevel1SkipData`
/// from the 32nd `flushDocBlock`: the entry is written before the span's
/// bytes but describes the state at its end.
fn write_level1_span(
    out: &mut Vec<u8>,
    span: &[(i32, i32)],
    prev_doc_id: i32,
    level1_last_doc_id: &mut i32,
    ctx: &mut BlockWriteCtx<'_, '_>,
) -> i32 {
    let (index_has_freq, norms) = (ctx.index_has_freq, ctx.norms);
    debug_assert_eq!(span.len(), LEVEL1_NUM_DOCS as usize);

    // Build the span's 32 full blocks into a scratch buffer first so the
    // level-1 entry's byte-length field is known before the entry header is
    // written (same "measure by building into scratch first" approach
    // `write_full_block` uses for `blockLength`).
    let mut span_bytes = Vec::new();
    let mut prev = prev_doc_id;
    for block in span.chunks(BLOCK_SIZE as usize) {
        prev = write_full_block(&mut span_bytes, block, prev, ctx);
    }
    let last_doc_id = prev;

    // This span's own impacts, on the same basis as [`write_full_block`]'s --
    // accumulated over the whole 8192-document span, so the frontier bounds
    // every level-0 block beneath it, as a level-1 impact must.
    let mut level1_impacts = Vec::new();
    // `numImpactBytes` is sampled here, before the pos/pay sub-fields are
    // appended to the same scratch buffer (`Lucene104PostingsWriter.java:
    // 507-521`), so it counts the impact bytes only.
    let mut level1_scratch = Vec::new();
    if index_has_freq {
        let span_impacts = competitive_impacts(span, norms);
        // See `write_full_block`'s identical assertion.
        debug_assert!(!span_impacts.is_empty());
        write_impacts(&span_impacts, &mut level1_impacts);
        ctx.maxima
            .observe_level1(span_impacts.len() as i32, level1_impacts.len());
        level1_scratch.extend_from_slice(&level1_impacts);
        ctx.skip.write_level1(&mut level1_scratch);
    }

    // `read_level1_entry` computes `doc_end_fp` as this vlong's value added
    // to `r.position()` measured right after the vlong itself -- i.e.
    // *before* the freq-gated `skip1EndFP`/`numImpactBytes` fields below are
    // read. So the vlong must span every byte from there through the end of
    // the whole entry+span, not just `span_bytes` alone: the freq-gated
    // header contributes `2 (skip1EndFP) + 2 (numImpactBytes) + the impact
    // bytes themselves` whenever `index_has_freq`.
    // ARITH: `level1_scratch` and `span_bytes` are in-memory scratch buffers
    // this call just built; both lengths are bounded by `isize::MAX` and the
    // constants added to them are 2 and 4.
    #[allow(clippy::arithmetic_side_effects)]
    let freq_header_len: usize = if index_has_freq {
        4 + level1_scratch.len()
    } else {
        0
    };
    // Java's `writeVInt(docID - level1LastDocID)` is an `int` subtraction; see
    // `write_full_block`'s per-doc delta for why a valid caller never wraps.
    let doc_delta = last_doc_id.wrapping_sub(*level1_last_doc_id);
    out.write_vint(doc_delta);
    // ARITH: both are lengths of in-memory scratch buffers built by this
    // call, each bounded by `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    let doc_end_delta = (freq_header_len + span_bytes.len()) as i64;
    out.write_vlong(doc_end_delta);
    if index_has_freq {
        // skip1EndFP delta: `numImpactBytes`'s own 2 bytes plus everything
        // that follows it inside the entry -- the impact bytes and then the
        // pos/pay sub-fields.
        // ARITH: as above.
        #[allow(clippy::arithmetic_side_effects)]
        let skip1_end_fp = (2 + level1_scratch.len()) as i16;
        out.write_i16(skip1_end_fp);
        out.write_i16(level1_impacts.len() as i16);
        out.write_bytes(&level1_scratch);
    }
    out.write_bytes(&span_bytes);

    *level1_last_doc_id = last_doc_id;
    last_doc_id
}

/// Writes one term's `.doc` tail-block bytes (the `docFreq % BLOCK_SIZE`
/// remainder, or the whole term when `docFreq < BLOCK_SIZE`) — the exact
/// inverse of `crate::postings::read_tail_block`. `prev_doc_id` is `-1` when
/// there are no preceding full blocks for this term, or the last full
/// block's last doc ID otherwise (full-block chaining within one term, see
/// [`write_full_block`]) — a term's postings never share a running doc-ID
/// base with another *term*, only across blocks within the same term.
fn write_tail_block(
    out: &mut Vec<u8>,
    docs: &[(i32, i32)],
    prev_doc_id: i32,
    index_has_freq: bool,
) {
    let mut raw = Vec::with_capacity(docs.len());
    let mut prev = prev_doc_id;
    for &(doc_id, freq) in docs {
        // See `write_full_block`'s per-doc delta.
        let delta = doc_id.wrapping_sub(prev) as u32;
        prev = doc_id;
        if index_has_freq {
            raw.push((delta << 1) | if freq == 1 { 1 } else { 0 });
        } else {
            raw.push(delta);
        }
    }
    out.write_group_vints(&raw);
    if index_has_freq {
        for &(_, freq) in docs {
            if freq != 1 {
                out.write_vint(freq);
            }
        }
    }
}

/// Writes one term's whole `.pos` (and, when `has_offsets`/`has_payloads`,
/// `.pay`) byte range: zero or more full 256-position `PForUtil` blocks
/// ([`write_full_position_block`]/[`write_full_payload_length_block`]/
/// [`write_full_offset_block`]) followed by a group-varint-free vint tail for
/// the `total_term_freq % BLOCK_SIZE` remainder — the exact write-side
/// inverse of `crate::postings::read_positions`'s `num_full_blocks`/
/// `tail_count` split. `positions` is one `Vec<i32>` per doc (parallel to
/// that term's `docs`), each holding the doc's absolute, ascending occurrence
/// positions — see [`TermPostings`]'s `positions` field doc comment for the
/// exact input shape. `offsets`/`payloads` are only consulted when
/// `has_offsets`/`has_payloads` respectively, in which case each must be the
/// same shape (one entry per doc, matching `positions[i].len()`) — see
/// [`TermPostings`]'s `offsets`/`payloads` field doc comments.
///
/// Position deltas (and, when present, payload lengths/bytes and offset
/// start-deltas/lengths) are buffered into one flat, cross-doc sequence first
/// (resetting to each doc's absolute first position/offset at that doc's
/// first occurrence, exactly like `read_positions`'s own flat
/// `pos_deltas`/`payload_*`/`offset_*` before it re-chops the sequence by
/// `freqs`) so that a 256-occurrence chunk spanning a doc boundary is still
/// encoded as a single full block — matching real Lucene's own
/// `addPosition`/`posBufferUpto == BLOCK_SIZE` flush timing, which is
/// entirely doc-boundary-agnostic (`Lucene104PostingsWriter.java:315-355`).
///
/// Wire order when both payloads and offsets are present, in both the full
/// block and vint-tail paths, is always **payload fields before offset
/// fields** — `Lucene104PostingsWriter.addPosition`
/// (`Lucene104PostingsWriter.java:316-353`, full block) and `finishTerm`
/// (`Lucene104PostingsWriter.java:598-633`, vint tail) both write the payload
/// length/bytes immediately after the position delta and before any offset
/// fields — matched exactly here and by `crate::postings::read_positions`'s
/// existing (unmodified) decode order.
///
/// Returns this term's [`PositionLayout`]: the `.pos`/`.pay` file pointer at
/// every full-block boundary (what `.doc`'s level-0/level-1 skip records
/// point at, see [`PosSkipWriter`]) plus its `lastPosBlockOffset` — how many
/// `.pos` bytes the full blocks took, i.e. the offset *within this term's
/// `.pos` range* at which the vint tail begins.
/// `Lucene104PostingsWriter.finishTerm` samples exactly that
/// (`posOut.getFilePointer() - posStartFP`, taken **before** the vint tail is
/// written) and `Lucene104PostingsReader.reset` turns it back into
/// `lastPosBlockFP = posStartFP + lastPosBlockOffset`, the file pointer at
/// which `refillPositions` switches from `PForUtil` blocks to
/// `refillLastPositionBlock`. It is load-bearing for real Lucene, and for this
/// port's own skip-driven single-document
/// position walk too, which jumps into the middle of `.pos` and so cannot
/// re-derive the split from a running occurrence count.
#[allow(clippy::too_many_arguments)]
fn write_position_tail<T: TermSource>(
    pos_out: &mut Vec<u8>,
    pay_out: &mut Vec<u8>,
    (pos_base, pay_base): (u64, u64),
    term: &T,
    has_offsets: bool,
    has_payloads: bool,
    scratch: &mut PositionScratch,
) -> PositionLayout {
    let (payload_bytes, payload_lengths) = (term.payload_bytes(), term.payload_lengths());
    let pos_out_start = pos_out.len();
    let pay_out_start = pay_out.len();
    let has_offsets_or_payloads = has_offsets || has_payloads;
    // Only the full blocks' end pointers go in these: most terms have none,
    // and an empty `Vec` costs no allocation.
    let mut pos_block_fp: Vec<u64> = Vec::new();
    let mut pay_block_fp: Vec<u64> = Vec::new();
    let PositionScratch {
        deltas,
        offset_start_deltas,
        offset_lengths,
    } = scratch;
    deltas.clear();
    offset_start_deltas.clear();
    offset_lengths.clear();
    // Borrowed, not rebuilt: `TermPostings` already carries the payload run in
    // exactly the layout this function consumes -- the whole term's payloads
    // concatenated, one length per occurrence -- so there is nothing to
    // flatten here any more.
    for doc_idx in 0..term.position_lists() {
        let doc_positions = term.positions(doc_idx);
        let doc_offsets = if has_offsets {
            term.offsets(doc_idx)
        } else {
            &[]
        };
        let mut prev = 0i32;
        let mut prev_start_offset = 0i32;
        for (occ_idx, &p) in doc_positions.iter().enumerate() {
            // ARITH: `validate_field` bounds positions to `0..=i32::MAX` and
            // makes them strictly ascending within a doc, and `prev` starts at
            // 0, so `p - prev` is in `0..=i32::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            let pos_delta = p - prev;
            deltas.push(pos_delta);
            prev = p;
            if has_offsets {
                let (start_offset, end_offset) = doc_offsets[occ_idx];
                // ARITH: `validate_field` requires `start_offset >=
                // last_start_offset` (from a base of 0, so every offset is
                // non-negative) and `end_offset >= start_offset`, so both
                // differences are in `0..=i32::MAX`.
                #[allow(clippy::arithmetic_side_effects)]
                let (start_delta, length) =
                    (start_offset - prev_start_offset, end_offset - start_offset);
                offset_start_deltas.push(start_delta);
                offset_lengths.push(length);
                prev_start_offset = start_offset;
            }
        }
    }

    // Running index into `payload_bytes` for the full-block path: each
    // block's payload byte run is a variable-length slice (unlike the
    // fixed-256-wide position/offset arrays), so its bounds must be tracked
    // by summing consumed lengths as blocks are emitted, exactly like
    // `Lucene104PostingsWriter`'s own `payloadByteUpto`/`payloadBytesReadUpto`
    // accumulators.
    let mut payload_bytes_upto = 0usize;

    let mut start = 0usize;
    // ARITH: `start = end` at the bottom of the loop, so `start` only advances
    // by the `BLOCK_SIZE` the condition just proved is left; `start <=
    // deltas.len()` at every test.
    #[allow(clippy::arithmetic_side_effects)]
    while deltas.len() - start >= BLOCK_SIZE as usize {
        let end = start + BLOCK_SIZE as usize;
        write_full_position_block(pos_out, &deltas[start..end]);
        if has_payloads {
            let block_len: usize = payload_lengths[start..end]
                .iter()
                .map(|&l| l as usize)
                .sum();
            write_full_payload_length_block(
                pay_out,
                &payload_lengths[start..end],
                // ARITH: `payload_bytes` is the concatenation of every
                // occurrence's payload in order, and `block_len` is the sum of
                // this block's lengths, so the running cursor plus this
                // block's bytes never passes the end of that concatenation.
                #[allow(clippy::arithmetic_side_effects)]
                &payload_bytes[payload_bytes_upto..payload_bytes_upto + block_len],
            );
            // ARITH: same concatenation invariant as the slice above.
            #[allow(clippy::arithmetic_side_effects)]
            {
                payload_bytes_upto += block_len;
            }
        }
        if has_offsets {
            write_full_offset_block(
                pay_out,
                &offset_start_deltas[start..end],
                &offset_lengths[start..end],
            );
        }
        pos_block_fp.push(pos_base.saturating_add(pos_out.len() as u64));
        if has_offsets_or_payloads {
            pay_block_fp.push(pay_base.saturating_add(pay_out.len() as u64));
        }
        start = end;
    }
    // Sampled here, before the vint tail -- exactly where
    // `Lucene104PostingsWriter.finishTerm` samples it.
    // ARITH: `pos_out_start` was `pos_out.len()` on entry and `pos_out` only
    // grows.
    #[allow(clippy::arithmetic_side_effects)]
    let last_pos_block_offset = pos_out.len() - pos_out_start;

    // Vint tail (`refillLastPositionBlock`'s write-side inverse,
    // `Lucene104PostingsWriter.finishTerm`): a plain vint position delta per
    // occurrence (or, when `has_payloads`, the delta shifted left one bit
    // with bit 0 signaling "payload length changed", followed by the new
    // length only when it changed and the payload bytes themselves whenever
    // the (possibly-reused) length is non-zero — `Lucene104PostingsWriter
    // .java:598-617`), then, only when `has_offsets`, an offset
    // start-delta/length pair whose length is only re-written when it
    // changes from the previous occurrence's (`Lucene104PostingsWriter.java:
    // 622-632`). The payload-length and offset-length repeat-suppression
    // states are each independent, term-scoped accumulators (reset at the
    // start of this vint tail, not carried over from any preceding full
    // blocks — full blocks store every length as a raw `PForUtil` value with
    // no suppression at all, so there is nothing to carry over even if there
    // were preceding full blocks).
    let mut last_payload_length = -1i32; // force the first occurrence's length to be written
    let mut last_offset_length = -1i32; // force the first occurrence's length to be written
    let mut payload_bytes_read_upto = payload_bytes_upto;
    for i in start..deltas.len() {
        let delta = deltas[i];
        if has_payloads {
            // `validate_field` proved every length fits an `i32`.
            let length = payload_lengths[i] as i32;
            if length != last_payload_length {
                last_payload_length = length;
                pos_out.write_vint((delta << 1) | 1);
                pos_out.write_vint(length);
            } else {
                pos_out.write_vint(delta << 1);
            }
            if length != 0 {
                let len = length as usize;
                // ARITH: same concatenation invariant as the full-block
                // cursor above; this cursor resumes where that one stopped.
                #[allow(clippy::arithmetic_side_effects)]
                let payload =
                    &payload_bytes[payload_bytes_read_upto..payload_bytes_read_upto + len];
                pos_out.write_bytes(payload);
                // ARITH: same concatenation invariant as the slice above.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    payload_bytes_read_upto += len;
                }
            }
        } else {
            pos_out.write_vint(delta);
        }

        if has_offsets {
            let start_delta = offset_start_deltas[i];
            let length = offset_lengths[i];
            if length != last_offset_length {
                pos_out.write_vint((start_delta << 1) | 1);
                pos_out.write_vint(length);
                last_offset_length = length;
            } else {
                pos_out.write_vint(start_delta << 1);
            }
        }
    }

    PositionLayout {
        pos_start: pos_base.saturating_add(pos_out_start as u64),
        pay_start: has_offsets_or_payloads.then_some(pay_base.saturating_add(pay_out_start as u64)),
        pos_block_fp,
        pay_block_fp,
        last_pos_block_offset,
    }
}

/// Where a term's `.pos`/`.pay` full blocks begin, so `.doc`'s level-0 and
/// level-1 skip records can point at them.
///
/// Real Lucene never needs this: `Lucene104PostingsWriter` interleaves the
/// three files and samples `posOut.getFilePointer()`/`posBufferUpto` live at
/// every `flushDocBlock`. This writer builds each file whole, one term at a
/// time, so the same samples are *reconstructed* — which is exactly as exact,
/// because the flush schedule is arithmetic: a `.pos` block closes every 256
/// occurrences, doc-boundary-agnostic.
/// [`write_position_tail`]'s per-occurrence buffers, kept across a field's
/// terms so each term reuses the last one's allocations instead of growing
/// three fresh vectors push by push.
#[derive(Default)]
struct PositionScratch {
    deltas: Vec<i32>,
    offset_start_deltas: Vec<i32>,
    offset_lengths: Vec<i32>,
}

struct PositionLayout {
    /// The term's `posStartFP`.
    pos_start: u64,
    /// Its `.pay` start; `None` when the field has neither offsets nor
    /// payloads (no `.pay` bytes exist, and no `.pay` skip fields are
    /// written).
    pay_start: Option<u64>,
    /// `.pos` absolute offset after `k + 1` full blocks, for `k` in
    /// `0..num_full_blocks`.
    pos_block_fp: Vec<u64>,
    /// The same for `.pay`, empty when `pay_start` is `None`.
    pay_block_fp: Vec<u64>,
    /// See [`write_position_tail`]'s return doc.
    last_pos_block_offset: usize,
}

/// `Lucene104PostingsWriter`'s `posOut.getFilePointer()` / `posBufferUpto` /
/// `payOut.getFilePointer()` / `payloadByteUpto` at one doc-block boundary.
struct PosSkipSample {
    pos_fp: u64,
    pos_upto: u8,
    pay_fp: u64,
    pay_upto: i32,
}

impl PositionLayout {
    /// The four values `flushDocBlock`/`writeLevel1SkipData` would have
    /// sampled after `occ` occurrences have been buffered.
    ///
    /// `occ` never exceeds this term's `totalTermFreq`, so `q` never exceeds
    /// `num_full_blocks` and every lookup is in range: [`validate_field`] runs
    /// before any of this and enforces `positions[i].len() == freq` for every
    /// document, which is what makes the caller's running occurrence count and
    /// this layout's block count the same quantity. All three lookups are
    /// still written as fallible ones rather than one panicking index beside
    /// two checked ones -- if that invariant is ever broken the failure should
    /// be a wrong pointer a differential test catches, not a panic inside a
    /// writer.
    ///
    /// `payload_lengths` is the term's per-occurrence payload lengths, empty
    /// when the field has no payloads (`payloadByteUpto` then stays `0`).
    fn sample(&self, occ: u64, payload_lengths: &[u32]) -> PosSkipSample {
        // ARITH: `BLOCK_SIZE` is the non-zero constant 256, so neither the
        // division nor the remainder can trap.
        #[allow(clippy::arithmetic_side_effects)]
        let (q, r) = (
            (occ / BLOCK_SIZE as u64) as usize,
            (occ % BLOCK_SIZE as u64) as u8,
        );
        // `.pos`/`.pay` offset after `q` full blocks.
        let after = |start: u64, ends: &[u64]| match q.checked_sub(1) {
            None => start,
            Some(k) => ends.get(k).copied().unwrap_or(0),
        };
        PosSkipSample {
            pos_fp: after(self.pos_start, &self.pos_block_fp),
            pos_upto: r,
            pay_fp: self
                .pay_start
                .map_or(0, |start| after(start, &self.pay_block_fp)),
            pay_upto: payload_byte_upto(payload_lengths, occ),
        }
    }
}

/// `payloadByteUpto` once `occ` occurrences are buffered: the payload bytes
/// of the occurrences since the last full 256-occurrence block, since
/// `addPosition`'s flush resets it to `0` the moment a block fills. Summed
/// at each (rare) skip sample instead of kept per occurrence for every term
/// of the field until its `.doc` pass.
// ARITH: `occ % BLOCK_SIZE <= occ`, so the subtraction cannot underflow, and
// the at most 255 summed `u32` lengths fit a `u64`.
#[allow(clippy::arithmetic_side_effects)]
fn payload_byte_upto(payload_lengths: &[u32], occ: u64) -> i32 {
    let block_start = (occ - occ % BLOCK_SIZE as u64) as usize;
    payload_lengths
        .get(block_start..occ as usize)
        .map_or(0, |run| run.iter().map(|&l| u64::from(l)).sum::<u64>()) as i32
}

/// The running `level0LastPosFP`/`level1LastPosFP` (and their `.pay` twins)
/// that turn [`PositionLayout`]'s absolute samples into the *deltas* the wire
/// carries, plus the running occurrence count that indexes it.
///
/// One per term. `Lucene104PostingsWriter` keeps exactly these four fields and
/// resets them at `startTerm` (`Lucene104PostingsWriter.java:244-247`).
struct PosSkipWriter<'a> {
    /// `None` for a field that does not index positions: no pos/pay skip
    /// sub-fields exist on the wire at all, and every method here is a no-op.
    layout: Option<&'a PositionLayout>,
    /// The term's payload lengths, empty when the field has no payloads.
    payload_lengths: &'a [u32],
    has_offsets_or_payloads: bool,
    /// Occurrences of the documents written so far — `flushDocBlock`'s
    /// implicit "everything `addPosition` has seen" watermark.
    occ: u64,
    level0_last_pos_fp: u64,
    level0_last_pay_fp: u64,
    level1_last_pos_fp: u64,
    level1_last_pay_fp: u64,
}

impl<'a> PosSkipWriter<'a> {
    fn new(
        layout: Option<&'a PositionLayout>,
        payload_lengths: &'a [u32],
        has_offsets_or_payloads: bool,
    ) -> Self {
        let pos_start = layout.map_or(0, |l| l.pos_start);
        let pay_start = layout.and_then(|l| l.pay_start).unwrap_or(0);
        PosSkipWriter {
            layout,
            payload_lengths,
            has_offsets_or_payloads,
            occ: 0,
            level0_last_pos_fp: pos_start,
            level0_last_pay_fp: pay_start,
            level1_last_pos_fp: pos_start,
            level1_last_pay_fp: pay_start,
        }
    }

    /// Accounts for one `.doc` block's documents having been added, before
    /// its skip record is written — `addPosition` runs for every occurrence
    /// of every document in the block, then `flushDocBlock` samples.
    fn add_block_docs(&mut self, block: &[(i32, i32)]) {
        if self.layout.is_some() {
            // ARITH: every freq is `>= 1` and `<= i32::MAX` (`validate_field`),
            // and a term has at most `i32::MAX` documents, so the running
            // occurrence count stays below 2^62.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.occ += block.iter().map(|&(_, f)| f as u64).sum::<u64>();
            }
        }
    }

    /// `flushDocBlock`'s `if (writePositions)` region
    /// (`Lucene104PostingsWriter.java:413-422`), written into the level-0
    /// record right after the impacts run.
    fn write_level0(&mut self, out: &mut Vec<u8>) {
        let Some(layout) = self.layout else { return };
        let s = layout.sample(self.occ, self.payload_lengths);
        // `sample` falls back to 0 for an out-of-range block index rather
        // than panicking (see its doc comment), and a 0 there would make this
        // delta underflow. Java's is a `long` subtraction that wraps, and a
        // wrapped pointer delta is exactly the "wrong pointer a differential
        // test catches" that fallback is written for -- an underflow panic
        // inside the writer is not.
        out.write_vlong(s.pos_fp.wrapping_sub(self.level0_last_pos_fp) as i64);
        out.write_byte(s.pos_upto);
        self.level0_last_pos_fp = s.pos_fp;
        if self.has_offsets_or_payloads {
            out.write_vlong(s.pay_fp.wrapping_sub(self.level0_last_pay_fp) as i64);
            out.write_vint(s.pay_upto);
            self.level0_last_pay_fp = s.pay_fp;
        }
    }

    /// `writeLevel1SkipData`'s `if (writePositions)` region
    /// (`Lucene104PostingsWriter.java:513-521`), written into the level-1
    /// entry's `scratchOutput` right after the impacts run — so it lands
    /// *inside* the `skip1EndFP` extent but *outside* `numImpactBytes`.
    fn write_level1(&mut self, out: &mut Vec<u8>) {
        let Some(layout) = self.layout else { return };
        let s = layout.sample(self.occ, self.payload_lengths);
        // `sample` falls back to 0 for an out-of-range block index rather
        // than panicking (see its doc comment), and a 0 there would make this
        // delta underflow. Java's is a `long` subtraction that wraps, and a
        // wrapped pointer delta is exactly the "wrong pointer a differential
        // test catches" that fallback is written for -- an underflow panic
        // inside the writer is not.
        out.write_vlong(s.pos_fp.wrapping_sub(self.level1_last_pos_fp) as i64);
        out.write_byte(s.pos_upto);
        self.level1_last_pos_fp = s.pos_fp;
        if self.has_offsets_or_payloads {
            out.write_vlong(s.pay_fp.wrapping_sub(self.level1_last_pay_fp) as i64);
            out.write_vint(s.pay_upto);
            self.level1_last_pay_fp = s.pay_fp;
        }
    }
}

/// Writes one full 256-occurrence `.pos` `PForUtil` block — no skip header
/// at all (unlike [`write_full_block`]'s `.doc` full blocks): `.pos` full
/// blocks are just a bare `for_util::pfor_encode`'d array of position deltas,
/// read back by a plain `for_util::pfor_decode` call with no header framing
/// whatsoever, per `crate::postings::read_positions`'s `num_full_blocks` loop.
/// `deltas` must be exactly `BLOCK_SIZE` (256) position deltas.
fn write_full_position_block(out: &mut Vec<u8>, deltas: &[i32]) {
    debug_assert_eq!(deltas.len(), BLOCK_SIZE as usize);
    let mut vals = [0u32; for_util::BLOCK_SIZE];
    for (v, &d) in vals.iter_mut().zip(deltas) {
        *v = d as u32;
    }
    for_util::pfor_encode(&mut vals, out);
}

/// Writes one full 256-occurrence `.pay` offset block: two back-to-back
/// bare `PForUtil` arrays (offset start-deltas, then offset lengths), same
/// "no skip header at all" shape as [`write_full_position_block`] — the
/// exact write-side inverse of `crate::postings::read_positions`'s
/// `has_offsets` full-block branch (`Lucene104PostingsWriter.java:350-353`:
/// `pforUtil.encode(offsetStartDeltaBuffer, payOut);
/// pforUtil.encode(offsetLengthBuffer, payOut);`). Both slices must be
/// exactly `BLOCK_SIZE` (256) long.
fn write_full_offset_block(out: &mut Vec<u8>, start_deltas: &[i32], lengths: &[i32]) {
    debug_assert_eq!(start_deltas.len(), BLOCK_SIZE as usize);
    debug_assert_eq!(lengths.len(), BLOCK_SIZE as usize);
    let mut starts = [0u32; for_util::BLOCK_SIZE];
    for (v, &d) in starts.iter_mut().zip(start_deltas) {
        *v = d as u32;
    }
    for_util::pfor_encode(&mut starts, out);
    let mut lens = [0u32; for_util::BLOCK_SIZE];
    for (v, &l) in lens.iter_mut().zip(lengths) {
        *v = l as u32;
    }
    for_util::pfor_encode(&mut lens, out);
}

/// Writes one full 256-occurrence `.pay` payload block: a bare `PForUtil`
/// array of raw (unsuppressed — see the module/`write_position_tail` doc
/// comments, full blocks never suppress repeated lengths, only the vint tail
/// does) payload lengths, followed by a vint byte-count and that many raw
/// payload bytes — the exact write-side inverse of `crate::postings::
/// read_positions`'s `has_payloads` full-block branch
/// (`Lucene104PostingsWriter.java:344-349`: `pforUtil.encode(payloadLengthBuffer,
/// payOut); payOut.writeVInt(payloadByteUpto); payOut.writeBytes(payloadBytes,
/// 0, payloadByteUpto);`). Always written *before* this same block's offset
/// fields (see [`write_full_offset_block`]) when both are present, matching
/// `addPosition`'s own payload-then-offsets order. `lengths` must be exactly
/// `BLOCK_SIZE` (256) long; `bytes` must be exactly `lengths.iter().sum()`
/// long.
fn write_full_payload_length_block(out: &mut Vec<u8>, lengths: &[u32], bytes: &[u8]) {
    debug_assert_eq!(lengths.len(), BLOCK_SIZE as usize);
    debug_assert_eq!(
        lengths.iter().map(|&l| l as usize).sum::<usize>(),
        bytes.len()
    );
    let mut lens = [0u32; for_util::BLOCK_SIZE];
    lens.copy_from_slice(lengths);
    for_util::pfor_encode(&mut lens, out);
    out.write_vint(bytes.len() as i32);
    out.write_bytes(bytes);
}

/// Writes every term's per-term postings metadata bytes — the write-side
/// inverse of `crate::postings::decode_term_metadata` (restricted to this
/// writer's own scope: `payStartFP` only appears when the field indexes
/// offsets or stores payloads; `lastPosBlockOffset` carries the real offset
/// of the vint position tail, exactly when `decode_term_metadata`'s own
/// `total_term_freq > BLOCK_SIZE` gate requires it). Takes the same branch
/// `Lucene104PostingsWriter.encodeTerm` takes for every term: the zigzag
/// singleton-doc-delta branch (bit 0 set) between two consecutive singleton
/// terms that share a `docStartFP`, the plain `docStartFP`-delta branch
/// otherwise.
///
/// `doc_start_fp`/`pos_start_fp`/`pay_start_fp` deltas are threaded exactly
/// like `SegmentTermsEnumFrame.metaDataUpto`/`absolute` on the read side: the
/// first term of `indices` -- one block's terms, in block order, which the
/// block writer hands over a block at a time -- decodes against `TermMetadata::EMPTY`
/// (`doc_start_fp`/`pos_start_fp`/`pay_start_fp == 0`), every subsequent term
/// against the *previous* term's already-written value — so this writer must
/// emit the same running delta, not each term's absolute offset. Unlike
/// `doc_start_fp`, `pos_start_fp`/`pay_start_fp` never have a singleton-skip
/// special case: every term that indexes positions/offsets writes real
/// `.pos`/`.pay` bytes and so always advances them, even when `docFreq == 1`
/// pulses its `.doc` entry away.
#[allow(clippy::too_many_arguments)]
fn write_term_metadata(
    out: &mut Vec<u8>,
    terms: &[blocktree_writer::BlockTerm<'_>],
    singletons: &[i32],
    indices: &[usize],
    doc_start_fp: &[u64],
    pos_start_fp: &[u64],
    pay_start_fp: &[u64],
    last_pos_block_offset: &[i64],
    index_has_positions: bool,
    index_has_offsets_or_payloads: bool,
) {
    // `lastState`, reset to `EMPTY_STATE` for the block's first term
    // (`absolute`): no singleton, every pointer 0.
    let mut base_doc_start_fp = 0u64;
    let mut base_singleton: i32 = -1;
    let mut base_pos_start_fp = 0u64;
    let mut base_pay_start_fp = 0u64;
    for &i in indices {
        let singleton = singletons[i];
        let this_fp = doc_start_fp[i];
        if base_singleton != -1 && singleton != -1 && this_fp == base_doc_start_fp {
            // Runs of rare terms (IDs) share a `.doc` pointer; the doc id is
            // written as a zigzag delta from the previous singleton's.
            // Two `i32` doc ids: the difference always fits an `i64`.
            let delta = i64::from(singleton).wrapping_sub(i64::from(base_singleton));
            let zigzag = ((delta << 1) ^ (delta >> 63)) as u64;
            out.write_vlong(((zigzag << 1) | 1) as i64);
        } else {
            let delta = this_fp.wrapping_sub(base_doc_start_fp);
            out.write_vlong((delta << 1) as i64);
            if singleton != -1 {
                out.write_vint(singleton);
            }
        }
        base_doc_start_fp = this_fp;
        base_singleton = singleton;

        if index_has_positions {
            let this_pos_fp = pos_start_fp[i];
            let pos_delta = this_pos_fp.wrapping_sub(base_pos_start_fp);
            out.write_vlong(pos_delta as i64);
            base_pos_start_fp = this_pos_fp;

            if index_has_offsets_or_payloads {
                let this_pay_fp = pay_start_fp[i];
                let pay_delta = this_pay_fp.wrapping_sub(base_pay_start_fp);
                out.write_vlong(pay_delta as i64);
                base_pay_start_fp = this_pay_fp;
            }

            // `lastPosBlockOffset`: only present on the wire when
            // `total_term_freq > BLOCK_SIZE` (`decode_term_metadata`'s
            // gate, strictly greater -- exactly `BLOCK_SIZE` occurrences fill
            // one full block with no tail after it, so the real writer only
            // emits this field once there is a genuine vint tail to point at).
            //
            // The value is the byte offset, relative to this term's
            // `posStartFP`, at which that vint tail begins -- i.e. exactly how
            // many bytes this term's full `PForUtil` position blocks took
            // (`write_position_tail`'s return value, sampled the same place
            // `Lucene104PostingsWriter.finishTerm` samples
            // `posOut.getFilePointer() - posStartFP`). This port's own
            // `read_positions` re-derives the block/tail split from
            // `total_term_freq`, but `postings::read_occurrences_for_doc` --
            // which jumps into the middle of `.pos` from `.doc`'s skip data
            // and so has no occurrence count to re-derive anything from --
            // reads it, and so does real Lucene:
            // `Lucene104PostingsReader.reset` computes
            // `lastPosBlockFP = posStartFP + lastPosBlockOffset` and
            // `refillPositions` switches to `refillLastPositionBlock` the
            // moment `posIn.getFilePointer()` equals it. Writing a constant 0
            // here would make that comparison true at the term's very first
            // position block, so real Lucene would decode a `PForUtil` block as
            // if it were the vint tail; the skip-driven walk reads it too
            // (`postings_skip_pointers.rs`).
            if terms[i].total_term_freq > BLOCK_SIZE as i64 {
                out.write_vlong(last_pos_block_offset[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests;
