//! Write side for a **single field's** term dictionary + postings —
//! `.doc`/`.tim`/`.tip`/`.tmd` — narrowly scoped to be the exact inverse of
//! what `crate::postings`/`crate::blocktree`'s existing (unmodified) read
//! side already decodes for the shapes below. Nothing here duplicates that
//! decode logic; this module only emits bytes, and the differential tests in
//! `crates/lucene-search` prove those bytes read back correctly through the
//! real, pre-existing `blocktree::open`/`postings::DocInput` functions.
//!
//! # Scope (read this before assuming more than it proves)
//!
//! - **One or more fields per call**, each independently written (`numFields`
//!   in `.tmd` is `inputs.len()`).
//! - **Exactly one physical `.tim` block per field, under one
//!   `SIGN_NO_CHILDREN` `.tip` root — never a split trie.** This is the
//!   load-bearing scope restriction. A `SIGN_MULTI_CHILDREN` writer existed
//!   briefly (one leaf block per leading byte, an `ARRAY`-strategy root with
//!   no output of its own) and was removed: **real Lucene cannot read it.**
//!   `SegmentTermsEnum` starts by loading the root *block*, and a root node
//!   carrying children but no output hands `loadBlock` an `fp` of `-1`. Two
//!   terms differing in their first byte were enough to trip it (see
//!   `docs/sweep/findings.md`, "The term dictionary could not survive a
//!   second leading byte"). **Explicitly still unimplemented**: non-leaf
//!   blocks whose entries are sub-block pointers, floor sub-blocks, any
//!   second trie level, and the `ARRAY`/`BITS`/`REVERSE_ARRAY` child-label
//!   strategies (`crate::blocktree`'s read side supports all three; this
//!   writer emits none of them). The cost is that a term lookup within a
//!   field scans the field's single block instead of descending a trie —
//!   the block-tree navigation item already filed in the sweep findings.
//! - **`docFreq` of any size is now supported for the `.doc` doc-delta/freq
//!   stream**: every complete 256-doc chunk of a term's postings is emitted
//!   as a full `ForUtil`/`PForUtil`-encoded block ([`write_full_block`],
//!   reusing `crate::for_util::for_encode`/`pfor_encode` directly — no
//!   bit-packing is reimplemented here), preceded by a level-0 skip header
//!   the existing, unmodified `crate::postings::read_full_block_header`/
//!   `decode_full_block_body` already parses. The `docFreq % BLOCK_SIZE`
//!   remainder still uses the group-varint tail-block path. Doc deltas
//!   always take the plain positive-`bitsPerValue` `ForUtil` shape (never
//!   the `bitsPerValue == 0` "all-256-consecutive" or `bitsPerValue < 0`
//!   dense-bitset alternate encodings the real writer sometimes prefers for
//!   space — see `docs/parity.md` for that scope cut). Each block carries
//!   **one impact, `(maxFreq, norm = 1)`** rather than a real
//!   `CompetitiveImpactAccumulator` run: [`FieldPostingsInput`] carries no
//!   norms, so the accumulator has nothing to accumulate against. Norm 1 is
//!   the highest-scoring norm, so the bound is *sound* but loose — it costs
//!   query-time pruning, never a wrong answer. (An empty impacts region is
//!   not an option: real Lucene rejects the segment with "Got empty list of
//!   impacts".) **`docFreq >= LEVEL1_NUM_DOCS` (8192) is now
//!   supported too**: for every complete span of [`crate::postings::LEVEL1_FACTOR`] (32) full
//!   level-0 blocks, a level-1 skip entry ([`write_level1_span`]) is emitted
//!   immediately before them — the exact write-side inverse of
//!   `crate::postings::read_level1_entry`/`LazyDocsCursor::skip_level1_to`.
//!   The level-1 entry carries the same single `(maxFreq, norm = 1)` impact,
//!   maximised over the whole 8192-doc span so it bounds every level-0 block
//!   beneath it, and — since `c20-postings-skip` — the `indexHasPos`-gated
//!   `.pos`/`.pay` sub-fields too ([`PosSkipWriter::write_level1`]), which is
//!   what lets a positions-indexing field exceed `BLOCK_SIZE` at all. **There is no
//!   further per-term docFreq ceiling**: the reader has no level-2 skip
//!   structure (`Lucene104` postings only ever have levels 0 and 1), so a
//!   term spanning any number of level-1 spans plus a final partial span
//!   round-trips the same way arbitrarily large `docFreq` already did below
//!   `LEVEL1_NUM_DOCS`.
//! - **Term frequency, positions, and now offsets too — still no
//!   payloads.** `IndexOptions::Docs`/`DocsAndFreqs`/
//!   `DocsAndFreqsAndPositions`/`DocsAndFreqsAndPositionsAndOffsets`/
//!   `DocsAndCustomFreqs` are all accepted — `DocsAndCustomFreqs` is
//!   wire-identical to `DocsAndFreqs` (real Lucene's `writeFreqs` derives from
//!   `IndexOptions.subsumes(DOCS_AND_FREQS)`, which the two share; they only
//!   differ in how the freq value is *interpreted* by the caller, never in
//!   encoding), so no separate code path is needed for it here; `.pos` is only
//!   written once a field indexes positions, and
//!   `.pay` is only written once a field indexes offsets (this writer never
//!   has payloads, so `.pay` is never opened for that reason alone). This
//!   mirrors `flush_stored_only_segment`'s own historical "start with the
//!   smallest defensible slice" precedent (see
//!   `crate::term_vectors::write_best_speed`'s positions-only cut for
//!   another example of the same policy).
//! - **`total_term_freq` of any size is now supported for the `.pos`/`.pay`
//!   position/offset streams too**: every complete 256-occurrence chunk of a
//!   term's positions (buffered across doc boundaries, matching real
//!   `Lucene104PostingsWriter.addPosition`'s `posBufferUpto == BLOCK_SIZE`
//!   flush timing) is emitted as a full `PForUtil`-encoded block
//!   ([`write_full_position_block`], reusing `crate::for_util::pfor_encode`
//!   directly) — and, when the field indexes offsets, that same chunk's
//!   offset start-deltas/lengths are emitted as a full `PForUtil`-encoded
//!   `.pay` block right alongside it ([`write_full_offset_block`]) — with the
//!   `total_term_freq % BLOCK_SIZE` remainder still using the vint-tail path
//!   (`refillLastPositionBlock`-equivalent, offset start-delta/length pairs
//!   inlined in `.pos` right after each occurrence's position delta).
//!   Unlike `.doc` full blocks, a `.pos`/`.pay` full block has **no skip
//!   header at all** — it's read back by bare, unframed
//!   `for_util::pfor_decode` calls, per `crate::postings::read_positions`'s
//!   `num_full_blocks` loop — so a `.pos`/`.pay` block carries no skip data
//!   of its own. The skip data that locates them lives in `.doc`: every
//!   level-0 block header and level-1 span entry of a positions-indexing
//!   field carries the `.pos`/`.pay` file pointer and buffer offset its
//!   documents' occurrences start at ([`PosSkipWriter`]), which is what lets
//!   a reader `advance(doc)` and jump `.pos` without walking the postings
//!   list. This writer builds each file whole rather than interleaving them,
//!   so it lays `.pos`/`.pay` out first and reconstructs the samples real
//!   Lucene takes live (see [`PositionLayout`]); the flush schedule is pure
//!   arithmetic (one `.pos` block per 256 occurrences, doc-boundary-agnostic),
//!   so the reconstruction is exact rather than approximate. **`docFreq` has
//!   no positions-specific ceiling any more** — `c20-postings-skip` closed
//!   the gap that used to force one.
//! - **`docFreq == 1` is pulsed into the term dictionary**, exactly like the
//!   real writer (`Lucene104PostingsWriter.java:568-577`): no `.doc` bytes at
//!   all for a singleton term, matching what `postings::singleton_postings`
//!   already expects to read back.
//!
//! # Caller obligations (not re-validated beyond what's cheap to check)
//!
//! `terms` must already be sorted ascending by term bytes with no
//! duplicates, and each term's `docs` must be sorted ascending by doc ID with
//! no duplicates and every `freq >= 1` — the same invariant
//! `indexing_chain::InMemoryInvertedIndex`'s `BTreeMap`/per-term sort already
//! guarantees for its `Vec<PostingEntry>`. Violating this produces incorrect
//! (but not memory-unsafe) output; [`write_single_field`] only checks the
//! cheap structural invariants explicitly listed above (sortedness of terms,
//! `docFreq` bound, `index_options`).
//!
//! # Wire format written (mirrors `crate::blocktree`/`crate::postings`'s own
//! module docs, writer side)
//!
//! - `.doc`: `IndexHeader(codec="Lucene104PostingsWriterDoc")`, then, for
//!   each non-singleton term in order, its tail-block bytes (group-varint
//!   `(docDelta << 1) | (freq == 1 ? 1 : 0)` values when `index_options`
//!   carries freqs, else plain `docDelta`, followed by one plain vint per
//!   `freq != 1` doc, in doc order) — see `crate::postings::read_tail_block`
//!   for the exact inverse. `Footer`.
//! - `.pos` (only when `index_options` indexes positions —
//!   `DocsAndFreqsAndPositions` or `DocsAndFreqsAndPositionsAndOffsets`):
//!   `IndexHeader(codec="Lucene104PostingsWriterPos")`, then, for each term
//!   that indexes positions, zero or more full 256-occurrence `PForUtil`
//!   blocks followed by a vint tail for the remainder — plain `posDelta`
//!   vints (accumulator reset to 0 at each doc's first occurrence; no
//!   payload bit-packing, since this writer never has payloads), each
//!   optionally followed, when the field also indexes offsets, by an
//!   `(offsetStartDelta << 1) | changed` vint and, only when `changed`, an
//!   offset-length vint — see `crate::postings::read_positions`'s tail-block
//!   branch (`has_payloads == false`) for the exact inverse. `Footer`.
//! - `.pay` (only when `index_options` is
//!   `DocsAndFreqsAndPositionsAndOffsets`): `IndexHeader(codec=
//!   "Lucene104PostingsWriterPay")`, then, for each term's full
//!   256-occurrence `.pos` blocks, that same chunk's offset start-deltas
//!   then offset lengths as two back-to-back bare `PForUtil` arrays (no
//!   payload-length/payload-bytes fields, since this writer never has
//!   payloads) — see `crate::postings::read_positions`'s `has_offsets`
//!   full-block branch for the exact inverse. `Footer`.
//! - `.tim`: `IndexHeader(codec="BlockTreeTermsDict")`, then one physical
//!   block per field, each block being (`entCount << 1 | 1` code,
//!   `isLeafBlock` + `NO_COMPRESSION` code, suffix bytes, suffix lengths,
//!   per-term stats, per-term postings metadata — see
//!   [`write_term_metadata`]), `Footer`.
//! - `.tip`: `IndexHeader(codec="BlockTreeTermsIndex")`, then, per field, one
//!   `SIGN_NO_CHILDREN`/`hasTerms`/no-floor root node pointing at that
//!   field's single `.tim` block — see [`write_leaf_node`]. `Footer`.
//! - `.tmd`: `IndexHeader(codec="BlockTreeTermsMeta")`, the postings writer's
//!   own embedded header (`IndexHeader(codec="Lucene104PostingsWriterTerms")`,
//!   `indexBlockSize = 256`), `numFields = inputs.len()`, then each field's
//!   record (`fieldNumber, numTerms, sumTotalTermFreq/sumDocFreq, docCount, minTerm/maxTerm,
//!   indexStart/rootFP/indexEnd`), `indexLength`, `termsLength`, `Footer`.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_output::DataOutput;

use crate::blocktree::{
    LEAF_NODE_HAS_TERMS, POSTINGS_BLOCK_SIZE, POSTINGS_TERMS_CODEC, POSTINGS_VERSION_CURRENT,
    SIGN_NO_CHILDREN, TERMS_CODEC_NAME, TERMS_INDEX_CODEC_NAME, TERMS_META_CODEC_NAME,
    VERSION_CURRENT as BLOCKTREE_VERSION_CURRENT,
};
use crate::field_infos::IndexOptions;
use crate::for_util;
use crate::postings::{
    BLOCK_SIZE, DOC_CODEC, LEVEL1_NUM_DOCS, META_CODEC, PAY_CODEC, POS_CODEC,
    VERSION_CURRENT as DOC_VERSION_CURRENT,
};

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("write_single_field: terms must be non-empty")]
    EmptyTerms,
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
        "write_single_field: term at index {index}, doc index {doc_index} has non-ascending or \
         duplicate positions -- positions must strictly increase within a doc"
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
/// Flat rather than the nested `Vec<Vec<Vec<u8>>>` it used to be, because
/// that shape cost a heap object per occurrence *and* a vector header per
/// posting entry, on both sides of the `lucene-index` boundary: c23 measured
/// it at 26 us/doc and ~190 MB per 50 000 documents with an all-empty-payload
/// control costing the same, which is what identifies the slot rather than
/// the bytes as the cost. This is also the exact layout
/// [`write_position_tail`] already had to build internally, and the layout
/// `Lucene104PostingsWriter` accumulates into its own `payloadBytes` /
/// `payloadLengthBuffer`, so the writer now borrows the caller's run instead
/// of re-flattening it.
#[derive(Debug, Clone, Default)]
pub struct TermPostings {
    pub term: Vec<u8>,
    pub docs: Vec<(i32, i32)>,
    pub positions: Vec<Vec<i32>>,
    pub offsets: Vec<Vec<(i32, i32)>>,
    pub payload_bytes: Vec<u8>,
    pub payload_lengths: Vec<u32>,
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

/// Input to [`write_single_field`]: one field's whole term dictionary,
/// already fully materialized and sorted.
pub struct FieldPostingsInput<'a> {
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
    pub terms: &'a [TermPostings],
}

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
pub fn write_single_field(
    input: &FieldPostingsInput<'_>,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    write_fields(std::slice::from_ref(input), segment_id, segment_suffix)
}

/// Writes `.doc`/`.tim`/`.tip`/`.tmd` bytes for **one or more** fields in a
/// single segment — see the module doc for the exact per-field scope and
/// wire format, each of which applies independently to every field in
/// `inputs`. `numFields` in the resulting `.tmd` is `inputs.len()`; each
/// field still gets its own single `.tim` block and single root `.tip` trie
/// node (no multi-block/multi-level-trie support here, see the module doc),
/// but all fields' blocks/nodes/records are interleaved into the *same*
/// physical `.doc`/`.pos`/`.tim`/`.tip`/`.tmd` byte buffers, exactly like a
/// real multi-field segment. `segment_id`/`segment_suffix` must match what
/// the caller will later open the files with (`blocktree::open`/
/// `postings::DocInput::open` both check them).
pub fn write_fields(
    inputs: &[FieldPostingsInput<'_>],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<Output> {
    if inputs.is_empty() {
        return Err(Error::EmptyTerms);
    }
    for input in inputs {
        validate_field(input)?;
    }

    // ---- .doc ----
    let mut maxima = PostingsMaxima::default();
    let mut doc = Vec::new();
    codec_util::write_index_header(
        &mut doc,
        DOC_CODEC,
        DOC_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    // ---- .pos ----
    // Only written at all if at least one field indexes positions, exactly
    // like a real segment has no `.pos` file when no field needs one.
    let any_positions = inputs
        .iter()
        .any(|input| input.index_options.subsumes_positions());
    let mut pos = Vec::new();
    if any_positions {
        codec_util::write_index_header(
            &mut pos,
            POS_CODEC,
            DOC_VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
    }

    // ---- .pay ----
    // Only written at all if at least one field indexes offsets and/or has
    // payloads, same "no file needed" convention as `.pos`.
    let any_offsets = inputs
        .iter()
        .any(|input| input.index_options.subsumes_offsets());
    let any_payloads = inputs.iter().any(|input| input.has_payloads);
    let mut pay = Vec::new();
    if any_offsets || any_payloads {
        codec_util::write_index_header(
            &mut pay,
            PAY_CODEC,
            DOC_VERSION_CURRENT,
            segment_id,
            segment_suffix,
        );
    }

    // ---- .tim / .tip headers ----
    let mut tim = Vec::new();
    codec_util::write_index_header(
        &mut tim,
        TERMS_CODEC_NAME,
        BLOCKTREE_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    let mut tip = Vec::new();
    codec_util::write_index_header(
        &mut tip,
        TERMS_INDEX_CODEC_NAME,
        BLOCKTREE_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );

    // ---- .tmd header ----
    let mut tmd = Vec::new();
    codec_util::write_index_header(
        &mut tmd,
        TERMS_META_CODEC_NAME,
        BLOCKTREE_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    codec_util::write_index_header(
        &mut tmd,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    );
    tmd.write_vint(POSTINGS_BLOCK_SIZE);
    tmd.write_vint(inputs.len() as i32); // numFields

    for input in inputs {
        let index_has_positions = input.index_options.subsumes_positions();
        let index_has_freq = input.index_options != IndexOptions::Docs;

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
        // shared `.pos` buffer, same convention as `doc_start_fp` below. Left
        // at `0` (never read, see `write_term_metadata`) when this field
        // doesn't index positions.
        let index_has_offsets = input.index_options.subsumes_offsets();
        let index_has_payloads = input.has_payloads;
        let index_has_offsets_or_payloads = index_has_offsets || index_has_payloads;
        let mut pos_start_fp = vec![0u64; input.terms.len()];
        let mut pay_start_fp = vec![0u64; input.terms.len()];
        // `lastPosBlockOffset` per term: where this term's vint position tail
        // starts, relative to its own `posStartFP`. Only written to the term
        // metadata when `totalTermFreq > BLOCK_SIZE` (see
        // [`write_term_metadata`]), but computed for every term because
        // [`write_position_tail`] is the only place that knows it.
        let mut last_pos_block_offset = vec![0i64; input.terms.len()];
        let mut layouts: Vec<Option<PositionLayout>> = Vec::new();
        if index_has_positions {
            for (i, t) in input.terms.iter().enumerate() {
                pos_start_fp[i] = pos.len() as u64;
                if index_has_offsets_or_payloads {
                    pay_start_fp[i] = pay.len() as u64;
                }
                let layout = write_position_tail(
                    &mut pos,
                    &mut pay,
                    &t.positions,
                    &t.offsets,
                    &t.payload_bytes,
                    &t.payload_lengths,
                    index_has_offsets,
                    index_has_payloads,
                );
                last_pos_block_offset[i] = layout.last_pos_block_offset as i64;
                layouts.push(Some(layout));
            }
        } else {
            layouts.resize_with(input.terms.len(), || None);
        }

        // `doc_start_fp[i]` is term `i`'s byte offset into the *shared* `.doc`
        // buffer (relative to the whole file including its header — the same
        // absolute convention `postings::TermMetadata::doc_start_fp` decodes
        // into) where its tail block begins, or `0` for a singleton term
        // (never read for singletons, see `postings::singleton_postings`).
        let mut doc_start_fp = vec![0u64; input.terms.len()];
        for (i, t) in input.terms.iter().enumerate() {
            if t.docs.len() == 1 {
                continue;
            }
            doc_start_fp[i] = doc.len() as u64;

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
            let mut skip = PosSkipWriter::new(layouts[i].as_ref(), index_has_offsets_or_payloads);
            // `docFreq >= LEVEL1_NUM_DOCS` (8192): emit a level-1 skip entry
            // before every complete span of `LEVEL1_FACTOR` (32) full
            // level-0 blocks, mirroring `DocInput::read_postings`'s own
            // `doc_count_left >= LEVEL1_NUM_DOCS` loop exactly.
            // ARITH: `start` only ever advances by the very amount the loop
            // condition just proved is left (`len - start >= N` before
            // `start += N`), so `start <= t.docs.len()` holds at every test
            // and `start + N <= len` at every slice.
            #[allow(clippy::arithmetic_side_effects)]
            while t.docs.len() - start >= LEVEL1_NUM_DOCS as usize {
                let span = &t.docs[start..start + LEVEL1_NUM_DOCS as usize];
                prev_doc_id = write_level1_span(
                    &mut doc,
                    span,
                    prev_doc_id,
                    &mut level1_last_doc_id,
                    index_has_freq,
                    &mut maxima,
                    &mut skip,
                );
                start += LEVEL1_NUM_DOCS as usize;
            }
            // ARITH: same invariant as the level-1 loop above.
            #[allow(clippy::arithmetic_side_effects)]
            while t.docs.len() - start >= BLOCK_SIZE as usize {
                let block = &t.docs[start..start + BLOCK_SIZE as usize];
                prev_doc_id = write_full_block(
                    &mut doc,
                    block,
                    prev_doc_id,
                    index_has_freq,
                    &mut maxima,
                    &mut skip,
                );
                start += BLOCK_SIZE as usize;
            }
            if start < t.docs.len() {
                write_tail_block(&mut doc, &t.docs[start..], prev_doc_id, index_has_freq);
            }
        }

        // ---- this field's .tim block + .tip node ----
        // Every term goes in one leaf block under a single
        // `SIGN_NO_CHILDREN` trie root, whatever leading bytes the terms
        // span. This writer previously split a field spanning several
        // leading bytes into one leaf block per byte under a
        // `SIGN_MULTI_CHILDREN` root -- which real Lucene cannot read: its
        // terms enum starts by loading the root *block*, and that root node
        // carried children but no output of its own, so `loadBlock` was
        // handed -1. Two terms differing in their first byte were enough
        // (`docs/sweep/findings.md`, "The term dictionary could not survive
        // a second leading byte").
        //
        // Modelling that properly means non-leaf blocks whose entries are
        // sub-block pointers, which this writer does not have yet. Until it
        // does, one block is both correct and what real Lucene already
        // validates -- it is the shape every passing fixture here has always
        // produced. The cost is that term lookup within a field is a scan of
        // the single block rather than a trie descent, which is the
        // block-tree navigation item already filed in the sweep findings.
        let block_fp = write_tim_block(
            &mut tim,
            input.terms,
            &doc_start_fp,
            &pos_start_fp,
            &pay_start_fp,
            &last_pos_block_offset,
            input.index_options,
            index_has_positions,
            index_has_offsets_or_payloads,
        );
        let index_start = tip.len();
        let root_fp_abs = write_leaf_node(&mut tip, block_fp as u64);
        let index_end = tip.len();
        // ARITH: `write_leaf_node` returns the `tip.len()` it saw on entry,
        // which is exactly `index_start`, so this is 0.
        #[allow(clippy::arithmetic_side_effects)]
        let root_fp = root_fp_abs - index_start;

        // ---- this field's .tmd record ----
        tmd.write_vint(input.field_number);
        let num_terms = input.terms.len() as i64;
        tmd.write_vlong(num_terms);
        let sum_doc_freq: i64 = input.terms.iter().map(|t| t.docs.len() as i64).sum();
        let sum_total_term_freq: i64 = if input.index_options == IndexOptions::Docs {
            sum_doc_freq
        } else {
            input
                .terms
                .iter()
                .flat_map(|t| t.docs.iter())
                .map(|&(_, f)| f as i64)
                .sum()
        };
        if input.index_options != IndexOptions::Docs {
            tmd.write_vlong(sum_total_term_freq);
        }
        tmd.write_vlong(sum_doc_freq);
        tmd.write_vint(input.doc_count);
        let min_term = &input.terms[0].term;
        // ARITH: `validate_field` rejected an empty `terms`.
        #[allow(clippy::arithmetic_side_effects)]
        let max_term = &input.terms[input.terms.len() - 1].term;
        tmd.write_vint(min_term.len() as i32);
        tmd.write_bytes(min_term);
        tmd.write_vint(max_term.len() as i32);
        tmd.write_bytes(max_term);
        tmd.write_vlong(index_start as i64);
        tmd.write_vlong(root_fp as i64);
        tmd.write_vlong(index_end as i64);
    }

    codec_util::write_footer(&mut doc);
    if any_positions {
        codec_util::write_footer(&mut pos);
    }
    if any_offsets || any_payloads {
        codec_util::write_footer(&mut pay);
    }
    codec_util::write_footer(&mut tim);
    codec_util::write_footer(&mut tip);

    // Both lengths are the *whole* file, footer included: Java writes each
    // footer and only then records `getFilePointer()`, and its reader feeds
    // these straight to `CodecUtil.retrieveChecksum`, which rejects any file
    // whose real length disagrees.
    tmd.write_i64(tip.len() as i64); // indexLength
    tmd.write_i64(tim.len() as i64); // termsLength
    codec_util::write_footer(&mut tmd);

    // `.psm`, mirroring `Lucene104PostingsWriter.close()`: the impact maxima
    // first, then each postings file's length *including* its footer (Java
    // records `getFilePointer()` after writing the footer). Lucene sizes its
    // impact-decoding buffers from these maxima before reading any block, so
    // they must cover what was actually written.
    let mut psm = Vec::new();
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
    psm.write_i64(doc.len() as i64);
    if any_positions {
        psm.write_i64(pos.len() as i64);
        if any_offsets || any_payloads {
            psm.write_i64(pay.len() as i64);
        }
    }
    codec_util::write_footer(&mut psm);

    Ok(Output {
        doc,
        psm,
        pos,
        pay,
        tim,
        tip,
        tmd,
    })
}

/// Writes the one physical `.tim` leaf block a field gets, for `terms` (the
/// field's whole already-sorted term list), returning the block's absolute
/// byte offset into `tim`.
///
/// Each term is stored with its **full** bytes as the block's "suffix",
/// matching the empty path prefix of the `SIGN_NO_CHILDREN` root
/// [`write_leaf_node`] writes. There is no prefix to strip because there is no
/// enclosing trie node to have encoded one -- a `strip_prefix_len` parameter
/// existed while this writer emitted `SIGN_MULTI_CHILDREN` roots and was
/// removed with them (see this module's doc comment for why real Lucene cannot
/// read that shape).
///
/// `doc_start_fp`/`pos_start_fp` must be the same length as `terms`; metadata
/// deltas are threaded fresh starting from `TermMetadata::EMPTY`
/// (`write_term_metadata`'s `base_doc_start_fp`/`base_pos_start_fp` both start
/// at 0), matching `SegmentTermsEnumFrame`'s per-frame reset that the read side
/// (`crate::blocktree::decode_block`) already assumes.
#[allow(clippy::too_many_arguments)]
fn write_tim_block(
    tim: &mut Vec<u8>,
    terms: &[TermPostings],
    doc_start_fp: &[u64],
    pos_start_fp: &[u64],
    pay_start_fp: &[u64],
    last_pos_block_offset: &[i64],
    index_options: IndexOptions,
    index_has_positions: bool,
    index_has_offsets_or_payloads: bool,
) -> usize {
    let block_fp = tim.len();
    let ent_count = terms.len() as u32;
    let code = (ent_count << 1) | 1; // isLastInFloor
    tim.write_vint(code as i32);

    let mut suffix_bytes = Vec::new();
    let mut suffix_lengths = Vec::new();
    let mut stats = Vec::new();
    for t in terms {
        let suffix = &t.term[..];
        suffix_bytes.write_bytes(suffix);
        suffix_lengths.write_vint(suffix.len() as i32);
        let doc_freq = t.docs.len() as u32;
        let total_term_freq: i64 = t.docs.iter().map(|&(_, f)| f as i64).sum();
        stats.write_vint((doc_freq << 1) as i32); // never singleton-run-encoded
        if index_options != IndexOptions::Docs {
            // ARITH: `total_term_freq` is the sum of this term's per-doc
            // freqs, each `>= 1` (`validate_field`), over exactly `doc_freq`
            // documents -- so it is at least `doc_freq` and the difference is
            // non-negative.
            #[allow(clippy::arithmetic_side_effects)]
            let total_term_freq_delta = total_term_freq - doc_freq as i64;
            stats.write_vlong(total_term_freq_delta);
        }
    }

    let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // isLeafBlock, NO_COMPRESSION
    tim.write_vlong(code_l as i64);
    tim.write_bytes(&suffix_bytes);

    tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
    tim.write_bytes(&suffix_lengths);

    tim.write_vint(stats.len() as i32);
    tim.write_bytes(&stats);

    let mut meta = Vec::new();
    write_term_metadata(
        &mut meta,
        terms,
        doc_start_fp,
        pos_start_fp,
        pay_start_fp,
        last_pos_block_offset,
        index_has_positions,
        index_has_offsets_or_payloads,
    );
    tim.write_vint(meta.len() as i32);
    tim.write_bytes(&meta);

    block_fp
}

/// Writes one `SIGN_NO_CHILDREN`/`hasTerms`/no-floor `.tip` node pointing at
/// `block_fp` (a `.tim` block's absolute offset), returning this node's own
/// absolute offset into `tip`. It is the only node this writer emits per
/// field: the field's `.tip` root and its single `.tim` block's index entry
/// are the same node.
fn write_leaf_node(tip: &mut Vec<u8>, block_fp: u64) -> usize {
    let fp = tip.len();
    // keep it simple: always 8 bytes, same as blocktree.rs's test Builder
    let output_fp_bytes = 8usize;
    // ARITH: `output_fp_bytes` is the literal 8 on the line above.
    #[allow(clippy::arithmetic_side_effects)]
    let header =
        (SIGN_NO_CHILDREN as u8) | ((output_fp_bytes as u8 - 1) << 2) | (LEAF_NODE_HAS_TERMS as u8);
    tip.push(header);
    tip.extend_from_slice(&block_fp.to_le_bytes());
    tip.extend_from_slice(&0u64.to_le_bytes()); // 8-byte over-read pad, `load_node`'s SIGN_NO_CHILDREN reads up to fp+1..fp+9
    fp
}

/// Validates one field's structural invariants (sortedness, `docFreq`/
/// `totalTermFreq` bounds, positions shape) — the exact same checks
/// `write_single_field` ran inline before this became a per-field helper
/// shared by [`write_fields`]'s loop.
fn validate_field(input: &FieldPostingsInput<'_>) -> Result<()> {
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
        if w[0].term >= w[1].term {
            // ARITH: `i` indexes `terms.windows(2)`, so `i + 1 < terms.len()`.
            #[allow(clippy::arithmetic_side_effects)]
            let at = i + 1;
            return Err(Error::TermsNotSorted(at));
        }
    }
    let index_has_positions = input.index_options.subsumes_positions();
    let index_has_offsets = input.index_options.subsumes_offsets();
    for (i, t) in input.terms.iter().enumerate() {
        if t.docs.is_empty() {
            return Err(Error::EmptyPostings(i));
        }
        // Checked on the first doc only: the ascending check below carries it
        // to the rest. This is what bounds every `docID - lastDocID` delta
        // below to `1..=i32::MAX`, and it is the same rejection
        // `Lucene104PostingsWriter.startDoc` makes.
        if t.docs[0].0 < 0 {
            return Err(Error::NegativeDocId { index: i });
        }
        for (j, &(_, freq)) in t.docs.iter().enumerate() {
            if freq < 1 {
                return Err(Error::NonPositiveFreq { index: i });
            }
            // ARITH: guarded by `j > 0`.
            #[allow(clippy::arithmetic_side_effects)]
            let out_of_order = j > 0 && t.docs[j - 1].0 >= t.docs[j].0;
            if out_of_order {
                return Err(Error::DocIdsNotSorted { index: i });
            }
        }
        if index_has_positions {
            if t.positions.len() != t.docs.len() {
                return Err(Error::MissingPositions {
                    index: i,
                    doc_index: t.positions.len(),
                });
            }
            for (j, (&(_, freq), positions)) in t.docs.iter().zip(&t.positions).enumerate() {
                if positions.len() != freq as usize {
                    return Err(Error::PositionsFreqMismatch {
                        index: i,
                        doc_index: j,
                        positions: positions.len(),
                        freq,
                    });
                }
                if positions.windows(2).any(|w| w[0] >= w[1]) {
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
                if t.offsets.len() != t.docs.len() {
                    return Err(Error::MissingOffsets {
                        index: i,
                        doc_index: t.offsets.len(),
                    });
                }
                for (j, (&(_, freq), doc_offsets)) in t.docs.iter().zip(&t.offsets).enumerate() {
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
                let total_term_freq: usize = t.docs.iter().map(|&(_, freq)| freq as usize).sum();
                if t.payload_lengths.len() != total_term_freq {
                    return Err(Error::PayloadsFreqMismatch {
                        index: i,
                        payload_lengths: t.payload_lengths.len(),
                        total_term_freq,
                    });
                }
                // ARITH: a `u32` widened to `usize` cannot overflow the sum
                // unless the run has more occurrences than `usize::MAX / 2^32`
                // -- more than the address space holds -- and the sum is
                // checked against `payload_bytes.len()` immediately below.
                #[allow(clippy::arithmetic_side_effects)]
                let expected: usize = t.payload_lengths.iter().map(|&l| l as usize).sum();
                if let Some((k, &length)) = t
                    .payload_lengths
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
                if t.payload_bytes.len() != expected {
                    return Err(Error::PayloadBytesMismatch {
                        index: i,
                        payload_bytes: t.payload_bytes.len(),
                        expected,
                    });
                }
            }
        }
    }
    Ok(())
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
    index_has_freq: bool,
    maxima: &mut PostingsMaxima,
    skip: &mut PosSkipWriter<'_>,
) -> i32 {
    debug_assert_eq!(block.len(), BLOCK_SIZE as usize);
    // `addPosition` has run for every occurrence of this block's documents by
    // the time `flushDocBlock` samples the `.pos`/`.pay` pointers.
    skip.add_block_docs(block);

    // Everything from here down is what `blockLength` measures (i.e. what
    // `read_full_block_header` reads as `body_end - r.position()`
    // immediately after `blockLength` itself) -- build it in a scratch
    // buffer first so `blockLength`'s value is known before the header is
    // written.
    let mut rest = Vec::new();
    if index_has_freq {
        // One impact per block, `(maxFreq, norm = 1)`: the highest frequency
        // this block contains paired with the shortest possible field length,
        // which is an upper bound on any score the block can produce. Impacts
        // bound scores for dynamic pruning, so a loose bound only costs
        // pruning opportunities while a low one would drop real hits.
        //
        // An *empty* region is not the conservative choice it looks like:
        // Lucene rejects the segment outright with "Got empty list of impacts
        // on level 0".
        //
        // `Lucene104PostingsWriter.writeImpacts` encodes each impact against
        // the previous one, starting from `(0, 0)`, so this single entry is
        // `freqDelta = maxFreq - 1` and `normDelta = 0` -- the folded
        // single-byte form, with no zig-zag long following.
        let max_freq = block.iter().map(|&(_, f)| f).max().unwrap_or(1).max(1);
        let mut impacts = Vec::new();
        // ARITH: `.max(1)` on the line above puts `max_freq` in `1..=i32::MAX`,
        // so `max_freq - 1` is in `0..=i32::MAX - 1`. (Rust's `<<` checks the
        // shift *amount*, not the value, so the `<< 1` is Java's `int` shift
        // bit for bit.)
        #[allow(clippy::arithmetic_side_effects)]
        let freq_delta = max_freq - 1;
        impacts.write_vint(freq_delta << 1);
        maxima.observe_level0(1, impacts.len());
        rest.write_vint(impacts.len() as i32);
        rest.write_bytes(&impacts);
        // `.pos`/`.pay` skip pointers, immediately after the impacts run and
        // still inside `writeFreqs` -- `Lucene104PostingsWriter.java:413-422`.
        skip.write_level0(&mut rest);
    }
    // Java's `numSkipBytes = level0Output.size()` is sampled exactly here --
    // after the impacts region *and* the pos/pay skip fields, before the doc
    // deltas and freqs are appended.
    let impacts_region_len = rest.len() as i64;

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
        for_util::for_encode(&mut deltas, bits_per_value, &mut rest);
    } else {
        // Dense unary bit-set encoding: doc IDs are the set-bit positions
        // (ascending) in a `num_bit_set_longs`-word bitset based at
        // `prev_doc_id + 1`, matching `FixedBitSet`'s word/bit layout
        // (word = bit_index / 64, bit = bit_index % 64).
        let mut words = vec![0u64; num_bit_set_longs as usize];
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
        for word in &words {
            rest.write_i64(*word as i64);
        }
    }

    if index_has_freq {
        let mut freqs = [0u32; for_util::BLOCK_SIZE];
        for (i, &(_, freq)) in block.iter().enumerate() {
            freqs[i] = freq as u32;
        }
        for_util::pfor_encode(&mut freqs, &mut rest);
    }

    // `numSkipBytes` spans the two header fields below plus the impacts
    // region -- landing a reader that skips this block on the first doc-delta
    // byte. This port's own reader parses the field and ignores it, deriving
    // the same position from `blockLength`; real Lucene's `skipLevel0To`
    // seeks by it, and a zero here sends it *backwards* to re-read the header
    // as block data and then off the end of the file. Measure the fields by
    // building them first, as Java does with its `scratchOutput`.
    let mut header = Vec::new();
    // Java writes `docBuffer[BLOCK_SIZE - 1] - level0LastDocID` as an `int`;
    // see the per-doc delta above.
    write_vint15(&mut header, last_doc_id.wrapping_sub(prev_doc_id));
    write_vlong15(&mut header, rest.len() as i64);
    // ARITH: both terms are lengths of in-memory scratch buffers this call
    // just built -- a few KiB each, and both bounded by `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    let num_skip_bytes = impacts_region_len + header.len() as i64;
    out.write_vlong(num_skip_bytes);
    out.write_bytes(&header);
    out.write_bytes(&rest);

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
    index_has_freq: bool,
    maxima: &mut PostingsMaxima,
    skip: &mut PosSkipWriter<'_>,
) -> i32 {
    debug_assert_eq!(span.len(), LEVEL1_NUM_DOCS as usize);

    // Build the span's 32 full blocks into a scratch buffer first so the
    // level-1 entry's byte-length field is known before the entry header is
    // written (same "measure by building into scratch first" approach
    // `write_full_block` uses for `blockLength`).
    let mut span_bytes = Vec::new();
    let mut prev = prev_doc_id;
    for block in span.chunks(BLOCK_SIZE as usize) {
        prev = write_full_block(&mut span_bytes, block, prev, index_has_freq, maxima, skip);
    }
    let last_doc_id = prev;

    // This span's own single impact, on the same `(maxFreq, norm = 1)` basis
    // as [`write_full_block`]'s -- the max is over the whole 8192-doc span, so
    // it bounds every level-0 block beneath it, as a level-1 impact must.
    let mut level1_impacts = Vec::new();
    // `numImpactBytes` is sampled here, before the pos/pay sub-fields are
    // appended to the same scratch buffer (`Lucene104PostingsWriter.java:
    // 507-521`), so it counts the impact bytes only.
    let mut level1_scratch = Vec::new();
    if index_has_freq {
        let max_freq = span.iter().map(|&(_, f)| f).max().unwrap_or(1).max(1);
        // ARITH: as in `write_full_block` -- `.max(1)` bounds `max_freq` below.
        #[allow(clippy::arithmetic_side_effects)]
        let freq_delta = max_freq - 1;
        level1_impacts.write_vint(freq_delta << 1);
        maxima.observe_level1(1, level1_impacts.len());
        level1_scratch.extend_from_slice(&level1_impacts);
        skip.write_level1(&mut level1_scratch);
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
/// `refillLastPositionBlock`. It is load-bearing for real Lucene, and — since
/// `c20-postings-skip` — for this port's own skip-driven single-document
/// position walk too, which jumps into the middle of `.pos` and so cannot
/// re-derive the split from a running occurrence count.
#[allow(clippy::too_many_arguments)]
fn write_position_tail(
    pos_out: &mut Vec<u8>,
    pay_out: &mut Vec<u8>,
    positions: &[Vec<i32>],
    offsets: &[Vec<(i32, i32)>],
    payload_bytes: &[u8],
    payload_lengths: &[u32],
    has_offsets: bool,
    has_payloads: bool,
) -> PositionLayout {
    let pos_out_start = pos_out.len();
    let pay_out_start = pay_out.len();
    let has_offsets_or_payloads = has_offsets || has_payloads;
    let mut pos_block_fp: Vec<u64> = vec![pos_out_start as u64];
    let mut pay_block_fp: Vec<u64> = if has_offsets_or_payloads {
        vec![pay_out_start as u64]
    } else {
        Vec::new()
    };
    let mut deltas = Vec::new();
    let mut offset_start_deltas = Vec::new();
    let mut offset_lengths = Vec::new();
    // Borrowed, not rebuilt: `TermPostings` already carries the payload run in
    // exactly the layout this function consumes -- the whole term's payloads
    // concatenated, one length per occurrence -- so there is nothing to
    // flatten here any more.
    for (doc_idx, doc_positions) in positions.iter().enumerate() {
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
                let (start_offset, end_offset) = offsets[doc_idx][occ_idx];
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
        pos_block_fp.push(pos_out.len() as u64);
        if has_offsets_or_payloads {
            pay_block_fp.push(pay_out.len() as u64);
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

    // `payloadByteUpto`'s running value, per occurrence: entry `i` is the
    // number of payload bytes written by the occurrences before `i` *since
    // the last full-block flush*. Only the doc-block boundaries are ever
    // sampled from it, but it is the only place the per-occurrence lengths
    // are still in hand. Empty when the field has no payloads, in which case
    // `Lucene104PostingsWriter` leaves `payloadByteUpto` at `0` and writes
    // that (`addPosition` only touches it under `writePayloads`).
    let payload_prefix = if has_payloads {
        // ARITH: one more entry than there are occurrences, and
        // `payload_lengths` is an in-memory `Vec` whose length is bounded by
        // `isize::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        let mut prefix = Vec::with_capacity(payload_lengths.len() + 1);
        let mut acc = 0u64;
        prefix.push(0u64);
        for (i, &l) in payload_lengths.iter().enumerate() {
            // ARITH: `l` is a `payload.len() as i32`, so it is non-negative
            // and `acc` is bounded by the total payload bytes this term
            // holds -- an in-memory length, far below 2^64. `i + 1` cannot
            // overflow for the same reason, and `BLOCK_SIZE` is a non-zero
            // constant.
            #[allow(clippy::arithmetic_side_effects)]
            {
                acc += l as u64;
                // `addPosition`'s flush resets `payloadByteUpto` to `0` the moment
                // the 256th occurrence of a block has been buffered, so the
                // boundary entry is `0`, not the block's total.
                if (i + 1) % BLOCK_SIZE as usize == 0 {
                    acc = 0;
                }
            }
            prefix.push(acc);
        }
        prefix
    } else {
        Vec::new()
    };

    PositionLayout {
        pos_block_fp,
        pay_block_fp,
        payload_prefix,
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
struct PositionLayout {
    /// `.pos` absolute offset after `k` full blocks, for `k` in
    /// `0..=num_full_blocks`. Entry `0` is the term's `posStartFP`.
    pos_block_fp: Vec<u64>,
    /// The same for `.pay`. Empty when the field has neither offsets nor
    /// payloads (no `.pay` bytes exist, and no `.pay` skip fields are
    /// written).
    pay_block_fp: Vec<u64>,
    /// `payloadByteUpto` before occurrence `i`, reset at every full-block
    /// boundary. Empty when the field has no payloads.
    payload_prefix: Vec<u64>,
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
    fn sample(&self, occ: u64) -> PosSkipSample {
        // ARITH: `BLOCK_SIZE` is the non-zero constant 256, so neither the
        // division nor the remainder can trap.
        #[allow(clippy::arithmetic_side_effects)]
        let (q, r) = (
            (occ / BLOCK_SIZE as u64) as usize,
            (occ % BLOCK_SIZE as u64) as u8,
        );
        PosSkipSample {
            pos_fp: self.pos_block_fp.get(q).copied().unwrap_or(0),
            pos_upto: r,
            pay_fp: self.pay_block_fp.get(q).copied().unwrap_or(0),
            pay_upto: self.payload_prefix.get(occ as usize).copied().unwrap_or(0) as i32,
        }
    }
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
    fn new(layout: Option<&'a PositionLayout>, has_offsets_or_payloads: bool) -> Self {
        let pos_start = layout.map_or(0, |l| l.pos_block_fp[0]);
        let pay_start = layout.map_or(0, |l| l.pay_block_fp.first().copied().unwrap_or(0));
        PosSkipWriter {
            layout,
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
        let s = layout.sample(self.occ);
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
        let s = layout.sample(self.occ);
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
/// `total_term_freq > BLOCK_SIZE` gate requires it). Always takes the
/// bit-clear ("absolute-ish
/// `docStartFP` delta") branch, never the zigzag-singleton-delta branch —
/// this writer has no need for that alternate encoding's extra compactness.
///
/// `doc_start_fp`/`pos_start_fp`/`pay_start_fp` deltas are threaded exactly
/// like `SegmentTermsEnumFrame.metaDataUpto`/`absolute` on the read side: the
/// first term in the (only) block decodes against `TermMetadata::EMPTY`
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
    terms: &[TermPostings],
    doc_start_fp: &[u64],
    pos_start_fp: &[u64],
    pay_start_fp: &[u64],
    last_pos_block_offset: &[i64],
    index_has_positions: bool,
    index_has_offsets_or_payloads: bool,
) {
    let mut base_doc_start_fp = 0u64;
    let mut base_pos_start_fp = 0u64;
    let mut base_pay_start_fp = 0u64;
    for (i, t) in terms.iter().enumerate() {
        let doc_freq = t.docs.len();
        // Singleton terms never advance `doc_start_fp` (no `.doc` bytes are
        // written for them, see `write_single_field`), so their delta is 0
        // and the running base is left unchanged for the next term.
        let this_fp = if doc_freq == 1 {
            base_doc_start_fp
        } else {
            doc_start_fp[i]
        };
        let delta = this_fp.wrapping_sub(base_doc_start_fp);
        out.write_vlong(((delta << 1) as i64) & !1); // bit 0 clear: absolute-ish delta branch
        if doc_freq == 1 {
            out.write_vint(t.docs[0].0);
        }
        base_doc_start_fp = this_fp;

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
            // here -- which this writer did until the M2 sweep -- made that
            // comparison true at the term's very first position block, so
            // real Lucene decoded a `PForUtil` block as if it were the vint
            // tail. It was unreachable from this port's own round-trip tests
            // when b5 fixed it; c20's skip-driven walk makes it reachable
            // (`postings_skip_pointers.rs`).
            let total_term_freq: i64 = t.docs.iter().map(|&(_, f)| f as i64).sum();
            if total_term_freq > BLOCK_SIZE as i64 {
                out.write_vlong(last_pos_block_offset[i]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // A test's `i + 1` is not a length read off disk; see
    // `docs/arithmetic-gate.md`'s "Test code" section.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::blocktree::{self, FieldTerms};
    use crate::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, VectorEncoding,
        VectorSimilarityFunction,
    };
    use crate::postings::DocInput;

    const SEG_ID: [u8; ID_LENGTH] = [9u8; ID_LENGTH];
    const SUFFIX: &str = "";

    /// Builds a [`TermPostings`] flat payload run from one occurrence's
    /// payload per element, in doc order and then occurrence order -- which
    /// is the whole layout: doc boundaries live in `docs`' frequencies, not
    /// in the run.
    fn payload_run(occurrences: &[&[u8]]) -> (Vec<u8>, Vec<u32>) {
        let mut bytes = Vec::new();
        let mut lengths = Vec::new();
        for occurrence in occurrences {
            lengths.push(occurrence.len() as u32);
            bytes.extend_from_slice(occurrence);
        }
        (bytes, lengths)
    }

    fn field_info(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options,
            doc_values_type: DocValuesType::None,
            doc_values_skip_index_type: DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: VectorEncoding::Float32,
            vector_similarity_function: VectorSimilarityFunction::Euclidean,
        }
    }

    /// Same as [`field_info`] but with `store_payloads` set — needed for
    /// every payload round-trip test below, since [`FieldTerms::positions`]/
    /// [`crate::blocktree::FieldTerms`] reads `has_payloads` off the opened
    /// `FieldInfo`, not off the writer's [`FieldPostingsInput::has_payloads`]
    /// (which only controls what bytes this writer emits).
    fn field_info_with_payloads(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
        FieldInfo {
            store_payloads: true,
            ..field_info(number, name, index_options)
        }
    }

    fn open_written<'a>(
        output: &'a Output,
        field_infos: &FieldInfos,
        max_doc: i32,
    ) -> (blocktree::BlockTreeFields, DocInput<'a>) {
        let fields = blocktree::open(
            &output.tim,
            &output.tip,
            &output.tmd,
            field_infos,
            &SEG_ID,
            SUFFIX,
            max_doc,
        )
        .expect("write_single_field's own bytes must open cleanly");
        let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");
        (fields, doc_in)
    }

    /// Mixed singleton/multi-doc terms, round-tripped through the existing
    /// unmodified `blocktree::open` + `postings::DocInput` read side (no
    /// query layer here — see
    /// `crates/lucene-search/tests/postings_writer_round_trip.rs` for the
    /// required end-to-end `search_term_query` proof, which lives in
    /// `lucene-search` rather than here since this crate must not depend
    /// upward on `lucene-search`, see the `architecture` skill).
    #[test]
    fn mixed_singleton_and_multi_doc_terms_round_trip() {
        let terms = vec![
            TermPostings {
                term: b"fox".to_vec(),
                docs: vec![(1, 2), (4, 1), (7, 3)],
                ..Default::default()
            },
            TermPostings {
                term: b"quick".to_vec(),
                docs: vec![(4, 1)], // singleton
                ..Default::default()
            },
            TermPostings {
                term: b"the".to_vec(),
                docs: vec![(0, 1), (1, 1), (4, 2), (7, 1)],
                ..Default::default()
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 8,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

        let fis = FieldInfos {
            fields: vec![field_info(0, "body", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 8);

        let field = fields.field("body").unwrap();
        assert_eq!(field.num_terms, 3);
        assert_eq!(field.min_term, b"fox");
        assert_eq!(field.max_term, b"the");

        let postings = field.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![1, 4, 7]);
        assert_eq!(postings.freqs, vec![2, 1, 3]);

        let postings = field.postings(b"quick", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![4]);
        assert_eq!(postings.freqs, vec![1]);

        let postings = field.postings(b"the", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 1, 4, 7]);
        assert_eq!(postings.freqs, vec![1, 1, 2, 1]);

        assert!(field.seek_exact(b"missing").is_none());
    }

    /// Byte-level correctness on `docFreq`/`totalTermFreq`/`seek_exact`
    /// alone (no query layer), for `IndexOptions::Docs` (no freqs at all —
    /// `totalTermFreq == docFreq` aliasing) to make sure that branch, not
    /// just `DocsAndFreqs`, round-trips.
    #[test]
    fn docs_only_index_options_round_trips() {
        let terms = vec![
            TermPostings {
                term: b"a".to_vec(),
                docs: vec![(0, 1), (2, 1)],
                ..Default::default()
            },
            TermPostings {
                term: b"b".to_vec(),
                docs: vec![(1, 1)],
                ..Default::default()
            },
        ];
        let input = FieldPostingsInput {
            field_number: 3,
            index_options: IndexOptions::Docs,
            doc_count: 3,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(3, "f", IndexOptions::Docs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 3);
        let field: &FieldTerms = fields.field("f").unwrap();
        assert_eq!(
            field.seek_exact(b"a"),
            Some(blocktree::TermStats {
                doc_freq: 2,
                total_term_freq: 2
            })
        );
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 2]);
        assert_eq!(postings.freqs, vec![1, 1]); // freqs default to 1 when the field has no freqs

        let postings = field.postings(b"b", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![1]);
    }

    /// All terms singleton (`docFreq == 1`): no `.doc` bytes are needed at
    /// all — `postings()` must still resolve every term purely from the
    /// term-dictionary metadata (`postings::singleton_postings`).
    #[test]
    fn all_singleton_terms_need_no_doc_file() {
        let terms = vec![
            TermPostings {
                term: b"alpha".to_vec(),
                docs: vec![(2, 1)],
                ..Default::default()
            },
            TermPostings {
                term: b"beta".to_vec(),
                docs: vec![(5, 4)],
                ..Default::default()
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let fields = blocktree::open(
            &output.tim,
            &output.tip,
            &output.tmd,
            &fis,
            &SEG_ID,
            SUFFIX,
            6,
        )
        .unwrap();
        let field = fields.field("f").unwrap();
        // No `.doc` file opened at all -- `doc_in: None` is fine since every
        // term here is a singleton.
        let postings = field.postings(b"beta", None).unwrap().unwrap();
        assert_eq!(postings.docs, vec![5]);
        assert_eq!(postings.freqs, vec![4]);
    }

    #[test]
    fn rejects_empty_terms() {
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 0,
            has_payloads: false,
            terms: &[],
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::EmptyTerms)
        ));
    }

    #[test]
    fn rejects_unsorted_terms() {
        let terms = vec![
            TermPostings {
                term: b"b".to_vec(),
                docs: vec![(0, 1)],
                ..Default::default()
            },
            TermPostings {
                term: b"a".to_vec(),
                docs: vec![(0, 1)],
                ..Default::default()
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::TermsNotSorted(1))
        ));
    }

    #[test]
    fn rejects_duplicate_terms() {
        let terms = vec![
            TermPostings {
                term: b"a".to_vec(),
                docs: vec![(0, 1)],
                ..Default::default()
            },
            TermPostings {
                term: b"a".to_vec(),
                docs: vec![(1, 1)],
                ..Default::default()
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::TermsNotSorted(1))
        ));
    }

    #[test]
    fn rejects_empty_postings_for_a_term() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![],
            ..Default::default()
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 0,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::EmptyPostings(0))
        ));
    }

    #[test]
    fn rejects_non_ascending_doc_ids() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(2, 1), (1, 1)],
            ..Default::default()
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 3,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::DocIdsNotSorted { index: 0 })
        ));
    }

    #[test]
    fn rejects_duplicate_doc_ids() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(1, 1), (1, 1)],
            ..Default::default()
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::DocIdsNotSorted { index: 0 })
        ));
    }

    #[test]
    fn rejects_non_positive_freq() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 0), (1, 1)],
            ..Default::default()
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 2,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::NonPositiveFreq { index: 0 })
        ));
    }

    /// Round-trips a `docFreq` at each level-1-relevant boundary through the
    /// existing, unmodified `blocktree::open`/`DocInput::read_postings` --
    /// asserting the full doc/freq lists, not just "didn't error". Covers:
    /// exactly `LEVEL1_NUM_DOCS` (one level-1 span, no remainder), one more
    /// than that (one span + a one-doc tail), and two full level-1 spans
    /// back to back, proving `write_level1_span`'s `level1_last_doc_id`/
    /// `prev_doc_id` threading across more than one span.
    #[test]
    fn docfreq_at_level1_boundaries_round_trips() {
        for doc_freq in [LEVEL1_NUM_DOCS, LEVEL1_NUM_DOCS + 1, 2 * LEVEL1_NUM_DOCS] {
            let term = varied_docs_term(b"a", doc_freq);
            let max_doc = term.docs.last().unwrap().0 + 1;
            let terms = vec![term.clone()];
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: term.docs.len() as i32,
                has_payloads: false,
                terms: &terms,
            };
            let output = write_single_field(&input, &SEG_ID, SUFFIX)
                .unwrap_or_else(|e| panic!("doc_freq={doc_freq}: {e}"));
            let fis = FieldInfos {
                fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
            };
            let (fields, doc_in) = open_written(&output, &fis, max_doc);
            let field = fields.field("f").unwrap();
            assert_eq!(
                field.seek_exact(b"a").unwrap().doc_freq,
                doc_freq,
                "doc_freq={doc_freq}"
            );
            let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
            let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
            let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
            assert_eq!(postings.docs, expected_docs, "doc_freq={doc_freq}");
            assert_eq!(postings.freqs, expected_freqs, "doc_freq={doc_freq}");
        }
    }

    /// Same boundaries as [`docfreq_at_level1_boundaries_round_trips`] but
    /// through [`crate::postings::DocInput::lazy_cursor`]'s `advance`, which
    /// is what actually exercises `LazyDocsCursor::skip_level1_to` --
    /// jumping straight past whole level-1 spans without decoding their
    /// level-0 blocks. Advancing to the very last doc after a full span (or
    /// two) proves the skip landed in the right place.
    #[test]
    fn docfreq_at_level1_boundaries_advance_via_lazy_cursor() {
        for doc_freq in [LEVEL1_NUM_DOCS, LEVEL1_NUM_DOCS + 1, 2 * LEVEL1_NUM_DOCS] {
            let term = varied_docs_term(b"a", doc_freq);
            let max_doc = term.docs.last().unwrap().0 + 1;
            let terms = vec![term.clone()];
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: term.docs.len() as i32,
                has_payloads: false,
                terms: &terms,
            };
            let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
            let fis = FieldInfos {
                fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
            };
            let (fields, doc_in) = open_written(&output, &fis, max_doc);
            let field = fields.field("f").unwrap();
            let mut cursor = field
                .lazy_postings(b"a", &doc_in)
                .unwrap()
                .expect("term must exist");
            let last_doc = term.docs.last().unwrap().0;
            assert_eq!(
                cursor.advance(last_doc).unwrap(),
                last_doc,
                "doc_freq={doc_freq}"
            );
        }
    }

    /// b5's F6, closed: `PostingsEnum` flags. With
    /// [`crate::postings::PostingsFlags::DocsOnly`] every frequency block on
    /// the wire is stepped over (`PForUtil.skip`) instead of unpacked, and
    /// every reported frequency is `1` -- so the *doc ids* must be
    /// bit-for-bit what the freqs-decoding path produces, across all three
    /// block shapes at once (full level-0 blocks, a group-varint tail, and a
    /// level-1 span).
    #[test]
    fn docs_only_flags_decode_the_same_doc_ids_and_report_freq_one() {
        use crate::postings::PostingsFlags;
        for doc_freq in [
            300,                  // one full block + an 44-doc tail
            2 * BLOCK_SIZE,       // full blocks, no tail
            LEVEL1_NUM_DOCS + 77, // a whole level-1 span plus a tail
        ] {
            let term = irregular_docs_term(b"a", doc_freq);
            let max_doc = term.docs.last().unwrap().0 + 1;
            let terms = vec![term.clone()];
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: term.docs.len() as i32,
                has_payloads: false,
                terms: &terms,
            };
            let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
            let fis = FieldInfos {
                fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
            };
            let (fields, doc_in) = open_written(&output, &fis, max_doc);
            let field = fields.field("f").unwrap();

            let with_freqs = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
            let docs_only = field
                .postings_with_flags(b"a", Some(&doc_in), PostingsFlags::DocsOnly)
                .unwrap()
                .unwrap();
            assert_eq!(docs_only.docs, with_freqs.docs, "doc_freq={doc_freq}");
            assert!(
                with_freqs.freqs.iter().any(|&f| f != 1),
                "doc_freq={doc_freq}: the fixture must have real freqs, or \
                 this test proves nothing"
            );
            assert!(
                docs_only.freqs.iter().all(|&f| f == 1),
                "doc_freq={doc_freq}: DocsOnly must report freq 1 everywhere"
            );

            // ... and the same through the lazy cursor, walked to exhaustion.
            let mut lazy = field
                .lazy_postings_with_flags(b"a", &doc_in, PostingsFlags::DocsOnly)
                .unwrap()
                .unwrap();
            let mut walked = Vec::with_capacity(doc_freq as usize);
            loop {
                let d = lazy.next_doc().unwrap();
                if d == crate::postings::NO_MORE_DOCS {
                    break;
                }
                assert_eq!(lazy.freq(), Some(1), "doc_freq={doc_freq}");
                walked.push(d);
            }
            assert_eq!(walked, with_freqs.docs, "doc_freq={doc_freq} lazy walk");

            // A skip-heavy walk exercises the same skip on the advance path.
            let mut lazy = field
                .lazy_postings_with_flags(b"a", &doc_in, PostingsFlags::DocsOnly)
                .unwrap()
                .unwrap();
            let last = *with_freqs.docs.last().unwrap();
            assert_eq!(lazy.advance(last).unwrap(), last, "doc_freq={doc_freq}");
        }
    }

    /// A field that indexes no frequencies at all is unaffected by the flag:
    /// there is no frequency block on the wire to skip, and both paths report
    /// `1`.
    #[test]
    fn docs_only_flags_are_a_no_op_for_a_field_without_frequencies() {
        use crate::postings::PostingsFlags;
        let doc_freq = 300;
        let mut term = irregular_docs_term(b"a", doc_freq);
        for d in &mut term.docs {
            d.1 = 1;
        }
        let max_doc = term.docs.last().unwrap().0 + 1;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::Docs,
            doc_count: term.docs.len() as i32,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let with_freqs = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let docs_only = field
            .postings_with_flags(b"a", Some(&doc_in), PostingsFlags::DocsOnly)
            .unwrap()
            .unwrap();
        assert_eq!(docs_only.docs, with_freqs.docs);
        assert_eq!(docs_only.freqs, with_freqs.freqs);
    }

    /// Same as [`docfreq_at_level1_boundaries_round_trips`] but with
    /// [`irregular_docs_term`]'s non-constant doc-ID gaps and widely varying
    /// freqs instead of [`varied_docs_term`]'s constant delta-of-2 -- a
    /// delta/length-accounting bug in `write_level1_span` could plausibly
    /// only surface once the span's actual byte length varies unpredictably
    /// with real data, not a uniform pattern.
    #[test]
    fn docfreq_at_level1_boundary_with_irregular_gaps_and_varying_freqs() {
        let doc_freq = LEVEL1_NUM_DOCS + 100;
        let term = irregular_docs_term(b"a", doc_freq);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: term.docs.len() as i32,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        assert_eq!(field.seek_exact(b"a").unwrap().doc_freq, doc_freq);
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs);
        assert_eq!(postings.freqs, expected_freqs);

        // Also confirm the lazy cursor can advance straight to the last doc
        // (exercising skip_level1_to against this same irregular span).
        let mut cursor = field
            .lazy_postings(b"a", &doc_in)
            .unwrap()
            .expect("term must exist");
        let last_doc = term.docs.last().unwrap().0;
        assert_eq!(cursor.advance(last_doc).unwrap(), last_doc);
    }

    /// The write-side analogue of `postings`'s own
    /// `lazy_cursor_advance_skips_whole_corrupted_level1_span_without_decoding_it`
    /// test: writes a real level-1 span via [`write_level1_span`], then
    /// corrupts its first level-0 block's header bytes in place. An
    /// `advance()` to a doc in the trailing tail (past the whole span) must
    /// still succeed -- proving `skip_level1_to` jumped straight to
    /// `doc_end_fp` without ever reading the corrupted block 0 header. A
    /// control `advance()` to a target inside the span forces that same
    /// header to be decoded and must surface the corruption, confirming the
    /// first assertion wasn't passing by luck (e.g. because the corruption
    /// was inert).
    #[test]
    fn writer_level1_span_advance_past_it_skips_corrupted_first_block_header() {
        let doc_freq = LEVEL1_NUM_DOCS + 8;
        // `irregular_docs_term` (not `varied_docs_term`'s constant delta-2
        // docs): a constant delta of 2 makes `docRange == BLOCK_SIZE * 2`
        // land exactly on the writer's bit-set-vs-packed boundary (see
        // `write_full_block`'s doc comment), so the first level-0 block
        // would be written in the dense bit-set shape rather than the
        // generic packed shape this test's header corruption assumes.
        // `irregular_docs_term`'s widely varying deltas keep the packed
        // shape (IndexOptions::Docs below -> freq ignored anyway).
        let term = irregular_docs_term(b"a", doc_freq);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::Docs,
            doc_count: term.docs.len() as i32,
            has_payloads: false,
            terms: &terms,
        };
        let mut output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

        // Locate the level-1 span's first byte in `.doc`: the term's only
        // level-1 entry starts right after the `.doc` index header (this is
        // the field's one and only term), and -- since `IndexOptions::Docs`
        // has no freq, `index_has_freq` is false, so the entry is just
        // `vint(doc_delta)` then `vlong(span_len)` with no freq-gated
        // header fields (see `write_level1_span`'s doc comment) -- the span
        // bytes start immediately after those two fields.
        use lucene_store::data_input::{DataInput, SliceInput};
        let mut r = SliceInput::new(&output.doc);
        codec_util::check_index_header(&mut r, DOC_CODEC, 0, DOC_VERSION_CURRENT, &SEG_ID, SUFFIX)
            .unwrap();
        r.read_vint().unwrap(); // doc_delta
        r.read_vlong().unwrap(); // span_len
        let span_start = r.position();

        // Corrupt the first level-0 block's header (`level0NumBytes`
        // vlong + `docDelta`/`blockLength` fields) and well into its body
        // with bytes whose continuation bits never terminate -- 40 bytes
        // (not just the ~5-byte header) because `write_full_block` can pick
        // any of three doc-delta shapes, and a shorter corrupted run was
        // observed to occasionally decode "successfully" (silently wrong,
        // not an error) for the wider dense-bit-set body; 40 bytes reliably
        // errors regardless of which shape this block took.
        for b in output.doc[span_start..span_start + 40].iter_mut() {
            *b = 0xFF;
        }

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();

        // Past the whole (corrupted) span, in the tail: must succeed.
        let last_doc = term.docs.last().unwrap().0;
        let mut cursor = field
            .lazy_postings(b"a", &doc_in)
            .unwrap()
            .expect("term must exist");
        assert_eq!(cursor.advance(last_doc).unwrap(), last_doc);

        // Control: a target inside the span forces decoding the corrupted
        // block 0 header, which must surface an error.
        let mut cursor2 = field
            .lazy_postings(b"a", &doc_in)
            .unwrap()
            .expect("term must exist");
        assert!(cursor2.advance(100).is_err());
    }

    /// `docFreq == LEVEL1_NUM_DOCS - 1` (8191): the largest term size this
    /// writer accepts, one doc short of the rejection boundary tested above.
    /// Round-tripped through the unmodified reader, not just checked for an
    /// `Ok` result.
    #[test]
    fn docfreq_one_less_than_level1_num_docs_is_accepted() {
        let doc_freq = LEVEL1_NUM_DOCS - 1;
        let term = varied_docs_term(b"a", doc_freq);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        assert_eq!(field.seek_exact(b"a").unwrap().doc_freq, doc_freq);
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs);
        assert_eq!(postings.freqs, expected_freqs);
    }

    #[test]
    fn rejects_unsupported_index_options() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::None,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::UnsupportedIndexOptions(IndexOptions::None))
        ));
    }

    /// `IndexOptions::DocsAndCustomFreqs` is wire-identical to `DocsAndFreqs`
    /// (see the module doc): this round-trips a multi-doc, multi-freq term
    /// through the real writer + unmodified reader under that option, proving
    /// it's accepted end-to-end rather than just not-rejected.
    #[test]
    fn docs_and_custom_freqs_round_trips_like_docs_and_freqs() {
        let term = TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 3), (1, 1), (5, 7)],
            ..Default::default()
        };
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndCustomFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndCustomFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        assert_eq!(field.seek_exact(b"a").unwrap().doc_freq, 3);
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs);
        assert_eq!(postings.freqs, expected_freqs);
    }

    /// Many terms, each with several docs, all under `BLOCK_SIZE` -- checks
    /// the running `doc_start_fp` delta-threading across more than a
    /// handful of terms (the earlier tests only ever have 2-3 terms).
    #[test]
    fn many_terms_many_docs_each() {
        let mut terms = Vec::new();
        for i in 0..20 {
            let term = format!("term{i:02}").into_bytes();
            let docs: Vec<(i32, i32)> = (0..5).map(|d| (i * 5 + d, (d + 1))).collect();
            terms.push(TermPostings {
                term,
                docs,
                ..Default::default()
            });
        }
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 100,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 100);
        let field = fields.field("f").unwrap();
        assert_eq!(field.num_terms, 20);
        for i in 0..20 {
            let term = format!("term{i:02}").into_bytes();
            let postings = field.postings(&term, Some(&doc_in)).unwrap().unwrap();
            let expected_docs: Vec<i32> = (0..5).map(|d| i * 5 + d).collect();
            let expected_freqs: Vec<i32> = (0..5).map(|d| d + 1).collect();
            assert_eq!(postings.docs, expected_docs, "term{i:02}");
            assert_eq!(postings.freqs, expected_freqs, "term{i:02}");
        }
    }

    /// 26 terms, one per lowercase letter (`"a0".."z0"`), so the field spans
    /// 26 distinct leading bytes.
    ///
    /// This test was written for a multi-block writer that split such a field
    /// into one `.tim` block per leading byte under a `SIGN_MULTI_CHILDREN`
    /// root; that writer was removed because real Lucene cannot read the shape
    /// (see this module's doc comment). What it proves now is the property
    /// that outlived it and is the reason the split was attempted: **a field
    /// whose terms span many leading bytes still reads back term-for-term**,
    /// through the unmodified `blocktree::open`/`postings::DocInput`, from the
    /// single block this writer emits. Every term is looked up independently,
    /// not just the first and last. See
    /// `crates/lucene-search/tests/postings_writer_round_trip.rs`'s
    /// `term_query_finds_correct_docs_across_multiple_tim_blocks` for the same
    /// property through a real `search_term_query`.
    #[test]
    fn a_field_spanning_every_lowercase_leading_byte_reads_back_term_for_term() {
        let mut terms = Vec::new();
        for (i, c) in (b'a'..=b'z').enumerate() {
            let term = vec![c, b'0'];
            let docs: Vec<(i32, i32)> = (0..3).map(|d| ((i as i32) * 3 + d, d + 1)).collect();
            terms.push(TermPostings {
                term,
                docs,
                ..Default::default()
            });
        }
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 78,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 78);
        let field = fields.field("f").unwrap();
        assert_eq!(field.num_terms, 26);
        assert_eq!(field.min_term, b"a0");
        assert_eq!(field.max_term, b"z0");
        for (i, c) in (b'a'..=b'z').enumerate() {
            let term = vec![c, b'0'];
            let postings = field.postings(&term, Some(&doc_in)).unwrap().unwrap();
            let expected_docs: Vec<i32> = (0..3).map(|d| (i as i32) * 3 + d).collect();
            let expected_freqs: Vec<i32> = (0..3).map(|d| d + 1).collect();
            assert_eq!(postings.docs, expected_docs, "term index {i}");
            assert_eq!(postings.freqs, expected_freqs, "term index {i}");
        }
        // A term that doesn't exist must still miss cleanly across a
        // multi-child trie (not just the single-block case).
        assert!(field.seek_exact(b"zz").is_none());
    }

    fn leading_byte_group_terms(n: u8) -> Vec<TermPostings> {
        (0..n)
            .map(|i| TermPostings {
                term: vec![i, b'x'],
                docs: vec![(i as i32, 1)],
                ..Default::default()
            })
            .collect()
    }

    /// Terms spanning many distinct leading bytes all land in the one block
    /// this writer emits, with no per-byte grouping and so no upper bound on
    /// how many distinct leading bytes a field may have. The previous
    /// per-leading-byte split capped this at 33 *and* produced a `.tip` root
    /// real Lucene could not follow -- see this module's `.tim`/`.tip`
    /// comment. 40 is past that old cap on purpose.
    #[test]
    fn many_distinct_leading_bytes_all_round_trip_through_one_block() {
        for n in [2u8, 33, 34, 40] {
            let terms = leading_byte_group_terms(n);
            let input = FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: n as i32,
                has_payloads: false,
                terms: &terms,
            };
            let output = write_single_field(&input, &SEG_ID, SUFFIX)
                .unwrap_or_else(|e| panic!("{n} leading bytes must succeed: {e}"));
            let fis = FieldInfos {
                fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
            };
            let fields = blocktree::open(
                &output.tim,
                &output.tip,
                &output.tmd,
                &fis,
                &SEG_ID,
                SUFFIX,
                n as i32,
            )
            .unwrap_or_else(|e| panic!("{n}-leading-byte output must open: {e}"));
            let f = fields.field("f").unwrap();
            assert_eq!(f.num_terms, n as i64, "n={n}");
            for i in 0..n {
                let stats = f.seek_exact(&[i, b'x']).unwrap_or_else(|| {
                    panic!("term {i} missing with n={n}");
                });
                assert_eq!(stats.doc_freq, 1, "term {i} with n={n}");
            }
        }
    }

    /// A field whose first term is the empty byte string, alongside terms with
    /// several distinct leading bytes.
    ///
    /// The empty term has no leading byte at all, which is what made the
    /// removed multi-block writer fall back to a single block for such a
    /// field. Every field now takes that path, so what this pins is the
    /// remaining half: the empty term is a legal term, sorts first, and reads
    /// back as itself rather than as the block's prefix.
    #[test]
    fn an_empty_term_alongside_distinct_leading_bytes_reads_back_as_itself() {
        let terms = vec![
            TermPostings {
                term: b"".to_vec(),
                docs: vec![(0, 1)],
                ..Default::default()
            },
            TermPostings {
                term: b"m".to_vec(),
                docs: vec![(1, 1)],
                ..Default::default()
            },
            TermPostings {
                term: b"z".to_vec(),
                docs: vec![(2, 1)],
                ..Default::default()
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 3,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 3);
        let field = fields.field("f").unwrap();
        assert_eq!(field.num_terms, 3);
        assert_eq!(
            field.postings(b"", Some(&doc_in)).unwrap().unwrap().docs,
            vec![0]
        );
        assert_eq!(
            field.postings(b"m", Some(&doc_in)).unwrap().unwrap().docs,
            vec![1]
        );
        assert_eq!(
            field.postings(b"z", Some(&doc_in)).unwrap().unwrap().docs,
            vec![2]
        );
    }

    /// Positions write-side byte-level round trip through the existing
    /// unmodified `postings::read_positions` (no query layer here -- see
    /// `crates/lucene-search/tests/postings_writer_round_trip.rs`'s
    /// `phrase_query_finds_correct_docs_over_freshly_written_positions` for
    /// the required phrase-query capstone proof). Covers a singleton term
    /// (`"beta"`, `docFreq == 1`, still needs `.pos` bytes since positions
    /// are independent of the `.doc` singleton-pulsing optimization), a
    /// multi-doc term, and per-doc freq > 1 (multiple occurrences in one
    /// doc), to exercise the position-accumulator reset at each doc's first
    /// occurrence.
    #[test]
    fn positions_round_trip_via_read_positions() {
        let terms = vec![
            TermPostings {
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
                term: b"alpha".to_vec(),
                docs: vec![(0, 2), (3, 1)],
                positions: vec![vec![1, 4], vec![2]],
                offsets: Vec::new(),
            },
            TermPostings {
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
                term: b"beta".to_vec(),
                docs: vec![(1, 3)], // singleton doc, but freq == 3 occurrences
                positions: vec![vec![0, 5, 6]],
                offsets: Vec::new(),
            },
        ];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 3,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

        let fis = FieldInfos {
            fields: vec![field_info(
                0,
                "body",
                IndexOptions::DocsAndFreqsAndPositions,
            )],
        };
        let (fields, doc_in) = open_written(&output, &fis, 4);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");

        let field = fields.field("body").unwrap();

        let positions = field
            .positions(b"alpha", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(
            positions[0].iter().map(|p| p.position).collect::<Vec<_>>(),
            vec![1, 4]
        );
        assert_eq!(
            positions[1].iter().map(|p| p.position).collect::<Vec<_>>(),
            vec![2]
        );

        let positions = field
            .positions(b"beta", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), 1);
        assert_eq!(
            positions[0].iter().map(|p| p.position).collect::<Vec<_>>(),
            vec![0, 5, 6]
        );
    }

    #[test]
    fn rejects_missing_positions_when_index_options_needs_them() {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![], // no positions supplied, but index_options needs them
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::MissingPositions {
                index: 0,
                doc_index: 0
            })
        ));
    }

    #[test]
    fn rejects_positions_freq_mismatch() {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![1]], // only 1 position but freq == 2
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::PositionsFreqMismatch {
                index: 0,
                doc_index: 0,
                positions: 1,
                freq: 2,
            })
        ));
    }

    #[test]
    fn rejects_non_ascending_positions_within_a_doc() {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![3, 3]], // duplicate, not strictly ascending
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::PositionsNotAscending {
                index: 0,
                doc_index: 0,
            })
        ));
    }

    /// Two fields ("title": term-freq only, "body": positions) written in
    /// ONE [`write_fields`] call, sharing the same physical `.doc`/`.pos`/
    /// `.tim`/`.tip`/`.tmd` buffers — `numFields == 2` in `.tmd`, and each
    /// field must be independently seekable/queryable through the existing
    /// unmodified `blocktree::open` read side with no cross-contamination
    /// (see `crates/lucene-search/tests/postings_writer_round_trip.rs`'s
    /// `multi_field_segment_term_queries_are_isolated_per_field` for the
    /// required real `search_term_query` end-to-end proof of the same
    /// property).
    #[test]
    fn write_fields_two_fields_share_one_tmd_and_stay_isolated() {
        let title_terms = vec![
            TermPostings {
                term: b"rust".to_vec(),
                docs: vec![(0, 1)],
                ..Default::default()
            },
            TermPostings {
                term: b"tokyo".to_vec(),
                docs: vec![(1, 1)],
                ..Default::default()
            },
        ];
        let body_terms = vec![
            TermPostings {
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
                term: b"fox".to_vec(),
                docs: vec![(0, 1), (2, 1)],
                positions: vec![vec![3], vec![0]],
                offsets: Vec::new(),
            },
            TermPostings {
                payload_bytes: Vec::new(),
                payload_lengths: Vec::new(),
                term: b"rust".to_vec(), // same bytes as "title"'s term, different field
                docs: vec![(1, 2)],
                positions: vec![vec![0, 5]],
                offsets: Vec::new(),
            },
        ];
        let inputs = vec![
            FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 2,
                has_payloads: false,
                terms: &title_terms,
            },
            FieldPostingsInput {
                field_number: 1,
                index_options: IndexOptions::DocsAndFreqsAndPositions,
                doc_count: 3,
                has_payloads: false,
                terms: &body_terms,
            },
        ];
        let output = write_fields(&inputs, &SEG_ID, SUFFIX).unwrap();

        let fis = FieldInfos {
            fields: vec![
                field_info(0, "title", IndexOptions::DocsAndFreqs),
                field_info(1, "body", IndexOptions::DocsAndFreqsAndPositions),
            ],
        };
        let fields = blocktree::open(
            &output.tim,
            &output.tip,
            &output.tmd,
            &fis,
            &SEG_ID,
            SUFFIX,
            3,
        )
        .expect("write_fields' own bytes must open cleanly");
        assert!(fields.field("title").is_some());
        assert!(fields.field("body").is_some());

        let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");

        let title = fields.field("title").unwrap();
        assert_eq!(title.num_terms, 2);
        let p = title.postings(b"rust", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(p.docs, vec![0]);
        assert!(title.seek_exact(b"fox").is_none()); // no cross-contamination from "body"

        let body = fields.field("body").unwrap();
        assert_eq!(body.num_terms, 2);
        let p = body.postings(b"fox", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(p.docs, vec![0, 2]);
        // "rust" exists in both fields with different postings -- prove
        // "body"'s copy is independent of "title"'s.
        let p = body.postings(b"rust", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(p.docs, vec![1]);
        assert_eq!(p.freqs, vec![2]);
        let positions = body
            .positions(b"rust", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(
            positions[0]
                .iter()
                .map(|pp| pp.position)
                .collect::<Vec<_>>(),
            vec![0, 5]
        );
    }

    #[test]
    fn write_fields_rejects_an_empty_inputs_slice() {
        assert!(matches!(
            write_fields(&[], &SEG_ID, SUFFIX),
            Err(Error::EmptyTerms)
        ));
    }

    #[test]
    fn write_fields_three_fields_each_stay_isolated() {
        let a_terms = vec![TermPostings {
            term: b"alpha".to_vec(),
            docs: vec![(0, 1)],
            ..Default::default()
        }];
        let b_terms = vec![TermPostings {
            term: b"beta".to_vec(),
            docs: vec![(1, 1)],
            ..Default::default()
        }];
        let c_terms = vec![TermPostings {
            term: b"gamma".to_vec(),
            docs: vec![(2, 1)],
            ..Default::default()
        }];
        let inputs = vec![
            FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 1,
                has_payloads: false,
                terms: &a_terms,
            },
            FieldPostingsInput {
                field_number: 1,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 1,
                has_payloads: false,
                terms: &b_terms,
            },
            FieldPostingsInput {
                field_number: 2,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 1,
                has_payloads: false,
                terms: &c_terms,
            },
        ];
        let output = write_fields(&inputs, &SEG_ID, SUFFIX).unwrap();

        let fis = FieldInfos {
            fields: vec![
                field_info(0, "a", IndexOptions::DocsAndFreqs),
                field_info(1, "b", IndexOptions::DocsAndFreqs),
                field_info(2, "c", IndexOptions::DocsAndFreqs),
            ],
        };
        let fields = blocktree::open(
            &output.tim,
            &output.tip,
            &output.tmd,
            &fis,
            &SEG_ID,
            SUFFIX,
            3,
        )
        .expect("write_fields' own bytes must open cleanly for 3 fields");

        let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");
        let a = fields.field("a").unwrap();
        assert_eq!(
            a.postings(b"alpha", Some(&doc_in)).unwrap().unwrap().docs,
            vec![0]
        );
        assert!(a.seek_exact(b"beta").is_none());
        assert!(a.seek_exact(b"gamma").is_none());

        let b = fields.field("b").unwrap();
        assert_eq!(
            b.postings(b"beta", Some(&doc_in)).unwrap().unwrap().docs,
            vec![1]
        );
        assert!(b.seek_exact(b"alpha").is_none());
        assert!(b.seek_exact(b"gamma").is_none());

        let c = fields.field("c").unwrap();
        assert_eq!(
            c.postings(b"gamma", Some(&doc_in)).unwrap().unwrap().docs,
            vec![2]
        );
        assert!(c.seek_exact(b"alpha").is_none());
        assert!(c.seek_exact(b"beta").is_none());
    }

    /// `total_term_freq >= BLOCK_SIZE` alone (via a single doc with a huge
    /// freq, so `docFreq == 1`) is no longer rejected -- only `docFreq >=
    /// BLOCK_SIZE` is, per [`Error::DocFreqTooLargeForPositions`]'s doc
    /// comment. This is the "one doc, many positions" full-position-block
    /// case (see [`positions_full_block_from_one_doc_round_trips`] for the
    /// round-trip proof); this test only checks it no longer errors.
    #[test]
    fn total_term_freq_at_or_above_block_size_from_one_doc_is_now_accepted() {
        let positions: Vec<Vec<i32>> = vec![(0..BLOCK_SIZE).collect()];
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, BLOCK_SIZE)],
            positions,
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        write_single_field(&input, &SEG_ID, SUFFIX)
            .expect("docFreq == 1 stays well under BLOCK_SIZE; only docFreq is now bounded");
    }

    /// `docFreq >= BLOCK_SIZE` while indexing positions used to be rejected
    /// (`Error::DocFreqTooLargeForPositions`), because a `.doc` full block of
    /// a positions-indexing field carries pos/pay skip sub-fields this writer
    /// did not emit. It emits them now ([`PosSkipWriter`]), so the shape is
    /// accepted -- and the level-0 header it produces is read back by the
    /// unmodified `crate::postings::read_full_block_header`, which would
    /// mis-frame the block if the sub-fields were missing or misplaced.
    #[test]
    fn doc_freq_at_or_above_block_size_while_indexing_positions_round_trips() {
        let doc_freq = BLOCK_SIZE + 3;
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: (0..doc_freq).map(|i| (i * 2, 1 + (i % 3))).collect(),
            positions: (0..doc_freq)
                .map(|i| (0..1 + (i % 3)).map(|k| k * 2 + (i % 5)).collect())
                .collect(),
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: doc_freq,
            has_payloads: false,
            terms: &terms,
        };
        let out = write_single_field(&input, &SEG_ID, SUFFIX).expect("write");
        let fis = FieldInfos {
            fields: vec![field_info(0, "f0", IndexOptions::DocsAndFreqsAndPositions)],
        };
        let (fields, doc_in) = open_written(&out, &fis, doc_freq * 2);
        let f = fields.field("f0").expect("field");
        let postings = f
            .postings(b"a", Some(&doc_in))
            .expect("postings")
            .expect("term present");
        assert_eq!(
            postings.docs,
            (0..doc_freq).map(|i| i * 2).collect::<Vec<i32>>()
        );
        assert_eq!(
            postings.freqs,
            (0..doc_freq).map(|i| 1 + (i % 3)).collect::<Vec<i32>>()
        );
    }

    /// Builds a term with `doc_freq` docs, doc IDs `0, 2, 4, .. 2*(doc_freq-1)`
    /// (varied deltas, not all-1, so `write_full_block` never takes a trivial
    /// all-equal-delta shortcut) and per-doc freq `1 + (doc_index % 5)`
    /// (varied, some `!= 1`, so the tail-block's freq-exception path and the
    /// full block's `pfor_encode` both see non-trivial input).
    fn varied_docs_term(term: &[u8], doc_freq: i32) -> TermPostings {
        TermPostings {
            term: term.to_vec(),
            docs: (0..doc_freq).map(|i| (i * 2, 1 + (i % 5))).collect(),
            ..Default::default()
        }
    }

    /// Unlike [`varied_docs_term`] (a constant doc-delta of 2), this
    /// produces genuinely irregular gaps -- deltas cycling through
    /// 1/1/1/50/1/1/1/300/... -- and widely varying freqs (1 up to 1000),
    /// exercising `write_full_block`'s per-block `bits_required(max_delta)`
    /// computation against a real spread of values rather than one that
    /// happens to be uniform.
    fn irregular_docs_term(term: &[u8], doc_freq: i32) -> TermPostings {
        let deltas = [1i32, 1, 1, 50, 1, 1, 1, 300];
        let mut doc_id = 0i32;
        let mut docs = Vec::with_capacity(doc_freq as usize);
        for i in 0..doc_freq {
            if i > 0 {
                doc_id += deltas[(i as usize) % deltas.len()];
            }
            let freq = 1 + (i * 37) % 1000;
            docs.push((doc_id, freq));
        }
        TermPostings {
            term: term.to_vec(),
            docs,
            ..Default::default()
        }
    }

    /// Calls [`write_full_block`] directly with `index_has_freq: false` (so
    /// `rest` -- the block body -- starts immediately with the `bitsPerValue`
    /// token, no impacts-length prefix in front of it) and returns that
    /// token, decoded straight off the wire bytes: `level0NumBytes` (plain
    /// vlong, always `0` here), then `vint15`/`vlong15` (the doc-delta and
    /// blockLength header fields), then the token byte itself. This lets
    /// tests assert *which shape the writer picked* (the byte value), not
    /// just that the reader can still decode whatever shape came out.
    fn full_block_bits_per_value_token(block: &[(i32, i32)], prev_doc_id: i32) -> i8 {
        use lucene_store::data_input::{DataInput, SliceInput};
        let mut out = Vec::new();
        write_full_block(
            &mut out,
            block,
            prev_doc_id,
            false,
            &mut PostingsMaxima::default(),
            &mut PosSkipWriter::new(None, false),
        );
        let mut r = SliceInput::new(&out);
        let _level0_num_bytes = r.read_vlong().unwrap();
        // vint15: i16, non-negative fast path or a following vint for the
        // high bits -- our test blocks' doc deltas are always small enough
        // for the fast path, but handle both for robustness.
        let s = r.read_i16().unwrap();
        if s < 0 {
            r.read_vint().unwrap();
        }
        // vlong15: same shape, long-widening.
        let s = r.read_i16().unwrap();
        if s < 0 {
            r.read_vlong().unwrap();
        }
        r.read_byte().unwrap() as i8
    }

    /// All 256 doc deltas are exactly 1 (a term present in 256 consecutive
    /// docs with no gaps) -- real Lucene's `docRange == BLOCK_SIZE` case.
    /// Asserts the writer picks the `bitsPerValue == 0` "all-256-consecutive"
    /// marker (not just that the block round-trips), then round-trips the
    /// whole term through the unmodified reader and checks the exact doc ID
    /// sequence.
    #[test]
    fn full_block_all_consecutive_picks_zero_token() {
        let block: Vec<(i32, i32)> = (0..BLOCK_SIZE).map(|i| (i, 1)).collect();
        assert_eq!(full_block_bits_per_value_token(&block, -1), 0);

        let term = TermPostings {
            term: b"a".to_vec(),
            docs: block.clone(),
            ..Default::default()
        };
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        assert_eq!(postings.docs, expected_docs);
    }

    /// A block dense enough that the bit-set shape beats the next
    /// `bitsPerValue` step: 256 docs packed into the smallest possible
    /// doc-ID span (deltas of 1 except the very last delta of 2, so
    /// `docRange == 257` -- one more than `BLOCK_SIZE`, avoiding the
    /// `docRange == BLOCK_SIZE` all-consecutive shortcut while staying as
    /// dense as possible). `numBitSetLongs = bits2words(257) = 5`, so the
    /// bit set costs `5 * 64 = 320` bits, while the next `bitsPerValue` step
    /// above `bitsRequired(2) = 2` is `3`, costing `3 * 256 = 768` bits --
    /// the bit set wins. Asserts the writer picks `bitsPerValue < 0` with
    /// the expected `numLongs`, then round-trips through the unmodified
    /// reader.
    #[test]
    fn full_block_dense_picks_bitset_token() {
        let mut block: Vec<(i32, i32)> = (0..BLOCK_SIZE).map(|i| (i, 1)).collect();
        let last = block.len() - 1;
        block[last].0 += 1; // doc IDs 0..254, then 256 (skipping 255): docRange == 257.
        let token = full_block_bits_per_value_token(&block, -1);
        assert_eq!(token, -5);

        let term = TermPostings {
            term: b"a".to_vec(),
            docs: block.clone(),
            ..Default::default()
        };
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        assert_eq!(postings.docs, expected_docs);
    }

    /// The band that separates Lucene **10.5.0** -- the version this port
    /// pins -- from Lucene `main`'s later, looser rule.
    ///
    /// 10.5.0's `Lucene104PostingsWriter.flushDocBlock` picks packed FOR when
    /// `numBitsNextBitsPerValue <= docRange`, i.e. against the bit set's
    /// *unrounded* size. `main` compares against
    /// `numBitSetLongs * Long.SIZE == ceil(docRange/64)*64` instead, which is
    /// `>= docRange`, so every block landing in
    /// `docRange < numBitsNext <= ceil(docRange/64)*64` gets packed FOR on
    /// `main` and the bit set on 10.5.0. Both shapes are legal and this
    /// port's reader takes either, so the round-trip assertions in the tests
    /// around this one pass whichever way it goes; only asserting the chosen
    /// token catches a version mix-up here.
    ///
    /// Construction: 208 deltas of 3 and 48 of 2, so `docRange == 720` and
    /// `bitsRequired(3) == 2`, giving `numBitsNextBitsPerValue == 3 * 256 ==
    /// 768`. Then `768 > 720` (10.5.0: a 12-long bit set) while
    /// `768 <= bits2words(720) * 64 == 12 * 64 == 768` (`main`: packed FOR at
    /// `bitsPerValue == 2`). The band is real but narrow, which is why an
    /// arbitrary block does not land in it -- the sibling
    /// `full_block_dense_picks_bitset_token` (`docRange == 257`) pins the
    /// side of the comparison where both versions agree.
    #[test]
    fn full_block_encoding_choice_matches_lucene_in_the_disputed_band() {
        let deltas: Vec<i32> = std::iter::repeat_n(3, 208)
            .chain(std::iter::repeat_n(2, 48))
            .collect();
        assert_eq!(deltas.len(), BLOCK_SIZE as usize);
        let doc_range: i32 = deltas.iter().sum();
        assert_eq!(doc_range, 720, "test construction");

        let mut doc_id = -1;
        let block: Vec<(i32, i32)> = deltas
            .iter()
            .map(|d| {
                doc_id += d;
                (doc_id, 1)
            })
            .collect();

        // 10.5.0 compares 768 against the unrounded docRange of 720, so the
        // packed branch loses and a 12-long bit set is written. (Lucene
        // `main` compares against 12 * 64 == 768 and would write packed FOR
        // at bitsPerValue == 2 here; asserting `2` means the port drifted
        // onto `main`.)
        assert_eq!(full_block_bits_per_value_token(&block, -1), -12);

        let term = TermPostings {
            term: b"a".to_vec(),
            docs: block.clone(),
            ..Default::default()
        };
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        assert_eq!(postings.docs, expected_docs);
    }

    /// A block whose deltas alternate 1/100 -- sparse enough (`docRange`
    /// around 12,900) that the dense bit-set shape (`~203` words, `~12,992`
    /// bits) is no cheaper than the next `bitsPerValue` step above
    /// `bitsRequired(100) == 7` (`8 * 256 == 2048` bits) -- confirms the
    /// plain positive-`bitsPerValue` `ForUtil` path (pre-existing behavior)
    /// is still chosen when neither special shape wins.
    #[test]
    fn full_block_irregular_picks_plain_packed_token() {
        let block: Vec<(i32, i32)> = (0..BLOCK_SIZE)
            .scan(0i32, |doc_id, i| {
                if i > 0 {
                    *doc_id += if i % 2 == 0 { 1 } else { 100 };
                }
                Some((*doc_id, 1))
            })
            .collect();
        let token = full_block_bits_per_value_token(&block, -1);
        assert_eq!(token, 7); // bitsRequired(100) == 7.

        let term = TermPostings {
            term: b"a".to_vec(),
            docs: block.clone(),
            ..Default::default()
        };
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        assert_eq!(postings.docs, expected_docs);
    }

    /// `docFreq == BLOCK_SIZE` (256): exactly one full block, no tail block
    /// at all -- the boundary the module doc's "no per-term upper bound"
    /// claim rests on. Round-tripped through the existing, unmodified
    /// `blocktree::open`/`DocInput::read_postings` (not just "didn't
    /// panic" -- every doc/freq is asserted).
    #[test]
    fn docfreq_exactly_one_full_block_no_tail() {
        let term = varied_docs_term(b"a", BLOCK_SIZE);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs);
        assert_eq!(postings.freqs, expected_freqs);
    }

    /// `docFreq == BLOCK_SIZE + 1` (257): one full block plus a one-doc
    /// tail block, proving `prev_doc_id` threads correctly from the full
    /// block into the tail block's delta base.
    #[test]
    fn docfreq_one_full_block_plus_one_doc_tail() {
        let term = varied_docs_term(b"a", BLOCK_SIZE + 1);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs);
        assert_eq!(postings.freqs, expected_freqs);
    }

    /// `docFreq == 600`: two full blocks plus an 88-doc tail, exercising
    /// full-block-to-full-block `prev_doc_id` chaining (not just
    /// full-block-to-tail).
    #[test]
    fn docfreq_spans_multiple_full_blocks_plus_tail() {
        let doc_freq = 600;
        assert_eq!(doc_freq / BLOCK_SIZE, 2);
        assert_eq!(doc_freq % BLOCK_SIZE, 88);
        let term = varied_docs_term(b"a", doc_freq);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        assert_eq!(field.seek_exact(b"a").unwrap().doc_freq, doc_freq);
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs);
        assert_eq!(postings.freqs, expected_freqs);
    }

    /// `docFreq == 2 * BLOCK_SIZE` with irregular, non-constant doc-ID gaps
    /// and widely varying freqs (see [`irregular_docs_term`]) -- every
    /// other full-block test in this module uses a constant doc-delta,
    /// which can't distinguish "the per-block bit width was computed from
    /// the real max delta in that block" from "it happened to be right
    /// because every delta was identical."
    #[test]
    fn docfreq_spans_full_blocks_with_irregular_gaps_and_varying_freqs() {
        let doc_freq = 2 * BLOCK_SIZE;
        let term = irregular_docs_term(b"a", doc_freq);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs);
        assert_eq!(postings.freqs, expected_freqs);
    }

    /// A field with `IndexOptions::Docs` (no freqs at all) at `docFreq ==
    /// BLOCK_SIZE` still round-trips through a full block -- proves the
    /// `index_has_freq == false` branch (no impacts field, no `pfor_encode`
    /// freq body) is wired correctly too, not just the freq-carrying case.
    #[test]
    fn docfreq_exactly_one_full_block_no_freqs() {
        let doc_freq = BLOCK_SIZE;
        let docs: Vec<(i32, i32)> = (0..doc_freq).map(|i| (i * 3, 1)).collect();
        let max_doc = docs.last().unwrap().0 + 1;
        let doc_count = docs.len() as i32;
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: docs.clone(),
            ..Default::default()
        }];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::Docs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();
        let postings = field.postings(b"a", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = docs.iter().map(|&(d, _)| d).collect();
        assert_eq!(postings.docs, expected_docs);
    }

    /// Two fields written in one [`write_fields`] call, only one of which
    /// has a full-block term (`docFreq == BLOCK_SIZE`) -- proves full-block
    /// emission for one field doesn't corrupt or bleed into a neighboring
    /// field's own (small, tail-only) postings, mirroring this module's
    /// established multi-field-isolation pattern
    /// (`write_fields_two_fields_share_one_tmd_and_stay_isolated`).
    #[test]
    fn full_block_field_and_small_field_stay_isolated() {
        let full_term = varied_docs_term(b"big", BLOCK_SIZE);
        let small_terms = vec![TermPostings {
            term: b"small".to_vec(),
            docs: vec![(0, 1), (2, 3)],
            ..Default::default()
        }];
        let full_max_doc = full_term.docs.last().unwrap().0 + 1;
        let full_doc_count = full_term.docs.len() as i32;
        let full_terms = vec![full_term.clone()];
        let inputs = vec![
            FieldPostingsInput {
                field_number: 0,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: full_doc_count,
                has_payloads: false,
                terms: &full_terms,
            },
            FieldPostingsInput {
                field_number: 1,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 2,
                has_payloads: false,
                terms: &small_terms,
            },
        ];
        let max_doc = full_max_doc.max(3);
        let output = write_fields(&inputs, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![
                field_info(0, "big_field", IndexOptions::DocsAndFreqs),
                field_info(1, "small_field", IndexOptions::DocsAndFreqs),
            ],
        };
        let fields = blocktree::open(
            &output.tim,
            &output.tip,
            &output.tmd,
            &fis,
            &SEG_ID,
            SUFFIX,
            max_doc,
        )
        .unwrap();
        let doc_in = DocInput::open(&output.doc, &SEG_ID, SUFFIX).expect("open .doc");

        let big = fields.field("big_field").unwrap();
        let big_postings = big.postings(b"big", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = full_term.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = full_term.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(big_postings.docs, expected_docs);
        assert_eq!(big_postings.freqs, expected_freqs);

        let small = fields.field("small_field").unwrap();
        let small_postings = small.postings(b"small", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(small_postings.docs, vec![0, 2]);
        assert_eq!(small_postings.freqs, vec![1, 3]);
    }

    /// Several terms in one field, some below and some spanning full
    /// blocks, each independently seekable -- proves full-block emission
    /// doesn't disturb the term-dictionary metadata threading
    /// (`doc_start_fp` deltas) for neighboring terms in the same block-tree
    /// leaf block.
    #[test]
    fn mixed_small_and_full_block_terms_in_one_field() {
        let small = TermPostings {
            term: b"small".to_vec(),
            docs: vec![(0, 2), (5, 1)],
            ..Default::default()
        };
        let big = varied_docs_term(b"zzz", BLOCK_SIZE + 10);
        let max_doc = big.docs.last().unwrap().0 + 1;
        let doc_count = small.docs.len() as i32 + big.docs.len() as i32;
        let terms = vec![small.clone(), big.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let field = fields.field("f").unwrap();

        let postings = field.postings(b"small", Some(&doc_in)).unwrap().unwrap();
        assert_eq!(postings.docs, vec![0, 5]);
        assert_eq!(postings.freqs, vec![2, 1]);

        let postings = field.postings(b"zzz", Some(&doc_in)).unwrap().unwrap();
        let expected_docs: Vec<i32> = big.docs.iter().map(|&(d, _)| d).collect();
        let expected_freqs: Vec<i32> = big.docs.iter().map(|&(_, f)| f).collect();
        assert_eq!(postings.docs, expected_docs);
        assert_eq!(postings.freqs, expected_freqs);
    }

    /// Builds a term whose `total_term_freq` is exactly `total`, spread
    /// across a handful of docs (`docFreq` well under `BLOCK_SIZE`, so
    /// [`Error::DocFreqTooLargeForPositions`] never trips) with genuinely
    /// irregular per-occurrence position deltas -- cycling through
    /// 1/1/4/1/1/30/1/1/2/... rather than a uniform delta, so a bug in
    /// [`write_full_position_block`]'s flat cross-doc buffering (e.g. an
    /// off-by-one at a doc boundary, or the accumulator failing to reset at
    /// each doc's first occurrence) would produce a wrong position sequence
    /// rather than silently passing on uniform test data. Occurrences are
    /// spread across `num_docs` docs as evenly as possible (the last doc
    /// absorbing any remainder), so a 256-or-257-long chunk genuinely spans
    /// several doc boundaries.
    fn irregular_positions_term(term: &[u8], total: i32, num_docs: i32) -> TermPostings {
        let delta_cycle = [1i32, 1, 4, 1, 1, 30, 1, 1, 2, 7];
        let base_freq = total / num_docs;
        let mut freqs = vec![base_freq; num_docs as usize];
        freqs[(num_docs - 1) as usize] += total - base_freq * num_docs;

        let mut docs = Vec::with_capacity(num_docs as usize);
        let mut positions = Vec::with_capacity(num_docs as usize);
        let mut cycle_idx = 0usize;
        for (doc_idx, &freq) in freqs.iter().enumerate() {
            let doc_id = (doc_idx as i32) * 3; // arbitrary but strictly ascending
            docs.push((doc_id, freq));
            let mut doc_positions = Vec::with_capacity(freq as usize);
            let mut pos = 0i32;
            for _ in 0..freq {
                pos += delta_cycle[cycle_idx % delta_cycle.len()];
                cycle_idx += 1;
                doc_positions.push(pos);
            }
            positions.push(doc_positions);
        }
        TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: term.to_vec(),
            docs,
            positions,
            offsets: Vec::new(),
        }
    }

    /// `total_term_freq == BLOCK_SIZE` (256): exactly one full `.pos`
    /// `PForUtil` block, no vint tail at all -- the boundary
    /// [`write_position_tail`]'s "no per-term upper bound" claim rests on.
    /// Round-tripped through the existing, unmodified
    /// `crate::postings::read_positions` (via `FieldTerms::positions`),
    /// asserting the exact irregular position sequence per doc, not just
    /// counts.
    #[test]
    fn total_term_freq_exactly_one_full_position_block_round_trips() {
        let term = irregular_positions_term(b"a", BLOCK_SIZE, 5);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let expected_positions = term.positions.clone();
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqsAndPositions)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), expected_positions.len());
        for (doc_idx, (got, expected)) in positions.iter().zip(&expected_positions).enumerate() {
            let got_positions: Vec<i32> = got.iter().map(|p| p.position).collect();
            assert_eq!(&got_positions, expected, "doc index {doc_idx}");
        }
    }

    /// `total_term_freq == BLOCK_SIZE + 1` (257): one full `.pos` block plus
    /// a single-occurrence vint tail, proving the flat cross-doc delta
    /// buffer's `start` offset threads correctly from the full block into
    /// the tail. Same irregular-delta construction and per-doc assertion
    /// style as [`total_term_freq_exactly_one_full_position_block_round_trips`].
    /// `lastPosBlockOffset` must be the real byte offset of the vint position
    /// tail, not the constant `0` this writer emitted until the M2 sweep.
    ///
    /// This port's own `read_positions` re-derives the full-block/tail split
    /// from `total_term_freq` and never reads the field, so every round-trip
    /// test in this module passes either way. Real Lucene does not:
    /// `Lucene104PostingsReader.reset` computes `lastPosBlockFP = posStartFP +
    /// lastPosBlockOffset` and `refillPositions` switches to
    /// `refillLastPositionBlock` the instant `posIn.getFilePointer()` reaches
    /// it -- with `0` that is true at the term's very first position block, so
    /// Lucene decodes a `PForUtil` block as a vint tail.
    ///
    /// Asserted two ways: the metadata bytes actually written into `.tim`
    /// decode back (through the unmodified `postings::decode_term_metadata`)
    /// to the expected offset, and reading `.pos` from `posStartFP + that
    /// offset` really does land on the vint tail -- the last
    /// `totalTermFreq % BLOCK_SIZE` position deltas, in order.
    #[test]
    fn last_pos_block_offset_locates_the_vint_position_tail() {
        use lucene_store::data_input::{DataInput, SliceInput};

        // One doc, 300 occurrences: one full 256-position `PForUtil` block
        // plus a 44-occurrence vint tail. `docFreq == 1` keeps this inside the
        // writer's positions scope (`docFreq < BLOCK_SIZE`) while
        // `totalTermFreq == 300 > BLOCK_SIZE` puts `lastPosBlockOffset` on the
        // wire.
        let total = 300usize;
        let positions: Vec<i32> = (0..total as i32).map(|i| i * 3).collect();
        let term = TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, total as i32)],
            positions: vec![positions.clone()],
            ..Default::default()
        };
        let terms = vec![term];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            terms: &terms,
        };
        let out = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();

        // Walk the single `.tim` leaf block down to its per-term metadata
        // region -- the block layout `write_tim_block` emits, read back in
        // wire order.
        let mut r = SliceInput::new(&out.tim);
        codec_util::check_index_header(
            &mut r,
            TERMS_CODEC_NAME,
            BLOCKTREE_VERSION_CURRENT,
            BLOCKTREE_VERSION_CURRENT,
            &SEG_ID,
            SUFFIX,
        )
        .unwrap();
        let code = r.read_vint().unwrap();
        assert_eq!(code >> 1, 1, "one term in the block");
        let code_l = r.read_vlong().unwrap();
        r.skip((code_l >> 3) as usize).unwrap(); // suffix bytes
        let lengths_code = r.read_vint().unwrap();
        r.skip((lengths_code >> 1) as usize).unwrap(); // suffix lengths
        let stats_len = r.read_vint().unwrap() as usize;
        r.skip(stats_len).unwrap(); // per-term stats
        let meta_len = r.read_vint().unwrap() as usize;
        let meta_start = r.position();
        let meta_bytes = r.slice(meta_start, meta_start + meta_len).unwrap();

        let mut mr = SliceInput::new(meta_bytes);
        let meta = crate::postings::decode_term_metadata(
            &mut mr,
            1, // docFreq
            true,
            crate::postings::TermMetadata::EMPTY,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
            total as i64,
        )
        .unwrap();
        assert_ne!(
            meta.last_pos_block_offset, 0,
            "lastPosBlockOffset must point past the full position block"
        );

        // It must land exactly on the vint tail: the remaining 44 position
        // deltas, plain vints, and nothing left over before the footer.
        let tail_fp = meta.pos_start_fp as i64 + meta.last_pos_block_offset;
        let mut pr = SliceInput::new(&out.pos);
        pr.seek(tail_fp as usize).unwrap();
        for i in (BLOCK_SIZE as usize)..total {
            assert_eq!(
                pr.read_vint().unwrap(),
                positions[i] - positions[i - 1],
                "occurrence {i} of the vint tail"
            );
        }
        assert_eq!(
            pr.position(),
            out.pos.len() - codec_util::FOOTER_LENGTH,
            "the vint tail must end exactly where the .pos footer begins"
        );
    }

    #[test]
    fn total_term_freq_one_full_position_block_plus_tail_round_trips() {
        let term = irregular_positions_term(b"a", BLOCK_SIZE + 1, 7);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let expected_positions = term.positions.clone();
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqsAndPositions)],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), expected_positions.len());
        for (doc_idx, (got, expected)) in positions.iter().zip(&expected_positions).enumerate() {
            let got_positions: Vec<i32> = got.iter().map(|p| p.position).collect();
            assert_eq!(&got_positions, expected, "doc index {doc_idx}");
        }
    }

    /// The exact test named in this task's requirements: `docs: vec![(0, 1),
    /// (2, 1)...]`-shaped round trip specifically for the "one doc, huge
    /// freq" full-position-block case accepted by
    /// [`total_term_freq_at_or_above_block_size_from_one_doc_is_now_accepted`]
    /// above -- proves that acceptance actually round-trips correctly, not
    /// just "doesn't error."
    #[test]
    fn positions_full_block_from_one_doc_round_trips() {
        let delta_cycle = [1i32, 1, 4, 1, 1, 30, 1, 1, 2, 7];
        let mut doc_positions = Vec::with_capacity(BLOCK_SIZE as usize);
        let mut pos = 0i32;
        for i in 0..BLOCK_SIZE {
            pos += delta_cycle[(i as usize) % delta_cycle.len()];
            doc_positions.push(pos);
        }
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, BLOCK_SIZE)],
            positions: vec![doc_positions.clone()],
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqsAndPositions)],
        };
        let (fields, doc_in) = open_written(&output, &fis, 1);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, None)
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), 1);
        let got_positions: Vec<i32> = positions[0].iter().map(|p| p.position).collect();
        assert_eq!(got_positions, doc_positions);
    }

    /// Derives deterministic, non-uniform `(startOffset, endOffset)` pairs
    /// from a term's already-built `positions` shape: `startOffset =
    /// position * 10` (strictly increasing since positions are strictly
    /// increasing within a doc, satisfying real Lucene's `startOffset >=
    /// lastStartOffset` assertion) and a length cycling through 1/3/2/5/1 so
    /// [`Error`]-free offset lengths aren't all identical (which would hide a
    /// bug where the writer always took the "length unchanged" tail-encoding
    /// branch).
    fn offsets_from_positions(positions: &[Vec<i32>]) -> Vec<Vec<(i32, i32)>> {
        let length_cycle = [1i32, 3, 2, 5, 1];
        let mut cycle_idx = 0usize;
        positions
            .iter()
            .map(|doc_positions| {
                doc_positions
                    .iter()
                    .map(|&p| {
                        let start = p * 10;
                        let len = length_cycle[cycle_idx % length_cycle.len()];
                        cycle_idx += 1;
                        (start, start + len)
                    })
                    .collect()
            })
            .collect()
    }

    /// Single position per doc, with offsets: every doc's lone occurrence
    /// still needs a correct `startOffsetDelta`/`length` pair even though
    /// there's only ever one occurrence to reset against per doc (no
    /// intra-doc delta to get wrong, but the accumulator must still reset to
    /// `0` at each new doc rather than leaking the previous doc's last
    /// `startOffset`). Round-tripped through the existing, unmodified
    /// `crate::postings::read_positions` (via `FieldTerms::positions`),
    /// asserting exact `(startOffset, endOffset)` pairs, not just positions.
    #[test]
    fn single_position_per_doc_with_offsets_round_trips() {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 1), (2, 1), (5, 1)],
            positions: vec![vec![0], vec![3], vec![7]],
            offsets: vec![vec![(0, 4)], vec![(30, 33)], vec![(70, 77)]],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 3,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(
                0,
                "f",
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            )],
        };
        let (fields, doc_in) = open_written(&output, &fis, 6);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let pay_in =
            crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), 3);
        let expected = [(0, 0, 4), (3, 30, 33), (7, 70, 77)];
        for (doc_idx, (got, &(pos, start, end))) in positions.iter().zip(&expected).enumerate() {
            assert_eq!(got.len(), 1, "doc index {doc_idx}");
            assert_eq!(got[0].position, pos, "doc index {doc_idx}");
            assert_eq!(got[0].start_offset, start, "doc index {doc_idx}");
            assert_eq!(got[0].end_offset, end, "doc index {doc_idx}");
        }
    }

    /// Multiple positions per doc, with offsets: confirms the
    /// `startOffsetDelta` is computed relative to the *previous occurrence in
    /// the same doc*, not the absolute `startOffset`, and that it resets to
    /// `0` at each new doc's first occurrence (not carried over from the
    /// previous doc's last `startOffset`).
    #[test]
    fn multi_position_per_doc_with_offsets_round_trips() {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 3), (4, 2)],
            positions: vec![vec![1, 4, 9], vec![0, 2]],
            offsets: vec![vec![(5, 9), (20, 24), (45, 50)], vec![(0, 3), (10, 15)]],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 2,
            has_payloads: false,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(
                0,
                "f",
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            )],
        };
        let (fields, doc_in) = open_written(&output, &fis, 5);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let pay_in =
            crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), 2);
        let got0: Vec<(i32, i32, i32)> = positions[0]
            .iter()
            .map(|p| (p.position, p.start_offset, p.end_offset))
            .collect();
        assert_eq!(got0, vec![(1, 5, 9), (4, 20, 24), (9, 45, 50)]);
        let got1: Vec<(i32, i32, i32)> = positions[1]
            .iter()
            .map(|p| (p.position, p.start_offset, p.end_offset))
            .collect();
        assert_eq!(got1, vec![(0, 0, 3), (2, 10, 15)]);
    }

    /// `total_term_freq` large enough to force at least one full
    /// `PForUtil`-encoded `.pos`/`.pay` block ([`write_full_position_block`]/
    /// [`write_full_offset_block`]), not just the vint-tail path -- proves
    /// the full-block offset encoding round-trips exact `(startOffset,
    /// endOffset)` pairs, including a length that changes from one
    /// occurrence to the next inside a full block (exercising
    /// `read_positions`'s `PForUtil`-decoded `offset_lengths` array, not the
    /// tail's "reuse unless changed" path). Occurrences span several docs
    /// (`docFreq` well under `BLOCK_SIZE`, so `Error::DocFreqTooLargeForPositions`
    /// never trips) via [`irregular_positions_term`], with offsets derived by
    /// [`offsets_from_positions`].
    #[test]
    fn total_term_freq_full_block_with_offsets_round_trips() {
        let mut term = irregular_positions_term(b"a", BLOCK_SIZE + 1, 5);
        term.offsets = offsets_from_positions(&term.positions);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let expected_positions = term.positions.clone();
        let expected_offsets = term.offsets.clone();
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: false,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info(
                0,
                "f",
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            )],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let pay_in =
            crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), expected_positions.len());
        for (doc_idx, (got, (expected_pos, expected_off))) in positions
            .iter()
            .zip(expected_positions.iter().zip(&expected_offsets))
            .enumerate()
        {
            let got_positions: Vec<i32> = got.iter().map(|p| p.position).collect();
            assert_eq!(&got_positions, expected_pos, "doc index {doc_idx}");
            let got_offsets: Vec<(i32, i32)> =
                got.iter().map(|p| (p.start_offset, p.end_offset)).collect();
            assert_eq!(&got_offsets, expected_off, "doc index {doc_idx}");
        }
    }

    #[test]
    fn rejects_missing_offsets_when_index_options_needs_them() {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            offsets: vec![], // no offsets supplied, but index_options needs them
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::MissingOffsets {
                index: 0,
                doc_index: 0
            })
        ));
    }

    #[test]
    fn rejects_offsets_freq_mismatch() {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![0, 1]],
            offsets: vec![vec![(0, 1)]], // only 1 offset pair but freq == 2
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::OffsetsFreqMismatch {
                index: 0,
                doc_index: 0,
                offsets: 1,
                freq: 2,
            })
        ));
    }

    #[test]
    fn rejects_invalid_offsets() {
        let terms = vec![TermPostings {
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![0, 1]],
            offsets: vec![vec![(5, 8), (3, 6)]], // startOffset decreases
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
            has_payloads: false,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::InvalidOffsets {
                index: 0,
                doc_index: 0,
                occurrence: 1,
            })
        ));
    }

    /// Single position per doc, with payloads and no offsets: every doc's
    /// lone occurrence gets its own distinct payload, so the vint-tail path's
    /// "reuse unless length changes" convention gets exercised across doc
    /// boundaries too (a length change is forced on essentially every
    /// occurrence here). Round-tripped through the existing, unmodified
    /// `crate::postings::read_positions` (via `FieldTerms::positions`).
    #[test]
    fn single_position_per_doc_with_payloads_round_trips() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1), (2, 1), (5, 1)],
            positions: vec![vec![0], vec![3], vec![7]],
            offsets: Vec::new(),
            payload_bytes: payload_run(&[b"x", b"yy", b"zzz"]).0,
            payload_lengths: payload_run(&[b"x", b"yy", b"zzz"]).1,
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 3,
            has_payloads: true,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info_with_payloads(
                0,
                "f",
                IndexOptions::DocsAndFreqsAndPositions,
            )],
        };
        let (fields, doc_in) = open_written(&output, &fis, 6);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let pay_in =
            crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), 3);
        let expected: [(i32, &[u8]); 3] = [(0, b"x"), (3, b"yy"), (7, b"zzz")];
        for (doc_idx, (got, &(pos, payload))) in positions.iter().zip(&expected).enumerate() {
            assert_eq!(got.len(), 1, "doc index {doc_idx}");
            assert_eq!(got[0].position, pos, "doc index {doc_idx}");
            assert_eq!(got[0].payload, payload, "doc index {doc_idx}");
        }
    }

    /// Multiple positions per doc, with payloads: one doc whose occurrences
    /// repeat the *same* payload bytes back-to-back (proving the vint tail's
    /// payload-length-unchanged suppression correctly reuses the previous
    /// length rather than re-writing it, per `Lucene104PostingsWriter.java:
    /// 604-617`) and another doc whose occurrences have varying-length
    /// payloads (forcing a length rewrite each time). Round-tripped through
    /// the existing, unmodified `crate::postings::read_positions`.
    #[test]
    fn multi_position_per_doc_with_payloads_round_trips() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 3), (4, 2)],
            positions: vec![vec![1, 4, 9], vec![0, 2]],
            offsets: Vec::new(),
            // doc 0: same 2-byte payload repeated for all 3 occurrences
            // (length-suppression path). doc 1: varying lengths (1 byte,
            // then 3 bytes).
            payload_bytes: payload_run(&[b"ab", b"ab", b"ab", b"c", b"def"]).0,
            payload_lengths: payload_run(&[b"ab", b"ab", b"ab", b"c", b"def"]).1,
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 2,
            has_payloads: true,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info_with_payloads(
                0,
                "f",
                IndexOptions::DocsAndFreqsAndPositions,
            )],
        };
        let (fields, doc_in) = open_written(&output, &fis, 5);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let pay_in =
            crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), 2);
        let got0: Vec<Vec<u8>> = positions[0].iter().map(|p| p.payload.clone()).collect();
        assert_eq!(got0, vec![b"ab".to_vec(), b"ab".to_vec(), b"ab".to_vec()]);
        let got1: Vec<Vec<u8>> = positions[1].iter().map(|p| p.payload.clone()).collect();
        assert_eq!(got1, vec![b"c".to_vec(), b"def".to_vec()]);
    }

    /// Payloads combined with offsets on the same field: proves the correct
    /// per-position wire interleaving (payload length/bytes *before* offset
    /// fields, in both the full-block `.pay` layout and the vint-tail `.pos`
    /// layout — see [`write_position_tail`]'s doc comment) by asserting both
    /// payload bytes and `(startOffset, endOffset)` decode correctly for the
    /// same occurrences.
    #[test]
    fn payloads_combined_with_offsets_round_trip() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 2), (3, 1)],
            positions: vec![vec![0, 2], vec![1]],
            offsets: vec![vec![(0, 3), (20, 22)], vec![(10, 15)]],
            payload_bytes: payload_run(&[b"p1", b"", b"p3"]).0,
            payload_lengths: payload_run(&[b"p1", b"", b"p3"]).1,
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 2,
            has_payloads: true,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info_with_payloads(
                0,
                "f",
                IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            )],
        };
        let (fields, doc_in) = open_written(&output, &fis, 4);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let pay_in =
            crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), 2);
        assert_eq!(positions[0].len(), 2);
        assert_eq!(positions[0][0].position, 0);
        assert_eq!(positions[0][0].payload, b"p1");
        assert_eq!(
            (positions[0][0].start_offset, positions[0][0].end_offset),
            (0, 3)
        );
        assert_eq!(positions[0][1].position, 2);
        assert!(positions[0][1].payload.is_empty());
        assert_eq!(
            (positions[0][1].start_offset, positions[0][1].end_offset),
            (20, 22)
        );
        assert_eq!(positions[1].len(), 1);
        assert_eq!(positions[1][0].position, 1);
        assert_eq!(positions[1][0].payload, b"p3");
        assert_eq!(
            (positions[1][0].start_offset, positions[1][0].end_offset),
            (10, 15)
        );
    }

    /// `total_term_freq` large enough to force at least one full `PForUtil`-
    /// encoded `.pos`/`.pay` block ([`write_full_position_block`]/
    /// [`write_full_payload_length_block`]), not just the vint-tail path --
    /// proves the full-block payload-length/bytes encoding round-trips
    /// exact payload bytes, including varying lengths inside a full block
    /// (exercising `read_positions`'s `PForUtil`-decoded `payload_lengths`
    /// array and the `.pay` byte-run it gates, not the tail's "reuse unless
    /// changed" path). Occurrences span several docs (`docFreq` well under
    /// `BLOCK_SIZE`, so `Error::DocFreqTooLargeForPositions` never trips) via
    /// [`irregular_positions_term`], with payload lengths cycling through
    /// 1/0/3/2 bytes (including an empty payload) so a bug that assumed every
    /// payload in a block has the same length would produce wrong bytes.
    #[test]
    fn total_term_freq_full_block_with_payloads_round_trips() {
        let mut term = irregular_positions_term(b"a", BLOCK_SIZE + 1, 5);
        let length_cycle = [1usize, 0, 3, 2];
        let mut next_byte = 0u8;
        // The per-doc nesting is the *expectation*; the term itself carries
        // the flat run, so this also pins that the two describe the same
        // occurrences in the same order.
        let expected_payloads: Vec<Vec<Vec<u8>>> = term
            .positions
            .iter()
            .map(|doc_positions| {
                doc_positions
                    .iter()
                    .enumerate()
                    .map(|(occ_idx, _)| {
                        let len = length_cycle[occ_idx % length_cycle.len()];
                        (0..len)
                            .map(|_| {
                                let b = next_byte;
                                next_byte = next_byte.wrapping_add(1);
                                b
                            })
                            .collect::<Vec<u8>>()
                    })
                    .collect()
            })
            .collect();
        for doc_payloads in &expected_payloads {
            for payload in doc_payloads {
                term.payload_lengths.push(payload.len() as u32);
                term.payload_bytes.extend_from_slice(payload);
            }
        }
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let expected_positions = term.positions.clone();
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            has_payloads: true,
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count,
            terms: &terms,
        };
        let output = write_single_field(&input, &SEG_ID, SUFFIX).unwrap();
        let fis = FieldInfos {
            fields: vec![field_info_with_payloads(
                0,
                "f",
                IndexOptions::DocsAndFreqsAndPositions,
            )],
        };
        let (fields, doc_in) = open_written(&output, &fis, max_doc);
        let pos_in =
            crate::postings::PosInput::open(&output.pos, &SEG_ID, SUFFIX).expect("open .pos");
        let pay_in =
            crate::postings::PayInput::open(&output.pay, &SEG_ID, SUFFIX).expect("open .pay");
        let field = fields.field("f").unwrap();

        let positions = field
            .positions(b"a", Some(&doc_in), &pos_in, Some(&pay_in))
            .unwrap()
            .unwrap();
        assert_eq!(positions.len(), expected_positions.len());
        for (doc_idx, (got, (expected_pos, expected_pay))) in positions
            .iter()
            .zip(expected_positions.iter().zip(&expected_payloads))
            .enumerate()
        {
            let got_positions: Vec<i32> = got.iter().map(|p| p.position).collect();
            assert_eq!(&got_positions, expected_pos, "doc index {doc_idx}");
            let got_payloads: Vec<Vec<u8>> = got.iter().map(|p| p.payload.clone()).collect();
            assert_eq!(&got_payloads, expected_pay, "doc index {doc_idx}");
        }
    }

    #[test]
    fn rejects_missing_payloads_when_has_payloads_is_set() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            offsets: Vec::new(),
            // no payload run supplied, but has_payloads is set
            payload_bytes: Vec::new(),
            payload_lengths: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: true,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::PayloadsFreqMismatch {
                index: 0,
                payload_lengths: 0,
                total_term_freq: 1,
            })
        ));
    }

    /// `permute_payload_run` has the direction hazard every reorder has, plus
    /// one of its own: the run carries no per-document index, so it has to be
    /// re-sliced from the **pre-permutation** occurrence counts. Pinned with a
    /// 3-cycle over three documents whose occurrence counts *and* payload
    /// lengths all differ, so neither a reversed permutation nor a
    /// fixed-width slicing bug can pass.
    #[test]
    fn permute_payload_run_moves_each_documents_payloads_with_it() {
        // doc order 0,1,2: doc 0 has 1 occurrence (1 byte), doc 1 has 2
        // (1 and 3 bytes), doc 2 has 1 (2 bytes).
        let bytes = vec![0xA0, 0xB0, 0xB1, 0xB2, 0xB3, 0xC0, 0xC1];
        let lengths = vec![1u32, 1, 3, 2];
        let counts = vec![1u32, 2, 1];
        let (got_bytes, got_lengths) = permute_payload_run(&bytes, &lengths, &counts, &[2, 0, 1]);
        assert_eq!(got_lengths, vec![2, 1, 1, 3]);
        assert_eq!(got_bytes, vec![0xC0, 0xC1, 0xA0, 0xB0, 0xB1, 0xB2, 0xB3]);
    }

    /// A truncated run must not panic: `PayloadsFreqMismatch` is where a
    /// caller that built the two out of step is meant to be told, and a slice
    /// panic here would take the process down first. The out-of-range
    /// permutation entry is the same rule from the other side.
    #[test]
    fn permute_payload_run_saturates_on_a_run_shorter_than_its_counts() {
        // Two documents claiming 1 and 3 occurrences over a run that only has
        // one length, and a permutation whose second entry names a document
        // that does not exist. Both truncations drop their run rather than
        // panicking or reading somebody else's bytes.
        let (bytes, lengths) = permute_payload_run(&[1, 2], &[1u32], &[1u32, 3], &[1, 5]);
        assert!(lengths.is_empty(), "got {lengths:?}");
        assert!(bytes.is_empty(), "got {bytes:?}");
        // The in-range half of the same run still moves correctly.
        let (bytes, lengths) = permute_payload_run(&[1, 2], &[1u32], &[1u32, 3], &[0, 1]);
        assert_eq!(lengths, vec![1]);
        assert_eq!(bytes, vec![1]);
    }

    /// The flat run's second invariant: the bytes must be exactly the
    /// concatenation its lengths describe. The nested shape could not express
    /// this state at all (a `Vec<u8>` is its own length), so it is new
    /// validation for a new representation rather than a re-spelling.
    #[test]
    fn rejects_payload_bytes_that_do_not_match_their_lengths() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![0, 1]],
            offsets: Vec::new(),
            // lengths sum to 5, but only 3 bytes are supplied
            payload_bytes: b"abc".to_vec(),
            payload_lengths: vec![2, 3],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: true,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::PayloadBytesMismatch {
                index: 0,
                payload_bytes: 3,
                expected: 5,
            })
        ));
    }

    /// A payload length is a `u32` in the run and an `int` on the wire, so a
    /// length past `i32::MAX` has to be rejected rather than written as a
    /// negative vint. Unreachable through `IndexWriter` (no allocation that
    /// large exists here) and reachable by any direct `postings_writer`
    /// caller, which is the whole reason the check is at the boundary.
    #[test]
    fn rejects_a_payload_length_that_does_not_fit_an_i32() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            offsets: Vec::new(),
            payload_bytes: Vec::new(),
            payload_lengths: vec![i32::MAX as u32 + 1],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: true,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::PayloadLengthTooLarge {
                index: 0,
                occurrence: 0,
                length: 2_147_483_648,
            })
        ));
    }

    #[test]
    fn rejects_payloads_freq_mismatch() {
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![0, 1]],
            offsets: Vec::new(),
            // only 1 payload length but freq == 2
            payload_bytes: payload_run(&[b"x"]).0,
            payload_lengths: payload_run(&[b"x"]).1,
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            has_payloads: true,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::PayloadsFreqMismatch {
                index: 0,
                payload_lengths: 1,
                total_term_freq: 2,
            })
        ));
    }
}
