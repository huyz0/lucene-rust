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
//! - **One physical `.tim` block per field, OR multiple sibling leaf blocks
//!   under a single multi-child `.tip` root — never a deeper/floor-split
//!   trie.** This is the load-bearing scope restriction, added in the
//!   "multi-block writer" task after the single-block-only writer proved out.
//!   The splitting policy is deliberately the simplest one that produces a
//!   *valid* trie without floor blocks or a second trie level: **group a
//!   field's (already-sorted) terms by their first byte.** If every term
//!   shares one leading byte (or there's only one term), the field still gets
//!   the original single-block/single-`SIGN_NO_CHILDREN`-root shape,
//!   unchanged from before. If the terms span 2..=33 distinct leading bytes,
//!   each group becomes its own leaf `.tim` block (storing only each term's
//!   bytes *after* the shared leading byte — that byte is the trie label, not
//!   stored twice) addressed by its own `SIGN_NO_CHILDREN` child node, and the
//!   field's `.tip` root is a single `SIGN_MULTI_CHILDREN` node (always
//!   `ChildSaveStrategy::ARRAY`, the simplest of the three label encodings to
//!   emit) whose children are exactly those leaf nodes, with no output of its
//!   own. **Explicitly still unimplemented**: floor sub-blocks (a single
//!   leading-byte group too large for one block), a second/deeper trie level
//!   (needed if 34+ distinct leading bytes appear, or if finer splitting
//!   within one leading byte is ever needed), and the `BITS`/`REVERSE_ARRAY`
//!   label-encoding strategies (read-side supports all three; this writer
//!   only ever emits `ARRAY`). A field needing more than 33 leading-byte
//!   groups is rejected with `Error::TooManyLeadingByteGroups` rather than
//!   silently misencoding the 5-bit strategy-byte-count field. A field
//!   containing an empty-byte-string term also falls back to the
//!   single-block path unconditionally (there's no leading byte to
//!   strip/route on for that term).
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
//!   space — see `docs/parity.md` for that scope cut) and impacts are always
//!   an empty byte region (no competitive-impact computation) — the reader
//!   accepts an empty impacts run and this writer never emits any queries
//!   that need real ones. **`docFreq >= LEVEL1_NUM_DOCS` (8192) is now
//!   supported too**: for every complete span of [`LEVEL1_FACTOR`] (32) full
//!   level-0 blocks, a level-1 skip entry ([`write_level1_span`]) is emitted
//!   immediately before them — the exact write-side inverse of
//!   `crate::postings::read_level1_entry`/`LazyDocsCursor::skip_level1_to`.
//!   Like level-0, the level-1 entry's impacts region is always empty (no
//!   competitive-impact computation at either level); since positions never
//!   co-occur with a full block in the first place (`total_term_freq <
//!   BLOCK_SIZE` is required whenever positions are indexed, and
//!   `docFreq >= LEVEL1_NUM_DOCS` implies `total_term_freq >= 8192`), the
//!   level-1 entry's `indexHasPos`-gated pos/pay sub-fields are never
//!   reachable from this writer and are simply never written. **There is no
//!   further per-term docFreq ceiling**: the reader has no level-2 skip
//!   structure (`Lucene104` postings only ever have levels 0 and 1), so a
//!   term spanning any number of level-1 spans plus a final partial span
//!   round-trips the same way arbitrarily large `docFreq` already did below
//!   `LEVEL1_NUM_DOCS`.
//! - **Term frequency, positions, and now offsets too — still no
//!   payloads.** `IndexOptions::Docs`/`DocsAndFreqs`/
//!   `DocsAndFreqsAndPositions`/`DocsAndFreqsAndPositionsAndOffsets` are all
//!   accepted; `.pos` is only written once a field indexes positions, and
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
//!   `num_full_blocks` loop — so no level-0/level-1-equivalent skip structure
//!   exists for positions/offsets to write the write-side inverse of in the
//!   first place. **Still deferred**: payloads (this writer has none at
//!   all, so that part of `read_positions`'s `has_payloads` branch is
//!   unreachable rather than untested). **`docFreq` must still stay below
//!   `BLOCK_SIZE` whenever a term indexes
//!   positions** ([`Error::DocFreqTooLargeForPositions`]): this writer's
//!   `.doc`-side [`write_full_block`] never emits the pos/pay skip
//!   sub-fields `read_full_block_header` expects on a full `.doc` block for
//!   a positions-indexing field, so a term can never be allowed to push its
//!   `.doc` stream into the full-block path while also indexing positions.
//!   Since `docFreq <= total_term_freq` always, this is a strictly separate,
//!   independent ceiling from the (now unbounded) `total_term_freq` one
//!   above — a term can have enormous `total_term_freq` from few, high-freq
//!   docs and still round-trip, as long as `docFreq` itself stays under
//!   `BLOCK_SIZE`.
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
//! - `.tim`: `IndexHeader(codec="BlockTreeTermsDict")`, then, per field, one
//!   physical block (single-block case) or one physical block per
//!   leading-byte group (multi-block case), each block being
//!   (`entCount << 1 | 1` code, `isLeafBlock` + `NO_COMPRESSION` code, suffix
//!   bytes, suffix lengths, per-term stats, per-term postings metadata — see
//!   [`write_term_metadata`]), `Footer`.
//! - `.tip`: `IndexHeader(codec="BlockTreeTermsIndex")`, then, per field,
//!   either one `SIGN_NO_CHILDREN`/`hasTerms`/no-floor root node pointing at
//!   the field's single `.tim` block (single-block case), or one
//!   `SIGN_NO_CHILDREN`/`hasTerms`/no-floor leaf node per leading-byte group
//!   followed by one `SIGN_MULTI_CHILDREN`/`ChildSaveStrategy::ARRAY` root
//!   node (no output of its own) whose children are exactly those leaf nodes
//!   (multi-block case) — see [`write_multi_children_root`]. `Footer`.
//! - `.tmd`: `IndexHeader(codec="BlockTreeTermsMeta")`, the postings writer's
//!   own embedded header (`IndexHeader(codec="Lucene104PostingsWriterTerms")`,
//!   `indexBlockSize = 256`), `numFields = inputs.len()`, then each field's
//!   record (`fieldNumber, numTerms, sumTotalTermFreq/sumDocFreq, docCount, minTerm/maxTerm,
//!   indexStart/rootFP/indexEnd`), `indexLength`, `termsLength`, `Footer`.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_output::DataOutput;
use std::ops::Range;

use crate::blocktree::{
    CHILD_STRATEGY_ARRAY, LEAF_NODE_HAS_TERMS, POSTINGS_BLOCK_SIZE, POSTINGS_TERMS_CODEC,
    POSTINGS_VERSION_CURRENT, SIGN_MULTI_CHILDREN, SIGN_NO_CHILDREN, TERMS_CODEC_NAME,
    TERMS_INDEX_CODEC_NAME, TERMS_META_CODEC_NAME, VERSION_CURRENT as BLOCKTREE_VERSION_CURRENT,
};
use crate::field_infos::IndexOptions;
use crate::for_util;
use crate::postings::{
    write_group_vints, BLOCK_SIZE, DOC_CODEC, LEVEL1_NUM_DOCS, PAY_CODEC, POS_CODEC,
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
    #[error("write_single_field: term at index {index} has freq < 1")]
    NonPositiveFreq { index: usize },
    #[error(
        "write_single_field: only IndexOptions::Docs/DocsAndFreqs/DocsAndFreqsAndPositions/\
         DocsAndFreqsAndPositionsAndOffsets is supported, got {0:?}"
    )]
    UnsupportedIndexOptions(IndexOptions),
    #[error(
        "write_single_field: term at index {index} indexes positions and has docFreq \
         {doc_freq} >= BLOCK_SIZE ({BLOCK_SIZE}); this writer's `.doc` full-block path never \
         emits the pos/pay skip sub-fields a positions-indexing field's full block needs, so \
         docFreq must stay below BLOCK_SIZE whenever positions are indexed (total_term_freq \
         itself has no such ceiling — see the module doc)"
    )]
    DocFreqTooLargeForPositions { index: usize, doc_freq: i64 },
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
        "write_fields: field has {0} distinct leading-byte groups, but this writer's multi-child \
         trie root only supports 2..=33 children (ChildSaveStrategy::ARRAY's 5-bit strategy-byte-count field)"
    )]
    TooManyLeadingByteGroups(usize),
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
#[derive(Debug, Clone, Default)]
pub struct TermPostings {
    pub term: Vec<u8>,
    pub docs: Vec<(i32, i32)>,
    pub positions: Vec<Vec<i32>>,
    pub offsets: Vec<Vec<(i32, i32)>>,
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
    // Only written at all if at least one field indexes offsets (this
    // writer never has payloads), same "no file needed" convention as
    // `.pos`.
    let any_offsets = inputs
        .iter()
        .any(|input| input.index_options.subsumes_offsets());
    let mut pay = Vec::new();
    if any_offsets {
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
            // `docFreq >= LEVEL1_NUM_DOCS` (8192): emit a level-1 skip entry
            // before every complete span of `LEVEL1_FACTOR` (32) full
            // level-0 blocks, mirroring `DocInput::read_postings`'s own
            // `doc_count_left >= LEVEL1_NUM_DOCS` loop exactly.
            while t.docs.len() - start >= LEVEL1_NUM_DOCS as usize {
                let span = &t.docs[start..start + LEVEL1_NUM_DOCS as usize];
                prev_doc_id = write_level1_span(
                    &mut doc,
                    span,
                    prev_doc_id,
                    &mut level1_last_doc_id,
                    index_has_freq,
                );
                start += LEVEL1_NUM_DOCS as usize;
            }
            while t.docs.len() - start >= BLOCK_SIZE as usize {
                let block = &t.docs[start..start + BLOCK_SIZE as usize];
                prev_doc_id = write_full_block(&mut doc, block, prev_doc_id, index_has_freq);
                start += BLOCK_SIZE as usize;
            }
            if start < t.docs.len() {
                write_tail_block(&mut doc, &t.docs[start..], prev_doc_id, index_has_freq);
            }
        }

        // `pos_start_fp[i]` is term `i`'s absolute byte offset into the
        // shared `.pos` buffer, same convention as `doc_start_fp` above. Left
        // at `0` (never read, see `write_term_metadata`) when this field
        // doesn't index positions.
        let index_has_offsets = input.index_options.subsumes_offsets();
        let mut pos_start_fp = vec![0u64; input.terms.len()];
        let mut pay_start_fp = vec![0u64; input.terms.len()];
        if index_has_positions {
            for (i, t) in input.terms.iter().enumerate() {
                pos_start_fp[i] = pos.len() as u64;
                if index_has_offsets {
                    pay_start_fp[i] = pay.len() as u64;
                }
                write_position_tail(
                    &mut pos,
                    &mut pay,
                    &t.positions,
                    &t.offsets,
                    index_has_offsets,
                );
            }
        }

        // ---- this field's .tim block(s) + .tip node(s) ----
        // See the module doc's "Scope" section: a field whose terms span
        // only one leading byte (or has a single term, or contains an
        // empty-byte-string term) keeps the original single-block/
        // single-`SIGN_NO_CHILDREN`-root shape; a field spanning 2..=33
        // distinct leading bytes gets one leaf block per leading-byte group
        // under a `SIGN_MULTI_CHILDREN` root (see [`write_multi_children_root`]).
        let groups = group_terms_by_leading_byte(input.terms);
        let (index_start, root_fp, index_end) = if groups.len() <= 1 {
            let block_fp = write_tim_block(
                &mut tim,
                input.terms,
                &doc_start_fp,
                &pos_start_fp,
                &pay_start_fp,
                0,
                input.index_options,
                index_has_positions,
            );
            let index_start = tip.len();
            let root_fp_abs = write_leaf_node(&mut tip, block_fp as u64);
            let index_end = tip.len();
            (index_start, root_fp_abs - index_start, index_end)
        } else {
            if groups.len() > 33 {
                return Err(Error::TooManyLeadingByteGroups(groups.len()));
            }
            let index_start = tip.len();
            let mut labels = Vec::with_capacity(groups.len());
            let mut child_fps_abs = Vec::with_capacity(groups.len());
            for (label, range) in &groups {
                let group_terms = &input.terms[range.clone()];
                let block_fp = write_tim_block(
                    &mut tim,
                    group_terms,
                    &doc_start_fp[range.clone()],
                    &pos_start_fp[range.clone()],
                    &pay_start_fp[range.clone()],
                    1, // strip the shared leading byte (it's the trie label)
                    input.index_options,
                    index_has_positions,
                );
                let child_fp_abs = write_leaf_node(&mut tip, block_fp as u64);
                labels.push(*label);
                child_fps_abs.push(child_fp_abs);
            }
            let root_fp_abs = write_multi_children_root(&mut tip, &labels, &child_fps_abs);
            let index_end = tip.len();
            (index_start, root_fp_abs - index_start, index_end)
        };

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
    if any_offsets {
        codec_util::write_footer(&mut pay);
    }
    codec_util::write_footer(&mut tim);
    codec_util::write_footer(&mut tip);

    tmd.write_i64(tip.len() as i64 - codec_util::FOOTER_LENGTH as i64); // indexLength
    tmd.write_i64(tim.len() as i64 - codec_util::FOOTER_LENGTH as i64); // termsLength
    codec_util::write_footer(&mut tmd);

    Ok(Output {
        doc,
        pos,
        pay,
        tim,
        tip,
        tmd,
    })
}

/// Groups `terms` (already sorted ascending, per [`write_fields`]'s caller
/// obligations) into maximal runs sharing the same first byte, returning
/// `(label, range)` pairs in ascending label order — the splitting policy
/// described in the module doc's "Scope" section. Falls back to a single
/// group spanning every term (i.e. the caller takes the single-block path)
/// whenever splitting wouldn't be safe: `terms` is empty, has only one term,
/// contains an empty-byte-string term (no leading byte to strip/route on), or
/// every term happens to share one leading byte already.
fn group_terms_by_leading_byte(terms: &[TermPostings]) -> Vec<(u8, Range<usize>)> {
    if terms.len() <= 1 || terms.iter().any(|t| t.term.is_empty()) {
        return vec![(0, 0..terms.len())];
    }
    let mut groups = Vec::new();
    let mut start = 0;
    for i in 1..=terms.len() {
        if i == terms.len() || terms[i].term[0] != terms[start].term[0] {
            groups.push((terms[start].term[0], start..i));
            start = i;
        }
    }
    groups
}

/// Writes one physical `.tim` leaf block for `terms` (a contiguous,
/// already-sorted slice — either a whole field in the single-block case, or
/// one leading-byte group in the multi-block case), returning the block's
/// absolute byte offset into `tim`. `strip_prefix_len` is `0` for the
/// single-block case (the block stores each term's full bytes as its
/// "suffix", matching the trie root's empty path prefix) or `1` for a
/// leading-byte group (the block stores only the bytes *after* the shared
/// leading byte, which the enclosing `SIGN_MULTI_CHILDREN` trie node already
/// encodes as that child's label — see [`crate::blocktree::collect_leaf_blocks`]'s
/// doc comment for why a block only ever stores its own suffix). `doc_start_fp`/
/// `pos_start_fp` must be the same length as `terms` and already sliced to
/// line up with it; metadata deltas are threaded fresh starting from
/// `TermMetadata::EMPTY` for this block alone (`write_term_metadata`'s
/// `base_doc_start_fp`/`base_pos_start_fp` both start at 0 here), matching
/// `SegmentTermsEnumFrame`'s per-frame reset the read side
/// (`crate::blocktree::decode_block`) already assumes — blocks never share
/// metadata-delta state across a floor split *or* across sibling leaf blocks.
#[allow(clippy::too_many_arguments)]
fn write_tim_block(
    tim: &mut Vec<u8>,
    terms: &[TermPostings],
    doc_start_fp: &[u64],
    pos_start_fp: &[u64],
    pay_start_fp: &[u64],
    strip_prefix_len: usize,
    index_options: IndexOptions,
    index_has_positions: bool,
) -> usize {
    let block_fp = tim.len();
    let ent_count = terms.len() as u32;
    let code = (ent_count << 1) | 1; // isLastInFloor
    tim.write_vint(code as i32);

    let mut suffix_bytes = Vec::new();
    let mut suffix_lengths = Vec::new();
    let mut stats = Vec::new();
    for t in terms {
        let suffix = &t.term[strip_prefix_len..];
        suffix_bytes.write_bytes(suffix);
        suffix_lengths.write_vint(suffix.len() as i32);
        let doc_freq = t.docs.len() as u32;
        let total_term_freq: i64 = t.docs.iter().map(|&(_, f)| f as i64).sum();
        stats.write_vint((doc_freq << 1) as i32); // never singleton-run-encoded
        if index_options != IndexOptions::Docs {
            stats.write_vlong(total_term_freq - doc_freq as i64);
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
        index_has_positions,
        index_options.subsumes_offsets(),
    );
    tim.write_vint(meta.len() as i32);
    tim.write_bytes(&meta);

    block_fp
}

/// Writes one `SIGN_NO_CHILDREN`/`hasTerms`/no-floor `.tip` node pointing at
/// `block_fp` (a `.tim` block's absolute offset), returning this node's own
/// absolute offset into `tip` — shared by the single-block root and, in the
/// multi-block case, every one of the `SIGN_MULTI_CHILDREN` root's leaf
/// children (see [`write_multi_children_root`]).
fn write_leaf_node(tip: &mut Vec<u8>, block_fp: u64) -> usize {
    let fp = tip.len();
    let output_fp_bytes = 8usize; // keep it simple: always 8 bytes, same as blocktree.rs's test Builder
    let header =
        (SIGN_NO_CHILDREN as u8) | ((output_fp_bytes as u8 - 1) << 2) | (LEAF_NODE_HAS_TERMS as u8);
    tip.push(header);
    tip.extend_from_slice(&block_fp.to_le_bytes());
    tip.extend_from_slice(&0u64.to_le_bytes()); // 8-byte over-read pad, `load_node`'s SIGN_NO_CHILDREN reads up to fp+1..fp+9
    fp
}

/// Writes one `SIGN_MULTI_CHILDREN` root node (`ChildSaveStrategy::ARRAY`,
/// no output of its own) whose children are exactly the leaf nodes at
/// `child_fps_abs` (already written into `tip`, one per entry of `labels`, in
/// the same ascending-label order), returning the root node's own absolute
/// offset into `tip`.
///
/// Mirrors `TrieReader.loadMultiChildrenNode`'s read side
/// (`crate::blocktree::load_node`) for the "no own output" branch
/// (`term & 0x20 == 0`, so `strategy_fp = fp + 3` — a 3-byte header packing
/// `sign`/`childrenDeltaFpBytes`/`hasOutput=0`/`childSaveStrategy`/
/// `strategyBytes`/`minChildrenLabel`) followed by `ChildSaveStrategy::ARRAY`'s
/// layout (`crate::blocktree::multi_children_labels_and_fps`'s `ARRAY` arm):
/// `labels.len() - 1` raw label bytes (every label after `min_label`, which
/// the header already carries) then `labels.len()` children-delta-fp entries,
/// each `children_delta_fp_bytes` (fixed at 8 here, matching [`write_leaf_node`]'s
/// own 8-byte output-fp convention) little-endian bytes encoding
/// `this_root_fp - child_fp` (the read side's "delta from parent" convention;
/// [`crate::blocktree::multi_children_labels_and_fps`] rejects a delta
/// exceeding the parent's own fp, so every child must already be written
/// — hence must be written to `tip` before this call).
///
/// `labels.len()` must be in `2..=33` (checked by the caller,
/// `Error::TooManyLeadingByteGroups`) — the header's 5-bit
/// `strategyBytes - 1` field can only address `labels.len() - 1` in `1..=32`.
fn write_multi_children_root(tip: &mut Vec<u8>, labels: &[u8], child_fps_abs: &[usize]) -> usize {
    debug_assert_eq!(labels.len(), child_fps_abs.len());
    debug_assert!((2..=33).contains(&labels.len()));

    let root_fp = tip.len();
    let child_count = labels.len() as u32;
    let children_delta_fp_bytes = 8u32; // field value; raw header bits are (this - 1)
    let strategy_bytes = child_count - 1; // field value; raw header bits are (this - 1)
    let min_label = labels[0];

    let term_u32: u32 = SIGN_MULTI_CHILDREN
        | ((children_delta_fp_bytes - 1) << 2)
        // bit 5 (0x20, "has own output") deliberately left clear: this root
        // has no terms of its own, only children.
        | (CHILD_STRATEGY_ARRAY << 9)
        | ((strategy_bytes - 1) << 11)
        | ((min_label as u32) << 16);
    let header_bytes = term_u32.to_le_bytes();
    tip.push(header_bytes[0]);
    tip.push(header_bytes[1]);
    tip.push(header_bytes[2]);

    // ChildSaveStrategy::ARRAY: `labels[1..]` written verbatim (ascending, one
    // byte each) right after the header.
    for &label in &labels[1..] {
        tip.push(label);
    }
    // Then one children-delta-fp entry per label (including `labels[0]`),
    // in the same order, each `this_root_fp - child_fp`, little-endian,
    // `children_delta_fp_bytes` (8) bytes wide.
    for &child_fp in child_fps_abs {
        let delta = (root_fp - child_fp) as u64;
        tip.extend_from_slice(&delta.to_le_bytes());
    }

    root_fp
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
    ) {
        return Err(Error::UnsupportedIndexOptions(input.index_options));
    }
    if input.terms.is_empty() {
        return Err(Error::EmptyTerms);
    }
    for (i, w) in input.terms.windows(2).enumerate() {
        if w[0].term >= w[1].term {
            return Err(Error::TermsNotSorted(i + 1));
        }
    }
    let index_has_positions = input.index_options.subsumes_positions();
    let index_has_offsets = input.index_options.subsumes_offsets();
    for (i, t) in input.terms.iter().enumerate() {
        if t.docs.is_empty() {
            return Err(Error::EmptyPostings(i));
        }
        for (j, &(_, freq)) in t.docs.iter().enumerate() {
            if freq < 1 {
                return Err(Error::NonPositiveFreq { index: i });
            }
            if j > 0 && t.docs[j - 1].0 >= t.docs[j].0 {
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
            let doc_freq = t.docs.len() as i64;
            if doc_freq >= BLOCK_SIZE as i64 {
                return Err(Error::DocFreqTooLargeForPositions { index: i, doc_freq });
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

/// Writes one full 256-doc `.doc` block — a level-0 skip header
/// (`level0NumBytes` skip pointer, `docDelta`, `blockLength`, an always-empty
/// impacts region) followed by the doc-delta/freq body — the exact
/// write-side inverse of `crate::postings::read_full_block_header`/
/// `decode_full_block_body`. This deliberately never writes the header's
/// pos/pay skip fields (only present on the wire when the field indexes
/// positions *and* has freqs): a term can only reach this full-block path at
/// `docFreq >= BLOCK_SIZE`, and [`validate_field`]'s
/// [`Error::DocFreqTooLargeForPositions`] check already rejects exactly that
/// combination (`docFreq >= BLOCK_SIZE` while positions are indexed) before a
/// full block is ever built — positions genuinely cannot co-occur with this
/// function today. `block` must be exactly `BLOCK_SIZE` (256)
/// `(doc_id, freq)` pairs, ascending. Returns `block`'s last doc ID, which
/// the caller threads through as `prev_doc_id` for the next full block or
/// the trailing tail block (`Lucene104PostingsReader.prefixSum`'s running
/// per-term base).
///
/// Doc deltas always take the plain positive-`bitsPerValue` `ForUtil` shape
/// (`decode_full_block_body`'s `bitsPerValue > 0` branch) — this writer never
/// emits the `bitsPerValue == 0` "all-256-consecutive" or `bitsPerValue < 0`
/// dense-bitset alternate encodings the real writer sometimes picks for
/// space efficiency (see the module doc's scope section and
/// `docs/parity.md`). Freqs (when `index_has_freq`) go through
/// `for_util::pfor_encode` directly — its on-wire token/body shape is byte-
/// identical to what `for_util::pfor_decode` (called from
/// `decode_full_block_body`) expects, so no re-derivation of that format
/// happens here.
fn write_full_block(
    out: &mut Vec<u8>,
    block: &[(i32, i32)],
    prev_doc_id: i32,
    index_has_freq: bool,
) -> i32 {
    debug_assert_eq!(block.len(), BLOCK_SIZE as usize);

    // Everything from here down is what `blockLength` measures (i.e. what
    // `read_full_block_header` reads as `body_end - r.position()`
    // immediately after `blockLength` itself) -- build it in a scratch
    // buffer first so `blockLength`'s value is known before the header is
    // written.
    let mut rest = Vec::new();
    if index_has_freq {
        rest.write_vint(0); // impacts byte-length: always an empty region
                            // (no competitive-impact computation, see the module doc).
    }

    let mut deltas = [0u32; for_util::BLOCK_SIZE];
    let mut prev = prev_doc_id;
    let mut max_delta = 0u32;
    for (i, &(doc_id, _)) in block.iter().enumerate() {
        let delta = (doc_id - prev) as u32;
        deltas[i] = delta;
        max_delta = max_delta.max(delta);
        prev = doc_id;
    }
    // `bits_required` returns 0 only for an all-zero input; every delta here
    // is `>= 1` (ascending, no duplicates), so `max_delta >= 1` and this is
    // always `>= 1` in practice -- `.max(1)` just keeps the invariant
    // explicit rather than relying on that fact silently.
    let bits_per_value = for_util::bits_required(max_delta).max(1);
    rest.write_byte(bits_per_value as u8);
    for_util::for_encode(&deltas, bits_per_value, &mut rest);

    if index_has_freq {
        let mut freqs = [0u32; for_util::BLOCK_SIZE];
        for (i, &(_, freq)) in block.iter().enumerate() {
            freqs[i] = freq as u32;
        }
        for_util::pfor_encode(&mut freqs, &mut rest);
    }

    let last_doc_id = block[block.len() - 1].0;
    out.write_vlong(0); // level0NumBytes: a skip pointer this reader parses
                        // but never uses (see read_full_block_header), so any
                        // valid vlong is fine here.
    write_vint15(out, last_doc_id - prev_doc_id);
    write_vlong15(out, rest.len() as i64);
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
/// entry's freq-gated metadata) and a `numImpactBytes` `i16`, both fixed at
/// `2`/`0` here: the impacts region is always empty (no competitive-impact
/// computation at level 1, mirroring [`write_full_block`]'s own level-0
/// choice), so `skip1EndFP` only ever needs to span the two bytes of
/// `numImpactBytes` itself. The `indexHasPos`-gated pos/pay sub-fields
/// `read_level1_entry` supports are never written: `index_has_pos` is always
/// false on this path, since a term reaching `docFreq >= LEVEL1_NUM_DOCS`
/// implies `docFreq >= BLOCK_SIZE`, which [`validate_field`]'s
/// [`Error::DocFreqTooLargeForPositions`] check already rejects whenever
/// positions are indexed — the same reasoning [`write_full_block`]'s own doc
/// comment gives for why its header never writes pos/pay skip fields either.
fn write_level1_span(
    out: &mut Vec<u8>,
    span: &[(i32, i32)],
    prev_doc_id: i32,
    level1_last_doc_id: &mut i32,
    index_has_freq: bool,
) -> i32 {
    debug_assert_eq!(span.len(), LEVEL1_NUM_DOCS as usize);

    // Build the span's 32 full blocks into a scratch buffer first so the
    // level-1 entry's byte-length field is known before the entry header is
    // written (same "measure by building into scratch first" approach
    // `write_full_block` uses for `blockLength`).
    let mut span_bytes = Vec::new();
    let mut prev = prev_doc_id;
    for block in span.chunks(BLOCK_SIZE as usize) {
        prev = write_full_block(&mut span_bytes, block, prev, index_has_freq);
    }
    let last_doc_id = prev;

    // `read_level1_entry` computes `doc_end_fp` as this vlong's value added
    // to `r.position()` measured right after the vlong itself -- i.e.
    // *before* the freq-gated `skip1EndFP`/`numImpactBytes` fields below are
    // read. So the vlong must span every byte from there through the end of
    // the whole entry+span, not just `span_bytes` alone: the freq-gated
    // header contributes `2 (skip1EndFP) + 2 (numImpactBytes) + 0 (impact
    // bytes, always empty)` extra bytes whenever `index_has_freq`.
    let freq_header_len: usize = if index_has_freq { 4 } else { 0 };
    let doc_delta = last_doc_id - *level1_last_doc_id;
    out.write_vint(doc_delta);
    out.write_vlong((freq_header_len + span_bytes.len()) as i64);
    if index_has_freq {
        out.write_i16(2); // skip1EndFP delta: exactly `numImpactBytes`'s 2 bytes, since
                          // no impact bytes and no pos/pay sub-fields follow (see doc comment).
        out.write_i16(0); // numImpactBytes: always an empty impacts region.
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
        let delta = (doc_id - prev) as u32;
        prev = doc_id;
        if index_has_freq {
            raw.push((delta << 1) | if freq == 1 { 1 } else { 0 });
        } else {
            raw.push(delta);
        }
    }
    write_group_vints(out, &raw);
    if index_has_freq {
        for &(_, freq) in docs {
            if freq != 1 {
                out.write_vint(freq);
            }
        }
    }
}

/// Writes one term's whole `.pos` (and, when `has_offsets`, `.pay`) byte
/// range: zero or more full 256-position `PForUtil` blocks
/// ([`write_full_position_block`]/[`write_full_offset_block`]) followed by a
/// group-varint-free vint tail for the `total_term_freq % BLOCK_SIZE`
/// remainder — the exact write-side inverse of `crate::postings::
/// read_positions`'s `num_full_blocks`/`tail_count` split (`has_payloads ==
/// false` throughout: this writer never has payloads, see the module doc).
/// `positions` is one `Vec<i32>` per doc (parallel to that term's `docs`),
/// each holding the doc's absolute, ascending occurrence positions —
/// see [`TermPostings`]'s `positions` field doc comment for the exact input
/// shape. `offsets` is only consulted when `has_offsets`, in which case it
/// must be the same shape (one `Vec<(i32, i32)>` per doc, `offsets[i].len()
/// == positions[i].len()`) — see [`TermPostings`]'s `offsets` field doc
/// comment.
///
/// Position deltas (and, when `has_offsets`, offset start-deltas/lengths)
/// are buffered into one flat, cross-doc sequence first (resetting to each
/// doc's absolute first position/offset at that doc's first occurrence,
/// exactly like `read_positions`'s own flat `pos_deltas`/`offset_*` before
/// it re-chops the sequence by `freqs`) so that a 256-occurrence chunk
/// spanning a doc boundary is still encoded as a single full block — matching
/// real Lucene's own `addPosition`/`posBufferUpto == BLOCK_SIZE` flush
/// timing, which is entirely doc-boundary-agnostic
/// (`Lucene104PostingsWriter.java:315-355`).
fn write_position_tail(
    pos_out: &mut Vec<u8>,
    pay_out: &mut Vec<u8>,
    positions: &[Vec<i32>],
    offsets: &[Vec<(i32, i32)>],
    has_offsets: bool,
) {
    let mut deltas = Vec::new();
    let mut offset_start_deltas = Vec::new();
    let mut offset_lengths = Vec::new();
    for (doc_idx, doc_positions) in positions.iter().enumerate() {
        let mut prev = 0i32;
        let mut prev_start_offset = 0i32;
        for (occ_idx, &p) in doc_positions.iter().enumerate() {
            deltas.push(p - prev);
            prev = p;
            if has_offsets {
                let (start_offset, end_offset) = offsets[doc_idx][occ_idx];
                offset_start_deltas.push(start_offset - prev_start_offset);
                offset_lengths.push(end_offset - start_offset);
                prev_start_offset = start_offset;
            }
        }
    }

    let mut start = 0usize;
    while deltas.len() - start >= BLOCK_SIZE as usize {
        let end = start + BLOCK_SIZE as usize;
        write_full_position_block(pos_out, &deltas[start..end]);
        if has_offsets {
            write_full_offset_block(
                pay_out,
                &offset_start_deltas[start..end],
                &offset_lengths[start..end],
            );
        }
        start = end;
    }

    // Vint tail (`refillLastPositionBlock`'s write-side inverse,
    // `Lucene104PostingsWriter.finishTerm`, `writePayloads == false` branch
    // throughout): a plain vint position delta per occurrence, then, only
    // when `has_offsets`, an offset start-delta/length pair whose length is
    // only re-written when it changes from the previous occurrence's
    // (`Lucene104PostingsWriter.java:622-632`).
    let mut last_offset_length = -1i32; // force the first occurrence's length to be written
    for i in start..deltas.len() {
        pos_out.write_vint(deltas[i]);
        if has_offsets {
            let delta = offset_start_deltas[i];
            let length = offset_lengths[i];
            if length != last_offset_length {
                pos_out.write_vint((delta << 1) | 1);
                pos_out.write_vint(length);
                last_offset_length = length;
            } else {
                pos_out.write_vint(delta << 1);
            }
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

/// Writes every term's per-term postings metadata bytes — the write-side
/// inverse of `crate::postings::decode_term_metadata` (restricted to this
/// writer's own scope: no payloads, so `payStartFP` only ever appears when
/// the field indexes offsets; `lastPosBlockOffset` is written -- always `0`,
/// since this reader never re-derives or acts on the value, see below --
/// exactly when `decode_term_metadata`'s own `total_term_freq > BLOCK_SIZE`
/// gate requires it). Always takes the bit-clear ("absolute-ish
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
    index_has_positions: bool,
    index_has_offsets: bool,
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

            if index_has_offsets {
                let this_pay_fp = pay_start_fp[i];
                let pay_delta = this_pay_fp.wrapping_sub(base_pay_start_fp);
                out.write_vlong(pay_delta as i64);
                base_pay_start_fp = this_pay_fp;
            }

            // `lastPosBlockOffset`: only present on the wire when
            // `total_term_freq > BLOCK_SIZE` (`decode_term_metadata`'s
            // `total_term_freq > BLOCK_SIZE` gate, strictly greater --
            // exactly `BLOCK_SIZE` needs no tail block at all, so the real
            // writer only ever emits this field once there's a genuine tail
            // to point at). This reader never re-derives or uses the value
            // (`read_positions`'s doc comment: it computes the full-block
            // count from `total_term_freq` itself instead), so any valid
            // vlong round-trips correctly -- same "the reader parses it but
            // never acts on it" shape as [`write_full_block`]'s
            // `level0NumBytes` skip pointer.
            let total_term_freq: i64 = t.docs.iter().map(|&(_, f)| f as i64).sum();
            if total_term_freq > BLOCK_SIZE as i64 {
                out.write_vlong(0);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::blocktree::{self, FieldTerms};
    use crate::field_infos::{
        DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos, VectorEncoding,
        VectorSimilarityFunction,
    };
    use crate::postings::DocInput;

    const SEG_ID: [u8; ID_LENGTH] = [9u8; ID_LENGTH];
    const SUFFIX: &str = "";

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
        let term = varied_docs_term(b"a", doc_freq); // IndexOptions::Docs below -> freq ignored
        let max_doc = term.docs.last().unwrap().0 + 1;
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::Docs,
            doc_count: term.docs.len() as i32,
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
        // vlong + `docDelta`/`blockLength` fields) with bytes whose
        // continuation bits never terminate within the block -- any decode
        // attempt of this header must error out.
        for b in output.doc[span_start..span_start + 8].iter_mut() {
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
            index_options: IndexOptions::DocsAndCustomFreqs,
            doc_count: 1,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::UnsupportedIndexOptions(
                IndexOptions::DocsAndCustomFreqs
            ))
        ));
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

    /// Forces the multi-block/multi-child-trie path this task added: 26
    /// terms, one per lowercase letter (`"a0".."z0"`), so every term is its
    /// own leading-byte group -- 26 physical `.tim` blocks under one
    /// `SIGN_MULTI_CHILDREN` `.tip` root, well above the "does it even split"
    /// bar of 2 blocks. Every term is looked up independently (not just
    /// first/last) through the existing, unmodified `blocktree::open`/
    /// `postings::DocInput`, proving `group_terms_by_leading_byte`/
    /// `write_multi_children_root`'s child ordering, per-block suffix
    /// stripping, and per-block metadata-delta reset (each block restarts
    /// `doc_start_fp`/`pos_start_fp` threading from zero -- see
    /// `write_tim_block`'s doc comment) are all correct, not just the "it
    /// happens to work for the first block" case. See
    /// `crates/lucene-search/tests/postings_writer_round_trip.rs`'s
    /// `term_query_finds_correct_docs_across_multiple_tim_blocks` for the
    /// required real `search_term_query` end-to-end proof of the same
    /// property.
    #[test]
    fn many_leading_byte_groups_force_multi_child_trie_root() {
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

    /// A field with 40 distinct leading bytes exceeds this writer's
    /// multi-child root capacity (2..=33 children, see the module doc) and
    /// must fail loudly rather than silently misencode the 5-bit
    /// `strategyBytes` header field.
    #[test]
    fn rejects_field_needing_more_than_33_leading_byte_groups() {
        let mut terms = Vec::new();
        for i in 0..40u8 {
            terms.push(TermPostings {
                term: vec![i, b'x'],
                docs: vec![(i as i32, 1)],
                ..Default::default()
            });
        }
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 40,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::TooManyLeadingByteGroups(40))
        ));
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

    #[test]
    fn exactly_33_leading_byte_groups_succeeds() {
        let terms = leading_byte_group_terms(33);
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 33,
            terms: &terms,
        };
        let output =
            write_single_field(&input, &SEG_ID, SUFFIX).expect("exactly 33 groups must succeed");
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
            33,
        )
        .expect("33-group output must open cleanly");
        let f = fields.field("f").unwrap();
        assert_eq!(f.num_terms, 33);
    }

    #[test]
    fn exactly_34_leading_byte_groups_is_rejected() {
        let terms = leading_byte_group_terms(34);
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqs,
            doc_count: 34,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::TooManyLeadingByteGroups(34))
        ));
    }

    /// A field with an empty-byte-string term falls back to the single-block
    /// path even when the remaining terms would otherwise split into several
    /// leading-byte groups -- there's no leading byte to strip/route on for
    /// the empty term, so `group_terms_by_leading_byte` must not attempt to
    /// split at all in that case.
    #[test]
    fn empty_term_falls_back_to_single_block_even_with_other_distinct_leading_bytes() {
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
                term: b"alpha".to_vec(),
                docs: vec![(0, 2), (3, 1)],
                positions: vec![vec![1, 4], vec![2]],
                offsets: Vec::new(),
            },
            TermPostings {
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
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![], // no positions supplied, but index_options needs them
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
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
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![1]], // only 1 position but freq == 2
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
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
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![3, 3]], // duplicate, not strictly ascending
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
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
                term: b"fox".to_vec(),
                docs: vec![(0, 1), (2, 1)],
                positions: vec![vec![3], vec![0]],
                offsets: Vec::new(),
            },
            TermPostings {
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
                terms: &title_terms,
            },
            FieldPostingsInput {
                field_number: 1,
                index_options: IndexOptions::DocsAndFreqsAndPositions,
                doc_count: 3,
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
                terms: &a_terms,
            },
            FieldPostingsInput {
                field_number: 1,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 1,
                terms: &b_terms,
            },
            FieldPostingsInput {
                field_number: 2,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 1,
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
            term: b"a".to_vec(),
            docs: vec![(0, BLOCK_SIZE)],
            positions,
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
            terms: &terms,
        };
        write_single_field(&input, &SEG_ID, SUFFIX)
            .expect("docFreq == 1 stays well under BLOCK_SIZE; only docFreq is now bounded");
    }

    /// `docFreq >= BLOCK_SIZE` while indexing positions is still rejected --
    /// this is the ceiling [`Error::DocFreqTooLargeForPositions`] actually
    /// protects (the `.doc`-side full-block path's missing pos/pay skip
    /// fields), replacing the old `total_term_freq`-based check.
    #[test]
    fn rejects_doc_freq_at_or_above_block_size_when_indexing_positions() {
        let doc_freq = BLOCK_SIZE;
        let terms = vec![TermPostings {
            term: b"a".to_vec(),
            docs: (0..doc_freq).map(|i| (i, 1)).collect(),
            positions: (0..doc_freq).map(|i| vec![i]).collect(),
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: doc_freq,
            terms: &terms,
        };
        assert!(matches!(
            write_single_field(&input, &SEG_ID, SUFFIX),
            Err(Error::DocFreqTooLargeForPositions { index: 0, doc_freq })
            if doc_freq == BLOCK_SIZE as i64
        ));
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
                terms: &full_terms,
            },
            FieldPostingsInput {
                field_number: 1,
                index_options: IndexOptions::DocsAndFreqs,
                doc_count: 2,
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
    #[test]
    fn total_term_freq_one_full_position_block_plus_tail_round_trips() {
        let term = irregular_positions_term(b"a", BLOCK_SIZE + 1, 7);
        let max_doc = term.docs.last().unwrap().0 + 1;
        let doc_count = term.docs.len() as i32;
        let expected_positions = term.positions.clone();
        let terms = vec![term.clone()];
        let input = FieldPostingsInput {
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
            term: b"a".to_vec(),
            docs: vec![(0, BLOCK_SIZE)],
            positions: vec![doc_positions.clone()],
            offsets: Vec::new(),
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositions,
            doc_count: 1,
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
            term: b"a".to_vec(),
            docs: vec![(0, 1), (2, 1), (5, 1)],
            positions: vec![vec![0], vec![3], vec![7]],
            offsets: vec![vec![(0, 4)], vec![(30, 33)], vec![(70, 77)]],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 3,
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
            term: b"a".to_vec(),
            docs: vec![(0, 3), (4, 2)],
            positions: vec![vec![1, 4, 9], vec![0, 2]],
            offsets: vec![vec![(5, 9), (20, 24), (45, 50)], vec![(0, 3), (10, 15)]],
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 2,
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
            term: b"a".to_vec(),
            docs: vec![(0, 1)],
            positions: vec![vec![0]],
            offsets: vec![], // no offsets supplied, but index_options needs them
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
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
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![0, 1]],
            offsets: vec![vec![(0, 1)]], // only 1 offset pair but freq == 2
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
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
            term: b"a".to_vec(),
            docs: vec![(0, 2)],
            positions: vec![vec![0, 1]],
            offsets: vec![vec![(5, 8), (3, 6)]], // startOffset decreases
        }];
        let input = FieldPostingsInput {
            field_number: 0,
            index_options: IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            doc_count: 1,
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
}
