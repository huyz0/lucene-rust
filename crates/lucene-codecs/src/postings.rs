//! Port of `org.apache.lucene.codecs.lucene104.Lucene104PostingsReader`'s
//! `.doc`/`.pos`/`.pay` file decode — read-only, scoped to
//! **`IndexOptions.DOCS`/`DOCS_AND_FREQS`/`DOCS_AND_FREQS_AND_POSITIONS`/
//! `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`** (incl. payloads) at any
//! `docFreq`, including `docFreq >= LEVEL1_NUM_DOCS` (32 * `BLOCK_SIZE` =
//! 8192), whose interleaved level-1 skip entries both paths now handle (see
//! "`docFreq >= LEVEL1_NUM_DOCS`" below). Two decode strategies are
//! available: a full forward scan (a
//! sequential `nextDoc()`/`nextPosition()`-equivalent, or the whole-term
//! eager [`DocInput::read_postings`]) and a genuinely lazy decode-on-demand
//! `advance()` ([`LazyDocsCursor`]) — see "`advance()`: two APIs, two decode
//! strategies" below for which to use when. See "Positions/offsets/payloads
//! (`.pos`/`.pay`)" below for that half of the decode; everything above it in
//! this doc comment covers `.doc` alone, same as before that was added.
//!
//! ## Three shapes of a term's `.doc` bytes
//!
//! `Lucene104PostingsWriter.finishTerm` special-cases `docFreq == 1` by pulsing
//! the single doc ID into the term dictionary itself (see
//! `Lucene104PostingsWriter.java:568-577`): no bytes are written to `.doc` at
//! all for a singleton term ([`singleton_postings`]). For `1 < docFreq <
//! BLOCK_SIZE`, `flushDocBlock(true)` never reaches the packed-int/bit-set
//! branch (that path only runs when `docBufferUpto == BLOCK_SIZE`,
//! `Lucene104PostingsWriter.java:392-461`) — instead it takes the
//! `PostingsUtil.writeVIntBlock` branch (`Lucene104PostingsWriter.java:394-395`),
//! a much simpler group-varint + trailing-vint-freq-exceptions encoding with no
//! skip data, no impacts, and no `ForUtil`/`PForUtil` bit-packing at all
//! ([`read_tail_block`]). For `docFreq >= BLOCK_SIZE`, one or more full
//! 256-doc blocks precede that same tail-block encoding for the
//! `docFreq % BLOCK_SIZE` remainder (zero full blocks' worth of tail bytes if
//! `docFreq` is an exact multiple of `BLOCK_SIZE` — the last full block is
//! still written via the full-block path in that case, see
//! `Lucene104PostingsWriter.finishTerm`/`flushDocBlock`). Each full block is
//! `ForUtil`/`PForUtil`-encoded ([`read_full_block`], ported in
//! [`crate::for_util`]) and prefixed by a level-0 skip header
//! (`Lucene104PostingsWriter.flushDocBlock`'s `else` branch) that this reader
//! parses field-by-field in wire order rather than exploiting its skip
//! pointers.
//!
//! ## Wire format: the tail block (`docFreq % BLOCK_SIZE`, or the whole term
//! for `docFreq < BLOCK_SIZE`)
//!
//! `docFreq % BLOCK_SIZE` (or `docFreq`, if `< BLOCK_SIZE`) group-varint-encoded
//! values (`GroupVIntUtil`/`DataInput::read_group_vints`, already ported in
//! `lucene-store`), each packing `(docDelta << 1) | (freq == 1 ? 1 : 0)` when
//! the field has freqs (`PostingsUtil.java:39-52`), or plain `docDelta` when it
//! doesn't (`IndexOptions::Docs`). Immediately after, in doc order, one plain
//! vint per doc whose packed bit was 0 (i.e. freq != 1) carries that doc's
//! actual freq. Doc IDs are delta-coded from a base of the previous block's
//! last doc ID, or `-1` if there is no previous block
//! (`Lucene104PostingsReader.prefixSum`, `Lucene104PostingsReader.java:194-200`).
//!
//! ## Wire format: a full 256-doc block
//!
//! Per full block, in order (`BlockPostingsEnum.refillFullBlock` plus the
//! level-0 header that precedes it): `level0NumBytes` (vlong, skip-pointer —
//! parsed but unused by this reader, see [`LazyDocsCursor`]'s doc comment for
//! why), `docDelta` (`writeVInt15`-encoded — this block's last doc ID minus
//! the previous block's, used by [`LazyDocsCursor`]/[`read_full_block_header`]
//! to decide whether to skip the block without decoding it, discarded by the
//! eager [`read_full_block`]), `blockLength` (`writeVLong15`-encoded — the
//! byte length, from right after this field, of everything through the end
//! of the block, i.e. impacts/pos-pay fields plus the body; used the same
//! way as `docDelta` to compute where the block ends without decoding it);
//! then, only when the field has freqs, an impacts byte-length (vlong) and
//! that many impact bytes (competitive-scoring metadata — parsed-and-discarded,
//! see "Deferred"); then a 1-byte `bitsPerValue` token selecting how the
//! block's 256 doc deltas are packed
//! (`> 0`: `ForUtil`-bit-packed body, `numBytes(bitsPerValue)` bytes; `== 0`:
//! no bytes, every delta is 1 — "all 256 docs in the block are consecutive";
//! `< 0`: a `-bitsPerValue`-long bit-set encoding — the 256 doc IDs are the
//! ascending set-bit positions, based at the previous block's last doc + 1);
//! then, only when the field has freqs, a `PForUtil`-encoded
//! (patched frame-of-reference, i.e. bit-packed plus up to 7 byte-patched
//! exceptions) block of 256 raw freq values.
//!
//! ## Per-term metadata (`decodeTerm`)
//!
//! The blocktree term dictionary's per-term metadata bytes (previously skipped
//! by `blocktree.rs`, see its module doc) encode `Lucene104PostingsReader.decodeTerm`
//! (`Lucene104PostingsReader.java:213-251`), scoped here to the no-positions
//! case: one vlong whose low bit selects between an absolute-ish `docStartFP`
//! delta (bit clear — `termState.docStartFP += l >>> 1`, plus a raw vint
//! `singletonDocID` when `docFreq == 1`) or a zigzag `singletonDocID` delta
//! relative to the *previous term in the same block* (bit set — only legal for
//! a non-absolute decode, i.e. not the first term after a block load; see
//! `SegmentTermsEnumFrame.java:471,506,509`: `absolute = metaDataUpto == 0`).
//!
//! ## Positions/offsets/payloads (`.pos`/`.pay`)
//!
//! For `IndexOptions::DocsAndFreqsAndPositions` and up, `decode_term_metadata`
//! also decodes `posStartFP`/`payStartFP`/`lastPosBlockOffset`
//! (`Lucene104PostingsReader.java:237-250`), and a full `.doc` block's
//! level-0 header carries extra pos/pay skip-pointer fields
//! (`Lucene104PostingsReader.java:754-761`, parsed-and-discarded by
//! [`read_full_block`] same as the rest of that header). The actual
//! position/offset/payload bytes live entirely in `.pos`/`.pay`
//! ([`PosInput`]/[`PayInput`], opened the same way as [`DocInput`]), not
//! `.doc`, as **one flat sequence of `totalTermFreq` occurrences** rather
//! than one block per doc — `Lucene104PostingsWriter.addPosition` buffers and
//! flushes 256 occurrences at a time *across* doc boundaries, only resetting
//! the position/offset delta accumulator to 0 at each doc's first occurrence
//! (`Lucene104PostingsReader.java:1298-1304`, mirroring
//! `Lucene104PostingsWriter.startDoc`'s `lastPosition = 0; lastStartOffset =
//! 0;`). [`read_positions`] decodes that flat sequence — zero or more full
//! `ForUtil`/`PForUtil` blocks of `BLOCK_SIZE` (positions reuse the exact
//! same 256-wide block size as `.doc`, confirmed from
//! `Lucene104PostingsFormat.BLOCK_SIZE = ForUtil.BLOCK_SIZE` rather than
//! assumed from an older Lucene version's separate/smaller position block
//! size) for `totalTermFreq / BLOCK_SIZE` full groups — payload lengths and
//! offset start-deltas/lengths are themselves bulk `PForUtil`-encoded per
//! block, with that block's payload bytes batched into one run right after —
//! then a `refillLastPositionBlock`-style vint tail for the
//! `totalTermFreq % BLOCK_SIZE` remainder, where payload bytes are inlined in
//! `.pos` immediately after each occurrence's length instead, and a
//! payload/offset length is only re-written when it changes from the
//! previous occurrence's (reused otherwise) — then re-chops the flat sequence
//! into per-doc groups using the term's already-decoded `Postings::freqs`.
//!
//! ## `advance()`: two APIs, two decode strategies
//!
//! [`PostingsCursor`] gives `advance(target)`/`next_doc()` **interface**
//! parity with `PostingsEnum` as a binary search over
//! [`DocInput::read_postings`]'s already-fully-decoded `Vec<i32>` — simple,
//! correct, but not lazy: every block is decoded up front regardless of what
//! the caller ends up needing. [`LazyDocsCursor`] (opened via
//! [`DocInput::lazy_cursor`]) is the genuinely lazy sibling: it decodes one
//! `.doc` block at a time, and `advance()` uses each full block's own
//! level-0 header (`docDelta`/`blockLength`) to jump straight past a whole
//! block's body — never running `ForUtil`/`PForUtil` decode on it — whenever
//! the header proves the block's entire doc range is behind the target. See
//! [`LazyDocsCursor`]'s own doc comment for the precise, load-bearing
//! boundary of what that does and does not skip (short version: full blocks
//! are skippable at any `docFreq >= BLOCK_SIZE`, whole 32-block spans are
//! skippable via level-1 entries at `docFreq >= LEVEL1_NUM_DOCS`, and the
//! tail block never is). Pick [`PostingsCursor`] when the term's postings
//! are small
//! enough that eager decode is cheap or a caller already has a
//! fully-materialized [`Postings`] on hand; pick [`LazyDocsCursor`] when a
//! caller wants real skip-past-undecoded-blocks behavior (e.g. a
//! conjunction query intersecting a large postings list against a much
//! smaller one) or wants to stop decoding early without paying for the rest
//! of the term up front.
//!
//! ## `docFreq >= LEVEL1_NUM_DOCS` (level-1 skip entries)
//!
//! Above `LEVEL1_NUM_DOCS` (8192) the `.doc` stream interleaves a level-1
//! skip entry ([`read_level1_entry`]) before every span of [`LEVEL1_FACTOR`]
//! (32) full level-0 blocks, for as long as at least `LEVEL1_NUM_DOCS` docs
//! remain. Both paths handle this now: `read_postings` consumes each entry's
//! bytes and decodes its 32 blocks (materializing everything, no jumping),
//! and [`LazyDocsCursor`] uses the entry's `doc_delta`/`doc_end_fp` to jump
//! straight past a whole 32-block span whose last doc is behind the caller's
//! `advance()` target — the coarser level-1 counterpart to the level-0
//! skip-past-one-block described above. Positions inherit this via the same
//! `docFreq` gate through `postings()`.
//!
//! ## Deferred (all rejected with [`Error::Unsupported`])
//!
//! - `IndexOptions::DocsAndCustomFreqs` — real Lucene never writes this for
//!   an ordinary indexed text field, so it's out of scope here.
//! - Impacts (`ImpactsEnum`, `CompetitiveImpactAccumulator`, competitive-scoring
//!   metadata) — see `docs/parity.md`.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use crate::field_infos::IndexOptions;
use crate::for_util;

/// `Lucene104PostingsFormat.DOC_CODEC`.
const DOC_CODEC: &str = "Lucene104PostingsWriterDoc";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;
/// `ForUtil.BLOCK_SIZE` (== `Lucene104PostingsFormat.BLOCK_SIZE`).
pub const BLOCK_SIZE: i32 = 256;
/// `Lucene104PostingsFormat.LEVEL1_NUM_DOCS` (`LEVEL1_FACTOR(=32) * BLOCK_SIZE`):
/// below this many docs, a term's `.doc` bytes contain only level-0 skip
/// headers (no level-1 entries) — see the module doc's "Deferred" section.
const LEVEL1_NUM_DOCS: i32 = 32 * BLOCK_SIZE;
/// `Lucene104PostingsFormat.LEVEL1_FACTOR`: one level-1 skip entry precedes a
/// span of exactly this many consecutive full level-0 blocks
/// (`32 * BLOCK_SIZE == LEVEL1_NUM_DOCS` docs).
const LEVEL1_FACTOR: usize = 32;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("decodeTerm: singleton-delta bit set on an absolute (first-in-block) decode")]
    AbsoluteSingletonDelta,
    #[error("decodeTerm: singleton-delta bit set but no previous singleton to delta from")]
    NoPreviousSingleton,
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Per-term postings location, decoded from the blocktree's per-term metadata
/// bytes (`Lucene104PostingsReader.decodeTerm`, no-positions subset). `-1` for
/// `singleton_doc_id` means "not a singleton" (`docFreq > 1`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermMetadata {
    pub doc_start_fp: u64,
    pub singleton_doc_id: i32,
    /// Only meaningful for `IndexOptions::DocsAndFreqsAndPositions` and up;
    /// `0` otherwise (never read in that case).
    pub pos_start_fp: u64,
    /// Only meaningful when the field has offsets or payloads; `0` otherwise.
    pub pay_start_fp: u64,
    /// `Lucene104PostingsReader.decodeTerm`'s `lastPosBlockOffset`: `-1` when
    /// `totalTermFreq <= BLOCK_SIZE` (no trailing vint-encoded position
    /// block after the full `ForUtil`/`PForUtil` blocks — either there are no
    /// full blocks at all and everything is the vint tail, or the term ends
    /// exactly on a full-block boundary and there is no tail at all).
    pub last_pos_block_offset: i64,
}

impl Default for TermMetadata {
    fn default() -> Self {
        Self::EMPTY
    }
}

impl TermMetadata {
    /// `IntBlockTermState`'s "empty" starting state (`EMPTY_STATE` /
    /// `absolute == true` semantics): zero `docStartFP`/`posStartFP`/
    /// `payStartFP`, no singleton yet.
    pub const EMPTY: TermMetadata = TermMetadata {
        doc_start_fp: 0,
        singleton_doc_id: -1,
        pos_start_fp: 0,
        pay_start_fp: 0,
        last_pos_block_offset: -1,
    };
}

/// `Lucene104PostingsReader.decodeTerm`, restricted to fields with no
/// positions (`IndexOptions::Docs`/`DocsAndFreqs`) — the `posStartFP`/
/// `payStartFP`/`lastPosBlockOffset` fields never appear on the wire for
/// those. `absolute` mirrors `SegmentTermsEnumFrame`'s `metaDataUpto == 0`:
/// true only for the first term decoded after loading a `.tim` block, false
/// for every subsequent term in that same block (deltas are relative to the
/// previous term's decoded state, `prev`).
///
/// `index_options`/`has_payloads`/`total_term_freq` drive the
/// positions/offsets/payloads-specific fields
/// (`Lucene104PostingsReader.java:237-250`): a `posStartFP` delta vlong when
/// `index_options` indexes positions; then, only when it also indexes
/// offsets or the field stores payloads, a `payStartFP` delta vlong; then,
/// only when `total_term_freq > BLOCK_SIZE`, a `lastPosBlockOffset` vlong
/// locating the final vint-encoded position block (see
/// `read_positions`/`PosPayInput`). `total_term_freq` must be the *this
/// term's* decoded total, not the previous term's — same as `doc_freq`.
pub fn decode_term_metadata(
    r: &mut SliceInput,
    doc_freq: i32,
    absolute: bool,
    prev: TermMetadata,
    index_options: IndexOptions,
    has_payloads: bool,
    total_term_freq: i64,
) -> Result<TermMetadata> {
    // `Lucene104PostingsReader.decodeTerm` zeroes every FP accumulator before
    // applying this term's deltas when `absolute` is set (a fresh term-dict
    // block always starts its first term's FPs from 0), rather than basing
    // them on whatever `prev` happened to carry in from the caller.
    let base = if absolute { TermMetadata::EMPTY } else { prev };

    let l = r.read_vlong()? as u64;
    let (doc_start_fp, singleton_doc_id) = if l & 1 == 0 {
        let doc_start_fp = base.doc_start_fp.wrapping_add(l >> 1);
        let singleton_doc_id = if doc_freq == 1 { r.read_vint()? } else { -1 };
        (doc_start_fp, singleton_doc_id)
    } else {
        if absolute {
            return Err(Error::AbsoluteSingletonDelta);
        }
        if prev.singleton_doc_id == -1 {
            return Err(Error::NoPreviousSingleton);
        }
        let delta = lucene_util::zigzag::decode(l >> 1);
        (
            prev.doc_start_fp,
            (prev.singleton_doc_id as i64 + delta) as i32,
        )
    };

    let mut pos_start_fp = base.pos_start_fp;
    let mut pay_start_fp = base.pay_start_fp;
    let mut last_pos_block_offset: i64 = -1;
    if index_options.subsumes_positions() {
        pos_start_fp = pos_start_fp.wrapping_add(r.read_vlong()? as u64);
        if index_options.subsumes_offsets() || has_payloads {
            pay_start_fp = pay_start_fp.wrapping_add(r.read_vlong()? as u64);
        }
        if total_term_freq > BLOCK_SIZE as i64 {
            last_pos_block_offset = r.read_vlong()?;
        }
    }

    Ok(TermMetadata {
        doc_start_fp,
        singleton_doc_id,
        pos_start_fp,
        pay_start_fp,
        last_pos_block_offset,
    })
}

/// One term's decoded `(docID, freq)` pairs, in ascending doc-ID order.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Postings {
    pub docs: Vec<i32>,
    pub freqs: Vec<i32>,
}

/// An opened `.doc` file (header/footer validated once), ready for
/// per-term seeks. Mirrors `Lucene104PostingsReader`'s `docIn`, minus
/// everything this slice doesn't support (positions, skip data, impacts).
pub struct DocInput<'a> {
    buf: &'a [u8],
}

impl<'a> DocInput<'a> {
    /// Validates the `.doc` file's index header and footer checksum framing
    /// (`Lucene104PostingsReader`'s constructor, `Lucene104PostingsReader.java:134-140`).
    pub fn open(doc: &'a [u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<Self> {
        let mut r = SliceInput::new(doc);
        codec_util::check_index_header(
            &mut r,
            DOC_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        codec_util::retrieve_checksum(doc)?;
        Ok(DocInput { buf: doc })
    }

    /// Decodes a term's `(docID, freq)` pairs for any `docFreq > 1`
    /// (`docFreq == 1` singletons are pulsed into the term dictionary, see
    /// [`singleton_postings`]). Dispatches on `doc_freq` the same way
    /// `BlockPostingsEnum.refillDocs` does: zero or more full 256-doc PFOR
    /// blocks (`refillFullBlock`, [`read_full_block`]) followed by at most
    /// one group-varint tail block for the `docFreq % BLOCK_SIZE` remainder
    /// (`refillRemainder`'s non-singleton branch,
    /// `Lucene104PostingsReader.java:647-656`, [`read_tail_block`]).
    ///
    /// `index_options`/`has_payloads` may indicate a field with
    /// positions/offsets/payloads (`IndexOptions::DocsAndFreqsAndPositions`
    /// and up): the `.doc` file's full-block level-0 header carries extra
    /// pos/pay skip fields in that case (`Lucene104PostingsReader.java:754-761`),
    /// parsed here for wire-order correctness even though this reader never
    /// decodes `.pos`/`.pay` itself (see [`read_positions`] for that).
    pub fn read_postings(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
        has_payloads: bool,
    ) -> Result<Postings> {
        if doc_freq <= 1 {
            return Err(Error::Unsupported(
                "docFreq <= 1: use singleton_postings instead (no .doc bytes are written)",
            ));
        }
        if !matches!(
            index_options,
            IndexOptions::Docs
                | IndexOptions::DocsAndFreqs
                | IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        ) {
            return Err(Error::Unsupported(
                "IndexOptions::DocsAndCustomFreqs is not supported in this slice",
            ));
        }
        let index_has_freq = index_options != IndexOptions::Docs;
        let index_has_pos = index_options.subsumes_positions();
        let index_has_offsets_or_payloads = index_options.subsumes_offsets() || has_payloads;

        let mut r = SliceInput::new(self.buf);
        r.seek(meta.doc_start_fp as usize)?;

        let n = doc_freq as usize;
        let mut docs = Vec::with_capacity(n);
        let mut freqs = Vec::with_capacity(n);

        let mut prev_doc_id: i32 = -1;
        let mut doc_count_left = doc_freq;

        // `docFreq >= LEVEL1_NUM_DOCS` (8192): the `.doc` stream interleaves a
        // level-1 skip entry before every span of `LEVEL1_FACTOR` (32) full
        // level-0 blocks, for as long as at least `LEVEL1_NUM_DOCS` docs
        // remain (`Lucene104PostingsReader.skipLevel1To`'s
        // `docCountLeft < LEVEL1_NUM_DOCS` stop condition). This eager path
        // materializes every doc regardless, so it consumes each level-1
        // entry's bytes (via [`read_level1_entry`], shared with the lazy
        // cursor) purely to stay aligned, then decodes the 32 blocks that
        // follow. Once fewer than `LEVEL1_NUM_DOCS` docs remain, no more
        // level-1 entries appear and the remaining full blocks + tail decode
        // exactly like a sub-8192 term.
        while doc_count_left >= LEVEL1_NUM_DOCS {
            read_level1_entry(
                &mut r,
                index_has_freq,
                index_has_pos,
                index_has_offsets_or_payloads,
            )?;
            for _ in 0..LEVEL1_FACTOR {
                let (block_docs, block_freqs) = read_full_block(
                    &mut r,
                    prev_doc_id,
                    index_has_freq,
                    index_has_pos,
                    index_has_offsets_or_payloads,
                )?;
                prev_doc_id = block_docs[block_docs.len() - 1];
                docs.extend_from_slice(&block_docs);
                freqs.extend_from_slice(&block_freqs);
                doc_count_left -= BLOCK_SIZE;
            }
        }

        while doc_count_left >= BLOCK_SIZE {
            let (block_docs, block_freqs) = read_full_block(
                &mut r,
                prev_doc_id,
                index_has_freq,
                index_has_pos,
                index_has_offsets_or_payloads,
            )?;
            prev_doc_id = block_docs[block_docs.len() - 1];
            docs.extend_from_slice(&block_docs);
            freqs.extend_from_slice(&block_freqs);
            doc_count_left -= BLOCK_SIZE;
        }
        if doc_count_left > 0 {
            read_tail_block(
                &mut r,
                prev_doc_id,
                doc_count_left as usize,
                index_has_freq,
                &mut docs,
                &mut freqs,
            )?;
        }

        Ok(Postings { docs, freqs })
    }

    /// Opens a [`LazyDocsCursor`] over this term's `(docID, freq)` pairs:
    /// blocks are decoded on demand, and a full block whose entire doc range
    /// is behind the caller's `advance()` target is skipped without ever
    /// running `ForUtil`/`PForUtil` decode on it (see [`LazyDocsCursor`]'s
    /// own doc comment for exactly what "skipped" means here). Validation
    /// (`doc_freq <= 1`, `IndexOptions::DocsAndCustomFreqs`) mirrors
    /// [`Self::read_postings`] exactly — same scope, different decode
    /// strategy. `docFreq >= LEVEL1_NUM_DOCS` is supported by both: this
    /// cursor additionally jumps whole 32-block level-1 spans (see
    /// [`Self::skip_level1_to`]).
    pub fn lazy_cursor(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
        has_payloads: bool,
    ) -> Result<LazyDocsCursor<'a>> {
        if doc_freq <= 1 {
            return Err(Error::Unsupported(
                "docFreq <= 1: use singleton_postings instead (no .doc bytes are written)",
            ));
        }
        if !matches!(
            index_options,
            IndexOptions::Docs
                | IndexOptions::DocsAndFreqs
                | IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        ) {
            return Err(Error::Unsupported(
                "IndexOptions::DocsAndCustomFreqs is not supported in this slice",
            ));
        }
        let mut r = SliceInput::new(self.buf);
        r.seek(meta.doc_start_fp as usize)?;

        // Mirror `Lucene104PostingsReader.BlockPostingsEnum.reset`'s level-1
        // setup (`Lucene104PostingsReader.java:559-568`): below
        // `LEVEL1_NUM_DOCS` there are no level-1 entries on the wire, so pin
        // `level1_last_doc_id` at NO_MORE_DOCS to disable the level-1 skip
        // path entirely (`target > NO_MORE_DOCS` is never true). At or above
        // it, start the running last-doc at `-1` with `level1_doc_end_fp`
        // pointing at the first level-1 entry (which sits at `docStartFP`).
        let level1_last_doc_id = if doc_freq < LEVEL1_NUM_DOCS {
            NO_MORE_DOCS
        } else {
            -1
        };

        Ok(LazyDocsCursor {
            r,
            index_has_freq: index_options != IndexOptions::Docs,
            index_has_pos: index_options.subsumes_positions(),
            index_has_offsets_or_payloads: index_options.subsumes_offsets() || has_payloads,
            doc_freq,
            prev_doc_id: -1,
            doc_count_left: doc_freq,
            level1_last_doc_id,
            level1_doc_end_fp: meta.doc_start_fp as usize,
            level1_doc_count_upto: 0,
            block_docs: [0; BLOCK_SIZE as usize],
            block_freqs: [0; BLOCK_SIZE as usize],
            block_len: 0,
            block_pos: 0,
            doc_id: -1,
        })
    }
}

/// `Lucene104PostingsFormat.POS_CODEC`.
const POS_CODEC: &str = "Lucene104PostingsWriterPos";
/// `Lucene104PostingsFormat.PAY_CODEC`.
const PAY_CODEC: &str = "Lucene104PostingsWriterPay";

/// One decoded position occurrence — `PostingsEnum.nextPosition()` bundled
/// with `startOffset()`/`endOffset()`/`getPayload()` for a single occurrence
/// of a term in one doc.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Position {
    pub position: i32,
    /// `-1` when the field doesn't index offsets (`PostingsEnum.startOffset`'s
    /// own no-offsets contract).
    pub start_offset: i32,
    /// `-1` when the field doesn't index offsets.
    pub end_offset: i32,
    /// Empty when this occurrence has no payload, or the field doesn't store
    /// payloads at all (`PostingsEnum.getPayload() == null`).
    pub payload: Vec<u8>,
}

/// An opened `.pos` file (header/footer validated once), analogous to
/// [`DocInput`].
pub struct PosInput<'a> {
    buf: &'a [u8],
}

impl<'a> PosInput<'a> {
    /// `Lucene104PostingsReader`'s constructor, the `.pos` branch
    /// (`Lucene104PostingsReader.java:142-149`).
    pub fn open(pos: &'a [u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<Self> {
        let mut r = SliceInput::new(pos);
        codec_util::check_index_header(
            &mut r,
            POS_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        codec_util::retrieve_checksum(pos)?;
        Ok(PosInput { buf: pos })
    }
}

/// An opened `.pay` file (header/footer validated once), analogous to
/// [`DocInput`]. Only opened for fields with offsets and/or payloads
/// (`Lucene104PostingsReader.java:151-161`).
pub struct PayInput<'a> {
    buf: &'a [u8],
}

impl<'a> PayInput<'a> {
    pub fn open(pay: &'a [u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<Self> {
        let mut r = SliceInput::new(pay);
        codec_util::check_index_header(
            &mut r,
            PAY_CODEC,
            VERSION_START,
            VERSION_CURRENT,
            segment_id,
            segment_suffix,
        )?;
        codec_util::retrieve_checksum(pay)?;
        Ok(PayInput { buf: pay })
    }
}

/// Decodes every position (and, if the field has them, offset/payload)
/// occurrence for a term, in doc order — `PostingsEnum.nextPosition()`/
/// `startOffset()`/`endOffset()`/`getPayload()` for every doc this term
/// occurs in — given that term's already-decoded per-doc frequencies
/// (`Postings::freqs`, in the same doc order [`DocInput::read_postings`] or
/// [`singleton_postings`] produced) and per-term metadata.
///
/// Scoped like [`DocInput::read_postings`]: **sequential decode only** (no
/// skip-ahead), any `total_term_freq` this port's fixtures or a realistic
/// term would produce. Positions/payloads/offsets live in wholly separate
/// `.pos`/`.pay` files from `.doc`, as **one flat sequence of
/// `total_term_freq` occurrences**, not one block per doc — the writer
/// buffers/flushes 256 occurrences at a time across doc boundaries
/// (`Lucene104PostingsWriter.addPosition`'s `posBufferUpto == BLOCK_SIZE`
/// flush), only resetting the position/offset accumulator to 0 at each
/// doc's *first* occurrence (`Lucene104PostingsReader.java:1298-1304`,
/// mirroring `Lucene104PostingsWriter.startDoc`'s `lastPosition = 0;
/// lastStartOffset = 0;`). This decodes that whole flat sequence first
/// (full `ForUtil`/`PForUtil` blocks of `BLOCK_SIZE`, i.e. `for_util::
/// BLOCK_SIZE` == the same 256 as `.doc`'s block size — confirmed from
/// `Lucene104PostingsFormat.BLOCK_SIZE = ForUtil.BLOCK_SIZE`, not a
/// separate/older 128-wide position block size — then a `refillLastPositionBlock`-style
/// vint tail for the `total_term_freq % BLOCK_SIZE` remainder), then
/// re-chops it into per-doc groups using `freqs`.
pub fn read_positions(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    freqs: &[i32],
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
) -> Result<Vec<Vec<Position>>> {
    if !index_options.subsumes_positions() {
        return Err(Error::Unsupported(
            "read_positions needs a field with IndexOptions::DocsAndFreqsAndPositions or higher",
        ));
    }
    let has_offsets = index_options.subsumes_offsets();

    let mut pos_r = SliceInput::new(pos.buf);
    pos_r.seek(meta.pos_start_fp as usize)?;
    let mut pay_r = pay.map(|p| SliceInput::new(p.buf));
    if let Some(r) = pay_r.as_mut() {
        r.seek(meta.pay_start_fp as usize)?;
    }

    let n = total_term_freq as usize;
    let mut pos_deltas: Vec<i32> = Vec::with_capacity(n);
    let mut payload_lengths: Vec<i32> = Vec::with_capacity(if has_payloads { n } else { 0 });
    let mut payload_bytes: Vec<u8> = Vec::new();
    let mut offset_start_deltas: Vec<i32> = Vec::with_capacity(if has_offsets { n } else { 0 });
    let mut offset_lengths: Vec<i32> = Vec::with_capacity(if has_offsets { n } else { 0 });

    // `meta.last_pos_block_offset` (already decoded by `decode_term_metadata`)
    // tells us exactly where the vint tail block begins on the wire, which is
    // equivalent to (but doesn't require us to re-derive live, unlike the
    // real reader's `posIn.getFilePointer() == lastPosBlockFP` check) simply
    // computing how many full 256-position blocks precede it from
    // `total_term_freq` itself.
    let num_full_blocks = n / BLOCK_SIZE as usize;
    let tail_count = n % BLOCK_SIZE as usize;

    // `.pay` is only ever touched by full PForUtil blocks (the vint tail's
    // payload bytes live inline in `.pos`, see below) -- so a term whose
    // whole `total_term_freq` fits in the tail never needs it, even for a
    // field with offsets/payloads.
    if num_full_blocks > 0 && (has_offsets || has_payloads) && pay.is_none() {
        return Err(Error::Unsupported(
            "read_positions needs an opened .pay file: this field has offsets or payloads and \
             total_term_freq spans at least one full 256-position block",
        ));
    }

    for _ in 0..num_full_blocks {
        let mut deltas = [0u32; for_util::BLOCK_SIZE];
        for_util::pfor_decode(&mut pos_r, &mut deltas)?;
        pos_deltas.extend(deltas.iter().map(|&d| d as i32));

        if has_payloads {
            let pay_r = pay_r
                .as_mut()
                .expect("checked above: has_payloads implies pay.is_some()");
            let mut lens = [0u32; for_util::BLOCK_SIZE];
            for_util::pfor_decode(pay_r, &mut lens)?;
            let num_bytes = pay_r.read_vint()? as usize;
            let start = payload_bytes.len();
            payload_bytes.resize(start + num_bytes, 0);
            pay_r.read_bytes(&mut payload_bytes[start..])?;
            payload_lengths.extend(lens.iter().map(|&l| l as i32));
        }
        if has_offsets {
            let pay_r = pay_r
                .as_mut()
                .expect("checked above: has_offsets implies pay.is_some()");
            let mut starts = [0u32; for_util::BLOCK_SIZE];
            for_util::pfor_decode(pay_r, &mut starts)?;
            let mut lens = [0u32; for_util::BLOCK_SIZE];
            for_util::pfor_decode(pay_r, &mut lens)?;
            offset_start_deltas.extend(starts.iter().map(|&s| s as i32));
            offset_lengths.extend(lens.iter().map(|&l| l as i32));
        }
    }

    if tail_count > 0 {
        // Vint tail block (`refillLastPositionBlock`,
        // `Lucene104PostingsReader.java:1176-1216`): a real reverse-engineered
        // detail, not guessed by analogy with the full-block path above —
        // payload bytes are inlined in `.pos` immediately after their length,
        // not batched separately in `.pay`, and a payload/offset length is
        // only written when it *changes* from the previous occurrence's
        // (bit 0 of the vint code), reusing the last value otherwise.
        let mut last_payload_length = 0i32;
        let mut last_offset_length = 0i32;
        for _ in 0..tail_count {
            let code = pos_r.read_vint()?;
            if has_payloads {
                if code & 1 != 0 {
                    last_payload_length = pos_r.read_vint()?;
                }
                pos_deltas.push(code >> 1);
                if last_payload_length != 0 {
                    let start = payload_bytes.len();
                    payload_bytes.resize(start + last_payload_length as usize, 0);
                    pos_r.read_bytes(&mut payload_bytes[start..])?;
                }
                payload_lengths.push(last_payload_length);
            } else {
                pos_deltas.push(code);
            }

            if has_offsets {
                let delta_code = pos_r.read_vint()?;
                if delta_code & 1 != 0 {
                    last_offset_length = pos_r.read_vint()?;
                }
                offset_start_deltas.push(delta_code >> 1);
                offset_lengths.push(last_offset_length);
            }
        }
    }

    // Re-chop the flat, `total_term_freq`-long sequence into per-doc groups
    // using `freqs`, resetting the position/offset accumulator to 0 at each
    // doc's first occurrence (deltas are only ever relative to the previous
    // occurrence of the *same* doc, never across a doc boundary — see this
    // function's doc comment).
    let mut payload_upto = 0usize;
    let mut idx = 0usize;
    let mut result = Vec::with_capacity(freqs.len());
    for &freq in freqs {
        let mut position = 0i32;
        let mut start_offset_acc = 0i32;
        let mut doc_positions = Vec::with_capacity(freq.max(0) as usize);
        for _ in 0..freq {
            // `freqs` is decoded independently (from `.doc`) of `n =
            // total_term_freq` (from the term dictionary): nothing on the
            // wire guarantees they agree, so a corrupted `.doc`/`.tim`/`.tmd`
            // could otherwise walk `idx` past the end of the flat
            // `pos_deltas`/`payload_lengths`/`offset_*` arrays and panic on
            // out-of-bounds indexing instead of surfacing a decode error.
            if idx >= pos_deltas.len() {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "sum of per-doc freqs exceeds total_term_freq".into(),
                )));
            }
            position += pos_deltas[idx];
            let payload = if has_payloads {
                let len = payload_lengths[idx] as usize;
                let start = payload_upto;
                let end = start + len;
                if end > payload_bytes.len() {
                    return Err(Error::Store(lucene_store::Error::Corrupted(
                        "payload length exceeds decoded payload bytes".into(),
                    )));
                }
                payload_upto = end;
                payload_bytes[start..end].to_vec()
            } else {
                Vec::new()
            };
            let (start_offset, end_offset) = if has_offsets {
                let s = start_offset_acc + offset_start_deltas[idx];
                let e = s + offset_lengths[idx];
                start_offset_acc = s;
                (s, e)
            } else {
                (-1, -1)
            };
            doc_positions.push(Position {
                position,
                start_offset,
                end_offset,
                payload,
            });
            idx += 1;
        }
        result.push(doc_positions);
    }

    if idx != n {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "sum of per-doc freqs is less than total_term_freq".into(),
        )));
    }

    Ok(result)
}

/// `writeVInt15`'s companion reader
/// (`Lucene104PostingsReader.readVInt15`): a 2-byte fast path for values that
/// fit in 15 bits, else the top bit of the `short` flags a following vint
/// carrying the remaining high bits (`value = (s & 0x7FFF) | (extra << 15)`).
fn read_vint15(r: &mut SliceInput) -> Result<i32> {
    let s = r.read_i16()?;
    if s >= 0 {
        Ok(s as i32)
    } else {
        Ok((s as i32 & 0x7FFF) | (r.read_vint()? << 15))
    }
}

/// `Lucene104PostingsReader.readVLong15`, the `long`-widening sibling of
/// [`read_vint15`].
fn read_vlong15(r: &mut SliceInput) -> Result<i64> {
    let s = r.read_i16()?;
    if s >= 0 {
        Ok(s as i64)
    } else {
        Ok((s as i64 & 0x7FFF) | (r.read_vlong()? << 15))
    }
}

/// One level-1 skip entry, decoded from the `.doc` stream (present before
/// every span of [`LEVEL1_FACTOR`] full level-0 blocks while at least
/// `LEVEL1_NUM_DOCS` docs remain). Mirrors the fields
/// `Lucene104PostingsReader.skipLevel1To` reads
/// (`Lucene104PostingsReader.java:691-713`).
struct Level1Entry {
    /// vInt delta added to the running `level1LastDocID` (starts at `-1`),
    /// giving this span's last (max) doc ID across its 32 blocks.
    doc_delta: i32,
    /// `level1DocEndFP`: absolute byte offset (into the whole `.doc` buffer)
    /// of the START of the *next* level-1 entry, i.e. one past this span's own
    /// 32 level-0 blocks. Seeking straight here skips the whole span.
    doc_end_fp: usize,
}

/// Reads one level-1 skip entry at `r`'s current position (which must be a
/// level-1 entry boundary — `docStartFP` for span 0, or the previous span's
/// `doc_end_fp` afterward), leaving `r` positioned at the first level-0 block
/// header of the span. Shared by [`DocInput::read_postings`] (eager, discards
/// the return) and [`LazyDocsCursor::skip_level1_to`] (lazy, uses `doc_delta`
/// and `doc_end_fp` to decide whether to jump the whole span).
///
/// The impacts bytes (competitive-scoring metadata for the whole span) are
/// always skipped, never decoded — this port doesn't do impacts yet, same as
/// the level-0 header path in [`read_full_block_header`]. The pos/pay
/// sub-fields are parsed for wire-order correctness even though this reader
/// never uses them to seek `.pos`/`.pay`.
fn read_level1_entry(
    r: &mut SliceInput,
    index_has_freq: bool,
    index_has_pos: bool,
    index_has_offsets_or_payloads: bool,
) -> Result<Level1Entry> {
    // Steps 1-2 are always present, even for `IndexOptions::Docs` (no freqs):
    // `skipLevel1To` calls `readVInt`/`readVLong` unconditionally before the
    // `indexHasFreq` gate — plain vint/vlong, NOT `readVInt15`/`readVLong15`.
    let doc_delta = r.read_vint()?;
    let delta = r.read_vlong()? as usize;
    let doc_end_fp = delta.wrapping_add(r.position());

    if index_has_freq {
        // `skip1EndFP` (step 3a): a plain 2-byte `readShort`, the byte length
        // from right here to the end of this entry's metadata. Only used as a
        // consistency check (Java asserts `getFilePointer() == skip1EndFP`).
        let skip1_end_fp = (r.read_i16()? as i64 + r.position() as i64) as usize;
        // `numImpactBytes` (step 3b): another plain `readShort`, non-negative
        // (a length). Then skip that many raw impact bytes (step 3c).
        let num_impact_bytes = r.read_i16()? as usize;
        r.skip(num_impact_bytes)?;
        if index_has_pos {
            let _pos_end_fp_delta = r.read_vlong()?;
            let _pos_buffer_upto = r.read_byte()?;
            if index_has_offsets_or_payloads {
                let _pay_end_fp_delta = r.read_vlong()?;
                let _pay_buffer_upto = r.read_vint()?;
            }
        }
        debug_assert_eq!(r.position(), skip1_end_fp);
    }

    Ok(Level1Entry {
        doc_delta,
        doc_end_fp,
    })
}

/// A full block's level-0 skip header, decoded up to (but not including) the
/// block body (the `bitsPerValue` token and everything after it). This is the
/// part of `doMoveToNextLevel0Block`/`skipLevel0To`
/// (`Lucene104PostingsReader.java:739-762`, `818-871`) both code paths always
/// read — real Lucene's `advance()` uses exactly this much information (a
/// block's last doc ID, plus where its body starts and ends) to decide
/// whether to decode the body or `docIn.seek()` straight past it.
///
/// **What is genuinely skippable vs. what must still be touched**: every
/// field here is a small fixed-width or vint/vlong-prefixed value (including
/// the impacts byte run, whose *length* is read but whose *bytes* are
/// skipped via [`DataInput::skip`] rather than decoded) — so determining
/// `last_doc_id` and `body_start`/`body_len` never runs `ForUtil`/`PForUtil`
/// decode, which is the expensive part of a block (bit-unpacking 256 values).
/// That decode work is exactly what [`LazyDocsCursor`] avoids for a block
/// this header proves is entirely before the caller's target.
struct FullBlockHeader {
    /// This block's last (highest) doc ID — `prev_doc_id + docDelta`, proven
    /// consistent with the body's own delta-decoded last entry by every
    /// existing fixture/unit test that decodes both (see `read_full_block`).
    last_doc_id: i32,
    /// Byte offset (into the same buffer `r` reads from) where the block's
    /// body (`bitsPerValue` token onward) begins.
    body_start: usize,
    /// Byte offset where the block's body ends, i.e. where the next block's
    /// own level-0 header (or the tail block, or the term's end) begins.
    body_end: usize,
}

/// Reads one full block's level-0 header (see [`FullBlockHeader`]) without
/// touching the body. `r` is left positioned at `body_start` on return.
fn read_full_block_header(
    r: &mut SliceInput,
    prev_doc_id: i32,
    index_has_freq: bool,
    index_has_pos: bool,
    index_has_offsets_or_payloads: bool,
) -> Result<FullBlockHeader> {
    let _level0_num_bytes = r.read_vlong()?;
    let doc_delta = read_vint15(r)?;
    let last_doc_id = prev_doc_id + doc_delta;
    let block_length = read_vlong15(r)?;
    // `level0DocEndFP` in `Lucene104PostingsReader.doMoveToNextLevel0Block`
    // (`Lucene104PostingsReader.java:743-744`) is computed *immediately*
    // after reading `blockLength`, i.e. before the impacts/pos/pay fields
    // are read -- `blockLength` therefore measures from here (not from
    // `body_start` below) through the end of the whole block, so it
    // includes the impacts-length-prefixed bytes and pos/pay skip fields,
    // not just the `bitsPerValue`-onward body.
    let body_end = r.position() + block_length as usize;
    if index_has_freq {
        // Impacts byte-length is a plain vint here (`doMoveToNextLevel0Block`,
        // `Lucene104PostingsReader.java:746`), unlike level-1's vlong-prefixed
        // `numSkipBytes` -- confirmed against the reader source rather than
        // assumed from the tail-block/level-1 shape.
        let impacts_len = r.read_vint()? as usize;
        r.skip(impacts_len)?;

        // Level-0 pos/pay skip data (`Lucene104PostingsReader.java:754-761`):
        // parsed for wire-order correctness (this reader never skips ahead
        // with it for `.pos`/`.pay` themselves, only for `.doc`) only when
        // the field indexes positions.
        if index_has_pos {
            let _pos_end_fp_delta = r.read_vlong()?;
            let _pos_buffer_upto = r.read_byte()?;
            if index_has_offsets_or_payloads {
                let _pay_end_fp_delta = r.read_vlong()?;
                let _pay_buffer_upto = r.read_vint()?;
            }
        }
    }

    let body_start = r.position();
    Ok(FullBlockHeader {
        last_doc_id,
        body_start,
        body_end,
    })
}

/// Decodes a full block's body (the `bitsPerValue` token onward) — `r` must
/// already be positioned at [`FullBlockHeader::body_start`]. Shared by
/// [`read_full_block`] (eager path) and [`LazyDocsCursor`] (lazy path) so
/// there is exactly one body decoder to keep in sync with `ForUtil`/
/// `PForUtil`.
fn decode_full_block_body(
    r: &mut SliceInput,
    prev_doc_id: i32,
    index_has_freq: bool,
) -> Result<([i32; BLOCK_SIZE as usize], [i32; BLOCK_SIZE as usize])> {
    let bits_per_value_byte = r.read_byte()? as i8;
    let mut docs = [0i32; BLOCK_SIZE as usize];
    if bits_per_value_byte > 0 {
        let mut doc_deltas = [0u32; for_util::BLOCK_SIZE];
        for_util::for_decode(bits_per_value_byte as u32, r, &mut doc_deltas)?;
        let mut sum: i64 = prev_doc_id as i64;
        for (d, &delta) in docs.iter_mut().zip(doc_deltas.iter()) {
            sum += delta as i64;
            *d = sum as i32;
        }
    } else if bits_per_value_byte == 0 {
        // "0 is used to record that all 256 docs in the block are
        // consecutive" (`Lucene104PostingsReader.refillFullBlock`): every
        // delta is 1, no bytes follow.
        for (i, d) in docs.iter_mut().enumerate() {
            *d = prev_doc_id + 1 + i as i32;
        }
    } else {
        // Dense/unary bit-set encoding of doc deltas (`bitsPerValue < 0`,
        // `numLongs = -bitsPerValue`): the block's 256 doc IDs are the
        // positions of the set bits (ascending) in a `numLongs`-word bitset
        // based at `prevDocID + 1`, rather than a packed-delta array. The
        // writer picks this over `ForUtil`-packed deltas whenever it's
        // strictly more storage-efficient (`Lucene104PostingsWriter.
        // flushDocBlock`'s `numBitsNextBitsPerValue <=
        // numBitSetLongs*Long.SIZE` check) -- real-world dense postings (a
        // term present in every document of a run, e.g. this port's own
        // `big`/"everywhere" fixture) commonly take this path, so it isn't
        // an edge case to skip.
        let num_longs = (-(bits_per_value_byte as i32)) as usize;
        let mut words = vec![0u64; num_longs];
        for w in words.iter_mut() {
            *w = r.read_i64()? as u64;
        }
        let doc_bit_set_base = prev_doc_id as i64 + 1;
        let mut found = 0usize;
        'words: for (word_idx, &word) in words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let bit = bits.trailing_zeros() as i64;
                docs[found] = (doc_bit_set_base + (word_idx as i64) * 64 + bit) as i32;
                found += 1;
                if found == BLOCK_SIZE as usize {
                    break 'words;
                }
                bits &= bits - 1; // clear lowest set bit
            }
        }
        if found != BLOCK_SIZE as usize {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "full-block doc bit-set has fewer than BLOCK_SIZE set bits".into(),
            )));
        }
    }

    let mut freqs = [1i32; BLOCK_SIZE as usize];
    if index_has_freq {
        let mut freq_words = [0u32; for_util::BLOCK_SIZE];
        for_util::pfor_decode(r, &mut freq_words)?;
        for (f, &w) in freqs.iter_mut().zip(freq_words.iter()) {
            *f = w as i32;
        }
    }

    Ok((docs, freqs))
}

/// One full 256-doc block (`BlockPostingsEnum.refillFullBlock` plus the
/// level-0 skip header that precedes every full block on the wire,
/// `Lucene104PostingsWriter.flushDocBlock`'s `else` branch —
/// `docBufferUpto == BLOCK_SIZE`). Thin wrapper over
/// [`read_full_block_header`] + [`decode_full_block_body`]: this always
/// decodes the body, since the eager [`DocInput::read_postings`] caller
/// wants every doc regardless of what the header says — see
/// [`LazyDocsCursor`] for the decode-on-demand path that actually uses the
/// header to skip a block's body.
fn read_full_block(
    r: &mut SliceInput,
    prev_doc_id: i32,
    index_has_freq: bool,
    index_has_pos: bool,
    index_has_offsets_or_payloads: bool,
) -> Result<([i32; BLOCK_SIZE as usize], [i32; BLOCK_SIZE as usize])> {
    let header = read_full_block_header(
        r,
        prev_doc_id,
        index_has_freq,
        index_has_pos,
        index_has_offsets_or_payloads,
    )?;
    debug_assert_eq!(r.position(), header.body_start);
    let result = decode_full_block_body(r, prev_doc_id, index_has_freq)?;
    debug_assert_eq!(r.position(), header.body_end);
    Ok(result)
}

/// The `docFreq % BLOCK_SIZE` remainder after zero or more full blocks
/// (`BlockPostingsEnum.refillRemainder`'s non-singleton branch): the same
/// group-varint + trailing-vint-freq-exceptions scheme the pre-existing
/// single-block (`docFreq < BLOCK_SIZE`) path already implements, just with
/// `prev_doc_id` seeded from the last full block instead of always `-1`.
#[allow(clippy::too_many_arguments)]
fn read_tail_block(
    r: &mut SliceInput,
    prev_doc_id: i32,
    count: usize,
    index_has_freq: bool,
    docs: &mut Vec<i32>,
    freqs: &mut Vec<i32>,
) -> Result<()> {
    let mut raw = vec![0u64; count];
    r.read_group_vints(&mut raw)?;

    let start = docs.len();
    docs.resize(start + count, 0);
    freqs.resize(start + count, 1);
    if index_has_freq {
        for ((d, f), &v) in docs[start..]
            .iter_mut()
            .zip(freqs[start..].iter_mut())
            .zip(raw.iter())
        {
            *f = (v & 1) as i32;
            *d = (v >> 1) as i32;
        }
        for f in freqs[start..].iter_mut() {
            if *f == 0 {
                *f = r.read_vint()?;
            }
        }
    } else {
        for (d, &v) in docs[start..].iter_mut().zip(raw.iter()) {
            *d = v as i32;
        }
    }

    let mut sum: i64 = prev_doc_id as i64;
    for d in docs[start..].iter_mut() {
        sum += *d as i64;
        *d = sum as i32;
    }

    Ok(())
}

/// `docFreq == 1`: the single doc/freq is reconstructed entirely from the
/// term dictionary's metadata (`termState.singletonDocID`) and
/// `totalTermFreq` (implicitly the one doc's freq) — no `.doc` file access,
/// matching `BlockPostingsEnum.refillRemainder`'s singleton branch
/// (`Lucene104PostingsReader.java:640-646`).
pub fn singleton_postings(meta: TermMetadata, total_term_freq: i64) -> Result<Postings> {
    if meta.singleton_doc_id < 0 {
        return Err(Error::NoPreviousSingleton);
    }
    Ok(Postings {
        docs: vec![meta.singleton_doc_id],
        freqs: vec![total_term_freq as i32],
    })
}

/// `PostingsEnum.NO_MORE_DOCS` (`DocIdSetIterator.NO_MORE_DOCS`).
pub const NO_MORE_DOCS: i32 = i32::MAX;

/// An `advance()`-shaped cursor over an **already fully-materialized**
/// [`Postings`] — **not** real skip-ahead.
///
/// This is deliberately *not* Lucene's `Lucene104PostingsReader.
/// BlockPostingsEnum.advance()`: that method jumps between undecoded `.doc`
/// blocks using the level-0/level-1 skip pointers this module's decode
/// functions already parse-and-discard (see the module doc's "Deferred:
/// skip-ahead" section) — it can skip an entire 256-doc block's bytes
/// without ever decoding them. `DocInput::read_postings` above still fully
/// decodes every block up front into one `Vec<i32>` per term (the
/// eager-materialization design this whole file already commits to, same
/// tradeoff as `BlockTree`'s `TermsEnum`/`IndexedDISI`/the terms
/// dictionary/`BlockPackedReaderIterator` — see those modules' doc
/// comments). Given that, `advance()` here is simply a binary search over
/// the already-decoded `docs` array: it has `advance()`'s *interface*
/// (`PostingsEnum.advance(target)`'s doc-jump semantics, useful for a
/// conjunction/phrase-query caller that wants to intersect two postings
/// lists without linearly walking both) but none of the "skip bytes we
/// never decode" *performance* benefit real Lucene's skip data exists for —
/// every byte of the term's postings is decoded by `read_postings` before
/// this cursor ever runs. A real lazy skip-ahead (extending `DocInput` with
/// a stateful decode-on-demand iterator that uses the level-0 skip pointers
/// to jump between undecoded blocks) is tracked as future work in
/// `docs/parity.md` — do not read this type as proof that lazy wire-level
/// skipping exists.
///
/// Mirrors `DocIdSetIterator`'s contract: a cursor starts positioned before
/// the first doc (`doc_id() == -1`), `next_doc()`/`advance()` move strictly
/// forward, and both return [`NO_MORE_DOCS`] once exhausted. Advancing to a
/// target at or before the current doc ID is a documented **no-op** (returns
/// the current doc ID unchanged) rather than an error or a rewind — real
/// Lucene's contract technically forbids calling `advance()` with a target
/// `<= docID()` (`PostingsEnum`'s Javadoc), but callers here get a safe,
/// well-defined no-op instead of undefined behavior, since binary-searching
/// backward would be either wrong (if implemented as "search from the
/// start") or silently a no-op anyway (if implemented as "search from
/// current" like this one is) — better to name the guaranteed behavior than
/// leave it to accident.
pub struct PostingsCursor<'p> {
    postings: &'p Postings,
    /// Index into `postings.docs`/`postings.freqs` of the current position.
    /// `postings.docs.len()` once exhausted.
    idx: usize,
    /// Whether `next_doc()`/`advance()` has been called at least once
    /// (`doc_id()` reports `-1` until then, matching `DocIdSetIterator`'s
    /// "positioned before the first doc" starting state).
    started: bool,
}

impl<'p> PostingsCursor<'p> {
    /// A fresh cursor, positioned before the first doc.
    pub fn new(postings: &'p Postings) -> Self {
        PostingsCursor {
            postings,
            idx: 0,
            started: false,
        }
    }

    /// The current doc ID: `-1` before the first `next_doc()`/`advance()`
    /// call, [`NO_MORE_DOCS`] once exhausted, otherwise the doc ID at the
    /// cursor's position.
    pub fn doc_id(&self) -> i32 {
        if !self.started {
            -1
        } else if self.idx >= self.postings.docs.len() {
            NO_MORE_DOCS
        } else {
            self.postings.docs[self.idx]
        }
    }

    /// The current doc's frequency, or `None` before the first
    /// `next_doc()`/`advance()` call or once exhausted (mirrors `doc_id()`'s
    /// three-state contract; there is no freq to report in either edge
    /// case).
    pub fn freq(&self) -> Option<i32> {
        if self.started && self.idx < self.postings.docs.len() {
            Some(self.postings.freqs[self.idx])
        } else {
            None
        }
    }

    /// `PostingsEnum.nextDoc()`: moves to the next doc, returning its ID (or
    /// [`NO_MORE_DOCS`] if there isn't one).
    pub fn next_doc(&mut self) -> i32 {
        if !self.started {
            self.started = true;
            // idx is already 0 (the first doc, if any).
        } else if self.idx < self.postings.docs.len() {
            self.idx += 1;
        }
        self.doc_id()
    }

    /// `PostingsEnum.advance(target)`: moves forward to the first doc ID
    /// `>= target`, returning it (or [`NO_MORE_DOCS`] if none remains).
    /// Binary searches the already-decoded `docs` array from the current
    /// position onward (never backward — see this type's doc comment for
    /// why a `target <= doc_id()` is a documented no-op rather than an
    /// error).
    pub fn advance(&mut self, target: i32) -> i32 {
        self.started = true;
        let start = self.idx.min(self.postings.docs.len());
        let offset = self.postings.docs[start..].partition_point(|&d| d < target);
        self.idx = start + offset;
        self.doc_id()
    }
}

/// A genuinely lazy `(docID, freq)` iterator: decodes one block at a time
/// on demand, and — for `advance()` targets beyond a not-yet-decoded full
/// block's entire doc range — skips that block's body without ever running
/// `ForUtil`/`PForUtil` decode on it, using the level-0 header's own
/// `docDelta`/`blockLength` fields (see [`FullBlockHeader`]).
///
/// ## What is actually skipped, and under what conditions
///
/// This is the honest boundary the module doc's "Deferred" section asks for:
///
/// - **Full blocks (`BLOCK_SIZE` = 256 docs each) are skippable at zero
///   decode cost.** A full block's level-0 header (`level0NumBytes`,
///   `docDelta`, `blockLength`, plus impacts/pos/pay skip fields when the
///   field has freqs/positions) is *always* read to reach the next block —
///   there is no way to avoid touching those handful of vint/vlong/byte
///   fields — but reading them never invokes `ForUtil`/`PForUtil` (the
///   bit-unpacking of 256 packed values, the actual expensive part of a
///   block). If `advance(target)` finds `target > header.last_doc_id`, it
///   jumps straight to `header.body_end` and moves to the next block without
///   decoding this one's body at all. This works for **every** full block a
///   term has, regardless of `docFreq` — it does not require
///   `docFreq >= LEVEL1_NUM_DOCS` (8192). The 8192 threshold is what real
///   Lucene's **level-1** skip list needs (skipping *32 full blocks at once*
///   without reading even their level-0 headers); level-0 skip-past-one-block
///   is available and used here for any term with at least one full block
///   (`docFreq >= BLOCK_SIZE` = 256), which this port's fixtures already
///   exercise (see `blocktree_fixtures.rs`'s "big"/"everywhere" field).
/// - **The tail block (`docFreq % BLOCK_SIZE` remainder, or the entire term
///   when `docFreq < BLOCK_SIZE`) carries no skip data at all** — real
///   Lucene's own `PostingsUtil.writeVIntBlock` format has no level-0 header,
///   no length prefix, nothing to jump past. Reaching the tail always means
///   decoding it in full (`read_tail_block`), lazy or not. This matches real
///   `Lucene104PostingsReader.refillRemainder`, which has no skip variant
///   either.
/// - **`docFreq >= LEVEL1_NUM_DOCS` (8192): whole 32-block spans are
///   skippable via the level-1 entry.** Above that threshold the `.doc`
///   stream interleaves a level-1 skip entry before each span of 32 full
///   level-0 blocks. [`Self::skip_level1_to`] reads that entry and, when the
///   span's last doc (`level1_last_doc_id`) is still behind the target,
///   `seek()`s straight to the next entry — jumping all 32 blocks without
///   reading even their individual level-0 headers, the coarser counterpart
///   to the per-block skip above. Only once fewer than `LEVEL1_NUM_DOCS`
///   docs remain does it fall back to the one-header-at-a-time level-0 path.
/// - **Early exit still pays off even without any skip**: unlike
///   [`DocInput::read_postings`], which always decodes the *entire* term
///   up front, this cursor decodes blocks one at a time, so a caller that
///   stops early (e.g. a conjunction query whose other clause is exhausted
///   first) never decodes the remaining blocks regardless of whether they
///   were skippable via header comparison.
///
/// `.pos`/`.pay` are untouched by this cursor (same scope as `DocInput`
/// itself) — a caller needing positions still goes through
/// [`crate::postings::read_positions`] separately, sequentially, once it
/// knows which docs it wants.
#[derive(Debug)]
pub struct LazyDocsCursor<'a> {
    r: SliceInput<'a>,
    index_has_freq: bool,
    index_has_pos: bool,
    index_has_offsets_or_payloads: bool,
    /// This term's total `docFreq` — needed to recompute `doc_count_left` at
    /// each level-1 span boundary (`docFreq - level1_doc_count_upto`), exactly
    /// like `Lucene104PostingsReader.skipLevel1To`.
    doc_freq: i32,
    /// Last doc ID that is either fully decoded-and-consumed-past or
    /// skipped-past — the delta base for the next block's doc IDs.
    prev_doc_id: i32,
    /// Docs not yet decoded or skipped (full blocks + the trailing tail, if
    /// any).
    doc_count_left: i32,
    /// `level1LastDocID`: the highest doc ID covered by the current level-1
    /// span, or [`NO_MORE_DOCS`] once past the last level-1 entry (or always,
    /// for a `docFreq < LEVEL1_NUM_DOCS` term with no level-1 entries). An
    /// `advance(target)` with `target > level1_last_doc_id` triggers
    /// [`Self::skip_level1_to`] to jump whole 32-block spans.
    level1_last_doc_id: i32,
    /// `level1DocEndFP`: absolute byte offset of the next level-1 entry (one
    /// past the current span's 32 level-0 blocks). Where
    /// [`Self::skip_level1_to`] seeks to skip a whole span.
    level1_doc_end_fp: usize,
    /// `level1DocCountUpto`: how many docs precede the current level-1 span
    /// (always a multiple of `LEVEL1_NUM_DOCS`).
    level1_doc_count_upto: i32,
    block_docs: [i32; BLOCK_SIZE as usize],
    block_freqs: [i32; BLOCK_SIZE as usize],
    /// Number of valid entries in `block_docs`/`block_freqs` (`BLOCK_SIZE`
    /// for a full block, `docFreq % BLOCK_SIZE` for the tail, `0` when no
    /// block is currently loaded).
    block_len: usize,
    /// Index into `block_docs`/`block_freqs` of the current position.
    block_pos: usize,
    /// `-1` before the first `next_doc()`/`advance()` call,
    /// [`NO_MORE_DOCS`] once exhausted, otherwise the current doc ID.
    doc_id: i32,
}

impl<'a> LazyDocsCursor<'a> {
    /// The current doc ID (see the `doc_id` field's doc comment for the
    /// three-state contract).
    pub fn doc_id(&self) -> i32 {
        self.doc_id
    }

    /// The current doc's frequency, or `None` before the first
    /// `next_doc()`/`advance()` call or once exhausted.
    pub fn freq(&self) -> Option<i32> {
        if self.doc_id != -1 && self.doc_id != NO_MORE_DOCS {
            Some(self.block_freqs[self.block_pos])
        } else {
            None
        }
    }

    /// `PostingsEnum.nextDoc()`: moves to the next doc, returning its ID (or
    /// [`NO_MORE_DOCS`] if there isn't one). Implemented as `advance(doc_id +
    /// 1)`, saturating rather than overflowing once already at
    /// [`NO_MORE_DOCS`].
    pub fn next_doc(&mut self) -> Result<i32> {
        let target = if self.doc_id == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        } else {
            self.doc_id.saturating_add(1)
        };
        self.advance(target)
    }

    /// `PostingsEnum.advance(target)`: moves forward to the first doc ID
    /// `>= target`, returning it (or [`NO_MORE_DOCS`] if none remains).
    /// Advancing to a target at or before the current doc ID is a documented
    /// no-op (same contract as [`PostingsCursor::advance`]).
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        if self.doc_id == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        if target <= self.doc_id {
            return Ok(self.doc_id);
        }

        // First, try the already-decoded current block (covers the common
        // "advance a little" and "nextDoc" cases without touching the wire
        // at all).
        if self.block_pos < self.block_len {
            let offset =
                self.block_docs[self.block_pos..self.block_len].partition_point(|&d| d < target);
            if self.block_pos + offset < self.block_len {
                self.block_pos += offset;
                self.doc_id = self.block_docs[self.block_pos];
                return Ok(self.doc_id);
            }
            // Target is beyond every doc left in this block: fall through
            // to load the next one.
            self.block_pos = self.block_len;
        }

        loop {
            // Level-1 skip: before touching any level-0 block, jump past whole
            // 32-block spans that are entirely behind `target`
            // (`Lucene104PostingsReader.moveToNextLevel0Block`/`doAdvanceShallow`
            // both call `skipLevel1To(target)` first). For a
            // `docFreq < LEVEL1_NUM_DOCS` term `level1_last_doc_id` is pinned
            // at NO_MORE_DOCS, so this never fires.
            if target > self.level1_last_doc_id {
                self.skip_level1_to(target)?;
            }

            if self.doc_count_left == 0 {
                self.block_len = 0;
                self.block_pos = 0;
                self.doc_id = NO_MORE_DOCS;
                return Ok(NO_MORE_DOCS);
            }

            if self.doc_count_left >= BLOCK_SIZE {
                let header = read_full_block_header(
                    &mut self.r,
                    self.prev_doc_id,
                    self.index_has_freq,
                    self.index_has_pos,
                    self.index_has_offsets_or_payloads,
                )?;

                if header.last_doc_id < target {
                    // The whole block is behind `target`: jump straight to
                    // its end, never decoding the body (the actual `ForUtil`/
                    // `PForUtil` bit-unpack this cursor avoids).
                    self.r.seek(header.body_end)?;
                    self.prev_doc_id = header.last_doc_id;
                    self.doc_count_left -= BLOCK_SIZE;
                    continue;
                }

                // Target lands inside this block (or the block's last doc
                // is still < target is false, i.e. >= target): decode it.
                debug_assert_eq!(self.r.position(), header.body_start);
                let (docs, freqs) =
                    decode_full_block_body(&mut self.r, self.prev_doc_id, self.index_has_freq)?;
                self.block_docs = docs;
                self.block_freqs = freqs;
                self.block_len = BLOCK_SIZE as usize;
                self.prev_doc_id = header.last_doc_id;
                self.doc_count_left -= BLOCK_SIZE;

                let offset = self.block_docs.partition_point(|&d| d < target);
                self.block_pos = offset;
                self.doc_id = self.block_docs[offset];
                return Ok(self.doc_id);
            }

            // The tail block: no skip data on the wire at all, must decode.
            let count = self.doc_count_left as usize;
            let mut docs = Vec::with_capacity(count);
            let mut freqs = Vec::with_capacity(count);
            read_tail_block(
                &mut self.r,
                self.prev_doc_id,
                count,
                self.index_has_freq,
                &mut docs,
                &mut freqs,
            )?;
            self.block_docs[..count].copy_from_slice(&docs);
            self.block_freqs[..count].copy_from_slice(&freqs);
            self.block_len = count;
            self.doc_count_left = 0;

            let offset = self.block_docs[..count].partition_point(|&d| d < target);
            self.block_pos = offset;
            self.doc_id = if offset < count {
                self.block_docs[offset]
            } else {
                NO_MORE_DOCS
            };
            return Ok(self.doc_id);
        }
    }

    /// Port of `Lucene104PostingsReader.skipLevel1To`
    /// (`Lucene104PostingsReader.java:674-719`): consume level-1 skip entries,
    /// jumping straight past whole 32-block spans whose last doc is still
    /// behind `target`, until either a span that contains `target` is reached
    /// (leaving `r` at that span's first level-0 block header, with
    /// `level1_last_doc_id >= target`) or fewer than `LEVEL1_NUM_DOCS` docs
    /// remain (leaving `r` at the trailing sub-8192 region of full blocks +
    /// tail, with `level1_last_doc_id == NO_MORE_DOCS`). The subsequent
    /// level-0 loop in [`Self::advance`] then takes over exactly as it does
    /// for a sub-8192 term.
    ///
    /// Each iteration re-seeks to `level1_doc_end_fp` (the known span
    /// boundary) and recomputes `doc_count_left` from `level1_doc_count_upto`,
    /// so the caller's running `doc_count_left`/`r` position before the call
    /// don't matter — this is what lets a whole span be skipped even from the
    /// middle of the previous one.
    fn skip_level1_to(&mut self, target: i32) -> Result<()> {
        loop {
            self.prev_doc_id = self.level1_last_doc_id;
            self.r.seek(self.level1_doc_end_fp)?;
            self.doc_count_left = self.doc_freq - self.level1_doc_count_upto;
            self.level1_doc_count_upto += LEVEL1_NUM_DOCS;

            if self.doc_count_left < LEVEL1_NUM_DOCS {
                // Fewer than a full span remains: no level-1 entry precedes it.
                // `r` is now at the first of the trailing level-0 blocks.
                self.level1_last_doc_id = NO_MORE_DOCS;
                break;
            }

            let entry = read_level1_entry(
                &mut self.r,
                self.index_has_freq,
                self.index_has_pos,
                self.index_has_offsets_or_payloads,
            )?;
            self.level1_last_doc_id += entry.doc_delta;
            self.level1_doc_end_fp = entry.doc_end_fp;

            if self.level1_last_doc_id >= target {
                // `target` is within this span: `r` is positioned at the
                // span's first level-0 block header.
                break;
            }
            // The whole span is behind `target`: loop again, which re-seeks to
            // `level1_doc_end_fp` (past this span's 32 blocks) without touching
            // any of their level-0 headers.
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::data_output::DataOutput;

    /// Test-only encoder for `GroupVIntUtil.writeGroupVInts`'s wire format
    /// (groups of 4 values, 1 flag byte packing each value's byte-length minus
    /// one, then that many little-endian bytes per value; a final partial
    /// group of fewer than 4 falls back to plain vints) — mirrors this
    /// project's pattern of small test-only encoders (see `data_input.rs`'s
    /// own tests) rather than adding a writer this port doesn't otherwise need
    /// yet.
    fn write_group_vints(out: &mut Vec<u8>, values: &[u32]) {
        let mut i = 0;
        while i + 4 <= values.len() {
            let chunk = &values[i..i + 4];
            let lens: Vec<u8> = chunk
                .iter()
                .map(|&v| {
                    let bytes = if v == 0 {
                        1
                    } else {
                        4 - (v.leading_zeros() / 8)
                    };
                    (bytes - 1) as u8
                })
                .collect();
            let flag = (lens[0] << 6) | (lens[1] << 4) | (lens[2] << 2) | lens[3];
            out.push(flag);
            for (j, &v) in chunk.iter().enumerate() {
                let n = lens[j] as usize + 1;
                out.extend_from_slice(&v.to_le_bytes()[..n]);
            }
            i += 4;
        }
        while i < values.len() {
            out.write_vint(values[i] as i32);
            i += 1;
        }
    }

    fn header_and_footer(codec: &str, id: &[u8; ID_LENGTH]) -> (Vec<u8>, Vec<u8>) {
        let mut before = Vec::new();
        codec_util::write_index_header(&mut before, codec, VERSION_CURRENT, id, "");
        let mut after = Vec::new();
        codec_util::write_footer(&mut after);
        (before, after)
    }

    #[test]
    fn open_rejects_bad_header() {
        let id = [1u8; ID_LENGTH];
        let mut doc = Vec::new();
        codec_util::write_index_header(&mut doc, "WrongCodec", VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut doc);
        assert!(DocInput::open(&doc, &id, "").is_err());
    }

    #[test]
    fn read_postings_two_docs_with_freqs() {
        let id = [2u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        // docFreq=2: deltas [3, 2] (docIDs 2 and 4), freqs [2, 1].
        // group-varint packing: (delta<<1)|(freq==1?1:0)
        write_group_vints(&mut doc, &[3 << 1, (2 << 1) | 1]);
        doc.write_vint(2); // explicit freq for the first doc (freq != 1)
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, 2, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(postings.docs, vec![2, 4]);
        assert_eq!(postings.freqs, vec![2, 1]);
    }

    #[test]
    fn read_postings_docs_only_no_freqs() {
        let id = [3u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        // docFreq=3, plain deltas (no freq bit-packing): docIDs 0,1,5 -> deltas 1,1,4
        write_group_vints(&mut doc, &[1, 1, 4]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, 3, IndexOptions::Docs, false)
            .unwrap();
        assert_eq!(postings.docs, vec![0, 1, 5]);
        assert_eq!(postings.freqs, vec![1, 1, 1]);
    }

    #[test]
    fn read_postings_all_freq_one_docs_only_bit_path() {
        // Every doc has freq==1 (bit set), so no trailing freq vints at all --
        // exercises the branch where the second (freq-exception) loop in
        // `read_postings` never fires.
        let id = [6u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        // docIDs 0, 3, 4 (deltas 1, 3, 1), freq==1 for all -> bit always set.
        write_group_vints(&mut doc, &[(1 << 1) | 1, (3 << 1) | 1, (1 << 1) | 1]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, 3, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(postings.docs, vec![0, 3, 4]);
        assert_eq!(postings.freqs, vec![1, 1, 1]);
    }

    #[test]
    fn read_postings_block_size_minus_one_docs() {
        // docFreq == BLOCK_SIZE - 1 (255): the largest docFreq this slice's
        // group-varint (non-PFOR) path supports -- one below the boundary
        // where `read_postings` rejects with Unsupported.
        let id = [7u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        let n = (BLOCK_SIZE - 1) as usize;
        // Consecutive doc IDs 0..n, delta=1 each, freq==2 for every doc (bit
        // clear) so every doc also needs a trailing freq vint.
        let deltas: Vec<u32> = (0..n).map(|_| 1u32 << 1).collect();
        write_group_vints(&mut doc, &deltas);
        for _ in 0..n {
            doc.write_vint(2);
        }
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, n as i32, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(postings.docs, (0..n as i32).collect::<Vec<_>>());
        assert!(postings.freqs.iter().all(|&f| f == 2));
        assert_eq!(postings.freqs.len(), n);
    }

    #[test]
    fn read_postings_rejects_singleton_doc_freq() {
        let id = [4u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        doc.extend_from_slice(&footer);
        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp: 0,
            singleton_doc_id: 7,
            ..TermMetadata::EMPTY
        };
        let err = input
            .read_postings(meta, 1, IndexOptions::Docs, false)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    /// Writes a level-1 skip entry for `IndexOptions::Docs` (no freqs): just
    /// the vInt doc-delta and the vLong `doc_end_fp` delta, which (measured
    /// from right after the vLong) must equal `span.len()` so the entry points
    /// exactly past its own span. See [`read_level1_entry`].
    fn write_level1_entry_docs(doc: &mut Vec<u8>, doc_delta: i32, span: &[u8]) {
        doc.write_vint(doc_delta);
        doc.write_vlong(span.len() as i64);
    }

    #[test]
    fn read_postings_level1_span_plus_tail() {
        // docFreq == LEVEL1_NUM_DOCS + 8 (8200): one level-1 entry, then a
        // span of 32 all-consecutive full blocks (docs 0..8191), then an
        // 8-doc group-varint tail (docs 8192..8199) with no more level-1
        // entries.
        let id = [5u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        let mut span = Vec::new();
        for _ in 0..LEVEL1_FACTOR {
            write_full_block(&mut span, false, 0);
        }
        // doc_delta 8192 -> the span's last doc is -1 + 8192 = 8191.
        write_level1_entry_docs(&mut doc, LEVEL1_NUM_DOCS, &span);
        doc.extend_from_slice(&span);
        // Tail: 8 consecutive docs (deltas all 1) from prevDocID 8191.
        write_group_vints(&mut doc, &[1; 8]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, LEVEL1_NUM_DOCS + 8, IndexOptions::Docs, false)
            .unwrap();
        assert_eq!(postings.docs, (0..LEVEL1_NUM_DOCS + 8).collect::<Vec<_>>());
        assert!(postings.freqs.iter().all(|&f| f == 1));
    }

    #[test]
    fn read_level1_entry_decodes_all_fields() {
        // A DocsAndFreqsAndPositions(+payloads) level-1 entry, hand-built, to
        // confirm every field is read in the right order and doc_delta/
        // doc_end_fp come out exactly right (and the internal skip1EndFP
        // consistency check holds -- it's a debug_assert that fires in tests).
        let mut bytes = Vec::new();
        bytes.write_vint(8191); // doc_delta
        bytes.write_vlong(100); // doc_end_fp delta
        let pos_after_vlong = bytes.len();
        // freq metadata: skip1EndFP short, then numImpactBytes short, impacts,
        // then pos (vlong+byte) and pay (vlong+vint) sub-fields.
        // metadata after short1 = short2 (2) + 3 impact bytes + pos(vlong 1 +
        // byte 1) + pay(vlong 1 + vint 1) = 2 + 3 + 4 = 9.
        bytes.write_i16(9); // skip1EndFP offset
        bytes.write_i16(3); // numImpactBytes
        bytes.write_bytes(&[0xDE, 0xAD, 0xBE]); // 3 impact bytes (skipped)
        bytes.write_vlong(50); // posEndFP delta (discarded)
        bytes.write_byte(7); // posBufferUpto (discarded)
        bytes.write_vlong(60); // payEndFP delta (discarded)
        bytes.write_vint(9); // payBufferUpto (discarded)
        let metadata_end = bytes.len();

        let mut r = SliceInput::new(&bytes);
        let entry = read_level1_entry(&mut r, true, true, true).unwrap();
        assert_eq!(entry.doc_delta, 8191);
        assert_eq!(entry.doc_end_fp, 100 + pos_after_vlong);
        // r is left at the start of the span's first level-0 block header.
        assert_eq!(r.position(), metadata_end);
    }

    #[test]
    fn read_level1_entry_docs_only_reads_just_delta_fields() {
        // IndexOptions::Docs: no freq gate, so only the vInt doc-delta and
        // vLong doc_end_fp delta are present -- no impacts/pos/pay.
        let mut bytes = Vec::new();
        bytes.write_vint(500);
        bytes.write_vlong(40);
        let pos_after_vlong = bytes.len();
        bytes.write_bytes(&[0x11, 0x22]); // "span" bytes that must NOT be read
        let mut r = SliceInput::new(&bytes);
        let entry = read_level1_entry(&mut r, false, false, false).unwrap();
        assert_eq!(entry.doc_delta, 500);
        assert_eq!(entry.doc_end_fp, 40 + pos_after_vlong);
        assert_eq!(r.position(), pos_after_vlong);
    }

    /// Test-only encoder for a full 256-doc block's level-0 header +
    /// doc-delta/freq payload (`Lucene104PostingsWriter.flushDocBlock`'s
    /// `else` branch), specialized to the `bitsPerValue == 0`
    /// ("all consecutive") doc-delta encoding and (optionally) the
    /// `PForUtil` all-equal fast path for freqs, since those need no
    /// `ForUtil`/`PForUtil` packed body to hand-construct — the
    /// lane-interleaved bit-packed paths are exercised by the
    /// `for_util` module's own tests and by the `GenBlockTree.java`
    /// differential fixture (real `IndexWriter` bytes).
    ///
    /// `docDelta` and `blockLength` are real, consistent header fields here
    /// (not filler) — [`LazyDocsCursor`]'s skip-ahead relies on them being
    /// accurate, unlike the pre-lazy-cursor version of this helper, which
    /// only needed `read_full_block`'s wire-order-only decode to work.
    /// `docDelta` is always `BLOCK_SIZE` (256), matching the "all 256 deltas
    /// are 1" body this helper always writes.
    fn write_full_block(out: &mut Vec<u8>, index_has_freq: bool, freq_value: i32) {
        let mut body = Vec::new();
        if index_has_freq {
            // impacts: empty (zero-length section is valid — a reader that
            // needed them would just see none).
            body.write_vlong(0);
        }
        body.write_byte(0); // bitsPerValue == 0: all 256 deltas are 1.
        if index_has_freq {
            body.write_byte(0); // PForUtil token: bitsPerValue=0, numExceptions=0
            body.write_vint(freq_value);
        }

        // `blockLength` is measured from right after this field (i.e. from
        // right here) through the end of the whole block -- see
        // `read_full_block_header`'s doc comment -- so it equals `body.len()`
        // exactly, since this helper writes no impacts/pos/pay bytes before
        // `body` starts recording (`index_has_freq`'s impacts-length vlong
        // above is itself part of `body`).
        out.write_vlong(body.len() as i64); // level0NumBytes (not used to skip in these tests)
        out.write_i16(BLOCK_SIZE as i16); // docDelta via writeVInt15
        out.write_i16(body.len() as i16); // blockLength via writeVLong15
        out.write_bytes(&body);
    }

    #[test]
    fn read_postings_exactly_one_full_block_no_tail() {
        // docFreq == BLOCK_SIZE (256): one full block, no tail bytes at all.
        let id = [8u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_full_block(&mut doc, true, 3);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(postings.docs, (0..BLOCK_SIZE).collect::<Vec<_>>());
        assert!(postings.freqs.iter().all(|&f| f == 3));
        assert_eq!(postings.docs.len(), BLOCK_SIZE as usize);
    }

    #[test]
    fn read_postings_one_full_block_plus_one_doc_tail() {
        // docFreq == BLOCK_SIZE + 1 (257): one full block (docs 0..256) then
        // a 1-doc group-varint tail block, prevDocID chained from the full
        // block's last doc (255).
        let id = [9u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_full_block(&mut doc, true, 3);
        // Tail: 1 doc, delta=5 from prevDocID=255 -> docID 260, freq=7 (bit
        // clear, explicit freq vint follows).
        write_group_vints(&mut doc, &[5 << 1]);
        doc.write_vint(7);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, BLOCK_SIZE + 1, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(postings.docs.len(), BLOCK_SIZE as usize + 1);
        assert_eq!(
            &postings.docs[..BLOCK_SIZE as usize],
            &(0..BLOCK_SIZE).collect::<Vec<_>>()[..]
        );
        assert_eq!(postings.docs[BLOCK_SIZE as usize], 260);
        assert_eq!(postings.freqs[BLOCK_SIZE as usize], 7);
        assert!(postings.freqs[..BLOCK_SIZE as usize]
            .iter()
            .all(|&f| f == 3));
    }

    #[test]
    fn read_postings_multi_block_docs_only_no_freqs() {
        // IndexOptions::Docs (no freqs): full block omits impacts and the
        // PForUtil freq block entirely.
        let id = [10u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_full_block(&mut doc, false, 0);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap();
        assert_eq!(postings.docs, (0..BLOCK_SIZE).collect::<Vec<_>>());
        assert!(postings.freqs.iter().all(|&f| f == 1));
    }

    #[test]
    fn read_full_block_bitset_encoding_decodes_dense_docs() {
        // bitsPerValue < 0: dense unary bit-set doc-delta encoding. 8 words
        // of 0x5555...5 (every even bit set, 32 per word) give exactly
        // BLOCK_SIZE (256) set bits at positions 0,2,4,...,510 -- docIDs
        // 0,2,...,510 since prevDocID=-1 puts docBitSetBase at 0. This is
        // the same branch real Lucene picks for the `big`/"everywhere"
        // fixture term (see `blocktree_fixtures.rs`).
        let id = [11u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        let mut body = Vec::new();
        let num_longs = 8u8;
        body.write_byte((-(num_longs as i8)) as u8);
        for _ in 0..num_longs {
            body.write_bytes(&0x5555_5555_5555_5555u64.to_le_bytes());
        }
        doc.write_vlong(body.len() as i64);
        doc.write_i16(1);
        doc.write_i16(body.len() as i16);
        doc.write_bytes(&body);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap();
        let expected: Vec<i32> = (0..BLOCK_SIZE).map(|i| i * 2).collect();
        assert_eq!(postings.docs, expected);
        assert!(postings.freqs.iter().all(|&f| f == 1));
    }

    #[test]
    fn read_full_block_bitset_encoding_rejects_too_few_set_bits() {
        // A corrupted/truncated bit-set with fewer than BLOCK_SIZE set bits
        // must be a decode error, not a silently short postings list.
        let id = [12u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        let mut body = Vec::new();
        body.write_byte((-4i8) as u8); // 4 longs = 256 bits, but none set
        body.extend_from_slice(&[0u8; 32]);
        doc.write_vlong(body.len() as i64);
        doc.write_i16(1);
        doc.write_i16(body.len() as i16);
        doc.write_bytes(&body);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let err = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn read_full_block_packed_encoding_decodes_bit_packed_deltas() {
        // bitsPerValue > 0: the `for_util::for_decode` packed-delta branch,
        // the encoding real Lucene picks when the doc IDs are neither fully
        // consecutive nor dense enough for the bit-set path (see the
        // `bitsPerValue < 0` test above). Deltas alternate 1/3 (needs 2 bits
        // per value), encoded via `for_util`'s own test-only encoder so this
        // exercises the exact same wire format `for_decode` expects.
        let id = [13u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        let mut deltas = [0u32; for_util::BLOCK_SIZE];
        for (i, d) in deltas.iter_mut().enumerate() {
            *d = if i % 2 == 0 { 1 } else { 3 };
        }
        let bits_per_value = 2u32;
        let packed = for_util::test_support::encode_block(&deltas, bits_per_value);

        let mut body = Vec::new();
        body.write_byte(bits_per_value as u8);
        body.write_bytes(&packed);
        doc.write_vlong(body.len() as i64);
        doc.write_i16(1);
        doc.write_i16(body.len() as i16);
        doc.write_bytes(&body);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap();
        let mut expected = Vec::with_capacity(BLOCK_SIZE as usize);
        let mut sum = -1i32;
        for &d in &deltas {
            sum += d as i32;
            expected.push(sum);
        }
        assert_eq!(postings.docs, expected);
    }

    #[test]
    fn singleton_postings_reconstructs_from_metadata() {
        let meta = TermMetadata {
            doc_start_fp: 123,
            singleton_doc_id: 9,
            ..TermMetadata::EMPTY
        };
        let postings = singleton_postings(meta, 4).unwrap();
        assert_eq!(postings.docs, vec![9]);
        assert_eq!(postings.freqs, vec![4]);
    }

    #[test]
    fn singleton_postings_rejects_non_singleton_metadata() {
        let meta = TermMetadata {
            doc_start_fp: 0,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        assert!(singleton_postings(meta, 1).is_err());
    }

    fn postings(docs: &[i32], freqs: &[i32]) -> Postings {
        Postings {
            docs: docs.to_vec(),
            freqs: freqs.to_vec(),
        }
    }

    #[test]
    fn cursor_starts_before_first_doc() {
        let p = postings(&[2, 5, 9], &[1, 1, 1]);
        let cursor = PostingsCursor::new(&p);
        assert_eq!(cursor.doc_id(), -1);
        assert_eq!(cursor.freq(), None);
    }

    #[test]
    fn cursor_next_doc_walks_in_order() {
        let p = postings(&[2, 5, 9], &[3, 4, 5]);
        let mut cursor = PostingsCursor::new(&p);
        assert_eq!(cursor.next_doc(), 2);
        assert_eq!(cursor.freq(), Some(3));
        assert_eq!(cursor.next_doc(), 5);
        assert_eq!(cursor.freq(), Some(4));
        assert_eq!(cursor.next_doc(), 9);
        assert_eq!(cursor.freq(), Some(5));
        assert_eq!(cursor.next_doc(), NO_MORE_DOCS);
        assert_eq!(cursor.freq(), None);
        // Calling next_doc() again once exhausted stays exhausted (idempotent).
        assert_eq!(cursor.next_doc(), NO_MORE_DOCS);
    }

    #[test]
    fn cursor_advance_before_first_doc_lands_on_first() {
        let p = postings(&[2, 5, 9], &[1, 1, 1]);
        let mut cursor = PostingsCursor::new(&p);
        // target 0 is before the first doc (2): should land on 2.
        assert_eq!(cursor.advance(0), 2);
        assert_eq!(cursor.freq(), Some(1));
    }

    #[test]
    fn cursor_advance_exact_match() {
        let p = postings(&[2, 5, 9], &[1, 2, 3]);
        let mut cursor = PostingsCursor::new(&p);
        assert_eq!(cursor.advance(5), 5);
        assert_eq!(cursor.freq(), Some(2));
    }

    #[test]
    fn cursor_advance_between_docs_lands_on_next_higher() {
        let p = postings(&[2, 5, 9], &[1, 1, 1]);
        let mut cursor = PostingsCursor::new(&p);
        // target 6 is between 5 and 9: should land on 9.
        assert_eq!(cursor.advance(6), 9);
    }

    #[test]
    fn cursor_advance_past_last_doc_exhausts() {
        let p = postings(&[2, 5, 9], &[1, 1, 1]);
        let mut cursor = PostingsCursor::new(&p);
        assert_eq!(cursor.advance(100), NO_MORE_DOCS);
        assert_eq!(cursor.freq(), None);
        // Once exhausted, further advances stay exhausted.
        assert_eq!(cursor.advance(200), NO_MORE_DOCS);
    }

    #[test]
    fn cursor_advance_on_empty_postings() {
        let p = postings(&[], &[]);
        let mut cursor = PostingsCursor::new(&p);
        assert_eq!(cursor.doc_id(), -1);
        assert_eq!(cursor.advance(0), NO_MORE_DOCS);
        assert_eq!(cursor.freq(), None);
    }

    #[test]
    fn cursor_advance_to_doc_before_current_is_a_documented_no_op() {
        // advance() to a target <= the current doc ID does not rewind: it
        // is a documented no-op (binary search never looks backward from
        // the cursor's current index) rather than an error.
        let p = postings(&[2, 5, 9, 20], &[1, 1, 1, 1]);
        let mut cursor = PostingsCursor::new(&p);
        assert_eq!(cursor.advance(9), 9);
        assert_eq!(cursor.advance(5), 9, "no-op: target is behind current doc");
        assert_eq!(cursor.advance(9), 9, "no-op: target equals current doc");
        // Cursor can still move forward normally afterward.
        assert_eq!(cursor.advance(20), 20);
    }

    #[test]
    fn cursor_advance_then_next_doc_continues_from_landed_position() {
        let p = postings(&[2, 5, 9, 20], &[1, 2, 3, 4]);
        let mut cursor = PostingsCursor::new(&p);
        assert_eq!(cursor.advance(6), 9);
        assert_eq!(cursor.next_doc(), 20);
        assert_eq!(cursor.freq(), Some(4));
        assert_eq!(cursor.next_doc(), NO_MORE_DOCS);
    }

    #[test]
    fn cursor_advance_to_no_more_docs_target_exhausts() {
        let p = postings(&[2, 5], &[1, 1]);
        let mut cursor = PostingsCursor::new(&p);
        assert_eq!(cursor.advance(NO_MORE_DOCS), NO_MORE_DOCS);
    }

    #[test]
    fn lazy_cursor_rejects_singleton_doc_freq() {
        let id = [20u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        doc.extend_from_slice(&footer);
        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp: 0,
            singleton_doc_id: 7,
            ..TermMetadata::EMPTY
        };
        let err = input
            .lazy_cursor(meta, 1, IndexOptions::Docs, false)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn lazy_cursor_level1_sequential_next_doc_matches_read_postings() {
        // docFreq == LEVEL1_NUM_DOCS + 8 (8200): one level-1 span (32 blocks)
        // + an 8-doc tail. A full `next_doc()` walk through the lazy cursor
        // (which reads the level-1 entry, decodes each of the span's 32
        // blocks on demand, then the tail) must match the eager
        // `read_postings` result byte-for-byte across the span boundary.
        let id = [21u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        let mut span = Vec::new();
        for _ in 0..LEVEL1_FACTOR {
            write_full_block(&mut span, false, 0);
        }
        write_level1_entry_docs(&mut doc, LEVEL1_NUM_DOCS, &span);
        doc.extend_from_slice(&span);
        write_group_vints(&mut doc, &[1; 8]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let doc_freq = LEVEL1_NUM_DOCS + 8;
        let eager = input
            .read_postings(meta, doc_freq, IndexOptions::Docs, false)
            .unwrap();

        let mut cursor = input
            .lazy_cursor(meta, doc_freq, IndexOptions::Docs, false)
            .unwrap();
        let mut lazy_docs = Vec::new();
        loop {
            let d = cursor.next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            lazy_docs.push(d);
        }
        assert_eq!(lazy_docs, eager.docs);
    }

    #[test]
    fn lazy_cursor_advance_skips_whole_corrupted_level1_span_without_decoding_it() {
        // The level-1 analogue of
        // `lazy_cursor_advance_skips_corrupted_earlier_block_without_decoding_it`:
        // an entire 32-block level-1 span is corrupt (its first block has a
        // valid level-0 frame but a zero-bit bit-set body, the rest is
        // garbage). `advance(target)` to a doc in the *tail* past the whole
        // span must succeed -- proving `skip_level1_to` jumps straight to the
        // level-1 entry's `doc_end_fp` without ever reading a single level-0
        // block header in the span. A control `advance()` into the span still
        // surfaces the corruption, confirming the skip wasn't luck.
        let id = [22u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        // The span: block 0 has a valid frame (docDelta=256 -> last doc 255)
        // but a corrupt zero-bit bit-set body; blocks 1..31 are pure garbage
        // neither tested path ever reads (the skip jumps over them via
        // doc_end_fp; the control errors on block 0 before reaching them).
        let mut span = Vec::new();
        let mut body0 = Vec::new();
        body0.write_byte((-4i8) as u8); // 4 longs = 256 bits, none set -> corrupt
        body0.extend_from_slice(&[0u8; 32]);
        span.write_vlong(body0.len() as i64); // level0NumBytes
        span.write_i16(BLOCK_SIZE as i16); // docDelta = 256
        span.write_i16(body0.len() as i16); // blockLength
        span.write_bytes(&body0);
        span.extend_from_slice(&[0xFFu8; 64]); // garbage stand-in for blocks 1..31

        // Level-1 entry: doc_delta 8192 -> span last doc 8191; doc_end_fp
        // lands exactly at the tail (span.len() bytes past the vLong).
        write_level1_entry_docs(&mut doc, LEVEL1_NUM_DOCS, &span);
        doc.extend_from_slice(&span);
        // Valid 8-doc tail (docs 8192..8199), chained from prevDocID 8191.
        write_group_vints(&mut doc, &[1; 8]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let doc_freq = LEVEL1_NUM_DOCS + 8;

        let mut cursor = input
            .lazy_cursor(meta, doc_freq, IndexOptions::Docs, false)
            .unwrap();
        // Target 8195 is in the tail, past the whole corrupt span.
        assert_eq!(cursor.advance(8195).unwrap(), 8195);
        assert_eq!(cursor.freq(), Some(1));

        // Control: a target inside the span forces decoding block 0's corrupt
        // body, which must surface the corruption error.
        let mut cursor2 = input
            .lazy_cursor(meta, doc_freq, IndexOptions::Docs, false)
            .unwrap();
        let err = cursor2.advance(100).unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn lazy_cursor_sequential_next_doc_matches_read_postings() {
        // Two full blocks (docs 0..256, 256..512) plus a 3-doc tail: proves
        // the lazy per-block decode-on-demand path produces byte-identical
        // results to the eager whole-term `read_postings` across both the
        // full-block/full-block and full-block/tail-block boundaries.
        let id = [22u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_full_block(&mut doc, true, 3);
        write_full_block(&mut doc, true, 4);
        write_group_vints(&mut doc, &[5 << 1, 1 << 1, 2 << 1]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let doc_freq = 2 * BLOCK_SIZE + 3;
        let eager = input
            .read_postings(meta, doc_freq, IndexOptions::DocsAndFreqs, false)
            .unwrap();

        let mut cursor = input
            .lazy_cursor(meta, doc_freq, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        let mut lazy_docs = Vec::new();
        let mut lazy_freqs = Vec::new();
        loop {
            let d = cursor.next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            lazy_docs.push(d);
            lazy_freqs.push(cursor.freq().unwrap());
        }
        assert_eq!(lazy_docs, eager.docs);
        assert_eq!(lazy_freqs, eager.freqs);
    }

    #[test]
    fn lazy_cursor_advance_skips_corrupted_earlier_block_without_decoding_it() {
        // Block 0 (docs 0..256) is deliberately corrupt: a dense bit-set
        // encoding (`bitsPerValue == -4`) with zero bits actually set, which
        // `decode_full_block_body` rejects with `Error::Store(Corrupted)` --
        // see `read_full_block_bitset_encoding_rejects_too_few_set_bits`.
        // Block 1 (docs 256..511) is a normal, valid all-consecutive block.
        // `advance(300)` lands in block 1: if the cursor decoded block 0's
        // body along the way (as the eager `read_postings` path always
        // does), this test would fail with a decode error instead of
        // returning doc 300 -- proving the skip genuinely bypasses
        // `ForUtil`/`PForUtil` decode for a block it can prove is entirely
        // behind the target, not just "returns the right answer by luck".
        let id = [23u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        // Corrupt block 0: IndexOptions::Docs (no freq field), docDelta=256
        // (claims last doc 255, consistent with a real all-256-bit block),
        // but the body's bit-set has no bits set at all.
        let mut corrupt_body = Vec::new();
        corrupt_body.write_byte((-4i8) as u8); // 4 longs = 256 bits
        corrupt_body.extend_from_slice(&[0u8; 32]); // none set -- corrupt
        doc.write_vlong(corrupt_body.len() as i64); // level0NumBytes (unused)
        doc.write_i16(BLOCK_SIZE as i16); // docDelta = 256
        doc.write_i16(corrupt_body.len() as i16); // blockLength
        doc.write_bytes(&corrupt_body);

        // Block 1: valid, all-consecutive, no freq field (Docs mode).
        write_full_block(&mut doc, false, 0);

        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let mut cursor = input
            .lazy_cursor(meta, 2 * BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap();

        let result = cursor.advance(300).unwrap();
        assert_eq!(result, 300);
        assert_eq!(cursor.freq(), Some(1));

        // Sanity check the other direction: actually decoding block 0 (a
        // target inside it) must surface the corruption error, confirming
        // the earlier success was really a skip and not an accidental pass.
        let mut cursor2 = input
            .lazy_cursor(meta, 2 * BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap();
        let err = cursor2.advance(10).unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn lazy_cursor_advance_to_doc_before_current_is_a_no_op() {
        let id = [24u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_group_vints(&mut doc, &[(3 << 1) | 1, (3 << 1) | 1]); // docs 2, 5 (deltas 3,3 from prev=-1), freq=1 each
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let mut cursor = input
            .lazy_cursor(meta, 2, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(cursor.advance(5).unwrap(), 5);
        // Advancing "backward" to a target at/before the current doc is a
        // documented no-op, matching `PostingsCursor::advance`'s contract.
        assert_eq!(cursor.advance(3).unwrap(), 5);
        assert_eq!(cursor.advance(5).unwrap(), 5);
    }

    #[test]
    fn lazy_cursor_advance_past_last_doc_returns_no_more_docs() {
        let id = [25u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_group_vints(&mut doc, &[(3 << 1) | 1, (3 << 1) | 1]); // docs 2, 5 (deltas 3,3 from prev=-1)
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let mut cursor = input
            .lazy_cursor(meta, 2, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(cursor.advance(100).unwrap(), NO_MORE_DOCS);
        assert_eq!(cursor.freq(), None);
        // Once exhausted, further `next_doc()`/`advance()` calls stay
        // `NO_MORE_DOCS` rather than erroring or wrapping around.
        assert_eq!(cursor.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(cursor.advance(1).unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn lazy_cursor_next_doc_from_start_walks_in_order() {
        let id = [26u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_group_vints(&mut doc, &[(3 << 1) | 1, (3 << 1) | 1]); // docs 2, 5 (deltas 3,3 from prev=-1)
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let mut cursor = input
            .lazy_cursor(meta, 2, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(cursor.doc_id(), -1);
        assert_eq!(cursor.next_doc().unwrap(), 2);
        assert_eq!(cursor.next_doc().unwrap(), 5);
        assert_eq!(cursor.next_doc().unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn decode_term_metadata_absolute_then_delta_docstart() {
        let mut bytes = Vec::new();
        // absolute: docStartFP delta=10 (l = 10<<1 = 20), docFreq>1 so no singleton vint
        bytes.write_vlong(20);
        // second term in same block: docStartFP delta=5 (l = 5<<1 = 10)
        bytes.write_vlong(10);
        let mut r = SliceInput::new(&bytes);

        let first = decode_term_metadata(
            &mut r,
            2,
            true,
            TermMetadata::EMPTY,
            IndexOptions::DocsAndFreqs,
            false,
            2,
        )
        .unwrap();
        assert_eq!(first.doc_start_fp, 10);
        assert_eq!(first.singleton_doc_id, -1);

        let second = decode_term_metadata(
            &mut r,
            2,
            false,
            first,
            IndexOptions::DocsAndFreqs,
            false,
            2,
        )
        .unwrap();
        assert_eq!(second.doc_start_fp, 15);
    }

    #[test]
    fn decode_term_metadata_absolute_resets_fps_even_with_stale_prev() {
        // `Lucene104PostingsReader.decodeTerm` zeroes docStartFP/posStartFP/
        // payStartFP before applying deltas whenever `absolute` is set --
        // regardless of what `prev` carries in. A caller passing a non-empty
        // `prev` alongside `absolute=true` (e.g. a future multi-block
        // BlockTree reader that doesn't reset `prev_meta` per block) must
        // still get FPs computed as deltas-from-zero, not
        // deltas-from-`prev`.
        let mut bytes = Vec::new();
        // docStartFP delta=7 (l = 7<<1 = 14, docFreq>1 so no singleton vint).
        bytes.write_vlong(14);
        // posStartFP delta=3.
        bytes.write_vlong(3);
        let mut r = SliceInput::new(&bytes);

        let stale_prev = TermMetadata {
            doc_start_fp: 1000,
            pos_start_fp: 2000,
            pay_start_fp: 3000,
            singleton_doc_id: -1,
            last_pos_block_offset: -1,
        };
        let decoded = decode_term_metadata(
            &mut r,
            2,
            true,
            stale_prev,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
            1,
        )
        .unwrap();
        assert_eq!(decoded.doc_start_fp, 7, "should be 0 + 7, not 1000 + 7");
        assert_eq!(decoded.pos_start_fp, 3, "should be 0 + 3, not 2000 + 3");
    }

    #[test]
    fn decode_term_metadata_singleton_absolute_then_zigzag_delta() {
        let mut bytes = Vec::new();
        // absolute singleton: docStartFP delta=0 (l=0), then raw vint singletonDocID=7
        bytes.write_vlong(0);
        bytes.write_vint(7);
        // next term: singleton delta of +3 via zigzag, flag bit set
        let zz = lucene_util::zigzag::encode(3);
        bytes.write_vlong(((zz as i64) << 1) | 1);
        let mut r = SliceInput::new(&bytes);

        let first = decode_term_metadata(
            &mut r,
            1,
            true,
            TermMetadata::EMPTY,
            IndexOptions::DocsAndFreqs,
            false,
            2,
        )
        .unwrap();
        assert_eq!(first.singleton_doc_id, 7);

        let second = decode_term_metadata(
            &mut r,
            1,
            false,
            first,
            IndexOptions::DocsAndFreqs,
            false,
            2,
        )
        .unwrap();
        assert_eq!(second.singleton_doc_id, 10);
        assert_eq!(second.doc_start_fp, first.doc_start_fp);
    }

    #[test]
    fn decode_term_metadata_rejects_absolute_singleton_delta() {
        let mut bytes = Vec::new();
        bytes.write_vlong(1); // flag bit set on what must be an absolute decode
        let mut r = SliceInput::new(&bytes);
        let err = decode_term_metadata(
            &mut r,
            1,
            true,
            TermMetadata::EMPTY,
            IndexOptions::DocsAndFreqs,
            false,
            2,
        )
        .unwrap_err();
        assert!(matches!(err, Error::AbsoluteSingletonDelta));
    }

    #[test]
    fn decode_term_metadata_rejects_delta_with_no_previous_singleton() {
        let mut bytes = Vec::new();
        bytes.write_vlong(1); // flag bit set, non-absolute
        let mut r = SliceInput::new(&bytes);
        let err = decode_term_metadata(
            &mut r,
            1,
            false,
            TermMetadata::EMPTY,
            IndexOptions::DocsAndFreqs,
            false,
            2,
        )
        .unwrap_err();
        assert!(matches!(err, Error::NoPreviousSingleton));
    }

    #[test]
    fn decode_term_metadata_with_positions_reads_pos_and_pay_fps() {
        // IndexOptions::DocsAndFreqsAndPositions, no offsets, has_payloads
        // true: posStartFP delta, then payStartFP delta (payloads alone
        // trigger it, per `Lucene104PostingsReader.java:239-242`), then
        // (totalTermFreq > BLOCK_SIZE) a lastPosBlockOffset vlong.
        let mut bytes = Vec::new();
        bytes.write_vlong(4); // docStartFP delta = 2 (l=4, bit0 clear -> 4>>1=2)
        bytes.write_vlong(7); // posStartFP delta
        bytes.write_vlong(11); // payStartFP delta (has_payloads=true)
        bytes.write_vlong(300); // lastPosBlockOffset (totalTermFreq=BLOCK_SIZE+1 > BLOCK_SIZE)
        let mut r = SliceInput::new(&bytes);

        let meta = decode_term_metadata(
            &mut r,
            5,
            true,
            TermMetadata::EMPTY,
            IndexOptions::DocsAndFreqsAndPositions,
            true,
            BLOCK_SIZE as i64 + 1,
        )
        .unwrap();
        assert_eq!(meta.doc_start_fp, 2);
        assert_eq!(meta.pos_start_fp, 7);
        assert_eq!(meta.pay_start_fp, 11);
        assert_eq!(meta.last_pos_block_offset, 300);
    }

    #[test]
    fn decode_term_metadata_with_positions_no_last_pos_block_offset_when_small() {
        // totalTermFreq <= BLOCK_SIZE: no lastPosBlockOffset vlong on the wire.
        let mut bytes = Vec::new();
        bytes.write_vlong(0); // docStartFP delta = 0
        bytes.write_vlong(3); // posStartFP delta
        let mut r = SliceInput::new(&bytes);

        let meta = decode_term_metadata(
            &mut r,
            2,
            true,
            TermMetadata::EMPTY,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
            BLOCK_SIZE as i64,
        )
        .unwrap();
        assert_eq!(meta.pos_start_fp, 3);
        assert_eq!(meta.pay_start_fp, 0);
        assert_eq!(meta.last_pos_block_offset, -1);
        // No bytes left to read (would error if the writer had emitted a
        // pay/lastPosBlockOffset field this decode didn't consume).
        assert!(r.read_vlong().is_err());
    }

    #[test]
    fn decode_term_metadata_offsets_without_payloads_still_reads_pay_fp() {
        // DocsAndFreqsAndPositionsAndOffsets subsumes offsets, so payStartFP
        // is written even when has_payloads=false.
        let mut bytes = Vec::new();
        bytes.write_vlong(0);
        bytes.write_vlong(5); // posStartFP delta
        bytes.write_vlong(9); // payStartFP delta (offsets, not payloads)
        let mut r = SliceInput::new(&bytes);

        let meta = decode_term_metadata(
            &mut r,
            2,
            true,
            TermMetadata::EMPTY,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            false,
            BLOCK_SIZE as i64,
        )
        .unwrap();
        assert_eq!(meta.pos_start_fp, 5);
        assert_eq!(meta.pay_start_fp, 9);
    }

    fn pos_header_and_footer(id: &[u8; ID_LENGTH]) -> (Vec<u8>, Vec<u8>) {
        let mut before = Vec::new();
        codec_util::write_index_header(&mut before, POS_CODEC, VERSION_CURRENT, id, "");
        let mut after = Vec::new();
        codec_util::write_footer(&mut after);
        (before, after)
    }

    fn pay_header_and_footer(id: &[u8; ID_LENGTH]) -> (Vec<u8>, Vec<u8>) {
        let mut before = Vec::new();
        codec_util::write_index_header(&mut before, PAY_CODEC, VERSION_CURRENT, id, "");
        let mut after = Vec::new();
        codec_util::write_footer(&mut after);
        (before, after)
    }

    #[test]
    fn read_positions_single_position_no_offsets_no_payloads() {
        // One doc, one occurrence: total_term_freq=1 < BLOCK_SIZE, so it's
        // entirely the vint tail (`refillLastPositionBlock`), no PForUtil
        // blocks at all. code = posDelta (no payload bit-packing since
        // has_payloads=false).
        let id = [20u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        pos.write_vint(42); // posDelta = 42
        pos.extend_from_slice(&pos_footer);

        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            ..TermMetadata::EMPTY
        };
        let result = read_positions(
            &pos_in,
            None,
            meta,
            &[1],
            1,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), 1);
        assert_eq!(result[0][0].position, 42);
        assert_eq!(result[0][0].start_offset, -1);
        assert_eq!(result[0][0].end_offset, -1);
        assert!(result[0][0].payload.is_empty());
    }

    #[test]
    fn read_positions_multiple_positions_with_payload_on_some_occurrences() {
        // Two docs sharing one term: doc0 has 2 occurrences (positions 1
        // and 3, first with a payload, second without -- payload length
        // changes 2 -> 0, both written explicitly since the length changed
        // each time); doc1 has 1 occurrence with no payload, reusing the
        // last (0) length so no length vint is written for it.
        let id = [21u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;

        // doc0, occurrence 0: posDelta=1, payload length changes to 2 (bit set).
        pos.write_vint((1 << 1) | 1);
        pos.write_vint(2);
        pos.write_bytes(&[0xAA, 0xBB]);
        // doc0, occurrence 1: posDelta=2 (position 1+2=3), payload length changes to 0.
        pos.write_vint((2 << 1) | 1);
        pos.write_vint(0);
        // doc1, occurrence 0: posDelta=5 (fresh accumulator, position=5),
        // payload length unchanged (still 0), so bit is clear.
        pos.write_vint(5 << 1);

        pos.extend_from_slice(&pos_footer);

        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            ..TermMetadata::EMPTY
        };
        let result = read_positions(
            &pos_in,
            None,
            meta,
            &[2, 1],
            3,
            IndexOptions::DocsAndFreqsAndPositions,
            true,
        )
        .unwrap();
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].len(), 2);
        assert_eq!(result[0][0].position, 1);
        assert_eq!(result[0][0].payload, vec![0xAA, 0xBB]);
        assert_eq!(result[0][1].position, 3);
        assert!(result[0][1].payload.is_empty());
        assert_eq!(result[1].len(), 1);
        assert_eq!(result[1][0].position, 5);
        assert!(result[1][0].payload.is_empty());
    }

    #[test]
    fn read_positions_with_offsets() {
        // One doc, two occurrences, offsets but no payloads: payIn carries
        // the offset start-delta/length vint pairs interleaved with .pos'
        // posDelta vints (no payload bit-packing on the posDelta code since
        // has_payloads=false).
        let id = [22u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;

        // occurrence 0: posDelta=1 (position=1), offset [0,3) (length 3, bit set).
        pos.write_vint(1);
        pos.write_vint(1); // offset start delta = 0, bit set (length changes)
        pos.write_vint(3);
        // occurrence 1: posDelta=1 (position=2), offset [4,7) (start delta=4
        // from lastStartOffset=0, length still 3 so bit clear, reused).
        pos.write_vint(1);
        pos.write_vint(4 << 1);

        pos.extend_from_slice(&pos_footer);

        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            ..TermMetadata::EMPTY
        };
        let result = read_positions(
            &pos_in,
            None,
            meta,
            &[2],
            2,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            false,
        )
        .unwrap();
        assert_eq!(result[0][0].position, 1);
        assert_eq!(result[0][0].start_offset, 0);
        assert_eq!(result[0][0].end_offset, 3);
        assert_eq!(result[0][1].position, 2);
        assert_eq!(result[0][1].start_offset, 4);
        assert_eq!(result[0][1].end_offset, 7);
    }

    #[test]
    fn read_positions_rejects_offsets_without_pay_input() {
        // total_term_freq spans one full block (BLOCK_SIZE), which is what
        // actually requires `.pay` for a field with offsets -- a term whose
        // whole total_term_freq fit in the vint tail wouldn't need it (see
        // `read_positions`'s own doc comment), so this deliberately uses a
        // full-block-sized total to exercise the real requirement.
        let id = [23u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        pos.write_byte(0);
        pos.write_vint(1);
        pos.extend_from_slice(&pos_footer);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let err = read_positions(
            &pos_in,
            None,
            TermMetadata::EMPTY,
            &[BLOCK_SIZE],
            BLOCK_SIZE as i64,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn read_positions_rejects_freqs_summing_past_total_term_freq() {
        // `freqs` (decoded independently from `.doc`) claiming more
        // occurrences than `total_term_freq` (decoded from the term
        // dictionary) is corrupted input -- must be a decode error, not an
        // out-of-bounds panic when the re-chop loop runs past the end of the
        // flat `pos_deltas` array.
        let id = [27u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        pos.write_vint(1); // one posDelta -- total_term_freq=1 worth of data
        pos.extend_from_slice(&pos_footer);

        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            ..TermMetadata::EMPTY
        };
        // freqs claims 2 occurrences for a single doc, but total_term_freq
        // (and thus the decoded pos_deltas array) only has 1.
        let err = read_positions(
            &pos_in,
            None,
            meta,
            &[2],
            1,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn read_positions_rejects_freqs_summing_below_total_term_freq() {
        // The reverse mismatch: freqs under-claim occurrences relative to
        // total_term_freq. Must also be a decode error, not a silent partial
        // decode that drops leftover positions.
        let id = [28u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        pos.write_vint(1);
        pos.write_vint(1);
        pos.extend_from_slice(&pos_footer);

        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            ..TermMetadata::EMPTY
        };
        let err = read_positions(
            &pos_in,
            None,
            meta,
            &[1],
            2,
            IndexOptions::DocsAndFreqsAndPositions,
            false,
        )
        .unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn read_positions_rejects_non_position_index_options() {
        let id = [24u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        pos.extend_from_slice(&pos_footer);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let err = read_positions(
            &pos_in,
            None,
            TermMetadata::EMPTY,
            &[1],
            1,
            IndexOptions::DocsAndFreqs,
            false,
        )
        .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn read_positions_exactly_one_full_block_boundary() {
        // total_term_freq == BLOCK_SIZE (256) exactly: one full PForUtil
        // block, no vint tail at all -- exercises `num_full_blocks == 1,
        // tail_count == 0`. All 256 posDeltas equal 1 (positions 1..=256,
        // one doc's occurrences), payload lengths all equal 0 (still needs a
        // PForUtil block + a zero-length `numBytes` vint on `.pay`, matching
        // what the real writer emits even for an all-empty-payload block).
        let id = [25u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        pos.write_byte(0); // PForUtil token: bitsPerValue=0, numExceptions=0
        pos.write_vint(1); // fill value: posDelta=1 for all 256 occurrences
        pos.extend_from_slice(&pos_footer);

        let (mut pay, pay_footer) = pay_header_and_footer(&id);
        let pay_start_fp = pay.len() as u64;
        pay.write_byte(0); // payloadLengthBuffer PForUtil token: all-equal
        pay.write_vint(0); // fill value: length 0 for all 256
        pay.write_vint(0); // numBytes = 0 (no payload bytes follow)
        pay.extend_from_slice(&pay_footer);

        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let pay_in = PayInput::open(&pay, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            pay_start_fp,
            ..TermMetadata::EMPTY
        };
        let result = read_positions(
            &pos_in,
            Some(&pay_in),
            meta,
            &[BLOCK_SIZE],
            BLOCK_SIZE as i64,
            IndexOptions::DocsAndFreqsAndPositions,
            true,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), BLOCK_SIZE as usize);
        let expected: Vec<i32> = (1..=BLOCK_SIZE).collect();
        assert_eq!(
            result[0].iter().map(|p| p.position).collect::<Vec<_>>(),
            expected
        );
        assert!(result[0].iter().all(|p| p.payload.is_empty()));
    }

    #[test]
    fn read_positions_full_block_with_offsets_no_payloads() {
        // Same full-PForUtil-block shape as
        // `read_positions_exactly_one_full_block_boundary`, but for a field
        // with offsets and no payloads: `.pay` carries only the two
        // offset-start-delta/offset-length PForUtil blocks (no payload-length
        // block, no payload bytes), matching `read_positions`'s `has_offsets`
        // branch. This was previously untested at every level (fixture and
        // unit) -- a mismatch in the payload/offset `.pay` cursor ordering
        // would silently produce wrong offsets undetected.
        let id = [26u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        pos.write_byte(0); // PForUtil token: bitsPerValue=0, numExceptions=0
        pos.write_vint(1); // fill value: posDelta=1 for all 256 occurrences
        pos.extend_from_slice(&pos_footer);

        let (mut pay, pay_footer) = pay_header_and_footer(&id);
        let pay_start_fp = pay.len() as u64;
        pay.write_byte(0); // offsetStartDeltaBuffer PForUtil token: all-equal
        pay.write_vint(2); // fill value: start delta = 2 for all 256
        pay.write_byte(0); // offsetLengthBuffer PForUtil token: all-equal
        pay.write_vint(5); // fill value: length = 5 for all 256
        pay.extend_from_slice(&pay_footer);

        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let pay_in = PayInput::open(&pay, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            pay_start_fp,
            ..TermMetadata::EMPTY
        };
        let result = read_positions(
            &pos_in,
            Some(&pay_in),
            meta,
            &[BLOCK_SIZE],
            BLOCK_SIZE as i64,
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            false,
        )
        .unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].len(), BLOCK_SIZE as usize);
        for (i, p) in result[0].iter().enumerate() {
            let expected_start = (i as i32 + 1) * 2;
            assert_eq!(p.start_offset, expected_start, "occurrence {i}");
            assert_eq!(p.end_offset, expected_start + 5, "occurrence {i}");
            assert!(p.payload.is_empty());
        }
    }
}
