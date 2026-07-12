//! Port of `org.apache.lucene.codecs.lucene104.Lucene104PostingsReader`'s
//! `.doc`/`.pos`/`.pay` file decode — read-only, scoped to **sequential
//! decode** (a full forward scan, i.e. `nextDoc()`/`nextPosition()`-equivalent;
//! no skip-ahead/`advance()`) of **`IndexOptions.DOCS`/`DOCS_AND_FREQS`/
//! `DOCS_AND_FREQS_AND_POSITIONS`/`DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`**
//! (incl. payloads) for any `docFreq < LEVEL1_NUM_DOCS` (32 * `BLOCK_SIZE` =
//! 8192 — see "Deferred" below). See "Positions/offsets/payloads
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
//! parsed but unused, see "Deferred"), `docDelta` (`writeVInt15`-encoded,
//! skip-pointer — parsed but unused), `blockLength` (`writeVLong15`-encoded,
//! skip-pointer — parsed but unused); then, only when the field has freqs, an
//! impacts byte-length (vlong) and that many impact bytes (competitive-scoring
//! metadata — parsed-and-discarded, see "Deferred"); then a 1-byte
//! `bitsPerValue` token selecting how the block's 256 doc deltas are packed
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
//! ## Deferred (all rejected with [`Error::Unsupported`])
//!
//! - **Skip-ahead (`advance()`)**: the level-0/level-1 skip pointers
//!   (`level0NumBytes`/`blockLength`/level-1 headers) are parsed for wire-order
//!   correctness but never used to jump forward; every block is decoded in
//!   full. Correct but not fast for a searcher that only wants a subrange —
//!   fine for a full-scan merge/`nextDoc()` consumer. Follow-up work.
//! - **`docFreq >= LEVEL1_NUM_DOCS`** (8192): level-1 skip entries start
//!   appearing inline in the `.doc` stream every 32 full blocks, which this
//!   reader does not parse (this also blocks positions, since `positions()`
//!   goes through the same `docFreq` gate via `postings()`).
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

        // This slice only supports `docFreq < LEVEL1_NUM_DOCS`
        // (`Lucene104PostingsFormat.LEVEL1_NUM_DOCS` = 32 * BLOCK_SIZE =
        // 8192): the reader only ever seeks straight to `docStartFP` and
        // walks level-0 blocks sequentially (no level-1 skip entries appear
        // on the wire below that threshold, see the module doc's "Deferred"
        // section) — this covers every term this port's fixtures or a
        // realistic single-segment merge would produce short of an
        // enormous posting list.
        if doc_freq >= LEVEL1_NUM_DOCS {
            return Err(Error::Unsupported(
                "docFreq >= LEVEL1_NUM_DOCS: level-1 skip data not supported in this slice",
            ));
        }

        let mut prev_doc_id: i32 = -1;
        let mut doc_count_left = doc_freq;
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

/// One full 256-doc block (`BlockPostingsEnum.refillFullBlock` plus the
/// level-0 skip header that precedes every full block on the wire,
/// `Lucene104PostingsWriter.flushDocBlock`'s `else` branch —
/// `docBufferUpto == BLOCK_SIZE`).
///
/// Scoped to **sequential decode only**: the level-0 header's
/// `level0NumBytes`/`docDelta`/`blockLength`/impacts fields exist so
/// `advance()` can skip whole blocks without decoding them; this reader
/// parses every field in wire order instead of seeking with them (no bytes
/// are skipped blindly), which is sufficient for a full forward scan
/// (`nextDoc()`-equivalent) but not for skip-ahead — see the module doc's
/// "Deferred" section and `docs/parity.md`.
fn read_full_block(
    r: &mut SliceInput,
    prev_doc_id: i32,
    index_has_freq: bool,
    index_has_pos: bool,
    index_has_offsets_or_payloads: bool,
) -> Result<([i32; BLOCK_SIZE as usize], [i32; BLOCK_SIZE as usize])> {
    let _level0_num_bytes = r.read_vlong()?;
    let _doc_delta = read_vint15(r)?;
    let _block_length = read_vlong15(r)?;
    if index_has_freq {
        // Impacts byte-length is a plain vint here (`doMoveToNextLevel0Block`,
        // `Lucene104PostingsReader.java:746`), unlike level-1's vlong-prefixed
        // `numSkipBytes` -- confirmed against the reader source rather than
        // assumed from the tail-block/level-1 shape.
        let impacts_len = r.read_vint()? as usize;
        r.skip(impacts_len)?;

        // Level-0 pos/pay skip data (`Lucene104PostingsReader.java:754-761`):
        // parsed for wire-order correctness (this reader never skips ahead
        // with it, see the module doc) only when the field indexes
        // positions.
        if index_has_pos {
            let _pos_end_fp_delta = r.read_vlong()?;
            let _pos_buffer_upto = r.read_byte()?;
            if index_has_offsets_or_payloads {
                let _pay_end_fp_delta = r.read_vlong()?;
                let _pay_buffer_upto = r.read_vint()?;
            }
        }
    }

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

    #[test]
    fn read_postings_rejects_level1_doc_freq() {
        // docFreq >= LEVEL1_NUM_DOCS (8192): out of scope, see the module doc.
        let id = [5u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        doc.extend_from_slice(&footer);
        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp: 0,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let err = input
            .read_postings(meta, LEVEL1_NUM_DOCS, IndexOptions::DocsAndFreqs, false)
            .unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
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

        out.write_vlong(body.len() as i64); // level0NumBytes (unused by the reader)
        out.write_i16(1); // docDelta via writeVInt15 (unused by the reader)
        out.write_i16(body.len() as i16); // blockLength via writeVLong15 (unused)
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
