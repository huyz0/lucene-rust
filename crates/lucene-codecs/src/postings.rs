//! Port of `org.apache.lucene.codecs.lucene104.Lucene104PostingsReader`'s
//! `.doc`/`.pos`/`.pay` file decode — read-only, scoped to
//! **`IndexOptions.DOCS`/`DOCS_AND_FREQS`/`DOCS_AND_FREQS_AND_POSITIONS`/
//! `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`/`DOCS_AND_CUSTOM_FREQS`** (incl.
//! payloads) at any
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
//! `ForUtil`/`PForUtil`-encoded ([`read_full_block_header`] +
//! [`decode_full_block_body`], ported in
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
//! eager path), `blockLength` (`writeVLong15`-encoded — the
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
//! [`read_full_block_header`] same as the rest of that header). The actual
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
//! ## `IndexOptions::DocsAndCustomFreqs`
//!
//! Wire-identical to `IndexOptions::DocsAndFreqs`: real Lucene's
//! `Lucene104PostingsReader`/`Lucene104PostingsWriter` derive `indexHasFreq`/
//! `writeFreqs` from `IndexOptions.subsumes(DOCS_AND_FREQS)`
//! (`IndexOptions.java`'s `subsumes` override), which is `true` for both
//! variants — they differ only in how the caller *interprets* the freq value
//! (a term count vs. an opaque per-doc "custom" score, e.g. for similarity
//! implementations that want an arbitrary integer instead of a real
//! occurrence count), never in how it's encoded or decoded. So this decoder
//! treats it exactly like `DocsAndFreqs` (same `index_has_freq` derivation,
//! same false `subsumes_positions()`/`subsumes_offsets()`) with no separate
//! code path.
//!
//! ## Deferred (all rejected with [`Error::Unsupported`])
//!
//! - Impacts (`ImpactsEnum`, `CompetitiveImpactAccumulator`, competitive-scoring
//!   metadata) — see `docs/parity.md`.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::field_infos::IndexOptions;
use crate::for_util::{self, ForUtil};

/// `Lucene104PostingsFormat.DOC_CODEC`.
pub(crate) const DOC_CODEC: &str = "Lucene104PostingsWriterDoc";
/// `Lucene104PostingsFormat.META_CODEC` -- the `.psm` metadata file's codec.
pub(crate) const META_CODEC: &str = "Lucene104PostingsWriterMeta";
const VERSION_START: i32 = 0;
pub(crate) const VERSION_CURRENT: i32 = 0;
/// `ForUtil.BLOCK_SIZE` (== `Lucene104PostingsFormat.BLOCK_SIZE`).
pub const BLOCK_SIZE: i32 = 256;
/// `Lucene104PostingsFormat.LEVEL1_NUM_DOCS` (`LEVEL1_FACTOR(=32) * BLOCK_SIZE`):
/// below this many docs, a term's `.doc` bytes contain only level-0 skip
/// headers (no level-1 entries) — see the module doc's "Deferred" section.
pub(crate) const LEVEL1_NUM_DOCS: i32 = 32 * BLOCK_SIZE;
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
        // Zigzag-decoded off disk, so the addition itself can overflow `i64`
        // on a corrupt `.tim`: wrap rather than panic in a debug build, the
        // same rule the `doc_start_fp`/`pos_start_fp` accumulators above
        // already follow.
        let delta = lucene_util::zigzag::decode(l >> 1);
        (
            prev.doc_start_fp,
            (prev.singleton_doc_id as i64).wrapping_add(delta) as i32,
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

/// One term's decoded `(docID, freq)` pairs, in ascending doc-ID order, plus
/// the competitive impacts `DocInput::read_postings` captured in passing while
/// it was already decoding each level-0 header and level-1 entry — no second
/// decode pass, just retaining what [`LazyDocsCursor`] retains too instead of
/// discarding it (see this struct's `level0_impacts`/`level1_impacts` fields
/// and [`PostingsCursor::level0_impacts`]/[`PostingsCursor::level1_impacts`]).
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Postings {
    pub docs: Vec<i32>,
    pub freqs: Vec<i32>,
    /// One entry per full level-0 block actually present on the wire, in
    /// ascending order: `(last_doc_id_in_block, impacts_for_that_block)`.
    /// Empty when the term has no full blocks (`docFreq < BLOCK_SIZE`) or the
    /// field has no freqs. Since blocks are decoded in ascending, contiguous
    /// doc-ID order with no gaps, [`PostingsCursor::level0_impacts`] finds the
    /// block covering a given doc ID with a single binary search over the
    /// `last_doc_id_in_block` column (the first entry whose bound is `>=` the
    /// target doc ID) rather than needing an explicit per-doc index.
    pub level0_impacts: Vec<(i32, Impacts)>,
    /// One entry per level-1 skip entry actually present on the wire (only
    /// non-empty once `docFreq >= LEVEL1_NUM_DOCS`), in ascending order:
    /// `(last_doc_id_in_span, impacts_for_that_span)`. Empty below
    /// `LEVEL1_NUM_DOCS` docs or for a field with no freqs. Looked up the same
    /// way as `level0_impacts`.
    pub level1_impacts: Vec<(i32, Impacts)>,
}

// Tradeoff, not a bug: `level0_impacts`/`level1_impacts` are populated on
// every `read_postings` call for a field with freqs, one small `Vec`
// allocation per full block/span, even for callers that never call
// `PostingsCursor::level0_impacts`/`level1_impacts`. This mirrors
// `docs`/`freqs` themselves already being fully eager (the whole point of
// `read_postings` vs. `DocInput::lazy_cursor`), and the added cost is
// proportional to `docFreq / BLOCK_SIZE` (one entry per ~256 docs), not
// per-doc -- negligible next to the `docs`/`freqs` `Vec<i32>`s already
// dominating this struct's size. A caller that wants to avoid this
// entirely already has `DocInput::lazy_cursor`/`LazyDocsCursor` as the
// decode-on-demand alternative.

/// Binary-searches a `(last_doc_id_covered, impacts)` column (as stored in
/// [`Postings::level0_impacts`]/[`Postings::level1_impacts`]) for the entry
/// covering `doc_id`, given the entries are in ascending, contiguous,
/// non-overlapping doc-ID order (true for both columns: level-0/level-1
/// blocks/spans are decoded back-to-back with no gaps in doc-ID coverage).
/// Returns an empty slice if `doc_id` falls in a trailing region no entry
/// covers (the tail block, or — for `level1_impacts` — any full blocks/tail
/// past the last level-1 span), matching [`LazyDocsCursor`]'s contract of
/// reporting no impacts there.
fn find_impacts(column: &[(i32, Impacts)], doc_id: i32) -> &[Impact] {
    let pos = column.partition_point(|&(last, _)| last < doc_id);
    column
        .get(pos)
        .map(|(_, impacts)| impacts.as_slice())
        .unwrap_or(&[])
}

/// A single competitive `(freq, norm)` pair, port of `Impact`
/// (`org.apache.lucene.codecs.Impact`): "per-document scoring factors" used by
/// `ImpactsEnum`/`CompetitiveImpactAccumulator` for WAND/MAXSCORE query-time
/// pruning. `norm` is whatever `NumericDocValues.longValue()` the field's
/// `.nvd` produces at write time — for the default similarity this is a
/// single signed byte widened to `long` (`Similarity.computeNorm`'s common
/// case), but the encoding here (see [`decode_impacts`]) is norm-width-
/// agnostic, so this type stores the full `i64` rather than assuming a byte.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Impact {
    /// Term frequency of the term in the document (or the max across a block/
    /// span of documents, for the level-0/level-1 skip metadata this is
    /// decoded from).
    pub freq: i32,
    /// Norm factor of the document (or the min-competitive norm across a
    /// block/span, paired with `freq` the same way).
    pub norm: i64,
}

/// A block's or span's list of competitive `(freq, norm)` pairs, ordered by
/// strictly increasing `freq` and strictly increasing (unsigned) `norm` —
/// the same invariant `CompetitiveImpactAccumulator.getCompetitiveFreqNormPairs`
/// guarantees on the write side (`CompetitiveImpactAccumulator.java:100-119`):
/// each successive entry has both a higher freq *and* a higher norm than the
/// previous one, so a scorer can stop scanning the list as soon as it finds
/// an entry whose norm is competitive enough, using that entry's freq as an
/// upper bound.
pub type Impacts = Vec<Impact>;

/// Decodes one level-0 or level-1 impacts byte run (`Lucene104PostingsReader
/// .readImpacts`, `Lucene104PostingsReader.java:1447-1467`): a flat sequence
/// of `(freqDelta, normDelta?)` pairs with no length prefix of their own — the
/// caller already knows the byte run's length (from the level-0/level-1
/// header's own length-prefixed impacts field) and passes exactly that slice.
///
/// Each entry is a vint `freqDelta` whose low bit selects the norm-delta
/// encoding: bit clear means "norm increased by exactly 1 from the previous
/// entry" (the common case — most Lucene norms are monotonically increasing
/// small integers), encoded in zero extra bytes; bit set means an explicit
/// zigzag-encoded `normDelta` vlong follows (`writeImpacts`,
/// `Lucene104PostingsWriter.java:540-556`: `out.writeVInt((freqDelta << 1) |
/// 1); out.writeZLong(normDelta)`). `freq`/`norm` both accumulate from a
/// `(0, 0)` starting point and both deltas are stored as "one less than the
/// true delta" (`impact.freq - previous.freq - 1`), so decode always adds
/// back that `+ 1`. An empty byte slice decodes to an empty list (this can't
/// currently happen on the write side for a full block/span — `writeImpacts`
/// is only called once `docBufferUpto == BLOCK_SIZE` docs have been
/// accumulated, and `CompetitiveImpactAccumulator` always has ≥1 entry once
/// any doc has been added — but the decoder doesn't assume it, matching
/// Java's loop which is driven purely by `in.getPosition() < in.length()`).
pub fn decode_impacts(bytes: &[u8]) -> Result<Impacts> {
    let mut out = Impacts::new();
    decode_impacts_into(bytes, &mut out)?;
    Ok(out)
}

/// [`decode_impacts`] into a caller-owned buffer, so a cursor walking block
/// after block reuses one allocation instead of making one per block. Lucene
/// does the same, decoding `level0SerializedImpacts` into a reusable
/// `FreqAndNormBuffer` (`Lucene104PostingsReader.readImpacts`).
pub fn decode_impacts_into(bytes: &[u8], impacts: &mut Impacts) -> Result<()> {
    let mut r = SliceInput::new(bytes);
    let mut freq: i32 = 0;
    let mut norm: i64 = 0;
    impacts.clear();
    while r.position() < bytes.len() {
        let freq_delta = r.read_vint()?;
        // `1 + (freqDelta >>> 1)` is `int` arithmetic in Java's `readImpacts`
        // and wraps; `(freq_delta as u32) >> 1` reaches `i32::MAX` from a
        // five-byte varint with the sign bit set, so the `1 +` itself is an
        // overflow on a corrupt `.doc` before the accumulator even sees it.
        freq = freq.wrapping_add(1i32.wrapping_add(((freq_delta as u32) >> 1) as i32));
        if freq_delta & 1 != 0 {
            norm = norm.wrapping_add(1i64.wrapping_add(r.read_zlong()?));
        } else {
            norm = norm.wrapping_add(1);
        }
        impacts.push(Impact { freq, norm });
    }
    Ok(())
}

/// The part of `PostingsEnum`'s feature-flag mask a `.doc` decoder can act
/// on.
///
/// Java's `PostingsEnum` exposes six constants -- `NONE`, `FREQS`,
/// `POSITIONS`, `OFFSETS`, `PAYLOADS`, `ALL` -- but they form a chain:
/// `POSITIONS` subsumes `FREQS`, `OFFSETS` and `PAYLOADS` subsume
/// `POSITIONS`, and `NONE` is `DOCS`. `Lucene104PostingsReader` derives
/// exactly one boolean from that mask before it touches `.doc`
/// (`needsFreq = indexHasFreq && featureRequested(flags, FREQS)`); the rest
/// gate `.pos`/`.pay`, which in this port are a separate call
/// ([`read_positions`]) a docs-only caller simply does not make. So this
/// enum is that one boolean, named after the flags it stands for rather
/// than after its implementation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PostingsFlags {
    /// `PostingsEnum.NONE` / `PostingsEnum.DOCS`: doc ids only. Every
    /// returned frequency is `1`, and the `.doc` file's frequency blocks are
    /// stepped over (`PForUtil.skip`) instead of unpacked.
    ///
    /// This is what a constant-score query, an `Occur::FILTER`-shaped clause,
    /// a `TermInSetQuery` and a delete-by-term resolution all want: they read
    /// doc ids and never call `freq()`.
    DocsOnly,
    /// `PostingsEnum.FREQS` and every flag above it: frequencies are decoded.
    #[default]
    Freqs,
}

impl PostingsFlags {
    /// `PostingsEnum.featureRequested(flags, PostingsEnum.FREQS)`.
    fn needs_freq(self) -> bool {
        matches!(self, PostingsFlags::Freqs)
    }
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
    /// blocks (`refillFullBlock`, [`read_full_block_header`] +
    /// [`decode_full_block_body`]) followed by at most
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
        self.read_postings_with_flags(
            meta,
            doc_freq,
            index_options,
            has_payloads,
            PostingsFlags::Freqs,
        )
    }

    /// [`Self::read_postings`] with the consumer's `PostingsEnum` flags:
    /// [`PostingsFlags::DocsOnly`] makes every frequency block on the wire a
    /// `PForUtil.skip` instead of a 256-value unpack, and leaves
    /// [`Postings::freqs`] filled with `1`.
    pub fn read_postings_with_flags(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
        has_payloads: bool,
        flags: PostingsFlags,
    ) -> Result<Postings> {
        let needs_freq = flags.needs_freq();
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
                | IndexOptions::DocsAndCustomFreqs
        ) {
            return Err(Error::Unsupported(
                "IndexOptions::None is not supported in this slice",
            ));
        }
        let index_has_freq = index_options != IndexOptions::Docs;
        let index_has_pos = index_options.subsumes_positions();
        let index_has_offsets_or_payloads = index_options.subsumes_offsets() || has_payloads;

        let mut r = SliceInput::new(self.buf);
        r.seek(meta.doc_start_fp as usize)?;

        // `doc_freq` is read out of the term dictionary, so it must not size
        // an allocation on its own (the `doc_freq <= 1` guard above is what
        // makes the `as usize` cast itself non-negative): a corrupt one turns `with_capacity` into
        // an abort (allocation failure is not a catchable error), which
        // through the FFI is a dead JVM. Capping the reservation at the
        // `.doc` file's own length keeps every real term a single allocation
        // -- a document never costs less than a byte outside a packed block
        // -- and a dense block simply grows the `Vec` once.
        let n = (doc_freq as usize).min(self.buf.len());
        let mut docs = Vec::with_capacity(n);
        let mut freqs = Vec::with_capacity(n);
        let mut level0_impacts: Vec<(i32, Impacts)> = Vec::new();
        let mut level1_impacts: Vec<(i32, Impacts)> = Vec::new();

        let mut prev_doc_id: i32 = -1;
        let mut doc_count_left = doc_freq;
        // One scratch and one pair of block buffers for the whole term, not
        // one per block -- `Lucene104PostingsReader.BlockPostingsEnum` holds
        // the equivalent as instance fields for the life of the enumeration.
        let mut scratch = BlockScratch::new();
        let mut block_docs = [0i32; BLOCK_SIZE as usize];
        let mut block_freqs = [1i32; BLOCK_SIZE as usize];
        // Mirrors `LazyDocsCursor`'s `level1_last_doc_id` accumulator (starts
        // at -1, `+= doc_delta` per level-1 entry) purely to record each
        // span's covering doc-ID bound in `level1_impacts` above.
        let mut level1_last_doc_id: i32 = -1;

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
            let entry = read_level1_entry(
                &mut r,
                index_has_freq,
                index_has_pos,
                index_has_offsets_or_payloads,
            )?;
            // Off disk; see `read_full_block_header`'s `doc_delta`.
            level1_last_doc_id = level1_last_doc_id.wrapping_add(entry.doc_delta);
            if index_has_freq {
                level1_impacts.push((level1_last_doc_id, decode_impacts(entry.impact_bytes)?));
            }
            for _ in 0..LEVEL1_FACTOR {
                let header = read_full_block_header(
                    &mut r,
                    prev_doc_id,
                    index_has_freq,
                    index_has_pos,
                    index_has_offsets_or_payloads,
                )?;
                decode_full_block_body(
                    &mut r,
                    prev_doc_id,
                    index_has_freq,
                    needs_freq,
                    &mut scratch,
                    &mut block_docs,
                    &mut block_freqs,
                )?;
                check_wire_position(r.position(), header.body_end, "full block body")?;
                prev_doc_id = header.last_doc_id;
                if index_has_freq {
                    level0_impacts.push((header.last_doc_id, decode_impacts(header.impact_bytes)?));
                }
                docs.extend_from_slice(&block_docs);
                freqs.extend_from_slice(&block_freqs);
                // ARITH: the enclosing `while` runs only while
                // `doc_count_left >= LEVEL1_NUM_DOCS`, and `LEVEL1_NUM_DOCS ==
                // LEVEL1_FACTOR * BLOCK_SIZE` is exactly what this inner
                // `for _ in 0..LEVEL1_FACTOR` subtracts, so the counter is
                // still `>= 0` after the last of the 32 iterations.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    doc_count_left -= BLOCK_SIZE;
                }
            }
        }

        while doc_count_left >= BLOCK_SIZE {
            let header = read_full_block_header(
                &mut r,
                prev_doc_id,
                index_has_freq,
                index_has_pos,
                index_has_offsets_or_payloads,
            )?;
            decode_full_block_body(
                &mut r,
                prev_doc_id,
                index_has_freq,
                needs_freq,
                &mut scratch,
                &mut block_docs,
                &mut block_freqs,
            )?;
            check_wire_position(r.position(), header.body_end, "full block body")?;
            prev_doc_id = header.last_doc_id;
            if index_has_freq {
                level0_impacts.push((header.last_doc_id, decode_impacts(header.impact_bytes)?));
            }
            docs.extend_from_slice(&block_docs);
            freqs.extend_from_slice(&block_freqs);
            // ARITH: the enclosing `while` runs only while
            // `doc_count_left >= BLOCK_SIZE`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                doc_count_left -= BLOCK_SIZE;
            }
        }
        if doc_count_left > 0 {
            let start = docs.len();
            let count = doc_count_left as usize;
            // ARITH: both loops above ran until `doc_count_left < BLOCK_SIZE`,
            // so `count < 256`; `start` is the length of a live `Vec<i32>`,
            // which is at most `isize::MAX / 4`. The sum cannot reach
            // `usize::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            let end = start + count;
            debug_assert!(count < BLOCK_SIZE as usize);
            docs.resize(end, 0);
            freqs.resize(end, 1);
            read_tail_block(
                &mut r,
                prev_doc_id,
                index_has_freq,
                needs_freq,
                &mut docs[start..],
                &mut freqs[start..],
            )?;
        }

        Ok(Postings {
            docs,
            freqs,
            level0_impacts,
            level1_impacts,
        })
    }

    /// Opens a [`LazyDocsCursor`] over this term's `(docID, freq)` pairs:
    /// blocks are decoded on demand, and a full block whose entire doc range
    /// is behind the caller's `advance()` target is skipped without ever
    /// running `ForUtil`/`PForUtil` decode on it (see [`LazyDocsCursor`]'s
    /// own doc comment for exactly what "skipped" means here). Validation
    /// (`doc_freq <= 1`, `IndexOptions::None`) mirrors
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
        self.lazy_cursor_with_flags(
            meta,
            doc_freq,
            index_options,
            has_payloads,
            PostingsFlags::Freqs,
        )
    }

    /// [`Self::lazy_cursor`] with the consumer's `PostingsEnum` flags: with
    /// [`PostingsFlags::DocsOnly`] every refilled block skips its frequency
    /// block instead of unpacking it, and [`LazyDocsCursor::freq`] always
    /// answers `1`.
    pub fn lazy_cursor_with_flags(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
        has_payloads: bool,
        flags: PostingsFlags,
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
                | IndexOptions::DocsAndCustomFreqs
        ) {
            return Err(Error::Unsupported(
                "IndexOptions::None is not supported in this slice",
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
            needs_freq: flags.needs_freq(),
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
            scratch: BlockScratch::new(),
            pending: None,
            level0_last_doc_id: -1,
            level0_impacts: Impacts::new(),
            level1_impacts: Impacts::new(),
            // `BlockPostingsEnum.reset` (`Lucene104PostingsReader.java:
            // 517-525`): both levels start at the term's own `.pos`/`.pay`
            // start, with nothing consumed.
            level0_pos: PosCursorState {
                pos_fp: meta.pos_start_fp,
                pay_fp: meta.pay_start_fp,
                pos_buffer_upto: 0,
            },
            level1_pos: PosCursorState {
                pos_fp: meta.pos_start_fp,
                pay_fp: meta.pay_start_fp,
                pos_buffer_upto: 0,
            },
            block_pos_origin: PosCursorState {
                pos_fp: meta.pos_start_fp,
                pay_fp: meta.pay_start_fp,
                pos_buffer_upto: 0,
            },
        })
    }
}

/// `Lucene104PostingsFormat.POS_CODEC`.
pub(crate) const POS_CODEC: &str = "Lucene104PostingsWriterPos";
/// `Lucene104PostingsFormat.PAY_CODEC`.
pub(crate) const PAY_CODEC: &str = "Lucene104PostingsWriterPay";

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

/// The flat, `total_term_freq`-long streams a term's `.pos`/`.pay` data
/// decodes into, before they are re-chopped into per-document groups.
///
/// Split out so a caller wanting only positions -- phrase matching, which is
/// the caller that matters for speed -- can assemble them without building a
/// `Vec<Position>` per document. See [`read_positions_flat`].
struct PositionStreams {
    pos_deltas: Vec<i32>,
    payload_lengths: Vec<i32>,
    payload_bytes: Vec<u8>,
    offset_start_deltas: Vec<i32>,
    offset_lengths: Vec<i32>,
}

/// Decodes a term's `.pos` (and `.pay`, where the field has offsets or
/// payloads) into flat streams: the wire-format half of [`read_positions`].
/// The re-chopping half lives in its callers, because they disagree about what
/// shape they want it in.
fn decode_position_streams(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
) -> Result<PositionStreams> {
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

    let n = wire_count(total_term_freq, "total_term_freq")?;
    // `total_term_freq` comes off disk, so it sizes nothing directly:
    // `with_capacity(n)` on a corrupt term dictionary is an allocation of
    // arbitrary size, which fails by *aborting* the process rather than by
    // returning an error -- through the FFI, a dead JVM with no exception.
    // Reserving at most one entry per byte of `.pos` keeps the real case a
    // single allocation (a position never costs less than a byte in the vint
    // tail) and the corrupt case bounded by a file that actually exists; a
    // genuinely denser packed block just grows the `Vec` once more.
    let reserve = n.min(pos.buf.len());
    let mut pos_deltas: Vec<i32> = Vec::with_capacity(reserve);
    let mut payload_lengths: Vec<i32> = Vec::with_capacity(if has_payloads { reserve } else { 0 });
    let mut payload_bytes: Vec<u8> = Vec::new();
    let mut offset_start_deltas: Vec<i32> =
        Vec::with_capacity(if has_offsets { reserve } else { 0 });
    let mut offset_lengths: Vec<i32> = Vec::with_capacity(if has_offsets { reserve } else { 0 });

    // `meta.last_pos_block_offset` (already decoded by `decode_term_metadata`)
    // tells us exactly where the vint tail block begins on the wire, which is
    // equivalent to (but doesn't require us to re-derive live, unlike the
    // real reader's `posIn.getFilePointer() == lastPosBlockFP` check) simply
    // computing how many full 256-position blocks precede it from
    // `total_term_freq` itself.
    let (num_full_blocks, tail_count) = full_blocks_and_tail(n);

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
            // `read_length`, not `read_vint as usize`: a negative or
            // longer-than-the-file byte count would otherwise size the
            // `resize` below straight off disk.
            let num_bytes = pay_r.read_length("payload block")?;
            let start = payload_bytes.len();
            payload_bytes.resize(add_wire_offset(start, num_bytes)?, 0);
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
                // `code >>> 1`; see `refill_last_position_block`. The two
                // decoders are asserted equivalent by
                // `postings_wanted_docs.rs`, so they must agree here too.
                pos_deltas.push(((code as u32) >> 1) as i32);
                if last_payload_length != 0 {
                    // Read off disk as a plain vint, so negative and absurd
                    // are both reachable: bound it by the bytes that are
                    // actually there before it sizes an allocation.
                    let len = wire_length(last_payload_length as i64, "tail payload")?;
                    if len > pos_r.remaining() {
                        return Err(corrupted(format!(
                            "tail payload length {len} exceeds {} remaining .pos bytes",
                            pos_r.remaining()
                        )));
                    }
                    let start = payload_bytes.len();
                    payload_bytes.resize(add_wire_offset(start, len)?, 0);
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
                offset_start_deltas.push(((delta_code as u32) >> 1) as i32);
                offset_lengths.push(last_offset_length);
            }
        }
    }

    Ok(PositionStreams {
        pos_deltas,
        payload_lengths,
        payload_bytes,
        offset_start_deltas,
        offset_lengths,
    })
}

/// Positions only, in one flat array with per-document start offsets, instead
/// of a `Vec<Vec<Position>>`.
///
/// Returns `(positions, doc_starts)`: document `i`'s positions are
/// `positions[doc_starts[i] as usize..doc_starts[i + 1] as usize]`, and
/// `doc_starts` has `freqs.len() + 1` entries so the last document needs no
/// special case.
///
/// Why it exists: phrase matching went through [`read_positions`], which
/// allocates one `Vec<Position>` per matching document -- roughly five million
/// allocations for a query on a high-frequency term -- and about half of a
/// phrase query's runtime was in `malloc`/`free`/`memcpy` as a result. This
/// makes two allocations regardless of how many documents match, and drops the
/// offset and payload fields a phrase matcher never reads. Lucene does not
/// materialize a per-document container at all: `BlockPostingsEnum` walks
/// `nextPosition()` out of a reusable buffer, which is further still than this
/// goes.
pub fn read_positions_flat(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    freqs: &[i32],
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
) -> Result<(Vec<i32>, Vec<u32>)> {
    let streams =
        decode_position_streams(pos, pay, meta, total_term_freq, index_options, has_payloads)?;
    let pos_deltas = streams.pos_deltas;
    // `wire_count`, not `as usize`: a negative `totalTermFreq` sign-extends to
    // ~2^64 and would make the `idx != n` reconciliation below unsatisfiable
    // in a way that reads as "fewer occurrences than claimed" rather than as
    // the corrupt term dictionary it is. `decode_position_streams` rejects it
    // too, but this file's rule is that the check lives where the value is
    // used, not two calls away.
    let n = wire_count(total_term_freq, "total_term_freq")?;

    // Capped for the same reason [`decode_position_streams`] caps its own
    // reservations: `n` is `total_term_freq` off disk.
    let mut positions: Vec<i32> = Vec::with_capacity(n.min(pos_deltas.len()));
    // ARITH: `freqs` is a live slice, so `freqs.len() <= isize::MAX` and the
    // `+ 1` cannot reach `usize::MAX`. `freqs` is this port's own decoded
    // `Postings::freqs`, not a length read off disk.
    #[allow(clippy::arithmetic_side_effects)]
    let mut doc_starts: Vec<u32> = Vec::with_capacity(freqs.len() + 1);
    let mut idx = 0usize;
    for &freq in freqs {
        doc_starts.push(positions.len() as u32);
        // Deltas reset at each document's first occurrence: they are only ever
        // relative to the previous occurrence of the *same* document.
        let mut position = 0i32;
        for _ in 0..freq {
            // Same guard as `read_positions`: `freqs` comes from `.doc` and
            // `total_term_freq` from the term dictionary, and nothing on the
            // wire makes them agree, so a corrupt segment must surface a
            // decode error rather than index past the end.
            if idx >= pos_deltas.len() {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "sum of per-doc freqs exceeds total_term_freq".into(),
                )));
            }
            // `wrapping_add`, like [`SinkCursor::emit`]: the delta is a
            // `.pos` file value widened from `u32`, so a corrupt block
            // overflows the accumulator, and an overflow panic in a debug
            // build of the FFI takes the JVM down. Java accumulates in an
            // `int` and wraps.
            position = position.wrapping_add(pos_deltas[idx]);
            positions.push(position);
            // ARITH: the guard three lines up returned unless
            // `idx < pos_deltas.len()`, and a live slice's length is at most
            // `isize::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                idx += 1;
            }
        }
    }
    doc_starts.push(positions.len() as u32);
    if idx != n {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "sum of per-doc freqs is less than total_term_freq".into(),
        )));
    }
    Ok((positions, doc_starts))
}

/// Where one wanted document's occurrences sit in a term's flat,
/// `total_term_freq`-long occurrence stream: `[from, to)`.
type OccurrenceRange = (usize, usize);

/// What a [`walk_wanted_occurrences`] pass keeps of each occurrence it reaches.
///
/// Two implementations, monomorphized so neither pays for the other's work:
/// [`PositionsOnly`] for phrase matching, [`FullOccurrences`] for
/// highlighting. `NEEDS_EXTRAS` decides whether `.pay`'s offset and payload
/// blocks are unpacked or stepped over with `PForUtil.skip`, which is the same
/// job `PostingsEnum`'s flag mask does one level up in Java.
trait OccurrenceSink {
    /// Whether this sink reads offsets and payloads at all.
    const NEEDS_EXTRAS: bool;

    /// A wanted document is starting. Issued exactly once per entry in
    /// `wanted`, in `wanted` order, before any of that document's
    /// occurrences -- including for entries that turn out to have none, so
    /// the per-document index stays aligned with `wanted`.
    fn open_doc(&mut self);

    /// One occurrence of the document currently open. `start_offset`/
    /// `end_offset` are `-1` and `payload` is empty where the field does not
    /// index them (or the sink did not ask), matching
    /// `PostingsEnum.startOffset`'s and `getPayload`'s own no-data contracts.
    fn push(&mut self, position: i32, start_offset: i32, end_offset: i32, payload: &[u8]);
}

/// [`OccurrenceSink`] for phrase matching: positions in one flat `Vec`, with
/// per-document start indices. No offsets, no payloads, no per-document
/// container.
#[derive(Debug)]
struct PositionsOnly {
    positions: Vec<i32>,
    doc_starts: Vec<u32>,
}

impl OccurrenceSink for PositionsOnly {
    const NEEDS_EXTRAS: bool = false;

    #[inline]
    fn open_doc(&mut self) {
        self.doc_starts.push(self.positions.len() as u32);
    }

    #[inline]
    fn push(&mut self, position: i32, _start_offset: i32, _end_offset: i32, _payload: &[u8]) {
        self.positions.push(position);
    }
}

/// [`OccurrenceSink`] for highlighting: whole [`Position`] records --
/// `nextPosition()`, `startOffset()`, `endOffset()`, `getPayload()` -- in the
/// same flat shape.
#[derive(Debug)]
struct FullOccurrences {
    occurrences: Vec<Position>,
    doc_starts: Vec<u32>,
}

impl OccurrenceSink for FullOccurrences {
    const NEEDS_EXTRAS: bool = true;

    #[inline]
    fn open_doc(&mut self) {
        self.doc_starts.push(self.occurrences.len() as u32);
    }

    #[inline]
    fn push(&mut self, position: i32, start_offset: i32, end_offset: i32, payload: &[u8]) {
        self.occurrences.push(Position {
            position,
            start_offset,
            end_offset,
            payload: payload.to_vec(),
        });
    }
}

/// The walk's position within `wanted`: which range is being collected,
/// whether its [`OccurrenceSink::open_doc`] has been issued yet, and the
/// per-document position/offset accumulators.
///
/// Both accumulators reset at each document's *first* occurrence, never
/// across a document boundary -- the read-side mirror of
/// `Lucene104PostingsWriter.startDoc`'s `lastPosition = 0; lastStartOffset =
/// 0;`. That reset is also what makes skipping documents sound: a document's
/// positions and offsets are self-contained, so nothing carried over from the
/// documents this pass skipped is needed to decode the ones it keeps.
struct SinkCursor {
    /// Index into the range list of the document being collected.
    w: usize,
    /// Whether `open_doc` has already been issued for `w`.
    open: bool,
    position: i32,
    offset: i32,
}

impl SinkCursor {
    fn new() -> Self {
        SinkCursor {
            w: 0,
            open: false,
            position: 0,
            offset: 0,
        }
    }

    /// Brings the cursor to occurrence `g`: closes every range that ends at
    /// or before it (issuing the `open_doc` of any that never got one, so a
    /// wanted document with no occurrences still takes its slot), then opens
    /// the next range once `g` reaches its first occurrence.
    #[inline]
    fn sync<S: OccurrenceSink>(&mut self, sink: &mut S, ranges: &[OccurrenceRange], g: usize) {
        while self.w < ranges.len() && ranges[self.w].1 <= g {
            if !self.open {
                sink.open_doc();
            }
            // ARITH: the loop condition established `self.w < ranges.len()`,
            // and a live slice's length is at most `isize::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.w += 1;
            }
            self.open = false;
        }
        if self.w < ranges.len() && !self.open && g >= ranges[self.w].0 {
            sink.open_doc();
            self.open = true;
            self.position = 0;
            self.offset = 0;
        }
    }

    /// Accumulates one occurrence's deltas and hands the absolute values to
    /// the sink. `offsets` is `None` when the field has no offsets or the
    /// sink did not ask for them.
    #[inline]
    fn emit<S: OccurrenceSink>(
        &mut self,
        sink: &mut S,
        position_delta: i32,
        offsets: Option<(i32, i32)>,
        payload: &[u8],
    ) {
        // `wrapping_add`, not `+`: every one of these deltas is a file value,
        // so a corrupt `.pos`/`.pay` can overflow the accumulator, and an
        // overflow panic in a debug build of the FFI takes the JVM down.
        // Java accumulates in an `int` and wraps silently; a wrapped position
        // is a wrong answer for a corrupt file, not a crash.
        self.position = self.position.wrapping_add(position_delta);
        let (start_offset, end_offset) = match offsets {
            Some((start_delta, length)) => {
                self.offset = self.offset.wrapping_add(start_delta);
                (self.offset, self.offset.wrapping_add(length))
            }
            None => (-1, -1),
        };
        sink.push(self.position, start_offset, end_offset, payload);
    }

    /// Issues the `open_doc` of every range the stream never reached, so the
    /// per-document index always has one entry per `wanted` entry.
    fn finish<S: OccurrenceSink>(&mut self, sink: &mut S, ranges: &[OccurrenceRange]) {
        while self.w < ranges.len() {
            if !self.open {
                sink.open_doc();
            }
            // ARITH: the loop condition established `self.w < ranges.len()`,
            // and a live slice's length is at most `isize::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.w += 1;
            }
            self.open = false;
        }
    }
}

/// Each `wanted` entry's occurrence range, in `wanted` order, plus the
/// validation that `freqs` and `total_term_freq` agree.
///
/// One pass over `freqs`, accumulating as it goes and emitting a range each
/// time it reaches the next `wanted` index. The prefix-sum array this
/// replaced was `docFreq + 1` entries long whatever the caller asked for --
/// 20 MB for a five-million-document term, to answer a question about one
/// document. Java needs neither: `.doc`'s skip data carries the `.pos` file
/// pointers, so `advance` never sums a frequency at all (see
/// [`walk_wanted_occurrences`]).
///
/// An entry the doc list does not have, or one that is not strictly after the
/// entry before it (an unsorted or repeated `wanted`), gets an empty range at
/// the current watermark: it keeps its slot in the result and contributes
/// nothing. The alternative -- indexing a prefix-sum array by whatever the
/// caller passed -- is an out-of-bounds panic on a caller's mistake, which is
/// what this used to be.
///
/// `freqs` is decoded from `.doc` and `n` comes from the term dictionary;
/// nothing on the wire makes them agree, so their disagreement is a decode
/// error rather than an index past the end of something.
fn wanted_ranges(wanted: &[usize], freqs: &[i32], n: usize) -> Result<Vec<OccurrenceRange>> {
    #[inline]
    fn add(acc: &mut u64, freq: i32, n: usize) -> Result<()> {
        if freq < 0 {
            return Err(corrupted(format!("negative per-doc frequency {freq}")));
        }
        // ARITH: on entry `*acc <= n <= u32::MAX` (the check below returned
        // otherwise, and `n` came through `wire_count`), and `freq` is a
        // non-negative `i32`, so the sum is under 2^33.
        #[allow(clippy::arithmetic_side_effects)]
        {
            *acc += freq as u64;
        }
        if *acc > n as u64 {
            return Err(corrupted(
                "sum of per-doc freqs disagrees with total_term_freq",
            ));
        }
        Ok(())
    }

    let mut ranges = Vec::with_capacity(wanted.len());
    // Frequencies of `freqs[..next]` are already in `acc`.
    let mut acc: u64 = 0;
    let mut next = 0usize;
    let mut watermark = 0usize;
    for &d in wanted {
        let range = if d < freqs.len() && d >= next {
            for &freq in &freqs[next..d] {
                add(&mut acc, freq, n)?;
            }
            next = d;
            let freq = freqs[d];
            if freq < 0 {
                return Err(corrupted(format!("negative per-doc frequency {freq}")));
            }
            // `acc + freq` is where this document's occurrences end. It is
            // checked against `n` here rather than only later, when the
            // running sum reaches `freqs[d]` -- both because a `usize` on a
            // 32-bit target has no room to spare above `n` (`wire_count` caps
            // `n` at `u32::MAX`, which is `usize::MAX` there), and because a
            // wrapped end is a silently wrong occurrence range rather than a
            // crash.
            let from = acc as usize;
            let to = acc
                .checked_add(freq as u64)
                .filter(|&to| to <= n as u64)
                .ok_or_else(|| corrupted("sum of per-doc freqs disagrees with total_term_freq"))?;
            (from, to as usize)
        } else {
            (watermark, watermark)
        };
        let range = if range.0 >= watermark {
            range
        } else {
            (watermark, watermark)
        };
        watermark = range.1;
        ranges.push(range);
    }
    // Finish the sum, so a frequency list that disagrees with
    // `total_term_freq` is still rejected even when nothing after the last
    // wanted document was needed.
    for &freq in &freqs[next..] {
        add(&mut acc, freq, n)?;
    }
    if acc != n as u64 {
        return Err(corrupted(
            "sum of per-doc freqs disagrees with total_term_freq",
        ));
    }
    Ok(ranges)
}

/// Steps over one full 256-occurrence block of `.pos` (and its `.pay`
/// companions) without unpacking anything: `PForUtil.skip` reads the token
/// byte and seeks past the packed body, where a decode would bit-unpack 256
/// values.
///
/// This is the block-level half of what makes a wanted-documents walk cheap.
/// The occurrence-level half is that a skipped block's positions are never
/// needed: positions and offsets restart at every document's first
/// occurrence, so nothing decoded here would have been carried forward.
fn skip_position_block(
    pos_r: &mut SliceInput,
    pay_r: Option<&mut SliceInput>,
    has_payloads: bool,
    has_offsets: bool,
) -> Result<()> {
    for_util::pfor_skip(pos_r)?;
    if !has_payloads && !has_offsets {
        return Ok(());
    }
    let r = pay_r.expect("checked by the caller: .pay is opened for offsets or payloads");
    if has_payloads {
        for_util::pfor_skip(r)?;
        let num_bytes = r.read_length("payload block")?;
        r.skip(num_bytes)?;
    }
    if has_offsets {
        for_util::pfor_skip(r)?;
        for_util::pfor_skip(r)?;
    }
    Ok(())
}

/// What a position walk asks of each block: which of `.pos`/`.pay`'s streams
/// exist on the wire, and which of them this walk actually unpacks.
///
/// The `has_*`/`want_*` split is `Lucene104PostingsReader`'s
/// `indexHasOffsets`/`needsOffsets` pair: a stream that exists but is not
/// wanted is still *stepped over*, because the streams are interleaved and
/// nothing else gives their length.
#[derive(Debug, Clone, Copy)]
struct PositionWants {
    has_offsets: bool,
    has_payloads: bool,
    want_offsets: bool,
    want_payloads: bool,
}

/// `BlockPostingsEnum`'s position buffers: the fixed 256-entry arrays
/// `refillPositions` fills, plus the block's payload byte run.
///
/// Held by the walker for the length of a walk and refilled in place, which is
/// what Lucene does with its `posDeltaBuffer`/`payloadLengthBuffer`/
/// `offsetStartDeltaBuffer`/`offsetLengthBuffer`/`payloadBytes` instance
/// fields. `len` is `BLOCK_SIZE` for a full block and
/// `totalTermFreq % BLOCK_SIZE` for the vint tail.
struct PositionBlock {
    pos_deltas: [u32; for_util::BLOCK_SIZE],
    offset_start_deltas: [u32; for_util::BLOCK_SIZE],
    offset_lengths: [u32; for_util::BLOCK_SIZE],
    payload_lengths: [u32; for_util::BLOCK_SIZE],
    /// The block's payloads, concatenated; occurrence `i`'s payload is the
    /// `payload_lengths[i]` bytes at `sum(payload_lengths[..i])`. Empty
    /// unless `want_payloads`.
    payload_bytes: Vec<u8>,
    len: usize,
}

impl PositionBlock {
    fn new() -> Self {
        PositionBlock {
            pos_deltas: [0; for_util::BLOCK_SIZE],
            offset_start_deltas: [0; for_util::BLOCK_SIZE],
            offset_lengths: [0; for_util::BLOCK_SIZE],
            payload_lengths: [0; for_util::BLOCK_SIZE],
            payload_bytes: Vec::new(),
            len: 0,
        }
    }

    /// Occurrence `i`'s payload, or `&[]` when payloads are not being read.
    ///
    /// `payload_upto` is the caller's running byte offset; every bound is
    /// checked because the lengths come off `.pay`/`.pos` and nothing on the
    /// wire ties them to the byte run's actual length.
    #[inline]
    fn payload(&self, payload_upto: usize, length: usize) -> Result<&[u8]> {
        payload_upto
            .checked_add(length)
            .and_then(|end| self.payload_bytes.get(payload_upto..end))
            .ok_or_else(|| corrupted("payload lengths overrun the block's payload byte run"))
    }
}

/// `Lucene104PostingsReader.refillPositions`' full-block branch plus
/// `refillOffsetsOrPayloads`: one 256-occurrence `PForUtil` block of `.pos`,
/// and the `.pay` blocks that go with it.
fn refill_full_position_block(
    pos_r: &mut SliceInput,
    pay_r: Option<&mut SliceInput>,
    wants: PositionWants,
    for_util_state: &mut for_util::ForUtil,
    block: &mut PositionBlock,
) -> Result<()> {
    for_util_state.pfor_decode(pos_r, &mut block.pos_deltas)?;
    block.len = for_util::BLOCK_SIZE;
    block.payload_bytes.clear();
    if !wants.has_offsets && !wants.has_payloads {
        return Ok(());
    }
    let r = pay_r.expect("checked by the caller: .pay is opened for offsets or payloads");
    if wants.has_payloads {
        if wants.want_payloads {
            for_util_state.pfor_decode(r, &mut block.payload_lengths)?;
        } else {
            for_util::pfor_skip(r)?;
        }
        // `read_length`, not `read_vint() as usize`: a negative or
        // longer-than-the-file byte count would otherwise size the `resize`
        // below straight off disk.
        let num_bytes = r.read_length("payload block")?;
        if wants.want_payloads {
            block.payload_bytes.resize(num_bytes, 0);
            r.read_bytes(&mut block.payload_bytes)?;
        } else {
            r.skip(num_bytes)?;
        }
    }
    if wants.has_offsets {
        if wants.want_offsets {
            for_util_state.pfor_decode(r, &mut block.offset_start_deltas)?;
            for_util_state.pfor_decode(r, &mut block.offset_lengths)?;
        } else {
            for_util::pfor_skip(r)?;
            for_util::pfor_skip(r)?;
        }
    }
    Ok(())
}

/// `Lucene104PostingsReader.refillLastPositionBlock`: the trailing
/// `totalTermFreq % BLOCK_SIZE` occurrences, vint-coded in `.pos` alone --
/// payload bytes inlined right after their length, and a payload/offset
/// length written only when it *changes* from the previous occurrence's (bit
/// 0 of the vint code), reusing the last value otherwise. `.pay` is not
/// touched at all.
fn refill_last_position_block(
    pos_r: &mut SliceInput,
    wants: PositionWants,
    count: usize,
    block: &mut PositionBlock,
) -> Result<()> {
    debug_assert!(count <= for_util::BLOCK_SIZE);
    block.len = count;
    block.payload_bytes.clear();
    let mut last_payload_length = 0i32;
    let mut last_offset_length = 0i32;
    for i in 0..count {
        let code = pos_r.read_vint()?;
        if wants.has_payloads {
            if code & 1 != 0 {
                last_payload_length = pos_r.read_vint()?;
            }
            // `code >>> 1` in Java, not `>>`: `code` is a signed vint and
            // `(delta << 1) | 1` is negative for a delta at or above 2^30,
            // which `IndexWriter.MAX_POSITION` permits. An arithmetic shift
            // would sign-extend and recover the wrong delta.
            block.pos_deltas[i] = (code as u32) >> 1;
            // Read off disk as a plain vint, so negative and absurd are both
            // reachable: bound it by the bytes that are actually there before
            // it sizes a copy.
            let length = wire_length(last_payload_length as i64, "tail payload")?;
            if length > pos_r.remaining() {
                return Err(corrupted(format!(
                    "tail payload length {length} exceeds {} remaining .pos bytes",
                    pos_r.remaining()
                )));
            }
            if wants.want_payloads {
                block.payload_lengths[i] = length as u32;
                let start = block.payload_bytes.len();
                block
                    .payload_bytes
                    .resize(add_wire_offset(start, length)?, 0);
                pos_r.read_bytes(&mut block.payload_bytes[start..])?;
            } else {
                pos_r.skip(length)?;
            }
        } else {
            block.pos_deltas[i] = code as u32;
        }

        if wants.has_offsets {
            let delta_code = pos_r.read_vint()?;
            if delta_code & 1 != 0 {
                last_offset_length = pos_r.read_vint()?;
            }
            if wants.want_offsets {
                // `deltaCode >>> 1`, for the same reason as `code` above.
                block.offset_start_deltas[i] = (delta_code as u32) >> 1;
                block.offset_lengths[i] = last_offset_length as u32;
            }
        }
    }
    Ok(())
}

/// Where a term's vint position tail begins, as
/// `Lucene104PostingsReader.reset` computes `lastPosBlockFP`
/// (`Lucene104PostingsReader.java:526-532`).
///
/// `None` means "this term has no vint tail": `totalTermFreq == BLOCK_SIZE`
/// exactly, so its `.pos` is one full block and nothing else. Java writes
/// `-1`, a file pointer no `.pos` position can equal, for the same reason.
///
/// This is the field `b5` found being written as a constant `0`
/// (`docs/sweep/m2/b5-postings.md` F4). Nothing in this port read it back
/// then, because every position walk started at the term's `posStartFP` and
/// counted occurrences. A walk that *jumps into the middle* of `.pos` has no
/// occurrence count to derive the split from, so this is now the only thing
/// that tells a full block from the tail -- exactly the role it has in
/// `refillPositions`.
fn last_pos_block_fp(meta: TermMetadata, total_term_freq: i64) -> Option<u64> {
    match total_term_freq.cmp(&(BLOCK_SIZE as i64)) {
        std::cmp::Ordering::Less => Some(meta.pos_start_fp),
        std::cmp::Ordering::Equal => None,
        std::cmp::Ordering::Greater => Some(
            meta.pos_start_fp
                .wrapping_add(meta.last_pos_block_offset as u64),
        ),
    }
}

/// `PostingsEnum.advance(doc)` followed by that document's `nextPosition()` /
/// `startOffset()` / `endOffset()` / `getPayload()`, starting from the
/// `.pos`/`.pay` origin `.doc`'s own skip data gave us rather than from the
/// term's first occurrence.
///
/// This is `Lucene104PostingsReader`'s `seekPosData` + `skipPositions` +
/// `nextPosition` trio. The difference from [`walk_wanted_occurrences`] is
/// entirely in where it starts: that one addresses `.pos` by a running
/// frequency sum over the whole doc list, this one is handed a file pointer.
///
/// Every value it trusts comes off disk, so: the block-skipping loop is
/// bounded by the `.pos` file itself (a `PForUtil` block is never zero bytes,
/// and `pfor_skip` fails at EOF), `to_skip` is checked against the landing
/// block's own length rather than indexing it, and `freq` is checked against
/// the occurrences the streams actually yield.
#[allow(clippy::too_many_arguments)]
fn walk_document_occurrences<S: OccurrenceSink>(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    origin: PositionOrigin,
    freq: usize,
    last_pos_block: Option<u64>,
    tail_count: usize,
    wants: PositionWants,
    sink: &mut S,
) -> Result<()> {
    if (wants.has_offsets || wants.has_payloads) && pay.is_none() {
        return Err(Error::Unsupported(
            "positions need an opened .pay file: this field has offsets or payloads",
        ));
    }
    // `usize::MAX` on a 32-bit target where the pointer does not fit: the
    // seek then fails as the out-of-range read it is, rather than silently
    // truncating to an in-range address.
    let mut pos_r = SliceInput::new(pos.buf);
    pos_r.seek(usize::try_from(origin.pos_fp).unwrap_or(usize::MAX))?;
    let mut pay_r = pay.map(|p| SliceInput::new(p.buf));
    if let Some(r) = pay_r.as_mut() {
        r.seek(usize::try_from(origin.pay_fp).unwrap_or(usize::MAX))?;
    }

    // `skipPositions`' whole-block loop: step over the blocks entirely behind
    // the target document, one token byte and a seek per stream.
    let mut to_skip = origin.skip;
    while to_skip >= for_util::BLOCK_SIZE as u64 {
        // `assert posIn.getFilePointer() != lastPosBlockFP` in Java: the vint
        // tail is the last block there is, so a skip that reaches it and
        // still wants to step past it means the skip data and the term's
        // `totalTermFreq` disagree.
        if last_pos_block == Some(pos_r.position() as u64) {
            return Err(corrupted(
                "the .doc skip data asks to step past the last .pos block",
            ));
        }
        skip_position_block(
            &mut pos_r,
            pay_r.as_mut(),
            wants.has_payloads,
            wants.has_offsets,
        )?;
        // ARITH: the loop condition is `to_skip >= for_util::BLOCK_SIZE`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            to_skip -= for_util::BLOCK_SIZE as u64;
        }
    }

    let mut for_util_state = for_util::ForUtil::new();
    let mut block = PositionBlock::new();
    // `refillPositions`' one-line dispatch: the vint tail is recognised by
    // the file pointer alone. It is also the *last* block, so once it has
    // been decoded there is nothing after it -- a walk still wanting
    // occurrences at that point has been handed a frequency the term does not
    // have, and must say so rather than decoding the footer as a `PForUtil`
    // block.
    let mut tail_decoded = false;
    let mut refill = |pos_r: &mut SliceInput,
                      pay_r: Option<&mut SliceInput>,
                      block: &mut PositionBlock,
                      tail_decoded: &mut bool|
     -> Result<()> {
        if *tail_decoded {
            return Err(corrupted(
                "a document's frequency outruns the occurrences .pos holds for the term",
            ));
        }
        if last_pos_block == Some(pos_r.position() as u64) {
            *tail_decoded = true;
            refill_last_position_block(pos_r, wants, tail_count, block)
        } else {
            refill_full_position_block(pos_r, pay_r, wants, &mut for_util_state, block)
        }
    };
    refill(&mut pos_r, pay_r.as_mut(), &mut block, &mut tail_decoded)?;

    let mut upto = to_skip as usize;
    if upto > block.len {
        return Err(corrupted(format!(
            "the .doc skip data lands {upto} occurrences into a {}-occurrence .pos block",
            block.len
        )));
    }
    // `skipPositions`' `payloadByteUpto = sumOverRange(payloadLengthBuffer, 0,
    // toSkip)`. Saturating, because the lengths are `.pay` values: an
    // overflowed offset then fails `PositionBlock::payload`'s bounds check
    // rather than panicking in a debug build.
    let mut payload_upto = 0usize;
    if wants.want_payloads {
        for &l in &block.payload_lengths[..upto] {
            payload_upto = payload_upto.saturating_add(l as usize);
        }
    }

    // The occurrence budget. `freq` reaches here from `.doc` and is checked
    // against `total_term_freq`, but `total_term_freq` is itself an
    // unvalidated `.tim` vlong -- so a corrupt segment where the two *agree*
    // has nothing stopping the loop below except `.pos` running out. That is
    // not a panic, it is worse: a minimal `PForUtil` block is a couple of
    // bytes for 256 values, so the walk can produce on the order of a hundred
    // `Position` records (each with its own payload `Vec`) per byte of `.pos`
    // before it EOFs -- an allocation blow-up, and an allocation failure
    // *aborts*, which no `catch_unwind` at the FFI boundary can intercept.
    //
    // The batch walker has no such exposure: `wanted_ranges` rejects unless
    // the frequency list sums to `total_term_freq` exactly. A single-document
    // walk cannot have that invariant, so it gets an explicit ceiling
    // instead: `BLOCK_SIZE` occurrences per byte of `.pos` after the origin
    // is the densest any block can be (a 256-value `PForUtil` block is at
    // least its own token byte).
    let pos_bytes_left = pos.buf.len().saturating_sub(pos_r.position());
    let max_occurrences = pos_bytes_left.saturating_mul(for_util::BLOCK_SIZE);
    if freq > max_occurrences {
        return Err(corrupted(format!(
            "document frequency {freq} cannot fit in the {pos_bytes_left} .pos bytes that \
             follow this document's first occurrence"
        )));
    }

    let ranges = [(0usize, freq)];
    let mut cursor = SinkCursor::new();
    for g in 0..freq {
        if upto == block.len {
            refill(&mut pos_r, pay_r.as_mut(), &mut block, &mut tail_decoded)?;
            upto = 0;
            payload_upto = 0;
            if block.len == 0 {
                // An empty vint tail (`totalTermFreq % BLOCK_SIZE == 0` with a
                // `lastPosBlockOffset` that points at it anyway).
                return Err(corrupted(
                    "a document's frequency outruns the occurrences .pos holds for the term",
                ));
            }
        }
        cursor.sync(sink, &ranges, g);
        // Pins the invariant the `upto += 1` below rests on, and with it the
        // in-bounds-ness of every fixed-256-entry index in this loop body.
        debug_assert!(upto < block.len && block.len <= for_util::BLOCK_SIZE);
        let payload_length = if wants.want_payloads {
            block.payload_lengths[upto] as usize
        } else {
            0
        };
        let payload = if wants.want_payloads {
            block.payload(payload_upto, payload_length)?
        } else {
            &[][..]
        };
        let offsets = wants.want_offsets.then(|| {
            (
                block.offset_start_deltas[upto] as i32,
                block.offset_lengths[upto] as i32,
            )
        });
        cursor.emit(sink, block.pos_deltas[upto] as i32, offsets, payload);
        payload_upto = payload_upto.saturating_add(payload_length);
        // ARITH: `upto <= block.len` on entry to the loop body (checked once
        // before the loop, and restored to 0 by the refill above whenever it
        // reaches `block.len`), and `block.len` is `for_util::BLOCK_SIZE` for
        // a full block or `tail_count < BLOCK_SIZE` for the vint tail -- so
        // `upto <= 256` here and the increment stays inside a `usize`. This
        // is also what makes the fixed-256-entry indexing above in-bounds.
        #[allow(clippy::arithmetic_side_effects)]
        {
            upto += 1;
        }
    }
    cursor.finish(sink, &ranges);
    Ok(())
}

/// One forward pass over a term's `.pos`/`.pay` streams that materializes only
/// the documents `wanted` names, as indices into the term's own doc list.
///
/// This is `PostingsEnum.advance(doc)` followed by `nextPosition()` /
/// `startOffset()` / `endOffset()` / `getPayload()`, for a batch of documents
/// instead of one -- the shape both callers actually want, and the shape that
/// lets the batch be walked in a single pass.
///
/// # What it skips, and what it cannot
///
/// A full block that holds no wanted occurrence is stepped over with
/// [`skip_position_block`]: one token byte and a seek per stream, instead of
/// three `PForUtil` unpacks of 256 values each. Once the last wanted document
/// is behind it the pass returns immediately, leaving the rest of the term's
/// `.pos`/`.pay` untouched -- so highlighting the first document of a
/// million-document postings list reads a handful of blocks, not all of them.
///
/// What it still pays is one token byte and a seek per intervening block,
/// because a `PForUtil` block's length is only knowable from its own header --
/// and, before that, the caller's whole `freqs` list, because this addresses
/// `.pos` by a running frequency sum.
///
/// [`read_occurrences_for_doc`] pays neither: `.doc`'s level-0/level-1 skip
/// data carries the `.pos`/`.pay` file pointers, so it jumps straight to the
/// block it needs. It answers about *one* document, which is what the
/// highlighter asks and what the skip data can address; this batch form is
/// still what phrase matching wants, because its `wanted` set is a large
/// fraction of the term's doc list and it has already decoded that list to
/// intersect it. See `docs/sweep/m2/c20-postings-skip.md`.
#[allow(clippy::too_many_arguments)]
fn walk_wanted_occurrences<S: OccurrenceSink>(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    freqs: &[i32],
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
    wanted: &[usize],
    sink: &mut S,
) -> Result<()> {
    if !index_options.subsumes_positions() {
        return Err(Error::Unsupported(
            "positions need a field with IndexOptions::DocsAndFreqsAndPositions or higher",
        ));
    }
    let has_offsets = index_options.subsumes_offsets();
    let want_offsets = S::NEEDS_EXTRAS && has_offsets;
    let want_payloads = S::NEEDS_EXTRAS && has_payloads;
    let n = wire_count(total_term_freq, "total_term_freq")?;

    let ranges = wanted_ranges(wanted, freqs, n)?;

    let (num_full_blocks, tail_count) = full_blocks_and_tail(n);
    // `.pay` is only ever touched by full `PForUtil` blocks (the vint tail's
    // payload bytes live inline in `.pos`), so a term whose whole
    // `total_term_freq` fits in the tail never needs it, even for a field
    // with offsets or payloads.
    if num_full_blocks > 0 && (has_offsets || has_payloads) && pay.is_none() {
        return Err(Error::Unsupported(
            "positions need an opened .pay file: this field has offsets or payloads and \
             total_term_freq spans at least one full 256-position block",
        ));
    }

    let mut pos_r = SliceInput::new(pos.buf);
    pos_r.seek(meta.pos_start_fp as usize)?;
    let mut pay_r = pay.map(|p| SliceInput::new(p.buf));
    if let Some(r) = pay_r.as_mut() {
        r.seek(meta.pay_start_fp as usize)?;
    }

    let wants = PositionWants {
        has_offsets,
        has_payloads,
        want_offsets,
        want_payloads,
    };
    let mut for_util_state = for_util::ForUtil::new();
    let mut block = PositionBlock::new();
    let mut cursor = SinkCursor::new();
    let mut g = 0usize;

    'stream: {
        for _ in 0..num_full_blocks {
            cursor.sync(sink, &ranges, g);
            if cursor.w == ranges.len() {
                break 'stream;
            }
            // ARITH: `g` is `k * BLOCK_SIZE` on the `k`-th (0-based) iteration
            // of a `for _ in 0..num_full_blocks`, so `block_end` here is
            // `(k + 1) * BLOCK_SIZE <= num_full_blocks * BLOCK_SIZE`, which
            // `full_blocks_and_tail` makes `<= n` -- and `n` is a `usize`.
            #[allow(clippy::arithmetic_side_effects)]
            let block_end = g + for_util::BLOCK_SIZE;
            debug_assert!(block_end <= n);
            if ranges[cursor.w].0 >= block_end {
                skip_position_block(&mut pos_r, pay_r.as_mut(), has_payloads, has_offsets)?;
                g = block_end;
                continue;
            }

            refill_full_position_block(
                &mut pos_r,
                pay_r.as_mut(),
                wants,
                &mut for_util_state,
                &mut block,
            )?;
            let mut payload_upto = 0usize;
            for i in 0..for_util::BLOCK_SIZE {
                // ARITH: `i < BLOCK_SIZE` and `g + BLOCK_SIZE == block_end`,
                // which the `debug_assert` above bounds by `n`, so this sum is
                // `< n` and cannot overflow a `usize`.
                #[allow(clippy::arithmetic_side_effects)]
                let occurrence = g + i;
                cursor.sync(sink, &ranges, occurrence);
                let payload_length = if want_payloads {
                    block.payload_lengths[i] as usize
                } else {
                    0
                };
                if cursor.open {
                    let payload = if want_payloads {
                        block.payload(payload_upto, payload_length)?
                    } else {
                        &[][..]
                    };
                    let offsets = want_offsets.then(|| {
                        (
                            block.offset_start_deltas[i] as i32,
                            block.offset_lengths[i] as i32,
                        )
                    });
                    cursor.emit(sink, block.pos_deltas[i] as i32, offsets, payload);
                }
                payload_upto = payload_upto.saturating_add(payload_length);
            }
            g = block_end;
        }

        // Vint tail block (`refillLastPositionBlock`): decoded whole, because
        // payload bytes are inlined in `.pos` right after their length and a
        // payload/offset length is only written when it *changes* from the
        // previous occurrence's (bit 0 of the vint code) -- so every
        // occurrence has to be walked even where none of them is wanted. It
        // is at most `BLOCK_SIZE - 1` occurrences, so decoding it whole
        // rather than streaming it costs nothing worth a second decoder.
        if tail_count > 0 {
            // The early exit c15 established: if every wanted document is
            // already behind us, the tail is never touched at all. Checked
            // before the refill, not after, so a walk that ends on the last
            // full block reads none of it.
            cursor.sync(sink, &ranges, g);
            if cursor.w == ranges.len() {
                break 'stream;
            }
            refill_last_position_block(&mut pos_r, wants, tail_count, &mut block)?;
            let mut payload_upto = 0usize;
            for i in 0..tail_count {
                // ARITH: after the full-block loop `g == num_full_blocks *
                // BLOCK_SIZE` and `i < tail_count == n % BLOCK_SIZE`, so this
                // sum is `< n` and cannot overflow a `usize`.
                #[allow(clippy::arithmetic_side_effects)]
                let occurrence = g + i;
                cursor.sync(sink, &ranges, occurrence);
                if cursor.w == ranges.len() {
                    break 'stream;
                }
                let payload_length = if want_payloads {
                    block.payload_lengths[i] as usize
                } else {
                    0
                };
                if cursor.open {
                    let payload = if want_payloads {
                        block.payload(payload_upto, payload_length)?
                    } else {
                        &[][..]
                    };
                    let offsets = want_offsets.then(|| {
                        (
                            block.offset_start_deltas[i] as i32,
                            block.offset_lengths[i] as i32,
                        )
                    });
                    cursor.emit(sink, block.pos_deltas[i] as i32, offsets, payload);
                }
                payload_upto = payload_upto.saturating_add(payload_length);
            }
        }
    }

    cursor.finish(sink, &ranges);
    Ok(())
}

/// The positions of just the documents `wanted` names, given as sorted indices
/// into the term's own doc list (i.e. into `freqs`).
///
/// Phrase matching only ever needs positions for documents in the
/// intersection of its terms' postings lists, which is a fraction of any one
/// term's. Building them all is wasted: on the M1 corpus `phrase t0 t1`
/// materialized every position of `t0` -- roughly 15 million, 60 MB -- to look
/// at the 2.2 million documents the intersection actually contains.
///
/// Returns `(positions, doc_starts)` addressed by position *within `wanted`*:
/// `wanted[i]`'s positions are `positions[doc_starts[i]..doc_starts[i + 1]]`,
/// and `doc_starts` has `wanted.len() + 1` entries so the last document needs
/// no special case. A `wanted` entry the doc list does not have, or one that
/// is not strictly after the entry before it, keeps its slot and yields no
/// positions (see [`wanted_ranges`]).
///
/// See [`walk_wanted_occurrences`] for what this does and does not skip on
/// the wire.
#[allow(clippy::too_many_arguments)]
pub fn read_positions_for_docs(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    freqs: &[i32],
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
    wanted: &[usize],
) -> Result<(Vec<i32>, Vec<u32>)> {
    // ARITH: `wanted` is a live slice, so `wanted.len() <= isize::MAX` and
    // the `+ 1` cannot reach `usize::MAX`. It is the caller's own document
    // list, not a count read off disk.
    #[allow(clippy::arithmetic_side_effects)]
    let doc_starts = Vec::with_capacity(wanted.len() + 1);
    let mut sink = PositionsOnly {
        positions: Vec::new(),
        doc_starts,
    };
    walk_wanted_occurrences(
        pos,
        pay,
        meta,
        freqs,
        total_term_freq,
        index_options,
        has_payloads,
        wanted,
        &mut sink,
    )?;
    sink.doc_starts.push(sink.positions.len() as u32);
    Ok((sink.positions, sink.doc_starts))
}

/// [`read_positions_for_docs`]'s offsets- and payloads-carrying sibling:
/// whole [`Position`] records for just the documents `wanted` names.
///
/// This is what `PostingsEnum.advance(doc)` + `nextPosition()` /
/// `startOffset()` / `endOffset()` / `getPayload()` gives Java's
/// `PostingsOffsetStrategy`, and it exists because the only offset-carrying
/// accessor this port had ([`read_positions`]) returns every document's
/// offsets, so highlighting one document cost a full postings sweep plus a
/// `Vec<Position>` per document in the term.
///
/// Same result shape and same `wanted` contract as
/// [`read_positions_for_docs`]: `wanted[i]`'s occurrences are
/// `occurrences[doc_starts[i]..doc_starts[i + 1]]`.
/// `start_offset`/`end_offset` are `-1` for a field that does not index
/// offsets, and `payload` is empty where an occurrence has none -- the same
/// values [`read_positions`] reports.
#[allow(clippy::too_many_arguments)]
pub fn read_occurrences_for_docs(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    freqs: &[i32],
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
    wanted: &[usize],
) -> Result<(Vec<Position>, Vec<u32>)> {
    // ARITH: `wanted` is a live slice, so `wanted.len() <= isize::MAX` and
    // the `+ 1` cannot reach `usize::MAX`. It is the caller's own document
    // list, not a count read off disk.
    #[allow(clippy::arithmetic_side_effects)]
    let doc_starts = Vec::with_capacity(wanted.len() + 1);
    let mut sink = FullOccurrences {
        occurrences: Vec::new(),
        doc_starts,
    };
    walk_wanted_occurrences(
        pos,
        pay,
        meta,
        freqs,
        total_term_freq,
        index_options,
        has_payloads,
        wanted,
        &mut sink,
    )?;
    sink.doc_starts.push(sink.occurrences.len() as u32);
    Ok((sink.occurrences, sink.doc_starts))
}

/// One document's occurrences, reached by `advance(doc_id)` over `.doc`'s
/// skip data -- Java's `PostingsOffsetStrategy.getOffsetsEnum` shape, and the
/// accessor a highlighter wants.
///
/// `Ok(None)` when the term's postings do not contain `doc_id`.
///
/// # What this costs, and why it is not what [`read_occurrences_for_docs`]
/// costs
///
/// [`read_occurrences_for_docs`] takes the term's whole frequency list and
/// addresses `.pos` by a running sum over it, so it pays a full doc-list
/// decode however few documents it is asked for. This pays neither: `.doc`'s
/// level-1 entries jump 8 192 documents at a time and its level-0 headers 256,
/// and each of those records carries the `.pos`/`.pay` file pointer its
/// documents' occurrences start at ([`LazyDocsCursor::position_origin`]). The
/// only frequencies summed are the ones in the target's own 256-document
/// block, and the only `.pos`/`.pay` bytes read are the blocks holding its
/// occurrences.
///
/// Requires `doc_freq > 1`: a singleton term has no `.doc` bytes at all
/// (`Lucene104PostingsWriter.finishTerm` pulses it into the term dictionary),
/// so there is no skip data to walk -- use [`read_occurrences_for_docs`] with
/// `wanted = [0]` for those, which is what `blocktree` does.
#[allow(clippy::too_many_arguments)]
pub fn read_occurrences_for_doc(
    doc: &DocInput<'_>,
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    doc_freq: i32,
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
    doc_id: i32,
) -> Result<Option<Vec<Position>>> {
    if !index_options.subsumes_positions() {
        return Err(Error::Unsupported(
            "positions need a field with IndexOptions::DocsAndFreqsAndPositions or higher",
        ));
    }
    let n = wire_count(total_term_freq, "total_term_freq")?;
    let mut cursor = doc.lazy_cursor(meta, doc_freq, index_options, has_payloads)?;
    if cursor.advance(doc_id)? != doc_id {
        return Ok(None);
    }
    let freq = cursor.freq().unwrap_or(0);
    if freq < 0 || freq as i64 > total_term_freq {
        return Err(corrupted(format!(
            "document {doc_id} claims frequency {freq}, which is not in \
             0..={total_term_freq} (the term's totalTermFreq)"
        )));
    }
    let origin = cursor
        .position_origin()?
        .expect("advance landed on a real document, so the cursor is positioned");
    if origin.skip > n as u64 {
        return Err(corrupted(format!(
            "the .doc skip data places document {doc_id}'s occurrences {} past a term with \
             only {n} of them",
            origin.skip
        )));
    }

    let has_offsets = index_options.subsumes_offsets();
    let wants = PositionWants {
        has_offsets,
        has_payloads,
        want_offsets: has_offsets,
        want_payloads: has_payloads,
    };
    let mut sink = FullOccurrences {
        occurrences: Vec::new(),
        doc_starts: Vec::with_capacity(2),
    };
    walk_document_occurrences(
        pos,
        pay,
        origin,
        freq as usize,
        last_pos_block_fp(meta, total_term_freq),
        n % for_util::BLOCK_SIZE,
        wants,
        &mut sink,
    )?;
    Ok(Some(sink.occurrences))
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
    let has_offsets = index_options.subsumes_offsets();
    // `wire_count`, not `as usize`; see [`read_positions_flat`].
    let n = wire_count(total_term_freq, "total_term_freq")?;
    let PositionStreams {
        pos_deltas,
        payload_lengths,
        payload_bytes,
        offset_start_deltas,
        offset_lengths,
    } = decode_position_streams(pos, pay, meta, total_term_freq, index_options, has_payloads)?;

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
        // `freq` is a `.doc` value: cap the reservation at the occurrences
        // that actually decoded, so a corrupt frequency cannot ask the
        // allocator for an arbitrary amount before the loop below rejects it.
        let mut doc_positions = Vec::with_capacity((freq.max(0) as usize).min(pos_deltas.len()));
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
            // Same `wrapping_add` rule as [`read_positions_flat`] and
            // [`SinkCursor::emit`] -- and it must be the same, or the two
            // readers disagree on exactly the corrupt segment where
            // `postings_wanted_docs.rs` asserts they agree.
            position = position.wrapping_add(pos_deltas[idx]);
            let payload = if has_payloads {
                // `payload_lengths` holds a full block's `.pay` payload-length
                // stream, `PForUtil`-decoded as `u32` and stored as `i32`: a
                // corrupt block yields a *negative* length here, and
                // `negative as usize` sign-extends to ~2^64. The old
                // `start + len` then wrapped to a value *below* `start`,
                // passed the `end > payload_bytes.len()` check, and panicked
                // in `payload_bytes[start..end]` with "slice index starts at
                // .. but ends at ..". Reject the length itself, and fold the
                // addition and the bound into the same `get`, exactly as
                // `PositionBlock::payload` already does for the walkers.
                let end = usize::try_from(payload_lengths[idx])
                    .ok()
                    .and_then(|len| payload_upto.checked_add(len));
                let Some(bytes) = end.and_then(|end| payload_bytes.get(payload_upto..end)) else {
                    return Err(corrupted(format!(
                        "payload length {} at occurrence {idx} exceeds the {} decoded \
                         payload bytes from offset {payload_upto}",
                        payload_lengths[idx],
                        payload_bytes.len()
                    )));
                };
                let payload = bytes.to_vec();
                payload_upto = end.expect("Some: the `get` above only ran on a Some end");
                payload
            } else {
                Vec::new()
            };
            let (start_offset, end_offset) = if has_offsets {
                let s = start_offset_acc.wrapping_add(offset_start_deltas[idx]);
                let e = s.wrapping_add(offset_lengths[idx]);
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
            // ARITH: the guard at the top of this loop body returned unless
            // `idx < pos_deltas.len()`, and a live slice's length is at most
            // `isize::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                idx += 1;
            }
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
struct Level1Entry<'a> {
    /// vInt delta added to the running `level1LastDocID` (starts at `-1`),
    /// giving this span's last (max) doc ID across its 32 blocks.
    doc_delta: i32,
    /// `level1DocEndFP`: absolute byte offset (into the whole `.doc` buffer)
    /// of the START of the *next* level-1 entry, i.e. one past this span's own
    /// 32 level-0 blocks. Seeking straight here skips the whole span.
    doc_end_fp: usize,
    /// The competitive `(freq, norm)` pairs covering the *whole* 32-block
    /// span (`level1CompetitiveFreqNormAccumulator`,
    /// `Lucene104PostingsWriter.java:481-489`: every level-0 block's own
    /// accumulator is merged into this one via `addAll` as each block is
    /// flushed, and it's only cleared once this level-1 entry has been
    /// written) — a strict superset/merge of any one level-0 block's impacts
    /// within the span, not just the first or last block's. Empty when the
    /// field has no freqs (`index_has_freq == false`, in which case no
    /// impacts field exists on the wire at all).
    ///
    /// Kept **undecoded**, as a zero-copy view into the mapped `.doc` bytes,
    /// exactly as [`FullBlockHeader::impact_bytes`] is and for the same
    /// reason: `skipLevel1To` decodes the run only when
    /// `needsImpacts && level1LastDocID >= target` and `skipBytes` past it
    /// otherwise, so a walk crossing 610 spans to reach a document does not
    /// decode -- or allocate -- 610 impact lists it never looks at. (b5 F7,
    /// recorded there as bounded; c20 measured it as the dominant residual of
    /// a skip-driven `advance` on a five-million-document term.)
    impact_bytes: &'a [u8],
    /// The span's `.pos`/`.pay` skip pointers, or `None` when the field does
    /// not index positions (in which case the sub-fields do not exist on the
    /// wire). See [`PosSkip`].
    pos_skip: Option<PosSkip>,
}

/// One level-0 or level-1 record's `.pos`/`.pay` skip pointers:
/// `posEndFPDelta`/`posBufferUpto` and, when the field also has offsets or
/// payloads, `payEndFPDelta`/`payloadByteUpto`
/// (`Lucene104PostingsReader.readLevel0PosData` and the identical block
/// inside `skipLevel1To`).
///
/// Every field is a delta or an in-block offset, never an absolute position:
/// a reader accumulates `pos_end_fp_delta` across every record it passes from
/// the term's `posStartFP` onward, which is why a skip that jumps a level-1
/// span has to take the span's own accumulated value rather than re-deriving
/// one.
#[derive(Debug, Clone, Copy, Default)]
struct PosSkip {
    /// Added to the running `.pos` pointer. A vlong off disk, so it is
    /// accumulated with `wrapping_add` and only validated when the resulting
    /// pointer is actually seeked to.
    pos_end_fp_delta: u64,
    /// `posBufferUpto`: how many occurrences of the 256-wide `.pos` block at
    /// the new pointer belong to *earlier* documents. A single byte on the
    /// wire, so it is `0..=255` however corrupt the file is -- it can never
    /// index past a 256-entry block.
    pos_buffer_upto: u8,
    /// Added to the running `.pay` pointer; `0` when the field has neither
    /// offsets nor payloads (the sub-field does not exist on the wire).
    pay_end_fp_delta: u64,
}

/// Reads one level-1 skip entry at `r`'s current position (which must be a
/// level-1 entry boundary — `docStartFP` for span 0, or the previous span's
/// `doc_end_fp` afterward), leaving `r` positioned at the first level-0 block
/// header of the span. Shared by [`DocInput::read_postings`] (eager, discards
/// the return) and [`LazyDocsCursor::skip_level1_to`] (lazy, uses `doc_delta`
/// and `doc_end_fp` to decide whether to jump the whole span).
///
/// The impacts bytes (competitive-scoring metadata for the whole span) are
/// decoded into [`Level1Entry::impacts`] via [`decode_impacts`]. The pos/pay
/// sub-fields are parsed for wire-order correctness even though this reader
/// never uses them to seek `.pos`/`.pay`.
fn read_level1_entry<'a>(
    r: &mut SliceInput<'a>,
    index_has_freq: bool,
    index_has_pos: bool,
    index_has_offsets_or_payloads: bool,
) -> Result<Level1Entry<'a>> {
    // Steps 1-2 are always present, even for `IndexOptions::Docs` (no freqs):
    // `skipLevel1To` calls `readVInt`/`readVLong` unconditionally before the
    // `indexHasFreq` gate — plain vint/vlong, NOT `readVInt15`/`readVLong15`.
    let doc_delta = r.read_vint()?;
    let delta = r.read_vlong()? as usize;
    let doc_end_fp = delta.wrapping_add(r.position());

    let mut impact_bytes: &[u8] = &[];
    let mut pos_skip: Option<PosSkip> = None;
    if index_has_freq {
        // `skip1EndFP` (step 3a): a plain 2-byte `readShort`, the byte length
        // from right here to the end of this entry's metadata. Only used as a
        // consistency check (Java asserts `getFilePointer() == skip1EndFP`).
        //
        // A negative `short` (or a position near the end of a huge file) is
        // reachable off disk, so this is folded through `checked_add` and
        // `usize::try_from` rather than `as usize`, which would turn a
        // negative offset into ~2^64 and merely *look* like a mismatch.
        let Some(skip1_end_fp) = (r.read_i16()? as i64)
            .checked_add(r.position() as i64)
            .and_then(|fp| usize::try_from(fp).ok())
        else {
            return Err(corrupted(
                "level-1 skip entry: skip1EndFP is not a representable file offset",
            ));
        };
        // `numImpactBytes` (step 3b): another plain `readShort`, non-negative
        // (a length). Then decode that many raw impact bytes (step 3c).
        //
        // Signed on the wire, so a corrupt `.doc` can put a negative here.
        // `as usize` would sign-extend that to ~2^64 and the `impact_start +`
        // below would overflow -- a panic in a debug build, which through the
        // FFI is a dead JVM. Java gets a `NegativeArraySizeException`; this
        // port reports the corruption it is.
        let num_impact_bytes = wire_length(r.read_i16()? as i64, "level-1 impacts")?;
        let impact_start = r.position();
        impact_bytes = r.slice(
            impact_start,
            add_wire_offset(impact_start, num_impact_bytes)?,
        )?;
        r.skip(num_impact_bytes)?;
        if index_has_pos {
            pos_skip = Some(read_pos_skip(r, index_has_offsets_or_payloads)?);
        }
        check_wire_position(r.position(), skip1_end_fp, "level-1 skip entry")?;
    }

    Ok(Level1Entry {
        doc_delta,
        doc_end_fp,
        impact_bytes,
        pos_skip,
    })
}

/// `Lucene104PostingsReader.readLevel0PosData` (and the byte-identical block
/// inside `skipLevel1To`): the `.pos`/`.pay` skip sub-fields of one level-0
/// header or level-1 entry.
///
/// `payloadByteUpto` is read and discarded on purpose, and it is the one
/// field here that is genuinely not needed -- **on every path**, not just
/// most. Java stores it (`seekPosData`'s `payloadByteUpto = payUpto`) and
/// then overwrites it before it can be read:
///
/// - `seekPosData` also sets `posBufferUpto = BLOCK_SIZE`, so `skipPositions`'
///   `leftInBlock` is `0` and it always takes the `else` branch, which ends
///   `payloadByteUpto = sumOverRange(payloadLengthBuffer, 0, toSkip)`;
/// - and when `toSkip == 0` `skipPositions` does not run at all, but
///   `nextPosition` then refills, and both `refillOffsetsOrPayloads` and
///   `refillLastPositionBlock` assign `payloadByteUpto = 0`.
///
/// So recomputing it from the landing block's own decoded lengths is not an
/// approximation of Java, it is the same number by a route that does not let
/// a file-derived value index a byte run. It is still *parsed*, because the
/// bytes are there and nothing after them is self-delimiting.
fn read_pos_skip(r: &mut SliceInput, index_has_offsets_or_payloads: bool) -> Result<PosSkip> {
    let pos_end_fp_delta = r.read_vlong()? as u64;
    let pos_buffer_upto = r.read_byte()?;
    let mut pay_end_fp_delta = 0u64;
    if index_has_offsets_or_payloads {
        pay_end_fp_delta = r.read_vlong()? as u64;
        let _pay_buffer_upto = r.read_vint()?;
    }
    Ok(PosSkip {
        pos_end_fp_delta,
        pos_buffer_upto,
        pay_end_fp_delta,
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
/// field here is a small fixed-width or vint/vlong-prefixed value, including
/// the impacts byte run (captured undecoded as
/// [`FullBlockHeader::impact_bytes`], a borrow of the mapped file, and decoded
/// only if a caller asks for a bound) — so determining `last_doc_id` and `body_start`/`body_len` never runs
/// `ForUtil`/`PForUtil` decode, which is the expensive part of a block
/// (bit-unpacking 256 values). That decode work is exactly what
/// [`LazyDocsCursor`] avoids for a block this header proves is entirely
/// before the caller's target.
struct FullBlockHeader<'a> {
    /// This block's last (highest) doc ID — `prev_doc_id + docDelta`, proven
    /// consistent with the body's own delta-decoded last entry by every
    /// existing fixture/unit test that decodes both (see `read_full_block_header`).
    last_doc_id: i32,
    /// Byte offset (into the same buffer `r` reads from) where the block's
    /// body (`bitsPerValue` token onward) begins.
    body_start: usize,
    /// Byte offset where the block's body ends, i.e. where the next block's
    /// own level-0 header (or the tail block, or the term's end) begins.
    body_end: usize,
    /// The competitive `(freq, norm)` pairs for *this one* level-0 block
    /// (`level0FreqNormAccumulator.getCompetitiveFreqNormPairs()`,
    /// `Lucene104PostingsWriter.java:397-402`) — reset to empty after every
    /// block is flushed, unlike the level-1 span's merged accumulator. Empty
    /// when the field has no freqs.
    ///
    /// Kept **undecoded**, as a zero-copy view into the mapped `.doc` bytes.
    /// Lucene holds the same run as a `BytesRef` (`level0SerializedImpacts`)
    /// and calls `readImpacts` only when `getImpacts()` asks for it. Decoding
    /// it here instead cost 9.75% of a term query's profile plus its allocator
    /// traffic, most of that on blocks the cursor went on to skip without ever
    /// looking at a bound.
    impact_bytes: &'a [u8],
    /// This block's `.pos`/`.pay` skip pointers, or `None` when the field
    /// does not index positions (or has no freqs, in which case the whole
    /// freq-gated region including these is absent). See [`PosSkip`].
    pos_skip: Option<PosSkip>,
}

/// Reads one full block's level-0 header (see [`FullBlockHeader`]) without
/// touching the body. `r` is left positioned at `body_start` on return.
fn read_full_block_header<'a>(
    r: &mut SliceInput<'a>,
    prev_doc_id: i32,
    index_has_freq: bool,
    index_has_pos: bool,
    index_has_offsets_or_payloads: bool,
) -> Result<FullBlockHeader<'a>> {
    // `level0NumBytes` (`numSkipBytes` on the write side): the byte length,
    // from right here, of everything up to the block *body* -- the two
    // `vint15`/`vlong15` header fields, the impacts run, and the pos/pay skip
    // sub-fields. `skipLevel0To` uses it to `seek` straight past a block's
    // metadata when it wants neither impacts nor positions; this reader
    // computes the same position by parsing, and so can *check* the two
    // agree.
    //
    // That check earns its keep now that the region it spans contains two
    // variable-width sub-fields whose very presence is decided by
    // `index_has_pos`/`index_has_offsets_or_payloads` -- values that come from
    // `FieldInfos`, not from this stream. A field opened with the wrong
    // `has_payloads`, or a corrupt vlong, otherwise mis-frames `body_start`
    // and the body decodes plausible garbage with no error at all. It is the
    // level-0 twin of `read_level1_entry`'s `skip1EndFP` check.
    let level0_num_bytes = r.read_vlong()?;
    let skip0_end = add_wire_offset(
        r.position(),
        wire_length(level0_num_bytes, "level-0 skip header")?,
    )?;
    let doc_delta = read_vint15(r)?;
    // `read_vint15` off disk: a corrupt `.doc` overflows the running doc id,
    // which is a debug-build panic. The claim is checked against the block
    // body anyway (`advance`'s post-refill search, and `check_wire_position`),
    // so a wrapped value is caught there rather than here.
    let last_doc_id = prev_doc_id.wrapping_add(doc_delta);
    let block_length = read_vlong15(r)?;
    // `level0DocEndFP` in `Lucene104PostingsReader.doMoveToNextLevel0Block`
    // (`Lucene104PostingsReader.java:743-744`) is computed *immediately*
    // after reading `blockLength`, i.e. before the impacts/pos/pay fields
    // are read -- `blockLength` therefore measures from here (not from
    // `body_start` below) through the end of the whole block, so it
    // includes the impacts-length-prefixed bytes and pos/pay skip fields,
    // not just the `bitsPerValue`-onward body.
    // `block_length` is a `vlong15` off disk: negative or absurd values are
    // reachable on a corrupt `.doc`, and `position() + (negative as usize)`
    // overflows -- a panic in a debug build. A bad-but-representable value
    // still ends up out of range, and the `seek`/`slice` that uses `body_end`
    // reports that as EOF.
    let body_end = add_wire_offset(r.position(), wire_length(block_length, "level-0 block")?)?;
    let mut impact_bytes: &[u8] = &[];
    let mut pos_skip: Option<PosSkip> = None;
    if index_has_freq {
        // Impacts byte-length is a plain vint here (`doMoveToNextLevel0Block`,
        // `Lucene104PostingsReader.java:746`), unlike level-1's vlong-prefixed
        // `numSkipBytes` -- confirmed against the reader source rather than
        // assumed from the tail-block/level-1 shape.
        // `read_vint` decodes a negative value from a five-byte varint with
        // the sign bit set, so this length is not trustworthy without a
        // check: `read_length` rejects negatives and anything longer than the
        // bytes actually left, which is what keeps `impacts_start +` below
        // from overflowing (a debug-build panic) on a corrupt `.doc`.
        let impacts_len = r.read_length("level-0 impacts")?;
        let impacts_start = r.position();
        impact_bytes = r.slice(impacts_start, add_wire_offset(impacts_start, impacts_len)?)?;
        r.skip(impacts_len)?;

        // Level-0 pos/pay skip data (`Lucene104PostingsReader.java:754-761`,
        // `readLevel0PosData`): where in `.pos`/`.pay` this block's documents'
        // occurrences begin. [`LazyDocsCursor`] accumulates it so a
        // positional walk can seek straight there instead of summing every
        // preceding document's frequency.
        if index_has_pos {
            pos_skip = Some(read_pos_skip(r, index_has_offsets_or_payloads)?);
        }
    }

    let body_start = r.position();
    check_wire_position(body_start, skip0_end, "level-0 skip header")?;
    Ok(FullBlockHeader {
        last_doc_id,
        body_start,
        body_end,
        impact_bytes,
        pos_skip,
    })
}

/// The buffers a full-block decode needs but does not produce: the packed
/// word array `ForUtil` unpacks into, the bit-set words the dense doc encoding
/// reads, and the `ForUtil` scratch itself.
///
/// These were locals inside `decode_full_block_body`, so a 256-doc block paid
/// for roughly 8 KiB of zeroed and copied stack -- of which 2 KiB was the
/// output -- plus a heap allocation on the dense path. Lucene's
/// `BlockPostingsEnum` holds every one of these as an instance field and
/// reuses them for the whole enumeration; a cursor walking a long posting list
/// should do the same. Callers that decode one block still just construct one.
#[derive(Debug)]
struct BlockScratch {
    for_util: ForUtil,
    /// `ForUtil`/`PForUtil` output: doc deltas, then reused for freqs.
    words: [u32; for_util::BLOCK_SIZE],
    /// Dense doc-encoding bit set. `-bitsPerValue` is read from an `i8`, so
    /// `numLongs` can never exceed 128 whatever the file says.
    bitset: [u64; 128],
}

impl BlockScratch {
    fn new() -> Self {
        Self {
            for_util: ForUtil::new(),
            words: [0u32; for_util::BLOCK_SIZE],
            bitset: [0u64; 128],
        }
    }
}

/// Test-only instrumentation counting full block *body* decodes -- the
/// `ForUtil`/`PForUtil` bit-unpack of 256 documents.
///
/// Exists to separate two things a profile conflates: blocks whose *header*
/// was read to make a skip decision, and blocks whose body was actually
/// unpacked. Lucene draws that line explicitly -- `advanceShallow` moves the
/// impacts forward without touching the body, and only `advance` decodes -- so
/// a port that decodes every block it merely considers is doing work Lucene
/// never does, and this counter is how that shows up as a number rather than
/// as a suspicion.
#[cfg(any(test, feature = "test-support"))]
pub mod test_only_block_decode_counter {
    use std::cell::Cell;

    thread_local! {
        static DECODES: Cell<u64> = const { Cell::new(0) };
    }

    pub fn record_decode() {
        DECODES.with(|c| c.set(c.get().wrapping_add(1)));
    }

    pub fn reset() {
        DECODES.with(|c| c.set(0));
    }

    pub fn count() -> u64 {
        DECODES.with(|c| c.get())
    }
}

/// `Error::Store(Corrupted)` with `msg` -- the one thing every wire-level
/// disagreement in this module returns, spelled once.
fn corrupted(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// A byte length read off disk, validated before it is used to size or offset
/// anything.
///
/// The `.doc`/`.pos`/`.pay` streams write lengths as `readShort`, `readVInt`
/// and `readVLong15`, all of which can decode to a **negative** number from
/// corrupt bytes. `negative as usize` sign-extends to roughly `2^64`, and the
/// `base + len` that follows then overflows -- which in a debug build is a
/// panic, and a panic in a debug build of the FFI takes the JVM down with it
/// (`c8`'s finding 16 and the lazy-cursor tail block are the same class).
/// Java's readers get an exception out of the negative array size instead.
fn wire_length(value: i64, what: &str) -> Result<usize> {
    if value < 0 {
        return Err(corrupted(format!("{what}: negative byte length {value}")));
    }
    Ok(value as usize)
}

/// `base + len` where `len` came off disk: an overflow is corruption, not an
/// arithmetic bug of ours, so it is reported rather than panicked on. The
/// resulting offset is still bounds-checked by whatever `slice`/`seek` uses
/// it -- this only guarantees the addition itself is answerable.
fn add_wire_offset(base: usize, len: usize) -> Result<usize> {
    base.checked_add(len).ok_or_else(|| {
        corrupted(format!(
            "byte length {len} overflows the file offset {base}"
        ))
    })
}

/// A count of documents or occurrences read off disk, as a `usize` that is
/// safe to compare, divide and size with.
///
/// The two rejections are different in kind and are reported as such. A
/// negative count is corruption: no writer emits one. A count above
/// `u32::MAX` is merely past **this port's** ceiling -- `totalTermFreq` is a
/// `long` in Lucene, and a stop-word in a segment of hundreds of millions of
/// documents really can exceed 2^32 occurrences -- because the flat position
/// streams here index with `u32`. That is a limitation to name, not a damaged
/// file to report.
fn wire_count(value: i64, what: &str) -> Result<usize> {
    if value < 0 {
        return Err(corrupted(format!("{what}: negative count {value}")));
    }
    if value > u32::MAX as i64 {
        return Err(Error::Unsupported(
            "total_term_freq exceeds u32::MAX: this port's flat position streams are indexed \
             with u32, so a term with more than 2^32 occurrences cannot be walked",
        ));
    }
    Ok(value as usize)
}

/// Splits `n` occurrences into whole 256-wide `PForUtil` blocks plus the
/// `refillLastPositionBlock` vint tail -- `totalTermFreq / BLOCK_SIZE` and
/// `totalTermFreq % BLOCK_SIZE`, the two numbers every `.pos` walk starts
/// from.
// ARITH: `BLOCK_SIZE` is the compile-time constant 256, so neither the
// division nor the remainder can divide by zero, and neither can overflow
// (`usize / c` and `usize % c` are both bounded by `n`).
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn full_blocks_and_tail(n: usize) -> (usize, usize) {
    (n / BLOCK_SIZE as usize, n % BLOCK_SIZE as usize)
}

/// Checks that a decode landed exactly where a **length field read off the
/// same file** said it would, and returns a decode error if it did not.
///
/// The `.doc` stream records two such lengths redundantly: a level-0 header's
/// `level0NumBytes` (the byte length of the block body that follows) and a
/// level-1 entry's `skip1EndFP` (the byte length of that entry's own
/// metadata). Neither is derivable from the data it measures, so on a corrupt
/// file the decode ends somewhere else, and every byte after it is garbage.
///
/// Java asserts both (`assert docIn.getFilePointer() == blockEndFP`,
/// `assert docIn.getFilePointer() == skip1EndFP`), which is a no-op without
/// `-ea`. This port had `debug_assert_eq!`, which is a no-op in release and a
/// *panic* in debug — neither is what the rest of this module does with a
/// corrupt file, and a panic in a debug build of the FFI takes the JVM down.
/// `debug_assert` is for invariants this code's own arithmetic guarantees;
/// these are values read off disk. Both are hard decode errors now.
/// (`c9-check-index`'s byte-flipping sweep reaches the first one.)
fn check_wire_position(position: usize, expected: usize, what: &str) -> Result<()> {
    if position != expected {
        return Err(corrupted(format!(
            "{what}: decode ended at {position} but the file's own length field claims {expected}"
        )));
    }
    Ok(())
}

/// Decodes a full block's body (the `bitsPerValue` token onward) — `r` must
/// already be positioned at [`FullBlockHeader::body_start`]. Shared by
/// [`DocInput::read_postings`] (eager path) and [`LazyDocsCursor`] (lazy path) so
/// there is exactly one body decoder to keep in sync with `ForUtil`/
/// `PForUtil`.
fn decode_full_block_body(
    r: &mut SliceInput,
    prev_doc_id: i32,
    index_has_freq: bool,
    needs_freq: bool,
    scratch: &mut BlockScratch,
    docs: &mut [i32; BLOCK_SIZE as usize],
    freqs: &mut [i32; BLOCK_SIZE as usize],
) -> Result<()> {
    #[cfg(any(test, feature = "test-support"))]
    test_only_block_decode_counter::record_decode();
    let bits_per_value_byte = r.read_byte()? as i8;
    if bits_per_value_byte > 0 {
        let doc_deltas = &mut scratch.words;
        scratch
            .for_util
            .decode(bits_per_value_byte as u32, r, doc_deltas)?;
        // ARITH: `sum` starts inside `i32`'s range and gains at most
        // `u32::MAX` per iteration over the fixed 256-entry `doc_deltas`, so
        // `|sum| < 2^31 + 256 * 2^32 < 2^41` -- three orders of magnitude
        // inside `i64`. The hottest loop in the file; no per-doc check.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let mut sum: i64 = prev_doc_id as i64;
            for (d, &delta) in docs.iter_mut().zip(doc_deltas.iter()) {
                sum += delta as i64;
                *d = sum as i32;
            }
        }
    } else if bits_per_value_byte == 0 {
        // "0 is used to record that all 256 docs in the block are
        // consecutive" (`Lucene104PostingsReader.refillFullBlock`): every
        // delta is 1, no bytes follow.
        for (i, d) in docs.iter_mut().enumerate() {
            // `prev_doc_id` descends from file deltas; `wrapping_add` for the
            // same reason every other accumulator in this file uses it.
            //
            // ARITH: `docs` is `[i32; BLOCK_SIZE]`, so `i < 256` and
            // `1 + i as i32 <= 256`.
            #[allow(clippy::arithmetic_side_effects)]
            let delta = 1 + i as i32;
            *d = prev_doc_id.wrapping_add(delta);
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
        // ARITH: this branch is `bits_per_value_byte < 0` on an `i8`, so the
        // widened value is in `-128..=-1` and its negation in `1..=128` --
        // `i32::MIN`, the one input unary `-` cannot answer, is unreachable.
        #[allow(clippy::arithmetic_side_effects)]
        let num_longs = (-(bits_per_value_byte as i32)) as usize;
        // `bits_per_value_byte` is an `i8`, so `num_longs <= 128` however
        // corrupt the file is -- which is also Lucene's own bound
        // (`assert numBitSetLongs <= BLOCK_SIZE / 2`). A fixed scratch array
        // is therefore always large enough, and this path stops allocating.
        let words = &mut scratch.bitset[..num_longs];
        for w in words.iter_mut() {
            *w = r.read_i64()? as u64;
        }
        // ARITH: `prev_doc_id` is an `i32`, so `+ 1` is exact in `i64`;
        // `word_idx < num_longs <= 128` so `word_idx * 64 <= 8128`; `bit` is
        // a `trailing_zeros` of a non-zero `u64`, so `0..=63`; and the
        // running sum stays under `2^31 + 8191 < 2^32`. `found` is capped at
        // `BLOCK_SIZE` by the check inside the loop, which is also what keeps
        // `docs[found]` in bounds. `bits - 1` is guarded by `bits != 0`.
        #[allow(clippy::arithmetic_side_effects)]
        let found = {
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
            found
        };
        if found != BLOCK_SIZE as usize {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "full-block doc bit-set has fewer than BLOCK_SIZE set bits".into(),
            )));
        }
    }

    if index_has_freq && needs_freq {
        let freq_words = &mut scratch.words;
        scratch.for_util.pfor_decode(r, freq_words)?;
        for (f, &w) in freqs.iter_mut().zip(freq_words.iter()) {
            *f = w as i32;
        }
    } else {
        if index_has_freq {
            // `Lucene104PostingsReader.refillFullBlock`'s `needsFreq ==
            // false` branch: the freq block is still *there* on the wire, so
            // it has to be stepped over -- one token byte and a seek, versus
            // a 256-value `PForUtil` unpack.
            for_util::pfor_skip(r)?;
        }
        // A field without freqs -- or a consumer that did not ask for them
        // ([`PostingsFlags::DocsOnly`]) -- scores every occurrence as 1.
        // Lucene fills `freqBuffer` the same way rather than branching per
        // doc downstream; where it instead defers via `freqFP`, this port
        // makes "not requested" mean "not available", which is exactly the
        // `PostingsEnum.NONE`/`DOCS` contract.
        freqs.fill(1);
    }

    Ok(())
}

/// The `docFreq % BLOCK_SIZE` remainder after zero or more full blocks
/// (`BlockPostingsEnum.refillRemainder`'s non-singleton branch): the same
/// group-varint + trailing-vint-freq-exceptions scheme the pre-existing
/// single-block (`docFreq < BLOCK_SIZE`) path already implements, just with
/// `prev_doc_id` seeded from the last full block instead of always `-1`.
/// Decodes into caller-owned `docs`/`freqs` slices, whose common length is
/// the block's doc count. Writing straight into the destination is what
/// `Lucene104PostingsReader.refillRemainder` does too -- it decodes into the
/// enumeration's own `docBuffer`/`freqBuffer` -- and it matters here because a
/// term with `docFreq < BLOCK_SIZE` is *entirely* a tail block, so a cursor
/// over one of those (the common shape by a wide margin) used to pay two
/// `Vec` allocations plus two 256-entry copies to hand the result back into
/// the buffers it already owned.
fn read_tail_block(
    r: &mut SliceInput,
    prev_doc_id: i32,
    index_has_freq: bool,
    needs_freq: bool,
    docs: &mut [i32],
    freqs: &mut [i32],
) -> Result<()> {
    let count = docs.len();
    debug_assert_eq!(freqs.len(), count);
    if count >= BLOCK_SIZE as usize {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "tail block must hold fewer than BLOCK_SIZE docs".into(),
        )));
    }
    // On the stack, not the heap: a tail block is by definition shorter than
    // `BLOCK_SIZE`, and this is the only decode a `docFreq < BLOCK_SIZE` term
    // ever does, so a per-call allocation here is a per-*term* allocation.
    // Lucene reads group-varints straight into its long-lived `docBuffer`.
    let mut raw = [0u64; BLOCK_SIZE as usize];
    let raw = &mut raw[..count];
    r.read_group_vints(raw)?;

    if index_has_freq && needs_freq {
        for ((d, f), &v) in docs.iter_mut().zip(freqs.iter_mut()).zip(raw.iter()) {
            *f = (v & 1) as i32;
            *d = (v >> 1) as i32;
        }
        for f in freqs.iter_mut() {
            if *f == 0 {
                *f = r.read_vint()?;
            }
        }
    } else if index_has_freq {
        // `PostingsUtil.readVIntBlock`'s `decodeFreq == false` branch: the
        // low bit is still the freq flag, so the doc deltas need shifting,
        // but the trailing freq-exception vints are never read. Safe because
        // the tail block is the last thing a term writes to `.doc`.
        for (d, &v) in docs.iter_mut().zip(raw.iter()) {
            *d = (v >> 1) as i32;
        }
        freqs.fill(1);
    } else {
        for (d, &v) in docs.iter_mut().zip(raw.iter()) {
            *d = v as i32;
        }
        // A field without freqs scores every occurrence as 1, same as
        // `decode_full_block_body`'s own no-freq branch.
        freqs.fill(1);
    }

    // ARITH: `sum` starts inside `i32`'s range and gains an `i32` per
    // iteration over a slice the `count >= BLOCK_SIZE` guard above caps at
    // 255 entries, so `|sum| < 2^31 * 256 < 2^39`.
    #[allow(clippy::arithmetic_side_effects)]
    {
        let mut sum: i64 = prev_doc_id as i64;
        for d in docs.iter_mut() {
            sum += *d as i64;
            *d = sum as i32;
        }
    }

    Ok(())
}

/// `GroupVIntUtil.writeGroupVInts`'s wire format (groups of 4 values, one
/// flag byte packing each value's byte-length minus one, then that many
/// little-endian bytes per value; a final partial group of fewer than 4
/// falls back to plain vints) — the write-side companion to
/// [`DataInput::read_group_vints`], needed by [`crate::postings_writer`]'s
/// tail-block encoder ([`read_tail_block`]'s exact inverse).
// ARITH: `values` is a live slice, so `values.len() <= isize::MAX` and
// `i <= values.len()` throughout (it only ever advances to a bound the loop
// condition just proved), which keeps `i + 4` and `i + 1` inside a `usize`.
// `v.leading_zeros() / 8 <= 3` on the `v != 0` branch, so `4 - ..` is `>= 1`
// and `bytes - 1` is `>= 0`; `lens[j] <= 3`, so `lens[j] + 1 <= 4`. Nothing
// here comes off disk -- this is the encoder `postings_writer` calls with
// values it produced.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn write_group_vints(out: &mut impl DataOutput, values: &[u32]) {
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
        out.write_byte(flag);
        for (j, &v) in chunk.iter().enumerate() {
            let n = lens[j] as usize + 1;
            out.write_bytes(&v.to_le_bytes()[..n]);
        }
        i += 4;
    }
    while i < values.len() {
        out.write_vint(values[i] as i32);
        i += 1;
    }
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
        // No `.doc` bytes at all for a singleton -- no impacts to report,
        // same as `PostingsCursor::level0_impacts`/`level1_impacts`'s
        // "no covering entry" empty-slice contract.
        level0_impacts: Vec::new(),
        level1_impacts: Vec::new(),
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

    /// `ImpactsEnum.getImpacts()`'s level-0 result, conceptually — same
    /// meaning as [`LazyDocsCursor::level0_impacts`]: the competitive
    /// `(freq, norm)` pairs for the level-0 block currently covering
    /// `doc_id()`. Empty before the first `next_doc()`/`advance()` call, once
    /// exhausted, when `doc_id()` falls in the trailing tail block (no
    /// level-0 impacts exist on the wire for it), or for a field with no
    /// freqs. Looks up [`Postings::level0_impacts`] (captured once, up front,
    /// by [`DocInput::read_postings`]) rather than decoding anything itself.
    pub fn level0_impacts(&self) -> &[Impact] {
        if !self.started || self.idx >= self.postings.docs.len() {
            return &[];
        }
        find_impacts(&self.postings.level0_impacts, self.postings.docs[self.idx])
    }

    /// `ImpactsEnum.getImpacts()`'s level-1 result, conceptually — same
    /// meaning as [`LazyDocsCursor::level1_impacts`]: the competitive
    /// `(freq, norm)` pairs merged across the whole 32-block level-1 span
    /// currently covering `doc_id()`. Empty below `LEVEL1_NUM_DOCS` docs, for
    /// a field with no freqs, before the first `next_doc()`/`advance()` call,
    /// once exhausted, or once `doc_id()` is past the last level-1 span (the
    /// trailing sub-8192 region of full blocks + tail).
    pub fn level1_impacts(&self) -> &[Impact] {
        if !self.started || self.idx >= self.postings.docs.len() {
            return &[];
        }
        find_impacts(&self.postings.level1_impacts, self.postings.docs[self.idx])
    }

    /// `PostingsEnum.nextDoc()`: moves to the next doc, returning its ID (or
    /// [`NO_MORE_DOCS`] if there isn't one).
    pub fn next_doc(&mut self) -> i32 {
        if !self.started {
            self.started = true;
            // idx is already 0 (the first doc, if any).
        } else if self.idx < self.postings.docs.len() {
            // ARITH: guarded by the `else if` -- a live `Vec`'s length is at
            // most `isize::MAX`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.idx += 1;
            }
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
        // ARITH: `partition_point` over `docs[start..]` returns at most that
        // sub-slice's length, so `start + offset <= docs.len()`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.idx = start + offset;
        }
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
    /// `needsFreq`: whether the consumer asked for frequencies
    /// ([`PostingsFlags`]). When false the frequency block of every refilled
    /// level-0 block is stepped over rather than unpacked, and every
    /// [`Self::freq`] answers `1`.
    needs_freq: bool,
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
    /// The most recently decoded level-0 block's competitive impacts
    /// (`ImpactsEnum.getImpacts()` level `0`, conceptually) — empty for the
    /// tail block (no impacts on the wire there at all) or a field with no
    /// freqs. Valid for whichever block `block_docs`/`block_freqs` currently
    /// hold, i.e. the block containing `doc_id`.
    level0_impacts: Impacts,
    /// The current level-1 span's merged competitive impacts (`getImpacts()`
    /// level `1`, conceptually) — empty below `LEVEL1_NUM_DOCS` docs (no
    /// level-1 entries on the wire) or for a field with no freqs.
    level1_impacts: Impacts,
    /// Decode buffers, held for the life of the cursor rather than rebuilt per
    /// block — see [`BlockScratch`].
    scratch: BlockScratch,
    /// A block whose level-0 header and impacts have been read but whose body
    /// has **not** been bit-unpacked — `Lucene104PostingsReader`'s
    /// `needsRefilling` state, reached through
    /// [`LazyDocsCursor::advance_shallow`].
    pending: Option<PendingBlock>,
    /// `level0LastDocID`: the highest doc ID of the block this cursor is
    /// currently positioned on, whether that block is merely shallow-positioned
    /// or fully decoded. [`NO_MORE_DOCS`] in the tail block (which carries no
    /// header, so its extent is not known without decoding) and once exhausted;
    /// `-1` before the first move.
    level0_last_doc_id: i32,
    /// `level0PosEndFP`/`level0BlockPosUpto`/`level0PayEndFP`: the running
    /// `.pos`/`.pay` state at the *end* of the last level-0 header this
    /// cursor read, i.e. at the start of the block after it. Seeded from the
    /// term's `posStartFP`/`payStartFP`, which is the state at the start of
    /// block 0.
    level0_pos: PosCursorState,
    /// `level1PosEndFP`/`level1BlockPosUpto`/`level1PayEndFP`: the same at
    /// the end of the last level-1 span entry read.
    /// [`Self::skip_level1_to`] copies it into `level0_pos` at the top of
    /// every iteration, exactly as `skipLevel1To` does.
    level1_pos: PosCursorState,
    /// The snapshot of `level0_pos` taken *before* the header of the block
    /// the cursor is currently positioned on -- Java's `posFP`/`posUpto`/
    /// `payFP` locals in `skipLevel0To`, which is what `seekPosData` is then
    /// handed. This is the `.pos`/`.pay` origin of the current block's
    /// documents.
    block_pos_origin: PosCursorState,
}

/// A running `.pos`/`.pay` position: the absolute file pointers plus how many
/// occurrences of the `.pos` block at `pos_fp` belong to earlier documents.
///
/// The `.doc` file carries these as deltas at every level-0 block header and
/// level-1 span entry; a cursor accumulates them. `pos_fp`/`pay_fp` are
/// `wrapping_add`ed from file values and only validated when a seek actually
/// uses them.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
struct PosCursorState {
    pos_fp: u64,
    pay_fp: u64,
    /// `0..=255`: a single wire byte, so it can never index past a block.
    pos_buffer_upto: u8,
}

impl PosCursorState {
    /// `level0PosEndFP += docIn.readVLong(); level0BlockPosUpto =
    /// docIn.readByte() & 0xFF;` and the `.pay` pair -- one skip record's
    /// worth of movement.
    ///
    /// `wrapping_add`: both deltas are file values, and the only thing that
    /// can be said about a corrupt one is that the seek it eventually feeds
    /// will fail. Adding them with `+` would instead panic in a debug build,
    /// which through the FFI is a dead JVM with no exception to catch.
    #[inline]
    fn advance(&mut self, skip: PosSkip) {
        self.pos_fp = self.pos_fp.wrapping_add(skip.pos_end_fp_delta);
        self.pay_fp = self.pay_fp.wrapping_add(skip.pay_end_fp_delta);
        self.pos_buffer_upto = skip.pos_buffer_upto;
    }
}

/// Where in `.pos`/`.pay` one document's occurrences begin, as
/// `PostingsEnum.advance(doc)` leaves it.
///
/// This is `Lucene104PostingsReader`'s post-`seekPosData` state for one
/// document: seek `.pos` to `pos_fp` and `.pay` to `pay_fp`, step over `skip`
/// occurrences, and the next occurrence read is the document's first. `skip`
/// is `posBufferUpto` from the skip data plus the frequencies of the
/// documents ahead of the target *within its own level-0 block* -- Java's
/// `accumulatePendingPositions`/`skipPositions` pair, which never looks at a
/// frequency outside the current block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PositionOrigin {
    pub pos_fp: u64,
    pub pay_fp: u64,
    pub skip: u64,
}

/// `VectorUtil.findNextGEQ`: index of the first entry `>= target` in an
/// ascending buffer, or `buf.len()` if there is none.
///
/// A **linear scan**, which is what Lucene uses and is not the obvious choice --
/// `partition_point` is a binary search over a sorted slice, which is what this
/// port did. Over a 256-entry block that is eight unpredictable branches, on the
/// order of 120 cycles once mispredictions are counted, where scoring the
/// document costs about ten. The linear scan's branch is taken every iteration
/// until the one that ends it, so it predicts perfectly and vectorizes, and the
/// distances are short in practice.
///
/// It is why block-max pruning kept measuring as a *regression* here: three
/// separate attempts at MAXSCORE and WAND all skipped real work and all came out
/// slower, because every skip went through `advance` and paid more than the
/// scoring it avoided.
#[inline]
fn find_next_geq(buf: &[i32], target: i32) -> usize {
    buf.iter().position(|&d| d >= target).unwrap_or(buf.len())
}

/// A level-0 block positioned but not decoded: everything
/// [`LazyDocsCursor::refill`] needs to unpack it later, and nothing more.
///
/// This is the whole point of `advanceShallow`. Deciding whether a block can
/// contain a competitive document needs its header and its impacts, which are a
/// handful of vints; unpacking its 256 documents costs orders of magnitude more
/// and is pure waste if the block loses. Lucene keeps exactly this split.
#[derive(Debug, Clone, Copy)]
struct PendingBlock {
    /// The previous block's last doc ID -- the delta base the body decodes
    /// against. Held because `prev_doc_id` must not advance past this block
    /// until the body is actually decoded.
    base_doc_id: i32,
    last_doc_id: i32,
    body_start: usize,
    body_end: usize,
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

    /// `ImpactsEnum.getImpacts()`'s level-0 result, conceptually: the
    /// competitive `(freq, norm)` pairs for the level-0 block the cursor is
    /// currently positioned in (i.e. covering `doc_id`). Empty before the
    /// first `next_doc()`/`advance()` call, once exhausted, for the trailing
    /// tail block (no level-0 impacts exist on the wire for it), or for a
    /// field with no freqs.
    pub fn level0_impacts(&self) -> &[Impact] {
        &self.level0_impacts
    }

    /// `ImpactsEnum.getImpacts()`'s level-1 result, conceptually: the
    /// competitive `(freq, norm)` pairs merged across the *whole* 32-block
    /// level-1 span currently covering `doc_id`. Empty below
    /// `LEVEL1_NUM_DOCS` docs (no level-1 entries on the wire at that point)
    /// or for a field with no freqs.
    pub fn level1_impacts(&self) -> &[Impact] {
        &self.level1_impacts
    }

    /// The last (highest) doc ID covered by the level-0 block the cursor is
    /// currently positioned in (`self.prev_doc_id`, set from `header.last_doc_id`
    /// when a full block is decoded in [`Self::advance`] and left unchanged
    /// while `block_pos` walks within it). Only meaningful while
    /// [`Self::level0_impacts`] is non-empty (i.e. a real full block, not the
    /// trailing tail block, which never updates `prev_doc_id` and always
    /// reports empty impacts) — a caller that has proven a block's
    /// [`Self::level0_impacts`] can't beat its threshold calls
    /// `advance(this value + 1)` to skip straight past the rest of the block
    /// without decoding any more of it (see `search_term_query_scored_maxscore`
    /// in `lucene-search` for the real caller).
    /// The highest doc ID covered by the current level-1 span (32 level-0
    /// blocks), or [`NO_MORE_DOCS`] when this term has no level-1 entries on
    /// the wire (`docFreq < LEVEL1_NUM_DOCS`) or the cursor is past the last
    /// one.
    ///
    /// Pairs with [`Self::level1_impacts`] to let a caller skip a whole span at
    /// once instead of a block at a time -- real Lucene's
    /// `MaxScoreCache.getSkipUpTo`/`getSkipLevel` walk levels and skip at the
    /// highest level whose bound is still under the threshold, which is up to
    /// 32x fewer skip decisions than level-0 skipping alone.
    pub fn level1_last_doc_id(&self) -> i32 {
        self.level1_last_doc_id
    }

    /// Last doc ID of the most recently *decoded* block.
    ///
    /// Predates [`Self::advance_shallow`] and deliberately still means what it
    /// always meant, because callers use it as a re-check watermark and a
    /// shallow-aware value would poison that: in the tail block -- which has no
    /// header, so no known extent -- the shallow answer is [`NO_MORE_DOCS`],
    /// and a caller latching that as "already checked up to here" stops
    /// re-evaluating its bound for the rest of the query. That regression was
    /// caught by `boolean_query_scored_maxscore_matches_eager_ffi_path_and_actually_skips_blocks`,
    /// whose skip counter went to zero while its results stayed correct.
    ///
    /// New code walking blocks on impacts alone wants
    /// [`Self::level0_last_doc_id`] instead.
    pub fn current_block_last_doc_id(&self) -> i32 {
        self.prev_doc_id
    }

    /// `Impacts.getDocIdUpTo(0)`: the highest doc ID of the block the cursor is
    /// positioned on, valid after a shallow move as well as a full one, and the
    /// doc ID up to which [`Self::level0_impacts`] describes.
    ///
    /// [`NO_MORE_DOCS`] in the tail block and once exhausted; `-1` before the
    /// first move. A caller must therefore treat `NO_MORE_DOCS` as "extent
    /// unknown, cannot skip", not as "finished" -- which is also why
    /// [`Self::level0_impacts`] is empty in exactly those states.
    pub fn level0_last_doc_id(&self) -> i32 {
        self.level0_last_doc_id
    }

    /// Where the current document's occurrences begin in `.pos`/`.pay`, from
    /// `.doc`'s own skip data -- `Lucene104PostingsReader`'s `seekPosData`
    /// arguments plus `accumulatePendingPositions`' in-block frequency sum,
    /// bundled into one value.
    ///
    /// This is what makes a positional `advance(doc)` cost a skip rather than
    /// a walk: `.doc`'s level-0 header and level-1 entry each carry the
    /// `.pos`/`.pay` pointer their documents' occurrences start at, so the
    /// only frequencies that have to be summed are the ones in the current
    /// 256-document block. Nothing before it is read.
    ///
    /// `None` before the first `next_doc()`/`advance()` and once exhausted.
    /// `Err` when the cursor was opened with [`PostingsFlags::DocsOnly`]
    /// (the frequencies the sum needs were skipped, not decoded) or when a
    /// decoded frequency is negative.
    pub fn position_origin(&self) -> Result<Option<PositionOrigin>> {
        if self.doc_id == -1 || self.doc_id == NO_MORE_DOCS {
            return Ok(None);
        }
        if !self.needs_freq {
            return Err(Error::Unsupported(
                "position_origin needs a cursor opened with PostingsFlags::Freqs: the \
                 in-block frequency sum it adds to the skip data was never decoded",
            ));
        }
        // `posBufferUpto` is one wire byte, so at most 255; each frequency is
        // bounded below by the check just under this, and there are at most
        // `BLOCK_SIZE` of them, so the sum cannot overflow a `u64`.
        let mut skip = self.block_pos_origin.pos_buffer_upto as u64;
        for &freq in &self.block_freqs[..self.block_pos] {
            let freq = u64::try_from(freq).map_err(|_| {
                corrupted(format!(
                    "negative per-doc frequency {freq} in the current .doc block"
                ))
            })?;
            skip = skip.wrapping_add(freq);
        }
        Ok(Some(PositionOrigin {
            pos_fp: self.block_pos_origin.pos_fp,
            pay_fp: self.block_pos_origin.pay_fp,
            skip,
        }))
    }

    /// `PostingsEnum.nextDoc()`: moves to the next doc, returning its ID (or
    /// [`NO_MORE_DOCS`] if there isn't one).
    ///
    /// The common case is one slot along in the block already decoded, and it
    /// is written that way. Lucene's is the same three lines:
    ///
    /// ```java
    /// if (doc == level0LastDocID) { moveToNextLevel0Block(); }
    /// return this.doc = docBuffer[docBufferUpto++];
    /// ```
    ///
    /// This used to delegate to `advance(doc_id + 1)`, which is correct but
    /// pays `advance`'s `partition_point` -- a binary search over up to 256
    /// entries, eight unpredictable branches, to find an offset that is always
    /// exactly 1. It measured 14.5 ns per document against Lucene's 2.9 ns,
    /// a 5x gap that did not move when the `ForUtil` kernel underneath it got
    /// 3x faster, because block decode was never where the time went.
    ///
    /// Falls back to `advance` at a block boundary and before the first call,
    /// which is where the state machine actually lives.
    ///
    /// `pending.is_some()` is the `needsRefilling` case, and it takes the same
    /// fallback: a shallow move ([`Self::advance_shallow`]) leaves the cursor
    /// positioned on a block whose body has not been unpacked, so
    /// `block_docs`/`block_len` still describe the *previous* block and every
    /// document left in them is behind the shallow position. Answering from
    /// them would hand back a document the cursor has already moved past --
    /// a backwards `nextDoc()`, which `DocIdSetIterator` forbids. Lucene's
    /// `nextDoc` gates on exactly this (`if (doc == level0LastDocID ||
    /// needsRefilling)`), and `advance` here already carried the same guard.
    #[inline]
    pub fn next_doc(&mut self) -> Result<i32> {
        if self.doc_id == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        // Invariant, established by every path in `advance` that sets
        // `doc_id` to a real doc: `block_docs[block_pos] == doc_id` whenever
        // `block_pos < block_len`. So the next document, if this block still
        // has one, is the next slot -- no search needed.
        // ARITH: `block_pos` is only ever assigned `0`, an offset strictly
        // inside the current block, or `block_len` itself, and `block_len` is
        // `BLOCK_SIZE` or the tail's `count < BLOCK_SIZE` -- so
        // `block_pos <= 256` and the `+ 1` cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        let next = self.block_pos + 1;
        debug_assert!(self.block_pos <= self.block_len && self.block_len <= BLOCK_SIZE as usize);
        if self.pending.is_none() && next < self.block_len {
            self.block_pos = next;
            self.doc_id = self.block_docs[next];
            return Ok(self.doc_id);
        }
        self.advance(self.doc_id.saturating_add(1))
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
        // `pending.is_none()` matters: a shallow move positions past the
        // decoded block without touching `block_docs`, so those documents are
        // stale and must not answer an advance.
        if self.pending.is_none() && self.block_pos < self.block_len {
            let offset = find_next_geq(&self.block_docs[self.block_pos..self.block_len], target);
            // ARITH: `find_next_geq` returns at most the length of the slice
            // it was given, so `block_pos + offset <= block_len <=
            // BLOCK_SIZE`.
            #[allow(clippy::arithmetic_side_effects)]
            let landing = self.block_pos + offset;
            if landing < self.block_len {
                self.block_pos = landing;
                self.doc_id = self.block_docs[self.block_pos];
                return Ok(self.doc_id);
            }
            // Target is beyond every doc left in this block: fall through
            // to load the next one.
            self.block_pos = self.block_len;
        }

        // Position on the block that can contain `target`, reading headers and
        // impacts only. This is where the block-at-a-time walking happens, and
        // it no longer unpacks anything on the way.
        self.advance_shallow(target)?;

        if self.pending.is_some() {
            // A full block, positioned but not decoded. Now it is genuinely
            // needed, so pay for it.
            self.refill()?;
            let offset = find_next_geq(&self.block_docs, target);
            // `advance_shallow` only stops on a block whose header claims
            // `last_doc_id >= target`, so a well-formed block always has a
            // match. A corrupt `.doc` can claim one and then decode a body
            // whose last doc is smaller -- nothing on the wire ties the
            // level-0 header's `docDelta` to the body's own deltas -- and
            // indexing at `BLOCK_SIZE` would panic instead of surfacing that.
            if offset >= self.block_len {
                return Err(corrupted(
                    "full block's decoded doc IDs do not reach the last doc ID its level-0 \
                     header claims",
                ));
            }
            self.block_pos = offset;
            self.doc_id = self.block_docs[offset];
            return Ok(self.doc_id);
        }

        if self.doc_count_left == 0 {
            self.block_len = 0;
            self.block_pos = 0;
            self.doc_id = NO_MORE_DOCS;
            return Ok(NO_MORE_DOCS);
        }

        // The tail block: no skip data on the wire at all, so there is nothing
        // to decide from and it must be decoded.
        //
        // `Lucene104PostingsReader.refillRemainder` asserts
        // `docCountLeft >= 0 && docCountLeft < BLOCK_SIZE`. Here that is not an
        // invariant of our own arithmetic: `doc_count_left` starts from the
        // term's `docFreq`, which is read off disk, so a corrupt `.tim` can
        // leave a remainder at or past `BLOCK_SIZE` and slicing the fixed-size
        // block array by it panics instead of reporting the corruption.
        if self.doc_count_left < 0 || self.doc_count_left >= BLOCK_SIZE {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "tail block claims {} remaining documents, which is not in 0..{BLOCK_SIZE}",
                self.doc_count_left
            ))));
        }
        let count = self.doc_count_left as usize;
        read_tail_block(
            &mut self.r,
            self.prev_doc_id,
            self.index_has_freq,
            self.needs_freq,
            &mut self.block_docs[..count],
            &mut self.block_freqs[..count],
        )?;
        self.block_len = count;
        self.doc_count_left = 0;
        // The tail block has no level-0 skip header (and hence no impacts) on
        // the wire at all (`Lucene104PostingsReader.refillRemainder`'s
        // non-singleton branch never touches `level0SerializedImpacts`).
        self.level0_impacts.clear();

        let offset = find_next_geq(&self.block_docs[..count], target);
        self.block_pos = offset;
        self.doc_id = if offset < count {
            self.block_docs[offset]
        } else {
            NO_MORE_DOCS
        };
        Ok(self.doc_id)
    }

    /// `ImpactsEnum.advanceShallow(target)`: move to the first block that can
    /// contain `target`, reading each candidate block's level-0 header and
    /// impacts and **decoding no document bodies at all**. Returns that block's
    /// `level0LastDocID` -- the doc ID up to which
    /// [`Self::level0_impacts`] is valid -- or [`NO_MORE_DOCS`] when the cursor
    /// has run into the tail block or past the end.
    ///
    /// ## Why this exists
    ///
    /// This is the single largest structural divergence the M1.6 sweep found.
    /// Without it, deciding "can this block hold a competitive document?"
    /// required [`Self::advance`], which decodes the block first -- so a
    /// scoring loop paid the `ForUtil`/`PForUtil` unpack of 256 documents for
    /// every block it then discarded. Counted on the M1 corpus, `and t0 tz`
    /// unpacked 6,067,712 documents to score 80,226 of them: **1.3% of the
    /// decode work was used.** Lucene never does that; `advanceShallow` sets
    /// `needsRefilling` and `refillDocs()` runs only if the block survives.
    ///
    /// After this returns, [`Self::doc_id`] and [`Self::freq`] still describe
    /// the *previous* position -- nothing about the new block is readable until
    /// [`Self::advance`] or [`Self::next_doc`] materializes it. That is the same
    /// contract Lucene's `ImpactsEnum` has, and it is what lets a caller walk
    /// blocks on impacts alone.
    pub fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        // Already positioned on a block that covers `target`, decoded or not.
        if target <= self.level0_last_doc_id {
            return Ok(self.level0_last_doc_id);
        }

        // A shallow block that `target` has moved past: skip it without ever
        // decoding it. This is the case that saves the work.
        if let Some(p) = self.pending.take() {
            self.r.seek(p.body_end)?;
            self.prev_doc_id = p.last_doc_id;
            // ARITH: a `PendingBlock` is only ever created on the
            // `doc_count_left >= BLOCK_SIZE` branch of the loop below, and
            // nothing reassigns or decrements `doc_count_left` while one is
            // pending (`refill` is the only other consumer and it takes the
            // same `pending`), so the counter is still `>= BLOCK_SIZE` here.
            #[allow(clippy::arithmetic_side_effects)]
            {
                debug_assert!(self.doc_count_left >= BLOCK_SIZE);
                self.doc_count_left -= BLOCK_SIZE;
            }
        }

        loop {
            // Level-1 skip: jump past whole 32-block spans that are entirely
            // behind `target` before looking at any level-0 header, exactly as
            // `doAdvanceShallow` does.
            if target > self.level1_last_doc_id {
                self.skip_level1_to(target)?;
            }

            if self.doc_count_left == 0 {
                self.level0_impacts.clear();
                self.level0_last_doc_id = NO_MORE_DOCS;
                return Ok(NO_MORE_DOCS);
            }

            // Sampled here, at the top of the iteration, exactly as
            // `skipLevel0To` samples its `posFP`/`posUpto`/`payFP` locals:
            // this is the `.pos`/`.pay` state at the start of whichever block
            // the loop is about to look at.
            let origin = self.level0_pos;

            if self.doc_count_left >= BLOCK_SIZE {
                let header = read_full_block_header(
                    &mut self.r,
                    self.prev_doc_id,
                    self.index_has_freq,
                    self.index_has_pos,
                    self.index_has_offsets_or_payloads,
                )?;

                if let Some(skip) = header.pos_skip {
                    self.level0_pos.advance(skip);
                }

                if header.last_doc_id < target {
                    self.r.seek(header.body_end)?;
                    self.prev_doc_id = header.last_doc_id;
                    // ARITH: guarded by the enclosing
                    // `if self.doc_count_left >= BLOCK_SIZE`.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        self.doc_count_left -= BLOCK_SIZE;
                    }
                    continue;
                }

                // `skipLevel0To`'s `posFP`/`posUpto`/`payFP` locals, handed
                // to `seekPosData` once the loop settles: the state sampled
                // *before* this block's own header advanced it, which is
                // where this block's documents' occurrences begin.
                self.block_pos_origin = origin;

                // Impacts, yes -- a handful of vints, and the whole reason a
                // caller is here. Body, no.
                decode_impacts_into(header.impact_bytes, &mut self.level0_impacts)?;
                self.level0_last_doc_id = header.last_doc_id;
                self.pending = Some(PendingBlock {
                    base_doc_id: self.prev_doc_id,
                    last_doc_id: header.last_doc_id,
                    body_start: header.body_start,
                    body_end: header.body_end,
                });
                return Ok(header.last_doc_id);
            }

            // The tail carries no header, so its extent and impacts are unknown
            // without decoding it. Empty impacts mean "no bound available",
            // which every caller in this port treats as "cannot skip" -- the
            // same conservative answer Lucene reaches with its dummy
            // `freq = Integer.MAX_VALUE` impact. Its `.pos`/`.pay` origin is
            // known, though: `skipLevel0To`'s `else` branch breaks with the
            // same snapshot every other exit uses.
            self.block_pos_origin = origin;
            self.level0_impacts.clear();
            self.level0_last_doc_id = NO_MORE_DOCS;
            return Ok(NO_MORE_DOCS);
        }
    }

    /// `Lucene104PostingsReader.refillDocs`: unpack the block
    /// [`Self::advance_shallow`] positioned on. A no-op when there is nothing
    /// pending, so it is safe to call unconditionally before reading documents.
    fn refill(&mut self) -> Result<()> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        self.r.seek(p.body_start)?;
        decode_full_block_body(
            &mut self.r,
            p.base_doc_id,
            self.index_has_freq,
            self.needs_freq,
            &mut self.scratch,
            &mut self.block_docs,
            &mut self.block_freqs,
        )?;
        check_wire_position(self.r.position(), p.body_end, "full block body")?;
        self.block_len = BLOCK_SIZE as usize;
        self.block_pos = 0;
        self.prev_doc_id = p.last_doc_id;
        // ARITH: same invariant as `advance_shallow`'s own `pending.take()`
        // -- a `PendingBlock` exists only for a block `advance_shallow`
        // reached with `doc_count_left >= BLOCK_SIZE`, and this is the only
        // other place that consumes one.
        #[allow(clippy::arithmetic_side_effects)]
        {
            debug_assert!(self.doc_count_left >= BLOCK_SIZE);
            self.doc_count_left -= BLOCK_SIZE;
        }
        Ok(())
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
            // `skipLevel1To`'s `level0PosEndFP = level1PosEndFP` and friends:
            // jumping to a span boundary resets the level-0 `.pos`/`.pay`
            // state to the span's own, which is the whole reason the level-1
            // entry carries it. Without this a positional walk after a
            // level-1 jump would address `.pos` from wherever the last
            // level-0 header left it, i.e. from the wrong span.
            self.level0_pos = self.level1_pos;
            // Both operands descend from `docFreq`, a term-dictionary value:
            // `wrapping_*`, so a corrupt one produces a wrong (and then
            // rejected) span rather than a debug-build panic.
            self.doc_count_left = self.doc_freq.wrapping_sub(self.level1_doc_count_upto);
            self.level1_doc_count_upto = self.level1_doc_count_upto.wrapping_add(LEVEL1_NUM_DOCS);

            if self.doc_count_left < LEVEL1_NUM_DOCS {
                // Fewer than a full span remains: no level-1 entry precedes it.
                // `r` is now at the first of the trailing level-0 blocks.
                self.level1_last_doc_id = NO_MORE_DOCS;
                self.level1_impacts = Impacts::new();
                break;
            }

            let entry = read_level1_entry(
                &mut self.r,
                self.index_has_freq,
                self.index_has_pos,
                self.index_has_offsets_or_payloads,
            )?;
            // Off disk; see `read_full_block_header`'s `doc_delta`.
            self.level1_last_doc_id = self.level1_last_doc_id.wrapping_add(entry.doc_delta);
            self.level1_doc_end_fp = entry.doc_end_fp;
            if let Some(skip) = entry.pos_skip {
                self.level1_pos.advance(skip);
            }

            if self.level1_last_doc_id >= target {
                // `target` is within this span: `r` is positioned at the
                // span's first level-0 block header. This is also
                // `skipLevel1To`'s `needsImpacts && level1LastDocID >= target`
                // gate -- the impacts of a span being jumped over are never
                // decoded, and on a long postings list that is most of them.
                decode_impacts_into(entry.impact_bytes, &mut self.level1_impacts)?;
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
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one (`docs/arithmetic-gate.md`, "Test code").
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_store::data_output::DataOutput;

    /// Test-only encoder mirroring `Lucene104PostingsWriter.writeImpacts`
    /// (`Lucene104PostingsWriter.java:540-556`) exactly, so
    /// `decode_impacts`'s own tests can round-trip through the real writer
    /// logic rather than hand-picked bytes only.
    fn write_impacts(out: &mut Vec<u8>, impacts: &[Impact]) {
        let mut prev_freq = 0i32;
        let mut prev_norm = 0i64;
        for impact in impacts {
            let freq_delta = impact.freq - prev_freq - 1;
            let norm_delta = impact.norm - prev_norm - 1;
            if norm_delta == 0 {
                out.write_vint(freq_delta << 1);
            } else {
                out.write_vint((freq_delta << 1) | 1);
                out.write_zlong(norm_delta);
            }
            prev_freq = impact.freq;
            prev_norm = impact.norm;
        }
    }

    #[test]
    fn decode_impacts_empty_bytes_is_empty_list() {
        assert_eq!(decode_impacts(&[]).unwrap(), Vec::new());
    }

    #[test]
    fn decode_impacts_single_entry_implicit_norm_delta() {
        // freq=5, norm=1 (the common "norm only ever increases by 1" case,
        // encoded with the low bit clear and no zlong at all).
        let impacts = vec![Impact { freq: 5, norm: 1 }];
        let mut bytes = Vec::new();
        write_impacts(&mut bytes, &impacts);
        assert_eq!(bytes.len(), 1); // single vint, no zlong byte
        assert_eq!(decode_impacts(&bytes).unwrap(), impacts);
    }

    #[test]
    fn decode_impacts_multiple_entries_with_explicit_norm_deltas() {
        // Strictly increasing freq and (unsigned) norm, some entries with a
        // norm jump bigger than 1 (forces the zlong branch), including a
        // negative-looking (but unsigned-larger) norm value to exercise
        // zigzag encoding of a negative delta relative to the accumulated
        // norm (Lucene norms are `long`, not necessarily positive bytes).
        let impacts = vec![
            Impact { freq: 1, norm: 1 },
            Impact { freq: 3, norm: 2 },   // normDelta 0 -> implicit
            Impact { freq: 10, norm: 50 }, // normDelta 47 -> explicit zlong
            Impact {
                freq: 20,
                norm: i64::MAX,
            }, // huge normDelta -> explicit zlong, multi-byte
        ];
        let mut bytes = Vec::new();
        write_impacts(&mut bytes, &impacts);
        assert_eq!(decode_impacts(&bytes).unwrap(), impacts);
    }

    #[test]
    fn decode_impacts_rejects_truncated_final_vint() {
        // A freqDelta vint whose continuation bit is set but with no
        // following byte in the slice -- decode_impacts must surface this as
        // an EOF error, not silently stop or panic.
        let bytes = [0x80u8]; // continuation bit set, no next byte
        assert!(decode_impacts(&bytes).is_err());
    }

    #[test]
    fn decode_impacts_rejects_truncated_zlong_after_explicit_flag() {
        // freqDelta with the low bit set (an explicit normDelta follows) but
        // the zlong bytes are missing entirely.
        let mut bytes = Vec::new();
        bytes.write_vint(1); // freqDelta << 1 | 1, i.e. freqDelta=0, explicit flag
                             // no zlong bytes follow
        assert!(decode_impacts(&bytes).is_err());
    }

    #[test]
    fn an_impacts_freq_delta_at_the_top_of_i32_wraps_rather_than_panicking() {
        // `readImpacts` accumulates `freq += 1 + (freqDelta >>> 1)` in a Java
        // `int`, which wraps. `freqDelta` is a `readVInt`, and a five-byte
        // varint with the sign bit set decodes to a negative `i32`, so
        // `(freqDelta as u32) >> 1` reaches `i32::MAX` -- and the `1 +` in
        // front of it overflowed *before* the accumulator's `wrapping_add`
        // ever saw it. That is a debug-build panic, which through the FFI is
        // a dead JVM, for an impacts run a corrupt `.doc` can carry.
        //
        // A single-bit flip cannot produce this five-byte pattern, which is
        // why the byte-flip sweeps do not reach it.
        let mut bytes = Vec::new();
        bytes.write_vint(-2); // freqDelta == -2: (0xFFFF_FFFE >> 1) == i32::MAX,
                              // and the low bit is clear so no zlong follows
        let impacts = decode_impacts(&bytes).expect("a wrapped freq, not a panic");
        assert_eq!(impacts.len(), 1);
        // 0 + 1 + i32::MAX, wrapped: exactly what Java's `int` produces.
        assert_eq!(impacts[0].freq, i32::MIN);
        assert_eq!(impacts[0].norm, 1); // low bit clear -> implicit +1
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

    /// `IndexOptions::DocsAndCustomFreqs` is wire-identical to `DocsAndFreqs`
    /// (see the module doc): the exact same `.doc` bytes as
    /// `read_postings_all_freq_one_docs_only_bit_path` above decode
    /// identically when read under this option, for both the eager
    /// [`DocInput::read_postings`] and the lazy [`DocInput::lazy_cursor`]
    /// path.
    #[test]
    fn read_postings_docs_and_custom_freqs_matches_docs_and_freqs() {
        let id = [6u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_group_vints(&mut doc, &[(1 << 1) | 1, (3 << 1) | 1, (1 << 1) | 1]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, 3, IndexOptions::DocsAndCustomFreqs, false)
            .unwrap();
        assert_eq!(postings.docs, vec![0, 3, 4]);
        assert_eq!(postings.freqs, vec![1, 1, 1]);

        let mut cursor = input
            .lazy_cursor(meta, 3, IndexOptions::DocsAndCustomFreqs, false)
            .unwrap();
        let mut docs = Vec::new();
        loop {
            let d = cursor.next_doc().unwrap();
            if d == NO_MORE_DOCS {
                break;
            }
            docs.push(d);
        }
        assert_eq!(docs, vec![0, 3, 4]);
    }

    /// `IndexOptions::None` isn't a valid postings-carrying option (real
    /// Lucene never writes `.doc` bytes for an unindexed field) -- both
    /// decode entry points reject it, distinct from the now-accepted
    /// `DocsAndCustomFreqs` case above.
    #[test]
    fn read_postings_rejects_index_options_none() {
        let id = [6u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        doc.extend_from_slice(&footer);
        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        assert!(matches!(
            input.read_postings(meta, 3, IndexOptions::None, false),
            Err(Error::Unsupported(_))
        ));
        assert!(matches!(
            input.lazy_cursor(meta, 3, IndexOptions::None, false),
            Err(Error::Unsupported(_))
        ));
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

    /// Writes a level-1 skip entry for a field with freqs but no positions
    /// (`IndexOptions::DocsAndFreqs`), with a real (possibly non-empty)
    /// impacts byte run — the freq-gated counterpart to
    /// [`write_level1_entry_docs`], needed to exercise `PostingsCursor::
    /// level1_impacts`/`Postings::level1_impacts` with real level-1 impacts
    /// data. See [`read_level1_entry`] for the exact field layout this
    /// mirrors.
    fn write_level1_entry_with_impacts(
        doc: &mut Vec<u8>,
        doc_delta: i32,
        span: &[u8],
        impacts: &[Impact],
    ) {
        doc.write_vint(doc_delta);
        doc.write_vlong(span.len() as i64);
        let mut impact_bytes = Vec::new();
        write_impacts(&mut impact_bytes, impacts);
        // `skip1EndFP` is measured from right after the short that carries it
        // (i.e. from the start of `numImpactBytes`) through the end of this
        // entry's freq-gated metadata: here that's just `numImpactBytes` (2
        // bytes) plus the impact bytes themselves (no pos/pay sub-fields,
        // since this helper is scoped to DocsAndFreqs).
        doc.write_i16((2 + impact_bytes.len()) as i16);
        doc.write_i16(impact_bytes.len() as i16);
        doc.write_bytes(&impact_bytes);
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
    fn postings_cursor_level0_impacts_varies_within_block_and_across_blocks() {
        // docFreq == 2 * BLOCK_SIZE (512): two full blocks, each with its own
        // distinct (freq, norm) impacts list -- mirrors the "impacts varying
        // within a level-0 block" real-Lucene case
        // (`big_field_impacts_match_real_lucene_impacts_enum`), but hand-built
        // so `PostingsCursor::level0_impacts`/`Postings::level0_impacts` are
        // exercised directly against known-good expected values rather than
        // only cross-checked against the lazy path or real Lucene bytes.
        let id = [20u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        let block0_impacts = vec![Impact { freq: 1, norm: 1 }, Impact { freq: 5, norm: 10 }];
        let block1_impacts = vec![Impact { freq: 2, norm: 3 }];
        write_full_block_with_impacts(&mut doc, true, 3, &block0_impacts);
        write_full_block_with_impacts(&mut doc, true, 7, &block1_impacts);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, 2 * BLOCK_SIZE, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(
            postings.level0_impacts,
            vec![
                (BLOCK_SIZE - 1, block0_impacts.clone()),
                (2 * BLOCK_SIZE - 1, block1_impacts.clone()),
            ]
        );

        let mut cursor = PostingsCursor::new(&postings);
        assert_eq!(cursor.advance(0), 0);
        assert_eq!(cursor.level0_impacts(), block0_impacts.as_slice());
        assert!(cursor.level1_impacts().is_empty());

        // Exact last doc of block 0 -- the boundary value a `<` vs `<=` bug
        // in `find_impacts`'s binary search would only surface at.
        assert_eq!(cursor.advance(BLOCK_SIZE - 1), BLOCK_SIZE - 1);
        assert_eq!(cursor.level0_impacts(), block0_impacts.as_slice());

        assert_eq!(cursor.advance(BLOCK_SIZE), BLOCK_SIZE);
        assert_eq!(cursor.level0_impacts(), block1_impacts.as_slice());
        assert!(cursor.level1_impacts().is_empty());
    }

    #[test]
    fn postings_cursor_level1_impacts_reachable_across_span() {
        // docFreq == LEVEL1_NUM_DOCS + 8 (8200): one level-1 entry carrying a
        // real (non-empty) merged span impacts list, a span of 32 full blocks
        // (each with its own distinct level-0 impacts), then an 8-doc tail --
        // mirrors "l1_field_impacts_match_real_lucene_impacts_enum" but hand-
        // built so both `level0_impacts()` (varying block-to-block within the
        // span) and `level1_impacts()` (constant across the whole span, empty
        // once past it in the tail) are checked against known-good values.
        let id = [21u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        let first_block_impacts = vec![Impact { freq: 1, norm: 1 }];
        let last_block_impacts = vec![Impact { freq: 4, norm: 9 }];
        let span_impacts = vec![Impact { freq: 4, norm: 9 }, Impact { freq: 8, norm: 20 }];

        let mut span = Vec::new();
        write_full_block_with_impacts(&mut span, true, 1, &first_block_impacts);
        for _ in 1..LEVEL1_FACTOR - 1 {
            write_full_block_with_impacts(&mut span, true, 1, &[]);
        }
        write_full_block_with_impacts(&mut span, true, 1, &last_block_impacts);

        write_level1_entry_with_impacts(&mut doc, LEVEL1_NUM_DOCS, &span, &span_impacts);
        doc.extend_from_slice(&span);
        // Tail: 8 consecutive docs (deltas all 1) from prevDocID 8191, no
        // freq exceptions (freq==1 bit path).
        write_group_vints(&mut doc, &[(1 << 1) | 1; 8]);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let postings = input
            .read_postings(meta, LEVEL1_NUM_DOCS + 8, IndexOptions::DocsAndFreqs, false)
            .unwrap();
        assert_eq!(
            postings.level1_impacts,
            vec![(LEVEL1_NUM_DOCS - 1, span_impacts.clone())]
        );

        let mut cursor = PostingsCursor::new(&postings);

        // First doc of the span: level-0 impacts are the first block's own,
        // level-1 impacts are the whole span's merged list.
        assert_eq!(cursor.advance(0), 0);
        assert_eq!(cursor.level0_impacts(), first_block_impacts.as_slice());
        assert_eq!(cursor.level1_impacts(), span_impacts.as_slice());

        // Last full block of the span: level-0 impacts change, level-1
        // impacts stay the same (whole-span merge, not per-block).
        assert_eq!(
            cursor.advance(LEVEL1_NUM_DOCS - BLOCK_SIZE),
            LEVEL1_NUM_DOCS - BLOCK_SIZE
        );
        assert_eq!(cursor.level0_impacts(), last_block_impacts.as_slice());
        assert_eq!(cursor.level1_impacts(), span_impacts.as_slice());

        // Exact last doc of the entire span -- the boundary value a `<` vs
        // `<=` bug in `find_impacts` would only surface at, distinct from
        // the "first doc of the last block" check just above.
        assert_eq!(cursor.advance(LEVEL1_NUM_DOCS - 1), LEVEL1_NUM_DOCS - 1);
        assert_eq!(cursor.level0_impacts(), last_block_impacts.as_slice());
        assert_eq!(cursor.level1_impacts(), span_impacts.as_slice());

        // Into the trailing tail (past the level-1 span entirely): both are
        // empty, same as `LazyDocsCursor`'s contract for the tail block.
        assert_eq!(cursor.advance(LEVEL1_NUM_DOCS), LEVEL1_NUM_DOCS);
        assert!(cursor.level0_impacts().is_empty());
        assert!(cursor.level1_impacts().is_empty());
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
                            // 3 valid impact bytes: freqDelta=0, normDelta=0 (implicit +1 each)
                            // three times over, decoding to (freq=1,norm=1), (2,2), (3,3).
        bytes.write_bytes(&[0x00, 0x00, 0x00]);
        bytes.write_vlong(50); // posEndFP delta
        bytes.write_byte(7); // posBufferUpto
        bytes.write_vlong(60); // payEndFP delta
        bytes.write_vint(9); // payloadByteUpto (parsed, then recomputed -- see
                             // `read_pos_skip`)
        let metadata_end = bytes.len();

        let mut r = SliceInput::new(&bytes);
        let entry = read_level1_entry(&mut r, true, true, true).unwrap();
        assert_eq!(entry.doc_delta, 8191);
        assert_eq!(entry.doc_end_fp, 100 + pos_after_vlong);
        // r is left at the start of the span's first level-0 block header.
        assert_eq!(r.position(), metadata_end);
        assert_eq!(
            decode_impacts(entry.impact_bytes).unwrap(),
            vec![
                Impact { freq: 1, norm: 1 },
                Impact { freq: 2, norm: 2 },
                Impact { freq: 3, norm: 3 },
            ]
        );
        // The `.pos`/`.pay` skip pointers are retained now, not discarded.
        let skip = entry.pos_skip.expect("a positions-indexing field has them");
        assert_eq!(skip.pos_end_fp_delta, 50);
        assert_eq!(skip.pos_buffer_upto, 7);
        assert_eq!(skip.pay_end_fp_delta, 60);
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
    /// only needed `read_full_block_header`'s wire-order-only decode to work.
    /// `docDelta` is always `BLOCK_SIZE` (256), matching the "all 256 deltas
    /// are 1" body this helper always writes.
    fn write_full_block(out: &mut Vec<u8>, index_has_freq: bool, freq_value: i32) {
        write_full_block_with_impacts(out, index_has_freq, freq_value, &[]);
    }

    /// Same as [`write_full_block`], but with a real (possibly non-empty)
    /// level-0 impacts byte run instead of always writing an empty one —
    /// needed to exercise `PostingsCursor::level0_impacts`/`Postings::
    /// level0_impacts` with real per-block impacts data rather than a
    /// perpetually-empty section.
    fn write_full_block_with_impacts(
        out: &mut Vec<u8>,
        index_has_freq: bool,
        freq_value: i32,
        impacts: &[Impact],
    ) {
        // The metadata region (impacts, and the pos/pay skip fields this
        // helper's fields never have) is measured separately from the block
        // body, because `level0NumBytes` spans the former plus the two header
        // fields while `blockLength` spans the former plus the latter.
        let mut meta_region = Vec::new();
        if index_has_freq {
            let mut impact_bytes = Vec::new();
            write_impacts(&mut impact_bytes, impacts);
            // Impacts byte-length is a plain vint on the wire here
            // (`read_full_block_header`'s doc comment) — `write_vint` matches
            // that (not `write_vlong`, even though the two happen to agree
            // byte-for-byte for any value small enough to fit both).
            meta_region.write_vint(impact_bytes.len() as i32);
            meta_region.write_bytes(&impact_bytes);
        }
        let mut block_body = Vec::new();
        block_body.write_byte(0); // bitsPerValue == 0: all 256 deltas are 1.
        if index_has_freq {
            block_body.write_byte(0); // PForUtil token: bitsPerValue=0, numExceptions=0
            block_body.write_vint(freq_value);
        }

        // `blockLength` is measured from right after this field (i.e. from
        // right here) through the end of the whole block -- see
        // `read_full_block_header`'s doc comment.
        //
        // `level0NumBytes` (`numSkipBytes`) is measured from right after
        // *itself* through the start of the block body, i.e. the two
        // fixed-width header fields plus the metadata region --
        // `Lucene104PostingsWriter.flushDocBlock`'s `numSkipBytes =
        // level0Output.size()` sampled after the impacts/pos/pay region, then
        // `+= scratchOutput.size()`. This helper used to write `body.len()`
        // here, which is neither, and no reader noticed until
        // `read_full_block_header` started checking the two agree.
        let header_fields_len = 4; // vint15 docDelta + vlong15 blockLength
        out.write_vlong((meta_region.len() + header_fields_len) as i64);
        out.write_i16(BLOCK_SIZE as i16); // docDelta via writeVInt15
        out.write_i16((meta_region.len() + block_body.len()) as i16); // blockLength via writeVLong15
        out.write_bytes(&meta_region);
        out.write_bytes(&block_body);
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
        doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
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
        doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
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
        let mut source = deltas;
        let bits_per_value = 2u32;
        let mut packed = Vec::new();
        // `for_encode` collapses lanes in place (as Java's `ForUtil.encode`
        // does), so hand it a scratch copy -- `deltas` is the expectation below.
        for_util::for_encode(&mut source, bits_per_value, &mut packed);

        let mut body = Vec::new();
        body.write_byte(bits_per_value as u8);
        body.write_bytes(&packed);
        doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
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
            level0_impacts: Vec::new(),
            level1_impacts: Vec::new(),
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
        span.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
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
    fn lazy_cursor_next_doc_after_advance_shallow_moves_to_the_shallow_block() {
        // Two full blocks (docs 0..255 and 256..511). Walk into block 0, then
        // shallow-advance into block 1 and call `next_doc()`.
        //
        // `advance_shallow` reads block 1's header and impacts but leaves its
        // body unpacked, so `block_docs` still holds *block 0*. Answering
        // `next_doc()` out of that buffer hands back doc 1 -- behind the
        // position the shallow move just established, and described by block
        // 1's impacts rather than its own block's. Lucene gates its `nextDoc`
        // on exactly this (`if (doc == level0LastDocID || needsRefilling)`)
        // and lands on the shallow block's first doc, 256.
        let id = [41u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        write_full_block(&mut doc, false, 0);
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

        assert_eq!(cursor.next_doc().unwrap(), 0);
        assert_eq!(cursor.advance_shallow(300).unwrap(), 2 * BLOCK_SIZE - 1);
        assert_eq!(
            cursor.next_doc().unwrap(),
            BLOCK_SIZE,
            "next_doc must materialize the shallow-positioned block, not answer \
             from the stale previous one"
        );
        // And the walk continues from there, still in order.
        assert_eq!(cursor.next_doc().unwrap(), BLOCK_SIZE + 1);
    }

    #[test]
    fn a_level1_entry_that_disagrees_with_its_own_skip1_end_fp_is_rejected() {
        // The level-1 entry's `skip1EndFP` short is the byte length of that
        // entry's own metadata, read off the same file as the metadata it
        // measures. Java asserts it; this port used to `debug_assert_eq!` it,
        // i.e. panic in a debug build on a corrupt `.doc`. Build a valid
        // `docFreq >= LEVEL1_NUM_DOCS` term's first level-1 entry, then
        // corrupt only that short.
        let id = [42u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        // Level-1 entry: docDelta, span length, then (freqs) skip1EndFP,
        // numImpactBytes, impact bytes.
        doc.write_vint(1000); // docDelta
        doc.write_vlong(0); // span byte length (unused by this path)
        let skip1_pos = doc.len();
        doc.write_i16(0); // skip1EndFP placeholder -- patched below
        doc.write_i16(0); // numImpactBytes = 0
        let after = doc.len();
        doc.extend_from_slice(&footer);
        // Honest value first: from just after the skip1EndFP short to `after`.
        let honest = (after - (skip1_pos + 2)) as i16;
        doc[skip1_pos..skip1_pos + 2].copy_from_slice(&honest.to_le_bytes());

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        // Sanity: with the honest length the entry parses (it fails later, on
        // the blocks that are not there, but not on this check).
        let honest_err = input
            .read_postings(meta, LEVEL1_NUM_DOCS, IndexOptions::DocsAndFreqs, false)
            .unwrap_err();
        assert!(
            !format!("{honest_err}").contains("level-1 skip entry"),
            "the honest entry must not trip the skip1EndFP check: {honest_err}"
        );

        // Now lie about it by one byte.
        doc[skip1_pos..skip1_pos + 2].copy_from_slice(&(honest + 1).to_le_bytes());
        let input = DocInput::open(&doc, &id, "").unwrap();
        let err = input
            .read_postings(meta, LEVEL1_NUM_DOCS, IndexOptions::DocsAndFreqs, false)
            .unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
                && format!("{err}").contains("level-1 skip entry"),
            "expected a Corrupted decode error naming the level-1 entry, got {err:?}"
        );
    }

    /// A length read off disk must never become an unchecked `as usize`.
    ///
    /// `numImpactBytes` in a level-1 entry is a signed `readShort`; a corrupt
    /// `.doc` can put a negative there, and `negative as usize` sign-extends
    /// to ~2^64, so the `impact_start + len` that follows overflows -- which
    /// in a debug build is a panic, and a panic in a debug build of the FFI
    /// takes the JVM down. Java throws instead. Same class as the four
    /// `debug_assert` sites c8 converted and the lazy-cursor tail block.
    #[test]
    fn a_level1_entry_with_a_negative_impacts_length_is_rejected() {
        let id = [70u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        doc.write_vint(1000); // docDelta
        doc.write_vlong(0); // span byte length
        doc.write_i16(0); // skip1EndFP (never reached)
        doc.write_i16(-1); // numImpactBytes: impossible
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let err = input
            .read_postings(meta, LEVEL1_NUM_DOCS, IndexOptions::DocsAndFreqs, false)
            .unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
                && format!("{err}").contains("level-1 impacts"),
            "expected a Corrupted decode error naming the impacts length, got {err:?}"
        );
    }

    /// The level-0 header's impacts length is a `vint`, which decodes to a
    /// negative from a five-byte varint with the sign bit set -- the same
    /// overflow-into-panic shape as the level-1 short above.
    #[test]
    fn a_level0_header_with_a_negative_impacts_length_is_rejected() {
        let id = [71u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        doc.write_vlong(4); // level0NumBytes
        doc.write_i16(256); // docDelta
        doc.write_i16(4); // blockLength
        doc.write_vint(-1); // impacts byte length: impossible
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let err = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::DocsAndFreqs, false)
            .unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "expected a Corrupted decode error, got {err:?}"
        );
    }

    /// `blockLength` is a `vlong15`: a corrupt one can be negative, and
    /// `position() + (negative as usize)` overflows.
    #[test]
    fn a_level0_header_with_a_negative_block_length_is_rejected() {
        let id = [72u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        doc.write_vlong(4); // level0NumBytes
        doc.write_i16(256); // docDelta
                            // `read_vlong15`: a negative short means "the low 15 bits, plus a
                            // vlong of high bits shifted up by 15". `1 << 48` shifted by 15 sets
                            // the sign bit of the i64.
        doc.write_i16(-32768i16); // 0x8000: low 15 bits are 0, more follows
        doc.write_vlong(1i64 << 48);
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
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
                && format!("{err}").contains("level-0 block"),
            "expected a Corrupted decode error naming the block length, got {err:?}"
        );
    }

    /// The lazy cursor can reach its tail block with a `doc_count_left` that
    /// is not a legal tail size, because that number descends from `docFreq`
    /// -- a term-dictionary value, not one of our own.
    ///
    /// The route: a level-0 header whose `docDelta` overshoots what its body
    /// decodes (nothing on the wire ties the two together). `advance` walks
    /// off the end of the decoded block, `advance_shallow` returns early
    /// because the target is still within the *claimed* extent, and the tail
    /// path is entered with a whole block's worth of documents still
    /// outstanding. Slicing the fixed 256-entry block array by that used to
    /// panic; `Lucene104PostingsReader.refillRemainder` asserts the same
    /// bound.
    #[test]
    fn a_tail_block_larger_than_block_size_is_rejected_rather_than_panicking() {
        let id = [73u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        let mut body = Vec::new();
        body.write_byte(0); // bitsPerValue == 0: docs 0..255
        doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
        doc.write_i16(1000); // docDelta: claims the block reaches doc 999
        doc.write_i16(body.len() as i16);
        doc.write_bytes(&body);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        // 600 documents: one full block, leaving 344 -- more than a tail can
        // hold -- once the block is consumed.
        let mut cursor = input
            .lazy_cursor(meta, 600, IndexOptions::Docs, false)
            .unwrap();
        assert_eq!(cursor.next_doc().unwrap(), 0);
        assert_eq!(cursor.advance(255).unwrap(), 255);
        let err = cursor.advance(500).unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
                && format!("{err}").contains("tail block"),
            "expected a Corrupted decode error naming the tail block, got {err:?}"
        );
    }

    /// A `total_term_freq` that no file could hold is rejected before it sizes
    /// or divides anything -- and the two rejections are different in kind:
    /// negative is a corrupt term dictionary, while above `u32::MAX` is a
    /// real value this port's `u32`-indexed flat streams cannot address.
    #[test]
    fn an_impossible_total_term_freq_is_rejected() {
        let id = [74u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        pos.write_vint(1);
        pos.extend_from_slice(&pos_footer);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            ..TermMetadata::EMPTY
        };
        for total_term_freq in [-1i64, u32::MAX as i64 + 1, i64::MAX] {
            let err = read_positions_for_docs(
                &pos_in,
                None,
                meta,
                &[1],
                total_term_freq,
                IndexOptions::DocsAndFreqsAndPositions,
                false,
                &[0],
            )
            .unwrap_err();
            if total_term_freq < 0 {
                assert!(
                    matches!(err, Error::Store(lucene_store::Error::Corrupted(_)))
                        && format!("{err}").contains("negative count"),
                    "a negative total_term_freq is corruption, got {err:?}"
                );
            } else {
                assert!(
                    matches!(err, Error::Unsupported(_)) && format!("{err}").contains("u32::MAX"),
                    "a total_term_freq past this port's ceiling is unsupported, not \
                     corrupt, got {err:?}"
                );
            }
            // The whole-term reader guards the same value the same way.
            assert!(read_positions(
                &pos_in,
                None,
                meta,
                &[1],
                total_term_freq,
                IndexOptions::DocsAndFreqsAndPositions,
                false,
            )
            .is_err());
        }
    }

    /// A tail payload whose length runs past the end of `.pos` must be a
    /// decode error, not a multi-gigabyte `resize` driven straight off disk.
    #[test]
    fn a_tail_payload_longer_than_the_file_is_rejected() {
        let id = [75u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        pos.write_vint((3 << 1) | 1); // posDelta=3, payload length follows
        pos.write_vint(1_000_000); // ... and it is longer than the whole file
        pos.extend_from_slice(&pos_footer);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            ..TermMetadata::EMPTY
        };
        // Both the whole-term reader and the wanted-documents walk.
        assert!(read_positions(
            &pos_in,
            None,
            meta,
            &[1],
            1,
            IndexOptions::DocsAndFreqsAndPositions,
            true,
        )
        .is_err());
        assert!(read_occurrences_for_docs(
            &pos_in,
            None,
            meta,
            &[1],
            1,
            IndexOptions::DocsAndFreqsAndPositions,
            true,
            &[0],
        )
        .is_err());
    }

    /// A field that does not index positions has no `.pos` data to walk, in
    /// either shape.
    #[test]
    fn wanted_documents_need_a_positions_indexing_field() {
        let id = [76u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        pos.write_vint(1);
        pos.extend_from_slice(&pos_footer);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            ..TermMetadata::EMPTY
        };
        for options in [IndexOptions::Docs, IndexOptions::DocsAndFreqs] {
            assert!(matches!(
                read_positions_for_docs(&pos_in, None, meta, &[1], 1, options, false, &[0]),
                Err(Error::Unsupported(_))
            ));
            assert!(matches!(
                read_occurrences_for_docs(&pos_in, None, meta, &[1], 1, options, false, &[0]),
                Err(Error::Unsupported(_))
            ));
        }
    }

    #[test]
    fn a_block_body_that_disagrees_with_its_headers_byte_length_is_rejected() {
        // A level-0 header records the body's byte length (`level0NumBytes`);
        // the body's own encoding records nothing that ties it to that. Here
        // the header claims one byte more than the body actually occupies, so
        // after decoding it the reader sits one byte short of where the header
        // says the block ends -- and every following block would decode as
        // garbage. Java only `assert`s this; this port used to
        // `debug_assert_eq!` it, i.e. panic in debug and silently continue in
        // release. Both paths that decode a full block must now return a
        // decode error instead.
        let id = [42u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        let mut body = Vec::new();
        body.write_byte(0); // bitsPerValue == 0: 256 consecutive docs, 0..255.
        doc.write_vlong(4); // level0NumBytes (honest -- the lie under test is blockLength)
        doc.write_i16(256); // docDelta (honest)
        doc.write_i16(body.len() as i16 + 1); // blockLength, matching the lie
        doc.write_bytes(&body);
        doc.write_byte(0); // the extra byte the header accounts for
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        // Eager path.
        let err = input
            .read_postings(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "expected a Corrupted decode error from read_postings, got {err:?}"
        );
        // Lazy path: `advance` refills the block through the same check.
        let mut cursor = input
            .lazy_cursor(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap();
        let err = cursor.advance(10).unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "expected a Corrupted decode error from the cursor, got {err:?}"
        );
    }

    #[test]
    fn lazy_cursor_advance_rejects_a_block_body_that_undershoots_its_header() {
        // A level-0 header's `docDelta` and the block body's own deltas are
        // independent on the wire: nothing forces them to agree. Here the
        // header claims the block's last doc is 999 while the body decodes to
        // docs 0..255, so `advance_shallow` stops on the block (999 >= 500)
        // and the post-refill search for the first doc >= 500 finds nothing.
        // That used to index one past the block buffer and panic.
        let id = [42u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;

        let mut body = Vec::new();
        body.write_byte(0); // bitsPerValue == 0: 256 consecutive docs, 0..255.
        doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
        doc.write_i16(1000); // docDelta -- a lie: the body only reaches 255.
        doc.write_i16(body.len() as i16); // blockLength (honest, so the seek lands right)
        doc.write_bytes(&body);
        doc.extend_from_slice(&footer);

        let input = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let mut cursor = input
            .lazy_cursor(meta, BLOCK_SIZE, IndexOptions::Docs, false)
            .unwrap();
        let err = cursor.advance(500).unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
            "expected a Corrupted decode error, got {err:?}"
        );
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
        doc.write_vlong(4); // level0NumBytes: the two header fields, no metadata region
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

    /// `level0NumBytes` is the level-0 header's own self-description: the
    /// byte length from just after it to the start of the block body. This
    /// reader reaches the same position by parsing the fields in between, so
    /// a disagreement means the two do not describe the same block -- a
    /// corrupt vlong, or a `FieldInfos` whose `has_payloads` does not match
    /// the bytes, which decides whether two of those fields are there at all.
    ///
    /// Without the check the block body decodes from the wrong offset and
    /// produces plausible garbage silently. `read_level1_entry` has always
    /// had the equivalent check (`skip1EndFP`); this is its level-0 twin.
    #[test]
    fn a_level0_header_whose_num_skip_bytes_disagrees_with_its_fields_is_rejected() {
        let id = [76u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        let mut body = Vec::new();
        body.write_byte(0); // bitsPerValue == 0: 256 consecutive docs.
                            // The two header fields are 4 bytes and there is no metadata region
                            // (IndexOptions::Docs), so the honest value is 4.
        doc.write_vlong(5);
        doc.write_i16(BLOCK_SIZE as i16);
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
            .expect_err("the header does not describe its own fields");
        assert!(
            format!("{err}").contains("level-0 skip header"),
            "unexpected error: {err}"
        );
    }

    /// A frequency big enough to outrun `.pos` many times over must be
    /// rejected up front, not walked until the file EOFs.
    ///
    /// It is not a panic that is at stake -- every index is bounded -- but an
    /// allocation blow-up: each refill yields up to 256 more `Position`
    /// records, each with its own payload `Vec`, and a `PForUtil` block can be
    /// as little as its own token byte. An allocation failure *aborts*, which
    /// no `catch_unwind` at the FFI boundary can intercept, so the ceiling has
    /// to be checked before the loop rather than discovered inside it.
    #[test]
    fn a_frequency_that_dwarfs_the_pos_file_is_rejected_before_the_walk() {
        let id = [77u8; ID_LENGTH];
        let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2, 3]);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let err = walk_one(
            &pos_in,
            PositionOrigin {
                pos_fp: pos_start_fp,
                pay_fp: 0,
                skip: 0,
            },
            10_000_000,
            Some(pos_start_fp),
            3,
        )
        .expect_err("ten million occurrences do not fit in a handful of bytes");
        assert!(
            format!("{err}").contains("cannot fit in the"),
            "unexpected error: {err}"
        );
    }

    /// `Lucene104PostingsReader.reset`'s three-way `lastPosBlockFP` rule
    /// (`Lucene104PostingsReader.java:526-532`), which is the only thing that
    /// separates a full `PForUtil` block from the vint tail once a walk has
    /// jumped into the middle of `.pos`.
    #[test]
    fn last_pos_block_fp_matches_lucene_reset() {
        let meta = TermMetadata {
            pos_start_fp: 1000,
            last_pos_block_offset: 77,
            ..TermMetadata::EMPTY
        };
        // Below one full block: everything is the tail, which therefore
        // starts at the term's own posStartFP.
        assert_eq!(last_pos_block_fp(meta, 5), Some(1000));
        assert_eq!(last_pos_block_fp(meta, BLOCK_SIZE as i64 - 1), Some(1000));
        // Exactly one full block: no tail at all, and Java's `-1` sentinel
        // (a pointer nothing can equal) becomes `None` here.
        assert_eq!(last_pos_block_fp(meta, BLOCK_SIZE as i64), None);
        // Past a full block: posStartFP + lastPosBlockOffset.
        assert_eq!(last_pos_block_fp(meta, BLOCK_SIZE as i64 + 1), Some(1077));
    }

    /// A `.pos` stream holding `count` single-vint occurrences with no
    /// offsets or payloads, plus the term's `posStartFP`.
    fn tail_only_pos(id: &[u8; ID_LENGTH], deltas: &[i32]) -> (Vec<u8>, u64) {
        let (mut pos, footer) = pos_header_and_footer(id);
        let pos_start_fp = pos.len() as u64;
        for &d in deltas {
            pos.write_vint(d);
        }
        pos.extend_from_slice(&footer);
        (pos, pos_start_fp)
    }

    fn positions_only_wants() -> PositionWants {
        PositionWants {
            has_offsets: false,
            has_payloads: false,
            want_offsets: false,
            want_payloads: false,
        }
    }

    fn walk_one(
        pos_in: &PosInput<'_>,
        origin: PositionOrigin,
        freq: usize,
        last_pos_block: Option<u64>,
        tail_count: usize,
    ) -> Result<Vec<Position>> {
        let mut sink = FullOccurrences {
            occurrences: Vec::new(),
            doc_starts: Vec::new(),
        };
        walk_document_occurrences(
            pos_in,
            None,
            origin,
            freq,
            last_pos_block,
            tail_count,
            positions_only_wants(),
            &mut sink,
        )?;
        Ok(sink.occurrences)
    }

    /// The honest case, so the three rejections below cannot pass for the
    /// wrong reason: a skip of two occurrences into a five-occurrence tail
    /// yields the third and fourth, with the position accumulator restarting
    /// at the document's first occurrence.
    #[test]
    fn a_skip_driven_walk_starts_at_the_occurrence_the_origin_names() {
        let id = [70u8; ID_LENGTH];
        let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2, 3, 4, 5]);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let got = walk_one(
            &pos_in,
            PositionOrigin {
                pos_fp: pos_start_fp,
                pay_fp: 0,
                skip: 2,
            },
            2,
            Some(pos_start_fp),
            5,
        )
        .unwrap();
        assert_eq!(got.iter().map(|p| p.position).collect::<Vec<_>>(), [3, 7]);
    }

    /// `skipPositions`' `assert posIn.getFilePointer() != lastPosBlockFP`:
    /// the vint tail is the last block there is, so skip data asking to step
    /// a whole block past it is a disagreement between `.doc` and the term's
    /// `totalTermFreq`, not something to seek blindly on.
    #[test]
    fn a_skip_that_steps_past_the_vint_tail_is_rejected() {
        let id = [71u8; ID_LENGTH];
        let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2, 3]);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let err = walk_one(
            &pos_in,
            PositionOrigin {
                pos_fp: pos_start_fp,
                pay_fp: 0,
                skip: BLOCK_SIZE as u64,
            },
            1,
            Some(pos_start_fp),
            3,
        )
        .expect_err("the tail is the last block");
        assert!(
            format!("{err}").contains("step past the last .pos block"),
            "unexpected error: {err}"
        );
    }

    /// The landing offset inside the block is a file-derived value too: it
    /// must be checked against the block's own length, not used to index it.
    #[test]
    fn a_skip_landing_past_the_blocks_own_length_is_rejected() {
        let id = [72u8; ID_LENGTH];
        let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2, 3]);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let err = walk_one(
            &pos_in,
            PositionOrigin {
                pos_fp: pos_start_fp,
                pay_fp: 0,
                skip: 5,
            },
            1,
            Some(pos_start_fp),
            3,
        )
        .expect_err("5 occurrences into a 3-occurrence block");
        assert!(
            format!("{err}").contains("occurrences into a 3-occurrence"),
            "unexpected error: {err}"
        );
    }

    /// A frequency that outruns the occurrences `.pos` actually holds must
    /// stop, not spin: the empty tail block is reached, refilled to zero
    /// occurrences, and the walk reports the disagreement.
    #[test]
    fn a_frequency_longer_than_the_position_stream_is_rejected() {
        let id = [73u8; ID_LENGTH];
        let (pos, pos_start_fp) = tail_only_pos(&id, &[1, 2]);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let err = walk_one(
            &pos_in,
            PositionOrigin {
                pos_fp: pos_start_fp,
                pay_fp: 0,
                skip: 0,
            },
            3,
            Some(pos_start_fp),
            2,
        )
        .expect_err("only two occurrences exist");
        assert!(
            format!("{err}").contains("outruns the occurrences"),
            "unexpected error: {err}"
        );
    }

    /// A `.pos` pointer beyond the file is an EOF, not a panic or a silently
    /// truncated address.
    #[test]
    fn a_position_origin_past_the_end_of_the_pos_file_is_an_error() {
        let id = [74u8; ID_LENGTH];
        let (pos, _pos_start_fp) = tail_only_pos(&id, &[1, 2]);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        assert!(walk_one(
            &pos_in,
            PositionOrigin {
                pos_fp: u64::MAX,
                pay_fp: 0,
                skip: 0,
            },
            1,
            None,
            0,
        )
        .is_err());
    }

    /// `position_origin` adds the current `.doc` block's frequencies to the
    /// skip data, so it needs frequencies: a `DocsOnly` cursor skipped them
    /// on the wire and must say so rather than summing the `1`s it filled in.
    #[test]
    fn position_origin_needs_a_freqs_cursor() {
        let id = [75u8; ID_LENGTH];
        let (mut doc, doc_footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        // Two docs in a group-varint tail block, deltas 3 and 4 (doc IDs 2
        // and 6) with freq 1 each -- `(delta << 1) | 1` is the freq-is-one
        // packing `read_tail_block` decodes.
        write_group_vints(&mut doc, &[(3 << 1) | 1, (4 << 1) | 1]);
        doc.extend_from_slice(&doc_footer);
        let doc_in = DocInput::open(&doc, &id, "").unwrap();
        let meta = TermMetadata {
            doc_start_fp,
            ..TermMetadata::EMPTY
        };
        let mut cursor = doc_in
            .lazy_cursor_with_flags(
                meta,
                2,
                IndexOptions::DocsAndFreqsAndPositions,
                false,
                PostingsFlags::DocsOnly,
            )
            .unwrap();
        // Before the first move there is no position to report at all.
        assert!(cursor.position_origin().unwrap().is_none());
        assert_eq!(cursor.next_doc().unwrap(), 2);
        let err = cursor
            .position_origin()
            .expect_err("DocsOnly skipped the frequencies the sum needs");
        assert!(
            format!("{err}").contains("PostingsFlags::Freqs"),
            "unexpected error: {err}"
        );
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

    /// Recomputes the trailing 8-byte CRC32 of a codec-footer-terminated
    /// buffer in place, so a byte flip in the body is not simply "caught" by
    /// the checksum -- c15/c19/c25's shape, and `blocktree.rs`'s
    /// `resign_footer` twin.
    fn resign_footer(buf: &mut [u8]) {
        let len = buf.len();
        let checksum = crc32fast::hash(&buf[..len - 8]) as u64;
        buf[len - 8..].copy_from_slice(&checksum.to_be_bytes());
    }

    /// Drives every `.doc`-only reader in this module over one term and
    /// reports whether any of them refused the bytes. Deliberately does *not*
    /// require the three to agree: on a corrupt file they legitimately can
    /// disagree, and the point of the sweep is that none of them panics or
    /// aborts.
    fn exercise_doc_readers(doc: &[u8], id: &[u8; ID_LENGTH], meta: TermMetadata, df: i32) -> bool {
        let Ok(input) = DocInput::open(doc, id, "") else {
            return true;
        };
        let opts = IndexOptions::DocsAndFreqs;
        let mut rejected = false;
        if input.read_postings(meta, df, opts, false).is_err() {
            rejected = true;
        }
        if input
            .read_postings_with_flags(meta, df, opts, false, PostingsFlags::DocsOnly)
            .is_err()
        {
            rejected = true;
        }
        // A full sequential walk of the lazy cursor, then a skipping one --
        // the two reach different code (`refill` vs `advance_shallow`'s
        // seek-past, and `skip_level1_to`'s span jump).
        match input.lazy_cursor(meta, df, opts, false) {
            Err(_) => rejected = true,
            Ok(mut c) => {
                // `df + 1` steps is a bound, not a heuristic: every step
                // either moves within a decoded block or consumes wire, and
                // the term claims `df` documents.
                for _ in 0..=df {
                    match c.next_doc() {
                        Ok(NO_MORE_DOCS) => break,
                        Ok(_) => {
                            let _ = c.freq();
                            let _ = c.level0_impacts();
                            let _ = c.level1_impacts();
                        }
                        Err(_) => {
                            rejected = true;
                            break;
                        }
                    }
                }
            }
        }
        match input.lazy_cursor(meta, df, opts, false) {
            Err(_) => rejected = true,
            Ok(mut c) => {
                let mut target = 0i32;
                while target < df {
                    match c.advance_shallow(target) {
                        Ok(_) => {}
                        Err(_) => {
                            rejected = true;
                            break;
                        }
                    }
                    match c.advance(target) {
                        Ok(NO_MORE_DOCS) => break,
                        Ok(_) => {}
                        Err(_) => {
                            rejected = true;
                            break;
                        }
                    }
                    target = target.saturating_add(700);
                }
            }
        }
        rejected
    }

    /// A `.doc` level-0 header for a positions-indexing field: the same
    /// shape [`write_full_block_with_impacts`] writes, plus
    /// `readLevel0PosData`'s four sub-fields (`posEndFPDelta`,
    /// `posBufferUpto`, `payEndFPDelta`, `payloadByteUpto`) inside the
    /// freq-gated region. Needed so a corruption sweep can reach
    /// [`read_pos_skip`] and [`LazyDocsCursor::position_origin`], which the
    /// no-positions helper never emits a byte of.
    fn write_full_block_with_pos_skip(
        out: &mut Vec<u8>,
        pos_end_fp_delta: i64,
        pos_buffer_upto: u8,
        pay_end_fp_delta: i64,
    ) {
        let mut meta_region = Vec::new();
        let mut impact_bytes = Vec::new();
        write_impacts(&mut impact_bytes, &[Impact { freq: 1, norm: 1 }]);
        meta_region.write_vint(impact_bytes.len() as i32);
        meta_region.write_bytes(&impact_bytes);
        meta_region.write_vlong(pos_end_fp_delta);
        meta_region.write_byte(pos_buffer_upto);
        meta_region.write_vlong(pay_end_fp_delta);
        meta_region.write_vint(0); // payloadByteUpto

        let mut block_body = Vec::new();
        block_body.write_byte(0); // bitsPerValue == 0: all 256 deltas are 1.
        block_body.write_byte(0); // PForUtil freq token: all-equal
        block_body.write_vint(1); // every freq is 1

        out.write_vlong((meta_region.len() + 4) as i64);
        out.write_i16(BLOCK_SIZE as i16); // docDelta via writeVInt15
        out.write_i16((meta_region.len() + block_body.len()) as i16); // blockLength
        out.write_bytes(&meta_region);
        out.write_bytes(&block_body);
    }

    /// Re-signed single-byte corruption sweep over the `.doc` of a
    /// *positions-indexing* term, driven through
    /// [`read_occurrences_for_doc`] -- the skip-data path, which is the half
    /// [`every_resigned_single_byte_doc_corruption_is_an_error_or_a_clean_decode`]
    /// cannot reach because its field indexes no positions.
    ///
    /// Every value this reaches comes off the flipped `.doc`: the level-0
    /// header's `posEndFPDelta`/`posBufferUpto`/`payEndFPDelta`, the
    /// frequencies the in-block sum adds up, and the `docDelta`/`blockLength`
    /// that frame them. `.pos`/`.pay` are left intact so a rejection is
    /// attributable to the `.doc` bytes alone.
    #[test]
    fn every_resigned_single_byte_positional_doc_corruption_is_an_error_or_a_clean_decode() {
        let id = [42u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        for _ in 0..2 {
            pos.write_byte(0); // full block: all 256 posDeltas are 1
            pos.write_vint(1);
        }
        let pos_blocks_len = pos.len() as u64 - pos_start_fp;
        for _ in 0..2 {
            pos.write_vint(2); // vint tail: (posDelta 1 << 1) | keep-length
        }
        pos.extend_from_slice(&pos_footer);

        let (mut pay, pay_footer) = pay_header_and_footer(&id);
        let pay_start_fp = pay.len() as u64;
        for _ in 0..2 {
            pay.write_byte(0); // payloadLengthBuffer: all zero
            pay.write_vint(0);
            pay.write_vint(0); // numBytes == 0
        }
        let pay_block_len = (pay.len() as u64 - pay_start_fp) / 2;
        pay.extend_from_slice(&pay_footer);

        let (mut doc, doc_footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        for _ in 0..2 {
            write_full_block_with_pos_skip(
                &mut doc,
                (pos_blocks_len / 2) as i64,
                0,
                pay_block_len as i64,
            );
        }
        write_group_vints(&mut doc, &[3; 2]); // tail: 2 docs, delta 1, freq 1
        doc.extend_from_slice(&doc_footer);

        let df = 2 * BLOCK_SIZE + 2;
        let ttf = df as i64; // every document has exactly one occurrence
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            pos_start_fp,
            pay_start_fp,
            last_pos_block_offset: pos_blocks_len as i64,
        };
        let opts = IndexOptions::DocsAndFreqsAndPositions;
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let pay_in = PayInput::open(&pay, &id, "").unwrap();

        let exercise = |doc: &[u8]| -> bool {
            let Ok(doc_in) = DocInput::open(doc, &id, "") else {
                return true;
            };
            let mut rejected = doc_in.read_postings(meta, df, opts, true).is_err();
            for target in [0, 1, 7, 255, 256, 300, 511, 512, 513] {
                match read_occurrences_for_doc(
                    &doc_in,
                    &pos_in,
                    Some(&pay_in),
                    meta,
                    df,
                    ttf,
                    opts,
                    true,
                    target,
                ) {
                    Ok(_) => {}
                    Err(_) => rejected = true,
                }
            }
            rejected
        };
        assert!(!exercise(&doc));

        let body = doc_start_fp as usize..doc.len() - codec_util::FOOTER_LENGTH;
        let mut total = 0usize;
        let mut rejected = 0usize;
        for off in body {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = doc.clone();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if exercise(&corrupt) {
                    rejected += 1;
                }
            }
        }
        // Measured when this was written: 49 of 60 (81.7%). As with the
        // sibling sweep, the bar is "no panic, no abort" plus a floor -- the
        // eleven accepted flips all land in a `posEndFPDelta`/`payEndFPDelta`
        // or an impacts byte, where a different-but-in-range value produces a
        // different-but-well-formed answer the format cannot contradict.
        assert_eq!(total, 60);
        assert!(
            rejected >= 45,
            "only {rejected} of {total} re-signed positional .doc corruptions were rejected"
        );
    }

    /// Re-signed single-byte corruption sweep over a whole `.doc` term body:
    /// flip one bit, **re-sign the codec footer so the checksum still
    /// passes**, and require a typed error or a clean decode from every
    /// `.doc` reader -- never a panic, and never an allocation abort
    /// `catch_unwind` cannot intercept, which through the FFI is a dead JVM.
    ///
    /// The term is deliberately the richest shape this format has: one
    /// level-1 skip entry with a real impacts run, the 32 full level-0 blocks
    /// it spans (each with its own impacts and its own `PForUtil` freq
    /// block), and a group-varint tail. So the sweep reaches
    /// `read_level1_entry`'s `skip1EndFP`/`numImpactBytes`, every level-0
    /// header field including `level0NumBytes`/`docDelta`/`blockLength`, all
    /// three `bitsPerValue` body encodings (a flip of the token byte reaches
    /// the packed and bit-set branches as well as the all-consecutive one),
    /// and the tail block's own doc-count bound.
    #[test]
    fn every_resigned_single_byte_doc_corruption_is_an_error_or_a_clean_decode() {
        let id = [41u8; ID_LENGTH];
        let (mut doc, footer) = header_and_footer(DOC_CODEC, &id);
        let doc_start_fp = doc.len() as u64;
        let mut span = Vec::new();
        for i in 0..LEVEL1_FACTOR {
            let impacts = vec![Impact {
                freq: 1 + i as i32,
                norm: 2 + i as i64,
            }];
            write_full_block_with_impacts(&mut span, true, 1 + (i as i32 % 5), &impacts);
        }
        write_level1_entry_with_impacts(
            &mut doc,
            LEVEL1_NUM_DOCS,
            &span,
            &[Impact { freq: 9, norm: 4 }],
        );
        doc.extend_from_slice(&span);
        // Tail: 8 documents, delta 1, freq 1 -- `(delta << 1) | 1`.
        write_group_vints(&mut doc, &[3; 8]);
        doc.extend_from_slice(&footer);

        let df = LEVEL1_NUM_DOCS + 8;
        let meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        // The clean file must decode, or the sweep proves nothing.
        assert!(!exercise_doc_readers(&doc, &id, meta, df));

        let body = doc_start_fp as usize..doc.len() - codec_util::FOOTER_LENGTH;
        let mut total = 0usize;
        let mut rejected = 0usize;
        for off in body {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = doc.clone();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if exercise_doc_readers(&corrupt, &id, meta, df) {
                    rejected += 1;
                }
            }
        }

        // Measured when this was written: 589 of 744 (79.2%). The rest decode
        // to a different but self-consistent postings list -- flipping a
        // freq's fill value, or an impacts delta, is a wrong answer the
        // format has no way to notice, which is exactly what the checksum
        // this sweep defeats exists for. c19/c25 measured 44/99 on `.tip`,
        // 85/99 on `.nvm`, 18/99 on `.dvd` and 15/43 on `.tvd`, so a rate
        // below 100% is the norm.
        //
        // What this pins is that nothing *panics or aborts*, plus a floor so
        // a future change that stops bounding something fails loudly.
        assert_eq!(total, 744);
        assert!(
            rejected >= 570,
            "only {rejected} of {total} re-signed .doc corruptions were rejected"
        );
    }

    #[test]
    fn a_payload_length_block_claiming_a_negative_length_is_rejected_not_a_panic() {
        // `.pay`'s payload-length stream is `PForUtil`-decoded as `u32` and
        // kept as `i32`, so a corrupt block can hand `read_positions` a
        // *negative* length. `negative as usize` sign-extends to ~2^64 and
        // the old `payload_upto + len` either panicked outright ("attempt to
        // add with overflow", debug) or wrapped to a value *below*
        // `payload_upto`, slipped past the `end > payload_bytes.len()` check
        // and panicked in `payload_bytes[start..end]` ("slice index starts at
        // 256 but ends at 255", release). Two full blocks, because the first
        // one has to move `payload_upto` off zero for the wrap to be the
        // interesting case rather than a plain "longer than the file".
        let id = [40u8; ID_LENGTH];
        let (mut pos, pos_footer) = pos_header_and_footer(&id);
        let pos_start_fp = pos.len() as u64;
        for _ in 0..2 {
            pos.write_byte(0); // PForUtil token: all 256 posDeltas equal
            pos.write_vint(1);
        }
        pos.extend_from_slice(&pos_footer);

        let (mut pay, pay_footer) = pay_header_and_footer(&id);
        let pay_start_fp = pay.len() as u64;
        // Block 0: every payload one byte long, 256 payload bytes.
        pay.write_byte(0);
        pay.write_vint(1);
        pay.write_vint(BLOCK_SIZE);
        pay.write_bytes(&[7u8; BLOCK_SIZE as usize]);
        // Block 1: every payload length is 0xFFFF_FFFF, i.e. `-1` as an
        // `i32`, with no payload bytes at all behind it.
        pay.write_byte(0);
        pay.write_vint(-1);
        pay.write_vint(0);
        pay.extend_from_slice(&pay_footer);

        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let pay_in = PayInput::open(&pay, &id, "").unwrap();
        let meta = TermMetadata {
            pos_start_fp,
            pay_start_fp,
            ..TermMetadata::EMPTY
        };
        let err = read_positions(
            &pos_in,
            Some(&pay_in),
            meta,
            &[2 * BLOCK_SIZE],
            2 * BLOCK_SIZE as i64,
            IndexOptions::DocsAndFreqsAndPositions,
            true,
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::Store(lucene_store::Error::Corrupted(ref m)) if m.contains("payload length")),
            "expected a corruption report, got {err:?}"
        );
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

    // ---------------------------------------------------------------------
    // Min-of-N A/B harness (see `docs/sweep/m2/c27-*`). Criterion is unusable
    // on this machine -- c24 measured the same code at 83/91/129 us across
    // three consecutive runs -- so this reports the **minimum** of 40 timed
    // repetitions, which is the statistic a noisy shared machine does not
    // corrupt upward. Run with:
    //
    //   ARITH_PERF_REPS=40 cargo test --release -p lucene-codecs --lib \
    //       arith_gate_perf -- --nocapture
    // ---------------------------------------------------------------------

    fn perf_doc_file(id: &[u8; ID_LENGTH]) -> (Vec<u8>, u64, i32) {
        let (mut doc, footer) = header_and_footer(DOC_CODEC, id);
        let doc_start_fp = doc.len() as u64;
        let mut span = Vec::new();
        for _ in 0..LEVEL1_FACTOR {
            write_full_block_with_impacts(&mut span, true, 3, &[Impact { freq: 2, norm: 5 }]);
        }
        write_level1_entry_with_impacts(
            &mut doc,
            LEVEL1_NUM_DOCS,
            &span,
            &[Impact { freq: 9, norm: 4 }],
        );
        doc.extend_from_slice(&span);
        write_group_vints(&mut doc, &[3; 8]);
        doc.extend_from_slice(&footer);
        (doc, doc_start_fp, LEVEL1_NUM_DOCS + 8)
    }

    /// 16 full `.pos` blocks (4 096 occurrences, one per document) with a
    /// one-byte payload each, plus the matching `.pay`.
    fn perf_pos_pay(id: &[u8; ID_LENGTH]) -> (Vec<u8>, u64, Vec<u8>, u64, usize) {
        const BLOCKS: usize = 16;
        let (mut pos, pos_footer) = pos_header_and_footer(id);
        let pos_start_fp = pos.len() as u64;
        for _ in 0..BLOCKS {
            pos.write_byte(0);
            pos.write_vint(1);
        }
        pos.extend_from_slice(&pos_footer);

        let (mut pay, pay_footer) = pay_header_and_footer(id);
        let pay_start_fp = pay.len() as u64;
        for _ in 0..BLOCKS {
            pay.write_byte(0);
            pay.write_vint(1); // every payload one byte long
            pay.write_vint(BLOCK_SIZE);
            pay.write_bytes(&[9u8; BLOCK_SIZE as usize]);
        }
        pay.extend_from_slice(&pay_footer);
        (
            pos,
            pos_start_fp,
            pay,
            pay_start_fp,
            BLOCKS * BLOCK_SIZE as usize,
        )
    }

    fn min_of<F: FnMut()>(reps: usize, mut f: F) -> std::time::Duration {
        let mut best = std::time::Duration::MAX;
        for _ in 0..reps {
            let t = std::time::Instant::now();
            f();
            best = best.min(t.elapsed());
        }
        best
    }

    /// Runs one repetition of each arm under `cargo test` -- enough to keep
    /// the harness itself exercised and honest -- and a real measurement when
    /// `ARITH_PERF_REPS` asks for one.
    #[test]
    fn arith_gate_perf() {
        let id = [90u8; ID_LENGTH];
        let (doc, doc_start_fp, df) = perf_doc_file(&id);
        let doc_in = DocInput::open(&doc, &id, "").unwrap();
        let doc_meta = TermMetadata {
            doc_start_fp,
            singleton_doc_id: -1,
            ..TermMetadata::EMPTY
        };
        let opts = IndexOptions::DocsAndFreqs;

        let (pos, pos_start_fp, pay, pay_start_fp, ttf) = perf_pos_pay(&id);
        let pos_in = PosInput::open(&pos, &id, "").unwrap();
        let pay_in = PayInput::open(&pay, &id, "").unwrap();
        let pos_meta = TermMetadata {
            pos_start_fp,
            pay_start_fp,
            ..TermMetadata::EMPTY
        };
        let freqs = vec![1i32; ttf];
        let wanted: Vec<usize> = (0..ttf).step_by(4).collect();
        let popts = IndexOptions::DocsAndFreqsAndPositions;

        // 1 under `cargo test`; set `ARITH_PERF_REPS=40` for a measurement.
        let reps: usize = std::env::var("ARITH_PERF_REPS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(1);

        let read_postings = min_of(reps, || {
            let p = doc_in.read_postings(doc_meta, df, opts, false).unwrap();
            std::hint::black_box(&p);
        });
        let lazy_walk = min_of(reps, || {
            let mut c = doc_in.lazy_cursor(doc_meta, df, opts, false).unwrap();
            let mut sum = 0i64;
            loop {
                match c.next_doc().unwrap() {
                    NO_MORE_DOCS => break,
                    d => sum += d as i64,
                }
            }
            std::hint::black_box(sum);
        });
        let wanted_walk = min_of(reps, || {
            let r = read_positions_for_docs(
                &pos_in,
                Some(&pay_in),
                pos_meta,
                &freqs,
                ttf as i64,
                popts,
                true,
                &wanted,
            )
            .unwrap();
            std::hint::black_box(&r);
        });
        let whole_term_positions = min_of(reps, || {
            let r = read_positions(
                &pos_in,
                Some(&pay_in),
                pos_meta,
                &freqs,
                ttf as i64,
                popts,
                true,
            )
            .unwrap();
            std::hint::black_box(&r);
        });

        println!(
            "ARITHPERF read_postings={:?} lazy_walk={:?} wanted_walk={:?} whole_term_positions={:?}",
            read_postings, lazy_walk, wanted_walk, whole_term_positions
        );
    }
}
