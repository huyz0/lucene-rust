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

use crate::field_infos::IndexOptions;
use crate::for_util::{self, ForUtil};

/// Which postings format wrote a segment's `.doc`/`.pos`/`.pay` -- the
/// `PerFieldPostingsFormat.format` a field's `FieldInfo` names, and the
/// generation every file of that format announces in its header
/// (`<name>PostingsWriterDoc`/`Pos`/`Pay`).
///
/// `Lucene104` is this module's own reader. The retired generations are read
/// by [`crate::backward_codecs`]: their `.doc` block framing and bit packing
/// differ (see `backward_codecs::postings`), while their `.pos`/`.pay` layout
/// is `Lucene104`'s at a 128-value block, which [`read_positions`] and its
/// siblings handle directly.
///
/// This is how the read path supports retired formats: a [`DocInput`]/[`PosInput`]/
/// [`PayInput`] learns its generation from its own header, the term
/// dictionary learns it from `FieldInfo` (`decodeTerm` differs too), and
/// every decode entry point dispatches on it. For a retired generation the
/// lazy cursors serve a term from its fully decoded postings: correct, but
/// without `Lucene104`'s block skipping -- see `docs/parity.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PostingsFormat {
    /// `backward_codecs.lucene90.Lucene90PostingsFormat` (9.0-9.10).
    Lucene90,
    /// `backward_codecs.lucene99.Lucene99PostingsFormat` (9.11).
    Lucene99,
    /// `backward_codecs.lucene912.Lucene912PostingsFormat` (9.12, 10.0).
    Lucene912,
    /// `backward_codecs.lucene101.Lucene101PostingsFormat` (10.1, 10.2).
    Lucene101,
    /// `backward_codecs.lucene103.Lucene103PostingsFormat` (10.3): the
    /// `Lucene101` postings under the trie term dictionary.
    Lucene103,
    /// `codecs.lucene104.Lucene104PostingsFormat`.
    #[default]
    Lucene104,
}

impl PostingsFormat {
    /// Every generation, oldest first.
    pub const ALL: [PostingsFormat; 6] = [
        PostingsFormat::Lucene90,
        PostingsFormat::Lucene99,
        PostingsFormat::Lucene912,
        PostingsFormat::Lucene101,
        PostingsFormat::Lucene103,
        PostingsFormat::Lucene104,
    ];

    /// `PostingsFormat.getName()`.
    pub fn name(self) -> &'static str {
        match self {
            PostingsFormat::Lucene90 => "Lucene90",
            PostingsFormat::Lucene99 => "Lucene99",
            PostingsFormat::Lucene912 => "Lucene912",
            PostingsFormat::Lucene101 => "Lucene101",
            PostingsFormat::Lucene103 => "Lucene103",
            PostingsFormat::Lucene104 => "Lucene104",
        }
    }

    /// `PostingsFormat.forName`, restricted to the formats this port reads.
    pub fn from_name(name: &str) -> Option<PostingsFormat> {
        Self::ALL.into_iter().find(|f| f.name() == name)
    }

    /// The format a field's `FieldInfo` names
    /// (`PerFieldPostingsFormat.format`), defaulting to `Lucene104` for a
    /// field without the attribute -- this port's own writer records none.
    /// A completion field's postings are in its format's delegate.
    pub fn of_field(field: &crate::field_infos::FieldInfo) -> Option<PostingsFormat> {
        match field
            .attributes
            .iter()
            .find(|(k, _)| k == "PerFieldPostingsFormat.format")
        {
            Some((_, v)) => Self::from_name(v).or_else(|| Self::completion_delegate(v)),
            None => Some(PostingsFormat::Lucene104),
        }
    }

    /// `CompletionXXPostingsFormat.delegatePostingsFormat()` of the release
    /// that wrote a completion field under `name`
    /// ([`crate::per_field_postings::COMPLETION_FORMAT_NAMES`]), as the
    /// `Lucene90` block tree reader -- the only one that asks -- needs it.
    /// Java reads the delegate's name out of the `.cmp` instead, which
    /// matters for `Completion101` alone: Lucene 10.1-10.2 wrote it over
    /// `Lucene101`, 10.3 over `Lucene103`, whose dictionary the current
    /// block tree reader opens without asking.
    pub fn completion_delegate(name: &str) -> Option<PostingsFormat> {
        match name {
            "Completion90" => Some(PostingsFormat::Lucene90),
            "Completion99" => Some(PostingsFormat::Lucene99),
            "Completion912" => Some(PostingsFormat::Lucene912),
            "Completion101" => Some(PostingsFormat::Lucene101),
            "Completion104" => Some(PostingsFormat::Lucene104),
            _ => None,
        }
    }

    /// The generation a postings file's header names: the codec name up to
    /// `PostingsWriter`. `None` for a file of no postings format.
    ///
    /// Read in place, without allocating: [`DocInput::validated`] calls this
    /// on every search's open of a segment. A codec name is ASCII shorter
    /// than 128 bytes (`CodecUtil.writeHeader`), so its length is one byte.
    fn of_header(buf: &[u8]) -> Option<PostingsFormat> {
        let magic = buf.get(..4)?;
        if magic != codec_util::CODEC_MAGIC.to_be_bytes() {
            return None;
        }
        let len = usize::from(*buf.get(4)?);
        let name = buf.get(5..5usize.checked_add(len)?)?;
        let name = std::str::from_utf8(name).ok()?;
        let prefix = name.split("PostingsWriter").next()?;
        Self::from_name(prefix)
    }

    /// `BLOCK_SIZE`: 256 for `Lucene104`, 128 for every retired generation.
    pub fn block_size(self) -> usize {
        match self {
            PostingsFormat::Lucene104 => BLOCK_SIZE as usize,
            _ => crate::backward_codecs::for_util::BLOCK_SIZE,
        }
    }

    /// The word the generation's `ForUtil` packs into.
    pub(crate) fn word(self) -> crate::backward_codecs::for_util::Word {
        use crate::backward_codecs::for_util::Word;
        match self {
            PostingsFormat::Lucene90 | PostingsFormat::Lucene99 | PostingsFormat::Lucene912 => {
                Word::Long
            }
            _ => Word::Int,
        }
    }

    /// Whether the term dictionary is `Lucene90BlockTreeTermsReader`'s (an
    /// FST index) rather than `Lucene103BlockTreeTermsReader`'s (a trie).
    pub fn uses_fst_terms_index(self) -> bool {
        matches!(
            self,
            PostingsFormat::Lucene90
                | PostingsFormat::Lucene99
                | PostingsFormat::Lucene912
                | PostingsFormat::Lucene101
        )
    }
}

/// `decodeTerm` for any [`PostingsFormat`]: the `Lucene104` wire layout at
/// the generation's block size, plus -- for `Lucene90`/`Lucene99`, whose skip
/// data trails the postings -- the `skipOffset` vlong of a term with more
/// than one block of documents, read and dropped (this port's readers never
/// seek to skip data; see [`PostingsFormat`]).
#[allow(clippy::too_many_arguments)]
#[inline]
pub fn decode_term_metadata_for(
    format: PostingsFormat,
    r: &mut SliceInput,
    doc_freq: i32,
    absolute: bool,
    prev: TermMetadata,
    index_options: IndexOptions,
    has_payloads: bool,
    total_term_freq: i64,
) -> Result<TermMetadata> {
    let block = format.block_size() as i64;
    let mut meta = decode_term_metadata_with_block(
        r,
        doc_freq,
        absolute,
        prev,
        index_options,
        has_payloads,
        total_term_freq,
        block,
    )?;
    if matches!(format, PostingsFormat::Lucene90 | PostingsFormat::Lucene99)
        && i64::from(doc_freq) > block
    {
        meta.skip_offset = r.read_vlong()?;
    }
    Ok(meta)
}

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
pub(crate) const LEVEL1_FACTOR: usize = 32;

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
    /// `IntBlockTermState.skipOffset` of the trailing-skip generations
    /// (`Lucene90`, `Lucene99`): where the term's skip list starts, relative
    /// to `doc_start_fp`, or `-1` when `docFreq <= 128` and it has none.
    /// Always `-1` for every other generation, whose skip data is inline.
    pub skip_offset: i64,
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
        skip_offset: -1,
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
    decode_term_metadata_with_block(
        r,
        doc_freq,
        absolute,
        prev,
        index_options,
        has_payloads,
        total_term_freq,
        BLOCK_SIZE as i64,
    )
}

/// [`decode_term_metadata`] with the format's `BLOCK_SIZE`, which decides
/// whether a `lastPosBlockOffset` is on the wire.
#[allow(clippy::too_many_arguments)]
#[inline(always)]
fn decode_term_metadata_with_block(
    r: &mut SliceInput,
    doc_freq: i32,
    absolute: bool,
    prev: TermMetadata,
    index_options: IndexOptions,
    has_payloads: bool,
    total_term_freq: i64,
    block_size: i64,
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
        if total_term_freq > block_size {
            last_pos_block_offset = r.read_vlong()?;
        }
    }

    Ok(TermMetadata {
        doc_start_fp,
        singleton_doc_id,
        pos_start_fp,
        pay_start_fp,
        last_pos_block_offset,
        skip_offset: -1,
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
    impacts.clear();
    for_each_impact(bytes, |freq, norm| impacts.push(Impact { freq, norm }))
}

/// [`decode_impacts_into`]'s walk, handing each `(freq, norm)` to `f` as it is
/// decoded instead of storing it: a block-max bound folds the pairs straight
/// into its maximum, with no list written and read back. One-byte varints --
/// nearly every delta, since both accumulate from the previous pair -- are
/// read off the slice directly; a longer one, or a truncated one, goes through
/// [`SliceInput`], so values and errors are [`decode_impacts`]'.
// ARITH: `at` only steps over bytes of `bytes` (`at < bytes.len()` on the fast
// path, `+ position()` of a reader over `bytes[at..]` on the slow one); `freq`
// and `norm` accumulate with wrapping arithmetic, as Java's `int`/`long` do.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
pub fn for_each_impact(bytes: &[u8], mut f: impl FnMut(i32, i64)) -> Result<()> {
    let mut at = 0usize;
    let mut freq: i32 = 0;
    let mut norm: i64 = 0;
    while let Some(&b) = bytes.get(at) {
        let freq_delta = if b < 0x80 {
            at += 1;
            i32::from(b)
        } else {
            let mut r = SliceInput::new(&bytes[at..]);
            let v = r.read_vint()?;
            at += r.position();
            v
        };
        // `1 + (freqDelta >>> 1)` is `int` arithmetic in Java's `readImpacts`
        // and wraps; `(freq_delta as u32) >> 1` reaches `i32::MAX` from a
        // five-byte varint with the sign bit set, so the `1 +` itself is an
        // overflow on a corrupt `.doc` before the accumulator even sees it.
        freq = freq.wrapping_add(1i32.wrapping_add(((freq_delta as u32) >> 1) as i32));
        if freq_delta & 1 != 0 {
            let delta = match bytes.get(at) {
                Some(&z) if z < 0x80 => {
                    at += 1;
                    i64::from(z >> 1) ^ -i64::from(z & 1)
                }
                _ => {
                    let mut r = SliceInput::new(&bytes[at..]);
                    let v = r.read_zlong()?;
                    at += r.position();
                    v
                }
            };
            norm = norm.wrapping_add(1i64.wrapping_add(delta));
        } else {
            norm = norm.wrapping_add(1);
        }
        f(freq, norm);
    }
    Ok(())
}

/// The current block's level-0 impacts as a cursor holds them: still encoded
/// (not asked for since the block changed), or already decoded.
pub enum Level0Impacts<'b> {
    /// The serialized run, for [`for_each_impact`]; may be empty.
    Encoded(&'b [u8]),
    /// The decoded list, empty for a block with no impacts.
    Decoded(&'b [Impact]),
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
    /// `TermsEnum.impacts(PostingsEnum.FREQS)`: frequencies are decoded, and
    /// so is each skipped-to block's impacts, for block-max pruning.
    #[default]
    Freqs,
    /// `TermsEnum.postings(PostingsEnum.FREQS)`: frequencies without impacts,
    /// what a scorer that never prunes (`COMPLETE`, `TOP_DOCS_WITH_SCORES`)
    /// asks for. Block headers it skips past are stepped over rather than
    /// read, and [`LazyDocsCursor::level0_impacts`] stays empty -- the
    /// "unknown, bound by the global maximum" answer every caller already
    /// takes from a tail block.
    FreqsNoImpacts,
}

impl PostingsFlags {
    /// `PostingsEnum.featureRequested(flags, PostingsEnum.FREQS)`.
    fn needs_freq(self) -> bool {
        matches!(self, PostingsFlags::Freqs | PostingsFlags::FreqsNoImpacts)
    }

    fn needs_impacts(self) -> bool {
        matches!(self, PostingsFlags::Freqs)
    }
}

/// An opened `.doc` file (header/footer validated once), ready for
/// per-term seeks. Mirrors `Lucene104PostingsReader`'s `docIn`, minus
/// everything this reader does not read (positions, skip data, impacts).
pub struct DocInput<'a> {
    buf: &'a [u8],
    format: PostingsFormat,
}

/// `(codec name, VERSION_START, VERSION_CURRENT)` of one of a format's
/// postings files: `kind` is `Doc`, `Pos` or `Pay`.
fn postings_header(format: PostingsFormat, kind: &str) -> (String, i32, i32) {
    let current = match format {
        // `Lucene90PostingsFormat.VERSION_CURRENT` and
        // `Lucene101PostingsFormat.VERSION_DENSE_BLOCKS_AS_BITSETS`.
        PostingsFormat::Lucene90 | PostingsFormat::Lucene101 => 1,
        PostingsFormat::Lucene104 => VERSION_CURRENT,
        _ => 0,
    };
    let codec = if format == PostingsFormat::Lucene104 {
        match kind {
            "Doc" => DOC_CODEC,
            "Pos" => POS_CODEC,
            _ => PAY_CODEC,
        }
        .to_string()
    } else {
        format!("{}PostingsWriter{kind}", format.name())
    };
    (codec, VERSION_START, current)
}

/// Opens one postings file of whichever generation its header names: the
/// shared half of [`DocInput::open`]/[`PosInput::open`]/[`PayInput::open`].
fn open_postings_file(
    buf: &[u8],
    kind: &str,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Result<PostingsFormat> {
    let format = PostingsFormat::of_header(buf).unwrap_or_default();
    let (codec, start, current) = postings_header(format, kind);
    let mut r = SliceInput::new(buf);
    codec_util::check_index_header(&mut r, &codec, start, current, segment_id, segment_suffix)?;
    codec_util::retrieve_checksum(buf)?;
    Ok(format)
}

impl<'a> DocInput<'a> {
    /// The file's bytes, header to footer.
    pub fn bytes(&self) -> &'a [u8] {
        self.buf
    }

    /// A buffer [`DocInput::open`] has already accepted: no header or footer is
    /// read again. For a reader that validates a segment's postings once and
    /// opens them per search. The generation is re-read from the header's
    /// codec name, a few bytes.
    pub fn validated(buf: &'a [u8]) -> Self {
        DocInput {
            buf,
            format: PostingsFormat::of_header(buf).unwrap_or_default(),
        }
    }
    /// Validates the `.doc` file's index header and footer checksum framing
    /// (`Lucene104PostingsReader`'s constructor, `Lucene104PostingsReader.java:134-140`),
    /// for whichever [`PostingsFormat`] the header names.
    pub fn open(doc: &'a [u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<Self> {
        let format = open_postings_file(doc, "Doc", segment_id, segment_suffix)?;
        Ok(DocInput { buf: doc, format })
    }

    /// The postings format that wrote this `.doc`.
    pub fn format(&self) -> PostingsFormat {
        self.format
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
        let mut out = Postings::default();
        self.read_postings_into(meta, doc_freq, index_options, has_payloads, flags, &mut out)?;
        Ok(out)
    }

    /// [`Self::read_postings_with_flags`] into `out`, reusing its buffers: a
    /// caller decoding term after term (a merge) allocates once, not twice
    /// per term. Under [`PostingsFlags::FreqsNoImpacts`] the impacts are
    /// stepped over, as `postings(FREQS)` steps over them in Java. On error
    /// `out` holds whatever was decoded so far.
    ///
    /// # Errors
    /// As [`Self::read_postings_with_flags`].
    pub fn read_postings_into(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
        has_payloads: bool,
        flags: PostingsFlags,
        out: &mut Postings,
    ) -> Result<()> {
        let needs_freq = flags.needs_freq();
        if self.format != PostingsFormat::Lucene104 {
            let mut p = crate::backward_codecs::postings::read_postings(
                self.buf,
                self.format,
                meta,
                doc_freq,
                index_options,
            )?;
            if !needs_freq {
                p.freqs.fill(1);
            }
            *out = p;
            return Ok(());
        }
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
            return Err(Error::Unsupported("IndexOptions::None is not supported"));
        }
        let index_has_freq = index_options != IndexOptions::Docs;
        let index_has_pos = index_options.subsumes_positions();
        let index_has_offsets_or_payloads = index_options.subsumes_offsets() || has_payloads;
        let wants_impacts = index_has_freq && flags != PostingsFlags::FreqsNoImpacts;

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
        let Postings {
            docs,
            freqs,
            level0_impacts,
            level1_impacts,
        } = out;
        docs.clear();
        freqs.clear();
        level0_impacts.clear();
        level1_impacts.clear();
        docs.reserve(n);
        freqs.reserve(n);

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
            if wants_impacts {
                level1_impacts.push((level1_last_doc_id, decode_impacts(entry.impact_bytes)?));
            }
            for _ in 0..LEVEL1_FACTOR {
                let header = read_full_block_header(
                    &mut r,
                    prev_doc_id,
                    index_has_freq,
                    index_has_pos,
                    index_has_offsets_or_payloads,
                    true,
                    true,
                )?;
                decode_full_block_body(
                    &mut r,
                    prev_doc_id,
                    index_has_freq,
                    needs_freq,
                    &mut scratch,
                    &mut block_docs,
                    &mut block_freqs,
                    false,
                    false,
                )?;
                check_wire_position(r.position(), header.body_end, "full block body")?;
                prev_doc_id = header.last_doc_id;
                if wants_impacts {
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
                true,
                true,
            )?;
            decode_full_block_body(
                &mut r,
                prev_doc_id,
                index_has_freq,
                needs_freq,
                &mut scratch,
                &mut block_docs,
                &mut block_freqs,
                false,
                false,
            )?;
            check_wire_position(r.position(), header.body_end, "full block body")?;
            prev_doc_id = header.last_doc_id;
            if wants_impacts {
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

        Ok(())
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
    /// [`LazyDocsCursor::skip_level1_to`]).
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
        self.lazy_cursor_for(meta, doc_freq, index_options, has_payloads, flags)
    }

    /// A [`LazyDocsCursor`] over a pulsed single-document term (`docFreq ==
    /// 1`), which has no `.doc` bytes at all: its one document is
    /// `meta.singleton_doc_id` and its frequency the term's `totalTermFreq`,
    /// as Lucene's `BlockPostingsEnum.reset` sets them for a singleton. The
    /// cursor behaves like one already holding a one-document tail block --
    /// no impacts, so a scorer treats it as unskippable -- which is what lets a
    /// disjunction or conjunction carry such a term as a leg instead of giving
    /// up on the whole segment.
    pub fn singleton_cursor(
        &self,
        meta: TermMetadata,
        total_term_freq: i64,
        index_options: IndexOptions,
        has_payloads: bool,
        flags: PostingsFlags,
    ) -> Result<LazyDocsCursor<'a>> {
        if meta.singleton_doc_id < 0 {
            return Err(Error::Unsupported(
                "singleton_cursor needs a pulsed term (docFreq == 1)",
            ));
        }
        let mut cursor = self.lazy_cursor_for(meta, 1, index_options, has_payloads, flags)?;
        cursor.doc_count_left = 0;
        cursor.level1_last_doc_id = NO_MORE_DOCS;
        cursor.block_docs[0] = meta.singleton_doc_id;
        // A cursor that asked for frequencies starts with a zeroed array, so
        // the slot is always written: the term's `totalTermFreq` when the field
        // indexes frequencies, 1 when it does not -- what every other block
        // shape reports for a field without freqs (`read_tail_block`,
        // `decode_full_block_body`), and what Lucene returns.
        if cursor.needs_freq {
            cursor.block_freqs[0] = if cursor.index_has_freq {
                i32::try_from(total_term_freq)
                    .map_err(|_| corrupted("singleton term's totalTermFreq overflows an int"))?
            } else {
                1
            };
        }
        cursor.block_len = 1;
        Ok(cursor)
    }

    /// What a cursor over `meta`'s postings starts from: the `.doc` reader
    /// at the term's start, the backward codecs' skip list, and the level-1
    /// setup -- the half of [`Self::lazy_cursor_for`] that
    /// [`Self::reset_lazy_cursor`] shares.
    /// The first document of a term whose postings are a single tail block
    /// (`1 < docFreq < BLOCK_SIZE`, in a format that writes them from
    /// `docStartFP` against a previous document of `-1`), read from the
    /// block's first group-varint value alone; `None` for any other term.
    ///
    /// A walk that only wants a term's documents below some cutoff learns
    /// from this alone that a rare term starts past it, without decoding
    /// the up to 255 deltas [`read_tail_block`] would.
    pub fn tail_only_first_doc(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
    ) -> Result<Option<i32>> {
        let trailing = matches!(
            self.format,
            PostingsFormat::Lucene90 | PostingsFormat::Lucene99
        );
        let bs = self.format.block_size() as i32;
        if trailing || doc_freq <= 1 || doc_freq >= bs || index_options == IndexOptions::None {
            return Ok(None);
        }
        let mut r = SliceInput::new(self.buf);
        r.seek(meta.doc_start_fp as usize)?;
        // `GroupVIntUtil.readGroupVInts`: full groups of four first (a flag
        // byte, then each value's 1-4 little-endian bytes), plain vints for
        // a remainder -- so the first value leads a group when there are at
        // least four.
        let raw = if doc_freq >= 4 {
            let flag = r.read_byte()?;
            let mut word = [0u8; 4];
            r.read_bytes(&mut word[..usize::from(flag >> 6) + 1])?;
            u32::from_le_bytes(word)
        } else {
            r.read_vint()? as u32
        };
        // As `read_tail_block`: the low bit is the freq-is-one flag when the
        // field has frequencies.
        let delta = if index_options == IndexOptions::Docs {
            raw
        } else {
            raw >> 1
        };
        Ok(Some((-1i32).wrapping_add(delta as i32)))
    }

    fn cursor_prelude(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
        has_payloads: bool,
    ) -> Result<CursorPrelude<'a>> {
        if !matches!(
            index_options,
            IndexOptions::Docs
                | IndexOptions::DocsAndFreqs
                | IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
                | IndexOptions::DocsAndCustomFreqs
        ) {
            return Err(Error::Unsupported("IndexOptions::None is not supported"));
        }
        let trailing = matches!(
            self.format,
            PostingsFormat::Lucene90 | PostingsFormat::Lucene99
        );
        // `BlockImpactsDocsEnum`'s `skipper.init(docStartFP + skipOffset,
        // docStartFP, posStartFP, payStartFP, docFreq)`: only a term with more
        // than one block has a skip list at all.
        let skip = if trailing && meta.skip_offset >= 0 && doc_freq > 1 {
            let fields = crate::backward_codecs::skip_list::SkipFields {
                has_pos: index_options.subsumes_positions(),
                has_offsets: index_options.subsumes_offsets(),
                has_payloads,
            };
            Some(Box::new(crate::backward_codecs::skip_list::SkipList::new(
                self.buf,
                fields,
                (meta.doc_start_fp as i64).wrapping_add(meta.skip_offset),
                meta.doc_start_fp as i64,
                meta.pos_start_fp as i64,
                meta.pay_start_fp as i64,
                doc_freq,
            )?))
        } else {
            None
        };
        let mut r = SliceInput::new(self.buf);
        r.seek(meta.doc_start_fp as usize)?;

        // Mirror `Lucene104PostingsReader.BlockPostingsEnum.reset`'s level-1
        // setup (`Lucene104PostingsReader.java:559-568`): below
        // `LEVEL1_NUM_DOCS` there are no level-1 entries on the wire, so pin
        // `level1_last_doc_id` at NO_MORE_DOCS to disable the level-1 skip
        // path entirely (`target > NO_MORE_DOCS` is never true). At or above
        // it, start the running last-doc at `-1` with `level1_doc_end_fp`
        // pointing at the first level-1 entry (which sits at `docStartFP`).
        let bs = self.format.block_size() as i32;
        // ARITH: `block_size()` is 128 or 256.
        #[allow(clippy::arithmetic_side_effects)]
        let level1_docs = bs * 32;
        let level1_last_doc_id = if doc_freq < level1_docs || trailing {
            NO_MORE_DOCS
        } else {
            -1
        };
        Ok(CursorPrelude {
            r,
            skip,
            trailing,
            bs,
            level1_docs,
            level1_last_doc_id,
        })
    }

    /// `termsEnum.postings(reuse, flags)` with `reuse` a cursor this `.doc`
    /// handed out for another term of the same field: `cursor` re-pointed at
    /// `meta`'s postings as [`Self::lazy_cursor_with_flags`] would open them,
    /// its block buffers kept rather than built again. Those are kilobytes,
    /// and a multi-term query's bit-set union opens a cursor per term --
    /// thousands for a short prefix -- which is why Lucene's `TermsEnum`
    /// takes a `reuse` at all.
    pub fn reset_lazy_cursor(
        &self,
        cursor: &mut LazyDocsCursor<'a>,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
        has_payloads: bool,
        flags: PostingsFlags,
    ) -> Result<()> {
        if doc_freq <= 1 {
            return Err(Error::Unsupported(
                "docFreq <= 1: use singleton_postings instead (no .doc bytes are written)",
            ));
        }
        let CursorPrelude {
            r,
            skip,
            trailing,
            bs,
            level1_docs,
            level1_last_doc_id,
        } = self.cursor_prelude(meta, doc_freq, index_options, has_payloads)?;
        let c = cursor;
        if !flags.needs_freq() && (c.needs_freq || c.block_freqs[0] != 1) {
            // A cursor that never decodes frequencies reads all ones.
            c.block_freqs = [1; BLOCK_SIZE as usize];
        }
        c.r = r;
        c.index_has_freq = index_options != IndexOptions::Docs;
        c.needs_freq = flags.needs_freq();
        c.needs_impacts = flags.needs_impacts();
        c.needs_pos = false;
        c.index_has_pos = index_options.subsumes_positions();
        c.index_has_offsets_or_payloads = index_options.subsumes_offsets() || has_payloads;
        c.doc_freq = doc_freq;
        c.prev_doc_id = if trailing { 0 } else { -1 };
        c.doc_count_left = doc_freq;
        c.level1_last_doc_id = level1_last_doc_id;
        c.level1_doc_end_fp = meta.doc_start_fp as usize;
        c.level1_doc_count_upto = 0;
        c.block_len = 0;
        c.block_pos = 0;
        c.doc_id = -1;
        c.bits = None;
        c.bits_stepped = false;
        c.pending = None;
        c.level0_last_doc_id = -1;
        c.level0_impacts = Impacts::new();
        c.level0_impact_bytes = &[];
        c.level0_impacts_stale = false;
        c.level1_impacts = Impacts::new();
        let start = PosCursorState {
            pos_fp: meta.pos_start_fp,
            pay_fp: meta.pay_start_fp,
            pos_buffer_upto: 0,
        };
        c.level0_pos = start;
        c.level1_pos = start;
        c.block_pos_origin = start;
        c.block_gen = 0;
        c.format = self.format;
        c.bs = bs;
        c.level1_docs = level1_docs;
        c.skip = skip;
        Ok(())
    }

    /// The shared body of [`Self::lazy_cursor_with_flags`] and
    /// [`Self::singleton_cursor`], past their `docFreq` checks.
    fn lazy_cursor_for(
        &self,
        meta: TermMetadata,
        doc_freq: i32,
        index_options: IndexOptions,
        has_payloads: bool,
        flags: PostingsFlags,
    ) -> Result<LazyDocsCursor<'a>> {
        let CursorPrelude {
            r,
            skip,
            trailing,
            bs,
            level1_docs,
            level1_last_doc_id,
        } = self.cursor_prelude(meta, doc_freq, index_options, has_payloads)?;

        Ok(LazyDocsCursor {
            r,
            index_has_freq: index_options != IndexOptions::Docs,
            needs_freq: flags.needs_freq(),
            needs_impacts: flags.needs_impacts(),
            needs_pos: false,
            index_has_pos: index_options.subsumes_positions(),
            index_has_offsets_or_payloads: index_options.subsumes_offsets() || has_payloads,
            doc_freq,
            // `Lucene90`/`Lucene99` code a term's first document as its own id
            // (`accum = 0`); every later generation as one past `-1`.
            prev_doc_id: if trailing { 0 } else { -1 },
            doc_count_left: doc_freq,
            level1_last_doc_id,
            level1_doc_end_fp: meta.doc_start_fp as usize,
            level1_doc_count_upto: 0,
            block_docs: [0; BLOCK_SIZE as usize],
            // All ones for a cursor that never decodes frequencies: the block
            // decoder then leaves the array alone (`unit_freqs`).
            block_freqs: [if flags.needs_freq() { 0 } else { 1 }; BLOCK_SIZE as usize],
            block_len: 0,
            block_pos: 0,
            doc_id: -1,
            bits: None,
            bits_stepped: false,
            scratch: BlockScratch::new(),
            pending: None,
            level0_last_doc_id: -1,
            level0_impacts: Impacts::new(),
            level0_impact_bytes: &[],
            level0_impacts_stale: false,
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
            block_gen: 0,
            format: self.format,
            bs,
            level1_docs,
            skip,
        })
    }
}

/// [`DocInput::cursor_prelude`]'s result.
struct CursorPrelude<'a> {
    r: SliceInput<'a>,
    skip: Option<Box<crate::backward_codecs::skip_list::SkipList<'a>>>,
    trailing: bool,
    bs: i32,
    level1_docs: i32,
    level1_last_doc_id: i32,
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
    format: PostingsFormat,
}

impl<'a> PosInput<'a> {
    /// The file's bytes, header to footer.
    pub fn bytes(&self) -> &'a [u8] {
        self.buf
    }

    /// A buffer [`PosInput::open`] has already accepted: no header or footer is
    /// read again. For a reader that validates a segment's postings once and
    /// opens them per search.
    pub fn validated(buf: &'a [u8]) -> Self {
        PosInput {
            buf,
            format: PostingsFormat::of_header(buf).unwrap_or_default(),
        }
    }
    /// `Lucene104PostingsReader`'s constructor, the `.pos` branch
    /// (`Lucene104PostingsReader.java:142-149`), for whichever
    /// [`PostingsFormat`] the header names.
    pub fn open(pos: &'a [u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<Self> {
        let format = open_postings_file(pos, "Pos", segment_id, segment_suffix)?;
        Ok(PosInput { buf: pos, format })
    }

    /// The postings format that wrote this `.pos`.
    pub fn format(&self) -> PostingsFormat {
        self.format
    }
}

/// An opened `.pay` file (header/footer validated once), analogous to
/// [`DocInput`]. Only opened for fields with offsets and/or payloads
/// (`Lucene104PostingsReader.java:151-161`).
pub struct PayInput<'a> {
    buf: &'a [u8],
}

impl<'a> PayInput<'a> {
    /// A buffer [`PayInput::open`] has already accepted: no header or footer is
    /// read again. For a reader that validates a segment's postings once and
    /// opens them per search.
    pub fn validated(buf: &'a [u8]) -> Self {
        PayInput { buf }
    }
    /// The `.pay` branch of the postings reader's constructor, for whichever
    /// [`PostingsFormat`] the header names. `.pay` is only ever read alongside
    /// its `.pos`, whose generation decides the layout.
    pub fn open(pay: &'a [u8], segment_id: &[u8; ID_LENGTH], segment_suffix: &str) -> Result<Self> {
        open_postings_file(pay, "Pay", segment_id, segment_suffix)?;
        Ok(PayInput { buf: pay })
    }
}

/// The flat, `total_term_freq`-long streams a term's `.pos`/`.pay` data
/// decodes into, before they are re-chopped into per-document groups.
///
/// Split out so a caller wanting only positions -- phrase matching, which is
/// the caller that matters for speed -- can assemble them without building a
/// `Vec<Position>` per document. See [`read_positions_flat`].
#[derive(Default)]
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
    let mut streams = PositionStreams::default();
    decode_position_streams_into(
        pos,
        pay,
        meta,
        total_term_freq,
        index_options,
        has_payloads,
        &mut streams,
    )?;
    Ok(streams)
}

/// [`decode_position_streams`] into `streams`, reusing its buffers.
fn decode_position_streams_into(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
    streams: &mut PositionStreams,
) -> Result<()> {
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
    let PositionStreams {
        pos_deltas,
        payload_lengths,
        payload_bytes,
        offset_start_deltas,
        offset_lengths,
    } = streams;
    pos_deltas.clear();
    payload_lengths.clear();
    payload_bytes.clear();
    offset_start_deltas.clear();
    offset_lengths.clear();
    pos_deltas.reserve(reserve);
    if has_payloads {
        payload_lengths.reserve(reserve);
    }
    if has_offsets {
        offset_start_deltas.reserve(reserve);
        offset_lengths.reserve(reserve);
    }

    // `meta.last_pos_block_offset` (already decoded by `decode_term_metadata`)
    // tells us exactly where the vint tail block begins on the wire, which is
    // equivalent to (but doesn't require us to re-derive live, unlike the
    // real reader's `posIn.getFilePointer() == lastPosBlockFP` check) simply
    // computing how many full 256-position blocks precede it from
    // `total_term_freq` itself.
    let format = pos.format;
    let (num_full_blocks, tail_count) = if format == PostingsFormat::Lucene104 {
        full_blocks_and_tail(n)
    } else {
        let b = format.block_size();
        // ARITH: `b` is 128.
        #[allow(clippy::arithmetic_side_effects)]
        let split = (n / b, n % b);
        split
    };

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
        pfor_block(format, &mut pos_r, |d| pos_deltas.push(d as i32))?;

        if has_payloads {
            let pay_r = pay_r
                .as_mut()
                .expect("checked above: has_payloads implies pay.is_some()");
            let mut lens: Vec<u32> = Vec::with_capacity(format.block_size());
            pfor_block(format, pay_r, |l| lens.push(l))?;
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
            pfor_block(format, pay_r, |s| offset_start_deltas.push(s as i32))?;
            pfor_block(format, pay_r, |l| offset_lengths.push(l as i32))?;
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

    Ok(())
}

/// One full `.pos`/`.pay` `PForUtil` block of `format`, each value handed to
/// `sink` in order: 256 values through [`for_util::pfor_decode`] for
/// `Lucene104`, 128 through the retired generations' decoder otherwise.
fn pfor_block(format: PostingsFormat, r: &mut SliceInput, mut sink: impl FnMut(u32)) -> Result<()> {
    if format == PostingsFormat::Lucene104 {
        let mut block = [0u32; for_util::BLOCK_SIZE];
        for_util::pfor_decode(r, &mut block)?;
        block.iter().for_each(|&v| sink(v));
    } else {
        let mut block = [0u64; crate::backward_codecs::for_util::BLOCK_SIZE];
        crate::backward_codecs::for_util::pfor_decode(r, format.word(), &mut block)?;
        block.iter().for_each(|&v| sink(v as u32));
    }
    Ok(())
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
    doc_starts.push(stream_len(positions.len())?);
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
/// Implementations are monomorphized so each pays only for its own work:
/// [`PositionsOnly`] for phrase matching, [`FullOccurrences`] for
/// highlighting, and a shared-terms merge sink that streams into its target.
/// `NEEDS_EXTRAS` decides whether `.pay`'s offset and payload
/// blocks are unpacked or stepped over with `PForUtil.skip`, which is the same
/// job `PostingsEnum`'s flag mask does one level up in Java.
pub(crate) trait OccurrenceSink {
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
/// time it reaches the next `wanted` index. A prefix-sum array would be
/// `docFreq + 1` entries long whatever the caller asked for -- 20 MB for a
/// five-million-document term, to answer a question about one document. Java
/// needs neither: `.doc`'s skip data carries the `.pos` file
/// pointers, so `advance` never sums a frequency at all (see
/// [`walk_wanted_occurrences`]).
///
/// An entry the doc list does not have, or one that is not strictly after the
/// entry before it (an unsorted or repeated `wanted`), gets an empty range at
/// the current watermark: it keeps its slot in the result and contributes
/// nothing. The alternative -- indexing a prefix-sum array by whatever the
/// caller passed -- is an out-of-bounds panic on a caller's mistake.
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
        // ARITH: on entry `*acc <= n <= i64::MAX` (the check below returned
        // otherwise, and `n` came through `wire_count`), and `freq` is a
        // non-negative `i32`, so the sum is under 2^64.
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
            // running sum reaches `freqs[d]` -- both because a `usize` has no
            // room to spare above `n` when `n` is `usize::MAX` (`wire_count`
            // lets through any non-negative count that fits), and because a
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

/// [`walk_document_occurrences`]' decode buffers, held by the caller so a
/// walk per document does not zero four 256-entry arrays (and a payload
/// buffer) each time.
pub(crate) struct WalkScratch {
    for_util: for_util::ForUtil,
    block: PositionBlock,
}

impl WalkScratch {
    pub(crate) fn new() -> Self {
        WalkScratch {
            for_util: for_util::ForUtil::new(),
            block: PositionBlock::new(),
        }
    }
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
/// A position walk from the term's `posStartFP` can count occurrences, but a
/// walk that *jumps into the middle* of `.pos` has no occurrence count to
/// derive the split from, so this is the only thing that tells a full block
/// from the tail -- exactly the role it has in `refillPositions`.
fn last_pos_block_fp(meta: TermMetadata, total_term_freq: i64) -> Option<u64> {
    last_pos_block_fp_for(meta, total_term_freq, BLOCK_SIZE as i64)
}

/// [`last_pos_block_fp`] for a generation of block size `bs` (128 for every
/// retired one): the vint tail's start, `None` when the positions end on a
/// block boundary.
fn last_pos_block_fp_for(meta: TermMetadata, total_term_freq: i64, bs: i64) -> Option<u64> {
    match total_term_freq.cmp(&bs) {
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
    scratch: &mut WalkScratch,
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

    let WalkScratch {
        for_util: for_util_state,
        block,
    } = scratch;
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
            refill_full_position_block(pos_r, pay_r, wants, for_util_state, block)
        }
    };
    refill(&mut pos_r, pay_r.as_mut(), block, &mut tail_decoded)?;

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
            refill(&mut pos_r, pay_r.as_mut(), block, &mut tail_decoded)?;
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
/// highlighter asks and what the skip data can address; the all-documents form
/// is still what phrase matching wants, because its `wanted` set is a large
/// fraction of the term's doc list and it has already decoded that list to
/// intersect it.
#[allow(clippy::too_many_arguments)]
pub(crate) fn walk_wanted_occurrences<S: OccurrenceSink>(
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
            // The early exit: if every wanted document is
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
/// term's. Building them all is wasted: on the benchmark corpus `phrase t0 t1`
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
    if pos.format != PostingsFormat::Lucene104 {
        // A retired format: every delta, then the wanted documents' ranges
        // of them, each summed from 0 (a document's first delta is absolute).
        let n = wire_count(total_term_freq, "total_term_freq")?;
        let deltas = crate::backward_codecs::postings::read_position_deltas(
            pos.buf,
            pos.format,
            meta,
            total_term_freq,
            index_options.subsumes_offsets(),
            has_payloads,
        )?;
        let ranges = wanted_ranges(wanted, freqs, n)?;
        let mut positions = Vec::new();
        let mut doc_starts = Vec::with_capacity(ranges.len().saturating_add(1));
        for (a, b) in ranges {
            doc_starts.push(positions.len() as u32);
            let mut p = 0i32;
            for &d in deltas.get(a..b).unwrap_or(&[]) {
                p = p.wrapping_add(d);
                positions.push(p);
            }
        }
        doc_starts.push(stream_len(positions.len())?);
        return Ok((positions, doc_starts));
    }
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
    sink.doc_starts.push(stream_len(sink.positions.len())?);
    Ok((sink.positions, sink.doc_starts))
}

/// [`read_positions_for_docs`]'s offsets- and payloads-carrying sibling:
/// whole [`Position`] records for just the documents `wanted` names.
///
/// This is what `PostingsEnum.advance(doc)` + `nextPosition()` /
/// `startOffset()` / `endOffset()` / `getPayload()` gives Java's
/// `PostingsOffsetStrategy`, and it exists because the offset-carrying
/// [`read_positions`] returns every document's offsets, so highlighting one
/// document would cost a full postings sweep plus a `Vec<Position>` per
/// document in the term.
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
    if pos.format != PostingsFormat::Lucene104 {
        // A retired format: every occurrence, then the wanted documents'.
        let n = wire_count(total_term_freq, "total_term_freq")?;
        let ranges = wanted_ranges(wanted, freqs, n)?;
        let all = read_positions(
            pos,
            pay,
            meta,
            freqs,
            total_term_freq,
            index_options,
            has_payloads,
        )?;
        let flat: Vec<Position> = all.into_iter().flatten().collect();
        let mut occurrences = Vec::new();
        let mut doc_starts = Vec::with_capacity(ranges.len().saturating_add(1));
        for (a, b) in ranges {
            doc_starts.push(occurrences.len() as u32);
            occurrences.extend_from_slice(flat.get(a..b).unwrap_or(&[]));
        }
        doc_starts.push(stream_len(occurrences.len())?);
        return Ok((occurrences, doc_starts));
    }
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
    sink.doc_starts.push(stream_len(sink.occurrences.len())?);
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
    if doc.format != PostingsFormat::Lucene104 {
        // A retired format has no skip data this port walks: find the
        // document in the decoded list, then its occurrences.
        let postings = doc.read_postings(meta, doc_freq, index_options, has_payloads)?;
        let Ok(i) = postings.docs.binary_search(&doc_id) else {
            return Ok(None);
        };
        let mut all = read_positions(
            pos,
            pay,
            meta,
            &postings.freqs,
            total_term_freq,
            index_options,
            has_payloads,
        )?;
        if i >= all.len() {
            return Err(corrupted("document past the term's decoded positions"));
        }
        return Ok(Some(all.swap_remove(i)));
    }
    let mut cursor = doc.lazy_cursor(meta, doc_freq, index_options, has_payloads)?;
    cursor.track_positions();
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
        &mut WalkScratch::new(),
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

/// One term's absolute occurrence streams, retaining one allocation per stream.
///
/// [`read_occurrences_flat`] shares the wire decoder with [`read_positions`],
/// transforms its position/offset arrays in place, and moves its payload bytes.
/// Document boundaries are private and validated before exposing borrowed slices.
/// This representation targets measured merge decode/materialization cost; it
/// does not skip validation of occurrences belonging to deleted documents.
#[derive(Debug, Default)]
pub struct FlatOccurrences {
    positions: Vec<i32>,
    start_offsets: Vec<i32>,
    end_offsets: Vec<i32>,
    payload_lengths: Vec<u32>,
    payload_bytes: Vec<u8>,
    doc_starts: Vec<usize>,
    payload_starts: Vec<usize>,
    has_offsets: bool,
    has_payloads: bool,
}

/// Borrowed occurrences of one document in a [`FlatOccurrences`] result.
/// Offsets and payload streams are empty when the field does not index them.
/// Otherwise the offset/length slices have one entry per position; payload
/// lengths partition `payload_bytes` in occurrence order, including zero lengths.
#[derive(Debug, PartialEq, Eq)]
pub struct DocOccurrences<'a> {
    pub positions: &'a [i32],
    pub start_offsets: &'a [i32],
    pub end_offsets: &'a [i32],
    pub payload_lengths: &'a [u32],
    pub payload_bytes: &'a [u8],
}

impl FlatOccurrences {
    /// The decoded occurrences at this source document's postings-list index.
    /// Returns `None` outside that list or if any private boundary is invalid.
    pub fn document(&self, index: usize) -> Option<DocOccurrences<'_>> {
        let next = index.checked_add(1)?;
        let start = *self.doc_starts.get(index)?;
        let end = *self.doc_starts.get(next)?;
        let positions = self.positions.get(start..end)?;
        let (start_offsets, end_offsets) = if self.has_offsets {
            (
                self.start_offsets.get(start..end)?,
                self.end_offsets.get(start..end)?,
            )
        } else {
            (&[][..], &[][..])
        };
        let (payload_lengths, payload_bytes) = if self.has_payloads {
            let from = *self.payload_starts.get(index)?;
            let to = *self.payload_starts.get(next)?;
            (
                self.payload_lengths.get(start..end)?,
                self.payload_bytes.get(from..to)?,
            )
        } else {
            (&[][..], &[][..])
        };
        Some(DocOccurrences {
            positions,
            start_offsets,
            end_offsets,
            payload_lengths,
            payload_bytes,
        })
    }
}

/// [`read_positions`] without intermediate per-document or per-payload buffers.
///
/// Java: `Lucene104PostingsReader.BlockPostingsEnum.nextPosition`; accumulators
/// reset at document boundaries and wrap as Java `int`s do. Retired formats use
/// their own 128-occurrence blocks through the same wire decoder. Every source
/// occurrence is checked before the result is returned, including those a merge
/// later discards. Frequency reconciliation and payload range errors match
/// [`read_positions`].
pub fn read_occurrences_flat(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    freqs: &[i32],
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
) -> Result<FlatOccurrences> {
    let mut out = FlatOccurrences::default();
    read_occurrences_flat_into(
        pos,
        pay,
        meta,
        freqs,
        total_term_freq,
        index_options,
        has_payloads,
        &mut out,
    )?;
    Ok(out)
}

/// [`read_occurrences_flat`] into `out`, reusing its buffers: a merge reads
/// every term of every source through one. On error `out` is left empty or
/// partly filled, never with stale boundaries over fresh streams.
///
/// # Errors
/// As [`read_occurrences_flat`].
#[allow(clippy::too_many_arguments)]
pub fn read_occurrences_flat_into(
    pos: &PosInput<'_>,
    pay: Option<&PayInput<'_>>,
    meta: TermMetadata,
    freqs: &[i32],
    total_term_freq: i64,
    index_options: IndexOptions,
    has_payloads: bool,
    out: &mut FlatOccurrences,
) -> Result<()> {
    let has_offsets = index_options.subsumes_offsets();
    let n = wire_count(total_term_freq, "total_term_freq")?;
    // The previous term's buffers become this one's streams; the payload
    // lengths change type in place (same layout, so the collect reuses the
    // allocation).
    let mut streams = PositionStreams {
        pos_deltas: std::mem::take(&mut out.positions),
        payload_lengths: std::mem::take(&mut out.payload_lengths)
            .into_iter()
            .map(|len| len as i32)
            .collect(),
        payload_bytes: std::mem::take(&mut out.payload_bytes),
        offset_start_deltas: std::mem::take(&mut out.start_offsets),
        offset_lengths: std::mem::take(&mut out.end_offsets),
    };
    let mut doc_starts = std::mem::take(&mut out.doc_starts);
    let mut payload_starts = std::mem::take(&mut out.payload_starts);
    decode_position_streams_into(
        pos,
        pay,
        meta,
        total_term_freq,
        index_options,
        has_payloads,
        &mut streams,
    )?;
    let PositionStreams {
        mut pos_deltas,
        payload_lengths,
        payload_bytes,
        mut offset_start_deltas,
        mut offset_lengths,
    } = streams;
    // ARITH: `freqs` is a live slice of i32, so its length is at most
    // isize::MAX / 4; adding the terminal boundary cannot overflow usize.
    #[allow(clippy::arithmetic_side_effects)]
    let boundaries = freqs.len() + 1;
    doc_starts.clear();
    payload_starts.clear();
    doc_starts.reserve(boundaries);
    if has_payloads {
        payload_starts.reserve(boundaries);
    }
    let mut payload_upto = 0usize;
    let mut idx = 0usize;
    for &freq in freqs {
        doc_starts.push(idx);
        if has_payloads {
            payload_starts.push(payload_upto);
        }
        let mut position = 0i32;
        let mut start_offset = 0i32;
        for _ in 0..freq {
            if idx >= pos_deltas.len() {
                return Err(corrupted("sum of per-doc freqs exceeds total_term_freq"));
            }
            position = position.wrapping_add(pos_deltas[idx]);
            pos_deltas[idx] = position;
            if has_payloads {
                let end = usize::try_from(payload_lengths[idx])
                    .ok()
                    .and_then(|len| payload_upto.checked_add(len));
                let Some(end) = end.filter(|&end| payload_bytes.get(payload_upto..end).is_some())
                else {
                    return Err(corrupted(format!(
                        "payload length {} at occurrence {idx} exceeds the {} decoded \
                         payload bytes from offset {payload_upto}",
                        payload_lengths[idx],
                        payload_bytes.len()
                    )));
                };
                payload_upto = end;
            }
            if has_offsets {
                start_offset = start_offset.wrapping_add(offset_start_deltas[idx]);
                offset_start_deltas[idx] = start_offset;
                offset_lengths[idx] = start_offset.wrapping_add(offset_lengths[idx]);
            }
            // ARITH: the loop's guard established `idx < pos_deltas.len()`,
            // and a live slice is at most isize::MAX elements long.
            #[allow(clippy::arithmetic_side_effects)]
            {
                idx += 1;
            }
        }
    }
    doc_starts.push(idx);
    if has_payloads {
        payload_starts.push(payload_upto);
    }
    if idx != n {
        return Err(corrupted(
            "sum of per-doc freqs is less than total_term_freq",
        ));
    }
    // Every retained length was checked non-negative and within payload_bytes
    // above. Converting the stream does not allocate one buffer per occurrence.
    let payload_lengths = payload_lengths.into_iter().map(|len| len as u32).collect();
    *out = FlatOccurrences {
        positions: pos_deltas,
        start_offsets: offset_start_deltas,
        end_offsets: offset_lengths,
        payload_lengths,
        payload_bytes,
        doc_starts,
        payload_starts,
        has_offsets,
        has_payloads,
    };
    Ok(())
}

/// `writeVInt15`'s companion reader
/// (`Lucene104PostingsReader.readVInt15`): a 2-byte fast path for values that
/// fit in 15 bits, else the top bit of the `short` flags a following vint
/// carrying the remaining high bits (`value = (s & 0x7FFF) | (extra << 15)`).
#[inline]
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
#[inline]
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
    /// exactly as [`FullBlockHeader::impact_bytes`] is and for the same reason:
    /// `skipLevel1To` decodes the run only when `needsImpacts &&
    /// level1LastDocID >= target` and `skipBytes` past it otherwise, so a walk
    /// crossing 610 spans to reach a document does not decode -- or allocate --
    /// 610 impact lists it never looks at. (Otherwise it is the dominant cost
    /// of a skip-driven `advance` on a five-million-document term.)
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
/// decoded into [`Level1Entry::impact_bytes`] via [`decode_impacts`]. The pos/pay
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
/// only if a caller asks for a bound) — so determining `last_doc_id` and
/// `body_start`/`body_len` never runs `ForUtil`/`PForUtil` decode, which is the
/// expensive part of a block (bit-unpacking 256 values). That decode work is
/// exactly what [`LazyDocsCursor`] avoids for a block this header proves is
/// entirely before the caller's target.
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
    needs_impacts: bool,
    needs_pos: bool,
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
    // A cursor that wants neither the impacts nor the `.pos`/`.pay` origin
    // never comes here: `LazyDocsCursor::skip_level0_headers` is
    // `skipLevel0To`'s `docIn.seek(skip0End)` for it.
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
        if needs_impacts {
            let impacts_start = r.position();
            impact_bytes = r.slice(impacts_start, add_wire_offset(impacts_start, impacts_len)?)?;
        }
        r.skip(impacts_len)?;

        // Level-0 pos/pay skip data (`Lucene104PostingsReader.java:754-761`,
        // `readLevel0PosData`): where in `.pos`/`.pay` this block's documents'
        // occurrences begin. [`LazyDocsCursor`] accumulates it so a
        // positional walk can seek straight there instead of summing every
        // preceding document's frequency.
        if index_has_pos {
            if needs_pos {
                pos_skip = Some(read_pos_skip(r, index_has_offsets_or_payloads)?);
            } else {
                r.seek(skip0_end)?;
            }
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
    /// `docCumulativeWordPopCounts`: set bits in `bitset[..w]`, filled when a
    /// bit-set block is kept, so a document's rank is one lookup and one
    /// popcount rather than a count over every word in between.
    word_ranks: [u16; 128],
}

impl BlockScratch {
    fn new() -> Self {
        Self {
            for_util: ForUtil::new(),
            words: [0u32; for_util::BLOCK_SIZE],
            bitset: [0u64; 128],
            word_ranks: [0u16; 128],
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
/// A document's frequencies claim more positions than the term's
/// `totalTermFreq` left in `.pos`.
fn positions_overrun() -> Error {
    corrupted("positions run past the term's totalTermFreq")
}

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
/// A negative count is corruption: no writer emits one. Any other value is a
/// real one -- `totalTermFreq` is a `long` in Lucene, and a stop-word in a
/// segment of hundreds of millions of documents really can exceed 2^32
/// occurrences (`IndexWriter.MAX_DOCS` documents, each with an `int`
/// frequency) -- and every walk here counts occurrences in `usize`, so only a
/// 32-bit target, whose `usize` cannot hold such a count, refuses it. The
/// whole-term readers that *materialise* a stream still index it with `u32`
/// ([`stream_len`]); the per-document and wanted-documents walks and
/// [`PositionsCursor`] never hold more than the documents asked for.
fn wire_count(value: i64, what: &str) -> Result<usize> {
    if value < 0 {
        return Err(corrupted(format!("{what}: negative count {value}")));
    }
    usize::try_from(value).map_err(|_| {
        Error::Unsupported(
            "total_term_freq exceeds this platform's usize: a term with more occurrences \
             than a 32-bit target can count cannot be walked there",
        )
    })
}

/// The `u32` end offset of a materialised occurrence stream: the flat
/// readers ([`read_positions_flat`], [`read_positions_for_docs`],
/// [`read_occurrences_for_docs`]) address their output with `u32` document
/// starts, so a stream longer than `u32::MAX` entries -- 16 GiB of positions
/// held in memory at once -- is refused rather than truncated. Checked once,
/// on the final length: every earlier start is at most it.
fn stream_len(len: usize) -> Result<u32> {
    u32::try_from(len).map_err(|_| {
        Error::Unsupported(
            "a materialised position stream past u32::MAX entries: walk the term with \
             PositionsCursor or per document instead",
        )
    })
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
#[inline]
fn check_wire_position(position: usize, expected: usize, what: &str) -> Result<()> {
    if position != expected {
        return Err(wire_position_mismatch(position, expected, what));
    }
    Ok(())
}

/// [`check_wire_position`]'s error, kept out of line so the check itself
/// inlines into the block loops as one compare.
#[cold]
#[inline(never)]
fn wire_position_mismatch(position: usize, expected: usize, what: &str) -> Error {
    corrupted(format!(
        "{what}: decode ended at {position} but the file's own length field claims {expected}"
    ))
}

/// Scratch for decoding doc blocks ([`Self::decode`]), reused across blocks.
pub struct DocBlockDecoder {
    scratch: Box<BlockScratch>,
    docs: Box<[i32; BLOCK_SIZE as usize]>,
    freqs: Box<[i32; BLOCK_SIZE as usize]>,
}

impl Default for DocBlockDecoder {
    fn default() -> Self {
        Self {
            scratch: Box::new(BlockScratch::new()),
            docs: Box::new([0; BLOCK_SIZE as usize]),
            freqs: Box::new([0; BLOCK_SIZE as usize]),
        }
    }
}

impl DocBlockDecoder {
    /// Appends the `(doc, freq)` pairs of a block
    /// `crate::postings_writer::encode_doc_block` wrote from `prev_doc_id`
    /// onto `out`, those from document `from` on; a frequency of 1 for each
    /// without `index_has_freq`.
    pub fn decode(
        &mut self,
        bytes: &[u8],
        prev_doc_id: i32,
        index_has_freq: bool,
        from: i32,
        out: &mut Vec<(i32, i32)>,
    ) -> Result<()> {
        let mut r = SliceInput::new(bytes);
        decode_full_block_body(
            &mut r,
            prev_doc_id,
            index_has_freq,
            index_has_freq,
            &mut self.scratch,
            &mut self.docs,
            &mut self.freqs,
            false,
            false,
        )?;
        let first = self.docs.partition_point(|&d| d < from);
        out.extend(
            self.docs[first..]
                .iter()
                .copied()
                .zip(self.freqs[first..].iter().copied()),
        );
        Ok(())
    }
}

/// What [`decode_full_block_body`] left behind for the block's doc ids.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BodyShape {
    /// All `BLOCK_SIZE` doc ids are in `docs`.
    Docs,
    /// A bit-set block kept as its words, `scratch.bitset[..words]`: the
    /// block's documents are `base + i` for each set bit `i`, and the `k`-th
    /// set bit is the document `freqs[k]` belongs to.
    Bits { base: i32, words: usize },
}

/// Decodes a full block's body (the `bitsPerValue` token onward) — `r` must
/// already be positioned at [`FullBlockHeader::body_start`]. Shared by
/// [`DocInput::read_postings`] (eager path) and [`LazyDocsCursor`] (lazy path) so
/// there is exactly one body decoder to keep in sync with `ForUtil`/
/// `PForUtil`.
///
/// With `keep_bitset`, a bit-set-encoded block is **not** expanded into
/// `docs`: its words are left in `scratch.bitset` and the block comes back as
/// [`BodyShape::Bits`], for a cursor that answers `nextDoc`/`advance` from
/// the bit set the way Lucene's `BlockPostingsEnum` does (`nextSetBit`,
/// never a materialized array). Freqs are decoded either way.
#[allow(clippy::too_many_arguments)]
fn decode_full_block_body(
    r: &mut SliceInput,
    prev_doc_id: i32,
    index_has_freq: bool,
    needs_freq: bool,
    scratch: &mut BlockScratch,
    docs: &mut [i32; BLOCK_SIZE as usize],
    freqs: &mut [i32; BLOCK_SIZE as usize],
    keep_bitset: bool,
    unit_freqs: bool,
) -> Result<BodyShape> {
    let mut shape = BodyShape::Docs;
    #[cfg(any(test, feature = "test-support"))]
    test_only_block_decode_counter::record_decode();
    let bits_per_value_byte = r.read_byte()? as i8;
    if bits_per_value_byte > 0 {
        let doc_deltas = &mut scratch.words;
        scratch
            .for_util
            .decode(bits_per_value_byte as u32, r, doc_deltas)?;
        // Deltas to doc ids: a running sum over the block, in wrapping
        // 32-bit arithmetic (equal bit for bit to `sum as i32` of the wide
        // sum), vectorized -- LLVM will not vectorize a scan on its
        // own, and it is what Lucene's `ForDeltaUtil` decode ends in. A
        // corrupt delta wraps rather than panics; `advance`'s post-refill
        // check against the block header catches the result.
        lucene_util::simd::prefix_sum_256(prev_doc_id, doc_deltas, docs);
    } else if bits_per_value_byte == 0 && keep_bitset {
        // "0 is used to record that all 256 docs in the block are
        // consecutive" (`Lucene104PostingsReader.refillFullBlock`), which
        // Lucene keeps as a bit set with the first 256 bits on
        // (`docBitSet.set(0, BLOCK_SIZE)`) and answers `advance` from with
        // `nextSetBit`. The same here: a term in nearly every document is
        // almost all such blocks, and expanding each into an array to scan is
        // measurably slower.
        const WORDS: usize = BLOCK_SIZE as usize / 64;
        scratch.bitset[..WORDS].fill(u64::MAX);
        shape = BodyShape::Bits {
            base: prev_doc_id.wrapping_add(1),
            words: WORDS,
        };
    } else if bits_per_value_byte == 0 {
        // As above, expanded: every delta is 1, no bytes follow.
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
        // One bounds check for the whole run (`readLongs`), rather than a
        // checked `read_i64` per word.
        // ARITH: `num_longs <= 128`, so the byte count is at most 1024.
        #[allow(clippy::arithmetic_side_effects)]
        let byte_len = num_longs * 8;
        let bytes = r.as_slice().get(..byte_len).ok_or_else(|| {
            Error::Store(lucene_store::Error::Eof {
                offset: r.position(),
            })
        })?;
        for (w, b) in words.iter_mut().zip(bytes.chunks_exact(8)) {
            *w = u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]);
        }
        r.skip(byte_len)?;
        // Doc ids are `docBitSetBase + bit`, with `docBitSetBase = prevDocID +
        // 1`, in the same wrapping arithmetic the rest of this decoder uses
        // for values derived from the wire.
        let base = prev_doc_id.wrapping_add(1);
        // A well-formed block sets exactly `BLOCK_SIZE` bits (Lucene asserts
        // it). Anything else is rejected: fewer leaves the block short, and
        // more would, in a kept bit set, make a docs-only `advance` return the
        // extra documents while a freqs cursor fails its rank check -- three
        // answers for one corrupt block.
        // A plain sum, not a saturating one: at most 128 words of 64 bits is
        // 8192, and without the per-word overflow branch the popcounts
        // pipeline instead of forming a chain (the saturating fold was a
        // third of a bit-set refill).
        let set: u32 = words.iter().map(|w| w.count_ones()).sum();
        if set != BLOCK_SIZE as u32 {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "full-block doc bit-set does not have exactly BLOCK_SIZE set bits".into(),
            )));
        }
        if keep_bitset {
            shape = BodyShape::Bits {
                base,
                words: num_longs,
            };
        } else {
            lucene_util::simd::expand_bitset(words, base, docs);
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
        // `PostingsEnum.NONE`/`DOCS` contract. A cursor that never decodes
        // frequencies keeps its array at 1 from the start (`unit_freqs`), and
        // rewriting 256 ones per block would be the one store loop left in a
        // docs-only advance.
        if !unit_freqs {
            freqs.fill(1);
        }
    }

    Ok(shape)
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
/// over one of those (the common shape by a wide margin) needs no `Vec`
/// allocations or 256-entry copies to hand the result back.
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
    // Straight into the doc buffer, as Lucene reads group-varints into its
    // long-lived `docBuffer`: this is the only decode a `docFreq <
    // BLOCK_SIZE` term ever does, so a staging buffer zeroed here is zeroed
    // once per *term*.
    r.read_group_vints_i32(docs)?;

    if index_has_freq && needs_freq {
        for (d, f) in docs.iter_mut().zip(freqs.iter_mut()) {
            let v = *d as u32;
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
        for d in docs.iter_mut() {
            *d = ((*d as u32) >> 1) as i32;
        }
        freqs.fill(1);
    } else {
        // The deltas as written. A field without freqs scores every
        // occurrence as 1, same as `decode_full_block_body`'s own no-freq
        // branch.
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
    //
    // SENTINEL: `-1` = "not started" and `NO_MORE_DOCS` = "exhausted", both
    // outside the doc-id domain -- Java's `DocIdSetIterator` contract.
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

/// ORs bits `lo..hi` of `src` into `dst`, bit `i` landing on bit `i + off`;
/// bits that would land outside `dst` are dropped.
fn or_shifted_bits(dst: &mut [u64], src: &[u64], lo: usize, hi: usize, off: i64) {
    let hi = hi.min(src.len().saturating_mul(64));
    if lo >= hi {
        return;
    }
    // ARITH: `lo < hi <= 64 * src.len()`, so the word indexes are in `src`;
    // the shifts are by `0..64`; `pos` is a bit offset well inside `i64`.
    #[allow(clippy::arithmetic_side_effects)]
    for (w, &word) in src
        .iter()
        .enumerate()
        .take(((hi - 1) >> 6) + 1)
        .skip(lo >> 6)
    {
        let mut x = word;
        if w == lo >> 6 {
            x &= u64::MAX << (lo & 63);
        }
        if w == (hi - 1) >> 6 && hi & 63 != 0 {
            x &= u64::MAX >> (64 - (hi & 63));
        }
        if x == 0 {
            continue;
        }
        let pos = off + (w as i64) * 64;
        let (q, r) = (pos.div_euclid(64), pos.rem_euclid(64) as u32);
        if let Some(d) = usize::try_from(q).ok().and_then(|q| dst.get_mut(q)) {
            *d |= x << r;
        }
        if r != 0 {
            if let Some(d) = usize::try_from(q + 1).ok().and_then(|q| dst.get_mut(q)) {
                *d |= x >> (64 - r);
            }
        }
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
    /// `needsImpacts`: whether a caller will ever ask for this cursor's
    /// impacts. A docs-only cursor never does, so every block header's impact
    /// run is stepped over unread, as `skipLevel0To` steps over it.
    needs_impacts: bool,
    /// `needsPos`: whether the `.pos`/`.pay` origin of each block is tracked
    /// ([`Self::position_origin`], [`PositionsCursor`]). Off unless a positions
    /// reader opts in with [`Self::track_positions`]; when off, the pos/pay
    /// skip fields of every block header are stepped over unread.
    needs_pos: bool,
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
    /// hold, i.e. the block containing `doc_id`. Decoded from
    /// `level0_impact_bytes` on first use ([`Self::level0_impacts`]).
    level0_impacts: Impacts,
    /// The current block's serialized impacts, not yet decoded into
    /// `level0_impacts` while `level0_impacts_stale`: `skipLevel0To` keeps
    /// the bytes and `getImpacts` decodes them, so a cursor advanced through
    /// many blocks -- the dense clause of a conjunction, landing on a new
    /// block every hundred candidates -- decodes only the blocks whose bound
    /// is asked for.
    level0_impact_bytes: &'a [u8],
    level0_impacts_stale: bool,
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
    /// Bumped every time a block body (full or tail) is decoded into
    /// `block_docs`/`block_freqs`, so a [`PositionsCursor`] riding on this
    /// cursor can tell that the `.doc` block under it changed.
    block_gen: u64,
    /// `Some((base, words))` while the current block is a bit-set block kept
    /// unexpanded ([`BodyShape::Bits`]): its documents are `base + i` for the
    /// set bits of `scratch.bitset[..words]`, `block_docs` is not filled, and
    /// `block_pos` is the current document's rank among them -- which is
    /// what keeps `block_freqs[block_pos]` its frequency.
    bits: Option<(i32, usize)>,
    /// A docs-only `next_doc` has been answered from `bits` since the last
    /// `advance`: the next one expands the block instead. See
    /// [`Self::next_doc`].
    bits_stepped: bool,
    /// The generation that wrote the term. `Lucene912`/`Lucene101`/`Lucene103`
    /// share `Lucene104`'s `.doc` layout -- the same level-0 headers, level-1
    /// entries and group-varint tail -- at half the block size, and differ only
    /// in how a block body is packed (`refill`).
    format: PostingsFormat,
    /// `ForUtil.BLOCK_SIZE` of [`Self::format`]: 256, or 128 for a retired
    /// generation. The block arrays are always 256 wide; a 128-document
    /// block fills their front half.
    bs: i32,
    /// `LEVEL1_NUM_DOCS` of [`Self::format`]: 32 blocks.
    level1_docs: i32,
    /// The trailing multi-level skip list of a `Lucene90`/`Lucene99` term
    /// with more than one block: where the level-0 headers of the inline
    /// generations come from for these ([`Self::advance_shallow`]). `None`
    /// for every other term.
    skip: Option<Box<crate::backward_codecs::skip_list::SkipList<'a>>>,
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
    // Eight lanes at a time on AVX2 builds, as Lucene's Panama
    // `VectorUtilSupport.findNextGEQ` does; the scalar scan elsewhere.
    lucene_util::simd::find_next_geq(buf, target)
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

    /// Opts this cursor into tracking each block's `.pos`/`.pay` origin
    /// (`needsPos`), which [`Self::position_origin`] and [`PositionsCursor`]
    /// read. Must be called before the cursor first moves; a cursor that never
    /// reads positions leaves it off and steps over that part of every block
    /// header.
    ///
    /// Crate-private because the "before the first move" rule is only a
    /// `debug_assert`: turned on later, the blocks already crossed never added
    /// their skip deltas and the origins would be silently wrong. Its two
    /// callers set it immediately after construction.
    pub(crate) fn track_positions(&mut self) {
        debug_assert_eq!(self.doc_id, -1, "track_positions after the cursor moved");
        self.needs_pos = true;
    }

    /// Whether the field was indexed with frequencies (`indexHasFreq`):
    /// without them every document's frequency is 1 and no block carries
    /// impacts.
    pub fn has_freqs(&self) -> bool {
        self.index_has_freq
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

    /// [`Self::level0_impacts`] without decoding: a caller that only folds
    /// the pairs (a block-max bound) walks the bytes once with
    /// [`for_each_impact`] and leaves the list undecoded for anyone else.
    pub fn level0_impacts_view(&self) -> Level0Impacts<'_> {
        if self.level0_impacts_stale {
            Level0Impacts::Encoded(self.level0_impact_bytes)
        } else {
            Level0Impacts::Decoded(&self.level0_impacts)
        }
    }

    /// `ImpactsEnum.getImpacts()`'s level-0 result, conceptually: the
    /// competitive `(freq, norm)` pairs for the level-0 block the cursor is
    /// currently positioned in (i.e. covering `doc_id`). Empty before the
    /// first `next_doc()`/`advance()` call, once exhausted, for the trailing
    /// tail block (no level-0 impacts exist on the wire for it), or for a
    /// field with no freqs.
    ///
    /// Decoded here, on first use after the block changed. Bytes that do not
    /// decode (a truncated varint in a corrupt `.doc`) give no impacts: "no
    /// bound", on which every caller declines to skip -- the conservative
    /// answer, never a wrong skip.
    pub fn level0_impacts(&mut self) -> &[Impact] {
        if self.level0_impacts_stale {
            self.level0_impacts_stale = false;
            if decode_impacts_into(self.level0_impact_bytes, &mut self.level0_impacts).is_err() {
                self.level0_impacts.clear();
                self.level0_impacts_stale = false;
            }
        }
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

    /// `PostingsEnum.docIDRunEnd()`: one past the last document of the run
    /// of consecutive documents the current one starts. Every document in
    /// `[doc_id(), doc_id_run_end())` is in this term's postings.
    ///
    /// A fully dense block (a bit-set block whose four words are all ones:
    /// 256 consecutive documents) runs to the end of the block, and to the end
    /// of its level-1 span when that span holds as many documents as it spans
    /// -- `BlockPostingsEnum`'s override, exactly.
    ///
    /// Anywhere else Lucene answers `DocIdSetIterator`'s conservative
    /// `docID() + 1`. This answers the true run within the decoded block: the
    /// next clear bit of a bit-set block, or the first gap in the decoded doc
    /// ids. The contract allows any end whose documents all match, and a
    /// `ReqExclBulkScorer` over a nearly dense excluded term then jumps each
    /// run once instead of stepping through it a document at a time.
    pub fn doc_id_run_end(&self) -> i32 {
        // Only a `Lucene104` block is ever kept as bits (`refill`).
        const DENSE_WORDS: usize = (BLOCK_SIZE as usize) / 64;
        let doc = self.doc_id;
        if doc < 0 || doc == NO_MORE_DOCS || self.pending.is_some() {
            return doc.saturating_add(1);
        }
        if let Some((base, words)) = self.bits {
            if words == DENSE_WORDS
                && self.scratch.bitset[..DENSE_WORDS]
                    .iter()
                    .all(|&w| w == u64::MAX)
            {
                // ARITH: `doc_count_left <= doc_freq`, both non-negative;
                // the doc ids are below `NO_MORE_DOCS`, so the spans and
                // the `+ 1` cannot overflow except at `NO_MORE_DOCS`,
                // which a dense block never ends on.
                let level0_doc_count_upto = self.doc_freq.wrapping_sub(self.doc_count_left);
                let level1_is_dense = self
                    .level1_last_doc_id
                    .wrapping_sub(self.level0_last_doc_id)
                    == self
                        .level1_doc_count_upto
                        .wrapping_sub(level0_doc_count_upto);
                if level1_is_dense {
                    return self.level1_last_doc_id.saturating_add(1);
                }
                return self.level0_last_doc_id.saturating_add(1);
            }
            if doc < base {
                return doc.saturating_add(1);
            }
            // The next clear bit after this document's, capped at the block's
            // last document.
            let rel = doc.wrapping_sub(base) as u32 as usize;
            let clear = lucene_util::fixed_bit_set::next_clear_bit_in_words(
                &self.scratch.bitset[..words],
                rel,
            );
            let end = base.saturating_add(i32::try_from(clear).unwrap_or(i32::MAX));
            return end
                .min(self.prev_doc_id.saturating_add(1))
                .max(doc.saturating_add(1));
        }
        // A decoded block: `block_docs[block_pos] == doc_id`.
        let (mut j, len) = (self.block_pos, self.block_len);
        if j >= len || self.block_docs[j] != doc {
            return doc.saturating_add(1);
        }
        // ARITH: `j + 1 < len <= BLOCK_SIZE`; doc ids ascend below
        // `NO_MORE_DOCS`, so the `+ 1`s cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        while j + 1 < len && self.block_docs[j + 1] == self.block_docs[j] + 1 {
            j += 1;
        }
        self.block_docs[j].saturating_add(1)
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
    ///
    /// Crate-private with [`Self::track_positions`], which it depends on:
    /// outside the crate no cursor could have opted in, so it could only
    /// ever answer `Err`.
    pub(crate) fn position_origin(&self) -> Result<Option<PositionOrigin>> {
        if self.doc_id == -1 || self.doc_id == NO_MORE_DOCS {
            return Ok(None);
        }
        if !self.needs_freq {
            return Err(Error::Unsupported(
                "position_origin needs a cursor opened with PostingsFlags::Freqs: the \
                 in-block frequency sum it adds to the skip data was never decoded",
            ));
        }
        if !self.needs_pos {
            return Err(Error::Unsupported(
                "position_origin needs a cursor that tracks positions (track_positions): \
                 without it the .pos/.pay origin of each block is never read",
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
    /// Delegating to `advance(doc_id + 1)` would be correct but pays
    /// `advance`'s `partition_point` -- a binary search over up to 256 entries,
    /// eight unpredictable branches, to find an offset that is always exactly 1.
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
        // A sequential walk reads the rest of the block document by document,
        // so a kept bit set is expanded once here (a set-bit search per call
        // measured 20% slower than the array on `t1`/`tz`; answering only the
        // first sixteen calls from the bits still cost sequential walks 5-9%
        // and bought an early-stopping union almost nothing); expanding also
        // settles `block_pos`, which a docs-only `advance` leaves stale.
        //
        // The exception is the first `next_doc` after an `advance` on a
        // docs-only cursor: that is `ReqExclBulkScorer` stepping past one
        // excluded document, which then advances again -- usually into another
        // block, so expanding this one would be wasted. It is answered from
        // the bits, `BlockPostingsEnum.nextDoc`'s `nextSetBit`, and only a
        // second consecutive call expands.
        if let Some((base, n)) = self.bits {
            if self.pending.is_none() {
                if !self.needs_freq && !self.bits_stepped && self.doc_id >= base {
                    self.bits_stepped = true;
                    // ARITH: `doc_id >= base`, and both lie in this block.
                    let from = (self.doc_id.wrapping_sub(base) as u32 as usize).saturating_add(1);
                    let words = &self.scratch.bitset[..n];
                    if let Some(bit) =
                        lucene_util::fixed_bit_set::next_set_bit_in_words(words, from)
                    {
                        let doc = base.wrapping_add(bit as i32);
                        if doc <= self.prev_doc_id {
                            self.doc_id = doc;
                            return Ok(doc);
                        }
                    }
                    // The block has nothing after this document: the next one
                    // is in a later block, so expanding this one is wasted.
                    return self.advance(self.doc_id.saturating_add(1));
                }
                self.materialize_bits();
            }
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

    /// `PostingsEnum.nextPostings(upTo, buffer)` (`BlockPostingsEnum`'s
    /// override): the current document and every later one in the **same
    /// decoded block** that is below `up_to`, with their frequencies, copied
    /// into `docs`/`freqs` (both cleared first). The cursor is left on the
    /// first document *not* returned -- the next one in the block, or the
    /// first one `>= up_to`, decoding the next block if it has to -- exactly
    /// as Java's trailing `advance(upTo)` leaves it.
    ///
    /// Empty output means the cursor is at or past `up_to` (or exhausted).
    ///
    /// This is what lets a scorer work a block at a time: one bulk copy of the
    /// already-decoded block instead of a `next_doc()` + `freq()` round trip
    /// per document, each returning a `Result`.
    ///
    /// Must be called with the cursor positioned on a document (after
    /// `next_doc`/`advance`), as Java asserts `needsRefilling == false`. A
    /// shallow move that has not been materialized is resolved by decoding the
    /// pending block and starting at its first document, since every document
    /// left in the previous block is behind the shallow target.
    pub fn next_postings(
        &mut self,
        up_to: i32,
        docs: &mut Vec<i32>,
        freqs: &mut Vec<i32>,
    ) -> Result<()> {
        self.next_postings_at_most(up_to, usize::MAX, docs, freqs)
    }

    /// [`Self::next_postings`], returning at most `max` documents (`max >=
    /// 1`): a caller that must stop after a given number of matches -- a
    /// conjunction filling `Scorer.nextDocsAndScores`' batch of 64 -- takes
    /// no document from the block it cannot use.
    pub fn next_postings_at_most(
        &mut self,
        up_to: i32,
        max: usize,
        docs: &mut Vec<i32>,
        freqs: &mut Vec<i32>,
    ) -> Result<()> {
        docs.clear();
        freqs.clear();
        if self.pending.is_some() {
            self.refill()?;
            self.materialize_bits();
            self.doc_id = self.block_docs[0];
        }
        // The rest of this block is copied out as an array.
        self.materialize_bits();
        if self.doc_id < 0 || self.doc_id >= up_to {
            return Ok(());
        }
        let start = self.block_pos;
        let len = self.block_len;
        debug_assert!(start < len && self.block_docs[start] == self.doc_id);
        let block = &self.block_docs[start..len];
        // `computeBufferEndBoundary`: the whole rest of the block when its
        // last document is under `up_to`, else the first one that is not.
        let block_last = block.last().copied().unwrap_or(NO_MORE_DOCS);
        let n = if block_last < up_to {
            block.len()
        } else {
            find_next_geq(block, up_to)
        }
        .min(max.max(1));
        docs.extend_from_slice(&block[..n]);
        freqs.extend_from_slice(&self.block_freqs[start..len][..n]);
        // ARITH: `n <= len - start`, so `start + n <= len <= BLOCK_SIZE`.
        #[allow(clippy::arithmetic_side_effects)]
        let next = start + n;
        if next < len {
            self.block_pos = next;
            self.doc_id = self.block_docs[next];
            return Ok(());
        }
        // The block is used up: move on exactly as `advance` past its last
        // document would, which decodes the next block (if any).
        self.block_pos = len;
        self.advance(block_last.saturating_add(1))?;
        Ok(())
    }

    /// `PostingsEnum.advance(target)`: moves forward to the first doc ID
    /// `>= target`, returning it (or [`NO_MORE_DOCS`] if none remains).
    /// Advancing to a target at or before the current doc ID is a documented
    /// no-op (same contract as [`PostingsCursor::advance`]).
    ///
    /// The common case -- `target` inside the block already decoded as an
    /// array -- is this small inlined function: one test against the block's
    /// last document, then [`find_next_geq`]. Everything else (a bit-set
    /// block, crossing into later blocks, the end) is [`Self::advance_slow`].
    #[inline]
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        self.bits_stepped = false;
        let (pos, len) = (self.block_pos, self.block_len);
        // A cursor on a kept bit set, with the target inside the block the
        // header describes: Lucene's `docBitSet.nextSetBit(target -
        // docBitSetBase)`, inline -- plus, when frequencies are read, the
        // document's rank among the block's set bits.
        if let Some((base, n)) = self.bits {
            if target > self.doc_id
                && target <= self.prev_doc_id
                && self.pending.is_none()
                && target > base
            {
                let rel = target.wrapping_sub(base) as u32 as usize;
                let words = &self.scratch.bitset[..n];
                if let Some(bit) = lucene_util::fixed_bit_set::next_set_bit_in_words(words, rel) {
                    let doc = base.wrapping_add(bit as i32);
                    if doc <= self.prev_doc_id {
                        if self.needs_freq {
                            // The rank keeps `block_freqs[block_pos]` this
                            // document's frequency: `bits_advance`'s
                            // cumulative popcount, inline.
                            // ARITH: `bit < 128 * 64`, so `bit >> 6 < 128`
                            // indexes both arrays; the shift is at most 63;
                            // the sum is at most 8192.
                            #[allow(clippy::arithmetic_side_effects)]
                            let rank = {
                                let w = bit >> 6;
                                let below = (1u64 << (bit & 63)) - 1;
                                usize::from(self.scratch.word_ranks[w])
                                    + (words[w] & below).count_ones() as usize
                            };
                            if rank >= self.block_len {
                                return self.advance_slow(target);
                            }
                            self.block_pos = rank;
                        }
                        self.doc_id = doc;
                        return Ok(doc);
                    }
                }
                // No such bit, or one past the header's last document:
                // corrupt, and `advance_slow` reports it the same way.
            }
            return self.advance_slow(target);
        }
        // `target > doc_id` also excludes an exhausted cursor (`doc_id ==
        // NO_MORE_DOCS == i32::MAX`); `pos < len` makes `len - 1` a real slot.
        if target > self.doc_id
            && self.pending.is_none()
            && self.bits.is_none()
            && pos < len
            && target <= self.block_docs[len.wrapping_sub(1)]
        {
            // The block's last document is `>= target`, so the search lands
            // inside the block.
            // Lucene's own step (`findNextGEQ(docBuffer, target,
            // docBufferUpto, ...)`): a predictable branch on the entry eight
            // ahead, then one count. A vector compare at every step made each
            // landing wait on a mask. The search starts at `pos` itself, not
            // one past it: before the first move that entry has not been
            // returned yet.
            // ARITH: the search returns at most the length it is given, and
            // less here since the last document qualifies, so `landing < len`.
            #[allow(clippy::arithmetic_side_effects)]
            let landing =
                pos + lucene_util::simd::find_next_geq_v1(&self.block_docs[pos..len], target);
            // The count is the answer only on an ascending block; a corrupt
            // one that is not falls through to the exact scan.
            let doc = self.block_docs[landing];
            if doc >= target {
                self.block_pos = landing;
                self.doc_id = doc;
                return Ok(doc);
            }
        }
        self.advance_slow(target)
    }

    /// Whether the postings hold exactly `target`, answered from the current
    /// bit-set block when it covers `target`: `Some(true)` positions the
    /// cursor there (as [`Self::advance`] would), `Some(false)` leaves it
    /// where it was -- no scan to the next set bit, which a caller only
    /// testing membership (`ScorerUtil.applyRequiredClause`) never needs.
    /// `None` when the block does not answer (an array block, a target past
    /// it, a pending block): use [`Self::advance`].
    #[inline]
    pub fn advance_exact_in_bits(&mut self, target: i32) -> Option<bool> {
        let (base, n) = self.bits?;
        if !(target > self.doc_id
            && target <= self.prev_doc_id
            && self.pending.is_none()
            && target > base)
        {
            return None;
        }
        let rel = target.wrapping_sub(base) as u32 as usize;
        let words = self.scratch.bitset.get(..n)?;
        // ARITH: shifting right by 6 and masking with 63 cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        let (w, b) = (rel >> 6, rel & 63);
        let word = *words.get(w)?;
        if (word >> b) & 1 == 0 {
            return Some(false);
        }
        if self.needs_freq {
            // As `advance`: the rank keeps `block_freqs[block_pos]` this
            // document's frequency.
            // ARITH: `w < n <= 128`, the shift is at most 63, the sum at
            // most 8192.
            #[allow(clippy::arithmetic_side_effects)]
            let rank = usize::from(*self.scratch.word_ranks.get(w)?)
                + (word & ((1u64 << b) - 1)).count_ones() as usize;
            if rank >= self.block_len {
                return None;
            }
            self.block_pos = rank;
        }
        self.bits_stepped = false;
        self.doc_id = target;
        Some(true)
    }

    /// Marks every document from the current one up to (not including) `end`
    /// in `words`, a bit set whose bit `i` is document `base + i`, and leaves
    /// the cursor on the first document `>= end` (returned): a window of
    /// `BooleanScorer`'s bulk union. A kept bit-set block is ORed in a word at
    /// a time rather than expanded document by document -- which is how
    /// Lucene's own `DocIdSetIterator.intoBitSet` override for
    /// `BlockPostingsEnum` fills a window from a dense block.
    ///
    /// The caller guarantees `base <= doc_id` and `end - base <= 64 *
    /// words.len()`; a document outside the window (only a corrupt block has
    /// one) is dropped rather than written out of bounds.
    pub fn into_window(&mut self, base: i32, end: i32, words: &mut [u64]) -> Result<i32> {
        let mut doc = self.doc_id;
        while doc < end {
            if let (Some((bbase, n)), None, false) = (self.bits, self.pending, self.needs_freq) {
                if doc >= bbase {
                    let last = self.prev_doc_id;
                    // ARITH: `bbase <= doc <= last`, all doc ids of one block,
                    // and `end - bbase` is only used when `end <= last`.
                    let lo = doc.wrapping_sub(bbase) as u32 as usize;
                    let hi = if last < end {
                        last
                    } else {
                        end.wrapping_sub(1)
                    };
                    let hi = (hi.wrapping_sub(bbase) as u32 as usize).saturating_add(1);
                    let src = self.scratch.bitset.get(..n).unwrap_or(&[]);
                    // ARITH: two `i32`s widened to `i64` cannot overflow a subtraction.
                    #[allow(clippy::arithmetic_side_effects)]
                    let off = i64::from(bbase) - i64::from(base);
                    or_shifted_bits(words, src, lo, hi, off);
                    if last < end {
                        doc = self.advance(last.saturating_add(1))?;
                        continue;
                    }
                    // `last >= end` is a set bit of this block, so there is one.
                    let from = (end.wrapping_sub(bbase) as u32 as usize).max(lo);
                    let bit = lucene_util::fixed_bit_set::next_set_bit_in_words(src, from)
                        .ok_or_else(|| corrupted("bit-set block ends before its last doc ID"))?;
                    self.bits_stepped = false;
                    self.doc_id = bbase.wrapping_add(bit as i32);
                    return Ok(self.doc_id);
                }
            }
            // An array block already decoded: the rest of it below `end` in
            // one pass over the buffer, as `intoBitSet` loops over its
            // `docBuffer`, rather than a `next_doc` round trip per document.
            if self.pending.is_none()
                && self.bits.is_none()
                && self.block_pos < self.block_len
                && doc >= base
            {
                let start = self.block_pos;
                let block = &self.block_docs[start..self.block_len];
                let mut k = 0usize;
                if block.last().is_some_and(|&last| last < end) {
                    // The whole rest of the block is in the window: no
                    // per-document bound test.
                    for &d in block {
                        // ARITH: `base <= doc <= d < end`, as below.
                        let i = d.wrapping_sub(base) as u32 as usize;
                        if let Some(w) = words.get_mut(i >> 6) {
                            *w |= 1u64 << (i & 63);
                        }
                    }
                    k = block.len();
                }
                while k < block.len() && block[k] < end {
                    // ARITH: `base <= doc <= block[k] < end`, so the offset is
                    // non-negative and below the window's length.
                    let i = block[k].wrapping_sub(base) as u32 as usize;
                    if let Some(w) = words.get_mut(i >> 6) {
                        *w |= 1u64 << (i & 63);
                    }
                    k = k.saturating_add(1);
                }
                if let Some(&next) = block.get(k) {
                    self.block_pos = start.saturating_add(k);
                    self.doc_id = next;
                    return Ok(next);
                }
                // The whole block is in the window: stand on its last document
                // (the `block_docs[block_pos] == doc_id` invariant) and step
                // into the next block.
                self.block_pos = self.block_len.saturating_sub(1);
                self.doc_id = self.block_docs[self.block_pos];
                doc = self.next_doc()?;
                continue;
            }
            if doc >= base {
                // ARITH: `base <= doc < end`, so the offset is non-negative and
                // below the window's length.
                let i = doc.wrapping_sub(base) as u32 as usize;
                if let Some(w) = words.get_mut(i >> 6) {
                    *w |= 1u64 << (i & 63);
                }
            }
            doc = self.next_doc()?;
        }
        Ok(doc)
    }

    /// Positions past the last document, as walking to the end would.
    fn exhaust(&mut self) {
        self.pending = None;
        self.bits = None;
        self.block_len = 0;
        self.block_pos = 0;
        self.doc_count_left = 0;
        self.level0_last_doc_id = NO_MORE_DOCS;
        self.level1_last_doc_id = NO_MORE_DOCS;
        self.level0_impacts.clear();
        self.level0_impacts_stale = false;
        self.level1_impacts.clear();
        self.doc_id = NO_MORE_DOCS;
    }

    /// [`Self::advance`] for everything but a target inside the current
    /// array block.
    fn advance_slow(&mut self, target: i32) -> Result<i32> {
        if self.doc_id == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        if target <= self.doc_id {
            return Ok(self.doc_id);
        }
        if target == NO_MORE_DOCS {
            // No document can be at or past `NO_MORE_DOCS`, so the walk over
            // every remaining skip entry that would prove it is skipped (a
            // deliberate difference from `BlockPostingsEnum.advance`, which
            // walks them; skip data past the current position is then never
            // read, so a corrupt entry there is not reported here): a
            // scorer that knows nothing more can compete (a constant score
            // below the bar) exhausts its postings this way.
            self.exhaust();
            return Ok(NO_MORE_DOCS);
        }

        // First, try the already-decoded current block (covers the common
        // "advance a little" and "nextDoc" cases without touching the wire
        // at all).
        // `pending.is_none()` matters: a shallow move positions past the
        // decoded block without touching `block_docs`, so those documents are
        // stale and must not answer an advance.
        if self.pending.is_none() && self.block_pos < self.block_len && self.bits.is_some() {
            // A kept bit set: `nextSetBit`, if the block reaches `target`
            // (`prev_doc_id` is its header's last document).
            if target <= self.prev_doc_id {
                return self.bits_advance(target);
            }
            self.block_pos = self.block_len;
        }
        // Only when the block's last document reaches `target`: otherwise the
        // scan could only come up empty (Lucene's `target > level0LastDocID`
        // test goes straight to the skip data), and scanning the rest of a
        // block to learn that was half of what a long advance cost.
        if self.pending.is_none() && self.block_pos < self.block_len {
            if target <= self.block_docs[self.block_len.wrapping_sub(1)] {
                let offset =
                    find_next_geq(&self.block_docs[self.block_pos..self.block_len], target);
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
            // needed, so pay for it -- except that a bit-set block is kept as
            // its bits and searched, never expanded.
            self.refill()?;
            if self.bits.is_some() {
                return self.bits_advance(target);
            }
            // Counted rather than scanned: the landing is anywhere in a fresh
            // block, so an early-exit scan ends on a branch it cannot predict
            // (`lucene_util::simd::count_less_than`).
            let offset = if self.block_len == BLOCK_SIZE as usize {
                lucene_util::simd::count_less_than(&self.block_docs, target)
            } else {
                // A retired generation's 128-document block: the back half
                // of the array is stale.
                find_next_geq(&self.block_docs[..self.block_len], target)
            };
            // `advance_shallow` only stops on a block whose header claims
            // `last_doc_id >= target`, so a well-formed block always has a
            // match. A corrupt `.doc` can claim one and then decode a body
            // whose last doc is smaller -- nothing on the wire ties the
            // level-0 header's `docDelta` to the body's own deltas -- and
            // indexing at `BLOCK_SIZE` would panic instead of surfacing that.
            // A body that does not ascend can make the count land short of
            // `target`, which is the same corruption seen from the other side.
            if offset >= self.block_len && self.level0_last_doc_id == NO_MORE_DOCS {
                // A block of unknown extent (see `trailing_advance_shallow`)
                // that ends before `target`: it was the term's last.
                self.block_pos = self.block_len;
                self.doc_id = NO_MORE_DOCS;
                return Ok(NO_MORE_DOCS);
            }
            if offset >= self.block_len || self.block_docs[offset] < target {
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
        if self.doc_count_left < 0 || self.doc_count_left >= self.bs {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "tail block claims {} remaining documents, which is not in 0..{}",
                self.doc_count_left, self.bs
            ))));
        }
        let count = self.doc_count_left as usize;
        if self.format == PostingsFormat::Lucene90 {
            crate::backward_codecs::postings::read_tail_block_90(
                &mut self.r,
                self.prev_doc_id,
                self.index_has_freq,
                &mut self.block_docs[..count],
                &mut self.block_freqs[..count],
            )?;
        } else {
            read_tail_block(
                &mut self.r,
                self.prev_doc_id,
                self.index_has_freq,
                self.needs_freq,
                &mut self.block_docs[..count],
                &mut self.block_freqs[..count],
            )?;
        }
        self.block_len = count;
        self.bits = None;
        self.doc_count_left = 0;
        self.block_gen = self.block_gen.wrapping_add(1);
        // The tail block has no level-0 skip header (and hence no impacts) on
        // the wire at all (`Lucene104PostingsReader.refillRemainder`'s
        // non-singleton branch never touches `level0SerializedImpacts`).
        self.level0_impacts.clear();
        self.level0_impacts_stale = false;

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
    /// Without it, deciding "can this block hold a competitive document?"
    /// required [`Self::advance`], which decodes the block first -- so a
    /// scoring loop paid the `ForUtil`/`PForUtil` unpack of 256 documents for
    /// every block it then discarded. Counted on the benchmark corpus, `and t0 tz`
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
                debug_assert!(self.doc_count_left >= self.bs);
                self.doc_count_left -= self.bs;
            }
        }

        loop {
            // Level-1 skip: jump past whole 32-block spans that are entirely
            // behind `target` before looking at any level-0 header, exactly as
            // `doAdvanceShallow` does.
            // A trailing-skip term has no inline level-1 entries: its
            // `level1_*` describe skip-list level 1 (`trailing_advance_shallow`).
            if target > self.level1_last_doc_id && self.skip.is_none() {
                self.skip_level1_to(target)?;
            }

            if self.doc_count_left == 0 {
                self.level0_impacts.clear();
                self.level0_impacts_stale = false;
                self.level0_last_doc_id = NO_MORE_DOCS;
                return Ok(NO_MORE_DOCS);
            }

            // Sampled here, at the top of the iteration, exactly as
            // `skipLevel0To` samples its `posFP`/`posUpto`/`payFP` locals:
            // this is the `.pos`/`.pay` state at the start of whichever block
            // the loop is about to look at.
            let origin = self.level0_pos;

            if matches!(
                self.format,
                PostingsFormat::Lucene90 | PostingsFormat::Lucene99
            ) {
                return self.trailing_advance_shallow(target);
            }

            if self.doc_count_left >= self.bs && !self.needs_pos {
                if let Some(last) = self.skip_level0_headers(target)? {
                    return Ok(last);
                }
                continue;
            }

            if self.doc_count_left >= self.bs {
                let header = read_full_block_header(
                    &mut self.r,
                    self.prev_doc_id,
                    self.index_has_freq,
                    self.index_has_pos,
                    self.index_has_offsets_or_payloads,
                    self.needs_impacts,
                    self.needs_pos,
                )?;

                if let Some(skip) = header.pos_skip {
                    self.level0_pos.advance(skip);
                }

                if header.last_doc_id < target {
                    self.r.seek(header.body_end)?;
                    self.prev_doc_id = header.last_doc_id;
                    // ARITH: guarded by the enclosing
                    // `if self.doc_count_left >= self.bs`.
                    #[allow(clippy::arithmetic_side_effects)]
                    {
                        self.doc_count_left -= self.bs;
                    }
                    continue;
                }

                // See `skip_level0_headers`: the header after this block.
                self.r.prefetch(header.body_end);

                // `skipLevel0To`'s `posFP`/`posUpto`/`payFP` locals, handed
                // to `seekPosData` once the loop settles: the state sampled
                // *before* this block's own header advanced it, which is
                // where this block's documents' occurrences begin.
                self.block_pos_origin = origin;

                // Impacts, yes -- a handful of vints, and the whole reason a
                // caller is here. Body, no.
                self.level0_impact_bytes = header.impact_bytes;
                self.level0_impacts_stale = true;
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
            self.level0_impacts_stale = false;
            self.level0_last_doc_id = NO_MORE_DOCS;
            return Ok(NO_MORE_DOCS);
        }
    }

    /// [`Self::advance_shallow`]'s level-0 walk for a cursor that wants no
    /// `.pos`/`.pay` origins -- `skipLevel0To` with `needsPos == false`,
    /// which reads three fields per block and seeks, and the impacts of the
    /// block it stops on when they are wanted (`needsImpacts && found`),
    /// kept undecoded until asked for. The same parse as [`read_full_block_header`] on
    /// that branch, with the cursor's state held in locals across blocks
    /// rather than reloaded and stored back each time: this loop is most of
    /// what a long `advance` over a dense term costs.
    ///
    /// `Some(last_doc_id)` once positioned (shallowly) on the block holding
    /// `target`; `None` when fewer than `BLOCK_SIZE` documents remain (the
    /// tail is next) before one does,
    /// with the cursor's state written back for
    /// [`Self::advance_shallow`] to carry on from.
    fn skip_level0_headers(&mut self, target: i32) -> Result<Option<i32>> {
        let mut r = self.r.clone();
        let mut prev = self.prev_doc_id;
        let mut left = self.doc_count_left;
        let found = loop {
            let level0_num_bytes = r.read_vlong()?;
            let skip0_end = add_wire_offset(
                r.position(),
                wire_length(level0_num_bytes, "level-0 skip header")?,
            )?;
            // Wrapping for the reason `read_full_block_header` gives.
            let last = prev.wrapping_add(read_vint15(&mut r)?);
            let block_length = read_vlong15(&mut r)?;
            let body_end =
                add_wire_offset(r.position(), wire_length(block_length, "level-0 block")?)?;
            // `skipLevel0To` reads a block's impacts only when it stops on it
            // (`needsImpacts && found`) and seeks past everything else.
            let mut impacts: &[u8] = &[];
            if self.index_has_freq {
                if last >= target && self.needs_impacts {
                    let len = r.read_length("level-0 impacts")?;
                    let start = r.position();
                    let end = add_wire_offset(start, len)?;
                    // The impacts lie inside this header: a length reaching
                    // past it would read the block body as impacts, and a
                    // wrong bound skips blocks that compete.
                    if end > skip0_end {
                        return Err(corrupted("level-0 impacts run past their skip header"));
                    }
                    impacts = r.slice(start, end)?;
                }
                r.seek(skip0_end)?;
            }
            check_wire_position(r.position(), skip0_end, "level-0 skip header")?;
            if last >= target {
                // The next header, read by the next `advance` past this
                // block: its line loads while this body is decoded.
                r.prefetch(body_end);
                break Some((last, r.position(), body_end, impacts));
            }
            r.seek(body_end)?;
            prev = last;
            // ARITH: `advance_shallow` enters with `left >= BLOCK_SIZE`, and
            // the loop only goes round again while that still holds.
            #[allow(clippy::arithmetic_side_effects)]
            {
                left -= self.bs;
            }
            // No level-1 test: `advance_shallow` ran `skip_level1_to` first,
            // so the span in hand reaches `target` and the walk cannot run
            // into the next level-1 entry before finding it.
            debug_assert!(target <= self.level1_last_doc_id);
            if left < self.bs {
                break None;
            }
        };
        self.r = r;
        self.prev_doc_id = prev;
        self.doc_count_left = left;
        let Some((last, body_start, body_end, impacts)) = found else {
            return Ok(None);
        };
        // No `.pos`/`.pay` skip data was parsed, so the origin sampled before
        // the walk is still this block's.
        self.block_pos_origin = self.level0_pos;
        self.level0_impacts.clear();
        self.level0_impact_bytes = impacts;
        self.level0_impacts_stale = self.needs_impacts;
        self.level0_last_doc_id = last;
        self.pending = Some(PendingBlock {
            base_doc_id: prev,
            last_doc_id: last,
            body_start,
            body_end,
        });
        Ok(Some(last))
    }

    /// [`Self::advance_shallow`] for a `Lucene90`/`Lucene99` term: its block
    /// headers are the entries of the trailing skip list, so this is
    /// `BlockImpactsDocsEnum.advanceShallow` -- `skipTo(target)` when the
    /// target is past the block the skip list describes, jumping as many
    /// blocks (and levels) as it can, then the entry it stopped on gives the
    /// block's last document, its end and its impacts. The one full block
    /// with no entry, the last of a term that ends on a block boundary, is
    /// positioned with an unknown extent, like a tail.
    ///
    /// Called with documents left and no pending block.
    fn trailing_advance_shallow(&mut self, target: i32) -> Result<i32> {
        let Some(skip) = self.skip.as_mut() else {
            // `docFreq <= 128`: one block (or a tail alone) and no skip list,
            // so no extent, and its occurrences start where the term's do.
            self.block_pos_origin = self.level0_pos;
            self.level0_impacts.clear();
            self.level0_impacts_stale = false;
            self.level0_last_doc_id = NO_MORE_DOCS;
            if self.doc_count_left < self.bs {
                return Ok(NO_MORE_DOCS);
            }
            self.level0_impacts.clear();
            self.level0_impacts_stale = false;
            self.level0_last_doc_id = NO_MORE_DOCS;
            self.pending = Some(PendingBlock {
                base_doc_id: self.prev_doc_id,
                last_doc_id: NO_MORE_DOCS,
                body_start: self.r.position(),
                body_end: usize::MAX,
            });
            return Ok(NO_MORE_DOCS);
        };
        // SENTINEL-OK: an unpositioned list's `-1` is below every target
        // (`advance_shallow` takes only `target >= 0`), so it always skips --
        // `BlockImpactsDocsEnum.nextSkipDoc`'s own initial `-1`.
        if target > skip.next_skip_doc() {
            // `newDocUpto = skipper.skipTo(target) + 1`: the documents before
            // the block that can hold `target`.
            // A target of 0 (the first `nextDoc`) is read as 1: the reader
            // starts with `skipDoc[0] == 0`, so `skipTo(0)` would load no
            // entry at all and leave the first block without its extent.
            let new_doc_upto = skip.skip_to(target.max(1))?.wrapping_add(1);
            let consumed = i64::from(self.doc_freq.wrapping_sub(self.doc_count_left));
            if new_doc_upto >= consumed {
                let left = i64::from(self.doc_freq).wrapping_sub(new_doc_upto);
                self.doc_count_left = i32::try_from(left)
                    .ok()
                    .filter(|&l| l >= 0)
                    .ok_or_else(|| corrupted(format!("skip list skipped {new_doc_upto} docs")))?;
                self.prev_doc_id = skip.last_doc();
                let at = usize::try_from(skip.doc_pointer())
                    .map_err(|_| corrupted("negative skip doc pointer"))?;
                self.r.seek(at)?;
                // `getImpacts()` level 1: the skip list's level-1 entry that
                // covers `target` -- eight blocks' merged impacts, which is
                // what lets a scorer skip more than one block per decision.
                match skip.level1() {
                    Some((last, bytes)) if self.needs_impacts => {
                        if last != self.level1_last_doc_id
                            && decode_impacts_into(bytes, &mut self.level1_impacts).is_err()
                        {
                            self.level1_impacts.clear();
                        }
                        self.level1_last_doc_id = last;
                    }
                    _ => {
                        self.level1_impacts.clear();
                        self.level1_last_doc_id = NO_MORE_DOCS;
                    }
                }
                // `seekPosData`'s arguments: where this block's occurrences
                // start. A block reached sequentially keeps the origin the
                // previous skip left, which the entry just read replaces.
                let (pos_fp, pos_upto, pay_fp) = skip.pos_state();
                self.block_pos_origin = PosCursorState {
                    pos_fp: pos_fp as u64,
                    pay_fp: pay_fp as u64,
                    pos_buffer_upto: u8::try_from(pos_upto)
                        .map_err(|_| corrupted(format!("posBufferUpto {pos_upto}")))?,
                };
            }
        }
        let body_start = self.r.position();
        if self.doc_count_left < self.bs {
            // The skip landed in the tail.
            self.level0_impacts.clear();
            self.level0_impacts_stale = false;
            self.level0_last_doc_id = NO_MORE_DOCS;
            return Ok(NO_MORE_DOCS);
        }
        // SENTINEL-OK: `skip_to` ran above (every `target >= 0` passes the
        // `-1` test), so the list is positioned and this is a document or
        // `i32::MAX`, never `-1`.
        let last = skip.next_skip_doc();
        if last == i32::MAX {
            self.level0_impacts.clear();
            self.level0_impacts_stale = false;
            self.level0_last_doc_id = NO_MORE_DOCS;
            self.pending = Some(PendingBlock {
                base_doc_id: self.prev_doc_id,
                last_doc_id: NO_MORE_DOCS,
                body_start,
                body_end: usize::MAX,
            });
            return Ok(NO_MORE_DOCS);
        }
        let body_end = usize::try_from(skip.next_doc_pointer())
            .map_err(|_| corrupted("negative skip doc pointer"))?;
        self.level0_impact_bytes = if self.needs_impacts {
            skip.level0_impacts()
        } else {
            &[]
        };
        self.level0_impacts.clear();
        self.level0_impacts_stale = self.needs_impacts;
        self.level0_last_doc_id = last;
        self.pending = Some(PendingBlock {
            base_doc_id: self.prev_doc_id,
            last_doc_id: last,
            body_start,
            body_end,
        });
        Ok(last)
    }

    /// `Lucene104PostingsReader.refillDocs`: unpack the block
    /// [`Self::advance_shallow`] positioned on. A no-op when there is nothing
    /// pending, so it is safe to call unconditionally before reading documents.
    fn refill(&mut self) -> Result<()> {
        let Some(p) = self.pending.take() else {
            return Ok(());
        };
        self.r.seek(p.body_start)?;
        let shape = if self.format == PostingsFormat::Lucene104 {
            decode_full_block_body(
                &mut self.r,
                p.base_doc_id,
                self.index_has_freq,
                self.needs_freq,
                &mut self.scratch,
                &mut self.block_docs,
                &mut self.block_freqs,
                true,
                !self.needs_freq,
            )?
        } else {
            // A retired generation's 128 documents, bit-set blocks expanded.
            crate::backward_codecs::postings::decode_block_body(
                &mut self.r,
                self.format,
                p.base_doc_id,
                self.index_has_freq,
                self.needs_freq,
                &mut self.block_docs,
                &mut self.block_freqs,
            )?;
            BodyShape::Docs
        };
        if p.body_end != usize::MAX {
            check_wire_position(self.r.position(), p.body_end, "full block body")?;
        }
        self.block_len = self.bs as usize;
        self.block_gen = self.block_gen.wrapping_add(1);
        self.block_pos = 0;
        self.prev_doc_id = if p.last_doc_id == NO_MORE_DOCS {
            // No header gave the extent: the block itself does.
            // ARITH: `bs` is 128 or 256.
            #[allow(clippy::arithmetic_side_effects)]
            self.block_docs[self.bs as usize - 1]
        } else {
            p.last_doc_id
        };
        self.bits = match shape {
            BodyShape::Bits { base, words } => {
                // Only a cursor that tracks ranks (for its frequencies) needs
                // the cumulative counts; a docs-only one recovers a rank on
                // demand (`materialize_bits`).
                // ARITH: at most 128 words of 64 bits, so every running total
                // is at most 8192 and fits a `u16`.
                #[allow(clippy::arithmetic_side_effects)]
                if self.needs_freq {
                    let mut running = 0u16;
                    for (rank, word) in self.scratch.word_ranks[..words]
                        .iter_mut()
                        .zip(&self.scratch.bitset[..words])
                    {
                        *rank = running;
                        running += word.count_ones() as u16;
                    }
                }
                Some((base, words))
            }
            BodyShape::Docs => None,
        };
        // ARITH: same invariant as `advance_shallow`'s own `pending.take()`
        // -- a `PendingBlock` exists only for a block `advance_shallow`
        // reached with `doc_count_left >= BLOCK_SIZE`, and this is the only
        // other place that consumes one.
        #[allow(clippy::arithmetic_side_effects)]
        {
            debug_assert!(self.doc_count_left >= self.bs);
            self.doc_count_left -= self.bs;
        }
        Ok(())
    }

    /// `advance(target)` inside the bit-set block [`Self::bits`] describes:
    /// the first set bit at or after `target`, found word by word
    /// (`FixedBitSet.nextSetBit`, which is Lucene's whole answer here), with
    /// `block_pos` moved to that document's rank from the per-word cumulative
    /// popcounts filled at refill (`BlockScratch::word_ranks`).
    ///
    /// The caller has established that the block's header claims a last
    /// document `>= target`; a bit set that disagrees -- no such bit, or one
    /// past the block's 256th document -- is reported as corruption, exactly
    /// as the expanded path reports it.
    fn bits_advance(&mut self, target: i32) -> Result<i32> {
        let Some((base, n)) = self.bits else {
            return Err(corrupted(
                "bit-set advance on a block that is not a bit set",
            ));
        };
        let words = &self.scratch.bitset[..n];
        // `target > base` whenever the cursor is inside or entering this
        // block (its base is one past the previous block's last document);
        // anything else starts at the block's first bit.
        let rel = if target > base {
            target.wrapping_sub(base) as u32 as usize
        } else {
            0
        };
        let not_reached = || {
            corrupted(
                "full block's decoded doc IDs do not reach the last doc ID its level-0 \
                 header claims",
            )
        };
        let bit = lucene_util::fixed_bit_set::next_set_bit_in_words(words, rel)
            .ok_or_else(not_reached)?;
        // `bit < 8192`, so the cast is exact; wrapping for the same reason
        // every doc id derived from the wire is.
        let doc = base.wrapping_add(bit as i32);
        if !self.needs_freq {
            // Lucene's docs-only branch: no position in the block is tracked
            // (every frequency reads as 1); `materialize_bits` works the rank
            // out if a sequential read ever needs it. The document must still
            // lie within what the block's header claims.
            if doc > self.prev_doc_id {
                return Err(not_reached());
            }
            self.doc_id = doc;
            return Ok(doc);
        }
        // `docCumulativeWordPopCounts[w] + bitCount(word & below)`: the set bits
        // in the words before this one, then those below it in its own -- one
        // lookup and one popcount, whatever the distance moved.
        // ARITH: `bit < 128 * 64`, so `bit >> 6 < 128` indexes both arrays;
        // `1 << (bit & 63)` shifts by at most 63 and is at least 1; the sum is
        // at most 8192.
        #[allow(clippy::arithmetic_side_effects)]
        let rank = {
            let w = bit >> 6;
            let below = (1u64 << (bit & 63)) - 1;
            usize::from(self.scratch.word_ranks[w]) + (words[w] & below).count_ones() as usize
        };
        if rank >= self.block_len {
            return Err(not_reached());
        }
        self.block_pos = rank;
        self.doc_id = doc;
        Ok(doc)
    }

    /// Expands a kept bit-set block into `block_docs`, for a caller that
    /// reads the doc ids as an array ([`Self::next_postings`]).
    fn materialize_bits(&mut self) {
        if let Some((base, n)) = self.bits.take() {
            let words = &self.scratch.bitset[..n];
            // A docs-only `advance` does not track the rank; recover it from
            // the current document when that document is in this block.
            if !self.needs_freq && self.doc_id >= base {
                let rel = self.doc_id.wrapping_sub(base) as u32 as usize;
                // ARITH: shifting right by 6 and masking with 63 cannot
                // overflow; a `rel` past this block (a corrupt document) only
                // makes `w` pass `n`, which the `min`/`get` below absorb.
                #[allow(clippy::arithmetic_side_effects)]
                let (w, b) = (rel >> 6, rel & 63);
                let below: usize = words[..w.min(n)]
                    .iter()
                    .map(|x| x.count_ones() as usize)
                    .fold(0usize, usize::saturating_add);
                let below_mask = u64::MAX
                    .checked_shr(64u32.saturating_sub(b as u32))
                    .unwrap_or(0);
                let within = words.get(w).map_or(0, |x| (x & below_mask).count_ones());
                self.block_pos = below.saturating_add(within as usize).min(self.block_len);
            }
            lucene_util::simd::expand_bitset(words, base, &mut self.block_docs);
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
            self.level1_doc_count_upto = self.level1_doc_count_upto.wrapping_add(self.level1_docs);

            if self.doc_count_left < self.level1_docs {
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
                // The `needsImpacts` half: a cursor that never asks for
                // impacts (level 0 already skips its own the same way) does
                // not decode this span's either -- a merged level-1 list can
                // hold hundreds of pairs.
                if self.needs_impacts {
                    decode_impacts_into(entry.impact_bytes, &mut self.level1_impacts)?;
                }
                break;
            }
            // The whole span is behind `target`: loop again, which re-seeks to
            // `level1_doc_end_fp` (past this span's 32 blocks) without touching
            // any of their level-0 headers.
        }
        Ok(())
    }
}

/// `BlockPostingsEnum` opened with `PostingsEnum.POSITIONS`: a
/// [`LazyDocsCursor`] plus a lazy `.pos` reader. Nothing in `.pos` is touched
/// until a document's positions are asked for, and the positions of every
/// document that is never asked about are stepped over -- inside the current
/// block by an index bump, across whole blocks by `PForUtil.skip` -- rather
/// than decoded.
///
/// This is what an exact phrase needs: its conjunction visits a small
/// fraction of each term's documents, and of those only the ones that survive
/// the score check need positions at all. The eager
/// [`read_positions_for_docs`] this replaces for phrases decoded every
/// position of every candidate into one `Vec` before matching a single one.
///
/// # How the `.pos` stream is kept in step with `.doc`
///
/// Lucene tracks a running `posPendingCount` and, at each `.doc` block
/// boundary, either seeks `.pos` to the block's recorded origin or -- when that
/// origin lies inside the positions block already decoded -- adds the
/// remaining frequencies of the old block to the pending count
/// (`seekPosData`). This port reaches the same state without remembering the
/// old block: the origin names the positions block *and* the index inside it
/// where the new block's first document starts, so when that block is the one
/// in the buffer the cursor simply moves its index there.
///
/// Within a `.doc` block, the positions of the documents between the last one
/// read and the current one -- plus any unread positions of the last one --
/// are the in-block frequency sum Java's `accumulatePendingPositions` adds.
pub struct PositionsCursor<'a> {
    docs: LazyDocsCursor<'a>,
    pos_r: SliceInput<'a>,
    for_util: for_util::ForUtil,
    block: PositionBlock,
    /// Next unread entry of `block.pos_deltas`; `== block.len` means the next
    /// read refills.
    buf_upto: usize,
    /// `.pos` offset the block in `block` was decoded from; `u64::MAX` when
    /// the buffer holds nothing.
    buf_fp: u64,
    /// Where the vint-coded tail block starts (`lastPosBlockFP`), and how many
    /// occurrences it holds.
    last_pos_block_fp: Option<u64>,
    tail_count: usize,
    wants: PositionWants,
    /// [`LazyDocsCursor::block_gen`] when this cursor last lined `.pos` up
    /// with the `.doc` block.
    block_gen: u64,
    /// First document (index into the `.doc` block) whose positions have not
    /// been accounted for.
    pos_doc_upto: usize,
    /// Occurrences to step over before the next document's first one.
    pending: u64,
    /// The document whose positions are being read, or `-1`.
    pos_doc: i32,
    /// Occurrences of `pos_doc` not yet read.
    ///
    /// A `u64`, not the `u32` a frequency fits in, on purpose: next to the
    /// `i32` `position`, LLVM fused `doc_left -= 1` and the position add into
    /// one two-lane vector op whose 64-bit load spans the two 32-bit stores
    /// `start_doc` makes -- a store-forwarding stall on every document's first
    /// position, and most of what a positions walk cost.
    doc_left: u64,
    position: i32,
    /// The generation that wrote `.pos`: `Lucene104` packs 256 occurrences a
    /// block, every retired generation 128 with its own `PForUtil` -- the
    /// layout is otherwise the same, down to the vint tail.
    format: PostingsFormat,
    /// [`Self::occurrences_into`]'s decode buffers, made on its first call.
    occ_scratch: Option<Box<WalkScratch>>,
    /// `.pay`, once [`Self::read_payloads`] asked for payloads: every full
    /// positions block then decodes its payload block too, and every block
    /// stepped over steps over its `.pay` half.
    pay_r: Option<SliceInput<'a>>,
    /// The next unread payload byte of `block.payload_bytes`.
    payload_upto: usize,
    /// The last position's payload: its start in `block.payload_bytes` and
    /// its length.
    payload: (usize, usize),
}

impl<'a> PositionsCursor<'a> {
    /// `docs` must have been opened with [`PostingsFlags::Freqs`] on a field
    /// indexed with positions; `meta`/`total_term_freq` are the term's.
    pub(crate) fn new(
        docs: LazyDocsCursor<'a>,
        pos: &PosInput<'a>,
        meta: TermMetadata,
        total_term_freq: i64,
        has_offsets: bool,
        has_payloads: bool,
    ) -> Result<Self> {
        let mut docs = docs;
        docs.track_positions();
        let bs = pos.format.block_size() as i64;
        // `total_term_freq % BLOCK_SIZE`, which is in `0..256`.
        let tail_count = total_term_freq.rem_euclid(bs) as usize;
        Ok(PositionsCursor {
            docs,
            pos_r: SliceInput::new(pos.buf),
            for_util: for_util::ForUtil::new(),
            block: PositionBlock::new(),
            buf_upto: 0,
            buf_fp: u64::MAX,
            last_pos_block_fp: last_pos_block_fp_for(meta, total_term_freq, bs),
            tail_count,
            // Offsets and payloads are never read here, but in the vint tail
            // they are interleaved with the positions in `.pos` and must be
            // stepped over; full blocks keep them in `.pay`, which this
            // cursor never opens.
            wants: PositionWants {
                has_offsets,
                has_payloads,
                want_offsets: false,
                want_payloads: false,
            },
            block_gen: u64::MAX,
            pos_doc_upto: 0,
            pending: 0,
            pos_doc: -1,
            doc_left: 0,
            position: 0,
            format: pos.format,
            occ_scratch: None,
            pay_r: None,
            payload_upto: 0,
            payload: (0, 0),
        })
    }

    /// `PostingsEnum.PAYLOADS`: from here on [`Self::payload`] answers each
    /// position's payload, read as the positions are -- `.pay` stepped in
    /// lockstep with `.pos`, each payload block decoded once -- rather than
    /// walked again per document as [`Self::occurrences_into`] does.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for a field without payloads, a retired
    /// format's cursor, or a cursor that has already read positions.
    pub fn read_payloads(&mut self, pay: &PayInput<'a>) -> Result<()> {
        if !self.wants.has_payloads {
            return Err(Error::Unsupported(
                "read_payloads needs a field with payloads",
            ));
        }
        if self.format != PostingsFormat::Lucene104 {
            return Err(Error::Unsupported(
                "read_payloads reads Lucene104 postings only",
            ));
        }
        if self.buf_fp != u64::MAX || self.pos_doc != -1 {
            return Err(Error::Unsupported(
                "read_payloads must be asked before the first position is read",
            ));
        }
        self.pay_r = Some(SliceInput::new(pay.buf));
        self.wants.want_payloads = true;
        Ok(())
    }

    /// `PostingsEnum.getPayload()` after [`Self::next_position`]: `None`
    /// where the position has none, payloads are not being read, or no
    /// position of the current document has been read yet (the last one
    /// read was another document's).
    ///
    /// # Errors
    /// Payload lengths that overrun the block's payload bytes.
    pub fn payload(&self) -> Result<Option<&[u8]>> {
        let (start, len) = self.payload;
        if self.pay_r.is_none() || len == 0 || self.docs.doc_id != self.pos_doc {
            return Ok(None);
        }
        Ok(Some(self.block.payload(start, len)?))
    }

    /// The payload bytes of the block's occurrences `from..to`.
    fn payload_span(&self, from: usize, to: usize) -> usize {
        self.block.payload_lengths[from..to]
            .iter()
            .fold(0usize, |acc, &l| acc.saturating_add(l as usize))
    }

    /// The current document's occurrences with their offsets and payloads
    /// (`nextPosition()` and `startOffset()`/`endOffset()`/`getPayload()`
    /// for each, as a cursor opened with `PostingsEnum.ALL` reads them),
    /// replacing `out`'s contents.
    ///
    /// Read from `.pos`/`.pay` at the origin the `.doc` skip data gives the
    /// document -- [`read_occurrences_for_doc`]'s walk, without its term
    /// lookup and skip-list descent per call, since this cursor is already on
    /// the document -- and independently of [`Self::next_position`]'s
    /// stream, which is left where it was.
    ///
    /// # Errors
    /// [`Error::Unsupported`] off a document, or for a retired format's
    /// cursor (whose documents [`read_occurrences_for_doc`] reads); the
    /// walk's corruption errors.
    pub fn occurrences_into(
        &mut self,
        pos: &PosInput<'_>,
        pay: Option<&PayInput<'_>>,
        out: &mut Vec<Position>,
    ) -> Result<()> {
        if self.format != PostingsFormat::Lucene104 {
            return Err(Error::Unsupported(
                "occurrences_into reads Lucene104 postings only",
            ));
        }
        let origin = self.docs.position_origin()?.ok_or(Error::Unsupported(
            "occurrences_into needs the cursor positioned on a document",
        ))?;
        let freq = usize::try_from(self.freq())
            .ok()
            .filter(|&f| f > 0)
            .ok_or_else(|| corrupted("document frequency is not positive"))?;
        let wants = PositionWants {
            has_offsets: self.wants.has_offsets,
            has_payloads: self.wants.has_payloads,
            want_offsets: self.wants.has_offsets,
            want_payloads: self.wants.has_payloads,
        };
        let scratch = self
            .occ_scratch
            .get_or_insert_with(|| Box::new(WalkScratch::new()));
        out.clear();
        let mut sink = FullOccurrences {
            occurrences: std::mem::take(out),
            doc_starts: Vec::new(),
        };
        let walked = walk_document_occurrences(
            pos,
            pay,
            origin,
            freq,
            self.last_pos_block_fp,
            self.tail_count,
            wants,
            scratch,
            &mut sink,
        );
        *out = sink.occurrences;
        walked
    }

    /// The underlying document cursor, for everything that is not a position:
    /// `advance_shallow`, impacts, block boundaries.
    #[inline]
    pub fn docs(&self) -> &LazyDocsCursor<'a> {
        &self.docs
    }

    #[inline]
    pub fn docs_mut(&mut self) -> &mut LazyDocsCursor<'a> {
        &mut self.docs
    }

    #[inline]
    pub fn doc_id(&self) -> i32 {
        self.docs.doc_id()
    }

    #[inline]
    pub fn next_doc(&mut self) -> Result<i32> {
        self.docs.next_doc()
    }

    #[inline]
    pub fn advance(&mut self, target: i32) -> Result<i32> {
        self.docs.advance(target)
    }

    /// The current document's frequency: how many times
    /// [`Self::next_position`] may be called for it.
    #[inline]
    pub fn freq(&self) -> i32 {
        self.docs.freq().unwrap_or(1)
    }

    /// `PostingsEnum.nextPosition()`: the current document's next position,
    /// ascending. Callable at most [`Self::freq`] times per document.
    ///
    /// The common step -- the document's positions already lined up, the
    /// next one in the decoded block, no payloads -- is small enough to
    /// inline into the caller's loop, where it cannot fail and so builds no
    /// `Result` through memory; a document's first position, a refill and
    /// payloads go through [`Self::next_position_slow`].
    #[inline]
    pub fn next_position(&mut self) -> Result<i32> {
        match self.try_next_position() {
            Some(p) => Ok(p),
            None => self.next_position_slow(),
        }
    }

    /// [`Self::next_position`]'s infallible step: `None` where it needs
    /// [`Self::next_position_slow`] (a new document, an exhausted block, a
    /// cursor reading payloads, or a document with no position left).
    #[inline]
    pub fn try_next_position(&mut self) -> Option<i32> {
        // Payloads first: a cursor reading them never takes this step, and
        // should not pay for a `try_start_doc` call per document either.
        if self.pay_r.is_some()
            || (self.docs.doc_id != self.pos_doc && !self.try_start_doc())
            || self.doc_left == 0
            || self.buf_upto >= self.block.len
        {
            return None;
        }
        let delta = *self.block.pos_deltas.get(self.buf_upto)?;
        // ARITH: `buf_upto < block.len <= BLOCK_SIZE` was just checked, and
        // a position is `wrapping_add`ed exactly as Java's `int` sum wraps;
        // `doc_left > 0` was just checked.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.position = self.position.wrapping_add(delta as i32);
            self.buf_upto += 1;
            self.doc_left -= 1;
        }
        Some(self.position)
    }

    /// [`Self::start_doc`]'s common case -- the document in the `.doc`
    /// block `.pos` is already lined up with, its first position inside the
    /// decoded positions block, no payloads -- without building a `Result`:
    /// `false`, having changed nothing, where `start_doc` must run.
    #[inline(never)]
    fn try_start_doc(&mut self) -> bool {
        let doc = self.docs.doc_id;
        if doc < 0
            || doc == NO_MORE_DOCS
            || self.docs.pending.is_some()
            || self.docs.block_gen != self.block_gen
            || self.pay_r.is_some()
        {
            return false;
        }
        let block_pos = self.docs.block_pos;
        let Some((&freq, between)) = self
            .docs
            .block_freqs
            .get(self.pos_doc_upto..=block_pos)
            .and_then(<[i32]>::split_last)
        else {
            return false;
        };
        let mut skip = self.pending.wrapping_add(self.doc_left);
        for &f in between {
            if f <= 0 {
                return false;
            }
            skip = skip.wrapping_add(f as u64);
        }
        let Some(left) = self.block.len.checked_sub(self.buf_upto) else {
            return false;
        };
        if freq <= 0 || skip > left as u64 {
            return false;
        }
        // ARITH: `skip <= block.len - buf_upto` and `block_pos` indexes
        // `block_freqs`, both just checked.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.buf_upto += skip as usize;
            self.pos_doc_upto = block_pos + 1;
        }
        self.pending = 0;
        self.doc_left = freq as u64;
        self.position = 0;
        self.pos_doc = doc;
        true
    }

    /// [`Self::next_position`] in full.
    #[inline(never)]
    fn next_position_slow(&mut self) -> Result<i32> {
        if self.docs.doc_id != self.pos_doc {
            self.start_doc()?;
        }
        if self.doc_left == 0 {
            return Err(Error::Unsupported(
                "next_position called more times than the document's frequency",
            ));
        }
        if self.buf_upto >= self.block.len {
            self.refill()?;
            if self.buf_upto >= self.block.len {
                return Err(positions_overrun());
            }
        }
        if self.pay_r.is_some() {
            let len = self.block.payload_lengths[self.buf_upto] as usize;
            self.payload = (self.payload_upto, len);
            self.payload_upto = self.payload_upto.saturating_add(len);
        }
        // ARITH: `buf_upto < block.len <= BLOCK_SIZE` after the refill above,
        // and a position is `wrapping_add`ed exactly as Java's `int` sum
        // wraps; `doc_left > 0` was just checked.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.position = self
                .position
                .wrapping_add(self.block.pos_deltas[self.buf_upto] as i32);
            self.buf_upto += 1;
            self.doc_left -= 1;
        }
        Ok(self.position)
    }

    /// Every position of the current document not yet read, appended to
    /// `out` in order: [`Self::next_position`] `doc_left` times, with the
    /// checks made once per positions block instead of once per position. A
    /// phrase scorer reads each term's positions whole before matching them.
    pub fn positions_into(&mut self, out: &mut Vec<i32>) -> Result<()> {
        if self.docs.doc_id != self.pos_doc {
            self.start_doc()?;
        }
        while self.doc_left > 0 {
            if self.buf_upto >= self.block.len {
                self.refill()?;
                // An empty refill (the frequencies claim more positions than
                // `totalTermFreq` left) would never make progress.
                if self.buf_upto >= self.block.len {
                    return Err(positions_overrun());
                }
            }
            // ARITH: `buf_upto < block.len <= BLOCK_SIZE` after the refill,
            // so `take <= block.len - buf_upto` stays in the block and
            // `take <= doc_left`; positions wrap as Java's `int` sum does.
            #[allow(clippy::arithmetic_side_effects)]
            {
                let take = (self.block.len - self.buf_upto).min(self.doc_left as usize);
                if self.pay_r.is_some() {
                    let bytes = self.payload_span(self.buf_upto, self.buf_upto + take);
                    self.payload_upto = self.payload_upto.saturating_add(bytes);
                    self.payload = (0, 0);
                }
                let deltas = &self.block.pos_deltas[self.buf_upto..self.buf_upto + take];
                let mut p = self.position;
                out.extend(deltas.iter().map(|&d| {
                    p = p.wrapping_add(d as i32);
                    p
                }));
                self.position = p;
                self.buf_upto += take;
                self.doc_left -= take as u64;
            }
        }
        Ok(())
    }

    /// Lines `.pos` up with the current document: `seekPosData` if the `.doc`
    /// block changed, then `accumulatePendingPositions`.
    fn start_doc(&mut self) -> Result<()> {
        let doc = self.docs.doc_id;
        if doc < 0 || doc == NO_MORE_DOCS || self.docs.pending.is_some() {
            return Err(Error::Unsupported(
                "next_position needs the cursor positioned on a decoded document",
            ));
        }
        if self.docs.block_gen != self.block_gen {
            self.block_gen = self.docs.block_gen;
            let origin = self.docs.block_pos_origin;
            self.pos_doc_upto = 0;
            self.doc_left = 0;
            let start = origin.pos_buffer_upto as usize;
            if origin.pos_fp == self.buf_fp && start <= self.block.len && start >= self.buf_upto {
                // The new block's first document starts inside the positions
                // block already decoded: move to it, decode nothing.
                if self.pay_r.is_some() {
                    let bytes = self.payload_span(self.buf_upto, start);
                    self.payload_upto = self.payload_upto.saturating_add(bytes);
                }
                self.buf_upto = start;
                self.pending = 0;
            } else {
                let fp = usize::try_from(origin.pos_fp)
                    .map_err(|_| corrupted("positions file pointer overflows usize"))?;
                self.pos_r.seek(fp)?;
                if let Some(pay_r) = self.pay_r.as_mut() {
                    let pay_fp = usize::try_from(origin.pay_fp)
                        .map_err(|_| corrupted("payloads file pointer overflows usize"))?;
                    pay_r.seek(pay_fp)?;
                }
                self.buf_fp = u64::MAX;
                self.block.len = 0;
                self.buf_upto = 0;
                self.pending = origin.pos_buffer_upto as u64;
            }
        }
        let block_pos = self.docs.block_pos;
        let mut skip = self.pending.wrapping_add(self.doc_left);
        // Documents between the last one read and this one; none at all on a
        // sequential walk, which is the common case this keeps branch-cheap.
        if self.pos_doc_upto < block_pos {
            for &freq in &self.docs.block_freqs[self.pos_doc_upto..block_pos] {
                let freq = u64::try_from(freq).map_err(|_| {
                    corrupted("negative per-doc frequency in the current .doc block")
                })?;
                skip = skip.wrapping_add(freq);
            }
        }
        if skip != 0 {
            self.skip_positions(skip)?;
        }
        self.pending = 0;
        // ARITH: `block_pos < block_len <= BLOCK_SIZE` for a decoded document.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.pos_doc_upto = block_pos + 1;
        }
        let freq = self.docs.block_freqs[block_pos];
        self.doc_left = u64::try_from(freq)
            .ok()
            .filter(|&f| f > 0)
            .ok_or_else(|| corrupted(format!("document frequency {freq} is not positive")))?;
        self.position = 0;
        self.payload = (0, 0);
        self.pos_doc = doc;
        Ok(())
    }

    /// `skipPositions`: step over `n` occurrences -- inside the buffer by an
    /// index bump, across whole blocks with `PForUtil.skip`, then one refill
    /// for the block the next position is in.
    fn skip_positions(&mut self, n: u64) -> Result<()> {
        // ARITH: `buf_upto <= block.len` is this cursor's invariant.
        #[allow(clippy::arithmetic_side_effects)]
        let left = (self.block.len - self.buf_upto) as u64;
        if n <= left {
            // ARITH: `n <= left` fits, and lands at most on `block.len`.
            #[allow(clippy::arithmetic_side_effects)]
            let to = self.buf_upto + n as usize;
            if self.pay_r.is_some() {
                let bytes = self.payload_span(self.buf_upto, to);
                self.payload_upto = self.payload_upto.saturating_add(bytes);
            }
            self.buf_upto = to;
            return Ok(());
        }
        // ARITH: `n > left` was just established.
        #[allow(clippy::arithmetic_side_effects)]
        let mut n = n - left;
        self.buf_upto = self.block.len;
        let bs = self.format.block_size() as u64;
        while n >= bs {
            if Some(self.pos_r.position() as u64) == self.last_pos_block_fp {
                return Err(corrupted(
                    "skipping positions ran into the vint tail with a full block left to skip",
                ));
            }
            if self.format == PostingsFormat::Lucene104 {
                match self.pay_r.as_mut() {
                    // `.pay` keeps step: its payload (and offset) blocks
                    // follow the positions blocks one for one.
                    Some(pay_r) => skip_position_block(
                        &mut self.pos_r,
                        Some(pay_r),
                        self.wants.has_payloads,
                        self.wants.has_offsets,
                    )?,
                    None => skip_position_block(&mut self.pos_r, None, false, false)?,
                }
            } else {
                crate::backward_codecs::for_util::pfor_skip(&mut self.pos_r, self.format.word())?;
            }
            // ARITH: guarded by the loop condition.
            #[allow(clippy::arithmetic_side_effects)]
            {
                n -= bs;
            }
        }
        if n > 0 {
            self.refill()?;
            if n > self.block.len as u64 {
                return Err(corrupted("positions to skip overrun the positions block"));
            }
            self.buf_upto = n as usize;
            if self.pay_r.is_some() {
                self.payload_upto = self.payload_span(0, self.buf_upto);
            }
        }
        Ok(())
    }

    /// `refillPositions`: the next positions block, full (`PForUtil`) or the
    /// vint tail.
    fn refill(&mut self) -> Result<()> {
        let fp = self.pos_r.position() as u64;
        if Some(fp) == self.last_pos_block_fp {
            refill_last_position_block(
                &mut self.pos_r,
                self.wants,
                self.tail_count,
                &mut self.block,
            )?;
        } else if self.format == PostingsFormat::Lucene104 {
            match self.pay_r.as_mut() {
                Some(pay_r) => refill_full_position_block(
                    &mut self.pos_r,
                    Some(pay_r),
                    self.wants,
                    &mut self.for_util,
                    &mut self.block,
                )?,
                None => {
                    self.for_util
                        .pfor_decode(&mut self.pos_r, &mut self.block.pos_deltas)?;
                    self.block.len = for_util::BLOCK_SIZE;
                }
            }
        } else {
            // A retired generation's 128 deltas, in the front of the buffer.
            let mut words = [0u64; crate::backward_codecs::for_util::BLOCK_SIZE];
            crate::backward_codecs::for_util::pfor_decode(
                &mut self.pos_r,
                self.format.word(),
                &mut words,
            )?;
            for (d, &w) in self.block.pos_deltas.iter_mut().zip(words.iter()) {
                *d = w as u32;
            }
            self.block.len = crate::backward_codecs::for_util::BLOCK_SIZE;
        }
        self.buf_fp = fp;
        self.buf_upto = 0;
        self.payload_upto = 0;
        Ok(())
    }
}
#[cfg(test)]
mod tests;
