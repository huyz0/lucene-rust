//! Port of `org.apache.lucene.codecs.lucene103.blocktree.Lucene103BlockTreeTermsReader`
//! (`.tim` term dictionary + `.tip` term index + `.tmd` per-field metadata),
//! including its **lazy** `SegmentTermsEnum`/`SegmentTermsEnumFrame`
//! navigation.
//!
//! Note on naming: the pinned Lucene version (10.5.0) uses
//! `Lucene104PostingsFormat`, whose term dictionary is
//! `Lucene103BlockTreeTermsReader`/`Writer` (package
//! `o.a.l.codecs.lucene103.blocktree`) — *not* the `lucene90.blocktree`
//! classes, which live in `backward-codecs` and are out of scope for this
//! port (see PLAN.md's "pin one Lucene version" rule). The `.tip` term index
//! in this version is **not an FST** — Lucene 10.x replaced it with a
//! purpose-built binary trie (`TrieReader`/`TrieBuilder`), a flatter,
//! pointer-chasing encoding of the same "prefix trie whose leaves are term
//! blocks" idea `fst.rs`'s module doc describes for the *older* format.
//! `fst.rs` remains useful groundwork (arc-lookup style reasoning, shared
//! `codec_util` header handling) but is not used by this module — and no
//! codec in `lucene/core` references `o.a.l.util.fst` either, so that is
//! parity, not a gap.
//!
//! ## Wire format
//!
//! - `.tmd` (`TERMS_META_EXTENSION`): `IndexHeader(codec="BlockTreeTermsMeta")`,
//!   then the postings reader's own `init` header
//!   (`IndexHeader(codec="Lucene104PostingsWriterTerms")` + `indexBlockSize: vint`,
//!   which must equal `Lucene104PostingsFormat.BLOCK_SIZE` = 256 for this pinned
//!   version), then `numFields: vint`, then per field: `fieldNumber: vint`,
//!   `numTerms: vlong`, a `sumTotalTermFreq`/`sumDocFreq` pair (see
//!   [`read_freq_pair`] for the DOCS-only aliasing trick), `docCount: vint`,
//!   `minTerm`/`maxTerm` (vint-length-prefixed byte arrays), and finally
//!   `indexStart`/`rootFP`/`indexEnd` (three vlongs locating this field's root
//!   node in `.tip`). After the field loop: `indexLength: i64`, `termsLength: i64`,
//!   `Footer`.
//! - `.tip` (`TERMS_INDEX_EXTENSION`): `IndexHeader(codec="BlockTreeTermsIndex")`,
//!   then every field's trie nodes packed back to back (each field's node
//!   region spans `[indexStart, indexEnd)` from its `.tmd` record), `Footer`.
//!   A trie node's header byte packs a 2-bit `sign` selecting one of three
//!   encodings (`SIGN_NO_CHILDREN`/`SIGN_SINGLE_CHILD_*`/`SIGN_MULTI_CHILDREN`);
//!   see `TrieReader.java`/`TrieBuilder.java` for the full byte-packing scheme
//!   ([`load_node`] is a direct transliteration of `TrieReader.load`/
//!   `loadLeafNode`/`loadSingleChildNode`/`loadMultiChildrenNode`, and
//!   [`lookup_child`] of `TrieReader.lookupChild` plus all three
//!   `ChildSaveStrategy` decodings).
//! - `.tim` (`TERMS_EXTENSION`): `IndexHeader(codec="BlockTreeTermsDict")`, then
//!   every field's blocks packed back to back, each block laid out as
//!   `SegmentTermsEnumFrame.loadBlock` reads it (see [`Frame::load_block`]).
//!
//! ## Shape: lazy frames, not a materialized dictionary
//!
//! [`open`] reads **only** the `.tmd` records — per-field counts, min/max
//! term, and the `(indexStart, rootFP, indexEnd)` triple locating the field's
//! trie — and then stops, exactly like `Lucene103BlockTreeTermsReader`'s
//! constructor plus one `FieldReader` per field. No `.tim` block is touched
//! until a lookup asks for one.
//!
//! A lookup runs [`SegmentTermsEnum`], the port of Java's class of the same
//! name: it walks the `.tip` trie one label at a time
//! ([`lookup_child`]), pushes a [`Frame`] per trie node that carries an
//! output block, picks the one floor sub-block whose label range covers the
//! target ([`Frame::scan_to_floor_frame`]), loads *that* block and no other
//! ([`Frame::load_block`]), and scans or binary-searches its suffix bytes
//! ([`Frame::scan_to_term`]). Per-term postings metadata is decoded lazily on
//! top of that, only up to the term actually landed on
//! ([`Frame::decode_meta_data`]), so a terms-only consumer never pays for it.
//! `next()` walks blocks and in-block sub-block pointers with the same frame
//! stack, never re-consulting the trie.
//!
//! **Why lazy.** Materializing every field's terms at open costs
//! `O(all terms in the segment)` in time and memory (35 ms for the benchmark
//! corpus' 579k-term segment, where Lucene's whole `DirectoryReader.open`
//! costs 0.34 ms), and a search engine reopens readers on every refresh.
//!
//! ## Fallible lookups
//!
//! Decoding a block can fail on corrupt bytes, and with lazy loading that
//! failure necessarily surfaces at **lookup** time rather than at [`open`] —
//! which is also where real Lucene surfaces it (`loadBlock` throws
//! `CorruptIndexException` from inside `TermsEnum.seekExact`). Every lookup
//! therefore has a `Result`-returning form: [`FieldTerms::try_seek_exact`],
//! [`TermsEnum::try_next`], [`TermsEnum::try_seek_ceil`]. The older
//! infallible spellings ([`FieldTerms::seek_exact`], [`TermsEnum::next`],
//! [`TermsEnum::seek_ceil`]) are kept for callers that have no error channel;
//! they report a corrupt block as "no such term"/end-of-terms, and each says
//! so in its own doc comment. New code should prefer the `try_` forms.
//!
//! ## Suffix compression
//!
//! `CompressionAlgorithm::LZ4` (reusing `crate::lz4::decompress`) and
//! `LowercaseAscii` (a standalone port of
//! `LowercaseAsciiCompression.decompress`, see [`decompress_lowercase_ascii`])
//! are both decoded, alongside `NO_COMPRESSION`, and this port's own
//! blocktree writer emits all three exactly where Java does
//! (`crate::blocktree_writer`). Code `3` (never assigned to a
//! `CompressionAlgorithm` constant) is rejected in a stock dictionary, as
//! `Error::Store(Corrupted)`, matching `CompressionAlgorithm.byCode`'s own
//! `IllegalArgumentException`.
//!
//! ## Zstd blocks
//!
//! A dictionary written with `postings_writer::WriteOptions::zstd_term_blocks`
//! (the `ZstdTerms104` postings format) carries [`ZSTD_TERMS_CODEC_NAME`] and
//! its siblings in place of the three stock header names, so stock Lucene
//! refuses it. Its blocks may use code `3`: a whole block after its token as
//! one bare zstd frame (`crate::blocktree_writer::BlockZstd` has the layout),
//! decoded against the field's dictionary, which ends its `.tmd` record. The
//! decoded sections are read exactly as an ordinary block's.

use std::sync::{Arc, Mutex};

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use crate::automaton::{ByteDfa, CompiledDfa, DfaWalker, LazyDfa, TermAutomaton, Verdict, DEAD};
use crate::field_infos::{FieldInfos, IndexOptions};
use crate::fuzzy::FuzzyMatch;
use crate::postings::{self, DocInput, Postings, TermMetadata};

/// One term's postings plus its positions in flat form: `(postings, positions,
/// doc_starts)`, where document `i`'s positions are
/// `positions[doc_starts[i] as usize..doc_starts[i + 1] as usize]`. See
/// [`FieldTerms::positions_flat`].
pub type FlatPositions = (Postings, Vec<i32>, Vec<u32>);
use crate::regexp::RegexpPattern;
use crate::wildcard::WildcardPattern;

pub(crate) const TERMS_CODEC_NAME: &str = "BlockTreeTermsDict";
pub(crate) const TERMS_INDEX_CODEC_NAME: &str = "BlockTreeTermsIndex";
pub(crate) const TERMS_META_CODEC_NAME: &str = "BlockTreeTermsMeta";
/// The headers of a dictionary whose blocks may be zstd blocks
/// (`blocktree_writer::BlockZstd`) and whose `.tmd` field records each end
/// with the field's zstd dictionary (`vInt` length, then the bytes; 0 for
/// none). Stock Lucene rejects these names, so it never misreads a zstd block.
pub(crate) const ZSTD_TERMS_CODEC_NAME: &str = "ZstdBlockTreeTermsDict";
pub(crate) const ZSTD_TERMS_INDEX_CODEC_NAME: &str = "ZstdBlockTreeTermsIndex";
pub(crate) const ZSTD_TERMS_META_CODEC_NAME: &str = "ZstdBlockTreeTermsMeta";
const VERSION_START: i32 = 0;
pub(crate) const VERSION_CURRENT: i32 = 0;

/// `Lucene104PostingsFormat.TERMS_CODEC` — the postings writer's own header,
/// embedded in the `.tmd` stream right after BlockTree's own index header.
pub(crate) const POSTINGS_TERMS_CODEC: &str = "Lucene104PostingsWriterTerms";
/// `Lucene103PostingsFormat.TERMS_CODEC`: the 10.3 postings under this same
/// trie dictionary.
const LUCENE103_POSTINGS_TERMS_CODEC: &str = "Lucene103PostingsWriterTerms";

/// The postings reader's terms codec that follows `.tmd`'s own index header
/// (`PostingsReaderBase.init`), or `None` when `.tmd` does not start with a
/// `BlockTreeTermsMeta` header for this segment at any version either
/// dictionary generation writes.
fn postings_terms_codec(
    tmd: &[u8],
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
) -> Option<String> {
    let mut r = SliceInput::new(tmd);
    codec_util::check_index_header(
        &mut r,
        TERMS_META_CODEC_NAME,
        0,
        crate::backward_codecs::blocktree::VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )
    .ok()?;
    codec_util::peek_codec_name(r.as_slice())
}
pub(crate) const POSTINGS_VERSION_START: i32 = 0;
pub(crate) const POSTINGS_VERSION_CURRENT: i32 = 0;
/// `Lucene104PostingsFormat.BLOCK_SIZE` (= `ForUtil.BLOCK_SIZE`), the postings
/// block size the `.tmd` stream's `indexBlockSize` field must match.
pub(crate) const POSTINGS_BLOCK_SIZE: i32 = 256;

/// `TrieBuilder.SIGN_NO_CHILDREN` — a leaf trie node (no children).
pub(crate) const SIGN_NO_CHILDREN: u32 = 0x00;
/// `TrieBuilder.SIGN_SINGLE_CHILD_WITH_OUTPUT`.
pub(crate) const SIGN_SINGLE_CHILD_WITH_OUTPUT: u32 = 0x01;
/// `TrieBuilder.SIGN_SINGLE_CHILD_WITHOUT_OUTPUT`.
pub(crate) const SIGN_SINGLE_CHILD_WITHOUT_OUTPUT: u32 = 0x02;
/// `TrieBuilder.SIGN_MULTI_CHILDREN`.
pub(crate) const SIGN_MULTI_CHILDREN: u32 = 0x03;
/// `TrieBuilder.LEAF_NODE_HAS_TERMS` (`1 << 5`).
pub(crate) const LEAF_NODE_HAS_TERMS: u32 = 1 << 5;
/// `TrieBuilder.LEAF_NODE_HAS_FLOOR` (`1 << 6`).
pub(crate) const LEAF_NODE_HAS_FLOOR: u32 = 1 << 6;
/// `TrieBuilder.NON_LEAF_NODE_HAS_TERMS` (`1L << 1`) — the equivalent flag
/// packed into a non-leaf node's *encoded output fp* (`encodeFP`), not its
/// header byte, since non-leaf nodes' header bits are all spoken for by
/// child-pointer bookkeeping.
pub(crate) const NON_LEAF_NODE_HAS_TERMS: u64 = 1 << 1;
/// `TrieBuilder.NON_LEAF_NODE_HAS_FLOOR` (`1L << 0`).
pub(crate) const NON_LEAF_NODE_HAS_FLOOR: u64 = 1;
/// `TrieBuilder.ChildSaveStrategy.REVERSE_ARRAY.code`.
pub(crate) const CHILD_STRATEGY_REVERSE_ARRAY: u32 = 0;
/// `TrieBuilder.ChildSaveStrategy.ARRAY.code`.
pub(crate) const CHILD_STRATEGY_ARRAY: u32 = 1;
/// `TrieBuilder.ChildSaveStrategy.BITS.code`.
pub(crate) const CHILD_STRATEGY_BITS: u32 = 2;

/// The fewest bytes one `.tmd` per-field record can occupy: nine
/// single-byte-minimum values -- `fieldNumber`, `numTerms`,
/// `sumTotalTermFreq`, `sumDocFreq`, `docCount`, the two length prefixes of
/// `minTerm`/`maxTerm`, and `indexStart`/`rootFP`/`indexEnd` -- less the one
/// `sumDocFreq` that `IndexOptions::Docs` aliases away. Used only as a ceiling
/// on `numFields`, so undercounting is the safe direction.
pub(crate) const MIN_FIELD_RECORD_BYTES: usize = 9;

const BYTES_MINUS_1_MASK: [u64; 8] = [
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    FieldInfos(#[from] crate::field_infos::Error),
    #[error("invalid numFields: {0}")]
    InvalidNumFields(i32),
    #[error("invalid field number: {0}")]
    InvalidFieldNumber(i32),
    #[error("illegal numTerms for field number: {0}")]
    IllegalNumTerms(i32),
    #[error("invalid docCount: {doc_count} maxDoc: {max_doc}")]
    InvalidDocCount { doc_count: i32, max_doc: i32 },
    #[error("invalid sumDocFreq: {sum_doc_freq} docCount: {doc_count}")]
    InvalidSumDocFreq { sum_doc_freq: i64, doc_count: i32 },
    #[error("invalid sumTotalTermFreq: {sum_total_term_freq} sumDocFreq: {sum_doc_freq}")]
    InvalidSumTotalTermFreq {
        sum_total_term_freq: i64,
        sum_doc_freq: i64,
    },
    #[error("duplicate field: {0}")]
    DuplicateField(String),
    #[error(
        "index-time postings BLOCK_SIZE ({found}) != read-time BLOCK_SIZE ({POSTINGS_BLOCK_SIZE})"
    )]
    UnexpectedBlockSize { found: i32 },
    #[error(transparent)]
    Postings(#[from] postings::Error),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
    /// Another terms dictionary a per-field group was routed to failed to
    /// decode, while [`crate::per_field_postings::open_groups`] read it.
    #[error("{format} terms dictionary: {message}")]
    ForeignDictionary {
        format: &'static str,
        message: String,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// A term found by [`FieldTerms::seek_term_state`]: its statistics plus the
/// postings pointers [`FieldTerms::lazy_postings_for`] opens it from --
/// `TermsEnum.termState()`. Only meaningful for the field that produced it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SeekedTerm {
    pub stats: TermStats,
    meta: TermMetadata,
}

impl SeekedTerm {
    /// `IntBlockTermState.singletonDocID`: the one document of a term pulsed
    /// into the dictionary (`docFreq == 1`), read with no `.doc` access;
    /// `None` for any other term.
    pub fn singleton_doc(&self) -> Option<i32> {
        (self.stats.doc_freq == 1 && self.meta.singleton_doc_id >= 0)
            .then_some(self.meta.singleton_doc_id)
    }
}

/// `docFreq`/`totalTermFreq` for one found term — the entirety of what this
/// slice can read back for a term (no postings/doc-ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermStats {
    pub doc_freq: i32,
    pub total_term_freq: i64,
}

/// `TermsEnum.SeekStatus`-equivalent: the outcome of [`TermsEnum::seek_ceil`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekStatus {
    /// The target term itself was present.
    Found,
    /// The target term was absent; the enum is positioned on the smallest
    /// term greater than the target.
    NotFound,
    /// No term in the field is >= the target; the enum is positioned past
    /// the last term (a following [`TermsEnum::next`] returns `None`).
    End,
}

// ---------------------------------------------------------------------------
// Byte-cursor helpers over an in-memory region (Java's `ByteArrayDataInput`).
//
// A frame keeps four such regions (`suffixBytes`/`suffixLengthBytes`/
// `statBytes`/`bytes`) plus a read position into each, exactly like
// `SegmentTermsEnumFrame`. Reading through a `SliceInput` per entry would
// rebuild the reader on every `next()`; these read straight out of the `Vec`
// with the position passed by reference.
// ---------------------------------------------------------------------------

fn eof_err(what: &str) -> Error {
    Error::Store(lucene_store::Error::Corrupted(format!(
        "terms block {what} region read past its end"
    )))
}

// ARITH: `buf.get(*pos)` returned `Some`, so `*pos < buf.len()`; a slice's
// length is at most `isize::MAX`, so `*pos + 1 <= isize::MAX` and cannot
// overflow `usize`.
#[allow(clippy::arithmetic_side_effects)]
fn read_byte_at(buf: &[u8], pos: &mut usize, what: &str) -> Result<u8> {
    let b = *buf.get(*pos).ok_or_else(|| eof_err(what))?;
    *pos += 1;
    Ok(b)
}

/// `DataInput.readVInt` over an in-memory region. Bit-for-bit the same
/// 5-group decoding, including the sign bit the last group can carry.
// ARITH: `shift` starts at 7 and is only incremented inside a loop whose
// guard is `shift <= 28`, so it takes the values 7, 14, 21, 28, 35 and stops.
// Every `<< shift` therefore runs with `shift <= 28 < 32` (legal for `i32`),
// and `shift + 7 <= 35` cannot overflow.
#[allow(clippy::arithmetic_side_effects)]
fn read_vint_at(buf: &[u8], pos: &mut usize, what: &str) -> Result<i32> {
    let mut b = read_byte_at(buf, pos, what)? as i32;
    let mut v = b & 0x7F;
    let mut shift = 7;
    while b & 0x80 != 0 && shift <= 28 {
        b = read_byte_at(buf, pos, what)? as i32;
        v |= (b & 0x7F) << shift;
        shift += 7;
    }
    Ok(v)
}

/// `DataInput.readVLong` over an in-memory region.
// ARITH: `shift` starts at 7 and is only incremented inside a loop whose
// guard is `shift <= 63`, so it takes the values 7, 14, ..., 63, 70 and stops.
// Every `<< shift` therefore runs with `shift <= 63 < 64` (legal for `i64`),
// and `shift + 7 <= 70` cannot overflow.
#[allow(clippy::arithmetic_side_effects)]
fn read_vlong_at(buf: &[u8], pos: &mut usize, what: &str) -> Result<i64> {
    let mut b = read_byte_at(buf, pos, what)? as i64;
    let mut v = b & 0x7F;
    let mut shift = 7;
    while b & 0x80 != 0 && shift <= 63 {
        b = read_byte_at(buf, pos, what)? as i64;
        v |= (b & 0x7F) << shift;
        shift += 7;
    }
    Ok(v)
}

/// Makes `buf` hold at least `len` readable bytes, growing it but never
/// shrinking it -- Java's `ArrayUtil.oversize` reuse, where the *logical*
/// length is carried separately (`numSuffixBytes` and friends) so a block
/// smaller than its predecessor costs nothing. Zero-filling only the growth
/// keeps a re-loaded block from paying to blank bytes it is about to
/// overwrite.
///
/// The allocation is fallible on purpose: the length comes straight off the
/// wire (a block's *decompressed* suffix length), so a corrupt header must
/// produce an error rather than the process abort `vec![0u8; n]` would give
/// -- an abort cannot be caught at the FFI boundary. Java's `new byte[n]`
/// throws `OutOfMemoryError`, which a caller can catch.
fn fit_buf(buf: &mut Vec<u8>, len: usize, what: &str) -> Result<()> {
    if buf.len() < len {
        // ARITH: guarded by `buf.len() < len` one line up.
        #[allow(clippy::arithmetic_side_effects)]
        let extra = len - buf.len();
        buf.try_reserve(extra).map_err(|_| {
            Error::Store(lucene_store::Error::Corrupted(format!(
                "cannot allocate {len} bytes for {what}"
            )))
        })?;
        buf.resize(len, 0);
    }
    Ok(())
}

/// What a dictionary's zstd blocks (`blocktree_writer::BlockZstd`) decode
/// with: the field's trained dictionary, if it has one.
#[derive(Debug, Default)]
pub(crate) struct ZstdBlocks {
    dict: Option<lucene_zstd::DecoderDictionary>,
}

/// A field's `.tim` bytes, and whether (and how) its blocks may be zstd
/// blocks: `None` for a stock dictionary, where the zstd code is corrupt.
#[derive(Debug, Clone, Copy)]
struct Tim<'a> {
    bytes: &'a [u8],
    zstd: Option<&'a ZstdBlocks>,
}

impl std::ops::Deref for Tim<'_> {
    type Target = [u8];
    #[inline]
    fn deref(&self) -> &[u8] {
        self.bytes
    }
}

/// The most bytes per entry a zstd block's suffix-lengths, stats and
/// metadata sections can take together (a sub-block pointer, a term's stats
/// and its postings metadata are a few `vLong`s each), so a corrupt plain
/// length cannot buy a large allocation.
const ZSTD_SECTION_BYTES_PER_ENTRY: usize = 256;

/// `BytesRefBuilder`: a growable byte buffer with a logical length that can
/// be shorter than the allocation, so `setByteAt`/`setLength` behave the way
/// `SegmentTermsEnum.term` does while walking a target term's bytes.
#[derive(Debug, Clone, Default)]
struct TermBuf {
    bytes: Vec<u8>,
    len: usize,
}

impl TermBuf {
    #[inline]
    fn get(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    #[inline]
    fn set_length(&mut self, n: usize) {
        if self.bytes.len() < n {
            self.bytes.resize(n, 0);
        }
        self.len = n;
    }

    /// # Panics
    ///
    /// Never for any `i` this module produces: every call site passes an index
    /// into a caller-supplied `target: &[u8]` (`target_upto < target.len()`),
    /// so `i < isize::MAX`.
    // ARITH: `i` indexes a live `&[u8]`, whose length is at most `isize::MAX`,
    // so `i + 1 <= isize::MAX` cannot overflow `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn set_byte_at(&mut self, i: usize, b: u8) {
        debug_assert!(i < isize::MAX as usize);
        if self.bytes.len() <= i {
            self.bytes.resize(i + 1, 0);
        }
        self.bytes[i] = b;
    }

    fn copy_from(&mut self, src: &[u8]) {
        self.bytes.clear();
        self.bytes.extend_from_slice(src);
        self.len = src.len();
    }

    fn clear(&mut self) {
        self.len = 0;
    }
}

/// One entry of [`SegmentTermsEnum`]'s frame stack — the port of
/// `SegmentTermsEnumFrame`. Holds one `.tim` block's four decoded regions and
/// the cursors into them; a stack of these (one per trie node along the
/// current term's path that carried an output block) is the *entire*
/// in-memory footprint of a term lookup.
#[derive(Debug, Clone, Default)]
struct Frame {
    /// Index in the stack (`SegmentTermsEnumFrame.ord`).
    ord: usize,
    has_terms: bool,
    has_terms_orig: bool,
    is_floor: bool,

    fp: usize,
    fp_orig: usize,
    fp_end: usize,

    /// Position in the field's `.tip` slice of this node's floor data
    /// (`numFollowFloorBlocks`), for `rewind` -- or in [`Self::floor_buf`]
    /// when [`Self::floor_owned`].
    rewind_pos: usize,
    /// Position in the field's `.tip` slice of the next unread floor record
    /// (in [`Self::floor_buf`] when [`Self::floor_owned`]).
    floor_data_pos: usize,
    /// The frame's own copy of its index output, for a frame pushed from a
    /// retired FST index: there the floor data is the tail of an output the
    /// arcs along the path assemble (`SegmentTermsEnum.pushFrame(arc,
    /// frameData, length)`'s `frameData`), not a region of `.tip`.
    /// A high-water-mark buffer like the block regions below.
    floor_buf: Vec<u8>,
    /// Floor data is read from [`Self::floor_buf`] rather than the field's
    /// `.tip` slice.
    floor_owned: bool,
    num_follow_floor_blocks: i32,
    /// `256` once the last floor block has been selected, so no target label
    /// can be `>=` it (Java uses the same sentinel).
    next_floor_label: u32,

    prefix_length: usize,
    /// Entries in the currently-loaded block.
    ///
    /// **Invariant (`ENT_COUNT`), established by [`Frame::load_block`] and
    /// relied on by every `// ARITH:` proof below:** `1 <= ent_count <=
    /// i32::MAX`. The upper half is structural -- `ent_count` is
    /// `(code as u32) >> 1` for a `code: i32`, so it cannot exceed
    /// `u32::MAX >> 1 == i32::MAX` -- and the lower half plus the tighter
    /// `ent_count <= suffix_length_bytes_len` bound are checked explicitly in
    /// `load_block`. Consequently `ent_count as i32 >= 1` is never negative,
    /// and `next_ent`, which every scan keeps in `0..=ent_count`, can always
    /// be incremented without overflowing `i32`.
    ent_count: u32,
    /// Which entry is read next, or `-1` when the block is not loaded.
    next_ent: i32,
    is_last_in_floor: bool,
    is_leaf_block: bool,
    all_equal: bool,
    last_sub_fp: i64,

    /// The four per-block regions. Each `Vec` is a high-water-mark buffer
    /// reused across `load_block` calls; the bytes that belong to the block
    /// currently loaded are `[..*_len]` (Java's `byte[]` + `numBytes` pair).
    suffix_bytes: Vec<u8>,
    suffix_bytes_len: usize,
    suffixes_pos: usize,
    suffix_length_bytes: Vec<u8>,
    suffix_length_bytes_len: usize,
    suffix_lengths_pos: usize,
    stat_bytes: Vec<u8>,
    stat_bytes_len: usize,
    stats_pos: usize,
    meta_bytes: Vec<u8>,
    meta_bytes_len: usize,
    meta_pos: usize,

    stats_singleton_run_length: u32,
    meta_data_upto: u32,
    /// A zstd block's plain bytes, reused across loads.
    zstd_plain: Vec<u8>,

    // `BlockTermState`.
    doc_freq: i32,
    total_term_freq: i64,
    term_block_ord: u32,
    meta: TermMetadata,

    start_byte_pos: usize,
    suffix_length: usize,
    sub_code: u64,
}

impl Frame {
    /// `SegmentTermsEnumFrame.getTermBlockOrd()`.
    #[inline]
    fn term_block_ord(&self) -> u32 {
        if self.is_leaf_block {
            self.next_ent.max(0) as u32
        } else {
            self.term_block_ord
        }
    }

    /// `SegmentTermsEnumFrame.setFloorData`: the floor record layout is
    /// `numFollowFloorBlocks: vint`, then `numFollowFloorBlocks` times
    /// `(floorLeadByte: byte, code: vlong)` where
    /// `code = (subFP - fpOrig) << 1 | hasTerms`. The *first* label is read
    /// here and each following one at the end of a `scan_to_floor_frame`
    /// step, exactly as Java splits it.
    fn set_floor_data(&mut self, index: &[u8], floor_data_fp: usize) -> Result<()> {
        if self.floor_owned {
            let buf = std::mem::take(&mut self.floor_buf);
            let r = self.set_floor_data_in(&buf, floor_data_fp);
            self.floor_buf = buf;
            return r;
        }
        self.set_floor_data_in(index, floor_data_fp)
    }

    /// [`Self::set_floor_data`] over the bytes that hold this frame's floor
    /// data.
    fn set_floor_data_in(&mut self, index: &[u8], floor_data_fp: usize) -> Result<()> {
        self.rewind_pos = floor_data_fp;
        let mut pos = floor_data_fp;
        self.num_follow_floor_blocks = read_vint_at(index, &mut pos, "floor data")?;
        if self.num_follow_floor_blocks <= 0 {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "invalid numFollowFloorBlocks: {}",
                self.num_follow_floor_blocks
            ))));
        }
        self.next_floor_label = read_byte_at(index, &mut pos, "floor data")? as u32;
        self.floor_data_pos = pos;
        Ok(())
    }

    /// `SegmentTermsEnumFrame.rewind`: back to this frame's *first* block and
    /// its first entry.
    ///
    /// Java unconditionally sets `nextEnt = -1`, which forces the block to be
    /// re-read from `.tim` even when the frame already holds exactly that
    /// block. When the frame is already parked on `fpOrig` this port instead
    /// resets the four region cursors in place, which lands in precisely the
    /// state [`Frame::load_block`] would produce (see the assignments at the
    /// end of that method) without re-decoding anything -- the optimization
    /// Lucene left commented out in `rewind()`'s body. It matters because
    /// this port pools one enum per field for `seek_exact`, so consecutive
    /// lookups that share a block would otherwise reload it every time.
    ///
    /// The reset restores every field `load_block`'s tail sets, with one
    /// exception: `is_last_in_floor`, which `scan_to_floor_frame` can only
    /// have changed by also moving `fp` off `fp_orig` -- and that takes the
    /// reload branch instead. The single input that could break the
    /// equivalence is a floor record encoding a zero delta (`code >> 1 == 0`),
    /// which no writer emits and which would make a floor block its own
    /// successor.
    fn rewind(&mut self, index: &[u8]) -> Result<()> {
        if self.next_ent != -1 && self.fp == self.fp_orig {
            self.reset_cursors();
        } else {
            self.fp = self.fp_orig;
            self.next_ent = -1;
        }
        self.has_terms = self.has_terms_orig;
        if self.is_floor {
            let rewind_pos = self.rewind_pos;
            self.set_floor_data(index, rewind_pos)?;
        }
        Ok(())
    }

    /// The subset of [`Frame::load_block`]'s tail that depends only on the
    /// already-decoded regions -- see [`Frame::rewind`].
    fn reset_cursors(&mut self) {
        self.suffixes_pos = 0;
        self.suffix_lengths_pos = 0;
        self.stats_pos = 0;
        self.meta_pos = 0;
        self.stats_singleton_run_length = 0;
        self.meta_data_upto = 0;
        self.term_block_ord = 0;
        self.meta = TermMetadata::EMPTY;
        self.next_ent = 0;
        self.last_sub_fp = -1;
    }

    /// `SegmentTermsEnumFrame.loadBlock`: decodes the block header and copies
    /// (decompressing where needed) the four per-block regions. Per-term
    /// stats and postings metadata are *not* decoded here -- that is
    /// [`Frame::decode_meta_data`]'s job, run only up to the term actually
    /// landed on.
    fn load_block(&mut self, tim: Tim<'_>) -> Result<()> {
        if self.next_ent != -1 {
            // Already loaded.
            return Ok(());
        }
        let mut r = SliceInput::new(tim.bytes);
        r.seek(self.fp)?;

        let code = r.read_vint()?;
        self.ent_count = (code as u32) >> 1;
        if self.ent_count == 0 {
            // Java `assert entCount > 0`, disabled in production; an empty
            // block would otherwise decode as a silently empty term range.
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "empty terms block".into(),
            )));
        }
        self.is_last_in_floor = (code & 1) != 0;

        let code_l = r.read_vlong()? as u64;
        self.is_leaf_block = (code_l & 0x04) != 0;
        let num_suffix_bytes = (code_l >> 3) as usize;
        let compression_alg = code_l & 0x03;
        if compression_alg == crate::blocktree_writer::COMPRESSION_ZSTD {
            if let Some(zstd) = tim.zstd {
                return self.load_zstd_block(&mut r, zstd, num_suffix_bytes);
            }
        }
        // `numSuffixBytes` is the *decompressed* length, so only the
        // uncompressed case can be bounds-checked against what is left.
        if compression_alg == 0 && num_suffix_bytes > r.remaining() {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "terms block suffix length {num_suffix_bytes} exceeds {} remaining bytes",
                r.remaining()
            ))));
        }
        fit_buf(
            &mut self.suffix_bytes,
            num_suffix_bytes,
            "terms block suffix bytes",
        )?;
        self.suffix_bytes_len = num_suffix_bytes;
        let suffixes = &mut self.suffix_bytes[..num_suffix_bytes];
        match compression_alg {
            // `CompressionAlgorithm.NO_COMPRESSION.read`.
            0 => r.read_bytes(suffixes)?,
            // `CompressionAlgorithm.LOWERCASE_ASCII.read`.
            1 => decompress_lowercase_ascii(&mut r, suffixes)?,
            // `CompressionAlgorithm.LZ4.read`.
            2 => {
                crate::lz4::decompress_slice(&mut r, num_suffix_bytes, suffixes, 0)?;
            }
            _ => {
                // `code_l & 0x03` is masked to 2 bits, so `3` is the only
                // remaining value, a zstd block, which only a zstd
                // dictionary may hold; `CompressionAlgorithm.byCode` throws
                // `IllegalArgumentException` for it too.
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "illegal compression algorithm code (3) for a terms block".into(),
                )));
            }
        }
        self.suffixes_pos = 0;
        self.read_sections(&mut r)?;
        // Sub-blocks of one floor block are written back to back, so the next
        // floor block starts exactly here.
        self.fp_end = r.position();
        Ok(())
    }

    /// The rest of a zstd block after its token: `vInt(plainLength)`,
    /// `vInt(frameLength)` and a bare zstd frame holding the suffix bytes
    /// and then the three sections [`Self::read_sections`] reads.
    // ARITH: `ent_count <= i32::MAX` (`ENT_COUNT`) times a small constant,
    // plus a suffix length already bounded by the plain length it is
    // compared with; `position + frame_len` is checked by `read_region_len`.
    #[allow(clippy::arithmetic_side_effects)]
    fn load_zstd_block(
        &mut self,
        r: &mut SliceInput<'_>,
        zstd: &ZstdBlocks,
        num_suffix_bytes: usize,
    ) -> Result<()> {
        let corrupt = |m: String| Error::Store(lucene_store::Error::Corrupted(m));
        let plain_len = r.read_vint()?;
        let frame_len = read_region_len(r, "zstd terms block")?;
        let max_plain = (self.ent_count as usize)
            .saturating_mul(ZSTD_SECTION_BYTES_PER_ENTRY)
            .saturating_add(num_suffix_bytes);
        if plain_len < 0
            || (plain_len as usize) < num_suffix_bytes
            || plain_len as usize > max_plain
        {
            return Err(corrupt(format!(
                "zstd terms block of {plain_len} bytes for {num_suffix_bytes} suffix bytes and {} entries",
                self.ent_count
            )));
        }
        let plain_len = plain_len as usize;
        let frame = &r.as_slice()[..frame_len];
        r.seek(r.position() + frame_len)?;
        let mut plain = std::mem::take(&mut self.zstd_plain);
        plain.clear();
        lucene_zstd::with_bare_decompressor(|d| {
            d.decompress(frame, zstd.dict.as_ref(), plain_len, &mut plain)
        })
        .map_err(|e| corrupt(format!("zstd terms block: {e}")))?;
        fit_buf(
            &mut self.suffix_bytes,
            num_suffix_bytes,
            "terms block suffix bytes",
        )?;
        self.suffix_bytes_len = num_suffix_bytes;
        self.suffix_bytes[..num_suffix_bytes].copy_from_slice(&plain[..num_suffix_bytes]);
        self.suffixes_pos = 0;
        let mut sections = SliceInput::new(&plain[num_suffix_bytes..]);
        let read = self.read_sections(&mut sections);
        let left = sections.as_slice().len();
        self.zstd_plain = plain;
        read?;
        if left != 0 {
            return Err(corrupt(format!(
                "{left} bytes after a zstd terms block's sections"
            )));
        }
        self.fp_end = r.position();
        Ok(())
    }

    /// A block's suffix-lengths, stats and metadata sections, which follow
    /// its suffix bytes.
    fn read_sections(&mut self, r: &mut SliceInput<'_>) -> Result<()> {
        let raw = r.read_vint()? as u32;
        self.all_equal = (raw & 1) != 0;
        let num_suffix_length_bytes = (raw >> 1) as usize;
        // `allEqual` replicates a single byte, so only the non-replicated
        // case has to fit in what is left of the file.
        if !self.all_equal && num_suffix_length_bytes > r.remaining() {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "terms block suffix-lengths length {num_suffix_length_bytes} exceeds {} remaining bytes",
                r.remaining()
            ))));
        }
        // `Lucene103BlockTreeTermsWriter.writeBlock` pushes exactly one vint
        // into `suffixLengthsWriter` per entry (a term's `suffix`, or a
        // sub-block's `(suffix << 1) | 1`, plus a further vlong for the
        // sub-block delta), and a vint is never shorter than one byte -- so a
        // well-formed block always has `numSuffixLengthBytes >= entCount`, in
        // the `allEqual` case too (`allEqual` replicates the blob's *bytes*,
        // it does not shorten the blob). Checking it here, once per block load
        // rather than once per entry, is what bounds `ent_count` by a real
        // buffer length for every scan below -- see `binary_search_term_leaf`,
        // whose bisection indices are `entCount`-derived.
        if self.ent_count as usize > num_suffix_length_bytes {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "terms block entCount {} exceeds its {num_suffix_length_bytes}-byte \
                 suffix-lengths region",
                self.ent_count
            ))));
        }
        fit_buf(
            &mut self.suffix_length_bytes,
            num_suffix_length_bytes,
            "terms block suffix lengths",
        )?;
        self.suffix_length_bytes_len = num_suffix_length_bytes;
        let suffix_lengths = &mut self.suffix_length_bytes[..num_suffix_length_bytes];
        if self.all_equal {
            let b = r.read_byte()?;
            suffix_lengths.fill(b);
        } else {
            r.read_bytes(suffix_lengths)?;
        }
        self.suffix_lengths_pos = 0;

        let num_stat_bytes = read_region_len(r, "terms block stats")?;
        fit_buf(&mut self.stat_bytes, num_stat_bytes, "terms block stats")?;
        self.stat_bytes_len = num_stat_bytes;
        r.read_bytes(&mut self.stat_bytes[..num_stat_bytes])?;
        self.stats_pos = 0;

        self.stats_singleton_run_length = 0;
        self.meta_data_upto = 0;
        self.meta = TermMetadata::EMPTY;
        self.term_block_ord = 0;
        self.next_ent = 0;
        self.last_sub_fp = -1;

        let num_meta_bytes = read_region_len(r, "terms block metadata")?;
        fit_buf(&mut self.meta_bytes, num_meta_bytes, "terms block metadata")?;
        self.meta_bytes_len = num_meta_bytes;
        r.read_bytes(&mut self.meta_bytes[..num_meta_bytes])?;
        self.meta_pos = 0;
        Ok(())
    }

    /// `SegmentTermsEnumFrame.loadNextFloorBlock`.
    fn load_next_floor_block(&mut self, tim: Tim<'_>) -> Result<()> {
        self.fp = self.fp_end;
        self.next_ent = -1;
        self.load_block(tim)
    }

    /// `SegmentTermsEnumFrame.scanToFloorFrame`: picks the one floor
    /// sub-block whose lead-byte range covers `target`, without reading any
    /// of the others' blocks.
    fn scan_to_floor_frame(&mut self, index: &[u8], target: &[u8]) -> Result<()> {
        if !self.is_floor || target.len() <= self.prefix_length {
            return Ok(());
        }
        self.scan_to_floor_label(index, target[self.prefix_length] as u32)
    }

    /// [`Self::scan_to_floor_frame`] for a lead byte on its own: moves to the
    /// floor block whose label range holds `target_label`, reading only floor
    /// records (forward from where the last scan stopped).
    fn scan_to_floor_label(&mut self, index: &[u8], target_label: u32) -> Result<()> {
        if !self.is_floor {
            return Ok(());
        }
        if target_label < self.next_floor_label {
            // Already on the correct block.
            return Ok(());
        }
        if self.floor_owned {
            let buf = std::mem::take(&mut self.floor_buf);
            let r = self.scan_floor_records(&buf, target_label);
            self.floor_buf = buf;
            return r;
        }
        self.scan_floor_records(index, target_label)
    }

    /// [`Self::scan_to_floor_label`]'s walk over the floor records, in the
    /// bytes that hold them.
    fn scan_floor_records(&mut self, index: &[u8], target_label: u32) -> Result<()> {
        let mut new_fp;
        let mut pos = self.floor_data_pos;
        loop {
            if self.num_follow_floor_blocks <= 0 {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "floor block list exhausted before reaching the target label".into(),
                )));
            }
            let code = read_vlong_at(index, &mut pos, "floor data")? as u64;
            // Java's `newFP = fpOrig + (code >>> 1)`. The delta is at most
            // `2^63 - 1` (a vlong) and `fp_orig` at most `isize::MAX`, so on a
            // 64-bit target the sum provably cannot wrap -- but it was written
            // as a `wrapping_add`, which reads as if wrapping were *intended*,
            // and a wrap here would land back inside the `.tim` at an offset
            // that decodes as a perfectly valid but different block: a silent
            // wrong answer rather than an error. Making it checked costs one
            // branch per floor step and states the fact instead of assuming
            // it. The resulting fp is bounds-checked by `load_block`'s seek.
            new_fp = usize::try_from(code >> 1)
                .ok()
                .and_then(|delta| self.fp_orig.checked_add(delta))
                .ok_or_else(|| {
                    Error::Store(lucene_store::Error::Corrupted(format!(
                        "floor block delta {} overflows the parent fp {}",
                        code >> 1,
                        self.fp_orig
                    )))
                })?;
            self.has_terms = (code & 1) != 0;

            self.is_last_in_floor = self.num_follow_floor_blocks == 1;
            // ARITH: the loop head returns `Corrupted` unless
            // `num_follow_floor_blocks >= 1`, so this cannot underflow.
            #[allow(clippy::arithmetic_side_effects)]
            {
                self.num_follow_floor_blocks -= 1;
            }

            if self.is_last_in_floor {
                self.next_floor_label = 256;
                break;
            }
            self.next_floor_label = read_byte_at(index, &mut pos, "floor data")? as u32;
            if target_label < self.next_floor_label {
                break;
            }
        }
        self.floor_data_pos = pos;
        if new_fp != self.fp {
            // Force a reload of the block we just switched to.
            self.next_ent = -1;
            self.fp = new_fp;
        }
        Ok(())
    }

    /// The suffix bytes of the entry the cursor last read.
    // ARITH: `start_byte_pos + suffix_length` is written by exactly two
    // places, and both bound it by `suffix_bytes_len <= suffix_bytes.len()`:
    // `take_suffix` (a `checked_add` plus an explicit `end > suffix_bytes_len`
    // rejection) and `binary_search_term_leaf` (whose
    // `suffix_length * ent_count <= suffix_bytes_len` precondition is checked
    // once, before the bisection, and whose indices never exceed
    // `ent_count - 1`). A slice length is at most `isize::MAX`, so the sum
    // cannot overflow `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn suffix(&self) -> &[u8] {
        debug_assert!(self.start_byte_pos + self.suffix_length <= self.suffix_bytes.len());
        &self.suffix_bytes[self.start_byte_pos..self.start_byte_pos + self.suffix_length]
    }

    /// Reads the next entry's suffix length and advances the suffix cursor
    /// past its bytes, bounds-checking both regions.
    fn take_suffix(&mut self, suffix_length: usize) -> Result<()> {
        self.suffix_length = suffix_length;
        self.start_byte_pos = self.suffixes_pos;
        let end = self
            .suffixes_pos
            .checked_add(suffix_length)
            .ok_or_else(|| eof_err("suffix"))?;
        if end > self.suffix_bytes_len {
            return Err(eof_err("suffix"));
        }
        self.suffixes_pos = end;
        Ok(())
    }

    /// `SegmentTermsEnumFrame.fillTerm`.
    // ARITH: `prefix_length` is the length of a byte prefix already
    // materialized in `term` and `suffix_length` is bounded by
    // `suffix_bytes_len <= suffix_bytes.len()` (see [`Frame::suffix`]). Both
    // are lengths of live allocations, hence each at most `isize::MAX`, so
    // their sum is at most `2 * isize::MAX = usize::MAX - 1`.
    #[allow(clippy::arithmetic_side_effects)]
    fn fill_term(&self, term: &mut TermBuf) {
        let total = self.prefix_length + self.suffix_length;
        term.set_length(total);
        term.bytes[self.prefix_length..total].copy_from_slice(self.suffix());
    }

    /// `SegmentTermsEnumFrame.nextLeaf`.
    // ARITH: the guard immediately above the increment leaves
    // `0 <= next_ent < ent_count as i32`, and `ent_count <= i32::MAX` by the
    // `ENT_COUNT` invariant, so `next_ent + 1 <= i32::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn next_leaf(&mut self, term: &mut TermBuf) -> Result<()> {
        if self.next_ent < 0 || self.next_ent >= self.ent_count as i32 {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "terms block entry cursor ran past entCount".into(),
            )));
        }
        self.next_ent += 1;
        let len = read_vint_at(
            &self.suffix_length_bytes[..self.suffix_length_bytes_len],
            &mut self.suffix_lengths_pos,
            "suffix lengths",
        )? as usize;
        self.take_suffix(len)?;
        self.fill_term(term);
        Ok(())
    }

    /// `SegmentTermsEnumFrame.nextNonLeaf`: returns `true` when the entry it
    /// landed on is a sub-block pointer rather than a term.
    // ARITH: the two guards at the head of the loop body leave
    // `0 <= next_ent < ent_count as i32 <= i32::MAX` (`ENT_COUNT`), so
    // `next_ent + 1` cannot overflow. `term_block_ord` counts *terms* within
    // the same block, so it is bounded by `ent_count <= i32::MAX < u32::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_non_leaf(&mut self, tim: Tim<'_>, term: &mut TermBuf) -> Result<bool> {
        loop {
            if self.next_ent == self.ent_count as i32 {
                self.load_next_floor_block(tim)?;
                if self.is_leaf_block {
                    self.next_leaf(term)?;
                    return Ok(false);
                }
                continue;
            }
            if self.next_ent < 0 || self.next_ent > self.ent_count as i32 {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "terms block entry cursor ran past entCount".into(),
                )));
            }
            self.next_ent += 1;
            let code = read_vint_at(
                &self.suffix_length_bytes[..self.suffix_length_bytes_len],
                &mut self.suffix_lengths_pos,
                "suffix lengths",
            )? as u32;
            self.take_suffix((code >> 1) as usize)?;
            self.fill_term(term);
            if (code & 1) == 0 {
                // A normal term.
                self.sub_code = 0;
                self.term_block_ord += 1;
                return Ok(false);
            }
            // A sub-block; make its fp absolute.
            self.sub_code = read_vlong_at(
                &self.suffix_length_bytes[..self.suffix_length_bytes_len],
                &mut self.suffix_lengths_pos,
                "suffix lengths",
            )? as u64;
            self.last_sub_fp = self.absolute_sub_fp()?;
            return Ok(true);
        }
    }

    /// `fp - subCode`, rejecting a delta that would point at or past this
    /// block (a corrupt chain that would otherwise recurse forever).
    // ARITH: the guard rejects `sub_code as usize > self.fp`, so the
    // subtraction runs only for `sub_code as usize <= self.fp` and cannot
    // underflow.
    #[allow(clippy::arithmetic_side_effects)]
    fn absolute_sub_fp(&self) -> Result<i64> {
        if self.sub_code == 0 || self.sub_code as usize > self.fp {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "terms block sub-block delta fp exceeds parent fp".into(),
            )));
        }
        Ok((self.fp - self.sub_code as usize) as i64)
    }

    /// The next entry of the loaded block, leaving the term buffer alone --
    /// what `IntersectTermsEnumFrame` reads: the suffix is only looked at
    /// through [`Self::suffix`] until the automaton says the entry matters.
    /// `true` for a sub-block pointer (its fp in `last_sub_fp`). Unlike
    /// [`Self::next`] it never moves on to the next floor block; the caller
    /// sees `next_ent == ent_count` and decides.
    // ARITH: the guard leaves `0 <= next_ent < ent_count <= i32::MAX`
    // (`ENT_COUNT`), so `next_ent + 1` cannot overflow; `term_block_ord`
    // counts terms within this block and is bounded by `ent_count`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    fn next_entry(&mut self) -> Result<bool> {
        if self.next_ent < 0 || self.next_ent >= self.ent_count as i32 {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "terms block entry cursor ran past entCount".into(),
            )));
        }
        self.next_ent += 1;
        let code = read_vint_at(
            &self.suffix_length_bytes[..self.suffix_length_bytes_len],
            &mut self.suffix_lengths_pos,
            "suffix lengths",
        )? as u32;
        if self.is_leaf_block {
            self.take_suffix(code as usize)?;
            return Ok(false);
        }
        self.take_suffix((code >> 1) as usize)?;
        if (code & 1) == 0 {
            self.sub_code = 0;
            self.term_block_ord += 1;
            return Ok(false);
        }
        self.sub_code = read_vlong_at(
            &self.suffix_length_bytes[..self.suffix_length_bytes_len],
            &mut self.suffix_lengths_pos,
            "suffix lengths",
        )? as u64;
        self.last_sub_fp = self.absolute_sub_fp()?;
        Ok(true)
    }

    /// `SegmentTermsEnumFrame.next`.
    fn next(&mut self, tim: Tim<'_>, term: &mut TermBuf) -> Result<bool> {
        if self.is_leaf_block {
            self.next_leaf(term)?;
            Ok(false)
        } else {
            self.next_non_leaf(tim, term)
        }
    }

    /// `SegmentTermsEnumFrame.scanToSubBlock`: re-positions a parent frame on
    /// the sub-block entry a `next()` popped back out of.
    // ARITH: `self.fp - sub_fp as usize` is guarded by the
    // `sub_fp < 0 || sub_fp as usize >= self.fp` rejection just above it.
    // `next_ent + 1` runs only after the loop head rejected
    // `next_ent >= ent_count as i32`, so `next_ent < ent_count <= i32::MAX`
    // (`ENT_COUNT`). `term_block_ord` counts terms within one block and so is
    // bounded by `ent_count`.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_sub_block(&mut self, sub_fp: i64) -> Result<()> {
        if self.last_sub_fp == sub_fp {
            return Ok(());
        }
        if sub_fp < 0 || sub_fp as usize >= self.fp {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "sub-block fp is not below its parent".into(),
            )));
        }
        let target_sub_code = (self.fp - sub_fp as usize) as u64;
        loop {
            if self.next_ent >= self.ent_count as i32 {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "sub-block pointer not found in its parent block".into(),
                )));
            }
            self.next_ent += 1;
            let code = read_vint_at(
                &self.suffix_length_bytes[..self.suffix_length_bytes_len],
                &mut self.suffix_lengths_pos,
                "suffix lengths",
            )? as u32;
            self.take_suffix((code >> 1) as usize)?;
            if (code & 1) != 0 {
                let sub_code = read_vlong_at(
                    &self.suffix_length_bytes[..self.suffix_length_bytes_len],
                    &mut self.suffix_lengths_pos,
                    "suffix lengths",
                )? as u64;
                if sub_code == target_sub_code {
                    self.last_sub_fp = sub_fp;
                    return Ok(());
                }
            } else {
                self.term_block_ord += 1;
            }
        }
    }

    /// `SegmentTermsEnumFrame.scanToTerm`. The `bool` in the result is Java's
    /// inline "recurse into the sub-frame(s)" step, which needs the enum's
    /// frame stack and so is done by the caller.
    fn scan_to_term(
        &mut self,
        term: &mut TermBuf,
        target: &[u8],
        exact_only: bool,
        term_exists: &mut bool,
    ) -> Result<(SeekStatus, bool)> {
        if self.is_leaf_block {
            if self.all_equal {
                Ok((
                    self.binary_search_term_leaf(term, target, exact_only, term_exists)?,
                    false,
                ))
            } else {
                Ok((
                    self.scan_to_term_leaf(term, target, exact_only, term_exists)?,
                    false,
                ))
            }
        } else {
            self.scan_to_term_non_leaf(term, target, exact_only, term_exists)
        }
    }

    /// The target's suffix, i.e. everything past this block's shared prefix.
    #[inline]
    fn target_suffix<'t>(&self, target: &'t [u8]) -> &'t [u8] {
        target.get(self.prefix_length..).unwrap_or(&[])
    }

    /// `SegmentTermsEnumFrame.scanToTermLeaf`.
    // ARITH: the entry guard is `next_ent >= ent_count as i32` (Java writes
    // `==`; `>=` is the same test for every reachable state, since every
    // writer of `next_ent` keeps it in `0..=ent_count`, and it is what makes
    // the loop's increment provably safe rather than merely unreachable). So
    // the loop body runs only with `next_ent < ent_count <= i32::MAX`
    // (`ENT_COUNT`), and it re-establishes that by breaking as soon as
    // `next_ent >= ent_count`.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_term_leaf(
        &mut self,
        term: &mut TermBuf,
        target: &[u8],
        exact_only: bool,
        term_exists: &mut bool,
    ) -> Result<SeekStatus> {
        *term_exists = true;
        self.sub_code = 0;
        if self.next_ent >= self.ent_count as i32 {
            if exact_only {
                self.fill_term(term);
            }
            return Ok(SeekStatus::End);
        }
        loop {
            self.next_ent += 1;
            let len = read_vint_at(
                &self.suffix_length_bytes[..self.suffix_length_bytes_len],
                &mut self.suffix_lengths_pos,
                "suffix lengths",
            )? as usize;
            self.take_suffix(len)?;
            match self.suffix().cmp(self.target_suffix(target)) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Greater => {
                    self.fill_term(term);
                    return Ok(SeekStatus::NotFound);
                }
                std::cmp::Ordering::Equal => {
                    self.fill_term(term);
                    return Ok(SeekStatus::Found);
                }
            }
            if self.next_ent >= self.ent_count as i32 {
                break;
            }
        }
        if exact_only {
            self.fill_term(term);
        }
        Ok(SeekStatus::End)
    }

    /// `SegmentTermsEnumFrame.binarySearchTermLeaf`: the `allEqual` fast path,
    /// where every suffix in the block has the same length so entry `i`'s
    /// bytes start at `i * suffixLength` and the scan becomes a bisection.
    // ARITH: in the order the operations appear --
    //
    // * `ent_count as i32 - 1 >= 0` because `ent_count >= 1` (`ENT_COUNT`).
    // * `start` is `next_ent`, which the entry guard pins to
    //   `0 <= next_ent < ent_count`, and `end` is `ent_count - 1`; both are
    //   therefore in `0..=i32::MAX - 1`, so `start as u32 + end as u32` is at
    //   most `2 * (i32::MAX - 1) < u32::MAX` and cannot overflow. This is the
    //   port of Java's `(start + end) >>> 1`: the *unsigned* shift is what
    //   keeps the midpoint correct once the sum passes `i32::MAX`.
    // * `mid` stays in `start..=end`, so `mid + 1 <= ent_count <= i32::MAX`
    //   and `mid - 1 >= -1`.
    // * every `start_byte_pos`/`suffixes_pos` expression is at most
    //   `ent_count * suffix_length`, which the `checked_mul` guard above the
    //   bisection has already bounded by `suffix_bytes_len`. For the
    //   `start_byte_pos += suffix_length` step specifically: `cmp == Less` on
    //   the final iteration means `start` became `mid + 1 > end`, so
    //   `mid == end`, and that branch also requires `end < ent_count - 1`,
    //   giving `mid <= ent_count - 2` and `(mid + 2) * suffix_length <=
    //   suffix_bytes_len`.
    #[allow(clippy::arithmetic_side_effects)]
    fn binary_search_term_leaf(
        &mut self,
        term: &mut TermBuf,
        target: &[u8],
        exact_only: bool,
        term_exists: &mut bool,
    ) -> Result<SeekStatus> {
        *term_exists = true;
        self.sub_code = 0;
        if self.next_ent < 0 || self.next_ent >= self.ent_count as i32 {
            if exact_only {
                self.fill_term(term);
            }
            return Ok(SeekStatus::End);
        }
        self.suffix_length = read_vint_at(
            &self.suffix_length_bytes[..self.suffix_length_bytes_len],
            &mut self.suffix_lengths_pos,
            "suffix lengths",
        )? as usize;
        if self
            .suffix_length
            .checked_mul(self.ent_count as usize)
            .is_none_or(|n| n > self.suffix_bytes_len)
        {
            return Err(eof_err("suffix"));
        }
        let mut start = self.next_ent;
        let mut end = self.ent_count as i32 - 1;
        let mut cmp = std::cmp::Ordering::Equal;
        while start <= end {
            debug_assert!(start >= 0 && end >= 0);
            // `(start + end) >>> 1` in Java: the sum is formed in `u32` so the
            // midpoint stays correct (and the shift stays logical) even when
            // it exceeds `i32::MAX`.
            let mid = ((start as u32 + end as u32) >> 1) as i32;
            self.next_ent = mid + 1;
            self.start_byte_pos = mid as usize * self.suffix_length;
            cmp = self.suffix().cmp(self.target_suffix(target));
            match cmp {
                std::cmp::Ordering::Less => start = mid + 1,
                std::cmp::Ordering::Greater => end = mid - 1,
                std::cmp::Ordering::Equal => {
                    self.suffixes_pos = self.start_byte_pos + self.suffix_length;
                    self.fill_term(term);
                    return Ok(SeekStatus::Found);
                }
            }
        }
        if end < self.ent_count as i32 - 1 {
            // The bisection ended on a smaller term and a greater one exists:
            // advance onto it.
            if cmp == std::cmp::Ordering::Less {
                self.start_byte_pos += self.suffix_length;
                self.next_ent += 1;
            }
            self.suffixes_pos = self.start_byte_pos + self.suffix_length;
            self.fill_term(term);
            Ok(SeekStatus::NotFound)
        } else {
            self.suffixes_pos = self.start_byte_pos + self.suffix_length;
            if exact_only {
                self.fill_term(term);
            }
            Ok(SeekStatus::End)
        }
    }

    /// `SegmentTermsEnumFrame.scanToTermNonLeaf`.
    // ARITH: the `while next_ent < ent_count as i32` head means the increment
    // runs only with `next_ent < ent_count <= i32::MAX` (`ENT_COUNT`).
    // `term_block_ord` counts terms within this one block, so it is bounded by
    // `ent_count` too.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_term_non_leaf(
        &mut self,
        term: &mut TermBuf,
        target: &[u8],
        exact_only: bool,
        term_exists: &mut bool,
    ) -> Result<(SeekStatus, bool)> {
        if self.next_ent == self.ent_count as i32 {
            if exact_only {
                self.fill_term(term);
                *term_exists = self.sub_code == 0;
            }
            return Ok((SeekStatus::End, false));
        }
        while self.next_ent < self.ent_count as i32 {
            self.next_ent += 1;
            let code = read_vint_at(
                &self.suffix_length_bytes[..self.suffix_length_bytes_len],
                &mut self.suffix_lengths_pos,
                "suffix lengths",
            )? as u32;
            self.take_suffix((code >> 1) as usize)?;
            *term_exists = (code & 1) == 0;
            if *term_exists {
                self.term_block_ord += 1;
                self.sub_code = 0;
            } else {
                self.sub_code = read_vlong_at(
                    &self.suffix_length_bytes[..self.suffix_length_bytes_len],
                    &mut self.suffix_lengths_pos,
                    "suffix lengths",
                )? as u64;
                self.last_sub_fp = self.absolute_sub_fp()?;
            }
            match self.suffix().cmp(self.target_suffix(target)) {
                std::cmp::Ordering::Less => {}
                std::cmp::Ordering::Greater => {
                    self.fill_term(term);
                    // Positioned on a sub-block whose terms are all after the
                    // target: a ceiling seek has to descend into it to reach
                    // the actual next term.
                    return Ok((SeekStatus::NotFound, !exact_only && !*term_exists));
                }
                std::cmp::Ordering::Equal => {
                    // Cannot be a sub-block: the index would have routed the
                    // seek into that sub-block from the start.
                    self.fill_term(term);
                    return Ok((SeekStatus::Found, false));
                }
            }
        }
        if exact_only {
            self.fill_term(term);
        }
        Ok((SeekStatus::End, false))
    }

    /// `SegmentTermsEnumFrame.decodeMetaData`: catches the per-term stats and
    /// postings-metadata streams up to the term the cursor is on, and no
    /// further. A terms-only consumer never triggers it.
    // ARITH: `stats_singleton_run_length -= 1` is guarded by the `> 0` test on
    // the line above it. `meta_data_upto += 1` runs under
    // `meta_data_upto < limit`, and `limit` is `term_block_ord()`, which is
    // either `next_ent.max(0)` or `term_block_ord` -- both bounded by
    // `ent_count <= i32::MAX` (`ENT_COUNT`), well inside `u32`.
    #[allow(clippy::arithmetic_side_effects)]
    fn decode_meta_data(
        &mut self,
        index_options: IndexOptions,
        has_payloads: bool,
        format: postings::PostingsFormat,
        postings_base: [u64; 3],
    ) -> Result<()> {
        let limit = self.term_block_ord();
        if limit == 0 {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "asked for term metadata with no term read from this block".into(),
            )));
        }
        let mut absolute = self.meta_data_upto == 0;
        let mut r = SliceInput::new(&self.meta_bytes[..self.meta_bytes_len]);
        r.seek(self.meta_pos)?;
        while self.meta_data_upto < limit {
            if self.stats_singleton_run_length > 0 {
                self.doc_freq = 1;
                self.total_term_freq = 1;
                self.stats_singleton_run_length -= 1;
            } else {
                let token = read_vint_at(
                    &self.stat_bytes[..self.stat_bytes_len],
                    &mut self.stats_pos,
                    "stats",
                )?;
                if token & 1 == 1 {
                    self.doc_freq = 1;
                    self.total_term_freq = 1;
                    self.stats_singleton_run_length = (token as u32) >> 1;
                } else {
                    self.doc_freq = ((token as u32) >> 1) as i32;
                    self.total_term_freq = if index_options == IndexOptions::Docs {
                        self.doc_freq as i64
                    } else {
                        // `StatsWriter` writes `totalTermFreq - docFreq`, so
                        // Java's `state.docFreq + statsReader.readVLong()` is
                        // a plain `long` add that silently wraps on a corrupt
                        // stats blob, yielding a *negative* totalTermFreq that
                        // every scorer downstream treats as a real frequency.
                        // Report the overflow instead.
                        let delta = read_vlong_at(
                            &self.stat_bytes[..self.stat_bytes_len],
                            &mut self.stats_pos,
                            "stats",
                        )?;
                        i64::from(self.doc_freq).checked_add(delta).ok_or_else(|| {
                            Error::Store(lucene_store::Error::Corrupted(format!(
                                "totalTermFreq overflows: docFreq={} + delta={delta}",
                                self.doc_freq
                            )))
                        })?
                    };
                }
            }

            let (doc_freq, total_term_freq) = (self.doc_freq, self.total_term_freq);
            let meta = &mut self.meta;
            *meta = postings::decode_term_metadata_for(
                format,
                &mut r,
                doc_freq,
                absolute,
                *meta,
                index_options,
                has_payloads,
                total_term_freq,
            )?;
            if absolute {
                // A block's first term is absolute and every later one a
                // delta from it, so shifting the first shifts them all: into
                // this field's format's part of the reader's buffers.
                meta.doc_start_fp = meta.doc_start_fp.saturating_add(postings_base[0]);
                meta.pos_start_fp = meta.pos_start_fp.saturating_add(postings_base[1]);
                meta.pay_start_fp = meta.pay_start_fp.saturating_add(postings_base[2]);
            }
            self.meta_pos = r.position();
            self.meta_data_upto += 1;
            absolute = false;
        }
        self.term_block_ord = self.meta_data_upto;
        Ok(())
    }
}

/// The reusable half of a [`SegmentTermsEnum`]: the frame stack, the current
/// term's bytes, and where in the stack we are. Split out from the enum so
/// [`FieldTerms`] can pool one and hand it to every `&self` lookup, keeping
/// the last-loaded blocks warm across calls.
#[derive(Debug)]
struct EnumState {
    term: TermBuf,
    stack: Vec<Frame>,
    /// Index of the current frame, or `-1` when the enum is unpositioned
    /// (Java's `currentFrame == staticFrame`).
    current: i32,
    /// Java's `nodes`: `nodes[i]` is the index node the last seek reached
    /// after consuming `term[..i]` (`nodes[0]` the root). Valid up to
    /// [`Self::valid_index_prefix`]; a seek sharing that prefix with the
    /// current term resumes from these instead of walking down from the
    /// root again.
    nodes: Vec<IndexNode>,
    /// Java's `validIndexPrefix`: how many leading bytes of [`Self::term`]
    /// the frame stack and [`Self::nodes`] still describe -- the current
    /// frame's prefix after a seek, cut back by `next()` as it pops frames.
    valid_index_prefix: usize,
    /// Java's `targetBeforeCurrentLength`: the frames at or below this ord
    /// are positioned at or before the target of the seek in progress, so a
    /// push that lands on one holding the same block keeps its cursor rather
    /// than rewinding it (`pushFrame(node, fp, length)`); `-1` rewinds every
    /// such frame.
    target_before_current_length: i32,
    /// Java's `termExists`: the entry the cursor is on is a term rather than
    /// a sub-block pointer. Read by the seek-state reuse: a seek to exactly
    /// the current term answers from it without touching a block.
    term_exists: bool,
    /// The enum is parked on a real term (so `term()`/`docFreq()` are
    /// meaningful). False before the first call and after end-of-terms.
    on_term: bool,
    /// Java's `eof` (an assert-only flag there): repeated `next()` past the
    /// end keeps returning "no more terms" instead of walking off the stack.
    eof: bool,
    /// Set for [`FieldTerms::term_block`]: an exact seek stops where it
    /// would load the block, recording its `.tim` offset in
    /// [`Self::prepared_fp`].
    prepare_only: bool,
    prepared_fp: Option<usize>,
}

impl EnumState {
    /// Unpositioned, as a new enum is, keeping the frames' buffers and the
    /// stack (`SegmentTermsEnum::reset`, without the enum).
    fn unposition(&mut self) {
        self.current = -1;
        self.term.clear();
        self.term_exists = false;
        self.on_term = false;
        self.eof = false;
        self.valid_index_prefix = 0;
        self.target_before_current_length = -1;
    }
}

impl Drop for TermsEnum<'_> {
    /// Returns the state to the field's pool, unless the pool meanwhile
    /// holds a state with more frames.
    fn drop(&mut self) {
        if let Ok(mut pooled) = self.field.scratch.slot().try_lock() {
            if pooled.stack.len() <= self.st.stack.len() {
                let mut st = std::mem::take(&mut self.st);
                st.unposition();
                *pooled = st;
            }
        }
    }
}

/// How many [`EnumState`]s a field pools: enough that a search pool's
/// threads (slots are handed out in thread-creation order) each find their
/// own, few enough that an idle field holds well under a kilobyte.
const SCRATCH_SHARDS: usize = 8;

/// A field's pooled lookup states, one per thread slot (modulo
/// [`SCRATCH_SHARDS`]). A single shared state made every thread but one
/// fall back to a cold state per lookup under concurrent search: four
/// threads seeking one field took 3.5x as long as one.
#[derive(Debug, Default)]
struct ScratchPool([Mutex<EnumState>; SCRATCH_SHARDS]);

impl ScratchPool {
    /// The calling thread's state. Callers still `try_lock`: two threads
    /// can share a slot, and the loser runs on a fresh state, as before.
    fn slot(&self) -> &Mutex<EnumState> {
        use std::sync::atomic::{AtomicUsize, Ordering};
        static NEXT: AtomicUsize = AtomicUsize::new(0);
        thread_local! {
            static SLOT: usize = NEXT.fetch_add(1, Ordering::Relaxed) % SCRATCH_SHARDS;
        }
        &self.0[SLOT.with(|s| *s)]
    }
}

impl Default for EnumState {
    /// Unpositioned: no frame, nothing to reuse.
    fn default() -> Self {
        EnumState {
            term: TermBuf::default(),
            stack: Vec::new(),
            current: -1,
            nodes: Vec::new(),
            valid_index_prefix: 0,
            target_before_current_length: -1,
            term_exists: false,
            on_term: false,
            eof: false,
            prepare_only: false,
            prepared_fp: None,
        }
    }
}

/// Port of `SegmentTermsEnum` -- the whole lazy navigator. Borrows the
/// field's `.tim`/`.tip` bytes and a (possibly pooled) [`EnumState`].
///
/// **Seek-state reuse.** As Java's `prepareSeekExact` and `seekCeil` do, a
/// seek on a positioned enum first compares the target with the current
/// term over the prefix the frame stack still describes
/// ([`EnumState::valid_index_prefix`]), resumes the trie walk from the
/// node stored for the shared prefix ([`EnumState::nodes`]) and the frame
/// for it (`lastFrame`), and -- when the target sorts after the current
/// term -- leaves every frame up to the old current one where its cursor
/// stands ([`EnumState::target_before_current_length`]), so a sorted
/// stream of seeks scans each block forward once instead of restarting at
/// the root and the block's first entry for every target. A target before
/// the current term rewinds the frame it diverges in; the exact current
/// term (`termExists`) answers without reading anything. Until the
/// spatial prefix-tree traversals (heatmaps, RPT intersects) nothing here
/// issued sorted seeks, and every seek restarted from the root; the
/// in-place [`Frame::rewind`] and the per-field pooled state (which make
/// any access order cheap to repeat) stay.
///
/// Unported, with the same status as before:
/// `seekExact(BytesRef, TermState)`/`termState()`/`ord()` (no `TermStates`
/// reuse exists in this port's search layer) and `prefetchBlock`
/// (`IndexInput.prefetch`, unmeasurable against a warm page cache).
struct SegmentTermsEnum<'a> {
    field: &'a FieldTerms,
    st: &'a mut EnumState,
}

impl<'a> SegmentTermsEnum<'a> {
    #[inline]
    fn tim(&self) -> Tim<'a> {
        self.field.tim()
    }

    /// The field's own trie: its `[indexStart, indexEnd)` region of `.tip`.
    /// Empty for a retired FST index, whose frames carry their floor data
    /// themselves ([`Frame::floor_buf`]).
    #[inline]
    fn index(&self) -> &'a [u8] {
        self.field.index.trie_bytes()
    }

    #[inline]
    fn cur(&mut self) -> &mut Frame {
        let ord = self.st.current.max(0) as usize;
        &mut self.st.stack[ord]
    }

    #[inline]
    fn cur_ref(&self) -> &Frame {
        &self.st.stack[self.st.current.max(0) as usize]
    }

    /// The stack index the next `pushFrame` fills, i.e. `currentFrame.ord + 1`.
    ///
    /// A `checked_add` rather than an `// ARITH:` proof, because there is no
    /// honest proof to write: the only structural bound on stack depth is the
    /// strictly-decreasing sub-block fp chain, so it is `.tim`'s own length
    /// that limits how deep a corrupt file can drive the descent, and that is
    /// a 64-bit quantity. The check costs one branch per *block* push, which
    /// [`Frame::load_block`] dwarfs.
    fn next_ord(&self) -> Result<usize> {
        let ord = self.st.current.checked_add(1).ok_or_else(|| {
            Error::Store(lucene_store::Error::Corrupted(
                "terms frame stack depth overflowed".into(),
            ))
        })?;
        debug_assert!(ord >= 0, "st.current is never below -1");
        Ok(ord.max(0) as usize)
    }

    /// `SegmentTermsEnum.getFrame(ord)`.
    fn ensure_frame(&mut self, ord: usize) {
        while self.st.stack.len() <= ord {
            let n = self.st.stack.len();
            self.st.stack.push(Frame {
                ord: n,
                next_ent: -1,
                ..Frame::default()
            });
        }
    }

    /// `SegmentTermsEnum.pushFrame(node, length)` -- or, for a retired FST
    /// index, `Lucene90`'s `pushFrame(arc, frameData, length)`, which decodes
    /// the same three facts (block fp, `hasTerms`, floor data) from the
    /// output the arcs to `node` assembled.
    fn push_frame_node(&mut self, node: &IndexNode, length: usize) -> Result<()> {
        let ord = self.next_ord()?;
        self.ensure_frame(ord);
        let index = self.index();
        let field = self.field;
        let f = &mut self.st.stack[ord];
        let fp = match node {
            IndexNode::Trie(node) => {
                f.floor_owned = false;
                f.has_terms = node.has_terms;
                f.has_terms_orig = node.has_terms;
                f.is_floor = node.floor_data_fp.is_some();
                if let Some(fdp) = node.floor_data_fp {
                    f.set_floor_data(index, fdp)?;
                }
                node.output_fp.unwrap_or_default()
            }
            IndexNode::Fst(node) => {
                let block = field.index.fst()?.block(node, &mut f.floor_buf)?;
                f.floor_owned = true;
                f.has_terms = block.has_terms;
                f.has_terms_orig = block.has_terms;
                f.is_floor = block.floor_start.is_some();
                if let Some(start) = block.floor_start {
                    f.set_floor_data(index, start)?;
                }
                block.fp
            }
        };
        let fp = usize::try_from(fp).map_err(|_| {
            Error::Store(lucene_store::Error::Corrupted(format!(
                "terms block fp {fp} out of range"
            )))
        })?;
        self.push_frame_fp(fp, length)
    }

    /// `SegmentTermsEnum.pushFrame(node, fp, length)` -- the shared tail, also
    /// used on its own for a frame pushed by following an in-block sub-block
    /// pointer (Java's `pushFrame(null, fp, length)`).
    fn push_frame_fp(&mut self, fp: usize, length: usize) -> Result<()> {
        let ord = self.next_ord()?;
        self.ensure_frame(ord);
        let index = self.index();
        let keep = ord as i64 <= i64::from(self.st.target_before_current_length);
        let f = &mut self.st.stack[ord];
        if f.fp_orig == fp && f.next_ent != -1 && f.prefix_length == length {
            // Same block as last time this ord was used. A frame at or below
            // `targetBeforeCurrentLength` stands at or before the target of
            // this seek, so its cursor is kept and the scan resumes from it;
            // any other is rewound -- in place, keeping its decoded regions.
            if !keep {
                f.rewind(index)?;
            }
        } else {
            f.next_ent = -1;
            f.prefix_length = length;
            f.term_block_ord = 0;
            f.fp = fp;
            f.fp_orig = fp;
            f.last_sub_fp = -1;
        }
        self.st.current = ord as i32;
        Ok(())
    }

    /// Pushes the "next"-style frame Java writes as `pushFrame(null, fp, len)`:
    /// a frame reached by an in-block sub-block pointer, which must be treated
    /// as un-floored even if the block it loads says otherwise (Java's "even
    /// if it's floor'd we must pretend it isn't so we don't try to scan to the
    /// right floor frame").
    fn push_next_frame(&mut self, fp: i64, length: usize) -> Result<()> {
        if fp < 0 {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "sub-block fp was never set".into(),
            )));
        }
        let ord = self.next_ord()?;
        self.ensure_frame(ord);
        {
            let f = &mut self.st.stack[ord];
            f.is_floor = false;
            f.has_terms = true;
            f.has_terms_orig = true;
        }
        self.push_frame_fp(fp as usize, length)
    }

    fn load_current_block(&mut self) -> Result<()> {
        let tim = self.tim();
        self.cur().load_block(tim)
    }

    /// Puts the enum back in its unpositioned state, keeping every frame's
    /// decoded block so the next lookup can reuse it.
    fn reset(&mut self) {
        self.st.unposition();
    }

    /// `nodes[0] = trieReader.root` (`getNode` for the root slot).
    fn set_root_node(&mut self, root: IndexNode) {
        match self.st.nodes.first_mut() {
            Some(slot) => *slot = root,
            None => self.st.nodes.push(root),
        }
    }

    /// Stores the node reached after `depth` target bytes
    /// (`lookupChild(label, node, getNode(depth))`); the slots below it are
    /// already the path's.
    fn set_node(&mut self, depth: usize, node: IndexNode) {
        if depth < self.st.nodes.len() {
            self.st.nodes[depth] = node;
        } else {
            debug_assert_eq!(depth, self.st.nodes.len());
            self.st.nodes.push(node);
        }
    }

    /// The opening of `prepareSeekExact`/`seekCeil` on a positioned enum:
    /// how far the target shares the prefix the frame stack still
    /// describes. Returns the bytes matched (`targetUpto`), the frame for
    /// that prefix (`lastFrame.ord`) and how the current term compares with
    /// the target; `None` when the stored state cannot describe the term
    /// (it always can, but a stale pooled state must restart from the root
    /// rather than index out of bounds).
    // ARITH: `target_upto < target_limit <= term.len`, and `last_frame`
    // counts frames along the path, each one a pushed stack entry, so both
    // increments stay below lengths of live allocations.
    #[allow(clippy::arithmetic_side_effects)]
    fn shared_prefix(&self, target: &[u8]) -> Option<(usize, usize, std::cmp::Ordering)> {
        use std::cmp::Ordering;
        let term = self.st.term.get();
        let target_limit = target.len().min(self.st.valid_index_prefix).min(term.len());
        if self.st.nodes.len() <= target_limit || self.st.current < 0 {
            return None;
        }
        let mut target_upto = 0usize;
        let mut last_frame = 0usize;
        let mut cmp = Ordering::Equal;
        while target_upto < target_limit {
            cmp = term[target_upto].cmp(&target[target_upto]);
            if cmp != Ordering::Equal {
                break;
            }
            if self.st.nodes[1 + target_upto].has_output() {
                last_frame += 1;
            }
            target_upto += 1;
        }
        if cmp == Ordering::Equal {
            cmp = term[target_upto..].cmp(&target[target_upto..]);
        }
        if last_frame > self.st.current as usize {
            return None;
        }
        Some((target_upto, last_frame, cmp))
    }

    fn root(&self) -> Result<IndexNode> {
        self.field.index.root()
    }

    /// `SegmentTermsEnum.seekExact(BytesRef)` (`prepareSeekExact` without
    /// prefetch, then its supplier).
    ///
    /// A seek that fails part way leaves the pooled state unpositioned: a
    /// half-updated frame stack must not be resumed by the next seek.
    fn seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        // `prepareSeekExact`'s first line: the field's recorded min/max term
        // bound every possible hit, so an out-of-range target never touches
        // the trie at all -- nor the seek state, which stays where it was.
        if self.field.num_terms > 0
            && (target < self.field.min_term.as_slice() || target > self.field.max_term.as_slice())
        {
            self.st.on_term = false;
            return Ok(false);
        }
        let r = self.seek_exact_inner(target);
        if r.is_err() {
            self.reset();
        }
        r
    }

    // ARITH: `target_upto` is only incremented inside `while target_upto <
    // target.len()`, so it never exceeds `target.len()`, which as a slice
    // length is at most `isize::MAX`; `target_upto + 1` and `1 + target_upto`
    // therefore cannot overflow `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_exact_inner(&mut self, target: &[u8]) -> Result<bool> {
        use std::cmp::Ordering;
        self.st.eof = false;
        self.st.on_term = false;
        let index = self.index();
        self.st.target_before_current_length = self.st.current;
        let mut target_upto = match self.shared_prefix(target) {
            Some((target_upto, last_frame, cmp)) => {
                match cmp {
                    // The target is after the current term: every frame up
                    // to the current one stands before it.
                    Ordering::Less => self.st.current = last_frame as i32,
                    // Before it: rewind the frame where they diverge, and
                    // every frame pushed above it.
                    Ordering::Greater => {
                        self.st.target_before_current_length = last_frame as i32;
                        self.st.current = last_frame as i32;
                        self.cur().rewind(index)?;
                    }
                    Ordering::Equal => {
                        if self.st.term_exists {
                            self.st.on_term = true;
                            return Ok(true);
                        }
                        // The current "term" is a sub-block or a prefix a
                        // failed seek left: walk on from where it stands.
                    }
                }
                target_upto
            }
            None => {
                self.reset();
                self.st.target_before_current_length = -1;
                let root = self.root()?;
                self.set_root_node(root);
                self.push_node_frame_at(0)?;
                0
            }
        };

        while target_upto < target.len() {
            let target_label = target[target_upto];
            let next = self
                .field
                .index
                .child(index, &self.st.nodes[target_upto], target_label)?;
            match next {
                None => {
                    // The index is exhausted: this frame's block is the only
                    // one that could hold the target.
                    self.st.valid_index_prefix = self.cur_ref().prefix_length;
                    self.cur().scan_to_floor_frame(index, target)?;
                    if !self.cur_ref().has_terms {
                        self.st.term_exists = false;
                        self.st.term.set_byte_at(target_upto, target_label);
                        self.st.term.set_length(1 + target_upto);
                        return Ok(false);
                    }
                    return self.load_and_scan_exact(target);
                }
                Some(next_node) => {
                    self.st.term.set_byte_at(target_upto, target_label);
                    let has_output = next_node.has_output();
                    target_upto += 1;
                    self.set_node(target_upto, next_node);
                    if has_output {
                        self.push_node_frame_at(target_upto)?;
                    }
                }
            }
        }

        self.st.valid_index_prefix = self.cur_ref().prefix_length;
        self.cur().scan_to_floor_frame(index, target)?;
        if !self.cur_ref().has_terms {
            self.st.term_exists = false;
            self.st.term.set_length(target_upto);
            return Ok(false);
        }
        self.load_and_scan_exact(target)
    }

    /// `pushFrame(node, length)` for the node stored at `nodes[depth]`,
    /// whose prefix is `depth` bytes long.
    fn push_node_frame_at(&mut self, depth: usize) -> Result<()> {
        // The node is moved out for the push (which borrows the stack) and
        // put back: no clone of an FST node's output bytes per frame.
        let node = std::mem::replace(
            &mut self.st.nodes[depth],
            IndexNode::Trie(TrieNode::default()),
        );
        let r = self.push_frame_node(&node, depth);
        self.st.nodes[depth] = node;
        r
    }

    fn load_and_scan_exact(&mut self, target: &[u8]) -> Result<bool> {
        if self.st.prepare_only {
            self.st.prepared_fp = Some(self.cur_ref().fp);
            return Ok(false);
        }
        self.load_current_block()?;
        let ord = self.st.current.max(0) as usize;
        let EnumState {
            term,
            stack,
            term_exists,
            ..
        } = &mut *self.st;
        let (status, _) = stack[ord].scan_to_term(term, target, true, term_exists)?;
        let found = status == SeekStatus::Found;
        self.st.on_term = found;
        Ok(found)
    }

    /// `SegmentTermsEnum.seekCeil(BytesRef)`. A seek that fails part way
    /// leaves the state unpositioned, as [`Self::seek_exact`] does.
    fn seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let r = self.seek_ceil_inner(target);
        if r.is_err() {
            self.reset();
        }
        r
    }

    // ARITH: as in [`SegmentTermsEnum::seek_exact_inner`] -- `target_upto` is
    // only incremented under `while target_upto < target.len()`, and a slice
    // length is at most `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_ceil_inner(&mut self, target: &[u8]) -> Result<SeekStatus> {
        use std::cmp::Ordering;
        self.st.eof = false;
        self.st.on_term = false;
        let index = self.index();
        self.st.target_before_current_length = self.st.current;
        let mut target_upto = match self.shared_prefix(target) {
            Some((target_upto, last_frame, cmp)) => {
                match cmp {
                    Ordering::Less => self.st.current = last_frame as i32,
                    Ordering::Greater => {
                        // `seekCeil` writes 0 here where `prepareSeekExact`
                        // writes `lastFrame.ord`: kept as Java has it.
                        self.st.target_before_current_length = 0;
                        self.st.current = last_frame as i32;
                        self.cur().rewind(index)?;
                    }
                    Ordering::Equal => {
                        if self.st.term_exists {
                            self.st.on_term = true;
                            return Ok(SeekStatus::Found);
                        }
                    }
                }
                target_upto
            }
            None => {
                self.reset();
                self.st.target_before_current_length = -1;
                let root = self.root()?;
                self.set_root_node(root);
                self.push_node_frame_at(0)?;
                0
            }
        };

        while target_upto < target.len() {
            let target_label = target[target_upto];
            let next = self
                .field
                .index
                .child(index, &self.st.nodes[target_upto], target_label)?;
            match next {
                None => {
                    self.st.valid_index_prefix = self.cur_ref().prefix_length;
                    self.cur().scan_to_floor_frame(index, target)?;
                    return self.load_and_scan_ceil(target);
                }
                Some(next_node) => {
                    self.st.term.set_byte_at(target_upto, target_label);
                    let has_output = next_node.has_output();
                    target_upto += 1;
                    self.set_node(target_upto, next_node);
                    if has_output {
                        self.push_node_frame_at(target_upto)?;
                    }
                }
            }
        }

        self.st.valid_index_prefix = self.cur_ref().prefix_length;
        self.cur().scan_to_floor_frame(index, target)?;
        self.load_and_scan_ceil(target)
    }

    // ARITH: `prefix_length + suffix_length` is the same sum
    // [`Frame::fill_term`] forms and is bounded the same way -- two lengths of
    // live allocations, each at most `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn load_and_scan_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.load_current_block()?;
        let ord = self.st.current.max(0) as usize;
        let (status, descend) = {
            let EnumState {
                term,
                stack,
                term_exists,
                ..
            } = &mut *self.st;
            stack[ord].scan_to_term(term, target, false, term_exists)?
        };

        if descend {
            // The scan stopped on a sub-block that sorts after the target;
            // the true ceiling is that sub-block's first term.
            let (sub_fp, length) = {
                let f = self.cur_ref();
                (f.last_sub_fp, f.prefix_length + f.suffix_length)
            };
            self.push_next_frame(sub_fp, length)?;
            self.load_current_block()?;
            loop {
                let tim = self.tim();
                let ord = self.st.current.max(0) as usize;
                let EnumState { term, stack, .. } = &mut *self.st;
                if !stack[ord].next(tim, term)? {
                    break;
                }
                let sub_fp = self.cur_ref().last_sub_fp;
                let length = self.st.term.len;
                self.push_next_frame(sub_fp, length)?;
                self.load_current_block()?;
            }
            // The last `next()` landed on a term (Java's frame sets
            // `termExists`).
            self.st.term_exists = true;
            self.st.on_term = true;
            return Ok(SeekStatus::NotFound);
        }

        if status == SeekStatus::End {
            // Past the last term of this block; the ceiling, if any, is the
            // next term in document order.
            self.st.term.copy_from(target);
            self.st.term_exists = false;
            if self.next()? {
                return Ok(SeekStatus::NotFound);
            }
            return Ok(SeekStatus::End);
        }
        self.st.on_term = true;
        Ok(status)
    }

    /// `SegmentTermsEnum.next()`: `true` when it advanced onto a term.
    fn next(&mut self) -> Result<bool> {
        if self.st.eof {
            return Ok(false);
        }
        if self.st.current < 0 {
            let node = self.root()?;
            self.set_root_node(node);
            self.push_node_frame_at(0)?;
            self.load_current_block()?;
        }
        self.st.target_before_current_length = self.st.current;

        // Pop finished blocks.
        loop {
            let (next_ent, ent_count, is_last_in_floor, ord, fp_orig) = {
                let f = self.cur_ref();
                (
                    f.next_ent,
                    f.ent_count as i32,
                    f.is_last_in_floor,
                    f.ord,
                    f.fp_orig,
                )
            };
            if next_ent != ent_count {
                break;
            }
            if !is_last_in_floor {
                let tim = self.tim();
                self.cur().load_next_floor_block(tim)?;
                break;
            }
            if ord == 0 {
                self.st.eof = true;
                self.st.on_term = false;
                self.st.term_exists = false;
                self.st.term.clear();
                self.st.valid_index_prefix = 0;
                let index = self.index();
                self.cur().rewind(index)?;
                return Ok(false);
            }
            // ARITH: `ord` read three lines up is `cur_ref().ord`, which
            // `ensure_frame` sets to the frame's own index in `st.stack`, i.e.
            // to `st.current`; the `ord == 0` branch above has already
            // returned, so `st.current >= 1` here and the decrement cannot go
            // below 0.
            #[allow(clippy::arithmetic_side_effects)]
            {
                debug_assert_eq!(ord as i32, self.st.current);
                self.st.current -= 1;
            }
            let parent = self.st.current.max(0) as usize;
            let needs_reposition = {
                let f = &self.st.stack[parent];
                f.next_ent == -1 || f.last_sub_fp != fp_orig as i64
            };
            if needs_reposition {
                // We popped into a frame that is either not loaded or not
                // scanned to the right entry. Borrow the term and the frame
                // as disjoint fields so repositioning costs no allocation.
                let index = self.index();
                let tim = self.tim();
                let EnumState { term, stack, .. } = &mut *self.st;
                let f = &mut stack[parent];
                f.scan_to_floor_frame(index, term.get())?;
                f.load_block(tim)?;
                f.scan_to_sub_block(fp_orig as i64)?;
            }
            self.st.valid_index_prefix = self
                .st
                .valid_index_prefix
                .min(self.st.stack[parent].prefix_length);
        }

        loop {
            let tim = self.tim();
            let is_sub = {
                let ord = self.st.current.max(0) as usize;
                let EnumState { term, stack, .. } = &mut *self.st;
                stack[ord].next(tim, term)?
            };
            if !is_sub {
                self.st.term_exists = true;
                self.st.on_term = true;
                return Ok(true);
            }
            let sub_fp = self.cur_ref().last_sub_fp;
            let length = self.st.term.len;
            self.push_next_frame(sub_fp, length)?;
            self.load_current_block()?;
        }
    }

    /// `SegmentTermsEnum.docFreq()`/`totalTermFreq()`.
    fn stats(&mut self) -> Result<TermStats> {
        let (index_options, has_payloads, format, base) = (
            self.field.index_options,
            self.field.has_payloads,
            self.field.postings_format,
            self.field.postings_base,
        );
        let f = self.cur();
        f.decode_meta_data(index_options, has_payloads, format, base)?;
        Ok(TermStats {
            doc_freq: f.doc_freq,
            total_term_freq: f.total_term_freq,
        })
    }

    /// `SegmentTermsEnum.postings()`'s half of `decodeMetaData`: the postings
    /// file pointers for the term the enum is parked on.
    fn stats_and_meta(&mut self) -> Result<(TermStats, TermMetadata)> {
        let (index_options, has_payloads, format, base) = (
            self.field.index_options,
            self.field.has_payloads,
            self.field.postings_format,
            self.field.postings_base,
        );
        let f = self.cur();
        f.decode_meta_data(index_options, has_payloads, format, base)?;
        Ok((
            TermStats {
                doc_freq: f.doc_freq,
                total_term_freq: f.total_term_freq,
            },
            f.meta,
        ))
    }

    /// Where the cursor sits, for the intersect iterators' skip accounting:
    /// which frame, which block, which entry.
    fn position(&self) -> (i32, usize, i32) {
        if self.st.current < 0 {
            return (-1, 0, -1);
        }
        let f = self.cur_ref();
        (self.st.current, f.fp, f.next_ent)
    }
}

/// `TermsEnum`-equivalent: ordered enumeration (`next()`) and nearest-match
/// seeking (`seekCeil()`) over one field's term dictionary, backed by the
/// lazy [`SegmentTermsEnum`] frame stack.
#[derive(Debug)]
pub struct TermsEnum<'a> {
    field: &'a FieldTerms,
    st: EnumState,
}

impl<'a> TermsEnum<'a> {
    /// `Terms.iterator()`: an unpositioned enum. It borrows the field's
    /// pooled state when no lookup holds it -- its frames keep their block
    /// buffers, so an enum per query (a spatial traversal, a multi-term
    /// rewrite) does not allocate and grow a dozen of them again each
    /// time, the cost Java's TLAB hides -- and hands it back on drop.
    fn new(field: &'a FieldTerms) -> Self {
        let mut st = match field.scratch.slot().try_lock() {
            Ok(mut pooled) => std::mem::take(&mut *pooled),
            Err(_) => EnumState::default(),
        };
        st.unposition();
        Self { field, st }
    }

    fn ste(&mut self) -> SegmentTermsEnum<'_> {
        SegmentTermsEnum {
            field: self.field,
            st: &mut self.st,
        }
    }

    /// `TermsEnum.next()`: advance to the next term in sorted order, with its
    /// `docFreq`/`totalTermFreq`, or `None` at end-of-terms.
    ///
    /// Loads at most one further `.tim` block per call, and only decodes the
    /// postings metadata of the term it lands on.
    ///
    /// **Java's `next()` is [`Self::try_next_term`], not this.** Real
    /// `SegmentTermsEnum.next()` decodes only the term bytes and leaves
    /// `decodeMetaData` to `docFreq()`/`totalTermFreq()`/`postings()`; this
    /// spelling is the two fused, kept because most callers do want both and
    /// fusing them saves them a second call. A caller that only needs the
    /// term bytes should use [`Self::try_next_term`] and ask for
    /// [`Self::try_stats`] on the terms it keeps.
    pub fn try_next(&mut self) -> Result<Option<(&[u8], TermStats)>> {
        let mut ste = self.ste();
        if !ste.next()? {
            return Ok(None);
        }
        let stats = ste.stats()?;
        Ok(Some((self.st.term.get(), stats)))
    }

    /// `SegmentTermsEnum.next()` exactly: advance to the next term and hand
    /// back **only its bytes**, leaving `decodeMetaData` undone.
    ///
    /// A scan that filters on the term
    /// bytes -- a wildcard/regexp/fuzzy intersection, a term-range delete, a
    /// `CheckIndex` pass -- discards `docFreq`/`totalTermFreq` for every term
    /// it rejects, and through [`Self::try_next`] it paid a stats vint and a
    /// full [`TermMetadata`] decode (`.doc`/`.pos`/`.pay` file pointers,
    /// singleton pulsing) to produce them. Java never does: `next()` returns a
    /// `BytesRef` and the metadata is decoded lazily, at most once per frame
    /// position, by whichever accessor first asks.
    ///
    /// Pair it with [`Self::try_stats`], which decodes on demand and is
    /// idempotent per position -- so `try_next_term()` then `try_stats()` on
    /// every term costs exactly what [`Self::try_next`] does, and the saving
    /// is real only for the terms whose stats are never asked for.
    pub fn try_next_term(&mut self) -> Result<Option<&[u8]>> {
        if !self.ste().next()? {
            return Ok(None);
        }
        Ok(Some(self.st.term.get()))
    }

    /// The bytes of the term the cursor is parked on, without decoding
    /// anything -- `TermsEnum.term()`. `None` before the first advance, past
    /// the end, or after a [`Self::try_seek_ceil`] that returned
    /// [`SeekStatus::End`].
    pub fn term(&self) -> Option<&[u8]> {
        self.st.on_term.then(|| self.st.term.get())
    }

    /// `TermsEnum.docFreq()`/`totalTermFreq()`: the current term's stats,
    /// decoded now. `None` in exactly the positions [`Self::term`] returns
    /// `None` for.
    ///
    /// Decoding is memoised per frame position by
    /// `SegmentTermsEnumFrame.metaDataUpto`, so calling this twice on one
    /// term costs one decode, and calling it after [`Self::try_next`] (which
    /// has already decoded) costs none.
    pub fn try_stats(&mut self) -> Result<Option<TermStats>> {
        if !self.st.on_term {
            return Ok(None);
        }
        self.ste().stats().map(Some)
    }

    /// [`Self::try_stats`] plus the postings pointers: the term the enum is
    /// parked on as a [`SeekedTerm`], to open postings from without seeking
    /// again. One metadata decode either way.
    pub fn try_seeked_term(&mut self) -> Result<Option<SeekedTerm>> {
        if !self.st.on_term {
            return Ok(None);
        }
        let (stats, meta) = self.ste().stats_and_meta()?;
        Ok(Some(SeekedTerm { stats, meta }))
    }

    /// [`Self::try_next`] with the error dropped: a corrupt block reads as
    /// end-of-terms.
    ///
    /// **Test convenience only.** No production caller in this workspace uses
    /// this spelling (re-checkable by marking these four methods
    /// `#[deprecated]` and building `--all-targets`). It survives because a
    /// test that has just built its own bytes gains nothing from an error
    /// channel; a decoder that degrades corruption to "no such term" is how a
    /// corrupt index reads as an empty one.
    ///
    /// Named to mirror Java's `TermsEnum.next()` rather than
    /// `std::iter::Iterator::next`: a real `Iterator` impl would need `Item`
    /// to borrow from `self`, which that trait cannot express.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&[u8], TermStats)> {
        self.try_next().unwrap_or(None)
    }

    /// `TermsEnum.seekCeil(BytesRef)`: position on the smallest term
    /// `>= target` and report whether that was an exact match, a ceiling
    /// match, or that no such term exists.
    pub fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.ste().seek_ceil(target)
    }

    /// [`Self::try_seek_ceil`] with the error dropped: a corrupt block reads
    /// as [`SeekStatus::End`]. **Test convenience only** -- see
    /// [`Self::next`].
    pub fn seek_ceil(&mut self, target: &[u8]) -> SeekStatus {
        self.try_seek_ceil(target).unwrap_or(SeekStatus::End)
    }

    /// The term/stats the cursor is currently on — `None` before the first
    /// `next()`/`seek_ceil()` call, past the end, or when `seek_ceil`
    /// returned [`SeekStatus::End`].
    pub fn try_current(&mut self) -> Result<Option<(&[u8], TermStats)>> {
        if !self.st.on_term {
            return Ok(None);
        }
        let stats = self.ste().stats()?;
        Ok(Some((self.st.term.get(), stats)))
    }

    /// [`Self::try_current`] with the error dropped. **Test convenience
    /// only** -- see [`Self::next`].
    pub fn current(&mut self) -> Option<(&[u8], TermStats)> {
        self.try_current().unwrap_or(None)
    }

    /// `TermsEnum.postings(...)` at the cursor's **current** position:
    /// the same result as [`FieldTerms::postings`] for the current term, but
    /// with no dictionary seek — the term's postings metadata is already
    /// decoded in the cursor's own frame.
    ///
    /// This is what makes a genuinely streaming merge possible. Java's
    /// `FieldsConsumer.merge` pulls one `TermsEnum` forward per sub-reader and
    /// asks each for the current term's `PostingsEnum`; asking through
    /// `FieldTerms::postings(term, ..)` instead means re-seeking from the trie
    /// root for a term the cursor is already standing on.
    ///
    /// `None` before the first `next()`/`seek_ceil()`, past the end, or after
    /// a `seek_ceil` that returned [`SeekStatus::End`].
    pub fn try_current_postings(
        &mut self,
        doc_in: Option<&DocInput<'_>>,
    ) -> Result<Option<Postings>> {
        if !self.st.on_term {
            return Ok(None);
        }
        let (stats, meta) = self.ste().stats_and_meta()?;
        Ok(Some(self.field.postings_from(stats, meta, doc_in)?))
    }

    /// [`Self::try_current_postings`] plus the current term's positions
    /// (with offsets/payloads, when the field indexes them) — decoded from
    /// the one metadata read, so a positional merge costs one traversal step
    /// rather than two seeks and two docs/freqs decodes.
    ///
    /// Needs a field with [`IndexOptions::DocsAndFreqsAndPositions`] or
    /// higher, exactly like [`FieldTerms::positions`].
    pub fn try_current_postings_and_positions(
        &mut self,
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
    ) -> Result<Option<(Postings, Vec<Vec<postings::Position>>)>> {
        if !self.st.on_term {
            return Ok(None);
        }
        let (stats, meta) = self.ste().stats_and_meta()?;
        let docs = self.field.postings_from(stats, meta, doc_in)?;
        let positions = postings::read_positions(
            pos_in,
            pay_in,
            meta,
            &docs.freqs,
            stats.total_term_freq,
            self.field.index_options,
            self.field.has_payloads,
        )?;
        Ok(Some((docs, positions)))
    }

    /// The current term's docs and complete occurrence streams, without a
    /// `Position` object or payload allocation per occurrence. A merge can
    /// select each live document's slices before building its writer inputs.
    ///
    /// Java: `TermsEnum.postings` and `PostingsEnum.nextPosition`. Like
    /// [`Self::try_current_postings_and_positions`], this reads metadata and
    /// docs once and validates the whole source stream before live filtering.
    /// Returns `None` when the cursor is not positioned on a term.
    pub fn try_current_postings_and_occurrences_flat(
        &mut self,
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
    ) -> Result<Option<(Postings, postings::FlatOccurrences)>> {
        if !self.st.on_term {
            return Ok(None);
        }
        let (stats, meta) = self.ste().stats_and_meta()?;
        let docs = self.field.postings_from(stats, meta, doc_in)?;
        let occurrences = postings::read_occurrences_flat(
            pos_in,
            pay_in,
            meta,
            &docs.freqs,
            stats.total_term_freq,
            self.field.index_options,
            self.field.has_payloads,
        )?;
        Ok(Some((docs, occurrences)))
    }

    /// [`Self::try_current_postings`] into `out`, reusing its buffers, with
    /// `postings(FREQS)`'s flags: no impacts. What a merge reads term after
    /// term. `false` (and `out` untouched) when not standing on a term.
    ///
    /// # Errors
    /// As [`Self::try_current_postings`].
    pub fn try_current_postings_into(
        &mut self,
        doc_in: Option<&DocInput<'_>>,
        out: &mut Postings,
    ) -> Result<bool> {
        if !self.st.on_term {
            return Ok(false);
        }
        let (stats, meta) = self.ste().stats_and_meta()?;
        self.field.postings_into(stats, meta, doc_in, out)?;
        Ok(true)
    }

    /// [`Self::try_current_postings_and_occurrences_flat`] into `docs` and
    /// `occurrences`, reusing their buffers; impacts are not read.
    ///
    /// # Errors
    /// As [`Self::try_current_postings_and_occurrences_flat`].
    pub fn try_current_postings_and_occurrences_flat_into(
        &mut self,
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
        docs: &mut Postings,
        occurrences: &mut postings::FlatOccurrences,
    ) -> Result<bool> {
        if !self.st.on_term {
            return Ok(false);
        }
        let (stats, meta) = self.ste().stats_and_meta()?;
        self.field.postings_into(stats, meta, doc_in, docs)?;
        postings::read_occurrences_flat_into(
            pos_in,
            pay_in,
            meta,
            &docs.freqs,
            stats.total_term_freq,
            self.field.index_options,
            self.field.has_payloads,
            occurrences,
        )?;
        Ok(true)
    }
}

/// One field's term dictionary: the `.tmd` record plus the `.tim`/`.tip`
/// bytes to navigate on demand — the port of `FieldReader`.
///
/// Cloning shares the underlying `.tim`/`.tip` buffers (`Arc`) and starts the
/// clone with an empty lookup scratch.
/// Shared, type-erased ownership of one whole codec file's bytes.
///
/// [`FieldTerms`] navigates the segment's `.tim`/`.tip` for as long as it
/// lives, so it has to *own* a share of them. An `Arc<[u8]>` owns its own
/// allocation: building one from a `memmap2::Mmap` (what
/// `lucene_store::MmapDirectory` hands back, and Lucene's own
/// `MMapDirectory` equivalent) copies the whole file. On the benchmark
/// corpus that was 199 µs of a 579 µs `DirectoryReader::open` -- for a 4.7 MB
/// `.tim` whose bytes were already resident in the page cache.
///
/// Erasing the owner instead lets `open_shared` take an `Arc<Input>` (a
/// mapping, or a `Vec` someone else already owns) unchanged. The cost is one
/// virtual `as_ref` per [`SegmentTermsEnum::tim`]/[`SegmentTermsEnum::index`]
/// call -- both of which are hoisted into a local at the top of each lookup,
/// so it is a handful of predicted indirect calls per *seek*, not per byte
/// or per term. Measured: no change to `blocktree_open`'s seek cases.
pub type SharedBytes = Arc<dyn AsRef<[u8]> + Send + Sync>;

/// A field's trie: `[start, end)` of `bytes`, rooted at `root_fp`.
#[derive(Clone)]
struct TrieSlice {
    bytes: SharedBytes,
    start: usize,
    root_fp: usize,
    end: usize,
}

impl TrieSlice {
    fn bytes(&self) -> &[u8] {
        &self.bytes.as_ref().as_ref()[self.start..self.end]
    }
}

/// A retired format's FST terms index, walked in place
/// (`crate::backward_codecs::blocktree`).
pub(crate) use crate::backward_codecs::blocktree::{FstNode, FstTermsIndex};

/// A node of a field's terms index: a trie node, or -- for a retired FST
/// index -- the arc reaching a prefix plus the output its path assembled.
/// [`SegmentTermsEnum`] steps through either the same way: a child per
/// target byte, a frame pushed wherever the node carries a block.
#[derive(Debug, Clone)]
pub(crate) enum IndexNode {
    Trie(TrieNode),
    Fst(FstNode),
}

impl IndexNode {
    /// `Node.hasOutput()` / `Arc.isFinal()`: the prefix names a block.
    #[inline]
    fn has_output(&self) -> bool {
        match self {
            IndexNode::Trie(n) => n.output_fp.is_some(),
            IndexNode::Fst(n) => n.has_block(),
        }
    }
}

/// Where a field's terms index lives.
#[derive(Clone)]
enum TermsIndex {
    /// `.tip`'s trie, read in place (`Lucene103BlockTreeTermsReader`).
    InPlace(TrieSlice),
    /// A retired format's FST in `.tip`, read in place
    /// (`Lucene90BlockTreeTermsReader`'s `FieldReader.index`).
    Fst(Arc<FstTermsIndex>),
}

impl TermsIndex {
    /// The trie's bytes, which trie nodes and their floor data index into;
    /// empty for an FST index, whose frames own their floor data.
    #[inline]
    fn trie_bytes(&self) -> &[u8] {
        match self {
            TermsIndex::InPlace(t) => t.bytes(),
            TermsIndex::Fst(_) => &[],
        }
    }

    /// The FST index, for a node that came from one.
    fn fst(&self) -> Result<&FstTermsIndex> {
        match self {
            TermsIndex::Fst(f) => Ok(f),
            TermsIndex::InPlace(_) => Err(wrong_index_kind()),
        }
    }

    /// The root node (`TrieReader.root` / `FST.getFirstArc`).
    fn root(&self) -> Result<IndexNode> {
        match self {
            TermsIndex::InPlace(t) => Ok(IndexNode::Trie(load_node(t.bytes(), t.root_fp)?)),
            TermsIndex::Fst(f) => Ok(IndexNode::Fst(f.root())),
        }
    }

    /// `TrieReader.lookupChild` / `FST.findTargetArc`; `trie` is
    /// [`Self::trie_bytes`], hoisted by the caller.
    #[inline]
    fn child(&self, trie: &[u8], node: &IndexNode, label: u8) -> Result<Option<IndexNode>> {
        match node {
            IndexNode::Trie(n) => Ok(lookup_child(trie, n, label)?.map(IndexNode::Trie)),
            IndexNode::Fst(n) => Ok(self.fst()?.child(n, label)?.map(IndexNode::Fst)),
        }
    }

    /// The block `node` names, if any.
    fn block_fp(&self, node: &IndexNode) -> Result<Option<u64>> {
        match node {
            IndexNode::Trie(n) => Ok(n.output_fp),
            IndexNode::Fst(n) => self.fst()?.block_fp(n),
        }
    }
}

fn wrong_index_kind() -> Error {
    Error::Store(lucene_store::Error::Corrupted(
        "terms index node of the wrong kind for its field".into(),
    ))
}

impl std::fmt::Debug for TermsIndex {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TermsIndex::InPlace(t) => {
                write!(f, "InPlace({}..{}, root {})", t.start, t.end, t.root_fp)
            }
            TermsIndex::Fst(i) => write!(f, "Fst({i:?})"),
        }
    }
}

pub struct FieldTerms {
    pub num_terms: i64,
    pub sum_total_term_freq: i64,
    pub sum_doc_freq: i64,
    pub doc_count: i32,
    pub min_term: Vec<u8>,
    pub max_term: Vec<u8>,
    index_options: IndexOptions,
    has_payloads: bool,
    /// The postings format that wrote this field (its `FieldInfo`'s
    /// `PerFieldPostingsFormat.format`): each generation's `decodeTerm`
    /// differs, so the dictionary has to know which one it is decoding.
    postings_format: postings::PostingsFormat,
    /// The whole segment's `.tim`, shared by every field.
    tim: SharedBytes,
    /// Set for a dictionary with zstd blocks.
    zstd: Option<Arc<ZstdBlocks>>,
    /// The field's terms index, read in place from `.tip`: a trie, or a
    /// retired format's FST.
    index: TermsIndex,
    /// Pooled [`EnumState`]s so the `&self` lookups
    /// ([`Self::seek_exact`], [`Self::postings`], ...) keep the last-loaded
    /// blocks warm across calls instead of re-decoding them. Java gets the
    /// same effect from the caller holding a `TermsEnum`; this port's API
    /// takes a term per call, so the reuse has to live here -- one per
    /// thread slot, so concurrent searches each keep their own warm state.
    scratch: ScratchPool,
    /// The field's term n-gram index ([`crate::term_ngram`]), built the
    /// first time it is worth it: [`NGRAM_BUILD_AFTER`] regexp walks that
    /// had to cover the whole dictionary. A field never asked such a query
    /// never pays for one.
    ngram: std::sync::OnceLock<Option<Arc<crate::term_ngram::TermNgramIndex>>>,
    wide_walks: std::sync::atomic::AtomicU32,
    /// Where this field's `.doc`, `.pos` and `.pay` begin inside the buffers
    /// the segment's reader hands the postings decoder: `[0; 3]` for a
    /// segment with one postings format, and each format's offset into the
    /// concatenation for a segment `PerFieldPostingsFormat` split across
    /// several ([`BlockTreeFields::combine`]). Added to every term's decoded
    /// file pointers.
    postings_base: [u64; 3],
    /// Terms a reader view does not show ([`BlockTreeFields::without_terms`]): a lookup of one
    /// finds nothing. Set only for the `_field_names` field of a reader that hides fields.
    hidden_terms: Option<Arc<std::collections::HashSet<Vec<u8>>>>,
}

impl std::fmt::Debug for FieldTerms {
    /// Hand-written because [`SharedBytes`] is type-erased and has no
    /// `Debug`. Reports the two buffers by *length*, which is what the
    /// derived form would have printed for an `Arc<[u8]>` anyway minus a
    /// megabytes-long byte dump.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FieldTerms")
            .field("num_terms", &self.num_terms)
            .field("sum_total_term_freq", &self.sum_total_term_freq)
            .field("sum_doc_freq", &self.sum_doc_freq)
            .field("doc_count", &self.doc_count)
            .field("min_term", &self.min_term)
            .field("max_term", &self.max_term)
            .field("index_options", &self.index_options)
            .field("has_payloads", &self.has_payloads)
            .field("tim_len", &self.tim.as_ref().as_ref().len())
            .field("index", &self.index)
            .finish_non_exhaustive()
    }
}

impl Clone for FieldTerms {
    fn clone(&self) -> Self {
        Self {
            num_terms: self.num_terms,
            sum_total_term_freq: self.sum_total_term_freq,
            sum_doc_freq: self.sum_doc_freq,
            doc_count: self.doc_count,
            min_term: self.min_term.clone(),
            max_term: self.max_term.clone(),
            index_options: self.index_options,
            has_payloads: self.has_payloads,
            postings_format: self.postings_format,
            tim: Arc::clone(&self.tim),
            zstd: self.zstd.clone(),
            index: self.index.clone(),
            scratch: ScratchPool::default(),
            ngram: self.ngram.clone(),
            wide_walks: std::sync::atomic::AtomicU32::new(
                self.wide_walks.load(std::sync::atomic::Ordering::Relaxed),
            ),
            postings_base: self.postings_base,
            hidden_terms: self.hidden_terms.clone(),
        }
    }
}

impl FieldTerms {
    /// The field's `.tim` as a block loader reads it.
    #[inline]
    fn tim(&self) -> Tim<'_> {
        Tim {
            bytes: self.tim.as_ref().as_ref(),
            zstd: self.zstd.as_deref(),
        }
    }

    /// The field's `IndexOptions` (`Terms.hasPositions`/`hasOffsets` and the
    /// rest read it).
    pub fn index_options(&self) -> IndexOptions {
        self.index_options
    }

    /// Whether the field stores payloads (`Terms.hasPayloads()`).
    pub fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    /// Runs `f` against the pooled lookup state.
    fn with_scratch<T>(&self, f: impl FnOnce(&mut SegmentTermsEnum<'_>) -> Result<T>) -> Result<T> {
        // `try_lock`, never `lock`: a `BlockTreeFields` is shared by every
        // concurrently-searching thread (`lucene-ffi` hands the same `Arc` to
        // all of them), so blocking here would serialize what is otherwise a
        // read-only, embarrassingly parallel lookup. A thread that loses the
        // race just runs on its own fresh state -- same answers, only without
        // the warm blocks -- which is exactly what Java's callers get, since
        // there each thread holds its own `TermsEnum`.
        match self.scratch.slot().try_lock() {
            Ok(mut guard) => f(&mut SegmentTermsEnum {
                field: self,
                st: &mut guard,
            }),
            // Contended, or poisoned by a panic that may have left a frame
            // half-updated -- either way the pooled state is unusable for
            // this call, and a fresh one gives the same answers.
            Err(_) => {
                let mut st = EnumState {
                    current: -1,
                    ..EnumState::default()
                };
                f(&mut SegmentTermsEnum {
                    field: self,
                    st: &mut st,
                })
            }
        }
    }

    /// Whether a reader view hides `term` ([`BlockTreeFields::without_terms`]).
    #[inline]
    fn term_hidden(&self, term: &[u8]) -> bool {
        self.hidden_terms.as_ref().is_some_and(|h| h.contains(term))
    }

    /// `TermsEnum.seekExact(BytesRef)` + `docFreq()`/`totalTermFreq()`:
    /// walks the `.tip` trie to the one block that could hold `term`, loads
    /// it, and decodes only that term's stats.
    pub fn try_seek_exact(&self, term: &[u8]) -> Result<Option<TermStats>> {
        if self.term_hidden(term) {
            return Ok(None);
        }
        self.with_scratch(|ste| {
            if !ste.seek_exact(term)? {
                return Ok(None);
            }
            Ok(Some(ste.stats()?))
        })
    }

    /// [`Self::try_seek_exact`] with the error dropped: a corrupt block reads
    /// as "no such term". **Test convenience only** -- see
    /// [`TermsEnum::next`].
    pub fn seek_exact(&self, term: &[u8]) -> Option<TermStats> {
        self.try_seek_exact(term).unwrap_or(None)
    }

    /// The segment's whole `.tim`, which [`Self::term_block`] points into.
    pub fn terms_file(&self) -> &[u8] {
        self.tim.as_ref().as_ref()
    }

    /// `TermsEnum.prepareSeekExact`'s I/O hint: the first byte of the
    /// `.tim` block that could hold `term`, found from the terms index
    /// alone (no `.tim` byte is read), or `None` when no block could. A
    /// query hands these to [`lucene_store::prefetch`] for all its terms
    /// before seeking any, so their blocks arrive together.
    pub fn term_block(&self, term: &[u8]) -> Result<Option<&[u8]>> {
        if self.term_hidden(term) {
            return Ok(None);
        }
        let mut st = EnumState {
            prepare_only: true,
            ..EnumState::default()
        };
        SegmentTermsEnum {
            field: self,
            st: &mut st,
        }
        .seek_exact(term)?;
        let tim = self.terms_file();
        Ok(st
            .prepared_fp
            .and_then(|fp| tim.get(fp..fp.saturating_add(1))))
    }

    /// `seekExact` plus `decodeMetaData`, exposing just the postings file
    /// pointers (`docStartFP`/`posStartFP`/`payStartFP`/`lastPosBlockOffset`).
    ///
    /// Callers that want to decode postings should use the accessors on this
    /// type rather than driving `crate::postings` themselves; this exists for
    /// tests that need to prove a *specific* metadata field is load-bearing,
    /// which they can only do by handing the reader a doctored one.
    pub fn term_metadata(&self, term: &[u8]) -> Result<Option<TermMetadata>> {
        Ok(self.term_state(term)?.map(|(_, meta)| meta))
    }

    /// `seekExact` plus `decodeMetaData` — the stats *and* the postings file
    /// pointers, in one trie walk.
    pub(crate) fn term_state(&self, term: &[u8]) -> Result<Option<(TermStats, TermMetadata)>> {
        if self.term_hidden(term) {
            return Ok(None);
        }
        self.with_scratch(|ste| {
            if !ste.seek_exact(term)? {
                return Ok(None);
            }
            Ok(Some(ste.stats_and_meta()?))
        })
    }

    /// `TermsEnum.seekExact(term)` then `termState()`: the term's statistics
    /// and postings pointers from **one** trie walk, to open postings from
    /// with [`Self::lazy_postings_for`] -- what Lucene's `TermStates` caches
    /// per leaf so a query seeks each term once. Calling [`Self::try_seek_exact`]
    /// for the statistics and then [`Self::lazy_postings`] walks the trie
    /// twice.
    pub fn seek_term_state(&self, term: &[u8]) -> Result<Option<SeekedTerm>> {
        Ok(self
            .term_state(term)?
            .map(|(stats, meta)| SeekedTerm { stats, meta }))
    }

    /// `postings.intoBitSet(maxDoc, bitSet, 0)` for a term already found by
    /// [`Self::seek_term_state`]: every document of `term` below `end` set
    /// in `words` (bit `i` is document `i`), as `DocIdSetBuilder` gathers a
    /// multi-term query's terms. A pulsed singleton sets its one bit from the
    /// term metadata; a real posting list is read through `reuse`, the
    /// cursor of the term before (`termsEnum.postings(reuse, NONE)`), built
    /// on the first one -- it is kilobytes of block buffers, and a short
    /// prefix has thousands of terms.
    pub fn or_docs_into<'d>(
        &self,
        term: &SeekedTerm,
        doc_in: &DocInput<'d>,
        end: i32,
        words: &mut [u64],
        reuse: &mut Option<postings::LazyDocsCursor<'d>>,
    ) -> Result<()> {
        if term.stats.doc_freq == 1 {
            let doc = term.meta.singleton_doc_id;
            if (0..end).contains(&doc) {
                let i = doc as u32 as usize;
                if let Some(w) = words.get_mut(i >> 6) {
                    *w |= 1u64 << (i & 63);
                }
            }
            return Ok(());
        }
        let cursor = match reuse {
            Some(cursor) => {
                doc_in.reset_lazy_cursor(
                    cursor,
                    term.meta,
                    term.stats.doc_freq,
                    self.index_options,
                    self.has_payloads,
                    postings::PostingsFlags::DocsOnly,
                )?;
                cursor
            }
            None => reuse.insert(doc_in.lazy_cursor_with_flags(
                term.meta,
                term.stats.doc_freq,
                self.index_options,
                self.has_payloads,
                postings::PostingsFlags::DocsOnly,
            )?),
        };
        cursor.next_doc()?;
        cursor.into_window(0, end, words)?;
        Ok(())
    }

    /// [`DocInput::tail_only_first_doc`] for a term found on this field.
    pub fn tail_only_first_doc(
        &self,
        term: &SeekedTerm,
        doc_in: &DocInput<'_>,
    ) -> Result<Option<i32>> {
        Ok(doc_in.tail_only_first_doc(term.meta, term.stats.doc_freq, self.index_options)?)
    }

    /// `TermsEnum.postings(reuse, flags)`: [`Self::lazy_postings_for`] into
    /// the cursor in `reuse`, reset in place, or a new one there when it is
    /// empty. A cursor is kilobytes of block buffers; a walk over thousands of
    /// expanded terms that opened one per term spent a third of its time
    /// building and moving them. `term` must have `docFreq >= 2` -- a pulsed
    /// term has no `.doc` bytes to reset onto (`Error::Unsupported`).
    pub fn reuse_postings_for<'d, 'r>(
        &self,
        term: &SeekedTerm,
        doc_in: &DocInput<'d>,
        flags: postings::PostingsFlags,
        reuse: &'r mut Option<postings::LazyDocsCursor<'d>>,
    ) -> Result<&'r mut postings::LazyDocsCursor<'d>> {
        match reuse {
            Some(cursor) => {
                doc_in.reset_lazy_cursor(
                    cursor,
                    term.meta,
                    term.stats.doc_freq,
                    self.index_options,
                    self.has_payloads,
                    flags,
                )?;
                Ok(cursor)
            }
            None => {
                if term.stats.doc_freq <= 1 {
                    return Err(postings::Error::Unsupported(
                        "docFreq <= 1: a pulsed term has no .doc bytes to reuse a cursor on",
                    )
                    .into());
                }
                Ok(reuse.insert(doc_in.lazy_cursor_with_flags(
                    term.meta,
                    term.stats.doc_freq,
                    self.index_options,
                    self.has_payloads,
                    flags,
                )?))
            }
        }
    }

    /// The docs and freqs of a term restored from its [`SeekedTerm`]
    /// (`seekExact(term, state)`, then `postings`): no dictionary seek.
    pub fn postings_for(
        &self,
        term: &SeekedTerm,
        doc_in: Option<&DocInput<'_>>,
    ) -> Result<Postings> {
        self.postings_from(term.stats, term.meta, doc_in)
    }

    /// [`Self::postings_for`] plus the term's positions, as
    /// [`TermsEnum::try_current_postings_and_positions`] reads them.
    pub fn postings_and_positions_for(
        &self,
        term: &SeekedTerm,
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
    ) -> Result<(Postings, Vec<Vec<postings::Position>>)> {
        let docs = self.postings_from(term.stats, term.meta, doc_in)?;
        let positions = postings::read_positions(
            pos_in,
            pay_in,
            term.meta,
            &docs.freqs,
            term.stats.total_term_freq,
            self.index_options,
            self.has_payloads,
        )?;
        Ok((docs, positions))
    }

    /// `TermsEnum.postings(null, flags)` on a term already found by
    /// [`Self::seek_term_state`] on this same field: no second seek.
    pub fn lazy_postings_for<'d>(
        &self,
        term: &SeekedTerm,
        doc_in: &DocInput<'d>,
        flags: postings::PostingsFlags,
    ) -> Result<postings::LazyDocsCursor<'d>> {
        // A pulsed single-document term has no `.doc` bytes; its one document
        // lives in the term metadata. Opened as a one-document cursor so every
        // caller -- a disjunction leg, a conjunction leg -- can take it like
        // any other term.
        if term.stats.doc_freq == 1 {
            return Ok(doc_in.singleton_cursor(
                term.meta,
                term.stats.total_term_freq,
                self.index_options,
                self.has_payloads,
                flags,
            )?);
        }
        Ok(doc_in.lazy_cursor_with_flags(
            term.meta,
            term.stats.doc_freq,
            self.index_options,
            self.has_payloads,
            flags,
        )?)
    }

    /// `Terms.iterator()`-equivalent: a cursor positioned before the first
    /// term, ready for [`TermsEnum::next`]/[`TermsEnum::seek_ceil`].
    pub fn iter(&self) -> TermsEnum<'_> {
        TermsEnum::new(self)
    }

    /// `Terms.intersect(CompiledAutomaton, BytesRef)`-equivalent for a glob
    /// pattern: every term (in sorted order) matching `pattern`, with its
    /// stats.
    ///
    /// Seeks to the pattern's literal prefix and walks forward until the
    /// prefix runs out, so blocks outside the prefix range are never loaded —
    /// the same pruning real Lucene's `IntersectTermsEnum` gets from its
    /// automaton's dead states, restricted to what
    /// [`WildcardPattern::literal_prefix`] can prove. A pattern with no
    /// literal prefix (`*foo`) still walks the whole field; so does Lucene's,
    /// since a leading `.*` has no dead prefix.
    pub fn intersect<'a>(
        &'a self,
        pattern: &'a WildcardPattern,
    ) -> impl Iterator<Item = Result<(Vec<u8>, TermStats)>> + 'a {
        self.intersect_states(pattern)
            .map(|r| r.map(|(t, s)| (t, s.stats)))
    }

    /// [`Self::intersect`] yielding each match's [`SeekedTerm`], so its
    /// postings open with [`Self::lazy_postings_for`] and no second seek --
    /// what `MultiTermQuery`'s rewrite keeps in its `TermStates`.
    pub fn intersect_states<'a>(
        &'a self,
        pattern: &'a WildcardPattern,
    ) -> impl Iterator<Item = Result<(Vec<u8>, SeekedTerm)>> + 'a {
        Intersect::new(
            self,
            DfaFiltered::new(PrefixMatcher(pattern), pattern.to_dfa_cached()),
            pattern.literal_prefix(),
        )
    }

    /// `CompiledAutomaton.getTermsEnum(terms)`: by the automaton's type,
    /// nothing (`NONE`), every term (`ALL`), the one term (`SINGLE`, a
    /// `SingleTermsEnum`), or `Terms.intersect(compiled, null)` (`NORMAL`:
    /// the block-tree walk stepping the compiled byte automaton, the
    /// `IntersectTermsEnum`'s place). Every term in order, with its stats
    /// and postings pointers.
    pub fn compiled_terms<'a>(
        &'a self,
        compiled: &'a lucene_util::automaton::CompiledAutomaton,
    ) -> CompiledTermsEnum<'a> {
        use lucene_util::automaton::AutomatonType;
        match compiled.automaton_type {
            AutomatonType::NONE => CompiledTermsEnum::Empty,
            AutomatonType::ALL => CompiledTermsEnum::All(self.iter()),
            AutomatonType::SINGLE => {
                CompiledTermsEnum::Single(compiled.term.as_deref().map(|t| (self, t)))
            }
            AutomatonType::NORMAL => match compiled.byte_runnable() {
                Some(run) => CompiledTermsEnum::Normal(Box::new(DfaIntersect::new(
                    self,
                    crate::automaton::RunnableTermAutomaton(run),
                ))),
                None => CompiledTermsEnum::Empty,
            },
        }
    }

    /// `FuzzyQuery`-equivalent term matching: every term within `pattern`'s
    /// edit-distance budget, in sorted order, with its stats. Same shape as
    /// [`Self::intersect`], with `pattern`'s required `prefixLength`-byte
    /// exact prefix as the seek target.
    pub fn fuzzy_intersect<'a>(&'a self, pattern: &'a FuzzyMatch<'a>) -> FuzzyIntersect<'a> {
        let matcher = FuzzyMatcher {
            pattern,
            max_edits: pattern.max_edits(),
            last_edits: 0,
        };
        // `FuzzyTermsEnum`'s `terms.intersect(compiled, lastTerm)`: the
        // block-tree walk stepping the Levenshtein automaton, which skips
        // every block whose prefix it proves dead. The exact distance still
        // confirms each term (the budget can fall mid-walk).
        if let Some(compiled) = pattern.cached_compiled_dfa() {
            return FuzzyIntersect {
                walk: FuzzyWalk::Dfa(Box::new(DfaIntersect::new(self, compiled)), matcher),
                field: self,
                last_term: Vec::new(),
                resume_after: None,
            };
        }
        let prefix = pattern.literal_prefix().to_vec();
        FuzzyIntersect {
            walk: FuzzyWalk::Scan(Box::new(Intersect::new(
                self,
                DfaFiltered::new(matcher, None),
                prefix,
            ))),
            field: self,
            last_term: Vec::new(),
            resume_after: None,
        }
    }

    /// `RegexpQuery`-equivalent term matching, with the dead-prefix **block
    /// skip**: when [`RegexpPattern::dead_prefix_len`] proves no term sharing
    /// the current term's `k`-byte prefix can match, the walk seeks straight
    /// past that whole prefix range, which with lazy frames means the blocks
    /// under it are never loaded. That is what real Lucene's
    /// `IntersectTermsEnum` gets from a `ByteRunAutomaton` entering a dead
    /// state.
    pub fn regexp_intersect<'a>(
        &'a self,
        pattern: &'a RegexpPattern,
    ) -> impl Iterator<Item = Result<(Vec<u8>, TermStats)>> + 'a {
        self.regexp_intersect_states(pattern)
            .map(|r| r.map(|(t, s)| (t, s.stats)))
    }

    /// [`Self::regexp_intersect`] yielding each match's [`SeekedTerm`]; see
    /// [`Self::intersect_states`].
    pub fn regexp_intersect_states<'a>(
        &'a self,
        pattern: &'a RegexpPattern,
    ) -> impl Iterator<Item = Result<(Vec<u8>, SeekedTerm)>> + 'a {
        if let Some(compiled) = pattern.compiled() {
            // No literal prefix: the walk would visit the whole dictionary.
            // If the pattern forces some literal text, the n-gram index can
            // name the few terms containing it instead.
            if pattern.literal_prefix().is_empty() {
                let keys = pattern.required_grams();
                if !keys.is_empty() {
                    if let Some(index) = self.ngram_index_for_wide_query() {
                        let limit = index.num_terms() / NGRAM_MAX_CANDIDATE_SHARE;
                        if let Some(candidates) = index.candidates(&keys, limit) {
                            return RegexpIntersect::Ngram(NgramIntersect {
                                field: self,
                                index,
                                candidates: candidates.into_iter(),
                                compiled,
                            });
                        }
                    }
                }
            }
            return RegexpIntersect::Dfa(DfaIntersect::new(self, compiled));
        }
        // Too large to determinize whole: determinize only what the
        // dictionary reaches. Too large even as an NFA: the matcher alone.
        match pattern.to_nfa() {
            Some((nfa, start, end)) => RegexpIntersect::Lazy(Box::new(DfaIntersect::new(
                self,
                LazyDfa::new(nfa, start, end),
            ))),
            None => RegexpIntersect::Scan(Intersect::new(
                self,
                DfaFiltered::new(RegexpMatcher(pattern), None),
                pattern.literal_prefix(),
            )),
        }
    }

    /// Whether this field has built its term n-gram index -- for tests that
    /// must know the n-gram path was really taken. Not for production use.
    #[doc(hidden)]
    pub fn ngram_index_built(&self) -> bool {
        self.ngram.get().is_some_and(Option::is_some)
    }

    /// [`Self::regexp_intersect_states`] with the lazily determinized automaton
    /// forced, even for a pattern the eager one could be built for -- so the
    /// lazy walk can be checked against real Lucene on every fixture pattern,
    /// not only on the rare ones that need it -- with an entry budget
    /// (`automaton::MAX_LAZY_SET_ENTRIES` in production), so running out of
    /// it can be tested too. `None` when the pattern has no NFA. Not for
    /// production use.
    #[doc(hidden)]
    pub fn regexp_intersect_states_lazy<'a>(
        &'a self,
        pattern: &'a RegexpPattern,
        limit: usize,
    ) -> Option<impl Iterator<Item = Result<(Vec<u8>, SeekedTerm)>> + 'a> {
        let (nfa, start, end) = pattern.to_nfa()?;
        // `ALL` semantics come from the eager automaton's totality check; a
        // total pattern is not what this entry point exists to test.
        Some(DfaIntersect::new(
            self,
            LazyDfa::with_limit(nfa, start, end, limit),
        ))
    }

    /// `seekExact(term)` followed by `PostingsEnum` iteration
    /// (`postingsReader.postings(...)`, `DOCS_AND_FREQS` mode) — decodes the
    /// term's actual `(docID, freq)` pairs. `doc_in` is `None` for fields
    /// where a `.doc` file was never opened; passing `None` for a found term
    /// whose `docFreq > 1` is an error, since that path needs `.doc` bytes.
    pub fn postings(&self, term: &[u8], doc_in: Option<&DocInput<'_>>) -> Result<Option<Postings>> {
        self.postings_with_flags(term, doc_in, postings::PostingsFlags::Freqs)
    }

    /// [`Self::postings`] with the consumer's `PostingsEnum` flags -- Java's
    /// `TermsEnum.postings(reuse, flags)`. With
    /// [`postings::PostingsFlags::DocsOnly`] the `.doc` file's frequency
    /// blocks are stepped over rather than unpacked, and every returned
    /// frequency is `1`.
    pub fn postings_with_flags(
        &self,
        term: &[u8],
        doc_in: Option<&DocInput<'_>>,
        flags: postings::PostingsFlags,
    ) -> Result<Option<Postings>> {
        let Some((stats, meta)) = self.term_state(term)? else {
            return Ok(None);
        };
        if stats.doc_freq == 1 {
            return Ok(Some(postings::singleton_postings(
                meta,
                stats.total_term_freq,
            )?));
        }
        let doc_in = doc_in.ok_or(Error::Unsupported(
            "postings() needs an opened .doc file for docFreq > 1 terms",
        ))?;
        Ok(Some(doc_in.read_postings_with_flags(
            meta,
            stats.doc_freq,
            self.index_options,
            self.has_payloads,
            flags,
        )?))
    }

    /// `seekExact(term)` followed by opening a [`postings::LazyDocsCursor`]:
    /// the decode-on-demand sibling of [`Self::postings`] (see that method
    /// and `crate::postings`'s module doc for the shared scope/validation).
    pub fn lazy_postings<'d>(
        &self,
        term: &[u8],
        doc_in: &DocInput<'d>,
    ) -> Result<Option<postings::LazyDocsCursor<'d>>> {
        self.lazy_postings_with_flags(term, doc_in, postings::PostingsFlags::Freqs)
    }

    /// [`Self::lazy_postings`] with the consumer's `PostingsEnum` flags --
    /// the decode-on-demand sibling of [`Self::postings_with_flags`].
    pub fn lazy_postings_with_flags<'d>(
        &self,
        term: &[u8],
        doc_in: &DocInput<'d>,
        flags: postings::PostingsFlags,
    ) -> Result<Option<postings::LazyDocsCursor<'d>>> {
        let Some((stats, meta)) = self.term_state(term)? else {
            return Ok(None);
        };
        Ok(Some(doc_in.lazy_cursor_with_flags(
            meta,
            stats.doc_freq,
            self.index_options,
            self.has_payloads,
            flags,
        )?))
    }

    /// `TermsEnum.postings(reuse, PostingsEnum.POSITIONS)`: a lazy cursor over
    /// `term`'s documents that decodes a document's positions only when
    /// [`postings::PositionsCursor::next_position`] asks for them, stepping
    /// over everyone else's. `Ok(None)` when the term is absent.
    ///
    /// Needs a field indexed with positions; a pulsed singleton term
    /// (`docFreq == 1`, no `.doc` bytes) is refused the same way
    /// [`Self::lazy_postings`] refuses it.
    pub fn lazy_positions<'d>(
        &self,
        term: &[u8],
        doc_in: &DocInput<'d>,
        pos_in: &postings::PosInput<'d>,
    ) -> Result<Option<postings::PositionsCursor<'d>>> {
        let Some((stats, meta)) = self.term_state(term)? else {
            return Ok(None);
        };
        if !matches!(
            self.index_options,
            IndexOptions::DocsAndFreqsAndPositions
                | IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        ) {
            return Err(Error::Postings(postings::Error::Unsupported(
                "lazy_positions needs a field indexed with positions",
            )));
        }
        let docs = doc_in.lazy_cursor_with_flags(
            meta,
            stats.doc_freq,
            self.index_options,
            self.has_payloads,
            postings::PostingsFlags::Freqs,
        )?;
        Ok(Some(postings::PositionsCursor::new(
            docs,
            pos_in,
            meta,
            stats.total_term_freq,
            self.index_options == IndexOptions::DocsAndFreqsAndPositionsAndOffsets,
            self.has_payloads,
        )?))
    }

    /// `postings(term, doc_in)` followed by `PostingsEnum.nextPosition()`/
    /// `startOffset()`/`endOffset()`/`getPayload()` for every occurrence in
    /// every doc — needs a field with `IndexOptions::DocsAndFreqsAndPositions`
    /// or higher.
    pub fn positions(
        &self,
        term: &[u8],
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
    ) -> Result<Option<Vec<Vec<postings::Position>>>> {
        let Some((stats, meta)) = self.term_state(term)? else {
            return Ok(None);
        };
        let doc_postings = self.postings_from(stats, meta, doc_in)?;
        Ok(Some(postings::read_positions(
            pos_in,
            pay_in,
            meta,
            &doc_postings.freqs,
            stats.total_term_freq,
            self.index_options,
            self.has_payloads,
        )?))
    }

    /// The `postings`-decoding half of [`Self::postings`], for callers that
    /// already hold the term's state.
    fn postings_from(
        &self,
        stats: TermStats,
        meta: TermMetadata,
        doc_in: Option<&DocInput<'_>>,
    ) -> Result<Postings> {
        if stats.doc_freq == 1 {
            return Ok(postings::singleton_postings(meta, stats.total_term_freq)?);
        }
        let doc_in = doc_in.ok_or(Error::Unsupported(
            "postings() needs an opened .doc file for docFreq > 1 terms",
        ))?;
        Ok(doc_in.read_postings(meta, stats.doc_freq, self.index_options, self.has_payloads)?)
    }

    /// [`Self::postings_from`] into `out`, reusing its buffers, without
    /// impacts ([`postings::PostingsFlags::FreqsNoImpacts`]).
    fn postings_into(
        &self,
        stats: TermStats,
        meta: TermMetadata,
        doc_in: Option<&DocInput<'_>>,
        out: &mut Postings,
    ) -> Result<()> {
        if stats.doc_freq == 1 {
            if meta.singleton_doc_id < 0 {
                return Err(postings::Error::NoPreviousSingleton.into());
            }
            out.docs.clear();
            out.freqs.clear();
            out.level0_impacts.clear();
            out.level1_impacts.clear();
            out.docs.push(meta.singleton_doc_id);
            out.freqs.push(stats.total_term_freq as i32);
            return Ok(());
        }
        let doc_in = doc_in.ok_or(Error::Unsupported(
            "postings() needs an opened .doc file for docFreq > 1 terms",
        ))?;
        Ok(doc_in.read_postings_into(
            meta,
            stats.doc_freq,
            self.index_options,
            self.has_payloads,
            postings::PostingsFlags::FreqsNoImpacts,
            out,
        )?)
    }

    /// Positions for just the documents `wanted` names, as indices into this
    /// term's own doc list -- see [`postings::read_positions_for_docs`].
    ///
    /// `freqs` must be this term's per-document frequencies in the same order
    /// and with the same live-document filtering the caller used to build
    /// `wanted`, because the two index the same list.
    #[allow(clippy::too_many_arguments)]
    pub fn positions_for_docs(
        &self,
        term: &[u8],
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
        freqs: &[i32],
        total_term_freq: i64,
        wanted: &[usize],
    ) -> Result<(Vec<i32>, Vec<u32>)> {
        let _ = doc_in;
        let Some((_, meta)) = self.term_state(term)? else {
            // ARITH: `wanted` is a slice, so its length is at most
            // `isize::MAX` and `+ 1` cannot overflow `usize`.
            #[allow(clippy::arithmetic_side_effects)]
            let offsets = vec![0; wanted.len() + 1];
            return Ok((Vec::new(), offsets));
        };
        Ok(postings::read_positions_for_docs(
            pos_in,
            pay_in,
            meta,
            freqs,
            total_term_freq,
            self.index_options,
            self.has_payloads,
            wanted,
        )?)
    }

    /// [`Self::positions_for_docs`]'s offsets- and payloads-carrying sibling:
    /// whole [`postings::Position`] records for just the documents `wanted`
    /// names -- see [`postings::read_occurrences_for_docs`].
    ///
    /// Same `freqs`/`wanted` contract as [`Self::positions_for_docs`].
    #[allow(clippy::too_many_arguments)]
    pub fn occurrences_for_docs(
        &self,
        term: &[u8],
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
        freqs: &[i32],
        total_term_freq: i64,
        wanted: &[usize],
    ) -> Result<(Vec<postings::Position>, Vec<u32>)> {
        let _ = doc_in;
        let Some((_, meta)) = self.term_state(term)? else {
            // ARITH: `wanted` is a slice, so its length is at most
            // `isize::MAX` and `+ 1` cannot overflow `usize`.
            #[allow(clippy::arithmetic_side_effects)]
            let offsets = vec![0; wanted.len() + 1];
            return Ok((Vec::new(), offsets));
        };
        Ok(postings::read_occurrences_for_docs(
            pos_in,
            pay_in,
            meta,
            freqs,
            total_term_freq,
            self.index_options,
            self.has_payloads,
            wanted,
        )?)
    }

    /// `postings(term).advance(doc_id)` followed by `nextPosition()` /
    /// `startOffset()` / `endOffset()` / `getPayload()` for **that one
    /// document** -- Java's `PostingsOffsetStrategy.getOffsetsEnum` shape, and
    /// the accessor a highlighter wants.
    ///
    /// `Ok(None)` means the term is absent from the field, or present and not
    /// in `doc_id` -- the same "missing is not an error" convention
    /// [`Self::postings`] uses. For a well-formed segment an empty `Vec` never
    /// comes back, because a document in a term's postings has at least one
    /// occurrence of it; a `.doc` claiming `freq == 0` for it would, and that
    /// is not something the position walk rejects (only a *negative*
    /// frequency is corruption there).
    ///
    /// Costs `.doc`'s skip data down to the one 256-document block holding
    /// `doc_id`, plus the `.pos`/`.pay` blocks that actually hold its
    /// occurrences -- not, as [`Self::positions`] would, every document's,
    /// and not the term's whole doc list. See
    /// [`postings::read_occurrences_for_doc`].
    pub fn occurrences_for_doc(
        &self,
        term: &[u8],
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
        doc_id: i32,
    ) -> Result<Option<Vec<postings::Position>>> {
        let Some((stats, meta)) = self.term_state(term)? else {
            return Ok(None);
        };
        if stats.doc_freq == 1 {
            // A singleton term is pulsed into the term dictionary: no `.doc`
            // bytes exist at all, so there is no skip data to walk and the
            // whole of the term's `.pos` range belongs to that one document.
            if meta.singleton_doc_id != doc_id {
                return Ok(None);
            }
            let (occurrences, _starts) = postings::read_occurrences_for_docs(
                pos_in,
                pay_in,
                meta,
                &[stats.total_term_freq as i32],
                stats.total_term_freq,
                self.index_options,
                self.has_payloads,
                &[0],
            )?;
            return Ok(Some(occurrences));
        }
        let doc_in = doc_in.ok_or(Error::Unsupported(
            "occurrences_for_doc() needs an opened .doc file for docFreq > 1 terms",
        ))?;
        Ok(postings::read_occurrences_for_doc(
            doc_in,
            pos_in,
            pay_in,
            meta,
            stats.doc_freq,
            stats.total_term_freq,
            self.index_options,
            self.has_payloads,
            doc_id,
        )?)
    }

    /// [`FieldTerms::positions`] in the flat shape phrase matching wants: one
    /// positions array plus per-document start offsets, rather than a `Vec`
    /// per document. Returns the postings alongside, because a caller needs
    /// the doc IDs and this has already decoded them to get the freqs.
    pub fn positions_flat(
        &self,
        term: &[u8],
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
    ) -> Result<Option<FlatPositions>> {
        let Some((stats, meta)) = self.term_state(term)? else {
            return Ok(None);
        };
        let doc_postings = self.postings_from(stats, meta, doc_in)?;
        let (positions, doc_starts) = postings::read_positions_flat(
            pos_in,
            pay_in,
            meta,
            &doc_postings.freqs,
            stats.total_term_freq,
            self.index_options,
            self.has_payloads,
        )?;
        Ok(Some((doc_postings, positions, doc_starts)))
    }
}

/// All fields' term dictionaries for one segment, keyed by field name.
#[derive(Debug, Clone, Default)]
pub struct BlockTreeFields {
    fields: Vec<(String, FieldTerms)>,
}

impl BlockTreeFields {
    /// A dictionary of already-opened fields, in `.tmd` order.
    pub(crate) fn from_fields(fields: Vec<(String, FieldTerms)>) -> Self {
        BlockTreeFields { fields }
    }

    /// This dictionary with `terms` of `field` not found by an exact lookup: a reader that hides
    /// fields hides their names in `_field_names`, whose terms are the names of the fields a
    /// document has. Lookups by term see nothing of them; a walk over the field's terms still
    /// lists them, which nothing a search runs does for that field.
    #[must_use]
    pub fn without_terms(&self, field: &str, terms: std::collections::HashSet<Vec<u8>>) -> Self {
        let terms = Arc::new(terms);
        let mut out = self.clone();
        for (name, f) in &mut out.fields {
            if name == field {
                f.hidden_terms = Some(Arc::clone(&terms));
            }
        }
        out
    }

    /// This dictionary without the fields `hidden` names: what a reader that hides them sees
    /// (`Terms` for such a field is `null`). The other fields are shared, not re-read.
    #[must_use]
    pub fn without(&self, hidden: &std::collections::HashSet<String>) -> Self {
        BlockTreeFields {
            fields: self
                .fields
                .iter()
                .filter(|(name, _)| !hidden.contains(name))
                .cloned()
                .collect(),
        }
    }

    /// `PerFieldPostingsFormat.FieldsReader`: one dictionary over the
    /// dictionaries of every postings format a segment's fields were routed
    /// to (one `.tim`/`.tip`/`.tmd` per format and suffix), each group's
    /// fields decoding their postings at `base` -- where that group's
    /// `.doc`/`.pos`/`.pay` begin in the buffers the caller concatenated.
    /// A field named by two groups is corrupt ([`Error::DuplicateField`]).
    pub fn combine(groups: Vec<(BlockTreeFields, [u64; 3])>) -> Result<Self> {
        let mut fields: Vec<(String, FieldTerms)> = Vec::new();
        for (group, base) in groups {
            for (name, mut terms) in group.fields {
                if fields.iter().any(|(n, _)| *n == name) {
                    return Err(Error::DuplicateField(name));
                }
                terms.postings_base = base;
                fields.push((name, terms));
            }
        }
        Ok(BlockTreeFields { fields })
    }
}

impl FieldTerms {
    /// One field's dictionary over a `.tim` and a retired FST terms index
    /// read in place -- what [`open_shared`] builds per `.tmd` record, for
    /// `crate::backward_codecs::blocktree`.
    pub(crate) fn from_parts(
        stats: (i64, i64, i64, i32),
        min_term: Vec<u8>,
        max_term: Vec<u8>,
        field_info: &crate::field_infos::FieldInfo,
        postings_format: postings::PostingsFormat,
        tim: SharedBytes,
        index: FstTermsIndex,
    ) -> Self {
        let (num_terms, sum_total_term_freq, sum_doc_freq, doc_count) = stats;
        FieldTerms {
            num_terms,
            sum_total_term_freq,
            sum_doc_freq,
            doc_count,
            min_term,
            max_term,
            index_options: field_info.index_options,
            has_payloads: field_info.store_payloads,
            postings_format,
            tim,
            zstd: None,
            index: TermsIndex::Fst(Arc::new(index)),
            scratch: ScratchPool::default(),
            ngram: std::sync::OnceLock::new(),
            wide_walks: std::sync::atomic::AtomicU32::new(0),
            postings_base: [0; 3],
            hidden_terms: None,
        }
    }

    /// The postings format that wrote this field.
    pub fn postings_format(&self) -> postings::PostingsFormat {
        self.postings_format
    }
}

impl BlockTreeFields {
    /// Replaces field `name`'s `sumTotalTermFreq`, `sumDocFreq` and
    /// `docCount` (no-op for an absent field): for a dictionary that holds a
    /// subset of a field's terms and must report the whole field's
    /// collection statistics, such as a search's view of the writer's buffer.
    /// The number of terms stays the dictionary's own, which lookups rely on.
    pub fn set_field_stats(
        &mut self,
        name: &str,
        sum_total_term_freq: i64,
        sum_doc_freq: i64,
        doc_count: i32,
    ) {
        if let Some((_, f)) = self.fields.iter_mut().find(|(n, _)| n == name) {
            f.sum_total_term_freq = sum_total_term_freq;
            f.sum_doc_freq = sum_doc_freq;
            f.doc_count = doc_count;
        }
    }

    pub fn field(&self, name: &str) -> Option<&FieldTerms> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, f)| f)
    }

    /// Every field's name paired with its term dictionary, in the order
    /// `.tmd` listed them -- used by callers (e.g. `CheckIndex`-equivalent's
    /// postings re-derivation) that need to walk *every* field's *every*
    /// term rather than looking one up by name via [`Self::field`].
    pub fn iter_fields(&self) -> impl Iterator<Item = (&str, &FieldTerms)> {
        self.fields.iter().map(|(n, f)| (n.as_str(), f))
    }

    /// Every field's terms with their statistics and postings metadata, as
    /// a [`crate::terms_dictionary::TermsDictionaryFormat`] whose dictionary
    /// is a block tree under a wrapper (`BloomFilteringPostingsFormat`)
    /// reads its group back. `field_infos` numbers the fields.
    ///
    /// # Errors
    /// A block that does not decode, or a field `field_infos` lacks.
    pub fn dictionary_fields(
        &self,
        field_infos: &FieldInfos,
    ) -> Result<Vec<crate::postings_writer::DictionaryField>> {
        let mut out = Vec::with_capacity(self.fields.len());
        for (name, terms) in &self.fields {
            let info = field_infos.field_by_name(name).ok_or_else(|| {
                Error::Store(lucene_store::Error::Corrupted(format!(
                    "field {name} not in FieldInfos"
                )))
            })?;
            let mut field = crate::postings_writer::DictionaryField {
                field_number: info.number,
                index_options: terms.index_options(),
                has_payloads: terms.has_payloads(),
                doc_count: terms.doc_count,
                terms: Vec::new(),
            };
            let mut it = terms.iter();
            while let Some(bytes) = it.try_next_term()? {
                let bytes = bytes.to_vec();
                // The iterator is on this term already: its block holds the
                // term's state, so no second walk from the root.
                let Some(SeekedTerm { stats, meta }) = it.try_seeked_term()? else {
                    continue;
                };
                field.terms.push(crate::postings_writer::DictionaryTerm {
                    bytes,
                    doc_freq: stats.doc_freq,
                    total_term_freq: stats.total_term_freq,
                    meta,
                });
            }
            out.push(field);
        }
        Ok(out)
    }

    /// A fields producer for a segment with no postings at all (no
    /// `.tim`/`.tip`/`.tmd` files) -- e.g. a stored-fields-only segment,
    /// where `FieldInfos.hasPostings()` is false for every field. Every
    /// lookup on this behaves exactly like a real segment whose term
    /// dictionary happens to be empty.
    pub fn empty() -> Self {
        BlockTreeFields { fields: Vec::new() }
    }
}

// ---------------------------------------------------------------------------
// Automaton-free term intersection over the lazy enum.
// ---------------------------------------------------------------------------

/// What a term-intersection walk needs from a pattern: accept/reject one
/// term, and (optionally) prove that a whole prefix range cannot match.
trait TermMatcher {
    /// Whether this matcher can ever prove a prefix dead. `false` makes the
    /// whole skip path -- the virtual call, the attempt counter and the
    /// give-up test -- fold away at monomorphization, rather than being
    /// re-evaluated per non-matching term to reach a compile-time constant.
    const CAN_SKIP: bool = false;

    /// Whether [`Self::dead_prefix_len`] is free -- answered from state
    /// [`Self::matches`] already computed, as a DFA's dead state is. Then
    /// asking costs nothing, and only a skip actually taken counts toward
    /// [`Intersect`]'s give-up heuristic.
    const FREE_DEAD_PREFIX: bool = false;

    /// [`Self::FREE_DEAD_PREFIX`] for this instance: a matcher whose answer
    /// is free only when it holds some state (a DFA that may have been too
    /// large to build) overrides this.
    fn free_dead_prefix(&self) -> bool {
        Self::FREE_DEAD_PREFIX
    }

    fn matches(&mut self, term: &[u8]) -> bool;

    /// `k` such that no term starting with `term[..k]` can match, or `None`.
    /// The [`IntersectTermsEnum`-equivalent](FieldTerms::regexp_intersect)
    /// skip; a matcher that cannot prove this simply never skips.
    fn dead_prefix_len(&self, _term: &[u8]) -> Option<usize> {
        None
    }

    /// Where to seek after the non-matching `term`, when the matcher can say
    /// exactly: `Some(Some(target))` seeks to `target`, the smallest term that
    /// could still match; `Some(None)` means no later term can match. `None`
    /// leaves the choice to [`Self::dead_prefix_len`].
    fn skip_target(&self, _term: &[u8]) -> Option<Option<Vec<u8>>> {
        None
    }
}

struct PrefixMatcher<'a>(&'a WildcardPattern);
impl TermMatcher for PrefixMatcher<'_> {
    fn matches(&mut self, term: &[u8]) -> bool {
        self.0.matches(term)
    }
}

struct FuzzyMatcher<'a, 'b> {
    pattern: &'a FuzzyMatch<'b>,
    /// The budget **currently in force**, which starts at the pattern's own
    /// `maxEdits` and can only fall -- see [`FuzzyIntersect::set_max_edits`].
    max_edits: u8,
    /// The exact edit distance of the last term this matcher **accepted**, so
    /// a consumer scoring that term does not run the DP a second time for an
    /// answer already computed -- see [`FuzzyIntersect::last_edits`]. Java gets
    /// this for free: `FuzzyTermsEnum.next` computes `ed` and sets
    /// `boostAtt` in the same method.
    last_edits: usize,
}
impl TermMatcher for FuzzyMatcher<'_, '_> {
    fn matches(&mut self, term: &[u8]) -> bool {
        match self.pattern.edits_within(term, self.max_edits) {
            Some(ed) => {
                self.last_edits = ed;
                true
            }
            None => false,
        }
    }
}

/// [`FieldTerms::fuzzy_intersect`]'s walk, as a named type so a caller can
/// tighten the edit budget **while it is running**.
///
/// That is `FuzzyTermsEnum`'s `MaxNonCompetitiveBoostAttribute` channel:
/// `TopTermsRewrite.collectTerms` publishes the worst boost still in its
/// size-`maxExpansions` queue, `FuzzyTermsEnum.next` notices it changed, and
/// `bottomChanged` drops `maxEdits` for as long as no term at that distance
/// could still compete --
///
/// ```java
/// while (maxEdits > 0) {
///   float maxBoost = 1.0f - ((float) maxEdits / (float) termLength);
///   if (bottom < maxBoost || (bottom == maxBoost && termAfter == false)) break;
///   maxEdits--;
/// }
/// ```
///
/// -- then swaps in `automata[maxEdits]`, seeked back to where it was
/// (`getAutomatonEnum(maxEdits, lastTerm)`). The block-tree walk does the
/// same: it restarts over the smaller automaton, which prunes far more of
/// the dictionary, and passes over the terms up to the last one it yielded
/// (`resume_after`) -- a restart from the root standing in for the seek,
/// which a smaller automaton keeps cheap. The forward scan (no automaton)
/// needs no swap: tightening the exact test takes effect on the next term
/// and the position is already correct.
pub struct FuzzyIntersect<'a> {
    walk: FuzzyWalk<'a>,
    field: &'a FieldTerms,
    /// The term this walk last yielded: `lastTerm`.
    last_term: Vec<u8>,
    /// After an automaton swap, the terms at or before this one were already
    /// walked under the larger automaton and are passed over.
    resume_after: Option<Vec<u8>>,
}

/// How a [`FuzzyIntersect`] finds its candidates.
enum FuzzyWalk<'a> {
    /// The block-tree intersection with the Levenshtein DFA, each term it
    /// accepts confirmed by the exact matcher.
    Dfa(
        Box<DfaIntersect<'a, std::sync::Arc<crate::automaton::CompiledDfa>>>,
        FuzzyMatcher<'a, 'a>,
    ),
    /// A pattern too large for a DFA: the forward scan, every term tested.
    /// Boxed like the DFA walk: both carry a terms enum's frame stack.
    Scan(Box<Intersect<'a, DfaFiltered<FuzzyMatcher<'a, 'a>>>>),
}

impl<'a> FuzzyIntersect<'a> {
    fn matcher(&self) -> &FuzzyMatcher<'a, 'a> {
        match &self.walk {
            FuzzyWalk::Dfa(_, m) => m,
            FuzzyWalk::Scan(scan) => &scan.matcher.inner,
        }
    }

    fn matcher_mut(&mut self) -> &mut FuzzyMatcher<'a, 'a> {
        match &mut self.walk {
            FuzzyWalk::Dfa(_, m) => m,
            FuzzyWalk::Scan(scan) => &mut scan.matcher.inner,
        }
    }

    /// `bottomChanged`'s `actualEnum = getAutomatonEnum(maxEdits, lastTerm)`.
    ///
    /// Only ever lowers: a budget at or above the one in force is ignored,
    /// because widening mid-scan would make the walk yield terms it had
    /// already rejected further back, which no caller could interpret.
    pub fn set_max_edits(&mut self, max_edits: u8) {
        let m = self.matcher_mut();
        if max_edits >= m.max_edits {
            return;
        }
        m.max_edits = max_edits;
        let pattern = m.pattern;
        if let FuzzyWalk::Dfa(walk, _) = &mut self.walk {
            if let Some(compiled) = pattern.cached_compiled_dfa_within(max_edits) {
                **walk = DfaIntersect::new(self.field, compiled);
                self.resume_after = Some(self.last_term.clone());
            }
        }
    }

    /// The budget currently in force.
    pub fn max_edits(&self) -> u8 {
        self.matcher().max_edits
    }

    /// The exact edit distance of the term this walk last yielded -- what
    /// `FuzzyTermsEnum.next` has in hand when it sets `BoostAttribute`. Pair
    /// it with [`crate::fuzzy::FuzzyMatch::boost_from_edits`] rather than
    /// re-deriving the distance from the term.
    ///
    /// Meaningless before the first yielded term (it reads `0`, which is also
    /// a legitimate distance), which is why it is not an `Option`: every
    /// caller reads it immediately after a `Some` from [`Iterator::next`].
    pub fn last_edits(&self) -> usize {
        self.matcher().last_edits
    }
}

impl Iterator for FuzzyIntersect<'_> {
    type Item = Result<(Vec<u8>, TermStats)>;

    fn next(&mut self) -> Option<Self::Item> {
        match &mut self.walk {
            FuzzyWalk::Dfa(walk, matcher) => loop {
                match walk.next()? {
                    Ok((term, seeked)) => {
                        if let Some(after) = &self.resume_after {
                            if term.as_slice() <= after.as_slice() {
                                continue;
                            }
                            self.resume_after = None;
                        }
                        if matcher.matches(&term) {
                            self.last_term.clear();
                            self.last_term.extend_from_slice(&term);
                            return Some(Ok((term, seeked.stats)));
                        }
                    }
                    Err(e) => return Some(Err(e)),
                }
            },
            FuzzyWalk::Scan(scan) => scan.next().map(|r| r.map(|(t, s)| (t, s.stats))),
        }
    }
}

struct RegexpMatcher<'a>(&'a RegexpPattern);
impl TermMatcher for RegexpMatcher<'_> {
    const CAN_SKIP: bool = true;

    fn matches(&mut self, term: &[u8]) -> bool {
        self.0.matches(term)
    }
    fn dead_prefix_len(&self, term: &[u8]) -> Option<usize> {
        self.0.dead_prefix_len(term)
    }
}

/// A [`TermMatcher`] with a superset [`ByteDfa`] in front of it -- what
/// `CompiledAutomaton` gives `IntersectTermsEnum`.
///
/// The DFA is walked incrementally over the sorted terms ([`DfaWalker`]), so
/// each term costs only the bytes it does not share with the one before. A
/// term the DFA rejects is rejected without asking the exact matcher; a term
/// it accepts is confirmed by the exact matcher, which stays the authority
/// (the DFA may accept a superset -- see `crate::automaton`). And a prefix
/// the DFA proves dead becomes the skip [`Intersect`] seeks past, exactly and
/// for free, where the automaton-free matchers had to search for it.
///
/// `dfa` is `None` when the pattern could not be determinized within
/// `automaton`'s limits; the wrapper then defers to the exact matcher alone.
struct DfaFiltered<M: TermMatcher> {
    inner: M,
    dfa: Option<(std::sync::Arc<ByteDfa>, DfaWalker)>,
    /// The dead-prefix length [`Self::matches`] found for the term it was
    /// last handed, for [`TermMatcher::dead_prefix_len`] on the same term.
    last_dead: Option<usize>,
}

impl<M: TermMatcher> DfaFiltered<M> {
    fn new(inner: M, dfa: Option<std::sync::Arc<ByteDfa>>) -> Self {
        Self {
            inner,
            dfa: dfa.map(|d| {
                let w = DfaWalker::new(&d);
                (d, w)
            }),
            last_dead: None,
        }
    }
}

impl<M: TermMatcher> TermMatcher for DfaFiltered<M> {
    const CAN_SKIP: bool = true;
    const FREE_DEAD_PREFIX: bool = true;

    /// Free only with a DFA: without one (the pattern was too large to
    /// determinize) the question goes to the inner matcher's own
    /// `dead_prefix_len`, which is not free and must count toward the walk's
    /// give-up heuristic like any other matcher's.
    fn free_dead_prefix(&self) -> bool {
        self.dfa.is_some()
    }

    fn matches(&mut self, term: &[u8]) -> bool {
        if let Some((dfa, walker)) = self.dfa.as_mut() {
            match walker.feed(dfa, term) {
                Verdict::DeadAt(k) => {
                    self.last_dead = Some(k);
                    return false;
                }
                Verdict::Reject => {
                    self.last_dead = None;
                    return false;
                }
                Verdict::Accept => self.last_dead = None,
            }
        }
        self.inner.matches(term)
    }

    fn dead_prefix_len(&self, term: &[u8]) -> Option<usize> {
        if self.dfa.is_some() {
            self.last_dead
        } else {
            self.inner.dead_prefix_len(term)
        }
    }

    fn skip_target(&self, term: &[u8]) -> Option<Option<Vec<u8>>> {
        let (dfa, walker) = self.dfa.as_ref()?;
        let dead_at = self.last_dead?;
        Some(walker.next_live_after(dfa, term, dead_at))
    }
}
/// Attempts before [`Intersect`] judges whether skipping is paying.
const SKIP_WARMUP: u32 = 128;

/// Entries a skip must save on average to be worth its cost. A skip is a
/// fresh `seekCeil` -- a trie descent plus, usually, a block load -- where
/// staying put is one `next()`.
const SKIP_MIN_SAVING: u64 = 16;

/// Credit given to a skip that left the block it started in: it provably
/// avoided loading at least the rest of that block, and typically several
/// whole blocks, which no entry count can express.
const SKIP_BLOCK_CREDIT: u64 = 64;

/// The saving [`SKIP_WARMUP`] attempts have to have produced for the skip
/// heuristic to stay on. A compile-time constant, so the multiplication is
/// not arithmetic on any value that came off disk.
// ARITH: both factors are literals; `128 * 16 = 2048` is evaluated at compile
// time and cannot overflow `u64`.
#[allow(clippy::arithmetic_side_effects)]
const SKIP_WARMUP_BUDGET: u64 = SKIP_WARMUP as u64 * SKIP_MIN_SAVING;

/// The shared body of [`FieldTerms::intersect`]/[`FieldTerms::fuzzy_intersect`]/
/// [`FieldTerms::regexp_intersect`]: seek to the literal prefix, walk forward
/// while the prefix holds, and skip provably-dead prefix ranges with a
/// `seekCeil` that never loads the blocks it jumps over.
struct Intersect<'a, M: TermMatcher> {
    enum_: TermsEnum<'a>,
    matcher: M,
    prefix: Vec<u8>,
    started: bool,
    done: bool,
    skip_attempts: u32,
    skipped: u64,
    skip_enabled: bool,
}

impl<'a, M: TermMatcher> Intersect<'a, M> {
    fn new(field: &'a FieldTerms, matcher: M, prefix: Vec<u8>) -> Self {
        Self {
            enum_: field.iter(),
            matcher,
            prefix,
            started: false,
            done: false,
            skip_attempts: 0,
            skipped: 0,
            skip_enabled: true,
        }
    }

    /// Positions on the first candidate; `false` when there is none.
    fn start(&mut self) -> Result<bool> {
        self.started = true;
        if self.prefix.is_empty() {
            return Ok(self.enum_.try_next_term()?.is_some());
        }
        Ok(match self.enum_.try_seek_ceil(&self.prefix)? {
            SeekStatus::Found | SeekStatus::NotFound => true,
            SeekStatus::End => false,
        })
    }

    /// The body of [`Iterator::next`], with the error channel Java's
    /// `IntersectTermsEnum.next()` has (it throws `IOException`).
    fn next_result(&mut self) -> Result<Option<(Vec<u8>, SeekedTerm)>> {
        if self.done {
            return Ok(None);
        }
        if !self.started && !self.start()? {
            self.done = true;
            return Ok(None);
        }
        loop {
            // `term()`, not `try_current()`: `IntersectTermsEnum` walks on the
            // term bytes alone and only the terms it *yields* need stats, so
            // decoding `docFreq`/`totalTermFreq`/`TermMetadata` here would
            // charge every rejected term for numbers this loop throws away.
            let Some(term) = self.enum_.term() else {
                self.done = true;
                return Ok(None);
            };
            // A byte loop, not `starts_with`: the prefix is a byte or two and
            // this runs for every term the walk visits, where a `memcmp` call
            // per term was a tenth of a fuzzy expansion.
            let n = self.prefix.len();
            if !term
                .get(..n)
                .is_some_and(|h| h.iter().zip(&self.prefix).all(|(a, b)| a == b))
            {
                self.done = true;
                return Ok(None);
            }
            if self.matcher.matches(term) {
                let bytes = term.to_vec();
                // Only now, for a term that is actually being handed out.
                //
                // `term()` was `Some`, so `try_stats()` cannot be `None` --
                // both gate on the same `st.on_term` and nothing between them
                // moves the cursor. Written as an `else` rather than an
                // `expect` anyway: this is a decode path reachable from the
                // JVM through a wildcard/regexp/fuzzy expansion, and a panic
                // must never cross that boundary (AGENTS.md invariant 5). The
                // unreachable branch reads as end-of-terms, which is what the
                // `term()` arm above already does.
                let Some(seeked) = self.enum_.try_seeked_term()? else {
                    self.done = true;
                    return Ok(None);
                };
                if self.enum_.try_next_term()?.is_none() {
                    self.done = true;
                }
                return Ok(Some((bytes, seeked)));
            }

            // Not a match. Either step to the next term, or -- when the
            // pattern proves this whole prefix range is dead -- seek past it.
            if !M::CAN_SKIP || !self.skip_enabled {
                if self.enum_.try_next_term()?.is_none() {
                    self.done = true;
                    return Ok(None);
                }
                continue;
            }

            // Asking the matcher is itself the cost the give-up below
            // measures: a pattern whose language is prefix-closed (`cat.*`,
            // `t.*99`) never yields a dead prefix, so every one of these
            // calls is pure loss. Count the *question*, not just the jumps.
            // A matcher that can name the next viable term (a DFA's
            // `nextString`) is asked first; `Some(None)` means no later term can
            // match at all, which ends the walk outright.
            let target = match self.matcher.skip_target(term) {
                Some(None) => {
                    self.done = true;
                    return Ok(None);
                }
                Some(Some(target)) => Some(target),
                None => self
                    .matcher
                    .dead_prefix_len(term)
                    .filter(|&k| k <= term.len())
                    .and_then(|k| prefix_upper_bound(&term[..k])),
            };
            // A heuristic counter: `saturating_add` is the honest semantics
            // here, because a saturated attempt count can only leave the
            // skip heuristic permanently on or off -- it can never change
            // which terms the intersection yields.
            if target.is_some() || !self.matcher.free_dead_prefix() {
                self.skip_attempts = self.skip_attempts.saturating_add(1);
            }

            match target {
                None => {
                    if self.enum_.try_next_term()?.is_none() {
                        self.done = true;
                        return Ok(None);
                    }
                }
                Some(upper) => {
                    let before = self.enum_.ste().position();
                    match self.enum_.try_seek_ceil(&upper)? {
                        SeekStatus::Found | SeekStatus::NotFound => {}
                        SeekStatus::End => {
                            self.done = true;
                            return Ok(None);
                        }
                    }
                    let after = self.enum_.ste().position();
                    let credit = if before.0 == after.0 && before.1 == after.1 {
                        // Same block: the entry cursor says exactly how many
                        // terms the jump stepped over. Both operands are
                        // `next_ent` values, bounded only by a block's
                        // `entCount`, so their difference can leave `i32`;
                        // `saturating_sub` is honest for a counter that only
                        // ever steers the heuristic.
                        let stepped = after.2.saturating_sub(before.2).max(1);
                        // ARITH: `stepped >= 1` after `.max(1)` and is
                        // non-negative, so the cast is exact and `- 1` cannot
                        // underflow.
                        #[allow(clippy::arithmetic_side_effects)]
                        let credit = stepped as u64 - 1;
                        credit
                    } else {
                        // A different block: the jump provably avoided
                        // loading at least the rest of the one it started in,
                        // which no entry count can express.
                        SKIP_BLOCK_CREDIT
                    };
                    self.skipped = self.skipped.saturating_add(credit);
                }
            }

            if self.skip_attempts == SKIP_WARMUP && self.skipped < SKIP_WARMUP_BUDGET {
                self.skip_enabled = false;
            }
        }
    }
}

impl<M: TermMatcher> Iterator for Intersect<'_, M> {
    /// A corrupt `.tim` block ends the walk **with an error**, as
    /// `IntersectTermsEnum.next()`'s `IOException` does, rather than by
    /// quietly reporting fewer matching terms: a truncated term expansion is a
    /// wrong hit set, and every consumer of these iterators is inside a
    /// `Result`-returning function already.
    ///
    /// Each match comes with its [`SeekedTerm`] -- the walk decoded the
    /// postings pointers anyway -- so a caller can open its postings with
    /// [`FieldTerms::lazy_postings_for`] instead of seeking the term again.
    type Item = Result<(Vec<u8>, SeekedTerm)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_result() {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => None,
            Err(e) => {
                // One error per walk: the cursor's frame stack is not
                // recoverable, and Java's enumeration is likewise dead once
                // `loadBlock` has thrown.
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// `IntersectTermsEnum`: the term dictionary walked in lockstep with a
/// [`ByteDfa`], block by block.
///
/// Each frame of the walk remembers the automaton state its block's shared
/// prefix leads to, and every entry is judged by running only its *suffix*
/// bytes, in place in the block, from that state. A term entry is yielded
/// when its state accepts; a sub-block entry is descended into only when its
/// state is still live -- a dead one is never loaded, which is how a pattern
/// with no literal prefix (`[a-z][0-9]{2}`) still skips most of the
/// dictionary. Nothing is copied into the term buffer until an entry is
/// kept, and no term is compared with the one before it.
///
/// This replaces [`Intersect`]'s forward scan for regexp, which fed every
/// term (all of its bytes, found by comparing it with the previous one)
/// through a [`DfaWalker`] and could only skip by re-seeking from the root.
pub struct DfaIntersect<'a, A: TermAutomaton> {
    field: &'a FieldTerms,
    st: EnumState,
    compiled: A,
    /// `states[ord]`: the automaton state after frame `ord`'s block prefix.
    states: Vec<u32>,
    /// `nodes[ord]`: the index trie node for frame `ord`'s prefix, when the
    /// walk found it -- the source of the frame's floor data, which is what
    /// lets a floor block whose lead bytes the automaton cannot take be
    /// skipped without loading it.
    nodes: Vec<Option<IndexNode>>,
    /// `rest_dead[ord]`: an entry of frame `ord` sorted past every byte its
    /// state can take, so the rest of the frame -- this block and its later
    /// floor blocks -- is skipped.
    rest_dead: Vec<bool>,
    /// Every block the walk has loaded, by file pointer. A well-formed
    /// dictionary is a tree, so an honest walk loads each block once; a
    /// corrupt term index can point its floor skips and sub-block descents
    /// back into blocks already walked and fan out without end
    /// (`check_index`'s corrupted-`.tip` sweep found this). A second load of
    /// the same block is reported as corruption -- exactly, with no size
    /// estimate to get wrong on compressed blocks.
    visited: lucene_util::fx_hash::FxHashSet<usize>,
    started: bool,
    done: bool,
}

impl<'a, A: TermAutomaton> DfaIntersect<'a, A> {
    /// `compiled` carries `CompiledAutomaton`'s `AUTOMATON_TYPE.ALL` swap
    /// already made ([`CompiledDfa::new`]): `.*` and `@` enumerate the whole
    /// dictionary, ill-formed UTF-8 terms included.
    fn new(field: &'a FieldTerms, compiled: A) -> Self {
        Self {
            field,
            st: EnumState {
                current: -1,
                ..EnumState::default()
            },
            compiled,
            states: Vec::new(),
            nodes: Vec::new(),
            rest_dead: Vec::new(),
            visited: Default::default(),
            started: false,
            done: false,
        }
    }

    fn ste(&mut self) -> SegmentTermsEnum<'_> {
        SegmentTermsEnum {
            field: self.field,
            st: &mut self.st,
        }
    }

    /// Records the block just loaded into frame `ord`; see [`Self::visited`].
    fn record_load(&mut self, ord: usize) -> Result<()> {
        if !self.visited.insert(self.st.stack[ord].fp) {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "term dictionary walk reached the same block twice: the term index points back into blocks already walked"
                    .into(),
            )));
        }
        Ok(())
    }

    fn set_state(&mut self, ord: usize, state: u32, node: Option<IndexNode>) {
        if self.states.len() <= ord {
            self.states.resize(ord.saturating_add(1), DEAD);
            self.nodes.resize(ord.saturating_add(1), None);
            self.rest_dead.resize(ord.saturating_add(1), false);
        }
        self.states[ord] = state;
        self.nodes[ord] = node;
        self.rest_dead[ord] = false;
    }

    /// The floor block of frame `ord` that holds the first lead byte at or
    /// after `from` the automaton can still take: `IntersectTermsEnumFrame`'s
    /// floor skip. `false` when no later floor block can hold one. Only a
    /// frame pushed from its trie node carries floor data; any other steps
    /// through its floor blocks in order.
    fn seek_live_floor(&mut self, ord: usize, from: u32) -> Result<bool> {
        let loaded = self.seek_live_floor_unbudgeted(ord, from)?;
        if loaded {
            self.record_load(ord)?;
        }
        Ok(loaded)
    }

    fn seek_live_floor_unbudgeted(&mut self, ord: usize, from: u32) -> Result<bool> {
        let index = self.field.index.trie_bytes();
        let tim = self.field.tim();
        let s = self.states[ord];
        let f = &mut self.st.stack[ord];
        if !f.is_floor {
            if from == 0 {
                f.load_block(tim)?;
            } else {
                f.load_next_floor_block(tim)?;
            }
            return Ok(true);
        }
        let Some(label) = self
            .compiled
            .first_live_from(s, from)
            .ok_or_else(too_complex)?
        else {
            // Nothing further, but the first block still holds the term that
            // equals the prefix itself.
            if from == 0 && self.compiled.is_accept(s) {
                f.load_block(tim)?;
                return Ok(true);
            }
            return Ok(false);
        };
        let label = if from == 0 && self.compiled.is_accept(s) {
            0
        } else {
            u32::from(label)
        };
        if label >= f.next_floor_label || from > 0 {
            f.scan_to_floor_label(index, label)?;
            if from > 0 && f.next_ent != -1 {
                // The scan stayed on the block just finished: no floor
                // block after it holds that byte.
                return Ok(false);
            }
        }
        f.load_block(tim)?;
        Ok(true)
    }

    // ARITH: `st.current` is decremented only after the `ord == 0` return,
    // so it stays `>= 0` while the walk runs; `prefix_length +
    // suffix_length` is bounded as in [`Frame::fill_term`].
    #[allow(clippy::arithmetic_side_effects)]
    fn next_result(&mut self) -> Result<Option<(Vec<u8>, SeekedTerm)>> {
        if self.done {
            return Ok(None);
        }
        if !self.started {
            self.started = true;
            let start = self.compiled.start();
            if start == DEAD || self.field.num_terms == 0 {
                self.done = true;
                return Ok(None);
            }
            let mut ste = self.ste();
            let root = ste.root()?;
            ste.push_frame_node(&root, 0)?;
            self.set_state(0, start, Some(root));
            if !self.seek_live_floor(0, 0)? {
                self.done = true;
                return Ok(None);
            }
        }
        loop {
            let ord = self.st.current.max(0) as usize;
            let f = &mut self.st.stack[ord];
            if f.next_ent == f.ent_count as i32 || self.rest_dead[ord] {
                if !f.is_last_in_floor && !self.rest_dead[ord] {
                    let from = f.next_floor_label;
                    if self.seek_live_floor(ord, from)? {
                        continue;
                    }
                }
                if ord == 0 {
                    self.done = true;
                    self.st.on_term = false;
                    return Ok(None);
                }
                // The parent's cursor is already just past the sub-block
                // entry that led here: a depth-first walk never moves it.
                self.st.current -= 1;
                continue;
            }
            let is_sub = f.next_entry()?;
            let mut s = self.states[ord];
            if !is_sub && self.compiled.utf8_total(s) {
                if std::str::from_utf8(f.suffix()).is_err() {
                    continue;
                }
                f.fill_term(&mut self.st.term);
                self.st.on_term = true;
                let (stats, meta) = self.ste().stats_and_meta()?;
                return Ok(Some((
                    self.st.term.get().to_vec(),
                    SeekedTerm { stats, meta },
                )));
            }
            // Entries are sorted, so one whose first suffix byte is past the
            // last byte this block's prefix state can take ends the block --
            // `IntersectTermsEnum`'s jump to `entCount` -- and with it every
            // floor block after this one.
            if let Some(&first) = f.suffix().first() {
                if self
                    .compiled
                    .last_live_byte(s)
                    .ok_or_else(too_complex)?
                    .is_none_or(|max| first > max)
                {
                    self.rest_dead[ord] = true;
                    continue;
                }
            }
            for &b in f.suffix() {
                s = self.compiled.step(s, b).ok_or_else(too_complex)?;
                if s == DEAD {
                    break;
                }
            }
            if s == DEAD {
                continue;
            }
            if is_sub {
                let (sub_fp, length) = (f.last_sub_fp, f.prefix_length + f.suffix_length);
                f.fill_term(&mut self.st.term);
                // The child's trie node, reached from this frame's by the
                // entry's suffix bytes: it carries the floor data. Accepted
                // only when it names the very block the entry points at.
                let terms_index = &self.field.index;
                let index = terms_index.trie_bytes();
                let mut node = self.nodes.get(ord).cloned().flatten();
                for &b in &self.st.term.get()[self.st.stack[ord].prefix_length..] {
                    node = match node {
                        Some(n) => terms_index.child(index, &n, b)?,
                        None => None,
                    };
                }
                let node = match node {
                    Some(n) if terms_index.block_fp(&n)? == Some(sub_fp as u64) => Some(n),
                    _ => None,
                };
                let mut ste = self.ste();
                match &node {
                    Some(n) => ste.push_frame_node(n, length)?,
                    None => ste.push_next_frame(sub_fp, length)?,
                }
                let child = self.st.current.max(0) as usize;
                self.set_state(child, s, node);
                if !self.seek_live_floor(child, 0)? {
                    self.st.current -= 1;
                }
                continue;
            }
            if !self.compiled.is_accept(s) {
                continue;
            }
            f.fill_term(&mut self.st.term);
            self.st.on_term = true;
            let (stats, meta) = self.ste().stats_and_meta()?;
            return Ok(Some((
                self.st.term.get().to_vec(),
                SeekedTerm { stats, meta },
            )));
        }
    }
}

/// What [`FieldTerms::compiled_terms`] returns: `TermsEnum.EMPTY`, the
/// field's own `TermsEnum`, a `SingleTermsEnum`, or the automaton walk.
pub enum CompiledTermsEnum<'a> {
    Empty,
    All(TermsEnum<'a>),
    /// The field and the one term, until it is looked up.
    Single(Option<(&'a FieldTerms, &'a [u8])>),
    Normal(Box<DfaIntersect<'a, crate::automaton::RunnableTermAutomaton<'a>>>),
}

impl Iterator for CompiledTermsEnum<'_> {
    /// A corrupt block ends the enumeration with an error.
    type Item = Result<(Vec<u8>, SeekedTerm)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            CompiledTermsEnum::Empty => None,
            CompiledTermsEnum::All(te) => {
                let term = match te.try_next_term() {
                    Ok(Some(t)) => t.to_vec(),
                    Ok(None) => return None,
                    Err(e) => {
                        *self = CompiledTermsEnum::Empty;
                        return Some(Err(e));
                    }
                };
                match te.try_seeked_term() {
                    Ok(Some(s)) => Some(Ok((term, s))),
                    Ok(None) => None,
                    Err(e) => {
                        *self = CompiledTermsEnum::Empty;
                        Some(Err(e))
                    }
                }
            }
            CompiledTermsEnum::Single(pending) => {
                let (field, term) = pending.take()?;
                match field.seek_term_state(term) {
                    Ok(Some(s)) => Some(Ok((term.to_vec(), s))),
                    Ok(None) => None,
                    Err(e) => Some(Err(e)),
                }
            }
            CompiledTermsEnum::Normal(walk) => walk.next(),
        }
    }
}

impl<A: TermAutomaton> Iterator for DfaIntersect<'_, A> {
    /// As [`Intersect`]'s: a corrupt block ends the walk with an error.
    type Item = Result<(Vec<u8>, SeekedTerm)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.next_result() {
            Ok(Some(item)) => Some(Ok(item)),
            Ok(None) => None,
            Err(e) => {
                self.done = true;
                Some(Err(e))
            }
        }
    }
}

/// Dictionary-wide regexp walks a field serves before it builds its term
/// n-gram index: a one-off query never pays for the build, a field that keeps
/// being asked "contains" questions does, once.
const NGRAM_BUILD_AFTER: u32 = 2;

/// The n-gram path is taken only when its candidates are at most this
/// fraction (`1 / n`) of the dictionary; past that, verifying candidates one
/// by one (each a dictionary lookup) costs more than walking.
const NGRAM_MAX_CANDIDATE_SHARE: usize = 8;

/// Fields with more terms than this never build an n-gram index. It costs
/// about 30 bytes per term (6 MB for the benchmark's 200 000-term field),
/// outside the JVM's view and its circuit breakers, so it is kept to a size
/// that is small next to the index it serves: about 60 MB at the cap.
const NGRAM_MAX_TERMS: i64 = 2_000_000;

impl FieldTerms {
    /// The field's n-gram index, if one is built or this query is the one
    /// that makes it worth building ([`NGRAM_BUILD_AFTER`]); `None`
    /// otherwise, and for a field too large or unreadable to index.
    fn ngram_index_for_wide_query(&self) -> Option<Arc<crate::term_ngram::TermNgramIndex>> {
        if let Some(built) = self.ngram.get() {
            return built.clone();
        }
        let seen = self
            .wide_walks
            .fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if seen < NGRAM_BUILD_AFTER || self.num_terms > NGRAM_MAX_TERMS {
            return None;
        }
        self.ngram
            .get_or_init(|| {
                // Streamed: each term goes straight from the dictionary into
                // the builder, with no intermediate copy of the dictionary.
                let mut e = self.iter();
                let mut b = crate::term_ngram::TermNgramBuilder::default();
                loop {
                    match e.try_next_term() {
                        Ok(Some(t)) => b.push(t)?,
                        Ok(None) => break,
                        Err(_) => return None,
                    }
                }
                b.finish().map(Arc::new)
            })
            .clone()
    }
}

/// [`FieldTerms::regexp_intersect_states`] answered from the term n-gram
/// index: the terms containing every gram the pattern forces, in dictionary
/// order, each confirmed by the automaton and then looked up for its state.
struct NgramIntersect<'a> {
    field: &'a FieldTerms,
    index: Arc<crate::term_ngram::TermNgramIndex>,
    candidates: std::vec::IntoIter<u32>,
    compiled: Arc<CompiledDfa>,
}

impl Iterator for NgramIntersect<'_> {
    type Item = Result<(Vec<u8>, SeekedTerm)>;

    fn next(&mut self) -> Option<Self::Item> {
        for ord in self.candidates.by_ref() {
            let term = self.index.term(ord);
            if !self.compiled.dfa.run(term) {
                continue;
            }
            return match self.field.seek_term_state(term) {
                Ok(Some(seeked)) => Some(Ok((term.to_vec(), seeked))),
                // The index was built from this dictionary, so the term is
                // there; a miss can only be corruption.
                Ok(None) => Some(Err(Error::Store(lucene_store::Error::Corrupted(
                    "term from the n-gram index is missing from its dictionary".into(),
                )))),
                Err(e) => Some(Err(e)),
            };
        }
        None
    }
}

/// A lazy automaton ran out of states: the pattern is too complex to run,
/// as `TooComplexToDeterminizeException` reports it in Lucene.
fn too_complex() -> Error {
    Error::Unsupported("regexp too complex to run: its automaton exceeded the lazy state budget")
}

/// [`FieldTerms::regexp_intersect_states`]'s walk: the lockstep automaton
/// walk when the pattern determinizes, the matcher-driven scan when it is
/// too large to.
enum RegexpIntersect<'a> {
    Dfa(DfaIntersect<'a, std::sync::Arc<CompiledDfa>>),
    Lazy(Box<DfaIntersect<'a, LazyDfa>>),
    Ngram(NgramIntersect<'a>),
    Scan(Intersect<'a, DfaFiltered<RegexpMatcher<'a>>>),
}

impl Iterator for RegexpIntersect<'_> {
    type Item = Result<(Vec<u8>, SeekedTerm)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            RegexpIntersect::Dfa(w) => w.next(),
            RegexpIntersect::Lazy(w) => w.next(),
            RegexpIntersect::Ngram(w) => w.next(),
            RegexpIntersect::Scan(w) => w.next(),
        }
    }
}

/// The exclusive upper bound of the sorted range whose bytes all start with
/// `prefix`: `prefix` with its last byte incremented (dropping any trailing
/// `0xFF` bytes first, since those can't be incremented in place -- e.g.
/// `[0x61, 0xFF]` -> `[0x62]`). `None` when `prefix` is empty (no useful
/// bound -- every term is in range) or entirely `0xFF` bytes (no finite byte
/// string is an upper bound).
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut upper = prefix.to_vec();
    while let Some(&last) = upper.last() {
        if last == 0xFF {
            upper.pop();
        } else {
            // ARITH: the `last == 0xFF` branch above took every byte that
            // cannot be incremented, so `last <= 0xFE` here.
            #[allow(clippy::arithmetic_side_effects)]
            {
                *upper.last_mut().unwrap() += 1;
            }
            return Some(upper);
        }
    }
    None
}

/// `Lucene103BlockTreeTermsReader.readBytesRef` -- the `.tmd`'s
/// `minTerm`/`maxTerm`.
///
/// Java sizes `new BytesRef(numBytes)` straight from the vint and lets a
/// corrupt one raise `OutOfMemoryError`, which a caller can catch. Here the
/// same `vec![0u8; len]` would be an *abort* on allocation failure, which
/// `catch_unwind` at the FFI boundary cannot intercept -- so the length is
/// bounded by the bytes actually left in the `.tmd` first. That cannot reject
/// a file Lucene wrote: `readBytesRef` is immediately followed by
/// `readBytes(bytes, 0, numBytes)`, so a well-formed record always has at
/// least `numBytes` left.
pub(crate) fn read_bytes_ref(input: &mut SliceInput) -> Result<Vec<u8>> {
    let len = input.read_vint()?;
    if len < 0 {
        return Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "invalid bytes length: {len}"
        ))));
    }
    let len = len as usize;
    if len > input.remaining() {
        return Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "bytes length {len} exceeds {} remaining bytes",
            input.remaining()
        ))));
    }
    let mut buf = vec![0u8; len];
    input.read_bytes(&mut buf)?;
    Ok(buf)
}

/// Reads the `sumTotalTermFreq`/`sumDocFreq` pair, aliasing the single value
/// written when `IndexOptions::Docs` (frequencies aren't stored at all, so
/// `sumTotalTermFreq == sumDocFreq` and only one vlong is on the wire) —
/// mirrors `Lucene103BlockTreeTermsReader`'s constructor exactly.
pub(crate) fn read_freq_pair(
    input: &mut SliceInput,
    index_options: IndexOptions,
) -> Result<(i64, i64)> {
    let first = input.read_vlong()?;
    if index_options == IndexOptions::Docs {
        Ok((first, first))
    } else {
        let sum_doc_freq = input.read_vlong()?;
        Ok((first, sum_doc_freq))
    }
}

/// One decoded trie node (`TrieReader.Node`), covering all three shapes
/// (`SIGN_NO_CHILDREN`/`SIGN_SINGLE_CHILD_*`/`SIGN_MULTI_CHILDREN`) in a
/// single struct rather than Java's shape-specific fields left unset —
/// simpler than a Rust enum-per-shape here since [`load_node`] always fills
/// every field it needs for that shape and callers only ever read the
/// fields relevant to `node.sign`, mirroring how `TrieReader.Node` itself
/// mixes single-child/multi-child fields in one class.
#[derive(Debug, Clone, Copy, Default)]
pub(crate) struct TrieNode {
    sign: u32,
    /// This node's own file pointer within the field's `.tip` index slice.
    fp: usize,
    /// `Node.outputFp`/`Node.hasOutput()` — `None` when this node has no
    /// terms/sub-block of its own (an internal node that exists purely to
    /// route to deeper children).
    output_fp: Option<u64>,
    has_terms: bool,
    /// `Node.floorDataFp`/`Node.isFloor()`.
    floor_data_fp: Option<usize>,
    /// Single-child only: `Node.childDeltaFp`/`Node.minChildrenLabel`.
    child_delta_fp: u64,
    min_children_label: u8,
    /// Multi-children only: `Node.strategyFp`/`childSaveStrategy`/
    /// `strategyBytes`/`childrenDeltaFpBytes` (`minChildrenLabel` above is
    /// shared with the single-child case; multi packs it in the same role).
    strategy_fp: usize,
    child_save_strategy: u32,
    strategy_bytes: usize,
    children_delta_fp_bytes: usize,
}

/// `TrieReader.access.readLong(fp)`.
///
/// **This is the guard the whole trie decoder rests on.** Returning `Ok`
/// establishes `fp + 8 <= slice.len()`, i.e. `fp <= slice.len() - 8`, which is
/// what makes every `fp + k` (`k <= 11`) offset in [`load_node`] safe from
/// overflow. `fp + 8` must therefore *not* be computed directly: `rootFP`
/// comes off the `.tmd` as a `vlong` and is cast with `as usize`, so a
/// negative one arrives here as `usize::MAX`, where `fp + 8` panics in a debug
/// build and wraps to `7` -- passing the bound check -- in a release one, only
/// to panic on the slice index below.
fn read_u64_at(slice: &[u8], fp: usize) -> Result<u64> {
    let Some(end) = fp.checked_add(8).filter(|&end| end <= slice.len()) else {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "trie node read past end of index slice".into(),
        )));
    };
    Ok(u64::from_le_bytes(slice[fp..end].try_into().unwrap()))
}

fn read_u8_at(slice: &[u8], fp: usize) -> Result<u8> {
    slice.get(fp).copied().ok_or_else(|| {
        Error::Store(lucene_store::Error::Corrupted(
            "trie node read past end of index slice".into(),
        ))
    })
}

/// Reads `n_bytes` (1..=8) little-endian bytes starting at `fp` into a
/// `u64` — `TrieBuilder.writeLongNBytes`'s read-side inverse, used for the
/// multi-children children-fp array (`TrieReader.lookupChild`'s
/// `BYTES_MINUS_1_MASK`-free array-read, since here `n_bytes` is already a
/// byte count rather than a "minus 1" nibble).
fn read_u64_n_bytes(slice: &[u8], fp: usize, n_bytes: usize) -> Result<u64> {
    // The only caller passes `TrieNode::children_delta_fp_bytes`, which
    // `load_node` builds as `((term >> 2) & 0x07) + 1`, so it is in `1..=8`.
    debug_assert!((1..=8).contains(&n_bytes));
    let Some(end) = fp.checked_add(n_bytes).filter(|&end| end <= slice.len()) else {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "trie children-fp array read past end of index slice".into(),
        )));
    };
    let mut v = 0u64;
    // ARITH: `i` indexes `slice[fp..end]`, whose length is `n_bytes <= 8`, so
    // `i <= 7` and the shift amount `8 * i` is at most **56** -- the largest
    // legal `u64` shift is 63, and 64 is the one that panics.
    #[allow(clippy::arithmetic_side_effects)]
    for (i, &b) in slice[fp..end].iter().enumerate() {
        v |= (b as u64) << (8 * i);
    }
    Ok(v)
}

/// Reads one trie node at `fp` within `slice` (the field's `[indexStart,
/// indexEnd)` region of `.tip`) — `TrieReader.load`, dispatching on `sign`
/// to `loadLeafNode`/`loadSingleChildNode`/`loadMultiChildrenNode`.
// ARITH: the first statement is `read_u64_at(slice, fp)?`, which returns `Ok`
// only after establishing `fp + 8 <= slice.len()` with a `checked_add`. So for
// the whole body `fp <= slice.len() - 8 <= isize::MAX - 8`, and every offset
// formed from it is `fp + k` for a `k` bounded by the header's own bit fields:
//
// * `fp_bytes_minus1`, `child_delta_bytes_minus1` and `encoded_bytes_minus1`
//   are all `(term >> n) & 0x07`, hence at most 7;
// * `children_delta_fp_bytes` is `((term >> 2) & 0x07) + 1 <= 8`;
// * `strategy_bytes` is `((term >> 11) & 0x1F) + 1 <= 32`;
// * `children_num` is `read_u8_at(..) as u64 + 1 <= 256`.
//
// The widest offset any branch forms is the multi-children floor pointer,
// `fp + 4 + encoded_bytes_minus1 + 1 + strategy_bytes + children_num *
// children_delta_fp_bytes <= fp + 12 + 32 + 256 * 8 = fp + 2092`, which cannot
// overflow `usize`. The offsets are *not* bounds-checked here, deliberately:
// every one of them is later handed to `read_u64_at`/`read_u8_at`/
// `read_u64_n_bytes`, which do the bounds check at the point of use, exactly
// as `TrieReader` leaves it to `RandomAccessInput`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn load_node(slice: &[u8], fp: usize) -> Result<TrieNode> {
    let word = read_u64_at(slice, fp)?;
    let term = word as u32;
    let sign = term & 0x03;

    match sign {
        SIGN_NO_CHILDREN => {
            // loadLeafNode: [floor data][output fp][1x|floor|terms|3b fpBytes|2b sign]
            let fp_bytes_minus1 = (term >> 2) & 0x07;
            let output_fp = if fp_bytes_minus1 <= 6 {
                (word >> 8) & BYTES_MINUS_1_MASK[fp_bytes_minus1 as usize]
            } else {
                read_u64_at(slice, fp + 1)?
            };
            let has_terms = (term & LEAF_NODE_HAS_TERMS) != 0;
            let floor_data_fp = if (term & LEAF_NODE_HAS_FLOOR) != 0 {
                Some(fp + 2 + fp_bytes_minus1 as usize)
            } else {
                None
            };
            Ok(TrieNode {
                sign,
                fp,
                output_fp: Some(output_fp),
                has_terms,
                floor_data_fp,
                child_delta_fp: 0,
                min_children_label: 0,
                strategy_fp: 0,
                child_save_strategy: 0,
                strategy_bytes: 0,
                children_delta_fp_bytes: 0,
            })
        }
        SIGN_SINGLE_CHILD_WITH_OUTPUT | SIGN_SINGLE_CHILD_WITHOUT_OUTPUT => {
            // loadSingleChildNode: [floor][encoded output fp][child fp][label]
            // [3b encoded output fp bytes|3b child fp bytes|2b sign]
            let child_delta_bytes_minus1 = (term >> 2) & 0x07;
            let l = if child_delta_bytes_minus1 <= 5 {
                word >> 16
            } else {
                read_u64_at(slice, fp + 2)?
            };
            let child_delta_fp = l & BYTES_MINUS_1_MASK[child_delta_bytes_minus1 as usize];
            let min_children_label = ((term >> 8) & 0xFF) as u8;

            if sign == SIGN_SINGLE_CHILD_WITHOUT_OUTPUT {
                Ok(TrieNode {
                    sign,
                    fp,
                    output_fp: None,
                    has_terms: false,
                    floor_data_fp: None,
                    child_delta_fp,
                    min_children_label,
                    strategy_fp: 0,
                    child_save_strategy: 0,
                    strategy_bytes: 0,
                    children_delta_fp_bytes: 0,
                })
            } else {
                let encoded_bytes_minus1 = (term >> 5) & 0x07;
                let offset = fp + child_delta_bytes_minus1 as usize + 3;
                let encoded_fp =
                    read_u64_at(slice, offset)? & BYTES_MINUS_1_MASK[encoded_bytes_minus1 as usize];
                let output_fp = encoded_fp >> 2;
                let has_terms = (encoded_fp & NON_LEAF_NODE_HAS_TERMS) != 0;
                let floor_data_fp = if (encoded_fp & NON_LEAF_NODE_HAS_FLOOR) != 0 {
                    Some(offset + encoded_bytes_minus1 as usize + 1)
                } else {
                    None
                };
                Ok(TrieNode {
                    sign,
                    fp,
                    output_fp: Some(output_fp),
                    has_terms,
                    floor_data_fp,
                    child_delta_fp,
                    min_children_label,
                    strategy_fp: 0,
                    child_save_strategy: 0,
                    strategy_bytes: 0,
                    children_delta_fp_bytes: 0,
                })
            }
        }
        SIGN_MULTI_CHILDREN => {
            // loadMultiChildrenNode: [floor][children fps][strategy data]
            // [children count if floor][encoded output fp][label]
            // [5b strategy bytes|2b strategy|3b encoded fp bytes|1b has
            //  output|3b children fp bytes|2b sign]
            let children_delta_fp_bytes = (((term >> 2) & 0x07) + 1) as usize;
            let child_save_strategy = (term >> 9) & 0x03;
            let strategy_bytes = (((term >> 11) & 0x1F) + 1) as usize;
            let min_children_label = ((term >> 16) & 0xFF) as u8;

            if (term & 0x20) != 0 {
                let encoded_bytes_minus1 = (term >> 6) & 0x07;
                let l = if encoded_bytes_minus1 <= 4 {
                    word >> 24
                } else {
                    read_u64_at(slice, fp + 3)?
                };
                let encoded_fp = l & BYTES_MINUS_1_MASK[encoded_bytes_minus1 as usize];
                let output_fp = encoded_fp >> 2;
                let has_terms = (encoded_fp & NON_LEAF_NODE_HAS_TERMS) != 0;
                let (strategy_fp, floor_data_fp) = if (encoded_fp & NON_LEAF_NODE_HAS_FLOOR) != 0 {
                    let offset = fp + 4 + encoded_bytes_minus1 as usize;
                    let children_num = (read_u8_at(slice, offset)? as u64) + 1;
                    let sfp = offset + 1;
                    (
                        sfp,
                        Some(
                            sfp + strategy_bytes
                                + (children_num as usize) * children_delta_fp_bytes,
                        ),
                    )
                } else {
                    (fp + 4 + encoded_bytes_minus1 as usize, None)
                };
                Ok(TrieNode {
                    sign,
                    fp,
                    output_fp: Some(output_fp),
                    has_terms,
                    floor_data_fp,
                    child_delta_fp: 0,
                    min_children_label,
                    strategy_fp,
                    child_save_strategy,
                    strategy_bytes,
                    children_delta_fp_bytes,
                })
            } else {
                Ok(TrieNode {
                    sign,
                    fp,
                    output_fp: None,
                    has_terms: false,
                    floor_data_fp: None,
                    child_delta_fp: 0,
                    min_children_label,
                    strategy_fp: fp + 3,
                    child_save_strategy,
                    strategy_bytes,
                    children_delta_fp_bytes,
                })
            }
        }
        _ => unreachable!("sign is masked to 2 bits"),
    }
}

/// `TrieReader.lookupChild`: the child of `parent` labelled `target_label`,
/// or `None` when there is none. This is the one trie operation a lazy seek
/// needs — one label at a time down the target term's own bytes, never
/// enumerating a node's children.
// ARITH: in the order the operations appear --
//
// * both `parent.fp - delta` subtractions are guarded by an explicit
//   `delta > parent.fp` rejection on the two lines above them;
// * `strategy_fp + strategy_bytes + position * children_delta_fp_bytes` is
//   bounded by `parent.fp + 11 + 32 + 255 * 8 = parent.fp + 2083`, because
//   `load_node` sets `strategy_fp <= parent.fp + 11` and `strategy_bytes <= 32`
//   (see its own `// ARITH:` block), `children_delta_fp_bytes <= 8`, and
//   `position` is a label-derived index that [`child_position`] never returns
//   above 255 (each of its three strategies is proved there). `parent.fp` is
//   itself at most `slice.len() - 8`, so the sum cannot overflow `usize`.
#[allow(clippy::arithmetic_side_effects)]
fn lookup_child(slice: &[u8], parent: &TrieNode, target_label: u8) -> Result<Option<TrieNode>> {
    match parent.sign {
        SIGN_NO_CHILDREN => Ok(None),
        SIGN_SINGLE_CHILD_WITH_OUTPUT | SIGN_SINGLE_CHILD_WITHOUT_OUTPUT => {
            if target_label != parent.min_children_label {
                return Ok(None);
            }
            if (parent.child_delta_fp as usize) > parent.fp {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "trie child delta fp exceeds parent fp".into(),
                )));
            }
            let fp = parent.fp - parent.child_delta_fp as usize;
            Ok(Some(load_node(slice, fp)?))
        }
        SIGN_MULTI_CHILDREN => {
            let min_label = parent.min_children_label;
            let position = if target_label == min_label {
                0
            } else if target_label > min_label {
                child_position(slice, parent, target_label)?
            } else {
                -1
            };
            if position < 0 {
                return Ok(None);
            }
            debug_assert!(
                position <= 255,
                "child_position stays within one label byte"
            );
            let off = parent.strategy_fp
                + parent.strategy_bytes
                + position as usize * parent.children_delta_fp_bytes;
            let delta = read_u64_n_bytes(slice, off, parent.children_delta_fp_bytes)?;
            if (delta as usize) > parent.fp {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "trie child delta fp exceeds parent fp".into(),
                )));
            }
            Ok(Some(load_node(slice, parent.fp - delta as usize)?))
        }
        _ => unreachable!("sign is masked to 2 bits"),
    }
}

/// `TrieBuilder.ChildSaveStrategy.{BITS,ARRAY,REVERSE_ARRAY}.lookup`: the
/// index of `target_label` among a multi-children node's children, or `-1`.
///
/// `BITS` is decoded byte-wise rather than through Java's 64-bit
/// `RandomAccessInput.readLong`. The answer is identical — the long reads are
/// little-endian, so bit `i` of word `w` is bit `i & 7` of byte `8w + (i >> 3)`,
/// and `Long.bitCount` over the words below the target equals a byte
/// popcount over the same bytes — and it avoids Java's habit of reading eight
/// bytes past the end of the strategy region (harmless there because the
/// extra bits are masked off, but a bounds error for a slice that ends at the
/// trie region's last node).
///
/// The returned index is always `< 0` (absent) or `<= 255`; [`lookup_child`]'s
/// `// ARITH:` proof depends on that upper bound.
// ARITH: `min_label` and `target` are both `u8`s widened to `i32`, so every
// `target - min_label`, `max_label - min_label` and `... - low` is a
// difference of values in `0..=255` minus a further `low <= strategy_bytes <=
// 32`: all of them stay inside `-32..=255`, far from an `i32` boundary. This
// is also the one place in the module where a *sign* mistake would matter --
// `(target - min_label) as usize` sign-extends a negative difference into a
// huge index -- so the sole caller establishes `target > min_label` before
// entering (see the `target_label > min_label` arm in [`lookup_child`]) and
// the `debug_assert!` below pins it.
//
// `strategy_bytes` is `((term >> 11) & 0x1F) + 1`, hence in `1..=32`
// (`load_node`), so: `strategy_bytes * 8 <= 256`; `strategy_bytes as i32 - 1
// >= 0`; `strategy_bytes as i32 - 2 >= 0` on the path that reaches it, which
// the `strategy_bytes == 1` early return guards. Both bisections keep
// `0 <= low, high <= 31`, so `low + high <= 62` and `mid +/- 1` are trivially
// in range. `fp + i` / `fp + mid as usize` / `offset + mid as usize` add at
// most 32 to `strategy_fp <= slice.len() + 3`. The `BITS` accumulator sums at
// most 31 byte popcounts plus 7, so `pos <= 255`; `mask - 1` is safe because
// `mask = 1 << (bit_index & 7) >= 1`.
// SENTINEL: `-1` = "this node has no child with `target_label`". Outside the
// domain of the result (a child position is `>= 0`), so every call site has
// to test it -- see `docs/mechanical-gates.md`'s `sentinel-callers` rule.
#[allow(clippy::arithmetic_side_effects)]
fn child_position(slice: &[u8], node: &TrieNode, target_label: u8) -> Result<i32> {
    let fp = node.strategy_fp;
    let strategy_bytes = node.strategy_bytes;
    let min_label = node.min_children_label as i32;
    let target = target_label as i32;
    debug_assert!(
        target > min_label,
        "callers only probe labels above the minimum"
    );
    debug_assert!((1..=32).contains(&strategy_bytes));

    match node.child_save_strategy {
        CHILD_STRATEGY_BITS => {
            let bit_index = (target - min_label) as usize;
            if bit_index >= strategy_bytes * 8 {
                return Ok(-1);
            }
            let byte_index = bit_index >> 3;
            let byte = read_u8_at(slice, fp + byte_index)?;
            let mask = 1u8 << (bit_index & 7);
            if byte & mask == 0 {
                return Ok(-1);
            }
            let mut pos = 0i32;
            for i in 0..byte_index {
                pos += read_u8_at(slice, fp + i)?.count_ones() as i32;
            }
            pos += (byte & (mask - 1)).count_ones() as i32;
            Ok(pos)
        }
        CHILD_STRATEGY_ARRAY => {
            let (mut low, mut high) = (0i32, strategy_bytes as i32 - 1);
            while low <= high {
                let mid = (low + high) >> 1;
                let mid_label = read_u8_at(slice, fp + mid as usize)? as i32;
                match mid_label.cmp(&target) {
                    std::cmp::Ordering::Less => low = mid + 1,
                    std::cmp::Ordering::Greater => high = mid - 1,
                    std::cmp::Ordering::Equal => return Ok(mid + 1),
                }
            }
            Ok(-1)
        }
        CHILD_STRATEGY_REVERSE_ARRAY => {
            let max_label = read_u8_at(slice, fp)? as i32;
            let offset = fp + 1;
            if target >= max_label {
                return Ok(if target == max_label {
                    max_label - min_label - strategy_bytes as i32 + 1
                } else {
                    -1
                });
            }
            if strategy_bytes == 1 {
                return Ok(target - min_label);
            }
            let (mut low, mut high) = (0i32, strategy_bytes as i32 - 2);
            while low <= high {
                let mid = (low + high) >> 1;
                let mid_label = read_u8_at(slice, offset + mid as usize)? as i32;
                match mid_label.cmp(&target) {
                    std::cmp::Ordering::Less => low = mid + 1,
                    std::cmp::Ordering::Greater => high = mid - 1,
                    // An explicitly-absent label.
                    std::cmp::Ordering::Equal => return Ok(-1),
                }
            }
            Ok(target - min_label - low)
        }
        other => Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "invalid child save strategy code: {other}"
        )))),
    }
}

fn read_region_len(r: &mut SliceInput, what: &str) -> Result<usize> {
    let len = r.read_vint()?;
    if len < 0 {
        return Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "negative {what} length: {len}"
        ))));
    }
    let len = len as usize;
    if len > r.remaining() {
        return Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "{what} length {len} exceeds {} remaining bytes",
            r.remaining()
        ))));
    }
    Ok(len)
}

/// Port of `LowercaseAsciiCompression.decompress` (`o.a.l.util.compress`):
/// undoes the 4-into-3-byte 6-bit pack (bytes mostly in `[0x1F,0x3F)` /
/// `[0x5F,0x7F)`, i.e. ASCII digits/lowercase/`.`/`-`/`_`) plus a trailing
/// exception list for the rare non-compressible byte. `out.len()` is the
/// *original* (decompressed) length, matching Java's `len` parameter; the
/// compressed byte count (`compressedLen = len - len/4`) is derived from it,
/// not read from the stream.
// ARITH: `saved == len >> 2 <= len` so `len - saved` cannot underflow, and the
// three indices in step 2 are `compressed_len + i`, `saved + i` and
// `2 * saved + i` for `i < saved`; since `saved <= len / 4` and
// `compressed_len == len - saved`, all three are `< len == out.len()` and so
// cannot overflow `usize` either. In step 4 the loop's own `i >= out.len()`
// rejection means the accumulator enters every iteration with
// `i < out.len() <= isize::MAX`, and one iteration adds at most 255.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn decompress_lowercase_ascii(r: &mut SliceInput, out: &mut [u8]) -> Result<()> {
    let len = out.len();
    let saved = len >> 2;
    let compressed_len = len - saved;

    // 1. Copy the packed bytes.
    r.read_bytes(&mut out[..compressed_len])?;

    // 2. Restore the leading 2 bits of each packed byte into whole bytes.
    for i in 0..saved {
        out[compressed_len + i] = ((out[i] & 0xC0) >> 2)
            | ((out[saved + i] & 0xC0) >> 4)
            | ((out[(saved << 1) + i] & 0xC0) >> 6);
    }

    // 3. Move back to the original range.
    for b in out.iter_mut() {
        *b = ((*b & 0x1F) | 0x20 | ((*b & 0x20) << 1)).wrapping_sub(1);
    }

    // 4. Restore exceptions.
    let num_exceptions = r.read_vint()?;
    let mut i: usize = 0;
    for _ in 0..num_exceptions {
        i += r.read_byte()? as usize;
        if i >= out.len() {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "lowercase-ASCII exception index out of range".into(),
            )));
        }
        out[i] = r.read_byte()?;
    }

    Ok(())
}

/// Decodes one physical `.tim` block at `fp` in full — every entry, its
/// stats and its postings metadata — recursing into any in-block sub-block
/// pointer with that entry's own key bytes as the child's prefix.
///
/// Test-only. The reader itself never does this: [`SegmentTermsEnum`] loads a
/// block and reads only the entries a seek or scan actually walks over. This
/// exists so the block-format unit tests can assert on one block's whole
/// contents in isolation, which is what they were written against.
#[cfg(test)]
fn decode_block(
    tim: &[u8],
    fp: usize,
    index_options: IndexOptions,
    has_payloads: bool,
) -> Result<Vec<(Vec<u8>, TermStats, TermMetadata)>> {
    let mut out = Vec::new();
    decode_block_at_depth(tim, fp, index_options, has_payloads, 0, &[], &mut out)?;
    Ok(out)
}

// ARITH: `depth` is rejected above `10_000` on the function's first statement,
// so `depth + 1` is at most `10_001`.
#[allow(clippy::arithmetic_side_effects)]
#[cfg(test)]
fn decode_block_at_depth(
    tim: &[u8],
    fp: usize,
    index_options: IndexOptions,
    has_payloads: bool,
    depth: u32,
    prefix: &[u8],
    out: &mut Vec<(Vec<u8>, TermStats, TermMetadata)>,
) -> Result<()> {
    if depth > 10_000 {
        return Err(Error::Unsupported(
            "terms block sub-block nesting too deep (possible cycle)",
        ));
    }
    let mut frame = Frame {
        fp,
        fp_orig: fp,
        next_ent: -1,
        prefix_length: prefix.len(),
        ..Frame::default()
    };
    let tim = Tim {
        bytes: tim,
        zstd: None,
    };
    frame.load_block(tim)?;
    let mut term = TermBuf::default();
    term.copy_from(prefix);
    for _ in 0..frame.ent_count {
        if frame.next(tim, &mut term)? {
            let sub_fp = frame.last_sub_fp;
            let child_prefix = term.get().to_vec();
            decode_block_at_depth(
                tim.bytes,
                sub_fp as usize,
                index_options,
                has_payloads,
                depth + 1,
                &child_prefix,
                out,
            )?;
            continue;
        }
        frame.decode_meta_data(
            index_options,
            has_payloads,
            postings::PostingsFormat::Lucene104,
            [0; 3],
        )?;
        out.push((
            term.get().to_vec(),
            TermStats {
                doc_freq: frame.doc_freq,
                total_term_freq: frame.total_term_freq,
            },
            frame.meta,
        ));
    }
    Ok(())
}

/// Opens a `.tim`/`.tip`/`.tmd` triple already read whole into memory.
///
/// Copies `tim`/`tip` into shared buffers, since the returned
/// [`BlockTreeFields`] keeps navigating them for as long as it lives; use
/// [`open_shared`] to hand over buffers the caller already owns and skip the
/// copy.
pub fn open(
    tim: &[u8],
    tip: &[u8],
    tmd: &[u8],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    max_doc: i32,
) -> Result<BlockTreeFields> {
    open_shared(
        Arc::new(tim.to_vec()),
        Arc::new(tip.to_vec()),
        tmd,
        field_infos,
        segment_id,
        segment_suffix,
        max_doc,
    )
}

/// [`open`] without the `.tim`/`.tip` copy: the port of
/// `Lucene103BlockTreeTermsReader`'s constructor.
///
/// Reads the codec headers, then one `FieldReader`-equivalent record per
/// field out of `.tmd` — counts, min/max term, and the
/// `(indexStart, rootFP, indexEnd)` triple locating the field's trie — and
/// validates the recorded `.tip`/`.tim` lengths and footers. **No `.tim`
/// block is read**; that happens per lookup, in [`SegmentTermsEnum`].
pub fn open_shared(
    tim: SharedBytes,
    tip: SharedBytes,
    tmd: &[u8],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    max_doc: i32,
) -> Result<BlockTreeFields> {
    // `Lucene90BlockTreeTermsReader` (the FST-indexed dictionary of every
    // postings format from `Lucene90` to `Lucene101`) and
    // `Lucene103BlockTreeTermsReader` write the same three codec names at
    // overlapping versions; what tells them apart is the postings reader's
    // own header that follows in `.tmd`, which the older formats all name
    // `Lucene90PostingsWriterTerms`.
    if postings_terms_codec(tmd, segment_id, segment_suffix).as_deref()
        == Some(crate::backward_codecs::blocktree::POSTINGS_TERMS_CODEC)
    {
        return crate::backward_codecs::blocktree::open(
            tim,
            tip,
            tmd,
            field_infos,
            segment_id,
            segment_suffix,
            max_doc,
        );
    }
    let tim_bytes: &[u8] = tim.as_ref().as_ref();
    let tip_bytes: &[u8] = tip.as_ref().as_ref();
    let zstd = codec_util::peek_codec_name(tim_bytes).as_deref() == Some(ZSTD_TERMS_CODEC_NAME);
    let (terms_codec, index_codec, meta_codec) = if zstd {
        (
            ZSTD_TERMS_CODEC_NAME,
            ZSTD_TERMS_INDEX_CODEC_NAME,
            ZSTD_TERMS_META_CODEC_NAME,
        )
    } else {
        (
            TERMS_CODEC_NAME,
            TERMS_INDEX_CODEC_NAME,
            TERMS_META_CODEC_NAME,
        )
    };
    let mut tim_input = SliceInput::new(tim_bytes);
    let tim_header = codec_util::check_index_header(
        &mut tim_input,
        terms_codec,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut tip_input = SliceInput::new(tip_bytes);
    codec_util::check_index_header(
        &mut tip_input,
        index_codec,
        tim_header.version,
        tim_header.version,
        segment_id,
        segment_suffix,
    )?;

    let mut tmd_input = SliceInput::new(tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        meta_codec,
        tim_header.version,
        tim_header.version,
        segment_id,
        segment_suffix,
    )?;

    // PostingsReaderBase.init: the postings writer's own header, embedded in
    // the same .tmd stream right after BlockTree's index header.
    //
    // `Lucene103PostingsFormat` (10.3) shares this dictionary with a 128-doc
    // block and its own terms codec; everything else about the two is read
    // by the same code, the postings generation travelling with each field.
    let terms_format = if codec_util::peek_codec_name(tmd_input.as_slice()).as_deref()
        == Some(LUCENE103_POSTINGS_TERMS_CODEC)
    {
        postings::PostingsFormat::Lucene103
    } else {
        postings::PostingsFormat::Lucene104
    };
    codec_util::check_index_header(
        &mut tmd_input,
        if terms_format == postings::PostingsFormat::Lucene103 {
            LUCENE103_POSTINGS_TERMS_CODEC
        } else {
            POSTINGS_TERMS_CODEC
        },
        POSTINGS_VERSION_START,
        POSTINGS_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    let index_block_size = tmd_input.read_vint()?;
    if index_block_size as usize != terms_format.block_size() {
        return Err(Error::UnexpectedBlockSize {
            found: index_block_size,
        });
    }

    let num_fields = tmd_input.read_vint()?;
    if num_fields < 0 {
        return Err(Error::InvalidNumFields(num_fields));
    }
    // Java presizes an `IntObjectHashMap<>(numFields)` from the same vint and
    // survives a corrupt one with an `OutOfMemoryError` a caller can catch.
    // Here `Vec::with_capacity` on a `(String, FieldTerms)` -- a couple of
    // hundred bytes an element -- *aborts* on allocation failure, which
    // `catch_unwind` at the FFI boundary cannot intercept: four flipped bytes
    // would buy a several-hundred-gigabyte reservation and a dead JVM. The
    // `.tmd` stream is its own ceiling, since every field record spends at
    // least `MIN_FIELD_RECORD_BYTES` on the wire.
    if num_fields as usize > tmd_input.remaining() / MIN_FIELD_RECORD_BYTES {
        return Err(Error::InvalidNumFields(num_fields));
    }

    let mut fields: Vec<(String, FieldTerms)> = Vec::with_capacity(num_fields as usize);
    for _ in 0..num_fields {
        let field_number = tmd_input.read_vint()?;
        let num_terms = tmd_input.read_vlong()?;
        if num_terms <= 0 {
            return Err(Error::IllegalNumTerms(field_number));
        }
        let field_info = field_infos
            .field_by_number(field_number)
            .ok_or(Error::InvalidFieldNumber(field_number))?;
        let postings_format = terms_format;

        let (sum_total_term_freq, sum_doc_freq) =
            read_freq_pair(&mut tmd_input, field_info.index_options)?;
        let doc_count = tmd_input.read_vint()?;
        let min_term = read_bytes_ref(&mut tmd_input)?;
        let mut max_term = read_bytes_ref(&mut tmd_input)?;
        if num_terms == 1 {
            max_term = min_term.clone();
        }

        if !(0..=max_doc).contains(&doc_count) {
            return Err(Error::InvalidDocCount { doc_count, max_doc });
        }
        if sum_doc_freq < doc_count as i64 {
            return Err(Error::InvalidSumDocFreq {
                sum_doc_freq,
                doc_count,
            });
        }
        if sum_total_term_freq < sum_doc_freq {
            return Err(Error::InvalidSumTotalTermFreq {
                sum_total_term_freq,
                sum_doc_freq,
            });
        }

        let index_start = tmd_input.read_vlong()? as usize;
        let root_fp = tmd_input.read_vlong()? as usize;
        let index_end = tmd_input.read_vlong()? as usize;

        if index_end > tip_bytes.len() || index_start > index_end {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "field index region out of bounds".into(),
            )));
        }
        // `FieldReader`'s constructor stops here; loading the root node is
        // the one extra O(1) check this port keeps, so a `rootFP` outside the
        // field's own trie region is rejected at open rather than at the
        // first lookup.
        load_node(&tip_bytes[index_start..index_end], root_fp)?;
        let zstd = if zstd {
            let len = read_region_len(&mut tmd_input, "zstd terms dictionary")?;
            // In place: `DecoderDictionary::new` copies what it keeps.
            let dict = (len > 0)
                .then(|| lucene_zstd::DecoderDictionary::new(&tmd_input.as_slice()[..len]));
            tmd_input.skip(len)?;
            Some(Arc::new(ZstdBlocks { dict }))
        } else {
            None
        };

        if fields.iter().any(|(n, _)| n == &field_info.name) {
            return Err(Error::DuplicateField(field_info.name.clone()));
        }
        fields.push((
            field_info.name.clone(),
            FieldTerms {
                num_terms,
                sum_total_term_freq,
                sum_doc_freq,
                doc_count,
                min_term,
                max_term,
                index_options: field_info.index_options,
                has_payloads: field_info.store_payloads,
                postings_format,
                tim: Arc::clone(&tim),
                zstd,
                index: TermsIndex::InPlace(TrieSlice {
                    bytes: Arc::clone(&tip),
                    start: index_start,
                    root_fp,
                    end: index_end,
                }),
                scratch: ScratchPool::default(),
                ngram: std::sync::OnceLock::new(),
                wide_walks: std::sync::atomic::AtomicU32::new(0),
                postings_base: [0; 3],
                hidden_terms: None,
            },
        ));
    }

    let index_length = tmd_input.read_i64()?;
    let terms_length = tmd_input.read_i64()?;
    codec_util::check_footer(&mut tmd_input, tmd.len())?;

    // `Lucene103BlockTreeTermsReader`'s constructor ends with
    // `CodecUtil.retrieveChecksum(indexIn, indexLength)` /
    // `retrieveChecksum(termsIn, termsLength)` -- the *expected-length*
    // overload, which rejects a file that is too short **and** one that is
    // too long, then checks the footer.
    if index_length < 0 || terms_length < 0 {
        return Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "negative recorded .tip/.tim length: {index_length}/{terms_length}"
        ))));
    }
    codec_util::retrieve_checksum_with_expected_length(tip_bytes, index_length as usize)?;
    codec_util::retrieve_checksum_with_expected_length(tim_bytes, terms_length as usize)?;

    Ok(BlockTreeFields { fields })
}

#[cfg(test)]
mod tests;
