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
//! **Why this replaced the previous design.** Until this port's `c1` batch,
//! `open` recursively visited every trie node, expanded every floor block,
//! decoded every `.tim` block and merged every field's terms into one sorted
//! array. It gave the same answers but cost `O(all terms in the segment)` in
//! both time and memory at open: **35.4 ms** to open the M1 benchmark
//! corpus' single 579k-term segment where real Lucene's whole
//! `DirectoryReader.open` costs 0.34 ms, and one live copy of every term's
//! bytes plus a 64-byte record for as long as the reader lived. A search
//! engine reopens readers on every refresh, so that was the largest
//! architectural divergence left in the read path (finding A1 in
//! `docs/sweep/m2/LEDGER.md`). See `docs/sweep/m2/c1-lazy-blocktree.md` for
//! the before/after numbers.
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
//! are both decoded, alongside `NO_COMPRESSION`. This port's own blocktree
//! *writer* only ever emits `NO_COMPRESSION`; the other two exist to read
//! real Lucene-written segments. Only code `3` (never assigned to a
//! `CompressionAlgorithm` constant) is rejected, as `Error::Store(Corrupted)`,
//! matching `CompressionAlgorithm.byCode`'s own `IllegalArgumentException`.

use std::sync::{Arc, Mutex};

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

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
const VERSION_START: i32 = 0;
pub(crate) const VERSION_CURRENT: i32 = 0;

/// `Lucene104PostingsFormat.TERMS_CODEC` — the postings writer's own header,
/// embedded in the `.tmd` stream right after BlockTree's own index header.
pub(crate) const POSTINGS_TERMS_CODEC: &str = "Lucene104PostingsWriterTerms";
const POSTINGS_VERSION_START: i32 = 0;
pub(crate) const POSTINGS_VERSION_CURRENT: i32 = 0;
/// `Lucene104PostingsFormat.BLOCK_SIZE` (= `ForUtil.BLOCK_SIZE`), the postings
/// block size the `.tmd` stream's `indexBlockSize` field must match.
pub(crate) const POSTINGS_BLOCK_SIZE: i32 = 256;

/// `TrieBuilder.SIGN_NO_CHILDREN` — a leaf trie node (no children).
pub(crate) const SIGN_NO_CHILDREN: u32 = 0x00;
/// `TrieBuilder.SIGN_SINGLE_CHILD_WITH_OUTPUT`.
const SIGN_SINGLE_CHILD_WITH_OUTPUT: u32 = 0x01;
/// `TrieBuilder.SIGN_SINGLE_CHILD_WITHOUT_OUTPUT`.
const SIGN_SINGLE_CHILD_WITHOUT_OUTPUT: u32 = 0x02;
/// `TrieBuilder.SIGN_MULTI_CHILDREN`.
pub(crate) const SIGN_MULTI_CHILDREN: u32 = 0x03;
/// `TrieBuilder.LEAF_NODE_HAS_TERMS` (`1 << 5`).
pub(crate) const LEAF_NODE_HAS_TERMS: u32 = 1 << 5;
/// `TrieBuilder.LEAF_NODE_HAS_FLOOR` (`1 << 6`).
const LEAF_NODE_HAS_FLOOR: u32 = 1 << 6;
/// `TrieBuilder.NON_LEAF_NODE_HAS_TERMS` (`1L << 1`) — the equivalent flag
/// packed into a non-leaf node's *encoded output fp* (`encodeFP`), not its
/// header byte, since non-leaf nodes' header bits are all spoken for by
/// child-pointer bookkeeping.
const NON_LEAF_NODE_HAS_TERMS: u64 = 1 << 1;
/// `TrieBuilder.NON_LEAF_NODE_HAS_FLOOR` (`1L << 0`).
const NON_LEAF_NODE_HAS_FLOOR: u64 = 1;
/// `TrieBuilder.ChildSaveStrategy.REVERSE_ARRAY.code`.
const CHILD_STRATEGY_REVERSE_ARRAY: u32 = 0;
/// `TrieBuilder.ChildSaveStrategy.ARRAY.code`.
pub(crate) const CHILD_STRATEGY_ARRAY: u32 = 1;
/// `TrieBuilder.ChildSaveStrategy.BITS.code`.
const CHILD_STRATEGY_BITS: u32 = 2;

/// The fewest bytes one `.tmd` per-field record can occupy: nine
/// single-byte-minimum values -- `fieldNumber`, `numTerms`,
/// `sumTotalTermFreq`, `sumDocFreq`, `docCount`, the two length prefixes of
/// `minTerm`/`maxTerm`, and `indexStart`/`rootFP`/`indexEnd` -- less the one
/// `sumDocFreq` that `IndexOptions::Docs` aliases away. Used only as a ceiling
/// on `numFields`, so undercounting is the safe direction.
const MIN_FIELD_RECORD_BYTES: usize = 9;

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
}

pub type Result<T> = std::result::Result<T, Error>;

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
    /// (`numFollowFloorBlocks`), for `rewind`.
    rewind_pos: usize,
    /// Position in the field's `.tip` slice of the next unread floor record.
    floor_data_pos: usize,
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
    fn load_block(&mut self, tim: &[u8]) -> Result<()> {
        if self.next_ent != -1 {
            // Already loaded.
            return Ok(());
        }
        let mut r = SliceInput::new(tim);
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
                crate::lz4::decompress(&mut r, num_suffix_bytes, suffixes, 0)?;
            }
            _ => {
                // `code_l & 0x03` is masked to 2 bits, so `3` is the only
                // remaining value; `CompressionAlgorithm.byCode` throws
                // `IllegalArgumentException` for it too.
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "illegal compression algorithm code (3) for a terms block".into(),
                )));
            }
        }
        self.suffixes_pos = 0;

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

        let num_stat_bytes = read_region_len(&mut r, "terms block stats")?;
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

        let num_meta_bytes = read_region_len(&mut r, "terms block metadata")?;
        fit_buf(&mut self.meta_bytes, num_meta_bytes, "terms block metadata")?;
        self.meta_bytes_len = num_meta_bytes;
        r.read_bytes(&mut self.meta_bytes[..num_meta_bytes])?;
        self.meta_pos = 0;

        // Sub-blocks of one floor block are written back to back, so the next
        // floor block starts exactly here.
        self.fp_end = r.position();
        Ok(())
    }

    /// `SegmentTermsEnumFrame.loadNextFloorBlock`.
    fn load_next_floor_block(&mut self, tim: &[u8]) -> Result<()> {
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
        let target_label = target[self.prefix_length] as u32;
        if target_label < self.next_floor_label {
            // Already on the correct block.
            return Ok(());
        }

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
    fn next_non_leaf(&mut self, tim: &[u8], term: &mut TermBuf) -> Result<bool> {
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

    /// `SegmentTermsEnumFrame.next`.
    fn next(&mut self, tim: &[u8], term: &mut TermBuf) -> Result<bool> {
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
    fn decode_meta_data(&mut self, index_options: IndexOptions, has_payloads: bool) -> Result<()> {
        let limit = self.term_block_ord();
        if limit == 0 {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "asked for term metadata with no term read from this block".into(),
            )));
        }
        let mut absolute = self.meta_data_upto == 0;
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
            let Frame {
                meta_bytes,
                meta_bytes_len,
                meta_pos,
                meta,
                ..
            } = self;
            let mut r = SliceInput::new(&meta_bytes[..*meta_bytes_len]);
            r.seek(*meta_pos)?;
            *meta = postings::decode_term_metadata(
                &mut r,
                doc_freq,
                absolute,
                *meta,
                index_options,
                has_payloads,
                total_term_freq,
            )?;
            *meta_pos = r.position();

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
#[derive(Debug, Default)]
struct EnumState {
    term: TermBuf,
    stack: Vec<Frame>,
    /// Index of the current frame, or `-1` when the enum is unpositioned
    /// (Java's `currentFrame == staticFrame`).
    current: i32,
    /// Java's `termExists`: the entry the cursor is on is a term rather than
    /// a sub-block pointer. Written wherever Java writes it and read by
    /// nothing here yet -- both of Java's consumers,
    /// `seekExact(BytesRef, TermState)` and the seek-state-reuse prologue,
    /// are deliberately unported (see [`SegmentTermsEnum`]'s doc comment).
    /// Kept rather than dropped because `scanToTermNonLeaf` is the only place
    /// that can compute it, so a later port of either would otherwise have to
    /// re-thread it back through three call layers.
    term_exists: bool,
    /// The enum is parked on a real term (so `term()`/`docFreq()` are
    /// meaningful). False before the first call and after end-of-terms.
    on_term: bool,
    /// Java's `eof` (an assert-only flag there): repeated `next()` past the
    /// end keeps returning "no more terms" instead of walking off the stack.
    eof: bool,
}

/// Port of `SegmentTermsEnum` -- the whole lazy navigator. Borrows the
/// field's `.tim`/`.tip` bytes and a (possibly pooled) [`EnumState`].
///
/// **What is not ported.** Java's `prepareSeekExact` and `seekCeil` open with
/// a branch that reuses the *previous* seek's frame stack when the new target
/// shares a prefix with the current term (`validIndexPrefix`, `nodes[]`,
/// `lastFrame`). It is a pure optimization for seeking in sorted order; every
/// seek here restarts from the root instead, which is exactly Java's own
/// `currentFrame == staticFrame` path. What that branch mainly buys --
/// not re-loading a block the frame already holds -- is recovered by
/// [`Frame::rewind`]'s in-place cursor reset plus the per-field pooled state,
/// which apply to *any* access order rather than only a sorted one. See
/// `docs/sweep/m2/c1-lazy-blocktree.md`.
///
/// Also unported, with the same status as before this batch:
/// `seekExact(BytesRef, TermState)`/`termState()`/`ord()` (no `TermStates`
/// reuse exists in this port's search layer) and `prefetchBlock`
/// (`IndexInput.prefetch`, unmeasurable against a warm page cache).
struct SegmentTermsEnum<'a> {
    field: &'a FieldTerms,
    st: &'a mut EnumState,
}

impl<'a> SegmentTermsEnum<'a> {
    #[inline]
    fn tim(&self) -> &'a [u8] {
        self.field.tim.as_ref().as_ref()
    }

    /// The field's own `[indexStart, indexEnd)` region of `.tip`.
    #[inline]
    fn index(&self) -> &'a [u8] {
        &self.field.tip.as_ref().as_ref()[self.field.index_start..self.field.index_end]
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

    /// `SegmentTermsEnum.pushFrame(node, length)`.
    fn push_frame_node(&mut self, node: &TrieNode, length: usize) -> Result<()> {
        let ord = self.next_ord()?;
        self.ensure_frame(ord);
        let index = self.index();
        let f = &mut self.st.stack[ord];
        f.has_terms = node.has_terms;
        f.has_terms_orig = node.has_terms;
        f.is_floor = node.floor_data_fp.is_some();
        if let Some(fdp) = node.floor_data_fp {
            f.set_floor_data(index, fdp)?;
        }
        let fp = node.output_fp.unwrap_or_default() as usize;
        self.push_frame_fp(fp, length)
    }

    /// `SegmentTermsEnum.pushFrame(node, fp, length)` -- the shared tail, also
    /// used on its own for a frame pushed by following an in-block sub-block
    /// pointer (Java's `pushFrame(null, fp, length)`).
    fn push_frame_fp(&mut self, fp: usize, length: usize) -> Result<()> {
        let ord = self.next_ord()?;
        self.ensure_frame(ord);
        let index = self.index();
        let f = &mut self.st.stack[ord];
        if f.fp_orig == fp && f.next_ent != -1 && f.prefix_length == length {
            // Same block as last time this ord was used: keep its decoded
            // regions and just rewind the cursors.
            f.rewind(index)?;
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
        self.st.current = -1;
        self.st.term.clear();
        self.st.term_exists = false;
        self.st.on_term = false;
        self.st.eof = false;
    }

    fn root(&self) -> Result<TrieNode> {
        load_node(self.index(), self.field.root_fp)
    }

    /// `SegmentTermsEnum.seekExact(BytesRef)`.
    // ARITH: `target_upto` is only incremented inside `while target_upto <
    // target.len()`, so it never exceeds `target.len()`, which as a slice
    // length is at most `isize::MAX`; `target_upto + 1` and `1 + target_upto`
    // therefore cannot overflow `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        // `prepareSeekExact`'s first line: the field's recorded min/max term
        // bound every possible hit, so an out-of-range target never touches
        // the trie at all.
        if self.field.num_terms > 0
            && (target < self.field.min_term.as_slice() || target > self.field.max_term.as_slice())
        {
            self.reset();
            return Ok(false);
        }
        self.reset();

        let index = self.index();
        let mut node = self.root()?;
        let mut target_upto = 0usize;
        self.push_frame_node(&node, 0)?;

        while target_upto < target.len() {
            let target_label = target[target_upto];
            match lookup_child(index, &node, target_label)? {
                None => {
                    // The index is exhausted: this frame's block is the only
                    // one that could hold the target.
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
                    node = next_node;
                    target_upto += 1;
                    if node.output_fp.is_some() {
                        self.push_frame_node(&node, target_upto)?;
                    }
                }
            }
        }

        self.cur().scan_to_floor_frame(index, target)?;
        if !self.cur_ref().has_terms {
            self.st.term_exists = false;
            self.st.term.set_length(target_upto);
            return Ok(false);
        }
        self.load_and_scan_exact(target)
    }

    fn load_and_scan_exact(&mut self, target: &[u8]) -> Result<bool> {
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

    /// `SegmentTermsEnum.seekCeil(BytesRef)`.
    // ARITH: as in [`SegmentTermsEnum::seek_exact`] -- `target_upto` is only
    // incremented under `while target_upto < target.len()`, and a slice length
    // is at most `isize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.reset();
        let index = self.index();
        let mut node = self.root()?;
        let mut target_upto = 0usize;
        self.push_frame_node(&node, 0)?;

        while target_upto < target.len() {
            let target_label = target[target_upto];
            match lookup_child(index, &node, target_label)? {
                None => {
                    self.cur().scan_to_floor_frame(index, target)?;
                    return self.load_and_scan_ceil(target);
                }
                Some(next_node) => {
                    self.st.term.set_byte_at(target_upto, target_label);
                    node = next_node;
                    target_upto += 1;
                    if node.output_fp.is_some() {
                        self.push_frame_node(&node, target_upto)?;
                    }
                }
            }
        }

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
            self.push_frame_node(&node, 0)?;
            self.load_current_block()?;
        }

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
        let (index_options, has_payloads) = (self.field.index_options, self.field.has_payloads);
        let f = self.cur();
        f.decode_meta_data(index_options, has_payloads)?;
        Ok(TermStats {
            doc_freq: f.doc_freq,
            total_term_freq: f.total_term_freq,
        })
    }

    /// `SegmentTermsEnum.postings()`'s half of `decodeMetaData`: the postings
    /// file pointers for the term the enum is parked on.
    fn stats_and_meta(&mut self) -> Result<(TermStats, TermMetadata)> {
        let (index_options, has_payloads) = (self.field.index_options, self.field.has_payloads);
        let f = self.cur();
        f.decode_meta_data(index_options, has_payloads)?;
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
    fn new(field: &'a FieldTerms) -> Self {
        Self {
            field,
            st: EnumState {
                current: -1,
                ..EnumState::default()
            },
        }
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
    pub fn try_next(&mut self) -> Result<Option<(&[u8], TermStats)>> {
        let mut ste = self.ste();
        if !ste.next()? {
            return Ok(None);
        }
        let stats = ste.stats()?;
        Ok(Some((self.st.term.get(), stats)))
    }

    /// [`Self::try_next`] with the error dropped: a corrupt block reads as
    /// end-of-terms.
    ///
    /// **Test convenience only.** No production caller in this workspace uses
    /// this spelling any more (`c39-codecs-readpath` migrated the last of
    /// them, and the migration is re-checkable by marking these four methods
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
}

/// One field's term dictionary: the `.tmd` record plus the `.tim`/`.tip`
/// bytes to navigate on demand — the port of `FieldReader`.
///
/// Cloning shares the underlying `.tim`/`.tip` buffers (`Arc`) and starts the
/// clone with an empty lookup scratch.
/// Shared, type-erased ownership of one whole codec file's bytes.
///
/// [`FieldTerms`] navigates the segment's `.tim`/`.tip` for as long as it
/// lives, so it has to *own* a share of them. `Arc<[u8]>` was the obvious
/// choice and was what this held until c12, but an `Arc<[u8]>` owns its own
/// allocation: building one from a `memmap2::Mmap` (what
/// `lucene_store::MmapDirectory` hands back, and Lucene's own
/// `MMapDirectory` equivalent) copies the whole file. On the M1 benchmark
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

pub struct FieldTerms {
    pub num_terms: i64,
    pub sum_total_term_freq: i64,
    pub sum_doc_freq: i64,
    pub doc_count: i32,
    pub min_term: Vec<u8>,
    pub max_term: Vec<u8>,
    index_options: IndexOptions,
    has_payloads: bool,
    /// The whole segment's `.tim`, shared by every field.
    tim: SharedBytes,
    /// The whole segment's `.tip`, shared by every field; this field's trie
    /// occupies `[index_start, index_end)`.
    tip: SharedBytes,
    index_start: usize,
    index_end: usize,
    root_fp: usize,
    /// One pooled [`EnumState`] so the `&self` lookups
    /// ([`Self::seek_exact`], [`Self::postings`], ...) keep the last-loaded
    /// blocks warm across calls instead of re-decoding them. Java gets the
    /// same effect from the caller holding a `TermsEnum`; this port's API
    /// takes a term per call, so the reuse has to live here.
    scratch: Mutex<EnumState>,
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
            .field("tip_len", &self.tip.as_ref().as_ref().len())
            .field("index_start", &self.index_start)
            .field("index_end", &self.index_end)
            .field("root_fp", &self.root_fp)
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
            tim: Arc::clone(&self.tim),
            tip: Arc::clone(&self.tip),
            index_start: self.index_start,
            index_end: self.index_end,
            root_fp: self.root_fp,
            scratch: Mutex::new(EnumState {
                current: -1,
                ..EnumState::default()
            }),
        }
    }
}

impl FieldTerms {
    /// Runs `f` against the pooled lookup state.
    fn with_scratch<T>(&self, f: impl FnOnce(&mut SegmentTermsEnum<'_>) -> Result<T>) -> Result<T> {
        // `try_lock`, never `lock`: a `BlockTreeFields` is shared by every
        // concurrently-searching thread (`lucene-ffi` hands the same `Arc` to
        // all of them), so blocking here would serialize what is otherwise a
        // read-only, embarrassingly parallel lookup. A thread that loses the
        // race just runs on its own fresh state -- same answers, only without
        // the warm blocks -- which is exactly what Java's callers get, since
        // there each thread holds its own `TermsEnum`.
        match self.scratch.try_lock() {
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

    /// `TermsEnum.seekExact(BytesRef)` + `docFreq()`/`totalTermFreq()`:
    /// walks the `.tip` trie to the one block that could hold `term`, loads
    /// it, and decodes only that term's stats.
    pub fn try_seek_exact(&self, term: &[u8]) -> Result<Option<TermStats>> {
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
    fn term_state(&self, term: &[u8]) -> Result<Option<(TermStats, TermMetadata)>> {
        self.with_scratch(|ste| {
            if !ste.seek_exact(term)? {
                return Ok(None);
            }
            Ok(Some(ste.stats_and_meta()?))
        })
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
        Intersect::new(self, PrefixMatcher(pattern), pattern.literal_prefix())
    }

    /// `FuzzyQuery`-equivalent term matching: every term within `pattern`'s
    /// edit-distance budget, in sorted order, with its stats. Same shape as
    /// [`Self::intersect`], with `pattern`'s required `prefixLength`-byte
    /// exact prefix as the seek target.
    pub fn fuzzy_intersect<'a>(&'a self, pattern: &'a FuzzyMatch<'a>) -> FuzzyIntersect<'a> {
        let prefix = pattern.literal_prefix().to_vec();
        FuzzyIntersect {
            inner: Intersect::new(
                self,
                FuzzyMatcher {
                    pattern,
                    max_edits: pattern.max_edits(),
                    last_edits: 0,
                },
                prefix,
            ),
        }
    }

    /// `RegexpQuery`-equivalent term matching, with the dead-prefix **block
    /// skip**: when [`RegexpPattern::dead_prefix_len`] proves no term sharing
    /// the current term's `k`-byte prefix can match, the walk seeks straight
    /// past that whole prefix range, which with lazy frames means the blocks
    /// under it are never loaded. That is what real Lucene's
    /// `IntersectTermsEnum` gets from a `ByteRunAutomaton` entering a dead
    /// state; see `docs/sweep/m2/c1-lazy-blocktree.md` for the measurement.
    pub fn regexp_intersect<'a>(
        &'a self,
        pattern: &'a RegexpPattern,
    ) -> impl Iterator<Item = Result<(Vec<u8>, TermStats)>> + 'a {
        Intersect::new(self, RegexpMatcher(pattern), pattern.literal_prefix())
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
    /// and not, as this used to, the term's whole doc list. See
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

    fn matches(&mut self, term: &[u8]) -> bool;

    /// `k` such that no term starting with `term[..k]` can match, or `None`.
    /// The [`IntersectTermsEnum`-equivalent](FieldTerms::regexp_intersect)
    /// skip; a matcher that cannot prove this simply never skips.
    fn dead_prefix_len(&self, _term: &[u8]) -> Option<usize> {
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
/// (`getAutomatonEnum(maxEdits, lastTerm)`). This walk needs no re-seek: it is
/// a forward scan over one sorted range, so tightening the predicate takes
/// effect on the next term and the position is already correct. What it buys
/// is the same thing Java's automaton swap buys -- every remaining term is
/// tested against a narrower band, and the length filter rejects far more of
/// them outright.
pub struct FuzzyIntersect<'a> {
    inner: Intersect<'a, FuzzyMatcher<'a, 'a>>,
}

impl FuzzyIntersect<'_> {
    /// `bottomChanged`'s `actualEnum = getAutomatonEnum(maxEdits, lastTerm)`.
    ///
    /// Only ever lowers: a budget at or above the one in force is ignored,
    /// because widening mid-scan would make the walk yield terms it had
    /// already rejected further back, which no caller could interpret.
    pub fn set_max_edits(&mut self, max_edits: u8) {
        if max_edits < self.inner.matcher.max_edits {
            self.inner.matcher.max_edits = max_edits;
        }
    }

    /// The budget currently in force.
    pub fn max_edits(&self) -> u8 {
        self.inner.matcher.max_edits
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
        self.inner.matcher.last_edits
    }
}

impl Iterator for FuzzyIntersect<'_> {
    type Item = Result<(Vec<u8>, TermStats)>;

    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
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
            return Ok(self.enum_.try_next()?.is_some());
        }
        Ok(match self.enum_.try_seek_ceil(&self.prefix)? {
            SeekStatus::Found | SeekStatus::NotFound => true,
            SeekStatus::End => false,
        })
    }

    /// The body of [`Iterator::next`], with the error channel Java's
    /// `IntersectTermsEnum.next()` has (it throws `IOException`).
    fn next_result(&mut self) -> Result<Option<(Vec<u8>, TermStats)>> {
        if self.done {
            return Ok(None);
        }
        if !self.started && !self.start()? {
            self.done = true;
            return Ok(None);
        }
        loop {
            let Some((term, stats)) = self.enum_.try_current()? else {
                self.done = true;
                return Ok(None);
            };
            if !term.starts_with(&self.prefix) {
                self.done = true;
                return Ok(None);
            }
            if self.matcher.matches(term) {
                let item = (term.to_vec(), stats);
                if self.enum_.try_next()?.is_none() {
                    self.done = true;
                }
                return Ok(Some(item));
            }

            // Not a match. Either step to the next term, or -- when the
            // pattern proves this whole prefix range is dead -- seek past it.
            if !M::CAN_SKIP || !self.skip_enabled {
                if self.enum_.try_next()?.is_none() {
                    self.done = true;
                    return Ok(None);
                }
                continue;
            }

            // Asking the matcher is itself the cost the give-up below
            // measures: a pattern whose language is prefix-closed (`cat.*`,
            // `t.*99`) never yields a dead prefix, so every one of these
            // calls is pure loss. Count the *question*, not just the jumps.
            let target = self
                .matcher
                .dead_prefix_len(term)
                .filter(|&k| k <= term.len())
                .and_then(|k| prefix_upper_bound(&term[..k]));
            // A heuristic counter: `saturating_add` is the honest semantics
            // here, because a saturated attempt count can only leave the
            // skip heuristic permanently on or off -- it can never change
            // which terms the intersection yields.
            self.skip_attempts = self.skip_attempts.saturating_add(1);

            match target {
                None => {
                    if self.enum_.try_next()?.is_none() {
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
    type Item = Result<(Vec<u8>, TermStats)>;

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
fn read_bytes_ref(input: &mut SliceInput) -> Result<Vec<u8>> {
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
fn read_freq_pair(input: &mut SliceInput, index_options: IndexOptions) -> Result<(i64, i64)> {
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
#[derive(Debug, Clone, Copy)]
struct TrieNode {
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
fn load_node(slice: &[u8], fp: usize) -> Result<TrieNode> {
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
fn decompress_lowercase_ascii(r: &mut SliceInput, out: &mut [u8]) -> Result<()> {
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
    frame.load_block(tim)?;
    let mut term = TermBuf::default();
    term.copy_from(prefix);
    for _ in 0..frame.ent_count {
        if frame.next(tim, &mut term)? {
            let sub_fp = frame.last_sub_fp;
            let child_prefix = term.get().to_vec();
            decode_block_at_depth(
                tim,
                sub_fp as usize,
                index_options,
                has_payloads,
                depth + 1,
                &child_prefix,
                out,
            )?;
            continue;
        }
        frame.decode_meta_data(index_options, has_payloads)?;
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
    let tim_bytes: &[u8] = tim.as_ref().as_ref();
    let tip_bytes: &[u8] = tip.as_ref().as_ref();
    let mut tim_input = SliceInput::new(tim_bytes);
    let tim_header = codec_util::check_index_header(
        &mut tim_input,
        TERMS_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut tip_input = SliceInput::new(tip_bytes);
    codec_util::check_index_header(
        &mut tip_input,
        TERMS_INDEX_CODEC_NAME,
        tim_header.version,
        tim_header.version,
        segment_id,
        segment_suffix,
    )?;

    let mut tmd_input = SliceInput::new(tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        TERMS_META_CODEC_NAME,
        tim_header.version,
        tim_header.version,
        segment_id,
        segment_suffix,
    )?;

    // PostingsReaderBase.init: the postings writer's own header, embedded in
    // the same .tmd stream right after BlockTree's index header.
    codec_util::check_index_header(
        &mut tmd_input,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_START,
        POSTINGS_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    let index_block_size = tmd_input.read_vint()?;
    if index_block_size != POSTINGS_BLOCK_SIZE {
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
                tim: Arc::clone(&tim),
                tip: Arc::clone(&tip),
                index_start,
                index_end,
                root_fp,
                scratch: Mutex::new(EnumState {
                    current: -1,
                    ..EnumState::default()
                }),
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
mod tests {
    // The arithmetic gate is about values read off disk; a fixture builder's
    // `i + 1` is not one. See `docs/arithmetic-gate.md`, "Test code".
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use crate::field_infos::FieldInfo;
    use lucene_store::data_output::DataOutput;

    /// Test-only: every child of `node`, found by probing
    /// [`lookup_child`] with all 256 labels.
    ///
    /// Production never enumerates a trie node's children -- a seek walks one
    /// label at a time down the target term's own bytes -- but the structural
    /// tests below want the whole subtree, and probing every label exercises
    /// each `ChildSaveStrategy`'s hit *and* miss paths rather than a
    /// separate "list all" decoder that production would not use.
    fn trie_children(slice: &[u8], node: &TrieNode) -> Result<Vec<(u8, TrieNode)>> {
        let mut out = Vec::new();
        for label in 0..=u8::MAX {
            if let Some(child) = lookup_child(slice, node, label)? {
                out.push((label, child));
            }
        }
        Ok(out)
    }

    /// Test-only: every physical `.tim` block one trie node's output
    /// addresses -- the base block plus each follow-on floor sub-block --
    /// driven through the production [`Frame::scan_to_floor_frame`] one
    /// boundary label at a time. Production only ever asks for the *one*
    /// block a target term falls in.
    fn expand_floor(
        index: &[u8],
        base_fp: u64,
        base_has_terms: bool,
        floor_data_fp: Option<usize>,
    ) -> Result<Vec<(u64, bool)>> {
        let Some(fdp) = floor_data_fp else {
            return Ok(vec![(base_fp, base_has_terms)]);
        };
        let mut f = Frame {
            fp: base_fp as usize,
            fp_orig: base_fp as usize,
            has_terms: base_has_terms,
            has_terms_orig: base_has_terms,
            is_floor: true,
            next_ent: -1,
            ..Frame::default()
        };
        f.set_floor_data(index, fdp)?;
        let mut out = Vec::new();
        loop {
            out.push((f.fp as u64, f.has_terms));
            if f.next_floor_label > u8::MAX as u32 {
                return Ok(out);
            }
            let label = f.next_floor_label as u8;
            f.scan_to_floor_frame(index, &[label])?;
        }
    }

    /// Test-only: every physical `.tim` block reachable from `node`, paired
    /// with the trie label path that led to it. This is what `open` used to
    /// do for every field of every segment before the lazy port; it survives
    /// only as a way for the fixture tests to assert on a field's whole
    /// block structure.
    fn collect_leaf_blocks(
        slice: &[u8],
        node: &TrieNode,
        depth: u32,
        prefix: &mut Vec<u8>,
        out: &mut Vec<(u64, Vec<u8>)>,
    ) -> Result<()> {
        if depth > 10_000 {
            return Err(Error::Unsupported("trie nesting too deep (possible cycle)"));
        }
        if let Some(fp) = node.output_fp {
            for (block_fp, has_terms) in
                expand_floor(slice, fp, node.has_terms, node.floor_data_fp)?
            {
                // `hasTerms == false` means the block holds nothing but
                // pointers to further-nested sub-blocks.
                if has_terms {
                    out.push((block_fp, prefix.clone()));
                }
            }
        }
        for (label, child) in trie_children(slice, node)? {
            prefix.push(label);
            let r = collect_leaf_blocks(slice, &child, depth + 1, prefix, out);
            prefix.pop();
            r?;
        }
        Ok(())
    }

    /// `DataOutput::write_vlong` without its non-negative assertion: the
    /// ten-group encoding a corrupt file can legally contain.
    fn write_vlong_allowing_negative(out: &mut Vec<u8>, v: i64) {
        let mut v = v as u64;
        while v & !0x7F != 0 {
            out.push(((v & 0x7F) as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
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
            doc_values_type: crate::field_infos::DocValuesType::None,
            doc_values_skip_index_type: crate::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: crate::field_infos::VectorEncoding::Float32,
            vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
        }
    }

    /// Hand-builds a single-field, single-block `.tim`/`.tip`/`.tmd` triple
    /// (terms `["a", "ab", "b"]`, docFreq/totalTermFreq = 1/1, 2/3, 1/1) —
    /// this port's own encoder, test-only, to exercise error/boundary paths
    /// a real (small) fixture never reaches. Mirrors the pattern used by
    /// `codec_util.rs`/`segment_info.rs`'s own test-only encoders.
    struct Builder {
        id: [u8; ID_LENGTH],
        suffix: String,
        ov: Overrides,
    }

    /// Deliberate corruptions [`Builder`] can bake into an otherwise
    /// well-formed, correctly-footered triple, so a test can aim one header
    /// field at a decoder path without hand-assembling the whole file.
    #[derive(Default, Clone)]
    struct Overrides {
        /// `.tmd` `numFields`.
        num_fields: Option<i32>,
        /// `.tmd` `minTerm`'s length prefix (the bytes written stay the
        /// real ones).
        min_term_len: Option<i32>,
        /// `.tmd` `rootFP`.
        root_fp: Option<i64>,
        /// The per-term `totalTermFreq - docFreq` vlong in the `.tim` stats
        /// region, for every term.
        ttf_delta: Option<i64>,
        /// The `.tim` block header's `entCount`.
        ent_count: Option<u32>,
        /// `(declared region length, Some(byte) => write it as `allEqual`)`
        /// for the `.tim` suffix-lengths region.
        suffix_lengths_region: Option<(usize, Option<u8>)>,
    }

    impl Builder {
        fn new() -> Self {
            Builder {
                id: [7u8; ID_LENGTH],
                suffix: String::new(),
                ov: Overrides::default(),
            }
        }

        fn build(
            &self,
            index_options: IndexOptions,
            terms: &[(&str, u32, u64)],
        ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
            // .tim
            let mut tim = Vec::new();
            codec_util::write_index_header(
                &mut tim,
                TERMS_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            let block_fp = tim.len();

            let ent_count = self.ov.ent_count.unwrap_or(terms.len() as u32);
            let code = (ent_count << 1) | 1; // isLastInFloor
            tim.write_vint(code as i32);

            let mut suffix_bytes = Vec::new();
            let mut suffix_lengths = Vec::new();
            let mut stats = Vec::new();
            for (term, doc_freq, total_term_freq) in terms {
                suffix_bytes.extend_from_slice(term.as_bytes());
                suffix_lengths.write_vint(term.len() as i32);
                let token = (*doc_freq as i32) << 1; // never singleton-run-encoded, for test simplicity
                stats.write_vint(token);
                if index_options != IndexOptions::Docs {
                    let delta = self
                        .ov
                        .ttf_delta
                        .unwrap_or((*total_term_freq as i64) - (*doc_freq as i64));
                    stats.write_vlong(delta);
                }
            }

            let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // isLeafBlock, NO_COMPRESSION
            tim.write_vlong(code_l as i64);
            tim.write_bytes(&suffix_bytes);

            match self.ov.suffix_lengths_region {
                None => {
                    tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
                    tim.write_bytes(&suffix_lengths);
                }
                Some((declared, None)) => {
                    tim.write_vint((declared as i32) << 1);
                    tim.write_bytes(&suffix_lengths);
                }
                Some((declared, Some(byte))) => {
                    tim.write_vint(((declared as i32) << 1) | 1);
                    tim.push(byte);
                }
            }

            tim.write_vint(stats.len() as i32);
            tim.write_bytes(&stats);

            // Postings metadata: one entry per term via the bit=0
            // (docStartFP-delta) branch, legal regardless of `absolute` --
            // these seek_exact-only tests don't exercise postings decode, so
            // the fake docStartFP/singletonDocID values are never read back.
            let mut meta = Vec::new();
            for (_, doc_freq, _) in terms {
                meta.write_vlong(10 << 1);
                if *doc_freq == 1 {
                    meta.write_vint(0);
                }
            }
            tim.write_vint(meta.len() as i32);
            tim.write_bytes(&meta);

            codec_util::write_footer(&mut tim);

            // .tip: root node (SIGN_NO_CHILDREN), hasTerms, no floor.
            let mut tip = Vec::new();
            codec_util::write_index_header(
                &mut tip,
                TERMS_INDEX_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            let index_start = tip.len();
            let root_fp = 0usize;
            let output_fp_bytes = 8usize; // keep it simple: always 8 bytes
            let header = (SIGN_NO_CHILDREN as u8)
                | ((output_fp_bytes as u8 - 1) << 2)
                | (LEAF_NODE_HAS_TERMS as u8);
            tip.push(header);
            tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
            tip.extend_from_slice(&0u64.to_le_bytes()); // 8-byte over-read pad
            let index_end = tip.len();
            codec_util::write_footer(&mut tip);

            // .tmd
            let mut tmd = Vec::new();
            codec_util::write_index_header(
                &mut tmd,
                TERMS_META_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            codec_util::write_index_header(
                &mut tmd,
                POSTINGS_TERMS_CODEC,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            tmd.write_vint(POSTINGS_BLOCK_SIZE);

            tmd.write_vint(self.ov.num_fields.unwrap_or(1)); // numFields
            tmd.write_vint(0); // field number
            let num_terms = terms.len() as i64;
            tmd.write_vlong(num_terms);
            let sum_doc_freq: i64 = terms.iter().map(|(_, d, _)| *d as i64).sum();
            let sum_total_term_freq: i64 = if index_options == IndexOptions::Docs {
                sum_doc_freq
            } else {
                terms.iter().map(|(_, _, t)| *t as i64).sum()
            };
            if index_options != IndexOptions::Docs {
                tmd.write_vlong(sum_total_term_freq);
            }
            tmd.write_vlong(sum_doc_freq);
            tmd.write_vint(1); // docCount
            let min_term = terms[0].0.as_bytes();
            let max_term = terms[terms.len() - 1].0.as_bytes();
            tmd.write_vint(self.ov.min_term_len.unwrap_or(min_term.len() as i32));
            tmd.write_bytes(min_term);
            tmd.write_vint(max_term.len() as i32);
            tmd.write_bytes(max_term);
            tmd.write_vlong(index_start as i64);
            match self.ov.root_fp {
                // `DataOutput::write_vlong` refuses a negative value (Java's
                // `writeVLong` asserts the same), but the *wire* format can
                // carry one in its tenth group -- and that is exactly what a
                // corrupt `.tmd` looks like to `as usize`.
                Some(v) => write_vlong_allowing_negative(&mut tmd, v),
                None => tmd.write_vlong(root_fp as i64),
            }
            tmd.write_vlong(index_end as i64);

            tmd.write_i64(tip.len() as i64); // indexLength
            tmd.write_i64((tim.len()) as i64); // termsLength
            codec_util::write_footer(&mut tmd);

            (tim, tip, tmd)
        }
    }

    #[test]
    fn seek_exact_found_and_not_found() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();
        assert_eq!(field.num_terms, 3);
        assert_eq!(field.sum_doc_freq, 4);
        assert_eq!(field.sum_total_term_freq, 5);
        assert_eq!(field.min_term, b"a");
        assert_eq!(field.max_term, b"b");

        assert_eq!(
            field.seek_exact(b"ab"),
            Some(TermStats {
                doc_freq: 2,
                total_term_freq: 3
            })
        );
        assert_eq!(
            field.seek_exact(b"a"),
            Some(TermStats {
                doc_freq: 1,
                total_term_freq: 1
            })
        );
        assert_eq!(field.seek_exact(b"missing"), None);
        assert_eq!(field.seek_exact(b""), None);
        assert!(fields.field("nope").is_none());
    }

    #[test]
    fn intersect_prefix_and_wildcard_over_materialized_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[
                ("apple", 1, 1),
                ("application", 1, 1),
                ("apply", 1, 1),
                ("banana", 1, 1),
                ("band", 1, 1),
            ],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        // Prefix: "app*" -> apple, application, apply (in sorted order).
        let pattern = WildcardPattern::new(b"app*");
        let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
        assert_eq!(
            got,
            vec![
                b"apple".to_vec(),
                b"application".to_vec(),
                b"apply".to_vec()
            ]
        );

        // "?" wildcard: "ban?" matches nothing here ("band" is 4 bytes so
        // "ban?" matches "band" exactly -- exercise it precisely).
        let pattern = WildcardPattern::new(b"ban?");
        let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
        assert_eq!(got, vec![b"band".to_vec()]);

        // No literal prefix ("*" in the middle only): "*ana*" -> banana.
        let pattern = WildcardPattern::new(b"*ana*");
        let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
        assert_eq!(got, vec![b"banana".to_vec()]);

        // Matches everything.
        let pattern = WildcardPattern::new(b"*");
        assert_eq!(field.intersect(&pattern).count(), 5);

        // Matches nothing: valid prefix range, no candidate satisfies the
        // rest of the pattern.
        let pattern = WildcardPattern::new(b"app??????");
        assert_eq!(field.intersect(&pattern).count(), 0);

        // Matches nothing: prefix outside the field's term range entirely.
        let pattern = WildcardPattern::new(b"zzz*");
        assert_eq!(field.intersect(&pattern).count(), 0);

        // Exact-match pattern (no wildcard bytes at all) behaves like
        // seek_exact.
        let pattern = WildcardPattern::new(b"banana");
        let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
        assert_eq!(got, vec![b"banana".to_vec()]);

        // PrefixQuery-shaped constructor.
        let pattern = WildcardPattern::prefix(b"ban");
        let got: Vec<Vec<u8>> = field.intersect(&pattern).map(|r| r.unwrap().0).collect();
        assert_eq!(got, vec![b"banana".to_vec(), b"band".to_vec()]);
    }

    /// Backs `FuzzyQuery::max_expansions`'s (task #221, `lucene-search`
    /// crate) selection-policy claim: `fuzzy_intersect` returns a lazy
    /// iterator over this field's already-fully-decoded, in-memory sorted
    /// entries (the whole term dictionary is materialized once at segment
    /// open, not decoded on demand here -- see this module's own doc
    /// comment), filtering one candidate at a time as it's pulled. `.take(n)`
    /// stops pulling once `n` matches are found, so it skips the fuzzy-match
    /// predicate/allocation for every entry past the cap, but it does not
    /// skip any term-dictionary decode work (that already happened). This
    /// test builds a term dictionary with more than 50 terms that all match
    /// the same fuzzy pattern (all single-byte substitutions of a
    /// fixed-length target, so every one is within `max_edits` of it) and
    /// checks that `.take(50)` yields exactly the first 50 in sorted order,
    /// while the untruncated iterator yields every one of them.
    #[test]
    fn fuzzy_intersect_take_truncates_the_lazy_walk() {
        let b = Builder::new();
        // 60 five-byte terms "aaaaa".."aaaaz"-ish, each one substitution away
        // from "aaaaa" in its last byte -- all within max_edits=1 of it, and
        // already in sorted order since they differ only in their last byte.
        // Offsets 0..60 land in ASCII 33..93 ('!'..']'), all single-byte
        // printable characters, so the resulting `String`s stay valid UTF-8
        // and byte-sort exactly like their trailing byte value.
        let mut owned_terms: Vec<String> = (0u8..60)
            .map(|i| format!("aaaa{}", (33 + i) as char))
            .collect();
        owned_terms.sort();
        let terms: Vec<(&str, u32, u64)> = owned_terms.iter().map(|t| (t.as_str(), 1, 1)).collect();
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, &terms);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        let pattern = FuzzyMatch::new(b"aaaaa", 1, 0, true);

        // Untruncated: every one of the 60 terms matches.
        assert_eq!(field.fuzzy_intersect(&pattern).count(), 60);

        // Truncated to `max_expansions`-shaped 50: exactly 50 matches, and
        // they are the first 50 in sorted term-dictionary order (not an
        // arbitrary/unstable 50).
        let capped: Vec<Vec<u8>> = field
            .fuzzy_intersect(&pattern)
            .take(50)
            .map(|r| r.unwrap().0)
            .collect();
        assert_eq!(capped.len(), 50);
        let expected: Vec<Vec<u8>> = owned_terms[..50]
            .iter()
            .map(|t| t.as_bytes().to_vec())
            .collect();
        assert_eq!(capped, expected);

        // A cap that never binds (more than the total match count) is a
        // no-op, same "fewer matches than the limit" regression-safety shape
        // the task requires.
        assert_eq!(field.fuzzy_intersect(&pattern).take(1000).count(), 60);
    }

    #[test]
    fn regexp_intersect_over_materialized_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[
                ("apple", 1, 1),
                ("application", 1, 1),
                ("apply", 1, 1),
                ("banana", 1, 1),
                ("band", 1, 1),
            ],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        // Literal-prefix-narrowed range: "appl.*" -> apple, application,
        // apply (in sorted order).
        let pattern = RegexpPattern::new(b"appl.*").unwrap();
        let got: Vec<Vec<u8>> = field
            .regexp_intersect(&pattern)
            .map(|r| r.unwrap().0)
            .collect();
        assert_eq!(
            got,
            vec![
                b"apple".to_vec(),
                b"application".to_vec(),
                b"apply".to_vec()
            ]
        );

        // Alternation has no useful literal prefix (falls back to a full
        // scan) but still matches correctly.
        let pattern = RegexpPattern::new(b"banana|band").unwrap();
        let got: Vec<Vec<u8>> = field
            .regexp_intersect(&pattern)
            .map(|r| r.unwrap().0)
            .collect();
        assert_eq!(got, vec![b"banana".to_vec(), b"band".to_vec()]);

        // Whole-term-match: "ban" alone matches neither "banana" nor "band".
        let pattern = RegexpPattern::new(b"ban").unwrap();
        assert_eq!(field.regexp_intersect(&pattern).count(), 0);

        // Matches nothing: prefix outside the field's term range entirely.
        let pattern = RegexpPattern::new(b"zzz.*").unwrap();
        assert_eq!(field.regexp_intersect(&pattern).count(), 0);
    }

    /// The dead-prefix skip must never change *which* terms come back, only
    /// how many are tested. Checked against a brute-force filter over a
    /// dictionary shaped like the search benchmark's (`t0`..`t2999`), on the
    /// interior-constrained pattern family the skip exists for.
    #[test]
    fn regexp_intersect_skip_agrees_with_a_brute_force_scan() {
        let terms: Vec<String> = (0..3000).map(|i| format!("t{i}")).collect();
        let mut sorted: Vec<&str> = terms.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        let entries: Vec<(&str, u32, u64)> = sorted.iter().map(|t| (*t, 1u32, 1u64)).collect();

        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, &entries);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        for src in [
            "t1[0-9]",
            "t1.",
            "t[0-9]{3}",
            "t.*9",
            "t1|t22|t333",
            "t9*",
            "t(1|2)[0-9][0-9]",
            "zzz.*",
            "t1[0-9]&t.*",
            "t<1-30>",
        ] {
            let pattern = RegexpPattern::new(src.as_bytes()).unwrap();
            let expected: Vec<&str> = sorted
                .iter()
                .copied()
                .filter(|t| pattern.matches(t.as_bytes()))
                .collect();
            let got: Vec<String> = field
                .regexp_intersect(&pattern)
                .map(|r| String::from_utf8(r.unwrap().0).unwrap())
                .collect();
            let expected: Vec<String> = expected.iter().map(|t| t.to_string()).collect();
            assert_eq!(got, expected, "pattern {src}");
        }
    }

    #[test]
    fn prefix_upper_bound_handles_ff_bytes_and_empty() {
        assert_eq!(prefix_upper_bound(b""), None);
        assert_eq!(prefix_upper_bound(&[0xFF]), None);
        assert_eq!(prefix_upper_bound(&[0xFF, 0xFF]), None);
        assert_eq!(prefix_upper_bound(b"a"), Some(b"b".to_vec()));
        assert_eq!(prefix_upper_bound(&[b'a', 0xFF]), Some(vec![b'b']));
        assert_eq!(prefix_upper_bound(b"app"), Some(b"apq".to_vec()));
    }

    #[test]
    fn single_term_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        assert_eq!(field.min_term, field.max_term);
        assert_eq!(
            field.seek_exact(b"only"),
            Some(TermStats {
                doc_freq: 1,
                total_term_freq: 1
            })
        );
        assert_eq!(field.seek_exact(b"other"), None);
    }

    #[test]
    fn terms_enum_next_walks_all_terms_in_order() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();
        let mut it = field.iter();
        assert_eq!(it.current(), None);
        assert_eq!(
            it.next(),
            Some((
                b"a".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );
        assert_eq!(
            it.current(),
            Some((
                b"a".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );
        assert_eq!(
            it.next(),
            Some((
                b"ab".as_slice(),
                TermStats {
                    doc_freq: 2,
                    total_term_freq: 3
                }
            ))
        );
        assert_eq!(
            it.next(),
            Some((
                b"b".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );
        assert_eq!(it.next(), None);
        // Idempotent past the end.
        assert_eq!(it.next(), None);
        assert_eq!(it.current(), None);
    }

    #[test]
    fn terms_enum_seek_ceil_found_notfound_end_and_continues() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        // Exact match.
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"ab"), SeekStatus::Found);
        assert_eq!(
            it.current(),
            Some((
                b"ab".as_slice(),
                TermStats {
                    doc_freq: 2,
                    total_term_freq: 3
                }
            ))
        );
        // next() after seekCeil continues past the found term.
        assert_eq!(
            it.next(),
            Some((
                b"b".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );

        // Ceiling match: falls strictly between "a" and "ab".
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"aa"), SeekStatus::NotFound);
        assert_eq!(
            it.current(),
            Some((
                b"ab".as_slice(),
                TermStats {
                    doc_freq: 2,
                    total_term_freq: 3
                }
            ))
        );

        // Before the first term: ceiling is the first term.
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b""), SeekStatus::NotFound);
        assert_eq!(
            it.current(),
            Some((
                b"a".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );

        // After the last term: no ceiling exists.
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"z"), SeekStatus::End);
        assert_eq!(it.current(), None);
        assert_eq!(it.next(), None);
    }

    /// `TermsEnum`'s past-the-end cursor states. A real writer never emits a
    /// zero-term field (`open()` itself rejects `numTerms <= 0`), so the
    /// smallest dictionary that can stand in for one is a single-term field
    /// walked past its only term: `next()` must stay idempotent at the end
    /// (Java's `eof`), `current()` must report nothing, and a `seek_ceil`
    /// past the last term must report `End` and leave the cursor unpositioned.
    #[test]
    fn terms_enum_past_the_end_is_idempotent() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 1).unwrap();
        let field = fields.field("text").unwrap();

        let mut it = field.iter();
        assert!(it.next().is_some());
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
        assert_eq!(it.current(), None);

        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"zzz"), SeekStatus::End);
        assert_eq!(it.current(), None);
        assert_eq!(it.next(), None);

        // A seek *after* end-of-terms must still answer for its target, not
        // fall out of `next()`'s `eof` guard: `SegmentTermsEnum::reset` has to
        // clear `eof`, and only a walk-then-seek order can prove it does.
        let mut it = field.iter();
        while it.next().is_some() {}
        assert_eq!(it.seek_ceil(b"only"), SeekStatus::Found);
        assert_eq!(it.current().unwrap().0, b"only");
        assert_eq!(it.seek_ceil(b"a"), SeekStatus::NotFound);
        assert_eq!(it.current().unwrap().0, b"only");
        assert_eq!(it.seek_ceil(b"zzz"), SeekStatus::End);

        // A field with no `.tim`/`.tip`/`.tmd` at all has no fields to
        // iterate in the first place.
        assert!(BlockTreeFields::empty().field("text").is_none());
        assert_eq!(BlockTreeFields::empty().iter_fields().count(), 0);
    }

    #[test]
    fn terms_enum_single_term_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        let mut it = field.iter();
        assert_eq!(
            it.next(),
            Some((
                b"only".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );
        assert_eq!(it.next(), None);

        let mut it2 = field.iter();
        assert_eq!(it2.seek_ceil(b"only"), SeekStatus::Found);
        assert_eq!(it2.next(), None);
    }

    #[test]
    fn docs_only_index_options_omits_total_term_freq_field() {
        // IndexOptions::Docs never writes a distinct sumTotalTermFreq, and
        // per-term stats never write the extra totalTermFreq vlong either.
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("x", 3, 3), ("y", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        assert_eq!(field.sum_total_term_freq, field.sum_doc_freq);
        assert_eq!(
            field.seek_exact(b"x"),
            Some(TermStats {
                doc_freq: 3,
                total_term_freq: 3
            })
        );
    }

    /// `CodecUtil.retrieveChecksum(IndexInput, long expectedLength)` rejects
    /// "file too long" as well as "truncated file": the recorded length must
    /// equal the file's own. Only the truncation half used to be checked, so
    /// a `.tmd` claiming a shorter `.tip`/`.tim` than the one it describes
    /// was accepted -- and the footer was then read from the wrong offset.
    #[test]
    fn recorded_tip_or_tim_length_shorter_than_the_file_is_rejected() {
        let b = Builder::new();
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
        // Sanity: unmodified, it opens.
        assert!(open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).is_ok());

        for which in ["tip", "tim"] {
            let mut tim = tim.clone();
            let mut tip = tip.clone();
            // One trailing byte past the recorded length -- Lucene's "file
            // too long" case.
            if which == "tip" {
                tip.push(0);
            } else {
                tim.push(0);
            }
            let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap_err();
            assert!(
                matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
                "{which}: {err:?}"
            );
        }
    }

    #[test]
    fn invalid_num_fields_rejected() {
        let mut tmd = Vec::new();
        let id = [1u8; ID_LENGTH];
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(-1); // invalid numFields
        codec_util::write_footer(&mut tmd);

        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tim);
        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tip);

        let fis = FieldInfos { fields: vec![] };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::InvalidNumFields(-1)));
    }

    #[test]
    fn unexpected_postings_block_size_rejected() {
        let mut tmd = Vec::new();
        let id = [1u8; ID_LENGTH];
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(128); // wrong block size
        codec_util::write_footer(&mut tmd);

        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tim);
        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tip);

        let fis = FieldInfos { fields: vec![] };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::UnexpectedBlockSize { found: 128 }));
    }

    #[test]
    fn multi_children_node_with_invalid_strategy_code_rejected() {
        // childSaveStrategy code 3 doesn't exist (only 0/1/2 are defined) ->
        // a structural Corrupted error, not silently misdecoded.
        let mut slice = vec![0u8; 24];
        let term: u32 = SIGN_MULTI_CHILDREN | (3 << 9); // invalid strategy code
        slice[0..3].copy_from_slice(&term.to_le_bytes()[0..3]);
        let err = load_node(&slice, 0)
            .and_then(|node| lookup_child(&slice, &node, 1))
            .unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn single_child_trie_node_with_output_and_floor_round_trips() {
        // Build a leaf child at fp=0, then a SIGN_SINGLE_CHILD_WITH_OUTPUT
        // parent at fp=16 with its own output+floor data, pointing at the
        // child via a 1-byte delta -- exercises loadSingleChildNode's
        // "has output" branch end to end (TrieReader.loadSingleChildNode).
        let mut slice = vec![0u8; 40];
        // Child: SIGN_NO_CHILDREN, 1-byte output fp = 42, hasTerms.
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 42;

        let parent_fp = 16usize;
        let child_delta_fp: u8 = parent_fp as u8; // 16
        let label: u8 = b'x';
        // encodeFP: (floor?1:0) | (hasTerms?2:0) | (fp << 2); output fp = 20.
        let encoded_fp: u64 = NON_LEAF_NODE_HAS_FLOOR | NON_LEAF_NODE_HAS_TERMS | (20 << 2);
        assert!(encoded_fp <= 0xFF, "fits in 1 byte for this test");

        // childDeltaFpBytesMinus1 = 0 (1 byte), encodedOutputFpBytesMinus1 = 0 (1 byte)
        let term: u32 = SIGN_SINGLE_CHILD_WITH_OUTPUT;
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        slice[parent_fp + 1] = label;
        slice[parent_fp + 2] = child_delta_fp;
        slice[parent_fp + 3] = encoded_fp as u8;
        // Floor data right after the 1-byte encoded output fp: one follow
        // block, floorLeadByte='y', code = (5 << 1) | 1 (hasTerms).
        let floor_fp = parent_fp + 4;
        slice[floor_fp] = 1; // numFollowFloorBlocks vint
        slice[floor_fp + 1] = b'y';
        slice[floor_fp + 2] = (5 << 1) | 1; // code vlong

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.sign, SIGN_SINGLE_CHILD_WITH_OUTPUT);
        assert_eq!(node.output_fp, Some(20));
        assert!(node.has_terms);
        assert_eq!(node.floor_data_fp, Some(floor_fp));
        assert_eq!(node.min_children_label, label);

        let blocks = expand_floor(&slice, 20, true, node.floor_data_fp).unwrap();
        assert_eq!(blocks, vec![(20, true), (25, true)]);

        let child_fp = parent_fp - node.child_delta_fp as usize;
        assert_eq!(child_fp, 0);
        let child = load_node(&slice, child_fp).unwrap();
        assert_eq!(child.output_fp, Some(42));
        assert!(child.has_terms);
    }

    #[test]
    fn single_child_without_output_has_no_own_block() {
        let mut slice = vec![0u8; 24];
        let parent_fp = 8usize;
        // childDeltaFpBytesMinus1 = 0 (1 byte)
        let term: u32 = SIGN_SINGLE_CHILD_WITHOUT_OUTPUT;
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        slice[parent_fp + 1] = b'q';
        slice[parent_fp + 2] = 8; // child delta fp -> child at fp 0

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.output_fp, None);
        assert!(!node.has_terms);
        assert_eq!(node.min_children_label, b'q');
    }

    /// Builds a `SIGN_MULTI_CHILDREN` node (no output of its own) with two
    /// children under the given `strategy`, and asserts `multi_children_fps`
    /// recovers exactly the two child fps regardless of which
    /// `ChildSaveStrategy` encoded them -- `TrieReader.lookupChild`'s three
    /// strategies (`BITS`/`ARRAY`/`REVERSE_ARRAY`), generalized to "list all"
    /// (see [`multi_children_labels_and_fps`]'s doc comment for why).
    fn build_and_check_multi_children(strategy: u32, strategy_bytes_region: &[u8]) {
        let mut slice = vec![0u8; 32];
        // Child A: leaf, output fp = 10, hasTerms, at fp 0.
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 10;
        // Child B: leaf, output fp = 20, hasTerms, at fp 2.
        slice[2] = LEAF_NODE_HAS_TERMS as u8;
        slice[3] = 20;

        let parent_fp = 8usize;
        let min_label = b'a';
        let strategy_bytes = strategy_bytes_region.len();
        // childrenDeltaFpBytesMinus1 = 0 (1 byte), no output
        let term: u32 = SIGN_MULTI_CHILDREN
            | (strategy << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());

        let strategy_fp = parent_fp + 3;
        slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(strategy_bytes_region);
        let fps_fp = strategy_fp + strategy_bytes;
        slice[fps_fp] = parent_fp as u8; // delta to child A
        slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.output_fp, None);
        assert_eq!(node.child_save_strategy, strategy);
        assert_eq!(node.strategy_bytes, strategy_bytes);

        let mut child_fps: Vec<usize> = trie_children(&slice, &node)
            .unwrap()
            .into_iter()
            .map(|(_, child)| child.fp)
            .collect();
        child_fps.sort_unstable();
        assert_eq!(child_fps, vec![0, 2]);

        let mut collected = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut collected).unwrap();
        let mut fps: Vec<u64> = collected.iter().map(|(fp, _)| *fp).collect();
        fps.sort_unstable();
        assert_eq!(fps, vec![10, 20]);
    }

    #[test]
    fn multi_children_array_strategy() {
        // ARRAY: labels[1..] stored explicitly ('b' = 0x62), minLabel='a'
        // implicit.
        build_and_check_multi_children(CHILD_STRATEGY_ARRAY, b"b");
    }

    #[test]
    fn multi_children_bits_strategy() {
        // BITS: byteDistance = 'b'-'a'+1 = 2 -> 1 byte; bit0 (label 'a')
        // and bit1 (label 'b') both set -> 0b011 = 3.
        build_and_check_multi_children(CHILD_STRATEGY_BITS, &[0b011]);
    }

    #[test]
    fn multi_children_reverse_array_strategy() {
        // REVERSE_ARRAY: byte0 = maxLabel ('b'), no missing labels between
        // 'a' and 'b' (they're consecutive) -> exactly 1 byte.
        build_and_check_multi_children(CHILD_STRATEGY_REVERSE_ARRAY, b"b");
    }

    #[test]
    fn multi_children_reverse_array_strategy_with_gap() {
        // Labels 'a' and 'd' (a gap of 'b','c' in between): byteDistance=4,
        // labelCnt=2 -> strategyBytes = 4-2+1 = 3: [maxLabel='d', 'b', 'c'].
        let mut slice = vec![0u8; 32];
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 10;
        slice[2] = LEAF_NODE_HAS_TERMS as u8;
        slice[3] = 20;

        let parent_fp = 8usize;
        let min_label = b'a';
        let strategy_bytes_region = *b"dbc";
        let strategy_bytes = strategy_bytes_region.len();
        let term: u32 = SIGN_MULTI_CHILDREN
            | (CHILD_STRATEGY_REVERSE_ARRAY << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        let strategy_fp = parent_fp + 3;
        slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(&strategy_bytes_region);
        let fps_fp = strategy_fp + strategy_bytes;
        slice[fps_fp] = parent_fp as u8; // delta to child A (fp 0)
        slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B (fp 2)

        let node = load_node(&slice, parent_fp).unwrap();
        let mut child_fps: Vec<usize> = trie_children(&slice, &node)
            .unwrap()
            .into_iter()
            .map(|(_, child)| child.fp)
            .collect();
        child_fps.sort_unstable();
        assert_eq!(child_fps, vec![0, 2]);
        // The two labels the encoding lists as *absent* between 'a' and 'd'
        // must miss, not resolve to a neighbour.
        assert!(lookup_child(&slice, &node, b'b').unwrap().is_none());
        assert!(lookup_child(&slice, &node, b'c').unwrap().is_none());
    }

    /// `REVERSE_ARRAY` with *two* gaps, so `lookup`'s bisection over the
    /// absent-label list takes its `midLabel > target` branch and returns
    /// through the `target - minLabel - low` arithmetic rather than hitting an
    /// exact miss. Labels 'a', 'c', 'e'; absent 'b', 'd'; strategy region
    /// `[maxLabel='e', 'b', 'd']` (`needBytes` = 5 - 3 + 1 = 3).
    #[test]
    fn multi_children_reverse_array_strategy_with_two_gaps() {
        let mut slice = vec![0u8; 40];
        // Three leaf children at fp 0, 2, 4 with distinct output fps.
        for (i, out) in [(0usize, 10u8), (2, 20), (4, 30)] {
            slice[i] = LEAF_NODE_HAS_TERMS as u8;
            slice[i + 1] = out;
        }

        let parent_fp = 12usize;
        let min_label = b'a';
        let strategy_bytes_region = *b"ebd";
        let strategy_bytes = strategy_bytes_region.len();
        let term: u32 = SIGN_MULTI_CHILDREN
            | (CHILD_STRATEGY_REVERSE_ARRAY << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        let strategy_fp = parent_fp + 3;
        slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(&strategy_bytes_region);
        let fps_fp = strategy_fp + strategy_bytes;
        slice[fps_fp] = parent_fp as u8; // 'a' -> fp 0
        slice[fps_fp + 1] = (parent_fp - 2) as u8; // 'c' -> fp 2
        slice[fps_fp + 2] = (parent_fp - 4) as u8; // 'e' -> fp 4

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.child_save_strategy, CHILD_STRATEGY_REVERSE_ARRAY);
        for (label, fp, output) in [(b'a', 0usize, 10u64), (b'c', 2, 20), (b'e', 4, 30)] {
            let child = lookup_child(&slice, &node, label)
                .unwrap()
                .unwrap_or_else(|| panic!("label {} should resolve", label as char));
            assert_eq!(child.fp, fp, "label {}", label as char);
            assert_eq!(child.output_fp, Some(output), "label {}", label as char);
        }
        for label in *b"`bdf" {
            assert!(
                lookup_child(&slice, &node, label).unwrap().is_none(),
                "label {} should miss",
                label as char
            );
        }
    }

    #[test]
    fn collect_leaf_blocks_skips_output_with_no_terms() {
        // A node whose own output has hasTerms=false (a pointer-only block,
        // e.g. a coarser prefix the writer recursed past rather than
        // floor-split) contributes no block of its own -- it's skipped, not
        // an error, since any real terms under that prefix are reachable
        // through this node's own children instead (see
        // `collect_leaf_blocks`'s doc comment). Here the node has
        // `SIGN_NO_CHILDREN`, so skipping it means zero blocks collected.
        let mut slice = vec![0u8; 16];
        slice[0] = 0; // sign=0, fpBytesMinus1=0, no LEAF_NODE_HAS_TERMS bit
        slice[1] = 5;
        let node = load_node(&slice, 0).unwrap();
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn expand_floor_rejects_negative_num_follow() {
        let mut buf = Vec::new();
        buf.write_vint(-1);
        let err = expand_floor(&buf, 0, true, Some(0)).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn expand_floor_no_floor_data_returns_just_the_base_block() {
        let blocks = expand_floor(&[], 7, true, None).unwrap();
        assert_eq!(blocks, vec![(7, true)]);
    }

    /// End-to-end: a field whose terms span two floor sub-blocks under one
    /// leaf trie node (`LEAF_NODE_HAS_FLOOR`), with a hand-built
    /// `.tim`/`.tip`/`.tmd` triple no real (small) fixture reaches.
    ///
    /// Exercises the two things floor blocks demand of the lazy reader:
    /// `scanToFloorFrame` must pick the *one* sub-block a target's lead byte
    /// falls in (so a term in the second block is not looked for in the
    /// first), and `next()` must chain from one floor block to the next off
    /// the `isLastInFloor` bit rather than stopping at the first block's end.
    #[test]
    fn open_floor_field_walks_both_blocks() {
        let id = [9u8; ID_LENGTH];
        let suffix = String::new();

        fn write_leaf_block(
            tim: &mut Vec<u8>,
            terms: &[(&str, u32, u64)],
            is_last_in_floor: bool,
        ) -> usize {
            let block_fp = tim.len();
            let ent_count = terms.len() as u32;
            tim.write_vint(((ent_count << 1) | u32::from(is_last_in_floor)) as i32);

            let mut suffix_bytes = Vec::new();
            let mut suffix_lengths = Vec::new();
            let mut stats = Vec::new();
            for (term, doc_freq, total_term_freq) in terms {
                suffix_bytes.extend_from_slice(term.as_bytes());
                suffix_lengths.write_vint(term.len() as i32);
                stats.write_vint((*doc_freq as i32) << 1);
                stats.write_vlong((*total_term_freq as i64) - (*doc_freq as i64));
            }
            let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04;
            tim.write_vlong(code_l as i64);
            tim.write_bytes(&suffix_bytes);
            tim.write_vint((suffix_lengths.len() as i32) << 1);
            tim.write_bytes(&suffix_lengths);
            tim.write_vint(stats.len() as i32);
            tim.write_bytes(&stats);

            let mut meta = Vec::new();
            for (_, doc_freq, _) in terms {
                meta.write_vlong(10 << 1);
                if *doc_freq == 1 {
                    meta.write_vint(0);
                }
            }
            tim.write_vint(meta.len() as i32);
            tim.write_bytes(&meta);
            block_fp
        }

        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, &suffix);
        // Terms are sorted within each block, and split across the floor
        // boundary at 'm' exactly as `Lucene103BlockTreeTermsWriter` would.
        let block0_fp = write_leaf_block(&mut tim, &[("a", 1, 1), ("b", 1, 1)], false);
        let block1_fp = write_leaf_block(&mut tim, &[("m", 1, 1), ("z", 2, 5)], true);
        codec_util::write_footer(&mut tim);

        let mut tip = Vec::new();
        codec_util::write_index_header(
            &mut tip,
            TERMS_INDEX_CODEC_NAME,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        let index_start = tip.len();
        // Leaf root, floor: header | outputFp(block0_fp, 8 bytes to keep it
        // simple) | floor data.
        let header = LEAF_NODE_HAS_TERMS as u8 | LEAF_NODE_HAS_FLOOR as u8 | (7 << 2);
        tip.push(header);
        tip.extend_from_slice(&(block0_fp as u64).to_le_bytes());
        tip.write_vint(1); // numFollowFloorBlocks
        tip.write_byte(b'm'); // floorLeadByte for block1
        tip.write_vlong((((block1_fp - block0_fp) as i64) << 1) | 1); // code, hasTerms
        tip.extend_from_slice(&0u64.to_le_bytes()); // over-read pad
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        let mut tmd = Vec::new();
        codec_util::write_index_header(
            &mut tmd,
            TERMS_META_CODEC_NAME,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        codec_util::write_index_header(
            &mut tmd,
            POSTINGS_TERMS_CODEC,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1); // numFields
        tmd.write_vint(0); // field number
        tmd.write_vlong(4); // numTerms
        tmd.write_vlong(8); // sumTotalTermFreq = 1+1+1+5
        tmd.write_vlong(5); // sumDocFreq = 1+1+1+2
        tmd.write_vint(1); // docCount
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vint(1);
        tmd.write_bytes(b"z");
        tmd.write_vlong(index_start as i64);
        tmd.write_vlong(0); // root fp within index slice
        tmd.write_vlong(index_end as i64);
        tmd.write_i64(tip.len() as i64); // indexLength: the whole file, footer included
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &id, &suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        assert_eq!(field.num_terms, 4);

        // Every term is reachable, whichever floor sub-block holds it.
        for (term, expected) in [("a", (1, 1)), ("b", (1, 1)), ("m", (1, 1)), ("z", (2, 5))] {
            let stats = field.seek_exact(term.as_bytes()).unwrap();
            assert_eq!(stats.doc_freq, expected.0, "term={term}");
            assert_eq!(stats.total_term_freq, expected.1, "term={term}");
        }
        // A miss whose lead byte falls in the *second* floor block, and one
        // that falls in the first: both must scan only their own block.
        assert!(field.seek_exact(b"missing").is_none());
        assert!(field.seek_exact(b"c").is_none());

        // `next()` must chain across the floor boundary.
        let mut it = field.iter();
        let mut walked = Vec::new();
        while let Some((term, stats)) = it.next() {
            walked.push((String::from_utf8(term.to_vec()).unwrap(), stats.doc_freq));
        }
        assert_eq!(
            walked,
            vec![
                ("a".to_string(), 1),
                ("b".to_string(), 1),
                ("m".to_string(), 1),
                ("z".to_string(), 2),
            ]
        );

        // And so must `seek_ceil` into the second block from a target in the
        // first block's range.
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"c"), SeekStatus::NotFound);
        assert_eq!(it.current().unwrap().0, b"m");
    }

    #[test]
    fn empty_terms_block_rejected() {
        let id = [2u8; ID_LENGTH];
        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        let block_fp = tim.len();
        tim.write_vint(1); // entCount=0, isLastInFloor=true -> code = 0<<1|1 = 1
        codec_util::write_footer(&mut tim);

        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        let index_start = tip.len();
        let header = LEAF_NODE_HAS_TERMS as u8; // SIGN_NO_CHILDREN, 1-byte fp
        tip.push(header);
        tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
        tip.extend_from_slice(&0u64.to_le_bytes());
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        let mut tmd = Vec::new();
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1);
        tmd.write_vint(0);
        tmd.write_vlong(1); // numTerms must be >0 to pass that check; block itself will be empty
        tmd.write_vlong(0); // sumDocFreq (Docs aliasing)
        tmd.write_vint(0); // docCount
        tmd.write_vint(0);
        tmd.write_bytes(&[]);
        tmd.write_vint(0);
        tmd.write_bytes(&[]);
        tmd.write_vlong(index_start as i64);
        tmd.write_vlong(0);
        tmd.write_vlong(index_end as i64);
        tmd.write_i64(tip.len() as i64); // indexLength: the whole file, footer included
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        // `open` no longer reads any `.tim` block (see the module doc), so a
        // block this corrupt is rejected where real Lucene rejects it too:
        // from inside the lookup that loads it.
        let fields = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap();
        let field = fields.field("f").unwrap();
        let err = field.try_seek_exact(b"").unwrap_err();
        assert!(matches!(err, Error::Store(_)));
        // The infallible spelling degrades it to "no such term".
        assert!(field.seek_exact(b"").is_none());
    }

    /// A corrupt block is now discovered where real Lucene discovers it --
    /// inside the lookup that loads it, not at `open` -- and every lookup's
    /// `Result`-returning form surfaces it. The infallible spellings degrade
    /// to "no such term"/end-of-terms instead, which is the one behaviour
    /// this batch deliberately traded away (see the module doc).
    #[test]
    fn a_corrupt_block_errors_from_the_lookup_not_from_open() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("alpha", 1, 1), ("beta", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::Docs)],
        };

        // Find the block's suffix-lengths region and inflate the first
        // entry's length past the end of the suffix bytes, which is what
        // `scanToTermLeaf`'s cursor has to reject.
        let header = codec_util::index_header_length(TERMS_CODEC_NAME, &b.suffix);
        let mut r = SliceInput::new(&tim);
        r.seek(header).unwrap();
        let _code = r.read_vint().unwrap();
        let code_l = r.read_vlong().unwrap() as u64;
        let num_suffix_bytes = (code_l >> 3) as usize;
        r.seek(r.position() + num_suffix_bytes).unwrap();
        let _num_suffix_length_bytes = r.read_vint().unwrap();
        let first_length_at = r.position();
        assert_eq!(tim[first_length_at], b"alpha".len() as u8);

        let mut corrupt = tim.clone();
        corrupt[first_length_at] = 100;
        let fields = open(&corrupt, &tip, &tmd, &fis, &b.id, &b.suffix, 5)
            .expect("open reads no .tim block, so it cannot see this");
        let field = fields.field("text").unwrap();

        let err = field.try_seek_exact(b"alpha").unwrap_err();
        assert!(matches!(err, Error::Store(_)), "{err:?}");
        assert!(field.seek_exact(b"alpha").is_none());

        let mut it = field.iter();
        assert!(it.try_next().is_err());
        let mut it = field.iter();
        assert_eq!(it.next(), None);
        let mut it = field.iter();
        assert!(it.try_seek_ceil(b"alpha").is_err());
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"alpha"), SeekStatus::End);

        // The intersect iterators run over the same enum and now surface the
        // same error rather than ending quietly: a truncated term expansion
        // is a wrong hit set, not a smaller one.
        let pattern = WildcardPattern::new(b"a*");
        let items: Vec<_> = field.intersect(&pattern).collect();
        assert_eq!(items.len(), 1, "one error, then the walk is over");
        assert!(matches!(items[0], Err(Error::Store(_))), "{:?}", items[0]);
    }

    /// `SegmentTermsEnum.seekExact`'s "index exhausted and this frame has no
    /// terms" fast path: the trie routed us to a node whose own `.tim` block
    /// holds nothing but sub-block pointers, so the target cannot exist and
    /// no block is loaded at all.
    ///
    /// Uses `blocktree_child_strategies_index`, whose "arraystrat" field has
    /// exactly that shape at the root (see
    /// `open_field_with_no_terms_in_root_block_still_finds_every_term`), and
    /// picks a lead byte inside the field's min/max range that is *not* one
    /// of the root's five child labels, so the very first `lookupChild`
    /// misses while the current frame is the no-terms root.
    #[test]
    fn seek_exact_stops_at_a_trie_node_with_no_terms() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_child_strategies_index/"
        );
        let text = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTreeChildStrategies)");
        let kv: Vec<(String, String)> = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let get = |k: &str| {
            kv.iter()
                .find(|(kk, _)| kk == k)
                .map(|(_, v)| v.as_str())
                .unwrap_or_else(|| panic!("manifest key {k} missing"))
        };
        let mut id = [0u8; ID_LENGTH];
        let hex = get("id_hex");
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = get("segment_suffix").to_string();
        let max_doc: i32 = get("max_doc").parse().unwrap();
        let read = |n: &str| std::fs::read(format!("{dir}{n}.raw")).unwrap();
        let field_infos = crate::field_infos::parse(&read(get("fnm_file_name")), &id, "").unwrap();
        let fields = open(
            &read(get("tim_file_name")),
            &read(get("tip_file_name")),
            &read(get("tmd_file_name")),
            &field_infos,
            &id,
            &suffix,
            max_doc,
        )
        .unwrap();
        let field = fields.field("arraystrat").unwrap();

        // Every lead byte strictly between the field's min and max term that
        // has no root child: `seek_exact` must miss without loading a block.
        let lo = field.min_term[0];
        let hi = field.max_term[0];
        let mut checked = 0;
        for lead in (lo + 1)..hi {
            let target = [lead, b'x'];
            if field.seek_exact(&target).is_some() {
                continue;
            }
            assert_eq!(field.try_seek_exact(&target).unwrap(), None);
            checked += 1;
        }
        assert!(
            checked > 0,
            "expected at least one absent lead byte inside [{lo}, {hi}]"
        );
    }

    /// The pooled lookup scratch must never make concurrent searchers wait on
    /// each other: `BlockTreeFields` is shared by every search thread through
    /// one `Arc`, so a blocking lock here would serialize the whole term
    /// dictionary. A thread that loses the race runs on its own state and
    /// must get identical answers.
    #[test]
    fn concurrent_lookups_do_not_serialize_and_agree() {
        let b = Builder::new();
        let owned: Vec<String> = (0..200).map(|i| format!("term{i:04}")).collect();
        let mut sorted: Vec<&str> = owned.iter().map(String::as_str).collect();
        sorted.sort_unstable();
        let entries: Vec<(&str, u32, u64)> = sorted
            .iter()
            .enumerate()
            .map(|(i, t)| (*t, i as u32 + 1, i as u64 + 1))
            .collect();
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, &entries);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields =
            std::sync::Arc::new(open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 500).unwrap());

        let mut handles = Vec::new();
        for _ in 0..8 {
            let fields = std::sync::Arc::clone(&fields);
            let expected: Vec<(String, u32)> = entries
                .iter()
                .map(|(t, df, _)| ((*t).to_string(), *df))
                .collect();
            handles.push(std::thread::spawn(move || {
                let field = fields.field("text").unwrap();
                for _ in 0..20 {
                    for (term, df) in &expected {
                        let stats = field
                            .try_seek_exact(term.as_bytes())
                            .expect("intact bytes")
                            .unwrap_or_else(|| panic!("term {term} not found"));
                        assert_eq!(stats.doc_freq, *df as i32);
                    }
                    assert!(field.seek_exact(b"nope").is_none());
                }
            }));
        }
        for h in handles {
            h.join().expect("no thread panicked");
        }
    }

    /// A cloned `FieldTerms` shares the segment's `.tim`/`.tip` buffers and
    /// gets its own lookup scratch, so both copies keep working
    /// independently -- `BlockTreeFields` is `Clone` and callers hold it
    /// behind an `Arc`, so this is a real path.
    #[test]
    fn cloning_a_field_keeps_both_copies_usable() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("alpha", 3, 3), ("beta", 4, 4)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let copy = fields.clone();
        for f in [&fields, &copy] {
            let field = f.field("text").unwrap();
            assert_eq!(field.seek_exact(b"alpha").unwrap().doc_freq, 3);
            assert_eq!(field.seek_exact(b"beta").unwrap().doc_freq, 4);
            assert!(field.seek_exact(b"gamma").is_none());
        }
    }

    /// `open_shared` is `open` without the `.tim`/`.tip` copy: same answers,
    /// same errors, buffers the caller already owns.
    #[test]
    fn open_shared_reads_the_same_dictionary_without_copying() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("alpha", 3, 3), ("beta", 4, 4)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::Docs)],
        };
        let shared_tim: SharedBytes = Arc::new(tim.clone());
        let shared_tip: SharedBytes = Arc::new(tip.clone());
        let fields = open_shared(
            SharedBytes::clone(&shared_tim),
            SharedBytes::clone(&shared_tip),
            &tmd,
            &fis,
            &b.id,
            &b.suffix,
            5,
        )
        .unwrap();
        let field = fields.field("text").unwrap();
        assert_eq!(field.seek_exact(b"beta").unwrap().doc_freq, 4);
        // The caller's buffers are shared, not copied: the reader holds a
        // reference to the very same allocation.
        assert!(Arc::strong_count(&shared_tim) > 1);
    }

    #[test]
    fn read_bytes_ref_rejects_negative_length() {
        let mut buf = Vec::new();
        buf.write_vint(-1);
        let mut input = SliceInput::new(&buf);
        let err = read_bytes_ref(&mut input).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn load_node_leaf_eight_byte_output_fp() {
        // fpBytesMinus1 == 7 forces a fresh 8-byte read at fp+1.
        let mut slice = Vec::new();
        let header: u8 = LEAF_NODE_HAS_TERMS as u8 | (7 << 2); // sign=0, fpBytesMinus1=7
        slice.push(header);
        let big_fp: u64 = 0x0102_0304_0506_0708;
        slice.extend_from_slice(&big_fp.to_le_bytes()); // read fresh at fp+1
        slice.extend_from_slice(&0u64.to_le_bytes()); // over-read padding

        let node = load_node(&slice, 0).unwrap();
        assert_eq!(node.output_fp, Some(big_fp));
        assert!(node.has_terms);
    }

    #[test]
    fn load_node_rejects_truncated_slice() {
        let slice = [0u8; 4];
        let err = load_node(&slice, 0).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    /// Hand-builds a two-level `.tim` byte sequence: a leaf child block
    /// (term `"zz"`, docFreq/totalTermFreq 1/1) followed by a non-leaf parent
    /// block whose two entries are a real term (`"aa"`) and a sub-block
    /// pointer (key byte `b`) resolving back to the child block via
    /// `parent_fp - subCode` — the genuine "multi-level blocktree" case
    /// (`SegmentTermsEnumFrame.nextNonLeaf`'s `code & 1` sub-block bit),
    /// distinct from the `.tip` trie's own multi-level nesting (already
    /// covered by [`collect_leaf_blocks`]'s tests) and from floor blocks.
    /// Confirms `decode_block` recurses into the sub-block and reattaches its
    /// key byte as a prefix, producing all three terms in the same block's
    /// entry list.
    #[test]
    fn decode_block_recurses_into_sub_block() {
        let mut tim = Vec::new();

        // --- child (leaf) block: one term "zz", docFreq=1/totalTermFreq=1 ---
        let child_fp = tim.len();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
        let child_suffix = b"zz";
        let child_code_l = ((child_suffix.len() as u64) << 3) | 0x04; // leaf, no compression
        tim.write_vlong(child_code_l as i64);
        tim.write_bytes(child_suffix);
        tim.write_vint((1i32 << 1) | 1); // allEqual, logical len 1
        tim.write_byte(2); // suffix length 2
        let mut child_stats = Vec::new();
        child_stats.write_vint(1 << 1); // token&1==0, docFreq=1
        tim.write_vint(child_stats.len() as i32);
        tim.write_bytes(&child_stats);
        let mut child_meta = Vec::new();
        child_meta.write_vlong(10 << 1); // docStartFP delta=10, absolute
        child_meta.write_vint(0); // singleton_doc_id (docFreq==1)
        tim.write_vint(child_meta.len() as i32);
        tim.write_bytes(&child_meta);

        // --- parent (non-leaf) block: term "aa" + sub-block "b" -> child ---
        let parent_fp = tim.len();
        tim.write_vint((2 << 1) | 1); // entCount=2, isLastInFloor
        let parent_suffix_bytes = b"ab"; // "a" (term "aa"'s suffix) then "b" (sub-block key)
        let parent_code_l = (parent_suffix_bytes.len() as u64) << 3; // non-leaf, no compression
        tim.write_vlong(parent_code_l as i64);
        tim.write_bytes(parent_suffix_bytes);

        let mut suffix_lengths = Vec::new();
        suffix_lengths.write_vint(1 << 1); // entry 0: suffix len 1, not a sub-block
        suffix_lengths.write_vint((1 << 1) | 1); // entry 1: suffix len 1, IS a sub-block
        let sub_code = (parent_fp - child_fp) as i64;
        suffix_lengths.write_vlong(sub_code); // entry 1's subCode
        tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
        tim.write_bytes(&suffix_lengths);

        let mut parent_stats = Vec::new();
        parent_stats.write_vint(1 << 1); // entry 0 ("aa"): docFreq=1
        tim.write_vint(parent_stats.len() as i32);
        tim.write_bytes(&parent_stats);

        let mut parent_meta = Vec::new();
        parent_meta.write_vlong(5 << 1); // entry 0's docStartFP delta=5, absolute
        parent_meta.write_vint(0); // singleton_doc_id
        tim.write_vint(parent_meta.len() as i32);
        tim.write_bytes(&parent_meta);

        let entries = decode_block(&tim, parent_fp, IndexOptions::Docs, false).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, b"a");
        assert_eq!(entries[0].1.doc_freq, 1);
        assert_eq!(entries[1].0, b"bzz");
        assert_eq!(entries[1].1.doc_freq, 1);
        assert_eq!(entries[1].1.total_term_freq, 1);
    }

    #[test]
    fn decode_block_rejects_sub_block_delta_fp_past_parent() {
        let mut tim = Vec::new();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
        let suffix_bytes = b"x";
        let code_l = (suffix_bytes.len() as u64) << 3; // non-leaf
        tim.write_vlong(code_l as i64);
        tim.write_bytes(suffix_bytes);
        let mut suffix_lengths = Vec::new();
        suffix_lengths.write_vint((1 << 1) | 1); // suffix len 1, is a sub-block
        suffix_lengths.write_vlong(1_000_000); // subCode far exceeding this block's own fp
        tim.write_vint((suffix_lengths.len() as i32) << 1);
        tim.write_bytes(&suffix_lengths);
        tim.write_vint(0); // no stat bytes
        tim.write_vint(0); // no meta bytes

        let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    /// A `vint` byte count for one of a block's uncompressed regions that
    /// exceeds the file must be an error, not an allocation. Java's
    /// `new byte[n]` throws; `vec![0u8; n]` would abort the process, and an
    /// abort cannot be caught at the FFI boundary.
    #[test]
    fn decode_block_rejects_region_lengths_larger_than_the_file() {
        // Base block: one leaf entry, then a stats length that claims far
        // more bytes than the file holds.
        fn block_with_stats_len(stats_len: i32) -> Vec<u8> {
            let mut tim = Vec::new();
            tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
            let suffix_bytes = b"x";
            tim.write_vlong((((suffix_bytes.len() as u64) << 3) | 0x04) as i64); // leaf, no compression
            tim.write_bytes(suffix_bytes);
            let mut suffix_lengths = Vec::new();
            suffix_lengths.write_vint(1);
            tim.write_vint((suffix_lengths.len() as i32) << 1);
            tim.write_bytes(&suffix_lengths);
            tim.write_vint(stats_len);
            tim
        }

        for stats_len in [i32::MAX, -1] {
            let tim = block_with_stats_len(stats_len);
            let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
            assert!(
                matches!(err, Error::Store(lucene_store::Error::Corrupted(_))),
                "stats_len={stats_len} gave {err:?}"
            );
        }
    }

    /// The uncompressed suffix region gets the same treatment, but against
    /// `remaining()` rather than a plain sign check: `numSuffixBytes` is a
    /// `vlong`-derived field with 61 usable bits.
    #[test]
    fn decode_block_rejects_uncompressed_suffix_length_past_the_file() {
        let mut tim = Vec::new();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
        tim.write_vlong(((1u64 << 40) << 3 | 0x04) as i64); // leaf, no compression, absurd length
        let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn decode_block_rejects_illegal_compression_code() {
        // `code_l & 0x03 == 3` never corresponds to a `CompressionAlgorithm`
        // enum constant (only 0/NO_COMPRESSION, 1/LOWERCASE_ASCII,
        // 2/LZ4 are assigned) -- real Lucene's `CompressionAlgorithm.byCode`
        // throws for it too.
        let mut tim = Vec::new();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
        tim.write_vlong(0x04 | 0x03); // isLeafBlock, illegal compressionAlg=3
        let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn load_node_multi_children_with_output_and_floor() {
        // SIGN_MULTI_CHILDREN with its own output+floor data (the
        // `hasOutput`/`NON_LEAF_NODE_HAS_FLOOR` branch of
        // `loadMultiChildrenNode`, not yet exercised by the no-output
        // multi-children tests above).
        let mut slice = vec![0u8; 48];
        let parent_fp = 16usize;
        let min_label = b'a';
        let strategy_bytes_region = *b"b"; // ARRAY: one extra label 'b'.
        let strategy_bytes = strategy_bytes_region.len();
        let encoded_bytes_minus1 = 0u32; // 1-byte encoded output fp.
                                         // childrenDeltaFpBytesMinus1 = 0 (1 byte)
        let term: u32 = SIGN_MULTI_CHILDREN
            | (1 << 5) // has output
            | (encoded_bytes_minus1 << 6)
            | (CHILD_STRATEGY_ARRAY << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());

        // encodeFP: (floor?1:0) | (hasTerms?2:0) | (fp << 2); output fp = 9.
        let encoded_fp: u64 = NON_LEAF_NODE_HAS_FLOOR | NON_LEAF_NODE_HAS_TERMS | (9 << 2);
        assert!(encoded_fp <= 0xFF);
        // The 3-byte header only fills the low 24 bits of `term`, so byte
        // offset +3 (the word's 4th byte) is already the start of the
        // encoded-output-fp region, not part of the header -- matches
        // `loadMultiChildrenNode`'s `termLong >>> 24` inline read.
        let encoded_fp_off = parent_fp + 3;
        slice[encoded_fp_off] = encoded_fp as u8;

        // "has floor" branch: one byte childrenNum-1, then strategy bytes,
        // then children fps, then floor data.
        let children_num_off = encoded_fp_off + 1;
        slice[children_num_off] = 1; // childrenNum - 1 = 1 -> 2 children
        let strategy_fp = children_num_off + 1;
        slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(&strategy_bytes_region);
        let fps_fp = strategy_fp + strategy_bytes;
        // Two children, both leaf nodes at fp 0 and fp 2.
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 30;
        slice[2] = LEAF_NODE_HAS_TERMS as u8;
        slice[3] = 40;
        slice[fps_fp] = parent_fp as u8; // delta to child A (fp 0)
        slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B (fp 2)
        let floor_fp = fps_fp + 2;
        slice[floor_fp] = 1; // numFollowFloorBlocks
        slice[floor_fp + 1] = b'z';
        slice[floor_fp + 2] = (3 << 1) | 1; // code

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.output_fp, Some(9));
        assert!(node.has_terms);
        assert_eq!(node.floor_data_fp, Some(floor_fp));
        assert_eq!(node.strategy_fp, strategy_fp);

        let mut out = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap();
        let mut fps: Vec<u64> = out.iter().map(|(fp, _)| *fp).collect();
        fps.sort_unstable();
        // Own output expands to blocks at fp 9 and fp 9+3=12, plus children
        // at fp 30 and fp 40.
        assert_eq!(fps, vec![9, 12, 30, 40]);
    }

    #[test]
    fn collect_leaf_blocks_rejects_trie_nesting_too_deep() {
        let mut slice = vec![0u8; 16];
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 5;
        let node = load_node(&slice, 0).unwrap();
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        let err = collect_leaf_blocks(&slice, &node, 10_001, &mut prefix, &mut out).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn collect_leaf_blocks_rejects_single_child_delta_exceeding_parent_fp() {
        let mut slice = vec![0u8; 16];
        let term: u32 = SIGN_SINGLE_CHILD_WITHOUT_OUTPUT;
        slice[0..4].copy_from_slice(&term.to_le_bytes());
        slice[1] = b'x';
        slice[2] = 100; // child delta fp (100) > parent fp (0)
        let node = load_node(&slice, 0).unwrap();
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        let err = collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn multi_children_fps_rejects_delta_exceeding_parent_fp() {
        let mut slice = vec![0u8; 24];
        let parent_fp = 8usize;
        let term: u32 = SIGN_MULTI_CHILDREN | (CHILD_STRATEGY_ARRAY << 9);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        let strategy_fp = parent_fp + 3;
        slice[strategy_fp] = b'b'; // ARRAY strategy, one extra label
        slice[strategy_fp + 1] = 100; // delta (100) > parent fp (8)
        let node = load_node(&slice, parent_fp).unwrap();
        // Label 0 is the node's own `minChildrenLabel`, i.e. child position
        // 0, whose delta is the corrupt one.
        let err = lookup_child(&slice, &node, 0).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn decode_block_singleton_run_length_and_all_equal_suffixes() {
        // Hand-build a block with allEqual suffix lengths and a singleton
        // run (three consecutive docFreq=1/totalTermFreq=1 terms encoded via
        // the run-length token) to exercise both branches `Builder` (which
        // always emits per-entry non-run tokens and variable suffix
        // lengths) never reaches.
        let mut tim = Vec::new();
        let terms = ["aa", "bb", "cc"];
        let ent_count = terms.len() as u32;
        tim.write_vint(((ent_count << 1) | 1) as i32); // isLastInFloor

        let suffix_bytes: Vec<u8> = terms.iter().flat_map(|t| t.bytes()).collect();
        let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // leaf, no compression
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&suffix_bytes);

        // allEqual suffix lengths: all terms are 2 bytes. The logical array
        // size is still entCount (one vint-encoded length per entry) even
        // though only a single physical byte is written on disk.
        tim.write_vint(((ent_count as i32) << 1) | 1);
        tim.write_byte(2);

        // stats: one run-length token covering all three (docFreq=1 each).
        let mut stats = Vec::new();
        stats.write_vint((3 << 1) | 1); // token&1==1 -> singleton run of length 3
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        // Postings metadata: three singleton entries, each via the bit=0
        // (docStartFP-delta) branch of `decode_term_metadata` -- legal
        // whether or not `absolute` is set, unlike the zigzag-delta branch.
        let mut meta = Vec::new();
        for singleton_doc_id in [0i32, 1, 2] {
            meta.write_vlong(0); // docStartFP delta = 0
            meta.write_vint(singleton_doc_id);
        }
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);

        let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
        assert_eq!(entries.len(), 3);
        for (term, stats, _meta) in &entries {
            assert_eq!(term.len(), 2);
            assert_eq!(stats.doc_freq, 1);
            assert_eq!(stats.total_term_freq, 1);
        }
        assert_eq!(entries[0].0, b"aa");
        assert_eq!(entries[2].0, b"cc");
    }

    #[test]
    fn decode_block_lz4_compressed_suffixes() {
        // Hand-built block using `code_l & 0x03 == 2` (LZ4) with the suffix
        // bytes actually run through this port's own `crate::lz4::compress`
        // (a real, general-purpose LZ4 compressor, not a fake/no-op one --
        // see `lz4.rs`'s module doc), then decoded back via `decode_block`'s
        // new LZ4 dispatch arm. This is a hand-built *test vector* (compress
        // + decompress round-trip through this port's own LZ4, cross-checked
        // separately against real Lucene bytes by
        // `tests/blocktree_compressed_fixture.rs`, which decodes an actual
        // `Lucene103BlockTreeTermsWriter`-produced LZ4 block).
        let mut tim = Vec::new();
        let terms = ["aaaaaaaa", "aaaaaaab", "aaaaaaac", "aaaaaaad"];
        let ent_count = terms.len() as u32;
        tim.write_vint(((ent_count << 1) | 1) as i32);

        let suffix_bytes: Vec<u8> = terms.iter().flat_map(|t| t.bytes()).collect();
        let compressed = crate::lz4::compress(&suffix_bytes);
        // Sanity: this input is repetitive enough that LZ4 actually shrinks
        // it -- otherwise this test wouldn't be exercising anything real.
        assert!(compressed.len() < suffix_bytes.len());

        let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04 | 0x02; // leaf, LZ4
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&compressed);

        tim.write_vint(((ent_count as i32) << 1) | 1); // allEqual suffix lengths
        tim.write_byte(8);

        let mut stats = Vec::new();
        stats.write_vint((ent_count << 1 | 1) as i32); // singleton run of length 4
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        let mut meta = Vec::new();
        for singleton_doc_id in 0..ent_count as i32 {
            meta.write_vlong(0);
            meta.write_vint(singleton_doc_id);
        }
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);

        let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
        assert_eq!(entries.len(), 4);
        for (i, (term, stats, _meta)) in entries.iter().enumerate() {
            assert_eq!(term, terms[i].as_bytes());
            assert_eq!(stats.doc_freq, 1);
            assert_eq!(stats.total_term_freq, 1);
        }
    }

    #[test]
    fn decompress_lowercase_ascii_matches_real_lucene_compress_output() {
        // Real Lucene bytes: generated by directly invoking
        // `org.apache.lucene.util.compress.LowercaseAsciiCompression.compress`
        // (from the pinned lucene-core-10.5.0.jar) on the ASCII string below,
        // which mixes lowercase letters, digits, `.`/`-`/`_` (all
        // compressible) with two exceptions (`Z`, `!`, both outside the
        // compressible ranges) to exercise the exception-list decode branch
        // too. Not embedded in an actual on-disk `.tim` block -- see
        // `tests/blocktree_compressed_fixture.rs`'s module doc for why
        // forcing a real `IndexWriter` to choose `LOWERCASE_ASCII` (as
        // opposed to `LZ4` or `NO_COMPRESSION`) for this port's own fixtures
        // wasn't achieved in reasonable effort, and why this vector is the
        // honest fallback for that one mode.
        let original = b"the-quick_brown.fox.jumps_over-42.lazy_dogs.1234567890Z!abcdefghij";
        let compressed_hex = "7569664ef236aaa4aca0a3b3b0b8af8fa7b0b90fab362e3174607077a6b38e95134fad62fbbaa0e53068b4cf125394d5161701365a";
        let compressed: Vec<u8> = (0..compressed_hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&compressed_hex[i..i + 2], 16).unwrap())
            .collect();

        let mut r = SliceInput::new(&compressed);
        let mut out = vec![0u8; original.len()];
        decompress_lowercase_ascii(&mut r, &mut out).unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn decompress_lowercase_ascii_rejects_out_of_range_exception_index() {
        // Hand-built, not real-Lucene-generated: 4-byte output (saved=1,
        // compressed_len=3), 3 arbitrary packed bytes, then a single
        // exception whose delta (10) pushes the cumulative index to 10,
        // past `out.len()` (4) -- must error before even reading the
        // exception's replacement value byte.
        let compressed: Vec<u8> = vec![0x61, 0x62, 0x63, 0x01, 0x0A];
        let mut r = SliceInput::new(&compressed);
        let mut out = vec![0u8; 4];
        let err = decompress_lowercase_ascii(&mut r, &mut out).unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn decode_block_lowercase_ascii_compressed_suffixes() {
        // Same real-Lucene-generated compressed bytes as the standalone
        // decompress test above, this time threaded through the full
        // `decode_block` dispatch (`code_l & 0x03 == 1`) with a single
        // whole-block suffix rather than per-term suffix lengths (the term
        // boundary doesn't line up with the compression -- LowercaseAscii
        // compresses the concatenated suffix blob as one unit, same as
        // LZ4 -- so this test uses one giant "term" spanning the whole
        // decompressed suffix, which is enough to prove the dispatch wires
        // the compression-alg byte, the decompressed length, and the
        // decoded bytes together correctly).
        let mut tim = Vec::new();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor

        let original = b"the-quick_brown.fox.jumps_over-42.lazy_dogs.1234567890Z!abcdefghij";
        let compressed_hex = "7569664ef236aaa4aca0a3b3b0b8af8fa7b0b90fab362e3174607077a6b38e95134fad62fbbaa0e53068b4cf125394d5161701365a";
        let compressed: Vec<u8> = (0..compressed_hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&compressed_hex[i..i + 2], 16).unwrap())
            .collect();

        let code_l = ((original.len() as u64) << 3) | 0x04 | 0x01; // leaf, LOWERCASE_ASCII
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&compressed);

        tim.write_vint((1 << 1) | 1); // allEqual, single entry -> irrelevant, but still 1 length byte
        tim.write_byte(original.len() as u8);

        let mut stats = Vec::new();
        stats.write_vint(1 << 1 | 1); // singleton run of length 1
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        let mut meta = Vec::new();
        meta.write_vlong(0);
        meta.write_vint(0);
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);

        let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, original);
        assert_eq!(entries[0].1.doc_freq, 1);
        assert_eq!(entries[0].1.total_term_freq, 1);
    }

    /// `.tmd` with caller-chosen per-field statistics, so each of `open`'s
    /// record-level validations can be driven on its own. The `.tim`/`.tip`
    /// come from [`Builder::build`] unchanged -- only the metadata varies.
    #[allow(clippy::too_many_arguments)]
    fn tmd_with(
        id: &[u8; ID_LENGTH],
        suffix: &str,
        num_terms: i64,
        sum_total_term_freq: i64,
        sum_doc_freq: i64,
        doc_count: i32,
        index_length: i64,
        terms_length: i64,
    ) -> Vec<u8> {
        let index_start = codec_util::index_header_length(TERMS_INDEX_CODEC_NAME, suffix);
        // `Builder::build`'s root node: header byte + an 8-byte output fp +
        // 8 bytes of over-read pad.
        let index_end = index_start + 17;

        let mut tmd = Vec::new();
        codec_util::write_index_header(
            &mut tmd,
            TERMS_META_CODEC_NAME,
            VERSION_CURRENT,
            id,
            suffix,
        );
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, id, suffix);
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1); // numFields
        tmd.write_vint(0); // field number
        tmd.write_vlong(num_terms);
        tmd.write_vlong(sum_total_term_freq);
        tmd.write_vlong(sum_doc_freq);
        tmd.write_vint(doc_count);
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vlong(index_start as i64);
        tmd.write_vlong(0); // root fp
        tmd.write_vlong(index_end as i64);
        tmd.write_i64(index_length);
        tmd.write_i64(terms_length);
        codec_util::write_footer(&mut tmd);
        tmd
    }

    /// The four `.tmd` record checks `Lucene103BlockTreeTermsReader`'s
    /// constructor makes before it will trust a field: `numTerms > 0`,
    /// `sumDocFreq >= docCount`, `sumTotalTermFreq >= sumDocFreq`, and
    /// non-negative recorded `.tip`/`.tim` lengths. These are the only
    /// validations left at open now that no block is read there, so each one
    /// is worth its own case.
    #[test]
    fn tmd_record_level_validations_rejected() {
        let b = Builder::new();
        let (tim, tip, _) = b.build(IndexOptions::DocsAndFreqs, &[("a", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let ok_lengths = (tip.len() as i64, tim.len() as i64);
        let open_with = |tmd: Vec<u8>| open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5);

        // Sanity: the same helper with legal values opens cleanly, so each
        // rejection below is attributable to the one value it changed.
        assert!(open_with(tmd_with(
            &b.id,
            &b.suffix,
            1,
            1,
            1,
            1,
            ok_lengths.0,
            ok_lengths.1
        ))
        .is_ok());

        let err = open_with(tmd_with(
            &b.id,
            &b.suffix,
            0,
            1,
            1,
            1,
            ok_lengths.0,
            ok_lengths.1,
        ))
        .unwrap_err();
        assert!(matches!(err, Error::IllegalNumTerms(0)), "{err:?}");

        // sumDocFreq (1) < docCount (2).
        let err = open_with(tmd_with(
            &b.id,
            &b.suffix,
            1,
            5,
            1,
            2,
            ok_lengths.0,
            ok_lengths.1,
        ))
        .unwrap_err();
        assert!(matches!(err, Error::InvalidSumDocFreq { .. }), "{err:?}");

        // sumTotalTermFreq (1) < sumDocFreq (3).
        let err = open_with(tmd_with(
            &b.id,
            &b.suffix,
            1,
            1,
            3,
            1,
            ok_lengths.0,
            ok_lengths.1,
        ))
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidSumTotalTermFreq { .. }),
            "{err:?}"
        );

        for (index_length, terms_length) in [(-1, ok_lengths.1), (ok_lengths.0, -1)] {
            let err = open_with(tmd_with(
                &b.id,
                &b.suffix,
                1,
                1,
                1,
                1,
                index_length,
                terms_length,
            ))
            .unwrap_err();
            assert!(matches!(err, Error::Store(_)), "{err:?}");
        }
    }

    /// `TermMatcher`'s default `dead_prefix_len` -- the "this matcher can
    /// never prove a prefix dead" answer that `CAN_SKIP == false` folds the
    /// call site away for. Instantiated here so the default body is still
    /// compiled and checked even though no production matcher reaches it.
    #[test]
    fn a_matcher_that_cannot_skip_reports_no_dead_prefix() {
        struct NeverSkips;
        impl TermMatcher for NeverSkips {
            fn matches(&mut self, term: &[u8]) -> bool {
                term == b"yes"
            }
        }
        const { assert!(!NeverSkips::CAN_SKIP) };
        assert!(NeverSkips.matches(b"yes"));
        assert_eq!(NeverSkips.dead_prefix_len(b"anything"), None);
    }

    /// Every postings-facing entry point must report a term that is not in
    /// the dictionary as "nothing here" rather than erroring or reaching for
    /// a `.doc`/`.pos` file -- the miss path each of them takes off the same
    /// lazy `seekExact`.
    #[test]
    fn postings_entry_points_report_a_missing_term_as_none() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, &[("alpha", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        // A `.pos` file with nothing but its framing: none of these calls
        // gets far enough to read a byte of it.
        let mut pos_bytes = Vec::new();
        codec_util::write_index_header(
            &mut pos_bytes,
            postings::POS_CODEC,
            postings::VERSION_CURRENT,
            &b.id,
            &b.suffix,
        );
        codec_util::write_footer(&mut pos_bytes);
        let pos_in = postings::PosInput::open(&pos_bytes, &b.id, &b.suffix).unwrap();

        assert!(field.postings(b"beta", None).unwrap().is_none());
        assert!(field
            .positions(b"beta", None, &pos_in, None)
            .unwrap()
            .is_none());
        assert!(field
            .positions_flat(b"beta", None, &pos_in, None)
            .unwrap()
            .is_none());
        let (positions, starts) = field
            .positions_for_docs(b"beta", None, &pos_in, None, &[], 0, &[])
            .unwrap();
        assert!(positions.is_empty());
        assert_eq!(starts, vec![0]);
    }

    /// The trie region's two raw readers must reject a read that runs off the
    /// end of the field's own `[indexStart, indexEnd)` slice rather than
    /// panicking on the slice index -- a corrupt `.tmd` can point `rootFP`
    /// anywhere inside it.
    #[test]
    fn trie_reads_past_the_index_slice_are_errors() {
        let slice = [0u8; 4];
        assert!(matches!(read_u64_at(&slice, 0), Err(Error::Store(_))));
        assert!(matches!(read_u8_at(&slice, 9), Err(Error::Store(_))));
        assert!(matches!(
            read_u64_n_bytes(&slice, 2, 8),
            Err(Error::Store(_))
        ));
        // In range: the low bytes of a little-endian run.
        assert_eq!(read_u64_n_bytes(&[1, 2, 3, 4], 1, 2).unwrap(), 0x0302);
    }

    /// A dead-prefix skip that jumps past the *last* term of the field must
    /// end the walk, not fall off the end of the frame stack -- the
    /// `SeekStatus::End` arm of `Intersect`'s skip.
    #[test]
    fn regexp_intersect_skipping_past_the_last_term_ends_the_walk() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("za", 1, 1), ("zb", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        // `z[0-9]` shares the literal prefix `z` with both terms and matches
        // neither, and each of them is a dead prefix (nothing extending "za"
        // or "zb" can match a two-byte pattern), so the walk skips past "zb"
        // and off the end of the dictionary.
        let pattern = RegexpPattern::new(b"z[0-9]").unwrap();
        assert_eq!(field.regexp_intersect(&pattern).count(), 0);

        // The same pattern with a match in range still finds it.
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("z1", 1, 1), ("za", 1, 1)]);
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();
        let got: Vec<Vec<u8>> = field
            .regexp_intersect(&pattern)
            .map(|r| r.unwrap().0)
            .collect();
        assert_eq!(got, vec![b"z1".to_vec()]);
    }

    /// `TrieReader.load`'s two out-of-word fallbacks: when a node's packed fp
    /// needs more bytes than fit alongside the header in the first 8-byte
    /// word, Java re-reads a whole `long` at `fp + 2` (single child) or
    /// `fp + 3` (multi children) instead of shifting the word it already has.
    /// Neither is reachable from any fixture, because both need a `.tim`
    /// large enough to push a block fp past ~2^40.
    #[test]
    fn load_node_reads_a_second_word_for_wide_file_pointers() {
        // Single child, 7-byte child delta fp -> `read_u64_at(fp + 2)`.
        let mut slice = vec![0u8; 32];
        slice[0] = LEAF_NODE_HAS_TERMS as u8; // leaf child at fp 0, output fp 42
        slice[1] = 42;
        let parent_fp = 16usize;
        let child_delta_bytes_minus1 = 6u32;
        slice[parent_fp] =
            (SIGN_SINGLE_CHILD_WITHOUT_OUTPUT | (child_delta_bytes_minus1 << 2)) as u8;
        slice[parent_fp + 1] = b'q';
        slice[parent_fp + 2] = parent_fp as u8; // delta 16 -> child at fp 0
        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.sign, SIGN_SINGLE_CHILD_WITHOUT_OUTPUT);
        assert_eq!(node.child_delta_fp, parent_fp as u64);
        let child = lookup_child(&slice, &node, b'q').unwrap().expect("child q");
        assert_eq!(child.output_fp, Some(42));
        assert!(lookup_child(&slice, &node, b'r').unwrap().is_none());

        // Multi children with an output, 6-byte encoded output fp ->
        // `read_u64_at(fp + 3)`.
        let mut slice = vec![0u8; 48];
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 30;
        slice[2] = LEAF_NODE_HAS_TERMS as u8;
        slice[3] = 40;
        let parent_fp = 16usize;
        let encoded_bytes_minus1 = 5u32;
        let min_label = b'a';
        let strategy_bytes = 1usize; // ARRAY: one extra label
        let term: u32 = SIGN_MULTI_CHILDREN
            | (1 << 5)
            | (encoded_bytes_minus1 << 6)
            | (CHILD_STRATEGY_ARRAY << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        // encodeFP with hasTerms, no floor, output fp 9; written at fp + 3 so
        // the 6-byte read starting there sees exactly it.
        slice[parent_fp + 3] = ((9u64 << 2) | NON_LEAF_NODE_HAS_TERMS) as u8;
        let strategy_fp = parent_fp + 4 + encoded_bytes_minus1 as usize;
        slice[strategy_fp] = b'b';
        slice[strategy_fp + 1] = parent_fp as u8; // 'a' -> fp 0
        slice[strategy_fp + 2] = (parent_fp - 2) as u8; // 'b' -> fp 2

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.output_fp, Some(9));
        assert!(node.has_terms);
        assert_eq!(node.floor_data_fp, None);
        assert_eq!(node.strategy_fp, strategy_fp);
        for (label, output) in [(b'a', 30u64), (b'b', 40)] {
            let child = lookup_child(&slice, &node, label)
                .unwrap()
                .unwrap_or_else(|| panic!("label {} should resolve", label as char));
            assert_eq!(child.output_fp, Some(output));
        }
    }

    #[test]
    fn invalid_field_number_rejected() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        // FieldInfos has no field numbered 0.
        let fis = FieldInfos {
            fields: vec![field_info(9, "other", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap_err();
        assert!(matches!(err, Error::InvalidFieldNumber(0)));
    }

    #[test]
    fn invalid_doc_count_rejected() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        // docCount (1, baked into Builder::build) exceeds maxDoc=0.
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidDocCount { .. }));
    }

    #[test]
    fn duplicate_field_rejected() {
        let id = [3u8; ID_LENGTH];
        let mut tmd = Vec::new();
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(2); // numFields

        // Build a single shared .tim block (one term "a") and .tip root node
        // that both field records point at, so the same field *name* is
        // reachable twice (two field numbers mapping to fields named "f").
        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        let block_fp = tim.len();
        tim.write_vint((1 << 1) | 1);
        tim.write_vlong(((1u64 << 3) | 0x04) as i64);
        tim.write_bytes(b"a");
        tim.write_vint(1 << 1);
        tim.write_bytes(&[1]);
        let mut stats = Vec::new();
        stats.write_vint(1 << 1); // docFreq=1, non-singleton token
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);
        let mut meta = Vec::new();
        meta.write_vlong(0); // docStartFP delta = 0
        meta.write_vint(0); // singletonDocID (docFreq == 1)
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);
        codec_util::write_footer(&mut tim);

        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        let index_start = tip.len();
        let header = LEAF_NODE_HAS_TERMS as u8;
        tip.push(header);
        tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
        tip.extend_from_slice(&0u64.to_le_bytes());
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        for field_number in [0i32, 1i32] {
            tmd.write_vint(field_number);
            tmd.write_vlong(1); // numTerms
            tmd.write_vlong(1); // sumDocFreq (Docs aliasing)
            tmd.write_vint(1); // docCount
            tmd.write_vint(1);
            tmd.write_bytes(b"a");
            tmd.write_vint(1);
            tmd.write_bytes(b"a");
            tmd.write_vlong(index_start as i64);
            tmd.write_vlong(0);
            tmd.write_vlong(index_end as i64);
        }
        tmd.write_i64(tip.len() as i64); // indexLength: the whole file, footer included
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![
                field_info(0, "f", IndexOptions::Docs),
                field_info(1, "f", IndexOptions::Docs),
            ],
        };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::DuplicateField(_)));
    }

    #[test]
    fn index_region_out_of_bounds_rejected() {
        let b = Builder::new();
        let (tim, tip, _tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        let id = b.id;
        let suffix = b.suffix.clone();

        // Hand-build a .tmd whose indexEnd points past the end of .tip.
        let mut tmd = Vec::new();
        codec_util::write_index_header(
            &mut tmd,
            TERMS_META_CODEC_NAME,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        codec_util::write_index_header(
            &mut tmd,
            POSTINGS_TERMS_CODEC,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1);
        tmd.write_vint(0);
        tmd.write_vlong(1);
        tmd.write_vlong(1);
        tmd.write_vint(1);
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vlong(0);
        tmd.write_vlong(0);
        tmd.write_vlong((tip.len() + 100) as i64); // out of bounds indexEnd
        tmd.write_i64(tip.len() as i64);
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &id, &suffix, 5).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    /// Structural proof (not just "lookups still work") that
    /// `fixtures/data/blocktree_multilevel_index/` -- 8000 pseudo-random
    /// terms, regenerated via `fixtures/src/GenBlockTreeMultilevel.java` --
    /// actually forces real Lucene to write a genuine **non-leaf** `.tim`
    /// block (some of its entries are in-block pointers to further-nested
    /// sub-blocks, not raw term suffixes) reachable from this field's `.tip`
    /// trie, i.e. the "root block -> internal block -> leaf block" case this
    /// module's `decode_block`/`decode_block_at_depth` now decode. Walks the
    /// same trie [`collect_leaf_blocks`] would, independently re-deriving
    /// which physical `.tim` blocks are leaf vs. non-leaf by peeking each
    /// one's own `isLeafBlock` bit -- this test would fail (assert
    /// `saw_non_leaf_block`) if a future regen of this fixture, or a change
    /// to real Lucene's own writer heuristics, stopped producing one, which
    /// is exactly the failure mode a purely behavioral "every term still
    /// findable" test could miss (it'd stay green even if this fixture
    /// degenerated to an all-leaf-blocks shape). The full differential
    /// (every term findable via the public API, matching real Lucene's own
    /// ground truth) lives in `crates/lucene-codecs/tests/blocktree_multilevel_fixture.rs`,
    /// same split as every other real-bytes fixture test in this crate:
    /// external test = public-API differential, in-crate test = structural
    /// invariant only reachable with this module's private internals.
    #[test]
    fn multilevel_fixture_reaches_a_genuine_non_leaf_block() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_multilevel_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTreeMultilevel)");
        let kv: std::collections::HashMap<String, String> = manifest
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let read_raw = |name: &str| {
            std::fs::read(format!("{dir}{name}.raw"))
                .unwrap_or_else(|_| panic!("missing {name}.raw"))
        };

        let tmd = read_raw(kv.get("tmd_file_name").unwrap());
        let tip = read_raw(kv.get("tip_file_name").unwrap());
        let tim = read_raw(kv.get("tim_file_name").unwrap());
        let fnm = read_raw(kv.get("fnm_file_name").unwrap());
        let id_hex = kv.get("id_hex").unwrap();
        let mut id = [0u8; ID_LENGTH];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = kv.get("segment_suffix").unwrap();
        let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

        // Re-derive "many"'s index_start/root_fp/index_end the same way
        // `open()` does, so this test doesn't need any new `pub` surface on
        // this module just to expose them.
        let mut tmd_input = SliceInput::new(&tmd);
        codec_util::check_index_header(
            &mut tmd_input,
            TERMS_META_CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        codec_util::check_index_header(
            &mut tmd_input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        let _index_block_size = tmd_input.read_vint().unwrap();
        let num_fields = tmd_input.read_vint().unwrap();
        let mut field_index = None;
        for _ in 0..num_fields {
            let field_number = tmd_input.read_vint().unwrap();
            let _num_terms = tmd_input.read_vlong().unwrap();
            let fi = field_infos.field_by_number(field_number).unwrap();
            read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
            let _doc_count = tmd_input.read_vint().unwrap();
            let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
            let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
            let index_start = tmd_input.read_vlong().unwrap() as usize;
            let root_fp = tmd_input.read_vlong().unwrap() as usize;
            let index_end = tmd_input.read_vlong().unwrap() as usize;
            if fi.name == "many" {
                field_index = Some((index_start, root_fp, index_end));
            }
        }
        let (index_start, root_fp, index_end) = field_index.expect("field \"many\" in .tmd");

        let index_slice = &tip[index_start..index_end];
        let root = load_node(index_slice, root_fp).unwrap();
        // This field's 8000 pseudo-random lowercase terms cover all 26
        // letters at depth 0, so the root's `(minLabel, maxLabel, labelCnt)`
        // is `('a', 'z', 26)` -- a fully dense range, for which
        // `TrieBuilder.ChildSaveStrategy.choose`'s cost formula picks
        // `REVERSE_ARRAY` (`needBytes` = `26 - 26 + 1` = 1, beating both
        // `ARRAY` = 25 and `BITS` = `ceil(26/8)` = 4). This is this fixture's
        // real-Lucene-forced multi-children strategy; `ARRAY` and `BITS` are
        // covered by the dedicated `blocktree_child_strategies_index`
        // fixture instead (see `child_strategies_fixture_forces_array_and_bits_strategies`).
        assert_eq!(root.child_save_strategy, CHILD_STRATEGY_REVERSE_ARRAY);
        assert_eq!(root.strategy_bytes, 1);
        assert_eq!(root.min_children_label, b'a');
        let mut blocks = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(index_slice, &root, 0, &mut prefix, &mut blocks).unwrap();
        assert!(
            blocks.len() > 1,
            "expected the trie to reach more than one physical block"
        );

        // Peek each reached block's own isLeafBlock bit directly (the same
        // two reads `decode_block_at_depth` starts with) without doing a
        // full decode -- purely structural.
        let mut saw_non_leaf_block = false;
        for (block_fp, _prefix) in &blocks {
            let mut r = SliceInput::new(&tim);
            r.seek(*block_fp as usize).unwrap();
            let _code = r.read_vint().unwrap();
            let code_l = r.read_vlong().unwrap() as u64;
            if (code_l & 0x04) == 0 {
                saw_non_leaf_block = true;
            }
        }
        assert!(
            saw_non_leaf_block,
            "expected at least one physical .tim block reachable from the \"many\" \
             field's trie to be non-leaf (isLeafBlock == false) -- this fixture is \
             supposed to force real Lucene into a genuine multi-level blocktree \
             (root block -> internal block -> leaf block), not just multiple \
             sibling leaf blocks/floor blocks under one trie node"
        );

        // And the full round trip through the *unmodified* public API must
        // still recover every term correctly despite that non-leaf block
        // (this is the behavioral half; the fuller differential -- matching
        // real Lucene's own sorted term list -- lives in
        // `tests/blocktree_multilevel_fixture.rs`).
        let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
        let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
        let field = fields.field("many").unwrap();
        let num_terms: i64 = kv.get("field.many.numTerms").unwrap().parse().unwrap();
        assert_eq!(field.num_terms, num_terms);
        let mut it = field.iter();
        while let Some((term, stats)) = it.next() {
            assert_eq!(field.seek_exact(term).unwrap(), stats);
        }
    }

    /// Real-Lucene-fixture differential test proving `ChildSaveStrategy::ARRAY`
    /// and `ChildSaveStrategy::BITS` (the two of the three real
    /// `TrieBuilder.ChildSaveStrategy` label-encodings that
    /// `multilevel_fixture_reaches_a_genuine_non_leaf_block`'s "many" field
    /// does *not* land on -- that root happens to pick `REVERSE_ARRAY`, see
    /// that test) are each forced onto a real `.tip` trie root node and
    /// decode correctly.
    ///
    /// `fixtures/src/GenBlockTreeChildStrategies.java` builds two fields
    /// whose terms' leading bytes were hand-picked so
    /// `TrieBuilder.ChildSaveStrategy.choose`'s own `needBytes` cost formula
    /// -- BITS: `ceil((maxLabel-minLabel+1)/8)`, ARRAY: `labelCnt-1`,
    /// REVERSE_ARRAY: `(maxLabel-minLabel+1)-labelCnt+1` -- picks a distinct
    /// winner for each field (see that file's module doc for the exact
    /// arithmetic): "arraystrat" (5 labels spanning printable-ASCII, distance
    /// 94: BITS=12, ARRAY=4, REVERSE_ARRAY=90 -> ARRAY wins) and "bitsstrat"
    /// (9 labels spaced 5 apart, distance 41: BITS=6, ARRAY=8,
    /// REVERSE_ARRAY=33 -> BITS wins). This test decodes each field's root
    /// trie node and asserts the exact `child_save_strategy` code real
    /// Lucene's writer chose, then round-trips every term through the
    /// public `open`/`seek_exact` API to prove the decode is not just
    /// structurally plausible but actually correct.
    #[test]
    fn child_strategies_fixture_forces_array_and_bits_strategies() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_child_strategies_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTreeChildStrategies)");
        let kv: std::collections::HashMap<String, String> = manifest
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let read_raw = |name: &str| {
            std::fs::read(format!("{dir}{name}.raw"))
                .unwrap_or_else(|_| panic!("missing {name}.raw"))
        };
        let tmd = read_raw(kv.get("tmd_file_name").unwrap());
        let tip = read_raw(kv.get("tip_file_name").unwrap());
        let tim = read_raw(kv.get("tim_file_name").unwrap());
        let fnm = read_raw(kv.get("fnm_file_name").unwrap());
        let id_hex = kv.get("id_hex").unwrap();
        let mut id = [0u8; ID_LENGTH];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = kv.get("segment_suffix").unwrap();
        let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

        let mut tmd_input = SliceInput::new(&tmd);
        codec_util::check_index_header(
            &mut tmd_input,
            TERMS_META_CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        codec_util::check_index_header(
            &mut tmd_input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        let _index_block_size = tmd_input.read_vint().unwrap();
        let num_fields = tmd_input.read_vint().unwrap();
        let mut field_index: std::collections::HashMap<String, (usize, usize, usize)> =
            std::collections::HashMap::new();
        for _ in 0..num_fields {
            let field_number = tmd_input.read_vint().unwrap();
            let _num_terms = tmd_input.read_vlong().unwrap();
            let fi = field_infos.field_by_number(field_number).unwrap();
            read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
            let _doc_count = tmd_input.read_vint().unwrap();
            let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
            let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
            let index_start = tmd_input.read_vlong().unwrap() as usize;
            let root_fp = tmd_input.read_vlong().unwrap() as usize;
            let index_end = tmd_input.read_vlong().unwrap() as usize;
            field_index.insert(fi.name.clone(), (index_start, root_fp, index_end));
        }

        // "arraystrat": 5 labels, distance 94 (0x21..=0x7e) -> ARRAY (code 1).
        let (index_start, root_fp, index_end) = field_index["arraystrat"];
        let root = load_node(&tip[index_start..index_end], root_fp).unwrap();
        assert_eq!(
            root.child_save_strategy, CHILD_STRATEGY_ARRAY,
            "expected real Lucene to pick ChildSaveStrategy.ARRAY for \"arraystrat\"'s \
             root node (needBytes: BITS=12, ARRAY=4, REVERSE_ARRAY=90)"
        );
        assert_eq!(root.strategy_bytes, 4); // labelCnt - 1 = 5 - 1
        assert_eq!(root.min_children_label, 0x21);

        // "bitsstrat": 9 labels, distance 41 (0x21..=0x49) -> BITS (code 2).
        let (index_start, root_fp, index_end) = field_index["bitsstrat"];
        let root = load_node(&tip[index_start..index_end], root_fp).unwrap();
        assert_eq!(
            root.child_save_strategy, CHILD_STRATEGY_BITS,
            "expected real Lucene to pick ChildSaveStrategy.BITS for \"bitsstrat\"'s \
             root node (needBytes: BITS=6, ARRAY=8, REVERSE_ARRAY=33)"
        );
        assert_eq!(root.strategy_bytes, 6); // ceil(41 / 8)
        assert_eq!(root.min_children_label, 0x21);

        // Full round trip through the unmodified public API: every term in
        // both fields must be findable via seek_exact, proving the ARRAY and
        // BITS label decodes (not just the strategy *code*) are correct.
        let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
        let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
        for name in ["arraystrat", "bitsstrat"] {
            let field = fields.field(name).unwrap();
            let expected_count: i64 = kv
                .get(&format!("field.{name}.count"))
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(field.num_terms, expected_count);
            let mut counted = 0i64;
            let mut count_it = field.iter();
            while count_it.next().is_some() {
                counted += 1;
            }
            assert_eq!(counted, expected_count);
            let terms_tsv = std::fs::read_to_string(format!("{dir}{name}.terms.tsv")).unwrap();
            let expected_terms: Vec<&str> = terms_tsv.lines().collect();
            assert_eq!(expected_terms.len() as i64, expected_count);
            for term in &expected_terms {
                let stats = field
                    .seek_exact(term.as_bytes())
                    .unwrap_or_else(|| panic!("term {term:?} not found in field {name}"));
                assert_eq!(stats.doc_freq, 1);
                assert_eq!(stats.total_term_freq, 1);
            }
            let mut it = field.iter();
            while let Some((term, stats)) = it.next() {
                assert_eq!(field.seek_exact(term).unwrap(), stats);
            }
        }
    }

    /// Real-Lucene-fixture differential test for the "root block with no
    /// terms of its own (all sub-blocks)" shape: a field whose first terms
    /// already diverge enough on their leading byte that real Lucene's
    /// writer gives every top-level leading-byte group its own `.tim` block
    /// before the root ever accumulates `minItemsInBlock` raw terms of its
    /// own -- `writeBlocks(0, count)` (`Lucene103BlockTreeTermsWriter.java`)
    /// then sees only `PendingBlock` entries at the root, never a loose
    /// `PendingTerm`, so the root `PendingBlock`'s own `hasTerms` is `false`.
    ///
    /// Reuses `fixtures/src/GenBlockTreeChildStrategies.java`'s existing
    /// "arraystrat"/"bitsstrat" fields (5 and 9 distinct leading bytes, 30
    /// terms each, comfortably above the default `minItemsInBlock=25`) rather
    /// than adding a new generator -- confirmed structurally here (`assert!
    /// (!root.has_terms)`) to be exactly this shape, which the
    /// `child_strategies_fixture_forces_array_and_bits_strategies` test above
    /// already opens successfully but never asserted the root's own
    /// `has_terms` bit against.
    ///
    /// This is the fixture that caught `open()`'s previous
    /// `Error::Unsupported("root block with no terms (all sub-blocks) ...")`
    /// rejection being unreachable-by-correct-data dead code: real Lucene's
    /// `PendingBlock.compileIndex` always merges every sub-block's own
    /// compiled trie into its parent's, so the `.tip` trie structurally
    /// mirrors the `.tim` block hierarchy at *every* level including the
    /// root -- `collect_leaf_blocks` already walks past a no-terms output
    /// into that node's trie children unconditionally (see its own doc
    /// comment), so `blocks` was never actually empty for this shape; the
    /// check was simply wrong, not a real gap, and was removed rather than
    /// specially handled.
    #[test]
    fn open_field_with_no_terms_in_root_block_still_finds_every_term() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_child_strategies_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTreeChildStrategies)");
        let kv: std::collections::HashMap<String, String> = manifest
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let read_raw = |name: &str| {
            std::fs::read(format!("{dir}{name}.raw"))
                .unwrap_or_else(|_| panic!("missing {name}.raw"))
        };
        let tmd = read_raw(kv.get("tmd_file_name").unwrap());
        let tip = read_raw(kv.get("tip_file_name").unwrap());
        let tim = read_raw(kv.get("tim_file_name").unwrap());
        let fnm = read_raw(kv.get("fnm_file_name").unwrap());
        let id_hex = kv.get("id_hex").unwrap();
        let mut id = [0u8; ID_LENGTH];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = kv.get("segment_suffix").unwrap();
        let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

        let mut tmd_input = SliceInput::new(&tmd);
        codec_util::check_index_header(
            &mut tmd_input,
            TERMS_META_CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        codec_util::check_index_header(
            &mut tmd_input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        let _index_block_size = tmd_input.read_vint().unwrap();
        let num_fields = tmd_input.read_vint().unwrap();
        let mut field_index: std::collections::HashMap<String, (usize, usize, usize)> =
            std::collections::HashMap::new();
        for _ in 0..num_fields {
            let field_number = tmd_input.read_vint().unwrap();
            let _num_terms = tmd_input.read_vlong().unwrap();
            let fi = field_infos.field_by_number(field_number).unwrap();
            read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
            let _doc_count = tmd_input.read_vint().unwrap();
            let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
            let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
            let index_start = tmd_input.read_vlong().unwrap() as usize;
            let root_fp = tmd_input.read_vlong().unwrap() as usize;
            let index_end = tmd_input.read_vlong().unwrap() as usize;
            field_index.insert(fi.name.clone(), (index_start, root_fp, index_end));
        }

        // Structural confirmation, for both fields, that the root trie
        // node's own output really is `hasTerms == false` -- this is the
        // exact shape this test exists to cover, not an incidental fact.
        for name in ["arraystrat", "bitsstrat"] {
            let (index_start, root_fp, index_end) = field_index[name];
            let index_slice = &tip[index_start..index_end];
            let root = load_node(index_slice, root_fp).unwrap();
            assert!(
                !root.has_terms,
                "expected field {name:?}'s root .tip trie node to have \
                 hasTerms == false (every leading-byte group forming its \
                 own sub-block before the root ever keeps a loose term) -- \
                 if this fails, GenBlockTreeChildStrategies.java's label \
                 counts no longer force this shape and this test needs a \
                 different fixture"
            );
            assert_ne!(
                root.sign, SIGN_NO_CHILDREN,
                "a no-terms root with zero trie children would mean every \
                 term is unreachable -- this test's whole point is a \
                 no-terms root that still has children"
            );

            let mut blocks = Vec::new();
            let mut prefix = Vec::new();
            collect_leaf_blocks(index_slice, &root, 0, &mut prefix, &mut blocks).unwrap();
            assert!(
                !blocks.is_empty(),
                "field {name:?}: the previous (removed) `blocks.is_empty()` \
                 check would have wrongly rejected this fixture as \
                 unsupported"
            );
        }

        // Behavioral proof: seek_exact for every known term, plus full
        // ordered TermsEnum::next() enumeration, both still work correctly
        // through a root block that itself holds no terms.
        let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
        let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
        for name in ["arraystrat", "bitsstrat"] {
            let field = fields.field(name).unwrap();
            let terms_tsv = std::fs::read_to_string(format!("{dir}{name}.terms.tsv")).unwrap();
            let mut expected_terms: Vec<&str> = terms_tsv.lines().collect();
            expected_terms.sort_unstable();

            for term in &expected_terms {
                field
                    .seek_exact(term.as_bytes())
                    .unwrap_or_else(|| panic!("term {term:?} not found in field {name}"));
            }

            let mut enumerated: Vec<Vec<u8>> = Vec::new();
            let mut cursor = field.iter();
            while let Some((term, _stats)) = cursor.next() {
                enumerated.push(term.to_vec());
            }
            let expected_bytes: Vec<Vec<u8>> = expected_terms
                .iter()
                .map(|t| t.as_bytes().to_vec())
                .collect();
            assert_eq!(
                enumerated, expected_bytes,
                "field {name:?}: TermsEnum::next() must yield every term in \
                 sorted order even though the root block contributes none \
                 of its own"
            );
        }
    }

    /// Structural proof that `fixtures/data/blocktree_deep_nesting_index/`
    /// (2000 real-Lucene-written terms over a deliberately narrow `{a,b}`
    /// alphabet, `minItemsInBlock=2`/`maxItemsInBlock=4` -- see
    /// `fixtures/src/GenBlockTreeDeepNesting.java`'s module doc for why a
    /// narrow alphabet plus small block-size thresholds is what actually
    /// forces this, where `blocktree_multilevel_index`'s wide-alphabet/
    /// default-thresholds fixture plateaus at a single non-leaf layer no
    /// matter how many terms are added) forces real Lucene to write a
    /// **chain of 4 or more nested non-leaf `.tim` blocks** -- i.e. the
    /// `decode_block`/`decode_block_at_depth` sub-block recursion (see the
    /// module doc's "Multi-level blocktree tries" section) is exercised at
    /// real depth, not just depth 1 (root block -> one internal block ->
    /// leaf, `multilevel_fixture_reaches_a_genuine_non_leaf_block`'s shape).
    ///
    /// This independently re-derives nesting depth via its own from-scratch
    /// walk of each reachable physical block's own bytes (peeking
    /// `isLeafBlock`/`subFP` exactly like [`decode_block_at_depth`] does, but
    /// without calling it, so this test cannot pass merely because
    /// `decode_block_at_depth`'s own bookkeeping is self-consistent) and
    /// asserts a *minimum* depth was reached, not just "some non-leaf block
    /// exists" -- the failure mode a weaker assertion would miss is exactly
    /// a future fixture regen or upstream writer change collapsing back to
    /// shallow nesting while still leaving *some* non-leaf block around.
    #[test]
    fn deep_nesting_fixture_reaches_at_least_four_levels() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_deep_nesting_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTreeDeepNesting)");
        let kv: std::collections::HashMap<String, String> = manifest
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let read_raw = |name: &str| {
            std::fs::read(format!("{dir}{name}.raw"))
                .unwrap_or_else(|_| panic!("missing {name}.raw"))
        };

        let tmd = read_raw(kv.get("tmd_file_name").unwrap());
        let tip = read_raw(kv.get("tip_file_name").unwrap());
        let tim = read_raw(kv.get("tim_file_name").unwrap());
        let fnm = read_raw(kv.get("fnm_file_name").unwrap());
        let id_hex = kv.get("id_hex").unwrap();
        let mut id = [0u8; ID_LENGTH];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = kv.get("segment_suffix").unwrap();
        let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

        // Re-derive "many"'s index_start/root_fp/index_end the same way
        // `open()` does, mirroring `multilevel_fixture_reaches_a_genuine_non_leaf_block`.
        let mut tmd_input = SliceInput::new(&tmd);
        codec_util::check_index_header(
            &mut tmd_input,
            TERMS_META_CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        codec_util::check_index_header(
            &mut tmd_input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        let _index_block_size = tmd_input.read_vint().unwrap();
        let num_fields = tmd_input.read_vint().unwrap();
        let mut field_index = None;
        for _ in 0..num_fields {
            let field_number = tmd_input.read_vint().unwrap();
            let _num_terms = tmd_input.read_vlong().unwrap();
            let fi = field_infos.field_by_number(field_number).unwrap();
            read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
            let _doc_count = tmd_input.read_vint().unwrap();
            let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
            let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
            let index_start = tmd_input.read_vlong().unwrap() as usize;
            let root_fp = tmd_input.read_vlong().unwrap() as usize;
            let index_end = tmd_input.read_vlong().unwrap() as usize;
            if fi.name == "many" {
                field_index = Some((index_start, root_fp, index_end));
            }
        }
        let (index_start, root_fp, index_end) = field_index.expect("field \"many\" in .tmd");

        let index_slice = &tip[index_start..index_end];
        let root = load_node(index_slice, root_fp).unwrap();
        let mut blocks = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(index_slice, &root, 0, &mut prefix, &mut blocks).unwrap();
        assert!(
            blocks.len() > 1,
            "expected the trie to reach more than one physical block"
        );

        // Independently walks a chain of sub-block pointers starting at a
        // trie-reachable block, returning the number of blocks chained
        // (1 for a leaf block with no sub-block entries at all). Reads only
        // the header + suffix-lengths stream (skipping stats/meta bytes
        // wholesale, since sub-block-vs-term entry order and the subFP
        // delta live entirely in the suffix-lengths stream) -- deliberately
        // not sharing any code with `decode_block_at_depth`.
        fn block_chain_depth(tim: &[u8], fp: usize) -> usize {
            let mut r = SliceInput::new(tim);
            r.seek(fp).unwrap();
            let code = r.read_vint().unwrap();
            let ent_count = (code as u32) >> 1;
            let code_l = r.read_vlong().unwrap() as u64;
            let is_leaf_block = (code_l & 0x04) != 0;
            let num_suffix_bytes = (code_l >> 3) as usize;
            let compression_alg = code_l & 0x03;
            let mut suffix_bytes = vec![0u8; num_suffix_bytes];
            match compression_alg {
                0 => r.read_bytes(&mut suffix_bytes).unwrap(),
                1 => decompress_lowercase_ascii(&mut r, &mut suffix_bytes).unwrap(),
                2 => {
                    crate::lz4::decompress(&mut r, num_suffix_bytes, &mut suffix_bytes, 0).unwrap();
                }
                _ => panic!("illegal compression code"),
            }
            let num_suffix_length_bytes_raw = r.read_vint().unwrap() as u32;
            let all_equal = (num_suffix_length_bytes_raw & 1) != 0;
            let num_suffix_length_bytes = (num_suffix_length_bytes_raw >> 1) as usize;
            let mut suffix_length_bytes = vec![0u8; num_suffix_length_bytes];
            if all_equal {
                let b = r.read_byte().unwrap();
                suffix_length_bytes.fill(b);
            } else {
                r.read_bytes(&mut suffix_length_bytes).unwrap();
            }

            if is_leaf_block {
                return 1;
            }
            let mut suffix_lengths_reader = SliceInput::new(&suffix_length_bytes);
            let mut max_child_depth = 0usize;
            for _ in 0..ent_count {
                let entry_code = suffix_lengths_reader.read_vint().unwrap() as u32;
                let is_sub_block = (entry_code & 1) != 0;
                if is_sub_block {
                    let sub_code = suffix_lengths_reader.read_vlong().unwrap() as u64;
                    let sub_fp = fp - sub_code as usize;
                    max_child_depth = max_child_depth.max(block_chain_depth(tim, sub_fp));
                }
            }
            1 + max_child_depth
        }

        let mut max_depth = 0usize;
        for (block_fp, _prefix) in &blocks {
            max_depth = max_depth.max(block_chain_depth(&tim, *block_fp as usize));
        }
        assert!(
            max_depth >= 4,
            "expected at least a 4-block-deep chain of nested non-leaf .tim \
             blocks reachable from the \"many\" field's trie (root -> \
             internal -> internal -> ... -> leaf), got max_depth={max_depth} \
             -- if this fails, GenBlockTreeDeepNesting.java's term \
             shape/block-size thresholds no longer force this, and either \
             the fixture needs retuning or real Lucene's writer heuristics \
             changed"
        );

        // And the full round trip through the *unmodified* public API must
        // still recover every term correctly despite this much deeper
        // nesting (this is the behavioral half; the fuller differential --
        // matching real Lucene's own sorted term list -- lives in
        // `tests/blocktree_deep_nesting_fixture.rs`).
        let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
        let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
        let field = fields.field("many").unwrap();
        let num_terms: i64 = kv.get("field.many.numTerms").unwrap().parse().unwrap();
        assert_eq!(field.num_terms, num_terms);
        let mut it = field.iter();
        while let Some((term, stats)) = it.next() {
            assert_eq!(field.seek_exact(term).unwrap(), stats);
        }
    }

    // -----------------------------------------------------------------
    // c27: the arithmetic gate. One regression test per fix, each written
    // against the *unfixed* code first.
    // -----------------------------------------------------------------

    fn default_field_infos() -> FieldInfos {
        FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        }
    }

    const SAMPLE_TERMS: &[(&str, u32, u64)] = &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)];

    /// `rootFP` is a `.tmd` vlong that `open` casts with `as usize`, so a
    /// negative one arrives at `load_node` as `usize::MAX`. `read_u64_at`'s
    /// bound used to be `fp + 8 > slice.len()`, which panics on that in a
    /// debug build and wraps to `7` -- passing the bound -- in a release one,
    /// only to panic on the slice index right after.
    #[test]
    fn absurd_root_fp_is_a_decode_error_not_an_overflow() {
        let mut b = Builder::new();
        b.ov.root_fp = Some(-1);
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
        let err = open(
            &tim,
            &tip,
            &tmd,
            &default_field_infos(),
            &b.id,
            &b.suffix,
            5,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("trie node read past end of index slice"),
            "{err}"
        );
    }

    /// `numFields` sized a `Vec<(String, FieldTerms)>` reservation directly.
    /// A `FieldTerms` is a couple of hundred bytes, so a corrupt vint bought
    /// a several-hundred-gigabyte reservation -- and an allocation failure
    /// *aborts*, which `catch_unwind` at the FFI boundary cannot intercept.
    #[test]
    fn absurd_num_fields_errors_instead_of_reserving_for_it() {
        let mut b = Builder::new();
        b.ov.num_fields = Some(i32::MAX);
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
        let err = open(
            &tim,
            &tip,
            &tmd,
            &default_field_infos(),
            &b.id,
            &b.suffix,
            5,
        )
        .unwrap_err();
        assert!(
            matches!(err, Error::InvalidNumFields(n) if n == i32::MAX),
            "{err}"
        );
    }

    /// `minTerm`/`maxTerm` are vint-length-prefixed and sized `vec![0u8; n]`
    /// straight off that vint. Java's `new BytesRef(numBytes)` raises a
    /// catchable `OutOfMemoryError`; the Rust equivalent aborts.
    #[test]
    fn absurd_min_term_length_errors_instead_of_allocating_for_it() {
        let mut b = Builder::new();
        b.ov.min_term_len = Some(i32::MAX);
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
        let err = open(
            &tim,
            &tip,
            &tmd,
            &default_field_infos(),
            &b.id,
            &b.suffix,
            5,
        )
        .unwrap_err();
        assert!(
            format!("{err}").contains("exceeds") && format!("{err}").contains("remaining bytes"),
            "{err}"
        );
    }

    /// `totalTermFreq = docFreq + <vlong>` is a `long` add in Java, which
    /// wraps silently; in Rust it panics in a debug build and, in a release
    /// one, produces a *negative* frequency every scorer downstream would
    /// treat as real.
    #[test]
    fn total_term_freq_overflow_is_a_decode_error() {
        let mut b = Builder::new();
        b.ov.ttf_delta = Some(i64::MAX);
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
        let fields = open(
            &tim,
            &tip,
            &tmd,
            &default_field_infos(),
            &b.id,
            &b.suffix,
            5,
        )
        .unwrap();
        let field = fields.field("text").unwrap();
        let err = field.try_seek_exact(b"a").unwrap_err();
        assert!(
            format!("{err}").contains("totalTermFreq overflows"),
            "{err}"
        );
    }

    /// A block's `entCount` drives every scan cursor and
    /// `binary_search_term_leaf`'s bisection bounds. The writer emits at
    /// least one vint per entry into the suffix-lengths region, so
    /// `entCount > numSuffixLengthBytes` is corruption -- and checking it
    /// once per block load is what bounds `ent_count` for every per-entry
    /// proof in this module.
    #[test]
    fn ent_count_past_the_suffix_lengths_region_is_a_decode_error() {
        let mut b = Builder::new();
        b.ov.ent_count = Some(50);
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
        let fields = open(
            &tim,
            &tip,
            &tmd,
            &default_field_infos(),
            &b.id,
            &b.suffix,
            5,
        )
        .unwrap();
        let field = fields.field("text").unwrap();
        let err = field.try_seek_exact(b"a").unwrap_err();
        assert!(
            format!("{err}").contains("exceeds its 3-byte suffix-lengths region"),
            "{err}"
        );
    }

    /// `binarySearchTermLeaf`'s midpoint is `(start + end) >>> 1` in Java --
    /// an *unsigned* shift, which is what keeps the midpoint correct once the
    /// sum passes `Integer.MAX_VALUE`. This port had `>>`, so the same sum
    /// panicked in a debug build and produced a negative `mid` in a release
    /// one. Driven at the `Frame` level because reaching it through
    /// `load_block` would need a gigabyte-scale suffix-lengths region.
    #[test]
    fn binary_search_midpoint_does_not_overflow_for_a_huge_ent_count() {
        let mut frame = Frame {
            ent_count: i32::MAX as u32,
            next_ent: 0,
            is_leaf_block: true,
            all_equal: true,
            // One vint, `0`: every suffix in the block is zero bytes long,
            // so the `suffix_length * ent_count <= suffix_bytes_len` guard
            // passes with an empty suffix region and the bisection runs its
            // full ~31 steps against `start + end` near `2 * i32::MAX`.
            suffix_length_bytes: vec![0u8],
            suffix_length_bytes_len: 1,
            ..Frame::default()
        };
        let mut term = TermBuf::default();
        let mut term_exists = false;
        let status = frame
            .binary_search_term_leaf(&mut term, b"zzz", true, &mut term_exists)
            .unwrap();
        // Every suffix is empty, so every one of them sorts before "zzz":
        // the bisection walks to the very last entry and reports End.
        assert_eq!(status, SeekStatus::End);
        assert_eq!(frame.next_ent, i32::MAX);
    }

    /// Recomputes the trailing 8-byte CRC32 of a codec-footer-terminated
    /// buffer in place, so a byte flip in the body is not simply "caught" by
    /// the checksum -- c15/c19/c25's shape.
    fn resign_footer(buf: &mut [u8]) {
        let len = buf.len();
        let checksum = crc32fast::hash(&buf[..len - 8]) as u64;
        buf[len - 8..].copy_from_slice(&checksum.to_be_bytes());
    }

    /// Re-signed single-byte corruption sweep over the whole `.tim` block
    /// body, the whole `.tip` trie region and the whole `.tmd` record region:
    /// flip one bit, re-sign the footer so the CRC cannot "catch" the
    /// corruption, and require a typed error or a clean decode --
    /// never a panic, and never an allocation abort `catch_unwind` cannot
    /// intercept.
    ///
    /// The assertion is deliberately *not* "every corruption is rejected":
    /// plenty of single-bit flips produce a self-consistent dictionary that
    /// answers differently but well-formedly, which is exactly what the
    /// checksum this sweep defeats exists for.
    #[test]
    fn every_resigned_single_byte_terms_dict_corruption_is_an_error_or_a_clean_decode() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::DocsAndFreqs, SAMPLE_TERMS);
        let fis = default_field_infos();
        // Bodies, past the index header and short of the footer. The `.tmd`
        // also records both other files' lengths in its last 16 body bytes; a
        // flip there is caught by `retrieve_checksum_with_expected_length`
        // rather than by anything semantic, so it is excluded as
        // uninteresting.
        let tim_body = 8..tim.len() - codec_util::FOOTER_LENGTH;
        let tip_body = 8..tip.len() - codec_util::FOOTER_LENGTH;
        let tmd_body = 8..tmd.len() - codec_util::FOOTER_LENGTH - 16;

        let mut total = 0usize;
        let mut rejected = 0usize;

        let exercise = |tim: &[u8], tip: &[u8], tmd: &[u8]| -> bool {
            match open(tim, tip, tmd, &fis, &b.id, &b.suffix, 5) {
                Err(_) => true,
                Ok(fields) => {
                    let Some(field) = fields.field("text") else {
                        return true;
                    };
                    let mut bad = false;
                    for (term, _, _) in SAMPLE_TERMS {
                        if field.try_seek_exact(term.as_bytes()).is_err() {
                            bad = true;
                        }
                    }
                    let mut it = field.iter();
                    if it.try_seek_ceil(b"aa").is_err() {
                        bad = true;
                    }
                    let mut it = field.iter();
                    loop {
                        match it.try_next() {
                            Ok(Some(_)) => {}
                            Ok(None) => break,
                            Err(_) => {
                                bad = true;
                                break;
                            }
                        }
                    }
                    bad
                }
            }
        };

        for off in tim_body {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = tim.clone();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if exercise(&corrupt, &tip, &tmd) {
                    rejected += 1;
                }
            }
        }
        for off in tip_body {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = tip.clone();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if exercise(&tim, &corrupt, &tmd) {
                    rejected += 1;
                }
            }
        }
        for off in tmd_body {
            for mask in [0x01u8, 0x80] {
                let mut corrupt = tmd.clone();
                corrupt[off] ^= mask;
                resign_footer(&mut corrupt);
                total += 1;
                if exercise(&tim, &tip, &corrupt) {
                    rejected += 1;
                }
            }
        }

        // Measured when this was written: 391 of 436. The rest decode to a
        // different but self-consistent dictionary -- c19 measured 44 of 99
        // on `.tip` alone and c25 15 of 43 on `.tvd`, so a rate below 100% is
        // the norm, not a gap. What this pins is that nothing *panics or
        // aborts*, plus a floor so a future change that stops bounding
        // something fails loudly.
        assert_eq!(total, 436);
        assert!(
            rejected >= 385,
            "only {rejected} of {total} re-signed .tim/.tip/.tmd corruptions were rejected"
        );
    }
}
