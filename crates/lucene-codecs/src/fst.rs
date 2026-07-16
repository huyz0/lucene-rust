//! Port of `org.apache.lucene.util.fst.FST` (read side) plus a from-scratch,
//! simplified construction path (`build_fst`) usable by this module's own
//! reader.
//!
//! This module implements enough of Lucene's FST (finite state transducer)
//! format to look up a byte-sequence key in an already-built, on-disk FST and
//! recover its accumulated output (`Util.get(fst, BytesRef)`). It does **not**
//! port real Lucene's *incremental* construction algorithm
//! (`FSTCompiler`/`Builder`'s node-freezing, streaming/memory-bounded build)
//! -- that remains a separate, much larger undertaking (see `docs/parity.md`).
//! Instead, `build_fst` (see the "FST construction" section near the bottom
//! of this file) builds a full in-memory trie upfront and serializes it
//! bottom-up to the same byte format `Fst::read`/`Fst::get` below already
//! parse -- correct and round-trip-verified against this module's own
//! unmodified reader. It **does** reproduce real `FSTCompiler`'s suffix
//! sharing / minimization (via a `NodeHash`-equivalent dedup table keyed on
//! each node's fully-resolved arc signature -- see `NodeHash`/`build_node`),
//! just via whole-tree hash-consing rather than incremental freezing; it does
//! not reproduce fixed-length arc nodes or output pushing. See that section's
//! doc comment for the precise list of what's reproduced and what's deferred.
//!
//! ## Scope of this slice
//!
//! - **On-heap only.** Real Lucene's `FST` can be backed by an off-heap
//!   (mmap'd) `OffHeapFSTStore` or an on-heap `OnHeapFSTStore`. Only the
//!   on-heap representation is ported: the FST body is a plain `Vec<u8>`
//!   (`OnHeapFSTStore` with `bytesArray != null` writes/reads the body as one
//!   contiguous forward byte array — see `OnHeapFSTStore.java`).
//! - **Single output type on the wire: `BytesRef`-shaped
//!   (`ByteSequenceOutputs`).** This is the output type real Lucene uses for
//!   the term index FST (`Lucene90BlockTreeTermsReader`'s `.tip` FST maps
//!   term-prefix byte sequences to concatenated block-pointer byte
//!   sequences), so it's the one needed to eventually navigate BlockTree.
//!   `Fst`/`build_fst`/`build_node` are hardcoded to `Vec<u8>` outputs and
//!   stay that way. Typed output values (`PositiveIntOutputs`,
//!   `ByteSequenceOutputs`, `PairOutputs<A, B>`) are layered *on top of* that
//!   single wire type via the `Outputs` trait and `build_fst_typed`/
//!   `Fst::get_typed`, which just encode/decode a typed value to/from the
//!   `Vec<u8>` this module already stores -- see the "Typed output values"
//!   section near the bottom of this file for why that's a faithful (not a
//!   corner-cut) port of `PairOutputs` given this builder never pushes
//!   output prefixes toward the root.
//! - **Variable-length ("list") arc nodes, plus all three fixed-length arc
//!   node encodings.** Real Lucene's `FSTCompiler` also emits fixed-length
//!   arc nodes for binary search (`ARCS_FOR_BINARY_SEARCH`), direct
//!   addressing (`ARCS_FOR_DIRECT_ADDRESSING`) and continuous ranges
//!   (`ARCS_FOR_CONTINUOUS`) once a node has "enough" arcs to make the
//!   space/speed tradeoff worth it. `ARCS_FOR_BINARY_SEARCH` is ported
//!   (`find_target_arc`'s sparse binary search over fixed-size arc slots,
//!   `read_arc`'s matching `BIT_TARGET_NEXT` target rule); `ARCS_FOR_DIRECT_ADDRESSING`
//!   is also ported (node header's label range + presence bitset ("bit
//!   table"), then fixed-size arc slots addressed only for present labels via
//!   `BitTableUtil`-equivalent bit counting -- see `bit_table_*` helpers below).
//!   `ARCS_FOR_CONTINUOUS` is ported too (`find_target_arc_continuous`):
//!   simpler than direct addressing since every label in the range is
//!   guaranteed present, so there's no presence bit-table and no per-arc
//!   label byte at all -- just `firstLabel` plus a direct
//!   `label - firstLabel` index into fixed-size arc slots.
//! - **Lookup (`get`) and full forward enumeration (`iter`), not seek.** `get`
//!   mirrors `Util.get(FST, BytesRef)`: walk arcs for a specific key and
//!   return whether it's accepted plus its output. `iter`/`FstEnum` mirrors
//!   `BytesRefFSTEnum`'s full ascending-order walk (`next()`) over every
//!   accepted key -- but not its `seekCeil`/`seekFloor`/`seekExact` (needed
//!   for prefix/range-bounded iteration, not a full walk); see `FstEnum`'s
//!   doc comment for the precise scope and `docs/parity.md` for what's
//!   deferred. `IntsRefFSTEnum` (the `IntsRef`-input analogue) is also not
//!   ported, since this module only has a `BytesRef`-shaped output type to
//!   begin with -- see the point above.
//!
//! ## Wire format
//!
//! FST metadata (`FST.FSTMetadata.save`/`FST.readMetadata`) is a normal codec
//! header (`CodecUtil.writeHeader`/`checkHeader`, name `"FST"`, versions 6..9)
//! followed by:
//!
//! ```text
//! AcceptsEmpty(byte)         --> 1 if the FST accepts the empty string, else 0
//! [EmptyOutputLen(vint), EmptyOutputBytesReversed]   -- only if AcceptsEmpty==1
//! InputType(byte)            --> 0=BYTE1, 1=BYTE2, 2=BYTE4
//! StartNode(vlong)
//! NumBytes(vlong)
//! ```
//!
//! `NumBytes` raw FST body bytes follow immediately (`OnHeapFSTStore`'s
//! forward `readBytes` call) -- in this port's fixtures, metadata and body
//! share one file, exactly like `FST.save(Path)` does (`save(out, out)`).
//!
//! The FST body itself is read via a *reverse* byte cursor
//! (`ReverseBytesReader`): `readByte()` returns `bytes[pos]` and then
//! decrements `pos`. Node/arc encoding is otherwise a conventional
//! byte-oriented record format (label, optional output, optional final
//! output, target) whose exact per-arc flag bits are ported verbatim from
//! `FST.java`'s `readArc`/`findTargetArc`/`readFirstRealTargetArc`.

use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::error::Error as StoreError;

/// Errors specific to FST decoding, layered on top of `lucene_store::Error`.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] StoreError),
    #[error("unsupported FST feature: {0}")]
    Unsupported(String),
    #[error("corrupt FST: {0}")]
    Corrupt(String),
}

pub type Result<T> = std::result::Result<T, Error>;

const FILE_FORMAT_NAME: &str = "FST";
const VERSION_START: i32 = 6;
const VERSION_CONTINUOUS_ARCS: i32 = 9;
const VERSION_CURRENT: i32 = VERSION_CONTINUOUS_ARCS;

const BIT_FINAL_ARC: u8 = 1 << 0;
const BIT_LAST_ARC: u8 = 1 << 1;
const BIT_TARGET_NEXT: u8 = 1 << 2;
const BIT_STOP_NODE: u8 = 1 << 3;
const BIT_ARC_HAS_OUTPUT: u8 = 1 << 4;
const BIT_ARC_HAS_FINAL_OUTPUT: u8 = 1 << 5;

const ARCS_FOR_BINARY_SEARCH: u8 = BIT_ARC_HAS_FINAL_OUTPUT;
const ARCS_FOR_DIRECT_ADDRESSING: u8 = 1 << 6;
const ARCS_FOR_CONTINUOUS: u8 = ARCS_FOR_DIRECT_ADDRESSING + ARCS_FOR_BINARY_SEARCH;

const FINAL_END_NODE: i64 = -1;
const NON_FINAL_END_NODE: i64 = 0;

/// `FST.END_LABEL`: the label of the "fake" virtual arc `read_first_target_arc`
/// inserts to represent a node's own acceptance, before falling through to
/// its real outgoing arcs (if any). Only meaningful to enumeration
/// (`FstEnum`/`read_first_target_arc`/`read_next_arc`) -- `get`/
/// `find_target_arc` never see or produce this value (see
/// `find_target_arc`'s own doc comment on why `label_to_match` is always a
/// real byte there).
const END_LABEL: i32 = -1;

fn flag(flags: u8, bit: u8) -> bool {
    flags & bit != 0
}

/// `FST.INPUT_TYPE`: the width of input labels. The BlockTree term index
/// always uses `BYTE1` (raw term bytes); the others are ported for
/// completeness of metadata parsing only.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    Byte1,
    Byte2,
    Byte4,
}

/// A reverse byte-array cursor over the FST body, matching
/// `ReverseBytesReader`: `read_byte` returns the byte at the current position
/// and then moves the cursor *backwards*. Positions are absolute indices into
/// the backing array (never negative in valid usage, but kept as `i64` since
/// Java's `getPosition()`/`setPosition()` use `long`/int freely and a few call
/// sites subtract before checking).
struct BytesReader<'a> {
    bytes: &'a [u8],
    pos: i64,
}

impl<'a> BytesReader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn get_position(&self) -> i64 {
        self.pos
    }

    fn set_position(&mut self, pos: i64) {
        self.pos = pos;
    }

    fn skip_bytes(&mut self, count: i64) {
        self.pos -= count;
    }

    fn read_byte(&mut self) -> Result<u8> {
        let idx = self.pos;
        if idx < 0 || idx as usize >= self.bytes.len() {
            return Err(Error::Corrupt(format!(
                "FST byte-reader position {idx} out of range (len={})",
                self.bytes.len()
            )));
        }
        self.pos -= 1;
        Ok(self.bytes[idx as usize])
    }

    fn read_bytes(&mut self, out: &mut [u8]) -> Result<()> {
        for slot in out.iter_mut() {
            *slot = self.read_byte()?;
        }
        Ok(())
    }

    /// `DataInput.readVInt`: identical varint algorithm regardless of the
    /// direction bytes are physically stored in, since it only ever calls
    /// `read_byte()` in sequence.
    fn read_vint(&mut self) -> Result<i32> {
        let mut b = self.read_byte()?;
        let mut v = (b & 0x7f) as i32;
        let mut shift = 7;
        while b & 0x80 != 0 {
            b = self.read_byte()?;
            v |= ((b & 0x7f) as i32).wrapping_shl(shift);
            shift += 7;
        }
        Ok(v)
    }

    fn read_vlong(&mut self) -> Result<i64> {
        let mut b = self.read_byte()?;
        let mut v = (b & 0x7f) as i64;
        let mut shift = 7;
        while b & 0x80 != 0 {
            b = self.read_byte()?;
            v |= ((b & 0x7f) as i64).wrapping_shl(shift);
            shift += 7;
        }
        Ok(v)
    }

    fn skip_output(&mut self) -> Result<()> {
        let len = self.read_vint()?;
        if len != 0 {
            self.skip_bytes(len as i64);
        }
        Ok(())
    }

    fn read_output(&mut self) -> Result<Vec<u8>> {
        let len = self.read_vint()?;
        if len == 0 {
            return Ok(Vec::new());
        }
        let mut out = vec![0u8; len as usize];
        self.read_bytes(&mut out)?;
        Ok(out)
    }
}

/// `FST.getNumPresenceBytes`: number of bytes needed to hold one presence bit
/// per label in a range of `label_range` labels.
fn num_presence_bytes(label_range: i32) -> i32 {
    (label_range + 7) >> 3
}

/// `BitTableUtil.isBitSet`, pre-positioned via `arc.bit_table_start`
/// (`FST.Arc.BitTable.isBitSet`). `arc.node_flags` must be
/// `ARCS_FOR_DIRECT_ADDRESSING`.
fn bit_table_is_bit_set(bit_index: i32, arc: &Arc, r: &mut BytesReader) -> Result<bool> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    r.set_position(arc.bit_table_start);
    r.skip_bytes((bit_index >> 3) as i64);
    let b = r.read_byte()?;
    Ok(b & (1u8 << (bit_index & 7)) != 0)
}

/// `BitTableUtil.countBits`, pre-positioned via `arc.bit_table_start`: the
/// total number of set bits in the whole bit-table, i.e. the number of arcs
/// actually present in this direct-addressing node.
fn bit_table_count_bits(arc: &Arc, r: &mut BytesReader) -> Result<i32> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    r.set_position(arc.bit_table_start);
    let num_bytes = num_presence_bytes(arc.num_arcs);
    let mut count = 0i32;
    for _ in 0..num_bytes {
        count += r.read_byte()?.count_ones() as i32;
    }
    Ok(count)
}

/// `BitTableUtil.countBitsUpTo`, pre-positioned via `arc.bit_table_start`:
/// the number of set bits strictly before `bit_index` -- i.e. `bit_index`'s
/// corresponding `presence_index` if its bit is set.
fn bit_table_count_bits_up_to(bit_index: i32, arc: &Arc, r: &mut BytesReader) -> Result<i32> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    r.set_position(arc.bit_table_start);
    let full_bytes = bit_index >> 3;
    let mut count = 0i32;
    for _ in 0..full_bytes {
        count += r.read_byte()?.count_ones() as i32;
    }
    let rem_bits = bit_index & 7;
    if rem_bits != 0 {
        let b = r.read_byte()?;
        let mask = (1u8 << rem_bits) - 1;
        count += (b & mask).count_ones() as i32;
    }
    Ok(count)
}

/// `BitTableUtil.nextBitSet`, pre-positioned via `arc.bit_table_start`: the
/// index of the next set bit strictly after `bit_index` (which may be `-1`,
/// meaning "the first set bit"), or `-1` if none.
fn bit_table_next_bit_set(bit_index: i32, arc: &Arc, r: &mut BytesReader) -> Result<i32> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    r.set_position(arc.bit_table_start);
    let bit_table_bytes = num_presence_bytes(arc.num_arcs);
    let mut byte_index = bit_index / 8;
    let shift = ((bit_index + 1) & 7) as u32;
    let mask: i32 = -1i32 << shift;
    let mut i: i32;
    if mask == -1 && bit_index != -1 {
        r.skip_bytes((byte_index + 1) as i64);
        i = 0;
    } else {
        r.skip_bytes(byte_index as i64);
        let b = r.read_byte()? as i32;
        i = (b & 0xff) & mask;
    }
    while i == 0 {
        byte_index += 1;
        if byte_index == bit_table_bytes {
            return Ok(-1);
        }
        i = (r.read_byte()? as i32) & 0xff;
    }
    Ok(i.trailing_zeros() as i32 + (byte_index << 3))
}

/// `BitTableUtil.previousBitSet`, pre-positioned via `arc.bit_table_start`:
/// the index of the next set bit strictly *before* `bit_index`, or `-1` if
/// none. Needed by `FstEnum`'s `seek_floor` support (`FSTEnum`'s
/// `findNextFloorArcDirectAddressing`/`doSeekFloorArrayDirectAddressing`) --
/// `bit_table_next_bit_set`'s mirror-image counterpart.
fn bit_table_previous_bit_set(bit_index: i32, arc: &Arc, r: &mut BytesReader) -> Result<i32> {
    debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
    debug_assert!(bit_index >= 0);
    r.set_position(arc.bit_table_start);
    let mut byte_index = bit_index >> 3;
    r.skip_bytes(byte_index as i64);
    let mask = (1i32 << (bit_index & 7)) - 1;
    let mut i = (r.read_byte()? as i32) & mask;
    while i == 0 {
        if byte_index == 0 {
            return Ok(-1);
        }
        byte_index -= 1;
        r.skip_bytes(-2);
        i = r.read_byte()? as i32;
    }
    Ok(31 - i.leading_zeros() as i32 + (byte_index << 3))
}

/// `ByteSequenceOutputs.add`: concatenate prefix output with an arc's own
/// output.
fn output_add(prefix: &[u8], output: &[u8]) -> Vec<u8> {
    if prefix.is_empty() {
        return output.to_vec();
    }
    if output.is_empty() {
        return prefix.to_vec();
    }
    let mut out = Vec::with_capacity(prefix.len() + output.len());
    out.extend_from_slice(prefix);
    out.extend_from_slice(output);
    out
}

/// `FST.Arc<T>`, restricted to the fields needed for variable-length
/// ("list") arc nodes plus the three array-node encodings this port decodes
/// (`ARCS_FOR_BINARY_SEARCH`, `ARCS_FOR_DIRECT_ADDRESSING`,
/// `ARCS_FOR_CONTINUOUS` -- the latter two share the `first_label`/
/// `pos_arcs_start` fields, `ARCS_FOR_CONTINUOUS` just never needs
/// `bit_table_start`/`presence_index` since every label in its range is
/// present).
#[derive(Debug, Clone, Default)]
pub struct Arc {
    label: i32,
    output: Vec<u8>,
    target: i64,
    flags: u8,
    next_final_output: Vec<u8>,
    next_arc: i64,
    /// `FST.Arc.nodeFlags`: the node header's own flags byte (as opposed to
    /// `flags`, this arc's *own* per-arc flags byte). `0` for list-encoded
    /// nodes (never one of the two special constants below); one of
    /// `ARCS_FOR_BINARY_SEARCH`/`ARCS_FOR_DIRECT_ADDRESSING` for a fixed-
    /// length-arc node. Needed because both share a non-zero `bytes_per_arc`
    /// but decode differently (label storage, target-skip arithmetic).
    node_flags: u8,
    /// `FST.Arc.bytesPerArc`: 0 for list-encoded (variable length arc) nodes;
    /// non-zero for `ARCS_FOR_BINARY_SEARCH`/`ARCS_FOR_DIRECT_ADDRESSING`
    /// fixed-length-arc nodes.
    bytes_per_arc: i32,
    /// `FST.Arc.numArcs`: only meaningful when `bytes_per_arc != 0`. For
    /// `ARCS_FOR_BINARY_SEARCH` this is the arc-array size; for
    /// `ARCS_FOR_DIRECT_ADDRESSING` this is the label range (`numArcs` in
    /// `FST.java`'s own naming, despite actually holding a range width).
    num_arcs: i32,
    /// `FST.Arc.posArcsStart`: only meaningful when `bytes_per_arc != 0`.
    pos_arcs_start: i64,
    /// `FST.Arc.arcIdx`: this arc's index within its fixed-length-arc node's
    /// slots (only meaningful/maintained when `bytes_per_arc != 0`). Needed
    /// by `read_next_real_arc`'s fixed-length-arc branches to advance
    /// incrementally through a node's arcs during enumeration (`get`/
    /// `find_target_arc` never need this: they jump straight to one matched
    /// slot and stop).
    arc_idx: i32,
    /// `FST.Arc.bitTableStart`: start position of the presence bit-table for
    /// an `ARCS_FOR_DIRECT_ADDRESSING` node (only meaningful when
    /// `node_flags == ARCS_FOR_DIRECT_ADDRESSING`).
    bit_table_start: i64,
    /// `FST.Arc.firstLabel`: first (lowest) label of an
    /// `ARCS_FOR_DIRECT_ADDRESSING` node's label range (only meaningful when
    /// `node_flags == ARCS_FOR_DIRECT_ADDRESSING`).
    first_label: i32,
    /// `FST.Arc.presenceIndex`: this arc's index among only the *present*
    /// labels of an `ARCS_FOR_DIRECT_ADDRESSING` node (i.e. the count of set
    /// bits before `arc_idx` in the bit-table) -- only meaningful when
    /// `node_flags == ARCS_FOR_DIRECT_ADDRESSING`.
    presence_index: i32,
}

impl Arc {
    pub fn label(&self) -> i32 {
        self.label
    }

    pub fn output(&self) -> &[u8] {
        &self.output
    }

    pub fn target(&self) -> i64 {
        self.target
    }

    fn flag(&self, bit: u8) -> bool {
        flag(self.flags, bit)
    }

    pub fn is_last(&self) -> bool {
        self.flag(BIT_LAST_ARC)
    }

    pub fn is_final(&self) -> bool {
        self.flag(BIT_FINAL_ARC)
    }

    pub fn next_final_output(&self) -> &[u8] {
        &self.next_final_output
    }
}

fn target_has_arcs(arc: &Arc) -> bool {
    arc.target > 0
}

/// Parsed `FST.FSTMetadata`.
#[derive(Debug, Clone)]
pub struct FstMetadata {
    pub input_type: InputType,
    pub empty_output: Option<Vec<u8>>,
    pub start_node: i64,
    pub version: i32,
    pub num_bytes: i64,
}

/// The FST body ("`OnHeapFSTStore`'s bytes"), either owned by this `Fst` or
/// borrowed from a caller-owned byte slice.
///
/// ## On "off-heap" storage in this port
///
/// Real Lucene's `OffHeapFSTStore` backs the FST body with a slice into a
/// memory-mapped file, so opening a large FST never pays for a full extra
/// heap copy of its bytes -- the OS page cache is the only backing store,
/// and pages are faulted in lazily on access. This port's `Directory`
/// abstraction (`lucene_store::directory::MmapDirectory`) already provides
/// exactly that: `MmapDirectory::open` returns `Input::Mapped(memmap2::Mmap)`,
/// a real OS memory mapping that `Deref`s to `&[u8]` without ever copying the
/// file into a `Vec<u8>`.
///
/// What `FstBytes::Borrowed` adds on top is the missing piece for FSTs
/// specifically: previously, the *only* way to construct an `Fst` was
/// `Fst::read`, which unconditionally allocates a fresh `Vec<u8>` and copies
/// `num_bytes` of body into it (see `Fst::read`'s `Owned` path below) --
/// *even if* the caller's `input` was itself already a zero-copy view over an
/// `Input::Mapped` mmap. That copy is real, avoidable off-heap-storage cost:
/// for a large `.tip` FST, it's a full second copy of already-resident,
/// already-addressable bytes. `Fst::read_borrowed` (below) instead slices the
/// body directly out of the caller's `'a`-lifetime buffer via
/// `SliceInput::slice`/`SliceInput::as_slice`, which for an `Input::Mapped`-
/// backed `SliceInput` is a sub-range of the mmap itself -- no allocation, no
/// copy, and (because `memmap2::Mmap` pages are only faulted in as touched)
/// genuinely OS-page-cache-backed the same way real Lucene's
/// `OffHeapFSTStore` is.
///
/// This is **not** a claim that this module itself performs `mmap(2)` --
/// it doesn't, and doesn't need to: that's `lucene-store`'s job, already
/// done. What's added here is the FST-level plumbing (a lifetime-generic
/// `Fst<'a>` and a body representation that can be either owned or borrowed)
/// so an `Fst` can be constructed *without* forcing a copy when the caller
/// already holds zero-copy, mmap-backed bytes. If this port had no zero-copy
/// `Directory` backend at all, the honest version of this task would be
/// scoped down further to "borrowing an arbitrary caller-owned `&[u8]`,
/// which happens not to be backed by mmap" -- that is not the case here,
/// since `MmapDirectory` already exists and is exercised by
/// `crates/lucene-store/src/directory.rs`'s own tests.
#[derive(Debug, Clone)]
enum FstBytes<'a> {
    Owned(Vec<u8>),
    Borrowed(&'a [u8]),
}

impl std::ops::Deref for FstBytes<'_> {
    type Target = [u8];
    fn deref(&self) -> &[u8] {
        match self {
            FstBytes::Owned(v) => v,
            FstBytes::Borrowed(s) => s,
        }
    }
}

/// A `ByteSequenceOutputs`-typed FST, read from bytes written by real
/// Lucene's `FST.save`.
///
/// `'a` is `'static` for `Fst::read`'s owned path (`Fst<'static>`, i.e.
/// plain `Fst` in the common case) and the lifetime of the caller's backing
/// buffer for `Fst::read_borrowed`'s zero-copy path -- see `FstBytes`'s doc
/// comment above for what "off-heap" honestly means in this port.
#[derive(Debug, Clone)]
pub struct Fst<'a> {
    metadata: FstMetadata,
    bytes: FstBytes<'a>,
}

/// Metadata common to both `Fst::read` (owned) and `Fst::read_borrowed`
/// (zero-copy) -- everything in the wire format up to, but not including,
/// the raw body bytes themselves, which each caller handles differently
/// (copy vs. borrow).
struct ParsedMetadata {
    input_type: InputType,
    empty_output: Option<Vec<u8>>,
    start_node: i64,
    version: i32,
    num_bytes: i64,
}

fn read_fst_metadata_prefix(input: &mut SliceInput) -> Result<ParsedMetadata> {
    let header = codec_util::check_header(input, FILE_FORMAT_NAME, VERSION_START, VERSION_CURRENT)?;
    let version = header.version;

    let empty_output = if input.read_byte()? == 1 {
        let num_bytes = input.read_vint()? as usize;
        let mut reversed = vec![0u8; num_bytes];
        input.read_bytes(&mut reversed)?;
        // FSTMetadata.save writes the empty output's bytes reversed so
        // that reading them back with a reverse BytesReader (starting
        // at the last byte) reproduces `outputs.readFinalOutput` in the
        // original order. We only need the plain byte sequence, so
        // reverse it back here instead of building a one-shot reverse
        // reader.
        reversed.reverse();
        Some(reversed)
    } else {
        None
    };

    let input_type = match input.read_byte()? {
        0 => InputType::Byte1,
        1 => InputType::Byte2,
        2 => InputType::Byte4,
        other => return Err(Error::Corrupt(format!("invalid FST input type {other}"))),
    };
    let start_node = input.read_vlong()?;
    let num_bytes = input.read_vlong()?;
    if num_bytes < 0 {
        return Err(Error::Corrupt(format!("negative FST numBytes {num_bytes}")));
    }

    Ok(ParsedMetadata {
        input_type,
        empty_output,
        start_node,
        version,
        num_bytes,
    })
}

impl Fst<'static> {
    /// Port of `FST.readMetadata` + the `FST(FSTMetadata, DataInput)`
    /// constructor's `OnHeapFSTStore` body read, both operating on the same
    /// forward cursor (matching `FST.save(Path)`/`FST.read(Path, Outputs)`,
    /// which write/read metadata and body through one stream).
    ///
    /// Allocates a fresh `Vec<u8>` and copies the body into it (real
    /// Lucene's `OnHeapFSTStore` semantics). See `Fst::read_borrowed` for the
    /// zero-copy alternative when the caller's `input` is itself backed by
    /// mmap'd (or otherwise already-owned) bytes.
    pub fn read(input: &mut SliceInput) -> Result<Fst<'static>> {
        let meta = read_fst_metadata_prefix(input)?;
        let mut bytes = vec![0u8; meta.num_bytes as usize];
        input.read_bytes(&mut bytes)?;

        Ok(Fst {
            metadata: FstMetadata {
                input_type: meta.input_type,
                empty_output: meta.empty_output,
                start_node: meta.start_node,
                version: meta.version,
                num_bytes: meta.num_bytes,
            },
            bytes: FstBytes::Owned(bytes),
        })
    }
}

impl<'a> Fst<'a> {
    /// Zero-copy equivalent of `Fst::read`: parses the same metadata but,
    /// instead of copying `num_bytes` of body into a new `Vec<u8>`, borrows
    /// them directly out of `input`'s own `'a`-lifetime backing buffer via
    /// `SliceInput::slice` -- no allocation for the body at all.
    ///
    /// This is real Lucene's `OffHeapFSTStore` distinction *iff* `input`
    /// itself is backed by a zero-copy source (an `Input::Mapped` mmap, via
    /// `lucene_store::directory::MmapDirectory` + `SliceInput::new` over its
    /// `Deref<Target = [u8]>`): in that case, the returned `Fst<'a>` never
    /// materializes a second full-size copy of the FST body, matching real
    /// Lucene's mmap'd-FST cost model. If `input` instead already wraps an
    /// owned `Vec<u8>` (e.g. from `Directory::open` on a non-mmap backend),
    /// this still avoids the *extra* copy `Fst::read` would have made, but
    /// obviously doesn't retroactively make the caller's own buffer
    /// OS-mmap'd -- see `FstBytes`'s doc comment for the precise claim.
    pub fn read_borrowed(input: &mut SliceInput<'a>) -> Result<Fst<'a>> {
        let meta = read_fst_metadata_prefix(input)?;
        let start = input.position();
        let end = start
            .checked_add(meta.num_bytes as usize)
            .ok_or_else(|| Error::Corrupt(format!("FST numBytes {} overflows", meta.num_bytes)))?;
        let bytes = input.slice(start, end)?;
        input.seek(end)?;

        Ok(Fst {
            metadata: FstMetadata {
                input_type: meta.input_type,
                empty_output: meta.empty_output,
                start_node: meta.start_node,
                version: meta.version,
                num_bytes: meta.num_bytes,
            },
            bytes: FstBytes::Borrowed(bytes),
        })
    }

    pub fn metadata(&self) -> &FstMetadata {
        &self.metadata
    }

    /// `true` if this `Fst`'s body is borrowed (zero-copy) rather than owned
    /// by this struct -- exposed for tests/diagnostics; not needed for
    /// lookup itself, which goes through `reader()`/`Deref` either way.
    pub fn is_borrowed(&self) -> bool {
        matches!(self.bytes, FstBytes::Borrowed(_))
    }

    fn reader(&self) -> BytesReader<'_> {
        BytesReader::new(&self.bytes)
    }

    /// `FST.readLabel`, `BYTE1` only: `get` (the only arc-traversal entry
    /// point this slice implements) rejects any FST whose
    /// `metadata.input_type != Byte1` before ever calling this, so
    /// `BYTE2`/`BYTE4` label decoding (relevant only to non-term-index FSTs)
    /// is out of scope here -- see the module docs and `Fst::get`.
    fn read_label(&self, r: &mut BytesReader) -> Result<i32> {
        debug_assert_eq!(self.metadata.input_type, InputType::Byte1);
        Ok(r.read_byte()? as i32)
    }

    /// `FST.getFirstArc`: the virtual incoming arc to the start node.
    pub fn first_arc(&self) -> Arc {
        let (flags, next_final_output) = match &self.metadata.empty_output {
            Some(out) if !out.is_empty() => (
                BIT_FINAL_ARC | BIT_LAST_ARC | BIT_ARC_HAS_FINAL_OUTPUT,
                out.clone(),
            ),
            Some(_) => (BIT_FINAL_ARC | BIT_LAST_ARC, Vec::new()),
            None => (BIT_LAST_ARC, Vec::new()),
        };
        Arc {
            target: self.metadata.start_node,
            flags,
            next_final_output,
            ..Default::default()
        }
    }

    /// `FST.readArc`: decode one arc, assuming the flags byte has already
    /// been read into `arc.flags` and `r` is positioned right after it.
    fn read_arc(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        arc.label = if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING
            || arc.node_flags == ARCS_FOR_CONTINUOUS
        {
            // Direct-addressing and continuous nodes never store the label
            // explicitly -- it's implied by the arc's position in the label
            // range (`FST.readArc`'s `arc.label = arc.firstLabel() +
            // arc.arcIdx()` branch, shared by both encodings).
            arc.first_label + arc.arc_idx
        } else {
            self.read_label(r)?
        };

        arc.output = if flag(arc.flags, BIT_ARC_HAS_OUTPUT) {
            r.read_output()?
        } else {
            Vec::new()
        };

        arc.next_final_output = if flag(arc.flags, BIT_ARC_HAS_FINAL_OUTPUT) {
            r.read_output()?
        } else {
            Vec::new()
        };

        if flag(arc.flags, BIT_STOP_NODE) {
            arc.target = if flag(arc.flags, BIT_FINAL_ARC) {
                FINAL_END_NODE
            } else {
                NON_FINAL_END_NODE
            };
            arc.next_arc = r.get_position();
        } else if flag(arc.flags, BIT_TARGET_NEXT) {
            arc.next_arc = r.get_position();
            if !flag(arc.flags, BIT_LAST_ARC) {
                if arc.bytes_per_arc == 0 {
                    // List-encoded node: must scan past the remaining
                    // sibling arcs to find this arc's implicit target.
                    self.seek_to_next_node(r)?;
                } else {
                    // Fixed-length-arc node (`ARCS_FOR_BINARY_SEARCH` or
                    // `ARCS_FOR_DIRECT_ADDRESSING`): the target is simply the
                    // position right before the fixed arcs array starts
                    // (`FST.readArc`'s `bytesPerArc() != 0` branch) -- no scan
                    // needed since every arc's on-disk size is known. For
                    // direct addressing, the array only holds *present* arcs,
                    // so its size is the bit-table's set-bit count, not the
                    // label range (`num_arcs`).
                    let num_arcs = if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING {
                        bit_table_count_bits(arc, r)?
                    } else {
                        arc.num_arcs
                    };
                    r.set_position(arc.pos_arcs_start - arc.bytes_per_arc as i64 * num_arcs as i64);
                }
            }
            arc.target = r.get_position();
        } else {
            arc.target = r.read_vlong()?;
            arc.next_arc = r.get_position();
        }
        Ok(())
    }

    /// `FST.seekToNextNode`: skip arcs (list encoding only) until the last
    /// arc of the current node has been consumed.
    fn seek_to_next_node(&self, r: &mut BytesReader) -> Result<()> {
        loop {
            let flags = r.read_byte()?;
            self.read_label(r)?;
            if flag(flags, BIT_ARC_HAS_OUTPUT) {
                r.skip_output()?;
            }
            if flag(flags, BIT_ARC_HAS_FINAL_OUTPUT) {
                r.skip_output()?;
            }
            if !flag(flags, BIT_STOP_NODE) && !flag(flags, BIT_TARGET_NEXT) {
                r.read_vlong()?;
            }
            if flag(flags, BIT_LAST_ARC) {
                return Ok(());
            }
        }
    }

    /// `FST.readPresenceBytes`: record the bit-table's start position and
    /// skip past its `getNumPresenceBytes(arc.numArcs)` bytes (the presence
    /// bits themselves are read lazily, on demand, via the `bit_table_*`
    /// helpers -- matching real Lucene's own "don't read them here, just
    /// skip them" comment).
    fn read_presence_bytes(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        debug_assert!(arc.bytes_per_arc > 0);
        debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
        arc.bit_table_start = r.get_position();
        r.skip_bytes(num_presence_bytes(arc.num_arcs) as i64);
        Ok(())
    }

    /// `FST.readArcByDirectAddressing(Arc, BytesReader, int, int)`: seek to
    /// the arc slot for the (already known-present) `range_index`/
    /// `presence_index` pair and decode it via the shared `read_arc`.
    fn read_arc_by_direct_addressing(
        &self,
        arc: &mut Arc,
        r: &mut BytesReader,
        range_index: i32,
        presence_index: i32,
    ) -> Result<()> {
        r.set_position(arc.pos_arcs_start - presence_index as i64 * arc.bytes_per_arc as i64);
        arc.arc_idx = range_index;
        arc.presence_index = presence_index;
        arc.flags = r.read_byte()?;
        self.read_arc(arc, r)
    }

    /// `FST.findTargetArc`'s `ARCS_FOR_BINARY_SEARCH` branch: `follow`'s
    /// target node has already been confirmed (by the caller) to be a fixed-
    /// length-arc, binary-searchable node, with `r` positioned right after
    /// the node header's `flags` byte. Ported field-for-field from
    /// `FST.java`'s `findTargetArc`: `numArcs`/`bytesPerArc` (both `vint`),
    /// then a sparse binary search over `numArcs` fixed-size arc slots, each
    /// addressed as `posArcsStart - bytesPerArc * idx` (slot `idx`'s flags
    /// byte sits one byte after that address, mirroring the `+1`/`-1`
    /// asymmetry in `FST.java`'s own address arithmetic between the
    /// mid-search label peek and the final `readNextRealArc` call).
    fn find_target_arc_binary_search(
        &self,
        label_to_match: i32,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        let num_arcs = r.read_vint()?;
        let bytes_per_arc = r.read_vint()?;
        let pos_arcs_start = r.get_position();

        let mut low = 0i32;
        let mut high = num_arcs - 1;
        while low <= high {
            let mid = (low + high) >> 1;
            // +1 to skip over the mid arc's flags byte, matching
            // `FST.java`'s `posArcsStart - (bytesPerArc * mid + 1)`.
            r.set_position(pos_arcs_start - (bytes_per_arc as i64 * mid as i64 + 1));
            let mid_label = self.read_label(r)?;
            match mid_label.cmp(&label_to_match) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid - 1,
                std::cmp::Ordering::Equal => {
                    r.set_position(pos_arcs_start - bytes_per_arc as i64 * mid as i64);
                    let flags = r.read_byte()?;
                    let mut arc = Arc {
                        flags,
                        node_flags: ARCS_FOR_BINARY_SEARCH,
                        bytes_per_arc,
                        num_arcs,
                        pos_arcs_start,
                        ..Default::default()
                    };
                    self.read_arc(&mut arc, r)?;
                    return Ok(Some(arc));
                }
            }
        }
        Ok(None)
    }

    /// `FST.findTargetArc`'s `ARCS_FOR_DIRECT_ADDRESSING` branch: `follow`'s
    /// target node has already been confirmed (by the caller) to be a direct-
    /// addressing node, with `r` positioned right after the node header's
    /// `flags` byte. Ported field-for-field from `FST.java`: label range
    /// (`numArcs`, despite the name) + `bytesPerArc` (both `vint`), the
    /// presence bit-table (skipped, not read yet), `firstLabel`, then a
    /// direct `label_to_match - first_label` index into the label range --
    /// rejected outright if out of range or (if in range) not actually
    /// present in the bit-table, otherwise decoded via
    /// `read_arc_by_direct_addressing` at its bit-table-derived presence
    /// index.
    fn find_target_arc_direct_addressing(
        &self,
        label_to_match: i32,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        let num_arcs = r.read_vint()?; // Label range.
        let bytes_per_arc = r.read_vint()?;
        let mut arc = Arc {
            node_flags: ARCS_FOR_DIRECT_ADDRESSING,
            num_arcs,
            bytes_per_arc,
            ..Default::default()
        };
        self.read_presence_bytes(&mut arc, r)?;
        arc.first_label = self.read_label(r)?;
        arc.pos_arcs_start = r.get_position();

        let arc_index = label_to_match - arc.first_label;
        if arc_index < 0 || arc_index >= arc.num_arcs {
            return Ok(None); // Before or after the label range.
        }
        if !bit_table_is_bit_set(arc_index, &arc, r)? {
            return Ok(None); // Arc missing in the range.
        }
        let presence_index = bit_table_count_bits_up_to(arc_index, &arc, r)?;
        self.read_arc_by_direct_addressing(&mut arc, r, arc_index, presence_index)?;
        Ok(Some(arc))
    }

    /// `FST.findTargetArc`'s `ARCS_FOR_CONTINUOUS` branch: `follow`'s target
    /// node has already been confirmed (by the caller) to be a continuous
    /// node, with `r` positioned right after the node header's `flags` byte.
    /// Simpler than `ARCS_FOR_DIRECT_ADDRESSING`'s equivalent since every
    /// label in `[firstLabel, firstLabel + numArcs)` is guaranteed present --
    /// no presence bit-table at all, so a label range hit always resolves to
    /// a real arc. Ported field-for-field from `FST.java`'s `findTargetArc`:
    /// `numArcs`/`bytesPerArc` (both `vint`), then `firstLabel`, then a direct
    /// `label_to_match - first_label` index into the range, rejected if out
    /// of range, otherwise decoded via `read_next_real_arc` from `arc_idx =
    /// arc_index - 1` (mirroring `FST.java`'s own `arc.arcIdx = arcIndex - 1;
    /// return readNextRealArc(arc, in);` -- `read_next_real_arc`'s shared
    /// `bytes_per_arc != 0` branch increments `arc_idx` back to `arc_index`
    /// and seeks to that slot, reusing the exact arithmetic
    /// `ARCS_FOR_BINARY_SEARCH` already established).
    fn find_target_arc_continuous(
        &self,
        label_to_match: i32,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        let num_arcs = r.read_vint()?;
        let bytes_per_arc = r.read_vint()?;
        let first_label = self.read_label(r)?;
        let pos_arcs_start = r.get_position();

        let arc_index = label_to_match - first_label;
        if arc_index < 0 || arc_index >= num_arcs {
            return Ok(None); // Before or after the label range.
        }

        let mut arc = Arc {
            node_flags: ARCS_FOR_CONTINUOUS,
            num_arcs,
            bytes_per_arc,
            first_label,
            pos_arcs_start,
            arc_idx: arc_index - 1,
            ..Default::default()
        };
        self.read_next_real_arc(&mut arc, r)?;
        Ok(Some(arc))
    }

    /// `FST.findTargetArc`: find the arc leaving `follow` labeled
    /// `label_to_match`, or `None` if there is none. List-encoded,
    /// `ARCS_FOR_BINARY_SEARCH`, `ARCS_FOR_DIRECT_ADDRESSING` and
    /// `ARCS_FOR_CONTINUOUS` nodes.
    ///
    /// `label_to_match == END_LABEL` is also supported (needed by
    /// `FstEnum::do_seek_exact`, which -- like real `FSTEnum.doSeekExact` --
    /// re-derives the "fake" accepting arc this way once the target key's
    /// bytes are exhausted): re-synthesizes the same synthetic arc
    /// `read_first_target_arc` would insert, without touching `r` at all.
    /// `Fst::get`'s loop never does this (it calls this once per actual key
    /// byte and then checks `arc.is_final()` directly instead), so for that
    /// caller `label_to_match` is always a real byte value (0..=255).
    fn find_target_arc(
        &self,
        label_to_match: i32,
        follow: &Arc,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        if label_to_match == END_LABEL {
            return Ok(if follow.is_final() {
                let mut arc = Arc {
                    label: END_LABEL,
                    output: follow.next_final_output().to_vec(),
                    ..Default::default()
                };
                if follow.target() <= 0 {
                    arc.flags = BIT_LAST_ARC;
                } else {
                    // NOTE: next_arc is a node (not an address!) in this case.
                    arc.flags = 0;
                    arc.next_arc = follow.target();
                }
                arc.node_flags = arc.flags;
                Some(arc)
            } else {
                None
            });
        }
        debug_assert!((0..=255).contains(&label_to_match));
        if !target_has_arcs(follow) {
            return Ok(None);
        }

        r.set_position(follow.target());
        let flags = r.read_byte()?;
        if flags == ARCS_FOR_BINARY_SEARCH {
            return self.find_target_arc_binary_search(label_to_match, r);
        }
        if flags == ARCS_FOR_DIRECT_ADDRESSING {
            return self.find_target_arc_direct_addressing(label_to_match, r);
        }
        if flags == ARCS_FOR_CONTINUOUS {
            return self.find_target_arc_continuous(label_to_match, r);
        }

        // Linear scan (list-encoded nodes).
        let mut arc = Arc {
            flags,
            ..Default::default()
        };
        loop {
            let flags = arc.flags;
            let pos = r.get_position();
            let label = self.read_label(r)?;
            if label == label_to_match {
                r.set_position(pos);
                self.read_arc(&mut arc, r)?;
                return Ok(Some(arc));
            } else if label > label_to_match || flag(flags, BIT_LAST_ARC) {
                // Either past the label in sorted order, or this was the
                // last arc of the node -- either way, no match.
                return Ok(None);
            } else {
                if flag(flags, BIT_ARC_HAS_OUTPUT) {
                    r.skip_output()?;
                }
                if flag(flags, BIT_ARC_HAS_FINAL_OUTPUT) {
                    r.skip_output()?;
                }
                if !flag(flags, BIT_STOP_NODE) && !flag(flags, BIT_TARGET_NEXT) {
                    r.read_vlong()?;
                }
                arc.flags = r.read_byte()?;
            }
        }
    }

    // --- Seek support helpers (`FSTEnum`'s array/direct/continuous "read
    // arc N", "read last arc" and "peek next label" primitives) -----------
    //
    // These are additional per-arc random-access primitives `find_target_arc`
    // and the ordered-enumeration methods above never needed (they either
    // jump straight to one matched arc and stop, or only ever advance
    // forward one arc at a time): reading an arbitrary arc by index within a
    // fixed-length-arc node, reading a node's *last* arc directly (without
    // scanning through every arc first), and peeking at the label of the
    // arc following the current one without mutating it. `FstEnum`'s
    // `seek_ceil`/`seek_floor`/`seek_exact` (below) are built on top of
    // these plus the existing `find_target_arc`/`read_first_real_target_arc`/
    // `read_next_real_arc`.

    /// `FST.readArcByIndex`: read the arc at (0-based) slot `idx` of an
    /// `ARCS_FOR_BINARY_SEARCH` node -- `arc` must already carry that node's
    /// header fields (`pos_arcs_start`/`bytes_per_arc`/`node_flags`), as set
    /// up by `find_target_arc_binary_search`/`read_first_real_target_arc`/
    /// `find_arc_binary_search`.
    fn read_arc_by_index(&self, arc: &mut Arc, r: &mut BytesReader, idx: i32) -> Result<()> {
        debug_assert!(arc.bytes_per_arc > 0);
        debug_assert_eq!(arc.node_flags, ARCS_FOR_BINARY_SEARCH);
        debug_assert!(idx >= 0 && idx < arc.num_arcs);
        r.set_position(arc.pos_arcs_start - idx as i64 * arc.bytes_per_arc as i64);
        arc.arc_idx = idx;
        arc.flags = r.read_byte()?;
        self.read_arc(arc, r)
    }

    /// `FST.readArcByContinuous`: read the arc at (0-based) range-index
    /// `range_index` of an `ARCS_FOR_CONTINUOUS` node.
    fn read_arc_by_continuous(
        &self,
        arc: &mut Arc,
        r: &mut BytesReader,
        range_index: i32,
    ) -> Result<()> {
        debug_assert!(range_index >= 0 && range_index < arc.num_arcs);
        r.set_position(arc.pos_arcs_start - range_index as i64 * arc.bytes_per_arc as i64);
        arc.arc_idx = range_index;
        arc.flags = r.read_byte()?;
        self.read_arc(arc, r)
    }

    /// `FST.readLastArcByDirectAddressing`: read an `ARCS_FOR_DIRECT_ADDRESSING`
    /// node's last (highest-labeled) present arc directly, without scanning
    /// through the whole bit-table arc by arc.
    fn read_last_arc_by_direct_addressing(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        let presence_index = bit_table_count_bits(arc, r)? - 1;
        let range_index = arc.num_arcs - 1;
        self.read_arc_by_direct_addressing(arc, r, range_index, presence_index)
    }

    /// `FST.readLastArcByContinuous`: read an `ARCS_FOR_CONTINUOUS` node's
    /// last arc.
    fn read_last_arc_by_continuous(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        let range_index = arc.num_arcs - 1;
        self.read_arc_by_continuous(arc, r, range_index)
    }

    /// `FST.readLastTargetArc`: follow `follow` and read the *last* arc of
    /// its target node (as opposed to `read_first_real_target_arc`'s first
    /// arc) -- needed by `FstEnum::push_last`, `seek_floor`'s "beyond the
    /// last arc of this range" cases.
    fn read_last_target_arc(&self, follow: &Arc, r: &mut BytesReader) -> Result<Arc> {
        if !target_has_arcs(follow) {
            debug_assert!(follow.is_final());
            return Ok(Arc {
                label: END_LABEL,
                target: FINAL_END_NODE,
                output: follow.next_final_output().to_vec(),
                flags: BIT_LAST_ARC,
                node_flags: BIT_LAST_ARC,
                ..Default::default()
            });
        }

        r.set_position(follow.target());
        let flags = r.read_byte()?;
        let mut arc = Arc {
            node_flags: flags,
            ..Default::default()
        };
        if flags == ARCS_FOR_BINARY_SEARCH
            || flags == ARCS_FOR_DIRECT_ADDRESSING
            || flags == ARCS_FOR_CONTINUOUS
        {
            arc.num_arcs = r.read_vint()?;
            arc.bytes_per_arc = r.read_vint()?;
            if flags == ARCS_FOR_DIRECT_ADDRESSING {
                self.read_presence_bytes(&mut arc, r)?;
                arc.first_label = self.read_label(r)?;
                arc.pos_arcs_start = r.get_position();
                self.read_last_arc_by_direct_addressing(&mut arc, r)?;
            } else if flags == ARCS_FOR_BINARY_SEARCH {
                arc.arc_idx = arc.num_arcs - 2;
                arc.pos_arcs_start = r.get_position();
                self.read_next_real_arc(&mut arc, r)?;
            } else {
                arc.first_label = self.read_label(r)?;
                arc.pos_arcs_start = r.get_position();
                self.read_last_arc_by_continuous(&mut arc, r)?;
            }
        } else {
            arc.flags = flags;
            arc.bytes_per_arc = 0;
            while !flag(arc.flags, BIT_LAST_ARC) {
                self.read_label(r)?;
                if flag(arc.flags, BIT_ARC_HAS_OUTPUT) {
                    r.skip_output()?;
                }
                if flag(arc.flags, BIT_ARC_HAS_FINAL_OUTPUT) {
                    r.skip_output()?;
                }
                if !flag(arc.flags, BIT_STOP_NODE) && !flag(arc.flags, BIT_TARGET_NEXT) {
                    r.read_vlong()?;
                }
                arc.flags = r.read_byte()?;
            }
            // Undo the flags byte just read.
            r.skip_bytes(-1);
            arc.next_arc = r.get_position();
            self.read_next_real_arc(&mut arc, r)?;
        }
        debug_assert!(arc.is_last());
        Ok(arc)
    }

    /// `FST.readNextArcLabel`: peek at the label of the arc immediately
    /// following `arc` (which must not be the node's last arc) without
    /// mutating `arc` itself.
    fn read_next_arc_label(&self, arc: &Arc, r: &mut BytesReader) -> Result<i32> {
        debug_assert!(!arc.is_last());
        if arc.label == END_LABEL {
            r.set_position(arc.next_arc);
            let flags = r.read_byte()?;
            if flags == ARCS_FOR_BINARY_SEARCH
                || flags == ARCS_FOR_DIRECT_ADDRESSING
                || flags == ARCS_FOR_CONTINUOUS
            {
                let num_arcs = r.read_vint()?;
                r.read_vint()?; // bytes_per_arc, unused here.
                if flags == ARCS_FOR_BINARY_SEARCH {
                    r.read_byte()?; // Skip the arc's own flags byte.
                } else if flags == ARCS_FOR_DIRECT_ADDRESSING {
                    r.skip_bytes(num_presence_bytes(num_arcs) as i64);
                }
            }
        } else {
            match arc.node_flags {
                ARCS_FOR_BINARY_SEARCH => {
                    r.set_position(
                        arc.pos_arcs_start
                            - (1 + arc.arc_idx) as i64 * arc.bytes_per_arc as i64
                            - 1,
                    );
                }
                ARCS_FOR_DIRECT_ADDRESSING => {
                    let next_index = bit_table_next_bit_set(arc.arc_idx, arc, r)?;
                    debug_assert!(next_index != -1);
                    return Ok(arc.first_label + next_index);
                }
                ARCS_FOR_CONTINUOUS => {
                    return Ok(arc.first_label + arc.arc_idx + 1);
                }
                _ => {
                    debug_assert_eq!(arc.bytes_per_arc, 0);
                    r.set_position(arc.next_arc - 1);
                }
            }
        }
        self.read_label(r)
    }

    /// `Util.binarySearch`: binary search an `ARCS_FOR_BINARY_SEARCH` node's
    /// arcs for `target_label`, starting from `arc.arc_idx` (usually `0`, but
    /// `find_next_floor_arc_binary_search` resumes from a nonzero index).
    /// Returns the matching slot index if found, else `-1 - insertion_point`
    /// (matching `Collections.binarySearch`'s / `Util.binarySearch`'s own
    /// negative-encoding convention).
    fn find_arc_binary_search(
        &self,
        arc: &Arc,
        target_label: i32,
        r: &mut BytesReader,
    ) -> Result<i32> {
        debug_assert_eq!(arc.node_flags, ARCS_FOR_BINARY_SEARCH);
        let mut low = arc.arc_idx;
        let mut high = arc.num_arcs - 1;
        while low <= high {
            let mid = (low + high) >> 1;
            r.set_position(arc.pos_arcs_start);
            r.skip_bytes(arc.bytes_per_arc as i64 * mid as i64 + 1);
            let mid_label = self.read_label(r)?;
            match mid_label.cmp(&target_label) {
                std::cmp::Ordering::Less => low = mid + 1,
                std::cmp::Ordering::Greater => high = mid - 1,
                std::cmp::Ordering::Equal => return Ok(mid),
            }
        }
        Ok(-1 - low)
    }

    /// `FSTEnum.findNextFloorArcBinarySearch`: given `arc` positioned at its
    /// node's first arc, find and read the arc whose label is strictly less
    /// than `target_label` (skipping the first arc, which has already been
    /// read and is used as-is if it's already the floor arc).
    fn find_next_floor_arc_binary_search(
        &self,
        arc: &mut Arc,
        target_label: i32,
        r: &mut BytesReader,
    ) -> Result<()> {
        debug_assert_eq!(arc.node_flags, ARCS_FOR_BINARY_SEARCH);
        debug_assert_eq!(arc.arc_idx, 0);
        if arc.num_arcs > 1 {
            let idx = self.find_arc_binary_search(arc, target_label, r)?;
            debug_assert!(idx != -1);
            if idx > 1 {
                self.read_arc_by_index(arc, r, idx - 1)?;
            } else if idx < -2 {
                self.read_arc_by_index(arc, r, -2 - idx)?;
            }
        }
        Ok(())
    }

    /// `FSTEnum.findNextFloorArcDirectAddressing`: direct-addressing analogue
    /// of `find_next_floor_arc_binary_search`.
    fn find_next_floor_arc_direct_addressing(
        &self,
        arc: &mut Arc,
        target_label: i32,
        r: &mut BytesReader,
    ) -> Result<()> {
        debug_assert_eq!(arc.node_flags, ARCS_FOR_DIRECT_ADDRESSING);
        debug_assert_eq!(arc.label, arc.first_label);
        if arc.num_arcs > 1 {
            let target_index = target_label - arc.first_label;
            debug_assert!(target_index >= 0);
            if target_index >= arc.num_arcs {
                self.read_last_arc_by_direct_addressing(arc, r)?;
            } else {
                let floor_index = bit_table_previous_bit_set(target_index, arc, r)?;
                if floor_index > 0 {
                    let presence_index = bit_table_count_bits_up_to(floor_index, arc, r)?;
                    self.read_arc_by_direct_addressing(arc, r, floor_index, presence_index)?;
                }
            }
        }
        Ok(())
    }

    /// `FSTEnum.findNextFloorArcContinuous`: continuous-node analogue of
    /// `find_next_floor_arc_binary_search`.
    fn find_next_floor_arc_continuous(
        &self,
        arc: &mut Arc,
        target_label: i32,
        r: &mut BytesReader,
    ) -> Result<()> {
        debug_assert_eq!(arc.node_flags, ARCS_FOR_CONTINUOUS);
        debug_assert_eq!(arc.label, arc.first_label);
        if arc.num_arcs > 1 {
            let target_index = target_label - arc.first_label;
            debug_assert!(target_index >= 0);
            if target_index >= arc.num_arcs {
                self.read_last_arc_by_continuous(arc, r)?;
            } else {
                self.read_arc_by_continuous(arc, r, target_index - 1)?;
            }
        }
        Ok(())
    }

    // --- Ordered enumeration (`BytesRefFSTEnum`) --------------------------
    //
    // The three methods below are `FST.java`'s `readFirstRealTargetArc`/
    // `readNextRealArc`/`readFirstTargetArc`/`readNextArc`, ported field-for-
    // field. Unlike `find_target_arc` (which only ever needs to *find* one
    // specific labeled arc and can stop as soon as it does), full ordered
    // enumeration needs to walk every arc of every node it visits in turn --
    // real Lucene's `readNextRealArc` is exactly that "read the arc after
    // this one" primitive, generic over list-encoded and all three
    // fixed-length-arc node encodings alike, which is what `find_target_arc`
    // never needed and so never implemented.

    /// `FST.readNextRealArc`: advance `arc` in-place to the arc immediately
    /// following it within the same node (never the virtual `END_LABEL`
    /// arc -- callers must route through `read_next_arc` for that). `arc`
    /// must already be positioned at a real arc of the node (as set up by
    /// `read_first_real_target_arc` or a previous call to this method).
    fn read_next_real_arc(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING {
            // Direct-addressing node: advance to the next *present* label in
            // the range (the bit-table's next set bit after `arc_idx`, which
            // is `-1` on the very first call).
            let next_index = bit_table_next_bit_set(arc.arc_idx, arc, r)?;
            return self.read_arc_by_direct_addressing(arc, r, next_index, arc.presence_index + 1);
        } else if arc.bytes_per_arc != 0 {
            // `ARCS_FOR_BINARY_SEARCH` or `ARCS_FOR_CONTINUOUS` fixed-length-arc
            // node: advance to the next fixed-size slot by index (identical
            // arithmetic for both -- `ARCS_FOR_CONTINUOUS` never needs to
            // consult a presence bit-table since every slot in its range is
            // present).
            arc.arc_idx += 1;
            debug_assert!(arc.arc_idx >= 0 && arc.arc_idx < arc.num_arcs);
            r.set_position(arc.pos_arcs_start - arc.bytes_per_arc as i64 * arc.arc_idx as i64);
            arc.flags = r.read_byte()?;
        } else {
            // List-encoded node: `arc.next_arc` is the position `read_arc`
            // already computed as "where the next arc's flags byte lives".
            r.set_position(arc.next_arc);
            arc.flags = r.read_byte()?;
        }
        self.read_arc(arc, r)
    }

    /// `FST.readFirstRealTargetArc`: read the first real (non-virtual) arc
    /// of the node at `node_address`, handling list-encoded and all three
    /// fixed-length-arc node headers (mirrors `FST.readFirstArcInfo` followed
    /// by one `readNextRealArc` call).
    fn read_first_real_target_arc(&self, node_address: i64, r: &mut BytesReader) -> Result<Arc> {
        r.set_position(node_address);
        let flags = r.read_byte()?;
        let mut arc = Arc {
            node_flags: flags,
            ..Default::default()
        };
        if flags == ARCS_FOR_BINARY_SEARCH {
            let num_arcs = r.read_vint()?;
            let bytes_per_arc = r.read_vint()?;
            arc.num_arcs = num_arcs;
            arc.bytes_per_arc = bytes_per_arc;
            arc.pos_arcs_start = r.get_position();
            arc.arc_idx = -1;
        } else if flags == ARCS_FOR_DIRECT_ADDRESSING {
            let num_arcs = r.read_vint()?; // Label range.
            let bytes_per_arc = r.read_vint()?;
            arc.num_arcs = num_arcs;
            arc.bytes_per_arc = bytes_per_arc;
            self.read_presence_bytes(&mut arc, r)?;
            arc.first_label = self.read_label(r)?;
            arc.presence_index = -1;
            arc.pos_arcs_start = r.get_position();
            arc.arc_idx = -1;
        } else if flags == ARCS_FOR_CONTINUOUS {
            let num_arcs = r.read_vint()?;
            let bytes_per_arc = r.read_vint()?;
            arc.num_arcs = num_arcs;
            arc.bytes_per_arc = bytes_per_arc;
            arc.first_label = self.read_label(r)?;
            arc.pos_arcs_start = r.get_position();
            arc.arc_idx = -1;
        } else {
            // List-encoded node: `next_arc` re-anchors `read_next_real_arc`
            // back at this node's first arc (its flags byte gets re-read,
            // matching `FST.java`'s own re-read of the same byte here).
            arc.next_arc = node_address;
            arc.bytes_per_arc = 0;
            arc.node_flags = 0;
        }
        self.read_next_real_arc(&mut arc, r)?;
        Ok(arc)
    }

    /// `FST.readFirstTargetArc`: read the first arc leaving `follow`'s
    /// target node -- if `follow` is itself an accepting (final) arc, that
    /// "first arc" is a synthetic `END_LABEL` arc representing acceptance at
    /// this node, inserted *before* any of the node's real outgoing arcs
    /// (matching real Lucene's enumeration order: a node's own key, if any,
    /// sorts before all of its longer descendants).
    fn read_first_target_arc(&self, follow: &Arc, r: &mut BytesReader) -> Result<Arc> {
        if follow.is_final() {
            let mut arc = Arc {
                label: END_LABEL,
                output: follow.next_final_output().to_vec(),
                flags: BIT_FINAL_ARC,
                target: FINAL_END_NODE,
                ..Default::default()
            };
            if follow.target() <= 0 {
                arc.flags |= BIT_LAST_ARC;
            } else {
                // Not really an address here -- a node address to resume
                // real-arc enumeration from once this virtual arc is
                // exhausted (see `read_next_arc`).
                arc.next_arc = follow.target();
            }
            Ok(arc)
        } else {
            self.read_first_real_target_arc(follow.target(), r)
        }
    }

    /// `FST.readNextArc`: advance `arc` to the arc following it, transparently
    /// handling the case where `arc` was itself the synthetic `END_LABEL` arc
    /// (in which case "next" means the target node's first *real* arc).
    fn read_next_arc(&self, arc: &Arc, r: &mut BytesReader) -> Result<Arc> {
        if arc.label == END_LABEL {
            if arc.next_arc <= 0 {
                return Err(Error::Corrupt(
                    "read_next_arc called on the last arc of a node".to_string(),
                ));
            }
            self.read_first_real_target_arc(arc.next_arc, r)
        } else {
            let mut next = arc.clone();
            self.read_next_real_arc(&mut next, r)?;
            Ok(next)
        }
    }

    /// Port of `BytesRefFSTEnum`'s full forward walk (`FSTEnum.doNext`/
    /// `pushFirst`, no seek support -- see `Fst::iter`'s doc comment):
    /// returns an iterator over every `(key, output)` pair this FST accepts,
    /// in ascending key order.
    pub fn iter(&self) -> Result<FstEnum<'_, 'a>> {
        if self.metadata.input_type != InputType::Byte1 {
            return Err(Error::Unsupported(
                "Fst::iter only supports INPUT_TYPE.BYTE1 (term-index FSTs)".to_string(),
            ));
        }
        Ok(FstEnum::new(self))
    }

    /// Port of `Util.get(FST, BytesRef)`: look up `key` and return its
    /// accumulated output if the FST accepts it, else `None`.
    pub fn get(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        if self.metadata.input_type != InputType::Byte1 {
            return Err(Error::Unsupported(
                "Fst::get only supports INPUT_TYPE.BYTE1 (term-index FSTs)".to_string(),
            ));
        }

        let mut r = self.reader();
        let mut arc = self.first_arc();
        // The virtual first arc isn't itself the answer for an empty key --
        // Util.get walks `input.length` arcs and only then checks
        // `arc.isFinal()`, so an empty key's answer is the empty output.
        let mut output: Vec<u8> = Vec::new();

        for &b in key {
            match self.find_target_arc(b as i32, &arc, &mut r)? {
                Some(next) => {
                    output = output_add(&output, next.output());
                    arc = next;
                }
                None => return Ok(None),
            }
        }

        if arc.is_final() {
            Ok(Some(output_add(&output, arc.next_final_output())))
        } else {
            Ok(None)
        }
    }

    /// Port of `BytesRefFSTEnum.seekExact`: look up `key` and return its
    /// accumulated output if the FST accepts it, else `None`.
    ///
    /// This legitimately delegates to `Fst::get` rather than re-implementing
    /// an independent arc-by-arc descent: `get` *is* that descent (walk one
    /// arc per key byte via `find_target_arc`, never a linear scan over the
    /// FST's full contents), which is exactly what real `FSTEnum.doSeekExact`
    /// does too (its main loop is `fst.findTargetArc(targetLabel, arc,
    /// getArc(upto), fstReader)` called once per key byte) -- the only
    /// difference is `doSeekExact` additionally maintains `FSTEnum`'s
    /// persistent per-depth arc/output stack so a subsequent `next()` can
    /// resume enumeration from where the seek landed, which this
    /// non-iterator convenience method has no need for. See
    /// `FstEnum::seek_exact` for the stateful, stack-maintaining equivalent
    /// used when seeking is followed by continued enumeration.
    pub fn seek_exact(&self, key: &[u8]) -> Result<Option<Vec<u8>>> {
        self.get(key)
    }
}

/// Port of `BytesRefFSTEnum`, restricted to a full forward walk (real
/// Lucene's `next()`) -- **not** ported: `seekCeil`/`seekFloor`/`seekExact`
/// (`FSTEnum.doSeekCeil`/`doSeekFloor`/`doSeekExact`, `rewindPrefix`'s shared-
/// prefix-aware re-seek), which real Lucene needs for range/prefix queries
/// over a term dictionary but which is a substantially larger increment on
/// top of this (each seek variant has its own list/binary-search/direct-
/// addressing/continuous dispatch -- see `FSTEnum.java`). This type only
/// walks every accepted key in ascending order from the start, which is
/// already enough to enumerate any FST's full contents; see `docs/parity.md`
/// for the deferred seek support and `IntsRefFSTEnum`'s int-sequence variant
/// (this port only has `BytesRef`-shaped output, so only the `BytesRef` key
/// side of enumeration -- i.e. this type -- was ported; real Lucene's
/// `IntsRefFSTEnum` enumerates `IntsRef` *inputs* against a differently-typed
/// FST and shares the same `FSTEnum` base class this port's `read_first_target_arc`
/// et al. mirror, but was not itself needed for any consumer in this port).
///
/// Constructed via `Fst::iter`. Implements `Iterator<Item = Result<(Vec<u8>,
/// Vec<u8>)>>`, yielding `(key, output)` pairs in ascending key order.
pub struct FstEnum<'f, 'a> {
    fst: &'f Fst<'a>,
    r: BytesReader<'f>,
    /// `FSTEnum.arcs`: `arcs[0]` is the virtual incoming arc to the start
    /// node (`Fst::first_arc`); `arcs[i]` (`i >= 1`) is the arc last read at
    /// depth `i`, potentially the synthetic `END_LABEL` arc representing
    /// acceptance at that depth.
    arcs: Vec<Arc>,
    /// `FSTEnum.output`: `outputs[i]` is the cumulative output through
    /// depth `i` (`outputs[0]` is always empty -- no output accumulated for
    /// the virtual root arc).
    outputs: Vec<Vec<u8>>,
    /// `BytesRefFSTEnum.current`: `labels[i]` (`i >= 1`) is the key byte
    /// chosen at depth `i` -- i.e. the current key, once accepted, is
    /// `labels[1..upto]`.
    labels: Vec<u8>,
    /// `FSTEnum.upto`.
    upto: usize,
    /// Set once enumeration has yielded every key, so a subsequent `next()`
    /// call keeps returning `None` (a fused iterator) instead of
    /// misreading `upto == 0`'s *other* meaning -- "not yet started" -- and
    /// restarting from the first key. `upto` alone can't distinguish these
    /// two states: both leave it at `0`.
    done: bool,
    /// `BytesRefFSTEnum.target`: the key most recently passed to
    /// `seek_ceil`/`seek_floor`/`seek_exact`, byte-indexed from `0` (unlike
    /// `labels`, which is 1-indexed by depth) -- empty/unused until the
    /// first seek call.
    target: Vec<u8>,
    /// `FSTEnum.targetLength`: `target.len()`, cached since `get_target_label`
    /// (`BytesRefFSTEnum.getTargetLabel`) is called once per arc visited
    /// during a seek.
    target_length: usize,
}

impl<'f, 'a> FstEnum<'f, 'a> {
    fn new(fst: &'f Fst<'a>) -> Self {
        FstEnum {
            r: fst.reader(),
            fst,
            arcs: vec![fst.first_arc()],
            outputs: vec![Vec::new()],
            labels: vec![0u8],
            upto: 0,
            done: false,
            target: Vec::new(),
            target_length: 0,
        }
    }

    fn set_arc(&mut self, idx: usize, arc: Arc) {
        if self.arcs.len() <= idx {
            self.arcs.resize(idx + 1, Arc::default());
        }
        self.arcs[idx] = arc;
    }

    fn set_output(&mut self, idx: usize, output: Vec<u8>) {
        if self.outputs.len() <= idx {
            self.outputs.resize(idx + 1, Vec::new());
        }
        self.outputs[idx] = output;
    }

    fn set_label(&mut self, idx: usize, label: u8) {
        if self.labels.len() <= idx {
            self.labels.resize(idx + 1, 0u8);
        }
        self.labels[idx] = label;
    }

    /// `FSTEnum.pushFirst`: from `self.arcs[self.upto]`, repeatedly descend
    /// via `read_first_target_arc` until landing on an accepting (synthetic
    /// `END_LABEL`) arc -- the smallest-keyed accepted descendant of the
    /// node `self.upto` started at.
    fn push_first(&mut self) -> Result<()> {
        loop {
            let arc = self.arcs[self.upto].clone();
            let cum_output = output_add(&self.outputs[self.upto - 1], arc.output());
            self.set_output(self.upto, cum_output);
            if arc.label() == END_LABEL {
                break;
            }
            self.set_label(self.upto, arc.label() as u8);
            self.upto += 1;
            let next = self.fst.read_first_target_arc(&arc, &mut self.r)?;
            self.set_arc(self.upto, next);
        }
        Ok(())
    }

    /// `FSTEnum.pushLast`: from `self.arcs[self.upto]`, repeatedly descend
    /// via `read_last_target_arc` until landing on an accepting (synthetic
    /// `END_LABEL`) arc -- the largest-keyed accepted descendant of the node
    /// `self.upto` started at. `push_first`'s mirror image, needed by
    /// `seek_floor`'s "target falls strictly between two sibling arcs, or
    /// past the last one" cases.
    fn push_last(&mut self) -> Result<()> {
        loop {
            let arc = self.arcs[self.upto].clone();
            self.set_label(self.upto, arc.label() as u8);
            let cum_output = output_add(&self.outputs[self.upto - 1], arc.output());
            self.set_output(self.upto, cum_output);
            if arc.label() == END_LABEL {
                break;
            }
            self.upto += 1;
            let next = self.fst.read_last_target_arc(&arc, &mut self.r)?;
            self.set_arc(self.upto, next);
        }
        Ok(())
    }

    // --- getTargetLabel/getCurrentLabel/setCurrentLabel/incr -------------

    /// `BytesRefFSTEnum.getTargetLabel`.
    fn get_target_label(&self) -> i32 {
        if self.upto - 1 == self.target_length {
            END_LABEL
        } else {
            self.target[self.upto - 1] as i32
        }
    }

    /// `BytesRefFSTEnum.getCurrentLabel`.
    fn get_current_label(&self) -> i32 {
        self.labels[self.upto] as i32
    }

    /// `FSTEnum.rewindPrefix`: rewind `self.upto` back to the end of the
    /// shared prefix between the enum's current position and `self.target`,
    /// so `do_seek_ceil`/`do_seek_floor`/`do_seek_exact`'s main loops only
    /// need to (re)walk the target's differing suffix.
    fn rewind_prefix(&mut self) -> Result<()> {
        if self.upto == 0 {
            let root = self.arcs[0].clone();
            if !root.is_final() && !target_has_arcs(&root) {
                // Degenerate FST that accepts no keys at all (see the same
                // guard in `advance`) -- leave `self.upto` at `0` so the
                // seek drivers below treat this as "nothing to find" without
                // dereferencing a nonexistent node.
                return Ok(());
            }
            self.upto = 1;
            let first = self.fst.read_first_target_arc(&root, &mut self.r)?;
            self.set_arc(1, first);
            return Ok(());
        }

        let current_limit = self.upto;
        self.upto = 1;
        while self.upto < current_limit && self.upto <= self.target_length + 1 {
            let cmp = self.get_current_label() - self.get_target_label();
            if cmp < 0 {
                // Seek forward: the shared prefix ends here.
                break;
            } else if cmp > 0 {
                // Seek backwards -- reset this arc to the first arc of its
                // parent node.
                let parent = self.arcs[self.upto - 1].clone();
                let first = self.fst.read_first_target_arc(&parent, &mut self.r)?;
                self.set_arc(self.upto, first);
                break;
            }
            self.upto += 1;
        }
        Ok(())
    }

    /// `FSTEnum.rollbackToLastForkThenPush`: the current node's label range
    /// is exhausted with no ceiling arc found -- back up depth by depth
    /// until a node with an unvisited next-sibling arc is found, take that
    /// sibling, then `push_first` from there (the smallest-keyed accepted
    /// descendant of that sibling). If no such node exists (`self.upto`
    /// reaches `0`), there is no key `>=` the target anywhere in the FST.
    fn rollback_to_last_fork_then_push(&mut self) -> Result<()> {
        if self.upto == 0 {
            return Ok(());
        }
        self.upto -= 1;
        loop {
            if self.upto == 0 {
                return Ok(());
            }
            let prev = self.arcs[self.upto].clone();
            if !prev.is_last() {
                let next = self.fst.read_next_arc(&prev, &mut self.r)?;
                self.set_arc(self.upto, next);
                self.push_first()?;
                return Ok(());
            }
            self.upto -= 1;
        }
    }

    /// `FSTEnum.backtrackToFloorArc`: backtracks until it finds a node whose
    /// first arc is before the (possibly-updated, after backing up a level)
    /// target label, then on that node finds the arc just before the target
    /// label and `push_last`s from there. If `self.upto` reaches `0`, there
    /// is no key `<=` the target anywhere in the FST.
    fn backtrack_to_floor_arc(&mut self) -> Result<()> {
        loop {
            let target_label = self.get_target_label();
            let parent = self.arcs[self.upto - 1].clone();
            let mut arc = self.fst.read_first_target_arc(&parent, &mut self.r)?;
            if arc.label() < target_label {
                if !arc.is_last() {
                    if arc.bytes_per_arc != 0 && arc.label() != END_LABEL {
                        match arc.node_flags {
                            ARCS_FOR_BINARY_SEARCH => self.fst.find_next_floor_arc_binary_search(
                                &mut arc,
                                target_label,
                                &mut self.r,
                            )?,
                            ARCS_FOR_DIRECT_ADDRESSING => {
                                self.fst.find_next_floor_arc_direct_addressing(
                                    &mut arc,
                                    target_label,
                                    &mut self.r,
                                )?
                            }
                            _ => {
                                debug_assert_eq!(arc.node_flags, ARCS_FOR_CONTINUOUS);
                                self.fst.find_next_floor_arc_continuous(
                                    &mut arc,
                                    target_label,
                                    &mut self.r,
                                )?
                            }
                        }
                    } else {
                        while !arc.is_last()
                            && self.fst.read_next_arc_label(&arc, &mut self.r)? < target_label
                        {
                            arc = self.fst.read_next_arc(&arc, &mut self.r)?;
                        }
                    }
                }
                self.set_arc(self.upto, arc);
                self.push_last()?;
                return Ok(());
            }
            self.upto -= 1;
            if self.upto == 0 {
                return Ok(());
            }
        }
    }

    /// Shared tail of every "found an exact match at this depth" branch
    /// across `seek_ceil`'s/`seek_exact`'s per-encoding helpers: records the
    /// matched arc/output, then either stops (if it's the synthetic
    /// `END_LABEL` acceptance arc) or descends one level and continues the
    /// seek loop from the newly matched arc's target's first arc.
    fn finish_seek_match(&mut self, arc: Arc, target_label: i32) -> Result<bool> {
        let cum_output = output_add(&self.outputs[self.upto - 1], arc.output());
        self.set_output(self.upto, cum_output);
        self.set_arc(self.upto, arc.clone());
        if target_label == END_LABEL {
            return Ok(false);
        }
        self.set_label(self.upto, arc.label() as u8);
        self.upto += 1;
        let next = self.fst.read_first_target_arc(&arc, &mut self.r)?;
        self.set_arc(self.upto, next);
        Ok(true)
    }

    /// Shared tail of every "ceiling arc found, but past an exact match" seek
    /// branch: record `arc` then `push_first` from it.
    fn finish_seek_push_first(&mut self, arc: Arc) -> Result<()> {
        self.set_arc(self.upto, arc);
        self.push_first()
    }

    /// Shared tail of every "floor arc found, but before an exact match" seek
    /// branch: record `arc` then `push_last` from it.
    fn finish_seek_push_last(&mut self, arc: Arc) -> Result<()> {
        self.set_arc(self.upto, arc);
        self.push_last()
    }

    // --- doSeekCeil ---------------------------------------------------------

    fn seek_ceil_list(&mut self, target_label: i32) -> Result<bool> {
        let arc = self.arcs[self.upto].clone();
        if arc.label() == target_label {
            self.finish_seek_match(arc, target_label)
        } else if arc.label() > target_label {
            self.finish_seek_push_first(arc)?;
            Ok(false)
        } else if arc.is_last() {
            self.rollback_to_last_fork_then_push()?;
            Ok(false)
        } else {
            let next = self.fst.read_next_arc(&arc, &mut self.r)?;
            self.set_arc(self.upto, next);
            Ok(true)
        }
    }

    fn seek_ceil_binary_search(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        let idx = self
            .fst
            .find_arc_binary_search(&arc, target_label, &mut self.r)?;
        if idx >= 0 {
            self.fst.read_arc_by_index(&mut arc, &mut self.r, idx)?;
            self.finish_seek_match(arc, target_label)
        } else {
            let idx = -1 - idx;
            if idx == arc.num_arcs {
                // Dead end: target is after the last arc.
                self.fst.read_arc_by_index(&mut arc, &mut self.r, idx - 1)?;
                debug_assert!(arc.is_last());
                self.rollback_to_last_fork_then_push()?;
            } else {
                self.fst.read_arc_by_index(&mut arc, &mut self.r, idx)?;
                self.finish_seek_push_first(arc)?;
            }
            Ok(false)
        }
    }

    fn seek_ceil_direct_addressing(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        let target_index = target_label - arc.first_label;
        if target_index >= arc.num_arcs {
            self.rollback_to_last_fork_then_push()?;
            Ok(false)
        } else {
            let clamped_index = if target_index < 0 { -1 } else { target_index };
            if clamped_index >= 0 && bit_table_is_bit_set(clamped_index, &arc, &mut self.r)? {
                let presence_index = bit_table_count_bits_up_to(clamped_index, &arc, &mut self.r)?;
                self.fst.read_arc_by_direct_addressing(
                    &mut arc,
                    &mut self.r,
                    clamped_index,
                    presence_index,
                )?;
                self.finish_seek_match(arc, target_label)
            } else {
                let ceil_index = bit_table_next_bit_set(clamped_index, &arc, &mut self.r)?;
                debug_assert!(ceil_index != -1);
                let presence_index = bit_table_count_bits_up_to(ceil_index, &arc, &mut self.r)?;
                self.fst.read_arc_by_direct_addressing(
                    &mut arc,
                    &mut self.r,
                    ceil_index,
                    presence_index,
                )?;
                self.finish_seek_push_first(arc)?;
                Ok(false)
            }
        }
    }

    fn seek_ceil_continuous(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        let target_index = target_label - arc.first_label;
        if target_index >= arc.num_arcs {
            self.rollback_to_last_fork_then_push()?;
            Ok(false)
        } else if target_index < 0 {
            self.fst.read_arc_by_continuous(&mut arc, &mut self.r, 0)?;
            debug_assert!(arc.label() > target_label);
            self.finish_seek_push_first(arc)?;
            Ok(false)
        } else {
            self.fst
                .read_arc_by_continuous(&mut arc, &mut self.r, target_index)?;
            self.finish_seek_match(arc, target_label)
        }
    }

    /// `FSTEnum.doSeekCeil`: seeks to the smallest accepted key `>=`
    /// `self.target`.
    fn do_seek_ceil(&mut self) -> Result<()> {
        self.rewind_prefix()?;
        if self.upto == 0 {
            // Degenerate, key-less FST -- see `rewind_prefix`'s guard.
            return Ok(());
        }
        loop {
            let target_label = self.get_target_label();
            let arc = self.arcs[self.upto].clone();
            let keep_going = if arc.bytes_per_arc != 0 && arc.label() != END_LABEL {
                match arc.node_flags {
                    ARCS_FOR_DIRECT_ADDRESSING => self.seek_ceil_direct_addressing(target_label)?,
                    ARCS_FOR_BINARY_SEARCH => self.seek_ceil_binary_search(target_label)?,
                    _ => {
                        debug_assert_eq!(arc.node_flags, ARCS_FOR_CONTINUOUS);
                        self.seek_ceil_continuous(target_label)?
                    }
                }
            } else {
                self.seek_ceil_list(target_label)?
            };
            if !keep_going {
                break;
            }
        }
        Ok(())
    }

    // --- doSeekFloor --------------------------------------------------------

    fn seek_floor_list(&mut self, target_label: i32) -> Result<bool> {
        let arc = self.arcs[self.upto].clone();
        if arc.label() == target_label {
            self.finish_seek_match(arc, target_label)
        } else if arc.label() > target_label {
            self.seek_floor_list_backtrack()?;
            Ok(false)
        } else if !arc.is_last() {
            if self.fst.read_next_arc_label(&arc, &mut self.r)? > target_label {
                self.finish_seek_push_last(arc)?;
                Ok(false)
            } else {
                let next = self.fst.read_next_arc(&arc, &mut self.r)?;
                self.set_arc(self.upto, next);
                Ok(true)
            }
        } else {
            self.finish_seek_push_last(arc)?;
            Ok(false)
        }
    }

    /// `doSeekFloorList`'s own inline backward walk (distinct from
    /// `backtrack_to_floor_arc`, which the array-encoding seek-floor variants
    /// use): correct for a rewound node of *any* encoding since it drives
    /// entirely through the generic `read_next_arc`/`read_next_arc_label`
    /// dispatchers rather than an encoding-specific fast path -- just
    /// potentially slower.
    fn seek_floor_list_backtrack(&mut self) -> Result<()> {
        loop {
            let target_label = self.get_target_label();
            let parent = self.arcs[self.upto - 1].clone();
            let mut arc = self.fst.read_first_target_arc(&parent, &mut self.r)?;
            if arc.label() < target_label {
                while !arc.is_last()
                    && self.fst.read_next_arc_label(&arc, &mut self.r)? < target_label
                {
                    arc = self.fst.read_next_arc(&arc, &mut self.r)?;
                }
                self.set_arc(self.upto, arc);
                self.push_last()?;
                return Ok(());
            }
            self.upto -= 1;
            if self.upto == 0 {
                return Ok(());
            }
        }
    }

    fn seek_floor_binary_search(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        let idx = self
            .fst
            .find_arc_binary_search(&arc, target_label, &mut self.r)?;
        if idx >= 0 {
            self.fst.read_arc_by_index(&mut arc, &mut self.r, idx)?;
            self.finish_seek_match(arc, target_label)
        } else if idx == -1 {
            self.backtrack_to_floor_arc()?;
            Ok(false)
        } else {
            self.fst
                .read_arc_by_index(&mut arc, &mut self.r, -2 - idx)?;
            self.finish_seek_push_last(arc)?;
            Ok(false)
        }
    }

    fn seek_floor_direct_addressing(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        let target_index = target_label - arc.first_label;
        if target_index < 0 {
            self.backtrack_to_floor_arc()?;
            Ok(false)
        } else if target_index >= arc.num_arcs {
            self.fst
                .read_last_arc_by_direct_addressing(&mut arc, &mut self.r)?;
            self.finish_seek_push_last(arc)?;
            Ok(false)
        } else if bit_table_is_bit_set(target_index, &arc, &mut self.r)? {
            let presence_index = bit_table_count_bits_up_to(target_index, &arc, &mut self.r)?;
            self.fst.read_arc_by_direct_addressing(
                &mut arc,
                &mut self.r,
                target_index,
                presence_index,
            )?;
            self.finish_seek_match(arc, target_label)
        } else {
            let floor_index = bit_table_previous_bit_set(target_index, &arc, &mut self.r)?;
            let presence_index = bit_table_count_bits_up_to(floor_index, &arc, &mut self.r)?;
            self.fst.read_arc_by_direct_addressing(
                &mut arc,
                &mut self.r,
                floor_index,
                presence_index,
            )?;
            self.finish_seek_push_last(arc)?;
            Ok(false)
        }
    }

    fn seek_floor_continuous(&mut self, target_label: i32) -> Result<bool> {
        let mut arc = self.arcs[self.upto].clone();
        let target_index = target_label - arc.first_label;
        if target_index < 0 {
            self.backtrack_to_floor_arc()?;
            Ok(false)
        } else if target_index >= arc.num_arcs {
            self.fst
                .read_last_arc_by_continuous(&mut arc, &mut self.r)?;
            self.finish_seek_push_last(arc)?;
            Ok(false)
        } else {
            self.fst
                .read_arc_by_continuous(&mut arc, &mut self.r, target_index)?;
            self.finish_seek_match(arc, target_label)
        }
    }

    /// `FSTEnum.doSeekFloor`: seeks to the largest accepted key `<=`
    /// `self.target`.
    fn do_seek_floor(&mut self) -> Result<()> {
        self.rewind_prefix()?;
        if self.upto == 0 {
            // Degenerate, key-less FST -- see `rewind_prefix`'s guard.
            return Ok(());
        }
        loop {
            let target_label = self.get_target_label();
            let arc = self.arcs[self.upto].clone();
            let keep_going = if arc.bytes_per_arc != 0 && arc.label() != END_LABEL {
                match arc.node_flags {
                    ARCS_FOR_DIRECT_ADDRESSING => {
                        self.seek_floor_direct_addressing(target_label)?
                    }
                    ARCS_FOR_BINARY_SEARCH => self.seek_floor_binary_search(target_label)?,
                    _ => {
                        debug_assert_eq!(arc.node_flags, ARCS_FOR_CONTINUOUS);
                        self.seek_floor_continuous(target_label)?
                    }
                }
            } else {
                self.seek_floor_list(target_label)?
            };
            if !keep_going {
                break;
            }
        }
        Ok(())
    }

    // --- doSeekExact ---------------------------------------------------------

    /// `FSTEnum.doSeekExact`: seeks to exactly `self.target`, returning
    /// whether it was found -- unlike `seek_ceil`/`seek_floor`, short-circuits
    /// as soon as a byte fails to match rather than continuing to search for
    /// a nearby key.
    fn do_seek_exact(&mut self) -> Result<bool> {
        self.rewind_prefix()?;
        if self.upto == 0 {
            // Degenerate, key-less FST -- see `rewind_prefix`'s guard.
            return Ok(false);
        }
        let mut arc = self.arcs[self.upto - 1].clone();
        let mut target_label = self.get_target_label();
        loop {
            match self.fst.find_target_arc(target_label, &arc, &mut self.r)? {
                None => {
                    let first = self.fst.read_first_target_arc(&arc, &mut self.r)?;
                    self.set_arc(self.upto, first);
                    return Ok(false);
                }
                Some(next_arc) => {
                    self.set_arc(self.upto, next_arc.clone());
                    let cum_output = output_add(&self.outputs[self.upto - 1], next_arc.output());
                    self.set_output(self.upto, cum_output);
                    if target_label == END_LABEL {
                        return Ok(true);
                    }
                    self.set_label(self.upto, target_label as u8);
                    self.upto += 1;
                    target_label = self.get_target_label();
                    arc = next_arc;
                }
            }
        }
    }

    /// Reads out `(key, output)` at the enum's current position, or `None`
    /// if the last seek found nothing (`self.upto == 0`) -- shared tail of
    /// `seek_ceil`/`seek_floor` (`BytesRefFSTEnum.setResult`).
    fn current_result(&self) -> Option<(Vec<u8>, Vec<u8>)> {
        if self.upto == 0 {
            None
        } else {
            Some((
                self.labels[1..self.upto].to_vec(),
                self.outputs[self.upto].clone(),
            ))
        }
    }

    /// Port of `BytesRefFSTEnum.seekCeil`: seeks to the smallest accepted
    /// key `>=` `target`, returning `(key, output)` if one exists.
    pub fn seek_ceil(&mut self, target: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        self.target = target.to_vec();
        self.target_length = target.len();
        self.do_seek_ceil()?;
        let result = self.current_result();
        self.done = result.is_none();
        Ok(result)
    }

    /// Port of `BytesRefFSTEnum.seekFloor`: seeks to the largest accepted key
    /// `<=` `target`, returning `(key, output)` if one exists.
    pub fn seek_floor(&mut self, target: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        self.target = target.to_vec();
        self.target_length = target.len();
        self.do_seek_floor()?;
        let result = self.current_result();
        self.done = result.is_none();
        Ok(result)
    }

    /// Port of `BytesRefFSTEnum.seekExact`: seeks to exactly `target`,
    /// returning `(target, output)` if the FST accepts it, else `None` --
    /// short-circuits as soon as a byte fails to match, unlike
    /// `seek_ceil`/`seek_floor`. Unlike `Fst::seek_exact` (a stateless
    /// convenience wrapper around `Fst::get`), this maintains `FstEnum`'s
    /// per-depth arc/output stack so a subsequent `next()` call can resume
    /// ordered enumeration from the found key.
    pub fn seek_exact(&mut self, target: &[u8]) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        self.target = target.to_vec();
        self.target_length = target.len();
        let found = self.do_seek_exact()?;
        if found {
            let result = self.current_result();
            self.done = result.is_none();
            Ok(result)
        } else {
            self.done = true;
            Ok(None)
        }
    }

    /// `FSTEnum.doNext` + `BytesRefFSTEnum.setResult`: advance to the next
    /// accepted key in ascending order, or `Ok(None)` once every key has
    /// been visited.
    fn advance(&mut self) -> Result<Option<(Vec<u8>, Vec<u8>)>> {
        if self.done {
            return Ok(None);
        }
        if self.upto == 0 {
            let root = self.arcs[0].clone();
            if !root.is_final() && !target_has_arcs(&root) {
                // Degenerate FST that accepts no keys at all (not even the
                // empty string) -- `read_first_target_arc` would otherwise
                // try to dereference a nonexistent node at address 0/below.
                // Real Lucene-written FSTs never hit this (every FST accepts
                // at least one key), but this port's `build_fst` allows
                // constructing one from an empty key set (see
                // `build_fst_empty_key_set_never_accepts_anything`).
                self.done = true;
                return Ok(None);
            }
            self.upto = 1;
            let first = self.fst.read_first_target_arc(&root, &mut self.r)?;
            self.set_arc(1, first);
        } else {
            loop {
                if self.arcs[self.upto].is_last() {
                    self.upto -= 1;
                    if self.upto == 0 {
                        self.done = true;
                        return Ok(None);
                    }
                } else {
                    break;
                }
            }
            let cur = self.arcs[self.upto].clone();
            let next = self.fst.read_next_arc(&cur, &mut self.r)?;
            self.set_arc(self.upto, next);
        }
        self.push_first()?;
        let key = self.labels[1..self.upto].to_vec();
        let output = self.outputs[self.upto].clone();
        Ok(Some((key, output)))
    }
}

impl Iterator for FstEnum<'_, '_> {
    type Item = Result<(Vec<u8>, Vec<u8>)>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.advance() {
            Ok(Some(pair)) => Some(Ok(pair)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

// --- FST construction (simplified FSTCompiler) -----------------------------
//
// Real Lucene's `FSTCompiler` (`org.apache.lucene.util.fst.FSTCompiler`) is an
// *incremental*, memory-bounded builder: it consumes keys one at a time in
// sorted order, keeps only a stack of `UnCompiledNode`s for the current key's
// path, "freezes" (compiles to bytes) each node as soon as it's known no
// longer to change, and de-duplicates frozen nodes via a node hash table so
// that any suffix shared by two or more keys is written to the byte store
// exactly once (suffix sharing / minimization). That incremental,
// hash-consing algorithm is a large undertaking on its own.
//
// **What's implemented here instead** is a simpler, from-scratch construction
// that is *correct* (produces bytes `Fst::read`/`Fst::get` decode correctly)
// but *not* minimal:
//
// 1. Build an ordinary in-memory trie from the full sorted `(key, output)`
//    sequence (not incremental/streaming -- the whole key set is held in
//    memory as a trie before any bytes are written).
// 2. Serialize the trie bottom-up (post-order) into the exact list-encoded
//    ("variable length arc") node/arc byte format `Fst::read`'s
//    `read_arc`/`find_target_arc` already knows how to parse -- see the
//    module doc for the field layout and `tests::build_single_key_fst`
//    (which this generalizes to a full trie with branching) for the
//    address-ordering contract those functions rely on.
//
// Deliberately **not** done, all deferred to a future slice along with real
// `FSTCompiler` itself:
// - **Incremental/streaming construction.** Real `FSTCompiler` consumes keys
//   one at a time and only ever holds the current key's path of
//   not-yet-frozen nodes in memory, freezing (and dedup-checking) each node
//   as soon as no further key can extend it. This builder instead
//   materializes the entire trie in memory upfront and serializes it
//   bottom-up in one pass. **Suffix sharing / minimization itself is done**
//   (see `NodeHash`/`build_node` below): two keys that share a *suffix* (not
//   just a prefix) -- e.g. `"cat"` and `"bat"` sharing the final `"at"` --
//   get a single shared copy of that suffix's nodes in the byte store, same
//   as real `FSTCompiler`'s node hash table would produce, just discovered by
//   hash-consing the whole tree at once rather than incrementally.
// - **Fixed-length arc nodes** (`ARCS_FOR_BINARY_SEARCH`/`_DIRECT_ADDRESSING`/
//   `_CONTINUOUS`) and the `BIT_TARGET_NEXT` adjacent-node compaction: every
//   arc here writes an explicit `vlong` target rather than relying on
//   physical adjacency, and every node is list-encoded. Larger output, but
//   still within what `Fst::read` supports (which itself doesn't decode the
//   array-node encodings -- see the module doc).
// - **Output pushing down shared prefixes.** Real Lucene's outputs
//   (`ByteSequenceOutputs`) get pushed as far toward the root as possible so
//   that arcs shared by many keys carry a common output prefix once. This
//   builder instead puts each key's *entire* output on the single arc
//   leading into its accepting node (mirroring the existing hand-built
//   `tests::build_single_key_fst` shape), relying on `Fst::get`'s
//   `output_add` accumulation to still assemble the right final bytes.
// - Only `ByteSequenceOutputs`/`INPUT_TYPE.BYTE1`/on-heap, matching the
//   reader's own scope.
//
// A node in the trie: `children` keyed by the next input byte (a `BTreeMap`
// so iteration order is the ascending label order the wire format requires),
// plus whether this node itself is an accepting state for some key and, if
// so, that key's output.
#[derive(Debug, Default)]
struct TrieNode {
    children: std::collections::BTreeMap<u8, TrieNode>,
    is_final: bool,
    final_output: Vec<u8>,
}

impl TrieNode {
    fn insert(&mut self, key: &[u8], output: Vec<u8>) {
        let mut node = self;
        for &b in key {
            node = node.children.entry(b).or_default();
        }
        node.is_final = true;
        node.final_output = output;
    }
}

fn write_vint(out: &mut Vec<u8>, mut v: i32) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u32) >> 7) as i32;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
}

fn write_vlong(out: &mut Vec<u8>, mut v: i64) {
    loop {
        let mut b = (v & 0x7f) as u8;
        v = ((v as u64) >> 7) as i64;
        if v != 0 {
            b |= 0x80;
            out.push(b);
        } else {
            out.push(b);
            break;
        }
    }
}

/// Appends one arc's bytes to `bytes` given its fields in logical (forward
/// read) order, returning that arc's address (the position `find_target_arc`
/// would `set_position` to land on this arc's flags byte). See the module
/// doc / `tests::append_arc_logical` for why the physical bytes are the
/// reverse of `logical`.
fn append_arc_logical(bytes: &mut Vec<u8>, logical: &[u8]) -> i64 {
    for &b in logical.iter().rev() {
        bytes.push(b);
    }
    (bytes.len() - 1) as i64
}

/// One arc's structural identity for node-hash-consing purposes: the exact
/// fields that determine the bytes `build_node` would write for it (label,
/// finality + final output, and -- for non-stop arcs -- the *already
/// resolved* address of the target node). Two nodes whose full ordered arc
/// list compares equal are guaranteed to serialize to byte-identical arcs
/// (see `build_node`), so this doubles as the dedup key.
type ArcSignature = (u8, bool, Vec<u8>, Option<i64>);

/// Real `FSTCompiler`'s `NodeHash` maps a *frozen* node's arc signature to
/// the address it was already written at, so that a later node with an
/// identical signature (typically a shared suffix reached via a different
/// prefix) reuses that address instead of re-serializing a duplicate copy.
/// This is the same table, keyed on the whole-tree-at-once builder's
/// equivalent of "frozen node": one whose subtree has already been fully
/// recursively resolved to final addresses.
type NodeHash = std::collections::HashMap<Vec<ArcSignature>, i64>;

/// Serializes one trie node's children into the byte store, recursing into
/// any child that itself has further children first (post-order: a child
/// node's address must be known before the arc pointing at it can write its
/// `vlong` target). Returns this node's own address (the address of the arc
/// for its *smallest*-labeled child -- see the module doc's node/arc address
/// ordering contract). Panics only if called on a node with no children
/// (callers only recurse into/start from non-empty nodes).
///
/// Before writing any new bytes for this node, checks `node_hash` for an
/// already-frozen node with the exact same ordered arc signature (same
/// labels, same finality/final-outputs, same child target addresses) and, if
/// found, returns its existing address unchanged -- this is the suffix
/// sharing / minimization step: any suffix reachable from two or more
/// different paths in the trie collapses to a single copy in the byte store.
fn build_node(node: &TrieNode, bytes: &mut Vec<u8>, node_hash: &mut NodeHash) -> i64 {
    let labels: Vec<u8> = node.children.keys().copied().collect();
    assert!(!labels.is_empty(), "build_node requires at least one arc");

    let mut child_addr: std::collections::HashMap<u8, i64> = std::collections::HashMap::new();
    for &label in &labels {
        let child = &node.children[&label];
        if !child.children.is_empty() {
            child_addr.insert(label, build_node(child, bytes, node_hash));
        }
    }

    // Signature in ascending label order (independent of the descending
    // order arcs are physically appended in below) -- just a lookup key, so
    // canonical ordering only needs to be consistent across calls.
    let signature: Vec<ArcSignature> = labels
        .iter()
        .map(|&label| {
            let child = &node.children[&label];
            let has_children = !child.children.is_empty();
            (
                label,
                child.is_final,
                child.final_output.clone(),
                if has_children {
                    Some(child_addr[&label])
                } else {
                    None
                },
            )
        })
        .collect();

    if let Some(&existing_addr) = node_hash.get(&signature) {
        return existing_addr;
    }

    // Arcs must be appended in *descending* label order: the first one
    // appended lands at the lowest address and is read *last* when scanning
    // ascending from the node's start address, so it's the one carrying
    // BIT_LAST_ARC.
    let mut node_addr = 0i64;
    for (i, &label) in labels.iter().rev().enumerate() {
        let child = &node.children[&label];

        let mut flags = 0u8;
        if i == 0 {
            flags |= BIT_LAST_ARC;
        }
        if child.is_final {
            flags |= BIT_FINAL_ARC;
            if !child.final_output.is_empty() {
                flags |= BIT_ARC_HAS_FINAL_OUTPUT;
            }
        }

        let has_children = !child.children.is_empty();
        if !has_children {
            flags |= BIT_STOP_NODE;
        }

        let mut logical = Vec::new();
        logical.push(flags);
        logical.push(label);
        if flag(flags, BIT_ARC_HAS_FINAL_OUTPUT) {
            write_vint(&mut logical, child.final_output.len() as i32);
            logical.extend_from_slice(&child.final_output);
        }
        if has_children {
            write_vlong(&mut logical, child_addr[&label]);
        }

        node_addr = append_arc_logical(bytes, &logical);
    }

    node_hash.insert(signature, node_addr);
    node_addr
}

/// Errors specific to `build_fst`'s input contract.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum BuildError {
    #[error("build_fst requires keys in strictly ascending sorted order (duplicate or out-of-order key at index {index})")]
    NotSorted { index: usize },
}

/// A minimal from-scratch equivalent of real Lucene's `FSTCompiler` for the
/// `ByteSequenceOutputs`/`INPUT_TYPE.BYTE1`/on-heap slice this module already
/// reads -- see the section doc above for exactly which of real
/// `FSTCompiler`'s behaviors (suffix sharing, output pushing, fixed-length
/// arc nodes) are and aren't reproduced.
///
/// `entries` must already be sorted in strictly ascending key order (as real
/// `FSTCompiler.add` requires of its caller) -- checked, not silently
/// tolerated.
pub fn build_fst(entries: &[(Vec<u8>, Vec<u8>)]) -> std::result::Result<Fst<'static>, BuildError> {
    for i in 1..entries.len() {
        if entries[i - 1].0 >= entries[i].0 {
            return Err(BuildError::NotSorted { index: i });
        }
    }

    let mut empty_output: Option<Vec<u8>> = None;
    let mut root = TrieNode::default();
    for (key, output) in entries {
        if key.is_empty() {
            empty_output = Some(output.clone());
        } else {
            root.insert(key, output.clone());
        }
    }

    let mut bytes = Vec::new();
    let start_node = if root.children.is_empty() {
        // No non-empty keys at all: an empty body with a start node address
        // of 0 is never dereferenced by `Fst::get` since it only calls
        // `find_target_arc` when `target_has_arcs` (`target > 0`), and 0
        // fails that check.
        0
    } else {
        let mut node_hash = NodeHash::new();
        build_node(&root, &mut bytes, &mut node_hash)
    };

    Ok(Fst {
        metadata: FstMetadata {
            input_type: InputType::Byte1,
            empty_output,
            start_node,
            version: VERSION_CURRENT,
            num_bytes: bytes.len() as i64,
        },
        bytes: FstBytes::Owned(bytes),
    })
}

/// Serializes a built `Fst` back into the exact on-disk byte layout
/// `Fst::read` parses (codec header + `FSTMetadata.save` fields + raw body),
/// so a caller can round-trip through the real, unmodified `Fst::read` entry
/// point rather than only exercising the in-memory `Fst` this module's own
/// builder happens to construct directly.
pub fn write_fst(fst: &Fst<'_>) -> Vec<u8> {
    let mut out = Vec::new();
    codec_util::write_header(&mut out, FILE_FORMAT_NAME, VERSION_CURRENT);

    match &fst.metadata.empty_output {
        Some(output) => {
            out.push(1);
            let mut reversed = output.clone();
            reversed.reverse();
            write_vint(&mut out, reversed.len() as i32);
            out.extend_from_slice(&reversed);
        }
        None => out.push(0),
    }

    out.push(match fst.metadata.input_type {
        InputType::Byte1 => 0,
        InputType::Byte2 => 1,
        InputType::Byte4 => 2,
    });
    write_vlong(&mut out, fst.metadata.start_node);
    write_vlong(&mut out, fst.metadata.num_bytes);
    out.extend_from_slice(&fst.bytes);
    out
}

// --- Typed output values (`PositiveIntOutputs`, `PairOutputs`) ------------
//
// Everything above this point stores exactly one output type on the wire:
// a raw byte sequence (`ByteSequenceOutputs`, matching `Fst::get`'s
// `Result<Option<Vec<u8>>>` and `build_fst`'s `(Vec<u8>, Vec<u8>)` entries).
// There is no generic `Outputs<T>`/arc-output abstraction underneath the
// reader or `build_node` -- both are hardcoded to `Vec<u8>` -- and this
// builder never pushes a shared output prefix up toward the root (see the
// "FST construction" section doc above: "Output pushing down shared
// prefixes ... deferred"). That absence of output-pushing is exactly what
// makes the two typed output values below simple to add correctly: each
// key's *entire* output value lives on the single arc leading to its
// accepting node (same as today), so combining "two arcs from different
// nodes get merged/shared" never has to reconcile an output already pushed
// partway down one path with a different one pushed down another -- there
// is no pushed output to reconcile. `build_node`'s existing `NodeHash`
// dedup keys on each node's exact arc signature, which includes the
// (encoded) final output bytes -- so two subtrees whose accepting arcs
// carry different typed output values naturally get different signatures
// and are never incorrectly merged; two subtrees whose typed values are
// equal (and whose deeper structure also matches) naturally *are* merged,
// which is the correct, space-saving behavior and is exercised by
// `tests::pair_outputs_shared_suffix_nodes_keep_distinct_first_components`
// below.
//
// Rather than threading a generic output type through `Fst`/`TrieNode`/
// `build_node` (a much larger change touching the reader too), typed
// support is layered *on top of* the existing byte-sequence machinery: an
// `Outputs` trait describes how to encode a typed value to/from the
// `Vec<u8>` this module already reads and writes, and `build_fst_typed`/
// `Fst::get_typed` are thin encode/decode wrappers around `build_fst`/
// `Fst::get`. This intentionally does not attempt real Lucene's separate
// `add`/`common`/`subtract` operations (`Outputs.add/common/subtract`) --
// those exist in real `FSTCompiler` purely to support output-pushing
// (finding/removing a shared output prefix so it can be hoisted onto a
// shared arc); since this builder never pushes outputs, there is no shared
// prefix to find or subtract, so those operations would have no caller and
// are omitted rather than stubbed out dishonestly.

/// A typed FST output value, encoded to/from the raw `Vec<u8>` this module's
/// `Fst`/`build_fst` already store on the wire. `Self::Value` mirrors real
/// Lucene's `T` type parameter of `Outputs<T>`; `Self` itself is a
/// zero-sized marker type selecting *which* codec applies (matching real
/// Lucene's `Outputs<T>` being a singleton descriptor object distinct from
/// the values `T` it produces).
pub trait Outputs {
    /// The typed output value this codec encodes/decodes, e.g. `i64` for
    /// `PositiveIntOutputs` or `Vec<u8>` for `ByteSequenceOutputs`.
    type Value: Clone + PartialEq;

    /// `Outputs.getNoOutput()`: the identity/absent value (real Lucene's
    /// `NO_OUTPUT` sentinel), e.g. `0i64` or the empty byte sequence. Must
    /// encode to the empty `Vec<u8>` so it lines up with this module's
    /// existing "empty output bytes -> `BIT_ARC_HAS_(FINAL_)OUTPUT` unset"
    /// convention in `build_node`/`read_arc` -- i.e. a key whose typed
    /// output is `zero()` costs no extra output bytes on the wire, same as
    /// today's plain `Vec<u8>` empty-output keys.
    fn zero() -> Self::Value;

    /// Encode `value` to the byte sequence `build_fst`/`Fst::get` store and
    /// return respectively.
    fn encode(value: &Self::Value) -> Vec<u8>;

    /// Decode a byte sequence previously produced by `encode` (only ever
    /// called on bytes this same `Outputs` impl produced, never validated
    /// against a foreign encoding).
    fn decode(bytes: &[u8]) -> Self::Value;
}

/// Forward (non-reversed) `vint` reader over a plain slice -- the encode
/// direction's counterpart to `write_vint` (also forward) and distinct from
/// `BytesReader::read_vint`, which walks its cursor *backwards* to match the
/// on-disk FST body's own reverse-read convention. Typed output *values*
/// (this section) are encoded/decoded independently of that body layout, in
/// ordinary forward byte order, since they're just the payload `build_node`
/// copies verbatim into an arc's final-output bytes. Returns `(value,
/// bytes_consumed)`.
fn read_vint_forward(bytes: &[u8]) -> (i32, usize) {
    let mut idx = 0usize;
    let mut b = bytes[idx];
    idx += 1;
    let mut v = (b & 0x7f) as i32;
    let mut shift = 7;
    while b & 0x80 != 0 {
        b = bytes[idx];
        idx += 1;
        v |= ((b & 0x7f) as i32) << shift;
        shift += 7;
    }
    (v, idx)
}

/// Forward `vlong` reader, `read_vint_forward`'s 64-bit counterpart (used by
/// `PositiveIntOutputs::decode`).
fn read_vlong_forward(bytes: &[u8]) -> (i64, usize) {
    let mut idx = 0usize;
    let mut b = bytes[idx];
    idx += 1;
    let mut v = (b & 0x7f) as i64;
    let mut shift = 7;
    while b & 0x80 != 0 {
        b = bytes[idx];
        idx += 1;
        v |= ((b & 0x7f) as i64) << shift;
        shift += 7;
    }
    (v, idx)
}

/// Port of `PositiveIntOutputs`: an FST output type whose value is a single
/// non-negative `i64` (real Lucene rejects negative values; this port
/// mirrors that with a `debug_assert`, since silently wrapping/rejecting in
/// release builds isn't this slice's concern and every current caller is a
/// test). `NO_OUTPUT` is `0`, matching real Lucene, which is why it encodes
/// to the empty byte sequence (see `Outputs::zero`'s doc comment).
pub struct PositiveIntOutputs;

impl Outputs for PositiveIntOutputs {
    type Value = i64;

    fn zero() -> i64 {
        0
    }

    fn encode(value: &i64) -> Vec<u8> {
        debug_assert!(
            *value >= 0,
            "PositiveIntOutputs requires non-negative values, got {value}"
        );
        if *value == 0 {
            return Vec::new();
        }
        let mut out = Vec::new();
        write_vlong(&mut out, *value);
        out
    }

    fn decode(bytes: &[u8]) -> i64 {
        if bytes.is_empty() {
            return 0;
        }
        let (value, consumed) = read_vlong_forward(bytes);
        debug_assert_eq!(
            consumed,
            bytes.len(),
            "PositiveIntOutputs::decode: trailing bytes"
        );
        value
    }
}

/// Port of `ByteSequenceOutputs`, expressed as an `Outputs` impl: the
/// identity codec, matching what `build_fst`/`Fst::get` already do directly
/// for plain `(Vec<u8>, Vec<u8>)` callers. Exists so `PairOutputs<A, B>` can
/// use it as either component type (e.g. `PairOutputs<PositiveIntOutputs,
/// ByteSequenceOutputs>`, the shape real Lucene's synonym/suggest
/// infrastructure uses for `PairOutputs<Long, BytesRef>`).
pub struct ByteSequenceOutputs;

impl Outputs for ByteSequenceOutputs {
    type Value = Vec<u8>;

    fn zero() -> Vec<u8> {
        Vec::new()
    }

    fn encode(value: &Vec<u8>) -> Vec<u8> {
        value.clone()
    }

    fn decode(bytes: &[u8]) -> Vec<u8> {
        bytes.to_vec()
    }
}

/// Port of `PairOutputs.Pair<A, B>`: a value combining two independent
/// component output values.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Pair<A, B> {
    pub first: A,
    pub second: B,
}

/// Port of `PairOutputs<A, B>`: combines two independent `Outputs` codecs
/// (`A`, `B`) into one `Outputs` impl over `Pair<A::Value, B::Value>`, so a
/// single FST can map each key to both an `A`-typed and a `B`-typed output
/// simultaneously (e.g. a weight *and* a payload per key, as real Lucene's
/// synonym/suggest FSTs do with `PairOutputs<Long, BytesRef>`).
///
/// Encoding: if both components encode to the empty byte sequence (i.e. the
/// pair *is* `zero()`), the whole pair also encodes to the empty byte
/// sequence -- preserving the "no output bytes for a no-op key" convention
/// `Outputs::zero`'s doc comment describes, rather than always paying a
/// length-prefix byte even for an all-zero pair. Otherwise: the first
/// component's encoded length (`vint`), then its encoded bytes, then the
/// second component's encoded bytes filling the remainder (no length prefix
/// needed for the second component since it's always everything left).
pub struct PairOutputs<A, B>(std::marker::PhantomData<(A, B)>);

impl<A: Outputs, B: Outputs> Outputs for PairOutputs<A, B> {
    type Value = Pair<A::Value, B::Value>;

    fn zero() -> Self::Value {
        Pair {
            first: A::zero(),
            second: B::zero(),
        }
    }

    fn encode(value: &Self::Value) -> Vec<u8> {
        let first = A::encode(&value.first);
        let second = B::encode(&value.second);
        if first.is_empty() && second.is_empty() {
            return Vec::new();
        }
        let mut out = Vec::new();
        write_vint(&mut out, first.len() as i32);
        out.extend_from_slice(&first);
        out.extend_from_slice(&second);
        out
    }

    fn decode(bytes: &[u8]) -> Self::Value {
        if bytes.is_empty() {
            return Self::zero();
        }
        let (first_len, header_len) = read_vint_forward(bytes);
        let first_len = first_len as usize;
        let first_bytes = &bytes[header_len..header_len + first_len];
        let second_bytes = &bytes[header_len + first_len..];
        Pair {
            first: A::decode(first_bytes),
            second: B::decode(second_bytes),
        }
    }
}

/// Typed equivalent of `build_fst`: encodes each entry's `O::Value` output
/// via `O::encode` before delegating to the existing byte-sequence builder.
/// `entries` must be sorted in strictly ascending key order, exactly like
/// `build_fst`.
pub fn build_fst_typed<O: Outputs>(
    entries: &[(Vec<u8>, O::Value)],
) -> std::result::Result<Fst<'static>, BuildError> {
    let byte_entries: Vec<(Vec<u8>, Vec<u8>)> = entries
        .iter()
        .map(|(key, value)| (key.clone(), O::encode(value)))
        .collect();
    build_fst(&byte_entries)
}

impl<'a> Fst<'a> {
    /// Typed equivalent of `Fst::get`: looks up `key` and decodes its
    /// accumulated output bytes via `O::decode`.
    pub fn get_typed<O: Outputs>(&self, key: &[u8]) -> Result<Option<O::Value>> {
        Ok(self.get(key)?.map(|bytes| O::decode(&bytes)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- Hand-built FST bytes ---------------------------------------------
    //
    // These unit tests build minimal, hand-crafted FST bodies (list-encoded
    // nodes only) to exercise `Fst::get`'s own boundary/error handling --
    // the differential tests in `tests/fst_fixtures.rs` cover real,
    // Lucene-written FSTs (with genuine arc sharing); see the
    // `test-coverage` skill on why both layers are needed.

    fn write_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_vlong(out: &mut Vec<u8>, mut v: i64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u64) >> 7) as i64;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    /// Builds a single-key FST body (list-encoded) accepting exactly `key`,
    /// with final output `output`, using the classic "chain of arcs, each
    /// node has exactly one arc, output lives on the arc immediately
    /// preceding acceptance" shape. Bytes are laid out in *forward* address
    /// order (index 0 = lowest address) since a `BytesReader` seeks directly
    /// by absolute position; only reading direction is reversed.
    ///
    /// Node for byte `key[i]` sits at a higher address than node for
    /// `key[i+1]`; the last node (accepting) is the lowest-addressed / first
    /// bytes in the array.
    /// Appends one arc's bytes to `bytes`, given its fields in *logical*
    /// (forward) read order (`flags`, `label`, ...). Because `BytesReader`
    /// reads by decrementing its position (`ReverseBytesReader`), the
    /// physical array must hold each field's bytes at *decreasing*
    /// addresses relative to reading order -- i.e. the logical byte
    /// sequence appended here must be pushed in reverse, so that the
    /// address of `logical[0]` (the flags byte, returned as this arc's
    /// node address) ends up higher than `logical[1]`, and so on. This
    /// mirrors how real `FSTCompiler` grows its byte store from high
    /// addresses down to low ones.
    fn append_arc_logical(bytes: &mut Vec<u8>, logical: &[u8]) -> i64 {
        for &b in logical.iter().rev() {
            bytes.push(b);
        }
        (bytes.len() - 1) as i64
    }

    /// Builds a minimal, list-encoded (no array/direct-addressing nodes)
    /// FST body accepting exactly `key` with final output `output`: one
    /// node per key byte, each with exactly one arc (`BIT_LAST_ARC`),
    /// explicit (non-`BIT_TARGET_NEXT`) `vlong` targets throughout, and the
    /// arc for the last byte carrying `BIT_STOP_NODE | BIT_FINAL_ARC` (plus
    /// the final output, if non-empty) instead of a target.
    ///
    /// Nodes are appended starting from the *last* key byte, so each
    /// non-final node's target (the already-appended next node's address)
    /// is known by the time it's written.
    fn build_single_key_fst(key: &[u8], output: &[u8]) -> (Vec<u8>, i64) {
        let mut bytes: Vec<u8> = Vec::new();
        let mut next_target: Option<i64> = None;
        let mut addrs = vec![0i64; key.len()];

        for i in (0..key.len()).rev() {
            let b = key[i];
            let is_last_byte = i == key.len() - 1;

            let mut logical = Vec::new();
            let mut flags = BIT_LAST_ARC;
            if is_last_byte {
                flags |= BIT_FINAL_ARC | BIT_STOP_NODE;
                if !output.is_empty() {
                    flags |= BIT_ARC_HAS_FINAL_OUTPUT;
                }
                logical.push(flags);
                logical.push(b);
                if !output.is_empty() {
                    write_vint(&mut logical, output.len() as i32);
                    logical.extend_from_slice(output);
                }
            } else {
                logical.push(flags);
                logical.push(b);
                write_vlong(
                    &mut logical,
                    next_target.expect("target known from reverse pass"),
                );
            }

            let addr = append_arc_logical(&mut bytes, &logical);
            addrs[i] = addr;
            next_target = Some(addr);
        }

        let start_node = addrs[0];
        (bytes, start_node)
    }

    fn fst_from_body(
        bytes: Vec<u8>,
        start_node: i64,
        input_type: InputType,
        empty_output: Option<Vec<u8>>,
    ) -> Fst<'static> {
        Fst {
            metadata: FstMetadata {
                input_type,
                empty_output,
                start_node,
                version: VERSION_CURRENT,
                num_bytes: bytes.len() as i64,
            },
            bytes: FstBytes::Owned(bytes),
        }
    }

    #[test]
    fn single_key_found_with_output() {
        let (bytes, start) = build_single_key_fst(b"cat", b"1");
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(b"cat").unwrap(), Some(b"1".to_vec()));
    }

    #[test]
    fn single_key_not_present_wrong_label() {
        let (bytes, start) = build_single_key_fst(b"cat", b"1");
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(b"dog").unwrap(), None);
    }

    #[test]
    fn single_key_not_present_prefix_of_key() {
        let (bytes, start) = build_single_key_fst(b"cat", b"1");
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        // "ca" is a proper prefix that isn't itself accepted.
        assert_eq!(fst.get(b"ca").unwrap(), None);
    }

    #[test]
    fn single_key_not_present_extends_past_key() {
        let (bytes, start) = build_single_key_fst(b"cat", b"1");
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        // "cats" walks past the accepting node: the last arc is BIT_STOP_NODE
        // with no outgoing arcs, so target_has_arcs is false.
        assert_eq!(fst.get(b"cats").unwrap(), None);
    }

    #[test]
    fn empty_key_accepted_via_empty_output() {
        let (bytes, start) = build_single_key_fst(b"x", b"ignored");
        let fst = fst_from_body(bytes, start, InputType::Byte1, Some(b"empty-out".to_vec()));
        assert_eq!(fst.get(b"").unwrap(), Some(b"empty-out".to_vec()));
    }

    #[test]
    fn empty_key_rejected_when_fst_does_not_accept_it() {
        let (bytes, start) = build_single_key_fst(b"x", b"1");
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(b"").unwrap(), None);
    }

    #[test]
    fn empty_output_with_zero_length_bytes_still_accepts_empty_key() {
        let (bytes, start) = build_single_key_fst(b"x", b"1");
        // emptyOutput != None but zero-length: still "accepts empty string",
        // just with a NO_OUTPUT (empty) final output -- mirrors
        // `getFirstArc`'s `emptyOutput != NO_OUTPUT` check.
        let fst = fst_from_body(bytes, start, InputType::Byte1, Some(Vec::new()));
        assert_eq!(fst.get(b"").unwrap(), Some(Vec::new()));
    }

    #[test]
    fn non_byte1_input_type_is_unsupported_for_get() {
        let (bytes, start) = build_single_key_fst(b"x", b"1");
        let fst = fst_from_body(bytes, start, InputType::Byte2, None);
        assert!(matches!(fst.get(b"x"), Err(Error::Unsupported(_))));
    }

    // --- `ARCS_FOR_DIRECT_ADDRESSING` fixed-length-arc nodes ---------------
    //
    // Hand-built (not real-Lucene-written) node in this encoding, to pin down
    // `find_target_arc_direct_addressing`/`bit_table_*`'s byte-level field
    // layout before/alongside the real-fixture differential test in
    // `tests/fst_direct_addressing_fixtures.rs`.

    /// Builds a single root node spanning the label range
    /// `[first_label, first_label + num_arcs)`, encoded as
    /// `ARCS_FOR_DIRECT_ADDRESSING`: a presence bit-table (one bit per label
    /// in the range) followed by `first_label`, then one fixed-size arc slot
    /// per *present* label only (ascending order), each a final, accepting
    /// stop node whose 1-byte output is given by `present`. `present`'s
    /// labels must all fall within the range and need not be pre-sorted.
    /// Returns `(body_bytes, start_node_address)`.
    fn build_direct_addressing_node(
        first_label: u8,
        num_arcs: i32,
        present: &[(u8, u8)],
    ) -> (Vec<u8>, i64) {
        let num_presence_bytes = num_presence_bytes(num_arcs) as usize;
        let mut presence = vec![0u8; num_presence_bytes];
        for &(label, _) in present {
            let idx = (label - first_label) as usize;
            presence[idx / 8] |= 1 << (idx % 8);
        }

        let mut sorted_present = present.to_vec();
        sorted_present.sort_by_key(|(l, _)| *l);
        let num_present = sorted_present.len();

        let mut logical_arcs: Vec<Vec<u8>> = Vec::with_capacity(num_present);
        for (i, &(_label, output)) in sorted_present.iter().enumerate() {
            let mut logical = Vec::new();
            let mut flags = BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT;
            if i == num_present - 1 {
                flags |= BIT_LAST_ARC;
            }
            logical.push(flags);
            // No label byte: direct-addressing arcs never store the label
            // explicitly, it's implied by position in the range.
            write_vint(&mut logical, 1);
            logical.push(output);
            logical_arcs.push(logical);
        }
        let bytes_per_arc = logical_arcs.iter().map(|a| a.len()).max().unwrap_or(0);

        let mut logical = vec![ARCS_FOR_DIRECT_ADDRESSING];
        write_vint(&mut logical, num_arcs);
        write_vint(&mut logical, bytes_per_arc as i32);
        logical.extend_from_slice(&presence);
        logical.push(first_label);
        for arc in &logical_arcs {
            let mut padded = arc.clone();
            padded.resize(bytes_per_arc, 0u8);
            logical.extend_from_slice(&padded);
        }

        let mut bytes = Vec::new();
        let node_addr = append_arc_logical(&mut bytes, &logical);
        (bytes, node_addr)
    }

    #[test]
    fn direct_addressing_node_finds_every_present_label() {
        let first_label = b'a';
        let num_arcs = (b'z' - b'a' + 1) as i32;
        let present = [
            (b'a', 1u8),
            (b'c', 2u8),
            (b'f', 3u8),
            (b'm', 4u8),
            (b'z', 5u8),
        ];
        let (bytes, start) = build_direct_addressing_node(first_label, num_arcs, &present);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        for (label, output) in present {
            assert_eq!(
                fst.get(&[label]).unwrap(),
                Some(vec![output]),
                "label {label} should resolve to {output}"
            );
        }
    }

    #[test]
    fn direct_addressing_node_rejects_absent_labels_in_and_around_range() {
        let first_label = b'a';
        let num_arcs = (b'z' - b'a' + 1) as i32;
        let present = [
            (b'a', 1u8),
            (b'c', 2u8),
            (b'f', 3u8),
            (b'm', 4u8),
            (b'z', 5u8),
        ];
        let (bytes, start) = build_direct_addressing_node(first_label, num_arcs, &present);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        // In-range but not present (bit clear), plus strictly before/after
        // the label range entirely.
        for absent in [b'b', b'd', b'g', b'n', b'y', 0u8, 0xffu8] {
            assert_eq!(
                fst.get(&[absent]).unwrap(),
                None,
                "label {absent} should be absent"
            );
        }
    }

    #[test]
    fn direct_addressing_node_enumerates_every_present_label_in_ascending_order() {
        let first_label = b'a';
        let num_arcs = (b'z' - b'a' + 1) as i32;
        let present = [
            (b'm', 4u8),
            (b'a', 1u8),
            (b'z', 5u8),
            (b'c', 2u8),
            (b'f', 3u8),
        ];
        let (bytes, start) = build_direct_addressing_node(first_label, num_arcs, &present);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        let mut expected: Vec<(Vec<u8>, Vec<u8>)> =
            present.iter().map(|&(l, o)| (vec![l], vec![o])).collect();
        expected.sort();
        let got: Vec<(Vec<u8>, Vec<u8>)> = fst.iter().unwrap().collect::<Result<_>>().unwrap();
        assert_eq!(got, expected);
    }

    // --- `ARCS_FOR_CONTINUOUS` fixed-length-arc nodes ----------------------
    //
    // Hand-built (not real-Lucene-written) node in this encoding, to pin down
    // `find_target_arc_continuous`'s byte-level field layout before/alongside
    // the real-fixture differential test in
    // `tests/fst_continuous_fixtures.rs`.

    /// Builds a single root node spanning the label range
    /// `[first_label, first_label + labels.len())`, encoded as
    /// `ARCS_FOR_CONTINUOUS`: no presence bit-table and no per-arc label byte
    /// at all (every label in the range is present by construction), just
    /// `first_label` followed by one fixed-size arc slot per label in
    /// ascending order, each a final, accepting stop node whose 1-byte output
    /// is `outputs[i]`. Returns `(body_bytes, start_node_address)`.
    fn build_continuous_node(first_label: u8, outputs: &[u8]) -> (Vec<u8>, i64) {
        let num_arcs = outputs.len();
        let mut logical_arcs: Vec<Vec<u8>> = Vec::with_capacity(num_arcs);
        for (i, &output) in outputs.iter().enumerate() {
            let mut logical = Vec::new();
            let mut flags = BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT;
            if i == num_arcs - 1 {
                flags |= BIT_LAST_ARC;
            }
            logical.push(flags);
            // No label byte: continuous arcs never store the label
            // explicitly, it's implied by position in the range.
            write_vint(&mut logical, 1);
            logical.push(output);
            logical_arcs.push(logical);
        }
        let bytes_per_arc = logical_arcs.iter().map(|a| a.len()).max().unwrap_or(0);

        let mut logical = vec![ARCS_FOR_CONTINUOUS];
        write_vint(&mut logical, num_arcs as i32);
        write_vint(&mut logical, bytes_per_arc as i32);
        logical.push(first_label);
        for arc in &logical_arcs {
            let mut padded = arc.clone();
            padded.resize(bytes_per_arc, 0u8);
            logical.extend_from_slice(&padded);
        }

        let mut bytes = Vec::new();
        let node_addr = append_arc_logical(&mut bytes, &logical);
        (bytes, node_addr)
    }

    #[test]
    fn continuous_node_finds_every_present_label() {
        let first_label = b'a';
        let outputs = [1u8, 2, 3, 4, 5];
        let (bytes, start) = build_continuous_node(first_label, &outputs);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        for (i, &output) in outputs.iter().enumerate() {
            let label = first_label + i as u8;
            assert_eq!(
                fst.get(&[label]).unwrap(),
                Some(vec![output]),
                "label {label} should resolve to {output}"
            );
        }
    }

    #[test]
    fn continuous_node_rejects_labels_outside_the_range() {
        let first_label = b'a';
        let outputs = [1u8, 2, 3, 4, 5]; // Covers 'a'..='e'.
        let (bytes, start) = build_continuous_node(first_label, &outputs);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        for absent in [b'`', b'f', b'z', 0u8, 0xffu8] {
            assert_eq!(
                fst.get(&[absent]).unwrap(),
                None,
                "label {absent} should be absent"
            );
        }
    }

    #[test]
    fn continuous_node_enumerates_every_label_in_ascending_order() {
        let first_label = b'a';
        let outputs = [1u8, 2, 3, 4, 5];
        let (bytes, start) = build_continuous_node(first_label, &outputs);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        let expected: Vec<(Vec<u8>, Vec<u8>)> = outputs
            .iter()
            .enumerate()
            .map(|(i, &o)| (vec![first_label + i as u8], vec![o]))
            .collect();
        let got: Vec<(Vec<u8>, Vec<u8>)> = fst.iter().unwrap().collect::<Result<_>>().unwrap();
        assert_eq!(got, expected);
    }

    // --- `ARCS_FOR_BINARY_SEARCH` fixed-length-arc nodes ------------------
    //
    // Hand-built (not real-Lucene-written) node in this encoding, to pin
    // down `find_target_arc_binary_search`'s byte-level field layout and
    // binary-search bounds before/alongside the real-fixture differential
    // test in `tests/fst_fixtures.rs`.

    /// Builds a single root node with `labels.len()` arcs (ascending,
    /// distinct byte labels) encoded as `ARCS_FOR_BINARY_SEARCH`: each arc is
    /// a fixed-size slot of `bytes_per_arc` bytes (padded with trailing zero
    /// bytes after its real content -- real Lucene pads the same way so
    /// every slot can be addressed via `posArcsStart - bytesPerArc * idx`).
    /// Every arc is a final, accepting stop node whose 1-byte output is
    /// `outputs[i]`. Returns `(body_bytes, start_node_address)`.
    fn build_binary_search_node(labels: &[u8], outputs: &[u8]) -> (Vec<u8>, i64) {
        assert_eq!(labels.len(), outputs.len());
        let num_arcs = labels.len();

        // Build each arc's logical (forward-read) bytes first, to find the
        // max length -> bytes_per_arc (every slot is padded to this size).
        let mut logical_arcs: Vec<Vec<u8>> = Vec::with_capacity(num_arcs);
        for (i, &label) in labels.iter().enumerate() {
            let mut logical = Vec::new();
            let mut flags = BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT;
            if i == num_arcs - 1 {
                // Only the highest-labeled arc (the last slot, at the
                // lowest address) is actually the node's last arc --
                // `is_last()` must be false for every other slot so
                // enumeration (`FstEnum`) knows to keep advancing through
                // the node instead of stopping after the first arc it
                // reads. (Real Lucene-written binary-search nodes get this
                // right by construction; this hand-built fixture previously
                // set the bit on every slot, which `find_target_arc`'s
                // direct binary-search jump never noticed but `FstEnum`'s
                // incremental walk does.)
                flags |= BIT_LAST_ARC;
            }
            logical.push(flags);
            logical.push(label);
            write_vint(&mut logical, 1);
            logical.push(outputs[i]);
            logical_arcs.push(logical);
        }
        let bytes_per_arc = logical_arcs.iter().map(|a| a.len()).max().unwrap();

        // Build the *entire* node (header + every fixed-size slot) as one
        // logical (forward-read-order) byte sequence, then push it via
        // `append_arc_logical`'s single whole-blob reversal -- this keeps
        // every field's relative addressing (header bytes, then slot 0,
        // slot 1, ...) consistent with the reverse `BytesReader`'s
        // decreasing-position reads, exactly the way a real node's bytes
        // are laid out. `append_arc_logical` returns the flags byte's
        // address (`node_addr`), matching what `find_target_arc` seeks to.
        let mut logical = vec![ARCS_FOR_BINARY_SEARCH];
        write_vint(&mut logical, num_arcs as i32);
        write_vint(&mut logical, bytes_per_arc as i32);
        for arc in &logical_arcs {
            let mut padded = arc.clone();
            padded.resize(bytes_per_arc, 0u8);
            logical.extend_from_slice(&padded);
        }

        let mut bytes = Vec::new();
        let node_addr = append_arc_logical(&mut bytes, &logical);
        (bytes, node_addr)
    }

    #[test]
    fn binary_search_node_finds_every_label() {
        let labels = [b'a', b'c', b'f', b'm', b'z'];
        let outputs = [1u8, 2, 3, 4, 5];
        let (bytes, start) = build_binary_search_node(&labels, &outputs);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        for (label, output) in labels.iter().zip(outputs.iter()) {
            assert_eq!(
                fst.get(&[*label]).unwrap(),
                Some(vec![*output]),
                "label {label} should resolve to {output}"
            );
        }
    }

    #[test]
    fn binary_search_node_rejects_absent_labels() {
        let labels = [b'a', b'c', b'f', b'm', b'z'];
        let outputs = [1u8, 2, 3, 4, 5];
        let (bytes, start) = build_binary_search_node(&labels, &outputs);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        // Before the first label, between two labels, and after the last.
        for absent in [b'0', b'b', b'd', b'g', b'n', 0xffu8] {
            assert_eq!(fst.get(&[absent]).unwrap(), None);
        }
    }

    #[test]
    fn read_rejects_bad_input_type_byte() {
        let mut meta = Vec::new();
        codec_util::write_header(&mut meta, FILE_FORMAT_NAME, VERSION_CURRENT);
        meta.push(0); // no empty output
        meta.push(9); // invalid input type marker
        let mut input = SliceInput::new(&meta);
        assert!(matches!(Fst::read(&mut input), Err(Error::Corrupt(_))));
    }

    #[test]
    fn read_rejects_wrong_codec_header() {
        let mut meta = Vec::new();
        codec_util::write_header(&mut meta, "NotFST", VERSION_CURRENT);
        let mut input = SliceInput::new(&meta);
        assert!(matches!(Fst::read(&mut input), Err(Error::Store(_))));
    }

    #[test]
    fn read_metadata_round_trips_through_real_header_and_body() {
        // Full end-to-end `Fst::read` (metadata + body via one cursor),
        // independent of the real-Lucene differential fixtures: hand-builds
        // the exact byte layout `FSTMetadata.save` produces for a
        // no-empty-output, BYTE1, single-key FST.
        let (body, start_node) = build_single_key_fst(b"ab", b"42");

        let mut file = Vec::new();
        codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
        file.push(0); // no empty output
        file.push(0); // INPUT_TYPE.BYTE1
        write_vlong(&mut file, start_node);
        write_vlong(&mut file, body.len() as i64);
        file.extend_from_slice(&body);

        let mut input = SliceInput::new(&file);
        let fst = Fst::read(&mut input).unwrap();
        assert_eq!(fst.metadata().input_type, InputType::Byte1);
        assert_eq!(fst.metadata().start_node, start_node);
        assert_eq!(fst.metadata().num_bytes, body.len() as i64);
        assert_eq!(fst.get(b"ab").unwrap(), Some(b"42".to_vec()));
        assert_eq!(fst.get(b"ac").unwrap(), None);
    }

    #[test]
    fn read_rejects_truncated_body() {
        // A numBytes claiming more body than actually follows must fail
        // (as an Eof from the shared SliceInput), not silently succeed with
        // a short/garbage body.
        let mut file = Vec::new();
        codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
        file.push(0);
        file.push(0);
        write_vlong(&mut file, 0);
        write_vlong(&mut file, 1000); // claims 1000 body bytes, none follow
        let mut input = SliceInput::new(&file);
        assert!(matches!(Fst::read(&mut input), Err(Error::Store(_))));
    }

    #[test]
    fn read_borrowed_rejects_truncated_body_same_as_read() {
        // Same malformed bytes as `read_rejects_truncated_body`, through
        // `read_borrowed` -- `SliceInput::slice`'s bounds check must reject
        // this the same way `read_bytes` does for the owned path, not
        // silently return a short/garbage slice.
        let mut file = Vec::new();
        codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
        file.push(0);
        file.push(0);
        write_vlong(&mut file, 0);
        write_vlong(&mut file, 1000); // claims 1000 body bytes, none follow
        let mut input = SliceInput::new(&file);
        assert!(matches!(
            Fst::read_borrowed(&mut input),
            Err(Error::Store(_))
        ));
    }

    #[test]
    fn read_rejects_negative_num_bytes() {
        // `readVLong`'s bit-shifting can decode a hand-crafted (never
        // written by a real, well-behaved encoder) byte sequence as a
        // negative `i64`; `Fst::read` must reject that outright via its own
        // explicit guard rather than trying `vec![0u8; num_bytes as usize]`
        // with a negative-cast-to-usize length.
        let mut file = Vec::new();
        codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
        file.push(0);
        file.push(0);
        write_vlong(&mut file, 0);
        write_vlong(&mut file, -1); // decodes to a negative numBytes
        let mut input = SliceInput::new(&file);
        assert!(matches!(Fst::read(&mut input), Err(Error::Corrupt(_))));
    }

    #[test]
    fn read_accepts_empty_string_via_empty_output() {
        // Full `Fst::read` path for the `acceptsEmpty == 1` branch: the
        // empty-string output is stored length-prefixed and *reversed* on
        // disk (`FSTMetadata.save`), matching what a real Lucene FST with a
        // key equal to the empty string would produce.
        let (body, start_node) = build_single_key_fst(b"x", b"ignored");
        let empty_output = b"root-output".to_vec();
        let mut reversed = empty_output.clone();
        reversed.reverse();

        let mut file = Vec::new();
        codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
        file.push(1); // accepts empty string
        write_vint(&mut file, reversed.len() as i32);
        file.extend_from_slice(&reversed);
        file.push(0); // INPUT_TYPE.BYTE1
        write_vlong(&mut file, start_node);
        write_vlong(&mut file, body.len() as i64);
        file.extend_from_slice(&body);

        let mut input = SliceInput::new(&file);
        let fst = Fst::read(&mut input).unwrap();
        assert_eq!(fst.metadata().empty_output, Some(empty_output.clone()));
        assert_eq!(fst.get(b"").unwrap(), Some(empty_output));
    }

    #[test]
    fn read_parses_byte2_and_byte4_input_types() {
        for (marker, expected) in [(1u8, InputType::Byte2), (2u8, InputType::Byte4)] {
            let mut file = Vec::new();
            codec_util::write_header(&mut file, FILE_FORMAT_NAME, VERSION_CURRENT);
            file.push(0); // no empty output
            file.push(marker);
            write_vlong(&mut file, 0); // startNode
            write_vlong(&mut file, 0); // numBytes: empty body
            let mut input = SliceInput::new(&file);
            let fst = Fst::read(&mut input).unwrap();
            assert_eq!(fst.metadata().input_type, expected);
        }
    }

    #[test]
    fn get_errors_on_out_of_range_start_node() {
        // A start_node address beyond the body's bounds must surface as an
        // error from the reverse byte cursor, not a panic or silent
        // misdecode -- exercises `BytesReader::read_byte`'s own bounds
        // check directly (real Lucene-written FSTs never do this; this is
        // this decoder's own defensive boundary).
        let fst = fst_from_body(vec![1, 2, 3], 100, InputType::Byte1, None);
        assert!(matches!(fst.get(b"a"), Err(Error::Corrupt(_))));
    }

    #[test]
    fn multi_byte_vint_output_length_round_trips() {
        // Forces `read_vint`'s (and the test-only `write_vint`'s) multi-byte
        // continuation branch: an output of 200 bytes needs a 2-byte vint
        // length prefix (127 is the largest 1-byte vint).
        let big_output = vec![7u8; 200];
        let (bytes, start) = build_single_key_fst(b"k", &big_output);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(b"k").unwrap(), Some(big_output));
    }

    #[test]
    fn multi_byte_vlong_target_round_trips() {
        // Forces `read_vlong`'s multi-byte continuation branch for an
        // explicit arc target: a long enough key chain pushes early node
        // addresses well past 127 (the largest 1-byte vlong value).
        let key: Vec<u8> = (0..100u8).collect(); // 100 distinct bytes, deep chain
        let (bytes, start) = build_single_key_fst(&key, b"end");
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(&key).unwrap(), Some(b"end".to_vec()));
        assert!(bytes_len_forces_multi_byte_vlong(&fst));
    }

    /// Sanity check that the previous test's key is actually long enough to
    /// push at least one node address past the 1-byte vlong threshold (127)
    /// -- guards against the test silently stopping being meaningful if the
    /// key length above is ever shortened.
    fn bytes_len_forces_multi_byte_vlong(fst: &Fst) -> bool {
        fst.bytes.len() > 127
    }

    // --- `output_add` (free function) ---------------------------------

    #[test]
    fn output_add_covers_all_branches() {
        assert_eq!(output_add(&[], &[]), Vec::<u8>::new());
        assert_eq!(output_add(&[], &[1, 2]), vec![1, 2]);
        assert_eq!(output_add(&[1, 2], &[]), vec![1, 2]);
        assert_eq!(output_add(&[1, 2], &[3, 4]), vec![1, 2, 3, 4]);
    }

    // --- `Arc` accessors -------------------------------------------------

    #[test]
    fn arc_accessors() {
        let (bytes, start) = build_single_key_fst(b"q", b"9");
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        let mut r = fst.reader();
        let first = fst.first_arc();
        let arc = fst
            .find_target_arc(b'q' as i32, &first, &mut r)
            .unwrap()
            .unwrap();
        assert_eq!(arc.label(), b'q' as i32);
        assert_eq!(arc.output(), &[] as &[u8]);
        assert_eq!(arc.target(), FINAL_END_NODE);
        assert!(arc.is_last());
        assert!(arc.is_final());
        assert_eq!(arc.next_final_output(), b"9");
    }

    // --- Hand-built multi-arc nodes ---------------------------------------
    //
    // `build_single_key_fst` only ever emits single-arc nodes with explicit
    // vlong targets. Real Lucene-written FSTs (see `tests/fst_fixtures.rs`)
    // exercise genuine branching, but happened to never use the
    // `BIT_TARGET_NEXT` compaction for a *non-last* sibling arc for this
    // fixture's particular key set, nor a final-output-bearing arc that gets
    // skipped over mid-scan, nor a non-final stop node, nor a
    // zero-length-but-flagged output -- so those `read_arc`/
    // `seek_to_next_node`/`find_target_arc` branches need bytes built by
    // hand here instead.

    /// Builds a two-arc root node: `'a'` (not last, `BIT_TARGET_NEXT`,
    /// pointing implicitly at a one-arc child node for `'b'`) and `'z'`
    /// (last, itself accepting). Exercises `read_arc`'s `BIT_TARGET_NEXT`
    /// non-last branch, which calls `seek_to_next_node` to skip past `'z'`
    /// before landing on the `'b'`-node's address.
    fn build_target_next_branching_fst(b_output: &[u8], z_output: &[u8]) -> (Vec<u8>, i64) {
        let mut bytes = Vec::new();

        // Child node for 'b' (address = addr_b), reached only via 'a's
        // implicit BIT_TARGET_NEXT target.
        let mut node_b = Vec::new();
        let mut flags_b = BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE;
        if !b_output.is_empty() {
            flags_b |= BIT_ARC_HAS_FINAL_OUTPUT;
        }
        node_b.push(flags_b);
        node_b.push(b'b');
        if !b_output.is_empty() {
            write_vint(&mut node_b, b_output.len() as i32);
            node_b.extend_from_slice(b_output);
        }
        let addr_b = append_arc_logical(&mut bytes, &node_b);

        // Root's 'z' arc (appended directly after node_b, so it stays
        // adjacent -- this is what makes 'a's implicit TARGET_NEXT target
        // land exactly on addr_b once 'z' is fully skipped).
        let mut arc_z = Vec::new();
        let mut flags_z = BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE;
        if !z_output.is_empty() {
            flags_z |= BIT_ARC_HAS_FINAL_OUTPUT;
        }
        arc_z.push(flags_z);
        arc_z.push(b'z');
        if !z_output.is_empty() {
            write_vint(&mut arc_z, z_output.len() as i32);
            arc_z.extend_from_slice(z_output);
        }
        append_arc_logical(&mut bytes, &arc_z);

        // Root's 'a' arc: not last, BIT_TARGET_NEXT, no output of its own --
        // its target is implicit (wherever the byte stream lands right
        // after skipping every remaining sibling, i.e. addr_b).
        let arc_a = vec![BIT_TARGET_NEXT, b'a'];
        let addr_a = append_arc_logical(&mut bytes, &arc_a);

        assert!(addr_b < addr_a); // sanity: node_b really is "below" root
        (bytes, addr_a)
    }

    #[test]
    fn target_next_non_last_arc_reaches_child_via_seek_to_next_node() {
        let (bytes, start) = build_target_next_branching_fst(b"B", b"Z");
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(b"ab").unwrap(), Some(b"B".to_vec()));
        assert_eq!(fst.get(b"z").unwrap(), Some(b"Z".to_vec()));
        // 'a' alone is a structural-only arc (not final): present in the
        // FST's arcs but not an accepted key.
        assert_eq!(fst.get(b"a").unwrap(), None);
        assert_eq!(fst.get(b"ac").unwrap(), None);
    }

    #[test]
    fn skips_final_output_of_a_non_matching_sibling_arc_during_scan() {
        // Root arcs: 'a' (itself accepting, has a final output, NOT last)
        // and 'z' (last, accepting). Looking up "z" must skip clean over
        // 'a's final-output bytes (`find_target_arc`'s
        // `BIT_ARC_HAS_FINAL_OUTPUT` skip branch) before reaching 'z'.
        let mut bytes = Vec::new();

        let mut arc_z = vec![
            BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT,
            b'z',
        ];
        write_vint(&mut arc_z, 1);
        arc_z.push(b'Z');
        append_arc_logical(&mut bytes, &arc_z);

        let mut arc_a = vec![
            BIT_FINAL_ARC | BIT_ARC_HAS_FINAL_OUTPUT | BIT_STOP_NODE,
            b'a',
        ];
        write_vint(&mut arc_a, 1);
        arc_a.push(b'A');
        let start = append_arc_logical(&mut bytes, &arc_a);

        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(b"z").unwrap(), Some(b"Z".to_vec()));
        assert_eq!(fst.get(b"a").unwrap(), Some(b"A".to_vec()));
    }

    #[test]
    fn stop_node_without_final_arc_is_a_non_accepting_dead_end() {
        // A `BIT_STOP_NODE` arc without `BIT_FINAL_ARC` represents Lucene's
        // "non-final dead-end state" (`FST.NON_FINAL_END_NODE`): the label
        // matches structurally but the key isn't accepted there.
        let arc = vec![BIT_LAST_ARC | BIT_STOP_NODE, b'q'];
        let mut bytes = Vec::new();
        let start = append_arc_logical(&mut bytes, &arc);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(b"q").unwrap(), None);
    }

    #[test]
    fn zero_length_flagged_output_is_read_as_empty() {
        // BIT_ARC_HAS_OUTPUT set but the stored vint length is 0: a
        // defensive case a real encoder wouldn't produce (it would just
        // clear the flag), but `read_output`'s `len == 0` branch must still
        // decode it as an empty output rather than misreading subsequent
        // bytes as output content.
        let mut bytes = Vec::new();
        let mut arc = vec![
            BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_OUTPUT,
            b'r',
        ];
        write_vint(&mut arc, 0); // output length 0, despite the flag being set
        let start = append_arc_logical(&mut bytes, &arc);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        assert_eq!(fst.get(b"r").unwrap(), Some(Vec::new()));
    }

    // --- `build_fst` (simplified FSTCompiler) -----------------------------
    //
    // These round-trip *through the existing, unmodified* `Fst::read`/
    // `Fst::get` -- the whole point of this builder is that its output is
    // usable by the reader that was already written and tested above, not
    // just "some bytes this module's own code happens to also understand".

    fn assert_round_trips(
        entries: &[(Vec<u8>, Vec<u8>)],
        present: &[(&[u8], &[u8])],
        absent: &[&[u8]],
    ) {
        let fst = build_fst(entries).unwrap();
        let file = write_fst(&fst);
        let mut input = SliceInput::new(&file);
        let read_back = Fst::read(&mut input).unwrap();

        for (key, expected) in present {
            assert_eq!(
                read_back.get(key).unwrap(),
                Some(expected.to_vec()),
                "key {key:?} should resolve to {expected:?}"
            );
            // Also check the in-memory `Fst` `build_fst` returns directly,
            // independent of the `write_fst`/`Fst::read` serialization round
            // trip.
            assert_eq!(fst.get(key).unwrap(), Some(expected.to_vec()));
        }
        for key in absent {
            assert_eq!(
                read_back.get(key).unwrap(),
                None,
                "key {key:?} should be absent"
            );
            assert_eq!(fst.get(key).unwrap(), None);
        }
    }

    /// The same 7-key shape as `fixtures/src/GenFst.java`'s fixture (shared
    /// prefixes within each group, two disjoint groups, plus a lone
    /// single-byte key), so this test can be eyeballed against that fixture's
    /// differential test for a sanity cross-check even though this builder
    /// doesn't consume real Lucene bytes.
    fn seven_key_fixture() -> Vec<(Vec<u8>, Vec<u8>)> {
        vec![
            (b"app".to_vec(), b"1".to_vec()),
            (b"apple".to_vec(), b"2".to_vec()),
            (b"application".to_vec(), b"3".to_vec()),
            (b"banana".to_vec(), b"4".to_vec()),
            (b"band".to_vec(), b"5".to_vec()),
            (b"bandana".to_vec(), b"6".to_vec()),
            (b"z".to_vec(), b"7".to_vec()),
        ]
    }

    #[test]
    fn build_fst_seven_key_fixture_round_trips_through_real_reader() {
        let entries = seven_key_fixture();
        assert_round_trips(
            &entries,
            &[
                (b"app", b"1"),
                (b"apple", b"2"),
                (b"application", b"3"),
                (b"banana", b"4"),
                (b"band", b"5"),
                (b"bandana", b"6"),
                (b"z", b"7"),
            ],
            &[
                b"ap",         // proper prefix of app, not itself accepted
                b"appl",       // proper prefix of apple/application
                b"applicatio", // proper prefix of application
                b"appz",       // diverges mid-key
                b"ban",        // proper prefix of banana/band/bandana
                b"bandanas",   // extends past an accepting node
                b"",           // no empty-string key in this fixture
                b"zz",         // extends past z's accepting stop node
                b"missing-entirely",
            ],
        );
    }

    #[test]
    fn build_fst_single_key() {
        let entries = vec![(b"cat".to_vec(), b"1".to_vec())];
        assert_round_trips(&entries, &[(b"cat", b"1")], &[b"ca", b"cats", b"dog", b""]);
    }

    #[test]
    fn build_fst_empty_key_set_never_accepts_anything() {
        let fst = build_fst(&[]).unwrap();
        assert_eq!(fst.get(b"").unwrap(), None);
        assert_eq!(fst.get(b"anything").unwrap(), None);
    }

    #[test]
    fn build_fst_accepts_the_empty_string_key_via_empty_output() {
        let entries = vec![
            (Vec::new(), b"root".to_vec()),
            (b"x".to_vec(), b"1".to_vec()),
        ];
        assert_round_trips(&entries, &[(b"", b"root"), (b"x", b"1")], &[b"xx", b"y"]);
    }

    #[test]
    fn build_fst_one_key_is_a_proper_prefix_of_another() {
        // "band" is itself accepted *and* has a further child ('a' -> "na")
        // continuing to "bandana" -- exercises the non-stop-node branch of
        // `build_node` where a child is both final and has its own children.
        let entries = vec![
            (b"band".to_vec(), b"short".to_vec()),
            (b"bandana".to_vec(), b"long".to_vec()),
        ];
        assert_round_trips(
            &entries,
            &[(b"band", b"short"), (b"bandana", b"long")],
            &[b"ban", b"bandan", b"bandanas"],
        );
    }

    #[test]
    fn build_fst_rejects_unsorted_input() {
        let entries = vec![(b"b".to_vec(), Vec::new()), (b"a".to_vec(), Vec::new())];
        assert_eq!(
            build_fst(&entries).unwrap_err(),
            BuildError::NotSorted { index: 1 }
        );
    }

    #[test]
    fn build_fst_rejects_duplicate_key() {
        let entries = vec![(b"a".to_vec(), Vec::new()), (b"a".to_vec(), Vec::new())];
        assert_eq!(
            build_fst(&entries).unwrap_err(),
            BuildError::NotSorted { index: 1 }
        );
    }

    #[test]
    fn build_fst_many_keys_forces_multi_byte_vlong_targets() {
        // Enough keys/branching that at least one child node's address
        // exceeds the 1-byte vlong threshold (127), forcing `write_vlong`'s
        // (and `read_vlong`'s, on the read side already covered above)
        // multi-byte continuation branch through the builder's own target
        // encoding.
        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u16..200)
            .map(|i| {
                let key = format!("key{i:04}").into_bytes();
                let output = i.to_le_bytes().to_vec();
                (key, output)
            })
            .collect();
        let fst = build_fst(&entries).unwrap();
        assert!(fst.metadata().num_bytes > 127);
        for (key, output) in &entries {
            assert_eq!(fst.get(key).unwrap(), Some(output.clone()));
        }
        assert_eq!(fst.get(b"key9999").unwrap(), None);
    }

    #[test]
    fn build_fst_many_keys_round_trips_through_write_fst_and_both_readers() {
        // The 200-key/multi-byte-vlong-target case above only queries the
        // freshly-built in-memory `Fst` directly -- it never serializes via
        // `write_fst` and never re-parses via `Fst::read`/`Fst::read_borrowed`,
        // so the multi-byte vlong target shape was never exercised through
        // either actual reader, only through the builder's own in-memory
        // representation. This closes that gap.
        let entries: Vec<(Vec<u8>, Vec<u8>)> = (0u16..200)
            .map(|i| {
                let key = format!("key{i:04}").into_bytes();
                let output = i.to_le_bytes().to_vec();
                (key, output)
            })
            .collect();
        let built = build_fst(&entries).unwrap();
        let bytes = write_fst(&built);

        let mut owned_input = SliceInput::new(&bytes);
        let read_fst = Fst::read(&mut owned_input).unwrap();
        for (key, output) in &entries {
            assert_eq!(read_fst.get(key).unwrap(), Some(output.clone()));
        }
        assert_eq!(read_fst.get(b"key9999").unwrap(), None);

        let mut borrowed_input = SliceInput::new(&bytes);
        let borrowed_fst = Fst::read_borrowed(&mut borrowed_input).unwrap();
        assert!(borrowed_fst.is_borrowed());
        for (key, output) in &entries {
            assert_eq!(borrowed_fst.get(key).unwrap(), Some(output.clone()));
        }
        assert_eq!(borrowed_fst.get(b"key9999").unwrap(), None);
    }

    // --- `Fst::read_borrowed` (zero-copy body) ----------------------------

    /// Builds one `(codec header + metadata + body)` byte buffer via
    /// `build_fst`/`write_fst`, then confirms `Fst::read` (owned copy) and
    /// `Fst::read_borrowed` (zero-copy) agree on every present/absent key --
    /// same fixture shape `build_fst_seven_key_fixture_round_trips_through_real_reader`
    /// already uses for the owned path.
    #[test]
    fn read_borrowed_matches_read_for_same_bytes() {
        let entries = seven_key_fixture();
        let built = build_fst(&entries).unwrap();
        let file = write_fst(&built);

        let mut owned_input = SliceInput::new(&file);
        let owned = Fst::read(&mut owned_input).unwrap();

        let mut borrowed_input = SliceInput::new(&file);
        let borrowed = Fst::read_borrowed(&mut borrowed_input).unwrap();

        assert!(!owned.is_borrowed());
        assert!(borrowed.is_borrowed());

        let present: &[(&[u8], &[u8])] = &[
            (b"app", b"1"),
            (b"apple", b"2"),
            (b"application", b"3"),
            (b"banana", b"4"),
            (b"band", b"5"),
            (b"bandana", b"6"),
            (b"z", b"7"),
        ];
        for (key, expected) in present {
            assert_eq!(owned.get(key).unwrap(), Some(expected.to_vec()));
            assert_eq!(borrowed.get(key).unwrap(), Some(expected.to_vec()));
        }
        for key in [b"ap".as_slice(), b"appz", b"missing-entirely"] {
            assert_eq!(owned.get(key).unwrap(), None);
            assert_eq!(borrowed.get(key).unwrap(), None);
        }
    }

    /// Structural proof (not just a runtime output check) that
    /// `Fst::read_borrowed` really borrows rather than allocating its own
    /// second copy of the body: `FstBytes::Borrowed` holds a `&[u8]`, which
    /// (unlike `FstBytes::Owned(Vec<u8>)`) cannot itself own heap
    /// allocation -- if the body were copied, this variant would have had
    /// to be `Vec<u8>` to hold it. This is the honest, provable half of
    /// "no extra full-size copy happens": a `&[u8]` is a pointer+length,
    /// never a second buffer.
    #[test]
    fn read_borrowed_body_is_a_slice_not_a_second_owned_buffer() {
        let entries = seven_key_fixture();
        let built = build_fst(&entries).unwrap();
        let file = write_fst(&built);

        let mut input = SliceInput::new(&file);
        let fst = Fst::read_borrowed(&mut input).unwrap();
        match &fst.bytes {
            FstBytes::Borrowed(slice) => {
                // The slice's address must fall inside `file`'s own
                // allocation -- proof this is a view into the caller's
                // buffer, not a copy living at some independent address.
                let file_start = file.as_ptr() as usize;
                let file_end = file_start + file.len();
                let slice_start = slice.as_ptr() as usize;
                assert!(slice_start >= file_start && slice_start <= file_end);
                assert_eq!(slice.len(), fst.metadata.num_bytes as usize);
            }
            FstBytes::Owned(_) => panic!("read_borrowed must produce FstBytes::Borrowed"),
        }
    }

    /// End-to-end through this port's real zero-copy `Directory` backend
    /// (`MmapDirectory`): writes an FST file to disk, opens it via
    /// `MmapDirectory` (an actual OS `mmap(2)`, `lucene_store::directory`'s
    /// `Input::Mapped`), and confirms `Fst::read_borrowed` over that mapped
    /// buffer resolves keys correctly -- this is the concrete scenario the
    /// module doc on `FstBytes` describes: a large FST backed by mmap'd
    /// bytes, opened without a second full-size heap copy.
    #[test]
    fn read_borrowed_over_a_real_mmap_directory_input() {
        use lucene_store::directory::{Directory, MmapDirectory};
        use lucene_store::DataOutput;

        let entries = seven_key_fixture();
        let built = build_fst(&entries).unwrap();
        let file_bytes = write_fst(&built);

        let mut root = std::env::temp_dir();
        root.push(format!(
            "lucene-rust-fst-mmap-test-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&root).unwrap();

        let dir = MmapDirectory::open(&root);
        let mut out = dir.create_output("fst.bin").unwrap();
        out.write_bytes(&file_bytes);
        out.close().unwrap();

        let mapped = dir.open("fst.bin").unwrap();
        let mut input = SliceInput::new(&mapped);
        let fst = Fst::read_borrowed(&mut input).unwrap();
        assert!(fst.is_borrowed());
        assert_eq!(fst.get(b"app").unwrap(), Some(b"1".to_vec()));
        assert_eq!(fst.get(b"bandana").unwrap(), Some(b"6".to_vec()));
        assert_eq!(fst.get(b"missing").unwrap(), None);

        std::fs::remove_dir_all(&root).ok();
    }

    #[test]
    fn build_fst_long_output_forces_multi_byte_vint_length() {
        // Every other test's outputs are <=4 bytes, so the final-output
        // length's `write_vint` call never exercises its own multi-byte
        // (>127) continuation branch -- distinct from the vlong target
        // encoding the test above already stresses. A 200-byte output
        // forces that branch.
        let long_output: Vec<u8> = (0u16..200).map(|i| (i % 256) as u8).collect();
        let entries: Vec<(Vec<u8>, Vec<u8>)> = vec![
            (b"a".to_vec(), vec![]),
            (b"ab".to_vec(), long_output.clone()),
        ];
        let fst = build_fst(&entries).unwrap();
        assert_eq!(fst.get(b"a").unwrap(), Some(vec![]));
        assert_eq!(fst.get(b"ab").unwrap(), Some(long_output));
        assert_eq!(fst.get(b"abc").unwrap(), None);
    }

    // --- `Fst::iter` (full ordered enumeration) ---------------------------

    fn collect_iter(fst: &Fst) -> Vec<(Vec<u8>, Vec<u8>)> {
        fst.iter()
            .expect("BYTE1 fixture should support iter")
            .collect::<Result<_>>()
            .expect("enumeration should not error")
    }

    #[test]
    fn iter_over_empty_fst_yields_nothing() {
        let fst = build_fst(&[]).unwrap();
        assert_eq!(collect_iter(&fst), Vec::<(Vec<u8>, Vec<u8>)>::new());
    }

    #[test]
    fn iter_over_single_key_fst_yields_that_one_key() {
        let entries = vec![(b"cat".to_vec(), b"1".to_vec())];
        let fst = build_fst(&entries).unwrap();
        assert_eq!(collect_iter(&fst), vec![(b"cat".to_vec(), b"1".to_vec())]);
    }

    #[test]
    fn iter_stays_exhausted_after_yielding_none() {
        // Found in review: `advance()` used `upto == 0` to mean both
        // "not yet started" and "just finished" -- calling `next()` again
        // after it returned `None` re-entered the "not yet started" branch
        // and silently restarted enumeration from the first key instead of
        // staying exhausted, violating the standard Rust `Iterator`
        // contract (an iterator should keep returning `None` once it has).
        let entries = vec![(b"cat".to_vec(), b"1".to_vec())];
        let fst = build_fst(&entries).unwrap();
        let mut it = fst.iter().unwrap();
        assert_eq!(
            it.next().unwrap().unwrap(),
            (b"cat".to_vec(), b"1".to_vec())
        );
        assert!(it.next().is_none());
        // The bug: this second post-exhaustion call would incorrectly
        // yield "cat" again instead of staying `None`.
        assert!(it.next().is_none());
        assert!(it.next().is_none());
    }

    #[test]
    fn iter_over_single_empty_string_key_yields_it() {
        // The only accepted key is the empty string itself (via emptyOutput),
        // with no other real arcs at all -- exercises `read_first_target_arc`'s
        // `follow.target() <= 0` branch (the synthetic END_LABEL arc is also
        // BIT_LAST_ARC, so enumeration must stop immediately after it).
        let entries = vec![(Vec::new(), b"root".to_vec())];
        let fst = build_fst(&entries).unwrap();
        assert_eq!(collect_iter(&fst), vec![(Vec::new(), b"root".to_vec())]);
    }

    #[test]
    fn iter_over_seven_key_fixture_yields_ascending_sorted_order() {
        let entries = seven_key_fixture();
        let fst = build_fst(&entries).unwrap();
        assert_eq!(collect_iter(&fst), entries);
    }

    #[test]
    fn iter_over_empty_string_plus_other_keys() {
        // Empty string accepted *and* the root has further real children --
        // exercises `read_first_target_arc`'s `follow.target() > 0` branch
        // (the synthetic END_LABEL arc is not last, so `read_next_arc` must
        // fall through to the root node's first real arc afterwards).
        let entries = vec![
            (Vec::new(), b"root".to_vec()),
            (b"x".to_vec(), b"1".to_vec()),
        ];
        let fst = build_fst(&entries).unwrap();
        assert_eq!(collect_iter(&fst), entries);
    }

    #[test]
    fn iter_over_binary_search_root_node_yields_ascending_order() {
        let labels = [b'a', b'c', b'f', b'm', b'z'];
        let outputs = [1u8, 2, 3, 4, 5];
        let (bytes, start) = build_binary_search_node(&labels, &outputs);
        let fst = fst_from_body(bytes, start, InputType::Byte1, None);
        let expected: Vec<(Vec<u8>, Vec<u8>)> = labels
            .iter()
            .zip(outputs.iter())
            .map(|(&l, &o)| (vec![l], vec![o]))
            .collect();
        assert_eq!(collect_iter(&fst), expected);
    }

    #[test]
    fn iter_errors_on_non_byte1_input_type() {
        let (bytes, start) = build_single_key_fst(b"x", b"1");
        let fst = fst_from_body(bytes, start, InputType::Byte2, None);
        assert!(matches!(fst.iter(), Err(Error::Unsupported(_))));
    }

    // --- Seek support (`seek_exact`/`seek_ceil`/`seek_floor`) --------------
    //
    // These exercise the hand-built, list-encoded seven-key fixture already
    // used above -- `tests/fst_seek_fixtures.rs` covers the same operations
    // against real, Lucene-written fixtures spanning all four node encodings
    // (list, binary search, direct addressing, continuous).

    #[test]
    fn fst_seek_exact_on_present_and_absent_keys() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        assert_eq!(fst.seek_exact(b"app").unwrap(), Some(b"1".to_vec()));
        assert_eq!(fst.seek_exact(b"bandana").unwrap(), Some(b"6".to_vec()));
        assert_eq!(fst.seek_exact(b"appl").unwrap(), None);
        assert_eq!(fst.seek_exact(b"missing").unwrap(), None);

        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_exact(b"z").unwrap(),
            Some((b"z".to_vec(), b"7".to_vec()))
        );
        assert_eq!(e.seek_exact(b"appl").unwrap(), None);
    }

    #[test]
    fn fst_enum_seek_ceil_lands_between_two_keys() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        // "appl" sits strictly between "app" and "apple".
        assert_eq!(
            e.seek_ceil(b"appl").unwrap(),
            Some((b"apple".to_vec(), b"2".to_vec()))
        );
        // "ban" sits strictly between "bandana"'s prefix and "banana".
        assert_eq!(
            e.seek_ceil(b"ban").unwrap(),
            Some((b"banana".to_vec(), b"4".to_vec()))
        );
    }

    #[test]
    fn fst_enum_seek_ceil_before_first_key_finds_first_key() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_ceil(b"").unwrap(),
            Some((b"app".to_vec(), b"1".to_vec()))
        );
        assert_eq!(
            e.seek_ceil(b"AAA").unwrap(),
            Some((b"app".to_vec(), b"1".to_vec()))
        );
    }

    #[test]
    fn fst_enum_seek_ceil_past_last_key_finds_nothing() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(e.seek_ceil(b"zz").unwrap(), None);
        assert_eq!(e.seek_ceil(b"zzzzzzzzzz").unwrap(), None);
    }

    #[test]
    fn fst_enum_seek_ceil_on_exact_key_returns_it() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_ceil(b"banana").unwrap(),
            Some((b"banana".to_vec(), b"4".to_vec()))
        );
    }

    #[test]
    fn fst_enum_seek_floor_lands_between_two_keys() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_floor(b"appl").unwrap(),
            Some((b"app".to_vec(), b"1".to_vec()))
        );
        assert_eq!(
            e.seek_floor(b"bane").unwrap(),
            Some((b"bandana".to_vec(), b"6".to_vec()))
        );
    }

    #[test]
    fn fst_enum_seek_floor_past_last_key_finds_last_key() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_floor(b"zzzzzzzzzz").unwrap(),
            Some((b"z".to_vec(), b"7".to_vec()))
        );
    }

    #[test]
    fn fst_enum_seek_floor_before_first_key_finds_nothing() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(e.seek_floor(b"").unwrap(), None);
        assert_eq!(e.seek_floor(b"AAA").unwrap(), None);
    }

    #[test]
    fn fst_enum_seek_floor_on_exact_key_returns_it() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_floor(b"band").unwrap(),
            Some((b"band".to_vec(), b"5".to_vec()))
        );
    }

    #[test]
    fn fst_enum_sequential_seeks_forward_and_backward_use_rewind_prefix() {
        // Exercises `rewind_prefix`'s forward (cmp<0) and backward (cmp>0)
        // branches across successive seeks on the same `FstEnum`, not just a
        // single fresh seek from the start.
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_ceil(b"application").unwrap(),
            Some((b"application".to_vec(), b"3".to_vec()))
        );
        // Backward: "ap" < "application"'s shared prefix.
        assert_eq!(
            e.seek_ceil(b"ap").unwrap(),
            Some((b"app".to_vec(), b"1".to_vec()))
        );
        // Forward again, past everything already visited.
        assert_eq!(
            e.seek_ceil(b"c").unwrap(),
            Some((b"z".to_vec(), b"7".to_vec()))
        );
        // "z" is the last key -- a plain `next()` after landing on it (via
        // seek) correctly reports the enumeration is exhausted, same as real
        // Lucene's `FSTEnum` (seeking only repositions the enum; it doesn't
        // change what "the next key after this one" means).
        assert!(e.next().is_none());
    }

    #[test]
    fn fst_enum_seek_on_empty_string_key_fixture() {
        let entries = vec![
            (Vec::new(), b"root".to_vec()),
            (b"x".to_vec(), b"1".to_vec()),
        ];
        let fst = build_fst(&entries).unwrap();
        assert_eq!(fst.seek_exact(b"").unwrap(), Some(b"root".to_vec()));

        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_ceil(b"").unwrap(),
            Some((Vec::new(), b"root".to_vec()))
        );
        assert_eq!(
            e.seek_floor(b"").unwrap(),
            Some((Vec::new(), b"root".to_vec()))
        );
        assert_eq!(
            e.seek_ceil(b"w").unwrap(),
            Some((b"x".to_vec(), b"1".to_vec()))
        );
        assert_eq!(
            e.seek_floor(b"w").unwrap(),
            Some((Vec::new(), b"root".to_vec()))
        );
    }

    #[test]
    fn fst_enum_seek_on_empty_fst_finds_nothing() {
        let fst = build_fst(&[]).unwrap();
        let mut e = fst.iter().unwrap();
        assert_eq!(e.seek_ceil(b"anything").unwrap(), None);
        assert_eq!(e.seek_floor(b"anything").unwrap(), None);
        assert_eq!(fst.seek_exact(b"").unwrap(), None);
    }

    #[test]
    fn fst_enum_next_resumes_after_seek_following_full_exhaustion() {
        // Regression test: `seek_ceil`/`seek_floor`/`seek_exact` must clear
        // `done` (a Rust-only fused-iterator flag with no Java equivalent) so
        // that repositioning a fully-exhausted `FstEnum` and calling `next()`
        // again resumes ordered enumeration instead of short-circuiting.
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        while e.next().is_some() {}
        assert_eq!(
            e.seek_ceil(b"ban").unwrap(),
            Some((b"banana".to_vec(), b"4".to_vec()))
        );
        assert_eq!(
            e.next().unwrap().unwrap(),
            (b"band".to_vec(), b"5".to_vec())
        );
        assert_eq!(
            e.next().unwrap().unwrap(),
            (b"bandana".to_vec(), b"6".to_vec())
        );
    }

    #[test]
    fn fst_enum_next_resumes_after_seek_floor_following_full_exhaustion() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        while e.next().is_some() {}
        assert_eq!(
            e.seek_floor(b"band").unwrap(),
            Some((b"band".to_vec(), b"5".to_vec()))
        );
        assert_eq!(
            e.next().unwrap().unwrap(),
            (b"bandana".to_vec(), b"6".to_vec())
        );
    }

    #[test]
    fn fst_enum_next_resumes_after_seek_exact_following_full_exhaustion() {
        let fst = build_fst(&seven_key_fixture()).unwrap();
        let mut e = fst.iter().unwrap();
        while e.next().is_some() {}
        assert_eq!(
            e.seek_exact(b"banana").unwrap(),
            Some((b"banana".to_vec(), b"4".to_vec()))
        );
        assert_eq!(
            e.next().unwrap().unwrap(),
            (b"band".to_vec(), b"5".to_vec())
        );
    }

    // --- Suffix sharing / minimization proof tests -------------------------
    //
    // These prove `build_node`'s `NodeHash` dedup actually collapses
    // structurally-identical nodes reached via different prefixes into a
    // single shared address, rather than merely producing a correct (but
    // possibly non-minimal) FST. A "round-trips correctly" test alone would
    // pass even for the old, pre-dedup naive trie builder -- these instead
    // inspect the compiled node addresses directly.

    /// Walks `prefix` from the FST's root and returns the address of the
    /// node reached after consuming it (i.e. the `target()` of the arc for
    /// `prefix`'s last byte) -- the same "which node did we land on" question
    /// `Fst::get` answers per-byte, just exposed here instead of being
    /// discarded after the final `is_final()` check.
    fn node_after_prefix(fst: &Fst<'_>, prefix: &[u8]) -> i64 {
        let mut r = fst.reader();
        let mut arc = fst.first_arc();
        for &b in prefix {
            arc = fst
                .find_target_arc(b as i32, &arc, &mut r)
                .unwrap()
                .unwrap_or_else(|| panic!("prefix {prefix:?} not found in FST"));
        }
        arc.target()
    }

    #[test]
    fn build_fst_shares_identical_suffix_node_across_two_prefixes() {
        // "bat" and "cat" share the final "at" suffix: the node reached
        // after 'b' and the node reached after 'c' are structurally
        // identical (single child arc 'a', itself leading to a single child
        // arc 't', final, no output) and must be written to the byte store
        // exactly once and reused, i.e. `node_after_prefix(.., b"b")` and
        // `node_after_prefix(.., b"c")` must be the *same* address.
        let entries = vec![(b"bat".to_vec(), Vec::new()), (b"cat".to_vec(), Vec::new())];
        let fst = build_fst(&entries).unwrap();

        let after_b = node_after_prefix(&fst, b"b");
        let after_c = node_after_prefix(&fst, b"c");
        assert_eq!(
            after_b, after_c,
            "the shared \"at\" suffix must compile to one shared node address, not two independent copies"
        );

        // Sharing a node must not corrupt lookups through either path.
        assert_eq!(fst.get(b"bat").unwrap(), Some(Vec::new()));
        assert_eq!(fst.get(b"cat").unwrap(), Some(Vec::new()));
        assert_eq!(fst.get(b"at").unwrap(), None);
        assert_eq!(fst.get(b"ba").unwrap(), None);
    }

    #[test]
    fn build_fst_shares_classic_mop_pop_stop_top_suffix_node() {
        // Real Lucene's own canonical `FSTCompiler` suffix-sharing example:
        // "mop"/"moth"/"pop"/"star"/"stop"/"top". The nodes reached after
        // "po", "sto", and "to" are each a single final 'p' arc (stop node,
        // no output) and so must all three collapse to the identical shared
        // address, while the node after "mo" (which additionally has a 'th'
        // child) must NOT be that same address.
        let entries = vec![
            (b"mop".to_vec(), Vec::new()),
            (b"moth".to_vec(), Vec::new()),
            (b"pop".to_vec(), Vec::new()),
            (b"star".to_vec(), Vec::new()),
            (b"stop".to_vec(), Vec::new()),
            (b"top".to_vec(), Vec::new()),
        ];
        let fst = build_fst(&entries).unwrap();

        let after_po = node_after_prefix(&fst, b"po");
        let after_sto = node_after_prefix(&fst, b"sto");
        let after_to = node_after_prefix(&fst, b"to");
        let after_mo = node_after_prefix(&fst, b"mo");

        assert_eq!(
            after_po, after_sto,
            "\"po\" and \"sto\" both end in a bare final 'p' node and must share it"
        );
        assert_eq!(
            after_sto, after_to,
            "\"sto\" and \"to\" both end in a bare final 'p' node and must share it"
        );
        assert_ne!(
            after_mo, after_po,
            "\"mo\" has an extra 'th' child so its node must NOT be reused for the plain-'p' shared node"
        );

        // Regression: get/seek/enumerate all still correct against an FST
        // that now has genuinely shared/multiply-referenced nodes.
        for (key, _) in &entries {
            assert_eq!(fst.get(key).unwrap(), Some(Vec::new()), "get({key:?})");
        }
        assert_eq!(collect_iter(&fst), entries);

        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_ceil(b"mos").unwrap(),
            Some((b"moth".to_vec(), Vec::new()))
        );
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_floor(b"stoz").unwrap(),
            Some((b"stop".to_vec(), Vec::new()))
        );
        let mut e = fst.iter().unwrap();
        assert_eq!(
            e.seek_exact(b"top").unwrap(),
            Some((b"top".to_vec(), Vec::new()))
        );
    }

    #[test]
    fn build_fst_dedup_keeps_compiled_size_sublinear_in_repeated_suffix_count() {
        // A naive (non-deduplicating) trie builder writes one independent
        // copy of the shared tail's arcs per distinct prefix, so its total
        // size grows linearly with `num_prefixes * tail.len()`. With
        // suffix-sharing minimization, the tail's arcs are written once and
        // every prefix's last arc just targets that single shared address,
        // so total size grows roughly like `num_prefixes + tail.len()`
        // instead. Assert the actual compiled size is well under the naive
        // linear bound to prove sharing, not just correctness, occurred.
        const TAIL: &[u8] = b"xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"; // 44 bytes
        let prefixes: &[&[u8]] = &[b"aa", b"bb", b"cc", b"dd", b"ee", b"ff", b"gg", b"hh"];

        let mut entries: Vec<(Vec<u8>, Vec<u8>)> = prefixes
            .iter()
            .map(|p| {
                let mut key = p.to_vec();
                key.extend_from_slice(TAIL);
                (key, Vec::new())
            })
            .collect();
        entries.sort();

        let fst = build_fst(&entries).unwrap();

        // Naive bound: each of the 8 prefixes would need its own full,
        // independent copy of every arc down the shared tail. Each arc here
        // is at least 3 bytes (flags + label + 1-byte vlong target), so a
        // fully-unshared build would need well over
        // `prefixes.len() * TAIL.len() * 3` bytes for the tails alone.
        let naive_unshared_lower_bound = prefixes.len() * TAIL.len() * 3;
        assert!(
            (fst.metadata.num_bytes as usize) < naive_unshared_lower_bound,
            "compiled size {} should be far smaller than the naive no-sharing bound {} \
             if suffix sharing actually collapsed the repeated tail to one copy",
            fst.metadata.num_bytes,
            naive_unshared_lower_bound
        );

        // And every key must still resolve correctly through the shared tail.
        for (key, output) in &entries {
            assert_eq!(fst.get(key).unwrap().as_ref(), Some(output));
        }

        // Direct structural check: the node reached after each distinct
        // 2-byte prefix must be the *same* address (all of them lead into
        // the one shared `TAIL` chain).
        let first_addr = node_after_prefix(&fst, prefixes[0]);
        for p in &prefixes[1..] {
            assert_eq!(
                node_after_prefix(&fst, p),
                first_addr,
                "prefix {p:?} should land on the same shared tail node as {:?}",
                prefixes[0]
            );
        }
    }

    // --- Typed outputs: `PositiveIntOutputs` / `PairOutputs` ---------------

    type WeightedEntry = (Vec<u8>, Pair<i64, Vec<u8>>);

    #[test]
    fn positive_int_outputs_round_trips_values_including_zero() {
        let entries: Vec<(Vec<u8>, i64)> = vec![
            (b"a".to_vec(), 0),
            (b"apple".to_vec(), 1),
            (b"banana".to_vec(), 300),
            (b"grape".to_vec(), i64::MAX),
        ];
        let fst = build_fst_typed::<PositiveIntOutputs>(&entries).unwrap();
        for (key, value) in &entries {
            assert_eq!(
                fst.get_typed::<PositiveIntOutputs>(key).unwrap(),
                Some(*value),
                "get_typed({key:?})"
            );
        }
        assert_eq!(
            fst.get_typed::<PositiveIntOutputs>(b"missing").unwrap(),
            None
        );

        // NO_OUTPUT (0) must encode to zero extra bytes, same as a plain
        // empty `Vec<u8>` output today.
        assert_eq!(PositiveIntOutputs::encode(&0), Vec::<u8>::new());
    }

    #[test]
    fn pair_outputs_round_trips_both_components_over_a_shared_prefix_key_set() {
        // A realistic key set with both shared prefixes ("band"/"banana"/
        // "bandana") and a shared suffix ("cat"/"bat", stressing interaction
        // with `build_node`'s suffix-sharing dedup) -- every key gets a
        // distinct (weight, payload) pair and both components must read
        // back correctly regardless of which nodes ended up shared.
        type Weighted = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
        let entries: Vec<WeightedEntry> = vec![
            (
                b"banana".to_vec(),
                Pair {
                    first: 4,
                    second: b"fruit".to_vec(),
                },
            ),
            (
                b"band".to_vec(),
                Pair {
                    first: 5,
                    second: b"music".to_vec(),
                },
            ),
            (
                b"bandana".to_vec(),
                Pair {
                    first: 6,
                    second: b"headwear".to_vec(),
                },
            ),
            (
                b"bat".to_vec(),
                Pair {
                    first: 1,
                    second: Vec::new(),
                },
            ),
            (
                b"cat".to_vec(),
                Pair {
                    first: 2,
                    second: b"meow".to_vec(),
                },
            ),
        ];
        let sorted = {
            let mut e = entries.clone();
            e.sort_by(|a, b| a.0.cmp(&b.0));
            e
        };
        let fst = build_fst_typed::<Weighted>(&sorted).unwrap();

        for (key, value) in &entries {
            let got = fst.get_typed::<Weighted>(key).unwrap();
            assert_eq!(got.as_ref(), Some(value), "get_typed({key:?})");
        }
        assert_eq!(fst.get_typed::<Weighted>(b"ba").unwrap(), None);
    }

    #[test]
    fn pair_outputs_shared_suffix_nodes_keep_distinct_first_components() {
        // Highest-risk interaction with suffix sharing: "bat" and "cat" would
        // share their tail node under plain `ByteSequenceOutputs` (see
        // `build_fst_shares_identical_suffix_node_across_two_prefixes`
        // above), but here their *entire* output lives on that same final
        // arc and differs per key (different `Pair` values) -- so the two
        // subtrees must NOT collapse to one shared node this time, and each
        // key's distinct pair must still read back correctly.
        type Weighted = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
        let entries: Vec<WeightedEntry> = vec![
            (
                b"bat".to_vec(),
                Pair {
                    first: 1,
                    second: b"flying".to_vec(),
                },
            ),
            (
                b"cat".to_vec(),
                Pair {
                    first: 2,
                    second: b"purring".to_vec(),
                },
            ),
        ];
        let fst = build_fst_typed::<Weighted>(&entries).unwrap();

        let after_b = node_after_prefix(&fst, b"b");
        let after_c = node_after_prefix(&fst, b"c");
        assert_ne!(
            after_b, after_c,
            "distinct per-key Pair outputs on the shared-shape \"at\" tail must prevent node sharing"
        );

        assert_eq!(
            fst.get_typed::<Weighted>(b"bat").unwrap(),
            Some(Pair {
                first: 1,
                second: b"flying".to_vec()
            })
        );
        assert_eq!(
            fst.get_typed::<Weighted>(b"cat").unwrap(),
            Some(Pair {
                first: 2,
                second: b"purring".to_vec()
            })
        );
    }

    #[test]
    fn pair_outputs_zero_value_encodes_to_no_extra_bytes() {
        type IntPair = PairOutputs<PositiveIntOutputs, PositiveIntOutputs>;
        let zero = <IntPair as Outputs>::zero();
        assert_eq!(IntPair::encode(&zero), Vec::<u8>::new());
        assert_eq!(IntPair::decode(&[]), zero);
    }
}
