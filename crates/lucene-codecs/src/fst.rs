//! Port of `org.apache.lucene.util.fst.FST` (read side) plus a from-scratch,
//! simplified construction path (`build_fst`) usable by this module's own
//! reader.
//!
//! This module implements enough of Lucene's FST (finite state transducer)
//! format to look up a byte-sequence key in an already-built, on-disk FST and
//! recover its accumulated output (`Util.get(fst, BytesRef)`). It does **not**
//! port real Lucene's *incremental* construction algorithm
//! (`FSTCompiler`/`Builder`'s node-freezing, suffix-sharing node hash table)
//! -- that remains a separate, much larger undertaking (see `docs/parity.md`).
//! Instead, `build_fst` (see the "FST construction" section near the bottom
//! of this file) builds a full in-memory trie and serializes it directly to
//! the same byte format `Fst::read`/`Fst::get` below already parse -- correct
//! and round-trip-verified against this module's own unmodified reader, but
//! without real `FSTCompiler`'s suffix sharing/minimization or fixed-length
//! arc nodes. See that section's doc comment for the precise list of what's
//! reproduced and what's deferred.
//!
//! ## Scope of this slice
//!
//! - **On-heap only.** Real Lucene's `FST` can be backed by an off-heap
//!   (mmap'd) `OffHeapFSTStore` or an on-heap `OnHeapFSTStore`. Only the
//!   on-heap representation is ported: the FST body is a plain `Vec<u8>`
//!   (`OnHeapFSTStore` with `bytesArray != null` writes/reads the body as one
//!   contiguous forward byte array — see `OnHeapFSTStore.java`).
//! - **Single output type: `BytesRef`-shaped (`ByteSequenceOutputs`).** This
//!   is the output type real Lucene uses for the term index FST
//!   (`Lucene90BlockTreeTermsReader`'s `.tip` FST maps term-prefix byte
//!   sequences to concatenated block-pointer byte sequences), so it's the one
//!   needed to eventually navigate BlockTree. Other output types
//!   (`PositiveIntOutputs`, `PairOutputs`, ...) are not implemented.
//! - **Variable-length ("list") arc nodes, plus `ARCS_FOR_BINARY_SEARCH`
//!   fixed-length arc nodes.** Real Lucene's `FSTCompiler` also emits
//!   fixed-length arc nodes for binary search (`ARCS_FOR_BINARY_SEARCH`),
//!   direct addressing (`ARCS_FOR_DIRECT_ADDRESSING`) and continuous ranges
//!   (`ARCS_FOR_CONTINUOUS`) once a node has "enough" arcs to make the
//!   space/speed tradeoff worth it. `ARCS_FOR_BINARY_SEARCH` is ported
//!   (`find_target_arc`'s sparse binary search over fixed-size arc slots,
//!   `read_arc`'s matching `BIT_TARGET_NEXT` target rule); reading
//!   `ARCS_FOR_DIRECT_ADDRESSING`/`ARCS_FOR_CONTINUOUS` nodes is not yet
//!   implemented -- encountering one still returns
//!   `Error::Unsupported("FST array-encoded node ...")` rather than silently
//!   misdecoding. Small term-index FSTs (few dozen distinct labels per node)
//!   normally stay in list form, but large/dense FSTs whose compiler picked
//!   one of the two remaining encodings for some node are a real gap -- see
//!   `docs/parity.md`.
//! - **Lookup only, not enumeration.** `get` mirrors `Util.get(FST, BytesRef)`:
//!   walk arcs for a specific key and return whether it's accepted plus its
//!   output. Full ordered enumeration (`BytesRefFSTEnum`/`IntsRefFSTEnum`) is
//!   not ported.
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
/// ("list") arc nodes -- see the module doc for why fixed-length (array)
/// node fields (`bytesPerArc`, `posArcsStart`, `arcIdx`, `numArcs`, the
/// direct-addressing bit-table fields) are omitted entirely.
#[derive(Debug, Clone, Default)]
pub struct Arc {
    label: i32,
    output: Vec<u8>,
    target: i64,
    flags: u8,
    next_final_output: Vec<u8>,
    next_arc: i64,
    /// `FST.Arc.bytesPerArc`: 0 for list-encoded (variable length arc) nodes;
    /// non-zero for `ARCS_FOR_BINARY_SEARCH` fixed-length-arc nodes (the only
    /// array encoding this port currently decodes -- see the module doc).
    bytes_per_arc: i32,
    /// `FST.Arc.numArcs`: only meaningful when `bytes_per_arc != 0`.
    num_arcs: i32,
    /// `FST.Arc.posArcsStart`: only meaningful when `bytes_per_arc != 0`.
    pos_arcs_start: i64,
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
    /// List-encoded nodes only (see module docs).
    fn read_arc(&self, arc: &mut Arc, r: &mut BytesReader) -> Result<()> {
        arc.label = self.read_label(r)?;

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
                    // `ARCS_FOR_BINARY_SEARCH` fixed-length-arc node: the
                    // target is simply the position right before the fixed
                    // arcs array starts (`FST.readArc`'s `bytesPerArc() != 0`
                    // branch) -- no scan needed since every arc's on-disk
                    // size is known.
                    r.set_position(
                        arc.pos_arcs_start - arc.bytes_per_arc as i64 * arc.num_arcs as i64,
                    );
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

    fn reject_if_array_node(flags: u8) -> Result<()> {
        if flags == ARCS_FOR_DIRECT_ADDRESSING || flags == ARCS_FOR_CONTINUOUS {
            return Err(Error::Unsupported(format!(
                "FST array-encoded node (flags={flags:#x}); only list-encoded (variable length arc) and ARCS_FOR_BINARY_SEARCH nodes are supported in this slice"
            )));
        }
        Ok(())
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

    /// `FST.findTargetArc`: find the arc leaving `follow` labeled
    /// `label_to_match`, or `None` if there is none. List-encoded and
    /// `ARCS_FOR_BINARY_SEARCH` nodes only (see module docs for
    /// `ARCS_FOR_DIRECT_ADDRESSING`/`ARCS_FOR_CONTINUOUS`, which remain
    /// rejected via `Error::Unsupported`).
    ///
    /// Note: real Lucene's `findTargetArc` also accepts `label_to_match ==
    /// END_LABEL` (used by `FSTEnum`/enumeration consumers to re-derive the
    /// "fake" accepting arc). `Util.get`'s loop -- the only caller here --
    /// never does that: it calls this once per actual key byte and then
    /// checks `arc.is_final()` directly (see `Fst::get`), so that branch
    /// is omitted as unreachable dead code in this slice's scope.
    /// `label_to_match` is therefore always a real byte value (0..=255).
    fn find_target_arc(
        &self,
        label_to_match: i32,
        follow: &Arc,
        r: &mut BytesReader,
    ) -> Result<Option<Arc>> {
        debug_assert!((0..=255).contains(&label_to_match));
        if !target_has_arcs(follow) {
            return Ok(None);
        }

        r.set_position(follow.target());
        let flags = r.read_byte()?;
        if flags == ARCS_FOR_BINARY_SEARCH {
            return self.find_target_arc_binary_search(label_to_match, r);
        }
        Self::reject_if_array_node(flags)?;

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
                Self::reject_if_array_node(arc.flags)?;
            }
        }
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
// - **Suffix sharing / minimization.** Two keys that happen to share a
//   *suffix* (not just a prefix) -- e.g. `"cat"` and `"bat"` sharing the
//   final `"at"` -- get two independent copies of that suffix's nodes here,
//   whereas real Lucene's node hash table would collapse them into one
//   shared node reachable from both `'c'` and `'b'` arcs. This only affects
//   output size/compactness, not correctness: `Fst::get` still resolves every
//   key to its correct output either way.
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

/// Serializes one trie node's children into the byte store, recursing into
/// any child that itself has further children first (post-order: a child
/// node's address must be known before the arc pointing at it can write its
/// `vlong` target). Returns this node's own address (the address of the arc
/// for its *smallest*-labeled child -- see the module doc's node/arc address
/// ordering contract). Panics only if called on a node with no children
/// (callers only recurse into/start from non-empty nodes).
fn build_node(node: &TrieNode, bytes: &mut Vec<u8>) -> i64 {
    let labels: Vec<u8> = node.children.keys().copied().collect();
    assert!(!labels.is_empty(), "build_node requires at least one arc");

    let mut child_addr: std::collections::HashMap<u8, i64> = std::collections::HashMap::new();
    for &label in &labels {
        let child = &node.children[&label];
        if !child.children.is_empty() {
            child_addr.insert(label, build_node(child, bytes));
        }
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
        build_node(&root, &mut bytes)
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

    #[test]
    fn direct_addressing_node_is_rejected_not_misdecoded() {
        // A node header byte equal to ARCS_FOR_DIRECT_ADDRESSING must be
        // rejected outright rather than silently misparsed -- this encoding
        // is not yet implemented (see module doc).
        let mut bytes = vec![0u8];
        let node_addr = bytes.len() as i64;
        bytes.push(ARCS_FOR_DIRECT_ADDRESSING);
        let fst = fst_from_body(bytes, node_addr, InputType::Byte1, None);
        let err = fst.get(b"a").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn continuous_arcs_node_is_rejected_not_misdecoded() {
        // Same, for ARCS_FOR_CONTINUOUS.
        let mut bytes = vec![0u8];
        let node_addr = bytes.len() as i64;
        bytes.push(ARCS_FOR_CONTINUOUS);
        let fst = fst_from_body(bytes, node_addr, InputType::Byte1, None);
        let err = fst.get(b"a").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
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
            let flags = BIT_LAST_ARC | BIT_FINAL_ARC | BIT_STOP_NODE | BIT_ARC_HAS_FINAL_OUTPUT;
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
}
