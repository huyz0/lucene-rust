//! Port of `org.apache.lucene.util.fst.FST` — read side only.
//!
//! This module implements enough of Lucene's FST (finite state transducer)
//! format to look up a byte-sequence key in an already-built, on-disk FST and
//! recover its accumulated output (`Util.get(fst, BytesRef)`). It does **not**
//! port FST *construction* (`FSTCompiler`) — that lives on the write path and
//! is a separate, much larger undertaking (see `docs/parity.md`).
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
//! - **Only variable-length ("list") arc nodes.** Real Lucene's `FSTCompiler`
//!   also emits fixed-length arc nodes for binary search
//!   (`ARCS_FOR_BINARY_SEARCH`), direct addressing (`ARCS_FOR_DIRECT_ADDRESSING`)
//!   and continuous ranges (`ARCS_FOR_CONTINUOUS`) once a node has "enough"
//!   arcs to make the space/speed tradeoff worth it. Reading those array-style
//!   nodes is not implemented; encountering one returns
//!   `Error::Unsupported("FST array-encoded node ...")` rather than silently
//!   misdecoding. Small term-index FSTs (few dozen distinct labels per node)
//!   normally stay in list form, but this is a real gap for large/dense FSTs
//!   -- see `docs/parity.md`.
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
#[derive(Debug, Clone)]
pub struct Arc {
    label: i32,
    output: Vec<u8>,
    target: i64,
    flags: u8,
    next_final_output: Vec<u8>,
    next_arc: i64,
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

/// An on-heap, `ByteSequenceOutputs`-typed FST, read from bytes written by
/// real Lucene's `FST.save`.
#[derive(Debug, Clone)]
pub struct Fst {
    metadata: FstMetadata,
    bytes: Vec<u8>,
}

impl Fst {
    /// Port of `FST.readMetadata` + the `FST(FSTMetadata, DataInput)`
    /// constructor's `OnHeapFSTStore` body read, both operating on the same
    /// forward cursor (matching `FST.save(Path)`/`FST.read(Path, Outputs)`,
    /// which write/read metadata and body through one stream).
    pub fn read(input: &mut SliceInput) -> Result<Fst> {
        let header =
            codec_util::check_header(input, FILE_FORMAT_NAME, VERSION_START, VERSION_CURRENT)?;
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
        let mut bytes = vec![0u8; num_bytes as usize];
        input.read_bytes(&mut bytes)?;

        Ok(Fst {
            metadata: FstMetadata {
                input_type,
                empty_output,
                start_node,
                version,
                num_bytes,
            },
            bytes,
        })
    }

    pub fn metadata(&self) -> &FstMetadata {
        &self.metadata
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
            label: 0,
            output: Vec::new(),
            target: self.metadata.start_node,
            flags,
            next_final_output,
            next_arc: 0,
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
                // Must scan past the remaining sibling arcs to find this
                // arc's target (list-encoded nodes only; see module docs
                // for why the fixed-length-array `bytesPerArc != 0` branch
                // is not ported).
                self.seek_to_next_node(r)?;
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
        if flags == ARCS_FOR_BINARY_SEARCH
            || flags == ARCS_FOR_DIRECT_ADDRESSING
            || flags == ARCS_FOR_CONTINUOUS
        {
            return Err(Error::Unsupported(format!(
                "FST array-encoded node (flags={flags:#x}); only list-encoded (variable length arc) nodes are supported in this slice"
            )));
        }
        Ok(())
    }

    /// `FST.findTargetArc`: find the arc leaving `follow` labeled
    /// `label_to_match`, or `None` if there is none. List-encoded nodes only.
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
        Self::reject_if_array_node(flags)?;

        // Linear scan (the only node encoding this slice supports).
        let mut arc = Arc {
            label: 0,
            output: Vec::new(),
            target: 0,
            flags,
            next_final_output: Vec::new(),
            next_arc: 0,
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
    ) -> Fst {
        Fst {
            metadata: FstMetadata {
                input_type,
                empty_output,
                start_node,
                version: VERSION_CURRENT,
                num_bytes: bytes.len() as i64,
            },
            bytes,
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
    fn array_encoded_node_is_rejected_not_misdecoded() {
        // A node header byte equal to ARCS_FOR_BINARY_SEARCH must be
        // rejected outright rather than silently misparsed as a list node.
        let mut bytes = vec![0u8];
        let node_addr = bytes.len() as i64;
        bytes.push(ARCS_FOR_BINARY_SEARCH);
        let fst = fst_from_body(bytes, node_addr, InputType::Byte1, None);
        let err = fst.get(b"a").unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
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
}
