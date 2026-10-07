//! `org.apache.lucene.analysis.morph.TokenInfoFST`: a dictionary's
//! surface forms as an `FST<Long>` (`PositiveIntOutputs`), with the root
//! arcs of a label range cached.
//!
//! The reader is the read path of Lucene 10.5.0's `FST` for `Long` outputs:
//! `FST.readMetadata` and the `OnHeapFSTStore` body, `getFirstArc`,
//! `findTargetArc` over all four node encodings (variable-length arc lists,
//! `ARCS_FOR_BINARY_SEARCH`, `ARCS_FOR_DIRECT_ADDRESSING` with its presence
//! bit table, `ARCS_FOR_CONTINUOUS`), `ReverseBytesReader`, `BitTableUtil`
//! and `PositiveIntOutputs` (a vlong, `0` being no output), BYTE1/BYTE2
//! (both byte orders)/BYTE4 labels. Enumeration (`readNextArc`,
//! `readLastTargetArc`, `IntsRefFSTEnum`) is not ported: nothing walks a
//! dictionary's FST in order.
//!
//! Differs:
//! - `lucene-codecs` has a general FST port, but this crate sits on
//!   `lucene-util` only, and the Viterbi loop reads one arc per input unit,
//!   so this is a separate allocation-free reader of `Long`-output FSTs
//!   (an [`FstArc`] is `Copy`, its outputs plain `i64`s).
//! - A user dictionary is compiled by Java's `FSTCompiler` into an FST; here
//!   it is a trie over the same UTF-16 units ([`TokenInfoFst::from_sorted`])
//!   whose arcs carry no output and whose final arcs carry the entry's
//!   ordinal as their final output. Every walk sums to the same ordinal at
//!   the same arcs, which is all a lookup observes. A repeated key is
//!   `FSTCompiler`'s `UnsupportedOperationException` (`PositiveIntOutputs`
//!   cannot merge two outputs).
//! - Bytes outside the FST (a corrupt or truncated file) are an
//!   [`AnalysisError::Io`] where Java throws
//!   `ArrayIndexOutOfBoundsException`; every read is bounds-checked, every
//!   loop advances through the bytes, so a hostile FST ends in an error.

use super::resource::{io_error, ResourceInput};
use crate::AnalysisError;

/// `FST.FILE_FORMAT_NAME`.
const FILE_FORMAT_NAME: &str = "FST";
/// `FST.VERSION_START`.
const VERSION_START: i32 = 6;
/// `FST.VERSION_LITTLE_ENDIAN`.
const VERSION_LITTLE_ENDIAN: i32 = 8;
/// `FST.VERSION_CURRENT` (`VERSION_CONTINUOUS_ARCS`).
const VERSION_CURRENT: i32 = 9;

const BIT_FINAL_ARC: u8 = 1;
const BIT_LAST_ARC: u8 = 1 << 1;
const BIT_TARGET_NEXT: u8 = 1 << 2;
const BIT_STOP_NODE: u8 = 1 << 3;
const BIT_ARC_HAS_OUTPUT: u8 = 1 << 4;
const BIT_ARC_HAS_FINAL_OUTPUT: u8 = 1 << 5;
const ARCS_FOR_BINARY_SEARCH: u8 = BIT_ARC_HAS_FINAL_OUTPUT;
const ARCS_FOR_DIRECT_ADDRESSING: u8 = 1 << 6;
const ARCS_FOR_CONTINUOUS: u8 = ARCS_FOR_DIRECT_ADDRESSING + ARCS_FOR_BINARY_SEARCH;

/// `FST.END_LABEL`.
pub const END_LABEL: i32 = -1;
/// `FST.FINAL_END_NODE`.
const FINAL_END_NODE: i64 = -1;
/// `FST.NON_FINAL_END_NODE`.
const NON_FINAL_END_NODE: i64 = 0;

/// `FST.INPUT_TYPE`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InputType {
    /// `BYTE1`.
    Byte1,
    /// `BYTE2`.
    Byte2,
    /// `BYTE4`.
    Byte4,
}

/// `FST.Arc<Long>`: `Copy`, outputs as `i64` (`0` is `NO_OUTPUT`).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct FstArc {
    label: i32,
    output: i64,
    next_final_output: i64,
    target: i64,
    flags: u8,
    node_flags: u8,
    next_arc: i64,
    bytes_per_arc: i32,
    num_arcs: i32,
    pos_arcs_start: i64,
    arc_idx: i32,
    bit_table_start: i64,
    first_label: i32,
    presence_index: i32,
}

impl FstArc {
    /// `label()`.
    pub fn label(&self) -> i32 {
        self.label
    }
    /// `output()`.
    #[inline]
    pub fn output(&self) -> i64 {
        self.output
    }
    /// `nextFinalOutput()`.
    #[inline]
    pub fn next_final_output(&self) -> i64 {
        self.next_final_output
    }
    /// `target()`.
    pub fn target(&self) -> i64 {
        self.target
    }
    /// `isFinal()`.
    #[inline]
    pub fn is_final(&self) -> bool {
        self.flags & BIT_FINAL_ARC != 0
    }
    /// `isLast()`.
    pub fn is_last(&self) -> bool {
        self.flags & BIT_LAST_ARC != 0
    }
    fn flag(&self, bit: u8) -> bool {
        self.flags & bit != 0
    }
}

fn out_of_bounds(pos: i64) -> AnalysisError {
    io_error(
        "ArrayIndexOutOfBoundsException",
        format!("FST position {pos} is outside the FST"),
    )
}

/// `ReverseBytesReader`: reads go from `pos` down.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: i64,
}

impl Reader<'_> {
    #[inline]
    fn read_byte(&mut self) -> Result<u8, AnalysisError> {
        let b = usize::try_from(self.pos)
            .ok()
            .and_then(|p| self.bytes.get(p))
            .copied()
            .ok_or_else(|| out_of_bounds(self.pos))?;
        self.pos = self.pos.wrapping_sub(1);
        Ok(b)
    }

    fn skip_bytes(&mut self, count: i64) {
        self.pos = self.pos.saturating_sub(count);
    }

    /// `DataInput.readShort()`: little-endian in read order.
    #[inline]
    fn read_short(&mut self) -> Result<u16, AnalysisError> {
        // Bytes `pos` then `pos - 1`: one bounds check for both.
        let hi = usize::try_from(self.pos).map_err(|_| out_of_bounds(self.pos))?;
        let pair = hi
            .checked_sub(1)
            .and_then(|lo| self.bytes.get(lo..=hi))
            .ok_or_else(|| {
                // Java fails on the first byte it cannot read.
                if self.bytes.get(hi).is_none() {
                    out_of_bounds(self.pos)
                } else {
                    out_of_bounds(self.pos.wrapping_sub(1))
                }
            })?;
        self.pos = self.pos.wrapping_sub(2);
        Ok(u16::from_le_bytes([pair[1], pair[0]]))
    }

    fn read_vint(&mut self) -> Result<i32, AnalysisError> {
        let mut result: u32 = 0;
        for shift in [0u32, 7, 14, 21] {
            let b = self.read_byte()?;
            result |= u32::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(result as i32);
            }
        }
        let b = self.read_byte()?;
        result |= u32::from(b & 0x0F) << 28;
        if b & 0xF0 == 0 {
            return Ok(result as i32);
        }
        Err(io_error(
            "IOException",
            "Invalid vInt detected (too many bits)",
        ))
    }

    fn read_vlong(&mut self) -> Result<i64, AnalysisError> {
        let mut result: u64 = 0;
        for shift in [0u32, 7, 14, 21, 28, 35, 42, 49, 56] {
            let b = self.read_byte()?;
            result |= u64::from(b & 0x7F) << shift;
            if b & 0x80 == 0 {
                return Ok(result as i64);
            }
        }
        Err(io_error(
            "IOException",
            "Invalid vLong detected (negative values disallowed)",
        ))
    }
}

/// An `FST<Long>` as Lucene saves it: metadata, then the node bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
struct Fst {
    bytes: Vec<u8>,
    input_type: InputType,
    version: i32,
    start_node: i64,
    empty_output: Option<i64>,
}

/// `pos - idx * bytesPerArc` (and the like): position arithmetic in `i64`,
/// where Java's `long` math cannot overflow for an `int` index and width.
#[inline]
fn arc_pos(start: i64, idx: i32, bytes_per_arc: i32) -> i64 {
    start.wrapping_sub(i64::from(idx).wrapping_mul(i64::from(bytes_per_arc)))
}

impl Fst {
    /// `new FST<>(FST.readMetadata(in, PositiveIntOutputs), in)`.
    fn read(input: &mut ResourceInput<'_>) -> Result<Self, AnalysisError> {
        let version = input.check_header(FILE_FORMAT_NAME, VERSION_START, VERSION_CURRENT)?;
        let empty_output = if input.read_byte()? == 1 {
            let num_bytes = input.read_vint()?;
            let num_bytes = usize::try_from(num_bytes)
                .map_err(|_| io_error("NegativeArraySizeException", num_bytes))?;
            let empty = input.read_bytes(num_bytes)?;
            // De-serialize the empty-string output through a reverse reader
            // at its last byte.
            let mut r = Reader {
                bytes: empty,
                pos: i64::try_from(num_bytes).unwrap_or(0).wrapping_sub(1),
            };
            Some(r.read_vlong()?)
        } else {
            None
        };
        let input_type = match input.read_byte()? {
            0 => InputType::Byte1,
            1 => InputType::Byte2,
            2 => InputType::Byte4,
            t => {
                return Err(io_error(
                    "CorruptIndexException",
                    format!("invalid input type {}", t as i8),
                ))
            }
        };
        let start_node = input.read_vlong()?;
        let num_bytes = input.read_vlong()?;
        let num_bytes = usize::try_from(num_bytes)
            .ok()
            .filter(|&n| n <= input.remaining())
            .ok_or_else(|| io_error("EOFException", "read past EOF"))?;
        let bytes = input.read_bytes(num_bytes)?.to_vec();
        Ok(Fst {
            bytes,
            input_type,
            version,
            start_node,
            empty_output,
        })
    }

    fn reader(&self) -> Reader<'_> {
        Reader {
            bytes: &self.bytes,
            pos: 0,
        }
    }

    /// `readLabel`.
    #[inline]
    fn read_label(&self, r: &mut Reader<'_>) -> Result<i32, AnalysisError> {
        Ok(match self.input_type {
            InputType::Byte1 => i32::from(r.read_byte()?),
            InputType::Byte2 if self.version < VERSION_LITTLE_ENDIAN => {
                i32::from(r.read_short()?.swap_bytes())
            }
            InputType::Byte2 => i32::from(r.read_short()?),
            InputType::Byte4 => r.read_vint()?,
        })
    }

    /// `getFirstArc`.
    fn first_arc(&self) -> FstArc {
        let mut arc = FstArc::default();
        match self.empty_output {
            Some(out) => {
                arc.flags = BIT_FINAL_ARC | BIT_LAST_ARC;
                arc.next_final_output = out;
                if out != 0 {
                    arc.flags |= BIT_ARC_HAS_FINAL_OUTPUT;
                }
            }
            None => arc.flags = BIT_LAST_ARC,
        }
        arc.target = self.start_node;
        arc
    }

    /// `readPresenceBytes`.
    fn read_presence_bytes(arc: &mut FstArc, r: &mut Reader<'_>) {
        arc.bit_table_start = r.pos;
        r.skip_bytes(i64::from(num_presence_bytes(arc.num_arcs)));
    }

    /// `readFirstArcInfo` of a list node (the only kind `findTargetArc`
    /// scans linearly; the fixed-length-arc branch serves enumeration,
    /// which is not ported).
    fn read_first_arc_info(node: i64, arc: &mut FstArc) {
        arc.next_arc = node;
        arc.bytes_per_arc = 0;
    }

    /// `readNextRealArc` for the arc after `arc.arc_idx` of a binary-search
    /// or continuous node (the direct-addressing and list branches serve
    /// enumeration, which is not ported).
    fn read_next_real_arc(
        &self,
        arc: &mut FstArc,
        r: &mut Reader<'_>,
    ) -> Result<(), AnalysisError> {
        arc.arc_idx = arc.arc_idx.wrapping_add(1);
        r.pos = arc_pos(arc.pos_arcs_start, arc.arc_idx, arc.bytes_per_arc);
        arc.flags = r.read_byte()?;
        self.read_arc(arc, r)
    }

    /// `readArcByDirectAddressing(arc, in, rangeIndex, presenceIndex)`.
    fn read_arc_by_direct_addressing(
        &self,
        arc: &mut FstArc,
        r: &mut Reader<'_>,
        range_index: i32,
        presence_index: i32,
    ) -> Result<(), AnalysisError> {
        r.pos = arc_pos(arc.pos_arcs_start, presence_index, arc.bytes_per_arc);
        arc.arc_idx = range_index;
        arc.presence_index = presence_index;
        arc.flags = r.read_byte()?;
        self.read_arc(arc, r)
    }

    /// `readArc`: the flags byte has been read.
    fn read_arc(&self, arc: &mut FstArc, r: &mut Reader<'_>) -> Result<(), AnalysisError> {
        arc.label = if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING
            || arc.node_flags == ARCS_FOR_CONTINUOUS
        {
            arc.first_label.wrapping_add(arc.arc_idx)
        } else {
            self.read_label(r)?
        };
        arc.output = if arc.flag(BIT_ARC_HAS_OUTPUT) {
            r.read_vlong()?
        } else {
            0
        };
        arc.next_final_output = if arc.flag(BIT_ARC_HAS_FINAL_OUTPUT) {
            r.read_vlong()?
        } else {
            0
        };
        if arc.flag(BIT_STOP_NODE) {
            arc.target = if arc.flag(BIT_FINAL_ARC) {
                FINAL_END_NODE
            } else {
                NON_FINAL_END_NODE
            };
            arc.next_arc = r.pos;
        } else if arc.flag(BIT_TARGET_NEXT) {
            arc.next_arc = r.pos;
            if !arc.flag(BIT_LAST_ARC) {
                if arc.bytes_per_arc == 0 {
                    self.seek_to_next_node(r)?;
                } else {
                    let num_arcs = if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING {
                        count_bits(arc, r)?
                    } else {
                        arc.num_arcs
                    };
                    r.pos = arc_pos(arc.pos_arcs_start, num_arcs, arc.bytes_per_arc);
                }
            }
            arc.target = r.pos;
        } else {
            arc.target = r.read_vlong()?;
            arc.next_arc = r.pos;
        }
        Ok(())
    }

    /// `seekToNextNode`.
    fn seek_to_next_node(&self, r: &mut Reader<'_>) -> Result<(), AnalysisError> {
        loop {
            let flags = r.read_byte()?;
            self.read_label(r)?;
            if flags & BIT_ARC_HAS_OUTPUT != 0 {
                r.read_vlong()?;
            }
            if flags & BIT_ARC_HAS_FINAL_OUTPUT != 0 {
                r.read_vlong()?;
            }
            if flags & BIT_STOP_NODE == 0 && flags & BIT_TARGET_NEXT == 0 {
                r.read_vlong()?;
            }
            if flags & BIT_LAST_ARC != 0 {
                return Ok(());
            }
        }
    }

    /// `findTargetArc(labelToMatch, follow, arc, in)`.
    fn find_target_arc(
        &self,
        label: i32,
        follow: &FstArc,
    ) -> Result<Option<FstArc>, AnalysisError> {
        let mut arc = FstArc::default();
        if label == END_LABEL {
            if !follow.is_final() {
                return Ok(None);
            }
            if follow.target <= 0 {
                arc.flags = BIT_LAST_ARC;
            } else {
                arc.flags = 0;
                arc.next_arc = follow.target;
            }
            arc.output = follow.next_final_output;
            arc.label = END_LABEL;
            arc.node_flags = arc.flags;
            return Ok(Some(arc));
        }
        if follow.target <= 0 {
            return Ok(None);
        }
        let mut r = self.reader();
        r.pos = follow.target;
        let flags = r.read_byte()?;
        arc.node_flags = flags;
        if flags == ARCS_FOR_DIRECT_ADDRESSING {
            arc.num_arcs = r.read_vint()?;
            arc.bytes_per_arc = r.read_vint()?;
            Self::read_presence_bytes(&mut arc, &mut r);
            arc.first_label = self.read_label(&mut r)?;
            arc.pos_arcs_start = r.pos;
            let arc_index = label.wrapping_sub(arc.first_label);
            if arc_index < 0 || arc_index >= arc.num_arcs || !is_bit_set(arc_index, &arc, &mut r)? {
                return Ok(None);
            }
            let presence = count_bits_up_to(arc_index, &arc, &mut r)?;
            self.read_arc_by_direct_addressing(&mut arc, &mut r, arc_index, presence)?;
            return Ok(Some(arc));
        } else if flags == ARCS_FOR_BINARY_SEARCH {
            arc.num_arcs = r.read_vint()?;
            arc.bytes_per_arc = r.read_vint()?;
            arc.pos_arcs_start = r.pos;
            let (mut low, mut high) = (0i32, arc.num_arcs.wrapping_sub(1));
            while low <= high {
                // (low + high) >>> 1
                let mid = ((low as u32).wrapping_add(high as u32) >> 1) as i32;
                // +1 to skip over flags
                r.pos = arc_pos(arc.pos_arcs_start, mid, arc.bytes_per_arc).wrapping_sub(1);
                let mid_label = self.read_label(&mut r)?;
                let cmp = mid_label.wrapping_sub(label);
                if cmp < 0 {
                    low = mid.wrapping_add(1);
                } else if cmp > 0 {
                    high = mid.wrapping_sub(1);
                } else {
                    arc.arc_idx = mid.wrapping_sub(1);
                    self.read_next_real_arc(&mut arc, &mut r)?;
                    return Ok(Some(arc));
                }
            }
            return Ok(None);
        } else if flags == ARCS_FOR_CONTINUOUS {
            arc.num_arcs = r.read_vint()?;
            arc.bytes_per_arc = r.read_vint()?;
            arc.first_label = self.read_label(&mut r)?;
            arc.pos_arcs_start = r.pos;
            let arc_index = label.wrapping_sub(arc.first_label);
            if arc_index < 0 || arc_index >= arc.num_arcs {
                return Ok(None);
            }
            arc.arc_idx = arc_index.wrapping_sub(1);
            self.read_next_real_arc(&mut arc, &mut r)?;
            return Ok(Some(arc));
        }
        // Linear scan
        Self::read_first_arc_info(follow.target, &mut arc);
        r.pos = arc.next_arc;
        loop {
            arc.flags = r.read_byte()?;
            let flags = arc.flags;
            let pos = r.pos;
            let l = self.read_label(&mut r)?;
            if l == label {
                r.pos = pos;
                self.read_arc(&mut arc, &mut r)?;
                return Ok(Some(arc));
            } else if l > label || arc.is_last() {
                return Ok(None);
            }
            if flags & BIT_ARC_HAS_OUTPUT != 0 {
                r.read_vlong()?;
            }
            if flags & BIT_ARC_HAS_FINAL_OUTPUT != 0 {
                r.read_vlong()?;
            }
            if flags & BIT_STOP_NODE == 0 && flags & BIT_TARGET_NEXT == 0 {
                r.read_vlong()?;
            }
        }
    }
}

/// `FST.getNumPresenceBytes`.
fn num_presence_bytes(label_range: i32) -> i32 {
    label_range.wrapping_add(7) >> 3
}

/// `BitTable.isBitSet` / `BitTableUtil.isBitSet`.
fn is_bit_set(bit_index: i32, arc: &FstArc, r: &mut Reader<'_>) -> Result<bool, AnalysisError> {
    r.pos = arc.bit_table_start;
    r.skip_bytes(i64::from(bit_index >> 3));
    Ok(r.read_byte()? & (1u8 << (bit_index & 7)) != 0)
}

/// `BitTable.countBits` / `BitTableUtil.countBits`.
fn count_bits(arc: &FstArc, r: &mut Reader<'_>) -> Result<i32, AnalysisError> {
    r.pos = arc.bit_table_start;
    let bytes = num_presence_bytes(arc.num_arcs);
    let mut count = 0i32;
    for _ in 0..bytes.max(0) {
        count = count.wrapping_add(r.read_byte()?.count_ones() as i32);
    }
    Ok(count)
}

/// `BitTable.countBitsUpTo` / `BitTableUtil.countBitsUpTo`: the set bits
/// before `bit_index`.
fn count_bits_up_to(
    bit_index: i32,
    arc: &FstArc,
    r: &mut Reader<'_>,
) -> Result<i32, AnalysisError> {
    // The table is read backwards from `bit_table_start`: its first `full`
    // bytes are `bytes[start - full + 1..=start]`, so the whole-byte part is
    // one slice popcount (Java's `BitTableUtil` reads longs for the same
    // reason) rather than a checked read per byte.
    let full = i64::from(bit_index.max(0) >> 3);
    let start = arc.bit_table_start;
    let lo = start.wrapping_sub(full).wrapping_add(1);
    let slice = usize::try_from(lo)
        .ok()
        .zip(usize::try_from(start).ok())
        .and_then(|(lo, hi)| {
            if full == 0 {
                Some(&r.bytes[..0])
            } else {
                r.bytes.get(lo..=hi)
            }
        })
        .ok_or_else(|| out_of_bounds(lo))?;
    let mut count = 0u32;
    let mut chunks = slice.chunks_exact(8);
    for c in &mut chunks {
        let mut w = [0u8; 8];
        w.copy_from_slice(c);
        count = count.wrapping_add(u64::from_ne_bytes(w).count_ones());
    }
    for b in chunks.remainder() {
        count = count.wrapping_add(b.count_ones());
    }
    r.pos = start.wrapping_sub(full);
    let remaining = bit_index & 7;
    if remaining != 0 {
        let mask = (1u8 << remaining).wrapping_sub(1);
        count = count.wrapping_add((r.read_byte()? & mask).count_ones());
    }
    Ok(count as i32)
}

/// One node of a user dictionary's trie: its arcs by label, and the ordinal
/// of the entry ending here.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
struct TrieNode {
    labels: Vec<u16>,
    targets: Vec<u32>,
    ord: Option<i64>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Automaton {
    Fst(Fst),
    Trie(Vec<TrieNode>),
}

/// `TokenInfoFST`: the automaton and its root-arc cache.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenInfoFst {
    automaton: Automaton,
    cache_ceiling: i32,
    cache_floor: i32,
    /// Per cached label, `0` for no arc or `i + 1` for `root_arcs[i]`: a
    /// dense index (Java holds an `Arc` per slot) so the 28,608 Kuromoji
    /// slots cost 4 bytes each, and the arcs that exist sit together.
    root_index: Vec<u32>,
    root_arcs: Vec<FstArc>,
}

impl TokenInfoFst {
    /// `new TokenInfoFST(FST.read(in), cacheCeiling, cacheFloor)` over a
    /// `$fst.dat` file's bytes.
    pub fn read(bytes: &[u8], cache_ceiling: i32, cache_floor: i32) -> Result<Self, AnalysisError> {
        let fst = Fst::read(&mut ResourceInput::new(bytes))?;
        Self::new(Automaton::Fst(fst), cache_ceiling, cache_floor)
    }

    /// The trie of a user dictionary: `keys` in the order `FSTCompiler`
    /// receives them (sorted, as UTF-16 units), the `i`th mapping to `i`.
    pub fn from_sorted(
        keys: &[Vec<u16>],
        cache_ceiling: i32,
        cache_floor: i32,
    ) -> Result<Self, AnalysisError> {
        let mut nodes = vec![TrieNode::default()];
        for (ord, key) in keys.iter().enumerate() {
            let mut node = 0usize;
            for &unit in key {
                node = match nodes[node].labels.binary_search(&unit) {
                    Ok(i) => nodes[node].targets[i] as usize,
                    Err(i) => {
                        let child = nodes.len();
                        nodes[node].labels.insert(i, unit);
                        nodes[node].targets.insert(i, child as u32);
                        nodes.push(TrieNode::default());
                        child
                    }
                };
            }
            if nodes[node].ord.is_some() {
                return Err(AnalysisError::IllegalState(
                    "UnsupportedOperationException".to_string(),
                ));
            }
            nodes[node].ord = Some(ord as i64);
        }
        Self::new(Automaton::Trie(nodes), cache_ceiling, cache_floor)
    }

    fn new(
        automaton: Automaton,
        cache_ceiling: i32,
        cache_floor: i32,
    ) -> Result<Self, AnalysisError> {
        if cache_ceiling < cache_floor {
            return Err(AnalysisError::IllegalArgument(format!(
                "cacheCeiling must be larger than cacheFloor; cacheCeiling={cache_ceiling}, cacheFloor={cache_floor}"
            )));
        }
        let mut fst = TokenInfoFst {
            automaton,
            cache_ceiling,
            cache_floor,
            root_index: Vec::new(),
            root_arcs: Vec::new(),
        };
        // Java: cacheRootArcs
        let first = fst.first_arc();
        let (mut index, mut arcs) = (Vec::new(), Vec::new());
        for label in cache_floor..=cache_ceiling {
            match fst.find(label, &first)? {
                Some(arc) => {
                    arcs.push(arc);
                    index.push(u32::try_from(arcs.len()).unwrap_or(u32::MAX));
                }
                None => index.push(0),
            }
        }
        fst.root_index = index;
        fst.root_arcs = arcs;
        Ok(fst)
    }

    /// `getFirstArc`.
    pub fn first_arc(&self) -> FstArc {
        match &self.automaton {
            Automaton::Fst(f) => f.first_arc(),
            Automaton::Trie(nodes) => FstArc {
                flags: match nodes.first().and_then(|n| n.ord) {
                    Some(_) => BIT_FINAL_ARC | BIT_LAST_ARC,
                    None => BIT_LAST_ARC,
                },
                next_final_output: nodes.first().and_then(|n| n.ord).unwrap_or(0),
                target: 0,
                ..FstArc::default()
            },
        }
    }

    fn find(&self, label: i32, follow: &FstArc) -> Result<Option<FstArc>, AnalysisError> {
        match &self.automaton {
            Automaton::Fst(f) => f.find_target_arc(label, follow),
            Automaton::Trie(nodes) => {
                let (Ok(unit), Some(node)) = (
                    u16::try_from(label),
                    usize::try_from(follow.target)
                        .ok()
                        .and_then(|n| nodes.get(n)),
                ) else {
                    return Ok(None);
                };
                let Ok(i) = node.labels.binary_search(&unit) else {
                    return Ok(None);
                };
                let target = node.targets[i];
                let ord = nodes.get(target as usize).and_then(|n| n.ord);
                Ok(Some(FstArc {
                    label,
                    flags: if ord.is_some() { BIT_FINAL_ARC } else { 0 },
                    next_final_output: ord.unwrap_or(0),
                    target: i64::from(target),
                    ..FstArc::default()
                }))
            }
        }
    }

    /// `findTargetArc(ch, follow, arc, useCache, fstReader)`: the arc
    /// leaving `follow` with label `ch`; with `use_cache` and `ch` in the
    /// cached range, the cached root arc (whatever `follow` is, as in Java).
    #[inline]
    pub fn find_target_arc(
        &self,
        ch: i32,
        follow: &FstArc,
        use_cache: bool,
    ) -> Result<Option<FstArc>, AnalysisError> {
        if use_cache && ch >= self.cache_floor && ch <= self.cache_ceiling {
            let slot = usize::try_from(ch.wrapping_sub(self.cache_floor)).unwrap_or(usize::MAX);
            let i = self.root_index.get(slot).copied().unwrap_or(0);
            return Ok(usize::try_from(i)
                .ok()
                .and_then(|i| i.checked_sub(1))
                .and_then(|i| self.root_arcs.get(i))
                .copied());
        }
        self.find(ch, follow)
    }

    /// `FST.getEmptyOutput()` (`None` for a user dictionary's trie unless an
    /// entry is empty).
    pub fn empty_output(&self) -> Option<i64> {
        match &self.automaton {
            Automaton::Fst(f) => f.empty_output,
            Automaton::Trie(nodes) => nodes.first().and_then(|n| n.ord),
        }
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::morph::resource::test_util::{header, vlong};

    fn units(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    /// Walks `key` from the first arc; the output sum and final output at
    /// the last unit, `None` when an arc is missing.
    fn walk(f: &TokenInfoFst, key: &[u16]) -> Option<(i64, bool, i64)> {
        let mut arc = f.first_arc();
        let mut out = 0;
        for (i, &u) in key.iter().enumerate() {
            arc = f.find_target_arc(i32::from(u), &arc, i == 0).unwrap()?;
            out += arc.output();
        }
        Some((out, arc.is_final(), arc.next_final_output()))
    }

    #[test]
    fn trie_maps_keys_to_ordinals() {
        let keys: Vec<Vec<u16>> = ["ab", "abc", "b", "日本"]
            .iter()
            .map(|s| units(s))
            .collect();
        let f = TokenInfoFst::from_sorted(&keys, 0x30FF, 0x3040).unwrap();
        assert_eq!(walk(&f, &units("ab")), Some((0, true, 0)));
        assert_eq!(walk(&f, &units("abc")), Some((0, true, 1)));
        assert_eq!(walk(&f, &units("a")), Some((0, false, 0)));
        assert_eq!(walk(&f, &units("日本")), Some((0, true, 3)));
        assert_eq!(walk(&f, &units("x")), None);
        assert_eq!(walk(&f, &units("abcd")), None);
        assert!(!f.first_arc().is_final());
        assert_eq!(f.empty_output(), None);
        let first = f.first_arc();
        assert_eq!(f.find_target_arc(-1, &first, false).unwrap(), None);
        // Cached root arcs: a cached label answers whatever the follow arc.
        let k: Vec<Vec<u16>> = vec![units("あい")];
        let f = TokenInfoFst::from_sorted(&k, 0x30FF, 0x3040).unwrap();
        let a = f.find_target_arc(0x3042, &first, true).unwrap().unwrap();
        assert_eq!(f.find_target_arc(0x3042, &a, true).unwrap(), Some(a));
        assert_eq!(f.find_target_arc(0x3042, &a, false).unwrap(), None);
        assert_eq!(a.label(), 0x3042);
        assert!(!a.is_last());
        assert!(a.target() > 0);
        // An empty key is the empty output; a repeated key fails.
        let e = TokenInfoFst::from_sorted(&[vec![], units("a")], 1, 0).unwrap();
        assert!(e.first_arc().is_final());
        assert_eq!(e.empty_output(), Some(0));
        assert!(TokenInfoFst::from_sorted(&[units("a"), units("a")], 1, 0).is_err());
        assert!(TokenInfoFst::from_sorted(&[], 0, 1).is_err());
    }

    /// An FST file whose body is `body` (bytes in file order; the reader
    /// starts at `start`).
    fn fst_file(input_type: u8, empty: Option<u64>, start: u64, body: &[u8]) -> Vec<u8> {
        let mut b = header("FST", 9);
        match empty {
            Some(v) => {
                let mut e = Vec::new();
                vlong(&mut e, v);
                e.reverse();
                b.push(1);
                vlong(&mut b, e.len() as u64);
                b.extend_from_slice(&e);
            }
            None => b.push(0),
        }
        b.push(input_type);
        vlong(&mut b, start);
        vlong(&mut b, body.len() as u64);
        b.extend_from_slice(body);
        b
    }

    #[test]
    fn hand_built_list_node() {
        // One list node at 4..1 (read downward): arc 'a' (final, last, stop
        // node, output 5). Bytes in read order: flags, label, vlong output.
        let flags = BIT_FINAL_ARC | BIT_LAST_ARC | BIT_STOP_NODE | BIT_ARC_HAS_OUTPUT;
        let body = [0u8, 5, b'a', flags];
        let f = TokenInfoFst::read(&fst_file(0, Some(7), 3, &body), 0, 0).unwrap();
        assert_eq!(f.empty_output(), Some(7));
        let first = f.first_arc();
        assert!(first.is_final());
        assert_eq!(first.next_final_output(), 7);
        let a = f
            .find_target_arc(i32::from(b'a'), &first, false)
            .unwrap()
            .unwrap();
        assert_eq!(
            (a.output(), a.is_final(), a.target()),
            (5, true, FINAL_END_NODE)
        );
        assert_eq!(
            f.find_target_arc(i32::from(b'b'), &first, false).unwrap(),
            None
        );
        assert_eq!(f.find_target_arc(0, &first, false).unwrap(), None);
        assert_eq!(f.find_target_arc(i32::from(b'a'), &a, false).unwrap(), None);
        // END_LABEL off a final arc, with and without a target.
        let end = f
            .find_target_arc(END_LABEL, &first, false)
            .unwrap()
            .unwrap();
        assert_eq!(
            (end.label(), end.output(), end.is_last()),
            (END_LABEL, 7, false)
        );
        let end = f.find_target_arc(END_LABEL, &a, false).unwrap().unwrap();
        assert!(end.is_last());
        let mut not_final = first;
        not_final.flags = 0;
        assert_eq!(
            f.find_target_arc(END_LABEL, &not_final, false).unwrap(),
            None
        );
        // A zero empty output carries no final-output flag.
        let z = TokenInfoFst::read(&fst_file(0, Some(0), 3, &body), 0, 0).unwrap();
        assert_eq!(z.first_arc().flags & BIT_ARC_HAS_FINAL_OUTPUT, 0);
    }

    #[test]
    fn hand_built_target_next_arcs() {
        // Read order from the start node: arc 'a' (target-next, output 3,
        // final output 5), arc 'b' (last, stop, final), then the next node:
        // arc 'c' (last, stop, final, output 7).
        let read_order = [
            BIT_TARGET_NEXT | BIT_ARC_HAS_OUTPUT | BIT_ARC_HAS_FINAL_OUTPUT,
            b'a',
            3,
            5,
            BIT_LAST_ARC | BIT_STOP_NODE | BIT_FINAL_ARC,
            b'b',
            BIT_LAST_ARC | BIT_STOP_NODE | BIT_FINAL_ARC | BIT_ARC_HAS_OUTPUT,
            b'c',
            7,
        ];
        let mut body = read_order.to_vec();
        body.reverse();
        let f = TokenInfoFst::read(&fst_file(0, None, 8, &body), 0, 0).unwrap();
        let first = f.first_arc();
        let a = f
            .find_target_arc(i32::from(b'a'), &first, false)
            .unwrap()
            .unwrap();
        assert_eq!(
            (a.output(), a.next_final_output(), a.is_final()),
            (3, 5, false)
        );
        let c = f
            .find_target_arc(i32::from(b'c'), &a, false)
            .unwrap()
            .unwrap();
        assert_eq!((c.output(), c.is_final()), (7, true));
        let b = f
            .find_target_arc(i32::from(b'b'), &first, false)
            .unwrap()
            .unwrap();
        assert!(b.is_final() && b.is_last());
        assert_eq!(
            f.find_target_arc(i32::from(b'A'), &first, false).unwrap(),
            None
        );
        assert_eq!(
            f.find_target_arc(i32::from(b'z'), &first, false).unwrap(),
            None
        );
        // A non-final stop arc ends at no node.
        let mut odd = read_order.to_vec();
        odd[4] = BIT_LAST_ARC | BIT_STOP_NODE;
        odd.reverse();
        let g = TokenInfoFst::read(&fst_file(0, None, 8, &odd), 0, 0).unwrap();
        let b = g
            .find_target_arc(i32::from(b'b'), &g.first_arc(), false)
            .unwrap()
            .unwrap();
        assert_eq!((b.is_final(), b.target()), (false, NON_FINAL_END_NODE));
    }

    #[test]
    fn corrupt_files_fail_without_panicking() {
        let flags = BIT_FINAL_ARC | BIT_LAST_ARC | BIT_STOP_NODE | BIT_ARC_HAS_OUTPUT;
        let file = fst_file(1, None, 4, &[0u8, 5, b'a', 0, flags]);
        assert!(TokenInfoFst::read(&file, 0x100, 0).is_ok());
        for cut in 0..file.len() {
            let _ = TokenInfoFst::read(&file[..cut], 0x100, 0);
        }
        for i in 0..file.len() {
            let mut f = file.clone();
            f[i] ^= 0xFF;
            if let Ok(t) = TokenInfoFst::read(&f, 0x100, 0) {
                let _ = walk(&t, &units("a"));
            }
        }
        let bad_type = fst_file(7, None, 1, &[0]);
        let e = TokenInfoFst::read(&bad_type, 0, 0).unwrap_err();
        assert!(e.to_string().contains("invalid input type 7"), "{e}");
        let mut neg = header("FST", 9);
        neg.extend_from_slice(&[1, 0xFF, 0xFF, 0xFF, 0xFF, 0x0F]);
        assert!(TokenInfoFst::read(&neg, 0, 0).is_err());
        let start_outside = fst_file(0, None, 99, &[0, 0]);
        assert!(TokenInfoFst::read(&start_outside, 0x100, 0).is_err());
    }

    #[test]
    fn bit_table_helpers() {
        // A table of two bytes 0b0000_0101, 0b1000_0000 at positions 1, 0
        // (read downward from 1).
        let bytes = [0b1000_0000u8, 0b0000_0101];
        let arc = FstArc {
            bit_table_start: 1,
            num_arcs: 16,
            ..FstArc::default()
        };
        let mut r = Reader {
            bytes: &bytes,
            pos: 0,
        };
        assert!(is_bit_set(0, &arc, &mut r).unwrap());
        assert!(!is_bit_set(1, &arc, &mut r).unwrap());
        assert!(is_bit_set(15, &arc, &mut r).unwrap());
        assert_eq!(count_bits(&arc, &mut r).unwrap(), 3);
        assert_eq!(count_bits_up_to(2, &arc, &mut r).unwrap(), 1);
        assert_eq!(count_bits_up_to(8, &arc, &mut r).unwrap(), 2);
        assert_eq!(count_bits_up_to(15, &arc, &mut r).unwrap(), 2);
        assert_eq!(num_presence_bytes(9), 2);
    }

    #[test]
    fn readers_decode_labels_and_varints() {
        let bytes = [0x01u8, 0x02, 0x81, 0x01, 0xFF, 0xFF, 0xFF, 0xFF, 0x7F];
        let mut r = Reader {
            bytes: &bytes,
            pos: 1,
        };
        assert_eq!(r.read_short().unwrap(), 0x0102);
        let mut r = Reader {
            bytes: &bytes,
            pos: 1,
        };
        let fst = |input_type, version| Fst {
            bytes: Vec::new(),
            input_type,
            version,
            start_node: 0,
            empty_output: None,
        };
        assert_eq!(fst(InputType::Byte2, 7).read_label(&mut r).unwrap(), 0x0201);
        let mut r = Reader {
            bytes: &[0x01, 0x81],
            pos: 1,
        };
        assert_eq!(fst(InputType::Byte4, 9).read_label(&mut r).unwrap(), 0x81);
        let five = [0x1F, 0xFF, 0xFF, 0xFF, 0xFF];
        let mut r = Reader {
            bytes: &five,
            pos: 4,
        };
        assert!(r.read_vint().is_err());
        let ok = [0x0F, 0xFF, 0xFF, 0xFF, 0xFF];
        let mut r = Reader { bytes: &ok, pos: 4 };
        assert_eq!(r.read_vint().unwrap(), -1);
        let ten = [0xFF; 10];
        let mut r = Reader {
            bytes: &ten,
            pos: 9,
        };
        assert!(r.read_vlong().is_err());
        let mut r = Reader {
            bytes: &ten,
            pos: -1,
        };
        assert!(r.read_byte().is_err());
    }
}
