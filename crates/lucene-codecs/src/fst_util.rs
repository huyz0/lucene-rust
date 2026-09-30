//! Port of `org.apache.lucene.util.fst.Util` over a typed FST reader.
//!
//! `crate::fst::Fst` decodes arcs as byte sequences only. The `Util`
//! algorithms need arcs decoded through any [`FstOutputs`] (`shortestPaths`
//! compares `Long` weights; suggesters use `PairOutputs`), so this module
//! carries [`TypedFst`], a port of `FST<T>`'s arc-reading surface
//! (`getFirstArc`, `readFirstTargetArc`, `readNextArc`, `findTargetArc`,
//! `readLastTargetArc`, the by-index/by-range readers for all three
//! fixed-length encodings, `BitTableUtil`), generic over the outputs. On top
//! of it: `Util.get`, `TopNSearcher`/`shortestPaths`, `readCeilArc`,
//! `binarySearch`, `toDot`, and the `toUTF16`/`toUTF32`/`toIntsRef`/
//! `toBytesRef` conversions.

use std::cmp::Ordering;
use std::fmt::Write as _;

use lucene_store::codec_util;
use lucene_store::data_input::{DataInput, SliceInput};

use crate::fst::{
    InputType, ARCS_FOR_BINARY_SEARCH, ARCS_FOR_CONTINUOUS, ARCS_FOR_DIRECT_ADDRESSING,
    BIT_ARC_HAS_FINAL_OUTPUT, BIT_ARC_HAS_OUTPUT, BIT_FINAL_ARC, BIT_LAST_ARC, BIT_STOP_NODE,
    BIT_TARGET_NEXT, FILE_FORMAT_NAME, FINAL_END_NODE, NON_FINAL_END_NODE, VERSION_CONTINUOUS_ARCS,
    VERSION_LITTLE_ENDIAN,
};
use crate::fst_compiler::{CompiledFst, FstOutputs, FstReadError, ReverseReader};

/// `FST.END_LABEL`.
pub const END_LABEL: i32 = -1;

type Result<T> = std::result::Result<T, FstReadError>;

fn corrupt(msg: impl Into<String>) -> FstReadError {
    FstReadError(msg.into())
}

/// `FST.Arc<T>`.
#[derive(Debug, Clone, PartialEq)]
pub struct Arc<V> {
    label: i32,
    output: V,
    target: i64,
    flags: u8,
    next_final_output: V,
    next_arc: i64,
    node_flags: u8,
    bytes_per_arc: i32,
    pos_arcs_start: i64,
    arc_idx: i32,
    num_arcs: i32,
    bit_table_start: i64,
    first_label: i32,
    presence_index: i32,
}

impl<V: Clone> Arc<V> {
    fn new(no_output: V) -> Self {
        Arc {
            label: 0,
            output: no_output.clone(),
            target: 0,
            flags: 0,
            next_final_output: no_output,
            next_arc: 0,
            node_flags: 0,
            bytes_per_arc: 0,
            pos_arcs_start: 0,
            arc_idx: 0,
            num_arcs: 0,
            bit_table_start: 0,
            first_label: 0,
            presence_index: 0,
        }
    }
    /// `label()`.
    pub fn label(&self) -> i32 {
        self.label
    }
    /// `output()`.
    pub fn output(&self) -> &V {
        &self.output
    }
    /// `target()`.
    pub fn target(&self) -> i64 {
        self.target
    }
    /// `flags()`.
    pub fn flags(&self) -> u8 {
        self.flags
    }
    /// `nextFinalOutput()`.
    pub fn next_final_output(&self) -> &V {
        &self.next_final_output
    }
    /// `isLast()`.
    pub fn is_last(&self) -> bool {
        self.flags & BIT_LAST_ARC != 0
    }
    /// `isFinal()`.
    pub fn is_final(&self) -> bool {
        self.flags & BIT_FINAL_ARC != 0
    }
    /// `nodeFlags()`.
    pub fn node_flags(&self) -> u8 {
        self.node_flags
    }
    /// `bytesPerArc()`.
    pub fn bytes_per_arc(&self) -> i32 {
        self.bytes_per_arc
    }
    /// `numArcs()` (the label range for direct addressing).
    pub fn num_arcs(&self) -> i32 {
        self.num_arcs
    }
    /// `arcIdx()`.
    pub fn arc_idx(&self) -> i32 {
        self.arc_idx
    }
    fn flag(&self, bit: u8) -> bool {
        self.flags & bit != 0
    }
}

/// `FST.targetHasArcs(arc)`.
pub fn target_has_arcs<V>(arc: &Arc<V>) -> bool {
    arc.target > 0
}

/// `FST<T>` over an on-heap body, decoding outputs through `O`.
pub struct TypedFst<O: FstOutputs> {
    input_type: InputType,
    empty_output: Option<O::Value>,
    start_node: i64,
    version: i32,
    bytes: Vec<u8>,
}

impl<O: FstOutputs> Clone for TypedFst<O> {
    fn clone(&self) -> Self {
        TypedFst {
            input_type: self.input_type,
            empty_output: self.empty_output.clone(),
            start_node: self.start_node,
            version: self.version,
            bytes: self.bytes.clone(),
        }
    }
}

impl<O: FstOutputs> std::fmt::Debug for TypedFst<O> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TypedFst")
            .field("input_type", &self.input_type)
            .field("empty_output", &self.empty_output)
            .field("start_node", &self.start_node)
            .field("version", &self.version)
            .field("num_bytes", &self.bytes.len())
            .finish()
    }
}

impl<O: FstOutputs> TypedFst<O> {
    /// An FST straight from the compiler.
    pub fn from_compiled(fst: CompiledFst<O::Value>) -> Self {
        TypedFst {
            input_type: fst.input_type,
            empty_output: fst.empty_output,
            start_node: fst.start_node,
            version: fst.version,
            bytes: fst.bytes,
        }
    }

    /// `FST.readMetadata` + the on-heap body (`FST.save(out, out)` bytes).
    pub fn read(saved: &[u8]) -> Result<Self> {
        let mut input = SliceInput::new(saved);
        let store = |e: lucene_store::error::Error| corrupt(e.to_string());
        let version = codec_util::check_header(
            &mut input,
            FILE_FORMAT_NAME,
            VERSION_LITTLE_ENDIAN,
            VERSION_CONTINUOUS_ARCS,
        )
        .map_err(store)?
        .version;
        let empty_output = if input.read_byte().map_err(store)? == 1 {
            let n = input.read_vint().map_err(store)?;
            let n = usize::try_from(n).map_err(|_| corrupt("negative empty-output length"))?;
            let mut buf = vec![0u8; n.min(input.len())];
            input.read_bytes(&mut buf).map_err(store)?;
            let mut r = ReverseReader::new(&buf);
            r.set_position((n as i64).saturating_sub(1));
            Some(O::read_final_output(&mut r)?)
        } else {
            None
        };
        let input_type = match input.read_byte().map_err(store)? {
            0 => InputType::Byte1,
            1 => InputType::Byte2,
            2 => InputType::Byte4,
            t => return Err(corrupt(format!("invalid input type {t}"))),
        };
        let start_node = input.read_vlong().map_err(store)?;
        let num_bytes = input.read_vlong().map_err(store)?;
        let remaining = input.len().saturating_sub(input.position());
        let n = usize::try_from(num_bytes)
            .ok()
            .filter(|&n| n <= remaining)
            .ok_or_else(|| corrupt(format!("numBytes {num_bytes} past the end")))?;
        let mut bytes = vec![0u8; n];
        input.read_bytes(&mut bytes).map_err(store)?;
        Ok(TypedFst {
            input_type,
            empty_output,
            start_node,
            version,
            bytes,
        })
    }

    /// `inputType`.
    pub fn input_type(&self) -> InputType {
        self.input_type
    }

    /// `emptyOutput`.
    pub fn empty_output(&self) -> Option<&O::Value> {
        self.empty_output.as_ref()
    }

    /// `getBytesReader()`.
    pub fn bytes_reader(&self) -> ReverseReader<'_> {
        ReverseReader::new(&self.bytes)
    }

    /// `readLabel(in)`.
    pub fn read_label(&self, input: &mut ReverseReader<'_>) -> Result<i32> {
        Ok(match self.input_type {
            InputType::Byte1 => i32::from(input.read_byte()?),
            InputType::Byte2 => i32::from(input.read_short()? as u16),
            InputType::Byte4 => input.read_vint()?,
        })
    }

    /// `getFirstArc(arc)`.
    pub fn first_arc(&self) -> Arc<O::Value> {
        let no = O::no_output();
        let mut arc = Arc::new(no.clone());
        match &self.empty_output {
            Some(e) => {
                arc.flags = BIT_FINAL_ARC | BIT_LAST_ARC;
                arc.next_final_output = e.clone();
                if *e != no {
                    arc.flags |= BIT_ARC_HAS_FINAL_OUTPUT;
                }
            }
            None => {
                arc.flags = BIT_LAST_ARC;
                arc.next_final_output = no.clone();
            }
        }
        arc.output = no;
        arc.target = self.start_node;
        arc
    }

    fn is_fixed(flags: u8) -> bool {
        flags == ARCS_FOR_BINARY_SEARCH
            || flags == ARCS_FOR_DIRECT_ADDRESSING
            || flags == ARCS_FOR_CONTINUOUS
    }

    // ARITH: `(numArcs + 7) >> 3` over a vInt read from the body.
    #[allow(clippy::arithmetic_side_effects)]
    fn num_presence_bytes(num_arcs: i32) -> i64 {
        (i64::from(num_arcs) + 7) >> 3
    }

    fn read_presence_bytes(&self, arc: &mut Arc<O::Value>, input: &mut ReverseReader<'_>) {
        arc.bit_table_start = input.position();
        input.skip_bytes(Self::num_presence_bytes(arc.num_arcs));
    }

    /// `readLastTargetArc(follow, arc, in)`.
    // ARITH: `numArcs - 2` over a node's own header.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn read_last_target_arc(
        &self,
        follow: &Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<Arc<O::Value>> {
        let mut arc = Arc::new(O::no_output());
        if !target_has_arcs(follow) {
            arc.label = END_LABEL;
            arc.target = FINAL_END_NODE;
            arc.output = follow.next_final_output.clone();
            arc.flags = BIT_LAST_ARC;
            arc.node_flags = arc.flags;
            return Ok(arc);
        }
        input.set_position(follow.target);
        let flags = input.read_byte()?;
        arc.node_flags = flags;
        if Self::is_fixed(flags) {
            arc.num_arcs = input.read_vint()?;
            arc.bytes_per_arc = input.read_vint()?;
            if flags == ARCS_FOR_DIRECT_ADDRESSING {
                self.read_presence_bytes(&mut arc, input);
                arc.first_label = self.read_label(input)?;
                arc.pos_arcs_start = input.position();
                self.read_last_arc_by_direct_addressing(&mut arc, input)?;
            } else if flags == ARCS_FOR_BINARY_SEARCH {
                arc.arc_idx = arc.num_arcs - 2;
                arc.pos_arcs_start = input.position();
                self.read_next_real_arc(&mut arc, input)?;
            } else {
                arc.first_label = self.read_label(input)?;
                arc.pos_arcs_start = input.position();
                self.read_last_arc_by_continuous(&mut arc, input)?;
            }
        } else {
            arc.flags = flags;
            arc.bytes_per_arc = 0;
            while !arc.is_last() {
                self.read_label(input)?;
                if arc.flag(BIT_ARC_HAS_OUTPUT) {
                    O::read(input)?;
                }
                if arc.flag(BIT_ARC_HAS_FINAL_OUTPUT) {
                    O::read_final_output(input)?;
                }
                if !arc.flag(BIT_STOP_NODE) && !arc.flag(BIT_TARGET_NEXT) {
                    input.read_vlong()?;
                }
                arc.flags = input.read_byte()?;
            }
            input.skip_bytes(-1);
            arc.next_arc = input.position();
            self.read_next_real_arc(&mut arc, input)?;
        }
        Ok(arc)
    }

    /// `readFirstTargetArc(follow, arc, in)`.
    pub fn read_first_target_arc(
        &self,
        follow: &Arc<O::Value>,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<()> {
        if follow.is_final() {
            arc.label = END_LABEL;
            arc.output = follow.next_final_output.clone();
            arc.flags = BIT_FINAL_ARC;
            if follow.target <= 0 {
                arc.flags |= BIT_LAST_ARC;
            } else {
                arc.next_arc = follow.target;
            }
            arc.target = FINAL_END_NODE;
            arc.node_flags = arc.flags;
            Ok(())
        } else {
            self.read_first_real_target_arc(follow.target, arc, input)
        }
    }

    fn read_first_arc_info(
        &self,
        node_address: i64,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<()> {
        input.set_position(node_address);
        let flags = input.read_byte()?;
        arc.node_flags = flags;
        if Self::is_fixed(flags) {
            arc.num_arcs = input.read_vint()?;
            arc.bytes_per_arc = input.read_vint()?;
            arc.arc_idx = -1;
            if flags == ARCS_FOR_DIRECT_ADDRESSING {
                self.read_presence_bytes(arc, input);
                arc.first_label = self.read_label(input)?;
                arc.presence_index = -1;
            } else if flags == ARCS_FOR_CONTINUOUS {
                arc.first_label = self.read_label(input)?;
            }
            arc.pos_arcs_start = input.position();
        } else {
            arc.next_arc = node_address;
            arc.bytes_per_arc = 0;
        }
        Ok(())
    }

    /// `readFirstRealTargetArc(nodeAddress, arc, in)`.
    pub fn read_first_real_target_arc(
        &self,
        node_address: i64,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<()> {
        self.read_first_arc_info(node_address, arc, input)?;
        self.read_next_real_arc(arc, input)
    }

    /// `isExpandedTarget(follow, in)`.
    pub fn is_expanded_target(
        &self,
        follow: &Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<bool> {
        if !target_has_arcs(follow) {
            return Ok(false);
        }
        input.set_position(follow.target);
        Ok(Self::is_fixed(input.read_byte()?))
    }

    /// `readNextArc(arc, in)`.
    pub fn read_next_arc(
        &self,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<()> {
        if arc.label == END_LABEL {
            if arc.next_arc <= 0 {
                return Err(corrupt("cannot readNextArc when arc.isLast()=true"));
            }
            let next = arc.next_arc;
            self.read_first_real_target_arc(next, arc, input)
        } else {
            self.read_next_real_arc(arc, input)
        }
    }

    /// `readNextArcLabel(arc, in)`: the label of the arc after `arc`.
    // ARITH: arc slot offsets inside one node.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn read_next_arc_label(
        &self,
        arc: &Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<i32> {
        if arc.label == END_LABEL {
            input.set_position(arc.next_arc);
            let flags = input.read_byte()?;
            if Self::is_fixed(flags) {
                let num_arcs = input.read_vint()?;
                input.read_vint()?;
                if flags == ARCS_FOR_BINARY_SEARCH {
                    input.read_byte()?;
                } else if flags == ARCS_FOR_DIRECT_ADDRESSING {
                    input.skip_bytes(Self::num_presence_bytes(num_arcs));
                }
            }
        } else {
            match arc.node_flags {
                ARCS_FOR_BINARY_SEARCH => input.set_position(
                    arc.pos_arcs_start
                        - (1 + i64::from(arc.arc_idx)) * i64::from(arc.bytes_per_arc)
                        - 1,
                ),
                ARCS_FOR_DIRECT_ADDRESSING => {
                    let next = self.bit_next_set(arc.arc_idx, arc, input)?;
                    if next < 0 {
                        return Err(corrupt("no arc after the last present arc"));
                    }
                    return Ok(arc.first_label.wrapping_add(next));
                }
                ARCS_FOR_CONTINUOUS => {
                    return Ok(arc.first_label.wrapping_add(arc.arc_idx).wrapping_add(1));
                }
                _ => input.set_position(arc.next_arc - 1),
            }
        }
        self.read_label(input)
    }

    /// `readArcByIndex(arc, in, idx)` (binary-search nodes).
    // ARITH: `idx * bytesPerArc` over the node's own header values, in i64.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn read_arc_by_index(
        &self,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
        idx: i32,
    ) -> Result<()> {
        input.set_position(arc.pos_arcs_start - i64::from(idx) * i64::from(arc.bytes_per_arc));
        arc.arc_idx = idx;
        arc.flags = input.read_byte()?;
        self.read_arc(arc, input)
    }

    /// `readArcByContinuous(arc, in, rangeIndex)`.
    pub fn read_arc_by_continuous(
        &self,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
        range_index: i32,
    ) -> Result<()> {
        self.read_arc_by_index(arc, input, range_index)
    }

    /// `readArcByDirectAddressing(arc, in, rangeIndex)`.
    pub fn read_arc_by_direct_addressing(
        &self,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
        range_index: i32,
    ) -> Result<()> {
        let presence_index = self.bit_count_up_to(range_index, arc, input)?;
        self.read_arc_by_direct_addressing_at(arc, input, range_index, presence_index)
    }

    // ARITH: `presenceIndex * bytesPerArc` in i64.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_arc_by_direct_addressing_at(
        &self,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
        range_index: i32,
        presence_index: i32,
    ) -> Result<()> {
        input.set_position(
            arc.pos_arcs_start - i64::from(presence_index) * i64::from(arc.bytes_per_arc),
        );
        arc.arc_idx = range_index;
        arc.presence_index = presence_index;
        arc.flags = input.read_byte()?;
        self.read_arc(arc, input)
    }

    /// `readLastArcByDirectAddressing(arc, in)`.
    pub fn read_last_arc_by_direct_addressing(
        &self,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<()> {
        let presence_index = self.bit_count(arc, input)?.wrapping_sub(1);
        let last = arc.num_arcs.wrapping_sub(1);
        self.read_arc_by_direct_addressing_at(arc, input, last, presence_index)
    }

    /// `readLastArcByContinuous(arc, in)`.
    pub fn read_last_arc_by_continuous(
        &self,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<()> {
        let last = arc.num_arcs.wrapping_sub(1);
        self.read_arc_by_continuous(arc, input, last)
    }

    /// `readNextRealArc(arc, in)`.
    // ARITH: `arcIdx * bytesPerArc` in i64; `arcIdx + 1` stays below
    // `numArcs` for a well-formed walk (a corrupt one reads out of range and
    // errors).
    #[allow(clippy::arithmetic_side_effects)]
    pub fn read_next_real_arc(
        &self,
        arc: &mut Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<()> {
        match arc.node_flags {
            ARCS_FOR_BINARY_SEARCH | ARCS_FOR_CONTINUOUS => {
                arc.arc_idx = arc.arc_idx.wrapping_add(1);
                if arc.arc_idx < 0 || arc.arc_idx >= arc.num_arcs {
                    return Err(corrupt("read past the last arc"));
                }
                input.set_position(
                    arc.pos_arcs_start - i64::from(arc.arc_idx) * i64::from(arc.bytes_per_arc),
                );
                arc.flags = input.read_byte()?;
            }
            ARCS_FOR_DIRECT_ADDRESSING => {
                let next_index = self.bit_next_set(arc.arc_idx, arc, input)?;
                if next_index < 0 {
                    return Err(corrupt("read past the last arc"));
                }
                let presence = arc.presence_index.wrapping_add(1);
                return self.read_arc_by_direct_addressing_at(arc, input, next_index, presence);
            }
            _ => {
                input.set_position(arc.next_arc);
                arc.flags = input.read_byte()?;
            }
        }
        self.read_arc(arc, input)
    }

    // ARITH: target and slot offsets inside one node, in i64.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_arc(&self, arc: &mut Arc<O::Value>, input: &mut ReverseReader<'_>) -> Result<()> {
        if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING || arc.node_flags == ARCS_FOR_CONTINUOUS {
            arc.label = arc.first_label.wrapping_add(arc.arc_idx);
        } else {
            arc.label = self.read_label(input)?;
        }
        arc.output = if arc.flag(BIT_ARC_HAS_OUTPUT) {
            O::read(input)?
        } else {
            O::no_output()
        };
        arc.next_final_output = if arc.flag(BIT_ARC_HAS_FINAL_OUTPUT) {
            O::read_final_output(input)?
        } else {
            O::no_output()
        };
        if arc.flag(BIT_STOP_NODE) {
            arc.target = if arc.flag(BIT_FINAL_ARC) {
                FINAL_END_NODE
            } else {
                NON_FINAL_END_NODE
            };
            arc.next_arc = input.position();
        } else if arc.flag(BIT_TARGET_NEXT) {
            arc.next_arc = input.position();
            if !arc.flag(BIT_LAST_ARC) {
                if arc.bytes_per_arc == 0 {
                    self.seek_to_next_node(input)?;
                } else {
                    let num_arcs = if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING {
                        self.bit_count(arc, input)?
                    } else {
                        arc.num_arcs
                    };
                    input.set_position(
                        arc.pos_arcs_start - i64::from(arc.bytes_per_arc) * i64::from(num_arcs),
                    );
                }
            }
            arc.target = input.position();
        } else {
            arc.target = input.read_vlong()?;
            arc.next_arc = input.position();
        }
        Ok(())
    }

    /// `readEndArc(follow, arc)`: `None` when `follow` is not final.
    pub fn read_end_arc(follow: &Arc<O::Value>) -> Option<Arc<O::Value>> {
        if !follow.is_final() {
            return None;
        }
        let mut arc = Arc::new(O::no_output());
        if follow.target <= 0 {
            arc.flags = BIT_LAST_ARC;
        } else {
            arc.flags = 0;
            arc.next_arc = follow.target;
        }
        arc.output = follow.next_final_output.clone();
        arc.label = END_LABEL;
        Some(arc)
    }

    /// `findTargetArc(labelToMatch, follow, arc, in)`: the arc leaving
    /// `follow`'s target with `label`, or `None`.
    // ARITH: binary search midpoints and slot offsets in i64.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn find_target_arc(
        &self,
        label: i32,
        follow: &Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<Option<Arc<O::Value>>> {
        let mut arc = Arc::new(O::no_output());
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
            arc.output = follow.next_final_output.clone();
            arc.label = END_LABEL;
            arc.node_flags = arc.flags;
            return Ok(Some(arc));
        }
        if !target_has_arcs(follow) {
            return Ok(None);
        }
        input.set_position(follow.target);
        let flags = input.read_byte()?;
        arc.node_flags = flags;
        if flags == ARCS_FOR_DIRECT_ADDRESSING {
            arc.num_arcs = input.read_vint()?;
            arc.bytes_per_arc = input.read_vint()?;
            self.read_presence_bytes(&mut arc, input);
            arc.first_label = self.read_label(input)?;
            arc.pos_arcs_start = input.position();
            let idx = i64::from(label) - i64::from(arc.first_label);
            if idx < 0 || idx >= i64::from(arc.num_arcs) {
                return Ok(None);
            }
            if !self.bit_is_set(idx as i32, &arc, input)? {
                return Ok(None);
            }
            self.read_arc_by_direct_addressing(&mut arc, input, idx as i32)?;
            return Ok(Some(arc));
        } else if flags == ARCS_FOR_BINARY_SEARCH {
            arc.num_arcs = input.read_vint()?;
            arc.bytes_per_arc = input.read_vint()?;
            arc.pos_arcs_start = input.position();
            let mut low = 0i64;
            let mut high = i64::from(arc.num_arcs) - 1;
            while low <= high {
                let mid = (low + high) >> 1;
                input.set_position(arc.pos_arcs_start - (i64::from(arc.bytes_per_arc) * mid + 1));
                let mid_label = self.read_label(input)?;
                match mid_label.cmp(&label) {
                    Ordering::Less => low = mid + 1,
                    Ordering::Greater => high = mid - 1,
                    Ordering::Equal => {
                        arc.arc_idx = (mid - 1) as i32;
                        self.read_next_real_arc(&mut arc, input)?;
                        return Ok(Some(arc));
                    }
                }
            }
            return Ok(None);
        } else if flags == ARCS_FOR_CONTINUOUS {
            arc.num_arcs = input.read_vint()?;
            arc.bytes_per_arc = input.read_vint()?;
            arc.first_label = self.read_label(input)?;
            arc.pos_arcs_start = input.position();
            let idx = i64::from(label) - i64::from(arc.first_label);
            if idx < 0 || idx >= i64::from(arc.num_arcs) {
                return Ok(None);
            }
            arc.arc_idx = (idx - 1) as i32;
            self.read_next_real_arc(&mut arc, input)?;
            return Ok(Some(arc));
        }
        self.read_first_arc_info(follow.target, &mut arc, input)?;
        input.set_position(arc.next_arc);
        loop {
            let flags = input.read_byte()?;
            arc.flags = flags;
            let pos = input.position();
            let l = self.read_label(input)?;
            if l == label {
                input.set_position(pos);
                self.read_arc(&mut arc, input)?;
                return Ok(Some(arc));
            } else if l > label || arc.is_last() {
                return Ok(None);
            }
            if flags & BIT_ARC_HAS_OUTPUT != 0 {
                O::read(input)?;
            }
            if flags & BIT_ARC_HAS_FINAL_OUTPUT != 0 {
                O::read_final_output(input)?;
            }
            if flags & BIT_STOP_NODE == 0 && flags & BIT_TARGET_NEXT == 0 {
                input.read_vlong()?;
            }
        }
    }

    fn seek_to_next_node(&self, input: &mut ReverseReader<'_>) -> Result<()> {
        loop {
            let flags = input.read_byte()?;
            self.read_label(input)?;
            if flags & BIT_ARC_HAS_OUTPUT != 0 {
                O::read(input)?;
            }
            if flags & BIT_ARC_HAS_FINAL_OUTPUT != 0 {
                O::read_final_output(input)?;
            }
            if flags & BIT_STOP_NODE == 0 && flags & BIT_TARGET_NEXT == 0 {
                input.read_vlong()?;
            }
            if flags & BIT_LAST_ARC != 0 {
                return Ok(());
            }
        }
    }

    // --- BitTableUtil --------------------------------------------------

    fn bit_is_set(
        &self,
        bit: i32,
        arc: &Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<bool> {
        input.set_position(arc.bit_table_start);
        input.skip_bytes(i64::from(bit >> 3));
        Ok(input.read_byte()? & (1u8 << (bit & 7)) != 0)
    }

    // ARITH: popcounts of at most `numArcs` bits.
    #[allow(clippy::arithmetic_side_effects)]
    fn bit_count(&self, arc: &Arc<O::Value>, input: &mut ReverseReader<'_>) -> Result<i32> {
        input.set_position(arc.bit_table_start);
        let n = Self::num_presence_bytes(arc.num_arcs);
        let mut count = 0i32;
        for _ in 0..n {
            count += input.read_byte()?.count_ones() as i32;
        }
        Ok(count)
    }

    // ARITH: popcounts of at most `bit` bits.
    #[allow(clippy::arithmetic_side_effects)]
    fn bit_count_up_to(
        &self,
        bit: i32,
        arc: &Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<i32> {
        input.set_position(arc.bit_table_start);
        let mut count = 0i32;
        let full = bit >> 3;
        for _ in 0..full {
            count += input.read_byte()?.count_ones() as i32;
        }
        let rem = bit & 7;
        if rem != 0 {
            count += (input.read_byte()? & ((1u8 << rem) - 1)).count_ones() as i32;
        }
        Ok(count)
    }

    /// `BitTableUtil.nextBitSet`: the next set bit after `bit`, or -1.
    //
    // SENTINEL: `-1` = "no next present arc"; all three callers turn it into
    // an error or `None` before it can become an `arcIdx` (the c31 class in
    // `fst.rs`'s `bit_table_next_bit_set`).
    // ARITH: byte indices over the node's presence table.
    #[allow(clippy::arithmetic_side_effects)]
    fn bit_next_set(
        &self,
        bit: i32,
        arc: &Arc<O::Value>,
        input: &mut ReverseReader<'_>,
    ) -> Result<i32> {
        input.set_position(arc.bit_table_start);
        let table_bytes = Self::num_presence_bytes(arc.num_arcs);
        let mut byte_index = i64::from(bit.div_euclid(8).max(0));
        if bit == -1 {
            byte_index = 0;
        }
        let mask: i32 = -1i32 << ((bit + 1) & 7);
        let mut i: i32;
        if mask == -1 && bit != -1 {
            input.skip_bytes(byte_index + 1);
            byte_index += 1;
            if byte_index == table_bytes {
                return Ok(-1);
            }
            i = i32::from(input.read_byte()?);
        } else {
            input.skip_bytes(byte_index);
            i = i32::from(input.read_byte()?) & mask;
        }
        while i == 0 {
            byte_index += 1;
            if byte_index == table_bytes {
                return Ok(-1);
            }
            i = i32::from(input.read_byte()?);
        }
        Ok(i.trailing_zeros() as i32 + (byte_index as i32) * 8)
    }
}

// --- Util ------------------------------------------------------------------

/// `Util.get(fst, IntsRef)`.
pub fn get<O: FstOutputs>(fst: &TypedFst<O>, input: &[i32]) -> Result<Option<O::Value>> {
    let mut arc = fst.first_arc();
    let mut r = fst.bytes_reader();
    let mut output = O::no_output();
    for &label in input {
        match fst.find_target_arc(label, &arc, &mut r)? {
            None => return Ok(None),
            Some(next) => arc = next,
        }
        output = O::add(&output, &arc.output);
    }
    Ok(arc
        .is_final()
        .then(|| O::add(&output, &arc.next_final_output)))
}

/// `Util.get(fst, BytesRef)`.
pub fn get_bytes<O: FstOutputs>(fst: &TypedFst<O>, input: &[u8]) -> Result<Option<O::Value>> {
    get(fst, &to_ints_ref(input))
}

/// `Util.FSTPath<T>`.
#[derive(Debug, Clone)]
pub struct FstPath<V> {
    /// `arc`.
    pub arc: Arc<V>,
    /// `output`.
    pub output: V,
    /// `input`.
    pub input: Vec<i32>,
    /// `boost`.
    pub boost: f32,
    /// `context`.
    pub context: Option<String>,
    /// `payload`.
    pub payload: i32,
}

/// `Util.Result<T>`.
#[derive(Debug, Clone, PartialEq)]
pub struct TopResult<V> {
    /// `input`.
    pub input: Vec<i32>,
    /// `output`.
    pub output: V,
}

/// `Util.TopResults<T>`.
#[derive(Debug, Clone, PartialEq)]
pub struct TopResults<V> {
    /// `isComplete`.
    pub is_complete: bool,
    /// `topN`.
    pub top_n: Vec<TopResult<V>>,
}

type Cmp<'c, V> = Box<dyn Fn(&V, &V) -> Ordering + 'c>;
type PathCmp<'c, V> = Box<dyn Fn(&FstPath<V>, &FstPath<V>) -> Ordering + 'c>;

/// `Util.TopNSearcher<T>`: best-first search for the `top_n` smallest
/// outputs. `accept_partial_path`/`accept_result` are Java's protected
/// hooks.
pub struct TopNSearcher<'f, 'c, O: FstOutputs> {
    fst: &'f TypedFst<O>,
    top_n: usize,
    max_queue_depth: usize,
    comparator: Cmp<'c, O::Value>,
    path_comparator: PathCmp<'c, O::Value>,
    /// `TreeSet<FSTPath>` ordered by `path_comparator`; `None` once Java sets
    /// it to null.
    queue: Option<Vec<FstPath<O::Value>>>,
    /// `acceptPartialPath(path)`.
    pub accept_partial_path: Option<AcceptPartialPath<'c, O::Value>>,
    /// `acceptResult(input, output)`.
    pub accept_result: Option<AcceptResult<'c, O::Value>>,
}

/// `TopNSearcher.acceptPartialPath`'s override.
pub type AcceptPartialPath<'c, V> = Box<dyn FnMut(&FstPath<V>) -> bool + 'c>;
/// `TopNSearcher.acceptResult(IntsRef, T)`'s override.
pub type AcceptResult<'c, V> = Box<dyn FnMut(&[i32], &V) -> bool + 'c>;

impl<'f, 'c, O: FstOutputs + 'c> TopNSearcher<'f, 'c, O> {
    /// `new TopNSearcher(fst, topN, maxQueueDepth, comparator)`: ties broken
    /// by input.
    pub fn new(
        fst: &'f TypedFst<O>,
        top_n: usize,
        max_queue_depth: usize,
        comparator: impl Fn(&O::Value, &O::Value) -> Ordering + Clone + 'c,
    ) -> Self {
        let c2 = comparator.clone();
        TopNSearcher {
            fst,
            top_n,
            max_queue_depth,
            comparator: Box::new(comparator),
            path_comparator: Box::new(move |a, b| {
                c2(&a.output, &b.output).then_with(|| a.input.cmp(&b.input))
            }),
            queue: Some(Vec::new()),
            accept_partial_path: None,
            accept_result: None,
        }
    }

    fn accept_partial(&mut self, path: &FstPath<O::Value>) -> bool {
        self.accept_partial_path.as_mut().is_none_or(|f| f(path))
    }

    /// `addIfCompetitive(path)`.
    // ARITH: `max_queue_depth + 1` for a caller-chosen in-memory depth.
    #[allow(clippy::arithmetic_side_effects)]
    fn add_if_competitive(&mut self, path: &FstPath<O::Value>) {
        let output = O::add(&path.output, &path.arc.output);
        let Some(queue) = &self.queue else { return };
        if queue.len() == self.max_queue_depth {
            if let Some(bottom) = queue.last() {
                let comp = (self.path_comparator)(path, bottom);
                if comp == Ordering::Greater {
                    return;
                } else if comp == Ordering::Equal {
                    let mut extended = path.input.clone();
                    extended.push(path.arc.label);
                    if bottom.input.as_slice() < extended.as_slice() {
                        return;
                    }
                }
            }
        }
        let mut new_input = path.input.clone();
        new_input.push(path.arc.label);
        let new_path = FstPath {
            arc: path.arc.clone(),
            output,
            input: new_input,
            boost: path.boost,
            context: path.context.clone(),
            payload: path.payload,
        };
        if self.accept_partial(&new_path) {
            let cmp = &self.path_comparator;
            let Some(queue) = &mut self.queue else { return };
            match queue.binary_search_by(|p| cmp(p, &new_path)) {
                Ok(_) => {} // TreeSet.add of an equal element is a no-op.
                Err(pos) => queue.insert(pos, new_path),
            }
            if queue.len() == self.max_queue_depth + 1 {
                queue.pop();
            }
        }
    }

    /// `addStartPaths(node, startOutput, allowEmptyString, input)`.
    pub fn add_start_paths(
        &mut self,
        node: &Arc<O::Value>,
        start_output: O::Value,
        allow_empty_string: bool,
        input: Vec<i32>,
    ) -> Result<()> {
        self.add_start_paths_with(node, start_output, allow_empty_string, input, 0.0, None, -1)
    }

    /// `addStartPaths(node, startOutput, allowEmptyString, input, boost,
    /// context, payload)`.
    #[allow(clippy::too_many_arguments)]
    pub fn add_start_paths_with(
        &mut self,
        node: &Arc<O::Value>,
        start_output: O::Value,
        allow_empty_string: bool,
        input: Vec<i32>,
        boost: f32,
        context: Option<String>,
        payload: i32,
    ) -> Result<()> {
        let mut path = FstPath {
            arc: node.clone(),
            output: start_output,
            input,
            boost,
            context,
            payload,
        };
        let mut r = self.fst.bytes_reader();
        let follow = path.arc.clone();
        self.fst
            .read_first_target_arc(&follow, &mut path.arc, &mut r)?;
        loop {
            if allow_empty_string || path.arc.label != END_LABEL {
                self.add_if_competitive(&path);
            }
            if path.arc.is_last() {
                break;
            }
            self.fst.read_next_arc(&mut path.arc, &mut r)?;
        }
        Ok(())
    }

    /// `search()`.
    // ARITH: result counts bounded by `top_n`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn search(&mut self) -> Result<TopResults<O::Value>> {
        let mut results: Vec<TopResult<O::Value>> = Vec::new();
        let mut r = self.fst.bytes_reader();
        let no = O::no_output();
        let mut reject_count = 0usize;
        let mut scratch_arc = Arc::new(no.clone());
        while results.len() < self.top_n {
            let Some(queue) = &mut self.queue else { break };
            if queue.is_empty() {
                break;
            }
            let mut path = queue.remove(0);
            if !self.accept_partial(&path) {
                continue;
            }
            if path.arc.label == END_LABEL {
                path.input.pop();
                results.push(TopResult {
                    input: path.input,
                    output: path.output,
                });
                continue;
            }
            if results.len() == self.top_n - 1 && self.max_queue_depth == self.top_n {
                self.queue = None;
            }
            loop {
                let follow = path.arc.clone();
                self.fst
                    .read_first_target_arc(&follow, &mut path.arc, &mut r)?;
                let mut found_zero = false;
                let mut arc_copy_is_pending = false;
                loop {
                    if (self.comparator)(&no, &path.arc.output) == Ordering::Equal {
                        if self.queue.is_none() {
                            found_zero = true;
                            break;
                        } else if !found_zero {
                            arc_copy_is_pending = true;
                            found_zero = true;
                        } else {
                            self.add_if_competitive(&path);
                        }
                    } else if self.queue.is_some() {
                        self.add_if_competitive(&path);
                    }
                    if path.arc.is_last() {
                        break;
                    }
                    if arc_copy_is_pending {
                        scratch_arc = path.arc.clone();
                        arc_copy_is_pending = false;
                    }
                    self.fst.read_next_arc(&mut path.arc, &mut r)?;
                }
                if !found_zero {
                    return Err(corrupt("no zero-output arc on a shortest path"));
                }
                if self.queue.is_some() && !arc_copy_is_pending {
                    path.arc = scratch_arc.clone();
                }
                if path.arc.label == END_LABEL {
                    path.output = O::add(&path.output, &path.arc.output);
                    let accept = match &mut self.accept_result {
                        Some(f) => f(&path.input, &path.output),
                        None => true,
                    };
                    if accept {
                        results.push(TopResult {
                            input: path.input.clone(),
                            output: path.output.clone(),
                        });
                    } else {
                        reject_count += 1;
                    }
                    break;
                } else {
                    path.input.push(path.arc.label);
                    path.output = O::add(&path.output, &path.arc.output);
                    if !self.accept_partial(&path) {
                        break;
                    }
                }
            }
        }
        Ok(TopResults {
            is_complete: reject_count + self.top_n <= self.max_queue_depth,
            top_n: results,
        })
    }
}

/// `Util.shortestPaths(fst, fromNode, startOutput, comparator, topN,
/// allowEmptyString)`.
pub fn shortest_paths<O: FstOutputs>(
    fst: &TypedFst<O>,
    from_node: &Arc<O::Value>,
    start_output: O::Value,
    comparator: impl Fn(&O::Value, &O::Value) -> Ordering + Clone,
    top_n: usize,
    allow_empty_string: bool,
) -> Result<TopResults<O::Value>> {
    let mut searcher = TopNSearcher::new(fst, top_n, top_n, comparator);
    searcher.add_start_paths(from_node, start_output, allow_empty_string, Vec::new())?;
    searcher.search()
}

/// `Util.readCeilArc(label, fst, follow, arc, in)`: the first arc leaving
/// `follow`'s target whose label is `>= label`.
// ARITH: label differences in i64.
#[allow(clippy::arithmetic_side_effects)]
pub fn read_ceil_arc<O: FstOutputs>(
    label: i32,
    fst: &TypedFst<O>,
    follow: &Arc<O::Value>,
    input: &mut ReverseReader<'_>,
) -> Result<Option<Arc<O::Value>>> {
    if label == END_LABEL {
        return Ok(TypedFst::<O>::read_end_arc(follow));
    }
    if !target_has_arcs(follow) {
        return Ok(None);
    }
    let mut arc = Arc::new(O::no_output());
    fst.read_first_target_arc(follow, &mut arc, input)?;
    if arc.bytes_per_arc != 0 && arc.label != END_LABEL {
        if arc.node_flags == ARCS_FOR_DIRECT_ADDRESSING {
            let target_index = i64::from(label) - i64::from(arc.label);
            if target_index >= i64::from(arc.num_arcs) {
                return Ok(None);
            } else if target_index < 0 {
                return Ok(Some(arc));
            }
            let target_index = target_index as i32;
            if fst.bit_is_set(target_index, &arc, input)? {
                fst.read_arc_by_direct_addressing(&mut arc, input, target_index)?;
            } else {
                let ceil = fst.bit_next_set(target_index, &arc, input)?;
                if ceil < 0 {
                    return Err(corrupt("direct-addressing node without a last arc"));
                }
                fst.read_arc_by_direct_addressing(&mut arc, input, ceil)?;
            }
            return Ok(Some(arc));
        } else if arc.node_flags == ARCS_FOR_CONTINUOUS {
            let target_index = i64::from(label) - i64::from(arc.label);
            if target_index >= i64::from(arc.num_arcs) {
                return Ok(None);
            } else if target_index < 0 {
                return Ok(Some(arc));
            }
            fst.read_arc_by_continuous(&mut arc, input, target_index as i32)?;
            return Ok(Some(arc));
        }
        let mut idx = binary_search(fst, &arc, label)?;
        if idx >= 0 {
            fst.read_arc_by_index(&mut arc, input, idx)?;
            return Ok(Some(arc));
        }
        idx = -1 - idx;
        if idx == arc.num_arcs {
            return Ok(None);
        }
        fst.read_arc_by_index(&mut arc, input, idx)?;
        return Ok(Some(arc));
    }
    fst.read_first_real_target_arc(follow.target, &mut arc, input)?;
    loop {
        if arc.label >= label {
            return Ok(Some(arc));
        } else if arc.is_last() {
            return Ok(None);
        }
        fst.read_next_real_arc(&mut arc, input)?;
    }
}

/// `Util.binarySearch(fst, arc, targetLabel)`: over a binary-search node
/// from `arc.arcIdx()`; `-1 - insertionPoint` when absent.
// ARITH: midpoints and slot offsets in i64.
#[allow(clippy::arithmetic_side_effects)]
pub fn binary_search<O: FstOutputs>(
    fst: &TypedFst<O>,
    arc: &Arc<O::Value>,
    target_label: i32,
) -> Result<i32> {
    let mut input = fst.bytes_reader();
    let mut low = i64::from(arc.arc_idx);
    let mut high = i64::from(arc.num_arcs) - 1;
    while low <= high {
        let mid = (low + high) >> 1;
        input.set_position(arc.pos_arcs_start);
        input.skip_bytes(i64::from(arc.bytes_per_arc) * mid + 1);
        let mid_label = fst.read_label(&mut input)?;
        match mid_label.cmp(&target_label) {
            Ordering::Less => low = mid + 1,
            Ordering::Greater => high = mid - 1,
            Ordering::Equal => return Ok(mid as i32),
        }
    }
    Ok((-1 - low) as i32)
}

/// `Util.toDot(fst, out, sameRank, labelStates)`: Graphviz text, with
/// `output_to_string` standing in for `Outputs.outputToString`.
// ARITH: level counter.
#[allow(clippy::arithmetic_side_effects)]
pub fn to_dot<O: FstOutputs>(
    fst: &TypedFst<O>,
    same_rank: bool,
    label_states: bool,
    output_to_string: impl Fn(&O::Value) -> String,
) -> Result<String> {
    let mut out = String::new();
    let expanded_node_color = "blue";
    let start_arc = fst.first_arc();
    let mut this_level: Vec<Arc<O::Value>> = Vec::new();
    let mut next_level: Vec<Arc<O::Value>> = vec![start_arc.clone()];
    let mut same_level_states: Vec<i64> = Vec::new();
    let mut seen = std::collections::HashSet::new();
    seen.insert(start_arc.target);
    out.push_str("digraph FST {\n");
    out.push_str("  rankdir = LR; splines=true; concentrate=true; ordering=out; ranksep=2.5; \n");
    if !label_states {
        out.push_str("  node [shape=circle, width=.2, height=.2, style=filled]\n");
    }
    emit_dot_state(&mut out, "initial", Some("point"), Some("white"), Some(""));
    let no = O::no_output();
    let mut r = fst.bytes_reader();
    {
        let color = fst
            .is_expanded_target(&start_arc, &mut r)?
            .then_some(expanded_node_color);
        let (is_final, final_output) = if start_arc.is_final() {
            let fo =
                (start_arc.next_final_output != no).then(|| start_arc.next_final_output.clone());
            (true, fo)
        } else {
            (false, None)
        };
        let label = final_output
            .as_ref()
            .map(&output_to_string)
            .unwrap_or_default();
        emit_dot_state(
            &mut out,
            &start_arc.target.to_string(),
            Some(if is_final { "doublecircle" } else { "circle" }),
            color,
            Some(&label),
        );
    }
    let _ = writeln!(out, "  initial -> {}", start_arc.target);
    let mut level = 0;
    while !next_level.is_empty() {
        this_level.append(&mut next_level);
        level += 1;
        let _ = writeln!(out, "\n  // Transitions and states at level: {level}");
        while let Some(mut arc) = this_level.pop() {
            if !target_has_arcs(&arc) {
                continue;
            }
            let node = arc.target;
            fst.read_first_real_target_arc(node, &mut arc, &mut r)?;
            loop {
                if arc.target >= 0 && !seen.contains(&arc.target) {
                    let color = fst
                        .is_expanded_target(&arc, &mut r)?
                        .then_some(expanded_node_color);
                    let final_output = if arc.next_final_output != no {
                        output_to_string(&arc.next_final_output)
                    } else {
                        String::new()
                    };
                    emit_dot_state(
                        &mut out,
                        &arc.target.to_string(),
                        Some("circle"),
                        color,
                        Some(&final_output),
                    );
                    seen.insert(arc.target);
                    next_level.push(arc.clone());
                    same_level_states.push(arc.target);
                }
                let mut outs = if arc.output != no {
                    format!("/{}", output_to_string(&arc.output))
                } else {
                    String::new()
                };
                if !target_has_arcs(&arc) && arc.is_final() && arc.next_final_output != no {
                    outs = format!("{outs}/[{}]", output_to_string(&arc.next_final_output));
                }
                let arc_color = if arc.flag(BIT_TARGET_NEXT) {
                    "red"
                } else {
                    "black"
                };
                let _ = writeln!(
                    out,
                    "  {node} -> {} [label=\"{}{outs}\"{} color=\"{arc_color}\"]",
                    arc.target,
                    printable_label(arc.label),
                    if arc.is_final() {
                        " style=\"bold\""
                    } else {
                        ""
                    },
                );
                if arc.is_last() {
                    break;
                }
                fst.read_next_real_arc(&mut arc, &mut r)?;
            }
        }
        if same_rank && same_level_states.len() > 1 {
            out.push_str("  {rank=same; ");
            for s in &same_level_states {
                let _ = write!(out, "{}; ", *s as i32);
            }
            out.push_str(" }\n");
        }
        same_level_states.clear();
    }
    out.push_str("  -1 [style=filled, color=black, shape=doublecircle, label=\"\"]\n\n");
    out.push_str("  {rank=sink; -1 }\n");
    out.push_str("}\n");
    Ok(out)
}

fn emit_dot_state(
    out: &mut String,
    name: &str,
    shape: Option<&str>,
    color: Option<&str>,
    label: Option<&str>,
) {
    let _ = writeln!(
        out,
        "  {name} [{} {} {} ]",
        shape.map(|s| format!("shape={s}")).unwrap_or_default(),
        color.map(|c| format!("color={c}")).unwrap_or_default(),
        label.map_or("label=\"\"".to_string(), |l| format!("label=\"{l}\"")),
    );
}

fn printable_label(label: i32) -> String {
    if (0x20..=0x7d).contains(&label) && label != 0x22 && label != 0x5c {
        char::from(label as u8).to_string()
    } else {
        format!("0x{:x}", label)
    }
}

/// `Util.toUTF16(s)`: one int per UTF-16 code unit.
pub fn to_utf16(s: &str) -> Vec<i32> {
    s.encode_utf16().map(i32::from).collect()
}

/// `Util.toUTF32(s)`: one int per code point.
pub fn to_utf32(s: &str) -> Vec<i32> {
    s.chars().map(|c| c as i32).collect()
}

/// `Util.toUTF32(char[], offset, length)` over UTF-16 code units: unpaired
/// surrogates pass through as their own values (`Character.codePointAt`).
// ARITH: surrogate-pair arithmetic on 16-bit values.
#[allow(clippy::arithmetic_side_effects)]
pub fn to_utf32_units(units: &[u16]) -> Vec<i32> {
    let mut out = Vec::with_capacity(units.len());
    let mut i = 0;
    while i < units.len() {
        let hi = units[i];
        if (0xd800..0xdc00).contains(&hi) {
            if let Some(&lo) = units
                .get(i + 1)
                .filter(|&&lo| (0xdc00..0xe000).contains(&lo))
            {
                out.push(0x10000 + ((i32::from(hi) - 0xd800) << 10) + (i32::from(lo) - 0xdc00));
                i += 2;
                continue;
            }
        }
        out.push(i32::from(hi));
        i += 1;
    }
    out
}

/// `Util.toIntsRef(BytesRef)`.
pub fn to_ints_ref(input: &[u8]) -> Vec<i32> {
    input.iter().map(|&b| i32::from(b)).collect()
}

/// `Util.toBytesRef(IntsRef)`: each int truncated to a byte (Java asserts
/// `-128..=255`).
pub fn to_bytes_ref(input: &[i32]) -> Vec<u8> {
    input.iter().map(|&v| v as u8).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fst::PositiveIntOutputs;
    use crate::fst_compiler::FstCompilerBuilder;

    fn weights(entries: &[(&str, i64)]) -> TypedFst<PositiveIntOutputs> {
        let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
        for (k, v) in entries {
            c.add_bytes(k.as_bytes(), *v).unwrap();
        }
        TypedFst::from_compiled(c.compile().unwrap())
    }

    #[test]
    fn get_and_shortest_paths() {
        let fst = weights(&[
            ("a", 5),
            ("ab", 3),
            ("abc", 9),
            ("b", 1),
            ("bcd", 7),
            ("c", 2),
        ]);
        assert_eq!(get_bytes(&fst, b"ab").unwrap(), Some(3));
        assert_eq!(get_bytes(&fst, b"abd").unwrap(), None);
        assert_eq!(get_bytes(&fst, b"bc").unwrap(), None);
        let first = fst.first_arc();
        let top = shortest_paths(&fst, &first, 0, |a: &i64, b: &i64| a.cmp(b), 3, false).unwrap();
        let got: Vec<(Vec<u8>, i64)> = top
            .top_n
            .iter()
            .map(|r| (to_bytes_ref(&r.input), r.output))
            .collect();
        assert_eq!(
            got,
            vec![(b"b".to_vec(), 1), (b"c".to_vec(), 2), (b"ab".to_vec(), 3)]
        );
        assert!(top.is_complete);
        let saved = fst.clone();
        let dot = to_dot(&saved, true, true, |v| v.to_string()).unwrap();
        assert!(dot.starts_with("digraph FST {"));
        assert!(dot.contains("{rank=sink; -1 }"));
    }

    #[test]
    fn conversions() {
        assert_eq!(to_utf16("a\u{1F600}"), vec![97, 0xd83d, 0xde00]);
        assert_eq!(to_utf32("a\u{1F600}"), vec![97, 0x1F600]);
        assert_eq!(
            to_utf32_units(&[97, 0xd83d, 0xde00, 0xd800]),
            vec![97, 0x1F600, 0xd800]
        );
        assert_eq!(to_ints_ref(&[1, 255]), vec![1, 255]);
        assert_eq!(to_bytes_ref(&[1, 255, -1]), vec![1, 255, 255]);
        assert_eq!(printable_label(0x22), "0x22");
        assert_eq!(printable_label(b'a' as i32), "a");
    }

    #[test]
    fn read_rejects_bad_metadata() {
        assert!(TypedFst::<PositiveIntOutputs>::read(&[1, 2, 3]).is_err());
        let fst = weights(&[("x", 1)]);
        let bytes = crate::fst_compiler::CompiledFst {
            input_type: InputType::Byte1,
            empty_output: None,
            start_node: fst.start_node,
            version: fst.version,
            bytes: fst.bytes.clone(),
        }
        .save::<PositiveIntOutputs>();
        let back = TypedFst::<PositiveIntOutputs>::read(&bytes).unwrap();
        assert_eq!(get_bytes(&back, b"x").unwrap(), Some(1));
        assert!(TypedFst::<PositiveIntOutputs>::read(&bytes[..bytes.len() - 1]).is_err());
    }
}
