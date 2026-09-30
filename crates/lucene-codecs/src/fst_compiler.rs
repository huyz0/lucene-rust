//! Port of `org.apache.lucene.util.fst.FSTCompiler` and `NodeHash`: Lucene's
//! incremental, streaming FST construction, byte-identical to Lucene 10.5.0.
//!
//! This is the faithful port; `crate::fst::build_fst` (a whole-trie builder
//! with no output pushing and list-encoded nodes only) predates it and stays
//! for its existing callers. Everything that decides the bytes is reproduced:
//!
//! - the frontier of uncompiled nodes, `freezeTail`, and output pushing
//!   through [`FstOutputs::common`]/[`subtract`](FstOutputs::subtract)/
//!   [`add`](FstOutputs::add);
//! - `addNode`'s flag rules, including `BIT_TARGET_NEXT` against the last
//!   frozen node and the leading padding byte;
//! - all three fixed-length arc encodings (binary search, direct addressing
//!   with its expansion-credit heuristic, continuous) and the depth/arc-count
//!   rule that picks them;
//! - the stale bytes Java leaves in the padding of fixed-length arc slots:
//!   `scratchBytes` and `fixedLengthArcsBuffer` are reused across nodes
//!   without clearing, so the padding holds whatever an earlier node wrote
//!   there, and this port keeps both buffers with the same lifetimes (and
//!   `ArrayUtil.oversize` reallocation points) to reproduce it;
//! - `NodeHash`'s RAM-bounded two-generation suffix table
//!   (`suffixRAMLimitMB`): which nodes are shared depends only on the
//!   primary/fallback generations and their byte accounting, never on the
//!   hash function, so the tables are plain hash maps keyed by the node's
//!   arcs with Java's `ramBytesUsed` arithmetic deciding when a generation
//!   rolls over.
//!
//! `FST.FSTMetadata.save` + the on-heap body is [`CompiledFst::save`].

use std::collections::HashMap;
use std::fmt::Debug;
use std::hash::Hash;

use lucene_store::codec_util;

use crate::fst::{
    ByteSequenceOutputs, InputType, Pair, PairOutputs, PositiveIntOutputs, ARCS_FOR_BINARY_SEARCH,
    ARCS_FOR_CONTINUOUS, ARCS_FOR_DIRECT_ADDRESSING, BIT_ARC_HAS_FINAL_OUTPUT, BIT_ARC_HAS_OUTPUT,
    BIT_FINAL_ARC, BIT_LAST_ARC, BIT_STOP_NODE, BIT_TARGET_NEXT, FILE_FORMAT_NAME, FINAL_END_NODE,
    NON_FINAL_END_NODE, VERSION_CONTINUOUS_ARCS, VERSION_CURRENT,
};

/// `FST.VERSION_90`: the oldest version `FSTCompiler.Builder.setVersion`
/// accepts.
pub const VERSION_90: i32 = 8;

/// `FSTCompiler.DIRECT_ADDRESSING_MAX_OVERSIZING_FACTOR`.
pub const DIRECT_ADDRESSING_MAX_OVERSIZING_FACTOR: f32 = 1.0;
const FIXED_LENGTH_ARC_SHALLOW_DEPTH: usize = 3;
const FIXED_LENGTH_ARC_SHALLOW_NUM_ARCS: usize = 5;
const FIXED_LENGTH_ARC_DEEP_NUM_ARCS: usize = 10;
const DIRECT_ADDRESSING_MAX_OVERSIZE_WITH_CREDIT_FACTOR: f32 = 1.66;
const UNCOMPILED: i64 = i64::MIN;

// --- Outputs -----------------------------------------------------------------

/// Port of `org.apache.lucene.util.fst.Outputs<T>`: the output algebra
/// (`common`/`subtract`/`add`) the compiler pushes outputs with, and the wire
/// encoding. `Value` equality stands in for Java's `NO_OUTPUT` identity
/// checks: every Java `Outputs` returns the singleton exactly when the value
/// equals it (`validOutput`), so the two are the same test.
pub trait FstOutputs {
    /// Java's `T`.
    type Value: Clone + Eq + Hash + Debug;

    /// `getNoOutput()`.
    fn no_output() -> Self::Value;
    /// `common(output1, output2)`.
    fn common(a: &Self::Value, b: &Self::Value) -> Self::Value;
    /// `subtract(output, inc)`.
    fn subtract(output: &Self::Value, inc: &Self::Value) -> Self::Value;
    /// `add(prefix, output)`.
    fn add(prefix: &Self::Value, output: &Self::Value) -> Self::Value;
    /// `write(output, out)`.
    fn write(value: &Self::Value, out: &mut Vec<u8>);
    /// `writeFinalOutput(output, out)`.
    fn write_final_output(value: &Self::Value, out: &mut Vec<u8>) {
        Self::write(value, out);
    }
    /// `read(in)` over the FST's reverse byte cursor.
    fn read(input: &mut ReverseReader<'_>) -> Result<Self::Value, FstReadError>;
    /// `readFinalOutput(in)`.
    fn read_final_output(input: &mut ReverseReader<'_>) -> Result<Self::Value, FstReadError> {
        Self::read(input)
    }
    /// `merge(first, second)`: `None` where Java throws
    /// `UnsupportedOperationException` (every core `Outputs` but `NoOutputs`).
    fn merge(_first: &Self::Value, _second: &Self::Value) -> Option<Self::Value> {
        None
    }
}

/// A malformed FST body met while reading outputs or arcs.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
#[error("corrupt FST: {0}")]
pub struct FstReadError(pub String);

/// `FST.BytesReader` over an on-heap body: reads move *backwards*.
#[derive(Debug, Clone)]
pub struct ReverseReader<'a> {
    bytes: &'a [u8],
    pos: i64,
}

impl<'a> ReverseReader<'a> {
    /// A reader over `bytes`, positioned at 0.
    pub fn new(bytes: &'a [u8]) -> Self {
        ReverseReader { bytes, pos: 0 }
    }
    /// `getPosition()`.
    pub fn position(&self) -> i64 {
        self.pos
    }
    /// `setPosition(pos)`.
    pub fn set_position(&mut self, pos: i64) {
        self.pos = pos;
    }
    /// `readByte()`.
    pub fn read_byte(&mut self) -> Result<u8, FstReadError> {
        let b = usize::try_from(self.pos)
            .ok()
            .and_then(|p| self.bytes.get(p))
            .copied()
            .ok_or_else(|| FstReadError(format!("read past the body at {}", self.pos)))?;
        self.pos = self.pos.wrapping_sub(1);
        Ok(b)
    }
    /// `skipBytes(n)`.
    pub fn skip_bytes(&mut self, n: i64) {
        self.pos = self.pos.wrapping_sub(n);
    }
    /// `readVInt()`.
    pub fn read_vint(&mut self) -> Result<i32, FstReadError> {
        Ok(self.read_vlong_bounded(5)? as i32)
    }
    /// `readVLong()`.
    pub fn read_vlong(&mut self) -> Result<i64, FstReadError> {
        self.read_vlong_bounded(9)
    }
    // ARITH: `i < max <= 9`, so the shift stays below 64.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_vlong_bounded(&mut self, max: u32) -> Result<i64, FstReadError> {
        let mut v: u64 = 0;
        for i in 0..max {
            let b = self.read_byte()?;
            v |= u64::from(b & 0x7f) << (7 * i);
            if b & 0x80 == 0 {
                return Ok(v as i64);
            }
        }
        Err(FstReadError("variable-length integer too long".into()))
    }
    /// `readShort()` (little-endian, as every `VERSION_90`+ FST).
    pub fn read_short(&mut self) -> Result<i16, FstReadError> {
        let lo = self.read_byte()?;
        let hi = self.read_byte()?;
        Ok(i16::from_le_bytes([lo, hi]))
    }
}

fn write_vint(out: &mut Vec<u8>, v: i32) {
    let mut v = v as u32;
    while v & !0x7f != 0 {
        out.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn write_vlong(out: &mut Vec<u8>, v: i64) {
    let mut v = v as u64;
    while v & !0x7f != 0 {
        out.push((v & 0x7f) as u8 | 0x80);
        v >>= 7;
    }
    out.push(v as u8);
}

fn common_prefix_len<T: PartialEq>(a: &[T], b: &[T]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// Sequence `common`: the shared prefix (`Arrays.mismatch`).
fn seq_common<T: Clone + PartialEq>(a: &[T], b: &[T]) -> Vec<T> {
    a[..common_prefix_len(a, b)].to_vec()
}

/// Sequence `subtract`: `output` minus its prefix `inc`.
fn seq_subtract<T: Clone + PartialEq>(output: &[T], inc: &[T]) -> Vec<T> {
    debug_assert!(output.starts_with(inc), "subtract of a non-prefix");
    output[inc.len().min(output.len())..].to_vec()
}

fn seq_add<T: Clone>(prefix: &[T], output: &[T]) -> Vec<T> {
    let mut v = prefix.to_vec();
    v.extend_from_slice(output);
    v
}

fn read_len(input: &mut ReverseReader<'_>) -> Result<usize, FstReadError> {
    let len = input.read_vint()?;
    usize::try_from(len).map_err(|_| FstReadError(format!("negative output length {len}")))
}

impl FstOutputs for ByteSequenceOutputs {
    type Value = Vec<u8>;
    fn no_output() -> Vec<u8> {
        Vec::new()
    }
    fn common(a: &Vec<u8>, b: &Vec<u8>) -> Vec<u8> {
        seq_common(a, b)
    }
    fn subtract(output: &Vec<u8>, inc: &Vec<u8>) -> Vec<u8> {
        seq_subtract(output, inc)
    }
    fn add(prefix: &Vec<u8>, output: &Vec<u8>) -> Vec<u8> {
        seq_add(prefix, output)
    }
    fn write(value: &Vec<u8>, out: &mut Vec<u8>) {
        write_vint(out, value.len() as i32);
        out.extend_from_slice(value);
    }
    fn read(input: &mut ReverseReader<'_>) -> Result<Vec<u8>, FstReadError> {
        let len = read_len(input)?;
        (0..len).map(|_| input.read_byte()).collect()
    }
}

impl FstOutputs for PositiveIntOutputs {
    type Value = i64;
    fn no_output() -> i64 {
        0
    }
    fn common(a: &i64, b: &i64) -> i64 {
        if *a == 0 || *b == 0 {
            0
        } else {
            (*a).min(*b)
        }
    }
    fn subtract(output: &i64, inc: &i64) -> i64 {
        output.wrapping_sub(*inc)
    }
    fn add(prefix: &i64, output: &i64) -> i64 {
        prefix.wrapping_add(*output)
    }
    fn write(value: &i64, out: &mut Vec<u8>) {
        write_vlong(out, *value);
    }
    fn read(input: &mut ReverseReader<'_>) -> Result<i64, FstReadError> {
        input.read_vlong()
    }
}

/// Port of `IntSequenceOutputs`: an `IntsRef` output.
#[derive(Debug, Clone, Copy, Default)]
pub struct IntSequenceOutputs;

impl FstOutputs for IntSequenceOutputs {
    type Value = Vec<i32>;
    fn no_output() -> Vec<i32> {
        Vec::new()
    }
    fn common(a: &Vec<i32>, b: &Vec<i32>) -> Vec<i32> {
        seq_common(a, b)
    }
    fn subtract(output: &Vec<i32>, inc: &Vec<i32>) -> Vec<i32> {
        seq_subtract(output, inc)
    }
    fn add(prefix: &Vec<i32>, output: &Vec<i32>) -> Vec<i32> {
        seq_add(prefix, output)
    }
    fn write(value: &Vec<i32>, out: &mut Vec<u8>) {
        write_vint(out, value.len() as i32);
        for &v in value {
            write_vint(out, v);
        }
    }
    fn read(input: &mut ReverseReader<'_>) -> Result<Vec<i32>, FstReadError> {
        let len = read_len(input)?;
        (0..len).map(|_| input.read_vint()).collect()
    }
}

/// Port of `CharSequenceOutputs`: a `CharsRef` output (UTF-16 code units).
#[derive(Debug, Clone, Copy, Default)]
pub struct CharSequenceOutputs;

impl FstOutputs for CharSequenceOutputs {
    type Value = Vec<u16>;
    fn no_output() -> Vec<u16> {
        Vec::new()
    }
    fn common(a: &Vec<u16>, b: &Vec<u16>) -> Vec<u16> {
        seq_common(a, b)
    }
    fn subtract(output: &Vec<u16>, inc: &Vec<u16>) -> Vec<u16> {
        seq_subtract(output, inc)
    }
    fn add(prefix: &Vec<u16>, output: &Vec<u16>) -> Vec<u16> {
        seq_add(prefix, output)
    }
    fn write(value: &Vec<u16>, out: &mut Vec<u8>) {
        write_vint(out, value.len() as i32);
        for &c in value {
            write_vint(out, i32::from(c));
        }
    }
    fn read(input: &mut ReverseReader<'_>) -> Result<Vec<u16>, FstReadError> {
        let len = read_len(input)?;
        (0..len)
            .map(|_| input.read_vint().map(|c| c as u16))
            .collect()
    }
}

/// Port of `NoOutputs`: an FST that is only an automaton (every output is
/// `NO_OUTPUT`; duplicate inputs merge).
#[derive(Debug, Clone, Copy, Default)]
pub struct NoOutputs;

impl FstOutputs for NoOutputs {
    type Value = ();
    fn no_output() {}
    fn common(_: &(), _: &()) {}
    fn subtract(_: &(), _: &()) {}
    fn add(_: &(), _: &()) {}
    fn write(_: &(), _: &mut Vec<u8>) {}
    fn read(_: &mut ReverseReader<'_>) -> Result<(), FstReadError> {
        Ok(())
    }
    fn merge(_: &(), _: &()) -> Option<()> {
        Some(())
    }
}

impl<A: FstOutputs, B: FstOutputs> FstOutputs for PairOutputs<A, B> {
    type Value = Pair<A::Value, B::Value>;
    fn no_output() -> Self::Value {
        Pair {
            first: A::no_output(),
            second: B::no_output(),
        }
    }
    fn common(a: &Self::Value, b: &Self::Value) -> Self::Value {
        Pair {
            first: A::common(&a.first, &b.first),
            second: B::common(&a.second, &b.second),
        }
    }
    fn subtract(output: &Self::Value, inc: &Self::Value) -> Self::Value {
        Pair {
            first: A::subtract(&output.first, &inc.first),
            second: B::subtract(&output.second, &inc.second),
        }
    }
    fn add(prefix: &Self::Value, output: &Self::Value) -> Self::Value {
        Pair {
            first: A::add(&prefix.first, &output.first),
            second: B::add(&prefix.second, &output.second),
        }
    }
    fn write(value: &Self::Value, out: &mut Vec<u8>) {
        A::write(&value.first, out);
        B::write(&value.second, out);
    }
    fn read(input: &mut ReverseReader<'_>) -> Result<Self::Value, FstReadError> {
        let first = A::read(input)?;
        let second = B::read(input)?;
        Ok(Pair { first, second })
    }
}

// --- The compiler ---------------------------------------------------------

/// Why [`FstCompiler::add`] refused an input.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum CompileError {
    /// Inputs must arrive in sorted order (Java asserts it).
    #[error("inputs are added out of order")]
    OutOfOrder,
    /// A label is negative or too wide for the input type.
    #[error("label {label} does not fit input type {input_type:?}")]
    LabelOutOfRange { label: i32, input_type: InputType },
    /// A duplicate input under outputs that cannot merge (Java's
    /// `UnsupportedOperationException` from `Outputs.merge`).
    #[error("duplicate input and the outputs do not support merge")]
    MergeUnsupported,
    /// A `Builder` setting Java rejects.
    #[error("{0}")]
    IllegalArgument(String),
}

/// Port of `FSTCompiler.Builder`.
#[derive(Debug, Clone)]
pub struct FstCompilerBuilder {
    input_type: InputType,
    suffix_ram_limit_mb: f64,
    allow_fixed_length_arcs: bool,
    direct_addressing_max_oversizing_factor: f32,
    version: i32,
}

impl FstCompilerBuilder {
    /// `new Builder(inputType, outputs)` with Lucene's defaults: 32 MB suffix
    /// table, fixed-length arcs allowed, oversizing factor 1, current version.
    pub fn new(input_type: InputType) -> Self {
        FstCompilerBuilder {
            input_type,
            suffix_ram_limit_mb: 32.0,
            allow_fixed_length_arcs: true,
            direct_addressing_max_oversizing_factor: DIRECT_ADDRESSING_MAX_OVERSIZING_FACTOR,
            version: VERSION_CURRENT,
        }
    }

    /// `suffixRAMLimitMB(mb)`: 0 disables suffix sharing.
    pub fn suffix_ram_limit_mb(mut self, mb: f64) -> Result<Self, CompileError> {
        if mb.is_nan() || mb < 0.0 {
            return Err(CompileError::IllegalArgument(format!(
                "suffixRAMLimitMB must be >= 0; got: {mb}"
            )));
        }
        self.suffix_ram_limit_mb = mb;
        Ok(self)
    }

    /// `allowFixedLengthArcs(b)`.
    pub fn allow_fixed_length_arcs(mut self, allow: bool) -> Self {
        self.allow_fixed_length_arcs = allow;
        self
    }

    /// `directAddressingMaxOversizingFactor(f)`.
    pub fn direct_addressing_max_oversizing_factor(mut self, factor: f32) -> Self {
        self.direct_addressing_max_oversizing_factor = factor;
        self
    }

    /// `setVersion(v)`: `VERSION_90..=VERSION_CURRENT`.
    pub fn version(mut self, version: i32) -> Result<Self, CompileError> {
        if !(VERSION_90..=VERSION_CURRENT).contains(&version) {
            return Err(CompileError::IllegalArgument(format!(
                "Expected version in range [{VERSION_90}, {VERSION_CURRENT}], got {version}"
            )));
        }
        self.version = version;
        Ok(self)
    }

    /// `build()`.
    pub fn build<O: FstOutputs>(self) -> FstCompiler<O> {
        let dedup =
            (self.suffix_ram_limit_mb > 0.0).then(|| NodeHash::new(self.suffix_ram_limit_mb));
        FstCompiler {
            input_type: self.input_type,
            allow_fixed_length_arcs: self.allow_fixed_length_arcs,
            da_factor: self.direct_addressing_max_oversizing_factor,
            version: self.version,
            dedup,
            empty_output: None,
            last_input: Vec::new(),
            padding_byte_pending: true,
            frontier: (0..10).map(UncompiledNode::new).collect(),
            last_frozen_node: 0,
            num_bytes_per_arc: Vec::new(),
            num_label_bytes_per_arc: Vec::new(),
            fixed_buf: vec![0; 11],
            da_credit: 0,
            data: Vec::new(),
            scratch: Vec::new(),
            scratch_pos: 0,
            num_bytes_written: 1,
            arc_count: 0,
            node_count: 0,
            binary_search_node_count: 0,
            direct_addressing_node_count: 0,
            continuous_node_count: 0,
            last_read_len: 0,
            finished: false,
        }
    }
}

#[derive(Debug, Clone)]
struct UncompiledArc<V> {
    label: i32,
    target: i64,
    is_final: bool,
    output: V,
    next_final_output: V,
}

#[derive(Debug, Clone)]
struct UncompiledNode<V> {
    arcs: Vec<UncompiledArc<V>>,
    output: Option<V>,
    is_final: bool,
    depth: usize,
}

impl<V: Clone> UncompiledNode<V> {
    fn new(depth: usize) -> Self {
        UncompiledNode {
            arcs: Vec::new(),
            output: None,
            is_final: false,
            depth,
        }
    }
}

/// A node's identity for suffix sharing: its arcs, fully resolved.
type NodeKey<V> = Vec<(i32, i64, V, V, bool)>;

#[derive(Debug)]
struct Generation<V> {
    /// Node arcs -> (address, bytes a reader consumes through its last arc).
    map: HashMap<NodeKey<V>, (i64, i64)>,
    copied_bytes: i64,
}

impl<V> Generation<V> {
    fn new() -> Self {
        Generation {
            map: HashMap::new(),
            copied_bytes: 0,
        }
    }
}

/// `NodeHash`: the primary and fallback suffix tables.
#[derive(Debug)]
struct NodeHash<V> {
    primary: Generation<V>,
    fallback: Option<Generation<V>>,
    ram_limit_bytes: i64,
}

impl<V> NodeHash<V> {
    fn new(ram_limit_mb: f64) -> Self {
        let as_bytes = ram_limit_mb * 1024.0 * 1024.0;
        let ram_limit_bytes = if as_bytes >= i64::MAX as f64 {
            i64::MAX
        } else {
            as_bytes as i64
        };
        NodeHash {
            primary: Generation::new(),
            fallback: None,
            ram_limit_bytes,
        }
    }
}

/// `PackedInts.bitsRequired`.
fn bits_required(v: i64) -> i64 {
    i64::from(64u32.saturating_sub(v.leading_zeros())).max(1)
}

/// `ArrayUtil.oversize(minTargetSize, 1)` on a 64-bit JVM.
// ARITH: `min` is an in-memory buffer size far below `i32::MAX`.
#[allow(clippy::arithmetic_side_effects)]
fn oversize_bytes(min: usize) -> usize {
    if min == 0 {
        return 0;
    }
    let new_size = min + (min >> 3).max(3);
    (new_size + 7) & 0x7fff_fff8
}

/// Port of `FSTCompiler<T>`.
#[derive(Debug)]
pub struct FstCompiler<O: FstOutputs> {
    input_type: InputType,
    allow_fixed_length_arcs: bool,
    da_factor: f32,
    version: i32,
    dedup: Option<NodeHash<O::Value>>,
    empty_output: Option<O::Value>,
    last_input: Vec<i32>,
    padding_byte_pending: bool,
    frontier: Vec<UncompiledNode<O::Value>>,
    last_frozen_node: i64,
    num_bytes_per_arc: Vec<usize>,
    num_label_bytes_per_arc: Vec<usize>,
    /// `fixedLengthArcsBuffer`'s array: reused, reallocated (zeroed) only
    /// when too small.
    fixed_buf: Vec<u8>,
    da_credit: i64,
    data: Vec<u8>,
    /// `scratchBytes`: its content outlives each node (only the position
    /// resets), which is where fixed-length padding bytes come from.
    scratch: Vec<u8>,
    scratch_pos: usize,
    num_bytes_written: i64,
    arc_count: i64,
    node_count: i64,
    binary_search_node_count: i64,
    direct_addressing_node_count: i64,
    continuous_node_count: i64,
    /// Bytes a reader consumes through the last arc of the node `add_node`
    /// just wrote (what `NodeHash` copies when promoting from fallback).
    last_read_len: i64,
    finished: bool,
}

/// A compiled FST: `FST.FSTMetadata` plus the on-heap body.
#[derive(Debug, Clone, PartialEq)]
pub struct CompiledFst<V> {
    /// `inputType`.
    pub input_type: InputType,
    /// `emptyOutput`: the empty input's output, when it is accepted.
    pub empty_output: Option<V>,
    /// `startNode`.
    pub start_node: i64,
    /// `version`.
    pub version: i32,
    /// The body, as `ReadWriteDataOutput.writeTo` writes it.
    pub bytes: Vec<u8>,
}

/// Node counts `FSTCompiler` keeps (`getNodeCount`, `getArcCount`, and the
/// per-encoding counters).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompilerStats {
    pub node_count: i64,
    pub arc_count: i64,
    pub binary_search_node_count: i64,
    pub direct_addressing_node_count: i64,
    pub continuous_node_count: i64,
}

impl<O: FstOutputs> FstCompiler<O> {
    /// `getNodeCount()` and friends.
    // ARITH: node counts of an in-memory build.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn stats(&self) -> CompilerStats {
        CompilerStats {
            node_count: 1 + self.node_count,
            arc_count: self.arc_count,
            binary_search_node_count: self.binary_search_node_count,
            direct_addressing_node_count: self.direct_addressing_node_count,
            continuous_node_count: self.continuous_node_count,
        }
    }

    /// `fstSizeInBytes()`.
    pub fn fst_size_in_bytes(&self) -> i64 {
        self.num_bytes_written
    }

    fn check_label(&self, label: i32) -> Result<(), CompileError> {
        let max = match self.input_type {
            InputType::Byte1 => 255,
            InputType::Byte2 => 65535,
            InputType::Byte4 => i32::MAX,
        };
        if (0..=max).contains(&label) {
            Ok(())
        } else {
            Err(CompileError::LabelOutOfRange {
                label,
                input_type: self.input_type,
            })
        }
    }

    /// `add(BytesRef-as-IntsRef, output)` for byte keys.
    pub fn add_bytes(&mut self, input: &[u8], output: O::Value) -> Result<(), CompileError> {
        let ints: Vec<i32> = input.iter().map(|&b| i32::from(b)).collect();
        self.add(&ints, output)
    }

    /// `add(IntsRef input, T output)`: inputs in sorted order.
    // ARITH: indices over `input`/`frontier`, bounded by their lengths.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn add(&mut self, input: &[i32], output: O::Value) -> Result<(), CompileError> {
        if !self.last_input.is_empty() && input < self.last_input.as_slice() {
            return Err(CompileError::OutOfOrder);
        }
        for &l in input {
            self.check_label(l)?;
        }
        let no = O::no_output();
        if input.is_empty() {
            let merged = match &self.empty_output {
                Some(prev) => O::merge(prev, &output).ok_or(CompileError::MergeUnsupported)?,
                None => output,
            };
            self.frontier[0].is_final = true;
            self.empty_output = Some(merged);
            return Ok(());
        }
        let duplicate = input == self.last_input.as_slice();
        if duplicate && O::merge(&no, &no).is_none() {
            return Err(CompileError::MergeUnsupported);
        }

        let pos1 = common_prefix_len(&self.last_input, input);
        let prefix_len_plus1 = pos1 + 1;
        while self.frontier.len() < input.len() + 1 {
            let depth = self.frontier.len();
            self.frontier.push(UncompiledNode::new(depth));
        }

        self.freeze_tail(prefix_len_plus1);

        for idx in prefix_len_plus1..=input.len() {
            self.frontier[idx - 1].arcs.push(UncompiledArc {
                label: input[idx - 1],
                target: UNCOMPILED,
                is_final: false,
                output: no.clone(),
                next_final_output: no.clone(),
            });
        }

        let len = input.len();
        if self.last_input.len() != len || prefix_len_plus1 != len + 1 {
            self.frontier[len].is_final = true;
            self.frontier[len].output = Some(no.clone());
        }

        let mut output = output;
        for idx in 1..prefix_len_plus1 {
            let last_output = self.frontier[idx - 1]
                .arcs
                .last()
                .map(|a| a.output.clone())
                .unwrap_or_else(|| no.clone());
            let common = if last_output != no {
                let common = O::common(&output, &last_output);
                let word_suffix = O::subtract(&last_output, &common);
                if let Some(arc) = self.frontier[idx - 1].arcs.last_mut() {
                    arc.output = common.clone();
                }
                self.prepend_output(idx, &word_suffix);
                common
            } else {
                no.clone()
            };
            output = O::subtract(&output, &common);
        }

        if duplicate {
            let prev = self.frontier[len]
                .output
                .clone()
                .unwrap_or_else(|| no.clone());
            self.frontier[len].output =
                Some(O::merge(&prev, &output).ok_or(CompileError::MergeUnsupported)?);
        } else if let Some(arc) = self.frontier[prefix_len_plus1 - 1].arcs.last_mut() {
            arc.output = output;
        }
        self.last_input.clear();
        self.last_input.extend_from_slice(input);
        Ok(())
    }

    /// `UnCompiledNode.prependOutput`.
    fn prepend_output(&mut self, idx: usize, prefix: &O::Value) {
        let node = &mut self.frontier[idx];
        for arc in &mut node.arcs {
            arc.output = O::add(prefix, &arc.output);
        }
        if node.is_final {
            let out = node.output.clone().unwrap_or_else(O::no_output);
            node.output = Some(O::add(prefix, &out));
        }
    }

    /// `freezeTail(prefixLenPlus1)`.
    // ARITH: `idx >= down_to >= 1`.
    #[allow(clippy::arithmetic_side_effects)]
    fn freeze_tail(&mut self, prefix_len_plus1: usize) {
        let down_to = prefix_len_plus1.max(1);
        let mut idx = self.last_input.len();
        while idx >= down_to {
            let prev = idx - 1;
            let next_final_output = self.frontier[idx]
                .output
                .clone()
                .unwrap_or_else(O::no_output);
            let is_final = self.frontier[idx].is_final;
            let node = self.compile_node(idx);
            if let Some(arc) = self.frontier[prev].arcs.last_mut() {
                debug_assert_eq!(arc.label, self.last_input[prev]);
                arc.target = node;
                arc.next_final_output = next_final_output;
                arc.is_final = is_final;
            }
            idx -= 1;
        }
    }

    /// `compileNode(nodeIn)`: freezes frontier node `idx` and clears it.
    fn compile_node(&mut self, idx: usize) -> i64 {
        let bytes_pos_start = self.num_bytes_written;
        let node_in = std::mem::replace(&mut self.frontier[idx], UncompiledNode::new(idx));
        let node = if self.dedup.is_some() {
            if node_in.arcs.is_empty() {
                let n = self.add_node(&node_in);
                self.last_frozen_node = n;
                n
            } else {
                self.dedup_add(&node_in)
            }
        } else {
            self.add_node(&node_in)
        };
        if self.num_bytes_written != bytes_pos_start {
            self.last_frozen_node = node;
        }
        // `nodeIn.clear()`: keep the arc allocation for reuse.
        let mut cleared = node_in;
        cleared.arcs.clear();
        cleared.is_final = false;
        cleared.output = None;
        self.frontier[idx] = cleared;
        node
    }

    /// `NodeHash.add(nodeIn)`.
    // ARITH: Java's `long` RAM accounting over in-memory table sizes.
    #[allow(clippy::arithmetic_side_effects)]
    fn dedup_add(&mut self, node_in: &UncompiledNode<O::Value>) -> i64 {
        let key: NodeKey<O::Value> = node_in
            .arcs
            .iter()
            .map(|a| {
                (
                    a.label,
                    a.target,
                    a.output.clone(),
                    a.next_final_output.clone(),
                    a.is_final,
                )
            })
            .collect();
        if let Some(&(addr, _)) = self.dedup.as_ref().and_then(|h| h.primary.map.get(&key)) {
            return addr;
        }
        let from_fallback = self
            .dedup
            .as_ref()
            .and_then(|h| h.fallback.as_ref())
            .and_then(|f| f.map.get(&key))
            .copied();
        let (addr, entry_read_len, copied) = match from_fallback {
            Some((addr, read_len)) => (addr, read_len, read_len),
            None => {
                let addr = self.add_node(node_in);
                (addr, self.last_read_len, self.scratch_pos as i64)
            }
        };
        let Some(hash) = self.dedup.as_mut() else {
            return addr;
        };
        hash.primary.map.insert(key, (addr, entry_read_len));
        hash.primary.copied_bytes += copied;
        let count = hash.primary.map.len() as i64;
        let copied_bytes = hash.primary.copied_bytes;
        let ram_bytes_used = count * 2 * bits_required(addr) / 8
            + count * 2 * bits_required(copied_bytes) / 8
            + copied_bytes;
        if ram_bytes_used >= hash.ram_limit_bytes / 2 {
            let old = std::mem::replace(&mut hash.primary, Generation::new());
            hash.fallback = Some(old);
        }
        addr
    }

    fn should_expand_node_with_fixed_length_arcs(&self, node: &UncompiledNode<O::Value>) -> bool {
        self.allow_fixed_length_arcs
            && ((node.depth <= FIXED_LENGTH_ARC_SHALLOW_DEPTH
                && node.arcs.len() >= FIXED_LENGTH_ARC_SHALLOW_NUM_ARCS)
                || node.arcs.len() >= FIXED_LENGTH_ARC_DEEP_NUM_ARCS)
    }

    // ARITH: `scratch_pos + n` stays within an in-memory node's size.
    #[allow(clippy::arithmetic_side_effects)]
    fn scratch_write(&mut self, bytes: &[u8]) {
        let end = self.scratch_pos + bytes.len();
        if self.scratch.len() < end {
            self.scratch.resize(end, 0);
        }
        self.scratch[self.scratch_pos..end].copy_from_slice(bytes);
        self.scratch_pos = end;
    }

    fn write_label(&self, out: &mut Vec<u8>, label: i32) {
        match self.input_type {
            InputType::Byte1 => out.push(label as u8),
            InputType::Byte2 => out.extend_from_slice(&(label as i16).to_le_bytes()),
            InputType::Byte4 => write_vint(out, label),
        }
    }

    /// `addNode(nodeIn)`: serialises one node into `data`, returns its
    /// address (the address of its last byte).
    // ARITH: arc and byte counts of one in-memory node; addresses are the
    // running body length.
    #[allow(clippy::arithmetic_side_effects)]
    fn add_node(&mut self, node_in: &UncompiledNode<O::Value>) -> i64 {
        let num_arcs = node_in.arcs.len();
        if num_arcs == 0 {
            return if node_in.is_final {
                FINAL_END_NODE
            } else {
                NON_FINAL_END_NODE
            };
        }
        self.scratch_pos = 0;
        let fixed = self.should_expand_node_with_fixed_length_arcs(node_in);
        if fixed && self.num_bytes_per_arc.len() < num_arcs {
            self.num_bytes_per_arc.resize(num_arcs, 0);
            self.num_label_bytes_per_arc.resize(num_arcs, 0);
        }
        self.arc_count += num_arcs as i64;
        let no = O::no_output();
        let last_arc = num_arcs - 1;
        let mut last_arc_start = 0usize;
        let mut max_bytes_per_arc = 0usize;
        let mut max_bytes_per_arc_without_label = 0usize;
        let mut tmp = Vec::new();
        for (arc_idx, arc) in node_in.arcs.iter().enumerate() {
            let mut flags = 0u8;
            if arc_idx == last_arc {
                flags += BIT_LAST_ARC;
            }
            if self.last_frozen_node == arc.target && !fixed {
                flags += BIT_TARGET_NEXT;
            }
            if arc.is_final {
                flags += BIT_FINAL_ARC;
                if arc.next_final_output != no {
                    flags += BIT_ARC_HAS_FINAL_OUTPUT;
                }
            }
            let target_has_arcs = arc.target > 0;
            if !target_has_arcs {
                flags += BIT_STOP_NODE;
            }
            if arc.output != no {
                flags += BIT_ARC_HAS_OUTPUT;
            }
            self.scratch_write(&[flags]);
            tmp.clear();
            self.write_label(&mut tmp, arc.label);
            let num_label_bytes = tmp.len();
            if arc.output != no {
                O::write(&arc.output, &mut tmp);
            }
            if arc.next_final_output != no {
                O::write_final_output(&arc.next_final_output, &mut tmp);
            }
            if target_has_arcs && flags & BIT_TARGET_NEXT == 0 {
                write_vlong(&mut tmp, arc.target);
            }
            let chunk = std::mem::take(&mut tmp);
            self.scratch_write(&chunk);
            tmp = chunk;
            if fixed {
                let num_arc_bytes = self.scratch_pos - last_arc_start;
                self.num_bytes_per_arc[arc_idx] = num_arc_bytes;
                self.num_label_bytes_per_arc[arc_idx] = num_label_bytes;
                last_arc_start = self.scratch_pos;
                max_bytes_per_arc = max_bytes_per_arc.max(num_arc_bytes);
                max_bytes_per_arc_without_label =
                    max_bytes_per_arc_without_label.max(num_arc_bytes - num_label_bytes);
            }
        }
        self.last_read_len = self.scratch_pos as i64;

        if fixed {
            let first = node_in.arcs[0].label;
            let label_range = (node_in.arcs[last_arc].label - first + 1) as usize;
            let continuous = label_range == num_arcs;
            if continuous && self.version >= VERSION_CONTINUOUS_ARCS {
                self.write_node_for_direct_addressing_or_continuous(
                    node_in,
                    max_bytes_per_arc_without_label,
                    label_range,
                    true,
                );
                self.continuous_node_count += 1;
            } else if self.should_expand_node_with_direct_addressing(
                num_arcs,
                max_bytes_per_arc,
                max_bytes_per_arc_without_label,
                label_range,
            ) {
                self.write_node_for_direct_addressing_or_continuous(
                    node_in,
                    max_bytes_per_arc_without_label,
                    label_range,
                    false,
                );
                self.direct_addressing_node_count += 1;
            } else {
                self.write_node_for_binary_search(num_arcs, max_bytes_per_arc);
                self.binary_search_node_count += 1;
            }
        }

        self.scratch[..self.scratch_pos].reverse();
        if self.padding_byte_pending {
            self.data.push(0);
            self.padding_byte_pending = false;
        }
        self.data
            .extend_from_slice(&self.scratch[..self.scratch_pos]);
        self.num_bytes_written += self.scratch_pos as i64;
        self.node_count += 1;
        self.num_bytes_written - 1
    }

    /// `shouldExpandNodeWithDirectAddressing`.
    // ARITH: node sizes in bytes; Java's `int`/`float` arithmetic.
    #[allow(clippy::arithmetic_side_effects)]
    fn should_expand_node_with_direct_addressing(
        &mut self,
        num_arcs: usize,
        num_bytes_per_arc: usize,
        max_bytes_per_arc_without_label: usize,
        label_range: usize,
    ) -> bool {
        let size_for_binary_search = (num_bytes_per_arc * num_arcs) as i32;
        let size_for_direct_addressing = (num_presence_bytes(label_range)
            + self.num_label_bytes_per_arc[0]
            + max_bytes_per_arc_without_label * num_arcs)
            as i32;
        let allowed_oversize = (size_for_binary_search as f32 * self.da_factor) as i32;
        let expansion_cost = size_for_direct_addressing - allowed_oversize;
        if expansion_cost <= 0
            || (self.da_credit >= i64::from(expansion_cost)
                && size_for_direct_addressing as f32
                    <= allowed_oversize as f32 * DIRECT_ADDRESSING_MAX_OVERSIZE_WITH_CREDIT_FACTOR)
        {
            self.da_credit -= i64::from(expansion_cost);
            return true;
        }
        false
    }

    /// `writeNodeForBinarySearch`: spreads the compact arcs into
    /// `max_bytes_per_arc`-wide slots behind a header, moving each arc right
    /// in place (the gaps keep whatever `scratch` held there).
    // ARITH: slot offsets inside one node, `dest_pos >= src_pos` throughout
    // (Java asserts it).
    #[allow(clippy::arithmetic_side_effects)]
    fn write_node_for_binary_search(&mut self, num_arcs: usize, max_bytes_per_arc: usize) {
        let mut header = vec![ARCS_FOR_BINARY_SEARCH];
        write_vint(&mut header, num_arcs as i32);
        write_vint(&mut header, max_bytes_per_arc as i32);
        self.fixed_buf[..header.len()].copy_from_slice(&header);
        let header_len = header.len();
        let mut src_pos = self.scratch_pos;
        let mut dest_pos = header_len + num_arcs * max_bytes_per_arc;
        if dest_pos > src_pos {
            if self.scratch.len() < dest_pos {
                self.scratch.resize(dest_pos, 0);
            }
            self.scratch_pos = dest_pos;
            for arc_idx in (0..num_arcs).rev() {
                dest_pos -= max_bytes_per_arc;
                let arc_len = self.num_bytes_per_arc[arc_idx];
                src_pos -= arc_len;
                if src_pos != dest_pos {
                    self.scratch
                        .copy_within(src_pos..src_pos + arc_len, dest_pos);
                }
            }
        }
        self.scratch[..header_len].copy_from_slice(&header);
        // A reader stops after the last arc's own bytes, not its padding.
        self.last_read_len = (header_len
            + (num_arcs - 1) * max_bytes_per_arc
            + self.num_bytes_per_arc[num_arcs - 1]) as i64;
    }

    /// `writeNodeForDirectAddressingOrContinuous`.
    // ARITH: offsets inside `fixed_buf`, which is sized to
    // `11 + presence + total_arc_bytes` before any is used.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_node_for_direct_addressing_or_continuous(
        &mut self,
        node_in: &UncompiledNode<O::Value>,
        max_bytes_per_arc_without_label: usize,
        label_range: usize,
        continuous: bool,
    ) {
        let num_arcs = node_in.arcs.len();
        let header_max_len = 11;
        let num_presence = if continuous {
            0
        } else {
            num_presence_bytes(label_range)
        };
        let mut src_pos = self.scratch_pos;
        let total_arc_bytes =
            self.num_label_bytes_per_arc[0] + num_arcs * max_bytes_per_arc_without_label;
        let mut buffer_offset = header_max_len + num_presence + total_arc_bytes;
        if self.fixed_buf.len() < buffer_offset {
            self.fixed_buf = vec![0; oversize_bytes(buffer_offset)];
        }
        for arc_idx in (0..num_arcs).rev() {
            buffer_offset -= max_bytes_per_arc_without_label;
            let src_arc_len = self.num_bytes_per_arc[arc_idx];
            src_pos -= src_arc_len;
            let label_len = self.num_label_bytes_per_arc[arc_idx];
            self.fixed_buf[buffer_offset] = self.scratch[src_pos];
            let remaining = src_arc_len - 1 - label_len;
            if remaining != 0 {
                let from = src_pos + 1 + label_len;
                self.fixed_buf[buffer_offset + 1..buffer_offset + 1 + remaining]
                    .copy_from_slice(&self.scratch[from..from + remaining]);
            }
            if arc_idx == 0 {
                buffer_offset -= label_len;
                self.fixed_buf[buffer_offset..buffer_offset + label_len]
                    .copy_from_slice(&self.scratch[src_pos + 1..src_pos + 1 + label_len]);
            }
        }
        let mut header = vec![if continuous {
            ARCS_FOR_CONTINUOUS
        } else {
            ARCS_FOR_DIRECT_ADDRESSING
        }];
        write_vint(&mut header, label_range as i32);
        write_vint(&mut header, max_bytes_per_arc_without_label as i32);
        self.fixed_buf[..header.len()].copy_from_slice(&header);
        self.scratch_pos = 0;
        self.scratch_write(&header);
        if !continuous {
            self.write_presence_bits(node_in);
        }
        let arcs = self.fixed_buf[buffer_offset..buffer_offset + total_arc_bytes].to_vec();
        self.scratch_write(&arcs);
        let last = num_arcs - 1;
        self.last_read_len = (header.len()
            + num_presence
            + self.num_label_bytes_per_arc[0]
            + last * max_bytes_per_arc_without_label
            + (self.num_bytes_per_arc[last] - self.num_label_bytes_per_arc[last]))
            as i64;
    }

    /// `writePresenceBits`.
    // ARITH: labels are strictly increasing, so each step is positive.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_presence_bits(&mut self, node_in: &UncompiledNode<O::Value>) {
        let mut presence_bits: u8 = 1;
        let mut presence_index = 0i32;
        let mut previous_label = node_in.arcs[0].label;
        for arc in &node_in.arcs[1..] {
            presence_index += arc.label - previous_label;
            while presence_index >= 8 {
                self.scratch_write(&[presence_bits]);
                presence_bits = 0;
                presence_index -= 8;
            }
            presence_bits |= 1 << presence_index;
            previous_label = arc.label;
        }
        self.scratch_write(&[presence_bits]);
    }

    /// `compile()`: freezes what is left and returns the FST, or `None` when
    /// nothing (not even the empty input) was added -- or when this compiler
    /// already compiled (Java's "already finished"). [`Self::stats`] stays
    /// readable afterwards, as in Java.
    pub fn compile(&mut self) -> Option<CompiledFst<O::Value>> {
        if self.finished {
            return None;
        }
        self.freeze_tail(0);
        if self.frontier[0].arcs.is_empty() {
            self.empty_output.as_ref()?;
            debug_assert!(self.padding_byte_pending);
            self.data.push(0);
            self.padding_byte_pending = false;
        }
        let mut start_node = self.compile_node(0);
        if start_node == FINAL_END_NODE && self.empty_output.is_some() {
            start_node = 0;
        }
        debug_assert_eq!(self.num_bytes_written, self.data.len() as i64);
        self.finished = true;
        Some(CompiledFst {
            input_type: self.input_type,
            empty_output: self.empty_output.take(),
            start_node,
            version: self.version,
            bytes: std::mem::take(&mut self.data),
        })
    }
}

/// `FST.getNumPresenceBytes(labelRange)`.
// ARITH: a label range of at most 2^31.
#[allow(clippy::arithmetic_side_effects)]
fn num_presence_bytes(label_range: usize) -> usize {
    (label_range + 7) >> 3
}

impl<V> CompiledFst<V> {
    /// `FST.save(out, out)`: `FSTMetadata.save` (header, reversed empty
    /// output, input type, start node, byte count) then the body.
    pub fn save<O: FstOutputs<Value = V>>(&self) -> Vec<u8> {
        let mut out = Vec::new();
        codec_util::write_header(&mut out, FILE_FORMAT_NAME, VERSION_CURRENT);
        match &self.empty_output {
            Some(v) => {
                out.push(1);
                let mut buf = Vec::new();
                O::write_final_output(v, &mut buf);
                buf.reverse();
                write_vint(&mut out, buf.len() as i32);
                out.extend_from_slice(&buf);
            }
            None => out.push(0),
        }
        out.push(match self.input_type {
            InputType::Byte1 => 0,
            InputType::Byte2 => 1,
            InputType::Byte4 => 2,
        });
        write_vlong(&mut out, self.start_node);
        write_vlong(&mut out, self.bytes.len() as i64);
        out.extend_from_slice(&self.bytes);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn build<O: FstOutputs>(entries: &[(&[u8], O::Value)]) -> CompiledFst<O::Value> {
        let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<O>();
        for (k, v) in entries {
            c.add_bytes(k, v.clone()).unwrap();
        }
        c.compile().unwrap()
    }

    #[test]
    fn outputs_algebra() {
        assert_eq!(PositiveIntOutputs::common(&5, &3), 3);
        assert_eq!(PositiveIntOutputs::common(&0, &3), 0);
        assert_eq!(PositiveIntOutputs::subtract(&5, &3), 2);
        assert_eq!(PositiveIntOutputs::add(&5, &3), 8);
        assert_eq!(
            ByteSequenceOutputs::common(&b"abc".to_vec(), &b"abd".to_vec()),
            b"ab"
        );
        assert_eq!(
            ByteSequenceOutputs::subtract(&b"abc".to_vec(), &b"ab".to_vec()),
            b"c"
        );
        assert_eq!(IntSequenceOutputs::add(&vec![1], &vec![2]), vec![1, 2]);
        assert_eq!(
            IntSequenceOutputs::common(&vec![1, 2], &vec![1, 3]),
            vec![1]
        );
        assert_eq!(IntSequenceOutputs::subtract(&vec![1, 2], &vec![1]), vec![2]);
        assert_eq!(CharSequenceOutputs::common(&vec![7, 8], &vec![7]), vec![7]);
        assert_eq!(
            CharSequenceOutputs::subtract(&vec![7, 8], &vec![7]),
            vec![8]
        );
        assert_eq!(CharSequenceOutputs::add(&vec![7], &vec![8]), vec![7, 8]);
        type P = PairOutputs<PositiveIntOutputs, ByteSequenceOutputs>;
        let a = Pair {
            first: 4i64,
            second: b"xy".to_vec(),
        };
        let b = Pair {
            first: 6i64,
            second: b"xz".to_vec(),
        };
        let c = P::common(&a, &b);
        assert_eq!(
            c,
            Pair {
                first: 4,
                second: b"x".to_vec()
            }
        );
        assert_eq!(P::add(&c, &P::subtract(&b, &c)), b);
        NoOutputs::common(&(), &());
        NoOutputs::subtract(&(), &());
        NoOutputs::add(&(), &());
        assert_eq!(PositiveIntOutputs::merge(&1, &2), None);

        let mut buf = Vec::new();
        P::write(&b, &mut buf);
        IntSequenceOutputs::write(&vec![300, 1], &mut buf);
        CharSequenceOutputs::write(&vec![65], &mut buf);
        NoOutputs::write(&(), &mut buf);
        buf.reverse();
        let mut r = ReverseReader::new(&buf);
        r.set_position(buf.len() as i64 - 1);
        assert_eq!(P::read(&mut r).unwrap(), b);
        assert_eq!(IntSequenceOutputs::read(&mut r).unwrap(), vec![300, 1]);
        assert_eq!(
            CharSequenceOutputs::read_final_output(&mut r).unwrap(),
            vec![65]
        );
        NoOutputs::read(&mut r).unwrap();
        assert_eq!(r.position(), -1);
        assert!(r.read_byte().is_err());
    }

    #[test]
    fn reader_errors() {
        let bad = [0x80u8; 12];
        let mut r = ReverseReader::new(&bad);
        r.set_position(11);
        assert!(r.read_vlong().is_err());
        let neg = [0x0f, 0xff, 0xff, 0xff, 0xff];
        let mut r = ReverseReader::new(&neg);
        r.set_position(4);
        assert!(read_len(&mut r).is_err());
        let s = [0x01, 0x02];
        let mut r = ReverseReader::new(&s);
        r.set_position(1);
        assert_eq!(r.read_short().unwrap(), i16::from_le_bytes([2, 1]));
        r.set_position(1);
        r.skip_bytes(1);
        assert_eq!(r.read_byte().unwrap(), 1);
    }

    #[test]
    fn builder_and_input_checks() {
        assert!(FstCompilerBuilder::new(InputType::Byte1)
            .suffix_ram_limit_mb(-1.0)
            .is_err());
        assert!(FstCompilerBuilder::new(InputType::Byte1)
            .version(7)
            .is_err());
        assert!(FstCompilerBuilder::new(InputType::Byte1).version(8).is_ok());
        let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<PositiveIntOutputs>();
        assert!(matches!(
            c.add(&[256], 1),
            Err(CompileError::LabelOutOfRange { .. })
        ));
        c.add_bytes(b"b", 1).unwrap();
        assert_eq!(c.add_bytes(b"a", 1), Err(CompileError::OutOfOrder));
        assert_eq!(c.add_bytes(b"b", 2), Err(CompileError::MergeUnsupported));
        let mut e = FstCompilerBuilder::new(InputType::Byte2).build::<PositiveIntOutputs>();
        e.add(&[], 3).unwrap();
        assert_eq!(e.add(&[], 4), Err(CompileError::MergeUnsupported));
        assert!(e.add(&[70000], 1).is_err());
        e.add(&[65535], 1).unwrap();
        assert!(FstCompilerBuilder::new(InputType::Byte4)
            .build::<NoOutputs>()
            .add(&[-1], ())
            .is_err());
        let mut none = FstCompilerBuilder::new(InputType::Byte1).build::<NoOutputs>();
        assert!(none.compile().is_none());
    }

    #[test]
    fn shared_suffixes_and_pushed_outputs() {
        let f =
            build::<PositiveIntOutputs>(&[(b"cat", 5), (b"cats", 7), (b"dog", 5), (b"dogs", 7)]);
        // "at"/"og" tails share one "s"-final node; outputs pushed to the root.
        assert!(f.start_node > 0);
        let saved = f.save::<PositiveIntOutputs>();
        assert!(saved.len() > f.bytes.len());
        let empty = {
            let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<ByteSequenceOutputs>();
            c.add_bytes(b"", b"e".to_vec()).unwrap();
            c.compile().unwrap()
        };
        assert_eq!(empty.start_node, 0);
        assert_eq!(empty.bytes, vec![0]);
        // ... accepts-empty, len 2, reversed [len 1, 'e'], BYTE1, start 0, 1 byte, body.
        let saved = empty.save::<ByteSequenceOutputs>();
        assert!(saved.ends_with(&[1, 2, b'e', 1, 0, 0, 1, 0]), "{saved:?}");
    }

    #[test]
    fn no_outputs_merge_duplicates_and_fixed_nodes() {
        let mut c = FstCompilerBuilder::new(InputType::Byte1).build::<NoOutputs>();
        c.add_bytes(b"", ()).unwrap();
        c.add_bytes(b"", ()).unwrap();
        for k in [
            b"a".as_slice(),
            b"a",
            b"b",
            b"c",
            b"d",
            b"e",
            b"f",
            b"h",
            b"z",
        ] {
            c.add_bytes(k, ()).unwrap();
        }
        let f = c.compile().unwrap();
        assert!(c.compile().is_none(), "already finished");
        let s = c.stats();
        assert_eq!(s.arc_count, 8);
        assert_eq!(s.node_count, 2);
        assert_eq!(
            s.binary_search_node_count + s.direct_addressing_node_count,
            1
        );
        assert_eq!(c.fst_size_in_bytes(), f.bytes.len() as i64);
        assert_eq!(oversize_bytes(0), 0);
        assert_eq!(oversize_bytes(20), 24);
        assert_eq!(bits_required(0), 1);
        assert_eq!(bits_required(255), 8);
    }
}
