//! Port of `org.apache.lucene.util.bkd.BKDWriter`, byte-identical to Lucene
//! 10.5.0, with the pieces it drives: `DocIdsWriter`'s write side (every
//! encoding), `MutablePointTreeReaderUtils` (sort, `sortByDim`,
//! `partition`), `BKDRadixSelector` (heap and offline selection,
//! `heapRadixSort`), `HeapPointWriter`, and `OfflinePointWriter`/`Reader`.
//!
//! All three of Java's entry points are here, as each writes different bytes:
//!
//! - [`BkdWriter::write_field`] -- the flush path, over a
//!   [`MutablePointTree`] (`writeField1Dim` for one dimension,
//!   `writeFieldNDims` otherwise);
//! - [`BkdWriter::add`] + [`BkdWriter::finish`] -- points streamed in, spilled
//!   to temp files past `maxMBSortInHeap`, the tree built by radix selection
//!   (what `Lucene90PointsWriter` uses to merge multi-dimensional fields);
//! - [`BkdWriter::merge`] -- the one-dimensional merge of sorted sources.
//!
//! Where points compare equal under the key an algorithm sorts or selects by
//! (the keys omit the non-split index dimensions), the arrangement they end
//! up in is written to disk; that is why the selection and sort algorithms
//! are swap-for-swap ports (`lucene_util::sorter`) rather than calls to
//! `slice::sort`. `IntroSelector`'s unseeded shuffle (reached only when its
//! recursion budget runs out) is the one place Java's own output is not
//! reproducible; this port seeds it.

use std::cmp::Ordering;

use lucene_store::codec_util::{self, FOOTER_LENGTH, FOOTER_MAGIC};
use lucene_store::data_output::DataOutput;
use lucene_store::directory::Directory;
use lucene_store::index_output::{FsIndexOutput, IndexOutput};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::numeric_utils;
use lucene_util::sorter::{
    intro_select, intro_sort, msb_radix_sort, radix_select, IntroTarget, RadixTarget,
};
use lucene_util::splittable_random::SplittableRandom;

/// `BKDWriter.CODEC_NAME`.
pub const CODEC_NAME: &str = "BKD";
/// `BKDWriter.VERSION_META_FILE`.
pub const VERSION_META_FILE: i32 = 9;
/// `BKDWriter.VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21`.
pub const VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21: i32 = 10;
/// `BKDWriter.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21;
/// `BKDWriter.DEFAULT_MAX_MB_SORT_IN_HEAP`.
pub const DEFAULT_MAX_MB_SORT_IN_HEAP: f64 = 16.0;
/// `BKDConfig.DEFAULT_MAX_POINTS_IN_LEAF_NODE`.
pub const DEFAULT_MAX_POINTS_IN_LEAF_NODE: usize = 512;
const SPLITS_BEFORE_EXACT_BOUNDS: i32 = 4;

/// Errors of the BKD writer.
#[derive(Debug, thiserror::Error)]
pub enum BkdError {
    /// A configuration or call Java rejects.
    #[error("{0}")]
    IllegalArgument(String),
    /// Temp-file I/O.
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
}

type Result<T> = std::result::Result<T, BkdError>;

fn illegal(msg: impl Into<String>) -> BkdError {
    BkdError::IllegalArgument(msg.into())
}

/// `org.apache.lucene.util.bkd.BKDConfig`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BkdConfig {
    /// `numDims`.
    pub num_dims: usize,
    /// `numIndexDims`.
    pub num_index_dims: usize,
    /// `bytesPerDim`.
    pub bytes_per_dim: usize,
    /// `maxPointsInLeafNode`.
    pub max_points_in_leaf_node: usize,
}

impl BkdConfig {
    /// `new BKDConfig(...)` with Java's checks.
    pub fn new(
        num_dims: usize,
        num_index_dims: usize,
        bytes_per_dim: usize,
        max_points_in_leaf_node: usize,
    ) -> Result<Self> {
        if !(1..=16).contains(&num_dims) {
            return Err(illegal(format!(
                "numDims must be 1 .. 16 (got: {num_dims})"
            )));
        }
        if !(1..=8).contains(&num_index_dims) {
            return Err(illegal(format!(
                "numIndexDims must be 1 .. 8 (got: {num_index_dims})"
            )));
        }
        if num_index_dims > num_dims {
            return Err(illegal(format!(
                "numIndexDims cannot exceed numDims ({num_dims}) (got: {num_index_dims})"
            )));
        }
        if bytes_per_dim == 0 || bytes_per_dim > 16 {
            return Err(illegal(format!(
                "bytesPerDim must be 1 .. 16; got {bytes_per_dim}"
            )));
        }
        if max_points_in_leaf_node == 0 || max_points_in_leaf_node > (i32::MAX - 8) as usize {
            return Err(illegal(format!(
                "maxPointsInLeafNode must be > 0; got {max_points_in_leaf_node}"
            )));
        }
        Ok(BkdConfig {
            num_dims,
            num_index_dims,
            bytes_per_dim,
            max_points_in_leaf_node,
        })
    }
    /// `packedBytesLength()`.
    // ARITH: at most 16 * 16.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn packed_bytes_length(&self) -> usize {
        self.num_dims * self.bytes_per_dim
    }
    /// `packedIndexBytesLength()`.
    // ARITH: at most 8 * 16.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn packed_index_bytes_length(&self) -> usize {
        self.num_index_dims * self.bytes_per_dim
    }
    /// `bytesPerDoc()`: packed value plus a 4-byte doc id.
    // ARITH: at most 256 + 4.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn bytes_per_doc(&self) -> usize {
        self.packed_bytes_length() + 4
    }
}

// --- helpers -------------------------------------------------------------

/// `BKDUtil.commonPrefixLength*`: length of the common prefix of two
/// `n`-byte values.
fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
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

/// `FixedBitSet.bits2words`.
// ARITH: a doc-id range.
#[allow(clippy::arithmetic_side_effects)]
fn bits2words(n: i64) -> i64 {
    if n == 0 {
        0
    } else {
        ((n - 1) >> 6) + 1
    }
}

// --- DocIdsWriter ----------------------------------------------------------

const CONTINUOUS_IDS: i8 = -2;
const BITSET_IDS: i8 = -1;
const DELTA_BPV_16: i8 = 16;
const BPV_21: i8 = 21;
const BPV_24: i8 = 24;
const BPV_32: i8 = 32;

/// `DocIdsWriter.writeDocIds(docIds, 0, count, out)`: picks the encoding
/// exactly as Java does (continuous run, bitset for dense sorted ids, 16-bit
/// deltas, 21-/24-bit packing on `VERSION_VECTORIZE_...` and later, else 32
/// bits).
// ARITH: doc ids are non-negative `i32`s; ranges and shifts are Java's `int`
// arithmetic on them.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_doc_ids(doc_ids: &[i32], version: i32, out: &mut Vec<u8>) {
    let count = doc_ids.len();
    let mut strictly_sorted = true;
    let mut min = doc_ids[0];
    let mut max = doc_ids[0];
    for i in 1..count {
        let last = doc_ids[i - 1];
        let current = doc_ids[i];
        if last >= current {
            strictly_sorted = false;
        }
        min = min.min(current);
        max = max.max(current);
    }
    let min2max = max.wrapping_sub(min).wrapping_add(1);
    if strictly_sorted {
        if min2max as usize == count {
            out.push(CONTINUOUS_IDS as u8);
            write_vint(out, doc_ids[0]);
            return;
        } else if i64::from(min2max) <= (count as i64) << 4 {
            out.push(BITSET_IDS as u8);
            write_ids_as_bitset(doc_ids, out);
            return;
        }
    }
    if min2max <= 0xFFFF {
        out.push(DELTA_BPV_16 as u8);
        let mut scratch: Vec<i32> = doc_ids.iter().map(|d| d - min).collect();
        write_vint(out, min);
        let half = count >> 1;
        for i in 0..half {
            scratch[i] = scratch[half + i] | (scratch[i] << 16);
        }
        for s in &scratch[..half] {
            out.extend_from_slice(&s.to_le_bytes());
        }
        if count & 1 == 1 {
            out.extend_from_slice(&(scratch[count - 1] as i16).to_le_bytes());
        }
    } else if max <= 0x1FFFFF && version >= VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21 {
        out.push(BPV_21 as u8);
        let one_third = (count / 3) & !0xF;
        let num_ints = one_third * 2;
        let mut scratch: Vec<i32> = doc_ids[..num_ints].iter().map(|d| d << 11).collect();
        for i in 0..one_third {
            let d = doc_ids[i + num_ints];
            scratch[i] |= d & 0x7FF;
            scratch[i + one_third] |= ((d as u32 >> 11) & 0x7FF) as i32;
        }
        for s in &scratch {
            out.extend_from_slice(&s.to_le_bytes());
        }
        let mut i = one_third * 3;
        while i + 2 < count {
            let l = i64::from(doc_ids[i])
                | (i64::from(doc_ids[i + 1]) << 21)
                | (i64::from(doc_ids[i + 2]) << 42);
            out.extend_from_slice(&l.to_le_bytes());
            i += 3;
        }
        while i < count {
            out.extend_from_slice(&(doc_ids[i] as i16).to_le_bytes());
            out.push((doc_ids[i] as u32 >> 16) as u8);
            i += 1;
        }
    } else if max <= 0xFFFFFF {
        out.push(BPV_24 as u8);
        if version < VERSION_VECTORIZE_BPV24_AND_INTRODUCE_BPV21 {
            write_scalar_ints24(doc_ids, out);
        } else {
            let quarter = count >> 2;
            let num_ints = quarter * 3;
            let mut scratch: Vec<i32> = doc_ids[..num_ints].iter().map(|d| d << 8).collect();
            for i in 0..quarter {
                let d = doc_ids[i + num_ints];
                scratch[i] |= d & 0xFF;
                scratch[i + quarter] |= ((d as u32 >> 8) & 0xFF) as i32;
                scratch[i + quarter * 2] |= (d as u32 >> 16) as i32;
            }
            for s in &scratch {
                out.extend_from_slice(&s.to_le_bytes());
            }
            for &d in &doc_ids[quarter << 2..] {
                out.extend_from_slice(&(d as i16).to_le_bytes());
                out.push((d as u32 >> 16) as u8);
            }
        }
    } else {
        out.push(BPV_32 as u8);
        for d in doc_ids {
            out.extend_from_slice(&d.to_le_bytes());
        }
    }
}

// ARITH: 24-bit packing of non-negative doc ids.
#[allow(clippy::arithmetic_side_effects)]
fn write_scalar_ints24(doc_ids: &[i32], out: &mut Vec<u8>) {
    let count = doc_ids.len();
    let mut i = 0;
    while i + 7 < count {
        let d: Vec<i64> = doc_ids[i..i + 8].iter().map(|&x| i64::from(x)).collect();
        let l1 = (d[0] & 0xffffff) << 40 | (d[1] & 0xffffff) << 16 | ((d[2] >> 8) & 0xffff);
        let l2 = (d[2] & 0xff) << 56
            | (d[3] & 0xffffff) << 32
            | (d[4] & 0xffffff) << 8
            | ((d[5] >> 16) & 0xff);
        let l3 = (d[5] & 0xffff) << 48 | (d[6] & 0xffffff) << 24 | (d[7] & 0xffffff);
        out.extend_from_slice(&l1.to_le_bytes());
        out.extend_from_slice(&l2.to_le_bytes());
        out.extend_from_slice(&l3.to_le_bytes());
        i += 8;
    }
    for &d in &doc_ids[i..] {
        out.extend_from_slice(&((d as u32 >> 8) as i16).to_le_bytes());
        out.push(d as u8);
    }
}

// ARITH: word indices over a doc-id range already bounded by 16 * count.
#[allow(clippy::arithmetic_side_effects)]
fn write_ids_as_bitset(doc_ids: &[i32], out: &mut Vec<u8>) {
    let min = doc_ids[0];
    let max = doc_ids[doc_ids.len() - 1];
    let offset_words = min >> 6;
    let offset_bits = offset_words << 6;
    let total_word_count = bits2words(i64::from(max - offset_bits + 1));
    write_vint(out, offset_words);
    write_vint(out, total_word_count as i32);
    let mut current_word = 0u64;
    let mut current_word_index = 0i32;
    for &d in doc_ids {
        let index = d - offset_bits;
        let next_word_index = index >> 6;
        if current_word_index < next_word_index {
            out.extend_from_slice(&current_word.to_le_bytes());
            current_word = 0;
            current_word_index += 1;
            while current_word_index < next_word_index {
                current_word_index += 1;
                out.extend_from_slice(&0u64.to_le_bytes());
            }
        }
        current_word |= 1u64 << (index & 63);
    }
    out.extend_from_slice(&current_word.to_le_bytes());
}

// --- MutablePointTree -----------------------------------------------------

/// `org.apache.lucene.codecs.MutablePointTree` over flat arrays: what the
/// flush path hands the writer (`PointValuesWriter`'s buffered points, in
/// insertion order). The writer reorders it in place.
#[derive(Debug, Clone, Default)]
pub struct MutablePointTree {
    /// Doc id of point `i`.
    pub docs: Vec<i32>,
    /// Packed value of point `i` at `i * stride..`.
    pub values: Vec<u8>,
    stride: usize,
}

impl MutablePointTree {
    /// An empty tree of `packed_bytes_length`-byte values.
    pub fn new(packed_bytes_length: usize) -> Self {
        MutablePointTree {
            docs: Vec::new(),
            values: Vec::new(),
            stride: packed_bytes_length,
        }
    }
    /// Appends a point.
    pub fn push(&mut self, packed_value: &[u8], doc: i32) {
        self.values.extend_from_slice(packed_value);
        self.docs.push(doc);
    }
    /// `size()`.
    pub fn len(&self) -> usize {
        self.docs.len()
    }
    /// `size() == 0`.
    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }
    /// `getValue(i)`.
    // ARITH: i < len.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn value(&self, i: usize) -> &[u8] {
        &self.values[i * self.stride..(i + 1) * self.stride]
    }
    // ARITH: i < len, k < stride.
    #[allow(clippy::arithmetic_side_effects)]
    fn byte_at(&self, i: usize, k: usize) -> u8 {
        self.values[i * self.stride + k]
    }
    // ARITH: i, j < len.
    #[allow(clippy::arithmetic_side_effects)]
    fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        self.docs.swap(i, j);
        let s = self.stride;
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let (lo, hi) = self.values.split_at_mut(b * s);
        lo[a * s..(a + 1) * s].swap_with_slice(&mut hi[..s]);
    }
}

// --- point stores ----------------------------------------------------------

/// `HeapPointWriter`: points as `packed value + big-endian doc id` records.
#[derive(Debug)]
struct HeapPoints {
    block: Vec<u8>,
    size: usize,
    next_write: usize,
    bpd: usize,
}

impl HeapPoints {
    // ARITH: an in-memory point count.
    #[allow(clippy::arithmetic_side_effects)]
    fn new(config: &BkdConfig, size: usize) -> Self {
        HeapPoints {
            block: vec![0; config.bytes_per_doc() * size],
            size,
            next_write: 0,
            bpd: config.bytes_per_doc(),
        }
    }
    // ARITH: next_write < size.
    #[allow(clippy::arithmetic_side_effects)]
    fn append(&mut self, packed_value_doc: &[u8]) {
        let pos = self.next_write * self.bpd;
        self.block[pos..pos + self.bpd].copy_from_slice(packed_value_doc);
        self.next_write += 1;
    }
    // ARITH: i < size.
    #[allow(clippy::arithmetic_side_effects)]
    fn record(&self, i: usize) -> &[u8] {
        &self.block[i * self.bpd..(i + 1) * self.bpd]
    }
    // ARITH: i < size, k < bpd.
    #[allow(clippy::arithmetic_side_effects)]
    fn byte_at(&self, i: usize, k: usize) -> i32 {
        i32::from(self.block[i * self.bpd + k])
    }
    // ARITH: i, j < size.
    #[allow(clippy::arithmetic_side_effects)]
    fn swap(&mut self, i: usize, j: usize) {
        if i == j {
            return;
        }
        let s = self.bpd;
        let (a, b) = if i < j { (i, j) } else { (j, i) };
        let (lo, hi) = self.block.split_at_mut(b * s);
        lo[a * s..(a + 1) * s].swap_with_slice(&mut hi[..s]);
    }
}

/// `OfflinePointWriter`: a temp file of records, closed by a codec footer.
struct OfflinePoints {
    name: String,
    out: Option<FsIndexOutput>,
    count: u64,
}

enum PointStore {
    Heap(HeapPoints),
    Offline(OfflinePoints),
}

type SharedStore = std::rc::Rc<std::cell::RefCell<PointStore>>;

/// `BKDRadixSelector.PathSlice`.
#[derive(Clone)]
struct PathSlice {
    writer: SharedStore,
    start: u64,
    count: u64,
}

fn shared(store: PointStore) -> SharedStore {
    std::rc::Rc::new(std::cell::RefCell::new(store))
}

/// Where the tree's leaves ended up, to write the index from
/// (`BKDTreeLeafNodes` plus what `writeIndex` needs).
#[derive(Debug, Clone)]
pub struct BkdIndexPlan {
    leaf_fps: Vec<i64>,
    /// Split value `i` at `i * bytes_per_dim`.
    split_values: Vec<u8>,
    split_dims: Vec<u8>,
    data_start_fp: i64,
}

/// `Long.toString(n, 36)`.
// ARITH: digit extraction.
#[allow(clippy::arithmetic_side_effects)]
fn base36(mut n: u64) -> String {
    if n == 0 {
        return "0".into();
    }
    let mut d = Vec::new();
    while n > 0 {
        let x = (n % 36) as u8;
        d.push(if x < 10 { b'0' + x } else { b'a' + x - 10 });
        n /= 36;
    }
    d.reverse();
    String::from_utf8(d).unwrap_or_default()
}

/// `org.apache.lucene.util.bkd.BKDWriter`.
pub struct BkdWriter<'d> {
    config: BkdConfig,
    version: i32,
    total_point_count: u64,
    max_points_sort_in_heap: usize,
    temp_dir: Option<&'d dyn Directory>,
    temp_prefix: String,
    next_temp: u64,
    docs_seen: FixedBitSet,
    min_packed: Vec<u8>,
    max_packed: Vec<u8>,
    point_count: u64,
    finished: bool,
    point_writer: Option<PointStore>,
    common_prefix_lengths: Vec<usize>,
    random: SplittableRandom,
}

impl<'d> BkdWriter<'d> {
    /// `new BKDWriter(maxDoc, tempDir, tempFileNamePrefix, config,
    /// maxMBSortInHeap, totalPointCount, version)`. `temp_dir` is needed only
    /// when `add` spills (more than `maxMBSortInHeap` worth of points).
    pub fn new(
        max_doc: usize,
        temp_dir: Option<&'d dyn Directory>,
        temp_prefix: &str,
        config: BkdConfig,
        max_mb_sort_in_heap: f64,
        total_point_count: u64,
        version: i32,
    ) -> Result<Self> {
        if !(4..=VERSION_CURRENT).contains(&version) {
            return Err(illegal(format!("Version out of range: {version}")));
        }
        if max_mb_sort_in_heap.is_nan() || max_mb_sort_in_heap < 0.0 {
            return Err(illegal(format!(
                "maxMBSortInHeap must be >= 0.0 (got: {max_mb_sort_in_heap})"
            )));
        }
        let max_points_sort_in_heap =
            ((max_mb_sort_in_heap * 1024.0 * 1024.0) / config.bytes_per_doc() as f64) as i32;
        if max_points_sort_in_heap < config.max_points_in_leaf_node as i32 {
            return Err(illegal(format!(
                "maxMBSortInHeap={max_mb_sort_in_heap} only allows for maxPointsSortInHeap={max_points_sort_in_heap}, \
                 but this is less than maxPointsInLeafNode={}; either increase maxMBSortInHeap or decrease maxPointsInLeafNode",
                config.max_points_in_leaf_node
            )));
        }
        Ok(BkdWriter {
            config,
            version,
            total_point_count,
            max_points_sort_in_heap: max_points_sort_in_heap as usize,
            temp_dir,
            temp_prefix: temp_prefix.to_string(),
            next_temp: 0,
            docs_seen: FixedBitSet::new(max_doc),
            min_packed: vec![0; config.packed_index_bytes_length()],
            max_packed: vec![0; config.packed_index_bytes_length()],
            point_count: 0,
            finished: false,
            point_writer: None,
            common_prefix_lengths: vec![0; config.num_dims],
            random: SplittableRandom::new(0x5eed),
        })
    }

    /// The number of points written.
    pub fn point_count(&self) -> u64 {
        self.point_count
    }

    // --- temp files ------------------------------------------------------

    /// `Directory.createTempOutput(prefix, "bkd_" + desc, ctx)`.
    // ARITH: a temp-file counter.
    #[allow(clippy::arithmetic_side_effects)]
    fn create_temp(&mut self, desc: &str) -> Result<OfflinePoints> {
        let dir = self
            .temp_dir
            .ok_or_else(|| illegal("spilling points needs a temp directory"))?;
        let existing = dir.list_all()?;
        loop {
            let n = self.next_temp;
            self.next_temp += 1;
            let name = format!("{}_bkd_{desc}_{}.tmp", self.temp_prefix, base36(n));
            if !existing.contains(&name) {
                let out = dir.create_output(&name)?;
                return Ok(OfflinePoints {
                    name,
                    out: Some(out),
                    count: 0,
                });
            }
        }
    }

    fn close_store(&self, store: &mut PointStore) -> Result<()> {
        if let PointStore::Offline(o) = store {
            if let Some(mut out) = o.out.take() {
                out.write_bytes(&FOOTER_MAGIC.to_be_bytes());
                out.write_bytes(&0u32.to_be_bytes());
                let c = out.checksum();
                out.write_bytes(&c.to_be_bytes());
                out.close()?;
            }
        }
        Ok(())
    }

    fn destroy_store(&self, store: &PointStore) -> Result<()> {
        if let (PointStore::Offline(o), Some(dir)) = (store, self.temp_dir) {
            dir.delete_file(&o.name)?;
        }
        Ok(())
    }

    // ARITH: a point count.
    #[allow(clippy::arithmetic_side_effects)]
    fn store_append(store: &mut PointStore, record: &[u8]) {
        match store {
            PointStore::Heap(h) => h.append(record),
            PointStore::Offline(o) => {
                if let Some(out) = o.out.as_mut() {
                    out.write_bytes(record);
                }
                o.count += 1;
            }
        }
    }

    /// The records `start..start + count` of an offline store (read back
    /// whole; the footer is verified).
    // ARITH: record offsets.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_offline(&self, name: &str, start: u64, count: u64) -> Result<Vec<u8>> {
        let dir = self.temp_dir.ok_or_else(|| illegal("no temp directory"))?;
        let input = dir.open(name)?;
        let bpd = self.config.bytes_per_doc() as u64;
        let body = input.len().saturating_sub(FOOTER_LENGTH) as u64;
        if (start + count) * bpd > body {
            return Err(illegal(format!(
                "requested slice is beyond the length of this file: start={start} length={count} tempFileName={name}"
            )));
        }
        let mut footer = lucene_store::data_input::SliceInput::new(&input);
        footer.seek(input.len() - FOOTER_LENGTH)?;
        codec_util::check_footer(&mut footer, input.len())?;
        Ok(input[(start * bpd) as usize..((start + count) * bpd) as usize].to_vec())
    }

    /// `getPointWriter(count, desc)`: on heap when at most half the heap
    /// budget, else a temp file.
    fn get_point_writer(&mut self, count: u64, desc: &str) -> Result<PointStore> {
        if count <= (self.max_points_sort_in_heap / 2) as u64 {
            Ok(PointStore::Heap(HeapPoints::new(
                &self.config,
                count as usize,
            )))
        } else {
            Ok(PointStore::Offline(self.create_temp(desc)?))
        }
    }

    // --- add / finish ----------------------------------------------------------

    /// `BKDWriter.add(packedValue, docID)`.
    // ARITH: dimension offsets, a point count.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn add(&mut self, packed_value: &[u8], doc_id: i32) -> Result<()> {
        let c = self.config;
        if packed_value.len() != c.packed_bytes_length() {
            return Err(illegal(format!(
                "packedValue should be length={} (got: {})",
                c.packed_bytes_length(),
                packed_value.len()
            )));
        }
        if self.point_count >= self.total_point_count {
            return Err(illegal(format!(
                "totalPointCount={} was passed when we were created, but we just hit {} values",
                self.total_point_count,
                self.point_count + 1
            )));
        }
        if self.point_count == 0 {
            let writer = if self.total_point_count > self.max_points_sort_in_heap as u64 {
                PointStore::Offline(self.create_temp("spill")?)
            } else {
                PointStore::Heap(HeapPoints::new(&c, self.total_point_count as usize))
            };
            self.point_writer = Some(writer);
            let n = c.packed_index_bytes_length();
            self.min_packed.copy_from_slice(&packed_value[..n]);
            self.max_packed.copy_from_slice(&packed_value[..n]);
        } else {
            for dim in 0..c.num_index_dims {
                let o = dim * c.bytes_per_dim;
                let v = &packed_value[o..o + c.bytes_per_dim];
                if v < &self.min_packed[o..o + c.bytes_per_dim] {
                    self.min_packed[o..o + c.bytes_per_dim].copy_from_slice(v);
                } else if v > &self.max_packed[o..o + c.bytes_per_dim] {
                    self.max_packed[o..o + c.bytes_per_dim].copy_from_slice(v);
                }
            }
        }
        let mut record = packed_value.to_vec();
        record.extend_from_slice(&doc_id.to_be_bytes());
        if let Some(w) = self.point_writer.as_mut() {
            Self::store_append(w, &record);
        }
        self.point_count += 1;
        // FBS: Java's `docsSeen.set(docID)` over a `FixedBitSet(maxDoc)`;
        // an id past `maxDoc` is the caller's contract violation.
        if (doc_id as usize) < self.docs_seen.len() {
            self.docs_seen.set(doc_id as usize);
        }
        Ok(())
    }

    /// `BKDWriter.finish(metaOut, indexOut, dataOut)`: builds the tree of the
    /// added points into `data`; `None` when nothing was added.
    // ARITH: leaf counts from the point count.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn finish(&mut self, data: &mut Vec<u8>) -> Result<Option<BkdIndexPlan>> {
        if self.finished {
            return Err(illegal("already finished"));
        }
        if self.point_count == 0 {
            return Ok(None);
        }
        self.finished = true;
        let mut writer = self
            .point_writer
            .take()
            .ok_or_else(|| illegal("no points"))?;
        self.close_store(&mut writer)?;
        let c = self.config;
        let max_leaf = c.max_points_in_leaf_node as u64;
        let num_leaves = self.point_count.div_ceil(max_leaf) as usize;
        let mut split_values = vec![0u8; (num_leaves - 1) * c.bytes_per_dim];
        let mut split_dims = vec![0u8; num_leaves - 1];
        let mut leaf_fps: Vec<i64> = Vec::with_capacity(num_leaves);
        let data_start_fp = data.len() as i64;
        let points = PathSlice {
            writer: shared(writer),
            start: 0,
            count: self.point_count,
        };
        let mut parent_splits = vec![0i32; c.num_index_dims];
        let (min, max) = (self.min_packed.clone(), self.max_packed.clone());
        self.build_path(
            0,
            num_leaves,
            points,
            data,
            min,
            max,
            &mut parent_splits,
            &mut split_values,
            &mut split_dims,
            &mut leaf_fps,
            num_leaves,
        )?;
        Ok(Some(BkdIndexPlan {
            leaf_fps,
            split_values,
            split_dims,
            data_start_fp,
        }))
    }

    /// `BKDWriter.split`: the dimension to split on.
    // ARITH: split counts and offsets.
    #[allow(clippy::arithmetic_side_effects)]
    fn split(&self, min: &[u8], max: &[u8], parent_splits: &[i32]) -> usize {
        let c = self.config;
        let max_num_splits = parent_splits.iter().copied().max().unwrap_or(0);
        for (dim, &splits) in parent_splits.iter().enumerate().take(c.num_index_dims) {
            let o = dim * c.bytes_per_dim;
            if splits < max_num_splits / 2
                && min[o..o + c.bytes_per_dim] != max[o..o + c.bytes_per_dim]
            {
                return dim;
            }
        }
        let mut split_dim = usize::MAX;
        let mut best = vec![0u8; c.bytes_per_dim];
        let mut diff = vec![0u8; c.bytes_per_dim];
        for dim in 0..c.num_index_dims {
            // `NumericUtils.subtract` cannot underflow: max >= min.
            let _ = numeric_utils::subtract(c.bytes_per_dim, dim, max, min, &mut diff);
            if split_dim == usize::MAX || diff > best {
                best.copy_from_slice(&diff);
                split_dim = dim;
            }
        }
        split_dim
    }

    /// `BKDWriter.build(... PathSlice ...)`.
    #[allow(clippy::too_many_arguments, clippy::arithmetic_side_effects)]
    // ARITH: leaf counts and offsets inside the tree being built.
    fn build_path(
        &mut self,
        leaves_offset: usize,
        num_leaves: usize,
        points: PathSlice,
        out: &mut Vec<u8>,
        mut min_packed: Vec<u8>,
        mut max_packed: Vec<u8>,
        parent_splits: &mut [i32],
        split_values: &mut [u8],
        split_dims: &mut [u8],
        leaf_fps: &mut Vec<i64>,
        total_num_leaves: usize,
    ) -> Result<()> {
        let c = self.config;
        if num_leaves == 1 {
            // switchToHeap when the slice lives in a temp file.
            let is_heap = matches!(&*points.writer.borrow(), PointStore::Heap(_));
            let (heap_rc, from, to) = if is_heap {
                (
                    points.writer.clone(),
                    points.start as usize,
                    (points.start + points.count) as usize,
                )
            } else {
                let name = match &*points.writer.borrow() {
                    PointStore::Offline(o) => o.name.clone(),
                    PointStore::Heap(_) => unreachable!("checked above"),
                };
                let bytes = self.read_offline(&name, points.start, points.count)?;
                let mut heap = HeapPoints::new(&c, points.count as usize);
                for r in bytes.chunks(c.bytes_per_doc()) {
                    heap.append(r);
                }
                self.destroy_store(&points.writer.borrow())?;
                (shared(PointStore::Heap(heap)), 0, points.count as usize)
            };
            let mut store = heap_rc.borrow_mut();
            let PointStore::Heap(heap) = &mut *store else {
                unreachable!("a heap store");
            };
            // computeCommonPrefixLength: `scratch` keeps the first record.
            let first: Vec<u8> = heap.record(from)[..c.packed_bytes_length()].to_vec();
            self.common_prefix_lengths.fill(c.bytes_per_dim);
            for i in from + 1..to {
                let r = heap.record(i);
                for dim in 0..c.num_dims {
                    if self.common_prefix_lengths[dim] != 0 {
                        let o = dim * c.bytes_per_dim;
                        let p = common_prefix(
                            &first[o..o + c.bytes_per_dim],
                            &r[o..o + c.bytes_per_dim],
                        );
                        self.common_prefix_lengths[dim] = self.common_prefix_lengths[dim].min(p);
                    }
                }
            }
            let mut sorted_dim = 0;
            let mut sorted_dim_cardinality = usize::MAX;
            for dim in 0..c.num_dims {
                let prefix = self.common_prefix_lengths[dim];
                if prefix < c.bytes_per_dim {
                    let o = dim * c.bytes_per_dim;
                    let mut used = [false; 256];
                    for i in from..to {
                        used[heap.record(i)[o + prefix] as usize] = true;
                    }
                    let cardinality = used.iter().filter(|u| **u).count();
                    if cardinality < sorted_dim_cardinality {
                        sorted_dim = dim;
                        sorted_dim_cardinality = cardinality;
                    }
                }
            }
            let prefix = self.common_prefix_lengths[sorted_dim];
            self.heap_radix_sort(heap, from, to, sorted_dim, prefix);
            // HeapPointWriter.computeCardinality
            let mut leaf_cardinality = 1;
            for i in from + 1..to {
                let (prev, next) = (heap.record(i - 1), heap.record(i));
                for dim in 0..c.num_dims {
                    let s = dim * c.bytes_per_dim + self.common_prefix_lengths[dim];
                    let e = dim * c.bytes_per_dim + c.bytes_per_dim;
                    if prev[s..e] != next[s..e] {
                        leaf_cardinality += 1;
                        break;
                    }
                }
            }
            leaf_fps.push(out.len() as i64);
            let count = to - from;
            let mut docs = Vec::with_capacity(count);
            let mut values = Vec::with_capacity(count * c.packed_bytes_length());
            for i in from..to {
                let r = heap.record(i);
                let p = c.packed_bytes_length();
                values.extend_from_slice(&r[..p]);
                docs.push(i32::from_be_bytes([r[p], r[p + 1], r[p + 2], r[p + 3]]));
            }
            drop(store);
            let cpl = self.common_prefix_lengths.clone();
            write_leaf(
                &c,
                self.version,
                out,
                &docs,
                &values,
                &first,
                cpl,
                sorted_dim,
                leaf_cardinality,
            );
            return Ok(());
        }
        let split_dim = if c.num_index_dims == 1 {
            0
        } else {
            if num_leaves != total_num_leaves
                && c.num_index_dims > 2
                && parent_splits.iter().sum::<i32>() % SPLITS_BEFORE_EXACT_BOUNDS == 0
            {
                self.compute_bounds_path(&points, &mut min_packed, &mut max_packed)?;
            }
            self.split(&min_packed, &max_packed, parent_splits)
        };
        let num_left = get_num_left_leaf_nodes(num_leaves);
        let left_count = (num_left * c.max_points_in_leaf_node) as u64;
        let o = split_dim * c.bytes_per_dim;
        let common_prefix_len = common_prefix(
            &min_packed[o..o + c.bytes_per_dim],
            &max_packed[o..o + c.bytes_per_dim],
        );
        let (split_value, left, right) = self.select(
            points.clone(),
            points.start,
            points.start + points.count,
            points.start + left_count,
            split_dim,
            common_prefix_len,
        )?;
        let right_offset = leaves_offset + num_left;
        let split_offset = right_offset - 1;
        split_dims[split_offset] = split_dim as u8;
        let address = split_offset * c.bytes_per_dim;
        split_values[address..address + c.bytes_per_dim].copy_from_slice(&split_value);
        let mut min_split = min_packed.clone();
        let mut max_split = max_packed.clone();
        min_split[o..o + c.bytes_per_dim].copy_from_slice(&split_value);
        max_split[o..o + c.bytes_per_dim].copy_from_slice(&split_value);
        parent_splits[split_dim] += 1;
        self.build_path(
            leaves_offset,
            num_left,
            left,
            out,
            min_packed,
            max_split,
            parent_splits,
            split_values,
            split_dims,
            leaf_fps,
            total_num_leaves,
        )?;
        self.build_path(
            right_offset,
            num_leaves - num_left,
            right,
            out,
            min_split,
            max_packed,
            parent_splits,
            split_values,
            split_dims,
            leaf_fps,
            total_num_leaves,
        )?;
        parent_splits[split_dim] -= 1;
        Ok(())
    }

    /// `computePackedValueBounds(PathSlice, ...)`.
    // ARITH: record offsets.
    #[allow(clippy::arithmetic_side_effects)]
    fn compute_bounds_path(&self, slice: &PathSlice, min: &mut [u8], max: &mut [u8]) -> Result<()> {
        let c = self.config;
        let records = self.slice_records(slice)?;
        let mut it = records.chunks(c.bytes_per_doc());
        let Some(first) = it.next() else {
            return Ok(());
        };
        let n = c.packed_index_bytes_length();
        min.copy_from_slice(&first[..n]);
        max.copy_from_slice(&first[..n]);
        for r in it {
            for dim in 0..c.num_index_dims {
                let o = dim * c.bytes_per_dim;
                let v = &r[o..o + c.bytes_per_dim];
                if v < &min[o..o + c.bytes_per_dim] {
                    min[o..o + c.bytes_per_dim].copy_from_slice(v);
                } else if v > &max[o..o + c.bytes_per_dim] {
                    max[o..o + c.bytes_per_dim].copy_from_slice(v);
                }
            }
        }
        Ok(())
    }

    /// A slice's records, in order.
    // ARITH: record offsets.
    #[allow(clippy::arithmetic_side_effects)]
    fn slice_records(&self, slice: &PathSlice) -> Result<Vec<u8>> {
        match &*slice.writer.borrow() {
            PointStore::Heap(h) => {
                let bpd = h.bpd;
                Ok(
                    h.block[slice.start as usize * bpd..(slice.start + slice.count) as usize * bpd]
                        .to_vec(),
                )
            }
            PointStore::Offline(o) => self.read_offline(&o.name, slice.start, slice.count),
        }
    }
}

/// `BKDWriter.getNumLeftLeafNodes`.
// ARITH: a leaf count above 1.
#[allow(clippy::arithmetic_side_effects)]
fn get_num_left_leaf_nodes(num_leaves: usize) -> usize {
    let last_full_level = usize::BITS - 1 - num_leaves.leading_zeros();
    let leaves_full_level = 1usize << last_full_level;
    let mut num_left = leaves_full_level / 2;
    let unbalanced = num_leaves - leaves_full_level;
    num_left += unbalanced.min(num_left);
    num_left
}

// --- BKDRadixSelector --------------------------------------------------------

/// `RadixTarget` over a heap store's records, keyed by
/// `split-dim suffix | data dims | doc id` from byte `prefix` on.
struct HeapKey<'a> {
    heap: &'a mut HeapPoints,
    dim_offset: usize,
    /// `bytesPerDim - commonPrefixLength`: negative once the common prefix
    /// runs past the split dimension into the data dims and doc id.
    dim_cmp_bytes: i64,
    /// `packedIndexBytesLength - dimCmpBytes`.
    data_offset: usize,
}

impl RadixTarget for HeapKey<'_> {
    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
    }
    // ARITH: offsets inside a record.
    #[allow(clippy::arithmetic_side_effects)]
    fn byte_at(&mut self, i: usize, k: usize) -> i32 {
        let pos = if (k as i64) < self.dim_cmp_bytes {
            self.dim_offset + k
        } else {
            self.data_offset + k
        };
        self.heap.byte_at(i, pos)
    }
}

/// The fallback `IntroSelector`/`IntroSorter` of `heapRadixSelect`/
/// `heapRadixSort`: whole split dimension (when bytes of it remain), then
/// data dims and doc id.
struct HeapCompare<'a> {
    heap: &'a mut HeapPoints,
    compare_dim: bool,
    dim_start: usize,
    bytes_per_dim: usize,
    index_len: usize,
    pivot: Vec<u8>,
}

impl HeapCompare<'_> {
    // ARITH: offsets inside a record.
    #[allow(clippy::arithmetic_side_effects)]
    fn key<'r>(&self, r: &'r [u8]) -> (&'r [u8], &'r [u8]) {
        (
            &r[self.dim_start..self.dim_start + self.bytes_per_dim],
            &r[self.index_len..],
        )
    }
}

impl IntroTarget for HeapCompare<'_> {
    fn swap(&mut self, i: usize, j: usize) {
        self.heap.swap(i, j);
    }
    fn set_pivot(&mut self, i: usize) {
        let r = self.heap.record(i);
        let (d, rest) = self.key(r);
        self.pivot.clear();
        self.pivot.extend_from_slice(d);
        self.pivot.extend_from_slice(rest);
    }
    // ARITH: offsets inside the pivot.
    #[allow(clippy::arithmetic_side_effects)]
    fn compare_pivot(&mut self, j: usize) -> i32 {
        let r = self.heap.record(j);
        let (d, rest) = self.key(r);
        if self.compare_dim {
            let c = self.pivot[..self.bytes_per_dim].cmp(d);
            if c != Ordering::Equal {
                return c as i32;
            }
        }
        self.pivot[self.bytes_per_dim..].cmp(rest) as i32
    }
    fn compare(&mut self, i: usize, j: usize) -> i32 {
        let (ri, rj) = (self.heap.record(i), self.heap.record(j));
        let (di, resti) = self.key(ri);
        let (dj, restj) = self.key(rj);
        if self.compare_dim {
            let c = di.cmp(dj);
            if c != Ordering::Equal {
                return c as i32;
            }
        }
        resti.cmp(restj) as i32
    }
}

impl<'d> BkdWriter<'d> {
    /// `bytesSorted`: split dim + data dims + doc id.
    // ARITH: at most 16 * 16 + 4.
    #[allow(clippy::arithmetic_side_effects)]
    fn bytes_sorted(&self) -> usize {
        let c = self.config;
        c.bytes_per_dim + (c.num_dims - c.num_index_dims) * c.bytes_per_dim + 4
    }

    /// `BKDRadixSelector.heapRadixSelect`: the split value.
    // ARITH: key offsets.
    #[allow(clippy::arithmetic_side_effects)]
    fn heap_radix_select(
        &mut self,
        heap: &mut HeapPoints,
        dim: usize,
        from: usize,
        to: usize,
        partition_point: usize,
        common_prefix_length: usize,
    ) -> Vec<u8> {
        let c = self.config;
        let dim_cmp_bytes = c.bytes_per_dim as i64 - common_prefix_length as i64;
        let max_len = self.bytes_sorted() - common_prefix_length;
        let mut random = std::mem::replace(&mut self.random, SplittableRandom::new(0));
        {
            let mut key = HeapKey {
                heap,
                dim_offset: dim * c.bytes_per_dim + common_prefix_length,
                dim_cmp_bytes,
                data_offset: (c.packed_index_bytes_length() as i64 - dim_cmp_bytes) as usize,
            };
            radix_select(
                &mut key,
                max_len,
                from,
                to,
                partition_point,
                &mut |k: &mut HeapKey<'_>, d, f, t, kk| {
                    let skipped = d + common_prefix_length;
                    let mut cmp = HeapCompare {
                        heap: k.heap,
                        compare_dim: skipped < c.bytes_per_dim,
                        dim_start: dim * c.bytes_per_dim,
                        bytes_per_dim: c.bytes_per_dim,
                        index_len: c.packed_index_bytes_length(),
                        pivot: Vec::new(),
                    };
                    intro_select(&mut cmp, f, t, kk, &mut random);
                },
            );
        }
        self.random = random;
        let r = heap.record(partition_point);
        r[dim * c.bytes_per_dim..(dim + 1) * c.bytes_per_dim].to_vec()
    }

    /// `BKDRadixSelector.heapRadixSort`.
    // ARITH: key offsets.
    #[allow(clippy::arithmetic_side_effects)]
    fn heap_radix_sort(
        &self,
        heap: &mut HeapPoints,
        from: usize,
        to: usize,
        dim: usize,
        common_prefix_length: usize,
    ) {
        let c = self.config;
        let dim_cmp_bytes = c.bytes_per_dim as i64 - common_prefix_length as i64;
        let max_len = self.bytes_sorted() - common_prefix_length;
        let mut key = HeapKey {
            heap,
            dim_offset: dim * c.bytes_per_dim + common_prefix_length,
            dim_cmp_bytes,
            data_offset: (c.packed_index_bytes_length() as i64 - dim_cmp_bytes) as usize,
        };
        msb_radix_sort(
            &mut key,
            max_len,
            from,
            to,
            &mut |k: &mut HeapKey<'_>, kk, f, t| {
                let skipped = kk + common_prefix_length;
                let mut cmp = HeapCompare {
                    heap: k.heap,
                    compare_dim: skipped < c.bytes_per_dim,
                    dim_start: dim * c.bytes_per_dim,
                    bytes_per_dim: c.bytes_per_dim,
                    index_len: c.packed_index_bytes_length(),
                    pivot: Vec::new(),
                };
                intro_sort(&mut cmp, f, t);
            },
        );
    }

    /// `BKDRadixSelector.select`: partitions `points` around
    /// `partition_point` by `dim`, returning the split value and the two
    /// halves.
    // ARITH: point counts of the slice.
    #[allow(clippy::arithmetic_side_effects)]
    fn select(
        &mut self,
        points: PathSlice,
        from: u64,
        to: u64,
        partition_point: u64,
        dim: usize,
        dim_common_prefix: usize,
    ) -> Result<(Vec<u8>, PathSlice, PathSlice)> {
        let is_heap = matches!(&*points.writer.borrow(), PointStore::Heap(_));
        if is_heap {
            let value = {
                let mut store = points.writer.borrow_mut();
                let PointStore::Heap(heap) = &mut *store else {
                    unreachable!("checked above");
                };
                self.heap_radix_select(
                    heap,
                    dim,
                    from as usize,
                    to as usize,
                    partition_point as usize,
                    dim_common_prefix,
                )
            };
            let left = PathSlice {
                writer: points.writer.clone(),
                start: from,
                count: partition_point - from,
            };
            let right = PathSlice {
                writer: points.writer,
                start: partition_point,
                count: to - partition_point,
            };
            return Ok((value, left, right));
        }
        let mut left = self.get_point_writer(partition_point - from, &format!("left{dim}"))?;
        let mut right = self.get_point_writer(to - partition_point, &format!("right{dim}"))?;
        let value = self.build_histogram_and_partition(
            points.writer,
            &mut left,
            &mut right,
            from,
            to,
            partition_point,
            0,
            dim_common_prefix,
            dim,
        )?;
        self.close_store(&mut left)?;
        self.close_store(&mut right)?;
        Ok((
            value,
            PathSlice {
                writer: shared(left),
                start: 0,
                count: partition_point - from,
            },
            PathSlice {
                writer: shared(right),
                start: 0,
                count: to - partition_point,
            },
        ))
    }

    /// `getBucket(offset, commonPrefixPosition, pointValue)` over a record.
    // ARITH: offsets inside a record.
    #[allow(clippy::arithmetic_side_effects)]
    fn bucket(&self, record: &[u8], offset: usize, cpp: usize) -> usize {
        let c = self.config;
        if cpp < c.bytes_per_dim {
            record[offset + cpp] as usize
        } else {
            record[c.packed_index_bytes_length() + cpp - c.bytes_per_dim] as usize
        }
    }

    /// `findCommonPrefixAndHistogram` over an offline store.
    // ARITH: key offsets and counts.
    #[allow(clippy::arithmetic_side_effects)]
    fn find_common_prefix_and_histogram(
        &self,
        records: &[u8],
        dim: usize,
        dim_common_prefix: usize,
        histogram: &mut [u64; 256],
        partition_bucket: &mut [usize],
    ) -> usize {
        let c = self.config;
        let bytes_sorted = self.bytes_sorted();
        let mut cpp = bytes_sorted;
        let offset = dim * c.bytes_per_dim;
        let bpd = c.bytes_per_doc();
        let mut recs = records.chunks(bpd);
        let first = recs.next().expect("a non-empty slice");
        let mut scratch = vec![0u8; bytes_sorted];
        scratch[..c.bytes_per_dim].copy_from_slice(&first[offset..offset + c.bytes_per_dim]);
        scratch[c.bytes_per_dim..].copy_from_slice(&first[c.packed_index_bytes_length()..bpd]);
        let n = records.len() / bpd;
        let mut i = 1usize;
        while i < n {
            let r = &records[i * bpd..(i + 1) * bpd];
            if cpp == dim_common_prefix {
                histogram[self.bucket(r, offset, cpp)] += 1;
                for j in i + 1..n {
                    let r2 = &records[j * bpd..(j + 1) * bpd];
                    histogram[self.bucket(r2, offset, cpp)] += 1;
                }
                break;
            }
            let start = dim_common_prefix.min(c.bytes_per_dim);
            let end = cpp.min(c.bytes_per_dim);
            let mismatch = scratch[start..end]
                .iter()
                .zip(&r[offset + start..offset + end])
                .position(|(a, b)| a != b);
            match mismatch {
                None => {
                    if cpp > c.bytes_per_dim {
                        let tie_start = c.packed_index_bytes_length();
                        let tie_end = tie_start + cpp - c.bytes_per_dim;
                        let k = scratch[c.bytes_per_dim..cpp]
                            .iter()
                            .zip(&r[tie_start..tie_end])
                            .position(|(a, b)| a != b);
                        if let Some(k) = k {
                            cpp = c.bytes_per_dim + k;
                            histogram.fill(0);
                            histogram[scratch[cpp] as usize] = i as u64;
                        }
                    }
                }
                Some(j) => {
                    cpp = dim_common_prefix + j;
                    histogram.fill(0);
                    histogram[scratch[cpp] as usize] = i as u64;
                }
            }
            if cpp != bytes_sorted {
                histogram[self.bucket(r, offset, cpp)] += 1;
            }
            i += 1;
        }
        for (b, s) in partition_bucket.iter_mut().zip(&scratch).take(cpp) {
            *b = *s as usize;
        }
        cpp
    }

    /// `buildHistogramAndPartition`.
    #[allow(clippy::too_many_arguments, clippy::arithmetic_side_effects)]
    // ARITH: point counts of the slice and key positions.
    fn build_histogram_and_partition(
        &mut self,
        points: SharedStore,
        left: &mut PointStore,
        right: &mut PointStore,
        from: u64,
        to: u64,
        partition_point: u64,
        iteration: usize,
        base_common_prefix: usize,
        dim: usize,
    ) -> Result<Vec<u8>> {
        let c = self.config;
        let bytes_sorted = self.bytes_sorted();
        let name = match &*points.borrow() {
            PointStore::Offline(o) => o.name.clone(),
            PointStore::Heap(_) => return Err(illegal("offline partition over a heap store")),
        };
        let records = self.read_offline(&name, from, to - from)?;
        let mut histogram = [0u64; 256];
        let mut partition_bucket = vec![0usize; bytes_sorted];
        let common_prefix = self.find_common_prefix_and_histogram(
            &records,
            dim,
            base_common_prefix,
            &mut histogram,
            &mut partition_bucket,
        );
        let offset = dim * c.bytes_per_dim;
        let bpd = c.bytes_per_doc();
        let partition_value =
            |pb: &[usize]| -> Vec<u8> { pb[..c.bytes_per_dim].iter().map(|&b| b as u8).collect() };
        if common_prefix == bytes_sorted {
            // Every key equal: the first `partition_point - from` go left.
            let pos = common_prefix - 1;
            self.offline_partition(
                &records,
                left,
                right,
                None,
                offset,
                pos,
                &partition_bucket,
                partition_point - from,
            );
            self.destroy_store(&points.borrow())?;
            return Ok(partition_value(&partition_bucket));
        }
        let mut left_count = 0u64;
        for (i, &size) in histogram.iter().enumerate() {
            if left_count + size > partition_point - from {
                partition_bucket[common_prefix] = i;
                break;
            }
            left_count += size;
        }
        let delta = histogram[partition_bucket[common_prefix]];
        if common_prefix == bytes_sorted - 1 {
            let tie = partition_point - from - left_count;
            self.offline_partition(
                &records,
                left,
                right,
                None,
                offset,
                common_prefix,
                &partition_bucket,
                tie,
            );
            self.destroy_store(&points.borrow())?;
            return Ok(partition_value(&partition_bucket));
        }
        // getDeltaPointWriter: heap if it fits beside left/right's heap use.
        let used: usize = [&*left, &*right]
            .iter()
            .map(|s| match s {
                PointStore::Heap(h) => h.size,
                PointStore::Offline(_) => 0,
            })
            .sum();
        let mut delta_points = if delta <= (self.max_points_sort_in_heap - used) as u64 {
            PointStore::Heap(HeapPoints::new(&c, delta as usize))
        } else {
            PointStore::Offline(self.create_temp(&format!("delta{iteration}"))?)
        };
        self.offline_partition(
            &records,
            left,
            right,
            Some(&mut delta_points),
            offset,
            common_prefix,
            &partition_bucket,
            0,
        );
        self.destroy_store(&points.borrow())?;
        self.close_store(&mut delta_points)?;
        let new_partition_point = partition_point - from - left_count;
        match delta_points {
            PointStore::Heap(mut heap) => {
                let count = heap.next_write;
                let value = self.heap_radix_select(
                    &mut heap,
                    dim,
                    0,
                    count,
                    new_partition_point as usize,
                    common_prefix + 1,
                );
                for i in 0..count {
                    let r = heap.record(i).to_vec();
                    if (i as u64) < new_partition_point {
                        Self::store_append(left, &r);
                    } else {
                        Self::store_append(right, &r);
                    }
                }
                let _ = bpd;
                Ok(value)
            }
            PointStore::Offline(o) => {
                let count = o.count;
                self.build_histogram_and_partition(
                    shared(PointStore::Offline(o)),
                    left,
                    right,
                    0,
                    count,
                    new_partition_point,
                    iteration + 1,
                    common_prefix + 1,
                    dim,
                )
            }
        }
    }

    /// `offlinePartition`: streams every record left, right, or (tied at
    /// `byte_position`) into `delta` -- or, at the last key byte, the first
    /// `tiebreak` ties left.
    #[allow(clippy::too_many_arguments, clippy::arithmetic_side_effects)]
    // ARITH: a tie-break counter.
    fn offline_partition(
        &self,
        records: &[u8],
        left: &mut PointStore,
        right: &mut PointStore,
        mut delta: Option<&mut PointStore>,
        offset: usize,
        byte_position: usize,
        partition_bucket: &[usize],
        tiebreak: u64,
    ) {
        let bytes_sorted = self.bytes_sorted();
        let mut tiebreak_counter = 0u64;
        for r in records.chunks(self.config.bytes_per_doc()) {
            let bucket = self.bucket(r, offset, byte_position);
            let pb = partition_bucket[byte_position];
            if bucket < pb {
                Self::store_append(left, r);
            } else if bucket > pb {
                Self::store_append(right, r);
            } else if byte_position == bytes_sorted - 1 {
                if tiebreak_counter < tiebreak {
                    Self::store_append(left, r);
                    tiebreak_counter += 1;
                } else {
                    Self::store_append(right, r);
                }
            } else if let Some(d) = delta.as_deref_mut() {
                Self::store_append(d, r);
            }
        }
    }
}

// --- MutablePointTreeReaderUtils ------------------------------------------

/// `partition`'s radix key: split-dim suffix, data dims, then the doc id's
/// top `bits_per_doc_id` bits big-endian.
struct TreeKey<'a> {
    tree: &'a mut MutablePointTree,
    dim_offset: usize,
    dim_cmp_bytes: usize,
    data_cmp_bytes: usize,
    index_len: usize,
    bits_per_doc_id: i32,
}

impl RadixTarget for TreeKey<'_> {
    fn swap(&mut self, i: usize, j: usize) {
        self.tree.swap(i, j);
    }
    // ARITH: key offsets; shifts below 32.
    #[allow(clippy::arithmetic_side_effects)]
    fn byte_at(&mut self, i: usize, k: usize) -> i32 {
        if k < self.dim_cmp_bytes {
            i32::from(self.tree.byte_at(i, self.dim_offset + k))
        } else if k < self.data_cmp_bytes {
            i32::from(
                self.tree
                    .byte_at(i, self.index_len + k - self.dim_cmp_bytes),
            )
        } else {
            let shift = self.bits_per_doc_id - (((k - self.data_cmp_bytes + 1) as i32) << 3);
            ((self.tree.docs[i] as u32 >> shift.max(0)) & 0xff) as i32
        }
    }
}

/// `partition`'s fallback `IntroSelector` for key byte `k`: the whole split
/// dimension (while `k` is inside it), the data dims from where `k` points,
/// then the doc id.
struct TreePartitionCompare<'a> {
    tree: &'a mut MutablePointTree,
    k: usize,
    dim_cmp_bytes: usize,
    data_cmp_bytes: usize,
    dim_start: usize,
    bytes_per_dim: usize,
    data_start: usize,
    data_end: usize,
    pivot: Vec<u8>,
    pivot_doc: i32,
}

impl IntroTarget for TreePartitionCompare<'_> {
    fn swap(&mut self, i: usize, j: usize) {
        self.tree.swap(i, j);
    }
    fn set_pivot(&mut self, i: usize) {
        self.pivot.clear();
        self.pivot.extend_from_slice(self.tree.value(i));
        self.pivot_doc = self.tree.docs[i];
    }
    // ARITH: offsets inside a value.
    #[allow(clippy::arithmetic_side_effects)]
    fn compare_pivot(&mut self, j: usize) -> i32 {
        let v = self.tree.value(j);
        if self.k < self.dim_cmp_bytes {
            let s = self.dim_start;
            let c = self.pivot[s..s + self.bytes_per_dim].cmp(&v[s..s + self.bytes_per_dim]);
            if c != Ordering::Equal {
                return c as i32;
            }
        }
        if self.k < self.data_cmp_bytes {
            let c =
                self.pivot[self.data_start..self.data_end].cmp(&v[self.data_start..self.data_end]);
            if c != Ordering::Equal {
                return c as i32;
            }
        }
        self.pivot_doc.wrapping_sub(self.tree.docs[j])
    }
}

/// `sortByDim`'s `IntroSorter`: the sorted dimension, then data dims, then
/// doc id.
struct TreeDimCompare<'a> {
    tree: &'a mut MutablePointTree,
    start: usize,
    bytes_per_dim: usize,
    index_len: usize,
    packed_len: usize,
    pivot: Vec<u8>,
    pivot_doc: i32,
}

impl IntroTarget for TreeDimCompare<'_> {
    fn swap(&mut self, i: usize, j: usize) {
        self.tree.swap(i, j);
    }
    fn set_pivot(&mut self, i: usize) {
        self.pivot.clear();
        self.pivot.extend_from_slice(self.tree.value(i));
        self.pivot_doc = self.tree.docs[i];
    }
    // ARITH: offsets inside a value.
    #[allow(clippy::arithmetic_side_effects)]
    fn compare_pivot(&mut self, j: usize) -> i32 {
        let v = self.tree.value(j);
        let s = self.start;
        let mut c = self.pivot[s..s + self.bytes_per_dim].cmp(&v[s..s + self.bytes_per_dim]);
        if c == Ordering::Equal {
            c = self.pivot[self.index_len..self.packed_len]
                .cmp(&v[self.index_len..self.packed_len]);
            if c == Ordering::Equal {
                return self.pivot_doc.wrapping_sub(self.tree.docs[j]);
            }
        }
        c as i32
    }
}

/// `PackedInts.bitsRequired(maxDoc - 1)`.
fn bits_required(v: i64) -> i32 {
    // leading_zeros is at most 64.
    (64u32.saturating_sub(v.max(0).leading_zeros()) as i32).max(1)
}

impl<'d> BkdWriter<'d> {
    /// `BKDWriter.writeField(metaOut, indexOut, dataOut, name, reader)`: the
    /// flush path. Reorders `values` and writes the leaves into `data`;
    /// `None` when there are no points.
    pub fn write_field(
        &mut self,
        data: &mut Vec<u8>,
        values: &mut MutablePointTree,
    ) -> Result<Option<BkdIndexPlan>> {
        if values.stride != self.config.packed_bytes_length() {
            return Err(illegal("packed value length does not match the config"));
        }
        if self.config.num_dims == 1 {
            self.write_field_1dim(data, values)
        } else {
            self.write_field_ndims(data, values)
        }
    }

    /// `writeField1Dim`: `MutablePointTreeReaderUtils.sort`, then the
    /// one-dimension writer. The sort's key is the whole packed value then the
    /// doc id, which is a total order on distinct points, so a stable
    /// comparison sort yields exactly Java's permutation.
    // ARITH: shifts build a key of at most 12 value bytes + 4 doc bytes = 128
    // bits.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_field_1dim(
        &mut self,
        data: &mut Vec<u8>,
        values: &mut MutablePointTree,
    ) -> Result<Option<BkdIndexPlan>> {
        let n = values.len();
        let mut one = OneDimWriter::new(self, data)?;
        let stride = values.stride;
        if stride <= 12 && values.docs.iter().all(|&d| d >= 0) {
            // Stage-3 fast path: the (value, doc) key packed into one `u128`
            // (value bytes big-endian on top, the non-negative doc below), so
            // the sort compares integers instead of slices. Points with equal
            // keys are identical, so an unstable sort writes the same bytes.
            let mut keyed: Vec<(u128, u32)> = (0..n)
                .map(|i| {
                    let mut k = 0u128;
                    for &b in values.value(i) {
                        k = (k << 8) | u128::from(b);
                    }
                    ((k << 32) | u128::from(values.docs[i] as u32), i as u32)
                })
                .collect();
            keyed.sort_unstable();
            for (_, i) in keyed {
                let i = i as usize;
                one.add(self, data, values.value(i), values.docs[i])?;
            }
        } else {
            let mut order: Vec<usize> = (0..n).collect();
            order.sort_by(|&a, &b| {
                values
                    .value(a)
                    .cmp(values.value(b))
                    .then(values.docs[a].cmp(&values.docs[b]))
            });
            for i in order {
                one.add(self, data, values.value(i), values.docs[i])?;
            }
        }
        one.finish(self, data)
    }

    /// `writeFieldNDims`.
    // ARITH: leaf counts from the point count.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_field_ndims(
        &mut self,
        data: &mut Vec<u8>,
        values: &mut MutablePointTree,
    ) -> Result<Option<BkdIndexPlan>> {
        if self.point_count != 0 {
            return Err(illegal("cannot mix add and writeField"));
        }
        if self.finished {
            return Err(illegal("already finished"));
        }
        self.finished = true;
        self.point_count = values.len() as u64;
        if self.point_count == 0 {
            return Ok(None);
        }
        let c = self.config;
        let num_leaves = values.len().div_ceil(c.max_points_in_leaf_node);
        let mut split_values = vec![0u8; (num_leaves - 1) * c.bytes_per_dim];
        let mut split_dims = vec![0u8; num_leaves - 1];
        let mut leaf_fps = vec![0i64; num_leaves];
        let (mut min, mut max) = (self.min_packed.clone(), self.max_packed.clone());
        compute_bounds_tree(&c, values, 0, values.len(), &mut min, &mut max);
        self.min_packed = min.clone();
        self.max_packed = max.clone();
        for &d in &values.docs {
            // FBS: `docsSeen` is `FixedBitSet(maxDoc)`; ids are the caller's.
            if (d as usize) < self.docs_seen.len() {
                self.docs_seen.set(d as usize);
            }
        }
        let data_start_fp = data.len() as i64;
        let mut parent_splits = vec![0i32; c.num_index_dims];
        let n = values.len();
        self.build_tree(
            0,
            num_leaves,
            values,
            0,
            n,
            data,
            min,
            max,
            &mut parent_splits,
            &mut split_values,
            &mut split_dims,
            &mut leaf_fps,
        )?;
        Ok(Some(BkdIndexPlan {
            leaf_fps,
            split_values,
            split_dims,
            data_start_fp,
        }))
    }

    /// `BKDWriter.build(... MutablePointTree ...)`.
    #[allow(clippy::too_many_arguments, clippy::arithmetic_side_effects)]
    // ARITH: leaf counts and offsets inside the tree being built.
    fn build_tree(
        &mut self,
        leaves_offset: usize,
        num_leaves: usize,
        values: &mut MutablePointTree,
        from: usize,
        to: usize,
        out: &mut Vec<u8>,
        mut min_packed: Vec<u8>,
        mut max_packed: Vec<u8>,
        parent_splits: &mut [i32],
        split_values: &mut [u8],
        split_dims: &mut [u8],
        leaf_fps: &mut [i64],
    ) -> Result<()> {
        let c = self.config;
        if num_leaves == 1 {
            let count = to - from;
            self.common_prefix_lengths.fill(c.bytes_per_dim);
            let first = values.value(from).to_vec();
            for i in from + 1..to {
                let v = values.value(i);
                for dim in 0..c.num_dims {
                    let o = dim * c.bytes_per_dim;
                    let p =
                        common_prefix(&first[o..o + c.bytes_per_dim], &v[o..o + c.bytes_per_dim]);
                    self.common_prefix_lengths[dim] = self.common_prefix_lengths[dim].min(p);
                }
            }
            // Java counts the byte after the prefix from `from + 1` on.
            let mut sorted_dim = 0;
            let mut sorted_dim_cardinality = usize::MAX;
            for dim in 0..c.num_dims {
                let prefix = self.common_prefix_lengths[dim];
                if prefix < c.bytes_per_dim {
                    let mut used = [false; 256];
                    for i in from + 1..to {
                        used[values.byte_at(i, dim * c.bytes_per_dim + prefix) as usize] = true;
                    }
                    let cardinality = used.iter().filter(|u| **u).count();
                    if cardinality < sorted_dim_cardinality {
                        sorted_dim = dim;
                        sorted_dim_cardinality = cardinality;
                    }
                }
            }
            {
                let mut cmp = TreeDimCompare {
                    tree: values,
                    start: sorted_dim * c.bytes_per_dim,
                    bytes_per_dim: c.bytes_per_dim,
                    index_len: c.packed_index_bytes_length(),
                    packed_len: c.packed_bytes_length(),
                    pivot: Vec::new(),
                    pivot_doc: -1,
                };
                intro_sort(&mut cmp, from, to);
            }
            let mut leaf_cardinality = 1;
            let mut comparator = values.value(from).to_vec();
            for i in from + 1..to {
                let v = values.value(i);
                for dim in 0..c.num_dims {
                    let s = dim * c.bytes_per_dim;
                    if v[s..s + c.bytes_per_dim] != comparator[s..s + c.bytes_per_dim] {
                        leaf_cardinality += 1;
                        comparator.copy_from_slice(v);
                        break;
                    }
                }
            }
            leaf_fps[leaves_offset] = out.len() as i64;
            let docs: Vec<i32> = values.docs[from..to].to_vec();
            let vals: Vec<u8> = values.values[from * values.stride..to * values.stride].to_vec();
            let first_sorted = values.value(from).to_vec();
            let _ = count;
            let cpl = self.common_prefix_lengths.clone();
            write_leaf(
                &c,
                self.version,
                out,
                &docs,
                &vals,
                &first_sorted,
                cpl,
                sorted_dim,
                leaf_cardinality,
            );
            return Ok(());
        }
        let split_dim = if c.num_index_dims == 1 {
            0
        } else {
            if num_leaves != leaf_fps.len()
                && c.num_index_dims > 2
                && parent_splits.iter().sum::<i32>() % SPLITS_BEFORE_EXACT_BOUNDS == 0
            {
                compute_bounds_tree(&c, values, from, to, &mut min_packed, &mut max_packed);
            }
            self.split(&min_packed, &max_packed, parent_splits)
        };
        let num_left = get_num_left_leaf_nodes(num_leaves);
        let mid = from + num_left * c.max_points_in_leaf_node;
        let o = split_dim * c.bytes_per_dim;
        let common_prefix_len = common_prefix(
            &min_packed[o..o + c.bytes_per_dim],
            &max_packed[o..o + c.bytes_per_dim],
        );
        self.partition_tree(values, split_dim, common_prefix_len, from, to, mid);
        let right_offset = leaves_offset + num_left;
        let split_offset = right_offset - 1;
        split_dims[split_offset] = split_dim as u8;
        let split_value = values.value(mid)[o..o + c.bytes_per_dim].to_vec();
        let address = split_offset * c.bytes_per_dim;
        split_values[address..address + c.bytes_per_dim].copy_from_slice(&split_value);
        let mut min_split = min_packed[..c.packed_index_bytes_length()].to_vec();
        let mut max_split = max_packed[..c.packed_index_bytes_length()].to_vec();
        min_split[o..o + c.bytes_per_dim].copy_from_slice(&split_value);
        max_split[o..o + c.bytes_per_dim].copy_from_slice(&split_value);
        parent_splits[split_dim] += 1;
        self.build_tree(
            leaves_offset,
            num_left,
            values,
            from,
            mid,
            out,
            min_packed,
            max_split,
            parent_splits,
            split_values,
            split_dims,
            leaf_fps,
        )?;
        self.build_tree(
            right_offset,
            num_leaves - num_left,
            values,
            mid,
            to,
            out,
            min_split,
            max_packed,
            parent_splits,
            split_values,
            split_dims,
            leaf_fps,
        )?;
        parent_splits[split_dim] -= 1;
        Ok(())
    }

    /// `MutablePointTreeReaderUtils.partition`.
    // ARITH: key offsets.
    #[allow(clippy::arithmetic_side_effects)]
    fn partition_tree(
        &mut self,
        values: &mut MutablePointTree,
        split_dim: usize,
        common_prefix_len: usize,
        from: usize,
        to: usize,
        mid: usize,
    ) {
        let c = self.config;
        let dim_offset = split_dim * c.bytes_per_dim + common_prefix_len;
        let dim_cmp_bytes = c.bytes_per_dim - common_prefix_len;
        let data_cmp_bytes = (c.num_dims - c.num_index_dims) * c.bytes_per_dim + dim_cmp_bytes;
        let max_doc = self.docs_seen.len() as i64;
        let bits_per_doc_id = bits_required(max_doc - 1);
        let max_len = data_cmp_bytes + ((bits_per_doc_id + 7) / 8) as usize;
        let mut random = std::mem::replace(&mut self.random, SplittableRandom::new(0));
        let mut key = TreeKey {
            tree: values,
            dim_offset,
            dim_cmp_bytes,
            data_cmp_bytes,
            index_len: c.packed_index_bytes_length(),
            bits_per_doc_id,
        };
        radix_select(
            &mut key,
            max_len,
            from,
            to,
            mid,
            &mut |t: &mut TreeKey<'_>, k, f, e, kk| {
                let data_start = if k < dim_cmp_bytes {
                    c.packed_index_bytes_length()
                } else {
                    c.packed_index_bytes_length() + k - dim_cmp_bytes
                };
                let mut cmp = TreePartitionCompare {
                    tree: t.tree,
                    k,
                    dim_cmp_bytes,
                    data_cmp_bytes,
                    dim_start: split_dim * c.bytes_per_dim,
                    bytes_per_dim: c.bytes_per_dim,
                    data_start,
                    data_end: c.num_dims * c.bytes_per_dim,
                    pivot: Vec::new(),
                    pivot_doc: 0,
                };
                intro_select(&mut cmp, f, e, kk, &mut random);
            },
        );
        self.random = random;
    }

    /// `BKDWriter.merge`: the one-dimensional merge. Each source yields its
    /// points in its tree's leaf order (sorted by value, then doc), with
    /// docs already mapped and deleted ones dropped.
    pub fn merge(
        &mut self,
        data: &mut Vec<u8>,
        sources: Vec<Box<dyn Iterator<Item = (Vec<u8>, i32)> + '_>>,
    ) -> Result<Option<BkdIndexPlan>> {
        let bpd = self.config.bytes_per_dim;
        let mut heads: Vec<(Vec<u8>, i32, usize)> = Vec::new();
        let mut sources = sources;
        for (i, s) in sources.iter_mut().enumerate() {
            if let Some((v, d)) = s.next() {
                heads.push((v, d, i));
            }
        }
        let mut one = OneDimWriter::new(self, data)?;
        loop {
            let best = heads
                .iter()
                .enumerate()
                .min_by(|(_, a), (_, b)| a.0[..bpd].cmp(&b.0[..bpd]).then(a.1.cmp(&b.1)))
                .map(|(i, _)| i);
            let Some(bi) = best else { break };
            let (v, d, src) = heads[bi].clone();
            one.add(self, data, &v, d)?;
            match sources[src].next() {
                Some((nv, nd)) => heads[bi] = (nv, nd, src),
                None => {
                    heads.remove(bi);
                }
            }
        }
        one.finish(self, data)
    }

    /// `writeIndex`: the field's `.kdm` entry (after the caller's field
    /// number) and its packed index in `.kdi`.
    // ARITH: lengths of in-memory buffers.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn write_index(&self, meta: &mut Vec<u8>, index: &mut Vec<u8>, plan: &BkdIndexPlan) {
        let c = self.config;
        let packed_index = pack_index(&c, plan);
        codec_util::write_header(meta, CODEC_NAME, self.version);
        write_vint(meta, c.num_dims as i32);
        write_vint(meta, c.num_index_dims as i32);
        write_vint(meta, c.max_points_in_leaf_node as i32);
        write_vint(meta, c.bytes_per_dim as i32);
        write_vint(meta, plan.leaf_fps.len() as i32);
        meta.extend_from_slice(&self.min_packed);
        meta.extend_from_slice(&self.max_packed);
        write_vlong(meta, self.point_count as i64);
        write_vint(meta, self.docs_seen.cardinality() as i32);
        write_vint(meta, packed_index.len() as i32);
        meta.write_i64(plan.data_start_fp);
        meta.write_i64(index.len() as i64);
        index.extend_from_slice(&packed_index);
    }
}

/// `computePackedValueBounds(MutablePointTree, from, to, ...)`.
// ARITH: dimension offsets.
#[allow(clippy::arithmetic_side_effects)]
fn compute_bounds_tree(
    c: &BkdConfig,
    values: &MutablePointTree,
    from: usize,
    to: usize,
    min: &mut [u8],
    max: &mut [u8],
) {
    if from == to {
        return;
    }
    let n = c.packed_index_bytes_length();
    min.copy_from_slice(&values.value(from)[..n]);
    max.copy_from_slice(&values.value(from)[..n]);
    for i in from + 1..to {
        let v = values.value(i);
        for dim in 0..c.num_index_dims {
            let o = dim * c.bytes_per_dim;
            let s = &v[o..o + c.bytes_per_dim];
            if s < &min[o..o + c.bytes_per_dim] {
                min[o..o + c.bytes_per_dim].copy_from_slice(s);
            } else if s > &max[o..o + c.bytes_per_dim] {
                max[o..o + c.bytes_per_dim].copy_from_slice(s);
            }
        }
    }
}

/// `OneDimensionBKDWriter`: consumes points in sorted order, cutting a leaf
/// every `maxPointsInLeafNode`.
struct OneDimWriter {
    data_start_fp: i64,
    leaf_fps: Vec<i64>,
    leaf_start_values: Vec<u8>,
    leaf_values: Vec<u8>,
    leaf_docs: Vec<i32>,
    value_count: u64,
    leaf_cardinality: usize,
}

impl OneDimWriter {
    fn new(w: &mut BkdWriter<'_>, data: &[u8]) -> Result<Self> {
        if w.config.num_index_dims != 1 {
            return Err(illegal(format!(
                "config.numIndexDims() must be 1 but got {}",
                w.config.num_index_dims
            )));
        }
        if w.point_count != 0 {
            return Err(illegal("cannot mix add and merge"));
        }
        if w.finished {
            return Err(illegal("already finished"));
        }
        w.finished = true;
        Ok(OneDimWriter {
            data_start_fp: data.len() as i64,
            leaf_fps: Vec::new(),
            leaf_start_values: Vec::new(),
            leaf_values: Vec::new(),
            leaf_docs: Vec::new(),
            value_count: 0,
            leaf_cardinality: 0,
        })
    }

    // ARITH: leaf offsets and counts.
    #[allow(clippy::arithmetic_side_effects)]
    fn add(
        &mut self,
        w: &mut BkdWriter<'_>,
        data: &mut Vec<u8>,
        value: &[u8],
        doc: i32,
    ) -> Result<()> {
        let c = w.config;
        let leaf_count = self.leaf_docs.len();
        // Java offsets the previous value by `bytesPerDim` per point (the
        // same as `packedBytesLength` whenever this writer is reachable,
        // i.e. `numDims == 1`).
        if leaf_count == 0 || {
            let o = (leaf_count - 1) * c.bytes_per_dim;
            self.leaf_values[o..o + c.bytes_per_dim] != value[..c.bytes_per_dim]
        } {
            self.leaf_cardinality += 1;
        }
        self.leaf_values.extend_from_slice(value);
        self.leaf_docs.push(doc);
        // FBS: `docsSeen` is `FixedBitSet(maxDoc)`; ids are the caller's.
        if (doc as usize) < w.docs_seen.len() {
            w.docs_seen.set(doc as usize);
        }
        if self.value_count + self.leaf_docs.len() as u64 > w.total_point_count {
            return Err(illegal(format!(
                "totalPointCount={} was passed when we were created, but we just hit {} values",
                w.total_point_count,
                self.value_count + self.leaf_docs.len() as u64
            )));
        }
        if self.leaf_docs.len() == c.max_points_in_leaf_node {
            self.write_leaf_block(w, data);
        }
        Ok(())
    }

    // ARITH: leaf offsets.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_leaf_block(&mut self, w: &mut BkdWriter<'_>, data: &mut Vec<u8>) {
        let c = w.config;
        let p = c.packed_bytes_length();
        let n = c.packed_index_bytes_length();
        let count = self.leaf_docs.len();
        if self.value_count == 0 {
            w.min_packed.copy_from_slice(&self.leaf_values[..n]);
        }
        w.max_packed
            .copy_from_slice(&self.leaf_values[(count - 1) * p..(count - 1) * p + n]);
        self.value_count += count as u64;
        if !self.leaf_fps.is_empty() {
            self.leaf_start_values
                .extend_from_slice(&self.leaf_values[..n]);
        }
        self.leaf_fps.push(data.len() as i64);
        let mut cpl = vec![0usize; c.num_dims];
        cpl[0] = common_prefix(
            &self.leaf_values[..c.bytes_per_dim],
            &self.leaf_values[(count - 1) * p..(count - 1) * p + c.bytes_per_dim],
        );
        let first = self.leaf_values[..p].to_vec();
        write_leaf(
            &c,
            w.version,
            data,
            &self.leaf_docs,
            &self.leaf_values,
            &first,
            cpl,
            0,
            self.leaf_cardinality,
        );
        self.leaf_values.clear();
        self.leaf_docs.clear();
        self.leaf_cardinality = 0;
    }

    fn finish(mut self, w: &mut BkdWriter<'_>, data: &mut Vec<u8>) -> Result<Option<BkdIndexPlan>> {
        if !self.leaf_docs.is_empty() {
            self.write_leaf_block(w, data);
        }
        if self.value_count == 0 {
            return Ok(None);
        }
        w.point_count = self.value_count;
        Ok(Some(BkdIndexPlan {
            leaf_fps: self.leaf_fps,
            split_values: self.leaf_start_values,
            split_dims: Vec::new(),
            data_start_fp: self.data_start_fp,
        }))
    }
}

// --- leaves ------------------------------------------------------------------

/// A leaf: `writeLeafBlockDocs`, `writeCommonPrefixes` (from `first`), then
/// `writeLeafBlockPackedValues`. `values` holds the leaf's points in order.
#[allow(clippy::too_many_arguments, clippy::arithmetic_side_effects)]
// ARITH: offsets inside one leaf's values.
fn write_leaf(
    c: &BkdConfig,
    version: i32,
    out: &mut Vec<u8>,
    docs: &[i32],
    values: &[u8],
    first: &[u8],
    mut cpl: Vec<usize>,
    sorted_dim: usize,
    leaf_cardinality: usize,
) {
    let count = docs.len();
    let p = c.packed_bytes_length();
    let value = |i: usize| &values[i * p..(i + 1) * p];
    write_vint(out, count as i32);
    write_doc_ids(docs, version, out);
    for dim in 0..c.num_dims {
        write_vint(out, cpl[dim] as i32);
        let o = dim * c.bytes_per_dim;
        out.extend_from_slice(&first[o..o + cpl[dim]]);
    }
    let prefix_len_sum: usize = cpl.iter().sum();
    if prefix_len_sum == p {
        out.push(0xff);
        return;
    }
    let compressed_byte_offset = sorted_dim * c.bytes_per_dim + cpl[sorted_dim];
    let run_len = |start: usize, end: usize| -> usize {
        let b = value(start)[compressed_byte_offset];
        for i in start + 1..end {
            if value(i)[compressed_byte_offset] != b {
                return i - start;
            }
        }
        end - start
    };
    let (high_cost, low_cost) = if count == leaf_cardinality {
        (0usize, 1usize)
    } else {
        let mut num_run_lens = 0;
        let mut i = 0;
        while i < count {
            i += run_len(i, (i + 0xff).min(count));
            num_run_lens += 1;
        }
        (
            count * (p - prefix_len_sum - 1) + 2 * num_run_lens,
            leaf_cardinality * (p - prefix_len_sum + 1),
        )
    };
    if low_cost <= high_cost {
        out.push(0xfe);
        if c.num_index_dims != 1 {
            write_actual_bounds(c, out, &cpl, count, &value);
        }
        let mut scratch = value(0).to_vec();
        let mut cardinality = 1;
        let write_run = |out: &mut Vec<u8>, scratch: &[u8], cardinality: i32| {
            write_vint(out, cardinality);
            for j in 0..c.num_dims {
                let o = j * c.bytes_per_dim;
                out.extend_from_slice(&scratch[o + cpl[j]..o + c.bytes_per_dim]);
            }
        };
        for i in 1..count {
            let v = value(i);
            for dim in 0..c.num_dims {
                let s = dim * c.bytes_per_dim;
                if v[s..s + c.bytes_per_dim] != scratch[s..s + c.bytes_per_dim] {
                    write_run(out, &scratch, cardinality);
                    scratch.copy_from_slice(v);
                    cardinality = 1;
                    break;
                } else if dim == c.num_dims - 1 {
                    cardinality += 1;
                }
            }
        }
        write_run(out, &scratch, cardinality);
    } else {
        out.push(sorted_dim as u8);
        if c.num_index_dims != 1 {
            write_actual_bounds(c, out, &cpl, count, &value);
        }
        cpl[sorted_dim] += 1;
        let mut i = 0;
        while i < count {
            let len = run_len(i, (i + 0xff).min(count));
            out.push(value(i)[compressed_byte_offset]);
            out.push(len as u8);
            for k in i..i + len {
                let v = value(k);
                for dim in 0..c.num_dims {
                    let o = dim * c.bytes_per_dim;
                    out.extend_from_slice(&v[o + cpl[dim]..o + c.bytes_per_dim]);
                }
            }
            i += len;
        }
    }
}

/// `writeActualBounds`: per index dimension with a suffix, the leaf's min
/// and max suffix.
// ARITH: offsets inside one leaf.
#[allow(clippy::arithmetic_side_effects)]
fn write_actual_bounds<'v>(
    c: &BkdConfig,
    out: &mut Vec<u8>,
    cpl: &[usize],
    count: usize,
    value: &dyn Fn(usize) -> &'v [u8],
) {
    for (dim, &prefix) in cpl.iter().enumerate().take(c.num_index_dims) {
        let suffix = c.bytes_per_dim - prefix;
        if suffix > 0 {
            let o = dim * c.bytes_per_dim + prefix;
            let mut min = &value(0)[o..o + suffix];
            let mut max = min;
            for i in 1..count {
                let cand = &value(i)[o..o + suffix];
                if min > cand {
                    min = cand;
                } else if max < cand {
                    max = cand;
                }
            }
            out.extend_from_slice(min);
            out.extend_from_slice(max);
        }
    }
}

// --- packed index ---------------------------------------------------------------

/// `packIndex`.
// ARITH: at most 8 * 16 bytes.
#[allow(clippy::arithmetic_side_effects)]
fn pack_index(c: &BkdConfig, plan: &BkdIndexPlan) -> Vec<u8> {
    let mut blocks: Vec<Vec<u8>> = Vec::new();
    let mut last_split_values = vec![0u8; c.bytes_per_dim * c.num_index_dims];
    let mut negative_deltas = vec![false; c.num_index_dims];
    recurse_pack_index(
        c,
        plan,
        0,
        &mut blocks,
        &mut last_split_values,
        &mut negative_deltas,
        false,
        0,
        plan.leaf_fps.len(),
    );
    blocks.concat()
}

/// `recursePackIndex`; returns the bytes it contributed.
#[allow(clippy::too_many_arguments, clippy::arithmetic_side_effects)]
// ARITH: file-pointer deltas (leaves are written in order) and split-value
// byte arithmetic, all Java's own.
fn recurse_pack_index(
    c: &BkdConfig,
    plan: &BkdIndexPlan,
    min_block_fp: i64,
    blocks: &mut Vec<Vec<u8>>,
    last_split_values: &mut [u8],
    negative_deltas: &mut [bool],
    is_left: bool,
    leaves_offset: usize,
    num_leaves: usize,
) -> usize {
    let mut buf = Vec::new();
    if num_leaves == 1 {
        if is_left {
            return 0;
        }
        write_vlong(&mut buf, plan.leaf_fps[leaves_offset] - min_block_fp);
        let n = buf.len();
        blocks.push(buf);
        return n;
    }
    let left_block_fp = if is_left {
        min_block_fp
    } else {
        let fp = plan.leaf_fps[leaves_offset];
        write_vlong(&mut buf, fp - min_block_fp);
        fp
    };
    let num_left = get_num_left_leaf_nodes(num_leaves);
    let right_offset = leaves_offset + num_left;
    let split_offset = right_offset - 1;
    let split_dim = if plan.split_dims.is_empty() {
        0
    } else {
        plan.split_dims[split_offset] as usize
    };
    let bpd = c.bytes_per_dim;
    let sv = &plan.split_values[split_offset * bpd..(split_offset + 1) * bpd];
    let last = &last_split_values[split_dim * bpd..(split_dim + 1) * bpd];
    let prefix = common_prefix(sv, last);
    let first_diff_byte_delta = if prefix < bpd {
        let mut d = i32::from(sv[prefix]) - i32::from(last[prefix]);
        if negative_deltas[split_dim] {
            d = -d;
        }
        d
    } else {
        0
    };
    let code = (first_diff_byte_delta * (1 + bpd as i32) + prefix as i32) * c.num_index_dims as i32
        + split_dim as i32;
    write_vint(&mut buf, code);
    let suffix = bpd - prefix;
    if suffix > 1 {
        buf.extend_from_slice(&sv[prefix + 1..]);
    }
    let sav_split_value: Vec<u8> =
        last_split_values[split_dim * bpd + prefix..(split_dim + 1) * bpd].to_vec();
    last_split_values[split_dim * bpd + prefix..(split_dim + 1) * bpd]
        .copy_from_slice(&sv[prefix..]);
    let num_bytes = buf.len();
    blocks.push(buf);
    let idx_sav = blocks.len();
    blocks.push(Vec::new());
    let sav_negative = negative_deltas[split_dim];
    negative_deltas[split_dim] = true;
    let left_num_bytes = recurse_pack_index(
        c,
        plan,
        left_block_fp,
        blocks,
        last_split_values,
        negative_deltas,
        true,
        leaves_offset,
        num_left,
    );
    let mut bytes2 = Vec::new();
    if num_left != 1 {
        write_vint(&mut bytes2, left_num_bytes as i32);
    }
    let bytes2_len = bytes2.len();
    blocks[idx_sav] = bytes2;
    negative_deltas[split_dim] = false;
    let right_num_bytes = recurse_pack_index(
        c,
        plan,
        left_block_fp,
        blocks,
        last_split_values,
        negative_deltas,
        false,
        right_offset,
        num_leaves - num_left,
    );
    negative_deltas[split_dim] = sav_negative;
    last_split_values[split_dim * bpd + prefix..(split_dim + 1) * bpd]
        .copy_from_slice(&sav_split_value);
    num_bytes + bytes2_len + left_num_bytes + right_num_bytes
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;
    use lucene_store::data_input::SliceInput;

    fn round_trip(ids: &[i32], version: i32) -> i8 {
        let mut out = Vec::new();
        write_doc_ids(ids, version, &mut out);
        let mut input = SliceInput::new(&out);
        let back = crate::points::read_doc_ids(&mut input, ids.len()).unwrap();
        assert_eq!(back, ids, "version {version}, marker {}", out[0]);
        out[0] as i8
    }

    /// Writes at `VERSION_META_FILE` and decodes as Java's
    /// `DocIdsWriter.readScalarInts24`.
    fn scalar24_round_trip(ids: &[i32]) -> Vec<i32> {
        let mut out = Vec::new();
        write_doc_ids(ids, VERSION_META_FILE, &mut out);
        assert_eq!(out[0] as i8, BPV_24);
        let mut p = 1;
        let long = |p: &mut usize| {
            let v = i64::from_le_bytes(out[*p..*p + 8].try_into().unwrap());
            *p += 8;
            v
        };
        let mut back = Vec::new();
        let mut i = 0;
        while i + 7 < ids.len() {
            let (l1, l2, l3) = (long(&mut p), long(&mut p), long(&mut p));
            let u = |x: i64, s: u32| ((x as u64) >> s) as i64;
            back.push(u(l1, 40) as i32);
            back.push((u(l1, 16) & 0xffffff) as i32);
            back.push((((l1 & 0xffff) << 8) | u(l2, 56)) as i32);
            back.push((u(l2, 32) & 0xffffff) as i32);
            back.push((u(l2, 8) & 0xffffff) as i32);
            back.push((((l2 & 0xff) << 16) | u(l3, 48)) as i32);
            back.push((u(l3, 24) & 0xffffff) as i32);
            back.push((l3 & 0xffffff) as i32);
            i += 8;
        }
        while i < ids.len() {
            let s = i32::from(u16::from_le_bytes([out[p], out[p + 1]]));
            back.push((s << 8) | i32::from(out[p + 2]));
            p += 3;
            i += 1;
        }
        assert_eq!(p, out.len());
        back
    }

    #[test]
    fn doc_ids_every_encoding_round_trips() {
        let run: Vec<i32> = (100..400).collect();
        assert_eq!(round_trip(&run, VERSION_CURRENT), CONTINUOUS_IDS);
        let dense: Vec<i32> = (0..300).map(|i| 70 + i * 3).collect();
        assert_eq!(round_trip(&dense, VERSION_CURRENT), BITSET_IDS);
        let unsorted16: Vec<i32> = (0..301).map(|i| (i * 7919) % 60_000 + 5).collect();
        assert_eq!(round_trip(&unsorted16, VERSION_CURRENT), DELTA_BPV_16);
        let b21: Vec<i32> = (0..517).map(|i| (i * 104_729) % 0x1F_FFFF).collect();
        assert_eq!(round_trip(&b21, VERSION_CURRENT), BPV_21);
        let b24: Vec<i32> = (0..515).map(|i| (i * 1_299_709) % 0xFF_FFFF).collect();
        assert_eq!(round_trip(&b24, VERSION_CURRENT), BPV_24);
        // Before BPV_21 existed, 21-bit ids take the scalar 24-bit layout
        // (which the points reader, accepting BKD version 10 only, never sees).
        assert_eq!(scalar24_round_trip(&b21), b21);
        assert_eq!(scalar24_round_trip(&b24), b24);
        let b32: Vec<i32> = (0..77i64)
            .map(|i| (i * 40_000_000 % i64::from(i32::MAX)) as i32)
            .collect();
        assert_eq!(round_trip(&b32, VERSION_CURRENT), BPV_32);
        // A single id, and duplicates.
        assert_eq!(round_trip(&[5], VERSION_CURRENT), CONTINUOUS_IDS);
        assert_eq!(round_trip(&[9, 9, 9], VERSION_CURRENT), DELTA_BPV_16);
    }

    #[test]
    fn config_rejects_what_java_rejects() {
        assert!(BkdConfig::new(0, 1, 4, 512).is_err());
        assert!(BkdConfig::new(17, 1, 4, 512).is_err());
        assert!(BkdConfig::new(2, 0, 4, 512).is_err());
        assert!(BkdConfig::new(10, 9, 4, 512).is_err());
        assert!(BkdConfig::new(2, 3, 4, 512).is_err());
        assert!(BkdConfig::new(2, 2, 0, 512).is_err());
        assert!(BkdConfig::new(2, 2, 17, 512).is_err());
        assert!(BkdConfig::new(2, 2, 4, 0).is_err());
        let c = BkdConfig::new(3, 2, 4, 512).unwrap();
        assert_eq!(
            (
                c.packed_bytes_length(),
                c.packed_index_bytes_length(),
                c.bytes_per_doc()
            ),
            (12, 8, 16)
        );
    }

    #[test]
    fn writer_rejects_bad_calls() {
        let c = BkdConfig::new(1, 1, 4, 16).unwrap();
        assert!(BkdWriter::new(10, None, "_0", c, 16.0, 10, 3).is_err());
        assert!(BkdWriter::new(10, None, "_0", c, 16.0, 10, VERSION_CURRENT + 1).is_err());
        assert!(BkdWriter::new(10, None, "_0", c, -1.0, 10, VERSION_CURRENT).is_err());
        assert!(BkdWriter::new(10, None, "_0", c, f64::NAN, 10, VERSION_CURRENT).is_err());
        assert!(BkdWriter::new(10, None, "_0", c, 0.0, 10, VERSION_CURRENT).is_err());

        let mut w = BkdWriter::new(10, None, "_0", c, 16.0, 2, VERSION_CURRENT).unwrap();
        let mut data = Vec::new();
        // Nothing added: no tree.
        assert!(w.finish(&mut data).unwrap().is_none());
        assert!(w.add(&[0, 0, 0], 1).is_err());
        w.add(&[0, 0, 0, 1], 1).unwrap();
        w.add(&[0, 0, 0, 2], 2).unwrap();
        assert!(w.add(&[0, 0, 0, 3], 3).is_err(), "past totalPointCount");
        assert_eq!(w.point_count(), 2);
        assert!(w.finish(&mut data).unwrap().is_some());
        assert!(w.finish(&mut data).is_err(), "finished twice");

        let mut w = BkdWriter::new(10, None, "_0", c, 16.0, 2, VERSION_CURRENT).unwrap();
        let mut wrong = MutablePointTree::new(8);
        assert!(w.write_field(&mut data, &mut wrong).is_err());
        let mut empty = MutablePointTree::new(4);
        assert!(empty.is_empty());
        assert!(w.write_field(&mut data, &mut empty).unwrap().is_none());
    }

    #[test]
    fn spilling_needs_a_temp_dir() {
        let c = BkdConfig::new(1, 1, 4, 16).unwrap();
        // 0.0001 MB holds 13 points of 8 bytes: 100 points must spill.
        let r = BkdWriter::new(100, None, "_0", c, 0.0001, 100, VERSION_CURRENT);
        // maxPointsSortInHeap (13) < maxPointsInLeafNode (16) is itself refused.
        assert!(r.is_err());
        let mut w = BkdWriter::new(100, None, "_0", c, 0.0002, 100, VERSION_CURRENT).unwrap();
        assert!(w.add(&[0, 0, 0, 1], 1).is_err());
    }
}
