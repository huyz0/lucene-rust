//! Port of `org.apache.lucene.util.packed.PackedInts` and the in-memory half
//! of the packed-ints family: the random-access arrays that hold `n` values of
//! `bitsPerValue` bits each in a `long[]`, and the builders over them.
//!
//! | Java | Rust |
//! |---|---|
//! | `PackedInts` (helpers, `Format`, `fastestFormatAndBits`, `copy`) | this module |
//! | `PackedInts.Reader` / `PackedInts.Mutable` | [`Reader`] / [`Mutable`] |
//! | `PackedInts.NullReader` | [`NullReader`] |
//! | `PackedInts.getMutable` | [`get_mutable`] -> [`PackedMutable`] |
//! | `BulkOperation`, `BulkOperationPacked*`, `BulkOperationPackedSingleBlock` | [`bulk_operation`] |
//! | `Packed64` | [`packed64::Packed64`] |
//! | `Packed64SingleBlock` (+ its 14 per-width subclasses) | [`packed64_single_block::Packed64SingleBlock`] |
//! | `GrowableWriter` | [`growable_writer::GrowableWriter`] |
//! | `AbstractPagedMutable`, `PagedMutable`, `PagedGrowableWriter` | [`paged_mutable`] |
//! | `PackedLongValues`, `DeltaPackedLongValues`, `MonotonicLongValues` | [`packed_long_values`] |
//!
//! The serialized half (`PackedWriter`, `PackedReaderIterator`,
//! `PackedDataInput`/`Output`, `MonotonicBlockPacked*`, `DirectPacked64SingleBlockReader`)
//! reads and writes through `DataInput`/`DataOutput` and so lives in
//! `lucene-codecs` (`packed_ints.rs`, `packed_data.rs`, `monotonic_block_packed.rs`),
//! built on the bulk operations exported here.
//!
//! Rust-only changes, each forced by the language:
//!
//! - Java's class hierarchy (`Reader` <- `Mutable` <- `MutableImpl` <- `Packed64`)
//!   becomes two traits and a closed enum ([`PackedMutable`]) for what
//!   `getMutable` can return, so a page is not a trait object.
//! - Indices are `usize` (Java `int`); values are `i64` (Java `long`), and the
//!   backing words are `u64` so `>>>` is a plain `>>`.
//! - `ramBytesUsed` estimates a JVM object layout; the Rust types report the
//!   heap bytes they own instead ([`Mutable::ram_bytes_used`]).
//! - Java's `assert`s on caller contracts (index in range, value fits the
//!   width) become Rust's own slice bounds checks or `debug_assert!`s.

pub mod bulk_operation;
pub mod growable_writer;
pub mod packed64;
pub mod packed64_single_block;
pub mod packed_long_values;
pub mod paged_mutable;

pub use bulk_operation::BulkOperation;
pub use growable_writer::GrowableWriter;
pub use packed64::Packed64;
pub use packed64_single_block::Packed64SingleBlock;
pub use packed_long_values::{PackedLongValues, PackedLongValuesBuilder};
pub use paged_mutable::{PagedGrowableWriter, PagedMutable};

/// `PackedInts.FASTEST`: at most 700% memory overhead, always select a direct
/// implementation.
pub const FASTEST: f32 = 7.0;
/// `PackedInts.FAST`: at most 50% memory overhead.
pub const FAST: f32 = 0.5;
/// `PackedInts.DEFAULT`: at most 25% memory overhead.
pub const DEFAULT: f32 = 0.25;
/// `PackedInts.COMPACT`: no memory overhead at all.
pub const COMPACT: f32 = 0.0;
/// `PackedInts.DEFAULT_BUFFER_SIZE`: default amount of memory for bulk copies.
pub const DEFAULT_BUFFER_SIZE: usize = 1024;
/// `PackedInts.CODEC_NAME`.
pub const CODEC_NAME: &str = "PackedInts";
/// `PackedInts.VERSION_MONOTONIC_WITHOUT_ZIGZAG`.
pub const VERSION_MONOTONIC_WITHOUT_ZIGZAG: i32 = 2;
/// `PackedInts.VERSION_START`.
pub const VERSION_START: i32 = VERSION_MONOTONIC_WITHOUT_ZIGZAG;
/// `PackedInts.VERSION_CURRENT`.
pub const VERSION_CURRENT: i32 = VERSION_MONOTONIC_WITHOUT_ZIGZAG;

/// Errors of the packed-ints API: Java's `IllegalArgumentException`s.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum PackedError {
    #[error("{0}")]
    IllegalArgument(String),
}

/// Result alias for this module.
pub type Result<T> = std::result::Result<T, PackedError>;

fn illegal<T>(msg: impl Into<String>) -> Result<T> {
    Err(PackedError::IllegalArgument(msg.into()))
}

/// `PackedInts.checkVersion`.
pub fn check_version(version: i32) -> Result<()> {
    if version < VERSION_START {
        illegal(format!(
            "Version is too old, should be at least {VERSION_START} (got {version})"
        ))
    } else if version > VERSION_CURRENT {
        illegal(format!(
            "Version is too new, should be at most {VERSION_CURRENT} (got {version})"
        ))
    } else {
        Ok(())
    }
}

/// `PackedInts.Format`: how values are laid out in the backing blocks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Format {
    /// All bits written contiguously, most significant bit first.
    Packed,
    /// Each 64-bit block holds `64 / bitsPerValue` values, least significant
    /// first, with the remaining high bits wasted. Deprecated in Java, still
    /// readable and writable.
    PackedSingleBlock,
}

impl Format {
    /// `Format.getId`.
    pub fn id(self) -> i32 {
        match self {
            Format::Packed => 0,
            Format::PackedSingleBlock => 1,
        }
    }

    /// `Format.byId`.
    pub fn by_id(id: i32) -> Result<Format> {
        match id {
            0 => Ok(Format::Packed),
            1 => Ok(Format::PackedSingleBlock),
            _ => illegal(format!("Unknown format id: {id}")),
        }
    }

    /// `Format.byteCount`: bytes needed to store `value_count` values.
    pub fn byte_count(
        self,
        _packed_ints_version: i32,
        value_count: usize,
        bits_per_value: u32,
    ) -> u64 {
        debug_assert!(bits_per_value <= 64);
        match self {
            // Java: (long) Math.ceil((double) valueCount * bitsPerValue / 8).
            // The integer form is exact where the double one is, and for all
            // `int` counts the double one is exact too (< 2^53).
            Format::Packed => (value_count as u64)
                .saturating_mul(bits_per_value as u64)
                .div_ceil(8),
            Format::PackedSingleBlock => {
                (self.long_count(_packed_ints_version, value_count, bits_per_value) as u64)
                    .saturating_mul(8)
            }
        }
    }

    /// `Format.longCount`: 64-bit blocks needed to store `value_count` values.
    pub fn long_count(
        self,
        packed_ints_version: i32,
        value_count: usize,
        bits_per_value: u32,
    ) -> usize {
        match self {
            Format::Packed => {
                let byte_count = self.byte_count(packed_ints_version, value_count, bits_per_value);
                byte_count.div_ceil(8) as usize
            }
            Format::PackedSingleBlock => {
                let values_per_block = 64 / bits_per_value as usize;
                value_count.div_ceil(values_per_block)
            }
        }
    }

    /// `Format.isSupported`.
    pub fn is_supported(self, bits_per_value: u32) -> bool {
        match self {
            Format::Packed => (1..=64).contains(&bits_per_value),
            Format::PackedSingleBlock => packed64_single_block::is_supported(bits_per_value),
        }
    }

    /// `Format.overheadPerValue`, in bits.
    pub fn overhead_per_value(self, bits_per_value: u32) -> f32 {
        debug_assert!(self.is_supported(bits_per_value));
        match self {
            Format::Packed => 0.0,
            Format::PackedSingleBlock => {
                let values_per_block = 64 / bits_per_value;
                let overhead = 64 % bits_per_value;
                overhead as f32 / values_per_block as f32
            }
        }
    }

    /// `Format.overheadRatio`: overhead per value / bits per value.
    pub fn overhead_ratio(self, bits_per_value: u32) -> f32 {
        self.overhead_per_value(bits_per_value) / bits_per_value as f32
    }
}

/// `PackedInts.FormatAndBits`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct FormatAndBits {
    pub format: Format,
    pub bits_per_value: u32,
}

/// `PackedInts.fastestFormatAndBits`. `value_count` of `None` is Java's `-1`
/// ("unknown"), which the Java method does not otherwise use.
pub fn fastest_format_and_bits(
    _value_count: Option<usize>,
    bits_per_value: u32,
    acceptable_overhead_ratio: f32,
) -> FormatAndBits {
    // Java: Math.max(COMPACT, r) then Math.min(FASTEST, r); a NaN ratio stays
    // NaN through both (as through `clamp`) and truncates to 0 extra bits.
    let ratio = acceptable_overhead_ratio.clamp(COMPACT, FASTEST);
    let acceptable_overhead_per_value = ratio * bits_per_value as f32; // in bits
    let max_bits_per_value = bits_per_value as i64 + acceptable_overhead_per_value as i64;
    let bpv = bits_per_value as i64;
    let actual = if bpv <= 8 && max_bits_per_value >= 8 {
        8
    } else if bpv <= 16 && max_bits_per_value >= 16 {
        16
    } else if bpv <= 32 && max_bits_per_value >= 32 {
        32
    } else if bpv <= 64 && max_bits_per_value >= 64 {
        64
    } else {
        bits_per_value
    };
    FormatAndBits {
        format: Format::Packed,
        bits_per_value: actual,
    }
}

/// `PackedInts.bitsRequired(long)`: bits needed for values `0..=max_value`
/// (at least 1). Negative is Java's `IllegalArgumentException`.
pub fn bits_required(max_value: i64) -> Result<u32> {
    if max_value < 0 {
        return illegal(format!("maxValue must be non-negative (got: {max_value})"));
    }
    Ok(unsigned_bits_required(max_value as u64))
}

/// `PackedInts.unsignedBitsRequired(long)`: bits needed to store `bits` read
/// as unsigned (at least 1).
pub fn unsigned_bits_required(bits: u64) -> u32 {
    (64 - bits.leading_zeros()).max(1)
}

/// `PackedInts.unsignedBitsRequired(int)`.
pub fn unsigned_bits_required_u32(bits: u32) -> u32 {
    (32 - bits.leading_zeros()).max(1)
}

/// `PackedInts.maxValue`: the largest value `bits_per_value` bits hold
/// (`Long.MAX_VALUE` for 64, as in Java).
pub fn max_value(bits_per_value: u32) -> i64 {
    if bits_per_value >= 64 {
        i64::MAX
    } else {
        !(!0i64 << bits_per_value)
    }
}

/// `PackedInts.checkBlockSize`: `block_size` is a power of two in
/// `min..=max`; returns its log2.
pub fn check_block_size(
    block_size: usize,
    min_block_size: usize,
    max_block_size: usize,
) -> Result<u32> {
    if block_size < min_block_size || block_size > max_block_size {
        return illegal(format!(
            "blockSize must be >= {min_block_size} and <= {max_block_size}, got {block_size}"
        ));
    }
    if !block_size.is_power_of_two() {
        return illegal(format!(
            "blockSize must be a power of two, got {block_size}"
        ));
    }
    Ok(block_size.trailing_zeros())
}

/// `PackedInts.numBlocks`: blocks of `block_size` needed for `size` values.
pub fn num_blocks(size: u64, block_size: usize) -> Result<usize> {
    let n = size.div_ceil(block_size as u64);
    // Java computes this in an `int` and throws when it wrapped.
    if n > i32::MAX as u64 {
        return illegal("size is too large for this block size");
    }
    Ok(n as usize)
}

/// `PackedInts.Reader`: a read-only random-access array of values.
pub trait Reader {
    /// `get(int)`: the value at `index`. Panics past [`Reader::size`].
    fn get(&self, index: usize) -> i64;

    /// `size()`: the number of values.
    fn size(&self) -> usize;

    /// `get(int, long[], int, int)`: reads at least one and at most
    /// `arr.len()` values starting at `index` into `arr`; returns how many.
    fn get_bulk(&self, index: usize, arr: &mut [i64]) -> usize {
        debug_assert!(!arr.is_empty());
        debug_assert!(index < self.size());
        let gets = (self.size() - index).min(arr.len());
        for (o, slot) in arr[..gets].iter_mut().enumerate() {
            *slot = self.get(index + o);
        }
        gets
    }
}

/// `PackedInts.Mutable`: a packed array that can be modified.
pub trait Mutable: Reader {
    /// `getBitsPerValue()`.
    fn bits_per_value(&self) -> u32;

    /// `set(int, long)`.
    fn set(&mut self, index: usize, value: i64);

    /// `set(int, long[], int, int)`: sets at least one and at most `arr.len()`
    /// values starting at `index`; returns how many.
    fn set_bulk(&mut self, index: usize, arr: &[i64]) -> usize {
        default_set_bulk(self, index, arr)
    }

    /// `fill(int, int, long)`: sets `from..to` to `val`.
    fn fill(&mut self, from: usize, to: usize, val: i64) {
        default_fill(self, from, to, val)
    }

    /// `clear()`: sets every value to 0.
    fn clear(&mut self) {
        let size = self.size();
        self.fill(0, size, 0);
    }

    /// Heap bytes this array owns (Java's `ramBytesUsed`, measured for Rust).
    fn ram_bytes_used(&self) -> usize;
}

/// `Mutable.set(int, long[], int, int)`'s base implementation.
pub(crate) fn default_set_bulk<M: Mutable + ?Sized>(m: &mut M, index: usize, arr: &[i64]) -> usize {
    debug_assert!(!arr.is_empty());
    debug_assert!(index < m.size());
    let len = arr.len().min(m.size() - index);
    for (o, &v) in arr[..len].iter().enumerate() {
        m.set(index + o, v);
    }
    len
}

/// `Mutable.fill`'s base implementation.
pub(crate) fn default_fill<M: Mutable + ?Sized>(m: &mut M, from: usize, to: usize, val: i64) {
    debug_assert!(val <= max_value(m.bits_per_value()));
    debug_assert!(from <= to);
    for i in from..to {
        m.set(i, val);
    }
}

/// `PackedInts.NullReader`: `value_count` zeros, stored as nothing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NullReader {
    value_count: usize,
}

impl NullReader {
    /// `NullReader.forCount`.
    pub fn for_count(value_count: usize) -> Self {
        NullReader { value_count }
    }
}

impl Reader for NullReader {
    fn get(&self, _index: usize) -> i64 {
        0
    }
    fn size(&self) -> usize {
        self.value_count
    }
    fn get_bulk(&self, index: usize, arr: &mut [i64]) -> usize {
        debug_assert!(!arr.is_empty());
        debug_assert!(index < self.value_count);
        let len = arr.len().min(self.value_count - index);
        arr[..len].fill(0);
        len
    }
}

/// What `PackedInts.getMutable` can return: one of the two in-memory layouts.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PackedMutable {
    Packed64(Packed64),
    SingleBlock(Packed64SingleBlock),
}

impl PackedMutable {
    /// The layout this array uses.
    pub fn format(&self) -> Format {
        match self {
            PackedMutable::Packed64(_) => Format::Packed,
            PackedMutable::SingleBlock(_) => Format::PackedSingleBlock,
        }
    }

    /// The backing 64-bit blocks (Java's `blocks` field), for byte-exact
    /// comparison against Java and for serialization.
    pub fn blocks(&self) -> &[u64] {
        match self {
            PackedMutable::Packed64(p) => p.blocks(),
            PackedMutable::SingleBlock(p) => p.blocks(),
        }
    }
}

macro_rules! dispatch {
    ($self:ident, $p:ident => $e:expr) => {
        match $self {
            PackedMutable::Packed64($p) => $e,
            PackedMutable::SingleBlock($p) => $e,
        }
    };
}

impl Reader for PackedMutable {
    #[inline]
    fn get(&self, index: usize) -> i64 {
        dispatch!(self, p => p.get(index))
    }
    fn size(&self) -> usize {
        dispatch!(self, p => p.size())
    }
    fn get_bulk(&self, index: usize, arr: &mut [i64]) -> usize {
        dispatch!(self, p => p.get_bulk(index, arr))
    }
}

impl Mutable for PackedMutable {
    fn bits_per_value(&self) -> u32 {
        dispatch!(self, p => p.bits_per_value())
    }
    #[inline]
    fn set(&mut self, index: usize, value: i64) {
        dispatch!(self, p => p.set(index, value))
    }
    fn set_bulk(&mut self, index: usize, arr: &[i64]) -> usize {
        dispatch!(self, p => p.set_bulk(index, arr))
    }
    fn fill(&mut self, from: usize, to: usize, val: i64) {
        dispatch!(self, p => p.fill(from, to, val))
    }
    fn clear(&mut self) {
        dispatch!(self, p => Mutable::clear(p))
    }
    fn ram_bytes_used(&self) -> usize {
        dispatch!(self, p => p.ram_bytes_used())
    }
}

/// `PackedInts.getMutable(int, int, float)`.
pub fn get_mutable(
    value_count: usize,
    bits_per_value: u32,
    acceptable_overhead_ratio: f32,
) -> PackedMutable {
    let fb = fastest_format_and_bits(Some(value_count), bits_per_value, acceptable_overhead_ratio);
    get_mutable_with_format(value_count, fb.bits_per_value, fb.format)
}

/// `PackedInts.getMutable(int, int, Format)`. Panics for a
/// `PackedSingleBlock` width that format does not support, as Java throws.
pub fn get_mutable_with_format(
    value_count: usize,
    bits_per_value: u32,
    format: Format,
) -> PackedMutable {
    match format {
        Format::PackedSingleBlock => {
            PackedMutable::SingleBlock(Packed64SingleBlock::new(value_count, bits_per_value))
        }
        Format::Packed => PackedMutable::Packed64(Packed64::new(value_count, bits_per_value)),
    }
}

/// `PackedInts.copy(Reader, int, Mutable, int, int, int)`: copies
/// `src[src_pos..src_pos+len]` to `dest[dest_pos..]` using at most `mem`
/// bytes of buffer.
pub fn copy<R: Reader + ?Sized, M: Mutable + ?Sized>(
    src: &R,
    src_pos: usize,
    dest: &mut M,
    dest_pos: usize,
    len: usize,
    mem: usize,
) {
    debug_assert!(src_pos + len <= src.size());
    debug_assert!(dest_pos + len <= dest.size());
    let capacity = mem >> 3;
    if capacity == 0 {
        for i in 0..len {
            dest.set(dest_pos + i, src.get(src_pos + i));
        }
    } else if len > 0 {
        let mut buf = vec![0i64; capacity.min(len)];
        copy_with_buffer(src, src_pos, dest, dest_pos, len, &mut buf);
    }
}

/// `PackedInts.copy(Reader, int, Mutable, int, int, long[])`: the same with a
/// caller's buffer.
pub fn copy_with_buffer<R: Reader + ?Sized, M: Mutable + ?Sized>(
    src: &R,
    mut src_pos: usize,
    dest: &mut M,
    mut dest_pos: usize,
    mut len: usize,
    buf: &mut [i64],
) {
    assert!(!buf.is_empty());
    let mut remaining = 0usize;
    while len > 0 {
        let want = len.min(buf.len() - remaining);
        let read = src.get_bulk(src_pos, &mut buf[remaining..remaining + want]);
        debug_assert!(read > 0);
        src_pos += read;
        len -= read;
        remaining += read;
        let written = dest.set_bulk(dest_pos, &buf[..remaining]);
        debug_assert!(written > 0);
        dest_pos += written;
        if written < remaining {
            buf.copy_within(written..remaining, 0);
        }
        remaining -= written;
    }
    while remaining > 0 {
        let written = dest.set_bulk(dest_pos, &buf[..remaining]);
        dest_pos += written;
        remaining -= written;
        buf.copy_within(written..written + remaining, 0);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bits_required_matches_java() {
        assert_eq!(bits_required(0).unwrap(), 1);
        assert_eq!(bits_required(1).unwrap(), 1);
        assert_eq!(bits_required(2).unwrap(), 2);
        assert_eq!(bits_required(255).unwrap(), 8);
        assert_eq!(bits_required(256).unwrap(), 9);
        assert_eq!(bits_required(i64::MAX).unwrap(), 63);
        assert!(bits_required(-1).is_err());
        assert_eq!(unsigned_bits_required(u64::MAX), 64);
        assert_eq!(unsigned_bits_required(0), 1);
        assert_eq!(unsigned_bits_required_u32(0), 1);
        assert_eq!(unsigned_bits_required_u32(u32::MAX), 32);
        assert_eq!(unsigned_bits_required_u32(5), 3);
    }

    #[test]
    fn max_value_matches_java() {
        assert_eq!(max_value(1), 1);
        assert_eq!(max_value(8), 255);
        assert_eq!(max_value(63), i64::MAX);
        assert_eq!(max_value(64), i64::MAX);
    }

    #[test]
    fn version_checks() {
        assert!(check_version(VERSION_CURRENT).is_ok());
        assert!(check_version(1).is_err());
        assert!(check_version(3).is_err());
    }

    #[test]
    fn format_ids_and_counts() {
        assert_eq!(Format::by_id(0).unwrap(), Format::Packed);
        assert_eq!(Format::by_id(1).unwrap(), Format::PackedSingleBlock);
        assert!(Format::by_id(2).is_err());
        assert_eq!(Format::Packed.id(), 0);
        assert_eq!(Format::PackedSingleBlock.id(), 1);
        // 10 values of 7 bits: 70 bits -> 9 bytes -> 2 longs.
        assert_eq!(Format::Packed.byte_count(VERSION_CURRENT, 10, 7), 9);
        assert_eq!(Format::Packed.long_count(VERSION_CURRENT, 10, 7), 2);
        // single block, 7 bits: 9 values per block -> 2 blocks -> 16 bytes.
        assert_eq!(
            Format::PackedSingleBlock.long_count(VERSION_CURRENT, 10, 7),
            2
        );
        assert_eq!(
            Format::PackedSingleBlock.byte_count(VERSION_CURRENT, 10, 7),
            16
        );
        assert!(Format::Packed.is_supported(64));
        assert!(!Format::Packed.is_supported(0));
        assert!(Format::PackedSingleBlock.is_supported(21));
        assert!(!Format::PackedSingleBlock.is_supported(11));
        assert_eq!(Format::Packed.overhead_per_value(7), 0.0);
        // 64 % 7 = 1 wasted bit over 9 values.
        assert_eq!(Format::PackedSingleBlock.overhead_per_value(7), 1.0 / 9.0);
        assert_eq!(
            Format::PackedSingleBlock.overhead_ratio(7),
            (1.0 / 9.0) / 7.0
        );
    }

    #[test]
    fn fastest_format_rounds_up_within_the_overhead() {
        let f = |bpv, r| fastest_format_and_bits(Some(100), bpv, r).bits_per_value;
        assert_eq!(f(5, COMPACT), 5);
        assert_eq!(f(5, FASTEST), 8);
        assert_eq!(f(7, DEFAULT), 8); // 7 + 1.75 -> 8
        assert_eq!(f(6, DEFAULT), 6); // 6 + 1.5 -> 7
        assert_eq!(f(13, DEFAULT), 16);
        assert_eq!(f(25, DEFAULT), 25);
        assert_eq!(f(25, FAST), 32);
        assert_eq!(f(60, DEFAULT), 64);
        assert_eq!(f(64, COMPACT), 64);
        // Ratios outside [COMPACT, FASTEST] clamp.
        assert_eq!(f(5, -3.0), 5);
        assert_eq!(f(5, 100.0), 8);
        assert_eq!(
            fastest_format_and_bits(None, 3, FASTEST).format,
            Format::Packed
        );
    }

    #[test]
    fn block_size_checks() {
        assert_eq!(check_block_size(64, 64, 1 << 20).unwrap(), 6);
        assert!(check_block_size(32, 64, 1 << 20).is_err());
        assert!(check_block_size(1 << 21, 64, 1 << 20).is_err());
        assert!(check_block_size(96, 64, 1 << 20).is_err());
        assert_eq!(num_blocks(0, 64).unwrap(), 0);
        assert_eq!(num_blocks(64, 64).unwrap(), 1);
        assert_eq!(num_blocks(65, 64).unwrap(), 2);
        assert!(num_blocks(u64::MAX, 1).is_err());
    }

    #[test]
    fn null_reader_is_all_zeros() {
        let r = NullReader::for_count(10);
        assert_eq!(r.size(), 10);
        assert_eq!(r.get(3), 0);
        let mut arr = [7i64; 20];
        assert_eq!(r.get_bulk(4, &mut arr), 6);
        assert!(arr[..6].iter().all(|&v| v == 0));
        assert_eq!(arr[6], 7);
    }

    #[test]
    fn copy_between_layouts_with_every_buffer_size() {
        for mem in [0usize, 8, 16, 24, 64, 1024] {
            let mut src = get_mutable_with_format(300, 13, Format::Packed);
            for i in 0..300 {
                src.set(i, (i as i64 * 31) & 0x1fff);
            }
            let mut dest = get_mutable_with_format(400, 16, Format::PackedSingleBlock);
            copy(&src, 7, &mut dest, 50, 250, mem);
            for i in 0..250 {
                assert_eq!(dest.get(50 + i), src.get(7 + i), "mem={mem} i={i}");
            }
            assert_eq!(dest.get(49), 0);
            assert_eq!(dest.get(300), 0);
        }
    }

    #[test]
    fn packed_mutable_dispatch() {
        let mut m = get_mutable(100, 5, COMPACT);
        assert_eq!(m.format(), Format::Packed);
        assert_eq!(m.bits_per_value(), 5);
        m.fill(0, 100, 17);
        assert_eq!(m.get(99), 17);
        Mutable::clear(&mut m);
        assert_eq!(m.get(99), 0);
        assert_eq!(m.set_bulk(98, &[1, 2, 3]), 2);
        let mut out = [0i64; 2];
        assert_eq!(m.get_bulk(98, &mut out), 2);
        assert_eq!(out, [1, 2]);
        assert!(m.ram_bytes_used() >= m.blocks().len() * 8);
        let mut s = get_mutable_with_format(100, 5, Format::PackedSingleBlock);
        assert_eq!(s.format(), Format::PackedSingleBlock);
        s.set(3, 9);
        assert_eq!(s.get(3), 9);
        assert_eq!(s.blocks().len(), 100usize.div_ceil(12));
        Mutable::clear(&mut s);
        assert_eq!(s.get(3), 0);
    }
}
