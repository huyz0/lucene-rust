//! Port of `org.apache.lucene.util.packed.DirectReader.getInstance(...).get(index)`.
//!
//! [`get`] is one generic bit-position formula for any width, re-validated per
//! call. Lucene's fourteen width-specialized `DirectPackedReaderNN` classes
//! give each JIT call site a monomorphic reader; the equivalent here is
//! [`DirectReader::with_width`], which runs a caller's loop in a body compiled
//! for one width ([`FixedWidthReader`]). The engine's own readers do not use
//! it yet: they call [`get`] per value (dense doc values go through
//! `lucene_util::packed_longs::PackedLongs` instead). Shared by [`crate::doc_values`] (plain
//! bit-packed value arrays) and [`crate::direct_monotonic`] (each block's
//! deltas-from-expected-average array).
//!
//! `pub` for the same reason [`crate::for_util`] is: a per-value decode
//! primitive on the doc-values and monotonic-sequence paths, and
//! `DirectReader.getInstance` is public in Lucene, so the two can be
//! benchmarked directly against each other.

use lucene_store::Result;

/// `bits_per_value` must be one of the widths `DirectWriter` supports; anything
/// else is rejected here, exactly where `DirectReader.getInstance`'s `switch`
/// falls through to `IllegalArgumentException`. Java validates once per reader
/// and this validates once per read, but the alternative is worse: several
/// callers take `bits_per_value` straight off disk mid-lookup rather than at
/// parse time (`doc_values`' varying-bits-per-value block header,
/// `direct_monotonic`'s per-block `bpvs`), and an unsupported width reaches the
/// mask below as `1u64 << bits_per_value` -- a debug-build panic and, worse, a
/// silently masked shift returning a plausible wrong value in release.
///
/// `index` addresses the `index`-th `bits_per_value`-wide value packed
/// little-endian (LSB-first within each byte) starting at byte 0 of `slice`.
#[inline]
pub fn get(slice: &[u8], bits_per_value: u8, index: i64) -> Result<i64> {
    if !is_supported_bits(bits_per_value) {
        return Err(lucene_store::Error::Corrupted(format!(
            "unsupported DirectReader bitsPerValue: {bits_per_value}"
        )));
    }
    // `index * bits_per_value` fits comfortably in u64: `index` addresses an
    // element of an in-memory-decoded array, itself bounded by `slice.len() *
    // 8` bits (a real allocated buffer, far under u64::MAX). A wide u128
    // multiply here is unnecessary overhead on a hot per-value decode path
    // (called once per doc-values lookup / monotonic-sequence element).
    let bit_pos = (index as u64).wrapping_mul(bits_per_value as u64);
    let byte_pos =
        usize::try_from(bit_pos >> 3).map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let shift = (bit_pos & 7) as u32;
    // ARITH: `shift` is masked to `0..=7`, and `is_supported_bits` above caps
    // `bits_per_value` at 64. Those two alone would allow `bytes_needed == 9`,
    // which is *not* enough for the shift loop below -- `8 * 8 == 64` is
    // already a shift overflow on a `u64`. The real bound is tighter and comes
    // from `SUPPORTED_BITS`: `shift` is non-zero only when `bit_pos` is not a
    // multiple of 8, which for a supported width means the width itself is not
    // a multiple of 8, and those stop at 28. So either `shift == 0` and
    // `bytes_needed <= 8`, or `bits_per_value <= 28` and `bytes_needed <=
    // ceil(35 / 8) == 5`. Either way `bytes_needed <= 8` and `8 * i <= 56`.
    #[allow(clippy::arithmetic_side_effects)]
    let bytes_needed = (shift as usize + bits_per_value as usize).div_ceil(8);
    debug_assert!(
        bytes_needed <= 8,
        "bytes_needed={bytes_needed} would overflow the shift below"
    );

    // `index` reaches here from a doc-values ordinal or a monotonic-block
    // index, both of which can come off disk. On a 64-bit target `bit_pos` is
    // a `u64` and `bit_pos >> 3` therefore never exceeds `2^61 - 1`, so
    // `byte_pos + 8` could not overflow there anyway -- but on a 32-bit target
    // the `try_from` above admits anything up to `u32::MAX`, where it would.
    // One comparison against the slice length, hoisted out of both reads
    // below, makes every `byte_pos + n` here provably in range on every
    // target, since a slice length is at most `isize::MAX`. It also turns a
    // wildly out-of-range ordinal into an `Eof` before the two `slice.get`
    // calls rather than after.
    if byte_pos > slice.len() {
        return Err(lucene_store::Error::Eof { offset: byte_pos });
    }

    // One wide load whenever eight bytes are in range, which is what
    // `DirectWriter` pads its output for -- `padding_bytes_needed` below is
    // this port's copy of the same rule, and `Lucene90DocValuesConsumer` writes
    // that padding for exactly this reason. Every width `DirectWriter` supports
    // needs at most eight bytes: the non-byte-aligned ones stop at 28 bits
    // (five bytes at worst), and the byte-aligned ones never carry a shift.
    //
    // The byte-at-a-time loop below is the tail fallback: one load, one shift
    // and one OR per byte, so it gets linearly slower with the width, whereas
    // Lucene's `DirectPackedReaderNN` classes each do a single
    // `readInt`/`readLong`.
    // ARITH: `byte_pos <= slice.len() <= isize::MAX` was just checked, and
    // `bytes_needed <= 9`, so neither sum can overflow `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    let acc = if let Some(window) = slice.get(byte_pos..byte_pos + 8) {
        u64::from_le_bytes(window.try_into().expect("exactly 8 bytes"))
    } else {
        let Some(bytes) = slice.get(byte_pos..byte_pos + bytes_needed) else {
            return Err(lucene_store::Error::Eof { offset: byte_pos });
        };
        let mut acc: u64 = 0;
        // ARITH: `bytes.len() == bytes_needed <= 8` (see the bound above),
        // so `i <= 7` and `8 * i <= 56`, a legal `u64` shift.
        #[allow(clippy::arithmetic_side_effects)]
        for (i, &b) in bytes.iter().enumerate() {
            acc |= (b as u64) << (8 * i);
        }
        acc
    };
    let acc = acc >> shift;
    // ARITH: the `else` arm has `bits_per_value < 64` (the `== 64` case is the
    // arm above and `is_supported_bits` rejects everything over 64), so the
    // shift is in range and its result is at least 1.
    #[allow(clippy::arithmetic_side_effects)]
    let mask: u64 = if bits_per_value == 64 {
        u64::MAX
    } else {
        (1u64 << bits_per_value) - 1
    };
    Ok((acc & mask) as i64)
}

/// `DirectReader.getInstance(slice, bitsPerValue)`: a reader for one packed
/// array whose width is checked once, here, rather than on every read as
/// [`get`] must.
///
/// Lucene hands back one of fourteen width-specialized `LongValues` classes;
/// the call site sees a monomorphic `get` that is a load, a shift and a mask.
/// [`get`] re-validates the width and does its arithmetic with the width as a
/// variable: against a monomorphic Lucene call site that measured 0.57-0.94x
/// on every width but one. This validates once. [`Self::get`] still picks the
/// width per call, which a hot loop pays for (a per-call dispatch measured
/// slower than the generic formula); a loop should run inside
/// [`Self::with_width`], whose [`FixedWidthReader`] is 1.08-1.33x Lucene at
/// every width. Near the end of an unpadded slice both defer to [`get`], so
/// errors are unchanged.
#[derive(Debug, Clone, Copy)]
pub struct DirectReader<'a> {
    slice: &'a [u8],
    bits_per_value: u8,
}

impl<'a> DirectReader<'a> {
    /// `Err` for a width `DirectWriter` cannot emit, as [`get`] reports it.
    pub fn new(slice: &'a [u8], bits_per_value: u8) -> Result<Self> {
        if !is_supported_bits(bits_per_value) {
            return Err(lucene_store::Error::Corrupted(format!(
                "unsupported DirectReader bitsPerValue: {bits_per_value}"
            )));
        }
        Ok(DirectReader {
            slice,
            bits_per_value,
        })
    }

    /// The `index`-th value; the same answer and the same errors as [`get`].
    ///
    /// Dispatched on the width, per call, to a body where it is a constant;
    /// for one-off reads. A loop belongs in [`Self::with_width`]. Anything the
    /// fast body cannot answer -- an index near the end of an unpadded slice,
    /// a negative one -- goes to [`get`], so errors are unchanged.
    #[inline]
    pub fn get(&self, index: i64) -> Result<i64> {
        let fast = match self.bits_per_value {
            1 => read_at_width::<1>(self.slice, index),
            2 => read_at_width::<2>(self.slice, index),
            4 => read_at_width::<4>(self.slice, index),
            8 => read_at_width::<8>(self.slice, index),
            12 => read_at_width::<12>(self.slice, index),
            16 => read_at_width::<16>(self.slice, index),
            20 => read_at_width::<20>(self.slice, index),
            24 => read_at_width::<24>(self.slice, index),
            28 => read_at_width::<28>(self.slice, index),
            32 => read_at_width::<32>(self.slice, index),
            40 => read_at_width::<40>(self.slice, index),
            48 => read_at_width::<48>(self.slice, index),
            56 => read_at_width::<56>(self.slice, index),
            64 => read_at_width::<64>(self.slice, index),
            _ => None,
        };
        match fast {
            Some(v) => Ok(v),
            None => get(self.slice, self.bits_per_value, index),
        }
    }

    /// Runs `visitor` with this reader at a compile-time width: the dispatch
    /// happens once, here, and the visitor's loop is compiled per width.
    ///
    /// This is what gives a Rust loop what a monomorphic Lucene call site
    /// gets from `DirectPackedReaderNN`. [`Self::get`] dispatches on every
    /// call, and a hot loop pays that jump each time; a loop inside
    /// [`WidthVisitor::visit`] reads through [`FixedWidthReader::get`], whose
    /// width is a constant.
    pub fn with_width<V: WidthVisitor>(&self, visitor: V) -> V::Output {
        let slice = self.slice;
        match self.bits_per_value {
            1 => visitor.visit(FixedWidthReader::<1>::new(slice)),
            2 => visitor.visit(FixedWidthReader::<2>::new(slice)),
            4 => visitor.visit(FixedWidthReader::<4>::new(slice)),
            8 => visitor.visit(FixedWidthReader::<8>::new(slice)),
            12 => visitor.visit(FixedWidthReader::<12>::new(slice)),
            16 => visitor.visit(FixedWidthReader::<16>::new(slice)),
            20 => visitor.visit(FixedWidthReader::<20>::new(slice)),
            24 => visitor.visit(FixedWidthReader::<24>::new(slice)),
            28 => visitor.visit(FixedWidthReader::<28>::new(slice)),
            32 => visitor.visit(FixedWidthReader::<32>::new(slice)),
            40 => visitor.visit(FixedWidthReader::<40>::new(slice)),
            48 => visitor.visit(FixedWidthReader::<48>::new(slice)),
            56 => visitor.visit(FixedWidthReader::<56>::new(slice)),
            // `new` admits only supported widths, and 64 is the last.
            _ => visitor.visit(FixedWidthReader::<64>::new(slice)),
        }
    }
}

/// [`DirectReader::get`]'s fast body alone, without building a reader: the
/// `index`-th value, or `None` where [`get`] must answer (an unsupported
/// width, an index the fast read cannot reach).
#[inline(always)]
pub fn get_fast(slice: &[u8], bits_per_value: u8, index: i64) -> Option<i64> {
    match bits_per_value {
        1 => read_at_width::<1>(slice, index),
        2 => read_at_width::<2>(slice, index),
        4 => read_at_width::<4>(slice, index),
        8 => read_at_width::<8>(slice, index),
        12 => read_at_width::<12>(slice, index),
        16 => read_at_width::<16>(slice, index),
        20 => read_at_width::<20>(slice, index),
        24 => read_at_width::<24>(slice, index),
        28 => read_at_width::<28>(slice, index),
        32 => read_at_width::<32>(slice, index),
        40 => read_at_width::<40>(slice, index),
        48 => read_at_width::<48>(slice, index),
        56 => read_at_width::<56>(slice, index),
        64 => read_at_width::<64>(slice, index),
        _ => None,
    }
}

/// A computation over a [`DirectReader`] at a compile-time width; see
/// [`DirectReader::with_width`].
pub trait WidthVisitor {
    type Output;
    fn visit<const B: u32>(self, reader: FixedWidthReader<'_, B>) -> Self::Output;
}

/// A [`DirectReader`] whose width `B` is a constant.
#[derive(Debug, Clone, Copy)]
pub struct FixedWidthReader<'a, const B: u32> {
    slice: &'a [u8],
    // One comparison and an unchecked load per read, as `PackedLongs` does:
    // a safe slice read here costs two bounds checks, and measured 25%
    // behind Lucene's per-width readers.
    fast: lucene_util::packed_longs::FixedPackedLongs<'a, B>,
}

impl<'a, const B: u32> FixedWidthReader<'a, B> {
    fn new(slice: &'a [u8]) -> Self {
        FixedWidthReader {
            slice,
            fast: lucene_util::packed_longs::FixedPackedLongs::new(slice),
        }
    }

    /// The `index`-th value; the same answer and the same errors as [`get`].
    #[inline(always)]
    pub fn get(&self, index: i64) -> Result<i64> {
        // A negative index cast to `u64` is above every fast limit, so the
        // one comparison inside `get` also sends it to the checked path --
        // no separate sign test (Java's segment bounds check is one compare
        // too).
        if let Some(v) = self.fast.get(index as u64) {
            return Ok(v as i64);
        }
        get(self.slice, B as u8, index)
    }
}

/// The `index`-th value at the constant width `B`, or `None` where [`get`]
/// must answer (a negative or overflowing index, fewer bytes left than the
/// read needs). Byte-aligned widths are one plain load of exactly their
/// bytes, as `DirectPackedReader8/16/32/64` read them; the others a constant
/// multiply, shift and mask.
#[inline(always)]
fn read_at_width<const B: u8>(slice: &[u8], index: i64) -> Option<i64> {
    let index = u64::try_from(index).ok()?;
    let bit_pos = index.checked_mul(u64::from(B))?;
    let byte_pos = usize::try_from(bit_pos >> 3).ok()?;
    let tail = slice.get(byte_pos..)?;
    let value = match B {
        // Whole bytes: exactly the value's own bytes, as
        // `DirectPackedReader8/16/32/64` read them.
        8 => u64::from(*tail.first()?),
        16 => u64::from(u16::from_le_bytes(*tail.first_chunk::<2>()?)),
        32 => u64::from(u32::from_le_bytes(*tail.first_chunk::<4>()?)),
        64 => u64::from_le_bytes(*tail.first_chunk::<8>()?),
        // Widths dividing 8 never straddle a byte.
        1 | 2 | 4 => u64::from(*tail.first()? >> (bit_pos & 7)),
        // Everything else fits one eight-byte window: the shift is
        // non-zero only for widths up to 28, and `shift + B <= 35`.
        _ => u64::from_le_bytes(*tail.first_chunk::<8>()?) >> (bit_pos & 7),
    };
    let mask = u64::MAX
        .checked_shr(64u32.saturating_sub(u32::from(B)))
        .unwrap_or(0);
    Some((value & mask) as i64)
}

/// Port of `DirectWriter.add`/`flush`'s bit-packing (encode side of [`get`]):
/// packs `values` (each assumed to fit in `bits_per_value` unsigned bits) as
/// one little-endian, LSB-first-within-byte bitstream -- the exact inverse
/// of `get`'s formula, so this port doesn't need Java's thirteen
/// width-specialized encoders either.
// ARITH: every product is computed in `u128` from a slice length and a `u8`,
// so it cannot overflow. Inside the loop `bit_off` is masked to `0..=7`,
// `take = min(remaining, 8 - bit_off)` is in `0..=8`, `remaining` only ever
// decreases by `take` (and the loop stops at 0), and `bit_pos` advances by
// `take` a bounded number of times. This is the encode side: no operand here
// comes off disk.
#[allow(clippy::arithmetic_side_effects)]
pub fn encode(values: &[i64], bits_per_value: u8) -> Vec<u8> {
    let bpv = u32::from(bits_per_value.min(64));
    let total_bits = values.len() as u128 * u128::from(bpv);
    let n_bytes = total_bits.div_ceil(8) as usize;
    let mask = if bpv == 64 {
        u64::MAX
    } else {
        (1u64 << bpv) - 1
    };
    let mut out = Vec::with_capacity(n_bytes);
    // Little-endian bit packing, eight bytes at a time: `acc` holds the
    // `nbits < 64` bits not yet written, so adding one value (at most 64
    // bits) never overflows its 128.
    let (mut acc, mut nbits) = (0u128, 0u32);
    for &v in values {
        acc |= u128::from(v as u64 & mask) << nbits;
        nbits += bpv;
        if nbits >= 64 {
            out.extend_from_slice(&(acc as u64).to_le_bytes());
            acc >>= 64;
            nbits -= 64;
        }
    }
    out.extend_from_slice(&(acc as u64).to_le_bytes()[..nbits.div_ceil(8) as usize]);
    debug_assert_eq!(out.len(), n_bytes);
    out
}

/// `DirectWriter`'s supported bit widths -- `bitsRequired`/`unsignedBitsRequired`
/// always round up to one of these (`DirectWriter.roundBits`).
const SUPPORTED_BITS: [u32; 14] = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 40, 48, 56, 64];

/// [`SUPPORTED_BITS`] below 64, as a bit set, so the check in [`get`] is a
/// shift and a test rather than a scan of the table. 64 is handled separately
/// rather than widening this to `u128`: a 128-bit variable shift costs several
/// instructions, and this sits on a per-value decode path.
// ARITH: `i` indexes a 14-element array and the loop stops at its length.
// Evaluated at compile time, so an overflow here would be a build error, not
// a runtime panic.
#[allow(clippy::arithmetic_side_effects)]
const SUPPORTED_BITS_MASK: u64 = {
    let mut mask = 0u64;
    let mut i = 0;
    while i < SUPPORTED_BITS.len() {
        if SUPPORTED_BITS[i] < 64 {
            mask |= 1u64 << SUPPORTED_BITS[i];
        }
        i += 1;
    }
    mask
};

/// Port of `DirectWriter.checkBitsPerValue`'s membership test: is
/// `bits_per_value` one of the fourteen widths `DirectWriter` can emit?
#[inline]
pub(crate) fn is_supported_bits(bits_per_value: u8) -> bool {
    bits_per_value == 64
        || (bits_per_value < 64 && (SUPPORTED_BITS_MASK >> bits_per_value) & 1 != 0)
}

/// Port of `DirectWriter.unsignedBitsRequired`: the minimum bit width (among
/// [`SUPPORTED_BITS`]) that can hold `max_value` interpreted as unsigned.
// ARITH: `u64::leading_zeros()` returns `0..=64`, so `64 - lz` is in `0..=64`.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn unsigned_bits_required(max_value: i64) -> u8 {
    let bits = if max_value == 0 {
        1
    } else {
        64 - (max_value as u64).leading_zeros()
    };
    SUPPORTED_BITS
        .into_iter()
        .find(|&w| w >= bits)
        .unwrap_or(64) as u8
}

/// Port of `DirectWriter.paddingBytesNeeded`: extra zero bytes appended after
/// a block's packed data so a reader could always do one fixed-width
/// (u8/u16/u32/u64) read without touching the next block's bytes. This
/// port's own [`get`] is bounds-checked and never needs this, but the
/// padding is part of the on-disk byte layout (it shifts every subsequent
/// block's offset), so a writer must still emit it for wire compatibility.
// ARITH: each subtraction sits under the branch that establishes it --
// `64 - bits` only for `bits > 32`, and so on -- but that alone only bounds
// the *lower* end. The upper end comes from the call sites: both of them
// (`direct_monotonic::write`, `doc_values`' numeric writer) pass a
// width that `unsigned_bits_required` returned, which is one of the fourteen
// `SUPPORTED_BITS` and so never exceeds 64. A width read off disk must not be
// passed here; `get` is the entry point that validates one.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn padding_bytes_needed(bits_per_value: u8) -> usize {
    let padding_bits = if bits_per_value > 32 {
        64 - bits_per_value as u32
    } else if bits_per_value > 16 {
        32 - bits_per_value as u32
    } else if bits_per_value > 8 {
        16 - bits_per_value as u32
    } else {
        0
    };
    (padding_bits as usize).div_ceil(8)
}

#[cfg(test)]
mod tests;
