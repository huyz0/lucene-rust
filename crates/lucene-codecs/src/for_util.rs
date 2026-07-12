//! Port of `org.apache.lucene.codecs.lucene104.ForUtil`/`PForUtil`: the
//! bit-packed "patched frame-of-reference" (PFOR) bulk (de)coder used for a
//! full 256-value block (`ForUtil.BLOCK_SIZE`). Scoped to **decode only**
//! (read path first, see the `architecture` skill) and to the exact algorithm
//! Java uses — this is a direct transliteration of `ForUtil.decode1..decode16`/
//! `decodeSlow` and `PForUtil.decode`, not a redesign, because the on-disk
//! layout (which values land in which bits of which 32-bit little-endian word)
//! is the compatibility contract; a faster/SIMD re-expression of the same
//! bit-layout is future work (see `docs/parity.md`).
//!
//! ## Why it looks like scalar "SIMD-in-a-register" bit twiddling
//!
//! For `bitsPerValue <= 8` (`<= 16`), Java's *writer* first packs 4 (2)
//! consecutive values into one 32-bit int's 4 (2) byte (halfword) lanes
//! (`collapse8`/`collapse16`), then bit-packs *that* array with a "primitive
//! size" of 8 (16) instead of 32. Every mask/shift in `decode1..decode16` is
//! lane-replicated (`MASK8_x`/`MASK16_x` = the same `x`-bit mask repeated in
//! every byte/halfword lane) and every shift amount stays under the lane
//! width, so the four (two) lanes never interact — it's genuinely 4 (2)
//! independent bit-packed streams processed with one instruction stream, not
//! a different algorithm. [`expand8`]/[`expand16`] un-interleave the lanes
//! back into 256 individual values afterward. `bitsPerValue > 16` skips the
//! lane trick entirely (`decodeSlow`, plain 32-bit-wide packing).

use lucene_store::data_input::DataInput;
use lucene_store::Result;

/// `ForUtil.BLOCK_SIZE` / `PForUtil`'s implicit block width.
pub const BLOCK_SIZE: usize = 256;

/// `PForUtil.MAX_EXCEPTIONS`: at most 7 patched values per block (3 bits of a
/// token byte, `numExceptions = token >>> 5`).
const MAX_EXCEPTIONS: usize = 7;

#[inline]
fn mask32(bits: u32) -> u32 {
    if bits == 0 {
        0
    } else if bits >= 32 {
        u32::MAX
    } else {
        (1u32 << bits) - 1
    }
}

#[inline]
fn expand_mask16(m16: u32) -> u32 {
    m16 | (m16 << 16)
}

#[inline]
fn expand_mask8(m8: u32) -> u32 {
    expand_mask16(m8 | (m8 << 8))
}

/// `ForUtil.mask16`: an n-bit mask replicated into both 16-bit halfword lanes.
#[inline]
fn mask16(bits: u32) -> u32 {
    expand_mask16(mask32(bits))
}

/// `ForUtil.mask8`: an n-bit mask replicated into all four byte lanes.
#[inline]
fn mask8(bits: u32) -> u32 {
    expand_mask8(mask32(bits))
}

/// `ForUtil.expand8`: un-interleaves 64 four-byte-lane-packed ints (produced
/// by the decode1..decode8 helpers) into 256 individual values.
fn expand8(arr: &mut [u32; BLOCK_SIZE]) {
    for i in 0..64 {
        let l = arr[i];
        arr[i] = (l >> 24) & 0xFF;
        arr[64 + i] = (l >> 16) & 0xFF;
        arr[128 + i] = (l >> 8) & 0xFF;
        arr[192 + i] = l & 0xFF;
    }
}

/// `ForUtil.expand16`: un-interleaves 128 two-halfword-lane-packed ints into
/// 256 individual values.
fn expand16(arr: &mut [u32; BLOCK_SIZE]) {
    for i in 0..128 {
        let l = arr[i];
        arr[i] = (l >> 16) & 0xFFFF;
        arr[128 + i] = l & 0xFFFF;
    }
}

/// `PostingDecodingUtil.splitInts` (the default, non-vectorized
/// implementation shipped in `lucene101`/`lucene103`'s backward-codecs, which
/// is exactly what the JIT would otherwise auto-vectorize to): read `count`
/// little-endian words into `c[c_index..]`, then for every `i` and every `j`
/// with `b_shift - j*dec > 0`, extract `(c[c_index+i] >> (b_shift - j*dec)) &
/// b_mask` into `b[count*j + i]`; finally mask `c[c_index+i]` down to
/// `c_mask` in place (this last masked value is itself part of the decoded
/// output whenever `b` and `c` alias the same array at a disjoint offset —
/// see each `decodeN` call site).
#[allow(clippy::too_many_arguments)]
fn split_ints<R: DataInput>(
    r: &mut R,
    b: &mut [u32],
    c: &mut [u32],
    count: usize,
    b_shift: u32,
    dec: u32,
    b_mask: u32,
    c_index: usize,
    c_mask: u32,
) -> Result<()> {
    for k in 0..count {
        c[c_index + k] = r.read_u32_le()?;
    }
    // Java: `(bShift - 1) / dec` using signed int division truncating toward
    // zero; `bShift == 0` (only reachable for `bits_per_value == 32` via
    // `decode_slow`) still yields `maxIter == 0` (one iteration at shift 0),
    // matching `(-1)/dec == 0` in Java — hence the signed intermediate here.
    let max_iter = ((b_shift as i32 - 1) / dec as i32) as u32;
    for i in 0..count {
        let cv = c[c_index + i];
        for j in 0..=max_iter {
            let shift = b_shift.wrapping_sub(j * dec);
            b[count * (j as usize) + i] = (cv >> shift) & b_mask;
        }
        c[c_index + i] &= c_mask;
    }
    Ok(())
}

/// `ForUtil.decodeSlow`: the `bitsPerValue > 16` fallback (plain 32-bit-wide
/// packing, no lane interleaving).
fn decode_slow<R: DataInput>(
    bits_per_value: u32,
    r: &mut R,
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    let num_ints = (bits_per_value as usize) * 8;
    let mask = mask32(bits_per_value);
    let mut tmp = [0u32; BLOCK_SIZE];
    {
        let (b, _) = ints.split_at_mut(num_ints.max(1));
        split_ints(
            r,
            b,
            &mut tmp,
            num_ints,
            32 - bits_per_value,
            32,
            mask,
            0,
            u32::MAX,
        )?;
    }
    let remaining_bits_per_int = 32 - bits_per_value;
    let mask32_remaining = mask32(remaining_bits_per_int);
    let mut tmp_idx = 0usize;
    let mut remaining_bits = remaining_bits_per_int;
    for slot in ints.iter_mut().take(BLOCK_SIZE).skip(num_ints) {
        let mut b = bits_per_value - remaining_bits;
        let mut l = (tmp[tmp_idx] & mask32(remaining_bits)) << b;
        tmp_idx += 1;
        while b >= remaining_bits_per_int {
            b -= remaining_bits_per_int;
            l |= (tmp[tmp_idx] & mask32_remaining) << b;
            tmp_idx += 1;
        }
        if b > 0 {
            l |= (tmp[tmp_idx] >> (remaining_bits_per_int - b)) & mask32(b);
            remaining_bits = remaining_bits_per_int - b;
        } else {
            remaining_bits = remaining_bits_per_int;
        }
        *slot = l;
    }
    Ok(())
}

macro_rules! mask8_const {
    ($n:expr) => {
        mask8($n)
    };
}
macro_rules! mask16_const {
    ($n:expr) => {
        mask16($n)
    };
}

fn decode1<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    let (b, c) = ints.split_at_mut(56);
    split_ints(r, b, c, 8, 7, 1, mask8_const!(1), 0, mask8_const!(1))
}

fn decode2<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    let (b, c) = ints.split_at_mut(48);
    split_ints(r, b, c, 16, 6, 2, mask8_const!(2), 0, mask8_const!(2))
}

fn decode3<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(r, ints, tmp, 24, 5, 3, mask8_const!(3), 0, mask8_const!(2))?;
    let mask1 = mask8_const!(1);
    let mut tmp_idx = 0;
    let mut ints_idx = 48;
    for _ in 0..8 {
        let mut l0 = tmp[tmp_idx] << 1;
        l0 |= (tmp[tmp_idx + 1] >> 1) & mask1;
        ints[ints_idx] = l0;
        let mut l1 = (tmp[tmp_idx + 1] & mask1) << 2;
        l1 |= tmp[tmp_idx + 2];
        ints[ints_idx + 1] = l1;
        tmp_idx += 3;
        ints_idx += 2;
    }
    Ok(())
}

fn decode4<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    let (b, c) = ints.split_at_mut(32);
    split_ints(r, b, c, 32, 4, 4, mask8_const!(4), 0, mask8_const!(4))
}

fn decode5<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(r, ints, tmp, 40, 3, 5, mask8_const!(5), 0, mask8_const!(3))?;
    let (mask1, mask2) = (mask8_const!(1), mask8_const!(2));
    let mut tmp_idx = 0;
    let mut ints_idx = 40;
    for _ in 0..8 {
        let mut l0 = tmp[tmp_idx] << 2;
        l0 |= (tmp[tmp_idx + 1] >> 1) & mask2;
        ints[ints_idx] = l0;
        let mut l1 = (tmp[tmp_idx + 1] & mask1) << 4;
        l1 |= tmp[tmp_idx + 2] << 1;
        l1 |= (tmp[tmp_idx + 3] >> 2) & mask1;
        ints[ints_idx + 1] = l1;
        let mut l2 = (tmp[tmp_idx + 3] & mask2) << 3;
        l2 |= tmp[tmp_idx + 4];
        ints[ints_idx + 2] = l2;
        tmp_idx += 5;
        ints_idx += 3;
    }
    Ok(())
}

fn decode6<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(r, ints, tmp, 48, 2, 6, mask8_const!(6), 0, mask8_const!(2))?;
    let mut tmp_idx = 0;
    let mut ints_idx = 48;
    for _ in 0..16 {
        let l0 = (tmp[tmp_idx] << 4) | (tmp[tmp_idx + 1] << 2) | tmp[tmp_idx + 2];
        ints[ints_idx] = l0;
        tmp_idx += 3;
        ints_idx += 1;
    }
    Ok(())
}

fn decode7<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(r, ints, tmp, 56, 1, 7, mask8_const!(7), 0, mask8_const!(1))?;
    let mut tmp_idx = 0;
    let mut ints_idx = 56;
    for _ in 0..8 {
        let mut l0 = tmp[tmp_idx] << 6;
        l0 |= tmp[tmp_idx + 1] << 5;
        l0 |= tmp[tmp_idx + 2] << 4;
        l0 |= tmp[tmp_idx + 3] << 3;
        l0 |= tmp[tmp_idx + 4] << 2;
        l0 |= tmp[tmp_idx + 5] << 1;
        l0 |= tmp[tmp_idx + 6];
        ints[ints_idx] = l0;
        tmp_idx += 7;
        ints_idx += 1;
    }
    Ok(())
}

fn decode8<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    for slot in ints[0..64].iter_mut() {
        *slot = r.read_u32_le()?;
    }
    Ok(())
}

fn decode9<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(
        r,
        ints,
        tmp,
        72,
        7,
        9,
        mask16_const!(9),
        0,
        mask16_const!(7),
    )?;
    let (m1, m2, m3, m4, m5, m6, m8) = (
        mask16_const!(1),
        mask16_const!(2),
        mask16_const!(3),
        mask16_const!(4),
        mask16_const!(5),
        mask16_const!(6),
        mask16_const!(8),
    );
    let mut t = 0;
    let mut ii = 72;
    for _ in 0..8 {
        let mut l0 = tmp[t] << 2;
        l0 |= (tmp[t + 1] >> 5) & m2;
        ints[ii] = l0;
        let mut l1 = (tmp[t + 1] & m5) << 4;
        l1 |= (tmp[t + 2] >> 3) & m4;
        ints[ii + 1] = l1;
        let mut l2 = (tmp[t + 2] & m3) << 6;
        l2 |= (tmp[t + 3] >> 1) & m6;
        ints[ii + 2] = l2;
        let mut l3 = (tmp[t + 3] & m1) << 8;
        l3 |= tmp[t + 4] << 1;
        l3 |= (tmp[t + 5] >> 6) & m1;
        ints[ii + 3] = l3;
        let mut l4 = (tmp[t + 5] & m6) << 3;
        l4 |= (tmp[t + 6] >> 4) & m3;
        ints[ii + 4] = l4;
        let mut l5 = (tmp[t + 6] & m4) << 5;
        l5 |= (tmp[t + 7] >> 2) & m5;
        ints[ii + 5] = l5;
        let mut l6 = (tmp[t + 7] & m2) << 7;
        l6 |= tmp[t + 8];
        ints[ii + 6] = l6;
        let _ = m8;
        t += 9;
        ii += 7;
    }
    Ok(())
}

fn decode10<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(
        r,
        ints,
        tmp,
        80,
        6,
        10,
        mask16_const!(10),
        0,
        mask16_const!(6),
    )?;
    let (m2, m4) = (mask16_const!(2), mask16_const!(4));
    let mut t = 0;
    let mut ii = 80;
    for _ in 0..16 {
        let mut l0 = tmp[t] << 4;
        l0 |= (tmp[t + 1] >> 2) & m4;
        ints[ii] = l0;
        let mut l1 = (tmp[t + 1] & m2) << 8;
        l1 |= tmp[t + 2] << 2;
        l1 |= (tmp[t + 3] >> 4) & m2;
        ints[ii + 1] = l1;
        let mut l2 = (tmp[t + 3] & m4) << 6;
        l2 |= tmp[t + 4];
        ints[ii + 2] = l2;
        t += 5;
        ii += 3;
    }
    Ok(())
}

fn decode11<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(
        r,
        ints,
        tmp,
        88,
        5,
        11,
        mask16_const!(11),
        0,
        mask16_const!(5),
    )?;
    let (m1, m2, m3, m4) = (
        mask16_const!(1),
        mask16_const!(2),
        mask16_const!(3),
        mask16_const!(4),
    );
    let mut t = 0;
    let mut ii = 88;
    for _ in 0..8 {
        let mut l0 = tmp[t] << 6;
        l0 |= tmp[t + 1] << 1;
        l0 |= (tmp[t + 2] >> 4) & m1;
        ints[ii] = l0;
        let mut l1 = (tmp[t + 2] & m4) << 7;
        l1 |= tmp[t + 3] << 2;
        l1 |= (tmp[t + 4] >> 3) & m2;
        ints[ii + 1] = l1;
        let mut l2 = (tmp[t + 4] & m3) << 8;
        l2 |= tmp[t + 5] << 3;
        l2 |= (tmp[t + 6] >> 2) & m3;
        ints[ii + 2] = l2;
        let mut l3 = (tmp[t + 6] & m2) << 9;
        l3 |= tmp[t + 7] << 4;
        l3 |= (tmp[t + 8] >> 1) & m4;
        ints[ii + 3] = l3;
        let mut l4 = (tmp[t + 8] & m1) << 10;
        l4 |= tmp[t + 9] << 5;
        l4 |= tmp[t + 10];
        ints[ii + 4] = l4;
        t += 11;
        ii += 5;
    }
    Ok(())
}

fn decode12<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(
        r,
        ints,
        tmp,
        96,
        4,
        12,
        mask16_const!(12),
        0,
        mask16_const!(4),
    )?;
    let mut t = 0;
    let mut ii = 96;
    for _ in 0..32 {
        let l0 = (tmp[t] << 8) | (tmp[t + 1] << 4) | tmp[t + 2];
        ints[ii] = l0;
        t += 3;
        ii += 1;
    }
    Ok(())
}

fn decode13<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(
        r,
        ints,
        tmp,
        104,
        3,
        13,
        mask16_const!(13),
        0,
        mask16_const!(3),
    )?;
    let (m1, m2) = (mask16_const!(1), mask16_const!(2));
    let mut t = 0;
    let mut ii = 104;
    for _ in 0..8 {
        let mut l0 = tmp[t] << 10;
        l0 |= tmp[t + 1] << 7;
        l0 |= tmp[t + 2] << 4;
        l0 |= tmp[t + 3] << 1;
        l0 |= (tmp[t + 4] >> 2) & m1;
        ints[ii] = l0;
        let mut l1 = (tmp[t + 4] & m2) << 11;
        l1 |= tmp[t + 5] << 8;
        l1 |= tmp[t + 6] << 5;
        l1 |= tmp[t + 7] << 2;
        l1 |= (tmp[t + 8] >> 1) & m2;
        ints[ii + 1] = l1;
        let mut l2 = (tmp[t + 8] & m1) << 12;
        l2 |= tmp[t + 9] << 9;
        l2 |= tmp[t + 10] << 6;
        l2 |= tmp[t + 11] << 3;
        l2 |= tmp[t + 12];
        ints[ii + 2] = l2;
        t += 13;
        ii += 3;
    }
    Ok(())
}

fn decode14<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(
        r,
        ints,
        tmp,
        112,
        2,
        14,
        mask16_const!(14),
        0,
        mask16_const!(2),
    )?;
    let mut t = 0;
    let mut ii = 112;
    for _ in 0..16 {
        let mut l0 = tmp[t] << 12;
        l0 |= tmp[t + 1] << 10;
        l0 |= tmp[t + 2] << 8;
        l0 |= tmp[t + 3] << 6;
        l0 |= tmp[t + 4] << 4;
        l0 |= tmp[t + 5] << 2;
        l0 |= tmp[t + 6];
        ints[ii] = l0;
        t += 7;
        ii += 1;
    }
    Ok(())
}

fn decode15<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(
        r,
        ints,
        tmp,
        120,
        1,
        15,
        mask16_const!(15),
        0,
        mask16_const!(1),
    )?;
    let mut t = 0;
    let mut ii = 120;
    for _ in 0..8 {
        let mut l0 = tmp[t] << 14;
        l0 |= tmp[t + 1] << 13;
        l0 |= tmp[t + 2] << 12;
        l0 |= tmp[t + 3] << 11;
        l0 |= tmp[t + 4] << 10;
        l0 |= tmp[t + 5] << 9;
        l0 |= tmp[t + 6] << 8;
        l0 |= tmp[t + 7] << 7;
        l0 |= tmp[t + 8] << 6;
        l0 |= tmp[t + 9] << 5;
        l0 |= tmp[t + 10] << 4;
        l0 |= tmp[t + 11] << 3;
        l0 |= tmp[t + 12] << 2;
        l0 |= tmp[t + 13] << 1;
        l0 |= tmp[t + 14];
        ints[ii] = l0;
        t += 15;
        ii += 1;
    }
    Ok(())
}

fn decode16<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    for slot in ints[0..128].iter_mut() {
        *slot = r.read_u32_le()?;
    }
    Ok(())
}

/// `ForUtil.decode`: decode 256 packed integers of `bits_per_value` bits each
/// (`1..=32`) from `r` into `ints`.
pub fn for_decode<R: DataInput>(
    bits_per_value: u32,
    r: &mut R,
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    let mut tmp = [0u32; BLOCK_SIZE];
    match bits_per_value {
        1 => {
            decode1(r, ints)?;
            expand8(ints);
        }
        2 => {
            decode2(r, ints)?;
            expand8(ints);
        }
        3 => {
            decode3(r, &mut tmp, ints)?;
            expand8(ints);
        }
        4 => {
            decode4(r, ints)?;
            expand8(ints);
        }
        5 => {
            decode5(r, &mut tmp, ints)?;
            expand8(ints);
        }
        6 => {
            decode6(r, &mut tmp, ints)?;
            expand8(ints);
        }
        7 => {
            decode7(r, &mut tmp, ints)?;
            expand8(ints);
        }
        8 => {
            decode8(r, ints)?;
            expand8(ints);
        }
        9 => {
            decode9(r, &mut tmp, ints)?;
            expand16(ints);
        }
        10 => {
            decode10(r, &mut tmp, ints)?;
            expand16(ints);
        }
        11 => {
            decode11(r, &mut tmp, ints)?;
            expand16(ints);
        }
        12 => {
            decode12(r, &mut tmp, ints)?;
            expand16(ints);
        }
        13 => {
            decode13(r, &mut tmp, ints)?;
            expand16(ints);
        }
        14 => {
            decode14(r, &mut tmp, ints)?;
            expand16(ints);
        }
        15 => {
            decode15(r, &mut tmp, ints)?;
            expand16(ints);
        }
        16 => {
            decode16(r, ints)?;
            expand16(ints);
        }
        _ => decode_slow(bits_per_value, r, ints)?,
    }
    Ok(())
}

/// `numBytes(bitsPerValue)`: number of bytes a `for_decode` call at this
/// `bits_per_value` consumes from `r`. Not called by the sequential-decode
/// path yet (it never skips a block without decoding it, see
/// `postings.rs`'s module doc), but is the building block a future
/// skip-ahead (`advance()`) implementation needs to jump over an
/// undecoded block — kept alongside `for_decode`/`pfor_decode` rather than
/// re-derived later, and exercised directly by this module's own tests.
#[allow(dead_code)]
pub fn num_bytes(bits_per_value: u32) -> usize {
    (bits_per_value as usize) << 5
}

/// `PForUtil.decode`: decode 256 patched-FOR-encoded integers (a 1-byte
/// token, an optional [`for_decode`] body, then `numExceptions` `(index,
/// high-byte)` patches applied as `ints[index] |= patch << bits_per_value`).
pub fn pfor_decode<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    let token = r.read_byte()? as u32;
    let bits_per_value = token & 0x1f;
    if bits_per_value == 0 {
        let v = r.read_vint()? as u32;
        ints.fill(v);
    } else {
        for_decode(bits_per_value, r, ints)?;
    }
    let num_exceptions = (token >> 5) as usize;
    debug_assert!(num_exceptions <= MAX_EXCEPTIONS);
    for _ in 0..num_exceptions {
        let idx = r.read_byte()? as usize;
        let patch = r.read_byte()? as u32;
        ints[idx] |= patch << bits_per_value;
    }
    Ok(())
}

/// Test-only encoders mirroring `ForUtil.collapse8`/`collapse16` and the
/// generic body of `ForUtil.encode` (parameterized by `primitiveSize`), used
/// to independently round-trip `decode1..decode16` (the lane-interleaved
/// `bits_per_value <= 16` paths) without a JVM, and reused by `postings.rs`'s
/// own unit tests to hand-construct a `for_decode`-compatible packed
/// doc-delta block — the differential fixture (real `IndexWriter` bytes) is
/// still the primary ground truth per the `differential-testing` skill, this
/// covers bit-width boundaries a single fixture term won't happen to hit.
#[cfg(test)]
pub(crate) mod test_support {
    use super::*;

    fn collapse8(arr: &mut [u32; BLOCK_SIZE]) {
        for i in 0..64 {
            arr[i] = (arr[i] << 24) | (arr[64 + i] << 16) | (arr[128 + i] << 8) | arr[192 + i];
        }
    }

    fn collapse16(arr: &mut [u32; BLOCK_SIZE]) {
        for i in 0..128 {
            arr[i] = (arr[i] << 16) | arr[128 + i];
        }
    }

    fn mask_for(bits: u32, primitive_size: u32) -> u32 {
        match primitive_size {
            8 => mask8(bits),
            16 => mask16(bits),
            _ => mask32(bits),
        }
    }

    fn encode_generic(ints: &[u32], bits_per_value: u32, primitive_size: u32, out: &mut Vec<u8>) {
        let num_ints = (BLOCK_SIZE * primitive_size as usize) / 32;
        let num_ints_per_shift = (bits_per_value * 8) as usize;
        let mut tmp = vec![0u32; BLOCK_SIZE];
        let mut idx = 0usize;
        let mut shift: i32 = primitive_size as i32 - bits_per_value as i32;
        for slot in tmp.iter_mut().take(num_ints_per_shift) {
            *slot = ints[idx] << shift;
            idx += 1;
        }
        shift -= bits_per_value as i32;
        while shift >= 0 {
            for slot in tmp.iter_mut().take(num_ints_per_shift) {
                *slot |= ints[idx] << shift;
                idx += 1;
            }
            shift -= bits_per_value as i32;
        }

        let remaining_bits_per_int = (shift + bits_per_value as i32) as u32;
        let mask_remaining = mask_for(remaining_bits_per_int, primitive_size);
        let mut tmp_idx = 0usize;
        let mut remaining_bits_per_value = bits_per_value;
        while idx < num_ints {
            if remaining_bits_per_value >= remaining_bits_per_int {
                remaining_bits_per_value -= remaining_bits_per_int;
                tmp[tmp_idx] |= (ints[idx] >> remaining_bits_per_value) & mask_remaining;
                if remaining_bits_per_value == 0 {
                    idx += 1;
                    remaining_bits_per_value = bits_per_value;
                }
                tmp_idx += 1;
            } else {
                let mask1 = mask_for(remaining_bits_per_value, primitive_size);
                let mask2 = mask_for(
                    remaining_bits_per_int - remaining_bits_per_value,
                    primitive_size,
                );
                tmp[tmp_idx] |=
                    (ints[idx] & mask1) << (remaining_bits_per_int - remaining_bits_per_value);
                idx += 1;
                remaining_bits_per_value += bits_per_value - remaining_bits_per_int;
                tmp[tmp_idx] |= (ints[idx] >> remaining_bits_per_value) & mask2;
                tmp_idx += 1;
            }
        }

        for &w in tmp.iter().take(num_ints_per_shift) {
            out.extend_from_slice(&w.to_le_bytes());
        }
    }

    pub(crate) fn encode_block(values: &[u32; BLOCK_SIZE], bits_per_value: u32) -> Vec<u8> {
        let primitive_size = if bits_per_value <= 8 {
            8
        } else if bits_per_value <= 16 {
            16
        } else {
            32
        };
        let mut arr = *values;
        if primitive_size == 8 {
            collapse8(&mut arr);
        } else if primitive_size == 16 {
            collapse16(&mut arr);
        }
        let mut out = Vec::new();
        encode_generic(&arr, bits_per_value, primitive_size, &mut out);
        out
    }
}

#[cfg(test)]
mod tests {
    use super::test_support::encode_block;
    use super::*;
    use lucene_store::data_input::SliceInput;
    use lucene_store::data_output::DataOutput;

    #[test]
    fn for_decode_roundtrips_bits_1_to_16() {
        for bits in 1u32..=16 {
            let mut values = [0u32; BLOCK_SIZE];
            for (i, v) in values.iter_mut().enumerate() {
                *v = ((i as u32).wrapping_mul(2654435761) ^ (i as u32).rotate_left(3))
                    & mask32(bits);
            }
            let bytes = encode_block(&values, bits);
            assert_eq!(bytes.len(), num_bytes(bits), "bits_per_value={bits}");

            let mut r = SliceInput::new(&bytes);
            let mut decoded = [0u32; BLOCK_SIZE];
            for_decode(bits, &mut r, &mut decoded).unwrap();
            assert_eq!(decoded, values, "bits_per_value={bits}");
        }
    }

    #[test]
    fn decode_slow_roundtrips_bits_17_to_32() {
        for bits in [17u32, 20, 24, 28, 31, 32] {
            let mut values = [0u32; BLOCK_SIZE];
            for (i, v) in values.iter_mut().enumerate() {
                // Deterministic pseudo-random-ish pattern within [0, 2^bits).
                *v = ((i as u32).wrapping_mul(2654435761) ^ (i as u32)) & mask32(bits);
            }
            let bytes = encode_block(&values, bits);
            assert_eq!(bytes.len(), num_bytes(bits));

            let mut r = SliceInput::new(&bytes);
            let mut decoded = [0u32; BLOCK_SIZE];
            for_decode(bits, &mut r, &mut decoded).unwrap();
            assert_eq!(decoded, values, "bits_per_value={bits}");
        }
    }

    #[test]
    fn mask32_boundary_values() {
        assert_eq!(mask32(0), 0);
        assert_eq!(mask32(1), 1);
        assert_eq!(mask32(31), (1u32 << 31) - 1);
        assert_eq!(mask32(32), u32::MAX);
    }

    #[test]
    fn pfor_decode_all_equal_uses_vint_fast_path() {
        // token byte with bitsPerValue=0 and numExceptions=0, then a plain
        // vint carrying the single repeated value.
        let mut bytes = vec![0u8];
        bytes.write_vint(42);
        let mut r = SliceInput::new(&bytes);
        let mut ints = [0u32; BLOCK_SIZE];
        pfor_decode(&mut r, &mut ints).unwrap();
        assert!(ints.iter().all(|&v| v == 42));
    }

    #[test]
    fn pfor_decode_all_equal_with_exceptions() {
        // bitsPerValue=0 (all-equal base value), but 2 exceptions patch
        // specific slots to larger values via high bytes shifted by 0 bits.
        let num_exceptions = 2u8;
        let mut bytes = vec![num_exceptions << 5];
        bytes.write_vint(5);
        bytes.push(10); // exception at index 10
        bytes.push(3); // patch byte: ints[10] |= 3 << 0
        bytes.push(200); // exception at index 200
        bytes.push(1); // ints[200] |= 1 << 0
        let mut r = SliceInput::new(&bytes);
        let mut ints = [0u32; BLOCK_SIZE];
        pfor_decode(&mut r, &mut ints).unwrap();
        for (i, &v) in ints.iter().enumerate() {
            match i {
                10 => assert_eq!(v, 5 | 3),
                200 => assert_eq!(v, 5 | 1),
                _ => assert_eq!(v, 5),
            }
        }
    }

    #[test]
    fn for_decode_bits_per_value_one_all_zero() {
        // 8 zero words -> 64 zero collapsed ints -> expand8 -> all 256 zero.
        let bytes = vec![0u8; num_bytes(1)];
        let mut r = SliceInput::new(&bytes);
        let mut ints = [0u32; BLOCK_SIZE];
        for_decode(1, &mut r, &mut ints).unwrap();
        assert!(ints.iter().all(|&v| v == 0));
    }

    #[test]
    fn num_bytes_matches_bit_width() {
        assert_eq!(num_bytes(1), 32);
        assert_eq!(num_bytes(8), 256);
        assert_eq!(num_bytes(16), 512);
        assert_eq!(num_bytes(32), 1024);
    }
}
