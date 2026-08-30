//! Port of `org.apache.lucene.codecs.lucene104.ForUtil`/`PForUtil`: the
//! bit-packed "patched frame-of-reference" (PFOR) bulk (de)coder used for a
//! full 256-value block (`ForUtil.BLOCK_SIZE`). This is a direct
//! transliteration of `ForUtil.decode1..decode16`/`decodeSlow`/`encode` and
//! `PForUtil.decode`/`encode`, not a redesign, because the on-disk layout
//! (which values land in which bits of which 32-bit little-endian word) is
//! the compatibility contract; a faster/SIMD re-expression of the same
//! bit-layout is future work (see `docs/parity.md`).
//!
//! ## Encode side scope
//!
//! [`for_encode`]/[`pfor_encode`] are the production encoders (promoted from
//! a `#[cfg(test)]`-only helper that used to exist solely to exercise
//! [`for_decode`]'s round trip). [`pfor_encode`] is a full port of
//! `PForUtil.encode`, including the patched-exception selection loop
//! (histogram of required bit widths, prefer the smallest `bitsPerValue` in
//! `[maxBitsRequired - 8, maxBitsRequired]` that keeps the exception count at
//! or under `MAX_EXCEPTIONS = 7`, plus the `bitsPerValue == 0`/all-equal
//! fast path) — this is genuine `PForUtil`, not `ForUtil`-only with the label
//! borrowed; see `docs/parity.md` for exactly what still isn't wired to a
//! writer. **`crate::postings_writer` now calls [`for_encode`] (doc deltas,
//! plain FOR, never the `bitsPerValue == 0`/dense-bitset alternate shapes)
//! and [`pfor_encode`] (freqs)** for the `docFreq >= BLOCK_SIZE` full-block
//! case — see that module's `write_full_block`. `.pos` full blocks
//! (`total_term_freq >= BLOCK_SIZE`) still aren't wired to a writer; that's
//! out of scope here and tracked in `docs/parity.md` rather than silently
//! left unimplemented.
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

// `clippy::cast_sign_loss` is off workspace-wide (1 036 sites, most of them
// deliberate widenings -- see `docs/arithmetic-gate.md`'s "Lints considered
// and not adopted"), but this module is small enough to carry it: it fires on
// exactly three casts here, each of which has a written justification below.
// It is the lint that catches the "negative value sign-extended into a wider
// unsigned type" shape, which is what turned a `-1` into a `usize::MAX`
// length in batches b1 and c15 -- so keeping it on locks this decode kernel
// against a future `i32 as usize` slipping in unremarked.
#![deny(clippy::cast_sign_loss)]

use lucene_store::data_input::DataInput;
use lucene_store::data_output::DataOutput;
use lucene_store::Result;

/// `ForUtil.BLOCK_SIZE` / `PForUtil`'s implicit block width.
pub const BLOCK_SIZE: usize = 256;

/// `PForUtil`'s `static { assert ForUtil.BLOCK_SIZE <= 256 : "blocksize must
/// fit in one byte" }`, ported as a compile-time check and tightened to an
/// equality because both directions carry weight here:
///
/// * `<= 256` is what makes [`pfor_encode`]'s `i as u8` exception index
///   lossless, exactly as in Java;
/// * `>= 256` is what makes [`ForUtil::pfor_decode`]'s `ints[idx]` — the one
///   and only index in this module taken from a byte read off disk — in
///   bounds for *every* one of the 256 values that byte can hold, with no
///   runtime check needed on the hot patch loop.
const _: () = assert!(
    BLOCK_SIZE == 256,
    "PForUtil's exception index is a single byte: BLOCK_SIZE must be exactly 256"
);

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
        // ARITH: the two arms above have already returned for `bits == 0` and
        // `bits >= 32`, so here `bits` is in `1..=31`: `1u32 << bits` is a
        // shift by less than 32 (no shift overflow) yielding at least 2, so
        // the `- 1` cannot underflow.
        #[allow(clippy::arithmetic_side_effects)]
        {
            (1u32 << bits) - 1
        }
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
// ARITH: a fixed 64-iteration loop over this port's own scratch array.
// `i` is in `0..64`, so `64 + i`, `128 + i` and `192 + i` are at most 127,
// 191 and 255 -- every one of them a valid index into a `[u32; BLOCK_SIZE]`
// with `BLOCK_SIZE == 256`, and far from overflowing a `usize`. Nothing
// here comes off disk. Written with explicit lane offsets rather than four
// `split_at_mut` slices because the slice form loses the array's static
// length and measured 9-28% slower across `decode1`..`decode13` (see
// `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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
// ARITH: a fixed 128-iteration loop over this port's own scratch array;
// `i` is in `0..128`, so `128 + i` is at most 255, a valid index into a
// `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`. See `expand8` for why this
// keeps explicit lane offsets rather than `split_at_mut`.
#[allow(clippy::arithmetic_side_effects)]
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
// ARITH: this is the per-block scaffolding of the hot decode kernel (the
// per-value inner loop below contains no arithmetic at all), and every
// operand is a decode-shape constant rather than a value off disk:
//
//   * `count`, `dec`, `b_mask`, `c_mask` and `c_index` are literals at all 16
//     `decodeN` call sites, and `c_index` is `0` at every one of them;
//   * `b_shift` is a literal in `1..=7` at those call sites, and in
//     `decode_slow` it is `32 - bits_per_value` for a `bits_per_value` that
//     `ForUtil::decode` has already rejected outside `1..=32`, so
//     `b_shift <= 31` there too.
//
// From `dec >= 1` and `b_shift <= 31`: `b_shift as i32 - 1` is in `-1..=30`
// (no `i32` overflow), the division cannot trap, and
// `max_iter = (b_shift - 1) / dec <= 30`. By the definition of integer
// division `max_iter * dec <= b_shift - 1 < b_shift` whenever `b_shift >= 1`,
// and `max_iter == 0` when `b_shift == 0`, so for every `j <= max_iter` the
// product `j * dec` is at most `b_shift <= 31`: `b_shift - j * dec` never
// wraps and always stays a legal `u32` shift. `count * (max_iter + 1)` and
// `c_index + count` are bounded by `b.len()`/`c.len()` (both at most 256) --
// the four `debug_assert!`s pin exactly these four facts, so `cargo test`
// re-checks them for every width on every block it decodes.
#[allow(clippy::too_many_arguments, clippy::arithmetic_side_effects)]
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
    debug_assert!(dec >= 1, "split_ints dec must be positive, got {dec}");
    debug_assert!(b_shift <= 31, "split_ints b_shift out of range: {b_shift}");
    // Written with `checked_add`, not `c_index + count <= c.len()`: a bound
    // of the latter shape is self-defeating, because the very overflow it is
    // meant to exclude would wrap the sum down into range and pass the check.
    debug_assert!(
        c_index.checked_add(count).is_some_and(|end| end <= c.len()),
        "split_ints c range out of range"
    );
    r.read_u32s_le(&mut c[c_index..c_index + count])?;
    // Java: `(bShift - 1) / dec` using signed int division truncating toward
    // zero; `bShift == 0` (only reachable for `bits_per_value == 32` via
    // `decode_slow`) still yields `maxIter == 0` (one iteration at shift 0),
    // matching `(-1)/dec == 0` in Java — hence the signed intermediate here.
    // The `i32` intermediate is Java's own signed division (see the comment
    // above); the quotient is 0 at `b_shift == 0` and non-negative for every
    // other `b_shift`, so the cast back to `u32` cannot lose a sign bit.
    #[allow(clippy::cast_sign_loss)]
    let max_iter = ((b_shift as i32 - 1) / dec as i32) as u32;
    debug_assert!(
        count
            .checked_mul(max_iter as usize + 1)
            .is_some_and(|end| end <= b.len()),
        "split_ints b range out of range"
    );
    // Shift level outer, element inner -- Lucene's order, and its own comment
    // says why: "Process each shift level across all elements (better for
    // vectorization)". The transposed nest this port had first gives the inner
    // loop a variable trip count and a stride-`count` write, which is precisely
    // what stops a vectorizer. Slice-and-zip rather than indexing so the bounds
    // checks hoist out of the loop instead of repeating per element.
    let src = &c[c_index..c_index + count];
    for j in 0..=max_iter {
        let shift = b_shift - j * dec;
        let b_offset = count * j as usize;
        for (dst, &cv) in b[b_offset..b_offset + count].iter_mut().zip(src) {
            *dst = (cv >> shift) & b_mask;
        }
    }
    for v in c[c_index..c_index + count].iter_mut() {
        *v &= c_mask;
    }
    Ok(())
}

/// `ForUtil.decodeSlow`: the `bitsPerValue > 16` fallback (plain 32-bit-wide
/// packing, no lane interleaving).
///
/// # Panics (debug only)
///
/// The single caller is [`ForUtil::decode`]'s `_` match arm, reached only
/// after it has rejected `bits_per_value` outside `1..=32` and matched away
/// `1..=16`; the `debug_assert!` below pins that.
// ARITH: `bits_per_value` is in `17..=32` (see above), which bounds every
// operation here:
//   * `bits_per_value * 8 <= 256`, so `num_ints` fits and indexes `ints`;
//   * `32 - bits_per_value` is in `0..=15`, so `remaining_bits_per_int`
//     cannot underflow;
//   * `remaining_bits` is `remaining_bits_per_int` on entry and
//     `remaining_bits_per_int - b` for a `b` the enclosing `if` has already
//     shown is strictly below `remaining_bits_per_int`, so it is always in
//     `1..=15` whenever the loop body runs at all (for `bits_per_value == 32`
//     the loop body is unreachable: `num_ints == BLOCK_SIZE`). Hence
//     `b = bits_per_value - remaining_bits` is in `2..=30` (smallest at
//     `17 - 15`, largest at `31 - 1`): the subtraction cannot underflow and
//     `<< b` is a legal `u32` shift, comfortably short of the 32 that would
//     panic;
//   * `b -= remaining_bits_per_int` runs only under `b >= remaining_bits_per_int`
//     and `remaining_bits_per_int - b` only under `b < remaining_bits_per_int`;
//   * `tmp_idx` advances once per `remaining_bits_per_int` bits consumed and
//     the block's bit budget balances exactly --
//     `num_ints * (32 - bits_per_value) == (BLOCK_SIZE - num_ints) * bits_per_value`
//     -- so it ends at exactly `num_ints <= 256` (or at 0 when
//     `bits_per_value == 32`, where the loop does not run at all);
//     `debug_assert_eq!` at the bottom pins that, because overrunning it
//     would silently decode a value out of stale scratch rather than fail.
#[allow(clippy::arithmetic_side_effects)]
fn decode_slow<R: DataInput>(
    bits_per_value: u32,
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    debug_assert!(
        (17..=32).contains(&bits_per_value),
        "decode_slow bitsPerValue out of range: {bits_per_value}"
    );
    let num_ints = (bits_per_value as usize) * 8;
    let mask = mask32(bits_per_value);
    {
        let (b, _) = ints.split_at_mut(num_ints.max(1));
        split_ints(
            r,
            b,
            tmp,
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
    debug_assert_eq!(
        tmp_idx,
        // At `bits_per_value == 32` the packed body already covers all 256
        // values, the loop above never runs and no scratch word is touched.
        if num_ints == BLOCK_SIZE { 0 } else { num_ints },
        "decode_slow consumed the wrong number of scratch words"
    );
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

// ARITH: `decode3`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 8 times, so:
//   * the scratch cursor takes 0, 3, ..., 21 and reads at most `tmp[23]`,
//     finishing at 24 after the last `+= 3`;
//   * the output cursor takes 48, 50, ..., 62 and writes at
//     most `ints[63]`, finishing at 64 after the last `+= 2`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 64,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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

// ARITH: `decode5`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 8 times, so:
//   * the scratch cursor takes 0, 5, ..., 35 and reads at most `tmp[39]`,
//     finishing at 40 after the last `+= 5`;
//   * the output cursor takes 40, 43, ..., 61 and writes at
//     most `ints[63]`, finishing at 64 after the last `+= 3`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 64,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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

// ARITH: `decode6`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 16 times, so:
//   * the scratch cursor takes 0, 3, ..., 45 and reads at most `tmp[47]`,
//     finishing at 48 after the last `+= 3`;
//   * the output index is the bounded range `(48..).take(16)`, i.e.
//     48..=63, so every write lands in `ints`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 63,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
fn decode6<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(r, ints, tmp, 48, 2, 6, mask8_const!(6), 0, mask8_const!(2))?;
    let mut tmp_idx = 0;
    for ints_idx in (48..).take(16) {
        let l0 = (tmp[tmp_idx] << 4) | (tmp[tmp_idx + 1] << 2) | tmp[tmp_idx + 2];
        ints[ints_idx] = l0;
        tmp_idx += 3;
    }
    Ok(())
}

// ARITH: `decode7`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 8 times, so:
//   * the scratch cursor takes 0, 7, ..., 49 and reads at most `tmp[55]`,
//     finishing at 56 after the last `+= 7`;
//   * the output index is the bounded range `(56..).take(8)`, i.e.
//     56..=63, so every write lands in `ints`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 63,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
fn decode7<R: DataInput>(
    r: &mut R,
    tmp: &mut [u32; BLOCK_SIZE],
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    split_ints(r, ints, tmp, 56, 1, 7, mask8_const!(7), 0, mask8_const!(1))?;
    let mut tmp_idx = 0;
    for ints_idx in (56..).take(8) {
        let mut l0 = tmp[tmp_idx] << 6;
        l0 |= tmp[tmp_idx + 1] << 5;
        l0 |= tmp[tmp_idx + 2] << 4;
        l0 |= tmp[tmp_idx + 3] << 3;
        l0 |= tmp[tmp_idx + 4] << 2;
        l0 |= tmp[tmp_idx + 5] << 1;
        l0 |= tmp[tmp_idx + 6];
        ints[ints_idx] = l0;
        tmp_idx += 7;
    }
    Ok(())
}

fn decode8<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    r.read_u32s_le(&mut ints[0..64])
}

// ARITH: `decode9`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 8 times, so:
//   * the scratch cursor takes 0, 9, ..., 63 and reads at most `tmp[71]`,
//     finishing at 72 after the last `+= 9`;
//   * the output cursor takes 72, 79, ..., 121 and writes at
//     most `ints[127]`, finishing at 128 after the last `+= 7`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 128,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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
    let (m1, m2, m3, m4, m5, m6) = (
        mask16_const!(1),
        mask16_const!(2),
        mask16_const!(3),
        mask16_const!(4),
        mask16_const!(5),
        mask16_const!(6),
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
        t += 9;
        ii += 7;
    }
    Ok(())
}

// ARITH: `decode10`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 16 times, so:
//   * the scratch cursor takes 0, 5, ..., 75 and reads at most `tmp[79]`,
//     finishing at 80 after the last `+= 5`;
//   * the output cursor takes 80, 83, ..., 125 and writes at
//     most `ints[127]`, finishing at 128 after the last `+= 3`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 128,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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

// ARITH: `decode11`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 8 times, so:
//   * the scratch cursor takes 0, 11, ..., 77 and reads at most `tmp[87]`,
//     finishing at 88 after the last `+= 11`;
//   * the output cursor takes 88, 93, ..., 123 and writes at
//     most `ints[127]`, finishing at 128 after the last `+= 5`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 128,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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

// ARITH: `decode12`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 32 times, so:
//   * the scratch cursor takes 0, 3, ..., 93 and reads at most `tmp[95]`,
//     finishing at 96 after the last `+= 3`;
//   * the output index is the bounded range `(96..).take(32)`, i.e.
//     96..=127, so every write lands in `ints`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 127,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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
    for ii in (96..).take(32) {
        let l0 = (tmp[t] << 8) | (tmp[t + 1] << 4) | tmp[t + 2];
        ints[ii] = l0;
        t += 3;
    }
    Ok(())
}

// ARITH: `decode13`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 8 times, so:
//   * the scratch cursor takes 0, 13, ..., 91 and reads at most `tmp[103]`,
//     finishing at 104 after the last `+= 13`;
//   * the output cursor takes 104, 107, ..., 125 and writes at
//     most `ints[127]`, finishing at 128 after the last `+= 3`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 128,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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

// ARITH: `decode14`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 16 times, so:
//   * the scratch cursor takes 0, 7, ..., 105 and reads at most `tmp[111]`,
//     finishing at 112 after the last `+= 7`;
//   * the output index is the bounded range `(112..).take(16)`, i.e.
//     112..=127, so every write lands in `ints`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 127,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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
    for ii in (112..).take(16) {
        let mut l0 = tmp[t] << 12;
        l0 |= tmp[t + 1] << 10;
        l0 |= tmp[t + 2] << 8;
        l0 |= tmp[t + 3] << 6;
        l0 |= tmp[t + 4] << 4;
        l0 |= tmp[t + 5] << 2;
        l0 |= tmp[t + 6];
        ints[ii] = l0;
        t += 7;
    }
    Ok(())
}

// ARITH: `decode15`'s tail is a fixed unrolled loop over this port's own
// scratch arrays; not one operand comes off disk, and every shift distance
// is a literal under 32. The loop runs exactly 8 times, so:
//   * the scratch cursor takes 0, 15, ..., 105 and reads at most `tmp[119]`,
//     finishing at 120 after the last `+= 15`;
//   * the output index is the bounded range `(120..).take(8)`, i.e.
//     120..=127, so every write lands in `ints`.
// Both arrays are `[u32; BLOCK_SIZE]` with `BLOCK_SIZE == 256`, and the
// largest `usize` any of these additions produces is 127,
// so none can overflow. Kept as explicit indices rather than
// `chunks_exact`: the iterator form measured 10-48% slower on this decode
// kernel (see `bench_decode_and_encode_min_of_n`).
#[allow(clippy::arithmetic_side_effects)]
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
    for ii in (120..).take(8) {
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
    }
    Ok(())
}

fn decode16<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    r.read_u32s_le(&mut ints[0..128])
}

/// `ForUtil`, as an owner of the scratch buffer its decode paths need.
///
/// Lucene's `ForUtil` is an instance with `private final int[] tmp = new
/// int[BLOCK_SIZE]`, allocated once and reused for every block it ever
/// decodes. This port originally declared that buffer inside the decode
/// function, so every 256-value block paid a 1 KiB zero-fill, and
/// `decodeSlow` paid a second one -- roughly 9% of the measured cost at
/// `bits_per_value = 31`, invisible to a profile because it is spread across
/// every call. Holding it here restores Lucene's shape.
///
/// Callers that decode more than one block should keep one of these alive
/// across them; [`for_decode`]/[`pfor_decode`] remain for one-shot callers
/// (tests, the writer's round-trip checks) and construct one per call.
#[derive(Clone, Debug)]
pub struct ForUtil {
    tmp: [u32; BLOCK_SIZE],
}

impl Default for ForUtil {
    fn default() -> Self {
        Self::new()
    }
}

impl ForUtil {
    /// A decoder with a fresh scratch buffer.
    pub fn new() -> Self {
        Self {
            tmp: [0u32; BLOCK_SIZE],
        }
    }

    /// `ForUtil.decode`: decode 256 packed integers of `bits_per_value` bits
    /// each (`1..=32`) from `r` into `ints`, reusing this instance's scratch
    /// buffer.
    pub fn decode<R: DataInput>(
        &mut self,
        bits_per_value: u32,
        r: &mut R,
        ints: &mut [u32; BLOCK_SIZE],
    ) -> Result<()> {
        // Java has no explicit check here: `decodeSlow` indexes `MASKS32`
        // (`new int[32]`) with `bitsPerValue`, so anything outside `1..=31`
        // throws `ArrayIndexOutOfBoundsException` instead. This port's callers
        // take `bits_per_value` straight off disk -- `postings`' block header
        // is a signed byte, so a corrupt `.doc` can hand us up to 127 -- and
        // every path here returns `Result`, so a corrupt value has to be an
        // error, not a slice-index panic in `decode_slow`. `32` is accepted
        // (this port's `decode_slow`/`encode_generic` handle it; Java's
        // mask table stops one short), `0` is not: `pfor_decode` handles the
        // all-equal token before it ever gets here.
        if !(1..=32).contains(&bits_per_value) {
            return Err(lucene_store::Error::Corrupted(format!(
                "ForUtil bitsPerValue out of range: {bits_per_value}"
            )));
        }
        let tmp = &mut self.tmp;
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
                decode3(r, tmp, ints)?;
                expand8(ints);
            }
            4 => {
                decode4(r, ints)?;
                expand8(ints);
            }
            5 => {
                decode5(r, tmp, ints)?;
                expand8(ints);
            }
            6 => {
                decode6(r, tmp, ints)?;
                expand8(ints);
            }
            7 => {
                decode7(r, tmp, ints)?;
                expand8(ints);
            }
            8 => {
                decode8(r, ints)?;
                expand8(ints);
            }
            9 => {
                decode9(r, tmp, ints)?;
                expand16(ints);
            }
            10 => {
                decode10(r, tmp, ints)?;
                expand16(ints);
            }
            11 => {
                decode11(r, tmp, ints)?;
                expand16(ints);
            }
            12 => {
                decode12(r, tmp, ints)?;
                expand16(ints);
            }
            13 => {
                decode13(r, tmp, ints)?;
                expand16(ints);
            }
            14 => {
                decode14(r, tmp, ints)?;
                expand16(ints);
            }
            15 => {
                decode15(r, tmp, ints)?;
                expand16(ints);
            }
            16 => {
                decode16(r, ints)?;
                expand16(ints);
            }
            _ => decode_slow(bits_per_value, r, tmp, ints)?,
        }
        Ok(())
    }

    /// `PForUtil.decode`: decode 256 patched-FOR-encoded integers (a 1-byte
    /// token, an optional [`ForUtil::decode`] body, then `numExceptions`
    /// `(index, high-byte)` patches applied as `ints[index] |= patch <<
    /// bits_per_value`). Reuses this instance's scratch buffer.
    ///
    /// Every quantity this reads off disk is self-bounding, which is why
    /// there is no validation below beyond [`ForUtil::decode`]'s own width
    /// check: `bits_per_value` is `token & 0x1f`, i.e. `0..=31`, so
    /// `patch << bits_per_value` is always a legal `u32` shift;
    /// `num_exceptions` is `token >> 5`, i.e. `0..=7 == MAX_EXCEPTIONS`; and
    /// the patch index is a single byte, so it is `0..=255` and in bounds for
    /// `ints` by this module's `BLOCK_SIZE == 256` compile-time assertion.
    pub fn pfor_decode<R: DataInput>(
        &mut self,
        r: &mut R,
        ints: &mut [u32; BLOCK_SIZE],
    ) -> Result<()> {
        let token = r.read_byte()? as u32;
        let bits_per_value = token & 0x1f;
        if bits_per_value == 0 {
            // A corrupt vint can be negative, and the sign is deliberately
            // reinterpreted rather than rejected: Java is
            // `Arrays.fill(ints, 0, BLOCK_SIZE, in.readVInt())` into an
            // `int[]`, so it stores exactly these 32 bits too. This port
            // models Lucene's `int` as `u32` throughout the decode kernel, so
            // the cast is the identity on the bit pattern, not a widening --
            // it is the `i32 as usize` shape that this module's
            // `deny(cast_sign_loss)` exists to keep out, and there is none.
            #[allow(clippy::cast_sign_loss)]
            let v = r.read_vint()? as u32;
            ints.fill(v);
        } else {
            self.decode(bits_per_value, r, ints)?;
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

    /// `ForUtil.encode`: bit-pack 256 values, each already known to fit in
    /// `bits_per_value` bits (`1..=32`), and write them to `out`, reusing this
    /// instance's scratch buffer.
    ///
    /// `ints` is lane-collapsed **in place**, exactly like Java's
    /// `ForUtil.encode(int[] ints, int bitsPerValue, DataOutput out)`; the
    /// caller's array is scratch afterwards. The previous shape here took
    /// `&[u32; 256]` and copied, so every encoded block paid a 1 KiB memcpy on
    /// top of the 1 KiB zero-fill of a stack-local `tmp` -- work Java does
    /// neither of. Neither saving is separable in a microbenchmark: a
    /// repeatable bench of a destructive encoder has to restore its input each
    /// iteration, which reinstates exactly the copy that was removed. See
    /// `for_util/for_encode`'s `oneshot` vs `reused` arms in
    /// `benches/for_util_decode.rs` for the scratch-reuse half, and
    /// `docs/sweep/m2/b2-packed.md` for why the numbers there are inconclusive.
    ///
    /// # Panics
    ///
    /// If `bits_per_value` is outside `1..=32`. Java's `ForUtil.encode` has no
    /// such check and does not survive one either: at `bitsPerValue == 0` its
    /// `for (shift = shift - bitsPerValue; shift >= 0; shift -= bitsPerValue)`
    /// never advances and the encoder **spins forever**, and above 32 it walks
    /// off `MASKS32`. A hang is the one failure mode `catch_unwind` at the FFI
    /// boundary cannot turn back into a Java exception, so this port refuses
    /// the input instead — one comparison per 256-value block.
    pub fn encode<W: DataOutput>(
        &mut self,
        ints: &mut [u32; BLOCK_SIZE],
        bits_per_value: u32,
        out: &mut W,
    ) {
        assert!(
            (1..=32).contains(&bits_per_value),
            "ForUtil::encode bitsPerValue out of range: {bits_per_value}"
        );
        let primitive_size = if bits_per_value <= 8 {
            8
        } else if bits_per_value <= 16 {
            16
        } else {
            32
        };
        if primitive_size == 8 {
            collapse8(ints);
        } else if primitive_size == 16 {
            collapse16(ints);
        }
        encode_generic(ints, bits_per_value, primitive_size, out, &mut self.tmp);
    }
}

/// `numBytes(bitsPerValue)`: number of bytes a `for_decode` call at this
/// `bits_per_value` consumes from `r`. Not called by the sequential-decode
/// path yet (it never skips a block without decoding it, see
/// `postings.rs`'s module doc), but is the building block a future
/// skip-ahead (`advance()`) implementation needs to jump over an
/// undecoded block — kept alongside `for_decode`/`pfor_decode` rather than
/// re-derived later, and exercised directly by this module's own tests.
pub fn num_bytes(bits_per_value: u32) -> usize {
    (bits_per_value as usize) << 5
}

/// `PForUtil.skip`: consume exactly the bytes [`ForUtil::pfor_decode`] would
/// consume, without unpacking any of them.
///
/// This is what `Lucene104PostingsReader.refillFullBlock` calls in place of
/// `pforUtil.decode` when the consumer asked for `PostingsEnum.NONE`/`DOCS`
/// — one token byte, then a seek past the packed body and the exception
/// pairs. It reads the all-equal value with `read_vint` rather than Java's
/// `readVLong` for the same reason [`ForUtil::pfor_decode`] does: the two
/// agree byte-for-byte on every non-negative `i32`, and skip must consume
/// exactly what decode does — `pfor_skip_consumes_exactly_what_pfor_decode_does`
/// pins that.
pub fn pfor_skip<R: DataInput>(r: &mut R) -> Result<()> {
    let token = r.read_byte()? as u32;
    let bits_per_value = token & 0x1f;
    if bits_per_value == 0 {
        r.read_vint()?;
    } else {
        r.skip(num_bytes(bits_per_value))?;
    }
    // Each exception is an `(index, high-byte)` pair.
    r.skip(((token >> 5) as usize) << 1)?;
    Ok(())
}

/// One-shot [`ForUtil::pfor_decode`], for callers that decode a single block
/// and have no instance to reuse. A caller in a loop should hold a
/// [`ForUtil`], which is what makes its scratch buffer worth having.
pub fn pfor_decode<R: DataInput>(r: &mut R, ints: &mut [u32; BLOCK_SIZE]) -> Result<()> {
    ForUtil::new().pfor_decode(r, ints)
}

/// One-shot [`ForUtil::decode`], for callers that decode a single block and
/// have no instance to reuse. A caller in a loop should hold a [`ForUtil`].
pub fn for_decode<R: DataInput>(
    bits_per_value: u32,
    r: &mut R,
    ints: &mut [u32; BLOCK_SIZE],
) -> Result<()> {
    ForUtil::new().decode(bits_per_value, r, ints)
}

/// `ForUtil.collapse8`: interleave 4 consecutive values into one 32-bit int's
/// four byte lanes (the exact inverse of [`expand8`]).
// ARITH: the exact inverse of `expand8` and bounded identically -- `i` is
// in `0..64`, so the largest index formed is `192 + 63 == 255`.
#[allow(clippy::arithmetic_side_effects)]
fn collapse8(arr: &mut [u32; BLOCK_SIZE]) {
    for i in 0..64 {
        arr[i] = (arr[i] << 24) | (arr[64 + i] << 16) | (arr[128 + i] << 8) | arr[192 + i];
    }
}

/// `ForUtil.collapse16`: interleave 2 consecutive values into one 32-bit
/// int's two halfword lanes (the exact inverse of [`expand16`]).
// ARITH: the exact inverse of `expand16` and bounded identically -- `i` is
// in `0..128`, so the largest index formed is `128 + 127 == 255`.
#[allow(clippy::arithmetic_side_effects)]
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

/// `ForUtil.encode(int[], int, int, DataOutput, int[])`: the generic
/// bit-packing body shared by every `bits_per_value`, parameterized by
/// `primitive_size` (8/16 for the lane-interleaved `collapse8`/`collapse16`
/// paths, 32 for the `decodeSlow`-equivalent plain packing).
// ARITH: the sole caller is [`ForUtil::encode`], which asserts
// `bits_per_value` is in `1..=32` and then picks the smallest
// `primitive_size` in `{8, 16, 32}` that is at least `bits_per_value`. Both
// facts are re-pinned by the `debug_assert!`s below, and together they bound
// everything here:
//   * `BLOCK_SIZE * primitive_size <= 8192` and the divisor is the literal
//     32, so `num_ints = 8 * primitive_size` is in `{64, 128, 256}`;
//   * `bits_per_value * 8 <= 256`;
//   * `shift` starts at `primitive_size - bits_per_value >= 0` and each
//     `shift -= bits_per_value` is an `i32` bounded below by `-32`, so no
//     `i32` overflow and every `<< shift` executed under `shift >= 0` uses a
//     shift under 32;
//   * `idx` advances `8 * bits_per_value * floor(primitive_size /
//     bits_per_value) <= 8 * primitive_size == num_ints <= BLOCK_SIZE` times
//     across the two packing loops, so it never runs past `ints`;
//   * `remaining_bits_per_int = shift + bits_per_value` is in
//     `0..=bits_per_value - 1`; it is 0 exactly when `bits_per_value` divides
//     `primitive_size`, and in that case `idx == num_ints` already, so the
//     trailing `while` never runs with a zero stride (no hang, no `>> 32`);
//   * inside that `while`, `remaining_bits_per_value -= remaining_bits_per_int`
//     runs only under `>=`, `remaining_bits_per_int - remaining_bits_per_value`
//     only under `<`, and `bits_per_value - remaining_bits_per_int >= 1`
//     always; every shift distance stays strictly below `bits_per_value <= 31`
//     (`bits_per_value == 32` cannot reach this loop, per the point above).
#[allow(clippy::arithmetic_side_effects)]
fn encode_generic<W: DataOutput>(
    ints: &[u32],
    bits_per_value: u32,
    primitive_size: u32,
    out: &mut W,
    tmp: &mut [u32; BLOCK_SIZE],
) {
    debug_assert!(
        matches!(primitive_size, 8 | 16 | 32),
        "encode_generic primitiveSize must be 8/16/32, got {primitive_size}"
    );
    debug_assert!(
        (1..=primitive_size).contains(&bits_per_value),
        "encode_generic bitsPerValue {bits_per_value} out of range for primitiveSize {primitive_size}"
    );
    let num_ints = (BLOCK_SIZE * primitive_size as usize) / 32;
    let num_ints_per_shift = (bits_per_value * 8) as usize;
    // `tmp` arrives dirty and is not cleared: the first loop below *assigns*
    // (not ORs) every slot in `tmp[..num_ints_per_shift]`, and nothing past
    // that prefix is ever read. Java relies on the same property to reuse one
    // `ForUtil.tmp` field across every block it ever encodes.
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

    // The loop above exits with `shift` in `-bits_per_value..=-1`, so
    // `shift + bits_per_value` is in `0..=bits_per_value - 1`: non-negative,
    // and the cast cannot lose a sign bit.
    #[allow(clippy::cast_sign_loss)]
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
        out.write_bytes(&w.to_le_bytes());
    }
}

/// One-shot [`ForUtil::encode`], for callers that encode a single block and
/// have no instance to reuse. A caller in a loop should hold a [`ForUtil`],
/// which is what makes its scratch buffer worth having.
///
/// `values` is consumed destructively (lane-collapsed in place), exactly like
/// Java's `ForUtil.encode(int[] ints, ...)`.
pub fn for_encode<W: DataOutput>(values: &mut [u32; BLOCK_SIZE], bits_per_value: u32, out: &mut W) {
    ForUtil::new().encode(values, bits_per_value, out);
}

/// `PackedInts.bitsRequired(int)`: the minimum number of bits needed to
/// represent `v` unsigned. **Never returns 0** -- Java's
/// `PackedInts.unsignedBitsRequired` is `Math.max(1, 32 - nlz)` and documents
/// "this method returns at least 1", so `bitsRequired(0) == 1`.
///
/// The floor is load-bearing in [`pfor_encode`]'s histogram, not cosmetic. With
/// a 0 for zero values, the bit-width search could walk all the way down to
/// `b == 0` for any block whose non-zero values number at most `MAX_EXCEPTIONS`
/// and fit in 8 bits, where Java stops at `b == 1`: same decoded values, but
/// different bytes than Lucene writes for the same input.
// ARITH: `u32::leading_zeros` returns 0..=32 by definition, so `32 - ..` is
// in 0..=32 and cannot underflow.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
pub(crate) fn bits_required(v: u32) -> u32 {
    (32 - v.leading_zeros()).max(1)
}

/// `PForUtil.allEqual`.
fn all_equal(ints: &[u32; BLOCK_SIZE]) -> bool {
    ints.iter().all(|&v| v == ints[0])
}

/// `PForUtil.encode`: encode 256 integers, choosing the smallest
/// `bitsPerValue` that keeps at most [`MAX_EXCEPTIONS`] (7) values as
/// "patched" outliers (their low `bitsPerValue` bits are stored in the
/// packed body, their high bits as a separate `(index: u8, highBits: u8)`
/// patch list after it) — a direct port of the histogram-based bit-width
/// search in `PForUtil.encode`, not a simplified/`ForUtil`-only substitute.
/// `ints` is mutated in place exactly like the Java version (exception
/// values are masked down to `patchedBitsRequired` bits before the packed
/// body is written), so callers must pass their own scratch copy.
///
/// Every value must fit in 31 bits (`< 0x8000_0000`): the 1-byte token
/// stores `bitsPerValue` in its low 5 bits (`token & 0x1f`), so a
/// `bitsPerValue` of 32 would alias to `0` (the "all-equal" marker) and
/// silently corrupt the decode. This matches the real domain exactly --
/// Lucene doc deltas and term frequencies are non-negative Java `int`s, so
/// `PackedInts.bitsRequired` never returns 32 for them in practice.
pub fn pfor_encode<W: DataOutput>(ints: &mut [u32; BLOCK_SIZE], out: &mut W) {
    pfor_encode_with(&mut ForUtil::new(), ints, out)
}

fn pfor_encode_with<W: DataOutput>(
    for_util: &mut ForUtil,
    ints: &mut [u32; BLOCK_SIZE],
    out: &mut W,
) {
    let mut histogram = [0u32; 33];
    let mut max_bits_required = 0u32;
    // ARITH: exactly `BLOCK_SIZE` (256) increments are spread across the
    // histogram, so no bucket can pass 256; `bits_required` returns `1..=32`,
    // which is in range for the 33-entry array (Java's is `new int[32]` and
    // would throw for a value needing 32 bits -- see the `debug_assert!`
    // below, which is where this port makes that impossibility explicit).
    #[allow(clippy::arithmetic_side_effects)]
    for &v in ints.iter() {
        let bits = bits_required(v);
        histogram[bits as usize] += 1;
        max_bits_required = max_bits_required.max(bits);
    }
    // The 1-byte token carries `bitsPerValue` in 5 bits, so 32 would alias to
    // the `0` "all-equal" marker. Java can't hit this either -- `PForUtil`
    // feeds it non-negative `int`s and `PackedInts.bitsRequired` throws on a
    // negative -- but there the impossibility is enforced; here it is only
    // documented, so check it where the invariant actually lives.
    debug_assert!(
        max_bits_required <= 31,
        "pfor_encode values must fit in 31 bits, got one needing {max_bits_required}"
    );

    // We store patches on a byte, so bits can't be decreased by more than 8.
    let min_bits = max_bits_required.saturating_sub(8);
    let mut cumulative_exceptions = 0u32;
    let mut patched_bits_required = max_bits_required;
    let mut num_exceptions = 0u32;
    let mut b = max_bits_required;
    // ARITH: `b` strictly decreases from `max_bits_required` and the loop
    // breaks at `b == min_bits` *before* the decrement, so `b -= 1` runs only
    // under `b > min_bits >= 0`. Because each `b` is visited at most once and
    // the histogram's buckets sum to `BLOCK_SIZE` (256),
    // `cumulative_exceptions` is bounded by 256.
    #[allow(clippy::arithmetic_side_effects)]
    loop {
        if cumulative_exceptions as usize > MAX_EXCEPTIONS {
            break;
        }
        patched_bits_required = b;
        num_exceptions = cumulative_exceptions;
        cumulative_exceptions += histogram[b as usize];
        if b == min_bits {
            break;
        }
        b -= 1;
    }
    // `bits_required` never returns 0, so the buckets `1..=max_bits_required`
    // already account for all 256 values: reaching `b == 0` would need
    // `cumulative_exceptions <= MAX_EXCEPTIONS` at that point, but it is 256
    // there. That is what keeps `for_util.encode` below (which refuses a
    // `bitsPerValue` of 0) reachable only with a legal width.
    debug_assert!(
        (1..=32).contains(&patched_bits_required),
        "pfor_encode picked an unencodable bitsPerValue: {patched_bits_required}"
    );

    let max_unpatched_value = mask32(patched_bits_required);
    // The one reservation in this module. It is safe from the abort that a
    // disk-sized `Vec::with_capacity` risks on two counts: this is the encode
    // side, so nothing here was read from a file at all, and the search loop
    // above assigns `num_exceptions` only while `cumulative_exceptions` is
    // still at or under `MAX_EXCEPTIONS`, so it is at most 7.
    debug_assert!(num_exceptions as usize <= MAX_EXCEPTIONS);
    let mut exceptions: Vec<(u8, u8)> = Vec::with_capacity(num_exceptions as usize);
    if num_exceptions > 0 {
        for (i, v) in ints.iter_mut().enumerate() {
            if *v > max_unpatched_value {
                exceptions.push((i as u8, (*v >> patched_bits_required) as u8));
                *v &= max_unpatched_value;
            }
        }
        debug_assert_eq!(exceptions.len(), num_exceptions as usize);
    }

    if all_equal(ints) && max_bits_required <= 8 {
        // `PForUtil.encode`'s all-equal fast path pre-shifts each patch's
        // high byte left by `patchedBitsRequired` here since the packed body
        // is skipped entirely (a plain vint carries the single repeated
        // value instead); `pfor_decode`'s exception loop always shifts a
        // patch left by `bitsPerValue`, so pre-shifting by
        // `patched_bits_required` compensates for the `bitsPerValue == 0`
        // read in that branch.
        //
        // `patch << patched_bits_required` is a `u8` shift, so a
        // `patched_bits_required` of 8 would be a panic rather than a wrap.
        // It cannot happen: this branch requires `max_bits_required <= 8`, so
        // every value is at most 255, and `patched_bits_required == 8` makes
        // `max_unpatched_value == 255` -- no value can exceed it, so
        // `exceptions` is empty and the loop body never runs. Whenever a
        // patch does exist, `patched_bits_required <= 7`.
        debug_assert!(
            exceptions.is_empty() || patched_bits_required <= 7,
            "all-equal patch shift out of range: {patched_bits_required}"
        );
        out.write_byte((num_exceptions << 5) as u8);
        out.write_vint(ints[0] as i32);
        for &(idx, patch) in &exceptions {
            out.write_byte(idx);
            out.write_byte(patch << patched_bits_required);
        }
    } else {
        let token = ((num_exceptions << 5) | patched_bits_required) as u8;
        out.write_byte(token);
        for_util.encode(ints, patched_bits_required, out);
        for &(idx, patch) in &exceptions {
            out.write_byte(idx);
            out.write_byte(patch);
        }
    }
}

#[cfg(test)]
mod tests {
    // The gate is about values read off disk; a test's `i + 1` is not one.
    #![allow(clippy::arithmetic_side_effects)]

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
            let mut bytes = Vec::new();
            for_encode(&mut values.clone(), bits, &mut bytes);
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
            let mut bytes = Vec::new();
            for_encode(&mut values.clone(), bits, &mut bytes);
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
    fn pfor_skip_consumes_exactly_what_pfor_decode_does() {
        // `PForUtil.skip` has to land on the same byte `PForUtil.decode`
        // would, for every shape of block: the all-equal fast path, a packed
        // body, and either with exceptions. A skip that is off by one byte
        // corrupts every block after it, silently.
        let shapes: Vec<Vec<u32>> = vec![
            // all-equal, no exceptions
            vec![7; BLOCK_SIZE],
            // all-equal with two exceptions
            {
                let mut v = vec![7u32; BLOCK_SIZE];
                v[10] = 7 | (3 << 8);
                v[200] = 7 | (1 << 8);
                v
            },
            // a packed body, no exceptions
            (0..BLOCK_SIZE).map(|i| (i % 13) as u32).collect(),
            // a packed body with exceptions (a handful of outliers)
            {
                let mut v: Vec<u32> = (0..BLOCK_SIZE).map(|i| (i % 13) as u32).collect();
                v[3] = 100_000;
                v[250] = 70_000;
                v
            },
            // maximum width `pfor_encode` supports (31 bits)
            (0..BLOCK_SIZE)
                .map(|i| (u32::MAX >> 1) - i as u32)
                .collect(),
        ];
        for (shape_idx, values) in shapes.into_iter().enumerate() {
            let mut bytes = Vec::new();
            let mut ints = [0u32; BLOCK_SIZE];
            ints.copy_from_slice(&values);
            pfor_encode(&mut ints, &mut bytes);
            // Trailing sentinel so a skip that overshoots is visible.
            bytes.extend_from_slice(&[0xAB, 0xCD]);

            let mut decode_in = SliceInput::new(&bytes);
            let mut out = [0u32; BLOCK_SIZE];
            ForUtil::new()
                .pfor_decode(&mut decode_in, &mut out)
                .unwrap();
            assert_eq!(out.to_vec(), values, "shape {shape_idx} round trip");

            let mut skip_in = SliceInput::new(&bytes);
            pfor_skip(&mut skip_in).unwrap();
            assert_eq!(
                skip_in.position(),
                decode_in.position(),
                "shape {shape_idx}: skip and decode must consume the same bytes"
            );
        }
    }

    #[test]
    fn pfor_skip_reports_eof_rather_than_running_off_the_end() {
        let bytes = [1u8, 2, 3];
        let mut r = SliceInput::new(&bytes);
        assert!(pfor_skip(&mut r).is_err());
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

    // --- pfor_encode/for_encode round-trip tests -----------------------
    //
    // This repo's `BLOCK_SIZE` is 256 (real Lucene 10.5.0's
    // `Lucene104PostingsFormat`/`ForUtil.BLOCK_SIZE`), not the 128 this task
    // was originally scoped against -- these boundary tests exercise exactly
    // one full block (256 entries) plus the module's own decode-side
    // boundary cases (all-zero / full-32-bit) that a real-Lucene fixture
    // term won't reliably hit. The 127/129-style "one packed block + a
    // vint tail" boundary from the original task statement is a
    // `postings_writer`-level concern (choosing how many full blocks vs. a
    // tail to emit for a given docFreq); wiring `pfor_encode` into that
    // writer is out of scope here -- see this module's doc comment and
    // `docs/parity.md`.

    fn bits_required_for_test(v: u32) -> u32 {
        bits_required(v)
    }

    #[test]
    fn pfor_roundtrip_exactly_256_entries_no_exceptions() {
        // Every value fits in the same bit width -> patched_bits_required ==
        // max_bits_required, num_exceptions == 0.
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as u32) & 0xFF; // fits in 8 bits
        }
        let mut ints = values;
        let mut bytes = Vec::new();
        pfor_encode(&mut ints, &mut bytes);

        let mut r = SliceInput::new(&bytes);
        let mut decoded = [0u32; BLOCK_SIZE];
        pfor_decode(&mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn pfor_roundtrip_all_zero_needs_zero_bits() {
        let values = [0u32; BLOCK_SIZE];
        let mut ints = values;
        let mut bytes = Vec::new();
        pfor_encode(&mut ints, &mut bytes);
        // All-equal (0) with maxBitsRequired == 0 <= 8 takes the vint
        // fast path: 1 token byte + 1 vint byte for value 0.
        assert_eq!(bytes.len(), 2);

        let mut r = SliceInput::new(&bytes);
        let mut decoded = [1u32; BLOCK_SIZE];
        pfor_decode(&mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn pfor_roundtrip_full_31_bits_required() {
        // `pfor_encode`'s 1-byte token packs `numExceptions` into the top 3
        // bits and `bitsPerValue` into the bottom 5 (`token & 0x1f`), so
        // `bitsPerValue` can only ever range `0..=31` -- exactly matching the
        // real domain (Lucene doc deltas/freqs are non-negative Java `int`s,
        // so `bitsRequired` never exceeds 31 in practice; a `bitsPerValue`
        // of 32 would alias to the token's `0` == "all-equal" marker and is
        // out of scope here, just like it is for the real `PForUtil`).
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            // A couple of entries pinned to the largest representable value
            // so max_bits_required == 31 deterministically, plus varied
            // filler so it isn't all-equal.
            *v = if i % 7 == 0 {
                0x7FFF_FFFF
            } else {
                (i as u32).wrapping_mul(2654435761) & 0x7FFF_FFFF
            };
        }
        assert_eq!(bits_required_for_test(0x7FFF_FFFF), 31);

        let mut ints = values;
        let mut bytes = Vec::new();
        pfor_encode(&mut ints, &mut bytes);

        let mut r = SliceInput::new(&bytes);
        let mut decoded = [0u32; BLOCK_SIZE];
        pfor_decode(&mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn pfor_roundtrip_few_outliers_use_patched_exceptions() {
        // Almost every value fits in 1 bit; a handful of outliers need ~21
        // bits. `pfor_encode` patches those outliers as exceptions rather
        // than paying 21 bits for all 256 entries -- but the patch byte can
        // only absorb up to 8 bits of reduction (`minBits = maxBits - 8`),
        // so the packed body still costs `patched_bits_required = 13` bits
        // per entry here, not the theoretical minimum of 1 bit. That
        // 8-bit-reduction ceiling is a real `PForUtil` constraint (the patch
        // is stored in a single byte), not a bug in this port.
        let mut values = [1u32; BLOCK_SIZE];
        let outlier_indices = [3usize, 17, 100, 255];
        for &i in &outlier_indices {
            values[i] = 0x000F_FFF0 + i as u32; // needs 20 or 21 bits
        }
        let mut ints = values;
        let mut bytes = Vec::new();
        pfor_encode(&mut ints, &mut bytes);
        // 1 token byte + a 13-bit-per-value packed body (num_bytes(13)) + 2
        // bytes (index, patch) per exception.
        let expected_len = 1 + num_bytes(13) + outlier_indices.len() * 2;
        assert_eq!(bytes.len(), expected_len);
        assert!(
            bytes.len() < 1 + num_bytes(21),
            "patched encoding ({} bytes) should still beat a plain 21-bit body ({} bytes)",
            bytes.len(),
            1 + num_bytes(21)
        );

        let mut r = SliceInput::new(&bytes);
        let mut decoded = [0u32; BLOCK_SIZE];
        pfor_decode(&mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn decode_rejects_bits_per_value_outside_the_supported_range() {
        // A corrupt `.doc` block header can carry any signed byte; every
        // out-of-range width must be an error rather than an index panic.
        let bytes = vec![0u8; 1024];
        for bits in [0u32, 33, 64, 127] {
            let mut r = SliceInput::new(&bytes);
            let mut ints = [0u32; BLOCK_SIZE];
            assert!(
                ForUtil::new().decode(bits, &mut r, &mut ints).is_err(),
                "bits_per_value={bits} must be rejected"
            );
        }
    }

    #[test]
    fn bits_required_never_returns_zero_like_javas_packed_ints() {
        // `PackedInts.unsignedBitsRequired` is `Math.max(1, 32 - nlz)`.
        assert_eq!(bits_required(0), 1);
        assert_eq!(bits_required(1), 1);
        assert_eq!(bits_required(2), 2);
        assert_eq!(bits_required(0x7FFF_FFFF), 31);
        assert_eq!(bits_required(0xFFFF_FFFF), 32);
    }

    #[test]
    fn pfor_encode_mostly_zero_block_picks_javas_bit_width() {
        // 254 zeros, a single 1 and a single 200: the shape where a
        // `bitsRequired(0) == 0` floor would let the width search reach
        // `b == 0` and take the all-equal/vint path, where real `PForUtil`
        // stops at `b == 1` and emits a 1-bit packed body plus one exception.
        let mut values = [0u32; BLOCK_SIZE];
        values[100] = 1;
        values[200] = 200;
        let mut ints = values;
        let mut bytes = Vec::new();
        pfor_encode(&mut ints, &mut bytes);

        // token = (numExceptions << 5) | patchedBitsRequired = (1 << 5) | 1
        assert_eq!(bytes[0], 33);
        assert_eq!(bytes.len(), 1 + num_bytes(1) + 2);
        // The exception is index 200 with high bits 200 >> 1 == 100.
        assert_eq!(&bytes[bytes.len() - 2..], &[200u8, 100u8]);

        let mut r = SliceInput::new(&bytes);
        let mut decoded = [0u32; BLOCK_SIZE];
        pfor_decode(&mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn pfor_encode_matches_for_decode_directly() {
        // pfor_encode with bitsPerValue > 0 and no exceptions must be
        // readable by the lower-level for_decode once the 1-byte token is
        // skipped, proving pfor_encode's packed body is byte-identical to
        // for_encode's.
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = (i as u32) & 0x1F; // 5 bits, no exceptions
        }
        let mut ints = values;
        let mut bytes = Vec::new();
        pfor_encode(&mut ints, &mut bytes);

        let mut r = SliceInput::new(&bytes);
        let token = r.read_byte().unwrap();
        assert_eq!(token, 5); // bitsPerValue=5, numExceptions=0
        let mut decoded = [0u32; BLOCK_SIZE];
        for_decode(5, &mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    #[should_panic(expected = "bitsPerValue out of range")]
    fn encode_refuses_a_zero_bit_width_instead_of_spinning_forever() {
        // Java's `ForUtil.encode` loops `for (shift -= bitsPerValue; shift >=
        // 0; shift -= bitsPerValue)`: at `bitsPerValue == 0` the induction
        // variable never moves and the encoder hangs. A hang is the one
        // failure `catch_unwind` at the FFI boundary cannot convert back into
        // a Java exception, so this port refuses the width. Without the
        // `assert!` in `ForUtil::encode` this test does not fail — it never
        // returns at all.
        let mut values = [1u32; BLOCK_SIZE];
        let mut bytes = Vec::new();
        for_encode(&mut values, 0, &mut bytes);
    }

    #[test]
    #[should_panic(expected = "bitsPerValue out of range")]
    fn encode_refuses_a_width_above_32() {
        // Above 32 the very first `ints[idx] << shift` shifts by a negative
        // amount; Java walks off `MASKS32` instead. Either way the caller
        // deserves the width in the message, not a bare shift panic.
        let mut values = [1u32; BLOCK_SIZE];
        let mut bytes = Vec::new();
        for_encode(&mut values, 33, &mut bytes);
    }

    #[test]
    fn every_token_byte_decodes_or_errors_but_never_panics() {
        // A corrupt `.doc`/`.pos` block header is a single arbitrary byte:
        // `bitsPerValue = token & 0x1f` steers the body and `numExceptions =
        // token >> 5` steers the patch loop, whose index byte then indexes
        // `ints` directly. Sweep all 256 tokens against a body long enough
        // for any width (1 KiB packed body + 7 patch pairs) and against a
        // truncated one, and require every outcome to be `Ok` or a typed
        // error.
        for token in 0u16..=255 {
            for body in [0x00u8, 0xFF, 0xA5] {
                let mut bytes = vec![token as u8];
                bytes.extend(std::iter::repeat_n(body, num_bytes(32) + 2 * 7));
                let full_len = bytes.len();
                for len in [1usize, 2, 33, full_len / 2, full_len] {
                    let mut r = SliceInput::new(&bytes[..len]);
                    let mut ints = [0u32; BLOCK_SIZE];
                    let _ = ForUtil::new().pfor_decode(&mut r, &mut ints);
                    let mut r = SliceInput::new(&bytes[..len]);
                    let _ = pfor_skip(&mut r);
                }
            }
        }
    }

    #[test]
    fn decode_sweeps_every_width_over_an_all_ones_body() {
        // `decode_slow`'s scratch bookkeeping (`tmp_idx`) and `split_ints`'
        // shift/offset bookkeeping are pinned by `debug_assert!`s; run every
        // legal width over a saturated body so a debug test run actually
        // exercises those claims rather than trusting the comment.
        let bytes = vec![0xFFu8; num_bytes(32)];
        for bits in 1u32..=32 {
            let mut r = SliceInput::new(&bytes);
            let mut ints = [0u32; BLOCK_SIZE];
            ForUtil::new().decode(bits, &mut r, &mut ints).unwrap();
            let limit = mask32(bits);
            assert!(
                ints.iter().all(|&v| v <= limit),
                "bits={bits} decoded a value wider than the declared width"
            );
        }
    }

    #[test]
    fn all_widths_round_trip_including_the_encode_slow_boundary() {
        // Widths 1..=32 in one sweep, with a value pattern that fills the
        // full width. This is what pins the `chunks_exact` rewrite of
        // `decode3..decode15` and `expand8`/`expand16` against the previous
        // explicit-index form: a wrong stride or a wrong lane shows up as a
        // mismatched value here for the width that uses it.
        for bits in 1u32..=32 {
            let mut values = [0u32; BLOCK_SIZE];
            for (i, v) in values.iter_mut().enumerate() {
                *v = ((i as u32).wrapping_mul(2654435761) ^ (i as u32).rotate_left(11))
                    & mask32(bits);
            }
            // Force both extremes of the width into the block.
            values[0] = 0;
            values[BLOCK_SIZE - 1] = mask32(bits);
            let mut bytes = Vec::new();
            for_encode(&mut values.clone(), bits, &mut bytes);
            assert_eq!(bytes.len(), num_bytes(bits), "bits={bits}");
            let mut r = SliceInput::new(&bytes);
            let mut decoded = [0u32; BLOCK_SIZE];
            for_decode(bits, &mut r, &mut decoded).unwrap();
            assert_eq!(decoded, values, "bits={bits}");
        }
    }

    #[test]
    fn pfor_encode_all_equal_path_with_a_patch_shift() {
        // The one shape that exercises `patch << patched_bits_required` in
        // the all-equal branch: after masking, every value is equal, and the
        // patch's high byte has to be pre-shifted because the decoder's
        // exception loop will shift by `bitsPerValue == 0`. The shift is a
        // `u8` shift, so a `patched_bits_required` of 8 would panic — the
        // branch's `debug_assert!` states why it cannot be 8, and this test
        // is the one that runs it.
        let mut values = [3u32; BLOCK_SIZE];
        values[5] = 19;
        values[100] = 19;
        let mut ints = values;
        let mut bytes = Vec::new();
        pfor_encode(&mut ints, &mut bytes);
        // bitsPerValue == 0 (the all-equal marker) with two exceptions.
        assert_eq!(bytes[0], 2 << 5);

        let mut r = SliceInput::new(&bytes);
        let mut decoded = [0u32; BLOCK_SIZE];
        pfor_decode(&mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values);
    }

    #[test]
    fn default_is_a_fresh_decoder() {
        let mut fu = ForUtil::default();
        let bytes = vec![0u8; num_bytes(1)];
        let mut ints = [1u32; BLOCK_SIZE];
        fu.decode(1, &mut SliceInput::new(&bytes), &mut ints)
            .unwrap();
        assert!(ints.iter().all(|&v| v == 0));
    }

    // --- A/B microbenchmark -------------------------------------------
    //
    // Criterion is unusable on this machine (batch c24 measured the same code
    // at 83/91/129 us across three runs), so this is a min-of-N harness: the
    // minimum of N timed batches is the run least disturbed by the scheduler,
    // and it is repeatable to within ~0.3% here where the mean is not. This
    // is what the "iterators measured slower" claim in `decode3`..`decode15`
    // rests on. `#[ignore]`d; run with
    //   cargo test -p lucene-codecs --release --lib for_util::tests::bench \
    //     -- --ignored --nocapture --test-threads=1
    #[test]
    #[ignore = "microbenchmark"]
    fn bench_decode_and_encode_min_of_n() {
        use std::time::{Duration, Instant};
        fn min_of_n(n: u32, inner: u32, mut f: impl FnMut()) -> Duration {
            let mut best = Duration::from_secs(9999);
            for _ in 0..n {
                let t = Instant::now();
                for _ in 0..inner {
                    f();
                }
                best = best.min(t.elapsed());
            }
            best
        }
        for bits in [1u32, 3, 5, 6, 7, 9, 11, 13, 15, 16, 21, 31] {
            let mut values = [0u32; BLOCK_SIZE];
            for (i, v) in values.iter_mut().enumerate() {
                *v = ((i as u32).wrapping_mul(2654435761) ^ (i as u32).rotate_left(7))
                    & mask32(bits);
            }
            let mut bytes = Vec::new();
            for_encode(&mut values.clone(), bits, &mut bytes);
            let (mut fu, mut ints) = (ForUtil::new(), [0u32; BLOCK_SIZE]);
            let d = min_of_n(60, 4000, || {
                let mut r = SliceInput::new(std::hint::black_box(&bytes));
                fu.decode(bits, &mut r, &mut ints).unwrap();
                std::hint::black_box(&ints);
            });
            let mut out = Vec::with_capacity(num_bytes(bits));
            let e = min_of_n(60, 4000, || {
                let mut scratch = *std::hint::black_box(&values);
                out.clear();
                fu.encode(&mut scratch, bits, &mut out);
            });
            println!("bits={bits:2} decode={d:?} encode={e:?}");
        }
    }
}
