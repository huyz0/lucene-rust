//! IEEE-754 half-precision floats as Lucene's `HalfFloatPoint` encodes
//! them, bit for bit: `float` to half rounds to the nearest representable
//! half with ties to even (`roundShift`), denormal `float`s become zero,
//! overflow becomes infinity, and `NaN` keeps its top mantissa bits (after
//! `Float.floatToIntBits` canonicalises it).

/// `HalfFloatPoint.halfFloatToSortableShort(v)`.
pub fn half_float_to_sortable_short(v: f32) -> i16 {
    sortable_short_bits(half_float_to_short_bits(v))
}

/// `HalfFloatPoint.sortableShortToHalfFloat(bits)`.
pub fn sortable_short_to_half_float(bits: i16) -> f32 {
    short_bits_to_half_float(sortable_short_bits(bits))
}

/// `sortableShortBits(s)`: flips the non-sign bits of a negative short, so
/// the bits sort as the values do.
fn sortable_short_bits(s: i16) -> i16 {
    s ^ ((s >> 15) & 0x7fff)
}

/// `Float.floatToIntBits`: `NaN` canonicalised.
fn float_to_int_bits(v: f32) -> u32 {
    if v.is_nan() {
        0x7fc0_0000
    } else {
        v.to_bits()
    }
}

/// `halfFloatToShortBits(v)`.
///
/// ARITH: every intermediate is a 32-bit pattern handled with Java's
/// wrapping `int` semantics (`wrapping_*`); the values stay far inside
/// `u32` anyway -- `exp` is a byte, the mantissa 24 bits.
fn half_float_to_short_bits(v: f32) -> i16 {
    let float_bits = float_to_int_bits(v);
    let sign = float_bits >> 31;
    let mut exp = (float_bits >> 23) & 0xff;
    let mut mantissa = float_bits & 0x7f_ffff;
    if exp == 0xff {
        // NaN and the infinities.
        exp = 0x1f;
        mantissa >>= 23 - 10;
    } else if exp == 0 {
        // A denormal float is below the smallest half.
        mantissa = 0;
    } else {
        // `exp - 127 + 15`, signed.
        let e = i64::from(exp).wrapping_sub(112);
        if e >= 0x1f {
            exp = 0x1f;
            mantissa = 0;
        } else if e <= 0 {
            // `23 - 10 - exp + 1`.
            let shift = 14i64.wrapping_sub(e);
            if shift >= 32 {
                exp = 0;
                mantissa = 0;
            } else {
                mantissa |= 0x80_0000;
                mantissa = round_shift(mantissa, u32::try_from(shift).unwrap_or(31));
                exp = mantissa >> 10;
                mantissa &= 0x3ff;
            }
        } else {
            let e = u32::try_from(e).unwrap_or(0);
            mantissa = round_shift((e << 23) | mantissa, 23 - 10);
            exp = mantissa >> 10;
            mantissa &= 0x3ff;
        }
    }
    // `(short) (...)`: the low 16 bits.
    #[allow(clippy::cast_possible_truncation)]
    let bits = ((sign << 15) | (exp << 10) | mantissa) as u16;
    bits as i16
}

/// `roundShift(i, shift)`: `i / 2^shift`, rounded to nearest, ties to even.
fn round_shift(i: u32, shift: u32) -> u32 {
    debug_assert!(shift > 0 && shift < 32);
    let mut i = i.wrapping_add(1u32.wrapping_shl(shift.wrapping_sub(1)));
    i = i.wrapping_sub((i >> shift) & 1);
    i >> shift
}

/// `shortBitsToHalfFloat(s)`.
fn short_bits_to_half_float(s: i16) -> f32 {
    let s = u32::from(s as u16);
    let sign = s >> 15;
    let mut exp = (s >> 10) & 0x1f;
    let mut mantissa = s & 0x3ff;
    if exp == 0x1f {
        // NaN or the infinities.
        exp = 0xff;
        mantissa <<= 23 - 10;
    } else if mantissa == 0 && exp == 0 {
        // Zero.
    } else {
        if exp == 0 {
            // A denormal half is a normal float: `shift` is in 1..=10.
            let shift = mantissa.leading_zeros().wrapping_sub(32 - 11);
            mantissa = (mantissa << shift) & 0x3ff;
            exp = 1u32.wrapping_sub(shift);
        }
        exp = exp.wrapping_add(127 - 15);
        mantissa <<= 23 - 10;
    }
    f32::from_bits((sign << 31) | (exp << 23) | mantissa)
}
