//! `HalfFloatPoint`: a point of IEEE-754 half-precision floats, two bytes a
//! dimension, with its encoding helpers and query factories.
//!
//! The conversion is Java's bit for bit: `float` to half rounds to the
//! nearest representable half with ties to even (`roundShift`), denormal
//! `float`s become zero, overflow becomes infinity, and `NaN` keeps its top
//! mantissa bits (after `Float.floatToIntBits` canonicalises it).

use lucene_index::document::Number;

use super::{packed_indexable, point_type, PackedField};
use crate::document::{PointInSetQuery, PointRangeQuery};
use crate::Result;

/// `HalfFloatPoint`.
#[derive(Debug, Clone, PartialEq)]
pub struct HalfFloatPoint(PackedField);

impl HalfFloatPoint {
    /// `BYTES`.
    pub const BYTES: usize = 2;

    /// `new HalfFloatPoint(name, point...)`.
    pub fn new(name: impl Into<String>, point: &[f32]) -> Result<Self> {
        let packed = Self::pack(point)?;
        Ok(HalfFloatPoint(PackedField {
            name: name.into(),
            field_type: point_type(point.len(), 2)?,
            packed,
        }))
    }

    /// `nextUp(v)`: the smallest half float greater than `v` (`v` itself
    /// for `NaN` and `+Inf`).
    pub fn next_up(v: f32) -> f32 {
        if v.is_nan() || v == f32::INFINITY {
            return v;
        }
        let s = Self::half_float_to_sortable_short(v);
        let r = Self::sortable_short_to_half_float(s);
        if r <= v {
            return Self::sortable_short_to_half_float(s.wrapping_add(1));
        }
        r
    }

    /// `nextDown(v)`: the greatest half float less than `v` (`v` itself for
    /// `NaN` and `-Inf`).
    pub fn next_down(v: f32) -> f32 {
        if v.is_nan() || v == f32::NEG_INFINITY {
            return v;
        }
        let s = Self::half_float_to_sortable_short(v);
        let r = Self::sortable_short_to_half_float(s);
        if r >= v {
            return Self::sortable_short_to_half_float(s.wrapping_sub(1));
        }
        r
    }

    /// `halfFloatToSortableShort(v)`.
    pub fn half_float_to_sortable_short(v: f32) -> i16 {
        sortable_short_bits(half_float_to_short_bits(v))
    }

    /// `sortableShortToHalfFloat(bits)`.
    pub fn sortable_short_to_half_float(bits: i16) -> f32 {
        short_bits_to_half_float(sortable_short_bits(bits))
    }

    /// `pack(point...)`.
    pub fn pack(point: &[f32]) -> Result<Vec<u8>> {
        if point.is_empty() {
            return Err(super::illegal("point must not be 0 dimensions"));
        }
        Ok(point
            .iter()
            .flat_map(|&v| Self::encode_dimension(v))
            .collect())
    }

    /// `encodeDimension(value, dest, offset)`.
    pub fn encode_dimension(value: f32) -> [u8; 2] {
        short_to_sortable_bytes(Self::half_float_to_sortable_short(value))
    }

    /// `decodeDimension(value, offset)`: the dimension at the start of
    /// `value`.
    ///
    /// # Panics
    /// When `value` is shorter than two bytes.
    pub fn decode_dimension(value: &[u8]) -> f32 {
        Self::sortable_short_to_half_float(sortable_bytes_to_short(value))
    }

    /// `setFloatValues(point...)`.
    pub fn set_values(&mut self, point: &[f32]) -> Result<()> {
        self.0.check_dims(point.len())?;
        self.0.packed = Self::pack(point)?;
        Ok(())
    }

    /// `numericValue()`: the one dimension, as a `Float`.
    pub fn numeric(&self) -> Result<Number> {
        self.0.check_single()?;
        Ok(Number::Float(Self::decode_dimension(&self.0.packed)))
    }

    /// The packed point (`binaryValue()`).
    pub fn packed(&self) -> &[u8] {
        &self.0.packed
    }

    /// `newExactQuery(field, value)`.
    pub fn new_exact_query(field: &str, value: f32) -> Result<PointRangeQuery> {
        Self::new_range_query(field, value, value)
    }

    /// `newRangeQuery(field, lowerValue, upperValue)`: inclusive.
    pub fn new_range_query(field: &str, lower: f32, upper: f32) -> Result<PointRangeQuery> {
        Self::new_range_query_nd(field, &[lower], &[upper])
    }

    /// `newRangeQuery(field, lowerValue[], upperValue[])`.
    pub fn new_range_query_nd(
        field: &str,
        lower: &[f32],
        upper: &[f32],
    ) -> Result<PointRangeQuery> {
        PointRangeQuery::new(field, lower.len(), Self::pack(lower)?, Self::pack(upper)?)
    }

    /// `newSetQuery(field, values...)`. `PointInSetQuery` sorts the encoded
    /// points itself, and the encoding keeps Java's `Arrays.sort` order, so
    /// the values are not sorted first.
    pub fn new_set_query(field: &str, values: &[f32]) -> Result<PointInSetQuery> {
        PointInSetQuery::new(
            field,
            1,
            Self::BYTES,
            values.iter().map(|&v| Self::encode_dimension(v).to_vec()),
        )
    }
}

packed_indexable!(HalfFloatPoint, |p: &HalfFloatPoint| p.numeric().ok());

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

/// `shortToSortableBytes(value, result, offset)`: big-endian, sign bit
/// flipped.
fn short_to_sortable_bytes(value: i16) -> [u8; 2] {
    ((value as u16) ^ 0x8000).to_be_bytes()
}

/// `sortableBytesToShort(encoded, offset)`.
fn sortable_bytes_to_short(encoded: &[u8]) -> i16 {
    (u16::from_be_bytes([encoded[0], encoded[1]]) ^ 0x8000) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_and_rounds_to_even() {
        for v in [0.0f32, 1.0, -1.0, 65504.0, 0.5, -2.0, 6.103_515_6e-5] {
            assert_eq!(
                HalfFloatPoint::sortable_short_to_half_float(
                    HalfFloatPoint::half_float_to_sortable_short(v)
                ),
                v
            );
        }
        // 1 + 2^-11 is halfway between 1 and 1 + 2^-10: ties to even (1).
        let h = |v: f32| {
            HalfFloatPoint::sortable_short_to_half_float(
                HalfFloatPoint::half_float_to_sortable_short(v),
            )
        };
        assert_eq!(h(1.000_488_3), 1.0);
        assert_eq!(h(1.001_464_8), 1.001_953_1);
        assert_eq!(h(65520.0), f32::INFINITY);
        assert_eq!(h(-65520.0), f32::NEG_INFINITY);
        assert_eq!(h(1e-40), 0.0, "a denormal float is zero");
        assert_eq!(h(1e-8).to_bits(), 0, "below half's smallest denormal");
        assert!(h(f32::NAN).is_nan());
        assert_eq!(h(-0.0).to_bits(), 0x8000_0000);
    }

    #[test]
    fn next_up_and_down() {
        assert_eq!(HalfFloatPoint::next_up(1.0), 1.000_976_6);
        assert_eq!(HalfFloatPoint::next_down(1.0), 0.999_511_7);
        assert!(HalfFloatPoint::next_up(f32::NAN).is_nan());
        assert_eq!(HalfFloatPoint::next_up(f32::INFINITY), f32::INFINITY);
        assert_eq!(
            HalfFloatPoint::next_down(f32::NEG_INFINITY),
            f32::NEG_INFINITY
        );
        // A float between two halves steps to the nearer one on its side.
        assert_eq!(HalfFloatPoint::next_up(1.0001), 1.000_976_6);
        assert_eq!(HalfFloatPoint::next_down(1.0009), 1.0);
        assert_eq!(HalfFloatPoint::next_up(65504.0), f32::INFINITY);
    }

    #[test]
    fn fields_and_queries() {
        let mut p = HalfFloatPoint::new("h", &[1.5]).unwrap();
        assert_eq!(p.packed(), &HalfFloatPoint::encode_dimension(1.5));
        assert_eq!(p.numeric().unwrap(), Number::Float(1.5));
        assert_eq!(
            lucene_index::document::IndexableField::numeric_value(&p),
            Some(Number::Float(1.5))
        );
        p.set_values(&[2.5]).unwrap();
        assert_eq!(HalfFloatPoint::decode_dimension(p.packed()), 2.5);
        let e = p.set_values(&[1.0, 2.0]).unwrap_err();
        assert!(e
            .to_string()
            .contains("uses 1 dimensions; cannot change to (incoming) 2"));
        let two = HalfFloatPoint::new("h2", &[1.0, 2.0]).unwrap();
        let e = two.numeric().unwrap_err();
        assert!(e
            .to_string()
            .contains("cannot convert to a single numeric value"));
        assert_eq!(
            lucene_index::document::IndexableField::numeric_value(&two),
            None
        );
        assert!(HalfFloatPoint::new("h", &[]).is_err());
        let q = HalfFloatPoint::new_exact_query("h", 1.5).unwrap();
        assert_eq!(q.lower, q.upper);
        assert_eq!(q.bytes_per_dim, 2);
        let s = HalfFloatPoint::new_set_query("h", &[2.0, 1.0, 2.0]).unwrap();
        assert_eq!(s.points.len(), 2);
    }
}
