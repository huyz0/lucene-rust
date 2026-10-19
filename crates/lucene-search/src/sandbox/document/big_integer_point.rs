//! `BigIntegerPoint`: a point of 128-bit signed integers, sixteen bytes a
//! dimension, with its query factories.
//!
//! Deliberate difference: Java's `BigInteger` holds any integer and `pack`
//! refuses one outside `[MIN_VALUE, MAX_VALUE]`; here a dimension is an
//! `i128`, whose range is exactly that, so the refusal cannot arise. The
//! encoding is `NumericUtils.bigIntToSortableBytes(value, 16, ...)`:
//! big-endian two's complement with the sign bit flipped.

use super::{packed_indexable, point_type, PackedField};
use crate::document::{PointInSetQuery, PointRangeQuery};
use crate::Result;

/// `BigIntegerPoint`.
#[derive(Debug, Clone, PartialEq)]
pub struct BigIntegerPoint(PackedField);

impl BigIntegerPoint {
    /// `BYTES`.
    pub const BYTES: usize = 16;
    /// `MIN_VALUE`: `-2^127`.
    pub const MIN_VALUE: i128 = i128::MIN;
    /// `MAX_VALUE`: `2^127 - 1`.
    pub const MAX_VALUE: i128 = i128::MAX;

    /// `new BigIntegerPoint(name, point...)`.
    pub fn new(name: impl Into<String>, point: &[i128]) -> Result<Self> {
        let packed = Self::pack(point)?;
        Ok(BigIntegerPoint(PackedField {
            name: name.into(),
            field_type: point_type(point.len(), 16)?,
            packed,
        }))
    }

    /// `pack(point...)`.
    pub fn pack(point: &[i128]) -> Result<Vec<u8>> {
        if point.is_empty() {
            return Err(super::illegal("point must not be 0 dimensions"));
        }
        Ok(point
            .iter()
            .flat_map(|&v| Self::encode_dimension(v))
            .collect())
    }

    /// `encodeDimension(value, dest, offset)`.
    pub fn encode_dimension(value: i128) -> [u8; 16] {
        ((value as u128) ^ (1u128 << 127)).to_be_bytes()
    }

    /// `decodeDimension(value, offset)`: the dimension at the start of
    /// `value`.
    ///
    /// # Panics
    /// When `value` is shorter than sixteen bytes.
    pub fn decode_dimension(value: &[u8]) -> i128 {
        let mut b = [0u8; 16];
        b.copy_from_slice(&value[..16]);
        (u128::from_be_bytes(b) ^ (1u128 << 127)) as i128
    }

    /// `setBigIntegerValues(point...)`.
    pub fn set_values(&mut self, point: &[i128]) -> Result<()> {
        self.0.check_dims(point.len())?;
        self.0.packed = Self::pack(point)?;
        Ok(())
    }

    /// `numericValue()`: the one dimension (a `BigInteger` in Java, which
    /// the indexing chain's `Number` has no room for).
    pub fn numeric(&self) -> Result<i128> {
        self.0.check_single()?;
        Ok(Self::decode_dimension(&self.0.packed))
    }

    /// The packed point (`binaryValue()`).
    pub fn packed(&self) -> &[u8] {
        &self.0.packed
    }

    /// `newExactQuery(field, value)`.
    pub fn new_exact_query(field: &str, value: i128) -> Result<PointRangeQuery> {
        Self::new_range_query(field, value, value)
    }

    /// `newRangeQuery(field, lowerValue, upperValue)`: inclusive.
    pub fn new_range_query(field: &str, lower: i128, upper: i128) -> Result<PointRangeQuery> {
        Self::new_range_query_nd(field, &[lower], &[upper])
    }

    /// `newRangeQuery(field, lowerValue[], upperValue[])`.
    pub fn new_range_query_nd(
        field: &str,
        lower: &[i128],
        upper: &[i128],
    ) -> Result<PointRangeQuery> {
        PointRangeQuery::new(field, lower.len(), Self::pack(lower)?, Self::pack(upper)?)
    }

    /// `newSetQuery(field, values...)` (`PointInSetQuery` sorts the
    /// encoded points itself).
    pub fn new_set_query(field: &str, values: &[i128]) -> Result<PointInSetQuery> {
        PointInSetQuery::new(
            field,
            1,
            Self::BYTES,
            values.iter().map(|&v| Self::encode_dimension(v).to_vec()),
        )
    }
}

// `numericValue()` is a `BigInteger`, outside `Number`: the indexing chain
// only reads it for doc values, which this field has none of.
packed_indexable!(BigIntegerPoint, |_: &BigIntegerPoint| None);

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_util::numeric_utils::big_int_to_sortable_bytes;

    #[test]
    fn encoding_is_numeric_utils_big_int() {
        for v in [0i128, 1, -1, 200, -200, i128::MIN, i128::MAX, 1 << 100] {
            let mut want = [0u8; 16];
            big_int_to_sortable_bytes(&v.to_be_bytes(), 16, &mut want, 0).unwrap();
            assert_eq!(BigIntegerPoint::encode_dimension(v), want, "{v}");
            assert_eq!(BigIntegerPoint::decode_dimension(&want), v);
        }
        assert!(BigIntegerPoint::encode_dimension(-1) < BigIntegerPoint::encode_dimension(0));
    }

    #[test]
    fn field_and_queries() {
        let mut p = BigIntegerPoint::new("b", &[7]).unwrap();
        assert_eq!(p.numeric().unwrap(), 7);
        assert_eq!(
            lucene_index::document::IndexableField::numeric_value(&p),
            None
        );
        p.set_values(&[-9]).unwrap();
        assert_eq!(BigIntegerPoint::decode_dimension(p.packed()), -9);
        assert!(p.set_values(&[1, 2]).is_err());
        let two = BigIntegerPoint::new("b2", &[1, 2]).unwrap();
        assert!(two.numeric().is_err());
        assert!(BigIntegerPoint::new("b", &[]).is_err());
        let q = BigIntegerPoint::new_exact_query("b", 3).unwrap();
        assert_eq!(q.bytes_per_dim, 16);
        let s = BigIntegerPoint::new_set_query("b", &[3, 3, -1]).unwrap();
        assert_eq!(s.points.len(), 2);
    }
}
