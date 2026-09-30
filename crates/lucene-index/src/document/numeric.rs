//! The numeric point and numeric field types: `IntPoint`, `LongPoint`,
//! `FloatPoint`, `DoublePoint`, `BinaryPoint`, and the point-plus-doc-value
//! `IntField`, `LongField`, `FloatField`, `DoubleField` -- with
//! `NumericUtils`' sortable encodings they index.
//!
//! What reaches disk: a point is `numDims * bytesPerDim` packed bytes, each
//! dimension big-endian with its sign bit flipped (`intToSortableBytes` /
//! `longToSortableBytes`), a float or double first mapped to its sortable
//! integer (`floatToSortableInt` / `doubleToSortableLong`). A `*Field`
//! indexes the same one-dimension point and a `SORTED_NUMERIC` doc value: the
//! value itself for int/long, the sortable integer for float/double.
//!
//! The query factories (`newRangeQuery`, `newSetQuery`, ...) and sort fields
//! (`newSortField`) live in `lucene_search::document`, which reads what these
//! write.

use std::borrow::Cow;

use lucene_analysis::Analyzer;

use super::{
    illegal, DocValuesType, FieldTokens, FieldType, IndexableField, Number, Result, Store,
    StoredValue,
};

/// `NumericUtils.intToSortableBytes`.
pub fn int_to_sortable_bytes(value: i32) -> [u8; 4] {
    ((value as u32) ^ 0x8000_0000).to_be_bytes()
}

/// `NumericUtils.sortableBytesToInt`: the first four bytes of `encoded`.
///
/// # Panics
/// When `encoded` is shorter than four bytes (Java's
/// `ArrayIndexOutOfBoundsException`).
pub fn sortable_bytes_to_int(encoded: &[u8]) -> i32 {
    let b: [u8; 4] = encoded[..4].try_into().expect("four bytes");
    (u32::from_be_bytes(b) ^ 0x8000_0000) as i32
}

/// `NumericUtils.longToSortableBytes`.
pub fn long_to_sortable_bytes(value: i64) -> [u8; 8] {
    ((value as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
}

/// `NumericUtils.sortableBytesToLong`: the first eight bytes of `encoded`.
///
/// # Panics
/// When `encoded` is shorter than eight bytes.
pub fn sortable_bytes_to_long(encoded: &[u8]) -> i64 {
    let b: [u8; 8] = encoded[..8].try_into().expect("eight bytes");
    (u64::from_be_bytes(b) ^ 0x8000_0000_0000_0000) as i64
}

/// `NumericUtils.floatToSortableInt` (over `Float.floatToIntBits`, which
/// collapses every `NaN` to one).
pub fn float_to_sortable_int(value: f32) -> i32 {
    let bits = if value.is_nan() {
        0x7fc0_0000
    } else {
        value.to_bits() as i32
    };
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

/// `NumericUtils.sortableIntToFloat`.
pub fn sortable_int_to_float(encoded: i32) -> f32 {
    f32::from_bits((encoded ^ ((encoded >> 31) & 0x7fff_ffff)) as u32)
}

/// `NumericUtils.sortableDoubleBits`: the involution both directions share.
pub fn sortable_double_bits(bits: i64) -> i64 {
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

/// `NumericUtils.doubleToSortableLong` (over `Double.doubleToLongBits`).
pub fn double_to_sortable_long(value: f64) -> i64 {
    let bits = if value.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        value.to_bits() as i64
    };
    sortable_double_bits(bits)
}

/// `NumericUtils.sortableLongToDouble`.
pub fn sortable_long_to_double(encoded: i64) -> f64 {
    f64::from_bits(sortable_double_bits(encoded) as u64)
}

/// `FieldType` with `setDimensions(numDims, bytesPerDim)`, frozen.
fn point_type(num_dims: usize, bytes_per_dim: i32) -> Result<FieldType> {
    let dims = i32::try_from(num_dims).map_err(|_| illegal("too many dimensions"))?;
    let mut ft = FieldType::new();
    ft.set_dimensions(dims, bytes_per_dim)?;
    Ok(ft.frozen())
}

/// The `*Field` type: a one-dimension point and a `SORTED_NUMERIC` doc
/// value, stored when asked.
fn numeric_field_type(bytes: i32, store: Store) -> FieldType {
    let mut ft = FieldType::new();
    ft.set_dimensions(1, bytes)
        .expect("a one-dimension point of 4 or 8 bytes is valid");
    ft.set_doc_values_type(DocValuesType::SortedNumeric)
        .expect("unfrozen");
    if store == Store::Yes {
        ft.set_stored(true).expect("unfrozen");
    }
    ft.frozen()
}

/// A point field's packed value: the part every `*Point` shares.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct PackedPoint {
    name: String,
    field_type: FieldType,
    packed: Vec<u8>,
}

impl PackedPoint {
    fn new(name: String, field_type: FieldType, packed: Vec<u8>) -> Self {
        PackedPoint {
            name,
            field_type,
            packed,
        }
    }

    fn dims(&self) -> usize {
        usize::try_from(self.field_type.point_dimension_count()).unwrap_or(0)
    }

    /// The `setXxxValues(...)` dimension check.
    fn check_dims(&self, incoming: usize) -> Result<()> {
        if incoming != self.dims() {
            return Err(illegal(format!(
                "this field (name={}) uses {} dimensions; cannot change to (incoming) {incoming} \
                 dimensions",
                self.name,
                self.dims()
            )));
        }
        Ok(())
    }

    /// The `numericValue()` dimension check.
    fn check_single(&self) -> Result<()> {
        if self.dims() != 1 {
            return Err(super::Error::IllegalState(format!(
                "this field (name={}) uses {} dimensions; cannot convert to a single numeric value",
                self.name,
                self.dims()
            )));
        }
        Ok(())
    }
}

macro_rules! point_indexable {
    ($ty:ident) => {
        impl IndexableField for $ty {
            fn name(&self) -> &str {
                &self.0.name
            }
            fn field_type(&self) -> &FieldType {
                &self.0.field_type
            }
            fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
                Some(Cow::Borrowed(&self.0.packed))
            }
            fn numeric_value(&self) -> Option<Number> {
                self.numeric().ok()
            }
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }
    };
}

macro_rules! typed_point {
    (
        $(#[$m:meta])*
        $ty:ident, $prim:ty, $bytes:expr, $enc:expr, $dec:expr, $num:ident
    ) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ty(PackedPoint);

        impl $ty {
            /// `BYTES`: the width of one dimension.
            pub const BYTES: usize = $bytes;

            /// `new XxxPoint(name, point...)`.
            pub fn new(name: impl Into<String>, point: &[$prim]) -> Result<Self> {
                let packed = Self::pack(point)?;
                let ft = point_type(point.len(), $bytes as i32)?;
                Ok($ty(PackedPoint::new(name.into(), ft, packed)))
            }

            /// `pack(point...)`.
            pub fn pack(point: &[$prim]) -> Result<Vec<u8>> {
                if point.is_empty() {
                    return Err(illegal("point must not be 0 dimensions"));
                }
                let mut packed = Vec::with_capacity(point.len().saturating_mul($bytes));
                for &v in point {
                    packed.extend_from_slice(&Self::encode_dimension(v));
                }
                Ok(packed)
            }

            /// `encodeDimension(value, dest, offset)`.
            pub fn encode_dimension(value: $prim) -> [u8; $bytes] {
                ($enc)(value)
            }

            /// `decodeDimension(value, offset)`: the dimension at the start
            /// of `value`.
            ///
            /// # Panics
            /// When `value` is shorter than one dimension.
            pub fn decode_dimension(value: &[u8]) -> $prim {
                ($dec)(value)
            }

            /// `setXxxValues(point...)`.
            pub fn set_values(&mut self, point: &[$prim]) -> Result<()> {
                self.0.check_dims(point.len())?;
                self.0.packed = Self::pack(point)?;
                Ok(())
            }

            /// `numericValue()`: the one dimension's value.
            pub fn numeric(&self) -> Result<Number> {
                self.0.check_single()?;
                Ok(Number::$num(Self::decode_dimension(&self.0.packed)))
            }

            /// `binaryValue()`: the packed point.
            pub fn packed(&self) -> &[u8] {
                &self.0.packed
            }
        }

        point_indexable!($ty);
    };
}

typed_point!(
    /// `IntPoint`: an `int` point in 1..=16 dimensions.
    IntPoint, i32, 4, int_to_sortable_bytes, sortable_bytes_to_int, Int
);
typed_point!(
    /// `LongPoint`: a `long` point in 1..=16 dimensions.
    LongPoint, i64, 8, long_to_sortable_bytes, sortable_bytes_to_long, Long
);
typed_point!(
    /// `FloatPoint`: a `float` point in 1..=16 dimensions.
    FloatPoint,
    f32,
    4,
    |v: f32| int_to_sortable_bytes(float_to_sortable_int(v)),
    |b: &[u8]| sortable_int_to_float(sortable_bytes_to_int(b)),
    Float
);
typed_point!(
    /// `DoublePoint`: a `double` point in 1..=16 dimensions.
    DoublePoint,
    f64,
    8,
    |v: f64| long_to_sortable_bytes(double_to_sortable_long(v)),
    |b: &[u8]| sortable_long_to_double(sortable_bytes_to_long(b)),
    Double
);

impl FloatPoint {
    /// `FloatPoint.nextUp`: `Math.nextUp`, except `-0f` steps to `+0f`.
    pub fn next_up(f: f32) -> f32 {
        if f.to_bits() == 0x8000_0000 {
            return 0.0;
        }
        java_next_up_f32(f)
    }

    /// `FloatPoint.nextDown`: `Math.nextDown`, except `+0f` steps to `-0f`.
    pub fn next_down(f: f32) -> f32 {
        if f.to_bits() == 0 {
            return -0.0;
        }
        -java_next_up_f32(-f)
    }
}

impl DoublePoint {
    /// `DoublePoint.nextUp`: `Math.nextUp`, except `-0d` steps to `+0d`.
    pub fn next_up(d: f64) -> f64 {
        if d.to_bits() == 0x8000_0000_0000_0000 {
            return 0.0;
        }
        java_next_up_f64(d)
    }

    /// `DoublePoint.nextDown`: `Math.nextDown`, except `+0d` steps to `-0d`.
    pub fn next_down(d: f64) -> f64 {
        if d.to_bits() == 0 {
            return -0.0;
        }
        -java_next_up_f64(-d)
    }
}

/// `Math.nextUp(float)`: `NaN` and `+Inf` unchanged, `±0` to the smallest
/// subnormal.
fn java_next_up_f32(f: f32) -> f32 {
    if f.is_nan() || f == f32::INFINITY {
        return f;
    }
    if f == 0.0 {
        return f32::from_bits(1);
    }
    let bits = f.to_bits();
    f32::from_bits(if f > 0.0 {
        bits.wrapping_add(1)
    } else {
        bits.wrapping_sub(1)
    })
}

/// `Math.nextUp(double)`.
fn java_next_up_f64(d: f64) -> f64 {
    if d.is_nan() || d == f64::INFINITY {
        return d;
    }
    if d == 0.0 {
        return f64::from_bits(1);
    }
    let bits = d.to_bits();
    f64::from_bits(if d > 0.0 {
        bits.wrapping_add(1)
    } else {
        bits.wrapping_sub(1)
    })
}

/// `BinaryPoint`: a point of opaque, fixed-width byte dimensions.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryPoint(PackedPoint);

impl BinaryPoint {
    /// `new BinaryPoint(name, byte[]... point)`: every dimension the same,
    /// non-zero length.
    pub fn new(name: impl Into<String>, point: &[&[u8]]) -> Result<Self> {
        let bytes_per_dim = Self::check_point(point)?;
        let ft = point_type(point.len(), bytes_per_dim)?;
        Ok(BinaryPoint(PackedPoint::new(
            name.into(),
            ft,
            Self::pack(point)?,
        )))
    }

    /// `new BinaryPoint(name, packedPoint, type)`: already packed, the type
    /// given; the length must be `numDims * bytesPerDim`.
    pub fn from_packed(
        name: impl Into<String>,
        packed: Vec<u8>,
        field_type: FieldType,
    ) -> Result<Self> {
        let want = i64::from(field_type.point_dimension_count())
            .saturating_mul(i64::from(field_type.point_num_bytes()));
        if i64::try_from(packed.len()).ok() != Some(want) {
            return Err(illegal(format!(
                "packedPoint is length={} but type.pointDimensionCount()={} and \
                 type.pointNumBytes()={}",
                packed.len(),
                field_type.point_dimension_count(),
                field_type.point_num_bytes()
            )));
        }
        Ok(BinaryPoint(PackedPoint::new(
            name.into(),
            field_type,
            packed,
        )))
    }

    /// `BinaryPoint`'s `getType(point)` checks: the common dimension width.
    fn check_point(point: &[&[u8]]) -> Result<i32> {
        if point.is_empty() {
            return Err(illegal("point must not be 0 dimensions"));
        }
        let mut bytes_per_dim: Option<usize> = None;
        for dim in point {
            match bytes_per_dim {
                None => {
                    if dim.is_empty() {
                        return Err(illegal("point must not have 0-length values"));
                    }
                    bytes_per_dim = Some(dim.len());
                }
                Some(b) if b != dim.len() => {
                    return Err(illegal(format!(
                        "all dimensions must have same bytes length; got {b} and {}",
                        dim.len()
                    )));
                }
                Some(_) => {}
            }
        }
        i32::try_from(bytes_per_dim.unwrap_or(0)).map_err(|_| illegal("dimension too long"))
    }

    /// `BinaryPoint.pack(byte[]... point)`.
    pub fn pack(point: &[&[u8]]) -> Result<Vec<u8>> {
        Self::check_point(point)?;
        Ok(point.concat())
    }

    /// `binaryValue()`.
    pub fn packed(&self) -> &[u8] {
        &self.0.packed
    }
}

impl IndexableField for BinaryPoint {
    fn name(&self) -> &str {
        &self.0.name
    }
    fn field_type(&self) -> &FieldType {
        &self.0.field_type
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        Some(Cow::Borrowed(&self.0.packed))
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

macro_rules! numeric_field {
    (
        $(#[$m:meta])*
        $ty:ident, $prim:ty, $bytes:expr, $point:ident, $to_dv:expr, $stored:ident, $num:ident
    ) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ty {
            name: String,
            field_type: FieldType,
            value: $prim,
        }

        impl $ty {
            /// `new XxxField(name, value, store)`.
            pub fn new(name: impl Into<String>, value: $prim, store: Store) -> Self {
                $ty {
                    name: name.into(),
                    field_type: numeric_field_type($bytes, store),
                    value,
                }
            }

            /// The value as given.
            pub fn value(&self) -> $prim {
                self.value
            }

            /// `setXxxValue(value)`: the point, the doc value and the stored
            /// value all follow.
            pub fn set_value(&mut self, value: $prim) {
                self.value = value;
            }

            /// The `SORTED_NUMERIC` doc value this field indexes.
            pub fn doc_value(&self) -> i64 {
                ($to_dv)(self.value)
            }
        }

        impl IndexableField for $ty {
            fn name(&self) -> &str {
                &self.name
            }
            fn field_type(&self) -> &FieldType {
                &self.field_type
            }
            /// `binaryValue()`: the one-dimension point.
            fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
                Some(Cow::Owned($point::encode_dimension(self.value).to_vec()))
            }
            /// `numericValue()`: `fieldsData`, which is the doc value.
            fn numeric_value(&self) -> Option<Number> {
                Some(Number::$num(($to_dv)(self.value) as _))
            }
            fn stored_value(&self) -> Option<StoredValue> {
                self.field_type
                    .stored()
                    .then(|| StoredValue::$stored(self.value))
            }
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }
    };
}

numeric_field!(
    /// `IntField`: an `IntPoint` and a `SORTED_NUMERIC` doc value of the
    /// value.
    IntField, i32, 4, IntPoint, i64::from, Int, Int
);
numeric_field!(
    /// `LongField`: a `LongPoint` and a `SORTED_NUMERIC` doc value of the
    /// value.
    LongField, i64, 8, LongPoint, |v: i64| v, Long, Long
);
numeric_field!(
    /// `FloatField`: a `FloatPoint` and a `SORTED_NUMERIC` doc value of
    /// `floatToSortableInt(value)`.
    FloatField,
    f32,
    4,
    FloatPoint,
    |v: f32| i64::from(float_to_sortable_int(v)),
    Float,
    Long
);
numeric_field!(
    /// `DoubleField`: a `DoublePoint` and a `SORTED_NUMERIC` doc value of
    /// `doubleToSortableLong(value)`.
    DoubleField,
    f64,
    8,
    DoublePoint,
    double_to_sortable_long,
    Double,
    Long
);

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    #[test]
    fn sortable_encodings_round_trip_and_order() {
        let ints = [i32::MIN, -5, -1, 0, 1, 7, i32::MAX];
        for w in ints.windows(2) {
            assert!(int_to_sortable_bytes(w[0]) < int_to_sortable_bytes(w[1]));
        }
        for v in ints {
            assert_eq!(sortable_bytes_to_int(&int_to_sortable_bytes(v)), v);
        }
        let longs = [i64::MIN, -1, 0, 1, i64::MAX];
        for w in longs.windows(2) {
            assert!(long_to_sortable_bytes(w[0]) < long_to_sortable_bytes(w[1]));
        }
        for v in longs {
            assert_eq!(sortable_bytes_to_long(&long_to_sortable_bytes(v)), v);
        }
        let floats = [f32::NEG_INFINITY, -1.5, -0.0, 0.0, 2.5, f32::INFINITY];
        for w in floats.windows(2) {
            assert!(float_to_sortable_int(w[0]) < float_to_sortable_int(w[1]));
        }
        for v in floats {
            assert_eq!(
                sortable_int_to_float(float_to_sortable_int(v)).to_bits(),
                v.to_bits()
            );
        }
        assert_eq!(
            float_to_sortable_int(f32::NAN),
            float_to_sortable_int(f32::from_bits(0x7fc0_0001))
        );
        let doubles = [f64::NEG_INFINITY, -1.5, -0.0, 0.0, 2.5, f64::INFINITY];
        for w in doubles.windows(2) {
            assert!(double_to_sortable_long(w[0]) < double_to_sortable_long(w[1]));
        }
        for v in doubles {
            assert_eq!(
                sortable_long_to_double(double_to_sortable_long(v)).to_bits(),
                v.to_bits()
            );
        }
        assert_eq!(double_to_sortable_long(f64::NAN), 0x7ff8_0000_0000_0000);
    }

    #[test]
    fn points_pack_and_decode() {
        let p = IntPoint::new("f", &[1, -2]).unwrap();
        assert_eq!(p.packed().len(), 8);
        assert_eq!(IntPoint::decode_dimension(&p.packed()[4..]), -2);
        assert!(p.numeric().is_err(), "two dimensions have no single value");
        assert_eq!(p.numeric_value(), None);
        assert_eq!(p.field_type().point_dimension_count(), 2);
        assert!(IntPoint::new("f", &[]).is_err());
        let mut one = LongPoint::new("l", &[5]).unwrap();
        assert_eq!(one.numeric_value(), Some(Number::Long(5)));
        one.set_values(&[9]).unwrap();
        assert_eq!(one.numeric().unwrap(), Number::Long(9));
        assert!(one.set_values(&[1, 2]).is_err());
        assert_eq!(one.name(), "l");
        assert_eq!(one.binary_value().unwrap().len(), 8);
        assert!(one.token_stream(&Analyzer::keyword()).unwrap().is_none());
        let f = FloatPoint::new("f", &[1.5]).unwrap();
        assert_eq!(f.numeric().unwrap(), Number::Float(1.5));
        let d = DoublePoint::new("d", &[-2.25]).unwrap();
        assert_eq!(d.numeric().unwrap(), Number::Double(-2.25));
        assert!(IntPoint::new("f", &[0; 17]).is_err(), "too many dims");
    }

    #[test]
    fn next_up_and_down_cross_zero_as_lucene_does() {
        assert_eq!(FloatPoint::next_up(-0.0).to_bits(), 0);
        assert_eq!(FloatPoint::next_down(0.0).to_bits(), 0x8000_0000);
        assert_eq!(
            FloatPoint::next_up(1.0),
            f32::from_bits(1.0f32.to_bits() + 1)
        );
        assert_eq!(
            FloatPoint::next_down(1.0),
            f32::from_bits(1.0f32.to_bits() - 1)
        );
        assert_eq!(
            FloatPoint::next_up(-1.0),
            f32::from_bits((-1.0f32).to_bits() - 1)
        );
        assert_eq!(FloatPoint::next_up(f32::INFINITY), f32::INFINITY);
        assert!(FloatPoint::next_up(f32::NAN).is_nan());
        assert_eq!(FloatPoint::next_up(0.0), f32::from_bits(1));
        assert_eq!(DoublePoint::next_up(-0.0).to_bits(), 0);
        assert_eq!(DoublePoint::next_down(0.0).to_bits(), 0x8000_0000_0000_0000);
        assert_eq!(DoublePoint::next_up(0.0), f64::from_bits(1));
        assert_eq!(
            DoublePoint::next_down(1.0),
            f64::from_bits(1.0f64.to_bits() - 1)
        );
        assert_eq!(
            DoublePoint::next_up(-1.0),
            f64::from_bits((-1.0f64).to_bits() - 1)
        );
        assert_eq!(DoublePoint::next_up(f64::INFINITY), f64::INFINITY);
        assert!(DoublePoint::next_down(f64::NAN).is_nan());
    }

    #[test]
    fn binary_points_check_their_dimensions() {
        let p = BinaryPoint::new("b", &[b"ab", b"cd"]).unwrap();
        assert_eq!(p.packed(), b"abcd");
        assert_eq!(p.field_type().point_num_bytes(), 2);
        assert_eq!(p.name(), "b");
        assert_eq!(p.binary_value().unwrap().as_ref(), b"abcd");
        assert!(p.token_stream(&Analyzer::keyword()).unwrap().is_none());
        assert!(BinaryPoint::new("b", &[]).is_err());
        assert!(BinaryPoint::new("b", &[b""]).is_err());
        assert!(BinaryPoint::new("b", &[b"a", b"bc"]).is_err());
        let ft = p.field_type().clone();
        assert!(BinaryPoint::from_packed("b", b"wxyz".to_vec(), ft.clone()).is_ok());
        assert!(BinaryPoint::from_packed("b", b"wxy".to_vec(), ft).is_err());
    }

    #[test]
    fn numeric_fields_index_a_point_and_a_doc_value() {
        let f = IntField::new("i", -3, Store::Yes);
        assert_eq!(
            f.binary_value().unwrap().as_ref(),
            &int_to_sortable_bytes(-3)
        );
        assert_eq!(f.numeric_value(), Some(Number::Int(-3)));
        assert_eq!(f.stored_value(), Some(StoredValue::Int(-3)));
        assert_eq!(f.doc_value(), -3);
        assert!(f.field_type().stored());
        assert_eq!(
            f.field_type().doc_values_type(),
            DocValuesType::SortedNumeric
        );
        let mut f = FloatField::new("f", 1.25, Store::No);
        assert_eq!(f.stored_value(), None);
        assert_eq!(
            f.numeric_value(),
            Some(Number::Long(i64::from(float_to_sortable_int(1.25))))
        );
        f.set_value(2.0);
        assert_eq!(f.value(), 2.0);
        let d = DoubleField::new("d", -0.5, Store::Yes);
        assert_eq!(d.stored_value(), Some(StoredValue::Double(-0.5)));
        assert_eq!(d.doc_value(), double_to_sortable_long(-0.5));
        let l = LongField::new("l", 1 << 40, Store::No);
        assert_eq!(l.binary_value().unwrap().len(), 8);
        assert_eq!(l.name(), "l");
        assert!(l.token_stream(&Analyzer::keyword()).unwrap().is_none());
    }
}
