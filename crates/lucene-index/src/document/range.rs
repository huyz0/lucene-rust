//! Range fields: `IntRange`, `LongRange`, `FloatRange`, `DoubleRange` and
//! `InetAddressRange` (a box of 1..=4 dimensions indexed as a
//! `2 * dims`-dimension point), their doc-values twins
//! (`*RangeDocValuesField`, over `BinaryRangeDocValuesField`), and the
//! relation logic both query families share (`RangeFieldQuery.QueryType`).
//!
//! What reaches disk: the packed value is every dimension's minimum, then
//! every dimension's maximum, each in the point encoding of its type
//! (`NumericUtils.intToSortableBytes`, the sortable float/double integers, an
//! address's 16-byte form). The point field has `2 * dims` dimensions of that
//! width; the doc-values field stores the same bytes as a `BINARY` value.
//!
//! The query factories (`newIntersectsQuery`, `newWithinQuery`,
//! `newContainsQuery`, `newCrossesQuery`, `newSlowIntersectsQuery`) live in
//! `lucene_search::document`.

use std::borrow::Cow;
use std::cmp::Ordering;
use std::net::IpAddr;

use lucene_analysis::Analyzer;
use lucene_codecs::points::Relation;

use super::doc_values::BinaryDocValuesField;
use super::inet_address::InetAddressPoint;
use super::numeric::{
    double_to_sortable_long, float_to_sortable_int, int_to_sortable_bytes, long_to_sortable_bytes,
    sortable_bytes_to_int, sortable_bytes_to_long, sortable_int_to_float, sortable_long_to_double,
};
use super::{illegal, DocValuesType, FieldTokens, FieldType, IndexableField, Result, StoredValue};

/// `RangeFieldQuery.QueryType`: how a query box relates to an indexed box.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RangeQueryType {
    /// The indexed range shares at least one point with the query range.
    Intersects,
    /// The indexed range lies entirely inside the query range.
    Within,
    /// The indexed range contains the whole query range.
    Contains,
    /// The indexed range intersects, but is not within, the query range.
    Crosses,
}

/// `ArrayUtil.getUnsignedComparator(bytesPerDim)`: unsigned lexicographic
/// order of the `bytes_per_dim` bytes at each offset.
///
/// ARITH: every offset is below `2 * 4 * 16` (see [`RangeQueryType`]'s
/// callers), so `off + width` cannot overflow; a short slice panics on the
/// index, as Java's comparator throws.
#[allow(clippy::arithmetic_side_effects)]
fn cmp_at(a: &[u8], a_off: usize, b: &[u8], b_off: usize, width: usize) -> Ordering {
    a[a_off..a_off + width].cmp(&b[b_off..b_off + width])
}

impl RangeQueryType {
    /// `QueryType.compare(query, min, max, numDims, bytesPerDim, dim)`: one
    /// dimension of a cell against the query. `Crosses` has no per-dimension
    /// form (Java throws); [`Self::compare`] handles it across all
    /// dimensions.
    ///
    /// ARITH: (whole body) `dim < num_dims <= 4` and `bytes_per_dim <= 16`
    /// (the point limits), so every offset is below `2 * 4 * 16`.
    #[allow(clippy::arithmetic_side_effects)]
    fn compare_dim(
        self,
        query: &[u8],
        min: &[u8],
        max: &[u8],
        num_dims: usize,
        bytes_per_dim: usize,
        dim: usize,
    ) -> Relation {
        let min_off = dim * bytes_per_dim;
        let max_off = min_off + bytes_per_dim * num_dims;
        let c = |a: &[u8], ao: usize, b: &[u8], bo: usize| cmp_at(a, ao, b, bo, bytes_per_dim);
        match self {
            RangeQueryType::Intersects => {
                if c(query, max_off, min, min_off).is_lt()
                    || c(query, min_off, max, max_off).is_gt()
                {
                    return Relation::CellOutsideQuery;
                }
                if c(query, max_off, max, min_off).is_ge()
                    && c(query, min_off, min, max_off).is_le()
                {
                    return Relation::CellInsideQuery;
                }
                Relation::CellCrossesQuery
            }
            RangeQueryType::Within => {
                if c(query, max_off, min, max_off).is_lt()
                    || c(query, min_off, max, min_off).is_gt()
                {
                    return Relation::CellOutsideQuery;
                }
                if c(query, max_off, max, max_off).is_ge()
                    && c(query, min_off, min, min_off).is_le()
                {
                    return Relation::CellInsideQuery;
                }
                Relation::CellCrossesQuery
            }
            RangeQueryType::Contains => {
                if c(query, max_off, max, max_off).is_gt()
                    || c(query, min_off, min, min_off).is_lt()
                {
                    return Relation::CellOutsideQuery;
                }
                if c(query, max_off, min, max_off).is_le()
                    && c(query, min_off, max, min_off).is_ge()
                {
                    return Relation::CellInsideQuery;
                }
                Relation::CellCrossesQuery
            }
            RangeQueryType::Crosses => {
                unreachable!("CROSSES has no per-dimension relation; compare() handles it")
            }
        }
    }

    /// `QueryType.matches(query, packed, numDims, bytesPerDim, dim)`.
    ///
    /// ARITH: as [`Self::compare_dim`].
    #[allow(clippy::arithmetic_side_effects)]
    fn matches_dim(
        self,
        query: &[u8],
        packed: &[u8],
        num_dims: usize,
        bytes_per_dim: usize,
        dim: usize,
    ) -> bool {
        let min_off = dim * bytes_per_dim;
        let max_off = min_off + bytes_per_dim * num_dims;
        let c = |a: &[u8], ao: usize, b: &[u8], bo: usize| cmp_at(a, ao, b, bo, bytes_per_dim);
        match self {
            RangeQueryType::Intersects => {
                c(query, max_off, packed, min_off).is_ge()
                    && c(query, min_off, packed, max_off).is_le()
            }
            RangeQueryType::Within => {
                c(query, min_off, packed, min_off).is_le()
                    && c(query, max_off, packed, max_off).is_ge()
            }
            RangeQueryType::Contains => {
                c(query, min_off, packed, min_off).is_ge()
                    && c(query, max_off, packed, max_off).is_le()
            }
            RangeQueryType::Crosses => {
                unreachable!("CROSSES has no per-dimension match; matches() handles it")
            }
        }
    }

    /// `QueryType.compare(query, minPacked, maxPacked, numDims, bytesPerDim)`:
    /// a BKD cell `[min, max]` (each `2 * num_dims` dimensions) against the
    /// query box.
    pub fn compare(
        self,
        query: &[u8],
        min: &[u8],
        max: &[u8],
        num_dims: usize,
        bytes_per_dim: usize,
    ) -> Relation {
        if self == RangeQueryType::Crosses {
            let intersects =
                RangeQueryType::Intersects.compare(query, min, max, num_dims, bytes_per_dim);
            if intersects == Relation::CellOutsideQuery {
                return Relation::CellOutsideQuery;
            }
            let within = RangeQueryType::Within.compare(query, min, max, num_dims, bytes_per_dim);
            if within == Relation::CellInsideQuery {
                return Relation::CellOutsideQuery;
            }
            if intersects == Relation::CellInsideQuery && within == Relation::CellOutsideQuery {
                return Relation::CellInsideQuery;
            }
            return Relation::CellCrossesQuery;
        }
        let mut inside = true;
        for dim in 0..num_dims {
            match self.compare_dim(query, min, max, num_dims, bytes_per_dim, dim) {
                Relation::CellOutsideQuery => return Relation::CellOutsideQuery,
                Relation::CellInsideQuery => {}
                Relation::CellCrossesQuery => inside = false,
            }
        }
        if inside {
            Relation::CellInsideQuery
        } else {
            Relation::CellCrossesQuery
        }
    }

    /// `QueryType.matches(query, packed, numDims, bytesPerDim)`: one indexed
    /// box against the query box.
    pub fn matches(
        self,
        query: &[u8],
        packed: &[u8],
        num_dims: usize,
        bytes_per_dim: usize,
    ) -> bool {
        if self == RangeQueryType::Crosses {
            return RangeQueryType::Intersects.matches(query, packed, num_dims, bytes_per_dim)
                && !RangeQueryType::Within.matches(query, packed, num_dims, bytes_per_dim);
        }
        (0..num_dims).all(|dim| self.matches_dim(query, packed, num_dims, bytes_per_dim, dim))
    }
}

/// A range field's type: `2 * dims` point dimensions of `bytes`.
fn range_type(dims: usize, bytes: i32) -> Result<FieldType> {
    let d = i32::try_from(dims)
        .ok()
        .and_then(|d| d.checked_mul(2))
        .ok_or_else(|| illegal("too many dimensions"))?;
    let mut ft = FieldType::new();
    ft.set_dimensions(d, bytes)?;
    Ok(ft.frozen())
}

/// The `checkArgs(min, max)` every range type shares.
fn check_args<T>(min: &[T], max: &[T], class: &str) -> Result<()> {
    if min.is_empty() || max.is_empty() {
        return Err(illegal("min/max range values cannot be null or empty"));
    }
    if min.len() != max.len() {
        return Err(illegal("min/max ranges must agree"));
    }
    if min.len() > 4 {
        return Err(illegal(format!(
            "{class} does not support greater than 4 dimensions"
        )));
    }
    Ok(())
}

/// The range point: its packed bytes, shared by every range type.
#[derive(Debug, Clone, PartialEq)]
struct RangePoint {
    name: String,
    field_type: FieldType,
    packed: Vec<u8>,
}

impl RangePoint {
    fn dims(&self) -> usize {
        usize::try_from(self.field_type.point_dimension_count() / 2).unwrap_or(0)
    }

    fn check_dim(&self, dimension: usize) -> Result<()> {
        if dimension >= self.dims() {
            return Err(illegal(format!(
                "Index {dimension} out of bounds for length {}",
                self.dims()
            )));
        }
        Ok(())
    }
}

macro_rules! numeric_range {
    (
        $(#[$m:meta])*
        $ty:ident, $dv:ident, $prim:ty, $bytes:expr, $class:expr, $nan:expr,
        $enc:expr, $dec:expr
    ) => {
        $(#[$m])*
        #[derive(Debug, Clone, PartialEq)]
        pub struct $ty(RangePoint);

        impl $ty {
            /// `BYTES`: the width of one bound.
            pub const BYTES: usize = $bytes;

            /// `new XxxRange(name, min, max)`.
            pub fn new(name: impl Into<String>, min: &[$prim], max: &[$prim]) -> Result<Self> {
                check_args(min, max, $class)?;
                Ok($ty(RangePoint {
                    name: name.into(),
                    field_type: range_type(min.len(), $bytes)?,
                    packed: Self::encode(min, max)?,
                }))
            }

            /// `setRangeValues(min, max)`.
            pub fn set_range_values(&mut self, min: &[$prim], max: &[$prim]) -> Result<()> {
                check_args(min, max, $class)?;
                if min.len() != self.0.dims() || max.len() != self.0.dims() {
                    return Err(illegal(format!(
                        "field (name={}) uses {} dimensions; cannot change to (incoming) {} \
                         dimensions",
                        self.0.name,
                        self.0.dims(),
                        min.len()
                    )));
                }
                self.0.packed = Self::encode(min, max)?;
                Ok(())
            }

            /// `encode(min, max)` (`verifyAndEncode`): every minimum, then
            /// every maximum.
            pub fn encode(min: &[$prim], max: &[$prim]) -> Result<Vec<u8>> {
                check_args(min, max, $class)?;
                let mut mins = Vec::with_capacity(min.len().saturating_mul($bytes).saturating_mul(2));
                let mut maxs = Vec::with_capacity(max.len().saturating_mul($bytes));
                for (&lo, &hi) in min.iter().zip(max) {
                    if $nan(lo) {
                        return Err(illegal(format!(
                            "invalid min value (NaN) in {}", $class
                        )));
                    }
                    if $nan(hi) {
                        return Err(illegal(format!(
                            "invalid max value (NaN) in {}", $class
                        )));
                    }
                    if lo > hi {
                        return Err(illegal(format!(
                            "min value ({lo}) is greater than max value ({hi})"
                        )));
                    }
                    mins.extend_from_slice(&($enc)(lo));
                    maxs.extend_from_slice(&($enc)(hi));
                }
                mins.extend_from_slice(&maxs);
                Ok(mins)
            }

            /// `decodeMin(bytes, dimension)`.
            ///
            /// # Panics
            /// When `dimension` is outside the encoded box.
            //
            // ARITH: `dimension * BYTES` indexes a caller's slice; an
            // out-of-box dimension panics on the slice, as Java's array
            // access throws, and cannot overflow first for `dimension < 4`.
            #[allow(clippy::arithmetic_side_effects)]
            pub fn decode_min(b: &[u8], dimension: usize) -> $prim {
                ($dec)(&b[dimension * $bytes..])
            }

            /// `decodeMax(bytes, dimension)`.
            ///
            /// # Panics
            /// When `dimension` is outside the encoded box.
            //
            // ARITH: as `decode_min`; `len / 2` cannot overflow.
            #[allow(clippy::arithmetic_side_effects)]
            pub fn decode_max(b: &[u8], dimension: usize) -> $prim {
                ($dec)(&b[b.len() / 2 + dimension * $bytes..])
            }

            /// `getMin(dimension)`.
            pub fn min(&self, dimension: usize) -> Result<$prim> {
                self.0.check_dim(dimension)?;
                Ok(Self::decode_min(&self.0.packed, dimension))
            }

            /// `getMax(dimension)`.
            pub fn max(&self, dimension: usize) -> Result<$prim> {
                self.0.check_dim(dimension)?;
                Ok(Self::decode_max(&self.0.packed, dimension))
            }

            /// `binaryValue()`.
            pub fn packed(&self) -> &[u8] {
                &self.0.packed
            }
        }

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
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }

        $(#[$m])*
        ///
        /// The doc-values twin: the same packed box as a `BINARY` doc value
        /// (`BinaryRangeDocValuesField`).
        #[derive(Debug, Clone, PartialEq)]
        pub struct $dv {
            inner: BinaryRangeDocValuesField,
            min: Vec<$prim>,
            max: Vec<$prim>,
        }

        impl $dv {
            /// `new XxxRangeDocValuesField(field, min, max)`.
            pub fn new(field: impl Into<String>, min: &[$prim], max: &[$prim]) -> Result<Self> {
                let packed = $ty::encode(min, max)?;
                if min.iter().zip(max).any(|(lo, hi)| lo > hi) {
                    return Err(illegal("min should be less than max"));
                }
                Ok($dv {
                    inner: BinaryRangeDocValuesField::new(
                        field.into(),
                        packed,
                        min.len(),
                        $bytes,
                    ),
                    min: min.to_vec(),
                    max: max.to_vec(),
                })
            }

            /// `getMin(dimension)`.
            pub fn min(&self, dimension: usize) -> Result<$prim> {
                self.min
                    .get(dimension)
                    .copied()
                    .ok_or_else(|| illegal("Dimension out of valid range"))
            }

            /// `getMax(dimension)`.
            pub fn max(&self, dimension: usize) -> Result<$prim> {
                self.max
                    .get(dimension)
                    .copied()
                    .ok_or_else(|| illegal("Dimension out of valid range"))
            }

            /// The `BinaryRangeDocValuesField` this is.
            pub fn as_binary_range(&self) -> &BinaryRangeDocValuesField {
                &self.inner
            }
        }

        impl IndexableField for $dv {
            fn name(&self) -> &str {
                self.inner.name()
            }
            fn field_type(&self) -> &FieldType {
                self.inner.field_type()
            }
            fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
                self.inner.binary_value()
            }
            fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
                Ok(None)
            }
        }
    };
}

fn never_nan<T>(_: T) -> bool {
    false
}

numeric_range!(
    /// `IntRange`: an `int` box of 1..=4 dimensions.
    IntRange, IntRangeDocValuesField, i32, 4, "IntRange", never_nan,
    int_to_sortable_bytes, sortable_bytes_to_int
);
numeric_range!(
    /// `LongRange`: a `long` box of 1..=4 dimensions.
    LongRange, LongRangeDocValuesField, i64, 8, "LongRange", never_nan,
    long_to_sortable_bytes, sortable_bytes_to_long
);
numeric_range!(
    /// `FloatRange`: a `float` box of 1..=4 dimensions.
    FloatRange,
    FloatRangeDocValuesField,
    f32,
    4,
    "FloatRange",
    f32::is_nan,
    |v: f32| int_to_sortable_bytes(float_to_sortable_int(v)),
    |b: &[u8]| sortable_int_to_float(sortable_bytes_to_int(b))
);
numeric_range!(
    /// `DoubleRange`: a `double` box of 1..=4 dimensions.
    DoubleRange,
    DoubleRangeDocValuesField,
    f64,
    8,
    "DoubleRange",
    f64::is_nan,
    |v: f64| long_to_sortable_bytes(double_to_sortable_long(v)),
    |b: &[u8]| sortable_long_to_double(sortable_bytes_to_long(b))
);

/// `BinaryRangeDocValuesField`: a range box as a `BINARY` doc value, with
/// the shape a slow range query reads it by.
#[derive(Debug, Clone, PartialEq)]
pub struct BinaryRangeDocValuesField {
    inner: BinaryDocValuesField,
    num_dims: usize,
    num_bytes_per_dimension: usize,
}

impl BinaryRangeDocValuesField {
    /// `BinaryRangeDocValuesField(field, packedValue, numDims,
    /// numBytesPerDimension)`.
    pub fn new(
        field: String,
        packed_value: Vec<u8>,
        num_dims: usize,
        num_bytes_per_dimension: usize,
    ) -> Self {
        let mut ft = FieldType::new();
        ft.set_doc_values_type(DocValuesType::Binary)
            .expect("unfrozen");
        BinaryRangeDocValuesField {
            inner: BinaryDocValuesField::with_type(field, ft.frozen(), packed_value),
            num_dims,
            num_bytes_per_dimension,
        }
    }

    pub fn packed_value(&self) -> &[u8] {
        self.inner.value()
    }

    pub fn num_dims(&self) -> usize {
        self.num_dims
    }

    pub fn num_bytes_per_dimension(&self) -> usize {
        self.num_bytes_per_dimension
    }
}

impl IndexableField for BinaryRangeDocValuesField {
    fn name(&self) -> &str {
        self.inner.name()
    }
    fn field_type(&self) -> &FieldType {
        self.inner.field_type()
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        self.inner.binary_value()
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

/// `InetAddressRange`: a range of IPv4/IPv6 addresses, as one two-dimension
/// point of 16-byte addresses.
#[derive(Debug, Clone, PartialEq)]
pub struct InetAddressRange(RangePoint);

impl InetAddressRange {
    /// `BYTES`.
    pub const BYTES: usize = InetAddressPoint::BYTES;

    /// `new InetAddressRange(name, min, max)`.
    pub fn new(name: impl Into<String>, min: IpAddr, max: IpAddr) -> Result<Self> {
        Ok(InetAddressRange(RangePoint {
            name: name.into(),
            field_type: range_type(1, 16)?,
            packed: Self::encode(min, max)?,
        }))
    }

    /// `setRangeValues(min, max)`.
    pub fn set_range_values(&mut self, min: IpAddr, max: IpAddr) -> Result<()> {
        self.0.packed = Self::encode(min, max)?;
        Ok(())
    }

    /// `encode(min, max)`: both addresses in their 16-byte form, the minimum
    /// first.
    pub fn encode(min: IpAddr, max: IpAddr) -> Result<Vec<u8>> {
        let lo = InetAddressPoint::encode(min);
        let hi = InetAddressPoint::encode(max);
        if lo > hi {
            return Err(illegal(
                "min value cannot be greater than max value for InetAddressRange field",
            ));
        }
        Ok([lo, hi].concat())
    }

    pub fn packed(&self) -> &[u8] {
        &self.0.packed
    }
}

impl IndexableField for InetAddressRange {
    fn name(&self) -> &str {
        &self.0.name
    }
    fn field_type(&self) -> &FieldType {
        &self.0.field_type
    }
    fn binary_value(&self) -> Option<Cow<'_, [u8]>> {
        Some(Cow::Borrowed(&self.0.packed))
    }
    fn stored_value(&self) -> Option<StoredValue> {
        None
    }
    fn token_stream(&self, _analyzer: &Analyzer) -> Result<Option<FieldTokens>> {
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;

    fn int_box(min: &[i32], max: &[i32]) -> Vec<u8> {
        IntRange::encode(min, max).unwrap()
    }

    #[test]
    fn ranges_encode_mins_then_maxes() {
        let r = IntRange::new("r", &[1, -2], &[3, 4]).unwrap();
        assert_eq!(r.packed().len(), 16);
        assert_eq!(r.min(0).unwrap(), 1);
        assert_eq!(r.min(1).unwrap(), -2);
        assert_eq!(r.max(0).unwrap(), 3);
        assert_eq!(r.max(1).unwrap(), 4);
        assert!(r.min(2).is_err());
        assert_eq!(r.field_type().point_dimension_count(), 4);
        assert_eq!(r.name(), "r");
        assert!(r.token_stream(&Analyzer::keyword()).unwrap().is_none());
        assert!(IntRange::new("r", &[], &[]).is_err());
        assert!(IntRange::new("r", &[1], &[1, 2]).is_err());
        assert!(IntRange::new("r", &[1; 5], &[2; 5]).is_err());
        assert!(IntRange::new("r", &[5], &[4]).is_err());
        let mut l = LongRange::new("l", &[-5], &[5]).unwrap();
        l.set_range_values(&[0], &[1]).unwrap();
        assert_eq!(l.max(0).unwrap(), 1);
        assert!(l.set_range_values(&[0, 0], &[1, 1]).is_err());
        assert!(FloatRange::new("f", &[f32::NAN], &[1.0]).is_err());
        assert!(FloatRange::new("f", &[0.0], &[f32::NAN]).is_err());
        let d = DoubleRange::new("d", &[-1.5], &[2.5]).unwrap();
        assert_eq!(d.min(0).unwrap(), -1.5);
        assert_eq!(d.binary_value().unwrap().len(), 16);
    }

    #[test]
    fn doc_values_twins_carry_the_same_bytes() {
        let dv = IntRangeDocValuesField::new("r", &[1], &[9]).unwrap();
        assert_eq!(
            dv.binary_value().unwrap().as_ref(),
            &int_box(&[1], &[9])[..]
        );
        assert_eq!(dv.field_type().doc_values_type(), DocValuesType::Binary);
        assert_eq!(dv.min(0).unwrap(), 1);
        assert_eq!(dv.max(0).unwrap(), 9);
        assert!(dv.min(1).is_err());
        assert!(dv.max(1).is_err());
        assert_eq!(dv.name(), "r");
        assert_eq!(dv.as_binary_range().num_dims(), 1);
        assert_eq!(dv.as_binary_range().num_bytes_per_dimension(), 4);
        assert_eq!(dv.as_binary_range().packed_value().len(), 8);
        assert!(dv.token_stream(&Analyzer::keyword()).unwrap().is_none());
        let b = dv.as_binary_range().clone();
        assert_eq!(b.name(), "r");
        assert!(b.binary_value().is_some());
        assert!(b.token_stream(&Analyzer::keyword()).unwrap().is_none());
        assert!(DoubleRangeDocValuesField::new("d", &[2.0], &[1.0]).is_err());
    }

    #[test]
    fn relations_follow_the_query_type() {
        let q = int_box(&[10], &[20]);
        let m =
            |min: i32, max: i32, t: RangeQueryType| t.matches(&q, &int_box(&[min], &[max]), 1, 4);
        use RangeQueryType::*;
        assert!(m(15, 30, Intersects));
        assert!(!m(21, 30, Intersects));
        assert!(m(12, 18, Within));
        assert!(!m(5, 18, Within));
        assert!(m(5, 25, Contains));
        assert!(!m(12, 25, Contains));
        assert!(m(5, 15, Crosses));
        assert!(!m(12, 15, Crosses), "within is not crossing");
        // Cell relations over 2-dim (min,max) cells.
        let cell =
            |a: (i32, i32), b: (i32, i32)| (int_box(&[a.0], &[a.1]), int_box(&[b.0], &[b.1]));
        let rel = |t: RangeQueryType, a, b| {
            let (lo, hi) = cell(a, b);
            t.compare(&q, &lo, &hi, 1, 4)
        };
        assert_eq!(
            rel(Intersects, (30, 30), (40, 40)),
            Relation::CellOutsideQuery
        );
        assert_eq!(
            rel(Intersects, (12, 12), (15, 15)),
            Relation::CellInsideQuery
        );
        assert_eq!(
            rel(Intersects, (0, 0), (15, 30)),
            Relation::CellCrossesQuery
        );
        assert_eq!(rel(Within, (12, 12), (18, 18)), Relation::CellInsideQuery);
        assert_eq!(rel(Within, (25, 25), (30, 30)), Relation::CellOutsideQuery);
        assert_eq!(rel(Within, (0, 0), (15, 30)), Relation::CellCrossesQuery);
        assert_eq!(rel(Contains, (0, 25), (5, 30)), Relation::CellInsideQuery);
        assert_eq!(
            rel(Contains, (12, 12), (15, 15)),
            Relation::CellOutsideQuery
        );
        assert_eq!(rel(Contains, (0, 0), (15, 30)), Relation::CellCrossesQuery);
        assert_eq!(rel(Crosses, (30, 30), (40, 40)), Relation::CellOutsideQuery);
        assert_eq!(rel(Crosses, (12, 12), (18, 18)), Relation::CellOutsideQuery);
        assert_eq!(rel(Crosses, (0, 21), (5, 30)), Relation::CellInsideQuery);
        assert_eq!(rel(Crosses, (0, 0), (15, 30)), Relation::CellCrossesQuery);
    }

    #[test]
    fn inet_address_ranges() {
        let lo: IpAddr = "10.0.0.1".parse().unwrap();
        let hi: IpAddr = "10.0.0.9".parse().unwrap();
        let mut r = InetAddressRange::new("ip", lo, hi).unwrap();
        assert_eq!(r.packed().len(), 32);
        assert_eq!(r.field_type().point_dimension_count(), 2);
        assert!(InetAddressRange::new("ip", hi, lo).is_err());
        r.set_range_values(lo, lo).unwrap();
        assert_eq!(&r.packed()[..16], &r.packed()[16..]);
        assert_eq!(r.name(), "ip");
        assert_eq!(r.stored_value(), None);
        assert!(r.binary_value().is_some());
        assert!(r.token_stream(&Analyzer::keyword()).unwrap().is_none());
    }
}
