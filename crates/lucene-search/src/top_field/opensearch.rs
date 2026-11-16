//! OpenSearch's own comparator sources for keys Lucene has no type for, as
//! [`FieldComparatorSource`]s: `_shard_doc` ([`ShardDocSort`]), a
//! `half_float` field ([`HalfFloatSort`]) and an `unsigned_long` field
//! ([`UnsignedLongSort`]).
//!
//! None of them skips. OpenSearch's `HalfFloatComparator` and
//! `UnsignedLongComparator` do, with the field's points, once more than the
//! total-hits threshold's hits are in; a caller that wants their answer runs
//! these only where that cannot happen (or with skipping disabled), and
//! below the threshold the two agree hit for hit.

use std::cmp::Ordering;

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader};

use super::{
    float_to_sortable_int, java_round, FieldComparator, FieldComparatorSource, LeafCtx,
    LeafFieldComparator, Selector, SortError, SortType, SortValue,
};
use crate::directory_reader::SegmentReader;
use crate::Result;

/// `ShardDocFieldComparatorSource`: a document's key is `shardKeyPrefix |
/// (docBase + doc)`, the shard id in the high 32 bits.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShardDocSort {
    /// `((long) shardId) << 32`.
    pub prefix: i64,
}

impl FieldComparatorSource for ShardDocSort {
    fn new_comparator(&self, _: &str, _: usize, _: bool) -> Box<dyn FieldComparator> {
        Box::new(*self)
    }
}

impl FieldComparator for ShardDocSort {
    fn leaf<'a>(&self, ctx: LeafCtx<'a>) -> Result<Box<dyn LeafFieldComparator + 'a>> {
        Ok(Box::new(ShardDocLeaf {
            prefix: self.prefix,
            doc_base: ctx.doc_base,
        }))
    }

    fn compare_values(&self, a: &SortValue, b: &SortValue) -> Ordering {
        long_cmp(a, b, |a, b| a.cmp(&b))
    }
}

struct ShardDocLeaf {
    prefix: i64,
    doc_base: i32,
}

impl LeafFieldComparator for ShardDocLeaf {
    fn value(&mut self, doc: i32, _: f32) -> Result<SortValue> {
        // `shardKeyPrefix | (docBase + doc)`: the int sum, then widened.
        Ok(SortValue::Long(
            self.prefix | i64::from(self.doc_base.wrapping_add(doc)),
        ))
    }
}

/// `HalfFloatValuesComparatorSource` over a field with no nested path: the
/// field's values (`HalfFloatPoint.halfFloatToSortableShort`s) as doubles,
/// `MultiValueMode`'s pick over them, `missing` for a document without one,
/// the result cast to `float`. A hit's value is `float_to_sortable_int` of
/// that float, compared as `Float.compare` compares the floats.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfFloatSort {
    /// `Min`, `Max`, `Sum`, `Avg` or `Median`.
    pub selector: Selector,
    /// `missingObject(missingValue, reversed)`.
    pub missing: f32,
}

impl FieldComparatorSource for HalfFloatSort {
    fn new_comparator(&self, field: &str, _: usize, _: bool) -> Box<dyn FieldComparator> {
        Box::new(Comparator {
            field: field.to_string(),
            pick: Pick::HalfFloat(*self),
        })
    }

    fn value_string(&self, value: i64) -> String {
        crate::explain::java_float(super::sortable_int_to_float(value))
    }
}

/// `UnsignedLongValuesComparatorSource` over a field with no nested path:
/// the field's values (the unsigned longs' bits), `MultiValueMode`'s pick
/// for unsigned longs, `missing` for a document without one. A hit's value
/// is the unsigned long's bits, compared unsigned (`BigInteger.compareTo`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct UnsignedLongSort {
    /// `Min`, `Max`, `Sum`, `Avg` or `Median`.
    pub selector: Selector,
    /// `missingObject(missingValue, reversed).longValue()`.
    pub missing: i64,
}

impl FieldComparatorSource for UnsignedLongSort {
    fn new_comparator(&self, field: &str, _: usize, _: bool) -> Box<dyn FieldComparator> {
        Box::new(Comparator {
            field: field.to_string(),
            pick: Pick::Unsigned(*self),
        })
    }

    fn value_string(&self, value: i64) -> String {
        (value as u64).to_string()
    }
}

/// `DoubleValuesComparatorSource` over a `scaled_float` field with no nested
/// path: the field's values are longs (the value times the scaling factor,
/// rounded), read as doubles by multiplying with the inverse of the factor
/// (`ScaledFloatLeafFieldData`), `MultiValueMode`'s pick over those doubles,
/// `missing` for a document without one. A hit's value is
/// `double_to_sortable_long` of that double, compared as `Double.compare`
/// compares the doubles.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ScaledFloatSort {
    /// `Min`, `Max`, `Sum`, `Avg` or `Median`.
    pub selector: Selector,
    /// `missingObject(missingValue, reversed)`.
    pub missing: f64,
    /// `1d / scalingFactor`.
    pub inverse: f64,
}

impl FieldComparatorSource for ScaledFloatSort {
    fn new_comparator(&self, field: &str, _: usize, _: bool) -> Box<dyn FieldComparator> {
        Box::new(Comparator {
            field: field.to_string(),
            pick: Pick::Scaled(*self),
        })
    }

    fn value_string(&self, value: i64) -> String {
        crate::function::java_double(lucene_util::numeric_utils::sortable_long_to_double(value))
    }
}

#[derive(Clone, Copy)]
enum Pick {
    HalfFloat(HalfFloatSort),
    Unsigned(UnsignedLongSort),
    Scaled(ScaledFloatSort),
}

impl Pick {
    /// [`Self::value`] for a document with at most one value.
    fn single(self, value: Option<i64>) -> i64 {
        match (self, value) {
            (Pick::HalfFloat(s), None) => i64::from(float_to_sortable_int(s.missing)),
            (Pick::HalfFloat(_), Some(v)) => {
                let f = lucene_util::half_float::sortable_short_to_half_float(v as i16);
                i64::from(float_to_sortable_int(f))
            }
            (Pick::Scaled(s), None) => {
                lucene_util::numeric_utils::double_to_sortable_long(s.missing)
            }
            (Pick::Scaled(s), Some(v)) => {
                lucene_util::numeric_utils::double_to_sortable_long(v as f64 * s.inverse)
            }
            (Pick::Unsigned(s), None) => s.missing,
            (Pick::Unsigned(_), Some(v)) => v,
        }
    }

    /// The value a document with `values` (ascending as signed longs,
    /// possibly none) sorts by.
    fn value(self, values: &[i64], scratch: &mut Vec<i64>) -> i64 {
        match self {
            Pick::HalfFloat(s) => {
                scratch.clear();
                scratch.extend(values.iter().map(|&v| {
                    let f = lucene_util::half_float::sortable_short_to_half_float(v as i16);
                    i64::from(float_to_sortable_int(f))
                }));
                s.selector
                    .pick(SortType::Float, scratch)
                    .unwrap_or_else(|| i64::from(float_to_sortable_int(s.missing)))
            }
            Pick::Scaled(s) => {
                scratch.clear();
                scratch.extend(values.iter().map(|&v| {
                    lucene_util::numeric_utils::double_to_sortable_long(v as f64 * s.inverse)
                }));
                s.selector
                    .pick(SortType::Double, scratch)
                    .unwrap_or_else(|| {
                        lucene_util::numeric_utils::double_to_sortable_long(s.missing)
                    })
            }
            Pick::Unsigned(s) => {
                if values.is_empty() {
                    s.missing
                } else {
                    unsigned_pick(s.selector, values)
                }
            }
        }
    }

    fn unsigned(self) -> bool {
        matches!(self, Pick::Unsigned(_))
    }
}

/// `MultiValueMode.pick(SortedNumericUnsignedLongValues)` over a document's
/// values, ascending as signed longs, at least one -- `MIN`'s quirk
/// included: a first value of 0 among several gives the second.
pub(crate) fn unsigned_pick(selector: Selector, values: &[i64]) -> i64 {
    let n = values.len();
    match selector {
        Selector::Max => {
            let mut max = values[0];
            for &v in &values[1..] {
                if max < 0 && v >= 0 {
                    return max;
                }
                max = v;
            }
            max
        }
        Selector::Sum => values.iter().fold(0i64, |t, &v| t.wrapping_add(v)),
        Selector::Avg => {
            let total = values.iter().fold(0i64, |t, &v| t.wrapping_add(v));
            if n > 1 {
                divide_unsigned_and_round_up(total, n as i64)
            } else {
                total
            }
        }
        Selector::Median => {
            // `MultiValueMode.MEDIAN`'s unsigned pick: the values in signed order, the
            // ones past `i64::MAX` (negative) first; the middle of them counted from
            // the first non-negative one, wrapping round. Two middles are averaged
            // as `(sum >>> 1) + (sum & 1)` of their wrapping sum.
            let first = if values[0] >= 0 {
                0
            } else {
                values[1..]
                    .iter()
                    .position(|&v| v >= 0)
                    .map_or(0, |i| i + 1)
            };
            let at = ((n - 1) / 2 + first) % n;
            if n.is_multiple_of(2) {
                let sum = values[at].wrapping_add(values[(at + 1) % n]);
                (((sum as u64) >> 1) as i64).wrapping_add(sum & 1)
            } else {
                values[at]
            }
        }
        _ => {
            let min = values[0];
            if n == 1 || min > 0 {
                return min;
            }
            values[1..].iter().copied().find(|&v| v >= 0).unwrap_or(min)
        }
    }
}

/// `MultiValueMode.divideUnsignedAndRoundUp`.
pub(crate) fn divide_unsigned_and_round_up(dividend: i64, divisor: i64) -> i64 {
    let q = (((dividend as u64) >> 1) as i64 / divisor) << 1;
    let r = dividend.wrapping_sub(q.wrapping_mul(divisor));
    let quotient = q + ((r | !(r.wrapping_sub(divisor))) as u64 >> 63) as i64;
    let rem = r - ((!(r.wrapping_sub(divisor)) >> 63) & divisor);
    quotient.wrapping_add(java_round(rem as f64 / divisor as f64))
}

fn long_cmp(a: &SortValue, b: &SortValue, cmp: impl Fn(i64, i64) -> Ordering) -> Ordering {
    match (a, b) {
        (SortValue::Long(a), SortValue::Long(b)) => cmp(*a, *b),
        _ => Ordering::Equal,
    }
}

struct Comparator {
    field: String,
    pick: Pick,
}

impl FieldComparator for Comparator {
    fn leaf<'a>(&self, ctx: LeafCtx<'a>) -> Result<Box<dyn LeafFieldComparator + 'a>> {
        Ok(Box::new(Leaf {
            values: Values::open(ctx.reader, &self.field)?,
            pick: self.pick,
            buf: Vec::new(),
            scratch: Vec::new(),
        }))
    }

    fn compare_values(&self, a: &SortValue, b: &SortValue) -> Ordering {
        if self.pick.unsigned() {
            long_cmp(a, b, |a, b| (a as u64).cmp(&(b as u64)))
        } else {
            long_cmp(a, b, |a, b| a.cmp(&b))
        }
    }
}

struct Leaf<'a> {
    values: Values<'a>,
    pick: Pick,
    buf: Vec<i64>,
    scratch: Vec<i64>,
}

impl LeafFieldComparator for Leaf<'_> {
    fn value(&mut self, doc: i32, _: f32) -> Result<SortValue> {
        // A document with one value picks that value whatever the mode (a sum or an
        // average over one value is the value), so a `NUMERIC` column skips the
        // buffer and the selection.
        if let Values::Single(r) = &mut self.values {
            let v = r.value(doc).map_err(crate::Error::from)?;
            return Ok(SortValue::Long(self.pick.single(v)));
        }
        self.values.read(doc, &mut self.buf)?;
        Ok(SortValue::Long(
            self.pick.value(&self.buf, &mut self.scratch),
        ))
    }
}

/// `IndexNumericFieldData.load(context).getLongValues()`: the field's
/// `SORTED_NUMERIC` (or `NUMERIC`) column, none without the field, an error
/// for another doc-values type.
enum Values<'a> {
    Absent,
    Single(NumericReader<'a>),
    Multi(SortedNumericReader<'a>),
}

impl<'a> Values<'a> {
    fn open(reader: &'a SegmentReader, field: &str) -> Result<Self> {
        let Some(info) = reader.field_infos().fields.iter().find(|i| i.name == field) else {
            return Ok(Values::Absent);
        };
        let Some((meta, data)) = reader.doc_values_for_field(info.number) else {
            return Ok(Values::Absent);
        };
        Ok(if let Some(e) = meta.sorted_numeric_entry(info.number) {
            Values::Multi(SortedNumericReader::new(data, e))
        } else if let Some(e) = meta.numeric_entry(info.number) {
            Values::Single(NumericReader::new(data, e))
        } else if meta.binary_entry(info.number).is_some()
            || meta.sorted_entry(info.number).is_some()
            || meta.sorted_set_entry(info.number).is_some()
        {
            return Err(SortError::DocValuesType(field.to_string()).into());
        } else {
            Values::Absent
        })
    }

    fn read(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        match self {
            Values::Absent => {}
            Values::Single(r) => out.extend(r.value(doc).map_err(crate::Error::from)?),
            Values::Multi(r) => r.values(doc, out).map_err(crate::Error::from)?,
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
