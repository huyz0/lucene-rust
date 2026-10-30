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
    float_to_sortable_int, java_round, FieldComparator, FieldComparatorSource, LeafCtx, LeafFieldComparator,
    Selector, SortError, SortType, SortValue,
};
use crate::directory_reader::SegmentReader;
use crate::sandbox::document::HalfFloatPoint;
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
        Ok(SortValue::Long(self.prefix | i64::from(self.doc_base.wrapping_add(doc))))
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
    /// `Min`, `Max`, `Sum` or `Avg`.
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

#[derive(Clone, Copy)]
enum Pick {
    HalfFloat(HalfFloatSort),
    Unsigned(UnsignedLongSort),
}

impl Pick {
    /// The value a document with `values` (ascending as signed longs,
    /// possibly none) sorts by.
    fn value(self, values: &[i64], scratch: &mut Vec<i64>) -> i64 {
        match self {
            Pick::HalfFloat(s) => {
                scratch.clear();
                scratch.extend(values.iter().map(|&v| {
                    let f = HalfFloatPoint::sortable_short_to_half_float(v as i16);
                    i64::from(float_to_sortable_int(f))
                }));
                s.selector
                    .pick(SortType::Float, scratch)
                    .unwrap_or_else(|| i64::from(float_to_sortable_int(s.missing)))
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
fn unsigned_pick(selector: Selector, values: &[i64]) -> i64 {
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
fn divide_unsigned_and_round_up(dividend: i64, divisor: i64) -> i64 {
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
        self.values.read(doc, &mut self.buf)?;
        Ok(SortValue::Long(self.pick.value(&self.buf, &mut self.scratch)))
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
mod tests {
    use std::sync::Arc;

    use super::super::{search_sorted, FieldDoc, SortField};
    use super::*;
    use crate::directory_reader::DirectoryReader;
    use crate::query::{BooleanQuery, Clause, MatchAllDocsQuery, TermQuery};
    use lucene_store::FsDirectory;

    fn fixture() -> DirectoryReader {
        let dir = format!(
            "{}/../../fixtures/data/sorted_search_index",
            env!("CARGO_MANIFEST_DIR")
        );
        DirectoryReader::open(&FsDirectory::open(dir)).expect("open fixture")
    }

    fn query(term: Option<&str>) -> BooleanQuery {
        let mut q = BooleanQuery::new();
        match term {
            Some(t) => q.must.push(Clause::Term(TermQuery::new("body", t.as_bytes()))),
            None => q.must.push(Clause::MatchAllDocs(MatchAllDocsQuery::new(0))),
        }
        q
    }

    fn sorted(
        reader: &DirectoryReader,
        q: &BooleanQuery,
        sort: &[SortField],
        top_n: usize,
        after: Option<&FieldDoc>,
    ) -> Vec<FieldDoc> {
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        search_sorted(&segments, reader.segment_readers(), q, &norms, sort, top_n, u64::MAX, after)
            .unwrap()
            .hits
    }

    fn docs(hits: &[FieldDoc]) -> Vec<i32> {
        hits.iter().map(|h| h.doc).collect()
    }

    #[test]
    fn unsigned_picks_are_opensearchs() {
        let big = |v: u64| v as i64;
        // Ascending as signed: the values past i64::MAX come first.
        let values = [big(u64::MAX - 1), big(u64::MAX), 3, 9];
        assert_eq!(unsigned_pick(Selector::Min, &values), 3);
        assert_eq!(unsigned_pick(Selector::Max, &values), big(u64::MAX));
        assert_eq!(unsigned_pick(Selector::Max, &[2, 7]), 7);
        assert_eq!(unsigned_pick(Selector::Max, &[big(1 << 63), big(u64::MAX)]), big(u64::MAX));
        assert_eq!(unsigned_pick(Selector::Min, &[big(1 << 63), big(u64::MAX)]), big(1 << 63));
        // MultiValueMode.MIN's quirk: a leading 0 among several gives the next.
        assert_eq!(unsigned_pick(Selector::Min, &[0, 4, 6]), 4);
        assert_eq!(unsigned_pick(Selector::Min, &[0]), 0);
        assert_eq!(unsigned_pick(Selector::Sum, &[big(u64::MAX), 2]), 1);
        assert_eq!(unsigned_pick(Selector::Avg, &[5]), 5);
        assert_eq!(unsigned_pick(Selector::Avg, &[1, 2]), 2);
        for (total, n) in [(7u64, 2u64), (9, 4), (u64::MAX, 2), (u64::MAX - 5, 3), (1 << 63, 7), (10, 4)] {
            // Rounded half up, as BigInteger division then Math.round would.
            let want = ((u128::from(total) * 2 + u128::from(n)) / (2 * u128::from(n))) as u64;
            assert_eq!(divide_unsigned_and_round_up(total as i64, n as i64) as u64, want, "{total} / {n}");
        }
    }

    #[test]
    fn shard_doc_sorts_by_the_global_document() {
        let reader = fixture();
        let prefix = 3i64 << 32;
        let key = |reverse| SortField::with_source("_shard_doc", Arc::new(ShardDocSort { prefix }), reverse);
        let q = query(Some("w1"));
        let by_doc = sorted(&reader, &q, &[SortField::doc()], 100_000, None);
        let asc = sorted(&reader, &q, &[key(false)], 100_000, None);
        assert_eq!(docs(&asc), docs(&by_doc));
        assert!(asc.iter().all(|h| h.values == vec![prefix | i64::from(h.doc)]));
        let desc = sorted(&reader, &q, &[key(true)], 40, None);
        let mut want = docs(&by_doc);
        want.reverse();
        assert_eq!(docs(&desc), want[..40]);
        // search_after: the documents after the 40th, as the key orders them.
        let next = sorted(&reader, &q, &[key(true)], 40, desc.last());
        assert_eq!(docs(&next), want[40..80]);
        // Another shard's key: everything or nothing is after it.
        let other = FieldDoc { doc: 0, values: vec![4i64 << 32], terms: Vec::new() };
        assert_eq!(sorted(&reader, &q, &[key(true)], 40, Some(&other)).len(), 40);
        assert!(sorted(&reader, &q, &[key(false)], 40, Some(&other)).is_empty());
        assert_eq!(ShardDocSort { prefix }.sort_field_string("_shard_doc", false), "<custom:\"_shard_doc\">");
    }

    #[test]
    fn unsigned_long_orders_as_the_signed_key_over_small_values() {
        let reader = fixture();
        let q = query(Some("w0"));
        for (selector, reverse) in [
            (Selector::Min, false),
            (Selector::Max, true),
            (Selector::Sum, false),
            (Selector::Avg, true),
        ] {
            // Missing: the greatest unsigned long, as the greatest signed one.
            let source = UnsignedLongSort { selector, missing: -1 };
            let key = SortField::with_source("m", Arc::new(source), reverse);
            let signed = SortField {
                selector,
                missing: i64::MAX,
                ..SortField::numeric("m", SortType::Long, reverse)
            };
            let got = sorted(&reader, &q, &[key.clone(), SortField::doc()], 300, None);
            let want = sorted(&reader, &q, &[signed, SortField::doc()], 300, None);
            assert_eq!(docs(&got), docs(&want), "{selector:?}");
            for (g, w) in got.iter().zip(&want) {
                let w = if w.values[0] == i64::MAX { -1 } else { w.values[0] };
                // MIN's quirk aside, which a 0 among several values shows.
                if selector != Selector::Min {
                    assert_eq!(g.values[0], w, "{selector:?} doc {}", g.doc);
                }
            }
            let page = sorted(&reader, &q, &[key.clone(), SortField::doc()], 50, Some(&got[49]));
            assert_eq!(docs(&page), docs(&got[50..100]));
        }
        assert_eq!(UnsignedLongSort { selector: Selector::Min, missing: 0 }.value_string(-1), u64::MAX.to_string());
    }

    #[test]
    fn half_float_reads_sortable_shorts() {
        let reader = fixture();
        let q = query(None);
        for (selector, reverse) in [(Selector::Min, false), (Selector::Max, true), (Selector::Median, false)] {
            let source = HalfFloatSort { selector, missing: f32::INFINITY };
            let key = SortField::with_source("l", Arc::new(source), reverse);
            let signed = SortField {
                selector,
                missing: i64::MAX,
                ..SortField::numeric("l", SortType::Long, reverse)
            };
            let got = sorted(&reader, &q, &[key, SortField::doc()], 500, None);
            let want = sorted(&reader, &q, &[signed, SortField::doc()], 500, None);
            assert_eq!(docs(&got), docs(&want), "{selector:?}");
            for (g, w) in got.iter().zip(&want) {
                let f = if w.values[0] == i64::MAX {
                    f32::INFINITY
                } else {
                    HalfFloatPoint::sortable_short_to_half_float(w.values[0] as i16)
                };
                assert_eq!(g.values[0], i64::from(float_to_sortable_int(f)), "doc {}", g.doc);
            }
        }
        let s = HalfFloatSort { selector: Selector::Avg, missing: 0.0 };
        assert_eq!(s.value_string(i64::from(float_to_sortable_int(1.5))), "1.5");
        // A document's values averaged as doubles, then cast to float.
        let pick = Pick::HalfFloat(s);
        let raw = |f: f32| i64::from(HalfFloatPoint::half_float_to_sortable_short(f));
        let mut scratch = Vec::new();
        assert_eq!(pick.value(&[raw(1.0), raw(2.5)], &mut scratch), i64::from(float_to_sortable_int(1.75)));
        assert_eq!(pick.value(&[], &mut scratch), i64::from(float_to_sortable_int(0.0)));
    }

    #[test]
    fn wrong_doc_values_type_is_refused() {
        let dir = format!("{}/../../fixtures/data/keyword_sort_index", env!("CARGO_MANIFEST_DIR"));
        let reader = DirectoryReader::open(&FsDirectory::open(dir)).unwrap();
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        let key = SortField::with_source("k", Arc::new(UnsignedLongSort { selector: Selector::Min, missing: 0 }), false);
        let e = search_sorted(&segments, reader.segment_readers(), &query(None), &norms, &[key], 5, u64::MAX, None).unwrap_err();
        assert!(e.to_string().contains('k'), "{e}");
        let key = SortField::with_source("absent", Arc::new(HalfFloatSort { selector: Selector::Max, missing: 2.0 }), false);
        let hits = search_sorted(&segments, reader.segment_readers(), &query(None), &norms, &[key], 5, u64::MAX, None).unwrap().hits;
        assert!(hits.iter().all(|h| h.values[0] == i64::from(float_to_sortable_int(2.0))));
    }
}
