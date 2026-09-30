//! Port of `org.apache.lucene.index.SegmentOrder`: a view of a reader with
//! its segments reordered by a numeric sort's primary key, so the segments
//! most likely to hold the top hits are searched first and early termination
//! skips more of the rest.
//!
//! Each segment's sort value is its minimum (ascending) or maximum
//! (`reverse`) for the field -- from the doc-values skip index when the field
//! has one (`DocValuesSkipper`), else from its points' bounds
//! (`PointValues`), widened by the sort's missing value when some document
//! lacks the field. A segment with neither sorts as `Long.MIN_VALUE`
//! ascending, `Long.MAX_VALUE` descending. Segments with equal values keep
//! their order (Java's `Arrays.sort` over objects is stable, as `sort_by`
//! is).
//!
//! # What differs from Java
//!
//! - Java's `SortField.getMissingValue()` is `null` unless set, and `null`
//!   means "ignore documents without a value". [`SortField::missing`] is an
//!   `i64` defaulting to `0`, which cannot say "unset", so
//!   [`SegmentOrder::from_sort`] takes whether it was set.
//! - A non-numeric primary sort returns the reader itself in Java; here
//!   [`SegmentOrder::reorder`] returns a view in the same order (a
//!   [`DirectoryReader`] is not shared by reference).
//! - Java caches each segment's value per core key across `reorder` calls;
//!   the values are recomputed per call here (one metadata lookup per
//!   segment).

use crate::directory_reader::{DirectoryReader, SegmentReader};
use crate::top_field::{SortField, SortType};
use lucene_codecs::field_infos::DocValuesSkipIndexType;

/// The numeric key a [`SegmentOrder`] sorts segments by.
#[derive(Debug, Clone, PartialEq, Eq)]
struct NumericOrder {
    field: String,
    /// The missing value as a comparable long, or `None` when unset.
    missing: Option<i64>,
    reverse: bool,
    /// 4 for `INT`/`FLOAT` (`sortableBytesToInt`), 8 for `LONG`/`DOUBLE`
    /// (`sortableBytesToLong`).
    point_bytes: usize,
}

/// `SegmentOrder`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmentOrder {
    numeric: Option<NumericOrder>,
}

/// `NumericUtils.sortableBytesToInt(b, 0)` / `sortableBytesToLong(b, 0)`, as
/// a long; `None` for a value shorter than `len`.
fn decode_point(b: &[u8], len: usize) -> Option<i64> {
    match len {
        4 => {
            let bytes: [u8; 4] = b.get(..4)?.try_into().ok()?;
            Some(i64::from(i32::from_be_bytes(bytes) ^ i32::MIN))
        }
        _ => {
            let bytes: [u8; 8] = b.get(..8)?.try_into().ok()?;
            Some(i64::from_be_bytes(bytes) ^ i64::MIN)
        }
    }
}

impl NumericOrder {
    /// `NumericFieldReaderContextComparator.loadSortValue`.
    fn sort_value(&self, segment: &SegmentReader) -> i64 {
        let none = if self.reverse { i64::MAX } else { i64::MIN };
        let Some(info) = segment.field_infos().field_by_name(&self.field) else {
            return none;
        };
        let widen = |bound: i64, doc_count: i32| match self.missing {
            Some(missing) if doc_count != segment.max_doc => {
                if self.reverse {
                    bound.max(missing)
                } else {
                    bound.min(missing)
                }
            }
            _ => bound,
        };
        // `reader.getDocValuesSkipper(field)`: only a field indexed with a
        // skip index has one.
        if info.doc_values_skip_index_type != DocValuesSkipIndexType::None {
            if let Some(skipper) = segment
                .doc_values_for_field(info.number)
                .and_then(|(meta, _)| meta.skipper_meta(info.number))
            {
                let bound = if self.reverse {
                    skipper.max_value
                } else {
                    skipper.min_value
                };
                return widen(bound, skipper.doc_count);
            }
        }
        // `reader.getPointValues(field)`. A `.kdm` that fails to decode is
        // Java's caught `IOException`: as if there were nothing to read.
        if let Some(points) = segment.points_field(info.number) {
            let packed = if self.reverse {
                &points.max_packed_value
            } else {
                &points.min_packed_value
            };
            return match decode_point(packed, self.point_bytes) {
                Some(bound) => widen(bound, points.doc_count),
                None => none,
            };
        }
        none
    }
}

impl SegmentOrder {
    /// `SegmentOrder.fromSort(sort)` for the sort's primary key: a numeric
    /// key (`INT`/`LONG`/`FLOAT`/`DOUBLE`, `SortedNumericSortField`
    /// included) orders segments by it; anything else leaves them as they
    /// are. `missing_value_set` says whether the key's
    /// [`SortField::missing`] was set (`getMissingValue() != null`); it is
    /// already the comparable long `fromSort` converts to
    /// (`floatToSortableInt`, `doubleToSortableLong`).
    pub fn from_sort(primary: &SortField, missing_value_set: bool) -> Self {
        let point_bytes = match primary.ty {
            SortType::Int | SortType::Float => 4,
            SortType::Long | SortType::Double => 8,
            SortType::Score
            | SortType::Doc
            | SortType::String
            | SortType::StringVal
            | SortType::Custom(_) => return SegmentOrder { numeric: None },
        };
        SegmentOrder {
            numeric: Some(NumericOrder {
                field: primary.field.clone(),
                missing: missing_value_set.then_some(primary.missing),
                reverse: primary.reverse,
                point_bytes,
            }),
        }
    }

    /// `reorder(reader)`: `reader`'s segments, sorted by each one's value
    /// for the key (ascending, or descending when reversed).
    pub fn reorder(&self, reader: &DirectoryReader) -> DirectoryReader {
        let segments = reader.segment_readers();
        let mut order: Vec<usize> = (0..segments.len()).collect();
        if let Some(numeric) = &self.numeric {
            let values: Vec<i64> = segments.iter().map(|s| numeric.sort_value(s)).collect();
            order.sort_by(|&a, &b| {
                if numeric.reverse {
                    values[b].cmp(&values[a])
                } else {
                    values[a].cmp(&values[b])
                }
            });
        }
        reader.with_segment_order(&order)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn points_decode_as_sortable_ints_and_longs() {
        assert_eq!(decode_point(&[0x80, 0, 0, 5], 4), Some(5));
        assert_eq!(decode_point(&[0x7f, 0xff, 0xff, 0xfb], 4), Some(-5));
        assert_eq!(decode_point(&[0x80, 0, 0, 0, 0, 0, 0, 7], 8), Some(7));
        assert_eq!(decode_point(&[0, 0, 0, 0, 0, 0, 0, 0], 8), Some(i64::MIN));
        assert_eq!(decode_point(&[0x80, 0], 4), None);
        assert_eq!(decode_point(&[0x80, 0, 0, 0], 8), None);
    }

    #[test]
    fn a_non_numeric_primary_sort_orders_nothing() {
        for ty in [SortType::Score, SortType::Doc, SortType::String] {
            let sf = SortField {
                ty,
                ..SortField::numeric("f", SortType::Long, false)
            };
            assert_eq!(
                SegmentOrder::from_sort(&sf, true),
                SegmentOrder { numeric: None }
            );
        }
        let long = SegmentOrder::from_sort(&SortField::numeric("f", SortType::Long, true), false);
        assert_eq!(
            long.numeric,
            Some(NumericOrder {
                field: "f".into(),
                missing: None,
                reverse: true,
                point_bytes: 8,
            })
        );
    }
}
