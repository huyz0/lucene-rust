//! The point queries of the document package -- `PointRangeQuery` and
//! `PointInSetQuery` over packed values of any dimension count and width,
//! `IndexOrDocValuesQuery` -- and the factories of every point-indexed field:
//! `IntPoint`, `LongPoint`, `FloatPoint`, `DoublePoint`, `BinaryPoint`,
//! `InetAddressPoint`, and `IntField`/`LongField`/`FloatField`/`DoubleField`
//! (which pair the point query with a doc-values one, and sort by their
//! `SORTED_NUMERIC` values).
//!
//! `PointRangeQuery` and `PointInSetQuery` are the scorer tree's own
//! ([`crate::extended_query`]); as document-package queries they check the
//! field's point shape with Java's messages and run the tree's scorer
//! (`exec::ranges`). This module adds those checks and the typed encodings.

use std::net::IpAddr;

use lucene_index::document::{
    double_to_sortable_long, float_to_sortable_int, BinaryPoint, DoublePoint, FloatPoint,
    InetAddressPoint, IntPoint, LongPoint,
};

use super::doc_values_queries::{SortedNumericDocValuesRangeQuery, SortedNumericDocValuesSetQuery};
use super::{field_info, reader, Boosted, DocumentQuery, MatchNoDocs};
use crate::collector::ScoringCollector;
use crate::exec::{self, BoxScorer, Bulk, LeafContext, Mode};
pub use crate::extended_query::{PointInSetQuery, PointRangeQuery};
use crate::multi_segment::OpenSegment;
use crate::points_query::PointsInput;
use crate::top_field::{Selector, SortField, SortType};
use crate::{Error, Result};

fn illegal(message: impl Into<String>) -> Error {
    Error::DocumentQuery(message.into())
}

/// `PointRangeQuery.checkValidPointValues` and `PointInSetQuery`'s twin: the
/// segment's point shape against the query's. `None` when the segment has no
/// points for the field.
fn check_point_field(
    leaf: &OpenSegment<'_>,
    field: &str,
    num_dims: usize,
    bytes_per_dim: usize,
    what: &str,
) -> Result<Option<i32>> {
    let Some(info) = field_info(leaf, field)? else {
        return Ok(None);
    };
    if info.point_dimension_count == 0 {
        return Ok(None);
    }
    if usize::try_from(info.point_index_dimension_count).ok() != Some(num_dims) {
        return Err(illegal(format!(
            "field=\"{field}\" was indexed with {what}={} but this query has numDims={num_dims}",
            info.point_index_dimension_count
        )));
    }
    if usize::try_from(info.point_num_bytes).ok() != Some(bytes_per_dim) {
        return Err(illegal(format!(
            "field=\"{field}\" was indexed with bytesPerDim={} but this query has \
             bytesPerDim={bytes_per_dim}",
            info.point_num_bytes
        )));
    }
    Ok(Some(info.number))
}

/// Runs a scorer the tree builds over `leaf`'s points into `collector`,
/// live documents only: the `DocumentQuery` face of the scorer tree's point
/// queries.
fn score_points_leaf(
    leaf: &OpenSegment<'_>,
    collector: &mut dyn ScoringCollector,
    build: impl for<'c> FnOnce(&LeafContext<'c>, Mode) -> Result<Option<BoxScorer<'c>>>,
) -> Result<()> {
    let r = reader(leaf)?;
    let input = PointsInput {
        reader: r.points_reader()?,
        field_infos: r.field_infos(),
    };
    let ctx = LeafContext {
        points: Some(&input),
        max_doc: Some(r.max_doc),
        ..crate::aggs::plain_context(leaf)
    };
    let mode = Mode::of(&*collector);
    if let Some(scorer) = build(&ctx, mode)? {
        exec::score_segment(&mut Bulk::scorer(scorer), mode, leaf.live_docs, collector)?;
    }
    Ok(())
}

/// `PointRangeQuery` as a document-package query: the field's shape checked
/// with Java's messages, then the scorer tree's `PointRangeQuery`
/// ([`crate::extended_query::PointRangeQuery`], which this is).
impl DocumentQuery for PointRangeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        if check_point_field(
            leaf,
            &self.field,
            self.num_dims,
            self.bytes_per_dim,
            "numIndexDimensions",
        )?
        .is_none()
        {
            return Ok(());
        }
        score_points_leaf(leaf, collector, |ctx, mode| {
            exec::ranges::point_range(ctx, self, boost, mode)
        })
    }
}

/// `PointInSetQuery` as a document-package query, as [`PointRangeQuery`].
impl DocumentQuery for PointInSetQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        if check_point_field(
            leaf,
            &self.field,
            self.num_dims,
            self.bytes_per_dim,
            "numIndexDims",
        )?
        .is_none()
        {
            return Ok(());
        }
        score_points_leaf(leaf, collector, |ctx, mode| {
            exec::ranges::point_in_set(ctx, self, boost, mode)
        })
    }
}

/// `IndexOrDocValuesQuery`: the same hits two ways -- the points query and
/// the doc-values one. A segment missing either side's structure matches
/// nothing (Java's `null` scorer supplier); otherwise the points side runs
/// (see the module doc of [`super`]).
#[derive(Debug)]
pub struct IndexOrDocValuesQuery {
    pub field: String,
    pub index_query: Box<dyn DocumentQuery>,
    pub dv_query: Box<dyn DocumentQuery>,
}

impl IndexOrDocValuesQuery {
    pub fn new(
        field: impl Into<String>,
        index_query: Box<dyn DocumentQuery>,
        dv_query: Box<dyn DocumentQuery>,
    ) -> Self {
        IndexOrDocValuesQuery {
            field: field.into(),
            index_query,
            dv_query,
        }
    }
}

impl DocumentQuery for IndexOrDocValuesQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        let has_points = info.point_dimension_count != 0;
        let has_dv = info.doc_values_type != lucene_codecs::field_infos::DocValuesType::None;
        match (has_points, has_dv) {
            (true, true) => self.index_query.score_leaf(leaf, boost, collector),
            (true, false) | (false, true) => {
                // One side's structure is missing from this segment: Java's
                // weight returns no scorer at all.
                Ok(())
            }
            (false, false) => Ok(()),
        }
    }
}

fn check_same_len<T>(lower: &[T], upper: &[T]) -> Result<()> {
    if lower.len() != upper.len() {
        return Err(illegal(format!(
            "lowerPoint has length={} but upperPoint has different length={}",
            lower.len(),
            upper.len()
        )));
    }
    Ok(())
}

macro_rules! numeric_point_factories {
    ($(#[$m:meta])* $module:ident, $point:ident, $prim:ty) => {
        $(#[$m])*
        pub mod $module {
            use super::*;

            /// `newExactQuery(field, value)`.
            pub fn new_exact_query(field: &str, value: $prim) -> Result<PointRangeQuery> {
                new_range_query(field, value, value)
            }

            /// `newRangeQuery(field, lowerValue, upperValue)`: inclusive.
            pub fn new_range_query(field: &str, lower: $prim, upper: $prim) -> Result<PointRangeQuery> {
                new_range_query_nd(field, &[lower], &[upper])
            }

            /// `newRangeQuery(field, lowerValue[], upperValue[])`.
            pub fn new_range_query_nd(
                field: &str,
                lower: &[$prim],
                upper: &[$prim],
            ) -> Result<PointRangeQuery> {
                check_same_len(lower, upper)?;
                let lo = $point::pack(lower).map_err(|e| illegal(e.to_string()))?;
                let hi = $point::pack(upper).map_err(|e| illegal(e.to_string()))?;
                PointRangeQuery::new(field, lower.len(), lo, hi)
            }

            /// `newSetQuery(field, values...)`.
            pub fn new_set_query(field: &str, values: &[$prim]) -> Result<PointInSetQuery> {
                let mut sorted = values.to_vec();
                sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
                PointInSetQuery::new(
                    field,
                    1,
                    $point::BYTES,
                    sorted
                        .into_iter()
                        .map(|v| $point::encode_dimension(v).to_vec())
                        .collect::<Vec<Vec<u8>>>(),
                )
            }
        }
    };
}

numeric_point_factories!(
    /// `IntPoint`'s queries.
    int_point, IntPoint, i32
);
numeric_point_factories!(
    /// `LongPoint`'s queries.
    long_point, LongPoint, i64
);
numeric_point_factories!(
    /// `FloatPoint`'s queries (`FloatPoint.nextUp`/`nextDown` for exclusive
    /// bounds are `lucene_index::document::FloatPoint`'s).
    float_point, FloatPoint, f32
);
numeric_point_factories!(
    /// `DoublePoint`'s queries.
    double_point, DoublePoint, f64
);

/// `BinaryPoint`'s queries.
pub mod binary_point {
    use super::*;

    /// `newExactQuery(field, value)`.
    pub fn new_exact_query(field: &str, value: &[u8]) -> Result<PointRangeQuery> {
        new_range_query(field, value, value)
    }

    /// `newRangeQuery(field, lowerValue, upperValue)`: one dimension.
    pub fn new_range_query(field: &str, lower: &[u8], upper: &[u8]) -> Result<PointRangeQuery> {
        new_range_query_nd(field, &[lower], &[upper])
    }

    /// `newRangeQuery(field, lowerValue[][], upperValue[][])`.
    pub fn new_range_query_nd(
        field: &str,
        lower: &[&[u8]],
        upper: &[&[u8]],
    ) -> Result<PointRangeQuery> {
        let lo = BinaryPoint::pack(lower).map_err(|e| illegal(e.to_string()))?;
        let hi = BinaryPoint::pack(upper).map_err(|e| illegal(e.to_string()))?;
        PointRangeQuery::new(field, lower.len(), lo, hi)
    }

    /// `newSetQuery(field, values...)`: one-dimension values of one width; an
    /// empty set matches nothing.
    pub fn new_set_query(field: &str, values: &[&[u8]]) -> Result<Box<dyn DocumentQuery>> {
        let Some(first) = values.first() else {
            return Ok(Box::new(MatchNoDocs));
        };
        if let Some(bad) = values.iter().find(|v| v.len() != first.len()) {
            return Err(illegal(format!(
                "all byte[] must be the same length, but saw {} and {}",
                first.len(),
                bad.len()
            )));
        }
        Ok(Box::new(PointInSetQuery::new(
            field,
            1,
            first.len(),
            values.iter().map(|v| v.to_vec()).collect::<Vec<Vec<u8>>>(),
        )?))
    }
}

/// `InetAddressPoint`'s queries.
pub mod inet_address_point {
    use super::*;

    /// `newExactQuery(field, value)`.
    pub fn new_exact_query(field: &str, value: IpAddr) -> Result<PointRangeQuery> {
        new_range_query(field, value, value)
    }

    /// `newPrefixQuery(field, value, prefixLength)`: the addresses sharing
    /// the first `prefix_length` bits, in the address's own family.
    pub fn new_prefix_query(
        field: &str,
        value: IpAddr,
        prefix_length: u32,
    ) -> Result<PointRangeQuery> {
        let (lo, hi) = InetAddressPoint::prefix_bounds(value, prefix_length)
            .map_err(|e| illegal(e.to_string()))?;
        new_range_query(field, lo, hi)
    }

    /// `newRangeQuery(field, lowerValue, upperValue)`: inclusive.
    pub fn new_range_query(field: &str, lower: IpAddr, upper: IpAddr) -> Result<PointRangeQuery> {
        PointRangeQuery::new(
            field,
            1,
            InetAddressPoint::encode(lower).to_vec(),
            InetAddressPoint::encode(upper).to_vec(),
        )
    }

    /// `newSetQuery(field, values...)`.
    pub fn new_set_query(field: &str, values: &[IpAddr]) -> Result<PointInSetQuery> {
        PointInSetQuery::new(
            field,
            1,
            InetAddressPoint::BYTES,
            values
                .iter()
                .map(|v| InetAddressPoint::encode(*v).to_vec())
                .collect::<Vec<Vec<u8>>>(),
        )
    }
}

/// `SortedNumericSelector.Type`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NumericSelector {
    Min,
    Max,
}

fn sort_field(
    field: &str,
    ty: SortType,
    reverse: bool,
    selector: NumericSelector,
    missing: i64,
) -> SortField {
    let mut s = SortField::numeric(field, ty, reverse);
    s.selector = match selector {
        NumericSelector::Min => Selector::Min,
        NumericSelector::Max => Selector::Max,
    };
    s.missing = missing;
    s
}

macro_rules! numeric_field_factories {
    (
        $(#[$m:meta])* $module:ident, $points:ident, $prim:ty, $sort:expr, $to_dv:expr
        $(, $extra:item)*
    ) => {
        $(#[$m])*
        pub mod $module {
            use super::*;
            $($extra)*

            /// `newExactQuery(field, value)`.
            pub fn new_exact_query(field: &str, value: $prim) -> Result<IndexOrDocValuesQuery> {
                new_range_query(field, value, value)
            }

            /// `newRangeQuery(field, lowerValue, upperValue)`: the point range
            /// or, equivalently, the `SORTED_NUMERIC` doc-values range.
            pub fn new_range_query(
                field: &str,
                lower: $prim,
                upper: $prim,
            ) -> Result<IndexOrDocValuesQuery> {
                let dv_lo: i64 = ($to_dv)(lower);
                let dv_hi: i64 = ($to_dv)(upper);
                Ok(IndexOrDocValuesQuery::new(
                    field,
                    Box::new(super::$points::new_range_query(field, lower, upper)?),
                    Box::new(SortedNumericDocValuesRangeQuery::new(field, dv_lo, dv_hi)),
                ))
            }

            /// `newSetQuery(field, values...)`.
            pub fn new_set_query(field: &str, values: &[$prim]) -> Result<IndexOrDocValuesQuery> {
                let dv: Vec<i64> = values.iter().map(|&v| ($to_dv)(v)).collect();
                Ok(IndexOrDocValuesQuery::new(
                    field,
                    Box::new(super::$points::new_set_query(field, values)?),
                    Box::new(SortedNumericDocValuesSetQuery::new(field, dv)),
                ))
            }

            /// `newSortField(field, reverse, selector)`: a
            /// `SortedNumericSortField`, missing values sorting as `0`.
            pub fn new_sort_field(field: &str, reverse: bool, selector: NumericSelector) -> SortField {
                sort_field(field, $sort, reverse, selector, 0)
            }

            /// `newSortField(field, reverse, selector, missingValue)`.
            pub fn new_sort_field_with_missing(
                field: &str,
                reverse: bool,
                selector: NumericSelector,
                missing: $prim,
            ) -> SortField {
                sort_field(field, $sort, reverse, selector, ($to_dv)(missing))
            }
        }
    };
}

numeric_field_factories!(
    /// `IntField`'s queries and sort field.
    int_field, int_point, i32, SortType::Int, i64::from
);
numeric_field_factories!(
    /// `LongField`'s queries, sort field and distance feature query.
    long_field,
    long_point,
    i64,
    SortType::Long,
    |v: i64| v,
    /// `newDistanceFeatureQuery(field, weight, origin, pivotDistance)`: a
    /// [`LongDistanceFeatureQuery`](super::super::LongDistanceFeatureQuery),
    /// boosted by `weight` unless it is 1.
    pub fn new_distance_feature_query(
        field: &str,
        weight: f32,
        origin: i64,
        pivot_distance: i64,
    ) -> Result<Box<dyn DocumentQuery>> {
        let q = super::super::LongDistanceFeatureQuery::new(field, origin, pivot_distance)?;
        Ok(boxed_boost(Box::new(q), weight))
    }
);
numeric_field_factories!(
    /// `FloatField`'s queries and sort field (doc values are
    /// `floatToSortableInt`).
    float_field,
    float_point,
    f32,
    SortType::Float,
    |v: f32| i64::from(float_to_sortable_int(v))
);
numeric_field_factories!(
    /// `DoubleField`'s queries and sort field (doc values are
    /// `doubleToSortableLong`).
    double_field,
    double_point,
    f64,
    SortType::Double,
    double_to_sortable_long
);

pub(crate) fn boxed_boost(q: Box<dyn DocumentQuery>, weight: f32) -> Box<dyn DocumentQuery> {
    if weight != 1.0 {
        Box::new(Boosted::new(q, weight))
    } else {
        q
    }
}
