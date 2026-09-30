//! Range-field queries: `RangeFieldQuery` over the `2 * dims`-dimension
//! points of `IntRange`/`LongRange`/`FloatRange`/`DoubleRange`/
//! `InetAddressRange` (intersects, within, contains, crosses), and
//! `BinaryRangeFieldRangeQuery` over the same boxes stored as `BINARY` doc
//! values (`*RangeDocValuesField.newSlowIntersectsQuery`, through
//! `BinaryRangeDocValues`) -- with every range type's factories.
//!
//! The relation logic is `RangeFieldQuery.QueryType`'s, shared with the
//! writer side as [`RangeQueryType`].

use std::net::IpAddr;

use lucene_codecs::doc_values;
use lucene_codecs::field_infos::DocValuesType;
use lucene_codecs::points::{IntersectVisitor, Relation};
use lucene_index::document::{
    DoubleRange, FloatRange, InetAddressRange, IntRange, LongRange, RangeQueryType,
};

use super::{collect_live, field_info, reader, DocumentQuery};
use crate::collector::ScoringCollector;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

fn illegal(message: impl Into<String>) -> Error {
    Error::DocumentQuery(message.into())
}

/// `RangeFieldQuery`: the documents whose indexed box relates to the query
/// box `ranges` (every minimum, then every maximum) as `query_type` says, at
/// a constant score.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RangeFieldQuery {
    pub field: String,
    pub query_type: RangeQueryType,
    pub num_dims: usize,
    pub ranges: Vec<u8>,
    pub bytes_per_dim: usize,
}

impl RangeFieldQuery {
    /// `RangeFieldQuery(field, ranges, numDims, queryType)`.
    pub fn new(
        field: impl Into<String>,
        ranges: Vec<u8>,
        num_dims: usize,
        query_type: RangeQueryType,
    ) -> Result<Self> {
        if num_dims > 4 {
            return Err(illegal("dimension size cannot be greater than 4"));
        }
        if ranges.is_empty() {
            return Err(illegal("encoded ranges cannot be null or empty"));
        }
        let bytes_per_dim = ranges
            .len()
            .checked_div(num_dims.saturating_mul(2))
            .ok_or_else(|| illegal("numDims must be positive"))?;
        Ok(RangeFieldQuery {
            field: field.into(),
            query_type,
            num_dims,
            ranges,
            bytes_per_dim,
        })
    }
}

/// The `IntersectVisitor` of `RangeFieldQuery.createWeight`.
struct RangeVisitor<'q> {
    q: &'q RangeFieldQuery,
    docs: Vec<i32>,
}

impl IntersectVisitor for RangeVisitor<'_> {
    fn compare(&mut self, min_packed: &[u8], max_packed: &[u8]) -> Relation {
        self.q.query_type.compare(
            &self.q.ranges,
            min_packed,
            max_packed,
            self.q.num_dims,
            self.q.bytes_per_dim,
        )
    }

    fn visit(&mut self, doc_id: i32) {
        self.docs.push(doc_id);
    }

    fn visit_with_value(&mut self, doc_id: i32, packed_value: &[u8]) {
        if self.q.query_type.matches(
            &self.q.ranges,
            packed_value,
            self.q.num_dims,
            self.q.bytes_per_dim,
        ) {
            self.docs.push(doc_id);
        }
    }
}

impl DocumentQuery for RangeFieldQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        if info.point_dimension_count == 0 {
            return Ok(());
        }
        // `checkFieldInfo`.
        if usize::try_from(info.point_dimension_count / 2).ok() != Some(self.num_dims) {
            return Err(illegal(format!(
                "field=\"{}\" was indexed with numDims={} but this query has numDims={}",
                self.field,
                info.point_dimension_count / 2,
                self.num_dims
            )));
        }
        let r = reader(leaf)?;
        let points = r.points_reader()?;
        let Some(field) = points.field(info.number) else {
            return Ok(());
        };
        // Every document has a point, and the field's bounding box is inside
        // the query: every document matches.
        if field.doc_count == r.max_doc
            && self.query_type.compare(
                &self.ranges,
                &field.min_packed_value,
                &field.max_packed_value,
                self.num_dims,
                self.bytes_per_dim,
            ) == Relation::CellInsideQuery
        {
            for doc in 0..r.max_doc {
                collect_live(leaf, doc, boost, collector);
            }
            return Ok(());
        }
        let mut visitor = RangeVisitor {
            q: self,
            docs: Vec::new(),
        };
        points.intersect(info.number, &mut visitor)?;
        let mut docs = visitor.docs;
        docs.sort_unstable();
        docs.dedup();
        for doc in docs {
            collect_live(leaf, doc, boost, collector);
        }
        Ok(())
    }
}

/// `BinaryRangeDocValues`: a `BINARY` doc-values field read as range boxes
/// of `num_dims` dimensions of `num_bytes_per_dimension` bytes.
pub struct BinaryRangeDocValues<'a> {
    values: doc_values::BinaryReader<'a>,
    packed_len: usize,
}

impl std::fmt::Debug for BinaryRangeDocValues<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("BinaryRangeDocValues")
            .field("packed_len", &self.packed_len)
            .finish_non_exhaustive()
    }
}

impl<'a> BinaryRangeDocValues<'a> {
    pub fn new(
        values: doc_values::BinaryReader<'a>,
        num_dims: usize,
        num_bytes_per_dimension: usize,
    ) -> Self {
        BinaryRangeDocValues {
            values,
            packed_len: num_dims
                .saturating_mul(2)
                .saturating_mul(num_bytes_per_dimension),
        }
    }

    /// `advanceExact(doc)` then `getPackedValue()`: the document's box, or
    /// `None` without one.
    pub fn packed_value(&mut self, doc: i32) -> Result<Option<&'a [u8]>> {
        let Some(v) = self.values.value(doc)? else {
            return Ok(None);
        };
        v.get(..self.packed_len)
            .map(Some)
            .ok_or_else(|| illegal("a range doc value is shorter than its box"))
    }
}

/// `BinaryRangeFieldRangeQuery` (`*RangeDocValuesField.newSlowIntersectsQuery`):
/// the documents whose doc-values box intersects the query box, at a
/// constant score. `INTERSECTS` is the only relation it supports.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BinaryRangeFieldRangeQuery {
    pub field: String,
    pub query_packed_value: Vec<u8>,
    pub num_bytes_per_dimension: usize,
    pub num_dims: usize,
    pub query_type: RangeQueryType,
}

impl BinaryRangeFieldRangeQuery {
    pub fn new(
        field: impl Into<String>,
        query_packed_value: Vec<u8>,
        num_bytes_per_dimension: usize,
        num_dims: usize,
        query_type: RangeQueryType,
    ) -> Result<Self> {
        if query_type != RangeQueryType::Intersects {
            return Err(illegal(
                "INTERSECTS is the only query type supported for this field type right now",
            ));
        }
        Ok(BinaryRangeFieldRangeQuery {
            field: field.into(),
            query_packed_value,
            num_bytes_per_dimension,
            num_dims,
            query_type,
        })
    }
}

impl DocumentQuery for BinaryRangeFieldRangeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some(info) = field_info(leaf, &self.field)? else {
            return Ok(());
        };
        match info.doc_values_type {
            DocValuesType::None => return Ok(()),
            DocValuesType::Binary => {}
            other => {
                return Err(illegal(format!(
                    "unexpected docvalues type {} for field '{}' (expected=BINARY). Re-index with \
                     correct docvalues type.",
                    lucene_index::document::doc_values_type_name(other),
                    self.field
                )))
            }
        }
        let r = reader(leaf)?;
        let Some((meta, data)) = r.doc_values_for_field(info.number) else {
            return Ok(());
        };
        let Some(entry) = meta.binary_entry(info.number) else {
            return Ok(());
        };
        let mut values = BinaryRangeDocValues::new(
            doc_values::BinaryReader::new(data, entry),
            self.num_dims,
            self.num_bytes_per_dimension,
        );
        for doc in 0..r.max_doc {
            if let Some(packed) = values.packed_value(doc)? {
                if self.query_type.matches(
                    &self.query_packed_value,
                    packed,
                    self.num_dims,
                    self.num_bytes_per_dimension,
                ) {
                    collect_live(leaf, doc, boost, collector);
                }
            }
        }
        Ok(())
    }
}

fn check_args<T: PartialOrd>(min: &[T], max: &[T]) -> Result<()> {
    if min.is_empty() || max.is_empty() {
        return Err(illegal("min/max range values cannot be null or empty"));
    }
    if min.len() != max.len() {
        return Err(illegal("min/max ranges must agree"));
    }
    Ok(())
}

macro_rules! range_factories {
    ($(#[$m:meta])* $module:ident, $dv_module:ident, $ty:ident, $prim:ty) => {
        $(#[$m])*
        pub mod $module {
            use super::*;

            fn relation(
                field: &str,
                min: &[$prim],
                max: &[$prim],
                query_type: RangeQueryType,
            ) -> Result<RangeFieldQuery> {
                check_args(min, max)?;
                let ranges = $ty::encode(min, max).map_err(|e| illegal(e.to_string()))?;
                RangeFieldQuery::new(field, ranges, min.len(), query_type)
            }

            /// `newIntersectsQuery(field, min, max)`.
            pub fn new_intersects_query(field: &str, min: &[$prim], max: &[$prim]) -> Result<RangeFieldQuery> {
                relation(field, min, max, RangeQueryType::Intersects)
            }

            /// `newContainsQuery(field, min, max)`.
            pub fn new_contains_query(field: &str, min: &[$prim], max: &[$prim]) -> Result<RangeFieldQuery> {
                relation(field, min, max, RangeQueryType::Contains)
            }

            /// `newWithinQuery(field, min, max)`.
            pub fn new_within_query(field: &str, min: &[$prim], max: &[$prim]) -> Result<RangeFieldQuery> {
                relation(field, min, max, RangeQueryType::Within)
            }

            /// `newCrossesQuery(field, min, max)`.
            pub fn new_crosses_query(field: &str, min: &[$prim], max: &[$prim]) -> Result<RangeFieldQuery> {
                relation(field, min, max, RangeQueryType::Crosses)
            }
        }

        $(#[$m])*
        ///
        /// The doc-values twin's slow query.
        pub mod $dv_module {
            use super::*;

            /// `newSlowIntersectsQuery(field, min, max)`.
            pub fn new_slow_intersects_query(
                field: &str,
                min: &[$prim],
                max: &[$prim],
            ) -> Result<BinaryRangeFieldRangeQuery> {
                check_args(min, max)?;
                if min.iter().zip(max).any(|(lo, hi)| lo > hi) {
                    return Err(illegal("min should be less than max"));
                }
                let packed = $ty::encode(min, max).map_err(|e| illegal(e.to_string()))?;
                BinaryRangeFieldRangeQuery::new(
                    field,
                    packed,
                    $ty::BYTES,
                    min.len(),
                    RangeQueryType::Intersects,
                )
            }
        }
    };
}

range_factories!(
    /// `IntRange`'s queries.
    int_range, int_range_doc_values_field, IntRange, i32
);
range_factories!(
    /// `LongRange`'s queries.
    long_range, long_range_doc_values_field, LongRange, i64
);
range_factories!(
    /// `FloatRange`'s queries.
    float_range, float_range_doc_values_field, FloatRange, f32
);
range_factories!(
    /// `DoubleRange`'s queries.
    double_range, double_range_doc_values_field, DoubleRange, f64
);

/// `InetAddressRange`'s queries.
pub mod inet_address_range {
    use super::*;

    fn relation(
        field: &str,
        min: IpAddr,
        max: IpAddr,
        query_type: RangeQueryType,
    ) -> Result<RangeFieldQuery> {
        let ranges = InetAddressRange::encode(min, max).map_err(|e| illegal(e.to_string()))?;
        RangeFieldQuery::new(field, ranges, 1, query_type)
    }

    /// `newIntersectsQuery(field, min, max)`.
    pub fn new_intersects_query(field: &str, min: IpAddr, max: IpAddr) -> Result<RangeFieldQuery> {
        relation(field, min, max, RangeQueryType::Intersects)
    }

    /// `newContainsQuery(field, min, max)`.
    pub fn new_contains_query(field: &str, min: IpAddr, max: IpAddr) -> Result<RangeFieldQuery> {
        relation(field, min, max, RangeQueryType::Contains)
    }

    /// `newWithinQuery(field, min, max)`.
    pub fn new_within_query(field: &str, min: IpAddr, max: IpAddr) -> Result<RangeFieldQuery> {
        relation(field, min, max, RangeQueryType::Within)
    }

    /// `newCrossesQuery(field, min, max)`.
    pub fn new_crosses_query(field: &str, min: IpAddr, max: IpAddr) -> Result<RangeFieldQuery> {
        relation(field, min, max, RangeQueryType::Crosses)
    }
}
