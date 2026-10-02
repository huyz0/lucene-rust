//! The geo point and shape queries, sorts and nearest-neighbour search of Lucene's
//! `document` package, over the fields `lucene_index::document` writes
//! (`LatLonPoint`, `LatLonDocValuesField`, `XYPointField`,
//! `XYDocValuesField`):
//!
//! - the [`lat_lon_point`] module: `LatLonPoint.newBoxQuery` (a two-dimension
//!   `PointRangeQuery`, two of them across the dateline),
//!   `newDistanceQuery` ([`LatLonPointDistanceQuery`]), `newPolygonQuery` /
//!   `newGeometryQuery` ([`LatLonPointQuery`], `SpatialQuery`'s point
//!   visitor), `newDistanceFeatureQuery`
//!   ([`LatLonPointDistanceFeatureQuery`]) and `nearest` ([`nearest()`],
//!   `NearestNeighbor`);
//! - the [`lat_lon_doc_values_field`] module: `newSlowBoxQuery`
//!   ([`LatLonDocValuesBoxQuery`]), `newSlowDistanceQuery` /
//!   `newSlowPolygonQuery` / `newSlowGeometryQuery` ([`LatLonDocValuesQuery`])
//!   and `newDistanceSort` ([`LatLonPointSortField`]);
//! - the [`xy_point_field`] module: every query is an [`XYPointInGeometryQuery`];
//! - the [`xy_doc_values_field`] module: every query is an
//!   [`XYDocValuesPointInGeometryQuery`]; `newDistanceSort` is
//!   [`XYPointSortField`].
//!
//! and the shape queries, over the triangles and shape doc values
//! `lucene_index::document::{LatLonShape, XYShape}` write:
//!
//! - the [`lat_lon_shape`] module: `newBoxQuery`
//!   ([`LatLonShapeBoundingBoxQuery`], related in the encoded space; a
//!   `CONTAINS` box across the dateline is a [`MustConjunction`] of its
//!   halves), `newGeometryQuery` and its line/polygon/point/distance forms
//!   ([`LatLonShapeQuery`]), `newSlowDocValuesBoxQuery`
//!   ([`LatLonShapeDocValuesQuery`]);
//! - the [`xy_shape`] module: [`XYShapeQuery`] and [`XYShapeDocValuesQuery`].
//!
//! Each query follows its Java class's `Weight`: the same cell relations
//! (in the encoded space where Java relates encoded bounds), the same
//! per-point predicates (`GeoEncodingUtils`' grid predicates), the same
//! choice between a forward and an inverse tree walk, and the same leaf
//! handling, which matters because a cell's relation and a point's predicate
//! can disagree by a quantum at an edge. Hits are collected into a bitset
//! (Java's `DocIdSetBuilder`/`FixedBitSet`) and handed to the collector in
//! doc-id order at the constant score.
//!
//! # Deviations
//!
//! - `ShapeField.QueryRelation` lives here as [`QueryRelation`] (the rest of
//!   `ShapeField`, the encoding, is `lucene_index::document::ShapeField`).
//! - `equals`/`hashCode`/`toString`, `QueryVisitor` and `explain` are not
//!   ported, as for the rest of the package.

mod distance_feature;
mod doc_values;
pub mod geo3d;
mod nearest;
mod point_queries;
mod shape_queries;
mod sort;

use lucene_codecs::doc_values::{NumericReader, SortedNumericReader};
use lucene_codecs::field_infos::{DocValuesType, FieldInfo};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::geo::GeoError;

use super::{collect_live, field_info, reader, DocumentQuery};
use crate::collector::ScoringCollector;
use crate::directory_reader::SegmentReader;
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

pub use distance_feature::LatLonPointDistanceFeatureQuery;
pub use doc_values::{
    LatLonDocValuesBoxQuery, LatLonDocValuesQuery, XYDocValuesPointInGeometryQuery,
};
pub use geo3d::PointInGeo3DShapeQuery;
pub use nearest::{nearest, NearestHit, NearestHits};
pub use point_queries::{LatLonPointDistanceQuery, LatLonPointQuery, XYPointInGeometryQuery};
pub use shape_queries::{
    EncodedRectangle, LatLonShapeBoundingBoxQuery, LatLonShapeDocValuesQuery, LatLonShapeQuery,
    XYShapeDocValuesQuery, XYShapeQuery,
};
pub use sort::{
    Geo3DPointOutsideSortField, Geo3DPointSortField, LatLonPointSortField, SortedDistance,
    XYPointSortField,
};

fn illegal(message: impl Into<String>) -> Error {
    Error::DocumentQuery(message.into())
}

/// A geometry error as the query error Java's `IllegalArgumentException`
/// becomes.
fn geo(e: GeoError) -> Error {
    illegal(e.to_string())
}

/// `ShapeField.QueryRelation`: how an indexed geometry relates to the query
/// geometry for it to match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum QueryRelation {
    /// The indexed geometry shares a point with the query.
    Intersects,
    /// The indexed geometry lies within the query.
    Within,
    /// The indexed geometry shares no point with the query.
    Disjoint,
    /// The indexed geometry contains the query.
    Contains,
}

impl std::fmt::Display for QueryRelation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self {
            QueryRelation::Intersects => "INTERSECTS",
            QueryRelation::Within => "WITHIN",
            QueryRelation::Disjoint => "DISJOINT",
            QueryRelation::Contains => "CONTAINS",
        })
    }
}

/// A segment's `SORTED_NUMERIC` (or `NUMERIC`, as a singleton) doc values
/// for one field, read forward (`DocValues.getSortedNumeric`).
pub(crate) enum GeoValues<'a> {
    Numeric(NumericReader<'a>),
    Sorted(SortedNumericReader<'a>),
}

impl GeoValues<'_> {
    /// `cost()`: the documents with a value.
    pub(crate) fn cost(&self) -> i64 {
        match self {
            GeoValues::Numeric(r) => r.entry().num_values,
            GeoValues::Sorted(r) => i64::from(r.entry().num_docs_with_field),
        }
    }

    /// Replaces `out` with `doc`'s values (ascending; empty without any).
    pub(crate) fn values(&mut self, doc: i32, out: &mut Vec<i64>) -> Result<()> {
        out.clear();
        match self {
            GeoValues::Numeric(r) => {
                if let Some(v) = r.value(doc)? {
                    out.push(v);
                }
            }
            GeoValues::Sorted(r) => r.values(doc, out)?,
        }
        Ok(())
    }
}

/// `DocValues.getSortedNumeric(reader, field)`: the field's values, `None`
/// when the segment has none for it; a field of another doc-values type is
/// `DocValues.checkField`'s error.
pub(crate) fn sorted_numeric<'a>(
    leaf: &OpenSegment<'a>,
    info: &FieldInfo,
) -> Result<Option<GeoValues<'a>>> {
    sorted_numeric_in(reader(leaf)?, info)
}

/// [`sorted_numeric`] over a segment reader.
pub(crate) fn sorted_numeric_in<'a>(
    r: &'a SegmentReader,
    info: &FieldInfo,
) -> Result<Option<GeoValues<'a>>> {
    match info.doc_values_type {
        DocValuesType::None => return Ok(None),
        DocValuesType::Numeric | DocValuesType::SortedNumeric => {}
        other => {
            return Err(illegal(format!(
                "unexpected docvalues type {} for field '{}' (expected one of [SORTED_NUMERIC, \
                 NUMERIC]). Re-index with correct docvalues type.",
                lucene_index::document::doc_values_type_name(other),
                info.name
            )))
        }
    }
    let Some((meta, data)) = r.doc_values_for_field(info.number) else {
        return Ok(None);
    };
    if let Some(e) = meta.numeric_entry(info.number) {
        return Ok(Some(GeoValues::Numeric(NumericReader::new(data, e))));
    }
    Ok(meta
        .sorted_numeric_entry(info.number)
        .map(|e| GeoValues::Sorted(SortedNumericReader::new(data, e))))
}

/// `LeafReader.getSortedNumericDocValues(field)`: only a `SORTED_NUMERIC`
/// field has them (anything else is `null`, no error).
pub(crate) fn sorted_numeric_only<'a>(
    leaf: &OpenSegment<'a>,
    field: &str,
) -> Result<Option<GeoValues<'a>>> {
    match field_info(leaf, field)? {
        Some(info) if info.doc_values_type == DocValuesType::SortedNumeric => {
            sorted_numeric(leaf, info)
        }
        _ => Ok(None),
    }
}

/// Every live document of `bits`, ascending, at `score`.
pub(crate) fn collect_bits(
    leaf: &OpenSegment<'_>,
    bits: &FixedBitSet,
    score: f32,
    collector: &mut dyn ScoringCollector,
) {
    bits.for_each_set_bit(|doc| {
        if let Ok(doc) = i32::try_from(doc) {
            collect_live(leaf, doc, score, collector);
        }
    });
}

/// A non-negative count (`maxDoc`) as a bitset length.
#[inline]
pub(crate) fn idx(doc: i32) -> usize {
    usize::try_from(doc).unwrap_or(0)
}

/// `bits.set(doc)` for a doc id the tree handed out. One outside the
/// segment -- a corrupt `.kdd`; Java's `FixedBitSet` throws -- sets nothing
/// and is remembered in `bad` (the first one), which the walk's caller
/// turns into an error with [`check_walk`].
#[inline]
pub(crate) fn set_doc(bits: &mut FixedBitSet, doc: i32, bad: &mut Option<i32>) {
    match usize::try_from(doc) {
        Ok(d) if d < bits.len() => bits.set(d),
        _ => {
            bad.get_or_insert(doc);
        }
    }
}

/// `bits.clear(doc)`, checked as [`set_doc`].
#[inline]
pub(crate) fn clear_doc(bits: &mut FixedBitSet, doc: i32, bad: &mut Option<i32>) {
    match usize::try_from(doc) {
        Ok(d) if d < bits.len() => bits.clear(d),
        _ => {
            bad.get_or_insert(doc);
        }
    }
}

/// The corruption error for a points walk over `field` that named document
/// `doc` in a segment of `max_doc` documents.
pub(crate) fn out_of_segment(field: &str, doc: i32, max_doc: i32) -> Error {
    Error::Store(lucene_store::Error::Corrupted(format!(
        "points of field {field} name document {doc}, outside the segment's 0..{max_doc}"
    )))
}

/// After a walk: the error for the first out-of-segment doc id it met.
pub(crate) fn check_walk(bad: Option<i32>, field: &str, max_doc: i32) -> Result<()> {
    match bad {
        Some(doc) => Err(out_of_segment(field, doc, max_doc)),
        None => Ok(()),
    }
}

/// `bits.get(doc)`, bounded as [`set_doc`] (outside: `false`; the walk
/// records such a doc id when it sets or clears it).
#[inline]
pub(crate) fn get_doc(bits: &FixedBitSet, doc: i32) -> bool {
    usize::try_from(doc).is_ok_and(|d| d < bits.len() && bits.get(d))
}

/// `ConstantScoreQuery` over a `BooleanQuery` of `SHOULD` (`must ==
/// false`) or `MUST` clauses: what `LatLonPoint.newBoxQuery` builds across
/// the dateline and `newGeometryQuery` for a `CONTAINS` of several points.
/// Every match scores the boost.
#[derive(Debug)]
pub struct ConstantScoreBoolean {
    pub clauses: Vec<Box<dyn DocumentQuery>>,
    pub must: bool,
}

/// A collector keeping a segment's hits as a bitset.
struct BitsCollector(FixedBitSet);

impl ScoringCollector for BitsCollector {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        // The clauses collect only documents of this segment.
        let mut bad = None;
        set_doc(&mut self.0, doc_id, &mut bad);
        debug_assert!(
            bad.is_none(),
            "collected document {doc_id} outside the segment"
        );
    }
}

impl DocumentQuery for ConstantScoreBoolean {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let max_doc = idx(reader(leaf)?.max_doc);
        let mut acc: Option<FixedBitSet> = None;
        for clause in &self.clauses {
            let mut c = BitsCollector(FixedBitSet::new(max_doc));
            clause.score_leaf(leaf, 1.0, &mut c)?;
            acc = Some(match acc {
                None => c.0,
                Some(mut a) => {
                    if self.must {
                        a.and(&c.0);
                    } else {
                        a.or(&c.0);
                    }
                    a
                }
            });
        }
        if let Some(bits) = acc {
            // Every clause already dropped deleted documents.
            collect_bits(leaf, &bits, boost, collector);
        }
        Ok(())
    }
}

/// A `BooleanQuery` of `MUST` constant-score clauses that is *not* wrapped
/// in a `ConstantScoreQuery` -- what `LatLonShape.newBoxQuery` builds for a
/// `CONTAINS` box across the dateline: every match scores the sum of its
/// clauses' scores (`ConjunctionScorer`, summed in `double`). For that box
/// no document matches in practice -- each half reaches +-180, which a
/// shape's boundary cannot pass, and a boundary touching the box is
/// `NOTWITHIN` -- so the sum is checked by unit tests, not against Lucene.
#[derive(Debug)]
pub struct MustConjunction {
    pub clauses: Vec<Box<dyn DocumentQuery>>,
}

impl DocumentQuery for MustConjunction {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let mut sum = 0.0f64;
        for _ in &self.clauses {
            sum += f64::from(boost);
        }
        let mut hits = BitsCollector(FixedBitSet::new(idx(reader(leaf)?.max_doc)));
        let mut acc: Option<FixedBitSet> = None;
        for clause in &self.clauses {
            hits.0.clear_all();
            clause.score_leaf(leaf, 1.0, &mut hits)?;
            acc = Some(match acc {
                None => hits.0.clone(),
                Some(mut a) => {
                    a.and(&hits.0);
                    a
                }
            });
        }
        if let Some(bits) = acc {
            collect_bits(leaf, &bits, sum as f32, collector);
        }
        Ok(())
    }
}

/// `LatLonPoint`'s query factories and `nearest`.
pub mod lat_lon_point {
    use lucene_index::document::{int_to_sortable_bytes, LatLonPoint};
    use lucene_util::geo::{LatLonGeometry, Polygon};

    use super::super::{Boosted, MatchNoDocs, PointRangeQuery};
    use super::*;

    fn encode(e: lucene_index::document::Error) -> Error {
        illegal(e.to_string())
    }

    /// `newBoxQuery(field, minLatitude, maxLatitude, minLongitude,
    /// maxLongitude)`: a box that crosses the dateline when `max_longitude <
    /// min_longitude`.
    ///
    /// # Errors
    /// An invalid coordinate, with Java's message.
    pub fn new_box_query(
        field: &str,
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        let mut min_longitude = min_longitude;
        // Exact comparisons, as Java's: the quantized edges cannot be
        // reached any other way.
        #[allow(clippy::float_cmp)]
        {
            if min_latitude == 90.0 {
                return Ok(Box::new(MatchNoDocs));
            }
            if min_longitude == 180.0 {
                if max_longitude == 180.0 {
                    return Ok(Box::new(MatchNoDocs));
                } else if max_longitude < min_longitude {
                    min_longitude = -180.0;
                }
            }
        }
        let lower = LatLonPoint::encode_ceil(min_latitude, min_longitude).map_err(encode)?;
        let upper = LatLonPoint::encode(max_latitude, max_longitude).map_err(encode)?;
        if max_longitude < min_longitude {
            let mut left_open = lower;
            left_open[4..].copy_from_slice(&int_to_sortable_bytes(i32::MIN));
            let left = PointRangeQuery::new(field, 2, left_open.to_vec(), upper.to_vec())?;
            let mut right_open = upper;
            right_open[4..].copy_from_slice(&int_to_sortable_bytes(i32::MAX));
            let right = PointRangeQuery::new(field, 2, lower.to_vec(), right_open.to_vec())?;
            return Ok(Box::new(ConstantScoreBoolean {
                clauses: vec![Box::new(left), Box::new(right)],
                must: false,
            }));
        }
        Ok(Box::new(PointRangeQuery::new(
            field,
            2,
            lower.to_vec(),
            upper.to_vec(),
        )?))
    }

    /// `newDistanceQuery(field, latitude, longitude, radiusMeters)`.
    ///
    /// # Errors
    /// An invalid centre or radius, with Java's message.
    pub fn new_distance_query(
        field: &str,
        latitude: f64,
        longitude: f64,
        radius_meters: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        Ok(Box::new(LatLonPointDistanceQuery::new(
            field,
            latitude,
            longitude,
            radius_meters,
        )?))
    }

    /// `newPolygonQuery(field, polygons...)`.
    ///
    /// # Errors
    /// As [`new_geometry_query`].
    pub fn new_polygon_query(field: &str, polygons: &[Polygon]) -> Result<Box<dyn DocumentQuery>> {
        let geometries: Vec<LatLonGeometry> = polygons
            .iter()
            .cloned()
            .map(LatLonGeometry::Polygon)
            .collect();
        new_geometry_query(field, QueryRelation::Intersects, &geometries)
    }

    /// `newGeometryQuery(field, queryRelation, latLonGeometries...)`: a
    /// single rectangle or circle intersected is a box or distance query; a
    /// `CONTAINS` of points is a conjunction of one query per point (and of
    /// anything else matches nothing).
    ///
    /// # Errors
    /// An invalid geometry, or one the relation does not support, with
    /// Java's message.
    pub fn new_geometry_query(
        field: &str,
        query_relation: QueryRelation,
        geometries: &[LatLonGeometry],
    ) -> Result<Box<dyn DocumentQuery>> {
        if query_relation == QueryRelation::Intersects && geometries.len() == 1 {
            match &geometries[0] {
                LatLonGeometry::Rectangle(r) => {
                    return new_box_query(field, r.min_lat, r.max_lat, r.min_lon, r.max_lon)
                }
                LatLonGeometry::Circle(c) => {
                    return new_distance_query(field, c.lat(), c.lon(), c.radius())
                }
                _ => {}
            }
        }
        if query_relation == QueryRelation::Contains {
            let mut clauses: Vec<Box<dyn DocumentQuery>> = Vec::with_capacity(geometries.len());
            for g in geometries {
                if !matches!(g, LatLonGeometry::Point(_)) {
                    return Ok(Box::new(MatchNoDocs));
                }
                clauses.push(Box::new(LatLonPointQuery::new(
                    field,
                    QueryRelation::Contains,
                    std::slice::from_ref(g),
                )?));
            }
            return Ok(Box::new(ConstantScoreBoolean {
                clauses,
                must: true,
            }));
        }
        Ok(Box::new(LatLonPointQuery::new(
            field,
            query_relation,
            geometries,
        )?))
    }

    /// `newDistanceFeatureQuery(field, weight, originLat, originLon,
    /// pivotDistanceMeters)`: boosted by `weight` unless it is 1.
    ///
    /// # Errors
    /// An invalid origin or a pivot that is not positive, with Java's
    /// message.
    pub fn new_distance_feature_query(
        field: &str,
        weight: f32,
        origin_lat: f64,
        origin_lon: f64,
        pivot_distance_meters: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        let q = Box::new(LatLonPointDistanceFeatureQuery::new(
            field,
            origin_lat,
            origin_lon,
            pivot_distance_meters,
        )?);
        #[allow(clippy::float_cmp)]
        if weight != 1.0 {
            return Ok(Box::new(Boosted::new(q, weight)));
        }
        Ok(q)
    }

    pub use super::nearest::nearest;
}

/// `LatLonDocValuesField`'s query factories and distance sort.
pub mod lat_lon_doc_values_field {
    use lucene_util::geo::{Circle, LatLonGeometry, Polygon};

    use super::super::MatchNoDocs;
    use super::*;

    /// `newDistanceSort(field, latitude, longitude)`.
    ///
    /// # Errors
    /// An invalid origin, with Java's message.
    pub fn new_distance_sort(
        field: &str,
        latitude: f64,
        longitude: f64,
    ) -> Result<LatLonPointSortField> {
        LatLonPointSortField::new(field, latitude, longitude)
    }

    /// `newSlowBoxQuery(field, minLatitude, maxLatitude, minLongitude,
    /// maxLongitude)`.
    ///
    /// # Errors
    /// An invalid coordinate, with Java's message.
    pub fn new_slow_box_query(
        field: &str,
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        let mut min_longitude = min_longitude;
        #[allow(clippy::float_cmp)]
        {
            if min_latitude == 90.0 {
                return Ok(Box::new(MatchNoDocs));
            }
            if min_longitude == 180.0 {
                if max_longitude == 180.0 {
                    return Ok(Box::new(MatchNoDocs));
                } else if max_longitude < min_longitude {
                    min_longitude = -180.0;
                }
            }
        }
        Ok(Box::new(LatLonDocValuesBoxQuery::new(
            field,
            min_latitude,
            max_latitude,
            min_longitude,
            max_longitude,
        )?))
    }

    /// `newSlowDistanceQuery(field, latitude, longitude, radiusMeters)`.
    ///
    /// # Errors
    /// An invalid circle, with Java's message.
    pub fn new_slow_distance_query(
        field: &str,
        latitude: f64,
        longitude: f64,
        radius_meters: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        let circle = Circle::new(latitude, longitude, radius_meters).map_err(geo)?;
        new_slow_geometry_query(
            field,
            QueryRelation::Intersects,
            &[LatLonGeometry::Circle(circle)],
        )
    }

    /// `newSlowPolygonQuery(field, polygons...)`.
    ///
    /// # Errors
    /// As [`new_slow_geometry_query`].
    pub fn new_slow_polygon_query(
        field: &str,
        polygons: &[Polygon],
    ) -> Result<Box<dyn DocumentQuery>> {
        let geometries: Vec<LatLonGeometry> = polygons
            .iter()
            .cloned()
            .map(LatLonGeometry::Polygon)
            .collect();
        new_slow_geometry_query(field, QueryRelation::Intersects, &geometries)
    }

    /// `newSlowGeometryQuery(field, queryRelation, latLonGeometries...)`.
    ///
    /// # Errors
    /// An invalid geometry, or one the relation does not support, with
    /// Java's message.
    pub fn new_slow_geometry_query(
        field: &str,
        query_relation: QueryRelation,
        geometries: &[LatLonGeometry],
    ) -> Result<Box<dyn DocumentQuery>> {
        if query_relation == QueryRelation::Intersects && geometries.len() == 1 {
            if let LatLonGeometry::Rectangle(r) = &geometries[0] {
                return new_slow_box_query(field, r.min_lat, r.max_lat, r.min_lon, r.max_lon);
            }
        }
        if query_relation == QueryRelation::Contains
            && geometries
                .iter()
                .any(|g| !matches!(g, LatLonGeometry::Point(_)))
        {
            return Ok(Box::new(MatchNoDocs));
        }
        Ok(Box::new(LatLonDocValuesQuery::new(
            field,
            query_relation,
            geometries,
        )?))
    }
}

/// `XYPointField`'s query factories.
pub mod xy_point_field {
    use lucene_util::geo::{XYCircle, XYGeometry, XYPolygon, XYRectangle};

    use super::*;

    /// `newBoxQuery(field, minX, maxX, minY, maxY)`.
    ///
    /// # Errors
    /// An invalid rectangle, with Java's message.
    pub fn new_box_query(
        field: &str,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
    ) -> Result<XYPointInGeometryQuery> {
        let r = XYRectangle::new(min_x, max_x, min_y, max_y).map_err(geo)?;
        XYPointInGeometryQuery::new(field, &[XYGeometry::Rectangle(r)])
    }

    /// `newDistanceQuery(field, x, y, radius)`.
    ///
    /// # Errors
    /// An invalid circle, with Java's message.
    pub fn new_distance_query(
        field: &str,
        x: f32,
        y: f32,
        radius: f32,
    ) -> Result<XYPointInGeometryQuery> {
        let c = XYCircle::new(x, y, radius).map_err(geo)?;
        XYPointInGeometryQuery::new(field, &[XYGeometry::Circle(c)])
    }

    /// `newPolygonQuery(field, polygons...)`.
    ///
    /// # Errors
    /// No polygons, or an invalid one.
    pub fn new_polygon_query(
        field: &str,
        polygons: &[XYPolygon],
    ) -> Result<XYPointInGeometryQuery> {
        let g: Vec<XYGeometry> = polygons.iter().cloned().map(XYGeometry::Polygon).collect();
        XYPointInGeometryQuery::new(field, &g)
    }

    /// `newGeometryQuery(field, xyGeometries...)`.
    ///
    /// # Errors
    /// No geometries, or an invalid one.
    pub fn new_geometry_query(
        field: &str,
        geometries: &[XYGeometry],
    ) -> Result<XYPointInGeometryQuery> {
        XYPointInGeometryQuery::new(field, geometries)
    }
}

/// `XYDocValuesField`'s query factories and distance sort.
pub mod xy_doc_values_field {
    use lucene_util::geo::{XYCircle, XYGeometry, XYPolygon, XYRectangle};

    use super::*;

    /// `newDistanceSort(field, x, y)`.
    pub fn new_distance_sort(field: &str, x: f32, y: f32) -> XYPointSortField {
        XYPointSortField::new(field, x, y)
    }

    /// `newSlowBoxQuery(field, minX, maxX, minY, maxY)`.
    ///
    /// # Errors
    /// An invalid rectangle, with Java's message.
    pub fn new_slow_box_query(
        field: &str,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
    ) -> Result<XYDocValuesPointInGeometryQuery> {
        let r = XYRectangle::new(min_x, max_x, min_y, max_y).map_err(geo)?;
        XYDocValuesPointInGeometryQuery::new(field, &[XYGeometry::Rectangle(r)])
    }

    /// `newSlowDistanceQuery(field, x, y, radius)`.
    ///
    /// # Errors
    /// An invalid circle, with Java's message.
    pub fn new_slow_distance_query(
        field: &str,
        x: f32,
        y: f32,
        radius: f32,
    ) -> Result<XYDocValuesPointInGeometryQuery> {
        let c = XYCircle::new(x, y, radius).map_err(geo)?;
        XYDocValuesPointInGeometryQuery::new(field, &[XYGeometry::Circle(c)])
    }

    /// `newSlowPolygonQuery(field, polygons...)`.
    ///
    /// # Errors
    /// No polygons, or an invalid one.
    pub fn new_slow_polygon_query(
        field: &str,
        polygons: &[XYPolygon],
    ) -> Result<XYDocValuesPointInGeometryQuery> {
        let g: Vec<XYGeometry> = polygons.iter().cloned().map(XYGeometry::Polygon).collect();
        XYDocValuesPointInGeometryQuery::new(field, &g)
    }

    /// `newSlowGeometryQuery(field, geometries...)`.
    ///
    /// # Errors
    /// No geometries, or an invalid one.
    pub fn new_slow_geometry_query(
        field: &str,
        geometries: &[XYGeometry],
    ) -> Result<XYDocValuesPointInGeometryQuery> {
        XYDocValuesPointInGeometryQuery::new(field, geometries)
    }
}

/// `LatLonShape`'s query factories.
pub mod lat_lon_shape {
    use lucene_util::geo::{Circle, LatLonGeometry, Line, Point, Polygon, Rectangle};

    use super::*;

    /// `newBoxQuery(field, queryRelation, minLatitude, maxLatitude,
    /// minLongitude, maxLongitude)`: a `CONTAINS` box across the dateline
    /// is the conjunction of its two halves (a plain `BooleanQuery`, so a
    /// match would score twice the boost; see [`MustConjunction`] for why
    /// none happens in practice).
    ///
    /// # Errors
    /// An invalid rectangle, with Java's message.
    pub fn new_box_query(
        field: &str,
        query_relation: QueryRelation,
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        if query_relation == QueryRelation::Contains && min_longitude > max_longitude {
            return split_contains(
                field,
                min_latitude,
                max_latitude,
                min_longitude,
                max_longitude,
            );
        }
        let rectangle = Rectangle::new(min_latitude, max_latitude, min_longitude, max_longitude)
            .map_err(geo)?;
        Ok(Box::new(LatLonShapeBoundingBoxQuery::new(
            field,
            query_relation,
            rectangle,
        )?))
    }

    /// The two `MUST` halves of a `CONTAINS` box across the dateline.
    fn split_contains(
        field: &str,
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        let east = new_box_query(
            field,
            QueryRelation::Contains,
            min_latitude,
            max_latitude,
            min_longitude,
            180.0,
        )?;
        let west = new_box_query(
            field,
            QueryRelation::Contains,
            min_latitude,
            max_latitude,
            -180.0,
            max_longitude,
        )?;
        Ok(Box::new(MustConjunction {
            clauses: vec![east, west],
        }))
    }

    /// `newSlowDocValuesBoxQuery(field, queryRelation, minLatitude,
    /// maxLatitude, minLongitude, maxLongitude)`: as Java, a `CONTAINS` box
    /// across the dateline becomes the *indexed* box query's conjunction.
    ///
    /// # Errors
    /// An invalid rectangle, or `CONTAINS`, with Java's message.
    pub fn new_slow_doc_values_box_query(
        field: &str,
        query_relation: QueryRelation,
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        if query_relation == QueryRelation::Contains && min_longitude > max_longitude {
            return split_contains(
                field,
                min_latitude,
                max_latitude,
                min_longitude,
                max_longitude,
            );
        }
        let rectangle = Rectangle::new(min_latitude, max_latitude, min_longitude, max_longitude)
            .map_err(geo)?;
        Ok(Box::new(LatLonShapeDocValuesQuery::new(
            field,
            query_relation,
            &[LatLonGeometry::Rectangle(rectangle)],
        )?))
    }

    /// `newLineQuery(field, queryRelation, lines...)`.
    ///
    /// # Errors
    /// As [`new_geometry_query`].
    pub fn new_line_query(
        field: &str,
        query_relation: QueryRelation,
        lines: &[Line],
    ) -> Result<Box<dyn DocumentQuery>> {
        let g: Vec<LatLonGeometry> = lines.iter().cloned().map(LatLonGeometry::Line).collect();
        new_geometry_query(field, query_relation, &g)
    }

    /// `newPolygonQuery(field, queryRelation, polygons...)`.
    ///
    /// # Errors
    /// As [`new_geometry_query`].
    pub fn new_polygon_query(
        field: &str,
        query_relation: QueryRelation,
        polygons: &[Polygon],
    ) -> Result<Box<dyn DocumentQuery>> {
        let g: Vec<LatLonGeometry> = polygons
            .iter()
            .cloned()
            .map(LatLonGeometry::Polygon)
            .collect();
        new_geometry_query(field, query_relation, &g)
    }

    /// `newPointQuery(field, queryRelation, double[]... points)`: each
    /// point `[lat, lon]`.
    ///
    /// # Errors
    /// An invalid point, or as [`new_geometry_query`].
    pub fn new_point_query(
        field: &str,
        query_relation: QueryRelation,
        points: &[[f64; 2]],
    ) -> Result<Box<dyn DocumentQuery>> {
        let g = points
            .iter()
            .map(|p| Point::new(p[0], p[1]).map(LatLonGeometry::Point))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(geo)?;
        new_geometry_query(field, query_relation, &g)
    }

    /// `newDistanceQuery(field, queryRelation, circles...)`.
    ///
    /// # Errors
    /// As [`new_geometry_query`].
    pub fn new_distance_query(
        field: &str,
        query_relation: QueryRelation,
        circles: &[Circle],
    ) -> Result<Box<dyn DocumentQuery>> {
        let g: Vec<LatLonGeometry> = circles
            .iter()
            .cloned()
            .map(LatLonGeometry::Circle)
            .collect();
        new_geometry_query(field, query_relation, &g)
    }

    /// `newGeometryQuery(field, queryRelation, latLonGeometries...)`: a
    /// single rectangle is a box query; a `CONTAINS` of several geometries
    /// is a constant-score conjunction of one query per geometry.
    ///
    /// # Errors
    /// A line under `WITHIN`, or geometries `LatLonGeometry.create`
    /// rejects, with Java's message.
    pub fn new_geometry_query(
        field: &str,
        query_relation: QueryRelation,
        geometries: &[LatLonGeometry],
    ) -> Result<Box<dyn DocumentQuery>> {
        if let [geometry] = geometries {
            if let LatLonGeometry::Rectangle(r) = geometry {
                return new_box_query(
                    field,
                    query_relation,
                    r.min_lat,
                    r.max_lat,
                    r.min_lon,
                    r.max_lon,
                );
            }
            return Ok(Box::new(LatLonShapeQuery::new(
                field,
                query_relation,
                geometries,
            )?));
        }
        if query_relation == QueryRelation::Contains {
            // makeContainsGeometryQuery
            let mut clauses: Vec<Box<dyn DocumentQuery>> = Vec::with_capacity(geometries.len());
            for g in geometries {
                if let LatLonGeometry::Rectangle(r) = g {
                    // this handles rectangles across the dateline
                    clauses.push(new_box_query(
                        field,
                        QueryRelation::Contains,
                        r.min_lat,
                        r.max_lat,
                        r.min_lon,
                        r.max_lon,
                    )?);
                } else {
                    clauses.push(Box::new(LatLonShapeQuery::new(
                        field,
                        QueryRelation::Contains,
                        std::slice::from_ref(g),
                    )?));
                }
            }
            return Ok(Box::new(ConstantScoreBoolean {
                clauses,
                must: true,
            }));
        }
        Ok(Box::new(LatLonShapeQuery::new(
            field,
            query_relation,
            geometries,
        )?))
    }
}

/// `XYShape`'s query factories.
pub mod xy_shape {
    use lucene_util::geo::{XYCircle, XYGeometry, XYLine, XYPoint, XYPolygon, XYRectangle};

    use super::*;

    /// `newBoxQuery(field, queryRelation, minX, maxX, minY, maxY)`.
    ///
    /// # Errors
    /// An invalid rectangle, with Java's message.
    pub fn new_box_query(
        field: &str,
        query_relation: QueryRelation,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
    ) -> Result<Box<dyn DocumentQuery>> {
        let r = XYRectangle::new(min_x, max_x, min_y, max_y).map_err(geo)?;
        new_geometry_query(field, query_relation, &[XYGeometry::Rectangle(r)])
    }

    /// `newSlowDocValuesBoxQuery(field, queryRelation, minX, maxX, minY,
    /// maxY)`.
    ///
    /// # Errors
    /// An invalid rectangle, or `CONTAINS`, with Java's message.
    pub fn new_slow_doc_values_box_query(
        field: &str,
        query_relation: QueryRelation,
        min_x: f32,
        max_x: f32,
        min_y: f32,
        max_y: f32,
    ) -> Result<Box<dyn DocumentQuery>> {
        let r = XYRectangle::new(min_x, max_x, min_y, max_y).map_err(geo)?;
        Ok(Box::new(XYShapeDocValuesQuery::new(
            field,
            query_relation,
            &[XYGeometry::Rectangle(r)],
        )?))
    }

    /// `newLineQuery(field, queryRelation, lines...)`.
    ///
    /// # Errors
    /// As [`new_geometry_query`].
    pub fn new_line_query(
        field: &str,
        query_relation: QueryRelation,
        lines: &[XYLine],
    ) -> Result<Box<dyn DocumentQuery>> {
        let g: Vec<XYGeometry> = lines.iter().cloned().map(XYGeometry::Line).collect();
        new_geometry_query(field, query_relation, &g)
    }

    /// `newPolygonQuery(field, queryRelation, polygons...)`.
    ///
    /// # Errors
    /// As [`new_geometry_query`].
    pub fn new_polygon_query(
        field: &str,
        query_relation: QueryRelation,
        polygons: &[XYPolygon],
    ) -> Result<Box<dyn DocumentQuery>> {
        let g: Vec<XYGeometry> = polygons.iter().cloned().map(XYGeometry::Polygon).collect();
        new_geometry_query(field, query_relation, &g)
    }

    /// `newPointQuery(field, queryRelation, float[]... points)`: each point
    /// `[x, y]`.
    ///
    /// # Errors
    /// An invalid point, or as [`new_geometry_query`].
    pub fn new_point_query(
        field: &str,
        query_relation: QueryRelation,
        points: &[[f32; 2]],
    ) -> Result<Box<dyn DocumentQuery>> {
        let g = points
            .iter()
            .map(|p| XYPoint::new(p[0], p[1]).map(XYGeometry::Point))
            .collect::<std::result::Result<Vec<_>, _>>()
            .map_err(geo)?;
        new_geometry_query(field, query_relation, &g)
    }

    /// `newDistanceQuery(field, queryRelation, circles...)`.
    ///
    /// # Errors
    /// As [`new_geometry_query`].
    pub fn new_distance_query(
        field: &str,
        query_relation: QueryRelation,
        circles: &[XYCircle],
    ) -> Result<Box<dyn DocumentQuery>> {
        let g: Vec<XYGeometry> = circles.iter().cloned().map(XYGeometry::Circle).collect();
        new_geometry_query(field, query_relation, &g)
    }

    /// `newGeometryQuery(field, queryRelation, xyGeometries...)`: a
    /// `CONTAINS` of several geometries is a constant-score conjunction of
    /// one query per geometry.
    ///
    /// # Errors
    /// Geometries `XYGeometry.create` rejects, with Java's message.
    pub fn new_geometry_query(
        field: &str,
        query_relation: QueryRelation,
        geometries: &[XYGeometry],
    ) -> Result<Box<dyn DocumentQuery>> {
        if query_relation == QueryRelation::Contains && geometries.len() > 1 {
            let clauses = geometries
                .iter()
                .map(|g| new_geometry_query(field, query_relation, std::slice::from_ref(g)))
                .collect::<Result<Vec<_>>>()?;
            return Ok(Box::new(ConstantScoreBoolean {
                clauses,
                must: true,
            }));
        }
        Ok(Box::new(XYShapeQuery::new(
            field,
            query_relation,
            geometries,
        )?))
    }
}

#[cfg(test)]
mod tests;
