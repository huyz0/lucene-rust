//! The BKD-backed geo point queries: `LatLonPointDistanceQuery`,
//! `LatLonPointQuery` (with `SpatialQuery`'s relation scorers) and
//! `XYPointInGeometryQuery`.

use std::sync::Arc;

use lucene_codecs::field_infos::FieldInfo;
use lucene_codecs::points::{IntersectVisitor, PointsField, PointsReader, Relation};
use lucene_index::document::sortable_bytes_to_int;
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::geo::{
    Component2D, Component2DPredicate, DistancePredicate, GeoEncodingUtils, GeoError, GeoUtils,
    LatLonGeometry, Rectangle, WithinRelation, XYEncodingUtils, XYGeometry,
};

use super::{
    check_walk, clear_doc, collect_bits, geo, get_doc, idx, illegal, set_doc, QueryRelation,
};
use crate::collector::ScoringCollector;
use crate::document::{field_info, reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::points_query::estimate_doc_count;
use crate::Result;

/// The latitude (or `x`) of a packed point.
#[inline]
fn first(packed: &[u8]) -> i32 {
    sortable_bytes_to_int(packed)
}

/// The longitude (or `y`) of a packed point.
#[inline]
fn second(packed: &[u8]) -> i32 {
    sortable_bytes_to_int(&packed[4..])
}

/// `LatLonPoint.checkCompatible` / `XYPointField.checkCompatible`: a field
/// with points of another shape is an error, with `kind` naming the type.
fn check_compatible(info: &FieldInfo, kind: &str) -> Result<()> {
    check_shape(
        &info.name,
        info.point_dimension_count,
        info.point_num_bytes,
        kind,
    )
}

/// The tree itself must hold two indexed four-byte dimensions, whatever the
/// `FieldInfo` claims: every visitor here slices a packed value at byte 4.
/// Java does not check (`SpatialQuery`, the feature query and `nearest` would
/// throw decoding a short value); a corrupt or foreign field is an error
/// here rather than a panic.
pub(crate) fn check_points_shape(name: &str, pf: &PointsField) -> Result<()> {
    if pf.num_dims != 2 || pf.num_index_dims != 2 || pf.bytes_per_dim != 4 {
        return Err(illegal(format!(
            "field=\"{name}\" holds points of {} dimensions ({} indexed) of {} bytes, not a \
             geo point's two of four",
            pf.num_dims, pf.num_index_dims, pf.bytes_per_dim
        )));
    }
    Ok(())
}

fn check_shape(name: &str, dims: i32, bytes: i32, kind: &str) -> Result<()> {
    if dims != 0 && dims != 2 {
        return Err(illegal(format!(
            "field=\"{name}\" was indexed with numDims={dims} but this point type has numDims=2, \
             is the field really a {kind}?"
        )));
    }
    if bytes != 0 && bytes != 4 {
        return Err(illegal(format!(
            "field=\"{name}\" was indexed with bytesPerDim={bytes} but this point type has \
             bytesPerDim=4, is the field really a {kind}?"
        )));
    }
    Ok(())
}

/// The segment's points for `field`: its `FieldInfo`, the reader and the
/// field's tree, or `None` when it has no points here.
pub(crate) fn points_of<'a>(
    leaf: &OpenSegment<'a>,
    field: &str,
) -> Result<Option<(&'a FieldInfo, PointsReader<'a>)>> {
    let Some(info) = field_info(leaf, field)? else {
        return Ok(None);
    };
    if info.point_dimension_count == 0 {
        return Ok(None);
    }
    let points = reader(leaf)?.points_reader()?;
    if points.field(info.number).is_none() {
        return Ok(None);
    }
    Ok(Some((info, points)))
}

/// `SpatialQuery.transposeRelation`.
fn transpose(r: Relation) -> Relation {
    match r {
        Relation::CellInsideQuery => Relation::CellOutsideQuery,
        Relation::CellOutsideQuery => Relation::CellInsideQuery,
        Relation::CellCrossesQuery => Relation::CellCrossesQuery,
    }
}

// ---------------------------------------------------------------- distance

/// `LatLonPointDistanceQuery` (`LatLonPoint.newDistanceQuery`): the
/// documents with a point within `radius_meters` of the centre, by the
/// haversine sort key, at a constant score.
#[derive(Debug, Clone)]
pub struct LatLonPointDistanceQuery {
    pub field: String,
    pub latitude: f64,
    pub longitude: f64,
    pub radius_meters: f64,
    /// `createWeight`'s precomputed state.
    weight: DistanceWeight,
}

/// What `LatLonPointDistanceQuery.createWeight` computes once: the encoded
/// bounding box (two longitude ranges across the dateline), the sort key of
/// the radius, the axis latitude and the grid predicate.
#[derive(Debug, Clone)]
struct DistanceWeight {
    min_lat: i32,
    max_lat: i32,
    min_lon: i32,
    max_lon: i32,
    min_lon2: i32,
    sort_key: f64,
    axis_lat: f64,
    predicate: DistancePredicate,
}

impl LatLonPointDistanceQuery {
    /// `LatLonPointDistanceQuery(field, latitude, longitude, radiusMeters)`.
    ///
    /// # Errors
    /// A radius that is negative or not finite, or an invalid centre, with
    /// Java's message.
    pub fn new(
        field: impl Into<String>,
        latitude: f64,
        longitude: f64,
        radius_meters: f64,
    ) -> Result<Self> {
        if !radius_meters.is_finite() || radius_meters < 0.0 {
            return Err(illegal(format!(
                "radiusMeters: '{}' is invalid",
                lucene_util::geo::java_double_string(radius_meters)
            )));
        }
        GeoUtils::check_latitude(latitude).map_err(geo)?;
        GeoUtils::check_longitude(longitude).map_err(geo)?;
        let weight = Self::weight(latitude, longitude, radius_meters).map_err(geo)?;
        Ok(LatLonPointDistanceQuery {
            field: field.into(),
            latitude,
            longitude,
            radius_meters,
            weight,
        })
    }

    /// `createWeight`'s precomputation.
    fn weight(lat: f64, lon: f64, radius: f64) -> std::result::Result<DistanceWeight, GeoError> {
        let b = Rectangle::from_point_distance(lat, lon, radius)?;
        let min_lat = GeoEncodingUtils::encode_latitude(b.min_lat)?;
        let max_lat = GeoEncodingUtils::encode_latitude(b.max_lat)?;
        let (min_lon, max_lon, min_lon2) = if b.crosses_dateline() {
            (
                i32::MIN,
                GeoEncodingUtils::encode_longitude(b.max_lon)?,
                GeoEncodingUtils::encode_longitude(b.min_lon)?,
            )
        } else {
            (
                GeoEncodingUtils::encode_longitude(b.min_lon)?,
                GeoEncodingUtils::encode_longitude(b.max_lon)?,
                i32::MAX,
            )
        };
        Ok(DistanceWeight {
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            min_lon2,
            sort_key: GeoUtils::distance_query_sort_key(radius),
            axis_lat: Rectangle::axis_lat(lat, radius),
            predicate: GeoEncodingUtils::create_distance_predicate(lat, lon, radius)?,
        })
    }

    /// `matches(packedValue)`.
    #[inline]
    fn matches(&self, packed: &[u8]) -> bool {
        let w = &self.weight;
        let lat = first(packed);
        if lat > w.max_lat || lat < w.min_lat {
            return false;
        }
        let lon = second(packed);
        if (lon > w.max_lon || lon < w.min_lon) && lon < w.min_lon2 {
            return false;
        }
        w.predicate.test(lat, lon)
    }

    /// `relate(minPackedValue, maxPackedValue)`.
    fn relate(&self, min: &[u8], max: &[u8]) -> Relation {
        let w = &self.weight;
        let lat_lo = first(min);
        let lat_hi = first(max);
        if lat_lo > w.max_lat || lat_hi < w.min_lat {
            return Relation::CellOutsideQuery;
        }
        let lon_lo = second(min);
        let lon_hi = second(max);
        if (lon_lo > w.max_lon || lon_hi < w.min_lon) && lon_hi < w.min_lon2 {
            return Relation::CellOutsideQuery;
        }
        GeoUtils::relate(
            GeoEncodingUtils::decode_latitude(lat_lo),
            GeoEncodingUtils::decode_latitude(lat_hi),
            GeoEncodingUtils::decode_longitude(lon_lo),
            GeoEncodingUtils::decode_longitude(lon_hi),
            self.latitude,
            self.longitude,
            w.sort_key,
            w.axis_lat,
        )
        // A cell never crosses the dateline (its min is below its max), so
        // `relate`'s only throw cannot happen.
        .unwrap_or(Relation::CellCrossesQuery)
    }
}

/// The distance query's forward visitor (`getIntersectVisitor`).
struct DistanceVisitor<'q> {
    q: &'q LatLonPointDistanceQuery,
    result: FixedBitSet,
    /// The first doc id outside the segment (see [`set_doc`]).
    bad: Option<i32>,
}

impl IntersectVisitor for DistanceVisitor<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        self.q.relate(min, max)
    }
    fn visit(&mut self, doc_id: i32) {
        set_doc(&mut self.result, doc_id, &mut self.bad);
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        if self.q.matches(packed) {
            set_doc(&mut self.result, doc_id, &mut self.bad);
        }
    }
}

/// The distance query's inverse visitor (`getInverseIntersectVisitor`):
/// clears the documents that cannot match.
struct InverseDistanceVisitor<'q> {
    q: &'q LatLonPointDistanceQuery,
    result: FixedBitSet,
    /// The first doc id outside the segment (see [`clear_doc`]).
    bad: Option<i32>,
}

impl IntersectVisitor for InverseDistanceVisitor<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        transpose(self.q.relate(min, max))
    }
    fn visit(&mut self, doc_id: i32) {
        clear_doc(&mut self.result, doc_id, &mut self.bad);
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        if !self.q.matches(packed) {
            clear_doc(&mut self.result, doc_id, &mut self.bad);
        }
    }
}

impl DocumentQuery for LatLonPointDistanceQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some((info, points)) = points_of(leaf, &self.field)? else {
            return Ok(());
        };
        check_compatible(info, "LatLonPoint")?;
        let values = points.field(info.number).expect("checked by points_of");
        check_points_shape(&info.name, values)?;
        let max_doc = reader(leaf)?.max_doc;
        let size = idx(max_doc);
        if values.doc_count == max_doc && i64::from(values.doc_count) == values.point_count {
            // `cost()`: the forward visitor's estimate.
            let mut est = DistanceVisitor {
                q: self,
                result: FixedBitSet::new(0),
                bad: None,
            };
            let cost = estimate_doc_count(
                points.estimate_point_count(info.number, &mut est)?,
                values.point_count,
                values.doc_count,
            );
            if cost > i64::from(max_doc / 2) {
                let mut result = FixedBitSet::new(size);
                result.set_range(0, size);
                let mut v = InverseDistanceVisitor {
                    q: self,
                    result,
                    bad: None,
                };
                points.intersect(info.number, &mut v)?;
                check_walk(v.bad, &info.name, max_doc)?;
                collect_bits(leaf, &v.result, boost, collector);
                return Ok(());
            }
        }
        let mut v = DistanceVisitor {
            q: self,
            result: FixedBitSet::new(size),
            bad: None,
        };
        points.intersect(info.number, &mut v)?;
        check_walk(v.bad, &info.name, max_doc)?;
        collect_bits(leaf, &v.result, boost, collector);
        Ok(())
    }
}

// ---------------------------------------------------------------- spatial

/// `SpatialQuery.SpatialVisitor`: a query geometry's cell relation and its
/// per-value predicates -- over a packed point for `LatLonPointQuery`, over
/// an encoded triangle for the shape queries.
pub(crate) trait SpatialVisitor {
    /// `relate(minPackedValue, maxPackedValue)`.
    fn relate(&self, min: &[u8], max: &[u8]) -> Relation;
    /// `intersects()`.
    fn intersects(&self, packed: &[u8]) -> bool;
    /// `within()`: for a point the same test as `intersects()`.
    fn within(&self, packed: &[u8]) -> bool {
        self.intersects(packed)
    }
    /// `contains()`.
    fn contains(&self, packed: &[u8]) -> std::result::Result<WithinRelation, GeoError>;
    /// What calling `contains()` itself throws, before any value is tested
    /// (`LatLonShapeBoundingBoxQuery` refuses a box across the dateline).
    fn check_contains(&self) -> std::result::Result<(), GeoError> {
        Ok(())
    }

    /// `getInnerFunction(queryRelation)`.
    fn inner(&self, rel: QueryRelation, min: &[u8], max: &[u8]) -> Relation {
        let r = self.relate(min, max);
        if rel == QueryRelation::Disjoint {
            transpose(r)
        } else {
            r
        }
    }

    /// `getLeafPredicate(queryRelation)`.
    fn leaf(&self, rel: QueryRelation, packed: &[u8]) -> std::result::Result<bool, GeoError> {
        Ok(match rel {
            QueryRelation::Intersects => self.intersects(packed),
            QueryRelation::Within => self.within(packed),
            QueryRelation::Disjoint => !self.intersects(packed),
            QueryRelation::Contains => self.contains(packed)? == WithinRelation::Candidate,
        })
    }
}

/// The visitors of `SpatialQuery`'s scorers, as one type: which one is
/// `kind`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Walk {
    /// `getEstimateVisitor`: only `compare` is used.
    Estimate,
    /// `getSparseVisitor` / `getIntersectsDenseVisitor`.
    Forward,
    /// `getInverseDenseVisitor`.
    Inverse,
    /// `getDenseVisitor`.
    Dense,
    /// `getContainsDenseVisitor`.
    ContainsDense,
    /// `getShallowInverseDenseVisitor`.
    ShallowInverse,
    /// `hasAnyHits`' visitor; `found` ends the walk.
    AnyHits,
}

struct SpatialWalk<'v, V: ?Sized> {
    v: &'v V,
    rel: QueryRelation,
    walk: Walk,
    result: FixedBitSet,
    excluded: FixedBitSet,
    found: bool,
    /// The first error a `contains` test raised (Java throws it).
    error: Option<GeoError>,
    /// The first doc id outside the segment (see [`set_doc`]).
    bad: Option<i32>,
}

impl<'v, V: SpatialVisitor + ?Sized> SpatialWalk<'v, V> {
    fn new(v: &'v V, rel: QueryRelation, walk: Walk, max_doc: usize) -> Self {
        let excluded = if matches!(walk, Walk::Dense | Walk::ContainsDense) {
            FixedBitSet::new(max_doc)
        } else {
            FixedBitSet::new(0)
        };
        let result = if matches!(walk, Walk::Estimate | Walk::AnyHits) {
            FixedBitSet::new(0)
        } else {
            FixedBitSet::new(max_doc)
        };
        SpatialWalk {
            v,
            rel,
            walk,
            result,
            excluded,
            found: false,
            error: None,
            bad: None,
        }
    }

    fn leaf(&mut self, packed: &[u8]) -> bool {
        match self.v.leaf(self.rel, packed) {
            Ok(b) => b,
            Err(e) => {
                self.error.get_or_insert(e);
                false
            }
        }
    }
}

impl<V: SpatialVisitor + ?Sized> IntersectVisitor for SpatialWalk<'_, V> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        match self.walk {
            Walk::Inverse | Walk::ShallowInverse => transpose(self.v.inner(self.rel, min, max)),
            Walk::AnyHits => {
                if self.found {
                    return Relation::CellOutsideQuery;
                }
                let r = self.v.inner(self.rel, min, max);
                if r == Relation::CellInsideQuery {
                    self.found = true;
                    return Relation::CellOutsideQuery;
                }
                r
            }
            _ => self.v.inner(self.rel, min, max),
        }
    }

    fn visit(&mut self, doc_id: i32) {
        let d = doc_id;
        match self.walk {
            Walk::Estimate => {}
            Walk::Forward | Walk::Dense => set_doc(&mut self.result, d, &mut self.bad),
            Walk::Inverse | Walk::ShallowInverse => clear_doc(&mut self.result, d, &mut self.bad),
            Walk::ContainsDense => set_doc(&mut self.excluded, d, &mut self.bad),
            Walk::AnyHits => self.found = true,
        }
    }

    /// `visit(DocIdSetIterator)` / `visit(IntsRef)`: a cell inside the
    /// query, its documents taken as a run (`result.or(iterator)` and
    /// friends in Java) with one dispatch on the walk rather than one per
    /// document.
    fn visit_many(&mut self, doc_ids: &[i32]) {
        match self.walk {
            Walk::Estimate => {}
            Walk::Forward | Walk::Dense => {
                for &d in doc_ids {
                    set_doc(&mut self.result, d, &mut self.bad);
                }
            }
            Walk::Inverse | Walk::ShallowInverse => {
                for &d in doc_ids {
                    clear_doc(&mut self.result, d, &mut self.bad);
                }
            }
            Walk::ContainsDense => {
                for &d in doc_ids {
                    set_doc(&mut self.excluded, d, &mut self.bad);
                }
            }
            Walk::AnyHits => self.found |= !doc_ids.is_empty(),
        }
    }

    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        let d = doc_id;
        match self.walk {
            Walk::Estimate | Walk::ShallowInverse => {}
            Walk::Forward => {
                if !get_doc(&self.result, d) && self.leaf(packed) {
                    set_doc(&mut self.result, d, &mut self.bad);
                }
            }
            Walk::Inverse => {
                if get_doc(&self.result, d) && !self.leaf(packed) {
                    clear_doc(&mut self.result, d, &mut self.bad);
                }
            }
            Walk::Dense => {
                if !get_doc(&self.excluded, d) {
                    if self.leaf(packed) {
                        set_doc(&mut self.result, d, &mut self.bad);
                    } else {
                        set_doc(&mut self.excluded, d, &mut self.bad);
                    }
                }
            }
            Walk::ContainsDense => {
                if !get_doc(&self.excluded, d) {
                    match self.v.contains(packed) {
                        Ok(WithinRelation::Candidate) => {
                            set_doc(&mut self.result, d, &mut self.bad)
                        }
                        Ok(WithinRelation::NotWithin) => {
                            set_doc(&mut self.excluded, d, &mut self.bad)
                        }
                        Ok(WithinRelation::Disjoint) => {}
                        Err(e) => {
                            self.error.get_or_insert(e);
                        }
                    }
                }
            }
            Walk::AnyHits => {
                if !self.found && self.leaf(packed) {
                    self.found = true;
                }
            }
        }
    }
}

/// `SpatialQuery.getScorerSupplier` and `RelationScorerSupplier.getScorer`
/// over one segment: every live match of `visitor` under `rel`, at `score`.
pub(crate) fn spatial_score_leaf<V: SpatialVisitor + ?Sized>(
    leaf: &OpenSegment<'_>,
    info: &FieldInfo,
    points: &PointsReader<'_>,
    visitor: &V,
    rel: QueryRelation,
    score: f32,
    collector: &mut dyn ScoringCollector,
) -> Result<()> {
    let values: &PointsField = points.field(info.number).expect("checked by points_of");
    let max_doc = reader(leaf)?.max_doc;
    let size = idx(max_doc);
    let num = info.number;
    let doc_count = values.doc_count;
    let point_count = values.point_count;
    let single = i64::from(doc_count) == point_count;
    let r = visitor.inner(rel, &values.min_packed_value, &values.max_packed_value);
    if r == Relation::CellOutsideQuery
        || (r == Relation::CellInsideQuery && rel == QueryRelation::Contains)
    {
        return Ok(());
    }
    if doc_count == max_doc && r == Relation::CellInsideQuery {
        let mut all = FixedBitSet::new(size);
        all.set_range(0, size);
        collect_bits(leaf, &all, score, collector);
        return Ok(());
    }
    let run = |walk: Walk, init: Option<FixedBitSet>| -> Result<SpatialWalk<'_, V>> {
        let mut w = SpatialWalk::new(visitor, rel, walk, size);
        if let Some(bits) = init {
            w.result = bits;
        }
        points.intersect(num, &mut w)?;
        if let Some(e) = w.error.take() {
            return Err(geo(e));
        }
        check_walk(w.bad, &info.name, max_doc)?;
        Ok(w)
    };
    if rel != QueryRelation::Intersects && rel != QueryRelation::Contains && !single {
        // `hasAnyHits`: fast in the adversarial dense case with no match.
        if !run(Walk::AnyHits, None)?.found {
            return Ok(());
        }
    }
    let full = || {
        let mut all = FixedBitSet::new(size);
        all.set_range(0, size);
        all
    };
    let result = match rel {
        QueryRelation::Contains => {
            // `getContainsDenseVisitor` asks for `contains()` here.
            visitor.check_contains().map_err(geo)?;
            let mut w = run(Walk::ContainsDense, None)?;
            w.result.and_not(&w.excluded);
            w.result
        }
        QueryRelation::Within | QueryRelation::Disjoint if !single => {
            if doc_count == max_doc {
                run(Walk::Inverse, Some(full()))?.result
            } else {
                let mut w = run(Walk::Dense, None)?;
                w.result.and_not(&w.excluded);
                run(Walk::ShallowInverse, Some(w.result))?.result
            }
        }
        _ => {
            // `getSparseScorer`.
            if rel == QueryRelation::Disjoint && doc_count == max_doc && single {
                let mut est = SpatialWalk::new(visitor, rel, Walk::Estimate, 0);
                let cost = estimate_doc_count(
                    points.estimate_point_count(num, &mut est)?,
                    point_count,
                    doc_count,
                );
                if cost > i64::from(max_doc / 2) {
                    let r = run(Walk::Inverse, Some(full()))?.result;
                    collect_bits(leaf, &r, score, collector);
                    return Ok(());
                }
            }
            run(Walk::Forward, None)?.result
        }
    };
    collect_bits(leaf, &result, score, collector);
    Ok(())
}

/// `LatLonPointQuery` (`LatLonPoint.newPolygonQuery` / `newGeometryQuery`):
/// the documents whose points relate to the geometries as `query_relation`
/// says -- through `SpatialQuery`'s scorers -- at a constant score.
#[derive(Debug, Clone)]
pub struct LatLonPointQuery {
    pub field: String,
    pub query_relation: QueryRelation,
    pub geometries: Vec<LatLonGeometry>,
    visitor: Arc<LatLonPointVisitor>,
}

/// `LatLonPointQuery.getSpatialVisitor()`'s state: the geometries as one
/// `Component2D`, its grid predicate and its encoded bounding box.
struct LatLonPointVisitor {
    component: Arc<dyn Component2D>,
    predicate: Component2DPredicate<'static>,
    min_lat: i32,
    max_lat: i32,
    min_lon: i32,
    max_lon: i32,
}

impl std::fmt::Debug for LatLonPointVisitor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LatLonPointVisitor")
            .field("component", &self.component)
            .finish_non_exhaustive()
    }
}

impl LatLonPointQuery {
    /// `LatLonPointQuery(field, queryRelation, geometries...)`.
    ///
    /// # Errors
    /// A line under `WITHIN`, anything but points under `CONTAINS`, or a
    /// geometry `LatLonGeometry.create` rejects, with Java's message.
    pub fn new(
        field: impl Into<String>,
        query_relation: QueryRelation,
        geometries: &[LatLonGeometry],
    ) -> Result<Self> {
        if query_relation == QueryRelation::Within
            && geometries
                .iter()
                .any(|g| matches!(g, LatLonGeometry::Line(_)))
        {
            return Err(illegal(
                "LatLonPointQuery does not support WITHIN queries with line geometries",
            ));
        }
        if query_relation == QueryRelation::Contains
            && geometries
                .iter()
                .any(|g| !matches!(g, LatLonGeometry::Point(_)))
        {
            return Err(illegal(
                "LatLonPointQuery does not support CONTAINS queries with non-points geometries",
            ));
        }
        let component: Arc<dyn Component2D> =
            Arc::from(LatLonGeometry::create(geometries).map_err(geo)?);
        let predicate =
            GeoEncodingUtils::create_component_predicate_shared(component.clone()).map_err(geo)?;
        let visitor = LatLonPointVisitor {
            min_lat: GeoEncodingUtils::encode_latitude(component.min_y()).map_err(geo)?,
            max_lat: GeoEncodingUtils::encode_latitude(component.max_y()).map_err(geo)?,
            min_lon: GeoEncodingUtils::encode_longitude(component.min_x()).map_err(geo)?,
            max_lon: GeoEncodingUtils::encode_longitude(component.max_x()).map_err(geo)?,
            component,
            predicate,
        };
        Ok(LatLonPointQuery {
            field: field.into(),
            query_relation,
            geometries: geometries.to_vec(),
            visitor: Arc::new(visitor),
        })
    }
}

impl SpatialVisitor for LatLonPointVisitor {
    fn relate(&self, min: &[u8], max: &[u8]) -> Relation {
        let lat_lo = first(min);
        let lat_hi = first(max);
        if lat_lo > self.max_lat || lat_hi < self.min_lat {
            return Relation::CellOutsideQuery;
        }
        let lon_lo = second(min);
        let lon_hi = second(max);
        if lon_lo > self.max_lon || lon_hi < self.min_lon {
            return Relation::CellOutsideQuery;
        }
        self.component.relate(
            GeoEncodingUtils::decode_longitude(lon_lo),
            GeoEncodingUtils::decode_longitude(lon_hi),
            GeoEncodingUtils::decode_latitude(lat_lo),
            GeoEncodingUtils::decode_latitude(lat_hi),
        )
    }

    #[inline]
    fn intersects(&self, packed: &[u8]) -> bool {
        self.predicate.test(first(packed), second(packed))
    }

    fn contains(&self, packed: &[u8]) -> std::result::Result<WithinRelation, GeoError> {
        self.component.within_point(
            GeoEncodingUtils::decode_longitude(second(packed)),
            GeoEncodingUtils::decode_latitude(first(packed)),
        )
    }
}

impl DocumentQuery for LatLonPointQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some((info, points)) = points_of(leaf, &self.field)? else {
            return Ok(());
        };
        if let Some(pf) = points.field(info.number) {
            check_points_shape(&info.name, pf)?;
        }
        spatial_score_leaf(
            leaf,
            info,
            &points,
            self.visitor.as_ref(),
            self.query_relation,
            boost,
            collector,
        )
    }
}

// ---------------------------------------------------------------- cartesian

/// `XYPointInGeometryQuery` (every `XYPointField` query): the documents
/// with a point inside the geometries, at a constant score.
#[derive(Debug, Clone)]
pub struct XYPointInGeometryQuery {
    pub field: String,
    pub geometries: Vec<XYGeometry>,
    tree: Arc<dyn Component2D>,
}

impl XYPointInGeometryQuery {
    /// `XYPointInGeometryQuery(field, xyGeometries...)`.
    ///
    /// # Errors
    /// No geometries, or one `XYGeometry.create` rejects, with Java's
    /// message.
    pub fn new(field: impl Into<String>, geometries: &[XYGeometry]) -> Result<Self> {
        if geometries.is_empty() {
            return Err(illegal("geometries must not be empty"));
        }
        let tree = Arc::from(XYGeometry::create(geometries).map_err(geo)?);
        Ok(XYPointInGeometryQuery {
            field: field.into(),
            geometries: geometries.to_vec(),
            tree,
        })
    }
}

/// `XYPointInGeometryQuery.getIntersectVisitor`.
struct XYVisitor<'t> {
    tree: &'t dyn Component2D,
    result: FixedBitSet,
    /// The first doc id outside the segment (see [`set_doc`]).
    bad: Option<i32>,
}

impl XYVisitor<'_> {
    #[inline]
    fn decode(packed: &[u8]) -> (f64, f64) {
        (
            f64::from(XYEncodingUtils::decode(first(packed))),
            f64::from(XYEncodingUtils::decode(second(packed))),
        )
    }
}

impl IntersectVisitor for XYVisitor<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        let (min_x, min_y) = Self::decode(min);
        let (max_x, max_y) = Self::decode(max);
        self.tree.relate(min_x, max_x, min_y, max_y)
    }
    fn visit(&mut self, doc_id: i32) {
        set_doc(&mut self.result, doc_id, &mut self.bad);
    }
    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        let (x, y) = Self::decode(packed);
        if self.tree.contains(x, y) {
            set_doc(&mut self.result, doc_id, &mut self.bad);
        }
    }
}

impl DocumentQuery for XYPointInGeometryQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some((info, points)) = points_of(leaf, &self.field)? else {
            return Ok(());
        };
        check_compatible(info, "XYPoint")?;
        if let Some(pf) = points.field(info.number) {
            check_points_shape(&info.name, pf)?;
        }
        let max_doc = reader(leaf)?.max_doc;
        let mut v = XYVisitor {
            tree: self.tree.as_ref(),
            result: FixedBitSet::new(idx(max_doc)),
            bad: None,
        };
        points.intersect(info.number, &mut v)?;
        check_walk(v.bad, &info.name, max_doc)?;
        collect_bits(leaf, &v.result, boost, collector);
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A visitor whose answers are fixed: `relate` says `rel`, `intersects`
    /// says `hit`, `contains` says `within` (or fails).
    struct Fixed {
        rel: Relation,
        hit: bool,
        within: Option<WithinRelation>,
    }

    impl SpatialVisitor for Fixed {
        fn relate(&self, _min: &[u8], _max: &[u8]) -> Relation {
            self.rel
        }
        fn intersects(&self, _packed: &[u8]) -> bool {
            self.hit
        }
        fn contains(&self, _packed: &[u8]) -> std::result::Result<WithinRelation, GeoError> {
            self.within
                .ok_or_else(|| GeoError::IllegalArgument("no withinPoint".into()))
        }
    }

    #[test]
    fn spatial_walks_set_clear_and_exclude_as_java() {
        let v = Fixed {
            rel: Relation::CellCrossesQuery,
            hit: true,
            within: Some(WithinRelation::NotWithin),
        };
        let p = [0u8; 8];
        // CONTAINS: a cell inside excludes; NOTWITHIN excludes; CANDIDATE sets.
        let mut w = SpatialWalk::new(&v, QueryRelation::Contains, Walk::ContainsDense, 8);
        w.visit(1);
        w.visit_with_value(2, &p);
        assert!(get_doc(&w.excluded, 1) && get_doc(&w.excluded, 2));
        assert!(!get_doc(&w.result, 2));
        let cand = Fixed {
            within: Some(WithinRelation::Candidate),
            ..v
        };
        let mut w = SpatialWalk::new(&cand, QueryRelation::Contains, Walk::ContainsDense, 8);
        w.visit_with_value(3, &p);
        assert!(get_doc(&w.result, 3));
        assert_eq!(cand.leaf(QueryRelation::Contains, &p), Ok(true));
        // A failing `contains` is kept, and the walk goes on.
        let bad = Fixed { within: None, ..v };
        let mut w = SpatialWalk::new(&bad, QueryRelation::Contains, Walk::ContainsDense, 8);
        w.visit_with_value(4, &p);
        assert!(w.error.is_some());
        let mut w = SpatialWalk::new(&bad, QueryRelation::Contains, Walk::Forward, 8);
        w.visit_with_value(4, &p);
        assert!(w.error.is_some() && !get_doc(&w.result, 4));
        // The estimate walk only relates; any-hits stops at the first hit.
        let mut w = SpatialWalk::new(&v, QueryRelation::Within, Walk::Estimate, 8);
        w.visit(0);
        w.visit_with_value(0, &p);
        assert_eq!(w.compare(&p, &p), Relation::CellCrossesQuery);
        let inside = Fixed {
            rel: Relation::CellInsideQuery,
            ..v
        };
        let mut w = SpatialWalk::new(&inside, QueryRelation::Within, Walk::AnyHits, 0);
        assert_eq!(w.compare(&p, &p), Relation::CellOutsideQuery);
        assert!(w.found);
        assert_eq!(w.compare(&p, &p), Relation::CellOutsideQuery);
        let mut w = SpatialWalk::new(&v, QueryRelation::Within, Walk::AnyHits, 0);
        w.visit(5);
        assert!(w.found);
        // Dense: a failing point excludes its document for good.
        let miss = Fixed { hit: false, ..v };
        let mut w = SpatialWalk::new(&miss, QueryRelation::Within, Walk::Dense, 8);
        w.visit_with_value(6, &p);
        w.visit(7);
        assert!(get_doc(&w.excluded, 6) && get_doc(&w.result, 7));
        // The shallow inverse walk clears inside cells only.
        let mut w = SpatialWalk::new(&v, QueryRelation::Disjoint, Walk::ShallowInverse, 8);
        w.result.set_range(0, 8);
        w.visit_with_value(1, &p);
        w.visit(2);
        assert!(get_doc(&w.result, 1) && !get_doc(&w.result, 2));
        // Out-of-segment doc ids from a corrupt tree set nothing; the first
        // is remembered, and fails the walk.
        let mut w = SpatialWalk::new(&v, QueryRelation::Intersects, Walk::Forward, 8);
        w.visit(9);
        w.visit(-1);
        assert_eq!(w.result.cardinality(), 0);
        assert_eq!(w.bad, Some(9));
        let e = check_walk(w.bad, "f", 8).unwrap_err();
        assert!(
            e.to_string()
                .contains("points of field f name document 9, outside the segment's 0..8"),
            "{e}"
        );
        let mut w = SpatialWalk::new(&v, QueryRelation::Disjoint, Walk::Inverse, 8);
        w.visit_many(&[3, 99]);
        assert_eq!(w.bad, Some(99));
        let mut w = SpatialWalk::new(&v, QueryRelation::Within, Walk::ContainsDense, 8);
        w.visit_many(&[-5]);
        assert_eq!(w.bad, Some(-5));
        assert!(check_walk(None, "f", 8).is_ok());
    }

    #[test]
    fn transpose_swaps_inside_and_outside() {
        assert_eq!(
            transpose(Relation::CellInsideQuery),
            Relation::CellOutsideQuery
        );
        assert_eq!(
            transpose(Relation::CellOutsideQuery),
            Relation::CellInsideQuery
        );
        assert_eq!(
            transpose(Relation::CellCrossesQuery),
            Relation::CellCrossesQuery
        );
    }

    #[test]
    fn distance_query_validates_like_java() {
        let e = LatLonPointDistanceQuery::new("f", 0.0, 0.0, -1.0).unwrap_err();
        assert!(
            e.to_string().contains("radiusMeters: '-1.0' is invalid"),
            "{e}"
        );
        assert!(LatLonPointDistanceQuery::new("f", 0.0, 0.0, f64::NAN).is_err());
        assert!(LatLonPointDistanceQuery::new("f", 91.0, 0.0, 1.0).is_err());
        assert!(LatLonPointDistanceQuery::new("f", 0.0, 181.0, 1.0).is_err());
        let q = LatLonPointDistanceQuery::new("f", 10.0, 179.9, 50_000.0).unwrap();
        assert_eq!(q.weight.min_lon, i32::MIN, "the box crosses the dateline");
        let near = lucene_index::document::LatLonPoint::encode(10.0, -179.95).unwrap();
        let far = lucene_index::document::LatLonPoint::encode(10.0, 170.0).unwrap();
        assert!(q.matches(&near));
        assert!(!q.matches(&far));
        let lo = lucene_index::document::LatLonPoint::encode(-50.0, -50.0).unwrap();
        let hi = lucene_index::document::LatLonPoint::encode(-40.0, -40.0).unwrap();
        assert_eq!(q.relate(&lo, &hi), Relation::CellOutsideQuery);
    }

    #[test]
    fn point_query_rejects_unsupported_relations() {
        use lucene_util::geo::{Line, Point};
        let line = LatLonGeometry::Line(Line::new(&[0.0, 1.0], &[0.0, 1.0]).unwrap());
        let e = LatLonPointQuery::new("f", QueryRelation::Within, std::slice::from_ref(&line))
            .unwrap_err();
        assert!(e.to_string().contains("WITHIN queries with line"), "{e}");
        let e = LatLonPointQuery::new("f", QueryRelation::Contains, &[line]).unwrap_err();
        assert!(
            e.to_string().contains("CONTAINS queries with non-points"),
            "{e}"
        );
        assert!(LatLonPointQuery::new("f", QueryRelation::Intersects, &[]).is_err());
        let at = lucene_index::document::LatLonPoint::encode(1.0, 2.0).unwrap();
        let p = LatLonGeometry::Point(
            Point::new(
                GeoEncodingUtils::decode_latitude_bytes(&at, 0),
                GeoEncodingUtils::decode_longitude_bytes(&at, 4),
            )
            .unwrap(),
        );
        let q = LatLonPointQuery::new("f", QueryRelation::Contains, &[p]).unwrap();
        assert_eq!(q.visitor.contains(&at).unwrap(), WithinRelation::Candidate);
        assert!(format!("{q:?}").contains("LatLonPointVisitor"));
        assert!(XYPointInGeometryQuery::new("f", &[]).is_err());
    }

    #[test]
    fn check_compatible_messages() {
        let e = check_shape("f", 1, 4, "LatLonPoint").unwrap_err();
        assert!(e.to_string().contains("numDims=1"), "{e}");
        let e = check_shape("f", 2, 8, "XYPoint").unwrap_err();
        assert!(
            e.to_string().contains("bytesPerDim=8") && e.to_string().contains("XYPoint"),
            "{e}"
        );
        assert!(check_shape("f", 2, 4, "XYPoint").is_ok());
        assert!(check_shape("f", 0, 0, "XYPoint").is_ok());
    }
}
