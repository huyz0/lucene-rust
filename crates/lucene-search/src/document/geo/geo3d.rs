//! The spatial3d query and helpers: `PointInGeo3DShapeQuery` with
//! `PointInShapeIntersectVisitor`, and `Geo3DUtil`'s shape conversions --
//! over the `Geo3DPoint` x/y/z points `lucene_index::document` writes.
//!
//! The query walks the BKD tree as Java's does: a cell entirely around the
//! shape's (encoding-rounded) bounds is crossed without a relation test;
//! otherwise the cell, widened to the encoding's floor/ceil, becomes an
//! x/y/z solid whose `getRelationship` with the shape decides; a leaf value
//! inside the bounds is kept when the shape contains it. An exception a
//! shape method throws inside the walk (raised, see
//! `lucene_util::spatial3d::errors`) is the search's error, as Java's
//! would propagate.

use std::sync::Arc;

use lucene_codecs::points::{IntersectVisitor, Relation};
use lucene_index::document::geo3d::from_degrees;
use lucene_index::document::{sortable_bytes_to_int, Geo3DPoint};
use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::geo::{GeoUtils, Polygon};
use lucene_util::spatial3d::errors::catch;
use lucene_util::spatial3d::geo_area_factory::make_geo_area;
use lucene_util::spatial3d::geo_bbox_factory::make_geo_bbox;
use lucene_util::spatial3d::geo_circle_factory::make_geo_circle;
use lucene_util::spatial3d::geo_composite::GeoCompositePolygon;
use lucene_util::spatial3d::geo_path_factory::make_geo_path;
use lucene_util::spatial3d::geo_polygon_factory::{
    make_geo_polygon_with_holes, make_large_geo_polygon, PolygonDescription,
};
use lucene_util::spatial3d::{
    GeoAreaRelationship, GeoBBox, GeoCircle, GeoPath, GeoPoint, GeoPolygon, GeoShape, PlanetModel,
    XYZBounds,
};

use super::point_queries::points_of;
use super::{check_walk, collect_bits, idx, illegal, set_doc};
use crate::collector::ScoringCollector;
use crate::document::{reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::{Error, Result};

fn s3d(e: lucene_util::spatial3d::Error) -> Error {
    illegal(e.to_string())
}

fn geo(e: lucene_util::geo::GeoError) -> Error {
    illegal(e.to_string())
}

/// `Geo3DUtil.decodeValueFloor(x, planetModel)`: the low edge of an encoded
/// value's cell.
pub fn decode_value_floor(x: i32, planet_model: &PlanetModel) -> f64 {
    if x == planet_model.min_encoded_value {
        return -planet_model.max_value;
    }
    f64::from(x) * planet_model.decode
}

/// `Geo3DUtil.decodeValueCeil(x, planetModel)`: the high edge.
pub fn decode_value_ceil(x: i32, planet_model: &PlanetModel) -> f64 {
    if x == planet_model.max_encoded_value {
        return planet_model.max_value;
    }
    // `Math.nextDown((x + 1) * DECODE)`: `x + 1` is an int below
    // MAX_ENCODED_VALUE here, exact as a double.
    ((f64::from(x) + 1.0) * planet_model.decode).next_down()
}

/// A surface point at a latitude/longitude in degrees, checked as
/// `GeoUtils` checks.
fn point_at(pm: &Arc<PlanetModel>, latitude: f64, longitude: f64) -> Result<GeoPoint> {
    GeoUtils::check_latitude(latitude).map_err(geo)?;
    GeoUtils::check_longitude(longitude).map_err(geo)?;
    GeoPoint::from_lat_lon(pm, from_degrees(latitude), from_degrees(longitude)).map_err(s3d)
}

/// A `Polygon`'s ring as geo3d points: reversed, without the closing point
/// (`Geo3DUtil`'s loop, which does not check the coordinates).
fn ring(pm: &Arc<PlanetModel>, polygon: &Polygon) -> Result<Vec<GeoPoint>> {
    let lats = polygon.poly_lats();
    let lons = polygon.poly_lons();
    (0..lats.len().saturating_sub(1))
        .rev()
        .map(|i| {
            GeoPoint::from_lat_lon(pm, from_degrees(lats[i]), from_degrees(lons[i])).map_err(s3d)
        })
        .collect()
}

/// `Geo3DUtil.fromPolygon(planetModel, polygons...)`: one polygon, or a
/// composite of several (an empty composite for a degenerate one).
///
/// # Errors
/// No polygons, or the factory's exception.
pub fn from_polygon(pm: &Arc<PlanetModel>, polygons: &[Polygon]) -> Result<Arc<dyn GeoPolygon>> {
    if polygons.is_empty() {
        return Err(illegal("need at least one polygon"));
    }
    if let [only] = polygons {
        return Ok(match one_polygon(pm, only)? {
            Some(p) => p,
            None => Arc::new(GeoCompositePolygon::new(pm)),
        });
    }
    let mut composite = GeoCompositePolygon::new(pm);
    for p in polygons {
        if let Some(component) = one_polygon(pm, p)? {
            composite.add_shape(component).map_err(s3d)?;
        }
    }
    Ok(Arc::new(composite))
}

/// The private `fromPolygon(planetModel, polygon)`: holes first, each its
/// own polygon.
fn one_polygon(pm: &Arc<PlanetModel>, polygon: &Polygon) -> Result<Option<Arc<dyn GeoPolygon>>> {
    let mut holes = Vec::with_capacity(polygon.holes().len());
    for hole in polygon.holes() {
        if let Some(component) = one_polygon(pm, hole)? {
            holes.push(component);
        }
    }
    let points = ring(pm, polygon)?;
    catch(|| make_geo_polygon_with_holes(pm, &points, Some(holes), 0.0))
        .and_then(|r| r)
        .map_err(s3d)
}

/// `Geo3DUtil.fromLargePolygon(planetModel, polygons...)`.
///
/// # Errors
/// No polygons, or the factory's exception.
pub fn from_large_polygon(
    pm: &Arc<PlanetModel>,
    polygons: &[Polygon],
) -> Result<Arc<dyn GeoPolygon>> {
    if polygons.is_empty() {
        return Err(illegal("need at least one polygon"));
    }
    let descriptions = descriptions(pm, polygons)?;
    catch(|| make_large_geo_polygon(pm, &descriptions))
        .and_then(|r| r)
        .map_err(s3d)
}

/// `Geo3DUtil.convertToDescription(planetModel, polygons...)`.
fn descriptions(pm: &Arc<PlanetModel>, polygons: &[Polygon]) -> Result<Vec<PolygonDescription>> {
    polygons
        .iter()
        .map(|p| {
            Ok(PolygonDescription::with_holes(
                ring(pm, p)?,
                descriptions(pm, p.holes())?,
            ))
        })
        .collect()
}

/// `Geo3DUtil.fromPath(planetModel, pathLatitudes, pathLongitudes,
/// pathWidthMeters)`.
///
/// # Errors
/// Mismatched arrays, an invalid coordinate, or the factory's exception.
pub fn from_path(
    pm: &Arc<PlanetModel>,
    latitudes: &[f64],
    longitudes: &[f64],
    width_meters: f64,
) -> Result<Arc<dyn GeoPath>> {
    if latitudes.len() != longitudes.len() {
        return Err(illegal("same number of latitudes and longitudes required"));
    }
    let points = latitudes
        .iter()
        .zip(longitudes)
        .map(|(&lat, &lon)| point_at(pm, lat, lon))
        .collect::<Result<Vec<_>>>()?;
    let radius = width_meters / (pm.mean_radius() * pm.xy_scaling);
    catch(|| make_geo_path(pm, radius, &points))
        .and_then(|r| r)
        .map_err(s3d)
}

/// `Geo3DUtil.fromDistance(planetModel, latitude, longitude, radiusMeters)`.
///
/// # Errors
/// An invalid coordinate, or the factory's exception.
pub fn from_distance(
    pm: &Arc<PlanetModel>,
    latitude: f64,
    longitude: f64,
    radius_meters: f64,
) -> Result<Arc<dyn GeoCircle>> {
    GeoUtils::check_latitude(latitude).map_err(geo)?;
    GeoUtils::check_longitude(longitude).map_err(geo)?;
    let radius = radius_meters / pm.mean_radius();
    catch(|| make_geo_circle(pm, from_degrees(latitude), from_degrees(longitude), radius))
        .and_then(|r| r)
        .map_err(s3d)
}

/// `Geo3DUtil.fromBox(planetModel, minLatitude, maxLatitude, minLongitude,
/// maxLongitude)`.
///
/// # Errors
/// An invalid coordinate, or the factory's exception.
pub fn from_box(
    pm: &Arc<PlanetModel>,
    min_latitude: f64,
    max_latitude: f64,
    min_longitude: f64,
    max_longitude: f64,
) -> Result<Arc<dyn GeoBBox>> {
    GeoUtils::check_latitude(min_latitude).map_err(geo)?;
    GeoUtils::check_longitude(min_longitude).map_err(geo)?;
    GeoUtils::check_latitude(max_latitude).map_err(geo)?;
    GeoUtils::check_longitude(max_longitude).map_err(geo)?;
    make_geo_bbox(
        pm,
        from_degrees(max_latitude),
        from_degrees(min_latitude),
        from_degrees(min_longitude),
        from_degrees(max_longitude),
    )
    .map_err(s3d)
}

/// `PointInGeo3DShapeQuery` (`Geo3DPoint.newShapeQuery` and every other
/// `Geo3DPoint` query): the documents with a point inside the shape, at a
/// constant score.
#[derive(Clone)]
pub struct PointInGeo3DShapeQuery {
    pub field: String,
    pub shape: Arc<dyn GeoShape>,
    /// `shapeBounds`, Java's `getBounds` of the shape.
    bounds: XYZBounds,
}

impl std::fmt::Debug for PointInGeo3DShapeQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "PointInGeo3DShapeQuery({:?}, class {:?})",
            self.field,
            self.shape.class_code()
        )
    }
}

impl PointInGeo3DShapeQuery {
    /// `PointInGeo3DShapeQuery(field, shape)`.
    ///
    /// # Errors
    /// The shape's `getBounds` throws (raised; Java's constructor would
    /// throw).
    pub fn new(field: impl Into<String>, shape: Arc<dyn GeoShape>) -> Result<Self> {
        let bounds = catch(|| {
            let mut b = XYZBounds::new();
            shape.get_bounds(&mut b);
            b
        })
        .map_err(s3d)?;
        Ok(PointInGeo3DShapeQuery {
            field: field.into(),
            shape,
            bounds,
        })
    }
}

/// `PointInShapeIntersectVisitor`.
struct ShapeVisitor<'s> {
    shape: &'s dyn GeoShape,
    planet_model: &'s Arc<PlanetModel>,
    minimum_x: f64,
    maximum_x: f64,
    minimum_y: f64,
    maximum_y: f64,
    minimum_z: f64,
    maximum_z: f64,
    result: FixedBitSet,
    bad: Option<i32>,
    /// An exception relating a cell (`makeGeoArea`'s or the relation's).
    error: Option<lucene_util::spatial3d::Error>,
}

impl ShapeVisitor<'_> {
    fn cell(&self, min: &[u8], max: &[u8], d: usize) -> (f64, f64) {
        let at = d * 4;
        (
            decode_value_floor(sortable_bytes_to_int(&min[at..]), self.planet_model),
            decode_value_ceil(sortable_bytes_to_int(&max[at..]), self.planet_model),
        )
    }
}

impl IntersectVisitor for ShapeVisitor<'_> {
    fn compare(&mut self, min: &[u8], max: &[u8]) -> Relation {
        let (x_min, x_max) = self.cell(min, max, 0);
        let (y_min, y_max) = self.cell(min, max, 1);
        let (z_min, z_max) = self.cell(min, max, 2);
        if self.minimum_x >= x_min
            && self.maximum_x <= x_max
            && self.minimum_y >= y_min
            && self.maximum_y <= y_max
            && self.minimum_z >= z_min
            && self.maximum_z <= z_max
        {
            return Relation::CellCrossesQuery;
        }
        let relation = make_geo_area(self.planet_model, x_min, x_max, y_min, y_max, z_min, z_max)
            .and_then(|area| area.get_relationship(self.shape));
        match relation {
            Ok(GeoAreaRelationship::Contains) => Relation::CellInsideQuery,
            Ok(GeoAreaRelationship::Overlaps) | Ok(GeoAreaRelationship::Within) => {
                Relation::CellCrossesQuery
            }
            Ok(GeoAreaRelationship::Disjoint) => Relation::CellOutsideQuery,
            Err(e) => {
                self.error.get_or_insert(e);
                Relation::CellOutsideQuery
            }
        }
    }

    fn visit(&mut self, doc_id: i32) {
        set_doc(&mut self.result, doc_id, &mut self.bad);
    }

    fn visit_with_value(&mut self, doc_id: i32, packed: &[u8]) {
        let pm = self.planet_model;
        let x = Geo3DPoint::decode_dimension(packed, pm);
        let y = Geo3DPoint::decode_dimension(&packed[4..], pm);
        let z = Geo3DPoint::decode_dimension(&packed[8..], pm);
        if x >= self.minimum_x
            && x <= self.maximum_x
            && y >= self.minimum_y
            && y <= self.maximum_y
            && z >= self.minimum_z
            && z <= self.maximum_z
            && self.shape.is_within_xyz(x, y, z)
        {
            set_doc(&mut self.result, doc_id, &mut self.bad);
        }
    }
}

impl DocumentQuery for PointInGeo3DShapeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let Some((info, points)) = points_of(leaf, &self.field)? else {
            return Ok(());
        };
        // Java asserts twelve-byte values; a field of another shape is an
        // error here rather than a slice out of range.
        if let Some(pf) = points.field(info.number) {
            if pf.num_dims != 3 || pf.num_index_dims != 3 || pf.bytes_per_dim != 4 {
                return Err(illegal(format!(
                    "field=\"{}\" holds points of {} dimensions ({} indexed) of {} bytes, not a \
                     Geo3DPoint's three of four",
                    info.name, pf.num_dims, pf.num_index_dims, pf.bytes_per_dim
                )));
            }
        }
        let pm = self.shape.planet_model();
        let e = pm.doc_value_encoder();
        // The visitor's constructor unboxes the bounds (a shape's are set).
        let b = &self.bounds;
        let unset = || illegal("the shape has unset x/y/z bounds (NullPointerException in Java)");
        let max_doc = reader(leaf)?.max_doc;
        let mut v = ShapeVisitor {
            shape: &*self.shape,
            planet_model: pm,
            minimum_x: e.round_down_x(b.minimum_x().ok_or_else(unset)?),
            maximum_x: e.round_up_x(b.maximum_x().ok_or_else(unset)?),
            minimum_y: e.round_down_y(b.minimum_y().ok_or_else(unset)?),
            maximum_y: e.round_up_y(b.maximum_y().ok_or_else(unset)?),
            minimum_z: e.round_down_z(b.minimum_z().ok_or_else(unset)?),
            maximum_z: e.round_up_z(b.maximum_z().ok_or_else(unset)?),
            result: FixedBitSet::new(idx(max_doc)),
            bad: None,
            error: None,
        };
        let walked = catch(|| points.intersect(info.number, &mut v));
        walked.map_err(s3d)??;
        v.error.take().map_or(Ok(()), |e| Err(s3d(e)))?;
        check_walk(v.bad, &info.name, max_doc)?;
        collect_bits(leaf, &v.result, boost, collector);
        Ok(())
    }
}

/// `Geo3DPoint`'s query factories.
pub mod geo3d_point {
    use super::*;

    /// `newShapeQuery(field, shape)`.
    ///
    /// # Errors
    /// As [`PointInGeo3DShapeQuery::new`].
    pub fn new_shape_query(
        field: &str,
        shape: Arc<dyn GeoShape>,
    ) -> Result<Box<dyn DocumentQuery>> {
        Ok(Box::new(PointInGeo3DShapeQuery::new(field, shape)?))
    }

    /// `newDistanceQuery(field, planetModel, latitude, longitude,
    /// radiusMeters)`.
    ///
    /// # Errors
    /// As [`from_distance`].
    pub fn new_distance_query(
        field: &str,
        pm: &Arc<PlanetModel>,
        latitude: f64,
        longitude: f64,
        radius_meters: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        new_shape_query(
            field,
            from_distance(pm, latitude, longitude, radius_meters)?,
        )
    }

    /// `newBoxQuery(field, planetModel, minLatitude, maxLatitude,
    /// minLongitude, maxLongitude)`.
    ///
    /// # Errors
    /// As [`from_box`].
    pub fn new_box_query(
        field: &str,
        pm: &Arc<PlanetModel>,
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
    ) -> Result<Box<dyn DocumentQuery>> {
        new_shape_query(
            field,
            from_box(pm, min_latitude, max_latitude, min_longitude, max_longitude)?,
        )
    }

    /// `newPolygonQuery(field, planetModel, polygons...)`.
    ///
    /// # Errors
    /// As [`from_polygon`].
    pub fn new_polygon_query(
        field: &str,
        pm: &Arc<PlanetModel>,
        polygons: &[Polygon],
    ) -> Result<Box<dyn DocumentQuery>> {
        new_shape_query(field, from_polygon(pm, polygons)?)
    }

    /// `newLargePolygonQuery(field, planetModel, polygons...)`.
    ///
    /// # Errors
    /// As [`from_large_polygon`].
    pub fn new_large_polygon_query(
        field: &str,
        pm: &Arc<PlanetModel>,
        polygons: &[Polygon],
    ) -> Result<Box<dyn DocumentQuery>> {
        new_shape_query(field, from_large_polygon(pm, polygons)?)
    }

    /// `newPathQuery(field, pathLatitudes, pathLongitudes, pathWidthMeters,
    /// planetModel)`.
    ///
    /// # Errors
    /// As [`from_path`].
    pub fn new_path_query(
        field: &str,
        latitudes: &[f64],
        longitudes: &[f64],
        width_meters: f64,
        pm: &Arc<PlanetModel>,
    ) -> Result<Box<dyn DocumentQuery>> {
        new_shape_query(field, from_path(pm, latitudes, longitudes, width_meters)?)
    }
}

/// `Geo3DDocValuesField`'s sort factories.
pub mod geo3d_doc_values_field {
    use super::super::{Geo3DPointOutsideSortField, Geo3DPointSortField};
    use super::*;

    /// `newDistanceSort(field, latitude, longitude, maxRadiusMeters,
    /// planetModel)`.
    ///
    /// # Errors
    /// As [`from_distance`].
    pub fn new_distance_sort(
        field: &str,
        latitude: f64,
        longitude: f64,
        max_radius_meters: f64,
        pm: &Arc<PlanetModel>,
    ) -> Result<Geo3DPointSortField> {
        let shape = from_distance(pm, latitude, longitude, max_radius_meters)?;
        Ok(Geo3DPointSortField::new(field, pm, shape))
    }

    /// `newPathSort(field, pathLatitudes, pathLongitudes, pathWidthMeters,
    /// planetModel)`.
    ///
    /// # Errors
    /// As [`from_path`].
    pub fn new_path_sort(
        field: &str,
        latitudes: &[f64],
        longitudes: &[f64],
        width_meters: f64,
        pm: &Arc<PlanetModel>,
    ) -> Result<Geo3DPointSortField> {
        let shape = from_path(pm, latitudes, longitudes, width_meters)?;
        Ok(Geo3DPointSortField::new(field, pm, shape))
    }

    /// `newOutsideDistanceSort(field, latitude, longitude, maxRadiusMeters,
    /// planetModel)`.
    ///
    /// # Errors
    /// As [`from_distance`].
    pub fn new_outside_distance_sort(
        field: &str,
        latitude: f64,
        longitude: f64,
        max_radius_meters: f64,
        pm: &Arc<PlanetModel>,
    ) -> Result<Geo3DPointOutsideSortField> {
        let shape = from_distance(pm, latitude, longitude, max_radius_meters)?;
        Ok(Geo3DPointOutsideSortField::new(field, pm, shape))
    }

    /// `newOutsideBoxSort(field, minLatitude, maxLatitude, minLongitude,
    /// maxLongitude, planetModel)`.
    ///
    /// # Errors
    /// As [`from_box`].
    pub fn new_outside_box_sort(
        field: &str,
        min_latitude: f64,
        max_latitude: f64,
        min_longitude: f64,
        max_longitude: f64,
        pm: &Arc<PlanetModel>,
    ) -> Result<Geo3DPointOutsideSortField> {
        let shape = from_box(pm, min_latitude, max_latitude, min_longitude, max_longitude)?;
        Ok(Geo3DPointOutsideSortField::new(field, pm, shape))
    }

    /// `newOutsidePolygonSort(field, planetModel, polygons...)`.
    ///
    /// # Errors
    /// As [`from_polygon`].
    pub fn new_outside_polygon_sort(
        field: &str,
        pm: &Arc<PlanetModel>,
        polygons: &[Polygon],
    ) -> Result<Geo3DPointOutsideSortField> {
        Ok(Geo3DPointOutsideSortField::new(
            field,
            pm,
            from_polygon(pm, polygons)?,
        ))
    }

    /// `newOutsideLargePolygonSort(field, planetModel, polygons...)`.
    ///
    /// # Errors
    /// As [`from_large_polygon`].
    pub fn new_outside_large_polygon_sort(
        field: &str,
        pm: &Arc<PlanetModel>,
        polygons: &[Polygon],
    ) -> Result<Geo3DPointOutsideSortField> {
        Ok(Geo3DPointOutsideSortField::new(
            field,
            pm,
            from_large_polygon(pm, polygons)?,
        ))
    }

    /// `newOutsidePathSort(field, pathLatitudes, pathLongitudes,
    /// pathWidthMeters, planetModel)`.
    ///
    /// # Errors
    /// As [`from_path`].
    pub fn new_outside_path_sort(
        field: &str,
        latitudes: &[f64],
        longitudes: &[f64],
        width_meters: f64,
        pm: &Arc<PlanetModel>,
    ) -> Result<Geo3DPointOutsideSortField> {
        let shape = from_path(pm, latitudes, longitudes, width_meters)?;
        Ok(Geo3DPointOutsideSortField::new(field, pm, shape))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Clockwise, as `Geo3DUtil` expects (it reverses the ring).
    fn square(lat: f64, lon: f64) -> Polygon {
        let lats = [lat, lat + 1.0, lat + 1.0, lat, lat];
        let lons = [lon, lon, lon + 1.0, lon + 1.0, lon];
        Polygon::new(&lats, &lons, vec![]).unwrap()
    }

    #[test]
    fn factories_validate_their_arguments_as_java_does() {
        let pm = PlanetModel::wgs84();
        let e = from_polygon(&pm, &[]).err().unwrap();
        assert!(e.to_string().contains("need at least one polygon"), "{e}");
        assert!(from_large_polygon(&pm, &[]).is_err());
        let e = from_path(&pm, &[0.0], &[], 10.0).err().unwrap();
        assert!(e.to_string().contains("same number"), "{e}");
        let e = from_distance(&pm, 91.0, 0.0, 10.0).err().unwrap();
        assert!(e.to_string().contains("latitude"), "{e}");
        // Two polygons are a composite of both.
        let two = from_polygon(&pm, &[square(0.0, 0.0), square(10.0, 10.0)]).unwrap();
        assert!(two.is_within(&point_at(&pm, 0.5, 0.5).unwrap()));
        assert!(two.is_within(&point_at(&pm, 10.5, 10.5).unwrap()));
        assert!(!two.is_within(&point_at(&pm, 5.5, 5.5).unwrap()));
        let q = PointInGeo3DShapeQuery::new("f", two).unwrap();
        assert!(format!("{q:?}").starts_with("PointInGeo3DShapeQuery(\"f\""));
    }

    #[test]
    fn a_cell_the_area_factory_rejects_fails_the_walk() {
        let pm = PlanetModel::wgs84();
        let shape = from_distance(&pm, 0.0, 0.0, 1000.0).unwrap();
        let mut v = ShapeVisitor {
            shape: &*shape,
            planet_model: &pm,
            minimum_x: 0.0,
            maximum_x: 0.0,
            minimum_y: 0.0,
            maximum_y: 0.0,
            minimum_z: 0.0,
            maximum_z: 0.0,
            result: FixedBitSet::new(1),
            bad: None,
            error: None,
        };
        let packed = |value: f64| {
            let mut b = [0u8; 12];
            for d in 0..3 {
                Geo3DPoint::encode_dimension(value, &mut b[4 * d..], &pm).unwrap();
            }
            b
        };
        // A cell whose minimum lies above its maximum: no solid has it.
        let relation = v.compare(&packed(0.5), &packed(-0.5));
        assert_eq!(relation, Relation::CellOutsideQuery);
        let e = v.error.take().unwrap();
        assert!(e.to_string().contains("wrong order"), "{e}");
    }
}
