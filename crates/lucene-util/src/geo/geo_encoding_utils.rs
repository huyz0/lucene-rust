//! Port of `org.apache.lucene.geo.GeoEncodingUtils`: the 32-bit lat/lon
//! quantization `LatLonPoint` indexes, and the grid predicates the
//! doc-values and point queries test candidates with.

use super::component2d::Component2D;
use super::geo_utils::GeoUtils;
use super::rectangle::Rectangle;
use super::{GeoError, Relation};
use crate::numeric_utils;
use crate::sloppy_math;

/// Port of `org.apache.lucene.geo.GeoEncodingUtils`.
#[derive(Debug, Clone, Copy)]
pub struct GeoEncodingUtils;

const LAT_SCALE: f64 = (1u64 << 32) as f64 / 180.0;
const LAT_DECODE: f64 = 1.0 / LAT_SCALE;
const LON_SCALE: f64 = (1u64 << 32) as f64 / 360.0;
const LON_DECODE: f64 = 1.0 / LON_SCALE;

/// `Math.nextDown(double)` for a finite positive value.
#[inline]
fn next_down(v: f64) -> f64 {
    f64::from_bits(v.to_bits() - 1)
}

impl GeoEncodingUtils {
    /// `BITS`: bits per encoded dimension.
    pub const BITS: i16 = 32;
    /// `MIN_LON_ENCODED`: `encodeLongitude(-180)`.
    pub const MIN_LON_ENCODED: i32 = i32::MIN;
    /// `MAX_LON_ENCODED`: `encodeLongitude(180)`.
    pub const MAX_LON_ENCODED: i32 = i32::MAX;

    /// `encodeLatitude`: floor quantization.
    pub fn encode_latitude(latitude: f64) -> Result<i32, GeoError> {
        GeoUtils::check_latitude(latitude)?;
        let latitude = if latitude == 90.0 {
            next_down(latitude)
        } else {
            latitude
        };
        Ok((latitude / LAT_DECODE).floor() as i32)
    }

    /// `encodeLatitudeCeil`: ceil quantization.
    pub fn encode_latitude_ceil(latitude: f64) -> Result<i32, GeoError> {
        GeoUtils::check_latitude(latitude)?;
        let latitude = if latitude == 90.0 {
            next_down(latitude)
        } else {
            latitude
        };
        Ok((latitude / LAT_DECODE).ceil() as i32)
    }

    /// `encodeLongitude`: floor quantization.
    pub fn encode_longitude(longitude: f64) -> Result<i32, GeoError> {
        GeoUtils::check_longitude(longitude)?;
        let longitude = if longitude == 180.0 {
            next_down(longitude)
        } else {
            longitude
        };
        Ok((longitude / LON_DECODE).floor() as i32)
    }

    /// `encodeLongitudeCeil`: ceil quantization.
    pub fn encode_longitude_ceil(longitude: f64) -> Result<i32, GeoError> {
        GeoUtils::check_longitude(longitude)?;
        let longitude = if longitude == 180.0 {
            next_down(longitude)
        } else {
            longitude
        };
        Ok((longitude / LON_DECODE).ceil() as i32)
    }

    /// `decodeLatitude(int)`.
    #[inline]
    pub fn decode_latitude(encoded: i32) -> f64 {
        f64::from(encoded) * LAT_DECODE
    }

    /// `decodeLatitude(byte[], int)`.
    pub fn decode_latitude_bytes(src: &[u8], offset: usize) -> f64 {
        Self::decode_latitude(numeric_utils::sortable_bytes_to_int(src, offset))
    }

    /// `decodeLongitude(int)`.
    #[inline]
    pub fn decode_longitude(encoded: i32) -> f64 {
        f64::from(encoded) * LON_DECODE
    }

    /// `decodeLongitude(byte[], int)`.
    pub fn decode_longitude_bytes(src: &[u8], offset: usize) -> f64 {
        Self::decode_longitude(numeric_utils::sortable_bytes_to_int(src, offset))
    }

    /// `createDistancePredicate`.
    pub fn create_distance_predicate(
        lat: f64,
        lon: f64,
        radius_meters: f64,
    ) -> Result<DistancePredicate, GeoError> {
        let bounding_box = Rectangle::from_point_distance(lat, lon, radius_meters)?;
        let axis_lat = Rectangle::axis_lat(lat, radius_meters);
        let distance_sort_key = GeoUtils::distance_query_sort_key(radius_meters);
        let grid = create_sub_boxes(
            bounding_box.min_lat,
            bounding_box.max_lat,
            bounding_box.min_lon,
            bounding_box.max_lon,
            |b| {
                GeoUtils::relate(
                    b.min_lat,
                    b.max_lat,
                    b.min_lon,
                    b.max_lon,
                    lat,
                    lon,
                    distance_sort_key,
                    axis_lat,
                )
                // A grid cell never crosses the dateline (min <= max), so
                // Java's only throw cannot happen; "crosses" would be the
                // safe answer anyway (it defers to the exact check).
                .unwrap_or(Relation::CellCrossesQuery)
            },
        )?;
        Ok(DistancePredicate {
            grid,
            lat,
            lon,
            distance_key: distance_sort_key,
        })
    }

    /// `createComponentPredicate`.
    pub fn create_component_predicate(
        tree: &dyn Component2D,
    ) -> Result<Component2DPredicate<'_>, GeoError> {
        let grid = create_sub_boxes(
            tree.min_y(),
            tree.max_y(),
            tree.min_x(),
            tree.max_x(),
            |b| tree.relate(b.min_lon, b.max_lon, b.min_lat, b.max_lat),
        )?;
        Ok(Component2DPredicate {
            grid,
            tree: TreeRef::Borrowed(tree),
        })
    }

    /// [`Self::create_component_predicate`] over a shared tree: the
    /// predicate keeps the tree alive, so a query can hold both (Java's
    /// weight holds the tree and the predicate side by side).
    pub fn create_component_predicate_shared(
        tree: std::sync::Arc<dyn Component2D>,
    ) -> Result<Component2DPredicate<'static>, GeoError> {
        let grid = create_sub_boxes(
            tree.min_y(),
            tree.max_y(),
            tree.min_x(),
            tree.max_x(),
            |b| tree.relate(b.min_lon, b.max_lon, b.min_lat, b.max_lat),
        )?;
        Ok(Component2DPredicate {
            grid,
            tree: TreeRef::Shared(tree),
        })
    }
}

/// `GeoEncodingUtils.Grid`.
#[derive(Debug, Clone)]
struct Grid {
    lat_shift: i32,
    lon_shift: i32,
    lat_base: i32,
    lon_base: i32,
    max_lat_delta: i32,
    max_lon_delta: i32,
    relations: Vec<u8>,
}

const ARITY: i64 = 64;

/// `createSubBoxes`.
fn create_sub_boxes(
    shape_min_lat: f64,
    shape_max_lat: f64,
    shape_min_lon: f64,
    shape_max_lon: f64,
    mut box_to_relation: impl FnMut(&Rectangle) -> Relation,
) -> Result<Grid, GeoError> {
    let min_lat = GeoEncodingUtils::encode_latitude_ceil(shape_min_lat)?;
    let max_lat = GeoEncodingUtils::encode_latitude(shape_max_lat)?;
    let min_lon = GeoEncodingUtils::encode_longitude_ceil(shape_min_lon)?;
    let max_lon = GeoEncodingUtils::encode_longitude(shape_max_lon)?;
    if max_lat < min_lat || (shape_max_lon >= shape_min_lon && max_lon < min_lon) {
        // the box cannot match any quantized point
        return Ok(Grid {
            lat_shift: 1,
            lon_shift: 1,
            lat_base: 0,
            lon_base: 0,
            max_lat_delta: 0,
            max_lon_delta: 0,
            relations: Vec::new(),
        });
    }
    let (lat_shift, lat_base, max_lat_delta) = {
        let min_lat2 = i64::from(min_lat) - i64::from(i32::MIN);
        let max_lat2 = i64::from(max_lat) - i64::from(i32::MIN);
        let shift = compute_shift(min_lat2, max_lat2);
        let base = ((min_lat2 as u64) >> shift) as i32;
        let delta = (((max_lat2 as u64) >> shift) as i32) - base + 1;
        (shift, base, delta)
    };
    let (lon_shift, lon_base, max_lon_delta) = {
        let min_lon2 = i64::from(min_lon) - i64::from(i32::MIN);
        let mut max_lon2 = i64::from(max_lon) - i64::from(i32::MIN);
        if shape_max_lon < shape_min_lon {
            // crosses dateline
            max_lon2 += 1i64 << 32;
        }
        let shift = compute_shift(min_lon2, max_lon2);
        let base = ((min_lon2 as u64) >> shift) as i32;
        let delta = (((max_lon2 as u64) >> shift) as i32) - base + 1;
        (shift, base, delta)
    };
    let mut relations = vec![0u8; (max_lat_delta * max_lon_delta) as usize];
    for i in 0..max_lat_delta {
        for j in 0..max_lon_delta {
            // Java int arithmetic: wraps.
            let box_min_lat = ((lat_base + i) << lat_shift).wrapping_add(i32::MIN);
            let box_min_lon = ((lon_base + j) << lon_shift).wrapping_add(i32::MIN);
            let box_max_lat = box_min_lat.wrapping_add(1 << lat_shift).wrapping_sub(1);
            let box_max_lon = box_min_lon.wrapping_add(1 << lon_shift).wrapping_sub(1);
            // Java's `new Rectangle(...)` validates; decoded values are
            // always in range, so the struct is built directly.
            let rect = Rectangle {
                min_lat: GeoEncodingUtils::decode_latitude(box_min_lat),
                max_lat: GeoEncodingUtils::decode_latitude(box_max_lat),
                min_lon: GeoEncodingUtils::decode_longitude(box_min_lon),
                max_lon: GeoEncodingUtils::decode_longitude(box_max_lon),
            };
            relations[(i * max_lon_delta + j) as usize] = box_to_relation(&rect) as u8;
        }
    }
    // Java's Grid constructor rejects a shift outside 1..=31; compute_shift
    // cannot produce one for a 33-bit span with ARITY = 64.
    debug_assert!((1..=31).contains(&lat_shift) && (1..=31).contains(&lon_shift));
    Ok(Grid {
        lat_shift,
        lon_shift,
        lat_base,
        lon_base,
        max_lat_delta,
        max_lon_delta,
        relations,
    })
}

/// `computeShift`: the smallest shift (at least 1) that leaves fewer than
/// `ARITY` cells between `a` and `b`.
fn compute_shift(a: i64, b: i64) -> i32 {
    let mut shift = 1;
    loop {
        let delta = ((b as u64) >> shift) as i64 - ((a as u64) >> shift) as i64;
        if (0..ARITY).contains(&delta) {
            return shift;
        }
        shift += 1;
    }
}

impl Grid {
    /// The relation of the cell holding `(lat, lon)`, or `None` outside the
    /// grid (the shared head of both `test` methods).
    #[inline]
    fn relation(&self, lat: i32, lon: i32) -> Option<u8> {
        let lat2 = ((lat.wrapping_sub(i32::MIN) as u32) >> self.lat_shift) as i32;
        if lat2 < self.lat_base || lat2.wrapping_sub(self.lat_base) >= self.max_lat_delta {
            return None;
        }
        let mut lon2 = ((lon.wrapping_sub(i32::MIN) as u32) >> self.lon_shift) as i32;
        if lon2 < self.lon_base {
            // wrap
            lon2 = lon2.wrapping_add(1 << (32 - self.lon_shift));
        }
        if lon2.wrapping_sub(self.lon_base) >= self.max_lon_delta {
            return None;
        }
        Some(
            self.relations
                [((lat2 - self.lat_base) * self.max_lon_delta + (lon2 - self.lon_base)) as usize],
        )
    }
}

/// `GeoEncodingUtils.DistancePredicate`: a fast `test(lat, lon)` over
/// encoded points for a distance query.
#[derive(Debug, Clone)]
pub struct DistancePredicate {
    grid: Grid,
    lat: f64,
    lon: f64,
    distance_key: f64,
}

impl DistancePredicate {
    /// `test(int lat, int lon)`.
    pub fn test(&self, lat: i32, lon: i32) -> bool {
        match self.grid.relation(lat, lon) {
            None => false,
            Some(r) if r == Relation::CellCrossesQuery as u8 => {
                sloppy_math::haversin_sort_key(
                    GeoEncodingUtils::decode_latitude(lat),
                    GeoEncodingUtils::decode_longitude(lon),
                    self.lat,
                    self.lon,
                ) <= self.distance_key
            }
            Some(r) => r == Relation::CellInsideQuery as u8,
        }
    }
}

/// `GeoEncodingUtils.Component2DPredicate`: a fast `test(lat, lon)` over
/// encoded points for a `Component2D`.
#[derive(Debug, Clone)]
pub struct Component2DPredicate<'a> {
    grid: Grid,
    tree: TreeRef<'a>,
}

/// The tree a [`Component2DPredicate`] falls back to: borrowed, or shared
/// with its owner.
#[derive(Debug, Clone)]
enum TreeRef<'a> {
    Borrowed(&'a dyn Component2D),
    Shared(std::sync::Arc<dyn Component2D>),
}

impl Component2DPredicate<'_> {
    /// The tree the predicate was built over.
    pub fn tree(&self) -> &dyn Component2D {
        match &self.tree {
            TreeRef::Borrowed(t) => *t,
            TreeRef::Shared(t) => t.as_ref(),
        }
    }

    /// `test(int lat, int lon)`.
    pub fn test(&self, lat: i32, lon: i32) -> bool {
        match self.grid.relation(lat, lon) {
            None => false,
            Some(r) if r == Relation::CellCrossesQuery as u8 => self.tree().contains(
                GeoEncodingUtils::decode_longitude(lon),
                GeoEncodingUtils::decode_latitude(lat),
            ),
            Some(r) => r == Relation::CellInsideQuery as u8,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_predicate_answers_as_the_borrowed_one() {
        use crate::geo::{LatLonGeometry, Polygon};
        let poly = Polygon::new(
            &[0.0, 0.0, 10.0, 10.0, 0.0],
            &[0.0, 10.0, 10.0, 0.0, 0.0],
            vec![],
        )
        .unwrap();
        let tree = LatLonGeometry::Polygon(poly).to_component2d().unwrap();
        let shared: std::sync::Arc<dyn Component2D> = std::sync::Arc::from(tree);
        let borrowed = GeoEncodingUtils::create_component_predicate(shared.as_ref()).unwrap();
        let owned = GeoEncodingUtils::create_component_predicate_shared(shared.clone()).unwrap();
        assert_eq!(owned.tree().max_y(), 10.0);
        for lat in [-1.0, 0.0, 5.0, 9.99, 10.0, 20.0] {
            for lon in [-1.0, 0.0, 5.0, 10.0, 11.0] {
                let (a, b) = (
                    GeoEncodingUtils::encode_latitude(lat).unwrap(),
                    GeoEncodingUtils::encode_longitude(lon).unwrap(),
                );
                assert_eq!(borrowed.test(a, b), owned.test(a, b), "{lat} {lon}");
            }
        }
        assert!(owned.test(
            GeoEncodingUtils::encode_latitude(5.0).unwrap(),
            GeoEncodingUtils::encode_longitude(5.0).unwrap()
        ));
    }

    #[test]
    fn encode_extremes() {
        assert_eq!(
            GeoEncodingUtils::encode_longitude(-180.0).unwrap(),
            i32::MIN
        );
        assert_eq!(GeoEncodingUtils::encode_longitude(180.0).unwrap(), i32::MAX);
        assert_eq!(
            GeoEncodingUtils::encode_longitude_ceil(180.0).unwrap(),
            i32::MAX
        );
        assert_eq!(GeoEncodingUtils::encode_latitude(-90.0).unwrap(), i32::MIN);
        assert_eq!(GeoEncodingUtils::encode_latitude(90.0).unwrap(), i32::MAX);
        assert_eq!(
            GeoEncodingUtils::encode_latitude_ceil(90.0).unwrap(),
            i32::MAX
        );
        assert!(GeoEncodingUtils::encode_latitude(91.0).is_err());
        assert!(GeoEncodingUtils::encode_latitude_ceil(f64::NAN).is_err());
        assert!(GeoEncodingUtils::encode_longitude(181.0).is_err());
        assert!(GeoEncodingUtils::encode_longitude_ceil(-181.0).is_err());
        let mut bytes = [0u8; 4];
        numeric_utils::int_to_sortable_bytes(12345, &mut bytes, 0);
        assert_eq!(
            GeoEncodingUtils::decode_latitude_bytes(&bytes, 0),
            GeoEncodingUtils::decode_latitude(12345)
        );
        assert_eq!(
            GeoEncodingUtils::decode_longitude_bytes(&bytes, 0),
            GeoEncodingUtils::decode_longitude(12345)
        );
    }

    #[test]
    fn predicates_reject_invalid_shapes() {
        assert!(GeoEncodingUtils::create_distance_predicate(91.0, 0.0, 1.0).is_err());
        let off_globe = crate::geo::XYGeometry::Rectangle(
            crate::geo::XYRectangle::new(0.0, 1.0, 0.0, 100.0).unwrap(),
        )
        .to_component2d()
        .unwrap();
        assert!(GeoEncodingUtils::create_component_predicate(off_globe.as_ref()).is_err());
        let p = GeoEncodingUtils::create_distance_predicate(0.0, 0.0, 1000.0).unwrap();
        assert!(p.test(0, 0));
        assert!(!p.test(i32::MAX, 0));
    }

    #[test]
    fn empty_grid_matches_nothing() {
        // A box narrower than one quantization step holds no encoded point.
        let lat = GeoEncodingUtils::decode_latitude(1000);
        let grid =
            create_sub_boxes(lat + 1e-12, lat + 2e-12, 0.0, 1.0, |_| unreachable!()).unwrap();
        assert!(grid.relations.is_empty());
        assert_eq!(grid.relation(1000, 0), None);
    }
}
