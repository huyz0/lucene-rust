//! `XYZBounds` (`org.apache.lucene.spatial3d.geom.XYZBounds`): a shape's x,
//! y, z extent, each value widened by a fudge factor. Java's nullable
//! `Double` fields are `Option<f64>`.

use super::bounds::Bounds;
use super::geo_point::GeoPoint;
use super::membership::Membership;
use super::plane::Plane;
use super::planet_model::PlanetModel;
use super::vector::{Vector, MINIMUM_RESOLUTION};

/// Added to every recorded value (either way) so a shape's bounds always
/// contain it.
const FUDGE_FACTOR: f64 = MINIMUM_RESOLUTION * 1e3;

/// `XYZBounds`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct XYZBounds {
    min_x: Option<f64>,
    max_x: Option<f64>,
    min_y: Option<f64>,
    max_y: Option<f64>,
    min_z: Option<f64>,
    max_z: Option<f64>,
}

impl XYZBounds {
    /// An empty bounds.
    pub fn new() -> XYZBounds {
        XYZBounds::default()
    }

    /// `getMinimumX()`.
    pub fn minimum_x(&self) -> Option<f64> {
        self.min_x
    }

    /// `getMaximumX()`.
    pub fn maximum_x(&self) -> Option<f64> {
        self.max_x
    }

    /// `getMinimumY()`.
    pub fn minimum_y(&self) -> Option<f64> {
        self.min_y
    }

    /// `getMaximumY()`.
    pub fn maximum_y(&self) -> Option<f64> {
        self.max_y
    }

    /// `getMinimumZ()`.
    pub fn minimum_z(&self) -> Option<f64> {
        self.min_z
    }

    /// `getMaximumZ()`.
    pub fn maximum_z(&self) -> Option<f64> {
        self.max_z
    }

    /// `isSmallestMinX(planetModel)`.
    pub fn is_smallest_min_x(&self, pm: &PlanetModel) -> bool {
        self.min_x
            .is_some_and(|v| v - pm.minimum_x_value() < MINIMUM_RESOLUTION)
    }

    /// `isLargestMaxX(planetModel)`.
    pub fn is_largest_max_x(&self, pm: &PlanetModel) -> bool {
        self.max_x
            .is_some_and(|v| pm.maximum_x_value() - v < MINIMUM_RESOLUTION)
    }

    /// `isSmallestMinY(planetModel)`.
    pub fn is_smallest_min_y(&self, pm: &PlanetModel) -> bool {
        self.min_y
            .is_some_and(|v| v - pm.minimum_y_value() < MINIMUM_RESOLUTION)
    }

    /// `isLargestMaxY(planetModel)`.
    pub fn is_largest_max_y(&self, pm: &PlanetModel) -> bool {
        self.max_y
            .is_some_and(|v| pm.maximum_y_value() - v < MINIMUM_RESOLUTION)
    }

    /// `isSmallestMinZ(planetModel)`.
    pub fn is_smallest_min_z(&self, pm: &PlanetModel) -> bool {
        self.min_z
            .is_some_and(|v| v - pm.minimum_z_value() < MINIMUM_RESOLUTION)
    }

    /// `isLargestMaxZ(planetModel)`.
    pub fn is_largest_max_z(&self, pm: &PlanetModel) -> bool {
        self.max_z
            .is_some_and(|v| pm.maximum_z_value() - v < MINIMUM_RESOLUTION)
    }

    /// `overlaps(bounds)`: whether a corner of either box is inside the
    /// other.
    pub fn overlaps(&self, bounds: &XYZBounds) -> bool {
        is_corner_inside(self, bounds) || is_corner_inside(bounds, self)
    }

    /// `addBounds(bounds)`: widens **`bounds`** (the argument, as Java does)
    /// to include this one. Where this one has no value and `bounds` has
    /// one, Java unboxes a null (`NullPointerException`); here the
    /// comparison is false and `bounds` keeps its value.
    pub fn add_bounds(&self, bounds: &mut XYZBounds) {
        fn wider(mine: Option<f64>, theirs: &mut Option<f64>, larger: bool) {
            let replace = match (mine, *theirs) {
                (_, None) => true,
                (Some(m), Some(t)) => {
                    if larger {
                        m > t
                    } else {
                        m < t
                    }
                }
                (None, Some(_)) => false,
            };
            if replace {
                *theirs = mine;
            }
        }
        wider(self.max_x, &mut bounds.max_x, true);
        wider(self.min_x, &mut bounds.min_x, false);
        wider(self.max_y, &mut bounds.max_y, true);
        wider(self.min_y, &mut bounds.min_y, false);
        wider(self.max_z, &mut bounds.max_z, true);
        wider(self.min_z, &mut bounds.min_z, false);
    }

    /// `addXValue(double)`.
    pub fn add_x_value_f64(&mut self, x: f64) -> &mut XYZBounds {
        let small = x - FUDGE_FACTOR;
        if self.min_x.is_none_or(|m| m > small) {
            self.min_x = Some(small);
        }
        let large = x + FUDGE_FACTOR;
        if self.max_x.is_none_or(|m| m < large) {
            self.max_x = Some(large);
        }
        self
    }

    /// `addYValue(double)`.
    pub fn add_y_value_f64(&mut self, y: f64) -> &mut XYZBounds {
        let small = y - FUDGE_FACTOR;
        if self.min_y.is_none_or(|m| m > small) {
            self.min_y = Some(small);
        }
        let large = y + FUDGE_FACTOR;
        if self.max_y.is_none_or(|m| m < large) {
            self.max_y = Some(large);
        }
        self
    }

    /// `addZValue(double)`.
    pub fn add_z_value_f64(&mut self, z: f64) -> &mut XYZBounds {
        let small = z - FUDGE_FACTOR;
        if self.min_z.is_none_or(|m| m > small) {
            self.min_z = Some(small);
        }
        let large = z + FUDGE_FACTOR;
        if self.max_z.is_none_or(|m| m < large) {
            self.max_z = Some(large);
        }
        self
    }

    /// `isWithin(v)`.
    pub fn is_within(&self, v: &Vector) -> bool {
        self.is_within_xyz(v.x, v.y, v.z)
    }

    /// `isWithin(x, y, z)`: false while any bound is unset.
    pub fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        matches!(
            (self.min_x, self.max_x, self.min_y, self.max_y, self.min_z, self.max_z),
            (Some(a), Some(b), Some(c), Some(d), Some(e), Some(f))
                if x >= a && x <= b && y >= c && y <= d && z >= e && z <= f
        )
    }

    /// Java's `toString()`, `null` for an unset bound.
    pub fn java_to_string(&self) -> String {
        let s = |v: Option<f64>| v.map_or("null".into(), crate::geo::java_double_string);
        format!(
            "XYZBounds: [xmin={} xmax={} ymin={} ymax={} zmin={} zmax={}]",
            s(self.min_x),
            s(self.max_x),
            s(self.min_y),
            s(self.max_y),
            s(self.min_z),
            s(self.max_z)
        )
    }
}

fn all_set(b: &XYZBounds) -> Option<[f64; 6]> {
    Some([b.min_x?, b.max_x?, b.min_y?, b.max_y?, b.min_z?, b.max_z?])
}

fn is_corner_inside(one: &XYZBounds, other: &XYZBounds) -> bool {
    let (Some([ax, bx, ay, by, az, bz]), Some(o)) = (all_set(one), all_set(other)) else {
        return false;
    };
    let inside = |x: f64, y: f64, z: f64| {
        o[0] <= x && o[1] >= x && o[2] <= y && o[3] >= y && o[4] <= z && o[5] >= z
    };
    inside(ax, ay, az)
        || inside(bx, ay, az)
        || inside(ax, by, az)
        || inside(bx, by, az)
        || inside(ax, ay, bz)
        || inside(bx, ay, bz)
        || inside(ax, by, bz)
        || inside(bx, by, bz)
}

impl Bounds for XYZBounds {
    fn add_plane(
        &mut self,
        pm: &PlanetModel,
        plane: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds {
        plane.record_bounds_xyz(pm, self, bounds);
        self
    }

    fn add_horizontal_plane(
        &mut self,
        pm: &PlanetModel,
        _latitude: f64,
        horizontal_plane: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds {
        self.add_plane(pm, horizontal_plane, bounds)
    }

    fn add_vertical_plane(
        &mut self,
        pm: &PlanetModel,
        _longitude: f64,
        vertical_plane: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds {
        self.add_plane(pm, vertical_plane, bounds)
    }

    fn add_intersection(
        &mut self,
        pm: &PlanetModel,
        plane1: &Plane,
        plane2: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds {
        plane1.record_bounds_xyz_intersection(pm, self, plane2, bounds);
        self
    }

    fn add_point(&mut self, point: &GeoPoint) -> &mut dyn Bounds {
        self.add_x_value_f64(point.x);
        self.add_y_value_f64(point.y);
        self.add_z_value_f64(point.z)
    }

    fn add_x_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds {
        self.add_x_value_f64(point.x)
    }

    fn add_y_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds {
        self.add_y_value_f64(point.y)
    }

    fn add_z_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds {
        self.add_z_value_f64(point.z)
    }

    fn is_wide(&mut self) -> &mut dyn Bounds {
        self
    }

    fn no_longitude_bound(&mut self) -> &mut dyn Bounds {
        self
    }

    fn no_top_latitude_bound(&mut self) -> &mut dyn Bounds {
        self
    }

    fn no_bottom_latitude_bound(&mut self) -> &mut dyn Bounds {
        self
    }

    fn no_bound(&mut self, pm: &PlanetModel) -> &mut dyn Bounds {
        self.min_x = Some(pm.minimum_x_value());
        self.max_x = Some(pm.maximum_x_value());
        self.min_y = Some(pm.minimum_y_value());
        self.max_y = Some(pm.maximum_y_value());
        self.min_z = Some(pm.minimum_z_value());
        self.max_z = Some(pm.maximum_z_value());
        self
    }

    fn as_xyz_bounds(&mut self) -> Option<&mut XYZBounds> {
        Some(self)
    }
}
