//! `LatLonBounds` (`org.apache.lucene.spatial3d.geom.LatLonBounds`): a
//! shape's latitude/longitude extent. Java's nullable `Double` fields are
//! `Option<f64>`.

use super::bounds::Bounds;
use super::geo_point::GeoPoint;
use super::membership::Membership;
use super::plane::Plane;
use super::planet_model::PlanetModel;

use std::f64::consts::PI;

/// `LatLonBounds`.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LatLonBounds {
    no_longitude_bound: bool,
    no_top_latitude_bound: bool,
    no_bottom_latitude_bound: bool,
    min_latitude: Option<f64>,
    max_latitude: Option<f64>,
    left_longitude: Option<f64>,
    right_longitude: Option<f64>,
}

impl LatLonBounds {
    /// An empty bounds.
    pub fn new() -> LatLonBounds {
        LatLonBounds::default()
    }

    /// `toString()`.
    pub fn java_to_string(&self) -> String {
        let v = |no_bound: bool, value: Option<f64>| -> String {
            if no_bound {
                "no bound".to_string()
            } else {
                value.map_or("null".to_string(), crate::geo::java_double_string)
            }
        };
        format!(
            "LatLonBounds [minLat={}, maxLat={}, leftLon={}, rightLon={}]",
            v(self.no_bottom_latitude_bound, self.min_latitude),
            v(self.no_top_latitude_bound, self.max_latitude),
            v(self.no_longitude_bound, self.left_longitude),
            v(self.no_longitude_bound, self.right_longitude)
        )
    }

    /// `getMaxLatitude()`.
    pub fn max_latitude(&self) -> Option<f64> {
        self.max_latitude
    }

    /// `getMinLatitude()`.
    pub fn min_latitude(&self) -> Option<f64> {
        self.min_latitude
    }

    /// `getLeftLongitude()`.
    pub fn left_longitude(&self) -> Option<f64> {
        self.left_longitude
    }

    /// `getRightLongitude()`.
    pub fn right_longitude(&self) -> Option<f64> {
        self.right_longitude
    }

    /// `checkNoLongitudeBound()`.
    pub fn check_no_longitude_bound(&self) -> bool {
        self.no_longitude_bound
    }

    /// `checkNoTopLatitudeBound()`.
    pub fn check_no_top_latitude_bound(&self) -> bool {
        self.no_top_latitude_bound
    }

    /// `checkNoBottomLatitudeBound()`.
    pub fn check_no_bottom_latitude_bound(&self) -> bool {
        self.no_bottom_latitude_bound
    }

    fn add_latitude_bound(&mut self, latitude: f64) {
        if !self.no_top_latitude_bound && self.max_latitude.is_none_or(|m| latitude > m) {
            self.max_latitude = Some(latitude);
        }
        if !self.no_bottom_latitude_bound && self.min_latitude.is_none_or(|m| latitude < m) {
            self.min_latitude = Some(latitude);
        }
    }

    fn add_longitude_bound(&mut self, longitude: f64) {
        let mut longitude = longitude;
        match (self.left_longitude, self.right_longitude) {
            (None, None) => {
                self.left_longitude = Some(longitude);
                self.right_longitude = Some(longitude);
            }
            (left, right) => {
                // Java unboxes both; one null and one not cannot happen (they
                // are set and cleared together).
                let left = left.unwrap_or(f64::NAN);
                let right = right.unwrap_or(f64::NAN);
                let mut current_left_longitude = left;
                let mut current_right_longitude = right;
                if current_right_longitude < current_left_longitude {
                    current_right_longitude += 2.0 * PI;
                }
                if longitude < current_left_longitude {
                    longitude += 2.0 * PI;
                }
                if longitude < current_left_longitude || longitude > current_right_longitude {
                    let left_extension_amt = if longitude < current_left_longitude {
                        current_left_longitude - longitude
                    } else {
                        current_left_longitude + 2.0 * PI - longitude
                    };
                    let right_extension_amt = if longitude > current_right_longitude {
                        longitude - current_right_longitude
                    } else {
                        longitude + 2.0 * PI - current_right_longitude
                    };
                    if left_extension_amt < right_extension_amt {
                        current_left_longitude = left - left_extension_amt;
                        while current_left_longitude <= -PI {
                            current_left_longitude += 2.0 * PI;
                        }
                        self.left_longitude = Some(current_left_longitude);
                    } else {
                        current_right_longitude = right + right_extension_amt;
                        while current_right_longitude > PI {
                            current_right_longitude -= 2.0 * PI;
                        }
                        self.right_longitude = Some(current_right_longitude);
                    }
                }
            }
        }
        let left = self.left_longitude.unwrap_or(f64::NAN);
        let mut test_right_longitude = self.right_longitude.unwrap_or(f64::NAN);
        if test_right_longitude < left {
            test_right_longitude += PI * 2.0;
        }
        if test_right_longitude - left >= PI {
            self.no_longitude_bound = true;
            self.left_longitude = None;
            self.right_longitude = None;
        }
    }
}

impl Bounds for LatLonBounds {
    fn add_plane(
        &mut self,
        pm: &PlanetModel,
        plane: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds {
        plane.record_bounds_lat_lon(pm, self, bounds);
        self
    }

    fn add_horizontal_plane(
        &mut self,
        _pm: &PlanetModel,
        latitude: f64,
        _horizontal_plane: &Plane,
        _bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds {
        if !self.no_top_latitude_bound || !self.no_bottom_latitude_bound {
            self.add_latitude_bound(latitude);
        }
        self
    }

    fn add_vertical_plane(
        &mut self,
        _pm: &PlanetModel,
        longitude: f64,
        _vertical_plane: &Plane,
        _bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds {
        if !self.no_longitude_bound {
            self.add_longitude_bound(longitude);
        }
        self
    }

    fn add_intersection(
        &mut self,
        pm: &PlanetModel,
        plane1: &Plane,
        plane2: &Plane,
        bounds: &[&dyn Membership],
    ) -> &mut dyn Bounds {
        plane1.record_bounds_lat_lon_intersection(pm, self, plane2, bounds);
        self
    }

    fn add_point(&mut self, point: &GeoPoint) -> &mut dyn Bounds {
        if !self.no_longitude_bound {
            // Get a longitude value
            self.add_longitude_bound(point.longitude());
        }
        if !self.no_top_latitude_bound || !self.no_bottom_latitude_bound {
            // Compute a latitude value
            self.add_latitude_bound(point.latitude());
        }
        self
    }

    fn add_x_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds {
        if !self.no_longitude_bound {
            self.add_longitude_bound(point.longitude());
        }
        self
    }

    fn add_y_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds {
        if !self.no_longitude_bound {
            self.add_longitude_bound(point.longitude());
        }
        self
    }

    fn add_z_value(&mut self, point: &GeoPoint) -> &mut dyn Bounds {
        if !self.no_top_latitude_bound || !self.no_bottom_latitude_bound {
            self.add_latitude_bound(point.latitude());
        }
        self
    }

    fn is_wide(&mut self) -> &mut dyn Bounds {
        self.no_longitude_bound()
    }

    fn no_longitude_bound(&mut self) -> &mut dyn Bounds {
        self.no_longitude_bound = true;
        self.left_longitude = None;
        self.right_longitude = None;
        self
    }

    fn no_top_latitude_bound(&mut self) -> &mut dyn Bounds {
        self.no_top_latitude_bound = true;
        self.max_latitude = None;
        self
    }

    fn no_bottom_latitude_bound(&mut self) -> &mut dyn Bounds {
        self.no_bottom_latitude_bound = true;
        self.min_latitude = None;
        self
    }

    fn no_bound(&mut self, _pm: &PlanetModel) -> &mut dyn Bounds {
        self.no_longitude_bound();
        self.no_top_latitude_bound();
        self.no_bottom_latitude_bound()
    }
}
