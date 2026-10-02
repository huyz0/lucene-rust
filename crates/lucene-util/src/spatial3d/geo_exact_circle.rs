//! `GeoExactCircle` (`org.apache.lucene.spatial3d.geom.GeoExactCircle`): a
//! circle on an ellipsoid, approximated to a requested accuracy by slices,
//! each a plane through three points of the true (Vincenty) circle.

use super::prelude::*;
use super::shape::{impl_distance_shape, GeoCircle};

/// `ApproximationSlice`: a slice still being refined.
struct ApproximationSlice {
    plane: SidedPlane,
    end_point1: GeoPoint,
    point1_bearing: f64,
    end_point2: GeoPoint,
    point2_bearing: f64,
    middle_point: GeoPoint,
    middle_point_bearing: f64,
    must_split: bool,
}

impl ApproximationSlice {
    #[allow(clippy::too_many_arguments)]
    fn new(
        center: &GeoPoint,
        end_point1: GeoPoint,
        point1_bearing: f64,
        end_point2: GeoPoint,
        point2_bearing: f64,
        middle_point: GeoPoint,
        middle_point_bearing: f64,
        must_split: bool,
    ) -> Result<ApproximationSlice> {
        use crate::geo::java_double_string as d;
        // Construct the plane going through the three given points
        let Some(plane) = SidedPlane::construct_normalized_three_point_sided_plane(
            center,
            &end_point1,
            &end_point2,
            &middle_point,
        ) else {
            return Err(Error::IllegalArgument(format!(
                "Either circle is too small or accuracy is too high; could not construct a plane with endPoint1={} bearing {}, endPoint2={} bearing {}, middle={} bearing {}",
                end_point1,
                d(point1_bearing),
                end_point2,
                d(point2_bearing),
                middle_point,
                d(middle_point_bearing)
            )));
        };
        if plane.is_within_xyz(-center.x, -center.y, -center.z) {
            return Err(Error::IllegalArgument(format!(
                "Could not construct a valid plane for this planet model with endPoint1={} bearing {}, endPoint2={} bearing {}, middle={} bearing {}",
                end_point1,
                d(point1_bearing),
                end_point2,
                d(point2_bearing),
                middle_point,
                d(middle_point_bearing)
            )));
        }
        Ok(ApproximationSlice {
            plane,
            end_point1,
            point1_bearing,
            end_point2,
            point2_bearing,
            middle_point,
            middle_point_bearing,
            must_split,
        })
    }
}

/// `CircleSlice`: one final slice -- the circle plane bounded by the two
/// planes through its end points and the center.
#[derive(Debug, Clone)]
struct CircleSlice {
    notable_edge_points: [GeoPoint; 2],
    circle_plane: SidedPlane,
    plane1: SidedPlane,
    plane2: SidedPlane,
}

impl CircleSlice {
    fn new(
        circle_plane: SidedPlane,
        end_point1: GeoPoint,
        end_point2: GeoPoint,
        center: &GeoPoint,
        check: &GeoPoint,
    ) -> Result<CircleSlice> {
        let plane1 = SidedPlane::from_vectors(check, &end_point1, center)?;
        let plane2 = SidedPlane::from_vectors(check, &end_point2, center)?;
        Ok(CircleSlice {
            notable_edge_points: [end_point1, end_point2],
            circle_plane,
            plane1,
            plane2,
        })
    }
}

/// An exact circle.
#[derive(Debug, Clone)]
pub struct GeoExactCircle {
    planet_model: Arc<PlanetModel>,
    center: GeoPoint,
    radius: f64,
    actual_accuracy: f64,
    edge_points: [GeoPoint; 1],
    circle_slices: Vec<CircleSlice>,
}

impl GeoExactCircle {
    /// `GeoExactCircle(planetModel, lat, lon, radius, accuracy)`.
    pub fn new(
        planet_model: &Arc<PlanetModel>,
        lat: f64,
        lon: f64,
        radius: f64,
        accuracy: f64,
    ) -> Result<GeoExactCircle> {
        let pm = &**planet_model;
        if lat < -PI * 0.5 || lat > PI * 0.5 {
            return Err(illegal("Latitude out of bounds"));
        }
        if lon < -PI || lon > PI {
            return Err(illegal("Longitude out of bounds"));
        }
        if radius < 0.0 {
            return Err(illegal("Radius out of bounds"));
        }
        if radius < MINIMUM_RESOLUTION {
            return Err(illegal("Radius cannot be effectively zero"));
        }
        if pm.minimum_pole_distance - radius < MINIMUM_RESOLUTION {
            return Err(Error::IllegalArgument(format!(
                "Radius out of bounds. It cannot be bigger than {} for this planet model",
                crate::geo::java_double_string(pm.minimum_pole_distance)
            )));
        }
        let center = GeoPoint::from_lat_lon(pm, lat, lon)?;
        let actual_accuracy = if accuracy < MINIMUM_RESOLUTION {
            MINIMUM_RESOLUTION
        } else {
            accuracy
        };
        // We construct approximation planes until we have a low enough error
        // estimate
        let mut slices: Vec<ApproximationSlice> = Vec::with_capacity(100);
        // Construct four cardinal points, and then we'll build the first two
        // planes
        let north_point = pm.surface_point_on_bearing(&center, radius, 0.0)?;
        let south_point = pm.surface_point_on_bearing(&center, radius, PI)?;
        let east_point = pm.surface_point_on_bearing(&center, radius, PI * 0.5)?;
        let west_point = pm.surface_point_on_bearing(&center, radius, PI * 1.5)?;
        // Java's declare-then-assign, kept.
        #[allow(clippy::needless_late_init)]
        let edge_point;
        if pm.z_scaling > pm.xy_scaling {
            // z can be greater than x or y, so ellipse is longer in height than
            // width
            slices.push(ApproximationSlice::new(
                &center,
                east_point.clone(),
                PI * 0.5,
                west_point.clone(),
                PI * -0.5,
                north_point.clone(),
                0.0,
                true,
            )?);
            slices.push(ApproximationSlice::new(
                &center,
                west_point.clone(),
                PI * 1.5,
                east_point.clone(),
                PI * 0.5,
                south_point.clone(),
                PI,
                true,
            )?);
            edge_point = east_point;
        } else {
            // z will be less than x or y, so ellipse is shorter than it is tall
            slices.push(ApproximationSlice::new(
                &center,
                north_point.clone(),
                0.0,
                south_point.clone(),
                PI,
                east_point.clone(),
                PI * 0.5,
                true,
            )?);
            slices.push(ApproximationSlice::new(
                &center,
                south_point.clone(),
                PI,
                north_point.clone(),
                PI * 2.0,
                west_point.clone(),
                PI * 1.5,
                true,
            )?);
            edge_point = north_point;
        }
        let mut circle_slices = Vec::new();
        // Now, iterate over slices until we have converted all of them into
        // safe SidedPlanes.
        while let Some(this_slice) = slices.pop() {
            // Compute the midpoints between end points and middle point
            let interp_point1_bearing =
                (this_slice.point1_bearing + this_slice.middle_point_bearing) * 0.5;
            let interp_point1 =
                pm.surface_point_on_bearing(&center, radius, interp_point1_bearing)?;
            let interp_point2_bearing =
                (this_slice.point2_bearing + this_slice.middle_point_bearing) * 0.5;
            let interp_point2 =
                pm.surface_point_on_bearing(&center, radius, interp_point2_bearing)?;
            // Is this point on the plane? (that is, is the approximation good
            // enough?)
            if !this_slice.must_split
                && abs(this_slice.plane.evaluate(&interp_point1)) < actual_accuracy
                && abs(this_slice.plane.evaluate(&interp_point2)) < actual_accuracy
            {
                circle_slices.push(CircleSlice::new(
                    this_slice.plane,
                    this_slice.end_point1,
                    this_slice.end_point2,
                    &center,
                    &this_slice.middle_point,
                )?);
            } else {
                // Split the plane into two, and add it back to the end
                slices.push(ApproximationSlice::new(
                    &center,
                    this_slice.end_point1,
                    this_slice.point1_bearing,
                    this_slice.middle_point.clone(),
                    this_slice.middle_point_bearing,
                    interp_point1,
                    interp_point1_bearing,
                    false,
                )?);
                slices.push(ApproximationSlice::new(
                    &center,
                    this_slice.middle_point,
                    this_slice.middle_point_bearing,
                    this_slice.end_point2,
                    this_slice.point2_bearing,
                    interp_point2,
                    interp_point2_bearing,
                    false,
                )?);
            }
        }
        Ok(GeoExactCircle {
            planet_model: planet_model.clone(),
            center,
            radius,
            actual_accuracy,
            edge_points: [edge_point],
            circle_slices,
        })
    }

    /// `GeoExactCircle(planetModel, InputStream)`.
    pub fn read(planet_model: &Arc<PlanetModel>, input: &mut Input<'_>) -> Result<GeoExactCircle> {
        let lat = read_double(input)?;
        let lon = read_double(input)?;
        let radius = read_double(input)?;
        let accuracy = read_double(input)?;
        GeoExactCircle::new(planet_model, lat, lon, radius, accuracy)
    }

    fn distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        style.compute_distance(&self.center, x, y, z)
    }

    fn delta_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        self.distance(style, x, y, z) * 2.0
    }

    fn distance_bounds(
        &self,
        bounds: &mut dyn Bounds,
        _style: DistanceStyle,
        _distance_value: f64,
    ) -> Result<()> {
        // TBD: Compute actual bounds based on distance
        self.get_bounds(bounds);
        Ok(())
    }

    fn outside_distance(&self, style: DistanceStyle, x: f64, y: f64, z: f64) -> f64 {
        let mut outside_distance = f64::INFINITY;
        for slice in &self.circle_slices {
            let distance = style.compute_distance_to_plane(
                &self.planet_model,
                &slice.circle_plane,
                x,
                y,
                z,
                &[&slice.plane1, &slice.plane2],
            );
            if distance < outside_distance {
                outside_distance = distance;
            }
        }
        outside_distance
    }
}

impl SerializableObject for GeoExactCircle {
    fn write(&self, out: &mut Vec<u8>) -> Result<()> {
        write_double(out, self.center.latitude());
        write_double(out, self.center.longitude());
        write_double(out, self.radius);
        write_double(out, self.actual_accuracy);
        Ok(())
    }

    fn class_code(&self) -> Option<u8> {
        Some(37)
    }
}

impl_planet_object!(GeoExactCircle);
impl_membership_shape!(GeoExactCircle);
impl_base_area!(GeoExactCircle);
impl_distance_shape!(GeoExactCircle);

impl Membership for GeoExactCircle {
    fn is_within_xyz(&self, x: f64, y: f64, z: f64) -> bool {
        self.circle_slices.iter().any(|s| {
            s.circle_plane.is_within_xyz(x, y, z)
                && s.plane1.is_within_xyz(x, y, z)
                && s.plane2.is_within_xyz(x, y, z)
        })
    }
}

impl Bounded for GeoExactCircle {
    fn get_bounds(&self, bounds: &mut dyn Bounds) {
        let pm = &*self.planet_model;
        base_get_bounds(self, pm, bounds);
        bounds.add_point(&self.center);
        for slice in &self.circle_slices {
            bounds.add_plane(pm, &slice.circle_plane, &[&slice.plane1, &slice.plane2]);
            for point in &slice.notable_edge_points {
                bounds.add_point(point);
            }
        }
    }
}

impl GeoShape for GeoExactCircle {
    fn edge_points(&self) -> Cow<'_, [GeoPoint]> {
        Cow::Borrowed(&self.edge_points)
    }

    fn intersects(
        &self,
        p: &Plane,
        notable_points: &[GeoPoint],
        bounds: &[&dyn Membership],
    ) -> bool {
        self.circle_slices.iter().any(|s| {
            s.circle_plane.intersects(
                &self.planet_model,
                p,
                notable_points,
                &s.notable_edge_points,
                bounds,
                &[&s.plane1, &s.plane2],
            )
        })
    }
}

impl GeoAreaShape for GeoExactCircle {
    fn intersects_shape(&self, geo_shape: &dyn GeoShape) -> bool {
        self.circle_slices.iter().any(|s| {
            geo_shape.intersects(
                &s.circle_plane,
                &s.notable_edge_points,
                &[&s.plane1, &s.plane2],
            )
        })
    }
}

impl GeoSizeable for GeoExactCircle {
    fn radius(&self) -> f64 {
        self.radius
    }

    fn center(&self) -> GeoPoint {
        self.center.clone()
    }
}

impl GeoCircle for GeoExactCircle {}
