//! `DistanceStyle` and its five implementations (`ArcDistance`,
//! `LinearDistance`, `LinearSquaredDistance`, `NormalDistance`,
//! `NormalSquaredDistance` in `org.apache.lucene.spatial3d.geom`): how a
//! distance is measured, combined along a path, and -- for arc distance
//! only -- mapped back to points.
//!
//! Java's interface with singleton implementations is a `Copy` enum here.

use super::geo_point::GeoPoint;
use super::jmath::sqrt;
use super::membership::Membership;
use super::plane::Plane;
use super::planet_model::PlanetModel;
use super::{Error, Result};

/// A way of measuring distance.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DistanceStyle {
    /// `DistanceStyle.ARC`: the arc distance (radians on the unit sphere).
    Arc,
    /// `DistanceStyle.LINEAR`: the straight-line distance.
    Linear,
    /// `DistanceStyle.LINEAR_SQUARED`.
    LinearSquared,
    /// `DistanceStyle.NORMAL`: the perpendicular distance.
    Normal,
    /// `DistanceStyle.NORMAL_SQUARED`.
    NormalSquared,
}

fn not_reversible() -> Error {
    Error::IllegalState("Reverse mapping not implemented for this distance metric".into())
}

impl DistanceStyle {
    /// `computeDistance(point1, point2)`.
    pub fn compute_distance_points(self, point1: &GeoPoint, point2: &GeoPoint) -> f64 {
        match self {
            DistanceStyle::Arc => point1.arc_distance(point2),
            DistanceStyle::Linear => point1.linear_distance(point2),
            DistanceStyle::LinearSquared => point1.linear_distance_squared(point2),
            DistanceStyle::Normal => point1.normal_distance(point2),
            DistanceStyle::NormalSquared => point1.normal_distance_squared(point2),
        }
    }

    /// `computeDistance(point1, x2, y2, z2)`.
    pub fn compute_distance(self, point1: &GeoPoint, x2: f64, y2: f64, z2: f64) -> f64 {
        match self {
            DistanceStyle::Arc => point1.arc_distance_xyz(x2, y2, z2),
            DistanceStyle::Linear => point1.linear_distance_xyz(x2, y2, z2),
            DistanceStyle::LinearSquared => point1.linear_distance_squared_xyz(x2, y2, z2),
            DistanceStyle::Normal => point1.normal_distance_xyz(x2, y2, z2),
            DistanceStyle::NormalSquared => point1.normal_distance_squared_xyz(x2, y2, z2),
        }
    }

    /// `computeDistance(planetModel, plane, x, y, z, bounds)`: the distance
    /// from the point to the bounded plane.
    pub fn compute_distance_to_plane(
        self,
        pm: &PlanetModel,
        plane: &Plane,
        x: f64,
        y: f64,
        z: f64,
        bounds: &[&dyn Membership],
    ) -> f64 {
        match self {
            DistanceStyle::Arc => plane.arc_distance(pm, x, y, z, bounds),
            DistanceStyle::Linear => plane.linear_distance(pm, x, y, z, bounds),
            DistanceStyle::LinearSquared => plane.linear_distance_squared(pm, x, y, z, bounds),
            DistanceStyle::Normal => plane.normal_distance(x, y, z, bounds),
            DistanceStyle::NormalSquared => plane.normal_distance_squared(x, y, z, bounds),
        }
    }

    /// `toAggregationForm(distance)`: the squared styles sum square roots.
    pub fn to_aggregation_form(self, distance: f64) -> f64 {
        match self {
            DistanceStyle::LinearSquared | DistanceStyle::NormalSquared => sqrt(distance),
            _ => distance,
        }
    }

    /// `aggregateDistances(distances...)`: their sum.
    pub fn aggregate_distances(self, distances: &[f64]) -> f64 {
        let mut rval = 0.0;
        for d in distances {
            rval += d;
        }
        rval
    }

    /// `fromAggregationForm(aggregateDistance)`.
    pub fn from_aggregation_form(self, aggregate_distance: f64) -> f64 {
        match self {
            DistanceStyle::LinearSquared | DistanceStyle::NormalSquared => {
                aggregate_distance * aggregate_distance
            }
            _ => aggregate_distance,
        }
    }

    /// `findDistancePoints(planetModel, distanceValue, startPoint, plane,
    /// bounds)`: only arc distance has a reverse mapping.
    pub fn find_distance_points(
        self,
        pm: &PlanetModel,
        distance_value: f64,
        start_point: &GeoPoint,
        plane: &Plane,
        bounds: &[&dyn Membership],
    ) -> Result<Vec<GeoPoint>> {
        match self {
            DistanceStyle::Arc => {
                plane.find_arc_distance_points(pm, distance_value, start_point, bounds)
            }
            _ => Err(not_reversible()),
        }
    }

    /// `findMinimumArcDistance(planetModel, distanceValue)`.
    pub fn find_minimum_arc_distance(self, _pm: &PlanetModel, distance_value: f64) -> Result<f64> {
        match self {
            DistanceStyle::Arc => Ok(distance_value),
            _ => Err(not_reversible()),
        }
    }

    /// `findMaximumArcDistance(planetModel, distanceValue)`.
    pub fn find_maximum_arc_distance(self, _pm: &PlanetModel, distance_value: f64) -> Result<f64> {
        match self {
            DistanceStyle::Arc => Ok(distance_value),
            _ => Err(not_reversible()),
        }
    }
}
