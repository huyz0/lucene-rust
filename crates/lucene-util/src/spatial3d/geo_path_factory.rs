//! `GeoPathFactory` (`org.apache.lucene.spatial3d.geom.GeoPathFactory`).

use super::geo_degenerate_path::GeoDegeneratePath;
use super::geo_standard_path::GeoStandardPath;
use super::prelude::*;
use super::shape::GeoPath;

/// `makeGeoPath(planetModel, maxCutoffAngle, pathPoints)`: a degenerate
/// (zero-width) path below the angular resolution, a standard one
/// otherwise; consecutive numerically identical points are dropped first.
pub fn make_geo_path(
    planet_model: &Arc<PlanetModel>,
    max_cutoff_angle: f64,
    path_points: &[GeoPoint],
) -> Result<Arc<dyn GeoPath>> {
    let points = filter_points(path_points)?;
    if max_cutoff_angle < MINIMUM_ANGULAR_RESOLUTION {
        return GeoDegeneratePath::new(planet_model, &points)
            .map(|p| Arc::new(p) as Arc<dyn GeoPath>);
    }
    GeoStandardPath::new(planet_model, max_cutoff_angle, &points).map(|s| Arc::new(s) as _)
}

/// `filterPoints(pathPoints)`. Java indexes `pathPoints[length - 1]` of an
/// empty array (`ArrayIndexOutOfBoundsException`); that is an error here.
fn filter_points(path_points: &[GeoPoint]) -> Result<Vec<GeoPoint>> {
    let Some(last) = path_points.last() else {
        return Err(Error::ArrayIndexOutOfBounds(
            "Index -1 out of bounds for length 0".into(),
        ));
    };
    let mut no_identical_points = Vec::with_capacity(path_points.len());
    for w in path_points.windows(2) {
        if !w[0].is_numerically_identical(&w[1]) {
            no_identical_points.push(w[0].clone());
        }
    }
    no_identical_points.push(last.clone());
    Ok(no_identical_points)
}
