//! Port of `org.apache.lucene.geo.Line`.

use super::geo_utils::GeoUtils;
use super::polygon::Polygon;
use super::{java_double_string, java_max, java_min, GeoError};

/// Port of `org.apache.lucene.geo.Line`: a lat/lon polyline of at least two
/// points, with its bounding box.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    lats: Vec<f64>,
    lons: Vec<f64>,
    /// `minLat`.
    pub min_lat: f64,
    /// `maxLat`.
    pub max_lat: f64,
    /// `minLon`.
    pub min_lon: f64,
    /// `maxLon`.
    pub max_lon: f64,
}

impl Line {
    /// `new Line(lats, lons)`.
    pub fn new(lats: &[f64], lons: &[f64]) -> Result<Line, GeoError> {
        if lats.len() != lons.len() {
            return Err(GeoError::illegal("lats and lons must be equal length"));
        }
        if lats.len() < 2 {
            return Err(GeoError::illegal("at least 2 line points required"));
        }
        let mut min_lat = lats[0];
        let mut min_lon = lons[0];
        let mut max_lat = lats[0];
        let mut max_lon = lons[0];
        for i in 0..lats.len() {
            GeoUtils::check_latitude(lats[i])?;
            GeoUtils::check_longitude(lons[i])?;
            min_lat = java_min(lats[i], min_lat);
            min_lon = java_min(lons[i], min_lon);
            max_lat = java_max(lats[i], max_lat);
            max_lon = java_max(lons[i], max_lon);
        }
        Ok(Line {
            lats: lats.to_vec(),
            lons: lons.to_vec(),
            min_lat,
            max_lat,
            min_lon,
            max_lon,
        })
    }

    /// `numPoints()`.
    pub fn num_points(&self) -> usize {
        self.lats.len()
    }

    /// `getLat(vertex)`.
    pub fn lat(&self, vertex: usize) -> f64 {
        self.lats[vertex]
    }

    /// `getLon(vertex)`.
    pub fn lon(&self, vertex: usize) -> f64 {
        self.lons[vertex]
    }

    /// `getLats()`.
    pub fn lats(&self) -> &[f64] {
        &self.lats
    }

    /// `getLons()`.
    pub fn lons(&self) -> &[f64] {
        &self.lons
    }

    /// `toGeoJSON()`.
    pub fn to_geojson(&self) -> String {
        format!("[{}]", Polygon::vertices_to_geojson(&self.lats, &self.lons))
    }
}

impl std::fmt::Display for Line {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Line(")?;
        for i in 0..self.lats.len() {
            write!(
                f,
                "[{}, {}]",
                java_double_string(self.lons[i]),
                java_double_string(self.lats[i])
            )?;
        }
        f.write_str(")")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn construction() {
        let l = Line::new(&[1.0, 2.0, -3.0], &[10.0, -20.0, 30.0]).unwrap();
        assert_eq!(l.num_points(), 3);
        assert_eq!(
            (l.min_lat, l.max_lat, l.min_lon, l.max_lon),
            (-3.0, 2.0, -20.0, 30.0)
        );
        assert_eq!(l.lat(1), 2.0);
        assert_eq!(l.lon(2), 30.0);
        assert_eq!(l.lats(), &[1.0, 2.0, -3.0]);
        assert_eq!(l.lons().len(), 3);
        assert_eq!(l.to_string(), "Line([10.0, 1.0][-20.0, 2.0][30.0, -3.0])");
        assert_eq!(
            l.to_geojson(),
            "[[[10.0, 1.0], [-20.0, 2.0], [30.0, -3.0]]]"
        );
        assert_eq!(
            Line::new(&[1.0], &[1.0, 2.0]).unwrap_err().to_string(),
            "lats and lons must be equal length"
        );
        assert_eq!(
            Line::new(&[1.0], &[1.0]).unwrap_err().to_string(),
            "at least 2 line points required"
        );
        assert!(Line::new(&[1.0, 100.0], &[1.0, 2.0]).is_err());
    }
}
