//! Port of `org.apache.lucene.geo.Polygon`.

use super::geo_utils::{GeoUtils, WindingOrder};
use super::{java_double_string, java_max, java_min, GeoError};

/// Port of `org.apache.lucene.geo.Polygon`: a closed lat/lon ring with
/// optional holes (which may not have holes of their own).
#[derive(Debug, Clone, PartialEq)]
pub struct Polygon {
    poly_lats: Vec<f64>,
    poly_lons: Vec<f64>,
    holes: Vec<Polygon>,
    /// `minLat`.
    pub min_lat: f64,
    /// `maxLat`.
    pub max_lat: f64,
    /// `minLon`.
    pub min_lon: f64,
    /// `maxLon`.
    pub max_lon: f64,
    winding_order: WindingOrder,
}

impl Polygon {
    /// `new Polygon(polyLats, polyLons, holes...)`.
    pub fn new(
        poly_lats: &[f64],
        poly_lons: &[f64],
        holes: Vec<Polygon>,
    ) -> Result<Polygon, GeoError> {
        if poly_lats.len() != poly_lons.len() {
            return Err(GeoError::illegal(
                "polyLats and polyLons must be equal length",
            ));
        }
        if poly_lats.len() < 4 {
            return Err(GeoError::illegal("at least 4 polygon points required"));
        }
        let last = poly_lats.len() - 1;
        if poly_lats[0] != poly_lats[last] {
            return Err(GeoError::illegal(format!(
                "first and last points of the polygon must be the same (it must close itself): polyLats[0]={} polyLats[{}]={}",
                java_double_string(poly_lats[0]),
                last,
                java_double_string(poly_lats[last])
            )));
        }
        if poly_lons[0] != poly_lons[last] {
            return Err(GeoError::illegal(format!(
                "first and last points of the polygon must be the same (it must close itself): polyLons[0]={} polyLons[{}]={}",
                java_double_string(poly_lons[0]),
                last,
                java_double_string(poly_lons[last])
            )));
        }
        for i in 0..poly_lats.len() {
            GeoUtils::check_latitude(poly_lats[i])?;
            GeoUtils::check_longitude(poly_lons[i])?;
        }
        if holes.iter().any(|h| !h.holes.is_empty()) {
            return Err(GeoError::illegal(
                "holes may not contain holes: polygons may not nest.",
            ));
        }
        let mut min_lat = poly_lats[0];
        let mut max_lat = poly_lats[0];
        let mut min_lon = poly_lons[0];
        let mut max_lon = poly_lons[0];
        let mut winding_sum = 0f64;
        let num_pts = last;
        let mut j = 0;
        for i in 1..num_pts {
            min_lat = java_min(poly_lats[i], min_lat);
            max_lat = java_max(poly_lats[i], max_lat);
            min_lon = java_min(poly_lons[i], min_lon);
            max_lon = java_max(poly_lons[i], max_lon);
            // compute signed area
            winding_sum += (poly_lons[j] - poly_lons[num_pts])
                * (poly_lats[i] - poly_lats[num_pts])
                - (poly_lats[j] - poly_lats[num_pts]) * (poly_lons[i] - poly_lons[num_pts]);
            j = i;
        }
        Ok(Polygon {
            poly_lats: poly_lats.to_vec(),
            poly_lons: poly_lons.to_vec(),
            holes,
            min_lat,
            max_lat,
            min_lon,
            max_lon,
            winding_order: if winding_sum < 0.0 {
                WindingOrder::CCW
            } else {
                WindingOrder::CW
            },
        })
    }

    /// `numPoints()`.
    pub fn num_points(&self) -> usize {
        self.poly_lats.len()
    }

    /// `getPolyLats()`.
    pub fn poly_lats(&self) -> &[f64] {
        &self.poly_lats
    }

    /// `getPolyLat(vertex)`.
    pub fn poly_lat(&self, vertex: usize) -> f64 {
        self.poly_lats[vertex]
    }

    /// `getPolyLons()`.
    pub fn poly_lons(&self) -> &[f64] {
        &self.poly_lons
    }

    /// `getPolyLon(vertex)`.
    pub fn poly_lon(&self, vertex: usize) -> f64 {
        self.poly_lons[vertex]
    }

    /// `getHoles()`.
    pub fn holes(&self) -> &[Polygon] {
        &self.holes
    }

    /// `getWindingOrder()`.
    pub fn winding_order(&self) -> WindingOrder {
        self.winding_order
    }

    /// `numHoles()`.
    pub fn num_holes(&self) -> usize {
        self.holes.len()
    }

    /// `verticesToGeoJSON(lats, lons)`.
    pub fn vertices_to_geojson(lats: &[f64], lons: &[f64]) -> String {
        let mut sb = String::from("[");
        for i in 0..lats.len() {
            sb.push('[');
            sb.push_str(&java_double_string(lons[i]));
            sb.push_str(", ");
            sb.push_str(&java_double_string(lats[i]));
            sb.push(']');
            if i != lats.len() - 1 {
                sb.push_str(", ");
            }
        }
        sb.push(']');
        sb
    }

    /// `toGeoJSON()`.
    pub fn to_geojson(&self) -> String {
        let mut sb = String::from("[");
        sb.push_str(&Self::vertices_to_geojson(&self.poly_lats, &self.poly_lons));
        for hole in &self.holes {
            sb.push(',');
            sb.push_str(&Self::vertices_to_geojson(&hole.poly_lats, &hole.poly_lons));
        }
        sb.push(']');
        sb
    }

    /// `fromGeoJSON(geojson)`.
    pub fn from_geojson(geojson: &str) -> Result<Vec<Polygon>, GeoError> {
        super::simple_geojson_polygon_parser::SimpleGeoJSONPolygonParser::new(geojson).parse()
    }
}

impl std::fmt::Display for Polygon {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Polygon")?;
        for i in 0..self.poly_lats.len() {
            write!(
                f,
                "[{}, {}] ",
                java_double_string(self.poly_lats[i]),
                java_double_string(self.poly_lons[i])
            )?;
        }
        if !self.holes.is_empty() {
            f.write_str(", holes=[")?;
            for (i, h) in self.holes.iter().enumerate() {
                if i > 0 {
                    f.write_str(", ")?;
                }
                write!(f, "{h}")?;
            }
            f.write_str("]")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn square(min: f64, max: f64) -> Polygon {
        Polygon::new(
            &[min, min, max, max, min],
            &[min, max, max, min, min],
            vec![],
        )
        .unwrap()
    }

    #[test]
    fn construction_and_accessors() {
        let hole = square(1.0, 2.0);
        let p = Polygon::new(
            &[0.0, 0.0, 3.0, 3.0, 0.0],
            &[0.0, 3.0, 3.0, 0.0, 0.0],
            vec![hole.clone()],
        )
        .unwrap();
        assert_eq!(p.num_points(), 5);
        assert_eq!(p.num_holes(), 1);
        assert_eq!(p.holes()[0], hole);
        assert_eq!(p.poly_lat(2), 3.0);
        assert_eq!(p.poly_lon(1), 3.0);
        assert_eq!(p.poly_lats().len(), 5);
        assert_eq!(p.poly_lons().len(), 5);
        assert_eq!(
            (p.min_lat, p.max_lat, p.min_lon, p.max_lon),
            (0.0, 3.0, 0.0, 3.0)
        );
        assert_eq!(p.winding_order(), WindingOrder::CW);
        let ccw = Polygon::new(
            &[0.0, 3.0, 3.0, 0.0, 0.0],
            &[0.0, 0.0, 3.0, 3.0, 0.0],
            vec![],
        )
        .unwrap();
        assert_eq!(ccw.winding_order(), WindingOrder::CCW);
        assert!(p.to_string().starts_with("Polygon[0.0, 0.0] "));
        assert!(p.to_string().contains(", holes=[Polygon[1.0, 1.0] "));
        assert_eq!(
            hole.to_geojson(),
            "[[[1.0, 1.0], [2.0, 1.0], [2.0, 2.0], [1.0, 2.0], [1.0, 1.0]]]"
        );
        assert!(p.to_geojson().contains("]],[["));
    }

    #[test]
    fn validation() {
        let e = |r: Result<Polygon, GeoError>| r.unwrap_err().to_string();
        assert_eq!(
            e(Polygon::new(&[0.0; 4], &[0.0; 5], vec![])),
            "polyLats and polyLons must be equal length"
        );
        assert_eq!(
            e(Polygon::new(&[0.0; 3], &[0.0; 3], vec![])),
            "at least 4 polygon points required"
        );
        assert_eq!(
            e(Polygon::new(&[0.0, 1.0, 2.0, 3.0], &[0.0; 4], vec![])),
            "first and last points of the polygon must be the same (it must close itself): polyLats[0]=0.0 polyLats[3]=3.0"
        );
        assert_eq!(
            e(Polygon::new(&[0.0; 4], &[0.0, 1.0, 2.0, 3.0], vec![])),
            "first and last points of the polygon must be the same (it must close itself): polyLons[0]=0.0 polyLons[3]=3.0"
        );
        assert!(Polygon::new(&[0.0, 95.0, 1.0, 0.0], &[0.0; 4], vec![]).is_err());
        let with_hole = Polygon::new(
            &[0.0, 0.0, 3.0, 3.0, 0.0],
            &[0.0, 3.0, 3.0, 0.0, 0.0],
            vec![square(1.0, 2.0)],
        )
        .unwrap();
        assert_eq!(
            e(Polygon::new(
                &[0.0, 0.0, 9.0, 9.0, 0.0],
                &[0.0, 9.0, 9.0, 0.0, 0.0],
                vec![with_hole]
            )),
            "holes may not contain holes: polygons may not nest."
        );
    }
}
