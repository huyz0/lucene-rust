//! Port of `org.apache.lucene.geo.XYPolygon`.

use super::geo_utils::WindingOrder;
use super::xy_encoding_utils::XYEncodingUtils;
use super::{java_float_string, java_max_f32, java_min_f32, GeoError};

/// Port of `org.apache.lucene.geo.XYPolygon`: a closed cartesian ring with
/// optional holes.
#[derive(Debug, Clone, PartialEq)]
pub struct XYPolygon {
    x: Vec<f32>,
    y: Vec<f32>,
    holes: Vec<XYPolygon>,
    /// `minX`.
    pub min_x: f32,
    /// `maxX`.
    pub max_x: f32,
    /// `minY`.
    pub min_y: f32,
    /// `maxY`.
    pub max_y: f32,
    winding_order: WindingOrder,
}

impl XYPolygon {
    /// `new XYPolygon(x, y, holes...)`.
    pub fn new(x: &[f32], y: &[f32], holes: Vec<XYPolygon>) -> Result<XYPolygon, GeoError> {
        if x.len() != y.len() {
            return Err(GeoError::illegal("x and y must be equal length"));
        }
        if x.len() < 4 {
            return Err(GeoError::illegal("at least 4 polygon points required"));
        }
        let last = x.len() - 1;
        if x[0] != x[last] {
            return Err(GeoError::illegal(format!(
                "first and last points of the polygon must be the same (it must close itself): x[0]={} x[{}]={}",
                java_float_string(x[0]),
                last,
                java_float_string(x[last])
            )));
        }
        if y[0] != y[last] {
            return Err(GeoError::illegal(format!(
                "first and last points of the polygon must be the same (it must close itself): y[0]={} y[{}]={}",
                java_float_string(y[0]),
                last,
                java_float_string(y[last])
            )));
        }
        if holes.iter().any(|h| !h.holes.is_empty()) {
            return Err(GeoError::illegal(
                "holes may not contain holes: polygons may not nest.",
            ));
        }
        let mut min_x = XYEncodingUtils::check_val(x[0])?;
        let mut max_x = x[0];
        let mut min_y = XYEncodingUtils::check_val(y[0])?;
        let mut max_y = y[0];
        let mut winding_sum = 0f64;
        let num_pts = last;
        let mut j = 0;
        for i in 1..num_pts {
            min_x = java_min_f32(XYEncodingUtils::check_val(x[i])?, min_x);
            max_x = java_max_f32(x[i], max_x);
            min_y = java_min_f32(XYEncodingUtils::check_val(y[i])?, min_y);
            max_y = java_max_f32(y[i], max_y);
            // compute signed area -- in float, as Java writes it
            let area: f32 = (x[j] - x[num_pts]) * (y[i] - y[num_pts])
                - (y[j] - y[num_pts]) * (x[i] - x[num_pts]);
            winding_sum += f64::from(area);
            j = i;
        }
        Ok(XYPolygon {
            x: x.to_vec(),
            y: y.to_vec(),
            holes,
            min_x,
            max_x,
            min_y,
            max_y,
            winding_order: if winding_sum < 0.0 {
                WindingOrder::CCW
            } else {
                WindingOrder::CW
            },
        })
    }

    /// `numPoints()`.
    pub fn num_points(&self) -> usize {
        self.x.len()
    }

    /// `getPolyX()`.
    pub fn poly_x(&self) -> &[f32] {
        &self.x
    }

    /// `getPolyX(vertex)`.
    pub fn poly_x_at(&self, vertex: usize) -> f32 {
        self.x[vertex]
    }

    /// `getPolyY()`.
    pub fn poly_y(&self) -> &[f32] {
        &self.y
    }

    /// `getPolyY(vertex)`.
    pub fn poly_y_at(&self, vertex: usize) -> f32 {
        self.y[vertex]
    }

    /// `getHoles()`.
    pub fn holes(&self) -> &[XYPolygon] {
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

    /// `verticesToGeoJSON(xs, ys)`.
    pub fn vertices_to_geojson(xs: &[f32], ys: &[f32]) -> String {
        let mut sb = String::from("[");
        for i in 0..xs.len() {
            sb.push('[');
            sb.push_str(&java_float_string(xs[i]));
            sb.push_str(", ");
            sb.push_str(&java_float_string(ys[i]));
            sb.push(']');
            if i != xs.len() - 1 {
                sb.push_str(", ");
            }
        }
        sb.push(']');
        sb
    }
}

impl std::fmt::Display for XYPolygon {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("XYPolygon")?;
        for i in 0..self.x.len() {
            write!(
                f,
                "[{}, {}] ",
                java_float_string(self.x[i]),
                java_float_string(self.y[i])
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

    #[test]
    fn construction_and_validation() {
        let hole = XYPolygon::new(&[1.0, 2.0, 2.0, 1.0], &[1.0, 1.0, 2.0, 1.0], vec![]).unwrap();
        let p = XYPolygon::new(
            &[0.0, 3.0, 3.0, 0.0, 0.0],
            &[0.0, 0.0, 3.0, 3.0, 0.0],
            vec![hole.clone()],
        )
        .unwrap();
        let reversed = XYPolygon::new(
            &[0.0, 0.0, 3.0, 3.0, 0.0],
            &[0.0, 3.0, 3.0, 0.0, 0.0],
            vec![],
        )
        .unwrap();
        assert_ne!(p.winding_order(), reversed.winding_order());
        assert_eq!(p.num_points(), 5);
        assert_eq!(p.num_holes(), 1);
        assert_eq!(p.holes()[0], hole);
        assert_eq!((p.poly_x_at(1), p.poly_y_at(2)), (3.0, 3.0));
        assert_eq!(p.poly_x().len(), 5);
        assert_eq!(p.poly_y().len(), 5);
        assert_eq!((p.min_x, p.max_x, p.min_y, p.max_y), (0.0, 3.0, 0.0, 3.0));
        assert!(p.to_string().starts_with("XYPolygon[0.0, 0.0] "));
        assert!(p.to_string().contains("holes=[XYPolygon"));
        assert_eq!(
            XYPolygon::vertices_to_geojson(&[1.0, 2.0], &[3.0, 4.0]),
            "[[1.0, 3.0], [2.0, 4.0]]"
        );
        let e = |r: Result<XYPolygon, GeoError>| r.unwrap_err().to_string();
        assert_eq!(
            e(XYPolygon::new(&[0.0; 4], &[0.0; 5], vec![])),
            "x and y must be equal length"
        );
        assert_eq!(
            e(XYPolygon::new(&[0.0; 3], &[0.0; 3], vec![])),
            "at least 4 polygon points required"
        );
        assert_eq!(
            e(XYPolygon::new(&[0.0, 1.0, 1.0, 2.0], &[0.0; 4], vec![])),
            "first and last points of the polygon must be the same (it must close itself): x[0]=0.0 x[3]=2.0"
        );
        assert_eq!(
            e(XYPolygon::new(&[0.0; 4], &[0.0, 1.0, 1.0, 2.0], vec![])),
            "first and last points of the polygon must be the same (it must close itself): y[0]=0.0 y[3]=2.0"
        );
        assert_eq!(
            e(XYPolygon::new(
                &[0.0, 9.0, 9.0, 0.0],
                &[0.0, 0.0, 9.0, 0.0],
                vec![p]
            )),
            "holes may not contain holes: polygons may not nest."
        );
        assert!(XYPolygon::new(&[0.0, f32::NAN, 1.0, 0.0], &[0.0; 4], vec![]).is_err());
    }
}
