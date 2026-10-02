//! Port of `org.apache.lucene.geo.XYGeometry`: the abstract cartesian
//! geometry, and the factory that turns several into one `Component2D`.

use super::circle2d::Circle2D;
use super::component2d::Component2D;
use super::component_tree::ComponentTree;
use super::line2d::Line2D;
use super::point2d::Point2D;
use super::polygon2d::Polygon2D;
use super::rectangle2d::Rectangle2D;
use super::xy_circle::XYCircle;
use super::xy_line::XYLine;
use super::xy_point::XYPoint;
use super::xy_polygon::XYPolygon;
use super::xy_rectangle::XYRectangle;
use super::GeoError;

/// Port of `org.apache.lucene.geo.XYGeometry`: one of the cartesian
/// geometries.
#[derive(Debug, Clone, PartialEq)]
pub enum XYGeometry {
    /// An [`XYPoint`].
    Point(XYPoint),
    /// An [`XYLine`].
    Line(XYLine),
    /// An [`XYPolygon`].
    Polygon(XYPolygon),
    /// An [`XYCircle`].
    Circle(XYCircle),
    /// An [`XYRectangle`].
    Rectangle(XYRectangle),
}

impl XYGeometry {
    /// `Geometry.toComponent2D()`.
    pub fn to_component2d(&self) -> Result<Box<dyn Component2D>, GeoError> {
        Ok(match self {
            XYGeometry::Point(p) => Box::new(Point2D::create_xy(p)),
            XYGeometry::Line(l) => Box::new(Line2D::create_xy(l)),
            XYGeometry::Polygon(p) => Box::new(Polygon2D::create_xy(p)?),
            XYGeometry::Circle(c) => Box::new(Circle2D::create_xy(c)?),
            XYGeometry::Rectangle(r) => Box::new(Rectangle2D::create_xy(r)),
        })
    }

    /// `XYGeometry.create(geometries...)`.
    pub fn create(geometries: &[XYGeometry]) -> Result<Box<dyn Component2D>, GeoError> {
        if geometries.is_empty() {
            return Err(GeoError::illegal("geometries must not be empty"));
        }
        if geometries.len() == 1 {
            return geometries[0].to_component2d();
        }
        let components = geometries
            .iter()
            .map(XYGeometry::to_component2d)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ComponentTree::create(components))
    }
}
