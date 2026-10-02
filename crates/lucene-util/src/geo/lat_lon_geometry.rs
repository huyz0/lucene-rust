//! Port of `org.apache.lucene.geo.Geometry` and `LatLonGeometry`: the
//! abstract lat/lon geometry, and the factory that turns several into one
//! `Component2D`.

use super::circle::Circle;
use super::circle2d::Circle2D;
use super::component2d::Component2D;
use super::component_tree::ComponentTree;
use super::line::Line;
use super::line2d::Line2D;
use super::point::Point;
use super::point2d::Point2D;
use super::polygon::Polygon;
use super::polygon2d::Polygon2D;
use super::rectangle::Rectangle;
use super::rectangle2d::Rectangle2D;
use super::GeoError;

/// Port of `org.apache.lucene.geo.LatLonGeometry` (and its base
/// `Geometry`): one of the lat/lon geometries.
#[derive(Debug, Clone, PartialEq)]
pub enum LatLonGeometry {
    /// A [`Point`].
    Point(Point),
    /// A [`Line`].
    Line(Line),
    /// A [`Polygon`].
    Polygon(Polygon),
    /// A [`Circle`].
    Circle(Circle),
    /// A [`Rectangle`].
    Rectangle(Rectangle),
}

impl LatLonGeometry {
    /// `Geometry.toComponent2D()`.
    pub fn to_component2d(&self) -> Result<Box<dyn Component2D>, GeoError> {
        Ok(match self {
            LatLonGeometry::Point(p) => Box::new(Point2D::create(p)?),
            LatLonGeometry::Line(l) => Box::new(Line2D::create(l)),
            LatLonGeometry::Polygon(p) => Box::new(Polygon2D::create(p)?),
            LatLonGeometry::Circle(c) => Box::new(Circle2D::create(c)?),
            LatLonGeometry::Rectangle(r) => Rectangle2D::create(r)?,
        })
    }

    /// `LatLonGeometry.create(geometries...)`: one component, or a
    /// `ComponentTree` over several.
    pub fn create(geometries: &[LatLonGeometry]) -> Result<Box<dyn Component2D>, GeoError> {
        if geometries.is_empty() {
            return Err(GeoError::illegal("geometries must not be empty"));
        }
        if geometries.len() == 1 {
            return geometries[0].to_component2d();
        }
        let components = geometries
            .iter()
            .map(LatLonGeometry::to_component2d)
            .collect::<Result<Vec<_>, _>>()?;
        Ok(ComponentTree::create(components))
    }
}
