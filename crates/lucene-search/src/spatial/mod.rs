//! Lucene's `lucene-spatial-extras` (`org.apache.lucene.spatial`): the
//! [`SpatialStrategy`]s that index Spatial4j shapes into ordinary fields and
//! query them back -- over the shapes, prefix trees and `SpatialArgs` of
//! [`lucene_util::spatial4j`] and [`lucene_util::spatial_extras`].
//!
//! - [`prefix`]: `PrefixTreeStrategy` and its three kinds --
//!   [`RecursivePrefixTreeStrategy`] (any shape; intersects, within and
//!   contains by walking the indexed cells against the query shape's),
//!   [`TermQueryPrefixTreeStrategy`] (a `TermInSetQuery` of the query shape's
//!   cells) and [`NumberRangePrefixTreeStrategy`] (number and date ranges);
//!   the visiting queries, `PrefixTreeFacetCounter`,
//!   `HeatmapFacetCounter` and the field-cache distance source.
//! - [`bbox`]: `BBoxStrategy` (four doubles and a dateline flag) with its
//!   value source and the overlap-ratio similarity.
//! - [`vector`]: `PointVectorStrategy` (two doubles) and its distance source.
//! - [`serialized`]: `SerializedDVStrategy` (the shape's `BinaryCodec` bytes in
//!   a binary doc value, verified per document).
//! - [`composite`]: `CompositeSpatialStrategy` (a prefix-tree approximation
//!   verified against the serialized shape).
//! - [`util`]: the shape value sources and predicates the strategies share.
//!
//! # Shape
//!
//! A strategy is a [`SpatialStrategy`] trait object. Its fields are the
//! [`IndexableField`]s `lucene_index` indexes; its queries are
//! [`DocumentQuery`]s and its value sources [`DoubleValuesSource`]s, so they
//! run with the rest of `crate::document`. A Java setter on a strategy is a
//! `&mut self` method, called before the strategy is shared.
//!
//! # Deliberate differences
//!
//! - `equals`/`hashCode` and `QueryVisitor` are not ported, as elsewhere in
//!   `crate::document`.
//! - A query's `DocIdSet` is computed whole per segment (a bitset), then its
//!   live documents are collected in order -- what Java's constant-score
//!   scorer over the same set yields.
//! - Java caches a prefix tree field's points per `IndexReader`
//!   (`ShapeFieldCacheProvider`'s `WeakHashMap`); here the cache is built per
//!   `get_values` call (see [`util::PointPrefixTreeFieldCacheProvider`]).

pub mod bbox;
mod bool_query;
pub mod composite;
pub mod prefix;
pub mod serialized;
pub mod util;
pub mod vector;

use std::sync::Arc;

use lucene_index::document::IndexableField;
use lucene_util::spatial4j::{Point, Shape, SpatialContext};
use lucene_util::spatial_extras::query::SpatialArgs;

use crate::document::DocumentQuery;
use crate::values_source::DoubleValuesSource;
use crate::{Error, Result};

pub use bbox::BBoxStrategy;
pub use composite::CompositeSpatialStrategy;
pub use prefix::{
    NumberRangePrefixTreeStrategy, PrefixTreeStrategy, RecursivePrefixTreeStrategy,
    TermQueryPrefixTreeStrategy,
};
pub use serialized::SerializedDVStrategy;
pub use vector::PointVectorStrategy;

/// The fields one shape indexes as.
pub type Fields = Vec<Box<dyn IndexableField>>;

/// `SpatialStrategy`: how a shape becomes indexed fields, and how a
/// `SpatialArgs` becomes a query over them. `Display` is Java's `toString()`.
pub trait SpatialStrategy: std::fmt::Display + Send + Sync {
    /// `getSpatialContext()`.
    fn spatial_context(&self) -> &Arc<SpatialContext>;

    /// `getFieldName()`.
    fn field_name(&self) -> &str;

    /// `createIndexableFields(shape)`.
    ///
    /// # Errors
    /// A shape the strategy cannot index, with Java's exception.
    fn create_indexable_fields(&self, shape: &Arc<dyn Shape>) -> Result<Fields>;

    /// `makeDistanceValueSource(queryPoint, multiplier)`.
    ///
    /// # Errors
    /// `UnsupportedOperationException` for a strategy without distances.
    fn make_distance_value_source(
        &self,
        query_point: &Arc<dyn Point>,
        multiplier: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>>;

    /// `makeQuery(args)`.
    ///
    /// # Errors
    /// An unsupported operation or shape, with Java's exception.
    fn make_query(&self, args: &SpatialArgs) -> Result<Box<dyn DocumentQuery>>;

    /// The strategy as a `PrefixTreeStrategy`, which heatmaps need (Java's
    /// cast); `None` for the others.
    fn as_prefix_tree(&self) -> Option<&PrefixTreeStrategy> {
        None
    }
}

/// `SpatialStrategy`'s constructor checks: a field name is required.
pub(crate) fn check_field_name(field_name: &str) -> Result<()> {
    if field_name.is_empty() {
        return Err(Error::IllegalArgument("fieldName is required".into()));
    }
    Ok(())
}

/// `SpatialStrategy.toString()`: the class's simple name, the field and the
/// context.
pub(crate) fn strategy_string(class: &str, field: &str, ctx: &SpatialContext) -> String {
    format!("{class} field:{field} ctx={ctx}")
}

/// `makeDistanceValueSource(queryPoint)`: the multiplier 1.
///
/// # Errors
/// As [`SpatialStrategy::make_distance_value_source`].
pub fn make_distance_value_source(
    strategy: &dyn SpatialStrategy,
    query_point: &Arc<dyn Point>,
) -> Result<Arc<dyn DoubleValuesSource>> {
    strategy.make_distance_value_source(query_point, 1.0)
}

/// `makeRecipDistanceValueSource(queryShape)`: `c / (d + c)` of the distance
/// `d` from the shape's center, `c` a tenth of the distance from the center to
/// a corner of its bounding box (in `float`, as Java's).
///
/// # Errors
/// The shape's geometry errors, or the strategy's distance source's.
pub fn make_recip_distance_value_source(
    strategy: &dyn SpatialStrategy,
    query_shape: &Arc<dyn Shape>,
) -> Result<Arc<dyn DoubleValuesSource>> {
    let ctx = strategy.spatial_context();
    let bbox = query_shape.bounding_box()?;
    let corner = ctx.point_xy(bbox.min_x(), bbox.min_y())?;
    let diagonal_dist = ctx
        .dist_calc()
        .distance_xy(&*corner, bbox.max_x(), bbox.max_y())?;
    let dist_to_edge = diagonal_dist * 0.5;
    let c = (dist_to_edge as f32) * 0.1f32; // one tenth
    let center = query_shape.center()?;
    let distance = strategy.make_distance_value_source(&center, 1.0)?;
    Ok(Arc::new(util::ReciprocalDoubleValuesSource::new(
        f64::from(c),
        distance,
    )))
}

/// The Java class of a shape, for the messages that name it
/// (`shape.getClass()`).
pub(crate) fn shape_class(shape: &dyn Shape) -> String {
    use lucene_util::spatial4j::{CircleImpl, ShapeCollection};
    use lucene_util::spatial_extras::prefix_tree::{SpanUnitsNRShape, UnitNRShape};
    let any = shape.as_any();
    let name = if shape.as_point().is_some()
        && lucene_util::spatial_extras::spatial4j::geo3d_class_name(shape).is_none()
    {
        "org.locationtech.spatial4j.shape.impl.PointImpl"
    } else if let Some(name) = lucene_util::spatial_extras::spatial4j::geo3d_class_name(shape) {
        name
    } else if shape.as_rectangle().is_some() {
        "org.locationtech.spatial4j.shape.impl.RectangleImpl"
    } else if let Some(c) = any.downcast_ref::<CircleImpl>() {
        if c.context().is_some_and(|ctx| ctx.is_geo()) {
            "org.locationtech.spatial4j.shape.impl.GeoCircle"
        } else {
            "org.locationtech.spatial4j.shape.impl.CircleImpl"
        }
    } else if any.is::<ShapeCollection>() {
        "org.locationtech.spatial4j.shape.ShapeCollection"
    } else if any.is::<UnitNRShape>() {
        "org.apache.lucene.spatial.prefix.tree.NumberRangePrefixTree$NRCell"
    } else if any.is::<SpanUnitsNRShape>() {
        "org.apache.lucene.spatial.prefix.tree.NumberRangePrefixTree$SpanUnitsNRShape"
    } else {
        lucene_util::spatial4j::binary_codec::java_class_name(shape)
    };
    format!("class {name}")
}

/// Java's `ClassCastException` for casting `shape` to `target`.
pub(crate) fn class_cast(shape: &dyn Shape, target: &str) -> Error {
    let class = shape_class(shape);
    let class = class.strip_prefix("class ").unwrap_or(&class);
    Error::Spatial(lucene_util::spatial4j::Error::ClassCast(format!(
        "class {class} cannot be cast to class {target}"
    )))
}

#[cfg(test)]
mod tests;
