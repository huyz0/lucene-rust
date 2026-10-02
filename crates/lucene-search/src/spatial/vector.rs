//! `org.apache.lucene.spatial.vector`: [`PointVectorStrategy`] -- a point
//! indexes as two doubles (`__x`, `__y`), stored, as doc values and/or as
//! points -- and its [`DistanceValueSource`].

use std::fmt;
use std::sync::Arc;

use lucene_index::document::{
    DocValuesType, DoublePoint, Field, FieldType, NumericDocValuesField, StoredValue,
};
use lucene_util::spatial4j::{DistanceCalculator, Point, Rectangle, Shape, SpatialContext};
use lucene_util::spatial_extras::query::{SpatialArgs, SpatialOperation};

use super::bbox::numeric;
use super::bool_query::{docs_of, BoolQuery, ConstantScoreBool};
use super::util::WithDefault;
use super::{check_field_name, Fields, SpatialStrategy};
use crate::collector::ScoringCollector;
use crate::document::geo::idx;
use crate::document::{collect_live, double_point, reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::reader::NumericDocValues;
use crate::values_source::{
    doc_values_cacheable, BoxDoubleValues, DoubleValues, DoubleValuesSource, ValuesContext,
};
use crate::{Error, Result};

/// `PointVectorStrategy.SUFFIX_X`/`SUFFIX_Y`.
pub const SUFFIX_X: &str = "__x";
pub const SUFFIX_Y: &str = "__y";

fn unsupported(message: impl Into<String>) -> Error {
    Error::Spatial(lucene_util::spatial4j::Error::UnsupportedOperation(Some(
        message.into(),
    )))
}

/// `PointVectorStrategy`.
#[derive(Clone)]
pub struct PointVectorStrategy {
    ctx: Arc<SpatialContext>,
    field_name: String,
    field_name_x: String,
    field_name_y: String,
    has_stored: bool,
    has_doc_vals: bool,
    has_point_vals: bool,
}

impl fmt::Debug for PointVectorStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl PointVectorStrategy {
    /// `PointVectorStrategy.newInstance(ctx, fieldNamePrefix)`: points and
    /// doc values.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new_instance(ctx: Arc<SpatialContext>, field_name_prefix: &str) -> Result<Self> {
        Self::new(ctx, field_name_prefix, super::bbox::default_field_type())
    }

    /// `new PointVectorStrategy(ctx, fieldNamePrefix, fieldType)`.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new(
        ctx: Arc<SpatialContext>,
        field_name_prefix: &str,
        field_type: FieldType,
    ) -> Result<Self> {
        check_field_name(field_name_prefix)?;
        Ok(PointVectorStrategy {
            ctx,
            field_name: field_name_prefix.to_string(),
            field_name_x: format!("{field_name_prefix}{SUFFIX_X}"),
            field_name_y: format!("{field_name_prefix}{SUFFIX_Y}"),
            has_stored: field_type.stored(),
            has_doc_vals: field_type.doc_values_type() != DocValuesType::None,
            has_point_vals: field_type.point_dimension_count() > 0,
        })
    }

    /// `createIndexableFields(point)`: stored, doc values, points.
    ///
    /// # Errors
    /// Never for a finite point.
    pub fn create_point_fields(&self, point: &dyn Point) -> Result<Fields> {
        let illegal = |e: lucene_index::document::Error| Error::IllegalArgument(e.to_string());
        let mut fields: Fields = Vec::new();
        let values = [
            (&self.field_name_x, point.x()),
            (&self.field_name_y, point.y()),
        ];
        if self.has_stored {
            for (name, v) in values {
                fields.push(Box::new(Field::stored(
                    name.clone(),
                    StoredValue::Double(v),
                )));
            }
        }
        if self.has_doc_vals {
            for (name, v) in values {
                fields.push(Box::new(NumericDocValuesField::new(
                    name.clone(),
                    v.to_bits() as i64,
                )));
            }
        }
        if self.has_point_vals {
            for (name, v) in values {
                fields.push(Box::new(
                    DoublePoint::new(name.clone(), &[v]).map_err(illegal)?,
                ));
            }
        }
        Ok(fields)
    }

    /// `rangeQuery(field, min, max)`: an absent bound is infinite.
    fn range_query(&self, field: &str, min: Option<f64>, max: Option<f64>) -> Result<BoolQuery> {
        if self.has_point_vals {
            return Ok(BoolQuery::Clause(Box::new(double_point::new_range_query(
                field,
                min.unwrap_or(f64::NEG_INFINITY),
                max.unwrap_or(f64::INFINITY),
            )?)));
        }
        Err(unsupported("An index is required for this operation."))
    }

    /// `makeWithin(bbox)`: x within (either side of the dateline), y
    /// within.
    fn make_within(&self, bbox: &dyn Rectangle) -> Result<BoolQuery> {
        let (mut must, mut should, mut min_should_match) = (Vec::new(), Vec::new(), 0);
        if bbox.crosses_date_line() {
            // no data is beyond the world bounds
            should.push(self.range_query(&self.field_name_x, None, Some(bbox.max_x()))?);
            should.push(self.range_query(&self.field_name_x, Some(bbox.min_x()), None)?);
            min_should_match = 1; // at least one of the SHOULD
        } else {
            must.push(self.range_query(
                &self.field_name_x,
                Some(bbox.min_x()),
                Some(bbox.max_x()),
            )?);
        }
        must.push(self.range_query(&self.field_name_y, Some(bbox.min_y()), Some(bbox.max_y()))?);
        Ok(BoolQuery::Bool {
            must,
            should,
            must_not: Vec::new(),
            min_should_match,
        })
    }
}

impl fmt::Display for PointVectorStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&super::strategy_string(
            "PointVectorStrategy",
            &self.field_name,
            &self.ctx,
        ))
    }
}

impl SpatialStrategy for PointVectorStrategy {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.ctx
    }

    fn field_name(&self) -> &str {
        &self.field_name
    }

    fn create_indexable_fields(&self, shape: &Arc<dyn Shape>) -> Result<Fields> {
        match shape.as_point() {
            Some(p) => self.create_point_fields(p),
            None => Err(unsupported(format!("Can only index Point, not {shape}"))),
        }
    }

    fn make_distance_value_source(
        &self,
        query_point: &Arc<dyn Point>,
        multiplier: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>> {
        Ok(Arc::new(DistanceValueSource::new(
            self.clone(),
            query_point.clone(),
            multiplier,
        )))
    }

    fn make_query(&self, args: &SpatialArgs) -> Result<Box<dyn DocumentQuery>> {
        if !matches!(
            args.operation,
            SpatialOperation::Intersects | SpatialOperation::IsWithin
        ) {
            return Err(Error::Spatial(args.operation.unsupported()));
        }
        let shape = &args.shape;
        if let Some(bbox) = shape.as_rectangle() {
            return Ok(Box::new(ConstantScoreBool(self.make_within(bbox)?)));
        }
        if let Some(circle) = shape.as_circle() {
            let bbox = circle.bounding_box()?;
            let center = circle.center()?;
            return Ok(Box::new(DistanceRangeQuery {
                inner: ConstantScoreBool(self.make_within(&*bbox)?),
                distance_source: self.make_distance_value_source(&center, 1.0)?,
                limit: circle.radius(),
            }));
        }
        Err(unsupported(format!(
            "Only Rectangles and Circles are currently supported, found [{}]",
            super::shape_class(&**shape)
        )))
    }
}

/// `PointVectorStrategy.DistanceRangeQuery`: the bounding box's matches
/// whose distance is at most the limit.
#[derive(Debug)]
pub struct DistanceRangeQuery {
    inner: ConstantScoreBool,
    distance_source: Arc<dyn DoubleValuesSource>,
    limit: f64,
}

impl DocumentQuery for DistanceRangeQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let r = reader(leaf)?;
        let approximation = self.inner.0.matches(leaf, idx(r.max_doc))?;
        let ctx = ValuesContext::for_reader(r);
        let mut v = self.distance_source.get_values(&ctx, 0, None)?;
        for doc in docs_of(&approximation) {
            if v.advance_exact(doc)? && v.double_value()? <= self.limit {
                collect_live(leaf, doc, boost, collector);
            }
        }
        Ok(())
    }
}

/// `DistanceValueSource`: the distance from a point to each document's
/// point, times a multiplier; 180 times the multiplier without one.
pub struct DistanceValueSource {
    strategy: PointVectorStrategy,
    from: Arc<dyn Point>,
    multiplier: f64,
    null_value: f64,
}

impl DistanceValueSource {
    pub fn new(strategy: PointVectorStrategy, from: Arc<dyn Point>, multiplier: f64) -> Self {
        DistanceValueSource {
            strategy,
            from,
            multiplier,
            null_value: 180.0 * multiplier,
        }
    }
}

struct DistanceValues<'c> {
    pt_x: Option<Box<dyn NumericDocValues + 'c>>,
    pt_y: Option<Box<dyn NumericDocValues + 'c>>,
    from: Arc<dyn Point>,
    calculator: Arc<dyn DistanceCalculator>,
    multiplier: f64,
}

impl DoubleValues for DistanceValues<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        let x = match &mut self.pt_x {
            Some(dv) => dv.advance_exact(doc)?,
            None => false,
        };
        if !x {
            return Ok(false);
        }
        match &mut self.pt_y {
            Some(dv) => dv.advance_exact(doc),
            None => Ok(false),
        }
    }

    fn double_value(&mut self) -> Result<f64> {
        let x = f64::from_bits(self.pt_x.as_ref().map_or(0, |d| d.long_value()) as u64);
        let y = f64::from_bits(self.pt_y.as_ref().map_or(0, |d| d.long_value()) as u64);
        Ok(self.calculator.distance_xy(&*self.from, x, y)? * self.multiplier)
    }
}

impl DoubleValuesSource for DistanceValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
        _scores: Option<BoxDoubleValues<'c>>,
    ) -> Result<BoxDoubleValues<'c>> {
        let reader = ctx.leaf_reader(leaf)?;
        Ok(WithDefault::boxed(
            Box::new(DistanceValues {
                pt_x: numeric(reader, &self.strategy.field_name_x)?,
                pt_y: numeric(reader, &self.strategy.field_name_y)?,
                from: self.from.clone(),
                calculator: self.strategy.ctx.dist_calc().clone(),
                multiplier: self.multiplier,
            }),
            self.null_value,
        ))
    }

    fn needs_scores(&self) -> bool {
        false
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        doc_values_cacheable(ctx, leaf, &self.strategy.field_name_x)
            && doc_values_cacheable(ctx, leaf, &self.strategy.field_name_y)
    }

    fn describe(&self) -> String {
        format!("DistanceValueSource({}, {})", self.strategy, self.from)
    }
}
