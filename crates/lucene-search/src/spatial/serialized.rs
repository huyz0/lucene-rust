//! `org.apache.lucene.spatial.serialized`: [`SerializedDVStrategy`] -- the
//! shape's `BinaryCodec` bytes in a binary doc value, each document's shape
//! decoded and tested against the query shape.

use std::fmt;
use std::sync::Arc;

use lucene_index::document::BinaryDocValuesField;
use lucene_util::spatial4j::binary_codec::DataInput;
use lucene_util::spatial4j::{Point, Shape, SpatialContext};
use lucene_util::spatial_extras::query::SpatialArgs;

use super::util::{
    DistanceToShapeValueSource, ShapeValues, ShapeValuesPredicate, ShapeValuesSource,
};
use super::{check_field_name, Fields, SpatialStrategy};
use crate::collector::ScoringCollector;
use crate::document::{reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::reader::{BinaryDocValues, LeafReader};
use crate::values_source::{doc_values_cacheable, DoubleValuesSource, ValuesContext};
use crate::Result;

/// `SerializedDVStrategy`.
#[derive(Clone)]
pub struct SerializedDVStrategy {
    ctx: Arc<SpatialContext>,
    field_name: String,
}

impl fmt::Debug for SerializedDVStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl SerializedDVStrategy {
    /// `new SerializedDVStrategy(ctx, fieldName)`.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new(ctx: Arc<SpatialContext>, field_name: &str) -> Result<Self> {
        check_field_name(field_name)?;
        Ok(SerializedDVStrategy {
            ctx,
            field_name: field_name.to_string(),
        })
    }

    /// `makeShapeValueSource()`: each document's decoded shape.
    pub fn make_shape_value_source(&self) -> Arc<dyn ShapeValuesSource> {
        Arc::new(ShapeDocValueSource {
            field_name: self.field_name.clone(),
            ctx: self.ctx.clone(),
        })
    }
}

impl fmt::Display for SerializedDVStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&super::strategy_string(
            "SerializedDVStrategy",
            &self.field_name,
            &self.ctx,
        ))
    }
}

impl SpatialStrategy for SerializedDVStrategy {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        &self.ctx
    }

    fn field_name(&self) -> &str {
        &self.field_name
    }

    /// The shape's `BinaryCodec` bytes as a `BinaryDocValuesField`. (Java's
    /// `indexLastBufSize` is a buffer-size heuristic with no effect on the
    /// bytes.)
    fn create_indexable_fields(&self, shape: &Arc<dyn Shape>) -> Result<Fields> {
        let mut bytes = Vec::with_capacity(128);
        self.ctx.binary_codec().write_shape(&mut bytes, &**shape)?;
        Ok(vec![Box::new(BinaryDocValuesField::new(
            self.field_name.clone(),
            bytes,
        ))])
    }

    fn make_distance_value_source(
        &self,
        query_point: &Arc<dyn Point>,
        multiplier: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>> {
        Ok(Arc::new(DistanceToShapeValueSource::new(
            self.make_shape_value_source(),
            query_point.clone(),
            multiplier,
            &self.ctx,
        )))
    }

    fn make_query(&self, args: &SpatialArgs) -> Result<Box<dyn DocumentQuery>> {
        Ok(Box::new(PredicateValueSourceQuery {
            predicate: ShapeValuesPredicate::new(
                self.make_shape_value_source(),
                args.operation,
                args.shape.clone(),
            ),
        }))
    }
}

/// `SerializedDVStrategy.PredicateValueSourceQuery`: every live document
/// whose shape satisfies the predicate (`DocIdSetIterator.all(maxDoc)`
/// verified per document).
#[derive(Debug)]
pub struct PredicateValueSourceQuery {
    pub predicate: ShapeValuesPredicate,
}

impl DocumentQuery for PredicateValueSourceQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let r = reader(leaf)?;
        let ctx = ValuesContext::for_reader(r);
        let mut m = self.predicate.matcher(&ctx, 0)?;
        for doc in 0..r.max_doc {
            // `DefaultBulkScorer`: the live check before `matches()`, so a
            // deleted document's shape is never decoded.
            if !leaf.live_docs.is_none_or(|bits| bits.get_doc(doc)) {
                continue;
            }
            if m.matches(doc)? {
                collector.collect(doc, boost);
            }
        }
        Ok(())
    }
}

/// `SerializedDVStrategy.ShapeDocValueSource`: the shape decoded from each
/// document's binary doc value with the context's `BinaryCodec`.
pub struct ShapeDocValueSource {
    field_name: String,
    ctx: Arc<SpatialContext>,
}

impl fmt::Display for ShapeDocValueSource {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "shapeDocVal({})", self.field_name)
    }
}

struct DocShapes<'c> {
    values: Option<Box<dyn BinaryDocValues + 'c>>,
    ctx: Arc<SpatialContext>,
}

impl ShapeValues for DocShapes<'_> {
    fn advance_exact(&mut self, doc: i32) -> Result<bool> {
        match &mut self.values {
            Some(v) => v.advance_exact(doc),
            None => Ok(false),
        }
    }

    fn value(&mut self) -> Result<Arc<dyn Shape>> {
        let bytes = self.values.as_ref().map_or(&[][..], |v| v.binary_value());
        let mut input = DataInput::new(bytes);
        Ok(self.ctx.binary_codec().read_shape(&self.ctx, &mut input)?)
    }
}

impl ShapeValuesSource for ShapeDocValueSource {
    fn get_values<'c>(
        &self,
        ctx: &ValuesContext<'c>,
        leaf: usize,
    ) -> Result<Box<dyn ShapeValues + 'c>> {
        let reader = ctx.leaf_reader(leaf)?;
        Ok(Box::new(DocShapes {
            values: reader.binary_doc_values(&self.field_name)?,
            ctx: self.ctx.clone(),
        }))
    }

    fn is_cacheable(&self, ctx: &ValuesContext<'_>, leaf: usize) -> bool {
        doc_values_cacheable(ctx, leaf, &self.field_name)
    }
}
