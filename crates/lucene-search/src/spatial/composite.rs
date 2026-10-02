//! `org.apache.lucene.spatial.composite`: [`CompositeSpatialStrategy`] --
//! an RPT index for a fast approximation, a serialized shape to verify each
//! candidate -- with [`CompositeVerifyQuery`] and [`IntersectsRPTVerifyQuery`]
//! (an intersects walk that tells exact matches from candidates).

use std::fmt;
use std::sync::Arc;

use lucene_util::fixed_bit_set::FixedBitSet;
use lucene_util::spatial4j::{Point, Shape, SpatialContext, SpatialRelation};
use lucene_util::spatial_extras::prefix_tree::Cell;
use lucene_util::spatial_extras::query::{SpatialArgs, SpatialOperation};

use super::bool_query::docs_of;
use super::prefix::query::{visit, Traverser, VisitingQuery, Visitor};
use super::prefix::RecursivePrefixTreeStrategy;
use super::serialized::SerializedDVStrategy;
use super::util::ShapeValuesPredicate;
use super::{Fields, SpatialStrategy};
use crate::collector::ScoringCollector;
use crate::document::geo::{idx, set_doc};
use crate::document::{reader, DocumentQuery};
use crate::multi_segment::OpenSegment;
use crate::values_source::{DoubleValuesSource, ValuesContext};
use crate::{Error, Result};

/// `CompositeSpatialStrategy`.
#[derive(Clone)]
pub struct CompositeSpatialStrategy {
    field_name: String,
    index_strategy: RecursivePrefixTreeStrategy,
    geometry_strategy: SerializedDVStrategy,
    optimize_predicates: bool,
}

impl fmt::Debug for CompositeSpatialStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

impl CompositeSpatialStrategy {
    /// `new CompositeSpatialStrategy(fieldName, indexStrategy,
    /// geometryStrategy)`: the field name is unused but for `toString`.
    ///
    /// # Errors
    /// An empty field name.
    pub fn new(
        field_name: &str,
        index_strategy: RecursivePrefixTreeStrategy,
        geometry_strategy: SerializedDVStrategy,
    ) -> Result<Self> {
        super::check_field_name(field_name)?;
        Ok(CompositeSpatialStrategy {
            field_name: field_name.to_string(),
            index_strategy,
            geometry_strategy,
            optimize_predicates: true,
        })
    }

    /// `getIndexStrategy()`.
    pub fn index_strategy(&self) -> &RecursivePrefixTreeStrategy {
        &self.index_strategy
    }

    /// `getGeometryStrategy()`.
    pub fn geometry_strategy(&self) -> &SerializedDVStrategy {
        &self.geometry_strategy
    }

    /// `isOptimizePredicates()`.
    pub fn is_optimize_predicates(&self) -> bool {
        self.optimize_predicates
    }

    /// `setOptimizePredicates(v)`: intersects through
    /// [`IntersectsRPTVerifyQuery`] (the default), or every candidate
    /// verified.
    pub fn set_optimize_predicates(&mut self, v: bool) {
        self.optimize_predicates = v;
    }
}

impl fmt::Display for CompositeSpatialStrategy {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&super::strategy_string(
            "CompositeSpatialStrategy",
            &self.field_name,
            self.index_strategy.spatial_context(),
        ))
    }
}

impl SpatialStrategy for CompositeSpatialStrategy {
    fn spatial_context(&self) -> &Arc<SpatialContext> {
        self.index_strategy.spatial_context()
    }

    fn field_name(&self) -> &str {
        &self.field_name
    }

    fn create_indexable_fields(&self, shape: &Arc<dyn Shape>) -> Result<Fields> {
        let mut fields = self.index_strategy.create_indexable_fields(shape)?;
        fields.extend(self.geometry_strategy.create_indexable_fields(shape)?);
        Ok(fields)
    }

    fn make_distance_value_source(
        &self,
        _query_point: &Arc<dyn Point>,
        _multiplier: f64,
    ) -> Result<Arc<dyn DoubleValuesSource>> {
        Err(Error::Spatial(
            lucene_util::spatial4j::Error::UnsupportedOperation(None),
        ))
    }

    fn make_query(&self, args: &SpatialArgs) -> Result<Box<dyn DocumentQuery>> {
        let pred = args.operation;
        if matches!(
            pred,
            SpatialOperation::BBoxIntersects
                | SpatialOperation::BBoxWithin
                | SpatialOperation::IsDisjointTo
        ) {
            return Err(Error::Spatial(pred.unsupported()));
        }
        let predicate = ShapeValuesPredicate::new(
            self.geometry_strategy.make_shape_value_source(),
            pred,
            args.shape.clone(),
        );
        let ctx = self.spatial_context();
        if pred == SpatialOperation::Intersects && self.optimize_predicates {
            // the smart Intersects
            let grid = self.index_strategy.base().grid();
            // default to max precision
            let detail_level = grid.level_for_distance(args.resolve_dist_err(ctx, 0.0)?);
            return Ok(Box::new(IntersectsRPTVerifyQuery {
                q: VisitingQuery::new(
                    args.shape.clone(),
                    self.index_strategy.field_name(),
                    grid.clone(),
                    detail_level,
                    self.index_strategy.prefix_grid_scan_level(),
                ),
                predicate,
            }));
        }
        // The general path; all index matches get verified
        let mut index_args = if pred == SpatialOperation::Contains {
            args.clone()
        } else {
            let mut a = SpatialArgs::new(SpatialOperation::Intersects, args.shape.clone());
            a.set_dist_err(args.dist_err());
            a.set_dist_err_pct(args.dist_err_pct());
            a
        };
        if index_args.dist_err().is_none() && index_args.dist_err_pct().is_none() {
            index_args.set_dist_err_pct(Some(0.10));
        }
        let index_query = self.index_strategy.make_query(&index_args)?;
        Ok(Box::new(CompositeVerifyQuery {
            index_query,
            predicate,
        }))
    }
}

/// A collector keeping a segment's hits as a bitset.
struct Bits(FixedBitSet, Option<i32>);

impl ScoringCollector for Bits {
    fn collect(&mut self, doc_id: i32, _score: f32) {
        set_doc(&mut self.0, doc_id, &mut self.1);
    }
}

/// `CompositeVerifyQuery`: the index query's matches, each verified by the
/// predicate.
#[derive(Debug)]
pub struct CompositeVerifyQuery {
    pub index_query: Box<dyn DocumentQuery>,
    pub predicate: ShapeValuesPredicate,
}

impl DocumentQuery for CompositeVerifyQuery {
    fn rewrite(&self, leaves: &[OpenSegment<'_>]) -> Result<Option<Box<dyn DocumentQuery>>> {
        Ok(
            crate::document::rewrite(&*self.index_query, leaves)?.map(|index_query| {
                Box::new(CompositeVerifyQuery {
                    index_query,
                    predicate: self.predicate.clone(),
                }) as Box<dyn DocumentQuery>
            }),
        )
    }

    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let r = reader(leaf)?;
        let mut approx = Bits(FixedBitSet::new(idx(r.max_doc)), None);
        self.index_query.score_leaf(leaf, 1.0, &mut approx)?;
        let ctx = ValuesContext::for_reader(r);
        let mut m = self.predicate.matcher(&ctx, 0)?;
        for doc in docs_of(&approx.0) {
            if m.matches(doc)? {
                collector.collect(doc, boost);
            }
        }
        Ok(())
    }
}

/// `IntersectsRPTVerifyQuery`: an intersects walk that collects the
/// documents of cells within the query shape as exact matches and the rest
/// as candidates; only the candidates are verified.
#[derive(Debug)]
pub struct IntersectsRPTVerifyQuery {
    q: VisitingQuery,
    predicate: ShapeValuesPredicate,
}

/// `IntersectsDifferentiatingVisitor`.
struct DifferentiatingVisitor {
    approx: Option<FixedBitSet>,
    exact: Option<FixedBitSet>,
    approx_is_empty: bool,
    exact_is_empty: bool,
}

impl Visitor for DifferentiatingVisitor {
    fn start(&mut self, t: &mut Traverser<'_>) -> Result<()> {
        self.approx = Some(FixedBitSet::new(idx(t.max_doc)));
        self.exact = Some(FixedBitSet::new(idx(t.max_doc)));
        Ok(())
    }

    fn visit_prefix(
        &mut self,
        q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<bool> {
        if cell.shape_rel() == Some(SpatialRelation::Within) {
            self.exact_is_empty = false;
            t.collect_docs(self.exact.as_mut().expect("started"))?;
            return Ok(false);
        } else if cell.level() == q.base.detail_level {
            self.approx_is_empty = false;
            t.collect_docs(self.approx.as_mut().expect("started"))?;
            return Ok(false);
        }
        Ok(true)
    }

    fn visit_leaf(
        &mut self,
        _q: &VisitingQuery,
        t: &mut Traverser<'_>,
        cell: &dyn Cell,
    ) -> Result<()> {
        if cell.shape_rel() == Some(SpatialRelation::Within) {
            self.exact_is_empty = false;
            t.collect_docs(self.exact.as_mut().expect("started"))
        } else {
            self.approx_is_empty = false;
            t.collect_docs(self.approx.as_mut().expect("started"))
        }
    }
}

impl DocumentQuery for IntersectsRPTVerifyQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let r = reader(leaf)?;
        let mut t = Traverser::new(leaf, &self.q.base.field_name, &*self.q.base.grid)?;
        let mut v = DifferentiatingVisitor {
            approx: None,
            exact: None,
            approx_is_empty: true,
            exact_is_empty: true,
        };
        let had = visit(&self.q, &mut t, &mut v)?;
        t.check(&self.q.base.field_name)?;
        if !had {
            return Ok(());
        }
        // finish(): the exact set (if any) joins the candidates
        let exact = if v.exact_is_empty { None } else { v.exact };
        let approx = if v.approx_is_empty {
            exact.clone()
        } else {
            let mut a = v.approx.unwrap_or_else(|| FixedBitSet::new(idx(r.max_doc)));
            if let Some(e) = &exact {
                a.or(e);
            }
            Some(a)
        };
        let Some(approx) = approx else {
            return Ok(());
        };
        let ctx = ValuesContext::for_reader(r);
        let mut m = self.predicate.matcher(&ctx, 0)?;
        for doc in docs_of(&approx) {
            // `DefaultBulkScorer`: the live check first; an exact match
            // needs no verifying
            if leaf.live_docs.is_none_or(|bits| bits.get_doc(doc))
                && (exact.as_ref().is_some_and(|e| e.get_doc(doc)) || m.matches(doc)?)
            {
                collector.collect(doc, boost);
            }
        }
        Ok(())
    }
}
