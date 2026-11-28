//! `GROUP BY` of a keyword column, answered by the aggregation kernels.
//!
//! DataFusion's own hash aggregate materialises Arrow batches and hashes every
//! row (about 2.8x the kernels on one core). [`GroupByKernelRule`] replaces an
//! `Aggregate` directly over a table scan with a scan of a [`GroupTable`] that
//! runs [`Snapshot::group_by`], under a projection that restores the
//! aggregate's exact output names.
//!
//! The rule fires only where the answer is the one SQL gives: one keyword
//! group column that every document has (SQL would otherwise print a NULL
//! group), `count(*)` and `count`/`sum`/`min`/`max`/`avg` of a long column, no
//! `DISTINCT`, `FILTER` or `ORDER BY` inside an aggregate, and a scan with only
//! equality and range predicates and no limit. Sums and extremes the kernels' `f64` cannot hold
//! exactly are caught when the statement runs ([`KernelInexact`]); the engine
//! then plans the statement again without this rule.

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::tree_node::{Transformed, TreeNodeRecursion};
use datafusion::common::{Column, DataFusionError, Result, ScalarValue};
use datafusion::datasource::{provider_as_source, source_as_provider, TableProvider, TableType};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::logical_plan::{LogicalPlan, Projection, TableScanBuilder};
use datafusion::logical_expr::{Aggregate, Expr};
use datafusion::optimizer::{ApplyOrder, OptimizerConfig, OptimizerRule};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};

use crate::error::{external, KernelInexact};
use crate::group::AggFn;
use crate::pred::{self, Pred};
use crate::provider::LuceneTable;
use crate::snapshot::Snapshot;
use crate::spec::ColumnKind;

const SCAN_NAME: &str = "__group_kernels";

/// The rule; see the module documentation.
#[derive(Debug, Default)]
pub struct GroupByKernelRule;

fn long_column(snapshot: &Snapshot, e: &Expr) -> Option<usize> {
    let Expr::Column(c) = e else { return None };
    let i = snapshot
        .spec()
        .columns()
        .iter()
        .position(|col| col.name == c.name)?;
    (snapshot.spec().columns()[i].kind == ColumnKind::Long).then_some(i)
}

fn aggregate_fn(snapshot: &Snapshot, e: &Expr) -> Option<AggFn> {
    let f = match e {
        Expr::AggregateFunction(f) => f,
        _ => return None,
    };
    let p = &f.params;
    let plain = !p.distinct && p.filter.is_none() && p.order_by.is_empty();
    let arg = (plain && p.null_treatment.is_none() && p.args.len() == 1).then(|| &p.args[0])?;
    let name = f.func.name();
    if name == "count" {
        return match arg {
            Expr::Literal(ScalarValue::Int64(Some(1)), _) => Some(AggFn::CountStar),
            e => long_column(snapshot, e).map(AggFn::Count),
        };
    }
    // The analyser averages integers as `avg(CAST(x AS Float64))`.
    let arg = match arg {
        Expr::Cast(c) if name == "avg" && c.field.data_type() == &DataType::Float64 => &*c.expr,
        a => a,
    };
    let c = long_column(snapshot, arg)?;
    match name {
        "sum" => Some(AggFn::Sum(c)),
        "min" => Some(AggFn::Min(c)),
        "max" => Some(AggFn::Max(c)),
        "avg" => Some(AggFn::Avg(c)),
        _ => None,
    }
}

/// The kernels' answer for `agg`, when the rule can give it.
fn rewrite_aggregate(agg: &Aggregate) -> Result<Option<LogicalPlan>> {
    let LogicalPlan::TableScan(scan) = &*agg.input else {
        return Ok(None);
    };
    if scan.fetch.is_some() {
        return Ok(None);
    }
    let provider = source_as_provider(&scan.source)?;
    let Some(table) = provider.as_ref().downcast_ref::<LuceneTable>() else {
        return Ok(None);
    };
    let snapshot = table.snapshot();
    let Some(preds) = scan
        .filters
        .iter()
        .map(|e| pred::parse(snapshot.spec(), e).filter(|p| !p.is_scoring()))
        .collect::<Option<Vec<Pred>>>()
    else {
        return Ok(None);
    };
    let [Expr::Column(key)] = agg.group_expr.as_slice() else {
        return Ok(None);
    };
    let Some(key_index) = snapshot
        .spec()
        .columns()
        .iter()
        .position(|c| c.name == key.name && c.kind == ColumnKind::Keyword)
    else {
        return Ok(None);
    };
    if !snapshot.keyword_is_dense(key_index) {
        return Ok(None);
    }
    let Some(aggs) = agg
        .aggr_expr
        .iter()
        .map(|e| aggregate_fn(snapshot, e))
        .collect::<Option<Vec<AggFn>>>()
    else {
        return Ok(None);
    };
    // The scan's columns are the aggregate's output columns, in order.
    let fields: Vec<Field> = agg
        .schema
        .fields()
        .iter()
        .enumerate()
        .map(|(i, f)| Field::new(format!("c{i}"), f.data_type().clone(), f.is_nullable()))
        .collect();
    let table = Arc::new(GroupTable {
        snapshot: snapshot.clone(),
        key: key_index,
        aggs,
        preds,
        schema: Arc::new(Schema::new(fields)),
    });
    let new_scan = LogicalPlan::TableScan(
        TableScanBuilder::new(SCAN_NAME, provider_as_source(table)).build()?,
    );
    // The aggregate's output names: the group column first, then aggregates.
    let exprs: Vec<Expr> = agg
        .schema
        .iter()
        .enumerate()
        .map(|(i, (qualifier, field))| {
            Expr::Column(Column::new(Some(SCAN_NAME), format!("c{i}")))
                .alias_qualified(qualifier.cloned(), field.name())
        })
        .collect();
    Ok(Some(LogicalPlan::Projection(Projection::try_new(
        exprs,
        Arc::new(new_scan),
    )?)))
}

impl OptimizerRule for GroupByKernelRule {
    fn name(&self) -> &str {
        "group_by_kernels"
    }
    fn apply_order(&self) -> Option<ApplyOrder> {
        Some(ApplyOrder::BottomUp)
    }
    fn rewrite(
        &self,
        plan: LogicalPlan,
        _config: &dyn OptimizerConfig,
    ) -> Result<Transformed<LogicalPlan>> {
        if let LogicalPlan::Aggregate(agg) = &plan {
            if let Some(new) = rewrite_aggregate(agg)? {
                return Ok(Transformed::yes(new));
            }
        }
        Ok(Transformed::no(plan))
    }
}

/// The groups of one `GROUP BY`: a one-partition table whose rows are the
/// kernels' answer, columns `c0..` in the aggregate's output order.
struct GroupTable {
    snapshot: Arc<Snapshot>,
    key: usize,
    aggs: Vec<AggFn>,
    preds: Vec<Pred>,
    schema: SchemaRef,
}

impl fmt::Debug for GroupTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GroupTable(key={}, aggs={:?})", self.key, self.aggs)
    }
}

#[async_trait::async_trait]
impl TableProvider for GroupTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    async fn scan(
        &self,
        _state: &dyn Session,
        _projection: Option<&[usize]>,
        _filters: &[Expr],
        _limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(GroupScanExec {
            snapshot: self.snapshot.clone(),
            key: self.key,
            aggs: self.aggs.clone(),
            preds: self.preds.clone(),
            schema: self.schema.clone(),
            props: Arc::new(PlanProperties::new(
                EquivalenceProperties::new(self.schema.clone()),
                Partitioning::UnknownPartitioning(1),
                EmissionType::Final,
                Boundedness::Bounded,
            )),
        }))
    }
}

struct GroupScanExec {
    snapshot: Arc<Snapshot>,
    key: usize,
    aggs: Vec<AggFn>,
    preds: Vec<Pred>,
    schema: SchemaRef,
    props: Arc<PlanProperties>,
}

impl fmt::Debug for GroupScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "GroupScanExec")
    }
}

impl DisplayAs for GroupScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "GroupScanExec: key={}, aggs={:?}, preds={:?}",
            self.key, self.aggs, self.preds
        )
    }
}

impl ExecutionPlan for GroupScanExec {
    fn name(&self) -> &str {
        "GroupScanExec"
    }
    fn properties(&self) -> &Arc<PlanProperties> {
        &self.props
    }
    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }
    fn apply_expressions(
        &self,
        _f: &mut dyn FnMut(
            &Arc<dyn datafusion::physical_expr::PhysicalExpr>,
        ) -> Result<TreeNodeRecursion>,
    ) -> Result<TreeNodeRecursion> {
        Ok(TreeNodeRecursion::Continue)
    }
    fn with_new_children(
        self: Arc<Self>,
        _: Vec<Arc<dyn ExecutionPlan>>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        Ok(self)
    }
    fn execute(
        &self,
        _partition: usize,
        ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let scope = crate::scope::of(&ctx);
        let grouped = lucene_util::memory::within(scope.as_ref(), || {
            self.snapshot.group_by(self.key, &self.aggs, &self.preds)
        })
        .map_err(external)?
        .ok_or_else(|| {
            external(KernelInexact(
                "a document has no group key, or a sum or extreme is beyond 2^53".into(),
            ))
        })?;
        let mut columns = vec![grouped.keys];
        columns.extend(grouped.aggs);
        let batch = datafusion::arrow::array::RecordBatch::try_new(self.schema.clone(), columns)?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            futures::stream::iter(vec![Ok(batch)]),
        )))
    }
}

/// Whether `e` is the kernels' "cannot answer exactly" marker.
pub fn is_inexact(e: &DataFusionError) -> bool {
    matches!(e.find_root(), DataFusionError::External(inner) if inner.is::<KernelInexact>())
}
