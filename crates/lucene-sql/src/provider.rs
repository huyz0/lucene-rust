//! The DataFusion table over a [`Snapshot`]. A scan without predicates reads
//! the doc values of the segments, one partition per group of segments; a scan
//! with pushed-down predicates runs one native search over the whole index in
//! a single partition (so the cross-segment pruning bound and the nearest
//! neighbours are exactly the direct API's), and `ORDER BY _score DESC LIMIT k`
//! reaches it as the scan's `fetch` ([`crate::rule::ScoreTopKRule`]).

use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{BooleanArray, RecordBatch};
use datafusion::arrow::datatypes::SchemaRef;
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use lucene_search::collector::ScoringCollector;
use lucene_search::query::BooleanQuery;

use crate::columns::{Columns, WINDOW};
use crate::error::{external, Error};
use crate::native::{all_of, any_of, check_dimension, knn_in, term_clause, Hit};
use crate::pred::{self, Pred};
use crate::snapshot::{Searcher, Snapshot};

/// Neighbours a `knn()` predicate yields when no `LIMIT` reached the scan.
pub const KNN_DEFAULT_K: usize = 100;

/// Most scan partitions of an unfiltered scan; segments beyond it share one.
pub const MAX_PARTITIONS: usize = 16;

/// The table of a snapshot.
pub struct LuceneTable {
    snapshot: Arc<Snapshot>,
}

impl fmt::Debug for LuceneTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LuceneTable({})", self.snapshot.spec().name())
    }
}

impl LuceneTable {
    /// A table over `snapshot`.
    pub fn new(snapshot: Arc<Snapshot>) -> Self {
        Self { snapshot }
    }

    /// The snapshot behind it.
    pub fn snapshot(&self) -> &Arc<Snapshot> {
        &self.snapshot
    }
}

fn plan_err(msg: impl Into<String>) -> DataFusionError {
    DataFusionError::NotImplemented(msg.into())
}

/// Rejects predicate sets the scan cannot answer as one search.
fn check_preds(preds: &[Pred]) -> Result<()> {
    let scoring = preds.iter().filter(|p| p.is_scoring()).count();
    if scoring > 1 {
        return Err(plan_err(
            "a query can have one match_text or knn predicate (two would need a choice \
             between AND and OR the scan cannot make)",
        ));
    }
    Ok(())
}

#[async_trait::async_trait]
impl TableProvider for LuceneTable {
    fn schema(&self) -> SchemaRef {
        self.snapshot.spec().arrow_schema()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        let spec = self.snapshot.spec();
        let parsed: Vec<Option<Pred>> = filters.iter().map(|e| pred::parse(spec, e)).collect();
        check_preds(&parsed.iter().flatten().cloned().collect::<Vec<_>>())?;
        Ok(parsed
            .iter()
            .map(|p| match p {
                Some(_) => TableProviderFilterPushDown::Exact,
                None => TableProviderFilterPushDown::Unsupported,
            })
            .collect())
    }
    async fn scan(
        &self,
        _state: &dyn Session,
        projection: Option<&[usize]>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let spec = self.snapshot.spec();
        let preds: Vec<Pred> = filters
            .iter()
            .filter_map(|e| pred::parse(spec, e))
            .collect();
        check_preds(&preds)?;
        let projection: Vec<usize> = match projection {
            Some(p) => p.to_vec(),
            None => (0..=spec.columns().len()).collect(),
        };
        for &c in &projection {
            if let Some(col) = spec.columns().get(c) {
                if !col.kind.is_readable() {
                    return Err(plan_err(format!(
                        "column `{}` is a search column: use it in match_text or knn; \
                         it has no values to read (SELECT * needs its columns listed)",
                        col.name
                    )));
                }
            }
        }
        Ok(Arc::new(SegScanExec::new(
            self.snapshot.clone(),
            projection,
            preds,
            limit,
        )?))
    }
}

/// The scan: see the module documentation.
pub struct SegScanExec {
    snapshot: Arc<Snapshot>,
    projection: Vec<usize>,
    preds: Vec<Pred>,
    fetch: Option<usize>,
    /// Segment groups of an unfiltered scan, one per partition.
    groups: Vec<Vec<usize>>,
    schema: SchemaRef,
    props: Arc<PlanProperties>,
}

impl fmt::Debug for SegScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SegScanExec")
    }
}

impl SegScanExec {
    pub(crate) fn new(
        snapshot: Arc<Snapshot>,
        projection: Vec<usize>,
        preds: Vec<Pred>,
        fetch: Option<usize>,
    ) -> Result<Self> {
        let schema = Arc::new(snapshot.spec().arrow_schema().project(&projection)?);
        let groups = if preds.is_empty() {
            let n = snapshot.segment_count().clamp(1, MAX_PARTITIONS);
            let mut groups = vec![Vec::new(); n];
            for s in 0..snapshot.segment_count() {
                groups[s % n].push(s);
            }
            groups
        } else {
            vec![Vec::new()]
        };
        let props = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(groups.len()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Ok(Self {
            snapshot,
            projection,
            preds,
            fetch,
            groups,
            schema,
            props,
        })
    }

    /// This scan's predicates, in the order the plan holds them.
    pub fn preds(&self) -> &[Pred] {
        &self.preds
    }

    /// The same scan with every predicate literal that `lookup` knows replaced
    /// by what it returns; also how many were replaced.
    pub fn bind(&self, lookup: &dyn Fn(&str) -> Option<String>) -> (SegScanExec, usize) {
        let mut replaced = 0;
        let preds = self
            .preds
            .iter()
            .map(|p| match lookup(p.literal()) {
                Some(v) => {
                    replaced += 1;
                    p.with_literal(&v)
                }
                None => p.clone(),
            })
            .collect();
        let scan = SegScanExec {
            snapshot: self.snapshot.clone(),
            projection: self.projection.clone(),
            preds,
            fetch: self.fetch,
            groups: self.groups.clone(),
            schema: self.schema.clone(),
            props: self.props.clone(),
        };
        (scan, replaced)
    }
}

impl DisplayAs for SegScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SegScanExec: partitions={}, projection={:?}, preds={:?}, fetch={:?}",
            self.groups.len(),
            self.projection,
            self.preds,
            self.fetch
        )
    }
}

struct CollectAll(Vec<(i32, f32)>);

impl ScoringCollector for CollectAll {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.0.push((doc_id, score));
    }
}

/// The documents the predicates select, in the order the scan returns them.
pub(crate) fn search_hits(
    s: &Searcher<'_, '_>,
    snapshot: &Snapshot,
    preds: &[Pred],
    fetch: Option<usize>,
) -> crate::Result<Vec<Hit>> {
    let spec = snapshot.spec();
    let name = |c: usize| spec.columns()[c].name.clone();
    let filters: Vec<(String, String)> = preds
        .iter()
        .filter_map(|p| match p {
            Pred::Term { column, value } => Some((name(*column), value.clone())),
            _ => None,
        })
        .collect();
    let scoring = preds.iter().find(|p| p.is_scoring());
    match scoring {
        Some(Pred::Knn { column, vector }) => {
            let v = pred::parse_vector(vector).ok_or_else(|| {
                Error::unsupported(format!("`{vector}` is not a list of numbers"))
            })?;
            check_dimension(snapshot, *column, &v)?;
            let filter = (!filters.is_empty()).then(|| all_of(&filters));
            knn_in(
                s,
                &name(*column),
                &v,
                fetch.unwrap_or(KNN_DEFAULT_K),
                filter.as_ref(),
            )
        }
        Some(Pred::Match { column, text }) => {
            let terms = spec.text_analyzer(&name(*column)).terms(text);
            if terms.is_empty() || fetch == Some(0) {
                return Ok(Vec::new());
            }
            let mut q = any_of(&name(*column), &terms);
            q.filter = filters.iter().map(|(f, t)| term_clause(f, t)).collect();
            match fetch {
                Some(k) => {
                    let top = s.searcher.search(&q, k)?;
                    Ok(top
                        .score_docs
                        .iter()
                        .map(|h| (h.doc, Some(h.score)))
                        .collect())
                }
                None => collect_all(s, &q, true),
            }
        }
        _ => {
            let q = BooleanQuery {
                filter: filters.iter().map(|(f, t)| term_clause(f, t)).collect(),
                ..BooleanQuery::default()
            };
            let mut hits = collect_all(s, &q, false)?;
            if let Some(k) = fetch {
                hits.truncate(k);
            }
            Ok(hits)
        }
    }
}

fn collect_all(s: &Searcher<'_, '_>, q: &BooleanQuery, scored: bool) -> crate::Result<Vec<Hit>> {
    let mut c = CollectAll(Vec::new());
    s.searcher.search_collector(q, &mut c)?;
    Ok(c.0
        .into_iter()
        .map(|(d, sc)| (d, scored.then_some(sc)))
        .collect())
}

impl ExecutionPlan for SegScanExec {
    fn name(&self) -> &str {
        "SegScanExec"
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
        partition: usize,
        _ctx: Arc<TaskContext>,
    ) -> Result<SendableRecordBatchStream> {
        let batches = self.run(partition).map_err(|e| match e {
            Error::DataFusion(d) => d,
            other => external(other),
        })?;
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            self.schema.clone(),
            futures::stream::iter(batches.into_iter().map(Ok)),
        )))
    }
}

impl SegScanExec {
    fn run(&self, partition: usize) -> crate::Result<Vec<RecordBatch>> {
        let snapshot = &*self.snapshot;
        let mut cols = Columns::new(snapshot);
        let mut out = Vec::new();
        if !self.preds.is_empty() {
            let hits =
                snapshot.with_searcher(|s| search_hits(s, snapshot, &self.preds, self.fetch))?;
            for chunk in hits.chunks(WINDOW) {
                let arrays = cols.for_docs(&self.projection, chunk)?;
                out.push(self.batch(arrays, chunk.len())?);
            }
            return Ok(out);
        }
        let mut budget = self.fetch.unwrap_or(usize::MAX);
        for &seg in &self.groups[partition] {
            let leaf = snapshot.leaf(seg);
            let max = usize::try_from(leaf.max_doc).unwrap_or(0);
            let mut start = 0usize;
            while start < max && budget > 0 {
                let n = WINDOW.min(max - start);
                let live = leaf
                    .live_docs()
                    .map(|bits| BooleanArray::from_iter((0..n).map(|i| Some(bits.get(start + i)))));
                let kept = live
                    .as_ref()
                    .map_or(n, BooleanArray::true_count)
                    .min(budget);
                if kept > 0 {
                    let arrays = cols.for_window(
                        &self.projection,
                        seg,
                        i32::try_from(start).unwrap_or(i32::MAX),
                        n,
                        live.as_ref(),
                    )?;
                    let batch =
                        self.batch(arrays, live.as_ref().map_or(n, BooleanArray::true_count))?;
                    out.push(batch.slice(0, kept));
                    budget -= kept;
                }
                start += n;
            }
        }
        Ok(out)
    }

    fn batch(
        &self,
        arrays: Vec<datafusion::arrow::array::ArrayRef>,
        rows: usize,
    ) -> crate::Result<RecordBatch> {
        let options =
            datafusion::arrow::record_batch::RecordBatchOptions::new().with_row_count(Some(rows));
        RecordBatch::try_new_with_options(self.schema.clone(), arrays, &options)
            .map_err(Error::from)
    }
}
