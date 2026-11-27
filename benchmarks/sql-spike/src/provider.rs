//! DataFusion table provider over the index. Each group of segments (a
//! "slice") is one scan partition; doc values, stored ids and scores become
//! Arrow batches.

use std::any::Any;
use std::fmt;
use std::sync::Arc;

use datafusion::arrow::array::{
    ArrayRef, Float32Array, Int64Array, RecordBatch, StringArray, StringBuilder,
};
use datafusion::arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use datafusion::catalog::Session;
use datafusion::common::tree_node::TreeNodeRecursion;
use datafusion::common::{DataFusionError, Result, ScalarValue};
use datafusion::datasource::{TableProvider, TableType};
use datafusion::execution::{SendableRecordBatchStream, TaskContext};
use datafusion::logical_expr::{
    ColumnarValue, Expr, Operator, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    TableProviderFilterPushDown, Volatility,
};
use datafusion::physical_expr::EquivalenceProperties;
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
};
use lucene_codecs::doc_values::NumericReader;
use lucene_search::collector::{ScoringCollector, TopDocsCollector};
use lucene_search::global_ords::{keyword_column, Ords};
use lucene_search::knn_search::{knn_top_hits, KnnTarget};
use lucene_search::query::{BooleanQuery, Clause, TermQuery};

use crate::direct::KNN_CANDIDATES;
use crate::index::SpikeIndex;

pub const WINDOW: usize = 1024;
/// Nanoseconds spent in the native search of scans (bench diagnostics).
pub static SEARCH_NANOS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
/// Candidates a `knn()` predicate yields per partition when no `LIMIT k` was
/// pushed to the scan.
pub const KNN_DEFAULT_K: usize = 100;

pub fn table_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, true),
        Field::new("price", DataType::Int64, true),
        Field::new("cat", DataType::Utf8, true),
        Field::new("_score", DataType::Float32, true),
        Field::new(
            "vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                crate::VEC_DIM as i32,
            ),
            true,
        ),
        Field::new("body", DataType::Utf8, true),
    ]))
}

// --- predicate UDFs: never evaluated, only pushed down to the scan ---------

#[derive(Debug, PartialEq, Eq, Hash)]
struct Pushdown {
    name: &'static str,
    signature: Signature,
}

impl ScalarUDFImpl for Pushdown {
    fn name(&self) -> &str {
        self.name
    }
    fn signature(&self) -> &Signature {
        &self.signature
    }
    fn return_type(&self, _: &[DataType]) -> Result<DataType> {
        Ok(DataType::Boolean)
    }
    fn invoke_with_args(&self, _: ScalarFunctionArgs) -> Result<ColumnarValue> {
        Err(DataFusionError::Plan(format!(
            "{}() is only valid where it can be pushed down to the scan",
            self.name
        )))
    }
}

/// `match_text(body, 'w1 w2')`: a full-text predicate that scores.
pub fn match_text_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(Pushdown {
        name: "match_text",
        signature: Signature::any(2, Volatility::Immutable),
    })
}

/// `knn(vec, '0.1,0.2,...')`: a nearest-neighbour predicate that scores.
pub fn knn_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(Pushdown {
        name: "knn",
        signature: Signature::any(2, Volatility::Immutable),
    })
}

// --- pushed-down filters ----------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub enum Filter {
    Term { field: String, term: String },
    Match(Vec<String>),
    Knn(Vec<f32>),
}

fn lit_str(e: &Expr) -> Option<String> {
    match e {
        Expr::Literal(v, _) => v.try_as_str().flatten().map(str::to_string),
        _ => None,
    }
}

pub fn parse_filter(e: &Expr) -> Option<Filter> {
    match e {
        Expr::ScalarFunction(f) if f.name() == "match_text" && f.args.len() == 2 => {
            let text = lit_str(&f.args[1])?;
            Some(Filter::Match(
                text.split_whitespace().map(str::to_lowercase).collect(),
            ))
        }
        Expr::ScalarFunction(f) if f.name() == "knn" && f.args.len() == 2 => {
            let text = lit_str(&f.args[1])?;
            let v: Result<Vec<f32>, _> = text.split(',').map(|s| s.trim().parse()).collect();
            v.ok().map(Filter::Knn)
        }
        Expr::BinaryExpr(b) if b.op == Operator::Eq => {
            let (col, lit) = match (&*b.left, &*b.right) {
                (Expr::Column(c), l) => (c, l),
                (l, Expr::Column(c)) => (c, l),
                _ => return None,
            };
            if col.name != "id" && col.name != "cat" {
                return None;
            }
            Some(Filter::Term {
                field: col.name.clone(),
                term: lit_str(lit)?,
            })
        }
        _ => None,
    }
}

/// Segment groups, one scan partition each.
pub fn slices(ix: &SpikeIndex, partitions: usize) -> Vec<Vec<usize>> {
    let n = ix.segments.len();
    let partitions = partitions.clamp(1, n);
    let mut out = vec![Vec::new(); partitions];
    for s in 0..n {
        out[s % partitions].push(s);
    }
    out
}

pub struct LuceneTable {
    ix: &'static SpikeIndex,
    schema: SchemaRef,
    partitions: usize,
}

impl fmt::Debug for LuceneTable {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "LuceneTable")
    }
}

impl LuceneTable {
    pub fn new(ix: &'static SpikeIndex, partitions: usize) -> Self {
        Self {
            ix,
            schema: table_schema(),
            partitions,
        }
    }
}

#[async_trait::async_trait]
impl TableProvider for LuceneTable {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }
    fn table_type(&self) -> TableType {
        TableType::Base
    }
    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|e| {
                if parse_filter(e).is_some() {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
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
        let parsed: Vec<Filter> = filters.iter().filter_map(parse_filter).collect();
        let projection: Vec<usize> = match projection {
            Some(p) => p.to_vec(),
            None => (0..4).collect(),
        };
        if projection.contains(&4) || projection.contains(&5) {
            return Err(DataFusionError::NotImplemented(
                "the vec and body columns are search-only in the spike".into(),
            ));
        }
        Ok(Arc::new(SegScanExec::new(
            self.ix,
            slices(self.ix, self.partitions),
            projection,
            parsed,
            limit,
        )))
    }
}

// --- the scan ----------------------------------------------------------------

pub struct SegScanExec {
    ix: &'static SpikeIndex,
    slices: Vec<Vec<usize>>,
    projection: Vec<usize>,
    filters: Vec<Filter>,
    fetch: Option<usize>,
    schema: SchemaRef,
    props: Arc<PlanProperties>,
}

impl fmt::Debug for SegScanExec {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "SegScanExec")
    }
}

impl SegScanExec {
    fn new(
        ix: &'static SpikeIndex,
        slices: Vec<Vec<usize>>,
        projection: Vec<usize>,
        filters: Vec<Filter>,
        fetch: Option<usize>,
    ) -> Self {
        let full = table_schema();
        let schema = Arc::new(
            full.project(&projection)
                .expect("projection inside the table schema"),
        );
        let props = Arc::new(PlanProperties::new(
            EquivalenceProperties::new(schema.clone()),
            Partitioning::UnknownPartitioning(slices.len()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        ));
        Self {
            ix,
            slices,
            projection,
            filters,
            fetch,
            schema,
            props,
        }
    }
}

impl DisplayAs for SegScanExec {
    fn fmt_as(&self, _t: DisplayFormatType, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "SegScanExec: partitions={}, projection={:?}, filters={:?}, fetch={:?}",
            self.slices.len(),
            self.projection,
            self.filters,
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

/// Hits of one slice: global doc ids and scores.
fn matches(
    ix: &SpikeIndex,
    slice: &[usize],
    filters: &[Filter],
    fetch: Option<usize>,
) -> Result<Vec<(i32, f32)>> {
    let mut words: Vec<&str> = Vec::new();
    let mut terms: Vec<(&str, &str)> = Vec::new();
    let mut knn: Option<&[f32]> = None;
    for f in filters {
        match f {
            Filter::Match(w) => words.extend(w.iter().map(String::as_str)),
            Filter::Term { field, term } => terms.push((field, term)),
            Filter::Knn(v) => knn = Some(v),
        }
    }
    if let Some(v) = knn {
        if !words.is_empty() || !terms.is_empty() {
            return Err(DataFusionError::NotImplemented(
                "knn() with other predicates is not in the spike".into(),
            ));
        }
        let k = fetch.unwrap_or(KNN_DEFAULT_K);
        // `ix.segments[s]` carries its global doc base, so a one-segment
        // slice of the segment list returns global ids.
        let mut out = Vec::new();
        for &s in slice {
            let hits = knn_top_hits(
                &ix.segments[s..=s],
                "vec",
                &KnnTarget::Float(v.to_vec()),
                k,
                KNN_CANDIDATES.max(k),
                None,
            )
            .map_err(|e| DataFusionError::External(Box::new(e)))?;
            out.extend(hits.into_iter().map(|h| (h.doc_id, h.score)));
        }
        return Ok(out);
    }
    let q = BooleanQuery {
        must: terms
            .iter()
            .map(|(f, t)| Clause::Term(TermQuery::new(*f, t.as_bytes().to_vec())))
            .collect(),
        should: words
            .iter()
            .map(|w| Clause::Term(TermQuery::new("body", w.as_bytes().to_vec())))
            .collect(),
        minimum_should_match: usize::from(!words.is_empty()),
        ..BooleanQuery::default()
    };
    let ext = |e: lucene_search::Error| DataFusionError::External(Box::new(e));
    match (fetch, words.is_empty()) {
        // A score-ordered top-k pushed down by the optimiser rule.
        (Some(k), false) if slice.len() == ix.segments.len() => {
            // The whole index in one partition: the multi-segment search.
            let top = ix.searcher.search(&q, k).map_err(ext)?;
            Ok(top.score_docs.iter().map(|h| (h.doc, h.score)).collect())
        }
        (Some(k), false) => {
            let mut c = TopDocsCollector::with_total_hits_threshold(k, 1000);
            ix.searcher
                .search_slice_collector(&q, slice, &mut c)
                .map_err(ext)?;
            Ok(c.top_docs().iter().map(|h| (h.doc_id, h.score)).collect())
        }
        _ => {
            let mut c = CollectAll(Vec::new());
            ix.searcher
                .search_slice_collector(&q, slice, &mut c)
                .map_err(ext)?;
            let mut v = c.0;
            v.sort_unstable_by_key(|h| h.0);
            if let Some(k) = fetch {
                v.truncate(k);
            }
            Ok(v)
        }
    }
}

struct Cols<'a> {
    price: Vec<Option<NumericReader<'a>>>,
    cat: Vec<Option<Box<NumericReader<'a>>>>,
}

fn open_cols(ix: &'static SpikeIndex) -> Cols<'static> {
    let mut price = Vec::new();
    let mut cat = Vec::new();
    for (s, leaf) in ix.leaves().iter().enumerate() {
        price.push(
            leaf.doc_values_for_field(ix.fields[s].price)
                .and_then(|(m, d)| {
                    m.numeric_entry(ix.fields[s].price)
                        .map(|e| NumericReader::new(d, e))
                }),
        );
        cat.push(match keyword_column(leaf, "cat").unwrap() {
            Some((Ords::Single(r), _, _)) => Some(r),
            _ => None,
        });
    }
    Cols { price, cat }
}

/// Batches for a list of matched documents.
fn batch_for_docs(
    ix: &'static SpikeIndex,
    cols: &mut Cols<'static>,
    projection: &[usize],
    schema: &SchemaRef,
    docs: &[(i32, f32)],
) -> Result<RecordBatch> {
    let located: Vec<(usize, i32)> = docs.iter().map(|d| ix.locate(d.0)).collect();
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(projection.len());
    for &c in projection {
        let a: ArrayRef = match c {
            0 => {
                let mut b = StringBuilder::with_capacity(docs.len(), docs.len() * 12);
                for &(s, l) in &located {
                    b.append_value(ix.stored_id(s, l));
                }
                Arc::new(b.finish())
            }
            1 => Arc::new(Int64Array::from_iter(located.iter().map(|&(s, l)| {
                cols.price[s].as_mut().and_then(|r| r.value(l).unwrap())
            }))),
            2 => Arc::new(StringArray::from_iter(located.iter().map(|&(s, l)| {
                cols.cat[s]
                    .as_mut()
                    .and_then(|r| r.value(l).unwrap())
                    .map(|o| ix.cat_terms[s][o as usize].as_str())
            }))),
            3 => Arc::new(Float32Array::from_iter_values(docs.iter().map(|d| d.1))),
            _ => unreachable!("vec is rejected at plan time"),
        };
        arrays.push(a);
    }
    Ok(RecordBatch::try_new(schema.clone(), arrays)?)
}

/// A window of `n` consecutive documents of a segment.
fn batch_for_window(
    ix: &'static SpikeIndex,
    cols: &mut Cols<'static>,
    projection: &[usize],
    schema: &SchemaRef,
    seg: usize,
    start: i32,
    n: usize,
) -> Result<RecordBatch> {
    let mut vals = vec![0i64; n];
    let mut present = vec![0u64; n.div_ceil(64)];
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(projection.len());
    for &c in projection {
        let a: ArrayRef = match c {
            0 => {
                let mut b = StringBuilder::with_capacity(n, n * 12);
                for i in 0..n {
                    b.append_value(ix.stored_id(seg, start + i as i32));
                }
                Arc::new(b.finish())
            }
            1 => {
                let r = cols.price[seg].as_mut().expect("price column");
                r.fill_window(start, &mut vals, &mut present).unwrap();
                Arc::new(Int64Array::from(vals.clone()))
            }
            2 => {
                let r = cols.cat[seg].as_mut().expect("cat column");
                r.fill_window(start, &mut vals, &mut present).unwrap();
                let terms = &ix.cat_terms[seg];
                Arc::new(StringArray::from_iter_values(
                    vals.iter().map(|&o| terms[o as usize].as_str()),
                ))
            }
            3 => Arc::new(Float32Array::from(vec![0f32; n])),
            _ => unreachable!(),
        };
        arrays.push(a);
    }
    Ok(RecordBatch::try_new(schema.clone(), arrays)?)
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
        let ix = self.ix;
        let slice = self.slices[partition].clone();
        let projection = self.projection.clone();
        let filters = self.filters.clone();
        let fetch = self.fetch;
        let schema = self.schema.clone();
        let stream_schema = schema.clone();
        let mut state: Option<std::vec::IntoIter<Result<RecordBatch>>> = None;
        let mut started = false;
        let it = std::iter::from_fn(move || {
            if !started {
                started = true;
                let batches = (|| -> Result<Vec<Result<RecordBatch>>> {
                    let mut cols = open_cols(ix);
                    let mut out = Vec::new();
                    if filters.is_empty() {
                        let mut budget = fetch.unwrap_or(usize::MAX);
                        for &s in &slice {
                            let max = ix.leaves()[s].max_doc as usize;
                            let mut start = 0usize;
                            while start < max && budget > 0 {
                                let n = WINDOW.min(max - start).min(budget);
                                out.push(batch_for_window(
                                    ix,
                                    &mut cols,
                                    &projection,
                                    &schema,
                                    s,
                                    start as i32,
                                    n,
                                ));
                                start += n;
                                budget -= n;
                            }
                        }
                    } else {
                        let t = std::time::Instant::now();
                        let docs = matches(ix, &slice, &filters, fetch)?;
                        SEARCH_NANOS.fetch_add(
                            t.elapsed().as_nanos() as u64,
                            std::sync::atomic::Ordering::Relaxed,
                        );
                        for chunk in docs.chunks(WINDOW) {
                            out.push(batch_for_docs(ix, &mut cols, &projection, &schema, chunk));
                        }
                    }
                    Ok(out)
                })();
                state = Some(match batches {
                    Ok(b) => b.into_iter(),
                    Err(e) => vec![Err(e)].into_iter(),
                });
            }
            state.as_mut().and_then(Iterator::next)
        });
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            stream_schema,
            futures::stream::iter(it),
        )))
    }
}

pub fn _unused(_: &dyn Any, _: ScalarValue) {}
