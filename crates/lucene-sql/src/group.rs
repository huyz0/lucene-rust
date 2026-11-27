//! `GROUP BY` a keyword column answered by the aggregation kernels: a keyword
//! `terms` aggregation over every live document with one metric sub-aggregation
//! per aggregated column.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use lucene_aggs::bucket_aggs::{aggregate_tree, AggNode, AggResult, Globals, TermsOrder};
use lucene_aggs::metric_aggs::{
    MetricState, Source, ValueKind, NEED_COUNT, NEED_MIN_MAX, NEED_SUM,
};
use lucene_search::query::{BooleanQuery, Clause, MatchAllDocsQuery};

use crate::error::{Error, Result};
use crate::snapshot::Snapshot;
use crate::spec::ColumnKind;

/// Whole numbers a `f64` holds exactly.
const EXACT: f64 = 9_007_199_254_740_992.0;

/// An aggregate of a group, over a long column (indices into the table).
#[derive(Debug, Clone, PartialEq)]
pub enum AggFn {
    /// `count(*)`.
    CountStar,
    /// `count(col)`.
    Count(usize),
    /// `sum(col)`.
    Sum(usize),
    /// `min(col)`.
    Min(usize),
    /// `max(col)`.
    Max(usize),
    /// `avg(col)`.
    Avg(usize),
}

impl AggFn {
    fn column(&self) -> Option<usize> {
        match self {
            AggFn::CountStar => None,
            AggFn::Count(c) | AggFn::Sum(c) | AggFn::Min(c) | AggFn::Max(c) | AggFn::Avg(c) => {
                Some(*c)
            }
        }
    }
}

/// The groups of a `GROUP BY`: the keys in term order and one array per
/// aggregate, in the order asked.
pub struct Grouped {
    /// The group keys.
    pub keys: ArrayRef,
    /// One array per [`AggFn`].
    pub aggs: Vec<ArrayRef>,
}

impl Snapshot {
    /// Groups every live document by the keyword column `key` and computes
    /// `aggs` per group on the aggregation kernels.
    ///
    /// Returns `None` when the kernels cannot give exactly the answer a SQL
    /// engine would: a document without a `key` (SQL has a NULL group, the
    /// kernels none), or sums and extremes the kernels' `f64` cannot hold
    /// exactly. The caller then runs the statement the standard way.
    ///
    /// # Errors
    /// `key` is not a keyword column, an aggregated column is not a long
    /// column, or the index is unreadable.
    pub fn group_by(&self, key: usize, aggs: &[AggFn]) -> Result<Option<Grouped>> {
        let spec = self.spec();
        let key_col = spec
            .columns()
            .get(key)
            .filter(|c| c.kind == ColumnKind::Keyword)
            .ok_or_else(|| Error::unsupported("GROUP BY needs a keyword column"))?;
        let mut fields: Vec<(usize, u8)> = Vec::new();
        for a in aggs {
            let Some(c) = a.column() else { continue };
            if spec.columns().get(c).map(|c| &c.kind) != Some(&ColumnKind::Long) {
                return Err(Error::unsupported(
                    "an aggregate over a column needs a long column",
                ));
            }
            let needs = NEED_COUNT
                | NEED_MIN_MAX
                | if matches!(a, AggFn::Sum(_) | AggFn::Avg(_)) {
                    NEED_SUM
                } else {
                    0
                };
            match fields.iter_mut().find(|f| f.0 == c) {
                Some(f) => f.1 |= needs,
                None => fields.push((c, needs)),
            }
        }
        if !self.keyword_is_dense(key) {
            return Ok(None);
        }
        let subs: Vec<AggNode> = fields
            .iter()
            .map(|&(c, needs)| AggNode::Metric {
                field: spec.columns()[c].name.clone(),
                missing: None,
                kind: ValueKind::Long,
                source: Source::DocValues,
                needs,
                script: None,
            })
            .collect();
        let node = AggNode::Terms {
            field: key_col.name.clone(),
            missing: None,
            accepted: None,
            zero_buckets: false,
            shard_size: usize::MAX,
            order: TermsOrder::KeyAsc,
            subs,
        };
        let query = BooleanQuery {
            must: vec![Clause::MatchAllDocs(MatchAllDocsQuery::new(self.max_doc()))],
            ..BooleanQuery::default()
        };
        let ords: HashMap<String, Arc<_>> = HashMap::from([(
            key_col.name.clone(),
            self.reader().global_ords(&key_col.name)?,
        )]);
        let slices = vec![(0..self.segment_count()).collect::<Vec<_>>()];
        let mut out = self.with_searcher(|s| {
            Ok(aggregate_tree(
                s.segments,
                s.leaves,
                &query,
                &[node],
                &Globals::new(&ords),
                &slices,
                None,
            )?)
        })?;
        let Some(AggResult::Terms { buckets, subs }) = out.pop().and_then(|mut r| r.pop()) else {
            return Err(Error::unsupported("the kernels returned no terms result"));
        };
        let buckets = buckets.into_iter().next().map(|b| b.1).unwrap_or_default();
        let states: Vec<&Vec<MetricState>> = subs
            .iter()
            .map(|r| match r {
                AggResult::Metric(m) => Ok(m),
                _ => Err(Error::unsupported("the kernels returned no metric result")),
            })
            .collect::<Result<_>>()?;
        let keys: Vec<String> = buckets
            .iter()
            .map(|(k, _)| {
                String::from_utf8(k.clone())
                    .map_err(|_| Error::unsupported("a group key is not UTF-8"))
            })
            .collect::<Result<_>>()?;
        let mut arrays = Vec::with_capacity(aggs.len());
        for a in aggs {
            let state_of = |i: usize| -> Option<&MetricState> {
                let c = a.column()?;
                let f = fields.iter().position(|f| f.0 == c)?;
                states.get(f).and_then(|s| s.get(i))
            };
            let n = buckets.len();
            let array: ArrayRef = match a {
                AggFn::CountStar => Arc::new(Int64Array::from_iter_values(
                    buckets
                        .iter()
                        .map(|b| i64::try_from(b.1).unwrap_or(i64::MAX)),
                )),
                AggFn::Count(_) => Arc::new(Int64Array::from_iter_values((0..n).map(|i| {
                    state_of(i).map_or(0, |s| i64::try_from(s.count).unwrap_or(i64::MAX))
                }))),
                AggFn::Avg(_) => {
                    let mut v = Vec::with_capacity(n);
                    for i in 0..n {
                        let Some(s) = state_of(i) else {
                            return Ok(None);
                        };
                        if !sum_is_exact(s) {
                            return Ok(None);
                        }
                        v.push((s.count > 0).then(|| s.sum / s.count as f64));
                    }
                    Arc::new(Float64Array::from(v))
                }
                AggFn::Sum(_) | AggFn::Min(_) | AggFn::Max(_) => {
                    let mut v = Vec::with_capacity(n);
                    for i in 0..n {
                        let Some(s) = state_of(i) else {
                            return Ok(None);
                        };
                        let exact = match a {
                            AggFn::Sum(_) => sum_is_exact(s),
                            _ => s.count == 0 || extremes_are_exact(s),
                        };
                        if !exact {
                            return Ok(None);
                        }
                        let value = match a {
                            AggFn::Sum(_) => s.sum as i64,
                            AggFn::Min(_) => s.min as i64,
                            _ => s.max as i64,
                        };
                        v.push((s.count > 0).then_some(value));
                    }
                    Arc::new(Int64Array::from(v))
                }
            };
            arrays.push(array);
        }
        Ok(Some(Grouped {
            keys: Arc::new(StringArray::from_iter_values(keys)),
            aggs: arrays,
        }))
    }
}

fn extremes_are_exact(s: &MetricState) -> bool {
    s.min.abs() < EXACT && s.max.abs() < EXACT
}

/// Every partial sum of `count` values no larger than the extremes is exact
/// when `count * max(|min|, |max|)` stays below 2^53.
fn sum_is_exact(s: &MetricState) -> bool {
    s.count == 0 || (s.count as f64) * s.min.abs().max(s.max.abs()) < EXACT
}
