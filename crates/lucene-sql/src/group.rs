//! `GROUP BY` a keyword column answered by the aggregation kernels: a keyword
//! `terms` aggregation over every live document with one metric sub-aggregation
//! per aggregated column.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::arrow::array::{ArrayRef, Float64Array, Int64Array, StringArray};
use lucene_aggs::bucket_aggs::segment_cache::{AggSegmentCache, CacheRequest};
use lucene_aggs::bucket_aggs::{aggregate_tree_cached, AggNode, AggResult, Globals, TermsOrder};
use lucene_aggs::metric_aggs::{
    MetricState, Source, ValueKind, NEED_COUNT, NEED_MIN_MAX, NEED_SUM,
};

use crate::error::{Error, Result};
use crate::pred::Pred;
use crate::snapshot::Snapshot;
use crate::spec::ColumnKind;

/// Whole numbers a `f64` holds exactly.
const EXACT: f64 = 9_007_199_254_740_992.0;

/// A filter that keeps fewer than one document in this many is selective.
const SELECTIVE: i64 = 16;

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
    /// Groups every live document matching the equality and range `preds` by the keyword column `key` and computes
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
    pub fn group_by(&self, key: usize, aggs: &[AggFn], preds: &[Pred]) -> Result<Option<Grouped>> {
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
        let query = self.filter_query(preds);
        let ords: HashMap<String, Arc<_>> = HashMap::from([(
            key_col.name.clone(),
            self.reader().global_ords(&key_col.name)?,
        )]);
        let slices = vec![(0..self.segment_count()).collect::<Vec<_>>()];
        let cache = AggSegmentCache::global();
        let request = self
            .agg_identities()
            .filter(|_| cache.budget() > 0 && preds.is_empty())
            .map(|segments| CacheRequest { cache, segments });
        let mut out = self.with_searcher(|s| {
            Ok(aggregate_tree_cached(
                s.segments,
                s.leaves,
                &query,
                &[node],
                &Globals::new(&ords).with_bucket_limit(self.group_limit()),
                &slices,
                None,
                request.as_ref(),
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
        let counts: Vec<i64> = buckets
            .iter()
            .map(|b| i64::try_from(b.1).unwrap_or(i64::MAX))
            .collect();
        let Some(arrays) = agg_arrays(aggs, &counts, |a, i| {
            let c = a.column()?;
            let f = fields.iter().position(|f| f.0 == c)?;
            states.get(f).and_then(|s| s.get(i))
        }) else {
            return Ok(None);
        };
        Ok(Some(Grouped {
            keys: Arc::new(StringArray::from_iter_values(keys)),
            aggs: arrays,
        }))
    }

    /// The aggregates `aggs` over every live document matching `preds`, one
    /// row, on the aggregation kernels (`count(*)` from the index's counts).
    ///
    /// Returns `None` when the kernels cannot give exactly the answer a SQL
    /// engine would (see [`Self::group_by`]).
    ///
    /// # Errors
    /// An aggregated column is not a long column, or the index is unreadable.
    pub fn aggregate(&self, aggs: &[AggFn], preds: &[Pred]) -> Result<Option<Vec<ArrayRef>>> {
        let spec = self.spec();
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
        let matched = i64::try_from(self.count(preds)?).unwrap_or(i64::MAX);
        // The kernels visit the whole index; a filter that keeps a small part
        // of it is answered faster by reading the few documents, which is
        // what the standard path does.
        if !preds.is_empty() && matched.saturating_mul(SELECTIVE) < i64::from(self.max_doc()) {
            return Ok(None);
        }
        let mut states: Vec<Vec<MetricState>> = Vec::new();
        if !fields.is_empty() && matched > 0 {
            let nodes: Vec<AggNode> = fields
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
            let query = self.filter_query(preds);
            let ords = HashMap::new();
            let slices = vec![(0..self.segment_count()).collect::<Vec<_>>()];
            let cache = AggSegmentCache::global();
            let request = self
                .agg_identities()
                .filter(|_| cache.budget() > 0 && preds.is_empty())
                .map(|segments| CacheRequest { cache, segments });
            let mut out = self.with_searcher(|s| {
                Ok(aggregate_tree_cached(
                    s.segments,
                    s.leaves,
                    &query,
                    &nodes,
                    &Globals::new(&ords),
                    &slices,
                    None,
                    request.as_ref(),
                )?)
            })?;
            for r in out.pop().unwrap_or_default() {
                let AggResult::Metric(m) = r else {
                    return Err(Error::unsupported("the kernels returned no metric result"));
                };
                states.push(m);
            }
        }
        Ok(agg_arrays(aggs, &[matched], |a, _| {
            let c = a.column()?;
            let f = fields.iter().position(|f| f.0 == c)?;
            states.get(f).and_then(|s| s.first())
        }))
    }
}

/// One array per aggregate over `counts.len()` groups (`counts[i]` is group
/// `i`'s document count); `None` when a result is not exact in the kernels'
/// `f64`.
fn agg_arrays<'s>(
    aggs: &[AggFn],
    counts: &[i64],
    state_of: impl Fn(&AggFn, usize) -> Option<&'s MetricState>,
) -> Option<Vec<ArrayRef>> {
    let n = counts.len();
    let mut arrays = Vec::with_capacity(aggs.len());
    for a in aggs {
        let array: ArrayRef = match a {
            AggFn::CountStar => Arc::new(Int64Array::from(counts.to_vec())),
            AggFn::Count(_) => Arc::new(Int64Array::from_iter_values((0..n).map(|i| {
                state_of(a, i).map_or(0, |s| i64::try_from(s.count).unwrap_or(i64::MAX))
            }))),
            AggFn::Avg(_) => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let s = state_of(a, i)?;
                    if !sum_is_exact(s) {
                        return None;
                    }
                    v.push((s.count > 0).then(|| s.sum / s.count as f64));
                }
                Arc::new(Float64Array::from(v))
            }
            AggFn::Sum(_) | AggFn::Min(_) | AggFn::Max(_) => {
                let mut v = Vec::with_capacity(n);
                for i in 0..n {
                    let s = state_of(a, i)?;
                    let exact = match a {
                        AggFn::Sum(_) => sum_is_exact(s),
                        _ => s.count == 0 || extremes_are_exact(s),
                    };
                    if !exact {
                        return None;
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
    Some(arrays)
}

fn extremes_are_exact(s: &MetricState) -> bool {
    s.min.abs() < EXACT && s.max.abs() < EXACT
}

/// Every partial sum of `count` values no larger than the extremes is exact
/// when `count * max(|min|, |max|)` stays below 2^53.
fn sum_is_exact(s: &MetricState) -> bool {
    s.count == 0 || (s.count as f64) * s.min.abs().max(s.max.abs()) < EXACT
}
