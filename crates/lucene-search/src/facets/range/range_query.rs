//! `LongRange.getQuery` / `DoubleRange.getQuery`'s query: the documents
//! whose value (or one of whose values) is in the range, each scored the
//! boost (`ConstantScoreWeight`).
//!
//! Java's `ValueSourceQuery`/`MultiValueSourceQuery` lead with the
//! fast-match query's scorer and check each of its documents' values (a
//! `TwoPhaseIterator`); here the fast-match query is a filter beside the
//! range check, inside a constant-score query -- the same documents, scored
//! the same. The cost differs: a [`DocumentQuery`] pushes every match of a
//! leaf and has no two-phase iterator to be led by, so the range check reads
//! the values source for every document of the leaf (`O(maxDoc)`), where
//! Java reads it only for the fast-match query's candidates. Not
//! benchmarked; the `value_facets` pair measures the counting paths.

use std::fmt;
use std::sync::Arc;

use crate::collector::ScoringCollector;
use crate::document::{collect_live, reader, DocumentQuery};
use crate::extended_query::DocumentClause;
use crate::facets::multi_double_values_source::MultiDoubleValuesSource;
use crate::facets::multi_long_values_source::MultiLongValuesSource;
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause, ConstantScoreQuery};
use crate::values_source::{DoubleValuesSource, LongValuesSource, ValuesContext};
use crate::Result;

/// The values a [`RangeQuery`] checks, and the range they must fall in.
#[derive(Clone)]
pub(crate) enum RangeValues {
    Long {
        min: i64,
        max: i64,
        source: Arc<dyn LongValuesSource>,
    },
    MultiLong {
        min: i64,
        max: i64,
        source: Arc<dyn MultiLongValuesSource>,
    },
    Double {
        min: f64,
        max: f64,
        source: Arc<dyn DoubleValuesSource>,
    },
    MultiDouble {
        min: f64,
        max: f64,
        source: Arc<dyn MultiDoubleValuesSource>,
    },
}

impl RangeValues {
    fn describe(&self) -> String {
        match self {
            RangeValues::Long { source, .. } => source.describe(),
            RangeValues::MultiLong { source, .. } => source.describe(),
            RangeValues::Double { source, .. } => source.describe(),
            RangeValues::MultiDouble { source, .. } => source.describe(),
        }
    }
}

/// The range check as a document query (`ValueSourceQuery` /
/// `MultiValueSourceQuery` without the fast-match query).
#[derive(Clone)]
struct ValueSourceQuery {
    values: RangeValues,
}

impl fmt::Debug for ValueSourceQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ValueSourceQuery({})", self.values.describe())
    }
}

impl DocumentQuery for ValueSourceQuery {
    fn score_leaf(
        &self,
        leaf: &OpenSegment<'_>,
        boost: f32,
        collector: &mut dyn ScoringCollector,
    ) -> Result<()> {
        let r = reader(leaf)?;
        let ctx = ValuesContext::for_reader(r);
        let max_doc = r.max_doc;
        match &self.values {
            RangeValues::Long { min, max, source } => {
                let mut v = source.get_values(&ctx, 0, None)?;
                for doc in 0..max_doc {
                    if v.advance_exact(doc)? {
                        let x = v.long_value()?;
                        if x >= *min && x <= *max {
                            collect_live(leaf, doc, boost, collector);
                        }
                    }
                }
            }
            RangeValues::Double { min, max, source } => {
                let mut v = source.get_values(&ctx, 0, None)?;
                for doc in 0..max_doc {
                    if v.advance_exact(doc)? {
                        let x = v.double_value()?;
                        if x >= *min && x <= *max {
                            collect_live(leaf, doc, boost, collector);
                        }
                    }
                }
            }
            RangeValues::MultiLong { min, max, source } => {
                let mut v = source.get_values(&ctx, 0)?;
                for doc in 0..max_doc {
                    if !v.advance_exact(doc)? {
                        continue;
                    }
                    let mut i = 0i64;
                    while i < v.get_value_count() {
                        let x = v.next_value()?;
                        if x >= *min && x <= *max {
                            collect_live(leaf, doc, boost, collector);
                            break;
                        }
                        i = i.saturating_add(1);
                    }
                }
            }
            RangeValues::MultiDouble { min, max, source } => {
                let mut v = source.get_values(&ctx, 0)?;
                for doc in 0..max_doc {
                    if !v.advance_exact(doc)? {
                        continue;
                    }
                    let mut i = 0i64;
                    while i < v.get_value_count() {
                        let x = v.next_value()?;
                        if x >= *min && x <= *max {
                            collect_live(leaf, doc, boost, collector);
                            break;
                        }
                        i = i.saturating_add(1);
                    }
                }
            }
        }
        Ok(())
    }
}

/// `Range.getQuery`'s query: [`RangeQuery::clause`] is it as a clause,
/// its [`fmt::Display`] Java's `toString` (`Filter(<range>)`).
#[derive(Clone)]
pub struct RangeQuery {
    range: String,
    fast_match_query: Option<BooleanQuery>,
    values: RangeValues,
}

impl RangeQuery {
    pub(crate) fn new(
        range: String,
        fast_match_query: Option<BooleanQuery>,
        values: RangeValues,
    ) -> Self {
        RangeQuery {
            range,
            fast_match_query,
            values,
        }
    }

    /// The query as a clause of the scorer tree.
    pub fn clause(&self) -> Clause {
        let key = format!("Filter({}) over {}", self.range, self.values.describe());
        let check: Clause = DocumentClause::new(
            key,
            Arc::new(ValueSourceQuery {
                values: self.values.clone(),
            }),
        )
        .into();
        match &self.fast_match_query {
            None => check,
            Some(fast) => ConstantScoreQuery::new(
                BooleanQuery {
                    filter: vec![Clause::from(fast.clone()), check],
                    ..Default::default()
                },
                1.0,
            )
            .into(),
        }
    }
}

impl From<RangeQuery> for Clause {
    fn from(q: RangeQuery) -> Self {
        q.clause()
    }
}

impl fmt::Display for RangeQuery {
    /// `toString(field)`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Filter({})", self.range)
    }
}

impl fmt::Debug for RangeQuery {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Display::fmt(self, f)
    }
}

#[cfg(test)]
mod tests;
