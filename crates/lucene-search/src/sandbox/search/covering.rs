//! `CoveringQuery` and `CoveringScorer`: documents matching at least a
//! per-document number of the sub-queries, scored by the sum of the matching
//! ones' scores.
//!
//! The minimum comes from a `LongValuesSource`: at least `1`, and a document
//! without a value never matches (`Long.MAX_VALUE`). A constant minimum
//! rewrites, as Java's does, to a `BooleanQuery` of `SHOULD` clauses with
//! that `minimumNumberShouldMatch` -- or to no documents when it exceeds the
//! number of queries. The scorer's score sums the matching sub-scores in
//! `double` and rounds once to `float`.
//!
//! Differences: the scorer confirms each candidate as it goes (Java's
//! approximation stops at documents the two-phase check then rejects) -- the
//! same documents and scores; the minimum's source is read as it is, not
//! rewritten against the searcher first (no shipped `LongValuesSource`
//! needs it); `toString`, `explain`, `matches`, `visit`, `equals`/`hashCode`
//! beyond the queries and the source's description are not ported.

use std::sync::Arc;

use crate::exec::NO_MORE_DOCS;
use crate::exec::{build as exec_build, exact_advance, BoxScorer, LeafContext, Mode, Scorer};
use crate::extended_query::MAX_CLAUSE_COUNT;
use crate::query::{BooleanQuery, Clause, MatchNoDocsQuery};
use crate::values_source::{BoxLongValues, LongValuesSource, ValuesContext};
use crate::{Error, Result};

/// `CoveringQuery(queries, minimumNumberMatch)`.
#[derive(Clone)]
pub struct CoveringQuery {
    /// A multiset: a query given twice counts twice.
    pub queries: Vec<Clause>,
    pub minimum_number_match: Arc<dyn LongValuesSource>,
}

impl std::fmt::Debug for CoveringQuery {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "CoveringQuery(queries={:?}, minimumNumberMatch={})",
            self.queries,
            self.minimum_number_match.describe()
        )
    }
}

impl PartialEq for CoveringQuery {
    fn eq(&self, o: &Self) -> bool {
        let mut a: Vec<String> = self.queries.iter().map(|q| format!("{q:?}")).collect();
        let mut b: Vec<String> = o.queries.iter().map(|q| format!("{q:?}")).collect();
        a.sort();
        b.sort();
        a == b && self.minimum_number_match.describe() == o.minimum_number_match.describe()
    }
}

impl CoveringQuery {
    /// # Errors
    /// More than `IndexSearcher.getMaxClauseCount()` queries
    /// (`TooManyClauses`); a minimum that reads scores.
    pub fn new(
        queries: impl IntoIterator<Item = Clause>,
        minimum_number_match: Arc<dyn LongValuesSource>,
    ) -> Result<Self> {
        let queries: Vec<Clause> = queries.into_iter().collect();
        if queries.len() > MAX_CLAUSE_COUNT {
            return Err(Error::InvalidQuery("too many clauses".into()));
        }
        if minimum_number_match.needs_scores() {
            return Err(Error::InvalidQuery(
                "The minimum number of matches may not depend on the score.".into(),
            ));
        }
        Ok(Self {
            queries,
            minimum_number_match,
        })
    }

    /// `rewrite`'s constant case: `None` for a per-document minimum.
    pub fn rewrite(&self) -> Option<Clause> {
        let min = self.minimum_number_match.constant_value()?;
        if usize::try_from(min).map_or(min > 0, |m| m > self.queries.len()) {
            return Some(Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason(
                "More clauses are required to match than the number of clauses",
            )));
        }
        let msm = usize::try_from(min.max(1)).unwrap_or(1);
        let mut b = BooleanQuery::new();
        b.should = self.queries.clone();
        b.minimum_should_match = msm;
        Some(Clause::Boolean(Box::new(b)))
    }
}

/// `CoveringWeight.scorerSupplier(...).get(...)`.
pub(crate) fn scorer<'a>(
    ctx: &LeafContext<'a>,
    q: &CoveringQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    if let Some(rewritten) = q.rewrite() {
        return exec_build::build(ctx, &rewritten, boost, mode, top_level);
    }
    // The sub-scorers are never told a minimum competitive score.
    let sub_mode = if mode.needs_scores() {
        Mode::Complete
    } else {
        Mode::NoScores
    };
    let mut subs = Vec::new();
    for c in &q.queries {
        if let Some(s) = exec_build::build(ctx, c, boost, sub_mode, false)? {
            subs.push(s);
        }
    }
    if subs.is_empty() {
        return Ok(None);
    }
    let values = q
        .minimum_number_match
        .get_values(&ValuesContext::for_leaf(*ctx), 0, None)?;
    let cost = subs
        .iter()
        .map(|s| s.cost())
        .fold(0i64, i64::saturating_add);
    Ok(Some(Box::new(CoveringScorer {
        subs,
        docs: Vec::new(),
        values,
        doc: -1,
        cost,
    })))
}

/// `CoveringScorer`, its two-phase check folded into the iteration.
struct CoveringScorer<'a> {
    subs: Vec<BoxScorer<'a>>,
    /// Each sub-scorer's current (confirmed) document.
    docs: Vec<i32>,
    values: BoxLongValues<'a>,
    doc: i32,
    cost: i64,
}

impl CoveringScorer<'_> {
    /// `setMinMatch()` for `doc`, a document of the segment (Java's `1`
    /// past its last is never asked for here: the iteration ends first).
    fn min_match(&mut self, doc: i32) -> Result<i64> {
        Ok(if self.values.advance_exact(doc)? {
            self.values.long_value()?.max(1)
        } else {
            i64::MAX
        })
    }
}

impl Scorer for CoveringScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }

    fn advance(&mut self, mut target: i32) -> Result<i32> {
        if self.docs.is_empty() {
            self.docs = vec![-1; self.subs.len()];
        }
        loop {
            let mut candidate = NO_MORE_DOCS;
            for (s, d) in self.subs.iter_mut().zip(self.docs.iter_mut()) {
                if *d < target {
                    *d = exact_advance(s.as_mut(), target)?;
                }
                candidate = candidate.min(*d);
            }
            if candidate == NO_MORE_DOCS {
                self.doc = NO_MORE_DOCS;
                return Ok(NO_MORE_DOCS);
            }
            let count = self.docs.iter().filter(|&&d| d == candidate).count();
            let count = i64::try_from(count).unwrap_or(i64::MAX);
            if count >= self.min_match(candidate)? {
                self.doc = candidate;
                return Ok(candidate);
            }
            target = candidate.saturating_add(1);
        }
    }

    fn cost(&self) -> i64 {
        self.cost
    }

    fn score(&mut self) -> Result<f32> {
        let mut score = 0f64;
        for (s, &d) in self.subs.iter_mut().zip(&self.docs) {
            if d == self.doc {
                score += f64::from(s.score()?);
            }
        }
        Ok(score as f32)
    }

    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::INFINITY)
    }

    fn max_score_unbounded(&self) -> bool {
        true
    }
}
