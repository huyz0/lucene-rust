//! A scored `DisjunctionMaxQuery` whose disjuncts are all terms, nested in a
//! larger query: `DisjunctionMaxScorer` over `TermScorer`s, as the scorer tree
//! builds it (`multi_match` `best_fields` and `cross_fields` put one per query
//! term under a `BooleanQuery`).
//!
//! It answers what [`super::disjunction::DisjunctionScorer`] with
//! `Combine::Max` answers -- the same documents, the same scores, the same
//! block bounds -- with two differences in how: the clauses are concrete
//! `TermScorer`s rather than boxed scorers, and
//! [`Scorer::next_docs_and_scores`], which `MaxScoreBulkScorer` drives an
//! essential clause through, reads each term's postings a decoded block at a
//! time into a window of per-document maxima (as [`crate::bulk_scorer`]'s
//! top-level `DisMaxBulk` does) instead of merging the clauses document by
//! document.
//!
//! A document's score folds its clauses in query order where the heap folds
//! them in heap order: the maximum is the same either way, and the others'
//! `f64` sum of a few `f32`s is exact unless their magnitudes differ by more
//! than 2^28, which BM25 scores do not.

use lucene_util::fixed_bit_set::FixedBitSet;

use super::leaf::TermScorer;
use super::{Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::{sum_relative_error_bound, DocScores};
use crate::Result;

/// The documents a batch covers at most: `MaxScoreBulkScorer.INNER_WINDOW_SIZE`.
const WINDOW: i32 = 1 << 12;

pub(crate) struct TermDisMaxScorer<'a> {
    subs: Vec<TermScorer<'a>>,
    tie: f32,
    /// The smallest of the clauses' documents.
    doc: i32,
    cost: i64,
    /// A batch's window: which documents matched, each one's best clause
    /// score and the `f64` sum of its others.
    matches: FixedBitSet,
    max: Vec<f32>,
    others: Vec<f64>,
    buf: DocScores,
}

impl<'a> TermDisMaxScorer<'a> {
    /// `subs` at least two, unpositioned.
    pub(crate) fn new(subs: Vec<TermScorer<'a>>, tie: f32) -> Self {
        debug_assert!(subs.len() > 1);
        let cost = subs.iter().fold(0i64, |c, s| c.saturating_add(s.cost()));
        let doc = subs
            .iter()
            .map(Scorer::doc_id)
            .min()
            .unwrap_or(NO_MORE_DOCS);
        Self {
            subs,
            tie,
            doc,
            cost,
            matches: FixedBitSet::new(WINDOW as usize),
            max: vec![0.0; WINDOW as usize],
            others: vec![0.0; WINDOW as usize],
            buf: DocScores::default(),
        }
    }

    fn min_doc(&self) -> i32 {
        self.subs
            .iter()
            .map(Scorer::doc_id)
            .min()
            .unwrap_or(NO_MORE_DOCS)
    }
}

impl Scorer for TermDisMaxScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        let cur = self.doc;
        for s in &mut self.subs {
            if s.doc_id() == cur {
                s.next_doc()?;
            }
        }
        self.doc = self.min_doc();
        Ok(self.doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        for s in &mut self.subs {
            if s.doc_id() < target {
                s.advance(target)?;
            }
        }
        self.doc = self.min_doc();
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.cost
    }

    /// `DisjunctionMaxScorer.score`.
    fn score(&mut self) -> Result<f32> {
        let mut score_max = 0.0f32;
        let mut other_sum = 0.0f64;
        let doc = self.doc;
        for s in &mut self.subs {
            if s.doc_id() != doc {
                continue;
            }
            let sub = s.score()?;
            if sub >= score_max {
                other_sum += f64::from(score_max);
                score_max = sub;
            } else {
                other_sum += f64::from(sub);
            }
        }
        Ok((f64::from(score_max) + other_sum * f64::from(self.tie)) as f32)
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        let mut min = NO_MORE_DOCS;
        for s in &mut self.subs {
            if s.doc_id() <= target {
                min = min.min(s.advance_shallow(target)?);
            }
        }
        Ok(min)
    }

    /// `DisjunctionMaxScorer.getMaxScore`.
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut score_max = 0.0f32;
        let mut other_sum = 0.0f64;
        for s in &mut self.subs {
            if s.doc_id() <= up_to {
                let sub = s.max_score(up_to)?;
                if sub >= score_max {
                    other_sum += f64::from(score_max);
                    score_max = sub;
                } else {
                    other_sum += f64::from(sub);
                }
            }
        }
        if self.tie == 0.0 {
            return Ok(score_max);
        }
        other_sum *= 1.0 + 2.0 * sum_relative_error_bound(self.subs.len() - 1);
        Ok((f64::from(score_max) + other_sum * f64::from(self.tie)) as f32)
    }

    fn doc_id_run_end(&self) -> i32 {
        let mut end = self.doc.saturating_add(1);
        for s in &self.subs {
            if s.doc_id() == self.doc {
                end = end.max(s.doc_id_run_end());
            }
        }
        end
    }

    /// A dismax with no tie-breaker is bounded by its best clause alone.
    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if self.tie == 0.0 {
            for s in &mut self.subs {
                s.set_min_competitive_score(min)?;
            }
        }
        Ok(())
    }

    /// The matches in `[doc, min(up_to, doc + WINDOW))`: every clause's
    /// postings there a block at a time, folded per document.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&FixedBitSet>,
        out: &mut DocScores,
    ) -> Result<()> {
        out.docs.clear();
        out.scores.clear();
        let start = self.doc;
        if start >= up_to {
            return Ok(());
        }
        let end = up_to.min(start.saturating_add(WINDOW));
        for s in &mut self.subs {
            if s.doc_id() >= end {
                continue;
            }
            let leg = s.leg_mut();
            loop {
                leg.next_docs_and_scores(end, live_docs, &mut self.buf)?;
                if self.buf.docs.is_empty() {
                    break;
                }
                for (&doc, &sub) in self.buf.docs.iter().zip(&self.buf.scores) {
                    // ARITH: `start <= doc < end <= start + WINDOW`.
                    #[allow(clippy::arithmetic_side_effects)]
                    let i = (doc - start) as usize;
                    // FBS: `matches` holds `WINDOW` bits and `i < WINDOW`.
                    self.matches.set(i);
                    let score_max = self.max[i];
                    if sub >= score_max {
                        self.others[i] += f64::from(score_max);
                        self.max[i] = sub;
                    } else {
                        self.others[i] += f64::from(sub);
                    }
                }
            }
        }
        let tie = f64::from(self.tie);
        let (max, others) = (&mut self.max, &mut self.others);
        self.matches.for_each_set_bit(|i| {
            // ARITH: `i < WINDOW`, and `start + i < end`.
            #[allow(clippy::arithmetic_side_effects, clippy::cast_possible_wrap)]
            out.docs.push(start + i as i32);
            out.scores
                .push((f64::from(max[i]) + others[i] * tie) as f32);
            max[i] = 0.0;
            others[i] = 0.0;
        });
        self.matches.clear_all();
        self.doc = self.min_doc();
        Ok(())
    }
}
