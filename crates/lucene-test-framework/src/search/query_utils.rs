//! `QueryUtils`: sanity checks every query should pass on a searcher --
//! equality, scorers that agree with the collected hits however they are
//! advanced, bulk scorers that agree with scorers, explanations and
//! matches.
//!
//! The scorers are `lucene-search`'s, reached through its `test_support`
//! seam. `wrapUnderlyingReader` (the same checks through a `MultiReader`
//! padded with empty readers) has no counterpart yet, so
//! [`check_with_searcher`] runs the unwrapped checks only.

use lucene_search::collector::{ScoreMode, ScoringCollector};
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::{BooleanQuery, Clause};
use lucene_search::test_support::{self, LeafScorer, NO_MORE_DOCS};
use lucene_search::Result;
use lucene_util::java_random::JavaRandom;

use super::check_hits;

/// `QueryUtils.check(q)` / `checkHashEquals`: a query equals itself and a
/// copy of itself, and not a query of a different kind (Java's "whacky"
/// anonymous query). This port's queries are values without `hashCode`,
/// so equality is all there is to check.
pub fn check(query: &BooleanQuery) {
    check_equal(query, query);
    check_equal(query, &query.clone());
    let whacky = BooleanQuery {
        should: vec![Clause::Boolean(Box::new(query.clone()))],
        ..BooleanQuery::default()
    };
    check_unequal(query, &whacky);
}

/// `checkEqual`.
pub fn check_equal(q1: &BooleanQuery, q2: &BooleanQuery) {
    assert_eq!(q1, q2);
}

/// `checkUnequal`.
pub fn check_unequal(q1: &BooleanQuery, q2: &BooleanQuery) {
    assert!(q1 != q2, "{q1:?} equal to {q2:?}");
    assert!(q2 != q1, "{q2:?} equal to {q1:?}");
}

/// `checkExplanations(q, s)`: a deep [`check_hits::check_explanations`].
///
/// # Errors
/// Whatever searching or explaining reports.
pub fn check_explanations(query: &BooleanQuery, searcher: &IndexSearcher<'_, '_>) -> Result<()> {
    check_hits::check_explanations(query, searcher, true)
}

/// `QueryUtils.check(random, q, s)`: [`check`], then
/// [`check_first_skip_to`], [`check_skip_to`],
/// [`check_bulk_scorer_skip_to`], [`check_count`], [`check_explanations`]
/// and [`check_hits::check_matches`].
///
/// # Errors
/// Whatever searching reports.
pub fn check_with_searcher(
    random: &mut JavaRandom,
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
) -> Result<()> {
    check(query);
    check_first_skip_to(query, searcher)?;
    check_skip_to(query, searcher)?;
    check_bulk_scorer_skip_to(random, query, searcher)?;
    check_count(query, searcher)?;
    check_explanations(query, searcher)?;
    check_hits::check_matches(query, searcher)
}

/// The hits of `query` per segment, leaf-local, in collection order
/// (`ScoreMode.COMPLETE`).
fn hits_per_leaf(
    searcher: &IndexSearcher<'_, '_>,
    query: &BooleanQuery,
) -> Result<Vec<Vec<(i32, f32)>>> {
    let segments = searcher.segments();
    let mut per_leaf = vec![Vec::new(); segments.len()];
    for (doc, score) in check_hits::all_hits(searcher, query, ScoreMode::Complete)? {
        let leaf = searcher.segment_of(doc).expect("a hit is in a segment");
        per_leaf[leaf].push((doc - segments[leaf].doc_base, score));
    }
    Ok(per_leaf)
}

/// `assertNoPastSegmentEnd`: past `last_doc`, the leaf's scorer finds no
/// live document.
fn assert_no_past_segment_end(
    searcher: &IndexSearcher<'_, '_>,
    query: &BooleanQuery,
    leaf: usize,
    last_doc: i32,
) -> Result<()> {
    test_support::with_leaf(searcher, query, leaf, |ctx| {
        let Some(mut scorer) = ctx.scorer(ScoreMode::Complete)? else {
            return Ok(());
        };
        let live = ctx.live_docs();
        let mut more = false;
        let mut d = scorer.exact_advance(last_doc + 1)?;
        while d != NO_MORE_DOCS {
            if live.is_none_or(|l| l.get_doc(d)) {
                more = true;
                break;
            }
            d = scorer.exact_next()?;
        }
        assert!(
            !more,
            "query's last doc was {last_doc} but advance({}) got to {}",
            last_doc + 1,
            scorer.doc_id()
        );
        Ok(())
    })
}

/// `checkSkipTo`: the leaf scorer, moved by a pattern of `advance` and
/// `nextDoc` calls (`advance(docID() + 1)` for a skip), lands on each
/// collected document in turn with the collected score.
///
/// # Errors
/// Whatever searching reports.
pub fn check_skip_to(query: &BooleanQuery, searcher: &IndexSearcher<'_, '_>) -> Result<()> {
    const SKIP: u8 = 0;
    const NEXT: u8 = 1;
    const ORDERS: [&[u8]; 7] = [
        &[NEXT],
        &[SKIP],
        &[SKIP, NEXT],
        &[NEXT, SKIP],
        &[SKIP, SKIP, NEXT, NEXT],
        &[NEXT, NEXT, SKIP, SKIP],
        &[SKIP, SKIP, SKIP, NEXT, NEXT],
    ];
    const MAX_DIFF: f32 = 1e-5;
    let per_leaf = hits_per_leaf(searcher, query)?;
    for order in ORDERS {
        let mut opidx = 0usize;
        for (leaf, hits) in per_leaf.iter().enumerate() {
            let mut last_doc = -1;
            if !hits.is_empty() {
                test_support::with_leaf(searcher, query, leaf, |ctx| {
                    let mut scorer = ctx
                        .scorer(ScoreMode::Complete)?
                        .expect("a leaf with hits has a scorer");
                    for &(doc, score) in hits {
                        last_doc = doc;
                        let op = order[opidx % order.len()];
                        opidx += 1;
                        let more = if op == SKIP {
                            scorer.exact_advance(scorer.doc_id() + 1)? != NO_MORE_DOCS
                        } else {
                            scorer.exact_next()? != NO_MORE_DOCS
                        };
                        assert_doc_and_score(doc, score, more, &mut scorer, MAX_DIFF)?;
                    }
                    Ok(())
                })?;
            }
            assert_no_past_segment_end(searcher, query, leaf, last_doc)?;
        }
    }
    Ok(())
}

/// `assertDocAndScore`.
fn assert_doc_and_score(
    doc: i32,
    score: f32,
    more: bool,
    scorer: &mut LeafScorer<'_>,
    max_diff: f32,
) -> Result<()> {
    // Java scores before asserting `more`, reading whatever an exhausted
    // scorer returns; here scoring past the end is an error, so `more` is
    // asserted first.
    assert!(more, "scorer has no more docs, collector has {doc}");
    let scorer_doc = scorer.doc_id();
    let scorer_score = scorer.score()?;
    let scorer_score2 = scorer.score()?;
    assert_eq!(scorer_doc, doc, "scorerDoc={scorer_doc},doc={doc}");
    assert!(
        (score - scorer_score).abs() <= max_diff,
        "score={score}, scorerScore={scorer_score}"
    );
    assert!(
        (scorer_score2 - scorer_score).abs() <= max_diff,
        "scorerScorer={scorer_score}, scorerScore2={scorer_score2}"
    );
    Ok(())
}

/// `checkFirstSkipTo`: a fresh scorer advanced to any document between two
/// collected ones lands on the second, with its score. Java's stride:
/// `i += intervalTimes32++ / 1024`, so the first documents are checked many
/// times over and later ones ever more sparsely.
///
/// # Errors
/// Whatever searching reports.
pub fn check_first_skip_to(query: &BooleanQuery, searcher: &IndexSearcher<'_, '_>) -> Result<()> {
    const MAX_DIFF: f32 = 1e-3;
    let per_leaf = hits_per_leaf(searcher, query)?;
    let mut interval_times32: i64 = 32;
    for (leaf, hits) in per_leaf.iter().enumerate() {
        let mut last_doc = -1;
        if !hits.is_empty() {
            test_support::with_leaf(searcher, query, leaf, |ctx| {
                for &(doc, score) in hits {
                    let mut i = last_doc + 1;
                    while i <= doc {
                        let mut scorer = ctx
                            .scorer(ScoreMode::Complete)?
                            .expect("a leaf with hits has a scorer");
                        assert!(
                            scorer.exact_advance(i)? != NO_MORE_DOCS,
                            "query collected {doc} but advance({i}) says no more docs!"
                        );
                        assert_eq!(
                            doc,
                            scorer.doc_id(),
                            "query collected {doc} but advance({i}) got to {}",
                            scorer.doc_id()
                        );
                        let advance_score = scorer.score()?;
                        let again = scorer.score()?;
                        assert!(
                            (advance_score - again).abs() <= MAX_DIFF,
                            "unstable advance({i}) score!"
                        );
                        assert!(
                            (score - advance_score).abs() <= MAX_DIFF,
                            "query assigned doc {doc} a score of <{score}> but advance({i}) has <{advance_score}>!"
                        );
                        let step = interval_times32 / 1024;
                        interval_times32 += 1;
                        i += i32::try_from(step).unwrap_or(i32::MAX);
                    }
                    last_doc = doc;
                }
                Ok(())
            })?;
        }
        assert_no_past_segment_end(searcher, query, leaf, last_doc)?;
    }
    Ok(())
}

/// The collector of [`check_bulk_scorer_skip_to`]'s windows: each document
/// is the scorer's, with the scorer's score.
struct FollowScorer<'s, 'a> {
    scorer: &'s mut LeafScorer<'a>,
    min: i32,
    max: i32,
}

impl ScoringCollector for FollowScorer<'_, '_> {
    fn collect(&mut self, doc: i32, score: f32) {
        assert!(doc >= self.min);
        assert!(doc < self.max);
        assert_eq!(self.scorer.doc_id(), doc);
        let s = self.scorer.score().expect("scoring the scorer's document");
        assert!((s - score).abs() <= 0.01, "{s} != {score}");
        self.scorer.exact_next().expect("advancing the scorer");
    }
}

/// A collector that must see nothing.
struct NoMoreMatches;

impl ScoringCollector for NoMoreMatches {
    fn collect(&mut self, doc: i32, _score: f32) {
        panic!("no more matches expected, got {doc}");
    }
}

/// `checkBulkScorerSkipTo`: a bulk scorer scoring random windows collects
/// exactly what a scorer advanced alongside it finds.
///
/// # Errors
/// Whatever scoring reports.
pub fn check_bulk_scorer_skip_to(
    r: &mut JavaRandom,
    query: &BooleanQuery,
    searcher: &IndexSearcher<'_, '_>,
) -> Result<()> {
    for leaf in 0..searcher.segments().len() {
        test_support::with_leaf(searcher, query, leaf, |ctx| {
            let scorer = ctx.scorer(ScoreMode::Complete)?;
            let bulk = ctx.bulk_scorer(ScoreMode::Complete)?;
            let (mut scorer, mut bulk) = match (scorer, bulk) {
                (None, None) => return Ok(()),
                (Some(mut s), None) => {
                    assert_eq!(s.exact_next()?, NO_MORE_DOCS);
                    return Ok(());
                }
                (None, Some(mut b)) => {
                    b.score(&mut NoMoreMatches, None, 0, NO_MORE_DOCS)?;
                    return Ok(());
                }
                (Some(s), Some(b)) => (s, b),
            };
            let mut up_to = 0i32;
            loop {
                let min = up_to + r.next_int_bounded(5);
                let bound = if r.next_boolean() { 10 } else { 5000 };
                let max = min + 1 + r.next_int_bounded(bound);
                if scorer.doc_id() < min {
                    scorer.exact_advance(min)?;
                }
                let next = {
                    let mut follow = FollowScorer {
                        scorer: &mut scorer,
                        min,
                        max,
                    };
                    bulk.score(&mut follow, None, min, max)?
                };
                assert!(max <= next, "next={next} < max={max}");
                assert!(
                    next <= scorer.doc_id(),
                    "next={next} > scorer at {}",
                    scorer.doc_id()
                );
                up_to = max;
                if scorer.doc_id() == NO_MORE_DOCS {
                    bulk.score(&mut NoMoreMatches, None, up_to, NO_MORE_DOCS)?;
                    break;
                }
            }
            Ok(())
        })?;
    }
    Ok(())
}

/// Counts a segment's matches: one by one (Java's `collect(int)` per
/// document of a `DocIdStream`), or taking the bulk paths' counts as given
/// (`DocIdStream.count()`).
struct Count {
    n: u64,
    batched: bool,
    /// Whether a batch path was offered.
    saw_batch: bool,
}

impl ScoringCollector for Count {
    fn collect(&mut self, _doc: i32, _score: f32) {
        self.n += 1;
    }
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::CompleteNoScores
    }
    fn add_hits(&mut self, n: u64) -> bool {
        self.saw_batch = true;
        if self.batched {
            self.n += n;
        }
        self.batched
    }
    fn collect_range(&mut self, min: i32, max: i32, score: f32) {
        self.saw_batch = true;
        if self.batched {
            self.n += u64::try_from(max - min).unwrap_or(0);
        } else {
            for doc in min..max {
                self.collect(doc, score);
            }
        }
    }
}

/// `checkCount`: counting a segment's matches through the bulk scorers'
/// batch paths (`add_hits`, `collect_range` -- this port's `DocIdStream`)
/// gives what counting them one by one does.
///
/// # Errors
/// Whatever scoring reports.
pub fn check_count(query: &BooleanQuery, searcher: &IndexSearcher<'_, '_>) -> Result<()> {
    for leaf in 0..searcher.segments().len() {
        test_support::with_leaf(searcher, query, leaf, |ctx| {
            let Some(mut bulk) = ctx.bulk_scorer(ScoreMode::CompleteNoScores)? else {
                return Ok(());
            };
            let mut slow = Count {
                n: 0,
                batched: false,
                saw_batch: false,
            };
            bulk.score(&mut slow, ctx.live_docs(), 0, NO_MORE_DOCS)?;
            if !slow.saw_batch {
                return Ok(());
            }
            let Some(mut bulk) = ctx.bulk_scorer(ScoreMode::CompleteNoScores)? else {
                assert_eq!(slow.n, 0);
                return Ok(());
            };
            let mut fast = Count {
                n: 0,
                batched: true,
                saw_batch: false,
            };
            bulk.score(&mut fast, ctx.live_docs(), 0, NO_MORE_DOCS)?;
            assert_eq!(slow.n, fast.n, "batch count != one-by-one count");
            Ok(())
        })?;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counting_takes_a_batch_as_given_or_one_by_one() {
        let mut fast = Count {
            n: 0,
            batched: true,
            saw_batch: false,
        };
        assert!(fast.add_hits(3));
        fast.collect_range(2, 5, 1.0);
        assert_eq!((fast.n, fast.saw_batch), (6, true));
        let mut slow = Count {
            n: 0,
            batched: false,
            saw_batch: false,
        };
        assert!(!slow.add_hits(3));
        slow.collect_range(2, 5, 1.0);
        assert_eq!((slow.n, slow.saw_batch), (3, true));
        assert!(matches!(slow.score_mode(), ScoreMode::CompleteNoScores));
    }

    #[test]
    #[should_panic(expected = "no more matches expected, got 4")]
    fn a_match_past_the_end_fails() {
        NoMoreMatches.collect(4, 0.0);
    }
}
