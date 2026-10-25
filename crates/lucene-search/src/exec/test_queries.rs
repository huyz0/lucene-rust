//! The scorers of [`crate::test_support::TestQuery`]: lucene-test-framework's
//! `RandomApproximationQuery` (a scorer turned into a two-phase one whose
//! approximation adds random false positives) and the deliberately broken
//! scorers of `TestQueryUtils`, which `QueryUtils` must catch.
//!
//! Compiled only under `cfg(test)` or the `asserting` feature.

use lucene_util::java_random::JavaRandom;

use super::asserting::random_int_between;
use super::{build, exact_advance, exact_next, BoxScorer, LeafContext, Mode, Scorer, NO_MORE_DOCS};
use crate::test_support::{BrokenKind, TestKind, TestQuery};
use crate::Result;

/// `Weight.scorerSupplier(leaf).get(...)` for a [`TestQuery`]: the wrapped
/// query's scorer (`in.scorer(context)`, not top-level), wrapped.
pub(crate) fn build<'a>(
    ctx: &LeafContext<'a>,
    q: &TestQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    // Java returns inWeight directly for BlockScore without scores. Other
    // wrappers omit setTopLevelScoringClause on their scorer suppliers.
    let inner_top_level =
        top_level && !mode.needs_scores() && matches!(q.kind(), TestKind::BlockScore(_));
    let Some(inner) = build::build(ctx, q.query(), boost, mode, inner_top_level)? else {
        return Ok(None);
    };
    Ok(Some(match q.kind() {
        // Java returns the original weight when scores are not required.
        TestKind::BlockScore(_) if !mode.needs_scores() => inner,
        TestKind::BlockScore(length) => Box::new(BlockScoreScorer::new(inner, *length)?),
        TestKind::DisableBulk => inner,
        // `new RandomApproximationScorer(subScorer, new Random(random.nextLong()))`.
        TestKind::RandomApproximation(random) => {
            let seed = random
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .next_long();
            Box::new(RandomApproximationScorer::new(inner, JavaRandom::new(seed)))
        }
        TestKind::Broken(kind) => Box::new(BrokenScorer {
            inner,
            kind: *kind,
            doc: -1,
            last_was_advance: false,
            exhausted: false,
            last_scored_doc: -2,
            score_calls: 0,
        }),
    }))
}

/// `RandomApproximationScorer` with its `RandomTwoPhaseView` and
/// `RandomApproximation`: the approximation lands anywhere in
/// `[target, next match]`, and `matches()` confirms it is on the match.
struct RandomApproximationScorer<'a> {
    /// The wrapped scorer, iterated exactly (`scorer.iterator()`).
    inner: BoxScorer<'a>,
    random: JavaRandom,
    /// The approximation's document.
    doc: i32,
    /// The document `matches()` was last called on.
    last_doc: i32,
    match_cost: f32,
}

impl<'a> RandomApproximationScorer<'a> {
    fn new(inner: BoxScorer<'a>, mut random: JavaRandom) -> Self {
        let match_cost = random.next_float() * 200.0;
        Self {
            inner,
            random,
            doc: -1,
            last_doc: -1,
            match_cost,
        }
    }
}

impl Scorer for RandomApproximationScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc.saturating_add(1))
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        if self.inner.doc_id() < target {
            exact_advance(&mut *self.inner, target)?;
        }
        let disi = self.inner.doc_id();
        self.doc = if disi == NO_MORE_DOCS {
            NO_MORE_DOCS
        } else {
            random_int_between(&mut self.random, target, disi)
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn matches(&mut self) -> Result<bool> {
        assert!(
            self.doc != -1 && self.doc != NO_MORE_DOCS,
            "matches() should not be called on doc ID {}",
            self.doc
        );
        assert!(
            self.last_doc != self.doc,
            "matches() has been called twice on doc ID {}",
            self.doc
        );
        self.last_doc = self.doc;
        Ok(self.doc == self.inner.doc_id())
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
    fn score(&mut self) -> Result<f32> {
        self.inner.score()
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        // The approximation's false positives sit before the wrapped
        // scorer's next match, which is already ahead of them.
        let inner = self.inner.doc_id();
        let target = if inner > target && self.doc != inner {
            inner
        } else {
            target
        };
        self.inner.advance_shallow(target)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        self.inner.max_score(up_to)
    }
    fn doc_id_run_end(&self) -> i32 {
        // `disi.docIDRunEnd()` holds for the exact iterator only, which a
        // two-phase wrapped scorer's approximation is not.
        if self.doc == self.inner.doc_id() && !self.inner.two_phase() {
            self.inner.doc_id_run_end()
        } else {
            self.doc.saturating_add(1)
        }
    }
}

/// `TestQueryUtils`' broken scorers, one per [`BrokenKind`], over the
/// wrapped scorer's exact iterator.
struct BrokenScorer<'a> {
    inner: BoxScorer<'a>,
    kind: BrokenKind,
    doc: i32,
    /// `ScoreMismatchQuery`: whether the current document was reached by
    /// `advance`.
    last_was_advance: bool,
    /// `EarlyTerminationQuery`: whether the iterator has moved once.
    exhausted: bool,
    /// `UnstableScoreQuery`: the document scored last and the calls on it.
    last_scored_doc: i32,
    score_calls: u32,
}

impl Scorer for BrokenScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.last_was_advance = false;
        if self.kind == BrokenKind::EarlyTermination {
            if self.exhausted {
                self.doc = NO_MORE_DOCS;
                return Ok(self.doc);
            }
            self.exhausted = true;
        }
        self.doc = exact_next(&mut *self.inner)?;
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.last_was_advance = true;
        self.exhausted = true;
        self.doc = match self.kind {
            BrokenKind::Advance => NO_MORE_DOCS,
            BrokenKind::SkipTooFar => {
                let doc = exact_advance(&mut *self.inner, target)?;
                if doc == NO_MORE_DOCS {
                    doc
                } else {
                    exact_next(&mut *self.inner)?
                }
            }
            _ => exact_advance(&mut *self.inner, target)?,
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    fn score(&mut self) -> Result<f32> {
        let score = self.inner.score()?;
        Ok(match self.kind {
            BrokenKind::UnstableScore => {
                if self.doc != self.last_scored_doc {
                    self.last_scored_doc = self.doc;
                    self.score_calls = 0;
                }
                self.score_calls += 1;
                if self.score_calls.is_multiple_of(2) {
                    score + 1.0
                } else {
                    score
                }
            }
            BrokenKind::ScoreMismatch if self.last_was_advance => score + 100.0,
            _ => score,
        })
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let max = self.inner.max_score(up_to)?;
        Ok(match self.kind {
            BrokenKind::UnstableScore => max + 1.0,
            BrokenKind::ScoreMismatch => max + 100.0,
            _ => max,
        })
    }
}

/// Java: `BlockScoreQueryWrapper.createWeight().scorerSupplier`.
/// The first and last slots are the -1 and NO_MORE_DOCS sentinels;
/// blocks count those slots, not document IDs or just the matching docs.
struct BlockScoreScorer {
    docs: Box<[i32]>,
    scores: Box<[f32]>,
    block_length: i32,
    index: usize,
    last_shallow_target: i32,
}

impl BlockScoreScorer {
    fn new(mut inner: BoxScorer<'_>, block_length: i32) -> Result<Self> {
        let mut docs = vec![-1];
        let mut scores = vec![0.0];
        loop {
            let doc = exact_next(&mut *inner)?;
            docs.push(doc);
            if doc == NO_MORE_DOCS {
                scores.push(0.0);
                break;
            }
            scores.push(inner.score()?);
        }
        Ok(Self {
            docs: docs.into_boxed_slice(),
            scores: scores.into_boxed_slice(),
            block_length,
            index: 0,
            last_shallow_target: -1,
        })
    }

    // Java: Arrays.binarySearch and insertion point, over the entire array.
    fn position(&self, target: i32) -> usize {
        self.docs.binary_search(&target).unwrap_or_else(|i| i)
    }

    // ARITH: Java array index is nonnegative and fits i32; remainder is at most index, with zero intentionally rejected by the Java contract.
    #[allow(clippy::arithmetic_side_effects)]
    fn start_of_block(&self, target: i32) -> usize {
        let index = i32::try_from(self.position(target)).expect("Java array index fits i32");
        // Java accepts zero/negative lengths at construction; invalid use can panic.
        usize::try_from(index - index % self.block_length).expect("nonnegative block start")
    }

    // ARITH: The buffered docs always contain two sentinel slots, so len - 1 is nonnegative; Java int addition is explicitly wrapping.
    #[allow(clippy::arithmetic_side_effects)]
    fn end_of_block(&self, target: i32) -> usize {
        let start = i32::try_from(self.start_of_block(target)).expect("Java array index fits i32");
        let end = start
            .wrapping_add(self.block_length)
            .min(i32::try_from(self.docs.len() - 1).expect("Java array length fits i32"));
        usize::try_from(end).expect("negative block end")
    }
}

impl Scorer for BlockScoreScorer {
    fn doc_id(&self) -> i32 {
        self.docs[self.index]
    }
    // ARITH: The terminal sentinel is checked before incrementing, so index + 1 stays within docs.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_doc(&mut self) -> Result<i32> {
        assert_ne!(self.docs[self.index], NO_MORE_DOCS);
        self.index += 1;
        Ok(self.docs[self.index])
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.index = self.position(target);
        Ok(self.docs[self.index])
    }
    // ARITH: Construction always writes both sentinel slots, so len - 2 is nonnegative.
    #[allow(clippy::arithmetic_side_effects)]
    fn cost(&self) -> i64 {
        i64::try_from(self.docs.len() - 2).expect("Java array length fits i64")
    }
    fn score(&mut self) -> Result<f32> {
        Ok(self.scores[self.index])
    }
    // ARITH: Every stored document is at least -1, so subtracting one cannot underflow i32.
    #[allow(clippy::arithmetic_side_effects)]
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.last_shallow_target = target;
        if target == NO_MORE_DOCS {
            return Ok(NO_MORE_DOCS);
        }
        Ok(self.docs[self.end_of_block(target)] - 1)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut max = 0.0_f32;
        let start = self.start_of_block(self.docs[self.index].max(self.last_shallow_target));
        for j in start..self.docs.len() {
            if self.docs[j] > up_to {
                break;
            }
            // Java Math.max keeps the first NaN payload and +0 over -0.
            // Since max starts at +0, only a larger score or first NaN replaces it.
            let score = self.scores[j];
            if !max.is_nan() && (score.is_nan() || score > max) {
                max = score;
            }
        }
        Ok(max)
    }
}

#[cfg(test)]
mod block_score_tests {
    use super::*;
    use crate::Error;

    struct Input {
        docs: Vec<i32>,
        scores: Vec<f32>,
        index: usize,
        two_phase: bool,
        fail_iteration: bool,
        fail_score: bool,
    }
    impl Scorer for Input {
        fn doc_id(&self) -> i32 {
            self.docs[self.index]
        }
        fn next_doc(&mut self) -> Result<i32> {
            if self.fail_iteration {
                return Err(Error::MissingPosInput);
            }
            self.index += 1;
            Ok(self.docs[self.index])
        }
        fn advance(&mut self, target: i32) -> Result<i32> {
            self.index = self.docs.binary_search(&target).unwrap_or_else(|i| i);
            Ok(self.doc_id())
        }
        fn cost(&self) -> i64 {
            5
        }
        fn two_phase(&self) -> bool {
            self.two_phase
        }
        fn matches(&mut self) -> Result<bool> {
            Ok(self.doc_id() != 2)
        }
        fn score(&mut self) -> Result<f32> {
            if self.fail_score {
                return Err(Error::MissingPosInput);
            }
            Ok(self.scores[self.index])
        }
        fn max_score(&mut self, _: i32) -> Result<f32> {
            Ok(f32::INFINITY)
        }
    }
    fn input() -> Input {
        Input {
            docs: vec![-1, 0, 2, 8, 9, NO_MORE_DOCS],
            scores: vec![0.0, 2.0, 11.0, 3.0, 7.0, 0.0],
            index: 0,
            two_phase: false,
            fail_iteration: false,
            fail_score: false,
        }
    }
    #[test]
    fn two_phase_is_consumed_exactly_and_scores_are_eagerly_owned() {
        let mut inner = input();
        inner.two_phase = true;
        let mut s = BlockScoreScorer::new(Box::new(inner), 2).unwrap();
        assert_eq!(&*s.docs, &[-1, 0, 8, 9, NO_MORE_DOCS]);
        assert_eq!(&*s.scores, &[0.0, 2.0, 3.0, 7.0, 0.0]);
        assert!(!s.two_phase());
        assert_eq!(s.cost(), 3);
        assert_eq!(s.doc_id(), -1);
        assert_eq!(s.score().unwrap(), 0.0);
        assert_eq!(s.advance_shallow(0).unwrap(), 7);
        assert_eq!(s.max_score(7).unwrap(), 2.0);
        assert_eq!(s.advance(8).unwrap(), 8);
        assert_eq!(s.max_score(9).unwrap(), 7.0);
        assert_eq!(s.next_doc().unwrap(), 9);
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
        assert_eq!(s.score().unwrap(), 0.0);
        assert_eq!(s.advance_shallow(NO_MORE_DOCS).unwrap(), NO_MORE_DOCS);
        assert_eq!(s.max_score(NO_MORE_DOCS).unwrap(), 0.0);
    }
    #[test]
    fn empty_scorer_still_owns_both_sentinels() {
        let mut inner = input();
        inner.docs = vec![-1, NO_MORE_DOCS];
        inner.scores = vec![0.0, 0.0];
        let mut s = BlockScoreScorer::new(Box::new(inner), 1).unwrap();
        assert_eq!(s.cost(), 0);
        assert_eq!(s.max_score(NO_MORE_DOCS).unwrap(), 0.0);
        assert_eq!(s.advance_shallow(0).unwrap(), NO_MORE_DOCS - 1);
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
    }
    #[test]
    fn errors_while_buffering_are_returned_before_a_scorer_escapes() {
        let mut inner = input();
        inner.fail_iteration = true;
        assert!(matches!(
            BlockScoreScorer::new(Box::new(inner), 2),
            Err(Error::MissingPosInput)
        ));
        let mut inner = input();
        inner.fail_score = true;
        assert!(matches!(
            BlockScoreScorer::new(Box::new(inner), 2),
            Err(Error::MissingPosInput)
        ));
    }
    #[test]
    fn max_propagates_nan_and_starts_at_positive_zero_for_negative_scores() {
        let mut inner = input();
        inner.scores[1] = -4.0;
        let mut s = BlockScoreScorer::new(Box::new(inner), 2).unwrap();
        assert_eq!(s.max_score(0).unwrap().to_bits(), 0);
        s.scores[1] = f32::from_bits(0x7fc0_1234);
        assert_eq!(s.max_score(0).unwrap().to_bits(), 0x7fc0_1234);
        s.scores[2] = f32::from_bits(0x7fc0_5678);
        assert_eq!(s.max_score(NO_MORE_DOCS).unwrap().to_bits(), 0x7fc0_1234);
        s.scores[1] = -0.0;
        assert_eq!(s.max_score(0).unwrap().to_bits(), 0);
    }
    #[test]
    fn memory_of_the_eight_match_fixture_buffer() {
        let mut inner = input();
        inner.docs = vec![-1, 0, 2, 3, 6, 8, 9, 11, 14, NO_MORE_DOCS];
        inner.scores = vec![0.0, 0.25, 2.0, 1.0, 5.0, 0.5, 3.0, 7.0, 4.0, 0.0];
        let s = BlockScoreScorer::new(Box::new(inner), 3).unwrap();
        assert_eq!(s.docs.len(), 10);
        assert_eq!(s.scores.len(), 10);
        let shallow = std::mem::size_of_val(&s);
        let docs = s.docs.len() * std::mem::size_of::<i32>();
        let scores = s.scores.len() * std::mem::size_of::<f32>();
        println!("block_score_buffer_shallow_bytes={shallow} docs_bytes={docs} scores_bytes={scores} total_bytes={}", shallow + docs + scores);
        assert_eq!(docs, 40);
        assert_eq!(scores, 40);
        assert_eq!(shallow + docs + scores, 128);
    }
    #[test]
    #[should_panic]
    fn next_after_terminal_sentinel_fails() {
        let mut s = BlockScoreScorer::new(Box::new(input()), 2).unwrap();
        s.advance(NO_MORE_DOCS).unwrap();
        s.next_doc().unwrap();
    }
}
