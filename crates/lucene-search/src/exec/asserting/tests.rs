//! Every check of the asserting wrappers fires on the defect it targets,
//! and stays silent on a scorer that keeps its contract.

use super::*;
use crate::bulk_scorer::DocScores;

/// A scorer over `approx`, two-phase when `matching` is given; `bad`
/// breaks one part of its contract.
#[derive(Default)]
struct Fake {
    approx: Vec<i32>,
    matching: Option<Vec<i32>>,
    doc: i32,
    score: f32,
    max: f32,
    up_to: Option<i32>,
    run_end: Option<i32>,
    match_cost: f32,
    /// `next_doc` and `advance` go backwards.
    backwards: bool,
}

fn fake(approx: &[i32]) -> Fake {
    Fake {
        approx: approx.to_vec(),
        doc: -1,
        score: 1.0,
        max: 2.0,
        ..Fake::default()
    }
}

impl Scorer for Fake {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.advance(self.doc + 1)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = if self.backwards {
            target - 2
        } else {
            self.approx
                .iter()
                .copied()
                .find(|&d| d >= target)
                .unwrap_or(NO_MORE_DOCS)
        };
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.approx.len() as i64
    }
    fn two_phase(&self) -> bool {
        self.matching.is_some()
    }
    fn matches(&mut self) -> Result<bool> {
        Ok(self.matching.as_ref().is_none_or(|m| m.contains(&self.doc)))
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
    fn score(&mut self) -> Result<f32> {
        Ok(self.score)
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        Ok(self.up_to.unwrap_or(target))
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(self.max)
    }
    fn doc_id_run_end(&self) -> i32 {
        self.run_end.unwrap_or(self.doc + 1)
    }
}

/// `f` wrapped for a scored top-level `TOP_SCORES` search.
fn wrapped(f: Fake) -> AssertingScorer<'static> {
    AssertingScorer::new(Box::new(f), true, true)
}

#[test]
fn wrap_is_a_no_op_without_a_guard_and_guards_nest() {
    assert!(!active());
    let s = wrap(Box::new(fake(&[1])), Mode::TopScores, true);
    assert_eq!(s.doc_id(), -1);
    {
        let _outer = AssertingGuard::new(1);
        {
            let _inner = AssertingGuard::new(2);
            assert!(active());
        }
        assert!(active());
        assert!(AssertingBulk::installed(Some(5)).is_some());
        let mut s = wrap(Box::new(fake(&[1, 3])), Mode::Complete, true);
        assert_eq!(s.next_doc().unwrap(), 1);
        // `Mode::Complete`: no minimum competitive score.
        let err = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            s.set_min_competitive_score(1.0)
        }));
        assert!(err.is_err());
    }
    assert!(!active());
    assert!(AssertingBulk::installed(None).is_none());
}

#[test]
fn a_sound_scorer_passes_and_every_call_is_forwarded() {
    let mut f = fake(&[1, 2, 5, 9]);
    f.matching = Some(vec![2, 9]);
    f.match_cost = 3.0;
    let mut s = wrapped(f);
    assert_eq!(s.cost(), 4);
    assert!(s.two_phase());
    assert_eq!(s.match_cost(), 3.0);
    assert_eq!(s.advance_shallow(0).unwrap(), 0);
    assert_eq!(s.max_score(3).unwrap(), 2.0);
    assert_eq!(s.next_doc().unwrap(), 1);
    assert!(!s.matches().unwrap());
    assert_eq!(s.next_doc().unwrap(), 2);
    assert!(s.matches().unwrap());
    assert_eq!(s.score().unwrap(), 1.0);
    assert_eq!(s.doc_id_run_end(), 3);
    s.set_min_competitive_score(0.5).unwrap();
    s.set_min_competitive_score(0.5).unwrap();
    assert!(!s.max_score_unbounded());
    assert!(s.constant_bits().is_none());
    assert!(s.constant_score().is_none());
    assert!(s.contains(2).is_none());
    assert!(s.indri_boost().is_none());
    assert_eq!(s.smoothing_score(2).unwrap(), 0.0);
    assert!(!s.prefers_batches());
    assert!(!s.constant_scores());
    assert!(!s.batch_matches());
    let mut keep = [false; 2];
    s.matches_batch(&[5, 9], &mut keep).unwrap();
    assert_eq!(keep, [false, true]);
    assert_eq!(s.score().unwrap(), 1.0);
    s.matches_batch(&[], &mut []).unwrap();
    assert_eq!(s.advance(10).unwrap(), NO_MORE_DOCS);

    // Not two-phase: `matches` forwards, in any state.
    let mut s = wrapped(fake(&[0, 1, 2, 3]));
    assert_eq!(s.next_doc().unwrap(), 0);
    assert!(s.matches().unwrap());
    let mut out = DocScores::default();
    s.next_docs_and_scores(3, None, &mut out).unwrap();
    assert_eq!(out.docs, [0, 1, 2]);
    assert_eq!(s.doc_id(), 3);
    assert_eq!(s.score().unwrap(), 1.0);
    s.next_docs_and_scores(NO_MORE_DOCS, None, &mut out)
        .unwrap();
    assert_eq!(s.doc_id(), NO_MORE_DOCS);
}

/// One test per check: `$body` runs over `$s`, which must panic with
/// `$msg`.
macro_rules! fires {
    ($($name:ident: $msg:literal => |$s:ident| $body:block)*) => {$(
        #[test]
        #[should_panic(expected = $msg)]
        #[allow(unused_mut, unused_assignments)]
        fn $name() {
            let mut $s = wrapped(fake(&[1, 3]));
            $body
        }
    )*};
}

fires! {
    next_doc_after_the_end: "nextDoc() called after NO_MORE_DOCS" => |s| {
        s.advance(4).unwrap();
        s.next_doc().unwrap();
    }
    advance_after_the_end: "advance() called after NO_MORE_DOCS" => |s| {
        s.advance(4).unwrap();
        s.advance(5).unwrap();
    }
    advance_not_past_the_current_doc: "target must be > docID()" => |s| {
        s.next_doc().unwrap();
        s.advance(1).unwrap();
    }
    backwards_next_doc: "backwards nextDoc from -1 to -2" => |s| {
        s = wrapped(Fake { backwards: true, ..fake(&[]) });
        s.next_doc().unwrap();
    }
    backwards_advance: "backwards advance" => |s| {
        s = wrapped(Fake { backwards: true, ..fake(&[]) });
        s.advance(5).unwrap();
    }
    matches_twice: "matches() in state Iterating" => |s| {
        s = wrapped(Fake { matching: Some(vec![1]), ..fake(&[1]) });
        s.next_doc().unwrap();
        assert!(s.matches().unwrap());
        s.matches().unwrap();
    }
    score_without_scores: "score() on a scorer built without scores" => |s| {
        s = AssertingScorer::new(Box::new(fake(&[1])), false, false);
        s.next_doc().unwrap();
        s.score().unwrap();
    }
    score_on_an_unconfirmed_candidate: "score() in state Approximating" => |s| {
        s = wrapped(Fake { matching: Some(vec![1]), ..fake(&[1]) });
        s.next_doc().unwrap();
        s.score().unwrap();
    }
    score_before_the_first_doc: "score() in state Iterating" => |s| {
        s.score().unwrap();
    }
    nan_score: "NaN score" => |s| {
        s = wrapped(Fake { score: f32::NAN, ..fake(&[1]) });
        s.next_doc().unwrap();
        s.score().unwrap();
    }
    negative_score: "negative score" => |s| {
        s = wrapped(Fake { score: -0.0, ..fake(&[1]) });
        s.next_doc().unwrap();
        s.score().unwrap();
    }
    score_above_max_score: "> getMaxScore(1)" => |s| {
        s = wrapped(Fake { score: 3.0, ..fake(&[1]) });
        s.next_doc().unwrap();
        s.advance_shallow(1).unwrap();
        s.score().unwrap();
    }
    advance_shallow_without_scores: "advanceShallow on a scorer built without scores" => |s| {
        s = AssertingScorer::new(Box::new(fake(&[1])), false, false);
        s.advance_shallow(1).unwrap();
    }
    advance_shallow_backwards: "called on decreasing targets" => |s| {
        s.advance_shallow(5).unwrap();
        s.advance_shallow(4).unwrap();
    }
    advance_shallow_behind_the_doc: "target = 0 < docID = 1" => |s| {
        s.next_doc().unwrap();
        s.advance_shallow(0).unwrap();
    }
    advance_shallow_up_to_below_target: "upTo = 1 < target = 2" => |s| {
        s = wrapped(Fake { up_to: Some(1), ..fake(&[1]) });
        s.advance_shallow(2).unwrap();
    }
    next_doc_behind_the_shallow_target: "assertion failed" => |s| {
        s.advance_shallow(5).unwrap();
        s.next_doc().unwrap();
    }
    advance_behind_the_shallow_target: "assertion failed" => |s| {
        s.advance_shallow(5).unwrap();
        s.advance(2).unwrap();
    }
    max_score_without_scores: "getMaxScore on a scorer built without scores" => |s| {
        s = AssertingScorer::new(Box::new(fake(&[1])), false, false);
        s.max_score(1).unwrap();
    }
    max_score_below_the_shallow_target: "upTo = 2 < last target = 3" => |s| {
        s.advance_shallow(3).unwrap();
        s.max_score(2).unwrap();
    }
    max_score_unpositioned: "Cannot get max scores" => |s| {
        s.max_score(2).unwrap();
    }
    min_competitive_score_without_scores: "setMinCompetitiveScore without scores" => |s| {
        s = AssertingScorer::new(Box::new(fake(&[1])), false, false);
        s.set_min_competitive_score(1.0).unwrap();
    }
    min_competitive_score_below_the_top_level: "not the top-level scoring clause" => |s| {
        s = AssertingScorer::new(Box::new(fake(&[1])), true, false);
        s.set_min_competitive_score(1.0).unwrap();
    }
    min_competitive_score_nan: "assertion failed: !min.is_nan()" => |s| {
        s.set_min_competitive_score(f32::NAN).unwrap();
    }
    min_competitive_score_going_down: "min competitive score went down" => |s| {
        s.set_min_competitive_score(2.0).unwrap();
        s.set_min_competitive_score(1.0).unwrap();
    }
    doc_id_run_end_not_past_the_doc: "docIDRunEnd=1, docID=1" => |s| {
        s = wrapped(Fake { run_end: Some(1), ..fake(&[1]) });
        s.next_doc().unwrap();
        s.doc_id_run_end();
    }
    nan_match_cost: "assertion failed: !cost.is_nan()" => |s| {
        s = wrapped(Fake { match_cost: f32::NAN, ..fake(&[1]) });
        s.match_cost();
    }
    negative_match_cost: "assertion failed: cost >= 0.0" => |s| {
        s = wrapped(Fake { match_cost: -1.0, ..fake(&[1]) });
        s.match_cost();
    }
    batch_unpositioned: "nextDocsAndScores on an unpositioned scorer" => |s| {
        s.next_docs_and_scores(5, None, &mut DocScores::default()).unwrap();
    }
    matches_batch_after_the_end: "matches_batch after NO_MORE_DOCS" => |s| {
        s.advance(4).unwrap();
        s.matches_batch(&[5], &mut [false]).unwrap();
    }
    matches_batch_behind_the_doc: "matches_batch from 1 <= 1" => |s| {
        s.next_doc().unwrap();
        s.matches_batch(&[1], &mut [false]).unwrap();
    }
    matches_batch_unsorted: "unsorted batch" => |s| {
        s.matches_batch(&[3, 1], &mut [false; 2]).unwrap();
    }
}

#[test]
#[should_panic(expected = "Approximation and scorer don't advance synchronously")]
fn matches_on_a_moved_scorer() {
    let mut s = wrapped(Fake {
        matching: Some(vec![1, 2]),
        ..fake(&[1, 2])
    });
    s.next_doc().unwrap();
    // The scorer moved under the wrapper, which still thinks it is on 1.
    s.inner.next_doc().unwrap();
    s.matches().unwrap();
}

/// Every hit, as `(doc, score)`.
#[derive(Default)]
struct All(Vec<(i32, f32)>);

impl ScoringCollector for All {
    fn collect(&mut self, doc: i32, score: f32) {
        self.0.push((doc, score));
    }
    fn collect_many(&mut self, docs: &[i32], scores: &[f32], doc_base: i32) -> usize {
        for (&d, &s) in docs.iter().zip(scores) {
            self.0.push((d + doc_base, s));
        }
        docs.len()
    }
}

#[test]
fn the_leaf_collector_forwards_in_order_hits() {
    let mut all = All::default();
    let mut c = AssertingLeafCollector::new(&mut all, 0, 10);
    let _ = c.min_competitive_score();
    let _ = c.score_mode();
    let _ = c.pruning_threshold();
    let _ = c.constant_score_hits_needed();
    let _ = c.count_losing_hits(0, 0.0);
    let _ = c.add_hits(0);
    c.collect(1, 1.0);
    assert_eq!(c.collect_many(&[0, 1], &[2.0, 3.0], 2), 2);
    assert_eq!(c.collect_many(&[], &[], 0), 0);
    c.collect_range(5, 7, 4.0);
    c.collect(9, 0.0);
    assert_eq!(all.0[..3], [(1, 1.0), (2, 2.0), (3, 3.0)]);
    assert_eq!(*all.0.last().unwrap(), (9, 0.0));
}

macro_rules! collector_fires {
    ($($name:ident: $msg:literal => |$c:ident| $body:block)*) => {$(
        #[test]
        #[should_panic(expected = $msg)]
        fn $name() {
            let mut all = All::default();
            let mut $c = AssertingLeafCollector::new(&mut all, 2, 8);
            $body
        }
    )*};
}

collector_fires! {
    collect_out_of_order: "Out of order : 4 3" => |c| { c.collect(4, 1.0); c.collect(3, 1.0); }
    collect_below_the_window: "Out of range: 1 < 2" => |c| { c.collect(1, 1.0); }
    collect_past_the_window: "Out of range: 8 >= 8" => |c| { c.collect(8, 1.0); }
    collect_a_negative_score: "score=-1 for doc 3" => |c| { c.collect(3, -1.0); }
    collect_a_nan_score: "score=NaN for doc 3" => |c| { c.collect(3, f32::NAN); }
    collect_many_out_of_order: "Out of order : 4 3" => |c| { c.collect_many(&[4, 3], &[1.0; 2], 0); }
    collect_many_below_the_window: "Out of range: 1 < 2" => |c| { c.collect_many(&[0], &[1.0], 1); }
    collect_many_past_the_window: "Out of range: 9 >= 8" => |c| { c.collect_many(&[9], &[1.0], 0); }
    collect_many_a_negative_score: "score=-1 for doc 3" => |c| { c.collect_many(&[3], &[-1.0], 0); }
    collect_range_behind_the_last: "assertion failed: min > self.last_collected" => |c| {
        c.collect(5, 1.0);
        c.collect_range(4, 6, 1.0);
    }
    collect_range_empty: "assertion failed: max > min" => |c| { c.collect_range(4, 4, 1.0); }
    collect_range_below_the_window: "Out of range: 1 < 2" => |c| { c.collect_range(1, 3, 1.0); }
    collect_range_past_the_window: "Out of range: 8 >= 8" => |c| { c.collect_range(3, 9, 1.0); }
    collect_range_a_negative_score: "score=-1 for docs [3, 4)" => |c| { c.collect_range(3, 4, -1.0); }
}

/// The hits of a bulk scorer over `docs` under random windows, split into
/// `[0, mid)` and `[mid, max)` calls.
fn bulk_hits(seed: i64, docs: &[i32], mid: i32, max_doc: Option<i32>) -> Vec<(i32, f32)> {
    let mut bulk = Bulk::scorer(Box::new(fake(docs)));
    let mut a = AssertingBulk::new(JavaRandom::new(seed), max_doc);
    let mut all = All::default();
    let next = a
        .score(&mut bulk, Mode::Complete, None, &mut all, 0, mid)
        .unwrap();
    assert!(next >= mid);
    a.score(
        &mut bulk,
        Mode::Complete,
        None,
        &mut all,
        next,
        NO_MORE_DOCS,
    )
    .unwrap();
    all.0
}

#[test]
fn random_windows_collect_every_hit_once() {
    let docs: Vec<i32> = (0..3000).step_by(7).collect();
    let expected: Vec<(i32, f32)> = docs.iter().map(|&d| (d, 1.0)).collect();
    for seed in 0..20 {
        assert_eq!(bulk_hits(seed, &docs, 1500, Some(3000)), expected);
        assert_eq!(bulk_hits(seed, &docs, 1500, None), expected);
    }
    // A window ending past `maxDoc` returns `NO_MORE_DOCS`.
    let mut bulk = Bulk::scorer(Box::new(fake(&[1])));
    let mut a = AssertingBulk::new(JavaRandom::new(1), Some(2));
    let mut all = All::default();
    assert_eq!(
        a.score(&mut bulk, Mode::Complete, None, &mut all, 0, 2)
            .unwrap(),
        NO_MORE_DOCS
    );
}

#[test]
#[should_panic(expected = "Scoring backward: min=1 while previous max was max=5")]
fn bulk_scoring_backwards() {
    let mut bulk = Bulk::scorer(Box::new(fake(&[7])));
    let mut a = AssertingBulk::new(JavaRandom::new(1), None);
    let mut all = All::default();
    a.score(&mut bulk, Mode::Complete, None, &mut all, 0, 5)
        .unwrap();
    a.score(&mut bulk, Mode::Complete, None, &mut all, 1, 6)
        .unwrap();
}

#[test]
#[should_panic(expected = "max must be greater than min, got min=5, and max=4")]
fn bulk_window_inverted() {
    let mut bulk = Bulk::scorer(Box::new(fake(&[7])));
    let mut a = AssertingBulk::new(JavaRandom::new(1), None);
    a.score(&mut bulk, Mode::Complete, None, &mut All::default(), 5, 4)
        .unwrap();
}

#[test]
#[should_panic(expected = "a bulk scorer past maxDoc must be done")]
fn bulk_scorer_past_max_doc_not_done() {
    // A segment claiming 2 documents whose scorer has a match at 5.
    let mut bulk = Bulk::scorer(Box::new(fake(&[1, 5])));
    let mut a = AssertingBulk::new(JavaRandom::new(1), Some(2));
    a.score(&mut bulk, Mode::Complete, None, &mut All::default(), 0, 2)
        .unwrap();
}

#[test]
fn random_int_between_matches_random_numbers() {
    let mut r = JavaRandom::new(7);
    for _ in 0..100 {
        let v = random_int_between(&mut r, 3, 9);
        assert!((3..=9).contains(&v));
    }
    // The full `int` range draws through `nextLong(rnd, n)`'s rejection.
    let mut r = JavaRandom::new(7);
    let v = random_int_between(&mut r, i32::MIN, i32::MAX);
    let _ = v;
    assert_eq!(
        next_long(&mut JavaRandom::new(3), 8),
        next_long(&mut JavaRandom::new(3), 8)
    );
    for seed in 0..50 {
        let v = next_long(&mut JavaRandom::new(seed), 3);
        assert!((0..3).contains(&v));
    }
}
