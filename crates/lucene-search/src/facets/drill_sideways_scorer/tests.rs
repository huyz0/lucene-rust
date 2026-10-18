#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::collector::ScoreMode;
use crate::multi_segment::OpenSegment;

/// A scorer over a list of documents, each with whether its two-phase
/// confirmation passes, and a cost of the test's choosing (which picks the
/// strategy).
struct ListScorer {
    docs: Vec<(i32, bool)>,
    /// Index of the current document; `None` before the first.
    at: Option<usize>,
    cost: i64,
    two_phase: bool,
    match_cost: f32,
    /// Fail `next_doc` once this many have been asked for.
    fail_after: Option<usize>,
    calls: usize,
}

fn list(docs: &[i32], cost: i64) -> ListScorer {
    ListScorer {
        docs: docs.iter().map(|&d| (d, true)).collect(),
        at: None,
        cost,
        two_phase: false,
        match_cost: 0.0,
        fail_after: None,
        calls: 0,
    }
}

/// A two-phase scorer: `rejected` are on the approximation, but fail
/// `matches()`.
fn two_phase(docs: &[i32], rejected: &[i32], cost: i64, match_cost: f32) -> ListScorer {
    let mut all: Vec<(i32, bool)> = docs.iter().map(|&d| (d, true)).collect();
    all.extend(rejected.iter().map(|&d| (d, false)));
    all.sort_unstable();
    ListScorer {
        docs: all,
        two_phase: true,
        match_cost,
        ..list(&[], cost)
    }
}

impl Scorer for ListScorer {
    // SENTINEL: `-1` = "not positioned yet", `DocIdSetIterator.docID()`'s
    // contract; the scorer under test reads it only to advance past it.
    fn doc_id(&self) -> i32 {
        match self.at {
            None => -1,
            Some(i) => self.docs.get(i).map_or(NO_MORE_DOCS, |d| d.0),
        }
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.calls += 1;
        if self.fail_after.is_some_and(|n| self.calls > n) {
            return Err(Error::IllegalState("next_doc failed".into()));
        }
        self.at = Some(self.at.map_or(0, |i| i + 1));
        Ok(self.doc_id())
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.calls += 1;
        if self.fail_after.is_some_and(|n| self.calls > n) {
            return Err(Error::IllegalState("advance failed".into()));
        }
        let mut i = self.at.map_or(0, |i| i + 1);
        while self.docs.get(i).is_some_and(|d| d.0 < target) {
            i += 1;
        }
        self.at = Some(i);
        Ok(self.doc_id())
    }
    fn cost(&self) -> i64 {
        self.cost
    }
    fn two_phase(&self) -> bool {
        self.two_phase
    }
    fn matches(&mut self) -> Result<bool> {
        Ok(self.at.and_then(|i| self.docs.get(i)).is_some_and(|d| d.1))
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
    fn score(&mut self) -> Result<f32> {
        Ok(self.doc_id() as f32 / 10.0)
    }
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(f32::MAX)
    }
}

/// A collector recording what it was given.
#[derive(Default)]
struct Recorder {
    docs: Vec<(i32, f32)>,
    finished: usize,
    fail_collect: bool,
    fail_finish: bool,
}

impl<'a> SegmentCollector<'a> for Recorder {
    fn score_mode(&self) -> ScoreMode {
        ScoreMode::Complete
    }
    fn set_next_reader(&mut self, _ord: usize, _leaf: &OpenSegment<'a>) -> Result<()> {
        Ok(())
    }
    fn collect(&mut self, doc: i32, score: f32) -> Result<()> {
        if self.fail_collect {
            return Err(Error::IllegalState("collect failed".into()));
        }
        self.docs.push((doc, score));
        Ok(())
    }
    fn finish(&mut self) -> Result<()> {
        self.finished += 1;
        if self.fail_finish {
            return Err(Error::IllegalState("finish failed".into()));
        }
        Ok(())
    }
}

/// A test case: the base's documents (and the ones its confirmation
/// rejects), each dim's, and the live documents.
#[derive(Clone)]
struct Case {
    max_doc: i32,
    base: Vec<i32>,
    base_rejected: Vec<i32>,
    dims: Vec<Vec<i32>>,
    /// Per dim, the documents its confirmation rejects (two-phase when
    /// any).
    dims_rejected: Vec<Vec<i32>>,
    live: Option<Vec<bool>>,
}

/// The documents a drill-sideways search must report: a hit (every dim
/// matches) goes to the main collector and to every sideways collector, a
/// near miss (exactly one dim misses) to that dim's.
fn expected(c: &Case) -> (Vec<i32>, Vec<Vec<i32>>) {
    let mut hits = Vec::new();
    let mut sideways = vec![Vec::new(); c.dims.len()];
    for &doc in &c.base {
        if c.live.as_ref().is_some_and(|l| !l[doc as usize]) {
            continue;
        }
        let missed: Vec<usize> = (0..c.dims.len())
            .filter(|&d| !c.dims[d].contains(&doc))
            .collect();
        match missed[..] {
            [] => {
                hits.push(doc);
                for s in &mut sideways {
                    s.push(doc);
                }
            }
            [d] => sideways[d].push(doc),
            _ => {}
        }
    }
    (hits, sideways)
}

fn live_bits(c: &Case) -> Option<FixedBitSet> {
    c.live.as_ref().map(|l| {
        let mut bits = FixedBitSet::new(l.len());
        for (i, &v) in l.iter().enumerate() {
            if v {
                // FBS: `bits` is `FixedBitSet::new(l.len())`, `i` enumerates `l`.
                bits.set(i);
            }
        }
        bits
    })
}

/// Runs `c` with the given costs and `at_once`, returning the main
/// collector's and every sideways collector's documents and the strategy.
fn run(
    c: &Case,
    base_cost: i64,
    dim_costs: &[i64],
    at_once: bool,
    needs_scores: bool,
) -> (Vec<(i32, f32)>, Vec<Vec<i32>>, Strategy) {
    let base: BoxScorer<'_> = if c.base_rejected.is_empty() {
        Box::new(list(&c.base, base_cost))
    } else {
        Box::new(two_phase(&c.base, &c.base_rejected, base_cost, 1.0))
    };
    let dims: Vec<DocsAndCost<'_>> = c
        .dims
        .iter()
        .enumerate()
        .map(|(i, d)| {
            let rejected = c.dims_rejected.get(i).cloned().unwrap_or_default();
            let s: BoxScorer<'_> = if rejected.is_empty() {
                Box::new(list(d, dim_costs[i]))
            } else {
                Box::new(two_phase(d, &rejected, dim_costs[i], (10 - i) as f32))
            };
            DocsAndCost::new(s, i)
        })
        .collect();
    let mut scorer = DrillSidewaysScorer::new(c.max_doc, base, dims, at_once, needs_scores);
    let mut main = Recorder::default();
    let mut sideways: Vec<Recorder> = (0..c.dims.len()).map(|_| Recorder::default()).collect();
    STRATEGIES.with(|v| v.borrow_mut().clear());
    let live = live_bits(c);
    assert_eq!(
        scorer
            .score(&mut main, &mut sideways, live.as_ref(), 0, i32::MAX)
            .unwrap(),
        i32::MAX
    );
    for s in &sideways {
        assert_eq!(s.finished, 1);
    }
    let strategy = STRATEGIES.with(|v| v.borrow()[0]);
    (
        main.docs,
        sideways
            .into_iter()
            .map(|s| s.docs.into_iter().map(|d| d.0).collect())
            .collect(),
        strategy,
    )
}

/// A deterministic pseudo-random subset of `0..max_doc`, about one in
/// `every`.
fn subset(seed: u64, max_doc: i32, every: u64) -> Vec<i32> {
    let mut x = seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1;
    (0..max_doc)
        .filter(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x.is_multiple_of(every)
        })
        .collect()
}

fn case(seed: u64, max_doc: i32, dims: usize, deletions: bool) -> Case {
    Case {
        max_doc,
        base: subset(seed, max_doc, 2),
        base_rejected: Vec::new(),
        dims: (0..dims)
            .map(|d| subset(seed + 1 + d as u64, max_doc, 2))
            .collect(),
        dims_rejected: Vec::new(),
        live: deletions.then(|| (0..max_doc).map(|i| i % 7 != 3).collect()),
    }
}

fn check(c: &Case, base_cost: i64, dim_costs: &[i64], at_once: bool, want: Strategy) {
    let (hits, sideways, strategy) = run(c, base_cost, dim_costs, at_once, true);
    assert_eq!(strategy, want);
    let (want_hits, want_sideways) = expected(c);
    let got: Vec<i32> = hits.iter().map(|h| h.0).collect();
    assert_eq!(got, want_hits, "{want:?} hits");
    for &(doc, score) in &hits {
        assert_eq!(score, doc as f32 / 10.0);
    }
    for (d, (g, w)) in sideways.iter().zip(&want_sideways).enumerate() {
        let mut g = g.clone();
        g.sort_unstable();
        assert_eq!(&g, w, "{want:?} dim {d}");
    }
}

#[test]
fn every_strategy_reports_the_hits_and_near_misses() {
    for deletions in [false, true] {
        for seed in [1, 7, 42] {
            // Wider than two windows.
            let one = case(seed, 5000, 1, deletions);
            check(&one, 10, &[1000], false, Strategy::QueryFirstSingle);
            check(&one, 1000, &[1000], true, Strategy::QueryFirstSingle);
            check(&one, 1000, &[1000], false, Strategy::Union);
            let three = case(seed, 5000, 3, deletions);
            check(&three, 10, &[100, 100, 100], false, Strategy::QueryFirst);
            check(
                &three,
                1000,
                &[1000, 1000, 1000],
                true,
                Strategy::QueryFirst,
            );
            check(&three, 1000, &[1, 2, 3], false, Strategy::DrillDownAdvance);
            check(&three, 1000, &[1000, 1000, 1000], false, Strategy::Union);
        }
    }
}

#[test]
fn sparse_dims_exercise_the_runaway_dim() {
    // Dims that rarely match: the query-first path skips the base ahead to
    // whichever of two missing dims is nearer.
    for seed in [3, 5] {
        let mut c = case(seed, 3000, 3, true);
        c.dims = (0..3).map(|d| subset(seed * 10 + d, 3000, 9)).collect();
        check(&c, 10, &[500, 500, 500], false, Strategy::QueryFirst);
        check(&c, 1000, &[1, 2, 3], false, Strategy::DrillDownAdvance);
        check(&c, 1000, &[1000, 1000, 1000], false, Strategy::Union);
    }
}

#[test]
fn two_phase_confirmations_are_asked() {
    // Two-phase base and dims (a dim's rejected documents are not matches);
    // with deletions, so the drill-down-advance path confirms its second dim
    // too.
    for seed in [2, 9] {
        let mut c = case(seed, 4500, 3, true);
        c.base_rejected = subset(seed + 50, 4500, 5)
            .into_iter()
            .filter(|d| !c.base.contains(d))
            .collect();
        c.dims_rejected = (0..3)
            .map(|d| {
                subset(seed + 60 + d as u64, 4500, 4)
                    .into_iter()
                    .filter(|x| !c.dims[d].contains(x))
                    .collect()
            })
            .collect();
        check(&c, 10, &[100, 100, 100], false, Strategy::QueryFirst);
        check(&c, 1000, &[1, 2, 3], false, Strategy::DrillDownAdvance);
        check(&c, 1000, &[1000, 1000, 1000], false, Strategy::Union);
        let mut one = c.clone();
        one.dims.truncate(1);
        one.dims_rejected.truncate(1);
        check(&one, 10, &[1000], false, Strategy::QueryFirstSingle);
    }
}

#[test]
fn drill_down_advance_skips_the_second_dims_confirmation_without_deletions() {
    // Java's `acceptDocs == null || acceptDocs.get(doc) && matches()`: with
    // no deletions the second dim's approximation counts as a match.
    let c = Case {
        max_doc: 100,
        base: vec![1, 2, 3],
        base_rejected: Vec::new(),
        dims: vec![vec![1, 2, 3], vec![1, 3]],
        dims_rejected: vec![Vec::new(), vec![2]],
        live: None,
    };
    let (hits, sideways, strategy) = run(&c, 1000, &[1, 2], false, true);
    assert_eq!(strategy, Strategy::DrillDownAdvance);
    let hits: Vec<i32> = hits.iter().map(|h| h.0).collect();
    assert_eq!(hits, vec![1, 2, 3]);
    assert_eq!(sideways[1], vec![1, 2, 3]);
    // With deletions (none deleted) it is asked: 2 is a near miss of dim 1.
    let mut d = c.clone();
    d.live = Some(vec![true; 100]);
    let (hits, sideways, _) = run(&d, 1000, &[1, 2], false, true);
    let hits: Vec<i32> = hits.iter().map(|h| h.0).collect();
    assert_eq!(hits, vec![1, 3]);
    assert_eq!(sideways[1], vec![1, 2, 3]);
    assert_eq!(sideways[0], vec![1, 3]);
}

#[test]
fn scores_are_zero_when_not_needed() {
    let c = case(4, 300, 2, false);
    for (base_cost, costs, at_once) in [
        (10, [100, 100], false),
        (1000, [1, 2], false),
        (1000, [1000, 1000], false),
    ] {
        let (hits, _, _) = run(&c, base_cost, &costs, at_once, false);
        assert!(!hits.is_empty());
        assert!(hits.iter().all(|h| h.1 == 0.0));
    }
}

#[test]
fn an_empty_dim_leaves_only_its_near_misses() {
    let c = Case {
        max_doc: 50,
        base: vec![0, 5, 9, 20],
        base_rejected: Vec::new(),
        dims: vec![vec![0, 5, 20], Vec::new()],
        dims_rejected: Vec::new(),
        live: None,
    };
    let dims = vec![
        DocsAndCost::new(Box::new(list(&c.dims[0], 3)), 0),
        DocsAndCost::new(Box::new(EmptyScorer), 1),
    ];
    let mut scorer = DrillSidewaysScorer::new(50, Box::new(list(&c.base, 4)), dims, false, true);
    let mut main = Recorder::default();
    let mut sideways = vec![Recorder::default(), Recorder::default()];
    scorer
        .score(&mut main, &mut sideways, None, 0, i32::MAX)
        .unwrap();
    assert!(main.docs.is_empty());
    assert!(sideways[0].docs.is_empty());
    let docs: Vec<i32> = sideways[1].docs.iter().map(|d| d.0).collect();
    assert_eq!(docs, vec![0, 5, 20]);
    let mut e = EmptyScorer;
    assert_eq!(e.advance(3).unwrap(), NO_MORE_DOCS);
    assert_eq!(e.score().unwrap(), 0.0);
    assert_eq!(e.max_score(10).unwrap(), 0.0);
}

#[test]
fn refuses_a_partial_range() {
    let mut scorer = DrillSidewaysScorer::new(
        10,
        Box::new(list(&[1], 1)),
        vec![DocsAndCost::new(Box::new(list(&[1], 1)), 0)],
        false,
        true,
    );
    let mut main = Recorder::default();
    let mut sideways = vec![Recorder::default()];
    let err = scorer
        .score(&mut main, &mut sideways, None, 3, i32::MAX)
        .unwrap_err();
    assert!(matches!(err, Error::IllegalArgument(m) if m == "min must be 0, got 3"));
    let err = scorer
        .score(&mut main, &mut sideways, None, 0, 100)
        .unwrap_err();
    assert!(matches!(err, Error::IllegalArgument(m) if m == "maxDoc must be Integer.MAX_VALUE"));
    assert_eq!(sideways[0].finished, 0);
}

#[test]
fn finishes_the_sideways_collectors_when_scoring_fails() {
    let c = case(6, 200, 2, false);
    for (base_cost, costs) in [(10, [100, 100]), (1000, [1, 2]), (1000, [1000, 1000])] {
        // A failing main collector.
        let dims = vec![
            DocsAndCost::new(Box::new(list(&c.dims[0], costs[0])), 0),
            DocsAndCost::new(Box::new(list(&c.dims[1], costs[1])), 1),
        ];
        let mut scorer =
            DrillSidewaysScorer::new(200, Box::new(list(&c.base, base_cost)), dims, false, true);
        let mut main = Recorder {
            fail_collect: true,
            ..Recorder::default()
        };
        let mut sideways = vec![Recorder::default(), Recorder::default()];
        let err = scorer
            .score(&mut main, &mut sideways, None, 0, i32::MAX)
            .unwrap_err();
        assert!(matches!(err, Error::IllegalState(m) if m == "collect failed"));
        assert!(sideways.iter().all(|s| s.finished == 1));
    }
    // A failing finish is reported (the first), after every collector's.
    let dims = vec![
        DocsAndCost::new(Box::new(list(&c.dims[0], 100)), 0),
        DocsAndCost::new(Box::new(list(&c.dims[1], 100)), 1),
    ];
    let mut scorer = DrillSidewaysScorer::new(200, Box::new(list(&c.base, 10)), dims, false, true);
    let mut main = Recorder::default();
    let mut sideways = vec![
        Recorder {
            fail_finish: true,
            ..Recorder::default()
        },
        Recorder {
            fail_finish: true,
            ..Recorder::default()
        },
    ];
    let err = scorer
        .score(&mut main, &mut sideways, None, 0, i32::MAX)
        .unwrap_err();
    assert!(matches!(err, Error::IllegalState(m) if m == "finish failed"));
    assert!(sideways.iter().all(|s| s.finished == 1));
}

#[test]
fn a_failing_scorer_is_reported() {
    let c = case(8, 200, 2, false);
    for (base_cost, costs, which) in [
        (10, [100, 100], 0),
        (1000, [1, 2], 1),
        (1000, [1000, 1000], 2),
        (10, [100, 100], 2),
    ] {
        let mut base = list(&c.base, base_cost);
        let mut d0 = list(&c.dims[0], costs[0]);
        let mut d1 = list(&c.dims[1], costs[1]);
        match which {
            0 => base.fail_after = Some(3),
            1 => d0.fail_after = Some(3),
            _ => d1.fail_after = Some(3),
        }
        let dims = vec![
            DocsAndCost::new(Box::new(d0), 0),
            DocsAndCost::new(Box::new(d1), 1),
        ];
        let mut scorer = DrillSidewaysScorer::new(200, Box::new(base), dims, false, true);
        let mut main = Recorder::default();
        let mut sideways = vec![Recorder::default(), Recorder::default()];
        let r = scorer.score(&mut main, &mut sideways, None, 0, i32::MAX);
        assert!(matches!(r, Err(Error::IllegalState(_))), "{which}");
    }
}

#[test]
fn a_missing_sideways_collector_is_skipped() {
    // `collect_in` past the collectors given: nothing to collect into.
    let mut none: Vec<Recorder> = Vec::new();
    collect_in(&mut none, 2, 1, 0.0).unwrap();
    assert_eq!(slot_of(2049), 1);
    assert_eq!(slot_of(NO_MORE_DOCS), 2047);
}
