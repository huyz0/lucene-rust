#![cfg(test)]
// Test fixtures' own arithmetic -- see `docs/arithmetic-gate.md`'s "Test code".
#![allow(clippy::arithmetic_side_effects)]
use super::*;
use crate::exec::leaf::DocList;

fn bits(len: usize, set: &[usize]) -> Arc<FixedBitSet> {
    let mut b = FixedBitSet::new(len);
    for &i in set {
        b.set(i);
    }
    Arc::new(b)
}

fn list(docs: &[i32], scores: &[f32]) -> BoxScorer<'static> {
    Box::new(DocList::new(docs.to_vec(), scores.to_vec()))
}

#[test]
fn block_starts_and_java_math() {
    let b = bits(8, &[2, 5]);
    assert_eq!(block_start(&b, -1), 0);
    assert_eq!(block_start(&b, 1), 0);
    assert_eq!(block_start(&b, 3), 3);
    assert_eq!(block_start(&b, 100), 6);
    assert_eq!(block_start(&FixedBitSet::new(0), 3), 0);
    assert_eq!(next_set_bit(&b, -4), NO_MORE_DOCS);
    assert_eq!(next_set_bit(&b, 6), NO_MORE_DOCS);
    assert!(java_min(f64::NAN, 1.0).is_nan() && java_max(1.0, f64::NAN).is_nan());
    assert!(java_min(0.0, -0.0).is_sign_negative() && java_min(-0.0, 0.0).is_sign_negative());
    assert!(java_max(-0.0, 0.0).is_sign_positive() && java_max(0.0, -0.0).is_sign_positive());
    assert_eq!((java_min(1.0, 2.0), java_max(1.0, 2.0)), (1.0, 2.0));
    let mut none = ParentScore::new(ScoreMode::None);
    none.reset(3.0);
    none.add(4.0);
    assert_eq!(none.score(), 0.0);
    let mut avg = ParentScore::new(ScoreMode::Avg);
    avg.reset(1.0);
    avg.add(2.0);
    assert_eq!(avg.score(), 1.5);
}

/// `BlockJoinScorer` over children 0, 1 (parent 2) and 3 (parent 5).
#[test]
fn the_block_join_scorer_folds_children_into_parents() {
    for (mode, want) in [
        (ScoreMode::Total, [3.0, 4.0]),
        (ScoreMode::Max, [2.0, 4.0]),
        (ScoreMode::Min, [1.0, 4.0]),
    ] {
        let mut s =
            BlockJoinScorer::new(list(&[0, 1, 3], &[1.0, 2.0, 4.0]), bits(6, &[2, 5]), mode);
        assert_eq!(s.cost(), 3);
        assert_eq!(s.match_cost(), 10.0);
        assert_eq!(s.next_doc().unwrap(), 2);
        assert_eq!(s.score().unwrap(), want[0]);
        assert_eq!(s.score().unwrap(), want[0], "read again, not recomputed");
        assert_eq!(s.current_range(), (0, 1));
        assert_eq!(s.max_score(5).unwrap(), f32::INFINITY);
        assert!(s.max_score_unbounded());
        s.set_min_competitive_score(0.5).unwrap();
        assert_eq!(s.advance(3).unwrap(), 5);
        assert_eq!(s.score().unwrap(), want[1]);
        assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
    }
    let mut s = BlockJoinScorer::new(list(&[0], &[1.0]), bits(2, &[1]), ScoreMode::None);
    assert!(!s.max_score_unbounded(), "its child's maximum");
    s.next_doc().unwrap();
    s.max_score(1).unwrap();
    s.set_min_competitive_score(1.0).unwrap();
    // A child on a parent is Lucene's `IllegalStateException`.
    let mut s = BlockJoinScorer::new(list(&[0, 2], &[1.0, 1.0]), bits(3, &[2]), ScoreMode::Avg);
    assert_eq!(s.next_doc().unwrap(), 2);
    assert!(matches!(s.score(), Err(Error::IllegalState(_))));
}

struct Keep(Vec<(i32, f32)>, Option<f32>);
impl ScoringCollector for Keep {
    fn collect(&mut self, doc: i32, score: f32) {
        self.0.push((doc, score));
    }
    fn min_competitive_score(&self) -> Option<f32> {
        self.1
    }
    fn pruning_threshold(&self) -> Option<f32> {
        self.1
    }
}

#[test]
fn the_bulk_scorer_scores_whole_blocks_per_window() {
    let bulk = |mode| BlockJoinBulk {
        child: super::super::Bulk::scorer(list(&[0, 1, 3, 6], &[1.0, 2.0, 4.0, 8.0])),
        child_mode: Mode::Complete,
        parents: bits(8, &[2, 5, 7]),
        mode,
    };
    let mut b = bulk(ScoreMode::Total);
    let mut c = Keep(Vec::new(), None);
    assert_eq!(b.score(None, &mut c, 4, 4).unwrap(), 4, "an empty window");
    assert_eq!(b.score(None, &mut c, 3, 5).unwrap(), 5, "no parent in it");
    assert_eq!(b.score(None, &mut c, 0, 4).unwrap(), 4);
    assert_eq!(
        b.score(None, &mut c, 4, NO_MORE_DOCS).unwrap(),
        NO_MORE_DOCS
    );
    assert_eq!(c.0, [(2, 3.0), (5, 4.0), (7, 8.0)]);
    // The threshold reaches the child only for `Max`.
    let mut c = Keep(Vec::new(), Some(2.0));
    let w = BlockJoinCollector {
        inner: &mut c,
        parents: &bits(3, &[2]),
        mode: ScoreMode::Max,
        current_parent: -1,
        score: ParentScore::new(ScoreMode::Max),
        error: None,
    };
    assert_eq!(
        (w.min_competitive_score(), w.pruning_threshold()),
        (Some(2.0), Some(2.0))
    );
    let _ = w.score_mode();
    let w = BlockJoinCollector {
        mode: ScoreMode::Avg,
        ..w
    };
    assert_eq!(
        (w.min_competitive_score(), w.pruning_threshold()),
        (None, None)
    );
    assert_eq!(w.score_mode(), crate::collector::ScoreMode::Complete);
    // A child on a parent fails the scoring once the child is done.
    let mut b = BlockJoinBulk {
        child: super::super::Bulk::scorer(list(&[0, 2], &[1.0, 1.0])),
        child_mode: Mode::Complete,
        parents: bits(3, &[2]),
        mode: ScoreMode::Avg,
    };
    let mut c = Keep(Vec::new(), None);
    assert!(b.score(None, &mut c, 0, NO_MORE_DOCS).is_err());
}

#[test]
fn the_to_child_scorer_walks_each_parents_children() {
    let to_child = |parents: &[i32], set: Arc<FixedBitSet>| ToChildScorer {
        parent: list(parents, &vec![2.0; parents.len()]),
        parents: set,
        do_scores: true,
        parent_score: 0.0,
        child_doc: -1,
        parent_doc: 0,
    };
    // Parents 0 (no children), 3 (children 1, 2), 4 (none), 7 (5, 6).
    let set = bits(8, &[0, 3, 4, 7]);
    let mut s = to_child(&[0, 3, 4, 7], Arc::clone(&set));
    let mut got = Vec::new();
    while s.next_doc().unwrap() != NO_MORE_DOCS {
        got.push((s.doc_id(), s.score().unwrap()));
    }
    assert_eq!(got, [(1, 2.0), (2, 2.0), (5, 2.0), (6, 2.0)]);
    assert_eq!((s.cost(), s.max_score(1).unwrap()), (4, f32::INFINITY));
    let mut s = to_child(&[3, 4, 7], Arc::clone(&set));
    assert_eq!(s.advance(3).unwrap(), 5, "past a parent without children");
    assert_eq!(s.parent_doc(), 7);
    assert_eq!(s.advance(NO_MORE_DOCS).unwrap(), NO_MORE_DOCS);
    let mut s = to_child(&[3, 4], Arc::clone(&set));
    assert_eq!(s.advance(3).unwrap(), NO_MORE_DOCS);
    let mut s = to_child(&[3], Arc::clone(&set));
    assert_eq!(s.advance(3).unwrap(), NO_MORE_DOCS);
    // A parent query matching a child.
    let mut s = to_child(&[2], set);
    assert!(matches!(s.next_doc(), Err(Error::IllegalState(_))));
}

#[test]
fn the_parent_children_scorer_clips_to_one_block() {
    let mut s = ParentChildrenScorer {
        children: list(&[0, 2, 3, 6], &[1.0, 2.0, 3.0, 4.0]),
        first_child: 2,
        parent: 5,
        doc: -1,
    };
    assert_eq!((s.cost(), s.max_score(9).unwrap()), (3, f32::INFINITY));
    assert_eq!(s.next_doc().unwrap(), 2);
    assert_eq!(s.score().unwrap(), 2.0);
    assert_eq!(s.next_doc().unwrap(), 3);
    assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
}

#[test]
fn the_parents_children_scorer_limits_and_combines() {
    let scorer = |limit, parents: &[i32], children: &[i32]| ParentsChildrenScorer {
        parents: bits(9, &[3, 8]),
        parent: list(parents, &vec![1.0; parents.len()]),
        child: list(children, &vec![2.0; children.len()]),
        limit,
        do_scores: false,
        combiner: crate::join::ScoreCombiner::Sum,
        parent_score: 0.0,
        child_score: 0.0,
        parent_doc: 0,
        child_doc: -1,
        child_count: 0,
    };
    let mut s = scorer(1, &[3, 8], &[0, 1, 4, 5]);
    assert_eq!(s.cost(), 2);
    assert_eq!(s.next_doc().unwrap(), 0);
    assert_eq!(s.score().unwrap(), 1.0);
    assert_eq!(s.next_doc().unwrap(), 4, "one child a parent");
    assert_eq!(s.docs(), (8, 4));
    assert_eq!(s.max_score(9).unwrap(), f32::INFINITY);
    assert_eq!(s.next_doc().unwrap(), NO_MORE_DOCS);
    let mut s = scorer(
        crate::join::DEFAULT_CHILD_LIMIT_PER_PARENT,
        &[3, 8],
        &[0, 1, 4, 5],
    );
    assert_eq!(s.cost(), 4);
    assert_eq!(s.advance(1).unwrap(), 1);
    assert_eq!(s.advance(1).unwrap(), 1, "not past the current one");
    assert_eq!(s.advance(5).unwrap(), 5);
    // A parent query matching a child.
    let mut s = scorer(2, &[1], &[0]);
    assert!(matches!(s.next_doc(), Err(Error::IllegalState(_))));
}
