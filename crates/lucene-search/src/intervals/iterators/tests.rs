use super::*;

/// A scripted iterator: per document, its intervals `(start, end, gaps)`.
pub(crate) struct Scripted {
    docs: Vec<(i32, Vec<(i32, i32)>)>,
    at: Option<usize>,
    upto: usize,
    current: (i32, i32),
    cost: i64,
}

pub(crate) fn scripted(docs: &[(i32, &[(i32, i32)])]) -> BoxIntervals<'static> {
    Box::new(Scripted {
        docs: docs.iter().map(|(d, iv)| (*d, iv.to_vec())).collect(),
        at: None,
        upto: 0,
        current: (-1, -1),
        cost: docs.len() as i64,
    })
}

impl IntervalIterator for Scripted {
    // SENTINEL: `-1` = "not yet positioned", `DocIdSetIterator`'s own
    // unpositioned doc id; callers advance before reading a document.
    fn doc_id(&self) -> i32 {
        match self.at {
            None => -1,
            Some(i) => self.docs.get(i).map_or(NO_MORE_DOCS, |d| d.0),
        }
    }
    fn next_doc(&mut self) -> Result<i32> {
        let next = self.at.map_or(0, |i| (i + 1).min(self.docs.len()));
        self.at = Some(next);
        self.upto = 0;
        self.current = (-1, -1);
        Ok(self.doc_id())
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        loop {
            let d = self.next_doc()?;
            if d >= target {
                return Ok(d);
            }
        }
    }
    fn cost(&self) -> i64 {
        self.cost
    }
    fn start(&self) -> i32 {
        self.current.0
    }
    fn end(&self) -> i32 {
        self.current.1
    }
    fn gaps(&self) -> i32 {
        0
    }
    fn next_interval(&mut self) -> Result<i32> {
        let ivs = self
            .at
            .and_then(|i| self.docs.get(i))
            .map(|d| d.1.as_slice())
            .unwrap_or(&[]);
        self.current = ivs
            .get(self.upto)
            .copied()
            .unwrap_or((NO_MORE_INTERVALS, NO_MORE_INTERVALS));
        self.upto += 1;
        Ok(self.current.0)
    }
    fn match_cost(&self) -> f32 {
        2.0
    }
}

/// Every interval of every document: `(doc, start, end, gaps)`.
fn walk(mut it: BoxIntervals<'_>) -> Vec<(i32, i32, i32, i32)> {
    let mut out = Vec::new();
    while it.next_doc().unwrap() != NO_MORE_DOCS {
        while it.next_interval().unwrap() != NO_MORE_INTERVALS {
            out.push((it.doc_id(), it.start(), it.end(), it.gaps()));
        }
    }
    out
}

fn terms(positions: &[(i32, &[i32])]) -> BoxIntervals<'static> {
    let docs: Vec<(i32, Vec<(i32, i32)>)> = positions
        .iter()
        .map(|(d, ps)| (*d, ps.iter().map(|&p| (p, p)).collect()))
        .collect();
    Box::new(Scripted {
        docs,
        at: None,
        upto: 0,
        current: (-1, -1),
        cost: positions.len() as i64,
    })
}

#[test]
fn the_index_queue_is_lucenes_heap() {
    let keys = [5, 3, 3, 9, 1, 3];
    let less = |a: usize, b: usize| keys[a] < keys[b];
    let mut q = IndexQueue::new(2);
    assert_eq!(q.top(), None);
    assert_eq!(q.pop(&less), None);
    for i in 0..keys.len() {
        q.add(i, &less);
    }
    assert_eq!(q.size(), 6);
    assert_eq!(q.iter().count(), 6);
    let mut popped = Vec::new();
    while let Some(i) = q.pop(&less) {
        popped.push(keys[i]);
    }
    assert_eq!(popped, [1, 3, 3, 3, 5, 9]);
    q.add(3, &less);
    q.add(0, &less);
    assert_eq!(q.top(), Some(0));
    q.clear();
    assert_eq!(q.size(), 0);
}

#[test]
fn a_disjunction_merges_and_drops_contained_intervals() {
    let a = scripted(&[(1, &[(0, 0), (4, 4)]), (3, &[(2, 2)])]);
    let b = scripted(&[(1, &[(0, 1), (3, 5)]), (2, &[(7, 7)])]);
    let it = DisjunctionIntervals::new(vec![a, b]);
    assert_eq!(it.cost(), 4);
    assert_eq!(it.match_cost(), 4.0);
    assert_eq!(
        walk(Box::new(it)),
        [(1, 0, 0, 0), (1, 4, 4, 0), (2, 7, 7, 0), (3, 2, 2, 0)]
    );
    // Before its first interval and after its last.
    let mut it = DisjunctionIntervals::new(vec![terms(&[(0, &[1])])]);
    assert_eq!((it.start(), it.end(), it.gaps()), (-1, -1, 0));
    assert_eq!(it.current_ord(), None);
    it.next_doc().unwrap();
    assert_eq!(it.next_interval().unwrap(), 1);
    assert_eq!(it.current_ord(), Some(0));
    assert_eq!(it.next_interval().unwrap(), NO_MORE_INTERVALS);
    assert_eq!(
        (it.start(), it.end()),
        (NO_MORE_INTERVALS, NO_MORE_INTERVALS)
    );
    assert_eq!(it.next_interval().unwrap(), NO_MORE_INTERVALS);
    assert_eq!(it.advance(5).unwrap(), NO_MORE_DOCS);
}

#[test]
fn blocks_ordered_and_unordered_minimize_as_lucene() {
    let a = || terms(&[(0, &[0, 3, 6]), (1, &[2])]);
    let b = || terms(&[(0, &[1, 4, 8]), (1, &[0])]);
    assert_eq!(walk(block(vec![a(), b()])), [(0, 0, 1, 0), (0, 3, 4, 0)]);
    assert_eq!(
        walk(ordered(vec![a(), b()], None)),
        [(0, 0, 1, 0), (0, 3, 4, 0), (0, 6, 8, 1)]
    );
    assert_eq!(
        walk(unordered(vec![a(), b()], None)),
        [
            (0, 0, 1, 0),
            (0, 1, 3, 1),
            (0, 3, 4, 0),
            (0, 4, 6, 1),
            (0, 6, 8, 1),
            (1, 0, 2, 1)
        ]
    );
    // The match callback runs on every match the minimization settles on.
    let count = std::rc::Rc::new(std::cell::Cell::new(0));
    let c = std::rc::Rc::clone(&count);
    let cb: MatchCallback<'static> = Some(Box::new(move || {
        c.set(c.get() + 1);
        Ok(())
    }));
    assert_eq!(walk(ordered(vec![a(), b()], cb)).len(), 3);
    assert!(count.get() >= 3);
    let e: MatchCallback<'static> = Some(Box::new(|| Err(Error::Unsupported("x".into()))));
    let mut failing = ordered(vec![a(), b()], e);
    failing.next_doc().unwrap();
    assert!(failing.next_interval().is_err());
    // A conjunction is as costly as its cheapest leg, its match cost summed.
    let c = ordered(vec![a(), terms(&[(0, &[1])])], None);
    assert_eq!(c.cost(), 1);
    assert_eq!(c.match_cost(), 4.0);
}

#[test]
fn the_filtering_and_relative_iterators_filter_by_the_other_side() {
    let big = || scripted(&[(0, &[(0, 4), (6, 9)]), (2, &[(1, 1)])]);
    let small = || terms(&[(0, &[2, 7, 12]), (2, &[5])]);
    assert_eq!(
        walk(filtering(FilteringKind::Containing, big(), small())),
        [(0, 0, 4, 0), (0, 6, 9, 0)]
    );
    assert_eq!(
        walk(filtering(FilteringKind::ContainedBy, small(), big())),
        [(0, 2, 2, 0), (0, 7, 7, 0)]
    );
    assert_eq!(
        walk(filtering(FilteringKind::Overlapping, big(), small())),
        [(0, 0, 4, 0), (0, 6, 9, 0)]
    );
    let rel = |k, a, b| -> BoxIntervals<'static> { Box::new(RelativeIntervals::new(k, a, b)) };
    assert_eq!(
        walk(rel(RelativeKind::NotContaining, big(), small())),
        [(2, 1, 1, 0)]
    );
    assert_eq!(
        walk(rel(RelativeKind::NotContainedBy, small(), big())),
        [(0, 12, 12, 0), (2, 5, 5, 0)]
    );
    assert_eq!(
        walk(rel(RelativeKind::NonOverlapping, small(), big())),
        [(0, 12, 12, 0), (2, 5, 5, 0)]
    );
    let r = RelativeIntervals::new(RelativeKind::NonOverlapping, small(), big());
    assert_eq!(r.cost(), 2);
    assert_eq!(r.match_cost(), 4.0);
}

#[test]
fn filters_extensions_offsets_and_repeats() {
    let ab = || ordered(vec![terms(&[(0, &[0, 5])]), terms(&[(0, &[3, 6])])], None);
    let gaps: BoxIntervals<'static> =
        Box::new(FilteredIntervals::new(ab(), IntervalFilterKind::MaxGaps(1)));
    assert_eq!(walk(gaps), [(0, 5, 6, 0)]);
    let width: BoxIntervals<'static> = Box::new(FilteredIntervals::new(
        ab(),
        IntervalFilterKind::MaxWidth(4),
    ));
    assert_eq!(walk(width), [(0, 0, 3, 2), (0, 5, 6, 0)]);
    let ext: BoxIntervals<'static> = Box::new(ExtendedIntervals::new(terms(&[(0, &[1, 5])]), 2, 3));
    assert_eq!(walk(ext), [(0, 0, 4, 0), (0, 3, 8, 0)]);
    let mut ext = ExtendedIntervals::new(terms(&[(0, &[7])]), 0, i32::MAX);
    assert_eq!((ext.start(), ext.end()), (-1, -1));
    ext.advance(0).unwrap();
    ext.next_interval().unwrap();
    assert_eq!(ext.end(), NO_MORE_INTERVALS - 1);
    assert_eq!(ext.next_interval().unwrap(), NO_MORE_INTERVALS);
    assert_eq!(ext.end(), NO_MORE_INTERVALS);
    let before: BoxIntervals<'static> =
        Box::new(OffsetIntervals::new(terms(&[(0, &[0, 4])]), true));
    assert_eq!(walk(before), [(0, 0, 0, 0), (0, 3, 3, 0)]);
    let after: BoxIntervals<'static> =
        Box::new(OffsetIntervals::new(terms(&[(0, &[0, 4])]), false));
    assert_eq!(walk(after), [(0, 1, 1, 0), (0, 5, 5, 0)]);
    let mut edge =
        OffsetIntervals::new(scripted(&[(0, &[(0, i32::MAX), (0, i32::MAX - 1)])]), false);
    assert_eq!(edge.start(), -1);
    edge.next_doc().unwrap();
    assert_eq!(edge.next_interval().unwrap(), i32::MAX);
    assert_eq!(edge.next_interval().unwrap(), i32::MAX - 1);
    let rep: BoxIntervals<'static> =
        Box::new(DuplicateIntervals::new(terms(&[(0, &[1, 3, 4])]), 2));
    assert_eq!(walk(rep), [(0, 1, 3, 1), (0, 3, 4, 0)]);
    let mut short = DuplicateIntervals::new(terms(&[(0, &[1])]), 2);
    short.next_doc().unwrap();
    assert_eq!(short.next_interval().unwrap(), NO_MORE_INTERVALS);
    assert_eq!(short.next_interval().unwrap(), NO_MORE_INTERVALS);
    // The cache still holds the first copy and an unfilled slot.
    assert_eq!(short.width(), 2);
}

#[test]
fn at_least_n_of_m_minimizes_over_the_front_of_the_queue() {
    let a = || terms(&[(0, &[0, 6]), (1, &[2])]);
    let b = || terms(&[(0, &[2])]);
    let c = || terms(&[(0, &[3]), (1, &[9])]);
    let it = MinimumShouldMatchIntervals::new(vec![a(), b(), c()], 2, None);
    assert_eq!(it.cost(), 5);
    assert_eq!(
        walk(Box::new(it)),
        [(0, 0, 2, 1), (0, 2, 3, 0), (0, 3, 6, 2), (1, 2, 9, 6)]
    );
    let mut it = MinimumShouldMatchIntervals::new(vec![a(), b(), c()], 2, None);
    it.advance(0).unwrap();
    it.next_interval().unwrap();
    assert_eq!(it.current_iterators().len(), 2);
}
