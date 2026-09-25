//! `DisiWrapper`, `DisiPriorityQueue`, and the `DisjunctionScorer` family:
//! `DisjunctionSumScorer` and `DisjunctionMaxScorer` over a
//! `DisjunctionDISIApproximation`.

use super::{BoxScorer, Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::{sum_relative_error_bound, sum_upper_bound};
use crate::Result;

/// `DisiWrapper`: a sub-scorer and the state a disjunction keeps about it.
pub(crate) struct Disi<'a> {
    pub(crate) scorer: BoxScorer<'a>,
    /// The sub-scorer's document as the disjunction last saw it.
    pub(crate) doc: i32,
    pub(crate) cost: i64,
    pub(crate) match_cost: f32,
    /// `WANDScorer`'s scaled block maximum.
    pub(crate) scaled_max_score: i64,
}

impl<'a> Disi<'a> {
    pub(crate) fn new(scorer: BoxScorer<'a>) -> Self {
        let cost = scorer.cost();
        let match_cost = if scorer.two_phase() {
            scorer.match_cost()
        } else {
            0.0
        };
        Self {
            scorer,
            doc: -1,
            cost,
            match_cost,
            scaled_max_score: 0,
        }
    }
}

/// `DisiPriorityQueue`: a binary min-heap of indices into a `[Disi]`, ordered
/// by current document.
#[derive(Default)]
pub(crate) struct DisiQueue {
    heap: Vec<usize>,
    /// A handful of members kept unordered and scanned, not heaped: Lucene's
    /// `DisiPriorityQueue.ofMaxSize` picks `DisiPriorityQueue2` for two
    /// clauses for the same reason -- at this size the heap's bookkeeping
    /// costs more than comparing every member (a two-clause dismax spent a
    /// fifth of its time in `down_heap` and `top_list`). Members on the top
    /// document come out in member order rather than heap order; a
    /// disjunction's `f64` sum of a few `f32` scores is exact either way
    /// unless they differ in magnitude by more than 2^28.
    linear: bool,
}

/// Up to this many members, [`DisiQueue::for_disjunction`] scans instead of
/// heaping.
const LINEAR_MAX: usize = 8;

impl DisiQueue {
    pub(crate) fn with_capacity(n: usize) -> Self {
        Self {
            heap: Vec::with_capacity(n),
            linear: false,
        }
    }

    /// A queue for `DisjunctionScorer`, which only ever asks for the top, the
    /// members on its document, and a new top after the top moved.
    pub(crate) fn for_disjunction(n: usize) -> Self {
        Self {
            heap: Vec::with_capacity(n),
            linear: n <= LINEAR_MAX,
        }
    }

    /// The member on the smallest document, the first such in member order.
    #[inline]
    fn scan_min(&self, disis: &[Disi<'_>]) -> usize {
        let mut best = self.heap[0];
        for &i in &self.heap[1..] {
            if disis[i].doc < disis[best].doc {
                best = i;
            }
        }
        best
    }

    pub(crate) fn len(&self) -> usize {
        self.heap.len()
    }

    /// Linear mode needs `disis` to find the top; heap mode ignores it.
    pub(crate) fn top_of(&self, disis: &[Disi<'_>]) -> Option<usize> {
        if self.linear && !self.heap.is_empty() {
            return Some(self.scan_min(disis));
        }
        self.heap.first().copied()
    }

    pub(crate) fn top(&self) -> Option<usize> {
        debug_assert!(!self.linear, "a linear queue's top needs the members");
        self.heap.first().copied()
    }

    /// The heap's members, in heap order (`for (DisiWrapper w : head)`).
    pub(crate) fn members(&self) -> &[usize] {
        &self.heap
    }

    pub(crate) fn push(&mut self, disis: &[Disi<'_>], i: usize) {
        self.heap.push(i);
        if self.linear {
            return;
        }
        let mut at = self.heap.len() - 1;
        while at > 0 {
            let parent = (at - 1) / 2;
            if disis[self.heap[at]].doc >= disis[self.heap[parent]].doc {
                break;
            }
            self.heap.swap(at, parent);
            at = parent;
        }
    }

    pub(crate) fn pop(&mut self, disis: &[Disi<'_>]) -> Option<usize> {
        debug_assert!(!self.linear, "a disjunction's queue never pops");
        let last = self.heap.pop()?;
        if self.heap.is_empty() {
            return Some(last);
        }
        let top = std::mem::replace(&mut self.heap[0], last);
        self.down_heap(disis);
        Some(top)
    }

    /// `updateTop()`: restores the heap after the top's `doc` grew.
    pub(crate) fn update_top(&mut self, disis: &[Disi<'_>]) -> usize {
        if self.linear {
            return self.scan_min(disis);
        }
        self.down_heap(disis);
        self.heap[0]
    }

    /// `updateTop(replacement)`.
    pub(crate) fn replace_top(&mut self, disis: &[Disi<'_>], i: usize) -> usize {
        debug_assert!(!self.linear, "a disjunction's queue never replaces its top");
        self.heap[0] = i;
        self.update_top(disis)
    }

    fn down_heap(&mut self, disis: &[Disi<'_>]) {
        let n = self.heap.len();
        let mut at = 0;
        loop {
            let left = 2 * at + 1;
            if left >= n {
                break;
            }
            let right = left + 1;
            let child = if right < n && disis[self.heap[right]].doc < disis[self.heap[left]].doc {
                right
            } else {
                left
            };
            if disis[self.heap[child]].doc >= disis[self.heap[at]].doc {
                break;
            }
            self.heap.swap(at, child);
            at = child;
        }
    }

    /// `topList()`: every member on the top's document, into `out`.
    pub(crate) fn top_list(&self, disis: &[Disi<'_>], out: &mut Vec<usize>) {
        out.clear();
        if self.linear {
            if let Some(top) = self.top_of(disis) {
                let doc = disis[top].doc;
                out.extend(self.heap.iter().copied().filter(|&i| disis[i].doc == doc));
            }
            return;
        }
        let Some(&top) = self.heap.first() else {
            return;
        };
        let doc = disis[top].doc;
        // Breadth-first over heap positions, in `out` itself: a child can only
        // be on the top's document if its parent is.
        out.push(0);
        let mut k = 0;
        while k < out.len() {
            let at = out[k];
            for child in [2 * at + 1, 2 * at + 2] {
                if child < self.heap.len() && disis[self.heap[child]].doc == doc {
                    out.push(child);
                }
            }
            k += 1;
        }
        for at in out.iter_mut() {
            *at = self.heap[*at];
        }
    }
}

/// How a disjunction combines its matching sub-scores.
#[derive(Debug, Clone, Copy)]
pub(crate) enum Combine {
    /// `DisjunctionSumScorer`.
    Sum,
    /// `DisjunctionMaxScorer` with this tie-breaker multiplier.
    Max(f32),
}

/// `DisjunctionScorer`: the union of its sub-scorers' approximations, with
/// two-phase verification when any sub-scorer has one.
pub(crate) struct DisjunctionScorer<'a> {
    subs: Vec<Disi<'a>>,
    heap: DisiQueue,
    doc: i32,
    cost: i64,
    needs_scores: bool,
    two_phase: bool,
    match_cost: f32,
    combine: Combine,
    /// Sub-scorers verified to match the current document.
    verified: Vec<usize>,
    /// Two-phase sub-scorers on the current document not yet verified.
    unverified: Vec<usize>,
    scratch: Vec<usize>,
}

impl<'a> DisjunctionScorer<'a> {
    pub(crate) fn new(scorers: Vec<BoxScorer<'a>>, combine: Combine, needs_scores: bool) -> Self {
        debug_assert!(scorers.len() >= 2);
        let subs: Vec<Disi<'a>> = scorers.into_iter().map(Disi::new).collect();
        let mut two_phase = false;
        let mut sum_match_cost = 0.0f32;
        let mut sum_approx_cost = 0i64;
        let mut cost = 0i64;
        for w in &subs {
            let weight = w.cost.max(1);
            sum_approx_cost = sum_approx_cost.saturating_add(weight);
            if w.scorer.two_phase() {
                two_phase = true;
                sum_match_cost += w.match_cost * weight as f32;
            }
            cost = cost.saturating_add(w.cost);
        }
        let mut heap = DisiQueue::for_disjunction(subs.len());
        for i in 0..subs.len() {
            heap.push(&subs, i);
        }
        Self {
            heap,
            doc: -1,
            cost,
            needs_scores,
            two_phase,
            match_cost: if two_phase {
                sum_match_cost / sum_approx_cost as f32
            } else {
                0.0
            },
            combine,
            verified: Vec::with_capacity(subs.len()),
            unverified: Vec::with_capacity(subs.len()),
            scratch: Vec::with_capacity(subs.len()),
            subs,
        }
    }

    /// `getSubMatches()`: the sub-scorers matching the current document.
    fn sub_matches(&mut self) -> Result<()> {
        if !self.two_phase {
            self.heap.top_list(&self.subs, &mut self.verified);
            return Ok(());
        }
        for k in 0..self.unverified.len() {
            let i = self.unverified[k];
            if self.subs[i].scorer.matches()? {
                self.verified.push(i);
            }
        }
        self.unverified.clear();
        Ok(())
    }
}

impl Scorer for DisjunctionScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        let Some(mut top) = self.heap.top_of(&self.subs) else {
            self.doc = NO_MORE_DOCS;
            return Ok(self.doc);
        };
        let cur = self.subs[top].doc;
        loop {
            let w = &mut self.subs[top];
            w.doc = w.scorer.next_doc()?;
            top = self.heap.update_top(&self.subs);
            if self.subs[top].doc != cur {
                break;
            }
        }
        self.doc = self.subs[top].doc;
        Ok(self.doc)
    }

    fn advance(&mut self, target: i32) -> Result<i32> {
        let Some(mut top) = self.heap.top_of(&self.subs) else {
            self.doc = NO_MORE_DOCS;
            return Ok(self.doc);
        };
        while self.subs[top].doc < target {
            let w = &mut self.subs[top];
            w.doc = w.scorer.advance(target)?;
            top = self.heap.update_top(&self.subs);
        }
        self.doc = self.subs[top].doc;
        Ok(self.doc)
    }

    fn cost(&self) -> i64 {
        self.cost
    }

    fn two_phase(&self) -> bool {
        self.two_phase
    }

    /// `DisjunctionScorer.TwoPhase.matches`.
    fn matches(&mut self) -> Result<bool> {
        self.verified.clear();
        self.unverified.clear();
        self.heap.top_list(&self.subs, &mut self.scratch);
        for k in 0..self.scratch.len() {
            let i = self.scratch[k];
            if !self.subs[i].scorer.two_phase() {
                self.verified.push(i);
                if !self.needs_scores {
                    return Ok(true);
                }
            } else {
                self.unverified.push(i);
            }
        }
        if !self.verified.is_empty() {
            return Ok(true);
        }
        let subs = &self.subs;
        self.unverified
            .sort_by(|&a, &b| subs[a].match_cost.total_cmp(&subs[b].match_cost));
        while !self.unverified.is_empty() {
            let i = self.unverified.remove(0);
            if self.subs[i].scorer.matches()? {
                self.verified.push(i);
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn match_cost(&self) -> f32 {
        self.match_cost
    }

    fn score(&mut self) -> Result<f32> {
        self.sub_matches()?;
        match self.combine {
            Combine::Sum => {
                let mut score = 0.0f64;
                for k in 0..self.verified.len() {
                    let i = self.verified[k];
                    score += f64::from(self.subs[i].scorer.score()?);
                }
                Ok(score as f32)
            }
            Combine::Max(tie) => {
                let mut score_max = 0.0f32;
                let mut other_sum = 0.0f64;
                for k in 0..self.verified.len() {
                    let i = self.verified[k];
                    let sub = self.subs[i].scorer.score()?;
                    if sub >= score_max {
                        other_sum += f64::from(score_max);
                        score_max = sub;
                    } else {
                        other_sum += f64::from(sub);
                    }
                }
                Ok((f64::from(score_max) + other_sum * f64::from(tie)) as f32)
            }
        }
    }

    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        let mut min = NO_MORE_DOCS;
        for w in &mut self.subs {
            if w.scorer.doc_id() <= target {
                min = min.min(w.scorer.advance_shallow(target)?);
            }
        }
        Ok(min)
    }

    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        match self.combine {
            Combine::Sum => {
                let mut sum = 0.0f64;
                for w in &mut self.subs {
                    if w.scorer.doc_id() <= up_to {
                        sum += f64::from(w.scorer.max_score(up_to)?);
                    }
                }
                Ok(sum_upper_bound(sum, self.subs.len()) as f32)
            }
            Combine::Max(tie) => {
                let mut score_max = 0.0f32;
                let mut other_sum = 0.0f64;
                for w in &mut self.subs {
                    if w.scorer.doc_id() <= up_to {
                        let sub = w.scorer.max_score(up_to)?;
                        if sub >= score_max {
                            other_sum += f64::from(score_max);
                            score_max = sub;
                        } else {
                            other_sum += f64::from(sub);
                        }
                    }
                }
                if tie == 0.0 {
                    return Ok(score_max);
                }
                other_sum *= 1.0 + 2.0 * sum_relative_error_bound(self.subs.len() - 1);
                Ok((f64::from(score_max) + other_sum * f64::from(tie)) as f32)
            }
        }
    }

    /// `DisjunctionDISIApproximation.docIDRunEnd`: the longest run among the
    /// clauses on the current document.
    fn doc_id_run_end(&self) -> i32 {
        let mut end = self.doc.saturating_add(1);
        for w in &self.subs {
            if w.doc == self.doc {
                end = end.max(w.scorer.doc_id_run_end());
            }
        }
        end
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        // `DisjunctionSumScorer` inherits `Scorable`'s no-op; a dismax with no
        // tie-breaker is bounded by its best clause alone.
        if let Combine::Max(tie) = self.combine {
            if tie == 0.0 {
                for w in &mut self.subs {
                    w.scorer.set_min_competitive_score(min)?;
                }
            }
        }
        Ok(())
    }
}
