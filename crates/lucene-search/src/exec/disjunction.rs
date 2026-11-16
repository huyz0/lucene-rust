//! `DisiWrapper`, `DisiPriorityQueue`, and the `DisjunctionScorer` family:
//! `DisjunctionSumScorer` and `DisjunctionMaxScorer` over a
//! `DisjunctionDISIApproximation`.

use std::marker::PhantomData;

use super::{Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::{sum_relative_error_bound, sum_upper_bound};
use crate::Result;

/// `DisiWrapper`: a sub-scorer and the state a disjunction keeps about it.
///
/// `S` is the sub-scorers' type: `dyn Scorer` in the scorer tree, a concrete
/// scorer where a bulk scorer knows it (every call then monomorphic).
pub(crate) struct Disi<'a, S: Scorer + ?Sized + 'a = dyn Scorer + 'a> {
    pub(crate) scorer: Box<S>,
    /// The sub-scorer's document as the disjunction last saw it.
    pub(crate) doc: i32,
    pub(crate) cost: i64,
    pub(crate) match_cost: f32,
    /// `WANDScorer`'s scaled block maximum.
    pub(crate) scaled_max_score: i64,
    life: PhantomData<&'a ()>,
}

impl<'a, S: Scorer + ?Sized + 'a> Disi<'a, S> {
    pub(crate) fn new(scorer: Box<S>) -> Self {
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
            life: PhantomData,
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
    fn scan_min<S: Scorer + ?Sized>(&self, disis: &[Disi<'_, S>]) -> usize {
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

    /// Whether the members are scanned in member order rather than heaped.
    #[inline]
    fn is_linear(&self) -> bool {
        self.linear
    }

    /// Linear mode needs `disis` to find the top; heap mode ignores it.
    pub(crate) fn top_of<S: Scorer + ?Sized>(&self, disis: &[Disi<'_, S>]) -> Option<usize> {
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

    pub(crate) fn push<S: Scorer + ?Sized>(&mut self, disis: &[Disi<'_, S>], i: usize) {
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

    pub(crate) fn pop<S: Scorer + ?Sized>(&mut self, disis: &[Disi<'_, S>]) -> Option<usize> {
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
    pub(crate) fn update_top<S: Scorer + ?Sized>(&mut self, disis: &[Disi<'_, S>]) -> usize {
        if self.linear {
            return self.scan_min(disis);
        }
        self.down_heap(disis);
        self.heap[0]
    }

    /// `updateTop(replacement)`.
    pub(crate) fn replace_top<S: Scorer + ?Sized>(
        &mut self,
        disis: &[Disi<'_, S>],
        i: usize,
    ) -> usize {
        debug_assert!(!self.linear, "a disjunction's queue never replaces its top");
        self.heap[0] = i;
        self.update_top(disis)
    }

    fn down_heap<S: Scorer + ?Sized>(&mut self, disis: &[Disi<'_, S>]) {
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
    pub(crate) fn top_list<S: Scorer + ?Sized>(&self, disis: &[Disi<'_, S>], out: &mut Vec<usize>) {
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
pub(crate) struct DisjunctionScorer<'a, S: Scorer + ?Sized + 'a = dyn Scorer + 'a> {
    subs: Vec<Disi<'a, S>>,
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
    /// `DisjunctionMaxScorer.disjunctionBlockPropagator`: set only under
    /// `TOP_SCORES`.
    propagator: Option<BlockBoundaryPropagator>,
    /// The documents of the clauses read ahead by [`Scorer::next_docs_unscored`].
    window: Window,
}

/// How many documents [`Window`] spans at most.
const WINDOW_DOCS: i32 = 4096;

/// What [`DisjunctionScorer::next_docs_unscored`] read ahead: the union of
/// the clauses' documents in `base..end` as bits (`BooleanScorer`'s window),
/// every clause already past `end`. While `active`, the scorer's current
/// document is the window's next set bit.
#[derive(Default)]
struct Window {
    active: bool,
    base: i32,
    end: i32,
    bits: Vec<u64>,
    /// A clause's batch.
    batch: crate::bulk_scorer::DocScores,
    /// Scored: each clause's membership bits and scores, indexed from `base`.
    scored: bool,
    member: Vec<Vec<u64>>,
    scores: Vec<Vec<f32>>,
}

/// Most clauses a scored window sums.
const WINDOW_SCORED_CLAUSES: usize = 4;

impl Window {
    /// The first set bit at or after document `from`, as a document.
    fn next_from(&self, from: i32) -> Option<i32> {
        if from >= self.end {
            return None;
        }
        // ARITH: `base <= from < end`, a window of at most `WINDOW_DOCS`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            let at = (from - self.base).max(0) as usize;
            let mut w = at >> 6;
            let mut word = self.bits.get(w).copied()? & (u64::MAX << (at & 63));
            loop {
                if word != 0 {
                    return Some(self.base + (w << 6) as i32 + word.trailing_zeros() as i32);
                }
                w += 1;
                word = *self.bits.get(w)?;
            }
        }
    }
}

/// A clause a [`BlockBoundaryPropagator`] reads through.
pub(crate) trait Clause {
    type Scorer: Scorer + ?Sized;
    fn scorer(&mut self) -> &mut Self::Scorer;
}

impl<S: Scorer + ?Sized> Clause for Disi<'_, S> {
    type Scorer = S;
    fn scorer(&mut self) -> &mut S {
        &mut self.scorer
    }
}

/// `DisjunctionScoreBlockBoundaryPropagator`: block boundaries for a dis-max
/// under `TOP_SCORES`. The clauses are ordered by their global maximum score
/// (then cost); once the minimum competitive score exceeds a clause's global
/// maximum, that clause (and every lower one) no longer bounds the block, so
/// the boundary follows the "lead" clause and the stronger ones only.
#[derive(Debug)]
pub(crate) struct BlockBoundaryPropagator {
    /// Indices into the scorer's sub-scorers, ascending by global max score.
    order: Vec<usize>,
    max_scores: Vec<f32>,
    lead: usize,
}

impl BlockBoundaryPropagator {
    pub(crate) fn new<C: Clause>(subs: &mut [C]) -> Result<Self> {
        let mut keyed = Vec::with_capacity(subs.len());
        for (i, w) in subs.iter_mut().enumerate() {
            let s = w.scorer();
            s.advance_shallow(0)?;
            keyed.push((s.max_score(NO_MORE_DOCS)?, s.cost(), i));
        }
        // `Arrays.sort` on objects is stable; `Float.compareTo` is a total order.
        keyed.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
        let order: Vec<usize> = keyed.iter().map(|k| k.2).collect();
        let mut max_scores = Vec::with_capacity(order.len());
        for &i in &order {
            max_scores.push(subs[i].scorer().max_score(NO_MORE_DOCS)?);
        }
        Ok(Self {
            order,
            max_scores,
            lead: 0,
        })
    }

    pub(crate) fn advance_shallow<C: Clause>(&self, subs: &mut [C], target: i32) -> Result<i32> {
        // Clauses below the lead only propagate.
        for &i in &self.order[..self.lead] {
            let s = subs[i].scorer();
            if s.doc_id() < target {
                s.advance_shallow(target)?;
            }
        }
        // The lead and the clauses above it bound the block.
        let lead = subs[self.order[self.lead]].scorer();
        let lead_doc = lead.doc_id();
        let mut up_to = lead.advance_shallow(lead_doc.max(target))?;
        for &i in &self.order[self.lead + 1..] {
            let s = subs[i].scorer();
            if s.doc_id() <= target {
                up_to = up_to.min(s.advance_shallow(target)?);
            }
        }
        // The strongest clauses already beyond `target` end the block before
        // their next document, keeping them out of its maximum score.
        for &i in self.order[self.lead + 1..].iter().rev() {
            let doc = subs[i].scorer().doc_id();
            if doc > target {
                up_to = up_to.min(doc - 1);
            } else {
                break;
            }
        }
        Ok(up_to)
    }

    pub(crate) fn set_min_competitive_score(&mut self, min: f32) {
        while self.lead < self.max_scores.len() - 1 && min > self.max_scores[self.lead] {
            self.lead += 1;
        }
    }
}

impl<'a, S: Scorer + ?Sized + 'a> DisjunctionScorer<'a, S> {
    /// Installs the [`BlockBoundaryPropagator`], as `DisjunctionMaxScorer`
    /// does under `ScoreMode.TOP_SCORES`.
    pub(crate) fn with_block_propagator(mut self) -> Result<Self> {
        self.propagator = Some(BlockBoundaryPropagator::new(&mut self.subs)?);
        Ok(self)
    }

    pub(crate) fn new(scorers: Vec<Box<S>>, combine: Combine, needs_scores: bool) -> Self {
        debug_assert!(scorers.len() >= 2);
        let subs: Vec<Disi<'a, S>> = scorers.into_iter().map(Disi::new).collect();
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
            propagator: None,
            window: Window::default(),
        }
    }

    /// The smallest document the clauses are on.
    fn min_doc(&self) -> i32 {
        self.subs
            .iter()
            .map(|w| w.doc)
            .min()
            .unwrap_or(NO_MORE_DOCS)
    }

    /// Reads the clauses' documents from the current one on, below `up_to`,
    /// into the window; with `scored`, each clause's scores too.
    fn fill_window(&mut self, up_to: i32, scored: bool) -> Result<()> {
        let base = self.doc;
        let end = up_to.min(base.saturating_add(WINDOW_DOCS));
        // ARITH: `base < end <= base + WINDOW_DOCS`.
        #[allow(clippy::arithmetic_side_effects)]
        let len = (end - base) as usize;
        let win = &mut self.window;
        win.bits.clear();
        win.bits.resize(len.div_ceil(64), 0);
        win.scored = scored;
        if scored {
            win.member.resize_with(self.subs.len(), Vec::new);
            win.scores.resize_with(self.subs.len(), Vec::new);
        }
        for (i, w) in self.subs.iter_mut().enumerate() {
            if scored {
                win.member[i].clear();
                win.member[i].resize(len.div_ceil(64), 0);
                win.scores[i].resize(len, 0.0);
            }
            while w.doc < end {
                if scored {
                    w.scorer.next_docs_and_scores(end, None, &mut win.batch)?;
                } else {
                    w.scorer.next_docs_unscored(end, None, &mut win.batch)?;
                }
                for (k, &d) in win.batch.docs.iter().enumerate() {
                    // ARITH: `base <= d < end`.
                    #[allow(clippy::arithmetic_side_effects)]
                    let at = (d - base) as usize;
                    win.bits[at >> 6] |= 1 << (at & 63);
                    if scored {
                        win.member[i][at >> 6] |= 1 << (at & 63);
                        win.scores[i][at] = win.batch.scores[k];
                    }
                }
                w.doc = w.scorer.doc_id();
                if win.batch.docs.is_empty() {
                    break;
                }
            }
        }
        win.base = base;
        win.end = end;
        win.active = true;
        Ok(())
    }

    /// The score of `doc`, in the window: the members' scores summed in
    /// member order, as a linear queue's `topList()` is.
    fn window_score(&self, doc: i32) -> f32 {
        // ARITH: `base <= doc < end`.
        #[allow(clippy::arithmetic_side_effects)]
        let at = (doc - self.window.base) as usize;
        let mut score = 0.0f64;
        for (m, s) in self.window.member.iter().zip(&self.window.scores) {
            if m[at >> 6] >> (at & 63) & 1 == 1 {
                score += f64::from(s[at]);
            }
        }
        score as f32
    }

    /// The window's next document after the current one, or the clauses'
    /// smallest once it is spent.
    fn window_next(&mut self) -> i32 {
        match self.window.next_from(self.doc.saturating_add(1)) {
            Some(d) => self.doc = d,
            None => {
                self.window.active = false;
                self.doc = self.min_doc();
            }
        }
        self.doc
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

impl<S: Scorer + ?Sized> Scorer for DisjunctionScorer<'_, S> {
    fn doc_id(&self) -> i32 {
        self.doc
    }

    fn next_doc(&mut self) -> Result<i32> {
        if self.window.active {
            return Ok(self.window_next());
        }
        if self.heap.is_linear() {
            // The heap loop below, unrolled: every member on the current
            // document steps once, and the new document is the smallest.
            let cur = self.doc;
            let mut min = NO_MORE_DOCS;
            for w in &mut self.subs {
                if w.doc == cur {
                    w.doc = w.scorer.next_doc()?;
                }
                min = min.min(w.doc);
            }
            self.doc = min;
            return Ok(min);
        }
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
        if self.window.active {
            match self.window.next_from(target) {
                Some(d) => {
                    self.doc = d;
                    return Ok(d);
                }
                None => {
                    // Every clause is past the window; the clauses' own advance
                    // below takes it from there.
                    self.window.active = false;
                    self.doc = self.min_doc();
                    if self.doc >= target {
                        return Ok(self.doc);
                    }
                }
            }
        }
        if self.heap.is_linear() {
            // The heap loop below, unrolled: each member behind `target`
            // advances to it once, in whichever order.
            let mut min = NO_MORE_DOCS;
            for w in &mut self.subs {
                if w.doc < target {
                    w.doc = w.scorer.advance(target)?;
                }
                min = min.min(w.doc);
            }
            self.doc = min;
            return Ok(min);
        }
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
        if self.window.active && self.window.scored {
            return Ok(self.window_score(self.doc));
        }
        if !self.two_phase && self.heap.is_linear() {
            if let Combine::Sum = self.combine {
                // `topList()` of a linear queue is the members on the
                // document in member order; sum them without listing them.
                let doc = self.doc;
                let mut score = 0.0f64;
                for w in &mut self.subs {
                    if w.doc == doc {
                        score += f64::from(w.scorer.score()?);
                    }
                }
                return Ok(score as f32);
            }
        }
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
        if let Some(p) = &self.propagator {
            return p.advance_shallow(&mut self.subs, target);
        }
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
        if self.window.active {
            return self.doc.saturating_add(1);
        }
        let mut end = self.doc.saturating_add(1);
        for w in &self.subs {
            if w.doc == self.doc {
                end = end.max(w.scorer.doc_id_run_end());
            }
        }
        end
    }

    /// The window of `BooleanScorer`: the clauses' documents of a stretch
    /// read as bits, then listed in order -- a clause's batch each, not a
    /// virtual call per document.
    fn next_docs_unscored(
        &mut self,
        up_to: i32,
        live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        if self.needs_scores
            || self.two_phase
            || self.propagator.is_some()
            || !self.heap.is_linear()
        {
            return super::docs_one_by_one(self, up_to, live_docs, out);
        }
        out.docs.clear();
        out.scores.clear();
        while out.docs.len() < super::NEXT_DOCS_BATCH {
            if !self.window.active {
                if self.doc >= up_to {
                    break;
                }
                self.fill_window(up_to, false)?;
            }
            let doc = self.doc;
            if live_docs.is_none_or(|l| l.get_doc(doc)) {
                out.docs.push(doc);
                out.scores.push(0.0);
            }
            self.window_next();
        }
        Ok(())
    }

    /// The window of `BooleanScorer`, scored: each clause's batch of documents
    /// and scores into the window, then the documents listed in order with the
    /// members' scores summed in member order (as `score()` sums them).
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        if !self.needs_scores
            || self.two_phase
            || self.propagator.is_some()
            || !self.heap.is_linear()
            || !matches!(self.combine, Combine::Sum)
            || self.subs.len() > WINDOW_SCORED_CLAUSES
            || (self.window.active && !self.window.scored)
        {
            return super::docs_and_scores_one_by_one(self, up_to, live_docs, out);
        }
        out.docs.clear();
        out.scores.clear();
        while out.docs.len() < super::NEXT_DOCS_BATCH {
            if !self.window.active {
                if self.doc >= up_to {
                    break;
                }
                self.fill_window(up_to, true)?;
            }
            let doc = self.doc;
            if live_docs.is_none_or(|l| l.get_doc(doc)) {
                out.docs.push(doc);
                out.scores.push(self.window_score(doc));
            }
            self.window_next();
        }
        Ok(())
    }

    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        // `DisjunctionSumScorer` inherits `Scorable`'s no-op; a dismax with no
        // tie-breaker is bounded by its best clause alone.
        if let Some(p) = &mut self.propagator {
            p.set_min_competitive_score(min);
        }
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

#[cfg(test)]
mod tests;
