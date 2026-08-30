//! `SloppyPhraseMatcher`
//! (`/home/tuong/work/lucene-10.5.0/lucene/core/src/java/org/apache/lucene/search/SloppyPhraseMatcher.java`),
//! ported over already-decoded per-slot position lists.
//!
//! ## Why this exists at all
//!
//! Until this module, this port's sloppy phrase matching was **in-order only**:
//! it required `p_0 < p_1 < ... < p_{n-1}` in phrase order and admitted a match
//! when the summed gap slack fit the budget. Real Lucene admits **reordered**
//! matches -- `"quick fox"~2` matches a document reading `fox quick` -- so the
//! port under-matched every transposition, at every slop, silently.
//!
//! The whole mechanism is one line of `PhrasePositions`:
//!
//! ```text
//! position = postings.nextPosition() - offset;   // offset = slot index in the phrase
//! ```
//!
//! With every slot's positions shifted back by its own slot index, an exact
//! phrase is "all slots agree on one `position`", and `SloppyPhraseMatcher`'s
//! `matchLength = end - min(position)` is the width of the smallest window
//! covering one shifted position per slot. A transposition is not a special
//! case in that space; it is just a window of width 2. That is why Lucene needs
//! no ordering test and this port needed one.
//!
//! ## What `nextMatch` actually is
//!
//! `SloppyPhraseMatcher.nextMatch` is the classic "smallest range covering one
//! element from each of `k` sorted lists" walk: pop the currently-least
//! `PhrasePositions` off a `PhraseQueue`, record `end - position`, advance it,
//! and repeat. The refinement Lucene adds is that it keeps advancing the *same*
//! `pp` while it is still below the queue's next element, shrinking the current
//! window before re-queuing -- so successive `nextMatch()` calls emit a sequence
//! of locally-minimal `matchLength`s, and `PhraseScorer.score()` sums
//! `sloppyWeight() == 1f / (1f + matchLength)` over that sequence. This module
//! reproduces the walk statement for statement, so the *sequence* (and hence the
//! float sum, which is not associative) matches, not merely the final verdict.
//!
//! ## Repeats
//!
//! Two slots holding the same term must not settle on the same *raw* document
//! position -- `"a a"~9` does not match a document containing a single `a`.
//! Lucene handles that with `rptGroups`: repeating `PhrasePositions` are grouped,
//! spread apart at initialization (`advanceRepeatGroups`) and kept apart after
//! every advance (`advanceRpts`/`collide`). Both are ported here; see
//! [`PhraseRepeats`] for the one deliberate divergence in how the *groups* are
//! discovered.

use std::collections::{HashMap, HashSet};

/// Which slots of a phrase repeat each other, i.e. `SloppyPhraseMatcher`'s
/// `rptGroups`/`hasRpts`/`hasMultiTermRpts` triple, computed once per query
/// instead of once per scorer.
///
/// **The one deliberate divergence.** Java discovers the groups on the *first
/// candidate document* (`initFirstTime` -> `gatherRptGroups`) and, in the common
/// single-term case, groups two repeating `PhrasePositions` when their raw
/// positions coincide *in that document* rather than when their terms are equal.
/// For two slots holding the same term those two tests are the same test, since
/// identical terms have identical position lists and therefore identical first
/// positions. They differ only when two *different* repeating terms happen to
/// occupy the same position in whichever document Lucene happened to see first
/// (a same-position synonym), which makes Java's grouping depend on document
/// order. This port groups by term identity, which is stable and is what Java's
/// test is reaching for; the multi-term case below is Java's own term-graph
/// union, unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct PhraseRepeats {
    /// `PhrasePositions.rptGroup` per slot: a group id, or `-1` for a slot that
    /// repeats nothing.
    groups: Vec<i32>,
    /// `SloppyPhraseMatcher.hasRpts` -- some term occurs in two or more slots.
    has_rpts: bool,
    /// `SloppyPhraseMatcher.hasMultiTermRpts` -- some *repeating* slot accepts
    /// more than one term, which only a `MultiPhraseQuery` can produce.
    has_multi_term_rpts: bool,
}

impl PhraseRepeats {
    /// "No slot repeats any other": `hasRpts == false`, the case
    /// `SloppyPhraseMatcher.initSimple` exists for. Every `PhraseQuery` whose
    /// terms are pairwise distinct lands here.
    pub fn none(slots: usize) -> Self {
        Self {
            groups: vec![-1; slots],
            has_rpts: false,
            has_multi_term_rpts: false,
        }
    }

    /// [`Self::detect`] for a [`crate::query::PhraseQuery`], whose slots hold
    /// exactly one term each.
    pub fn for_phrase(terms: &[Vec<u8>]) -> Self {
        let slots: Vec<&[Vec<u8>]> = terms.iter().map(std::slice::from_ref).collect();
        Self::detect(&slots)
    }

    /// [`Self::detect`] for a [`crate::query::MultiPhraseQuery`], whose slots
    /// hold a set of accepted terms each (`MultiPhraseQuery.termArrays`).
    pub fn for_multi_phrase(term_arrays: &[Vec<Vec<u8>>]) -> Self {
        let slots: Vec<&[Vec<u8>]> = term_arrays.iter().map(Vec::as_slice).collect();
        Self::detect(&slots)
    }

    /// `initFirstTime`'s `repeatingTerms`/`repeatingPPs`/`gatherRptGroups`
    /// sequence: `slots[i]` is slot `i`'s accepted term set -- exactly one term
    /// for a `PhraseQuery`, one or more for a `MultiPhraseQuery`'s `termArrays`
    /// entry.
    ///
    /// Duplicate terms *within* one slot count toward the repetition tally, as
    /// Java's `repeatingTerms` (which iterates `pp.terms` without
    /// deduplicating) does.
    pub fn detect(slots: &[&[Vec<u8>]]) -> Self {
        // `repeatingTerms()`: a term is repeating once it has been seen twice,
        // and the terms are ordinaled in first-repeat order (Java's
        // `LinkedHashMap`), which is the order `termGroups` indexes by.
        let mut counts: HashMap<&[u8], usize> = HashMap::new();
        let mut repeating_ord: Vec<&[u8]> = Vec::new();
        for slot in slots {
            for term in *slot {
                let c = counts.entry(term.as_slice()).or_insert(0);
                *c += 1;
                if *c == 2 {
                    repeating_ord.push(term.as_slice());
                }
            }
        }
        if repeating_ord.is_empty() {
            return Self::none(slots.len());
        }
        let repeating: HashSet<&[u8]> = repeating_ord.iter().copied().collect();

        // `repeatingPPs()`: every slot mentioning at least one repeating term,
        // and `hasMultiTermRpts` iff one of those slots has more than one term.
        let mut rpp: Vec<usize> = Vec::new();
        let mut has_multi_term_rpts = false;
        for (i, slot) in slots.iter().enumerate() {
            if slot.iter().any(|t| repeating.contains(t.as_slice())) {
                rpp.push(i);
                has_multi_term_rpts |= slot.len() > 1;
            }
        }

        let mut groups = vec![-1i32; slots.len()];
        if has_multi_term_rpts {
            // `ppTermsBitSets` + `unionTermGroups` + `termGroups`: connected
            // components of the bipartite (slot, term) graph over the repeating
            // terms. Expressed as a union-find over term ordinals, which is the
            // same partition Java's repeated bit-set OR-and-remove computes.
            let ord_of: HashMap<&[u8], usize> = repeating_ord
                .iter()
                .enumerate()
                .map(|(i, t)| (*t, i))
                .collect();
            let mut parent: Vec<usize> = (0..repeating_ord.len()).collect();
            for &i in &rpp {
                let mut first: Option<usize> = None;
                for term in slots[i] {
                    let Some(&ord) = ord_of.get(term.as_slice()) else {
                        continue;
                    };
                    match first {
                        None => first = Some(ord),
                        Some(f) => union(&mut parent, f, ord),
                    }
                }
            }
            // Renumber the roots to dense group ids, in the order the groups are
            // first met, so the ids are deterministic.
            let mut group_of_root: HashMap<usize, i32> = HashMap::new();
            for &i in &rpp {
                for term in slots[i] {
                    let Some(&ord) = ord_of.get(term.as_slice()) else {
                        continue;
                    };
                    let root = find(&mut parent, ord);
                    let next = group_of_root.len() as i32;
                    let g = *group_of_root.entry(root).or_insert(next);
                    groups[i] = g;
                    break;
                }
            }
        } else {
            // Every repeating slot has exactly one term, so "same group" is
            // "same term" -- see this type's doc comment for why that is not
            // literally Java's first-document position test.
            let mut group_of_term: HashMap<&[u8], i32> = HashMap::new();
            for &i in &rpp {
                let term = slots[i][0].as_slice();
                let next = group_of_term.len() as i32;
                groups[i] = *group_of_term.entry(term).or_insert(next);
            }
        }

        Self {
            groups,
            has_rpts: true,
            has_multi_term_rpts,
        }
    }
}

/// Union-find `find` with path halving.
fn find(parent: &mut [usize], mut x: usize) -> usize {
    while parent[x] != x {
        parent[x] = parent[parent[x]];
        x = parent[x];
    }
    x
}

fn union(parent: &mut [usize], a: usize, b: usize) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        parent[rb] = ra;
    }
}

/// One `PhrasePositions`: a slot's already-decoded position list plus the
/// cursor, shifted position and repeat bookkeeping Java keeps on the object.
#[derive(Debug)]
struct Pp<'a> {
    positions: &'a [i32],
    /// Index of the *next* position to read -- Java's `count` counted down.
    idx: usize,
    /// `PhrasePositions.offset`: the slot's index within the phrase. This
    /// port's phrases are always at consecutive positions `0..n`, matching
    /// [`crate::query::PhraseQuery`]'s documented scope.
    offset: i32,
    /// `PhrasePositions.ord`.
    ord: usize,
    /// `PhrasePositions.position`: the raw position minus [`Pp::offset`].
    ///
    /// `i64`, where Java uses `int`: `end - position` is a difference of two
    /// values derived from positions read off disk, and a corrupt `.pos` can
    /// make that overflow `i32`. Widening costs nothing here (the walk is
    /// bounded by the list lengths, not by this arithmetic) and removes the
    /// only overflow in the module.
    position: i64,
    /// `PhrasePositions.rptGroup` -- `-1` for a slot that repeats nothing.
    rpt_group: i32,
    /// `PhrasePositions.rptInd`: this slot's index within its repeat group.
    rpt_ind: usize,
}

/// The matcher itself -- one instance per candidate document.
struct SloppyMatcher<'a> {
    pps: Vec<Pp<'a>>,
    /// `PhraseQueue`, as indices into [`Self::pps`] kept sorted ascending by
    /// `PhraseQueue.lessThan`'s key `(position, offset, ord)`. That key is a
    /// total order (offset alone already identifies a slot), so "the least
    /// element" is unambiguous and a sorted `Vec` and Java's binary heap
    /// necessarily pop the same element. A phrase has a handful of slots, so a
    /// sorted `Vec` with a `partition_point` insert beats a heap on both
    /// constant factor and code size.
    pq: Vec<usize>,
    /// `rptGroups`, flattened: group `g` is
    /// `rpt_slots[rpt_bounds[g]..rpt_bounds[g + 1]]`, ascending by offset.
    ///
    /// Flat rather than `Vec<Vec<usize>>` because `advanceRpts` walks a group
    /// while advancing the slots in it: with a nested `Vec` the borrow checker
    /// forces a clone of the group on **every advance**, i.e. one allocation
    /// per position of a repeating phrase. Indices copy out of `self` for free.
    rpt_slots: Vec<usize>,
    rpt_bounds: Vec<usize>,
    /// Scratch for `advanceRpts`' `FixedBitSet bits` and `rptStack`, reused
    /// across calls for the same reason.
    requeue: Vec<bool>,
    rpt_stack: Vec<usize>,
    has_rpts: bool,
    has_multi_term_rpts: bool,
    /// `SloppyPhraseMatcher.end`: the largest shifted position placed so far.
    end: i64,
    slop: i64,
    positioned: bool,
    /// `SloppyPhraseMatcher.matchLength`.
    match_length: i64,
}

impl<'a> SloppyMatcher<'a> {
    /// The constructor plus `resetPositions()`: place every slot and build the
    /// queue.
    ///
    /// `repeats.groups` must have one entry per slot -- it is
    /// [`PhraseRepeats::detect`]'s output for the same phrase.
    fn new(term_positions: &[&'a [i32]], repeats: &PhraseRepeats, slop: u32) -> Self {
        let group_count = if repeats.has_rpts {
            repeats
                .groups
                .iter()
                .copied()
                .max()
                .map_or(0, |m| (m + 1).max(0) as usize)
        } else {
            0
        };
        // `sortRptGroups` sorts each group by (query) offset; the slot index
        // *is* the offset, so visiting slots ascending already produces that
        // order, and a slot's position in its group is its `rptInd`.
        let mut rpt_bounds = vec![0usize; group_count.saturating_add(1)];
        for &g in &repeats.groups {
            if g >= 0 {
                rpt_bounds[(g as usize).saturating_add(1)] += 1;
            }
        }
        for g in 0..group_count {
            rpt_bounds[g + 1] += rpt_bounds[g];
        }
        let mut fill = rpt_bounds.clone();
        let mut rpt_slots = vec![0usize; rpt_bounds[group_count]];
        let mut pps: Vec<Pp<'a>> = term_positions
            .iter()
            .enumerate()
            .map(|(i, positions)| Pp {
                positions,
                idx: 0,
                offset: i as i32,
                ord: i,
                position: 0,
                rpt_group: repeats.groups[i],
                rpt_ind: 0,
            })
            .collect();
        for (slot, &g) in repeats.groups.iter().enumerate() {
            if g >= 0 {
                let at = fill[g as usize];
                rpt_slots[at] = slot;
                pps[slot].rpt_ind = at - rpt_bounds[g as usize];
                fill[g as usize] = at + 1;
            }
        }
        let widest = (0..group_count)
            .map(|g| rpt_bounds[g + 1] - rpt_bounds[g])
            .max()
            .unwrap_or(0);
        let slots = term_positions.len();
        let mut m = SloppyMatcher {
            pps,
            pq: Vec::with_capacity(slots),
            rpt_slots,
            rpt_bounds,
            requeue: vec![false; widest],
            rpt_stack: Vec::with_capacity(widest),
            has_rpts: repeats.has_rpts,
            has_multi_term_rpts: repeats.has_multi_term_rpts,
            end: i64::MIN,
            slop: i64::from(slop),
            positioned: false,
            match_length: i64::MAX,
        };
        m.positioned = m.init_phrase_positions();
        m
    }

    /// The slot indices of repeat group `g`, as a half-open range into
    /// [`Self::rpt_slots`].
    fn group_range(&self, g: usize) -> std::ops::Range<usize> {
        self.rpt_bounds[g]..self.rpt_bounds[g + 1]
    }

    /// `PhrasePositions.nextPosition()`.
    fn next_position(&mut self, i: usize) -> bool {
        let pp = &mut self.pps[i];
        match pp.positions.get(pp.idx) {
            Some(&raw) => {
                pp.position = i64::from(raw) - i64::from(pp.offset);
                pp.idx += 1;
                true
            }
            None => false,
        }
    }

    /// `PhrasePositions.firstPosition()`.
    fn first_position(&mut self, i: usize) {
        self.pps[i].idx = 0;
        self.next_position(i);
    }

    /// `advancePP` -- advance and update `end`.
    fn advance_pp(&mut self, i: usize) -> bool {
        if !self.next_position(i) {
            return false;
        }
        if self.pps[i].position > self.end {
            self.end = self.pps[i].position;
        }
        true
    }

    /// `tpPos(pp)`: the actual position in the document.
    fn tp_pos(&self, i: usize) -> i64 {
        self.pps[i]
            .position
            .saturating_add(i64::from(self.pps[i].offset))
    }

    /// `lesser(pp, pp2)`, compared by position then offset.
    fn lesser(&self, a: usize, b: usize) -> usize {
        let (pa, pb) = (&self.pps[a], &self.pps[b]);
        if pa.position < pb.position || (pa.position == pb.position && pa.offset < pb.offset) {
            a
        } else {
            b
        }
    }

    /// `collide(pp)`: the `rptInd` of another member of `pp`'s repeat group
    /// sitting on the same actual position, if any.
    fn collide(&self, i: usize) -> Option<usize> {
        let range = self.group_range(self.pps[i].rpt_group as usize);
        let tp = self.tp_pos(i);
        for at in range {
            let other = self.rpt_slots[at];
            if other != i && self.tp_pos(other) == tp {
                return Some(self.pps[other].rpt_ind);
            }
        }
        None
    }

    /// The slot at index `k` within `pp`'s repeat group -- Java's `rg[k]`.
    fn group_member(&self, pp: usize, k: usize) -> usize {
        self.rpt_slots[self.rpt_bounds[self.pps[pp].rpt_group as usize] + k]
    }

    fn pq_key(&self, i: usize) -> (i64, i32, usize) {
        (self.pps[i].position, self.pps[i].offset, self.pps[i].ord)
    }

    fn pq_add(&mut self, i: usize) {
        let key = self.pq_key(i);
        let at = self.pq.partition_point(|&j| self.pq_key(j) < key);
        self.pq.insert(at, i);
    }

    /// `pq.pop()`. The queue holds every slot bar the one `nextMatch` is
    /// currently advancing, and a phrase reaching here has at least two slots,
    /// so it is never empty at a call site that does not check.
    fn pq_pop(&mut self) -> usize {
        self.pq.remove(0)
    }

    /// `initPhrasePositions()`.
    fn init_phrase_positions(&mut self) -> bool {
        self.end = i64::MIN;
        if !self.has_rpts {
            self.init_simple();
            return true;
        }
        self.init_complex()
    }

    /// `initSimple()`.
    fn init_simple(&mut self) {
        self.pq.clear();
        for i in 0..self.pps.len() {
            self.first_position(i);
            if self.pps[i].position > self.end {
                self.end = self.pps[i].position;
            }
            self.pq_add(i);
        }
    }

    /// `initComplex()`.
    fn init_complex(&mut self) -> bool {
        for i in 0..self.pps.len() {
            self.first_position(i);
        }
        if !self.advance_repeat_groups() {
            return false;
        }
        self.fill_queue();
        true
    }

    /// `fillQueue()`.
    fn fill_queue(&mut self) {
        self.pq.clear();
        for i in 0..self.pps.len() {
            if self.pps[i].position > self.end {
                self.end = self.pps[i].position;
            }
            self.pq_add(i);
        }
    }

    /// `advanceRepeatGroups()`: spread each group out so it starts with no
    /// collisions, which is `nextMatch`'s precondition.
    fn advance_repeat_groups(&mut self) -> bool {
        for g in 0..self.rpt_bounds.len().saturating_sub(1) {
            let range = self.group_range(g);
            let len = range.len();
            if self.has_multi_term_rpts {
                // Some members may not collide, so the amount to advance is not
                // known up front.
                let mut i = 0usize;
                while i < len {
                    let mut incr = 1usize;
                    let pp = self.rpt_slots[range.start + i];
                    while let Some(k) = self.collide(pp) {
                        let pp2 = self.lesser(pp, self.rpt_slots[range.start + k]);
                        // At initialization this always advances the colliding
                        // slot with the higher offset.
                        if !self.advance_pp(pp2) {
                            return false;
                        }
                        if self.pps[pp2].rpt_ind < i {
                            // Java's "should not happen?" branch: re-run this
                            // group member rather than moving on. It terminates
                            // because `advance_pp` only ever moves forward
                            // through a finite list.
                            incr = 0;
                            break;
                        }
                    }
                    i = i.saturating_add(incr);
                }
            } else {
                // Known exactly how far to spread: the j-th member of the group
                // is advanced j times.
                for j in 1..len {
                    let slot = self.rpt_slots[range.start + j];
                    for _ in 0..j {
                        if !self.next_position(slot) {
                            return false;
                        }
                    }
                }
            }
        }
        true
    }

    /// `advanceRpts(pp)`: `pp` was just advanced; resolve the (at most one)
    /// collision that can have caused, and re-queue whatever had to move.
    fn advance_rpts(&mut self, pp: usize) -> bool {
        if self.pps[pp].rpt_group < 0 {
            return true;
        }
        let len = self.group_range(self.pps[pp].rpt_group as usize).len();
        self.requeue.clear();
        self.requeue.resize(len, false);
        let k0 = self.pps[pp].rpt_ind;
        let mut cur = pp;
        while let Some(k) = self.collide(cur) {
            cur = self.lesser(cur, self.group_member(cur, k));
            if !self.advance_pp(cur) {
                return false;
            }
            if k != k0 {
                // Mark only those currently in the queue: `pp` itself is not.
                self.requeue[k] = true;
            }
        }
        // Drain the queue until every slot that had to move has been seen, then
        // push them all back in the order Java's `rptStack` does.
        self.rpt_stack.clear();
        while self.requeue.iter().any(|&b| b) && !self.pq.is_empty() {
            let pp2 = self.pq_pop();
            self.rpt_stack.push(pp2);
            let ind = self.pps[pp2].rpt_ind;
            // Java tests `rptGroup >= 0` and the index alone, not group
            // membership, so a same-index member of a *different* group also
            // clears the bit. Kept as-is: changing it would change which
            // documents match.
            if self.pps[pp2].rpt_group >= 0 && ind < len && self.requeue[ind] {
                self.requeue[ind] = false;
            }
        }
        for at in (0..self.rpt_stack.len()).rev() {
            self.pq_add(self.rpt_stack[at]);
        }
        true
    }

    /// `nextMatch()`.
    fn next_match(&mut self) -> bool {
        if !self.positioned {
            return false;
        }
        let mut pp = self.pq_pop();
        self.match_length = self.end.saturating_sub(self.pps[pp].position);
        let mut next = self.pps[self.pq[0]].position;
        while self.advance_pp(pp) {
            if self.has_rpts && !self.advance_rpts(pp) {
                break; // pps exhausted
            }
            if self.pps[pp].position > next {
                // Done minimizing the current match length.
                self.pq_add(pp);
                if self.match_length <= self.slop {
                    return true;
                }
                pp = self.pq_pop();
                next = self.pps[self.pq[0]].position;
                self.match_length = self.end.saturating_sub(self.pps[pp].position);
            } else {
                let ml2 = self.end.saturating_sub(self.pps[pp].position);
                if ml2 < self.match_length {
                    self.match_length = ml2;
                }
            }
        }
        self.positioned = false;
        self.match_length <= self.slop
    }

    /// `sloppyWeight()`: `1f / (1f + matchLength)`.
    fn sloppy_weight(&self) -> f32 {
        1.0f32 / (1.0f32 + self.match_length as f32)
    }
}

/// Guards both entry points: Java reaches `SloppyPhraseMatcher` only through a
/// conjunction approximation over every slot's postings, so a slot with no
/// occurrence in this document never gets here, and a one-slot phrase is
/// rewritten to a `TermQuery` before a matcher is built.
enum Degenerate {
    /// Fewer than two slots, or a slot with no positions: handled without a
    /// matcher.
    Yes(f32),
    No,
}

fn degenerate(term_positions: &[&[i32]]) -> Degenerate {
    match term_positions.split_first() {
        None => Degenerate::Yes(0.0),
        Some((first, rest)) => {
            if rest.iter().any(|p| p.is_empty()) || first.is_empty() {
                Degenerate::Yes(0.0)
            } else if rest.is_empty() {
                // `PhraseQuery.Builder.build()` rewrites a one-term phrase to a
                // `TermQuery`, whose frequency is the term's own: every
                // occurrence is a zero-slack match.
                Degenerate::Yes(first.len() as f32)
            } else {
                Degenerate::No
            }
        }
    }
}

/// `SloppyPhraseMatcher`'s answer to "does this document match": Java's
/// `PhraseScorer.twoPhaseIterator().matches()`, which is `resetPositions()`
/// followed by one `nextMatch()`.
///
/// `term_positions[i]` is slot `i`'s sorted, ascending position list for one
/// document, in phrase order; `repeats` says which slots hold the same term
/// (use [`PhraseRepeats::none`] when they are pairwise distinct).
pub(crate) fn sloppy_phrase_matches(
    term_positions: &[&[i32]],
    repeats: &PhraseRepeats,
    slop: u32,
) -> bool {
    match degenerate(term_positions) {
        Degenerate::Yes(freq) => freq > 0.0,
        Degenerate::No => SloppyMatcher::new(term_positions, repeats, slop).next_match(),
    }
}

/// `PhraseScorer.score()`'s frequency for a sloppy phrase: the sum of
/// `sloppyWeight()` over the match sequence `nextMatch()` produces, starting
/// with the match `matches()` already found.
///
/// Returns `0.0` for a document that does not match, which is the same signal
/// [`sloppy_phrase_matches`] gives -- every match contributes a strictly
/// positive `1 / (1 + matchLength)`, and `matchLength` is never negative
/// (`end` is a maximum over the same positions the minimum is taken from).
pub(crate) fn sloppy_phrase_freq(
    term_positions: &[&[i32]],
    repeats: &PhraseRepeats,
    slop: u32,
) -> f32 {
    match degenerate(term_positions) {
        Degenerate::Yes(freq) => freq,
        Degenerate::No => {
            let mut m = SloppyMatcher::new(term_positions, repeats, slop);
            if !m.next_match() {
                return 0.0;
            }
            let mut freq = m.sloppy_weight();
            while m.next_match() {
                freq += m.sloppy_weight();
            }
            freq
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn terms(slots: &[&str]) -> Vec<Vec<Vec<u8>>> {
        slots
            .iter()
            .map(|s| vec![s.as_bytes().to_vec()])
            .collect::<Vec<_>>()
    }

    fn repeats_for(slots: &[&str]) -> PhraseRepeats {
        let owned = terms(slots);
        let refs: Vec<&[Vec<u8>]> = owned.iter().map(|s| s.as_slice()).collect();
        PhraseRepeats::detect(&refs)
    }

    fn matches(positions: &[&[i32]], slop: u32) -> bool {
        sloppy_phrase_matches(positions, &PhraseRepeats::none(positions.len()), slop)
    }

    fn freq(positions: &[&[i32]], slop: u32) -> f32 {
        sloppy_phrase_freq(positions, &PhraseRepeats::none(positions.len()), slop)
    }

    #[test]
    fn a_transposition_matches_at_slop_two_and_not_at_slop_one() {
        // The whole point of this module. Query "beta alpha" against a document
        // reading "alpha beta": beta is at 1, alpha at 0.
        assert!(!matches(&[&[1][..], &[0][..]], 1));
        assert!(matches(&[&[1][..], &[0][..]], 2));
        // Its match length is 2, so it weighs 1/3.
        assert_eq!(freq(&[&[1][..], &[0][..]], 2), 1.0 / 3.0);
    }

    #[test]
    fn an_in_order_gap_costs_the_gap_and_a_transposition_costs_twice_the_gap_plus_two() {
        // "alpha beta" over alpha@0 beta@3: two extra words, match length 2.
        assert!(!matches(&[&[0][..], &[3][..]], 1));
        assert!(matches(&[&[0][..], &[3][..]], 2));
        // The same pair queried the other way round: beta@3 alpha@0 shifts to
        // 3 and -1, a window of 4.
        assert!(!matches(&[&[3][..], &[0][..]], 3));
        assert!(matches(&[&[3][..], &[0][..]], 4));
    }

    #[test]
    fn adjacent_terms_match_at_slop_zero_and_weigh_one() {
        assert!(matches(&[&[0][..], &[1][..]], 0));
        assert_eq!(freq(&[&[0][..], &[1][..]], 0), 1.0);
        assert!(!matches(&[&[0][..], &[2][..]], 0));
    }

    #[test]
    fn a_three_term_phrase_uses_the_window_width_not_the_summed_gaps() {
        // a@0 b@5 c@1: shifted to 0, 4, -1 -- window 5. The in-order reading of
        // the same document could not match at any slop, since c precedes b.
        assert!(!matches(&[&[0][..], &[5][..], &[1][..]], 4));
        assert!(matches(&[&[0][..], &[5][..], &[1][..]], 5));
    }

    #[test]
    fn repeated_terms_may_not_share_a_position() {
        let rpt = repeats_for(&["a", "a"]);
        // One occurrence of `a` cannot satisfy both slots at any slop.
        assert!(!sloppy_phrase_matches(&[&[0][..], &[0][..]], &rpt, 100));
        // Two occurrences can.
        assert!(sloppy_phrase_matches(&[&[0, 1][..], &[0, 1][..]], &rpt, 0));
        assert_eq!(
            sloppy_phrase_freq(&[&[0, 1][..], &[0, 1][..]], &rpt, 0),
            1.0
        );
        // Without the repeat information the same input matches at slop 1,
        // which is exactly the wrong answer the grouping prevents.
        assert!(matches(&[&[0][..], &[0][..]], 1));
    }

    #[test]
    fn detect_finds_repeats_only_when_a_term_occurs_twice() {
        assert_eq!(repeats_for(&["a", "b"]), PhraseRepeats::none(2));
        let r = repeats_for(&["a", "b", "a"]);
        assert!(r.has_rpts);
        assert!(!r.has_multi_term_rpts);
        assert_eq!(r.groups, vec![0, -1, 0]);
        let r = repeats_for(&["a", "b", "a", "b"]);
        assert_eq!(r.groups, vec![0, 1, 0, 1]);
    }

    #[test]
    fn detect_unions_multi_term_slots_that_share_a_term() {
        // MultiPhraseQuery shape: slot 0 = {a, b}, slot 1 = {b, c}, slot 2 = {d}.
        // `b` repeats, so slots 0 and 1 are one group and slot 2 is in none.
        let owned: Vec<Vec<Vec<u8>>> = vec![
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"b".to_vec(), b"c".to_vec()],
            vec![b"d".to_vec()],
        ];
        let refs: Vec<&[Vec<u8>]> = owned.iter().map(|s| s.as_slice()).collect();
        let r = PhraseRepeats::detect(&refs);
        assert!(r.has_rpts);
        assert!(r.has_multi_term_rpts);
        assert_eq!(r.groups, vec![0, 0, -1]);
    }

    #[test]
    fn detect_unions_transitively_across_shared_terms() {
        // {a,b} {b,c} {c,d}: a chain, so all three end up in one group even
        // though slots 0 and 2 share no term directly.
        let owned: Vec<Vec<Vec<u8>>> = vec![
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"b".to_vec(), b"c".to_vec()],
            vec![b"c".to_vec(), b"d".to_vec()],
        ];
        let refs: Vec<&[Vec<u8>]> = owned.iter().map(|s| s.as_slice()).collect();
        let r = PhraseRepeats::detect(&refs);
        assert_eq!(r.groups, vec![0, 0, 0]);
    }

    #[test]
    fn multi_term_repeat_groups_still_keep_slots_off_one_position() {
        let owned: Vec<Vec<Vec<u8>>> = vec![
            vec![b"a".to_vec(), b"b".to_vec()],
            vec![b"b".to_vec(), b"c".to_vec()],
        ];
        let refs: Vec<&[Vec<u8>]> = owned.iter().map(|s| s.as_slice()).collect();
        let r = PhraseRepeats::detect(&refs);
        // A single shared position cannot satisfy both slots.
        assert!(!sloppy_phrase_matches(&[&[4][..], &[4][..]], &r, 50));
        // Two positions can.
        assert!(sloppy_phrase_matches(&[&[4, 5][..], &[4, 5][..]], &r, 1));
    }

    #[test]
    fn a_collision_created_mid_walk_is_resolved_and_the_queue_re_ordered() {
        // The `advanceRpts` path: with three occurrences of one term and a
        // two-slot repeat group, advancing the leading slot lands it on the
        // trailing slot's raw position, which Java resolves by advancing the
        // *lesser* of the two and re-queuing whatever moved. Nothing else in
        // this module's tests reaches it, because it needs a repeat group whose
        // members can still collide after initialization spread them apart.
        let rpt = repeats_for(&["a", "a"]);
        let three = [&[0i32, 1, 2][..], &[0i32, 1, 2][..]];
        assert!(sloppy_phrase_matches(&three, &rpt, 0));
        // At slop 0 the answer must be the exact matcher's: "a a" occurs at
        // (0,1) and (1,2), so twice, each weighing 1.
        assert_eq!(sloppy_phrase_freq(&three, &rpt, 0), 2.0);
        // Four occurrences, three matches.
        let four = [&[0i32, 1, 2, 3][..], &[0i32, 1, 2, 3][..]];
        assert_eq!(sloppy_phrase_freq(&four, &rpt, 0), 3.0);
        // A gap the group has to step over: occurrences at 0 and 5 are four
        // apart, so `"a a"` needs slop 4 and weighs 1/5.
        let apart = [&[0i32, 5][..], &[0i32, 5][..]];
        assert!(!sloppy_phrase_matches(&apart, &rpt, 3));
        assert!(sloppy_phrase_matches(&apart, &rpt, 4));
        assert_eq!(sloppy_phrase_freq(&apart, &rpt, 4), 1.0 / 5.0);
    }

    #[test]
    fn the_walk_shrinks_a_match_before_re_queuing() {
        // `nextMatch`'s "keep advancing the same pp while it is still below the
        // queue's next element" refinement: the first window found for a start
        // is not necessarily the tightest, and Lucene reports the tightest.
        // a@[0, 9], b@[7]: shifted to a=[0, 9], b=[6]. Popping a@0 gives
        // `end - 0 == 6`; advancing a to 9 is still not past the queue's next
        // element's position, and shrinks the window to `9 - 6 == 3`.
        let got = freq(&[&[0i32, 9][..], &[7i32][..]], 3);
        assert_eq!(got, 1.0 / 4.0, "the reported match length must be 3, not 6");
        assert!(!matches(&[&[0i32, 9][..], &[7i32][..]], 2));
    }

    #[test]
    fn a_repeat_group_that_runs_out_of_positions_does_not_match() {
        let rpt = repeats_for(&["a", "a", "a"]);
        // Only two occurrences for three slots.
        assert!(!sloppy_phrase_matches(
            &[&[0, 1][..], &[0, 1][..], &[0, 1][..]],
            &rpt,
            100
        ));
        assert_eq!(
            sloppy_phrase_freq(&[&[0, 1][..], &[0, 1][..], &[0, 1][..]], &rpt, 100),
            0.0
        );
    }

    #[test]
    fn degenerate_shapes_match_nothing_or_the_term_frequency() {
        assert!(!matches(&[], 5));
        assert!(!matches(&[&[][..]], 5));
        assert!(!matches(&[&[0][..], &[][..]], 5));
        assert_eq!(freq(&[], 5), 0.0);
        assert_eq!(freq(&[&[][..], &[1][..]], 5), 0.0);
        // A one-slot phrase degenerates to the term's own frequency.
        assert!(matches(&[&[2, 9][..]], 0));
        assert_eq!(freq(&[&[2, 9][..]], 0), 2.0);
    }

    #[test]
    fn several_occurrences_sum_their_weights() {
        // "a b" over a@[0,4] b@[1,7]: an adjacent pair (weight 1) and a pair
        // three apart (weight 1/3).
        let got = freq(&[&[0, 4][..], &[1, 7][..]], 3);
        assert_eq!(got, 1.0 + 1.0 / 3.0);
    }

    #[test]
    fn a_match_beyond_the_budget_contributes_nothing() {
        // a@0 b@6: window 5, over a budget of 2.
        assert_eq!(freq(&[&[0][..], &[6][..]], 2), 0.0);
    }

    #[test]
    fn slop_zero_agrees_with_exact_adjacency_on_every_small_case() {
        // Cross-check against the exact matcher's own rule for every pair of
        // three-position lists, which is the invariant that lets
        // `search_phrase_query` keep using the exact fast path at slop 0.
        for a in 0..8i32 {
            for b in 0..8i32 {
                let exact = b == a + 1;
                assert_eq!(matches(&[&[a][..], &[b][..]], 0), exact, "a={a} b={b}");
            }
        }
    }
}
