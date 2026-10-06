//! `IntervalIterator` and every implementation of it in
//! `org.apache.lucene.queries.intervals`, with the queues they run on
//! (`util.PriorityQueue`, the package's own `DisiPriorityQueue` and
//! `DisjunctionDISIApproximation`, and `ConjunctionDISI`).
//!
//! Each iterator is its Java class's state machine, method for method. Java
//! shares a sub-iterator between a parent and the parent's subclass fields
//! (`FilteringIntervalIterator`'s `a` and `b` are also its
//! `subIterators`); here a parent owns its sub-iterators and names them by
//! index. Java's `int` arithmetic wraps; so does this, explicitly.

use std::fmt;

use super::{IntervalFilterKind, IntervalsSource, PayloadFilter, NO_MORE_INTERVALS};
use crate::exec::span::LeafPositions;
use crate::exec::{LeafContext, NO_MORE_DOCS};
use crate::{Error, Result};

/// `IntervalIterator`: a `DocIdSetIterator` that, on each document, walks the
/// minimal intervals of its source.
pub(crate) trait IntervalIterator {
    /// `docID()`.
    fn doc_id(&self) -> i32;
    /// `nextDoc()`.
    fn next_doc(&mut self) -> Result<i32>;
    /// `advance(target)`.
    fn advance(&mut self, target: i32) -> Result<i32>;
    /// `cost()`.
    fn cost(&self) -> i64;
    /// `start()`: `-1` before the first interval, [`NO_MORE_INTERVALS`]
    /// after the last.
    fn start(&self) -> i32;
    /// `end()`.
    fn end(&self) -> i32;
    /// `gaps()`.
    fn gaps(&self) -> i32;
    /// `width()`: `end() - start() + 1`.
    fn width(&self) -> i32 {
        self.end().wrapping_sub(self.start()).wrapping_add(1)
    }
    /// `nextInterval()`: the next interval's start, or
    /// [`NO_MORE_INTERVALS`].
    fn next_interval(&mut self) -> Result<i32>;
    /// `matchCost()`.
    fn match_cost(&self) -> f32;
    /// `IntervalScorer.ensureFreq()`'s sum from the current interval on:
    /// `freq += 1.0 / max(length - minExtent + 1, 1)` per interval, in
    /// `double`, to the `float` sum. A method so that it is monomorphised
    /// per iterator: one virtual call per document, not three per interval.
    fn sum_freq(&mut self, min_extent: i32) -> Result<f32> {
        let mut freq = 0.0f32;
        loop {
            let length = self.end().wrapping_sub(self.start()).wrapping_add(1);
            let d = length.wrapping_sub(min_extent).wrapping_add(1).max(1);
            freq = (f64::from(freq) + 1.0 / f64::from(d)) as f32;
            if self.next_interval()? == NO_MORE_INTERVALS {
                return Ok(freq);
            }
        }
    }
}

impl fmt::Debug for dyn IntervalIterator + '_ {
    /// `IntervalIterator.toString`: `docID:[start->end]`.
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:[{}->{}]", self.doc_id(), self.start(), self.end())
    }
}

/// An interval iterator as its parent holds it: a term's inline, so the
/// leaf every source bottoms out in is called statically (and inlined)
/// rather than through a virtual call per document and per interval -- the
/// dispatch Java's JIT removes by inlining monomorphic call sites; any
/// other iterator boxed.
// The term held inline, not boxed, is the point: no pointer to chase on
// the hottest calls. Parents hold a few of these, never many.
#[allow(clippy::large_enum_variant)]
pub(crate) enum BoxIntervals<'a> {
    Term(TermIntervals<'a>),
    Dyn(Box<dyn IntervalIterator + 'a>),
}

impl<'a> BoxIntervals<'a> {
    /// Boxes `it` behind a virtual call.
    pub(crate) fn boxed(it: impl IntervalIterator + 'a) -> Self {
        BoxIntervals::Dyn(Box::new(it))
    }
}

impl fmt::Debug for BoxIntervals<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}:[{}->{}]", self.doc_id(), self.start(), self.end())
    }
}

/// Forwards to the variant, statically for a term.
macro_rules! dispatch {
    ($self:expr, $s:ident => $e:expr) => {
        match $self {
            BoxIntervals::Term($s) => $e,
            BoxIntervals::Dyn($s) => $e,
        }
    };
}

impl IntervalIterator for BoxIntervals<'_> {
    #[inline]
    fn doc_id(&self) -> i32 {
        dispatch!(self, s => s.doc_id())
    }
    #[inline]
    fn next_doc(&mut self) -> Result<i32> {
        dispatch!(self, s => s.next_doc())
    }
    #[inline]
    fn advance(&mut self, target: i32) -> Result<i32> {
        dispatch!(self, s => s.advance(target))
    }
    #[inline]
    fn cost(&self) -> i64 {
        dispatch!(self, s => s.cost())
    }
    #[inline]
    fn start(&self) -> i32 {
        dispatch!(self, s => s.start())
    }
    #[inline]
    fn end(&self) -> i32 {
        dispatch!(self, s => s.end())
    }
    #[inline]
    fn gaps(&self) -> i32 {
        dispatch!(self, s => s.gaps())
    }
    #[inline]
    fn width(&self) -> i32 {
        dispatch!(self, s => s.width())
    }
    #[inline]
    fn next_interval(&mut self) -> Result<i32> {
        dispatch!(self, s => s.next_interval())
    }
    #[inline]
    fn match_cost(&self) -> f32 {
        dispatch!(self, s => s.match_cost())
    }
    #[inline]
    fn sum_freq(&mut self, min_extent: i32) -> Result<f32> {
        dispatch!(self, s => s.sum_freq(min_extent))
    }
}

/// `MinimizingConjunctionIntervalsSource.MatchCallback`: run each time a
/// minimizing iterator settles on a match (the matches path caches its
/// sub-matches there).
pub(crate) type MatchCallback<'a> = Option<Box<dyn FnMut() -> Result<()> + 'a>>;

fn on_match(cb: &mut MatchCallback<'_>) -> Result<()> {
    match cb {
        Some(f) => f(),
        None => Ok(()),
    }
}

// ---------------------------------------------------------------------------
// util.PriorityQueue
// ---------------------------------------------------------------------------

/// `org.apache.lucene.util.PriorityQueue` over sub-iterator indices, its
/// `lessThan` passed to each call: the same 1-based heap, so equal elements
/// settle exactly where Java's do.
#[derive(Debug, Default)]
pub(crate) struct IndexQueue {
    heap: Vec<usize>,
    size: usize,
}

impl IndexQueue {
    pub(crate) fn new(max_size: usize) -> Self {
        IndexQueue {
            heap: vec![0; max_size.saturating_add(1).max(2)],
            size: 0,
        }
    }

    pub(crate) fn size(&self) -> usize {
        self.size
    }

    pub(crate) fn top(&self) -> Option<usize> {
        (self.size > 0).then(|| self.heap[1])
    }

    pub(crate) fn clear(&mut self) {
        self.size = 0;
    }

    /// The elements in heap order (`iterator()`).
    pub(crate) fn iter(&self) -> impl Iterator<Item = usize> + '_ {
        self.heap[1..=self.size].iter().copied()
    }

    pub(crate) fn add(&mut self, e: usize, less: &dyn Fn(usize, usize) -> bool) {
        let index = self.size.saturating_add(1);
        if index >= self.heap.len() {
            self.heap.resize(index.saturating_add(1), 0);
        }
        self.heap[index] = e;
        self.size = index;
        self.up_heap(index, less);
    }

    /// `updateTop()`: the top changed in place; re-sift it and return the
    /// new top.
    pub(crate) fn update_top(&mut self, less: &dyn Fn(usize, usize) -> bool) -> Option<usize> {
        self.down_heap(1, less);
        self.top()
    }

    pub(crate) fn pop(&mut self, less: &dyn Fn(usize, usize) -> bool) -> Option<usize> {
        if self.size == 0 {
            return None;
        }
        let result = self.heap[1];
        self.heap[1] = self.heap[self.size];
        self.size -= 1;
        self.down_heap(1, less);
        Some(result)
    }

    fn up_heap(&mut self, orig: usize, less: &dyn Fn(usize, usize) -> bool) {
        let mut i = orig;
        let node = self.heap[i];
        let mut j = i >> 1;
        while j > 0 && less(node, self.heap[j]) {
            self.heap[i] = self.heap[j];
            i = j;
            j >>= 1;
        }
        self.heap[i] = node;
    }

    fn down_heap(&mut self, mut i: usize, less: &dyn Fn(usize, usize) -> bool) {
        if self.size == 0 {
            return;
        }
        let node = self.heap[i];
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= self.size && less(self.heap[k], self.heap[j]) {
            j = k;
        }
        while j <= self.size && less(self.heap[j], node) {
            self.heap[i] = self.heap[j];
            i = j;
            j = i << 1;
            k = j + 1;
            if k <= self.size && less(self.heap[k], self.heap[j]) {
                j = k;
            }
        }
        self.heap[i] = node;
    }
}

// ---------------------------------------------------------------------------
// DisiPriorityQueue / DisjunctionDISIApproximation
// ---------------------------------------------------------------------------

/// The package's `DisiPriorityQueue` of `DisiWrapper`s, ordered by each
/// sub-iterator's current document.
#[derive(Debug)]
pub(crate) struct DisiQueue {
    /// `DisiWrapper.doc`, per sub-iterator.
    docs: Vec<i32>,
    /// `DisiWrapper.next`, per sub-iterator (`topList`'s links).
    next: Vec<Option<usize>>,
    heap: Vec<usize>,
    size: usize,
    /// `DisjunctionDISIApproximation.cost`.
    cost: i64,
    /// [`Self::top_list`]'s answer, kept to be refilled.
    list: Vec<usize>,
}

fn left_node(node: usize) -> usize {
    ((node + 1) << 1) - 1
}

fn parent_node(node: usize) -> Option<usize> {
    ((node + 1) >> 1).checked_sub(1)
}

impl DisiQueue {
    /// Every sub-iterator added in order, unpositioned (`doc == -1`).
    pub(crate) fn new(subs: &[BoxIntervals<'_>]) -> Self {
        let mut q = DisiQueue {
            docs: vec![-1; subs.len()],
            next: vec![None; subs.len()],
            heap: Vec::with_capacity(subs.len()),
            size: 0,
            cost: 0,
            list: Vec::with_capacity(subs.len()),
        };
        for (i, s) in subs.iter().enumerate() {
            q.cost = q.cost.wrapping_add(s.cost());
            q.heap.push(i);
            q.up_heap(i);
            q.size += 1;
        }
        q
    }

    fn top(&self) -> usize {
        self.heap[0]
    }

    pub(crate) fn top_doc(&self) -> i32 {
        self.heap.first().map_or(NO_MORE_DOCS, |&t| self.docs[t])
    }

    /// `topList()`: the sub-iterators on the top document, in the order
    /// Java's linked list holds them.
    pub(crate) fn top_list(&mut self) -> &[usize] {
        self.list.clear();
        if self.size == 0 {
            return &self.list;
        }
        let mut list = self.heap[0];
        self.next[list] = None;
        if self.size >= 3 {
            list = self.top_list_from(list, 1);
            list = self.top_list_from(list, 2);
        } else if self.size == 2 && self.docs[self.heap[1]] == self.docs[list] {
            let w = self.heap[1];
            self.next[w] = Some(list);
            list = w;
        }
        let mut at = Some(list);
        while let Some(w) = at {
            self.list.push(w);
            at = self.next[w];
        }
        &self.list
    }

    fn top_list_from(&mut self, mut list: usize, i: usize) -> usize {
        let w = self.heap[i];
        if self.docs[w] == self.docs[list] {
            self.next[w] = Some(list);
            list = w;
            let left = left_node(i);
            let right = left + 1;
            if right < self.size {
                list = self.top_list_from(list, left);
                list = self.top_list_from(list, right);
            } else if left < self.size && self.docs[self.heap[left]] == self.docs[list] {
                let l = self.heap[left];
                self.next[l] = Some(list);
                list = l;
            }
        }
        list
    }

    fn up_heap(&mut self, mut i: usize) {
        let node = self.heap[i];
        let node_doc = self.docs[node];
        let mut j = parent_node(i);
        while let Some(p) = j {
            if node_doc < self.docs[self.heap[p]] {
                self.heap[i] = self.heap[p];
                i = p;
                j = parent_node(p);
            } else {
                break;
            }
        }
        self.heap[i] = node;
    }

    fn down_heap(&mut self) {
        let size = self.size;
        let mut i = 0usize;
        let node = self.heap[0];
        let mut j = left_node(i);
        if j < size {
            let mut k = j + 1;
            if k < size && self.docs[self.heap[k]] < self.docs[self.heap[j]] {
                j = k;
            }
            if self.docs[self.heap[j]] < self.docs[node] {
                loop {
                    self.heap[i] = self.heap[j];
                    i = j;
                    j = left_node(i);
                    k = j + 1;
                    if k < size && self.docs[self.heap[k]] < self.docs[self.heap[j]] {
                        j = k;
                    }
                    if !(j < size && self.docs[self.heap[j]] < self.docs[node]) {
                        break;
                    }
                }
                self.heap[i] = node;
            }
        }
    }

    /// `DisjunctionDISIApproximation.nextDoc()`.
    pub(crate) fn next_doc(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<i32> {
        if self.size == 0 {
            return Ok(NO_MORE_DOCS);
        }
        let mut top = self.top();
        let doc = self.docs[top];
        loop {
            self.docs[top] = subs[top].next_doc()?;
            self.down_heap();
            top = self.top();
            if self.docs[top] != doc {
                return Ok(self.docs[top]);
            }
        }
    }

    /// `DisjunctionDISIApproximation.advance(target)`.
    pub(crate) fn advance(&mut self, subs: &mut [BoxIntervals<'_>], target: i32) -> Result<i32> {
        if self.size == 0 {
            return Ok(NO_MORE_DOCS);
        }
        let mut top = self.top();
        loop {
            self.docs[top] = subs[top].advance(target)?;
            self.down_heap();
            top = self.top();
            if self.docs[top] >= target {
                return Ok(self.docs[top]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// ConjunctionDISI
// ---------------------------------------------------------------------------

/// `ConjunctionUtils.intersectIterators` over interval iterators: a
/// `ConjunctionDISI`, its sub-iterators sorted by cost (stably, as
/// `timSort`), led by the cheapest.
#[derive(Debug)]
pub(crate) struct Conjunction {
    lead1: usize,
    lead2: usize,
    others: Vec<usize>,
}

impl Conjunction {
    pub(crate) fn new(subs: &[BoxIntervals<'_>]) -> Self {
        let mut order: Vec<usize> = (0..subs.len()).collect();
        order.sort_by_key(|&i| subs[i].cost());
        Conjunction {
            lead1: order[0],
            lead2: order.get(1).copied().unwrap_or(order[0]),
            others: order.iter().skip(2).copied().collect(),
        }
    }

    pub(crate) fn doc_id(&self, subs: &[BoxIntervals<'_>]) -> i32 {
        subs[self.lead1].doc_id()
    }

    pub(crate) fn cost(&self, subs: &[BoxIntervals<'_>]) -> i64 {
        subs[self.lead1].cost()
    }

    pub(crate) fn next_doc(&self, subs: &mut [BoxIntervals<'_>]) -> Result<i32> {
        let doc = subs[self.lead1].next_doc()?;
        self.do_next(subs, doc)
    }

    pub(crate) fn advance(&self, subs: &mut [BoxIntervals<'_>], target: i32) -> Result<i32> {
        let doc = subs[self.lead1].advance(target)?;
        self.do_next(subs, doc)
    }

    fn do_next(&self, subs: &mut [BoxIntervals<'_>], mut doc: i32) -> Result<i32> {
        'advance_head: loop {
            let next2 = subs[self.lead2].advance(doc)?;
            if next2 != doc {
                doc = subs[self.lead1].advance(next2)?;
                if next2 != doc {
                    continue;
                }
            }
            for &other in &self.others {
                if subs[other].doc_id() < doc {
                    let next = subs[other].advance(doc)?;
                    if next > doc {
                        doc = subs[self.lead1].advance(next)?;
                        continue 'advance_head;
                    }
                }
            }
            return Ok(doc);
        }
    }
}

/// `ConjunctionIntervalIterator`'s `matchCost`: the sub-iterators' summed.
fn summed_match_cost(subs: &[BoxIntervals<'_>]) -> f32 {
    subs.iter().fold(0.0f32, |acc, s| acc + s.match_cost())
}

// ---------------------------------------------------------------------------
// TermIntervalsSource / PayloadFilteredTermIntervalsSource
// ---------------------------------------------------------------------------

/// `TermIntervalsSource.TERM_POSNS_SEEK_OPS_PER_DOC`.
const TERM_POSNS_SEEK_OPS_PER_DOC: f32 = 256.0;
/// `TermIntervalsSource.TERM_OPS_PER_POS`.
const TERM_OPS_PER_POS: f32 = 7.0;

/// `TermIntervalsSource.termPositionsCost`.
pub(crate) fn term_positions_cost(doc_freq: i32, total_term_freq: i64) -> f32 {
    let exp_occurrences = total_term_freq as f32 / doc_freq as f32;
    TERM_POSNS_SEEK_OPS_PER_DOC + exp_occurrences * TERM_OPS_PER_POS
}

/// The payload half of a payload-filtered term: where to read a document's
/// occurrences, payloads included, and what to accept.
struct PayloadSide<'a> {
    filter: PayloadFilter,
    field_terms: &'a lucene_codecs::blocktree::FieldTerms,
    term: Vec<u8>,
    ctx: LeafContext<'a>,
    occurrences: Vec<lucene_codecs::postings::Position>,
}

/// `TermIntervalsSource.intervals`' and
/// `PayloadFilteredTermIntervalsSource.intervals`' anonymous iterators: one
/// interval per position, `[p, p]`.
pub(crate) struct TermIntervals<'a> {
    postings: LeafPositions<'a>,
    doc: i32,
    upto: i32,
    pos: i32,
    /// The document's positions, read on its first `nextInterval`.
    positions: Vec<i32>,
    loaded: bool,
    at: usize,
    cost: i64,
    match_cost: f32,
    payloads: Option<PayloadSide<'a>>,
    /// Whether the cursor reads each position's payload as it goes
    /// (`PostingsEnum.PAYLOADS`), rather than the document's occurrences
    /// being read whole on its first interval.
    stream: bool,
    /// Whether positions come one at a time off a lazy cursor
    /// (`nextPosition()`), rather than a pulsed singleton's decoded list.
    lazy: bool,
}

impl<'a> TermIntervals<'a> {
    fn reset(&mut self) {
        if self.doc == NO_MORE_DOCS {
            self.upto = -1;
            self.pos = NO_MORE_INTERVALS;
        } else {
            self.upto = i32::try_from(self.postings.freq_at(self.doc)).unwrap_or(i32::MAX);
            self.pos = -1;
        }
        self.loaded = false;
        self.at = 0;
    }

    fn load(&mut self) -> Result<()> {
        if self.loaded {
            return Ok(());
        }
        self.loaded = true;
        self.at = 0;
        match &mut self.payloads {
            None => {
                self.positions.clear();
                self.postings.positions_at(self.doc, &mut self.positions)?;
            }
            Some(p) => {
                self.postings.occurrences_at(
                    &p.ctx,
                    p.field_terms,
                    &p.term,
                    self.doc,
                    &mut p.occurrences,
                )?;
            }
        }
        Ok(())
    }
}

impl IntervalIterator for TermIntervals<'_> {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.doc = self.postings.next_doc(self.doc)?;
        self.reset();
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = self.postings.advance(target)?;
        self.reset();
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.cost
    }
    fn start(&self) -> i32 {
        self.pos
    }
    fn end(&self) -> i32 {
        self.pos
    }
    fn gaps(&self) -> i32 {
        0
    }
    fn next_interval(&mut self) -> Result<i32> {
        loop {
            if self.upto <= 0 {
                self.pos = NO_MORE_INTERVALS;
                return Ok(self.pos);
            }
            self.upto -= 1;
            if self.lazy && self.payloads.is_none() {
                self.pos = self.postings.next_position()?;
                return Ok(self.pos);
            }
            if self.stream {
                let (position, payload) = self.postings.next_position_with_payload()?;
                self.pos = position;
                if self
                    .payloads
                    .as_ref()
                    .is_some_and(|p| p.filter.test(payload))
                {
                    return Ok(position);
                }
                continue;
            }
            self.load()?;
            match &self.payloads {
                None => {
                    self.pos = self.positions.get(self.at).copied().unwrap_or(-1);
                    self.at += 1;
                    return Ok(self.pos);
                }
                Some(p) => {
                    let occ = p.occurrences.get(self.at);
                    self.at += 1;
                    self.pos = occ.map_or(-1, |o| o.position);
                    let payload = occ.map(|o| o.payload.as_slice()).filter(|b| !b.is_empty());
                    if p.filter.test(payload) {
                        return Ok(self.pos);
                    }
                }
            }
        }
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
}

/// The field's terms, checked as `TermIntervalsSource.intervals` checks
/// them: `None` without the field, an error without positions.
fn positions_field<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
) -> Result<Option<&'a lucene_codecs::blocktree::FieldTerms>> {
    let Some(ft) = ctx.fields.field(field) else {
        return Ok(None);
    };
    if !ft.index_options().subsumes_positions() {
        return Err(Error::IllegalArgument(format!(
            "Cannot create an IntervalIterator over field {field} because it has no indexed positions"
        )));
    }
    Ok(Some(ft))
}

/// `TermIntervalsSource.intervals(term, te)`, the term already known to be
/// in the field.
fn term_iterator<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
    term: &[u8],
    stats: lucene_codecs::blocktree::TermStats,
    payloads: Option<(PayloadFilter, &'a lucene_codecs::blocktree::FieldTerms)>,
) -> Result<BoxIntervals<'a>> {
    let Some(pos_in) = ctx.pos_in else {
        return Err(Error::MissingPosInput);
    };
    let mut postings = LeafPositions::open(ctx, pos_in, field, term)?;
    let stream = payloads.is_some() && postings.stream_payloads(ctx);
    let lazy = postings.is_lazy();
    Ok(BoxIntervals::Term(TermIntervals {
        postings,
        doc: -1,
        upto: 0,
        pos: -1,
        positions: Vec::new(),
        loaded: false,
        at: 0,
        cost: i64::from(stats.doc_freq),
        match_cost: term_positions_cost(stats.doc_freq, stats.total_term_freq),
        payloads: payloads.map(|(filter, field_terms)| PayloadSide {
            filter,
            field_terms,
            term: term.to_vec(),
            ctx: *ctx,
            occurrences: Vec::new(),
        }),
        stream,
        lazy,
    }))
}

// ---------------------------------------------------------------------------
// DisjunctionIntervalsSource.DisjunctionIntervalIterator
// ---------------------------------------------------------------------------

/// Which iterator a disjunction is on: `EMPTY`, `EXHAUSTED` or a sub.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Current {
    Empty,
    Exhausted,
    Sub(usize),
}

/// `DisjunctionIntervalsSource.DisjunctionIntervalIterator`.
pub(crate) struct DisjunctionIntervals<'a> {
    pub(crate) subs: Vec<BoxIntervals<'a>>,
    disi: DisiQueue,
    queue: IndexQueue,
    current: Current,
    match_cost: f32,
}

/// The interval queue's `lessThan`: by end, then the wider first.
fn end_then_wider<'s>(subs: &'s [BoxIntervals<'_>]) -> impl Fn(usize, usize) -> bool + 's {
    move |a, b| {
        let (ea, eb) = (subs[a].end(), subs[b].end());
        ea < eb || (ea == eb && subs[a].start() >= subs[b].start())
    }
}

/// The proximity queue's `lessThan`: by start, then the wider first.
fn start_then_wider<'s>(subs: &'s [BoxIntervals<'_>]) -> impl Fn(usize, usize) -> bool + 's {
    move |a, b| {
        let (sa, sb) = (subs[a].start(), subs[b].start());
        sa < sb || (sa == sb && subs[a].end() >= subs[b].end())
    }
}

impl<'a> DisjunctionIntervals<'a> {
    pub(crate) fn new(subs: Vec<BoxIntervals<'a>>) -> Self {
        let disi = DisiQueue::new(&subs);
        // `costsum += it.cost()`: the documents, not the match costs.
        let match_cost = subs.iter().fold(0.0f32, |acc, s| acc + s.cost() as f32);
        let n = subs.len();
        DisjunctionIntervals {
            subs,
            disi,
            queue: IndexQueue::new(n),
            current: Current::Empty,
            match_cost,
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.queue.clear();
        for &w in self.disi.top_list() {
            self.subs[w].next_interval()?;
            self.queue.add(w, &end_then_wider(&self.subs));
        }
        self.current = Current::Empty;
        Ok(())
    }

    /// `currentOrd()`: the index of the sub-iterator the current interval
    /// is from.
    pub(crate) fn current_ord(&self) -> Option<usize> {
        match self.current {
            Current::Sub(i) => Some(i),
            _ => None,
        }
    }
}

impl IntervalIterator for DisjunctionIntervals<'_> {
    fn doc_id(&self) -> i32 {
        self.disi.top_doc()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let doc = self.disi.next_doc(&mut self.subs)?;
        self.reset()?;
        Ok(doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.disi.advance(&mut self.subs, target)?;
        self.reset()?;
        Ok(doc)
    }
    fn cost(&self) -> i64 {
        self.disi.cost
    }
    // SENTINEL: `-1` = "not on an interval yet", `IntervalIterator`'s own
    // contract; callers compare it as a position, below every real one, as
    // Java's do.
    fn start(&self) -> i32 {
        match self.current {
            Current::Empty => -1,
            Current::Exhausted => NO_MORE_INTERVALS,
            Current::Sub(i) => self.subs[i].start(),
        }
    }
    // SENTINEL: `-1` = "not on an interval yet", `IntervalIterator`'s own
    // contract; callers compare it as a position, below every real one, as
    // Java's do.
    fn end(&self) -> i32 {
        match self.current {
            Current::Empty => -1,
            Current::Exhausted => NO_MORE_INTERVALS,
            Current::Sub(i) => self.subs[i].end(),
        }
    }
    fn gaps(&self) -> i32 {
        match self.current {
            Current::Sub(i) => self.subs[i].gaps(),
            // `EMPTY.gaps()`/`EXHAUSTED.gaps()` throw; never asked.
            _ => 0,
        }
    }
    fn next_interval(&mut self) -> Result<i32> {
        if matches!(self.current, Current::Empty | Current::Exhausted) {
            if let Some(top) = self.queue.top() {
                self.current = Current::Sub(top);
            }
            return Ok(self.start());
        }
        let (start, end) = (self.start(), self.end());
        while let Some(top) = self.queue.top() {
            let it = &self.subs[top];
            let contains =
                start >= it.start() && start <= it.end() && end >= it.start() && end <= it.end();
            if !contains {
                break;
            }
            let popped = self.queue.pop(&end_then_wider(&self.subs));
            if let Some(p) = popped {
                if self.subs[p].next_interval()? != NO_MORE_INTERVALS {
                    self.queue.add(p, &end_then_wider(&self.subs));
                }
            }
        }
        match self.queue.top() {
            None => {
                self.current = Current::Exhausted;
                Ok(NO_MORE_INTERVALS)
            }
            Some(top) => {
                self.current = Current::Sub(top);
                Ok(self.subs[top].start())
            }
        }
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
}

// ---------------------------------------------------------------------------
// The conjunctions: BLOCK, ORDERED, UNORDERED, and the filtering pairs
// ---------------------------------------------------------------------------

/// `ConjunctionIntervalIterator`'s state, shared by every conjunction.
struct ConjunctionState<'a> {
    subs: Vec<BoxIntervals<'a>>,
    approx: Conjunction,
    match_cost: f32,
}

impl<'a> ConjunctionState<'a> {
    fn new(subs: Vec<BoxIntervals<'a>>) -> Self {
        let approx = Conjunction::new(&subs);
        let match_cost = summed_match_cost(&subs);
        ConjunctionState {
            subs,
            approx,
            match_cost,
        }
    }
}

/// The per-class half of a `ConjunctionIntervalIterator`.
trait ConjunctionKind {
    /// `reset()`, called on each new document (not on `NO_MORE_DOCS`).
    fn reset(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<()>;
    fn start(&self, subs: &[BoxIntervals<'_>]) -> i32;
    fn end(&self, subs: &[BoxIntervals<'_>]) -> i32;
    fn gaps(&self, subs: &[BoxIntervals<'_>]) -> i32;
    fn width(&self, subs: &[BoxIntervals<'_>]) -> i32 {
        self.end(subs)
            .wrapping_sub(self.start(subs))
            .wrapping_add(1)
    }
    fn next_interval(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<i32>;
}

/// A `ConjunctionIntervalIterator` subclass: the shared state and the
/// class's own.
pub(crate) struct ConjunctionIntervals<'a, K> {
    state: ConjunctionState<'a>,
    kind: K,
}

impl<K: ConjunctionKind> IntervalIterator for ConjunctionIntervals<'_, K> {
    fn doc_id(&self) -> i32 {
        self.state.approx.doc_id(&self.state.subs)
    }
    fn next_doc(&mut self) -> Result<i32> {
        let doc = self.state.approx.next_doc(&mut self.state.subs)?;
        if doc != NO_MORE_DOCS {
            self.kind.reset(&mut self.state.subs)?;
        }
        Ok(doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.state.approx.advance(&mut self.state.subs, target)?;
        if doc != NO_MORE_DOCS {
            self.kind.reset(&mut self.state.subs)?;
        }
        Ok(doc)
    }
    fn cost(&self) -> i64 {
        self.state.approx.cost(&self.state.subs)
    }
    fn start(&self) -> i32 {
        self.kind.start(&self.state.subs)
    }
    fn end(&self) -> i32 {
        self.kind.end(&self.state.subs)
    }
    fn gaps(&self) -> i32 {
        self.kind.gaps(&self.state.subs)
    }
    fn width(&self) -> i32 {
        self.kind.width(&self.state.subs)
    }
    fn next_interval(&mut self) -> Result<i32> {
        self.kind.next_interval(&mut self.state.subs)
    }
    fn match_cost(&self) -> f32 {
        self.state.match_cost
    }
}

fn conjunction<'a, K: ConjunctionKind + 'a>(
    subs: Vec<BoxIntervals<'a>>,
    kind: K,
) -> BoxIntervals<'a> {
    BoxIntervals::boxed(ConjunctionIntervals {
        state: ConjunctionState::new(subs),
        kind,
    })
}

/// `BlockIntervalsSource.BlockIntervalIterator`.
struct Block {
    start: i32,
    end: i32,
}

impl ConjunctionKind for Block {
    fn reset(&mut self, _subs: &mut [BoxIntervals<'_>]) -> Result<()> {
        self.start = -1;
        self.end = -1;
        Ok(())
    }
    fn start(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        self.start
    }
    fn end(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        self.end
    }
    fn gaps(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        0
    }
    fn next_interval(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<i32> {
        if subs[0].next_interval()? == NO_MORE_INTERVALS {
            self.start = NO_MORE_INTERVALS;
            self.end = NO_MORE_INTERVALS;
            return Ok(NO_MORE_INTERVALS);
        }
        let mut i = 1usize;
        while i < subs.len() {
            while subs[i].start() <= subs[i - 1].end() {
                if subs[i].next_interval()? == NO_MORE_INTERVALS {
                    self.start = NO_MORE_INTERVALS;
                    self.end = NO_MORE_INTERVALS;
                    return Ok(NO_MORE_INTERVALS);
                }
            }
            if subs[i].start() == subs[i - 1].end().wrapping_add(1) {
                i += 1;
            } else {
                if subs[0].next_interval()? == NO_MORE_INTERVALS {
                    self.start = NO_MORE_INTERVALS;
                    self.end = NO_MORE_INTERVALS;
                    return Ok(NO_MORE_INTERVALS);
                }
                i = 1;
            }
        }
        self.start = subs[0].start();
        self.end = subs[subs.len() - 1].end();
        Ok(self.start)
    }
}

/// `BlockIntervalsSource.combine`.
pub(crate) fn block<'a>(subs: Vec<BoxIntervals<'a>>) -> BoxIntervals<'a> {
    conjunction(subs, Block { start: -1, end: -1 })
}

/// `OrderedIntervalsSource.OrderedIntervalIterator`.
struct Ordered<'a> {
    start: i32,
    end: i32,
    i: usize,
    slop: i32,
    on_match: MatchCallback<'a>,
}

impl ConjunctionKind for Ordered<'_> {
    fn reset(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<()> {
        subs[0].next_interval()?;
        self.i = 1;
        self.start = -1;
        self.end = -1;
        self.slop = -1;
        Ok(())
    }
    fn start(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        self.start
    }
    fn end(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        self.end
    }
    fn gaps(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        self.slop
    }
    fn next_interval(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<i32> {
        self.start = NO_MORE_INTERVALS;
        self.end = NO_MORE_INTERVALS;
        self.slop = NO_MORE_INTERVALS;
        let mut last_start = i32::MAX;
        let mut minimizing = false;
        let mut current_index = self.i;
        loop {
            let mut prev_end = subs[current_index - 1].end();
            loop {
                if prev_end >= last_start {
                    self.i = current_index;
                    return Ok(self.start);
                }
                if current_index == subs.len() {
                    break;
                }
                if minimizing && subs[current_index].start() > prev_end {
                    break;
                }
                loop {
                    if subs[current_index].end() >= last_start {
                        self.i = current_index;
                        return Ok(self.start);
                    }
                    let current_start = subs[current_index].next_interval()?;
                    if current_start == NO_MORE_INTERVALS {
                        self.i = current_index;
                        return Ok(self.start);
                    }
                    if current_start > prev_end {
                        break;
                    }
                }
                prev_end = subs[current_index].end();
                current_index += 1;
            }
            let start = subs[0].start();
            self.start = start;
            if start == NO_MORE_INTERVALS {
                self.i = current_index;
                self.end = NO_MORE_INTERVALS;
                return Ok(NO_MORE_INTERVALS);
            }
            let last = subs.len() - 1;
            let end = subs[last].end();
            self.end = end;
            let mut slop = end.wrapping_sub(start).wrapping_add(1);
            for s in subs.iter() {
                slop = slop.wrapping_sub(s.width());
            }
            self.slop = slop;
            on_match(&mut self.on_match)?;
            current_index = 1;
            if subs[0].next_interval()? == NO_MORE_INTERVALS {
                self.i = current_index;
                return Ok(self.start);
            }
            last_start = subs[last].start();
            minimizing = true;
        }
    }
}

/// `OrderedIntervalsSource.combine`.
pub(crate) fn ordered<'a>(
    subs: Vec<BoxIntervals<'a>>,
    on_match: MatchCallback<'a>,
) -> BoxIntervals<'a> {
    conjunction(
        subs,
        Ordered {
            start: -1,
            end: -1,
            i: 1,
            slop: 0,
            on_match,
        },
    )
}

/// `UnorderedIntervalsSource.UnorderedIntervalIterator`.
struct Unordered<'a> {
    queue: IndexQueue,
    start: i32,
    end: i32,
    slop: i32,
    queue_end: i32,
    on_match: MatchCallback<'a>,
}

impl Unordered<'_> {
    fn update_right_extreme(&mut self, it_end: i32) {
        if it_end > self.queue_end {
            self.queue_end = it_end;
        }
    }
}

impl ConjunctionKind for Unordered<'_> {
    fn reset(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<()> {
        self.queue_end = -1;
        self.start = -1;
        self.end = -1;
        self.queue.clear();
        for i in 0..subs.len() {
            if subs[i].next_interval()? == NO_MORE_INTERVALS {
                break;
            }
            self.queue.add(i, &start_then_wider(subs));
            let e = subs[i].end();
            self.update_right_extreme(e);
        }
        Ok(())
    }
    fn start(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        self.start
    }
    fn end(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        self.end
    }
    fn gaps(&self, _subs: &[BoxIntervals<'_>]) -> i32 {
        self.slop
    }
    fn next_interval(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<i32> {
        let n = subs.len();
        // First, find a matching interval.
        while self.queue.size() == n
            && self
                .queue
                .top()
                .is_some_and(|t| subs[t].start() == self.start)
        {
            let popped = self.queue.pop(&start_then_wider(subs));
            if let Some(it) = popped {
                if subs[it].next_interval()? != NO_MORE_INTERVALS {
                    self.queue.add(it, &start_then_wider(subs));
                    let e = subs[it].end();
                    self.update_right_extreme(e);
                }
            }
        }
        if self.queue.size() < n {
            self.start = NO_MORE_INTERVALS;
            self.end = NO_MORE_INTERVALS;
            return Ok(NO_MORE_INTERVALS);
        }
        // Then, minimize it.
        loop {
            let top = self.queue.top().unwrap_or(0);
            self.start = subs[top].start();
            self.end = self.queue_end;
            let mut slop = self.end.wrapping_sub(self.start).wrapping_add(1);
            for s in subs.iter() {
                slop = slop.wrapping_sub(s.width());
            }
            self.slop = slop;
            on_match(&mut self.on_match)?;
            if subs[top].end() == self.end {
                return Ok(self.start);
            }
            let popped = self.queue.pop(&start_then_wider(subs));
            if let Some(it) = popped {
                if subs[it].next_interval()? != NO_MORE_INTERVALS {
                    self.queue.add(it, &start_then_wider(subs));
                    let e = subs[it].end();
                    self.update_right_extreme(e);
                }
            }
            if !(self.queue.size() == n && self.end == self.queue_end) {
                break;
            }
        }
        Ok(self.start)
    }
}

/// `UnorderedIntervalsSource.combine`.
pub(crate) fn unordered<'a>(
    subs: Vec<BoxIntervals<'a>>,
    on_match: MatchCallback<'a>,
) -> BoxIntervals<'a> {
    let n = subs.len();
    conjunction(
        subs,
        Unordered {
            queue: IndexQueue::new(n),
            start: -1,
            end: -1,
            slop: 0,
            queue_end: 0,
            on_match,
        },
    )
}

/// The three `FilteringIntervalIterator`s: `a` (sub 0) filtered by `b`
/// (sub 1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum FilteringKind {
    /// `ContainingIntervalsSource`: `a` contains an interval of `b`.
    Containing,
    /// `ContainedByIntervalsSource`: `a` lies within an interval of `b`.
    ContainedBy,
    /// `OverlappingIntervalsSource`: `a` overlaps an interval of `b`.
    Overlapping,
}

struct Filtering {
    kind: FilteringKind,
    bpos: bool,
}

impl ConjunctionKind for Filtering {
    fn reset(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<()> {
        self.bpos = subs[1].next_interval()? != NO_MORE_INTERVALS;
        Ok(())
    }
    fn start(&self, subs: &[BoxIntervals<'_>]) -> i32 {
        if self.bpos {
            subs[0].start()
        } else {
            NO_MORE_INTERVALS
        }
    }
    fn end(&self, subs: &[BoxIntervals<'_>]) -> i32 {
        if self.bpos {
            subs[0].end()
        } else {
            NO_MORE_INTERVALS
        }
    }
    fn gaps(&self, subs: &[BoxIntervals<'_>]) -> i32 {
        subs[0].gaps()
    }
    fn next_interval(&mut self, subs: &mut [BoxIntervals<'_>]) -> Result<i32> {
        if !self.bpos {
            return Ok(NO_MORE_INTERVALS);
        }
        let (a, b) = subs.split_at_mut(1);
        let (a, b) = (&mut a[0], &mut b[0]);
        while a.next_interval()? != NO_MORE_INTERVALS {
            match self.kind {
                FilteringKind::Containing => {
                    while b.start() < a.start() && b.end() < a.end() {
                        if b.next_interval()? == NO_MORE_INTERVALS {
                            self.bpos = false;
                            return Ok(NO_MORE_INTERVALS);
                        }
                    }
                    if a.start() <= b.start() && a.end() >= b.end() {
                        return Ok(a.start());
                    }
                }
                FilteringKind::ContainedBy => {
                    while b.end() < a.end() {
                        if b.next_interval()? == NO_MORE_INTERVALS {
                            self.bpos = false;
                            return Ok(NO_MORE_INTERVALS);
                        }
                    }
                    if b.start() <= a.start() {
                        return Ok(a.start());
                    }
                }
                FilteringKind::Overlapping => {
                    while b.end() < a.start() {
                        if b.next_interval()? == NO_MORE_INTERVALS {
                            self.bpos = false;
                            return Ok(NO_MORE_INTERVALS);
                        }
                    }
                    if b.start() <= a.end() {
                        return Ok(a.start());
                    }
                }
            }
        }
        // `ContainingIntervalsSource` leaves `bpos` set here; the other two
        // clear it.
        if self.kind != FilteringKind::Containing {
            self.bpos = false;
        }
        Ok(NO_MORE_INTERVALS)
    }
}

/// A `FilteringIntervalIterator` over `a` and `b`.
pub(crate) fn filtering<'a>(
    kind: FilteringKind,
    a: BoxIntervals<'a>,
    b: BoxIntervals<'a>,
) -> BoxIntervals<'a> {
    conjunction(vec![a, b], Filtering { kind, bpos: false })
}

// ---------------------------------------------------------------------------
// MinimumShouldMatchIntervalsSource.MinimumShouldMatchIntervalIterator
// ---------------------------------------------------------------------------

/// `MinimumShouldMatchIntervalIterator`: an unordered conjunction of the
/// `minShouldMatch` sub-iterators at the front of a disjunction.
pub(crate) struct MinimumShouldMatchIntervals<'a> {
    pub(crate) subs: Vec<BoxIntervals<'a>>,
    disi: DisiQueue,
    proximity: IndexQueue,
    background: IndexQueue,
    match_cost: f32,
    min_should_match: usize,
    on_match: MatchCallback<'a>,
    start: i32,
    end: i32,
    queue_end: i32,
    slop: i32,
    lead: Option<usize>,
}

impl<'a> MinimumShouldMatchIntervals<'a> {
    pub(crate) fn new(
        subs: Vec<BoxIntervals<'a>>,
        min_should_match: usize,
        on_match: MatchCallback<'a>,
    ) -> Self {
        let disi = DisiQueue::new(&subs);
        let match_cost = summed_match_cost(&subs);
        let n = subs.len();
        MinimumShouldMatchIntervals {
            subs,
            disi,
            proximity: IndexQueue::new(min_should_match),
            background: IndexQueue::new(n),
            match_cost,
            min_should_match,
            on_match,
            start: 0,
            end: 0,
            queue_end: 0,
            slop: 0,
            lead: None,
        }
    }

    fn update_right_extreme(&mut self, it: usize) {
        let e = self.subs[it].end();
        if e > self.queue_end {
            self.queue_end = e;
        }
    }

    fn reset(&mut self) -> Result<()> {
        self.proximity.clear();
        self.background.clear();
        for &w in self.disi.top_list() {
            if self.subs[w].next_interval()? != NO_MORE_INTERVALS {
                self.background.add(w, &end_then_wider(&self.subs));
            }
        }
        self.queue_end = -1;
        for _ in 0..self.min_should_match {
            let Some(it) = self.background.pop(&end_then_wider(&self.subs)) else {
                break;
            };
            self.proximity.add(it, &start_then_wider(&self.subs));
            self.update_right_extreme(it);
        }
        self.start = -1;
        self.end = -1;
        Ok(())
    }

    /// `getCurrentIterators()`: the sub-iterators the current interval is
    /// made of.
    pub(crate) fn current_iterators(&self) -> Vec<usize> {
        let mut out = Vec::new();
        if let Some(lead) = self.lead {
            out.push(lead);
        }
        for it in self.proximity.iter() {
            if self.subs[it].end() <= self.end {
                out.push(it);
            }
        }
        out
    }
}

impl IntervalIterator for MinimumShouldMatchIntervals<'_> {
    fn doc_id(&self) -> i32 {
        self.disi.top_doc()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let doc = self.disi.next_doc(&mut self.subs)?;
        self.reset()?;
        Ok(doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.disi.advance(&mut self.subs, target)?;
        self.reset()?;
        Ok(doc)
    }
    fn cost(&self) -> i64 {
        self.disi.cost
    }
    fn start(&self) -> i32 {
        self.start
    }
    fn end(&self) -> i32 {
        self.end
    }
    fn gaps(&self) -> i32 {
        self.slop
    }
    fn next_interval(&mut self) -> Result<i32> {
        self.lead = None;
        let msm = self.min_should_match;
        // First, find a matching interval beyond the current start.
        while self.proximity.size() == msm
            && self
                .proximity
                .top()
                .is_some_and(|t| self.subs[t].start() == self.start)
        {
            let popped = self.proximity.pop(&start_then_wider(&self.subs));
            if let Some(it) = popped {
                if self.subs[it].next_interval()? != NO_MORE_INTERVALS {
                    self.background.add(it, &end_then_wider(&self.subs));
                    let next = self
                        .background
                        .pop(&end_then_wider(&self.subs))
                        .unwrap_or(it);
                    self.proximity.add(next, &start_then_wider(&self.subs));
                    self.update_right_extreme(next);
                }
            }
        }
        if self.proximity.size() < msm {
            self.start = NO_MORE_INTERVALS;
            self.end = NO_MORE_INTERVALS;
            return Ok(NO_MORE_INTERVALS);
        }
        // Then, minimize it.
        loop {
            on_match(&mut self.on_match)?;
            let top = self.proximity.top().unwrap_or(0);
            self.start = self.subs[top].start();
            self.end = self.queue_end;
            let mut slop = self.end.wrapping_sub(self.start).wrapping_add(1);
            for it in self.proximity.iter() {
                slop = slop.wrapping_sub(self.subs[it].width());
            }
            self.slop = slop;
            if self.subs[top].end() == self.end {
                return Ok(self.start);
            }
            self.lead = self.proximity.pop(&start_then_wider(&self.subs));
            if let Some(lead) = self.lead {
                if self.subs[lead].next_interval()? != NO_MORE_INTERVALS {
                    self.background.add(lead, &end_then_wider(&self.subs));
                }
                let popped = self.background.pop(&end_then_wider(&self.subs));
                if let Some(next) = popped {
                    self.proximity.add(next, &start_then_wider(&self.subs));
                    self.update_right_extreme(next);
                }
            }
            if !(self.proximity.size() == msm && self.end == self.queue_end) {
                break;
            }
        }
        Ok(self.start)
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
}

// ---------------------------------------------------------------------------
// RepeatingIntervalsSource.DuplicateIntervalIterator
// ---------------------------------------------------------------------------

/// `DuplicateIntervalIterator`: `copies` consecutive intervals of one
/// iterator taken as one interval, its width their summed widths.
pub(crate) struct DuplicateIntervals<'a> {
    inner: BoxIntervals<'a>,
    cache: Vec<i32>,
    cache_length: usize,
    cache_base: usize,
    started: bool,
    exhausted: bool,
}

impl<'a> DuplicateIntervals<'a> {
    pub(crate) fn new(inner: BoxIntervals<'a>, copies: usize) -> Self {
        DuplicateIntervals {
            inner,
            cache: vec![0; copies.saturating_mul(2)],
            cache_length: copies,
            cache_base: 0,
            started: false,
            exhausted: false,
        }
    }

    fn cache_next_interval(&mut self, line_pos: usize) -> Result<i32> {
        if self.inner.next_interval()? == NO_MORE_INTERVALS {
            self.exhausted = true;
            return Ok(NO_MORE_INTERVALS);
        }
        self.cache[line_pos * 2] = self.inner.start();
        self.cache[line_pos * 2 + 1] = self.inner.end();
        Ok(self.start())
    }

    fn clear(&mut self) {
        self.started = false;
        self.exhausted = false;
        self.cache.fill(-1);
    }
}

impl IntervalIterator for DuplicateIntervals<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.clear();
        self.inner.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.clear();
        self.inner.advance(target)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    fn start(&self) -> i32 {
        if self.exhausted {
            NO_MORE_INTERVALS
        } else {
            self.cache[(self.cache_base % self.cache_length) * 2]
        }
    }
    fn end(&self) -> i32 {
        if self.exhausted {
            NO_MORE_INTERVALS
        } else {
            let at = (self.cache_base + self.cache_length - 1) % self.cache_length;
            self.cache[at * 2 + 1]
        }
    }
    fn width(&self) -> i32 {
        let mut width = 0i32;
        for i in 0..self.cache_length {
            let pos = (self.cache_base + i) % self.cache_length;
            width = width
                .wrapping_add(self.cache[pos * 2])
                .wrapping_sub(self.cache[pos * 2 + 1])
                .wrapping_add(1);
        }
        width
    }
    fn gaps(&self) -> i32 {
        let outer = self.end().wrapping_sub(self.start()).wrapping_add(1);
        outer.wrapping_sub(self.width())
    }
    fn next_interval(&mut self) -> Result<i32> {
        if self.exhausted {
            return Ok(NO_MORE_INTERVALS);
        }
        if !self.started {
            for i in 0..self.cache_length {
                if self.cache_next_interval(i)? == NO_MORE_INTERVALS {
                    return Ok(NO_MORE_INTERVALS);
                }
            }
            self.cache_base = 0;
            self.started = true;
            return Ok(self.start());
        }
        let insert = (self.cache_base + self.cache_length) % self.cache_length;
        self.cache_base = (self.cache_base + 1) % self.cache_length;
        self.cache_next_interval(insert)
    }
    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
}

// ---------------------------------------------------------------------------
// IntervalFilter (FilteredIntervalsSource)
// ---------------------------------------------------------------------------

/// `IntervalFilter` with `FilteredIntervalsSource`'s `accept`.
pub(crate) struct FilteredIntervals<'a> {
    inner: BoxIntervals<'a>,
    filter: IntervalFilterKind,
}

impl<'a> FilteredIntervals<'a> {
    pub(crate) fn new(inner: BoxIntervals<'a>, filter: IntervalFilterKind) -> Self {
        FilteredIntervals { inner, filter }
    }

    fn accept(&self) -> bool {
        match self.filter {
            IntervalFilterKind::MaxGaps(n) => self.inner.gaps() <= n,
            IntervalFilterKind::MaxWidth(n) => {
                self.inner
                    .end()
                    .wrapping_sub(self.inner.start())
                    .wrapping_add(1)
                    <= n
            }
        }
    }
}

impl IntervalIterator for FilteredIntervals<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.inner.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.inner.advance(target)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    fn start(&self) -> i32 {
        self.inner.start()
    }
    fn end(&self) -> i32 {
        self.inner.end()
    }
    fn gaps(&self) -> i32 {
        self.inner.gaps()
    }
    fn next_interval(&mut self) -> Result<i32> {
        loop {
            let next = self.inner.next_interval()?;
            if next == NO_MORE_INTERVALS || self.accept() {
                return Ok(next);
            }
        }
    }
    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
}

// ---------------------------------------------------------------------------
// RelativeIterator: NOT_CONTAINING, NOT_CONTAINED_BY, NON_OVERLAPPING
// ---------------------------------------------------------------------------

/// The three `DifferenceIntervalsSource` iterators.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum RelativeKind {
    NotContaining,
    NotContainedBy,
    NonOverlapping,
}

/// A `RelativeIterator`: the minuend `a`'s intervals, those the subtrahend
/// `b` rules out skipped.
pub(crate) struct RelativeIntervals<'a> {
    a: BoxIntervals<'a>,
    b: BoxIntervals<'a>,
    bpos: bool,
    kind: RelativeKind,
}

impl<'a> RelativeIntervals<'a> {
    pub(crate) fn new(kind: RelativeKind, a: BoxIntervals<'a>, b: BoxIntervals<'a>) -> Self {
        RelativeIntervals {
            a,
            b,
            bpos: false,
            kind,
        }
    }

    fn reset(&mut self) -> Result<()> {
        let doc = self.a.doc_id();
        self.bpos =
            self.b.doc_id() == doc || (self.b.doc_id() < doc && self.b.advance(doc)? == doc);
        Ok(())
    }
}

impl IntervalIterator for RelativeIntervals<'_> {
    fn doc_id(&self) -> i32 {
        self.a.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        let doc = self.a.next_doc()?;
        self.reset()?;
        Ok(doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let doc = self.a.advance(target)?;
        self.reset()?;
        Ok(doc)
    }
    fn cost(&self) -> i64 {
        self.a.cost()
    }
    fn start(&self) -> i32 {
        self.a.start()
    }
    fn end(&self) -> i32 {
        self.a.end()
    }
    fn gaps(&self) -> i32 {
        self.a.gaps()
    }
    fn next_interval(&mut self) -> Result<i32> {
        if !self.bpos {
            return self.a.next_interval();
        }
        let (a, b) = (&mut self.a, &mut self.b);
        while a.next_interval()? != NO_MORE_INTERVALS {
            match self.kind {
                RelativeKind::NotContaining => {
                    while b.start() < a.start() && b.end() < a.end() {
                        if b.next_interval()? == NO_MORE_INTERVALS {
                            self.bpos = false;
                            return Ok(a.start());
                        }
                    }
                    if b.start() > a.end() {
                        return Ok(a.start());
                    }
                }
                RelativeKind::NotContainedBy => {
                    while b.end() < a.end() {
                        if b.next_interval()? == NO_MORE_INTERVALS {
                            return Ok(a.start());
                        }
                    }
                    if a.start() < b.start() {
                        return Ok(a.start());
                    }
                }
                RelativeKind::NonOverlapping => {
                    while b.end() < a.start() {
                        if b.next_interval()? == NO_MORE_INTERVALS {
                            self.bpos = false;
                            return Ok(a.start());
                        }
                    }
                    if b.start() > a.end() {
                        return Ok(a.start());
                    }
                }
            }
        }
        Ok(NO_MORE_INTERVALS)
    }
    fn match_cost(&self) -> f32 {
        self.a.match_cost() + self.b.match_cost()
    }
}

// ---------------------------------------------------------------------------
// ExtendedIntervalIterator, OffsetIntervalsSource's iterators
// ---------------------------------------------------------------------------

/// `ExtendedIntervalIterator`.
pub(crate) struct ExtendedIntervals<'a> {
    inner: BoxIntervals<'a>,
    before: i32,
    after: i32,
    positioned: bool,
}

impl<'a> ExtendedIntervals<'a> {
    pub(crate) fn new(inner: BoxIntervals<'a>, before: i32, after: i32) -> Self {
        ExtendedIntervals {
            inner,
            before,
            after,
            positioned: false,
        }
    }
}

impl IntervalIterator for ExtendedIntervals<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.positioned = false;
        self.inner.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.positioned = false;
        self.inner.advance(target)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    // SENTINEL: `-1` = "not on an interval yet", `IntervalIterator`'s own
    // contract; callers compare it as a position, below every real one, as
    // Java's do.
    fn start(&self) -> i32 {
        if !self.positioned {
            return -1;
        }
        let start = self.inner.start();
        if start == NO_MORE_INTERVALS {
            return NO_MORE_INTERVALS;
        }
        0.max(start.wrapping_sub(self.before))
    }
    // SENTINEL: `-1` = "not on an interval yet", `IntervalIterator`'s own
    // contract; callers compare it as a position, below every real one, as
    // Java's do.
    fn end(&self) -> i32 {
        if !self.positioned {
            return -1;
        }
        let end = self.inner.end();
        if end == NO_MORE_INTERVALS {
            return NO_MORE_INTERVALS;
        }
        let end = end.wrapping_add(self.after);
        if end < 0 || end == NO_MORE_INTERVALS {
            // overflow
            return NO_MORE_INTERVALS - 1;
        }
        end
    }
    fn gaps(&self) -> i32 {
        self.inner.gaps()
    }
    fn next_interval(&mut self) -> Result<i32> {
        self.positioned = true;
        self.inner.next_interval()?;
        Ok(self.start())
    }
    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
}

/// `OffsetIntervalsSource`'s two `OffsetIntervalIterator`s: a one-position
/// interval just before (`before`) or just after each of `inner`'s.
pub(crate) struct OffsetIntervals<'a> {
    inner: BoxIntervals<'a>,
    before: bool,
}

impl<'a> OffsetIntervals<'a> {
    pub(crate) fn new(inner: BoxIntervals<'a>, before: bool) -> Self {
        OffsetIntervals { inner, before }
    }
}

impl IntervalIterator for OffsetIntervals<'_> {
    fn doc_id(&self) -> i32 {
        self.inner.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.inner.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.inner.advance(target)
    }
    fn cost(&self) -> i64 {
        self.inner.cost()
    }
    // SENTINEL: `-1` = "not on an interval yet", `IntervalIterator`'s own
    // contract; callers compare it as a position, below every real one, as
    // Java's do.
    fn start(&self) -> i32 {
        if self.before {
            let pos = self.inner.start();
            if pos == -1 {
                return -1;
            }
            if pos == NO_MORE_INTERVALS {
                return NO_MORE_INTERVALS;
            }
            0.max(pos.wrapping_sub(1))
        } else {
            let pos = self.inner.end().wrapping_add(1);
            if pos == 0 {
                return -1;
            }
            if pos < 0 {
                // overflow
                return i32::MAX;
            }
            if pos == i32::MAX {
                return i32::MAX - 1;
            }
            pos
        }
    }
    fn end(&self) -> i32 {
        self.start()
    }
    fn gaps(&self) -> i32 {
        0
    }
    fn next_interval(&mut self) -> Result<i32> {
        self.inner.next_interval()?;
        Ok(self.start())
    }
    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
}

// ---------------------------------------------------------------------------
// IntervalsSource.intervals(field, ctx)
// ---------------------------------------------------------------------------

/// `IntervalsSource.intervals(field, ctx)`: the source's iterator over this
/// segment, `None` where Java returns `null` (no intervals for the field
/// here).
pub(crate) fn intervals<'a>(
    source: &IntervalsSource,
    field: &str,
    ctx: &LeafContext<'a>,
) -> Result<Option<BoxIntervals<'a>>> {
    use IntervalsSource as S;
    Ok(match source {
        S::Term(term) => {
            let Some(ft) = positions_field(ctx, field)? else {
                return Ok(None);
            };
            match ft.try_seek_exact(term)? {
                Some(stats) => Some(term_iterator(ctx, field, term, stats, None)?),
                None => None,
            }
        }
        S::PayloadFilteredTerm { term, filter } => {
            let Some(ft) = positions_field(ctx, field)? else {
                return Ok(None);
            };
            if !ft.has_payloads() {
                return Err(Error::IllegalArgument(format!(
                    "Cannot create a payload-filtered iterator over field {field} because it has no indexed payloads"
                )));
            }
            match ft.try_seek_exact(term)? {
                Some(stats) => Some(term_iterator(
                    ctx,
                    field,
                    term,
                    stats,
                    Some((filter.clone(), ft)),
                )?),
                None => None,
            }
        }
        S::Block(subs) => all_of(subs, field, ctx)?.map(block),
        S::Ordered(subs) => all_of(subs, field, ctx)?.map(|s| ordered(s, None)),
        S::Unordered(subs) => all_of(subs, field, ctx)?.map(|s| unordered(s, None)),
        S::Disjunction { sources, .. } => {
            let mut subs = Vec::with_capacity(sources.len());
            for s in sources {
                if let Some(it) = intervals(s, field, ctx)? {
                    subs.push(it);
                }
            }
            if subs.is_empty() {
                None
            } else {
                Some(BoxIntervals::boxed(DisjunctionIntervals::new(subs)))
            }
        }
        S::Repeating { source, count, .. } => {
            intervals(source, field, ctx)?.map(|it| -> BoxIntervals<'a> {
                BoxIntervals::boxed(DuplicateIntervals::new(
                    it,
                    usize::try_from(*count).unwrap_or(1),
                ))
            })
        }
        S::Filtered { source, filter } => {
            intervals(source, field, ctx)?.map(|it| -> BoxIntervals<'a> {
                BoxIntervals::boxed(FilteredIntervals::new(it, *filter))
            })
        }
        S::Extended {
            source,
            before,
            after,
        } => intervals(source, field, ctx)?.map(|it| -> BoxIntervals<'a> {
            BoxIntervals::boxed(ExtendedIntervals::new(it, *before, *after))
        }),
        S::Offset { source, before } => {
            intervals(source, field, ctx)?.map(|it| -> BoxIntervals<'a> {
                BoxIntervals::boxed(OffsetIntervals::new(it, *before))
            })
        }
        S::FixedField { field, source } => intervals(source, field, ctx)?,
        S::NoMatch(_) => None,
        S::Containing { big, small } => {
            pair(big, small, field, ctx)?.map(|(a, b)| filtering(FilteringKind::Containing, a, b))
        }
        S::ContainedBy { small, big } => {
            pair(small, big, field, ctx)?.map(|(a, b)| filtering(FilteringKind::ContainedBy, a, b))
        }
        S::Overlapping { source, reference } => pair(source, reference, field, ctx)?
            .map(|(a, b)| filtering(FilteringKind::Overlapping, a, b)),
        S::NotContaining {
            minuend,
            subtrahend,
        } => difference(RelativeKind::NotContaining, minuend, subtrahend, field, ctx)?,
        S::NotContainedBy {
            minuend,
            subtrahend,
        } => difference(
            RelativeKind::NotContainedBy,
            minuend,
            subtrahend,
            field,
            ctx,
        )?,
        S::NonOverlapping {
            minuend,
            subtrahend,
        } => difference(
            RelativeKind::NonOverlapping,
            minuend,
            subtrahend,
            field,
            ctx,
        )?,
        S::MinimumShouldMatch {
            sources,
            min_should_match,
        } => {
            let mut subs = Vec::with_capacity(sources.len());
            for s in sources {
                if let Some(it) = intervals(s, field, ctx)? {
                    subs.push(it);
                }
            }
            let msm = usize::try_from(*min_should_match).unwrap_or(0);
            if subs.len() < msm {
                None
            } else {
                Some(BoxIntervals::boxed(MinimumShouldMatchIntervals::new(
                    subs, msm, None,
                )))
            }
        }
        S::MultiTerm {
            pattern,
            max_expansions,
            name,
        } => {
            if ctx.fields.field(field).is_none() {
                return Ok(None);
            }
            let source = pattern.source(field);
            let terms = crate::exec::extended::expand_terms(ctx.fields, &source, None)?;
            let mut subs = Vec::new();
            for (term, seeked) in terms {
                subs.push(term_iterator(ctx, field, &term, seeked.stats, None)?);
                if subs.len() > *max_expansions {
                    return Err(Error::IllegalState(format!(
                        "Automaton [{name}] expanded to too many terms (limit {max_expansions})"
                    )));
                }
            }
            if subs.is_empty() {
                None
            } else {
                Some(BoxIntervals::boxed(DisjunctionIntervals::new(subs)))
            }
        }
    })
}

/// `ConjunctionIntervalsSource.intervals`: every sub-source's iterator, or
/// `None` when one has none.
fn all_of<'a>(
    sources: &[IntervalsSource],
    field: &str,
    ctx: &LeafContext<'a>,
) -> Result<Option<Vec<BoxIntervals<'a>>>> {
    let mut subs = Vec::with_capacity(sources.len());
    for s in sources {
        match intervals(s, field, ctx)? {
            Some(it) => subs.push(it),
            None => return Ok(None),
        }
    }
    Ok(Some(subs))
}

fn pair<'a>(
    a: &IntervalsSource,
    b: &IntervalsSource,
    field: &str,
    ctx: &LeafContext<'a>,
) -> Result<Option<(BoxIntervals<'a>, BoxIntervals<'a>)>> {
    let Some(a) = intervals(a, field, ctx)? else {
        return Ok(None);
    };
    let Some(b) = intervals(b, field, ctx)? else {
        return Ok(None);
    };
    Ok(Some((a, b)))
}

/// `DifferenceIntervalsSource.intervals`: the minuend alone when the
/// subtrahend has no iterator here.
fn difference<'a>(
    kind: RelativeKind,
    minuend: &IntervalsSource,
    subtrahend: &IntervalsSource,
    field: &str,
    ctx: &LeafContext<'a>,
) -> Result<Option<BoxIntervals<'a>>> {
    let Some(a) = intervals(minuend, field, ctx)? else {
        return Ok(None);
    };
    let Some(b) = intervals(subtrahend, field, ctx)? else {
        return Ok(Some(a));
    };
    Ok(Some(BoxIntervals::boxed(RelativeIntervals::new(
        kind, a, b,
    ))))
}

#[cfg(test)]
mod tests;
