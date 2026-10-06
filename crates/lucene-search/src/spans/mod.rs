//! `Spans` (`org.apache.lucene.queries.spans`, Lucene 10.5.0) as lazy
//! iterators, Java's classes method for method: `TermSpans`,
//! `NearSpansOrdered`/`NearSpansUnordered` over `ConjunctionSpans`, the
//! `SpanOrQuery` spans with its `SpanDisiPriorityQueue` and
//! `SpanPositionQueue`, `FilterSpans` (`SpanFirstQuery`,
//! `SpanPositionRangeQuery`, `SpanNotQuery`, `SpanPayloadCheckQuery`), the
//! `ContainSpans` of `SpanContainingQuery`/`SpanWithinQuery`, and
//! `PayloadScoreQuery`'s payload-collecting spans; with `SpanScorer`'s
//! sloppy frequency and `SpanWeight`'s similarity and explanation.
//!
//! The queries are [`SpanNode`]s, a `Clause::Extended` leaf (scored by
//! [`crate::exec::spans`]); the core span queries of terms, nears and ors
//! as a [`crate::query::Clause::Span`] keep their own materialized path
//! ([`crate::exec::span`]).
//!
//! # The two phases, as one
//!
//! Java's `Spans` is a `DocIdSetIterator` that may offer a
//! `TwoPhaseIterator`; every consumer here goes through the two-phase view.
//! So a [`Spans`] is its approximation ([`Spans::next_doc`],
//! [`Spans::advance`]) and its confirmation ([`Spans::matches`], which
//! leaves the spans before their first match, as `twoPhaseCurrentDocMatches`
//! does). A spans without approximation (a term's) confirms every document
//! its iterator stops on. Java's conjunction of sub-spans confirms each
//! sub-spans as its approximation moves; here the conjunction's
//! confirmation does, before its own -- the same documents, the same
//! positions.

pub mod payloads;

use lucene_codecs::blocktree::FieldTerms;
use lucene_codecs::postings::Position;

use crate::exec::span::LeafPositions;
use crate::exec::{LeafContext, NO_MORE_DOCS};
use crate::extended_query::MultiTermQuery;
use crate::index_searcher::IndexSearcher;
use crate::intervals::iterators::IndexQueue;
use crate::{Error, Result};

/// `Spans.NO_MORE_POSITIONS`.
pub const NO_MORE_POSITIONS: i32 = i32::MAX;

/// `SpanCollector`: what a span's leaves report through `collect`.
pub(crate) trait SpanCollector {
    /// `collectLeaf(postings, position, term)`: the term's occurrence at
    /// `position` of the current document, its payload read on demand.
    fn collect_leaf(&mut self, leaf: &mut TermSpans<'_>, position: i32) -> Result<()>;
    /// `reset()`.
    fn reset(&mut self);
}

/// `Spans` (with its two-phase view; see the module documentation).
pub(crate) trait Spans {
    /// The approximation's document.
    fn doc_id(&self) -> i32;
    /// Moves the approximation to its next document.
    fn next_doc(&mut self) -> Result<i32>;
    /// Moves the approximation to the first document at or after `target`.
    fn advance(&mut self, target: i32) -> Result<i32>;
    /// `cost()`.
    fn cost(&self) -> i64;
    /// `TwoPhaseIterator.matches()` (`true` for a spans without one): whether
    /// the current document has a span; the spans are then before it.
    fn matches(&mut self) -> Result<bool>;
    /// `TwoPhaseIterator.matchCost()`, or `positionsCost()` without one.
    fn match_cost(&self) -> f32;
    /// Whether Java's `asTwoPhaseIterator()` is non-null.
    fn two_phase(&self) -> bool;
    /// `nextStartPosition()`.
    fn next_start_position(&mut self) -> Result<i32>;
    /// `startPosition()`.
    fn start_position(&self) -> i32;
    /// `endPosition()`.
    fn end_position(&self) -> i32;
    /// `width()`.
    fn width(&self) -> i32;
    /// `collect(collector)`.
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()>;
    /// `doStartCurrentDoc()`: before the document's first span is scored.
    fn do_start_current_doc(&mut self) {}
    /// `doCurrentSpans()`: after each span is scored.
    fn do_current_spans(&mut self) -> Result<()> {
        Ok(())
    }
    /// [`sloppy_freq`] of the current document, monomorphised per spans:
    /// one virtual call per document instead of three per span.
    fn sloppy_freq(&mut self) -> Result<f32> {
        sloppy_freq(self)
    }
}

/// A spans as its parent holds it: a term's inline, so the leaf every span
/// query bottoms out in is called statically (and inlined) rather than
/// through a virtual call per document and per position -- the dispatch
/// Java's JIT removes by inlining the monomorphic call sites; any other
/// spans boxed.
// The term held inline, not boxed, is the point: no pointer to chase on
// the hottest calls. Parents hold a few of these, never many.
#[allow(clippy::large_enum_variant)]
pub(crate) enum BoxSpans<'a> {
    Term(TermSpans<'a>),
    Dyn(Box<dyn Spans + 'a>),
}

impl<'a> BoxSpans<'a> {
    /// Boxes `spans` behind a virtual call.
    pub(crate) fn boxed(spans: impl Spans + 'a) -> Self {
        BoxSpans::Dyn(Box::new(spans))
    }
}

/// Forwards to the variant, statically for a term.
macro_rules! dispatch {
    ($self:expr, $s:ident => $e:expr) => {
        match $self {
            BoxSpans::Term($s) => $e,
            BoxSpans::Dyn($s) => $e,
        }
    };
}

impl Spans for BoxSpans<'_> {
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
    fn matches(&mut self) -> Result<bool> {
        dispatch!(self, s => s.matches())
    }
    #[inline]
    fn match_cost(&self) -> f32 {
        dispatch!(self, s => s.match_cost())
    }
    #[inline]
    fn two_phase(&self) -> bool {
        dispatch!(self, s => s.two_phase())
    }
    #[inline]
    fn next_start_position(&mut self) -> Result<i32> {
        dispatch!(self, s => s.next_start_position())
    }
    #[inline]
    fn start_position(&self) -> i32 {
        dispatch!(self, s => s.start_position())
    }
    #[inline]
    fn end_position(&self) -> i32 {
        dispatch!(self, s => s.end_position())
    }
    #[inline]
    fn width(&self) -> i32 {
        dispatch!(self, s => s.width())
    }
    #[inline]
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()> {
        dispatch!(self, s => s.collect(collector))
    }
    #[inline]
    fn do_start_current_doc(&mut self) {
        dispatch!(self, s => s.do_start_current_doc())
    }
    #[inline]
    fn do_current_spans(&mut self) -> Result<()> {
        dispatch!(self, s => s.do_current_spans())
    }
    #[inline]
    fn sloppy_freq(&mut self) -> Result<f32> {
        dispatch!(self, s => s.sloppy_freq())
    }
}

// ---------------------------------------------------------------------------
// TermSpans
// ---------------------------------------------------------------------------

/// `SpanTermQuery.PHRASE_TO_SPAN_TERM_POSITIONS_COST`.
const PHRASE_TO_SPAN_TERM_POSITIONS_COST: f32 = 4.0;
/// `SpanTermQuery.TERM_POSNS_SEEK_OPS_PER_DOC`.
const TERM_POSNS_SEEK_OPS_PER_DOC: f32 = 128.0;
/// `SpanTermQuery.TERM_OPS_PER_POS`.
const TERM_OPS_PER_POS: f32 = 7.0;

/// `TermSpans`: one span `[p, p + 1)` per position of the term in the
/// document, its occurrences (payloads, offsets) read when collected.
pub(crate) struct TermSpans<'a> {
    postings: LeafPositions<'a>,
    field_terms: &'a FieldTerms,
    ctx: LeafContext<'a>,
    /// `(field, term)`.
    pub(crate) term: (String, Vec<u8>),
    doc: i32,
    freq: i32,
    count: i32,
    position: i32,
    positions: Vec<i32>,
    loaded: bool,
    /// The document's occurrences, payloads included, once asked for.
    occurrences: Vec<Position>,
    occurrences_loaded: bool,
    /// Positions come one at a time off a lazy cursor (`nextPosition()`)...
    lazy: bool,
    /// ...with their payloads (`PostingsEnum.PAYLOADS`), the current one
    /// kept here.
    stream: bool,
    payload: Option<Vec<u8>>,
    cost: i64,
    positions_cost: f32,
}

impl<'a> TermSpans<'a> {
    #[inline]
    fn reset(&mut self) {
        if self.doc != NO_MORE_DOCS {
            self.freq = i32::try_from(self.postings.freq_at(self.doc)).unwrap_or(i32::MAX);
            self.count = 0;
        }
        self.position = -1;
        self.loaded = false;
        self.occurrences_loaded = false;
    }

    /// The payload at the current position (`postings.getPayload()`),
    /// `None` where it has none.
    pub(crate) fn payload(&mut self) -> Result<Option<&[u8]>> {
        if self.stream {
            return Ok(self.payload.as_deref());
        }
        let occ = self.occurrence()?;
        Ok(occ.and_then(|o| (!o.payload.is_empty()).then_some(o.payload.as_slice())))
    }

    /// The occurrence at the current position: its payload and offsets.
    pub(crate) fn occurrence(&mut self) -> Result<Option<&Position>> {
        if !self.occurrences_loaded {
            self.occurrences_loaded = true;
            self.postings.occurrences_at(
                &self.ctx,
                self.field_terms,
                &self.term.1,
                self.doc,
                &mut self.occurrences,
            )?;
        }
        let at = usize::try_from(self.count).unwrap_or(0).checked_sub(1);
        Ok(at.and_then(|i| self.occurrences.get(i)))
    }
}

impl Spans for TermSpans<'_> {
    #[inline]
    fn doc_id(&self) -> i32 {
        self.doc
    }
    #[inline]
    fn next_doc(&mut self) -> Result<i32> {
        self.doc = self.postings.next_doc(self.doc)?;
        self.reset();
        Ok(self.doc)
    }
    #[inline]
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.doc = self.postings.advance(target)?;
        self.reset();
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.cost
    }
    fn matches(&mut self) -> Result<bool> {
        Ok(true)
    }
    fn match_cost(&self) -> f32 {
        self.positions_cost
    }
    fn two_phase(&self) -> bool {
        false
    }
    #[inline]
    fn next_start_position(&mut self) -> Result<i32> {
        if self.count == self.freq {
            self.position = NO_MORE_POSITIONS;
            return Ok(self.position);
        }
        if self.stream {
            let (position, payload) = self.postings.next_position_with_payload()?;
            self.position = position;
            match (&mut self.payload, payload) {
                (Some(buf), Some(p)) => {
                    buf.clear();
                    buf.extend_from_slice(p);
                }
                (slot, p) => *slot = p.map(<[u8]>::to_vec),
            }
            self.count += 1;
            return Ok(self.position);
        }
        if self.lazy {
            self.position = self.postings.next_position()?;
            self.count += 1;
            return Ok(self.position);
        }
        if !self.loaded {
            self.loaded = true;
            self.positions.clear();
            self.postings.positions_at(self.doc, &mut self.positions)?;
        }
        let at = usize::try_from(self.count).unwrap_or(0);
        self.position = self.positions.get(at).copied().unwrap_or(NO_MORE_POSITIONS);
        self.count += 1;
        Ok(self.position)
    }
    #[inline]
    fn start_position(&self) -> i32 {
        self.position
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    #[inline]
    fn end_position(&self) -> i32 {
        if self.position == -1 {
            -1
        } else if self.position != NO_MORE_POSITIONS {
            self.position.wrapping_add(1)
        } else {
            NO_MORE_POSITIONS
        }
    }
    fn width(&self) -> i32 {
        0
    }
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()> {
        let position = self.position;
        collector.collect_leaf(self, position)
    }
}

/// `SpanTermWeight.getSpans`: `None` when the term is not in this segment.
/// `payloads` is `requiredPostings` reaching `PAYLOADS`: the cursor then
/// reads each position's payload with it.
fn term_spans<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
    term: &[u8],
    payloads: bool,
) -> Result<Option<TermSpans<'a>>> {
    let Some(ft) = ctx.fields.field(field) else {
        return Ok(None);
    };
    let Some(stats) = ft.try_seek_exact(term)? else {
        return Ok(None);
    };
    if !ft.index_options().subsumes_positions() {
        return Err(Error::IllegalState(format!(
            "field \"{field}\" was indexed without position data; cannot run SpanTermQuery (term={})",
            String::from_utf8_lossy(term)
        )));
    }
    let Some(pos_in) = ctx.pos_in else {
        return Err(Error::MissingPosInput);
    };
    let exp = stats.total_term_freq as f32 / stats.doc_freq as f32;
    let positions_cost =
        (TERM_POSNS_SEEK_OPS_PER_DOC + exp * TERM_OPS_PER_POS) * PHRASE_TO_SPAN_TERM_POSITIONS_COST;
    let mut postings = LeafPositions::open(ctx, pos_in, field, term)?;
    let stream = payloads && ft.has_payloads() && postings.stream_payloads(ctx);
    let lazy = postings.is_lazy();
    Ok(Some(TermSpans {
        postings,
        field_terms: ft,
        ctx: *ctx,
        term: (field.to_string(), term.to_vec()),
        doc: -1,
        freq: 0,
        count: 0,
        position: -1,
        positions: Vec::new(),
        loaded: false,
        occurrences: Vec::new(),
        occurrences_loaded: false,
        lazy,
        stream,
        payload: None,
        cost: i64::from(stats.doc_freq),
        positions_cost,
    }))
}

// ---------------------------------------------------------------------------
// ConjunctionSpans: NearSpansOrdered, NearSpansUnordered, ContainSpans
// ---------------------------------------------------------------------------

/// `ConjunctionSpans`' doc-level half: a `ConjunctionDISI` over the
/// sub-spans' approximations, cheapest first, each confirmed in turn.
struct SpansConjunction {
    lead1: usize,
    lead2: usize,
    others: Vec<usize>,
    match_cost: f32,
}

impl SpansConjunction {
    fn new(subs: &[BoxSpans<'_>]) -> Self {
        let mut order: Vec<usize> = (0..subs.len()).collect();
        order.sort_by_key(|&i| subs[i].cost());
        SpansConjunction {
            lead1: order[0],
            lead2: order.get(1).copied().unwrap_or(order[0]),
            others: order.iter().skip(2).copied().collect(),
            match_cost: subs.iter().fold(0.0f32, |acc, s| acc + s.match_cost()),
        }
    }

    fn doc_id(&self, subs: &[BoxSpans<'_>]) -> i32 {
        subs[self.lead1].doc_id()
    }

    fn cost(&self, subs: &[BoxSpans<'_>]) -> i64 {
        subs[self.lead1].cost()
    }

    fn do_next(&self, subs: &mut [BoxSpans<'_>], mut doc: i32) -> Result<i32> {
        'advance_head: loop {
            if doc == NO_MORE_DOCS {
                return Ok(doc);
            }
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

    fn next_doc(&self, subs: &mut [BoxSpans<'_>]) -> Result<i32> {
        let doc = subs[self.lead1].next_doc()?;
        self.do_next(subs, doc)
    }

    fn advance(&self, subs: &mut [BoxSpans<'_>], target: i32) -> Result<i32> {
        let doc = subs[self.lead1].advance(target)?;
        self.do_next(subs, doc)
    }

    /// `ConjunctionTwoPhaseIterator.matches()`: every sub-spans confirmed.
    fn subs_match(subs: &mut [BoxSpans<'_>]) -> Result<bool> {
        for s in subs.iter_mut() {
            if !s.matches()? {
                return Ok(false);
            }
        }
        Ok(true)
    }
}

/// `NearSpansOrdered`.
pub(crate) struct NearSpansOrdered<'a> {
    subs: Vec<BoxSpans<'a>>,
    conj: SpansConjunction,
    at_first_in_current_doc: bool,
    one_exhausted_in_current_doc: bool,
    match_start: i32,
    match_end: i32,
    match_width: i32,
    allowed_slop: i32,
}

impl<'a> NearSpansOrdered<'a> {
    pub(crate) fn new(allowed_slop: i32, subs: Vec<BoxSpans<'a>>) -> Self {
        let conj = SpansConjunction::new(&subs);
        NearSpansOrdered {
            subs,
            conj,
            at_first_in_current_doc: true,
            one_exhausted_in_current_doc: false,
            match_start: -1,
            match_end: -1,
            match_width: -1,
            allowed_slop,
        }
    }

    /// `stretchToOrder()`.
    fn stretch_to_order(&mut self) -> Result<bool> {
        self.match_start = self.subs[0].start_position();
        self.match_width = 0;
        for i in 1..self.subs.len() {
            let prev_end = self.subs[i - 1].end_position();
            // `advancePosition(spans, prevSpans.endPosition())`.
            while self.subs[i].start_position() < prev_end {
                self.subs[i].next_start_position()?;
            }
            if self.subs[i].start_position() == NO_MORE_POSITIONS {
                self.one_exhausted_in_current_doc = true;
                return Ok(false);
            }
            self.match_width = self
                .match_width
                .wrapping_add(self.subs[i].start_position().wrapping_sub(prev_end));
        }
        self.match_end = self.subs[self.subs.len() - 1].end_position();
        Ok(true)
    }

    /// `twoPhaseCurrentDocMatches()`.
    fn current_doc_matches(&mut self) -> Result<bool> {
        self.one_exhausted_in_current_doc = false;
        while self.subs[0].next_start_position()? != NO_MORE_POSITIONS
            && !self.one_exhausted_in_current_doc
        {
            if self.stretch_to_order()? && self.match_width <= self.allowed_slop {
                self.at_first_in_current_doc = true;
                return Ok(true);
            }
        }
        Ok(false)
    }
}

impl Spans for NearSpansOrdered<'_> {
    fn doc_id(&self) -> i32 {
        self.conj.doc_id(&self.subs)
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.conj.next_doc(&mut self.subs)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.conj.advance(&mut self.subs, target)
    }
    fn cost(&self) -> i64 {
        self.conj.cost(&self.subs)
    }
    fn matches(&mut self) -> Result<bool> {
        Ok(SpansConjunction::subs_match(&mut self.subs)? && self.current_doc_matches()?)
    }
    fn match_cost(&self) -> f32 {
        self.conj.match_cost
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn next_start_position(&mut self) -> Result<i32> {
        if self.at_first_in_current_doc {
            self.at_first_in_current_doc = false;
            return Ok(self.match_start);
        }
        self.one_exhausted_in_current_doc = false;
        while self.subs[0].next_start_position()? != NO_MORE_POSITIONS
            && !self.one_exhausted_in_current_doc
        {
            if self.stretch_to_order()? && self.match_width <= self.allowed_slop {
                return Ok(self.match_start);
            }
        }
        self.match_start = NO_MORE_POSITIONS;
        self.match_end = NO_MORE_POSITIONS;
        Ok(NO_MORE_POSITIONS)
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn start_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else {
            self.match_start
        }
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn end_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else {
            self.match_end
        }
    }
    fn width(&self) -> i32 {
        self.match_width
    }
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()> {
        for s in &mut self.subs {
            s.collect(collector)?;
        }
        Ok(())
    }
}

/// `NearSpansUnordered.positionsOrdered`.
fn positions_ordered(a: &BoxSpans<'_>, b: &BoxSpans<'_>) -> bool {
    let (s1, s2) = (a.start_position(), b.start_position());
    if s1 == s2 {
        a.end_position() < b.end_position()
    } else {
        s1 < s2
    }
}

/// `NearSpansUnordered` and its `SpanTotalLengthEndPositionWindow`.
pub(crate) struct NearSpansUnordered<'a> {
    subs: Vec<BoxSpans<'a>>,
    conj: SpansConjunction,
    at_first_in_current_doc: bool,
    one_exhausted_in_current_doc: bool,
    allowed_slop: i32,
    window: IndexQueue,
    total_span_length: i32,
    max_end_position: i32,
}

impl<'a> NearSpansUnordered<'a> {
    pub(crate) fn new(allowed_slop: i32, subs: Vec<BoxSpans<'a>>) -> Self {
        let conj = SpansConjunction::new(&subs);
        let n = subs.len();
        NearSpansUnordered {
            subs,
            conj,
            at_first_in_current_doc: true,
            one_exhausted_in_current_doc: false,
            allowed_slop,
            window: IndexQueue::new(n),
            total_span_length: 0,
            max_end_position: -1,
        }
    }

    fn top(&self) -> usize {
        self.window.top().unwrap_or(0)
    }

    /// `startDocument()`.
    fn start_document(&mut self) -> Result<()> {
        self.window.clear();
        self.total_span_length = 0;
        self.max_end_position = -1;
        for i in 0..self.subs.len() {
            self.subs[i].next_start_position()?;
            let subs = &self.subs;
            self.window
                .add(i, &|a, b| positions_ordered(&subs[a], &subs[b]));
            let s = &self.subs[i];
            if s.end_position() > self.max_end_position {
                self.max_end_position = s.end_position();
            }
            let len = s.end_position().wrapping_sub(s.start_position());
            self.total_span_length = self.total_span_length.wrapping_add(len);
        }
        Ok(())
    }

    /// `nextPosition()`.
    fn next_position(&mut self) -> Result<bool> {
        let top = self.top();
        let s = &self.subs[top];
        let mut span_length = s.end_position().wrapping_sub(s.start_position());
        let next_start = self.subs[top].next_start_position()?;
        if next_start == NO_MORE_POSITIONS {
            return Ok(false);
        }
        self.total_span_length = self.total_span_length.wrapping_sub(span_length);
        let s = &self.subs[top];
        span_length = s.end_position().wrapping_sub(s.start_position());
        self.total_span_length = self.total_span_length.wrapping_add(span_length);
        if s.end_position() > self.max_end_position {
            self.max_end_position = s.end_position();
        }
        let subs = &self.subs;
        self.window
            .update_top(&|a, b| positions_ordered(&subs[a], &subs[b]));
        Ok(true)
    }

    /// `atMatch()`.
    fn at_match(&self) -> bool {
        let top_start = self.subs[self.top()].start_position();
        self.max_end_position
            .wrapping_sub(top_start)
            .wrapping_sub(self.total_span_length)
            <= self.allowed_slop
    }

    /// `twoPhaseCurrentDocMatches()`.
    fn current_doc_matches(&mut self) -> Result<bool> {
        self.start_document()?;
        loop {
            if self.at_match() {
                self.at_first_in_current_doc = true;
                self.one_exhausted_in_current_doc = false;
                return Ok(true);
            }
            if !self.next_position()? {
                return Ok(false);
            }
        }
    }
}

impl Spans for NearSpansUnordered<'_> {
    fn doc_id(&self) -> i32 {
        self.conj.doc_id(&self.subs)
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.conj.next_doc(&mut self.subs)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.conj.advance(&mut self.subs, target)
    }
    fn cost(&self) -> i64 {
        self.conj.cost(&self.subs)
    }
    fn matches(&mut self) -> Result<bool> {
        Ok(SpansConjunction::subs_match(&mut self.subs)? && self.current_doc_matches()?)
    }
    fn match_cost(&self) -> f32 {
        self.conj.match_cost
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn next_start_position(&mut self) -> Result<i32> {
        if self.at_first_in_current_doc {
            self.at_first_in_current_doc = false;
            return Ok(self.subs[self.top()].start_position());
        }
        loop {
            if !self.next_position()? {
                self.one_exhausted_in_current_doc = true;
                return Ok(NO_MORE_POSITIONS);
            }
            if self.at_match() {
                return Ok(self.subs[self.top()].start_position());
            }
        }
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn start_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else if self.one_exhausted_in_current_doc {
            NO_MORE_POSITIONS
        } else {
            self.subs[self.top()].start_position()
        }
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn end_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else if self.one_exhausted_in_current_doc {
            NO_MORE_POSITIONS
        } else {
            self.max_end_position
        }
    }
    fn width(&self) -> i32 {
        self.max_end_position
            .wrapping_sub(self.subs[self.top()].start_position())
    }
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()> {
        for s in &mut self.subs {
            s.collect(collector)?;
        }
        Ok(())
    }
}

/// `ContainSpans`: `SpanContainingQuery`'s (the big spans that contain a
/// little one) or `SpanWithinQuery`'s (the little spans within a big one).
pub(crate) struct ContainSpans<'a> {
    /// `[big, little]`.
    subs: Vec<BoxSpans<'a>>,
    conj: SpansConjunction,
    at_first_in_current_doc: bool,
    one_exhausted_in_current_doc: bool,
    /// `true`: `SpanContainingQuery` (the source is the big spans);
    /// `false`: `SpanWithinQuery` (the little).
    containing: bool,
}

impl<'a> ContainSpans<'a> {
    pub(crate) fn new(big: BoxSpans<'a>, little: BoxSpans<'a>, containing: bool) -> Self {
        let subs = vec![big, little];
        let conj = SpansConjunction::new(&subs);
        ContainSpans {
            subs,
            conj,
            at_first_in_current_doc: true,
            one_exhausted_in_current_doc: false,
            containing,
        }
    }

    fn source(&self) -> &BoxSpans<'a> {
        if self.containing {
            &self.subs[0]
        } else {
            &self.subs[1]
        }
    }

    /// The loop both `twoPhaseCurrentDocMatches` and `nextStartPosition`
    /// run: the next source span with a partner, or `None`.
    fn next_match(&mut self) -> Result<Option<i32>> {
        let (big, little) = self.subs.split_at_mut(1);
        let (big, little) = (&mut big[0], &mut little[0]);
        if self.containing {
            while big.next_start_position()? != NO_MORE_POSITIONS {
                while little.start_position() < big.start_position() {
                    if little.next_start_position()? == NO_MORE_POSITIONS {
                        self.one_exhausted_in_current_doc = true;
                        return Ok(None);
                    }
                }
                if big.end_position() >= little.end_position() {
                    return Ok(Some(big.start_position()));
                }
            }
        } else {
            while little.next_start_position()? != NO_MORE_POSITIONS {
                while big.end_position() < little.end_position() {
                    if big.next_start_position()? == NO_MORE_POSITIONS {
                        self.one_exhausted_in_current_doc = true;
                        return Ok(None);
                    }
                }
                if big.start_position() <= little.start_position() {
                    return Ok(Some(little.start_position()));
                }
            }
        }
        self.one_exhausted_in_current_doc = true;
        Ok(None)
    }
}

impl Spans for ContainSpans<'_> {
    fn doc_id(&self) -> i32 {
        self.conj.doc_id(&self.subs)
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.conj.next_doc(&mut self.subs)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.conj.advance(&mut self.subs, target)
    }
    fn cost(&self) -> i64 {
        self.conj.cost(&self.subs)
    }
    fn matches(&mut self) -> Result<bool> {
        if !SpansConjunction::subs_match(&mut self.subs)? {
            return Ok(false);
        }
        self.one_exhausted_in_current_doc = false;
        if self.next_match()?.is_some() {
            self.at_first_in_current_doc = true;
            return Ok(true);
        }
        Ok(false)
    }
    fn match_cost(&self) -> f32 {
        self.conj.match_cost
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn next_start_position(&mut self) -> Result<i32> {
        if self.at_first_in_current_doc {
            self.at_first_in_current_doc = false;
            return Ok(self.source().start_position());
        }
        Ok(self.next_match()?.unwrap_or(NO_MORE_POSITIONS))
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn start_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else if self.one_exhausted_in_current_doc {
            NO_MORE_POSITIONS
        } else {
            self.source().start_position()
        }
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn end_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else if self.one_exhausted_in_current_doc {
            NO_MORE_POSITIONS
        } else {
            self.source().end_position()
        }
    }
    fn width(&self) -> i32 {
        self.source().width()
    }
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()> {
        self.subs[0].collect(collector)?;
        self.subs[1].collect(collector)
    }
}

// ---------------------------------------------------------------------------
// SpanOrQuery's spans
// ---------------------------------------------------------------------------

/// `SpanOrWeight.getSpans`' anonymous spans: a disjunction of sub-spans by
/// document (`SpanDisiPriorityQueue`), merged by position
/// (`SpanPositionQueue`) within one.
pub(crate) struct OrSpans<'a> {
    subs: Vec<BoxSpans<'a>>,
    /// `SpanDisiWrapper.doc`, per sub.
    docs: Vec<i32>,
    /// `SpanDisiWrapper.next` (`topList`'s links).
    next: Vec<Option<usize>>,
    /// `lastApproxMatchDoc` / `lastApproxNonMatchDoc`, per sub.
    last_match: Vec<i32>,
    last_non_match: Vec<i32>,
    heap: Vec<usize>,
    by_position: IndexQueue,
    top_position: Option<usize>,
    last_doc_two_phase_matched: i32,
    two_phase: bool,
    match_cost: f32,
    cost: i64,
}

impl<'a> OrSpans<'a> {
    pub(crate) fn new(subs: Vec<BoxSpans<'a>>) -> Self {
        let n = subs.len();
        let mut s = OrSpans {
            docs: vec![-1; n],
            next: vec![None; n],
            last_match: vec![-2; n],
            last_non_match: vec![-2; n],
            heap: Vec::with_capacity(n),
            by_position: IndexQueue::new(n),
            top_position: None,
            last_doc_two_phase_matched: -1,
            two_phase: false,
            match_cost: 0.0,
            cost: subs.iter().fold(0i64, |acc, s| acc.wrapping_add(s.cost())),
            subs,
        };
        for i in 0..n {
            s.heap.push(i);
            s.up_heap(i);
        }
        // `asTwoPhaseIterator()`: the cost-weighted mean match cost of the
        // two-phase subs, else (none) the mean positions cost.
        let (mut sum_match, mut sum_approx) = (0.0f32, 0i64);
        let (mut sum_positions, mut sum_cost) = (0.0f32, 0i64);
        for &w in &s.heap {
            let sub = &s.subs[w];
            let cost_weight = if sub.cost() <= 1 { 1 } else { sub.cost() };
            if sub.two_phase() {
                sum_match += sub.match_cost() * cost_weight as f32;
                sum_approx = sum_approx.wrapping_add(cost_weight);
            }
            sum_positions += sub.match_cost() * cost_weight as f32;
            sum_cost = sum_cost.wrapping_add(cost_weight);
        }
        if sum_approx == 0 {
            s.match_cost = sum_positions / sum_cost as f32;
        } else {
            s.two_phase = true;
            s.match_cost = sum_match / sum_approx as f32;
        }
        s
    }

    fn up_heap(&mut self, mut i: usize) {
        let node = self.heap[i];
        let node_doc = self.docs[node];
        while i > 0 {
            let p = i.div_ceil(2) - 1;
            if node_doc < self.docs[self.heap[p]] {
                self.heap[i] = self.heap[p];
                i = p;
            } else {
                break;
            }
        }
        self.heap[i] = node;
    }

    fn down_heap(&mut self) {
        let size = self.heap.len();
        let mut i = 0usize;
        let node = self.heap[0];
        let mut j = 1usize;
        if j < size {
            let mut k = j + 1;
            if k < size && self.docs[self.heap[k]] < self.docs[self.heap[j]] {
                j = k;
            }
            if self.docs[self.heap[j]] < self.docs[node] {
                loop {
                    self.heap[i] = self.heap[j];
                    i = j;
                    j = ((i + 1) << 1) - 1;
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

    fn top(&self) -> usize {
        self.heap[0]
    }

    /// `topList()`, in Java's linked-list order.
    fn top_list(&mut self) -> Vec<usize> {
        let size = self.heap.len();
        let mut list = self.heap[0];
        self.next[list] = None;
        if size >= 3 {
            list = self.top_list_from(list, 1);
            list = self.top_list_from(list, 2);
        } else if size == 2 && self.docs[self.heap[1]] == self.docs[list] {
            let w = self.heap[1];
            self.next[w] = Some(list);
            list = w;
        }
        let mut out = Vec::new();
        let mut at = Some(list);
        while let Some(w) = at {
            out.push(w);
            at = self.next[w];
        }
        out
    }

    fn top_list_from(&mut self, mut list: usize, i: usize) -> usize {
        let size = self.heap.len();
        let w = self.heap[i];
        if self.docs[w] == self.docs[list] {
            self.next[w] = Some(list);
            list = w;
            let left = ((i + 1) << 1) - 1;
            let right = left + 1;
            if right < size {
                list = self.top_list_from(list, left);
                list = self.top_list_from(list, right);
            } else if left < size && self.docs[self.heap[left]] == self.docs[list] {
                let l = self.heap[left];
                self.next[l] = Some(list);
                list = l;
            }
        }
        list
    }

    fn by_position_less(&self) -> impl Fn(usize, usize) -> bool + '_ {
        move |a, b| {
            let (s1, s2) = (self.subs[a].start_position(), self.subs[b].start_position());
            if s1 < s2 {
                true
            } else if s1 == s2 {
                self.subs[a].end_position() < self.subs[b].end_position()
            } else {
                false
            }
        }
    }

    /// `fillPositionQueue()`.
    fn fill_position_queue(&mut self) -> Result<()> {
        for w in self.top_list() {
            let mut include = true;
            if self.last_doc_two_phase_matched == self.docs[w] && self.subs[w].two_phase() {
                // `matches()` said no, or (not asked yet) says no now.
                if self.last_non_match[w] == self.docs[w]
                    || (self.last_match[w] != self.docs[w] && !self.subs[w].matches()?)
                {
                    include = false;
                }
            }
            if include {
                self.subs[w].next_start_position()?;
                let mut q = std::mem::take(&mut self.by_position);
                q.add(w, &self.by_position_less());
                self.by_position = q;
            }
        }
        Ok(())
    }
}

impl Spans for OrSpans<'_> {
    fn doc_id(&self) -> i32 {
        self.docs[self.top()]
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.top_position = None;
        let mut top = self.top();
        let current = self.docs[top];
        loop {
            self.docs[top] = self.subs[top].next_doc()?;
            self.down_heap();
            top = self.top();
            if self.docs[top] != current {
                return Ok(self.docs[top]);
            }
        }
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.top_position = None;
        let mut top = self.top();
        loop {
            self.docs[top] = self.subs[top].advance(target)?;
            self.down_heap();
            top = self.top();
            if self.docs[top] >= target {
                return Ok(self.docs[top]);
            }
        }
    }
    fn cost(&self) -> i64 {
        self.cost
    }
    /// `twoPhaseCurrentDocMatches()`: a sub on the document that confirms
    /// it (a sub without two phases always does).
    fn matches(&mut self) -> Result<bool> {
        let list = self.top_list();
        let current = self.docs[list[0]];
        let mut matched = false;
        for w in list {
            if !self.subs[w].two_phase() {
                matched = true;
                break;
            }
            if self.subs[w].matches()? {
                self.last_match[w] = current;
                matched = true;
                break;
            }
            self.last_non_match[w] = current;
        }
        if !matched {
            return Ok(false);
        }
        self.last_doc_two_phase_matched = current;
        self.top_position = None;
        Ok(true)
    }
    fn match_cost(&self) -> f32 {
        self.match_cost
    }
    fn two_phase(&self) -> bool {
        self.two_phase
    }
    fn next_start_position(&mut self) -> Result<i32> {
        match self.top_position {
            None => {
                self.by_position.clear();
                self.fill_position_queue()?;
                self.top_position = self.by_position.top();
            }
            Some(top) => {
                self.subs[top].next_start_position()?;
                let mut q = std::mem::take(&mut self.by_position);
                q.update_top(&self.by_position_less());
                self.by_position = q;
                self.top_position = self.by_position.top();
            }
        }
        Ok(self
            .top_position
            .map_or(NO_MORE_POSITIONS, |t| self.subs[t].start_position()))
    }
    fn start_position(&self) -> i32 {
        self.top_position
            .map_or(-1, |t| self.subs[t].start_position())
    }
    fn end_position(&self) -> i32 {
        self.top_position
            .map_or(-1, |t| self.subs[t].end_position())
    }
    fn width(&self) -> i32 {
        self.top_position.map_or(0, |t| self.subs[t].width())
    }
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()> {
        match self.top_position {
            Some(t) => self.subs[t].collect(collector),
            None => Ok(()),
        }
    }
}

// ---------------------------------------------------------------------------
// FilterSpans
// ---------------------------------------------------------------------------

/// `FilterSpans.AcceptStatus`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum AcceptStatus {
    Yes,
    No,
    NoMoreInCurrentDoc,
}

/// What a `FilterSpans` accepts.
pub(crate) trait SpanFilter {
    /// `accept(candidate)`.
    fn accept(&mut self, candidate: &mut BoxSpans<'_>) -> Result<AcceptStatus>;
}

/// `FilterSpans`.
pub(crate) struct FilterSpans<'a, F> {
    inner: BoxSpans<'a>,
    filter: F,
    at_first_in_current_doc: bool,
    start_pos: i32,
}

impl<'a, F: SpanFilter> FilterSpans<'a, F> {
    pub(crate) fn new(inner: BoxSpans<'a>, filter: F) -> Self {
        FilterSpans {
            inner,
            filter,
            at_first_in_current_doc: false,
            start_pos: -1,
        }
    }

    /// `twoPhaseCurrentDocMatches()`.
    fn current_doc_matches(&mut self) -> Result<bool> {
        self.at_first_in_current_doc = false;
        self.start_pos = self.inner.next_start_position()?;
        loop {
            match self.filter.accept(&mut self.inner)? {
                AcceptStatus::Yes => {
                    self.at_first_in_current_doc = true;
                    return Ok(true);
                }
                AcceptStatus::No => {
                    self.start_pos = self.inner.next_start_position()?;
                    if self.start_pos == NO_MORE_POSITIONS {
                        self.start_pos = -1;
                        return Ok(false);
                    }
                }
                AcceptStatus::NoMoreInCurrentDoc => {
                    self.start_pos = -1;
                    return Ok(false);
                }
            }
        }
    }
}

impl<F: SpanFilter> Spans for FilterSpans<'_, F> {
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
    fn matches(&mut self) -> Result<bool> {
        Ok(self.inner.matches()? && self.current_doc_matches()?)
    }
    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
    fn two_phase(&self) -> bool {
        true
    }
    fn next_start_position(&mut self) -> Result<i32> {
        if self.at_first_in_current_doc {
            self.at_first_in_current_doc = false;
            return Ok(self.start_pos);
        }
        loop {
            self.start_pos = self.inner.next_start_position()?;
            if self.start_pos == NO_MORE_POSITIONS {
                return Ok(NO_MORE_POSITIONS);
            }
            match self.filter.accept(&mut self.inner)? {
                AcceptStatus::Yes => return Ok(self.start_pos),
                AcceptStatus::No => {}
                AcceptStatus::NoMoreInCurrentDoc => {
                    self.start_pos = NO_MORE_POSITIONS;
                    return Ok(NO_MORE_POSITIONS);
                }
            }
        }
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn start_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else {
            self.start_pos
        }
    }
    // SENTINEL: `-1` = "before the document's first span", `Spans`' own
    // contract (`startPosition`/`endPosition` before `nextStartPosition`);
    // callers compare it as a position, below every real one, as Java's do.
    fn end_position(&self) -> i32 {
        if self.at_first_in_current_doc {
            -1
        } else if self.start_pos != NO_MORE_POSITIONS {
            self.inner.end_position()
        } else {
            NO_MORE_POSITIONS
        }
    }
    fn width(&self) -> i32 {
        self.inner.width()
    }
    fn collect(&mut self, collector: &mut dyn SpanCollector) -> Result<()> {
        self.inner.collect(collector)
    }
    fn do_start_current_doc(&mut self) {
        self.inner.do_start_current_doc();
    }
    fn do_current_spans(&mut self) -> Result<()> {
        self.inner.do_current_spans()
    }
}

/// `SpanPositionRangeQuery.acceptPosition` (`SpanFirstQuery`'s is the
/// range from 0, with the same answers).
pub(crate) struct PositionRange {
    pub(crate) start: i32,
    pub(crate) end: i32,
}

impl SpanFilter for PositionRange {
    fn accept(&mut self, s: &mut BoxSpans<'_>) -> Result<AcceptStatus> {
        Ok(if s.start_position() >= self.end {
            AcceptStatus::NoMoreInCurrentDoc
        } else if s.start_position() >= self.start && s.end_position() <= self.end {
            AcceptStatus::Yes
        } else {
            AcceptStatus::No
        })
    }
}

/// `SpanNotWeight.getSpans`' filter: no exclude span within `pre`
/// positions before or `post` after.
pub(crate) struct NotFilter<'a> {
    exclude: BoxSpans<'a>,
    pre: i32,
    post: i32,
    last_approx_doc: i32,
    last_approx_result: bool,
}

impl SpanFilter for NotFilter<'_> {
    fn accept(&mut self, candidate: &mut BoxSpans<'_>) -> Result<AcceptStatus> {
        let doc = candidate.doc_id();
        let two_phase = self.exclude.two_phase();
        if doc > self.exclude.doc_id() {
            if two_phase {
                if self.exclude.advance(doc)? == doc {
                    self.last_approx_doc = doc;
                    self.last_approx_result = self.exclude.matches()?;
                }
            } else {
                self.exclude.advance(doc)?;
            }
        } else if two_phase && doc == self.exclude.doc_id() && doc != self.last_approx_doc {
            self.last_approx_doc = doc;
            self.last_approx_result = self.exclude.matches()?;
        }
        if doc != self.exclude.doc_id() || (doc == self.last_approx_doc && !self.last_approx_result)
        {
            return Ok(AcceptStatus::Yes);
        }
        if self.exclude.start_position() == -1 {
            self.exclude.next_start_position()?;
        }
        while self.exclude.end_position() <= candidate.start_position().wrapping_sub(self.pre) {
            if self.exclude.next_start_position()? == NO_MORE_POSITIONS {
                return Ok(AcceptStatus::Yes);
            }
        }
        if self.exclude.start_position().wrapping_sub(self.post) >= candidate.end_position() {
            Ok(AcceptStatus::Yes)
        } else {
            Ok(AcceptStatus::No)
        }
    }
}

// ---------------------------------------------------------------------------
// getSpans
// ---------------------------------------------------------------------------

/// `SpanWeight.getSpans(context, requiredPostings)`: the query's spans over
/// this segment, `None` where Java's is `null`. `payloads` is
/// `requiredPostings` at `PAYLOADS` (a payload query's clauses read their
/// terms' payloads). [`root_spans`] with a sink that boxes.
///
/// # Errors
/// A term of a field indexed without positions; a near query of fewer than
/// two clauses; a [`SpanNode::MultiTerm`] not rewritten first (`"Rewrite
/// first!"`).
pub(crate) fn spans_with<'a>(
    ctx: &LeafContext<'a>,
    q: &SpanNode,
    payloads: bool,
) -> Result<Option<BoxSpans<'a>>> {
    root_spans(ctx, q, payloads, Boxing)
}

/// What [`root_spans`] hands a query's spans to: generic over the spans'
/// type, so a scorer built from them is monomorphised for it.
pub(crate) trait SpansSink<'a>: Sized {
    type Out;
    fn sink<S: Spans + 'a>(self, spans: S) -> Self::Out;

    /// A term's spans: [`Self::sink`]'s unless the sink keeps terms apart
    /// ([`BoxSpans::Term`]).
    fn sink_term(self, spans: TermSpans<'a>) -> Self::Out {
        self.sink(spans)
    }

    /// Spans already boxed (a disjunction, a payload check, a clause's):
    /// [`Self::sink`]'s unless the sink would box them again.
    fn sink_boxed(self, spans: BoxSpans<'a>) -> Self::Out {
        self.sink(spans)
    }
}

/// [`spans_with`]'s sink: a term inline, everything else behind a box.
struct Boxing;

impl<'a> SpansSink<'a> for Boxing {
    type Out = BoxSpans<'a>;
    fn sink<S: Spans + 'a>(self, spans: S) -> BoxSpans<'a> {
        BoxSpans::boxed(spans)
    }
    fn sink_term(self, spans: TermSpans<'a>) -> BoxSpans<'a> {
        BoxSpans::Term(spans)
    }
    fn sink_boxed(self, spans: BoxSpans<'a>) -> BoxSpans<'a> {
        spans
    }
}

/// The query's spans handed to `sink`: a term, a first or position range, a
/// not, a near or a containment as their own types rather than boxed, so a
/// scorer over them calls its spans without a virtual call per document and
/// per span (the dispatch Java's JIT removes by inlining). Their clauses,
/// a disjunction and the payload queries go boxed ([`spans_with`]).
pub(crate) fn root_spans<'a, K: SpansSink<'a>>(
    ctx: &LeafContext<'a>,
    q: &SpanNode,
    payloads: bool,
    sink: K,
) -> Result<Option<K::Out>> {
    let sub = |q: &SpanNode| spans_with(ctx, q, payloads);
    let range = |s, start, end| FilterSpans::new(s, PositionRange { start, end });
    Ok(match q {
        SpanNode::Term { field, term } => {
            term_spans(ctx, field, term, payloads)?.map(|t| sink.sink_term(t))
        }
        SpanNode::Near {
            clauses,
            slop,
            in_order,
        } => match near_subs(ctx, q, clauses, payloads)? {
            None => None,
            Some(subs) if *in_order => Some(sink.sink(NearSpansOrdered::new(*slop, subs))),
            Some(subs) => Some(sink.sink(NearSpansUnordered::new(*slop, subs))),
        },
        SpanNode::Or { clauses } => {
            let mut subs = Vec::with_capacity(clauses.len());
            for c in clauses {
                if let Some(s) = sub(c)? {
                    subs.push(s);
                }
            }
            match subs.len() {
                0 => None,
                1 => subs.pop().map(|s| sink.sink_boxed(s)),
                _ => Some(sink.sink_boxed(BoxSpans::boxed(OrSpans::new(subs)))),
            }
        }
        SpanNode::First { inner, end } => sub(inner)?.map(|s| sink.sink(range(s, 0, *end))),
        SpanNode::PositionRange { inner, start, end } => {
            sub(inner)?.map(|s| sink.sink(range(s, *start, *end)))
        }
        SpanNode::Not {
            include,
            exclude,
            pre,
            post,
        } => {
            let Some(inc) = sub(include)? else {
                return Ok(None);
            };
            match sub(exclude)? {
                None => Some(sink.sink_boxed(inc)),
                Some(exc) => Some(sink.sink(FilterSpans::new(
                    inc,
                    NotFilter {
                        exclude: exc,
                        pre: *pre,
                        post: *post,
                        last_approx_doc: -1,
                        last_approx_result: false,
                    },
                ))),
            }
        }
        SpanNode::Containing { big, little } | SpanNode::Within { big, little } => {
            let Some(b) = sub(big)? else {
                return Ok(None);
            };
            let Some(l) = sub(little)? else {
                return Ok(None);
            };
            Some(sink.sink(ContainSpans::new(
                b,
                l,
                matches!(q, SpanNode::Containing { .. }),
            )))
        }
        SpanNode::FieldMasking { inner, .. } => sub(inner)?.map(|s| sink.sink_boxed(s)),
        // `requiredPostings.atLeast(PAYLOADS)` for both payload queries.
        SpanNode::PayloadCheck(p) => {
            spans_with(ctx, &p.inner, true)?.map(|s| sink.sink_boxed(payloads::check_spans(p, s)))
        }
        // Inside another span query a payload score query's weight is its
        // inner query's (`PayloadSpanWeight.getSpans`).
        SpanNode::PayloadScore(p) => spans_with(ctx, &p.inner, true)?.map(|s| sink.sink_boxed(s)),
        SpanNode::MultiTerm(_) => {
            return Err(Error::IllegalArgument("Rewrite first!".into()));
        }
    })
}

/// A near query's sub-spans, `None` when the field or a clause has none.
///
/// # Errors
/// Fewer than two clauses (`"Less than 2 subSpans.size()"`).
fn near_subs<'a>(
    ctx: &LeafContext<'a>,
    q: &SpanNode,
    clauses: &[SpanNode],
    payloads: bool,
) -> Result<Option<Vec<BoxSpans<'a>>>> {
    let Some(field) = q.field() else {
        return Ok(None);
    };
    if ctx.fields.field(field).is_none() {
        return Ok(None);
    }
    let mut subs = Vec::with_capacity(clauses.len());
    for c in clauses {
        match spans_with(ctx, c, payloads)? {
            Some(s) => subs.push(s),
            None => return Ok(None),
        }
    }
    if subs.len() < 2 {
        return Err(Error::IllegalArgument(format!(
            "Less than 2 subSpans.size():{}",
            subs.len()
        )));
    }
    Ok(Some(subs))
}

/// `getTermStates(weights)` as `buildSimWeight` sees them: the `(field,
/// term)`s whose statistics the query's similarity takes (`extractTermStates`:
/// a not query's include side, both sides of a containment).
pub(crate) fn weight_terms(q: &SpanNode, out: &mut Vec<(String, Vec<u8>)>) {
    match q {
        SpanNode::Term { field, term } => out.push((field.clone(), term.clone())),
        SpanNode::Near { clauses, .. } | SpanNode::Or { clauses } => {
            for c in clauses {
                weight_terms(c, out);
            }
        }
        SpanNode::First { inner, .. }
        | SpanNode::PositionRange { inner, .. }
        | SpanNode::FieldMasking { inner, .. } => weight_terms(inner, out),
        SpanNode::PayloadCheck(p) => weight_terms(&p.inner, out),
        SpanNode::PayloadScore(p) => weight_terms(&p.inner, out),
        SpanNode::Not { include, .. } => weight_terms(include, out),
        SpanNode::Containing { big, little } | SpanNode::Within { big, little } => {
            weight_terms(big, out);
            weight_terms(little, out);
        }
        SpanNode::MultiTerm(_) => {}
    }
}

/// Every term a query reads, for the statistics pass: [`weight_terms`] and
/// a not query's exclude side, whose own weight scores nothing but is built
/// all the same.
pub(crate) fn all_terms(q: &SpanNode, out: &mut Vec<(String, Vec<u8>)>) {
    match q {
        SpanNode::Not {
            include, exclude, ..
        } => {
            all_terms(include, out);
            all_terms(exclude, out);
        }
        SpanNode::Near { clauses, .. } | SpanNode::Or { clauses } => {
            for c in clauses {
                all_terms(c, out);
            }
        }
        SpanNode::First { inner, .. }
        | SpanNode::PositionRange { inner, .. }
        | SpanNode::FieldMasking { inner, .. } => all_terms(inner, out),
        SpanNode::PayloadCheck(p) => all_terms(&p.inner, out),
        SpanNode::PayloadScore(p) => all_terms(&p.inner, out),
        SpanNode::Containing { big, little } | SpanNode::Within { big, little } => {
            all_terms(big, out);
            all_terms(little, out);
        }
        other => weight_terms(other, out),
    }
}

/// The query a span query's weight is created from: Java's `createWeight`
/// hands a `FieldMaskingSpanQuery`'s to the masked query.
pub(crate) fn weight_query(q: &SpanNode) -> &SpanNode {
    match q {
        SpanNode::FieldMasking { inner, .. } => weight_query(inner),
        other => other,
    }
}

/// `SpanScorer.setFreqCurrentDoc()`: `sum(1 / (1 + width))` over the
/// document's spans, each added in `double` to the `float` sum.
pub(crate) fn sloppy_freq<S: Spans + ?Sized>(spans: &mut S) -> Result<f32> {
    let mut freq = 0.0f32;
    spans.do_start_current_doc();
    let mut start = spans.next_start_position()?;
    while start != NO_MORE_POSITIONS {
        freq = (f64::from(freq) + 1.0 / (1.0 + f64::from(spans.width()))) as f32;
        spans.do_current_spans()?;
        start = spans.next_start_position()?;
    }
    Ok(freq)
}

// ---------------------------------------------------------------------------
// The queries
// ---------------------------------------------------------------------------

/// A span query of `lucene-queries`' `spans` and `payloads` packages, as a
/// tree: each variant is one Java class.
#[derive(Debug, Clone, PartialEq)]
pub enum SpanNode {
    /// `SpanTermQuery`.
    Term { field: String, term: Vec<u8> },
    /// `SpanNearQuery`.
    Near {
        clauses: Vec<SpanNode>,
        slop: i32,
        in_order: bool,
    },
    /// `SpanOrQuery`.
    Or { clauses: Vec<SpanNode> },
    /// `SpanFirstQuery`: the spans ending at or before `end`.
    First { inner: Box<SpanNode>, end: i32 },
    /// `SpanPositionRangeQuery`: the spans in `[start, end)`.
    PositionRange {
        inner: Box<SpanNode>,
        start: i32,
        end: i32,
    },
    /// `SpanNotQuery`: `include`'s spans with no `exclude` span within `pre`
    /// positions before or `post` after.
    Not {
        include: Box<SpanNode>,
        exclude: Box<SpanNode>,
        pre: i32,
        post: i32,
    },
    /// `SpanContainingQuery`: `big`'s spans that contain one of `little`'s.
    Containing {
        big: Box<SpanNode>,
        little: Box<SpanNode>,
    },
    /// `SpanWithinQuery`: `little`'s spans within one of `big`'s.
    Within {
        big: Box<SpanNode>,
        little: Box<SpanNode>,
    },
    /// `FieldMaskingSpanQuery`: `inner`, reporting `field` as its own.
    FieldMasking { inner: Box<SpanNode>, field: String },
    /// `SpanMultiTermQueryWrapper`: the terms a multi-term query matches,
    /// as a span disjunction once the searcher rewrites it.
    MultiTerm(MultiTermQuery),
    /// `SpanPayloadCheckQuery`.
    PayloadCheck(Box<payloads::SpanPayloadCheckQuery>),
    /// `PayloadScoreQuery`.
    PayloadScore(Box<payloads::PayloadScoreQuery>),
}

impl SpanNode {
    /// `new SpanTermQuery(new Term(field, term))`.
    pub fn term(field: impl Into<String>, term: impl Into<Vec<u8>>) -> Self {
        SpanNode::Term {
            field: field.into(),
            term: term.into(),
        }
    }

    /// `new SpanNearQuery(clauses, slop, inOrder)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for clauses of different fields, as
    /// Java's constructor throws.
    pub fn near(clauses: Vec<SpanNode>, slop: i32, in_order: bool) -> Result<Self> {
        same_field(&clauses)?;
        Ok(SpanNode::Near {
            clauses,
            slop,
            in_order,
        })
    }

    /// `new SpanOrQuery(clauses)`.
    ///
    /// # Errors
    /// As [`Self::near`].
    pub fn or(clauses: Vec<SpanNode>) -> Result<Self> {
        same_field(&clauses)?;
        Ok(SpanNode::Or { clauses })
    }

    /// `new SpanFirstQuery(match, end)`.
    pub fn first(inner: SpanNode, end: i32) -> Self {
        SpanNode::First {
            inner: Box::new(inner),
            end,
        }
    }

    /// `new SpanPositionRangeQuery(match, start, end)`.
    pub fn position_range(inner: SpanNode, start: i32, end: i32) -> Self {
        SpanNode::PositionRange {
            inner: Box::new(inner),
            start,
            end,
        }
    }

    /// `new SpanNotQuery(include, exclude, pre, post)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when the two are of different fields.
    /// Negative distances allow that much overlap, as Java's do.
    pub fn not(include: SpanNode, exclude: SpanNode, pre: i32, post: i32) -> Result<Self> {
        if include.field().is_some()
            && exclude.field().is_some()
            && include.field() != exclude.field()
        {
            return Err(Error::IllegalArgument(
                "Clauses must have same field.".into(),
            ));
        }
        Ok(SpanNode::Not {
            include: Box::new(include),
            exclude: Box::new(exclude),
            pre,
            post,
        })
    }

    /// `new SpanNotQuery(include, exclude, dist, dist)`.
    ///
    /// # Errors
    /// As [`Self::not`].
    pub fn not_within(include: SpanNode, exclude: SpanNode, dist: i32) -> Result<Self> {
        Self::not(include, exclude, dist, dist)
    }

    /// `new SpanContainingQuery(big, little)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] when the two are of different fields.
    pub fn containing(big: SpanNode, little: SpanNode) -> Result<Self> {
        contain_fields(&big, &little)?;
        Ok(SpanNode::Containing {
            big: Box::new(big),
            little: Box::new(little),
        })
    }

    /// `new SpanWithinQuery(big, little)`.
    ///
    /// # Errors
    /// As [`Self::containing`].
    pub fn within(big: SpanNode, little: SpanNode) -> Result<Self> {
        contain_fields(&big, &little)?;
        Ok(SpanNode::Within {
            big: Box::new(big),
            little: Box::new(little),
        })
    }

    /// `new FieldMaskingSpanQuery(maskedQuery, maskedField)`.
    pub fn field_masking(inner: SpanNode, field: impl Into<String>) -> Self {
        SpanNode::FieldMasking {
            inner: Box::new(inner),
            field: field.into(),
        }
    }

    /// `new SpanMultiTermQueryWrapper(query)`.
    pub fn multi_term(query: MultiTermQuery) -> Self {
        SpanNode::MultiTerm(query)
    }

    /// `SpanQuery.getField()`.
    pub fn field(&self) -> Option<&str> {
        match self {
            SpanNode::Term { field, .. } | SpanNode::FieldMasking { field, .. } => Some(field),
            SpanNode::Near { clauses, .. } | SpanNode::Or { clauses } => {
                clauses.first().and_then(SpanNode::field)
            }
            SpanNode::First { inner, .. } | SpanNode::PositionRange { inner, .. } => inner.field(),
            SpanNode::Not { include, .. } => include.field(),
            SpanNode::Containing { big, .. } | SpanNode::Within { big, .. } => big.field(),
            SpanNode::MultiTerm(m) => Some(m.field()),
            SpanNode::PayloadCheck(p) => p.inner.field(),
            SpanNode::PayloadScore(p) => p.inner.field(),
        }
    }

    /// Whether the query must be rewritten against the searcher first: it
    /// holds a [`SpanNode::MultiTerm`].
    pub fn needs_rewrite(&self) -> bool {
        self.children().into_iter().any(SpanNode::needs_rewrite)
            || matches!(self, SpanNode::MultiTerm(_))
    }

    fn children(&self) -> Vec<&SpanNode> {
        match self {
            SpanNode::Term { .. } | SpanNode::MultiTerm(_) => Vec::new(),
            SpanNode::Near { clauses, .. } | SpanNode::Or { clauses } => clauses.iter().collect(),
            SpanNode::First { inner, .. }
            | SpanNode::PositionRange { inner, .. }
            | SpanNode::FieldMasking { inner, .. } => {
                vec![inner]
            }
            SpanNode::Not {
                include, exclude, ..
            } => vec![include, exclude],
            SpanNode::Containing { big, little } | SpanNode::Within { big, little } => {
                vec![big, little]
            }
            SpanNode::PayloadCheck(p) => vec![&p.inner],
            SpanNode::PayloadScore(p) => vec![&p.inner],
        }
    }

    /// `rewrite(indexSearcher)`, to the fixed point: each
    /// [`SpanNode::MultiTerm`] becomes the span disjunction of the terms it
    /// matches (`SCORING_SPAN_QUERY_REWRITE`, every term; or
    /// `TopTermsSpanBooleanQueryRewrite`, the first `size` of them for a
    /// top-terms rewrite method), in term order.
    ///
    /// # Errors
    /// What reading the term dictionaries reports.
    pub fn rewrite(&self, searcher: &IndexSearcher<'_, '_>) -> Result<SpanNode> {
        let b = |n: &SpanNode| -> Result<Box<SpanNode>> { Ok(Box::new(n.rewrite(searcher)?)) };
        Ok(match self {
            SpanNode::Term { .. } => self.clone(),
            SpanNode::Near {
                clauses,
                slop,
                in_order,
            } => SpanNode::Near {
                clauses: clauses
                    .iter()
                    .map(|c| c.rewrite(searcher))
                    .collect::<Result<_>>()?,
                slop: *slop,
                in_order: *in_order,
            },
            SpanNode::Or { clauses } => SpanNode::Or {
                clauses: clauses
                    .iter()
                    .map(|c| c.rewrite(searcher))
                    .collect::<Result<_>>()?,
            },
            SpanNode::First { inner, end } => SpanNode::First {
                inner: b(inner)?,
                end: *end,
            },
            SpanNode::PositionRange { inner, start, end } => SpanNode::PositionRange {
                inner: b(inner)?,
                start: *start,
                end: *end,
            },
            SpanNode::Not {
                include,
                exclude,
                pre,
                post,
            } => SpanNode::Not {
                include: b(include)?,
                exclude: b(exclude)?,
                pre: *pre,
                post: *post,
            },
            SpanNode::Containing { big, little } => SpanNode::Containing {
                big: b(big)?,
                little: b(little)?,
            },
            SpanNode::Within { big, little } => SpanNode::Within {
                big: b(big)?,
                little: b(little)?,
            },
            SpanNode::FieldMasking { inner, field } => SpanNode::FieldMasking {
                inner: b(inner)?,
                field: field.clone(),
            },
            SpanNode::MultiTerm(m) => rewrite_multi_term(searcher, m)?,
            SpanNode::PayloadCheck(p) => {
                let mut p = (**p).clone();
                p.inner = p.inner.rewrite(searcher)?;
                SpanNode::PayloadCheck(Box::new(p))
            }
            SpanNode::PayloadScore(p) => {
                let mut p = (**p).clone();
                p.inner = p.inner.rewrite(searcher)?;
                SpanNode::PayloadScore(Box::new(p))
            }
        })
    }
}

/// `SpanNearQuery`/`SpanOrQuery`'s constructor check: every clause of the
/// first's field.
fn same_field(clauses: &[SpanNode]) -> Result<()> {
    let mut field: Option<&str> = None;
    for c in clauses {
        let f = c.field();
        match field {
            None => field = f,
            Some(first) => {
                if f.is_some_and(|f| f != first) {
                    return Err(Error::IllegalArgument(
                        "Clauses must have same field.".into(),
                    ));
                }
            }
        }
    }
    Ok(())
}

/// `SpanContainQuery`'s constructor check.
fn contain_fields(big: &SpanNode, little: &SpanNode) -> Result<()> {
    if big.field() != little.field() {
        return Err(Error::IllegalArgument(
            "big and little not same field".into(),
        ));
    }
    Ok(())
}

/// `SpanMultiTermQueryWrapper.rewrite`: the terms of every segment, in
/// term order, as a `SpanOrQuery` of `SpanTermQuery`s.
fn rewrite_multi_term(searcher: &IndexSearcher<'_, '_>, m: &MultiTermQuery) -> Result<SpanNode> {
    use crate::extended_query::RewriteMethod;
    // `selectRewriteMethod`: a top-terms method keeps its size.
    let limit = match m.rewrite {
        RewriteMethod::TopTermsScoringBoolean(n)
        | RewriteMethod::TopTermsBoostOnlyBoolean(n)
        | RewriteMethod::TopTermsBlendedFreqScoring(n) => Some(n),
        _ => None,
    };
    let mut terms: std::collections::BTreeSet<Vec<u8>> = std::collections::BTreeSet::new();
    for seg in searcher.segments() {
        for (t, _) in crate::exec::extended::expand_terms(seg.fields, &m.source, limit)? {
            terms.insert(t);
        }
    }
    let field = m.field().to_string();
    let mut clauses: Vec<SpanNode> = terms
        .into_iter()
        .map(|t| SpanNode::term(field.clone(), t))
        .collect();
    if let Some(n) = limit {
        clauses.truncate(n);
    }
    Ok(SpanNode::Or { clauses })
}

impl std::fmt::Display for SpanNode {
    /// `toString()` (`toString("")`): a term as `field:text`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            SpanNode::Term { field, term } => {
                write!(f, "{field}:{}", String::from_utf8_lossy(term))
            }
            SpanNode::Near {
                clauses,
                slop,
                in_order,
            } => {
                f.write_str("spanNear([")?;
                join(f, clauses)?;
                write!(f, "], {slop}, {in_order})")
            }
            SpanNode::Or { clauses } => {
                f.write_str("spanOr([")?;
                join(f, clauses)?;
                f.write_str("])")
            }
            SpanNode::First { inner, end } => write!(f, "spanFirst({inner}, {end})"),
            SpanNode::PositionRange { inner, start, end } => {
                write!(f, "spanPosRange({inner}, {start}, {end})")
            }
            SpanNode::Not {
                include,
                exclude,
                pre,
                post,
            } => write!(f, "spanNot({include}, {exclude}, {pre}, {post})"),
            SpanNode::Containing { big, little } => write!(f, "SpanContaining({big}, {little})"),
            SpanNode::Within { big, little } => write!(f, "SpanWithin({big}, {little})"),
            SpanNode::FieldMasking { inner, field } => write!(f, "mask({inner}) as {field}"),
            SpanNode::MultiTerm(m) => write!(
                f,
                "SpanMultiTermQueryWrapper({})",
                multi_term_string(&m.source)
            ),
            SpanNode::PayloadCheck(p) => p.fmt(f),
            SpanNode::PayloadScore(p) => p.fmt(f),
        }
    }
}

/// The wrapped multi-term query's `toString()`.
fn multi_term_string(source: &crate::extended_query::MultiTermSource) -> String {
    use crate::extended_query::MultiTermSource as S;
    let text = |b: &[u8]| String::from_utf8_lossy(b).into_owned();
    // `TermRangeQuery.toString`: an open end is `*`, a `*` term escaped.
    let bound = |b: &Option<Vec<u8>>| match b {
        None => "*".to_string(),
        Some(t) => match term_to_string(t) {
            s if s == "*" => "\\*".to_string(),
            s => s,
        },
    };
    match source {
        S::Prefix(q) => format!("{}:{}*", q.field, text(&q.prefix)),
        S::Wildcard(q) => format!("{}:{}", q.field, text(&q.pattern)),
        S::Regexp(q) => format!("{}:/{}/", q.field, q.pattern),
        S::TermRange(q) => format!(
            "{}:{}{} TO {}{}",
            q.field,
            if q.include_lower { '[' } else { '{' },
            bound(&q.lower),
            bound(&q.upper),
            if q.include_upper { ']' } else { '}' }
        ),
        S::Automaton(_) => "AutomatonQuery".to_string(),
        S::TermSet(q) => format!("TermsQuery{{field={}}}", q.field),
    }
}

/// `Term.toString(BytesRef)`: the bytes as UTF-8 text when they are valid
/// UTF-8, else `BytesRef.toString()`'s hex (`[61 ff]`).
pub(crate) fn term_to_string(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(_) => {
            let hex: Vec<String> = bytes.iter().map(|b| format!("{b:x}")).collect();
            format!("[{}]", hex.join(" "))
        }
    }
}

fn join(f: &mut std::fmt::Formatter<'_>, clauses: &[SpanNode]) -> std::fmt::Result {
    for (i, c) in clauses.iter().enumerate() {
        if i > 0 {
            f.write_str(", ")?;
        }
        write!(f, "{c}")?;
    }
    Ok(())
}

impl From<&crate::query::SpanQuery> for SpanNode {
    /// The core span queries' tree in this one's terms.
    fn from(q: &crate::query::SpanQuery) -> Self {
        use crate::query::SpanQuery as Q;
        match q {
            Q::SpanTerm { field, term } => SpanNode::term(field.clone(), term.clone()),
            Q::SpanNear {
                clauses,
                slop,
                in_order,
            } => SpanNode::Near {
                clauses: clauses.iter().map(SpanNode::from).collect(),
                slop: i32::try_from(*slop).unwrap_or(i32::MAX),
                in_order: *in_order,
            },
            Q::SpanOr { clauses } => SpanNode::Or {
                clauses: clauses.iter().map(SpanNode::from).collect(),
            },
        }
    }
}

impl From<SpanNode> for crate::query::Clause {
    fn from(q: SpanNode) -> Self {
        crate::query::Clause::Extended(Box::new(crate::extended_query::ExtendedQuery::Span(q)))
    }
}

#[cfg(test)]
mod tests;
