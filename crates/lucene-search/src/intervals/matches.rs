//! `IntervalsSource.matches(field, ctx, doc)` and `IntervalQuery`'s
//! `Weight.matches`: `IntervalMatchesIterator` and its implementations
//! (`IntervalMatches.asMatches`/`wrapMatches`, `CachingMatchesIterator`,
//! `ConjunctionMatchesIterator`, the disjunction's, the repeat's and the
//! minimum-should-match's), over the same iterators the scorer runs.
//!
//! Java threads one `MatchesIterator` through two owners: the interval
//! iterator that wraps it (`wrapMatches`) advances it, and the matches
//! iterator built over that reads its offsets and sub-matches. Here the
//! shared iterator is an `Rc<RefCell<..>>` held by both.
//!
//! A document's occurrences of every term the source names (and every term
//! a multi-term source expands to) are read once, when the matches are
//! asked for, so the `Matches` own them; each `getMatches` builds the
//! iterators afresh over them, as Java's supplier does.
//!
//! `getQuery()` throws `UnsupportedOperationException` (or returns `null`)
//! for a conjunction's, a repeat's and a minimum-should-match's iterator in
//! Java; here those report the enclosing [`IntervalQuery`], since
//! [`MatchesIterator::query`] always has one.

use std::cell::RefCell;
use std::collections::HashMap;
use std::rc::Rc;
use std::sync::Arc;

use lucene_codecs::postings::Position;

use super::iterators::{self, BoxIntervals, IntervalIterator, MatchCallback, RelativeKind};
use super::{IntervalQuery, IntervalsSource, PayloadFilter, NO_MORE_INTERVALS};
use crate::exec::NO_MORE_DOCS;
use crate::matches::{disjunction, for_field, BoxMatches, BoxMatchesIterator, MatchesIterator};
use crate::multi_segment::OpenSegment;
use crate::query::{Clause, TermQuery};
use crate::{Error, Result};

/// `IntervalMatchesIterator`: a `MatchesIterator` that also knows its
/// match's `gaps()` and `width()`.
pub(crate) trait IntervalMatchesIterator {
    fn next(&mut self) -> Result<bool>;
    fn start_position(&self) -> i32;
    fn end_position(&self) -> i32;
    fn start_offset(&self) -> i32;
    fn end_offset(&self) -> i32;
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>>;
    /// `getQuery()`: an error where Java throws
    /// `UnsupportedOperationException`.
    fn query(&self) -> Result<Arc<Clause>>;
    fn gaps(&self) -> i32;
    fn width(&self) -> i32;
}

type Shared = Rc<RefCell<dyn IntervalMatchesIterator>>;

fn shared(mi: impl IntervalMatchesIterator + 'static) -> Shared {
    Rc::new(RefCell::new(mi))
}

/// A shared interval-matches iterator seen as a plain [`MatchesIterator`]
/// (what `FilterMatchesIterator` over it is in Java).
struct AsMatchesIterator {
    mi: Shared,
    query: Arc<Clause>,
}

impl MatchesIterator for AsMatchesIterator {
    fn next(&mut self) -> Result<bool> {
        self.mi.borrow_mut().next()
    }
    fn start_position(&self) -> i32 {
        self.mi.borrow().start_position()
    }
    fn end_position(&self) -> i32 {
        self.mi.borrow().end_position()
    }
    fn start_offset(&self) -> i32 {
        self.mi.borrow().start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.mi.borrow().end_offset()
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        self.mi.borrow_mut().sub_matches()
    }
    fn query(&self) -> &Clause {
        &self.query
    }
}

/// `mi` as a plain iterator; its query, or `fallback` where it has none.
fn as_plain(mi: Shared, fallback: &Arc<Clause>) -> BoxMatchesIterator {
    let query = mi.borrow().query().unwrap_or_else(|_| Arc::clone(fallback));
    Box::new(AsMatchesIterator { mi, query })
}

fn unsupported(what: &str) -> Error {
    Error::Unsupported(format!("{what} has no query"))
}

/// `ConjunctionMatchesIterator.SingletonMatchesIterator`: the sub-iterator's
/// current match, once.
struct Singleton {
    mi: Shared,
    query: Arc<Clause>,
    exhausted: bool,
}

impl MatchesIterator for Singleton {
    fn next(&mut self) -> Result<bool> {
        if self.exhausted {
            return Ok(false);
        }
        self.exhausted = true;
        Ok(true)
    }
    fn start_position(&self) -> i32 {
        self.mi.borrow().start_position()
    }
    fn end_position(&self) -> i32 {
        self.mi.borrow().end_position()
    }
    fn start_offset(&self) -> i32 {
        self.mi.borrow().start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.mi.borrow().end_offset()
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        self.mi.borrow_mut().sub_matches()
    }
    fn query(&self) -> &Clause {
        &self.query
    }
}

/// `sub.getSubMatches()`, else a singleton of `sub`'s current match.
fn sub_or_singleton(mi: &Shared, fallback: &Arc<Clause>) -> Result<BoxMatchesIterator> {
    let sub = mi.borrow_mut().sub_matches()?;
    Ok(match sub {
        Some(s) => s,
        None => {
            let query = mi.borrow().query().unwrap_or_else(|_| Arc::clone(fallback));
            Box::new(Singleton {
                mi: Rc::clone(mi),
                query,
                exhausted: false,
            })
        }
    })
}

// ---------------------------------------------------------------------------
// The leaves
// ---------------------------------------------------------------------------

/// `TermIntervalsSource.matches(te, doc, field)` and
/// `PayloadFilteredTermIntervalsSource.matches`' iterators over a term's
/// occurrences in the document.
struct TermMatches {
    occurrences: Arc<[Position]>,
    upto: usize,
    at: usize,
    pos: i32,
    filter: Option<PayloadFilter>,
    query: Arc<Clause>,
}

impl TermMatches {
    fn current(&self) -> Option<&Position> {
        self.at.checked_sub(1).and_then(|i| self.occurrences.get(i))
    }
}

impl IntervalMatchesIterator for TermMatches {
    fn next(&mut self) -> Result<bool> {
        loop {
            if self.upto == 0 {
                self.pos = NO_MORE_INTERVALS;
                return Ok(false);
            }
            self.upto -= 1;
            let occ = self.occurrences.get(self.at);
            self.at += 1;
            self.pos = occ.map_or(-1, |o| o.position);
            match &self.filter {
                None => return Ok(true),
                Some(f) => {
                    let payload = occ.map(|o| o.payload.as_slice()).filter(|p| !p.is_empty());
                    if f.test(payload) {
                        return Ok(true);
                    }
                }
            }
        }
    }
    fn start_position(&self) -> i32 {
        self.pos
    }
    fn end_position(&self) -> i32 {
        self.pos
    }
    fn start_offset(&self) -> i32 {
        self.current().map_or(-1, |o| o.start_offset)
    }
    fn end_offset(&self) -> i32 {
        self.current().map_or(-1, |o| o.end_offset)
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        Ok(None)
    }
    fn query(&self) -> Result<Arc<Clause>> {
        match &self.filter {
            None => Ok(Arc::clone(&self.query)),
            Some(_) => Err(unsupported("a payload-filtered term's match")),
        }
    }
    fn gaps(&self) -> i32 {
        0
    }
    fn width(&self) -> i32 {
        1
    }
}

/// `MultiTermIntervalsSource.matches`' iterator: the disjunction of the
/// expanded terms' matches.
struct MultiTermMatches {
    mi: BoxMatchesIterator,
}

impl IntervalMatchesIterator for MultiTermMatches {
    fn next(&mut self) -> Result<bool> {
        self.mi.next()
    }
    fn start_position(&self) -> i32 {
        self.mi.start_position()
    }
    fn end_position(&self) -> i32 {
        self.mi.end_position()
    }
    fn start_offset(&self) -> i32 {
        self.mi.start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.mi.end_offset()
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        self.mi.sub_matches()
    }
    fn query(&self) -> Result<Arc<Clause>> {
        Ok(Arc::new(self.mi.query().clone()))
    }
    fn gaps(&self) -> i32 {
        0
    }
    fn width(&self) -> i32 {
        1
    }
}

// ---------------------------------------------------------------------------
// IntervalMatches.wrapMatches / asMatches
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum WrapState {
    Unpositioned,
    Iterating,
    NoMoreIntervals,
    Exhausted,
}

/// `IntervalMatches.wrapMatches(mi, doc)`: a matches iterator as an
/// interval iterator over its one document.
struct WrapMatches {
    mi: Shared,
    doc: i32,
    state: WrapState,
}

impl IntervalIterator for WrapMatches {
    // SENTINEL: `-1` = "not yet positioned", `DocIdSetIterator`'s own
    // unpositioned doc id; callers advance before reading a document.
    fn doc_id(&self) -> i32 {
        match self.state {
            WrapState::Unpositioned => -1,
            WrapState::Iterating | WrapState::NoMoreIntervals => self.doc,
            WrapState::Exhausted => NO_MORE_DOCS,
        }
    }
    fn next_doc(&mut self) -> Result<i32> {
        Ok(match self.state {
            WrapState::Unpositioned => {
                self.state = WrapState::Iterating;
                self.doc
            }
            WrapState::Iterating | WrapState::NoMoreIntervals => {
                self.state = WrapState::Exhausted;
                NO_MORE_DOCS
            }
            WrapState::Exhausted => NO_MORE_DOCS,
        })
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        if target == self.doc {
            self.state = WrapState::Iterating;
            return Ok(self.doc);
        }
        self.state = WrapState::Exhausted;
        Ok(NO_MORE_DOCS)
    }
    fn cost(&self) -> i64 {
        1
    }
    fn start(&self) -> i32 {
        if self.state == WrapState::NoMoreIntervals {
            return NO_MORE_INTERVALS;
        }
        self.mi.borrow().start_position()
    }
    fn end(&self) -> i32 {
        if self.state == WrapState::NoMoreIntervals {
            return NO_MORE_INTERVALS;
        }
        self.mi.borrow().end_position()
    }
    fn gaps(&self) -> i32 {
        self.mi.borrow().gaps()
    }
    fn width(&self) -> i32 {
        self.mi.borrow().width()
    }
    fn next_interval(&mut self) -> Result<i32> {
        if self.mi.borrow_mut().next()? {
            return Ok(self.mi.borrow().start_position());
        }
        self.state = WrapState::NoMoreIntervals;
        Ok(NO_MORE_INTERVALS)
    }
    fn match_cost(&self) -> f32 {
        1.0
    }
}

fn wrap(mi: &Shared, doc: i32) -> BoxIntervals<'static> {
    BoxIntervals::boxed(WrapMatches {
        mi: Rc::clone(mi),
        doc,
        state: WrapState::Unpositioned,
    })
}

/// `IntervalMatches.asMatches(iterator, source, doc)`'s iterator: the
/// interval iterator's positions, `source`'s offsets and sub-matches.
struct AsMatches {
    it: BoxIntervals<'static>,
    source: Shared,
    cached: bool,
}

impl IntervalMatchesIterator for AsMatches {
    fn next(&mut self) -> Result<bool> {
        if self.cached {
            self.cached = false;
            return Ok(true);
        }
        Ok(self.it.next_interval()? != NO_MORE_INTERVALS)
    }
    fn start_position(&self) -> i32 {
        self.it.start()
    }
    fn end_position(&self) -> i32 {
        self.it.end()
    }
    fn start_offset(&self) -> i32 {
        self.source.borrow().start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.source.borrow().end_offset()
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        self.source.borrow_mut().sub_matches()
    }
    fn query(&self) -> Result<Arc<Clause>> {
        self.source.borrow().query()
    }
    fn gaps(&self) -> i32 {
        self.it.gaps()
    }
    fn width(&self) -> i32 {
        self.it.width()
    }
}

/// `IntervalMatches.asMatches`: `None` unless the iterator has an interval
/// on `doc`.
fn as_matches(mut it: BoxIntervals<'static>, source: Shared, doc: i32) -> Result<Option<Shared>> {
    if it.advance(doc)? != doc {
        return Ok(None);
    }
    if it.next_interval()? == NO_MORE_INTERVALS {
        return Ok(None);
    }
    Ok(Some(shared(AsMatches {
        it,
        source,
        cached: true,
    })))
}

// ---------------------------------------------------------------------------
// CachingMatchesIterator / ConjunctionMatchesIterator
// ---------------------------------------------------------------------------

/// `CachingMatchesIterator`: a sub-iterator whose sub-matches are copied
/// when its minimizing parent settles on a match (`cache()`), since the
/// minimization moves it on.
struct CachingMatches {
    inner: Shared,
    /// `[startPosition, endPosition, startOffset, endOffset]` per cached
    /// sub-match, with its query.
    cached: Vec<([i32; 4], Arc<Clause>)>,
}

impl CachingMatches {
    fn cache(&mut self) -> Result<()> {
        self.cached.clear();
        let mi = self.inner.borrow_mut().sub_matches()?;
        match mi {
            None => {
                let inner = self.inner.borrow();
                self.cached.push((
                    [
                        inner.start_position(),
                        inner.end_position(),
                        inner.start_offset(),
                        inner.end_offset(),
                    ],
                    inner.query()?,
                ));
            }
            Some(mut mi) => {
                while mi.next()? {
                    self.cached.push((
                        [
                            mi.start_position(),
                            mi.end_position(),
                            mi.start_offset(),
                            mi.end_offset(),
                        ],
                        Arc::new(mi.query().clone()),
                    ));
                }
            }
        }
        Ok(())
    }
}

impl IntervalMatchesIterator for CachingMatches {
    fn next(&mut self) -> Result<bool> {
        self.inner.borrow_mut().next()
    }
    fn start_position(&self) -> i32 {
        self.inner.borrow().start_position()
    }
    fn end_position(&self) -> i32 {
        self.inner.borrow().end_position()
    }
    fn start_offset(&self) -> i32 {
        self.cached.first().map_or(-1, |c| c.0[2])
    }
    fn end_offset(&self) -> i32 {
        self.cached.last().map_or(-1, |c| c.0[3])
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        Ok(Some(Box::new(CachedSubMatches {
            cached: self.cached.clone(),
            upto: 0,
        })))
    }
    fn query(&self) -> Result<Arc<Clause>> {
        match self.cached.first() {
            Some(c) => Ok(Arc::clone(&c.1)),
            None => self.inner.borrow().query(),
        }
    }
    fn gaps(&self) -> i32 {
        self.inner.borrow().gaps()
    }
    fn width(&self) -> i32 {
        self.inner.borrow().width()
    }
}

/// `CachingMatchesIterator.getSubMatches()`'s iterator over the cache.
struct CachedSubMatches {
    cached: Vec<([i32; 4], Arc<Clause>)>,
    upto: usize,
}

impl CachedSubMatches {
    fn current(&self) -> [i32; 4] {
        self.upto
            .checked_sub(1)
            .and_then(|i| self.cached.get(i))
            .map_or([-1; 4], |c| c.0)
    }
}

impl MatchesIterator for CachedSubMatches {
    fn next(&mut self) -> Result<bool> {
        self.upto += 1;
        Ok(self.upto <= self.cached.len())
    }
    fn start_position(&self) -> i32 {
        self.current()[0]
    }
    fn end_position(&self) -> i32 {
        self.current()[1]
    }
    fn start_offset(&self) -> i32 {
        self.current()[2]
    }
    fn end_offset(&self) -> i32 {
        self.current()[3]
    }
    fn query(&self) -> &Clause {
        let i = self
            .upto
            .saturating_sub(1)
            .min(self.cached.len().saturating_sub(1));
        &self.cached[i].1
    }
}

/// `MinimizingConjunctionIntervalsSource.cacheIterators`.
fn cache_iterators(subs: Vec<Rc<RefCell<CachingMatches>>>) -> MatchCallback<'static> {
    Some(Box::new(move || {
        for s in &subs {
            s.borrow_mut().cache()?;
        }
        Ok(())
    }))
}

/// `ConjunctionMatchesIterator`: the combined iterator's positions, the
/// sub-iterators' offsets and sub-matches.
struct ConjunctionMatches {
    it: BoxIntervals<'static>,
    subs: Vec<Shared>,
    cached: bool,
    query: Arc<Clause>,
}

impl IntervalMatchesIterator for ConjunctionMatches {
    fn next(&mut self) -> Result<bool> {
        if self.cached {
            self.cached = false;
            return Ok(true);
        }
        Ok(self.it.next_interval()? != NO_MORE_INTERVALS)
    }
    fn start_position(&self) -> i32 {
        self.it.start()
    }
    fn end_position(&self) -> i32 {
        self.it.end()
    }
    // SENTINEL: `-1` = "no offsets", `MatchesIterator`'s own contract,
    // reported to the caller as Java reports it.
    fn start_offset(&self) -> i32 {
        let mut start = i32::MAX;
        for s in &self.subs {
            let v = s.borrow().start_offset();
            if v == -1 {
                return -1;
            }
            start = start.min(v);
        }
        start
    }
    // SENTINEL: `-1` = "no offsets", `MatchesIterator`'s own contract,
    // reported to the caller as Java reports it.
    fn end_offset(&self) -> i32 {
        let mut end = -1;
        for s in &self.subs {
            let v = s.borrow().end_offset();
            if v == -1 {
                return -1;
            }
            end = end.max(v);
        }
        end
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        let mut subs = Vec::with_capacity(self.subs.len());
        for mi in &self.subs {
            subs.push(sub_or_singleton(mi, &self.query)?);
        }
        disjunction(subs)
    }
    fn query(&self) -> Result<Arc<Clause>> {
        Err(unsupported("a conjunction's match"))
    }
    fn gaps(&self) -> i32 {
        self.it.gaps()
    }
    fn width(&self) -> i32 {
        self.it.width()
    }
}

// ---------------------------------------------------------------------------
// The disjunction's, the repeat's and the minimum-should-match's iterators
// ---------------------------------------------------------------------------

/// `DisjunctionIntervalsSource.DisjunctionMatchesIterator`.
struct DisjunctionMatches {
    it: iterators::DisjunctionIntervals<'static>,
    subs: Vec<Shared>,
    query: Arc<Clause>,
}

impl DisjunctionMatches {
    fn current(&self) -> Option<&Shared> {
        self.it.current_ord().and_then(|i| self.subs.get(i))
    }
}

impl IntervalMatchesIterator for DisjunctionMatches {
    fn next(&mut self) -> Result<bool> {
        Ok(self.it.next_interval()? != NO_MORE_INTERVALS)
    }
    fn start_position(&self) -> i32 {
        self.it.start()
    }
    fn end_position(&self) -> i32 {
        self.it.end()
    }
    fn start_offset(&self) -> i32 {
        self.current().map_or(-1, |s| s.borrow().start_offset())
    }
    fn end_offset(&self) -> i32 {
        self.current().map_or(-1, |s| s.borrow().end_offset())
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        match self.current() {
            Some(s) => s.borrow_mut().sub_matches(),
            None => Ok(None),
        }
    }
    fn query(&self) -> Result<Arc<Clause>> {
        match self.current() {
            Some(s) => s.borrow().query(),
            None => Ok(Arc::clone(&self.query)),
        }
    }
    fn gaps(&self) -> i32 {
        self.it.gaps()
    }
    fn width(&self) -> i32 {
        self.it.width()
    }
}

/// `RepeatingIntervalsSource.DuplicateMatchesIterator`.
struct DuplicateMatches {
    subs: Vec<Shared>,
    cached: bool,
    query: Arc<Clause>,
}

impl DuplicateMatches {
    /// `DuplicateMatchesIterator.build`: `None` unless every copy can be
    /// moved onto its own interval.
    fn build(subs: Vec<Shared>, query: Arc<Clause>) -> Result<Option<Shared>> {
        let mut count = subs.len();
        while count > 0 {
            for _ in 0..count {
                if !subs[count - 1].borrow_mut().next()? {
                    return Ok(None);
                }
            }
            count -= 1;
        }
        Ok(Some(shared(DuplicateMatches {
            subs,
            cached: false,
            query,
        })))
    }
}

impl IntervalMatchesIterator for DuplicateMatches {
    fn next(&mut self) -> Result<bool> {
        if !self.cached {
            self.cached = true;
            return Ok(true);
        }
        let last = self.subs.len() - 1;
        if !self.subs[last].borrow_mut().next()? {
            return Ok(false);
        }
        for s in &self.subs[..last] {
            s.borrow_mut().next()?;
        }
        Ok(true)
    }
    fn start_position(&self) -> i32 {
        self.subs[0].borrow().start_position()
    }
    fn end_position(&self) -> i32 {
        self.subs[self.subs.len() - 1].borrow().end_position()
    }
    fn start_offset(&self) -> i32 {
        self.subs[0].borrow().start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.subs[self.subs.len() - 1].borrow().end_offset()
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        let mut subs = Vec::with_capacity(self.subs.len());
        for mi in &self.subs {
            subs.push(sub_or_singleton(mi, &self.query)?);
        }
        disjunction(subs)
    }
    fn query(&self) -> Result<Arc<Clause>> {
        Err(unsupported("a repeat's match"))
    }
    fn gaps(&self) -> i32 {
        let mut width = self
            .end_position()
            .wrapping_sub(self.start_position())
            .wrapping_add(1);
        for mi in &self.subs {
            let mi = mi.borrow();
            width = width.wrapping_sub(
                mi.end_position()
                    .wrapping_sub(mi.start_position())
                    .wrapping_add(1),
            );
        }
        width
    }
    fn width(&self) -> i32 {
        let mut width = 0i32;
        for mi in &self.subs {
            let mi = mi.borrow();
            width = width.wrapping_add(
                mi.end_position()
                    .wrapping_sub(mi.start_position())
                    .wrapping_add(1),
            );
        }
        width
    }
}

/// `MinimumShouldMatchIntervalsSource.MinimumMatchesIterator`.
struct MinimumMatches {
    it: iterators::MinimumShouldMatchIntervals<'static>,
    /// `lookup`: the caching iterator behind each sub-iterator, by index.
    subs: Vec<Rc<RefCell<CachingMatches>>>,
    cached: bool,
    query: Arc<Clause>,
}

impl IntervalMatchesIterator for MinimumMatches {
    fn next(&mut self) -> Result<bool> {
        if self.cached {
            self.cached = false;
            return Ok(true);
        }
        Ok(self.it.next_interval()? != NO_MORE_INTERVALS)
    }
    fn start_position(&self) -> i32 {
        self.it.start()
    }
    fn end_position(&self) -> i32 {
        self.it.end()
    }
    fn start_offset(&self) -> i32 {
        let mut start = i32::MAX;
        for i in self.it.current_iterators() {
            start = start.min(self.subs[i].borrow().start_offset());
        }
        start
    }
    fn end_offset(&self) -> i32 {
        let mut end = 0;
        for i in self.it.current_iterators() {
            end = end.max(self.subs[i].borrow().end_offset());
        }
        end
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        let mut mis = Vec::new();
        for i in self.it.current_iterators() {
            let cms = &self.subs[i];
            let sub = cms.borrow_mut().sub_matches()?;
            mis.push(match sub {
                Some(s) => s,
                None => {
                    let as_shared: Shared = Rc::clone(cms) as Shared;
                    as_plain(as_shared, &self.query)
                }
            });
        }
        disjunction(mis)
    }
    /// Java's `null`: the enclosing query.
    fn query(&self) -> Result<Arc<Clause>> {
        Ok(Arc::clone(&self.query))
    }
    fn gaps(&self) -> i32 {
        self.it.gaps()
    }
    fn width(&self) -> i32 {
        self.it.width()
    }
}

// ---------------------------------------------------------------------------
// A document's occurrences, read once
// ---------------------------------------------------------------------------

/// One term's occurrences in one document.
type Occurrences = Arc<[Position]>;

/// An expanded term and its occurrences in one document.
type TermOccurrences = (Vec<u8>, Occurrences);

/// Every term occurrence a source's matches read in one document.
#[derive(Default)]
struct DocOccurrences {
    /// `(field, term)`: the term's occurrences in the document, `None` when
    /// it is not in the field or not in the document.
    terms: HashMap<(String, Vec<u8>), Option<Occurrences>>,
    /// `(field, multi-term source)`: the expanded terms with occurrences in
    /// the document, in term order.
    multi: HashMap<(String, String), Vec<TermOccurrences>>,
}

fn multi_key(field: &str, source: &IntervalsSource) -> (String, String) {
    (field.to_string(), format!("{source:?}"))
}

/// Reads what `source`'s matches in `doc` need, with the checks Java's
/// `matches` makes on the way.
fn gather(
    seg: &OpenSegment<'_>,
    source: &IntervalsSource,
    field: &str,
    doc: i32,
    out: &mut DocOccurrences,
) -> Result<()> {
    use IntervalsSource as S;
    match source {
        S::Term(term) | S::PayloadFilteredTerm { term, .. } => {
            let key = (field.to_string(), term.clone());
            if out.terms.contains_key(&key) {
                return Ok(());
            }
            let occ = match seg.fields.field(field) {
                None => None,
                Some(ft) => {
                    if !ft.index_options().subsumes_positions() {
                        return Err(Error::IllegalArgument(format!(
                            "Cannot create an IntervalIterator over field {field} because it has no indexed positions"
                        )));
                    }
                    if matches!(source, S::PayloadFilteredTerm { .. }) && !ft.has_payloads() {
                        return Err(Error::IllegalArgument(format!(
                            "Cannot create a payload-filtered iterator over field {field} because it has no indexed payloads"
                        )));
                    }
                    let Some(pos_in) = seg.pos_in else {
                        return Err(Error::MissingPosInput);
                    };
                    ft.occurrences_for_doc(term, seg.doc_in, pos_in, seg.pay_in, doc)?
                        .map(Into::into)
                }
            };
            out.terms.insert(key, occ);
        }
        S::MultiTerm {
            pattern,
            max_expansions,
            ..
        } => {
            let key = multi_key(field, source);
            if out.multi.contains_key(&key) {
                return Ok(());
            }
            let mut found = Vec::new();
            if let (Some(ft), Some(pos_in)) = (seg.fields.field(field), seg.pos_in) {
                let src = pattern.source(field);
                let terms = crate::exec::extended::expand_terms(seg.fields, &src, None)?;
                let mut count = 0usize;
                for (term, _) in terms {
                    if let Some(occ) =
                        ft.occurrences_for_doc(&term, seg.doc_in, pos_in, seg.pay_in, doc)?
                    {
                        found.push((term.clone(), occ.into()));
                        // `count++ > maxExpansions`.
                        let before = count;
                        count += 1;
                        if before > *max_expansions {
                            return Err(Error::IllegalState(format!(
                                "Automaton {} expanded to too many terms (limit {max_expansions})",
                                String::from_utf8_lossy(&term)
                            )));
                        }
                    }
                }
            }
            out.multi.insert(key, found);
        }
        S::Block(subs) | S::Ordered(subs) | S::Unordered(subs) => {
            for s in subs {
                gather(seg, s, field, doc, out)?;
            }
        }
        S::Disjunction { sources, .. } | S::MinimumShouldMatch { sources, .. } => {
            for s in sources {
                gather(seg, s, field, doc, out)?;
            }
        }
        S::Repeating { source, .. }
        | S::Filtered { source, .. }
        | S::Extended { source, .. }
        | S::Offset { source, .. } => gather(seg, source, field, doc, out)?,
        S::FixedField { field, source } => gather(seg, source, field, doc, out)?,
        S::NoMatch(_) => {}
        S::Containing { big: a, small: b }
        | S::ContainedBy { small: a, big: b }
        | S::NotContaining {
            minuend: a,
            subtrahend: b,
        }
        | S::NotContainedBy {
            minuend: a,
            subtrahend: b,
        }
        | S::Overlapping {
            source: a,
            reference: b,
        }
        | S::NonOverlapping {
            minuend: a,
            subtrahend: b,
        } => {
            gather(seg, a, field, doc, out)?;
            gather(seg, b, field, doc, out)?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// IntervalsSource.matches(field, ctx, doc)
// ---------------------------------------------------------------------------

/// What every `matches` call of one document shares.
struct MatchContext<'o> {
    occ: &'o DocOccurrences,
    doc: i32,
    /// The enclosing query, for the iterators whose `getQuery` Java does
    /// not answer.
    query: Arc<Clause>,
}

/// `IntervalsSource.matches(field, ctx, doc)`.
fn source_matches(
    source: &IntervalsSource,
    field: &str,
    cx: &MatchContext<'_>,
) -> Result<Option<Shared>> {
    use IntervalsSource as S;
    let doc = cx.doc;
    Ok(match source {
        S::Term(term) | S::PayloadFilteredTerm { term, .. } => {
            let Some(Some(occ)) = cx.occ.terms.get(&(field.to_string(), term.clone())) else {
                return Ok(None);
            };
            let filter = match source {
                S::PayloadFilteredTerm { filter, .. } => Some(filter.clone()),
                _ => None,
            };
            let query = match &filter {
                None => Arc::new(Clause::Term(TermQuery::new(field, term.clone()))),
                Some(_) => Arc::clone(&cx.query),
            };
            Some(shared(TermMatches {
                occurrences: Arc::clone(occ),
                upto: occ.len(),
                at: 0,
                pos: -1,
                filter,
                query,
            }))
        }
        S::MultiTerm { .. } => {
            let Some(found) = cx.occ.multi.get(&multi_key(field, source)) else {
                return Ok(None);
            };
            let subs: Vec<BoxMatchesIterator> = found
                .iter()
                .map(|(term, occ)| {
                    let q = Arc::new(Clause::Term(TermQuery::new(field, term.clone())));
                    Box::new(crate::matches::ListMatchesIterator::for_term(q, occ))
                        as BoxMatchesIterator
                })
                .collect();
            disjunction(subs)?.map(|mi| shared(MultiTermMatches { mi }))
        }
        S::NoMatch(_) => None,
        S::FixedField { field, source } => source_matches(source, field, cx)?,
        S::Disjunction { sources, .. } => {
            let mut subs = Vec::with_capacity(sources.len());
            for s in sources {
                if let Some(mi) = source_matches(s, field, cx)? {
                    subs.push(mi);
                }
            }
            if subs.is_empty() {
                return Ok(None);
            }
            let wrapped = subs.iter().map(|m| wrap(m, doc)).collect();
            let mut it = iterators::DisjunctionIntervals::new(wrapped);
            if it.advance(doc)? != doc {
                return Ok(None);
            }
            Some(shared(DisjunctionMatches {
                it,
                subs,
                query: Arc::clone(&cx.query),
            }))
        }
        S::Repeating { source, count, .. } => {
            let mut subs = Vec::new();
            for _ in 0..*count {
                match source_matches(source, field, cx)? {
                    Some(mi) => subs.push(mi),
                    None => return Ok(None),
                }
            }
            DuplicateMatches::build(subs, Arc::clone(&cx.query))?
        }
        S::Filtered { source, filter } => {
            let Some(mi) = source_matches(source, field, cx)? else {
                return Ok(None);
            };
            let filtered =
                BoxIntervals::boxed(iterators::FilteredIntervals::new(wrap(&mi, doc), *filter));
            as_matches(filtered, mi, doc)?
        }
        S::Extended {
            source,
            before,
            after,
        } => {
            let Some(mi) = source_matches(source, field, cx)? else {
                return Ok(None);
            };
            let no_offsets = shared(NoOffsets { inner: mi });
            let wrapped = BoxIntervals::boxed(iterators::ExtendedIntervals::new(
                wrap(&no_offsets, doc),
                *before,
                *after,
            ));
            as_matches(wrapped, no_offsets, doc)?
        }
        S::Offset { source, before } => {
            let Some(mi) = source_matches(source, field, cx)? else {
                return Ok(None);
            };
            let it = BoxIntervals::boxed(iterators::OffsetIntervals::new(wrap(&mi, doc), *before));
            as_matches(it, mi, doc)?
        }
        S::NotContaining {
            minuend,
            subtrahend,
        } => difference(RelativeKind::NotContaining, minuend, subtrahend, field, cx)?,
        S::NotContainedBy {
            minuend,
            subtrahend,
        } => difference(RelativeKind::NotContainedBy, minuend, subtrahend, field, cx)?,
        S::NonOverlapping {
            minuend,
            subtrahend,
        } => difference(RelativeKind::NonOverlapping, minuend, subtrahend, field, cx)?,
        S::Block(sources) => {
            let Some(subs) = all_matches(sources, field, cx)? else {
                return Ok(None);
            };
            let it = iterators::block(subs.iter().map(|m| wrap(m, doc)).collect());
            conjunction_matches(it, subs, cx)?
        }
        S::Containing { big, small } => {
            let Some(subs) =
                all_matches(&[big.as_ref().clone(), small.as_ref().clone()], field, cx)?
            else {
                return Ok(None);
            };
            let it = iterators::filtering(
                iterators::FilteringKind::Containing,
                wrap(&subs[0], doc),
                wrap(&subs[1], doc),
            );
            conjunction_matches(it, subs, cx)?
        }
        S::ContainedBy { small, big } => {
            let Some(subs) =
                all_matches(&[small.as_ref().clone(), big.as_ref().clone()], field, cx)?
            else {
                return Ok(None);
            };
            let it = iterators::filtering(
                iterators::FilteringKind::ContainedBy,
                wrap(&subs[0], doc),
                wrap(&subs[1], doc),
            );
            // Only the small source's matches are reported.
            conjunction_matches(it, vec![Rc::clone(&subs[0])], cx)?
        }
        S::Overlapping { source, reference } => {
            let Some(subs) = all_matches(
                &[source.as_ref().clone(), reference.as_ref().clone()],
                field,
                cx,
            )?
            else {
                return Ok(None);
            };
            let it = iterators::filtering(
                iterators::FilteringKind::Overlapping,
                wrap(&subs[0], doc),
                wrap(&subs[1], doc),
            );
            conjunction_matches(it, vec![Rc::clone(&subs[0])], cx)?
        }
        S::Ordered(sources) | S::Unordered(sources) => {
            let mut caching = Vec::with_capacity(sources.len());
            for s in sources {
                match source_matches(s, field, cx)? {
                    Some(mi) => caching.push(Rc::new(RefCell::new(CachingMatches {
                        inner: mi,
                        cached: Vec::new(),
                    }))),
                    None => return Ok(None),
                }
            }
            let as_shared: Vec<Shared> = caching.iter().map(|c| Rc::clone(c) as Shared).collect();
            let wrapped = as_shared.iter().map(|m| wrap(m, doc)).collect();
            let callback = cache_iterators(caching);
            let mut it = if matches!(source, S::Ordered(_)) {
                iterators::ordered(wrapped, callback)
            } else {
                iterators::unordered(wrapped, callback)
            };
            if it.advance(doc)? != doc || it.next_interval()? == NO_MORE_INTERVALS {
                return Ok(None);
            }
            Some(shared(ConjunctionMatches {
                it,
                subs: as_shared,
                cached: true,
                query: Arc::clone(&cx.query),
            }))
        }
        S::MinimumShouldMatch {
            sources,
            min_should_match,
        } => {
            let mut caching = Vec::with_capacity(sources.len());
            for s in sources {
                if let Some(mi) = source_matches(s, field, cx)? {
                    caching.push(Rc::new(RefCell::new(CachingMatches {
                        inner: mi,
                        cached: Vec::new(),
                    })));
                }
            }
            let msm = usize::try_from(*min_should_match).unwrap_or(0);
            if caching.len() < msm {
                return Ok(None);
            }
            let wrapped = caching
                .iter()
                .map(|c| wrap(&(Rc::clone(c) as Shared), doc))
                .collect();
            let callback = cache_iterators(caching.clone());
            let mut it = iterators::MinimumShouldMatchIntervals::new(wrapped, msm, callback);
            if it.advance(doc)? != doc || it.next_interval()? == NO_MORE_INTERVALS {
                return Ok(None);
            }
            Some(shared(MinimumMatches {
                it,
                subs: caching,
                cached: true,
                query: Arc::clone(&cx.query),
            }))
        }
    })
}

/// `ExtendedIntervalsSource.matches`' delegate that hides offsets.
struct NoOffsets {
    inner: Shared,
}

impl IntervalMatchesIterator for NoOffsets {
    fn next(&mut self) -> Result<bool> {
        self.inner.borrow_mut().next()
    }
    fn start_position(&self) -> i32 {
        self.inner.borrow().start_position()
    }
    fn end_position(&self) -> i32 {
        self.inner.borrow().end_position()
    }
    // SENTINEL: `-1` = "no offsets", `MatchesIterator`'s own contract,
    // reported to the caller as Java reports it.
    fn start_offset(&self) -> i32 {
        -1
    }
    // SENTINEL: `-1` = "no offsets", `MatchesIterator`'s own contract,
    // reported to the caller as Java reports it.
    fn end_offset(&self) -> i32 {
        -1
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        self.inner.borrow_mut().sub_matches()
    }
    fn query(&self) -> Result<Arc<Clause>> {
        self.inner.borrow().query()
    }
    fn gaps(&self) -> i32 {
        self.inner.borrow().gaps()
    }
    fn width(&self) -> i32 {
        self.inner.borrow().width()
    }
}

/// `ConjunctionIntervalsSource.matches`' sub-iterators: all of them, or
/// `None`.
fn all_matches(
    sources: &[IntervalsSource],
    field: &str,
    cx: &MatchContext<'_>,
) -> Result<Option<Vec<Shared>>> {
    let mut subs = Vec::with_capacity(sources.len());
    for s in sources {
        match source_matches(s, field, cx)? {
            Some(mi) => subs.push(mi),
            None => return Ok(None),
        }
    }
    Ok(Some(subs))
}

/// The tail of `ConjunctionIntervalsSource.matches`: the combined iterator
/// on its first interval in the document, then `createMatchesIterator`.
fn conjunction_matches(
    mut it: BoxIntervals<'static>,
    subs: Vec<Shared>,
    cx: &MatchContext<'_>,
) -> Result<Option<Shared>> {
    if it.advance(cx.doc)? != cx.doc || it.next_interval()? == NO_MORE_INTERVALS {
        return Ok(None);
    }
    Ok(Some(shared(ConjunctionMatches {
        it,
        subs,
        cached: true,
        query: Arc::clone(&cx.query),
    })))
}

/// `DifferenceIntervalsSource.matches`.
fn difference(
    kind: RelativeKind,
    minuend: &IntervalsSource,
    subtrahend: &IntervalsSource,
    field: &str,
    cx: &MatchContext<'_>,
) -> Result<Option<Shared>> {
    let Some(min_it) = source_matches(minuend, field, cx)? else {
        return Ok(None);
    };
    let Some(sub_it) = source_matches(subtrahend, field, cx)? else {
        return Ok(Some(min_it));
    };
    let it = BoxIntervals::boxed(iterators::RelativeIntervals::new(
        kind,
        wrap(&min_it, cx.doc),
        wrap(&sub_it, cx.doc),
    ));
    as_matches(it, min_it, cx.doc)
}

/// `IntervalWeight.matches(context, doc)`: the source's matches in `doc`
/// (leaf-local), their `getQuery()` the [`IntervalQuery`].
pub(crate) fn interval_matches(
    seg: &OpenSegment<'_>,
    q: &IntervalQuery,
    clause: Arc<Clause>,
    doc: i32,
) -> Result<Option<BoxMatches>> {
    let mut occ = DocOccurrences::default();
    gather(seg, &q.source, &q.field, doc, &mut occ)?;
    let occ = Arc::new(occ);
    let source = Arc::new(q.source.clone());
    let field = q.field.clone();
    for_field(
        &q.field,
        Box::new(move || {
            let cx = MatchContext {
                occ: &occ,
                doc,
                query: Arc::clone(&clause),
            };
            Ok(source_matches(&source, &field, &cx)?.map(|mi| {
                Box::new(AsMatchesIterator {
                    mi,
                    query: Arc::clone(&clause),
                }) as BoxMatchesIterator
            }))
        }),
    )
}

#[cfg(test)]
mod tests;
