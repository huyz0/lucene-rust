//! The Matches API: `Weight.matches(context, doc)` for the scorer-tree
//! queries -- where in a matching document a query matched, as positions and
//! offsets per field.
//!
//! A port of Lucene 10.5.0's `Matches`, `MatchesIterator`, `MatchesUtils`
//! (`MATCH_WITH_NO_TERMS`, `fromSubMatches`, `forField`, `disjunction`),
//! `NamedMatches` (`findNamedMatches`), `TermMatchesIterator`,
//! `DisjunctionMatchesIterator` (`fromSubIterators`, `fromTermsEnum`),
//! `FilterMatchesIterator`, and the `matches` of `TermWeight`,
//! `PhraseWeight` (exact and sloppy, with `SloppyPhraseMatcher`'s lead
//! capture), `BooleanWeight`, `DisjunctionMaxWeight`, `ConstantScoreWeight`,
//! the multi-term wrapper (`AbstractMultiTermQueryConstantScoreWrapper`:
//! prefix, wildcard, regexp, term set) and `Weight`'s default (a clause
//! without positions -- match-all, points range, exists -- matches with no
//! terms when its scorer reaches the document).
//!
//! # How it is shaped
//!
//! A leaf's iterator is built from the document's decoded occurrences
//! ([`lucene_codecs::blocktree::FieldTerms::occurrences_for_doc`]) rather
//! than from a live `PostingsEnum`; a phrase's matches are computed when its
//! iterator is built (the same sequence Java's lazy `nextMatch` produces).
//! [`DisjunctionMatchesIterator`] merges its sub-iterators through a port of
//! `util/PriorityQueue`'s heap, whose tie order -- its `lessThan` calls equal
//! intervals less -- only the same heap reproduces.
//!
//! Like Java, matches ignore deletions: a deleted document still reports its
//! matches.
//!
//! Not ported: the `matches` of `FuzzyQuery` (its rewrite is a reader-wide
//! top-terms boolean), `MultiPhraseQuery` and span queries, which report
//! [`crate::Error::IllegalArgument`]; `NamedQuery` has no [`Clause`] variant,
//! so names are attached through [`named_matches`].
//!
//! Verified against Lucene by `tests/matches_fixtures.rs`
//! (`fixtures/src/GenMatches.java`).

use std::sync::Arc;

use lucene_codecs::postings::Position;

use crate::exec::{self, LeafContext, Mode, NO_MORE_DOCS};
use crate::multi_segment::OpenSegment;
use crate::query::{BooleanQuery, Clause, PhraseQuery, TermQuery};
use crate::sloppy_phrase::{sloppy_phrase_match_spans, PhraseRepeats};
use crate::{Error, Result};

/// `MatchesIterator`: the matches of one field of one document, ordered by
/// start position then end position.
pub trait MatchesIterator {
    /// `next()`: advances to the next match.
    fn next(&mut self) -> Result<bool>;
    /// `startPosition()`, `-1` without positions.
    fn start_position(&self) -> i32;
    /// `endPosition()`, `-1` without positions.
    fn end_position(&self) -> i32;
    /// `startOffset()`, `-1` without offsets.
    fn start_offset(&self) -> i32;
    /// `endOffset()`, `-1` without offsets.
    fn end_offset(&self) -> i32;
    /// `getSubMatches()`: the individual terms of the current match, or
    /// `None` at the leaf level (every iterator here is a leaf: terms and
    /// phrases).
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        Ok(None)
    }
    /// `getQuery()`: the query causing the current match.
    fn query(&self) -> &Clause;
}

pub type BoxMatchesIterator = Box<dyn MatchesIterator>;

/// `Matches`: a document's matches, per field.
pub trait Matches {
    /// `getMatches(field)`: a fresh iterator over the field's matches, or
    /// `None` when the field has none.
    fn get_matches(&self, field: &str) -> Result<Option<BoxMatchesIterator>>;
    /// `iterator()`: the fields with matches.
    fn fields(&self) -> Vec<String>;
    /// `getSubMatches()`.
    fn sub_matches(&self) -> Vec<&dyn Matches>;
    /// `this instanceof NamedMatches`.
    fn as_named(&self) -> Option<&NamedMatches> {
        None
    }
    /// `this == MatchesUtils.MATCH_WITH_NO_TERMS`.
    fn is_match_with_no_terms(&self) -> bool {
        false
    }
}

pub type BoxMatches = Box<dyn Matches>;

// ---------------------------------------------------------------------------
// MatchesUtils
// ---------------------------------------------------------------------------

/// `MatchesUtils.MATCH_WITH_NO_TERMS`: a match with nothing to point at
/// (a clause without positions).
#[derive(Debug, Default, Clone, Copy)]
pub struct MatchWithNoTerms;

impl Matches for MatchWithNoTerms {
    fn get_matches(&self, _field: &str) -> Result<Option<BoxMatchesIterator>> {
        Ok(None)
    }
    fn fields(&self) -> Vec<String> {
        Vec::new()
    }
    fn sub_matches(&self) -> Vec<&dyn Matches> {
        Vec::new()
    }
    fn is_match_with_no_terms(&self) -> bool {
        true
    }
}

/// A source of fresh iterators over one field's matches.
pub type IteratorFactory = dyn Fn() -> Result<Option<BoxMatchesIterator>> + Send + Sync;

/// `MatchesUtils.forField(field, supplier)`: one field's matches.
struct FieldMatches {
    field: String,
    make: Box<IteratorFactory>,
}

impl Matches for FieldMatches {
    fn get_matches(&self, field: &str) -> Result<Option<BoxMatchesIterator>> {
        if field != self.field {
            return Ok(None);
        }
        (self.make)()
    }
    fn fields(&self) -> Vec<String> {
        vec![self.field.clone()]
    }
    fn sub_matches(&self) -> Vec<&dyn Matches> {
        Vec::new()
    }
}

/// `MatchesUtils.forField(field, mis)`: `None` when the supplier's first
/// iterator is `None`.
pub fn for_field(field: &str, make: Box<IteratorFactory>) -> Result<Option<BoxMatches>> {
    if make()?.is_none() {
        return Ok(None);
    }
    Ok(Some(Box::new(FieldMatches {
        field: field.to_string(),
        make,
    })))
}

/// `MatchesUtils.fromSubMatches`'s amalgamation of two or more.
struct SubMatches {
    all: Vec<BoxMatches>,
}

impl SubMatches {
    fn with_terms(&self) -> impl Iterator<Item = &BoxMatches> {
        self.all.iter().filter(|m| !m.is_match_with_no_terms())
    }
}

impl Matches for SubMatches {
    fn get_matches(&self, field: &str) -> Result<Option<BoxMatchesIterator>> {
        let mut subs = Vec::new();
        for m in self.with_terms() {
            if let Some(it) = m.get_matches(field)? {
                subs.push(it);
            }
        }
        disjunction(subs)
    }
    /// `sm.stream().flatMap(...).distinct()`: first-seen order.
    fn fields(&self) -> Vec<String> {
        let mut out: Vec<String> = Vec::new();
        for m in self.with_terms() {
            for f in m.fields() {
                if !out.contains(&f) {
                    out.push(f);
                }
            }
        }
        out
    }
    fn sub_matches(&self) -> Vec<&dyn Matches> {
        self.all.iter().map(|m| m.as_ref()).collect()
    }
}

/// `MatchesUtils.fromSubMatches(subMatches)`: `None` for none;
/// `MATCH_WITH_NO_TERMS` when every one is; the one with terms when there is
/// one; else their amalgamation, whose iterators are disjunctions.
pub fn from_sub_matches(sub_matches: Vec<BoxMatches>) -> Option<BoxMatches> {
    if sub_matches.is_empty() {
        return None;
    }
    let with_terms = sub_matches
        .iter()
        .filter(|m| !m.is_match_with_no_terms())
        .count();
    if with_terms == 0 {
        return Some(Box::new(MatchWithNoTerms));
    }
    if with_terms == 1 {
        return sub_matches
            .into_iter()
            .find(|m| !m.is_match_with_no_terms());
    }
    Some(Box::new(SubMatches { all: sub_matches }))
}

/// `MatchesUtils.disjunction(subMatches)` /
/// `DisjunctionMatchesIterator.fromSubIterators`: `None` for none, the one
/// for one, else their merge.
pub fn disjunction(mut subs: Vec<BoxMatchesIterator>) -> Result<Option<BoxMatchesIterator>> {
    match subs.len() {
        0 => Ok(None),
        1 => Ok(subs.pop()),
        _ => Ok(Some(Box::new(DisjunctionMatchesIterator::new(subs)?))),
    }
}

// ---------------------------------------------------------------------------
// NamedMatches
// ---------------------------------------------------------------------------

/// `NamedMatches`: a query's matches under a name.
pub struct NamedMatches {
    name: String,
    inner: BoxMatches,
}

impl NamedMatches {
    pub fn new(name: impl Into<String>, inner: BoxMatches) -> Self {
        Self {
            name: name.into(),
            inner,
        }
    }

    /// `getName()`.
    pub fn name(&self) -> &str {
        &self.name
    }
}

impl Matches for NamedMatches {
    fn get_matches(&self, field: &str) -> Result<Option<BoxMatchesIterator>> {
        self.inner.get_matches(field)
    }
    fn fields(&self) -> Vec<String> {
        self.inner.fields()
    }
    fn sub_matches(&self) -> Vec<&dyn Matches> {
        vec![self.inner.as_ref()]
    }
    fn as_named(&self) -> Option<&NamedMatches> {
        Some(self)
    }
}

/// `NamedMatches.findNamedMatches(matches)`: every named node of the tree,
/// breadth first.
pub fn find_named_matches(matches: &dyn Matches) -> Vec<&NamedMatches> {
    let mut out = Vec::new();
    let mut queue: std::collections::VecDeque<&dyn Matches> = std::collections::VecDeque::new();
    queue.push_back(matches);
    while let Some(m) = queue.pop_front() {
        if let Some(n) = m.as_named() {
            out.push(n);
        }
        queue.extend(m.sub_matches());
    }
    out
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

/// `TermMatchesIterator`, and a phrase's `MatchesIterator` from
/// `PhraseWeight.matches`: a precomputed list of `[startPosition,
/// endPosition, startOffset, endOffset]` intervals.
pub struct ListMatchesIterator {
    query: Arc<Clause>,
    intervals: Vec<[i32; 4]>,
    /// The next interval to return: `next` moves the current one to `at - 1`.
    at: usize,
}

impl ListMatchesIterator {
    /// A term's occurrences in one document, in order (`TermMatchesIterator`).
    pub fn for_term(query: Arc<Clause>, occurrences: &[Position]) -> Self {
        Self {
            query,
            intervals: occurrences
                .iter()
                .map(|o| [o.position, o.position, o.start_offset, o.end_offset])
                .collect(),
            at: 0,
        }
    }

    fn current(&self) -> [i32; 4] {
        self.at
            .checked_sub(1)
            .and_then(|i| self.intervals.get(i))
            .copied()
            .unwrap_or([-1; 4])
    }
}

impl MatchesIterator for ListMatchesIterator {
    fn next(&mut self) -> Result<bool> {
        if self.at < self.intervals.len() {
            self.at += 1;
            Ok(true)
        } else {
            Ok(false)
        }
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
        &self.query
    }
}

/// `FilterMatchesIterator`: delegates everything to `inner`; wrap it to
/// override a method.
pub struct FilterMatchesIterator {
    pub inner: BoxMatchesIterator,
}

impl MatchesIterator for FilterMatchesIterator {
    fn next(&mut self) -> Result<bool> {
        self.inner.next()
    }
    fn start_position(&self) -> i32 {
        self.inner.start_position()
    }
    fn end_position(&self) -> i32 {
        self.inner.end_position()
    }
    fn start_offset(&self) -> i32 {
        self.inner.start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.inner.end_offset()
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        self.inner.sub_matches()
    }
    fn query(&self) -> &Clause {
        self.inner.query()
    }
}

/// `DisjunctionMatchesIterator`: several iterators merged by start then end
/// position (by offsets when neither has positions).
pub struct DisjunctionMatchesIterator {
    /// `PriorityQueue<MatchesIterator>`'s heap, 1-based.
    heap: Vec<Option<BoxMatchesIterator>>,
    size: usize,
    started: bool,
}

/// `DisjunctionMatchesIterator`'s `lessThan`: note it is true for equal
/// intervals.
fn disjunction_less(a: &dyn MatchesIterator, b: &dyn MatchesIterator) -> bool {
    if a.start_position() == -1 && b.start_position() == -1 {
        return a.start_offset() < b.start_offset()
            || (a.start_offset() == b.start_offset() && a.end_offset() <= b.end_offset());
    }
    a.start_position() < b.start_position()
        || (a.start_position() == b.start_position() && a.end_position() <= b.end_position())
}

impl DisjunctionMatchesIterator {
    fn new(subs: Vec<BoxMatchesIterator>) -> Result<Self> {
        let mut q = Self {
            heap: Vec::with_capacity(subs.len().saturating_add(1)),
            size: 0,
            started: false,
        };
        q.heap.push(None);
        for mut mi in subs {
            if mi.next()? {
                q.add(mi);
            }
        }
        Ok(q)
    }

    fn at(&self, i: usize) -> &dyn MatchesIterator {
        self.heap[i]
            .as_deref()
            .expect("heap slots 1..=size are occupied")
    }

    fn less(&self, i: usize, j: usize) -> bool {
        disjunction_less(self.at(i), self.at(j))
    }

    fn top(&self) -> &dyn MatchesIterator {
        self.at(1)
    }

    /// `PriorityQueue.add` + `upHeap`.
    fn add(&mut self, mi: BoxMatchesIterator) {
        self.size += 1;
        if self.heap.len() <= self.size {
            self.heap.push(Some(mi));
        } else {
            self.heap[self.size] = Some(mi);
        }
        let mut i = self.size;
        let mut j = i >> 1;
        while j > 0 && self.less(i, j) {
            self.heap.swap(i, j);
            i = j;
            j >>= 1;
        }
    }

    /// `PriorityQueue.downHeap(1)`, as the swaps of Java's hole-moving loop.
    fn down_heap(&mut self) {
        let mut i = 1;
        let mut j = i << 1;
        let mut k = j + 1;
        if k <= self.size && self.less(k, j) {
            j = k;
        }
        while j <= self.size && self.less(j, i) {
            self.heap.swap(i, j);
            i = j;
            j = i << 1;
            k = j + 1;
            if k <= self.size && self.less(k, j) {
                j = k;
            }
        }
    }

    /// `PriorityQueue.pop`.
    fn pop(&mut self) {
        if self.size == 0 {
            return;
        }
        self.heap.swap(1, self.size);
        self.heap[self.size] = None;
        self.size -= 1;
        if self.size > 0 {
            self.down_heap();
        }
    }
}

impl MatchesIterator for DisjunctionMatchesIterator {
    fn next(&mut self) -> Result<bool> {
        if !self.started {
            self.started = true;
            return Ok(self.size > 0);
        }
        if self.size == 0 {
            return Ok(false);
        }
        let advanced = match self.heap[1].as_mut() {
            Some(top) => top.next()?,
            None => false,
        };
        if !advanced {
            self.pop();
        }
        if self.size > 0 {
            self.down_heap();
            return Ok(true);
        }
        Ok(false)
    }
    fn start_position(&self) -> i32 {
        self.top().start_position()
    }
    fn end_position(&self) -> i32 {
        self.top().end_position()
    }
    fn start_offset(&self) -> i32 {
        self.top().start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.top().end_offset()
    }
    fn sub_matches(&mut self) -> Result<Option<BoxMatchesIterator>> {
        match self.heap.get_mut(1).and_then(Option::as_mut) {
            Some(top) => top.sub_matches(),
            None => Ok(None),
        }
    }
    fn query(&self) -> &Clause {
        self.top().query()
    }
}

// ---------------------------------------------------------------------------
// Weight.matches
// ---------------------------------------------------------------------------

/// The segment's readers, as a matches computation needs them (no live docs:
/// matches ignore deletions, as a `Scorer` does).
fn leaf_context<'a>(seg: &OpenSegment<'a>) -> LeafContext<'a> {
    LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: None,
        points: seg.points,
        norms: None,
        global: None,
        max_doc: seg.max_doc,
        cache: None,
        reader: seg.reader,
        similarity: None,
    }
}

/// `Weight.matches`'s default: `MATCH_WITH_NO_TERMS` when the clause's
/// scorer reaches `doc`, else `None`.
fn default_matches(seg: &OpenSegment<'_>, clause: &Clause, doc: i32) -> Result<Option<BoxMatches>> {
    let ctx = leaf_context(seg);
    let Some(mut scorer) = exec::build::build(&ctx, clause, 1.0, Mode::NoScores, true)? else {
        return Ok(None);
    };
    let found = exec::exact_advance(scorer.as_mut(), doc)?;
    Ok((found == doc && found != NO_MORE_DOCS).then(|| Box::new(MatchWithNoTerms) as BoxMatches))
}

/// The occurrences of `term` in `doc` (`PostingsEnum.OFFSETS`), or `None`
/// when the term does not occur there. A field indexed without positions
/// reports `freq` occurrences at position `-1` without offsets, as its
/// `PostingsEnum` does.
fn occurrences(
    seg: &OpenSegment<'_>,
    field: &str,
    term: &[u8],
    doc: i32,
) -> Result<Option<Vec<Position>>> {
    let Some(ft) = seg.fields.field(field) else {
        return Ok(None);
    };
    if let (Some(pos_in), true) = (seg.pos_in, ft.index_options().subsumes_positions()) {
        return Ok(ft.occurrences_for_doc(term, seg.doc_in, pos_in, seg.pay_in, doc)?);
    }
    let Some(postings) = ft.postings(term, seg.doc_in)? else {
        return Ok(None);
    };
    let Ok(i) = postings.docs.binary_search(&doc) else {
        return Ok(None);
    };
    let freq = postings.freqs.get(i).copied().unwrap_or(1).max(1);
    let none = Position {
        position: -1,
        start_offset: -1,
        end_offset: -1,
        payload: Vec::new(),
    };
    Ok(Some(vec![none; usize::try_from(freq).unwrap_or(1)]))
}

/// `TermWeight.matches`.
fn term_matches(
    seg: &OpenSegment<'_>,
    q: &TermQuery,
    clause: Arc<Clause>,
    doc: i32,
) -> Result<Option<BoxMatches>> {
    let Some(occ) = occurrences(seg, &q.field, &q.term, doc)? else {
        return Ok(None);
    };
    let occ: Arc<[Position]> = occ.into();
    for_field(
        &q.field,
        Box::new(move || {
            Ok(Some(
                Box::new(ListMatchesIterator::for_term(Arc::clone(&clause), &occ))
                    as BoxMatchesIterator,
            ))
        }),
    )
}

/// `ExactPhraseMatcher`'s per-term state.
struct PostingsAndPosition<'o> {
    occ: &'o [Position],
    pos: i32,
    up_to: usize,
    offset: i32,
}

/// `ExactPhraseMatcher.advancePosition`.
fn advance_position(p: &mut PostingsAndPosition<'_>, target: i32) -> bool {
    while p.pos < target {
        match p.occ.get(p.up_to) {
            Some(o) => {
                p.pos = o.position;
                p.up_to += 1;
            }
            None => return false,
        }
    }
    true
}

/// `ExactPhraseMatcher.nextMatch`.
fn exact_next_match(ps: &mut [PostingsAndPosition<'_>]) -> bool {
    let Some((lead, rest)) = ps.split_first_mut() else {
        return false;
    };
    match lead.occ.get(lead.up_to) {
        Some(o) => {
            lead.pos = o.position;
            lead.up_to += 1;
        }
        None => return false,
    }
    'advance_head: loop {
        let phrase_pos = lead.pos - lead.offset;
        for p in rest.iter_mut() {
            let expected = phrase_pos + p.offset;
            if !advance_position(p, expected) {
                break 'advance_head;
            }
            if p.pos != expected {
                let target = p.pos - p.offset + lead.offset;
                if advance_position(lead, target) {
                    continue 'advance_head;
                }
                break 'advance_head;
            }
        }
        return true;
    }
    false
}

/// Every exact match of the phrase whose terms' occurrences are `occs`, as
/// `ExactPhraseMatcher.startPosition()/endPosition()/startOffset()/
/// endOffset()` report them.
fn exact_phrase_spans(occs: &[Vec<Position>]) -> Vec<[i32; 4]> {
    let mut ps: Vec<PostingsAndPosition<'_>> = occs
        .iter()
        .enumerate()
        .map(|(i, occ)| PostingsAndPosition {
            occ,
            pos: -1,
            up_to: 0,
            offset: i32::try_from(i).unwrap_or(i32::MAX),
        })
        .collect();
    let mut out = Vec::new();
    while exact_next_match(&mut ps) {
        let first = &ps[0];
        let last = &ps[ps.len() - 1];
        let so = first.occ[first.up_to - 1].start_offset;
        let eo = last.occ[last.up_to - 1].end_offset;
        out.push([first.pos, last.pos, so, eo]);
    }
    out
}

/// `PhraseWeight.matches` for a `PhraseQuery` (after `rewrite`: no terms is
/// `MatchNoDocsQuery`, one term a `TermQuery`).
fn phrase_matches(
    seg: &OpenSegment<'_>,
    q: &PhraseQuery,
    clause: Arc<Clause>,
    doc: i32,
) -> Result<Option<BoxMatches>> {
    match q.terms.len() {
        0 => return Ok(None),
        1 => {
            let tq = TermQuery::new(q.field.clone(), q.terms[0].clone());
            let as_term = Arc::new(Clause::Term(tq.clone()));
            return term_matches(seg, &tq, as_term, doc);
        }
        _ => {}
    }
    let Some(ft) = seg.fields.field(&q.field) else {
        return Ok(None);
    };
    if !ft.index_options().subsumes_positions() || seg.pos_in.is_none() {
        return Err(Error::IllegalState(format!(
            "field \"{}\" was indexed without position data; cannot run PhraseQuery",
            q.field
        )));
    }
    let mut occs = Vec::with_capacity(q.terms.len());
    for t in &q.terms {
        match occurrences(seg, &q.field, t, doc)? {
            Some(o) => occs.push(o),
            None => return Ok(None),
        }
    }
    let spans = if q.slop == 0 {
        exact_phrase_spans(&occs)
    } else {
        let positions: Vec<Vec<i32>> = occs
            .iter()
            .map(|o| o.iter().map(|p| p.position).collect())
            .collect();
        let offsets: Vec<Vec<(i32, i32)>> = occs
            .iter()
            .map(|o| o.iter().map(|p| (p.start_offset, p.end_offset)).collect())
            .collect();
        let pos_refs: Vec<&[i32]> = positions.iter().map(Vec::as_slice).collect();
        let off_refs: Vec<&[(i32, i32)]> = offsets.iter().map(Vec::as_slice).collect();
        let repeats = PhraseRepeats::for_phrase(&q.terms);
        sloppy_phrase_match_spans(&pos_refs, &off_refs, &repeats, q.slop)
    };
    if spans.is_empty() {
        return Ok(None);
    }
    let spans: Arc<[[i32; 4]]> = spans.into();
    for_field(
        &q.field,
        Box::new(move || {
            Ok(Some(Box::new(ListMatchesIterator {
                query: Arc::clone(&clause),
                intervals: spans.to_vec(),
                at: 0,
            }) as BoxMatchesIterator))
        }),
    )
}

/// `AbstractMultiTermQueryConstantScoreWrapper.matches`:
/// `DisjunctionMatchesIterator.fromTermsEnum` over the query's terms in term
/// order.
fn multi_term_matches(
    seg: &OpenSegment<'_>,
    clause: Arc<Clause>,
    doc: i32,
) -> Result<Option<BoxMatches>> {
    let Some((field, terms, _)) = crate::expanded_terms(seg.fields, &clause)? else {
        return Ok(None);
    };
    let mut per_term: Vec<Arc<[Position]>> = Vec::new();
    for (term, _) in &terms {
        if let Some(o) = occurrences(seg, &field, term, doc)? {
            per_term.push(o.into());
        }
    }
    if per_term.is_empty() {
        return Ok(None);
    }
    let per_term: Arc<[Arc<[Position]>]> = per_term.into();
    for_field(
        &field,
        Box::new(move || {
            let subs = per_term
                .iter()
                .map(|occ| {
                    Box::new(ListMatchesIterator::for_term(Arc::clone(&clause), occ))
                        as BoxMatchesIterator
                })
                .collect();
            disjunction(subs)
        }),
    )
}

/// `BooleanWeight.matches`: `None` if a prohibited clause matches, a
/// required one does not, or fewer `SHOULD`s match than the minimum; else
/// the required and matching optional clauses' matches amalgamated.
fn boolean_matches(
    seg: &OpenSegment<'_>,
    q: &BooleanQuery,
    doc: i32,
) -> Result<Option<BoxMatches>> {
    let mut matches = Vec::new();
    // Java walks the clauses in the query's order; a prohibited or missing
    // required clause ends the walk either way, and only the amalgamation's
    // order (required then optional, as added) is observable.
    for c in &q.must_not {
        if leaf_matches(seg, c, doc)?.is_some() {
            return Ok(None);
        }
    }
    for c in q.must.iter().chain(&q.filter) {
        match leaf_matches(seg, c, doc)? {
            Some(m) => matches.push(m),
            None => return Ok(None),
        }
    }
    let mut should_count = 0usize;
    for c in &q.should {
        if let Some(m) = leaf_matches(seg, c, doc)? {
            matches.push(m);
            should_count += 1;
        }
    }
    if should_count < q.minimum_should_match {
        return Ok(None);
    }
    Ok(from_sub_matches(matches))
}

/// `Weight.matches(context, doc)` for `clause` over one segment, `doc`
/// leaf-local.
///
/// # Errors
/// [`Error::IllegalArgument`] for a clause whose positional matches are not
/// ported (fuzzy, multi-phrase, span); [`Error::IllegalState`] for a phrase
/// over a field without positions.
pub fn leaf_matches(
    seg: &OpenSegment<'_>,
    clause: &Clause,
    doc: i32,
) -> Result<Option<BoxMatches>> {
    match clause {
        Clause::Term(q) => term_matches(seg, q, Arc::new(clause.clone()), doc),
        Clause::Phrase(q) => phrase_matches(seg, q, Arc::new(clause.clone()), doc),
        Clause::Boolean(b) => boolean_matches(seg, b, doc),
        Clause::DisjunctionMax(d) => {
            let mut subs = Vec::new();
            for c in &d.disjuncts {
                if let Some(m) = leaf_matches(seg, c, doc)? {
                    subs.push(m);
                }
            }
            Ok(from_sub_matches(subs))
        }
        Clause::ConstantScore(c) => leaf_matches(seg, &c.inner, doc),
        Clause::Boost(b) => leaf_matches(seg, &b.inner, doc),
        Clause::Prefix(_) | Clause::Wildcard(_) | Clause::Regexp(_) | Clause::TermInSet(_) => {
            multi_term_matches(seg, Arc::new(clause.clone()), doc)
        }
        Clause::MatchNoDocs(_) => Ok(None),
        Clause::Fuzzy(_) | Clause::MultiPhrase(_) | Clause::Span(_) => Err(Error::IllegalArgument(
            format!("Weight.matches is not ported for {}", clause_name(clause)),
        )),
        // `Weight.matches`'s default: every other clause (match-all, points
        // range, exists, ...) matches with no terms where its scorer does.
        #[allow(unreachable_patterns)]
        _ => default_matches(seg, clause, doc),
    }
}

/// `IndexSearcher.createWeight(query).matches(leaf, doc)` for a global
/// document id.
///
/// # Errors
/// [`Error::IllegalArgument`] for a document outside every segment, and what
/// [`leaf_matches`] reports.
pub fn matches(
    searcher: &crate::index_searcher::IndexSearcher<'_, '_>,
    clause: &Clause,
    doc: i32,
) -> Result<Option<BoxMatches>> {
    let leaf = searcher
        .segment_of(doc)
        .ok_or_else(|| Error::IllegalArgument(format!("doc {doc} is in no segment")))?;
    let seg = &searcher.segments()[leaf];
    leaf_matches(seg, clause, doc - seg.doc_base)
}

/// `NamedMatches.wrapQuery(name, query)` over a boolean's `SHOULD` clauses,
/// then `findNamedMatches`: the names of the clauses in `named` that match
/// `doc` (leaf-local), each with its matches -- how a caller attaches names
/// without a `NamedQuery` clause.
pub fn named_matches(
    seg: &OpenSegment<'_>,
    named: &[(String, Clause)],
    doc: i32,
) -> Result<Vec<NamedMatches>> {
    let mut out = Vec::new();
    for (name, clause) in named {
        if let Some(m) = leaf_matches(seg, clause, doc)? {
            out.push(NamedMatches::new(name.clone(), m));
        }
    }
    Ok(out)
}

/// The simple class name of the Java query a clause is, as
/// `MatchesIterator.getQuery()` would report it.
pub fn clause_name(clause: &Clause) -> &'static str {
    match clause {
        Clause::Term(_) => "TermQuery",
        Clause::Phrase(_) => "PhraseQuery",
        Clause::Boolean(_) => "BooleanQuery",
        Clause::DisjunctionMax(_) => "DisjunctionMaxQuery",
        Clause::ConstantScore(_) => "ConstantScoreQuery",
        Clause::Boost(_) => "BoostQuery",
        Clause::Wildcard(_) => "WildcardQuery",
        Clause::Prefix(_) => "PrefixQuery",
        Clause::Fuzzy(_) => "FuzzyQuery",
        Clause::Regexp(_) => "RegexpQuery",
        Clause::Span(_) => "SpanQuery",
        Clause::PointsRange(_) => "PointRangeQuery",
        Clause::MatchAllDocs(_) => "MatchAllDocsQuery",
        Clause::MatchNoDocs(_) => "MatchNoDocsQuery",
        Clause::TermInSet(_) => "TermInSetQuery",
        Clause::MultiPhrase(_) => "MultiPhraseQuery",
        Clause::Exists(_) => "FieldExistsQuery",
        #[allow(unreachable_patterns)]
        _ => "Query",
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn term() -> Arc<Clause> {
        Arc::new(Clause::Term(TermQuery::new("f", b"t".to_vec())))
    }

    fn list(intervals: &[[i32; 4]]) -> BoxMatchesIterator {
        Box::new(ListMatchesIterator {
            query: term(),
            intervals: intervals.to_vec(),
            at: 0,
        })
    }

    fn drain(mut it: BoxMatchesIterator) -> Vec<[i32; 4]> {
        let mut out = Vec::new();
        while it.next().unwrap() {
            out.push([
                it.start_position(),
                it.end_position(),
                it.start_offset(),
                it.end_offset(),
            ]);
            assert!(it.sub_matches().unwrap().is_none());
            assert_eq!(clause_name(it.query()), "TermQuery");
        }
        out
    }

    #[test]
    fn disjunction_merges_by_position_then_offset() {
        let a = list(&[[1, 1, 0, 3], [5, 5, 20, 23]]);
        let b = list(&[[2, 4, 5, 15], [5, 5, 20, 23]]);
        let c = list(&[]);
        let d = disjunction(vec![a, b, c]).unwrap().unwrap();
        assert_eq!(
            drain(d),
            vec![[1, 1, 0, 3], [2, 4, 5, 15], [5, 5, 20, 23], [5, 5, 20, 23]]
        );
        // Without positions, by offsets.
        let a = list(&[[-1, -1, 7, 9]]);
        let b = list(&[[-1, -1, 1, 4], [-1, -1, 7, 8]]);
        let d = disjunction(vec![a, b]).unwrap().unwrap();
        assert_eq!(
            drain(d),
            vec![[-1, -1, 1, 4], [-1, -1, 7, 8], [-1, -1, 7, 9]]
        );
        assert!(disjunction(vec![]).unwrap().is_none());
        let one = disjunction(vec![list(&[[3, 3, 1, 2]])]).unwrap().unwrap();
        assert_eq!(drain(one), vec![[3, 3, 1, 2]]);
        let mut empty = DisjunctionMatchesIterator::new(vec![list(&[]), list(&[])]).unwrap();
        assert!(!empty.next().unwrap());
        assert!(!empty.next().unwrap());
        assert!(empty.sub_matches().unwrap().is_none());
    }

    #[test]
    fn filter_iterator_delegates() {
        let mut f = FilterMatchesIterator {
            inner: list(&[[4, 6, 10, 20]]),
        };
        assert!(f.next().unwrap());
        assert_eq!(
            [
                f.start_position(),
                f.end_position(),
                f.start_offset(),
                f.end_offset()
            ],
            [4, 6, 10, 20]
        );
        assert!(f.sub_matches().unwrap().is_none());
        assert_eq!(clause_name(f.query()), "TermQuery");
        assert!(!f.next().unwrap());
        let unstarted = ListMatchesIterator {
            query: term(),
            intervals: vec![[1, 1, 1, 1]],
            at: 0,
        };
        assert_eq!(unstarted.start_position(), -1);
    }

    fn field(name: &str, intervals: Vec<[i32; 4]>) -> BoxMatches {
        let q = term();
        for_field(
            name,
            Box::new(move || {
                Ok(Some(Box::new(ListMatchesIterator {
                    query: Arc::clone(&q),
                    intervals: intervals.clone(),
                    at: 0,
                }) as BoxMatchesIterator))
            }),
        )
        .unwrap()
        .unwrap()
    }

    #[test]
    fn from_sub_matches_amalgamates() {
        assert!(from_sub_matches(vec![]).is_none());
        let only_no_terms = from_sub_matches(vec![Box::new(MatchWithNoTerms)]).unwrap();
        assert!(only_no_terms.is_match_with_no_terms());
        assert!(only_no_terms.get_matches("f").unwrap().is_none());
        assert!(only_no_terms.fields().is_empty());
        let one = from_sub_matches(vec![
            Box::new(MatchWithNoTerms),
            field("a", vec![[0, 0, 0, 1]]),
        ])
        .unwrap();
        assert_eq!(one.fields(), vec!["a".to_string()]);
        assert!(one.sub_matches().is_empty());
        assert!(one.get_matches("b").unwrap().is_none());
        let many = from_sub_matches(vec![
            field("a", vec![[3, 3, 9, 10]]),
            Box::new(MatchWithNoTerms),
            field("b", vec![[1, 1, 0, 1]]),
            field("a", vec![[1, 1, 2, 3]]),
        ])
        .unwrap();
        assert_eq!(many.fields(), vec!["a".to_string(), "b".to_string()]);
        assert_eq!(many.sub_matches().len(), 4);
        assert_eq!(
            drain(many.get_matches("a").unwrap().unwrap()),
            vec![[1, 1, 2, 3], [3, 3, 9, 10]]
        );
        assert!(many.get_matches("c").unwrap().is_none());
        // `forField` of a supplier with nothing is no matches at all.
        assert!(for_field("x", Box::new(|| Ok(None))).unwrap().is_none());
    }

    #[test]
    fn named_matches_are_found_breadth_first() {
        let inner = NamedMatches::new("inner", field("a", vec![[0, 0, 0, 1]]));
        assert_eq!(inner.name(), "inner");
        assert_eq!(inner.fields(), vec!["a".to_string()]);
        assert!(inner.get_matches("a").unwrap().is_some());
        let tree = from_sub_matches(vec![
            Box::new(NamedMatches::new("x", Box::new(inner))),
            Box::new(NamedMatches::new("y", field("b", vec![[1, 1, 1, 2]]))),
        ])
        .unwrap();
        let names: Vec<&str> = find_named_matches(tree.as_ref())
            .iter()
            .map(|n| n.name())
            .collect();
        assert_eq!(names, vec!["x", "y", "inner"]);
    }

    fn occ(position: i32, start: i32, end: i32) -> Position {
        Position {
            position,
            start_offset: start,
            end_offset: end,
            payload: Vec::new(),
        }
    }

    #[test]
    fn exact_phrase_spans_follow_the_lead() {
        // "a b" over a(0) b(1) a(2) x(3) a(4) b(5).
        let a = vec![occ(0, 0, 1), occ(2, 4, 5), occ(4, 8, 9)];
        let b = vec![occ(1, 2, 3), occ(5, 10, 11)];
        assert_eq!(
            exact_phrase_spans(&[a.clone(), b.clone()]),
            vec![[0, 1, 0, 3], [4, 5, 8, 11]]
        );
        // "b a": b(1) a(2) only.
        assert_eq!(exact_phrase_spans(&[b, a]), vec![[1, 2, 2, 5]]);
        assert!(exact_phrase_spans(&[]).is_empty());
    }
}
