//! `SpanWeight`/`SpanScorer` (`org.apache.lucene.queries.spans`, Lucene
//! 10.5.0): a [`SpanQuery`] as a scorer in the tree, scored through the
//! searcher's similarity.
//!
//! `SpanScorer.setFreqCurrentDoc` walks the document's spans and sums
//! `1 / (1 + width)` over them -- in `double`, into a `float`, one span at a
//! time (`freq += (1.0 / (1.0 + spans.width()))`) -- and scores
//! `simScorer.score(freq, norm)`, the norm of the query's field as stored
//! (`1` without one). `SpanWeight.buildSimWeight` builds that scorer once,
//! from `searcher.collectionStatistics(query.getField())` and the
//! `termStatistics` of every distinct leaf term with a document, in `Term`
//! order (`SpanQuery.getTermStates` fills a `TreeMap`); with none, nothing
//! matches.
//!
//! `width()` is each `Spans`' own: `0` for a `TermSpans`, the summed gaps
//! between adjacent sub-spans (`matchWidth`) for `NearSpansOrdered`,
//! `maxEndPosition - top().startPosition()` for `NearSpansUnordered`, and the
//! current sub-span's for a `SpanOrQuery`.
//!
//! # What differs from Java
//!
//! Like [`super::extended`]'s phrase queries, the matches are resolved when
//! the scorer is built rather than through a lazy `Spans` iterator: every
//! document holding a leaf term is walked over its decoded positions
//! ([`crate::span_leaf_positions`]) with the same `NearSpans*` walks
//! ([`crate::near_spans`]) the matching path uses, and the scorer is a
//! [`DocList`]. A span's sequence here keeps every span the Java iterator
//! emits, repeats included (the matching path's extents drop consecutive
//! repeats, which `freq` must count). Two spans of a `SpanOrQuery` with the
//! same start and end come out in clause order; Java's `SpanPositionQueue`
//! leaves that tie unspecified.

use std::collections::HashMap;

use super::build::LeafContext;
use super::extended::{collection_of, norms_cursor, sim_scorer, term_entry};
use super::leaf::DocList;
use super::{BoxScorer, Mode, NO_MORE_DOCS};
use crate::query::SpanQuery;
use crate::Result;

/// One leaf `(field, term)`'s positions in one document (the tests'
/// input shape).
#[cfg(test)]
type DocPositions = HashMap<crate::SpanLeafKey, Vec<i32>>;

/// `SpanQuery.getField()`: a term's field, else the first clause's.
fn span_field(q: &SpanQuery) -> Option<&str> {
    match q {
        SpanQuery::SpanTerm { field, .. } => Some(field),
        SpanQuery::SpanNear { clauses, .. } | SpanQuery::SpanOr { clauses } => {
            clauses.first().and_then(span_field)
        }
    }
}

/// A [`SpanQuery`] with each `SpanTerm` replaced by its index among the
/// query's sorted, distinct leaves, so a document's positions are looked up
/// by index rather than hashed by `(field, term)`.
enum Compiled {
    Term(usize),
    Or(Vec<Compiled>),
    Near {
        clauses: Vec<Compiled>,
        slop: i64,
        in_order: bool,
    },
}

/// `(startPosition(), endPosition(), width())` of one emitted span.
type Emission = (i32, i32, i64);

/// Where a leaf's positions come from, document by document.
pub(crate) enum LeafPositions<'a> {
    /// The term is not in this segment.
    Absent,
    /// `TermsEnum.postings(POSITIONS)`: decoded only for the documents asked.
    Lazy(Box<lucene_codecs::postings::PositionsCursor<'a>>),
    /// A pulsed singleton's one posting (live documents only), decoded up
    /// front, and the entry the leaf is on.
    Flat(crate::TermDocPositions, usize),
}

impl<'a> LeafPositions<'a> {
    /// `field:term`'s positions in this segment: a lazy cursor when the term
    /// has a `.doc` stream, else its one (live) posting decoded.
    pub(crate) fn open(
        ctx: &LeafContext<'a>,
        pos_in: &lucene_codecs::postings::PosInput<'a>,
        field: &str,
        term: &[u8],
    ) -> Result<Self> {
        Ok(match (ctx.fields.field(field), ctx.doc_in) {
            (Some(field_terms), Some(doc_in))
                if field_terms
                    .try_seek_exact(term)?
                    .is_some_and(|stats| stats.doc_freq > 1) =>
            {
                field_terms
                    .lazy_positions(term, doc_in, pos_in)?
                    .map_or(LeafPositions::Absent, |c| LeafPositions::Lazy(Box::new(c)))
            }
            _ => crate::term_doc_positions(
                ctx.fields,
                ctx.doc_in,
                pos_in,
                ctx.pay_in,
                ctx.live_docs,
                field,
                term,
            )?
            .map_or(LeafPositions::Absent, |flat| LeafPositions::Flat(flat, 0)),
        })
    }

    /// `DocIdSetIterator.advance(target)` unless already there:
    /// the leaf's first document at or after `target`.
    pub(crate) fn advance(&mut self, target: i32) -> Result<i32> {
        Ok(match self {
            LeafPositions::Absent => NO_MORE_DOCS,
            LeafPositions::Lazy(cursor) => {
                if cursor.doc_id() < target {
                    cursor.advance(target)?
                } else {
                    cursor.doc_id()
                }
            }
            LeafPositions::Flat((docs, _, _), at) => {
                while *at < docs.len() && docs[*at] < target {
                    *at = at.saturating_add(1);
                }
                docs.get(*at).copied().unwrap_or(NO_MORE_DOCS)
            }
        })
    }

    /// `DocIdSetIterator.nextDoc()` from `current`, the document the leaf
    /// is on: a cursor's one-slot step rather than an `advance`.
    pub(crate) fn next_doc(&mut self, current: i32) -> Result<i32> {
        match self {
            LeafPositions::Lazy(cursor) => Ok(cursor.next_doc()?),
            _ => self.advance(current.saturating_add(1)),
        }
    }

    /// The leaf's frequency in `doc`: `0` when it is not on `doc`.
    pub(crate) fn freq_at(&self, doc: i32) -> u64 {
        match self {
            LeafPositions::Absent => 0,
            LeafPositions::Lazy(cursor) => {
                if cursor.doc_id() == doc {
                    u64::try_from(cursor.freq()).unwrap_or(0)
                } else {
                    0
                }
            }
            LeafPositions::Flat((docs, _, ranges), at) => {
                if docs.get(*at) == Some(&doc) {
                    let (from, to) = ranges[*at];
                    u64::from(to.saturating_sub(from))
                } else {
                    0
                }
            }
        }
    }

    /// The leaf's positions in `doc`, appended to `out`; nothing when the
    /// leaf is not on `doc`.
    pub(crate) fn positions_at(&mut self, doc: i32, out: &mut Vec<i32>) -> Result<()> {
        match self {
            LeafPositions::Absent => {}
            LeafPositions::Lazy(cursor) => {
                if cursor.doc_id() == doc {
                    cursor.positions_into(out)?;
                }
            }
            LeafPositions::Flat((docs, positions, ranges), at) => {
                if docs.get(*at) == Some(&doc) {
                    let (from, to) = ranges[*at];
                    out.extend_from_slice(&positions[from as usize..to as usize]);
                }
            }
        }
        Ok(())
    }

    /// Asks a lazy cursor to read each position's payload as it reads the
    /// position (`PostingsEnum.PAYLOADS`), for
    /// [`Self::next_position_with_payload`]; `false` where it will not (a
    /// pulsed singleton, a retired format, no `.pay`), and the leaf's
    /// occurrences are then read per document by [`Self::occurrences_at`].
    pub(crate) fn stream_payloads(&mut self, ctx: &LeafContext<'a>) -> bool {
        match (self, ctx.pay_in) {
            (LeafPositions::Lazy(cursor), Some(pay)) => cursor.read_payloads(pay).is_ok(),
            _ => false,
        }
    }

    /// Whether this is a lazy cursor, whose positions
    /// [`Self::next_position`] reads one at a time.
    pub(crate) fn is_lazy(&self) -> bool {
        matches!(self, LeafPositions::Lazy(_))
    }

    /// `nextPosition()` of the document a lazy cursor is on.
    pub(crate) fn next_position(&mut self) -> Result<i32> {
        match self {
            LeafPositions::Lazy(cursor) => Ok(cursor.next_position()?),
            _ => Err(crate::Error::IllegalState(
                "positions are streamed from a lazy cursor only".into(),
            )),
        }
    }

    /// `nextPosition()` and `getPayload()` of the document the cursor is on,
    /// once [`Self::stream_payloads`] said yes.
    pub(crate) fn next_position_with_payload(&mut self) -> Result<(i32, Option<&[u8]>)> {
        match self {
            LeafPositions::Lazy(cursor) => {
                let position = cursor.next_position()?;
                Ok((position, cursor.payload()?))
            }
            _ => Err(crate::Error::IllegalState(
                "payloads are streamed from a lazy cursor only".into(),
            )),
        }
    }

    /// The leaf's occurrences in `doc` -- positions with their offsets and
    /// payloads -- replacing `out`'s contents; nothing when the leaf is not
    /// on `doc`. A lazy cursor reads them where it stands
    /// (`PositionsCursor::occurrences_into`); a pulsed singleton, or a
    /// retired format's postings, through `field_terms`' one-document read.
    pub(crate) fn occurrences_at(
        &mut self,
        ctx: &LeafContext<'a>,
        field_terms: &lucene_codecs::blocktree::FieldTerms,
        term: &[u8],
        doc: i32,
        out: &mut Vec<lucene_codecs::postings::Position>,
    ) -> Result<()> {
        out.clear();
        let Some(pos_in) = ctx.pos_in else {
            return Err(crate::Error::MissingPosInput);
        };
        if let LeafPositions::Lazy(cursor) = self {
            if cursor.doc_id() != doc {
                return Ok(());
            }
            match cursor.occurrences_into(pos_in, ctx.pay_in, out) {
                Err(lucene_codecs::postings::Error::Unsupported(_)) => {}
                other => return other.map_err(Into::into),
            }
        }
        if self.freq_at(doc) == 0 {
            return Ok(());
        }
        if let Some(found) =
            field_terms.occurrences_for_doc(term, ctx.doc_in, pos_in, ctx.pay_in, doc)?
        {
            *out = found;
        }
        Ok(())
    }
}

/// How many times `c` names each leaf, when it is only terms and ors of
/// terms (every span `width() == 0`); `None` when it has a near.
fn term_counts(c: &Compiled, leaves: usize) -> Option<Vec<u64>> {
    fn walk(c: &Compiled, counts: &mut [u64]) -> bool {
        match c {
            Compiled::Term(i) => {
                counts[*i] = counts[*i].saturating_add(1);
                true
            }
            Compiled::Or(clauses) => clauses.iter().all(|cl| walk(cl, counts)),
            Compiled::Near { .. } => false,
        }
    }
    let mut counts = vec![0u64; leaves];
    walk(c, &mut counts).then_some(counts)
}

/// The span query's approximation advanced to `target`: the first document
/// at or after it every clause of each near and some clause of each or is
/// on (`ConjunctionDISI`/`DisjunctionDISIApproximation` over the term
/// spans), leaving every leaf cursor at or after `target`.
fn approximate(c: &Compiled, target: i32, sources: &mut [LeafPositions<'_>]) -> Result<i32> {
    match c {
        Compiled::Term(i) => sources[*i].advance(target),
        Compiled::Or(clauses) => {
            let mut min = NO_MORE_DOCS;
            for cl in clauses {
                min = min.min(approximate(cl, target, sources)?);
            }
            Ok(min)
        }
        Compiled::Near { clauses, .. } => {
            if clauses.is_empty() {
                return Ok(NO_MORE_DOCS);
            }
            let mut target = target;
            loop {
                let mut max = target;
                for cl in clauses {
                    let doc = approximate(cl, target, sources)?;
                    if doc == NO_MORE_DOCS {
                        return Ok(NO_MORE_DOCS);
                    }
                    max = max.max(doc);
                }
                if max == target {
                    return Ok(target);
                }
                target = max;
            }
        }
    }
}

fn compile(q: &SpanQuery, leaves: &[crate::SpanLeafKey]) -> Compiled {
    match q {
        SpanQuery::SpanTerm { field, term } => Compiled::Term(
            leaves
                .binary_search_by(|(f, t)| {
                    (f.as_str(), t.as_slice()).cmp(&(field.as_str(), term.as_slice()))
                })
                .expect("every leaf was collected"),
        ),
        SpanQuery::SpanOr { clauses } => {
            Compiled::Or(clauses.iter().map(|c| compile(c, leaves)).collect())
        }
        SpanQuery::SpanNear {
            clauses,
            slop,
            in_order,
        } => Compiled::Near {
            clauses: clauses.iter().map(|c| compile(c, leaves)).collect(),
            slop: i64::from(*slop),
            in_order: *in_order,
        },
    }
}

/// Buffers [`emit`] reuses from one document to the next.
#[derive(Default)]
struct Scratch {
    emissions: Vec<Vec<Emission>>,
    pairs: Vec<Vec<Vec<(i32, i32)>>>,
}

/// The `Spans` of `c` over one document (`pos[i]`: leaf `i`'s positions in
/// it, empty when the leaf is not in it), walked to `NO_MORE_POSITIONS`,
/// appended to `out` in emission order.
fn emit(c: &Compiled, pos: &[Vec<i32>], out: &mut Vec<Emission>, scratch: &mut Scratch) {
    match c {
        // `TermSpans`: one span per position, `width() == 0`.
        Compiled::Term(i) => {
            out.extend(pos[*i].iter().map(|&p| (p, p.saturating_add(1), 0)));
        }
        // `SpanOrQuery`'s spans: every clause matching in the document, merged
        // by `(start, end)` (`SpanPositionQueue`), repeats kept.
        Compiled::Or(clauses) => {
            let from = out.len();
            for cl in clauses {
                emit(cl, pos, out, scratch);
            }
            out[from..].sort_by_key(|&(s, e, _)| (s, e));
        }
        Compiled::Near {
            clauses,
            slop,
            in_order,
        } => {
            if clauses.is_empty() {
                return;
            }
            let mut per = scratch.pairs.pop().unwrap_or_default();
            per.resize_with(clauses.len(), Vec::new);
            let mut tmp = scratch.emissions.pop().unwrap_or_default();
            let mut all_match = true;
            for (cl, spans) in clauses.iter().zip(per.iter_mut()) {
                spans.clear();
                if all_match {
                    if let Compiled::Term(i) = cl {
                        spans.extend(pos[*i].iter().map(|&p| (p, p.saturating_add(1))));
                    } else {
                        tmp.clear();
                        emit(cl, pos, &mut tmp, scratch);
                        spans.extend(tmp.iter().map(|&(s, e, _)| (s, e)));
                    }
                    all_match = !spans.is_empty();
                }
            }
            scratch.emissions.push(tmp);
            if all_match {
                near(&per[..clauses.len()], *slop, *in_order, out);
            }
            scratch.pairs.push(per);
        }
    }
}

/// `NearSpansOrdered`/`NearSpansUnordered` over the clauses' spans in one
/// document, each match appended to `out` with its `width()`.
fn near(per: &[Vec<(i32, i32)>], slop: i64, in_order: bool, out: &mut Vec<Emission>) {
    const STACK: usize = 8;
    let mut on_stack: [&[(i32, i32)]; STACK] = [&[]; STACK];
    let on_heap: Vec<&[(i32, i32)]>;
    let slices: &[&[(i32, i32)]] = if per.len() <= STACK {
        for (slot, spans) in on_stack.iter_mut().zip(per) {
            *slot = spans.as_slice();
        }
        &on_stack[..per.len()]
    } else {
        on_heap = per.iter().map(Vec::as_slice).collect();
        &on_heap
    };
    if in_order {
        // `NearSpansOrdered.width()`: `matchWidth`, the gaps
        // `stretchToOrder` summed.
        crate::near_spans::ordered_walk(slices, slop, |_, start, end, width| {
            out.push((start, end, width));
        });
    } else {
        // `NearSpansUnordered.width()`:
        // `maxEndPosition - top().startPosition()`.
        crate::near_spans::for_each_unordered_match(slices, slop, |_, start, end| {
            out.push((start, end, i64::from(end) - i64::from(start)));
        });
    }
}

/// The leaves of an in-order `SpanNearQuery` of terms alone (two or more):
/// the shape [`ordered_terms_freq`] scores without materializing spans.
fn ordered_term_leaves(c: &Compiled) -> Option<(Vec<usize>, i64)> {
    let Compiled::Near {
        clauses,
        slop,
        in_order: true,
    } = c
    else {
        return None;
    };
    if clauses.len() < 2 {
        return None;
    }
    let leaves = clauses
        .iter()
        .map(|cl| match cl {
            Compiled::Term(i) => Some(*i),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()?;
    Some((leaves, *slop))
}

/// [`freq_of`] of [`emit`] for an in-order near of terms -- `clauses[i]` is
/// clause `i`'s leaf in `pos` -- fused: [`crate::near_spans::ordered_walk`]
/// over the positions themselves (a term's span is `[p, p + 1)`), each
/// match's `1 / (1 + width)` added as it is found, in the order `emit`
/// lists the matches. The same float sum, without the per-document span
/// lists.
///
/// With a slop of 0 every match has width 0 (the walk never moves a clause
/// before the previous one's end, so no gap is negative) and adds exactly
/// 1: the float sum is the number of matches, up to 2^24, where adding 1
/// to a `float` no longer changes it.
fn ordered_terms_freq(clauses: &[usize], slop: i64, pos: &[Vec<i32>]) -> f32 {
    if slop == 0 {
        let mut n = 0u32;
        ordered_terms_walk(clauses, slop, pos, |_| n = n.saturating_add(1));
        return n.min(1 << 24) as f32;
    }
    let mut freq = 0.0f32;
    ordered_terms_walk(clauses, slop, pos, |width| {
        freq = (f64::from(freq) + 1.0 / (1.0 + width as f64)) as f32;
    });
    freq
}

/// [`crate::near_spans::ordered_walk`] over the clauses' term positions,
/// `on_match` called with each match's width.
fn ordered_terms_walk(
    clauses: &[usize],
    slop: i64,
    pos: &[Vec<i32>],
    mut on_match: impl FnMut(i64),
) {
    if let [a, b] = clauses {
        let (first, second) = (&pos[*a], &pos[*b]);
        let mut at = 0usize;
        for &start in first {
            let first_end = start.saturating_add(1);
            while at < second.len() && second[at] < first_end {
                at += 1;
            }
            let Some(&next) = second.get(at) else {
                return;
            };
            let width = i64::from(next) - i64::from(first_end);
            if width <= slop {
                on_match(width);
            }
        }
        return;
    }
    if clauses.iter().any(|&i| pos[i].is_empty()) {
        return;
    }
    const STACK: usize = 8;
    let mut on_stack = [0usize; STACK];
    let mut on_heap = Vec::new();
    let cursor: &mut [usize] = if clauses.len() <= STACK {
        &mut on_stack[..clauses.len()]
    } else {
        on_heap.resize(clauses.len(), 0);
        &mut on_heap
    };
    for &start in &pos[clauses[0]] {
        let mut prev_end = start.saturating_add(1);
        let mut width: i64 = 0;
        for (k, &leaf) in clauses.iter().enumerate().skip(1) {
            let spans = &pos[leaf];
            let mut at = cursor[k];
            while at < spans.len() && spans[at] < prev_end {
                at += 1;
            }
            let Some(&next) = spans.get(at) else {
                return;
            };
            cursor[k] = at;
            width = width.saturating_add(i64::from(next) - i64::from(prev_end));
            prev_end = next.saturating_add(1);
        }
        if width <= slop {
            on_match(width);
        }
    }
}

/// `SpanScorer.setFreqCurrentDoc`: `sum(1 / (1 + width))` over the
/// document's spans, each step rounded to `float` as Java's compound
/// assignment rounds it; `0` when it has none.
fn freq_of(spans: &[Emission]) -> f32 {
    let mut freq = 0.0f32;
    for &(_, _, width) in spans {
        freq = (f64::from(freq) + 1.0 / (1.0 + width as f64)) as f32;
    }
    freq
}

/// [`emit`] over a test's per-leaf positions.
#[cfg(test)]
fn emissions(q: &SpanQuery, doc: &DocPositions) -> Vec<Emission> {
    let mut leaves = Vec::new();
    crate::collect_span_leaves(q, &mut leaves);
    leaves.sort_unstable();
    leaves.dedup();
    let compiled = compile(q, &leaves);
    let pos: Vec<Vec<i32>> = leaves
        .iter()
        .map(|k| doc.get(k).cloned().unwrap_or_default())
        .collect();
    let mut out = Vec::new();
    emit(&compiled, &pos, &mut out, &mut Scratch::default());
    out
}

#[cfg(test)]
fn sloppy_freq(q: &SpanQuery, doc: &DocPositions) -> f32 {
    freq_of(&emissions(q, doc))
}

/// `SpanWeight.scorerSupplier(context)`: the documents of this segment `q`
/// matches, with their scores when `mode` needs them, times `boost`
/// (`createWeight`'s, which `buildSimWeight` hands the similarity).
pub(crate) fn span<'a>(
    ctx: &LeafContext<'a>,
    q: &SpanQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    Ok(span_doc_scores(ctx, q, boost, mode.needs_scores())?
        .map(|(docs, scores)| -> BoxScorer<'a> { Box::new(DocList::new(docs, scores)) }))
}

/// [`span`]'s matches as ascending live documents and, when `needs_scores`,
/// their scores; `None` when nothing in this segment matches.
pub(crate) fn span_doc_scores(
    ctx: &LeafContext<'_>,
    q: &SpanQuery,
    boost: f32,
    needs_scores: bool,
) -> Result<Option<(Vec<i32>, Vec<f32>)>> {
    let Some(field) = span_field(q) else {
        return Ok(None);
    };
    // `buildSimWeight`: the distinct leaf terms, in `Term` order, that have a
    // document somewhere in the reader.
    let mut leaves = Vec::new();
    crate::collect_span_leaves(q, &mut leaves);
    leaves.sort_unstable();
    leaves.dedup();
    let mut term_stats = Vec::with_capacity(leaves.len());
    for (f, t) in &leaves {
        if let Some(entry) = term_entry(ctx, f, t)? {
            if entry.doc_freq > 0 {
                term_stats.push(entry.term_statistics());
            }
        }
    }
    if term_stats.is_empty() {
        // "no terms at all exist": no leaf has a document, so none matches.
        return Ok(None);
    }
    let scorer = if needs_scores {
        let first = leaves
            .iter()
            .find(|(f, _)| f == field)
            .map(|(_, t)| t.as_slice())
            .unwrap_or_default();
        // `searcher.collectionStatistics(field)` is `null` only when no
        // document has the field, and then no leaf of it has a document.
        let Some(collection) = term_entry(ctx, field, first)?
            .as_ref()
            .and_then(collection_of)
        else {
            return Ok(None);
        };
        Some(sim_scorer(ctx, field, boost, &collection, &term_stats))
    } else {
        None
    };
    // `SpanTermQuery`'s `TermSpans`: each distinct leaf's positions read per
    // document from a lazy cursor (a pulsed singleton, with no `.doc` stream
    // to walk, from its one decoded posting). The cursors themselves are the
    // approximation (`SpanNearQuery`'s conjunction, `SpanOrQuery`'s
    // disjunction), so only the documents the query can match are visited
    // and only their positions are decoded.
    let Some(pos_in) = ctx.pos_in else {
        return Err(crate::Error::MissingPosInput);
    };
    let mut sources = Vec::with_capacity(leaves.len());
    for (f, t) in &leaves {
        sources.push(LeafPositions::open(ctx, pos_in, f, t)?);
    }
    let compiled = compile(q, &leaves);
    let mut norms = norms_cursor(ctx, field);
    let (mut docs, mut scores) = (Vec::new(), Vec::new());
    let mut pos: Vec<Vec<i32>> = vec![Vec::new(); sources.len()];
    let mut spans = Vec::new();
    let mut scratch = Scratch::default();
    // A query of terms and ors of terms emits only width-0 spans.
    let term_counts = term_counts(&compiled, sources.len());
    // An in-order near of terms is scored straight from the positions.
    let ordered_terms = ordered_term_leaves(&compiled);
    let mut doc = approximate(&compiled, 0, &mut sources)?;
    while doc != NO_MORE_DOCS {
        if ctx.live_docs.is_none_or(|live| live.get_doc(doc)) {
            let freq = match &term_counts {
                // Every span is a term's, `width() == 0`: each adds exactly
                // 1 to the float sum, which is then the number of spans --
                // each leaf's frequency, times how often the query names it.
                Some(counts) => {
                    let mut n = 0u64;
                    for (source, &times) in sources.iter().zip(counts) {
                        n = n.saturating_add(source.freq_at(doc).saturating_mul(times));
                    }
                    n as f32
                }
                None => {
                    for (buf, source) in pos.iter_mut().zip(&mut sources) {
                        buf.clear();
                        source.positions_at(doc, buf)?;
                    }
                    match &ordered_terms {
                        Some((clauses, slop)) => ordered_terms_freq(clauses, *slop, &pos),
                        None => {
                            spans.clear();
                            emit(&compiled, &pos, &mut spans, &mut scratch);
                            freq_of(&spans)
                        }
                    }
                }
            };
            if freq != 0.0 {
                docs.push(doc);
                if let Some(scorer) = &scorer {
                    let norm = match norms.as_mut() {
                        Some(n) => n.norm_long(doc)?.unwrap_or(1),
                        None => 1,
                    };
                    scores.push(scorer.score(freq, norm));
                }
            }
        }
        doc = approximate(&compiled, doc.saturating_add(1), &mut sources)?;
    }
    if docs.is_empty() {
        return Ok(None);
    }
    Ok(Some((docs, scores)))
}

/// [`span_doc_scores`] under the default BM25 over one segment's inputs,
/// for the materializing and explain paths: `(doc, score)`, ascending.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve_span(
    fields: &lucene_codecs::blocktree::BlockTreeFields,
    doc_in: Option<&lucene_codecs::postings::DocInput<'_>>,
    pos_in: Option<&lucene_codecs::postings::PosInput<'_>>,
    pay_in: Option<&lucene_codecs::postings::PayInput<'_>>,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    norms: Option<&HashMap<String, crate::FieldNorms<'_>>>,
    global: Option<&crate::GlobalStats>,
    q: &SpanQuery,
) -> Result<Vec<(i32, f32)>> {
    let ctx = LeafContext {
        fields,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        points: None,
        norms,
        global,
        max_doc: None,
        cache: None,
        reader: None,
        similarity: None,
    };
    Ok(match span_doc_scores(&ctx, q, 1.0, true)? {
        Some((docs, scores)) => docs.into_iter().zip(scores).collect(),
        None => Vec::new(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn doc(entries: &[(&str, &[i32])]) -> DocPositions {
        entries
            .iter()
            .map(|(t, ps)| (("f".to_string(), t.as_bytes().to_vec()), ps.to_vec()))
            .collect()
    }

    fn term(t: &str) -> SpanQuery {
        SpanQuery::span_term("f", t)
    }

    /// The fused in-order near of terms scores as `emit` + `freq_of` do,
    /// bit for bit: random documents, two to five clauses (a repeated term
    /// among them), slops from 0 to 6, and empty clauses.
    #[test]
    fn the_fused_ordered_near_of_terms_scores_as_the_spans_do() {
        let mut seed = 0x2545_f491_4f6c_dd1du64;
        let mut next = move |n: u64| {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed % n
        };
        let names = ["a", "b", "c", "d"];
        let mut fused_hits = 0;
        for round in 0..3000 {
            let mut entries: Vec<(&str, Vec<i32>)> = Vec::new();
            for name in names {
                let mut ps: Vec<i32> = (0..next(12)).map(|_| next(40) as i32).collect();
                ps.sort_unstable();
                ps.dedup();
                entries.push((name, ps));
            }
            let d: DocPositions = entries
                .iter()
                .map(|(t, ps)| (("f".to_string(), t.as_bytes().to_vec()), ps.clone()))
                .collect();
            // up to ten clauses: past the eight the walks keep on the stack
            let n = if round % 50 == 0 {
                9 + round % 2
            } else {
                2 + (round % 4)
            };
            let clauses: Vec<SpanQuery> = (0..n).map(|_| term(names[next(4) as usize])).collect();
            let slop = next(7) as u32;
            let q = SpanQuery::span_near(clauses, slop, true);
            let mut leaves = Vec::new();
            crate::collect_span_leaves(&q, &mut leaves);
            leaves.sort_unstable();
            leaves.dedup();
            let compiled = compile(&q, &leaves);
            let pos: Vec<Vec<i32>> = leaves
                .iter()
                .map(|k| d.get(k).cloned().unwrap_or_default())
                .collect();
            let (fused, s) = ordered_term_leaves(&compiled).expect("an in-order near of terms");
            let got = ordered_terms_freq(&fused, s, &pos);
            assert_eq!(got.to_bits(), sloppy_freq(&q, &d).to_bits(), "{q:?} {d:?}");
            if got > 0.0 {
                fused_hits += 1;
            }
        }
        assert!(fused_hits > 300, "{fused_hits}");
        // only an in-order near of two or more terms is fused
        let leaves = vec![
            ("f".to_string(), b"a".to_vec()),
            ("f".to_string(), b"b".to_vec()),
        ];
        let unordered = compile(
            &SpanQuery::span_near([term("a"), term("b")], 1, false),
            &leaves,
        );
        assert!(ordered_term_leaves(&unordered).is_none());
        let one = compile(&SpanQuery::span_near([term("a")], 1, true), &leaves);
        assert!(ordered_term_leaves(&one).is_none());
        let nested = compile(
            &SpanQuery::span_near([term("a"), SpanQuery::span_or([term("a")])], 1, true),
            &leaves,
        );
        assert!(ordered_term_leaves(&nested).is_none());
        assert!(ordered_term_leaves(&Compiled::Term(0)).is_none());
        // an absent leaf is on no document; a near of no clauses matches none
        let mut absent = [LeafPositions::Absent];
        assert_eq!(absent[0].advance(0).unwrap(), NO_MORE_DOCS);
        assert_eq!(absent[0].freq_at(0), 0);
        let mut out = vec![];
        absent[0].positions_at(0, &mut out).unwrap();
        assert!(out.is_empty());
        let empty = Compiled::Near {
            clauses: vec![],
            slop: 0,
            in_order: true,
        };
        assert_eq!(approximate(&empty, 0, &mut absent).unwrap(), NO_MORE_DOCS);
    }

    #[test]
    fn a_term_span_has_width_zero_and_counts_every_position() {
        let d = doc(&[("a", &[0, 3, 7])]);
        assert_eq!(
            emissions(&term("a"), &d),
            vec![(0, 1, 0), (3, 4, 0), (7, 8, 0)]
        );
        assert_eq!(sloppy_freq(&term("a"), &d), 3.0);
        assert_eq!(sloppy_freq(&term("b"), &d), 0.0);
    }

    #[test]
    fn an_or_keeps_repeats_in_start_end_order() {
        let d = doc(&[("a", &[2, 5]), ("b", &[1, 5])]);
        let q = SpanQuery::span_or([term("a"), term("b"), term("a")]);
        assert_eq!(
            emissions(&q, &d),
            vec![
                (1, 2, 0),
                (2, 3, 0),
                (2, 3, 0),
                (5, 6, 0),
                (5, 6, 0),
                (5, 6, 0)
            ]
        );
        assert_eq!(sloppy_freq(&q, &d), 6.0);
    }

    #[test]
    fn near_widths_are_the_gaps_in_order_and_the_extent_out_of_order() {
        // a@0 b@2: in order, one gap of 1 -> width 1, freq 1/2.
        let d = doc(&[("a", &[0]), ("b", &[2])]);
        let ordered = SpanQuery::span_near([term("a"), term("b")], 1, true);
        assert_eq!(emissions(&ordered, &d), vec![(0, 3, 1)]);
        assert_eq!(sloppy_freq(&ordered, &d), 0.5);
        // Out of order: `maxEnd - start` = 3 -> freq 1/4.
        let unordered = SpanQuery::span_near([term("b"), term("a")], 1, false);
        assert_eq!(emissions(&unordered, &d), vec![(0, 3, 3)]);
        assert_eq!(sloppy_freq(&unordered, &d), 0.25);
        // Too far apart for slop 0; an empty clause list or a missing clause
        // matches nothing.
        let tight = SpanQuery::span_near([term("a"), term("b")], 0, true);
        assert!(emissions(&tight, &d).is_empty());
        assert!(emissions(&SpanQuery::span_near(std::iter::empty(), 0, true), &d).is_empty());
        let missing = SpanQuery::span_near([term("a"), term("z")], 5, false);
        assert!(emissions(&missing, &d).is_empty());
    }

    #[test]
    fn the_freq_sum_rounds_to_float_at_every_step() {
        // Three in-order spans with a gap of 2 (1/3 each): summed in `double` and rounded
        // to `float` each time, as `freq += 1.0 / (1.0 + width)` does.
        let d = doc(&[("a", &[0, 10, 20]), ("b", &[3, 13, 23])]);
        let q = SpanQuery::span_near([term("a"), term("b")], 2, true);
        let mut want = 0.0f32;
        for _ in 0..3 {
            want = (f64::from(want) + 1.0 / 3.0) as f32;
        }
        assert_eq!(sloppy_freq(&q, &d).to_bits(), want.to_bits());
    }

    #[test]
    fn the_query_field_is_the_first_leafs() {
        assert_eq!(span_field(&term("a")), Some("f"));
        let or = SpanQuery::span_or([SpanQuery::span_term("g", "x"), term("a")]);
        assert_eq!(span_field(&or), Some("g"));
        assert_eq!(span_field(&SpanQuery::span_or(std::iter::empty())), None);
    }
    /// A leaf's positions, payloads and occurrences, over the spans
    /// fixture's real postings: a lazy cursor's, a pulsed singleton's and an
    /// absent term's.
    #[test]
    fn leaf_positions_read_positions_payloads_and_occurrences() {
        let dir = std::path::Path::new(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/spans/index"
        ));
        let reader =
            crate::directory_reader::DirectoryReader::open(&lucene_store::FsDirectory::open(dir))
                .unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        for seg in &segments {
            let ctx = LeafContext {
                fields: seg.fields,
                doc_in: seg.doc_in,
                pos_in: seg.pos_in,
                pay_in: seg.pay_in,
                live_docs: seg.live_docs,
                points: None,
                norms: None,
                global: None,
                max_doc: seg.max_doc,
                cache: None,
                reader: None,
                similarity: None,
            };
            let pos_in = ctx.pos_in.unwrap();
            let ft = ctx.fields.field("pay").unwrap();
            for term in [b"apple".as_slice(), b"zeta", b"nosuch"] {
                let mut leaf = LeafPositions::open(&ctx, pos_in, "pay", term).unwrap();
                let mut streamed = LeafPositions::open(&ctx, pos_in, "pay", term).unwrap();
                let streams = streamed.stream_payloads(&ctx);
                assert_eq!(streams, leaf.is_lazy(), "{term:?}");
                let mut doc = leaf.next_doc(-1).unwrap();
                let mut out = Vec::new();
                while doc != NO_MORE_DOCS {
                    assert_eq!(streamed.advance(doc).unwrap(), doc);
                    leaf.occurrences_at(&ctx, ft, term, doc, &mut out).unwrap();
                    let want = ft
                        .occurrences_for_doc(term, ctx.doc_in, pos_in, ctx.pay_in, doc)
                        .unwrap()
                        .unwrap();
                    assert_eq!(out, want, "{term:?} doc {doc}");
                    if streams {
                        for o in &want {
                            let (p, payload) = streamed.next_position_with_payload().unwrap();
                            assert_eq!(p, o.position);
                            assert_eq!(payload.unwrap_or_default(), o.payload.as_slice());
                        }
                    } else {
                        assert!(streamed.next_position().is_err());
                        assert!(streamed.next_position_with_payload().is_err());
                    }
                    // Not on the document asked: nothing.
                    leaf.occurrences_at(&ctx, ft, term, doc + 1, &mut out)
                        .unwrap();
                    assert!(out.is_empty());
                    doc = leaf.next_doc(doc).unwrap();
                }
            }
        }
    }
}
