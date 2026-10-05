//! Per-segment scorers for [`crate::extended_query`]: each query's
//! `Weight.scorerSupplier`, reading the reader-wide statistics its
//! constructor would have gathered from [`crate::GlobalStats`] (or, with none
//! gathered, from the segment itself -- a single-segment search, where they
//! are the same).
//!
//! Every scorer here scores through [`crate::similarities`]: under the
//! default similarity that is [`crate::similarities::Bm25Similarity`]'s
//! `SimScorer`, the same `BM25Scorer` arithmetic the fast path computes.
//! Bounds are `MaxScoreCache.globalMaxScore`'s (`score(Float.MAX_VALUE, 1)`)
//! rather than merged impacts; a looser bound only means less pruning.

use std::ops::ControlFlow;
use std::sync::Arc;

use lucene_codecs::blocktree::{BlockTreeFields, SeekedTerm};
use lucene_codecs::postings::{Impact, LazyDocsCursor, PostingsFlags};

use super::build::{self, Child, LeafContext};
use super::disi_approx::{DisiApprox, DisiSub};
use super::leaf::{DocList, TermScorer};
use super::{BoxScorer, Mode, Scorer, NO_MORE_DOCS};
use crate::bulk_scorer::TermLeg;
use crate::extended_query::*;
use crate::field_norms::FieldNormsCursor;
use crate::query::{
    BooleanQuery, Clause, ConstantScoreQuery, MultiPhraseQuery, PhraseQuery, TermQuery,
};
use crate::similarities::{
    Bm25Similarity, CollectionStatistics, SimScorer, Similarity, TermStatistics,
};
use crate::{blocktree, sloppy_phrase, CollectionStats, Result};

// ---------------------------------------------------------------------------
// Statistics
// ---------------------------------------------------------------------------

/// `IndexSearcher.getSimilarity().scorer(...)`: the searcher's similarity, or
/// the default BM25 when the caller runs the fast path.
pub(crate) fn sim_scorer(
    ctx: &LeafContext<'_>,
    field: &str,
    boost: f32,
    collection: &CollectionStatistics,
    terms: &[TermStatistics],
) -> Arc<dyn SimScorer> {
    match ctx.similarity {
        Some(s) => s.scorer(field, boost, collection, terms),
        None => Bm25Similarity::default().scorer(field, boost, collection, terms),
    }
}

/// One term's reader-wide statistics with its field's (`docFreq` `0` when the
/// term is absent everywhere); `None` when no segment has the field. The
/// statistics pass records an entry for every term these queries name
/// ([`collect_terms`]); without one, this segment's own counters.
pub(crate) fn term_entry(
    ctx: &LeafContext<'_>,
    field: &str,
    term: &[u8],
) -> Result<Option<CollectionStats>> {
    if let Some(g) = ctx.global.and_then(|g| g.term(field, term)) {
        return Ok(Some(*g));
    }
    let Some(ft) = ctx.fields.field(field) else {
        return Ok(None);
    };
    let (doc_freq, total_term_freq) = match ft.try_seek_exact(term)? {
        Some(s) => (i64::from(s.doc_freq), s.total_term_freq),
        None => (0, 0),
    };
    Ok(Some(CollectionStats {
        doc_freq,
        doc_count: i64::from(ft.doc_count),
        total_term_freq,
        max_doc: i64::from(ctx.max_doc.unwrap_or(0)),
        sum_total_term_freq: ft.sum_total_term_freq,
        sum_doc_freq: ft.sum_doc_freq,
    }))
}

/// `searcher.collectionStatistics(field)` from an entry: `null` (here
/// `None`) when no document has the field.
pub(crate) fn collection_of(entry: &CollectionStats) -> Option<CollectionStatistics> {
    (entry.doc_count > 0).then(|| entry.collection_statistics())
}

/// The term's postings cursor in this segment, `None` when absent. `mode`
/// picks what `TermsEnum.impacts`/`postings(FREQS)` would decode.
fn cursor<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
    term: &[u8],
    flags: PostingsFlags,
) -> Result<Option<(LazyDocsCursor<'a>, SeekedTerm)>> {
    let Some(doc_in) = ctx.doc_in else {
        return Ok(None);
    };
    let Some(ft) = ctx.fields.field(field) else {
        return Ok(None);
    };
    let seeked = match ctx
        .global
        .and_then(|g| g.term_state(field, term, ctx.fields))
    {
        Some(found) => found,
        None => ft.seek_term_state(term)?,
    };
    let Some(seeked) = seeked else {
        return Ok(None);
    };
    Ok(Some((
        ft.lazy_postings_for(&seeked, doc_in, flags)?,
        seeked,
    )))
}

fn scoring_flags(mode: Mode) -> PostingsFlags {
    if mode == Mode::TopScores {
        PostingsFlags::Freqs
    } else {
        PostingsFlags::FreqsNoImpacts
    }
}

pub(crate) fn norms_cursor<'a>(
    ctx: &LeafContext<'a>,
    field: &str,
) -> Option<FieldNormsCursor<'a, 'a>> {
    ctx.norms.and_then(|m| m.get(field)).map(|n| n.cursor())
}

fn pe(e: lucene_codecs::postings::Error) -> crate::Error {
    blocktree::Error::Postings(e).into()
}

/// The terms and fields a query here scores from, for the statistics pass:
/// `(field, term)` pairs whose reader-wide statistics it must gather.
pub(crate) fn collect_terms(q: &ExtendedQuery, out: &mut Vec<(String, Vec<u8>)>) {
    match q {
        ExtendedQuery::Synonym(s) => {
            for (t, _) in &s.terms {
                out.push((s.field.clone(), t.clone()));
            }
        }
        ExtendedQuery::CombinedField(c) => {
            for (f, _) in &c.fields {
                out.push((f.clone(), c.term.clone()));
            }
        }
        ExtendedQuery::Blended(b) => {
            for (f, t, _) in &b.terms {
                out.push((f.clone(), t.clone()));
            }
        }
        ExtendedQuery::NGramPhrase(n) => {
            for t in &n.rewrite().terms {
                out.push((n.phrase.field.clone(), t.clone()));
            }
        }
        // `SpanWeight.buildSimWeight`: the terms the weight takes its
        // statistics from (and a not query's exclude side, built too).
        ExtendedQuery::Span(q) => crate::spans::all_terms(q, out),
        _ => {}
    }
}

/// The terms of a multi-phrase, for the statistics pass.
pub(crate) fn collect_multi_phrase_terms(q: &MultiPhraseQuery, out: &mut Vec<(String, Vec<u8>)>) {
    for alts in &q.term_arrays {
        for t in alts {
            out.push((q.field.clone(), t.clone()));
        }
    }
}

// ---------------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------------

/// `Weight.scorerSupplier(...).get(...)` for an extended query; `None` when
/// it matches nothing in this segment.
pub(crate) fn build<'a>(
    ctx: &LeafContext<'a>,
    q: &ExtendedQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    match q {
        ExtendedQuery::Synonym(s) => synonym(ctx, s, boost, mode),
        ExtendedQuery::CombinedField(c) => combined_field(ctx, c, boost, mode),
        ExtendedQuery::NGramPhrase(n) => {
            build::build(ctx, &Clause::Phrase(n.rewrite()), boost, mode, top_level)
        }
        ExtendedQuery::MultiTerm(m) => multi_term(ctx, m, boost, mode, top_level),
        ExtendedQuery::Blended(b) => blended(ctx, b, boost, mode, top_level),
        ExtendedQuery::IndriAnd(i) => indri_and(ctx, i, boost, mode),
        ExtendedQuery::LogOddsFusion(l) => log_odds(ctx, l, boost, mode),
        ExtendedQuery::BayesianScore(b) => bayesian(ctx, b, boost, mode, top_level),
        ExtendedQuery::DocAndScore(d) => doc_and_score(ctx, d, boost, mode),
        ExtendedQuery::NumericDocValuesRange(r) => {
            super::ranges::numeric_range(ctx, r, boost, mode)
        }
        ExtendedQuery::IndexSortRange(r) => {
            super::ranges::index_sort_range(ctx, r, boost, mode, top_level)
        }
        ExtendedQuery::PointRange(r) => super::ranges::point_range(ctx, r, boost, mode),
        ExtendedQuery::PointInSet(r) => super::ranges::point_in_set(ctx, r, boost, mode),
        ExtendedQuery::IndexOrDocValues(q) => {
            index_or_doc_values(ctx, q, boost, mode, top_level, None)
        }
        ExtendedQuery::Document(d) => super::ranges::document(ctx, d, boost, mode),
        ExtendedQuery::ToParentBlockJoin(q) => super::join::to_parent(ctx, q, boost, mode),
        ExtendedQuery::ToChildBlockJoin(q) => super::join::to_child(ctx, q, boost, mode),
        ExtendedQuery::ParentChildrenBlockJoin(q) => {
            super::join::parent_children(ctx, q, boost, mode)
        }
        ExtendedQuery::ParentsChildrenBlockJoin(q) => {
            super::join::parents_children(ctx, q, boost, mode)
        }
        ExtendedQuery::TermsIncludingScore(q) => {
            super::query_join::terms_including_score(ctx, q, boost, mode, top_level)
        }
        ExtendedQuery::GlobalOrdinals(q) => super::query_join::global_ordinals(ctx, q, boost, mode),
        ExtendedQuery::GlobalOrdinalsWithScore(q) => {
            super::query_join::global_ordinals_with_score(ctx, q, boost, mode)
        }
        ExtendedQuery::PointInSetIncludingScore(q) => {
            super::query_join::point_in_set_including_score(ctx, q)
        }
        ExtendedQuery::Function(q) => super::function::function_query(ctx, q, boost),
        ExtendedQuery::FunctionRange(q) => super::function::function_range(ctx, q),
        ExtendedQuery::FunctionMatch(q) => super::function::function_match(ctx, q, boost, mode),
        ExtendedQuery::Interval(q) => super::intervals::interval(ctx, q, boost, mode),
        ExtendedQuery::Span(q) => super::spans::span_node(ctx, q, boost, mode),
        ExtendedQuery::FunctionScore(q) => {
            super::function::function_score(ctx, q, boost, mode, top_level)
        }
        // `CommonTermsQuery` has no weight of its own: `rewrite(searcher)`
        // turns it into the boolean of its rare and frequent terms first.
        ExtendedQuery::CommonTerms(_) => Err(crate::Error::IllegalState(
            "CommonTermsQuery must be rewritten against the searcher \
             (rescorer::rewrite_rescore_clauses) before it is searched"
                .into(),
        )),
        // So has `MoreLikeThisQuery`: its rewrite finds the text's
        // interesting terms.
        ExtendedQuery::MoreLikeThis(_) => Err(crate::Error::IllegalState(
            "MoreLikeThisQuery must be rewritten against the searcher \
             (rescorer::rewrite_rescore_clauses) before it is searched"
                .into(),
        )),
        // `RescoreTopNQuery` has no weight of its own: `rewrite(searcher)`
        // turns it into a `DocAndScoreQuery` first.
        ExtendedQuery::RescoreTopN(_) => Err(crate::Error::IllegalState(
            "RescoreTopNQuery must be rewritten against the searcher \
             (rescorer::rewrite_rescore_clauses) before it is searched"
                .into(),
        )),
    }
}

// ---------------------------------------------------------------------------
// IndexOrDocValuesQuery
// ---------------------------------------------------------------------------

/// Whether a side of an `IndexOrDocValuesQuery` has a `ScorerSupplier` in this
/// segment -- `Weight.scorerSupplier` returning `null` for a field the segment
/// does not index that way -- and, for the index side, its `cost()`.
pub(crate) enum Supplier {
    /// No scorer supplier: the whole query matches nothing here.
    Absent,
    /// A supplier; its cost when it can be had without building the scorer
    /// (`PointRangeQuery`'s `estimateDocCount`), `None` otherwise.
    Present(Option<i64>),
}

/// The index side's supplier and `cost()`.
pub(crate) fn index_side(ctx: &LeafContext<'_>, clause: &Clause) -> Result<Supplier> {
    let Clause::Extended(e) = clause else {
        return Ok(Supplier::Present(None));
    };
    match e.as_ref() {
        ExtendedQuery::PointRange(q) => super::ranges::point_range_cost(ctx, q),
        ExtendedQuery::PointInSet(q) => Ok(match ctx.points {
            Some(p)
                if p.field_number(&q.field)
                    .and_then(|n| p.reader.field(n))
                    .is_some() =>
            {
                Supplier::Present(None)
            }
            _ => Supplier::Absent,
        }),
        _ => Ok(Supplier::Present(None)),
    }
}

/// Whether the doc-values side has a supplier: a doc-values query over a
/// field this segment has no doc values for has none.
fn dv_side_present(ctx: &LeafContext<'_>, clause: &Clause) -> Result<bool> {
    let field = match clause {
        Clause::Extended(e) => match e.as_ref() {
            ExtendedQuery::NumericDocValuesRange(q) => &q.field,
            _ => return Ok(true),
        },
        _ => return Ok(true),
    };
    let Some(reader) = ctx.reader else {
        return Err(crate::Error::MissingSegmentReader(field.to_string()));
    };
    Ok(reader
        .field_infos()
        .field_by_name(field)
        .is_some_and(|fi| fi.doc_values_type != lucene_codecs::field_infos::DocValuesType::None))
}

/// `IndexOrDocValuesQuery`'s `ScorerSupplier`: `get(leadCost)` with the lead
/// cost of the boolean it is a clause of, or -- `lead_cost` `None`, the query
/// run alone -- `bulkScorer()`, which always takes the index side.
pub(crate) fn index_or_doc_values<'a>(
    ctx: &LeafContext<'a>,
    q: &IndexOrDocValuesQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
    lead_cost: Option<i64>,
) -> Result<Option<BoxScorer<'a>>> {
    let Supplier::Present(cost) = index_side(ctx, &q.index_query)? else {
        return Ok(None);
    };
    if !dv_side_present(ctx, &q.dv_query)? {
        return Ok(None);
    }
    let plan = match lead_cost {
        None => crate::doc_value_query::IndexOrDocValuesPlan::Index,
        Some(lead) => crate::doc_value_query::plan_index_or_doc_values(cost, lead),
    };
    let side = match plan {
        crate::doc_value_query::IndexOrDocValuesPlan::Index => &q.index_query,
        crate::doc_value_query::IndexOrDocValuesPlan::DocValues => &q.dv_query,
    };
    #[cfg(test)]
    IODV_PLANS.with(|p| p.borrow_mut().push(plan));
    build::build(ctx, side, boost, mode, top_level)
}

#[cfg(test)]
thread_local! {
    /// Every side [`index_or_doc_values`] chose on this thread, for tests.
    pub(crate) static IODV_PLANS: std::cell::RefCell<Vec<crate::doc_value_query::IndexOrDocValuesPlan>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

// ---------------------------------------------------------------------------
// Positions: explicit phrase offsets
// ---------------------------------------------------------------------------

/// A phrase's frequency in one document when its terms sit at explicit
/// `offsets` (`PhraseQuery.Builder.add(term, position)`), `positions[i]`
/// being slot `i`'s raw positions. An exact phrase, or a sloppy one without
/// repeated terms, only ever compares `position - offset` across slots, so
/// the positions are rebased onto the slot index and handed to the matchers
/// written for implicit offsets. A sloppy phrase with repeats also compares
/// the raw positions (`tpPos`), so it runs the matcher with the offsets.
pub(crate) fn phrase_freq_at(
    positions: &[Vec<i32>],
    offsets: &[i32],
    repeats: &sloppy_phrase::PhraseRepeats,
    slop: u32,
    scratch: &mut sloppy_phrase::SloppyScratch,
    shifted: &mut Vec<Vec<i32>>,
) -> f32 {
    // The slots' position lists as slices: on the stack for an ordinary
    // phrase, so a document's match allocates nothing.
    const STACK: usize = 8;
    let mut on_stack: [&[i32]; STACK] = [&[]; STACK];
    let on_heap: Vec<&[i32]>;
    if slop > 0 && repeats.has_rpts {
        let slices: &[&[i32]] = if positions.len() <= STACK {
            for (slot, p) in on_stack.iter_mut().zip(positions) {
                *slot = p.as_slice();
            }
            &on_stack[..positions.len()]
        } else {
            on_heap = positions.iter().map(Vec::as_slice).collect();
            &on_heap
        };
        return sloppy_phrase::sloppy_phrase_freq_offsets_in(
            scratch, slices, repeats, slop, offsets,
        );
    }
    // Each slot's positions moved by its offset; a slot already at its own
    // index (the usual case) is read in place.
    shifted.resize_with(positions.len(), Vec::new);
    for (slot, (src, dst)) in positions.iter().zip(shifted.iter_mut()).enumerate() {
        dst.clear();
        let delta = slot as i64 - i64::from(offsets[slot]);
        if delta == 0 {
            continue;
        }
        dst.extend(src.iter().map(|&p| {
            i32::try_from(i64::from(p) + delta).unwrap_or(if delta < 0 {
                i32::MIN
            } else {
                i32::MAX
            })
        }));
    }
    let in_place = |slot: usize| slot as i64 == i64::from(offsets[slot]);
    let slices: &[&[i32]] = if positions.len() <= STACK {
        for (slot, s) in on_stack.iter_mut().enumerate().take(positions.len()) {
            *s = if in_place(slot) {
                positions[slot].as_slice()
            } else {
                shifted[slot].as_slice()
            };
        }
        &on_stack[..positions.len()]
    } else {
        on_heap = (0..positions.len())
            .map(|slot| {
                if in_place(slot) {
                    positions[slot].as_slice()
                } else {
                    shifted[slot].as_slice()
                }
            })
            .collect();
        &on_heap
    };
    if slop == 0 {
        crate::phrase_freq_exact(slices) as f32
    } else {
        sloppy_phrase::sloppy_phrase_freq_in(scratch, slices, repeats, slop)
    }
}

/// `PhraseWeight.scorer` for a phrase with explicit positions: the streaming
/// [`super::phrase::PhraseScorer`] with the offsets, or -- when a term is a
/// pulsed singleton with no `.doc` stream to walk -- the matches resolved up
/// front. Scores through the similarity (`PhraseWeight.getStats`).
pub(crate) fn positional_phrase<'a>(
    ctx: &LeafContext<'a>,
    p: &PhraseQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    if p.terms.len() < 2 {
        // `PhraseQuery.rewrite`: one term is a `TermQuery`.
        let Some(only) = p.terms.first() else {
            return Ok(None);
        };
        let t = Clause::Term(TermQuery::new(p.field.clone(), only.clone()));
        return build::build(ctx, &t, boost, mode, false);
    }
    // `PhraseQuery.rewrite`: positions rebased to start at 0.
    let raw = p.positions();
    let first = raw[0];
    let offsets: Vec<i32> = raw.iter().map(|&x| x - first).collect();
    let slots: Vec<Vec<Vec<u8>>> = p.terms.iter().map(|t| vec![t.clone()]).collect();
    let Some(field_terms) = ctx.fields.field(&p.field) else {
        return Ok(None);
    };
    let Some(pos_in) = ctx.pos_in else {
        return Err(crate::Error::MissingPosInput);
    };
    let mut stats = Vec::with_capacity(p.terms.len());
    let mut collection = None;
    let mut found = Vec::with_capacity(p.terms.len());
    for term in &p.terms {
        let Some(s) = field_terms.try_seek_exact(term)? else {
            return Ok(None);
        };
        found.push(s);
        let Some(entry) = term_entry(ctx, &p.field, term)? else {
            return Ok(None);
        };
        collection.get_or_insert(entry.collection_statistics());
        stats.push(entry.term_statistics());
    }
    let Some(collection) = collection else {
        return Ok(None);
    };
    let scorer = sim_scorer(ctx, &p.field, boost, &collection, &stats);
    let repeats = sloppy_phrase::PhraseRepeats::for_phrase(&p.terms);
    if let Some(doc_in) = ctx.doc_in.filter(|_| found.iter().all(|s| s.doc_freq > 1)) {
        let mut match_cost = 0.0f32;
        let mut terms = Vec::with_capacity(p.terms.len());
        for (slot, (term, s)) in p.terms.iter().zip(&found).enumerate() {
            match_cost +=
                super::phrase::term_positions_cost(i64::from(s.doc_freq), s.total_term_freq);
            let Some(cursor) = field_terms.lazy_positions(term, doc_in, pos_in)? else {
                return Ok(None);
            };
            terms.push(super::phrase::PhraseTerm {
                cursor,
                slot,
                cost: i64::from(s.doc_freq),
            });
        }
        return Ok(Some(Box::new(
            super::phrase::PhraseScorer::new(
                terms,
                0.0,
                p.slop,
                repeats,
                norms_cursor(ctx, &p.field),
                match_cost,
                mode == Mode::TopScores,
                mode.needs_scores(),
            )
            .with_sim_scorer(scorer)
            .with_offsets(offsets),
        )));
    }
    eager_phrase(
        ctx, pos_in, &p.field, &slots, &offsets, &repeats, p.slop, &scorer, mode,
    )
}

/// A phrase's matches resolved up front, each slot a set of alternative terms
/// (one term for a `PhraseQuery`), scored `simScorer.score(freq, norm)`.
#[allow(clippy::too_many_arguments)]
fn eager_phrase<'a>(
    ctx: &LeafContext<'a>,
    pos_in: &lucene_codecs::postings::PosInput<'a>,
    field: &str,
    slots: &[Vec<Vec<u8>>],
    offsets: &[i32],
    repeats: &sloppy_phrase::PhraseRepeats,
    slop: u32,
    scorer: &Arc<dyn SimScorer>,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    // `UnionPostingsEnum` per position over its alternatives' lazy
    // positions, the phrase's conjunction over those as its approximation:
    // only documents every position has are visited, and only their
    // positions decoded.
    let mut sources: Vec<Vec<super::span::LeafPositions<'a>>> = Vec::with_capacity(slots.len());
    for alts in slots {
        let mut slot = Vec::with_capacity(alts.len());
        for t in alts {
            slot.push(super::span::LeafPositions::open(ctx, pos_in, field, t)?);
        }
        sources.push(slot);
    }
    let mut norms = norms_cursor(ctx, field);
    let mut scratch = sloppy_phrase::SloppyScratch::default();
    let mut shifted = Vec::new();
    let mut merged = Vec::new();
    let mut positions: Vec<Vec<i32>> = vec![Vec::new(); slots.len()];
    let (mut docs, mut scores) = (Vec::new(), Vec::new());
    let mut doc = phrase_approximation(&mut sources, 0)?;
    while doc != NO_MORE_DOCS {
        if ctx.live_docs.is_none_or(|l| l.get_doc(doc)) {
            for (buf, slot) in positions.iter_mut().zip(&mut sources) {
                buf.clear();
                let mut split = 0;
                for (k, alt) in slot.iter_mut().enumerate() {
                    if k == 1 {
                        split = buf.len();
                    }
                    alt.positions_at(doc, buf)?;
                }
                // Sorted and **not** deduplicated, as `UnionPostingsEnum`'s
                // `PositionsQueue` yields them (see
                // `crate::multi_phrase_slot_positions`): two alternatives'
                // ascending runs merged, more sorted.
                match slot.len() {
                    0 | 1 => {}
                    2 => merge_runs(buf, split, &mut merged),
                    _ => buf.sort_unstable(),
                }
            }
            let freq = phrase_freq_at(
                &positions,
                offsets,
                repeats,
                slop,
                &mut scratch,
                &mut shifted,
            );
            if freq != 0.0 {
                docs.push(doc);
                if mode.needs_scores() {
                    let norm = match norms.as_mut() {
                        Some(n) => n.norm_long(doc)?.unwrap_or(1),
                        None => 1,
                    };
                    scores.push(scorer.score(freq, norm));
                }
            }
        }
        doc = phrase_approximation(&mut sources, doc.saturating_add(1))?;
    }
    if docs.is_empty() {
        return Ok(None);
    }
    Ok(Some(Box::new(DocList::new(docs, scores))))
}

/// `buf[..split]` and `buf[split..]`, each ascending, merged into one
/// ascending run, duplicates kept (`scratch` is reused across calls).
fn merge_runs(buf: &mut Vec<i32>, split: usize, scratch: &mut Vec<i32>) {
    let (a, b) = buf.split_at(split);
    if a.is_empty() || b.is_empty() || a[a.len() - 1] <= b[0] {
        return;
    }
    scratch.clear();
    let (mut i, mut j) = (0, 0);
    while i < a.len() && j < b.len() {
        if b[j] < a[i] {
            scratch.push(b[j]);
            j += 1;
        } else {
            scratch.push(a[i]);
            i += 1;
        }
    }
    scratch.extend_from_slice(&a[i..]);
    scratch.extend_from_slice(&b[j..]);
    std::mem::swap(buf, scratch);
}

/// The phrase's approximation advanced to `target`: the first document at
/// or after it on which some alternative of every position is
/// (`ConjunctionDISI` over `UnionPostingsEnum`s).
fn phrase_approximation(
    slots: &mut [Vec<super::span::LeafPositions<'_>>],
    target: i32,
) -> Result<i32> {
    let mut target = target;
    loop {
        let mut max = target;
        for slot in slots.iter_mut() {
            let mut min = NO_MORE_DOCS;
            for alt in slot.iter_mut() {
                min = min.min(alt.advance(target)?);
            }
            if min == NO_MORE_DOCS {
                return Ok(NO_MORE_DOCS);
            }
            max = max.max(min);
        }
        if max == target {
            return Ok(target);
        }
        target = max;
    }
}

/// `MultiPhraseQuery`'s weight: one position rewrites to a `BooleanQuery` of
/// `SHOULD` terms; otherwise `PhraseWeight` over `UnionPostingsEnum`s, the
/// similarity's scorer built from every present term of every position
/// (`allTermStats`, duplicates included), positions as given
/// (`MultiPhraseQuery` does not rebase them).
pub(crate) fn multi_phrase<'a>(
    ctx: &LeafContext<'a>,
    q: &MultiPhraseQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    if q.term_arrays.is_empty() {
        return Ok(None);
    }
    if q.term_arrays.len() == 1 {
        let should: Vec<Clause> = q.term_arrays[0]
            .iter()
            .map(|t| Clause::Term(TermQuery::new(q.field.clone(), t.clone())))
            .collect();
        let b = Clause::Boolean(Box::new(BooleanQuery::new().with_should(should)));
        return build::build(ctx, &b, boost, mode, false);
    }
    let mut all = Vec::new();
    let mut collection = None;
    for alts in &q.term_arrays {
        for t in alts {
            if let Some(entry) = term_entry(ctx, &q.field, t)? {
                if entry.doc_freq > 0 {
                    collection.get_or_insert(entry.collection_statistics());
                    all.push(entry.term_statistics());
                }
            }
        }
    }
    let Some(collection) = collection else {
        return Ok(None);
    };
    let Some(_) = ctx.fields.field(&q.field) else {
        return Ok(None);
    };
    let Some(pos_in) = ctx.pos_in else {
        return Err(crate::Error::MissingPosInput);
    };
    let scorer = sim_scorer(ctx, &q.field, boost, &collection, &all);
    let repeats = sloppy_phrase::PhraseRepeats::for_multi_phrase(&q.term_arrays);
    let offsets = q.positions();
    eager_phrase(
        ctx,
        pos_in,
        &q.field,
        &q.term_arrays,
        &offsets,
        &repeats,
        q.slop,
        &scorer,
        mode,
    )
}

// ---------------------------------------------------------------------------
// SynonymQuery
// ---------------------------------------------------------------------------

/// A postings cursor as a `DisiWrapper` member.
struct Member<'a> {
    cursor: LazyDocsCursor<'a>,
    cost: i64,
}

impl DisiSub for Member<'_> {
    fn doc_id(&self) -> i32 {
        self.cursor.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.cursor.next_doc().map_err(pe)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.cursor.advance(target).map_err(pe)
    }
    fn cost(&self) -> i64 {
        self.cost
    }
}

impl Member<'_> {
    fn freq(&self) -> i32 {
        self.cursor.freq().unwrap_or(1)
    }
}

fn synonym<'a>(
    ctx: &LeafContext<'a>,
    q: &SynonymQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    // `SynonymQuery.rewrite`: nothing is an empty boolean, and one unboosted
    // term is that term.
    if q.terms.is_empty() {
        return Ok(None);
    }
    if !mode.needs_scores() || (q.terms.len() == 1 && q.terms[0].1 == 1.0) {
        let should: Vec<Clause> = q
            .terms
            .iter()
            .map(|(t, _)| Clause::Term(TermQuery::new(q.field.clone(), t.clone())))
            .collect();
        let b = Clause::Boolean(Box::new(BooleanQuery::new().with_should(should)));
        return build::build(ctx, &b, boost, mode, false);
    }
    // `SynonymWeight`: the pseudo-term's docFreq is the largest, its
    // totalTermFreq the sum.
    let mut doc_freq = 0i64;
    let mut total_term_freq = 0i64;
    let mut collection = None;
    for (t, _) in &q.terms {
        if let Some(entry) = term_entry(ctx, &q.field, t)? {
            collection.get_or_insert(entry);
            if entry.doc_freq > 0 {
                doc_freq = doc_freq.max(entry.doc_freq);
                total_term_freq = total_term_freq.wrapping_add(entry.total_term_freq);
            }
        }
    }
    let Some(collection) = collection.as_ref().and_then(collection_of) else {
        return Ok(None);
    };
    if doc_freq == 0 {
        return Ok(None);
    }
    let pseudo = TermStatistics {
        doc_freq,
        total_term_freq,
    };
    let scorer = sim_scorer(ctx, &q.field, boost, &collection, &[pseudo]);
    let mut members = Vec::new();
    let mut boosts = Vec::new();
    for (t, b) in &q.terms {
        if let Some((cursor, seeked)) = self::cursor(ctx, &q.field, t, scoring_flags(mode))? {
            members.push(Member {
                cursor,
                cost: i64::from(seeked.stats.doc_freq),
            });
            boosts.push(*b);
        }
    }
    if members.is_empty() {
        return Ok(None);
    }
    Ok(Some(Box::new(
        FreqSumScorer::new(
            members,
            boosts,
            scorer,
            Norms::One(norms_cursor(ctx, &q.field)),
            false,
        )
        .with_impacts(mode == Mode::TopScores),
    )))
}

/// `SynonymQuery.mergeImpacts`' merge of several terms' competitive
/// impacts, each ascending by norm (as `Long.compareUnsigned` orders them)
/// with rising frequencies: walking the norms in order, the running sum of
/// every list's frequency so far -- which also covers the norms a list leaves
/// implicit -- kept wherever it rises. `None` for no list.
fn merge_impacts(lists: &[Vec<Impact>]) -> Option<Vec<Impact>> {
    match lists {
        [] => None,
        [only] => Some(only.clone()),
        _ => {
            // Per list: next index, the frequency already counted.
            let mut at = vec![0usize; lists.len()];
            let mut previous = vec![0i64; lists.len()];
            let mut sum_tf = 0i64;
            let mut merged: Vec<Impact> = Vec::new();
            loop {
                // The least norm among the lists' current impacts.
                let norm = lists
                    .iter()
                    .zip(&at)
                    .filter_map(|(l, &i)| l.get(i).map(|imp| imp.norm as u64))
                    .min();
                let Some(norm) = norm else {
                    return Some(merged);
                };
                for ((l, i), prev) in lists.iter().zip(at.iter_mut()).zip(previous.iter_mut()) {
                    if let Some(imp) = l.get(*i) {
                        if imp.norm as u64 == norm {
                            sum_tf = sum_tf.saturating_add(i64::from(imp.freq) - *prev);
                            *prev = i64::from(imp.freq);
                            *i = i.saturating_add(1);
                        }
                    }
                }
                let freq = sum_tf.min(i64::from(i32::MAX)) as i32;
                if merged.last().is_none_or(|last| freq > last.freq) {
                    merged.push(Impact {
                        freq,
                        norm: norm as i64,
                    });
                }
            }
        }
    }
}

/// How a [`FreqSumScorer`] reads a document's norm.
enum Norms<'a> {
    /// The field's own norm, `1` without one.
    One(Option<FieldNormsCursor<'a, 'a>>),
    /// `MultiNormsLeafSimScorer`: the weighted sum of every field's decoded
    /// length, re-encoded -- with [`DenseMemo`] when the fields allow it.
    Multi(
        Vec<(FieldNormsCursor<'a, 'a>, f32)>,
        Option<Box<DenseMemo<'a>>>,
    ),
}

/// [`Norms::Multi`] over one or two fields whose norms are dense, one byte
/// per document: the combined norm is then a function of the fields' norm
/// bytes alone, so it is computed once per distinct pair of bytes and
/// remembered, instead of decoding, weighting, rounding and re-encoding per
/// document (`MultiNormsLeafSimScorer`'s per-document work). The value
/// remembered is exactly what [`Norms::norm`]'s arithmetic gives for that
/// pair, under `advanceExact`'s rule (`batch == false`).
struct DenseMemo<'a> {
    bytes: [&'a [u8]; 2],
    weights: [f32; 2],
    two: bool,
    /// Per `a | b << 8`: the combined norm byte, or [`DenseMemo::UNSET`].
    memo: Box<[u16]>,
}

impl<'a> DenseMemo<'a> {
    const UNSET: u16 = u16::MAX;

    fn new(fields: &[(FieldNormsCursor<'a, 'a>, f32)]) -> Option<Box<Self>> {
        let (first, second) = match fields {
            [a] => (a, None),
            [a, b] => (a, Some(b)),
            _ => return None,
        };
        let a = first.0.dense_bytes()?;
        let b = match second {
            Some(f) => Some(f.0.dense_bytes()?),
            None => None,
        };
        Some(Box::new(DenseMemo {
            bytes: [a, b.unwrap_or(&[])],
            weights: [first.1, second.map_or(0.0, |f| f.1)],
            two: b.is_some(),
            memo: vec![Self::UNSET; if b.is_some() { 1 << 16 } else { 1 << 8 }].into_boxed_slice(),
        }))
    }

    /// The combined norm of `doc`, or `None` when some field's byte array
    /// does not cover it (the general path answers).
    #[inline]
    fn norm(&mut self, doc: i32) -> Option<i64> {
        let a = *self.bytes[0].get(doc as usize)?;
        let key = if self.two {
            let b = *self.bytes[1].get(doc as usize)?;
            usize::from(a) | usize::from(b) << 8
        } else {
            usize::from(a)
        };
        // `key < 1 << 16` (or `1 << 8` for one field): the memo's length.
        let slot = &mut self.memo[key];
        if *slot == Self::UNSET {
            let table = length_table();
            let mut acc = 0.0f32;
            acc += self.weights[0] * table[usize::from(a)];
            if self.two {
                acc += self.weights[1] * table[key >> 8];
            }
            *slot = u16::from(encode_multi_norm(acc, false) as u8);
        }
        Some(i64::from(*slot as u8 as i8))
    }
}

/// `SmallFloat.byte4ToInt` for every norm byte (`LENGTH_TABLE`).
fn length_table() -> &'static [f32; 256] {
    static TABLE: std::sync::OnceLock<[f32; 256]> = std::sync::OnceLock::new();
    TABLE.get_or_init(|| {
        std::array::from_fn(|i| lucene_util::small_float::byte4_to_int(i as u8) as f32)
    })
}

impl Norms<'_> {
    /// The norm `score` reads for `doc`. `batch` is
    /// `MultiFieldNormValues.longValues`' rule (a document no field has a
    /// norm for reads `1`), `false` its `advanceExact` rule (`0`).
    fn norm(&mut self, doc: i32, batch: bool) -> Result<i64> {
        match self {
            Norms::One(n) => Ok(match n {
                Some(n) => n.norm_long(doc)?.unwrap_or(1),
                None => 1,
            }),
            Norms::Multi(fields, memo) => {
                if !batch {
                    if let Some(norm) = memo.as_mut().and_then(|m| m.norm(doc)) {
                        return Ok(norm);
                    }
                }
                if fields.is_empty() {
                    return Ok(1);
                }
                let table = length_table();
                let mut acc = 0.0f32;
                for (cursor, weight) in fields.iter_mut() {
                    if let Some(norm) = cursor.norm_long(doc)? {
                        acc += *weight * table[usize::from(norm as u8)];
                    }
                }
                Ok(encode_multi_norm(acc, batch))
            }
        }
    }
}

/// `MultiFieldNormValues`' value for a document whose weighted, decoded
/// lengths sum to `acc`: `SmallFloat.intToByte4(Math.round(acc))`, read back
/// as the sign-extended byte `longValue()` returns -- or `1` under `batch`
/// (`longValues`' rule) when nothing was summed.
fn encode_multi_norm(acc: f32, batch: bool) -> i64 {
    if batch && acc == 0.0 {
        return 1;
    }
    let rounded = acc.round() as i64;
    let byte = lucene_util::small_float::int_to_byte4(
        u32::try_from(rounded.clamp(0, i64::from(i32::MAX))).unwrap_or(0),
    );
    i64::from(byte as i8)
}

/// `SynonymScorer` and `CombinedFieldScorer`: a disjunction of postings whose
/// score is `simScorer.score(sum(weight * freq), norm)`, the sum folded in
/// `topList()` order.
struct FreqSumScorer<'a> {
    approx: DisiApprox<Member<'a>>,
    /// Per member: the synonym's boost or the field's weight.
    weights: Vec<f32>,
    scorer: Arc<dyn SimScorer>,
    norms: Norms<'a>,
    /// `CombinedFieldScorer.freq`'s overflow guard.
    combined: bool,
    max: f32,
    list: Vec<usize>,
    /// `SynonymWeight`'s `ImpactsDISI` (`TOP_SCORES` only): the members'
    /// impacts bound each block, and blocks that cannot reach the minimum
    /// competitive score are skipped.
    impacts: bool,
    min_competitive: f32,
    /// The last document the current shallow bound covers.
    up_to: i32,
}

impl<'a> FreqSumScorer<'a> {
    fn new(
        members: Vec<Member<'a>>,
        weights: Vec<f32>,
        scorer: Arc<dyn SimScorer>,
        norms: Norms<'a>,
        combined: bool,
    ) -> Self {
        let max = if combined {
            // `CombinedFieldScorer`: `score(Float.POSITIVE_INFINITY, 1L)`.
            let m = scorer.score(f32::INFINITY, 1);
            if m.is_nan() {
                f32::INFINITY
            } else {
                m
            }
        } else {
            crate::bulk_scorer::sim_global_max(scorer.as_ref(), f32::MAX)
        };
        Self {
            approx: DisiApprox::new(members, i64::MAX),
            weights,
            scorer,
            norms,
            combined,
            max,
            list: Vec::new(),
            impacts: false,
            min_competitive: 0.0,
            up_to: -1,
        }
    }

    /// Prune on the members' impacts, as `SynonymWeight` does in
    /// `TOP_SCORES`.
    fn with_impacts(mut self, on: bool) -> Self {
        self.impacts = on;
        self
    }

    /// `Impacts.getDocIdUpTo(0)` of `SynonymQuery.mergeImpacts`' lead: the
    /// smallest block boundary at or after `target` among the members.
    fn shallow(&mut self, target: i32) -> Result<i32> {
        let mut up_to = NO_MORE_DOCS;
        for m in &mut self.approx.subs {
            up_to = up_to.min(m.cursor.advance_shallow(target).map_err(pe)?);
        }
        Ok(up_to)
    }

    /// `MaxScoreCache.getMaxScore` over `SynonymQuery.mergeImpacts`' merged
    /// impacts for the window ending at `up_to`: the largest score any
    /// `(freq, norm)` pair of the merge allows, or `None` when some member
    /// that can have a document there has no impacts reaching `up_to`
    /// (`mergeImpacts`' "impacts that trigger the maximum score").
    fn impact_bound(&mut self, up_to: i32) -> Option<f32> {
        let mut lists: Vec<Vec<Impact>> = Vec::new();
        for (m, &w) in self.approx.subs.iter_mut().zip(&self.weights) {
            if m.cursor.doc_id() > up_to {
                continue;
            }
            // `getLevel(impacts[i], docIdUpTo)`.
            let l0 = m.cursor.level0_last_doc_id();
            let l1 = m.cursor.level1_last_doc_id();
            let impacts: &[Impact] = if l0 != NO_MORE_DOCS && l0 >= up_to {
                m.cursor.level0_impacts()
            } else if l1 != NO_MORE_DOCS && l1 >= up_to {
                m.cursor.level1_impacts()
            } else {
                return None;
            };
            if impacts.is_empty() {
                return None;
            }
            lists.push(if w != 1.0 {
                impacts
                    .iter()
                    .map(|i| Impact {
                        freq: (i.freq as f32 * w).ceil() as i32,
                        norm: i.norm,
                    })
                    .collect()
            } else {
                impacts.to_vec()
            });
        }
        let merged = merge_impacts(&lists)?;
        let mut max = 0.0f32;
        for i in &merged {
            let s = self.scorer.score(i.freq as f32, i.norm);
            if s.is_nan() {
                return None;
            }
            max = max.max(s);
        }
        Some(max)
    }

    /// `ImpactsDISI.advanceTarget`: from `target`, past every block whose
    /// bound is below the minimum competitive score.
    fn competitive_target(&mut self, mut target: i32) -> Result<i32> {
        if !self.impacts || self.min_competitive <= 0.0 || target <= self.up_to {
            return Ok(target);
        }
        loop {
            self.up_to = self.shallow(target)?;
            if self.up_to == NO_MORE_DOCS {
                return Ok(target);
            }
            match self.impact_bound(self.up_to) {
                Some(bound) if bound < self.min_competitive => {
                    target = self.up_to.saturating_add(1);
                }
                _ => return Ok(target),
            }
        }
    }

    fn freq(&mut self) -> f32 {
        self.approx.top_list(&mut self.list);
        let mut freq = 0.0f32;
        for (k, &w) in self.list.iter().enumerate() {
            let f = self.approx.subs[w].freq();
            if self.combined {
                // `postingsEnum.freq() * weight`, then the overflow guard.
                let term = f as f32 * self.weights[w];
                freq = if k == 0 { term } else { freq + term };
                if freq < 0.0 {
                    return i32::MAX as f32;
                }
            } else {
                // `DisiWrapperFreq.freq()`: `boost * pe.freq()`.
                let term = self.weights[w] * f as f32;
                freq = if k == 0 { term } else { freq + term };
            }
        }
        freq
    }
}

impl Scorer for FreqSumScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.approx.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.impacts && self.min_competitive > 0.0 {
            let target = self.competitive_target(self.doc_id().saturating_add(1))?;
            return self.approx.advance(target);
        }
        self.approx.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        let target = self.competitive_target(target)?;
        self.approx.advance(target)
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        if !self.impacts {
            return Ok(NO_MORE_DOCS);
        }
        self.shallow(target)
    }
    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        self.min_competitive = min;
        Ok(())
    }
    fn cost(&self) -> i64 {
        self.approx.cost()
    }
    fn score(&mut self) -> Result<f32> {
        let doc = self.doc_id();
        let freq = self.freq();
        let norm = self.norms.norm(doc, false)?;
        Ok(self.scorer.score(freq, norm))
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        if self.impacts && up_to != NO_MORE_DOCS {
            if let Some(bound) = self.impact_bound(up_to) {
                return Ok(bound.min(self.max));
            }
        }
        Ok(self.max)
    }
    /// `CombinedFieldScorer.nextDocsAndScores`: the batch reads norms with
    /// `MultiFieldNormValues.longValues`' rule.
    fn next_docs_and_scores(
        &mut self,
        up_to: i32,
        live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
        out: &mut crate::bulk_scorer::DocScores,
    ) -> Result<()> {
        out.docs.clear();
        out.scores.clear();
        let mut doc = self.doc_id();
        while doc < up_to && out.docs.len() < super::NEXT_DOCS_BATCH {
            if live_docs.is_none_or(|l| l.get_doc(doc)) {
                let freq = self.freq();
                let norm = self.norms.norm(doc, self.combined)?;
                out.docs.push(doc);
                out.scores.push(self.scorer.score(freq, norm));
            }
            doc = self.next_doc()?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// CombinedFieldQuery
// ---------------------------------------------------------------------------

fn combined_field<'a>(
    ctx: &LeafContext<'a>,
    q: &CombinedFieldQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    if q.fields.is_empty() {
        return Ok(None);
    }
    if !mode.needs_scores() {
        let should: Vec<Clause> = q
            .fields
            .iter()
            .map(|(f, _)| Clause::Term(TermQuery::new(f.clone(), q.term.clone())))
            .collect();
        let b = Clause::Boolean(Box::new(BooleanQuery::new().with_should(should)));
        return build::build(ctx, &b, boost, mode, false);
    }
    Ok(combined_parts(ctx, q, boost)?.map(|p| -> BoxScorer<'a> {
        Box::new(FreqSumScorer::new(
            p.members, p.weights, p.scorer, p.norms, true,
        ))
    }))
}

/// What a scored `CombinedFieldQuery` is built from in one segment.
struct CombinedParts<'a> {
    members: Vec<Member<'a>>,
    weights: Vec<f32>,
    scorer: Arc<dyn SimScorer>,
    norms: Norms<'a>,
}

/// `CombinedFieldWeight.scorer`'s pieces; `None` when nothing matches here.
fn combined_parts<'a>(
    ctx: &LeafContext<'a>,
    q: &CombinedFieldQuery,
    boost: f32,
) -> Result<Option<CombinedParts<'a>>> {
    // `CombinedFieldWeight`: docFreq the largest, totalTermFreq the
    // weighted sum (`long += double`, a truncating cast each step).
    let mut doc_freq = 0i64;
    let mut total_term_freq = 0i64;
    // `mergeCollectionStatistics`.
    let (mut max_doc, mut doc_count, mut sum_ttf, mut sum_df) = (0i64, 0i64, 0i64, 0i64);
    for (field, weight) in &q.fields {
        let Some(entry) = term_entry(ctx, field, &q.term)? else {
            continue;
        };
        if entry.doc_freq > 0 {
            doc_freq = doc_freq.max(entry.doc_freq);
            total_term_freq =
                (total_term_freq as f64 + f64::from(*weight) * entry.total_term_freq as f64) as i64;
        }
        if entry.doc_count > 0 {
            max_doc = max_doc.max(entry.max_doc);
            doc_count = doc_count.max(entry.doc_count);
            sum_df = sum_df.max(entry.sum_doc_freq);
            sum_ttf =
                (sum_ttf as f64 + f64::from(*weight) * entry.sum_total_term_freq as f64) as i64;
        }
    }
    if doc_freq == 0 {
        return Ok(None);
    }
    let collection = CollectionStatistics {
        max_doc,
        doc_count,
        sum_total_term_freq: sum_ttf,
        sum_doc_freq: sum_df,
    };
    let pseudo = TermStatistics {
        doc_freq,
        total_term_freq: total_term_freq.max(1),
    };
    let scorer = sim_scorer(ctx, "pseudo_field", boost, &collection, &[pseudo]);
    let mut members = Vec::new();
    let mut weights = Vec::new();
    for (field, weight) in &q.fields {
        if let Some((cursor, seeked)) = cursor(ctx, field, &q.term, PostingsFlags::FreqsNoImpacts)?
        {
            members.push(Member {
                cursor,
                cost: i64::from(seeked.stats.doc_freq),
            });
            weights.push(*weight);
        }
    }
    if members.is_empty() {
        return Ok(None);
    }
    // `MultiNormsLeafSimScorer`: every field with norms, in field order.
    let norms: Vec<(FieldNormsCursor<'a, 'a>, f32)> = q
        .fields
        .iter()
        .filter_map(|(f, w)| norms_cursor(ctx, f).map(|c| (c, *w)))
        .collect();
    let norms = if norms.is_empty() {
        Norms::One(None)
    } else {
        let memo = DenseMemo::new(&norms);
        Norms::Multi(norms, memo)
    };
    Ok(Some(CombinedParts {
        members,
        weights,
        scorer,
        norms,
    }))
}

/// A scored top-level `CombinedFieldQuery` whose term is in at most two of
/// its fields, as its own bulk scorer ([`CombinedWindow`]): `None` for any
/// other shape (the scorer tree runs it, [`FreqSumScorer`]), `Some(None)`
/// when nothing matches here. With three or more members the frequencies
/// are summed in `topList()` order, which the window does not keep.
pub(crate) fn combined_field_bulk<'a>(
    ctx: &LeafContext<'a>,
    q: &CombinedFieldQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<Option<Box<CombinedWindow<'a>>>>> {
    if q.fields.is_empty() || !mode.needs_scores() {
        return Ok(None);
    }
    Ok(match combined_parts(ctx, q, boost)? {
        None => Some(None),
        Some(parts) if parts.members.len() <= 2 => Some(Some(Box::new(CombinedWindow::new(parts)))),
        Some(_) => None,
    })
}

/// The documents [`CombinedWindow`] gathers at once.
const COMBINED_WINDOW: usize = 4096;

/// `DefaultBulkScorer` over `CombinedFieldScorer` with one or two members,
/// computed a window of documents at a time instead of a document at a time.
///
/// Lucene's `CombinedFieldScorer` has no impacts and ignores the minimum
/// competitive score, so its bulk scorer visits, scores and collects every
/// document either field's postings hold. This does the same, in this
/// order per window of [`COMBINED_WINDOW`] documents: each member's postings
/// a decoded block at a time (`nextPostings`), its `freq * weight` added into
/// the document's slot; then the window's documents in order, with their
/// combined norms ([`Norms::norm`], the `advanceExact` rule `score()` reads);
/// then the BM25 scores of the whole batch (`BM25Scorer.score`'s arithmetic,
/// with no call per document, so it vectorizes); then the collector, every
/// live document in order.
///
/// The same values as the scorer tree, bit for bit: a document's frequency
/// is `t0`, `t1` or `t0 + t1` (`float` addition commutes, so the order
/// `topList()` would sum two in does not matter -- why this is limited to
/// two), the norm and score are the same functions, and every live document
/// is collected in ascending order, as `DefaultBulkScorer` would.
pub(crate) struct CombinedWindow<'a> {
    members: Vec<(Member<'a>, f32)>,
    scorer: Arc<dyn SimScorer>,
    norms: Norms<'a>,
    /// Per window slot: the summed `freq * weight` (zero between windows).
    acc: Box<[f32]>,
    /// Which window slots some member has a document in.
    bits: Box<[u64]>,
    block_docs: Vec<i32>,
    block_freqs: Vec<i32>,
    docs: Vec<i32>,
    freqs: Vec<f32>,
    norm_values: Vec<i64>,
    inverses: Vec<f32>,
    scores: Vec<f32>,
}

impl<'a> CombinedWindow<'a> {
    fn new(parts: CombinedParts<'a>) -> Self {
        Self {
            members: parts.members.into_iter().zip(parts.weights).collect(),
            scorer: parts.scorer,
            norms: parts.norms,
            acc: vec![0.0; COMBINED_WINDOW].into_boxed_slice(),
            bits: vec![0; COMBINED_WINDOW / 64].into_boxed_slice(),
            block_docs: Vec::new(),
            block_freqs: Vec::new(),
            docs: Vec::with_capacity(COMBINED_WINDOW),
            freqs: Vec::with_capacity(COMBINED_WINDOW),
            norm_values: Vec::with_capacity(COMBINED_WINDOW),
            inverses: Vec::with_capacity(COMBINED_WINDOW),
            scores: Vec::with_capacity(COMBINED_WINDOW),
        }
    }

    pub(crate) fn score<C: crate::collector::ScoringCollector + ?Sized>(
        &mut self,
        live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
        collector: &mut C,
        min: i32,
        max: i32,
    ) -> Result<i32> {
        for (m, _) in self.members.iter_mut() {
            if m.doc_id() < min {
                m.advance(min)?;
            }
        }
        loop {
            let base = self
                .members
                .iter()
                .map(|(m, _)| m.doc_id())
                .min()
                .unwrap_or(NO_MORE_DOCS);
            if base >= max {
                return Ok(base);
            }
            // ARITH: saturating; `end - base <= COMBINED_WINDOW`.
            let end = base.saturating_add(COMBINED_WINDOW as i32).min(max);
            self.fill(base, end)?;
            self.gather(base, live_docs)?;
            self.compute_scores();
            for (&doc, &score) in self.docs.iter().zip(&self.scores) {
                collector.collect(doc, score);
            }
        }
    }

    /// Every member's documents in `[base, end)` into the window: the slot's
    /// bit set, `freq * weight` added to its frequency.
    fn fill(&mut self, base: i32, end: i32) -> Result<()> {
        for (m, weight) in self.members.iter_mut() {
            while m.cursor.doc_id() < end {
                m.cursor
                    .next_postings(end, &mut self.block_docs, &mut self.block_freqs)
                    .map_err(pe)?;
                if self.block_docs.is_empty() {
                    break;
                }
                for (&doc, &freq) in self.block_docs.iter().zip(&self.block_freqs) {
                    // ARITH: `base <= doc < end <= base + COMBINED_WINDOW`.
                    let i = (doc - base) as usize;
                    // `postingsEnum.freq() * weight`, summed.
                    self.acc[i] += freq as f32 * *weight;
                    self.bits[i >> 6] |= 1u64 << (i & 63);
                }
            }
        }
        Ok(())
    }

    /// The window's live documents in order, with their frequencies, and
    /// the window cleared for the next.
    fn gather(
        &mut self,
        base: i32,
        live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    ) -> Result<()> {
        self.docs.clear();
        self.freqs.clear();
        for (w, word) in self.bits.iter_mut().enumerate() {
            let mut bits = std::mem::take(word);
            while bits != 0 {
                let i = w * 64 + bits.trailing_zeros() as usize;
                bits &= bits - 1;
                // ARITH: `i < COMBINED_WINDOW`, and the slot is a document.
                let doc = base + i as i32;
                let freq = std::mem::take(&mut self.acc[i]);
                if live_docs.is_none_or(|l| l.get_doc(doc)) {
                    self.docs.push(doc);
                    self.freqs.push(freq);
                }
            }
        }
        self.norm_values.clear();
        for &doc in &self.docs {
            self.norm_values.push(self.norms.norm(doc, false)?);
        }
        Ok(())
    }

    /// `simScorer.score(freq, norm)` for the gathered batch.
    fn compute_scores(&mut self) {
        self.scores.clear();
        match self.scorer.bm25_parts() {
            Some((weight, cache)) => {
                self.inverses.clear();
                self.inverses.extend(
                    self.norm_values
                        .iter()
                        .map(|&n| cache[usize::from(n as u8)]),
                );
                // `BM25Scorer.score`, term for term.
                self.scores.extend(
                    self.freqs
                        .iter()
                        .zip(&self.inverses)
                        .map(|(&freq, &inv)| weight - weight / (1.0 + freq * inv)),
                );
            }
            None => self
                .scorer
                .score_bulk(&self.freqs, &self.norm_values, &mut self.scores),
        }
    }
}

// ---------------------------------------------------------------------------
// BlendedTermQuery
// ---------------------------------------------------------------------------

/// `BlendedTermQuery.rewrite`: every term scores with the largest `docFreq`
/// and the summed `totalTermFreq`, then `BOOLEAN_REWRITE` or
/// `DisjunctionMaxRewrite`.
fn blended<'a>(
    ctx: &LeafContext<'a>,
    q: &BlendedTermQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    let children = blended_children(ctx, q, boost, mode)?;
    match q.rewrite {
        BlendedRewrite::Boolean => build::compose(
            Vec::new(),
            Vec::new(),
            children,
            Vec::new(),
            0,
            mode,
            top_level,
        ),
        BlendedRewrite::DisjunctionMax(tie) => {
            let mut terms: Vec<TermScorer<'a>> = children
                .into_iter()
                .map(|c| TermScorer::new(c.into_leg(), mode == Mode::TopScores))
                .collect();
            Ok(match terms.len() {
                0 => None,
                1 => terms.pop().map(|t| -> BoxScorer<'a> { Box::new(t) }),
                _ if mode.needs_scores() => Some(Box::new(
                    super::term_dismax::TermDisMaxScorer::new(terms, tie),
                )),
                _ => Some(Box::new(super::disjunction::DisjunctionScorer::new(
                    terms
                        .into_iter()
                        .map(|t| -> BoxScorer<'a> { Box::new(t) })
                        .collect(),
                    super::disjunction::Combine::Max(tie),
                    false,
                ))),
            })
        }
    }
}

/// `BlendedTermQuery.rewrite`'s term clauses, each a `TermQuery` over the
/// blended statistics (boosted when its boost is not 1), as the children of
/// the query it rewrites to.
fn blended_children<'a>(
    ctx: &LeafContext<'a>,
    q: &BlendedTermQuery,
    boost: f32,
    mode: Mode,
) -> Result<Vec<Child<'a>>> {
    let mut entries = Vec::with_capacity(q.terms.len());
    let mut df = 0i64;
    let mut ttf = 0i64;
    for (field, term, _) in &q.terms {
        let entry = term_entry(ctx, field, term)?;
        if let Some(e) = &entry {
            df = df.max(e.doc_freq);
            ttf = ttf.wrapping_add(e.total_term_freq);
        }
        entries.push(entry);
    }
    let mut children: Vec<Child<'a>> = Vec::new();
    for ((field, term, term_boost), entry) in q.terms.iter().zip(&entries) {
        let Some(entry) = entry else { continue };
        if df == 0 {
            continue;
        }
        let Some((cursor, seeked)) = cursor(
            ctx,
            field,
            term,
            if mode.needs_scores() {
                scoring_flags(mode)
            } else {
                PostingsFlags::DocsOnly
            },
        )?
        else {
            continue;
        };
        let cost = i64::from(seeked.stats.doc_freq);
        if !mode.needs_scores() {
            children.push(Child::Leg(Box::new(TermLeg::filter(cursor, cost))));
            continue;
        }
        let Some(collection) = collection_of(entry) else {
            continue;
        };
        let stats = TermStatistics {
            doc_freq: df,
            total_term_freq: ttf,
        };
        // `new BoostQuery(termQuery, boost)` when the boost is not 1.
        let b = if *term_boost != 1.0 {
            term_boost * boost
        } else {
            boost
        };
        let scorer = sim_scorer(ctx, field, b, &collection, &[stats]);
        children.push(Child::Leg(Box::new(TermLeg::scoring_sim(
            cursor,
            scorer,
            norms_cursor(ctx, field),
            cost,
        ))));
    }
    Ok(children)
}

/// `FuzzyQuery.rewrite`: `TopTermsBlendedFreqScoringRewrite` makes it a
/// [`BlendedTermQuery`] with `BOOLEAN_REWRITE` over the reader-wide
/// expansion, each term boosted by its `FuzzyTermsEnum` boost
/// (`Math.max(0, boost)`). `None` when the field is in no segment.
fn fuzzy_blended(ctx: &LeafContext<'_>, q: &crate::FuzzyQuery) -> Result<Option<BlendedTermQuery>> {
    let expansion = match ctx.global.and_then(|g| g.fuzzy(q)) {
        Some(e) => e.clone(),
        None => {
            let Some(ft) = ctx.fields.field(&q.field) else {
                return Ok(None);
            };
            crate::fuzzy_expansion_across_leaves(&[ft], q, i64::from(ft.doc_count))?
        }
    };
    Ok(Some(BlendedTermQuery::new(
        expansion
            .terms
            .iter()
            .map(|(t, b)| (q.field.clone(), t.clone(), b.max(0.0))),
        BlendedRewrite::Boolean,
    )?))
}

/// What a top-level clause rewrites to when that is a boolean, so
/// `bulk_clause` can hand it `BooleanWeight.bulkScorer` (Lucene rewrites
/// these queries before it creates a weight, and a pure disjunction of terms
/// is then scored by `MaxScoreBulkScorer`, not a `WANDScorer`).
pub(crate) enum BulkRewrite<'a> {
    /// The rewritten query, to be bulk-scored as any clause is.
    Clause(Clause),
    /// `BlendedTermQuery`'s `BOOLEAN_REWRITE`: its term clauses, already
    /// built over the blended statistics.
    Legs(Vec<Child<'a>>),
}

/// [`BulkRewrite`] for `clause`, when it is a scoring multi-term rewrite, a
/// blended query with the boolean rewrite or a fuzzy query; `None` for every
/// other clause (and when nothing reads a score).
pub(crate) fn bulk_rewrite<'a>(
    ctx: &LeafContext<'a>,
    clause: &Clause,
    boost: f32,
    mode: Mode,
) -> Result<Option<BulkRewrite<'a>>> {
    if !mode.needs_scores() {
        return Ok(None);
    }
    Ok(match clause {
        Clause::Extended(e) => match e.as_ref() {
            ExtendedQuery::MultiTerm(q)
                if matches!(
                    q.rewrite,
                    RewriteMethod::ScoringBoolean
                        | RewriteMethod::TopTermsScoringBoolean(_)
                        | RewriteMethod::TopTermsBoostOnlyBoolean(_)
                        | RewriteMethod::TopTermsBlendedFreqScoring(_)
                ) =>
            {
                rewritten(ctx, q)?.map(BulkRewrite::Clause)
            }
            ExtendedQuery::Blended(b) if b.rewrite == BlendedRewrite::Boolean => {
                Some(BulkRewrite::Legs(blended_children(ctx, b, boost, mode)?))
            }
            _ => None,
        },
        Clause::Fuzzy(f) => Some(BulkRewrite::Legs(fuzzy_children(ctx, f, boost, mode)?)),
        _ => None,
    })
}

/// `FuzzyQuery`'s scorer, under any similarity and score mode: its rewrite
/// ([`fuzzy_blended`]).
pub(crate) fn fuzzy_sim<'a>(
    ctx: &LeafContext<'a>,
    q: &crate::FuzzyQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(blended) = fuzzy_blended(ctx, q)? else {
        return Ok(None);
    };
    self::blended(ctx, &blended, boost, mode, top_level)
}

/// A fuzzy `SHOULD` clause of a boolean whose `minimumNumberShouldMatch` is
/// at most 1, flattened into it: `BooleanQuery.rewrite` inlines a nested
/// pure disjunction ("Flatten nested disjunctions"), and the fuzzy clause's
/// rewrite is one, so its term clauses become the enclosing boolean's own
/// `SHOULD` clauses -- summed with its other clauses in one disjunction, not
/// as a sub-total.
pub(crate) fn fuzzy_children<'a>(
    ctx: &LeafContext<'a>,
    q: &crate::FuzzyQuery,
    boost: f32,
    mode: Mode,
) -> Result<Vec<Child<'a>>> {
    match fuzzy_blended(ctx, q)? {
        Some(blended) => blended_children(ctx, &blended, boost, mode),
        None => Ok(Vec::new()),
    }
}

/// Records the reader-wide statistics of every term a fuzzy clause expanded
/// to, which a similarity other than BM25 reads (`totalTermFreq`, the
/// field's sums) and the BM25 fast path does not.
pub(crate) fn add_fuzzy_term_stats(
    global: &mut crate::GlobalStats,
    segments: &[crate::multi_segment::OpenSegment<'_>],
) -> Result<()> {
    let fuzzy: Vec<(String, Vec<Vec<u8>>)> = global
        .fuzzy_entries()
        .map(|(q, e)| {
            (
                q.field.clone(),
                e.terms.iter().map(|(t, _)| t.clone()).collect(),
            )
        })
        .collect();
    for (field, terms) in fuzzy {
        for term in terms {
            if global.term(&field, &term).is_some() {
                continue;
            }
            if let Some((stats, states)) =
                crate::multi_segment::global_term_stats_states(segments, &field, &term)?
            {
                global.insert_term_states(field.clone(), term, stats, states);
            }
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// MultiTermQuery rewrite methods
// ---------------------------------------------------------------------------

/// The terms `source` matches in one segment, in term order, each with its
/// stats and postings pointers -- at most `limit` of them (the first, which
/// are the smallest), `None` for every one.
///
/// `limit` is `TopTermsRewrite`'s queue under a constant boost: once the
/// queue is full, every later term of the segment sorts after its last and
/// is uncompetitive (`collect` returns before `termState()`), so the walk
/// stops there instead of decoding every remaining term's metadata.
pub(crate) fn expand_terms(
    fields: &BlockTreeFields,
    source: &MultiTermSource,
    limit: Option<usize>,
) -> Result<Vec<(Vec<u8>, SeekedTerm)>> {
    let mut out = Vec::new();
    visit_terms(fields, source, limit, &mut |term, seeked| {
        out.push((term, seeked));
        Ok(ControlFlow::Continue(()))
    })?;
    Ok(out)
}

/// [`expand_terms`], handing each term to `sink` as the walk reaches it
/// rather than collecting them. A sink that returns [`ControlFlow::Break`]
/// ends the walk there, as the constant-score wrappers `return` once a term
/// matches every document of the field.
pub(crate) fn visit_terms(
    fields: &BlockTreeFields,
    source: &MultiTermSource,
    limit: Option<usize>,
    sink: &mut dyn FnMut(Vec<u8>, SeekedTerm) -> Result<ControlFlow<()>>,
) -> Result<()> {
    let limit = limit.unwrap_or(usize::MAX);
    let mut take = |it: &mut dyn Iterator<
        Item = lucene_codecs::blocktree::Result<(Vec<u8>, SeekedTerm)>,
    >|
     -> Result<()> {
        for t in it.take(limit) {
            let (term, seeked) = t?;
            if sink(term, seeked)?.is_break() {
                break;
            }
        }
        Ok(())
    };
    match source {
        MultiTermSource::Prefix(p) => {
            let Some(ft) = fields.field(&p.field) else {
                return Ok(());
            };
            let pattern = lucene_codecs::wildcard::WildcardPattern::prefix(&p.prefix);
            let mut it = ft.intersect_states(&pattern);
            take(&mut it)
        }
        MultiTermSource::Wildcard(w) => {
            let Some(ft) = fields.field(&w.field) else {
                return Ok(());
            };
            let pattern = lucene_codecs::wildcard::WildcardPattern::new(&w.pattern);
            let mut it = ft.intersect_states(&pattern);
            take(&mut it)
        }
        MultiTermSource::Regexp(r) => {
            let Some(ft) = fields.field(&r.field) else {
                return Ok(());
            };
            let pattern = lucene_codecs::regexp::RegexpPattern::new(r.pattern.as_bytes())?;
            let mut it = ft.regexp_intersect_states(&pattern);
            take(&mut it)
        }
        MultiTermSource::TermRange(r) => {
            let Some(ft) = fields.field(&r.field) else {
                return Ok(());
            };
            let mut it = ft.iter();
            let mut on = match &r.lower {
                Some(lo) => !matches!(
                    it.try_seek_ceil(lo)?,
                    lucene_codecs::blocktree::SeekStatus::End
                ),
                None => it.try_next_term()?.is_some(),
            };
            let mut taken = 0usize;
            while on && taken < limit {
                let term = it.term().map(<[u8]>::to_vec).unwrap_or_default();
                let past = match &r.upper {
                    None => false,
                    Some(hi) if r.include_upper => term.as_slice() > hi.as_slice(),
                    Some(hi) => term.as_slice() >= hi.as_slice(),
                };
                if past {
                    break;
                }
                if r.accepts(&term) {
                    if let Some(seeked) = it.try_seeked_term()? {
                        taken = taken.saturating_add(1);
                        if sink(term, seeked)?.is_break() {
                            break;
                        }
                    }
                }
                on = it.try_next_term()?.is_some();
            }
            Ok(())
        }
        // `TermsQuery.getTermsEnum`: `TermsEnum.EMPTY` for no terms, else a
        // `SeekingTermSetTermsEnum` -- `FilteredTermsEnum.next` driven over
        // the field's terms, seeking to each set term in turn.
        MultiTermSource::TermSet(q) => {
            let Some(ft) = fields.field(&q.field) else {
                return Ok(());
            };
            if q.terms.is_empty() {
                return Ok(());
            }
            let mut filter = crate::join::SeekingTermSet::new(std::sync::Arc::clone(&q.terms));
            let mut it = ft.iter();
            let mut do_seek = true;
            let mut taken = 0usize;
            while taken < limit {
                let term = if do_seek {
                    do_seek = false;
                    let Some(t) = filter.next_seek().map(<[u8]>::to_vec) else {
                        break;
                    };
                    if it.try_seek_ceil(&t)? == lucene_codecs::blocktree::SeekStatus::End {
                        break;
                    }
                    it.term().map(<[u8]>::to_vec).unwrap_or_default()
                } else {
                    match it.try_next_term()? {
                        Some(t) => t.to_vec(),
                        None => break,
                    }
                };
                use crate::reader::filtered_terms_enum::AcceptStatus;
                let status = filter.accept_term(&term);
                if matches!(status, AcceptStatus::Yes | AcceptStatus::YesAndSeek) {
                    if let Some(seeked) = it.try_seeked_term()? {
                        taken = taken.saturating_add(1);
                        if sink(term, seeked)?.is_break() {
                            break;
                        }
                    }
                }
                match status {
                    AcceptStatus::YesAndSeek | AcceptStatus::NoAndSeek => do_seek = true,
                    AcceptStatus::End => break,
                    AcceptStatus::Yes | AcceptStatus::No => {}
                }
            }
            Ok(())
        }
        // `CompiledAutomaton.getTermsEnum`: `Terms.intersect` over the
        // compiled byte automaton, skipping every block it proves dead.
        MultiTermSource::Automaton(a) => {
            use lucene_util::automaton::CompiledAutomaton;
            let Some(ft) = fields.field(&a.field) else {
                return Ok(());
            };
            let compiled = CompiledAutomaton::with_options(&a.automaton, false, true, a.binary)
                .map_err(|e| crate::Error::InvalidQuery(format!("automaton: {e:?}")))?;
            let mut it = ft.compiled_terms(&compiled);
            take(&mut it)
        }
    }
}

/// `TermCollectingRewrite.collectTerms` over `segments` (each segment's
/// fields, in reader order) then the rewrite method's `build`: the boolean,
/// constant-scored boolean, or blended query the scoring rewrites produce.
/// `None` for the constant-score and doc-values rewrites, which run per
/// segment without a reader-wide rewrite.
pub(crate) fn rewrite_multi_term(
    segments: &[&BlockTreeFields],
    q: &MultiTermQuery,
) -> Result<Option<Clause>> {
    let field = q.field().to_string();
    let top = match q.rewrite {
        RewriteMethod::ConstantScoreBlended
        | RewriteMethod::ConstantScore
        | RewriteMethod::DocValues => return Ok(None),
        RewriteMethod::ScoringBoolean | RewriteMethod::ConstantScoreBoolean => None,
        RewriteMethod::TopTermsScoringBoolean(n)
        | RewriteMethod::TopTermsBoostOnlyBoolean(n)
        | RewriteMethod::TopTermsBlendedFreqScoring(n) => Some(n.min(MAX_CLAUSE_COUNT)),
    };
    // Term -> (docFreq, totalTermFreq) summed over the segments that have it.
    let mut terms: std::collections::BTreeMap<Vec<u8>, (i64, i64)> = Default::default();
    for fields in segments {
        for (term, seeked) in expand_terms(fields, &q.source, top)? {
            if let Some(limit) = top {
                // `TopTermsRewrite`'s queue with every boost `1`: a term is
                // kept when it is among the `size` smallest seen so far.
                if !terms.contains_key(&term) && terms.len() == limit {
                    let largest = terms.keys().next_back().cloned().unwrap_or_default();
                    if term > largest {
                        continue;
                    }
                    terms.remove(&largest);
                }
            }
            let e = terms.entry(term).or_insert((0, 0));
            e.0 += i64::from(seeked.stats.doc_freq);
            e.1 = e.1.wrapping_add(seeked.stats.total_term_freq);
            if top.is_none() && terms.len() > MAX_CLAUSE_COUNT {
                // `ScoringRewrite.checkMaxClauseCount`.
                return Err(crate::Error::InvalidQuery("too many clauses".into()));
            }
        }
    }
    let term_query = |t: &Vec<u8>, df: i64| {
        Clause::Term(TermQuery::new(field.clone(), t.clone()).with_doc_freq(df))
    };
    Ok(Some(match q.rewrite {
        RewriteMethod::ScoringBoolean | RewriteMethod::TopTermsScoringBoolean(_) => {
            Clause::Boolean(Box::new(
                BooleanQuery::new().with_should(
                    terms
                        .iter()
                        .map(|(t, &(df, _))| term_query(t, df))
                        .collect::<Vec<_>>(),
                ),
            ))
        }
        RewriteMethod::ConstantScoreBoolean => {
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                Clause::Boolean(Box::new(
                    BooleanQuery::new().with_should(
                        terms
                            .iter()
                            .map(|(t, &(df, _))| term_query(t, df))
                            .collect::<Vec<_>>(),
                    ),
                )),
                1.0,
            )))
        }
        RewriteMethod::TopTermsBoostOnlyBoolean(_) => Clause::Boolean(Box::new(
            BooleanQuery::new().with_should(
                terms
                    .iter()
                    .map(|(t, &(df, _))| {
                        Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                            term_query(t, df),
                            1.0,
                        )))
                    })
                    .collect::<Vec<_>>(),
            ),
        )),
        RewriteMethod::TopTermsBlendedFreqScoring(_) => {
            let blended = BlendedTermQuery::new(
                terms.keys().map(|t| (field.clone(), t.clone(), 1.0f32)),
                BlendedRewrite::Boolean,
            )?;
            Clause::from(blended)
        }
        _ => return Ok(None),
    }))
}

/// The reader-wide rewrite the statistics pass made for `q`, or this
/// segment's own.
fn rewritten(ctx: &LeafContext<'_>, q: &MultiTermQuery) -> Result<Option<Clause>> {
    let key = format!("{q:?}");
    if let Some(c) = ctx.global.and_then(|g| g.extended_rewrite(&key)) {
        return Ok(Some(c.clone()));
    }
    rewrite_multi_term(&[ctx.fields], q)
}

fn multi_term<'a>(
    ctx: &LeafContext<'a>,
    q: &MultiTermQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    match q.rewrite {
        RewriteMethod::ConstantScoreBlended | RewriteMethod::ConstantScore => {
            let blended = q.rewrite == RewriteMethod::ConstantScoreBlended;
            if blended {
                let classic = match &q.source {
                    MultiTermSource::Prefix(p) => Some(Clause::Prefix(p.clone())),
                    MultiTermSource::Wildcard(w) => Some(Clause::Wildcard(w.clone())),
                    MultiTermSource::Regexp(r) => Some(Clause::Regexp(r.clone())),
                    _ => None,
                };
                if let Some(c) = classic {
                    return build::build(ctx, &c, boost, mode, top_level);
                }
            }
            // Streamed, as the wrappers consume their `TermsEnum`: a short
            // automaton or range can match tens of thousands of terms.
            let mut stream = super::multi_term::StreamedTerms::new(ctx, q.field(), blended);
            visit_terms(ctx.fields, &q.source, None, &mut |term, seeked| {
                stream.push(term, seeked)?;
                Ok(if stream.settled() {
                    ControlFlow::Break(())
                } else {
                    ControlFlow::Continue(())
                })
            })?;
            stream.finish(ctx, q.field(), boost, mode)
        }
        // `DocValuesRewriteMethod.rewrite`: `new ConstantScoreQuery(new
        // MultiTermQueryDocValuesWrapper(query))`. The wrapper's weight is
        // created without scores, so the segment's query cache sees it as
        // Lucene's `LRUQueryCache` does.
        RewriteMethod::DocValues if mode.needs_scores() => {
            let wrapper = Clause::Extended(Box::new(ExtendedQuery::MultiTerm(q.clone())));
            let constant = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(wrapper, 1.0)));
            build::build(ctx, &constant, boost, mode, top_level)
        }
        RewriteMethod::DocValues => super::ranges::doc_values_rewrite(ctx, q, boost, mode),
        _ => match rewritten(ctx, q)? {
            Some(c) => build::build(ctx, &c, boost, mode, top_level),
            None => Ok(None),
        },
    }
}

// ---------------------------------------------------------------------------
// IndriAndQuery
// ---------------------------------------------------------------------------

/// `IndriAndScorer`: the disjunction of its clauses. Its score folds only
/// sub-scorers that are themselves `IndriScorer`s (a nested composite
/// `IndriAndScorer`) -- a term's `TermScorer` is not one, so it contributes
/// neither a score nor a boost, exactly as `scoreDoc`'s `instanceof` check
/// has it.
struct IndriAndScorer<'a> {
    approx: DisiApprox<BoxScorer<'a>>,
    boost: f32,
    /// The sub-scorers that are `IndriScorer`s, with their boosts: fixed
    /// when the scorer is built, so the per-document `instanceof` walk visits
    /// only these (none, for an Indri query of plain terms, whose every
    /// document then scores `0`).
    indri: Vec<(usize, f32)>,
}

impl<'a> IndriAndScorer<'a> {
    fn new(subs: Vec<BoxScorer<'a>>, boost: f32) -> Self {
        let indri = subs
            .iter()
            .enumerate()
            .filter_map(|(i, s)| s.indri_boost().map(|b| (i, b)))
            .collect();
        IndriAndScorer {
            approx: DisiApprox::new(subs, i64::MAX),
            boost,
            indri,
        }
    }

    /// `scoreDoc(subScorers, docId)`.
    fn score_doc(&mut self, doc: i32) -> Result<f32> {
        let mut score = 0.0f64;
        let mut boost_sum = 0.0f64;
        for &(i, b) in &self.indri {
            let sub = &mut self.approx.subs[i];
            // `score()` on the document, `smoothingScore(docId)` otherwise;
            // both are `scoreDoc` for a composite.
            let s = sub.smoothing_score(doc)?;
            score += f64::from(s) * f64::from(b);
            boost_sum += f64::from(b);
        }
        Ok(if boost_sum == 0.0 {
            0.0
        } else {
            (score / boost_sum) as f32
        })
    }
}

impl Scorer for IndriAndScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.approx.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.approx.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.approx.advance(target)
    }
    fn cost(&self) -> i64 {
        self.approx.cost()
    }
    fn score(&mut self) -> Result<f32> {
        let doc = self.doc_id();
        self.score_doc(doc)
    }
    /// `IndriDisjunctionScorer.getMaxScore`: `0`.
    fn max_score(&mut self, _up_to: i32) -> Result<f32> {
        Ok(0.0)
    }
    fn indri_boost(&self) -> Option<f32> {
        Some(self.boost)
    }
    fn smoothing_score(&mut self, doc: i32) -> Result<f32> {
        self.score_doc(doc)
    }
}

/// `IndriAndWeight`: every clause's weight is created with boost `1`; no
/// scorer is `null`, one is itself, more are an [`IndriAndScorer`] carrying
/// the query's boost.
fn indri_and<'a>(
    ctx: &LeafContext<'a>,
    q: &IndriAndQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let mut subs = Vec::new();
    for c in &q.clauses {
        if let Some(s) = build::build(ctx, c, 1.0, mode, false)? {
            subs.push(s);
        }
    }
    Ok(match subs.len() {
        0 => None,
        1 => subs.pop(),
        _ => Some(Box::new(IndriAndScorer::new(subs, boost))),
    })
}

// ---------------------------------------------------------------------------
// LogOddsFusionQuery
// ---------------------------------------------------------------------------

const CLAMP_MIN: f32 = 1e-7;
const CLAMP_MAX: f32 = 1.0 - 1e-7;

/// `LogOddsFusionScorer.logit`.
pub(crate) fn logit(p: f32) -> f32 {
    let clamped = p.clamp(CLAMP_MIN, CLAMP_MAX);
    f64::from(clamped / (1.0 - clamped)).ln() as f32
}

/// `LogOddsFusionScorer.sigmoid` / `BayesianScoreQuery.sigmoid`.
pub(crate) fn sigmoid(x: f32) -> f32 {
    if x >= 0.0 {
        (1.0 / (1.0 + (-f64::from(x)).exp())) as f32
    } else {
        let e = f64::from(x).exp();
        (e / (1.0 + e)) as f32
    }
}

/// `LogOddsFusionScorer.softplus`.
pub(crate) fn softplus(x: f32) -> f32 {
    if x > 20.0 {
        return x;
    }
    f64::from(x).exp().ln_1p() as f32
}

/// The fusion's parameters, shared by the scorer.
struct Fusion {
    total_clauses: usize,
    scaling: f32,
    weights: Option<Vec<f32>>,
    bounds: Option<(Vec<f32>, Vec<f32>)>,
}

impl Fusion {
    /// `gateLogit(rawLogit, signalIndex)`.
    fn gate(&self, raw: f32, i: usize) -> f32 {
        if let Some((min, max)) = &self.bounds {
            let range = max[i] - min[i];
            if range > 0.0 {
                return ((raw - min[i]) / range).clamp(0.0, 1.0);
            }
            return 0.5;
        }
        softplus(raw)
    }

    fn finish(&self, sum: f64) -> f32 {
        let scaled = if self.weights.is_some() {
            sum as f32 * self.scaling
        } else {
            (sum / self.total_clauses as f64) as f32 * self.scaling
        };
        sigmoid(scaled)
    }
}

struct LogOddsScorer<'a> {
    approx: DisiApprox<BoxScorer<'a>>,
    fusion: Fusion,
    two_phase: bool,
    list: Vec<usize>,
    verified: Vec<usize>,
}

impl Scorer for LogOddsScorer<'_> {
    fn doc_id(&self) -> i32 {
        self.approx.doc_id()
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.approx.next_doc()
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.approx.advance(target)
    }
    fn cost(&self) -> i64 {
        self.approx.cost()
    }
    fn two_phase(&self) -> bool {
        self.two_phase
    }
    /// `DisjunctionScorer.TwoPhase.matches`: the members on the document
    /// that match, the ones without a two-phase view first.
    fn matches(&mut self) -> Result<bool> {
        self.approx.top_list(&mut self.list);
        self.verified.clear();
        for &w in self.list.iter().rev() {
            if !self.approx.subs[w].two_phase() {
                self.verified.push(w);
            }
        }
        for &w in &self.list {
            if self.approx.subs[w].two_phase() && self.approx.subs[w].matches()? {
                self.verified.push(w);
            }
        }
        Ok(!self.verified.is_empty())
    }
    fn score(&mut self) -> Result<f32> {
        if !self.two_phase {
            self.approx.top_list(&mut self.list);
            self.verified.clone_from(&self.list);
        }
        let mut sum = 0.0f64;
        for k in 0..self.verified.len() {
            let w = self.verified[k];
            let sub = self.approx.subs[w].score()?;
            // `scorerIndexMap` only exists with weights; without, every
            // clause gates with index 0's bounds.
            let idx = if self.fusion.weights.is_some() { w } else { 0 };
            let gated = self.fusion.gate(logit(sub), idx);
            match &self.fusion.weights {
                Some(ws) => sum += f64::from(ws[w] * gated),
                None => sum += f64::from(gated),
            }
        }
        Ok(self.fusion.finish(sum))
    }
    /// `LogOddsFusionScorer.getMaxScore`.
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let mut sum = 0.0f64;
        for i in 0..self.approx.subs.len() {
            if self.approx.subs[i].doc_id() <= up_to {
                let m = self.approx.subs[i].max_score(up_to)?;
                let gated = self.fusion.gate(logit(m), i);
                match &self.fusion.weights {
                    Some(ws) => sum += f64::from(ws[i] * gated),
                    None => sum += f64::from(gated),
                }
            }
        }
        Ok(self.fusion.finish(sum))
    }
}

/// `LogOddsFusionWeight.scorerSupplier`: the clauses present in the segment;
/// one is itself, more are a [`LogOddsScorer`] over the active clauses'
/// weights and bounds, with `n^alpha` over **every** clause.
fn log_odds<'a>(
    ctx: &LeafContext<'a>,
    q: &LogOddsFusionQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let q = rewrite_log_odds(q);
    let q = match q {
        Rewritten::None => return Ok(None),
        Rewritten::One(c) => return build::build(ctx, &c, boost, mode, false),
        Rewritten::Fusion(q) => q,
    };
    let mut subs = Vec::new();
    let mut weights = q.weights.as_ref().map(|_| Vec::new());
    let mut bounds = q.logit_bounds.as_ref().map(|_| (Vec::new(), Vec::new()));
    for (i, c) in q.clauses.iter().enumerate() {
        if let Some(s) = build::build(ctx, c, boost, mode, false)? {
            subs.push(s);
            if let (Some(w), Some(all)) = (weights.as_mut(), q.weights.as_ref()) {
                w.push(all[i]);
            }
            if let (Some((lo, hi)), Some((all_lo, all_hi))) =
                (bounds.as_mut(), q.logit_bounds.as_ref())
            {
                lo.push(all_lo[i]);
                hi.push(all_hi[i]);
            }
        }
    }
    if subs.len() <= 1 {
        return Ok(subs.pop());
    }
    let total = q.clauses.len();
    let fusion = Fusion {
        total_clauses: total,
        scaling: (total as f64).powf(f64::from(q.alpha)) as f32,
        weights,
        bounds,
    };
    let two_phase = subs.iter().any(|s| s.two_phase());
    Ok(Some(Box::new(LogOddsScorer {
        approx: DisiApprox::new(subs, i64::MAX),
        fusion,
        two_phase,
        list: Vec::new(),
        verified: Vec::new(),
    })))
}

enum Rewritten {
    None,
    One(Clause),
    Fusion(LogOddsFusionQuery),
}

/// `LogOddsFusionQuery.rewrite`: no clause matches nothing, one is itself,
/// and `MatchNoDocsQuery` clauses drop out with their weights renormalised.
fn rewrite_log_odds(q: &LogOddsFusionQuery) -> Rewritten {
    if q.clauses.is_empty() {
        return Rewritten::None;
    }
    if q.clauses.len() == 1 {
        return Rewritten::One(q.clauses[0].clone());
    }
    let keep: Vec<usize> = (0..q.clauses.len())
        .filter(|&i| !matches!(q.clauses[i], Clause::MatchNoDocs(_)))
        .collect();
    if keep.len() == q.clauses.len() {
        return Rewritten::Fusion(q.clone());
    }
    match keep.len() {
        0 => Rewritten::None,
        1 => Rewritten::One(q.clauses[keep[0]].clone()),
        _ => {
            let weights = q.weights.as_ref().map(|w| {
                let mut kept: Vec<f32> = keep.iter().map(|&i| w[i]).collect();
                let sum: f32 = kept.iter().sum();
                if sum > 0.0 {
                    for x in kept.iter_mut() {
                        *x /= sum;
                    }
                }
                kept
            });
            let bounds = q.logit_bounds.as_ref().map(|(lo, hi)| {
                (
                    keep.iter().map(|&i| lo[i]).collect(),
                    keep.iter().map(|&i| hi[i]).collect(),
                )
            });
            Rewritten::Fusion(LogOddsFusionQuery {
                clauses: keep.iter().map(|&i| q.clauses[i].clone()).collect(),
                alpha: q.alpha,
                weights,
                logit_bounds: bounds,
            })
        }
    }
}

// ---------------------------------------------------------------------------
// BayesianScoreQuery
// ---------------------------------------------------------------------------

struct BayesianScorer<'a> {
    inner: BoxScorer<'a>,
    alpha: f32,
    beta: f32,
    logit_base_rate: f32,
}

impl BayesianScorer<'_> {
    fn transform(&self, s: f32) -> f32 {
        sigmoid(self.alpha * (s - self.beta) + self.logit_base_rate)
    }
}

impl Scorer for BayesianScorer<'_> {
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
    fn two_phase(&self) -> bool {
        self.inner.two_phase()
    }
    fn matches(&mut self) -> Result<bool> {
        self.inner.matches()
    }
    fn match_cost(&self) -> f32 {
        self.inner.match_cost()
    }
    fn score(&mut self) -> Result<f32> {
        let s = self.inner.score()?;
        Ok(self.transform(s))
    }
    fn advance_shallow(&mut self, target: i32) -> Result<i32> {
        self.inner.advance_shallow(target)
    }
    fn max_score(&mut self, up_to: i32) -> Result<f32> {
        let m = self.inner.max_score(up_to)?;
        Ok(self.transform(m))
    }
    /// `BayesianScoreScorer.setMinCompetitiveScore`: the threshold mapped
    /// back through the inverse sigmoid.
    fn set_min_competitive_score(&mut self, min: f32) -> Result<()> {
        if min > 0.0 && min < 1.0 {
            let clamped = min.clamp(1e-7, 1.0 - 1e-7);
            let logit_min = f64::from(clamped / (1.0 - clamped)).ln() as f32;
            let inner_min = (logit_min - self.logit_base_rate) / self.alpha + self.beta;
            self.inner.set_min_competitive_score(inner_min.max(0.0))?;
        }
        Ok(())
    }
    fn doc_id_run_end(&self) -> i32 {
        self.inner.doc_id_run_end()
    }
}

fn bayesian<'a>(
    ctx: &LeafContext<'a>,
    q: &BayesianScoreQuery,
    boost: f32,
    mode: Mode,
    top_level: bool,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(inner) = build::build(ctx, &q.query, boost, mode, top_level)? else {
        return Ok(None);
    };
    if !mode.needs_scores() {
        return Ok(Some(inner));
    }
    Ok(Some(Box::new(BayesianScorer {
        inner,
        alpha: q.alpha,
        beta: q.beta,
        logit_base_rate: q.logit_base_rate(),
    })))
}

// ---------------------------------------------------------------------------
// DocAndScoreQuery
// ---------------------------------------------------------------------------

/// `DocAndScoreQuery`'s scorer: this segment's slice of the fixed hits,
/// each scoring `score * boost`. The segment is found by its `docBase`,
/// which only a segment opened through its reader carries.
fn doc_and_score<'a>(
    ctx: &LeafContext<'a>,
    q: &DocAndScoreQuery,
    boost: f32,
    mode: Mode,
) -> Result<Option<BoxScorer<'a>>> {
    let Some(reader) = ctx.reader else {
        return Err(crate::Error::MissingSegmentReader(
            "DocAndScoreQuery".into(),
        ));
    };
    let Some((docs, scores)) = q.segment(reader.doc_base) else {
        return Ok(None);
    };
    if docs.is_empty() {
        return Ok(None);
    }
    let scores = if mode.needs_scores() {
        scores.iter().map(|&s| s * boost).collect()
    } else {
        Vec::new()
    };
    Ok(Some(Box::new(DocList::new(docs, scores))))
}

// ---------------------------------------------------------------------------
// Resolving outside the tree
// ---------------------------------------------------------------------------

/// The documents (and, with `scores`, each one's score) an extended query
/// matches in one segment, for the eager paths that resolve a clause to a
/// list: `resolve_clause_docs` and `clause_scores`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn resolve(
    fields: &BlockTreeFields,
    doc_in: Option<&lucene_codecs::postings::DocInput<'_>>,
    pos_in: Option<&lucene_codecs::postings::PosInput<'_>>,
    pay_in: Option<&lucene_codecs::postings::PayInput<'_>>,
    live_docs: Option<&lucene_util::fixed_bit_set::FixedBitSet>,
    points: Option<&crate::points_query::PointsInput<'_>>,
    norms: Option<&std::collections::HashMap<String, crate::FieldNorms<'_>>>,
    global: Option<&crate::GlobalStats>,
    q: &ExtendedQuery,
    scores: bool,
) -> Result<Vec<(i32, f32)>> {
    crate::explain::with_leaf_reader(|reader, similarity| {
        let ctx = LeafContext {
            fields,
            doc_in,
            pos_in,
            pay_in,
            live_docs,
            points,
            norms,
            global,
            // Set while `IndexSearcher.explain` runs: a block join's parent
            // filter needs the segment's size.
            max_doc: crate::explain::leaf().map(|(max_doc, _)| max_doc),
            cache: None,
            // Set while `IndexSearcher.explain` runs: a `DocAndScoreQuery` finds
            // its segment by the reader's doc base, and the scorer scores under
            // the searcher's similarity.
            reader,
            similarity,
        };
        let mode = if scores {
            Mode::Complete
        } else {
            Mode::NoScores
        };
        let Some(mut s) = build(&ctx, q, 1.0, mode, false)? else {
            return Ok(Vec::new());
        };
        let mut out = Vec::new();
        let mut doc = super::exact_next(&mut *s)?;
        while doc != NO_MORE_DOCS {
            if live_docs.is_none_or(|l| l.get_doc(doc)) {
                let score = if scores { s.score()? } else { 0.0 };
                out.push((doc, score));
            }
            doc = super::exact_next(&mut *s)?;
        }
        Ok(out)
    })
}

/// Every document `clause` matches in one segment, deletions not applied --
/// `global` the reader-wide preparation of its function queries, if any --
/// its scorer's iterator (`Weight.scorer(ctx).iterator()`) -- as a bit set
/// of `max_doc` documents. Collected through the clause's bulk scorer, a
/// window at a time rather than a scorer step each: the same documents.
pub(crate) fn segment_match_bits(
    seg: &crate::multi_segment::OpenSegment<'_>,
    clause: &Clause,
    global: Option<&crate::GlobalStats>,
) -> Result<lucene_util::fixed_bit_set::FixedBitSet> {
    use lucene_util::fixed_bit_set::FixedBitSet;
    struct Bits(FixedBitSet);
    impl crate::collector::ScoringCollector for Bits {
        fn collect(&mut self, doc_id: i32, _score: f32) {
            if let Ok(d) = usize::try_from(doc_id) {
                // FBS: a scorer returns this segment's documents, below
                // `maxDoc`, the set's length; the check keeps a corrupt one
                // from panicking.
                if d < self.0.len() {
                    self.0.set(d);
                }
            }
        }
        fn score_mode(&self) -> crate::collector::ScoreMode {
            crate::collector::ScoreMode::CompleteNoScores
        }
    }
    let ctx = LeafContext {
        fields: seg.fields,
        doc_in: seg.doc_in,
        pos_in: seg.pos_in,
        pay_in: seg.pay_in,
        live_docs: None,
        points: seg.points,
        norms: None,
        global,
        max_doc: seg.max_doc,
        cache: None,
        reader: seg.reader,
        similarity: None,
    };
    let mut bits = Bits(FixedBitSet::new(
        usize::try_from(seg.max_doc.unwrap_or(0)).unwrap_or(0),
    ));
    if let Some(mut bulk) = super::bulk_clause(&ctx, clause, 1.0, Mode::NoScores)? {
        super::score_segment(&mut bulk, Mode::NoScores, None, &mut bits)?;
    }
    Ok(bits.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn imp(freq: i32, norm: i64) -> Impact {
        Impact { freq, norm }
    }

    /// `SynonymQuery.mergeImpacts`: the implicit impacts between a list's
    /// own norms are counted (`{2, 10}` still bounds a document of norm 11),
    /// equal norms sum, a sum that does not rise is dropped, and norms order
    /// unsigned (a sign-extended byte norm sorts after every positive one).
    #[test]
    fn merged_impacts_sum_frequencies_by_norm_as_lucene_does() {
        assert!(merge_impacts(&[]).is_none());
        let one = vec![imp(2, 10), imp(4, 12)];
        assert_eq!(merge_impacts(std::slice::from_ref(&one)), Some(one.clone()));
        let other = vec![imp(1, 11), imp(3, 13)];
        assert_eq!(
            merge_impacts(&[one.clone(), other]),
            Some(vec![imp(2, 10), imp(3, 11), imp(5, 12), imp(7, 13)])
        );
        assert_eq!(
            merge_impacts(&[one, vec![imp(3, 10)]]),
            Some(vec![imp(5, 10), imp(7, 12)])
        );
        // Norm -1 is the largest unsigned: it comes last, and a frequency
        // that does not rise past the running maximum adds no impact.
        assert_eq!(
            merge_impacts(&[vec![imp(1, 5), imp(9, -1)], vec![imp(1, 3)]]),
            Some(vec![imp(1, 3), imp(2, 5), imp(10, -1)])
        );
        assert_eq!(
            merge_impacts(&[vec![imp(4, 1)], vec![imp(1, 2)], vec![imp(1, 1)]]),
            Some(vec![imp(5, 1), imp(6, 2)])
        );
    }

    #[test]
    fn logit_sigmoid_softplus_follow_java() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert!(sigmoid(-50.0) > 0.0 && sigmoid(-50.0) < 1e-20);
        assert_eq!(softplus(25.0), 25.0);
        assert!((softplus(0.0) - std::f32::consts::LN_2).abs() < 1e-7);
        assert!(logit(0.0) < -15.0);
        assert!(logit(1.0) > 15.0);
        assert_eq!(logit(0.5), 0.0);
    }

    #[test]
    fn explicit_offsets_rebase_onto_slots() {
        let mut scratch = sloppy_phrase::SloppyScratch::default();
        let mut shifted = Vec::new();
        // "a _ b": a at 3, b at 5, the phrase has a hole at 1.
        let positions = vec![vec![3], vec![5]];
        let none = sloppy_phrase::PhraseRepeats::none(2);
        assert_eq!(
            phrase_freq_at(&positions, &[0, 2], &none, 0, &mut scratch, &mut shifted),
            1.0
        );
        assert_eq!(
            phrase_freq_at(&positions, &[0, 1], &none, 0, &mut scratch, &mut shifted),
            0.0
        );
        // One position of slop covers the hole's absence.
        assert!(phrase_freq_at(&positions, &[0, 1], &none, 1, &mut scratch, &mut shifted) > 0.0);
    }

    #[test]
    fn log_odds_rewrite_drops_match_no_docs_and_renormalises() {
        let t = |s: &str| Clause::Term(TermQuery::new("f", s));
        let none = Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new());
        let q = LogOddsFusionQuery::new(
            [t("a"), none.clone(), t("b")],
            0.5,
            Some(vec![0.5, 0.25, 0.25]),
            Some((vec![0.0, 1.0, 2.0], vec![3.0, 4.0, 5.0])),
        )
        .unwrap();
        let Rewritten::Fusion(r) = rewrite_log_odds(&q) else {
            panic!("two clauses stay a fusion")
        };
        assert_eq!(r.clauses.len(), 2);
        assert_eq!(r.weights.unwrap(), vec![0.5 / 0.75, 0.25 / 0.75]);
        assert_eq!(r.logit_bounds.unwrap(), (vec![0.0, 2.0], vec![3.0, 5.0]));
        let one = LogOddsFusionQuery::new([t("a"), none.clone()], 0.5, None, None).unwrap();
        assert!(matches!(rewrite_log_odds(&one), Rewritten::One(_)));
        let zero = LogOddsFusionQuery::new([none.clone(), none], 0.5, None, None).unwrap();
        assert!(matches!(rewrite_log_odds(&zero), Rewritten::None));
    }

    #[test]
    fn multi_norms_reencode_the_weighted_length() {
        let mut none = Norms::Multi(Vec::new(), None);
        assert_eq!(none.norm(3, false).unwrap(), 1);
        let mut one = Norms::One(None);
        assert_eq!(one.norm(3, true).unwrap(), 1);
    }

    /// Dense one-byte norms, as `Lucene90NormsConsumer` writes them for an
    /// ordinary analyzed field: `bytes[doc]` is the document's norm.
    fn dense_norms(bytes: &[u8]) -> crate::field_norms::FieldNorms<'_> {
        let entry = lucene_codecs::norms::NormsEntry {
            field_number: 0,
            docs_with_field_offset: -1,
            docs_with_field_length: 0,
            jump_table_entry_count: -1,
            dense_rank_power: 0xFF,
            num_docs_with_field: bytes.len() as i32,
            bytes_per_norm: 1,
            norms_offset: 0,
        };
        crate::field_norms::FieldNorms::from_field_stats(bytes, entry, 100, 10)
    }

    /// [`DenseMemo`] is a cache of [`Norms::norm`]'s own arithmetic (which
    /// the `CombinedFieldQuery` fixtures verify against Lucene): it must give
    /// exactly the general path's value for every pair of norm bytes -- the
    /// ones that decode as negative `i8` included -- for one field and two,
    /// whatever the weights, on a first (computed) and a second (remembered)
    /// lookup, and step aside for a document its arrays do not cover.
    #[test]
    fn dense_memo_is_the_general_multi_norm_for_every_byte_pair() {
        let n = 1usize << 16;
        let low: Vec<u8> = (0..n).map(|d| (d & 0xFF) as u8).collect();
        let high: Vec<u8> = (0..n).map(|d| (d >> 8) as u8).collect();
        let (a, b) = (dense_norms(&low), dense_norms(&high));
        for (wa, wb) in [(1.0f32, 1.0f32), (2.5, 3.0), (1.0, 7.25), (13.0, 1.5)] {
            let fields = vec![(a.cursor(), wa), (b.cursor(), wb)];
            let mut memo = DenseMemo::new(&fields).expect("both fields are dense");
            let mut general = Norms::Multi(fields, None);
            for pass in 0..2 {
                for doc in 0..n as i32 {
                    let want = general.norm(doc, false).unwrap();
                    assert_eq!(
                        memo.norm(doc),
                        Some(want),
                        "{wa}/{wb} doc {doc} pass {pass}"
                    );
                }
            }
            assert_eq!(memo.norm(n as i32), None);
            assert_eq!(memo.norm(-1), None);
        }
        let fields = vec![(a.cursor(), 3.5f32)];
        let mut memo = DenseMemo::new(&fields).expect("one dense field");
        let mut general = Norms::Multi(fields, None);
        for doc in 0..256 {
            assert_eq!(memo.norm(doc), Some(general.norm(doc, false).unwrap()));
        }
        // Three fields, or a field that is not dense one-byte, have none.
        let three = vec![(a.cursor(), 1.0), (b.cursor(), 1.0), (a.cursor(), 1.0)];
        assert!(DenseMemo::new(&three).is_none());
        let unnormed = crate::field_norms::FieldNorms::unnormed(10, 1.0);
        assert!(DenseMemo::new(&[(unnormed.cursor(), 1.0)]).is_none());
        // `Norms::norm` answers from the memo under `advanceExact`'s rule
        // only; `longValues`' rule reads the fields.
        let fields = vec![(a.cursor(), 1.0f32), (b.cursor(), 1.0f32)];
        let memo = DenseMemo::new(&fields);
        let mut with_memo = Norms::Multi(fields, memo);
        let mut without = Norms::Multi(vec![(a.cursor(), 1.0), (b.cursor(), 1.0)], None);
        for doc in [0, 1, 255, 256, 40_000] {
            for batch in [false, true] {
                assert_eq!(
                    with_memo.norm(doc, batch).unwrap(),
                    without.norm(doc, batch).unwrap()
                );
            }
        }
    }
}
