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
use super::{BoxScorer, Mode};
use crate::query::SpanQuery;
use crate::Result;

/// One leaf `(field, term)`'s positions in one document.
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

/// The `Spans` of `q` over one document, walked to `NO_MORE_POSITIONS`: each
/// emitted span's `(startPosition(), endPosition(), width())`, in emission
/// order.
pub(crate) fn emissions(q: &SpanQuery, doc: &DocPositions) -> Vec<(i32, i32, i64)> {
    match q {
        // `TermSpans`: one span per position, `width() == 0`.
        SpanQuery::SpanTerm { field, term } => doc
            .get(&(field.clone(), term.clone()))
            .map(|ps| ps.iter().map(|&p| (p, p.saturating_add(1), 0)).collect())
            .unwrap_or_default(),
        // `SpanOrQuery`'s spans: every clause matching in the document, merged
        // by `(start, end)` (`SpanPositionQueue`), repeats kept.
        SpanQuery::SpanOr { clauses } => {
            let mut all: Vec<(i32, i32, i64)> =
                clauses.iter().flat_map(|c| emissions(c, doc)).collect();
            all.sort_by_key(|&(s, e, _)| (s, e));
            all
        }
        SpanQuery::SpanNear {
            clauses,
            slop,
            in_order,
        } => {
            if clauses.is_empty() {
                return Vec::new();
            }
            let per: Vec<Vec<(i32, i32)>> = clauses
                .iter()
                .map(|c| {
                    emissions(c, doc)
                        .into_iter()
                        .map(|(s, e, _)| (s, e))
                        .collect()
                })
                .collect();
            if per.iter().any(Vec::is_empty) {
                return Vec::new();
            }
            let slices: Vec<&[(i32, i32)]> = per.iter().map(Vec::as_slice).collect();
            let slop = i64::from(*slop);
            let mut out = Vec::new();
            if *in_order {
                // `NearSpansOrdered.width()`: `matchWidth`, the gaps
                // `stretchToOrder` summed.
                crate::near_spans::for_each_ordered_match(&slices, slop, |arr, start, end| {
                    let width = arr
                        .windows(2)
                        .map(|w| i64::from(w[1].0) - i64::from(w[0].1))
                        .fold(0i64, i64::saturating_add);
                    out.push((start, end, width));
                });
            } else {
                // `NearSpansUnordered.width()`:
                // `maxEndPosition - top().startPosition()`.
                crate::near_spans::for_each_unordered_match(&slices, slop, |_, start, end| {
                    out.push((start, end, i64::from(end) - i64::from(start)));
                });
            }
            out
        }
    }
}

/// `SpanScorer.setFreqCurrentDoc`: `sum(1 / (1 + width))` over the
/// document's spans, each step rounded to `float` as Java's compound
/// assignment rounds it; `0` when it has none.
pub(crate) fn sloppy_freq(q: &SpanQuery, doc: &DocPositions) -> f32 {
    let mut freq = 0.0f32;
    for (_, _, width) in emissions(q, doc) {
        freq = (f64::from(freq) + 1.0 / (1.0 + width as f64)) as f32;
    }
    freq
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
    let Some((candidates, per_leaf)) = crate::span_leaf_positions(
        ctx.fields,
        ctx.doc_in,
        ctx.pos_in,
        ctx.pay_in,
        ctx.live_docs,
        q,
    )?
    else {
        return Ok(None);
    };
    let mut norms = norms_cursor(ctx, field);
    let (mut docs, mut scores) = (Vec::new(), Vec::new());
    let mut doc_positions: DocPositions = HashMap::new();
    for doc in candidates {
        doc_positions.clear();
        for (key, map) in &per_leaf {
            if let Some(positions) = map.get(&doc) {
                doc_positions.insert(key.clone(), positions.clone());
            }
        }
        let freq = sloppy_freq(q, &doc_positions);
        if freq == 0.0 {
            continue;
        }
        docs.push(doc);
        if let Some(scorer) = &scorer {
            let norm = match norms.as_mut() {
                Some(n) => n.norm_long(doc)?.unwrap_or(1),
                None => 1,
            };
            scores.push(scorer.score(freq, norm));
        }
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
}
