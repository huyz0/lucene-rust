//! Span query matching: positions of span leaves, `SpanNear` matches and span hit extents.

use super::*;

/// Key type for [`span_matches_in_doc`]'s `doc_positions` map: one leaf
/// `SpanQuery::SpanTerm`'s `(field, term)` pair, the finest granularity a span
/// query ever needs a position list for (unlike `PhraseQuery`, a `SpanQuery`'s
/// leaves aren't all implicitly the same field — see [`SpanQuery`]'s doc
/// comment).
pub(crate) type SpanLeafKey = (String, Vec<u8>);

/// Computes `query`'s matching span ranges (`[start, end)` position pairs, real
/// `SpanTermQuery`/`SpanNearQuery`/`SpanOrQuery`'s per-doc result shape) against
/// one doc's already-decoded position lists, `doc_positions` -- one sorted,
/// ascending position list per distinct `SpanQuery::SpanTerm` leaf appearing
/// anywhere in `query` (a leaf whose `(field, term)` pair has no entry, or an
/// empty entry, is treated as "no occurrences in this doc", same convention
/// [`phrase_matches_in_doc`]'s callers already rely on -- see
/// [`span_doc_ids`]'s doc comment for how this map is built).
///
/// **Scope**: this is the direct, in-memory span computation
/// [`SpanQuery`]'s own doc comment describes (not a lazy `Spans` iterator) --
/// callers needing every matching span range for a doc call this once per
/// doc, the same shape `phrase_matches_in_doc`/[`sloppy_phrase`]
/// already use for `PhraseQuery`.
///
/// - `SpanQuery::SpanTerm`: every occurrence in `doc_positions` becomes exactly
///   one `(position, position + 1)` span -- real `SpanTermQuery`'s exact
///   semantics (`termFreq` occurrences, not just "does it occur").
/// - `SpanQuery::SpanOr`: the union (sorted, deduplicated) of every
///   sub-`SpanQuery`'s own spans -- real `SpanOrQuery`'s exact semantics.
/// - `SpanQuery::SpanNear`: delegates to [`span_near_matches`] -- see that
///   function's doc comment for the `slop`/`in_order` algorithm, including the
///   `in_order == false` any-order case that's this type's key differentiator
///   from `PhraseQuery`'s in-order-only sloppy matching.
///
/// Returned spans are sorted ascending and deduplicated (`(start, end)`
/// lexicographic order) regardless of variant, so a caller can treat "matches"
/// as simply "the returned `Vec` is non-empty" without caring which variant
/// produced it -- exactly how [`span_doc_ids`] uses this function.
pub(crate) fn span_matches_in_doc(
    query: &SpanQuery,
    doc_positions: &HashMap<SpanLeafKey, Vec<i32>>,
) -> Vec<(i32, i32)> {
    match query {
        SpanQuery::SpanTerm { field, term } => {
            let key = (field.clone(), term.clone());
            doc_positions
                .get(&key)
                .map(|positions| positions.iter().map(|&p| (p, p + 1)).collect())
                .unwrap_or_default()
        }
        SpanQuery::SpanOr { clauses } => {
            let mut spans: Vec<(i32, i32)> = clauses
                .iter()
                .flat_map(|clause| span_matches_in_doc(clause, doc_positions))
                .collect();
            spans.sort_unstable();
            spans.dedup();
            spans
        }
        SpanQuery::SpanNear {
            clauses,
            slop,
            in_order,
        } => span_near_matches(clauses, *slop, *in_order, doc_positions),
    }
}

/// [`SpanQuery::SpanNear`]'s matching algorithm: `NearSpansOrdered`'s or
/// `NearSpansUnordered`'s walk (`crate::near_spans`) over every clause's own
/// spans, which are computed recursively via [`span_matches_in_doc`] so a
/// `SpanNear` of `SpanNear`s composes for free.
///
/// **This is a walk, not a cartesian product.** Both Java classes are `Spans`
/// iterators whose sub-span cursors only ever move forward, and the sequence
/// of `(startPosition(), endPosition())` pairs they emit is *narrower* than
/// "every arrangement of one span per clause that fits the budget":
///
/// - `in_order == true` is `NearSpansOrdered`. `stretchToOrder` advances each
///   later clause to the first span starting at or after the previous
///   clause's end and never rewinds, so each position of clause 0 contributes
///   exactly one arrangement -- the minimum-slop one. Its own class doc:
///   "the formed spans only contains minimum slop matches."
/// - `in_order == false` is `NearSpansUnordered`. `atMatch()` is a pure width
///   test -- `maxEndPosition - top().startPosition() - totalSpanLength <=
///   slop` -- with **no** non-overlap requirement and no relative-order
///   requirement at all, evaluated at each step of a priority-queue walk that
///   advances only the least span. A reversed pair matches as long as the
///   width fits, and two clauses may settle on the *same* span, giving a
///   negative width, which matches: real Lucene returns doc 8555 -- a document
///   with a single `alpha` -- for `SpanNearQuery([alpha, alpha], 0, false)`.
///   Its `endPosition()` is the *running* `maxEndPosition`, which can exceed
///   the current arrangement's own maximum end.
///
/// Enumerating the whole cartesian product and reporting `(min start, max
/// end)` per accepted arrangement would produce extents Java's walk never
/// visits. The hit set would not change ([`span_doc_ids`] only asks whether
/// the span list is non-empty), but a nested `SpanNear`-of-`SpanNear`
/// consumes the inner clause's extents. Ground truth for the sequences themselves:
/// `fixtures/src/AppendSpanExtentManifest.java`'s `spanextent.*` keys, which
/// are real `Spans.nextStartPosition()`/`endPosition()` walks.
///
/// **Slop formula**: for a *non-overlapping* arrangement the two arms compute
/// the same number -- `sum(next.start - prev.end)` telescopes to
/// `maxEnd - minStart - sum(lengths)` -- so `slop` means the same thing on
/// both wherever both accept. It is **not** the same quantity as a sloppy
/// phrase's `matchLength` (see [`sloppy_phrase`]), which is a window width
/// over slot-shifted positions.
///
/// Returned spans are sorted ascending and deduplicated, which loses nothing:
/// both walks emit `(start, end)` pairs that are already non-decreasing in
/// both components (clause 0's cursor and the queue's minimum only rise, and
/// `maxEndPosition` only rises), so sorting is a no-op and `dedup` removes
/// only the consecutive repeats a caller asking "which extents exist" does not
/// want twice.
///
/// **Complexity**: linear in the total number of sub-spans, not exponential in
/// the clause count -- the cartesian product this replaced was the latter.
///
/// **Edge cases**: an empty `clauses` list, or any clause whose own spans are
/// empty (the sub-query doesn't occur at all in this doc), both yield no
/// spans -- a `SpanNear` needs every sub-clause to contribute at least one
/// occurrence.
pub(super) fn span_near_matches(
    clauses: &[SpanQuery],
    slop: u32,
    in_order: bool,
    doc_positions: &HashMap<SpanLeafKey, Vec<i32>>,
) -> Vec<(i32, i32)> {
    if clauses.is_empty() {
        return Vec::new();
    }
    let per_clause_spans: Vec<Vec<(i32, i32)>> = clauses
        .iter()
        .map(|clause| span_matches_in_doc(clause, doc_positions))
        .collect();
    if per_clause_spans.iter().any(Vec::is_empty) {
        return Vec::new();
    }
    let slices: Vec<&[(i32, i32)]> = per_clause_spans.iter().map(Vec::as_slice).collect();

    let slop = i64::from(slop);
    let mut results: Vec<(i32, i32)> = Vec::new();
    let mut record = |_: crate::near_spans::Arrangement<'_>, start: i32, end: i32| {
        results.push((start, end));
    };
    if in_order {
        crate::near_spans::for_each_ordered_match(&slices, slop, &mut record);
    } else {
        crate::near_spans::for_each_unordered_match(&slices, slop, &mut record);
    }
    results.sort_unstable();
    results.dedup();
    results
}

/// Collects every distinct `SpanQuery::SpanTerm` leaf's `(field, term)` pair
/// appearing anywhere in `query` (recursively through `SpanNear`/`SpanOr`),
/// deduplicated -- the set of position lists [`span_doc_ids`] needs to fetch
/// before it can evaluate [`span_matches_in_doc`] for any candidate doc.
pub(crate) fn collect_span_leaves(query: &SpanQuery, leaves: &mut Vec<SpanLeafKey>) {
    match query {
        SpanQuery::SpanTerm { field, term } => leaves.push((field.clone(), term.clone())),
        SpanQuery::SpanNear { clauses, .. } | SpanQuery::SpanOr { clauses } => {
            for clause in clauses {
                collect_span_leaves(clause, leaves);
            }
        }
    }
}

/// `SpanTermQuery.SpanTermWeight.getSpans`' positions check, reached as
/// `SpanWeight.getSpans` walks `query`: a term this segment has, in a field it
/// indexed without positions, is Java's `IllegalStateException`. The walk
/// stops where Java's does -- `SpanNearQuery` at its first clause without
/// spans, `SpanOrQuery` after every clause -- so a near whose earlier term
/// is absent here reports nothing. A query whose fields all have positions
/// (or are absent) is not walked.
pub(crate) fn require_span_positions(fields: &BlockTreeFields, query: &SpanQuery) -> Result<()> {
    fn lacks(fields: &BlockTreeFields, q: &SpanQuery) -> bool {
        match q {
            SpanQuery::SpanTerm { field, .. } => fields
                .field(field)
                .is_some_and(|ft| !ft.index_options().subsumes_positions()),
            SpanQuery::SpanNear { clauses, .. } | SpanQuery::SpanOr { clauses } => {
                clauses.iter().any(|c| lacks(fields, c))
            }
        }
    }
    /// Whether `q` has spans in this segment (Java's non-`null`).
    fn walk(fields: &BlockTreeFields, q: &SpanQuery) -> Result<bool> {
        match q {
            SpanQuery::SpanTerm { field, term } => {
                let Some(ft) = fields.field(field) else {
                    return Ok(false);
                };
                if ft.try_seek_exact(term)?.is_none() {
                    return Ok(false);
                }
                if !ft.index_options().subsumes_positions() {
                    return Err(Error::IllegalState(format!(
                        "field \"{field}\" was indexed without position data; cannot run SpanTermQuery (term={})",
                        String::from_utf8_lossy(term)
                    )));
                }
                Ok(true)
            }
            SpanQuery::SpanNear { clauses, .. } => {
                for c in clauses {
                    if !walk(fields, c)? {
                        return Ok(false);
                    }
                }
                Ok(true)
            }
            SpanQuery::SpanOr { clauses } => {
                let mut any = false;
                for c in clauses {
                    any |= walk(fields, c)?;
                }
                Ok(any)
            }
        }
    }
    if lacks(fields, query) {
        walk(fields, query)?;
    }
    Ok(())
}

/// [`Clause::Span`]'s matched doc-ID list: [`span_doc_extents`]
/// with the extents dropped.
///
/// [`span_doc_extents`] gathers every distinct leaf `(field, term)` pair
/// `query` touches (via [`collect_span_leaves`]), fetches each one's
/// live-filtered `doc_id -> position list` map (via [`term_doc_positions`],
/// the same helper [`search_phrase_query`] uses), then for every doc appearing
/// in **any** leaf's doc list (a safe, simple over-approximation of the true
/// candidate set -- see below for why that's fine) builds a per-doc
/// `doc_positions` map and runs [`span_matches_in_doc`] over it.
///
/// **Why "any leaf's doc list" instead of computing each variant's own tighter
/// candidate set** (e.g. a `SpanNear`'s candidates could be the *conjunction*
/// of its sub-clauses' doc lists, `SpanOr`'s the union): this port takes the
/// simpler, uniformly-correct union-of-every-leaf approach for
/// [`SpanQuery`]'s honestly-scoped MVP (see that type's doc comment) --
/// [`span_matches_in_doc`] itself already correctly reports "no match" for a
/// candidate doc that doesn't actually satisfy a `SpanNear`'s stricter
/// requirement, so the wider candidate set costs only some wasted position-
/// list lookups, never an incorrect result. A future optimization pass could
/// tighten this per-variant if profiling shows it matters (same "correctness
/// first, profile before optimizing" call this crate's other multi-term
/// matchers already make).
///
/// Returns an empty `Vec` -- not an error -- when `query` has no leaves at all
/// (a `SpanNear`/`SpanOr` with empty `clauses`, which can never match) or when
/// every leaf's field/term is missing from this segment. Requires `pos_in` (an
/// `Err(Error::MissingPosInput)` otherwise) whenever `query` has at least one
/// leaf -- unlike `PhraseQuery`, even a single-leaf `SpanTerm` needs a real
/// position list (its spans are per-occurrence `(position, position + 1)`
/// pairs, not just doc-level presence), so there is no single-term fast path
/// that skips positions the way [`search_phrase_query`] has for a length-1
/// phrase.
#[allow(clippy::too_many_arguments)]
pub(super) fn span_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &SpanQuery,
) -> Result<Vec<i32>> {
    Ok(
        span_doc_extents(fields, doc_in, pos_in, pay_in, live_docs, query)?
            .into_iter()
            .map(|(doc_id, _)| doc_id)
            .collect(),
    )
}

/// One document's matching span extents, as [`span_doc_extents`] returns them:
/// the document id and its `(startPosition(), endPosition())` pairs.
pub type DocSpans = (i32, Vec<(i32, i32)>);

/// Every matching document's own **span extents**, which is what real
/// `SpanWeight.getSpans(context, Postings.POSITIONS)` walked to
/// `NO_MORE_DOCS`/`NO_MORE_POSITIONS` produces: one
/// `(startPosition(), endPosition())` pair per emitted match, ascending, per
/// document, live documents only.
///
/// [`span_doc_ids`] is this with the extents dropped; a caller that wants the
/// ranges -- a highlighter, or anything composing spans further -- wants this.
/// A document with no matching span is not listed, so the returned document
/// list is exactly [`span_doc_ids`]'.
///
/// The extents are the walks' own (see [`span_near_matches`]), **not** the
/// bounding box of every arrangement that fits the slop budget: for
/// `in_order == true` Java emits only minimum-slop arrangements, and for
/// `in_order == false` its `endPosition()` is a running maximum. Ground truth
/// is `fixtures/src/AppendSpanExtentManifest.java`'s `spanextent.*` keys,
/// recorded from real `Spans` walks.
#[allow(clippy::too_many_arguments)]
pub fn span_doc_extents(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &SpanQuery,
) -> Result<Vec<DocSpans>> {
    let Some((candidate_docs, per_leaf_maps)) =
        span_leaf_positions(fields, doc_in, pos_in, pay_in, live_docs, query)?
    else {
        return Ok(Vec::new());
    };
    let mut result: Vec<DocSpans> = Vec::new();
    for doc_id in candidate_docs {
        let mut doc_positions: HashMap<SpanLeafKey, Vec<i32>> = HashMap::new();
        for (key, map) in &per_leaf_maps {
            if let Some(positions) = map.get(&doc_id) {
                doc_positions.insert(key.clone(), positions.clone());
            }
        }
        let extents = span_matches_in_doc(query, &doc_positions);
        if !extents.is_empty() {
            result.push((doc_id, extents));
        }
    }
    Ok(result)
}

/// Every leaf of a span query with its live documents' positions: the
/// candidate documents (any leaf's, ascending) and, per distinct leaf
/// `(field, term)` present in the segment, a `doc -> positions` map. `None`
/// when the query has no leaf. See [`span_doc_ids`] for why the candidates
/// are the union of every leaf's documents.
#[allow(clippy::type_complexity)]
pub(crate) fn span_leaf_positions(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &SpanQuery,
) -> Result<Option<(Vec<i32>, Vec<(SpanLeafKey, HashMap<i32, Vec<i32>>)>)>> {
    require_span_positions(fields, query)?;
    let mut leaves: Vec<SpanLeafKey> = Vec::new();
    collect_span_leaves(query, &mut leaves);
    // A segment without any leaf's field has no spans (`SpanTermWeight`
    // returns none), `.pos` file or not.
    if !leaves.iter().any(|(f, _)| fields.field(f).is_some()) {
        return Ok(None);
    }
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };
    leaves.sort_unstable();
    leaves.dedup();

    let mut candidate_docs: Vec<i32> = Vec::new();
    let mut per_leaf_maps: Vec<(SpanLeafKey, HashMap<i32, Vec<i32>>)> =
        Vec::with_capacity(leaves.len());
    for (field, term) in &leaves {
        let Some((docs, positions, spans)) =
            term_doc_positions(fields, doc_in, pos_in, pay_in, live_docs, field, term)?
        else {
            // A missing leaf term contributes no occurrences anywhere --
            // `span_matches_in_doc` already treats an absent map entry the
            // same way, so this leaf simply never adds candidate docs.
            continue;
        };
        // Spans are not on the measured hot path: rebuild the map the span
        // matcher expects from the flat positions and their per-doc spans.
        let map: HashMap<i32, Vec<i32>> = docs
            .iter()
            .copied()
            .zip(spans.iter())
            .map(|(doc, &(start, end))| (doc, positions[start as usize..end as usize].to_vec()))
            .collect();
        candidate_docs.extend(docs);
        per_leaf_maps.push(((field.clone(), term.clone()), map));
    }
    candidate_docs.sort_unstable();
    candidate_docs.dedup();
    Ok(Some((candidate_docs, per_leaf_maps)))
}

/// Executes `query` (see [`query::SpanQuery`] for the exact matching
/// semantics and this port's Spans-API-vs-direct-computation scope decision)
/// against one already-opened segment, feeding every matching **live** doc ID
/// to `collector` in ascending order -- same parameter contract as
/// [`search_phrase_query`], with `pos_in` required whenever `query` has at
/// least one leaf (see [`span_doc_ids`]'s doc comment for why `SpanQuery` has
/// no single-leaf fast path that skips positions the way a length-1
/// `PhraseQuery` does).
pub fn search_span_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &SpanQuery,
    collector: &mut C,
) -> Result<()> {
    for doc_id in span_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query)? {
        collector.collect(doc_id);
    }
    Ok(())
}
