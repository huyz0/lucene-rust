//! Phrase and multi-phrase query search: exact and sloppy position matching, scored and unscored.

use super::*;

/// Checks whether `term_positions` (one sorted, ascending position list per phrase
/// term, in phrase order, all for the *same* doc) has some base position `p` such
/// that `term_positions[i]` contains `p + i` for every `i` -- `ExactPhraseScorer`'s
/// core test (`org.apache.lucene.search.ExactPhraseScorer`, slop == 0 case), done
/// here as a straightforward candidate-and-check rather than Java's stateful
/// per-postings merge: every position in `term_positions[0]` is a candidate base
/// `p`, and each candidate is checked against every other term's position list via
/// binary search (each list is already sorted, since positions are decoded and
/// grouped in increasing order by [`lucene_codecs::postings::read_positions`]).
///
/// **Edge cases, verified by this function's own unit tests below**: an empty
/// `term_positions` (no terms at all) or any single empty position list (a term
/// with zero occurrences in this doc, which callers should never actually pass in
/// practice -- doc-level conjunction already guarantees every term occurs at least
/// once) both yield `false` rather than panicking. A single-term phrase
/// (`term_positions.len() == 1`) degenerates to "does this term occur at all in
/// this doc": the inner loop over `1..len` is empty, so the first candidate
/// position always succeeds. A repeated term (e.g. "the the") works unmodified --
/// the two position lists happen to be identical, but the check only ever compares
/// `p + i` against list `i`, never compares lists against each other by identity.
pub(crate) fn phrase_matches_in_doc(term_positions: &[&[i32]]) -> bool {
    phrase_freq_exact_impl(term_positions, true) != 0
}

/// Scratch cursors for the alignment walk, one per non-leading phrase term.
/// Stack-resident for any phrase this port can realistically be handed
/// (`MAX_INLINE_PHRASE_TERMS` slots), heap only past that -- the walk runs once
/// per candidate *document*, so a `Vec` here would be one allocation per
/// document, which is the exact shape finding #O15 spent a milestone removing
/// from the position stream.
pub(super) const MAX_INLINE_PHRASE_TERMS: usize = 32;

/// The one alignment walk behind [`phrase_matches_in_doc`] and
/// [`phrase_freq_exact`]. `stop_at_first` returns as soon as one alignment is
/// found (the boolean question) instead of counting them all.
///
/// **`ExactPhraseMatcher`'s shape, not a binary search.** `p0` walks the first
/// term's positions in ascending order, so every target `p0 + i` ascends too,
/// so a cursor per subsequent term never rewinds: the whole walk is one merge
/// pass over all the lists, `O(sum of list lengths)`, which is what
/// `ExactPhraseMatcher.nextMatch` does by advancing its `PostingsEnum`s
/// together.
///
/// This replaced a `binary_search` per term per candidate
/// (`O(|p0| · (n-1) · log|list|)`). Measured by
/// `benches/phrase_freq.rs` on 4096-position lists: **5.0x-6.8x** faster
/// (26.8 -> 4.9 µs at a 50% hit rate, 39.6 -> 5.8 µs for a three-term phrase),
/// and 6.8% slower on an 8-position list where both are ~18 ns. The same
/// change as the `next_doc`/`advance` binary searches.
pub(super) fn phrase_freq_exact_impl(term_positions: &[&[i32]], stop_at_first: bool) -> i32 {
    if let [first, second] = term_positions {
        return phrase_freq_exact_two(first, second, stop_at_first);
    }
    phrase_freq_exact_general(term_positions, stop_at_first)
}

/// [`phrase_freq_exact_impl`] for any number of terms.
pub(super) fn phrase_freq_exact_general(term_positions: &[&[i32]], stop_at_first: bool) -> i32 {
    let Some((first, rest)) = term_positions.split_first() else {
        return 0;
    };
    if rest.iter().any(|positions| positions.is_empty()) {
        return 0;
    }
    let mut inline = [0usize; MAX_INLINE_PHRASE_TERMS];
    let mut spilled;
    let cursors: &mut [usize] = if rest.len() <= MAX_INLINE_PHRASE_TERMS {
        &mut inline[..rest.len()]
    } else {
        spilled = vec![0usize; rest.len()];
        &mut spilled
    };

    let mut freq = 0;
    'candidate: for &p0 in first.iter() {
        for (i, positions) in rest.iter().enumerate() {
            let target = p0 + (i as i32 + 1);
            let mut c = cursors[i];
            while c < positions.len() && positions[c] < target {
                c += 1;
            }
            cursors[i] = c;
            if c == positions.len() {
                // This term is exhausted, and every later `p0` needs a strictly
                // larger target, so no later alignment can succeed either.
                break 'candidate;
            }
            if positions[c] != target {
                continue 'candidate;
            }
        }
        freq += 1;
        if stop_at_first {
            break;
        }
    }
    freq
}

/// [`phrase_freq_exact_impl`] for a two-term phrase -- every phrase in the
/// benchmark mix, and the common case: one merge of two ascending lists, counting each
/// `p` of `first` whose `p + 1` is in `second`. The general loop's cursor
/// table, slice-of-slices indexing and per-term bookkeeping were two thirds
/// of its cost on `"t0 t1"`, whose documents hold a handful of positions each.
pub(super) fn phrase_freq_exact_two(first: &[i32], second: &[i32], stop_at_first: bool) -> i32 {
    if !stop_at_first {
        return phrase_freq_exact_two_count(first, second);
    }
    let (mut i, mut j, mut freq) = (0, 0, 0);
    while let (Some(&p0), Some(&p1)) = (first.get(i), second.get(j)) {
        // The same `p0 + 1` the general loop aligns on.
        let target = p0 + 1;
        if p1 < target {
            j += 1;
            continue;
        }
        if p1 == target {
            freq += 1;
            if stop_at_first {
                break;
            }
        }
        // `j` stays: a multi-phrase position list is a union and may repeat
        // a position, and the next `p0` may equal this one.
        i += 1;
    }
    freq
}

/// [`phrase_freq_exact_two`]'s counting merge without a branch per step:
/// whether `second` or `first` moves on, and whether the step counts, are
/// both data, not control flow. A document holds a handful of positions per
/// term, and which list moves next is close to a coin flip, so the branchy
/// merge mispredicted most of its steps. The steps are the same: `j` moves
/// past a `second` position below `p0 + 1`; otherwise `i` moves, counting
/// when the position is exactly `p0 + 1` (and `j` stays, as there).
pub(super) fn phrase_freq_exact_two_count(first: &[i32], second: &[i32]) -> i32 {
    let (mut i, mut j, mut freq) = (0usize, 0usize, 0i32);
    while i < first.len() && j < second.len() {
        let target = first[i].wrapping_add(1);
        let p1 = second[j];
        let behind = p1 < target;
        freq += i32::from(p1 == target);
        j += usize::from(behind);
        i += usize::from(!behind);
    }
    freq
}

/// `ExactPhraseScorer`'s per-doc `phraseFreq`-equivalent: counts every valid base
/// position `p0` in `term_positions[0]` for which the rest of `term_positions`
/// align exactly (`term_positions[i]` contains `p0 + i` for every `i`) — the
/// same alignment condition [`phrase_matches_in_doc`] checks, except this counts
/// every satisfying `p0` instead of stopping at the first one.
///
/// **Why counting distinct `p0` values in the first term's own (already
/// deduplicated, strictly ascending) position list can't double-count the same
/// occurrence**: each `p0` is one real position of `term_positions[0]` in the
/// doc, and a real doc position occurs at most once in that list (positions are
/// decoded in strictly increasing order — see
/// [`lucene_codecs::postings::read_positions`]), so every counted match starts
/// at a genuinely distinct occurrence of the phrase's first word — this is
/// exactly `ExactPhraseScorer`'s own counting granularity: one match per
/// starting position of the phrase's first term that the rest of the phrase
/// aligns to. A repeated phrase (e.g. "the the" appearing twice, positions
/// 0,1,2,3 for "the") is counted once per valid starting position (0 and 2 for
/// non-overlapping repeats, or 0 *and* 1 if "the the the" — position 1 also
/// starts a valid "the the" alignment against positions 2 — matching real
/// Lucene's own per-start-position counting, which does not suppress
/// overlapping matches).
///
/// **Edge cases** (same contract as [`phrase_matches_in_doc`]): an empty
/// `term_positions`, or any single empty position list, both yield `0`. A
/// single-term phrase (`term_positions.len() == 1`) counts every occurrence of
/// that lone term (the inner alignment loop is empty, so every `p0` counts).
pub(crate) fn phrase_freq_exact(term_positions: &[&[i32]]) -> i32 {
    phrase_freq_exact_impl(term_positions, false)
}

/// One phrase-query term's live-filtered doc-ID list plus a `doc_id -> sorted
/// position list` map for that same term, or `None` when the field/term doesn't
/// exist (mirrors [`term_doc_ids`]'s "missing is not an error" convention). The map
/// (rather than a `Vec` aligned to the doc list) is what [`search_phrase_query`]
/// needs: after computing the doc-level conjunction across every term, it looks up
/// each candidate doc's position list per term by doc ID, not by index.
/// A term's live matching docs and, index-aligned with them, each doc's
/// positions.
///
/// Aligned vectors rather than a `HashMap<doc, positions>`: the phrase matcher
/// walks candidates in ascending doc order, so a per-term cursor index finds
/// each doc in amortised O(1) with no hashing and no per-doc clone. Building
/// the map cost one hash insertion per matching document -- ~5M for a
/// high-frequency term -- and looking a doc up then cloned its position vector.
///
/// Flat, not nested. The positions of every matching document live in one
/// `Vec<i32>`, and `spans[i]` is document `i`'s `(start, end)` into it. The
/// nested `Vec<Vec<i32>>` this replaced allocated once per matching document --
/// roughly five million allocations for a phrase query on a high-frequency term
/// -- and about half of such a query's runtime was in `malloc`/`free`/`memcpy`.
/// Lucene's `ExactPhraseMatcher` allocates nothing per document at all.
pub(crate) type TermDocPositions = (Vec<i32>, Vec<i32>, Vec<(u32, u32)>);

pub(super) fn term_doc_positions(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: &PosInput<'_>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    field: &str,
    term: &[u8],
) -> Result<Option<TermDocPositions>> {
    let Some(field_terms) = fields.field(field) else {
        return Ok(None);
    };
    let Some((postings, positions, doc_starts)) =
        field_terms.positions_flat(term, doc_in, pos_in, pay_in)?
    else {
        return Ok(None);
    };

    let mut docs = Vec::with_capacity(postings.docs.len());
    let mut spans: Vec<(u32, u32)> = Vec::with_capacity(postings.docs.len());
    for (i, doc_id) in postings.docs.into_iter().enumerate() {
        if !live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
            continue;
        }
        docs.push(doc_id);
        // `doc_starts` is indexed by the *unfiltered* document position and has
        // one extra entry, so the span is always in range and deleted documents
        // simply never get one pushed.
        spans.push((doc_starts[i], doc_starts[i + 1]));
    }
    Ok(Some((docs, positions, spans)))
}

/// `getPhraseMatcher`'s `fieldTerms.hasPositions() == false` check: a phrase
/// that `rewrite` leaves a phrase (two or more terms for `PhraseQuery`, two or
/// more positions for `MultiPhraseQuery`) over a field this segment indexed
/// without positions is Java's `IllegalStateException`, with the query as its
/// `toString()`. A segment without the field has no matcher, so no error;
/// the check comes before any term lookup, as Java's does.
pub(crate) fn require_phrase_positions(fields: &BlockTreeFields, q: &PhraseQuery) -> Result<()> {
    phrase_positions_check(field_index_options(fields, &q.field), q)
}

/// [`require_phrase_positions`] for `MultiPhraseQuery`.
pub(crate) fn require_multi_phrase_positions(
    fields: &BlockTreeFields,
    q: &query::MultiPhraseQuery,
) -> Result<()> {
    multi_phrase_positions_check(field_index_options(fields, &q.field), q)
}

/// [`require_phrase_positions`] given the field's index options in the leaf
/// (`None`: the leaf has no terms in it).
pub(crate) fn phrase_positions_check(
    options: Option<lucene_codecs::field_infos::IndexOptions>,
    q: &PhraseQuery,
) -> Result<()> {
    if q.terms.len() < 2 {
        return Ok(());
    }
    no_positions(options, &q.field, "PhraseQuery", || {
        Clause::Phrase(q.clone())
    })
}

/// [`phrase_positions_check`] for `MultiPhraseQuery`.
pub(crate) fn multi_phrase_positions_check(
    options: Option<lucene_codecs::field_infos::IndexOptions>,
    q: &query::MultiPhraseQuery,
) -> Result<()> {
    if q.term_arrays.len() < 2 {
        return Ok(());
    }
    no_positions(options, &q.field, "MultiPhraseQuery", || {
        Clause::MultiPhrase(q.clone())
    })
}

pub(super) fn field_index_options(
    fields: &BlockTreeFields,
    field: &str,
) -> Option<lucene_codecs::field_infos::IndexOptions> {
    fields.field(field).map(|ft| ft.index_options())
}

pub(super) fn no_positions(
    options: Option<lucene_codecs::field_infos::IndexOptions>,
    field: &str,
    name: &str,
    clause: impl FnOnce() -> Clause,
) -> Result<()> {
    match options {
        Some(o) if !o.subsumes_positions() => Err(Error::IllegalState(format!(
            "field \"{field}\" was indexed without position data; cannot run {name} (phrase={})",
            query_builder::to_query_string(Some(&clause()), "")
        ))),
        _ => Ok(()),
    }
}

/// Executes `query` (see [`query::PhraseQuery`] for `slop`'s exact semantics)
/// against one already-opened segment, feeding every matching **live** doc ID to
/// `collector` in ascending order -- same
/// parameter contract as [`search_term_query`], plus the segment's opened `.pos`/
/// `.pay` files (needed to check position alignment for a real, multi-term
/// phrase). Note `live_docs` sits *after* `pos_in`/`pay_in` here, unlike
/// [`search_term_query`]/[`search_boolean_query`]'s "`live_docs` right after
/// `doc_in`" ordering -- deliberate, to keep the two positions-file parameters
/// adjacent to each other and to `doc_in`.
///
/// - `pos_in`: the segment's opened `.pos` file. Required (an `Err(Error::
///   MissingPosInput)` otherwise) for any phrase with **more than one term** --
///   never touched for a single-term phrase, which degenerates to a plain
///   [`search_term_query`] call (see below). `None` is fine for that case.
/// - `pay_in`: the segment's opened `.pay` file, or `None` when the field has
///   neither offsets nor payloads, or its total occurrence count never spans a
///   full postings block -- same optionality contract as
///   [`lucene_codecs::blocktree::FieldTerms::positions`].
///
/// **Matching semantics**: a doc matches iff it contains every phrase term (a
/// pure doc-ID conjunction, computed first as a cheap pre-filter -- phrase match
/// implies term match, so this never does position work for a doc that couldn't
/// possibly qualify) *and* an alignment check finds a valid alignment for that
/// doc's per-term position lists: `query.slop == 0` uses
/// [`phrase_matches_in_doc`]'s exact-adjacency fast path (unchanged from before
/// `slop` existed), `query.slop > 0` runs
/// [`sloppy_phrase::sloppy_phrase_matches`], the ported `SloppyPhraseMatcher`
/// -- which admits terms appearing out of phrase order within the budget.
///
/// **Edge cases** (see `query::PhraseQuery`'s doc comment and this port's
/// `docs/parity.md` for the full accounting):
/// - **Empty `terms`**: matches nothing, mirroring real
///   `PhraseQuery.Builder.build()`'s `MatchNoDocsQuery` result for zero added
///   terms. Not an error.
/// - **Single term**: degenerates to [`search_term_query`] on a `TermQuery` for
///   that one term -- a length-1 phrase trivially "aligns" wherever the term
///   occurs, so there's no position work to do (also means a caller running only
///   single-term "phrase" queries never needs an opened `.pos` file at all).
/// - **A term missing from the field**: matches nothing, not an error -- same
///   convention as [`search_term_query`]/[`search_boolean_query`].
/// - **Duplicate terms** (e.g. "the the"): handled correctly by
///   [`phrase_matches_in_doc`] without special-casing -- see that function's doc
///   comment.
pub fn search_phrase_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
    collector: &mut C,
) -> Result<()> {
    require_phrase_positions(fields, query)?;
    if query.terms.is_empty() {
        return Ok(());
    }
    if query.terms.len() == 1 {
        let term_query = TermQuery::new(query.field.clone(), query.terms[0].clone());
        return search_term_query(fields, doc_in, live_docs, &term_query, collector);
    }
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };

    // Documents first, positions second -- the same order (and for the same
    // reason) as `search_phrase_query_scored_with_stats`: a phrase can only
    // match where every term does, so positions are needed only for the
    // intersection. Deliberately **unfiltered** by `live_docs` here, because
    // `positions_for_docs` indexes the wire stream by each document's running
    // frequency sum and needs the term's whole doc list; deletions are applied
    // to the candidate list below instead.
    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_freqs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            // A missing term means the phrase can never match -- same convention
            // as `term_doc_ids`/`search_term_query`.
            return Ok(());
        };
        per_term_docs.push(postings.docs);
        per_term_freqs.push(postings.freqs);
    }

    let per_term_docs_snapshot = per_term_docs.clone();
    let candidate_docs: Vec<i32> = Conjunction::new(
        per_term_docs
            .into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect(),
    )
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
    .collect();
    if candidate_docs.is_empty() {
        return Ok(());
    }

    // Each term's own indices for the candidates, ascending -- one walk per
    // term, no searching, because both lists are sorted.
    let mut per_term_positions: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_starts: Vec<Vec<u32>> = Vec::with_capacity(query.terms.len());
    for (t, term) in query.terms.iter().enumerate() {
        let docs = &per_term_docs_snapshot[t];
        let mut wanted = Vec::with_capacity(candidate_docs.len());
        let mut cursor = 0usize;
        for &doc_id in &candidate_docs {
            while cursor < docs.len() && docs[cursor] < doc_id {
                cursor += 1;
            }
            debug_assert!(
                cursor < docs.len() && docs[cursor] == doc_id,
                "candidates come from the intersection of these very lists"
            );
            wanted.push(cursor);
        }
        let stats = field_terms
            .try_seek_exact(term)?
            .expect("term presence was established above");
        let (positions, starts) = field_terms.positions_for_docs(
            term,
            doc_in,
            pos_in,
            pay_in,
            &per_term_freqs[t],
            stats.total_term_freq,
            &wanted,
        )?;
        per_term_positions.push(positions);
        per_term_starts.push(starts);
    }

    // Which slots hold the same term -- `SloppyPhraseMatcher`'s `rptGroups`,
    // which Java computes once per scorer and this port computes once per
    // query. Unused at `slop == 0`, where the exact matcher runs instead.
    let repeats = sloppy_phrase::PhraseRepeats::for_phrase(&query.terms);

    // Candidate `k` sits at index `k` in every term's positions, because
    // every term's `wanted` was built from the same candidate list.
    let mut term_positions: Vec<&[i32]> = Vec::with_capacity(per_term_positions.len());
    for (k, doc_id) in candidate_docs.into_iter().enumerate() {
        term_positions.clear();
        for t in 0..per_term_positions.len() {
            let start = per_term_starts[t][k] as usize;
            let end = per_term_starts[t][k + 1] as usize;
            term_positions.push(&per_term_positions[t][start..end]);
        }
        let is_match = if query.slop == 0 {
            phrase_matches_in_doc(&term_positions)
        } else {
            sloppy_phrase::sloppy_phrase_matches(&term_positions, &repeats, query.slop)
        };
        if is_match {
            collector.collect(doc_id);
        }
    }
    Ok(())
}

/// Scored sibling of [`search_phrase_query`]: same matching
/// semantics and parameter contract, but feeds each matched, live doc's BM25
/// score to a [`ScoringCollector`] instead of a plain [`Collector`].
///
/// **Formula, verified against real Lucene's `PhraseWeight`/`BM25Similarity`
/// source rather than guessed**: a multi-term phrase's `idf` is the *sum* of
/// each constituent term's own `idf(docFreq, docCount)` — this is
/// `BM25Similarity.idf(CollectionStatistics, TermStatistics[])`'s actual
/// behavior for a phrase's combined term statistics (it iterates every term
/// and sums each one's `idf`, then reports that sum as the phrase's overall
/// idf), not this port's invention. `tfNorm` is computed exactly like
/// [`term_doc_scores`]'s, except with the doc's **phrase frequency** in place
/// of a single term's `freq`:
/// - `query.slop == 0`: phrase frequency is [`phrase_freq_exact`]'s count of
///   valid alignments (`ExactPhraseScorer`'s real `phraseFreq` accumulation —
///   see that function's doc comment for the exact counting rule and why it
///   doesn't double-count).
/// - `query.slop > 0`: phrase frequency is
///   [`sloppy_phrase::sloppy_phrase_freq`], which is `PhraseScorer.score()`'s
///   own `freq = sloppyWeight(); while (nextMatch()) freq += sloppyWeight();`
///   over the ported `SloppyPhraseMatcher` walk -- a graduated
///   `1 / (1 + matchLength)` per match, summed in Lucene's own emission order
///   (`f32` addition is not associative, so the order is part of the answer).
///
/// `norms`/`collector`: same contract as [`search_term_query_scored`]'s.
#[allow(clippy::too_many_arguments)]
pub fn search_phrase_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    collector: &mut C,
) -> Result<()> {
    search_phrase_query_scored_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, query, norms, None, collector,
    )
}

/// [`search_phrase_query_scored`] taking reader-wide statistics, so a phrase
/// clause of a multi-segment search gets one idf per constituent term -- see
/// [`CollectionStats`]. `None` keeps this segment's own counters.
#[allow(clippy::too_many_arguments)]
pub fn search_phrase_query_scored_with_stats<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    require_phrase_positions(fields, query)?;
    if query.terms.is_empty() {
        return Ok(());
    }
    if query.terms.len() == 1 {
        let term_query = TermQuery::new(query.field.clone(), query.terms[0].clone());
        return search_term_query_scored_with_stats(
            fields,
            doc_in,
            live_docs,
            &term_query,
            norms,
            global,
            collector,
        );
    }
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };

    // Real BM25's phrase idf is the sum of every constituent term's own idf --
    // see this function's doc comment. A missing term means the phrase can
    // never match, same convention as `search_phrase_query`.
    // `BM25Similarity.idfExplain(TermStatistics[])` sums in a double and
    // casts once: an `f32` sum of three or more idfs can be an ulp off.
    let mut idf_acc = 0.0f64;
    let mut term_doc_freqs = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(stats) = field_terms.try_seek_exact(term)? else {
            return Ok(());
        };
        term_doc_freqs.push(stats.doc_freq);
        // Reader-wide statistics where the caller has them, exactly as the term
        // path does. A phrase's idf is the sum of its terms' idfs, so a
        // per-segment idf here is wrong in the same way and for the same
        // reason -- and it is what left two phrase queries disagreeing with
        // Java on the segmented corpus after every other query agreed.
        let (df, dc) = match global.and_then(|g| g.term(&query.field, term)) {
            Some(g) => (g.doc_freq, g.doc_count),
            None => (stats.doc_freq as i64, field_terms.doc_count as i64),
        };
        idf_acc += f64::from(similarity::idf(df, dc));
    }
    let idf_sum = idf_acc as f32;

    // `PhraseScorer` over lazy positions: a leapfrog over the terms' documents,
    // a `maxFreq` score check before any position is read, and positions
    // decoded only for the documents that survive it. See `phrase_scorer`.
    // Needs a real `.doc` stream for every term, so a pulsed singleton
    // (`docFreq == 1`, whose one posting lives in the term dictionary) keeps
    // the eager path below.
    if let Some(doc_in) = doc_in {
        if term_doc_freqs.iter().all(|&df| df > 1) {
            let mut cursors = Vec::with_capacity(query.terms.len());
            for (term, &df) in query.terms.iter().zip(&term_doc_freqs) {
                let Some(cursor) = field_terms.lazy_positions(term, doc_in, pos_in)? else {
                    return Ok(());
                };
                cursors.push((cursor, df as i64));
            }
            let repeats = sloppy_phrase::PhraseRepeats::for_phrase(&query.terms);
            return phrase_scorer::score_phrase(
                cursors,
                idf_sum,
                query.slop,
                &repeats,
                norms.map(|n| n.cursor()),
                live_docs,
                collector,
            );
        }
    }

    let matched = phrase_doc_freqs(field_terms, doc_in, pos_in, pay_in, live_docs, query)?;
    // One norms cursor for this scan; the matches ascend, so a sparse
    // field's `IndexedDISI` region is walked once, not once per document.
    let mut norms_cursor = norms.map(|n| n.cursor());
    for (doc_id, phrase_freq) in matched {
        let (field_length, avg_field_length) = match norms_cursor.as_mut() {
            Some(nc) => (nc.field_length(doc_id)?, nc.avg_field_length()),
            None => (
                similarity::UNNORMED_FIELD_LENGTH,
                similarity::UNNORMED_FIELD_LENGTH,
            ),
        };
        collector.collect(
            doc_id,
            similarity::do_score(
                idf_sum,
                phrase_freq,
                similarity::norm_inverse(
                    field_length,
                    avg_field_length,
                    similarity::DEFAULT_K1,
                    similarity::DEFAULT_B,
                ),
            ),
        );
    }
    Ok(())
}

/// The eager phrase path: every live document `query` matches in this
/// segment, ascending, with its phrase frequency (`ExactPhraseMatcher`'s
/// count, or `SloppyPhraseMatcher`'s summed weights). What a phrase whose
/// term is a pulsed singleton -- no `.doc` stream for a lazy cursor -- is
/// scored from, by BM25 below and by any other similarity in the scorer
/// tree. `field_terms` holds every one of the phrase's terms.
pub(crate) fn phrase_doc_freqs(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    pos_in: &PosInput<'_>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PhraseQuery,
) -> Result<Vec<(i32, f32)>> {
    // Documents first, positions second.
    //
    // A phrase can only match where every term does, so positions are needed
    // only for the intersection -- which for `phrase t0 t1` on the benchmark corpus is
    // 2.2M documents out of `t0`'s 5M. Fetching each term's positions up front
    // materialized every one of them, roughly 15 million for `t0` alone, to
    // look at less than half. Intersecting the (cheap) doc lists first and then
    // asking only for those documents' positions is what `ExactPhraseMatcher`
    // effectively does by advancing per candidate.
    // Deliberately **unfiltered** by `live_docs`: `positions_for_docs` indexes
    // the wire position stream by each document's running frequency sum and
    // rejects a frequency list whose total disagrees with the term's
    // `totalTermFreq`, so dropping deleted documents here makes every scored
    // phrase query on a segment with deletions fail outright
    // ("sum of per-doc freqs disagrees with total_term_freq"). Deletions are
    // applied to the candidate list below instead, which is where they belong:
    // a deleted document is not a hit, but it still occupies its slot in the
    // term's postings.
    let mut per_term_docs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_freqs: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    for term in &query.terms {
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            return Ok(Vec::new());
        };
        per_term_docs.push(postings.docs.clone());
        per_term_freqs.push(postings.freqs.clone());
    }

    let per_term_docs_snapshot = per_term_docs.clone();
    let candidate_docs: Vec<i32> = Conjunction::new(
        per_term_docs
            .into_iter()
            .map(|v| Box::new(v.into_iter()) as BoxDocIter<'static>)
            .collect(),
    )
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
    .collect();
    if candidate_docs.is_empty() {
        return Ok(Vec::new());
    }

    // Each term's own indices for the candidates, ascending -- one walk per
    // term, no searching, because both lists are sorted.
    let mut per_term_positions: Vec<Vec<i32>> = Vec::with_capacity(query.terms.len());
    let mut per_term_starts: Vec<Vec<u32>> = Vec::with_capacity(query.terms.len());
    for (t, term) in query.terms.iter().enumerate() {
        let docs = &per_term_docs_snapshot[t];
        let mut wanted = Vec::with_capacity(candidate_docs.len());
        let mut cursor = 0usize;
        for &doc_id in &candidate_docs {
            while cursor < docs.len() && docs[cursor] < doc_id {
                cursor += 1;
            }
            debug_assert!(
                cursor < docs.len() && docs[cursor] == doc_id,
                "candidates come from the intersection of these very lists"
            );
            wanted.push(cursor);
        }
        let stats = field_terms
            .try_seek_exact(term)?
            .expect("term presence was established above");
        let (positions, starts) = field_terms.positions_for_docs(
            term,
            doc_in,
            pos_in,
            pay_in,
            &per_term_freqs[t],
            stats.total_term_freq,
            &wanted,
        )?;
        per_term_positions.push(positions);
        per_term_starts.push(starts);
    }

    // One cursor per term, advanced in step with the ascending candidate order,
    // so each doc's positions are found without hashing and without cloning.
    // Candidate `k` sits at index `k` in every term's positions, because
    // `wanted` was built from the same candidate list for each term.
    let mut term_positions: Vec<&[i32]> = Vec::with_capacity(per_term_positions.len());
    // `SloppyPhraseMatcher`'s `rptGroups`, computed once per query -- see
    // `search_phrase_query`. Unused at `slop == 0`.
    let repeats = sloppy_phrase::PhraseRepeats::for_phrase(&query.terms);
    let mut out = Vec::with_capacity(candidate_docs.len());
    for (k, &doc_id) in candidate_docs.iter().enumerate() {
        term_positions.clear();
        for t in 0..per_term_positions.len() {
            let start = per_term_starts[t][k] as usize;
            let end = per_term_starts[t][k + 1] as usize;
            term_positions.push(&per_term_positions[t][start..end]);
        }
        // `PhraseScorer.score()`: the frequency fed to the similarity is
        // `ExactPhraseMatcher`'s match count for `slop == 0`, and
        // `SloppyPhraseMatcher`'s summed `1/(1+matchLength)` otherwise -- not
        // a flat `1` for "matched". See `sloppy_phrase::sloppy_phrase_freq`.
        let phrase_freq = if query.slop == 0 {
            phrase_freq_exact(&term_positions) as f32
        } else {
            sloppy_phrase::sloppy_phrase_freq(&term_positions, &repeats, query.slop)
        };
        if phrase_freq != 0.0 {
            out.push((doc_id, phrase_freq));
        }
    }
    Ok(out)
}

/// The documents containing at least one of `terms`, ascending -- one
/// position's `UnionFullPostingsEnum` doc stream. Absent terms contribute
/// nothing; an entirely absent set yields an empty list, which makes the whole
/// phrase match nothing.
///
/// **Not** filtered by `live_docs`: the caller applies deletions to the
/// candidate list, for the same reason `search_phrase_query_scored_with_stats`
/// does (`positions_for_docs` indexes the wire stream by a running frequency
/// sum that must total the term's `totalTermFreq`).
pub(crate) fn multi_phrase_slot_docs(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    terms: &[Vec<u8>],
) -> Result<Vec<i32>> {
    let mut merged = DocIdBitSet::default();
    for term in terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            merged.set(doc_id);
        }
    }
    Ok(merged.into_sorted_vec())
}

/// One position's merged, sorted, deduplicated positions for each document in
/// `candidates` (which must be ascending). Returns `(positions, starts)` with
/// `starts.len() == candidates.len() + 1`, the same flat shape
/// `FieldTerms::positions_for_docs` returns for a single term.
pub(crate) fn multi_phrase_slot_positions(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    pos_in: &PosInput<'_>,
    pay_in: Option<&PayInput<'_>>,
    terms: &[Vec<u8>],
    candidates: &[i32],
) -> Result<(Vec<i32>, Vec<u32>)> {
    let mut per_doc: Vec<Vec<i32>> = vec![Vec::new(); candidates.len()];
    for term in terms {
        let Some(stats) = field_terms.try_seek_exact(term)? else {
            continue;
        };
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            continue;
        };
        // `positions_for_docs` wants indices into the term's own **unfiltered**
        // doc sequence, and a frequency list totalling its `totalTermFreq` --
        // see `multi_phrase_slot_docs` on why deletions are not applied here.
        let term_docs = &postings.docs;
        let freqs = &postings.freqs;
        let mut wanted = Vec::new();
        let mut wanted_slots = Vec::new();
        let mut cursor = 0usize;
        for (slot, &doc_id) in candidates.iter().enumerate() {
            while cursor < term_docs.len() && term_docs[cursor] < doc_id {
                cursor += 1;
            }
            if cursor < term_docs.len() && term_docs[cursor] == doc_id {
                wanted.push(cursor);
                wanted_slots.push(slot);
            }
        }
        if wanted.is_empty() {
            continue;
        }
        let (positions, starts) = field_terms.positions_for_docs(
            term,
            doc_in,
            pos_in,
            pay_in,
            freqs,
            stats.total_term_freq,
            &wanted,
        )?;
        for (k, &slot) in wanted_slots.iter().enumerate() {
            let from = starts[k] as usize;
            let to = starts[k + 1] as usize;
            per_doc[slot].extend_from_slice(&positions[from..to]);
        }
    }

    let mut flat = Vec::new();
    let mut starts = Vec::with_capacity(candidates.len() + 1);
    starts.push(0u32);
    for doc_positions in &mut per_doc {
        // Sorted, and **not** deduplicated. `UnionPostingsEnum.freq()` builds
        // its `PositionsQueue` by draining every sub's positions and calling
        // `sort()` -- there is no dedup step, so a position reached by two
        // alternatives (the same term listed twice, or two terms the analyzer
        // stacked at one position) is yielded twice and the matcher counts the
        // alignment twice. Deduplicating here looks obviously right and is
        // wrong: real Lucene scores `pos:"(alpha alpha) beta"` at 0.87906057 on
        // this crate's fixture where the deduplicated form gives 0.6393168.
        // Recorded as `scoring.multiphrase.dup` and pinned by
        // `multi_phrase_query_scores_match_real_lucene_bit_for_bit`.
        doc_positions.sort_unstable();
        flat.extend_from_slice(doc_positions);
        starts.push(flat.len() as u32);
    }
    Ok((flat, starts))
}

/// [`search_phrase_query`]'s sibling for [`query::MultiPhraseQuery`]: matching
/// only, no scores. See that struct's doc comment for the exact semantics this
/// implements (and which parts of real `MultiPhraseQuery` are out of scope).
pub fn search_multi_phrase_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    collector: &mut C,
) -> Result<()> {
    for doc_id in multi_phrase_doc_ids(fields, doc_in, pos_in, pay_in, live_docs, query)? {
        collector.collect(doc_id);
    }
    Ok(())
}

/// The matching-only half of [`search_multi_phrase_query`], shared with
/// [`resolve_clause_docs`].
pub(super) fn multi_phrase_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
) -> Result<Vec<i32>> {
    let mut docs = Vec::new();
    struct Collect<'a>(&'a mut Vec<i32>);
    impl collector::ScoringCollector for Collect<'_> {
        fn collect(&mut self, doc_id: i32, _score: f32) {
            self.0.push(doc_id);
        }
    }
    let mut sink = Collect(&mut docs);
    multi_phrase_hits(
        fields, doc_in, pos_in, pay_in, live_docs, query, None, None, &mut sink,
    )?;
    Ok(docs)
}

/// Scored [`query::MultiPhraseQuery`] execution -- see that struct's doc
/// comment for the idf-over-every-term and merged-frequency rules taken from
/// `MultiPhraseWeight`.
#[allow(clippy::too_many_arguments)]
pub fn search_multi_phrase_query_scored<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    collector: &mut C,
) -> Result<()> {
    search_multi_phrase_query_scored_with_stats(
        fields, doc_in, pos_in, pay_in, live_docs, query, norms, None, collector,
    )
}

/// [`search_multi_phrase_query_scored`] taking reader-wide statistics -- see
/// [`CollectionStats`].
#[allow(clippy::too_many_arguments)]
pub fn search_multi_phrase_query_scored_with_stats<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    multi_phrase_hits(
        fields, doc_in, pos_in, pay_in, live_docs, query, norms, global, collector,
    )
}

/// The one implementation both [`search_multi_phrase_query`] and
/// [`search_multi_phrase_query_scored_with_stats`] run, so matching and scoring
/// can never disagree about which documents match .
#[allow(clippy::too_many_arguments)]
pub(super) fn multi_phrase_hits<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<()> {
    require_multi_phrase_positions(fields, query)?;
    // `MultiPhraseQuery.rewrite`: an empty `termArrays` is a
    // `MatchNoDocsQuery`.
    if query.term_arrays.is_empty() {
        return Ok(());
    }
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };

    // `MultiPhraseWeight`: the idf is summed over every term of every
    // position, and a term absent from this segment contributes nothing.
    // `BM25Similarity.idfExplain(TermStatistics[])` sums in a double and
    // casts once: an `f32` sum of three or more idfs can be an ulp off.
    let mut idf_acc = 0.0f64;
    let mut any_term_present = false;
    for alternatives in &query.term_arrays {
        for term in alternatives {
            let Some(stats) = field_terms.try_seek_exact(term)? else {
                continue;
            };
            any_term_present = true;
            let (df, dc) = match global.and_then(|g| g.term(&query.field, term)) {
                Some(g) => (g.doc_freq, g.doc_count),
                None => (stats.doc_freq as i64, field_terms.doc_count as i64),
            };
            idf_acc += f64::from(similarity::idf(df, dc));
        }
    }
    let idf_sum = idf_acc as f32;
    if !any_term_present {
        return Ok(());
    }

    // `MultiPhraseQuery.rewrite`'s "optimize one-term case": a single position
    // rewrites to a `BooleanQuery` of `SHOULD` `TermQuery`s. That is **not**
    // the same thing as a one-slot phrase over the merged union -- a
    // `BooleanQuery` scores each term with its *own* idf and its *own*
    // frequency and adds them, where the phrase path would use the summed idf
    // once against the merged frequency. Real Lucene's recorded scores for
    // this shape (`scoring.multiphrase.single` in the blocktree fixture's
    // manifest) are the boolean ones, which is what caught the difference.
    if query.term_arrays.len() == 1 {
        let mut summed: HashMap<i32, f32> = HashMap::new();
        for term in &query.term_arrays[0] {
            let term_query = TermQuery::new(query.field.clone(), term.clone());
            for (doc_id, score) in
                term_doc_scores(fields, doc_in, live_docs, &term_query, norms, global)?
            {
                *summed.entry(doc_id).or_insert(0.0) += score;
            }
        }
        let mut hits: Vec<(i32, f32)> = summed.into_iter().collect();
        hits.sort_unstable_by_key(|&(doc_id, _)| doc_id);
        for (doc_id, score) in hits {
            collector.collect(doc_id, score);
        }
        return Ok(());
    }

    // The matches ascend, so one norms cursor covers the whole scan.
    let mut norms_cursor = norms.map(|n| n.cursor());
    multi_phrase_freqs(
        field_terms,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        query,
        |doc_id, freq| {
            let norm_inverse = match norms_cursor.as_mut() {
                Some(nc) => nc.norm_inverse(doc_id)?,
                None => similarity::UNNORMED_NORM_INVERSE,
            };
            collector.collect(doc_id, similarity::do_score(idf_sum, freq, norm_inverse));
            Ok(())
        },
    )
}

/// `doc`'s frequency for a `MultiPhraseQuery` of two or more positions
/// (`PhraseMatcher`'s, what its weight's `SimScorer` reads), `None` when the
/// phrase does not match it: what an explanation without scores prints.
#[allow(clippy::too_many_arguments)]
pub(crate) fn multi_phrase_freq(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    doc: i32,
) -> Result<Option<f32>> {
    require_multi_phrase_positions(fields, query)?;
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(None);
    };
    let mut found = None;
    multi_phrase_freqs(
        field_terms,
        doc_in,
        pos_in,
        pay_in,
        live_docs,
        query,
        |doc_id, freq| {
            if doc_id == doc {
                found = Some(freq);
            }
            Ok(())
        },
    )?;
    Ok(found)
}

/// Every live document a `MultiPhraseQuery` of two or more positions
/// matches, ascending, with its phrase frequency: the matching half of
/// [`multi_phrase_hits`].
pub(super) fn multi_phrase_freqs(
    field_terms: &blocktree::FieldTerms,
    doc_in: Option<&DocInput<'_>>,
    pos_in: Option<&PosInput<'_>>,
    pay_in: Option<&PayInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &query::MultiPhraseQuery,
    mut on_match: impl FnMut(i32, f32) -> Result<()>,
) -> Result<()> {
    let Some(pos_in) = pos_in else {
        return Err(Error::MissingPosInput);
    };

    // One merged doc stream per position (`UnionFullPostingsEnum`'s doc side).
    let slot_docs: Vec<Vec<i32>> = query
        .term_arrays
        .iter()
        .map(|alts| multi_phrase_slot_docs(field_terms, doc_in, alts))
        .collect::<Result<Vec<_>>>()?;
    if slot_docs.iter().any(|docs| docs.is_empty()) {
        return Ok(());
    }

    // Documents first, positions second -- the same order `ExactPhraseMatcher`
    // effectively works in, and the reason the phrase path stopped decoding
    // every position of every document (finding #O15).
    let slot_count = slot_docs.len();
    let candidates: Vec<i32> = Conjunction::new(
        slot_docs
            .into_iter()
            .map(|docs| Box::new(docs.into_iter()) as BoxDocIter<'static>)
            .collect(),
    )
    .filter(|&doc_id| live_docs.is_none_or(|bits| bits.get_doc(doc_id)))
    .collect();
    if candidates.is_empty() {
        return Ok(());
    }

    let mut per_slot_positions = Vec::with_capacity(slot_count);
    let mut per_slot_starts = Vec::with_capacity(slot_count);
    for alternatives in &query.term_arrays {
        let (positions, starts) = multi_phrase_slot_positions(
            field_terms,
            doc_in,
            pos_in,
            pay_in,
            alternatives,
            &candidates,
        )?;
        per_slot_positions.push(positions);
        per_slot_starts.push(starts);
    }

    // See `search_phrase_query`'s own binding: `MultiPhraseQuery` slots hold a
    // *set* of terms each, which is the `hasMultiTermRpts` case of Java's
    // repeat detection.
    let repeats = sloppy_phrase::PhraseRepeats::for_multi_phrase(&query.term_arrays);

    let mut slot_positions: Vec<&[i32]> = Vec::with_capacity(slot_count);
    for (k, &doc_id) in candidates.iter().enumerate() {
        slot_positions.clear();
        for t in 0..per_slot_positions.len() {
            let from = per_slot_starts[t][k] as usize;
            let to = per_slot_starts[t][k + 1] as usize;
            slot_positions.push(&per_slot_positions[t][from..to]);
        }
        let freq = if query.slop == 0 {
            phrase_freq_exact(&slot_positions) as f32
        } else {
            sloppy_phrase::sloppy_phrase_freq(&slot_positions, &repeats, query.slop)
        };
        if freq == 0.0 {
            continue;
        }
        on_match(doc_id, freq)?;
    }
    Ok(())
}
