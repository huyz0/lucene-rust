//! Doc-id expansion of the multi-term queries: prefix, wildcard, fuzzy, regexp and term-in-set.

use super::*;

/// [`Clause::Prefix`]'s matched doc-ID list: same union-across-matching-terms
/// mechanism as [`wildcard_doc_ids`], built on
/// [`lucene_codecs::wildcard::WildcardPattern::prefix`] (a literal-bytes-only
/// pattern -- see [`PrefixQuery`]'s doc comment for why this avoids
/// `WildcardPattern::new`'s glob-escaping entirely) instead of a general glob
/// pattern. Returns an empty `Vec` -- not an error -- when `query.field`
/// doesn't exist in this segment, same "missing field means no matches"
/// convention every other clause follows.
pub(super) fn prefix_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &PrefixQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let pattern = WildcardPattern::prefix(&query.prefix);
    let matching_terms: Vec<Vec<u8>> = field_terms
        .intersect(&pattern)
        .map(|r| r.map(|(term, _stats)| term))
        .collect::<lucene_codecs::blocktree::Result<_>>()?;
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}

/// [`Clause::Wildcard`]'s matched doc-ID list: every term
/// [`lucene_codecs::blocktree::FieldTerms::intersect`] finds matching `query`'s
/// compiled pattern (for `query.field`) contributes its own postings' doc IDs,
/// **union**ed across every matching term (real `WildcardQuery`'s
/// `MultiTermQuery` matching contract -- a doc matches if *any* accepted term
/// occurs in it) and deduplicated (a doc can hold more than one term the
/// pattern accepts in a multi-valued field), then filtered by `live_docs` same
/// as [`term_doc_ids`]. Returns an empty `Vec` -- not an error -- when
/// `query.field` doesn't exist in this segment, matching every other clause's
/// "missing field means no matches" convention.
pub(super) fn wildcard_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &WildcardQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let pattern = WildcardPattern::new(&query.pattern);
    let matching_terms: Vec<Vec<u8>> = field_terms
        .intersect(&pattern)
        .map(|r| r.map(|(term, _stats)| term))
        .collect::<lucene_codecs::blocktree::Result<_>>()?;
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}

/// One term a [`FuzzyQuery`] expanded to, with the per-term boost real
/// Lucene's `FuzzyTermsEnum` publishes through its `BoostAttribute` and the
/// term's own document frequency.
#[derive(Debug, Clone)]
pub(super) struct ExpandedTerm {
    pub(super) term: Vec<u8>,
    /// `FuzzyTermsEnum`'s similarity, **not** clamped -- `TopTermsRewrite
    /// .build` does the `Math.max(0.0f, st.boost)` truncation when it turns
    /// the queue into clauses, and [`fuzzy_doc_scores`] does it there too.
    pub(super) boost: f32,
    pub(super) doc_freq: i64,
}

/// The result of expanding a [`FuzzyQuery`] against one segment's field: the
/// selected terms plus the **blended** document frequency every one of them
/// is scored with.
#[derive(Debug, Clone, Default)]
pub(super) struct FuzzyExpansion {
    pub(super) terms: Vec<ExpandedTerm>,
    /// `CollectionStatistics.docCount` for the field over whatever scope the
    /// expansion was run on -- one segment for the single-segment path, the
    /// whole reader for [`multi_segment::global_fuzzy_stats`]'s. It travels
    /// with the blended frequency because the two are one `TermStatistics`/
    /// `CollectionStatistics` pair in Java and separating them is how the
    /// per-leaf/reader-wide mismatch got in.
    pub(super) doc_count: i64,
    /// `BlendedTermQuery.rewrite`: `df = Math.max(df, ctx.docFreq())` across
    /// every selected term, then every term's `TermStates` is rebuilt with
    /// that artificial frequency. Blending is the whole point of
    /// `TopTermsBlendedFreqScoringRewrite`: without it "the rarest term
    /// typically ranks highest (often not useful eg in the set of expanded
    /// terms in a FuzzyQuery)".
    pub(super) blended_doc_freq: i64,
}

/// Expands `query` over `field_terms` exactly the way
/// `MultiTermQuery.TopTermsBlendedFreqScoringRewrite` does.
///
/// Four Lucene behaviours live here (as opposed to keeping the first
/// `max_expansions` terms in term-dictionary order and scoring every match a
/// flat `1.0`):
///
/// 1. **Per-term boost.** `FuzzyTermsEnum.next` sets `BoostAttribute` to
///    `1.0` for an exact match and `1 - ed/min(len(term), len(query term))`
///    otherwise -- see [`lucene_codecs::fuzzy::FuzzyMatch::boost`].
/// 2. **Top-N by boost, not by term order.** `TopTermsRewrite.collect` keeps
///    a size-`maxExpansions` priority queue whose worst element is the lowest
///    boost, with ties broken so the lexicographically **later** term is
///    dropped first (`boost == t.boost && bytes.compareTo(t.bytes.get()) > 0`
///    skips the candidate). [`fuzzy_expanded_terms_pruned`] keeps that queue
///    as a sorted `Vec`, in the final `(boost desc, bytes asc)` order.
/// 3. **Blended document frequency**, see [`FuzzyExpansion::blended_doc_freq`].
/// 4. **The `MaxNonCompetitiveBoostAttribute` feedback loop**: the worst
///    boost still in that queue is fed *back* into the term enumeration, which
///    drops its edit budget as soon as no term at the higher distance could
///    still compete (`FuzzyTermsEnum.bottomChanged`). See
///    [`lucene_codecs::blocktree::FuzzyIntersect::set_max_edits`].
///
/// The whole matching set is still *visited* -- `TopTermsRewrite.rewrite` runs
/// `collectTerms` over every term the automaton accepts -- but from the point
/// the queue fills, each remaining term is tested against a narrower band.
pub(super) fn fuzzy_expanded_terms(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
) -> Result<FuzzyExpansion> {
    fuzzy_expanded_terms_pruned(field_terms, query, true)
}

/// [`fuzzy_expanded_terms`] with the feedback loop switchable, so
/// `examples/fuzzy_pruning.rs` can time both arms in one process from one
/// build. Production always passes `true`; `false` is the pre-c40 behaviour
/// (`FuzzyTermsEnum` with `MaxNonCompetitiveBoostAttribute` never consulted),
/// and it must produce the **same expansion** -- which
/// `pruning_never_changes_the_selected_expansion` asserts.
pub(super) fn fuzzy_expanded_terms_pruned(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
    prune: bool,
) -> Result<FuzzyExpansion> {
    Ok(fuzzy_expanded_terms_instrumented(field_terms, query, prune)?.0)
}

/// [`fuzzy_expanded_terms_pruned`] also reporting the edit budget the walk
/// ended on -- `FuzzyTermsEnum.maxEdits` after the last `bottomChanged`. Only
/// [`bench_only_fuzzy_expansion`] wants it; the search path takes the tuple's
/// first half and drops it.
pub(super) fn fuzzy_expanded_terms_instrumented(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
    prune: bool,
) -> Result<(FuzzyExpansion, u8)> {
    let pattern = FuzzyMatch::new(
        &query.term,
        query.max_edits,
        query.prefix_length,
        query.transpositions,
    );
    let mut collector = FuzzyTopTerms::new(query, &pattern);
    let final_max_edits = collector.collect_segment(field_terms, &pattern, prune)?;
    Ok((
        collector.finish(field_terms.doc_count as i64),
        final_max_edits,
    ))
}

/// The terms [`fuzzy_expanded_terms`] selects, over a one-document leaf
/// whose terms are held in memory (`MemoryIndex`'s, each with `docFreq` 1),
/// in the queue's order.
pub(crate) fn fuzzy_expanded_terms_in_memory<'t>(
    terms: impl Iterator<Item = &'t [u8]>,
    query: &FuzzyQuery,
) -> Vec<Vec<u8>> {
    let pattern = FuzzyMatch::new(
        &query.term,
        query.max_edits,
        query.prefix_length,
        query.transpositions,
    );
    let mut collector = FuzzyTopTerms::new(query, &pattern);
    collector.collect_sorted(terms.map(|t| (t, 1)), &pattern);
    collector.queue.into_iter().map(|e| e.term).collect()
}

/// `TopTermsRewrite.rewrite`'s `stQueue` + `visitedTerms` + the shared
/// `MaxNonCompetitiveBoostAttribute`, as one object that
/// `TermCollectingRewrite.collectTerms` drives **once per leaf**.
///
/// Java's collector is created once and fed every leaf's `TermsEnum` in turn
/// (`for (LeafReaderContext context : topReaderContext.leaves())`), so the
/// queue, the visited-term map and the published max-non-competitive boost are
/// all reader-wide. That is what makes `BlendedTermQuery`'s `df` a reader-wide
/// number, and it is why this type exists rather than a per-segment loop:
/// a fuzzy clause scored from one segment's own `docFreq`/`docCount` is the
/// same defect [`CollectionStats`] exists to fix for `TermQuery`.
pub(super) struct FuzzyTopTerms {
    /// `Math.min(size, getMaxSize())`.
    pub(super) max_size: usize,
    /// `FuzzyAutomatonBuilder.getTermLength()`: the whole query term's
    /// codepoint count, which is `bottomChanged`'s denominator.
    pub(super) term_length: usize,
    /// The query's own `maxEdits`, which every leaf's `FuzzyTermsEnum` starts
    /// from (`this.maxEdits = this.automata.length - 1` in its constructor).
    pub(super) query_max_edits: u8,
    /// `stQueue`, as a sorted `Vec` of at most `max_size` entries in the final
    /// order (boost descending, then term bytes ascending) rather than a
    /// binary heap keyed on its inverse. Java's `ScoreTerm.compareTo` -- "if
    /// the boosts are equal, `other.bytes.compareTo(this.bytes)`" -- makes
    /// `peek()` the lowest boost and, among equal boosts, the
    /// lexicographically *largest* term; that is exactly the last element
    /// here. A phrase-sized `maxExpansions` (50 by default) makes the sorted
    /// insert cheaper than a heap, and it removes the final sort.
    pub(super) queue: Vec<ExpandedTerm>,
    /// `MaxNonCompetitiveBoostAttribute`'s two values, published on the
    /// collector's shared `AttributeSource` and therefore surviving from one
    /// leaf's enum to the next. `None` is Java's
    /// `(Float.NEGATIVE_INFINITY, null)` initial state.
    pub(super) published_bottom: Option<(f32, Vec<u8>)>,
}

impl FuzzyTopTerms {
    pub(super) fn new(query: &FuzzyQuery, pattern: &FuzzyMatch<'_>) -> Self {
        FuzzyTopTerms {
            max_size: query.max_expansions,
            term_length: pattern.term_length(),
            query_max_edits: query.max_edits,
            queue: Vec::new(),
            published_bottom: None,
        }
    }

    /// `FuzzyTermsEnum.bottomChanged`'s `while` loop against the currently
    /// published bottom, returning the tightened budget.
    ///
    /// `term_after` is Java's
    /// `bottomTerm == null || (lastTerm != null && lastTerm.compareTo(bottomTerm) >= 0)`.
    pub(super) fn tighten(&self, mut edits: u8, last_term: Option<&[u8]>) -> u8 {
        let (bottom, bottom_term) = match &self.published_bottom {
            // No bottom published yet: `bottom` is `Float.NEGATIVE_INFINITY`,
            // which is below every finite `maxBoost`, so the loop breaks at
            // once and nothing is tightened.
            None => return edits,
            Some((boost, term)) => (*boost, term.as_slice()),
        };
        let term_after = last_term.is_some_and(|t| t >= bottom_term);
        while edits > 0 {
            // `1.0f - ((float) maxEdits / (float) termLength)`. A zero-length
            // query term makes this `-inf`, which never breaks the loop -- the
            // same answer Java's float division gives.
            let max_boost = 1.0f32 - f32::from(edits) / (self.term_length as f32);
            if bottom < max_boost || (bottom == max_boost && !term_after) {
                break;
            }
            edits = edits.saturating_sub(1);
        }
        edits
    }

    /// `collectTerms`' body for one accepted term with its `boost` and
    /// `docFreq`: whether it was inserted into the queue (and so may have
    /// moved its bottom).
    pub(super) fn offer(&mut self, term: &[u8], boost: f32, doc_freq: i64) -> bool {
        // `collectTerms`' own rejection, *before* the visited-term lookup:
        // once the queue is full, a term that cannot beat the worst entry
        // is dropped without being inserted -- and, in a second leaf,
        // without its `docFreq` being accumulated either.
        // `maxSize == 0` makes the queue full from the start with no
        // bottom to compare against, so nothing is ever competitive --
        // which is the same "select nothing" a truncate-to-zero gives.
        if self.queue.len() == self.max_size {
            match self.queue.last() {
                None => return false,
                Some(bottom) => {
                    if boost < bottom.boost
                        || (boost == bottom.boost && term > bottom.term.as_slice())
                    {
                        return false;
                    }
                }
            }
        }

        // `ScoreTerm t = visitedTerms.get(bytes)`: a term already in the
        // queue only has its `docFreq` accumulated
        // (`t.termState.register(state, ord, docFreq, totalTermFreq)`),
        // and nothing is republished. Only reachable across leaves --
        // within one leaf the term dictionary yields each term once.
        //
        // The queue is sorted by `(boost desc, term asc)` and a term's
        // boost is a function of the term alone, so an entry for it can
        // only sit where it would be inserted: the binary search for the
        // insertion point answers the lookup too, where a scan compared
        // the term against every entry.
        let at = self
            .queue
            .partition_point(|e| e.boost > boost || (e.boost == boost && e.term.as_slice() < term));
        if let Some(entry) = self
            .queue
            .get_mut(at)
            .filter(|e| e.boost == boost && e.term == term)
        {
            entry.doc_freq = entry.doc_freq.saturating_add(doc_freq);
            return false;
        }
        self.queue.insert(
            at,
            ExpandedTerm {
                boost,
                term: term.to_vec(),
                doc_freq,
            },
        );
        // `if (stQueue.size() > maxSize) { st = stQueue.poll(); visitedTerms
        // .remove(st.bytes.get()); st.termState.clear(); }` -- an evicted
        // term loses the `docFreq` earlier leaves contributed, which is
        // Java's behaviour and not an oversight here.
        if self.queue.len() > self.max_size {
            self.queue.pop();
        }
        true
    }

    /// [`Self::collect_segment`] over a leaf whose terms are held in memory
    /// (`MemoryIndex`'s): each of `terms`, in term order with its `docFreq`,
    /// that the pattern accepts within the query's `maxEdits`. Without the
    /// feedback loop, which only prunes the walk and never changes the
    /// selected expansion (`pruning_never_changes_the_selected_expansion`).
    pub(super) fn collect_sorted<'t>(
        &mut self,
        terms: impl Iterator<Item = (&'t [u8], i64)>,
        pattern: &FuzzyMatch<'_>,
    ) {
        for (term, doc_freq) in terms {
            if let Some(ed) = pattern.edits_within(term, pattern.max_edits()) {
                self.offer(term, pattern.boost_from_edits(term, ed), doc_freq);
            }
        }
    }

    /// One leaf: `collectTerms`' inner `while ((bytes = termsEnum.next()) != null)`
    /// loop over `getTermsEnum(query, terms, collector.attributes)`.
    ///
    /// Returns the edit budget the leaf's walk ended on -- `FuzzyTermsEnum
    /// .maxEdits` after its last `bottomChanged`, which is the only
    /// *observable* of the feedback loop itself.
    pub(super) fn collect_segment(
        &mut self,
        field_terms: &lucene_codecs::blocktree::FieldTerms,
        pattern: &FuzzyMatch<'_>,
        prune: bool,
    ) -> Result<u8> {
        // `FuzzyTermsEnum`'s constructor: a *fresh* enum per leaf, starting
        // from the query's own `maxEdits`, then immediately
        // `bottom = maxBoostAtt.getMaxNonCompetitiveBoost(); bottomChanged(null)`
        // against the boost an earlier leaf published. `lastTerm` is null
        // there, so `termAfter` is `bottomTerm == null`, i.e. false whenever a
        // bottom exists.
        let mut effective_max_edits = self.query_max_edits;
        let mut walk = field_terms.fuzzy_intersect(pattern);
        if prune {
            let tightened = self.tighten(effective_max_edits, None);
            if tightened != effective_max_edits {
                effective_max_edits = tightened;
                walk.set_max_edits(tightened);
            }
        }
        while let Some(next) = walk.next() {
            let (term, stats) = next?;
            // `FuzzyTermsEnum.next` sets `boostAtt` from the `ed` it has
            // already computed; the walk carries the same distance out so this
            // does not run the banded DP a second time per accepted term.
            let boost = pattern.boost_from_edits(&term, walk.last_edits());

            if !self.offer(&term, boost, stats.doc_freq as i64) {
                continue;
            }

            // `maxBoostAtt.setMaxNonCompetitiveBoost(peek().boost)` +
            // `setCompetitiveTerm(peek().bytes)`, then `FuzzyTermsEnum.next`
            // noticing they changed and running `bottomChanged(lastTerm)`.
            //
            // Two details of *when* Java does this, both reproduced. It
            // publishes only from the insert branch -- `collectTerms` returns
            // before the publish block both when the term was not competitive
            // and when it was already visited -- and `next()` acts only when
            // the published pair actually *changed*, which is the same
            // condition, since an insert into a full queue always moves the
            // bottom. Without that gate this loop would re-run on later,
            // non-competitive terms and could decrement once more than Java
            // does (`termAfter` flips to true as the walk passes the bottom
            // term). That extra step is still result-preserving -- past the
            // bottom term no equal-boost term can displace it -- but it is not
            // what Java does, and "prunes a bit more than Lucene" is not a
            // divergence worth having.
            //
            // Java also delays the swap to the *next* `next()` call so the
            // current term's `docFreq()` stays valid; here the term has already
            // been consumed, so "after this term" is the same point.
            if self.queue.len() == self.max_size {
                let bottom = self.queue.last().expect("non-empty: len == max_size > 0");
                self.published_bottom = Some((bottom.boost, bottom.term.clone()));
                if prune && effective_max_edits > 0 {
                    let tightened = self.tighten(effective_max_edits, Some(&term));
                    if tightened != effective_max_edits {
                        effective_max_edits = tightened;
                        walk.set_max_edits(tightened);
                    }
                }
            }
        }
        Ok(effective_max_edits)
    }

    /// `BlendedTermQuery.rewrite`'s fold: `df = Math.max(df, ctx.docFreq())`
    /// over every selected term, with `doc_count` the matching
    /// `CollectionStatistics.docCount`.
    pub(super) fn finish(self, doc_count: i64) -> FuzzyExpansion {
        let blended_doc_freq = self.queue.iter().map(|t| t.doc_freq).max().unwrap_or(0);
        FuzzyExpansion {
            terms: self.queue,
            blended_doc_freq,
            doc_count,
        }
    }
}

/// `TopTermsRewrite.rewrite` run over **every leaf**, which is what
/// [`multi_segment::global_fuzzy_stats`] needs and what
/// [`fuzzy_expanded_terms`] deliberately is not.
///
/// `field_terms` is one entry per segment that has the field, in reader order
/// (`topReaderContext.leaves()`'s order, which is what decides the outcome of
/// an equal-boost tie and which leaf's `docFreq` an evicted term loses);
/// `doc_count` is the reader-wide `CollectionStatistics.docCount`, i.e. the
/// same segments' `docCount`s summed.
///
/// Pruning is always on here: it is a Lucene behaviour, not an option, and
/// `fuzzy_expanded_terms_pruned`'s switch exists only so
/// `examples/fuzzy_pruning.rs` can time both arms.
pub(crate) fn fuzzy_expansion_across_leaves(
    field_terms: &[&lucene_codecs::blocktree::FieldTerms],
    query: &FuzzyQuery,
    doc_count: i64,
) -> Result<FuzzyCollectionStats> {
    let pattern = FuzzyMatch::new(
        &query.term,
        query.max_edits,
        query.prefix_length,
        query.transpositions,
    );
    let mut collector = FuzzyTopTerms::new(query, &pattern);
    for leaf in field_terms {
        collector.collect_segment(leaf, &pattern, true)?;
    }
    let expansion = collector.finish(doc_count);
    Ok(FuzzyCollectionStats {
        terms: expansion
            .terms
            .iter()
            .map(|t| (t.term.clone(), t.boost))
            .collect(),
        blended_doc_freq: expansion.blended_doc_freq,
        doc_count: expansion.doc_count,
    })
}

/// One selected term as [`bench_only_fuzzy_expansion`] reports it:
/// `(term bytes, FuzzyTermsEnum's BoostAttribute, docFreq)`.
#[doc(hidden)]
pub type BenchExpandedTerm = (Vec<u8>, f32, i64);

/// Measurement-only entry point for `examples/fuzzy_pruning.rs`: the term
/// expansion [`fuzzy_doc_ids`]/[`fuzzy_doc_scores`] run, with the
/// `MaxNonCompetitiveBoostAttribute` feedback loop switchable so both arms are
/// one build in one process (criterion is unusable on this project's machine).
/// Returns the selected terms with their boosts, the blended document
/// frequency, and the edit budget the walk ended on -- the last of which is the
/// only *observable* of the feedback loop itself, so a test can require that it
/// actually fired.
///
/// `#[doc(hidden)] pub` rather than `#[cfg(feature = "test-support")]`, and
/// deliberately: a feature-gated example is one `cargo clippy --all-targets`
/// cannot build, and this project has already been bitten twice by a
/// measurement target left broken for several batches while a stale binary
/// kept producing plausible numbers (see `scripts/gate.sh`'s note on
/// `benchmarks/rust-runner`). The cost of the alternative is one hidden
/// function that forwards to private code.
#[doc(hidden)]
pub fn bench_only_fuzzy_expansion(
    field_terms: &lucene_codecs::blocktree::FieldTerms,
    query: &FuzzyQuery,
    prune: bool,
) -> Result<(Vec<BenchExpandedTerm>, i64, u8)> {
    let (expansion, final_max_edits) =
        fuzzy_expanded_terms_instrumented(field_terms, query, prune)?;
    Ok((
        expansion
            .terms
            .iter()
            .map(|t| (t.term.clone(), t.boost, t.doc_freq))
            .collect(),
        expansion.blended_doc_freq,
        final_max_edits,
    ))
}

/// [`Clause::Fuzzy`]'s matched doc-ID list (`max_expansions`
/// bounded): same union-across-matching-terms mechanism as
/// [`wildcard_doc_ids`]/[`prefix_doc_ids`], built on
/// [`lucene_codecs::blocktree::FieldTerms::fuzzy_intersect`] and
/// [`lucene_codecs::fuzzy::FuzzyMatch`] instead of a glob pattern. Returns an
/// empty `Vec` -- not an error -- when `query.field` doesn't exist in this
/// segment, same "missing field means no matches" convention every other
/// clause follows.
///
/// **`max_expansions` cap**: `fuzzy_intersect` returns a lazy `Iterator` over
/// this segment's field's already-fully-decoded, in-memory sorted term
/// entries (`BlockTreeFields` merges a field's whole dictionary into one
/// `Vec` at open time -- see that module's doc comment -- so there is no
/// on-demand term-dictionary decode left to short-circuit by this point).
/// `.take(query.max_expansions)` here stops pulling from that iterator once
/// the cap is hit, which does avoid running the fuzzy-match predicate and
/// allocating a result for every entry past the cap, but it does **not**
/// skip any decode/IO work, since that already happened when the segment was
/// opened.
///
/// **Selection when more terms match than `max_expansions` allows** is
/// [`fuzzy_expanded_terms`]'s, i.e. real Lucene's `TopTermsRewrite` priority
/// queue: highest `FuzzyTermsEnum` boost first, ties broken by ascending term
/// bytes.
pub(super) fn fuzzy_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &FuzzyQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let expansion = fuzzy_expanded_terms(field_terms, query)?;
    let mut acc = DocIdBitSet::default();
    for ExpandedTerm { term, .. } in &expansion.terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}

/// A lone [`FuzzyQuery`] scored the way Lucene scores it:
/// `TopTermsBlendedFreqScoringRewrite` makes it a `BlendedTermQuery` whose
/// `BOOLEAN_REWRITE` is a pure-`SHOULD` `BooleanQuery` of `TermQuery`s -- each
/// boosted by its edit distance, all sharing the blended document frequency --
/// and that disjunction is scored by `MaxScoreBulkScorer`
/// ([`bulk_scorer::score_disjunction`]).
///
/// [`fuzzy_doc_scores`] materialized every posting of every selected term
/// (up to `maxExpansions`, 1024 by default) into one list and summed scores
/// through a `HashMap<i32, f32>`, so a top-10 fuzzy query paid for every
/// matching document of every term. Here the terms whose block maxima cannot
/// reach the threshold become non-essential and are only advanced to the
/// candidates the others produce, and scores are summed in `f64` as Lucene
/// sums them.
///
/// A pulsed single-document term takes part as a one-document cursor
/// ([`lucene_codecs::blocktree::FieldTerms::lazy_postings_for`]).
pub(super) fn try_fuzzy_bulk<C: ScoringCollector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &FuzzyQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
    collector: &mut C,
) -> Result<bool> {
    let Some(doc_in) = doc_in else {
        return Ok(false);
    };
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(true); // no such field: nothing matches
    };
    let (selected, blended_doc_freq, doc_count) = match global.and_then(|g| g.fuzzy(query)) {
        Some(g) => (g.terms.clone(), g.blended_doc_freq, g.doc_count),
        None => {
            let expansion = fuzzy_expanded_terms(field_terms, query)?;
            (
                expansion
                    .terms
                    .iter()
                    .map(|t| (t.term.clone(), t.boost))
                    .collect::<Vec<_>>(),
                expansion.blended_doc_freq,
                expansion.doc_count,
            )
        }
    };
    let idf = similarity::idf(blended_doc_freq, doc_count);
    let mut legs: Vec<bulk_scorer::TermLeg<'_>> = Vec::with_capacity(selected.len());
    for (term, raw_boost) in &selected {
        let Some(seeked) = field_terms.seek_term_state(term)? else {
            continue;
        };
        let stats = seeked.stats;
        // A pulsed single-document term opens as a one-document cursor
        // (`lazy_postings_for`), so it is a leg like any other.
        let cursor = field_terms.lazy_postings_for(
            &seeked,
            doc_in,
            lucene_codecs::postings::PostingsFlags::Freqs,
        )?;
        legs.push(bulk_scorer::TermLeg::scoring(
            cursor,
            // `BoostQuery` folds the boost into the similarity weight.
            raw_boost.max(0.0) * idf,
            norms.map(|n| n.cursor()),
            stats.doc_freq as i64,
            (stats.total_term_freq - stats.doc_freq as i64 + 1).max(1) as f32,
        ));
    }
    bulk_scorer::score_disjunction(&mut legs, live_docs, collector)?;
    Ok(true)
}
/// [`Clause::Fuzzy`]'s BM25 score per matching, live doc.
///
/// Real `FuzzyQuery` is **not** a constant-scoring query: its default rewrite
/// method is `MultiTermQuery.TopTermsBlendedFreqScoringRewrite`, which turns
/// the expanded terms into a `BlendedTermQuery` -- a `BooleanQuery` of
/// `SHOULD` `BoostQuery(TermQuery(t), boost_t)` clauses whose `TermStates`
/// have all been rewritten to one blended document frequency. So a doc's
/// score is
///
/// ```text
/// sum over selected terms t of  max(0, boost_t) * BM25(df_blended, tf(t, doc), norm(doc))
/// ```
///
/// with `boost_t` from [`lucene_codecs::fuzzy::FuzzyMatch::boost`] and
/// `df_blended` from [`fuzzy_expanded_terms`]. `max(0, boost)` is
/// `TopTermsRewrite.build`'s own truncation ("we allow negative term scores
/// while collecting ... but truncate such boosts to 0.0f when building the
/// query"), which matters for one-and-two-character query terms whose
/// similarity really can go negative.
///
/// A flat `1.0` per fuzzy hit would change the top-k of every fuzzy query.
///
/// The blended `df`, the selected term set and the `docCount` are all
/// *reader-wide* statistics in Lucene: `TopTermsRewrite.rewrite` drives one
/// collector across every leaf and `BlendedTermQuery.rewrite` then folds a
/// single `df` over the result. `global` carries them
/// ([`multi_segment::global_fuzzy_stats`]); without it this expands and blends
/// within the one segment it is given, which is the right answer for a
/// single-segment index and the same per-leaf/reader-wide divergence
/// [`CollectionStats`] documents for `TermQuery` on any other.
pub(super) fn fuzzy_doc_scores(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &FuzzyQuery,
    norms: Option<&FieldNorms<'_>>,
    global: Option<&GlobalStats>,
) -> Result<HashMap<i32, f32>> {
    let mut scores: HashMap<i32, f32> = HashMap::new();
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(scores);
    };
    // Reader-wide expansion where the caller gathered one, this segment's own
    // otherwise. The terms are the *selection*, not just the statistics: which
    // terms win the `maxExpansions` queue is itself a reader-wide decision.
    let (selected, blended_doc_freq, doc_count) = match global.and_then(|g| g.fuzzy(query)) {
        Some(g) => (g.terms.clone(), g.blended_doc_freq, g.doc_count),
        None => {
            let expansion = fuzzy_expanded_terms(field_terms, query)?;
            (
                expansion
                    .terms
                    .iter()
                    .map(|t| (t.term.clone(), t.boost))
                    .collect(),
                expansion.blended_doc_freq,
                expansion.doc_count,
            )
        }
    };

    // Resolve every expanded term's live postings first, then score in one
    // pass, rather than scoring inside the per-term loop.
    //
    // Why: the norms lookup wants **ascending** documents. Each expanded term's
    // own postings ascend, but the next term restarts near doc 0, so scoring
    // term-by-term makes a `FieldNormsCursor` rewind once per expansion -- and
    // on a sparse field a rewind is a fresh `IndexedDISI` walk from the first
    // block header, turning O(region) into O(expansions x region). At the
    // measured 677 us per 100,000-document walk, a 50-term expansion would
    // spend ~34 ms in norms alone.
    let mut contributions: Vec<(i32, i32, f32)> = Vec::new(); // (doc, freq, boost)
    for (term, raw_boost) in &selected {
        let boost = raw_boost.max(0.0);
        let Some(postings) = field_terms.postings(term, doc_in)? else {
            continue;
        };
        for (&doc_id, &freq) in postings.docs.iter().zip(postings.freqs.iter()) {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
                contributions.push((doc_id, freq, boost));
            }
        }
    }

    // Sorting is what makes the cursor monotonic, and it is **stable**, which
    // is what keeps this bit-for-bit identical to the per-term loop it
    // replaces: a document's contributions keep their expansion-term order, so
    // the sequence of `+=` into `scores` -- and therefore every float bit --
    // is unchanged. `f32` addition is not associative, so that is a real
    // requirement, not a nicety.
    //
    // Only worth it when a lookup is actually order-sensitive. For the ordinary
    // dense one-byte field the lookup is an array index, so the sort would buy
    // nothing and is skipped.
    if norms.is_some_and(|n| n.prefers_ascending_lookups()) {
        contributions.sort_by_key(|&(doc_id, _, _)| doc_id);
    }

    // `BM25Similarity.scorer(boost, ...)` folds the boost into the *weight*
    // -- `this.weight = boost * idf.getValue().floatValue()` -- and
    // `BM25Scorer.score` then evaluates `weight - weight / (1 + freq *
    // normInverse)`. Multiplying the boost into the finished score instead is
    // the algebraically-equal, bitwise-different rewrite `similarity::do_score`
    // exists to warn about: it was one ULP out on this fixture's doc 2.
    let idf = similarity::idf(blended_doc_freq, doc_count);
    let mut norms_cursor = norms.map(|n| n.cursor());
    for (doc_id, freq, boost) in contributions {
        let norm_inverse = match norms_cursor.as_mut() {
            Some(nc) => nc.norm_inverse(doc_id)?,
            None => similarity::UNNORMED_NORM_INVERSE,
        };
        let score = similarity::do_score(boost * idf, freq as f32, norm_inverse);
        *scores.entry(doc_id).or_insert(0.0) += score;
    }
    Ok(scores)
}

/// [`Clause::Regexp`]'s matched doc-ID list: same
/// union-across-matching-terms mechanism as [`wildcard_doc_ids`]/
/// [`prefix_doc_ids`]/[`fuzzy_doc_ids`], built on
/// [`lucene_codecs::blocktree::FieldTerms::regexp_intersect`] and
/// [`lucene_codecs::regexp::RegexpPattern`] instead of a glob/edit-distance
/// pattern. Returns an empty `Vec` -- not an error -- when `query.field`
/// doesn't exist in this segment, same "missing field means no matches"
/// convention every other clause follows; a malformed `query.pattern`
/// (unsupported regexp syntax) instead surfaces as [`Error::Regexp`],
/// propagated via `?` from [`RegexpPattern::new`] -- distinct from the
/// missing-field/missing-term case because a bad pattern is a caller
/// mistake, not a legitimate "matches nothing" outcome.
pub(super) fn regexp_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &RegexpQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let pattern = RegexpPattern::new(query.pattern.as_bytes())?;
    let matching_terms: Vec<Vec<u8>> = field_terms
        .regexp_intersect(&pattern)
        .map(|r| r.map(|(term, _stats)| term))
        .collect::<lucene_codecs::blocktree::Result<_>>()?;
    let mut acc = DocIdBitSet::default();
    for term in &matching_terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}

/// [`Clause::TermInSet`]'s matched doc-ID list: each of `query.terms` is
/// resolved via the same per-term postings lookup [`term_doc_ids`] uses for a
/// plain [`crate::query::TermQuery`], and the results are **union**ed and
/// deduplicated -- the same "match any of several terms in one field" shape
/// [`wildcard_doc_ids`]/[`prefix_doc_ids`]/[`fuzzy_doc_ids`]/[`regexp_doc_ids`]
/// already implement for a computed (pattern-matched) term set, here applied
/// to an explicit, caller-supplied term list instead. A term absent from the
/// segment (or an absent `query.field` entirely) contributes no doc IDs, not
/// an error -- same "missing means no matches" convention every other clause
/// follows. An empty `query.terms` returns an empty `Vec` (matches nothing)
/// with no special-case branch needed, since the loop below simply doesn't run.
pub(super) fn term_in_set_doc_ids(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermInSetQuery,
) -> Result<Vec<i32>> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(Vec::new());
    };
    let mut acc = DocIdBitSet::default();
    for term in &query.terms {
        let Some(docs) = term_docs_only(field_terms, term, doc_in)? else {
            continue;
        };
        for doc_id in docs {
            if live_docs.is_none_or(|bits| bits.get_doc(doc_id)) {
                acc.set(doc_id);
            }
        }
    }
    let doc_ids = acc.into_sorted_vec();
    Ok(doc_ids)
}
