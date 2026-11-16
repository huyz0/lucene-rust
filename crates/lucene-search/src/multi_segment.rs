//! Multi-segment search: real `IndexSearcher`'s top-level fan-out --
//! run a query against every segment of an index, translate each segment's
//! locally-scored hits to global doc IDs, and merge into one globally-correct
//! `TopDocs` (`IndexSearcher.search`'s per-`LeafReaderContext` loop +
//! `TopDocs.merge`-equivalent).
//!
//! The single-segment query functions take one already-opened segment's
//! fields/doc-values/norms and returns doc IDs
//! **local** to that segment (`0..maxDoc` for that segment alone). A real
//! multi-segment index's global doc ID space is the concatenation of every
//! segment's local space in segment order -- segment `i`'s `doc_base` is the
//! sum of every earlier segment's `maxDoc` (`SegmentReader.docBase`, computed
//! by `IndexReaderContext.leaves()` in real Lucene; here each
//! [`OpenSegment`] carries its `doc_base`).
//!
//! ## Design: one generic fan-out+merge core, not one per query type
//!
//! Every scored query function this crate already has
//! (`search_term_query_scored`, `search_boolean_query_scored`,
//! `search_phrase_query_scored`, dismax scoring, etc.) shares the same output
//! shape once collected: a bounded, ranked `Vec<ScoreDoc>` (via
//! [`crate::collector::TopDocsCollector`]). The *only* per-segment-specific
//! part of "search every segment and merge" is *how* one segment's local hits
//! get produced -- the merge step itself (translate by `doc_base`, re-rank,
//! truncate to `top_n`) needs nothing but that `Vec<ScoreDoc>` and the
//! `doc_base`. [`merge_multi_segment_scored`] is that one shared core, taking
//! a per-segment closure; [`search_term_query_multi_segment`]/
//! [`search_boolean_query_multi_segment`] are thin, non-duplicated wrappers
//! around it. Adding a third query type's multi-segment entry point (phrase,
//! dismax, ...) means adding another equally thin wrapper, never a second copy
//! of the fan-out/merge logic itself -- this is the DRY design this task's
//! brief explicitly asks for, given every existing scored query type already
//! shares the `Vec<ScoreDoc>`-shaped output.
//!
//! Collecting each segment into its own bounded `TopDocsCollector::new(top_n)`
//! *before* merging (rather than collecting every match from every segment
//! into one giant unbounded list) is both what real
//! `IndexSearcher`/`TopFieldCollector` does per-leaf and provably lossless
//! here: a global top-`top_n` result can never need more than `top_n` hits
//! from any single segment, so truncating each segment's own contribution to
//! its local top-`top_n` first never discards a hit the global merge could
//! have used.
//!
//! The merge itself is implemented by feeding every segment's already-locally-
//! ranked, doc-base-translated hits through one more fresh
//! [`crate::collector::TopDocsCollector`] -- reusing that type's existing
//! score-descending/doc-ID-ascending-tie-break comparator
//! ([`crate::collector::rank_order`], `pub(crate)` so this link is checked) instead of
//! reimplementing the comparator a second time here. This is exactly real
//! Lucene's `TopDocs.merge`/`HitQueue` tie-break rule (verified against
//! `HitQueue.lessThan` in `collector.rs`'s own doc comment already) applied
//! twice -- once per segment, once again across segments -- which is
//! correct because "merge already-sorted lists with the same comparator"
//! composes: feeding already-ranked input through the same bounded top-n
//! collector a second time reproduces the same global ranking a single
//! flat collector over all segments' hits would have produced.
//!
//! ## The sequential entry points share one collector
//!
//! Per-segment collectors are what the *concurrent* entry points need (each
//! rayon task owns its own), but a sequential search that gave every leaf a
//! fresh collector threw away the one thing Lucene's single-threaded
//! `IndexSearcher.search` carries from leaf to leaf: the competitive
//! threshold. Each leaf restarted at "everything is competitive", filled its
//! own top-`top_n`, and only then began skipping blocks -- on the 15-segment
//! benchmark corpus that made 17 of 28 queries slower than Lucene while the
//! same queries on the merged index were all faster.
//! [`search_term_query_multi_segment`], [`search_boolean_query_multi_segment`]
//! and [`search_boolean_query_multi_segment_maxscore`] therefore run through
//! [`search_leaves_shared`]: one [`TopDocsCollector`] for the whole reader,
//! seen by each leaf through a doc-base-shifting
//! [`crate::collector::LeafCollector`]. The hits are the same (same comparator,
//! same global tie-break); only the amount of work differs.
//!
//! ## Scope decision: reader-wide BM25 idf, matching Lucene
//!
//! Real Lucene's default `BM25Similarity` computes `idf` from
//! `CollectionStatistics`/`TermStatistics` gathered **index-wide** across every
//! segment: `IndexSearcher.termStatistics`/`collectionStatistics` sum
//! `docFreq`/`docCount` over every leaf *before* `Similarity.scorer` ever runs,
//! and the resulting `Scorer` is then reused unchanged on each leaf. A term's
//! idf is therefore one number for the whole reader, not one per segment.
//!
//! This module reproduces that. [`global_term_stats`] does the summation for a
//! single term and [`global_boolean_stats`] for every leaf term a boolean query
//! mentions; the result is threaded into each per-segment search through the
//! `_with_stats` entry points in `lib.rs`, so every segment scores from the same
//! counters. `None` there means "use this segment's own counters", which is what
//! the single-segment entry points still pass and is equivalent whenever there
//! is only one segment.
//!
//! This was not always so, and the bug it caused is worth recording. Scoring
//! each segment from its own term dictionary made `body:t0`'s idf range 0.000473
//! to 0.000763 across the 15-segment corpus -- a 1.6x spread around the global
//! 0.000574 -- so the merged top-k filled from whichever segment happened to
//! make the term look rarest, and all 20 benchmark queries disagreed with Java
//! on their hit sets. Every differential fixture is a single segment, where the
//! two are identical, so nothing caught it until the multi-segment benchmark ran.

use crate::collector::{
    FieldValueDoc, LeafCollector, ScoreDoc, ScoringCollector, SortDirection, TopDocsCollector,
    TopFieldCollector, TotalHits,
};
use crate::field_norms::FieldNorms;
use crate::query::{BooleanQuery, Clause, TermQuery};
use crate::Result;

use std::collections::HashMap;
use std::time::Instant;

use lucene_codecs::blocktree::BlockTreeFields;
use lucene_codecs::postings::{DocInput, PayInput, PosInput};
use lucene_util::fixed_bit_set::FixedBitSet;

/// One already-opened segment's inputs, plus its `doc_base` (the segment's
/// starting global doc ID -- sum of every earlier segment's `maxDoc`, matching
/// real `SegmentReader.docBase`). Every field mirrors the identically-named
/// parameter every single-segment scored query function in this crate already
/// takes; see those functions' own doc comments (`lib.rs`) for what each one
/// means and when `None` is valid.
pub struct OpenSegment<'a> {
    pub fields: &'a BlockTreeFields,
    pub doc_in: Option<&'a DocInput<'a>>,
    pub pos_in: Option<&'a PosInput<'a>>,
    pub pay_in: Option<&'a PayInput<'a>>,
    pub live_docs: Option<&'a FixedBitSet>,
    /// This segment's starting global doc ID -- **the caller's
    /// responsibility to compute correctly** (sum of every earlier segment's
    /// `maxDoc`, in the same order `segments` is passed in every function in
    /// this module); a wrong `doc_base` here silently produces wrong global
    /// doc IDs with no way for this module to detect the mistake (it has no
    /// visibility into other segments' `maxDoc`).
    pub doc_base: i32,
    /// This segment's `maxDoc`, which a `MatchAllDocsQuery` clause matches
    /// every document below. `None` when not known here: such a clause then
    /// uses the `max_doc` it was built with, so a query decoded without one
    /// (the JVM's, whose match-all carries none) must come with this set.
    pub max_doc: Option<i32>,
    /// The segment's query cache (`LRUQueryCache` for its core), when it has
    /// one: repeatedly used non-scoring clauses are iterated from it.
    pub cache: Option<&'a crate::SegmentQueryCache>,
    /// The segment's points, when opened (a query with a points clause).
    pub points: Option<&'a crate::points_query::PointsInput<'a>>,
    /// The segment's reader, when the caller opened the segment through one:
    /// what a leaf reading norms or doc values (`FieldExistsQuery`) needs.
    pub reader: Option<&'a crate::directory_reader::SegmentReader>,
    /// Whether a sorted search's sort is a prefix of this segment's index
    /// sort (`TopFieldCollector.canEarlyTerminate`), so its documents come in
    /// the sort's order: set per request by a caller that knows both, and
    /// only a sorted search reads it.
    pub index_sort_prefix: bool,
    /// By field, `[numTerms, sumTotalTermFreq, sumDocFreq, docCount]` of a
    /// segment without a reader whose dictionary may lack a field its
    /// documents have (a buffer's terms read for one request): what the
    /// reader-wide statistics count of it for such a field.
    pub termless: Option<&'a std::collections::HashMap<String, [i64; 4]>>,
    /// The searcher's similarity (`IndexSearcher.setSimilarity`), every
    /// segment of a search holding the same one: what scores the request
    /// when the search function is not handed one itself. `None` is
    /// Lucene's default BM25, which the scorer tree computes on its fast
    /// path.
    pub similarity: Option<&'a dyn crate::similarities::Similarity>,
}

/// The shared fan-out+merge core (see this module's doc comment): runs
/// `per_segment_search` once per index in `0..segments_len` (each call
/// expected to fill `local` with that segment's own locally-ranked hits, in
/// **local** doc-ID space), translates every kept hit to global doc-ID space
/// via `doc_bases[i]`, and merges all segments' contributions into one
/// globally-ranked, `top_n`-truncated result.
///
/// `top_n == 0` is a defined "return nothing" edge case (every per-segment
/// collector and the final merge collector are all sized 0), matching
/// [`TopDocsCollector::new`]'s own `top_n == 0` contract.
///
/// A segment contributing zero matches (its `local` collector stays empty)
/// simply contributes nothing to the merge -- no special-casing needed, since
/// an empty `TopDocsCollector::top_docs()` slice iterates zero times.
pub fn merge_multi_segment_scored<F>(
    doc_bases: &[i32],
    top_n: usize,
    mut per_segment_search: F,
) -> Result<Vec<ScoreDoc>>
where
    F: FnMut(usize, &mut TopDocsCollector) -> Result<()>,
{
    let mut merged = TopDocsCollector::new(top_n);
    for (i, &doc_base) in doc_bases.iter().enumerate() {
        let mut local = TopDocsCollector::new(top_n);
        per_segment_search(i, &mut local)?;
        for hit in local.top_docs() {
            merged.collect(hit.doc_id + doc_base, hit.score);
        }
    }
    Ok(merged.top_docs().to_vec())
}

/// `IndexSearcher.search(query, n)` over several segments the way Lucene runs
/// it single-threaded: **one** [`TopDocsCollector`], handed to each leaf in
/// turn through a [`LeafCollector`] that shifts its doc ids by `doc_base`.
///
/// Returns the same hits as [`merge_multi_segment_scored`] -- the global top
/// `top_n` by score, ties to the lower global doc id -- but each leaf prunes
/// against the threshold every earlier leaf has already raised, instead of
/// starting from nothing. See [`LeafCollector`] for why leaves must be (and
/// here are) visited in ascending `doc_base` order.
pub fn search_leaves_shared<F>(
    doc_bases: &[i32],
    top_n: usize,
    per_segment_search: F,
) -> Result<Vec<ScoreDoc>>
where
    F: FnMut(usize, &mut LeafCollector<'_, TopDocsCollector>) -> Result<()>,
{
    // Visited in ascending `doc_base` order whatever order the caller lists
    // them in: the pruning rule (a later document loses a tie) is only sound
    // then. `per_segment_search` still receives each segment's own index.
    search_leaves_shared_counting(doc_bases, top_n, 0, per_segment_search).map(|(hits, _)| hits)
}

/// [`search_leaves_shared`] with Lucene's `totalHitsThreshold`
/// (`TopScoreDocCollectorManager(numHits, totalHitsThreshold)`): the shared
/// collector publishes no competitive score -- so no leaf may prune -- until it
/// has counted more than `total_hits_threshold` hits, and the returned
/// [`TotalHits`] is exact up to there and a lower bound
/// (`GreaterThanOrEqualTo`) past it. `u64::MAX` counts every hit exactly and
/// never prunes. `top_n` must be at least 1 for the count to mean anything: a
/// collector that keeps nothing collects nothing.
pub fn search_leaves_shared_counting<F>(
    doc_bases: &[i32],
    top_n: usize,
    total_hits_threshold: u64,
    mut per_segment_search: F,
) -> Result<(Vec<ScoreDoc>, TotalHits)>
where
    F: FnMut(usize, &mut LeafCollector<'_, TopDocsCollector>) -> Result<()>,
{
    let mut order: Vec<usize> = (0..doc_bases.len()).collect();
    order.sort_by_key(|&i| doc_bases[i]);
    let mut shared = TopDocsCollector::with_total_hits_threshold(top_n, total_hits_threshold);
    for i in order {
        let mut leaf = LeafCollector::new(&mut shared, doc_bases[i]);
        per_segment_search(i, &mut leaf)?;
    }
    Ok((shared.top_docs().to_vec(), shared.total_hits()))
}

/// `IndexSearcher.searchAfter(after, query, n)` over several segments: the same
/// fan-out and merge as [`merge_multi_segment_scored`], with every leaf's
/// collector told to drop the hits the previous page already returned.
///
/// **The doc-base translation is the whole of it.** `after.doc_id` is a
/// *global* doc id (that is what the caller was handed on page 1), and each
/// leaf's collector compares against *local* ones, so each gets
/// `after.doc_id - doc_base` -- exactly `TopScoreDocCollector.getLeafCollector`'s
/// `afterDoc = after.doc - context.docBase`. The result is deliberately allowed
/// to leave `[0, maxDoc)`:
///
/// - for a leaf **before** the one holding `after`, it is large and positive,
///   so every local doc id ties out and only the score test rejects -- correct,
///   since every one of those documents outranks `after` on a tie;
/// - for a leaf **after** it, it is negative, so no doc id can tie out -- also
///   correct, since none of those documents was on the previous page.
///
/// The merge collector is **not** given `after`: each leaf has already dropped
/// the previous page, so a second filter would drop nothing and could only
/// disagree.
///
/// `after`'s score must be the score that leaf's own search produces, or the
/// page boundary lands in the wrong place -- so a caller paginating must feed
/// back the `ScoreDoc` this function returned, unmodified, and must not change
/// the query between pages. That is Java's contract too.
pub fn merge_multi_segment_scored_after<F>(
    doc_bases: &[i32],
    top_n: usize,
    after: ScoreDoc,
    mut per_segment_search: F,
) -> Result<Vec<ScoreDoc>>
where
    F: FnMut(usize, &mut TopDocsCollector) -> Result<()>,
{
    let mut merged = TopDocsCollector::new(top_n);
    for (i, &doc_base) in doc_bases.iter().enumerate() {
        let mut local = TopDocsCollector::new(top_n).with_after(ScoreDoc {
            doc_id: after.doc_id.saturating_sub(doc_base),
            score: after.score,
        });
        per_segment_search(i, &mut local)?;
        for hit in local.top_docs() {
            merged.collect(hit.doc_id.saturating_add(doc_base), hit.score);
        }
    }
    Ok(merged.top_docs().to_vec())
}

/// [`merge_multi_segment_scored_concurrent`] with one
/// [`crate::collector::MaxScoreAccumulator`] shared across the leaves -- what
/// `TopScoreDocCollectorManager` builds whenever the search is concurrent, and
/// what its `null` stands in for when it is not.
///
/// Without it every leaf prunes only against its own queue, so a leaf that has
/// not yet found anything competitive scans all of its documents even though
/// another leaf has already proved a much higher bar. With it, each leaf
/// publishes the `(global doc id, score)` of its own worst kept hit and may
/// prune against the best pair any leaf has published.
///
/// `total_hits_threshold` is `TopScoreDocCollectorManager`'s: no leaf publishes
/// a threshold, into the accumulator or otherwise, until it has seen more than
/// that many hits, which is what keeps a caller's exact-count budget honest.
/// `u64::MAX` disables pruning entirely (`ScoreMode::COMPLETE`), which also
/// makes the accumulator inert -- correct rather than merely harmless, since an
/// exhaustive collector must not skip.
///
/// **The result does not depend on the scheduling.** The accumulator only ever
/// *raises* a leaf's pruning threshold to one another leaf has already proved
/// competitive, so a leaf that reads it late merely does more work; the hits it
/// keeps are the same. That is why the merge below is still the sequential,
/// input-ordered one [`merge_multi_segment_scored_concurrent`] uses.
pub fn merge_multi_segment_scored_concurrent_shared_max_score<F>(
    doc_bases: &[i32],
    top_n: usize,
    total_hits_threshold: u64,
    per_segment_search: F,
) -> Result<Vec<ScoreDoc>>
where
    F: Fn(usize, &mut TopDocsCollector) -> Result<()> + Sync,
{
    use rayon::prelude::*;

    let acc = std::sync::Arc::new(crate::collector::MaxScoreAccumulator::new());
    let per_segment_hits: Vec<Result<Vec<ScoreDoc>>> = doc_bases
        .par_iter()
        .enumerate()
        .map(|(i, &doc_base)| {
            let mut local =
                TopDocsCollector::with_total_hits_threshold(top_n, total_hits_threshold)
                    .with_shared_max_score(std::sync::Arc::clone(&acc), doc_base);
            per_segment_search(i, &mut local)?;
            Ok(local
                .top_docs()
                .iter()
                .map(|hit| ScoreDoc {
                    doc_id: hit.doc_id.saturating_add(doc_base),
                    score: hit.score,
                })
                .collect())
        })
        .collect();

    let mut merged = TopDocsCollector::new(top_n);
    for hits in per_segment_hits {
        for hit in hits? {
            merged.collect(hit.doc_id, hit.score);
        }
    }
    Ok(merged.top_docs().to_vec())
}

/// Deadline-aware sibling of [`merge_multi_segment_scored`] -- real Lucene's
/// `TimeLimitingBulkScorer`/`IndexSearcher.search`'s `queryTimeout` param stop
/// scoring early once a caller-supplied clock expires, returning whatever
/// hits were already collected rather than hanging or erroring
/// (`TimeLimitingBulkScorer.intersect` throws `TimeExceededException`, which
/// `IndexSearcher` catches and turns into a "partial" `TopDocs` --
/// `TopDocs.totalHits.relation ==
/// GREATER_THAN_OR_EQUAL_TO`). This port has no `TotalHits`/relation type
/// yet (`ScoreDoc`/`TopDocsCollector` only ever track exact counts -- see
/// `collector.rs`), so the partial/complete signal here is the plain `bool`
/// returned alongside the hits: `true` means the deadline was hit before
/// every segment could be searched, `false` means every segment ran to
/// completion exactly as [`merge_multi_segment_scored`] would have.
///
/// The deadline is checked at the one natural checkpoint this fan-out
/// already has without restructuring the collector API: **once per
/// segment**, before that segment's own `per_segment_search` call starts
/// (mirroring real Lucene's own per-segment/per-leaf timeout granularity --
/// `TimeLimitingBulkScorer` itself only checks every ~grow-doubling interval
/// of docs *within* one leaf, i.e. also not a true per-document check). A
/// segment that's already started always finishes -- this is a best-effort,
/// wall-clock deadline, not preemption.
///
/// `deadline: None` means "no timeout" -- behaves identically to
/// [`merge_multi_segment_scored`] (same hits, `timed_out` always `false`).
pub fn merge_multi_segment_scored_with_deadline<F>(
    doc_bases: &[i32],
    top_n: usize,
    deadline: Option<Instant>,
    mut per_segment_search: F,
) -> Result<(Vec<ScoreDoc>, bool)>
where
    F: FnMut(usize, &mut TopDocsCollector) -> Result<()>,
{
    let mut merged = TopDocsCollector::new(top_n);
    let mut timed_out = false;
    for (i, &doc_base) in doc_bases.iter().enumerate() {
        if let Some(d) = deadline {
            if Instant::now() >= d {
                timed_out = true;
                break;
            }
        }
        let mut local = TopDocsCollector::new(top_n);
        per_segment_search(i, &mut local)?;
        for hit in local.top_docs() {
            merged.collect(hit.doc_id + doc_base, hit.score);
        }
    }
    Ok((merged.top_docs().to_vec(), timed_out))
}

/// Concurrent sibling of [`merge_multi_segment_scored`] -- real Lucene's
/// `IndexSearcher` constructed with an `Executor` runs each leaf's search on
/// that executor and merges the partial `TopDocs` once every leaf finishes
/// (`IndexSearcher.slices`/`searchLeaf`). This port has no thread-pool
/// abstraction of its own (`rayon` is already a workspace dependency used
/// elsewhere in this crate), so the per-segment fan-out below is expressed as
/// a `rayon` `par_iter` instead of hand-rolling an `Executor`/`Future`
/// mechanism -- rayon's global pool plays the same role real Lucene's
/// `Executor` does here, minus any configuration knobs (see this crate's
/// `docs/parity.md` entry for exactly what that leaves out).
///
/// Every segment's own `TopDocsCollector::new(top_n)` + doc-base translation
/// happens independently inside the `par_iter` closure -- each segment's
/// contribution is a self-contained `Vec<ScoreDoc>` with no shared mutable
/// state between segments, so no locking is needed for the fan-out itself.
/// The final merge step is then run **sequentially**, feeding every
/// segment's already-doc-base-translated `Vec<ScoreDoc>` through one more
/// [`TopDocsCollector`] in segment order -- this is the exact same merge
/// [`merge_multi_segment_scored`] performs (same collector type, same
/// insertion order across segments, since `par_iter`'s `.collect()` preserves
/// input order regardless of which thread computed which element), which is
/// what makes the two functions' outputs provably identical for the same
/// input rather than merely usually-the-same.
///
/// `per_segment_search` must be `Fn` (not `FnMut`, unlike
/// [`merge_multi_segment_scored`]'s closure) and `Sync`, since `rayon` may
/// invoke it concurrently from multiple worker threads, one call per segment.
pub fn merge_multi_segment_scored_concurrent<F>(
    doc_bases: &[i32],
    top_n: usize,
    per_segment_search: F,
) -> Result<Vec<ScoreDoc>>
where
    F: Fn(usize, &mut TopDocsCollector) -> Result<()> + Sync,
{
    use rayon::prelude::*;

    let per_segment_hits: Vec<Result<Vec<ScoreDoc>>> = doc_bases
        .par_iter()
        .enumerate()
        .map(|(i, &doc_base)| {
            let mut local = TopDocsCollector::new(top_n);
            per_segment_search(i, &mut local)?;
            Ok(local
                .top_docs()
                .iter()
                .map(|hit| ScoreDoc {
                    doc_id: hit.doc_id + doc_base,
                    score: hit.score,
                })
                .collect())
        })
        .collect();

    let mut merged = TopDocsCollector::new(top_n);
    for hits in per_segment_hits {
        for hit in hits? {
            merged.collect(hit.doc_id, hit.score);
        }
    }
    Ok(merged.top_docs().to_vec())
}

/// Deadline-aware sibling of [`merge_multi_segment_scored_concurrent`]. Since
/// `rayon`'s `par_iter` dispatches work to its pool up front rather than one
/// segment at a time, there is no single "before segment `i`" checkpoint to
/// hook the way the sequential [`merge_multi_segment_scored_with_deadline`]
/// does -- instead, each segment's own closure checks the deadline itself,
/// right before calling `per_segment_search`, and contributes an empty
/// result (skips its own search) if the deadline has already passed. A
/// segment whose closure has already started `per_segment_search` still
/// always finishes (same best-effort contract as the sequential version --
/// this is not preemption). `timed_out` is `true` iff at least one segment
/// was skipped this way.
pub fn merge_multi_segment_scored_concurrent_with_deadline<F>(
    doc_bases: &[i32],
    top_n: usize,
    deadline: Option<Instant>,
    per_segment_search: F,
) -> Result<(Vec<ScoreDoc>, bool)>
where
    F: Fn(usize, &mut TopDocsCollector) -> Result<()> + Sync,
{
    use rayon::prelude::*;
    use std::sync::atomic::{AtomicBool, Ordering};

    let timed_out = AtomicBool::new(false);

    let per_segment_hits: Vec<Result<Vec<ScoreDoc>>> = doc_bases
        .par_iter()
        .enumerate()
        .map(|(i, &doc_base)| {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    timed_out.store(true, Ordering::Relaxed);
                    return Ok(Vec::new());
                }
            }
            let mut local = TopDocsCollector::new(top_n);
            per_segment_search(i, &mut local)?;
            Ok(local
                .top_docs()
                .iter()
                .map(|hit| ScoreDoc {
                    doc_id: hit.doc_id + doc_base,
                    score: hit.score,
                })
                .collect())
        })
        .collect();

    let mut merged = TopDocsCollector::new(top_n);
    for hits in per_segment_hits {
        for hit in hits? {
            merged.collect(hit.doc_id, hit.score);
        }
    }
    Ok((
        merged.top_docs().to_vec(),
        timed_out.load(Ordering::Relaxed),
    ))
}

/// Sum `docFreq` and `docCount` for one term across every segment, as
/// Lucene's `IndexSearcher` does before scoring any leaf.
///
/// Returns `Ok(None)` when no segment has the field, in which case the
/// per-segment values (also nothing) are already correct. A corrupt `.tim`
/// block in *any* segment is an `Err`: silently dropping that segment's
/// `docFreq` would change every other segment's idf, which is the
/// wrong-answer shape reader-wide statistics exist to prevent.
pub(crate) fn global_term_stats(
    segments: &[OpenSegment<'_>],
    field: &str,
    term: &[u8],
) -> crate::Result<Option<crate::CollectionStats>> {
    let mut stats = FieldSums::default();
    let mut seen = false;
    for seg in segments {
        stats.add_segment(seg);
        let Some(ft) = seg.fields.field(field) else {
            seen |= stats.add_termless(seg, field);
            continue;
        };
        seen = true;
        stats.add_field(ft);
        if let Some(term) = ft.try_seek_exact(term)? {
            stats.add_term(term.doc_freq, term.total_term_freq);
        }
    }
    Ok(seen.then_some(stats.0))
}

/// The running sums behind [`crate::CollectionStats`]: `IndexSearcher`'s
/// `collectionStatistics` over every leaf, and `TermStates`' per-term ones.
#[derive(Default)]
struct FieldSums(crate::CollectionStats);

impl FieldSums {
    /// `IndexReader.maxDoc()` counts every leaf, with the field or without.
    /// A segment opened without its `maxDoc` known adds nothing; no
    /// similarity reads `maxDoc`, and the sum is clamped to `docCount` when
    /// one is built from it.
    fn add_segment(&mut self, seg: &OpenSegment<'_>) {
        let max_doc = seg.max_doc.or(seg.reader.map(|r| r.max_doc)).unwrap_or(0);
        self.0.max_doc = self.0.max_doc.saturating_add(i64::from(max_doc.max(0)));
    }

    fn add_field(&mut self, ft: &lucene_codecs::blocktree::FieldTerms) {
        self.add_stats(ft.sum_total_term_freq, ft.sum_doc_freq, ft.doc_count);
    }

    fn add_stats(&mut self, sum_total_term_freq: i64, sum_doc_freq: i64, doc_count: i32) {
        let s = &mut self.0;
        s.doc_count = s.doc_count.saturating_add(i64::from(doc_count));
        s.sum_total_term_freq = s.sum_total_term_freq.saturating_add(sum_total_term_freq);
        s.sum_doc_freq = s.sum_doc_freq.saturating_add(sum_doc_freq);
    }

    /// A segment holding none of the field's terms here though its
    /// documents have the field ([`crate::directory_reader::SegmentReader::termless_field_stats`]):
    /// its statistics count all the same. Whether it had any.
    fn add_termless(&mut self, seg: &OpenSegment<'_>, field: &str) -> bool {
        if let Some((ttf, sdf, dc)) = seg.reader.and_then(|r| r.termless_field_stats(field)) {
            self.add_stats(ttf, sdf, dc);
            return true;
        }
        match seg
            .termless
            .filter(|_| seg.fields.field(field).is_none())
            .and_then(|m| m.get(field))
        {
            Some(&[_, ttf, sdf, dc]) => {
                self.add_stats(ttf, sdf, i32::try_from(dc).unwrap_or(i32::MAX));
                true
            }
            None => false,
        }
    }

    fn add_term(&mut self, doc_freq: i32, total_term_freq: i64) {
        let s = &mut self.0;
        s.doc_freq = s.doc_freq.saturating_add(i64::from(doc_freq));
        s.total_term_freq = s.total_term_freq.saturating_add(total_term_freq);
    }
}

/// Where each segment's seek found a term, by the segment's fields' address.
pub(crate) type TermStates = Vec<(usize, Option<lucene_codecs::blocktree::SeekedTerm>)>;

/// [`global_term_stats`], keeping where each segment's seek found the term
/// (by the segment's fields' address), for the scorers to open its postings
/// without seeking again -- Lucene's `TermStates.build`.
pub(crate) fn global_term_stats_states(
    segments: &[OpenSegment<'_>],
    field: &str,
    term: &[u8],
) -> crate::Result<Option<(crate::CollectionStats, TermStates)>> {
    let mut stats = FieldSums::default();
    let mut seen = false;
    let mut states = Vec::with_capacity(segments.len());
    for seg in segments {
        stats.add_segment(seg);
        let Some(ft) = seg.fields.field(field) else {
            seen |= stats.add_termless(seg, field);
            continue;
        };
        seen = true;
        stats.add_field(ft);
        let state = ft.seek_term_state(term)?;
        if let Some(s) = &state {
            stats.add_term(s.stats.doc_freq, s.stats.total_term_freq);
        }
        states.push((std::ptr::from_ref(seg.fields).addr(), state));
    }
    Ok(seen.then_some((stats.0, states)))
}

/// Sum `sumTotalTermFreq` and `docCount` for one field across every segment and
/// divide -- Java's `IndexSearcher.fieldStats(field)` feeding
/// `BM25Similarity.avgFieldLength(FieldStats)`.
///
/// This is `avgdl`'s half of the reader-wide-statistics rule
/// [`global_term_stats`] implements for `docFreq`/`docCount`. A
/// `SegmentReader` cannot compute it -- it sees only itself -- and
/// per-segment counters would give each leaf a different
/// length-normalization curve for the same term.
///
/// Returns `None` when no segment has the field, in which case there is
/// nothing to normalize and the per-segment answer (also nothing) is already
/// correct.
pub fn global_avg_field_length(segments: &[OpenSegment<'_>], field: &str) -> Option<f32> {
    let mut sum_total_term_freq = 0i64;
    let mut doc_count = 0i64;
    let mut seen = false;
    for seg in segments {
        let (ttf, dc) = match seg.fields.field(field) {
            Some(ft) => (ft.sum_total_term_freq, ft.doc_count),
            None => match seg.reader.and_then(|r| r.termless_field_stats(field)) {
                Some((ttf, _, dc)) => (ttf, dc),
                None => continue,
            },
        };
        seen = true;
        sum_total_term_freq += ttf;
        doc_count += dc as i64;
    }
    seen.then(|| crate::field_norms::avg_field_length(sum_total_term_freq, doc_count))
}

/// Reader-wide expansion and blended frequency for one [`crate::FuzzyQuery`]
/// clause:
/// `TopTermsRewrite.rewrite` driven across every leaf, then
/// `BlendedTermQuery.rewrite`'s `df = Math.max(df, ctx.docFreq())` fold.
///
/// Java's `TermCollectingRewrite.collectTerms` walks
/// `topReaderContext.leaves()` with **one** collector, so the
/// `maxExpansions` queue, the visited-term map (which accumulates a term's
/// `docFreq` across leaves) and the `MaxNonCompetitiveBoostAttribute` the
/// pruning loop reads are all reader-wide. Expanding per segment instead picks
/// a different term set in each leaf and blends a different frequency into it;
/// this is the fuzzy twin of [`global_term_stats`].
///
/// Returns `Ok(None)` when no segment has the field, in which case the
/// per-segment answer (also nothing) is already correct. A corrupt `.tim`
/// block in *any* segment is an `Err`, for the reason
/// [`global_term_stats`] gives.
pub fn global_fuzzy_stats(
    segments: &[OpenSegment<'_>],
    query: &crate::FuzzyQuery,
) -> crate::Result<Option<crate::FuzzyCollectionStats>> {
    let mut per_leaf: Vec<&lucene_codecs::blocktree::FieldTerms> = Vec::new();
    let mut doc_count = 0i64;
    for seg in segments {
        let Some(ft) = seg.fields.field(&query.field) else {
            continue;
        };
        doc_count += ft.doc_count as i64;
        per_leaf.push(ft);
    }
    if per_leaf.is_empty() {
        return Ok(None);
    }
    Ok(Some(crate::fuzzy_expansion_across_leaves(
        &per_leaf, query, doc_count,
    )?))
}

/// Reader-wide statistics for every leaf `Clause::Term` a boolean query
/// mentions, so each segment scores with one idf per term -- see
/// [`crate::CollectionStats`] -- plus every `Clause::Fuzzy`'s reader-wide
/// expansion, see [`global_fuzzy_stats`].
pub(crate) fn global_boolean_stats(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
) -> crate::Result<crate::GlobalStats> {
    global_boolean_stats_with_similarity(segments, query, None)
}

/// [`global_boolean_stats`] under a searcher's similarity (`None` the
/// default BM25), which the function queries' `createWeight` reads (a
/// `tf()`/`idf()`/`norm()` source scaled over the whole index).
pub(crate) fn global_boolean_stats_with_similarity(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    similarity: Option<&dyn crate::similarities::Similarity>,
) -> crate::Result<crate::GlobalStats> {
    reader_stats(segments, query, similarity, false)
}

/// The function queries' reader-wide state alone -- each function's
/// `createWeight` (a value source's context) or `rewrite(searcher)` (a values
/// source's) over every segment, under `similarity`, with the statistics the
/// queries they score read -- or `None` when `query` holds no function query.
///
/// What an entry point that does not score hands its leaves: Java creates
/// every weight over the whole searcher whatever the score mode, so a
/// `scale`'s minimum and maximum, a `docfreq`, an `IndexReaderFunctions`
/// statistic are the reader's on a sorted, counted, aggregated or terminated
/// search too; a leaf built without them refuses the function query
/// ([`crate::exec::function`]) rather than read one segment as the index.
pub(crate) fn global_function_stats(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    similarity: Option<&dyn crate::similarities::Similarity>,
) -> crate::Result<Option<crate::GlobalStats>> {
    let mut functions = Vec::new();
    crate::function::collect_functions(query, &mut functions);
    if functions.is_empty() {
        return Ok(None);
    }
    reader_stats(segments, query, similarity, true).map(Some)
}

/// The statistics a search hands its leaves: every scored term's
/// ([`global_boolean_stats_with_similarity`]) when `scores`, else only the
/// function queries' ([`global_function_stats`]).
pub fn leaf_stats(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    scores: bool,
    similarity: Option<&dyn crate::similarities::Similarity>,
) -> crate::Result<Option<crate::GlobalStats>> {
    if scores {
        global_boolean_stats_with_similarity(segments, query, similarity).map(Some)
    } else {
        global_function_stats(segments, query, similarity)
    }
}

/// [`global_boolean_stats_with_similarity`], or with `functions_only` just
/// what the function queries need: their sources' preparation and the
/// statistics of the queries they score.
fn reader_stats(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    similarity: Option<&dyn crate::similarities::Similarity>,
    functions_only: bool,
) -> crate::Result<crate::GlobalStats> {
    let similarity = similarity.or_else(|| segments.first().and_then(|s| s.similarity));
    fn walk_clause(c: &crate::query::Clause, out: &mut Collected) {
        use crate::query::Clause;
        match c {
            Clause::Term(t) => out
                .terms
                .push((t.field.clone(), t.term.clone(), t.doc_freq)),
            // A phrase's idf is the sum of its constituent terms' idfs, so each
            // of them needs the same reader-wide treatment. Missing this left
            // two phrase queries disagreeing with Java on the 15-segment corpus
            // after every other query had been fixed.
            Clause::Phrase(p) => {
                for term in &p.terms {
                    out.terms.push((p.field.clone(), term.clone(), None));
                }
            }
            // A fuzzy clause *does* score from term statistics -- its default
            // rewrite is `TopTermsBlendedFreqScoringRewrite`, not a
            // constant-score one -- so it needs the same reader-wide treatment,
            // in the shape `BlendedTermQuery` gives it.
            Clause::Fuzzy(f) => out.fuzzy.push(f.clone()),
            Clause::Boolean(inner) => walk(inner, out),
            Clause::DisjunctionMax(d) => {
                for inner in &d.disjuncts {
                    walk_clause(inner, out);
                }
            }
            Clause::Boost(b) => walk_clause(&b.inner, out),
            // `SpanWeight.buildSimWeight`: every leaf term's statistics.
            Clause::Span(s) => {
                let mut leaves = Vec::new();
                crate::collect_span_leaves(s, &mut leaves);
                out.terms
                    .extend(leaves.into_iter().map(|(f, t)| (f, t, None)));
            }
            Clause::MultiPhrase(m) => {
                let mut terms = Vec::new();
                crate::exec::extended::collect_multi_phrase_terms(m, &mut terms);
                out.terms
                    .extend(terms.into_iter().map(|(f, t)| (f, t, None)));
            }
            // The extended queries' own terms, their sub-clauses, and the
            // multi-term queries whose rewrite must see every segment.
            Clause::Extended(e) => {
                let mut terms = Vec::new();
                crate::exec::extended::collect_terms(e, &mut terms);
                out.terms
                    .extend(terms.into_iter().map(|(f, t)| (f, t, None)));
                if let crate::extended_query::ExtendedQuery::MultiTerm(m) = &**e {
                    out.multi_term.push(m.clone());
                }
                for child in e.children() {
                    walk_clause(child, out);
                }
            }
            // Everything else either does not score from term statistics
            // (`ConstantScore`, points, doc-values ranges) or expands to terms
            // and scores a flat `1.0` per match (`Wildcard`, `Prefix`,
            // `Regexp`), where a reader-wide statistic would have nothing to
            // change.
            _ => {}
        }
    }

    // `FuzzyQuery.rewrite` expands across the whole reader whether or not
    // the clause scores: one under a filter or a `constant_score` matches the
    // reader-wide expansion too.
    fn walk_unscored_fuzzy(c: &crate::query::Clause, out: &mut Vec<crate::FuzzyQuery>) {
        use crate::query::Clause;
        match c {
            Clause::Fuzzy(f) => out.push(f.clone()),
            Clause::Boolean(b) => {
                for inner in b
                    .must
                    .iter()
                    .chain(&b.should)
                    .chain(&b.filter)
                    .chain(&b.must_not)
                {
                    walk_unscored_fuzzy(inner, out);
                }
            }
            Clause::DisjunctionMax(d) => {
                for inner in &d.disjuncts {
                    walk_unscored_fuzzy(inner, out);
                }
            }
            Clause::Boost(b) => walk_unscored_fuzzy(&b.inner, out),
            Clause::ConstantScore(c) => walk_unscored_fuzzy(&c.inner, out),
            Clause::Extended(e) => {
                for child in e.children() {
                    walk_unscored_fuzzy(child, out);
                }
            }
            _ => {}
        }
    }

    fn walk(q: &BooleanQuery, out: &mut Collected) {
        // `filter` is deliberately not walked. `BooleanWeight`'s constructor
        // builds a filter clause's `Weight` with `ScoreMode.COMPLETE_NO_SCORES`,
        // and `TermQuery.TermWeight` then skips `searcher.termStatistics`
        // entirely ("we do not need the actual stats, use fake stats"). A term
        // reachable only through a filter clause therefore contributes no idf
        // to anything, and collecting reader-wide statistics for it would be a
        // cross-segment seek per term for a number nothing reads.
        for c in q
            .must
            .iter()
            .chain(q.should.iter())
            .chain(q.must_not.iter())
        {
            walk_clause(c, out);
        }
    }
    let mut collected = Collected::default();
    if !functions_only {
        walk(query, &mut collected);
    }
    // The queries a function query scores (a `QueryValueSource`'s,
    // `DoubleValuesSource.fromQuery`'s) score with reader-wide statistics
    // wherever the function query sits, as their `createWeight`'s
    // `searcher.createWeight(q, COMPLETE, 1)` does.
    let mut functions = Vec::new();
    crate::function::collect_functions(query, &mut functions);
    for f in &functions {
        for q in f.queries() {
            walk_clause(q, &mut collected);
        }
    }
    let mut all_fuzzy = Vec::new();
    let unscored = if functions_only {
        &[][..]
    } else {
        &[&query.must, &query.should, &query.filter, &query.must_not][..]
    };
    for c in unscored.iter().copied().flatten() {
        walk_unscored_fuzzy(c, &mut all_fuzzy);
    }
    for f in all_fuzzy {
        if !collected.fuzzy.contains(&f) {
            collected.fuzzy.push(f);
        }
    }
    let mut map = crate::GlobalStats::new();
    // `TermCollectingRewrite.collectTerms` across every leaf, then the
    // rewritten query's own terms.
    if !collected.multi_term.is_empty() {
        let fields: Vec<&lucene_codecs::blocktree::BlockTreeFields> =
            segments.iter().map(|s| s.fields).collect();
        for m in std::mem::take(&mut collected.multi_term) {
            if let Some(rewritten) = crate::exec::extended::rewrite_multi_term(&fields, &m)? {
                walk_clause(&rewritten, &mut collected);
                map.insert_extended_rewrite(format!("{m:?}"), rewritten);
            }
        }
    }
    for (field, term, doc_freq) in collected.terms {
        if let Some((mut stats, states)) = global_term_stats_states(segments, &field, &term)? {
            // A term carrying its own `TermStates` scores from their `docFreq`
            // (`TermWeight`'s `searcher.termStatistics(term, states.docFreq(),
            // ...)`); where it occurs, and the field's `docCount`, are the
            // reader's.
            if let Some(df) = doc_freq {
                stats.doc_freq = df;
            }
            map.insert_term_states(field, term, stats, states);
        }
    }
    for fuzzy in collected.fuzzy {
        if let Some(stats) = global_fuzzy_stats(segments, &fuzzy)? {
            map.insert_fuzzy(fuzzy, stats);
        }
    }
    // The expanded terms' own statistics, which the blended rewrite scores
    // each term from (`totalTermFreq`, the field's sums).
    crate::exec::extended::add_fuzzy_term_stats(&mut map, segments)?;
    let function_stats =
        crate::function::prepare_functions(segments, &functions, &map, similarity)?;
    map.set_functions(function_stats);
    if let Some(source) = similarity.and_then(|s| s.statistics_source()) {
        map.override_statistics(source)?;
    }
    Ok(map)
}

/// What [`global_boolean_stats`]' clause walk gathers: the leaf terms whose
/// idf must be reader-wide, and the fuzzy clauses whose whole *expansion*
/// must be.
#[derive(Default)]
struct Collected {
    /// `(field, term, the term's own docFreq if it carries one)`.
    terms: Vec<(String, Vec<u8>, Option<i64>)>,
    fuzzy: Vec<crate::FuzzyQuery>,
    /// Multi-term queries under a rewrite that needs every segment's terms.
    multi_term: Vec<crate::extended_query::MultiTermQuery>,
}

/// Multi-segment sibling of [`crate::search_term_query_scored`]: runs `query`
/// against every segment in `segments` (in the order given -- `doc_base`
/// values are used exactly as supplied, see [`OpenSegment::doc_base`]'s doc
/// comment), translates local doc IDs to global ones, and returns the
/// globally top-`top_n` hits by score (ties broken by ascending global doc
/// ID, matching [`crate::collector::TopDocsCollector`]'s own tie-break --
/// see this module's doc comment for why reusing that collector for the
/// merge step reproduces real Lucene's `HitQueue` tie-break globally, not
/// just per segment).
///
/// `norms`: per-segment, parallel to `segments` (`norms[i]` is segment `i`'s
/// opened norms for `query.field`, or `None` to fall back to the constant
/// approximation -- same meaning as `search_term_query_scored`'s own `norms`
/// parameter, just one per segment instead of one total).
///
/// idf comes from [`global_term_stats`]: `docFreq`/`docCount` summed across
/// every segment, so a term scores identically in each one -- see this
/// module's doc comment.
pub fn search_term_query_multi_segment(
    segments: &[OpenSegment<'_>],
    query: &TermQuery,
    norms: &[Option<&FieldNorms<'_>>],
    top_n: usize,
) -> Result<Vec<ScoreDoc>> {
    search_term_query_multi_segment_counting(segments, query, norms, top_n, 0).map(|(h, _)| h)
}

/// [`search_term_query_multi_segment`] that also counts total hits, under
/// Lucene's `totalHitsThreshold` -- see [`search_leaves_shared_counting`]. This
/// is `IndexSearcher.search(query, new TopScoreDocCollectorManager(n,
/// threshold))`, the call OpenSearch's query phase makes with
/// `track_total_hits` as the threshold.
pub fn search_term_query_multi_segment_counting(
    segments: &[OpenSegment<'_>],
    query: &TermQuery,
    norms: &[Option<&FieldNorms<'_>>],
    top_n: usize,
    total_hits_threshold: u64,
) -> Result<(Vec<ScoreDoc>, TotalHits)> {
    debug_assert_eq!(
        segments.len(),
        norms.len(),
        "one norms entry per segment expected"
    );
    // Reader-wide statistics, summed once, so every leaf scores with the same
    // idf -- what Lucene's IndexSearcher does via CollectionStatistics. Scoring
    // each segment from its own counters silently reorders results across a
    // multi-segment index; see CollectionStats.
    let found = global_term_stats_states(segments, &query.field, &query.term)?;
    // A term no segment holds matches nothing: no leaf to visit.
    if found.as_ref().is_none_or(|(stats, _)| stats.doc_freq == 0) {
        return Ok((
            Vec::new(),
            TotalHits {
                value: 0,
                relation: crate::collector::TotalHitsRelation::EqualTo,
            },
        ));
    }
    let (global, states) = found.map(|(g, s)| (Some(g), s)).unwrap_or_default();
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    search_leaves_shared_counting(&doc_bases, top_n, total_hits_threshold, |i, local| {
        let seg = &segments[i];
        // The statistics pass found the term absent here: nothing to read.
        let addr = std::ptr::from_ref(seg.fields).addr();
        if states.iter().any(|(a, s)| *a == addr && s.is_none()) {
            return Ok(());
        }
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_term_query_scored_maxscore_with_stats(
            seg.fields,
            seg.doc_in,
            seg.live_docs,
            query,
            seg_norms,
            global,
            local,
        )
    })
}

/// `IndexSearcher.searchAfter(after, new TermQuery(...), n)` across segments:
/// [`search_term_query_multi_segment`]'s next page.
///
/// `after` is a hit this function (or [`search_term_query_multi_segment`])
/// returned on an earlier page, in **global** doc-ID space and with the score it was
/// returned with -- see [`merge_multi_segment_scored_after`] for why both
/// halves matter and how the doc id is translated per leaf.
///
/// Reader-wide statistics are recomputed, which is what keeps the pages
/// consistent: page 2 must score with the same idf page 1 did, or the boundary
/// moves under the caller.
pub fn search_term_query_multi_segment_after(
    segments: &[OpenSegment<'_>],
    query: &TermQuery,
    norms: &[Option<&FieldNorms<'_>>],
    top_n: usize,
    after: ScoreDoc,
) -> Result<Vec<ScoreDoc>> {
    debug_assert_eq!(
        segments.len(),
        norms.len(),
        "one norms entry per segment expected"
    );
    let global = global_term_stats(segments, &query.field, &query.term)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored_after(&doc_bases, top_n, after, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_term_query_scored_maxscore_with_stats(
            seg.fields,
            seg.doc_in,
            seg.live_docs,
            query,
            seg_norms,
            global,
            local,
        )
    })
}

/// Concurrent sibling of [`search_term_query_multi_segment`], built on
/// [`merge_multi_segment_scored_concurrent_shared_max_score`] instead of
/// [`merge_multi_segment_scored`] -- searches every segment in parallel via
/// `rayon`, every leaf pruning against the best bar any leaf has proved (as
/// Java's concurrent `TopScoreDocCollectorManager` does), and merges the
/// results with the identical merge logic. See that function's doc comment
/// for why this produces byte-for-byte identical output to the sequential
/// path.
pub fn search_term_query_multi_segment_concurrent(
    segments: &[OpenSegment<'_>],
    query: &TermQuery,
    norms: &[Option<&FieldNorms<'_>>],
    top_n: usize,
) -> Result<Vec<ScoreDoc>> {
    debug_assert_eq!(
        segments.len(),
        norms.len(),
        "one norms entry per segment expected"
    );
    // Reader-wide statistics, exactly as the sequential path computes them.
    // Scoring each segment from its *own* `docFreq`/`docCount` (no stats)
    // would rank differently from the sequential path on any multi-segment
    // index, with an idf spread the module's doc comment records (1.6x across
    // a 15-segment corpus).
    let global = global_term_stats(segments, &query.field, &query.term)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored_concurrent_shared_max_score(&doc_bases, top_n, 0, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_term_query_scored_maxscore_with_stats(
            seg.fields,
            seg.doc_in,
            seg.live_docs,
            query,
            seg_norms,
            global,
            local,
        )
    })
}

/// Multi-segment sibling of [`crate::search_boolean_query_scored`] -- same
/// per-segment fan-out/merge as [`search_term_query_multi_segment`], built on
/// the same shared [`merge_multi_segment_scored`] core, generalized to a
/// `BooleanQuery` (and, since `search_boolean_query_scored` already resolves
/// nested `Clause::Boolean`/`Clause::DisjunctionMax`/etc. recursively, every
/// clause shape that function supports works unchanged here too).
///
/// `norms`: per-segment, parallel to `segments` (`norms[i]` is segment `i`'s
/// `HashMap<String, FieldNorms>` keyed by clause field, or `None` -- same
/// meaning as `search_boolean_query_scored`'s own `norms` parameter).
pub fn search_boolean_query_multi_segment(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    top_n: usize,
) -> Result<Vec<ScoreDoc>> {
    debug_assert_eq!(
        segments.len(),
        norms.len(),
        "one norms entry per segment expected"
    );
    let global = global_boolean_stats(segments, query)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    search_leaves_shared(&doc_bases, top_n, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_boolean_query_scored_segment(seg, query, seg_norms, Some(&global), local)
    })
}

/// Concurrent sibling of [`search_boolean_query_multi_segment`] -- same
/// relationship as [`search_term_query_multi_segment_concurrent`] has to
/// [`search_term_query_multi_segment`].
pub fn search_boolean_query_multi_segment_concurrent(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    top_n: usize,
) -> Result<Vec<ScoreDoc>> {
    debug_assert_eq!(
        segments.len(),
        norms.len(),
        "one norms entry per segment expected"
    );
    let global = global_boolean_stats(segments, query)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored_concurrent_shared_max_score(&doc_bases, top_n, 0, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_boolean_query_scored_segment(seg, query, seg_norms, Some(&global), local)
    })
}

/// Multi-segment sibling of [`crate::search_boolean_query_scored_maxscore`]
///: identical fan-out/merge shape to
/// [`search_boolean_query_multi_segment`] -- same [`merge_multi_segment_scored`]
/// core, same per-segment bounded `TopDocsCollector`, same doc-base
/// translation and cross-segment re-rank -- but each segment is searched with
/// the MAXSCORE-pruned scorer instead of the exhaustive one. MAXSCORE pruning
/// is a per-segment decision (`search_boolean_query_scored_maxscore` looks
/// only at that one segment's own block bounds/`docFreq`/`docCount`), so
/// fanning it out per-segment and merging with the exact same generic core
/// already used for the exhaustive path is a correct, non-duplicated
/// generalization: whatever `search_boolean_query_scored_maxscore` proves
/// about a single segment (identical output to the eager path, strictly
/// fewer blocks decoded when pruning fires) composes unchanged across
/// segments, because the merge step never looks at *how* a segment's hits
/// were produced, only at the resulting `Vec<ScoreDoc>` -- see this module's
/// own doc comment for why "merge already-ranked per-segment output" is
/// agnostic to the per-segment scorer used.
///
/// Falls back to the exhaustive scorer verbatim on any segment where
/// `search_boolean_query_scored_maxscore`'s own preconditions aren't met
/// (`must`/`must_not` non-empty, non-`Clause::Term` should-clauses, a term
/// with `docFreq <= 1`, etc. -- see that function's doc comment for the full
/// list); this is exactly `search_boolean_query_scored_maxscore`'s own
/// documented fallback behavior, not a new one introduced here.
///
/// `norms`: per-segment, parallel to `segments` -- same meaning as
/// [`search_boolean_query_multi_segment`]'s `norms` parameter.
pub fn search_boolean_query_multi_segment_maxscore(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    top_n: usize,
) -> Result<Vec<ScoreDoc>> {
    search_boolean_query_multi_segment_maxscore_counting(segments, query, norms, top_n, 0)
        .map(|(h, _)| h)
}

/// [`search_boolean_query_multi_segment_maxscore`] that also counts total
/// hits, under Lucene's `totalHitsThreshold` -- see
/// [`search_leaves_shared_counting`] and
/// [`search_term_query_multi_segment_counting`].
pub fn search_boolean_query_multi_segment_maxscore_counting(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    top_n: usize,
    total_hits_threshold: u64,
) -> Result<(Vec<ScoreDoc>, TotalHits)> {
    debug_assert_eq!(
        segments.len(),
        norms.len(),
        "one norms entry per segment expected"
    );
    // `BooleanQuery.rewrite`: a boolean of one scoring clause is that clause.
    // A lone term runs as the term query it rewrites to -- one dictionary
    // seek per segment, reused by its scorer, and no segment visited at all
    // for a term that none of them holds. It scores BM25: under the
    // searcher's own similarity the term runs as the scorer tree's.
    if let Some(term) = lone_term(query).filter(|_| segments.iter().all(|s| s.similarity.is_none()))
    {
        let term_norms: Vec<Option<&FieldNorms<'_>>> = norms
            .iter()
            .map(|m| m.and_then(|m| m.get(&term.field)))
            .collect();
        return search_term_query_multi_segment_counting(
            segments,
            term,
            &term_norms,
            top_n,
            total_hits_threshold,
        );
    }
    let rewritten = rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let global = global_boolean_stats(segments, query)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    search_leaves_shared_counting(&doc_bases, top_n, total_hits_threshold, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_boolean_query_scored_segment(seg, query, seg_norms, Some(&global), local)
    })
}

/// A boolean query made ready to score leaf by leaf: rewritten as the
/// searcher rewrites it (point ranges) and holding the statistics of every
/// segment it was prepared over, as a `Weight` created over the whole
/// searcher does whichever leaves it then scores.
pub struct PreparedBoolean {
    query: BooleanQuery,
    global: crate::GlobalStats,
}

impl PreparedBoolean {
    /// `query` over `segments`.
    pub fn new(segments: &[OpenSegment<'_>], query: &BooleanQuery) -> Result<Self> {
        let query = rewrite_points_ranges(query, segments).unwrap_or_else(|| query.clone());
        let global = global_boolean_stats(segments, &query)?;
        Ok(Self { query, global })
    }

    /// Every document of `segment` the query matches, with its score, in
    /// ascending (segment-local) doc ID order: an exhaustive scorer, nothing
    /// pruned.
    pub fn score_all(
        &self,
        segment: &OpenSegment<'_>,
        norms: Option<&HashMap<String, FieldNorms<'_>>>,
    ) -> Result<Vec<ScoreDoc>> {
        struct All(Vec<ScoreDoc>);
        impl crate::collector::ScoringCollector for All {
            fn collect(&mut self, doc_id: i32, score: f32) {
                self.0.push(ScoreDoc { doc_id, score });
            }
        }
        let mut all = All(Vec::new());
        crate::search_boolean_query_scored_segment(
            segment,
            &self.query,
            norms,
            Some(&self.global),
            &mut all,
        )?;
        let mut hits = all.0;
        if hits.windows(2).any(|w| w[0].doc_id > w[1].doc_id) {
            hits.sort_unstable_by_key(|h| h.doc_id);
        }
        Ok(hits)
    }
}

/// [`search_boolean_query_multi_segment_maxscore`] under `similarity` --
/// `IndexSearcher.setSimilarity(similarity)` then `search(query, topN)`.
///
/// Term and phrase clauses score through `similarity.scorer(field, boost,
/// collectionStatistics, termStatistics...)` as `TermWeight`/`PhraseWeight`
/// build it, from statistics summed across every segment
/// (`IndexSearcher.collectionStatistics`/`termStatistics`), each document by
/// `SimScorer.score(freq, norm)` with the field's stored norm, and the
/// scorer tree prunes with `MaxScoreCache`'s bounds for that scorer. The
/// norms must be each segment's stored ones for every scored field
/// (`DirectoryReader::field_norms_by_field`); a field absent from a
/// segment's map scores with the norm `1`, Java's value for a field without
/// norms.
///
/// A default [`crate::similarities::Bm25Similarity`] (and a per-field
/// wrapper that is one everywhere) runs the BM25 fast path unchanged. Every
/// scoring clause scores through the similarity: term, phrase, multi-phrase,
/// span, fuzzy (its blended rewrite) and the extended queries.
pub fn search_boolean_query_multi_segment_with_similarity(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    top_n: usize,
    similarity: &dyn crate::similarities::Similarity,
) -> Result<Vec<ScoreDoc>> {
    if similarity.is_default_bm25() {
        return search_boolean_query_multi_segment_maxscore(segments, query, norms, top_n);
    }
    debug_assert_eq!(
        segments.len(),
        norms.len(),
        "one norms entry per segment expected"
    );
    let rewritten = rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let global = global_boolean_stats(segments, query)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    search_leaves_shared(&doc_bases, top_n, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_boolean_query_scored_segment_with_similarity(
            seg, query, seg_norms, &global, similarity, local,
        )
    })
}

/// `IndexSearcher.setTimeout(queryTimeout)` then `search(query, topN)`:
/// every segment's bulk scorer is a `TimeLimitingBulkScorer`, which checks
/// `deadline` before each window of documents (100, growing by half), so a
/// long segment stops mid-way rather than only between segments. Returns
/// the hits collected before the deadline and whether it cut the search
/// short (`TopDocs.totalHits` becoming a lower bound, `partialResult`).
pub fn search_boolean_query_multi_segment_with_deadline(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    top_n: usize,
    deadline: Instant,
) -> Result<(Vec<ScoreDoc>, bool)> {
    let rewritten = rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let global = global_boolean_stats(segments, query)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    let mut timed_out = false;
    let (hits, skipped) =
        merge_multi_segment_scored_with_deadline(&doc_bases, top_n, Some(deadline), |i, local| {
            let seg = &segments[i];
            let ctx = crate::exec::LeafContext {
                fields: seg.fields,
                doc_in: seg.doc_in,
                pos_in: seg.pos_in,
                pay_in: seg.pay_in,
                live_docs: seg.live_docs,
                points: seg.points,
                norms: norms.get(i).copied().flatten(),
                global: Some(&global),
                max_doc: seg.max_doc,
                cache: seg.cache,
                reader: seg.reader,
                similarity: seg.similarity,
            };
            let mode = crate::exec::Mode::of(local);
            if let Some(mut bulk) = crate::exec::bulk_boolean(&ctx, query, 1.0, mode)? {
                timed_out |= crate::exec::score_segment_with_deadline(
                    &mut bulk,
                    mode,
                    seg.live_docs,
                    local,
                    seg.max_doc,
                    deadline,
                )?;
            }
            Ok(())
        })?;
    Ok((hits, timed_out || skipped))
}

/// The term a boolean of one scoring term clause rewrites to
/// (`BooleanQuery.rewrite`: a lone `MUST` clause with no `minimumShouldMatch`,
/// or a lone `SHOULD` clause with it at most 1).
fn lone_term(query: &BooleanQuery) -> Option<&TermQuery> {
    let clauses = query.must.len() + query.filter.len() + query.should.len() + query.must_not.len();
    if clauses != 1 {
        return None;
    }
    match (&query.must[..], &query.should[..]) {
        ([Clause::Term(t)], _) if query.minimum_should_match == 0 => Some(t),
        (_, [Clause::Term(t)]) if query.minimum_should_match <= 1 => Some(t),
        _ => None,
    }
    // The term path reads the reader's statistics; its own docFreq is the
    // boolean path's to apply.
    .filter(|t| t.doc_freq.is_none())
}

/// [`search_boolean_query_multi_segment_maxscore_counting`] behind
/// OpenSearch's `min_score`: a `MinimumScoreCollector` around the top-docs
/// collector, so only documents scoring at least `min` are kept and counted.
pub fn search_boolean_query_multi_segment_min_score(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    top_n: usize,
    total_hits_threshold: u64,
    min: f32,
) -> Result<(Vec<ScoreDoc>, TotalHits)> {
    let rewritten = rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let global = global_boolean_stats(segments, query)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    search_leaves_shared_counting(&doc_bases, top_n, total_hits_threshold, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        let mut min_score = crate::collector::MinScoreCollector::new(local, min);
        crate::search_boolean_query_scored_segment(
            seg,
            query,
            seg_norms,
            Some(&global),
            &mut min_score,
        )
    })
}

/// The `size: 0` count behind `min_score`: the live matches scoring at least
/// `min`, segment by segment, stopping at the first segment boundary past
/// `limit` -- `(count, is_lower_bound)`, as a threshold's count reports.
pub fn count_boolean_query_min_score(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    min: f32,
    limit: u64,
) -> Result<(u64, bool)> {
    struct Count(u64);
    impl crate::collector::ScoringCollector for Count {
        fn collect(&mut self, _doc_id: i32, _score: f32) {
            self.0 = self.0.saturating_add(1);
        }
    }
    let rewritten = rewrite_points_ranges(query, segments);
    let query = rewritten.as_ref().unwrap_or(query);
    let global = global_boolean_stats(segments, query)?;
    let mut count = Count(0);
    for (i, seg) in segments.iter().enumerate() {
        let seg_norms = norms.get(i).copied().flatten();
        let mut min_score = crate::collector::MinScoreCollector::new(&mut count, min);
        crate::search_boolean_query_scored_segment(
            seg,
            query,
            seg_norms,
            Some(&global),
            &mut min_score,
        )?;
        if count.0 > limit {
            return Ok((count.0, true));
        }
    }
    Ok((count.0, false))
}

/// `PointRangeQuery.rewrite` against the reader, then `BooleanQuery.rewrite`'s
/// removal of match-all `FILTER` clauses: a points range outside every
/// segment's values matches nothing, and one covering them where every
/// document has a value matches everything -- which a `FILTER` beside another
/// required clause then drops, so `+t #range(all)` runs as `t` alone. The JVM
/// rewrites before encoding; this is the same for a query built here. `None`
/// when nothing changes (or the segments' points are not open).
pub fn rewrite_points_ranges(
    query: &BooleanQuery,
    segments: &[OpenSegment<'_>],
) -> Option<BooleanQuery> {
    fn range(q: &crate::query::PointsRangeQuery, segments: &[OpenSegment<'_>]) -> Option<Clause> {
        let mut global: Option<(Vec<u8>, Vec<u8>)> = None;
        let mut every_doc = true;
        for seg in segments {
            let points = seg.points?;
            let field = points
                .field_number(&q.field)
                .and_then(|n| points.reader.field(n));
            let Some(field) = field else {
                every_doc = false;
                continue;
            };
            every_doc &= Some(field.doc_count) == seg.max_doc && field.num_dims == 1;
            global = Some(match global {
                None => (
                    field.min_packed_value.clone(),
                    field.max_packed_value.clone(),
                ),
                Some((lo, hi)) => (
                    lo.min(field.min_packed_value.clone()),
                    hi.max(field.max_packed_value.clone()),
                ),
            });
        }
        let Some((lo, hi)) = global else {
            return Some(Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new()));
        };
        let (min, max) = (
            crate::points_query::pack_i64(q.min),
            crate::points_query::pack_i64(q.max),
        );
        if max.as_slice() < lo.as_slice() || min.as_slice() > hi.as_slice() {
            return Some(Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new()));
        }
        if every_doc && min.as_slice() <= lo.as_slice() && max.as_slice() >= hi.as_slice() {
            return Some(Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
                i32::MAX,
            )));
        }
        None
    }
    fn clause(c: &Clause, segments: &[OpenSegment<'_>]) -> Option<Clause> {
        match c {
            Clause::PointsRange(q) => range(q, segments),
            Clause::Boolean(b) => boolean(b, segments).map(|b| Clause::Boolean(Box::new(b))),
            Clause::ConstantScore(cs) => clause(&cs.inner, segments).map(|inner| {
                let mut cs = (**cs).clone();
                cs.inner = Box::new(inner);
                Clause::ConstantScore(Box::new(cs))
            }),
            Clause::Boost(b) => clause(&b.inner, segments).map(|inner| {
                let mut b = (**b).clone();
                b.inner = Box::new(inner);
                Clause::Boost(Box::new(b))
            }),
            _ => None,
        }
    }
    fn list(v: &[Clause], segments: &[OpenSegment<'_>], changed: &mut bool) -> Vec<Clause> {
        v.iter()
            .map(|c| match clause(c, segments) {
                Some(n) => {
                    *changed = true;
                    n
                }
                None => c.clone(),
            })
            .collect()
    }
    fn boolean(b: &BooleanQuery, segments: &[OpenSegment<'_>]) -> Option<BooleanQuery> {
        let mut changed = false;
        let mut out = b.clone();
        out.must = list(&b.must, segments, &mut changed);
        out.filter = list(&b.filter, segments, &mut changed);
        out.should = list(&b.should, segments, &mut changed);
        out.must_not = list(&b.must_not, segments, &mut changed);
        if out.filter.len() > 1 || !out.must.is_empty() {
            let before = out.filter.len();
            out.filter.retain(|c| !matches!(c, Clause::MatchAllDocs(_)));
            changed |= out.filter.len() != before;
        }
        changed.then_some(out)
    }
    // Whether anything here can change: a points range, or a match-all a
    // `FILTER` list drops (`boolean`'s `retain`: one directly in it, beside
    // another filter or a `MUST`). Most queries have neither -- a match-all
    // anywhere else is left as it is -- and finding that out without
    // cloning the tree is most of a small query's rewrite.
    fn may_change(c: &Clause) -> bool {
        match c {
            Clause::PointsRange(_) => true,
            Clause::Boolean(b) => boolean_may_change(b),
            Clause::ConstantScore(cs) => may_change(&cs.inner),
            Clause::Boost(b) => may_change(&b.inner),
            _ => false,
        }
    }
    fn boolean_may_change(b: &BooleanQuery) -> bool {
        let drops_match_all = (b.filter.len() > 1 || !b.must.is_empty())
            && b.filter
                .iter()
                .any(|c| matches!(c, Clause::MatchAllDocs(_)));
        drops_match_all
            || b.must
                .iter()
                .chain(&b.filter)
                .chain(&b.should)
                .chain(&b.must_not)
                .any(may_change)
    }
    if !boolean_may_change(query) {
        return None;
    }
    boolean(query, segments)
}

/// Concurrent sibling of [`search_boolean_query_multi_segment_maxscore`] --
/// same relationship [`search_boolean_query_multi_segment_concurrent`] has to
/// [`search_boolean_query_multi_segment`]: fans the per-segment MAXSCORE
/// search out across `rayon`'s global pool via
/// [`merge_multi_segment_scored_concurrent`] instead of running segments one
/// at a time, merging with the identical sequential merge step -- see that
/// function's doc comment for why this is provably byte-for-byte identical
/// to the sequential path for the same input.
pub fn search_boolean_query_multi_segment_maxscore_concurrent(
    segments: &[OpenSegment<'_>],
    query: &BooleanQuery,
    norms: &[Option<&HashMap<String, FieldNorms<'_>>>],
    top_n: usize,
) -> Result<Vec<ScoreDoc>> {
    debug_assert_eq!(
        segments.len(),
        norms.len(),
        "one norms entry per segment expected"
    );
    let global = global_boolean_stats(segments, query)?;
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored_concurrent(&doc_bases, top_n, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_boolean_query_scored_segment(seg, query, seg_norms, Some(&global), local)
    })
}

// --- Sort-by-field multi-segment fan-out/merge ---
//
// Everything above this point merges *scored* (`ScoreDoc`) per-segment
// results through `TopDocsCollector`. `TopFieldCollector`/`FieldValueDoc`
// (`collector.rs`) and `sort_top_n_by_numeric_doc_value`/
// `search_numeric_range_sorted_by_field` (`doc_value_query.rs`) are that same
// shape's numeric-doc-value-sort sibling, but scoped to one already-opened
// segment -- there is no multi-segment fan-out/merge for a sort-by-field
// query yet. [`merge_multi_segment_by_field`] is the direct analogue of
// [`merge_multi_segment_scored`] for that shape: same per-segment-collect-
// then-doc-base-translate-then-re-collect pattern, just keyed by
// `(value, direction)` via [`TopFieldCollector`]/[`field_rank_order`]-
// equivalent ordering instead of by score.
//
// **Scope decision: sequential only, no `rayon`-concurrent sibling (yet).**
// [`merge_multi_segment_scored`] has a `_concurrent` twin
// ([`merge_multi_segment_scored_concurrent`]) because that was this crate's
// established pattern for scored multi-segment search. This task adds only
// the sequential path for sort-by-field: the per-segment work here (a
// doc-values sweep, see `doc_value_query::search_numeric_range`'s own "why a
// full sweep" note) has the same shape as the scored case and a concurrent
// version would follow the exact same recipe (`par_iter` over segments,
// each segment producing its own `Vec<FieldValueDoc>`, sequential final
// merge) -- but adding it here without a caller or test exercising it would
// be exactly the untested, unmotivated surface `rust-performance`/
// `test-coverage` warn against. It is listed in `docs/parity.md`.
use crate::doc_value_query::{self, MissingValue};
use lucene_codecs::doc_values::NumericEntry;

/// The shared fan-out+merge core for sort-by-field, the sort-by-field
/// analogue of [`merge_multi_segment_scored`]: runs `per_segment_search` once
/// per index in `0..doc_bases.len()` (each call expected to `offer` that
/// segment's own locally-ranked hits, in **local** doc-ID space, into the
/// `&mut TopFieldCollector` it's given), translates every kept hit to global
/// doc-ID space via `doc_bases[i]`, and merges all segments' contributions
/// into one globally-ranked (by `direction`, ties broken by ascending global
/// doc ID -- see [`crate::collector::TopFieldCollector`]'s own doc comment),
/// `top_n`-truncated result.
///
/// `top_n == 0` and an empty-contribution segment are defined the same way
/// [`merge_multi_segment_scored`] defines them (see that function's doc
/// comment) -- [`TopFieldCollector::new`] shares the same `top_n == 0`
/// contract as [`TopDocsCollector::new`].
pub fn merge_multi_segment_by_field<F>(
    doc_bases: &[i32],
    top_n: usize,
    direction: SortDirection,
    mut per_segment_search: F,
) -> Result<Vec<FieldValueDoc>>
where
    F: FnMut(usize, &mut TopFieldCollector) -> Result<()>,
{
    let mut merged = TopFieldCollector::new(top_n, direction);
    for (i, &doc_base) in doc_bases.iter().enumerate() {
        let mut local = TopFieldCollector::new(top_n, direction);
        per_segment_search(i, &mut local)?;
        for hit in local.top_docs() {
            merged.offer(hit.doc_id + doc_base, hit.value);
        }
    }
    Ok(merged.top_docs().to_vec())
}

/// One already-opened segment's doc-values inputs for a sort-by-field
/// multi-segment query -- the sort-by-field sibling of [`OpenSegment`],
/// scoped to what [`doc_value_query::search_numeric_range_sorted_by_field`]
/// needs per segment rather than to postings/term-dictionary access.
/// `doc_base` carries the exact same meaning and caller responsibility as
/// [`OpenSegment::doc_base`].
pub struct DocValueSegment<'a> {
    /// The `.dvd` bytes `range_entry` is an entry *of*.
    ///
    /// Separate from [`Self::sort_data`] because a segment's two fields do
    /// not always share one column: after a doc-values update the updated
    /// field is served from its own generation's `.dvd` and the other from
    /// the base one (`SegmentDocValuesProducer.dvProducersByField`). Pass the
    /// same slice twice when they do share one.
    pub range_data: &'a [u8],
    pub range_entry: &'a NumericEntry,
    /// The `.dvd` bytes `sort_entry` is an entry of -- see [`Self::range_data`].
    pub sort_data: &'a [u8],
    pub sort_entry: &'a NumericEntry,
    pub live_docs: Option<&'a lucene_util::fixed_bit_set::FixedBitSet>,
    pub max_doc: i32,
    pub doc_base: i32,
}

/// Multi-segment sibling of
/// [`doc_value_query::search_numeric_range_sorted_by_field`]: runs that exact
/// function against every segment in `segments` independently (matching
/// `[min, max]` on `range_entry`, then ranking the matches by `sort_entry`
/// per `direction`), translates each segment's local doc IDs to global via
/// `doc_base` (see [`DocValueSegment::doc_base`]), and merges into the
/// globally top-`top_n` hits via [`merge_multi_segment_by_field`] --
/// reusing [`TopFieldCollector`]'s existing ascending-doc-ID tie-break both
/// per segment and again across segments (same "merge already-sorted lists
/// with the same comparator composes" argument [`merge_multi_segment_scored`]'s
/// doc comment already makes for the scored case).
///
/// **Scope**: numeric doc-value fields only, single sort key -- the exact
/// same scope [`doc_value_query::search_numeric_range_sorted_by_field`]
/// already has (see that function's and `doc_value_query`'s own module doc
/// for the precise statement); this function adds only the multi-segment
/// fan-out/merge on top, no new capability. See `docs/parity.md` for this
/// task's entry.
pub fn search_numeric_range_sorted_by_field_multi_segment(
    segments: &[DocValueSegment<'_>],
    min: i64,
    max: i64,
    direction: SortDirection,
    missing: MissingValue,
    top_n: usize,
) -> Result<Vec<FieldValueDoc>> {
    // One collector for the whole reader, as Lucene's `TopFieldCollector` is:
    // each segment offers against the bottom the earlier ones set, rather than
    // filling a top-`top_n` of its own that the merge then mostly discards.
    // Same comparator and global tie-break as the per-segment merge, so the
    // same hits -- provided the segments come in ascending `doc_base` order,
    // which is what makes "ties go to the lower doc id" hold across them.
    let mut order: Vec<&DocValueSegment<'_>> = segments.iter().collect();
    order.sort_by_key(|s| s.doc_base);
    let mut collector = TopFieldCollector::new(top_n, direction);
    for seg in order {
        doc_value_query::collect_numeric_range_sorted_by_field(
            seg.range_data,
            seg.range_entry,
            seg.live_docs,
            seg.max_doc,
            min,
            max,
            seg.sort_data,
            seg.sort_entry,
            missing,
            seg.doc_base,
            &mut collector,
        )?;
    }
    Ok(collector.top_docs().to_vec())
}

/// A segment's BKD tree for the sort column: the `LongPoint` field indexed
/// under the same values as the doc-values column being sorted on --
/// Lucene's precondition for `NumericComparator` to use points at all.
#[derive(Clone, Copy)]
pub struct SortPoints<'a> {
    pub reader: &'a lucene_codecs::points::PointsReader<'a>,
    pub field_number: i32,
}

/// [`search_numeric_range_sorted_by_field_multi_segment`] that also skips
/// non-competitive documents with the sort field's points --
/// `NumericComparator`'s competitive iterator.
///
/// Once the shared collector is full, a later segment can only contribute a
/// document whose value strictly beats the collector's bottom (a tie loses on
/// doc id), so instead of scanning every document of the segment it asks the
/// BKD tree for the documents in `[min, bottom - 1]` (ascending; `[bottom + 1,
/// max]` descending) and offers only those. A segment whose points cannot hold
/// such a value costs one tree lookup. On a many-segment index most segments
/// are decided this way after the first fills the collector.
///
/// `points[i]` pairs with `segments[i]`; `None` scans that segment as the
/// plain function does. The pruning is used only where it cannot change the
/// result: the range and the sort are one column, a one-dimensional 8-byte
/// point field is supplied for it, and documents without a value are excluded
/// ([`MissingValue::Exclude`] -- a default value would have no point).
#[allow(clippy::too_many_arguments)]
pub fn search_numeric_range_sorted_by_field_multi_segment_with_points(
    segments: &[DocValueSegment<'_>],
    points: &[Option<SortPoints<'_>>],
    min: i64,
    max: i64,
    direction: SortDirection,
    missing: MissingValue,
    top_n: usize,
) -> Result<Vec<FieldValueDoc>> {
    let mut order: Vec<usize> = (0..segments.len()).collect();
    order.sort_by_key(|&i| segments[i].doc_base);
    let mut collector = TopFieldCollector::new(top_n, direction);
    for i in order {
        let seg = &segments[i];
        let same_column = std::ptr::eq(seg.range_data, seg.sort_data)
            && std::ptr::eq(seg.range_entry, seg.sort_entry);
        let usable = matches!(missing, MissingValue::Exclude)
            && same_column
            && points.get(i).copied().flatten().is_some_and(|p| {
                p.reader
                    .field(p.field_number)
                    .is_some_and(|f| f.num_dims == 1 && f.bytes_per_dim == 8)
            });
        if let (true, Some(bottom)) = (usable, collector.bottom_value()) {
            // The competitive sub-range of `[min, max]`; empty means nothing in
            // this segment can enter.
            let (lo, hi) = match direction {
                SortDirection::Ascending => (Some(min), bottom.checked_sub(1).map(|b| b.min(max))),
                SortDirection::Descending => (bottom.checked_add(1).map(|b| b.max(min)), Some(max)),
            };
            let (Some(lo), Some(hi)) = (lo, hi) else {
                continue;
            };
            if lo > hi {
                continue;
            }
            let Some(p) = points.get(i).copied().flatten() else {
                continue;
            };
            let mut hits = crate::collector::VecCollector::default();
            crate::points_query::search_points_range(
                p.reader,
                seg.live_docs,
                p.field_number,
                &crate::points_query::pack_i64(lo),
                &crate::points_query::pack_i64(hi),
                &mut hits,
            )?;
            let docs = hits.docs;
            let mut values =
                lucene_codecs::doc_values::NumericReader::new(seg.sort_data, seg.sort_entry);
            for doc in docs {
                // The doc-values column is the source of truth for the value
                // (as in Java); the points only chose which documents to read.
                if let Some(v) = values.value(doc)? {
                    if v >= min && v <= max {
                        // ARITH: a global doc id, below the reader's `max_doc`.
                        #[allow(clippy::arithmetic_side_effects)]
                        collector.offer(seg.doc_base + doc, v);
                    }
                }
            }
            continue;
        }
        doc_value_query::collect_numeric_range_sorted_by_field(
            seg.range_data,
            seg.range_entry,
            seg.live_docs,
            seg.max_doc,
            min,
            max,
            seg.sort_data,
            seg.sort_entry,
            missing,
            seg.doc_base,
            &mut collector,
        )?;
    }
    Ok(collector.top_docs().to_vec())
}

#[cfg(test)]
mod tests;
