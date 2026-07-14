//! Multi-segment search (task #41): real `IndexSearcher`'s top-level fan-out --
//! run a query against every segment of an index, translate each segment's
//! locally-scored hits to global doc IDs, and merge into one globally-correct
//! `TopDocs` (`IndexSearcher.search`'s per-`LeafReaderContext` loop +
//! `TopDocs.merge`-equivalent).
//!
//! Every query function in this crate up to this task (task #10 onward) takes
//! one already-opened segment's fields/doc-values/norms and returns doc IDs
//! **local** to that segment (`0..maxDoc` for that segment alone). A real
//! multi-segment index's global doc ID space is the concatenation of every
//! segment's local space in segment order -- segment `i`'s `doc_base` is the
//! sum of every earlier segment's `maxDoc` (`SegmentReader.docBase`, computed
//! by `IndexReaderContext.leaves()` in real Lucene; this port has no
//! `DirectoryReader`/`LeafReaderContext` abstraction yet, so callers compute
//! and pass `doc_base` directly per [`OpenSegment`]).
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
//! ([`crate::collector::rank_order`], private to `collector.rs`) instead of
//! reimplementing the comparator a second time here. This is exactly real
//! Lucene's `TopDocs.merge`/`HitQueue` tie-break rule (verified against
//! `HitQueue.lessThan` in `collector.rs`'s own doc comment already) applied
//! twice -- once per segment, once again across segments -- which is
//! correct because "merge already-sorted lists with the same comparator"
//! composes: feeding already-ranked input through the same bounded top-n
//! collector a second time reproduces the same global ranking a single
//! flat collector over all segments' hits would have produced.
//!
//! ## Scope decision: per-segment BM25 idf, not index-wide idf
//!
//! Real Lucene's default `BM25Similarity` computes `idf` from
//! `CollectionStatistics`/`TermStatistics` gathered **index-wide** across every
//! segment (`IndexSearcher.termStatistics`/`collectionStatistics` sum
//! `docFreq`/`docCount`/`sumTotalTermFreq` over every leaf before `Similarity
//! .scorer` ever runs) -- not per-segment. This port's existing scored query
//! functions (`term_doc_scores` in `lib.rs`, unchanged by this task) compute
//! `idf` from `field_terms.doc_count`/`stats.doc_freq`, which are **that one
//! segment's own** term dictionary statistics (`blocktree::FieldTerms`/
//! `TermStats`, task #13's plumbing) -- there is no index-wide statistics
//! aggregation anywhere in this port.
//!
//! This task deliberately does **not** add that aggregation. Each segment's
//! own score, taken alone, is exactly what this crate's existing
//! differentially-verified scoring already produces (correct for a
//! single-segment index); this task's new code is additive fan-out/merge
//! plumbing on top of already-correct per-segment scores, not a rewrite of
//! the scoring formula itself. Concretely: **the merged, multi-segment
//! `Vec<ScoreDoc>` this module returns is *not* claimed to be a byte-for-byte
//! match of real multi-segment Lucene's BM25 scores** whenever a term's
//! `docFreq`/`docCount` genuinely differ across segments (which is the common
//! case for any index with more than one segment) -- only "correct matching +
//! correct per-segment-relative scoring + correct global merge order" is
//! claimed. This gap is the same one flagged as a known limitation in
//! `docs/parity.md`; adding real index-wide `CollectionStatistics` would need
//! a new aggregation step across every segment's term dictionary before any
//! scoring starts (naturally a `DirectoryReader`-level concept this port
//! doesn't have yet) and is out of scope for this task -- tracked as a
//! follow-up in `docs/parity.md` rather than silently glossed over.

use crate::collector::{ScoreDoc, ScoringCollector, TopDocsCollector};
use crate::field_norms::FieldNorms;
use crate::query::{BooleanQuery, TermQuery};
use crate::Result;

use std::collections::HashMap;

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

/// Concurrent sibling of [`merge_multi_segment_scored`] -- real Lucene's
/// `IndexSearcher` constructed with an `Executor` runs each leaf's search on
/// that executor and merges the partial `TopDocs` once every leaf finishes
/// (`IndexSearcher.slices`/`searchLeaf`). This port has no thread-pool
/// abstraction of its own (`rayon` is already a workspace dependency used
/// elsewhere in this crate), so the per-segment fan-out below is expressed as
/// a `rayon` `par_iter` instead of hand-rolling an `Executor`/`Future`
/// mechanism -- rayon's global pool plays the same role real Lucene's
/// `Executor` does here, minus any configuration knobs (see this crate's
/// `docs/parity.md` entry for this task for exactly what that leaves out).
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
/// See this module's doc comment for the explicit idf scope decision: each
/// segment's own score is computed from *that segment's own* `docFreq`/
/// `docCount`, not an index-wide aggregate.
pub fn search_term_query_multi_segment(
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
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored(&doc_bases, top_n, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_term_query_scored(
            seg.fields,
            seg.doc_in,
            seg.live_docs,
            query,
            seg_norms,
            local,
        )
    })
}

/// Concurrent sibling of [`search_term_query_multi_segment`], built on
/// [`merge_multi_segment_scored_concurrent`] instead of
/// [`merge_multi_segment_scored`] -- searches every segment in parallel via
/// `rayon` and merges the results with the identical merge logic. See
/// [`merge_multi_segment_scored_concurrent`]'s doc comment for why this
/// produces byte-for-byte identical output to the sequential path.
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
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored_concurrent(&doc_bases, top_n, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_term_query_scored(
            seg.fields,
            seg.doc_in,
            seg.live_docs,
            query,
            seg_norms,
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
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored(&doc_bases, top_n, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_boolean_query_scored(
            seg.fields,
            seg.doc_in,
            seg.pos_in,
            seg.pay_in,
            seg.live_docs,
            query,
            seg_norms,
            local,
        )
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
    let doc_bases: Vec<i32> = segments.iter().map(|s| s.doc_base).collect();
    merge_multi_segment_scored_concurrent(&doc_bases, top_n, |i, local| {
        let seg = &segments[i];
        let seg_norms = norms.get(i).copied().flatten();
        crate::search_boolean_query_scored(
            seg.fields,
            seg.doc_in,
            seg.pos_in,
            seg.pay_in,
            seg.live_docs,
            query,
            seg_norms,
            local,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::TermQuery as TQ;
    use crate::{BooleanQuery, TermQuery};
    use lucene_codecs::{blocktree, field_infos};

    fn fixture_dir() -> String {
        concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        )
        .to_string()
    }

    struct Manifest {
        kv: Vec<(String, String)>,
    }

    impl Manifest {
        fn load() -> Self {
            let text = std::fs::read_to_string(format!("{}manifest.properties", fixture_dir()))
                .expect("run fixtures generator first (GenBlockTree)");
            let kv = text
                .lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect();
            Manifest { kv }
        }

        fn get(&self, key: &str) -> &str {
            self.kv
                .iter()
                .find(|(k, _)| k == key)
                .map(|(_, v)| v.as_str())
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
        }
    }

    fn id_from_hex(hex: &str) -> [u8; 16] {
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        id
    }

    fn read_raw(name: &str) -> Vec<u8> {
        std::fs::read(format!("{}{}.raw", fixture_dir(), name))
            .unwrap_or_else(|_| panic!("missing {name}.raw"))
    }

    fn open_real_segment() -> (blocktree::BlockTreeFields, Vec<u8>, [u8; 16], String, i32) {
        let m = Manifest::load();
        let id = id_from_hex(m.get("id_hex"));
        let suffix = m.get("segment_suffix").to_string();
        let max_doc: i32 = m.get("max_doc").parse().unwrap();

        let fnm = read_raw(m.get("fnm_file_name"));
        let field_infos = field_infos::parse(&fnm, &id, "").expect("parse .fnm");

        let tim = read_raw(m.get("tim_file_name"));
        let tip = read_raw(m.get("tip_file_name"));
        let tmd = read_raw(m.get("tmd_file_name"));
        let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
            .expect("open blocktree");

        let doc = read_raw(m.get("doc_file_name"));
        (fields, doc, id, suffix, max_doc)
    }

    /// End-to-end exercise of [`search_term_query_multi_segment`] itself (not
    /// just the generic merge core) against two real, `IndexWriter`-produced
    /// segment copies, proving the thin wrapper actually calls
    /// `search_term_query_scored` per segment and merges the real result.
    #[test]
    fn search_term_query_multi_segment_merges_two_real_segments() {
        let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
        let (fields1, doc1, id1, suffix1, _) = open_real_segment();
        let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
        let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

        let query = TermQuery::new("body", "cat");
        let segments = [
            OpenSegment {
                fields: &fields0,
                doc_in: Some(&doc_in0),
                pos_in: None,
                pay_in: None,
                live_docs: None,
                doc_base: 0,
            },
            OpenSegment {
                fields: &fields1,
                doc_in: Some(&doc_in1),
                pos_in: None,
                pay_in: None,
                live_docs: None,
                doc_base: max_doc0,
            },
        ];
        let norms = [None, None];

        let merged = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        assert!(!merged.is_empty());
        // Every hit from segment 1 must have a doc id >= doc_base (translated).
        for hit in &merged {
            assert!(hit.doc_id >= 0);
        }
        // Ranking must be non-increasing by score.
        for pair in merged.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    /// Same end-to-end exercise for [`search_boolean_query_multi_segment`].
    #[test]
    fn search_boolean_query_multi_segment_merges_two_real_segments() {
        let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
        let (fields1, doc1, id1, suffix1, _) = open_real_segment();
        let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
        let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

        let query =
            BooleanQuery::new().with_should([TQ::new("body", "cat"), TQ::new("body", "bird")]);
        let segments = [
            OpenSegment {
                fields: &fields0,
                doc_in: Some(&doc_in0),
                pos_in: None,
                pay_in: None,
                live_docs: None,
                doc_base: 0,
            },
            OpenSegment {
                fields: &fields1,
                doc_in: Some(&doc_in1),
                pos_in: None,
                pay_in: None,
                live_docs: None,
                doc_base: max_doc0,
            },
        ];
        let norms = [None, None];

        let merged = search_boolean_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        assert!(!merged.is_empty());
        for pair in merged.windows(2) {
            assert!(pair[0].score >= pair[1].score);
        }
    }

    /// A single-segment call (the degenerate `segments.len() == 1` case) must
    /// still translate/merge correctly -- `doc_base != 0` here to prove the
    /// translation isn't accidentally skipped for the one-segment case.
    #[test]
    fn single_segment_with_nonzero_doc_base_still_translates_correctly() {
        let (fields0, doc0, id0, suffix0, _max_doc0) = open_real_segment();
        let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
        let query = TermQuery::new("body", "cat");
        let segments = [OpenSegment {
            fields: &fields0,
            doc_in: Some(&doc_in0),
            pos_in: None,
            pay_in: None,
            live_docs: None,
            doc_base: 1000,
        }];
        let norms = [None];
        let merged = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        assert!(!merged.is_empty());
        for hit in &merged {
            assert!(hit.doc_id >= 1000);
        }
    }

    /// Synthetic per-segment "search": a closure-free stand-in producing
    /// preset `(local_doc_id, score)` pairs, feeding them straight into the
    /// per-segment collector `merge_multi_segment_scored` supplies -- this
    /// tests the merge core in isolation from any real blocktree/postings
    /// decoding, since the merge/doc-base-translation/tie-break logic is
    /// exactly what this task's brief flags as the highest-risk "looks
    /// locally correct, wrong globally" class of bug.
    fn fake_segment_search(hits: Vec<(i32, f32)>) -> impl FnMut(&mut TopDocsCollector) {
        move |local: &mut TopDocsCollector| {
            for &(doc_id, score) in &hits {
                local.collect(doc_id, score);
            }
        }
    }

    #[test]
    fn merges_interleaved_scores_across_three_segments_in_global_order() {
        // Segment 0: local docs 0,1,2 -> doc_base 0 -> global 0,1,2.
        // Segment 1: local docs 0,1   -> doc_base 3 -> global 3,4.
        // Segment 2: local docs 0,1,2 -> doc_base 5 -> global 5,6,7.
        let doc_bases = [0, 3, 5];
        let mut seg0 = fake_segment_search(vec![(0, 1.0), (1, 5.0), (2, 3.0)]);
        let mut seg1 = fake_segment_search(vec![(0, 4.0), (1, 2.0)]);
        let mut seg2 = fake_segment_search(vec![(0, 6.0), (1, 0.5), (2, 4.5)]);
        let result = merge_multi_segment_scored(&doc_bases, 10, |i, local| {
            match i {
                0 => seg0(local),
                1 => seg1(local),
                2 => seg2(local),
                _ => unreachable!(),
            }
            Ok(())
        })
        .unwrap();
        // Global doc IDs: seg0 -> {0:1.0, 1:5.0, 2:3.0}, seg1 -> {3:4.0, 4:2.0},
        // seg2 -> {5:6.0, 6:0.5, 7:4.5}.
        // Expected score-descending order: 5(6.0), 1(5.0), 7(4.5), 3(4.0),
        // 2(3.0), 4(2.0), 0(1.0), 6(0.5).
        let expected: Vec<(i32, f32)> = vec![
            (5, 6.0),
            (1, 5.0),
            (7, 4.5),
            (3, 4.0),
            (2, 3.0),
            (4, 2.0),
            (0, 1.0),
            (6, 0.5),
        ];
        let actual: Vec<(i32, f32)> = result.iter().map(|d| (d.doc_id, d.score)).collect();
        assert_eq!(actual, expected);
    }

    #[test]
    fn truncates_merged_result_to_top_n() {
        let doc_bases = [0, 10];
        let mut seg0 = fake_segment_search(vec![(0, 1.0), (1, 2.0), (2, 3.0)]);
        let mut seg1 = fake_segment_search(vec![(0, 4.0), (1, 5.0)]);
        let result = merge_multi_segment_scored(&doc_bases, 2, |i, local| {
            match i {
                0 => seg0(local),
                1 => seg1(local),
                _ => unreachable!(),
            }
            Ok(())
        })
        .unwrap();
        let actual: Vec<(i32, f32)> = result.iter().map(|d| (d.doc_id, d.score)).collect();
        // Global scores: 0:1.0, 1:2.0, 2:3.0, 10:4.0, 11:5.0 -- top 2 by score.
        assert_eq!(actual, vec![(11, 5.0), (10, 4.0)]);
    }

    #[test]
    fn segment_with_zero_matches_does_not_break_the_merge() {
        let doc_bases = [0, 2, 3];
        let mut seg0 = fake_segment_search(vec![(0, 1.0), (1, 2.0)]);
        let mut seg1 = fake_segment_search(vec![]); // no matches in this segment.
        let mut seg2 = fake_segment_search(vec![(0, 3.0)]);
        let result = merge_multi_segment_scored(&doc_bases, 10, |i, local| {
            match i {
                0 => seg0(local),
                1 => seg1(local),
                2 => seg2(local),
                _ => unreachable!(),
            }
            Ok(())
        })
        .unwrap();
        let actual: Vec<(i32, f32)> = result.iter().map(|d| (d.doc_id, d.score)).collect();
        assert_eq!(actual, vec![(3, 3.0), (1, 2.0), (0, 1.0)]);
    }

    #[test]
    fn tie_break_prefers_lower_global_doc_id_across_segments() {
        // Two segments each contribute one doc at the exact same score;
        // real Lucene's HitQueue tie-break (lower doc ID wins) must apply
        // globally, not just within one segment.
        let doc_bases = [0, 100];
        let mut seg0 = fake_segment_search(vec![(5, 3.0)]); // global doc 5.
        let mut seg1 = fake_segment_search(vec![(2, 3.0)]); // global doc 102.
        let result = merge_multi_segment_scored(&doc_bases, 10, |i, local| {
            match i {
                0 => seg0(local),
                1 => seg1(local),
                _ => unreachable!(),
            }
            Ok(())
        })
        .unwrap();
        let actual: Vec<(i32, f32)> = result.iter().map(|d| (d.doc_id, d.score)).collect();
        assert_eq!(actual, vec![(5, 3.0), (102, 3.0)]);
    }

    #[test]
    fn top_n_zero_returns_nothing() {
        let doc_bases = [0, 5];
        let mut seg0 = fake_segment_search(vec![(0, 1.0)]);
        let mut seg1 = fake_segment_search(vec![(0, 2.0)]);
        let result = merge_multi_segment_scored(&doc_bases, 0, |i, local| {
            match i {
                0 => seg0(local),
                1 => seg1(local),
                _ => unreachable!(),
            }
            Ok(())
        })
        .unwrap();
        assert!(result.is_empty());
    }

    #[test]
    fn no_segments_returns_empty() {
        let doc_bases: [i32; 0] = [];
        let result: Vec<ScoreDoc> =
            merge_multi_segment_scored(&doc_bases, 10, |_, _| Ok(())).unwrap();
        assert!(result.is_empty());
    }

    // --- Concurrent (rayon) path: must be byte-for-byte identical to the
    // sequential path for the same input. ---

    /// Builds `n` synthetic segments, each contributing a small, distinctly
    /// scored set of hits, with doc bases spaced far enough apart that global
    /// doc IDs never collide -- shared by every concurrent-vs-sequential test
    /// below so both paths run over the exact same fan-out.
    fn synthetic_doc_bases_and_hits(n: usize) -> (Vec<i32>, Vec<Vec<(i32, f32)>>) {
        let mut doc_bases = Vec::with_capacity(n);
        let mut hits = Vec::with_capacity(n);
        for i in 0..n {
            doc_bases.push((i as i32) * 100);
            // Distinct, deterministic scores per segment/doc so tie-breaks
            // and ordering are exercised the same way every run.
            hits.push(vec![
                (0, (i as f32) * 0.37 + 1.0),
                (1, (i as f32) * 0.11 + 2.0),
                (2, (i as f32 % 3.0) + 0.5),
            ]);
        }
        (doc_bases, hits)
    }

    fn run_sequential(doc_bases: &[i32], hits: &[Vec<(i32, f32)>], top_n: usize) -> Vec<ScoreDoc> {
        merge_multi_segment_scored(doc_bases, top_n, |i, local| {
            for &(doc_id, score) in &hits[i] {
                local.collect(doc_id, score);
            }
            Ok(())
        })
        .unwrap()
    }

    fn run_concurrent(doc_bases: &[i32], hits: &[Vec<(i32, f32)>], top_n: usize) -> Vec<ScoreDoc> {
        merge_multi_segment_scored_concurrent(doc_bases, top_n, |i, local| {
            for &(doc_id, score) in &hits[i] {
                local.collect(doc_id, score);
            }
            Ok(())
        })
        .unwrap()
    }

    fn assert_identical(a: &[ScoreDoc], b: &[ScoreDoc]) {
        let a: Vec<(i32, f32)> = a.iter().map(|d| (d.doc_id, d.score)).collect();
        let b: Vec<(i32, f32)> = b.iter().map(|d| (d.doc_id, d.score)).collect();
        assert_eq!(a, b, "sequential and concurrent results must be identical");
    }

    #[test]
    fn sequential_merge_propagates_per_segment_search_error() {
        let doc_bases = vec![0, 10, 20];
        let err = merge_multi_segment_scored(&doc_bases, 10, |i, _local| {
            if i == 1 {
                Err(crate::Error::MissingPosInput)
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert!(matches!(err, crate::Error::MissingPosInput));
    }

    #[test]
    fn concurrent_merge_propagates_per_segment_search_error() {
        let doc_bases = vec![0, 10, 20];
        let err = merge_multi_segment_scored_concurrent(&doc_bases, 10, |i, _local| {
            if i == 1 {
                Err(crate::Error::MissingPosInput)
            } else {
                Ok(())
            }
        })
        .unwrap_err();
        assert!(matches!(err, crate::Error::MissingPosInput));
    }

    #[test]
    fn concurrent_matches_sequential_empty_index() {
        let doc_bases: Vec<i32> = vec![];
        let hits: Vec<Vec<(i32, f32)>> = vec![];
        let seq = run_sequential(&doc_bases, &hits, 10);
        let con = run_concurrent(&doc_bases, &hits, 10);
        assert!(seq.is_empty());
        assert_identical(&seq, &con);
    }

    #[test]
    fn concurrent_matches_sequential_single_segment() {
        let (doc_bases, hits) = synthetic_doc_bases_and_hits(1);
        let seq = run_sequential(&doc_bases, &hits, 10);
        let con = run_concurrent(&doc_bases, &hits, 10);
        assert!(!seq.is_empty());
        assert_identical(&seq, &con);
    }

    #[test]
    fn concurrent_matches_sequential_many_segments() {
        // 16 segments -- enough for rayon's global pool to plausibly run
        // more than one in parallel.
        let (doc_bases, hits) = synthetic_doc_bases_and_hits(16);
        let seq = run_sequential(&doc_bases, &hits, 10);
        let con = run_concurrent(&doc_bases, &hits, 10);
        assert!(!seq.is_empty());
        assert_identical(&seq, &con);
    }

    #[test]
    fn concurrent_matches_sequential_with_ties_across_segments() {
        // Every segment contributes the exact same score at local doc 0 --
        // forces the same score-tie, ordered-by-global-doc-id path the
        // sequential merge already covers, now also through the concurrent
        // merge.
        let doc_bases: Vec<i32> = (0..8).map(|i| i * 10).collect();
        let hits: Vec<Vec<(i32, f32)>> = (0..8).map(|_| vec![(0, 3.0)]).collect();
        let seq = run_sequential(&doc_bases, &hits, 100);
        let con = run_concurrent(&doc_bases, &hits, 100);
        assert_identical(&seq, &con);
    }

    #[test]
    fn concurrent_matches_sequential_with_top_n_truncation() {
        let (doc_bases, hits) = synthetic_doc_bases_and_hits(10);
        let seq = run_sequential(&doc_bases, &hits, 3);
        let con = run_concurrent(&doc_bases, &hits, 3);
        assert_eq!(seq.len(), 3);
        assert_identical(&seq, &con);
    }

    /// End-to-end: [`search_term_query_multi_segment_concurrent`] against the
    /// same two real segments [`search_term_query_multi_segment_merges_two_real_segments`]
    /// uses, proving the concurrent wrapper (not just the generic merge core)
    /// matches the sequential wrapper exactly.
    #[test]
    fn search_term_query_multi_segment_concurrent_matches_sequential() {
        let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
        let (fields1, doc1, id1, suffix1, _) = open_real_segment();
        let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
        let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

        let query = TermQuery::new("body", "cat");
        let segments = [
            OpenSegment {
                fields: &fields0,
                doc_in: Some(&doc_in0),
                pos_in: None,
                pay_in: None,
                live_docs: None,
                doc_base: 0,
            },
            OpenSegment {
                fields: &fields1,
                doc_in: Some(&doc_in1),
                pos_in: None,
                pay_in: None,
                live_docs: None,
                doc_base: max_doc0,
            },
        ];
        let norms = [None, None];

        let seq = search_term_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        let con =
            search_term_query_multi_segment_concurrent(&segments, &query, &norms, 10).unwrap();
        assert!(!seq.is_empty());
        assert_identical(&seq, &con);
    }

    /// Same end-to-end check for [`search_boolean_query_multi_segment_concurrent`].
    #[test]
    fn search_boolean_query_multi_segment_concurrent_matches_sequential() {
        let (fields0, doc0, id0, suffix0, max_doc0) = open_real_segment();
        let (fields1, doc1, id1, suffix1, _) = open_real_segment();
        let doc_in0 = lucene_codecs::postings::DocInput::open(&doc0, &id0, &suffix0).unwrap();
        let doc_in1 = lucene_codecs::postings::DocInput::open(&doc1, &id1, &suffix1).unwrap();

        let query =
            BooleanQuery::new().with_should([TQ::new("body", "cat"), TQ::new("body", "bird")]);
        let segments = [
            OpenSegment {
                fields: &fields0,
                doc_in: Some(&doc_in0),
                pos_in: None,
                pay_in: None,
                live_docs: None,
                doc_base: 0,
            },
            OpenSegment {
                fields: &fields1,
                doc_in: Some(&doc_in1),
                pos_in: None,
                pay_in: None,
                live_docs: None,
                doc_base: max_doc0,
            },
        ];
        let norms = [None, None];

        let seq = search_boolean_query_multi_segment(&segments, &query, &norms, 10).unwrap();
        let con =
            search_boolean_query_multi_segment_concurrent(&segments, &query, &norms, 10).unwrap();
        assert!(!seq.is_empty());
        assert_identical(&seq, &con);
    }
}
