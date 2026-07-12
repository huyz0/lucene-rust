#![forbid(unsafe_code)]
//! lucene-search: see /PLAN.md for scope.
//!
//! First slice: single-segment `TermQuery` execution — find every live doc
//! ID matching one exact `(field, term)` pair, against a segment already
//! opened the way `crates/lucene-codecs/tests/blocktree_fixtures.rs` opens
//! one today (a `blocktree::BlockTreeFields` plus, when the term's `docFreq
//! > 1`, an opened `.doc` file).
//!
//! ## Scope of this slice (see PLAN.md's Phase 3 section for the full plan)
//!
//! **In scope:**
//! - [`query::TermQuery`]: field + exact term, no scoring metadata.
//! - [`search_term_query`]: `seekExact` the term via
//!   `blocktree::FieldTerms::postings`, then feed every `(docID, freq)` pair
//!   through a [`collector::Collector`], filtered by `live_docs` (deleted
//!   docs excluded — matches `IndexSearcher`'s `Bits liveDocs` handling,
//!   `null` meaning "no deletions, every doc is live").
//! - [`collector::CountCollector`]/[`collector::VecCollector`]: the two
//!   simplest observationally-useful collectors (`TotalHitCountCollector`
//!   and "give me the doc IDs", respectively).
//!
//! **Deliberately out of scope, left for later PLAN.md Phase 3 slices:**
//! - **Relevance scoring.** Real `TermQuery`'s `Weight`/`Scorer` pair always
//!   computes a BM25 (or configured `Similarity`) score per doc, using norms
//!   and collection statistics (`docFreq`/`sumTotalTermFreq`) this port can
//!   already read (`blocktree::TermStats`) but has no `Similarity` module
//!   for yet. A "does this doc match" query is honestly a different, simpler
//!   problem than "how well does this doc match", and PLAN.md's own phase
//!   plan lists Similarity/BM25 as a separate line item — this slice proves
//!   the matching/collection plumbing first, without inventing scoring math
//!   ahead of schedule.
//! - **`BooleanQuery`/conjunction/disjunction.** Not attempted here: it's a
//!   real, separate `DocIdSetIterator`-combination problem (conjunction
//!   advance-to-max-of-all, disjunction min-heap), not a trivial layer on
//!   top of a single `TermQuery`'s doc list.
//! - **Multi-segment search / `IndexSearcher`/`IndexReader` federation.**
//!   This module runs against one already-opened segment's term dictionary
//!   and postings file — there is no `SegmentReader`/`DirectoryReader`/
//!   `IndexReader` abstraction in this port yet (the write side only
//!   produces fully-stored-only segments so far — see
//!   `crates/lucene-index/examples/write_segment_infos_fixture.rs`'s module
//!   doc — and no unified "open every file this segment's `.si` names"
//!   reader exists on the read side either). Building that abstraction is
//!   its own task; this slice takes already-opened
//!   `blocktree::BlockTreeFields`/`postings::DocInput` values as parameters,
//!   the same shape the differential tests in
//!   `crates/lucene-codecs/tests/blocktree_fixtures.rs` already use.
//! - **Dynamic pruning (WAND/MAXSCORE), skip-ahead-driven early
//!   termination, `TopScoreDocCollector`.** All meaningless without scoring;
//!   deferred alongside it. This slice also doesn't use
//!   `postings::LazyDocsCursor`'s decode-on-demand `advance()` — a full
//!   `seekExact` + eager `postings()` materializes every matching doc up
//!   front (same tradeoff `blocktree.rs` itself already made for term
//!   lookup — see that module's doc comment — "correctness first, profile
//!   before optimizing" per the `rust-performance` skill). A future slice
//!   that adds a real multi-term/skip-driven query shape is the right place
//!   to switch to the lazy cursor for genuine sub-linear skipping.
//!
//! **Design note — why a plain function, not a `Weight`/`Scorer` trait
//! hierarchy:** real Lucene's `Query -> Weight -> Scorer/BulkScorer` chain
//! exists to support many query types composing arbitrarily (a
//! `BooleanQuery` wraps `Weight`s recursively) and per-segment reuse of
//! collection statistics across a multi-segment `IndexSearcher`. With
//! exactly one query type and exactly one segment, none of that
//! polymorphism has a second caller yet — introducing the trait hierarchy
//! now would be speculative generality with a single implementation, the
//! opposite of what `rust-performance` asks for. When `BooleanQuery` and
//! multi-segment search land, revisit whether an enum-based `Scorer`
//! (`rust-performance`'s "enums where the closed set allows" guidance)
//! earns its keep.
//!
//! Two concrete pieces of rework this design note defers, named explicitly
//! so the next contributor isn't surprised by their size: (1) **[`collector::Collector`]
//! itself will need a breaking signature change for relevance scoring** --
//! `collect(&mut self, doc_id: i32)` has no way to receive a score the way
//! real Lucene's `LeafCollector` does via `setScorer`/`Scorer.score()`; this
//! isn't a small addition, every existing `Collector` impl's signature
//! changes. (2) **`BooleanQuery` can't reuse `search_term_query`'s loop
//! body** -- postings are eagerly materialized into a `Vec` and filtered
//! inline rather than behind any `DocIdSetIterator`-shaped abstraction, so
//! conjunction/disjunction needs a rewrite from that shape, not an extension
//! of this one.

pub mod collector;
pub mod query;

pub use collector::{Collector, CountCollector, VecCollector};
pub use query::TermQuery;

use lucene_codecs::blocktree::{self, BlockTreeFields};
use lucene_codecs::postings::DocInput;
use lucene_util::fixed_bit_set::FixedBitSet;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    BlockTree(#[from] blocktree::Error),
}

pub type Result<T> = std::result::Result<T, Error>;

/// Executes `query` against one already-opened segment's term dictionary
/// (and, when needed, its `.doc` postings file), feeding every matching
/// **live** doc ID to `collector` in ascending order.
///
/// - `fields`: the segment's decoded term dictionaries
///   (`blocktree::open(...)`'s result).
/// - `doc_in`: the segment's opened `.doc` file, or `None` if the segment
///   has none opened. Only actually needed when the matched term's `docFreq
///   > 1` (see [`blocktree::FieldTerms::postings`]); a `None` here is fine
///   for a field where every term is a `docFreq == 1` singleton (pulsed
///   entirely into the term dictionary, e.g. this port's `id` fixture
///   field) — passing `None` for a term that turns out to need it surfaces
///   as an [`Error`].
/// - `live_docs`: the segment's `.liv` bitset (set bit == live), or `None`
///   for "no deletions in this segment" (mirrors `IndexSearcher`'s `Bits
///   liveDocs == null` convention) — every matched doc is then reported as
///   live.
///
/// Returns `Ok(())` with no doc reported to `collector` when the query's
/// field doesn't exist in this segment or the term isn't found in that
/// field's dictionary (mirrors `TermQuery.createWeight`'s `null`-`Scorer`
/// "no matches" outcome — not an error).
pub fn search_term_query<C: Collector>(
    fields: &BlockTreeFields,
    doc_in: Option<&DocInput<'_>>,
    live_docs: Option<&FixedBitSet>,
    query: &TermQuery,
    collector: &mut C,
) -> Result<()> {
    let Some(field_terms) = fields.field(&query.field) else {
        return Ok(());
    };
    let Some(postings) = field_terms.postings(&query.term, doc_in)? else {
        return Ok(());
    };
    for &doc_id in &postings.docs {
        let is_live = live_docs.is_none_or(|bits| bits.get(doc_id as usize));
        if is_live {
            collector.collect(doc_id);
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    // Reuses the same checked-in real-Lucene fixture
    // (`fixtures/data/blocktree_index/`) the differential test in
    // `crates/lucene-search/tests/term_query_fixtures.rs` opens -- that test
    // is the real-Lucene proof; these unit tests instead focus on
    // `search_term_query`'s own branches (missing field, missing term,
    // singleton no-`.doc`-needed path, the `.doc`-required error path,
    // `live_docs` filtering) using the same real segment data, rather than
    // hand-building a synthetic one (see the `test-coverage` skill: a real
    // fixture beats a hand-built one wherever one is already available).
    fn open_fixture() -> (BlockTreeFields, Option<DocInputOwned>) {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTree)");
        let get = |key: &str| -> String {
            manifest
                .lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        let id_hex = get("id_hex");
        let mut id = [0u8; 16];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = get("segment_suffix");
        let max_doc: i32 = get("max_doc").parse().unwrap();

        let read_raw = |name: &str| -> Vec<u8> {
            std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
        };
        let fnm = read_raw(&get("fnm_file_name"));
        let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");
        let tim = read_raw(&get("tim_file_name"));
        let tip = read_raw(&get("tip_file_name"));
        let tmd = read_raw(&get("tmd_file_name"));
        let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
            .expect("open blocktree");
        let doc = read_raw(&get("doc_file_name"));
        (fields, Some(DocInputOwned { doc, id, suffix }))
    }

    // Owns the `.doc` bytes + segment id/suffix so a `DocInput<'_>` can be
    // constructed with a lifetime tied to a local variable in each test
    // (`DocInput` borrows its buffer).
    struct DocInputOwned {
        doc: Vec<u8>,
        id: [u8; 16],
        suffix: String,
    }

    impl DocInputOwned {
        fn open(&self) -> DocInput<'_> {
            DocInput::open(&self.doc, &self.id, &self.suffix).expect("open .doc")
        }
    }

    #[test]
    fn missing_field_yields_no_matches() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = CountCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("nonexistent", "x"),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.count, 0);
    }

    #[test]
    fn missing_term_yields_no_matches() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "zzz-missing"),
            &mut c,
        )
        .unwrap();
        assert!(c.docs.is_empty());
    }

    #[test]
    fn multi_doc_term_collects_expected_docs_in_order() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut c = VecCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "cat"),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![0, 2]);
    }

    #[test]
    fn singleton_term_needs_no_doc_input() {
        let (fields, _doc) = open_fixture();
        let mut c = VecCollector::default();
        search_term_query(&fields, None, None, &TermQuery::new("id", "id2"), &mut c).unwrap();
        assert_eq!(c.docs, vec![2]);
    }

    #[test]
    fn multi_doc_term_without_doc_input_is_an_error() {
        let (fields, _doc) = open_fixture();
        let mut c = CountCollector::default();
        let err = search_term_query(&fields, None, None, &TermQuery::new("body", "cat"), &mut c)
            .unwrap_err();
        assert!(matches!(err, Error::BlockTree(_)));
    }

    #[test]
    fn live_docs_filters_out_deleted_docs() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let max_doc: i32 = {
            let dir = concat!(
                env!("CARGO_MANIFEST_DIR"),
                "/../../fixtures/data/blocktree_index/"
            );
            let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
            manifest
                .lines()
                .find_map(|l| l.strip_prefix("max_doc="))
                .unwrap()
                .parse()
                .unwrap()
        };
        let mut live_docs = FixedBitSet::new(max_doc as usize);
        for i in 0..max_doc {
            live_docs.set(i as usize);
        }
        // "cat" matches docs 0 and 2 (see manifest); mark doc 0 deleted.
        live_docs.clear(0);

        let mut c = VecCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            Some(&live_docs),
            &TermQuery::new("body", "cat"),
            &mut c,
        )
        .unwrap();
        assert_eq!(c.docs, vec![2]);
    }

    #[test]
    fn count_collector_matches_vec_collector_length() {
        let (fields, doc) = open_fixture();
        let doc_in = doc.as_ref().map(|d| d.open());
        let mut count = CountCollector::default();
        let mut docs = VecCollector::default();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "bird"),
            &mut count,
        )
        .unwrap();
        search_term_query(
            &fields,
            doc_in.as_ref(),
            None,
            &TermQuery::new("body", "bird"),
            &mut docs,
        )
        .unwrap();
        assert_eq!(count.count as usize, docs.docs.len());
    }
}
