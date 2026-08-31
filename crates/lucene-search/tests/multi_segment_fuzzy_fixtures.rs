//! **A `FuzzyQuery`'s expansion, blended frequency and scores are reader-wide,
//! proven against real Lucene on a real two-segment index.**
//!
//! `FuzzyQuery`'s default rewrite is
//! `MultiTermQuery.TopTermsBlendedFreqScoringRewrite`, and every part of it is
//! computed once across the whole reader:
//! `TermCollectingRewrite.collectTerms` drives one `TopTermsRewrite` collector
//! over `topReaderContext.leaves()`, so the `maxExpansions` queue picks one
//! term set for the whole reader and a term's `docFreq` accumulates across
//! leaves; then `BlendedTermQuery.rewrite` folds `df = max(df, ctx.docFreq())`
//! over those frequencies and scores every clause with it against the
//! reader-wide `CollectionStatistics.docCount`.
//!
//! This port expanded and blended within one segment (ledger item 9's
//! remaining half), which is the multi-term twin of the per-leaf idf defect
//! `multi_segment_scoring_fixtures.rs` covers for `TermQuery` -- and is
//! likewise invisible on a single segment, which is why every other fuzzy
//! fixture in this tree (all over `blocktree_index`) agrees with either
//! answer.
//!
//! The fixture's two segments are lopsided on exactly the axis that matters:
//! `dog`'s `docFreq` is 3 in each leaf and **6** reader-wide, `fox`'s is 1 and
//! 3 against **4**, so `max(df)` over the selected terms is 3 per leaf and 6
//! for the reader.
//!
//! Ground truth: `fixtures/src/AppendMultiSegmentFuzzyManifest.java`, which
//! records the rewritten query's own selected terms and boosts, each term's
//! reader-wide `docFreq`, and real `IndexSearcher` `TopDocs` as raw float bits
//! -- appended to the committed index without regenerating it.

#![allow(clippy::arithmetic_side_effects)] // Test arithmetic is not read off disk.

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{global_fuzzy_stats, search_boolean_query_multi_segment};
use lucene_search::{BooleanQuery, Clause, FuzzyQuery};
use lucene_store::FsDirectory;

fn fixture_dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/multi_segment_scoring_index"
    )
    .to_string()
}

struct Manifest(Vec<(String, String)>);

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}/manifest.properties", fixture_dir()))
            .expect("run scripts/gen-fixtures.sh --append-only first");
        Manifest(
            text.lines()
                .filter_map(|l| l.split_once('='))
                .map(|(k, v)| (k.to_string(), v.to_string()))
                .collect(),
        )
    }

    fn get(&self, key: &str) -> &str {
        self.0
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }

    fn query(&self, case: &str) -> FuzzyQuery {
        let g = |k: &str| self.get(&format!("fuzzymulti.{case}.{k}")).to_string();
        FuzzyQuery {
            field: g("field"),
            term: g("term").into_bytes(),
            max_edits: g("max_edits").parse().unwrap(),
            prefix_length: g("prefix_length").parse().unwrap(),
            transpositions: g("transpositions").parse().unwrap(),
            max_expansions: g("max_expansions").parse().unwrap(),
        }
    }

    fn list(&self, key: &str) -> Vec<String> {
        let raw = self.get(key);
        if raw.is_empty() {
            Vec::new()
        } else {
            raw.split(',').map(str::to_string).collect()
        }
    }

    fn cases(&self) -> Vec<String> {
        self.list("fuzzymulti.cases")
    }

    fn hits(&self, case: &str) -> Vec<(i32, f32)> {
        let raw = self.get(&format!("fuzzymulti.{case}.bits"));
        if raw.is_empty() {
            return Vec::new();
        }
        raw.split(',')
            .map(|pair| {
                let (doc, bits) = pair.split_once(':').expect("doc:bits");
                (
                    doc.parse().unwrap(),
                    f32::from_bits(bits.parse::<u32>().unwrap()),
                )
            })
            .collect()
    }
}

/// The *selection* and the blended frequency, straight off
/// `multi_segment::global_fuzzy_stats`.
///
/// A score comparison alone would let a port pick a different term set and
/// still land on the same numbers by luck; `TopTermsRewrite`'s queue is the
/// half that a per-segment expansion gets wrong first.
#[test]
fn reader_wide_fuzzy_expansion_matches_real_lucenes_rewrite() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    assert_eq!(reader.segment_readers().len(), 2, "two-segment fixture");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();

    let cases = m.cases();
    assert!(cases.len() >= 13, "the recorded case matrix shrank");
    let mut saw_blend_beyond_any_leaf = false;
    for case in &cases {
        let query = m.query(case);
        let stats = global_fuzzy_stats(&segments, &query)
            .unwrap_or_else(|e| panic!("{case}: {e}"))
            .unwrap_or_else(|| panic!("{case}: `body` exists in both segments"));

        // Java sorts the selected terms by bytes before building the query
        // (`ArrayUtil.timSort(scoreTerms, by bytes)`); this port keeps the
        // queue in `(boost desc, bytes asc)` order, so compare by bytes.
        let mut got: Vec<(String, u32)> = stats
            .terms
            .iter()
            .map(|(t, b)| (String::from_utf8(t.clone()).unwrap(), b.to_bits()))
            .collect();
        got.sort_by(|a, b| a.0.cmp(&b.0));
        let expected_terms = m.list(&format!("fuzzymulti.{case}.selected_terms"));
        let expected_boosts = m.list(&format!("fuzzymulti.{case}.selected_boost_bits"));
        let expected: Vec<(String, u32)> = expected_terms
            .iter()
            .cloned()
            .zip(expected_boosts.iter().map(|b| b.parse::<u32>().unwrap()))
            .collect();
        assert_eq!(
            got,
            expected,
            "{case}: selected terms/boosts differ from real Lucene's rewritten query \
             ({})",
            m.get(&format!("fuzzymulti.{case}.rewritten"))
        );

        let expected_blended: i64 = m
            .get(&format!("fuzzymulti.{case}.blended_doc_freq"))
            .parse()
            .unwrap();
        assert_eq!(
            stats.blended_doc_freq, expected_blended,
            "{case}: BlendedTermQuery's `df = max(df, ctx.docFreq())`"
        );

        // Reader-wide `CollectionStatistics.docCount`: both segments index
        // `body` for all four of their documents.
        assert_eq!(stats.doc_count, 8, "{case}: reader-wide docCount");

        // The fixture only proves anything where the reader-wide blend is a
        // number no single leaf could have produced.
        let per_leaf_max: i64 = m
            .list(&format!("fuzzymulti.{case}.selected_doc_freqs"))
            .iter()
            .map(|d| d.parse::<i64>().unwrap())
            .max()
            .unwrap_or(0);
        assert_eq!(per_leaf_max, expected_blended, "{case}: the fold is a max");
        if expected_blended > 4 {
            // 4 is the largest `docFreq` any one leaf can have here.
            saw_blend_beyond_any_leaf = true;
        }
    }
    assert!(
        saw_blend_beyond_any_leaf,
        "no case blends above a single leaf's maximum docFreq -- the fixture proves nothing"
    );
}

/// The scores, bit for bit, through the public multi-segment entry point.
#[test]
fn multi_segment_fuzzy_scores_match_real_lucene_bit_for_bit() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();

    for case in m.cases() {
        let query = m.query(&case);
        let boolean = BooleanQuery::new().with_should(vec![Clause::Fuzzy(query)]);
        let hits = search_boolean_query_multi_segment(&segments, &boolean, &norms, 20).unwrap();
        let expected = m.hits(&case);
        let got: Vec<(i32, f32)> = hits.iter().map(|h| (h.doc_id, h.score)).collect();
        assert_eq!(
            got.len(),
            expected.len(),
            "{case}: hit count -- got {got:?}, Lucene {expected:?}"
        );
        for ((gd, gs), (ed, es)) in got.iter().zip(&expected) {
            assert_eq!(
                gd, ed,
                "{case}: doc order -- got {got:?}, Lucene {expected:?}"
            );
            assert_eq!(
                gs.to_bits(),
                es.to_bits(),
                "{case}: doc {gd} scored {gs} ({:#x}), real Lucene {es} ({:#x})",
                gs.to_bits(),
                es.to_bits()
            );
        }
    }
}
