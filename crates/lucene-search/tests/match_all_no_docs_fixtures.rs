//! Differential/behavioral tests for `Clause::MatchAllDocs`/`Clause::MatchNoDocs`
//! (`MatchAllDocsQuery`/`MatchNoDocsQuery`-equivalents), against the same real
//! fixture segment `boolean_query_fixtures.rs` already uses
//! (`fixtures/data/blocktree_index/`; `max_doc` read from that fixture's own
//! `manifest.properties`, not hardcoded here).
//!
//! Covers:
//! - `MatchAllDocsQuery` over a segment with some docs marked deleted (via a
//!   hand-built `live_docs` bitset, since the fixture itself has no
//!   deletions) returns exactly the live doc set, never the deleted ones.
//! - `MatchNoDocsQuery` always returns empty, deletions or not.
//! - Both compose as a `BooleanQuery` `must` clause: `MatchAllDocsQuery` must
//!   alongside another clause behaves like that other clause alone;
//!   `MatchNoDocsQuery` as a must clause makes the whole query match nothing.
//! - Scored variants (`search_boolean_query_scored`): `MatchAllDocsQuery`
//!   contributes a flat `1.0` per live doc; `MatchNoDocsQuery` contributes
//!   nothing (empty result, since it never matches).

use lucene_codecs::blocktree;
use lucene_codecs::field_infos;
use lucene_codecs::postings::DocInput;
use lucene_search::{
    search_boolean_query, search_boolean_query_scored, BooleanQuery, Clause, MatchAllDocsQuery,
    MatchNoDocsQuery, TermQuery, VecCollector,
};
use lucene_util::fixed_bit_set::FixedBitSet;
use std::collections::BTreeSet;

fn dir() -> String {
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
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
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

    fn doc_set(&self, field: &str, term: &str) -> BTreeSet<i32> {
        self.get(&format!("field.{field}.term.{term}.postingsDocs"))
            .split(',')
            .map(|s| s.parse().unwrap())
            .collect()
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
    std::fs::read(format!("{}{}.raw", dir(), name)).unwrap_or_else(|_| panic!("missing {name}.raw"))
}

fn open_segment() -> (blocktree::BlockTreeFields, Vec<u8>, [u8; 16], String, i32) {
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

/// Builds a `live_docs` bitset over `0..max_doc` with `deleted` docs cleared
/// (bit unset) -- every other doc stays live, mirroring a real `.liv` file's
/// "set bit == live" convention.
fn live_docs_with_deletions(max_doc: i32, deleted: &[i32]) -> FixedBitSet {
    let mut bits = FixedBitSet::new(max_doc as usize);
    for doc_id in 0..max_doc {
        bits.set(doc_id as usize);
    }
    for &doc_id in deleted {
        bits.clear(doc_id as usize);
    }
    bits
}

#[test]
fn match_all_docs_returns_every_live_doc_not_the_deleted_ones() {
    let (fields, _doc, _id, _suffix, max_doc) = open_segment();
    assert!(max_doc > 3, "fixture sanity: needs at least 4 docs");

    // Delete docs 1 and 3 -- live set should be every doc except those two.
    let deleted = [1, 3];
    let live_docs = live_docs_with_deletions(max_doc, &deleted);
    let expected: Vec<i32> = (0..max_doc).filter(|d| !deleted.contains(d)).collect();

    let mut collector = VecCollector::default();
    let query = Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc));
    let boolean = BooleanQuery::new().with_must([query]);
    search_boolean_query(
        &fields,
        None,
        None,
        None,
        Some(&live_docs),
        None,
        &boolean,
        &mut collector,
    )
    .unwrap();
    assert_eq!(collector.docs, expected);
}

#[test]
fn match_all_docs_with_no_deletions_returns_every_doc() {
    let (fields, _doc, _id, _suffix, max_doc) = open_segment();

    let mut collector = VecCollector::default();
    let query = Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc));
    let boolean = BooleanQuery::new().with_must([query]);
    search_boolean_query(
        &fields,
        None,
        None,
        None,
        None,
        None,
        &boolean,
        &mut collector,
    )
    .unwrap();
    let expected: Vec<i32> = (0..max_doc).collect();
    assert_eq!(collector.docs, expected);
}

#[test]
fn match_no_docs_always_returns_empty_regardless_of_deletions() {
    let (fields, _doc, _id, _suffix, max_doc) = open_segment();

    // No deletions.
    let mut collector = VecCollector::default();
    let boolean = BooleanQuery::new().with_must([Clause::MatchNoDocs(MatchNoDocsQuery::new())]);
    search_boolean_query(
        &fields,
        None,
        None,
        None,
        None,
        None,
        &boolean,
        &mut collector,
    )
    .unwrap();
    assert!(collector.docs.is_empty());

    // With deletions -- still empty (MatchNoDocsQuery ignores live_docs entirely).
    let live_docs = live_docs_with_deletions(max_doc, &[0, 1, 2, 3, 4]);
    let mut collector2 = VecCollector::default();
    search_boolean_query(
        &fields,
        None,
        None,
        None,
        Some(&live_docs),
        None,
        &boolean,
        &mut collector2,
    )
    .unwrap();
    assert!(collector2.docs.is_empty());
}

#[test]
fn match_all_docs_as_must_clause_alongside_another_behaves_like_the_other_clause_alone() {
    let (fields, doc, id, suffix, max_doc) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");
    let m = Manifest::load();

    let expected = m.doc_set("body", "cat"); // {0, 2}
    assert_eq!(expected, BTreeSet::from([0, 2]), "fixture sanity");

    let mut collector = VecCollector::default();
    let boolean = BooleanQuery::new().with_must([
        Clause::Term(TermQuery::new("body", "cat")),
        Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc)),
    ]);
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &boolean,
        &mut collector,
    )
    .unwrap();
    assert_eq!(collector.docs, expected.into_iter().collect::<Vec<_>>());
}

#[test]
fn match_no_docs_as_must_clause_makes_the_whole_boolean_query_match_nothing() {
    let (fields, doc, id, suffix, _max_doc) = open_segment();
    let doc_in = DocInput::open(&doc, &id, &suffix).expect("open .doc");

    let mut collector = VecCollector::default();
    let boolean = BooleanQuery::new().with_must([
        Clause::Term(TermQuery::new("body", "cat")),
        Clause::MatchNoDocs(MatchNoDocsQuery::new()),
    ]);
    search_boolean_query(
        &fields,
        Some(&doc_in),
        None,
        None,
        None,
        None,
        &boolean,
        &mut collector,
    )
    .unwrap();
    assert!(collector.docs.is_empty());
}

#[test]
fn match_all_docs_scored_variant_scores_every_live_doc_flat_1_0() {
    let (fields, _doc, _id, _suffix, max_doc) = open_segment();
    let deleted = [1, 3];
    let live_docs = live_docs_with_deletions(max_doc, &deleted);

    let boolean =
        BooleanQuery::new().with_must([Clause::MatchAllDocs(MatchAllDocsQuery::new(max_doc))]);

    let mut collector = lucene_search::TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        None,
        None,
        None,
        Some(&live_docs),
        None,
        &boolean,
        None,
        &mut collector,
    )
    .unwrap();
    let mut hits: Vec<(i32, f32)> = collector
        .top_docs()
        .iter()
        .map(|d| (d.doc_id, d.score))
        .collect();
    hits.sort_by_key(|&(doc_id, _)| doc_id);
    // Every score ties at 1.0, so the top 10 are the 10 lowest-numbered live docs.
    let expected: Vec<(i32, f32)> = (0..max_doc)
        .filter(|d| !deleted.contains(d))
        .take(10)
        .map(|d| (d, 1.0))
        .collect();
    assert_eq!(hits, expected);
}

#[test]
fn match_no_docs_scored_variant_produces_no_hits() {
    let (fields, _doc, _id, _suffix, _max_doc) = open_segment();

    let boolean = BooleanQuery::new().with_must([Clause::MatchNoDocs(MatchNoDocsQuery::new())]);

    let mut collector = lucene_search::TopDocsCollector::new(10);
    search_boolean_query_scored(
        &fields,
        None,
        None,
        None,
        None,
        None,
        &boolean,
        None,
        &mut collector,
    )
    .unwrap();
    assert!(collector.top_docs().is_empty());
}
