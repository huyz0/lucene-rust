//! A phrase query across the boundary between two values of one multi-valued
//! field, checked against real Lucene's own answer.
//!
//! `fixtures/src/GenAnalysis.java` indexes the same values as repeated values
//! of one field through a real `IndexWriter` and records
//! `IndexSearcher.count` for each phrase; this runs the same phrases through
//! this port's `IndexWriter` -> `DirectoryReader` -> `search_phrase_query`
//! path and requires the same counts.
//!
//! The two cases here are the ones this writer can reproduce exactly: it
//! analyses with a plain `Analyzer::standard(None)`, so a case whose Lucene
//! analyzer had a stopword set would be indexed with different *terms*. The
//! stopword cases (`mv_trailing_stopwords`, `mv_stopwords_and_gap`) are
//! covered against the same ground truth one layer down, in
//! `lucene-index`'s `tests/multi_valued_fields.rs`, which drives
//! `invert_documents` with the fixture's own analyzer and compares every
//! position and offset.
//!
//! Lives in `lucene-search` because `lucene-index` must not depend on it (the
//! strictly downward dependency graph), the same reason
//! `index_writer_postings_fixtures.rs` does.

use lucene_codecs::field_infos::{FieldInfo, IndexOptions};
use lucene_codecs::stored_fields::{Document, FieldValue, StoredField};
use lucene_index::index_writer::IndexWriter;
use lucene_index::segment_info::LuceneVersion;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::{search_phrase_query, PhraseQuery, VecCollector};
use lucene_store::FsDirectory;
use lucene_util::test_support::TempDir;

fn version() -> LuceneVersion {
    LuceneVersion {
        major: 10,
        minor: 5,
        bugfix: 0,
    }
}

fn manifest() -> Vec<(String, String)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/analysis/manifest.properties"
    );
    std::fs::read_to_string(path)
        .expect("run scripts/gen-fixtures.sh --only GenAnalysis first")
        .lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn get<'a>(m: &'a [(String, String)], key: &str) -> &'a str {
    m.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("manifest key {key} missing"))
}

/// Indexes `values` as repeated values of `body` in one document and runs each
/// recorded phrase, requiring real Lucene's hit count.
fn check_case(case: &str) {
    let m = manifest();
    let values: Vec<&str> = get(&m, &format!("{case}.values")).split('|').collect();
    let position_gap: i32 = get(&m, &format!("{case}.position_increment_gap"))
        .parse()
        .unwrap();
    let offset_gap: i32 = get(&m, &format!("{case}.offset_gap")).parse().unwrap();
    let phrases = get(&m, &format!("{case}.phrases")).to_string();

    let path = TempDir::new(&format!("c40-{case}"));
    let dir = FsDirectory::open(&path);
    let fields = vec![
        FieldInfo::new("id", 0),
        FieldInfo::new("body", 1).with_index_options(IndexOptions::DocsAndFreqsAndPositions),
    ];
    let mut writer = IndexWriter::open(&dir, fields, "Lucene104", version()).unwrap();
    writer.set_position_increment_gap(position_gap);
    writer.set_offset_gap(offset_gap);
    writer.set_postings_field(Some("body")).unwrap();

    let mut doc_fields = vec![StoredField {
        field_number: 0,
        value: FieldValue::String(case.to_string()),
    }];
    for value in &values {
        doc_fields.push(StoredField {
            field_number: 1,
            value: FieldValue::String((*value).to_string()),
        });
    }
    writer
        .add_document(Document { fields: doc_fields })
        .unwrap();
    writer.commit().unwrap();

    let reader = DirectoryReader::open(&dir).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let seg = &segments[0];

    for entry in phrases.split(';') {
        let (name, expected) = entry.split_once('=').unwrap();
        let expected: usize = expected.parse().unwrap();
        // `GenAnalysis` records `{name, "t1 t2", slop}`; the name encodes the
        // words and slop, so recover them from the manifest's own phrase spec.
        let (words, slop) = phrase_spec(case, name);
        let mut collector = VecCollector::default();
        search_phrase_query(
            seg.fields,
            seg.doc_in,
            seg.pos_in,
            seg.pay_in,
            seg.live_docs,
            &PhraseQuery::new("body", words.iter().copied()).with_slop(slop),
            &mut collector,
        )
        .unwrap();
        assert_eq!(
            collector.docs.len(),
            expected,
            "{case}/{name}: phrase {words:?} at slop {slop} -- \
             real Lucene found {expected} hit(s), this port found {:?}",
            collector.docs
        );
    }
    std::fs::remove_dir_all(&path).ok();
}

/// The `{words, slop}` half of `GenAnalysis`' phrase specs, kept here rather
/// than in the manifest because the manifest records the *answer*, which is
/// the part that must come from Lucene.
fn phrase_spec(case: &str, name: &str) -> (Vec<&'static str>, u32) {
    match (case, name) {
        ("mv_default_gap", "across0") => (vec!["beta", "gamma"], 0),
        ("mv_default_gap", "within0") => (vec!["alpha", "beta"], 0),
        ("mv_default_gap", "reversedacross2") => (vec!["gamma", "beta"], 2),
        ("mv_gap_100", "across0") => (vec!["beta", "gamma"], 0),
        ("mv_gap_100", "across99") => (vec!["beta", "gamma"], 99),
        ("mv_gap_100", "across100") => (vec!["beta", "gamma"], 100),
        ("mv_gap_100", "within0") => (vec!["gamma", "delta"], 0),
        _ => panic!("unknown phrase spec {case}/{name}"),
    }
}

#[test]
fn phrases_across_a_default_gap_boundary_match_lucene() {
    check_case("mv_default_gap");
}

#[test]
fn a_position_increment_gap_stops_a_phrase_at_the_value_boundary() {
    check_case("mv_gap_100");
}
