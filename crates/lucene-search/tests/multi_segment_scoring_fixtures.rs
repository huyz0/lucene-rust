//! **Reader-wide statistics, proven against real Lucene on a real two-segment
//! index.**
//!
//! `IndexSearcher` computes `TermStats` and `FieldStats` once across the whole
//! reader and hands the same numbers to every leaf: `docFreq`/`docCount` for the
//! idf, and `sumTotalTermFreq / docCount` for `BM25Similarity`'s `avgdl`. A port
//! that derives either from the leaf it is scoring gives the same document a
//! different score depending on which segment it landed in, and the merged
//! top-k then fills from whichever segment makes the term look rarest or its
//! documents look shortest.
//!
//! Every other scoring fixture in this tree is a single segment, where per-leaf
//! and reader-wide are the same number by construction, so none of them can see
//! it -- which is why b13's F-26 (`avgdl` derived per leaf) survived two sweep
//! batches. `fixtures/src/GenMultiSegmentScoring.java` builds the shape that
//! can: two committed segments whose own `avgdl` values are 1.75 and 40.0
//! against a reader-wide 20.875, and a term (`fox`) whose `docFreq` is 1 of 4
//! in one leaf and 3 of 4 in the other.
//!
//! Comparison is on raw `f32` bits, like `bm25_scoring_fixtures.rs`: a
//! tolerance would not separate the two candidate answers at the top-k
//! boundary, and it is exactly the last-bit divergences that reorder hits.

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{
    search_boolean_query_multi_segment, search_term_query_multi_segment,
    search_term_query_multi_segment_concurrent,
};
use lucene_search::{BooleanQuery, Clause, TermQuery};
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
            .expect("run scripts/gen-fixtures.sh first (GenMultiSegmentScoring)");
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

    fn bits(&self, key: &str) -> u32 {
        self.get(key).parse().unwrap()
    }

    /// Real Lucene's recorded hits, in **global** doc-id space, reconstructed
    /// from `Float.floatToIntBits` so nothing passes through decimal rounding.
    fn lucene_hits(&self, key: &str) -> Vec<(i32, f32)> {
        let raw = self.get(&format!("{key}.bits"));
        if raw.is_empty() {
            return Vec::new();
        }
        raw.split(',')
            .map(|pair| {
                let (doc, bits) = pair.split_once(':').expect("doc:bits pair");
                (
                    doc.parse().unwrap(),
                    f32::from_bits(bits.parse::<u32>().unwrap()),
                )
            })
            .collect()
    }
}

fn assert_same_as_lucene(
    what: &str,
    got: &[lucene_search::collector::ScoreDoc],
    expected: &[(i32, f32)],
) {
    let got_pairs: Vec<(i32, f32)> = got.iter().map(|h| (h.doc_id, h.score)).collect();
    assert_eq!(
        got_pairs.len(),
        expected.len(),
        "{what}: hit count differs -- got {got_pairs:?}, Lucene {expected:?}"
    );
    for ((got_doc, got_score), (exp_doc, exp_score)) in got_pairs.iter().zip(expected) {
        assert_eq!(
            got_doc, exp_doc,
            "{what}: doc order differs -- got {got_pairs:?}, Lucene {expected:?}"
        );
        assert_eq!(
            got_score.to_bits(),
            exp_score.to_bits(),
            "{what}: doc {got_doc} scored {got_score} ({:#x}), real Lucene {exp_score} ({:#x})",
            got_score.to_bits(),
            exp_score.to_bits()
        );
    }
}

/// The fixture is only worth anything if the two segments really do disagree
/// about `avgdl`; pin that, so a regenerated corpus that happens to even out
/// fails here rather than silently making every test below vacuous.
#[test]
fn the_two_segments_own_avgdl_values_are_far_apart_and_neither_is_the_readers() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    assert_eq!(reader.segment_readers().len(), 2, "two-segment fixture");

    for (i, seg) in reader.segment_readers().iter().enumerate() {
        let (stf, dc) = seg.field_stats("body").expect("body has a term dictionary");
        assert_eq!(
            stf,
            m.get(&format!("segment.{i}.sum_total_term_freq"))
                .parse::<i64>()
                .unwrap()
        );
        assert_eq!(
            dc,
            m.get(&format!("segment.{i}.doc_count"))
                .parse::<i32>()
                .unwrap()
        );
    }

    let reader_wide = reader.avg_field_length("body").expect("body has norms");
    assert_eq!(
        reader_wide.to_bits(),
        m.bits("avgdl.bits"),
        "reader-wide avgdl must be `IndexSearcher.fieldStats`' sumTotalTermFreq/docCount"
    );

    // `multi_segment::global_avg_field_length` is the same sum taken over
    // `OpenSegment`s instead of `SegmentReader`s, for callers that hold the
    // former. Two independent implementations of one Java method is exactly
    // where a silent drift would live, so pin them to each other -- and to the
    // real Lucene value above, so "both wrong the same way" cannot pass.
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let via_segments = lucene_search::multi_segment::global_avg_field_length(&segments, "body")
        .expect("body has a term dictionary in both segments");
    assert_eq!(
        via_segments.to_bits(),
        reader_wide.to_bits(),
        "global_avg_field_length and DirectoryReader::avg_field_length must agree"
    );
    assert_eq!(
        lucene_search::multi_segment::global_avg_field_length(&segments, "no_such_field"),
        None,
        "a field no segment has yields None, not a divide-by-zero"
    );
    let leaf0 = f32::from_bits(m.bits("segment.0.avgdl.bits"));
    let leaf1 = f32::from_bits(m.bits("segment.1.avgdl.bits"));
    assert!(
        leaf1 / leaf0 > 10.0,
        "the fixture's whole point is a large per-leaf spread; got {leaf0} and {leaf1}"
    );
    assert!(
        reader_wide != leaf0 && reader_wide != leaf1,
        "the reader-wide value must differ from both leaves' own, or this fixture proves nothing"
    );
}

/// `search_term_query_multi_segment` must reproduce real Lucene's `TopDocs`
/// exactly -- same hits, same global doc ids, same order, same score bits.
#[test]
fn multi_segment_term_query_scores_match_real_lucene_bit_for_bit() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let owned = reader.field_norms("body");
    let norms: Vec<Option<&FieldNorms<'_>>> = owned.iter().map(Option::as_ref).collect();
    assert!(
        norms.iter().all(Option::is_some),
        "both segments index `body` with norms"
    );

    for (key, term) in [("scoring.term.fox", "fox"), ("scoring.term.dog", "dog")] {
        let hits =
            search_term_query_multi_segment(&segments, &TermQuery::new("body", term), &norms, 20)
                .unwrap();
        assert_same_as_lucene(key, &hits, &m.lucene_hits(key));
    }
}

/// The concurrent fan-out must be the sequential one's output, bit for bit --
/// the module's own claim, now checked against real Lucene rather than against
/// itself.
#[test]
fn the_concurrent_fan_out_matches_real_lucene_too() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let owned = reader.field_norms("body");
    let norms: Vec<Option<&FieldNorms<'_>>> = owned.iter().map(Option::as_ref).collect();

    let hits = search_term_query_multi_segment_concurrent(
        &segments,
        &TermQuery::new("body", "fox"),
        &norms,
        20,
    )
    .unwrap();
    assert_same_as_lucene(
        "scoring.term.fox (concurrent)",
        &hits,
        &m.lucene_hits("scoring.term.fox"),
    );
}

/// A two-clause disjunction across both segments: `BooleanWeight`'s additive
/// combination on top of the same reader-wide statistics.
#[test]
fn multi_segment_boolean_query_scores_match_real_lucene_bit_for_bit() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();
    let owned = reader.field_norms_by_field(&["body".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();

    let query = BooleanQuery::new().with_should(vec![
        Clause::Term(TermQuery::new("body", "fox")),
        Clause::Term(TermQuery::new("body", "dog")),
    ]);
    let hits = search_boolean_query_multi_segment(&segments, &query, &norms, 20).unwrap();
    assert_same_as_lucene(
        "scoring.boolean.should.fox.dog",
        &hits,
        &m.lucene_hits("scoring.boolean.should.fox.dog"),
    );
}

/// Negative control for the fix: scoring each leaf with **its own** `avgdl` --
/// which is what `SegmentReader::field_norms` alone can compute, and what this
/// port did before c6 -- must *not* reproduce Lucene's scores. Without this,
/// the tests above could pass for the wrong reason (e.g. if norms were silently
/// unused and both answers collapsed to the unnormed fallback).
#[test]
fn per_leaf_avgdl_does_not_reproduce_lucenes_scores() {
    let m = Manifest::load();
    let reader = DirectoryReader::open(&FsDirectory::open(fixture_dir())).expect("open reader");
    let opened = reader.open_segments().expect("open postings");
    let segments = opened.as_open_segments();

    // `SegmentReader::field_norms` derives `avgdl` from that segment's own
    // counters -- correct for a single-segment index, wrong here.
    let owned: Vec<Option<FieldNorms<'_>>> = reader
        .segment_readers()
        .iter()
        .map(|seg| seg.field_norms("body"))
        .collect();
    let norms: Vec<Option<&FieldNorms<'_>>> = owned.iter().map(Option::as_ref).collect();

    let hits =
        search_term_query_multi_segment(&segments, &TermQuery::new("body", "fox"), &norms, 20)
            .unwrap();
    let expected = m.lucene_hits("scoring.term.fox");
    assert_eq!(hits.len(), expected.len(), "the hit set is unaffected");
    let differs = hits
        .iter()
        .zip(&expected)
        .any(|(got, (_, exp))| got.score.to_bits() != exp.to_bits());
    assert!(
        differs,
        "per-leaf avgdl scored identically to reader-wide avgdl -- the fixture's \
         1.75-vs-40.0 spread should make that impossible, so the norms are \
         probably not being applied at all"
    );
}
