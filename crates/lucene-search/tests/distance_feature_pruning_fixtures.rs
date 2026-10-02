//! The distance feature queries' pruning across segments, differentially
//! against Lucene 10.5.0: `fixtures/src/GenDistanceFeaturePruning.java`
//! indexed four segments where a sixth of the documents sit exactly at the
//! origin -- so a top-n queue fills with hits scoring exactly the boost --
//! and recorded each top-n search's hits *and* its `totalHits` value and
//! relation. The count is what the pruning decides: how many documents the
//! scorer still hands the collector after `TopScoreDocCollector` pushes
//! `Math.nextUp(worst kept score)`, which it does after a competitive hit and
//! again when every leaf starts (`setScorer`). It also covers the geo query
//! with a NaN pivot, which Lucene never prunes.

#![allow(clippy::arithmetic_side_effects)]

use lucene_search::collector::TotalHitsRelation;
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::document::geo::lat_lon_point;
use lucene_search::document::{self as dq, long_field, DocumentQuery};
use lucene_search::multi_segment::OpenSegment;
use lucene_store::FsDirectory;

fn root() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data/distance_feature_pruning")
}

/// `GenDistanceFeaturePruning.scored(searcher, query, n)`.
fn scored(segments: &[OpenSegment<'_>], q: &dyn DocumentQuery, n: usize) -> String {
    let td = dq::search_top_docs(segments, q, n).unwrap();
    let rel = match td.total_hits.relation {
        TotalHitsRelation::EqualTo => "EQUAL_TO",
        TotalHitsRelation::GreaterThanOrEqualTo => "GREATER_THAN_OR_EQUAL_TO",
    };
    let hits: Vec<String> = td
        .score_docs
        .iter()
        .map(|s| format!("{}:{:x}", s.doc_id, s.score.to_bits()))
        .collect();
    let hits = hits.join(",");
    let hits = if td.score_docs.len() > 100 {
        format!("#{}", java_string_hash(&hits))
    } else {
        hits
    };
    format!("S\t{}\t{rel}\t{hits}", td.total_hits.value)
}

/// `String.hashCode()` of an ASCII string.
fn java_string_hash(s: &str) -> i32 {
    s.bytes()
        .fold(0i32, |h, b| h.wrapping_mul(31).wrapping_add(i32::from(b)))
}

/// The `totalHits` value of a `scored` line.
fn total(line: &str) -> &str {
    line.split('\t').nth(1).unwrap()
}

fn run(segments: &[OpenSegment<'_>], a: &[&str]) -> String {
    let weight: f32 = a[2].parse().unwrap();
    let n: usize = a[5].parse().unwrap();
    let q = match a[0] {
        "long" => long_field::new_distance_feature_query(
            a[1],
            weight,
            a[3].parse().unwrap(),
            a[4].parse().unwrap(),
        )
        .unwrap(),
        "geo" => {
            let (lat, lon) = a[3].split_once(',').unwrap();
            lat_lon_point::new_distance_feature_query(
                a[1],
                weight,
                lat.parse().unwrap(),
                lon.parse().unwrap(),
                a[4].parse().unwrap(),
            )
            .unwrap()
        }
        other => panic!("unknown kind {other}"),
    };
    scored(segments, q.as_ref(), n)
}

#[test]
fn distance_feature_pruning_counts_what_lucene_counts() {
    let reader =
        DirectoryReader::open(&FsDirectory::open(root().join("index"))).expect("open index");
    let opened = reader.open_segments().expect("open segments");
    let segments = opened.as_open_segments();
    assert_eq!(segments.len(), 4, "segments");
    let text = std::fs::read_to_string(root().join("queries.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenDistanceFeaturePruning");
    let mut failures = Vec::new();
    let mut n = 0;
    let mut pruned = 0;
    for line in text.lines() {
        let (query, want) = line.split_once("\t=>\t").unwrap();
        let a: Vec<&str> = query.split('\t').collect();
        let got = run(&segments, &a);
        // A NaN pivot scores every document NaN. Lucene never prunes it
        // (`Math.nextUp(NaN) > minCompetitiveScore` is false), which is what
        // the count checks. How a full queue of NaN hits churns and which
        // relation it reports are `TopScoreDocCollector`'s, not the query's,
        // and this port's collector orders NaN differently (`docs/parity.md`,
        // the `TopScoreDocCollector` row), so only the count is compared.
        let nan = a[4] == "NaN";
        if (nan && total(&got) != total(want)) || (!nan && got != want) {
            let short = |s: &str| s.chars().take(200).collect::<String>();
            failures.push(format!(
                "{query}\n  java: {}\n  rust: {}",
                short(want),
                short(&got)
            ));
        }
        pruned += usize::from(want.contains("GREATER_THAN_OR_EQUAL_TO"));
        n += 1;
    }
    assert!(
        failures.is_empty(),
        "{} of {n} queries differ:\n{}",
        failures.len(),
        failures[..failures.len().min(25)].join("\n")
    );
    assert!(n > 200, "{n} queries");
    assert!(pruned > 100, "only {pruned} queries pruned");
}
