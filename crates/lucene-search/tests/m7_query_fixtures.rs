//! M7's query and scoring half, differentially against Lucene 10.5.0:
//! `fixtures/src/GenM7Queries.java`.
//!
//! One three-segment, index-sorted index with deletions, searched with every
//! query M7 brings into the scorer tree -- synonyms, BM25F, n-gram phrases,
//! explicit phrase positions, multi-phrases, the multi-term rewrite methods,
//! blended terms, Indri, log-odds fusion, Bayesian calibration, doc-values and
//! index-sort ranges, multi-dimensional and 4/16-byte points, vector
//! similarity thresholds and KNN as a clause -- each under BM25 and, for the
//! scoring ones, `ClassicSimilarity` and a DFR similarity. Every top-20 hit
//! and its score bits must be Java's.

mod m7grammar;

use std::collections::HashMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::multi_segment::{
    search_boolean_query_multi_segment_with_deadline,
    search_boolean_query_multi_segment_with_similarity, OpenSegment,
};
use lucene_search::query::Clause;
use lucene_store::FsDirectory;
use m7grammar::{knn_clause, query, sim, Ctx, VectorFiles, QVEC};

fn data(path: &str) -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/data")
        .join(path)
}

fn want(hits: &str) -> Vec<(i32, u32)> {
    hits.split(',')
        .filter(|s| !s.is_empty())
        .map(|h| {
            let (d, b) = h.split_once(':').unwrap();
            (d.parse().unwrap(), u32::from_str_radix(b, 16).unwrap())
        })
        .collect()
}

#[test]
fn m7_queries_match_lucene_bit_for_bit() {
    let cases = check_fixture("m7_queries_index", &QVEC, 20, true);
    assert!(cases >= 200, "{cases} searches");
}

/// The vector queries over graphs large enough for filtered
/// (`FilteredHnswGraphSearcher`), patient and seeded walks to happen.
#[test]
fn m7_knn_queries_match_lucene_bit_for_bit() {
    let cases = check_fixture("m7_knn_index", &QVEC8, 50, false);
    assert!(cases >= 16, "{cases} searches");
}

/// Doc-values ranges over fields with a skip index (`SkipBlockRangeIterator`,
/// `DocValuesRangeIterator`), several skip blocks a segment.
#[test]
fn m7_doc_values_skip_ranges_match_lucene_bit_for_bit() {
    // The ranges must run on the skip-index path, not the plain-column one.
    let reader = DirectoryReader::open(&FsDirectory::open(data("m7_dv_index"))).unwrap();
    for seg in reader.segment_readers() {
        for field in ["sk", "msk"] {
            let number = seg.field_infos().field_by_name(field).unwrap().number;
            assert!(
                seg.doc_values_skip_index(number).unwrap().is_some(),
                "{field}"
            );
        }
        let body = seg.field_infos().field_by_name("body").unwrap().number;
        assert!(seg.doc_values_skip_index(body).unwrap().is_none());
    }
    let cases = check_fixture("m7_dv_index", &QVEC, 50, false);
    assert!(cases >= 9, "{cases} searches");
}

/// `GenM7Queries.QVEC8`.
const QVEC8: [f32; 8] = [0.1, -0.3, 0.25, 0.6, -0.05, 0.4, -0.2, 0.15];

/// Runs every search `name`'s `searches.tsv` records and fails on the first
/// difference from Lucene's hits; returns how many ran.
fn check_fixture(name: &str, qvec: &[f32], top: usize, estimator: bool) -> usize {
    let dir = data(name);
    let text = std::fs::read_to_string(dir.join("searches.tsv"))
        .expect("run scripts/gen-fixtures.sh --only GenM7Queries");
    let reader = DirectoryReader::open(&FsDirectory::open(&dir)).unwrap();
    assert!(reader.segment_readers().len() >= 2);
    assert!(
        reader
            .segment_readers()
            .iter()
            .filter(|s| s.live_docs().is_some())
            .count()
            >= 2
    );
    let mut opened = reader.open_segments().unwrap();
    opened.open_points().unwrap();
    let segments: Vec<OpenSegment<'_>> = opened.as_open_segments();
    let owned =
        reader.field_norms_by_field(&["body".to_string(), "title".to_string(), "gram".to_string()]);
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let vectors = VectorFiles::read(&dir, &reader);
    let knn = |op: &str, field: &str, arg: &str, clause: Option<Clause>| -> Option<Clause> {
        knn_clause(
            op, field, arg, clause, &reader, &segments, &norms, &vectors, qvec,
        )
    };
    let ctx = Ctx { knn: &knn };

    let (mut cases, mut skipped, mut failures) = (0, 0, Vec::new());
    for line in text.lines() {
        let [name, q, hits] = line.split('\t').collect::<Vec<_>>()[..] else {
            panic!("bad line {line}");
        };
        let Some(bq) = query(q, &ctx) else {
            skipped += 1;
            continue;
        };
        let similarity = sim(name);
        let got = match search_boolean_query_multi_segment_with_similarity(
            &segments,
            &bq,
            &norms,
            top,
            similarity.as_ref(),
        ) {
            Ok(got) => got,
            Err(e) => {
                failures.push(format!("{name} [{q}]: {e}"));
                continue;
            }
        };
        let got: Vec<(i32, u32)> = got.iter().map(|h| (h.doc_id, h.score.to_bits())).collect();
        cases += 1;
        let want = want(hits);
        if got != want {
            failures.push(format!("{name} [{q}]\n  rust {got:x?}\n  java {want:x?}"));
        }
        if name == "bm25" {
            // `TimeLimitingBulkScorer`: a deadline that never comes changes
            // nothing; one already past stops before the first window.
            let far = std::time::Instant::now() + std::time::Duration::from_secs(3600);
            let (timed, cut) =
                search_boolean_query_multi_segment_with_deadline(&segments, &bq, &norms, top, far)
                    .unwrap();
            let timed: Vec<(i32, u32)> = timed
                .iter()
                .map(|h| (h.doc_id, h.score.to_bits()))
                .collect();
            if timed != want || cut {
                failures.push(format!(
                    "deadline [{q}]\n  rust {timed:x?}\n  java {want:x?}"
                ));
            }
            let past = std::time::Instant::now();
            let (none, cut) =
                search_boolean_query_multi_segment_with_deadline(&segments, &bq, &norms, top, past)
                    .unwrap();
            assert!(none.is_empty() && cut, "an expired deadline scores nothing");
        }
    }
    // `BayesianScoreEstimator.estimate` over `body`.
    let est = if estimator {
        std::fs::read_to_string(dir.join("estimator.tsv")).unwrap()
    } else {
        String::new()
    };
    let mut estimates = 0;
    for line in est.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let (n, tokens, seed) = (
            f[0].parse().unwrap(),
            f[1].parse().unwrap(),
            f[2].parse().unwrap(),
        );
        let p =
            lucene_search::bayesian_estimator::estimate(&segments, &norms, "body", n, tokens, seed)
                .unwrap();
        let bits = |s: &str| u32::from_str_radix(s, 16).unwrap();
        let got = (p.alpha.to_bits(), p.beta.to_bits(), p.base_rate.to_bits());
        let want = (bits(f[3]), bits(f[4]), bits(f[5]));
        if got != want {
            failures.push(format!(
                "estimate {line}\n  rust {got:x?}\n  java {want:x?}"
            ));
        }
        estimates += 1;
    }
    assert_eq!(estimates, if estimator { 3 } else { 0 });
    eprintln!("{cases} searches compared, {skipped} skipped");
    assert_eq!(skipped, 0, "every recorded search runs");
    assert!(
        failures.is_empty(),
        "{} searches differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    cases
}
