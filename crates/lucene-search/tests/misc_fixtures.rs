//! M14 T14.3 `misc`, differentially against Lucene 10.5.0:
//! `fixtures/src/GenMisc.java`'s `misc/high_freq.tsv`.
//!
//! The index is one Lucene wrote over three commits (three segments); each
//! `HighFreqTerms.getHighFreqTerms` configuration Java ran (both comparators,
//! the whole index and one field, more terms asked for than exist) must return
//! the same terms, in the same order, with the same frequencies.

use std::collections::BTreeMap;

use lucene_search::directory_reader::DirectoryReader;
use lucene_search::misc::{get_high_freq_terms, TermOrder};
use lucene_search::Error;
use lucene_store::FsDirectory;

fn data() -> std::path::PathBuf {
    std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/misc")
}

/// `config`, `field` (`-` for the whole index), `numTerms`, and the rows Java
/// printed for it, each `term`, `totalTF`, `docFreq`.
type Expected = BTreeMap<(String, String, usize), Vec<(String, i64, i32)>>;

fn expected() -> Expected {
    let tsv = std::fs::read_to_string(data().join("high_freq.tsv")).unwrap();
    let mut out: Expected = BTreeMap::new();
    for line in tsv.lines() {
        let c: Vec<&str> = line.split('\t').collect();
        assert_eq!(c.len(), 6, "bad row {line:?}");
        let key = (c[0].to_string(), c[1].to_string(), c[2].parse().unwrap());
        out.entry(key).or_default().push((
            c[3].to_string(),
            c[4].parse().unwrap(),
            c[5].parse().unwrap(),
        ));
    }
    out
}

#[test]
fn high_freq_terms_match_lucene() {
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("high_freq"))).unwrap();
    let expected = expected();
    assert_eq!(
        expected.len(),
        6,
        "every configuration Java ran is compared"
    );
    for ((config, field, num), rows) in &expected {
        let order = match config.as_str() {
            "doc-freq" => TermOrder::DocFreq,
            "total-tf" => TermOrder::TotalTermFreq,
            other => panic!("unknown config {other}"),
        };
        let field = (field != "-").then_some(field.as_str());
        let got = get_high_freq_terms(&reader, *num, field, order).unwrap();
        let got: Vec<(String, i64, i32)> = got
            .iter()
            .map(|t| (t.term_text(), t.total_term_freq, t.doc_freq))
            .collect();
        assert_eq!(&got, rows, "{config} {field:?} {num}");
    }
}

#[test]
fn a_named_field_that_does_not_exist_is_refused() {
    let reader = DirectoryReader::open(&FsDirectory::open(data().join("high_freq"))).unwrap();
    let err = get_high_freq_terms(&reader, 5, Some("nope"), TermOrder::DocFreq).unwrap_err();
    assert!(
        matches!(&err, Error::IllegalState(m) if m == "field nope not found"),
        "{err:?}"
    );
}

/// `GenMiscSweetSpot`: `misc/sweet_spot.tsv`'s `F` (a function's value), `T`
/// (`toString`) and `S` (a hit's score) rows, for the default and a tuned
/// `SweetSpotSimilarity`.
mod sweet_spot {
    use super::data;
    use lucene_search::directory_reader::DirectoryReader;
    use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};
    use lucene_search::misc::SweetSpotSimilarity;
    use lucene_search::query::{BooleanQuery, Clause, TermQuery};
    use lucene_search::similarities::TfIdfSimilarity;
    use lucene_store::FsDirectory;

    fn make(config: &str) -> SweetSpotSimilarity {
        let mut s = SweetSpotSimilarity::default();
        if config == "tuned" {
            s.set_baseline_tf_factors(0.5, 2.0);
            s.set_hyperbolic_tf_factors(0.5, 3.0, 1.7, 6.0);
            s.set_length_norm_factors(3, 9, 0.7);
        }
        s
    }

    fn query(q: &str) -> BooleanQuery {
        let mut b = BooleanQuery::new();
        for part in q.split(' ') {
            let (f, t) = part.split_once(':').unwrap_or(("body", part));
            b.should.push(Clause::Term(TermQuery::new(f, t)));
        }
        b
    }

    #[test]
    fn functions_strings_and_scores_match_lucene() {
        let tsv = std::fs::read_to_string(data().join("sweet_spot.tsv")).unwrap();
        let reader = DirectoryReader::open(&FsDirectory::open(data().join("high_freq"))).unwrap();
        let opened = reader.open_segments().unwrap();
        let segments = opened.as_open_segments();
        let owned = reader.field_norms_by_field(&["body".to_string(), "title".to_string()]);
        let norms: Vec<SegmentNorms<'_, '_>> = owned.iter().map(Some).collect();
        let (mut functions, mut hits, mut strings) = (0, 0, 0);
        // The hits of each (config, query) in rank order, as Java listed them.
        let mut ranked: std::collections::BTreeMap<(String, String), Vec<(i32, u32)>> =
            Default::default();
        for line in tsv.lines() {
            let c: Vec<&str> = line.split('\t').collect();
            let s = make(c[1]);
            match c[0] {
                "F" => {
                    let bits = |v: f32| v.to_bits().to_string();
                    let got = match c[2] {
                        "baselineTf" => bits(s.baseline_tf(c[3].parse().unwrap())),
                        "hyperbolicTf" => bits(s.hyperbolic_tf(c[3].parse().unwrap())),
                        "tf" => bits(TfIdfSimilarity::tf(&s, c[3].parse().unwrap())),
                        "lengthNorm" => bits(s.length_norm(c[3].parse().unwrap())),
                        other => panic!("unknown function {other}"),
                    };
                    assert_eq!(got, c[4], "{line}");
                    functions += 1;
                }
                "T" => {
                    assert_eq!(s.to_string(), c[2], "{line}");
                    strings += 1;
                }
                "S" => {
                    ranked
                        .entry((c[1].to_string(), c[2].to_string()))
                        .or_default()
                        .push((c[4].parse().unwrap(), c[5].parse::<i32>().unwrap() as u32));
                    hits += 1;
                }
                other => panic!("unknown row {other}"),
            }
        }
        assert!(functions == 84 && strings == 2 && hits == 72);
        for ((config, q), expected) in &ranked {
            let sim = make(config);
            let mut searcher = IndexSearcher::new(&segments, &norms).unwrap();
            searcher.set_similarity(&sim);
            let td = searcher.search(&query(q), 10).unwrap();
            let got: Vec<(i32, u32)> = td
                .score_docs
                .iter()
                .map(|d| (d.doc, d.score.to_bits()))
                .collect();
            assert_eq!(&got, expected, "{config} {q}");
        }
    }
}
