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
