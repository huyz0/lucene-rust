//! Differential test against real Lucene's `StandardAnalyzer`
//! (StandardTokenizer + LowerCaseFilter + StopFilter): six cases covering
//! the position-increment-preservation rule when stopwords are removed
//! (mid-sentence, leading, trailing, consecutive, all-stopwords) plus a
//! mixed-case/punctuation sentence exercising the tokenizer, lowercasing,
//! and stopword removal together ("The" is itself a stopword once
//! lowercased). Regenerate with fixtures/src/GenAnalysis.java.

use lucene_analysis::Analyzer;
use std::collections::HashSet;

fn dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/analysis/").to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenAnalysis)");
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

fn expected_tokens(m: &Manifest, case: &str) -> Vec<(String, i32, i32, i32)> {
    let raw = m.get(&format!("{case}.tokens"));
    if raw.is_empty() {
        return Vec::new();
    }
    raw.split(';')
        .map(|entry| {
            let mut parts = entry.split(':');
            let term = parts.next().unwrap().to_string();
            let pos_inc: i32 = parts.next().unwrap().parse().unwrap();
            let offsets = parts.next().unwrap();
            let (start, end) = offsets.split_once(',').unwrap();
            (term, pos_inc, start.parse().unwrap(), end.parse().unwrap())
        })
        .collect()
}

fn actual_tokens(text: &str, stopwords: &HashSet<String>) -> Vec<(String, i32, i32, i32)> {
    Analyzer::standard(Some(stopwords))
        .analyze(text)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect()
}

#[test]
fn matches_real_standard_analyzer_across_all_cases() {
    let m = Manifest::load();
    let stopwords: HashSet<String> = ["the", "a", "of"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    for case in ["case1", "case2", "case3", "case4", "case5", "case6"] {
        let text = m.get(&format!("{case}.text"));
        let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
        let expected = expected_tokens(&m, case);
        assert_eq!(
            expected.len(),
            expected_count,
            "case {case}: manifest count mismatch"
        );

        let actual = actual_tokens(text, &stopwords);
        assert_eq!(
            actual, expected,
            "case {case} (text={text:?}) diverged from real Lucene"
        );
    }
}
