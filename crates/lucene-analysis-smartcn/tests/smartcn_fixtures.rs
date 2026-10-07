//! M12 T12.2: smartcn against Lucene 10.5.0.
//!
//! `fixtures/src/GenAnalysisSmartcn.java` writes `analysis_smartcn/`:
//! `HMMChineseTokenizer` over `corpus/analysis-chinese.txt` and 200 seeded
//! stress lines (`tok_hmm.tsv`), `SmartChineseAnalyzer` and the
//! `hmmChinese` factory over the corpus (`c_*.tsv`), `WordDictionary` and
//! `BigramDictionary` lookups (`words.tsv`, `bigrams.tsv`) and
//! `HHMMSegmenter.process` per corpus line (`paths.tsv`). Every row must be
//! equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::sync::Arc;

use lucene_analysis::factory::CustomAnalyzer;
use lucene_analysis::{Analyzer, CharArraySet};
use lucene_analysis_smartcn::hhmm::{BigramDictionary, HHMMSegmenter, WordDictionary};
use lucene_analysis_smartcn::tokenizer::hmm_chinese_tokenizer;
use lucene_analysis_smartcn::SmartChineseAnalyzer;
use support::{chain, comps, corpus, data_dir, esc, unesc};

fn read(file: &str) -> String {
    std::fs::read_to_string(data_dir("analysis_smartcn") + file).unwrap()
}

fn units(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

#[test]
fn word_dictionary_lookups_match_lucene() {
    let d = WordDictionary::get_instance();
    let mut n = 0;
    for line in read("words.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let w = units(&unesc(f[0]));
        let prefix = d.get_prefix_match(&w, 0);
        let eq = prefix >= 0 && d.is_equal(&w, prefix);
        let actual = format!("{}\t{}\t{}", d.get_frequency(&w), prefix, u8::from(eq));
        assert_eq!(actual, f[1..].join("\t"), "{}", f[0]);
        n += 1;
    }
    assert!(n > 5_000, "{n}");
}

#[test]
fn bigram_lookups_match_lucene() {
    let d = BigramDictionary::get_instance();
    let mut found = 0;
    for line in read("bigrams.tsv").lines() {
        let (pair, freq) = line.split_once('\t').unwrap();
        let actual = d.get_frequency(&units(&unesc(pair)));
        assert_eq!(actual.to_string(), freq, "{pair}");
        found += usize::from(actual > 0);
    }
    assert!(found > 100, "{found}");
}

#[test]
fn segmenter_paths_match_lucene() {
    let lines = corpus("analysis-chinese.txt");
    let seg = HHMMSegmenter::default();
    let mut actual = Vec::new();
    for (ln, line) in lines.iter().enumerate() {
        for t in seg.process(&units(line)).unwrap() {
            actual.push(format!(
                "{ln}\t{}\t{}\t{}\t{}\t{}",
                esc(&String::from_utf16_lossy(&t.char_array)),
                t.start_offset,
                t.end_offset,
                t.word_type,
                t.weight
            ));
        }
    }
    let text = read("paths.tsv");
    let expected: Vec<&str> = text.lines().collect();
    for (a, e) in actual.iter().zip(&expected) {
        assert_eq!(a, e);
    }
    assert_eq!(actual.len(), expected.len());
}

fn build(name: &str) -> Option<Analyzer> {
    Some(match name {
        "tok_hmm" => chain(|| comps(hmm_chinese_tokenizer())),
        "c_smart_default" => Analyzer::new(SmartChineseAnalyzer::default()),
        "c_smart_nostop" => Analyzer::new(SmartChineseAnalyzer::with_default_stop_words(false)),
        "c_smart_custom_stop" => Analyzer::new(SmartChineseAnalyzer::new(Some(Arc::new(
            CharArraySet::from_words(["的", "了", "是", ","], false),
        )))),
        "c_factory_hmm_lower" => {
            lucene_analysis_smartcn::register_factories().unwrap();
            CustomAnalyzer::builder()
                .with_tokenizer("hmmChinese", &[])
                .unwrap()
                .add_token_filter("lowercase", &[])
                .unwrap()
                .build()
                .unwrap()
                .into_analyzer()
        }
        _ => return None,
    })
}

#[test]
fn chains_match_lucene() {
    let corpus = corpus("analysis-chinese.txt");
    let mut lines = corpus.clone();
    lines.extend(read("stress.txt").lines().map(unesc));
    let names = support::fixture_names("analysis_smartcn")
        .into_iter()
        .filter(|n| n.starts_with("tok_") || n.starts_with("c_"));
    let (tok, rest): (std::collections::BTreeSet<String>, _) = names.partition(|n| n == "tok_hmm");
    assert_eq!(
        support::check_chains_over("analysis_smartcn", &lines, &tok, build),
        1
    );
    assert_eq!(
        support::check_chains_over("analysis_smartcn", &corpus, &rest, build),
        4
    );
}
