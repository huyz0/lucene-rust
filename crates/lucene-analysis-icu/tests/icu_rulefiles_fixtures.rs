//! `ICUTokenizerFactory`'s `rulefiles` against Lucene 10.5.0
//! (`fixtures/src/GenAnalysisIcuRules.java`): a `CustomAnalyzer` reading
//! Lucene's rule files from a directory, over the analysis-icu corpus, every
//! token row equal -- terms, offsets, types, position increments -- and, for
//! the configurations Lucene refuses, the same exception class.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use lucene_analysis::factory::CustomAnalyzer;
use support::{corpus, data_dir, normalise_expected};

const DIR: &str = "analysis_icu_rules";

fn read(file: &str) -> String {
    std::fs::read_to_string(data_dir(DIR) + file).unwrap()
}

#[test]
fn rulefiles_chains_match_lucene() {
    lucene_analysis_icu::register_factories().unwrap();
    let lines = corpus("analysis-icu.txt");
    let src = format!(
        "{}/../../fixtures/corpus/analysis-icu-rules/",
        env!("CARGO_MANIFEST_DIR")
    );
    let mut compared = 0;
    let mut refused = 0;
    for spec in read("rulefiles_specs.tsv").lines() {
        let (name, spec) = spec.split_once('\t').unwrap();
        let mut parts = spec.split(',');
        let factory = parts.next().unwrap();
        // the arguments: `rulefiles=Latn:a.rbbi_COMMA_Cyrl:b.rbbi`
        let args: Vec<(String, String)> = parts
            .map(|a| {
                let (k, v) = a.split_once('=').unwrap();
                (k.to_string(), v.replace("_COMMA_", ","))
            })
            .collect();
        let flat: Vec<&str> = args
            .iter()
            .flat_map(|(k, v)| [k.as_str(), v.as_str()])
            .collect();
        let built = CustomAnalyzer::builder_with_dir(&src)
            .and_then(|b| b.with_tokenizer(factory, &flat))
            .and_then(|b| b.build());
        let expected: Vec<String> = read(&format!("{name}.tsv"))
            .lines()
            .map(normalise_expected)
            .collect();
        match built {
            Ok(a) => {
                let a = a.into_analyzer();
                let mut actual = Vec::new();
                for (ln, line) in lines.iter().enumerate() {
                    support::analyze_line(&a, ln, line, &mut actual);
                }
                assert_eq!(actual, expected, "{name}: {spec}");
                compared += 1;
            }
            Err(e) => {
                assert_eq!(
                    vec![format!("B\t{}", e.java_class())],
                    expected,
                    "{name}: {spec}"
                );
                refused += 1;
            }
        }
    }
    assert_eq!(compared, 8, "chains compared");
    assert_eq!(refused, 4, "chains refused");
}
