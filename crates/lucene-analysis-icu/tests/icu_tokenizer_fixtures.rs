//! M12 T12.4: `ICUTokenizer` against Lucene 10.5.0 over ICU4J 77.1.
//!
//! `fixtures/src/GenAnalysisIcu.java` writes `analysis_icu/tok_*.tsv`: the
//! tokenizer in its four configurations over `corpus/analysis-icu.txt` and
//! 300 seeded stress lines (`tok_stress.txt`), every token's text, offsets,
//! position increment, type and script. Every row must be equal.
//!
//! One test, in the generator's order: the break engines are process-wide
//! and created on first use, as in Java (`icu4j/break_engines.rs`).
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use std::sync::Arc;

use lucene_analysis::{Analyzer, TokenStream};
use lucene_analysis_icu::{DefaultICUTokenizerConfig, ICUTokenizer, ScriptAttribute};
use support::{chain, comps, corpus, data_dir, esc, normalise_expected, unesc};

fn rows(a: &Analyzer, lines: &[String]) -> Vec<String> {
    let mut out = Vec::new();
    for (ln, line) in lines.iter().enumerate() {
        let mut ts = a.token_stream("f", line).unwrap();
        ts.reset().unwrap();
        while ts.increment_token().unwrap() {
            let at = ts.attributes();
            let script = at.custom::<ScriptAttribute>().unwrap();
            let mut reflected = String::new();
            lucene_analysis::CustomAttribute::reflect(script, &mut |_, _, v| {
                reflected = v.to_string()
            });
            out.push(format!(
                "T\t{ln}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                esc(at.term()),
                at.start_offset(),
                at.end_offset(),
                at.position_increment(),
                at.token_type(),
                script.code(),
                script.short_name(),
                reflected
            ));
        }
        ts.end().unwrap();
        let at = ts.attributes();
        out.push(format!(
            "E\t{ln}\t{}\t{}",
            at.start_offset(),
            at.end_offset()
        ));
        ts.close().unwrap();
    }
    out
}

#[test]
fn tokenizer_matches_lucene() {
    let mut lines = corpus("analysis-icu.txt");
    let stress = std::fs::read_to_string(data_dir("analysis_icu") + "tok_stress.txt").unwrap();
    lines.extend(stress.lines().map(unesc));
    assert_eq!(lines.len(), 79 + 300);
    for cjk in [true, false] {
        for myanmar in [true, false] {
            let name = format!(
                "tok_{}_{}",
                if cjk { "cjk" } else { "nocjk" },
                if myanmar { "mywords" } else { "mysyl" }
            );
            let a = chain(move || {
                comps(ICUTokenizer::with_config(Arc::new(
                    DefaultICUTokenizerConfig::new(cjk, myanmar),
                )))
            });
            let expected: Vec<String> =
                std::fs::read_to_string(data_dir("analysis_icu") + &name + ".tsv")
                    .unwrap()
                    .lines()
                    .map(normalise_expected)
                    .collect();
            let actual = rows(&a, &lines);
            for (i, (a, e)) in actual.iter().zip(&expected).enumerate() {
                assert_eq!(a, e, "{name}: row {i}");
            }
            assert_eq!(actual.len(), expected.len(), "{name}");
        }
    }
}
