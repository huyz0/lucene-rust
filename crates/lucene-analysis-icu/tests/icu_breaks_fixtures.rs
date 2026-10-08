//! `RuleBasedBreakIterator` over ICU's character, line, sentence and title
//! rules against ICU4J 77.1 (`fixtures/src/GenAnalysisIcuBreaks.java`):
//! every boundary, rule status and status vector over the analysis-icu
//! corpus and 200 seeded strings, equal.

use lucene_analysis_icu::icu4j::rbbi::{RuleBasedBreakIterator, DONE};

fn dir() -> String {
    format!(
        "{}/../../fixtures/data/analysis_icu_breaks",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn texts() -> Vec<Vec<u16>> {
    std::fs::read_to_string(format!("{}/breaks.txt", dir()))
        .unwrap()
        .lines()
        .map(|l| {
            l.split(' ')
                .filter(|u| !u.is_empty())
                .map(|u| u16::from_str_radix(u, 16).unwrap())
                .collect()
        })
        .collect()
}

#[test]
fn boundaries_match_icu4j() {
    let texts = texts();
    for name in ["char", "line", "sent", "title", "line_loose_cj", "sent_el"] {
        let brk = std::fs::read(format!("{}/{name}.brk", dir())).unwrap();
        let mut bi = RuleBasedBreakIterator::from_compiled_rules(&brk).unwrap();
        let expected = std::fs::read_to_string(format!("{}/{name}.tsv", dir())).unwrap();
        let mut n = 0;
        for (text, want) in texts.iter().zip(expected.lines()) {
            bi.set_text(text);
            let mut got = Vec::new();
            let mut b = bi.first();
            while b != DONE {
                let vec: Vec<String> = bi
                    .get_rule_status_vec()
                    .iter()
                    .map(|v| v.to_string())
                    .collect();
                got.push(format!("{b}:{}:{}", bi.get_rule_status(), vec.join(",")));
                b = bi.next();
            }
            assert_eq!(
                got.join(" "),
                want,
                "{name}: {:?}",
                String::from_utf16_lossy(text)
            );
            n += 1;
        }
        assert_eq!(n, texts.len(), "{name}");
    }
}
