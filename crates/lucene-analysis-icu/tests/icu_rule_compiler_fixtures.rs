//! The break rule compiler (`RuleBasedBreakIterator::from_rules`, ICU4J's
//! `new RuleBasedBreakIterator(String)`, what `ICUTokenizerFactory`'s
//! `rulefiles` runs) against ICU4J 77.1
//! (`fixtures/src/GenAnalysisIcuRules.java`): over Lucene's own rule files
//! and rule sets written for each rule feature, every boundary, rule status
//! and status vector of 360 texts equal; and for every rule set ICU4J
//! rejects, the same error code at the same line and column.

use lucene_analysis_icu::icu4j::rbbi::{RuleBasedBreakIterator, DONE};

fn dir() -> String {
    format!(
        "{}/../../fixtures/data/analysis_icu_rules",
        env!("CARGO_MANIFEST_DIR")
    )
}

fn texts() -> Vec<Vec<u16>> {
    std::fs::read_to_string(format!("{}/texts.txt", dir()))
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

/// Every boundary of `text` as `position:status:vector`, joined by spaces.
fn row(bi: &mut RuleBasedBreakIterator, text: &[u16]) -> String {
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
    got.join(" ")
}

/// The rule sets, by name.
fn names() -> Vec<String> {
    let mut names: Vec<String> = std::fs::read_dir(dir())
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".rbbi").map(str::to_string)
        })
        .collect();
    names.sort();
    names
}

/// ICU4J's `Error <code> at line <n> column <n>`, from its
/// `ERROR IllegalArgumentException: ...` line.
fn expected_error(tsv: &str) -> Option<String> {
    tsv.strip_prefix("ERROR ").map(|e| e.trim_end().to_string())
}

#[test]
fn compiled_rules_match_icu4j() {
    let texts = texts();
    let mut compiled = 0;
    let mut refused = 0;
    for name in names() {
        let rules = std::fs::read_to_string(format!("{}/{name}.rbbi", dir())).unwrap();
        let expected = std::fs::read_to_string(format!("{}/{name}.tsv", dir())).unwrap();
        let built = RuleBasedBreakIterator::from_rules(&rules);
        if let Some(java) = expected_error(&expected) {
            let err = built
                .err()
                .unwrap_or_else(|| panic!("{name}: ICU4J refuses: {java}"));
            let want = java
                .strip_prefix("IllegalArgumentException: ")
                .unwrap_or(&java);
            assert_eq!(err.message(), want, "{name}");
            refused += 1;
            continue;
        }
        let mut bi = match built {
            Ok(bi) => bi,
            Err(e) => panic!("{name}: {e}"),
        };
        for (text, want) in texts.iter().zip(expected.lines()) {
            assert_eq!(
                row(&mut bi, text),
                want,
                "{name}: {:?}",
                String::from_utf16_lossy(text)
            );
        }
        assert_eq!(expected.lines().count(), texts.len(), "{name}");
        compiled += 1;
    }
    assert_eq!(compiled, 32, "rule sets compiled");
    println!("{refused} rule sets refused");
}

/// `UCharacter.getCharFromExtendedName` over the names ICU4J reports for a
/// sample of every plane, its aliases, ideograph and Hangul names, the
/// `<type-hex>` forms and junk: the same code point (or none) for each.
#[test]
fn character_names_match_icu4j() {
    use lucene_analysis_icu::icu4j::unames::char_from_extended_name;
    let mut n = 0;
    for line in std::fs::read_to_string(format!("{}/names_probe.tsv", dir()))
        .unwrap()
        .lines()
    {
        let (name, cp) = line.rsplit_once('\t').unwrap();
        let name = unescape(name);
        let want: i64 = cp.parse().unwrap();
        let got = char_from_extended_name(&name).map_or(-1, i64::from);
        assert_eq!(got, want, "{name:?}");
        n += 1;
    }
    assert!(n > 4000, "{n} names probed");
}

/// `AnalysisRows.esc`'s inverse: a backslash, a tab, a newline and a
/// carriage return are escaped, and so is each control character or lone
/// surrogate (as `u` and four hex digits).
fn unescape(s: &str) -> String {
    let mut units: Vec<u16> = Vec::new();
    let mut it = s.chars().peekable();
    while let Some(c) = it.next() {
        if c == '\\' {
            match it.next() {
                Some('u') => {
                    let hex: String = (0..4).filter_map(|_| it.next()).collect();
                    units.push(u16::from_str_radix(&hex, 16).unwrap());
                }
                Some('t') => units.push(9),
                Some('n') => units.push(10),
                Some('r') => units.push(13),
                Some(other) => units.push(other as u16),
                None => units.push(u16::from(b'\\')),
            }
        } else {
            let mut b = [0u16; 2];
            units.extend_from_slice(c.encode_utf16(&mut b));
        }
    }
    String::from_utf16_lossy(&units)
}
