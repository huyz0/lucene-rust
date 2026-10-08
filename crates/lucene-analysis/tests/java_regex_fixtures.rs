//! M12 T12.7: `java.util.regex` beyond the shim -- `util/java_backtrack.rs`
//! (and the shim, for what it runs) against `Pattern` itself, over
//! `fixtures/src/GenJavaRegex.java`'s curated and generated patterns.

mod support;

use lucene_analysis::util::java_regex::{is_unsupported, JavaMatcher};
use lucene_analysis::util::JavaPattern;
use lucene_analysis::AnalysisError;
use support::{esc, normalise_expected, unesc};

/// `GenAnalysisCommon.regexRun`.
fn regex_run(p: &JavaPattern, input: &str) -> String {
    let mut m = JavaMatcher::new(p, input);
    let mut b = String::new();
    while m.try_find().unwrap() {
        b.push_str(&format!("({},{}", m.start(0), m.end(0)));
        for g in 1..=m.group_count() {
            b.push_str(&format!(" {}:{}", m.start(g), m.end(g)));
        }
        b.push(')');
    }
    let rep = p.replace(input, "<$0>", true).unwrap();
    format!("{b} rep={} m={}", esc(&rep), p.try_matches(input).unwrap())
}

/// The constructs still refused (see `util/java_backtrack.rs`).
const REFUSED: &[&str] = &["\\X", "\\X+", "\\N{LATIN SMALL LETTER A}"];

fn check(file: &str) -> (usize, usize) {
    let text = std::fs::read_to_string(support::data_dir("java_regex") + file).unwrap();
    let (mut compared, mut refused, mut failures) = (0, 0, Vec::new());
    let mut failed_patterns = std::collections::BTreeSet::new();
    let mut pattern: Option<Result<JavaPattern, AnalysisError>> = None;
    let mut source = String::new();
    for line in text.lines() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        if f[0] == "P" {
            source = unesc(f[1]);
            pattern = Some(JavaPattern::compile(&source));
            continue;
        }
        let (input, want) = (unesc(f[1]), normalise_expected(f[2]));
        if want == "SOE" {
            // Java's own `StackOverflowError` (deep.txt): the port has more
            // depth than Java's 1 MiB stack, so there is nothing to compare.
            continue;
        }
        match pattern.as_ref().unwrap() {
            Ok(p) => {
                let got = if want.starts_with("EXC") {
                    format!(
                        "compiled ({})",
                        if p.is_backtracking() { "bt" } else { "shim" }
                    )
                } else {
                    regex_run(p, &input)
                };
                if got != want && failed_patterns.insert(source.clone()) && failures.len() < 60 {
                    failures.push(format!(
                        "{source:?} on {input:?}:\n  java {want}\n  rust {got}"
                    ));
                }
                compared += 1;
            }
            Err(e) if is_unsupported(e) => {
                if !REFUSED.contains(&source.as_str()) && failed_patterns.insert(source.clone()) {
                    failures.push(format!("{source:?} refused: {e}"));
                }
                refused += 1;
            }
            Err(e) => {
                if !want.starts_with("EXC") && failed_patterns.insert(source.clone()) {
                    failures.push(format!("{source:?} rejected, Java gives {want}: {e}"));
                }
                compared += 1;
            }
        }
    }
    assert!(
        failures.is_empty(),
        "{file}: {} patterns fail:\n{}",
        failed_patterns.len(),
        failures.join("\n")
    );
    (compared, refused)
}

#[test]
fn curated_patterns_match_java() {
    let (compared, refused) = check("curated.txt");
    assert!(compared > 4000, "{compared}");
    assert!(refused <= REFUSED.len() * 40, "{refused}");
}

/// Inputs deep enough for thousands of backtracking entries, among them
/// under a negated lookaround or a zero-count quantifier (where an attempt
/// cut short must never read as a match).
#[test]
fn deep_inputs_match_java() {
    let (compared, refused) = check("deep.txt");
    assert!(compared >= 200, "{compared}");
    assert_eq!(refused, 0);
}

#[test]
fn generated_patterns_match_java() {
    let (compared, refused) = check("random.txt");
    assert!(compared >= 19_000, "{compared}");
    assert_eq!(refused, 0);
}
