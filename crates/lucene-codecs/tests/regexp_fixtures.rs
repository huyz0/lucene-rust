//! Differential test for [`lucene_codecs::regexp::RegexpPattern`] against real
//! Lucene's `RegExp` grammar.
//!
//! `fixtures/src/GenRegexp.java` builds each pattern exactly the way
//! `RegexpQuery(Term)` does -- `RegExp.ALL` syntax flags, no match flags, a
//! provider that knows no named automata, `Operations.determinize` at the
//! default work limit -- and records, for every term, whether the resulting
//! `ByteRunAutomaton` accepts its UTF-8 bytes. Patterns real Lucene rejects are
//! recorded as `ERR`.
//!
//! This is the check that a hand-written parser for a *non-PCRE* grammar needs:
//! the failure mode it guards against is not "fails to compile" but "compiles
//! and quietly means something else" -- `#`, `@`, `&`, `"..."` and `<n-m>` are
//! operators here and ordinary characters in every other regex dialect, while
//! `~` is an ordinary character here and an operator in some.
// Test-support code opts out of the arithmetic gate at the file boundary:
// the gate exists for values read off disk in production decode paths, not
// for a fixture builder's own index arithmetic. See
// `docs/arithmetic-gate.md`.
#![allow(clippy::arithmetic_side_effects)]

use lucene_codecs::regexp::RegexpPattern;

fn dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/regexp/").to_string()
}

fn terms() -> Vec<String> {
    let text = std::fs::read_to_string(format!("{}terms.txt", dir()))
        .expect("run fixtures generator first (GenRegexp)");
    // The file ends in a newline, so the last `lines()` entry is real content;
    // the *first* is deliberately empty (the empty term is a real case).
    text.lines().map(str::to_string).collect()
}

fn cases() -> Vec<(String, String)> {
    let text = std::fs::read_to_string(format!("{}cases.tsv", dir()))
        .expect("run fixtures generator first (GenRegexp)");
    text.lines()
        .map(|line| {
            let (pattern, mask) = line.split_once('\t').expect("every case line has a tab");
            (pattern.to_string(), mask.to_string())
        })
        .collect()
}

#[test]
fn matches_real_lucene_regexp_on_every_pattern_and_term() {
    let terms = terms();
    let cases = cases();
    assert!(cases.len() > 80, "fixture looks truncated: {}", cases.len());

    let mut divergences = Vec::new();
    for (pattern, mask) in &cases {
        match RegexpPattern::parse(pattern) {
            Err(err) => {
                if mask != "ERR" {
                    divergences.push(format!(
                        "{pattern:?}: this port rejected it ({err}), real Lucene compiled it"
                    ));
                }
            }
            Ok(compiled) => {
                if mask == "ERR" {
                    divergences.push(format!(
                        "{pattern:?}: this port compiled it, real Lucene rejected it"
                    ));
                    continue;
                }
                assert_eq!(
                    mask.len(),
                    terms.len(),
                    "{pattern:?}: mask width does not match terms.txt"
                );
                for (term, accepted) in terms.iter().zip(mask.chars()) {
                    let actual = compiled.matches(term.as_bytes());
                    let expected = accepted == '1';
                    if actual != expected {
                        divergences.push(format!(
                            "{pattern:?} vs term {term:?}: this port says {actual}, \
                             real Lucene says {expected}"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        divergences.is_empty(),
        "{} divergence(s) from real Lucene:\n{}",
        divergences.len(),
        divergences.join("\n")
    );
}

/// `literal_prefix` narrows `FieldTerms::regexp_intersect`'s scan to a
/// contiguous sorted range before any term is tested, so a prefix that is *not*
/// actually shared by every match would silently drop hits. It is allowed to be
/// shorter than `CompiledAutomaton.commonPrefix`; it is never allowed to be
/// wrong.
#[test]
fn literal_prefix_is_a_true_prefix_of_every_term_real_lucene_accepts() {
    let terms = terms();
    for (pattern, mask) in cases() {
        if mask == "ERR" {
            continue;
        }
        let Ok(compiled) = RegexpPattern::parse(&pattern) else {
            continue;
        };
        let prefix = compiled.literal_prefix();
        for (term, accepted) in terms.iter().zip(mask.chars()) {
            if accepted == '1' {
                assert!(
                    term.as_bytes().starts_with(&prefix),
                    "{pattern:?}: literal_prefix {prefix:?} is not a prefix of matching term \
                     {term:?}"
                );
            }
        }
    }
}

/// `dead_prefix_len` is what lets `FieldTerms::regexp_intersect` binary-search
/// past a whole run of the sorted term array -- the dead-state signal
/// `ByteRunAutomaton` gives `IntersectTermsEnum`. Getting it wrong drops
/// matches silently, so it is checked against real Lucene's own accept set:
/// for every term this port calls dead at length `k`, no term real Lucene
/// accepts may start with those `k` bytes.
#[test]
fn a_dead_prefix_excludes_every_term_real_lucene_accepts() {
    let terms = terms();
    for (pattern, mask) in cases() {
        if mask == "ERR" {
            continue;
        }
        let Ok(compiled) = RegexpPattern::parse(&pattern) else {
            continue;
        };
        for probe in &terms {
            let Some(k) = compiled.dead_prefix_len(probe.as_bytes()) else {
                continue;
            };
            let dead = &probe.as_bytes()[..k];
            for (term, accepted) in terms.iter().zip(mask.chars()) {
                assert!(
                    !(accepted == '1' && term.as_bytes().starts_with(dead)),
                    "{pattern:?}: prefix {:?} of {probe:?} declared dead, but real Lucene \
                     accepts {term:?}",
                    String::from_utf8_lossy(dead)
                );
            }
        }
    }
}
