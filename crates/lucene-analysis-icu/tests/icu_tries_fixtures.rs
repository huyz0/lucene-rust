//! `BytesTrie`/`CharsTrie` against ICU4J 77.1 (`fixtures/src/GenAnalysisIcuTries.java`):
//! tries built by ICU4J's builders (both options, every value size, up to
//! 66 KB so jump deltas take three bytes), walked unit by unit and by code
//! point -- every step's result, the final value and `current()` equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

use lucene_analysis_icu::icu4j::tries::{BytesTrie, CharsTrie, TrieResult};

fn ordinal(r: TrieResult) -> char {
    match r {
        TrieResult::NoMatch => '0',
        TrieResult::NoValue => '1',
        TrieResult::FinalValue => '2',
        TrieResult::IntermediateValue => '3',
    }
}

fn bytes(hex: &str) -> Vec<u8> {
    (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect()
}

fn units(hex: &str) -> Vec<u16> {
    (0..hex.len())
        .step_by(4)
        .map(|i| u16::from_str_radix(&hex[i..i + 4], 16).unwrap())
        .collect()
}

#[test]
fn tries_match_icu4j() {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/analysis_icu_tries/tries.txt"
    );
    let text = std::fs::read_to_string(path).unwrap();
    let mut kind = "";
    let mut trie_bytes: Vec<u8> = Vec::new();
    let mut trie_units: Vec<u16> = Vec::new();
    let (mut tries, mut queries) = (0, 0);
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "T" => {
                kind = if f[1] == "B" { "B" } else { "C" };
                if kind == "B" {
                    trie_bytes = bytes(f[3]);
                } else {
                    trie_units = units(f[3]);
                }
                tries += 1;
            }
            "Q" if kind == "B" => {
                let q = bytes(f[1]);
                let mut t = BytesTrie::new(&trie_bytes, 0);
                let mut steps = String::new();
                let mut last = None;
                for (i, &b) in q.iter().enumerate() {
                    let r = if i == 0 {
                        t.first(i32::from(b))
                    } else {
                        t.next(i32::from(b))
                    };
                    steps.push(ordinal(r));
                    last = Some(r);
                    if r == TrieResult::NoMatch {
                        break;
                    }
                }
                let value = match last {
                    Some(r) if r.has_value() => t.get_value().to_string(),
                    _ => "-".to_string(),
                };
                assert_eq!((steps.as_str(), value.as_str()), (f[2], f[3]), "{line}");
                assert_eq!(ordinal(t.current()).to_string(), f[4], "{line}");
                queries += 1;
            }
            "Q" | "P" => {
                let q = units(f[1]);
                let mut t = CharsTrie::new(&trie_units, 0);
                let mut steps = String::new();
                let mut last = None;
                if f[0] == "Q" {
                    for (i, &u) in q.iter().enumerate() {
                        let r = if i == 0 {
                            t.first(i32::from(u))
                        } else {
                            t.next(i32::from(u))
                        };
                        steps.push(ordinal(r));
                        last = Some(r);
                        if r == TrieResult::NoMatch {
                            break;
                        }
                    }
                } else {
                    let mut i = 0;
                    while i < q.len() {
                        // Java's codePointAt: a lone surrogate is itself.
                        let (u, next) = (i32::from(q[i]), q.get(i + 1).map(|&n| i32::from(n)));
                        let cp = match next {
                            Some(n)
                                if (0xd800..0xdc00).contains(&u)
                                    && (0xdc00..0xe000).contains(&n) =>
                            {
                                0x10000 + ((u - 0xd800) << 10) + (n - 0xdc00)
                            }
                            _ => u,
                        };
                        let first = i == 0;
                        i += if cp > 0xffff { 2 } else { 1 };
                        let r = if first {
                            t.first_for_code_point(cp)
                        } else {
                            t.next_for_code_point(cp)
                        };
                        steps.push(ordinal(r));
                        last = Some(r);
                        if r == TrieResult::NoMatch {
                            break;
                        }
                    }
                }
                let value = match last {
                    Some(r) if r.has_value() => t.get_value().to_string(),
                    _ => "-".to_string(),
                };
                assert_eq!((steps.as_str(), value.as_str()), (f[2], f[3]), "{line}");
                if f[0] == "Q" {
                    assert_eq!(ordinal(t.current()).to_string(), f[4], "{line}");
                }
                queries += 1;
            }
            _ => panic!("{line}"),
        }
    }
    assert_eq!(tries, 48);
    assert!(queries > 8_000, "{queries}");
}
