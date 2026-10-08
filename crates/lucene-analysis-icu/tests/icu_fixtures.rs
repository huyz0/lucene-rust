//! M12 T12.4: analysis-icu against Lucene 10.5.0 over ICU4J 77.1.
//!
//! `fixtures/src/GenAnalysisIcu.java` writes `analysis_icu/`: normalization
//! of every assigned 256-code-point block and of 600 stress strings through
//! six normalizers in four modes (`norm_blocks.tsv`, `norm_strings.tsv`),
//! `UnicodeSet` patterns (`unicode_sets.tsv`), and analysis chains over
//! `corpus/analysis-icu.txt` and the strings (`n_*`, `cf*`, `c_factory_*`).
//! Every row must be equal.
#![allow(clippy::arithmetic_side_effects)] // test code: no value read off disk

#[path = "../../lucene-analysis/tests/support/mod.rs"]
mod support;

use lucene_analysis::factory::CustomAnalyzer;
use lucene_analysis::util::WhitespaceTokenizer;
use lucene_analysis::{Analyzer, CharReader, KeywordTokenizer, TokenStream};
use lucene_analysis_icu::icu4j::normalizer2::{Mode, Normalizer2, QuickCheck};
use lucene_analysis_icu::icu4j::unicode_set::UnicodeSet;
use lucene_analysis_icu::{ICUFoldingFilter, ICUNormalizer2CharFilter, ICUNormalizer2Filter};
use support::{chain, chain_cf, comps, corpus, data_dir, esc, normalise_expected};

const DIR: &str = "analysis_icu";

fn read(file: &str) -> String {
    std::fs::read_to_string(data_dir(DIR) + file).unwrap()
}

/// `AnalysisRows.esc`'s inverse over UTF-16 units (a `\uXXXX` may be a lone
/// surrogate).
fn unesc_units(s: &str) -> Vec<u16> {
    let mut out = Vec::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            let mut b = [0u16; 2];
            out.extend_from_slice(c.encode_utf16(&mut b));
            continue;
        }
        match it.next() {
            Some('t') => out.push(9),
            Some('n') => out.push(10),
            Some('r') => out.push(13),
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(u16::from_str_radix(&h, 16).unwrap());
            }
            Some(o) => {
                let mut b = [0u16; 2];
                out.extend_from_slice(o.encode_utf16(&mut b));
            }
            None => out.push('\\' as u16),
        }
    }
    out
}

/// `AnalysisRows.esc` over UTF-16 units.
fn esc_units(s: &[u16]) -> String {
    let mut out = String::new();
    for (i, r) in char::decode_utf16(s.iter().copied()).enumerate() {
        let _ = i;
        match r {
            Ok(c) => out.push_str(&esc(&c.to_string())),
            Err(e) => out.push_str(&format!("\\u{:04X}", e.unpaired_surrogate())),
        }
    }
    out
}

fn fnv(s: &[u16]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    for &c in s {
        h = (h ^ u64::from(c >> 8)).wrapping_mul(0x0100_0000_01b3);
        h = (h ^ u64::from(c & 0xff)).wrapping_mul(0x0100_0000_01b3);
    }
    h
}

fn fnv_str(s: &str) -> u64 {
    let units: Vec<u16> = s.encode_utf16().collect();
    fnv(&units)
}

fn normalizer(form: &str, mode: &str) -> Normalizer2 {
    let mode = match mode {
        "compose" => Mode::Compose,
        "decompose" => Mode::Decompose,
        "fcd" => Mode::Fcd,
        _ => Mode::ComposeContiguous,
    };
    if form == "utr30" {
        let n = Normalizer2::utr30();
        if mode == Mode::Compose {
            return n;
        }
        return Normalizer2::from_data(include_bytes!("../src/resources/utr30.nrm"), mode).unwrap();
    }
    Normalizer2::get_instance(form, mode).unwrap()
}

fn qc(q: QuickCheck) -> &'static str {
    match q {
        QuickCheck::Yes => "Y",
        QuickCheck::No => "N",
        QuickCheck::Maybe => "M",
    }
}

#[test]
fn normalization_of_every_block_matches_icu4j() {
    let mut n = 0;
    let mut current: Option<(String, String, Normalizer2)> = None;
    for line in read("norm_blocks.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if current.as_ref().is_none_or(|c| c.0 != f[0] || c.1 != f[1]) {
            current = Some((f[0].into(), f[1].into(), normalizer(f[0], f[1])));
        }
        let norm = &current.as_ref().unwrap().2;
        let start = u32::from_str_radix(f[2], 16).unwrap();
        let mut s = Vec::new();
        let mut bounds = String::new();
        for c in start..start + 256 {
            if (0xd800..=0xdfff).contains(&c) {
                continue;
            }
            let ch = char::from_u32(c).unwrap();
            let mut b = [0u16; 2];
            s.extend_from_slice(ch.encode_utf16(&mut b));
            let c = c as i32;
            bounds.push(if norm.has_boundary_before(c) {
                '1'
            } else {
                '0'
            });
            bounds.push(if norm.has_boundary_after(c) { '1' } else { '0' });
            bounds.push(if norm.is_inert(c) { '1' } else { '0' });
        }
        let actual = format!(
            "{}\t{}\t{}\t{:x}\t{}\t{}\t{}\t{:x}",
            f[0],
            f[1],
            f[2],
            fnv(&norm.normalize(&s)),
            qc(norm.quick_check(&s)),
            u8::from(norm.is_normalized(&s)),
            norm.span_quick_check_yes(&s),
            fnv_str(&bounds)
        );
        assert_eq!(actual, line);
        n += 1;
    }
    assert!(n > 20_000, "{n}");
}

#[test]
fn normalization_of_stress_strings_matches_icu4j() {
    let strings: Vec<Vec<u16>> = read("norm_strings.txt").lines().map(unesc_units).collect();
    assert_eq!(strings.len(), 600);
    let mut n = 0;
    let mut current: Option<(String, String, Normalizer2)> = None;
    for line in read("norm_strings.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        if current.as_ref().is_none_or(|c| c.0 != f[0] || c.1 != f[1]) {
            current = Some((f[0].into(), f[1].into(), normalizer(f[0], f[1])));
        }
        let norm = &current.as_ref().unwrap().2;
        let s = &strings[f[2].parse::<usize>().unwrap()];
        let split = s.len() / 2;
        let mut first = norm.normalize(&s[..split]);
        norm.normalize_second_and_append(&mut first, &s[split..]);
        let actual = format!(
            "{}\t{}\t{}\t{}\t{}\t{}\t{}",
            f[0],
            f[1],
            f[2],
            esc_units(&norm.normalize(s)),
            qc(norm.quick_check(s)),
            norm.span_quick_check_yes(s),
            esc_units(&first)
        );
        assert_eq!(actual, line);
        n += 1;
    }
    assert_eq!(n, 600 * 24);
}

#[test]
fn unicode_set_patterns_match_icu4j() {
    let mut n = 0;
    for line in read("unicode_sets.tsv").lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let pattern = String::from_utf16(&unesc_units(f[0])).unwrap();
        let actual = match UnicodeSet::from_pattern(&pattern) {
            Ok(s) => {
                let mut ranges = String::new();
                let mut size = 0u64;
                for &(a, b) in s.ranges() {
                    ranges.push_str(&format!("{a:x}-{b:x},"));
                    size += u64::from(b - a + 1);
                }
                for st in s.strings() {
                    ranges.push_str(&format!("{{{}}}", esc_units(st)));
                    size += 1;
                }
                format!(
                    "{}\t{size}\t{}\t{:x}",
                    f[0],
                    s.ranges().len(),
                    fnv_str(&ranges)
                )
            }
            Err(e) => {
                let class = lucene_analysis::factory::FactoryError::from(e)
                    .java_class()
                    .to_string();
                format!("{}\tX\t{class}", f[0])
            }
        };
        assert_eq!(actual, line);
        n += 1;
    }
    assert!(n > 600, "{n}");
}

fn ws() -> WhitespaceTokenizer {
    WhitespaceTokenizer::new()
}

fn build(name: &str) -> Option<Analyzer> {
    let parts: Vec<&str> = name.split('_').collect();
    Some(match name {
        "n_kw_nfkc_cf" => chain(|| comps(ICUNormalizer2Filter::new(KeywordTokenizer::new()))),
        "n_ws_nfkc_cf" => chain(|| comps(ICUNormalizer2Filter::new(ws()))),
        "n_kw_folding" => chain(|| comps(ICUFoldingFilter::new(KeywordTokenizer::new()))),
        "n_ws_folding" => chain(|| comps(ICUFoldingFilter::new(ws()))),
        "cf_ws_nfkc_cf" => chain_cf(
            |r| Box::new(ICUNormalizer2CharFilter::new(r)) as Box<dyn CharReader>,
            || comps(ws()),
        ),
        "cf_kw_nfkc_cf" => chain_cf(
            |r| Box::new(ICUNormalizer2CharFilter::new(r)) as Box<dyn CharReader>,
            || comps(KeywordTokenizer::new()),
        ),
        _ if parts[0] == "n" && parts[1] == "kw" => {
            let mode = parts[parts.len() - 1].to_string();
            let form = parts[2..parts.len() - 1].join("_");
            let n = normalizer(&form, &mode);
            chain(move || {
                comps(ICUNormalizer2Filter::with_normalizer(
                    KeywordTokenizer::new(),
                    n.clone(),
                ))
            })
        }
        _ if parts[0].starts_with("cf") && parts[1] == "ws" => {
            let size: usize = parts[0][2..].parse().ok()?;
            let mode = parts[parts.len() - 1].to_string();
            let form = parts[2..parts.len() - 1].join("_");
            let n = normalizer(&form, &mode);
            chain_cf(
                move |r| {
                    Box::new(ICUNormalizer2CharFilter::with_buffer_size(
                        r,
                        n.clone(),
                        size,
                    )) as Box<dyn CharReader>
                },
                || comps(ws()),
            )
        }
        _ => return None,
    })
}

/// The lines the chains run over: the corpus, then the stress strings
/// without lone surrogates.
fn chain_lines() -> Vec<String> {
    let mut lines = corpus("analysis-icu.txt");
    for s in read("norm_strings.txt").lines() {
        if let Ok(s) = String::from_utf16(&unesc_units(s)) {
            lines.push(s);
        }
    }
    lines
}

#[test]
fn chains_match_lucene() {
    let names: std::collections::BTreeSet<String> = support::fixture_names(DIR)
        .into_iter()
        .filter(|n| n.starts_with("n_") || n.starts_with("cf"))
        .collect();
    assert_eq!(
        support::check_chains_over(DIR, &chain_lines(), &names, build),
        names.len()
    );
    assert!(names.len() > 30);
}

#[test]
fn factory_chains_match_lucene() {
    lucene_analysis_icu::register_factories().unwrap();
    let lines = corpus("analysis-icu.txt");
    let mut n = 0;
    for spec in read("factory_specs.tsv").lines() {
        let (name, spec) = spec.split_once('\t').unwrap();
        let mut b = CustomAnalyzer::builder();
        let built = (move || {
            for part in spec.split(' ') {
                let (kind, rest) = part.split_once(':').unwrap();
                let mut it = rest.split(',');
                let fname = it.next().unwrap();
                let args: Vec<(String, String)> = it
                    .map(|a| {
                        let (k, v) = a.split_once('=').unwrap();
                        (
                            k.to_string(),
                            v.replace("_SPACE_", " ").replace("_COMMA_", ","),
                        )
                    })
                    .collect();
                let args: Vec<&str> = args
                    .iter()
                    .flat_map(|(k, v)| [k.as_str(), v.as_str()])
                    .collect();
                b = match kind {
                    "c" => b.add_char_filter(fname, &args)?,
                    "t" => b.with_tokenizer(fname, &args)?,
                    _ => b.add_token_filter(fname, &args)?,
                };
            }
            b.build()
        })();
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
            }
            Err(e) => assert_eq!(
                vec![format!("B\t{}", e.java_class())],
                expected,
                "{name}: {spec}"
            ),
        }
        n += 1;
    }
    assert!(n > 15);
}

#[test]
fn whitespace_tokenizer_is_a_token_stream() {
    // Keeps the helper's trait bound honest.
    let t: Box<dyn TokenStream> = Box::new(ws());
    assert!(t.attributes().term().is_empty());
}
