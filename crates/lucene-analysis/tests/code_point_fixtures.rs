//! M11: every code point through the analysis-common filters whose output
//! depends on `java.lang.Character` (case mapping, categories, digits,
//! whitespace) -- `fixtures/data/analysis_common/codepoints.words`, written
//! by `fixtures/src/GenAnalysisCommon.java`.
//!
//! The port follows JDK 25 (Unicode 16.0), the JDK OpenSearch 3.8.0 bundles
//! and the oldest one with the FFM API the plugin needs on Unicode 16. CI's
//! fixture job runs JDK 21 (Unicode 15.0), so the generator leaves out the
//! code points whose `Character` properties differ between the two (its
//! `JDK21_JDK25_DIFFER`, the file's `X` rows) and is byte-identical under
//! either JDK; [`unicode_16_code_points_follow_jdk_25`] pins the port's
//! answers for those against JDK 25.

use std::collections::BTreeMap;

use lucene_analysis::cjk::{CJKWidthCharFilter, CJKWidthFilter};
use lucene_analysis::core_analysis::{DecimalDigitFilter, UpperCaseFilter};
use lucene_analysis::miscellaneous::{
    self as m, AsciiFoldingTokenFilter, ScandinavianFoldingFilter, ScandinavianNormalizationFilter,
    WordDelimiterGraphFilter,
};
use lucene_analysis::reader::CharReader;
use lucene_analysis::util::{LetterTokenizer, WhitespaceTokenizer};
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, KeywordTokenizer, LowerCaseFilter, TokenStream,
    TokenStreamComponents,
};

type Sink = Result<TokenStreamComponents, AnalysisError>;

struct Chain {
    components: Box<dyn Fn() -> Sink + Send + Sync>,
    width_char_filter: bool,
}

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Sink {
        (self.components)()
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        if self.width_char_filter {
            Box::new(CJKWidthCharFilter::new(reader))
        } else {
            reader
        }
    }
}

fn chain(width_char_filter: bool, f: impl Fn() -> Sink + Send + Sync + 'static) -> Analyzer {
    Analyzer::new(Chain {
        components: Box::new(f),
        width_char_filter,
    })
}

fn comps(s: impl TokenStream + 'static) -> Sink {
    Ok(TokenStreamComponents::new(s))
}

/// The generator's `codePointChains`.
fn chains() -> Vec<(&'static str, Analyzer)> {
    vec![
        (
            "fold",
            chain(false, || {
                comps(AsciiFoldingTokenFilter::new(KeywordTokenizer::new(), true))
            }),
        ),
        (
            "cjkw",
            chain(false, || {
                comps(CJKWidthFilter::new(KeywordTokenizer::new()))
            }),
        ),
        ("cjkwcf", chain(true, || comps(KeywordTokenizer::new()))),
        (
            "lower",
            chain(false, || {
                comps(LowerCaseFilter::new(KeywordTokenizer::new()))
            }),
        ),
        (
            "upper",
            chain(false, || {
                comps(UpperCaseFilter::new(KeywordTokenizer::new()))
            }),
        ),
        (
            "digit",
            chain(false, || {
                comps(DecimalDigitFilter::new(KeywordTokenizer::new()))
            }),
        ),
        (
            "scf",
            chain(false, || {
                comps(ScandinavianFoldingFilter::new(KeywordTokenizer::new()))
            }),
        ),
        (
            "scn",
            chain(false, || {
                comps(ScandinavianNormalizationFilter::new(KeywordTokenizer::new()))
            }),
        ),
        ("letter", chain(false, || comps(LetterTokenizer::new()))),
        ("ws", chain(false, || comps(WhitespaceTokenizer::new()))),
        (
            "wdgf",
            chain(false, || {
                comps(WordDelimiterGraphFilter::new(
                    KeywordTokenizer::new(),
                    m::GENERATE_WORD_PARTS
                        | m::GENERATE_NUMBER_PARTS
                        | m::SPLIT_ON_CASE_CHANGE
                        | m::SPLIT_ON_NUMERICS
                        | m::STEM_ENGLISH_POSSESSIVE,
                    None,
                )?)
            }),
        ),
    ]
}

/// The generator's `codePointRun`.
fn run(a: &Analyzer, cp: char) -> String {
    let text = format!("a{cp}B");
    let mut own = [0u16; 2];
    let own = cp.encode_utf16(&mut own);
    let mut out = String::new();
    let r = (|| -> Result<i32, AnalysisError> {
        let mut ts = a.token_stream("f", &text)?;
        let end = lucene_analysis::token_stream::consume(&mut ts, |x| {
            let units: Vec<u16> = x.term().encode_utf16().collect();
            let mut i = 0;
            while i < units.len() {
                if units[i..].starts_with(own) {
                    out.push_str("@.");
                    i += own.len();
                } else {
                    out.push_str(&format!("{:x}.", units[i]));
                    i += 1;
                }
            }
            out.push_str(&format!(
                ":{}-{}/{} ",
                x.start_offset(),
                x.end_offset(),
                x.position_increment()
            ));
        })?;
        Ok(end.end_offset())
    })();
    match r {
        Ok(e) => format!("{out}|{e}"),
        Err(e) => format!("{out}X{e:?}"),
    }
}

fn fixture() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/analysis_common/codepoints.words"
    ))
    .expect("codepoints.words (run scripts/gen-fixtures.sh --only GenAnalysisCommon)")
}

fn hex(s: &str) -> u32 {
    u32::from_str_radix(s, 16).unwrap()
}

/// The `X` rows: the code points left out.
fn skipped(text: &str) -> Vec<(u32, u32)> {
    text.lines()
        .filter_map(|l| l.strip_prefix("X\t"))
        .map(|l| {
            let (a, b) = l.split_once('\t').unwrap();
            (hex(a), hex(b))
        })
        .collect()
}

/// One chain's runs, as the generator writes them.
fn runs(name: &str, a: &Analyzer, skip: &[(u32, u32)]) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur: Option<(u32, u32, String)> = None;
    for cp in 0..=0x11_0000u32 {
        let r = char::from_u32(cp)
            .filter(|_| !skip.iter().any(|&(a, b)| (a..=b).contains(&cp)))
            .map(|c| run(a, c));
        if let Some((first, last, prev)) = &cur {
            if r.as_ref() != Some(prev) {
                out.push(format!("{name}\t{first:x}\t{last:x}\t{prev}"));
                cur = None;
            }
        }
        if let Some(r) = r {
            match &mut cur {
                Some((_, last, _)) => *last = cp,
                None => cur = Some((cp, cp, r)),
            }
        }
    }
    out
}

#[test]
fn every_code_point_matches_lucene() {
    let text = fixture();
    let skip = skipped(&text);
    assert_eq!(skip.len(), 55, "JDK21_JDK25_DIFFER ranges");
    let mut expected: BTreeMap<&str, Vec<&str>> = BTreeMap::new();
    for l in text.lines().filter(|l| !l.starts_with("X\t")) {
        expected
            .entry(l.split('\t').next().unwrap())
            .or_default()
            .push(l);
    }
    let chains = chains();
    assert_eq!(expected.len(), chains.len());
    std::thread::scope(|s| {
        for (name, a) in &chains {
            let (want, skip) = (&expected[name], &skip);
            s.spawn(move || {
                let got = runs(name, a, skip);
                for (g, w) in got.iter().zip(want.iter()) {
                    assert_eq!(g, w, "{name}");
                }
                assert_eq!(got.len(), want.len(), "{name}: runs");
            });
        }
    });
}

/// The code points whose `Character` properties changed between Unicode
/// 15.0 (JDK 21) and 16.0 (JDK 25) follow JDK 25 -- here pinned to what
/// JDK 25.0.4 answers (`Character.toUpperCase`/`toLowerCase`/`getType`).
#[test]
fn unicode_16_code_points_follow_jdk_25() {
    let lower = chain(false, || {
        comps(LowerCaseFilter::new(KeywordTokenizer::new()))
    });
    let upper = chain(false, || {
        comps(UpperCaseFilter::new(KeywordTokenizer::new()))
    });
    let letter = chain(false, || comps(LetterTokenizer::new()));
    let one = |a: &Analyzer, s: &str| -> Vec<String> {
        a.analyze(s).into_iter().map(|t| t.term).collect()
    };
    // New case pairs (JDK 21: no mapping).
    for (u, l) in [
        ('\u{A7CB}', '\u{264}'),
        ('\u{A7DC}', '\u{19B}'),
        ('\u{1C89}', '\u{1C8A}'),
        ('\u{A7DA}', '\u{A7DB}'),
    ] {
        assert_eq!(one(&lower, &u.to_string()), vec![l.to_string()], "{u:?}");
        assert_eq!(one(&upper, &l.to_string()), vec![u.to_string()], "{l:?}");
    }
    // New letters (JDK 21: unassigned, so a LetterTokenizer split there).
    for c in [
        '\u{10D50}',
        '\u{11380}',
        '\u{16D40}',
        '\u{1E5D0}',
        '\u{105C0}',
    ] {
        assert_eq!(
            one(&letter, &format!("a{c}b")),
            vec![format!("a{c}b")],
            "{c:?}"
        );
    }
    // Every skipped range is one the fixture names.
    let skip = skipped(&fixture());
    for c in [
        0x19Bu32, 0x264, 0x363, 0x897, 0x1C89, 0xA7CB, 0xA7DC, 0x1171E, 0x2EBF0,
    ] {
        assert!(skip.iter().any(|&(a, b)| (a..=b).contains(&c)), "{c:#x}");
    }
}
