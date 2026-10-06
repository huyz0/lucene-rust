//! M11 T11.4: Hunspell against Lucene on this repository's own dictionaries.
//!
//! `fixtures/src/GenHunspell.java` loads each `fixtures/corpus/hunspell/*.aff`
//! (with its `.dic`) twice -- case-sensitive and `ignoreCase` -- and records,
//! per word, `spell`, the stems `HunspellStemFilter` emits (dedup off, on,
//! longest only), `getRoots`, `analyzeSimpleWord`, `suggest`, a `Suggester`
//! with an n-gram `FragmentChecker` and `proceedPastRep`, and an n-gram
//! checker's verdict; per root, `lookupEntries` and `getAllWordForms`; per
//! dictionary, `generateAllSimpleWords`; for a dictionary Lucene refuses, the
//! exception. This loads the same files in Rust and requires every column to
//! be equal.

use std::sync::Arc;

use lucene_analysis::hunspell::{
    Dictionary, FragmentChecker, Hunspell, HunspellError, HunspellStemFilter, NGramFragmentChecker,
    Suggester, TimeoutPolicy, WordFormGenerator,
};
use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, KeywordTokenizer, TokenStream,
    TokenStreamComponents,
};

fn root() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/").to_string()
}

struct Chain {
    dictionary: Arc<Dictionary>,
    dedup: bool,
    longest_only: bool,
}

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(HunspellStemFilter::new(
            KeywordTokenizer::new(),
            Arc::clone(&self.dictionary),
            self.dedup,
            self.longest_only,
        )))
    }
}

fn stems(d: &Arc<Dictionary>, word: &str, dedup: bool, longest_only: bool) -> String {
    let a = Analyzer::new(Chain {
        dictionary: Arc::clone(d),
        dedup,
        longest_only,
    });
    let mut ts = a.token_stream("f", word).unwrap();
    ts.reset().unwrap();
    let mut out = Vec::new();
    while ts.increment_token().unwrap() {
        out.push(ts.attributes().term().to_string());
    }
    ts.end().unwrap();
    ts.close().unwrap();
    out.join("|")
}

/// Java's exception class for a load failure.
fn exception_name(e: &HunspellError) -> &'static str {
    match e {
        HunspellError::Parse { .. } => "ParseException",
        HunspellError::IllegalArgument(_) => "IllegalArgumentException",
        HunspellError::IllegalState(_) => "IllegalStateException",
        HunspellError::NumberFormat(_) => "NumberFormatException",
        HunspellError::IndexOutOfBounds(_) => "ArrayIndexOutOfBoundsException",
        HunspellError::NegativeArraySize(_) => "NegativeArraySizeException",
        HunspellError::UnsupportedCharset(_) => "UnsupportedCharsetException",
        HunspellError::IllegalCharsetName(_) => "IllegalCharsetNameException",
        HunspellError::UnmappableCharacter(_) => "UnmappableCharacterException",
        HunspellError::Unsupported(_) => "UnsupportedOperationException",
    }
}

/// Checks one `GenHunspell` output file against the port: whether the
/// dictionary was refused (as Lucene refused it), and how many words matched.
fn check_file(
    file: &str,
    aff: &[u8],
    dic: &[u8],
    ignore_case: bool,
    expected: &str,
) -> (bool, usize) {
    let mut words = 0;
    let dictionary = match Dictionary::new(aff, &[dic], ignore_case) {
        Ok(d) => Arc::new(d),
        Err(e) => {
            let line = expected.lines().next().unwrap_or("");
            let mut f = line.split('\t');
            assert_eq!(
                f.next(),
                Some("X"),
                "{file}: Lucene loads it, the port refuses: {e}"
            );
            assert_eq!(f.next(), Some(exception_name(&e)), "{file}: {e}");
            return (true, 0);
        }
    };
    let h = Hunspell::with_timeout_policy(&dictionary, TimeoutPolicy::NoTimeout, None);
    // A checker Lucene cannot build ("Too many collisions") is `!` and the
    // exception class in its column.
    let checker = NGramFragmentChecker::from_all_simple_words(2, &dictionary);
    let tuned = checker.as_ref().map(|c| {
        Suggester::new(&dictionary)
            .with_fragment_checker(c)
            .proceed_past_rep()
    });
    let roots: Vec<String> = expected
        .lines()
        .filter_map(|l| l.strip_prefix("E\t"))
        .map(|l| l.split('\t').next().unwrap().to_string())
        .collect();
    let root_refs: Vec<&str> = roots.iter().map(String::as_str).collect();
    let from_roots = NGramFragmentChecker::from_words(3, &root_refs);
    let gen = WordFormGenerator::new(&dictionary);
    for line in expected.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        match f[0] {
            "X" => panic!("{file}: Lucene refuses it ({}), the port loads it", f[1]),
            "E" => {
                let got = match dictionary.lookup_entries(f[1]) {
                    None => "null".to_string(),
                    Some(e) => e
                        .iter()
                        .map(ToString::to_string)
                        .collect::<Vec<_>>()
                        .join("|"),
                };
                assert_eq!(got, f[2], "{file}: lookupEntries({})", f[1]);
                let forms: Vec<String> = gen
                    .get_all_word_forms(f[1])
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                assert_eq!(forms.join("|"), f[3], "{file}: getAllWordForms({})", f[1]);
            }
            "G" => {
                let mut all = Vec::new();
                gen.generate_all_simple_words(&mut |aw| all.push(aw.to_string()));
                assert_eq!(all.join("|"), f[1], "{file}: generateAllSimpleWords");
            }
            "W" => {
                let w = f[1];
                assert_eq!(h.spell(w), f[2] == "1", "{file}: spell({w})");
                assert_eq!(
                    stems(&dictionary, w, false, false),
                    f[3],
                    "{file}: stems({w})"
                );
                assert_eq!(
                    stems(&dictionary, w, true, false),
                    f[4],
                    "{file}: uniqueStems({w})"
                );
                assert_eq!(
                    stems(&dictionary, w, true, true),
                    f[5],
                    "{file}: longestOnly({w})"
                );
                assert_eq!(h.get_roots(w).join("|"), f[6], "{file}: getRoots({w})");
                let analyses: Vec<String> = h
                    .analyze_simple_word(w)
                    .iter()
                    .map(ToString::to_string)
                    .collect();
                assert_eq!(analyses.join("|"), f[7], "{file}: analyzeSimpleWord({w})");
                // Java throws `StringIndexOutOfBoundsException` for a word
                // `IGNORE` empties; the port suggests nothing.
                let expect = |c: &str| {
                    if c.starts_with('!') {
                        String::new()
                    } else {
                        c.to_string()
                    }
                };
                assert_eq!(
                    h.suggest(w).unwrap().join("|"),
                    expect(f[8]),
                    "{file}: suggest({w})"
                );
                match &tuned {
                    Ok(tuned) => assert_eq!(
                        tuned.suggest_no_timeout(w).join("|"),
                        expect(f[9]),
                        "{file}: tuned suggest({w})"
                    ),
                    Err(e) => assert_eq!(format!("!{}", exception_name(e)), f[9], "{file}: {e}"),
                }
                match &from_roots {
                    Ok(from_roots) => {
                        let units: Vec<u16> = w.encode_utf16().collect();
                        let impossible =
                            from_roots.has_impossible_fragment_around(&units, 0, units.len());
                        // Java divides by zero over the empty `BitSet` of a
                        // dictionary with no roots (`!ArithmeticException`);
                        // the port's 64 empty bits make every fragment
                        // impossible.
                        assert_eq!(
                            impossible,
                            f[10] == "1" || f[10] == "!ArithmeticException",
                            "{file}: fragment({w}) (Lucene: {})",
                            f[10]
                        );
                    }
                    Err(e) => assert_eq!(format!("!{}", exception_name(e)), f[10], "{file}: {e}"),
                }
                words += 1;
            }
            other => panic!("{file}: unknown line kind {other}"),
        }
    }
    (false, words)
}

/// The corpus and generated-data roots: the committed ones, or (opt-in,
/// `scripts/check-hunspell-lucene-dictionaries.sh`) `HUNSPELL_CORPUS` and
/// `HUNSPELL_DATA`, where every file is checked and the failures listed.
fn roots() -> (String, String, bool) {
    match (
        std::env::var("HUNSPELL_CORPUS"),
        std::env::var("HUNSPELL_DATA"),
    ) {
        (Ok(c), Ok(d)) => (c, d, true),
        _ => (
            format!("{}corpus/hunspell/", root()),
            format!("{}data/hunspell/", root()),
            false,
        ),
    }
}

#[test]
fn hunspell_matches_lucene_on_every_dictionary() {
    let (corpus, data, opt_in) = roots();
    let mut names: Vec<String> = std::fs::read_dir(&corpus)
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".aff").map(str::to_string)
        })
        .collect();
    names.sort();
    let (mut words, mut files, mut refused) = (0, 0, 0);
    let mut failures = Vec::new();
    for name in &names {
        let aff = std::fs::read(format!("{corpus}{name}.aff")).unwrap();
        let dic = std::fs::read(format!("{corpus}{name}.dic")).unwrap_or_default();
        for ignore_case in [false, true] {
            let file = format!("{data}{name}{}.tsv", if ignore_case { ".ic" } else { "" });
            let expected = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file}: {e}"));
            files += 1;
            let run = || check_file(&file, &aff, &dic, ignore_case, &expected);
            let (was_refused, n) = if opt_in {
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(run)) {
                    Ok(r) => r,
                    Err(e) => {
                        let msg = e
                            .downcast_ref::<String>()
                            .cloned()
                            .or_else(|| e.downcast_ref::<&str>().map(|s| s.to_string()))
                            .unwrap_or_default();
                        failures.push(msg.lines().next().unwrap_or("").to_string());
                        (false, 0)
                    }
                }
            } else {
                run()
            };
            refused += usize::from(was_refused);
            words += n;
        }
    }
    eprintln!(
        "{files} files, {refused} refused, {words} words, {} failing",
        failures.len()
    );
    assert!(
        failures.is_empty(),
        "{} of {files} files differ:\n{}",
        failures.len(),
        failures.join("\n")
    );
    if !opt_in {
        assert!(files >= 110, "{files} fixture files");
        assert!(refused >= 58, "{refused} refused dictionaries");
        assert!(words >= 4400, "{words} words");
    }
}
