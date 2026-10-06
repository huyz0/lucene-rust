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
        HunspellError::Unsupported(_) => "UnsupportedOperationException",
    }
}

#[test]
fn hunspell_matches_lucene_on_every_dictionary() {
    let corpus = format!("{}corpus/hunspell/", root());
    let data = format!("{}data/hunspell/", root());
    let mut names: Vec<String> = std::fs::read_dir(&corpus)
        .unwrap()
        .filter_map(|e| {
            let n = e.unwrap().file_name().into_string().unwrap();
            n.strip_suffix(".aff").map(str::to_string)
        })
        .collect();
    names.sort();
    let (mut words, mut files, mut refused) = (0, 0, 0);
    for name in &names {
        let aff = std::fs::read(format!("{corpus}{name}.aff")).unwrap();
        let dic = std::fs::read(format!("{corpus}{name}.dic")).unwrap_or_default();
        for ignore_case in [false, true] {
            let file = format!("{data}{name}{}.tsv", if ignore_case { ".ic" } else { "" });
            let expected = std::fs::read_to_string(&file).unwrap_or_else(|e| panic!("{file}: {e}"));
            files += 1;
            let dictionary = match Dictionary::new(&aff, &[&dic], ignore_case) {
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
                    refused += 1;
                    continue;
                }
            };
            let h = Hunspell::with_timeout_policy(&dictionary, TimeoutPolicy::NoTimeout, None);
            let checker = NGramFragmentChecker::from_all_simple_words(2, &dictionary).unwrap();
            let tuned = Suggester::new(&dictionary)
                .with_fragment_checker(&checker)
                .proceed_past_rep();
            let roots: Vec<String> = expected
                .lines()
                .filter_map(|l| l.strip_prefix("E\t"))
                .map(|l| l.split('\t').next().unwrap().to_string())
                .collect();
            let root_refs: Vec<&str> = roots.iter().map(String::as_str).collect();
            let from_roots = NGramFragmentChecker::from_words(3, &root_refs).unwrap();
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
                        assert_eq!(
                            tuned.suggest_no_timeout(w).join("|"),
                            expect(f[9]),
                            "{file}: tuned suggest({w})"
                        );
                        let units: Vec<u16> = w.encode_utf16().collect();
                        let impossible =
                            from_roots.has_impossible_fragment_around(&units, 0, units.len());
                        assert_eq!(impossible, f[10] == "1", "{file}: fragment({w})");
                        words += 1;
                    }
                    other => panic!("{file}: unknown line kind {other}"),
                }
            }
        }
    }
    assert!(files >= 80, "{files} fixture files");
    assert!(refused >= 50, "{refused} refused dictionaries");
    assert!(words >= 4400, "{words} words");
}
