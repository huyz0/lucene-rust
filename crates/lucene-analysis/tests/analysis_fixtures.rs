//! Differential test against real Lucene's `StandardAnalyzer`
//! (StandardTokenizer + LowerCaseFilter + StopFilter): six cases covering
//! the position-increment-preservation rule when stopwords are removed
//! (mid-sentence, leading, trailing, consecutive, all-stopwords) plus a
//! mixed-case/punctuation sentence exercising the tokenizer, lowercasing,
//! and stopword removal together ("The" is itself a stopword once
//! lowercased). Regenerate with fixtures/src/GenAnalysis.java.

use lucene_analysis::{
    Analyzer, AsciiFoldingFilter, LowerCaseFilter, SnowballEnglishStemFilter, StopFilter,
};
use std::collections::HashSet;

fn dir() -> String {
    concat!(env!("CARGO_MANIFEST_DIR"), "/../../fixtures/data/analysis/").to_string()
}

struct Manifest {
    kv: Vec<(String, String)>,
}

impl Manifest {
    fn load() -> Self {
        let text = std::fs::read_to_string(format!("{}manifest.properties", dir()))
            .expect("run fixtures generator first (GenAnalysis)");
        let kv = text
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        Manifest { kv }
    }

    fn get(&self, key: &str) -> &str {
        self.kv
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
    }
}

fn expected_tokens(m: &Manifest, case: &str) -> Vec<(String, i32, i32, i32)> {
    let raw = m.get(&format!("{case}.tokens"));
    if raw.is_empty() {
        return Vec::new();
    }
    raw.split(';')
        .map(|entry| {
            let mut parts = entry.split(':');
            let term = parts.next().unwrap().to_string();
            let pos_inc: i32 = parts.next().unwrap().parse().unwrap();
            let offsets = parts.next().unwrap();
            let (start, end) = offsets.split_once(',').unwrap();
            (term, pos_inc, start.parse().unwrap(), end.parse().unwrap())
        })
        .collect()
}

fn actual_tokens(text: &str, stopwords: &HashSet<String>) -> Vec<(String, i32, i32, i32)> {
    Analyzer::standard(Some(stopwords))
        .analyze(text)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect()
}

#[test]
fn matches_real_standard_analyzer_across_all_cases() {
    let m = Manifest::load();
    let stopwords: HashSet<String> = ["the", "a", "of"]
        .into_iter()
        .map(|s| s.to_string())
        .collect();

    for case in ["case1", "case2", "case3", "case4", "case5", "case6"] {
        let text = m.get(&format!("{case}.text"));
        let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
        let expected = expected_tokens(&m, case);
        assert_eq!(
            expected.len(),
            expected_count,
            "case {case}: manifest count mismatch"
        );

        let actual = actual_tokens(text, &stopwords);
        assert_eq!(
            actual, expected,
            "case {case} (text={text:?}) diverged from real Lucene"
        );
    }
}

/// Converts a manifest's expected `(term, position_increment, start_char,
/// end_char)` tuples -- real Lucene reports offsets in `char`/UTF-16-code-unit
/// units, which coincide for this fixture's BMP-only text -- into
/// `(term, position_increment, start_byte, end_byte)`, matching this port's
/// `tokenize()`, which (despite its doc comment calling them "character
/// offsets") actually emits UTF-8 **byte** offsets via `char_indices()`. This
/// byte-vs-char-offset unit reconciliation only matters once non-ASCII text
/// is involved (every pre-existing `lucene-analysis` fixture case is plain
/// ASCII, where the two units coincide) -- the same kind of documented
/// byte-vs-codepoint scope decision `fuzzy.rs`/`wildcard.rs` already make
/// elsewhere in this port, not a bug in `AsciiFoldingFilter` itself.
fn char_offsets_to_byte_offsets(
    text: &str,
    expected: Vec<(String, i32, i32, i32)>,
) -> Vec<(String, i32, i32, i32)> {
    // char index -> byte index, plus one past the last char index -> text.len().
    let mut char_to_byte: Vec<usize> = text.char_indices().map(|(b, _)| b).collect();
    char_to_byte.push(text.len());

    expected
        .into_iter()
        .map(|(term, pos_inc, start_char, end_char)| {
            (
                term,
                pos_inc,
                char_to_byte[start_char as usize] as i32,
                char_to_byte[end_char as usize] as i32,
            )
        })
        .collect()
}

/// Task #64 (ASCIIFoldingFilter) cross-engine check: real
/// `ASCIIFoldingFilter` (fold only, no lowercasing) run over a string
/// containing several Latin-1/Latin-Extended-A diacritics, the special
/// eszett case, and a ligature ("café naïve Müller cœur straße"), recorded
/// by `fixtures/src/GenAnalysis.java`'s `fold_only` case. This asserts this
/// port's `AsciiFoldingFilter::apply` produces the same (term,
/// position_increment, offset-span) sequence as real Lucene, after
/// reconciling the char-vs-byte offset unit (see
/// `char_offsets_to_byte_offsets`).
#[test]
fn ascii_folding_matches_real_ascii_folding_filter() {
    let m = Manifest::load();
    let case = "fold_only";
    let text = m.get(&format!("{case}.text"));
    let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
    let expected = expected_tokens(&m, case);
    assert_eq!(expected.len(), expected_count, "manifest count mismatch");
    let expected = char_offsets_to_byte_offsets(text, expected);

    let tokens = lucene_analysis::tokenize(text);
    let actual: Vec<(String, i32, i32, i32)> = AsciiFoldingFilter::apply(tokens)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect();

    assert_eq!(actual, expected, "fold-only case diverged from real Lucene");
}

/// Task #207 (full UAX#29-style tokenizer) cross-engine check: bare real
/// `StandardTokenizer` output (no filters) over strings exercising combining
/// marks, CJK ideograph segmentation, precomposed and conjoining-Jamo Hangul
/// syllables, mixed CJK/Latin text, and midword punctuation (numeric
/// decimal/comma, acronym periods, apostrophe contraction) -- recorded by
/// `fixtures/src/GenAnalysis.java`'s `uax29_*` cases. Confirms this port's
/// `tokenize()` (now backed by the `unicode-segmentation` crate's UAX#29
/// word-boundary implementation) agrees with real Lucene on all of these,
/// after reconciling the char-vs-byte offset unit (see
/// `char_offsets_to_byte_offsets`).
#[test]
fn tokenize_matches_real_standard_tokenizer_on_uax29_cases() {
    let m = Manifest::load();
    for case in [
        "uax29_combining_mark",
        "uax29_cjk",
        "uax29_hangul_precomposed",
        "uax29_hangul_jamo",
        "uax29_mixed_cjk_latin",
        "uax29_midword_punct",
    ] {
        let text = m.get(&format!("{case}.text"));
        let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
        let expected = expected_tokens(&m, case);
        assert_eq!(
            expected.len(),
            expected_count,
            "case {case}: count mismatch"
        );
        let expected = char_offsets_to_byte_offsets(text, expected);

        let actual: Vec<(String, i32, i32, i32)> = lucene_analysis::tokenize(text)
            .into_iter()
            .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
            .collect();

        assert_eq!(
            actual, expected,
            "case {case} (text={text:?}) diverged from real Lucene"
        );
    }
}

/// Task #64 cross-engine check for the composed `Analyzer::with_ascii_folding`
/// chain (fold, then lowercase): `fixtures/src/GenAnalysis.java`'s
/// `fold_then_lower` case runs real `ASCIIFoldingFilter` followed by real
/// `LowerCaseFilter` over "Café Naïve ÉCOLE".
#[test]
fn ascii_folding_then_lowercase_matches_real_lucene() {
    let m = Manifest::load();
    let case = "fold_then_lower";
    let text = m.get(&format!("{case}.text"));
    let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
    let expected = expected_tokens(&m, case);
    assert_eq!(expected.len(), expected_count, "manifest count mismatch");
    let expected = char_offsets_to_byte_offsets(text, expected);

    let analyzer = Analyzer::standard(None).with_ascii_folding();
    let actual: Vec<(String, i32, i32, i32)> = analyzer
        .analyze(text)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect();

    assert_eq!(
        actual, expected,
        "fold-then-lower case diverged from real Lucene"
    );
}

/// Task #208 (second analyzer-chain producer, `Analyzer::keyword`) cross-engine
/// check: real `KeywordAnalyzer` (bare `KeywordTokenizer`, no filters) over a
/// handful of representative inputs -- a plain id-like string, a mixed-case
/// string with punctuation that would otherwise split under
/// `StandardAnalyzer`, embedded whitespace, non-ASCII text, and the empty
/// string -- recorded by `fixtures/src/GenAnalysis.java`'s `keyword_*` cases.
/// Confirms `Analyzer::keyword` always emits exactly the whole input as a
/// single unmodified token (case preserved, no splitting, no offset
/// adjustment), including real Lucene's non-obvious empty-input behavior:
/// `KeywordTokenizer` still emits one (empty) token rather than zero.
#[test]
fn keyword_analyzer_matches_real_keyword_analyzer() {
    let m = Manifest::load();
    for case in [
        "keyword_simple",
        "keyword_mixed_case_punct",
        "keyword_whitespace",
        "keyword_non_ascii",
        "keyword_empty",
    ] {
        let text = m.get(&format!("{case}.text"));
        let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
        let expected = expected_tokens(&m, case);
        assert_eq!(
            expected.len(),
            expected_count,
            "case {case}: manifest count mismatch"
        );
        let expected = char_offsets_to_byte_offsets(text, expected);

        let actual: Vec<(String, i32, i32, i32)> = Analyzer::keyword()
            .analyze(text)
            .into_iter()
            .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
            .collect();

        assert_eq!(
            actual, expected,
            "case {case} (text={text:?}) diverged from real Lucene"
        );
    }
}

/// Task #209 (Porter2/Snowball English stemmer) cross-engine check: real
/// `SnowballFilter` constructed with a real `EnglishStemmer`
/// (`org.tartarus.snowball.ext.EnglishStemmer`, generated from Snowball's
/// `english.sbl` -- the actual Porter2 algorithm, a different filter than
/// `EnglishAnalyzer`'s default classic-Porter `PorterStemFilter`), run over
/// `StandardTokenizer` + `LowerCaseFilter` output for a 112-word list
/// covering: the full step 1a plural family (`sses`/`ied`/`ies`/`ss`/`us`/
/// plain `s`), step 1b's `eed`/`eedly` protected-stem exceptions
/// (`proceed`/`exceed`/`succeed` staying unchanged) and its `ing`-only
/// special cases (`dying`/`lying`/`tying` -> `die`/`lie`/`tie`), the R1
/// irregular-prefix words (`arsenal`/`commune`/`emergency`/
/// `generalization`/`organization`/`pastime`/`university`/`generalize`/
/// `generous`/`lately`), the whole-word exception table
/// (`skis`/`skies`/`idly`/`gently`/`ugly`/`early`/`only`/`singly`/`sky`
/// plus the untouched `andes`/`atlas`/`bias`/`cosmos`/`news`/`howe`), the
/// full step 2/3/4 suffix families, step 5's `e`/`ll` handling
/// (`controll`->`control` vs. `roll` staying unchanged), and step 0's
/// apostrophe/possessive handling (`don't`/`doesn't`/`cats'`/`o'clock`/
/// `'tis`) -- recorded by `fixtures/src/GenAnalysis.java`'s
/// `snowball_english` case. Confirms this port's
/// `SnowballEnglishStemFilter` produces byte-for-byte identical terms (and
/// matching offsets/position-increments) to real Lucene's Porter2 stemmer.
#[test]
fn snowball_english_stemmer_matches_real_snowball_english_stemmer() {
    let m = Manifest::load();
    let case = "snowball_english";
    let text = m.get(&format!("{case}.text"));
    let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
    let expected = expected_tokens(&m, case);
    assert_eq!(expected.len(), expected_count, "manifest count mismatch");
    let expected = char_offsets_to_byte_offsets(text, expected);

    let tokens = lucene_analysis::tokenize(text);
    let tokens = lucene_analysis::LowerCaseFilter::apply(tokens);
    let actual: Vec<(String, i32, i32, i32)> = SnowballEnglishStemFilter::apply(tokens)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect();

    assert_eq!(
        actual, expected,
        "snowball_english case diverged from real Lucene"
    );
}

/// Task #220 (French default stopword list) cross-engine check: real
/// `StandardTokenizer` + `LowerCaseFilter` + `StopFilter` fed
/// `FrenchAnalyzer.getDefaultStopSet()` directly -- deliberately *not* the
/// full `FrenchAnalyzer` (no elision, no French stemming; see
/// [`lucene_analysis::FRENCH_STOP_WORDS`]'s doc comment for that scope
/// boundary) -- run over a French sentence containing five of the 154
/// default French stopwords ("le", "et", "la", "sont", "dans") interleaved
/// with three content words, recorded by `fixtures/src/GenAnalysis.java`'s
/// `french_stopwords` case. Confirms this port's `french_stop_words()`, fed
/// through the existing `StopFilter`, produces byte-identical (term,
/// position_increment, offset-span) output to real Lucene -- i.e. that the
/// 154-word list is not just a plausible-looking transcription but actually
/// matches real Lucene's stopword-removal behavior end-to-end, including
/// position-increment carry-over across the repeated consecutive stopwords.
#[test]
fn french_stop_words_match_real_french_analyzer_default_stop_set() {
    let m = Manifest::load();
    let case = "french_stopwords";
    let text = m.get(&format!("{case}.text"));
    let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
    let expected = expected_tokens(&m, case);
    assert_eq!(expected.len(), expected_count, "manifest count mismatch");
    let expected = char_offsets_to_byte_offsets(text, expected);

    let tokens = lucene_analysis::tokenize(text);
    let tokens = LowerCaseFilter::apply(tokens);
    let stopwords = lucene_analysis::french_stop_words();
    let actual: Vec<(String, i32, i32, i32)> = StopFilter::apply(tokens, &stopwords)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect();

    assert_eq!(
        actual, expected,
        "french_stopwords case diverged from real Lucene"
    );
}
