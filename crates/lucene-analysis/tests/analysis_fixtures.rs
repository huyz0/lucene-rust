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

/// Java's `String.length()` for `text`: its UTF-16 code-unit count, the unit
/// every offset in this fixture is expressed in on both sides.
fn utf16_len(text: &str) -> i32 {
    text.encode_utf16().count() as i32
}

/// Task #64 (ASCIIFoldingFilter) cross-engine check: real
/// `ASCIIFoldingFilter` (fold only, no lowercasing) run over a string
/// containing several Latin-1/Latin-Extended-A diacritics, the special
/// eszett case, and a ligature ("café naïve Müller cœur straße"), recorded
/// by `fixtures/src/GenAnalysis.java`'s `fold_only` case. This asserts this
/// port's `AsciiFoldingFilter::apply` produces the same (term,
/// position_increment, offset-span) sequence as real Lucene -- including that
/// the offsets are the *source* text's span, unchanged by the ligature
/// growing the term.
#[test]
fn ascii_folding_matches_real_ascii_folding_filter() {
    let m = Manifest::load();
    let case = "fold_only";
    let text = m.get(&format!("{case}.text"));
    let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
    let expected = expected_tokens(&m, case);
    assert_eq!(expected.len(), expected_count, "manifest count mismatch");

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
/// word-boundary implementation) agrees with real Lucene on all of these --
/// terms, position increments and offsets, compared verbatim, both sides in
/// Java `char`s.
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

/// Parses a `<term>:<posInc>:<posLen>:<start>,<end>` token list -- the shape
/// `GenAnalysis.analyzeGraph` writes for the `SynonymGraphFilter` cases,
/// which are the only ones that need `PositionLengthAttribute`.
fn expected_graph_tokens(m: &Manifest, case: &str) -> Vec<(String, i32, i32, i32, i32)> {
    let raw = m.get(&format!("{case}.tokens"));
    if raw.is_empty() {
        return Vec::new();
    }
    raw.split(';')
        .map(|entry| {
            let mut parts = entry.split(':');
            let term = parts.next().unwrap().to_string();
            let pos_inc: i32 = parts.next().unwrap().parse().unwrap();
            let pos_len: i32 = parts.next().unwrap().parse().unwrap();
            let (start, end) = parts.next().unwrap().split_once(',').unwrap();
            (
                term,
                pos_inc,
                pos_len,
                start.parse().unwrap(),
                end.parse().unwrap(),
            )
        })
        .collect()
}

/// Terms and position increments only, for the cases whose single input token
/// is hand-built here rather than produced by a tokenizer: what those cases
/// assert is the *term* the filter produces, and their span is
/// `0..utf16_len(text)` by construction. The offset unit itself is pinned by
/// the `utf16_*` cases at the bottom of this file.
fn terms_and_increments(tokens: Vec<lucene_analysis::Token>) -> Vec<(String, i32)> {
    tokens
        .into_iter()
        .map(|t| (t.term, t.position_increment))
        .collect()
}

fn expected_terms_and_increments(m: &Manifest, case: &str) -> Vec<(String, i32)> {
    expected_tokens(m, case)
        .into_iter()
        .map(|(term, pos_inc, _, _)| (term, pos_inc))
        .collect()
}

/// b8 sweep: real `PorterStemFilter` -- `EnglishAnalyzer`'s default stemmer,
/// a different algorithm from the Snowball English one checked above -- over
/// a 100-word vocabulary. This is the check that pinned four divergences at
/// once: the missing `k > k0 + 1` length guard (which stemmed `"s"` to the
/// empty string), the missing lowercase-ASCII precondition on Java's side
/// (so `"Cats"` really does stem to `"Cat"`), the `bli -> ble` and
/// `logi -> log` rules, and Java's "first suffix that matches wins even if
/// its measure test then fails" `switch`/`break` structure.
#[test]
fn porter_stemmer_matches_real_porter_stem_filter() {
    let m = Manifest::load();
    let case = "porter_english";
    let text = m.get(&format!("{case}.text"));
    let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
    let expected = expected_tokens(&m, case);
    assert_eq!(expected.len(), expected_count, "manifest count mismatch");

    // Java's chain is WhitespaceTokenizer + PorterStemFilter, with no
    // lowercasing: the words are space-separated single tokens, which is what
    // `tokenize` produces for them too.
    let tokens = lucene_analysis::tokenize(text);
    let actual: Vec<(String, i32, i32, i32)> = lucene_analysis::PorterStemFilter::apply(tokens)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect();

    assert_eq!(actual, expected, "porter_english diverged from real Lucene");
}

/// b8 sweep: `ASCIIFoldingFilter` over one token per Unicode block the real
/// filter covers. This port used to ship 92 of Lucene's 1242 mappings, so
/// every one of these cases folded to itself here and to ASCII there.
#[test]
fn ascii_folding_covers_every_block_real_lucene_folds() {
    let m = Manifest::load();
    for case in [
        "fold_latin_ext_a",
        "fold_latin_ext_b",
        "fold_vietnamese",
        "fold_punctuation",
        "fold_superscripts",
        "fold_enclosed_alnum",
        "fold_ligatures",
        "fold_fullwidth",
    ] {
        let text = m.get(&format!("{case}.text"));
        let expected = expected_terms_and_increments(&m, case);
        // KeywordTokenizer on Java's side: the whole input is one token.
        let tokens = vec![lucene_analysis::Token {
            term: text.to_string(),
            start_offset: 0,
            end_offset: utf16_len(text),
            position_increment: 1,
            position_length: 1,
        }];
        let actual = terms_and_increments(AsciiFoldingFilter::apply(tokens));
        assert_eq!(actual, expected, "{case} diverged from real Lucene");
    }
}

/// b8 sweep: `LowerCaseFilter` uses Java's **simple** per-codepoint
/// `Character.toLowerCase`, not full Unicode lowercasing. `U+0130` maps to a
/// single `i`, and a word-final capital sigma stays a plain `σ` -- both of
/// which Rust's `str::to_lowercase` gets differently.
#[test]
fn lowercase_filter_matches_javas_simple_case_mapping() {
    let m = Manifest::load();
    for case in [
        "lowercase_dotted_capital_i",
        "lowercase_greek_final_sigma",
        "lowercase_german_sharp_s",
    ] {
        let text = m.get(&format!("{case}.text"));
        let expected = expected_terms_and_increments(&m, case);
        let tokens = vec![lucene_analysis::Token {
            term: text.to_string(),
            start_offset: 0,
            end_offset: utf16_len(text),
            position_increment: 1,
            position_length: 1,
        }];
        let actual = terms_and_increments(LowerCaseFilter::apply(tokens));
        assert_eq!(actual, expected, "{case} diverged from real Lucene");
    }
}

/// b8 sweep: `NGramTokenFilter`/`EdgeNGramTokenFilter`. The two behaviours
/// this pins are the ones the port had wrong: an input token too short to
/// produce a gram still consumes a position (Java's `curPosIncr` carry), and
/// every gram keeps the **input token's** offsets because Java
/// `restoreState`s rather than calling `setOffset`.
#[test]
fn ngram_filters_match_real_lucene() {
    let m = Manifest::load();
    /// A gram filter under test, as `(fixture case name, filter)`.
    type GramCase = (
        &'static str,
        fn(Vec<lucene_analysis::Token>) -> Vec<lucene_analysis::Token>,
    );
    let cases: &[GramCase] = &[
        ("ngram_skipped_short_token", |t| {
            lucene_analysis::NGramTokenFilter::apply(t, 3, 3).unwrap()
        }),
        ("ngram_offsets_are_the_input_tokens", |t| {
            lucene_analysis::NGramTokenFilter::apply(t, 2, 3).unwrap()
        }),
        ("ngram_preserve_original", |t| {
            lucene_analysis::NGramTokenFilter::apply_preserving_original(t, 2, 3).unwrap()
        }),
        ("edge_ngram_basic", |t| {
            lucene_analysis::EdgeNGramTokenFilter::apply(t, 2, 3).unwrap()
        }),
        ("edge_ngram_preserve_original", |t| {
            lucene_analysis::EdgeNGramTokenFilter::apply_preserving_original(t, 2, 3).unwrap()
        }),
    ];
    for (case, apply) in cases {
        let text = m.get(&format!("{case}.text"));
        let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
        let expected = expected_tokens(&m, case);
        assert_eq!(
            expected.len(),
            expected_count,
            "{case}: manifest count mismatch"
        );
        let actual: Vec<(String, i32, i32, i32)> = apply(lucene_analysis::tokenize(text))
            .into_iter()
            .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
            .collect();
        assert_eq!(actual, expected, "{case} diverged from real Lucene");
    }
}

/// b8 sweep: `SynonymGraphFilter`'s emission order, position increments,
/// position lengths and offsets, all of which come out of the synonym graph's
/// nodes. This port used to emit the originals first with the synonym at
/// increment 0 (one position too late) and stamp the whole match's offsets
/// onto every token (making `startOffset` decrease, which real Lucene's
/// `IndexingChain` rejects outright).
#[test]
fn synonym_graph_filter_matches_real_lucene() {
    use lucene_analysis::{SynonymFilter, SynonymRule};

    fn rule(input: &str, outputs: &[&str]) -> SynonymRule {
        SynonymRule {
            input: input.split(' ').map(str::to_string).collect(),
            outputs: outputs
                .iter()
                .map(|o| o.split(' ').map(str::to_string).collect())
                .collect(),
        }
    }

    let m = Manifest::load();
    let cases: Vec<(&str, Vec<SynonymRule>)> = vec![
        ("syn_multiword_to_single", vec![rule("wi fi", &["wifi"])]),
        (
            "syn_two_alternatives",
            vec![rule("wi fi", &["wifi", "wireless"])],
        ),
        (
            "syn_multiword_to_multiword",
            vec![rule("new york", &["big apple"])],
        ),
        (
            "syn_single_to_multiword",
            vec![rule("usa", &["united states of america"])],
        ),
        ("syn_in_context", vec![rule("wi fi", &["wifi"])]),
    ];

    for (case, rules) in cases {
        let text = m.get(&format!("{case}.text"));
        let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
        let expected = expected_graph_tokens(&m, case);
        assert_eq!(
            expected.len(),
            expected_count,
            "{case}: manifest count mismatch"
        );
        let actual: Vec<(String, i32, i32, i32, i32)> =
            SynonymFilter::apply_multiword(lucene_analysis::tokenize(text), &rules)
                .into_iter()
                .map(|t| {
                    (
                        t.term,
                        t.position_increment,
                        t.position_length,
                        t.start_offset,
                        t.end_offset,
                    )
                })
                .collect();
        assert_eq!(actual, expected, "{case} diverged from real Lucene");
    }
}

// ---------------------------------------------------------------------------
// c33 sweep: the offset **unit**.
//
// `OffsetAttribute` reports UTF-16 code-unit (Java `char`) indices into the
// original text. This port used to emit UTF-8 **byte** offsets, and
// `lucene-search`'s highlighter used to read Unicode **scalar** counts, so all
// three units were in play at once. Every fixture case above is BMP-or-ASCII
// enough that at most two of the three separate; the `utf16_*` cases below are
// chosen so that all three differ, and they are compared **verbatim** -- there
// is deliberately no conversion helper left in this file for a wrong unit to
// hide behind.
// ---------------------------------------------------------------------------

/// The UTF-8 byte offset and the Unicode scalar index of the position that
/// `utf16_offset` names in `text` -- the two units this port has previously
/// mistaken for Lucene's. Used only by the negative controls below, to prove a
/// case actually separates the three.
fn other_units(text: &str, utf16_offset: i32) -> (i32, i32) {
    let (mut u16s, mut scalars, mut bytes) = (0i32, 0i32, 0i32);
    for c in text.chars() {
        if u16s >= utf16_offset {
            break;
        }
        u16s += c.len_utf16() as i32;
        scalars += 1;
        bytes += c.len_utf8() as i32;
    }
    (bytes, scalars)
}

/// Whether `case`'s recorded offsets are ones a UTF-8 byte-offset producer,
/// respectively a Unicode-scalar-offset producer, would get **wrong**.
///
/// Byte offsets separate from Java `char`s for any non-ASCII text; scalar
/// offsets separate only for supplementary-plane text. A case that defeats
/// neither is evidence about nothing.
fn units_this_case_defeats(m: &Manifest, case: &str, offsets: &[i32]) -> (bool, bool) {
    let text = m.get(&format!("{case}.text"));
    let (mut byte_differs, mut scalar_differs) = (false, false);
    for off in offsets {
        let (bytes, scalars) = other_units(text, *off);
        byte_differs |= bytes != *off;
        scalar_differs |= scalars != *off;
    }
    (byte_differs, scalar_differs)
}

/// Every offset `case` records, start and end.
fn case_offsets(m: &Manifest, case: &str) -> Vec<i32> {
    expected_tokens(m, case)
        .into_iter()
        .flat_map(|(_, _, start, end)| [start, end])
        .collect()
}

/// Asserts a byte-offset producer fails `case` -- the minimum bar for a case
/// to be evidence about the unit at all.
fn assert_defeats_byte_offsets(m: &Manifest, case: &str) {
    let (byte_differs, _) = units_this_case_defeats(m, case, &case_offsets(m, case));
    assert!(
        byte_differs,
        "{case}: a UTF-8 byte-offset producer would pass this case -- it does \
         not pin the unit"
    );
}

/// Asserts a *scalar*-offset producer fails `case` too, i.e. that the case is
/// supplementary-plane enough to separate all three units at once.
fn assert_defeats_scalar_offsets(m: &Manifest, case: &str) {
    let (byte_differs, scalar_differs) = units_this_case_defeats(m, case, &case_offsets(m, case));
    assert!(
        byte_differs && scalar_differs,
        "{case}: does not separate all three units"
    );
}

/// c33: `tokenize()`'s offsets are Java `char` indices, verbatim against a
/// real `StandardTokenizer` over text where UTF-8 bytes, Unicode scalars and
/// UTF-16 code units all disagree: a Latin-1 accented letter (1 char, 2
/// bytes), a CJK ideograph (1 char, 3 bytes), a decomposed combining mark (2
/// chars, 3 bytes), a supplementary-plane symbol (2 chars, 1 scalar, 4 bytes)
/// and a supplementary-plane *letter*, which is a token in its own right whose
/// span is two chars per scalar.
#[test]
fn tokenizer_offsets_are_java_char_indices() {
    let m = Manifest::load();
    for case in [
        "utf16_latin1",
        "utf16_cjk_offsets",
        "utf16_astral_symbol",
        "utf16_astral_letter",
        "utf16_combining_mark_offsets",
        "utf16_all_units",
    ] {
        let text = m.get(&format!("{case}.text"));
        let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
        let expected = expected_tokens(&m, case);
        assert_eq!(
            expected.len(),
            expected_count,
            "case {case}: manifest count mismatch"
        );
        let actual: Vec<(String, i32, i32, i32)> = lucene_analysis::tokenize(text)
            .into_iter()
            .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
            .collect();
        assert_eq!(
            actual, expected,
            "case {case} (text={text:?}) diverged from real Lucene"
        );
    }
    // Negative controls. Every case must defeat a byte-offset producer; the
    // supplementary-plane ones must defeat a scalar-offset producer as well,
    // which is what makes `utf16_all_units` a single case no wrong unit passes.
    for case in [
        "utf16_latin1",
        "utf16_cjk_offsets",
        "utf16_astral_symbol",
        "utf16_astral_letter",
        "utf16_combining_mark_offsets",
        "utf16_all_units",
    ] {
        assert_defeats_byte_offsets(&m, case);
    }
    for case in [
        "utf16_astral_symbol",
        "utf16_astral_letter",
        "utf16_all_units",
    ] {
        assert_defeats_scalar_offsets(&m, case);
    }
}

/// c33: an emoji shifts every later token by exactly **two** Java `char`s --
/// one scalar, four UTF-8 bytes.
///
/// Real `StandardTokenizer` also emits the emoji itself as a token; this
/// port's `unicode_word_indices` does not (b8's F40, recorded there and
/// unchanged here -- it needs `split_word_bounds` plus an Extended_Pictographic
/// pass, i.e. a tokenizer rewrite, not an offset fix). So this case asserts the
/// exact shape of that gap: the tokens this port does produce must match
/// Lucene's *verbatim*, offsets included, and the only ones missing must be the
/// non-alphanumeric ones.
#[test]
fn an_emoji_shifts_later_offsets_by_two_java_chars() {
    let m = Manifest::load();
    let case = "utf16_emoji";
    let text = m.get(&format!("{case}.text"));
    let expected = expected_tokens(&m, case);

    let alphanumeric: Vec<(String, i32, i32, i32)> = expected
        .iter()
        .filter(|(term, ..)| term.chars().any(char::is_alphanumeric))
        .cloned()
        .collect();
    assert_eq!(
        expected.len() - alphanumeric.len(),
        1,
        "{case}: expected exactly one token dropped by b8's F40 (the emoji)"
    );

    let actual: Vec<(String, i32, i32, i32)> = lucene_analysis::tokenize(text)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect();
    assert_eq!(
        actual, alphanumeric,
        "{case}: the tokens this port does produce diverged from real Lucene"
    );
    // The whole point: "beta" starts two chars after the emoji's own start,
    // which is neither its scalar index (7) nor its byte index (11).
    let beta = actual.last().expect("beta");
    assert_eq!((beta.0.as_str(), beta.2), ("beta", 9));
    assert_eq!(other_units(text, beta.2), (11, 8));
}

/// c33: `Analyzer::keyword`'s single token ends at Java's
/// `correctOffset(charCount)` -- a `char` count, so for `"id-<emoji>-e-acute"`
/// it is 7, not the 6 scalars or the 10 UTF-8 bytes.
#[test]
fn keyword_analyzer_offsets_are_java_char_indices() {
    let m = Manifest::load();
    let case = "utf16_keyword_astral";
    let text = m.get(&format!("{case}.text"));
    let expected = expected_tokens(&m, case);
    let actual: Vec<(String, i32, i32, i32)> = Analyzer::keyword()
        .analyze(text)
        .into_iter()
        .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
        .collect();
    assert_eq!(actual, expected, "{case} diverged from real Lucene");
    assert_defeats_scalar_offsets(&m, case);
    assert_eq!(text.chars().count(), 6);
    assert_eq!(text.len(), 10);
}

/// c33: a filter that changes a term's *length* still reports the source
/// text's span. `ASCIIFoldingFilter` grows `"stra<sharp-s>e"` (6 chars) to
/// `"strasse"` (7) behind a supplementary-plane letter that has already
/// shifted the offsets, so a producer that re-derived the span from the folded
/// term would be visibly wrong at both ends.
#[test]
fn ascii_folding_keeps_the_source_span_after_an_astral_token() {
    let m = Manifest::load();
    let case = "utf16_fold_after_astral";
    let text = m.get(&format!("{case}.text"));
    let expected = expected_tokens(&m, case);
    let actual: Vec<(String, i32, i32, i32)> =
        AsciiFoldingFilter::apply(lucene_analysis::tokenize(text))
            .into_iter()
            .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
            .collect();
    assert_eq!(actual, expected, "{case} diverged from real Lucene");
    assert_defeats_scalar_offsets(&m, case);
    let folded = actual.last().expect("folded token");
    assert_eq!(folded.0, "strasse");
    // 7 characters of term, a 6-char span.
    assert_eq!(folded.3 - folded.2, 6);
}

/// c33: same for `PorterStemFilter` (`"running" -> "run"`, `"fishes" ->
/// "fish"`), and the supplementary-plane token itself passes through the
/// stemmer untouched, keeping its own two-`char` span.
#[test]
fn porter_stem_keeps_the_source_span_after_an_astral_token() {
    let m = Manifest::load();
    let case = "utf16_porter_after_astral";
    let text = m.get(&format!("{case}.text"));
    let expected = expected_tokens(&m, case);
    let actual: Vec<(String, i32, i32, i32)> =
        lucene_analysis::PorterStemFilter::apply(lucene_analysis::tokenize(text))
            .into_iter()
            .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
            .collect();
    assert_eq!(actual, expected, "{case} diverged from real Lucene");
    assert_defeats_scalar_offsets(&m, case);
}

/// c33: the n-gram filters `restoreState()` the input token's offsets, so
/// every gram of a non-ASCII token reports that token's Java `char` span --
/// not the gram's own, and not a byte span. `EdgeNGramTokenFilter` also has to
/// keep slicing by **code point** (`Character.offsetByCodePoints`) while
/// *reporting* code units: the first gram of `"<astral A>bc"` is two code
/// points and four `char`s.
#[test]
fn ngram_filters_keep_the_input_tokens_java_char_span() {
    let m = Manifest::load();
    /// A gram filter under test, as `(fixture case name, filter)`.
    type GramCase = (
        &'static str,
        fn(Vec<lucene_analysis::Token>) -> Vec<lucene_analysis::Token>,
    );
    let cases: &[GramCase] = &[
        ("utf16_ngram_offsets", |t| {
            lucene_analysis::NGramTokenFilter::apply(t, 2, 3).unwrap()
        }),
        ("utf16_edge_ngram_offsets", |t| {
            lucene_analysis::EdgeNGramTokenFilter::apply(t, 2, 3).unwrap()
        }),
    ];
    for (case, apply) in cases {
        let text = m.get(&format!("{case}.text"));
        let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
        let expected = expected_tokens(&m, case);
        assert_eq!(
            expected.len(),
            expected_count,
            "{case}: manifest count mismatch"
        );
        let actual: Vec<(String, i32, i32, i32)> = apply(lucene_analysis::tokenize(text))
            .into_iter()
            .map(|t| (t.term, t.position_increment, t.start_offset, t.end_offset))
            .collect();
        assert_eq!(actual, expected, "{case} diverged from real Lucene");
        assert_defeats_scalar_offsets(&m, case);
    }
}

/// c33: `SynonymGraphFilter`'s collapsed match spans from the first matched
/// token's `startOffset` to the last one's `endOffset`, and the originals keep
/// their own -- in Java `char`s, behind a Latin-1 letter and in front of a CJK
/// one. The emitted `startOffset`s must still be **non-decreasing**, which is
/// the rule b8 fixed (`IndexingChain` rejects a decreasing one outright) and
/// which a unit change is exactly the kind of edit that could regress.
#[test]
fn synonym_graph_offsets_are_java_char_indices_and_non_decreasing() {
    use lucene_analysis::{SynonymFilter, SynonymRule};

    let m = Manifest::load();
    let case = "utf16_syn_multiword";
    let text = m.get(&format!("{case}.text"));
    let expected_count: usize = m.get(&format!("{case}.count")).parse().unwrap();
    let expected = expected_graph_tokens(&m, case);
    assert_eq!(
        expected.len(),
        expected_count,
        "{case}: manifest count mismatch"
    );
    let rules = vec![SynonymRule {
        input: vec!["wi".to_string(), "fi".to_string()],
        outputs: vec![vec!["wifi".to_string()]],
    }];
    let out = SynonymFilter::apply_multiword(lucene_analysis::tokenize(text), &rules);
    assert!(
        out.windows(2)
            .all(|w| w[0].start_offset <= w[1].start_offset),
        "{case}: startOffset went backwards -- IndexingChain rejects that"
    );
    let actual: Vec<(String, i32, i32, i32, i32)> = out
        .into_iter()
        .map(|t| {
            (
                t.term,
                t.position_increment,
                t.position_length,
                t.start_offset,
                t.end_offset,
            )
        })
        .collect();
    assert_eq!(actual, expected, "{case} diverged from real Lucene");
    // The graph cases carry position lengths, so `expected_tokens`' parser
    // does not apply -- check the unit separation directly against the CJK
    // token's start offset, 9 `char`s in: 11 UTF-8 bytes, 8 Unicode scalars.
    assert_eq!(
        other_units(text, 9),
        (11, 8),
        "{case} separates all three units"
    );
}
