//! M11 T11.1: the analysis-common differential harness.
//!
//! `fixtures/src/GenAnalysisCommon.java` runs every chain it names over each
//! line of `fixtures/corpus/analysis-common.txt` and writes
//! `fixtures/data/analysis_common/<chain>.tsv`: every token's term, offsets,
//! position increment and length, type, flags, payload, keyword flag and term
//! frequency, then the state `end()` leaves, or the exception a line throws.
//! This test builds the same chain in Rust ([`build`]), runs the same lines
//! through one reused [`Analyzer`], formats the rows the same way and compares
//! them line for line.
//!
//! A chain the port does not build yet is listed in [`PENDING`]; the test
//! fails on a fixture that is neither built nor pending, and on a pending name
//! with no fixture, so the two lists cannot drift from the generator.
//!
//! One deliberate normalisation: a Java term may hold an unpaired surrogate
//! (a filter cutting between the halves of a pair, `MinHashFilter`'s hash
//! chars); a Rust term cannot, and holds U+FFFD there -- which is also what
//! Java's `getBytesRef()`, the only way a term reaches an index, produces.
//! The expected rows' `\uD800`..`\uDFFF` escapes are compared as U+FFFD.

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::reader::CharReader;
use lucene_analysis::util::{LetterTokenizer, WhitespaceTokenizer};
use lucene_analysis::{
    core_analysis, AnalysisError, Analyzer, AnalyzerDefinition, CharArraySet, KeywordTokenizer,
    LowerCaseFilter, StandardAnalyzer, StandardTokenizer, StopFilter, TokenStream,
    TokenStreamComponents, ENGLISH_STOP_WORDS,
};

fn dir() -> String {
    concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/analysis_common/"
    )
    .to_string()
}

fn corpus() -> Vec<String> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/corpus/analysis-common.txt"
    );
    let text = std::fs::read_to_string(path).expect("the analysis-common corpus");
    // Split on '\n' only, as the generator does: lines hold U+2028 and U+0085.
    let mut lines: Vec<String> = text.split('\n').map(str::to_string).collect();
    if lines.last().is_some_and(String::is_empty) {
        lines.pop();
    }
    lines
}

/// The chains the generator writes that the port does not build yet.
const PENDING: &[&str] = &[
    "std_concatenate_graph",
    "ngram_tokenizer_1_2",
    "ngram_tokenizer_2_3",
    "edge_ngram_tokenizer_1_3",
    "std_ngram_filter_2_3",
    "std_ngram_filter_2_3_preserve",
    "std_edge_ngram_filter_1_4",
    "std_edge_ngram_filter_2_3_preserve",
    "std_shingle_default",
    "std_shingle_2_3_no_unigrams",
    "std_shingle_unigrams_if_none",
    "std_fixed_shingle_3",
    "pattern_tokenizer_split",
    "pattern_tokenizer_group",
    "simple_pattern_tokenizer",
    "simple_pattern_split_tokenizer",
    "ws_pattern_replace_all",
    "ws_pattern_replace_first",
    "ws_pattern_capture_group",
    "pattern_replace_char_filter",
    "path_hierarchy",
    "path_hierarchy_backslash_skip1",
    "reverse_path_hierarchy",
    "reverse_path_hierarchy_dot_skip1",
    "mapping_char_filter",
    "html_strip_standard",
    "html_strip_keyword",
    "html_strip_escaped_b",
    "std_common_grams",
    "std_common_grams_query",
    "cjk_analyzer",
    "std_cjk_bigram_unigrams",
    "std_cjk_bigram_han_only",
    "ws_cjk_width",
    "cjk_width_char_filter",
    "ws_delimited_payload_float",
    "ws_delimited_payload_int",
    "ws_delimited_payload_identity",
    "std_numeric_payload",
    "std_type_as_payload",
    "std_token_offset_payload",
    "ws_delimited_boost",
    "ws_shingle_minhash",
    "ws_minhash_64_buckets",
    "ws_minhash_no_rotation",
    "uax29_url_email_analyzer",
    "uax29_url_email_tokenizer",
    "std_kstem",
];

// ---------------------------------------------------------------- chains

type Sink = Result<TokenStreamComponents, AnalysisError>;
type ReaderWrap = Box<dyn Fn(Box<dyn CharReader>) -> Box<dyn CharReader> + Send + Sync>;

/// The generator's `chain(charFilters, tokenizer, filters)`: an analyzer
/// defined by a component factory and an `initReader` hook.
struct Chain {
    components: Box<dyn Fn() -> Sink + Send + Sync>,
    char_filters: Option<ReaderWrap>,
}

impl AnalyzerDefinition for Chain {
    fn create_components(&self, _field: &str) -> Sink {
        (self.components)()
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        match &self.char_filters {
            Some(wrap) => wrap(reader),
            None => reader,
        }
    }
}

fn chain(f: impl Fn() -> Sink + Send + Sync + 'static) -> Analyzer {
    Analyzer::new(Chain {
        components: Box::new(f),
        char_filters: None,
    })
}

fn comps(sink: impl TokenStream + 'static) -> Sink {
    Ok(TokenStreamComponents::new(sink))
}

fn set(ignore_case: bool, words: &[&str]) -> Arc<CharArraySet> {
    Arc::new(CharArraySet::from_words(words, ignore_case))
}

/// The generator's `WDGF_DEFAULT`.
const WDGF_DEFAULT: i32 = lucene_analysis::miscellaneous::GENERATE_WORD_PARTS
    | lucene_analysis::miscellaneous::GENERATE_NUMBER_PARTS
    | lucene_analysis::miscellaneous::SPLIT_ON_CASE_CHANGE
    | lucene_analysis::miscellaneous::SPLIT_ON_NUMERICS
    | lucene_analysis::miscellaneous::STEM_ENGLISH_POSSESSIVE;

fn english() -> Arc<CharArraySet> {
    Arc::new(CharArraySet::from_words(ENGLISH_STOP_WORDS, false))
}

/// The Rust twin of each generator chain, `None` while it is pending.
fn build(name: &str) -> Option<Analyzer> {
    use core_analysis::*;
    use lucene_analysis::en::*;
    use lucene_analysis::miscellaneous::{self as m, *};
    use lucene_analysis::util::{ElisionFilter, JavaPattern};
    Some(match name {
        // ---- core
        "standard_analyzer" => Analyzer::new(StandardAnalyzer::default()),
        "keyword_analyzer" => Analyzer::keyword(),
        "whitespace_analyzer" => Analyzer::new(WhitespaceAnalyzer::default()),
        "whitespace_max5" => Analyzer::new(WhitespaceAnalyzer::with_max_token_length(5)),
        "unicode_whitespace_analyzer" => Analyzer::new(UnicodeWhitespaceAnalyzer),
        "letter_tokenizer" => chain(|| comps(LetterTokenizer::new())),
        "letter_max3" => chain(|| comps(LetterTokenizer::with_max_token_len(3)?)),
        "simple_analyzer" => Analyzer::new(SimpleAnalyzer),
        "stop_analyzer" => Analyzer::new(StopAnalyzer::new(english())),
        "ws_lowercase" => chain(|| comps(LowerCaseFilter::new(WhitespaceTokenizer::new()))),
        "ws_uppercase" => chain(|| comps(UpperCaseFilter::new(WhitespaceTokenizer::new()))),
        "ws_decimal_digit" => chain(|| comps(DecimalDigitFilter::new(WhitespaceTokenizer::new()))),
        "std_stop_lower" => chain(|| {
            comps(StopFilter::new(
                LowerCaseFilter::new(StandardTokenizer::new()),
                english(),
            ))
        }),
        "std_type_drop_num" => chain(|| {
            comps(TypeTokenFilter::new(
                StandardTokenizer::new(),
                ["<NUM>"],
                false,
            ))
        }),
        "std_type_keep_alnum" => chain(|| {
            comps(TypeTokenFilter::new(
                StandardTokenizer::new(),
                ["<ALPHANUM>"],
                true,
            ))
        }),
        "keyword_tokenizer" => chain(|| comps(KeywordTokenizer::new())),

        // ---- miscellaneous
        "ws_wdgf_flatten" => chain(|| {
            comps(FlattenGraphFilter::new(WordDelimiterGraphFilter::new(
                WhitespaceTokenizer::new(),
                WDGF_DEFAULT | m::CATENATE_ALL | m::PRESERVE_ORIGINAL,
                None,
            )?))
        }),
        "std_ascii_folding" => chain(|| {
            comps(AsciiFoldingTokenFilter::new(
                StandardTokenizer::new(),
                false,
            ))
        }),
        "ws_ascii_folding_preserve" => chain(|| {
            comps(AsciiFoldingTokenFilter::new(
                WhitespaceTokenizer::new(),
                true,
            ))
        }),
        "ws_wdgf_default" => chain(|| {
            comps(WordDelimiterGraphFilter::new(
                WhitespaceTokenizer::new(),
                WDGF_DEFAULT,
                None,
            )?)
        }),
        "ws_wdgf_catenate" => chain(|| {
            comps(WordDelimiterGraphFilter::new(
                WhitespaceTokenizer::new(),
                WDGF_DEFAULT
                    | m::CATENATE_WORDS
                    | m::CATENATE_NUMBERS
                    | m::CATENATE_ALL
                    | m::PRESERVE_ORIGINAL,
                Some(set(false, &["AT&T", "j2se"])),
            )?)
        }),
        "ws_wdgf_offsets_off" => chain(|| {
            let table: Arc<[u8]> = Arc::from(&m::DEFAULT_WORD_DELIM_TABLE[..]);
            comps(WordDelimiterGraphFilter::with_table(
                WhitespaceTokenizer::new(),
                false,
                table,
                m::GENERATE_WORD_PARTS | m::CATENATE_WORDS,
                None,
            )?)
        }),
        "std_length_2_5" => chain(|| comps(LengthFilter::new(StandardTokenizer::new(), 2, 5)?)),
        "std_codepoint_count_1_3" => {
            chain(|| comps(CodepointCountFilter::new(StandardTokenizer::new(), 1, 3)?))
        }
        "keyword_trim" => chain(|| comps(TrimFilter::new(KeywordTokenizer::new()))),
        "std_truncate_4" => chain(|| {
            comps(TruncateTokenFilter::truncate_after_chars(
                StandardTokenizer::new(),
                4,
            )?)
        }),
        "std_truncate_cp_2" => chain(|| {
            comps(TruncateTokenFilter::truncate_after_code_points(
                StandardTokenizer::new(),
                2,
            )?)
        }),
        "std_limit_count_3" => chain(|| {
            comps(LimitTokenCountFilter::new(
                StandardTokenizer::new(),
                3,
                false,
            )?)
        }),
        "std_limit_count_3_all" => chain(|| {
            comps(LimitTokenCountFilter::new(
                StandardTokenizer::new(),
                3,
                true,
            )?)
        }),
        "std_limit_offset_20" => chain(|| {
            comps(LimitTokenOffsetFilter::new(
                StandardTokenizer::new(),
                20,
                false,
            )?)
        }),
        "std_limit_position_4" => chain(|| {
            comps(LimitTokenPositionFilter::new(
                StandardTokenizer::new(),
                4,
                false,
            )?)
        }),
        "ws_keyword_marker_porter" => chain(|| {
            comps(PorterStemFilter::new(SetKeywordMarkerFilter::new(
                LowerCaseFilter::new(WhitespaceTokenizer::new()),
                set(false, &["running", "happiness"]),
            )))
        }),
        "ws_pattern_keyword_porter" => chain(|| {
            comps(PorterStemFilter::new(
                PatternKeywordMarkerFilter::with_pattern(
                    LowerCaseFilter::new(WhitespaceTokenizer::new()),
                    JavaPattern::compile("[a-z]+ing")?,
                ),
            ))
        }),
        "ws_keyword_repeat_porter_dedup" => chain(|| {
            comps(RemoveDuplicatesTokenFilter::new(PorterStemFilter::new(
                KeywordRepeatFilter::new(LowerCaseFilter::new(WhitespaceTokenizer::new())),
            )))
        }),
        "ws_stemmer_override_porter" => chain(|| {
            let mut b = StemmerOverrideBuilder::new(true);
            b.add("running", "run!");
            b.add("Happiness", "joy");
            b.add("dogs", "dog");
            comps(PorterStemFilter::new(StemmerOverrideFilter::new(
                WhitespaceTokenizer::new(),
                b.build(),
            )))
        }),
        "std_elision" => chain(|| {
            comps(ElisionFilter::new(
                StandardTokenizer::new(),
                set(
                    true,
                    &[
                        "l", "m", "t", "qu", "n", "s", "j", "d", "c", "jusqu", "quoiqu", "lorsqu",
                        "puisqu",
                    ],
                ),
            ))
        }),
        "ws_capitalization" => {
            chain(|| comps(CapitalizationFilter::new(WhitespaceTokenizer::new())))
        }
        "ws_capitalization_custom" => chain(|| {
            comps(CapitalizationFilter::with_options(
                WhitespaceTokenizer::new(),
                false,
                Some(set(true, &["the", "and"])),
                true,
                Some(vec!["mc".to_string()]),
                2,
                4,
                6,
            )?)
        }),
        "ws_remove_duplicates" => chain(|| {
            comps(RemoveDuplicatesTokenFilter::new(LowerCaseFilter::new(
                WhitespaceTokenizer::new(),
            )))
        }),
        "std_fingerprint" => chain(|| {
            comps(FingerprintFilter::new(
                LowerCaseFilter::new(StandardTokenizer::new()),
                1024,
                ' ',
            ))
        }),
        "std_fingerprint_small" => {
            chain(|| comps(FingerprintFilter::new(StandardTokenizer::new(), 20, '_')))
        }
        "ws_delimited_term_frequency" => chain(|| {
            comps(DelimitedTermFrequencyTokenFilter::new(
                WhitespaceTokenizer::new(),
            ))
        }),
        "ws_protected_term" => chain(|| {
            comps(protected_term_filter(
                set(false, &["BROWN", "The"]),
                WhitespaceTokenizer::new(),
                LowerCaseFilter::new,
            ))
        }),
        "ws_conditional_lower" => chain(|| {
            comps(ConditionalTokenFilter::new(
                WhitespaceTokenizer::new(),
                |a: &AttributeSource| a.term_utf16_len() > 3,
                LowerCaseFilter::new,
            ))
        }),
        "ws_keep_word" => chain(|| {
            comps(KeepWordFilter::new(
                WhitespaceTokenizer::new(),
                set(true, &["the", "fox", "dog", "quick"]),
            ))
        }),
        "ws_hyphenated_words" => {
            chain(|| comps(HyphenatedWordsFilter::new(WhitespaceTokenizer::new())))
        }
        "std_type_as_synonym" => chain(|| {
            comps(TypeAsSynonymFilter::new(
                StandardTokenizer::new(),
                Some("_type_"),
                None,
                !0,
            ))
        }),
        "ws_scandinavian_folding" => {
            chain(|| comps(ScandinavianFoldingFilter::new(WhitespaceTokenizer::new())))
        }
        "ws_scandinavian_normalization" => chain(|| {
            comps(ScandinavianNormalizationFilter::new(
                WhitespaceTokenizer::new(),
            ))
        }),
        "std_fix_broken_offsets" => {
            chain(|| comps(FixBrokenOffsetsFilter::new(StandardTokenizer::new())))
        }
        "std_drop_if_flagged" => {
            chain(|| comps(DropIfFlaggedFilter::new(StandardTokenizer::new(), 1)))
        }

        // ---- en
        "english_analyzer" => Analyzer::new(EnglishAnalyzer::default()),
        "std_english_possessive" => {
            chain(|| comps(EnglishPossessiveFilter::new(StandardTokenizer::new())))
        }
        "std_porter" => chain(|| {
            comps(PorterStemFilter::new(LowerCaseFilter::new(
                StandardTokenizer::new(),
            )))
        }),
        "std_english_minimal" => chain(|| {
            comps(EnglishMinimalStemFilter::new(LowerCaseFilter::new(
                StandardTokenizer::new(),
            )))
        }),
        _ => return None,
    })
}

// ---------------------------------------------------------------- rows

/// The generator's `esc`.
fn esc(s: &str) -> String {
    let mut b = String::with_capacity(s.len());
    for ch in s.chars() {
        match ch {
            '\\' => b.push_str("\\\\"),
            '\t' => b.push_str("\\t"),
            '\n' => b.push_str("\\n"),
            '\r' => b.push_str("\\r"),
            c if (c as u32) < 0x20 => b.push_str(&format!("\\u{:04X}", c as u32)),
            c => b.push(c),
        }
    }
    b
}

fn hex(b: Option<&[u8]>) -> String {
    match b {
        None => "-".to_string(),
        Some(b) => b.iter().map(|x| format!("{x:02x}")).collect(),
    }
}

/// Java's exception class for an [`AnalysisError`].
fn exception_name(e: &AnalysisError) -> &'static str {
    match e {
        AnalysisError::IllegalArgument(m) if m.starts_with("NumberFormatException") => {
            "NumberFormatException"
        }
        AnalysisError::IllegalArgument(_) => "IllegalArgumentException",
        AnalysisError::IllegalState(_) => "IllegalStateException",
        AnalysisError::AlreadyClosed(_) => "AlreadyClosedException",
        AnalysisError::Io(_) => "IOException",
    }
}

fn token_row(ln: usize, a: &AttributeSource) -> String {
    let term = match a.bytes_term() {
        Some(bytes) => format!("#{}", hex(Some(bytes))),
        None => esc(a.term()),
    };
    format!(
        "T\t{ln}\t{term}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
        a.start_offset(),
        a.end_offset(),
        a.position_increment(),
        a.position_length(),
        esc(a.token_type()),
        a.flags(),
        hex(a.payload()),
        u8::from(a.is_keyword()),
        a.term_frequency()
    )
}

fn analyze_line(a: &Analyzer, ln: usize, line: &str, out: &mut Vec<String>) {
    let mut ts = match a.token_stream("f", line) {
        Ok(ts) => ts,
        Err(e) => {
            out.push(format!("X\t{ln}\t{}", exception_name(&e)));
            return;
        }
    };
    let run = |ts: &mut dyn TokenStream, out: &mut Vec<String>| -> Result<(), AnalysisError> {
        ts.reset()?;
        while ts.increment_token()? {
            out.push(token_row(ln, ts.attributes()));
        }
        ts.end()?;
        let e = ts.attributes();
        out.push(format!(
            "E\t{ln}\t{}\t{}\t{}",
            e.start_offset(),
            e.end_offset(),
            e.position_increment()
        ));
        Ok(())
    };
    if let Err(e) = run(&mut ts, out) {
        out.push(format!("X\t{ln}\t{}", exception_name(&e)));
    }
    let _ = ts.close();
}

/// Expected rows with Java's lone-surrogate escapes read as U+FFFD (see the
/// module docs).
fn normalise_expected(row: &str) -> String {
    let mut out = String::with_capacity(row.len());
    let mut rest = row;
    while let Some(i) = rest.find("\\u") {
        let (head, tail) = rest.split_at(i);
        out.push_str(head);
        // An escaped backslash before the `u` is a literal `\` then `u`.
        let escaped_backslash = head.chars().rev().take_while(|&c| c == '\\').count() % 2 == 1;
        let code = tail.get(2..6).and_then(|h| u32::from_str_radix(h, 16).ok());
        match code {
            Some(c) if !escaped_backslash && (0xD800..=0xDFFF).contains(&c) => {
                out.push('\u{FFFD}');
                rest = &tail[6..];
            }
            _ => {
                out.push_str(&tail[..2]);
                rest = &tail[2..];
            }
        }
    }
    out.push_str(rest);
    out
}

fn fixture_names() -> BTreeSet<String> {
    std::fs::read_dir(dir())
        .expect(
            "fixtures/data/analysis_common (run scripts/gen-fixtures.sh --only GenAnalysisCommon)",
        )
        .map(|e| e.unwrap().file_name().into_string().unwrap())
        .filter_map(|f| f.strip_suffix(".tsv").map(str::to_string))
        .collect()
}

#[test]
fn every_fixture_is_built_or_pending() {
    let names = fixture_names();
    assert!(names.len() > 100, "{} fixtures", names.len());
    for name in &names {
        let built = build(name).is_some();
        let pending = PENDING.contains(&name.as_str());
        assert!(built != pending, "{name}: built={built} pending={pending}");
    }
    for p in PENDING {
        assert!(names.contains(*p), "pending chain {p} has no fixture");
    }
}

#[test]
fn chains_match_lucene_token_for_token() {
    let lines = corpus();
    let mut checked = 0;
    for name in fixture_names() {
        let Some(analyzer) = build(&name) else {
            continue;
        };
        let expected_text = std::fs::read_to_string(format!("{}{name}.tsv", dir())).unwrap();
        let expected: Vec<String> = expected_text.lines().map(normalise_expected).collect();
        let mut actual = Vec::new();
        for (ln, line) in lines.iter().enumerate() {
            analyze_line(&analyzer, ln, line, &mut actual);
        }
        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(a, e, "{name}: row {i} differs");
        }
        assert_eq!(actual.len(), expected.len(), "{name}: row count");
        checked += 1;
    }
    assert!(checked >= 16, "only {checked} chains checked");
}

#[test]
fn normalising_reads_only_lone_surrogate_escapes() {
    assert_eq!(normalise_expected("a\\uD83Db"), "a\u{FFFD}b");
    assert_eq!(normalise_expected("a\\u0001b"), "a\\u0001b");
    assert_eq!(normalise_expected("a\\\\uD83D"), "a\\\\uD83D");
    assert_eq!(normalise_expected("\\u"), "\\u");
}
