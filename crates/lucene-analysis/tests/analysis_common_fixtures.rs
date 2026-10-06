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
const PENDING: &[&str] = &[];

/// Chains whose `java.util.regex` pattern the port rejects (it would mean
/// something else over the `regex` crate; see `util/java_regex.rs`): the
/// fixture records Lucene's tokens, and the port must refuse the pattern
/// with `IllegalArgument` rather than produce different ones.
const REJECTED: &[(&str, &str)] = &[
    ("keyword_pattern_replace_word_boundary", "\\bthe\\b"),
    ("keyword_pattern_replace_multiline", "(?m)^"),
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

/// [`chain`] with an `initReader` char filter.
fn chain_cf(
    cf: impl Fn(Box<dyn CharReader>) -> Box<dyn CharReader> + Send + Sync + 'static,
    f: impl Fn() -> Sink + Send + Sync + 'static,
) -> Analyzer {
    Analyzer::new(Chain {
        components: Box::new(f),
        char_filters: Some(Box::new(cf)),
    })
}

/// `tokenizer` then a `PatternReplaceFilter(pattern, replacement, all)`.
fn replace_chain<T: TokenStream + 'static>(
    tokenizer: fn() -> T,
    pattern: &'static str,
    replacement: &'static str,
) -> Analyzer {
    chain(move || {
        comps(lucene_analysis::pattern::PatternReplaceFilter::new(
            tokenizer(),
            lucene_analysis::util::JavaPattern::compile(pattern)?,
            Some(replacement),
            true,
        ))
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
    use lucene_analysis::boost::DelimitedBoostTokenFilter;
    use lucene_analysis::charfilter::{
        HTMLStripCharFilter, MappingCharFilter, NormalizeCharMapBuilder,
    };
    use lucene_analysis::cjk::{
        self, CJKAnalyzer, CJKBigramFilter, CJKWidthCharFilter, CJKWidthFilter,
    };
    use lucene_analysis::commongrams::{CommonGramsFilter, CommonGramsQueryFilter};
    use lucene_analysis::email::{UAX29URLEmailAnalyzer, UAX29URLEmailTokenizer};
    use lucene_analysis::en::*;
    use lucene_analysis::minhash::MinHashFilter;
    use lucene_analysis::miscellaneous::{self as m, *};
    use lucene_analysis::ngram::{EdgeNGramTokenFilter, NGramTokenFilter, NGramTokenizer};
    use lucene_analysis::path::{PathHierarchyTokenizer, ReversePathHierarchyTokenizer};
    use lucene_analysis::pattern::*;
    use lucene_analysis::payloads::*;
    use lucene_analysis::shingle::{FixedShingleFilter, ShingleFilter};
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

        // ---- ngram
        "ngram_tokenizer_1_2" => chain(|| comps(NGramTokenizer::new(1, 2)?)),
        "ngram_tokenizer_2_3" => chain(|| comps(NGramTokenizer::new(2, 3)?)),
        "edge_ngram_tokenizer_1_3" => chain(|| comps(NGramTokenizer::edge(1, 3)?)),
        "std_ngram_filter_2_3" => chain(|| {
            comps(NGramTokenFilter::new(
                StandardTokenizer::new(),
                2,
                3,
                false,
            )?)
        }),
        "std_ngram_filter_2_3_preserve" => {
            chain(|| comps(NGramTokenFilter::new(StandardTokenizer::new(), 2, 3, true)?))
        }
        "std_edge_ngram_filter_1_4" => chain(|| {
            comps(EdgeNGramTokenFilter::new(
                StandardTokenizer::new(),
                1,
                4,
                false,
            )?)
        }),
        "std_edge_ngram_filter_2_3_preserve" => chain(|| {
            comps(EdgeNGramTokenFilter::new(
                StandardTokenizer::new(),
                2,
                3,
                true,
            )?)
        }),

        // ---- shingle
        "std_shingle_default" => {
            chain(|| comps(ShingleFilter::new(StandardTokenizer::new(), 2, 2)?))
        }
        "std_shingle_2_3_no_unigrams" => chain(|| {
            let mut s =
                ShingleFilter::new(StopFilter::new(StandardTokenizer::new(), english()), 2, 3)?;
            s.set_output_unigrams(false);
            s.set_token_separator(Some("+"));
            s.set_filler_token(Some("*"));
            comps(s)
        }),
        "std_shingle_unigrams_if_none" => chain(|| {
            let mut s = ShingleFilter::new(StandardTokenizer::new(), 3, 3)?;
            s.set_output_unigrams(false);
            s.set_output_unigrams_if_no_shingles(true);
            comps(s)
        }),
        "std_fixed_shingle_3" => chain(|| {
            comps(FixedShingleFilter::new(
                StopFilter::new(StandardTokenizer::new(), english()),
                3,
            )?)
        }),

        // ---- pattern
        "pattern_tokenizer_split" => chain(|| {
            comps(PatternTokenizer::new(
                &JavaPattern::compile("[ ,;.]+")?,
                -1,
            )?)
        }),
        "pattern_tokenizer_group" => chain(|| {
            comps(PatternTokenizer::new(
                &JavaPattern::compile("([a-z]+)([0-9]*)")?,
                1,
            )?)
        }),
        "simple_pattern_tokenizer" => {
            chain(|| comps(SimplePatternTokenizer::new("[a-zA-Z]+[0-9]*")?))
        }
        "simple_pattern_split_tokenizer" => {
            chain(|| comps(SimplePatternSplitTokenizer::new("[ \t,;.]+")?))
        }
        "ws_pattern_replace_all" => chain(|| {
            comps(PatternReplaceFilter::new(
                WhitespaceTokenizer::new(),
                JavaPattern::compile("[aeiou]")?,
                Some("_"),
                true,
            ))
        }),
        "ws_pattern_replace_first" => chain(|| {
            comps(PatternReplaceFilter::new(
                WhitespaceTokenizer::new(),
                JavaPattern::compile("([a-z])([a-z]*)")?,
                Some("$2$1"),
                false,
            ))
        }),
        "ws_pattern_capture_group" => chain(|| {
            let pats = [
                JavaPattern::compile("([A-Z][a-z]+)")?,
                JavaPattern::compile("([0-9]+)")?,
            ];
            comps(PatternCaptureGroupTokenFilter::new(
                WhitespaceTokenizer::new(),
                true,
                &pats,
            ))
        }),
        "std_pattern_typing" => chain(|| {
            let rules = vec![
                PatternTypingRule {
                    pattern: JavaPattern::compile("^(\\d+)\\.(\\d+)$")?,
                    flags: 3,
                    type_template: "decimal_$1".into(),
                },
                PatternTypingRule {
                    pattern: JavaPattern::compile("^([A-Z])")?,
                    flags: 4,
                    type_template: "capital_$1".into(),
                },
            ];
            comps(PatternTypingFilter::new(StandardTokenizer::new(), rules))
        }),
        "pattern_replace_char_filter" => chain_cf(
            |r| {
                Box::new(PatternReplaceCharFilter::new(
                    JavaPattern::compile("([a-z]+)-([a-z]+)").unwrap(),
                    "$2_$1",
                    r,
                ))
            },
            || comps(WhitespaceTokenizer::new()),
        ),

        "keyword_pattern_replace_dot" => replace_chain(KeywordTokenizer::new, ".", "_"),
        "keyword_pattern_replace_dollar" => {
            replace_chain(KeywordTokenizer::new, "(\\S)\\s*$", "[$1]")
        }
        "keyword_pattern_replace_x_star" => replace_chain(KeywordTokenizer::new, "x*", "-"),
        "pattern_replace_char_filter_x_star" => chain_cf(
            |r| {
                Box::new(PatternReplaceCharFilter::new(
                    JavaPattern::compile("x*").unwrap(),
                    "-",
                    r,
                ))
            },
            || comps(KeywordTokenizer::new()),
        ),
        "ws_pattern_replace_ascii_case" => replace_chain(
            WhitespaceTokenizer::new,
            "(?i)[a-e\u{e9}]|stra\u{df}e|k",
            "#",
        ),
        "ws_pattern_replace_unicode_case" => replace_chain(
            WhitespaceTokenizer::new,
            "(?iu)[a-e\u{e9}]|stra\u{df}e|k|\u{3c3}",
            "#",
        ),
        "ws_pattern_replace_posix" => replace_chain(
            WhitespaceTokenizer::new,
            "\\p{Punct}|\\p{Upper}|[[:alpha:]]",
            "_",
        ),
        "pattern_tokenizer_h_v" => chain(|| {
            comps(PatternTokenizer::new(
                &JavaPattern::compile("[\\h\\v,]+")?,
                -1,
            )?)
        }),
        "pattern_tokenizer_categories" => chain(|| {
            comps(PatternTokenizer::new(
                &JavaPattern::compile("(\\p{L}+)|(\\p{Nd}+)")?,
                0,
            )?)
        }),
        "std_pattern_typing_classes" => chain(|| {
            let rules = vec![
                PatternTypingRule {
                    pattern: JavaPattern::compile("^\\p{Lu}\\p{Ll}+$")?,
                    flags: 1,
                    type_template: "title".into(),
                },
                PatternTypingRule {
                    pattern: JavaPattern::compile("(?iu)^\\w*(.)$")?,
                    flags: 2,
                    type_template: "end_$1".into(),
                },
            ];
            comps(PatternTypingFilter::new(StandardTokenizer::new(), rules))
        }),

        // ---- path
        "path_hierarchy" => chain(|| comps(PathHierarchyTokenizer::default())),
        "path_hierarchy_backslash_skip1" => {
            chain(|| comps(PathHierarchyTokenizer::new('\\', '/', 1)?))
        }
        "reverse_path_hierarchy" => chain(|| comps(ReversePathHierarchyTokenizer::default())),
        "reverse_path_hierarchy_dot_skip1" => {
            chain(|| comps(ReversePathHierarchyTokenizer::new('.', '.', 1)?))
        }

        // ---- charfilter
        "mapping_char_filter" => {
            let mut b = NormalizeCharMapBuilder::new();
            for (k, v) in [
                ("ä", "ae"),
                ("ß", "ss"),
                ("fox", "wolf"),
                ("qu", "kw"),
                ("the", ""),
                ("&", " and "),
                ("é", "e"),
            ] {
                b.add(k, v).unwrap();
            }
            let map = Arc::new(b.build());
            chain_cf(
                move |r| Box::new(MappingCharFilter::new(map.clone(), r)),
                || comps(WhitespaceTokenizer::new()),
            )
        }

        // ---- commongrams
        "std_common_grams" => chain(|| {
            comps(CommonGramsFilter::new(
                LowerCaseFilter::new(StandardTokenizer::new()),
                Some(english()),
            ))
        }),
        "std_common_grams_query" => chain(|| {
            comps(CommonGramsQueryFilter::new(CommonGramsFilter::new(
                LowerCaseFilter::new(StandardTokenizer::new()),
                Some(english()),
            )))
        }),

        // ---- cjk
        "cjk_analyzer" => Analyzer::new(CJKAnalyzer::default()),
        "std_cjk_bigram_unigrams" => chain(|| {
            comps(CJKBigramFilter::new(
                StandardTokenizer::new(),
                cjk::HAN | cjk::HIRAGANA_FLAG | cjk::KATAKANA_FLAG | cjk::HANGUL_FLAG,
                true,
            ))
        }),
        "std_cjk_bigram_han_only" => chain(|| {
            comps(CJKBigramFilter::new(
                StandardTokenizer::new(),
                cjk::HAN,
                false,
            ))
        }),
        "ws_cjk_width" => chain(|| comps(CJKWidthFilter::new(WhitespaceTokenizer::new()))),
        "cjk_width_char_filter" => chain_cf(
            |r| Box::new(CJKWidthCharFilter::new(r)),
            || comps(WhitespaceTokenizer::new()),
        ),

        // ---- payloads / boost
        "ws_delimited_payload_float" => chain(|| {
            comps(DelimitedPayloadTokenFilter::new(
                WhitespaceTokenizer::new(),
                u16::from(b'|'),
                FloatEncoder,
            ))
        }),
        "ws_delimited_payload_int" => chain(|| {
            comps(DelimitedPayloadTokenFilter::new(
                WhitespaceTokenizer::new(),
                u16::from(b'|'),
                IntegerEncoder,
            ))
        }),
        "ws_delimited_payload_identity" => chain(|| {
            comps(DelimitedPayloadTokenFilter::new(
                WhitespaceTokenizer::new(),
                u16::from(b'|'),
                IdentityEncoder,
            ))
        }),
        "std_numeric_payload" => chain(|| {
            comps(NumericPayloadTokenFilter::new(
                StandardTokenizer::new(),
                3.5,
                "<NUM>",
            ))
        }),
        "std_type_as_payload" => {
            chain(|| comps(TypeAsPayloadTokenFilter::new(StandardTokenizer::new())))
        }
        "std_token_offset_payload" => {
            chain(|| comps(TokenOffsetPayloadTokenFilter::new(StandardTokenizer::new())))
        }
        "ws_delimited_boost" => chain(|| {
            comps(DelimitedBoostTokenFilter::new(
                WhitespaceTokenizer::new(),
                '|',
            ))
        }),

        // ---- minhash
        "ws_shingle_minhash" => chain(|| {
            let mut s = ShingleFilter::new(WhitespaceTokenizer::new(), 2, 2)?;
            s.set_output_unigrams(false);
            comps(MinHashFilter::new(s, 4, 2, 1, true)?)
        }),
        "ws_minhash_64_buckets" => chain(|| {
            comps(MinHashFilter::new(
                WhitespaceTokenizer::new(),
                1,
                64,
                1,
                true,
            )?)
        }),
        "ws_minhash_no_rotation" => chain(|| {
            comps(MinHashFilter::new(
                WhitespaceTokenizer::new(),
                2,
                8,
                2,
                false,
            )?)
        }),

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
        "html_strip_standard" => chain_cf(
            |r| Box::new(HTMLStripCharFilter::new(r)),
            || comps(StandardTokenizer::new()),
        ),
        "html_strip_keyword" => chain_cf(
            |r| Box::new(HTMLStripCharFilter::new(r)),
            || comps(KeywordTokenizer::new()),
        ),
        "html_strip_escaped_b" => chain_cf(
            |r| Box::new(HTMLStripCharFilter::with_escaped_tags(r, ["b"])),
            || comps(WhitespaceTokenizer::new()),
        ),
        "uax29_url_email_analyzer" => Analyzer::new(UAX29URLEmailAnalyzer::default()),
        "uax29_url_email_tokenizer" => chain(|| comps(UAX29URLEmailTokenizer::new())),
        "std_kstem" => chain(|| {
            comps(KStemFilter::new(LowerCaseFilter::new(
                StandardTokenizer::new(),
            )))
        }),
        "std_concatenate_graph" => {
            chain(|| comps(ConcatenateGraphFilter::new(StandardTokenizer::new())))
        }
        "ws_wdgf_concatenate_graph" => chain(|| {
            comps(ConcatenateGraphFilter::new(WordDelimiterGraphFilter::new(
                WhitespaceTokenizer::new(),
                m::GENERATE_WORD_PARTS | m::CATENATE_ALL,
                None,
            )?))
        }),
        "std_english_minimal" => chain(|| {
            comps(EnglishMinimalStemFilter::new(LowerCaseFilter::new(
                StandardTokenizer::new(),
            )))
        }),
        "ws_limit_token_count_analyzer" => {
            lucene_analysis::miscellaneous::LimitTokenCountAnalyzer::new(ws_analyzer(), 3, false)
                .unwrap()
                .into_analyzer()
        }
        "simple_limit_token_count_consume_all" => {
            lucene_analysis::miscellaneous::LimitTokenCountAnalyzer::new(
                Analyzer::new(core_analysis::SimpleAnalyzer),
                2,
                true,
            )
            .unwrap()
            .into_analyzer()
        }
        "per_field_wrapper_field" | "per_field_wrapper_default" => {
            let field = if name.ends_with("field") { "f" } else { "g" };
            let mut map = std::collections::HashMap::new();
            map.insert(
                field.to_string(),
                Analyzer::new(core_analysis::SimpleAnalyzer),
            );
            lucene_analysis::miscellaneous::PerFieldAnalyzerWrapper::new(ws_analyzer(), map)
                .into_analyzer()
        }
        "shingle_analyzer_wrapper" => lucene_analysis::shingle::ShingleAnalyzerWrapper::new(
            Analyzer::new(StandardAnalyzer::default()),
            2,
            3,
        )
        .unwrap()
        .into_analyzer(),
        "shingle_analyzer_wrapper_options" => {
            lucene_analysis::shingle::ShingleAnalyzerWrapper::with_options(
                ws_analyzer(),
                2,
                3,
                Some("_"),
                false,
                true,
                Some("*"),
            )
            .unwrap()
            .into_analyzer()
        }
        _ => return None,
    })
}

/// `new WhitespaceAnalyzer()`.
fn ws_analyzer() -> Analyzer {
    Analyzer::new(core_analysis::WhitespaceAnalyzer::default())
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

/// The chains with a `BoostAttribute`, whose rows end with the boost's bits.
const BOOST_CHAINS: &[&str] = &["ws_delimited_boost"];

fn token_row(ln: usize, a: &AttributeSource, boost: bool) -> String {
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
    ) + &if boost {
        format!("\t{:x}", a.boost().to_bits())
    } else {
        String::new()
    }
}

fn analyze_line(a: &Analyzer, ln: usize, line: &str, boost: bool, out: &mut Vec<String>) {
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
            out.push(token_row(ln, ts.attributes(), boost));
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
        let rejected = REJECTED.iter().any(|(n, _)| n == name);
        assert!(
            u8::from(built) + u8::from(pending) + u8::from(rejected) == 1,
            "{name}: built={built} pending={pending} rejected={rejected}"
        );
    }
    for (name, pattern) in REJECTED {
        assert!(
            names.contains(*name),
            "rejected chain {name} has no fixture"
        );
        match lucene_analysis::util::JavaPattern::compile(pattern) {
            Err(AnalysisError::IllegalArgument(m)) => assert!(m.contains("unsupported"), "{m}"),
            other => panic!("{name}: {pattern} was not rejected: {other:?}"),
        }
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
            analyze_line(
                &analyzer,
                ln,
                line,
                BOOST_CHAINS.contains(&name.as_str()),
                &mut actual,
            );
        }
        for (i, (e, a)) in expected.iter().zip(&actual).enumerate() {
            assert_eq!(a, e, "{name}: row {i} differs");
        }
        assert_eq!(actual.len(), expected.len(), "{name}: row count");
        checked += 1;
    }
    assert!(checked >= 110, "only {checked} chains checked");
}

#[test]
fn normalising_reads_only_lone_surrogate_escapes() {
    assert_eq!(normalise_expected("a\\uD83Db"), "a\u{FFFD}b");
    assert_eq!(normalise_expected("a\\u0001b"), "a\\u0001b");
    assert_eq!(normalise_expected("a\\\\uD83D"), "a\\\\uD83D");
    assert_eq!(normalise_expected("\\u"), "\\u");
}

/// `stems.words`: KStem and Porter over the generator's stem x suffix
/// words, one keyword token each.
#[test]
fn kstem_and_porter_match_lucene_word_for_word() {
    use lucene_analysis::en::{KStemFilter, PorterStemFilter};
    let text = std::fs::read_to_string(format!("{}stems.words", dir())).unwrap();
    let kstem = chain(|| comps(KStemFilter::new(KeywordTokenizer::new())));
    let porter = chain(|| comps(PorterStemFilter::new(KeywordTokenizer::new())));
    let single = |a: &Analyzer, w: &str| a.analyze(w).pop().map(|t| t.term).unwrap_or_default();
    let mut n = 0;
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        assert_eq!(single(&kstem, f[0]), f[1], "kstem({})", f[0]);
        assert_eq!(single(&porter, f[0]), f[2], "porter({})", f[0]);
        n += 1;
    }
    assert!(n > 9000, "{n} words");
}

/// `urls.words`: `UAX29URLEmailTokenizer` over 3,000 random joins of URL-
/// and email-shaped fragments.
#[test]
fn uax29_url_email_matches_lucene_on_fragments() {
    use lucene_analysis::email::UAX29URLEmailTokenizer;
    use lucene_analysis::reader::StrReader;
    use lucene_analysis::token_stream::{consume, Tokenizer};
    let text = std::fs::read_to_string(format!("{}urls.words", dir())).unwrap();
    let unesc = |s: &str| {
        s.replace("\\t", "\t")
            .replace("\\\"", "\"")
            .replace("\\\\", "\\")
    };
    let mut t = UAX29URLEmailTokenizer::new();
    let mut n = 0;
    for line in text.lines() {
        let (input, expected) = line.split_once('\t').unwrap();
        let input = unesc(input);
        t.set_reader(Box::new(StrReader::new(input.as_str())))
            .unwrap();
        let mut got = String::new();
        let end = consume(&mut t, |a| {
            got.push_str(&format!(
                "{} {} {} {} {}|",
                esc(a.term()),
                a.token_type(),
                a.start_offset(),
                a.end_offset(),
                a.position_increment()
            ))
        })
        .unwrap();
        got.push_str(&format!(
            "{} {}",
            end.end_offset(),
            end.position_increment()
        ));
        assert_eq!(got, expected, "{input:?}");
        n += 1;
    }
    assert_eq!(n, 3000);
}

/// The generator's `esc`, undone (`\\`, `\t`, `\n`, `\r`, `\uXXXX`; the
/// inputs hold no lone surrogate).
fn unesc(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('t') => out.push('\t'),
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('u') => {
                let h: String = it.by_ref().take(4).collect();
                out.push(char::from_u32(u32::from_str_radix(&h, 16).unwrap()).unwrap());
            }
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

/// What the port does with one `regex.words` row, in the generator's form.
fn regex_run(p: &lucene_analysis::util::JavaPattern, input: &str) -> String {
    use lucene_analysis::util::java_regex::JavaMatcher;
    let mut m = JavaMatcher::new(p, input);
    let mut b = String::new();
    while m.find() {
        b.push_str(&format!("({},{}", m.start(0), m.end(0)));
        for g in 1..=m.group_count() {
            b.push_str(&format!(" {}:{}", m.start(g), m.end(g)));
        }
        b.push(')');
    }
    let rep = p.replace(input, "<$0>", true).unwrap();
    format!("{b} rep={} m={}", esc(&rep), p.matches(input))
}

/// The patterns `regex.words` runs that Java compiles and the port rejects
/// on purpose (see `util/java_regex.rs`'s module docs for why each).
const REGEX_REJECTED: &[&str] = &[
    "\\bfox",
    "\\b",
    "\\B",
    "(?m)^a",
    "(?m)a$",
    "(?x) a b",
    "(?U)\\w",
    "(a|)*",
    "(a*)+",
    "\\p{IsLatin}",
    "\\p{InGreek}",
    "\\p{IsAlphabetic}",
    "\\p{javaLowerCase}",
    "(a)\\1",
    "(?=a)",
    "(?<=a)b",
    "a++",
    "(?>a)",
    "\\Z",
    "\\G",
    "\\R",
    "\\X",
    "\\cA",
    "\\0101",
    "\\N{LATIN SMALL LETTER A}",
    "(?d).",
    "[a~~b]",
    "[&&a]",
    "[a&&]",
    "[a&&&b]",
    "\\b?",
];

/// `regex.words`: every pattern of the generator over every input --
/// `find()` spans and groups, `replaceAll("<$0>")`, `matches()` -- or the
/// exception Java throws compiling it.
#[test]
fn java_regex_matches_lucene_pattern_for_pattern() {
    use lucene_analysis::util::JavaPattern;
    let text = std::fs::read_to_string(format!("{}regex.words", dir())).unwrap();
    let (mut compared, mut refused) = (0, BTreeSet::new());
    for line in text.lines() {
        let f: Vec<&str> = line.splitn(3, '\t').collect();
        let (pattern, input, want) = (unesc(f[0]), unesc(f[1]), normalise_expected(f[2]));
        match JavaPattern::compile(&pattern) {
            Ok(p) => {
                assert!(!want.starts_with("EXC"), "{pattern:?}: Java threw {want}");
                assert!(
                    !REGEX_REJECTED.contains(&pattern.as_str()),
                    "{pattern:?} compiled"
                );
                assert_eq!(regex_run(&p, &input), want, "{pattern:?} on {input:?}");
                compared += 1;
            }
            Err(AnalysisError::IllegalArgument(_)) if want.starts_with("EXC") => {}
            Err(e) => {
                assert!(
                    REGEX_REJECTED.contains(&pattern.as_str()),
                    "{pattern:?} rejected but Java gives {want}: {e:?}"
                );
                refused.insert(pattern);
            }
        }
    }
    assert!(compared > 3000, "{compared} rows compared");
    assert_eq!(refused.len(), REGEX_REJECTED.len(), "{refused:?}");
}

/// `regex_ci.words`: which code points each case-insensitive form matches,
/// over every code point with a simple case mapping.
#[test]
fn java_regex_case_folding_matches_lucene() {
    use lucene_analysis::util::java_regex::JavaMatcher;
    use lucene_analysis::util::JavaPattern;
    let text = std::fs::read_to_string(format!("{}regex_ci.words", dir())).unwrap();
    let mut lines = text.lines();
    let inputs = [unesc(lines.next().unwrap()), unesc(lines.next().unwrap())];
    let mut n = 0;
    for line in lines {
        let f: Vec<&str> = line.split('\t').collect();
        let pattern = unesc(f[0]);
        let input = &inputs[usize::from(f[1] == "2")];
        let p = JavaPattern::compile(&pattern).unwrap();
        let mut m = JavaMatcher::new(&p, input);
        let mut got = Vec::new();
        while m.find() {
            got.push(m.start(0).to_string());
        }
        assert_eq!(got.join(","), f[2], "{pattern:?}");
        n += 1;
    }
    assert!(n > 4000, "{n} forms");
}

/// `concatenate.words`: `ConcatenateGraphFilter`'s indexed bytes with
/// separators above ASCII (Java casts the separator to a byte).
#[test]
fn concatenate_graph_separators_match_lucene_byte_for_byte() {
    use lucene_analysis::miscellaneous::ConcatenateGraphFilter;
    use lucene_analysis::reader::StrReader;
    use lucene_analysis::token_stream::{consume, Tokenizer};
    let text = std::fs::read_to_string(format!("{}concatenate.words", dir())).unwrap();
    let mut n = 0;
    for line in text.lines() {
        let f: Vec<&str> = line.split('\t').collect();
        let sep = match f[0] {
            "-" => None,
            h => Some(char::from_u32(u32::from_str_radix(h, 16).unwrap()).unwrap()),
        };
        let mut t = WhitespaceTokenizer::new();
        t.set_reader(Box::new(StrReader::new(unesc(f[1])))).unwrap();
        let mut c = ConcatenateGraphFilter::with_options(t, sep, true, 10_000);
        let mut got = String::new();
        consume(&mut c, |a| {
            got.push_str(&hex(Some(a.term_bytes())));
            got.push('|');
        })
        .unwrap();
        assert_eq!(got, f[2], "{line}");
        n += 1;
    }
    assert_eq!(n, 45);
}
