#![cfg(test)]
use super::*;

#[test]
fn lowercase_is_jdk_25s_not_the_toolchains() {
    // Unicode 17 letters (U+A7CE, U+16EA0): unassigned for JDK 25.
    for c in ['\u{A7CE}', '\u{A7D2}', '\u{16EA0}'] {
        assert_eq!(simple_to_lowercase(c), c, "{c:?}");
    }
    // Unicode 16 pairs JDK 21 lacks.
    assert_eq!(simple_to_lowercase('\u{A7CB}'), '\u{264}');
    assert_eq!(simple_to_lowercase('\u{A7DC}'), '\u{19B}');
    assert_eq!(simple_to_lowercase('\u{130}'), 'i');
    assert_eq!(simple_to_lowercase('Σ'), 'σ');
    assert_eq!(simple_to_lowercase('Q'), 'q');
}

#[test]
fn caseless_blocks_have_no_lowercase() {
    for cp in (0x0E00..=0x0EFF)
        .chain(0x3000..=0x9FFF)
        .chain(0xAC00..=0xD7AF)
        .chain(0x1F000..=0x1FBFF)
        .chain(0x20000..=0x3FFFF)
    {
        if let Some(c) = char::from_u32(cp) {
            assert!(is_caseless_block(c));
            assert_eq!(java_character::to_lower_case(cp), cp, "{cp:#x}");
        }
    }
    assert!(!is_caseless_block('A'));
    let mut t = "日本Ωx".to_string();
    lowercase_term(&mut t);
    assert_eq!(t, "日本ωx");
    let mut t = "日本".to_string();
    lowercase_term(&mut t);
    assert_eq!(t, "日本");
    // In place while widths agree; rebuilt from a width change (İ is two
    // bytes, i one; Ⱥ two, ⱥ three).
    let mut t = "ÄA\u{10400}İBȺÉ".to_string();
    lowercase_term(&mut t);
    assert_eq!(t, "äa\u{10428}ibⱥé");
}

fn tok(term: &str, start: i32, end: i32, pos_inc: i32) -> Token {
    Token {
        term: term.to_string(),
        start_offset: start,
        end_offset: end,
        position_increment: pos_inc,
        position_length: 1,
    }
}

fn tok_len(term: &str, start: i32, end: i32, pos_inc: i32, pos_len: i32) -> Token {
    Token {
        term: term.to_string(),
        start_offset: start,
        end_offset: end,
        position_increment: pos_inc,
        position_length: pos_len,
    }
}

#[test]
fn tokenize_multi_word_sentence() {
    let tokens = tokenize("The quick, brown fox!");
    assert_eq!(
        tokens,
        vec![
            tok("The", 0, 3, 1),
            tok("quick", 4, 9, 1),
            tok("brown", 11, 16, 1),
            tok("fox", 17, 20, 1),
        ]
    );
}

#[test]
fn tokenize_empty_text() {
    assert_eq!(tokenize(""), vec![]);
}

#[test]
fn tokenize_only_punctuation() {
    assert_eq!(tokenize("... !!! ,,,"), vec![]);
}

#[test]
fn tokenize_alphanumeric_run_kept_together() {
    let tokens = tokenize("abc123 456def");
    assert_eq!(
        tokens,
        vec![tok("abc123", 0, 6, 1), tok("456def", 7, 13, 1),]
    );
}

// -- Embedded numeric punctuation ("3.14", "1,000") --

#[test]
fn tokenize_number_with_embedded_period_stays_one_token() {
    // OLD (wrong) behavior: this split into "3" and "14".
    let tokens = tokenize("pi is 3.14 today");
    assert_eq!(
        tokens,
        vec![
            tok("pi", 0, 2, 1),
            tok("is", 3, 5, 1),
            tok("3.14", 6, 10, 1),
            tok("today", 11, 16, 1),
        ]
    );
}

#[test]
fn tokenize_number_with_embedded_comma_stays_one_token() {
    // OLD (wrong) behavior: this split into "1" and "000".
    let tokens = tokenize("1,000 dollars");
    assert_eq!(
        tokens,
        vec![tok("1,000", 0, 5, 1), tok("dollars", 6, 13, 1),]
    );
}

#[test]
fn tokenize_sentence_ending_period_after_number_still_splits() {
    // Adjacent case that must NOT be affected: a real sentence-ending
    // period (nothing alphanumeric follows it) still splits off.
    let tokens = tokenize("The total is 42. Done.");
    assert_eq!(
        tokens,
        vec![
            tok("The", 0, 3, 1),
            tok("total", 4, 9, 1),
            tok("is", 10, 12, 1),
            tok("42", 13, 15, 1),
            tok("Done", 17, 21, 1),
        ]
    );
}

// -- Acronym-style internal periods ("U.S.A.") --

#[test]
fn tokenize_acronym_kept_together() {
    // OLD (wrong) behavior: this split into "U", "S", "A".
    let tokens = tokenize("U.S.A. is here");
    assert_eq!(
        tokens,
        vec![
            tok("U.S.A", 0, 5, 1),
            tok("is", 7, 9, 1),
            tok("here", 10, 14, 1),
        ]
    );
}

#[test]
fn tokenize_trailing_sentence_period_after_word_still_splits() {
    // Adjacent case that must NOT be affected: a normal word followed by
    // a sentence-ending period still splits the period off.
    let tokens = tokenize("This is the end. Next sentence.");
    assert_eq!(
        tokens,
        vec![
            tok("This", 0, 4, 1),
            tok("is", 5, 7, 1),
            tok("the", 8, 11, 1),
            tok("end", 12, 15, 1),
            tok("Next", 17, 21, 1),
            tok("sentence", 22, 30, 1),
        ]
    );
}

// -- Internal apostrophes ("don't", "O'Brien") --

#[test]
fn tokenize_apostrophe_contraction_kept_together() {
    // OLD (wrong) behavior: this split into "don" and "t".
    let tokens = tokenize("don't stop");
    assert_eq!(tokens, vec![tok("don't", 0, 5, 1), tok("stop", 6, 10, 1),]);
}

#[test]
fn tokenize_apostrophe_name_kept_together() {
    // OLD (wrong) behavior: this split into "O" and "Brien".
    let tokens = tokenize("O'Brien arrived");
    assert_eq!(
        tokens,
        vec![tok("O'Brien", 0, 7, 1), tok("arrived", 8, 15, 1),]
    );
}

#[test]
fn tokenize_leading_apostrophe_not_absorbed() {
    // Adjacent case that must NOT be affected: an apostrophe with no
    // alphanumeric character before it (e.g. an opening quote) is a
    // plain separator, not part of the following word.
    let tokens = tokenize("'tis the season");
    assert_eq!(
        tokens,
        vec![
            tok("tis", 1, 4, 1),
            tok("the", 5, 8, 1),
            tok("season", 9, 15, 1),
        ]
    );
}

// -- UAX#29 extensions: combining marks, CJK, Hangul, ZWJ --

#[test]
fn tokenize_combining_mark_stays_attached_to_base_char() {
    // "e" + combining acute accent (U+0301), decomposed form of "é".
    // A naive per-char split would treat the combining mark as its own
    // boundary; UAX#29 (via WB's Extend rule) keeps it fused to "cafe".
    let tokens = tokenize("cafe\u{0301} today");
    assert_eq!(tokens.len(), 2);
    assert_eq!(tokens[0].term, "cafe\u{0301}");
    assert_eq!(tokens[0].start_offset, 0);
    // Five Java `char`s ("cafe" + the combining mark), six UTF-8 bytes.
    assert_eq!(tokens[0].end_offset, 5);
    assert_eq!(tokens[1].term, "today");
    assert_eq!((tokens[1].start_offset, tokens[1].end_offset), (6, 11));
}

#[test]
fn tokenize_cjk_ideographs_split_one_per_character() {
    // Each Han ideograph is its own token -- no word clustering across
    // CJK text, unlike Latin script.
    // Each ideograph is one UTF-16 code unit and three UTF-8 bytes.
    let tokens = tokenize("你好世界");
    assert_eq!(
        tokens,
        vec![
            tok("你", 0, 1, 1),
            tok("好", 1, 2, 1),
            tok("世", 2, 3, 1),
            tok("界", 3, 4, 1),
        ]
    );
}

#[test]
fn tokenize_precomposed_hangul_syllable_is_one_token() {
    let tokens = tokenize("안녕하세요");
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].term, "안녕하세요");
}

#[test]
fn tokenize_conjoining_hangul_jamo_cluster_into_one_syllable_token() {
    // Leading consonant + vowel + trailing consonant jamo (U+1100,
    // U+1161, U+11A8) compose the syllable "각"; UAX#29's Hangul WB
    // rules cluster them into one token, not three.
    let jamo = "\u{1100}\u{1161}\u{11A8}";
    let tokens = tokenize(jamo);
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].term, jamo);
}

#[test]
fn tokenize_mixed_cjk_and_latin() {
    let tokens = tokenize("hello 世界 world");
    assert_eq!(
        tokens,
        vec![
            tok("hello", 0, 5, 1),
            tok("世", 6, 7, 1),
            tok("界", 7, 8, 1),
            tok("world", 9, 14, 1),
        ]
    );
}

#[test]
fn tokenize_zwj_between_letters_does_not_split() {
    // A bare ZWJ (U+200D) between two letters is Extend-like and does
    // not introduce a word boundary.
    let joined = "a\u{200d}b";
    let tokens = tokenize(joined);
    assert_eq!(tokens.len(), 1);
    assert_eq!(tokens[0].term, joined);
}

#[test]
fn tokenize_emoji_is_its_own_token() {
    // Lucene's grammar emits an emoji as an <EMOJI> token (the old
    // unicode-segmentation stand-in dropped it). U+1F44D is one scalar,
    // **two** UTF-16 code units and four UTF-8 bytes, so "emoji" starts
    // at Java `char` 6, not scalar 5 or byte 8.
    let tokens = tokenize("test\u{1F44D}emoji");
    assert_eq!(
        tokens,
        vec![
            tok("test", 0, 4, 1),
            tok("\u{1F44D}", 4, 6, 1),
            tok("emoji", 6, 11, 1)
        ]
    );
}

#[test]
fn lowercase_filter_changes_case_not_offsets_or_positions() {
    let tokens = vec![tok("THE", 0, 3, 1), tok("Quick", 4, 9, 2)];
    let out = LowerCaseFilter::apply(tokens);
    assert_eq!(out, vec![tok("the", 0, 3, 1), tok("quick", 4, 9, 2),]);
}

// ---- the TokenStream lifecycle (`end()`) --------------------------

/// `Tokenizer.end()`'s `finalOffset = correctOffset(charCount)` is the
/// length of the **whole input**, not the end of the last token -- which is
/// what a multi-valued field's next value is offset by.
#[test]
fn tokenize_stream_ends_at_the_inputs_own_length_not_the_last_token() {
    let stream = tokenize_stream("fox   ");
    assert_eq!(stream.tokens.len(), 1);
    assert_eq!(stream.tokens[0].end_offset, 3);
    assert_eq!(stream.final_offset, 6);
    assert_eq!(stream.final_position_increment, 0);
    // UTF-16 code units, like every other offset in this crate.
    assert_eq!(tokenize_stream("\u{1D400} x").final_offset, 4);
    // Empty input: no tokens, and an end offset of 0.
    let empty = tokenize_stream("");
    assert!(empty.tokens.is_empty());
    assert_eq!(empty.final_offset, 0);
}

/// `FilteringTokenFilter.end()`: `skippedPositions` left over at end of
/// stream is published on the position-increment attribute, so trailing
/// stopwords still advance the field's position counter.
#[test]
fn stop_filter_end_publishes_the_trailing_skipped_positions() {
    let stopwords: HashSet<String> = ["the"].into_iter().map(String::from).collect();
    let stream = Analyzer::standard(Some(&stopwords)).analyze_stream("fox the the");
    assert_eq!(
        stream
            .tokens
            .iter()
            .map(|t| t.term.as_str())
            .collect::<Vec<_>>(),
        vec!["fox"]
    );
    assert_eq!(stream.final_position_increment, 2);
    assert_eq!(stream.final_offset, 11);

    // Nothing trailing to skip: the base `TokenStream.end()`'s 0.
    let stream = Analyzer::standard(Some(&stopwords)).analyze_stream("the fox");
    assert_eq!(stream.final_position_increment, 0);

    // Every token filtered out: all of them land on the final increment,
    // because no surviving token could carry them.
    let stream = Analyzer::standard(Some(&stopwords)).analyze_stream("the the the");
    assert!(stream.tokens.is_empty());
    assert_eq!(stream.final_position_increment, 3);
}

/// `StopFilter::apply` is `apply_to_stream` with the end-of-stream values
/// dropped, which is what every caller that only wants the tokens gets.
#[test]
fn stop_filter_apply_and_apply_to_stream_agree_on_the_tokens() {
    let stopwords: HashSet<String> = ["the"].into_iter().map(String::from).collect();
    let tokens = tokenize("fox the the");
    let via_apply = StopFilter::apply(tokens.clone(), &stopwords);
    let via_stream = StopFilter::apply_to_stream(
        AnalyzedTokens {
            tokens,
            final_position_increment: 0,
            final_offset: 11,
        },
        &stopwords,
    );
    assert_eq!(via_apply, via_stream.tokens);
    assert_eq!(via_stream.final_offset, 11);
}

/// `NGramTokenFilter.end()` / `EdgeNGramTokenFilter.end()`:
/// `posIncrAtt.setPositionIncrement(curPosIncr)` -- and note it **sets**
/// rather than adds, so an upstream filter's own end-of-stream increment is
/// discarded. Both halves are asserted, because the overwrite is the
/// surprising one.
#[test]
fn ngram_end_publishes_the_leftover_increment_and_overwrites_the_upstream_one() {
    // "ab" is shorter than min_gram 3, so it emits nothing and its
    // increment is still owed at end of stream.
    let stream = AnalyzedTokens {
        tokens: tokenize("abcd ab"),
        final_position_increment: 7,
        final_offset: 7,
    };
    let out = NGramTokenFilter::apply_to_stream(stream, 3, 3, false).unwrap();
    assert_eq!(
        out.tokens
            .iter()
            .map(|t| t.term.as_str())
            .collect::<Vec<_>>(),
        vec!["abc", "bcd"]
    );
    assert_eq!(out.final_position_increment, 1, "the skipped \"ab\"");
    assert_eq!(out.final_offset, 7);

    let stream = AnalyzedTokens {
        tokens: tokenize("abcd ab"),
        final_position_increment: 7,
        final_offset: 7,
    };
    let out = EdgeNGramTokenFilter::apply_to_stream(stream, 3, 3, false).unwrap();
    assert_eq!(
        out.tokens
            .iter()
            .map(|t| t.term.as_str())
            .collect::<Vec<_>>(),
        vec!["abc"]
    );
    assert_eq!(out.final_position_increment, 1);

    // preserveOriginal emits the short token, so nothing is owed.
    let stream = AnalyzedTokens {
        tokens: tokenize("abcd ab"),
        final_position_increment: 0,
        final_offset: 7,
    };
    let out = NGramTokenFilter::apply_to_stream(stream, 3, 3, true).unwrap();
    assert_eq!(out.final_position_increment, 0);
    assert!(out.tokens.iter().any(|t| t.term == "ab"));
}

/// A keyword analyzer's stream: one token, no swallowed positions, and the
/// `KeywordTokenizer.end()` final offset.
#[test]
fn keyword_analyzer_stream_ends_at_the_inputs_length() {
    let stream = Analyzer::keyword().analyze_stream("id-\u{1F600}");
    assert_eq!(stream.tokens.len(), 1);
    assert_eq!(stream.final_position_increment, 0);
    assert_eq!(stream.final_offset, 5);
}

/// The two gap accessors are Java's `Analyzer.getPositionIncrementGap` /
/// `getOffsetGap`, defaulting to Java's own `0` and `1`.
#[test]
fn the_analyzer_gaps_default_to_javas_zero_and_one() {
    let a = Analyzer::standard(None);
    assert_eq!(a.position_increment_gap(), 0);
    assert_eq!(a.offset_gap(), 1);
    let a = a.with_position_increment_gap(100).with_offset_gap(3);
    assert_eq!(a.position_increment_gap(), 100);
    assert_eq!(a.offset_gap(), 3);
    assert_eq!(Analyzer::keyword().offset_gap(), 1);
}

#[test]
fn stop_filter_bumps_next_position_increment() {
    // "the quick fox" with "the" as a stopword: "quick" should get
    // position_increment == 2 (1 from itself + 1 carried over from the
    // removed "the"), not 1.
    let tokens = tokenize("the quick fox");
    let tokens = LowerCaseFilter::apply(tokens);
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let out = StopFilter::apply(tokens, &stopwords);
    assert_eq!(out, vec![tok("quick", 4, 9, 2), tok("fox", 10, 13, 1),]);
}

#[test]
fn stop_filter_stopword_at_start() {
    let tokens = tokenize("the fox");
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let out = StopFilter::apply(tokens, &stopwords);
    assert_eq!(out, vec![tok("fox", 4, 7, 2)]);
}

#[test]
fn stop_filter_stopword_at_end() {
    let tokens = tokenize("fox the");
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let out = StopFilter::apply(tokens, &stopwords);
    assert_eq!(out, vec![tok("fox", 0, 3, 1)]);
}

#[test]
fn stop_filter_consecutive_stopwords_accumulate() {
    // "a the of fox" with "a"/"the"/"of" all stopwords: fox should carry
    // increment 1 (its own) + 3 removed = 4.
    let tokens = tokenize("a the of fox");
    let stopwords: HashSet<String> = ["a".to_string(), "the".to_string(), "of".to_string()]
        .into_iter()
        .collect();
    let out = StopFilter::apply(tokens, &stopwords);
    assert_eq!(out, vec![tok("fox", 9, 12, 4)]);
}

#[test]
fn stop_filter_all_stopwords_yields_empty_not_panic() {
    let tokens = tokenize("the a of");
    let stopwords: HashSet<String> = ["the".to_string(), "a".to_string(), "of".to_string()]
        .into_iter()
        .collect();
    let out = StopFilter::apply(tokens, &stopwords);
    assert_eq!(out, vec![]);
}

#[test]
fn english_stop_words_matches_real_lucene_canonical_list() {
    // Transcribed from real Lucene's
    // `org.apache.lucene.analysis.en.EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`
    // (the classic Lucene/Snowball English stop list). Review-confirmed
    // caveat, stated honestly: this literal is the same 33 words as
    // `ENGLISH_STOP_WORDS` itself, so this test guards against a future
    // edit letting the two lists drift apart -- it does not, on its own,
    // prove `ENGLISH_STOP_WORDS` was transcribed correctly from Lucene in
    // the first place (a one-time transcription error made once and
    // repeated in both places would still pass). That correctness claim
    // rests on careful transcription against the real Lucene source, not
    // on this test's structure.
    const CANONICAL_33: &[&str] = &[
        "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is",
        "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there",
        "these", "they", "this", "to", "was", "will", "with",
    ];
    assert_eq!(
        CANONICAL_33.len(),
        33,
        "the reference list itself must have exactly 33 entries"
    );
    let stopwords = english_stop_words();
    assert_eq!(
        stopwords.len(),
        33,
        "ENGLISH_STOP_WORDS must have exactly 33 entries, matching real Lucene"
    );
    for word in CANONICAL_33 {
        assert!(
            stopwords.contains(*word),
            "canonical Lucene English stopword {word:?} is missing from ENGLISH_STOP_WORDS"
        );
    }
    // No extras: every entry in this port's set must also appear in the
    // canonical list (catches an accidentally-added wrong/extra word).
    for word in &stopwords {
        assert!(
            CANONICAL_33.contains(&word.as_str()),
            "ENGLISH_STOP_WORDS contains {word:?}, which is not one of real Lucene's 33 \
                 canonical English stopwords"
        );
    }
}

#[test]
fn english_stop_words_case_is_already_lowercase() {
    // Real Lucene's set is populated with already-lowercase entries, and
    // StopFilter matches against already-lowercased terms (it runs after
    // LowerCaseFilter in the chain) -- so every entry here must be
    // lowercase, not merely "matched case-insensitively".
    for word in &ENGLISH_STOP_WORDS {
        assert_eq!(
            *word,
            word.to_lowercase(),
            "{word:?} must be stored lowercase"
        );
    }
}

#[test]
fn english_stop_words_does_not_false_positive_on_content_words() {
    // Representative non-stopwords that must survive StopFilter
    // untouched -- proves the set isn't overly broad (e.g. accidentally
    // matching real content words via substring/prefix matching instead
    // of exact string equality).
    let stopwords = english_stop_words();
    for word in ["search", "lucene", "rust", "document", "index", "query"] {
        assert!(!stopwords.contains(word), "{word:?} must NOT be a stopword");
    }
    let tokens = tokenize("search the lucene rust document index and query");
    let tokens = LowerCaseFilter::apply(tokens);
    let out = StopFilter::apply(tokens, &stopwords);
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    // "the" and "and" are real stopwords and must be removed; every
    // other word here is a real content word and must survive.
    assert_eq!(
        terms,
        vec!["search", "lucene", "rust", "document", "index", "query"]
    );
}

#[test]
fn english_stop_words_used_via_analyzer_standard() {
    // End-to-end: Analyzer::standard wired with the real default English
    // stop set behaves like real Lucene's EnglishAnalyzer/StandardAnalyzer
    // defaults for a sentence containing several of the 33 stopwords.
    let stopwords = english_stop_words();
    let analyzer = Analyzer::standard(Some(&stopwords));
    let out = analyzer.analyze("The quick fox will jump into the river");
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["quick", "fox", "jump", "river"]);
}

#[test]
fn french_stop_words_has_exactly_154_entries_matching_real_lucene() {
    // Real Lucene's FrenchAnalyzer.getDefaultStopSet() loads
    // french_stop.txt (Snowball's French stop list) via
    // WordlistLoader.getSnowballWordSet, which strips each line's `|`
    // comment and splits the remainder on whitespace. Transcribing that
    // file (org/apache/lucene/analysis/snowball/french_stop.txt in a
    // real Lucene 10.5.0 checkout) the same way yields exactly 154
    // surviving words -- several classic French stopwords (e.g. "son",
    // "sommes", "été", "est", "avions", "auras", "aura", "as", "fût")
    // are deliberately commented out of that file as homonyms of
    // unrelated common words, so this is not simply "every French
    // function word".
    let stopwords = french_stop_words();
    assert_eq!(
        stopwords.len(),
        154,
        "FRENCH_STOP_WORDS must have exactly 154 entries, matching real \
             Lucene's french_stop.txt after comment-stripping"
    );
    assert_eq!(FRENCH_STOP_WORDS.len(), 154);
    // Spot-check a representative sample spanning every distinct source
    // section of french_stop.txt: core articles/pronouns, single-letter
    // elision remnants, and inflected forms of être/avoir.
    for word in [
        "le", "la", "les", "un", "une", "et", "de", "du", "des", "dans", "sur", "pour", "qu",
        "que", "qui", "c", "d", "j", "l", "à", "m", "n", "s", "t", "y", "suis", "es", "sont",
        "serai", "était", "avons", "avez", "ont", "aurai", "eusse", "ceci", "cela", "quel", "sans",
        "soi",
    ] {
        assert!(
            stopwords.contains(word),
            "canonical French stopword {word:?} is missing from FRENCH_STOP_WORDS"
        );
    }
    // Deliberately-omitted homonyms (real Lucene's own comments explain
    // why): must NOT be present.
    for word in [
        "son", "sommes", "été", "étés", "est", "avions", "auras", "aura", "as", "fût",
    ] {
        assert!(
            !stopwords.contains(word),
            "{word:?} is a homonym real Lucene deliberately omits from its default \
                 French stop set -- it must not be in FRENCH_STOP_WORDS either"
        );
    }
}

#[test]
fn french_stop_words_case_is_already_lowercase() {
    for word in &FRENCH_STOP_WORDS {
        assert_eq!(
            *word,
            word.to_lowercase(),
            "{word:?} must be stored lowercase"
        );
    }
}

#[test]
fn french_stop_words_does_not_false_positive_on_content_words() {
    let stopwords = french_stop_words();
    for word in ["chat", "souris", "maison", "recherche", "document"] {
        assert!(!stopwords.contains(word), "{word:?} must NOT be a stopword");
    }
    let tokens = tokenize("Le chat et la souris sont dans la maison");
    let tokens = LowerCaseFilter::apply(tokens);
    let out = StopFilter::apply(tokens, &stopwords);
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["chat", "souris", "maison"]);
}

#[test]
fn analyzer_standard_full_pipeline() {
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let analyzer = Analyzer::standard(Some(&stopwords));
    let out = analyzer.analyze("The Quick, Brown FOX!");
    assert_eq!(
        out,
        vec![
            tok("quick", 4, 9, 2),
            tok("brown", 11, 16, 1),
            tok("fox", 17, 20, 1),
        ]
    );
}

#[test]
fn analyzer_standard_no_stopwords() {
    let analyzer = Analyzer::standard(None);
    let out = analyzer.analyze("Hello World");
    assert_eq!(out, vec![tok("hello", 0, 5, 1), tok("world", 6, 11, 1)]);
}

#[test]
fn ascii_folding_latin1_spot_checks() {
    let tokens = vec![
        tok("café", 0, 4, 1),
        tok("naïve", 0, 5, 1),
        tok("Müller", 0, 6, 1),
        tok("ñ", 0, 1, 1),
    ];
    let out = AsciiFoldingFilter::apply(tokens);
    assert_eq!(
        out,
        vec![
            tok("cafe", 0, 4, 1),
            tok("naive", 0, 5, 1),
            tok("Muller", 0, 6, 1),
            tok("n", 0, 1, 1),
        ]
    );
}

#[test]
fn ascii_folding_covers_every_documented_table_entry() {
    // Exhaustively spot-checks every char->replacement mapping this
    // filter documents, not just a handful -- so every match arm in
    // `fold_char` is actually exercised.
    let cases: &[(char, &str)] = &[
        ('À', "A"),
        ('Á', "A"),
        ('Â', "A"),
        ('Ã', "A"),
        ('Ä', "A"),
        ('Å', "A"),
        ('Æ', "AE"),
        ('Ç', "C"),
        ('È', "E"),
        ('É', "E"),
        ('Ê', "E"),
        ('Ë', "E"),
        ('Ì', "I"),
        ('Í', "I"),
        ('Î', "I"),
        ('Ï', "I"),
        ('Ð', "D"),
        ('Ñ', "N"),
        ('Ò', "O"),
        ('Ó', "O"),
        ('Ô', "O"),
        ('Õ', "O"),
        ('Ö', "O"),
        ('Ø', "O"),
        ('Ù', "U"),
        ('Ú', "U"),
        ('Û', "U"),
        ('Ü', "U"),
        ('Ý', "Y"),
        ('Þ', "TH"),
        ('ß', "ss"),
        ('à', "a"),
        ('á', "a"),
        ('â', "a"),
        ('ã', "a"),
        ('ä', "a"),
        ('å', "a"),
        ('æ', "ae"),
        ('ç', "c"),
        ('è', "e"),
        ('é', "e"),
        ('ê', "e"),
        ('ë', "e"),
        ('ì', "i"),
        ('í', "i"),
        ('î', "i"),
        ('ï', "i"),
        ('ð', "d"),
        ('ñ', "n"),
        ('ò', "o"),
        ('ó', "o"),
        ('ô', "o"),
        ('õ', "o"),
        ('ö', "o"),
        ('ø', "o"),
        ('ù', "u"),
        ('ú', "u"),
        ('û', "u"),
        ('ü', "u"),
        ('ý', "y"),
        ('ÿ', "y"),
        ('þ', "th"),
        ('Ą', "A"),
        ('ą', "a"),
        ('Ć', "C"),
        ('ć', "c"),
        ('Č', "C"),
        ('č', "c"),
        ('Ď', "D"),
        ('ď', "d"),
        ('Ę', "E"),
        ('ę', "e"),
        ('Ł', "L"),
        ('ł', "l"),
        ('Ń', "N"),
        ('ń', "n"),
        ('Ň', "N"),
        ('ň', "n"),
        ('Œ', "OE"),
        ('œ', "oe"),
        ('Ś', "S"),
        ('ś', "s"),
        ('Š', "S"),
        ('š', "s"),
        ('Ť', "T"),
        ('ť', "t"),
        ('Ź', "Z"),
        ('ź', "z"),
        ('Ž', "Z"),
        ('ž', "z"),
        ('Ż', "Z"),
        ('ż', "z"),
    ];
    for (c, expected) in cases {
        let tokens = vec![tok(&c.to_string(), 0, 1, 1)];
        let out = AsciiFoldingFilter::apply(tokens);
        assert_eq!(
            out,
            vec![tok(expected, 0, 1, 1)],
            "folding {c:?} should yield {expected:?}"
        );
    }
}

#[test]
fn ascii_folding_eszett_folds_to_ss() {
    let tokens = vec![tok("straße", 0, 6, 1)];
    let out = AsciiFoldingFilter::apply(tokens);
    assert_eq!(out, vec![tok("strasse", 0, 6, 1)]);
}

#[test]
fn ascii_folding_ligature_grows_term_but_not_offsets() {
    // "æ" (1 char) -> "ae" (2 chars): term grows, offsets untouched.
    let tokens = vec![tok("æther", 0, 5, 1), tok("cœur", 10, 14, 1)];
    let out = AsciiFoldingFilter::apply(tokens);
    assert_eq!(out, vec![tok("aether", 0, 5, 1), tok("coeur", 10, 14, 1),]);
    assert!(out[0].term.chars().count() > 5);
}

#[test]
fn ascii_folding_plain_ascii_passes_through_unmodified() {
    let tokens = vec![tok("hello", 0, 5, 1)];
    let out = AsciiFoldingFilter::apply(tokens.clone());
    assert_eq!(out, tokens);
}

#[test]
fn ascii_folding_mixed_diacritic_and_ascii_in_one_token() {
    let tokens = vec![tok("café123", 0, 7, 1)];
    let out = AsciiFoldingFilter::apply(tokens);
    assert_eq!(out, vec![tok("cafe123", 0, 7, 1)]);
}

#[test]
fn ascii_folding_char_outside_table_passes_through_unchanged() {
    // A Cyrillic character isn't in this filter's documented table --
    // it must survive untouched, not be dropped or panic.
    let tokens = vec![tok("привет", 0, 6, 1)];
    let out = AsciiFoldingFilter::apply(tokens.clone());
    assert_eq!(out, tokens);
}

#[test]
fn ascii_folding_composed_with_lowercase_folds_then_lowercases() {
    // Analyzer::with_ascii_folding applies folding before lowercasing:
    // "É" -> "E" -> "e".
    let analyzer = Analyzer::standard(None).with_ascii_folding();
    // Offsets are the *source* text's Java `char` spans: folding grows
    // no term here, but each accented letter is 2 UTF-8 bytes and 1
    // UTF-16 code unit, so these are 0,4 / 5,10 / 11,16 -- what real
    // Lucene's `fold_then_lower` fixture case records.
    let out = analyzer.analyze("Café Naïve ÉCOLE");
    assert_eq!(
        out,
        vec![
            tok("cafe", 0, 4, 1),
            tok("naive", 5, 10, 1),
            tok("ecole", 11, 16, 1),
        ]
    );
}

#[test]
fn porter_step1a_plural_forms() {
    let tokens = vec![
        tok("caresses", 0, 8, 1),
        tok("ponies", 0, 6, 1),
        tok("cats", 0, 4, 1),
        tok("caress", 0, 6, 1),
    ];
    let out = PorterStemFilter::apply(tokens);
    assert_eq!(
        out,
        vec![
            tok("caress", 0, 8, 1),
            tok("poni", 0, 6, 1),
            tok("cat", 0, 4, 1),
            tok("caress", 0, 6, 1),
        ]
    );
}

#[test]
fn porter_step1b_ed_ing_and_short_word_guard() {
    let cases: &[(&str, &str)] = &[
        ("feed", "feed"),   // *v* fails on stem "f" -- must NOT stem.
        ("agreed", "agre"), // m(stem)>0 for "eed" -> "ee", then 5a strips "e".
        ("plastered", "plaster"),
        ("bled", "bled"), // stem "bl" has no vowel -- must NOT stem.
        ("motoring", "motor"),
        ("sing", "sing"), // stem "s" has no vowel -- must NOT stem.
    ];
    for (input, expected) in cases {
        let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
        assert_eq!(out[0].term, *expected, "stemming {input:?}");
    }
}

#[test]
fn porter_stem_leaves_offsets_and_position_increment_untouched() {
    let tokens = vec![tok("running", 5, 12, 2)];
    let out = PorterStemFilter::apply(tokens);
    assert_eq!(out, vec![tok("run", 5, 12, 2)]);
}

#[test]
fn porter_stem_happiness_and_running() {
    assert_eq!(
        PorterStemFilter::apply(vec![tok("running", 0, 1, 1)])[0].term,
        "run"
    );
    assert_eq!(
        PorterStemFilter::apply(vec![tok("flies", 0, 1, 1)])[0].term,
        "fli"
    );
    assert_eq!(
        PorterStemFilter::apply(vec![tok("happiness", 0, 1, 1)])[0].term,
        "happi"
    );
}

/// `y` immediately after a CONSONANT must count as a VOWEL (Porter's own
/// reference rule, `case 'y': return (i==0) ? TRUE : !cons(i-1);`), not a
/// consonant, when deciding whether a stem "contains a vowel" for step
/// 1b's `-ing` removal guard. So `contains_vowel("fly")` is `true`, and
/// `-ing` strips "flying"/"trying" to "fly"/"try" -- step 1c's own, separate
/// `(*v*)` condition (checked against the letters preceding the trailing
/// `y`, i.e. "fl"/"tr") doesn't additionally fire here since neither
/// contains a vowel, so the final `y` is not further converted to `i`.
#[test]
fn porter_stem_y_after_consonant_is_a_vowel_not_a_consonant() {
    assert_eq!(
        PorterStemFilter::apply(vec![tok("flying", 0, 1, 1)])[0].term,
        "fly",
        "the -ing suffix must be stripped now that y-after-consonant counts as a vowel"
    );
    assert_eq!(
        PorterStemFilter::apply(vec![tok("trying", 0, 1, 1)])[0].term,
        "try",
        "the -ing suffix must be stripped now that y-after-consonant counts as a vowel"
    );
}

#[test]
fn porter_stem_step2_step3_step4_suffix_families() {
    let cases: &[(&str, &str)] = &[
        ("relational", "relat"),
        ("conditional", "condit"),
        ("rational", "ration"),
        ("valenci", "valenc"),
        ("hesitanci", "hesit"),
        ("digitizer", "digit"),
        ("conformabli", "conform"),
        ("radicalli", "radic"),
        ("differentli", "differ"),
        ("vileli", "vile"),
        ("analogousli", "analog"),
        ("vietnamization", "vietnam"),
        ("predication", "predic"),
        ("operator", "oper"),
        ("feudalism", "feudal"),
        ("decisiveness", "decis"),
        ("hopefulness", "hope"),
        ("callousness", "callous"),
        ("formaliti", "formal"),
        ("sensitiviti", "sensit"),
        ("sensibiliti", "sensibl"),
        ("triplicate", "triplic"),
        ("formative", "form"),
        ("formalize", "formal"),
        ("electriciti", "electr"),
        ("electrical", "electr"),
        ("hopeful", "hope"),
        ("goodness", "good"),
        ("revival", "reviv"),
        ("allowance", "allow"),
        ("inference", "infer"),
        ("airliner", "airlin"),
        ("gyroscopic", "gyroscop"),
        ("adjustable", "adjust"),
        ("defensible", "defens"),
        ("irritant", "irrit"),
        ("replacement", "replac"),
        ("adjustment", "adjust"),
        ("dependent", "depend"),
        ("adoption", "adopt"),
        ("homologou", "homolog"),
        ("communism", "commun"),
        ("activate", "activ"),
        ("angulariti", "angular"),
        ("homologous", "homolog"),
        ("effective", "effect"),
        ("bowdlerize", "bowdler"),
    ];
    for (input, expected) in cases {
        let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
        assert_eq!(out[0].term, *expected, "stemming {input:?}");
    }
}

#[test]
fn porter_stem_step5_final_e_and_double_l() {
    let cases: &[(&str, &str)] = &[
        ("probate", "probat"),
        ("rate", "rate"), // m==1 and IS cvc -- 'e' must survive.
        ("cease", "ceas"),
        ("controll", "control"),
        ("roll", "roll"), // m==1, not >1 -- must NOT collapse.
    ];
    for (input, expected) in cases {
        let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
        assert_eq!(out[0].term, *expected, "stemming {input:?}");
    }
}

/// Porter's 1980 paper illustrates steps 1a/1b/1c with their own worked
/// vocabulary (distinct from the step 2/3/4 list already covered by
/// `porter_stem_step2_step3_step4_suffix_families`); this test traces
/// that vocabulary directly against the implementation to close a real
/// gap in this port's test coverage: step 1a's plural forms
/// (`caresses`/`ponies`/`ties`/`caress`/`cats`); step 1b's guards *not*
/// firing when they shouldn't (`feed` hits the `-eed` rule but
/// `m(fe) == 0` so it stays `feed`; `bled`/`sing` have no vowel before
/// `-ed`/`-ing` so they stay unchanged too) versus firing correctly
/// (`agreed`->`agre`, `plastered`->`plaster`, `motoring`->`motor`), plus
/// each of the three post-deletion cleanup branches (`-at`/`-bl`/`-iz`
/// append via `sized`->`size`; double-consonant-drop via
/// `hopping`->`hop`/`tanned`->`tan`/`falling`->`fall`/`hissing`->`hiss`/
/// `fizzed`->`fizz`; plain deletion with no cleanup needed via
/// `failing`->`fail`/`filing`->`file`); and step 1c's `-y`->`-i`
/// conversion (`happy`->`happi`) versus its vowel guard not firing
/// (`sky`->`sky`, since `sk` contains no vowel).
#[test]
fn porter_stem_step1a_step1b_step1c_paper_vocabulary() {
    let cases: &[(&str, &str)] = &[
        // Step 1a.
        ("caresses", "caress"),
        ("ponies", "poni"),
        ("ties", "ti"),
        ("caress", "caress"),
        ("cats", "cat"),
        // Step 1b: `-eed` with m==0 stays put; `-ed`/`-ing` with no vowel
        // in the stem stays put too.
        ("feed", "feed"),
        ("agreed", "agre"),
        ("plastered", "plaster"),
        ("bled", "bled"),
        ("motoring", "motor"),
        ("sing", "sing"),
        // Step 1b post-deletion cleanup, all three branches.
        ("conflated", "conflat"),
        ("troubled", "troubl"),
        ("sized", "size"),
        ("hopping", "hop"),
        ("tanned", "tan"),
        ("falling", "fall"),
        ("hissing", "hiss"),
        ("fizzed", "fizz"),
        ("failing", "fail"),
        ("filing", "file"),
        // Step 1c.
        ("happy", "happi"),
        ("sky", "sky"),
    ];
    for (input, expected) in cases {
        let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
        assert_eq!(out[0].term, *expected, "stemming {input:?}");
    }
}

/// Java's `PorterStemmer` has **no** "lowercase ASCII only" guard: its
/// `cons()` treats every character that is not `a/e/i/o/u/y` as a
/// consonant, so an uppercase or accented word is stemmed like any other.
/// So `"Running"` is reduced to `"Run"`, which matters for any chain that
/// does not lowercase first.
#[test]
fn porter_stem_has_no_ascii_lowercase_guard() {
    // "Running": step 1 strips `-ing` and undoubles `nn`, exactly as it
    // would for "running".
    assert_eq!(
        PorterStemFilter::apply(vec![tok("Running", 0, 7, 1)])[0].term,
        "Run"
    );
    // Words with no matching suffix still come back untouched, and
    // nothing panics on non-ASCII or digits.
    for word in ["café", "123", "Cat"] {
        let out = PorterStemFilter::apply(vec![tok(word, 0, 3, 1)]);
        assert_eq!(out[0].term, word, "stemming {word:?}");
    }
    // ... and "Cats" loses its plural `s` the same way "cats" does.
    assert_eq!(
        PorterStemFilter::apply(vec![tok("Cats", 0, 4, 1)])[0].term,
        "Cat"
    );
}

/// A non-ASCII term is stemmed over its UTF-16 units, as Java's
/// `char[]`: a supplementary character is two consonants, a Latin-1
/// letter one. The expected stems are Lucene 10.5's `PorterStemFilter`.
#[test]
fn porter_stem_non_ascii_terms_over_utf16_units() {
    for (word, java) in [
        ("😀öible", "😀öibl"),
        ("éational", "éation"),
        ("😀angularant", "😀angular"),
        ("ürevivales", "ürevival"),
        ("é😀ies", "é😀i"),
        ("😀activable", "😀activ"),
        ("ü😀ization", "ü😀izat"),
        ("Lucene", "Lucen"),
        ("é😀", "é😀"),
    ] {
        assert_eq!(porter::stem(word), java, "stemming {word:?}");
    }
}

/// Java runs the six steps only when `k > k0 + 1`, so a one- or
/// two-character word is returned verbatim. Without that guard step 1a
/// deleted the whole of `"s"`, producing a **zero-length term**.
#[test]
fn porter_stem_leaves_one_and_two_character_words_alone() {
    for word in ["s", "as", "is", "us", "es", "ay", "a", ""] {
        let out = PorterStemFilter::apply(vec![tok(word, 0, 1, 1)]);
        assert_eq!(out[0].term, word, "stemming {word:?}");
    }
}

/// Three step-3/step-5 rules: `bli -> ble` (not `abli -> able`), `logi ->
/// log`, and a suffix whose measure test fails must stop the search rather
/// than fall through to a shorter suffix.
#[test]
fn porter_stem_first_matching_suffix_wins_even_when_its_measure_test_fails() {
    let cases: &[(&str, &str)] = &[
        // `bli -> ble`, not `abli -> able`.
        ("possibly", "possibl"),
        // The `g` group, which was absent.
        ("technology", "technolog"),
        ("apology", "apolog"),
        // ... and `logi` still only fires when `m(stem) > 0`, so "bio"
        // (measure 0) blocks the replacement without falling through.
        ("biology", "biologi"),
        // `-ment` matches first with m("argu") == 1, which is not > 1, so
        // Java stops there instead of falling through to `-ent`.
        ("argument", "argument"),
        // `-ization` matches with m("") == 0, so no `-ation` fallback.
        ("ization", "izat"),
    ];
    for (input, expected) in cases {
        let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
        assert_eq!(out[0].term, *expected, "stemming {input:?}");
    }
}

#[test]
fn analyzer_with_stemming_runs_after_stopwords() {
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let analyzer = Analyzer::standard(Some(&stopwords)).with_stemming();
    let out = analyzer.analyze("The Running Flies");
    assert_eq!(out, vec![tok("run", 4, 11, 2), tok("fli", 12, 17, 1),]);
}

#[test]
fn analyzer_default_has_no_stemming_backward_compatible() {
    let analyzer = Analyzer::standard(None);
    let out = analyzer.analyze("running");
    assert_eq!(out, vec![tok("running", 0, 7, 1)]);
}

#[test]
fn synonym_filter_injects_single_synonym_at_same_position() {
    let tokens = vec![tok("quick", 0, 5, 1)];
    let synonyms: HashMap<String, Vec<String>> = [("quick".to_string(), vec!["fast".to_string()])]
        .into_iter()
        .collect();
    let out = SynonymFilter::apply(tokens, &synonyms);
    assert_eq!(out, vec![tok("quick", 0, 5, 1), tok("fast", 0, 5, 0),]);
}

#[test]
fn synonym_filter_multiple_synonyms_all_same_position() {
    let tokens = vec![tok("quick", 0, 5, 1)];
    let synonyms: HashMap<String, Vec<String>> = [(
        "quick".to_string(),
        vec!["fast".to_string(), "speedy".to_string()],
    )]
    .into_iter()
    .collect();
    let out = SynonymFilter::apply(tokens, &synonyms);
    assert_eq!(
        out,
        vec![
            tok("quick", 0, 5, 1),
            tok("fast", 0, 5, 0),
            tok("speedy", 0, 5, 0),
        ]
    );
}

#[test]
fn synonym_filter_no_configured_synonym_passes_through_unchanged() {
    let tokens = vec![tok("hello", 0, 5, 1)];
    let synonyms: HashMap<String, Vec<String>> = [("quick".to_string(), vec!["fast".to_string()])]
        .into_iter()
        .collect();
    let out = SynonymFilter::apply(tokens.clone(), &synonyms);
    assert_eq!(out, tokens);
}

#[test]
fn synonym_filter_not_automatically_bidirectional() {
    // Configuring "quick" -> "fast" must NOT also expand "fast" ->
    // "quick" -- real Lucene requires explicit configuration in both
    // directions.
    let tokens = vec![tok("fast", 0, 4, 1)];
    let synonyms: HashMap<String, Vec<String>> = [("quick".to_string(), vec!["fast".to_string()])]
        .into_iter()
        .collect();
    let out = SynonymFilter::apply(tokens.clone(), &synonyms);
    assert_eq!(out, tokens);
}

#[test]
fn synonym_filter_composed_with_stop_filter_stopword_removed_before_expansion() {
    // "the quick fox" with "the" as a stopword and "quick" -> "fast"
    // configured: stopwords run first, so "the" is gone and never
    // considered for synonym expansion (it isn't in the map anyway, but
    // this also proves the ordering doesn't crash/misbehave on a
    // stopword-adjacent term); "quick" survives and still expands.
    let analyzer = {
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let synonyms: HashMap<String, Vec<String>> =
            [("quick".to_string(), vec!["fast".to_string()])]
                .into_iter()
                .collect();
        Analyzer::standard(Some(&stopwords)).with_synonyms(synonyms)
    };
    let out = analyzer.analyze("the quick fox");
    assert_eq!(
        out,
        vec![
            tok("quick", 4, 9, 2),
            tok("fast", 4, 9, 0),
            tok("fox", 10, 13, 1),
        ]
    );
}

#[test]
fn synonym_filter_stopword_itself_never_gets_expanded() {
    // If the stopword itself had a configured synonym, it must not
    // survive to be expanded, since it's removed before synonym
    // expansion runs.
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let synonyms: HashMap<String, Vec<String>> =
        [("the".to_string(), vec!["definite_article".to_string()])]
            .into_iter()
            .collect();
    let analyzer = Analyzer::standard(Some(&stopwords)).with_synonyms(synonyms);
    let out = analyzer.analyze("the fox");
    assert_eq!(out, vec![tok("fox", 4, 7, 2)]);
}

#[test]
fn synonym_filter_runs_after_stemming() {
    // Configuring the map with the STEMMED form ("run") as the key
    // proves synonyms see post-stemming terms, since stemming runs
    // before synonym expansion in the chain.
    let synonyms: HashMap<String, Vec<String>> = [("run".to_string(), vec!["sprint".to_string()])]
        .into_iter()
        .collect();
    let analyzer = Analyzer::standard(None)
        .with_stemming()
        .with_synonyms(synonyms);
    let out = analyzer.analyze("running");
    assert_eq!(out, vec![tok("run", 0, 7, 1), tok("sprint", 0, 7, 0),]);
}

#[test]
fn synonym_filter_apply_bidirectional_expands_both_directions() {
    // Only "cat" -> ["feline"] is configured; apply_bidirectional must
    // ALSO expand "feline" -> "cat" without that reverse mapping being
    // configured explicitly.
    let synonyms: HashMap<String, Vec<String>> = [("cat".to_string(), vec!["feline".to_string()])]
        .into_iter()
        .collect();

    let out_forward = SynonymFilter::apply_bidirectional(vec![tok("cat", 0, 3, 1)], &synonyms);
    assert_eq!(
        out_forward,
        vec![tok("cat", 0, 3, 1), tok("feline", 0, 3, 0)]
    );

    let out_reverse = SynonymFilter::apply_bidirectional(vec![tok("feline", 0, 6, 1)], &synonyms);
    assert_eq!(
        out_reverse,
        vec![tok("feline", 0, 6, 1), tok("cat", 0, 6, 0)]
    );
}

#[test]
fn synonym_filter_apply_non_bidirectional_still_unidirectional() {
    // The original `apply` entry point must remain completely unchanged:
    // with the same config, analyzing "feline" injects nothing.
    let synonyms: HashMap<String, Vec<String>> = [("cat".to_string(), vec!["feline".to_string()])]
        .into_iter()
        .collect();
    let tokens = vec![tok("feline", 0, 6, 1)];
    let out = SynonymFilter::apply(tokens.clone(), &synonyms);
    assert_eq!(out, tokens);
}

#[test]
fn synonym_filter_apply_bidirectional_no_duplicate_when_both_directions_configured() {
    // "cat" -> ["feline"] AND "feline" -> ["cat"] both explicitly
    // configured: the merged map must not inject "cat" (or "feline")
    // twice.
    let synonyms: HashMap<String, Vec<String>> = [
        ("cat".to_string(), vec!["feline".to_string()]),
        ("feline".to_string(), vec!["cat".to_string()]),
    ]
    .into_iter()
    .collect();

    let out_cat = SynonymFilter::apply_bidirectional(vec![tok("cat", 0, 3, 1)], &synonyms);
    assert_eq!(out_cat, vec![tok("cat", 0, 3, 1), tok("feline", 0, 3, 0)]);

    let out_feline = SynonymFilter::apply_bidirectional(vec![tok("feline", 0, 6, 1)], &synonyms);
    assert_eq!(
        out_feline,
        vec![tok("feline", 0, 6, 1), tok("cat", 0, 6, 0)]
    );
}

#[test]
fn synonym_filter_apply_bidirectional_multi_value_key_reverses_independently() {
    // "cat" -> ["feline", "kitty"]: the reverse mapping must produce two
    // SEPARATE entries, "feline" -> ["cat"] and "kitty" -> ["cat"] --
    // "feline" and "kitty" must NOT become synonyms of each other, since
    // the forward config never said that.
    let synonyms: HashMap<String, Vec<String>> = [(
        "cat".to_string(),
        vec!["feline".to_string(), "kitty".to_string()],
    )]
    .into_iter()
    .collect();

    let out_feline = SynonymFilter::apply_bidirectional(vec![tok("feline", 0, 6, 1)], &synonyms);
    assert_eq!(
        out_feline,
        vec![tok("feline", 0, 6, 1), tok("cat", 0, 6, 0)]
    );

    let out_kitty = SynonymFilter::apply_bidirectional(vec![tok("kitty", 0, 5, 1)], &synonyms);
    assert_eq!(out_kitty, vec![tok("kitty", 0, 5, 1), tok("cat", 0, 5, 0)]);

    // Forward direction still expands to BOTH synonyms, unaffected.
    let out_cat = SynonymFilter::apply_bidirectional(vec![tok("cat", 0, 3, 1)], &synonyms);
    assert_eq!(
        out_cat,
        vec![
            tok("cat", 0, 3, 1),
            tok("feline", 0, 3, 0),
            tok("kitty", 0, 3, 0),
        ]
    );
}

#[test]
fn synonym_filter_bidirectional_composed_with_stop_filter() {
    // Mirrors synonym_filter_composed_with_stop_filter_stopword_removed_before_expansion,
    // but with bidirectional mode on: "the cat fox" with "the" as a
    // stopword and "cat" -> ["feline"] configured bidirectionally --
    // stopwords still run first, and "cat" still expands to "feline".
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let synonyms: HashMap<String, Vec<String>> = [("cat".to_string(), vec!["feline".to_string()])]
        .into_iter()
        .collect();
    let analyzer = Analyzer::standard(Some(&stopwords)).with_bidirectional_synonyms(synonyms);
    let out = analyzer.analyze("the cat fox");
    assert_eq!(
        out,
        vec![
            tok("cat", 4, 7, 2),
            tok("feline", 4, 7, 0),
            tok("fox", 8, 11, 1),
        ]
    );
}

#[test]
fn synonym_filter_bidirectional_composed_with_stemming() {
    // Mirrors synonym_filter_runs_after_stemming, but bidirectional:
    // configuring the STEMMED form "run" -> ["sprint"] bidirectionally
    // means analyzing "sprint" (already the stemmed form of itself)
    // also injects "run".
    let synonyms: HashMap<String, Vec<String>> = [("run".to_string(), vec!["sprint".to_string()])]
        .into_iter()
        .collect();
    let analyzer = Analyzer::standard(None)
        .with_stemming()
        .with_bidirectional_synonyms(synonyms);

    let out_forward = analyzer.analyze("running");
    assert_eq!(
        out_forward,
        vec![tok("run", 0, 7, 1), tok("sprint", 0, 7, 0),]
    );

    let out_reverse = analyzer.analyze("sprint");
    assert_eq!(
        out_reverse,
        vec![tok("sprint", 0, 6, 1), tok("run", 0, 6, 0),]
    );
}

#[test]
fn synonym_filter_multiword_input_collapses_to_single_output_token() {
    // "wi fi" -> "wifi": a 2-token input phrase becomes 1 output token
    // with position_length == 2, marking it spans both original
    // positions. Offsets cover the whole matched span.
    let tokens = vec![tok("wi", 0, 2, 1), tok("fi", 3, 5, 1)];
    let rules = vec![SynonymRule {
        input: vec!["wi".to_string(), "fi".to_string()],
        outputs: vec![vec!["wifi".to_string()]],
    }];
    let out = SynonymFilter::apply_multiword(tokens, &rules);
    assert_eq!(
        out,
        // Verified against real `SynonymGraphFilter`:
        // `wifi:1:2:0,5;wi:0:1:0,2;fi:1:1:3,5`.
        vec![
            tok_len("wifi", 0, 5, 1, 2),
            tok("wi", 0, 2, 0),
            tok("fi", 3, 5, 1),
        ]
    );
}

#[test]
fn synonym_filter_single_word_input_expands_to_multiword_output() {
    // "usa" -> "united states of america": 1 input token becomes 4
    // chained output tokens, first at position_increment 0 (same slot as
    // "usa"), the rest at position_increment 1 each, all position_length
    // 1 (each occupies exactly one position on the output path).
    let tokens = vec![tok("usa", 0, 3, 1)];
    let rules = vec![SynonymRule {
        input: vec!["usa".to_string()],
        outputs: vec![vec![
            "united".to_string(),
            "states".to_string(),
            "of".to_string(),
            "america".to_string(),
        ]],
    }];
    let out = SynonymFilter::apply_multiword(tokens, &rules);
    assert_eq!(
        out,
        // Real `SynonymGraphFilter`:
        // `united:1:1:0,3;usa:0:4:0,3;states:1:1:0,3;of:1:1:0,3;america:1:1:0,3`.
        // Note the *original* gets position_length 4: both paths have to
        // rejoin at the same end node.
        vec![
            tok_len("united", 0, 3, 1, 1),
            tok_len("usa", 0, 3, 0, 4),
            tok_len("states", 0, 3, 1, 1),
            tok_len("of", 0, 3, 1, 1),
            tok_len("america", 0, 3, 1, 1),
        ]
    );
}

#[test]
fn synonym_filter_multiword_to_multiword() {
    // "new york" -> "big apple": a 2-token input phrase to a 2-token
    // output phrase. The output's first token gets position_length 1
    // (not the input's length of 2), since output.len() > 1.
    let tokens = vec![tok("new", 0, 3, 1), tok("york", 4, 8, 1)];
    let rules = vec![SynonymRule {
        input: vec!["new".to_string(), "york".to_string()],
        outputs: vec![vec!["big".to_string(), "apple".to_string()]],
    }];
    let out = SynonymFilter::apply_multiword(tokens, &rules);
    assert_eq!(
        out,
        // Real `SynonymGraphFilter`:
        // `big:1:1:0,8;new:0:2:0,3;apple:1:2:0,8;york:1:1:4,8`.
        vec![
            tok_len("big", 0, 8, 1, 1),
            tok_len("new", 0, 3, 0, 2),
            tok_len("apple", 0, 8, 1, 2),
            tok("york", 4, 8, 1),
        ]
    );
}

/// The graph's node counter has to keep advancing across tokens that
/// match nothing, or a later match starts at the wrong node. Real
/// `SynonymGraphFilter` over `"the wi fi router"` gives
/// `the:1:1:0,3;wifi:1:2:4,9;wi:0:1:4,6;fi:1:1:7,9;router:1:1:10,16`.
#[test]
fn synonym_filter_multiword_match_surrounded_by_pass_through_tokens() {
    let tokens = vec![
        tok("the", 0, 3, 1),
        tok("wi", 4, 6, 1),
        tok("fi", 7, 9, 1),
        tok("router", 10, 16, 1),
    ];
    let rules = vec![SynonymRule {
        input: vec!["wi".to_string(), "fi".to_string()],
        outputs: vec![vec!["wifi".to_string()]],
    }];
    let out = SynonymFilter::apply_multiword(tokens, &rules);
    assert_eq!(
        out,
        vec![
            tok("the", 0, 3, 1),
            tok_len("wifi", 4, 9, 1, 2),
            tok("wi", 4, 6, 0),
            tok("fi", 7, 9, 1),
            tok("router", 10, 16, 1),
        ]
    );
}

#[test]
fn synonym_filter_multiword_partial_prefix_does_not_match() {
    // "wi" alone (not followed by "fi") must NOT trigger the "wi fi"
    // rule -- neither mid-stream (followed by something else) nor as the
    // very last token (nothing following at all).
    let rules = vec![SynonymRule {
        input: vec!["wi".to_string(), "fi".to_string()],
        outputs: vec![vec!["wifi".to_string()]],
    }];

    let followed_by_other = vec![tok("wi", 0, 2, 1), tok("max", 3, 6, 1)];
    let out = SynonymFilter::apply_multiword(followed_by_other.clone(), &rules);
    assert_eq!(out, followed_by_other);

    let last_token = vec![tok("wi", 0, 2, 1)];
    let out = SynonymFilter::apply_multiword(last_token.clone(), &rules);
    assert_eq!(out, last_token);
}

#[test]
fn synonym_filter_multiword_no_rules_passes_through_unchanged() {
    let tokens = vec![tok("hello", 0, 5, 1), tok("world", 6, 11, 1)];
    let out = SynonymFilter::apply_multiword(tokens.clone(), &[]);
    assert_eq!(out, tokens);
}

#[test]
fn synonym_filter_multiword_prefers_longest_match() {
    // Both "new" -> "novel" and "new york" -> "nyc" configured; the
    // longer "new york" phrase should win over the shorter "new" rule
    // when both could match at the same starting position.
    let tokens = vec![tok("new", 0, 3, 1), tok("york", 4, 8, 1)];
    let rules = vec![
        SynonymRule {
            input: vec!["new".to_string()],
            outputs: vec![vec!["novel".to_string()]],
        },
        SynonymRule {
            input: vec!["new".to_string(), "york".to_string()],
            outputs: vec![vec!["nyc".to_string()]],
        },
    ];
    let out = SynonymFilter::apply_multiword(tokens, &rules);
    assert_eq!(
        out,
        // Real `SynonymGraphFilter`:
        // `nyc:1:2:0,8;new:0:1:0,3;york:1:1:4,8`.
        vec![
            tok_len("nyc", 0, 8, 1, 2),
            tok("new", 0, 3, 0),
            tok("york", 4, 8, 1),
        ]
    );
}

#[test]
fn synonym_filter_multiword_multiple_output_alternatives() {
    // A single multi-word input can have more than one alternative
    // output path (e.g. "wi fi" -> "wifi" or "wireless").
    let tokens = vec![tok("wi", 0, 2, 1), tok("fi", 3, 5, 1)];
    let rules = vec![SynonymRule {
        input: vec!["wi".to_string(), "fi".to_string()],
        outputs: vec![vec!["wifi".to_string()], vec!["wireless".to_string()]],
    }];
    let out = SynonymFilter::apply_multiword(tokens, &rules);
    assert_eq!(
        out,
        // Real `SynonymGraphFilter`:
        // `wifi:1:2:0,5;wireless:0:2:0,5;wi:0:1:0,2;fi:1:1:3,5`.
        vec![
            tok_len("wifi", 0, 5, 1, 2),
            tok_len("wireless", 0, 5, 0, 2),
            tok("wi", 0, 2, 0),
            tok("fi", 3, 5, 1),
        ]
    );
}

#[test]
fn synonym_filter_apply_and_apply_bidirectional_unaffected_by_multiword_addition() {
    // Sanity check the earlier single-word bidirectional task's behavior
    // is untouched by adding apply_multiword: same assertions as
    // synonym_filter_apply_bidirectional_expands_both_directions.
    let synonyms: HashMap<String, Vec<String>> = [("cat".to_string(), vec!["feline".to_string()])]
        .into_iter()
        .collect();

    let out_forward = SynonymFilter::apply_bidirectional(vec![tok("cat", 0, 3, 1)], &synonyms);
    assert_eq!(
        out_forward,
        vec![tok("cat", 0, 3, 1), tok("feline", 0, 3, 0)]
    );

    let out_reverse = SynonymFilter::apply_bidirectional(vec![tok("feline", 0, 6, 1)], &synonyms);
    assert_eq!(
        out_reverse,
        vec![tok("feline", 0, 6, 1), tok("cat", 0, 6, 0)]
    );
}

#[test]
fn analyzer_default_has_no_synonyms_backward_compatible() {
    let analyzer = Analyzer::standard(None);
    let out = analyzer.analyze("quick");
    assert_eq!(out, vec![tok("quick", 0, 5, 1)]);
}

#[test]
fn analyzer_default_has_no_folding_backward_compatible() {
    // Default Analyzer::standard (no with_ascii_folding call) leaves
    // diacritics as-is, only lowercasing -- unchanged behavior for every
    // existing caller (query_parser.rs, indexing_chain.rs).
    let analyzer = Analyzer::standard(None);
    let out = analyzer.analyze("Café");
    assert_eq!(out, vec![tok("café", 0, 4, 1)]);
}

// -- Analyzer::keyword --

#[test]
fn keyword_analyzer_emits_whole_text_as_one_unmodified_token() {
    let analyzer = Analyzer::keyword();
    let out = analyzer.analyze("Status: ACTIVE!");
    assert_eq!(out, vec![tok("Status: ACTIVE!", 0, 15, 1)]);
}

/// A pooled analyzer gives each document the previous one's string reader:
/// a longer text, a shorter one, an empty one and a supplementary-plane
/// one each tokenize exactly as with a new analyzer.
#[test]
fn consecutive_documents_reuse_the_string_reader() {
    let texts = [
        "a longer first document, with words",
        "short",
        "",
        "x\u{1f600}y z\u{10400}",
        "last one",
        "a longer first document, with words",
    ];
    let pooled_keyword = Analyzer::keyword();
    let pooled_standard = Analyzer::standard(None);
    for text in texts {
        assert_eq!(
            pooled_keyword.analyze(text),
            Analyzer::keyword().analyze(text),
            "{text:?}"
        );
        assert_eq!(
            pooled_standard.analyze(text),
            Analyzer::standard(None).analyze(text),
            "{text:?}"
        );
    }
}

#[test]
fn keyword_analyzer_does_not_split_on_whitespace() {
    // Distinct from Analyzer::standard/tokenize -- whitespace-separated
    // words are NOT split into separate tokens.
    let analyzer = Analyzer::keyword();
    let out = analyzer.analyze("the quick fox");
    assert_eq!(out, vec![tok("the quick fox", 0, 13, 1)]);
}

#[test]
fn keyword_analyzer_does_not_lowercase() {
    let analyzer = Analyzer::keyword();
    let out = analyzer.analyze("UPPER");
    assert_eq!(out, vec![tok("UPPER", 0, 5, 1)]);
}

#[test]
fn keyword_analyzer_ignores_stopwords_even_if_term_matches() {
    // A term that would be a stopword under Analyzer::standard's
    // pipeline is passed through untouched -- keyword semantics have no
    // stopword filtering at all.
    let analyzer = Analyzer::keyword();
    let out = analyzer.analyze("the");
    assert_eq!(out, vec![tok("the", 0, 3, 1)]);
}

#[test]
fn keyword_analyzer_empty_input_still_emits_one_empty_token() {
    // Matches real Lucene's KeywordTokenizer: incrementToken()
    // unconditionally succeeds once, even over zero characters.
    let analyzer = Analyzer::keyword();
    let out = analyzer.analyze("");
    assert_eq!(out, vec![tok("", 0, 0, 1)]);
}

#[test]
fn keyword_analyzer_end_offset_is_the_java_char_count() {
    // Java's KeywordTokenizer ends its one token at
    // `correctOffset(charCount)`. "id-<U+1F600>-<U+00E9>" is 6 Unicode
    // scalars, **7** UTF-16 code units and 10 UTF-8 bytes, so all three
    // candidate units are visibly distinct here.
    let text = "id-\u{1F600}-\u{E9}";
    assert_eq!(text.chars().count(), 6);
    assert_eq!(text.len(), 10);
    let out = Analyzer::keyword().analyze(text);
    assert_eq!(out, vec![tok(text, 0, 7, 1)]);
}

#[test]
fn tokenize_offsets_are_utf16_code_units_not_bytes_or_scalars() {
    // One text where all three units disagree at once: an accented letter
    // (1 char / 2 bytes), an ideograph (1 char / 3 bytes) and an astral
    // symbol (2 chars / 1 scalar / 4 bytes).
    let text = "alpha caf\u{E9} \u{4E16} \u{1D306} omega";
    let tokens = tokenize(text);
    assert_eq!(
        tokens,
        vec![
            tok("alpha", 0, 5, 1),
            tok("caf\u{E9}", 6, 10, 1),
            tok("\u{4E16}", 11, 12, 1),
            tok("omega", 16, 21, 1),
        ]
    );
    // Not the byte offsets (which would put "omega" at 21)...
    assert_eq!(text.len(), 26);
    // ...and not the scalar offsets either (which would put it at 15).
    assert_eq!(text.chars().count(), 20);
}

#[test]
fn utf16_len_agrees_with_encode_utf16_on_both_branches() {
    // The ASCII fast path and the general path must not disagree.
    for text in [
        "",
        "plain ascii",
        "caf\u{E9}",
        "\u{4E16}\u{754C}",
        "a\u{1F600}b",
        "e\u{301}",
    ] {
        assert_eq!(
            utf16_len(text),
            text.encode_utf16().count(),
            "utf16_len disagreed for {text:?}"
        );
    }
}

/// The word-at-a-time count agrees with `encode_utf16` on strings of every
/// length around the 8-byte word and 248-byte block boundaries.
#[test]
fn utf16_len_counts_words_and_blocks_exactly() {
    let alphabet = ["a", "\u{e9}", "\u{4e16}", "\u{1f600}", "z", "\u{10400}"];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    for len in (0..40).chain([247, 248, 249, 250, 495, 496, 497, 1000, 2500]) {
        let mut text = String::new();
        for _ in 0..len {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            text.push_str(alphabet[(state % alphabet.len() as u64) as usize]);
        }
        assert_eq!(
            utf16_len(&text),
            text.encode_utf16().count(),
            "{len} scalars"
        );
    }
    // the densest run of four-byte leads one block can hold
    let dense = "\u{1f600}".repeat(400);
    assert_eq!(utf16_len(&dense), 800);
}

#[test]
fn keyword_analyzer_builder_methods_have_no_effect() {
    // Calling any of Analyzer's filter-chain builders on a keyword
    // analyzer doesn't change analyze()'s output -- keyword mode
    // short-circuits before any of those fields are consulted.
    let mut synonyms = HashMap::new();
    synonyms.insert("the".to_string(), vec!["a".to_string()]);

    let plain = Analyzer::keyword().analyze("the");
    let with_everything = Analyzer::keyword()
        .with_ascii_folding()
        .with_stemming()
        .with_synonyms(synonyms)
        .analyze("the");
    assert_eq!(plain, with_everything);
    assert_eq!(plain, vec![tok("the", 0, 3, 1)]);
}

// -- NGramTokenFilter / EdgeNGramTokenFilter --

#[test]
fn ngram_filter_abcde_min2_max3_exact_gram_set_and_order() {
    let tokens = vec![tok("abcde", 0, 5, 1)];
    let out = NGramTokenFilter::apply(tokens, 2, 3).unwrap();
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["ab", "abc", "bc", "bcd", "cd", "cde", "de"]);
    // First gram keeps the original token's position_increment; every
    // subsequent gram from the same input token is position_increment 0.
    let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
    assert_eq!(pos_incs, vec![1, 0, 0, 0, 0, 0, 0]);
    assert!(out.iter().all(|t| t.position_length == 1));
    // Every gram carries the *input token's* offsets: Java restores the
    // captured state per gram and never calls `setOffset`.
    assert!(out.iter().all(|t| (t.start_offset, t.end_offset) == (0, 5)));
}

#[test]
fn edge_ngram_filter_abcde_min2_max4_exact_prefix_gram_set() {
    let tokens = vec![tok("abcde", 0, 5, 1)];
    let out = EdgeNGramTokenFilter::apply(tokens, 2, 4).unwrap();
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["ab", "abc", "abcd"]);
    let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
    assert_eq!(pos_incs, vec![1, 0, 0]);
    assert!(out.iter().all(|t| (t.start_offset, t.end_offset) == (0, 5)));
}

/// Java accumulates `curPosIncr` across input tokens that emit nothing,
/// so a token skipped for being shorter than `min_gram` still pushes the
/// next token's grams along.
#[test]
fn ngram_filter_carries_the_increment_of_a_skipped_short_token() {
    let tokens = vec![tok("big", 0, 3, 1), tok("a", 4, 5, 1), tok("cat", 6, 9, 1)];
    let out = NGramTokenFilter::apply(tokens, 3, 3).unwrap();
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["big", "cat"]);
    // "cat" is two positions after "big", not one: "a" produced no gram
    // but still consumed a position.
    let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
    assert_eq!(pos_incs, vec![1, 2]);
}

/// `NGramTokenFilter`'s `preserveOriginal` flag: too-short tokens are
/// kept (carrying the accumulated increment) and too-long tokens are
/// re-emitted after their grams at increment 0.
#[test]
fn ngram_filter_preserve_original_keeps_short_and_long_tokens() {
    let out = NGramTokenFilter::apply_preserving_original(
        vec![tok("a", 0, 1, 1), tok("abcd", 2, 6, 1)],
        2,
        3,
    )
    .unwrap();
    let observed: Vec<(&str, i32)> = out
        .iter()
        .map(|t| (t.term.as_str(), t.position_increment))
        .collect();
    assert_eq!(
        observed,
        vec![
            ("a", 1),
            ("ab", 1),
            ("abc", 0),
            ("bc", 0),
            ("bcd", 0),
            ("cd", 0),
            // "abcd" is longer than max_gram, so the original comes back.
            ("abcd", 0),
        ]
    );

    // A token exactly `max_gram` long is not duplicated.
    let out = NGramTokenFilter::apply_preserving_original(vec![tok("abc", 0, 3, 1)], 2, 3).unwrap();
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["ab", "abc", "bc"]);
}

#[test]
fn edge_ngram_filter_preserve_original_keeps_short_and_long_tokens() {
    let out = EdgeNGramTokenFilter::apply_preserving_original(
        vec![tok("a", 0, 1, 1), tok("abcd", 2, 6, 1)],
        2,
        3,
    )
    .unwrap();
    let observed: Vec<(&str, i32)> = out
        .iter()
        .map(|t| (t.term.as_str(), t.position_increment))
        .collect();
    assert_eq!(observed, vec![("a", 1), ("ab", 1), ("abc", 0), ("abcd", 0)]);
}

#[test]
fn ngram_filter_token_shorter_than_min_gram_produces_no_output() {
    let tokens = vec![tok("ab", 0, 2, 1)];
    let out = NGramTokenFilter::apply(tokens, 3, 5).unwrap();
    assert_eq!(out, vec![]);
}

#[test]
fn edge_ngram_filter_token_shorter_than_min_gram_produces_no_output() {
    let tokens = vec![tok("ab", 0, 2, 1)];
    let out = EdgeNGramTokenFilter::apply(tokens, 3, 5).unwrap();
    assert_eq!(out, vec![]);
}

#[test]
fn ngram_filter_min_gram_greater_than_max_gram_is_config_error() {
    let tokens = vec![tok("abcde", 0, 5, 1)];
    let err = NGramTokenFilter::apply(tokens, 4, 2).unwrap_err();
    assert!(err.contains("min_gram"));
}

#[test]
fn edge_ngram_filter_min_gram_greater_than_max_gram_is_config_error() {
    let tokens = vec![tok("abcde", 0, 5, 1)];
    let err = EdgeNGramTokenFilter::apply(tokens, 4, 2).unwrap_err();
    assert!(err.contains("min_gram"));
}

#[test]
fn ngram_filter_zero_or_negative_gram_sizes_are_config_errors() {
    let tokens = vec![tok("abcde", 0, 5, 1)];
    assert!(NGramTokenFilter::apply(tokens.clone(), 0, 3).is_err());
    assert!(NGramTokenFilter::apply(tokens.clone(), 1, 0).is_err());
    assert!(NGramTokenFilter::apply(tokens.clone(), -1, 3).is_err());
    assert!(NGramTokenFilter::apply(tokens, 2, -2).is_err());
}

#[test]
fn edge_ngram_filter_zero_or_negative_gram_sizes_are_config_errors() {
    let tokens = vec![tok("abcde", 0, 5, 1)];
    assert!(EdgeNGramTokenFilter::apply(tokens.clone(), 0, 3).is_err());
    assert!(EdgeNGramTokenFilter::apply(tokens, 1, -1).is_err());
}

#[test]
fn ngram_filter_single_character_token() {
    // A single-char token with min_gram == 1 produces exactly one gram
    // equal to the whole token.
    let tokens = vec![tok("a", 0, 1, 1)];
    let out = NGramTokenFilter::apply(tokens, 1, 3).unwrap();
    assert_eq!(out, vec![tok("a", 0, 1, 1)]);
}

#[test]
fn ngram_filter_multibyte_unicode_grams_by_codepoint_not_byte() {
    // "café" -- 'é' is 2 bytes in UTF-8, so byte-based gramming would
    // either split it into invalid UTF-8 or misalign lengths. Grammed by
    // codepoint (4 chars: c,a,f,é) with min=2/max=2: "ca","af","fé".
    let tokens = vec![tok("café", 0, 5, 1)];
    let out = NGramTokenFilter::apply(tokens, 2, 2).unwrap();
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["ca", "af", "fé"]);
    // Offsets stay the input token's, as Lucene leaves them.
    assert!(out.iter().all(|t| (t.start_offset, t.end_offset) == (0, 5)));
}

#[test]
fn edge_ngram_filter_multibyte_unicode_grams_by_codepoint_not_byte() {
    let tokens = vec![tok("café", 0, 5, 1)];
    let out = EdgeNGramTokenFilter::apply(tokens, 1, 4).unwrap();
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["c", "ca", "caf", "café"]);
    assert_eq!((out[3].start_offset, out[3].end_offset), (0, 5));
}

#[test]
fn ngram_filter_multiple_tokens_grammed_independently() {
    // Each input token is grammed on its own -- no gramming across token
    // boundaries.
    let tokens = vec![tok("ab", 0, 2, 1), tok("cd", 3, 5, 1)];
    let out = NGramTokenFilter::apply(tokens, 2, 2).unwrap();
    assert_eq!(out, vec![tok("ab", 0, 2, 1), tok("cd", 3, 5, 1)]);
}

#[test]
fn edge_ngram_filter_multiple_tokens_grammed_independently() {
    let tokens = vec![tok("abc", 0, 3, 1), tok("xyz", 4, 7, 1)];
    let out = EdgeNGramTokenFilter::apply(tokens, 1, 2).unwrap();
    let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["a", "ab", "x", "xy"]);
    // Second input token's grams carry its own position_increment on the
    // first gram, then 0 for the rest -- independent of the first
    // token's grams.
    let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
    assert_eq!(pos_incs, vec![1, 0, 1, 0]);
}

// -- SnowballEnglishStemFilter (Porter2/Snowball) --

fn snowball(term: &str) -> String {
    SnowballEnglishStemFilter::apply(vec![tok(term, 0, 1, 1)])[0]
        .term
        .clone()
}

#[test]
fn snowball_english_composed_via_analyzer_builder() {
    // Analyzer::with_snowball_stemming wires SnowballEnglishStemFilter
    // in as the analyzer's last stage, mirroring with_stemming's shape.
    let tokens = Analyzer::standard(None)
        .with_snowball_stemming()
        .analyze("running dogs");
    let terms: Vec<&str> = tokens.iter().map(|t| t.term.as_str()).collect();
    assert_eq!(terms, vec!["run", "dog"]);
}

#[test]
fn snowball_english_snowball_stemming_takes_precedence_over_classic() {
    // If both builders are enabled, the Snowball stemmer wins (no
    // double-stemming) -- use a word where the two algorithms are known
    // to diverge: classic Porter's step 1b cleanup only re-appends a
    // trailing `e` when the 2-letter "CVC" check applies (which
    // requires 3+ characters, so never fires for a 2-letter stem),
    // giving "owed" -> "ow"; Snowball's own short-syllable definition
    // explicitly covers the 2-letter vowel+consonant case, giving
    // "owed" -> "owe" (see the dedicated
    // `snowball_english_short_word_append_e_fallback` test).
    let tokens = Analyzer::standard(None)
        .with_stemming()
        .with_snowball_stemming()
        .analyze("owed");
    assert_eq!(tokens[0].term, "owe");
}

#[test]
fn snowball_english_leading_apostrophe_stripped_by_prelude() {
    // Real StandardTokenizer strips a bare leading apostrophe before
    // the stemmer ever sees it (confirmed by this crate's own
    // differential fixture, where "'tis" tokenizes to the term "tis"
    // directly) -- so this exercises SnowballEnglishStemFilter's own
    // prelude apostrophe-stripping directly, on a term a caller might
    // construct without going through tokenize() first.
    assert_eq!(snowball("'tis"), "tis");
}

#[test]
fn snowball_english_step0_apostrophe_suffix_variants() {
    // Step 0's three apostrophe-suffix patterns, longest-match-first:
    // trailing "'s'" (rare, but part of the real Among table), "'s"
    // (possessive), and a bare trailing "'".
    assert_eq!(snowball("ab's'"), "ab");
    assert_eq!(snowball("dog's"), "dog");
    assert_eq!(snowball("boys'"), "boy"); // "'" stripped, then plain "s" stripped too.
}

#[test]
fn snowball_english_step1a_suffix_family() {
    assert_eq!(snowball("caresses"), "caress"); // sses -> ss
    assert_eq!(snowball("ponies"), "poni"); // ies, stem len > 1 -> i
    assert_eq!(snowball("ties"), "tie"); // ies, stem len <= 1 -> ie
    assert_eq!(snowball("caress"), "caress"); // ss protected, no-op
    assert_eq!(snowball("virus"), "virus"); // us protected, no-op
    assert_eq!(snowball("cats"), "cat"); // plain s, vowel precedes
    assert_eq!(snowball("gas"), "gas"); // plain s, no vowel before the "as" -> unchanged
}

#[test]
fn snowball_english_exception1_whole_word_table() {
    // The whole-word exception table checked before anything else --
    // some entries remap, some are explicitly left unchanged.
    let cases: &[(&str, &str)] = &[
        ("skis", "ski"),
        ("skies", "sky"),
        ("dying", "die"),
        ("lying", "lie"),
        ("tying", "tie"),
        ("idly", "idl"),
        ("gently", "gentl"),
        ("ugly", "ugli"),
        ("early", "earli"),
        ("only", "onli"),
        ("singly", "singl"),
        ("andes", "andes"),
        ("atlas", "atlas"),
        ("bias", "bias"),
        ("cosmos", "cosmos"),
        ("howe", "howe"),
        ("news", "news"),
        ("sky", "sky"),
    ];
    for (input, expected) in cases {
        assert_eq!(snowball(input), *expected, "stemming {input:?}");
    }
}

#[test]
fn snowball_english_exception2_protected_stems() {
    // The second whole-word protected-stem table -- checked after
    // step 1a but before step 1b -- covers both the `-eed` family
    // (succeed/proceed/exceed) and the `-ing`-as-not-a-gerund family
    // (canning/inning/earring/herring/outing); all eight are left
    // completely unchanged.
    for word in [
        "succeed", "proceed", "exceed", "canning", "inning", "earring", "herring", "outing",
    ] {
        assert_eq!(snowball(word), word, "stemming {word:?}");
    }
}

#[test]
fn snowball_english_double_consonant_aeo_exception() {
    // a3_cleanup's narrow carve-out: a doubled consonant immediately
    // preceded by a/e/o is left alone when that's the *entire* stem
    // (three letters total) -- "added" -> stem "add" (3 letters,
    // starts with "a") keeps both d's, unlike "hopping"/"tanned" (the
    // ordinary double-consonant-drop case, already covered by the
    // cross-engine fixture).
    assert_eq!(snowball("added"), "add");
    assert_eq!(snowball("hopping"), "hop");
    assert_eq!(snowball("tanned"), "tan");
}

#[test]
fn snowball_english_short_word_append_e_fallback() {
    // a3_cleanup's "" fallback: append `e` back if the word is now
    // both short (R1 empty) and ends in a short syllable -- "hoped"
    // (stem "hop" after "-ed" deletion, itself CVC and exactly at R1's
    // start) restores the `e` to "hope"; "owed" (stem "ow", the
    // whole-word-is-vowel-then-consonant short-syllable case) restores
    // to "owe".
    assert_eq!(snowball("hoped"), "hope");
    assert_eq!(snowball("owed"), "owe");
}

#[test]
fn snowball_english_eed_family_ee_replacement() {
    // "-eed"/"-eedly", R1-gated, unconditionally replaced with "ee"
    // (the succ/proc/exc protection is handled globally by the
    // exception2 table, see the dedicated test above).
    assert_eq!(snowball("agreed"), "agre");
    assert_eq!(snowball("feed"), "feed"); // m(fe) == 0 at R1 -> R1 not yet reached, unchanged
}

#[test]
fn snowball_english_ed_ing_edly_ingly_shared_cleanup() {
    assert_eq!(snowball("motoring"), "motor"); // plain -ing deletion + cleanup
    assert_eq!(snowball("plastered"), "plaster"); // plain -ed deletion + cleanup
    assert_eq!(snowball("sing"), "sing"); // no vowel before "-ing" -> unchanged
    assert_eq!(snowball("bled"), "bled"); // no vowel before "-ed" -> unchanged
}

#[test]
fn snowball_english_step1c_trailing_y() {
    assert_eq!(snowball("cry"), "cri"); // y preceded by consonant -> i
    assert_eq!(snowball("toy"), "toy"); // y preceded by vowel -> unchanged
}

#[test]
fn snowball_english_step2_suffix_families() {
    assert_eq!(snowball("carelessly"), "careless"); // lessli -> less
                                                    // fulli -> ful (step2), then step3's own "ful" entry deletes
                                                    // it entirely (same R1 boundary reached again) -> "care".
    assert_eq!(snowball("carefully"), "care");
    assert_eq!(snowball("analogy"), "analog"); // "logi" (preceded by l) -> "log"
                                               // bli (not abli) -> ble (step2), then step5 deletes the
                                               // trailing "e" it just added (R1 holds, R2 doesn't, and the
                                               // resulting stem is not a short syllable) -> "trembl".
    assert_eq!(snowball("trembly"), "trembl");
    assert_eq!(snowball("national"), "nation"); // tional -> tion
}

#[test]
fn snowball_english_step3_and_step4_and_step5() {
    assert_eq!(snowball("triplicate"), "triplic"); // step3 icate -> ic
    assert_eq!(snowball("hopefulness"), "hope"); // step2 fulness -> ful, then step3 ful -> deleted
    assert_eq!(snowball("controll"), "control"); // step5 ll -> l (R2)
    assert_eq!(snowball("roll"), "roll"); // step5 ll unchanged, R2 not reached
    assert_eq!(snowball("rate"), "rate"); // step5 e kept: R1 holds, but ends in short syllable
    assert_eq!(snowball("probate"), "probat"); // step5 e dropped: R2 holds
}

#[test]
fn snowball_english_eedly_edly_ingly_variants() {
    // "eedly" (eed-family, replaced with "ee", then step5 deletes the
    // trailing "e" it left since R1 holds but R2 doesn't and the
    // result isn't a short syllable) -- "edly"/"ingly" (shared
    // delete+cleanup family, same as plain "ed"/"ing").
    assert_eq!(snowball("agreedly"), "agre");
    assert_eq!(snowball("reportedly"), "report");
    assert_eq!(snowball("lastingly"), "last");
}

#[test]
fn snowball_english_leading_y_marked_as_consonant() {
    // prelude marks a word-initial "y" as the internal 'Y' consonant
    // marker; postlude reverts it back to lowercase "y" in the output
    // regardless of whether any step touched it.
    assert_eq!(snowball("yellow"), "yellow");
}

#[test]
fn snowball_english_short_word_passes_through_unchanged() {
    // Below the 3-character minimum, per the algorithm's own domain.
    assert_eq!(snowball("at"), "at");
}

#[test]
fn snowball_english_non_domain_terms_pass_through_unchanged() {
    // Digits and non-ASCII letters are outside every grouping: Java's
    // EnglishStemmer leaves them verbatim. An upper-case head is not:
    // only the suffix is tested, so "Running" -> "Run", as in Java.
    for term in ["3.14", "café", "", "42"] {
        assert_eq!(snowball(term), term, "term {term:?}");
    }
    assert_eq!(snowball("Running"), "Run");
}
