//! Lucene's `TestMockAnalyzer` (test-framework's own tests), ported.
//!
//! `testChangeGaps` (LUCENE-5324) is not here: it is entirely an indexing
//! test -- a wrapper's gaps must show in indexed term-vector positions and
//! offsets -- and waits for the port's `RandomIndexWriter` (h4).

use std::sync::Arc;

use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, AnalyzerWrapper, CharReader, StrReader,
    TokenStream, TokenStreamComponents,
};
use lucene_test_framework::analysis::mock_filters::english_stopset;
use lucene_test_framework::analysis::mock_tokenizer::{self, MockTokenizer};
use lucene_test_framework::analysis::{
    assert_analyzes_to, assert_token_stream_contents, check_one_term, check_random_data, Expected,
    MockAnalyzer, MockCharFilter,
};
use lucene_test_framework::util::automaton_test_util::random_automaton;
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_test_framework::util::test_util;
use lucene_util::automaton::operations::{
    complement, determinize, union, DEFAULT_DETERMINIZE_WORK_LIMIT,
};
use lucene_util::automaton::{automata, Automaton, CharacterRunAutomaton, RegExp};

fn run_automaton(a: &Automaton) -> Arc<CharacterRunAutomaton> {
    let d = determinize(a, DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    Arc::new(CharacterRunAutomaton::new(&d).unwrap())
}

fn regexp(s: &str) -> Arc<CharacterRunAutomaton> {
    run_automaton(&RegExp::new(s).unwrap().to_automaton().unwrap())
}

/// An analyzer of one bare `MockTokenizer` (Java's anonymous
/// `Analyzer`s in these tests).
struct TokenizerOnly {
    dfa: Arc<CharacterRunAutomaton>,
    lower_case: bool,
    max_token_length: i32,
}

impl AnalyzerDefinition for TokenizerOnly {
    fn create_components(&self, _: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(
            MockTokenizer::with_max_token_length(
                Arc::clone(&self.dfa),
                self.lower_case,
                self.max_token_length,
            ),
        ))
    }
}

fn terms(output: &[&str]) -> Expected {
    Expected::terms(output)
}

/// Test a configuration that behaves a lot like WhitespaceAnalyzer.
#[test]
fn test_whitespace() {
    let mut t = LuceneTestCase::new("test_whitespace");
    let r = t.random();
    let a = MockAnalyzer::new(r).build();
    assert_analyzes_to(
        r,
        &a,
        "A bc defg hiJklmn opqrstuv wxy z ",
        &terms(&["a", "bc", "defg", "hijklmn", "opqrstuv", "wxy", "z"]),
    )
    .unwrap();
    assert_analyzes_to(
        r,
        &a,
        "aba cadaba shazam",
        &terms(&["aba", "cadaba", "shazam"]),
    )
    .unwrap();
    assert_analyzes_to(
        r,
        &a,
        "break on whitespace",
        &terms(&["break", "on", "whitespace"]),
    )
    .unwrap();
}

/// Test a configuration that behaves a lot like SimpleAnalyzer.
#[test]
fn test_simple() {
    let mut t = LuceneTestCase::new("test_simple");
    let r = t.random();
    let a = MockAnalyzer::with_tokenizer(r, mock_tokenizer::simple(), true).build();
    assert_analyzes_to(
        r,
        &a,
        "a-bc123 defg+hijklmn567opqrstuv78wxy_z ",
        &terms(&["a", "bc", "defg", "hijklmn", "opqrstuv", "wxy", "z"]),
    )
    .unwrap();
    assert_analyzes_to(
        r,
        &a,
        "aba4cadaba-Shazam",
        &terms(&["aba", "cadaba", "shazam"]),
    )
    .unwrap();
    assert_analyzes_to(
        r,
        &a,
        "break+on/Letters",
        &terms(&["break", "on", "letters"]),
    )
    .unwrap();
}

/// Test a configuration that behaves a lot like KeywordAnalyzer.
#[test]
fn test_keyword() {
    let mut t = LuceneTestCase::new("test_keyword");
    let r = t.random();
    let a = MockAnalyzer::with_tokenizer(r, mock_tokenizer::keyword(), false).build();
    assert_analyzes_to(
        r,
        &a,
        "a-bc123 defg+hijklmn567opqrstuv78wxy_z ",
        &terms(&["a-bc123 defg+hijklmn567opqrstuv78wxy_z "]),
    )
    .unwrap();
    assert_analyzes_to(r, &a, "aba4cadaba-Shazam", &terms(&["aba4cadaba-Shazam"])).unwrap();
    assert_analyzes_to(r, &a, "break+on/Nothing", &terms(&["break+on/Nothing"])).unwrap();
    // currently though emits no tokens for empty string
    assert_analyzes_to(r, &a, "", &terms(&[])).unwrap();
}

/// Test a configuration where each character is a term.
#[test]
fn test_single_char() {
    let mut t = LuceneTestCase::new("test_single_char");
    let r = t.random();
    let a = MockAnalyzer::with_tokenizer(r, regexp("."), false).build();
    assert_analyzes_to(
        r,
        &a,
        "foobar",
        &terms(&["f", "o", "o", "b", "a", "r"]).offsets(&[0, 1, 2, 3, 4, 5], &[1, 2, 3, 4, 5, 6]),
    )
    .unwrap();
    check_random_data(r, &a, 100).unwrap();
}

/// Test a configuration where two characters makes a term.
#[test]
fn test_two_chars() {
    let mut t = LuceneTestCase::new("test_two_chars");
    let r = t.random();
    let a = MockAnalyzer::with_tokenizer(r, regexp(".."), false).build();
    assert_analyzes_to(
        r,
        &a,
        "foobar",
        &terms(&["fo", "ob", "ar"]).offsets(&[0, 2, 4], &[2, 4, 6]),
    )
    .unwrap();
    // make sure when last term is a "partial" match that end() is correct
    assert_token_stream_contents(
        &mut a.token_stream("bogus", "fooba").unwrap(),
        &terms(&["fo", "ob"])
            .offsets(&[0, 2], &[2, 4])
            .pos_incs(&[1, 1])
            .final_offset(5),
    )
    .unwrap();
    check_random_data(r, &a, 100).unwrap();
}

/// Test a configuration where three characters makes a term.
#[test]
fn test_three_chars() {
    let mut t = LuceneTestCase::new("test_three_chars");
    let r = t.random();
    let a = MockAnalyzer::with_tokenizer(r, regexp("..."), false).build();
    assert_analyzes_to(
        r,
        &a,
        "foobar",
        &terms(&["foo", "bar"]).offsets(&[0, 3], &[3, 6]),
    )
    .unwrap();
    assert_token_stream_contents(
        &mut a.token_stream("bogus", "fooba").unwrap(),
        &terms(&["foo"])
            .offsets(&[0], &[3])
            .pos_incs(&[1])
            .final_offset(5),
    )
    .unwrap();
    check_random_data(r, &a, 100).unwrap();
}

/// Test a configuration where word starts with one uppercase.
#[test]
fn test_uppercase() {
    let mut t = LuceneTestCase::new("test_uppercase");
    let r = t.random();
    let a = MockAnalyzer::with_tokenizer(r, regexp("[A-Z][a-z]*"), false).build();
    assert_analyzes_to(
        r,
        &a,
        "FooBarBAZ",
        &terms(&["Foo", "Bar", "B", "A", "Z"]).offsets(&[0, 3, 6, 7, 8], &[3, 6, 7, 8, 9]),
    )
    .unwrap();
    assert_analyzes_to(
        r,
        &a,
        "aFooBar",
        &terms(&["Foo", "Bar"]).offsets(&[1, 4], &[4, 7]),
    )
    .unwrap();
    check_random_data(r, &a, 100).unwrap();
}

/// Test a configuration that behaves a lot like StopAnalyzer.
#[test]
fn test_stop() {
    let mut t = LuceneTestCase::new("test_stop");
    let r = t.random();
    let a =
        MockAnalyzer::with_automaton(r, mock_tokenizer::simple(), true, english_stopset()).build();
    assert_analyzes_to(
        r,
        &a,
        "the quick brown a fox",
        &terms(&["quick", "brown", "fox"]).pos_incs(&[2, 1, 2]),
    )
    .unwrap();
}

/// Test a configuration that behaves a lot like KeepWordFilter.
#[test]
fn test_keep() {
    let mut t = LuceneTestCase::new("test_keep");
    let r = t.random();
    let foo = automata::make_string("foo");
    let bar = automata::make_string("bar");
    let keep = complement(&union(&[&foo, &bar]), DEFAULT_DETERMINIZE_WORK_LIMIT).unwrap();
    let a = MockAnalyzer::with_automaton(r, mock_tokenizer::simple(), true, run_automaton(&keep))
        .build();
    assert_analyzes_to(
        r,
        &a,
        "quick foo brown bar bar fox foo",
        &terms(&["foo", "bar", "bar", "foo"]).pos_incs(&[2, 2, 1, 2]),
    )
    .unwrap();
}

/// Test a configuration that behaves a lot like LengthFilter.
#[test]
fn test_length() {
    let mut t = LuceneTestCase::new("test_length");
    let r = t.random();
    let a = MockAnalyzer::with_automaton(r, mock_tokenizer::whitespace(), true, regexp(".{5,}"))
        .build();
    assert_analyzes_to(
        r,
        &a,
        "ok toolong fine notfine",
        &terms(&["ok", "fine"]).pos_incs(&[1, 2]),
    )
    .unwrap();
}

/// Test MockTokenizer encountering a too long token.
#[test]
fn test_too_long_token() {
    let whitespace = Analyzer::new(TokenizerOnly {
        dfa: mock_tokenizer::whitespace(),
        lower_case: false,
        max_token_length: 5,
    });
    assert_token_stream_contents(
        &mut whitespace
            .token_stream("bogus", "test 123 toolong ok ")
            .unwrap(),
        &terms(&["test", "123", "toolo", "ng", "ok"])
            .offsets(&[0, 5, 9, 14, 17], &[4, 8, 14, 16, 19])
            .final_offset(20),
    )
    .unwrap();
    assert_token_stream_contents(
        &mut whitespace.token_stream("bogus", "test 123 toolo").unwrap(),
        &terms(&["test", "123", "toolo"])
            .offsets(&[0, 5, 9], &[4, 8, 14])
            .final_offset(14),
    )
    .unwrap();
}

#[test]
fn test_lucene_3042() {
    let mut t = LuceneTestCase::new("test_lucene_3042");
    let r = t.random();
    let analyzer = MockAnalyzer::new(r).build();
    {
        let mut stream = analyzer.token_stream("dummy", "t").unwrap();
        stream.reset().unwrap();
        while stream.increment_token().unwrap() {}
        stream.end().unwrap();
    }
    assert_analyzes_to(r, &analyzer, "t", &terms(&["t"])).unwrap();
}

/// blast some random strings through the analyzer
#[test]
fn test_random_strings() {
    let mut t = LuceneTestCase::new("test_random_strings");
    // Java evaluates `new MockAnalyzer(random())` before `atLeast(1000)`
    let a = MockAnalyzer::new(t.random()).build();
    let n = t.at_least(1000);
    check_random_data(t.random(), &a, n).unwrap();
}

/// blast some random strings through differently configured tokenizers
#[test]
fn test_random_regexps() {
    let mut t = LuceneTestCase::new("test_random_regexps");
    let iters = if t.nightly() {
        t.at_least(30)
    } else {
        t.at_least(1)
    };
    for _ in 0..iters {
        let r = t.random();
        let dfa = run_automaton(&random_automaton(r));
        let lower_case = r.next_boolean();
        let limit = test_util::next_int(r, 0, 500);
        let a = Analyzer::new(TokenizerOnly {
            dfa,
            lower_case,
            max_token_length: limit,
        });
        check_random_data(r, &a, 100).unwrap();
        a.close();
    }
}

#[test]
fn test_forward_offsets() {
    let mut t = LuceneTestCase::new("test_forward_offsets");
    let num = t.at_least(1000);
    let r = t.random();
    for _ in 0..num {
        let s = test_util::random_htmlish_string(r, 20);
        let charfilter = MockCharFilter::new(Box::new(StrReader::new(s)), 2).unwrap();
        let analyzer = MockAnalyzer::new(r).build();
        let mut ts = analyzer
            .token_stream_from_reader("bogus", Box::new(charfilter))
            .unwrap();
        ts.reset().unwrap();
        while ts.increment_token().unwrap() {}
        ts.end().unwrap();
    }
}

struct WrapReader(Analyzer);

impl AnalyzerWrapper for WrapReader {
    fn wrapped_analyzer(&self, _: &str) -> &Analyzer {
        &self.0
    }

    fn wrap_reader(&self, _: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(MockCharFilter::new(reader, 7).unwrap())
    }
}

/// LUCENE-5153: test that wrapping an analyzer's reader is allowed.
#[test]
fn test_wrap_reader() {
    let mut t = LuceneTestCase::new("test_wrap_reader");
    let r = t.random();
    let delegate = MockAnalyzer::new(r).build();
    let strategy = delegate.reuse_strategy();
    let a = Analyzer::with_reuse_strategy(WrapReader(delegate), strategy);
    check_one_term(r, &a, "abc", "aabc").unwrap();
}

/// What `MockTokenizer`'s checks catch, through an analyzer: a consumer that
/// skips `reset()` (Java's `checkResetException`) and one that stops early.
#[test]
fn mock_tokenizer_checks_fire_through_an_analyzer() {
    let mut t = LuceneTestCase::new("mock_tokenizer_checks_fire");
    let a = MockAnalyzer::new(t.random()).build();
    let mut ts = a.token_stream("f", "a b").unwrap();
    let e = ts.increment_token().unwrap_err();
    assert!(
        e.to_string()
            .contains("incrementToken() called while in wrong state"),
        "{e}"
    );
    ts.reset().unwrap();
    assert!(ts.increment_token().unwrap());
    let e = ts.end().unwrap_err();
    assert!(
        e.to_string()
            .contains("end() called in wrong state=INCREMENT!"),
        "{e}"
    );
    // a stream is dropped (and so closed) without a panic
    drop(ts);
    let quiet = MockAnalyzer::new(t.random()).enable_checks(false).build();
    let mut ts = quiet.token_stream("f", "a b").unwrap();
    ts.reset().unwrap();
    assert!(ts.increment_token().unwrap());
    ts.end().unwrap();
}

/// The [`TokenStream`] trait objects compose: a MockAnalyzer's stream is
/// usable as `&mut dyn TokenStream`.
#[test]
fn analyzer_streams_are_token_streams() {
    let mut t = LuceneTestCase::new("analyzer_streams_are_token_streams");
    let a = MockAnalyzer::new(t.random()).build();
    let ts: &mut dyn TokenStream = &mut a.token_stream("f", "x").unwrap();
    assert_token_stream_contents(ts, &terms(&["x"]).final_offset(1)).unwrap();
}
