#![cfg(test)]
use std::sync::atomic::{AtomicUsize, Ordering};

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::{AnalyzerDefinition, TokenStreamComponents, TokenizerInput};

use super::*;
use crate::analysis::mock_analyzer::{MockAnalyzer, MockSynonymAnalyzer};

/// One scripted token.
#[derive(Clone)]
struct T {
    term: &'static str,
    start: i32,
    end: i32,
    inc: i32,
    len: i32,
    ty: &'static str,
    keyword: bool,
    flags: i32,
    boost: f32,
    payload: Option<Vec<u8>>,
}

fn t(term: &'static str, start: i32, end: i32, inc: i32, len: i32) -> T {
    T {
        term,
        start,
        end,
        inc,
        len,
        ty: "word",
        keyword: false,
        flags: 0,
        boost: 1.0,
        payload: None,
    }
}

/// What `incrementToken` does before `reset`.
#[derive(Clone, Copy)]
enum NoReset {
    Done,
    Token,
    WrongError,
    Contract,
}

/// A tokenizer replaying `toks`, with switches for each contract
/// breach the assertions look for.
struct Scripted {
    atts: AttributeSource,
    input: TokenizerInput,
    toks: Vec<T>,
    upto: usize,
    clear: bool,
    end_clear: bool,
    final_offset: i32,
    reset_called: bool,
    no_reset: NoReset,
    /// Resets so far, shared across the analyzer's streams.
    resets: Arc<AtomicUsize>,
    /// From this many resets on, `incrementToken` fails.
    fail_from_reset: usize,
    /// Suffix the term with the reset count (an inconsistent analyzer).
    count_terms: bool,
}

fn scripted(toks: Vec<T>, final_offset: i32) -> Scripted {
    Scripted {
        atts: AttributeSource::new(),
        input: TokenizerInput::new(),
        toks,
        upto: 0,
        clear: true,
        end_clear: true,
        final_offset,
        reset_called: false,
        no_reset: NoReset::Done,
        resets: Arc::new(AtomicUsize::new(0)),
        fail_from_reset: usize::MAX,
        count_terms: false,
    }
}

impl TokenStream for Scripted {
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if !self.reset_called {
            match self.no_reset {
                NoReset::Done => return Ok(false),
                NoReset::Contract => return Err(AnalysisError::IllegalState("no".into())),
                NoReset::WrongError => return Err(AnalysisError::Io("odd".into())),
                NoReset::Token => {}
            }
        }
        if self.resets.load(Ordering::SeqCst) >= self.fail_from_reset {
            return Err(AnalysisError::Io("scripted failure".into()));
        }
        let Some(tok) = self.toks.get(self.upto).cloned() else {
            return Ok(false);
        };
        self.upto += 1;
        let a = &mut self.atts;
        if self.clear {
            a.clear_attributes();
        }
        if self.count_terms {
            a.set_term(&format!(
                "{}{}",
                tok.term,
                self.resets.load(Ordering::SeqCst)
            ));
        } else {
            a.set_term(tok.term);
        }
        a.set_offset(tok.start, tok.end)?;
        a.set_position_increment(tok.inc)?;
        a.set_position_length(tok.len)?;
        a.set_token_type(tok.ty);
        a.set_keyword(tok.keyword);
        a.set_flags(tok.flags);
        a.set_boost(tok.boost);
        a.set_payload(tok.payload);
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.upto = 0;
        self.reset_called = true;
        self.resets.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        if self.end_clear {
            self.atts.end_attributes();
        }
        self.atts.set_offset(self.final_offset, self.final_offset)
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.reset_called = false;
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }

    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
}

impl Tokenizer for Scripted {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}

/// An analyzer whose every stream is `make()`.
struct Def(Box<dyn Fn() -> Scripted + Send + Sync>);

impl AnalyzerDefinition for Def {
    fn create_components(&self, _: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new((self.0)()))
    }
}

fn analyzer(make: impl Fn() -> Scripted + Send + Sync + 'static) -> Analyzer {
    Analyzer::new(Def(Box::new(make)))
}

fn rich() -> Vec<T> {
    vec![
        T {
            ty: "alpha",
            keyword: true,
            flags: 3,
            boost: 2.0,
            payload: Some(vec![1]),
            ..t("a", 0, 1, 1, 1)
        },
        T {
            payload: Some(vec![2]),
            ..t("b", 2, 3, 1, 1)
        },
    ]
}

fn rich_expected() -> Expected {
    Expected::terms(&["a", "b"])
        .offsets(&[0, 2], &[1, 3])
        .types(&["alpha", "word"])
        .pos_incs(&[1, 1])
        .pos_lens(&[1, 1])
        .final_offset(3)
        .final_pos_inc(0)
        .keywords(&[true, false])
        .graph_offsets(true)
        .payloads(vec![Some(vec![1]), None])
        .flags(&[3, 0])
        .boosts(&[2.0, 1.0])
}

fn run(toks: Vec<T>, e: Expected) {
    let mut s = scripted(toks, 3);
    assert_token_stream_contents(&mut s, &e).unwrap();
}

fn run_rich(edit: impl FnOnce(&mut Expected)) {
    let mut e = rich_expected();
    edit(&mut e);
    run(rich(), e);
}

#[test]
fn every_attribute_checked_and_matching() {
    run_rich(|_| {});
}

#[test]
#[should_panic(expected = "term 0")]
fn wrong_term() {
    run_rich(|e| e.output[0] = "x".into());
}

#[test]
#[should_panic(expected = "startOffset 0")]
fn wrong_start_offset() {
    run_rich(|e| e.start_offsets = Some(vec![1, 2]));
}

#[test]
#[should_panic(expected = "endOffset 1")]
fn wrong_end_offset() {
    run_rich(|e| e.end_offsets = Some(vec![1, 2]));
}

#[test]
#[should_panic(expected = "type 1")]
fn wrong_type() {
    run_rich(|e| e.types = Some(vec!["alpha".into(), "x".into()]));
}

#[test]
#[should_panic(expected = "posIncrement 0")]
fn wrong_position_increment() {
    run_rich(|e| e.pos_increments = Some(vec![2, 1]));
}

#[test]
#[should_panic(expected = "posLength 0")]
fn wrong_position_length() {
    run_rich(|e| e.pos_lengths = Some(vec![2, 1]));
}

#[test]
#[should_panic(expected = "keywordAtt 0")]
fn wrong_keyword() {
    run_rich(|e| e.keyword_atts = Some(vec![false, false]));
}

#[test]
#[should_panic(expected = "flagsAtt 0")]
fn wrong_flags() {
    run_rich(|e| e.flags = Some(vec![1, 0]));
}

#[test]
#[should_panic(expected = "boostAtt 0")]
fn wrong_boost() {
    run_rich(|e| e.boost = Some(vec![1.0, 1.0]));
}

#[test]
#[should_panic(expected = "payloads 0")]
fn wrong_payload() {
    run_rich(|e| e.payloads = Some(vec![Some(vec![9]), None]));
}

#[test]
#[should_panic(expected = "finalOffset")]
fn wrong_final_offset() {
    run_rich(|e| e.final_offset = Some(4));
}

#[test]
#[should_panic(expected = "finalPosInc")]
fn wrong_final_position_increment() {
    run_rich(|e| e.final_pos_inc = Some(1));
}

#[test]
#[should_panic(expected = "first posIncrement must be >= 1")]
fn first_increment_zero() {
    run(
        vec![t("a", 0, 1, 0, 1)],
        Expected::terms(&["a"]).pos_incs(&[0]),
    );
}

#[test]
#[should_panic(expected = "startOffset (= 5) must be <= finalOffset (= 3)")]
fn start_past_final_offset() {
    run(
        vec![t("a", 5, 6, 1, 1)],
        Expected::terms(&["a"]).final_offset(3),
    );
}

#[test]
#[should_panic(expected = "endOffset must be <= finalOffset")]
fn end_past_final_offset() {
    run(
        vec![t("a", 1, 5, 1, 1)],
        Expected::terms(&["a"]).final_offset(3),
    );
}

#[test]
#[should_panic(expected = "offsets must not go backwards")]
fn offsets_going_backwards() {
    run(
        vec![t("a", 2, 3, 1, 1), t("b", 0, 1, 1, 1)],
        Expected::terms(&["a", "b"]).offsets(&[2, 0], &[3, 1]),
    );
}

#[test]
#[should_panic(expected = "inconsistent startOffset")]
fn inconsistent_start_offsets() {
    run(
        vec![t("a", 0, 1, 1, 1), t("b", 1, 2, 0, 1)],
        Expected::terms(&["a", "b"])
            .offsets(&[0, 1], &[1, 2])
            .pos_incs(&[1, 0])
            .pos_lens(&[1, 1]),
    );
}

#[test]
#[should_panic(expected = "inconsistent endOffset")]
fn inconsistent_end_offsets() {
    run(
        vec![t("a", 0, 2, 1, 2), t("b", 1, 3, 1, 1)],
        Expected::terms(&["a", "b"])
            .offsets(&[0, 1], &[2, 3])
            .pos_incs(&[1, 1])
            .pos_lens(&[2, 1]),
    );
}

#[test]
fn inconsistent_offsets_pass_when_the_graph_is_known_wrong() {
    run(
        vec![t("a", 0, 1, 1, 1), t("b", 1, 2, 0, 1)],
        Expected::terms(&["a", "b"])
            .offsets(&[0, 1], &[1, 2])
            .pos_incs(&[1, 0])
            .pos_lens(&[1, 1])
            .graph_offsets(false),
    );
}

#[test]
#[should_panic(expected = "more tokens than expected (expected count=1); extra token=b")]
fn extra_token() {
    run(rich(), Expected::terms(&["a"]));
}

#[test]
#[should_panic(expected = "token 2 does not exist")]
fn missing_token() {
    run(rich(), Expected::terms(&["a", "b", "c"]));
}

#[test]
#[should_panic(
    expected = "clearAttributes() was not called correctly in TokenStream chain at token 0"
)]
fn token_without_clear() {
    let mut s = scripted(rich(), 3);
    s.clear = false;
    assert_token_stream_contents(&mut s, &Expected::terms(&["a", "b"])).unwrap();
}

#[test]
#[should_panic(expected = "super.end()/clearAttributes() was not called correctly in end()")]
fn end_without_clear() {
    let mut s = scripted(rich(), 3);
    s.end_clear = false;
    assert_token_stream_contents(&mut s, &Expected::terms(&["a", "b"])).unwrap();
}

#[test]
fn reset_exception_accepts_false_or_illegal_state() {
    for mode in [NoReset::Done, NoReset::Contract] {
        let a = analyzer(move || Scripted {
            no_reset: mode,
            ..scripted(vec![t("a", 0, 1, 1, 1)], 1)
        });
        check_reset_exception(&a, "a").unwrap();
    }
}

#[test]
#[should_panic(expected = "didn't get expected exception when reset() not called")]
fn reset_exception_refuses_a_token() {
    let a = analyzer(|| Scripted {
        no_reset: NoReset::Token,
        ..scripted(vec![t("a", 0, 1, 1, 1)], 1)
    });
    check_reset_exception(&a, "a").unwrap();
}

#[test]
#[should_panic(expected = "got wrong exception when reset() not called")]
fn reset_exception_refuses_another_error() {
    let a = analyzer(|| Scripted {
        no_reset: NoReset::WrongError,
        ..scripted(vec![t("a", 0, 1, 1, 1)], 1)
    });
    check_reset_exception(&a, "a").unwrap();
}

#[test]
fn attribute_factory_draws_like_java() {
    let mut r = JavaRandom::new(42);
    let mut twin = JavaRandom::new(42);
    let mut seen = HashSet::new();
    for _ in 0..30 {
        let kind = new_attribute_factory(&mut r);
        let expected = match twin.next_int_bounded(3) {
            0 => AttributeFactoryKind::TokenStreamDefault,
            1 => AttributeFactoryKind::Token,
            _ => AttributeFactoryKind::Default,
        };
        assert_eq!(kind, expected);
        seen.insert(format!("{kind:?}"));
    }
    assert_eq!(seen.len(), 3);
}

#[test]
fn number_of_tokens() {
    assert_stream_has_number_of_tokens(&mut scripted(rich(), 3), 2).unwrap();
}

#[test]
#[should_panic(expected = "wrong number of tokens")]
fn wrong_number_of_tokens() {
    assert_stream_has_number_of_tokens(&mut scripted(rich(), 3), 3).unwrap();
}

#[test]
fn random_data_returns_the_analyzers_error() {
    let resets = Arc::new(AtomicUsize::new(0));
    let a = analyzer(move || Scripted {
        resets: Arc::clone(&resets),
        // check_reset_exception's reset passes; the first random
        // string's fails
        fail_from_reset: 2,
        ..scripted(vec![t("a", 0, 0, 1, 1)], 0)
    });
    let err = check_random_data(&mut JavaRandom::new(1), &a, 3).unwrap_err();
    assert_eq!(err, AnalysisError::Io("scripted failure".into()));
}

#[test]
#[should_panic(expected = "term 0")]
fn random_data_catches_an_inconsistent_analyzer() {
    let resets = Arc::new(AtomicUsize::new(0));
    let a = analyzer(move || Scripted {
        resets: Arc::clone(&resets),
        count_terms: true,
        ..scripted(vec![t("a", 0, 0, 1, 1)], 0)
    });
    check_random_data_with(&mut JavaRandom::new(1), &a, 5, 20, true, true).unwrap();
}

/// Java's inner `checkRandomData`, replayed draw by draw with an index
/// writer present: the field type's draws first, then per text the
/// string, the consistency check's draws and `nextInt(7)`.
fn replay_indexing(seed: i64, a: &Analyzer, iterations: i32) -> JavaRandom {
    let mut r = JavaRandom::new(seed);
    let term_vectors = r.next_boolean();
    if term_vectors {
        let _offsets = r.next_boolean();
        let positions = r.next_boolean();
        if positions {
            let _payloads = r.next_boolean();
        }
    }
    let _omit_norms = r.next_boolean();
    let _index_options = r.next_int_bounded(4);
    for _ in 0..iterations {
        let text = test_util::random_analysis_string(&mut r, 20, false);
        check_analysis_consistency(&mut r, a, false, &text, true).unwrap();
        let _pile_up = r.next_int_bounded(7);
    }
    r
}

#[test]
fn random_data_makes_the_index_writers_draws() {
    let a = MockAnalyzer::new(&mut JavaRandom::new(4)).build();
    for seed in 0..8 {
        let run = |index: bool| {
            let mut r = JavaRandom::new(seed);
            let c = RandomRun {
                iterations: 5,
                max_word_length: 20,
                use_char_filter: seed % 2 == 1,
                simple: false,
                graph_offsets_are_correct: true,
            };
            let tmp = TempDir::new("bttc-draws");
            let dir = FsDirectory::open(tmp.path());
            let iw = index.then(|| Mutex::new(random_index_writer(seed, &a, &dir)));
            check_random_data_inner(&mut r, &a, &c, iw.as_ref()).unwrap();
            if let Some(iw) = iw {
                close_and_check(iw, &dir);
            }
            r.next_long()
        };
        let indexed = run(true);
        assert_eq!(
            indexed,
            replay_indexing(seed, &a, 5).next_long(),
            "seed {seed}"
        );
        assert_ne!(indexed, run(false), "seed {seed}: the draws were skipped");
    }
}

#[test]
fn analysis_consistency_survives_both_evil_branches() {
    // evilness is nextInt(50) after nextInt(10): enough seeds reach 17
    // (a failing reader) and 7 (a consumer that stops early)
    let a = MockAnalyzer::new(&mut JavaRandom::new(3)).build();
    let (mut evil_reader, mut early_stop) = (0, 0);
    for seed in 0..300 {
        let mut probe = JavaRandom::new(seed);
        probe.next_int_bounded(10);
        match probe.next_int_bounded(50) {
            17 => evil_reader += 1,
            7 => early_stop += 1,
            _ => {}
        }
        let mut r = JavaRandom::new(seed);
        check_analysis_consistency(&mut r, &a, seed % 2 == 0, "some words to read", true).unwrap();
    }
    assert!(
        evil_reader > 0 && early_stop > 0,
        "{evil_reader} {early_stop}"
    );
}

#[test]
fn escape_is_javas() {
    assert_eq!(escape("a\n\r\"\\\u{e9}~"), "a\\n\\r\\\"\\\\\\u00e9~");
    assert_eq!(escape("\u{1f600}"), "\\ud83d\\ude00");
}

#[test]
fn mock_tokenizer_helpers() {
    let e = Expected::terms(&["Ab", "c"]).offsets(&[0, 3], &[2, 4]);
    assert_token_stream_contents(&mut whitespace_mock_tokenizer("Ab c").unwrap(), &e).unwrap();
    assert_token_stream_contents(
        &mut whitespace_mock_tokenizer_from(Box::new(StrReader::new("Ab c"))).unwrap(),
        &e,
    )
    .unwrap();
    let k = Expected::terms(&["Ab c"]);
    assert_token_stream_contents(&mut keyword_mock_tokenizer("Ab c").unwrap(), &k).unwrap();
    assert_token_stream_contents(
        &mut keyword_mock_tokenizer_from(Box::new(StrReader::new("Ab c"))).unwrap(),
        &k,
    )
    .unwrap();
}

#[test]
fn graph_strings() {
    let a = MockSynonymAnalyzer::analyzer();
    let want: HashSet<String> = ["guinea pig", "cavy"]
        .iter()
        .map(|s| s.to_string())
        .collect();
    assert_eq!(get_graph_strings_of(&a, "guinea pig").unwrap(), want);
    assert_graph_strings_of(
        &mut JavaRandom::new(5),
        &a,
        "guinea pig",
        &["cavy", "guinea pig"],
    )
    .unwrap();
}

#[test]
#[should_panic(expected = "Analyzer created unexpected string path: ")]
fn graph_strings_mismatch() {
    let a = MockSynonymAnalyzer::analyzer();
    assert_graph_strings(
        &mut a.token_stream("f", "guinea pig").unwrap(),
        &["guinea pig"],
    )
    .unwrap();
}

#[test]
fn analyzer_to_string() {
    let a = MockSynonymAnalyzer::analyzer();
    assert_eq!(
        to_string(&a, "guinea pig").unwrap(),
        "guinea at pos=0 to pos=1 offsets=0-6\n\
             cavy at pos=0 to pos=2 offsets=0-10\n\
             pig at pos=1 to pos=2 offsets=7-10\n"
    );
}

#[test]
fn check_one_term_and_the_clear_attribute() {
    let a = MockAnalyzer::new(&mut JavaRandom::new(1)).build();
    check_one_term(&mut JavaRandom::new(2), &a, "Word", "word").unwrap();
    let mut seen = Vec::new();
    let mut c = CheckClearAttributes::default();
    c.clear();
    c.reflect(&mut |class, key, value| seen.push(format!("{class}#{key}={value}")));
    assert_eq!(
            seen,
            ["org.apache.lucene.tests.analysis.BaseTokenStreamTestCase$CheckClearAttributesAttribute#clearCalled=true"]
        );
    assert!(c.get_and_reset_clear_called());
    assert!(!c.get_and_reset_clear_called());
    assert!(c.clone().get_and_reset_clear_called());
}
