//! Lucene's `TestLookaheadTokenFilter` and its `TrivialLookaheadFilter`,
//! ported.
//!
//! Java's anonymous analyzers draw from the test's `random()` inside
//! `createComponents`; here each analyzer holds its own random, seeded from
//! the test's.

use std::sync::Mutex;

use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, TokenFilter, TokenStream, TokenStreamComponents,
};
use lucene_test_framework::analysis::mock_tokenizer::{self, MockTokenizer};
use lucene_test_framework::analysis::{
    assert_analyzes_to, check_random_data_with, Expected, LookaheadHooks, LookaheadTokenFilter,
    MockRandomLookaheadTokenFilter, NoHooks,
};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;
use lucene_util::java_random::JavaRandom;

/// `IndexWriter.MAX_TERM_LENGTH / 2`.
const MAX_TOKEN_LENGTH: i32 = 32766 / 2;

fn tokenizer(random: &mut JavaRandom) -> MockTokenizer {
    MockTokenizer::with_max_token_length(
        mock_tokenizer::whitespace(),
        random.next_boolean(),
        MAX_TOKEN_LENGTH,
    )
}

struct RandomLookahead(Mutex<JavaRandom>);

impl AnalyzerDefinition for RandomLookahead {
    fn create_components(&self, _: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let mut random = self.0.lock().unwrap();
        let tokenizer = tokenizer(&mut random);
        Ok(TokenStreamComponents::new(
            MockRandomLookaheadTokenFilter::new(&mut random, tokenizer),
        ))
    }
}

fn max_length(t: &LuceneTestCase) -> i32 {
    if t.nightly() {
        8192
    } else {
        1024
    }
}

#[test]
fn test_random_strings() {
    let mut t = LuceneTestCase::new("test_random_strings");
    let max = max_length(&t);
    let iterations = 50 * t.multiplier();
    let r = t.random();
    let a = Analyzer::new(RandomLookahead(Mutex::new(JavaRandom::new(r.next_long()))));
    check_random_data_with(r, &a, iterations, max, false, true).unwrap();
}

/// `NeverPeeksLookaheadTokenFilter`: only ever calls `nextToken`.
struct NeverPeeks<I>(LookaheadTokenFilter<I>);

impl<I: TokenStream> TokenFilter for NeverPeeks<I> {
    type Input = I;

    fn input(&self) -> &I {
        self.0.input()
    }

    fn input_mut(&mut self) -> &mut I {
        self.0.input_mut()
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        self.0.next_token(&mut NoHooks)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.0.reset()
    }
}

struct NeverPeeksAnalyzer(Mutex<JavaRandom>);

impl AnalyzerDefinition for NeverPeeksAnalyzer {
    fn create_components(&self, _: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let tokenizer = tokenizer(&mut self.0.lock().unwrap());
        Ok(TokenStreamComponents::new(NeverPeeks(
            LookaheadTokenFilter::new(tokenizer),
        )))
    }
}

#[test]
fn test_never_calling_peek() {
    let mut t = LuceneTestCase::new("test_never_calling_peek");
    let max = max_length(&t);
    let iterations = 50 * t.multiplier();
    let r = t.random();
    let a = Analyzer::new(NeverPeeksAnalyzer(Mutex::new(JavaRandom::new(
        r.next_long(),
    ))));
    check_random_data_with(r, &a, iterations, max, false, true).unwrap();
}

/// `TrivialLookaheadFilter`: reads a sentence ahead, and after each
/// position inserts that position's "fact" (its term + "-huh?").
struct TrivialLookaheadFilter<I> {
    la: LookaheadTokenFilter<I, Option<String>>,
    hooks: InsertFacts,
}

struct InsertFacts {
    insert_upto: i32,
}

impl<I: TokenStream> LookaheadHooks<I, Option<String>> for InsertFacts {
    fn after_position(
        &mut self,
        la: &mut LookaheadTokenFilter<I, Option<String>>,
    ) -> Result<(), AnalysisError> {
        if self.insert_upto < la.output_pos {
            la.insert_token();
            let out = la.output_pos;
            // Java appends a null fact as "null"
            let fact = la
                .positions
                .get(out)
                .extra
                .clone()
                .unwrap_or_else(|| "null".into());
            let start = la.positions.get(out).start_offset;
            let end = la.positions.get(out + 1).end_offset;
            let atts = la.attributes_mut();
            atts.clear_attributes();
            atts.set_term(&fact);
            atts.set_position_increment(0)?;
            atts.set_offset(start, end)?;
            self.insert_upto = out;
        }
        Ok(())
    }
}

impl<I: TokenStream> TrivialLookaheadFilter<I> {
    fn new(input: I) -> Self {
        TrivialLookaheadFilter {
            la: LookaheadTokenFilter::new(input),
            hooks: InsertFacts { insert_upto: -1 },
        }
    }

    fn peek_sentence(&mut self) -> Result<(), AnalysisError> {
        let mut facts = Vec::new();
        loop {
            if self.la.peek_token()? {
                let term = self.la.input().attributes().term().to_string();
                facts.push(format!("{term}-huh?"));
                if term == "." {
                    break;
                }
            } else {
                break;
            }
        }
        let out = self.la.output_pos;
        for (x, fact) in facts.into_iter().enumerate() {
            self.la.positions.get(out + x as i32).extra = Some(fact);
        }
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for TrivialLookaheadFilter<I> {
    type Input = I;

    fn input(&self) -> &I {
        self.la.input()
    }

    fn input_mut(&mut self) -> &mut I {
        self.la.input_mut()
    }

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.la.positions.max_pos() < self.la.output_pos {
            self.peek_sentence()?;
        }
        self.la.next_token(&mut self.hooks)
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.la.reset()?;
        self.hooks.insert_upto = -1;
        Ok(())
    }
}

struct Trivial;

impl AnalyzerDefinition for Trivial {
    fn create_components(&self, _: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let source = MockTokenizer::new(mock_tokenizer::whitespace(), false);
        Ok(TokenStreamComponents::new(TrivialLookaheadFilter::new(
            source,
        )))
    }
}

#[test]
fn test_missed_first_token() {
    let mut t = LuceneTestCase::new("test_missed_first_token");
    let analyzer = Analyzer::new(Trivial);
    assert_analyzes_to(
        t.random(),
        &analyzer,
        "Only he who is running knows .",
        &Expected::terms(&[
            "Only",
            "Only-huh?",
            "he",
            "he-huh?",
            "who",
            "who-huh?",
            "is",
            "is-huh?",
            "running",
            "running-huh?",
            "knows",
            "knows-huh?",
            ".",
            ".-huh?",
        ]),
    )
    .unwrap();
}
