//! `MockAnalyzer`, `MockPayloadAnalyzer` and `MockSynonymAnalyzer`.
//!
//! Each is an [`AnalyzerDefinition`]; [`MockAnalyzer::build`] (and the
//! others' `analyzer()`) wrap it in an [`Analyzer`] with Java's reuse
//! strategy. Java's `MockAnalyzer` setters (`setEnableChecks`,
//! `setMaxTokenLength`, the gaps) are builder methods here, applied before
//! the analyzer is built -- every Java caller sets them before the first
//! `tokenStream`.

use std::collections::HashMap;
use std::sync::{Arc, Mutex};

use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, ReuseStrategy, TokenStream, TokenStreamComponents,
};
use lucene_util::automaton::CharacterRunAutomaton;
use lucene_util::java_random::JavaRandom;

use super::mock_filters::{
    empty_stopset, lock, MockFixedLengthPayloadFilter, MockLowerCaseFilter, MockPayloadFilter,
    MockTokenFilter, MockVariableLengthPayloadFilter, SharedRandom, TermFreqFn,
};
use super::mock_synonym::MockSynonymFilter;
use super::mock_tokenizer::{self, MockTokenizer, DEFAULT_MAX_TOKEN_LENGTH};
use crate::util::lucene_test_case;

/// `org.apache.lucene.tests.analysis.MockAnalyzer`: a [`MockTokenizer`]
/// and a [`MockTokenFilter`], plus (rarely, per field) random payloads.
pub struct MockAnalyzer {
    run_automaton: Arc<CharacterRunAutomaton>,
    lower_case: bool,
    filter: Arc<CharacterRunAutomaton>,
    custom_term_freq: Option<TermFreqFn>,
    position_increment_gap: i32,
    offset_gap: Option<i32>,
    random: SharedRandom,
    /// The payload choice made for each field, so a field stays consistent.
    previous_mappings: Mutex<HashMap<String, i32>>,
    enable_checks: bool,
    max_token_length: i32,
    nightly: bool,
    multiplier: i32,
}

impl MockAnalyzer {
    /// `MockAnalyzer(random)`: whitespace tokenization, lowercased, no stop
    /// words. Draws its own seed from `random`.
    pub fn new(random: &mut JavaRandom) -> Self {
        Self::with_automaton(random, mock_tokenizer::whitespace(), true, empty_stopset())
    }

    /// `MockAnalyzer(random, runAutomaton, lowerCase, filter)`.
    pub fn with_automaton(
        random: &mut JavaRandom,
        run_automaton: Arc<CharacterRunAutomaton>,
        lower_case: bool,
        filter: Arc<CharacterRunAutomaton>,
    ) -> Self {
        // `maybePayload`'s `rarely` sees the run's TESTS_NIGHTLY and
        // TESTS_MULTIPLIER, as Java's sees `TEST_NIGHTLY`/`RANDOM_MULTIPLIER`
        let (nightly, multiplier) = super::base_token_stream::knobs();
        MockAnalyzer {
            random: Arc::new(Mutex::new(JavaRandom::new(random.next_long()))),
            run_automaton,
            lower_case,
            filter,
            custom_term_freq: None,
            position_increment_gap: 0,
            offset_gap: None,
            previous_mappings: Mutex::new(HashMap::new()),
            enable_checks: true,
            max_token_length: DEFAULT_MAX_TOKEN_LENGTH,
            nightly,
            multiplier,
        }
    }

    /// `MockAnalyzer(random, runAutomaton, lowerCase)`.
    pub fn with_tokenizer(
        random: &mut JavaRandom,
        run_automaton: Arc<CharacterRunAutomaton>,
        lower_case: bool,
    ) -> Self {
        Self::with_automaton(random, run_automaton, lower_case, empty_stopset())
    }

    /// `MockAnalyzer(random, customTermFreq)`.
    pub fn with_term_freq(random: &mut JavaRandom, custom_term_freq: TermFreqFn) -> Self {
        let mut a = Self::new(random);
        a.custom_term_freq = Some(custom_term_freq);
        a
    }

    /// `setEnableChecks(enableChecks)`.
    pub fn enable_checks(mut self, enable_checks: bool) -> Self {
        self.enable_checks = enable_checks;
        self
    }

    /// `setMaxTokenLength(length)`.
    pub fn max_token_length(mut self, length: i32) -> Self {
        self.max_token_length = length;
        self
    }

    /// `setPositionIncrementGap(gap)`.
    pub fn position_increment_gap(mut self, gap: i32) -> Self {
        self.position_increment_gap = gap;
        self
    }

    /// `setOffsetGap(gap)`.
    pub fn offset_gap(mut self, gap: i32) -> Self {
        self.offset_gap = Some(gap);
        self
    }

    /// `TEST_NIGHTLY` and `RANDOM_MULTIPLIER`, which `rarely` (the payload
    /// choice) reads; Java reads them off the test run, the default here is
    /// a non-nightly run with multiplier 1.
    pub fn test_knobs(mut self, nightly: bool, multiplier: i32) -> Self {
        self.nightly = nightly;
        self.multiplier = multiplier;
        self
    }

    /// The analyzer, with Java's `PER_FIELD_REUSE_STRATEGY`.
    pub fn build(self) -> Analyzer {
        Analyzer::with_reuse_strategy(self, ReuseStrategy::PerField)
    }

    /// `maybePayload`: the payload kind for `field_name` -- `-1` none,
    /// `i32::MAX` variable length, else a fixed length -- chosen once per
    /// field.
    // SENTINEL: `-1` = no payload filter (Java's `maybePayload` value).
    fn payload_kind(&self, field_name: &str) -> i32 {
        let mut mappings = self
            .previous_mappings
            .lock()
            .unwrap_or_else(|p| p.into_inner());
        if let Some(&val) = mappings.get(field_name) {
            return val;
        }
        let mut random = lock(&self.random);
        let mut val = -1; // no payloads
        if lucene_test_case::rarely(&mut random, self.nightly, self.multiplier) {
            val = match random.next_int_bounded(3) {
                0 => -1,
                1 => i32::MAX,
                _ => random.next_int_bounded(12),
            };
        }
        // save it so we are consistent for this field
        mappings.insert(field_name.to_string(), val);
        val
    }
}

impl AnalyzerDefinition for MockAnalyzer {
    fn create_components(&self, field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let mut tokenizer = MockTokenizer::with_max_token_length(
            Arc::clone(&self.run_automaton),
            self.lower_case,
            self.max_token_length,
        );
        tokenizer.set_enable_checks(self.enable_checks);
        let filt = MockTokenFilter::with_term_freq(
            tokenizer,
            Arc::clone(&self.filter),
            self.custom_term_freq.clone(),
        );
        Ok(match self.payload_kind(field_name) {
            -1 => TokenStreamComponents::new(filt),
            i32::MAX => TokenStreamComponents::new(MockVariableLengthPayloadFilter::new(
                Arc::clone(&self.random),
                filt,
            )),
            len => TokenStreamComponents::new(MockFixedLengthPayloadFilter::new(
                Arc::clone(&self.random),
                filt,
                len,
            )?),
        })
    }

    fn normalize(&self, _field_name: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        if self.lower_case {
            Box::new(MockLowerCaseFilter::new(input))
        } else {
            input
        }
    }

    fn position_increment_gap(&self, _field_name: &str) -> i32 {
        self.position_increment_gap
    }

    fn offset_gap(&self, _field_name: &str) -> i32 {
        self.offset_gap.unwrap_or(1)
    }
}

/// `org.apache.lucene.tests.analysis.MockPayloadAnalyzer`: whitespace
/// tokens with `MockPayloadFilter`'s "pos: N" payloads.
pub struct MockPayloadAnalyzer;

impl MockPayloadAnalyzer {
    /// The analyzer (Java's default reuse strategy).
    pub fn analyzer() -> Analyzer {
        Analyzer::new(MockPayloadAnalyzer)
    }
}

impl AnalyzerDefinition for MockPayloadAnalyzer {
    fn create_components(&self, field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let tokenizer = MockTokenizer::new(mock_tokenizer::whitespace(), true);
        Ok(TokenStreamComponents::new(MockPayloadFilter::new(
            tokenizer, field_name,
        )))
    }
}

/// `org.apache.lucene.tests.analysis.MockSynonymAnalyzer`: a default
/// [`MockTokenizer`] and a [`MockSynonymFilter`].
pub struct MockSynonymAnalyzer;

impl MockSynonymAnalyzer {
    /// The analyzer (Java's default reuse strategy).
    pub fn analyzer() -> Analyzer {
        Analyzer::new(MockSynonymAnalyzer)
    }
}

impl AnalyzerDefinition for MockSynonymAnalyzer {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(MockSynonymFilter::new(
            MockTokenizer::default(),
        )))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::token_stream::consume;

    fn tokens(a: &Analyzer, field: &str, text: &str) -> Vec<(String, Option<Vec<u8>>)> {
        let mut ts = a.token_stream(field, text).unwrap();
        let mut out = Vec::new();
        consume(&mut ts, |t| {
            out.push((t.term().to_string(), t.payload().map(<[u8]>::to_vec)))
        })
        .unwrap();
        out
    }

    #[test]
    fn whitespace_lowercase_and_gaps() {
        let a = MockAnalyzer::new(&mut JavaRandom::new(0))
            .position_increment_gap(7)
            .offset_gap(3)
            .build();
        let got: Vec<_> = tokens(&a, "f", "Hello World")
            .into_iter()
            .map(|t| t.0)
            .collect();
        assert_eq!(got, ["hello", "world"]);
        assert_eq!(a.position_increment_gap_for_field("f"), 7);
        assert_eq!(a.offset_gap_for_field("f"), 3);
        assert_eq!(a.normalize("f", "AbC").unwrap(), b"abc");
        let d =
            MockAnalyzer::with_tokenizer(&mut JavaRandom::new(0), mock_tokenizer::keyword(), false)
                .build();
        assert_eq!(d.offset_gap_for_field("f"), 1);
        assert_eq!(d.normalize("f", "AbC").unwrap(), b"AbC");
    }

    #[test]
    fn every_payload_kind_is_chosen_and_kept_per_field() {
        // a high multiplier makes `rarely` likely, so all three kinds show
        let a = MockAnalyzer::new(&mut JavaRandom::new(1))
            .test_knobs(true, 100)
            .enable_checks(true)
            .max_token_length(10)
            .build();
        let mut kinds = std::collections::BTreeSet::new();
        for i in 0..200 {
            let field = format!("f{i}");
            let first = tokens(&a, &field, "a b c");
            let kind = match &first[0].1 {
                None => "none",
                Some(p)
                    if first
                        .iter()
                        .all(|t| t.1.as_ref().map(Vec::len) == Some(p.len())) =>
                {
                    "fixed"
                }
                Some(_) => "variable",
            };
            kinds.insert(kind);
            // the same field keeps its choice
            let again = tokens(&a, &field, "a b c");
            assert_eq!(again[0].1.is_some(), first[0].1.is_some());
        }
        assert!(
            kinds.contains("none") && kinds.contains("fixed"),
            "{kinds:?}"
        );
    }

    #[test]
    fn term_freq_function_is_applied() {
        let f: TermFreqFn = Arc::new(|t: &str| t.len() as i32);
        let a = MockAnalyzer::with_term_freq(&mut JavaRandom::new(0), f).build();
        let mut ts = a.token_stream("f", "abc d").unwrap();
        let mut freqs = Vec::new();
        consume(&mut ts, |t| freqs.push(t.term_frequency())).unwrap();
        assert_eq!(freqs, [3, 1]);
    }

    #[test]
    fn payload_and_synonym_analyzers() {
        let got = tokens(&MockPayloadAnalyzer::analyzer(), "f", "a b c");
        assert_eq!(got[0], ("a".into(), Some(b"pos: 0".to_vec())));
        let a = MockSynonymAnalyzer::analyzer();
        let got: Vec<_> = tokens(&a, "f", "dogs").into_iter().map(|t| t.0).collect();
        assert_eq!(got, ["dogs", "dog"]);
    }
}
