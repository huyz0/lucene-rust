//! `org.apache.lucene.tests.analysis`: the mock analysis chain and
//! `BaseTokenStreamTestCase`.
//!
//! The mocks are ordinary [`lucene_analysis`] streams -- a [`MockTokenizer`]
//! is a [`lucene_analysis::Tokenizer`], every mock filter a
//! [`lucene_analysis::TokenFilter`], [`MockAnalyzer`] an
//! [`lucene_analysis::AnalyzerDefinition`] -- so they chain with the
//! production analysis components. Randomness is
//! [`lucene_util::java_random::JavaRandom`], drawn in Java's order, so a
//! mock built from the same seed as its Java twin produces the same tokens
//! (`tests/test_framework_analysis_fixtures.rs` against
//! `GenTestFrameworkAnalysis`).
//!
//! Java's checks that throw (`IllegalStateException` from a mock's state
//! machine, `IOException` from [`CrankyTokenFilter`]) return an
//! [`lucene_analysis::AnalysisError`]; JUnit assertions
//! ([`base_token_stream`]'s `assert_*`) panic.
//!
//! Not ported: `CollationTestBase` (its subclasses test the JDK's and ICU's
//! collators against an index, which the ICU crate's own tests cover) and
//! `MockUTF16TermAttributeImpl`/`MockBytesAnalyzer` (a term attribute whose
//! bytes are UTF-16; the port's term attribute is not pluggable).

pub mod base_token_stream;
pub mod base_token_stream_factory;
pub mod canned;
pub mod cranky_token_filter;
pub mod lookahead;
pub mod mock_analyzer;
pub mod mock_char_filter;
pub mod mock_filters;
pub mod mock_reader_wrapper;
pub mod mock_synonym;
pub mod mock_tokenizer;
pub mod token;
pub mod token_stream_to_dot;
pub mod validating_token_filter;
pub mod vocabulary_assert;

pub use base_token_stream::{
    assert_analyzes_to, assert_graph_strings, assert_graph_strings_of,
    assert_stream_has_number_of_tokens, assert_token_stream_contents, check_analysis_consistency,
    check_one_term, check_random_data, check_random_data_with, check_reset_exception, escape,
    get_graph_strings, get_graph_strings_of, keyword_mock_tokenizer, keyword_mock_tokenizer_from,
    whitespace_mock_tokenizer, whitespace_mock_tokenizer_from, CheckClearAttributes, Expected,
};
pub use canned::{BinaryToken, CannedBinaryTokenStream, CannedTokenStream};
pub use cranky_token_filter::CrankyTokenFilter;
pub use lookahead::{
    LookaheadHooks, LookaheadTokenFilter, MockGraphTokenFilter, MockRandomLookaheadTokenFilter,
    NoHooks, Position,
};
pub use mock_analyzer::{MockAnalyzer, MockPayloadAnalyzer, MockSynonymAnalyzer};
pub use mock_char_filter::MockCharFilter;
pub use mock_filters::{
    MockFixedLengthPayloadFilter, MockHoleInjectingTokenFilter, MockLowerCaseFilter,
    MockPayloadFilter, MockTokenFilter, MockVariableLengthPayloadFilter, SimplePayloadFilter,
};
pub use mock_reader_wrapper::MockReaderWrapper;
pub use mock_synonym::MockSynonymFilter;
pub use mock_tokenizer::MockTokenizer;
pub use token::Token;
pub use token_stream_to_dot::TokenStreamToDot;
pub use validating_token_filter::ValidatingTokenFilter;
pub use vocabulary_assert::{assert_vocabulary, assert_vocabulary_tsv};
