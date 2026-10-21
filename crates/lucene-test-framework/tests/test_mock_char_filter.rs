//! Lucene's `TestMockCharFilter`, ported.

use lucene_analysis::{
    AnalysisError, Analyzer, AnalyzerDefinition, CharReader, TokenStreamComponents,
};
use lucene_test_framework::analysis::mock_tokenizer::{self, MockTokenizer};
use lucene_test_framework::analysis::{assert_analyzes_to, Expected, MockCharFilter};
use lucene_test_framework::util::lucene_test_case::LuceneTestCase;

struct WithCharFilter;

impl AnalyzerDefinition for WithCharFilter {
    fn create_components(&self, _: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(MockTokenizer::new(
            mock_tokenizer::whitespace(),
            false,
        )))
    }

    fn init_reader(&self, _: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(MockCharFilter::new(reader, 7).unwrap())
    }
}

#[test]
fn test() {
    let mut t = LuceneTestCase::new("test");
    let r = t.random();
    let analyzer = Analyzer::new(WithCharFilter);
    assert_analyzes_to(
        r,
        &analyzer,
        "ab",
        &Expected::terms(&["aab"]).offsets(&[0], &[2]),
    )
    .unwrap();
    assert_analyzes_to(
        r,
        &analyzer,
        "aba",
        &Expected::terms(&["aabaa"]).offsets(&[0], &[3]),
    )
    .unwrap();
    assert_analyzes_to(
        r,
        &analyzer,
        "abcdefga",
        &Expected::terms(&["aabcdefgaa"]).offsets(&[0], &[8]),
    )
    .unwrap();
}
