//! The `core` package's analyzers: `WhitespaceAnalyzer`,
//! `UnicodeWhitespaceAnalyzer`, `SimpleAnalyzer` and `StopAnalyzer`
//! (`KeywordAnalyzer` is [`crate::Analyzer::keyword`]). Each is an
//! [`AnalyzerDefinition`]; wrap it in [`crate::Analyzer::new`].

use std::sync::Arc;

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::token_stream::TokenStream;
use crate::util::char_tokenizer::DEFAULT_MAX_WORD_LEN;
use crate::util::{LetterTokenizer, UnicodeWhitespaceTokenizer, WhitespaceTokenizer};
use crate::{AnalysisError, CharArraySet, LowerCaseFilter, StopFilter, StopwordAnalyzerBase};

/// `org.apache.lucene.analysis.core.WhitespaceAnalyzer`.
#[derive(Debug, Clone, Copy)]
pub struct WhitespaceAnalyzer {
    max_token_length: usize,
}

impl Default for WhitespaceAnalyzer {
    /// `new WhitespaceAnalyzer()`: `maxTokenLength` 255.
    fn default() -> Self {
        Self::with_max_token_length(DEFAULT_MAX_WORD_LEN)
    }
}

impl WhitespaceAnalyzer {
    /// `new WhitespaceAnalyzer(int maxTokenLength)`; a bad length fails when
    /// the components are created, as Java's tokenizer constructor throws
    /// there.
    pub fn with_max_token_length(max_token_length: usize) -> Self {
        WhitespaceAnalyzer { max_token_length }
    }
}

impl AnalyzerDefinition for WhitespaceAnalyzer {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(
            WhitespaceTokenizer::with_max_token_len(self.max_token_length)?,
        ))
    }
}

/// `org.apache.lucene.analysis.core.UnicodeWhitespaceAnalyzer`.
#[derive(Debug, Clone, Copy, Default)]
pub struct UnicodeWhitespaceAnalyzer;

impl AnalyzerDefinition for UnicodeWhitespaceAnalyzer {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(UnicodeWhitespaceTokenizer::new()))
    }
}

/// `org.apache.lucene.analysis.core.SimpleAnalyzer`: `LetterTokenizer` +
/// `LowerCaseFilter`.
#[derive(Debug, Clone, Copy, Default)]
pub struct SimpleAnalyzer;

impl AnalyzerDefinition for SimpleAnalyzer {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(LowerCaseFilter::new(
            LetterTokenizer::new(),
        )))
    }

    fn normalize(&self, _field_name: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}

/// `org.apache.lucene.analysis.core.StopAnalyzer`: `LetterTokenizer` +
/// `LowerCaseFilter` + `StopFilter`.
#[derive(Debug, Clone)]
pub struct StopAnalyzer {
    base: StopwordAnalyzerBase,
}

impl StopAnalyzer {
    /// `new StopAnalyzer(CharArraySet)`.
    pub fn new(stop_words: Arc<CharArraySet>) -> Self {
        StopAnalyzer {
            base: StopwordAnalyzerBase::new(Some((*stop_words).clone())),
        }
    }

    /// `getStopwordSet()`.
    pub fn stopword_set(&self) -> &Arc<CharArraySet> {
        self.base.stopword_set()
    }
}

impl AnalyzerDefinition for StopAnalyzer {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        Ok(TokenStreamComponents::new(StopFilter::new(
            LowerCaseFilter::new(LetterTokenizer::new()),
            Arc::clone(self.base.stopword_set()),
        )))
    }

    fn normalize(&self, _field_name: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Analyzer;

    fn terms(a: &Analyzer, text: &str) -> Vec<String> {
        a.analyze(text).into_iter().map(|t| t.term).collect()
    }

    #[test]
    fn analyzers_tokenize_and_normalize() {
        let ws = Analyzer::new(WhitespaceAnalyzer::default());
        assert_eq!(terms(&ws, "Ab\u{00A0}c d"), vec!["Ab\u{00A0}c", "d"]);
        let short = Analyzer::new(WhitespaceAnalyzer::with_max_token_length(2));
        assert_eq!(terms(&short, "abc"), vec!["ab", "c"]);
        let bad = Analyzer::new(WhitespaceAnalyzer::with_max_token_length(0));
        assert!(bad.try_analyze_stream("x").is_err());
        let uw = Analyzer::new(UnicodeWhitespaceAnalyzer);
        assert_eq!(terms(&uw, "Ab\u{00A0}c"), vec!["Ab", "c"]);
        let simple = Analyzer::new(SimpleAnalyzer);
        assert_eq!(terms(&simple, "The 2 Dogs"), vec!["the", "dogs"]);
        assert_eq!(simple.normalize("f", "ABC").unwrap(), b"abc");
        let stop = StopAnalyzer::new(Arc::new(CharArraySet::from_words(["the"], false)));
        assert!(stop.stopword_set().contains("the"));
        let stop = Analyzer::new(stop);
        assert_eq!(terms(&stop, "The Dogs"), vec!["dogs"]);
        assert_eq!(stop.normalize("f", "XY").unwrap(), b"xy");
    }
}
