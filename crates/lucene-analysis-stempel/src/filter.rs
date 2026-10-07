//! `org.apache.lucene.analysis.stempel.StempelFilter`.

use lucene_analysis::token_stream::{TokenFilter, TokenStream};
use lucene_analysis::AnalysisError;

use crate::stemmer::StempelStemmer;

/// `StempelFilter.DEFAULT_MIN_LENGTH`.
pub const DEFAULT_MIN_LENGTH: usize = 3;

/// `StempelFilter`: each term of at least `min_length` UTF-16 units, not
/// marked a keyword, replaced by its stem when the stemmer has one.
pub struct StempelFilter<I> {
    input: I,
    stemmer: StempelStemmer,
    min_length: usize,
}

impl<I: TokenStream> StempelFilter<I> {
    /// `new StempelFilter(in, stemmer)`.
    pub fn new(input: I, stemmer: StempelStemmer) -> Self {
        StempelFilter {
            input,
            stemmer,
            min_length: DEFAULT_MIN_LENGTH,
        }
    }

    /// `new StempelFilter(in, stemmer, minLength)`; `IllegalArgumentException`
    /// below 1.
    pub fn with_min_length(
        input: I,
        stemmer: StempelStemmer,
        min_length: i32,
    ) -> Result<Self, AnalysisError> {
        let min_length = usize::try_from(min_length)
            .ok()
            .filter(|&m| m >= 1)
            .ok_or_else(|| AnalysisError::IllegalArgument("minLength must be >=1".into()))?;
        Ok(StempelFilter {
            input,
            stemmer,
            min_length,
        })
    }
}

impl<I: TokenStream> TokenFilter for StempelFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    // Java: StempelFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let attrs = self.input.attributes_mut();
        if !attrs.is_keyword() && attrs.term_utf16_len() >= self.min_length {
            let word: Vec<u16> = attrs.term().encode_utf16().collect();
            // A term of at least one unit is never the empty key a plain
            // trie throws on.
            if let Ok(Some(stem)) = self.stemmer.stem(&word) {
                attrs.set_term_utf16(&stem);
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stemmer::default_table;
    use lucene_analysis::KeywordTokenizer;

    #[test]
    fn min_length_is_checked() {
        let s = StempelStemmer::new(default_table());
        assert!(StempelFilter::with_min_length(KeywordTokenizer::new(), s.clone(), 0).is_err());
        assert!(StempelFilter::with_min_length(KeywordTokenizer::new(), s, 1).is_ok());
    }
}
