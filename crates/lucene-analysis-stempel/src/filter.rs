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
    /// The term as UTF-16 and its stem, reused per token.
    word: Vec<u16>,
    stem: Vec<u16>,
}

impl<I: TokenStream> StempelFilter<I> {
    /// `new StempelFilter(in, stemmer)`.
    pub fn new(input: I, stemmer: StempelStemmer) -> Self {
        StempelFilter {
            input,
            stemmer,
            min_length: DEFAULT_MIN_LENGTH,
            word: Vec::new(),
            stem: Vec::new(),
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
            min_length,
            ..Self::new(input, stemmer)
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
            self.word.clear();
            self.word.extend(attrs.term().encode_utf16());
            // A term of at least one unit is never the empty key a plain
            // trie throws on.
            if let Ok(true) = self.stemmer.stem_into(&self.word, &mut self.stem) {
                attrs.set_term_utf16(&self.stem);
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
