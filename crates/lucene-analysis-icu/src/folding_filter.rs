//! `org.apache.lucene.analysis.icu.ICUFoldingFilter`: search-term folding
//! from UTR #30 (case and accent folding, width and compatibility forms,
//! default ignorables removed) -- [`ICUNormalizer2Filter`] over Lucene's
//! `utr30.nrm` ([`Normalizer2::utr30`]).

use lucene_analysis::TokenStream;

use crate::icu4j::normalizer2::Normalizer2;
use crate::normalizer2_filter::ICUNormalizer2Filter;

/// `ICUFoldingFilter`: an [`ICUNormalizer2Filter`] whose normalizer is
/// `ICUFoldingFilter.NORMALIZER` unless one is given.
pub struct ICUFoldingFilter;

impl ICUFoldingFilter {
    /// `ICUFoldingFilter.NORMALIZER`.
    pub fn normalizer() -> Normalizer2 {
        Normalizer2::utr30()
    }

    /// `new ICUFoldingFilter(TokenStream)`.
    #[allow(clippy::new_ret_no_self)]
    pub fn new<I: TokenStream>(input: I) -> ICUNormalizer2Filter<I> {
        ICUNormalizer2Filter::with_normalizer(input, Self::normalizer())
    }

    /// `new ICUFoldingFilter(TokenStream, Normalizer2)`.
    pub fn with_normalizer<I: TokenStream>(
        input: I,
        normalizer: Normalizer2,
    ) -> ICUNormalizer2Filter<I> {
        ICUNormalizer2Filter::with_normalizer(input, normalizer)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::{KeywordTokenizer, StrReader, Tokenizer};

    #[test]
    fn folds() {
        for (text, want) in [("Résumé", "resume"), ("ＡＢＣ", "abc"), ("ﬁ", "fi")] {
            let mut t = KeywordTokenizer::new();
            t.set_reader(Box::new(StrReader::new(text))).unwrap();
            let mut s: Box<dyn TokenStream> = if want == "fi" {
                Box::new(ICUFoldingFilter::with_normalizer(
                    t,
                    ICUFoldingFilter::normalizer(),
                ))
            } else {
                Box::new(ICUFoldingFilter::new(t))
            };
            s.reset().unwrap();
            assert!(s.increment_token().unwrap());
            assert_eq!(s.attributes().term(), want);
        }
    }
}
