//! `org.apache.lucene.analysis.icu.ICUTransformFilter`: each term through
//! an ICU [`Transliterator`] (`filteredTransliterate(text, position,
//! false)` over the whole term).
//!
//! As in Java, a rule-based transliterator without a filter gets its
//! source set as its filter (`getSourceSet()`, characters no rule can
//! change are skipped without running the rules).

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

use crate::icu4j::translit::{Position, Transliterator};

/// `ICUTransformFilter`.
pub struct ICUTransformFilter<I = Box<dyn TokenStream>> {
    input: I,
    transform: Transliterator,
    units: Vec<u16>,
}

impl<I: TokenStream> ICUTransformFilter<I> {
    /// `ICUTransformFilter(input, transform)`.
    pub fn new(input: I, transform: Transliterator) -> Self {
        let mut transform = transform;
        if transform.filter().is_none() && transform.is_rule_based() {
            if let Ok(source) = transform.source_set() {
                if !source.is_empty() {
                    transform.set_filter(Some(source));
                }
            }
        }
        ICUTransformFilter {
            input,
            transform,
            units: Vec::new(),
        }
    }

    /// The transliterator (with the filter the constructor may have set).
    pub fn transliterator(&self) -> &Transliterator {
        &self.transform
    }
}

impl<I: TokenStream> TokenFilter for ICUTransformFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    // Java: ICUTransformFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let atts = self.input.attributes_mut();
        self.units.clear();
        self.units.extend(atts.term().encode_utf16());
        let length = i32::try_from(self.units.len())
            .map_err(|_| AnalysisError::IllegalArgument("term longer than 2^31 units".into()))?;
        let mut position = Position {
            context_start: 0,
            context_limit: length,
            start: 0,
            limit: length,
        };
        let before = self.units.clone();
        self.transform
            .filtered_transliterate(&mut self.units, &mut position)?;
        if self.units != before {
            atts.set_term_utf16(&self.units);
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::{KeywordTokenizer, StrReader, Tokenizer};

    fn run(t: Transliterator, text: &str) -> String {
        let mut k = KeywordTokenizer::new();
        k.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut s = ICUTransformFilter::new(k, t);
        s.reset().unwrap();
        assert!(s.increment_token().unwrap());
        let term = s.attributes().term().to_string();
        assert!(!s.increment_token().unwrap());
        term
    }

    #[test]
    fn transforms_terms() {
        let t = Transliterator::get_instance("Katakana-Hiragana", 0).unwrap();
        assert_eq!(run(t, "ヒラガナ"), "ひらがな");
        let t = Transliterator::create_from_rules("test", "a > b;", 0).unwrap();
        let k = KeywordTokenizer::new();
        let f = ICUTransformFilter::new(k, t);
        assert!(f.transliterator().filter().is_some());
        let t = Transliterator::create_from_rules("test", "a > b;", 0).unwrap();
        assert_eq!(run(t, "cab"), "cbb");
        let t = Transliterator::get_instance("Null", 0).unwrap();
        assert_eq!(run(t, "x"), "x");
    }
}
