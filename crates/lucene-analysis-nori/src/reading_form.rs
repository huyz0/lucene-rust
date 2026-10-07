//! `org.apache.lucene.analysis.ko.KoreanReadingFormFilter`: replaces a term
//! by its reading (`ReadingAttribute`, e.g. the Hangul of a Hanja word).

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

use crate::attributes::ReadingAttribute;

/// `KoreanReadingFormFilter`.
pub struct KoreanReadingFormFilter<I> {
    input: I,
}

impl<I: TokenStream> KoreanReadingFormFilter<I> {
    /// `new KoreanReadingFormFilter(input)`.
    pub fn new(mut input: I) -> Self {
        input.attributes_mut().add_custom::<ReadingAttribute>();
        KoreanReadingFormFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for KoreanReadingFormFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let atts = self.input.attributes_mut();
        if let Some(reading) = atts
            .custom::<ReadingAttribute>()
            .and_then(ReadingAttribute::reading)
        {
            atts.set_term(&reading);
        }
        Ok(true)
    }
}
