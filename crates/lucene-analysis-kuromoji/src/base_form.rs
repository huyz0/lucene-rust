//! `org.apache.lucene.analysis.ja.JapaneseBaseFormFilter`: replaces a term
//! by its base form (`BaseFormAttribute`), unless it is a keyword.

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

use crate::attributes::BaseFormAttribute;

/// `JapaneseBaseFormFilter`.
pub struct JapaneseBaseFormFilter<I> {
    input: I,
}

impl<I: TokenStream> JapaneseBaseFormFilter<I> {
    /// `new JapaneseBaseFormFilter(input)`.
    pub fn new(mut input: I) -> Self {
        input.attributes_mut().add_custom::<BaseFormAttribute>();
        JapaneseBaseFormFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for JapaneseBaseFormFilter<I> {
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
        if !atts.is_keyword() {
            if let Some(base) = atts
                .custom::<BaseFormAttribute>()
                .and_then(BaseFormAttribute::base_form)
            {
                atts.set_term(&base);
            }
        }
        Ok(true)
    }
}
