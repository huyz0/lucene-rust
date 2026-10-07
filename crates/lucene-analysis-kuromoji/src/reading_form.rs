//! `org.apache.lucene.analysis.ja.JapaneseReadingFormFilter`: replaces a
//! term by its reading (`ReadingAttribute`), in katakana or romanized.

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

use crate::attributes::ReadingAttribute;
use crate::dict::to_string_util;

const HIRAGANA_START: u16 = 0x3041;
const HIRAGANA_END: u16 = 0x3096;

fn is_hiragana(ch: u16) -> bool {
    (HIRAGANA_START..=HIRAGANA_END).contains(&ch)
}

/// `JapaneseReadingFormFilter`.
pub struct JapaneseReadingFormFilter<I> {
    input: I,
    use_romaji: bool,
}

impl<I: TokenStream> JapaneseReadingFormFilter<I> {
    /// `new JapaneseReadingFormFilter(input, useRomaji)`.
    pub fn new(mut input: I, use_romaji: bool) -> Self {
        input.attributes_mut().add_custom::<ReadingAttribute>();
        JapaneseReadingFormFilter { input, use_romaji }
    }
}

impl<I: TokenStream> TokenFilter for JapaneseReadingFormFilter<I> {
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
        let term: Vec<u16> = atts.term().encode_utf16().collect();
        let mut reading: Option<Vec<u16>> = atts
            .custom::<ReadingAttribute>()
            .and_then(ReadingAttribute::reading)
            .map(|r| r.encode_utf16().collect());
        if reading.is_none() && term.iter().any(|&c| is_hiragana(c)) {
            // When a term is OOV and contains hiragana, convert the term to
            // katakana and treat it as reading.
            reading = Some(
                term.iter()
                    .map(|&c| {
                        if is_hiragana(c) {
                            c.wrapping_add(0x60)
                        } else {
                            c
                        }
                    })
                    .collect(),
            );
        }
        if self.use_romaji {
            // if it's an OOV term, just try the term text
            let source = reading.as_deref().unwrap_or(&term);
            let romaji = to_string_util::romanization_utf16(source);
            atts.set_term_utf16(&romaji);
        } else if let Some(r) = reading {
            // just replace the term text with the reading, if it exists
            atts.set_term_utf16(&r);
        }
        Ok(true)
    }
}
