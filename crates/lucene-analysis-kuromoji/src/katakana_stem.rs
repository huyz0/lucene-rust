//! `org.apache.lucene.analysis.ja.JapaneseKatakanaStemFilter`: drops the
//! prolonged sound mark (ー) ending a long enough all-katakana term.

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

/// `JapaneseKatakanaStemFilter.DEFAULT_MINIMUM_LENGTH`.
pub const DEFAULT_MINIMUM_LENGTH: i32 = 4;
const HIRAGANA_KATAKANA_PROLONGED_SOUND_MARK: u16 = 0x30FC;

/// `JapaneseKatakanaStemFilter`.
pub struct JapaneseKatakanaStemFilter<I> {
    input: I,
    minimum_katakana_length: usize,
}

impl<I: TokenStream> JapaneseKatakanaStemFilter<I> {
    /// `new JapaneseKatakanaStemFilter(input, minimumLength)`
    /// (`IllegalArgumentException` below 1).
    pub fn new(input: I, minimum_length: i32) -> Result<Self, AnalysisError> {
        let minimum_katakana_length = usize::try_from(minimum_length)
            .ok()
            .filter(|&n| n >= 1)
            .ok_or_else(|| {
                AnalysisError::IllegalArgument("minimumLength must be >=1".to_string())
            })?;
        Ok(JapaneseKatakanaStemFilter {
            input,
            minimum_katakana_length,
        })
    }
}

/// `Character.UnicodeBlock.of(c) == KATAKANA` (U+30A0..U+30FF).
fn is_katakana(term: &[u16]) -> bool {
    term.iter().all(|&c| (0x30A0..=0x30FF).contains(&c))
}

impl<I: TokenStream> TokenFilter for JapaneseKatakanaStemFilter<I> {
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
        let min = self.minimum_katakana_length;
        let atts = self.input.attributes_mut();
        if !atts.is_keyword() {
            let term: Vec<u16> = atts.term().encode_utf16().collect();
            // Java: stem(term, length)
            if term.len() >= min
                && is_katakana(&term)
                && term.last() == Some(&HIRAGANA_KATAKANA_PROLONGED_SOUND_MARK)
            {
                atts.term_mut().pop();
            }
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn minimum_length_is_checked() {
        let e = JapaneseKatakanaStemFilter::new(lucene_analysis::KeywordTokenizer::new(), 0);
        assert_eq!(
            e.err().unwrap().to_string(),
            "illegal argument: minimumLength must be >=1"
        );
        assert!(is_katakana(&[0x30A2, 0x30FC]) && !is_katakana(&[0x3042]));
    }
}
