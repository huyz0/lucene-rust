//! `org.apache.lucene.analysis.en`: the streaming `PorterStemFilter`,
//! `EnglishPossessiveFilter`, `EnglishMinimalStemFilter` and
//! `EnglishAnalyzer`, `KStemFilter` (`kstem.rs`). The Porter algorithm itself is the crate's `porter`
//! module (`PorterStemmer`), shared with the older `Vec<Token>` API.

use std::sync::Arc;

mod kstem;
#[rustfmt::skip]
mod kstem_data;

pub use kstem::{KStemFilter, KStemmer};

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::miscellaneous::SetKeywordMarkerFilter;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::with_utf16_term;
use crate::{
    AnalysisError, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter,
    StopwordAnalyzerBase, ENGLISH_STOP_WORDS,
};

/// `org.apache.lucene.analysis.en.PorterStemFilter`: stems every non-keyword
/// term.
pub struct PorterStemFilter<I> {
    input: I,
}

impl<I: TokenStream> PorterStemFilter<I> {
    /// `new PorterStemFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        PorterStemFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for PorterStemFilter<I> {
    crate::filter_input!();
    // Java: PorterStemFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if !a.is_keyword() {
            let stemmed = crate::porter::stem(a.term());
            if stemmed != a.term() {
                a.set_term(&stemmed);
            }
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.en.EnglishPossessiveFilter`: strips a final
/// `'s`, `’s` or `＇s` (either case of `s`).
pub struct EnglishPossessiveFilter<I> {
    input: I,
    buf: Vec<u16>,
}

impl<I: TokenStream> EnglishPossessiveFilter<I> {
    /// `new EnglishPossessiveFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        EnglishPossessiveFilter {
            input,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for EnglishPossessiveFilter<I> {
    crate::filter_input!();
    // Java: EnglishPossessiveFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
            let n = b.len();
            if n >= 2
                && matches!(b[n - 2], 0x27 | 0x2019 | 0xFF07)
                && (b[n - 1] == u16::from(b's') || b[n - 1] == u16::from(b'S'))
            {
                b.truncate(n - 2);
                return true;
            }
            false
        });
        Ok(true)
    }
}

/// `EnglishMinimalStemmer.stem(char[], int)`: the new length (`s` may have
/// its third-last unit replaced by `y`).
fn minimal_stem(s: &mut [u16], len: usize) -> usize {
    let c = |i: usize| s[i];
    let is = |u: u16, ch: u8| u == u16::from(ch);
    if len < 3 || !is(c(len - 1), b's') {
        return len;
    }
    let pen = c(len - 2);
    if is(pen, b'u') || is(pen, b's') {
        return len;
    }
    if is(pen, b'e') {
        if len > 3 && is(c(len - 3), b'i') && !is(c(len - 4), b'a') && !is(c(len - 4), b'e') {
            s[len - 3] = u16::from(b'y');
            return len - 2;
        }
        let t = c(len - 3);
        if is(t, b'i') || is(t, b'a') || is(t, b'o') || is(t, b'e') {
            return len;
        }
    }
    len - 1
}

/// `org.apache.lucene.analysis.en.EnglishMinimalStemFilter`: plural `s`
/// removal (`EnglishMinimalStemmer`).
pub struct EnglishMinimalStemFilter<I> {
    input: I,
    buf: Vec<u16>,
}

impl<I: TokenStream> EnglishMinimalStemFilter<I> {
    /// `new EnglishMinimalStemFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        EnglishMinimalStemFilter {
            input,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for EnglishMinimalStemFilter<I> {
    crate::filter_input!();
    // Java: EnglishMinimalStemFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        if !self.input.attributes().is_keyword() {
            with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
                let n = b.len();
                let new_len = minimal_stem(b, n);
                b.truncate(new_len);
                true
            });
        }
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.en.EnglishAnalyzer`: `StandardTokenizer`,
/// `EnglishPossessiveFilter`, `LowerCaseFilter`, `StopFilter`, the
/// stem-exclusion `SetKeywordMarkerFilter`, `PorterStemFilter`.
#[derive(Debug, Clone)]
pub struct EnglishAnalyzer {
    base: StopwordAnalyzerBase,
    stem_exclusion_set: Arc<CharArraySet>,
}

impl Default for EnglishAnalyzer {
    /// `new EnglishAnalyzer()`: `ENGLISH_STOP_WORDS_SET`.
    fn default() -> Self {
        Self::new(
            CharArraySet::from_words(ENGLISH_STOP_WORDS, false),
            CharArraySet::empty(),
        )
    }
}

impl EnglishAnalyzer {
    /// `new EnglishAnalyzer(CharArraySet stopwords, CharArraySet stemExclusionSet)`.
    pub fn new(stopwords: CharArraySet, stem_exclusion_set: CharArraySet) -> Self {
        EnglishAnalyzer {
            base: StopwordAnalyzerBase::new(Some(stopwords)),
            stem_exclusion_set: Arc::new(stem_exclusion_set),
        }
    }
}

impl AnalyzerDefinition for EnglishAnalyzer {
    fn create_components(&self, _field_name: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let result = StopFilter::new(
            LowerCaseFilter::new(EnglishPossessiveFilter::new(StandardTokenizer::new())),
            Arc::clone(self.base.stopword_set()),
        );
        Ok(if self.stem_exclusion_set.is_empty() {
            TokenStreamComponents::new(PorterStemFilter::new(result))
        } else {
            TokenStreamComponents::new(PorterStemFilter::new(SetKeywordMarkerFilter::new(
                result,
                Arc::clone(&self.stem_exclusion_set),
            )))
        })
    }

    fn normalize(&self, _field_name: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::Canned;
    use crate::Analyzer;

    fn terms(ts: &mut dyn TokenStream) -> Vec<String> {
        let mut out = Vec::new();
        crate::token_stream::consume(ts, |a| out.push(a.term().to_string())).unwrap();
        out
    }

    #[test]
    fn porter_skips_keywords() {
        let mut c = Canned::parse("running:0:7:1:1 running:8:15:1:1");
        c.set_keywords(&[false, true]);
        assert_eq!(terms(&mut PorterStemFilter::new(c)), vec!["run", "running"]);
    }

    #[test]
    fn possessives_and_minimal_plurals() {
        let mut c = Canned::parse("a:0:1:1:1 b:0:1:1:1 c:0:1:1:1 d:0:1:1:1");
        c.set_terms(&["John's", "JAMES\u{2019}S", "x＇s", "'s"]);
        assert_eq!(
            terms(&mut EnglishPossessiveFilter::new(c)),
            vec!["John", "JAMES", "x", ""]
        );
        let words = [
            "ponies", "boxes", "bus", "class", "toys", "cats", "aries", "foes", "goes", "is", "x",
        ];
        let mut c = Canned::parse(&"w:0:1:1:1 ".repeat(words.len()));
        c.set_terms(&words);
        assert_eq!(
            terms(&mut EnglishMinimalStemFilter::new(c)),
            vec!["pony", "boxe", "bus", "class", "toy", "cat", "ary", "foes", "goes", "is", "x"]
        );
        let mut c = Canned::parse("dogs:0:4:1:1");
        c.set_keywords(&[true]);
        assert_eq!(terms(&mut EnglishMinimalStemFilter::new(c)), vec!["dogs"]);
    }

    #[test]
    fn analyzer_with_exclusions() {
        let a = Analyzer::new(EnglishAnalyzer::new(
            CharArraySet::from_words(["the"], false),
            CharArraySet::from_words(["running"], false),
        ));
        let t: Vec<String> = a
            .analyze("The running dogs")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert_eq!(t, vec!["running", "dog"]);
        assert_eq!(a.normalize("f", "ABC").unwrap(), b"abc");
    }
}
