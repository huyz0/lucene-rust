//! `org.apache.lucene.analysis.no`: `NorwegianAnalyzer` (Snowball), the light
//! and minimal stemmers (Bokmål and/or Nynorsk) and
//! `NorwegianNormalizationFilter`.

use std::sync::{Arc, LazyLock};

use crate::miscellaneous::{Folding, ScandinavianNormalizer};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::stemmer_util::{ends, ends_with};
use crate::{AnalysisError, CharArraySet};

use super::{mark_exclusions, snowball, snowball_set, std_lower_stop, CharStemmer, StemFilter};

/// `NorwegianAnalyzer.getDefaultStopSet()` (`snowball/norwegian_stop.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| snowball_set(include_str!("stopwords/norwegian_stop.txt")));

/// `NorwegianLightStemmer.BOKMAAL`.
pub const BOKMAAL: i32 = 1;
/// `NorwegianLightStemmer.NYNORSK`.
pub const NYNORSK: i32 = 2;

fn flags(flags: i32) -> Result<(bool, bool), AnalysisError> {
    if flags <= 0 || flags > BOKMAAL + NYNORSK {
        return Err(AnalysisError::IllegalArgument("invalid flags".into()));
    }
    Ok((flags & BOKMAAL != 0, flags & NYNORSK != 0))
}

/// `NorwegianLightStemmer`.
#[derive(Debug, Clone, Copy)]
pub struct NorwegianLightStemmer {
    bokmaal: bool,
    nynorsk: bool,
}

impl Default for NorwegianLightStemmer {
    /// `BOKMAAL`, as `new NorwegianLightStemFilter(input)`.
    fn default() -> Self {
        Self::new(BOKMAAL).expect("valid flags")
    }
}

impl NorwegianLightStemmer {
    /// `new NorwegianLightStemmer(flags)`.
    pub fn new(f: i32) -> Result<Self, AnalysisError> {
        let (bokmaal, nynorsk) = flags(f)?;
        Ok(NorwegianLightStemmer { bokmaal, nynorsk })
    }
}

impl CharStemmer for NorwegianLightStemmer {
    // Java: NorwegianLightStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let (b, n) = (self.bokmaal, self.nynorsk);
        let e = |s: &[u16], len: usize, x: &str| ends_with(s, len, x);
        if len > 4 && s[len - 1] == u16::from(b's') {
            len -= 1;
        }
        if len > 7
            && ((e(s, len, "heter") && b) || (e(s, len, "heten") && b) || (e(s, len, "heita") && n))
        {
            return len - 5;
        }
        if len > 8 && n && (e(s, len, "heiter") || e(s, len, "leiken") || e(s, len, "leikar")) {
            return len - 6;
        }
        if len > 5 && (e(s, len, "dom") || (e(s, len, "het") && b)) {
            return len - 3;
        }
        if len > 6 && n && (e(s, len, "heit") || e(s, len, "semd") || e(s, len, "leik")) {
            return len - 4;
        }
        if len > 7 && (e(s, len, "elser") || e(s, len, "elsen")) {
            return len - 5;
        }
        if len > 6
            && ((e(s, len, "ende") && b)
                || (e(s, len, "ande") && n)
                || e(s, len, "else")
                || (e(s, len, "este") && b)
                || (e(s, len, "aste") && n)
                || (e(s, len, "eren") && b)
                || (e(s, len, "aren") && n))
        {
            return len - 4;
        }
        if len > 5
            && ((e(s, len, "ere") && b)
                || (e(s, len, "are") && n)
                || (e(s, len, "est") && b)
                || (e(s, len, "ast") && n)
                || e(s, len, "ene")
                || (e(s, len, "ane") && n))
        {
            return len - 3;
        }
        if len > 4
            && (e(s, len, "er")
                || e(s, len, "en")
                || e(s, len, "et")
                || (e(s, len, "ar") && n)
                || (e(s, len, "st") && b)
                || e(s, len, "te"))
        {
            return len - 2;
        }
        if len > 3 && matches!(s[len - 1], 0x61 | 0x65 | 0x6E) {
            return len - 1;
        }
        len
    }
}

/// `NorwegianMinimalStemmer`.
#[derive(Debug, Clone, Copy)]
pub struct NorwegianMinimalStemmer {
    nynorsk: bool,
}

impl Default for NorwegianMinimalStemmer {
    /// `BOKMAAL`.
    fn default() -> Self {
        Self::new(BOKMAAL).expect("valid flags")
    }
}

impl NorwegianMinimalStemmer {
    /// `new NorwegianMinimalStemmer(flags)`.
    pub fn new(f: i32) -> Result<Self, AnalysisError> {
        let (_, nynorsk) = flags(f)?;
        Ok(NorwegianMinimalStemmer { nynorsk })
    }
}

impl CharStemmer for NorwegianMinimalStemmer {
    // Java: NorwegianMinimalStemmer.stem
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        let n = self.nynorsk;
        if len > 4 && s[len - 1] == u16::from(b's') {
            len -= 1;
        }
        if len > 5 && (ends!(s, len, "ene") || (ends!(s, len, "ane") && n)) {
            return len - 3;
        }
        if len > 4
            && (ends!(s, len, "er")
                || ends!(s, len, "en")
                || ends!(s, len, "et")
                || (ends!(s, len, "ar") && n))
        {
            return len - 2;
        }
        if len > 3 && matches!(s[len - 1], 0x61 | 0x65) {
            return len - 1;
        }
        len
    }
}

/// `NorwegianLightStemFilter`.
pub type NorwegianLightStemFilter<I> = StemFilter<I, NorwegianLightStemmer>;
/// `NorwegianMinimalStemFilter`.
pub type NorwegianMinimalStemFilter<I> = StemFilter<I, NorwegianMinimalStemmer>;

/// `NorwegianNormalizationFilter`: `ScandinavianNormalizer` with the `AE`,
/// `OE` and `AA` foldings.
pub struct NorwegianNormalizationFilter<I> {
    input: I,
    normalizer: ScandinavianNormalizer,
    buf: Vec<u16>,
}

impl<I: TokenStream> NorwegianNormalizationFilter<I> {
    /// `new NorwegianNormalizationFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        NorwegianNormalizationFilter {
            input,
            normalizer: ScandinavianNormalizer::new(&[Folding::Ae, Folding::Oe, Folding::Aa]),
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for NorwegianNormalizationFilter<I> {
    crate::filter_input!();

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let n = &self.normalizer;
        crate::util::with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
            n.process_token(b);
            true
        });
        Ok(true)
    }
}

language_analyzer! {
    /// `NorwegianAnalyzer`: `StandardTokenizer`, `LowerCaseFilter`,
    /// `StopFilter`, exclusions, Snowball `NorwegianStemmer`.
    NorwegianAnalyzer, DEFAULT_STOP_SET,
    components(s) {
        snowball(mark_exclusions(std_lower_stop(&s.stopwords), &s.exclusion), "Norwegian")
    }
    normalize(s, input) {
        crate::LowerCaseFilter::new(input)
    }
}
