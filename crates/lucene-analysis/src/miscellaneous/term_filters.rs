//! The `miscellaneous` package's in-place term rewriters: `TrimFilter`,
//! `TruncateTokenFilter`, `CapitalizationFilter`,
//! `ScandinavianFoldingFilter`, `ScandinavianNormalizationFilter` (and
//! `ScandinavianNormalizer`), `DelimitedTermFrequencyTokenFilter`.
//!
//! Each works, as Java's does, on the term's UTF-16 code units
//! ([`crate::util::with_utf16_term`]), so a cut between the halves of a
//! surrogate pair, or a `char`-wise case mapping, does what Java's does.

use std::sync::Arc;

use crate::java_character;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::with_utf16_term;
use crate::{AnalysisError, CharArraySet};

/// `org.apache.lucene.analysis.miscellaneous.TrimFilter`: strips leading and
/// trailing `Character.isWhitespace` units.
pub struct TrimFilter<I> {
    input: I,
    buf: Vec<u16>,
}

impl<I: TokenStream> TrimFilter<I> {
    /// `new TrimFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        TrimFilter {
            input,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for TrimFilter<I> {
    crate::filter_input!();
    // Java: TrimFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
            let len = b.len();
            if len == 0 {
                return false;
            }
            let ws = |u: u16| java_character::is_whitespace(u32::from(u));
            let mut start = 0;
            while start < len && ws(b[start]) {
                start += 1;
            }
            let mut end = len;
            while end > start && ws(b[end - 1]) {
                end -= 1;
            }
            if start > 0 || end < len {
                b.truncate(end);
                b.drain(..start);
                return true;
            }
            false
        });
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.miscellaneous.TruncateTokenFilter`.
pub struct TruncateTokenFilter<I> {
    input: I,
    truncate_after: usize,
    use_code_points: bool,
    buf: Vec<u16>,
}

impl<I: TokenStream> TruncateTokenFilter<I> {
    fn create(input: I, truncate_after: i32, use_code_points: bool) -> Result<Self, AnalysisError> {
        if truncate_after < 1 {
            return Err(AnalysisError::IllegalArgument(format!(
                "truncateAfter parameter must be a positive number: {truncate_after}"
            )));
        }
        Ok(TruncateTokenFilter {
            input,
            truncate_after: truncate_after as usize,
            use_code_points,
            buf: Vec::new(),
        })
    }

    /// `TruncateTokenFilter.truncateAfterChars` (and the deprecated
    /// `(TokenStream, int)` constructor): UTF-16 units.
    pub fn truncate_after_chars(input: I, n_chars: i32) -> Result<Self, AnalysisError> {
        Self::create(input, n_chars, false)
    }

    /// `TruncateTokenFilter.truncateAfterCodePoints`.
    pub fn truncate_after_code_points(input: I, n_code_points: i32) -> Result<Self, AnalysisError> {
        Self::create(input, n_code_points, true)
    }
}

impl<I: TokenStream> TokenFilter for TruncateTokenFilter<I> {
    crate::filter_input!();
    // Java: TruncateTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        if self.input.attributes().is_keyword() {
            return Ok(true);
        }
        let (after, code_points) = (self.truncate_after, self.use_code_points);
        with_utf16_term(self.input.attributes_mut(), &mut self.buf, |arr| {
            let len = arr.len();
            if len <= after {
                return false;
            }
            if code_points {
                let mut ofs = 0;
                let mut remaining = after;
                while ofs < len && remaining > 0 {
                    let hi = java_character::is_high_surrogate(arr[ofs]);
                    ofs += 1;
                    if hi && ofs < len && java_character::is_low_surrogate(arr[ofs]) {
                        ofs += 1;
                    }
                    remaining -= 1;
                }
                if remaining == 0 {
                    arr.truncate(ofs);
                    return true;
                }
                false
            } else {
                arr.truncate(after);
                true
            }
        });
        Ok(true)
    }
}

/// `CapitalizationFilter.DEFAULT_MAX_WORD_COUNT` / `DEFAULT_MAX_TOKEN_LENGTH`.
pub const CAPITALIZATION_DEFAULT_MAX: i32 = i32::MAX;

/// `Character.toLowerCase(char)`/`toUpperCase(char)` on one UTF-16 unit (a
/// surrogate maps to itself).
fn lower_unit(u: u16) -> u16 {
    u16::try_from(java_character::to_lower_case(u32::from(u))).unwrap_or(u)
}

fn upper_unit(u: u16) -> u16 {
    u16::try_from(java_character::to_upper_case(u32::from(u))).unwrap_or(u)
}

/// `org.apache.lucene.analysis.miscellaneous.CapitalizationFilter`.
pub struct CapitalizationFilter<I> {
    input: I,
    cfg: CapitalizationConfig,
    buf: Vec<u16>,
}

/// `CapitalizationFilter`'s settings.
struct CapitalizationConfig {
    only_first_word: bool,
    keep: Option<Arc<CharArraySet>>,
    force_first_letter: bool,
    ok_prefix: Option<Vec<Vec<u16>>>,
    min_word_length: usize,
    max_word_count: i32,
    max_token_length: i32,
}

impl<I: TokenStream> CapitalizationFilter<I> {
    /// `new CapitalizationFilter(TokenStream)`: only the first word, force
    /// the first letter, no limits.
    pub fn new(input: I) -> Self {
        Self::with_options(
            input,
            true,
            None,
            true,
            None,
            0,
            CAPITALIZATION_DEFAULT_MAX,
            CAPITALIZATION_DEFAULT_MAX,
        )
        .expect("the defaults are valid")
    }

    /// The full constructor.
    #[allow(clippy::too_many_arguments)]
    pub fn with_options(
        input: I,
        only_first_word: bool,
        keep: Option<Arc<CharArraySet>>,
        force_first_letter: bool,
        ok_prefix: Option<Vec<String>>,
        min_word_length: i32,
        max_word_count: i32,
        max_token_length: i32,
    ) -> Result<Self, AnalysisError> {
        if min_word_length < 0 {
            return Err(AnalysisError::IllegalArgument(
                "minWordLength must be greater than or equal to zero".into(),
            ));
        }
        if max_word_count < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxWordCount must be greater than zero".into(),
            ));
        }
        if max_token_length < 1 {
            return Err(AnalysisError::IllegalArgument(
                "maxTokenLength must be greater than zero".into(),
            ));
        }
        Ok(CapitalizationFilter {
            input,
            buf: Vec::new(),
            cfg: CapitalizationConfig {
                only_first_word,
                keep,
                force_first_letter,
                ok_prefix: ok_prefix
                    .map(|v| v.iter().map(|p| p.encode_utf16().collect()).collect()),
                min_word_length: min_word_length as usize,
                max_word_count,
                max_token_length,
            },
        })
    }
}

impl CapitalizationConfig {
    // Java: CapitalizationFilter.processWord
    fn process_word(&self, buffer: &mut [u16], offset: usize, length: usize, word_count: i32) {
        if length < 1 {
            return;
        }
        let word = &mut buffer[offset..offset + length];
        if self.only_first_word && word_count > 0 {
            for u in word.iter_mut() {
                *u = lower_unit(*u);
            }
            return;
        }
        if let Some(keep) = &self.keep {
            if keep.contains(&String::from_utf16_lossy(word)) {
                if word_count == 0 && self.force_first_letter {
                    word[0] = upper_unit(word[0]);
                }
                return;
            }
        }
        if length < self.min_word_length {
            return;
        }
        if let Some(prefixes) = &self.ok_prefix {
            for prefix in prefixes {
                if length >= prefix.len() && word[..prefix.len()] == prefix[..] {
                    return;
                }
            }
        }
        word[0] = upper_unit(word[0]);
        for u in word[1..].iter_mut() {
            *u = lower_unit(*u);
        }
    }
}

impl<I: TokenStream> TokenFilter for CapitalizationFilter<I> {
    crate::filter_input!();
    // Java: CapitalizationFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let cfg = &self.cfg;
        with_utf16_term(self.input.attributes_mut(), &mut self.buf, |term| {
            let len = term.len();
            let backup = (cfg.max_word_count < CAPITALIZATION_DEFAULT_MAX).then(|| term.clone());
            if len >= usize::try_from(cfg.max_token_length).unwrap_or(usize::MAX) {
                return false;
            }
            let mut word_count = 0i32;
            let mut last_word_start = 0usize;
            let mut i = 0usize;
            while i < len {
                let c = term[i];
                if c <= u16::from(b' ') || c == u16::from(b'.') {
                    let wlen = i - last_word_start;
                    if wlen > 0 {
                        cfg.process_word(term, last_word_start, wlen, word_count);
                        word_count += 1;
                        last_word_start = i + 1;
                        i += 1;
                    }
                }
                i += 1;
            }
            if last_word_start < len {
                cfg.process_word(term, last_word_start, len - last_word_start, word_count);
                word_count += 1;
            }
            if word_count > cfg.max_word_count {
                if let Some(b) = backup {
                    *term = b;
                }
            }
            true
        });
        Ok(true)
    }
}

const AA: u16 = 0x00C5; // Å
const AA_LOWER: u16 = 0x00E5; // å
const AE: u16 = 0x00C6; // Æ
const AE_LOWER: u16 = 0x00E6; // æ
const AE_SE: u16 = 0x00C4; // Ä
const AE_SE_LOWER: u16 = 0x00E4; // ä
const OE: u16 = 0x00D8; // Ø
const OE_LOWER: u16 = 0x00F8; // ø
const OE_SE: u16 = 0x00D6; // Ö
const OE_SE_LOWER: u16 = 0x00F6; // ö

fn is(u: u16, set: &[u8]) -> bool {
    set.iter().any(|&b| u == u16::from(b))
}

/// `org.apache.lucene.analysis.miscellaneous.ScandinavianFoldingFilter`.
pub struct ScandinavianFoldingFilter<I> {
    input: I,
    buf: Vec<u16>,
}

impl<I: TokenStream> ScandinavianFoldingFilter<I> {
    /// `new ScandinavianFoldingFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        ScandinavianFoldingFilter {
            input,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for ScandinavianFoldingFilter<I> {
    crate::filter_input!();
    // Java: ScandinavianFoldingFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
            let mut i = 0;
            while i < b.len() {
                let c = b[i];
                if c == AA_LOWER || c == AE_SE_LOWER || c == AE_LOWER {
                    b[i] = u16::from(b'a');
                } else if c == AA || c == AE_SE || c == AE {
                    b[i] = u16::from(b'A');
                } else if c == OE_LOWER || c == OE_SE_LOWER {
                    b[i] = u16::from(b'o');
                } else if c == OE || c == OE_SE {
                    b[i] = u16::from(b'O');
                } else if b.len() - 1 > i {
                    let next = b[i + 1];
                    if (is(c, b"aA") && is(next, b"aAeEoO")) || (is(c, b"oO") && is(next, b"eEoO"))
                    {
                        // StemmerUtil.delete(buffer, i + 1, length)
                        b.remove(i + 1);
                    }
                }
                i += 1;
            }
            true
        });
        Ok(true)
    }
}

/// `ScandinavianNormalizer.Foldings`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Folding {
    /// `aa`/`aA` -> `å`.
    Aa,
    /// `ao`/`aO` -> `å`.
    Ao,
    /// `ae`/`aE` -> `æ`.
    Ae,
    /// `oe`/`oE` -> `ø`.
    Oe,
    /// `oo`/`oO` -> `ø`.
    Oo,
}

/// `org.apache.lucene.analysis.miscellaneous.ScandinavianNormalizer`.
#[derive(Debug, Clone)]
pub struct ScandinavianNormalizer {
    foldings: Vec<Folding>,
}

impl ScandinavianNormalizer {
    /// `ScandinavianNormalizer.ALL_FOLDINGS`.
    pub const ALL_FOLDINGS: [Folding; 5] = [
        Folding::Aa,
        Folding::Ao,
        Folding::Ae,
        Folding::Oe,
        Folding::Oo,
    ];

    /// `new ScandinavianNormalizer(Set<Foldings>)`.
    pub fn new(foldings: &[Folding]) -> Self {
        ScandinavianNormalizer {
            foldings: foldings.to_vec(),
        }
    }

    fn has(&self, f: Folding) -> bool {
        self.foldings.contains(&f)
    }

    /// `processToken(char[], int)`: normalises `buffer` in place.
    pub fn process_token(&self, b: &mut Vec<u16>) {
        let mut i = 0;
        while i < b.len() {
            let c = b[i];
            if c == AE_SE_LOWER {
                b[i] = AE_LOWER;
            } else if c == AE_SE {
                b[i] = AE;
            } else if c == OE_SE_LOWER {
                b[i] = OE_LOWER;
            } else if c == OE_SE {
                b[i] = OE;
            } else if b.len() - 1 > i {
                let n = b[i + 1];
                let aa_ao = (self.has(Folding::Aa) && is(n, b"aA"))
                    || (self.has(Folding::Ao) && is(n, b"oO"));
                let ae = self.has(Folding::Ae) && is(n, b"eE");
                let oe_oo = (self.has(Folding::Oe) && is(n, b"eE"))
                    || (self.has(Folding::Oo) && is(n, b"oO"));
                let replacement = if c == u16::from(b'a') && aa_ao {
                    Some(AA_LOWER)
                } else if c == u16::from(b'A') && aa_ao {
                    Some(AA)
                } else if c == u16::from(b'a') && ae {
                    Some(AE_LOWER)
                } else if c == u16::from(b'A') && ae {
                    Some(AE)
                } else if c == u16::from(b'o') && oe_oo {
                    Some(OE_LOWER)
                } else if c == u16::from(b'O') && oe_oo {
                    Some(OE)
                } else {
                    None
                };
                if let Some(r) = replacement {
                    b.remove(i + 1);
                    b[i] = r;
                }
            }
            i += 1;
        }
    }
}

/// `org.apache.lucene.analysis.miscellaneous.ScandinavianNormalizationFilter`:
/// [`ScandinavianNormalizer`] with every folding.
pub struct ScandinavianNormalizationFilter<I> {
    input: I,
    normalizer: ScandinavianNormalizer,
    buf: Vec<u16>,
}

impl<I: TokenStream> ScandinavianNormalizationFilter<I> {
    /// `new ScandinavianNormalizationFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        ScandinavianNormalizationFilter {
            input,
            normalizer: ScandinavianNormalizer::new(&ScandinavianNormalizer::ALL_FOLDINGS),
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for ScandinavianNormalizationFilter<I> {
    crate::filter_input!();
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let n = &self.normalizer;
        with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
            n.process_token(b);
            true
        });
        Ok(true)
    }
}

/// `ArrayUtil.parseInt(char[], int, int)` (radix 10): Java's
/// `NumberFormatException` is an `IllegalArgument` whose message starts
/// with `NumberFormatException`.
pub(crate) fn parse_int(chars: &[u16]) -> Result<i32, AnalysisError> {
    let nfe = |m: &str| AnalysisError::IllegalArgument(format!("NumberFormatException: {m}"));
    if chars.is_empty() {
        return Err(nfe("chars length is 0"));
    }
    let negative = chars[0] == u16::from(b'-');
    let digits = if negative {
        if chars.len() == 1 {
            return Err(nfe("can't convert to an int"));
        }
        &chars[1..]
    } else {
        chars
    };
    let max = i32::MIN / 10;
    let mut result: i32 = 0;
    for &c in digits {
        let Some(digit) = java_character::decimal_digit_value(u32::from(c)) else {
            return Err(nfe("Unable to parse"));
        };
        if max > result {
            return Err(nfe("Unable to parse"));
        }
        let next = result.wrapping_mul(10).wrapping_sub(digit as i32);
        if next > result {
            return Err(nfe("Unable to parse"));
        }
        result = next;
    }
    if !negative {
        result = result.wrapping_neg();
        if result < 0 {
            return Err(nfe("Unable to parse"));
        }
    }
    Ok(result)
}

/// `DelimitedTermFrequencyTokenFilter.DEFAULT_DELIMITER`.
pub const DEFAULT_TERM_FREQUENCY_DELIMITER: char = '|';

/// `org.apache.lucene.analysis.miscellaneous.DelimitedTermFrequencyTokenFilter`:
/// `term|freq` sets the term frequency.
pub struct DelimitedTermFrequencyTokenFilter<I> {
    input: I,
    delimiter: u16,
    buf: Vec<u16>,
}

impl<I: TokenStream> DelimitedTermFrequencyTokenFilter<I> {
    /// `new DelimitedTermFrequencyTokenFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        Self::with_delimiter(input, DEFAULT_TERM_FREQUENCY_DELIMITER as u16)
    }

    /// `new DelimitedTermFrequencyTokenFilter(TokenStream, char)`.
    pub fn with_delimiter(input: I, delimiter: u16) -> Self {
        DelimitedTermFrequencyTokenFilter {
            input,
            delimiter,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for DelimitedTermFrequencyTokenFilter<I> {
    crate::filter_input!();
    // Java: DelimitedTermFrequencyTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let mut freq = None;
        let d = self.delimiter;
        with_utf16_term(self.input.attributes_mut(), &mut self.buf, |b| {
            match b.iter().position(|&u| u == d) {
                Some(i) => {
                    freq = Some(parse_int(&b[i + 1..]));
                    b.truncate(i);
                    true
                }
                None => false,
            }
        });
        if let Some(freq) = freq {
            self.input.attributes_mut().set_term_frequency(freq?)?;
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    fn terms(ts: &mut dyn TokenStream) -> Vec<String> {
        let mut out = Vec::new();
        crate::token_stream::consume(ts, |a| out.push(a.term().to_string())).unwrap();
        out
    }

    #[test]
    fn trim_and_truncate() {
        let mut f = TrimFilter::new(Canned::parse("x:0:1:1:1|1|0"));
        assert_eq!(render(&mut f), "x:0:1:1:1|1|0");
        let mut c = Canned::parse("a:0:1:1:1 b:0:1:1:1 c:0:1:1:1");
        c.set_terms(&[" \t ab ", "   ", ""]);
        assert_eq!(terms(&mut TrimFilter::new(c)), vec!["ab", "", ""]);
        let mut f = TruncateTokenFilter::truncate_after_chars(
            Canned::parse("abcdef:0:6:1:1 ab:7:9:1:1"),
            3,
        )
        .unwrap();
        assert_eq!(terms(&mut f), vec!["abc", "ab"]);
        let mut f = TruncateTokenFilter::truncate_after_code_points(
            Canned::parse("😀😀😀:0:6:1:1 a😀:7:9:1:1 😀:9:11:1:1"),
            2,
        )
        .unwrap();
        assert_eq!(terms(&mut f), vec!["😀😀", "a😀", "😀"]);
        let mut c = Canned::parse("abcdef:0:6:1:1");
        c.set_keywords(&[true]);
        let mut f = TruncateTokenFilter::truncate_after_chars(c, 2).unwrap();
        assert_eq!(terms(&mut f), vec!["abcdef"]);
        assert!(TruncateTokenFilter::truncate_after_chars(Canned::parse(""), 0).is_err());
    }

    #[test]
    fn capitalization_options() {
        let mut c = Canned::parse("x:0:1:1:1 y:0:1:1:1");
        c.set_terms(&["hELLO wORLD.again", "mcdonald the big"]);
        assert_eq!(
            terms(&mut CapitalizationFilter::new(c)),
            vec!["Hello world.again", "Mcdonald the big"]
        );
        let mut c = Canned::parse("x:0:1:1:1 y:0:1:1:1 z:0:1:1:1");
        c.set_terms(&["the quick", "mcdonald a", "one two three"]);
        let keep = Arc::new(CharArraySet::from_words(["the"], true));
        let mut f = CapitalizationFilter::with_options(
            c,
            false,
            Some(keep),
            true,
            Some(vec!["mc".into()]),
            2,
            2,
            100,
        )
        .unwrap();
        assert_eq!(
            terms(&mut f),
            vec!["The Quick", "mcdonald a", "one two three"]
        );
        let mut c = Canned::parse("abcdef:0:6:1:1");
        c.set_terms(&["the"]);
        let keep = Arc::new(CharArraySet::from_words(["the"], true));
        let mut f =
            CapitalizationFilter::with_options(c, true, Some(keep), false, None, 0, 9, 2).unwrap();
        assert_eq!(
            terms(&mut f),
            vec!["the"],
            "longer than maxTokenLength: untouched"
        );
        for (min, words, len) in [(-1, 1, 1), (0, 0, 1), (0, 1, 0)] {
            assert!(CapitalizationFilter::with_options(
                Canned::parse(""),
                true,
                None,
                true,
                None,
                min,
                words,
                len
            )
            .is_err());
        }
    }

    #[test]
    fn scandinavian() {
        let mut c = Canned::parse("x:0:1:1:1 y:0:1:1:1 z:0:1:1:1");
        c.set_terms(&["Blåbærsyltetøy", "räksmörgås", "AAOEaO"]);
        assert_eq!(
            terms(&mut ScandinavianFoldingFilter::new(c)),
            vec!["Blabarsyltetoy", "raksmorgas", "AOa"]
        );
        let mut c = Canned::parse("x:0:1:1:1 y:0:1:1:1 z:0:1:1:1");
        c.set_terms(&["räksmörgås", "Aaoe AE aO OO", "x"]);
        assert_eq!(
            terms(&mut ScandinavianNormalizationFilter::new(c)),
            vec!["ræksmørgås", "Åø Æ å Ø", "x"]
        );
        let mut b: Vec<u16> = "aaoo".encode_utf16().collect();
        ScandinavianNormalizer::new(&[Folding::Ae]).process_token(&mut b);
        assert_eq!(String::from_utf16(&b).unwrap(), "aaoo");
    }

    #[test]
    fn term_frequencies() {
        let mut c = Canned::parse("x:0:1:1:1 y:0:1:1:1");
        c.set_terms(&["a|3", "b"]);
        let mut f = DelimitedTermFrequencyTokenFilter::new(c);
        let mut out = Vec::new();
        f.reset().unwrap();
        while f.increment_token().unwrap() {
            out.push((
                f.attributes().term().to_string(),
                f.attributes().term_frequency(),
            ));
        }
        assert_eq!(out, [("a".to_string(), 3), ("b".to_string(), 1)]);
        let to16 = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        assert_eq!(parse_int(&to16("٣٤")).unwrap(), 34);
        assert_eq!(parse_int(&to16("-12")).unwrap(), -12);
        assert_eq!(parse_int(&to16("2147483647")).unwrap(), i32::MAX);
        assert_eq!(parse_int(&to16("-2147483648")).unwrap(), i32::MIN);
        for bad in ["", "-", "1.5", "2147483648", "99999999999"] {
            let e = parse_int(&to16(bad)).unwrap_err().to_string();
            assert!(e.contains("NumberFormatException"), "{bad}: {e}");
        }
        let mut c = Canned::parse("x:0:1:1:1");
        c.set_terms(&["a/0"]);
        let mut f = DelimitedTermFrequencyTokenFilter::with_delimiter(c, u16::from(b'/'));
        f.reset().unwrap();
        assert!(f.increment_token().is_err(), "frequency 0 is rejected");
    }
}
