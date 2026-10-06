//! `org.apache.lucene.analysis.cjk`: `CJKBigramFilter`, `CJKWidthFilter`,
//! `CJKWidthCharFilter` and `CJKAnalyzer`.

use std::sync::Arc;

use crate::analyzer::{AnalyzerDefinition, TokenStreamComponents};
use crate::attributes::State;
use crate::charfilter::OffsetCorrections;
use crate::java_character::{char_count, code_point_at, push_utf16};
use crate::reader::{CharFilter, CharReader};
use crate::standard::{HANGUL, HIRAGANA, IDEOGRAPHIC, KATAKANA, TOKEN_TYPES};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::with_utf16_term;
use crate::{
    AnalysisError, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter,
    StopwordAnalyzerBase,
};

/// `CJKBigramFilter.HAN`.
pub const HAN: i32 = 1;
/// `CJKBigramFilter.HIRAGANA`.
pub const HIRAGANA_FLAG: i32 = 2;
/// `CJKBigramFilter.KATAKANA`.
pub const KATAKANA_FLAG: i32 = 4;
/// `CJKBigramFilter.HANGUL`.
pub const HANGUL_FLAG: i32 = 8;
/// `CJKBigramFilter.DOUBLE_TYPE`.
pub const DOUBLE_TYPE: &str = "<DOUBLE>";
/// `CJKBigramFilter.SINGLE_TYPE`.
pub const SINGLE_TYPE: &str = "<SINGLE>";

/// `org.apache.lucene.analysis.cjk.CJKBigramFilter`: overlapping bigrams of
/// the CJK tokens `StandardTokenizer` emits one character at a time.
pub struct CJKBigramFilter<I> {
    input: I,
    /// The token types bigrammed (Java's `doHan`... fields).
    types: Vec<&'static str>,
    output_unigrams: bool,
    ngram_state: bool,
    buffer: Vec<u32>,
    start_offset: Vec<i32>,
    end_offset: Vec<i32>,
    index: usize,
    last_end_offset: i32,
    exhausted: bool,
    lone_state: Option<State>,
    scratch: Vec<u16>,
}

impl<I: TokenStream> CJKBigramFilter<I> {
    /// `new CJKBigramFilter(TokenStream, int flags, boolean outputUnigrams)`.
    pub fn new(input: I, flags: i32, output_unigrams: bool) -> Self {
        let mut types = Vec::new();
        for (flag, ty) in [
            (HAN, IDEOGRAPHIC),
            (HIRAGANA_FLAG, HIRAGANA),
            (KATAKANA_FLAG, KATAKANA),
            (HANGUL_FLAG, HANGUL),
        ] {
            if flags & flag != 0 {
                types.push(TOKEN_TYPES[ty]);
            }
        }
        CJKBigramFilter {
            input,
            types,
            output_unigrams,
            ngram_state: false,
            buffer: Vec::new(),
            start_offset: Vec::new(),
            end_offset: Vec::new(),
            index: 0,
            last_end_offset: 0,
            exhausted: false,
            lone_state: None,
            scratch: Vec::new(),
        }
    }

    /// `new CJKBigramFilter(TokenStream)`: every script, bigrams only.
    pub fn all(input: I) -> Self {
        Self::new(
            input,
            HAN | HIRAGANA_FLAG | KATAKANA_FLAG | HANGUL_FLAG,
            false,
        )
    }

    fn has_buffered_bigram(&self) -> bool {
        self.buffer.len() > self.index + 1
    }

    fn has_buffered_unigram(&self) -> bool {
        if self.output_unigrams {
            self.buffer.len() == self.index + 1
        } else {
            self.buffer.len() == 1 && self.index == 0
        }
    }

    // Java: CJKBigramFilter.doNext
    fn do_next(&mut self) -> Result<bool, AnalysisError> {
        if let Some(s) = self.lone_state.take() {
            self.input.attributes_mut().restore_state(&s);
            return Ok(true);
        }
        if self.exhausted {
            return Ok(false);
        }
        if self.input.increment_token()? {
            return Ok(true);
        }
        self.exhausted = true;
        Ok(false)
    }

    // Java: CJKBigramFilter.refill
    fn refill(&mut self) {
        if self.buffer.len() > 64 {
            let last = self.buffer.len() - 1;
            self.buffer.swap(0, last);
            self.start_offset.swap(0, last);
            self.end_offset.swap(0, last);
            self.buffer.truncate(1);
            self.start_offset.truncate(1);
            self.end_offset.truncate(1);
            self.index -= last;
        }
        let a = self.input.attributes();
        self.scratch.clear();
        self.scratch.extend(a.term().encode_utf16());
        let len = self.scratch.len();
        let mut start = a.start_offset();
        let end = a.end_offset();
        self.last_end_offset = end;
        // Java: offsets are spread per code point only when the token's
        // offsets span exactly its length.
        let spread = end - start == len as i32;
        let mut i = 0;
        while i < len {
            let cp = code_point_at(&self.scratch, i, len);
            let cp_len = char_count(cp);
            self.buffer.push(cp);
            self.start_offset.push(start);
            if spread {
                start += cp_len as i32;
                self.end_offset.push(start);
            } else {
                self.end_offset.push(end);
            }
            i += cp_len;
        }
    }

    fn set_term(&mut self, cps: &[u32]) {
        self.scratch.clear();
        for &cp in cps {
            push_utf16(&mut self.scratch, cp);
        }
        self.input.attributes_mut().set_term_utf16(&self.scratch);
    }

    // Java: CJKBigramFilter.flushBigram
    fn flush_bigram(&mut self) -> Result<(), AnalysisError> {
        self.input.attributes_mut().clear_attributes();
        let i = self.index;
        let pair = [self.buffer[i], self.buffer[i + 1]];
        self.set_term(&pair);
        let a = self.input.attributes_mut();
        a.set_offset(self.start_offset[i], self.end_offset[i + 1])?;
        a.set_token_type(DOUBLE_TYPE);
        if self.output_unigrams {
            a.set_position_increment(0)?;
            a.set_position_length(2)?;
        }
        self.index += 1;
        Ok(())
    }

    // Java: CJKBigramFilter.flushUnigram
    fn flush_unigram(&mut self) -> Result<(), AnalysisError> {
        self.input.attributes_mut().clear_attributes();
        let i = self.index;
        let one = [self.buffer[i]];
        self.set_term(&one);
        let a = self.input.attributes_mut();
        a.set_offset(self.start_offset[i], self.end_offset[i])?;
        a.set_token_type(SINGLE_TYPE);
        self.index += 1;
        Ok(())
    }

    fn clear_buffer(&mut self) {
        self.index = 0;
        self.buffer.clear();
        self.start_offset.clear();
        self.end_offset.clear();
    }
}

impl<I: TokenStream> TokenFilter for CJKBigramFilter<I> {
    crate::filter_input!();

    // Java: CJKBigramFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            if self.has_buffered_bigram() {
                if self.output_unigrams {
                    if self.ngram_state {
                        self.flush_bigram()?;
                    } else {
                        self.flush_unigram()?;
                        self.index -= 1;
                    }
                    self.ngram_state = !self.ngram_state;
                } else {
                    self.flush_bigram()?;
                }
                return Ok(true);
            } else if self.do_next()? {
                let a = self.input.attributes();
                if self.types.contains(&a.token_type()) {
                    if a.start_offset() != self.last_end_offset {
                        if self.has_buffered_unigram() {
                            self.lone_state = Some(a.capture_state());
                            self.flush_unigram()?;
                            return Ok(true);
                        }
                        self.clear_buffer();
                    }
                    self.refill();
                } else {
                    if self.has_buffered_unigram() {
                        self.lone_state = Some(a.capture_state());
                        self.flush_unigram()?;
                        return Ok(true);
                    }
                    return Ok(true);
                }
            } else {
                if self.has_buffered_unigram() {
                    self.flush_unigram()?;
                    return Ok(true);
                }
                return Ok(false);
            }
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.clear_buffer();
        self.last_end_offset = 0;
        self.lone_state = None;
        self.exhausted = false;
        self.ngram_state = false;
        Ok(())
    }
}

/// `CJKWidthFilter.KANA_NORM`: half-width katakana U+FF65..=U+FF9F to full width.
const KANA_NORM: [u16; 59] = [
    0x30fb, 0x30f2, 0x30a1, 0x30a3, 0x30a5, 0x30a7, 0x30a9, 0x30e3, 0x30e5, 0x30e7, 0x30c3, 0x30fc,
    0x30a2, 0x30a4, 0x30a6, 0x30a8, 0x30aa, 0x30ab, 0x30ad, 0x30af, 0x30b1, 0x30b3, 0x30b5, 0x30b7,
    0x30b9, 0x30bb, 0x30bd, 0x30bf, 0x30c1, 0x30c4, 0x30c6, 0x30c8, 0x30ca, 0x30cb, 0x30cc, 0x30cd,
    0x30ce, 0x30cf, 0x30d2, 0x30d5, 0x30d8, 0x30db, 0x30de, 0x30df, 0x30e0, 0x30e1, 0x30e2, 0x30e4,
    0x30e6, 0x30e8, 0x30e9, 0x30ea, 0x30eb, 0x30ec, 0x30ed, 0x30ef, 0x30f3, 0x3099, 0x309A,
];

/// `KANA_COMBINE_VOICED`, indexed from U+30A6.
const KANA_COMBINE_VOICED: &[u8] = &[
    78, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1,
    0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 8, 8, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
];

/// `KANA_COMBINE_HALF_VOICED`, indexed from U+30A6.
const KANA_COMBINE_HALF_VOICED: &[u8] = &[
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
];

const VOICED_MARK: u16 = 0xFF9E;
const SEMI_VOICED_MARK: u16 = 0xFF9F;

/// The voiced (or semi-voiced) form of `prev`, or `prev` itself.
fn combine_voice_mark(prev: u16, mark: u16) -> u16 {
    if (0x30A6..=0x30FD).contains(&prev) {
        let i = usize::from(prev - 0x30A6);
        let add = if mark == SEMI_VOICED_MARK {
            KANA_COMBINE_HALF_VOICED[i]
        } else {
            KANA_COMBINE_VOICED[i]
        };
        return prev + u16::from(add);
    }
    prev
}

/// `org.apache.lucene.analysis.cjk.CJKWidthFilter`: full-width ASCII to
/// ASCII, half-width katakana to full width (combining voice marks).
pub struct CJKWidthFilter<I> {
    input: I,
    buf: Vec<u16>,
}

impl<I: TokenStream> CJKWidthFilter<I> {
    /// `new CJKWidthFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        CJKWidthFilter {
            input,
            buf: Vec::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for CJKWidthFilter<I> {
    crate::filter_input!();
    // Java: CJKWidthFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        with_utf16_term(self.input.attributes_mut(), &mut self.buf, |text| {
            let mut changed = false;
            let mut i = 0;
            while i < text.len() {
                let ch = text[i];
                if (0xFF01..=0xFF5E).contains(&ch) {
                    text[i] = ch - 0xFEE0;
                    changed = true;
                } else if (0xFF65..=0xFF9F).contains(&ch) {
                    changed = true;
                    let combined = (ch == VOICED_MARK || ch == SEMI_VOICED_MARK)
                        && i > 0
                        && combine_voice_mark(text[i - 1], ch) != text[i - 1];
                    if combined {
                        text[i - 1] = combine_voice_mark(text[i - 1], ch);
                        text.remove(i);
                        continue;
                    }
                    text[i] = KANA_NORM[usize::from(ch - 0xFF65)];
                }
                i += 1;
            }
            changed
        });
        Ok(true)
    }
}

/// `org.apache.lucene.analysis.cjk.CJKWidthCharFilter`: [`CJKWidthFilter`]'s
/// folding before tokenization, with offset corrections.
pub struct CJKWidthCharFilter<R> {
    input: R,
    prev_char: Option<u16>,
    input_off: i32,
    corrections: OffsetCorrections,
    one: [u16; 1],
}

impl<R: CharReader> CJKWidthCharFilter<R> {
    /// `new CJKWidthCharFilter(Reader)`.
    pub fn new(input: R) -> Self {
        CJKWidthCharFilter {
            input,
            prev_char: None,
            input_off: 0,
            corrections: OffsetCorrections::default(),
            one: [0],
        }
    }

    fn input_read(&mut self) -> Result<Option<u16>, AnalysisError> {
        let n = self.input.read(&mut self.one)?;
        Ok((n == 1).then_some(self.one[0]))
    }

    // Java: CJKWidthCharFilter.read()
    fn read_one(&mut self) -> Result<Option<u16>, AnalysisError> {
        loop {
            let Some(ch) = self.input_read()? else {
                return Ok(self.prev_char.take());
            };
            self.input_off += 1;
            if ch == SEMI_VOICED_MARK || ch == VOICED_MARK {
                if let Some(prev) = self.prev_char {
                    let combined = combine_voice_mark(prev, ch);
                    if prev != combined {
                        self.prev_char = None;
                        let prev_diff = self.corrections.last_cumulative_diff();
                        self.corrections
                            .add(self.input_off - 1 - prev_diff, prev_diff + 1);
                        return Ok(Some(combined));
                    }
                }
            }
            let ret = self.prev_char;
            self.prev_char = Some(if (0xFF01..=0xFF5E).contains(&ch) {
                ch - 0xFEE0
            } else if (0xFF65..=0xFF9F).contains(&ch) {
                KANA_NORM[usize::from(ch - 0xFF65)]
            } else {
                ch
            });
            if ret.is_some() {
                return Ok(ret);
            }
        }
    }
}

impl<R: CharReader> CharFilter for CJKWidthCharFilter<R> {
    fn input(&self) -> &dyn CharReader {
        &self.input
    }

    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut self.input
    }

    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let mut n = 0;
        while n < buf.len() {
            match self.read_one()? {
                Some(c) => {
                    buf[n] = c;
                    n += 1;
                }
                None => break,
            }
        }
        Ok(n)
    }

    fn correct(&self, current_off: i32) -> i32 {
        self.corrections.correct(current_off)
    }
}

/// `cjk/stopwords.txt` from the Lucene 10.5.0 jar (Apache-2.0).
pub const CJK_STOP_WORDS: &[&str] = &[
    "a", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it", "no",
    "not", "of", "on", "or", "s", "such", "t", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with", "www",
];

/// `org.apache.lucene.analysis.cjk.CJKAnalyzer`: `StandardTokenizer`,
/// `CJKWidthFilter`, `LowerCaseFilter`, `CJKBigramFilter`, `StopFilter`.
#[derive(Debug, Clone)]
pub struct CJKAnalyzer {
    base: StopwordAnalyzerBase,
}

impl Default for CJKAnalyzer {
    /// `new CJKAnalyzer()`: [`CJK_STOP_WORDS`].
    fn default() -> Self {
        Self::new(CharArraySet::from_words(CJK_STOP_WORDS, false))
    }
}

impl CJKAnalyzer {
    /// `new CJKAnalyzer(CharArraySet)`.
    pub fn new(stopwords: CharArraySet) -> Self {
        CJKAnalyzer {
            base: StopwordAnalyzerBase::new(Some(stopwords)),
        }
    }
}

impl AnalyzerDefinition for CJKAnalyzer {
    fn create_components(&self, _field: &str) -> Result<TokenStreamComponents, AnalysisError> {
        let result = CJKBigramFilter::all(LowerCaseFilter::new(CJKWidthFilter::new(
            StandardTokenizer::new(),
        )));
        Ok(TokenStreamComponents::new(StopFilter::new(
            result,
            Arc::clone(self.base.stopword_set()),
        )))
    }

    fn normalize(&self, _field: &str, input: Box<dyn TokenStream>) -> Box<dyn TokenStream> {
        Box::new(LowerCaseFilter::new(CJKWidthFilter::new(input)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::util::canned::{render, Canned};
    use crate::Analyzer;

    fn ideo(spec: &str, types: &[&'static str]) -> Canned {
        let mut c = Canned::parse(spec);
        c.set_types(types);
        c
    }

    #[test]
    fn bigrams() {
        let i = TOKEN_TYPES[IDEOGRAPHIC];
        let c = ideo(
            "一:0:1:1:1 二:1:2:1:1 三:2:3:1:1 x:4:5:1:1 四:6:7:1:1|7|0",
            &[i, i, i, "<ALPHANUM>", i],
        );
        let mut f = CJKBigramFilter::all(c);
        assert_eq!(
            render(&mut f),
            "一二:0:2:1:1 二三:1:3:1:1 x:4:5:1:1 四:6:7:1:1|7|0"
        );
        let c = ideo("一:0:1:1:1 二:1:2:1:1 四:6:7:1:1|7|0", &[i, i, i]);
        let mut f = CJKBigramFilter::new(c, HAN, true);
        assert_eq!(
            render(&mut f),
            "一:0:1:1:1 一二:0:2:0:2 二:1:2:1:1 四:6:7:1:1|7|0"
        );
        let c = ideo("一二:0:5:1:1|5|0", &[i]);
        let mut f = CJKBigramFilter::all(c);
        assert_eq!(render(&mut f), "一二:0:5:1:1|5|0");
        let h = TOKEN_TYPES[HIRAGANA];
        let c = ideo("あ:0:1:1:1|1|0", &[h]);
        let mut f = CJKBigramFilter::new(c, HAN, false);
        assert_eq!(render(&mut f), "あ:0:1:1:1|1|0");
    }

    #[test]
    fn width_filter_and_char_filter() {
        let mut c = Canned::parse("x:0:1:1:1");
        c.set_terms(&["ＡＢｃ１ｶﾞｷﾟﾞｱ"]);
        let mut out = Vec::new();
        crate::token_stream::consume(&mut CJKWidthFilter::new(c), |a| {
            out.push(a.term().to_string())
        })
        .unwrap();
        assert_eq!(out, vec!["ABc1ガキ\u{309A}\u{3099}ア"]);
        let mut f = CJKWidthCharFilter::new(StrReader::new("ｶﾞＡ"));
        let mut buf = [0u16; 8];
        let n = f.read(&mut buf).unwrap();
        assert_eq!(String::from_utf16(&buf[..n]).unwrap(), "ガA");
        assert_eq!((f.correct_offset(1), f.correct_offset(2)), (2, 3));
    }

    #[test]
    fn analyzer() {
        let a = Analyzer::new(CJKAnalyzer::default());
        let t: Vec<String> = a
            .analyze("日本語 the ＡＢＣ")
            .into_iter()
            .map(|t| t.term)
            .collect();
        assert_eq!(t, vec!["日本", "本語", "abc"]);
        assert_eq!(a.normalize("f", "ＡＢ").unwrap(), b"ab");
    }
}
