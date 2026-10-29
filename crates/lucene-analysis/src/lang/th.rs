//! `org.apache.lucene.analysis.th`: `ThaiTokenizer` and `ThaiAnalyzer` --
//! **not supported by design**: they refuse with the
//! `UnsupportedOperationException` Lucene raises on a JRE without Thai
//! dictionary segmentation.
//!
//! `ThaiTokenizer` splits sentences with the JDK's sentence iterator
//! ([`crate::util::sentence_break`], ported) and words with
//! `BreakIterator.getWordInstance(th)`, the JDK's dictionary-based iterator.
//! Its output is a function of the JDK's Thai word list (`thai_dict`, in
//! `java.base`), which is GPL-2.0-with-Classpath-Exception data: it cannot be
//! vendored into this Apache-2.0 library, nor re-derived from the JDK's
//! answers without copying it, and no other dictionary (ICU's, LibThai's)
//! splits Thai the same way. So the port does what Lucene does on a JRE whose
//! word iterator does not segment Thai: [`DBBI_AVAILABLE`] is `false`, and
//! constructing the tokenizer -- directly, through [`ThaiAnalyzer`] or the
//! `thai` factory -- is an [`AnalysisError::UnsupportedOperation`] with
//! Java's message. `ThaiAnalyzer`'s stop set and `normalize` chain need no
//! dictionary and work. Thai text can be segmented with ICU's tokenizer
//! (`analysis-icu`, M12 T12.4) instead, as Lucene's own documentation
//! suggests.
//!
//! Lucene 10.6 (branch_10_6, not in the pinned 10.5.2 jars) adds Thai
//! components that need no dictionary, ported here as standalone pieces:
//! [`ThaiNormalizer`] and [`ThaiNormalizationFilter`], [`ThaiCharFilter`]
//! (#16717) and [`ThaiRepeatFilter`] (#16720). 10.6 also puts them in
//! `ThaiAnalyzer`'s chain and replaces its stop set (#16717, #16718); those
//! change the analyzer's output and wait for the 10.6.0 release, so
//! [`ThaiAnalyzer`] stays 10.5.2's.

use std::sync::{Arc, LazyLock};

use crate::attributes::StateSlot;
use crate::charfilter::OffsetCorrections;
use crate::core_analysis::DecimalDigitFilter;
use crate::reader::{CharFilter, CharReader};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::stemmer_util::delete;
use crate::{AnalysisError, CharArraySet, LowerCaseFilter};

use super::{comment_set, CharStemmer, StemFilter};

/// `ThaiTokenizer.DBBI_AVAILABLE`: whether the JRE's word iterator
/// segments Thai with a dictionary -- never, here (see the module docs).
pub const DBBI_AVAILABLE: bool = false;

/// The message of `ThaiTokenizer`'s `UnsupportedOperationException`.
pub const UNSUPPORTED: &str = "This JRE does not have support for Thai segmentation";

/// `ThaiAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/th_stopwords.txt")));

/// `ThaiTokenizer`: never constructed (see the module docs).
#[derive(Debug)]
pub struct ThaiTokenizer {
    _never: (),
}

impl ThaiTokenizer {
    /// `new ThaiTokenizer()`: Java's `UnsupportedOperationException` for a
    /// JRE without Thai segmentation.
    pub fn new() -> Result<Self, AnalysisError> {
        if DBBI_AVAILABLE {
            unreachable!("no Thai dictionary is shipped");
        }
        Err(AnalysisError::UnsupportedOperation(UNSUPPORTED.to_string()))
    }
}

/// `ThaiAnalyzer`: `ThaiTokenizer`, `LowerCaseFilter`,
/// `DecimalDigitFilter`, `StopFilter`. Its token streams refuse (see the
/// module docs); `normalize` works.
#[derive(Debug, Clone)]
pub struct ThaiAnalyzer {
    stopwords: Arc<CharArraySet>,
}

impl Default for ThaiAnalyzer {
    /// `new ThaiAnalyzer()`: the default stop set.
    fn default() -> Self {
        Self::new(&DEFAULT_STOP_SET)
    }
}

impl ThaiAnalyzer {
    /// `new ThaiAnalyzer(CharArraySet stopwords)`.
    pub fn new(stopwords: &CharArraySet) -> Self {
        ThaiAnalyzer {
            stopwords: super::copy_set(stopwords),
        }
    }

    /// `getStopwordSet()`.
    pub fn stopwords(&self) -> &CharArraySet {
        &self.stopwords
    }
}

impl crate::analyzer::AnalyzerDefinition for ThaiAnalyzer {
    fn create_components(
        &self,
        _field: &str,
    ) -> Result<crate::analyzer::TokenStreamComponents, AnalysisError> {
        let _ = ThaiTokenizer::new()?;
        unreachable!("ThaiTokenizer::new always refuses")
    }

    fn normalize(
        &self,
        _field: &str,
        input: Box<dyn crate::token_stream::TokenStream>,
    ) -> Box<dyn crate::token_stream::TokenStream> {
        Box::new(DecimalDigitFilter::new(LowerCaseFilter::new(input)))
    }
}

// ----------------------------------------------------------- ThaiNormalizer

/// `ThaiNormalizer.isToneMark`: Mai Ek .. Mai Chattawa.
fn is_tone_mark(c: u16) -> bool {
    (0x0E48..=0x0E4B).contains(&c)
}

/// `ThaiNormalizer.isToneMarkOrThanthakhat`.
fn is_tone_mark_or_thanthakhat(c: u16) -> bool {
    is_tone_mark(c) || c == 0x0E4C
}

/// `isAboveOrBelowVowel` (both classes; the char filter's
/// `isThaiAboveBelowVowel` is the same set).
fn is_above_or_below_vowel(c: u16) -> bool {
    c == 0x0E31 || (0x0E34..=0x0E3A).contains(&c) || c == 0x0E47
}

/// `ThaiNormalizer.isFollowVowel`.
fn is_follow_vowel(c: u16) -> bool {
    matches!(c, 0x0E30 | 0x0E32 | 0x0E33 | 0x0E45)
}

/// `ThaiNormalizer.isDanglingMark`.
fn is_dangling_mark(c: u16) -> bool {
    is_above_or_below_vowel(c) || is_tone_mark_or_thanthakhat(c) || c == 0x0E4D || c == 0x0E4E
}

/// `ThaiNormalizer.isDeduplicable`.
fn is_deduplicable(c: u16) -> bool {
    (0x0E30..=0x0E3A).contains(&c) || c == 0x0E45 || (0x0E47..=0x0E4E).contains(&c)
}

/// `org.apache.lucene.analysis.th.ThaiNormalizer` (Lucene 10.6, #16717):
/// removes ZWSP/ZWNJ and leading dangling marks, recomposes double Sara E
/// and decomposed Sara Am, maps Lakkhangyao to Sara Aa (not after Ru or
/// Lu), puts tone marks after above/below vowels and before follow vowels,
/// and collapses repeated vowels and marks.
#[derive(Debug, Default, Clone, Copy)]
pub struct ThaiNormalizer;

impl CharStemmer for ThaiNormalizer {
    // Java: ThaiNormalizer.normalize
    fn stem(&self, s: &mut Vec<u16>, mut len: usize) -> usize {
        if len == 0 {
            return 0;
        }

        // Step 1: remove zero-width characters
        let mut i = 0;
        while i < len {
            if s[i] == 0x200B || s[i] == 0x200C {
                len = delete(s, i, len);
            } else {
                i += 1;
            }
        }

        // remove leading dangling non-base marks
        while len > 0 && is_dangling_mark(s[0]) {
            len = delete(s, 0, len);
        }

        // Step 2: compositions (double Sara E, Sara Am, Lakkhangyao)
        let mut i = 0;
        while i < len {
            let c = s[i];
            // double Sara E -> Sara Ae
            if c == 0x0E40 && i + 1 < len && s[i + 1] == 0x0E40 {
                s[i] = 0x0E41;
                len = delete(s, i + 1, len);
                i += 1;
                continue;
            }
            // Nikhahit + optional tone mark + Sara Aa -> tone mark + Sara Am
            if c == 0x0E4D {
                if i + 1 < len && s[i + 1] == 0x0E32 {
                    s[i] = 0x0E33;
                    len = delete(s, i + 1, len);
                    i += 1;
                    continue;
                } else if i + 2 < len && is_tone_mark(s[i + 1]) && s[i + 2] == 0x0E32 {
                    s[i] = s[i + 1];
                    s[i + 1] = 0x0E33;
                    len = delete(s, i + 2, len);
                    i += 1;
                    continue;
                }
            }
            // tone mark + Nikhahit + Sara Aa -> tone mark + Sara Am
            if is_tone_mark(c) && i + 2 < len && s[i + 1] == 0x0E4D && s[i + 2] == 0x0E32 {
                s[i + 1] = 0x0E33;
                len = delete(s, i + 2, len);
                i += 1;
                continue;
            }
            // Lakkhangyao -> Sara Aa unless after Ru or Lu
            if c == 0x0E45 && (i == 0 || (s[i - 1] != 0x0E24 && s[i - 1] != 0x0E26)) {
                s[i] = 0x0E32;
            }
            i += 1;
        }

        // Step 3: canonical reordering
        let mut changed = true;
        while changed {
            changed = false;
            for i in 0..len.saturating_sub(1) {
                if (is_tone_mark_or_thanthakhat(s[i]) && is_above_or_below_vowel(s[i + 1]))
                    || (is_follow_vowel(s[i]) && is_tone_mark(s[i + 1]))
                {
                    s.swap(i, i + 1);
                    changed = true;
                }
            }
        }

        // Step 4: deduplicate repeated vowels, tone marks and diacritics
        let mut i = 0;
        while i + 1 < len {
            if is_deduplicable(s[i]) && s[i] == s[i + 1] {
                len = delete(s, i + 1, len);
            } else if is_tone_mark(s[i]) && is_tone_mark(s[i + 1]) {
                // two different tone marks: keep the last
                len = delete(s, i, len);
            } else {
                i += 1;
            }
        }
        len
    }
}

/// `org.apache.lucene.analysis.th.ThaiNormalizationFilter` (Lucene 10.6,
/// #16717): [`ThaiNormalizer`] over each term not marked a keyword.
pub type ThaiNormalizationFilter<I> = StemFilter<I, ThaiNormalizer>;

// ----------------------------------------------------------- ThaiCharFilter

/// `ThaiCharFilter.isThaiDiacritic`.
fn is_thai_diacritic(c: u16) -> bool {
    c == 0x0E31 || (0x0E34..=0x0E3A).contains(&c) || (0x0E47..=0x0E4E).contains(&c)
}

/// `ThaiCharFilter.isThaiTone`: the tone marks and Thanthakhat.
fn is_thai_tone(c: u16) -> bool {
    is_tone_mark_or_thanthakhat(c)
}

/// `org.apache.lucene.analysis.th.ThaiCharFilter` (Lucene 10.6, #16717):
/// before tokenization, removes ZWSP, ZWNJ, ZWJ and BOM, drops a diacritic
/// or tone mark repeating the previous output character, recomposes double
/// Sara E and decomposed Sara Am, swaps a tone mark before an above/below
/// vowel, and maps Lakkhangyao to Sara Aa (not after Ru or Lu), recording
/// an offset correction for every unit removed.
///
/// Differs: Java reads its input one `char` per `read()`; this reads it in
/// blocks into a staging buffer and takes one unit at a time from there,
/// which reads the same units in the same order.
pub struct ThaiCharFilter<R = Box<dyn CharReader>> {
    input: R,
    input_buffer: [u16; 8],
    in_buf_len: usize,
    eof: bool,
    output_off: i32,
    cumulative_diff: i32,
    /// `0` before the first output character, as in Java.
    prev_char: u16,
    /// `0` when none is pending, as in Java.
    pending_output_char: u16,
    corrections: OffsetCorrections,
    staged: Vec<u16>,
    staged_pos: usize,
    staged_len: usize,
}

impl<R: CharReader> ThaiCharFilter<R> {
    /// `new ThaiCharFilter(Reader)`.
    pub fn new(input: R) -> Self {
        ThaiCharFilter {
            input,
            input_buffer: [0; 8],
            in_buf_len: 0,
            eof: false,
            output_off: 0,
            cumulative_diff: 0,
            prev_char: 0,
            pending_output_char: 0,
            corrections: OffsetCorrections::default(),
            staged: Vec::new(),
            staged_pos: 0,
            staged_len: 0,
        }
    }

    /// `input.read()`: the next input unit, `None` at end of input.
    fn input_read(&mut self) -> Result<Option<u16>, AnalysisError> {
        if self.staged_pos == self.staged_len {
            if self.staged.is_empty() {
                self.staged = vec![0; 1024];
            }
            self.staged_len = self.input.read(&mut self.staged)?;
            self.staged_pos = 0;
            if self.staged_len == 0 {
                return Ok(None);
            }
        }
        let c = self.staged[self.staged_pos];
        self.staged_pos += 1;
        Ok(Some(c))
    }

    // Java: ThaiCharFilter.fillBuffer
    fn fill_buffer(&mut self) -> Result<(), AnalysisError> {
        while !self.eof && self.in_buf_len < 4 {
            match self.input_read()? {
                None => self.eof = true,
                Some(ch) => {
                    self.input_buffer[self.in_buf_len] = ch;
                    self.in_buf_len += 1;
                }
            }
        }
        Ok(())
    }

    // Java: ThaiCharFilter.shiftBuffer
    fn shift_buffer(&mut self, count: usize) {
        let remaining = self.in_buf_len - count;
        if remaining > 0 {
            self.input_buffer.copy_within(count..self.in_buf_len, 0);
        }
        self.in_buf_len = remaining;
    }

    /// `addOffCorrectMap(off, ++cumulativeDiff)`.
    fn drop_one(&mut self, off: i32) {
        self.cumulative_diff += 1;
        self.corrections.add(off, self.cumulative_diff);
    }

    /// Emits `c`: `prevChar = c; outputOff++; return c`.
    fn emit(&mut self, c: u16) -> Option<u16> {
        self.prev_char = c;
        self.output_off += 1;
        Some(c)
    }

    // Java: ThaiCharFilter.read()
    fn read_one(&mut self) -> Result<Option<u16>, AnalysisError> {
        if self.pending_output_char != 0 {
            let c = std::mem::take(&mut self.pending_output_char);
            return Ok(self.emit(c));
        }
        loop {
            self.fill_buffer()?;
            if self.in_buf_len == 0 {
                return Ok(None);
            }
            let mut c0 = self.input_buffer[0];
            let buf = self.input_buffer;
            let n = self.in_buf_len;

            // 1. zero-width character removal
            if matches!(c0, 0x200B | 0x200C | 0x200D | 0xFEFF) {
                self.shift_buffer(1);
                self.drop_one(self.output_off);
                continue;
            }
            // 2. consecutive identical diacritics / tone marks
            if is_thai_diacritic(c0) && c0 == self.prev_char {
                self.shift_buffer(1);
                self.drop_one(self.output_off);
                continue;
            }
            // 3. double Sara E -> Sara Ae
            if c0 == 0x0E40 && n >= 2 && buf[1] == 0x0E40 {
                self.shift_buffer(2);
                self.drop_one(self.output_off + 1);
                return Ok(self.emit(0x0E41));
            }
            // 4. Sara Am recomposition
            if c0 == 0x0E4D {
                // 4A: Nikhahit + Sara Aa -> Sara Am
                if n >= 2 && buf[1] == 0x0E32 {
                    self.shift_buffer(2);
                    self.drop_one(self.output_off + 1);
                    return Ok(self.emit(0x0E33));
                }
                // 4B: Nikhahit + tone + Sara Aa -> tone + Sara Am
                if n >= 3 && is_thai_tone(buf[1]) && buf[2] == 0x0E32 {
                    let tone = buf[1];
                    self.shift_buffer(3);
                    self.pending_output_char = 0x0E33;
                    self.drop_one(self.output_off + 2);
                    return Ok(self.emit(tone));
                }
            }
            // 5. tone mark before an above/below vowel: swap, then read on
            if is_thai_tone(c0) && n >= 2 && is_above_or_below_vowel(buf[1]) {
                self.input_buffer.swap(0, 1);
                continue;
            }
            // 6. Lakkhangyao -> Sara Aa unless after Ru or Lu
            if c0 == 0x0E45 && self.prev_char != 0x0E24 && self.prev_char != 0x0E26 {
                c0 = 0x0E32;
            }
            self.shift_buffer(1);
            return Ok(self.emit(c0));
        }
    }
}

impl<R: CharReader> CharFilter for ThaiCharFilter<R> {
    fn input(&self) -> &dyn CharReader {
        &self.input
    }

    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut self.input
    }

    // Java: ThaiCharFilter.read(char[], int, int)
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

// --------------------------------------------------------- ThaiRepeatFilter

/// `ThaiRepeatFilter.MAIYAMOK`: Thai Maiyamok (ๆ, U+0E46), the repetition
/// mark.
pub const MAIYAMOK: char = '\u{0E46}';

/// `org.apache.lucene.analysis.th.ThaiRepeatFilter` (Lucene 10.6, #16720):
/// a token of Maiyamok alone repeats the previous term once per mark (and
/// is dropped when no term precedes it); a term ending in Maiyamok is
/// emitted without the marks, then repeated once per mark. Each repetition
/// takes one mark's offsets and a position increment of 1.
pub struct ThaiRepeatFilter<I> {
    input: I,
    last_term: String,
    pending_repeats: usize,
    repeat_index: usize,
    repeat_starts: Vec<i32>,
    repeat_ends: Vec<i32>,
    saved_state: StateSlot,
}

impl<I: TokenStream> ThaiRepeatFilter<I> {
    /// `new ThaiRepeatFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        ThaiRepeatFilter {
            input,
            last_term: String::new(),
            pending_repeats: 0,
            repeat_index: 0,
            repeat_starts: vec![0; 4],
            repeat_ends: vec![0; 4],
            saved_state: StateSlot::new(),
        }
    }

    // Java: ThaiRepeatFilter.ensureRepeatCapacity
    fn ensure_repeat_capacity(&mut self, count: usize) {
        if count > self.repeat_starts.len() {
            self.repeat_starts = vec![0; count];
            self.repeat_ends = vec![0; count];
        }
    }
}

impl<I: TokenStream> TokenFilter for ThaiRepeatFilter<I> {
    crate::filter_input!();

    // Java: ThaiRepeatFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.pending_repeats > 0 {
            let a = self.input.attributes_mut();
            if let Some(state) = self.saved_state.get() {
                a.restore_state(state);
            }
            a.set_term(&self.last_term);
            a.set_position_increment(1)?;
            a.set_offset(
                self.repeat_starts[self.repeat_index],
                self.repeat_ends[self.repeat_index],
            )?;
            self.repeat_index += 1;
            self.pending_repeats -= 1;
            return Ok(true);
        }

        while self.input.increment_token()? {
            let a = self.input.attributes();
            let term = a.term();
            let orig_start = a.start_offset();
            let orig_end = a.end_offset();

            // Case 1: the token is all Maiyamok ("ๆ", "ๆๆ")
            if !term.is_empty() && term.chars().all(|c| c == MAIYAMOK) {
                if self.last_term.is_empty() {
                    // dangling Maiyamok at the start of the stream
                    continue;
                }
                let count = term.chars().count();
                let a = self.input.attributes_mut();
                a.set_term(&self.last_term);
                if count > 1 {
                    let first_end = orig_end.min(orig_start + 1);
                    a.set_offset(orig_start, first_end)?;
                    self.ensure_repeat_capacity(count - 1);
                    for i in 0..count - 1 {
                        let s = orig_end.min(orig_start + 1 + i as i32);
                        let e = if i == count - 2 {
                            orig_end
                        } else {
                            orig_end.min(s + 1)
                        };
                        self.repeat_starts[i] = s;
                        self.repeat_ends[i] = e;
                    }
                    self.saved_state.save(self.input.attributes());
                    self.pending_repeats = count - 1;
                    self.repeat_index = 0;
                }
                return Ok(true);
            }

            // Case 2: trailing Maiyamok ("เร็วๆ")
            let maiyamok_count = term.chars().rev().take_while(|&c| c == MAIYAMOK).count();
            if maiyamok_count > 0 {
                let base_bytes = term.len() - maiyamok_count * MAIYAMOK.len_utf8();
                let a = self.input.attributes_mut();
                a.term_mut().truncate(base_bytes);
                self.last_term.clear();
                self.last_term.push_str(a.term());

                let base_end = orig_start.max(orig_end - maiyamok_count as i32);
                a.set_offset(orig_start, base_end)?;

                self.ensure_repeat_capacity(maiyamok_count);
                for i in 0..maiyamok_count {
                    let s = orig_end.min(base_end + i as i32);
                    let e = if i == maiyamok_count - 1 {
                        orig_end
                    } else {
                        orig_end.min(s + 1)
                    };
                    self.repeat_starts[i] = s;
                    self.repeat_ends[i] = e;
                }
                self.saved_state.save(self.input.attributes());
                self.pending_repeats = maiyamok_count;
                self.repeat_index = 0;
                return Ok(true);
            }

            // a token without Maiyamok
            self.last_term.clear();
            self.last_term.push_str(term);
            return Ok(true);
        }
        Ok(false)
    }

    // Java: ThaiRepeatFilter.reset
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.last_term.clear();
        self.pending_repeats = 0;
        self.repeat_index = 0;
        self.saved_state.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Analyzer;

    #[test]
    fn the_tokenizer_refuses_as_on_a_jre_without_thai_segmentation() {
        let e = ThaiTokenizer::new().unwrap_err();
        assert_eq!(e, AnalysisError::UnsupportedOperation(UNSUPPORTED.into()));
        assert_eq!(
            e.to_string(),
            "unsupported operation: This JRE does not have support for Thai segmentation"
        );
        let a = Analyzer::new(ThaiAnalyzer::default());
        let e = a.token_stream("f", "ภาษาไทย").err().unwrap();
        assert!(matches!(e, AnalysisError::UnsupportedOperation(_)), "{e}");
    }

    #[test]
    fn stop_set_and_normalize_work() {
        let a = ThaiAnalyzer::default();
        assert!(a.stopwords().contains("และ"));
        assert_eq!(a.stopwords().len(), DEFAULT_STOP_SET.len());
        let a = Analyzer::new(ThaiAnalyzer::new(&CharArraySet::empty()));
        assert_eq!(a.normalize("f", "ABC ๑๒๓").unwrap(), b"abc 123");
    }
}

/// Lucene 10.6's `TestThaiCharFilter`, `TestThaiNormalizer` and
/// `TestThaiRepeatFilter`, with their inputs and expected tokens and
/// offsets; `whitespaceMockTokenizer` is [`WhitespaceTokenizer`] (the same
/// split on these inputs). The cases that need `ThaiTokenizer`
/// (`assumeTrue(DBBI_AVAILABLE)`) are skipped there as here.
#[cfg(test)]
mod tests_10_6 {
    use super::*;
    use crate::analyzer::collect_tokens;
    use crate::reader::StrReader;
    use crate::token_stream::Tokenizer;
    use crate::util::WhitespaceTokenizer;
    use crate::KeywordTokenizer;

    fn ws(reader: Box<dyn CharReader>) -> WhitespaceTokenizer {
        let mut t = WhitespaceTokenizer::new();
        t.set_reader(reader).unwrap();
        t
    }

    fn ws_text(text: &str) -> WhitespaceTokenizer {
        ws(Box::new(StrReader::new(text)))
    }

    fn char_filter(text: &str) -> Box<dyn CharReader> {
        Box::new(ThaiCharFilter::new(
            Box::new(StrReader::new(text)) as Box<dyn CharReader>
        ))
    }

    /// `assertTokenStreamContents(ts, terms, starts, ends, finalOffset)`.
    fn assert_ts(
        ts: &mut dyn TokenStream,
        terms: &[&str],
        starts: Option<&[i32]>,
        ends: Option<&[i32]>,
        final_offset: Option<i32>,
    ) {
        let got = collect_tokens(ts).unwrap();
        let t: Vec<&str> = got.tokens.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(t, terms);
        if let Some(starts) = starts {
            let s: Vec<i32> = got.tokens.iter().map(|t| t.start_offset).collect();
            assert_eq!(s, starts, "start offsets of {terms:?}");
        }
        if let Some(ends) = ends {
            let e: Vec<i32> = got.tokens.iter().map(|t| t.end_offset).collect();
            assert_eq!(e, ends, "end offsets of {terms:?}");
        }
        if let Some(f) = final_offset {
            assert_eq!(got.final_offset, f, "final offset of {terms:?}");
        }
        // every token here starts a position
        assert!(got.tokens.iter().all(|t| t.position_increment == 1));
    }

    fn check_char_filter(input: &str, term: &str, end: i32) {
        assert_ts(
            &mut ws(char_filter(input)),
            &[term],
            Some(&[0]),
            Some(&[end]),
            Some(end),
        );
    }

    // TestThaiCharFilter.testDoubleSaraE
    #[test]
    fn char_filter_double_sara_e() {
        check_char_filter("เเมว", "แมว", 4);
    }

    // TestThaiCharFilter.testSaraAmRecomposition
    #[test]
    fn char_filter_sara_am_recomposition() {
        check_char_filter("น\u{0E4D}\u{0E49}\u{0E32}", "น้ำ", 4);
        check_char_filter("ท\u{0E4D}\u{0E32}", "ทำ", 3);
    }

    // TestThaiCharFilter.testDuplicateDiacritics
    #[test]
    fn char_filter_duplicate_diacritics() {
        check_char_filter("ดีี", "ดี", 3);
        check_char_filter("ไม้้", "ไม้", 4);
    }

    // TestThaiCharFilter.testZeroWidth
    #[test]
    fn char_filter_zero_width() {
        check_char_filter("สวัส\u{200B}ดี", "สวัสดี", 7);
    }

    // TestThaiCharFilter.testToneReordering
    #[test]
    fn char_filter_tone_reordering() {
        check_char_filter("บ\u{0E48}\u{0E31}น", "บั่น", 4);
    }

    // TestThaiCharFilter.testLakkhangyao
    #[test]
    fn char_filter_lakkhangyao() {
        assert_ts(
            &mut ws(char_filter("ฤๅ ฦๅ กๅ")),
            &["ฤๅ", "ฦๅ", "กา"],
            Some(&[0, 3, 6]),
            Some(&[2, 5, 8]),
            Some(8),
        );
    }

    // TestThaiCharFilter.testEmpty
    #[test]
    fn char_filter_empty() {
        assert_ts(&mut ws(char_filter("")), &[], None, None, Some(0));
    }

    /// The rest of the char filter's paths, through `correct_offset`
    /// directly: ZWJ and BOM, a removal after a pending Sara Am, the
    /// leading `0` previous character, and reads one unit at a time.
    #[test]
    fn char_filter_offsets_and_small_reads() {
        // ก ZWJ BOM ํ ่ า ZWSP ข  ->  ก ่ ำ ข
        let text = "ก\u{200D}\u{FEFF}\u{0E4D}\u{0E48}\u{0E32}\u{200B}ข";
        let mut f = ThaiCharFilter::new(StrReader::new(text));
        let mut out = Vec::new();
        let mut one = [0u16; 1];
        while f.read(&mut one).unwrap() == 1 {
            out.push(one[0]);
        }
        assert_eq!(String::from_utf16(&out).unwrap(), "ก\u{0E48}\u{0E33}ข");
        let corrected: Vec<i32> = (0..=4).map(|o| f.correct_offset(o)).collect();
        assert_eq!(corrected, [0, 3, 4, 7, 8]);
        assert_eq!(f.read(&mut []).unwrap(), 0);
        f.close().unwrap();
    }

    /// `TestThaiCharFilter.testRandom`'s invariants (`checkRandomData`:
    /// offsets within the input and never going backwards) over a fixed
    /// pseudo-random Thai text.
    #[test]
    fn char_filter_offsets_stay_ordered_and_in_bounds() {
        const UNITS: [u16; 14] = [
            0x0E01, 0x0E24, 0x0E31, 0x0E32, 0x0E34, 0x0E40, 0x0E45, 0x0E47, 0x0E48, 0x0E4C, 0x0E4D,
            0x200B, 0xFEFF, 0x0020,
        ];
        let mut seed = 0x2545_F491_u32;
        for _ in 0..200 {
            let mut units = Vec::new();
            for _ in 0..(seed % 40) {
                seed ^= seed << 13;
                seed ^= seed >> 17;
                seed ^= seed << 5;
                units.push(UNITS[(seed % UNITS.len() as u32) as usize]);
            }
            let text = String::from_utf16(&units).unwrap();
            let got = collect_tokens(&mut ws(char_filter(&text))).unwrap();
            let mut last = 0;
            for t in &got.tokens {
                assert!(
                    last <= t.start_offset && t.start_offset <= t.end_offset,
                    "{text:?}"
                );
                assert!(t.end_offset <= units.len() as i32, "{text:?}");
                last = t.start_offset;
            }
            assert!(got.final_offset <= units.len() as i32);
        }
    }

    fn check_normalizer(input: &str, output: &str) {
        assert_ts(
            &mut ThaiNormalizationFilter::new(ws_text(input)),
            &[output],
            None,
            None,
            None,
        );
    }

    // TestThaiNormalizer.testDoubleSaraE
    #[test]
    fn normalizer_double_sara_e() {
        check_normalizer(
            "\u{0E40}\u{0E40}\u{0E1B}\u{0E25}\u{0E01}",
            "\u{0E41}\u{0E1B}\u{0E25}\u{0E01}",
        );
    }

    // TestThaiNormalizer.testSaraAmDecomposition
    #[test]
    fn normalizer_sara_am_decomposition() {
        check_normalizer(
            "\u{0E17}\u{0E4D}\u{0E32}\u{0E07}\u{0E32}\u{0E19}",
            "\u{0E17}\u{0E33}\u{0E07}\u{0E32}\u{0E19}",
        );
        check_normalizer(
            "\u{0E19}\u{0E4D}\u{0E49}\u{0E32}",
            "\u{0E19}\u{0E49}\u{0E33}",
        );
        check_normalizer(
            "\u{0E19}\u{0E49}\u{0E4D}\u{0E32}",
            "\u{0E19}\u{0E49}\u{0E33}",
        );
    }

    // TestThaiNormalizer.testSaraAmWithFollowedTone
    #[test]
    fn normalizer_sara_am_with_followed_tone() {
        check_normalizer("\u{0E19}\u{0E33}\u{0E49}", "\u{0E19}\u{0E49}\u{0E33}");
    }

    // TestThaiNormalizer.testReorderToneAndAboveVowel
    #[test]
    fn normalizer_reorder_tone_and_above_vowel() {
        check_normalizer("\u{0E01}\u{0E49}\u{0E34}", "\u{0E01}\u{0E34}\u{0E49}");
    }

    // TestThaiNormalizer.testDuplicateVowelsAndTones
    #[test]
    fn normalizer_duplicate_vowels_and_tones() {
        check_normalizer("\u{0E14}\u{0E35}\u{0E35}", "\u{0E14}\u{0E35}");
        check_normalizer(
            "\u{0E41}\u{0E21}\u{0E49}\u{0E49}",
            "\u{0E41}\u{0E21}\u{0E49}",
        );
        check_normalizer("\u{0E19}\u{0E32}\u{0E32}", "\u{0E19}\u{0E32}");
    }

    // TestThaiNormalizer.testConsecutiveDifferentTones
    #[test]
    fn normalizer_consecutive_different_tones() {
        check_normalizer(
            "\u{0E44}\u{0E21}\u{0E48}\u{0E49}",
            "\u{0E44}\u{0E21}\u{0E49}",
        );
    }

    // TestThaiNormalizer.testZeroWidthCharacters
    #[test]
    fn normalizer_zero_width_characters() {
        check_normalizer(
            "\u{0E20}\u{0E32}\u{0E29}\u{0E32}\u{200B}\u{0E44}\u{0E17}\u{0E22}",
            "\u{0E20}\u{0E32}\u{0E29}\u{0E32}\u{0E44}\u{0E17}\u{0E22}",
        );
        check_normalizer(
            "\u{0E20}\u{0E32}\u{0E29}\u{0E32}\u{200C}\u{0E44}\u{0E17}\u{0E22}",
            "\u{0E20}\u{0E32}\u{0E29}\u{0E32}\u{0E44}\u{0E17}\u{0E22}",
        );
    }

    // TestThaiNormalizer.testLakkhangyao
    #[test]
    fn normalizer_lakkhangyao() {
        check_normalizer("\u{0E24}\u{0E45}", "\u{0E24}\u{0E45}");
        check_normalizer("\u{0E01}\u{0E45}", "\u{0E01}\u{0E32}");
    }

    // TestThaiNormalizer.testDanglingMark
    #[test]
    fn normalizer_dangling_mark() {
        check_normalizer("\u{0E49}\u{0E01}\u{0E32}", "\u{0E01}\u{0E32}");
    }

    // TestThaiNormalizer.testEmptyTerm
    #[test]
    fn normalizer_empty_term() {
        let mut t = KeywordTokenizer::new();
        t.set_reader(Box::new(StrReader::new(""))).unwrap();
        assert_ts(
            &mut ThaiNormalizationFilter::new(t),
            &[""],
            None,
            None,
            None,
        );
    }

    /// The normalizer's remaining branches on the units directly, and the
    /// filter leaving a keyword alone.
    #[test]
    fn normalizer_edges_and_keywords() {
        let norm = |s: &str| {
            let mut u: Vec<u16> = s.encode_utf16().collect();
            let len = u.len();
            let n = ThaiNormalizer.stem(&mut u, len);
            String::from_utf16(&u[..n]).unwrap()
        };
        assert_eq!(norm(""), "");
        // only dangling marks and zero widths: nothing left
        assert_eq!(norm("\u{0E48}\u{200B}\u{0E4E}"), "");
        // Lakkhangyao first, after Lu, and a follow vowel before a tone
        assert_eq!(norm("\u{0E45}"), "\u{0E32}");
        assert_eq!(norm("\u{0E26}\u{0E45}"), "\u{0E26}\u{0E45}");
        assert_eq!(norm("\u{0E01}\u{0E30}\u{0E48}"), "\u{0E01}\u{0E48}\u{0E30}");
        // Thanthakhat before an above vowel
        assert_eq!(norm("\u{0E01}\u{0E4C}\u{0E34}"), "\u{0E01}\u{0E34}\u{0E4C}");

        let a = crate::miscellaneous::SetKeywordMarkerFilter::new(
            ws_text("\u{0E14}\u{0E35}\u{0E35}"),
            Arc::new(CharArraySet::from_words(
                ["\u{0E14}\u{0E35}\u{0E35}"],
                false,
            )),
        );
        assert_ts(
            &mut ThaiNormalizationFilter::new(a),
            &["\u{0E14}\u{0E35}\u{0E35}"],
            None,
            None,
            None,
        );
    }

    fn repeat(text: &str) -> ThaiRepeatFilter<WhitespaceTokenizer> {
        ThaiRepeatFilter::new(ws_text(text))
    }

    // TestThaiRepeatFilter.testStandaloneMaiyamok
    #[test]
    fn repeat_standalone_maiyamok() {
        assert_ts(
            &mut repeat("เร็ว ๆ"),
            &["เร็ว", "เร็ว"],
            Some(&[0, 5]),
            Some(&[4, 6]),
            None,
        );
    }

    // TestThaiRepeatFilter.testAttachedMaiyamok
    #[test]
    fn repeat_attached_maiyamok() {
        assert_ts(
            &mut repeat("เร็วๆ"),
            &["เร็ว", "เร็ว"],
            Some(&[0, 4]),
            Some(&[4, 5]),
            None,
        );
    }

    // TestThaiRepeatFilter.testMultipleMaiyamok
    #[test]
    fn repeat_multiple_maiyamok() {
        assert_ts(
            &mut repeat("มากๆๆ"),
            &["มาก", "มาก", "มาก"],
            Some(&[0, 3, 4]),
            Some(&[3, 4, 5]),
            None,
        );
        assert_ts(
            &mut repeat("มาก ๆ ๆ"),
            &["มาก", "มาก", "มาก"],
            Some(&[0, 4, 6]),
            Some(&[3, 5, 7]),
            None,
        );
    }

    // TestThaiRepeatFilter.testInSentence
    #[test]
    fn repeat_in_sentence() {
        assert_ts(
            &mut repeat("เด็ก ๆ กำลัง วิ่ง เล่น"),
            &["เด็ก", "เด็ก", "กำลัง", "วิ่ง", "เล่น"],
            Some(&[0, 5, 7, 13, 18]),
            Some(&[4, 6, 12, 17, 22]),
            None,
        );
    }

    // TestThaiRepeatFilter.testLeadingDanglingMaiyamok
    #[test]
    fn repeat_leading_dangling_maiyamok() {
        assert_ts(&mut repeat("ๆ ครับ"), &["ครับ"], None, None, None);
    }

    // TestThaiRepeatFilter.testEmpty
    #[test]
    fn repeat_empty() {
        assert_ts(&mut repeat(""), &[], None, None, None);
    }

    /// A standalone run of several marks, more marks than the initial
    /// capacity, and a reused filter forgetting the previous term.
    #[test]
    fn repeat_long_runs_and_reset() {
        assert_ts(
            &mut repeat("ดี ๆๆๆ"),
            &["ดี", "ดี", "ดี", "ดี"],
            Some(&[0, 3, 4, 5]),
            Some(&[2, 4, 5, 6]),
            None,
        );
        assert_ts(
            &mut repeat("ดีๆๆๆๆๆ"),
            &["ดี", "ดี", "ดี", "ดี", "ดี", "ดี"],
            Some(&[0, 2, 3, 4, 5, 6]),
            Some(&[2, 3, 4, 5, 6, 7]),
            None,
        );
        let mut f = repeat("ดี");
        assert_ts(&mut f, &["ดี"], None, None, None);
        f.input_mut()
            .set_reader(Box::new(StrReader::new("ๆ")))
            .unwrap();
        assert_ts(&mut f, &[], None, None, None);
    }
}
