//! `org.apache.lucene.analysis.th`: `ThaiTokenizer` and `ThaiAnalyzer`.
//!
//! `ThaiTokenizer` splits sentences with the JDK's sentence iterator
//! ([`crate::util::sentence_break`], ported) and words with
//! `BreakIterator.getWordInstance(th)`, the JDK's dictionary-based iterator.
//! The JDK's Thai word list (`thai_dict`, in `java.base`) is
//! GPL-2.0-with-Classpath-Exception data: it cannot be vendored into this
//! Apache-2.0 library, nor re-derived from the JDK's answers without copying
//! it (`docs/licences.md`). So the word iterator here is pluggable
//! ([`ThaiWordBreaker`]): `lucene-analysis-icu` installs ICU's word iterator
//! over ICU's Unicode-licensed Thai dictionary ([`install_word_breaker`]),
//! the dictionary Lucene's `ICUTokenizer`, Meilisearch's charabia (through
//! ICU4X) and Typesense segment Thai with. The sentence loop and the
//! letter-or-digit filter are Lucene's; the word boundaries inside Thai runs
//! are ICU's, which do not always match the JDK's (a different word list:
//! `docs/parity/analysis-lang.md`).
//!
//! Without an installed iterator the port does what Lucene does on a JRE
//! whose word iterator does not segment Thai: [`dbbi_available`] is `false`,
//! and constructing the tokenizer -- directly, through [`ThaiAnalyzer`] or
//! the `thai` factory -- is an [`AnalysisError::UnsupportedOperation`] with
//! Java's message. `ThaiAnalyzer`'s stop set and `normalize` chain need no
//! dictionary and always work.
//!
//! Lucene 10.6 added Thai components that need no dictionary:
//! [`ThaiNormalizer`] and [`ThaiNormalizationFilter`], [`ThaiCharFilter`]
//! (#16717) and [`ThaiRepeatFilter`] (#16720). [`ThaiAnalyzer`] runs them
//! as 10.6 does and uses its stop set (#16717, #16718).

use std::sync::{Arc, LazyLock, RwLock};

use crate::attributes::StateSlot;
use crate::charfilter::OffsetCorrections;
use crate::core_analysis::DecimalDigitFilter;
use crate::reader::{CharFilter, CharReader};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::util::stemmer_util::delete;
use crate::util::{Segmenter, SegmentingBase, SegmentingTokenizer};
use crate::{AnalysisError, CharArraySet, LowerCaseFilter};

use super::{comment_set, CharStemmer, StemFilter};

/// A word `BreakIterator` over one sentence (Java's `CharArrayIterator`
/// wrapper: positions are relative to the sentence's first unit).
pub trait ThaiWordBreaker: Send {
    /// `setText`: positions the iterator on the sentence's first boundary.
    fn set_text(&mut self, text: &[u16]);
    /// `current()`.
    fn current(&self) -> i32;
    /// `next()`: the following boundary, [`DONE`] past the last.
    fn next(&mut self) -> i32;
}

/// `BreakIterator.DONE`.
pub const DONE: i32 = -1;

type NewBreaker = fn() -> Box<dyn ThaiWordBreaker>;

static WORD_BREAKER: RwLock<Option<NewBreaker>> = RwLock::new(None);

static JVM_WORD_BREAKER: RwLock<Option<NewBreaker>> = RwLock::new(None);

/// Installs the host JVM's own Thai word iterator (the one Lucene's
/// `ThaiTokenizer` runs on), which [`thai_tokenizer_jvm`] clones instead of
/// the dictionary installed by [`install_word_breaker`].
pub fn install_jvm_word_breaker(new_breaker: fn() -> Box<dyn ThaiWordBreaker>) {
    *JVM_WORD_BREAKER.write().unwrap_or_else(|e| e.into_inner()) = Some(new_breaker);
}

/// Installs the Thai word iterator `ThaiTokenizer` clones per tokenizer
/// (`proto`). The first installation wins; `false` when one was installed
/// already.
pub fn install_word_breaker(new_breaker: fn() -> Box<dyn ThaiWordBreaker>) -> bool {
    let mut slot = WORD_BREAKER.write().unwrap_or_else(|e| e.into_inner());
    if slot.is_some() {
        return false;
    }
    *slot = Some(new_breaker);
    true
}

/// `ThaiTokenizer.DBBI_AVAILABLE`: whether a word iterator that segments
/// Thai is installed (see the module docs).
pub fn dbbi_available() -> bool {
    WORD_BREAKER
        .read()
        .unwrap_or_else(|e| e.into_inner())
        .is_some()
}

/// The message of `ThaiTokenizer`'s `UnsupportedOperationException`.
pub const UNSUPPORTED: &str = "This JRE does not have support for Thai segmentation";

/// `ThaiAnalyzer.getDefaultStopSet()` (`stopwords.txt`).
pub static DEFAULT_STOP_SET: LazyLock<Arc<CharArraySet>> =
    LazyLock::new(|| comment_set(include_str!("stopwords/th_stopwords.txt")));

/// The `ThaiTokenizer` half of `SegmentingTokenizerBase`: each sentence's
/// words from the installed [`ThaiWordBreaker`], those not starting with a
/// letter or digit skipped.
pub struct ThaiSegmenter {
    word_breaker: Box<dyn ThaiWordBreaker>,
    sentence_start: usize,
    sentence_end: usize,
}

impl std::fmt::Debug for ThaiSegmenter {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ThaiSegmenter")
            .field("sentence_start", &self.sentence_start)
            .field("sentence_end", &self.sentence_end)
            .finish_non_exhaustive()
    }
}

impl Segmenter for ThaiSegmenter {
    fn set_next_sentence(
        &mut self,
        base: &SegmentingBase,
        sentence_start: usize,
        sentence_end: usize,
    ) -> Result<(), AnalysisError> {
        self.sentence_start = sentence_start;
        self.sentence_end = sentence_end;
        let sentence = base
            .buffer()
            .get(sentence_start..sentence_end)
            .unwrap_or_default();
        self.word_breaker.set_text(sentence);
        Ok(())
    }

    fn increment_word(&mut self, base: &mut SegmentingBase) -> Result<bool, AnalysisError> {
        let mut start = self.word_breaker.current();
        if start == DONE {
            return Ok(false); // BreakIterator exhausted
        }
        // find the next set of boundaries, skipping over non-tokens
        let mut end = self.word_breaker.next();
        let sentence = base
            .buffer()
            .get(self.sentence_start..self.sentence_end)
            .unwrap_or_default();
        while end != DONE && !starts_with_letter_or_digit(sentence, start) {
            start = end;
            end = self.word_breaker.next();
        }
        if end == DONE {
            return Ok(false); // BreakIterator exhausted
        }
        let (Ok(s), Ok(e)) = (usize::try_from(start), usize::try_from(end)) else {
            return Ok(false);
        };
        let term = sentence.get(s..e).unwrap_or_default().to_vec();
        let at = base.offset().saturating_add(self.sentence_start as i32);
        let start_offset = base.correct_offset(at.saturating_add(start));
        let end_offset = base.correct_offset(at.saturating_add(end));
        let a = base.attributes_mut();
        a.clear_attributes();
        a.set_term_utf16(&term);
        a.set_offset(start_offset, end_offset)?;
        Ok(true)
    }
}

/// `Character.isLetterOrDigit(Character.codePointAt(buffer, sentenceStart +
/// start, sentenceEnd))`.
fn starts_with_letter_or_digit(sentence: &[u16], start: i32) -> bool {
    let Some(rest) = usize::try_from(start).ok().and_then(|s| sentence.get(s..)) else {
        return false;
    };
    match char::decode_utf16(rest.iter().copied()).next() {
        Some(Ok(c)) => crate::java_character::is_letter_or_digit(u32::from(c)),
        Some(Err(e)) => {
            crate::java_character::is_letter_or_digit(u32::from(e.unpaired_surrogate()))
        }
        None => false,
    }
}

/// `ThaiTokenizer`.
pub type ThaiTokenizer = SegmentingTokenizer<ThaiSegmenter>;

/// `new ThaiTokenizer()`: Java's `UnsupportedOperationException` when no
/// Thai word iterator is installed.
pub fn thai_tokenizer() -> Result<ThaiTokenizer, AnalysisError> {
    thai_tokenizer_with(&WORD_BREAKER)
}

/// [`thai_tokenizer`] over the JVM's word iterator ([`install_jvm_word_breaker`]).
pub fn thai_tokenizer_jvm() -> Result<ThaiTokenizer, AnalysisError> {
    thai_tokenizer_with(&JVM_WORD_BREAKER)
}

fn thai_tokenizer_with(slot: &RwLock<Option<NewBreaker>>) -> Result<ThaiTokenizer, AnalysisError> {
    let installed = *slot.read().unwrap_or_else(|e| e.into_inner());
    let Some(new_breaker) = installed else {
        return Err(AnalysisError::UnsupportedOperation(UNSUPPORTED.to_string()));
    };
    Ok(SegmentingTokenizer::new(ThaiSegmenter {
        word_breaker: new_breaker(),
        sentence_start: 0,
        sentence_end: 0,
    }))
}

/// `ThaiAnalyzer`: `ThaiCharFilter`, `ThaiTokenizer`, `LowerCaseFilter`,
/// `DecimalDigitFilter`, `ThaiNormalizationFilter`, `ThaiRepeatFilter`,
/// `StopFilter`. Its token streams refuse without an
/// installed word iterator (see the module docs); `normalize` always works.
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
        let result = DecimalDigitFilter::new(LowerCaseFilter::new(thai_tokenizer()?));
        let result = ThaiRepeatFilter::new(ThaiNormalizationFilter::new(result));
        Ok(crate::analyzer::TokenStreamComponents::new(
            crate::StopFilter::new(result, Arc::clone(&self.stopwords)),
        ))
    }

    fn normalize(
        &self,
        _field: &str,
        input: Box<dyn crate::token_stream::TokenStream>,
    ) -> Box<dyn crate::token_stream::TokenStream> {
        Box::new(ThaiNormalizationFilter::new(DecimalDigitFilter::new(
            LowerCaseFilter::new(input),
        )))
    }

    fn init_reader(&self, _field: &str, reader: Box<dyn CharReader>) -> Box<dyn CharReader> {
        Box::new(ThaiCharFilter::new(reader))
    }

    fn init_reader_for_normalization(
        &self,
        _field: &str,
        reader: Box<dyn CharReader>,
    ) -> Box<dyn CharReader> {
        Box::new(ThaiCharFilter::new(reader))
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
        // No word iterator is installed in this crate's tests.
        assert!(!dbbi_available());
        let Err(e) = thai_tokenizer() else {
            panic!("no word iterator is installed")
        };
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
mod tests_10_6;
