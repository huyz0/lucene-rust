//! `org.apache.lucene.analysis.ja.JapaneseIterationMarkCharFilter`:
//! replaces the iteration marks 々, ゝ, ゞ, ヽ and ヾ by the characters they
//! repeat (voiced by ゞ/ヾ). A run of marks repeats as many characters
//! before it; `。` ends what a mark can reach. The length never changes,
//! so offsets need no correction.

use lucene_analysis::charfilter::RollingCharBuffer;
use lucene_analysis::reader::{CharFilter, CharReader};
use lucene_analysis::AnalysisError;

/// `NORMALIZE_KANJI_DEFAULT`.
pub const NORMALIZE_KANJI_DEFAULT: bool = true;
/// `NORMALIZE_KANA_DEFAULT`.
pub const NORMALIZE_KANA_DEFAULT: bool = true;

const KANJI_ITERATION_MARK: u16 = 0x3005; // 々
const HIRAGANA_ITERATION_MARK: u16 = 0x309D; // ゝ
const HIRAGANA_VOICED_ITERATION_MARK: u16 = 0x309E; // ゞ
const KATAKANA_ITERATION_MARK: u16 = 0x30FD; // ヽ
const KATAKANA_VOICED_ITERATION_MARK: u16 = 0x30FE; // ヾ
const FULL_STOP_PUNCTUATION: u16 = 0x3002; // 。

/// Hiragana to dakuten map (lookup using code point - 0x304b (か)).
const H2D: [u16; 50] = [
    0x304c, 0x304c, 0x304e, 0x304e, 0x3050, 0x3050, 0x3052, 0x3052, 0x3054, 0x3054, 0x3056, 0x3056,
    0x3058, 0x3058, 0x305a, 0x305a, 0x305c, 0x305c, 0x305e, 0x305e, 0x3060, 0x3060, 0x3062, 0x3062,
    0x3063, 0x3065, 0x3065, 0x3067, 0x3067, 0x3069, 0x3069, 0x306a, 0x306b, 0x306c, 0x306d, 0x306e,
    0x3070, 0x3070, 0x3071, 0x3073, 0x3073, 0x3074, 0x3076, 0x3076, 0x3077, 0x3079, 0x3079, 0x307a,
    0x307c, 0x307c,
];
const HIRAGANA_KA: u16 = 0x304b;
const KATAKANA_KA: u16 = 0x30ab;

/// `lookup(c, map, offset)`: katakana uses the hiragana map shifted by
/// `カ - か`.
fn lookup(c: u16, offset: u16) -> u16 {
    match c.checked_sub(offset) {
        Some(i) if usize::from(i) < H2D.len() => {
            H2D[usize::from(i)].wrapping_add(offset.wrapping_sub(HIRAGANA_KA))
        }
        _ => c,
    }
}

fn inside(c: u16, offset: u16) -> bool {
    c.checked_sub(offset)
        .is_some_and(|i| usize::from(i) < H2D.len())
}

/// `JapaneseIterationMarkCharFilter`.
pub struct JapaneseIterationMarkCharFilter<R> {
    input: R,
    buffer: RollingCharBuffer,
    buffer_position: i32,
    iteration_marks_span_size: i32,
    iteration_mark_span_end_position: i32,
    normalize_kanji: bool,
    normalize_kana: bool,
}

impl<R: CharReader> JapaneseIterationMarkCharFilter<R> {
    /// `new JapaneseIterationMarkCharFilter(input, normalizeKanji,
    /// normalizeKana)`.
    pub fn new(input: R, normalize_kanji: bool, normalize_kana: bool) -> Self {
        JapaneseIterationMarkCharFilter {
            input,
            buffer: RollingCharBuffer::default(),
            buffer_position: 0,
            iteration_marks_span_size: 0,
            iteration_mark_span_end_position: 0,
            normalize_kanji,
            normalize_kana,
        }
    }

    fn get(&mut self, pos: i32) -> Result<Option<u16>, AnalysisError> {
        self.buffer.get(&mut self.input, pos)
    }

    // Java: read()
    fn read_one(&mut self) -> Result<Option<u16>, AnalysisError> {
        let Some(mut c) = self.get(self.buffer_position)? else {
            // End of input
            self.buffer.free_before(self.buffer_position);
            return Ok(None);
        };
        // Skip surrogate pair characters
        if (0xD800..=0xDFFF).contains(&c) {
            self.iteration_mark_span_end_position = self.buffer_position.wrapping_add(1);
        }
        // Free rolling buffer on full stop
        if c == FULL_STOP_PUNCTUATION {
            self.buffer.free_before(self.buffer_position);
            self.iteration_mark_span_end_position = self.buffer_position.wrapping_add(1);
        }
        // Normalize iteration mark
        if self.is_iteration_mark(c) {
            c = self.normalize_iteration_mark(c)?;
        }
        self.buffer_position = self.buffer_position.wrapping_add(1);
        Ok(Some(c))
    }

    fn normalize_iteration_mark(&mut self, c: u16) -> Result<u16, AnalysisError> {
        // Case 1: Inside an iteration mark span
        if self.buffer_position < self.iteration_mark_span_end_position {
            let src =
                self.source_character(self.buffer_position, self.iteration_marks_span_size)?;
            return Ok(self.normalize(src, c));
        }
        // Case 2: New iteration mark spans starts where the previous one
        // ended, which is illegal
        if self.buffer_position == self.iteration_mark_span_end_position {
            // Emit the illegal iteration mark and increase end position to
            // indicate that we can't start a new span on the next position
            // either
            self.iteration_mark_span_end_position =
                self.iteration_mark_span_end_position.wrapping_add(1);
            return Ok(c);
        }
        // Case 3: New iteration mark span
        self.iteration_marks_span_size = self.next_iteration_mark_span_size()?;
        self.iteration_mark_span_end_position = self
            .buffer_position
            .wrapping_add(self.iteration_marks_span_size);
        let src = self.source_character(self.buffer_position, self.iteration_marks_span_size)?;
        Ok(self.normalize(src, c))
    }

    fn next_iteration_mark_span_size(&mut self) -> Result<i32, AnalysisError> {
        let mut span_size: i32 = 0;
        let mut i = self.buffer_position;
        while let Some(c) = self.get(i)? {
            if !self.is_iteration_mark(c) {
                break;
            }
            span_size = span_size.wrapping_add(1);
            i = i.wrapping_add(1);
        }
        // Restrict span size so that we don't go past the previous end
        // position
        if self.buffer_position.wrapping_sub(span_size) < self.iteration_mark_span_end_position {
            span_size = self
                .buffer_position
                .wrapping_sub(self.iteration_mark_span_end_position);
        }
        Ok(span_size)
    }

    /// `sourceCharacter(position, spanSize)`: Java reads `(char)
    /// buffer.get(...)`, so a position past the end is `0xFFFF`.
    fn source_character(&mut self, position: i32, span_size: i32) -> Result<u16, AnalysisError> {
        Ok(self
            .get(position.wrapping_sub(span_size))?
            .unwrap_or(0xFFFF))
    }

    fn normalize(&self, c: u16, m: u16) -> u16 {
        if self.is_hiragana_iteration_mark(m) {
            return match m {
                HIRAGANA_ITERATION_MARK if self.is_dakuten(c, HIRAGANA_KA) => c.wrapping_sub(1),
                HIRAGANA_VOICED_ITERATION_MARK => lookup(c, HIRAGANA_KA),
                _ => c,
            };
        }
        if self.is_katakana_iteration_mark(m) {
            return match m {
                KATAKANA_ITERATION_MARK if self.is_dakuten(c, KATAKANA_KA) => c.wrapping_sub(1),
                KATAKANA_VOICED_ITERATION_MARK => lookup(c, KATAKANA_KA),
                _ => c,
            };
        }
        // If m is not kana and we are to normalize it, we assume it is
        // kanji and simply return it
        c
    }

    fn is_dakuten(&self, c: u16, offset: u16) -> bool {
        inside(c, offset) && c == lookup(c, offset)
    }

    fn is_iteration_mark(&self, c: u16) -> bool {
        self.is_kanji_iteration_mark(c)
            || self.is_hiragana_iteration_mark(c)
            || self.is_katakana_iteration_mark(c)
    }

    fn is_hiragana_iteration_mark(&self, c: u16) -> bool {
        self.normalize_kana && (c == HIRAGANA_ITERATION_MARK || c == HIRAGANA_VOICED_ITERATION_MARK)
    }

    fn is_katakana_iteration_mark(&self, c: u16) -> bool {
        self.normalize_kana && (c == KATAKANA_ITERATION_MARK || c == KATAKANA_VOICED_ITERATION_MARK)
    }

    fn is_kanji_iteration_mark(&self, c: u16) -> bool {
        self.normalize_kanji && c == KANJI_ITERATION_MARK
    }
}

impl<R: CharReader> CharFilter for JapaneseIterationMarkCharFilter<R> {
    fn input(&self) -> &dyn CharReader {
        &self.input
    }
    fn input_mut(&mut self) -> &mut dyn CharReader {
        &mut self.input
    }
    // Java: read(char[], int, int)
    fn read_filtered(&mut self, buf: &mut [u16]) -> Result<usize, AnalysisError> {
        let mut n = 0usize;
        for slot in buf.iter_mut() {
            match self.read_one()? {
                Some(c) => {
                    *slot = c;
                    n = n.wrapping_add(1);
                }
                None => break,
            }
        }
        Ok(n)
    }
    /// `correct(currentOff)`: the identity.
    fn correct(&self, current_off: i32) -> i32 {
        current_off
    }
}
