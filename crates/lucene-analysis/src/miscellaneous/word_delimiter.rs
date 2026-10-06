//! `WordDelimiterIterator` and `WordDelimiterGraphFilter`, over the term's
//! UTF-16 code units as Java's are (a surrogate is `ALPHA | DIGIT`, so a
//! pair is never split).

use std::sync::{Arc, LazyLock};

use crate::attributes::State;
use crate::java_character;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::{AnalysisError, CharArraySet};

const LOWER: i32 = 0x01;
const UPPER: i32 = 0x02;
const DIGIT: i32 = 0x04;
const SUBWORD_DELIM: i32 = 0x08;
/// `WordDelimiterIterator.ALPHA`.
pub const ALPHA: i32 = 0x03;
/// `WordDelimiterIterator.ALPHANUM`.
pub const ALPHANUM: i32 = 0x07;
/// `WordDelimiterIterator.DONE`.
const DONE: i32 = -1;
/// `WordDelimiterIterator.DONE`, for the deprecated `WordDelimiterFilter`.
pub(crate) const WORD_DELIMITER_DONE: i32 = DONE;

/// `WordDelimiterIterator.DEFAULT_WORD_DELIM_TABLE`: Latin-1 by
/// `Character.isLowerCase`/`isUpperCase`/`isDigit`. `isLowerCase` counts
/// `Other_Lowercase`, which in Latin-1 is `ª` and `º`.
pub static DEFAULT_WORD_DELIM_TABLE: LazyLock<[u8; 256]> = LazyLock::new(|| {
    let mut tab = [0u8; 256];
    for (i, slot) in tab.iter_mut().enumerate() {
        let cp = i as u32;
        let t = java_character::get_type(cp);
        let mut code = 0;
        if t == java_character::LOWERCASE_LETTER || cp == 0xAA || cp == 0xBA {
            code |= LOWER;
        } else if t == java_character::UPPERCASE_LETTER {
            code |= UPPER;
        } else if t == java_character::DECIMAL_DIGIT_NUMBER {
            code |= DIGIT;
        }
        if code == 0 {
            code = SUBWORD_DELIM;
        }
        *slot = code as u8;
    }
    tab
});

/// `WordDelimiterIterator.getType(int)`.
pub fn get_type(ch: u32) -> i32 {
    use java_character::*;
    match java_character::get_type(ch) {
        UPPERCASE_LETTER => UPPER,
        LOWERCASE_LETTER => LOWER,
        TITLECASE_LETTER
        | MODIFIER_LETTER
        | OTHER_LETTER
        | NON_SPACING_MARK
        | ENCLOSING_MARK
        | COMBINING_SPACING_MARK => ALPHA,
        DECIMAL_DIGIT_NUMBER | LETTER_NUMBER | OTHER_NUMBER => DIGIT,
        SURROGATE => ALPHA | DIGIT,
        _ => SUBWORD_DELIM,
    }
}

/// `WordDelimiterFilter.isAlpha`.
pub(crate) fn is_alpha_type(t: i32) -> bool {
    is_alpha(t)
}

/// `WordDelimiterFilter.isDigit`.
pub(crate) fn is_digit_type(t: i32) -> bool {
    is_digit(t)
}

fn is_alpha(t: i32) -> bool {
    t & ALPHA != 0
}
fn is_digit(t: i32) -> bool {
    t & DIGIT != 0
}
fn is_subword_delim(t: i32) -> bool {
    t & SUBWORD_DELIM != 0
}
fn is_upper(t: i32) -> bool {
    t & UPPER != 0
}

/// `org.apache.lucene.analysis.miscellaneous.WordDelimiterIterator`.
pub(crate) struct WordDelimiterIterator {
    text: Vec<u16>,
    length: i32,
    start_bounds: i32,
    end_bounds: i32,
    pub(crate) current: i32,
    pub(crate) end: i32,
    has_final_possessive: bool,
    split_on_case_change: bool,
    split_on_numerics: bool,
    stem_english_possessive: bool,
    char_type_table: Arc<[u8]>,
    skip_possessive: bool,
}

impl WordDelimiterIterator {
    pub(crate) fn new(table: Arc<[u8]>, case: bool, numerics: bool, possessive: bool) -> Self {
        WordDelimiterIterator {
            text: Vec::new(),
            length: 0,
            start_bounds: 0,
            end_bounds: 0,
            current: 0,
            end: 0,
            has_final_possessive: false,
            split_on_case_change: case,
            split_on_numerics: numerics,
            stem_english_possessive: possessive,
            char_type_table: table,
            skip_possessive: false,
        }
    }

    fn at(&self, i: i32) -> u16 {
        self.text[i as usize]
    }

    fn char_type(&self, ch: u16) -> i32 {
        match self.char_type_table.get(usize::from(ch)) {
            Some(&t) => i32::from(t as i8),
            None => get_type(u32::from(ch)),
        }
    }

    // Java: WordDelimiterIterator.next
    pub(crate) fn next(&mut self) -> i32 {
        self.current = self.end;
        if self.current == DONE {
            return DONE;
        }
        if self.skip_possessive {
            self.current += 2;
            self.skip_possessive = false;
        }
        let mut last_type = 0;
        while self.current < self.end_bounds {
            last_type = self.char_type(self.at(self.current));
            if !is_subword_delim(last_type) {
                break;
            }
            self.current += 1;
        }
        if self.current >= self.end_bounds {
            self.end = DONE;
            return DONE;
        }
        self.end = self.current + 1;
        while self.end < self.end_bounds {
            let t = self.char_type(self.at(self.end));
            if self.is_break(last_type, t) {
                break;
            }
            last_type = t;
            self.end += 1;
        }
        if self.end < self.end_bounds - 1 && self.ends_with_possessive(self.end + 2) {
            self.skip_possessive = true;
        }
        self.end
    }

    // Java: WordDelimiterIterator.type
    pub(crate) fn word_type(&self) -> i32 {
        if self.end == DONE {
            return 0;
        }
        match self.char_type(self.at(self.current)) {
            LOWER | UPPER => ALPHA,
            t => t,
        }
    }

    // Java: WordDelimiterIterator.setText
    pub(crate) fn set_text(&mut self, text: &[u16]) {
        self.text.clear();
        self.text.extend_from_slice(text);
        self.length = text.len() as i32;
        self.end_bounds = self.length;
        self.current = 0;
        self.start_bounds = 0;
        self.end = 0;
        self.skip_possessive = false;
        self.has_final_possessive = false;
        self.set_bounds();
    }

    // Java: WordDelimiterIterator.isBreak
    fn is_break(&self, last_type: i32, t: i32) -> bool {
        if t & last_type != 0 {
            return false;
        }
        // Java's three `return false` branches, in its order.
        let same_word = (!self.split_on_case_change && is_alpha(last_type) && is_alpha(t))
            || (is_upper(last_type) && is_alpha(t))
            || (!self.split_on_numerics
                && ((is_alpha(last_type) && is_digit(t)) || (is_digit(last_type) && is_alpha(t))));
        !same_word
    }

    // Java: WordDelimiterIterator.isSingleWord
    pub(crate) fn is_single_word(&self) -> bool {
        if self.has_final_possessive {
            self.current == self.start_bounds && self.end == self.end_bounds - 2
        } else {
            self.current == self.start_bounds && self.end == self.end_bounds
        }
    }

    // Java: WordDelimiterIterator.setBounds
    fn set_bounds(&mut self) {
        while self.start_bounds < self.length
            && is_subword_delim(self.char_type(self.at(self.start_bounds)))
        {
            self.start_bounds += 1;
        }
        while self.end_bounds > self.start_bounds
            && is_subword_delim(self.char_type(self.at(self.end_bounds - 1)))
        {
            self.end_bounds -= 1;
        }
        if self.ends_with_possessive(self.end_bounds) {
            self.has_final_possessive = true;
        }
        self.current = self.start_bounds;
    }

    // Java: WordDelimiterIterator.endsWithPossessive
    fn ends_with_possessive(&self, pos: i32) -> bool {
        self.stem_english_possessive
            && pos > 2
            && self.at(pos - 2) == u16::from(b'\'')
            && (self.at(pos - 1) == u16::from(b's') || self.at(pos - 1) == u16::from(b'S'))
            && is_alpha(self.char_type(self.at(pos - 3)))
            && (pos == self.end_bounds || is_subword_delim(self.char_type(self.at(pos))))
    }
}

/// `WordDelimiterGraphFilter.GENERATE_WORD_PARTS`.
pub const GENERATE_WORD_PARTS: i32 = 1;
/// `WordDelimiterGraphFilter.GENERATE_NUMBER_PARTS`.
pub const GENERATE_NUMBER_PARTS: i32 = 2;
/// `WordDelimiterGraphFilter.CATENATE_WORDS`.
pub const CATENATE_WORDS: i32 = 4;
/// `WordDelimiterGraphFilter.CATENATE_NUMBERS`.
pub const CATENATE_NUMBERS: i32 = 8;
/// `WordDelimiterGraphFilter.CATENATE_ALL`.
pub const CATENATE_ALL: i32 = 16;
/// `WordDelimiterGraphFilter.PRESERVE_ORIGINAL`.
pub const PRESERVE_ORIGINAL: i32 = 32;
/// `WordDelimiterGraphFilter.SPLIT_ON_CASE_CHANGE`.
pub const SPLIT_ON_CASE_CHANGE: i32 = 64;
/// `WordDelimiterGraphFilter.SPLIT_ON_NUMERICS`.
pub const SPLIT_ON_NUMERICS: i32 = 128;
/// `WordDelimiterGraphFilter.STEM_ENGLISH_POSSESSIVE`.
pub const STEM_ENGLISH_POSSESSIVE: i32 = 256;
/// `WordDelimiterGraphFilter.IGNORE_KEYWORDS`.
pub const IGNORE_KEYWORDS: i32 = 512;

/// `WordDelimiterGraphFilter.WordDelimiterConcatenation`.
#[derive(Default)]
struct Concatenation {
    buffer: Vec<u16>,
    start_part: i32,
    end_part: i32,
    start_pos: i32,
    word_type: i32,
    subword_count: i32,
}

impl Concatenation {
    fn clear(&mut self) {
        self.buffer.clear();
        self.start_part = 0;
        self.end_part = 0;
        self.word_type = 0;
        self.subword_count = 0;
    }
}

/// One buffered part: `bufferedParts[4*i..4*i+4]` and `bufferedTermParts[i]`.
#[derive(Debug, Clone)]
struct Part {
    term: Option<Vec<u16>>,
    start_pos: i32,
    end_pos: i32,
    start_part: i32,
    end_part: i32,
}

/// `org.apache.lucene.analysis.miscellaneous.WordDelimiterGraphFilter`.
pub struct WordDelimiterGraphFilter<I> {
    input: I,
    prot_words: Option<Arc<CharArraySet>>,
    flags: i32,
    parts: Vec<Part>,
    buffered_pos: usize,
    iterator: WordDelimiterIterator,
    concat: Concatenation,
    adjust_internal_offsets: bool,
    last_concat_count: i32,
    concat_all: Concatenation,
    accum_pos_inc: i32,
    saved_term: Vec<u16>,
    saved_start_offset: i32,
    saved_end_offset: i32,
    saved_state: Option<State>,
    last_start_offset: i32,
    adjusting_offsets: bool,
    word_pos: i32,
    scratch: Vec<u16>,
}

impl<I: TokenStream> WordDelimiterGraphFilter<I> {
    /// `new WordDelimiterGraphFilter(TokenStream, int, CharArraySet)`.
    pub fn new(
        input: I,
        flags: i32,
        prot_words: Option<Arc<CharArraySet>>,
    ) -> Result<Self, AnalysisError> {
        let table: Arc<[u8]> = Arc::from(&DEFAULT_WORD_DELIM_TABLE[..]);
        Self::with_table(input, false, table, flags, prot_words)
    }

    /// `new WordDelimiterGraphFilter(TokenStream, boolean adjustInternalOffsets,
    /// byte[] charTypeTable, int, CharArraySet)`.
    pub fn with_table(
        input: I,
        adjust_internal_offsets: bool,
        char_type_table: Arc<[u8]>,
        flags: i32,
        prot_words: Option<Arc<CharArraySet>>,
    ) -> Result<Self, AnalysisError> {
        let all = GENERATE_WORD_PARTS
            | GENERATE_NUMBER_PARTS
            | CATENATE_WORDS
            | CATENATE_NUMBERS
            | CATENATE_ALL
            | PRESERVE_ORIGINAL
            | SPLIT_ON_CASE_CHANGE
            | SPLIT_ON_NUMERICS
            | STEM_ENGLISH_POSSESSIVE
            | IGNORE_KEYWORDS;
        if flags & !all != 0 {
            return Err(AnalysisError::IllegalArgument(format!(
                "flags contains unrecognized flag: {flags}"
            )));
        }
        let has = |f: i32| flags & f != 0;
        Ok(WordDelimiterGraphFilter {
            input,
            prot_words,
            flags,
            parts: Vec::new(),
            buffered_pos: 0,
            iterator: WordDelimiterIterator::new(
                char_type_table,
                has(SPLIT_ON_CASE_CHANGE),
                has(SPLIT_ON_NUMERICS),
                has(STEM_ENGLISH_POSSESSIVE),
            ),
            concat: Concatenation::default(),
            adjust_internal_offsets,
            last_concat_count: 0,
            concat_all: Concatenation::default(),
            accum_pos_inc: 0,
            saved_term: Vec::new(),
            saved_start_offset: 0,
            saved_end_offset: 0,
            saved_state: None,
            last_start_offset: 0,
            adjusting_offsets: false,
            word_pos: 0,
            scratch: Vec::new(),
        })
    }

    fn has(&self, flag: i32) -> bool {
        self.flags & flag != 0
    }

    // Java: WordDelimiterGraphFilter.buffer
    fn buffer(
        &mut self,
        term: Option<Vec<u16>>,
        start_pos: i32,
        end_pos: i32,
        start_part: i32,
        end_part: i32,
    ) {
        debug_assert!(end_pos > start_pos);
        self.parts.push(Part {
            term,
            start_pos,
            end_pos,
            start_part,
            end_part,
        });
    }

    // Java: WordDelimiterGraphFilter.saveState
    fn save_state(&mut self) {
        let a = self.input.attributes();
        self.saved_start_offset = a.start_offset();
        self.saved_end_offset = a.end_offset();
        self.saved_state = Some(a.capture_state());
        self.saved_term.clear();
        self.saved_term.extend(a.term().encode_utf16());
    }

    fn should_concatenate(&self, t: i32) -> bool {
        (self.has(CATENATE_WORDS) && is_alpha(t)) || (self.has(CATENATE_NUMBERS) && is_digit(t))
    }

    fn should_generate_parts(&self, t: i32) -> bool {
        (self.has(GENERATE_WORD_PARTS) && is_alpha(t))
            || (self.has(GENERATE_NUMBER_PARTS) && is_digit(t))
    }

    // Java: WordDelimiterConcatenation.write
    fn write(&mut self, all: bool) {
        let c = if all { &self.concat_all } else { &self.concat };
        let (term, start_pos, start_part, end_part) =
            (c.buffer.clone(), c.start_pos, c.start_part, c.end_part);
        let word_pos = self.word_pos;
        self.buffer(Some(term), start_pos, word_pos, start_part, end_part);
    }

    // Java: WordDelimiterGraphFilter.flushConcatenation (of `concat`)
    fn flush_concatenation(&mut self) {
        if self.word_pos == self.concat.start_pos {
            self.word_pos += 1;
        }
        self.last_concat_count = self.concat.subword_count;
        if self.concat.subword_count != 1 || !self.should_generate_parts(self.concat.word_type) {
            self.write(false);
        }
        self.concat.clear();
    }

    // Java: WordDelimiterGraphFilter.concatenate
    fn concatenate(&mut self, all: bool) {
        let (cur, end, t, wp) = (
            self.iterator.current,
            self.iterator.end,
            self.iterator.word_type(),
            self.word_pos,
        );
        let c = if all {
            &mut self.concat_all
        } else {
            &mut self.concat
        };
        if c.buffer.is_empty() {
            c.word_type = t;
            c.start_part = cur;
            c.start_pos = wp;
        }
        c.buffer
            .extend_from_slice(&self.saved_term[cur as usize..end as usize]);
        c.subword_count += 1;
        c.end_part = end;
    }

    // Java: WordDelimiterGraphFilter.bufferWordParts
    fn buffer_word_parts(&mut self) {
        self.save_state();
        let saved_len = self.saved_term.len() as i32;
        self.adjusting_offsets = self.adjust_internal_offsets
            && self.saved_end_offset - self.saved_start_offset == saved_len;
        self.parts.clear();
        self.last_concat_count = 0;
        self.word_pos = 0;
        if self.has(PRESERVE_ORIGINAL) {
            self.buffer(None, 0, 1, 0, saved_len);
        }
        if self.iterator.is_single_word() {
            let (wp, cur, end) = (self.word_pos, self.iterator.current, self.iterator.end);
            self.buffer(None, wp, wp + 1, cur, end);
            self.word_pos += 1;
            self.iterator.next();
        } else {
            while self.iterator.end != DONE {
                let word_type = self.iterator.word_type();
                if !self.concat.buffer.is_empty() && (self.concat.word_type & word_type) == 0 {
                    self.flush_concatenation();
                }
                if self.should_concatenate(word_type) {
                    self.concatenate(false);
                }
                if self.has(CATENATE_ALL) {
                    self.concatenate(true);
                }
                if self.should_generate_parts(word_type) {
                    let (wp, cur, end) = (self.word_pos, self.iterator.current, self.iterator.end);
                    self.buffer(None, wp, wp + 1, cur, end);
                    self.word_pos += 1;
                }
                self.iterator.next();
            }
            if !self.concat.buffer.is_empty() {
                self.flush_concatenation();
            }
            if !self.concat_all.buffer.is_empty() {
                if self.concat_all.subword_count > self.last_concat_count {
                    if self.word_pos == self.concat_all.start_pos {
                        self.word_pos += 1;
                    }
                    self.write(true);
                }
                self.concat_all.clear();
            }
        }
        if self.has(PRESERVE_ORIGINAL) {
            if self.word_pos == 0 {
                self.word_pos += 1;
            }
            self.parts[0].end_pos = self.word_pos;
        }
        // Java: PositionSorter, an InPlaceMergeSorter (stable): by start
        // offset, then longer first.
        let from = usize::from(self.has(PRESERVE_ORIGINAL));
        self.parts[from..].sort_by(|i, j| {
            i.start_part
                .cmp(&j.start_part)
                .then(j.end_part.cmp(&i.end_part))
        });
        self.word_pos = 0;
        self.buffered_pos = 0;
    }
}

impl<I: TokenStream> TokenFilter for WordDelimiterGraphFilter<I> {
    crate::filter_input!();

    // Java: WordDelimiterGraphFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            if self.saved_state.is_none() {
                if !self.input.increment_token()? {
                    return Ok(false);
                }
                let a = self.input.attributes();
                if self.has(IGNORE_KEYWORDS) && a.is_keyword() {
                    return Ok(true);
                }
                self.accum_pos_inc += a.position_increment();
                self.scratch.clear();
                self.scratch.extend(a.term().encode_utf16());
                let term_length = self.scratch.len() as i32;
                let scratch = std::mem::take(&mut self.scratch);
                self.iterator.set_text(&scratch);
                self.scratch = scratch;
                self.iterator.next();
                let protected = self
                    .prot_words
                    .as_ref()
                    .is_some_and(|p| p.contains(a.term()));
                if (self.iterator.current == 0 && self.iterator.end == term_length) || protected {
                    let inc = self.accum_pos_inc;
                    self.input.attributes_mut().set_position_increment(inc)?;
                    self.accum_pos_inc = 0;
                    return Ok(true);
                }
                if self.iterator.end == DONE {
                    if !self.has(PRESERVE_ORIGINAL) {
                        continue;
                    }
                    self.accum_pos_inc = 0;
                    return Ok(true);
                }
                self.buffer_word_parts();
            }
            if self.buffered_pos < self.parts.len() {
                let part = self.parts[self.buffered_pos].clone();
                self.buffered_pos += 1;
                let (mut start_offset, mut end_offset) = if self.adjusting_offsets {
                    (
                        self.saved_start_offset + part.start_part,
                        self.saved_start_offset + part.end_part,
                    )
                } else {
                    (self.saved_start_offset, self.saved_end_offset)
                };
                start_offset = start_offset.max(self.last_start_offset);
                end_offset = end_offset.max(self.last_start_offset);
                let a = self.input.attributes_mut();
                a.clear_attributes();
                a.restore_state(
                    self.saved_state
                        .as_ref()
                        .expect("buffered parts imply a state"),
                );
                a.set_offset(start_offset, end_offset)?;
                self.last_start_offset = start_offset;
                match &part.term {
                    None => a.set_term_utf16(
                        &self.saved_term[part.start_part as usize..part.end_part as usize],
                    ),
                    Some(t) => a.set_term_utf16(t),
                }
                a.set_position_increment(self.accum_pos_inc + part.start_pos - self.word_pos)?;
                self.accum_pos_inc = 0;
                a.set_position_length(part.end_pos - part.start_pos)?;
                self.word_pos = part.start_pos;
                return Ok(true);
            }
            self.saved_state = None;
        }
    }

    // Java: WordDelimiterGraphFilter.reset
    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.accum_pos_inc = 0;
        self.saved_state = None;
        self.last_start_offset = 0;
        self.concat.clear();
        self.concat_all.clear();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    #[test]
    fn table_and_types() {
        assert_eq!(i32::from(DEFAULT_WORD_DELIM_TABLE[b'a' as usize]), LOWER);
        assert_eq!(i32::from(DEFAULT_WORD_DELIM_TABLE[0xAA]), LOWER);
        assert_eq!(i32::from(DEFAULT_WORD_DELIM_TABLE[b'Q' as usize]), UPPER);
        assert_eq!(i32::from(DEFAULT_WORD_DELIM_TABLE[b'7' as usize]), DIGIT);
        assert_eq!(
            i32::from(DEFAULT_WORD_DELIM_TABLE[b'-' as usize]),
            SUBWORD_DELIM
        );
        assert_eq!(get_type(0x4E2D), ALPHA);
        assert_eq!(get_type(0x2160), DIGIT);
        assert_eq!(get_type(0xD83D), ALPHA | DIGIT);
        assert_eq!(get_type(0x01C5), ALPHA);
        assert_eq!(get_type(0x0391), UPPER);
        assert_eq!(get_type(0x03B1), LOWER);
        assert_eq!(get_type(0x2014), SUBWORD_DELIM);
        assert_eq!(ALPHANUM, 7);
    }

    #[test]
    fn bad_flags_and_protected_words() {
        assert!(WordDelimiterGraphFilter::new(Canned::parse(""), 1 << 12, None).is_err());
        let prot = Arc::new(CharArraySet::from_words(["a-b"], false));
        let mut f = WordDelimiterGraphFilter::new(
            Canned::parse("a-b:0:3:1:1 c-d:4:7:1:1|7|0"),
            GENERATE_WORD_PARTS,
            Some(prot),
        )
        .unwrap();
        assert_eq!(render(&mut f), "a-b:0:3:1:1 c:4:7:1:1 d:4:7:1:1|7|0");
    }

    #[test]
    fn keywords_delimiters_only_and_adjusted_offsets() {
        let mut c = Canned::parse("a-b:0:3:1:1 --:4:6:1:1 c-d:7:10:1:1|10|0");
        c.set_keywords(&[true, false, false]);
        let mut f =
            WordDelimiterGraphFilter::new(c, GENERATE_WORD_PARTS | IGNORE_KEYWORDS, None).unwrap();
        assert_eq!(render(&mut f), "a-b:0:3:1:1 c:7:10:2:1 d:7:10:1:1|10|0");
        let mut f = WordDelimiterGraphFilter::new(
            Canned::parse("--:0:2:1:1|2|0"),
            GENERATE_WORD_PARTS | PRESERVE_ORIGINAL,
            None,
        )
        .unwrap();
        assert_eq!(render(&mut f), "--:0:2:1:1|2|0");
        let table: Arc<[u8]> = Arc::from(&DEFAULT_WORD_DELIM_TABLE[..]);
        let mut f = WordDelimiterGraphFilter::with_table(
            Canned::parse("ab-cd:0:5:1:1|5|0"),
            true,
            table,
            GENERATE_WORD_PARTS | CATENATE_WORDS,
            None,
        )
        .unwrap();
        assert_eq!(render(&mut f), "abcd:0:5:1:2 ab:0:2:0:1 cd:3:5:1:1|5|0");
    }
}
