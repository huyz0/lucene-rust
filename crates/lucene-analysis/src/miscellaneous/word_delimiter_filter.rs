//! `org.apache.lucene.analysis.miscellaneous.WordDelimiterFilter`, the
//! deprecated non-graph word delimiter (Lucene recommends
//! [`WordDelimiterGraphFilter`](super::WordDelimiterGraphFilter)): splits a
//! term into subwords and catenations, stacked on positions, ported state for
//! state over the shared `WordDelimiterIterator`.
//!
//! Java's `InPlaceMergeSorter` over the buffered tokens is a stable sort by
//! the same key (start offset, then position increment descending).

use std::sync::Arc;

use crate::attributes::State;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::{AnalysisError, CharArraySet};

use super::word_delimiter::{
    is_alpha_type, is_digit_type, WordDelimiterIterator, CATENATE_ALL, CATENATE_NUMBERS,
    CATENATE_WORDS, DEFAULT_WORD_DELIM_TABLE, GENERATE_NUMBER_PARTS, GENERATE_WORD_PARTS,
    IGNORE_KEYWORDS, PRESERVE_ORIGINAL, SPLIT_ON_CASE_CHANGE, SPLIT_ON_NUMERICS,
    STEM_ENGLISH_POSSESSIVE, WORD_DELIMITER_DONE,
};

/// `WordDelimiterFilter.WordDelimiterConcatenation`.
#[derive(Debug, Default)]
struct Concatenation {
    buffer: Vec<u16>,
    start_offset: i32,
    end_offset: i32,
    word_type: i32,
    subword_count: i32,
}

impl Concatenation {
    fn is_empty(&self) -> bool {
        self.buffer.is_empty()
    }

    fn clear(&mut self) {
        self.buffer.clear();
        self.start_offset = 0;
        self.end_offset = 0;
        self.word_type = 0;
        self.subword_count = 0;
    }
}

/// `WordDelimiterFilter`.
pub struct WordDelimiterFilter<I> {
    input: I,
    flags: i32,
    prot_words: Option<Arc<CharArraySet>>,
    iterator: WordDelimiterIterator,
    concat: Concatenation,
    last_concat_count: i32,
    concat_all: Concatenation,
    accum_pos_inc: i32,
    saved_buffer: Vec<u16>,
    saved_start_offset: i32,
    saved_end_offset: i32,
    saved_type: std::borrow::Cow<'static, str>,
    has_saved_state: bool,
    has_illegal_offsets: bool,
    has_output_token: bool,
    has_output_following_original: bool,
    /// `buffered`, `startOff`, `posInc`: the first `buffered_len` entries;
    /// those past it are spent states kept for their buffers (Java captures
    /// a new state per buffered token).
    buffered: Vec<(State, i32, i32)>,
    buffered_len: usize,
    buffered_pos: usize,
    first: bool,
    scratch: Vec<u16>,
}

impl<I: TokenStream> WordDelimiterFilter<I> {
    /// `new WordDelimiterFilter(TokenStream, int configurationFlags,
    /// CharArraySet protWords)`, over `DEFAULT_WORD_DELIM_TABLE`.
    pub fn new(input: I, flags: i32, prot_words: Option<Arc<CharArraySet>>) -> Self {
        Self::with_table(
            input,
            Arc::from(&DEFAULT_WORD_DELIM_TABLE[..]),
            flags,
            prot_words,
        )
    }

    /// `new WordDelimiterFilter(TokenStream, byte[] charTypeTable, int,
    /// CharArraySet)`.
    pub fn with_table(
        input: I,
        char_type_table: Arc<[u8]>,
        flags: i32,
        prot_words: Option<Arc<CharArraySet>>,
    ) -> Self {
        let has = |f: i32| flags & f != 0;
        WordDelimiterFilter {
            input,
            flags,
            prot_words,
            iterator: WordDelimiterIterator::new(
                char_type_table,
                has(SPLIT_ON_CASE_CHANGE),
                has(SPLIT_ON_NUMERICS),
                has(STEM_ENGLISH_POSSESSIVE),
            ),
            concat: Concatenation::default(),
            last_concat_count: 0,
            concat_all: Concatenation::default(),
            accum_pos_inc: 0,
            saved_buffer: Vec::new(),
            saved_start_offset: 0,
            saved_end_offset: 0,
            saved_type: std::borrow::Cow::Borrowed(""),
            has_saved_state: false,
            has_illegal_offsets: false,
            has_output_token: false,
            has_output_following_original: false,
            buffered: Vec::new(),
            buffered_len: 0,
            buffered_pos: 0,
            first: false,
            scratch: Vec::new(),
        }
    }

    fn has(&self, flag: i32) -> bool {
        self.flags & flag != 0
    }

    // Java: WordDelimiterFilter.buffer
    fn buffer(&mut self) {
        let a = self.input.attributes();
        let (start, inc) = (a.start_offset(), a.position_increment());
        match self.buffered.get_mut(self.buffered_len) {
            Some(slot) => {
                slot.0.clone_from(a);
                (slot.1, slot.2) = (start, inc);
            }
            None => self.buffered.push((a.capture_state(), start, inc)),
        }
        self.buffered_len += 1;
    }

    // Java: WordDelimiterFilter.saveState
    fn save_state(&mut self) {
        let a = self.input.attributes();
        self.saved_start_offset = a.start_offset();
        self.saved_end_offset = a.end_offset();
        // `scratch` holds this term's UTF-16 units (set by `increment`).
        self.saved_buffer.clear();
        self.saved_buffer.extend_from_slice(&self.scratch);
        self.has_illegal_offsets =
            self.saved_end_offset - self.saved_start_offset != self.saved_buffer.len() as i32;
        self.saved_type = a.token_type_cow().clone();
        self.has_saved_state = true;
    }

    // Java: WordDelimiterFilter.flushConcatenation
    fn flush_concatenation(&mut self, all: bool) -> Result<bool, AnalysisError> {
        let (count, word_type) = {
            let c = if all { &self.concat_all } else { &self.concat };
            (c.subword_count, c.word_type)
        };
        self.last_concat_count = count;
        if count != 1 || !self.should_generate_parts(word_type) {
            self.write_and_clear(all)?;
            return Ok(true);
        }
        if all {
            self.concat_all.clear();
        } else {
            self.concat.clear();
        }
        Ok(false)
    }

    fn should_concatenate(&self, t: i32) -> bool {
        (self.has(CATENATE_WORDS) && is_alpha_type(t))
            || (self.has(CATENATE_NUMBERS) && is_digit_type(t))
    }

    fn should_generate_parts(&self, t: i32) -> bool {
        (self.has(GENERATE_WORD_PARTS) && is_alpha_type(t))
            || (self.has(GENERATE_NUMBER_PARTS) && is_digit_type(t))
    }

    // Java: WordDelimiterFilter.concatenate
    fn concatenate(&mut self, all: bool) {
        let (current, end) = (self.iterator.current, self.iterator.end);
        let start = self.saved_start_offset;
        let part = &self.saved_buffer[current as usize..end as usize];
        let c = if all {
            &mut self.concat_all
        } else {
            &mut self.concat
        };
        if c.is_empty() {
            c.start_offset = start + current;
        }
        c.buffer.extend_from_slice(part);
        c.subword_count += 1;
        c.end_offset = start + end;
    }

    // Java: WordDelimiterFilter.generatePart
    fn generate_part(&mut self, is_single_word: bool) -> Result<(), AnalysisError> {
        let (current, end) = (self.iterator.current, self.iterator.end);
        let start_offset = self.saved_start_offset + current;
        let end_offset = self.saved_start_offset + end;
        let (s, e) = if self.has_illegal_offsets {
            if is_single_word && start_offset <= self.saved_end_offset {
                (start_offset, self.saved_end_offset)
            } else {
                (self.saved_start_offset, self.saved_end_offset)
            }
        } else {
            (start_offset, end_offset)
        };
        let inc = self.position(false);
        let a = self.input.attributes_mut();
        a.clear_attributes();
        a.set_term_utf16(&self.saved_buffer[current as usize..end as usize]);
        a.set_offset(s, e)?;
        a.set_position_increment(inc)?;
        a.set_token_type(self.saved_type.clone());
        Ok(())
    }

    // Java: WordDelimiterFilter.position
    fn position(&mut self, inject: bool) -> i32 {
        let pos_inc = self.accum_pos_inc;
        if self.has_output_token {
            self.accum_pos_inc = 0;
            return if inject { 0 } else { pos_inc.max(1) };
        }
        self.has_output_token = true;
        if !self.has_output_following_original {
            self.has_output_following_original = true;
            return 0;
        }
        self.accum_pos_inc = 0;
        pos_inc.max(1)
    }

    // Java: WordDelimiterConcatenation.write, then clear
    fn write_and_clear(&mut self, all: bool) -> Result<(), AnalysisError> {
        let (start, end) = {
            let c = if all { &self.concat_all } else { &self.concat };
            self.scratch.clear();
            self.scratch.extend_from_slice(&c.buffer);
            (c.start_offset, c.end_offset)
        };
        let (s, e) = if self.has_illegal_offsets {
            (self.saved_start_offset, self.saved_end_offset)
        } else {
            (start, end)
        };
        let inc = self.position(true);
        let a = self.input.attributes_mut();
        a.clear_attributes();
        a.set_term_utf16(&self.scratch);
        a.set_offset(s, e)?;
        a.set_position_increment(inc)?;
        a.set_token_type(self.saved_type.clone());
        self.accum_pos_inc = 0;
        if all {
            self.concat_all.clear();
        } else {
            self.concat.clear();
        }
        Ok(())
    }
}

impl<I: TokenStream> TokenFilter for WordDelimiterFilter<I> {
    crate::filter_input!();

    // Java: WordDelimiterFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        loop {
            if !self.has_saved_state {
                if !self.input.increment_token()? {
                    return Ok(false);
                }
                if self.has(IGNORE_KEYWORDS) && self.input.attributes().is_keyword() {
                    return Ok(true);
                }
                let a = self.input.attributes();
                self.scratch.clear();
                self.scratch.extend(a.term().encode_utf16());
                let term_length = self.scratch.len() as i32;
                self.accum_pos_inc += a.position_increment();
                let pos_inc = a.position_increment();
                self.iterator.set_text(&self.scratch);
                self.iterator.next();
                let protected = self
                    .prot_words
                    .as_ref()
                    .is_some_and(|p| p.contains(a.term()));
                if (self.iterator.current == 0 && self.iterator.end == term_length) || protected {
                    let inc = self.accum_pos_inc;
                    self.input.attributes_mut().set_position_increment(inc)?;
                    self.accum_pos_inc = 0;
                    self.first = false;
                    return Ok(true);
                }
                if self.iterator.end == WORD_DELIMITER_DONE && !self.has(PRESERVE_ORIGINAL) {
                    if pos_inc == 1 && !self.first {
                        self.accum_pos_inc -= 1;
                    }
                    continue;
                }
                self.save_state();
                self.has_output_token = false;
                self.has_output_following_original = !self.has(PRESERVE_ORIGINAL);
                self.last_concat_count = 0;
                if self.has(PRESERVE_ORIGINAL) {
                    let inc = self.accum_pos_inc;
                    self.input.attributes_mut().set_position_increment(inc)?;
                    self.accum_pos_inc = 0;
                    self.first = false;
                    return Ok(true);
                }
            }

            if self.iterator.end == WORD_DELIMITER_DONE {
                if !self.concat.is_empty() && self.flush_concatenation(false)? {
                    self.buffer();
                    continue;
                }
                if !self.concat_all.is_empty() {
                    if self.concat_all.subword_count > self.last_concat_count {
                        self.write_and_clear(true)?;
                        self.buffer();
                        continue;
                    }
                    self.concat_all.clear();
                }
                if self.buffered_pos < self.buffered_len {
                    if self.buffered_pos == 0 {
                        // InPlaceMergeSorter: stable, by start offset then
                        // position increment descending.
                        self.buffered[..self.buffered_len]
                            .sort_by(|x, y| x.1.cmp(&y.1).then(y.2.cmp(&x.2)));
                    }
                    let state = &self.buffered[self.buffered_pos].0;
                    self.buffered_pos += 1;
                    let a = self.input.attributes_mut();
                    a.clear_attributes();
                    a.restore_state(state);
                    if self.first && a.position_increment() == 0 {
                        a.set_position_increment(1)?;
                    }
                    self.first = false;
                    return Ok(true);
                }
                self.buffered_len = 0;
                self.buffered_pos = 0;
                self.has_saved_state = false;
                continue;
            }

            if self.iterator.is_single_word() {
                self.generate_part(true)?;
                self.iterator.next();
                self.first = false;
                return Ok(true);
            }

            let word_type = self.iterator.word_type();
            if !self.concat.is_empty() && self.concat.word_type & word_type == 0 {
                if self.flush_concatenation(false)? {
                    self.has_output_token = false;
                    self.buffer();
                    continue;
                }
                self.has_output_token = false;
            }
            if self.should_concatenate(word_type) {
                if self.concat.is_empty() {
                    self.concat.word_type = word_type;
                }
                self.concatenate(false);
            }
            if self.has(CATENATE_ALL) {
                self.concatenate(true);
            }
            if self.should_generate_parts(word_type) {
                self.generate_part(false)?;
                self.buffer();
            }
            self.iterator.next();
        }
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.has_saved_state = false;
        self.concat.clear();
        self.concat_all.clear();
        self.accum_pos_inc = 0;
        self.buffered_len = 0;
        self.buffered_pos = 0;
        self.first = true;
        Ok(())
    }
}
