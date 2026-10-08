//! `com.ibm.icu.text.RuleBasedBreakIterator`: the rule-driven state machine
//! over compiled `.brk` data ([`RbbiData`]), with its boundary cache and its
//! dictionary cache, which hands runs of dictionary characters to the
//! language break engines ([`break_engines`](crate::icu4j::break_engines)).
//!
//! Ported: the forward path Lucene's `ICUTokenizer` drives -- `setText`,
//! `first`, `next`, `current`, `getRuleStatus`, `getRuleStatusVec` --
//! through `handleNext` (look-ahead rules, the BOF category, rule status
//! indices), `BreakCache.next`/`populateFollowing`/`addFollowing` (the
//! 128-entry ring, six boundaries pre-fetched past a non-dictionary one) and
//! `DictionaryCache.following`/`populateDictionary`. Not ported: backward
//! iteration (`previous`, `preceding`, `following`, `isBoundary`, `last`,
//! `handleSafePrevious`, `populateNear`/`populatePreceding`) and building
//! rules from source (`RuleBasedBreakIterator(String)`,
//! `RBBIRuleBuilder`) -- the refusal is
//! [`crate::ICUTokenizerFactory`]'s `rulefiles`.

use std::sync::Arc;

use crate::icu4j::break_engines::{language_break_engine, DequeI};
use crate::icu4j::char_iter::{CharIter, DONE32};
use crate::icu4j::rbbi_data::{
    RbbiData, ACCEPTING, ACCEPTING_UNCONDITIONAL, LOOKAHEAD, NEXTSTATES, RBBI_BOF_REQUIRED, TAGSIDX,
};
use crate::IcuError;

/// `BreakIterator.DONE`.
pub const DONE: i32 = -1;

const START_STATE: u16 = 1;
const STOP_STATE: u16 = 0;
const CACHE_SIZE: usize = 128;

/// `RuleBasedBreakIterator.DictionaryCache`.
#[derive(Debug, Clone)]
struct DictionaryCache {
    breaks: DequeI,
    position_in_cache: i32,
    start: i32,
    limit: i32,
    first_rule_status_index: i32,
    other_rule_status_index: i32,
    boundary: i32,
    status_index: i32,
}

impl Default for DictionaryCache {
    fn default() -> Self {
        DictionaryCache {
            breaks: DequeI::default(),
            position_in_cache: -1,
            start: 0,
            limit: 0,
            first_rule_status_index: 0,
            other_rule_status_index: 0,
            boundary: 0,
            status_index: 0,
        }
    }
}

// ARITH: (the whole impl) cache positions index `breaks`, whose size is
// bounded by the text length.
#[allow(clippy::arithmetic_side_effects)]
impl DictionaryCache {
    fn reset(&mut self) {
        self.position_in_cache = -1;
        self.start = 0;
        self.limit = 0;
        self.first_rule_status_index = 0;
        self.other_rule_status_index = 0;
        self.breaks.remove_all_elements();
    }

    /// `following(fromPos)`.
    fn following(&mut self, from_pos: i32) -> bool {
        if from_pos >= self.limit || from_pos < self.start {
            self.position_in_cache = -1;
            return false;
        }
        let size = self.breaks.size();
        if self.position_in_cache >= 0
            && self.position_in_cache < size
            && self.breaks.element_at(self.position_in_cache) == from_pos
        {
            self.position_in_cache += 1;
            if self.position_in_cache >= size {
                self.position_in_cache = -1;
                return false;
            }
            self.boundary = self.breaks.element_at(self.position_in_cache);
            self.status_index = self.other_rule_status_index;
            return true;
        }
        self.position_in_cache = 0;
        while self.position_in_cache < size {
            let r = self.breaks.element_at(self.position_in_cache);
            if r > from_pos {
                self.boundary = r;
                self.status_index = self.other_rule_status_index;
                return true;
            }
            self.position_in_cache += 1;
        }
        self.position_in_cache = -1;
        false
    }
}

/// `RuleBasedBreakIterator`, forward iteration.
#[derive(Debug, Clone)]
pub struct RuleBasedBreakIterator {
    data: Arc<RbbiData>,
    text: Vec<u16>,
    position: i32,
    rule_status_index: i32,
    done: bool,
    look_ahead_matches: Vec<i32>,
    dictionary_char_count: i32,
    // BreakCache
    start_buf_idx: usize,
    end_buf_idx: usize,
    text_idx: i32,
    buf_idx: usize,
    boundaries: [i32; CACHE_SIZE],
    statuses: [i16; CACHE_SIZE],
    dictionary_cache: DictionaryCache,
}

#[inline]
fn mod_chunk(i: usize) -> usize {
    i & (CACHE_SIZE - 1)
}

impl RuleBasedBreakIterator {
    /// `getInstanceFromCompiledRules(bytes)`.
    pub fn from_compiled_rules(bytes: &[u8]) -> Result<RuleBasedBreakIterator, IcuError> {
        Ok(Self::from_data(Arc::new(RbbiData::get(bytes)?)))
    }

    /// An iterator over shared rules (`clone()` of a prototype).
    pub fn from_data(data: Arc<RbbiData>) -> RuleBasedBreakIterator {
        let n = usize::try_from(data.ftable.look_ahead_results_size).unwrap_or(0);
        let mut it = RuleBasedBreakIterator {
            data,
            text: Vec::new(),
            position: 0,
            rule_status_index: 0,
            done: false,
            look_ahead_matches: vec![0; n.min(1 << 16)],
            dictionary_char_count: 0,
            start_buf_idx: 0,
            end_buf_idx: 0,
            text_idx: 0,
            buf_idx: 0,
            boundaries: [0; CACHE_SIZE],
            statuses: [0; CACHE_SIZE],
            dictionary_cache: DictionaryCache::default(),
        };
        it.cache_reset(0, 0);
        it
    }

    /// The rules.
    pub fn data(&self) -> &Arc<RbbiData> {
        &self.data
    }

    /// `setText(CharacterIterator)` over a copy of `text` (Lucene's
    /// `CharArrayIterator` over the tokenizer's buffer), then `first()`.
    pub fn set_text(&mut self, text: &[u16]) {
        self.text.clear();
        self.text.extend_from_slice(text);
        self.cache_reset(0, 0);
        self.dictionary_cache.reset();
        self.first();
    }

    /// `first()`.
    pub fn first(&mut self) -> i32 {
        // fBreakCache.seek(0) holds after setText's reset.
        self.buf_idx = self.start_buf_idx;
        self.text_idx = self.boundaries[self.buf_idx];
        self.cache_current();
        self.position
    }

    /// `next()` (Java's name: a `BreakIterator`, not an `Iterator`).
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> i32 {
        self.cache_next();
        if self.done {
            DONE
        } else {
            self.position
        }
    }

    /// `current()`.
    pub fn current(&self) -> i32 {
        self.position
    }

    /// `getRuleStatus()`.
    // ARITH: status indices are table offsets (i32 from the data); a
    // corrupt one reads 0.
    pub fn get_rule_status(&self) -> i32 {
        let idx = self
            .rule_status_index
            .wrapping_add(self.data.status(self.rule_status_index));
        self.data.status(idx)
    }

    /// `getRuleStatusVec(fillInArray)`.
    pub fn get_rule_status_vec(&self) -> Vec<i32> {
        let n = self.data.status(self.rule_status_index);
        (1..=n.max(0))
            .map(|i| self.data.status(self.rule_status_index.wrapping_add(i)))
            .collect()
    }

    // --- handleNext --------------------------------------------------------

    /// `handleNext()`.
    // ARITH: positions are text indices; categories and states are u16
    // table values; `row + NEXTSTATES + category` stays far below
    // usize::MAX and reads 0 (STOP) outside the table.
    #[allow(clippy::arithmetic_side_effects)]
    fn handle_next(&mut self) -> i32 {
        self.rule_status_index = 0;
        self.dictionary_char_count = 0;
        let data = Arc::clone(&self.data);
        let table = &data.ftable;
        let trie = &data.trie;
        let mut text = CharIter::new(&self.text);
        let initial_position = self.position;
        text.set_index(usize::try_from(initial_position).unwrap_or(0));
        let mut result = initial_position;
        let mut c = text.current();
        if c >= 0xd800 {
            c = text.next_trail32(c);
            if c == DONE32 {
                self.done = true;
                return DONE;
            }
        }
        let mut state = START_STATE;
        let mut row = data.row_index(state);
        let mut category: u16 = 3;
        let flags = table.flags;
        let dict_start = table.dict_categories_start;
        // RBBI_START 0, RBBI_RUN 1, RBBI_END 2
        let mut mode = 1;
        if flags & RBBI_BOF_REQUIRED != 0 {
            category = 2;
            mode = 0;
        }
        while state != STOP_STATE {
            if c == DONE32 {
                if mode == 2 {
                    break;
                }
                mode = 2;
                category = 1;
            } else if mode == 1 {
                category = trie.get(c) as i16 as u16;
                if i32::from(category as i16) >= dict_start {
                    self.dictionary_char_count += 1;
                }
                c = text.next();
                if c >= 0xd800 {
                    c = text.next_trail32(c);
                }
            } else {
                mode = 1;
            }
            state = table.at(row + NEXTSTATES + usize::from(category));
            row = data.row_index(state);
            let accepting = table.at(row + ACCEPTING);
            if accepting == ACCEPTING_UNCONDITIONAL {
                result = text.index() as i32;
                if (0x10000..=0x10ffff).contains(&c) {
                    result -= 1;
                }
                self.rule_status_index = i32::from(table.at(row + TAGSIDX));
            } else if accepting > ACCEPTING_UNCONDITIONAL {
                let lookahead_result = self
                    .look_ahead_matches
                    .get(usize::from(accepting))
                    .copied()
                    .unwrap_or(-1);
                if lookahead_result >= 0 {
                    self.rule_status_index = i32::from(table.at(row + TAGSIDX));
                    self.position = lookahead_result;
                    return lookahead_result;
                }
            }
            let rule = table.at(row + LOOKAHEAD);
            if rule != 0 {
                let mut pos = text.index() as i32;
                if (0x10000..=0x10ffff).contains(&c) {
                    pos -= 1;
                }
                if let Some(slot) = self.look_ahead_matches.get_mut(usize::from(rule)) {
                    *slot = pos;
                }
            }
        }
        if result == initial_position {
            text.set_index(usize::try_from(initial_position).unwrap_or(0));
            text.next32();
            result = text.index() as i32;
            self.rule_status_index = 0;
        }
        self.position = result;
        result
    }

    // --- BreakCache (forward) -------------------------------------------------

    /// `BreakCache.reset(pos, ruleStatus)`.
    fn cache_reset(&mut self, pos: i32, rule_status: i32) {
        self.start_buf_idx = 0;
        self.end_buf_idx = 0;
        self.text_idx = pos;
        self.buf_idx = 0;
        self.boundaries[0] = pos;
        self.statuses[0] = rule_status as i16;
    }

    /// `BreakCache.current()`.
    fn cache_current(&mut self) -> i32 {
        self.position = self.text_idx;
        self.rule_status_index = i32::from(self.statuses[self.buf_idx]);
        self.done = false;
        self.text_idx
    }

    /// `BreakCache.next()`.
    fn cache_next(&mut self) {
        if self.buf_idx == self.end_buf_idx {
            self.done = !self.populate_following();
            self.position = self.text_idx;
            self.rule_status_index = i32::from(self.statuses[self.buf_idx]);
        } else {
            self.buf_idx = mod_chunk(self.buf_idx.wrapping_add(1));
            self.text_idx = self.boundaries[self.buf_idx];
            self.position = self.text_idx;
            self.rule_status_index = i32::from(self.statuses[self.buf_idx]);
        }
    }

    /// `BreakCache.populateFollowing()`.
    fn populate_following(&mut self) -> bool {
        let from_position = self.boundaries[self.end_buf_idx];
        let from_rule_status_idx = i32::from(self.statuses[self.end_buf_idx]);
        if self.dictionary_cache.following(from_position) {
            let (b, s) = (
                self.dictionary_cache.boundary,
                self.dictionary_cache.status_index,
            );
            self.add_following(b, s, true);
            return true;
        }
        self.position = from_position;
        let pos = self.handle_next();
        if pos == DONE {
            return false;
        }
        let rule_status_idx = self.rule_status_index;
        if self.dictionary_char_count > 0 {
            self.populate_dictionary(from_position, pos, from_rule_status_idx, rule_status_idx);
            if self.dictionary_cache.following(from_position) {
                let (b, s) = (
                    self.dictionary_cache.boundary,
                    self.dictionary_cache.status_index,
                );
                self.add_following(b, s, true);
                return true;
            }
        }
        self.add_following(pos, rule_status_idx, true);
        for _ in 0..6 {
            let pos = self.handle_next();
            if pos == DONE || self.dictionary_char_count > 0 {
                break;
            }
            let s = self.rule_status_index;
            self.add_following(pos, s, false);
        }
        true
    }

    /// `BreakCache.addFollowing(position, ruleStatusIdx, update)`.
    fn add_following(&mut self, position: i32, rule_status_idx: i32, update: bool) {
        let next_idx = mod_chunk(self.end_buf_idx.wrapping_add(1));
        if next_idx == self.start_buf_idx {
            self.start_buf_idx = mod_chunk(self.start_buf_idx.wrapping_add(6));
        }
        self.boundaries[next_idx] = position;
        self.statuses[next_idx] = rule_status_idx as i16;
        self.end_buf_idx = next_idx;
        if update {
            self.buf_idx = next_idx;
            self.text_idx = position;
        }
    }

    // --- DictionaryCache.populateDictionary ----------------------------------------

    /// `DictionaryCache.populateDictionary(startPos, endPos, firstRuleStatus,
    /// otherRuleStatus)`.
    // ARITH: positions are text indices; break counts are bounded by them.
    #[allow(clippy::arithmetic_side_effects)]
    fn populate_dictionary(
        &mut self,
        start_pos: i32,
        end_pos: i32,
        first_rule_status: i32,
        other_rule_status: i32,
    ) {
        if end_pos - start_pos <= 1 {
            return;
        }
        let cache = &mut self.dictionary_cache;
        cache.reset();
        cache.first_rule_status_index = first_rule_status;
        cache.other_rule_status_index = other_rule_status;
        let range_start = start_pos;
        let range_end = end_pos;
        let mut found_break_count = 0;
        let data = Arc::clone(&self.data);
        let mut text = CharIter::new(&self.text);
        text.set_index(usize::try_from(range_start).unwrap_or(0));
        let mut c = text.current32();
        let mut category = i32::from(data.trie.get(c) as i16);
        let dict_start = data.ftable.dict_categories_start;
        loop {
            let mut current;
            loop {
                current = text.index() as i32;
                if current >= range_end || category >= dict_start {
                    break;
                }
                c = text.next32();
                category = i32::from(data.trie.get(c) as i16);
            }
            if current >= range_end {
                break;
            }
            let lbe = language_break_engine(c);
            found_break_count +=
                lbe.find_breaks(&mut text, range_start, range_end, &mut cache.breaks);
            c = text.current32();
            category = i32::from(data.trie.get(c) as i16);
        }
        if found_break_count > 0 {
            if start_pos < cache.breaks.element_at(0) {
                cache.breaks.offer(start_pos);
            }
            if end_pos > cache.breaks.peek() {
                cache.breaks.push(end_pos);
            }
            cache.position_in_cache = 0;
            cache.start = cache.breaks.element_at(0);
            cache.limit = cache.breaks.peek();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const WORD: &[u8] = include_bytes!("../resources/word.brk");

    #[test]
    fn rejects_bad_rules() {
        assert!(RuleBasedBreakIterator::from_compiled_rules(&[0; 8]).is_err());
        // Every truncation fails cleanly; byte flips load or fail, and a
        // loaded iterator runs without panicking.
        for len in (0..WORD.len()).step_by(211) {
            assert!(RuleBasedBreakIterator::from_compiled_rules(&WORD[..len]).is_err());
        }
        let text = crate::icu4j::utf16::units("Hello, world 42! ภาษาไทย 東京 😀 x\u{301}");
        for at in (0..WORD.len()).step_by(37) {
            let mut b = WORD.to_vec();
            b[at] ^= 0xff;
            if let Ok(mut it) = RuleBasedBreakIterator::from_compiled_rules(&b) {
                it.set_text(&text);
                let mut n = 0;
                while it.next() != DONE && n < 1000 {
                    n += 1;
                    let _ = it.get_rule_status();
                }
            }
        }
    }

    #[test]
    fn forward_iteration_and_statuses() {
        let mut it = RuleBasedBreakIterator::from_compiled_rules(WORD).unwrap();
        let text = crate::icu4j::utf16::units("Hello, world 42");
        it.set_text(&text);
        assert_eq!(it.current(), 0);
        let mut bounds = Vec::new();
        loop {
            let b = it.next();
            if b == DONE {
                break;
            }
            bounds.push((b, it.get_rule_status(), it.get_rule_status_vec()));
        }
        assert_eq!(
            bounds.iter().map(|b| b.0).collect::<Vec<_>>(),
            vec![5, 6, 7, 12, 13, 15]
        );
        assert_eq!(bounds[0].1, 200);
        assert_eq!(bounds[0].2, vec![200]);
        assert_eq!(bounds[5].1, 100);
        assert_eq!(it.next(), DONE);
        // A long run of non-dictionary text crosses the 128-entry cache.
        let long = crate::icu4j::utf16::units(&"ab ".repeat(300));
        it.set_text(&long);
        let mut n = 0;
        while it.next() != DONE {
            n += 1;
        }
        assert_eq!(n, 600);
        it.set_text(&[]);
        assert_eq!(it.next(), DONE);
        assert!(it.data().rule_source.contains("$dictionary"));
        let _ = it.clone();
    }
}
