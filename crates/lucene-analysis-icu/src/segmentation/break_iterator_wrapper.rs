//! `org.apache.lucene.analysis.icu.segmentation.BreakIteratorWrapper`: a
//! rule-based break iterator whose rule status is replaced by
//! `EMOJI_SEQUENCE_STATUS` for a segment that starts with an emoji (a
//! keycap base or `©®™〰〽` only when an emoji presentation selector or a
//! combining keycap follows it).

use std::sync::OnceLock;

use crate::icu4j::rbbi::{RuleBasedBreakIterator, DONE};
use crate::icu4j::unicode_set::UnicodeSet;
use crate::segmentation::config::{EMOJI_SEQUENCE_STATUS, WORD_NONE};
use crate::segmentation::script_iterator::char_at_absolute;

fn emoji_rk() -> &'static UnicodeSet {
    static S: OnceLock<UnicodeSet> = OnceLock::new();
    S.get_or_init(|| {
        UnicodeSet::from_pattern("[\\u002a\\u00230-9©®™〰〽]").expect("EMOJI_RK parses")
    })
}

fn emoji() -> &'static UnicodeSet {
    static S: OnceLock<UnicodeSet> = OnceLock::new();
    S.get_or_init(|| {
        UnicodeSet::from_pattern("[[:Emoji:][:Extended_Pictographic:]]").expect("EMOJI parses")
    })
}

/// `BreakIteratorWrapper`.
#[derive(Debug, Clone)]
pub struct BreakIteratorWrapper {
    rbbi: RuleBasedBreakIterator,
    start: usize,
    status: i32,
}

impl BreakIteratorWrapper {
    /// `new BreakIteratorWrapper(rbbi)`.
    pub fn new(rbbi: RuleBasedBreakIterator) -> Self {
        BreakIteratorWrapper {
            rbbi,
            start: 0,
            status: WORD_NONE,
        }
    }

    /// `current()`.
    pub fn current(&self) -> i32 {
        self.rbbi.current()
    }

    /// `getRuleStatus()`.
    pub fn rule_status(&self) -> i32 {
        self.status
    }

    /// `next()`; `text` is the whole buffer `set_text` was given a part of.
    pub fn next(&mut self, text: &[u16]) -> i32 {
        let current = self.rbbi.current();
        let next = self.rbbi.next();
        self.status = self.calc_status(text, current, next);
        next
    }

    /// `calcStatus(current, next)`.
    fn calc_status(&self, text: &[u16], current: i32, next: i32) -> i32 {
        if next != DONE && self.is_emoji(text, current, next) {
            EMOJI_SEQUENCE_STATUS
        } else {
            self.rbbi.get_rule_status()
        }
    }

    /// `isEmoji(current, next)`.
    // ARITH: current and next are offsets within the segment.
    #[allow(clippy::arithmetic_side_effects)]
    fn is_emoji(&self, text: &[u16], current: i32, next: i32) -> bool {
        let begin = self.start + usize::try_from(current).unwrap_or(0);
        let end = self.start + usize::try_from(next).unwrap_or(0);
        let codepoint = char_at_absolute(text, 0, end, begin);
        if emoji().contains(codepoint) {
            if emoji_rk().contains(codepoint) {
                let trailer = begin + if codepoint > 0xffff { 2 } else { 1 };
                return trailer < end
                    && text
                        .get(trailer)
                        .is_some_and(|&u| u == 0xfe0f || u == 0x20e3);
            }
            return true;
        }
        false
    }

    /// `setText(text, start, length)`.
    // ARITH: start + length is within the buffer.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn set_text(&mut self, text: &[u16], start: usize, length: usize) {
        self.start = start;
        let end = (start + length).min(text.len());
        self.rbbi.set_text(text.get(start..end).unwrap_or(&[]));
        self.status = WORD_NONE;
    }
}
