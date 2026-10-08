//! `org.apache.lucene.analysis.icu.segmentation.CompositeBreakIterator`:
//! runs a [`ScriptIterator`] over the text and, within each script run, that
//! script's break iterator, one per script, created on first use and reused.

use std::collections::HashMap;
use std::sync::Arc;

use crate::icu4j::rbbi::DONE;
use crate::segmentation::break_iterator_wrapper::BreakIteratorWrapper;
use crate::segmentation::config::ICUTokenizerConfig;
use crate::segmentation::script_iterator::{ScriptIterator, COMMON};

/// `CompositeBreakIterator`.
pub struct CompositeBreakIterator {
    config: Arc<dyn ICUTokenizerConfig>,
    word_breakers: HashMap<i32, BreakIteratorWrapper>,
    rbbi: i32,
    script_iterator: ScriptIterator,
}

impl CompositeBreakIterator {
    /// `new CompositeBreakIterator(config)`.
    pub fn new(config: Arc<dyn ICUTokenizerConfig>) -> Self {
        let combine = config.combine_cj();
        CompositeBreakIterator {
            config,
            word_breakers: HashMap::new(),
            rbbi: COMMON,
            script_iterator: ScriptIterator::new(combine),
        }
    }

    fn breaker(&mut self, script: i32) -> &mut BreakIteratorWrapper {
        let config = &self.config;
        self.word_breakers
            .entry(script)
            .or_insert_with(|| BreakIteratorWrapper::new(config.get_break_iterator(script)))
    }

    /// `next()`: the next boundary in `text` (the buffer `set_text` was
    /// given), as an absolute offset, or `DONE`.
    // ARITH: script offsets are within the buffer.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn next(&mut self, text: &[u16]) -> i32 {
        let script = self.rbbi;
        let mut next = self.breaker(script).next(text);
        while next == DONE && self.script_iterator.next(text) {
            let code = self.script_iterator.script_code();
            self.rbbi = code;
            let start = self.script_iterator.script_start();
            let len = self.script_iterator.script_limit() - start;
            let b = self.breaker(code);
            b.set_text(text, start, len);
            next = b.next(text);
        }
        if next == DONE {
            DONE
        } else {
            next + self.script_iterator.script_start() as i32
        }
    }

    /// `current()`.
    // ARITH: as for `next`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn current(&mut self) -> i32 {
        let script = self.rbbi;
        let current = self.breaker(script).current();
        if current == DONE {
            DONE
        } else {
            current + self.script_iterator.script_start() as i32
        }
    }

    /// `getRuleStatus()`.
    pub fn rule_status(&mut self) -> i32 {
        let script = self.rbbi;
        self.breaker(script).rule_status()
    }

    /// `getScriptCode()`.
    pub fn script_code(&self) -> i32 {
        self.script_iterator.script_code()
    }

    /// `setText(text, start, length)`.
    // ARITH: script offsets are within the buffer.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn set_text(&mut self, text: &[u16], start: usize, length: usize) {
        self.script_iterator.set_text(start, length);
        if self.script_iterator.next(text) {
            let code = self.script_iterator.script_code();
            self.rbbi = code;
            let s = self.script_iterator.script_start();
            let len = self.script_iterator.script_limit() - s;
            self.breaker(code).set_text(text, s, len);
        } else {
            self.rbbi = COMMON;
            self.breaker(COMMON).set_text(text, 0, 0);
        }
    }
}
