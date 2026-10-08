//! `org.apache.lucene.analysis.icu.segmentation.ICUTokenizer`: breaks text
//! into words per UAX #29, per script ([`CompositeBreakIterator`]), reading
//! 4,096 units at a time and cutting a full buffer after its last white
//! space; each token carries its type and [`ScriptAttribute`].

use std::sync::Arc;

use lucene_analysis::attributes::AttributeSource;
use lucene_analysis::{AnalysisError, CharReader, TokenStream, Tokenizer, TokenizerInput};

use crate::icu4j::rbbi::DONE;
use crate::icu4j::uprops;
use crate::segmentation::composite_break_iterator::CompositeBreakIterator;
use crate::segmentation::config::{DefaultICUTokenizerConfig, ICUTokenizerConfig};
use crate::tokenattributes::ScriptAttribute;

const IOBUFFER: usize = 4096;

/// `ICUTokenizer`.
pub struct ICUTokenizer {
    atts: AttributeSource,
    input: TokenizerInput,
    buffer: Vec<u16>,
    length: i32,
    usable_length: i32,
    offset: i32,
    breaker: CompositeBreakIterator,
    config: Arc<dyn ICUTokenizerConfig>,
}

impl Default for ICUTokenizer {
    fn default() -> Self {
        Self::new()
    }
}

impl ICUTokenizer {
    /// `new ICUTokenizer()`: `DefaultICUTokenizerConfig(true, true)`.
    pub fn new() -> Self {
        Self::with_config(Arc::new(DefaultICUTokenizerConfig::new(true, true)))
    }

    /// `new ICUTokenizer(config)`.
    pub fn with_config(config: Arc<dyn ICUTokenizerConfig>) -> Self {
        let mut atts = AttributeSource::new();
        atts.add_custom::<ScriptAttribute>();
        ICUTokenizer {
            atts,
            input: TokenizerInput::new(),
            buffer: vec![0; IOBUFFER],
            length: 0,
            usable_length: 0,
            offset: 0,
            breaker: CompositeBreakIterator::new(Arc::clone(&config)),
            config,
        }
    }

    /// `findSafeEnd()`.
    // SENTINEL: `-1` = no white space to cut the buffer at.
    // ARITH: i indexes the buffer.
    #[allow(clippy::arithmetic_side_effects)]
    fn find_safe_end(&self) -> i32 {
        let mut i = self.length - 1;
        while i >= 0 {
            if uprops::is_whitespace(i32::from(self.buffer[i as usize])) {
                return i + 1;
            }
            i -= 1;
        }
        -1
    }

    /// `refill()`.
    // ARITH: lengths are within the 4,096-unit buffer.
    #[allow(clippy::arithmetic_side_effects)]
    fn refill(&mut self) -> Result<(), AnalysisError> {
        self.offset += self.usable_length;
        let leftover = (self.length - self.usable_length).max(0) as usize;
        let from = self.usable_length.max(0) as usize;
        self.buffer.copy_within(from..from + leftover, 0);
        let requested = IOBUFFER - leftover;
        let returned = self.read_fully(leftover, requested)?;
        self.length = (returned + leftover) as i32;
        if returned < requested {
            self.usable_length = self.length;
        } else {
            self.usable_length = self.find_safe_end();
            if self.usable_length < 0 {
                self.usable_length = self.length;
            }
        }
        let usable = self.usable_length.max(0) as usize;
        self.breaker.set_text(&self.buffer, 0, usable);
        Ok(())
    }

    /// `read(input, buffer, offset, length)`: reads until `length` units or
    /// the end.
    // ARITH: counts stay within `length`.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_fully(&mut self, offset: usize, length: usize) -> Result<usize, AnalysisError> {
        let mut remaining = length;
        let reader = self.input.reader()?;
        while remaining > 0 {
            let location = length - remaining;
            let count = reader.read(&mut self.buffer[offset + location..offset + length])?;
            if count == 0 {
                break;
            }
            remaining -= count;
        }
        Ok(length - remaining)
    }

    /// `incrementTokenBuffer()`.
    // ARITH: boundaries are offsets within the buffer.
    #[allow(clippy::arithmetic_side_effects)]
    fn increment_token_buffer(&mut self) -> Result<bool, AnalysisError> {
        let mut start = self.breaker.current();
        let mut end = self.breaker.next(&self.buffer);
        while end != DONE && self.breaker.rule_status() == 0 {
            start = end;
            end = self.breaker.next(&self.buffer);
        }
        if end == DONE {
            return Ok(false);
        }
        let (s, e) = (start.max(0) as usize, end.max(0) as usize);
        self.atts
            .set_term_utf16(&self.buffer[s..e.min(self.buffer.len())]);
        let so = self.input.correct_offset(self.offset + start);
        let eo = self.input.correct_offset(self.offset + end);
        self.atts.set_offset(so, eo)?;
        let script = self.breaker.script_code();
        let status = self.breaker.rule_status();
        self.atts
            .set_token_type(self.config.get_type(script, status));
        self.atts.add_custom::<ScriptAttribute>().set_code(script);
        Ok(true)
    }
}

impl TokenStream for ICUTokenizer {
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }
    fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    // Java: ICUTokenizer.incrementToken
    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        self.atts.clear_attributes();
        if self.length == 0 {
            self.refill()?;
        }
        while !self.increment_token_buffer()? {
            self.refill()?;
            if self.length <= 0 {
                return Ok(false);
            }
        }
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.input.reset();
        self.breaker.set_text(&self.buffer, 0, 0);
        self.length = 0;
        self.usable_length = 0;
        self.offset = 0;
        Ok(())
    }

    // ARITH: offsets within the input.
    #[allow(clippy::arithmetic_side_effects)]
    fn end(&mut self) -> Result<(), AnalysisError> {
        self.atts.end_attributes();
        let final_offset = if self.length < 0 {
            self.offset
        } else {
            self.offset + self.length
        };
        let o = self.input.correct_offset(final_offset);
        self.atts.set_offset(o, o)
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl Tokenizer for ICUTokenizer {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.input.set_reader(input)
    }
}
