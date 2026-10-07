//! `org.apache.lucene.analysis.util.SegmentingTokenizerBase`: a tokenizer
//! that reads its input in 1,024-unit windows, splits each window into
//! sentences with a `BreakIterator` and hands each sentence to a subclass,
//! which splits it into words.
//!
//! Java's abstract class is state plus two hooks; here the state is
//! [`SegmentingBase`], the hooks the [`Segmenter`] trait, and
//! [`SegmentingTokenizer`] the `final` driver joining them with a
//! [`BreakIterator`] (by default the JDK's sentence iterator,
//! [`SentenceBreakIterator`]).
//!
//! A window ends at the last line or paragraph separator in it
//! (`isSafeEnd`) unless the input is exhausted, so a sentence is never cut
//! by the buffer -- except one longer than the window with no separator,
//! which is cut where the window ends, as in Java.

use super::sentence_break::{BreakIterator, SentenceBreakIterator};
use crate::attributes::AttributeSource;
use crate::reader::CharReader;
use crate::token_stream::{TokenStream, Tokenizer, TokenizerInput};
use crate::AnalysisError;

/// `BUFFERMAX`: the window, in UTF-16 units.
pub const BUFFERMAX: usize = 1024;

/// The base class's state, which a [`Segmenter`]'s hooks read: the window
/// (`buffer`), its position in the input (`offset`), the attributes and the
/// input (for `correctOffset`).
pub struct SegmentingBase {
    buffer: Vec<u16>,
    /// `length`: units in the buffer.
    length: usize,
    /// `usableLength`: the units the break iterator sees.
    usable_length: usize,
    /// `offset`: the input position of `buffer[0]`.
    offset: i32,
    atts: AttributeSource,
    input: TokenizerInput,
}

impl SegmentingBase {
    fn new() -> Self {
        SegmentingBase {
            buffer: vec![0; BUFFERMAX],
            length: 0,
            usable_length: 0,
            offset: 0,
            atts: AttributeSource::new(),
            input: TokenizerInput::new(),
        }
    }

    /// `buffer`: the window (only its first [`Self::length`] units are
    /// input).
    pub fn buffer(&self) -> &[u16] {
        &self.buffer
    }

    /// `length`.
    pub fn length(&self) -> usize {
        self.length
    }

    /// `offset`: the input position of `buffer[0]`.
    pub fn offset(&self) -> i32 {
        self.offset
    }

    /// `correctOffset`.
    pub fn correct_offset(&self, off: i32) -> i32 {
        self.input.correct_offset(off)
    }

    /// The attributes, for `incrementWord` to fill.
    pub fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.atts
    }

    /// The attributes.
    pub fn attributes(&self) -> &AttributeSource {
        &self.atts
    }
}

/// The subclass of `SegmentingTokenizerBase`: its two abstract methods,
/// plus the `reset()` and `isSafeEnd` overrides Java allows.
pub trait Segmenter: Send {
    /// `setNextSentence(sentenceStart, sentenceEnd)`: the next sentence is
    /// `base.buffer()[sentence_start..sentence_end]`.
    fn set_next_sentence(
        &mut self,
        base: &SegmentingBase,
        sentence_start: usize,
        sentence_end: usize,
    );

    /// `incrementWord()`: the next word of the current sentence into the
    /// attributes, `false` when it has none left.
    fn increment_word(&mut self, base: &mut SegmentingBase) -> Result<bool, AnalysisError>;

    /// The subclass's `reset()` (after the base's).
    fn reset(&mut self) {}

    /// `isSafeEnd(ch)`: a unit a window may end after. Java's: CR, LF, NEL,
    /// U+2028, U+2029.
    fn is_safe_end(&self, ch: u16) -> bool {
        matches!(ch, 0x000D | 0x000A | 0x0085 | 0x2028 | 0x2029)
    }
}

/// `SegmentingTokenizerBase` with its subclass `S` and its iterator `B`.
pub struct SegmentingTokenizer<S: Segmenter, B: BreakIterator + Send = SentenceBreakIterator> {
    base: SegmentingBase,
    iterator: B,
    segmenter: S,
}

impl<S: Segmenter> SegmentingTokenizer<S, SentenceBreakIterator> {
    /// The tokenizer over the JDK's sentence iterator
    /// (`BreakIterator.getSentenceInstance(Locale.ROOT)`, as Thai's and
    /// smartcn's pass).
    pub fn new(segmenter: S) -> Self {
        Self::with_iterator(segmenter, SentenceBreakIterator::new())
    }
}

impl<S: Segmenter, B: BreakIterator + Send> SegmentingTokenizer<S, B> {
    /// `SegmentingTokenizerBase(BreakIterator)`.
    pub fn with_iterator(segmenter: S, iterator: B) -> Self {
        SegmentingTokenizer {
            base: SegmentingBase::new(),
            iterator,
            segmenter,
        }
    }

    /// The subclass.
    pub fn segmenter(&self) -> &S {
        &self.segmenter
    }

    /// `findSafeEnd()`: one past the last safe end, `None` without one.
    fn find_safe_end(&self) -> Option<usize> {
        let b = &self.base;
        (0..b.length)
            .rev()
            .find(|&i| self.segmenter.is_safe_end(b.buffer[i]))
            .map(|i| i + 1)
    }

    /// `refill()`.
    fn refill(&mut self) -> Result<(), AnalysisError> {
        let b = &mut self.base;
        b.offset += b.usable_length as i32;
        let leftover = b.length - b.usable_length;
        b.buffer.copy_within(b.usable_length..b.length, 0);
        let requested = BUFFERMAX - leftover;
        let returned = read_fully(b.input.reader()?, &mut b.buffer[leftover..])?;
        b.length = returned + leftover;
        if returned < requested {
            // reader has been emptied, process the rest
            self.base.usable_length = self.base.length;
        } else {
            // still more data to be read, find a safe-stopping place
            self.base.usable_length = self.find_safe_end().unwrap_or(self.base.length);
        }
        let b = &self.base;
        self.iterator.set_text(&b.buffer[..b.usable_length]);
        Ok(())
    }

    /// `incrementSentence()`.
    fn increment_sentence(&mut self) -> Result<bool, AnalysisError> {
        if self.base.length == 0 {
            // we must refill the buffer
            return Ok(false);
        }
        loop {
            let start = self.iterator.current();
            // find the next set of boundaries
            let Some(end) = self.iterator.next() else {
                return Ok(false); // BreakIterator exhausted
            };
            self.segmenter.set_next_sentence(&self.base, start, end);
            if self.segmenter.increment_word(&mut self.base)? {
                return Ok(true);
            }
        }
    }
}

/// `SegmentingTokenizerBase.read`: fills `buf` until the reader is empty,
/// returning how many units it read.
fn read_fully(input: &mut dyn CharReader, buf: &mut [u16]) -> Result<usize, AnalysisError> {
    let mut filled = 0;
    while filled < buf.len() {
        let count = input.read(&mut buf[filled..])?;
        if count == 0 {
            break; // EOF
        }
        filled += count;
    }
    Ok(filled)
}

impl<S: Segmenter, B: BreakIterator + Send> TokenStream for SegmentingTokenizer<S, B> {
    /// A source, not a wrapper: no conditional wrapper below it.
    fn conditional_root(&mut self) -> Option<&mut dyn std::any::Any> {
        None
    }

    fn attributes(&self) -> &AttributeSource {
        &self.base.atts
    }

    fn attributes_mut(&mut self) -> &mut AttributeSource {
        &mut self.base.atts
    }

    fn increment_token(&mut self) -> Result<bool, AnalysisError> {
        if self.base.length == 0 || !self.segmenter.increment_word(&mut self.base)? {
            while !self.increment_sentence()? {
                self.refill()?;
                if self.base.length == 0 {
                    // no more bytes to read
                    return Ok(false);
                }
            }
        }
        Ok(true)
    }

    fn reset(&mut self) -> Result<(), AnalysisError> {
        self.base.input.reset();
        self.iterator.set_text(&[]);
        self.base.length = 0;
        self.base.usable_length = 0;
        self.base.offset = 0;
        self.segmenter.reset();
        Ok(())
    }

    fn end(&mut self) -> Result<(), AnalysisError> {
        self.base.atts.end_attributes();
        let b = &self.base;
        let final_offset = b.correct_offset(b.offset + b.length as i32);
        self.base.atts.set_offset(final_offset, final_offset)
    }

    fn close(&mut self) -> Result<(), AnalysisError> {
        self.base.input.close()
    }

    fn as_tokenizer(&mut self) -> Option<&mut dyn Tokenizer> {
        Some(self)
    }
}

impl<S: Segmenter, B: BreakIterator + Send> Tokenizer for SegmentingTokenizer<S, B> {
    fn set_reader(&mut self, input: Box<dyn CharReader>) -> Result<(), AnalysisError> {
        self.base.input.set_reader(input)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::token_stream::consume;

    /// Lucene's test `WholeSentenceTokenizer`: each sentence one token.
    #[derive(Default)]
    struct WholeSentence {
        bounds: Option<(usize, usize)>,
    }

    impl Segmenter for WholeSentence {
        fn set_next_sentence(&mut self, _: &SegmentingBase, start: usize, end: usize) {
            self.bounds = Some((start, end));
        }
        fn increment_word(&mut self, base: &mut SegmentingBase) -> Result<bool, AnalysisError> {
            let Some((start, end)) = self.bounds.take() else {
                return Ok(false);
            };
            let (s, e) = (
                base.correct_offset(base.offset() + start as i32),
                base.correct_offset(base.offset() + end as i32),
            );
            let term = base.buffer()[start..end].to_vec();
            let a = base.attributes_mut();
            a.clear_attributes();
            a.set_term_utf16(&term);
            a.set_offset(s, e)?;
            Ok(true)
        }
        fn reset(&mut self) {
            self.bounds = None;
        }
    }

    fn run(text: &str) -> (Vec<(String, i32, i32)>, i32) {
        let mut t = SegmentingTokenizer::new(WholeSentence::default());
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut out = Vec::new();
        let end = consume(&mut t, |a| {
            out.push((a.term().to_string(), a.start_offset(), a.end_offset()))
        })
        .unwrap();
        assert!(t.as_tokenizer().is_some());
        assert!(t.conditional_root().is_none());
        assert!(t.segmenter().bounds.is_none());
        (out, end.end_offset())
    }

    #[test]
    fn sentences_across_windows() {
        let (toks, end) = run("One. Two! three");
        assert_eq!(
            toks,
            [
                ("One. ".into(), 0, 5),
                ("Two! ".into(), 5, 10),
                ("three".into(), 10, 15)
            ]
        );
        assert_eq!(end, 15);
        // A window ends at its last newline; sentences never straddle it.
        let line = "Abc def. Ghi jkl!\n";
        let text = line.repeat(100);
        let (toks, end) = run(&text);
        assert_eq!(end, text.len() as i32);
        assert_eq!(toks.len(), 200);
        assert!(toks.iter().all(|(t, s, e)| (e - s) as usize == t.len()));
        // No safe end: the window is cut at 1,024 units.
        let text = "a".repeat(2500);
        let (toks, _) = run(&text);
        let lens: Vec<usize> = toks.iter().map(|t| t.0.len()).collect();
        assert_eq!(lens, [1024, 1024, 452]);
        assert_eq!(run(""), (vec![], 0));
    }

    #[test]
    fn reading_before_reset_is_an_error() {
        let mut t = SegmentingTokenizer::new(WholeSentence::default());
        assert!(t.increment_token().is_err());
        assert!(WholeSentence::default().is_safe_end(0x2029));
        assert!(!WholeSentence::default().is_safe_end(u16::from(b'.')));
    }
}
