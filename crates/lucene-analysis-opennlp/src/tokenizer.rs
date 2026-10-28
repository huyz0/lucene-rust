//! `OpenNLPTokenizer`: `SegmentingTokenizerBase` over
//! `OpenNLPSentenceBreakIterator` (the sentence detector's sentence starts
//! as boundaries), each sentence split by the OpenNLP tokenizer, every
//! token carrying its sentence's index (`SentenceAttribute`) for the
//! filters after it.

use lucene_analysis::util::{BreakIterator, Segmenter, SegmentingBase, SegmentingTokenizer};
use lucene_analysis::AnalysisError;

use crate::tools::sentdetect::{SentenceDetector, Span};
use crate::tools::tokenize::Tokenizer;

/// `OpenNLPSentenceBreakIterator`, as `SegmentingTokenizerBase` drives it
/// (`setText`, `current`, `next`).
pub struct OpenNlpSentenceBreakIterator {
    detector: SentenceDetector,
    /// The text's length (`getEndIndex()`; the text starts at 0).
    end: usize,
    index: usize,
    current_sentence: usize,
    sentence_starts: Vec<usize>,
}

impl OpenNlpSentenceBreakIterator {
    /// The iterator over `detector`'s sentences.
    pub fn new(detector: SentenceDetector) -> Self {
        OpenNlpSentenceBreakIterator {
            detector,
            end: 0,
            index: 0,
            current_sentence: 0,
            sentence_starts: Vec::new(),
        }
    }

    /// `last()`.
    // ARITH: the sentence count is not zero there.
    #[allow(clippy::arithmetic_side_effects)]
    fn last(&mut self) -> usize {
        if self.sentence_starts.is_empty() {
            self.current_sentence = 0;
            self.index = 0;
        } else {
            self.current_sentence = self.sentence_starts.len() - 1;
            self.index = self.end;
        }
        self.index
    }
}

impl BreakIterator for OpenNlpSentenceBreakIterator {
    fn set_text(&mut self, text: &[u16]) {
        self.end = text.len();
        self.index = 0;
        self.current_sentence = 0;
        self.sentence_starts = self
            .detector
            .sent_pos_detect(text)
            .iter()
            .map(|s| s.start)
            .collect();
    }

    fn current(&self) -> usize {
        self.index
    }

    // ARITH: sentence indices below the sentence count.
    #[allow(clippy::arithmetic_side_effects)]
    fn next(&mut self) -> Option<usize> {
        if self.index == self.end || self.sentence_starts.is_empty() {
            None
        } else if self.current_sentence + 1 < self.sentence_starts.len() {
            self.current_sentence += 1;
            self.index = self.sentence_starts[self.current_sentence];
            Some(self.index)
        } else {
            Some(self.last())
        }
    }
}

/// `OpenNLPTokenizer`'s `setNextSentence`/`incrementWord`.
pub struct OpenNlpSegmenter {
    tokenizer: Tokenizer,
    term_spans: Option<Vec<Span>>,
    term_num: usize,
    sentence_start: usize,
    sentence_index: i32,
}

impl Segmenter for OpenNlpSegmenter {
    fn set_next_sentence(
        &mut self,
        base: &SegmentingBase,
        sentence_start: usize,
        sentence_end: usize,
    ) -> Result<(), AnalysisError> {
        self.sentence_start = sentence_start;
        let sentence = &base.buffer()[sentence_start..sentence_end];
        self.term_spans = Some(self.tokenizer.tokenize_pos(sentence));
        self.term_num = 0;
        self.sentence_index = self.sentence_index.wrapping_add(1);
        Ok(())
    }

    // ARITH: offsets within the 1,024-unit window.
    #[allow(clippy::arithmetic_side_effects)]
    fn increment_word(&mut self, base: &mut SegmentingBase) -> Result<bool, AnalysisError> {
        let Some(spans) = &self.term_spans else {
            return Ok(false);
        };
        let Some(&term) = spans.get(self.term_num) else {
            return Ok(false);
        };
        let start = self.sentence_start + term.start;
        let end = self.sentence_start + term.end;
        // offsets are i32 in Lucene: the window is at most BUFFERMAX units
        let (s, e) = (
            base.correct_offset(base.offset() + start as i32),
            base.correct_offset(base.offset() + end as i32),
        );
        let index = self.sentence_index;
        let (buffer, a) = base.buffer_and_attributes_mut();
        a.clear_attributes();
        a.set_term_utf16(&buffer[start..end]);
        a.set_offset(s, e)?;
        a.set_sentence_index(index);
        self.term_num += 1;
        Ok(true)
    }

    fn reset(&mut self) {
        self.term_spans = None;
        self.term_num = 0;
        self.sentence_start = 0;
        self.sentence_index = -1;
    }
}

/// `OpenNLPTokenizer`.
pub type OpenNlpTokenizer = SegmentingTokenizer<OpenNlpSegmenter, OpenNlpSentenceBreakIterator>;

/// `new OpenNLPTokenizer(factory, sentenceOp, tokenizerOp)`.
pub fn open_nlp_tokenizer(detector: SentenceDetector, tokenizer: Tokenizer) -> OpenNlpTokenizer {
    SegmentingTokenizer::with_iterator(
        OpenNlpSegmenter {
            tokenizer,
            term_spans: None,
            term_num: 0,
            sentence_start: 0,
            sentence_index: -1,
        },
        OpenNlpSentenceBreakIterator::new(detector),
    )
}
