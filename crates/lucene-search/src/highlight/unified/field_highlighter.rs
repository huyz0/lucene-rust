//! `FieldHighlighter`: scores a document's matches against the passages a
//! break iterator cuts, keeps the best, and formats them.

use std::cmp::Ordering;

use super::super::{Error, Result};
use super::break_iterators::{BreakIterator, DONE};
use super::passage::{HighlightObject, OffsetsEnum, Passage, PassageFormatter, PassageScorer};

/// The order the chosen passages are returned in (`passageSortComparator`).
pub type PassageOrder = fn(&Passage, &Passage) -> Ordering;

/// `UnifiedHighlighter.DEFAULT_PASSAGE_SORT_COMPARATOR`: by start offset.
pub fn passage_by_start_offset(a: &Passage, b: &Passage) -> Ordering {
    a.start_offset().cmp(&b.start_offset())
}

/// `FieldHighlighter`'s passage logic for one field. The offset strategy is
/// kept apart: [`FieldHighlighter::highlight_offsets_enums`] takes the
/// [`OffsetsEnum`] it produced.
pub struct FieldHighlighter {
    /// The field being highlighted.
    pub field: String,
    /// Stateful, as Java's.
    pub break_iterator: Box<dyn BreakIterator>,
    /// The scorer.
    pub passage_scorer: PassageScorer,
    /// `maxPassages`.
    pub max_passages: i32,
    /// `maxNoHighlightPassages`; `-1` falls back to `max_passages`.
    pub max_no_highlight_passages: i32,
    /// The formatter.
    pub passage_formatter: Box<dyn PassageFormatter>,
    /// `passageSortComparator`.
    pub passage_sort_comparator: PassageOrder,
}

impl FieldHighlighter {
    /// `getSummaryPassagesNoHighlight(maxPassages)`: the leading passages as
    /// they are, for a document with nothing to highlight.
    pub fn summary_passages_no_highlight(&mut self, max_passages: i32) -> Vec<Passage> {
        let mut passages = Vec::new();
        let mut pos = self.break_iterator.current();
        debug_assert_eq!(pos, 0);
        while i32::try_from(passages.len()).unwrap_or(i32::MAX) < max_passages {
            let next = self.break_iterator.next();
            if next == DONE {
                break;
            }
            let mut passage = Passage::new();
            passage.set_start_offset(pos);
            passage.set_end_offset(next);
            passages.push(passage);
            pos = next;
        }
        passages
    }

    /// `highlightOffsetsEnums(off)`: the algorithm treats sentence snippets
    /// as miniature documents, scoring each as
    /// `norm(sentenceStartOffset) * sum(weight * tf(freq))`.
    ///
    /// The break iterator must already hold the content (`setText`).
    pub fn highlight_offsets_enums(&mut self, off: &mut dyn OffsetsEnum) -> Result<Vec<Passage>> {
        let content_length = self.break_iterator.text_end();
        if !off.next_position()? {
            return Ok(Vec::new());
        }
        // Java: `new PriorityQueue<>(min(64, maxPassages + 1))` refuses a
        // `maxPassages` below 0, and the queue's `peek()` of a zero-length
        // queue is null, a `NullPointerException` at the first passage
        if self.max_passages < 1 {
            return Err(Error::IllegalArgument(format!(
                "maxPassages must be at least 1 to highlight a match, got {}",
                self.max_passages
            )));
        }
        // the queue keeps the `max_passages` best; the head is the weakest
        let mut queue: Vec<Passage> = Vec::new();
        // the current passage in progress: reset or added to the queue
        let mut passage = Passage::new();
        let mut last_passage_end = 0i32;
        loop {
            let start = off.start_offset();
            if start == -1 {
                return Err(Error::IllegalArgument(format!(
                    "field '{}' was indexed without offsets, cannot highlight",
                    self.field
                )));
            }
            let end = off.end_offset();
            // a match that straddles the end of the content is skipped
            let straddles = start < content_length && end > content_length;
            if !straddles {
                // see if this term should be part of a new passage
                if start >= passage.end_offset() {
                    passage = self.maybe_add_passage(&mut queue, passage, content_length);
                    // if we exceed the limit, we are done
                    if start >= content_length {
                        break;
                    }
                    // find the fragment from the middle of the match, so the
                    // result's length may be closer to fragsize
                    let center = start + (end - start) / 2;
                    let preceding = self.break_iterator.preceding((start + 1).max(center));
                    passage.set_start_offset(start.min(preceding.max(last_passage_end)));
                    let following = self.break_iterator.following((end - 1).min(center));
                    last_passage_end = end.max(following.min(content_length));
                    passage.set_end_offset(last_passage_end);
                }
                // add this term to the passage
                let term = off.term().expect("a highlighted match has a term").to_vec();
                passage.add_match(start, end, &term, off.freq());
            }
            if !off.next_position()? {
                break;
            }
        }
        self.maybe_add_passage(&mut queue, passage, content_length);
        queue.sort_by(self.passage_sort_comparator);
        Ok(queue)
    }

    /// The queue's order: weakest first, then the earlier start.
    fn weaker(left: &Passage, right: &Passage) -> Ordering {
        left.score()
            .partial_cmp(&right.score())
            .unwrap_or(Ordering::Equal)
            .then(left.start_offset().cmp(&right.start_offset()))
    }

    fn head(queue: &[Passage]) -> usize {
        let mut head = 0;
        for i in 1..queue.len() {
            if Self::weaker(&queue[i], &queue[head]) == Ordering::Less {
                head = i;
            }
        }
        head
    }

    fn maybe_add_passage(
        &self,
        queue: &mut Vec<Passage>,
        mut passage: Passage,
        content_length: i32,
    ) -> Passage {
        if passage.start_offset() == -1 {
            // empty passage, we can ignore it
            return passage;
        }
        passage.set_score(self.passage_scorer.score(&passage, content_length));
        let full = i32::try_from(queue.len()).unwrap_or(i32::MAX) == self.max_passages;
        if full && passage.score() < queue[Self::head(queue)].score() {
            // can't compete, just reset it
            passage.reset();
            return passage;
        }
        queue.push(passage);
        if i32::try_from(queue.len()).unwrap_or(i32::MAX) > self.max_passages {
            let head = Self::head(queue);
            let mut evicted = queue.swap_remove(head);
            evicted.reset();
            evicted
        } else {
            Passage::new()
        }
    }

    /// `highlightFieldForDoc` after the offsets are known: what the
    /// formatter makes of the passages, or `None` when the content is empty
    /// or nothing can be shown.
    pub fn highlight_content(
        &mut self,
        content: &[u16],
        off: &mut dyn OffsetsEnum,
    ) -> Result<Option<HighlightObject>> {
        let mut passages = self.highlight_offsets_enums(off)?;
        if passages.is_empty() {
            // no passages were returned, so ask for a default summary
            let n = if self.max_no_highlight_passages == -1 {
                self.max_passages
            } else {
                self.max_no_highlight_passages
            };
            passages = self.summary_passages_no_highlight(n);
        }
        if passages.is_empty() {
            Ok(None)
        } else {
            Ok(Some(
                self.passage_formatter.format_object(&passages, content),
            ))
        }
    }
}
