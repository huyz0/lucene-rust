//! OpenSearch's unified highlighter: `org.opensearch.lucene.search.uhighlight`
//! (`CustomUnifiedHighlighter`, `CustomFieldHighlighter`,
//! `CustomPassageFormatter`, `BoundedBreakIteratorScanner`, `Snippet`) over
//! this crate's [`UnifiedHighlighter`].
//!
//! What OpenSearch changes from Lucene's highlighter:
//!
//! - the offset source is the mapping's (postings with offsets, term
//!   vectors, or re-analysis), for the highlighted field and every field it
//!   masks, not the index's field infos;
//! - `WEIGHT_MATCHES` is off;
//! - passages are cut by the pre-LUCENE-9093 loop: a passage starts at the
//!   break before the match's start, not around its centre, and always comes
//!   back in start order;
//! - a document without a match shows its first `no_match_size` characters
//!   (to the next word boundary) of its first non-empty value, or nothing;
//! - each passage is its own [`Snippet`], trimmed, with its score and
//!   whether it holds a match.
//!
//! Differences from OpenSearch: `BoundedBreakIteratorScanner` throws on
//! `first`/`last`/`next`/`previous` and on offsets that go backwards; this
//! one answers its main iterator's boundary instead (the unified
//! highlighter never asks).

use std::sync::Arc;

use lucene_search::query::Clause;
use lucene_search::query_visitor::extract_terms;

use super::break_iterators::{
    sentence_instance, word_instance, BreakIterator, CustomSeparatorBreakIterator, DONE,
};
use super::passage::{EmptyOffsets, OffsetsEnum, Passage};
use super::strategies::{multi_fields_enum, FieldMatcher, LeafCtx, Strategy, MULTIVAL_SEP_CHAR};
use super::{FieldHighlighter, MaskedFields, UnifiedHighlighter};
use crate::highlighter::OffsetSource;
use crate::{Error, Result};

/// `HighlightUtils.PARAGRAPH_SEPARATOR` (U+2029).
pub const PARAGRAPH_SEPARATOR: u16 = 0x2029;

/// `Integer.MAX_VALUE - 1`: the passages kept when every fragment is asked
/// for (`number_of_fragments: 0`).
pub const ALL_PASSAGES: i32 = i32::MAX - 1;

/// `org.opensearch.lucene.search.uhighlight.Snippet`: one formatted passage.
#[derive(Debug, Clone, PartialEq)]
pub struct Snippet {
    /// The passage's text, marked up and trimmed, as UTF-16.
    pub text: Vec<u16>,
    /// The passage's score; NaN for a no-match summary.
    pub score: f32,
    /// Whether the passage holds a match.
    pub highlighted: bool,
}

// ---------------------------------------------------------------------------
// BoundedBreakIteratorScanner
// ---------------------------------------------------------------------------

/// `BoundedBreakIteratorScanner`: passages cut at the main iterator's
/// (sentence) boundaries, grown up to `max_len` characters, and a sentence
/// longer than that cut at the inner (word) iterator's boundaries around the
/// match. Only `preceding(offset)` with increasing offsets, each followed by
/// `following(offset - 1)`, are meaningful: the calls the unified
/// highlighter makes.
pub struct BoundedBreakIteratorScanner {
    main: Box<dyn BreakIterator>,
    inner: Box<dyn BreakIterator>,
    max_len: i32,
    last_preceding_offset: i32,
    window_start: i32,
    window_end: i32,
    inner_start: i32,
    inner_end: i32,
}

impl BoundedBreakIteratorScanner {
    /// `BoundedBreakIteratorScanner.getSentence(Locale.ROOT, maxLen)`.
    pub fn sentence(max_len: i32) -> Self {
        BoundedBreakIteratorScanner {
            main: sentence_instance(),
            inner: word_instance(),
            max_len,
            last_preceding_offset: -1,
            window_start: -1,
            window_end: -1,
            inner_start: -1,
            inner_end: 0,
        }
    }

    fn reset(&mut self) {
        self.last_preceding_offset = -1;
        self.window_start = -1;
        self.window_end = -1;
        self.inner_start = -1;
        self.inner_end = 0;
    }
}

impl BreakIterator for BoundedBreakIteratorScanner {
    fn set_text(&mut self, text: &[u16]) {
        self.reset();
        self.main.set_text(text);
        self.inner.set_text(text);
    }
    fn text_end(&self) -> i32 {
        self.main.text_end()
    }
    /// The last offset of the current split.
    fn current(&self) -> i32 {
        self.inner_end
    }
    fn first(&mut self) -> i32 {
        self.main.first()
    }
    fn last(&mut self) -> i32 {
        self.main.last()
    }
    fn next(&mut self) -> i32 {
        self.main.next()
    }
    fn previous(&mut self) -> i32 {
        self.main.previous()
    }
    fn preceding(&mut self, offset: i32) -> i32 {
        if offset > self.window_start && offset < self.window_end {
            self.inner_start = self.inner_end;
            self.inner_end = self.window_end;
        } else {
            self.inner_start = self.main.preceding(offset);
            self.window_start = self.inner_start;
            self.inner_end = self.main.following(offset - 1);
            self.window_end = self.inner_end;
            // expand to the next break until we reach max_len
            while self.inner_end - self.inner_start < self.max_len {
                let new_end = self.main.following(self.inner_end);
                if new_end == DONE || new_end - self.inner_start > self.max_len {
                    break;
                }
                self.inner_end = new_end;
                self.window_end = new_end;
            }
        }
        if self.inner_end - self.inner_start > self.max_len {
            // the current split is too big, so starting from the current
            // term we try to find boundaries on the left first
            if offset - self.max_len > self.inner_start {
                self.inner_start = self
                    .inner_start
                    .max(self.inner.preceding(offset - self.max_len));
            }
            // and then we try to expand the passage to the right with the
            // remaining size
            let remaining = (self.max_len - (offset - self.inner_start)).max(0);
            if offset + remaining < self.window_end {
                self.inner_end = self
                    .window_end
                    .min(self.inner.following(offset + remaining));
            }
        }
        self.last_preceding_offset = offset - 1;
        self.inner_start
    }
    /// Only after `preceding(offset + 1)`.
    fn following(&mut self, _offset: i32) -> i32 {
        self.inner_end
    }
}

// ---------------------------------------------------------------------------
// The highlighter
// ---------------------------------------------------------------------------

/// The break iterator passages are cut with (`UnifiedHighlighter.getBreakIterator`
/// in OpenSearch's fetch phase).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Breaks {
    /// `CustomSeparatorBreakIterator(MULTIVAL_SEP_CHAR)`: one passage per
    /// value (`number_of_fragments: 0`, or a field that is not tokenized).
    Values,
    /// `BreakIterator.getSentenceInstance(locale)`.
    Sentence,
    /// `BoundedBreakIteratorScanner.getSentence(locale, max_len)`.
    BoundedSentence(i32),
    /// `BreakIterator.getWordInstance(locale)`.
    Word,
}

impl Breaks {
    fn make(self) -> Box<dyn BreakIterator> {
        match self {
            Breaks::Values => Box::new(CustomSeparatorBreakIterator::new(MULTIVAL_SEP_CHAR)),
            Breaks::Sentence => sentence_instance(),
            Breaks::BoundedSentence(max_len) => {
                Box::new(BoundedBreakIteratorScanner::sentence(max_len))
            }
            Breaks::Word => word_instance(),
        }
    }
}

/// How a field is highlighted: `UnifiedHighlighter.buildHighlighter`'s
/// inputs.
pub struct Options {
    /// The field.
    pub field: String,
    /// The offset source the mapping implies.
    pub offset_source: OffsetSource,
    /// The break iterator.
    pub breaks: Breaks,
    /// `number_of_fragments`, [`ALL_PASSAGES`] for every one.
    pub max_passages: i32,
    /// `no_match_size`.
    pub no_match_size: i32,
    /// The first of `pre_tags`.
    pub pre_tag: String,
    /// The first of `post_tags`.
    pub post_tag: String,
    /// `encoder: html`.
    pub html: bool,
    /// Which query fields count as the highlighted one.
    pub field_matcher: FieldMatcher,
    /// `matched_fields`, masked into the highlighted one.
    pub masked_fields: Option<MaskedFields>,
}

/// `CustomUnifiedHighlighter`, built once per field and search: the field's
/// strategies and passage logic for the query.
pub struct CustomUnifiedHighlighter {
    strategies: Vec<Strategy>,
    highlighter: FieldHighlighter,
    no_match_size: i32,
    pre_tag: Vec<u16>,
    post_tag: Vec<u16>,
    html: bool,
}

impl CustomUnifiedHighlighter {
    /// `new CustomUnifiedHighlighter(...)`: `getFieldHighlighter(field,
    /// query, extractTerms(query), maxPassages)` without `WEIGHT_MATCHES`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for an `ANALYSIS` source without an
    /// analyzer, or one whose offset gap is not 1.
    pub fn new(
        options: Options,
        query: &Clause,
        analyzer: Option<Arc<lucene_analysis::Analyzer>>,
    ) -> Result<Self> {
        let breaks = options.breaks;
        let mut uh = UnifiedHighlighter::new(None, analyzer)
            .with_flags(true, true, true, false)
            .with_offset_source(options.offset_source)
            .with_field_matcher(options.field_matcher)
            .with_break_iterator(Arc::new(move || breaks.make()));
        if let Some(masked) = options.masked_fields {
            uh = uh.with_masked_fields(masked);
        }
        let all_terms = extract_terms(query);
        let (strategies, highlighter) =
            uh.field_highlighter(&options.field, query, &all_terms, options.max_passages)?;
        Ok(CustomUnifiedHighlighter {
            strategies,
            highlighter,
            no_match_size: options.no_match_size,
            pre_tag: options.pre_tag.encode_utf16().collect(),
            post_tag: options.post_tag.encode_utf16().collect(),
            html: options.html,
        })
    }

    /// Whether the query can match nothing in the field
    /// (`NoOpOffsetStrategy`): with `no_match_size` 0 there is then nothing
    /// to show for any document.
    pub fn is_noop(&self) -> bool {
        self.strategies.len() == 1 && self.strategies[0].source == OffsetSource::NoneNeeded
    }

    /// The fields whose strategy re-analyzes the content (`ANALYSIS`), so
    /// whose index analyzer the highlighter needs.
    pub fn analysis_fields(&self) -> impl Iterator<Item = &str> {
        self.strategies
            .iter()
            .filter(|s| s.source == OffsetSource::Analysis)
            .map(Strategy::field)
    }

    /// `no_match_size`.
    pub fn no_match_size(&self) -> i32 {
        self.no_match_size
    }

    /// The offset source the field is read from, once the query is known
    /// (`getOptimizedOffsetSource`).
    pub fn offset_source(&self) -> OffsetSource {
        self.strategies[0].source
    }

    /// `highlightField(reader, docId, loadFieldValue)` once the value is
    /// loaded (`content`, the values joined by [`MULTIVAL_SEP_CHAR`]): the
    /// snippets, empty when there is nothing to show. The caller checks
    /// [`Self::is_noop`] and the analyzed-length limits first.
    ///
    /// # Errors
    /// What reading the offsets fails with, and
    /// [`Error::IllegalArgument`] for a field indexed without offsets.
    pub fn highlight(
        &mut self,
        ctx: Option<&LeafCtx<'_>>,
        content: &[u16],
    ) -> Result<Vec<Snippet>> {
        // FieldHighlighter.highlightFieldForDoc
        if content.is_empty() {
            return Ok(Vec::new());
        }
        self.highlighter.break_iterator.set_text(content);
        let found = if self.strategies.len() == 1 {
            self.strategies[0].offsets_enum(ctx, content)?
        } else {
            multi_fields_enum(&self.strategies, ctx, content)?
        };
        let mut off: Box<dyn OffsetsEnum> = found.unwrap_or_else(|| Box::new(EmptyOffsets));
        let mut passages = self.highlight_offsets_enums(off.as_mut())?;
        if passages.is_empty() {
            passages = self.summary_passages_no_highlight(content);
        }
        Ok(self.format(&passages, content))
    }

    /// `CustomFieldHighlighter.highlightOffsetsEnums`: the pre-LUCENE-9093
    /// loop, a passage from the break before the match.
    fn highlight_offsets_enums(&mut self, off: &mut dyn OffsetsEnum) -> Result<Vec<Passage>> {
        let h = &mut self.highlighter;
        let content_length = h.break_iterator.text_end();
        if !off.next_position()? {
            return Ok(Vec::new());
        }
        if h.max_passages < 1 {
            // Java's PriorityQueue refuses a capacity below 1
            return Err(Error::IllegalArgument(format!(
                "maxPassages must be at least 1 to highlight a match, got {}",
                h.max_passages
            )));
        }
        let mut queue: Vec<Passage> = Vec::new();
        let mut passage = Passage::new();
        loop {
            let start = off.start_offset();
            if start == -1 {
                return Err(Error::IllegalArgument(format!(
                    "field '{}' was indexed without offsets, cannot highlight",
                    h.field
                )));
            }
            let end = off.end_offset();
            if !(start < content_length && end > content_length) {
                // see if this term should be part of a new passage
                if start >= passage.end_offset() {
                    passage = h.maybe_add_passage(&mut queue, passage, content_length);
                    // if we exceed the limit, we are done
                    if start >= content_length {
                        break;
                    }
                    passage.set_start_offset(h.break_iterator.preceding(start + 1).max(0));
                    passage.set_end_offset(h.break_iterator.following(start).min(content_length));
                }
                let term = off.term().unwrap_or_default().to_vec();
                passage.add_match(start, end, &term, off.freq());
            }
            if !off.next_position()? {
                break;
            }
        }
        h.maybe_add_passage(&mut queue, passage, content_length);
        // sort in ascending order (a stable sort over distinct starts)
        queue.sort_by_key(Passage::start_offset);
        Ok(queue)
    }

    /// `CustomFieldHighlighter.getSummaryPassagesNoHighlight`: the first
    /// non-empty value, cut at the first word boundary after
    /// `no_match_size` characters.
    fn summary_passages_no_highlight(&self, content: &[u16]) -> Vec<Passage> {
        if self.no_match_size <= 0 {
            return Vec::new();
        }
        let len = content.len();
        let mut pos = 0;
        while pos < len && content[pos] == MULTIVAL_SEP_CHAR {
            pos += 1;
        }
        if pos >= len {
            return Vec::new();
        }
        let mut end = content[pos..]
            .iter()
            .position(|&c| c == MULTIVAL_SEP_CHAR)
            .map_or(len, |p| p + pos);
        let no_match = usize::try_from(self.no_match_size).unwrap_or(0);
        if no_match + pos < end {
            let mut words = word_instance();
            words.set_text(content);
            // the next word boundary after no_match_size
            let following = words.following(i32::try_from(no_match + pos).unwrap_or(i32::MAX));
            end = usize::try_from(following).unwrap_or(len);
        }
        let mut passage = Passage::new();
        passage.set_score(f32::NAN);
        passage.set_start_offset(i32::try_from(pos).unwrap_or(i32::MAX));
        passage.set_end_offset(i32::try_from(end).unwrap_or(i32::MAX));
        vec![passage]
    }

    /// `CustomPassageFormatter.format`.
    fn format(&self, passages: &[Passage], content: &[u16]) -> Vec<Snippet> {
        let mut snippets = Vec::with_capacity(passages.len());
        for passage in passages {
            let mut sb: Vec<u16> = Vec::new();
            let mut pos = passage.start_offset();
            let n = passage.num_matches();
            let mut i = 0;
            while i < n {
                let start = passage.match_starts()[i];
                // append content before this start
                self.append(&mut sb, content, pos, start);
                let mut end = passage.match_ends()[i];
                // look ahead to expand 'end' past all overlapping
                while i + 1 < n && passage.match_starts()[i + 1] < end {
                    i += 1;
                    end = passage.match_ends()[i];
                }
                // in case the match straddles past the passage
                end = end.min(passage.end_offset());
                sb.extend_from_slice(&self.pre_tag);
                self.append(&mut sb, content, start, end);
                sb.extend_from_slice(&self.post_tag);
                pos = end;
                i += 1;
            }
            // a "term" from the analyzer could span a sentence boundary
            self.append(&mut sb, content, pos, pos.max(passage.end_offset()));
            // the separator between values, at the end of the snippet
            if matches!(
                sb.last(),
                Some(&PARAGRAPH_SEPARATOR) | Some(&MULTIVAL_SEP_CHAR)
            ) {
                sb.pop();
            }
            snippets.push(Snippet {
                text: trim(&sb).to_vec(),
                score: passage.score(),
                highlighted: n > 0,
            });
        }
        snippets
    }

    /// `encoder.encodeText(content.substring(start, end))`.
    fn append(&self, dest: &mut Vec<u16>, content: &[u16], start: i32, end: i32) {
        let s = usize::try_from(start).unwrap_or(0).min(content.len());
        let e = usize::try_from(end).unwrap_or(0).min(content.len());
        let slice = &content[s..e.max(s)];
        if !self.html {
            dest.extend_from_slice(slice);
            return;
        }
        // SimpleHTMLEncoder.htmlEncode
        for &ch in slice {
            let rep: &[u16] = match ch {
                0x22 => &QUOT,
                0x26 => &AMP,
                0x3C => &LT,
                0x3E => &GT,
                0x27 => &APOS,
                0x2F => &SLASH,
                _ => {
                    dest.push(ch);
                    continue;
                }
            };
            dest.extend_from_slice(rep);
        }
    }
}

const fn ascii<const N: usize>(s: &[u8; N]) -> [u16; N] {
    let mut out = [0u16; N];
    let mut i = 0;
    while i < N {
        out[i] = s[i] as u16;
        i += 1;
    }
    out
}

const QUOT: [u16; 6] = ascii(b"&quot;");
const AMP: [u16; 5] = ascii(b"&amp;");
const LT: [u16; 4] = ascii(b"&lt;");
const GT: [u16; 4] = ascii(b"&gt;");
const APOS: [u16; 6] = ascii(b"&#x27;");
const SLASH: [u16; 6] = ascii(b"&#x2F;");

/// `String.trim()`: without the leading and trailing units up to U+0020.
fn trim(s: &[u16]) -> &[u16] {
    let start = s.iter().position(|&c| c > 0x20).unwrap_or(s.len());
    let end = s.iter().rposition(|&c| c > 0x20).map_or(start, |p| p + 1);
    &s[start..end]
}

#[cfg(test)]
mod tests;
