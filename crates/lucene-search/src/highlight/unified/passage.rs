//! `Passage`, `PassageScorer`, `PassageFormatter` / `DefaultPassageFormatter`
//! and `OffsetsEnum`: what `FieldHighlighter` scores and formats.

use std::cmp::Ordering;
use std::collections::HashMap;

use super::super::Result;

/// `org.apache.lucene.search.uhighlight.Passage`: a span of the text and the
/// matches inside it. Offsets are UTF-16 code units.
#[derive(Debug, Clone)]
pub struct Passage {
    start_offset: i32,
    end_offset: i32,
    score: f32,
    match_starts: Vec<i32>,
    match_ends: Vec<i32>,
    match_terms: Vec<Vec<u8>>,
    match_term_freq_in_doc: Vec<i32>,
}

impl Default for Passage {
    fn default() -> Self {
        Self::new()
    }
}

impl Passage {
    /// `new Passage()`.
    pub fn new() -> Self {
        Passage {
            start_offset: -1,
            end_offset: -1,
            score: 0.0,
            match_starts: Vec::new(),
            match_ends: Vec::new(),
            match_terms: Vec::new(),
            match_term_freq_in_doc: Vec::new(),
        }
    }

    /// `addMatch(startOffset, endOffset, term, termFreqInDoc)`.
    pub fn add_match(&mut self, start: i32, end: i32, term: &[u8], term_freq_in_doc: i32) {
        debug_assert!(start >= self.start_offset && start <= self.end_offset);
        self.match_starts.push(start);
        self.match_ends.push(end);
        self.match_terms.push(term.to_vec());
        self.match_term_freq_in_doc.push(term_freq_in_doc);
    }

    /// `reset()`.
    pub fn reset(&mut self) {
        self.start_offset = -1;
        self.end_offset = -1;
        self.score = 0.0;
        self.match_starts.clear();
        self.match_ends.clear();
        self.match_terms.clear();
        self.match_term_freq_in_doc.clear();
    }

    /// `getStartOffset()`.
    pub fn start_offset(&self) -> i32 {
        self.start_offset
    }

    /// `getEndOffset()`.
    pub fn end_offset(&self) -> i32 {
        self.end_offset
    }

    /// `getLength()`.
    pub fn length(&self) -> i32 {
        self.end_offset - self.start_offset
    }

    /// `getScore()`.
    pub fn score(&self) -> f32 {
        self.score
    }

    /// `setScore(score)`.
    pub fn set_score(&mut self, score: f32) {
        self.score = score;
    }

    /// `getNumMatches()`.
    pub fn num_matches(&self) -> usize {
        self.match_starts.len()
    }

    /// `getMatchStarts()`, trimmed to `getNumMatches()`.
    pub fn match_starts(&self) -> &[i32] {
        &self.match_starts
    }

    /// `getMatchEnds()`, trimmed to `getNumMatches()`.
    pub fn match_ends(&self) -> &[i32] {
        &self.match_ends
    }

    /// `getMatchTerms()`, trimmed to `getNumMatches()`.
    pub fn match_terms(&self) -> &[Vec<u8>] {
        &self.match_terms
    }

    /// `getMatchTermFreqsInDoc()`, trimmed to `getNumMatches()`.
    pub fn match_term_freqs_in_doc(&self) -> &[i32] {
        &self.match_term_freq_in_doc
    }

    /// `setStartOffset(startOffset)`.
    pub fn set_start_offset(&mut self, start: i32) {
        self.start_offset = start;
    }

    /// `setEndOffset(endOffset)`.
    pub fn set_end_offset(&mut self, end: i32) {
        debug_assert!(self.start_offset <= end);
        self.end_offset = end;
    }
}

impl std::fmt::Display for Passage {
    /// `Passage.toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "Passage[{}-{}]{{", self.start_offset, self.end_offset)?;
        for i in 0..self.num_matches() {
            if i != 0 {
                f.write_str(",")?;
            }
            write!(
                f,
                "{}[{}-{}]",
                String::from_utf8_lossy(&self.match_terms[i]),
                self.match_starts[i] - self.start_offset,
                self.match_ends[i] - self.start_offset
            )?;
        }
        write!(f, "}}score={}", self.score)
    }
}

/// `PassageScorer`: BM25-flavoured scoring that treats each passage as a
/// miniature document.
#[derive(Debug, Clone, Copy)]
pub struct PassageScorer {
    k1: f32,
    b: f32,
    pivot: f32,
}

impl Default for PassageScorer {
    /// 1.2 and 0.75 are the BM25 defaults, 87 a typical English sentence
    /// length.
    fn default() -> Self {
        Self::new(1.2, 0.75, 87.0)
    }
}

impl PassageScorer {
    /// `new PassageScorer(k1, b, pivot)`.
    pub fn new(k1: f32, b: f32, pivot: f32) -> Self {
        PassageScorer { k1, b, pivot }
    }

    /// `weight(contentLength, totalTermFreq)`.
    pub fn weight(&self, content_length: i32, total_term_freq: i32) -> f32 {
        // approximate #docs from content length
        let num_docs = 1.0 + content_length as f32 / self.pivot;
        // numDocs is widened to double by `numDocs + 0.5D`
        (self.k1 + 1.0)
            * (1.0 + (f64::from(num_docs) + 0.5) / (f64::from(total_term_freq) + 0.5)).ln() as f32
    }

    /// `tf(freq, passageLen)`.
    pub fn tf(&self, freq: i32, passage_len: i32) -> f32 {
        let norm = self.k1 * ((1.0 - self.b) + self.b * (passage_len as f32 / self.pivot));
        freq as f32 / (freq as f32 + norm)
    }

    /// `norm(passageStart)`.
    pub fn norm(&self, passage_start: i32) -> f32 {
        // `pivot + passageStart` is float arithmetic before the log
        1.0 + 1.0 / f64::from(self.pivot + passage_start as f32).ln() as f32
    }

    /// `score(passage, contentLength)`.
    pub fn score(&self, passage: &Passage, content_length: i32) -> f32 {
        let mut index: HashMap<&[u8], usize> = HashMap::new();
        let hit_count = passage.num_matches();
        let mut term_freqs_in_passage = vec![0i32; hit_count];
        let mut term_freqs_in_doc = vec![0i32; hit_count];
        let mut unique = 0usize;
        for i in 0..hit_count {
            let term = passage.match_terms[i].as_slice();
            let idx = *index.entry(term).or_insert_with(|| {
                term_freqs_in_doc[unique] = passage.match_term_freq_in_doc[i];
                unique += 1;
                unique - 1
            });
            term_freqs_in_passage[idx] += 1;
        }
        let mut score = 0f64;
        for i in 0..unique {
            score += f64::from(
                self.tf(term_freqs_in_passage[i], passage.length())
                    * self.weight(content_length, term_freqs_in_doc[i]),
            );
        }
        score *= f64::from(self.norm(passage.start_offset()));
        score as f32
    }
}

/// What `PassageFormatter.format` returns: Java's `Object`, here anything
/// that prints (`toString()`, which `highlightFields` takes) and can be
/// downcast to what the formatter made.
pub trait HighlightValue: std::fmt::Display + std::any::Any + Send + Sync {
    /// The value, for a downcast.
    fn as_any(&self) -> &dyn std::any::Any;
}

impl<T: std::fmt::Display + std::any::Any + Send + Sync> HighlightValue for T {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

/// A formatted highlight, as `highlightFieldsAsObjects` returns them.
pub type HighlightObject = Box<dyn HighlightValue>;

/// `PassageFormatter`: the snippet from the top passages. `content` is the
/// UTF-16 text the offsets index.
pub trait PassageFormatter {
    /// `format(passages, content)` as text.
    fn format(&self, passages: &[Passage], content: &[u16]) -> String;

    /// `format(passages, content)` as the object it is: [`Self::format`]'s
    /// text unless the formatter makes something else, whose `Display` is
    /// then what `highlightFields` reports.
    fn format_object(&self, passages: &[Passage], content: &[u16]) -> HighlightObject {
        Box::new(self.format(passages, content))
    }
}

/// `DefaultPassageFormatter`.
#[derive(Debug, Clone)]
pub struct DefaultPassageFormatter {
    pre_tag: String,
    post_tag: String,
    ellipsis: String,
    escape: bool,
}

impl Default for DefaultPassageFormatter {
    fn default() -> Self {
        Self::new("<b>", "</b>", "... ", false)
    }
}

impl DefaultPassageFormatter {
    /// `new DefaultPassageFormatter(preTag, postTag, ellipsis, escape)`.
    pub fn new(pre_tag: &str, post_tag: &str, ellipsis: &str, escape: bool) -> Self {
        DefaultPassageFormatter {
            pre_tag: pre_tag.to_string(),
            post_tag: post_tag.to_string(),
            ellipsis: ellipsis.to_string(),
            escape,
        }
    }

    fn append(&self, dest: &mut Vec<u16>, content: &[u16], start: i32, end: i32) {
        let (s, e) = (clamp(start, content.len()), clamp(end, content.len()));
        if !self.escape {
            dest.extend_from_slice(&content[s..e.max(s)]);
            return;
        }
        // the OWASP rules
        for &ch in &content[s..e.max(s)] {
            let rep = match ch {
                0x26 => "&amp;",
                0x3C => "&lt;",
                0x3E => "&gt;",
                0x22 => "&quot;",
                0x27 => "&#x27;",
                0x2F => "&#x2F;",
                _ => {
                    dest.push(ch);
                    continue;
                }
            };
            dest.extend(rep.encode_utf16());
        }
    }
}

fn clamp(v: i32, len: usize) -> usize {
    usize::try_from(v.max(0)).unwrap_or(0).min(len)
}

impl PassageFormatter for DefaultPassageFormatter {
    fn format(&self, passages: &[Passage], content: &[u16]) -> String {
        let mut sb: Vec<u16> = Vec::new();
        let mut pos = 0i32;
        for passage in passages {
            // don't add ellipsis if its the first one, or if its connected
            if !sb.is_empty() && passage.start_offset() != pos {
                sb.extend(self.ellipsis.encode_utf16());
            }
            pos = passage.start_offset();
            let mut i = 0;
            while i < passage.num_matches() {
                let start = passage.match_starts()[i];
                debug_assert!(start >= pos && start < passage.end_offset());
                // append content before this start
                self.append(&mut sb, content, pos, start);
                let mut end = passage.match_ends()[i];
                debug_assert!(end > start);
                // overlapping terms: expand 'end' past all of them
                while i + 1 < passage.num_matches() && passage.match_starts()[i + 1] < end {
                    i += 1;
                    end = end.max(passage.match_ends()[i]);
                }
                // in case the match straddles past the passage
                end = end.min(passage.end_offset());
                sb.extend(self.pre_tag.encode_utf16());
                self.append(&mut sb, content, start, end);
                sb.extend(self.post_tag.encode_utf16());
                pos = end;
                i += 1;
            }
            // a "term" from the analyzer could span a sentence boundary
            self.append(&mut sb, content, pos, pos.max(passage.end_offset()));
            pos = passage.end_offset();
        }
        String::from_utf16_lossy(&sb)
    }
}

// ---------------------------------------------------------------------------
// OffsetsEnum
// ---------------------------------------------------------------------------

/// `OffsetsEnum`: a stream of (term, offsets) in offset order, one position
/// at a time. A fresh enum is positioned before its first match.
pub trait OffsetsEnum {
    /// `nextPosition()`.
    fn next_position(&mut self) -> Result<bool>;
    /// `freq()`.
    fn freq(&self) -> i32;
    /// `getTerm()`; `None` is Java's `null` (a wildcard automaton's enum).
    fn term(&self) -> Option<&[u8]>;
    /// `startOffset()`.
    fn start_offset(&self) -> i32;
    /// `endOffset()`.
    fn end_offset(&self) -> i32;
}

/// `OffsetsEnum.compareTo`: start, end, then term, a missing term last.
pub fn compare_offsets(a: &dyn OffsetsEnum, b: &dyn OffsetsEnum) -> Ordering {
    a.start_offset()
        .cmp(&b.start_offset())
        .then(a.end_offset().cmp(&b.end_offset()))
        .then_with(|| match (a.term(), b.term()) {
            (None, None) => Ordering::Equal,
            (None, Some(_)) => Ordering::Greater,
            (Some(_), None) => Ordering::Less,
            (Some(x), Some(y)) => x.cmp(y),
        })
}

/// `OffsetsEnum.EMPTY`.
#[derive(Debug, Default)]
pub struct EmptyOffsets;

impl OffsetsEnum for EmptyOffsets {
    fn next_position(&mut self) -> Result<bool> {
        Ok(false)
    }
    fn freq(&self) -> i32 {
        0
    }
    fn term(&self) -> Option<&[u8]> {
        panic!("UnsupportedOperationException")
    }
    fn start_offset(&self) -> i32 {
        panic!("UnsupportedOperationException")
    }
    fn end_offset(&self) -> i32 {
        panic!("UnsupportedOperationException")
    }
}

/// `OffsetsEnum.OfPostings`, with the occurrences read up front: one term, its
/// `freq`, and the offsets of each of its positions in the document.
#[derive(Debug)]
pub struct OfPostings {
    term: Vec<u8>,
    freq: i32,
    offsets: Vec<(i32, i32)>,
    next: usize,
}

impl OfPostings {
    /// `new OfPostings(term, freq, postingsEnum)`.
    pub fn new(term: Vec<u8>, freq: i32, offsets: Vec<(i32, i32)>) -> Self {
        OfPostings {
            term,
            freq,
            offsets,
            next: 0,
        }
    }
}

impl OffsetsEnum for OfPostings {
    fn next_position(&mut self) -> Result<bool> {
        if self.next < self.offsets.len() {
            self.next += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn freq(&self) -> i32 {
        self.freq
    }
    fn term(&self) -> Option<&[u8]> {
        Some(&self.term)
    }
    fn start_offset(&self) -> i32 {
        self.offsets[self.next - 1].0
    }
    fn end_offset(&self) -> i32 {
        self.offsets[self.next - 1].1
    }
}

/// An enum over matches that were all collected up front -- the shape of
/// `OfMatchesIteratorWithSubs`' output, `PhraseHelper`'s
/// `SpanCollectedOffsetsEnum` and the token-stream strategy's enum: each
/// entry is (term, start, end) with one shared or per-entry frequency.
#[derive(Debug)]
pub struct OfMatches {
    entries: Vec<MatchEntry>,
    next: usize,
}

/// One match of an [`OfMatches`].
#[derive(Debug, Clone)]
pub struct MatchEntry {
    /// The term (or label) the match is reported under.
    pub term: Vec<u8>,
    /// Start offset.
    pub start: i32,
    /// End offset.
    pub end: i32,
    /// `freq()` while this entry is current.
    pub freq: i32,
}

impl OfMatches {
    /// The entries are enumerated in the order given.
    pub fn new(entries: Vec<MatchEntry>) -> Self {
        OfMatches { entries, next: 0 }
    }
}

impl OffsetsEnum for OfMatches {
    fn next_position(&mut self) -> Result<bool> {
        if self.next < self.entries.len() {
            self.next += 1;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn freq(&self) -> i32 {
        self.entries[self.next - 1].freq
    }
    fn term(&self) -> Option<&[u8]> {
        Some(&self.entries[self.next - 1].term)
    }
    fn start_offset(&self) -> i32 {
        self.entries[self.next - 1].start
    }
    fn end_offset(&self) -> i32 {
        self.entries[self.next - 1].end
    }
}

/// `OffsetsEnum.MultiOffsetsEnum`: a merge of several enums in
/// [`compare_offsets`] order.
pub struct MultiOffsets {
    /// Min-heap by `compare_offsets` (kept sorted ascending by `sift`).
    queue: Vec<Box<dyn OffsetsEnum>>,
    started: bool,
}

impl MultiOffsets {
    /// `new MultiOffsetsEnum(inner)`.
    pub fn new(inner: Vec<Box<dyn OffsetsEnum>>) -> Result<Self> {
        let mut queue = Vec::new();
        for mut oe in inner {
            if oe.next_position()? {
                queue.push(oe);
            }
        }
        let mut me = MultiOffsets {
            queue,
            started: false,
        };
        me.sort();
        Ok(me)
    }

    fn sort(&mut self) {
        // `sort_by` is stable: equal entries keep arrival order
        self.queue.sort_by(|a, b| compare_offsets(&**a, &**b));
    }
}

impl OffsetsEnum for MultiOffsets {
    fn next_position(&mut self) -> Result<bool> {
        if !self.started {
            self.started = true;
            return Ok(!self.queue.is_empty());
        }
        if self.queue.is_empty() {
            return Ok(false);
        }
        let mut top = self.queue.remove(0);
        if top.next_position()? {
            // The rest is still sorted, so one binary search places `top`
            // where pushing it and re-running the stable sort would: after
            // every entry that does not compare greater.
            let at = self
                .queue
                .partition_point(|x| compare_offsets(&**x, &*top) != Ordering::Greater);
            self.queue.insert(at, top);
            return Ok(true);
        }
        Ok(!self.queue.is_empty())
    }
    fn freq(&self) -> i32 {
        self.queue[0].freq()
    }
    fn term(&self) -> Option<&[u8]> {
        self.queue[0].term()
    }
    fn start_offset(&self) -> i32 {
        self.queue[0].start_offset()
    }
    fn end_offset(&self) -> i32 {
        self.queue[0].end_offset()
    }
}

#[cfg(test)]
mod tests;
