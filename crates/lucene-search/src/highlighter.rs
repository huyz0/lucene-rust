//! Fragment assembly / highlighting over task #39's offset primitive (task #56).
//!
//! `crates/lucene-search/src/term_vectors_query.rs`'s `matched_term_offsets`
//! computes character-offset spans for matched terms in one document's
//! field -- exactly what a highlighter needs to know *where* the matches
//! are. This module is the next step: given those spans *plus* the original
//! field text (read from wherever the caller got it -- typically
//! `crates/lucene-codecs/src/stored_fields.rs`'s `StoredFieldsReader`, since
//! offsets alone don't carry the text), slice out short readable snippets
//! ("fragments") with the matches wrapped in a highlight marker, e.g.
//! `<b>term</b>` -- real Lucene's `UnifiedHighlighter`/`PassageFormatter`
//! default marker.
//!
//! ## Scope: a real, honestly-simplified MVP, not `UnifiedHighlighter`
//!
//! Real Lucene's `UnifiedHighlighter` uses a `BreakIterator` (locale-aware
//! sentence/word boundary detection, real NLP-adjacent work) to choose
//! passage boundaries, then scores candidate passages with `PassageScorer`
//! before picking the top N and rendering them with a `PassageFormatter`.
//! **This module ports the scorer and the formatter exactly; only the
//! boundary chooser is a simplification**, documented here as exactly that --
//! not a silent stand-in for `BreakIterator`:
//!
//! - Each match (or cluster of nearby matches) gets a fixed-size character
//!   window: `window_chars` before the earliest match in the cluster and
//!   `window_chars` after the latest, clamped to the text's start/end.
//! - The window's cut edges are snapped outward to the nearest whitespace
//!   boundary (so a fragment doesn't begin or end mid-word) where possible;
//!   if no whitespace is found before hitting the text boundary, it uses
//!   the boundary itself.
//! - **Passage selection is Java's**: every candidate gets a
//!   [`PassageScorer`] score (`k1 = 1.2`, `b = 0.75`, `pivot = 87`, with
//!   Java's exact `weight`/`tf`/`norm`/`score` formulas), the best
//!   `max_fragments` are kept, and only then are the survivors re-sorted into
//!   document order for rendering -- `FieldHighlighter.
//!   highlightOffsetsEnums`'s priority queue followed by its
//!   `passageSortComparator` sort. (This module used to truncate in plain
//!   document order, which silently discarded the best passages whenever they
//!   sat late in the text.)
//! - **Rendering is Java's** `DefaultPassageFormatter`: overlapping matches
//!   coalesce into one `pre`/`post`-wrapped span rather than nesting markers,
//!   the content is optionally HTML-escaped (`FragmentConfig::escape`) while
//!   the markers stay raw, and [`format_fragments`] joins non-contiguous
//!   passages with `FragmentConfig::ellipsis` (default `"... "`).
//!
//! ## Sentence-boundary snapping (opt-in via `FragmentConfig::snap_to_sentence`)
//!
//! `assemble_fragments`'s default behavior above (fixed-size char window,
//! snapped outward to whitespace) is unchanged. Setting
//! [`FragmentConfig::snap_to_sentence`] to `true` switches a fragment's edges
//! from that fixed window to the boundaries of the sentence(s) actually
//! containing its match(es) -- real `UnifiedHighlighter`'s
//! `BreakIterator.getSentenceInstance()`-based passage boundaries.
//!
//! - **The boundaries are UAX #29's**, via [`sentence_boundaries`] -- the same
//!   specification the JDK's `BreakIterator.getSentenceInstance` implements,
//!   not a heuristic. c12 replaced a hand-rolled terminator scan with an
//!   English abbreviation list; that list was a divergence from Java, which
//!   applies no abbreviation suppression at all. See
//!   [`sentence_boundaries`]'s own doc for the one tailoring that *is*
//!   applied (folding a whitespace-only run into the preceding sentence, which
//!   is what the JDK does and pure UAX #29 does not).
//! - [`split_sentence_boundaries`] is `SplittingBreakIterator`: sentence
//!   boundaries within each slice of the text between occurrences of a given
//!   character, for a multi-valued field whose values are joined with a
//!   separator that a passage must never straddle.
//! - A fragment's start snaps to the start of the sentence containing its
//!   cluster's earliest match; its end snaps to the end of the sentence
//!   containing its cluster's latest match (trailing whitespace trimmed).
//!   `window_chars` still governs which nearby matches get merged into one
//!   cluster, but no longer bounds the rendered fragment's size once
//!   sentence-snapped -- a fragment can be shorter *or* longer than the fixed
//!   window it would have used, since it's exactly the sentence(s), not a
//!   char count.
//! - If the text has no sentence terminator at all, the whole text is one
//!   sentence: the fragment still comes out sensible, never empty or
//!   panicking.
//!
//! ## Overlapping-window merging
//!
//! Two matches whose extended windows overlap (or abut) are merged into a
//! single fragment rather than two overlapping/duplicate ones -- this
//! mirrors real Lucene's passage-merging behavior and is the one piece of
//! this module's logic that is easy to get subtly wrong (see this module's
//! tests for the two-nearby-matches-in-one-fragment case, including marker
//! insertion for multiple matches within one merged window).
//!
//! ## Offset units: Java `char`s, i.e. UTF-16 code units
//!
//! Real Lucene's `OffsetAttribute` offsets are indices into the original
//! `String`, so they are **UTF-16 code units** -- Java `char`s -- and every
//! `PassageFormatter` slices the stored text with them via
//! `String.substring`. `crates/lucene-codecs/src/term_vectors.rs` and
//! `postings.rs` decode those `start_offset`/`end_offset` values verbatim off
//! disk, so that is the unit reaching this module from any real Lucene index.
//!
//! This module therefore measures **everything** in UTF-16 code units: a
//! [`TermOffsetSpan`]'s offsets, [`Fragment::start_offset`]/
//! [`Fragment::end_offset`], [`FragmentConfig::window_chars`], and the
//! `contentLength`/`passageLength` [`PassageScorer`] consumes (Java's are the
//! same `String` lengths). [`utf16_offset_to_byte`] converts one to a UTF-8
//! byte offset by walking `char_indices()` and accumulating `len_utf16()`, so
//! slicing `full_text` always lands on a valid UTF-8 boundary -- it cannot
//! panic on a multi-byte character even if the input offsets are wrong or
//! out of range (they are clamped to the text's UTF-16 length first).
//!
//! **This used to be Rust's `char` -- Unicode scalars -- which is not the
//! same unit** (M2 sweep `c29-search-carryovers`, closing c23's F13). The two
//! agree for the whole Basic Multilingual Plane and disagree for every
//! supplementary-plane character: an emoji is one Unicode scalar and two
//! UTF-16 code units, so a document containing one shifted every subsequent
//! offset by one per emoji and cut its snippets in the wrong place. Nothing
//! caught it because every fixture in this repo was ASCII, where UTF-8 bytes,
//! Unicode scalars and UTF-16 code units all coincide;
//! `crates/lucene-search/tests/highlighter_utf16_offsets_fixtures.rs` now
//! pins the unit against real Lucene's own `StandardAnalyzer` offsets over
//! text that separates all three.
//!
//! [`offsets_from_analysis`] used to be the one place a *different* unit
//! entered -- `lucene_analysis::Token`'s offsets were UTF-8 bytes and it
//! converted at the boundary. As of c33 that producer emits Java `char`s too,
//! so there is now exactly **one** offset unit from the analyzer through the
//! index to the rendered fragment, and no conversion anywhere in between.

/// One assembled, highlighted fragment of field text -- real Lucene's
/// `Passage`, after `PassageFormatter` has already rendered it.
#[derive(Debug, Clone, PartialEq)]
pub struct Fragment {
    /// The fragment's text, with each match wrapped in `pre`/`post` markers.
    pub text: String,
    /// The distinct matched terms found within this fragment, in the order
    /// they first appear (a repeated term is listed once).
    pub matched_terms: Vec<String>,
    /// `Passage.getStartOffset()` -- the fragment's start as a UTF-16
    /// code-unit (Java `char`) offset into the original `full_text`, the same
    /// unit [`TermOffsetSpan`] uses and the same one `Passage` reports.
    pub start_offset: usize,
    /// `Passage.getEndOffset()`, exclusive, in the same unit.
    pub end_offset: usize,
    /// `Passage.getScore()` -- [`PassageScorer`]'s BM25-shaped passage score,
    /// which is what decides *which* `max_fragments` fragments survive.
    pub score: f32,
}

/// Port of `org.apache.lucene.search.uhighlight.PassageScorer` -- the
/// BM25-shaped scorer that decides which passages a highlighter keeps when
/// there are more candidates than `maxPassages`.
///
/// Every formula and default below is Java's verbatim:
/// - `k1 = 1.2`, `b = 0.75`, `pivot = 87` (the defaults, chosen from the
///   "average length of an English sentence" note in Java's own doc comment).
/// - `weight(contentLength, totalTermFreq) = (k1 + 1) * log(1 + (numDocs +
///   0.5) / (totalTermFreq + 0.5))` where `numDocs = 1 + contentLength /
///   pivot` -- an idf-shaped term weight that treats the *document* as the
///   corpus and a pivot-sized slice of it as a "document".
/// - `tf(freq, passageLen) = freq / (freq + k1 * ((1 - b) + b * (passageLen /
///   pivot)))` -- the saturating term frequency, length-normalized against
///   `pivot` rather than an average field length.
/// - `norm(passageStart) = 1 + 1 / log(pivot + passageStart)` -- a mild
///   preference for passages earlier in the text.
/// - `score(passage, contentLength) = norm(startOffset) * sum over each
///   *distinct* matched term of tf(itsFreqInThisPassage, passageLength) *
///   weight(contentLength, itsFreqInTheWholeDocument)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PassageScorer {
    /// BM25 `k1`: controls term-frequency saturation.
    pub k1: f32,
    /// BM25 `b`: controls length normalization.
    pub b: f32,
    /// The pivot length used in place of an average field length.
    pub pivot: f32,
}

impl Default for PassageScorer {
    /// `new PassageScorer()`: `k1 = 1.2`, `b = 0.75`, `pivot = 87`.
    fn default() -> Self {
        PassageScorer {
            k1: 1.2,
            b: 0.75,
            pivot: 87.0,
        }
    }
}

impl PassageScorer {
    /// `PassageScorer.weight(int contentLength, int totalTermFreq)`.
    pub fn weight(&self, content_length: usize, total_term_freq: usize) -> f32 {
        let num_docs = 1.0 + content_length as f32 / self.pivot;
        (self.k1 + 1.0)
            * ((1.0 + (num_docs as f64 + 0.5) / (total_term_freq as f64 + 0.5)).ln()) as f32
    }

    /// `PassageScorer.tf(int freq, int passageLen)`.
    pub fn tf(&self, freq: usize, passage_len: usize) -> f32 {
        let norm = self.k1 * ((1.0 - self.b) + self.b * (passage_len as f32 / self.pivot));
        freq as f32 / (freq as f32 + norm)
    }

    /// `PassageScorer.norm(int passageStart)`.
    pub fn norm(&self, passage_start: usize) -> f32 {
        1.0 + 1.0 / ((self.pivot + passage_start as f32) as f64).ln() as f32
    }

    /// `PassageScorer.score(Passage, int contentLength)`: `norm(startOffset)`
    /// times the sum, over each **distinct** matched term in the passage, of
    /// `tf(that term's freq in this passage, passage length) * weight(content
    /// length, that term's freq in the whole document)`.
    ///
    /// `term_freqs` pairs each distinct term with `(freq in this passage,
    /// freq in the whole document)` -- Java reads the latter from
    /// `OffsetsEnum.freq()`; [`assemble_fragments`] derives it by counting the
    /// term's occurrences across *every* span it was handed, which is the
    /// same number whenever those spans come from
    /// [`crate::term_vectors_query::matched_term_offsets`] (it emits every
    /// occurrence of every matched term in the field).
    pub fn score(
        &self,
        term_freqs: &[(usize, usize)],
        passage_start: usize,
        passage_len: usize,
        content_length: usize,
    ) -> f32 {
        // Java accumulates into a `double` and casts once at the end.
        let mut score = 0f64;
        for &(freq_in_passage, freq_in_doc) in term_freqs {
            score += (self.tf(freq_in_passage, passage_len)
                * self.weight(content_length, freq_in_doc)) as f64;
        }
        (score * self.norm(passage_start) as f64) as f32
    }
}

/// Configuration for [`assemble_fragments`].
#[derive(Debug, Clone)]
pub struct FragmentConfig {
    /// Context to keep before the earliest match and after the latest match
    /// in a cluster, before whitespace-snapping and clamping to the text's
    /// bounds.
    ///
    /// Measured in **Java `char`s, i.e. UTF-16 code units** -- the unit every
    /// offset in this module is in, and the one Lucene's own passage lengths
    /// use. See this module's "Offset units" section.
    pub window_chars: usize,
    /// Marker inserted immediately before each matched term's text.
    pub pre: String,
    /// Marker inserted immediately after each matched term's text.
    pub post: String,
    /// Maximum number of fragments to return; later fragments (in
    /// left-to-right document order) beyond this count are dropped.
    pub max_fragments: usize,
    /// When `true`, a fragment's rendered start/end are snapped to the
    /// boundaries of the sentence(s) containing its match(es) instead of the
    /// fixed `window_chars` window -- see this module's doc comment section
    /// on sentence-boundary snapping for the exact (deliberately narrow)
    /// heuristic. Defaults to `false`, preserving this struct's prior
    /// fixed-window-only behavior for existing callers.
    pub snap_to_sentence: bool,
    /// `DefaultPassageFormatter`'s `ellipsis`: what [`format_fragments`] puts
    /// between two fragments that aren't contiguous in the original text.
    /// Java's default is `"... "` (with the trailing space).
    pub ellipsis: String,
    /// `DefaultPassageFormatter`'s `escape`: when `true`, every character of
    /// the original text that ends up in a fragment is HTML-escaped
    /// (`&` `<` `>` `"` `'` `/` -> `&amp;` `&lt;` `&gt;` `&quot;` `&#x27;`
    /// `&#x2F;`), while the `pre`/`post` markers themselves are emitted raw --
    /// exactly `DefaultPassageFormatter.append`'s split. Defaults to `false`,
    /// Java's default.
    pub escape: bool,
    /// The [`PassageScorer`] used to rank candidate fragments before
    /// `max_fragments` truncates them. Real Lucene *always* keeps the
    /// **highest-scoring** `maxPassages` passages (`FieldHighlighter
    /// .maybeAddPassage`'s priority queue) and only then re-sorts them into
    /// document order for rendering; this port did the same truncation in
    /// plain document order until this field existed.
    pub scorer: PassageScorer,
}

impl Default for FragmentConfig {
    /// `window_chars: 40`, `pre: "<b>"`, `post: "</b>"`, `ellipsis: "... "`,
    /// `escape: false` (real Lucene's `DefaultPassageFormatter` defaults),
    /// `max_fragments: 5`, `snap_to_sentence: false`, and the default
    /// [`PassageScorer`].
    fn default() -> Self {
        FragmentConfig {
            window_chars: 40,
            pre: "<b>".to_string(),
            post: "</b>".to_string(),
            max_fragments: 5,
            snap_to_sentence: false,
            ellipsis: "... ".to_string(),
            escape: false,
            scorer: PassageScorer::default(),
        }
    }
}

/// Port of `DefaultPassageFormatter.append`'s escaping branch: the exact six
/// characters Java escapes, and their exact replacements.
fn append_escaped(dest: &mut String, text: &str, escape: bool) {
    if !escape {
        dest.push_str(text);
        return;
    }
    for ch in text.chars() {
        match ch {
            '&' => dest.push_str("&amp;"),
            '<' => dest.push_str("&lt;"),
            '>' => dest.push_str("&gt;"),
            '"' => dest.push_str("&quot;"),
            '\'' => dest.push_str("&#x27;"),
            '/' => dest.push_str("&#x2F;"),
            other => dest.push(other),
        }
    }
}

/// Port of `DefaultPassageFormatter.format(Passage[], String)`'s outer loop:
/// concatenates `fragments`' already-rendered text, inserting
/// `config.ellipsis` between two fragments that are **not** contiguous in the
/// original text (`passage.getStartOffset() != pos`). `fragments` must be in
/// document order -- which is what [`assemble_fragments`] returns.
///
/// Real Lucene folds this into the same pass that inserts the markers;
/// splitting it out keeps [`assemble_fragments`]'s per-fragment result useful
/// on its own (an API this port already exposes over FFI) while still making
/// the single-string `UnifiedHighlighter.highlight` output reachable.
pub fn format_fragments(fragments: &[Fragment], config: &FragmentConfig) -> String {
    let mut out = String::new();
    let mut pos: Option<usize> = None;
    for fragment in fragments {
        if !out.is_empty() && pos != Some(fragment.start_offset) {
            out.push_str(&config.ellipsis);
        }
        out.push_str(&fragment.text);
        pos = Some(fragment.end_offset);
    }
    out
}

/// `BreakIterator.getSentenceInstance(Locale.ROOT)` -- every sentence
/// boundary in `text`, as byte offsets, always including `0` and
/// `text.len()`, ascending and deduplicated.
///
/// **This is UAX #29 sentence segmentation, not a heuristic.** Java's
/// `UnifiedHighlighter` breaks passages with `BreakIterator.getSentenceInstance`,
/// whose JDK implementation is a rule-based iterator over
/// [UAX #29](https://www.unicode.org/reports/tr29/)'s sentence-boundary rules
/// (SB1-SB12) with CLDR data. The `unicode-segmentation` crate -- already a
/// dependency of `lucene-analysis`, where it backs the UAX #29 *word*
/// tokenizer -- implements the same specification's sentence rules, so this is
/// the same algorithm rather than an approximation of it.
///
/// What it is *not*: locale-tailored. `getSentenceInstance(Locale)` can carry
/// per-locale rule tailorings; this is the untailored root behaviour, which is
/// what `UnifiedHighlighter`'s own default (`Locale.ROOT`) asks for.
///
/// ### What replacing the previous heuristic changed
///
/// Before c12 this was a hand-rolled scan for `.`/`!`/`?` followed by
/// whitespace and an uppercase letter, with a hardcoded English abbreviation
/// list ("Mr", "Dr", "St", ...) suppressing the break. That list was a
/// **divergence from Java, not a refinement of it**: run against the JDK,
/// `BreakIterator.getSentenceInstance(Locale.ROOT)` splits
/// `"Mr. Smith went home."` into `["Mr. ", "Smith went home."]` -- it applies
/// no abbreviation suppression at all, in either `Locale.ROOT` or
/// `Locale.ENGLISH`. The highlighter's passages are therefore now cut where
/// Lucene cuts them, including at the places the old list was trying to
/// protect. `sentence_boundaries_match_jdk_break_iterator` pins the JDK's
/// actual output for six texts.
pub fn sentence_boundaries(text: &str) -> Vec<usize> {
    use unicode_segmentation::UnicodeSegmentation;
    // `USentenceBounds::size_hint` computes `lower - 1` unguarded, so
    // collecting the bounds of an empty string underflows in a debug build
    // (unicode-segmentation 1.13.3, `sentence.rs:380`). An empty text has one
    // boundary anyway.
    if text.is_empty() {
        return vec![0];
    }
    let mut out: Vec<usize> = std::iter::once(0)
        .chain(text.split_sentence_bound_indices().map(|(i, _)| i))
        .collect();
    out.push(text.len());
    out.dedup();

    // One tailoring, in the direction of the spec of record. UAX #29's SB4
    // (`ParaSep ÷`) ends a sentence at *every* paragraph separator, so
    // `"Two.\n\nThree."` segments as `["Two.\n", "\n", "Three."]` -- a
    // "sentence" consisting of one newline. The JDK's
    // `BreakIterator.getSentenceInstance` folds that run into the preceding
    // sentence (`["Two.\n\n", "Three."]`), and the JDK is what
    // `UnifiedHighlighter` passages are cut with. Dropping any boundary whose
    // slice is entirely whitespace reproduces that exactly, and cannot affect
    // a boundary that starts real text.
    //
    // One backward pass, not repeated `Vec::remove`: a document with many
    // blank lines would otherwise be quadratic in the number of boundaries.
    // Walking from the end means each boundary is judged against the *kept*
    // boundary that follows it, which is the same question a forward
    // remove-in-place asks and the reason it had to shift the tail.
    let last = out.len() - 1;
    let mut next_kept = out[last];
    let mut keep = vec![true; out.len()];
    for i in (1..last).rev() {
        if text[out[i]..next_kept].trim().is_empty() {
            keep[i] = false;
        } else {
            next_kept = out[i];
        }
    }
    let mut kept = keep.iter();
    out.retain(|_| *kept.next().unwrap_or(&true));
    out
}

/// `SplittingBreakIterator(baseIter, sliceChar)`: virtually slices `text` on
/// every occurrence of `slice_char` and runs [`sentence_boundaries`] on each
/// slice, so the enclosed iterator never "sees" the splitting character.
///
/// Every `slice_char` position is itself a boundary, and so is the position
/// after it -- Java's "if the slice is 0-length ... that character is reported
/// as a boundary", which is what makes an adjacent pair, or one at either end
/// of the text, still produce boundaries.
///
/// `UnifiedHighlighter` uses this for a multi-valued field, whose values it
/// joins with `MULTIVAL_SEP_CHAR` (`'\0'`) so a passage can never straddle two
/// values of the same field.
pub fn split_sentence_boundaries(text: &str, slice_char: char) -> Vec<usize> {
    let mut out: Vec<usize> = Vec::new();
    let mut slice_start = 0usize;
    let push_slice = |out: &mut Vec<usize>, slice: &str, base: usize| {
        for b in sentence_boundaries(slice) {
            out.push(base + b);
        }
    };
    for (idx, c) in text.char_indices() {
        if c != slice_char {
            continue;
        }
        push_slice(&mut out, &text[slice_start..idx], slice_start);
        out.push(idx);
        out.push(idx + c.len_utf8());
        slice_start = idx + c.len_utf8();
    }
    push_slice(&mut out, &text[slice_start..], slice_start);
    out.push(text.len());
    out.sort_unstable();
    out.dedup();
    out
}

/// Byte offsets (into `text`) where each recognized sentence *begins* -- i.e.
/// [`sentence_boundaries`] without the trailing `text.len()` terminator, which
/// starts no sentence.
fn sentence_start_offsets(text: &str) -> Vec<usize> {
    let mut starts = sentence_boundaries(text);
    if starts.len() > 1 {
        starts.pop();
    }
    starts
}

/// Snaps `byte_offset` back to the start of the sentence containing it: the
/// largest recognized sentence-start `<= byte_offset`, or `0` if none (the
/// very first sentence always starts at `0`, so this never fails to find one
/// in a non-empty `sentence_starts`).
fn snap_start_to_sentence(sentence_starts: &[usize], byte_offset: usize) -> usize {
    sentence_starts
        .iter()
        .rev()
        .find(|&&s| s <= byte_offset)
        .copied()
        .unwrap_or(0)
}

/// Snaps `byte_offset` forward to the end of the sentence containing it: the
/// smallest recognized sentence-start `> byte_offset` (i.e. the next
/// sentence's start), or `text`'s length if `byte_offset`'s sentence is the
/// last one in the text. Trailing whitespace is trimmed off the result so a
/// sentence-snapped fragment doesn't end with dangling blank space.
fn snap_end_to_sentence(sentence_starts: &[usize], byte_offset: usize, text: &str) -> usize {
    let raw_end = sentence_starts
        .iter()
        .find(|&&s| s > byte_offset)
        .copied()
        .unwrap_or(text.len());
    let trimmed = text[..raw_end].trim_end();
    trimmed.len().max(byte_offset)
}

use crate::term_vectors_query::TermOffsetSpan;
use std::collections::HashMap;

/// Converts a UTF-16 code-unit (Java `char`) offset into `text` to a UTF-8
/// byte offset, clamped to `text`'s length -- never panics, never lands on a
/// non-UTF-8-boundary byte index, regardless of how out of range or
/// mis-unitted `utf16_offset` is.
///
/// An offset landing *inside* a surrogate pair (which no offset produced by a
/// Lucene analyzer ever does -- a token boundary is always a code-point
/// boundary) rounds down to that code point's start, which is the nearest
/// valid UTF-8 boundary and keeps the slice well-formed.
fn utf16_offset_to_byte(text: &str, utf16_offset: usize) -> usize {
    let mut units = 0usize;
    for (byte_idx, c) in text.char_indices() {
        let next = units + c.len_utf16();
        if utf16_offset < next {
            return byte_idx;
        }
        units = next;
    }
    text.len()
}

/// `text`'s length in UTF-16 code units -- what `String.length()` returns for
/// the same text in Java, and the unit every offset in this module is in.
fn utf16_len(text: &str) -> usize {
    text.chars().map(char::len_utf16).sum()
}

/// Snaps `byte_offset` outward (leftward for a window start, rightward for
/// a window end) to the nearest ASCII-or-Unicode whitespace boundary within
/// `text`, so a fragment window doesn't start or end mid-word. Falls back
/// to the original offset (which is always a valid char boundary, since
/// [`utf16_offset_to_byte`] only ever returns char-boundary indices) if no
/// whitespace is found before reaching the text's start/end.
fn snap_start_to_whitespace(text: &str, byte_offset: usize) -> usize {
    let before = &text[..byte_offset];
    match before.rfind(char::is_whitespace) {
        // Snap to just after the whitespace character found.
        Some(ws_byte_idx) => {
            let ws_char_len = before[ws_byte_idx..].chars().next().unwrap().len_utf8();
            ws_byte_idx + ws_char_len
        }
        None => 0,
    }
}

fn snap_end_to_whitespace(text: &str, byte_offset: usize) -> usize {
    let after = &text[byte_offset..];
    match after.find(char::is_whitespace) {
        Some(ws_byte_idx) => byte_offset + ws_byte_idx,
        None => text.len(),
    }
}

/// A cluster of one or more nearby matches sharing one merged window,
/// tracked in byte offsets (already converted from the spans' char
/// offsets) for slicing `full_text`.
struct Cluster {
    window_start: usize,
    window_end: usize,
    // Matches within this cluster, as (start_byte, end_byte, term) --
    // sorted ascending by start_byte, used to insert highlight markers.
    matches: Vec<(usize, usize, String)>,
}

/// Assembles highlighted text fragments from `full_text` and a set of
/// already-computed [`TermOffsetSpan`]s (e.g. from
/// [`crate::term_vectors_query::matched_term_offsets`]).
///
/// `spans` need not be sorted or non-overlapping on input; empty spans
/// (or an empty `full_text`) simply produce an empty `Vec<Fragment>` --
/// not an error, since "no matches" is a wholly ordinary caller state, not
/// a fault. Spans with `start_offset > end_offset` or that are entirely
/// out of `full_text`'s bounds are silently dropped (defensive, since a
/// caller may hand this stale offsets against different text without
/// intending a panic).
pub fn assemble_fragments(
    full_text: &str,
    spans: &[TermOffsetSpan],
    config: &FragmentConfig,
) -> Vec<Fragment> {
    if full_text.is_empty() || spans.is_empty() {
        return Vec::new();
    }

    let total_utf16 = utf16_len(full_text);

    // Convert every valid span to (byte_start, byte_end, utf16_start,
    // utf16_end, term) and its own raw (unmerged) window, sorted by match
    // start so overlap-merging can be a single left-to-right sweep. Window
    // arithmetic below is done entirely in UTF-16 code-unit space (matching
    // `config.window_chars`'s unit, which is Java's) before converting to
    // bytes, so it can never straddle a multi-byte character boundary
    // regardless of how far `window_chars` reaches from a match.
    let mut matches: Vec<(usize, usize, usize, usize, String)> = spans
        .iter()
        .filter(|s| s.start_offset >= 0 && s.end_offset >= s.start_offset)
        .filter(|s| (s.start_offset as usize) <= total_utf16)
        .map(|s| {
            let start_utf16 = s.start_offset as usize;
            let end_utf16 = (s.end_offset as usize).min(total_utf16);
            (
                utf16_offset_to_byte(full_text, start_utf16),
                utf16_offset_to_byte(full_text, end_utf16),
                start_utf16,
                end_utf16,
                s.term.clone(),
            )
        })
        .collect();
    matches.sort_by_key(|m| m.0);

    if matches.is_empty() {
        return Vec::new();
    }

    // Sweep matches left-to-right, merging into clusters whenever a
    // match's raw window overlaps (or abuts) the running cluster's window.
    let mut clusters: Vec<Cluster> = Vec::new();
    for (match_start, match_end, start_utf16, end_utf16, term) in matches {
        let raw_window_start_utf16 = start_utf16.saturating_sub(config.window_chars);
        let raw_window_end_utf16 = (end_utf16 + config.window_chars).min(total_utf16);
        let raw_window_start = utf16_offset_to_byte(full_text, raw_window_start_utf16);
        let raw_window_end = utf16_offset_to_byte(full_text, raw_window_end_utf16);
        let window_start = snap_start_to_whitespace(full_text, raw_window_start);
        let window_end = snap_end_to_whitespace(full_text, raw_window_end);

        match clusters.last_mut() {
            Some(last) if window_start <= last.window_end => {
                // Overlapping (or touching) window: merge into the same
                // fragment, extending its end if this match's window
                // reaches further right.
                last.window_end = last.window_end.max(window_end);
                last.matches.push((match_start, match_end, term));
            }
            _ => clusters.push(Cluster {
                window_start,
                window_end,
                matches: vec![(match_start, match_end, term)],
            }),
        }
    }

    // Sentence-snapping (opt-in): recompute each cluster's window from the
    // sentence(s) actually containing its matches, overriding the
    // fixed-window edges computed above. `window_chars` above still governs
    // merging (which matches share one cluster); this only changes what
    // gets rendered.
    if config.snap_to_sentence {
        let sentence_starts = sentence_start_offsets(full_text);
        for cluster in &mut clusters {
            let earliest_match_start = cluster.matches.iter().map(|m| m.0).min().unwrap();
            let latest_match_end = cluster.matches.iter().map(|m| m.1).max().unwrap();
            let start = snap_start_to_sentence(&sentence_starts, earliest_match_start);
            let end = snap_end_to_sentence(&sentence_starts, latest_match_end, full_text);
            cluster.window_start = start;
            cluster.window_end = end.max(start);
        }

        // Snapping can expand two clusters that didn't overlap under the
        // fixed `window_chars` window into the same (or an overlapping)
        // sentence -- e.g. two matches far apart in one long sentence. Without
        // this second sweep they'd render as separate, overlapping fragments
        // covering nearly the same text. Clusters are already sorted by
        // window_start (they were built via a left-to-right sweep over
        // matches sorted by start offset, and snapping only ever grows a
        // window, never reorders it), so a single left-to-right merge pass
        // is enough, identical in shape to the fixed-window sweep above.
        let mut merged: Vec<Cluster> = Vec::with_capacity(clusters.len());
        for cluster in clusters {
            match merged.last_mut() {
                Some(last) if cluster.window_start <= last.window_end => {
                    last.window_end = last.window_end.max(cluster.window_end);
                    last.matches.extend(cluster.matches);
                }
                _ => merged.push(cluster),
            }
        }
        clusters = merged;
    }

    // `FieldHighlighter.highlightOffsetsEnums`: every candidate passage is
    // scored, the best `maxPassages` are kept (its priority queue), and only
    // then are the survivors sorted back into document order for rendering
    // (`Arrays.sort(passages, passageSortComparator)`, whose default is by
    // start offset). Truncating in document order instead -- which is what
    // this function used to do -- silently drops the *best* passages whenever
    // they happen to sit late in the text.
    //
    // Java reads each term's whole-document frequency off `OffsetsEnum.freq()`;
    // here it is the term's occurrence count across every span the caller
    // handed in, which is the same number for spans produced by
    // `term_vectors_query::matched_term_offsets`.
    let mut doc_term_freqs: HashMap<&str, usize> = HashMap::new();
    for span in spans {
        *doc_term_freqs.entry(span.term.as_str()).or_insert(0) += 1;
    }

    let mut scored: Vec<(f32, Cluster)> = clusters
        .into_iter()
        .map(|cluster| {
            let start_utf16 = byte_offset_to_utf16(full_text, cluster.window_start);
            let end_utf16 = byte_offset_to_utf16(full_text, cluster.window_end);
            let mut passage_freqs: HashMap<&str, usize> = HashMap::new();
            for (_, _, term) in &cluster.matches {
                *passage_freqs.entry(term.as_str()).or_insert(0) += 1;
            }
            let term_freqs: Vec<(usize, usize)> = passage_freqs
                .iter()
                .map(|(term, &in_passage)| {
                    (
                        in_passage,
                        doc_term_freqs.get(term).copied().unwrap_or(in_passage),
                    )
                })
                .collect();
            let score = config.scorer.score(
                &term_freqs,
                start_utf16,
                end_utf16.saturating_sub(start_utf16),
                total_utf16,
            );
            (score, cluster)
        })
        .collect();

    if scored.len() > config.max_fragments {
        // Java's queue evicts the lowest score first, breaking a tie on the
        // *lower* start offset -- so the surviving order is score descending,
        // start offset descending on a tie.
        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| b.1.window_start.cmp(&a.1.window_start))
        });
        scored.truncate(config.max_fragments);
    }
    // Back into document order for rendering.
    scored.sort_by_key(|(_, cluster)| cluster.window_start);

    scored
        .into_iter()
        .map(|(score, cluster)| render_cluster(full_text, &cluster, config, score))
        .collect()
}

/// Inverse of [`utf16_offset_to_byte`]: how many UTF-16 code units precede
/// `byte_offset` in `text`. Used to report a fragment's boundaries (and to
/// feed [`PassageScorer`]) in the same unit [`TermOffsetSpan`] uses.
fn byte_offset_to_utf16(text: &str, byte_offset: usize) -> usize {
    utf16_len(&text[..byte_offset.min(text.len())])
}

/// Renders one cluster into a [`Fragment`], a port of
/// `DefaultPassageFormatter.format`'s per-passage body: emit the text between
/// matches (HTML-escaped when `config.escape`), wrap each match -- coalescing
/// any run of *overlapping* matches into a single wrapped span, exactly as
/// Java does -- in `pre`/`post`, and finish with the passage's trailing text.
fn render_cluster(
    full_text: &str,
    cluster: &Cluster,
    config: &FragmentConfig,
    score: f32,
) -> Fragment {
    // `matched_terms` in left-to-right first-occurrence order.
    let mut matched_terms: Vec<String> = Vec::new();
    for (_, _, term) in &cluster.matches {
        if !matched_terms.contains(term) {
            matched_terms.push(term.clone());
        }
    }

    // `DefaultPassageFormatter.format`'s inner loop, verbatim: walk the
    // matches left to right, emitting the (optionally escaped) text between
    // them, and coalescing any run of matches that *overlap* into one
    // pre/post-wrapped span (`while (i + 1 < numMatches && matchStarts[i + 1]
    // < end) end = max(end, matchEnds[++i])`). Rendering back-to-front
    // instead -- which is what this function used to do -- produces nested,
    // interleaved markers for overlapping matches (e.g. a term and a longer
    // term starting at the same offset), and cannot escape the text at all.
    let mut text = String::new();
    let mut pos = cluster.window_start;
    let mut i = 0usize;
    while i < cluster.matches.len() {
        let (match_start, match_end, _) = cluster.matches[i];
        let start = match_start.clamp(cluster.window_start, cluster.window_end);
        if start < pos {
            // Fully swallowed by the previous (coalesced) match.
            i += 1;
            continue;
        }
        append_escaped(&mut text, &full_text[pos..start], config.escape);

        let mut end = match_end;
        while i + 1 < cluster.matches.len() && cluster.matches[i + 1].0 < end {
            i += 1;
            end = end.max(cluster.matches[i].1);
        }
        let end = end.clamp(start, cluster.window_end);

        text.push_str(&config.pre);
        append_escaped(&mut text, &full_text[start..end], config.escape);
        text.push_str(&config.post);
        pos = end;
        i += 1;
    }
    append_escaped(
        &mut text,
        &full_text[pos.min(cluster.window_end)..cluster.window_end],
        config.escape,
    );

    Fragment {
        text,
        matched_terms,
        start_offset: byte_offset_to_utf16(full_text, cluster.window_start),
        end_offset: byte_offset_to_utf16(full_text, cluster.window_end),
        score,
    }
}

// ---------------------------------------------------------------------------
// `FieldOffsetStrategy`: where the offsets come from
// ---------------------------------------------------------------------------

/// `UnifiedHighlighter.OffsetSource` -- which of a field's stored structures
/// can answer "where in the original text does this term occur in this
/// document".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OffsetSource {
    /// The postings list carries character offsets
    /// (`IndexOptions.DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`).
    Postings,
    /// The field stores term vectors with offsets.
    TermVectors,
    /// Neither: the text has to be re-analyzed at highlight time.
    Analysis,
    /// Postings carry offsets *and* term vectors exist -- needed when the
    /// query has a multi-term component whose matching terms are only
    /// discoverable by scanning this document's own vocabulary.
    PostingsWithTermVectors,
    /// Nothing to highlight, so no offsets need reading at all
    /// (`NoOpOffsetStrategy`).
    NoneNeeded,
}

/// `UnifiedHighlighter.getOffsetSource(field)`: pick a source from the field's
/// own `FieldInfo`, in Java's exact order.
///
/// 1. offsets in the postings → [`OffsetSource::PostingsWithTermVectors`] if
///    the field also has term vectors, else [`OffsetSource::Postings`];
/// 2. term vectors → [`OffsetSource::TermVectors`] (Java notes it cannot check
///    here whether the vectors actually carry offsets; if they do not, the
///    failure surfaces later);
/// 3. otherwise → [`OffsetSource::Analysis`].
///
/// A field with no `FieldInfo` at all is Java's `fieldInfo == null`, also
/// [`OffsetSource::Analysis`] -- pass `None` for `index_options`.
pub fn offset_source_for_field(
    index_options: Option<lucene_codecs::field_infos::IndexOptions>,
    has_term_vectors: bool,
) -> OffsetSource {
    use lucene_codecs::field_infos::IndexOptions;
    match index_options {
        Some(IndexOptions::DocsAndFreqsAndPositionsAndOffsets) => {
            if has_term_vectors {
                OffsetSource::PostingsWithTermVectors
            } else {
                OffsetSource::Postings
            }
        }
        Some(_) if has_term_vectors => OffsetSource::TermVectors,
        // `None` is Java's `fieldInfo == null`, which falls straight through
        // to `ANALYSIS` -- it reads `hasTermVectors()` only *inside* the
        // `fieldInfo != null` branch, so a field with no `FieldInfo` cannot
        // report term vectors in the first place.
        _ => OffsetSource::Analysis,
    }
}

/// `UnifiedHighlighter.getOptimizedOffsetSource(components)`: adjust the
/// field-derived source for what the *query* actually needs.
///
/// - Nothing to highlight at all (no terms and no multi-term/rewriting part)
///   → [`OffsetSource::NoneNeeded`].
/// - [`OffsetSource::Postings`] with a multi-term part → downgraded to
///   [`OffsetSource::Analysis`], because matching a wildcard/regexp against
///   this one document's terms would otherwise mean scanning the whole field's
///   term dictionary.
/// - [`OffsetSource::PostingsWithTermVectors`] *without* a multi-term part →
///   upgraded to [`OffsetSource::Postings`]: the term vectors are not needed.
///
/// `has_multi_term_part` is Java's `mtqOrRewrite` (automata present, a phrase
/// helper that will rewrite, or an unrecognized query part); `has_terms` is
/// `components.terms() != null && terms.length > 0`.
pub fn optimize_offset_source(
    source: OffsetSource,
    has_multi_term_part: bool,
    has_terms: bool,
) -> OffsetSource {
    if !has_multi_term_part && !has_terms {
        return OffsetSource::NoneNeeded;
    }
    match source {
        OffsetSource::Postings if has_multi_term_part => OffsetSource::Analysis,
        OffsetSource::PostingsWithTermVectors if !has_multi_term_part => OffsetSource::Postings,
        other => other,
    }
}

/// `PostingsOffsetStrategy.getOffsetsEnum` -- offsets straight out of the
/// postings list, for a field indexed with
/// `DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS`.
///
/// One [`TermOffsetSpan`] per occurrence of each of `terms` in `doc_id`,
/// ordered the way `OffsetsEnum.compareTo` orders them (start offset, then end
/// offset, then the term) -- the same contract
/// [`crate::term_vectors_query::matched_term_offsets`] establishes for the
/// term-vector source, so the two are interchangeable inputs to
/// [`assemble_fragments`].
///
/// A term that is absent from the field, or present but not in this document,
/// contributes nothing. A field that does not index offsets yields spans whose
/// offsets are `-1` (`PostingsEnum.startOffset`'s own no-offsets contract);
/// those are dropped here rather than passed on, because a `-1` offset is not
/// a place in the text -- Java avoids the situation by never selecting this
/// strategy for such a field, which is exactly what
/// [`offset_source_for_field`] does.
///
/// # Cost
///
/// One call to
/// [`FieldTerms::occurrences_for_doc`][lucene_codecs::blocktree::FieldTerms::occurrences_for_doc]
/// per term, which is Java's `postingsEnum.advance(doc)` followed by walking
/// that one document's positions: `.doc`'s skip records down to the one
/// 256-document block holding `doc_id`, then only the `.pos`/`.pay` blocks
/// holding its own occurrences. Nothing else in the term is read.
///
/// It used to decode **every** document's positions and offsets and build a
/// `Vec<Position>` per document in the term, because
/// [`positions`][lucene_codecs::blocktree::FieldTerms::positions] was the only
/// offset-carrying accessor in this port -- so highlighting one document of a
/// high-`docFreq` term cost a full postings sweep (c12 §3.4). c15 closed that
/// with an offsets-carrying wanted-documents reader, which still decoded the
/// term's whole doc list to locate `doc_id`; c20 wired up `.doc`'s
/// `.pos`/`.pay` skip pointers so it no longer does. Measured in
/// `docs/sweep/m2/c15-postings-api.md` and
/// `docs/sweep/m2/c20-postings-skip.md`.
pub fn offsets_from_postings(
    fields: &lucene_codecs::blocktree::BlockTreeFields,
    doc_in: Option<&lucene_codecs::postings::DocInput<'_>>,
    pos_in: &lucene_codecs::postings::PosInput<'_>,
    pay_in: Option<&lucene_codecs::postings::PayInput<'_>>,
    field: &str,
    terms: &[&str],
    doc_id: i32,
) -> crate::Result<Vec<TermOffsetSpan>> {
    let mut spans = Vec::new();
    let Some(field_terms) = fields.field(field) else {
        return Ok(spans);
    };
    for term in terms {
        let Some(occurrences) =
            field_terms.occurrences_for_doc(term.as_bytes(), doc_in, pos_in, pay_in, doc_id)?
        else {
            continue;
        };
        for occurrence in &occurrences {
            if occurrence.start_offset < 0 || occurrence.end_offset < 0 {
                continue;
            }
            spans.push(TermOffsetSpan {
                term: (*term).to_string(),
                start_offset: occurrence.start_offset,
                end_offset: occurrence.end_offset,
            });
        }
    }
    sort_offsets_enum_order(&mut spans);
    Ok(spans)
}

/// `TokenStreamOffsetStrategy.getOffsetsEnum` -- re-analyze the document's own
/// stored text and keep the tokens whose term is one of `terms`.
///
/// This is Java's `ANALYSIS` source, and the one that needs no index support
/// at all: `AnalysisOffsetStrategy` runs the *index* analyzer over the content
/// so the tokens (and therefore the offsets) are the ones that were indexed.
/// `TokenStreamOffsetStrategy` is the cheap variant Java picks when the query
/// is pure term filtering with no position sensitivity -- it keeps a token
/// when its text is in the term set, which is precisely this.
///
/// **No unit conversion happens here.** `lucene_analysis::Token`'s offsets are
/// UTF-16 code units -- Java `char` indices into `content`, the same unit
/// [`TermOffsetSpan`] uses, the same unit real Lucene writes into a
/// `.pos`/`.tvd` and reports from an `OffsetAttribute`. c29 converted from
/// UTF-8 bytes here because the analyzer emitted bytes; c33 fixed the producer,
/// and the conversion came out with it -- a compensating conversion left in
/// place after its cause is fixed is exactly as wrong as the original defect.
pub fn offsets_from_analysis(
    analyzer: &lucene_analysis::Analyzer,
    content: &str,
    terms: &[&str],
) -> Vec<TermOffsetSpan> {
    let mut spans: Vec<TermOffsetSpan> = analyzer
        .analyze(content)
        .into_iter()
        .filter(|t| terms.contains(&t.term.as_str()))
        .map(|t| TermOffsetSpan {
            term: t.term,
            start_offset: t.start_offset,
            end_offset: t.end_offset,
        })
        .collect();
    sort_offsets_enum_order(&mut spans);
    spans
}

/// `OffsetSpanCollector.collectLeaf` at every position of one matched span:
/// the offsets of the occurrence each slot settled on, inserted into that
/// *term's* enum (Java keys the collector by term bytes, so two slots holding
/// one term share one enum and one dedup set).
fn collect_span<'t>(
    terms: &[&'t str],
    occurrences: &[Vec<lucene_codecs::postings::Position>],
    chosen: &[i32],
    per_term: &mut Vec<(&'t str, Vec<(i32, i32)>)>,
) {
    for (slot, &position) in chosen.iter().enumerate() {
        let slot_occurrences = &occurrences[slot];
        let Ok(at) = slot_occurrences.binary_search_by_key(&position, |o| o.position) else {
            continue;
        };
        let occurrence = &slot_occurrences[at];
        // An occurrence whose offsets are `-1` (positions indexed, offsets
        // not) is dropped rather than passed on.
        if occurrence.start_offset < 0 || occurrence.end_offset < occurrence.start_offset {
            continue;
        }
        let pair = (occurrence.start_offset, occurrence.end_offset);
        let entry = match per_term.iter_mut().find(|(t, _)| *t == terms[slot]) {
            Some(entry) => entry,
            None => {
                per_term.push((terms[slot], Vec::new()));
                per_term.last_mut().expect("just pushed")
            }
        };
        // `SpanCollectedOffsetsEnum.add`: sorted insert, dropping an offset
        // pair this term already has.
        if let Err(at) = entry.1.binary_search(&pair) {
            entry.1.insert(at, pair);
        }
    }
}

/// `PhraseHelper` -- the position-sensitive half of `FieldOffsetStrategy`:
/// **only the offsets that take part in an actual phrase match**, not every
/// occurrence of every phrase term.
///
/// This is the difference between highlighting `"quick brown"` as
/// `the <b>quick</b> <b>brown</b> fox, a quick red fox` and highlighting it as
/// `the <b>quick</b> <b>brown</b> fox, a <b>quick</b> red fox`. Java gets it
/// by running the query's `SpanQuery`s over the document and collecting each
/// matched span's *leaf* postings through an `OffsetSpanCollector`
/// (`PhraseHelper.createOffsetsEnumsForSpans`); [`offsets_from_postings`] and
/// [`crate::term_vectors_query::matched_term_offsets`] are the
/// position-*insensitive* sources it exists to correct.
///
/// - `terms` are the phrase's terms **in phrase order**, one per slot, and
///   `occurrences[i]` is that slot's term's occurrences in this one document,
///   ascending by position -- exactly what
///   [`lucene_codecs::blocktree::FieldTerms::occurrences_for_doc`] returns.
///   A repeated term (`"the the"`) is two slots sharing one occurrence list,
///   and Java likewise keys its collector by term bytes, so the two slots
///   contribute to one output enum.
/// - `slop` is the `PhraseQuery`'s, with `0` the exact-adjacency case.
///
/// Offsets are collected per term, kept ascending and **deduplicated** --
/// `SpanCollectedOffsetsEnum.add`'s insertion-sort-with-early-return, which is
/// what stops a position shared by two overlapping matches from being
/// highlighted twice -- and the result is in `OffsetsEnum.compareTo` order,
/// the same contract the two position-insensitive sources have, so it drops
/// straight into [`assemble_fragments`].
///
/// An occurrence whose offsets are `-1` (a field that indexes positions but
/// not offsets) is dropped rather than passed on, same as
/// [`offsets_from_postings`].
///
/// # Which matcher this is, and why it is not the scorer's
///
/// `PhraseHelper` does **not** run the query's own matcher. It runs the
/// `SpanQuery` `WeightedSpanTermExtractor.extract` rewrites the phrase into:
///
/// ```java
/// // sum position increments beyond 1
/// int positionGaps = 0;
/// int[] positions = phraseQuery.getPositions();
/// if (positions.length >= 2) {
///   positionGaps =
///       Math.max(0, positions[positions.length - 1] - positions[0] - positions.length + 1);
/// }
/// // if original slop is 0 then require inOrder
/// boolean inorder = (phraseQuery.getSlop() == 0);
/// SpanNearQuery sp = new SpanNearQuery(clauses, phraseQuery.getSlop() + positionGaps, inorder);
/// ```
///
/// so:
///
/// - **`slop == 0`** is `NearSpansOrdered`: each slot advanced to the smallest
///   position at or after the previous slot's end, with
///   `matchWidth = sum(start_i - end_{i-1})`. That is the in-order greedy walk
///   below, unchanged -- and it is why an exact phrase highlights exactly what
///   it matches.
/// - **`slop > 0`** is `NearSpansUnordered`, whose budget is a *different
///   quantity* from `SloppyPhraseMatcher`'s ([`crate::near_spans`] has the
///   comparison). This port enumerated in order here at every slop, so a
///   reordered occurrence was scored and never highlighted.
///
/// `positionGaps` is always `0` for this function: [`crate::query::PhraseQuery`]
/// holds `terms` with no per-term positions, i.e. Java's `positions[i] == i`.
///
/// The consequences of using the *span* matcher rather than the scorer's are
/// real and are Lucene's, not this port's: a reordered pair is highlighted at
/// slop 1 where the scorer needs slop 2, and `"alpha alpha"~2` is highlighted
/// in a document with a single `alpha` -- `SpanNearQuery` has no `rptGroups`,
/// so two slots holding one term may settle on one position. Both are pinned
/// against real Lucene's own `PhraseHelper` by the `highlight.*` entries in
/// `fixtures/data/blocktree_index/manifest.properties`.
///
/// `PhraseHelper`'s other half -- walking a whole `Query`
/// tree to *discover* which sub-queries are position-sensitive
/// (`WeightedSpanTermExtractor`) -- is the caller's job here: this function is
/// told the phrase.
pub fn phrase_match_offsets(
    terms: &[&str],
    occurrences: &[Vec<lucene_codecs::postings::Position>],
    slop: u32,
) -> Vec<TermOffsetSpan> {
    if terms.is_empty() || terms.len() != occurrences.len() {
        return Vec::new();
    }
    // One position list per slot, ascending -- the alignment walk's input.
    let positions: Vec<Vec<i32>> = occurrences
        .iter()
        .map(|slot| slot.iter().map(|o| o.position).collect())
        .collect();
    if positions.iter().any(|p| p.is_empty()) {
        return Vec::new();
    }

    // Java keys its collector by term bytes, so two slots holding the same
    // term share one output enum (and one dedup set).
    let mut per_term: Vec<(&str, Vec<(i32, i32)>)> = Vec::new();

    let slop = slop as i64;

    if slop == 0 {
        // `NearSpansOrdered`: `stretchToOrder` advances each later slot to the
        // smallest position at or after the previous slot's end, once per
        // position of the first slot, and charges
        // `matchWidth += spans.startPosition() - prevSpans.endPosition()`.
        let (first, rest) = positions.split_first().expect("non-empty, checked above");
        // Scratch reused across candidates: the position chosen for each slot.
        let mut chosen: Vec<i32> = vec![0; positions.len()];
        for &p0 in first.iter() {
            chosen[0] = p0;
            let mut prev = p0;
            let mut total_moves: i64 = 0;
            let mut matched = true;
            for (slot, slot_positions) in rest.iter().enumerate() {
                // Smallest position strictly greater than `prev`; the lists are
                // ascending, so `partition_point` finds it. (A term span ends
                // one past its position, so "at or after the previous end" is
                // "strictly after the previous position".)
                let idx = slot_positions.partition_point(|&x| x <= prev);
                let Some(&pos) = slot_positions.get(idx) else {
                    matched = false;
                    break;
                };
                total_moves += i64::from(pos - prev - 1);
                if total_moves > slop {
                    matched = false;
                    break;
                }
                prev = pos;
                chosen[slot + 1] = pos;
            }
            if matched {
                collect_span(terms, occurrences, &chosen, &mut per_term);
            }
        }
    } else {
        // `NearSpansUnordered`. Every slot's occurrences are one position
        // wide, so a slot's spans are `[p, p + 1)`.
        let spans: Vec<Vec<(i32, i32)>> = positions
            .iter()
            .map(|slot| slot.iter().map(|&p| (p, p.saturating_add(1))).collect())
            .collect();
        let slices: Vec<&[(i32, i32)]> = spans.iter().map(Vec::as_slice).collect();
        let mut chosen: Vec<i32> = vec![0; positions.len()];
        crate::near_spans::for_each_unordered_match(&slices, slop, |current| {
            for (slot, &(start, _)) in current.iter().enumerate() {
                chosen[slot] = start;
            }
            collect_span(terms, occurrences, &chosen, &mut per_term);
        });
    }

    let mut spans: Vec<TermOffsetSpan> = per_term
        .into_iter()
        .flat_map(|(term, pairs)| {
            pairs.into_iter().map(move |(start, end)| TermOffsetSpan {
                term: term.to_string(),
                start_offset: start,
                end_offset: end,
            })
        })
        .collect();
    sort_offsets_enum_order(&mut spans);
    spans
}

/// [`phrase_match_offsets`] over a real segment's postings: reads each phrase
/// term's occurrences in `doc_id` and keeps only those inside a phrase match.
///
/// The postings-source sibling of [`offsets_from_postings`], and a drop-in
/// replacement for it whenever the query is a phrase -- which is exactly the
/// choice `FieldOffsetStrategy` makes by consulting its `PhraseHelper`.
///
/// Costs one
/// [`occurrences_for_doc`][lucene_codecs::blocktree::FieldTerms::occurrences_for_doc]
/// per distinct phrase term (c15's/c20's skip-driven single-document read),
/// the same per-term cost [`offsets_from_postings`] pays -- the phrase
/// filtering itself is one merge pass over the decoded positions and reads
/// nothing further.
#[allow(clippy::too_many_arguments)] // The same eight inputs `offsets_from_postings` takes, plus the slop.
pub fn offsets_from_phrase(
    fields: &lucene_codecs::blocktree::BlockTreeFields,
    doc_in: Option<&lucene_codecs::postings::DocInput<'_>>,
    pos_in: &lucene_codecs::postings::PosInput<'_>,
    pay_in: Option<&lucene_codecs::postings::PayInput<'_>>,
    field: &str,
    phrase: &[&str],
    slop: u32,
    doc_id: i32,
) -> crate::Result<Vec<TermOffsetSpan>> {
    let Some(field_terms) = fields.field(field) else {
        return Ok(Vec::new());
    };
    let mut occurrences = Vec::with_capacity(phrase.len());
    for term in phrase {
        match field_terms.occurrences_for_doc(term.as_bytes(), doc_in, pos_in, pay_in, doc_id)? {
            // A phrase term absent from this document means no phrase match
            // at all, so there is nothing to highlight.
            None => return Ok(Vec::new()),
            Some(found) => occurrences.push(found),
        }
    }
    Ok(phrase_match_offsets(phrase, &occurrences, slop))
}

/// `OffsetsEnum.compareTo`: start offset, then end offset, then the term.
fn sort_offsets_enum_order(spans: &mut [TermOffsetSpan]) {
    spans.sort_by(|a, b| {
        a.start_offset
            .cmp(&b.start_offset)
            .then_with(|| a.end_offset.cmp(&b.end_offset))
            .then_with(|| a.term.cmp(&b.term))
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn span(term: &str, start: i32, end: i32) -> TermOffsetSpan {
        TermOffsetSpan {
            term: term.to_string(),
            start_offset: start,
            end_offset: end,
        }
    }

    fn cfg(window_chars: usize, max_fragments: usize) -> FragmentConfig {
        FragmentConfig {
            window_chars,
            max_fragments,
            ..FragmentConfig::default()
        }
    }

    fn sentence_cfg(window_chars: usize, max_fragments: usize) -> FragmentConfig {
        FragmentConfig {
            snap_to_sentence: true,
            ..cfg(window_chars, max_fragments)
        }
    }

    // --- `PassageScorer` (real Lucene's exact formulas) ---

    #[test]
    fn passage_scorer_defaults_are_javas() {
        let s = PassageScorer::default();
        assert_eq!(s.k1, 1.2);
        assert_eq!(s.b, 0.75);
        assert_eq!(s.pivot, 87.0);
    }

    #[test]
    fn passage_scorer_weight_tf_and_norm_match_the_java_formulas() {
        let s = PassageScorer::default();

        // weight(contentLength, totalTermFreq)
        //   = (k1 + 1) * ln(1 + (1 + contentLength/pivot + 0.5) / (ttf + 0.5))
        let content_length = 174usize; // exactly 2 pivots
        let ttf = 3usize;
        let num_docs = 1.0 + content_length as f32 / 87.0;
        let expected_weight =
            (1.2f32 + 1.0) * ((1.0 + (num_docs as f64 + 0.5) / (ttf as f64 + 0.5)).ln()) as f32;
        assert_eq!(s.weight(content_length, ttf), expected_weight);
        // Sanity: 174/87 == 2, so numDocs == 3.
        assert_eq!(num_docs, 3.0);

        // tf(freq, passageLen) = freq / (freq + k1 * ((1 - b) + b * len/pivot))
        let expected_tf = 2.0f32 / (2.0 + 1.2 * ((1.0 - 0.75) + 0.75 * (87.0 / 87.0)));
        assert_eq!(s.tf(2, 87), expected_tf);
        // At exactly one pivot the length factor collapses to k1 itself.
        assert_eq!(expected_tf, 2.0 / (2.0 + 1.2));

        // tf saturates: doubling freq must not double tf.
        assert!(s.tf(4, 87) < 2.0 * s.tf(2, 87));
        assert!(s.tf(4, 87) > s.tf(2, 87));
        // A longer passage is penalized.
        assert!(s.tf(2, 200) < s.tf(2, 20));

        // norm(passageStart) = 1 + 1/ln(pivot + start): earlier is better.
        assert_eq!(s.norm(0), 1.0 + 1.0 / (87.0f64.ln()) as f32);
        assert!(s.norm(0) > s.norm(500));
    }

    #[test]
    fn passage_scorer_score_is_norm_times_the_per_term_sum() {
        let s = PassageScorer::default();
        let term_freqs = [(2usize, 5usize), (1, 9)];
        let (start, len, content) = (10usize, 60usize, 400usize);
        let expected = ((s.tf(2, len) * s.weight(content, 5)) as f64
            + (s.tf(1, len) * s.weight(content, 9)) as f64)
            * s.norm(start) as f64;
        assert_eq!(s.score(&term_freqs, start, len, content), expected as f32);
    }

    #[test]
    fn max_fragments_keeps_the_highest_scoring_passages_not_the_first_ones() {
        // Three well-separated clusters: the first has one match, the last
        // has three of the same term packed together. Real Lucene keeps the
        // best `maxPassages` by `PassageScorer` and only then re-orders them
        // by offset -- so capping at 1 must keep the *dense* late cluster,
        // not the sparse early one.
        let pad = " pad".repeat(40);
        let mut text = String::from("cat");
        let mut spans = vec![span("cat", 0, 3)];
        text.push_str(&pad);
        text.push_str(&pad);
        let dense_start = text.chars().count();
        text.push_str("cat cat cat");
        for i in 0..3 {
            let s = (dense_start + i * 4) as i32;
            spans.push(span("cat", s, s + 3));
        }

        let all = assemble_fragments(&text, &spans, &cfg(5, 10));
        assert_eq!(all.len(), 2, "the two clusters must not merge");
        assert!(
            all[1].score > all[0].score,
            "the dense cluster must score higher: {} vs {}",
            all[1].score,
            all[0].score
        );

        let capped = assemble_fragments(&text, &spans, &cfg(5, 1));
        assert_eq!(capped.len(), 1);
        assert_eq!(
            capped[0].start_offset, all[1].start_offset,
            "capping must keep the best-scoring passage, not the first one"
        );
    }

    #[test]
    fn fragments_are_returned_in_document_order_after_score_selection() {
        // Even when selection is by score, the surviving fragments come back
        // in document order (`Arrays.sort(passages, passageSortComparator)`).
        let pad = " pad".repeat(40);
        let mut text = String::new();
        let mut spans = Vec::new();
        for _ in 0..4 {
            let start = text.chars().count() as i32;
            text.push_str("cat");
            spans.push(span("cat", start, start + 3));
            text.push_str(&pad);
        }
        let out = assemble_fragments(&text, &spans, &cfg(5, 3));
        assert_eq!(out.len(), 3);
        assert!(out
            .windows(2)
            .all(|w| w[0].start_offset < w[1].start_offset));
    }

    #[test]
    fn fragment_offsets_are_utf16_offsets_into_the_original_text() {
        // 'é' is two UTF-8 bytes but one UTF-16 code unit; the reported
        // offsets must be in the same unit `TermOffsetSpan` uses, not bytes.
        let text = "ééé cat ééé";
        let out = assemble_fragments(text, &[span("cat", 4, 7)], &cfg(100, 5));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].start_offset, 0);
        assert_eq!(out[0].end_offset, utf16_len(text));
    }

    // --- `DefaultPassageFormatter` ---

    #[test]
    fn format_fragments_joins_non_contiguous_passages_with_an_ellipsis() {
        let pad = " pad".repeat(30);
        let mut text = String::from("cat");
        let mut spans = vec![span("cat", 0, 3)];
        text.push_str(&pad);
        let second = text.chars().count() as i32;
        text.push_str("dog");
        spans.push(span("dog", second, second + 3));

        let config = cfg(5, 5);
        let fragments = assemble_fragments(&text, &spans, &config);
        assert_eq!(fragments.len(), 2);
        let formatted = format_fragments(&fragments, &config);
        assert!(
            formatted.contains("... "),
            "non-contiguous passages join with the default ellipsis: {formatted}"
        );
        assert!(formatted.contains("<b>cat</b>") && formatted.contains("<b>dog</b>"));
    }

    #[test]
    fn format_fragments_of_a_single_passage_has_no_ellipsis() {
        let config = cfg(20, 5);
        let fragments = assemble_fragments("cat car cat", &[span("cat", 0, 3)], &config);
        let formatted = format_fragments(&fragments, &config);
        assert!(!formatted.contains("..."));
        assert_eq!(formatted, fragments[0].text);
    }

    #[test]
    fn format_fragments_of_nothing_is_the_empty_string() {
        assert_eq!(format_fragments(&[], &FragmentConfig::default()), "");
    }

    #[test]
    fn escape_html_escapes_content_but_never_the_markers() {
        let text = "a <b>&\"'/ cat";
        let config = FragmentConfig {
            escape: true,
            ..cfg(100, 5)
        };
        let start = text.chars().count() as i32 - 3;
        let out = assemble_fragments(text, &[span("cat", start, start + 3)], &config);
        assert_eq!(out.len(), 1);
        // Java's `DefaultPassageFormatter.append` escapes exactly these six.
        assert!(out[0]
            .text
            .starts_with("a &lt;b&gt;&amp;&quot;&#x27;&#x2F; "));
        // The markers themselves are emitted raw.
        assert!(out[0].text.ends_with("<b>cat</b>"));
    }

    #[test]
    fn overlapping_matches_coalesce_into_one_marked_span() {
        // "cat" and "cats" both start at 0; Java's formatter merges any run
        // of overlapping matches into a single pre/post-wrapped span rather
        // than nesting the markers.
        let text = "cats sleep";
        let spans = [span("cat", 0, 3), span("cats", 0, 4)];
        let out = assemble_fragments(text, &spans, &cfg(100, 5));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "<b>cats</b> sleep");
        assert_eq!(
            out[0].matched_terms,
            vec!["cat".to_string(), "cats".to_string()]
        );
    }

    #[test]
    fn partially_overlapping_matches_extend_to_the_furthest_end() {
        let text = "abcdef ghi";
        // 0..4 and 2..6 overlap; the merged span must cover 0..6.
        let spans = [span("abcd", 0, 4), span("cdef", 2, 6)];
        let out = assemble_fragments(text, &spans, &cfg(100, 5));
        assert_eq!(out[0].text, "<b>abcdef</b> ghi");
    }

    #[test]
    fn empty_spans_yields_no_fragments() {
        let out = assemble_fragments("the quick brown fox", &[], &cfg(10, 5));
        assert!(out.is_empty());
    }

    #[test]
    fn empty_text_yields_no_fragments() {
        let out = assemble_fragments("", &[span("fox", 0, 3)], &cfg(10, 5));
        assert!(out.is_empty());
    }

    #[test]
    fn single_match_produces_one_windowed_highlighted_fragment() {
        let text = "the quick brown fox jumps over the lazy dog near the river bank today";
        // "fox" is at char offset 16..19.
        let spans = [span("fox", 16, 19)];
        let out = assemble_fragments(text, &spans, &cfg(10, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("<b>fox</b>"));
        assert_eq!(out[0].matched_terms, vec!["fox".to_string()]);
        // Window-snapping keeps whole words: no marker artifacts split
        // mid-word, and the fragment is shorter than the full text.
        assert!(out[0].text.len() < text.len());
    }

    #[test]
    fn two_nearby_matches_merge_into_one_fragment_with_both_highlighted() {
        // Two matches 8 chars apart with a 10-char window each side --
        // their raw windows overlap, so they must merge into one fragment,
        // and marker insertion for the second (rightward) match must not
        // corrupt the first (leftward) match's already-inserted markers.
        let text = "alpha cat runs beta car stops gamma delta epsilon zeta";
        //           0     6   10          15  20
        // "cat" at 6..9, "car" at 20..23 (14 chars apart).
        let cat_start = text.find("cat").unwrap() as i32;
        let car_start = text.find("car").unwrap() as i32;
        let spans = [
            span("cat", cat_start, cat_start + 3),
            span("car", car_start, car_start + 3),
        ];
        let out = assemble_fragments(text, &spans, &cfg(20, 5));
        assert_eq!(out.len(), 1, "nearby matches must merge into one fragment");
        assert!(out[0].text.contains("<b>cat</b>"));
        assert!(out[0].text.contains("<b>car</b>"));
        assert_eq!(
            out[0].matched_terms,
            vec!["cat".to_string(), "car".to_string()]
        );
        // Both original words must still be intact (no off-by-one marker
        // corruption from inserting the second match's markers first).
        let unmarked = out[0].text.replace("<b>", "").replace("</b>", "");
        assert!(unmarked.contains("cat"));
        assert!(unmarked.contains("car"));
    }

    #[test]
    fn two_far_apart_matches_produce_two_separate_fragments() {
        let mut text = String::from("cat ");
        text.push_str(&"filler ".repeat(30));
        text.push_str("car");
        let cat_start = 0i32;
        let car_start = text.rfind("car").unwrap() as i32;
        let spans = [
            span("cat", cat_start, cat_start + 3),
            span("car", car_start, car_start + 3),
        ];
        let out = assemble_fragments(&text, &spans, &cfg(10, 5));
        assert_eq!(out.len(), 2, "far-apart matches must not merge");
        assert!(out[0].text.contains("<b>cat</b>"));
        assert!(out[1].text.contains("<b>car</b>"));
    }

    #[test]
    fn window_clamps_at_text_start_and_end_without_panicking() {
        let text = "cat";
        let spans = [span("cat", 0, 3)];
        let out = assemble_fragments(text, &spans, &cfg(50, 5));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "<b>cat</b>");
    }

    #[test]
    fn match_at_very_start_and_very_end_of_longer_text_does_not_panic() {
        let text = "cat is here and the last word is car";
        let cat_span = span("cat", 0, 3);
        let car_start = text.rfind("car").unwrap() as i32;
        let car_span = span("car", car_start, car_start + 3);
        let out = assemble_fragments(text, &[cat_span, car_span], &cfg(3, 5));
        // Small window keeps them from merging; neither slice should panic
        // nor run off either end of `text`.
        assert!(!out.is_empty());
        for f in &out {
            assert!(!f.text.is_empty());
        }
    }

    #[test]
    fn max_fragments_truncates_extra_clusters() {
        // Five widely-separated matches, each far enough apart not to
        // merge, but capped to 2 fragments.
        let mut text = String::new();
        let mut spans = Vec::new();
        for _ in 0..5 {
            text.push_str("cat ");
            let start = text.len() as i32 - 4;
            spans.push(span("cat", start, start + 3));
            text.push_str(&"pad ".repeat(20));
        }
        let out = assemble_fragments(&text, &spans, &cfg(5, 2));
        assert_eq!(out.len(), 2);
    }

    /// Regression test for a real bug: window arithmetic must add
    /// `window_chars` (a CHAR count) to a CHAR offset, not a byte offset --
    /// mixing the two units can push the raw window boundary to land in the
    /// middle of a multi-byte character whenever one falls within
    /// `window_chars` of a match, which panics on slicing. This text places
    /// a 2-byte character ('é') exactly one char before the match, with a
    /// window small enough that byte/char-count-mixed arithmetic would land
    /// the window boundary inside that character's second byte.
    #[test]
    fn window_arithmetic_near_a_multi_byte_char_does_not_panic() {
        let text = "é match here";
        // "match" is at char offset 2..7 (é=0, space=1, m=2).
        let spans = [span("match", 2, 7)];
        let out = assemble_fragments(text, &spans, &cfg(1, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("<b>match</b>"));
    }

    #[test]
    fn multi_byte_utf8_match_does_not_panic_and_highlights_correctly() {
        // "café" -- 'é' is a 2-byte UTF-8 char but one Unicode scalar, at
        // char offset 3 (c-a-f-é), covering char offsets 0..4.
        let text = "café bar café bar café shop is nearby in the city center today for sure";
        let spans = [span("café", 0, 4)];
        let out = assemble_fragments(text, &spans, &cfg(5, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("<b>café</b>"));
    }

    #[test]
    fn out_of_range_and_invalid_spans_are_dropped_not_panicking() {
        let text = "cat dog";
        let spans = [
            span("bad", 100, 200), // entirely out of range
            span("bad2", 5, 2),    // end before start
            span("cat", 0, 3),     // valid
        ];
        let out = assemble_fragments(text, &spans, &cfg(10, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("<b>cat</b>"));
    }

    // Real-fixture-composed test: genuine offsets from task #3/#39's
    // checked-in Java-written fixture (`fixtures/data/term_vectors_index/`,
    // generated by `fixtures/src/GenTermVectors.java`), composed with the
    // REAL text those offsets describe. `GenTermVectors.java`'s doc 0
    // "text" field is built from a `CannedTokenStream` of three tokens --
    // "cat" at char offsets 0..3, "car" at 4..7, "cat" at 8..11 -- which
    // describes the literal text "cat car cat" (space at offset 3, space
    // at offset 7). This is not a made-up string: it is exactly what those
    // real, differentially-verified offsets denote.
    #[test]
    fn real_fixture_offsets_composed_with_their_real_field_text() {
        let full_text = "cat car cat";
        let spans = [span("cat", 0, 3), span("car", 4, 7), span("cat", 8, 11)];
        let out = assemble_fragments(full_text, &spans, &cfg(20, 5));
        // All three matches are within a 20-char window of each other, so
        // they merge into a single fragment spanning the whole text.
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "<b>cat</b> <b>car</b> <b>cat</b>");
        assert_eq!(
            out[0].matched_terms,
            vec!["cat".to_string(), "car".to_string()]
        );
    }

    // --- Sentence-boundary snapping (`snap_to_sentence: true`) ---

    #[test]
    fn sentence_snap_changes_output_vs_naive_fixed_window() {
        // A sentence boundary sits inside what a fixed 15-char window would
        // otherwise include: the naive window spills into the next
        // sentence's leading words, while sentence-snap mode must stop at
        // the sentence containing the match instead.
        let text = "Cats are great pets. Dogs are loyal companions too.";
        let start = text.find("great").unwrap() as i32;
        let end = start + "great".len() as i32;
        let spans = [span("great", start, end)];

        let naive = assemble_fragments(text, &spans, &cfg(15, 5));
        let snapped = assemble_fragments(text, &spans, &sentence_cfg(15, 5));

        assert_eq!(naive.len(), 1);
        assert_eq!(snapped.len(), 1);
        assert_ne!(
            naive[0].text, snapped[0].text,
            "sentence snapping must actually change the fragment boundaries"
        );
        // Sentence-snap keeps the whole first sentence, not a fragment of
        // the second one.
        assert!(snapped[0].text.starts_with("Cats are"));
        assert!(snapped[0].text.ends_with('.'));
        assert!(!snapped[0].text.contains("Dogs"));
    }

    #[test]
    fn sentence_snap_with_no_terminators_still_produces_whole_text_fragment() {
        let text = "no sentence terminators here just plain running text forever";
        let start = text.find("running").unwrap() as i32;
        let end = start + "running".len() as i32;
        let spans = [span("running", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(5, 5));
        assert_eq!(out.len(), 1);
        assert!(!out[0].text.is_empty());
        assert!(out[0].text.contains("<b>running</b>"));
        // No terminators at all -- the whole text is one "sentence".
        assert_eq!(out[0].text.replace("<b>", "").replace("</b>", ""), text);
    }

    // --- `FieldOffsetStrategy` selection -----------------------------------

    #[test]
    fn offset_source_follows_java_field_info_order() {
        use lucene_codecs::field_infos::IndexOptions;
        // Offsets in the postings win, and term vectors upgrade the answer.
        assert_eq!(
            offset_source_for_field(
                Some(IndexOptions::DocsAndFreqsAndPositionsAndOffsets),
                false
            ),
            OffsetSource::Postings
        );
        assert_eq!(
            offset_source_for_field(Some(IndexOptions::DocsAndFreqsAndPositionsAndOffsets), true),
            OffsetSource::PostingsWithTermVectors
        );
        // No offsets, but term vectors: Java takes them without checking they
        // carry offsets.
        assert_eq!(
            offset_source_for_field(Some(IndexOptions::DocsAndFreqsAndPositions), true),
            OffsetSource::TermVectors
        );
        assert_eq!(
            offset_source_for_field(Some(IndexOptions::Docs), true),
            OffsetSource::TermVectors
        );
        // Neither: re-analyze.
        assert_eq!(
            offset_source_for_field(Some(IndexOptions::DocsAndFreqs), false),
            OffsetSource::Analysis
        );
        // No FieldInfo at all is Java's `fieldInfo == null`: it never reaches
        // the `hasTermVectors()` test, so the "no FieldInfo but term vectors"
        // pair -- which cannot arise from a real reader anyway -- is ANALYSIS
        // too, not TERM_VECTORS.
        assert_eq!(offset_source_for_field(None, false), OffsetSource::Analysis);
        assert_eq!(offset_source_for_field(None, true), OffsetSource::Analysis);
    }

    #[test]
    fn optimized_offset_source_follows_javas_switch() {
        // Nothing to highlight at all.
        assert_eq!(
            optimize_offset_source(OffsetSource::Postings, false, false),
            OffsetSource::NoneNeeded
        );
        // A multi-term part makes the postings source useless (it would mean
        // scanning the whole term dictionary), so Java downgrades to ANALYSIS.
        assert_eq!(
            optimize_offset_source(OffsetSource::Postings, true, true),
            OffsetSource::Analysis
        );
        assert_eq!(
            optimize_offset_source(OffsetSource::Postings, false, true),
            OffsetSource::Postings
        );
        // Without a multi-term part the term vectors are not needed.
        assert_eq!(
            optimize_offset_source(OffsetSource::PostingsWithTermVectors, false, true),
            OffsetSource::Postings
        );
        assert_eq!(
            optimize_offset_source(OffsetSource::PostingsWithTermVectors, true, true),
            OffsetSource::PostingsWithTermVectors
        );
        // Everything else passes through, including a multi-term query with
        // no plain terms (Java's `mtqOrRewrite` alone keeps a source alive).
        assert_eq!(
            optimize_offset_source(OffsetSource::TermVectors, true, false),
            OffsetSource::TermVectors
        );
        assert_eq!(
            optimize_offset_source(OffsetSource::Analysis, false, true),
            OffsetSource::Analysis
        );
        assert_eq!(
            optimize_offset_source(OffsetSource::NoneNeeded, true, true),
            OffsetSource::NoneNeeded
        );
    }

    #[test]
    fn analysis_offset_strategy_keeps_only_matching_tokens_with_their_own_offsets() {
        let analyzer = lucene_analysis::Analyzer::standard(None);
        let content = "The quick brown fox jumps over the lazy dog";
        let spans = offsets_from_analysis(&analyzer, content, &["quick", "dog"]);
        let rendered: Vec<(&str, i32, i32)> = spans
            .iter()
            .map(|s| (s.term.as_str(), s.start_offset, s.end_offset))
            .collect();
        assert_eq!(
            rendered,
            vec![
                ("quick", 4, 9),
                ("dog", content.len() as i32 - 3, content.len() as i32),
            ]
        );
        // The offsets really do address the original text.
        for s in &spans {
            assert_eq!(
                &content[s.start_offset as usize..s.end_offset as usize],
                s.term
            );
        }
        // A term the analyzer never produces contributes nothing.
        assert!(offsets_from_analysis(&analyzer, content, &["unicorn"]).is_empty());
        assert!(offsets_from_analysis(&analyzer, "", &["quick"]).is_empty());
    }

    /// c33: [`offsets_from_analysis`] passes the analyzer's offsets through
    /// **unconverted**, because `lucene_analysis::Token` now reports the same
    /// UTF-16 code units this module works in.
    ///
    /// c29 converted here (the analyzer emitted UTF-8 bytes then). Leaving
    /// that conversion in after c33 fixed the producer would shift every
    /// non-ASCII highlight the other way, so this asserts the two sides are
    /// identical -- not merely that the result happens to be right.
    #[test]
    fn analysis_offsets_pass_through_unconverted_in_java_char_units() {
        let analyzer = lucene_analysis::Analyzer::standard(None);
        // "café naïve" is 12 UTF-8 bytes but 10 UTF-16 code units, so a
        // byte-offset producer and a converting consumer are both visible.
        let content = "café naïve dog";
        let raw: Vec<(i32, i32)> = analyzer
            .analyze(content)
            .into_iter()
            .filter(|t| t.term == "dog")
            .map(|t| (t.start_offset, t.end_offset))
            .collect();
        assert_eq!(
            raw,
            vec![(11, 14)],
            "the analyzer reports Java `char` offsets, which is what \
             OffsetAttribute would have reported for the same token"
        );

        let spans = offsets_from_analysis(&analyzer, content, &["dog"]);
        assert_eq!(
            spans
                .iter()
                .map(|s| (s.start_offset, s.end_offset))
                .collect::<Vec<_>>(),
            raw,
            "offsets_from_analysis must not convert anything"
        );

        // ...and they highlight the right word.
        let fragments = assemble_fragments(
            content,
            &spans,
            &FragmentConfig {
                window_chars: 100,
                ..FragmentConfig::default()
            },
        );
        assert_eq!(fragments[0].text, "café naïve <b>dog</b>");

        // An astral character separates Java `char`s from Unicode scalars as
        // well as from bytes: "beta" is at char 9, scalar 8, byte 11.
        let astral = "alpha \u{1F600} beta";
        let beta = offsets_from_analysis(&analyzer, astral, &["beta"]);
        assert_eq!(
            beta.iter()
                .map(|s| (s.start_offset, s.end_offset))
                .collect::<Vec<_>>(),
            vec![(9, 13)]
        );
        assert_eq!(
            assemble_fragments(
                astral,
                &beta,
                &FragmentConfig {
                    window_chars: 100,
                    ..FragmentConfig::default()
                },
            )[0]
            .text,
            "alpha \u{1F600} <b>beta</b>"
        );
    }

    #[test]
    fn utf16_offset_conversions_round_trip_and_clamp() {
        // "😀" is one Unicode scalar, 4 UTF-8 bytes, 2 UTF-16 code units.
        let text = "a😀b";
        assert_eq!(utf16_len(text), 4);
        assert_eq!(utf16_offset_to_byte(text, 0), 0);
        assert_eq!(utf16_offset_to_byte(text, 1), 1);
        // Inside the surrogate pair: rounds down to the emoji's own start.
        assert_eq!(utf16_offset_to_byte(text, 2), 1);
        assert_eq!(utf16_offset_to_byte(text, 3), 5);
        assert_eq!(utf16_offset_to_byte(text, 4), 6);
        // Past the end clamps rather than panicking.
        assert_eq!(utf16_offset_to_byte(text, 99), text.len());
        assert_eq!(byte_offset_to_utf16(text, 0), 0);
        assert_eq!(byte_offset_to_utf16(text, 1), 1);
        assert_eq!(byte_offset_to_utf16(text, 5), 3);
        assert_eq!(byte_offset_to_utf16(text, 6), 4);
        assert_eq!(byte_offset_to_utf16(text, 99), 4);
        assert_eq!(utf16_len(""), 0);
        assert_eq!(utf16_offset_to_byte("", 3), 0);
    }

    #[test]
    fn analysis_offsets_come_back_in_offsets_enum_order() {
        let analyzer = lucene_analysis::Analyzer::standard(None);
        let content = "dog cat dog";
        let spans = offsets_from_analysis(&analyzer, content, &["dog", "cat"]);
        let starts: Vec<i32> = spans.iter().map(|s| s.start_offset).collect();
        assert_eq!(starts, vec![0, 4, 8]);
    }

    #[test]
    fn sentence_snap_breaks_after_an_abbreviation_exactly_as_the_jdk_does() {
        // Until c12 this port carried a hardcoded English abbreviation list
        // that suppressed the break after "Mr.". That was a divergence, not a
        // refinement: `BreakIterator.getSentenceInstance(Locale.ROOT)` --
        // and `Locale.ENGLISH` -- splits this text as
        // `["Mr. ", "Smith arrived early. ", "He left before noon."]`, applying
        // no abbreviation suppression at all (verified by running the JDK; see
        // `sentence_boundaries_match_jdk_break_iterator`). The highlighter now
        // cuts where Lucene cuts.
        let text = "Mr. Smith arrived early. He left before noon.";
        let start = text.find("Smith").unwrap() as i32;
        let end = start + "Smith".len() as i32;
        let spans = [span("Smith", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(5, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.starts_with("<b>Smith</b>"));
        assert!(out[0].text.ends_with("early."));
        assert!(!out[0].text.contains("He left"));
    }

    /// The whole point of replacing the heuristic: the JDK's own answers,
    /// read from `fixtures/data/break_iterator/manifest.properties` --
    /// `BreakIterator.getSentenceInstance` run over each text by
    /// `fixtures/src/GenBreakIterator.java` and regenerable through
    /// `scripts/gen-fixtures.sh`, so a JDK or CLDR data change shows up as a
    /// failing check rather than as passages silently drifting away from
    /// Lucene's.
    ///
    /// Compares the sliced *substrings*, not offsets: `BreakIterator` reports
    /// UTF-16 offsets and this reports UTF-8 byte offsets, and every fixture
    /// text is ASCII precisely so the comparison never has to bridge the two.
    #[test]
    fn sentence_boundaries_match_the_jdks_break_iterator() {
        let text = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/break_iterator/manifest.properties"
        ))
        .expect("run scripts/gen-fixtures.sh first (GenBreakIterator)");
        let get = |key: &str| -> String {
            text.lines()
                .find_map(|l| l.strip_prefix(&format!("{key}=")))
                .unwrap_or_else(|| panic!("manifest key {key} missing"))
                .to_string()
        };
        // Inverse of `GenBreakIterator.escape`.
        let unescape = |s: &str| -> String {
            let mut out = String::new();
            let mut it = s.chars();
            while let Some(c) = it.next() {
                if c != '\\' {
                    out.push(c);
                    continue;
                }
                match it.next() {
                    Some('\\') => out.push('\\'),
                    Some('n') => out.push('\n'),
                    Some('u') => {
                        let hex: String = (0..4).filter_map(|_| it.next()).collect();
                        out.push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
                    }
                    other => panic!("unknown escape \\{other:?}"),
                }
            }
            out
        };
        let split = |s: &str| -> Vec<String> {
            if s.is_empty() {
                Vec::new()
            } else {
                s.split('\u{1}').map(&unescape).collect()
            }
        };

        let count: usize = get("count").parse().unwrap();
        assert!(count >= 10, "the fixture must actually carry texts");
        let mut checked_a_multi_sentence_case = false;
        for i in 0..count {
            let input = unescape(&get(&format!("text.{i}")));
            let want = split(&get(&format!("root.{i}")));
            // `UnifiedHighlighter`'s default is `Locale.ROOT`; the fixture
            // records `Locale.ENGLISH` too, and this port is untailored, so the
            // two must agree for the port to be right for either.
            assert_eq!(
                want,
                split(&get(&format!("english.{i}"))),
                "text {i}: ROOT and ENGLISH disagree, so an untailored port \
                 cannot match both"
            );

            let b = sentence_boundaries(&input);
            let got: Vec<String> = b
                .windows(2)
                .map(|w| input[w[0]..w[1]].to_string())
                .collect();
            assert_eq!(got, want, "text {i} ({input:?})");
            if want.len() > 1 {
                checked_a_multi_sentence_case = true;
            }
        }
        assert!(
            checked_a_multi_sentence_case,
            "a fixture where every text is one sentence proves nothing"
        );
    }

    /// Degenerate inputs the fixture cannot express as a boundary list.
    #[test]
    fn sentence_boundaries_of_degenerate_inputs() {
        assert_eq!(sentence_boundaries(""), vec![0]);
        // A text that is nothing but whitespace is one "sentence", not many.
        assert_eq!(sentence_boundaries("\n\n\n"), vec![0, 3]);
        assert_eq!(sentence_boundaries("   "), vec![0, 3]);
    }

    /// `SplittingBreakIterator(sentenceIterator, MULTIVAL_SEP_CHAR)`: the
    /// splitting character is always a boundary on both sides, and the
    /// enclosed iterator never sees it -- so a passage can never straddle two
    /// values of a multi-valued field.
    #[test]
    fn split_sentence_boundaries_reports_every_slice_char_as_a_boundary() {
        let text = "One. Two.\u{0}Three. Four.";
        let b = split_sentence_boundaries(text, '\u{0}');
        let sep = text.find('\u{0}').unwrap();
        assert!(b.contains(&sep), "the separator itself is a boundary");
        assert!(b.contains(&(sep + 1)), "and so is the position after it");
        for w in b.windows(2) {
            assert!(
                !(w[0] < sep && w[1] > sep + 1),
                "a slice must not straddle the separator"
            );
        }
        // Adjacent separators, and separators at either end, still produce
        // boundaries rather than empty confusion.
        let edges = split_sentence_boundaries("\u{0}\u{0}a\u{0}", '\u{0}');
        assert_eq!(edges, vec![0, 1, 2, 3, 4]);
    }

    #[test]
    fn sentence_snap_ordinal_number_is_not_mistaken_for_st_abbreviation() {
        // "st" is on ABBREVIATIONS (for "St." as in a street name), which
        // would incorrectly also suppress a sentence break after any
        // ordinal number ending in "st" (21st, 1st, 91st, ...) unless the
        // abbreviation check excludes an alphabetic run directly preceded
        // by a digit. Found in review: without that guard, this exact text
        // incorrectly merged into one fragment instead of breaking after
        // "21st.".
        let text = "He finished 21st. She started next.";
        let start = text.find("She").unwrap() as i32;
        let end = start + "She".len() as i32;
        let spans = [span("She", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(5, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.starts_with("<b>She</b>"));
        assert!(!out[0].text.contains("21st"));
    }

    #[test]
    fn sentence_snap_closing_quote_after_terminator_is_recognized() {
        // A terminator immediately followed by a closing quote, then
        // whitespace and an uppercase letter, must still be recognized as a
        // sentence break -- without quote-skipping, the char right after
        // the period is `"` (not whitespace), so the old heuristic would
        // fail to find the break here and spill into the quoted sentence.
        let text = "He said \"Stop.\" Then he left the room for good today.";
        let start = text.find("Then").unwrap() as i32;
        let end = start + "Then".len() as i32;
        let spans = [span("Then", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.starts_with("<b>Then</b>"));
        assert!(!out[0].text.contains("Stop"));
    }

    #[test]
    fn sentence_snap_closing_paren_after_terminator_is_recognized() {
        let text = "(See note.) Next sentence begins here and continues on.";
        let start = text.find("Next").unwrap() as i32;
        let end = start + "Next".len() as i32;
        let spans = [span("Next", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.starts_with("<b>Next</b>"));
        assert!(!out[0].text.contains("See note"));
    }

    #[test]
    fn sentence_snap_match_at_very_start_and_very_end_does_not_panic() {
        let text = "First sentence here. Middle sentence stands alone. Last sentence ends.";
        let first_word_end = "First".len() as i32;
        let last_word_start = text.rfind("Last").unwrap() as i32;
        let last_word_end = last_word_start + "Last".len() as i32;
        let spans = [
            span("First", 0, first_word_end),
            span("Last", last_word_start, last_word_end),
        ];

        let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
        assert!(!out.is_empty());
        for f in &out {
            assert!(!f.text.is_empty());
        }
        // The very-first match's fragment must start right at the text's
        // start (byte offset 0), not run off the front.
        let first_fragment = out
            .iter()
            .find(|f| f.text.contains("<b>First</b>"))
            .expect("a fragment containing the first match");
        assert!(first_fragment.text.starts_with("<b>First</b>"));
        // The very-last match's fragment must end at (or before) the text's
        // end, not run off the back.
        let last_fragment = out
            .iter()
            .find(|f| f.text.contains("<b>Last</b>"))
            .expect("a fragment containing the last match");
        assert!(last_fragment.text.ends_with('.'));
    }

    #[test]
    fn sentence_snap_re_merges_clusters_that_expand_into_the_same_sentence() {
        // A small window_chars keeps these two matches in separate clusters
        // under the fixed-window sweep (they're far apart), but they both
        // fall inside the same long sentence -- sentence-snapping expands
        // both clusters' windows to that whole sentence. Without a re-merge
        // pass after snapping, this produced two separate, nearly-identical,
        // overlapping fragments instead of one fragment with both matches
        // highlighted.
        let text = "One two three four five six seven eight nine ten eleven twelve \
                     thirteen fourteen fifteen sixteen. Next sentence word.";
        let one_start = text.find("One").unwrap() as i32;
        let one_end = one_start + "One".len() as i32;
        let sixteen_start = text.find("sixteen").unwrap() as i32;
        let sixteen_end = sixteen_start + "sixteen".len() as i32;
        let spans = [
            span("One", one_start, one_end),
            span("sixteen", sixteen_start, sixteen_end),
        ];

        let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
        assert_eq!(
            out.len(),
            1,
            "expected the two same-sentence clusters to merge into one fragment, got: {out:?}"
        );
        assert!(out[0].text.contains("<b>One</b>"));
        assert!(out[0].text.contains("<b>sixteen</b>"));
    }

    #[test]
    fn sentence_snap_lowercase_after_period_is_not_a_terminator() {
        // "3.5" -- a period followed by a lowercase/digit character is not
        // this heuristic's terminator (it requires whitespace then an
        // uppercase letter, or end-of-text), so the fragment must not break
        // there.
        let text = "The price is 3.5 and rising steadily today.";
        let start = text.find("3.5").unwrap() as i32;
        let end = start + "3.5".len() as i32;
        let spans = [span("3.5", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.starts_with("The price is"));
        assert!(out[0].text.ends_with("today."));
    }

    #[test]
    fn sentence_snap_consecutive_terminators_do_not_panic() {
        // "Really?!" and "Wow..." both have runs of terminator characters;
        // the heuristic must not double-count them or panic walking past
        // the run.
        let text = "Really?! Wow... That is surprising indeed today.";
        let start = text.find("Wow").unwrap() as i32;
        let end = start + "Wow".len() as i32;
        let spans = [span("Wow", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.contains("<b>Wow</b>"));
    }

    #[test]
    fn sentence_snap_single_word_no_terminator_no_whitespace() {
        let text = "Hello";
        let spans = [span("Hello", 0, 5)];
        let out = assemble_fragments(text, &spans, &sentence_cfg(3, 5));
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].text, "<b>Hello</b>");
    }
}
