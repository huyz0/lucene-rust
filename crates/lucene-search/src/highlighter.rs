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
//! passage boundaries, then scores candidate passages by term density
//! before picking the top N. **This module does neither.** Its passage
//! heuristic is deliberately simple and is documented here as exactly
//! that -- a simplification, not a silent stand-in for `BreakIterator`:
//!
//! - Each match (or cluster of nearby matches) gets a fixed-size character
//!   window: `window_chars` before the earliest match in the cluster and
//!   `window_chars` after the latest, clamped to the text's start/end.
//! - The window's cut edges are snapped outward to the nearest whitespace
//!   boundary (so a fragment doesn't begin or end mid-word) where possible;
//!   if no whitespace is found before hitting the text boundary, it uses
//!   the boundary itself.
//! - There is no term-density scoring -- fragments are emitted in
//!   left-to-right document order and simply truncated at `max_fragments`.
//!
//! ## Sentence-boundary snapping (opt-in via `FragmentConfig::snap_to_sentence`)
//!
//! `assemble_fragments`'s default behavior above (fixed-size char window,
//! snapped outward to whitespace) is unchanged. Setting
//! [`FragmentConfig::snap_to_sentence`] to `true` switches a fragment's edges
//! from that fixed window to the boundaries of the sentence(s) actually
//! containing its match(es) -- closer to real `UnifiedHighlighter`'s
//! `BreakIterator.getSentenceInstance()`-based passage boundaries, but with a
//! deliberately narrow, explicitly-scoped heuristic instead of ICU's full
//! Unicode sentence-segmentation algorithm (UAX #29-style, locale-aware,
//! abbreviation-dictionary-aware):
//!
//! - A sentence is considered to end at a `.`/`!`/`?` that is followed
//!   (after skipping any whitespace) by an uppercase letter, or by the end of
//!   the text.
//! - A fragment's start snaps to the start of the sentence containing its
//!   cluster's earliest match; its end snaps to the end of the sentence
//!   containing its cluster's latest match (trailing whitespace trimmed).
//!   `window_chars` still governs which nearby matches get merged into one
//!   cluster, but no longer bounds the rendered fragment's size once
//!   sentence-snapped -- a fragment can be shorter *or* longer than the fixed
//!   window it would have used, since it's exactly the sentence(s), not a
//!   char count.
//! - If the text has no recognized sentence terminator at all, the whole
//!   text is one sentence: the fragment still comes out sensible (the
//!   surrounding sentence extends to a document/text boundary), never empty
//!   or panicking.
//! - **A small, hardcoded, English-only abbreviation list** (see
//!   [`ABBREVIATIONS`]) suppresses a sentence break when the word
//!   immediately preceding the terminator (case-insensitively) is one of
//!   "Mr", "Mrs", "Ms", "Dr", "Jr", "Sr", "vs", "etc", "Inc", "St", "Prof",
//!   "Capt", "Co", "Ltd", "Gen" -- this closes the exact "Mr. Smith" false
//!   positive from an earlier version of this heuristic. **This list is not
//!   a comprehensive abbreviation dictionary and never will be**: any
//!   abbreviation not on it (e.g. "Gen." misspelled, or any non-English
//!   abbreviation) still produces the same false-positive sentence break as
//!   before -- there is no locale table or ICU `BreakIterator` behind this,
//!   by design (see this module's doc comment above and `docs/parity.md`'s
//!   highlighter row).
//! - **Closing quote/paren after the terminator**: a terminator immediately
//!   followed by a closing quote (`"`, `'`, U+201D `"`, U+2019 `'`) or
//!   paren/bracket (`)`, `]`, `}`) before any whitespace -- e.g. `He said
//!   "stop." Then left.` or `(See note.) Next sentence.` -- has that
//!   closing punctuation skipped before the whitespace+uppercase check, so
//!   the sentence break is still recognized right after the closing
//!   punctuation rather than missed entirely.
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
//! ## Offset units: char offsets, and this port's fixture data is ASCII-only
//!
//! Real Lucene's `OffsetAttribute`/`Analyzer` offsets are UTF-16 code-unit
//! offsets into the original `String`. `crates/lucene-codecs/src/
//! term_vectors.rs` decodes these `start_offset`/`end_offset` values
//! verbatim off disk -- whatever unit the indexing-time `Analyzer` wrote,
//! task #39/#3 never reinterpret them (see `term_vectors_query.rs`'s own
//! doc comment: "character offsets", inherited from real Lucene's
//! contract). This port's checked-in `fixtures/data/term_vectors_index/`
//! fixture (`fixtures/src/GenTermVectors.java`) only ever indexes ASCII
//! terms ("cat"/"car"/"dog"/"run"/"hello"), so UTF-16-code-unit, UTF-8-byte,
//! and Unicode-scalar-count offsets are numerically identical for it --
//! that fixture cannot, by itself, distinguish the three.
//!
//! Rather than assume ASCII-only callers forever, this module treats
//! incoming offsets as **Unicode scalar (`char`) counts** -- the same unit
//! Rust's own `str::chars()` naturally indexes by, and numerically
//! equivalent to UTF-16 code units for the entire Basic Multilingual Plane
//! (everything outside supplementary-plane astral characters, which is
//! effectively all real text). [`char_offset_to_byte`] converts a char
//! offset to a UTF-8 byte offset by walking `char_indices()`, so slicing
//! `full_text` is always on a valid UTF-8 boundary -- it cannot panic on a
//! multi-byte character even if the input offsets are wrong or
//! out-of-range (they're clamped to `full_text.chars().count()` first).
//! This is documented as a real design decision, not silently assumed.

/// One assembled, highlighted fragment of field text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Fragment {
    /// The fragment's text, with each match wrapped in `pre`/`post` markers.
    pub text: String,
    /// The distinct matched terms found within this fragment, in the order
    /// they first appear (a repeated term is listed once).
    pub matched_terms: Vec<String>,
}

/// Configuration for [`assemble_fragments`].
#[derive(Debug, Clone)]
pub struct FragmentConfig {
    /// Characters of context to keep before the earliest match and after
    /// the latest match in a cluster, before whitespace-snapping and
    /// clamping to the text's bounds.
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
}

impl Default for FragmentConfig {
    /// `window_chars: 40`, `pre: "<b>"`, `post: "</b>"` (real Lucene's
    /// `PassageFormatter` default markers), `max_fragments: 5`,
    /// `snap_to_sentence: false`.
    fn default() -> Self {
        FragmentConfig {
            window_chars: 40,
            pre: "<b>".to_string(),
            post: "</b>".to_string(),
            max_fragments: 5,
            snap_to_sentence: false,
        }
    }
}

/// Small, hardcoded, English-only list of common abbreviations whose
/// trailing `.` must not be treated as a sentence terminator by
/// [`sentence_start_offsets`], even when followed by a capitalized word --
/// e.g. "Mr. Smith" is one sentence, not two. Matched case-insensitively
/// against the word immediately preceding the terminator. **This is
/// intentionally a short fixed set, not a comprehensive dictionary**: any
/// abbreviation not listed here still triggers the same (documented) false
/// positive as before. See this module's doc comment section on
/// sentence-boundary snapping.
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "jr", "sr", "vs", "etc", "inc", "st", "prof", "capt", "co", "ltd",
    "gen",
];

/// Closing quote or paren/bracket characters that may sit between a
/// sentence terminator and the whitespace that follows it (e.g. `stop." `
/// or `note.) `) -- see this module's doc comment section on
/// sentence-boundary snapping.
fn is_closing_quote_or_paren(c: char) -> bool {
    matches!(c, '"' | '\'' | ')' | ']' | '}' | '\u{201D}' | '\u{2019}')
}

/// Returns whether the word ending immediately before `terminator_idx`
/// (the index, into `chars`, of the `.`/`!`/`?` itself) case-insensitively
/// matches one of [`ABBREVIATIONS`]. `chars` is a `char_indices()`-style
/// slice of the full text.
fn ends_with_abbreviation(chars: &[(usize, char)], terminator_idx: usize) -> bool {
    let mut j = terminator_idx;
    let mut word: Vec<char> = Vec::new();
    while j > 0 && chars[j - 1].1.is_alphabetic() {
        j -= 1;
        word.push(chars[j].1);
    }
    if word.is_empty() {
        return false;
    }
    // An alphabetic run directly preceded by a digit is an ordinal suffix
    // (21st./1st./2nd.), not the abbreviation "St." -- without this guard,
    // "st" (on the list for "St." as in "Main St.") would also suppress a
    // real sentence break after any ordinal ending the sentence, e.g. "He
    // finished 21st. She started next."
    if j > 0 && chars[j - 1].1.is_ascii_digit() {
        return false;
    }
    word.reverse();
    let word: String = word.into_iter().collect::<String>().to_lowercase();
    ABBREVIATIONS.contains(&word.as_str())
}

/// Byte offsets (into `text`) where each recognized sentence begins, always
/// including `0` and always sorted ascending. See this module's doc comment
/// section on sentence-boundary snapping for the exact terminator rule and
/// its documented scope (small fixed abbreviation list, closing-quote/paren
/// skipping, no locale/ICU semantics).
fn sentence_start_offsets(text: &str) -> Vec<usize> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let mut starts = vec![0usize];
    for i in 0..chars.len() {
        let (_, c) = chars[i];
        if c == '.' || c == '!' || c == '?' {
            if c == '.' && ends_with_abbreviation(&chars, i) {
                continue;
            }
            let mut j = i + 1;
            // Skip a closing quote/paren sitting between the terminator and
            // any following whitespace (e.g. `stop." Then`).
            while j < chars.len() && is_closing_quote_or_paren(chars[j].1) {
                j += 1;
            }
            while j < chars.len() && chars[j].1.is_whitespace() {
                j += 1;
            }
            if j < chars.len() && chars[j].1.is_uppercase() {
                starts.push(chars[j].0);
            }
        }
    }
    starts.sort_unstable();
    starts.dedup();
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

/// Converts a Unicode-scalar (`char`) offset into `text` to a UTF-8 byte
/// offset, clamped to `text`'s length -- never panics, never lands on a
/// non-UTF-8-boundary byte index, regardless of how out-of-range or
/// mis-unitted `char_offset` is.
fn char_offset_to_byte(text: &str, char_offset: usize) -> usize {
    match text.char_indices().nth(char_offset) {
        Some((byte_idx, _)) => byte_idx,
        None => text.len(),
    }
}

/// Snaps `byte_offset` outward (leftward for a window start, rightward for
/// a window end) to the nearest ASCII-or-Unicode whitespace boundary within
/// `text`, so a fragment window doesn't start or end mid-word. Falls back
/// to the original offset (which is always a valid char boundary, since
/// [`char_offset_to_byte`] only ever returns char-boundary indices) if no
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

    let total_chars = full_text.chars().count();

    // Convert every valid span to (byte_start, byte_end, char_start,
    // char_end, term) and its own raw (unmerged) window, sorted by match
    // start so overlap-merging can be a single left-to-right sweep. Window
    // arithmetic below is done entirely in CHAR space (matching
    // `config.window_chars`'s unit) before converting to bytes, so it can
    // never straddle a multi-byte character boundary regardless of how far
    // `window_chars` reaches from a match.
    let mut matches: Vec<(usize, usize, usize, usize, String)> = spans
        .iter()
        .filter(|s| s.start_offset >= 0 && s.end_offset >= s.start_offset)
        .filter(|s| (s.start_offset as usize) <= total_chars)
        .map(|s| {
            let start_char = s.start_offset as usize;
            let end_char = (s.end_offset as usize).min(total_chars);
            (
                char_offset_to_byte(full_text, start_char),
                char_offset_to_byte(full_text, end_char),
                start_char,
                end_char,
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
    for (match_start, match_end, start_char, end_char, term) in matches {
        let raw_window_start_char = start_char.saturating_sub(config.window_chars);
        let raw_window_end_char = (end_char + config.window_chars).min(total_chars);
        let raw_window_start = char_offset_to_byte(full_text, raw_window_start_char);
        let raw_window_end = char_offset_to_byte(full_text, raw_window_end_char);
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

    clusters
        .into_iter()
        .take(config.max_fragments)
        .map(|cluster| render_cluster(full_text, &cluster, config))
        .collect()
}

/// Renders one cluster into a [`Fragment`]: slices `full_text` to the
/// cluster's window, then inserts `pre`/`post` markers around each match --
/// working from the last match backward so an earlier insertion's byte
/// offsets are never invalidated by a later (rightward) one.
fn render_cluster(full_text: &str, cluster: &Cluster, config: &FragmentConfig) -> Fragment {
    let window_text = &full_text[cluster.window_start..cluster.window_end];

    let mut text = window_text.to_string();

    // `matched_terms` in left-to-right first-occurrence order, computed
    // independently of the (backward) marker-insertion pass below.
    let mut matched_terms: Vec<String> = Vec::new();
    for (_, _, term) in &cluster.matches {
        if !matched_terms.contains(term) {
            matched_terms.push(term.clone());
        }
    }

    // Insert markers back-to-front so each match's byte offsets (relative
    // to the window) stay valid for the next (earlier) insertion.
    for (match_start, match_end, _term) in cluster.matches.iter().rev() {
        // Offsets relative to the window's start; clamp defensively in
        // case a match's raw span reached beyond the (whitespace-snapped)
        // window edge.
        let rel_start = match_start
            .saturating_sub(cluster.window_start)
            .min(text.len());
        let rel_end = match_end
            .saturating_sub(cluster.window_start)
            .min(text.len())
            .max(rel_start);

        text.insert_str(rel_end, &config.post);
        text.insert_str(rel_start, &config.pre);
    }

    Fragment {
        text,
        matched_terms,
    }
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
            pre: "<b>".to_string(),
            post: "</b>".to_string(),
            max_fragments,
            snap_to_sentence: false,
        }
    }

    fn sentence_cfg(window_chars: usize, max_fragments: usize) -> FragmentConfig {
        FragmentConfig {
            snap_to_sentence: true,
            ..cfg(window_chars, max_fragments)
        }
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

    #[test]
    fn sentence_snap_abbreviation_list_closes_mr_smith_false_positive() {
        // Regression test: an earlier version of this heuristic treated
        // "Mr." as a sentence end (a documented false positive). With the
        // `ABBREVIATIONS` list, "Mr." followed by "Smith" is now correctly
        // NOT a sentence break -- the fragment must include the whole "Mr.
        // Smith arrived early." sentence, not start mid-sentence at "Smith".
        let text = "Mr. Smith arrived early. He left before noon.";
        let start = text.find("Smith").unwrap() as i32;
        let end = start + "Smith".len() as i32;
        let spans = [span("Smith", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(5, 5));
        assert_eq!(out.len(), 1);
        assert!(out[0].text.starts_with("Mr. <b>Smith</b>"));
        assert!(out[0].text.ends_with("early."));
        assert!(!out[0].text.contains("He left"));
    }

    #[test]
    fn sentence_snap_unlisted_abbreviation_still_breaks() {
        // "Gen." IS on the list (checks the negative would be pointless);
        // use a title that's deliberately NOT in `ABBREVIATIONS` to prove
        // this extension suppresses breaks only for the specific listed
        // abbreviations, not universally after every period. "Cmdr." is not
        // on the list, so the pre-existing (documented) false positive must
        // still occur here.
        let text = "Cmdr. Ripley reported in. She left before noon.";
        let start = text.find("Ripley").unwrap() as i32;
        let end = start + "Ripley".len() as i32;
        let spans = [span("Ripley", start, end)];

        let out = assemble_fragments(text, &spans, &sentence_cfg(5, 5));
        assert_eq!(out.len(), 1);
        // Not suppressed: "Cmdr." is (still, correctly per this heuristic's
        // documented scope) treated as a sentence end, so the fragment
        // starts at "Ripley", not "Cmdr. Ripley".
        assert!(out[0].text.starts_with("<b>Ripley</b>"));
        assert!(!out[0].text.contains("Cmdr."));
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
