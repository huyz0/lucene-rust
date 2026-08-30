//! The highlighter's **offset unit**, pinned against real Lucene.
//!
//! Lucene's `OffsetAttribute` offsets are indices into the original `String`,
//! so they are UTF-16 code units, and `DefaultPassageFormatter` slices the
//! stored text with them (`String.substring(start, end)`). Every offset this
//! port's highlighter consumes -- decoded verbatim off a `.tvd`/`.pos` by
//! `lucene-codecs`, or handed over the FFI by a JVM caller -- is therefore in
//! that unit.
//!
//! Nothing in this repo could catch a reader using the wrong one, because
//! every other fixture is pure ASCII, where UTF-8 bytes, Unicode scalars and
//! UTF-16 code units are all the same number. `fixtures/src/GenBreakIterator.java`
//! now also records, for texts that separate all three, every token a real
//! `StandardAnalyzer` produces as `term:start,end:slice` where `slice` is
//! Java's own `text.substring(start, end)`.
//!
//! The assertion is text-to-text: feed Lucene's `(start, end)` to
//! `assemble_fragments` and the marked-up region must be exactly the slice
//! Lucene's own `substring` produced. That leaves no room to re-derive the
//! offsets on the Rust side and accidentally agree with itself.
//!
//! Before `c29-search-carryovers` the highlighter read these offsets as
//! Unicode scalars, so `offset_text.2` (`"alpha 😀 beta 𝐀 gamma"`) highlighted
//! the wrong span for every token after the first emoji.

use lucene_search::highlighter::{assemble_fragments, FragmentConfig};
use lucene_search::term_vectors_query::TermOffsetSpan;

fn manifest() -> Vec<(String, String)> {
    let path = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/break_iterator/manifest.properties"
    );
    let text = std::fs::read_to_string(path)
        .expect("run scripts/gen-fixtures.sh first (GenBreakIterator)");
    text.lines()
        .filter_map(|l| l.split_once('='))
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn get<'m>(m: &'m [(String, String)], key: &str) -> &'m str {
    m.iter()
        .find(|(k, _)| k == key)
        .map(|(_, v)| v.as_str())
        .unwrap_or_else(|| panic!("manifest key {key} missing -- re-run scripts/gen-fixtures.sh"))
}

/// Undoes `GenBreakIterator.escape`.
fn unescape(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars();
    while let Some(c) = chars.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match chars.next() {
            Some('n') => out.push('\n'),
            Some('\\') => out.push('\\'),
            Some('u') => {
                // Only `` (the manifest separator) is ever emitted.
                let hex: String = chars.by_ref().take(4).collect();
                out.push(char::from_u32(u32::from_str_radix(&hex, 16).unwrap()).unwrap());
            }
            Some(other) => {
                out.push('\\');
                out.push(other);
            }
            None => out.push('\\'),
        }
    }
    out
}

struct Token {
    term: String,
    start: i32,
    end: i32,
    /// `text.substring(start, end)` as Java computed it.
    slice: String,
}

fn tokens(m: &[(String, String)], i: usize) -> Vec<Token> {
    let raw = get(m, &format!("offset_tokens.{i}"));
    if raw.is_empty() {
        return Vec::new();
    }
    raw.split('\u{1}')
        .map(|entry| {
            // `term:start,end:slice` -- split from the *left* twice, since a
            // slice may itself contain a `:`.
            let (term, rest) = entry.split_once(':').unwrap();
            let (span, slice) = rest.split_once(':').unwrap();
            let (start, end) = span.split_once(',').unwrap();
            Token {
                term: unescape(term),
                start: start.parse().unwrap(),
                end: end.parse().unwrap(),
                slice: unescape(slice),
            }
        })
        .collect()
}

fn texts(m: &[(String, String)]) -> Vec<String> {
    let count: usize = get(m, "offset_count").parse().unwrap();
    (0..count)
        .map(|i| unescape(get(m, &format!("offset_text.{i}"))))
        .collect()
}

/// The fixture is only evidence if its texts actually separate the three
/// candidate units. Assert that before relying on it.
#[test]
fn the_fixture_texts_separate_bytes_scalars_and_utf16_code_units() {
    let m = manifest();
    let all = texts(&m);
    assert!(!all.is_empty());

    // Every text has more UTF-8 bytes than UTF-16 code units (non-ASCII).
    for (i, text) in all.iter().enumerate() {
        let utf16: usize = text.chars().map(char::len_utf16).sum();
        let scalars = text.chars().count();
        assert_eq!(
            utf16,
            get(&m, &format!("offset_utf16_length.{i}"))
                .parse::<usize>()
                .unwrap(),
            "text {i}: this port's UTF-16 length must match Java's String.length()"
        );
        assert!(
            text.len() > utf16,
            "text {i} must have more UTF-8 bytes than UTF-16 code units"
        );
        assert!(scalars <= utf16);
    }

    // ...and at least one text also separates Unicode scalars from UTF-16
    // code units, which only a supplementary-plane character does.
    assert!(
        all.iter()
            .any(|t| t.chars().count() < t.chars().map(char::len_utf16).sum::<usize>()),
        "no fixture text contains a supplementary-plane character, so nothing \
         distinguishes a `char`-count reader from a UTF-16 one"
    );
}

/// The load-bearing assertion: highlighting with Lucene's own offsets marks
/// exactly the text Lucene's own `substring` selects.
#[test]
fn highlighting_with_lucenes_offsets_marks_exactly_lucenes_own_substring() {
    let m = manifest();
    let config = FragmentConfig {
        // A window big enough that every fragment is the whole text, so the
        // only thing under test is where the markers land.
        window_chars: 1000,
        max_fragments: 10,
        ..FragmentConfig::default()
    };

    let mut checked = 0usize;
    for (i, text) in texts(&m).iter().enumerate() {
        for token in tokens(&m, i) {
            let spans = [TermOffsetSpan {
                term: token.term.clone(),
                start_offset: token.start,
                end_offset: token.end,
            }];
            let fragments = assemble_fragments(text, &spans, &config);
            assert_eq!(fragments.len(), 1, "text {i} term {}", token.term);
            let marked = fragments[0]
                .text
                .split_once("<b>")
                .and_then(|(_, rest)| rest.split_once("</b>"))
                .map(|(inner, _)| inner.to_string())
                .unwrap_or_else(|| panic!("no marked span for {} in text {i}", token.term));
            assert_eq!(
                marked, token.slice,
                "text {i}: highlighting Lucene's offsets {}..{} for term {} must mark \
                 Lucene's own substring",
                token.start, token.end, token.term
            );
            checked += 1;
        }
    }
    assert!(checked >= 15, "only {checked} tokens checked");
}

/// A whole-text highlight of every token at once, which is what a real
/// highlight of a multi-term query does -- the fragment boundaries and the
/// reported `start_offset`/`end_offset` must also be in Lucene's unit.
#[test]
fn a_whole_text_fragments_offsets_are_reported_in_utf16_code_units() {
    let m = manifest();
    let config = FragmentConfig {
        window_chars: 1000,
        max_fragments: 1,
        ..FragmentConfig::default()
    };
    for (i, text) in texts(&m).iter().enumerate() {
        let spans: Vec<TermOffsetSpan> = tokens(&m, i)
            .into_iter()
            .map(|t| TermOffsetSpan {
                term: t.term,
                start_offset: t.start,
                end_offset: t.end,
            })
            .collect();
        let fragments = assemble_fragments(text, &spans, &config);
        assert_eq!(fragments.len(), 1);
        assert_eq!(fragments[0].start_offset, 0);
        assert_eq!(
            fragments[0].end_offset,
            get(&m, &format!("offset_utf16_length.{i}"))
                .parse::<usize>()
                .unwrap(),
            "text {i}: a fragment spanning the whole text must end at Java's \
             String.length(), not at its byte length or its scalar count"
        );
        // Stripping the markers must give the original text back untouched --
        // an offset in the wrong unit would have sliced mid-character or
        // duplicated/lost a character here.
        let stripped = fragments[0].text.replace("<b>", "").replace("</b>", "");
        assert_eq!(&stripped, text, "text {i}");
    }
}

/// The negative control the module doc names: reading Lucene's offsets as
/// Unicode scalars (what this port did before `c29-search-carryovers`) marks
/// the wrong span as soon as a supplementary-plane character precedes the
/// token. Without this, the test above could pass for a reader that got the
/// unit wrong in a way the BMP happens to hide.
#[test]
fn reading_the_same_offsets_as_unicode_scalars_would_mark_the_wrong_span() {
    let m = manifest();
    let mut disagreements = 0usize;
    for (i, text) in texts(&m).iter().enumerate() {
        for token in tokens(&m, i) {
            // What the old `char_indices().nth(n)` conversion would have
            // produced for the same offsets.
            let scalar_start = text
                .char_indices()
                .nth(token.start as usize)
                .map_or(text.len(), |(b, _)| b);
            let scalar_end = text
                .char_indices()
                .nth(token.end as usize)
                .map_or(text.len(), |(b, _)| b);
            if scalar_start <= scalar_end && text[scalar_start..scalar_end] != token.slice {
                disagreements += 1;
            }
        }
    }
    assert!(
        disagreements > 0,
        "the fixture must contain at least one token a scalar-count reader gets \
         wrong, or the unit under test is not actually pinned"
    );
}
