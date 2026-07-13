#![forbid(unsafe_code)]
//! lucene-analysis: see /PLAN.md for scope.
//!
//! A minimal, real analyzer chain mirroring Lucene's
//! `Analyzer`/`Tokenizer`/`TokenFilter` pipeline: a simplified word-boundary
//! tokenizer (not full UAX#29 Unicode text segmentation -- see the module
//! docs on [`tokenize`]), plus `LowerCaseFilter` and `StopFilter`.
//!
//! This crate sits below both `lucene-index` and `lucene-search` in the
//! workspace's downward dependency graph (it depends on nothing else in the
//! workspace), so either can depend on it without creating a cycle.

use std::collections::HashSet;

/// One analyzed token: term text plus the attributes real Lucene's
/// `CharTermAttribute`/`OffsetAttribute`/`PositionIncrementAttribute` carry.
///
/// `start_offset`/`end_offset` are character offsets into the original text
/// (matching this port's existing character-offset convention, e.g.
/// `TermOffsetSpan`). `position_increment` is the gap from the *previous
/// surviving* token's position (1 for immediately-adjacent tokens; see
/// [`StopFilter`] for how removed tokens affect this).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub term: String,
    pub start_offset: i32,
    pub end_offset: i32,
    pub position_increment: i32,
}

/// A simplified, real word-boundary tokenizer: splits on whitespace and
/// punctuation boundaries, keeping maximal runs of alphanumeric characters
/// (Unicode alphanumeric, via `char::is_alphanumeric`) as terms.
///
/// This mirrors the *core algorithm* of real Lucene's `StandardTokenizer`/
/// `WhitespaceTokenizer` -- split on non-alphanumeric boundaries -- but is
/// **not** a port of full UAX#29 Unicode Text Segmentation (which handles
/// things like combining marks, locale-specific word breaking, and complex
/// script segmentation). That's substantial, legitimately out-of-scope NLP
/// machinery; see `docs/parity.md` for the explicit scope note.
///
/// Every token gets `position_increment == 1` (tokenizers never skip
/// positions -- that only happens in filters, e.g. [`StopFilter`]).
pub fn tokenize(text: &str) -> Vec<Token> {
    let mut tokens = Vec::new();
    let mut chars = text.char_indices().peekable();
    while let Some(&(start, ch)) = chars.peek() {
        if !ch.is_alphanumeric() {
            chars.next();
            continue;
        }
        let mut end = start;
        let mut end_char_len = 0;
        while let Some(&(idx, c)) = chars.peek() {
            if !c.is_alphanumeric() {
                break;
            }
            end = idx;
            end_char_len = c.len_utf8();
            chars.next();
        }
        let end_offset = end + end_char_len;
        tokens.push(Token {
            term: text[start..end_offset].to_string(),
            start_offset: start as i32,
            end_offset: end_offset as i32,
            position_increment: 1,
        });
    }
    tokens
}

/// Real Lucene's `LowerCaseFilter`: lowercases each token's term text,
/// leaving offsets and position increments untouched.
pub struct LowerCaseFilter;

impl LowerCaseFilter {
    pub fn apply(tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.term = t.term.to_lowercase();
                t
            })
            .collect()
    }
}

/// Real Lucene's `StopFilter`: removes tokens whose term matches a
/// caller-supplied stopword set.
///
/// Position-increment preservation (real Lucene semantics, not "just drop
/// the removed token"): a removed stopword's own `position_increment` is
/// *not* discarded -- it is added onto the position increment of the next
/// surviving token, so the position gap it would have occupied is preserved.
/// Consecutive removed stopwords accumulate onto whichever token survives
/// next. If the text is nothing but stopwords, the output is empty (no
/// increment is left dangling anywhere since there's no surviving token to
/// carry it -- matching real Lucene, which simply produces zero tokens here
/// too).
pub struct StopFilter;

impl StopFilter {
    pub fn apply(tokens: Vec<Token>, stopwords: &HashSet<String>) -> Vec<Token> {
        let mut out = Vec::new();
        let mut pending_increment = 0;
        for mut t in tokens {
            if stopwords.contains(&t.term) {
                pending_increment += t.position_increment;
                continue;
            }
            t.position_increment += pending_increment;
            pending_increment = 0;
            out.push(t);
        }
        out
    }
}

/// An analyzer composing a tokenizer with a configurable filter chain.
///
/// At minimum applies [`LowerCaseFilter`]; optionally applies [`StopFilter`]
/// when stopwords are configured. Additional real-Lucene filters (stemming,
/// synonyms, ASCII-folding, etc.) are out of scope for this MVP -- see
/// `docs/parity.md`.
pub struct Analyzer {
    stopwords: Option<HashSet<String>>,
}

impl Analyzer {
    /// A "standard"-style analyzer: word-boundary tokenizer + lowercase +
    /// optional stopword removal, mirroring real Lucene's `StandardAnalyzer`
    /// (`StandardTokenizer` + `LowerCaseFilter` + `StopFilter`) at this
    /// crate's documented scope.
    pub fn standard(stopwords: Option<&HashSet<String>>) -> Self {
        Analyzer {
            stopwords: stopwords.cloned(),
        }
    }

    pub fn analyze(&self, text: &str) -> Vec<Token> {
        let tokens = tokenize(text);
        let tokens = LowerCaseFilter::apply(tokens);
        match &self.stopwords {
            Some(stopwords) => StopFilter::apply(tokens, stopwords),
            None => tokens,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(term: &str, start: i32, end: i32, pos_inc: i32) -> Token {
        Token {
            term: term.to_string(),
            start_offset: start,
            end_offset: end,
            position_increment: pos_inc,
        }
    }

    #[test]
    fn tokenize_multi_word_sentence() {
        let tokens = tokenize("The quick, brown fox!");
        assert_eq!(
            tokens,
            vec![
                tok("The", 0, 3, 1),
                tok("quick", 4, 9, 1),
                tok("brown", 11, 16, 1),
                tok("fox", 17, 20, 1),
            ]
        );
    }

    #[test]
    fn tokenize_empty_text() {
        assert_eq!(tokenize(""), vec![]);
    }

    #[test]
    fn tokenize_only_punctuation() {
        assert_eq!(tokenize("... !!! ,,,"), vec![]);
    }

    #[test]
    fn tokenize_alphanumeric_run_kept_together() {
        let tokens = tokenize("abc123 456def");
        assert_eq!(
            tokens,
            vec![tok("abc123", 0, 6, 1), tok("456def", 7, 13, 1),]
        );
    }

    #[test]
    fn lowercase_filter_changes_case_not_offsets_or_positions() {
        let tokens = vec![tok("THE", 0, 3, 1), tok("Quick", 4, 9, 2)];
        let out = LowerCaseFilter::apply(tokens);
        assert_eq!(out, vec![tok("the", 0, 3, 1), tok("quick", 4, 9, 2),]);
    }

    #[test]
    fn stop_filter_bumps_next_position_increment() {
        // "the quick fox" with "the" as a stopword: "quick" should get
        // position_increment == 2 (1 from itself + 1 carried over from the
        // removed "the"), not 1.
        let tokens = tokenize("the quick fox");
        let tokens = LowerCaseFilter::apply(tokens);
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let out = StopFilter::apply(tokens, &stopwords);
        assert_eq!(out, vec![tok("quick", 4, 9, 2), tok("fox", 10, 13, 1),]);
    }

    #[test]
    fn stop_filter_stopword_at_start() {
        let tokens = tokenize("the fox");
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let out = StopFilter::apply(tokens, &stopwords);
        assert_eq!(out, vec![tok("fox", 4, 7, 2)]);
    }

    #[test]
    fn stop_filter_stopword_at_end() {
        let tokens = tokenize("fox the");
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let out = StopFilter::apply(tokens, &stopwords);
        assert_eq!(out, vec![tok("fox", 0, 3, 1)]);
    }

    #[test]
    fn stop_filter_consecutive_stopwords_accumulate() {
        // "a the of fox" with "a"/"the"/"of" all stopwords: fox should carry
        // increment 1 (its own) + 3 removed = 4.
        let tokens = tokenize("a the of fox");
        let stopwords: HashSet<String> = ["a".to_string(), "the".to_string(), "of".to_string()]
            .into_iter()
            .collect();
        let out = StopFilter::apply(tokens, &stopwords);
        assert_eq!(out, vec![tok("fox", 9, 12, 4)]);
    }

    #[test]
    fn stop_filter_all_stopwords_yields_empty_not_panic() {
        let tokens = tokenize("the a of");
        let stopwords: HashSet<String> = ["the".to_string(), "a".to_string(), "of".to_string()]
            .into_iter()
            .collect();
        let out = StopFilter::apply(tokens, &stopwords);
        assert_eq!(out, vec![]);
    }

    #[test]
    fn analyzer_standard_full_pipeline() {
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let analyzer = Analyzer::standard(Some(&stopwords));
        let out = analyzer.analyze("The Quick, Brown FOX!");
        assert_eq!(
            out,
            vec![
                tok("quick", 4, 9, 2),
                tok("brown", 11, 16, 1),
                tok("fox", 17, 20, 1),
            ]
        );
    }

    #[test]
    fn analyzer_standard_no_stopwords() {
        let analyzer = Analyzer::standard(None);
        let out = analyzer.analyze("Hello World");
        assert_eq!(out, vec![tok("hello", 0, 5, 1), tok("world", 6, 11, 1)]);
    }
}
