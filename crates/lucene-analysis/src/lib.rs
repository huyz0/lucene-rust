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
/// **`start_offset`/`end_offset` are UTF-8 BYTE offsets into the original
/// text**, not character offsets -- this is a real, previously-mislabeled
/// discrepancy (surfaced by task #64's cross-engine testing against
/// non-ASCII text; real Lucene's own `OffsetAttribute` reports UTF-16
/// code-unit offsets, and this port's other char-offset-based APIs, e.g.
/// [`crate`]-external `TermOffsetSpan`/the highlighter, assume Unicode-scalar
/// (char) counts). [`tokenize`]'s own implementation builds these via
/// `char_indices()`/`len_utf8()`, which are byte positions, not char
/// positions -- confirmed to coincide with char offsets only for pure-ASCII
/// text, where every char is exactly one byte. **No live code path is
/// broken by this today** (nothing yet wires this crate's tokenizer output
/// into the char-offset-assuming consumers -- `lucene-index`'s
/// `indexing_chain` module currently just passes these offsets through
/// opaquely, with no persistence path to a codec yet), but this is a real
/// latent bug waiting to surface: once a future task wires tokenized output
/// into a real writer/highlighter pipeline, non-ASCII field text will
/// silently produce corrupted offset spans unless this unit mismatch is
/// resolved first (either by converting to char offsets here, or by every
/// downstream consumer explicitly treating these as byte offsets).
/// `position_increment` is the gap from the *previous surviving* token's
/// position (1 for immediately-adjacent tokens; see [`StopFilter`] for how
/// removed tokens affect this).
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

/// Real Lucene's `org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter`:
/// folds accented/diacritic Latin characters to their closest plain-ASCII
/// equivalent, leaving offsets and position increments untouched.
///
/// **Scope, stated explicitly (this is a deliberately-scoped subset of real
/// Lucene's much larger table, not the full port)**:
///
/// - **Covered**: the entire Latin-1 Supplement letter block (U+00C0-U+00DE
///   uppercase, U+00E0-U+00FE lowercase, i.e. À-Þ / à-þ, skipping U+00D7
///   `×` and U+00F7 `÷` which are math symbols, not letters), plus a
///   documented subset of Latin Extended-A covering the most common
///   Central/European diacritics: Ą/ą, Ć/ć, Ę/ę, Ł/ł, Ń/ń, Ś/ś, Ź/ź, Ż/ż
///   (Polish), Š/š, Č/č, Ž/ž, Ď/ď, Ť/ť, Ň/ň (Czech/Slovak/Baltic caron
///   forms). `Æ`/`æ` and `Œ`/`œ` fold to **two** ASCII characters (`AE`/`ae`
///   and `OE`/`oe` respectively) -- real Lucene's actual multi-char folding,
///   not an invented shortcut -- and `ß` folds to `ss` (real Lucene's actual
///   special case; it is emphatically not "b"/"beta").
/// - **Deferred, real follow-on work**: the rest of real Lucene's table --
///   the remainder of Latin Extended-A/B, Latin Extended Additional
///   (precomposed Vietnamese, etc.), and non-Latin scripts real
///   `ASCIIFoldingFilter` also folds (e.g. fullwidth Latin forms, some
///   Cyrillic/Greek-adjacent visual analogs). A character outside this
///   filter's documented table passes through **unchanged** (never dropped,
///   never a panic) -- see `docs/parity.md` for the itemized scope.
///
/// **Offsets are never adjusted for folding-driven length changes**: folding
/// `æ` -> `"ae"` grows a token's character count, but `start_offset`/
/// `end_offset` still refer to the *original* source text span -- this
/// matches real Lucene's `ASCIIFoldingFilter`, which does not touch
/// `OffsetAttribute` at all.
pub struct AsciiFoldingFilter;

impl AsciiFoldingFilter {
    /// Returns the ASCII fold for `c`, or `None` if `c` is outside this
    /// filter's documented table (caller should keep the original char).
    fn fold_char(c: char) -> Option<&'static str> {
        match c {
            // Latin-1 Supplement, uppercase letters (U+00C0-U+00DE, skipping
            // U+00D7 '×').
            'À' | 'Á' | 'Â' | 'Ã' | 'Ä' | 'Å' => Some("A"),
            'Æ' => Some("AE"),
            'Ç' => Some("C"),
            'È' | 'É' | 'Ê' | 'Ë' => Some("E"),
            'Ì' | 'Í' | 'Î' | 'Ï' => Some("I"),
            'Ð' => Some("D"),
            'Ñ' => Some("N"),
            'Ò' | 'Ó' | 'Ô' | 'Õ' | 'Ö' | 'Ø' => Some("O"),
            'Ù' | 'Ú' | 'Û' | 'Ü' => Some("U"),
            'Ý' => Some("Y"),
            'Þ' => Some("TH"),
            // Latin-1 Supplement, lowercase letters (U+00DF-U+00FE, skipping
            // U+00F7 '÷').
            'ß' => Some("ss"),
            'à' | 'á' | 'â' | 'ã' | 'ä' | 'å' => Some("a"),
            'æ' => Some("ae"),
            'ç' => Some("c"),
            'è' | 'é' | 'ê' | 'ë' => Some("e"),
            'ì' | 'í' | 'î' | 'ï' => Some("i"),
            'ð' => Some("d"),
            'ñ' => Some("n"),
            'ò' | 'ó' | 'ô' | 'õ' | 'ö' | 'ø' => Some("o"),
            'ù' | 'ú' | 'û' | 'ü' => Some("u"),
            'ý' | 'ÿ' => Some("y"),
            'þ' => Some("th"),
            // Latin Extended-A: common Central/Eastern European diacritics.
            'Ą' => Some("A"),
            'ą' => Some("a"),
            'Ć' => Some("C"),
            'ć' => Some("c"),
            'Č' => Some("C"),
            'č' => Some("c"),
            'Ď' => Some("D"),
            'ď' => Some("d"),
            'Ę' => Some("E"),
            'ę' => Some("e"),
            'Ł' => Some("L"),
            'ł' => Some("l"),
            'Ń' => Some("N"),
            'ń' => Some("n"),
            'Ň' => Some("N"),
            'ň' => Some("n"),
            'Œ' => Some("OE"),
            'œ' => Some("oe"),
            'Ś' => Some("S"),
            'ś' => Some("s"),
            'Š' => Some("S"),
            'š' => Some("s"),
            'Ť' => Some("T"),
            'ť' => Some("t"),
            'Ź' => Some("Z"),
            'ź' => Some("z"),
            'Ž' => Some("Z"),
            'ž' => Some("z"),
            'Ż' => Some("Z"),
            'ż' => Some("z"),
            _ => None,
        }
    }

    /// Folds each token's `term` character-by-character per the documented
    /// table above, leaving `start_offset`/`end_offset`/`position_increment`
    /// completely untouched even when folding changes the term's character
    /// length (e.g. a ligature growing to two ASCII characters).
    pub fn apply(tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                if t.term.is_ascii() {
                    return t;
                }
                let mut folded = String::with_capacity(t.term.len());
                for c in t.term.chars() {
                    match Self::fold_char(c) {
                        Some(replacement) => folded.push_str(replacement),
                        None => folded.push(c),
                    }
                }
                t.term = folded;
                t
            })
            .collect()
    }
}

/// An analyzer composing a tokenizer with a configurable filter chain.
///
/// At minimum applies [`LowerCaseFilter`]; optionally applies [`StopFilter`]
/// when stopwords are configured, and optionally applies
/// [`AsciiFoldingFilter`] when enabled via [`Analyzer::with_ascii_folding`].
/// Additional real-Lucene filters (stemming, synonyms, etc.) are out of
/// scope for this MVP -- see `docs/parity.md`.
pub struct Analyzer {
    stopwords: Option<HashSet<String>>,
    ascii_folding: bool,
}

impl Analyzer {
    /// A "standard"-style analyzer: word-boundary tokenizer + lowercase +
    /// optional stopword removal, mirroring real Lucene's `StandardAnalyzer`
    /// (`StandardTokenizer` + `LowerCaseFilter` + `StopFilter`) at this
    /// crate's documented scope. ASCII-folding is off by default -- use
    /// [`Analyzer::with_ascii_folding`] to enable it -- so every existing
    /// caller's behavior is unchanged.
    pub fn standard(stopwords: Option<&HashSet<String>>) -> Self {
        Analyzer {
            stopwords: stopwords.cloned(),
            ascii_folding: false,
        }
    }

    /// Enables [`AsciiFoldingFilter`] in this analyzer's chain. Filter
    /// order: tokenize -> **fold** -> lowercase -> stopwords. Folding runs
    /// before lowercasing so that an uppercase accented letter (e.g. `É`)
    /// folds straight to its ASCII letter (`E`) and then gets lowercased
    /// along with every other token in the same pass, rather than needing
    /// its own case-conversion step; this also means stopword matching (which
    /// happens last, against already-lowercased terms) sees the fully
    /// folded-and-lowercased form regardless of the input's original
    /// diacritics/casing.
    pub fn with_ascii_folding(mut self) -> Self {
        self.ascii_folding = true;
        self
    }

    pub fn analyze(&self, text: &str) -> Vec<Token> {
        let tokens = tokenize(text);
        let tokens = if self.ascii_folding {
            AsciiFoldingFilter::apply(tokens)
        } else {
            tokens
        };
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

    #[test]
    fn ascii_folding_latin1_spot_checks() {
        let tokens = vec![
            tok("café", 0, 4, 1),
            tok("naïve", 0, 5, 1),
            tok("Müller", 0, 6, 1),
            tok("ñ", 0, 1, 1),
        ];
        let out = AsciiFoldingFilter::apply(tokens);
        assert_eq!(
            out,
            vec![
                tok("cafe", 0, 4, 1),
                tok("naive", 0, 5, 1),
                tok("Muller", 0, 6, 1),
                tok("n", 0, 1, 1),
            ]
        );
    }

    #[test]
    fn ascii_folding_covers_every_documented_table_entry() {
        // Exhaustively spot-checks every char->replacement mapping this
        // filter documents, not just a handful -- so every match arm in
        // `fold_char` is actually exercised.
        let cases: &[(char, &str)] = &[
            ('À', "A"),
            ('Á', "A"),
            ('Â', "A"),
            ('Ã', "A"),
            ('Ä', "A"),
            ('Å', "A"),
            ('Æ', "AE"),
            ('Ç', "C"),
            ('È', "E"),
            ('É', "E"),
            ('Ê', "E"),
            ('Ë', "E"),
            ('Ì', "I"),
            ('Í', "I"),
            ('Î', "I"),
            ('Ï', "I"),
            ('Ð', "D"),
            ('Ñ', "N"),
            ('Ò', "O"),
            ('Ó', "O"),
            ('Ô', "O"),
            ('Õ', "O"),
            ('Ö', "O"),
            ('Ø', "O"),
            ('Ù', "U"),
            ('Ú', "U"),
            ('Û', "U"),
            ('Ü', "U"),
            ('Ý', "Y"),
            ('Þ', "TH"),
            ('ß', "ss"),
            ('à', "a"),
            ('á', "a"),
            ('â', "a"),
            ('ã', "a"),
            ('ä', "a"),
            ('å', "a"),
            ('æ', "ae"),
            ('ç', "c"),
            ('è', "e"),
            ('é', "e"),
            ('ê', "e"),
            ('ë', "e"),
            ('ì', "i"),
            ('í', "i"),
            ('î', "i"),
            ('ï', "i"),
            ('ð', "d"),
            ('ñ', "n"),
            ('ò', "o"),
            ('ó', "o"),
            ('ô', "o"),
            ('õ', "o"),
            ('ö', "o"),
            ('ø', "o"),
            ('ù', "u"),
            ('ú', "u"),
            ('û', "u"),
            ('ü', "u"),
            ('ý', "y"),
            ('ÿ', "y"),
            ('þ', "th"),
            ('Ą', "A"),
            ('ą', "a"),
            ('Ć', "C"),
            ('ć', "c"),
            ('Č', "C"),
            ('č', "c"),
            ('Ď', "D"),
            ('ď', "d"),
            ('Ę', "E"),
            ('ę', "e"),
            ('Ł', "L"),
            ('ł', "l"),
            ('Ń', "N"),
            ('ń', "n"),
            ('Ň', "N"),
            ('ň', "n"),
            ('Œ', "OE"),
            ('œ', "oe"),
            ('Ś', "S"),
            ('ś', "s"),
            ('Š', "S"),
            ('š', "s"),
            ('Ť', "T"),
            ('ť', "t"),
            ('Ź', "Z"),
            ('ź', "z"),
            ('Ž', "Z"),
            ('ž', "z"),
            ('Ż', "Z"),
            ('ż', "z"),
        ];
        for (c, expected) in cases {
            let tokens = vec![tok(&c.to_string(), 0, 1, 1)];
            let out = AsciiFoldingFilter::apply(tokens);
            assert_eq!(
                out,
                vec![tok(expected, 0, 1, 1)],
                "folding {c:?} should yield {expected:?}"
            );
        }
    }

    #[test]
    fn ascii_folding_eszett_folds_to_ss() {
        let tokens = vec![tok("straße", 0, 6, 1)];
        let out = AsciiFoldingFilter::apply(tokens);
        assert_eq!(out, vec![tok("strasse", 0, 6, 1)]);
    }

    #[test]
    fn ascii_folding_ligature_grows_term_but_not_offsets() {
        // "æ" (1 char) -> "ae" (2 chars): term grows, offsets untouched.
        let tokens = vec![tok("æther", 0, 5, 1), tok("cœur", 10, 14, 1)];
        let out = AsciiFoldingFilter::apply(tokens);
        assert_eq!(out, vec![tok("aether", 0, 5, 1), tok("coeur", 10, 14, 1),]);
        assert!(out[0].term.chars().count() > 5);
    }

    #[test]
    fn ascii_folding_plain_ascii_passes_through_unmodified() {
        let tokens = vec![tok("hello", 0, 5, 1)];
        let out = AsciiFoldingFilter::apply(tokens.clone());
        assert_eq!(out, tokens);
    }

    #[test]
    fn ascii_folding_mixed_diacritic_and_ascii_in_one_token() {
        let tokens = vec![tok("café123", 0, 7, 1)];
        let out = AsciiFoldingFilter::apply(tokens);
        assert_eq!(out, vec![tok("cafe123", 0, 7, 1)]);
    }

    #[test]
    fn ascii_folding_char_outside_table_passes_through_unchanged() {
        // A Cyrillic character isn't in this filter's documented table --
        // it must survive untouched, not be dropped or panic.
        let tokens = vec![tok("привет", 0, 6, 1)];
        let out = AsciiFoldingFilter::apply(tokens.clone());
        assert_eq!(out, tokens);
    }

    #[test]
    fn ascii_folding_composed_with_lowercase_folds_then_lowercases() {
        // Analyzer::with_ascii_folding applies folding before lowercasing:
        // "É" -> "E" -> "e".
        let analyzer = Analyzer::standard(None).with_ascii_folding();
        let out = analyzer.analyze("Café Naïve ÉCOLE");
        assert_eq!(
            out,
            vec![
                tok("cafe", 0, 5, 1),
                tok("naive", 6, 12, 1),
                tok("ecole", 13, 19, 1),
            ]
        );
    }

    #[test]
    fn analyzer_default_has_no_folding_backward_compatible() {
        // Default Analyzer::standard (no with_ascii_folding call) leaves
        // diacritics as-is, only lowercasing -- unchanged behavior for every
        // existing caller (query_parser.rs, indexing_chain.rs).
        let analyzer = Analyzer::standard(None);
        let out = analyzer.analyze("Café");
        assert_eq!(out, vec![tok("café", 0, 5, 1)]);
    }
}
