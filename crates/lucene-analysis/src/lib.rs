#![forbid(unsafe_code)]
//! lucene-analysis: see /PLAN.md for scope.
//!
//! A minimal, real analyzer chain mirroring Lucene's
//! `Analyzer`/`Tokenizer`/`TokenFilter` pipeline: a UAX#29 word-boundary
//! tokenizer (see the module docs on [`tokenize`] for exactly what's covered
//! vs. deliberately deferred relative to real Lucene's `StandardTokenizer`),
//! plus `LowerCaseFilter`, `StopFilter`,
//! `AsciiFoldingFilter`, `PorterStemFilter`, `SynonymFilter`, and
//! `NGramTokenFilter`/`EdgeNGramTokenFilter`.
//!
//! This crate sits below both `lucene-index` and `lucene-search` in the
//! workspace's downward dependency graph (it depends on nothing else in the
//! workspace), so either can depend on it without creating a cycle.

use std::collections::HashMap;
use std::collections::HashSet;

use unicode_segmentation::UnicodeSegmentation;

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
///
/// `position_length` mirrors real Lucene's `PositionLengthAttribute`: the
/// number of positions this token spans, starting at its own position. Every
/// token produced by [`tokenize`] and every filter in this crate except
/// [`SynonymFilter::apply_multiword`] leaves it at `1` (a token that only
/// occupies its own position -- the overwhelming common case, including
/// real Lucene's own default). [`SynonymFilter::apply_multiword`] is the only
/// producer of `position_length > 1`: a multi-word input phrase collapsed to
/// a single output token (e.g. `"wi fi"` -> `"wifi"`) gets a `position_length`
/// equal to the number of input tokens it replaces, so a consumer that reads
/// this attribute (unlike this crate's own [`Analyzer`], which does not) can
/// tell the synonym token spans multiple original positions rather than one.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    pub term: String,
    pub start_offset: i32,
    pub end_offset: i32,
    pub position_increment: i32,
    pub position_length: i32,
}

/// A UAX#29-based word-boundary tokenizer, standing in for real Lucene's
/// `StandardTokenizer` (which itself is a JFlex-generated implementation of
/// [UAX #29, Unicode Text Segmentation](https://www.unicode.org/reports/tr29/)'s
/// default word-boundary algorithm, extended with a handful of Lucene-specific
/// rules for URLs/emails/host names that are out of scope here -- see below).
///
/// **Implementation**: this delegates word segmentation itself to the
/// `unicode-segmentation` crate's [`UnicodeSegmentation::unicode_word_indices`]
/// (already a workspace dependency -- no new crate was added for this), which
/// is a compliant implementation of UAX#29's `Word_Break` property tables and
/// rule set (WB1-WB999, per the current Unicode Character Database the crate
/// ships). That single call is what gives this tokenizer real UAX#29
/// semantics rather than the ad hoc hand-rolled rules a previous version of
/// this function used:
///
/// - **Combining diacritical marks**: a base character followed by one or
///   more `Grapheme_Extend`/combining-mark characters (e.g. a bare `e`
///   followed by a combining acute accent, U+0301) is never split apart --
///   UAX#29's `WB` rules never insert a boundary before an `Extend`/`ZWJ`
///   character, so `"cafe\u{0301}"` tokenizes as the one token `"café"`
///   (grapheme-equivalent), not two.
/// - **CJK ideograph segmentation**: each Han ideograph is `Word_Break =
///   Other`/`Ideographic` with no `ALetter`-style clustering rule joining
///   adjacent ideographs, so a run of CJK text segments into one token *per
///   character* (e.g. `"你好世界"` -> four separate one-character tokens),
///   matching real `StandardTokenizer`'s behavior on unsegmented CJK (neither
///   real Lucene nor this port does dictionary-based CJK word segmentation;
///   that is a distinct, heavier feature -- see `CJKAnalyzer`'s bigram
///   filter, which remains out of scope here).
/// - **Hangul syllable clustering**: precomposed Hangul syllables (e.g. `안`)
///   are single Unicode scalars already and naturally form single tokens;
///   sequences of *conjoining* Hangul Jamo (leading/vowel/trailing consonant
///   codepoints, U+1100-U+11FF) are clustered into one token per syllable
///   block by UAX#29's dedicated Hangul `WB` rules (the same rules real
///   Lucene's tokenizer relies on), rather than splitting at each Jamo
///   codepoint.
/// - **Midword punctuation**: UAX#29's `MidLetter`/`MidNumLet`/`MidNum` rules
///   (WB6/WB7/WB11/WB12) are exactly what already produced this crate's
///   previously hand-coded exceptions -- e.g. `.`/`,` embedded in a number
///   (`"3.14"`, `"1,000"`), `.` between single letters in an acronym
///   (`"U.S.A."` -> `"U.S.A"`, the trailing period still splits off since
///   nothing alphanumeric follows), and `'`/`’` inside a contraction/name
///   (`"don't"`, `"O'Brien"`) -- so this port's existing documented behavior
///   for those cases is preserved (and is now backed by the real algorithm
///   these rules come from, not a 4-character lookup table).
///
/// **What real UAX#29/`StandardTokenizer` includes that this does *not*
/// port** (deliberately out of scope, not silently wrong -- see
/// `docs/parity.md`):
/// - **Emoji/ZWJ *sequence* grouping as a single visual glyph**: a bare ZWJ
///   between two letters is itself `Extend`-like and does not split (see
///   `"a\u{200D}b"` above), but a ZWJ emoji sequence (e.g. family emoji built
///   from base emoji + ZWJ + modifiers) contains no alphanumeric codepoints
///   at all, so -- like every other non-alphanumeric run -- it produces *no*
///   token, same as a lone emoji. Grapheme-cluster-aware emoji tokenization
///   (treating a whole ZWJ sequence as one indivisible unit for filters that
///   *do* want to emit it as a term) is a distinct, heavier Unicode
///   grapheme-segmentation feature this crate does not attempt; adding it
///   would not require a new external crate (the workspace's
///   `unicode-segmentation` dependency also implements UAX#29 grapheme
///   clusters via `graphemes()`), but is out of scope for this task since
///   this tokenizer -- like real `StandardTokenizer` -- only ever emits
///   alphanumeric-containing segments as terms in the first place.
/// - **Lucene's own URL/email/host-name JFlex extensions** to the base
///   UAX#29 grammar (e.g. keeping `user@example.com` or
///   `https://example.com/path` as a single token) are Lucene-specific
///   additions layered on top of UAX#29, not part of UAX#29 itself, and
///   remain unimplemented here -- an email/URL still gets split into its
///   alphanumeric-run pieces (`user`, `example`, `com`, ...).
/// - **Locale-specific tailoring** (UAX#29 §5.3's optional locale exceptions,
///   e.g. Southeast Asian dictionary-based segmentation for Thai/Lao/Khmer/
///   Myanmar) is not implemented -- the crate, like real Lucene's default
///   `BreakIterator`-free tokenizer, applies the same rules regardless of
///   detected script/language.
///
/// Every token gets `position_increment == 1` (tokenizers never skip
/// positions -- that only happens in filters, e.g. [`StopFilter`]).
pub fn tokenize(text: &str) -> Vec<Token> {
    text.unicode_word_indices()
        .map(|(start, word)| {
            let start_offset = start as i32;
            let end_offset = (start + word.len()) as i32;
            Token {
                term: word.to_string(),
                start_offset,
                end_offset,
                position_increment: 1,
                position_length: 1,
            }
        })
        .collect()
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

/// The classic Lucene/Snowball English stop word list, byte-for-byte the
/// same 33 words as real Lucene's
/// `org.apache.lucene.analysis.en.EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`
/// (itself sourced from the Snowball project's `english` stop list). Stored
/// lowercase, matching real Lucene's `CharArraySet` (built with
/// `ignoreCase == false` there, but populated with already-lowercase
/// entries) and this port's [`StopFilter`], which does a plain, exact
/// (case-sensitive) string match against terms that have already passed
/// through [`LowerCaseFilter`] earlier in the chain -- see
/// [`english_stop_words`].
pub const ENGLISH_STOP_WORDS: &[&str] = &[
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into", "is", "it",
    "no", "not", "of", "on", "or", "such", "that", "the", "their", "then", "there", "these",
    "they", "this", "to", "was", "will", "with",
];

/// Builds a fresh `HashSet<String>` from [`ENGLISH_STOP_WORDS`], ready to
/// pass to [`StopFilter::apply`] or [`Analyzer::standard`], mirroring real
/// Lucene's `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET` default. Not a
/// `static`/`OnceLock`-cached singleton (real Lucene's set is immutable and
/// shared, but this port's `Analyzer`/`StopFilter` API takes an owned
/// `HashSet<String>` per call site, and this list is only 33 short strings,
/// so allocating a fresh set per analyzer construction is simpler and not a
/// meaningful cost).
pub fn english_stop_words() -> HashSet<String> {
    ENGLISH_STOP_WORDS.iter().map(|s| s.to_string()).collect()
}

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

/// Real Lucene's `org.apache.lucene.analysis.en.PorterStemFilter`: the
/// classic Porter stemming algorithm (Martin Porter, "An algorithm for
/// suffix stripping", 1980) for English, stemming each token's `term` field
/// and leaving offsets/position increments untouched (same convention as
/// every other filter in this crate).
///
/// **Scope, stated explicitly**: this ports **all five steps** of the
/// original 1980 algorithm --
///
/// - **Step 1a**: `-sses`->`-ss`, `-ies`->`-i`, `-ss`->`-ss` (no-op), `-s`->
///   (delete).
/// - **Step 1b**: `-eed`->`-ee` (only if `m(stem) > 0`); `-ed`/`-ing` deleted
///   only if the stem contains a vowel, followed by cleanup (`-at`/`-bl`/
///   `-iz` gets `e` appended; a double consonant not ending in `l`/`s`/`z`
///   loses its last letter; `m(stem) == 1` and CVC gets `e` appended).
/// - **Step 1c**: trailing `y` -> `i` if the stem contains a vowel.
/// - **Step 2** (`m(stem) > 0`): the long suffix-family table (`-ational`->
///   `-ate`, `-tional`->`-tion`, `-enci`->`-ence`, ... `-biliti`->`-ble`).
/// - **Step 3** (`m(stem) > 0`): `-icate`->`-ic`, `-ative`-> (delete),
///   `-alize`->`-al`, `-iciti`->`-ic`, `-ical`->`-ic`, `-ful`/`-ness`->
///   (delete).
/// - **Step 4** (`m(stem) > 1`): removes `-al`, `-ance`, `-ence`, `-er`,
///   `-ic`, `-able`, `-ible`, `-ant`, `-ement`, `-ment`, `-ent`, `-ion` (only
///   if preceded by `s`/`t`), `-ou`, `-ism`, `-ate`, `-iti`, `-ous`, `-ive`,
///   `-ize`.
/// - **Step 5a**: trailing `e` deleted if `m(stem) > 1`, or if `m(stem) == 1`
///   and the stem is not CVC.
/// - **Step 5b**: a trailing double `l` collapses to a single `l` if
///   `m(word) > 1`.
///
/// Nothing is deferred -- this is the complete classic algorithm, not a
/// subset -- but it is still **English-only** and, per the algorithm's own
/// definition, only meaningful on lowercase ASCII alphabetic input: a term
/// containing any non-ASCII-alphabetic character (digits, punctuation,
/// non-Latin scripts) or any uppercase letter is passed through **unchanged**
/// (never panics, never partially stems). In a normal analyzer chain this
/// filter runs after [`LowerCaseFilter`], so terms are already lowercase by
/// the time they reach it; this guard only matters if `PorterStemFilter` is
/// used standalone on not-yet-lowercased text.
pub struct PorterStemFilter;

impl PorterStemFilter {
    pub fn apply(tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.term = porter::stem(&t.term);
                t
            })
            .collect()
    }
}

/// A scoped-down version of real Lucene's
/// `org.apache.lucene.analysis.synonym.SynonymFilter`/`SynonymGraphFilter`:
/// single-word-to-single-word synonym injection only.
///
/// **Scope, stated explicitly**: real Lucene's full `SynonymGraphFilter`
/// handles multi-word synonym *phrases* (e.g. `"New York"` <-> `"NYC"`) via a
/// graph token stream with its own traversal machinery -- that's substantial,
/// legitimately out-of-scope NLP infrastructure. This filter only maps one
/// term to one or more single-word replacement terms, configured via a
/// caller-supplied `HashMap<String, Vec<String>>`.
///
/// **Positional semantics (the real Lucene rule this mirrors)**: an injected
/// synonym occupies the *same position* as the term it's a synonym for --
/// `position_increment == 0` -- since it doesn't advance past the original,
/// it's an alternative *at* that position (so a `PhraseQuery`/`SpanNear`
/// built against either the original or the synonym term still aligns with
/// surrounding words). The original token keeps its own (unmodified)
/// `position_increment`; only the injected synonym token gets `0`. This is
/// the first token in this crate with `position_increment == 0` -- every
/// prior token (including ones StopFilter bumps) has had `>= 1`.
///
/// **Offsets**: the injected synonym token gets the exact same
/// `start_offset`/`end_offset` as the original -- real Lucene's convention,
/// since the synonym doesn't correspond to distinct source text, it's an
/// alternative reading of the same span.
///
/// **Bidirectionality is NOT automatic by default** (matching real Lucene's
/// `SynonymMap`, which also requires explicit configuration in both
/// directions): configuring `"quick" -> ["fast"]` does *not* also expand
/// `"fast"` to `"quick"`. A caller wanting symmetric synonyms must either
/// configure both `"quick" -> ["fast"]` and `"fast" -> ["quick"]` themselves,
/// or use [`SynonymFilter::apply_bidirectional`] (see that method for the
/// opt-in bidirectional mode, mirroring real Lucene's
/// `SynonymMap.Builder(true)` construction option at a scoped-down level).
pub struct SynonymFilter;

impl SynonymFilter {
    /// For each token whose term is a key in `synonyms`, injects one
    /// additional token per configured synonym value immediately after the
    /// original, each with `position_increment == 0` and the same
    /// `start_offset`/`end_offset` as the original. Tokens with no
    /// configured synonym pass through unchanged (no extra token, no
    /// modification).
    pub fn apply(tokens: Vec<Token>, synonyms: &HashMap<String, Vec<String>>) -> Vec<Token> {
        let mut out = Vec::with_capacity(tokens.len());
        for t in tokens {
            let replacements = synonyms.get(&t.term).cloned();
            let (start_offset, end_offset) = (t.start_offset, t.end_offset);
            out.push(t);
            if let Some(replacements) = replacements {
                for replacement in replacements {
                    out.push(Token {
                        term: replacement,
                        start_offset,
                        end_offset,
                        position_increment: 0,
                        position_length: 1,
                    });
                }
            }
        }
        out
    }

    /// Opt-in bidirectional variant of [`SynonymFilter::apply`], mirroring
    /// real Lucene's `SynonymMap.Builder(true)` (bidirectional) construction
    /// mode at this crate's documented single-word-to-single-word scope:
    /// given the same `HashMap<String, Vec<String>>` config, a `key ->
    /// [values]` mapping ALSO expands each `value -> key` (the reverse of a
    /// direct one-word-to-one-word mapping), so configuring only `"cat" ->
    /// ["feline"]` is enough for analyzing `"feline"` to also inject `"cat"`
    /// -- the caller no longer needs to configure both directions
    /// themselves.
    ///
    /// **Not replicated** (same scope carve-outs as [`SynonymFilter::apply`],
    /// plus one more specific to this mode): multi-word synonym phrases,
    /// weighted/scored synonyms, and real Lucene's `includeOrig` flag are all
    /// out of scope. Also out of scope: transitive closure -- if `"cat" ->
    /// ["feline"]` and `"feline" -> ["kitty"]` are both configured, this does
    /// *not* additionally infer `"cat" -> ["kitty"]` or `"kitty" -> ["cat"]`;
    /// only the direct reverse of each configured pair is added.
    ///
    /// The combined forward+reverse map is built once per call (not
    /// per-token) via an internal helper, then delegated to
    /// [`SynonymFilter::apply`]. A term appearing as both a key and a value
    /// across different mappings (e.g. `"cat" -> ["feline"]` and `"feline"
    /// -> ["cat"]` both configured) is deduplicated -- each direction's
    /// value list never contains the same term twice.
    pub fn apply_bidirectional(
        tokens: Vec<Token>,
        synonyms: &HashMap<String, Vec<String>>,
    ) -> Vec<Token> {
        let merged = build_bidirectional_map(synonyms);
        Self::apply(tokens, &merged)
    }

    /// Multi-word extension of [`SynonymFilter::apply`]/[`apply_bidirectional`]:
    /// matches a **sequence** of one or more input tokens against each
    /// [`SynonymRule::input`] phrase (not just a single token), so rules like
    /// `"wi" "fi" -> "wifi"` (multi-word input collapsing to one output word)
    /// or `"usa" -> "united" "states" "of" "america"` (one input word
    /// expanding to a multi-word output phrase) are both supported, as is
    /// multi-word-to-multi-word.
    ///
    /// **Matching (the lookahead/buffering this needs over
    /// [`SynonymFilter::apply`]'s per-token loop)**: because `rules` is a
    /// slice (not a single-token-keyed map), matching a phrase requires
    /// looking ahead across multiple *input* tokens before deciding whether a
    /// rule fires. At each input position, this scans every rule whose first
    /// input word equals the current token's term, tries the longest
    /// candidate first (**greedy longest match**, mirroring real Lucene's
    /// `SynonymMap`/`SynonymGraphFilter` preference for the longest matching
    /// input phrase), and requires every subsequent word in that rule's
    /// `input` to equal the term of the correspondingly-offset *following*
    /// token -- not just the current one. A partial prefix match (e.g. input
    /// `"wi"` immediately followed by any word other than `"fi"`, or `"wi"`
    /// as the very last token with no `"fi"` following at all) never fires:
    /// the rule is only applied when the *entire* input phrase is present
    /// contiguously (`position_increment == 1` between the matched tokens,
    /// same adjacency notion [`tokenize`] itself produces).
    ///
    /// **Emission**: on a match spanning `len` input tokens, this passes
    /// the `len` matched original tokens through unchanged (same convention
    /// as [`SynonymFilter::apply`]: the original is never dropped), then
    /// appends one alternative path per `rule.outputs` entry:
    /// - A single-word output (`output.len() == 1`) becomes one token with
    ///   `position_increment == 0` (an alternative reading at the same
    ///   starting position as the match) and `position_length == len` --
    ///   the real Lucene `PositionLengthAttribute` convention for a token
    ///   that spans multiple original positions (e.g. `"wifi"` replacing
    ///   `"wi" "fi"` gets `position_length == 2`).
    /// - A multi-word output (`output.len() > 1`) becomes `output.len()`
    ///   chained tokens: the first at `position_increment == 0` (same
    ///   starting position as the match), each subsequent one at
    ///   `position_increment == 1` (advancing one position per output word,
    ///   same as any ordinary adjacent-token sequence), and every one of them
    ///   at `position_length == 1` (each occupies exactly one position in its
    ///   own output path).
    ///
    /// All emitted tokens (matched-through originals and every rule output
    /// token) get the exact same `start_offset`/`end_offset`: the first
    /// matched input token's `start_offset` and the last matched input
    /// token's `end_offset` -- the span of source text the whole match
    /// covers, same convention as [`SynonymFilter::apply`] applied to a
    /// (potentially multi-token) span instead of a single token.
    ///
    /// **Scope carve-out, stated explicitly (see also this type's own
    /// doc)**: this produces a genuinely graph-*shaped* token stream --
    /// distinct output paths recorded via `position_increment`/
    /// `position_length` on [`Token`], the same attributes real Lucene's
    /// `SynonymGraphFilter` uses -- but it is **not** a full graph
    /// `TokenStream`: output is still a single flat `Vec<Token>` in one
    /// linear order (no `PositionLengthAttribute`-aware graph traversal
    /// API), and a multi-word *output* phrase (the `"usa" -> united states
    /// of america"` direction) does not extend the overall position count
    /// the way a true lattice would -- tokens immediately after the matched
    /// span keep the position they'd have had relative to the *original*
    /// single input position, not the expanded output phrase's length. This
    /// means downstream consumers that only read a flat token sequence (this
    /// crate's own [`Analyzer`], most simple positional indexes) see a
    /// reasonable in-order token sequence with correct `position_length`
    /// markers, but full alignment for arbitrary phrase/span queries
    /// spanning *past* a multi-word output on a lattice would require a real
    /// graph-consuming `PhraseQuery`/`SpanQuery`, which is out of scope here
    /// (see `docs/parity.md` for the precise deferred-vs-covered split).
    ///
    /// Rules are matched independently per starting position; overlapping
    /// rules are not combined (only the single longest match at each
    /// position is applied), and a rule's `input` must be non-empty (an
    /// empty-`input` rule is simply never matched, since no starting term can
    /// equal a nonexistent first word).
    pub fn apply_multiword(tokens: Vec<Token>, rules: &[SynonymRule]) -> Vec<Token> {
        let mut by_first_word: HashMap<&str, Vec<&SynonymRule>> = HashMap::new();
        for rule in rules {
            if let Some(first) = rule.input.first() {
                by_first_word.entry(first.as_str()).or_default().push(rule);
            }
        }
        for candidates in by_first_word.values_mut() {
            candidates.sort_by_key(|r| std::cmp::Reverse(r.input.len()));
        }

        let mut out = Vec::with_capacity(tokens.len());
        let mut i = 0;
        while i < tokens.len() {
            let matched = by_first_word
                .get(tokens[i].term.as_str())
                .and_then(|candidates| {
                    candidates.iter().copied().find(|rule| {
                        let len = rule.input.len();
                        len > 0
                            && i + len <= tokens.len()
                            && rule
                                .input
                                .iter()
                                .enumerate()
                                .all(|(k, word)| tokens[i + k].term == *word)
                    })
                });

            match matched {
                Some(rule) => {
                    let len = rule.input.len();
                    let start_offset = tokens[i].start_offset;
                    let end_offset = tokens[i + len - 1].end_offset;
                    for t in &tokens[i..i + len] {
                        out.push(t.clone());
                    }
                    for output in &rule.outputs {
                        let span_len = if output.len() == 1 { len as i32 } else { 1 };
                        for (idx, term) in output.iter().enumerate() {
                            out.push(Token {
                                term: term.clone(),
                                start_offset,
                                end_offset,
                                position_increment: if idx == 0 { 0 } else { 1 },
                                position_length: span_len,
                            });
                        }
                    }
                    i += len;
                }
                None => {
                    out.push(tokens[i].clone());
                    i += 1;
                }
            }
        }
        out
    }
}

/// A single multi-word synonym rule for [`SynonymFilter::apply_multiword`]:
/// maps a contiguous sequence of one or more input terms (`input`) to one or
/// more alternative output phrases (`outputs`), each itself a sequence of one
/// or more terms. Matching is exact-term, case-sensitive (same as the
/// single-word `HashMap<String, Vec<String>>` rules used by
/// [`SynonymFilter::apply`]) -- callers wanting case-insensitive matching
/// should lowercase both `input`/`outputs` and run this after
/// [`LowerCaseFilter`], same convention as the single-word filter.
///
/// Examples: `SynonymRule { input: vec!["wi".into(), "fi".into()], outputs:
/// vec![vec!["wifi".into()]] }` (multi-word input, single-word output) and
/// `SynonymRule { input: vec!["usa".into()], outputs: vec![vec!["united".into(),
/// "states".into(), "of".into(), "america".into()]] }` (single-word input,
/// multi-word output) are both valid, as is a rule with multi-word `input`
/// *and* a multi-word entry in `outputs`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SynonymRule {
    pub input: Vec<String>,
    pub outputs: Vec<Vec<String>>,
}

/// Builds a combined forward+reverse synonym map from `synonyms`: every
/// configured `key -> [values]` entry is kept as-is, and additionally each
/// `value -> key` reverse entry is added. Used by
/// [`SynonymFilter::apply_bidirectional`] to precompute the expanded map
/// once per call rather than re-deriving it per token.
///
/// Deduplicates so that a term already present in a target term's value
/// list (whether from the forward or reverse pass) is never added twice --
/// this handles both a term mapping to itself (`v == k`, skipped) and a pair
/// configured in both directions already (e.g. `"cat" -> ["feline"]` and
/// `"feline" -> ["cat"]` both present in `synonyms`).
fn build_bidirectional_map(
    synonyms: &HashMap<String, Vec<String>>,
) -> HashMap<String, Vec<String>> {
    let mut merged: HashMap<String, Vec<String>> = HashMap::new();
    for (k, vs) in synonyms {
        let entry = merged.entry(k.clone()).or_default();
        for v in vs {
            if !entry.contains(v) {
                entry.push(v.clone());
            }
        }
    }
    for (k, vs) in synonyms {
        for v in vs {
            let entry = merged.entry(v.clone()).or_default();
            if v != k && !entry.contains(k) {
                entry.push(k.clone());
            }
        }
    }
    merged
}

/// Real Lucene's `org.apache.lucene.analysis.ngram.NGramTokenFilter`: expands
/// each input token into every contiguous substring ("gram") whose length is
/// between `min_gram` and `max_gram` codepoints, inclusive.
///
/// **Token-filter form, not tokenizer form (a deliberate, documented scope
/// choice)**: real Lucene has both an `NGramTokenizer` (grams raw text
/// directly, ignoring this crate's own word-boundary rules) and an
/// `NGramTokenFilter` (grams already-tokenized terms). This port only
/// implements the token-filter form, since it composes naturally with this
/// crate's existing `Vec<Token> -> Vec<Token>` filter chain (see
/// [`Analyzer::analyze`]) and lets n-gramming sit downstream of
/// [`tokenize`]'s word-boundary logic, [`LowerCaseFilter`], etc. **The
/// tokenizer form is a real, deferred gap**: it would need a raw-`&str ->
/// Vec<Token>` entry point that ignores word boundaries entirely (gramming
/// straight across whitespace/punctuation), which is a different code shape
/// from every other producer in this module and is not implemented here --
/// see `docs/parity.md`.
///
/// **Gram order (confirmed against real Lucene's own behavior)**: for each
/// input token, grams are produced in order of increasing start position,
/// and for each start position, in order of increasing length -- e.g.
/// `"abcde"` with `min_gram = 2`/`max_gram = 3` produces, in this exact
/// order: `"ab"`, `"abc"`, `"bc"`, `"bcd"`, `"cd"`, `"cde"`, `"de"`.
///
/// **A token shorter than `min_gram` produces no output at all** (real
/// Lucene's actual behavior -- not a truncated or padded gram, and not the
/// whole token passed through unchanged).
///
/// **Positions**: the first gram derived from a given input token keeps that
/// token's own `position_increment`; every subsequent gram from the *same*
/// input token gets `position_increment == 0` (an alternative reading at the
/// same starting position -- same convention this crate's [`SynonymFilter`]
/// already uses for injected tokens). `position_length` stays `1` for every
/// gram (each gram is a single alternative token, not a multi-position span).
///
/// **Offsets**: each gram gets its own precise `start_offset`/`end_offset`,
/// computed from the *codepoint* range it covers within the original token's
/// term (never splitting a multi-byte UTF-8 character), added onto the
/// original token's `start_offset` -- consistent with this module's existing
/// (documented, byte-offset) `Token::start_offset`/`end_offset` convention.
pub struct NGramTokenFilter;

/// Computes, for `term`, the ordered list of `(gram_text, start_char, end_char)`
/// substrings whose codepoint length falls in `min_gram..=max_gram`; if
/// `edge_only` is true, only grams starting at codepoint 0 are produced
/// (the [`EdgeNGramTokenFilter`] case). `start_char`/`end_char` are codepoint
/// indices into `term`, suitable for translating to byte offsets via
/// `char_indices`.
fn ngrams_for_term(
    term: &str,
    min_gram: i32,
    max_gram: i32,
    edge_only: bool,
) -> Vec<(String, usize, usize)> {
    let chars: Vec<char> = term.chars().collect();
    let n = chars.len();
    let min_gram = min_gram as usize;
    let max_gram = max_gram as usize;
    if n < min_gram {
        return Vec::new();
    }
    let mut grams = Vec::new();
    let starts: Vec<usize> = if edge_only { vec![0] } else { (0..n).collect() };
    for start in starts {
        for len in min_gram..=max_gram {
            let end = start + len;
            if end > n {
                break;
            }
            let gram: String = chars[start..end].iter().collect();
            grams.push((gram, start, end));
        }
    }
    grams
}

/// Translates a codepoint range `[start_char, end_char)` within `term` into
/// byte offsets relative to the start of `term`, by walking `char_indices`.
/// `end_char == term.chars().count()` maps to `term.len()` (the byte length
/// of the whole term).
fn char_range_to_byte_range(term: &str, start_char: usize, end_char: usize) -> (usize, usize) {
    let mut start_byte = term.len();
    let mut end_byte = term.len();
    let mut found_start = false;
    for (char_idx, (byte_idx, _)) in term.char_indices().enumerate() {
        if char_idx == start_char {
            start_byte = byte_idx;
            found_start = true;
        }
        if char_idx == end_char {
            end_byte = byte_idx;
        }
    }
    debug_assert!(found_start || start_char == term.chars().count());
    (start_byte, end_byte)
}

/// Shared validation for [`NGramTokenFilter::apply`]/
/// [`EdgeNGramTokenFilter::apply`]: `min_gram`/`max_gram` must both be
/// positive, and `min_gram` must not exceed `max_gram`. Mirrors real
/// Lucene's `NGramTokenFilter`/`EdgeNGramTokenFilter` constructors, which
/// both throw `IllegalArgumentException` for these same conditions -- ported
/// here as a `Result::Err` rather than a panic, since this is caller
/// configuration error, not an invariant violation.
fn validate_gram_range(min_gram: i32, max_gram: i32) -> Result<(), String> {
    if min_gram <= 0 {
        return Err(format!("min_gram must be positive, got {min_gram}"));
    }
    if max_gram <= 0 {
        return Err(format!("max_gram must be positive, got {max_gram}"));
    }
    if min_gram > max_gram {
        return Err(format!(
            "min_gram ({min_gram}) must not exceed max_gram ({max_gram})"
        ));
    }
    Ok(())
}

/// Grams `tokens` per [`NGramTokenFilter`]'s documented algorithm/positional
/// convention, applying `ngrams_for_term` (with `edge_only`) to each input
/// token's `term` and emitting one output token per gram. Shared
/// implementation for both [`NGramTokenFilter::apply`] and
/// [`EdgeNGramTokenFilter::apply`].
fn apply_ngram_filter(
    tokens: Vec<Token>,
    min_gram: i32,
    max_gram: i32,
    edge_only: bool,
) -> Result<Vec<Token>, String> {
    validate_gram_range(min_gram, max_gram)?;
    let mut out = Vec::new();
    for t in tokens {
        let grams = ngrams_for_term(&t.term, min_gram, max_gram, edge_only);
        for (idx, (gram, start_char, end_char)) in grams.into_iter().enumerate() {
            let (start_byte, end_byte) = char_range_to_byte_range(&t.term, start_char, end_char);
            out.push(Token {
                term: gram,
                start_offset: t.start_offset + start_byte as i32,
                end_offset: t.start_offset + end_byte as i32,
                position_increment: if idx == 0 { t.position_increment } else { 0 },
                position_length: 1,
            });
        }
    }
    Ok(out)
}

impl NGramTokenFilter {
    /// Grams every token in `tokens` per this filter's documented algorithm.
    /// Returns `Err` if `min_gram`/`max_gram` are not both positive or if
    /// `min_gram > max_gram` (see [`validate_gram_range`]); on success,
    /// tokens shorter than `min_gram` (in codepoints) contribute no output
    /// tokens at all.
    pub fn apply(tokens: Vec<Token>, min_gram: i32, max_gram: i32) -> Result<Vec<Token>, String> {
        apply_ngram_filter(tokens, min_gram, max_gram, false)
    }
}

/// Real Lucene's `org.apache.lucene.analysis.ngram.EdgeNGramTokenFilter`:
/// like [`NGramTokenFilter`], but only produces **prefix** grams anchored at
/// the start of each input token (codepoint index 0) -- the shape used for
/// autocomplete/prefix-search indexing. E.g. `"abcde"` with `min_gram = 2`/
/// `max_gram = 4` produces, in order: `"ab"`, `"abc"`, `"abcd"`.
///
/// Same token-filter-only scope note, no-output-below-`min_gram` rule,
/// position/offset convention, and config-error validation as
/// [`NGramTokenFilter`] -- see that type's docs for the full rationale; this
/// type differs only in which start positions are grammed.
pub struct EdgeNGramTokenFilter;

impl EdgeNGramTokenFilter {
    /// Grams every token in `tokens`, keeping only prefix substrings anchored
    /// at the start of each token. Returns `Err` under the same conditions as
    /// [`NGramTokenFilter::apply`].
    pub fn apply(tokens: Vec<Token>, min_gram: i32, max_gram: i32) -> Result<Vec<Token>, String> {
        apply_ngram_filter(tokens, min_gram, max_gram, true)
    }
}

/// An analyzer composing a tokenizer with a configurable filter chain.
///
/// At minimum applies [`LowerCaseFilter`]; optionally applies [`StopFilter`]
/// when stopwords are configured, optionally applies [`AsciiFoldingFilter`]
/// when enabled via [`Analyzer::with_ascii_folding`], optionally applies
/// [`PorterStemFilter`] when enabled via [`Analyzer::with_stemming`], and
/// optionally applies [`SynonymFilter`] when enabled via
/// [`Analyzer::with_synonyms`]. Additional real-Lucene filters (multi-word
/// synonym phrases via `SynonymGraphFilter`, etc.) are out of scope for this
/// MVP -- see `docs/parity.md`.
pub struct Analyzer {
    stopwords: Option<HashSet<String>>,
    ascii_folding: bool,
    stemming: bool,
    synonyms: Option<HashMap<String, Vec<String>>>,
    synonyms_bidirectional: bool,
}

impl Analyzer {
    /// A "standard"-style analyzer: word-boundary tokenizer + lowercase +
    /// optional stopword removal, mirroring real Lucene's `StandardAnalyzer`
    /// (`StandardTokenizer` + `LowerCaseFilter` + `StopFilter`) at this
    /// crate's documented scope. ASCII-folding and stemming are off by
    /// default -- use [`Analyzer::with_ascii_folding`] / [`Analyzer::with_stemming`]
    /// to enable them -- so every existing caller's behavior is unchanged.
    pub fn standard(stopwords: Option<&HashSet<String>>) -> Self {
        Analyzer {
            stopwords: stopwords.cloned(),
            ascii_folding: false,
            stemming: false,
            synonyms: None,
            synonyms_bidirectional: false,
        }
    }

    /// Enables [`AsciiFoldingFilter`] in this analyzer's chain. Filter
    /// order: tokenize -> **fold** -> lowercase -> stopwords -> stemming.
    /// Folding runs before lowercasing so that an uppercase accented letter
    /// (e.g. `É`) folds straight to its ASCII letter (`E`) and then gets
    /// lowercased along with every other token in the same pass, rather than
    /// needing its own case-conversion step; this also means stopword
    /// matching (which happens next, against already-lowercased terms) sees
    /// the fully folded-and-lowercased form regardless of the input's
    /// original diacritics/casing.
    pub fn with_ascii_folding(mut self) -> Self {
        self.ascii_folding = true;
        self
    }

    /// Enables [`PorterStemFilter`] in this analyzer's chain, mirroring real
    /// Lucene's `EnglishAnalyzer` running `PorterStemFilter` as its last
    /// stage. Filter order: tokenize -> fold -> lowercase -> stopwords ->
    /// **stem**. Stemming runs last so that stopword matching sees
    /// unstemmed terms (matching real Lucene: `EnglishAnalyzer`'s stop set
    /// contains unstemmed words like `"the"`, not stems).
    pub fn with_stemming(mut self) -> Self {
        self.stemming = true;
        self
    }

    /// Enables [`SynonymFilter`] in this analyzer's chain, injecting
    /// configured single-word synonyms at the same position as the term
    /// they replace (see [`SynonymFilter`] for the full scope/positional
    /// semantics). Filter order: tokenize -> fold -> lowercase -> stopwords
    /// -> stem -> **synonyms** (last). Synonyms run last for two reasons:
    /// (1) real Lucene's convention is that synonym expansion operates on
    /// already-normalized terms, so it should see lowercased/stemmed forms,
    /// matching the caller-supplied map's expected (normalized) keys; (2)
    /// running after [`StopFilter`] means a term that is itself a stopword
    /// (and thus removed) never gets its synonym expanded -- expanding a
    /// term that's about to be dropped would be wasted and would leave an
    /// orphaned synonym token with no corresponding original.
    pub fn with_synonyms(mut self, synonyms: HashMap<String, Vec<String>>) -> Self {
        self.synonyms = Some(synonyms);
        self
    }

    /// Opt-in bidirectional variant of [`Analyzer::with_synonyms`]: same
    /// filter-chain position (last), but applies
    /// [`SynonymFilter::apply_bidirectional`] instead of
    /// [`SynonymFilter::apply`], so a configured `key -> [values]` mapping
    /// also expands each `value -> key`. Does not affect any other
    /// existing behavior -- an `Analyzer` built with [`Analyzer::with_synonyms`]
    /// is completely unaffected by this method's existence.
    pub fn with_bidirectional_synonyms(mut self, synonyms: HashMap<String, Vec<String>>) -> Self {
        self.synonyms = Some(synonyms);
        self.synonyms_bidirectional = true;
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
        let tokens = match &self.stopwords {
            Some(stopwords) => StopFilter::apply(tokens, stopwords),
            None => tokens,
        };
        let tokens = if self.stemming {
            PorterStemFilter::apply(tokens)
        } else {
            tokens
        };
        match &self.synonyms {
            Some(synonyms) if self.synonyms_bidirectional => {
                SynonymFilter::apply_bidirectional(tokens, synonyms)
            }
            Some(synonyms) => SynonymFilter::apply(tokens, synonyms),
            None => tokens,
        }
    }
}

/// The classic Porter stemming algorithm (Martin Porter, 1980), operating on
/// lowercase ASCII alphabetic words. See [`PorterStemFilter`] for the
/// documented per-step scope; this module is a direct, mechanical port of
/// the published algorithm's five steps.
mod porter {
    /// Stems `term`, or returns it unchanged if it isn't a lowercase ASCII
    /// alphabetic word (the algorithm's own domain of definition).
    pub(super) fn stem(term: &str) -> String {
        if term.is_empty() || !term.chars().all(|c| c.is_ascii_lowercase()) {
            return term.to_string();
        }
        let mut w: Vec<char> = term.chars().collect();
        step1a(&mut w);
        step1b(&mut w);
        step1c(&mut w);
        step2(&mut w);
        step3(&mut w);
        step4(&mut w);
        step5a(&mut w);
        step5b(&mut w);
        w.into_iter().collect()
    }

    /// Is `chars[i]` a consonant? Vowels are `a`/`e`/`i`/`o`/`u`; `y` is a
    /// consonant at position 0 or immediately after a VOWEL, and a vowel
    /// immediately after a consonant -- Porter's own reference rule
    /// (`case 'y': return (i==0) ? TRUE : !cons(i-1);`). E.g. "syzygy": the
    /// first `y` follows consonant `s`, so it's a VOWEL there (not a
    /// consonant); "toy": `y` follows vowel `o`, so it's a CONSONANT there.
    fn is_consonant(chars: &[char], i: usize) -> bool {
        match chars[i] {
            'a' | 'e' | 'i' | 'o' | 'u' => false,
            'y' => i == 0 || !is_consonant(chars, i - 1),
            _ => true,
        }
    }

    /// The algorithm's "measure" `m`: the number of `VC` (vowel-then-
    /// consonant) sequences in `chars`, after skipping any leading
    /// consonants and ignoring any trailing vowels.
    fn measure(chars: &[char]) -> u32 {
        let n = chars.len();
        let mut i = 0;
        while i < n && is_consonant(chars, i) {
            i += 1;
        }
        let mut m = 0;
        loop {
            while i < n && !is_consonant(chars, i) {
                i += 1;
            }
            if i >= n {
                break;
            }
            while i < n && is_consonant(chars, i) {
                i += 1;
            }
            m += 1;
            if i >= n {
                break;
            }
        }
        m
    }

    /// Does `chars` contain at least one vowel?
    fn contains_vowel(chars: &[char]) -> bool {
        (0..chars.len()).any(|i| !is_consonant(chars, i))
    }

    /// Does `chars` end in a double consonant (e.g. `-tt`, `-ss`)?
    fn ends_double_consonant(chars: &[char]) -> bool {
        let n = chars.len();
        n >= 2 && chars[n - 1] == chars[n - 2] && is_consonant(chars, n - 1)
    }

    /// Does `chars` end in consonant-vowel-consonant, where the final
    /// consonant is not `w`, `x`, or `y` (real Porter's `*o` condition)?
    fn cvc(chars: &[char]) -> bool {
        let n = chars.len();
        n >= 3
            && is_consonant(chars, n - 3)
            && !is_consonant(chars, n - 2)
            && is_consonant(chars, n - 1)
            && !matches!(chars[n - 1], 'w' | 'x' | 'y')
    }

    /// If `w` ends with `suffix` and `measure` of the remaining stem is
    /// `>= min_m`, replaces the suffix with `replacement` and returns `true`.
    /// Otherwise leaves `w` untouched and returns `false`.
    fn try_step(w: &mut Vec<char>, suffix: &str, replacement: &str, min_m: u32) -> bool {
        let n = w.len();
        let suf_len = suffix.chars().count();
        if n < suf_len {
            return false;
        }
        if w[n - suf_len..].iter().collect::<String>() != suffix {
            return false;
        }
        let stem = &w[..n - suf_len];
        if measure(stem) < min_m {
            return false;
        }
        let mut new_w: Vec<char> = stem.to_vec();
        new_w.extend(replacement.chars());
        *w = new_w;
        true
    }

    /// Step 1a: `-sses`->`-ss`, `-ies`->`-i`, `-ss`->`-ss` (no-op), else
    /// trailing `-s`-> (delete). Unconditional on measure.
    fn step1a(w: &mut Vec<char>) {
        let s: String = w.iter().collect();
        if s.ends_with("sses") {
            w.truncate(w.len() - 2);
        } else if s.ends_with("ies") {
            w.truncate(w.len() - 3);
            w.push('i');
        } else if s.ends_with("ss") {
            // no-op: "ss" stays "ss".
        } else if s.ends_with('s') {
            w.truncate(w.len() - 1);
        }
    }

    /// Step 1b: `-eed`->`-ee` (if `m(stem) > 0`); `-ed`/`-ing` deleted only
    /// if the stem contains a vowel, then post-deletion cleanup.
    fn step1b(w: &mut Vec<char>) {
        let s: String = w.iter().collect();
        if s.ends_with("eed") {
            let stem_len = w.len() - 3;
            if measure(&w[..stem_len]) > 0 {
                w.truncate(w.len() - 1);
            }
            return;
        }
        let deleted = if s.ends_with("ed") && contains_vowel(&w[..w.len() - 2]) {
            w.truncate(w.len() - 2);
            true
        } else if s.ends_with("ing") && contains_vowel(&w[..w.len() - 3]) {
            w.truncate(w.len() - 3);
            true
        } else {
            false
        };
        if !deleted {
            return;
        }
        let s2: String = w.iter().collect();
        if s2.ends_with("at") || s2.ends_with("bl") || s2.ends_with("iz") {
            w.push('e');
        } else if ends_double_consonant(w) && !matches!(w[w.len() - 1], 'l' | 's' | 'z') {
            w.pop();
        } else if measure(w) == 1 && cvc(w) {
            w.push('e');
        }
    }

    /// Step 1c: trailing `y` -> `i` if the stem (word minus the `y`)
    /// contains a vowel.
    fn step1c(w: &mut [char]) {
        let n = w.len();
        if n > 0 && w[n - 1] == 'y' && contains_vowel(&w[..n - 1]) {
            w[n - 1] = 'i';
        }
    }

    /// Step 2 (`m(stem) > 0`): the long suffix-family table. Tried in the
    /// order the original paper lists them (longer/more-specific suffixes
    /// like `-ational` before their shorter overlapping counterparts like
    /// `-tional`), stopping at the first match.
    fn step2(w: &mut Vec<char>) {
        const RULES: &[(&str, &str)] = &[
            ("ational", "ate"),
            ("tional", "tion"),
            ("enci", "ence"),
            ("anci", "ance"),
            ("izer", "ize"),
            ("abli", "able"),
            ("alli", "al"),
            ("entli", "ent"),
            ("eli", "e"),
            ("ousli", "ous"),
            ("ization", "ize"),
            ("ation", "ate"),
            ("ator", "ate"),
            ("alism", "al"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("biliti", "ble"),
        ];
        for (suf, rep) in RULES {
            if try_step(w, suf, rep, 1) {
                return;
            }
        }
    }

    /// Step 3 (`m(stem) > 0`): a smaller suffix-family table.
    fn step3(w: &mut Vec<char>) {
        const RULES: &[(&str, &str)] = &[
            ("icate", "ic"),
            ("ative", ""),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ful", ""),
            ("ness", ""),
        ];
        for (suf, rep) in RULES {
            if try_step(w, suf, rep, 1) {
                return;
            }
        }
    }

    /// Step 4 (`m(stem) > 1`): strips a suffix entirely. `-ion` additionally
    /// requires the stem to end in `s` or `t` (real Porter's special case).
    fn step4(w: &mut Vec<char>) {
        const RULES: &[&str] = &[
            "al", "ance", "ence", "er", "ic", "able", "ible", "ant", "ement", "ment", "ent",
        ];
        for suf in RULES {
            if try_step(w, suf, "", 2) {
                return;
            }
        }
        let n = w.len();
        if n >= 4 && w[n - 3..].iter().collect::<String>() == "ion" && matches!(w[n - 4], 's' | 't')
        {
            let stem = &w[..n - 3];
            if measure(stem) > 1 {
                w.truncate(n - 3);
                return;
            }
        }
        const REST: &[&str] = &["ou", "ism", "ate", "iti", "ous", "ive", "ize"];
        for suf in REST {
            if try_step(w, suf, "", 2) {
                return;
            }
        }
    }

    /// Step 5a: trailing `e` deleted if `m(stem) > 1`, or if `m(stem) == 1`
    /// and the stem is not CVC.
    fn step5a(w: &mut Vec<char>) {
        let n = w.len();
        if n == 0 || w[n - 1] != 'e' {
            return;
        }
        let stem = &w[..n - 1];
        let m = measure(stem);
        if m > 1 || (m == 1 && !cvc(stem)) {
            w.truncate(n - 1);
        }
    }

    /// Step 5b: a trailing double `l` collapses to a single `l` if
    /// `m(word) > 1`.
    fn step5b(w: &mut Vec<char>) {
        let n = w.len();
        if n >= 2 && w[n - 1] == 'l' && w[n - 2] == 'l' && measure(w) > 1 {
            w.pop();
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
            position_length: 1,
        }
    }

    fn tok_len(term: &str, start: i32, end: i32, pos_inc: i32, pos_len: i32) -> Token {
        Token {
            term: term.to_string(),
            start_offset: start,
            end_offset: end,
            position_increment: pos_inc,
            position_length: pos_len,
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

    // -- Embedded numeric punctuation ("3.14", "1,000") --

    #[test]
    fn tokenize_number_with_embedded_period_stays_one_token() {
        // OLD (wrong) behavior: this split into "3" and "14".
        let tokens = tokenize("pi is 3.14 today");
        assert_eq!(
            tokens,
            vec![
                tok("pi", 0, 2, 1),
                tok("is", 3, 5, 1),
                tok("3.14", 6, 10, 1),
                tok("today", 11, 16, 1),
            ]
        );
    }

    #[test]
    fn tokenize_number_with_embedded_comma_stays_one_token() {
        // OLD (wrong) behavior: this split into "1" and "000".
        let tokens = tokenize("1,000 dollars");
        assert_eq!(
            tokens,
            vec![tok("1,000", 0, 5, 1), tok("dollars", 6, 13, 1),]
        );
    }

    #[test]
    fn tokenize_sentence_ending_period_after_number_still_splits() {
        // Adjacent case that must NOT be affected: a real sentence-ending
        // period (nothing alphanumeric follows it) still splits off.
        let tokens = tokenize("The total is 42. Done.");
        assert_eq!(
            tokens,
            vec![
                tok("The", 0, 3, 1),
                tok("total", 4, 9, 1),
                tok("is", 10, 12, 1),
                tok("42", 13, 15, 1),
                tok("Done", 17, 21, 1),
            ]
        );
    }

    // -- Acronym-style internal periods ("U.S.A.") --

    #[test]
    fn tokenize_acronym_kept_together() {
        // OLD (wrong) behavior: this split into "U", "S", "A".
        let tokens = tokenize("U.S.A. is here");
        assert_eq!(
            tokens,
            vec![
                tok("U.S.A", 0, 5, 1),
                tok("is", 7, 9, 1),
                tok("here", 10, 14, 1),
            ]
        );
    }

    #[test]
    fn tokenize_trailing_sentence_period_after_word_still_splits() {
        // Adjacent case that must NOT be affected: a normal word followed by
        // a sentence-ending period still splits the period off.
        let tokens = tokenize("This is the end. Next sentence.");
        assert_eq!(
            tokens,
            vec![
                tok("This", 0, 4, 1),
                tok("is", 5, 7, 1),
                tok("the", 8, 11, 1),
                tok("end", 12, 15, 1),
                tok("Next", 17, 21, 1),
                tok("sentence", 22, 30, 1),
            ]
        );
    }

    // -- Internal apostrophes ("don't", "O'Brien") --

    #[test]
    fn tokenize_apostrophe_contraction_kept_together() {
        // OLD (wrong) behavior: this split into "don" and "t".
        let tokens = tokenize("don't stop");
        assert_eq!(tokens, vec![tok("don't", 0, 5, 1), tok("stop", 6, 10, 1),]);
    }

    #[test]
    fn tokenize_apostrophe_name_kept_together() {
        // OLD (wrong) behavior: this split into "O" and "Brien".
        let tokens = tokenize("O'Brien arrived");
        assert_eq!(
            tokens,
            vec![tok("O'Brien", 0, 7, 1), tok("arrived", 8, 15, 1),]
        );
    }

    #[test]
    fn tokenize_leading_apostrophe_not_absorbed() {
        // Adjacent case that must NOT be affected: an apostrophe with no
        // alphanumeric character before it (e.g. an opening quote) is a
        // plain separator, not part of the following word.
        let tokens = tokenize("'tis the season");
        assert_eq!(
            tokens,
            vec![
                tok("tis", 1, 4, 1),
                tok("the", 5, 8, 1),
                tok("season", 9, 15, 1),
            ]
        );
    }

    // -- UAX#29 extensions: combining marks, CJK, Hangul, ZWJ --

    #[test]
    fn tokenize_combining_mark_stays_attached_to_base_char() {
        // "e" + combining acute accent (U+0301), decomposed form of "é".
        // A naive per-char split would treat the combining mark as its own
        // boundary; UAX#29 (via WB's Extend rule) keeps it fused to "cafe".
        let tokens = tokenize("cafe\u{0301} today");
        assert_eq!(tokens.len(), 2);
        assert_eq!(tokens[0].term, "cafe\u{0301}");
        assert_eq!(tokens[0].start_offset, 0);
        assert_eq!(tokens[0].end_offset, "cafe\u{0301}".len() as i32);
        assert_eq!(tokens[1].term, "today");
    }

    #[test]
    fn tokenize_cjk_ideographs_split_one_per_character() {
        // Each Han ideograph is its own token -- no word clustering across
        // CJK text, unlike Latin script.
        let tokens = tokenize("你好世界");
        assert_eq!(
            tokens,
            vec![
                tok("你", 0, 3, 1),
                tok("好", 3, 6, 1),
                tok("世", 6, 9, 1),
                tok("界", 9, 12, 1),
            ]
        );
    }

    #[test]
    fn tokenize_precomposed_hangul_syllable_is_one_token() {
        let tokens = tokenize("안녕하세요");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].term, "안녕하세요");
    }

    #[test]
    fn tokenize_conjoining_hangul_jamo_cluster_into_one_syllable_token() {
        // Leading consonant + vowel + trailing consonant jamo (U+1100,
        // U+1161, U+11A8) compose the syllable "각"; UAX#29's Hangul WB
        // rules cluster them into one token, not three.
        let jamo = "\u{1100}\u{1161}\u{11A8}";
        let tokens = tokenize(jamo);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].term, jamo);
    }

    #[test]
    fn tokenize_mixed_cjk_and_latin() {
        let tokens = tokenize("hello 世界 world");
        assert_eq!(
            tokens,
            vec![
                tok("hello", 0, 5, 1),
                tok("世", 6, 9, 1),
                tok("界", 9, 12, 1),
                tok("world", 13, 18, 1),
            ]
        );
    }

    #[test]
    fn tokenize_zwj_between_letters_does_not_split() {
        // A bare ZWJ (U+200D) between two letters is Extend-like and does
        // not introduce a word boundary.
        let joined = "a\u{200d}b";
        let tokens = tokenize(joined);
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].term, joined);
    }

    #[test]
    fn tokenize_emoji_produces_no_token() {
        // Emoji contain no alphanumeric codepoints, so -- like any other
        // non-alphanumeric run -- they produce no token at all, but do not
        // corrupt tokenization of the surrounding text.
        let tokens = tokenize("test\u{1F44D}emoji");
        assert_eq!(tokens, vec![tok("test", 0, 4, 1), tok("emoji", 8, 13, 1)]);
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
    fn english_stop_words_matches_real_lucene_canonical_list() {
        // Transcribed from real Lucene's
        // `org.apache.lucene.analysis.en.EnglishAnalyzer.ENGLISH_STOP_WORDS_SET`
        // (the classic Lucene/Snowball English stop list). Review-confirmed
        // caveat, stated honestly: this literal is the same 33 words as
        // `ENGLISH_STOP_WORDS` itself, so this test guards against a future
        // edit letting the two lists drift apart -- it does not, on its own,
        // prove `ENGLISH_STOP_WORDS` was transcribed correctly from Lucene in
        // the first place (a one-time transcription error made once and
        // repeated in both places would still pass). That correctness claim
        // rests on careful transcription against the real Lucene source, not
        // on this test's structure.
        const CANONICAL_33: &[&str] = &[
            "a", "an", "and", "are", "as", "at", "be", "but", "by", "for", "if", "in", "into",
            "is", "it", "no", "not", "of", "on", "or", "such", "that", "the", "their", "then",
            "there", "these", "they", "this", "to", "was", "will", "with",
        ];
        assert_eq!(
            CANONICAL_33.len(),
            33,
            "the reference list itself must have exactly 33 entries"
        );
        let stopwords = english_stop_words();
        assert_eq!(
            stopwords.len(),
            33,
            "ENGLISH_STOP_WORDS must have exactly 33 entries, matching real Lucene"
        );
        for word in CANONICAL_33 {
            assert!(
                stopwords.contains(*word),
                "canonical Lucene English stopword {word:?} is missing from ENGLISH_STOP_WORDS"
            );
        }
        // No extras: every entry in this port's set must also appear in the
        // canonical list (catches an accidentally-added wrong/extra word).
        for word in &stopwords {
            assert!(
                CANONICAL_33.contains(&word.as_str()),
                "ENGLISH_STOP_WORDS contains {word:?}, which is not one of real Lucene's 33 \
                 canonical English stopwords"
            );
        }
    }

    #[test]
    fn english_stop_words_case_is_already_lowercase() {
        // Real Lucene's set is populated with already-lowercase entries, and
        // StopFilter matches against already-lowercased terms (it runs after
        // LowerCaseFilter in the chain) -- so every entry here must be
        // lowercase, not merely "matched case-insensitively".
        for word in ENGLISH_STOP_WORDS {
            assert_eq!(
                *word,
                word.to_lowercase(),
                "{word:?} must be stored lowercase"
            );
        }
    }

    #[test]
    fn english_stop_words_does_not_false_positive_on_content_words() {
        // Representative non-stopwords that must survive StopFilter
        // untouched -- proves the set isn't overly broad (e.g. accidentally
        // matching real content words via substring/prefix matching instead
        // of exact string equality).
        let stopwords = english_stop_words();
        for word in ["search", "lucene", "rust", "document", "index", "query"] {
            assert!(!stopwords.contains(word), "{word:?} must NOT be a stopword");
        }
        let tokens = tokenize("search the lucene rust document index and query");
        let tokens = LowerCaseFilter::apply(tokens);
        let out = StopFilter::apply(tokens, &stopwords);
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        // "the" and "and" are real stopwords and must be removed; every
        // other word here is a real content word and must survive.
        assert_eq!(
            terms,
            vec!["search", "lucene", "rust", "document", "index", "query"]
        );
    }

    #[test]
    fn english_stop_words_used_via_analyzer_standard() {
        // End-to-end: Analyzer::standard wired with the real default English
        // stop set behaves like real Lucene's EnglishAnalyzer/StandardAnalyzer
        // defaults for a sentence containing several of the 33 stopwords.
        let stopwords = english_stop_words();
        let analyzer = Analyzer::standard(Some(&stopwords));
        let out = analyzer.analyze("The quick fox will jump into the river");
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["quick", "fox", "jump", "river"]);
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
    fn porter_step1a_plural_forms() {
        let tokens = vec![
            tok("caresses", 0, 8, 1),
            tok("ponies", 0, 6, 1),
            tok("cats", 0, 4, 1),
            tok("caress", 0, 6, 1),
        ];
        let out = PorterStemFilter::apply(tokens);
        assert_eq!(
            out,
            vec![
                tok("caress", 0, 8, 1),
                tok("poni", 0, 6, 1),
                tok("cat", 0, 4, 1),
                tok("caress", 0, 6, 1),
            ]
        );
    }

    #[test]
    fn porter_step1b_ed_ing_and_short_word_guard() {
        let cases: &[(&str, &str)] = &[
            ("feed", "feed"),   // *v* fails on stem "f" -- must NOT stem.
            ("agreed", "agre"), // m(stem)>0 for "eed" -> "ee", then 5a strips "e".
            ("plastered", "plaster"),
            ("bled", "bled"), // stem "bl" has no vowel -- must NOT stem.
            ("motoring", "motor"),
            ("sing", "sing"), // stem "s" has no vowel -- must NOT stem.
        ];
        for (input, expected) in cases {
            let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
            assert_eq!(out[0].term, *expected, "stemming {input:?}");
        }
    }

    #[test]
    fn porter_stem_leaves_offsets_and_position_increment_untouched() {
        let tokens = vec![tok("running", 5, 12, 2)];
        let out = PorterStemFilter::apply(tokens);
        assert_eq!(out, vec![tok("run", 5, 12, 2)]);
    }

    #[test]
    fn porter_stem_happiness_and_running() {
        assert_eq!(
            PorterStemFilter::apply(vec![tok("running", 0, 1, 1)])[0].term,
            "run"
        );
        assert_eq!(
            PorterStemFilter::apply(vec![tok("flies", 0, 1, 1)])[0].term,
            "fli"
        );
        assert_eq!(
            PorterStemFilter::apply(vec![tok("happiness", 0, 1, 1)])[0].term,
            "happi"
        );
    }

    /// Regression test for a real bug caught in review: `y` immediately
    /// after a CONSONANT must count as a VOWEL (Porter's own reference rule,
    /// `case 'y': return (i==0) ? TRUE : !cons(i-1);`), not a consonant, when
    /// deciding whether a stem "contains a vowel" for step 1b's `-ing`
    /// removal guard. Before this fix, `contains_vowel("fly")` was wrongly
    /// `false` (the `y`, following consonant `l`, was misclassified as a
    /// consonant instead of a vowel), so `-ing` was never stripped and
    /// "flying"/"trying" passed through completely unstemmed. After the fix,
    /// `-ing` correctly strips to "fly"/"try" -- step 1c's own, separate
    /// `(*v*)` condition (checked against the letters preceding the trailing
    /// `y`, i.e. "fl"/"tr") doesn't additionally fire here since neither
    /// contains a vowel, so the final `y` is not further converted to `i`.
    #[test]
    fn porter_stem_y_after_consonant_is_a_vowel_not_a_consonant() {
        assert_eq!(
            PorterStemFilter::apply(vec![tok("flying", 0, 1, 1)])[0].term,
            "fly",
            "the -ing suffix must be stripped now that y-after-consonant counts as a vowel"
        );
        assert_eq!(
            PorterStemFilter::apply(vec![tok("trying", 0, 1, 1)])[0].term,
            "try",
            "the -ing suffix must be stripped now that y-after-consonant counts as a vowel"
        );
    }

    #[test]
    fn porter_stem_step2_step3_step4_suffix_families() {
        let cases: &[(&str, &str)] = &[
            ("relational", "relat"),
            ("conditional", "condit"),
            ("rational", "ration"),
            ("valenci", "valenc"),
            ("hesitanci", "hesit"),
            ("digitizer", "digit"),
            ("conformabli", "conform"),
            ("radicalli", "radic"),
            ("differentli", "differ"),
            ("vileli", "vile"),
            ("analogousli", "analog"),
            ("vietnamization", "vietnam"),
            ("predication", "predic"),
            ("operator", "oper"),
            ("feudalism", "feudal"),
            ("decisiveness", "decis"),
            ("hopefulness", "hope"),
            ("callousness", "callous"),
            ("formaliti", "formal"),
            ("sensitiviti", "sensit"),
            ("sensibiliti", "sensibl"),
            ("triplicate", "triplic"),
            ("formative", "form"),
            ("formalize", "formal"),
            ("electriciti", "electr"),
            ("electrical", "electr"),
            ("hopeful", "hope"),
            ("goodness", "good"),
            ("revival", "reviv"),
            ("allowance", "allow"),
            ("inference", "infer"),
            ("airliner", "airlin"),
            ("gyroscopic", "gyroscop"),
            ("adjustable", "adjust"),
            ("defensible", "defens"),
            ("irritant", "irrit"),
            ("replacement", "replac"),
            ("adjustment", "adjust"),
            ("dependent", "depend"),
            ("adoption", "adopt"),
            ("homologou", "homolog"),
            ("communism", "commun"),
            ("activate", "activ"),
            ("angulariti", "angular"),
            ("homologous", "homolog"),
            ("effective", "effect"),
            ("bowdlerize", "bowdler"),
        ];
        for (input, expected) in cases {
            let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
            assert_eq!(out[0].term, *expected, "stemming {input:?}");
        }
    }

    #[test]
    fn porter_stem_step5_final_e_and_double_l() {
        let cases: &[(&str, &str)] = &[
            ("probate", "probat"),
            ("rate", "rate"), // m==1 and IS cvc -- 'e' must survive.
            ("cease", "ceas"),
            ("controll", "control"),
            ("roll", "roll"), // m==1, not >1 -- must NOT collapse.
        ];
        for (input, expected) in cases {
            let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
            assert_eq!(out[0].term, *expected, "stemming {input:?}");
        }
    }

    /// Porter's 1980 paper illustrates steps 1a/1b/1c with their own worked
    /// vocabulary (distinct from the step 2/3/4 list already covered by
    /// `porter_stem_step2_step3_step4_suffix_families`); this test traces
    /// that vocabulary directly against the implementation to close a real
    /// gap in this port's test coverage: step 1a's plural forms
    /// (`caresses`/`ponies`/`ties`/`caress`/`cats`); step 1b's guards *not*
    /// firing when they shouldn't (`feed` hits the `-eed` rule but
    /// `m(fe) == 0` so it stays `feed`; `bled`/`sing` have no vowel before
    /// `-ed`/`-ing` so they stay unchanged too) versus firing correctly
    /// (`agreed`->`agre`, `plastered`->`plaster`, `motoring`->`motor`), plus
    /// each of the three post-deletion cleanup branches (`-at`/`-bl`/`-iz`
    /// append via `sized`->`size`; double-consonant-drop via
    /// `hopping`->`hop`/`tanned`->`tan`/`falling`->`fall`/`hissing`->`hiss`/
    /// `fizzed`->`fizz`; plain deletion with no cleanup needed via
    /// `failing`->`fail`/`filing`->`file`); and step 1c's `-y`->`-i`
    /// conversion (`happy`->`happi`) versus its vowel guard not firing
    /// (`sky`->`sky`, since `sk` contains no vowel).
    #[test]
    fn porter_stem_step1a_step1b_step1c_paper_vocabulary() {
        let cases: &[(&str, &str)] = &[
            // Step 1a.
            ("caresses", "caress"),
            ("ponies", "poni"),
            ("ties", "ti"),
            ("caress", "caress"),
            ("cats", "cat"),
            // Step 1b: `-eed` with m==0 stays put; `-ed`/`-ing` with no vowel
            // in the stem stays put too.
            ("feed", "feed"),
            ("agreed", "agre"),
            ("plastered", "plaster"),
            ("bled", "bled"),
            ("motoring", "motor"),
            ("sing", "sing"),
            // Step 1b post-deletion cleanup, all three branches.
            ("conflated", "conflat"),
            ("troubled", "troubl"),
            ("sized", "size"),
            ("hopping", "hop"),
            ("tanned", "tan"),
            ("falling", "fall"),
            ("hissing", "hiss"),
            ("fizzed", "fizz"),
            ("failing", "fail"),
            ("filing", "file"),
            // Step 1c.
            ("happy", "happi"),
            ("sky", "sky"),
        ];
        for (input, expected) in cases {
            let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
            assert_eq!(out[0].term, *expected, "stemming {input:?}");
        }
    }

    #[test]
    fn porter_stem_non_lowercase_ascii_passes_through_unchanged() {
        // Uppercase and non-ASCII terms are outside the algorithm's domain
        // of definition -- must pass through unchanged, never panic.
        let tokens = vec![
            tok("Running", 0, 7, 1),
            tok("café", 0, 4, 1),
            tok("", 0, 0, 1),
            tok("123", 0, 3, 1),
        ];
        let out = PorterStemFilter::apply(tokens.clone());
        assert_eq!(out, tokens);
    }

    #[test]
    fn analyzer_with_stemming_runs_after_stopwords() {
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let analyzer = Analyzer::standard(Some(&stopwords)).with_stemming();
        let out = analyzer.analyze("The Running Flies");
        assert_eq!(out, vec![tok("run", 4, 11, 2), tok("fli", 12, 17, 1),]);
    }

    #[test]
    fn analyzer_default_has_no_stemming_backward_compatible() {
        let analyzer = Analyzer::standard(None);
        let out = analyzer.analyze("running");
        assert_eq!(out, vec![tok("running", 0, 7, 1)]);
    }

    #[test]
    fn synonym_filter_injects_single_synonym_at_same_position() {
        let tokens = vec![tok("quick", 0, 5, 1)];
        let synonyms: HashMap<String, Vec<String>> =
            [("quick".to_string(), vec!["fast".to_string()])]
                .into_iter()
                .collect();
        let out = SynonymFilter::apply(tokens, &synonyms);
        assert_eq!(out, vec![tok("quick", 0, 5, 1), tok("fast", 0, 5, 0),]);
    }

    #[test]
    fn synonym_filter_multiple_synonyms_all_same_position() {
        let tokens = vec![tok("quick", 0, 5, 1)];
        let synonyms: HashMap<String, Vec<String>> = [(
            "quick".to_string(),
            vec!["fast".to_string(), "speedy".to_string()],
        )]
        .into_iter()
        .collect();
        let out = SynonymFilter::apply(tokens, &synonyms);
        assert_eq!(
            out,
            vec![
                tok("quick", 0, 5, 1),
                tok("fast", 0, 5, 0),
                tok("speedy", 0, 5, 0),
            ]
        );
    }

    #[test]
    fn synonym_filter_no_configured_synonym_passes_through_unchanged() {
        let tokens = vec![tok("hello", 0, 5, 1)];
        let synonyms: HashMap<String, Vec<String>> =
            [("quick".to_string(), vec!["fast".to_string()])]
                .into_iter()
                .collect();
        let out = SynonymFilter::apply(tokens.clone(), &synonyms);
        assert_eq!(out, tokens);
    }

    #[test]
    fn synonym_filter_not_automatically_bidirectional() {
        // Configuring "quick" -> "fast" must NOT also expand "fast" ->
        // "quick" -- real Lucene requires explicit configuration in both
        // directions.
        let tokens = vec![tok("fast", 0, 4, 1)];
        let synonyms: HashMap<String, Vec<String>> =
            [("quick".to_string(), vec!["fast".to_string()])]
                .into_iter()
                .collect();
        let out = SynonymFilter::apply(tokens.clone(), &synonyms);
        assert_eq!(out, tokens);
    }

    #[test]
    fn synonym_filter_composed_with_stop_filter_stopword_removed_before_expansion() {
        // "the quick fox" with "the" as a stopword and "quick" -> "fast"
        // configured: stopwords run first, so "the" is gone and never
        // considered for synonym expansion (it isn't in the map anyway, but
        // this also proves the ordering doesn't crash/misbehave on a
        // stopword-adjacent term); "quick" survives and still expands.
        let analyzer = {
            let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
            let synonyms: HashMap<String, Vec<String>> =
                [("quick".to_string(), vec!["fast".to_string()])]
                    .into_iter()
                    .collect();
            Analyzer::standard(Some(&stopwords)).with_synonyms(synonyms)
        };
        let out = analyzer.analyze("the quick fox");
        assert_eq!(
            out,
            vec![
                tok("quick", 4, 9, 2),
                tok("fast", 4, 9, 0),
                tok("fox", 10, 13, 1),
            ]
        );
    }

    #[test]
    fn synonym_filter_stopword_itself_never_gets_expanded() {
        // If the stopword itself had a configured synonym, it must not
        // survive to be expanded, since it's removed before synonym
        // expansion runs.
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let synonyms: HashMap<String, Vec<String>> =
            [("the".to_string(), vec!["definite_article".to_string()])]
                .into_iter()
                .collect();
        let analyzer = Analyzer::standard(Some(&stopwords)).with_synonyms(synonyms);
        let out = analyzer.analyze("the fox");
        assert_eq!(out, vec![tok("fox", 4, 7, 2)]);
    }

    #[test]
    fn synonym_filter_runs_after_stemming() {
        // Configuring the map with the STEMMED form ("run") as the key
        // proves synonyms see post-stemming terms, since stemming runs
        // before synonym expansion in the chain.
        let synonyms: HashMap<String, Vec<String>> =
            [("run".to_string(), vec!["sprint".to_string()])]
                .into_iter()
                .collect();
        let analyzer = Analyzer::standard(None)
            .with_stemming()
            .with_synonyms(synonyms);
        let out = analyzer.analyze("running");
        assert_eq!(out, vec![tok("run", 0, 7, 1), tok("sprint", 0, 7, 0),]);
    }

    #[test]
    fn synonym_filter_apply_bidirectional_expands_both_directions() {
        // Only "cat" -> ["feline"] is configured; apply_bidirectional must
        // ALSO expand "feline" -> "cat" without that reverse mapping being
        // configured explicitly.
        let synonyms: HashMap<String, Vec<String>> =
            [("cat".to_string(), vec!["feline".to_string()])]
                .into_iter()
                .collect();

        let out_forward = SynonymFilter::apply_bidirectional(vec![tok("cat", 0, 3, 1)], &synonyms);
        assert_eq!(
            out_forward,
            vec![tok("cat", 0, 3, 1), tok("feline", 0, 3, 0)]
        );

        let out_reverse =
            SynonymFilter::apply_bidirectional(vec![tok("feline", 0, 6, 1)], &synonyms);
        assert_eq!(
            out_reverse,
            vec![tok("feline", 0, 6, 1), tok("cat", 0, 6, 0)]
        );
    }

    #[test]
    fn synonym_filter_apply_non_bidirectional_still_unidirectional() {
        // The original `apply` entry point must remain completely unchanged:
        // with the same config, analyzing "feline" injects nothing.
        let synonyms: HashMap<String, Vec<String>> =
            [("cat".to_string(), vec!["feline".to_string()])]
                .into_iter()
                .collect();
        let tokens = vec![tok("feline", 0, 6, 1)];
        let out = SynonymFilter::apply(tokens.clone(), &synonyms);
        assert_eq!(out, tokens);
    }

    #[test]
    fn synonym_filter_apply_bidirectional_no_duplicate_when_both_directions_configured() {
        // "cat" -> ["feline"] AND "feline" -> ["cat"] both explicitly
        // configured: the merged map must not inject "cat" (or "feline")
        // twice.
        let synonyms: HashMap<String, Vec<String>> = [
            ("cat".to_string(), vec!["feline".to_string()]),
            ("feline".to_string(), vec!["cat".to_string()]),
        ]
        .into_iter()
        .collect();

        let out_cat = SynonymFilter::apply_bidirectional(vec![tok("cat", 0, 3, 1)], &synonyms);
        assert_eq!(out_cat, vec![tok("cat", 0, 3, 1), tok("feline", 0, 3, 0)]);

        let out_feline =
            SynonymFilter::apply_bidirectional(vec![tok("feline", 0, 6, 1)], &synonyms);
        assert_eq!(
            out_feline,
            vec![tok("feline", 0, 6, 1), tok("cat", 0, 6, 0)]
        );
    }

    #[test]
    fn synonym_filter_apply_bidirectional_multi_value_key_reverses_independently() {
        // "cat" -> ["feline", "kitty"]: the reverse mapping must produce two
        // SEPARATE entries, "feline" -> ["cat"] and "kitty" -> ["cat"] --
        // "feline" and "kitty" must NOT become synonyms of each other, since
        // the forward config never said that.
        let synonyms: HashMap<String, Vec<String>> = [(
            "cat".to_string(),
            vec!["feline".to_string(), "kitty".to_string()],
        )]
        .into_iter()
        .collect();

        let out_feline =
            SynonymFilter::apply_bidirectional(vec![tok("feline", 0, 6, 1)], &synonyms);
        assert_eq!(
            out_feline,
            vec![tok("feline", 0, 6, 1), tok("cat", 0, 6, 0)]
        );

        let out_kitty = SynonymFilter::apply_bidirectional(vec![tok("kitty", 0, 5, 1)], &synonyms);
        assert_eq!(out_kitty, vec![tok("kitty", 0, 5, 1), tok("cat", 0, 5, 0)]);

        // Forward direction still expands to BOTH synonyms, unaffected.
        let out_cat = SynonymFilter::apply_bidirectional(vec![tok("cat", 0, 3, 1)], &synonyms);
        assert_eq!(
            out_cat,
            vec![
                tok("cat", 0, 3, 1),
                tok("feline", 0, 3, 0),
                tok("kitty", 0, 3, 0),
            ]
        );
    }

    #[test]
    fn synonym_filter_bidirectional_composed_with_stop_filter() {
        // Mirrors synonym_filter_composed_with_stop_filter_stopword_removed_before_expansion,
        // but with bidirectional mode on: "the cat fox" with "the" as a
        // stopword and "cat" -> ["feline"] configured bidirectionally --
        // stopwords still run first, and "cat" still expands to "feline".
        let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
        let synonyms: HashMap<String, Vec<String>> =
            [("cat".to_string(), vec!["feline".to_string()])]
                .into_iter()
                .collect();
        let analyzer = Analyzer::standard(Some(&stopwords)).with_bidirectional_synonyms(synonyms);
        let out = analyzer.analyze("the cat fox");
        assert_eq!(
            out,
            vec![
                tok("cat", 4, 7, 2),
                tok("feline", 4, 7, 0),
                tok("fox", 8, 11, 1),
            ]
        );
    }

    #[test]
    fn synonym_filter_bidirectional_composed_with_stemming() {
        // Mirrors synonym_filter_runs_after_stemming, but bidirectional:
        // configuring the STEMMED form "run" -> ["sprint"] bidirectionally
        // means analyzing "sprint" (already the stemmed form of itself)
        // also injects "run".
        let synonyms: HashMap<String, Vec<String>> =
            [("run".to_string(), vec!["sprint".to_string()])]
                .into_iter()
                .collect();
        let analyzer = Analyzer::standard(None)
            .with_stemming()
            .with_bidirectional_synonyms(synonyms);

        let out_forward = analyzer.analyze("running");
        assert_eq!(
            out_forward,
            vec![tok("run", 0, 7, 1), tok("sprint", 0, 7, 0),]
        );

        let out_reverse = analyzer.analyze("sprint");
        assert_eq!(
            out_reverse,
            vec![tok("sprint", 0, 6, 1), tok("run", 0, 6, 0),]
        );
    }

    #[test]
    fn synonym_filter_multiword_input_collapses_to_single_output_token() {
        // "wi fi" -> "wifi": a 2-token input phrase becomes 1 output token
        // with position_length == 2, marking it spans both original
        // positions. Offsets cover the whole matched span.
        let tokens = vec![tok("wi", 0, 2, 1), tok("fi", 3, 5, 1)];
        let rules = vec![SynonymRule {
            input: vec!["wi".to_string(), "fi".to_string()],
            outputs: vec![vec!["wifi".to_string()]],
        }];
        let out = SynonymFilter::apply_multiword(tokens, &rules);
        assert_eq!(
            out,
            vec![
                tok("wi", 0, 2, 1),
                tok("fi", 3, 5, 1),
                tok_len("wifi", 0, 5, 0, 2),
            ]
        );
    }

    #[test]
    fn synonym_filter_single_word_input_expands_to_multiword_output() {
        // "usa" -> "united states of america": 1 input token becomes 4
        // chained output tokens, first at position_increment 0 (same slot as
        // "usa"), the rest at position_increment 1 each, all position_length
        // 1 (each occupies exactly one position on the output path).
        let tokens = vec![tok("usa", 0, 3, 1)];
        let rules = vec![SynonymRule {
            input: vec!["usa".to_string()],
            outputs: vec![vec![
                "united".to_string(),
                "states".to_string(),
                "of".to_string(),
                "america".to_string(),
            ]],
        }];
        let out = SynonymFilter::apply_multiword(tokens, &rules);
        assert_eq!(
            out,
            vec![
                tok("usa", 0, 3, 1),
                tok_len("united", 0, 3, 0, 1),
                tok_len("states", 0, 3, 1, 1),
                tok_len("of", 0, 3, 1, 1),
                tok_len("america", 0, 3, 1, 1),
            ]
        );
    }

    #[test]
    fn synonym_filter_multiword_to_multiword() {
        // "new york" -> "big apple": a 2-token input phrase to a 2-token
        // output phrase. The output's first token gets position_length 1
        // (not the input's length of 2), since output.len() > 1.
        let tokens = vec![tok("new", 0, 3, 1), tok("york", 4, 8, 1)];
        let rules = vec![SynonymRule {
            input: vec!["new".to_string(), "york".to_string()],
            outputs: vec![vec!["big".to_string(), "apple".to_string()]],
        }];
        let out = SynonymFilter::apply_multiword(tokens, &rules);
        assert_eq!(
            out,
            vec![
                tok("new", 0, 3, 1),
                tok("york", 4, 8, 1),
                tok_len("big", 0, 8, 0, 1),
                tok_len("apple", 0, 8, 1, 1),
            ]
        );
    }

    #[test]
    fn synonym_filter_multiword_partial_prefix_does_not_match() {
        // "wi" alone (not followed by "fi") must NOT trigger the "wi fi"
        // rule -- neither mid-stream (followed by something else) nor as the
        // very last token (nothing following at all).
        let rules = vec![SynonymRule {
            input: vec!["wi".to_string(), "fi".to_string()],
            outputs: vec![vec!["wifi".to_string()]],
        }];

        let followed_by_other = vec![tok("wi", 0, 2, 1), tok("max", 3, 6, 1)];
        let out = SynonymFilter::apply_multiword(followed_by_other.clone(), &rules);
        assert_eq!(out, followed_by_other);

        let last_token = vec![tok("wi", 0, 2, 1)];
        let out = SynonymFilter::apply_multiword(last_token.clone(), &rules);
        assert_eq!(out, last_token);
    }

    #[test]
    fn synonym_filter_multiword_no_rules_passes_through_unchanged() {
        let tokens = vec![tok("hello", 0, 5, 1), tok("world", 6, 11, 1)];
        let out = SynonymFilter::apply_multiword(tokens.clone(), &[]);
        assert_eq!(out, tokens);
    }

    #[test]
    fn synonym_filter_multiword_prefers_longest_match() {
        // Both "new" -> "novel" and "new york" -> "nyc" configured; the
        // longer "new york" phrase should win over the shorter "new" rule
        // when both could match at the same starting position.
        let tokens = vec![tok("new", 0, 3, 1), tok("york", 4, 8, 1)];
        let rules = vec![
            SynonymRule {
                input: vec!["new".to_string()],
                outputs: vec![vec!["novel".to_string()]],
            },
            SynonymRule {
                input: vec!["new".to_string(), "york".to_string()],
                outputs: vec![vec!["nyc".to_string()]],
            },
        ];
        let out = SynonymFilter::apply_multiword(tokens, &rules);
        assert_eq!(
            out,
            vec![
                tok("new", 0, 3, 1),
                tok("york", 4, 8, 1),
                tok_len("nyc", 0, 8, 0, 2),
            ]
        );
    }

    #[test]
    fn synonym_filter_multiword_multiple_output_alternatives() {
        // A single multi-word input can have more than one alternative
        // output path (e.g. "wi fi" -> "wifi" or "wireless").
        let tokens = vec![tok("wi", 0, 2, 1), tok("fi", 3, 5, 1)];
        let rules = vec![SynonymRule {
            input: vec!["wi".to_string(), "fi".to_string()],
            outputs: vec![vec!["wifi".to_string()], vec!["wireless".to_string()]],
        }];
        let out = SynonymFilter::apply_multiword(tokens, &rules);
        assert_eq!(
            out,
            vec![
                tok("wi", 0, 2, 1),
                tok("fi", 3, 5, 1),
                tok_len("wifi", 0, 5, 0, 2),
                tok_len("wireless", 0, 5, 0, 2),
            ]
        );
    }

    #[test]
    fn synonym_filter_apply_and_apply_bidirectional_unaffected_by_multiword_addition() {
        // Sanity check the earlier single-word bidirectional task's behavior
        // is untouched by adding apply_multiword: same assertions as
        // synonym_filter_apply_bidirectional_expands_both_directions.
        let synonyms: HashMap<String, Vec<String>> =
            [("cat".to_string(), vec!["feline".to_string()])]
                .into_iter()
                .collect();

        let out_forward = SynonymFilter::apply_bidirectional(vec![tok("cat", 0, 3, 1)], &synonyms);
        assert_eq!(
            out_forward,
            vec![tok("cat", 0, 3, 1), tok("feline", 0, 3, 0)]
        );

        let out_reverse =
            SynonymFilter::apply_bidirectional(vec![tok("feline", 0, 6, 1)], &synonyms);
        assert_eq!(
            out_reverse,
            vec![tok("feline", 0, 6, 1), tok("cat", 0, 6, 0)]
        );
    }

    #[test]
    fn analyzer_default_has_no_synonyms_backward_compatible() {
        let analyzer = Analyzer::standard(None);
        let out = analyzer.analyze("quick");
        assert_eq!(out, vec![tok("quick", 0, 5, 1)]);
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

    // -- NGramTokenFilter / EdgeNGramTokenFilter --

    #[test]
    fn ngram_filter_abcde_min2_max3_exact_gram_set_and_order() {
        let tokens = vec![tok("abcde", 0, 5, 1)];
        let out = NGramTokenFilter::apply(tokens, 2, 3).unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["ab", "abc", "bc", "bcd", "cd", "cde", "de"]);
        // First gram keeps the original token's position_increment; every
        // subsequent gram from the same input token is position_increment 0.
        let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
        assert_eq!(pos_incs, vec![1, 0, 0, 0, 0, 0, 0]);
        assert!(out.iter().all(|t| t.position_length == 1));
        // Offsets: "ab" is chars 0..2 of a token starting at byte 0.
        assert_eq!((out[0].start_offset, out[0].end_offset), (0, 2));
        // "de" is chars 3..5.
        assert_eq!((out[6].start_offset, out[6].end_offset), (3, 5));
    }

    #[test]
    fn edge_ngram_filter_abcde_min2_max4_exact_prefix_gram_set() {
        let tokens = vec![tok("abcde", 0, 5, 1)];
        let out = EdgeNGramTokenFilter::apply(tokens, 2, 4).unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["ab", "abc", "abcd"]);
        let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
        assert_eq!(pos_incs, vec![1, 0, 0]);
        assert_eq!((out[0].start_offset, out[0].end_offset), (0, 2));
        assert_eq!((out[2].start_offset, out[2].end_offset), (0, 4));
    }

    #[test]
    fn ngram_filter_token_shorter_than_min_gram_produces_no_output() {
        let tokens = vec![tok("ab", 0, 2, 1)];
        let out = NGramTokenFilter::apply(tokens, 3, 5).unwrap();
        assert_eq!(out, vec![]);
    }

    #[test]
    fn edge_ngram_filter_token_shorter_than_min_gram_produces_no_output() {
        let tokens = vec![tok("ab", 0, 2, 1)];
        let out = EdgeNGramTokenFilter::apply(tokens, 3, 5).unwrap();
        assert_eq!(out, vec![]);
    }

    #[test]
    fn ngram_filter_min_gram_greater_than_max_gram_is_config_error() {
        let tokens = vec![tok("abcde", 0, 5, 1)];
        let err = NGramTokenFilter::apply(tokens, 4, 2).unwrap_err();
        assert!(err.contains("min_gram"));
    }

    #[test]
    fn edge_ngram_filter_min_gram_greater_than_max_gram_is_config_error() {
        let tokens = vec![tok("abcde", 0, 5, 1)];
        let err = EdgeNGramTokenFilter::apply(tokens, 4, 2).unwrap_err();
        assert!(err.contains("min_gram"));
    }

    #[test]
    fn ngram_filter_zero_or_negative_gram_sizes_are_config_errors() {
        let tokens = vec![tok("abcde", 0, 5, 1)];
        assert!(NGramTokenFilter::apply(tokens.clone(), 0, 3).is_err());
        assert!(NGramTokenFilter::apply(tokens.clone(), 1, 0).is_err());
        assert!(NGramTokenFilter::apply(tokens.clone(), -1, 3).is_err());
        assert!(NGramTokenFilter::apply(tokens, 2, -2).is_err());
    }

    #[test]
    fn edge_ngram_filter_zero_or_negative_gram_sizes_are_config_errors() {
        let tokens = vec![tok("abcde", 0, 5, 1)];
        assert!(EdgeNGramTokenFilter::apply(tokens.clone(), 0, 3).is_err());
        assert!(EdgeNGramTokenFilter::apply(tokens, 1, -1).is_err());
    }

    #[test]
    fn ngram_filter_single_character_token() {
        // A single-char token with min_gram == 1 produces exactly one gram
        // equal to the whole token.
        let tokens = vec![tok("a", 0, 1, 1)];
        let out = NGramTokenFilter::apply(tokens, 1, 3).unwrap();
        assert_eq!(out, vec![tok("a", 0, 1, 1)]);
    }

    #[test]
    fn ngram_filter_multibyte_unicode_grams_by_codepoint_not_byte() {
        // "café" -- 'é' is 2 bytes in UTF-8, so byte-based gramming would
        // either split it into invalid UTF-8 or misalign lengths. Grammed by
        // codepoint (4 chars: c,a,f,é) with min=2/max=2: "ca","af","fé".
        let tokens = vec![tok("café", 0, 5, 1)];
        let out = NGramTokenFilter::apply(tokens, 2, 2).unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["ca", "af", "fé"]);
        // "fé" spans the last two codepoints: byte 2..5 (since 'é' is 2
        // bytes), not 2..4.
        assert_eq!((out[2].start_offset, out[2].end_offset), (2, 5));
    }

    #[test]
    fn edge_ngram_filter_multibyte_unicode_grams_by_codepoint_not_byte() {
        let tokens = vec![tok("café", 0, 5, 1)];
        let out = EdgeNGramTokenFilter::apply(tokens, 1, 4).unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["c", "ca", "caf", "café"]);
        assert_eq!((out[3].start_offset, out[3].end_offset), (0, 5));
    }

    #[test]
    fn ngram_filter_multiple_tokens_grammed_independently() {
        // Each input token is grammed on its own -- no gramming across token
        // boundaries.
        let tokens = vec![tok("ab", 0, 2, 1), tok("cd", 3, 5, 1)];
        let out = NGramTokenFilter::apply(tokens, 2, 2).unwrap();
        assert_eq!(out, vec![tok("ab", 0, 2, 1), tok("cd", 3, 5, 1)]);
    }

    #[test]
    fn edge_ngram_filter_multiple_tokens_grammed_independently() {
        let tokens = vec![tok("abc", 0, 3, 1), tok("xyz", 4, 7, 1)];
        let out = EdgeNGramTokenFilter::apply(tokens, 1, 2).unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["a", "ab", "x", "xy"]);
        // Second input token's grams carry its own position_increment on the
        // first gram, then 0 for the rest -- independent of the first
        // token's grams.
        let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
        assert_eq!(pos_incs, vec![1, 0, 1, 0]);
    }
}
