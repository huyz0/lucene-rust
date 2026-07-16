#![forbid(unsafe_code)]
//! lucene-analysis: see /PLAN.md for scope.
//!
//! A minimal, real analyzer chain mirroring Lucene's
//! `Analyzer`/`Tokenizer`/`TokenFilter` pipeline: a simplified word-boundary
//! tokenizer (not full UAX#29 Unicode text segmentation -- see the module
//! docs on [`tokenize`]), plus `LowerCaseFilter`, `StopFilter`,
//! `AsciiFoldingFilter`, `PorterStemFilter`, and `SynonymFilter`.
//!
//! This crate sits below both `lucene-index` and `lucene-search` in the
//! workspace's downward dependency graph (it depends on nothing else in the
//! workspace), so either can depend on it without creating a cycle.

use std::collections::HashMap;
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

/// True for characters this tokenizer treats as *internal word connectors*:
/// kept as part of a token when they sit strictly between two alphanumeric
/// characters (e.g. the `.` in `"3.14"`, the `,` in `"1,000"`, the `'`/`’` in
/// `"don't"`/`"O'Brien"`), but still treated as a plain separator everywhere
/// else (e.g. a sentence-ending `"sentence."` still splits off the period,
/// since nothing alphanumeric follows it).
fn is_word_connector(c: char) -> bool {
    matches!(c, '.' | ',' | '\'' | '\u{2019}')
}

/// A simplified, real word-boundary tokenizer: splits on whitespace and
/// punctuation boundaries, keeping maximal runs of alphanumeric characters
/// (Unicode alphanumeric, via `char::is_alphanumeric`) as terms, plus three
/// narrowly-scoped UAX#29-inspired extensions (see [`is_word_connector`]):
///
/// - **Embedded numeric punctuation**: `.`/`,` between two alphanumeric
///   characters stays part of the token, so `"3.14"` and `"1,000"` are each
///   one token rather than splitting at the punctuation.
/// - **Acronym-style internal periods**: the same rule means `"U.S.A."`
///   tokenizes as `"U.S.A"` (the trailing period, followed by nothing
///   alphanumeric, is still dropped) rather than three separate
///   single-letter tokens.
/// - **Internal apostrophes**: `'`/`’` between two alphanumeric characters
///   stays part of the token, so `"don't"` and `"O'Brien"` survive as single
///   tokens instead of splitting at the apostrophe.
///
/// A connector followed by whitespace, end-of-text, or any other
/// non-alphanumeric character is *not* absorbed -- e.g. a real
/// sentence-ending period (`"end of sentence."`) still splits the trailing
/// `.` off, since nothing alphanumeric follows it.
///
/// This mirrors the *core algorithm* of real Lucene's `StandardTokenizer`/
/// `WhitespaceTokenizer` -- split on non-alphanumeric boundaries, with a
/// small set of hand-picked exceptions -- but is **not** a port of full
/// UAX#29 Unicode Text Segmentation (which additionally handles combining
/// marks, locale-specific word breaking, CJK/complex script segmentation,
/// and full email/URL recognition). That's substantial, legitimately
/// out-of-scope NLP machinery; see `docs/parity.md` for the explicit scope
/// note.
///
/// Every token gets `position_increment == 1` (tokenizers never skip
/// positions -- that only happens in filters, e.g. [`StopFilter`]).
pub fn tokenize(text: &str) -> Vec<Token> {
    let chars: Vec<(usize, char)> = text.char_indices().collect();
    let n = chars.len();
    let mut tokens = Vec::new();
    let mut i = 0;
    while i < n {
        let (start, ch) = chars[i];
        if !ch.is_alphanumeric() {
            i += 1;
            continue;
        }
        // `last_included` tracks the index of the last char absorbed into
        // this token; it only ever advances over alphanumeric chars or
        // connectors that are themselves followed by another alphanumeric
        // char, so a token never ends on a bare connector.
        let mut last_included = i;
        let mut j = i + 1;
        while j < n {
            let (_, c) = chars[j];
            if c.is_alphanumeric() {
                last_included = j;
                j += 1;
                continue;
            }
            if is_word_connector(c) && j + 1 < n && chars[j + 1].1.is_alphanumeric() {
                // Absorb the connector provisionally; `last_included` is
                // only bumped to it once the following alphanumeric char is
                // also absorbed (next loop iteration), so a trailing
                // connector at the very end of the run is never included.
                j += 1;
                continue;
            }
            break;
        }
        let (end_start, end_ch) = chars[last_included];
        let end_offset = end_start + end_ch.len_utf8();
        tokens.push(Token {
            term: text[start..end_offset].to_string(),
            start_offset: start as i32,
            end_offset: end_offset as i32,
            position_increment: 1,
        });
        i = last_included + 1;
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
}
