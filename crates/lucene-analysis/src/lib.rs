#![forbid(unsafe_code)]
//! lucene-analysis: Lucene's analysis model (`org.apache.lucene.analysis`,
//! `.standard`, `.tokenattributes`), plus a few analysis-common filters.
//!
//! # The model
//!
//! - [`TokenStream`] -- Java's streaming lifecycle (`reset`,
//!   `increment_token`, `end`, `close`) over one shared [`AttributeSource`]
//!   carrying every core attribute ([`attributes`] explains the fixed-struct
//!   shape).
//! - [`Tokenizer`] (reads a [`CharReader`], corrects offsets through
//!   [`CharFilter`]s), [`TokenFilter`] (implement it to write a filter; chains
//!   compose by ownership), [`FilteringTokenFilter`], [`CachingTokenFilter`],
//!   [`GraphTokenFilter`].
//! - [`Analyzer`] over an [`AnalyzerDefinition`] (`createComponents`,
//!   `normalize`, `initReader`, the gaps), with [`ReuseStrategy`],
//!   [`AnalyzerWrapper`] and [`DelegatingAnalyzerWrapper`];
//!   [`StopwordAnalyzerBase`], [`CharArraySet`], [`wordlist_loader`].
//! - [`StandardTokenizer`] -- Lucene's JFlex UAX#29 scanner, tables and all --
//!   and [`StandardAnalyzer`].
//! - [`token_stream_to_automaton`] / [`automaton_to_token_stream`].
//! - [`factory`] -- the factory SPI (`TokenizerFactory.forName("standard", args)`
//!   and the rest) and [`factory::CustomAnalyzer`]: analyzers from Java's
//!   configuration text.
//!
//! Offsets everywhere are **UTF-16 code units** (Java `char` indices); see
//! [`Token`] and [`reader`].
//!
//! The older materialised API -- [`tokenize`], [`Token`], [`AnalyzedTokens`],
//! the `Vec<Token>` filter functions and [`Analyzer::standard`]'s builders --
//! is kept for the workspace's callers and now runs on the streaming model.
//!
//! This crate sits below both `lucene-index` and `lucene-search` in the
//! workspace's downward dependency graph (its only workspace dependency is
//! `lucene-util`, for the automaton type the graph converters build), so
//! either can depend on it without creating a cycle.

use std::collections::HashMap;
use std::collections::HashSet;
use std::sync::Arc;

/// The `input`/`input_mut` half of a [`TokenFilter`] impl whose input is the
/// field `input` of type `I`.
macro_rules! filter_input {
    () => {
        type Input = I;
        fn input(&self) -> &I {
            &self.input
        }
        fn input_mut(&mut self) -> &mut I {
            &mut self.input
        }
    };
}
pub(crate) use filter_input;

mod analyzer;
pub mod attributes;
mod automaton;
pub mod boost;
mod char_array_set;
pub mod charfilter;
pub mod cjk;
pub mod classic;
pub mod commongrams;
pub mod compound;
pub mod core_analysis;
pub mod data_pack;
pub mod email;
pub mod en;
pub mod factory;
pub mod graph_finite_strings;
mod graph_token_filter;
pub mod hunspell;
pub mod java_character;
mod java_float;
pub mod java_unicode_script;
#[rustfmt::skip]
mod java_character_tables;
mod keyword_tokenizer;
pub mod lang;
mod legacy;
pub mod minhash;
pub mod miscellaneous;
pub mod morph;
pub mod ngram;
pub mod path;
pub mod pattern;
pub mod payloads;
pub mod reader;
pub mod reverse;
pub mod shingle;
pub mod sinks;
pub mod snowball;
pub mod standard;
mod stopword_analyzer_base;
pub mod synonym;
pub mod token_stream;
pub mod util;
pub mod wikipedia;
pub mod wordlist_loader;

pub use analyzer::{
    collect_tokens, Analyzer, AnalyzerDefinition, AnalyzerTokenStream, AnalyzerWrapper,
    DelegatingAnalyzerWrapper, ReuseStrategy, SourceFn, TokenStreamComponents,
};
pub use attributes::{AttrValue, AttributeSource, CustomAttribute, State, StateSlot, DEFAULT_TYPE};
pub use automaton::{
    automaton_to_token_stream, token_stream_to_automaton, Automaton, AutomatonBuilder,
    TokenStreamToAutomaton, Transition, HOLE, POS_SEP,
};
pub use char_array_set::CharArraySet;
pub use graph_finite_strings::{FiniteStringsTokenStream, GraphTokenStreamFiniteStrings};
pub use graph_token_filter::{GraphTokenFilter, MAX_GRAPH_STACK_SIZE, MAX_TOKEN_CACHE_SIZE};
pub use keyword_tokenizer::KeywordTokenizer;
pub use reader::{CharFilter, CharReader, StrReader, UnitsReader};
pub use standard::{StandardAnalyzer, StandardTokenizer};
pub use stopword_analyzer_base::StopwordAnalyzerBase;
pub use token_stream::{
    Accept, CachingTokenFilter, FilteringTokenFilter, TokenFilter, TokenStream, Tokenizer,
    TokenizerInput,
};

/// What Java analysis throws, as a value.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum AnalysisError {
    /// `IllegalArgumentException`: a bad setting or attribute value.
    #[error("illegal argument: {0}")]
    IllegalArgument(String),
    /// `IllegalStateException`: a broken `TokenStream` contract.
    #[error("illegal state: {0}")]
    IllegalState(String),
    /// `AlreadyClosedException`.
    #[error("already closed: {0}")]
    AlreadyClosed(String),
    /// `IOException` from a reader.
    #[error("i/o: {0}")]
    Io(String),
    /// `UnsupportedOperationException`: an operation this port refuses by
    /// design (`ThaiTokenizer` without an installed Thai word iterator, see
    /// `lang::th`).
    #[error("unsupported operation: {0}")]
    UnsupportedOperation(String),
}

/// One analyzed token: term text plus the attributes real Lucene's
/// `CharTermAttribute`/`OffsetAttribute`/`PositionIncrementAttribute` carry.
///
/// **`start_offset`/`end_offset` are UTF-16 code-unit offsets into the
/// original text** -- Java `char` indices, exactly what real Lucene's
/// `OffsetAttribute.startOffset()`/`endOffset()` report, and exactly what
/// `text.substring(start, end)` slices with on the JVM side. They are *not*
/// UTF-8 byte offsets and *not* Unicode scalar (`char` in Rust) counts; the
/// three units coincide only for pure-ASCII text, differ for any non-Latin
/// text, and all three differ for supplementary-plane text (an emoji is 1
/// scalar, 2 UTF-16 code units and 4 UTF-8 bytes).
///
/// This is a real unit, not a convention: these offsets are written verbatim
/// into `.pos`/`.pay`/`.tvd` by `lucene-index`'s `indexing_chain` +
/// `IndexWriter`, read back by real Lucene as `startOffset()`/`endOffset()`,
/// and consumed by `lucene-search`'s highlighter, which slices Java `char`
/// spans. `CheckIndex` never compares an offset against the text it indexes,
/// so nothing but a differential fixture catches a wrong unit here -- see
/// `crates/lucene-analysis/tests/analysis_fixtures.rs`, whose `utf16_*`
/// cases pin every producer against a real Lucene analyzer over text that
/// separates all three units.
///
/// Every filter in this crate that changes a term's length
/// ([`AsciiFoldingFilter`], [`PorterStemFilter`],
/// [`SnowballEnglishStemFilter`], [`LowerCaseFilter`]) leaves the *input
/// token's* offsets untouched, as Java's do -- an offset span always refers
/// to the original source text, never to the rewritten term.
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

/// A whole analyzed field value: the tokens, plus the two values real
/// Lucene's `TokenStream.end()` leaves behind in the attribute source.
///
/// # Why this type exists
///
/// Java's analysis is a *stream* with a lifecycle -- `reset()`,
/// `incrementToken()*`, `end()`, `close()` -- and `end()` is not a formality.
/// `IndexingChain.PerField.invertTokenStream` reads two attributes after it:
///
/// ```java
/// stream.end();
/// invertState.position += invertState.posIncrAttribute.getPositionIncrement();
/// invertState.offset += invertState.offsetAttribute.endOffset();
/// ```
///
/// This crate's filters are `Vec<Token> -> Vec<Token>` functions, which is the
/// right Rust shape for the token sequence but has nowhere to put those two
/// end-of-stream values -- so they were simply dropped, and the two things
/// they decide were wrong:
///
/// 1. **A document whose last tokens were all filtered out did not advance the
///    position counter.** `FilteringTokenFilter.end()` adds its
///    `skippedPositions` to the increment, so `"fox the the"` leaves the
///    field's position counter two past `fox`, not on it. Invisible in a
///    single-valued field, and the whole story in a multi-valued one.
/// 2. **The next value of a multi-valued field started at offset 0**, because
///    `finalOffset` -- the value's own length -- is what `invertState.offset`
///    accumulates.
///
/// [`final_position_increment`](Self::final_position_increment) and
/// [`final_offset`](Self::final_offset) are exactly those two attribute reads.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AnalyzedTokens {
    /// The tokens `incrementToken()` produced, in order.
    pub tokens: Vec<Token>,
    /// `PositionIncrementAttribute.getPositionIncrement()` **after** `end()`.
    ///
    /// `TokenStream.end()` (the base implementation) clears the attributes and
    /// sets this to `0`; each filter that swallowed positions adds them back
    /// on the way out. So it is `0` for a stream nothing filtered, and the
    /// number of positions the trailing filtered-out tokens occupied
    /// otherwise.
    pub final_position_increment: i32,
    /// `OffsetAttribute.endOffset()` **after** `end()` -- Java's
    /// `Tokenizer.end()`'s `finalOffset = correctOffset(charCount)`, i.e. the
    /// length of the whole input in **UTF-16 code units** (see [`Token`] for
    /// the unit), not the end of the last token.
    pub final_offset: i32,
}

/// Lucene's `StandardTokenizer` (default `maxTokenLength` 255) over `text`,
/// materialised: the term text, offsets, position increments and lengths of
/// every token. See [`StandardTokenizer`] for the tokenizer itself (and its
/// token types, which [`Token`] does not carry).
///
/// The scanner is Lucene's own, so emoji (`<EMOJI>` tokens, ZWJ sequences
/// whole), Southeast Asian runs and `maxTokenLength` splitting all match.
///
/// Every token gets `position_increment == 1` unless a token longer than the
/// maximum is skipped (see [`StandardTokenizer`]); offsets are UTF-16 code
/// units (see [`Token`]).
pub fn tokenize(text: &str) -> Vec<Token> {
    tokenize_stream(text).tokens
}

/// [`tokenize`] as a whole stream, i.e. with `Tokenizer.end()` run:
/// [`AnalyzedTokens::final_offset`] is Java's `finalOffset =
/// correctOffset(charCount)` -- the length of the **whole input** in UTF-16
/// code units, not the end of the last token (`"fox   "` ends at 6, not 3).
pub fn tokenize_stream(text: &str) -> AnalyzedTokens {
    let mut tok = StandardTokenizer::new();
    tok.set_reader(Box::new(StrReader::new(text)))
        .and_then(|()| collect_tokens(&mut tok))
        .expect("StandardTokenizer over an in-memory string cannot fail")
}

/// Java's `String.length()` for the same text: the number of UTF-16 code
/// units `s` encodes to, i.e. one per BMP scalar and two per
/// supplementary-plane scalar. ASCII (the common case) is its byte length.
#[inline]
pub(crate) fn utf16_len(s: &str) -> usize {
    if s.is_ascii() {
        return s.len();
    }
    // One unit per scalar (every byte that is not a continuation byte,
    // `10xxxxxx`) and a second for a supplementary one (a four-byte lead,
    // `0xf0` and up): the byte length less the continuation bytes plus the
    // four-byte leads, counted eight bytes to a step. Each step adds a 0/1 per
    // byte lane; the lanes are summed before any can pass 31, so that their
    // total fits one byte.
    const LANES: u64 = 0x0101_0101_0101_0101;
    let bytes = s.as_bytes();
    let (mut continuations, mut four_byte_leads) = (0usize, 0usize);
    for block in bytes.chunks(31 * 8) {
        let (mut cont_lanes, mut lead_lanes) = (0u64, 0u64);
        let mut words = block.chunks_exact(8);
        for w in &mut words {
            let w = u64::from_le_bytes([w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7]]);
            cont_lanes = cont_lanes.wrapping_add((w >> 7) & !(w >> 6) & LANES);
            lead_lanes = lead_lanes.wrapping_add((w >> 4) & (w >> 5) & (w >> 6) & (w >> 7) & LANES);
        }
        continuations += (cont_lanes.wrapping_mul(LANES) >> 56) as usize;
        four_byte_leads += (lead_lanes.wrapping_mul(LANES) >> 56) as usize;
        for &b in words.remainder() {
            continuations += usize::from(b & 0xc0 == 0x80);
            four_byte_leads += usize::from(b >= 0xf0);
        }
    }
    bytes.len() - continuations + four_byte_leads
}

/// Lucene's `org.apache.lucene.analysis.LowerCaseFilter`: lowercases each
/// token's term with Java's simple per-code-point mapping
/// ([`simple_to_lowercase`]), leaving every other attribute untouched.
pub struct LowerCaseFilter<I = Box<dyn TokenStream>> {
    input: I,
    /// The lowercased term is built here and swapped in, so a token that
    /// changes case costs no allocation.
    scratch: String,
}

/// Java's `CharacterUtils.toLowerCase` applied to one codepoint.
///
/// That method is
/// `Character.toChars(Character.toLowerCase(codePointAt(...)))` written
/// **back into the same buffer at the same index**, so it is the *simple*,
/// strictly 1:1 case mapping -- not the full mapping with its expansions and
/// context rules (`İ` lowercases to `i`, not `i` + U+0307; `"ΟΔΟΣ"` to
/// `"οδοσ"`, no final sigma).
///
/// The mapping is [`java_character::to_lower_case`]: JDK 25's
/// `Character.toLowerCase` (Unicode 16.0), the JDK OpenSearch 3.9.0 bundles,
/// read from the same table `lucene-util`'s `CaseFolding` uses. Rust's
/// `char::to_lowercase` follows the toolchain's Unicode version (17.0 on
/// Rust 1.97) and lowercased 28 code points Unicode 16 leaves unassigned.
#[inline]
pub(crate) fn simple_to_lowercase(c: char) -> char {
    if c.is_ascii() {
        return c.to_ascii_lowercase();
    }
    // The BMP's table lookup is cheaper than the block test.
    if u32::from(c) >= 0x10000 && is_caseless_block(c) {
        return c;
    }
    char::from_u32(java_character::to_lower_case(u32::from(c))).unwrap_or(c)
}

/// Blocks with no case mappings at all (Thai/Lao, CJK punctuation through
/// the unified ideographs, Hangul syllables, emoji and the supplementary
/// ideographic planes): the table search would only return the character
/// itself. `caseless_blocks_have_no_lowercase`
/// checks every code point in them.
#[inline]
fn is_caseless_block(c: char) -> bool {
    matches!(c as u32, 0x0E00..=0x0EFF | 0x3000..=0x9FFF | 0xAC00..=0xD7AF | 0x1F000..=0x1FBFF | 0x20000..=0x3FFFF)
}

/// `CharacterUtils.toLowerCase` over a whole term, in place; a term that
/// lowercasing leaves unchanged (most non-Latin text) is not rebuilt.
fn lowercase_term(term: &mut String) {
    lowercase_term_with(term, &mut String::new());
}

/// The byte index of the first code point of `b` (valid UTF-8) that
/// [`simple_to_lowercase`] changes.
fn first_to_lowercase(b: &[u8]) -> Option<usize> {
    let mut i = 0;
    while i < b.len() {
        let x = u32::from(b[i]);
        if x < 0x80 {
            if b[i].is_ascii_uppercase() {
                return Some(i);
            }
            i += 1;
            continue;
        }
        let cont = |k: usize| u32::from(b[i + k] & 0x3F);
        let (cp, w) = if x < 0xE0 {
            (((x & 0x1F) << 6) | cont(1), 2)
        } else if x < 0xF0 {
            (((x & 0x0F) << 12) | (cont(1) << 6) | cont(2), 3)
        } else {
            (
                ((x & 0x07) << 18) | (cont(1) << 12) | (cont(2) << 6) | cont(3),
                4,
            )
        };
        let changes = if cp < 0x10000 {
            java_character::to_lower_case(cp) != cp
        } else {
            char::from_u32(cp).is_some_and(|c| simple_to_lowercase(c) != c)
        };
        if changes {
            return Some(i);
        }
        i += w;
    }
    None
}

/// [`lowercase_term`]: a code point whose lowercase has the same UTF-8
/// width is rewritten in place (Java's in-place `Character.toChars` write);
/// from the first one that does not, the rest is built in `scratch` and
/// swapped in.
fn lowercase_term_with(term: &mut String, scratch: &mut String) {
    // Most terms need no change: find the first unit that does, decoding
    // the UTF-8 by hand.
    let Some(mut i) = first_to_lowercase(term.as_bytes()) else {
        return;
    };
    while i < term.len() {
        if term.as_bytes()[i] < 0x80 {
            term[i..i + 1].make_ascii_lowercase();
            i += 1;
            continue;
        }
        let Some(c) = term[i..].chars().next() else {
            break;
        };
        let w = c.len_utf8();
        let l = simple_to_lowercase(c);
        if l != c {
            if l.len_utf8() != w {
                scratch.clear();
                scratch.push_str(&term[..i]);
                scratch.extend(term[i..].chars().map(simple_to_lowercase));
                std::mem::swap(term, scratch);
                return;
            }
            term.replace_range(i..i + w, l.encode_utf8(&mut [0; 4]));
        }
        i += w;
    }
}

impl<I: TokenStream> LowerCaseFilter<I> {
    /// `new LowerCaseFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        LowerCaseFilter {
            input,
            scratch: String::new(),
        }
    }
}

impl LowerCaseFilter {
    /// The filter over a materialised token list.
    pub fn apply(tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                lowercase_term(&mut t.term);
                t
            })
            .collect()
    }
}

impl<I: TokenStream> TokenFilter for LowerCaseFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.input.increment_token()? {
            lowercase_term_with(self.input.attributes_mut().term_mut(), &mut self.scratch);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// Lucene's `org.apache.lucene.analysis.StopFilter`: a
/// [`FilteringTokenFilter`] dropping the tokens whose term is in a
/// [`CharArraySet`].
///
/// Position-increment preservation (real Lucene semantics, not "just drop
/// the removed token"): a removed stopword's own `position_increment` is
/// *not* discarded -- it is added onto the position increment of the next
/// surviving token, so the position gap it would have occupied is preserved,
/// and trailing removed stopwords add theirs to the increment `end()`
/// reports. If the text is nothing but stopwords, the output is empty.
pub struct StopFilter<I = Box<dyn TokenStream>> {
    inner: FilteringTokenFilter<I, StopWords>,
}

/// `StopFilter.accept()`: `!stopWords.contains(term)`.
pub struct StopWords(Arc<CharArraySet>);

impl Accept for StopWords {
    fn accept(&mut self, attributes: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok(!self.0.contains(attributes.term()))
    }
}

impl<I: TokenStream> StopFilter<I> {
    /// `new StopFilter(TokenStream, CharArraySet)`.
    pub fn new(input: I, stop_words: Arc<CharArraySet>) -> Self {
        StopFilter {
            inner: FilteringTokenFilter::new(input, StopWords(stop_words)),
        }
    }
}

impl<I: TokenStream> TokenFilter for StopFilter<I> {
    type Input = FilteringTokenFilter<I, StopWords>;
    fn input(&self) -> &Self::Input {
        &self.inner
    }
    fn input_mut(&mut self) -> &mut Self::Input {
        &mut self.inner
    }
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        self.inner.increment_token()
    }
}

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
pub const ENGLISH_STOP_WORDS: [&str; 33] = [
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

/// The default French stop word list, byte-for-byte the same 154 words as
/// real Lucene's `org.apache.lucene.analysis.fr.FrenchAnalyzer.getDefaultStopSet()`
/// (loaded there from `french_stop.txt`, a resource under
/// `org/apache/lucene/analysis/snowball/` -- itself sourced verbatim from the
/// Snowball project's French stop list, <https://snowballstem.org/algorithms/french/stop.txt>).
/// This list was transcribed directly from that resource file as it exists in
/// a real `/home/tuong/work/lucene` 10.5.0 checkout
/// (`lucene/analysis/common/src/resources/org/apache/lucene/analysis/snowball/french_stop.txt`),
/// stripping the file's `|`-delimited comments and blank lines the same way
/// real Lucene's `WordlistLoader.getSnowballWordSet` does (split remaining
/// text on whitespace, one word per surviving line) -- not reconstructed from
/// memory. Stored lowercase, matching real Lucene's `CharArraySet` there
/// (built with `ignoreCase == false`, populated with already-lowercase
/// entries) and this port's [`StopFilter`], which does a plain, exact
/// (case-sensitive) string match against terms that have already passed
/// through [`LowerCaseFilter`] earlier in the chain -- see
/// [`french_stop_words`].
///
/// **Scope boundary, stated explicitly**: this is *only* the default
/// stopword list real `FrenchAnalyzer` uses, not a port of `FrenchAnalyzer`
/// itself. Real Lucene's `FrenchAnalyzer` also performs French elision
/// filtering (`ElisionFilter` with `DEFAULT_ARTICLES` -- stripping leading
/// `l'`/`d'`/`qu'`/etc. from words like `l'homme` -> `homme`) and French
/// stemming (`FrenchLightStemmer` via `SnowballFilter`/`FrenchLightStemFilter`).
/// Neither elision nor stemming is ported here or anywhere else in this
/// crate -- this constant and [`french_stop_words`] exist solely so a caller
/// can feed this port's existing, language-agnostic [`StopFilter`] a real
/// French default stopword set, the same way [`ENGLISH_STOP_WORDS`] feeds it
/// an English one. A composed `Analyzer::french()`-style helper mirroring the
/// full `FrenchAnalyzer` pipeline (tokenize -> elide -> lowercase -> stopword
/// -> stem) is not provided.
pub const FRENCH_STOP_WORDS: [&str; 154] = [
    "au", "aux", "avec", "ce", "ces", "dans", "de", "des", "du", "elle", "en", "et", "eux", "il",
    "je", "la", "le", "leur", "lui", "ma", "mais", "me", "même", "mes", "moi", "mon", "ne", "nos",
    "notre", "nous", "on", "ou", "par", "pas", "pour", "qu", "que", "qui", "sa", "se", "ses",
    "sur", "ta", "te", "tes", "toi", "ton", "tu", "un", "une", "vos", "votre", "vous", "c", "d",
    "j", "l", "à", "m", "n", "s", "t", "y", "étée", "étées", "étant", "suis", "es", "êtes", "sont",
    "serai", "seras", "sera", "serons", "serez", "seront", "serais", "serait", "serions", "seriez",
    "seraient", "étais", "était", "étions", "étiez", "étaient", "fus", "fut", "fûmes", "fûtes",
    "furent", "sois", "soit", "soyons", "soyez", "soient", "fusse", "fusses", "fussions",
    "fussiez", "fussent", "ayant", "eu", "eue", "eues", "eus", "ai", "avons", "avez", "ont",
    "aurai", "aurons", "aurez", "auront", "aurais", "aurait", "aurions", "auriez", "auraient",
    "avais", "avait", "aviez", "avaient", "eut", "eûmes", "eûtes", "eurent", "aie", "aies", "ait",
    "ayons", "ayez", "aient", "eusse", "eusses", "eût", "eussions", "eussiez", "eussent", "ceci",
    "cela", "celà", "cet", "cette", "ici", "ils", "les", "leurs", "quel", "quels", "quelle",
    "quelles", "sans", "soi",
];

/// Builds a fresh `HashSet<String>` from [`FRENCH_STOP_WORDS`], ready to pass
/// to [`StopFilter::apply`], mirroring real Lucene's
/// `FrenchAnalyzer.getDefaultStopSet()` default. See [`FRENCH_STOP_WORDS`]'s
/// doc comment for the sourcing and the explicit scope boundary (list only,
/// no elision, no stemming). Not a `static`/`OnceLock`-cached singleton, for
/// the same reason [`english_stop_words`] isn't.
pub fn french_stop_words() -> HashSet<String> {
    FRENCH_STOP_WORDS.iter().map(|s| s.to_string()).collect()
}

impl StopFilter {
    pub fn apply(tokens: Vec<Token>, stopwords: &HashSet<String>) -> Vec<Token> {
        Self::apply_to_stream(
            AnalyzedTokens {
                tokens,
                final_position_increment: 0,
                final_offset: 0,
            },
            stopwords,
        )
        .tokens
    }

    /// [`Self::apply`] with `FilteringTokenFilter`'s **`end()`** as well as its
    /// `incrementToken()`.
    ///
    /// ```java
    /// public void end() throws IOException {
    ///   super.end();
    ///   posIncrAtt.setPositionIncrement(posIncrAtt.getPositionIncrement() + skippedPositions);
    /// }
    /// ```
    ///
    /// `skippedPositions` at end of stream is what the trailing stopwords in
    /// `"fox the the"` left behind: `apply` drops them with nowhere to put
    /// their increments, and this carries them out on
    /// [`AnalyzedTokens::final_position_increment`], which is what
    /// `IndexingChain` adds to the field's position counter.
    pub fn apply_to_stream(stream: AnalyzedTokens, stopwords: &HashSet<String>) -> AnalyzedTokens {
        let AnalyzedTokens {
            tokens,
            final_position_increment,
            final_offset,
        } = stream;
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
        AnalyzedTokens {
            tokens: out,
            final_position_increment: final_position_increment + pending_increment,
            final_offset,
        }
    }
}

/// Real Lucene's `org.apache.lucene.analysis.miscellaneous.ASCIIFoldingFilter`:
/// folds accented/diacritic Latin characters to their closest plain-ASCII
/// equivalent, leaving offsets and position increments untouched.
///
/// **Coverage: the whole table, not a subset.** Every codepoint real
/// `ASCIIFoldingFilter.foldToASCII` rewrites is in [`FOLD_TABLE`] -- all 1242
/// of them, generated from the real filter rather than hand-transcribed. That
/// spans Latin-1 Supplement, Latin Extended-A/B, IPA Extensions, Phonetic
/// Extensions, Latin Extended Additional (the precomposed Vietnamese
/// letters), General Punctuation and superscripts/subscripts, Letterlike
/// Symbols, Enclosed Alphanumerics, Dingbats, Latin Extended-C/D,
/// Supplemental Punctuation, Alphabetic Presentation Forms (the `ﬁ`/`ﬂ`
/// ligatures) and Halfwidth/Fullwidth Forms -- the same list this filter's
/// own javadoc enumerates.
///
/// A codepoint outside the table passes through unchanged (never dropped,
/// never a panic), which is also what Lucene does.
///
/// **Offsets are never adjusted for folding-driven length changes**: folding
/// `æ` -> `"ae"` grows a token's character count, but `start_offset`/
/// `end_offset` still refer to the *original* source text span -- this
/// matches real Lucene's `ASCIIFoldingFilter`, which does not touch
/// `OffsetAttribute` at all.
pub struct AsciiFoldingFilter;

/// Real Lucene's `ASCIIFoldingFilter.foldToASCII` table, in full: every
/// BMP codepoint that filter maps to something other than itself, paired
/// with its ASCII replacement, sorted by codepoint so [`AsciiFoldingFilter::fold_char`] can
/// binary-search it.
///
/// Generated by folding every codepoint `0..0x10000` through the real
/// `ASCIIFoldingFilter.foldToASCII(char[], int, char[], int, int)` from
/// `lucene-analysis-common-10.5.0.jar` and keeping the ones that changed:
/// **1242 entries**, which is the same count as the `case '\uXXXX'` labels
/// in that method's switch. `fixtures/src/GenAnalysis.java`'s
/// `ascii_folding_table` case re-derives it at fixture-generation time and
/// `crates/lucene-analysis/tests/analysis_fixtures.rs` asserts this table
/// reproduces it entry for entry, so it cannot drift silently.
///
/// The filter's own `char[]`-based signature means the table is BMP-only;
/// no supplementary codepoint folds in Lucene either.
static FOLD_TABLE: [(char, &str); 1242] = [
    ('\u{00AB}', "\""),
    ('\u{00B2}', "2"),
    ('\u{00B3}', "3"),
    ('\u{00B9}', "1"),
    ('\u{00BB}', "\""),
    ('\u{00C0}', "A"),
    ('\u{00C1}', "A"),
    ('\u{00C2}', "A"),
    ('\u{00C3}', "A"),
    ('\u{00C4}', "A"),
    ('\u{00C5}', "A"),
    ('\u{00C6}', "AE"),
    ('\u{00C7}', "C"),
    ('\u{00C8}', "E"),
    ('\u{00C9}', "E"),
    ('\u{00CA}', "E"),
    ('\u{00CB}', "E"),
    ('\u{00CC}', "I"),
    ('\u{00CD}', "I"),
    ('\u{00CE}', "I"),
    ('\u{00CF}', "I"),
    ('\u{00D0}', "D"),
    ('\u{00D1}', "N"),
    ('\u{00D2}', "O"),
    ('\u{00D3}', "O"),
    ('\u{00D4}', "O"),
    ('\u{00D5}', "O"),
    ('\u{00D6}', "O"),
    ('\u{00D8}', "O"),
    ('\u{00D9}', "U"),
    ('\u{00DA}', "U"),
    ('\u{00DB}', "U"),
    ('\u{00DC}', "U"),
    ('\u{00DD}', "Y"),
    ('\u{00DE}', "TH"),
    ('\u{00DF}', "ss"),
    ('\u{00E0}', "a"),
    ('\u{00E1}', "a"),
    ('\u{00E2}', "a"),
    ('\u{00E3}', "a"),
    ('\u{00E4}', "a"),
    ('\u{00E5}', "a"),
    ('\u{00E6}', "ae"),
    ('\u{00E7}', "c"),
    ('\u{00E8}', "e"),
    ('\u{00E9}', "e"),
    ('\u{00EA}', "e"),
    ('\u{00EB}', "e"),
    ('\u{00EC}', "i"),
    ('\u{00ED}', "i"),
    ('\u{00EE}', "i"),
    ('\u{00EF}', "i"),
    ('\u{00F0}', "d"),
    ('\u{00F1}', "n"),
    ('\u{00F2}', "o"),
    ('\u{00F3}', "o"),
    ('\u{00F4}', "o"),
    ('\u{00F5}', "o"),
    ('\u{00F6}', "o"),
    ('\u{00F8}', "o"),
    ('\u{00F9}', "u"),
    ('\u{00FA}', "u"),
    ('\u{00FB}', "u"),
    ('\u{00FC}', "u"),
    ('\u{00FD}', "y"),
    ('\u{00FE}', "th"),
    ('\u{00FF}', "y"),
    ('\u{0100}', "A"),
    ('\u{0101}', "a"),
    ('\u{0102}', "A"),
    ('\u{0103}', "a"),
    ('\u{0104}', "A"),
    ('\u{0105}', "a"),
    ('\u{0106}', "C"),
    ('\u{0107}', "c"),
    ('\u{0108}', "C"),
    ('\u{0109}', "c"),
    ('\u{010A}', "C"),
    ('\u{010B}', "c"),
    ('\u{010C}', "C"),
    ('\u{010D}', "c"),
    ('\u{010E}', "D"),
    ('\u{010F}', "d"),
    ('\u{0110}', "D"),
    ('\u{0111}', "d"),
    ('\u{0112}', "E"),
    ('\u{0113}', "e"),
    ('\u{0114}', "E"),
    ('\u{0115}', "e"),
    ('\u{0116}', "E"),
    ('\u{0117}', "e"),
    ('\u{0118}', "E"),
    ('\u{0119}', "e"),
    ('\u{011A}', "E"),
    ('\u{011B}', "e"),
    ('\u{011C}', "G"),
    ('\u{011D}', "g"),
    ('\u{011E}', "G"),
    ('\u{011F}', "g"),
    ('\u{0120}', "G"),
    ('\u{0121}', "g"),
    ('\u{0122}', "G"),
    ('\u{0123}', "g"),
    ('\u{0124}', "H"),
    ('\u{0125}', "h"),
    ('\u{0126}', "H"),
    ('\u{0127}', "h"),
    ('\u{0128}', "I"),
    ('\u{0129}', "i"),
    ('\u{012A}', "I"),
    ('\u{012B}', "i"),
    ('\u{012C}', "I"),
    ('\u{012D}', "i"),
    ('\u{012E}', "I"),
    ('\u{012F}', "i"),
    ('\u{0130}', "I"),
    ('\u{0131}', "i"),
    ('\u{0132}', "IJ"),
    ('\u{0133}', "ij"),
    ('\u{0134}', "J"),
    ('\u{0135}', "j"),
    ('\u{0136}', "K"),
    ('\u{0137}', "k"),
    ('\u{0138}', "q"),
    ('\u{0139}', "L"),
    ('\u{013A}', "l"),
    ('\u{013B}', "L"),
    ('\u{013C}', "l"),
    ('\u{013D}', "L"),
    ('\u{013E}', "l"),
    ('\u{013F}', "L"),
    ('\u{0140}', "l"),
    ('\u{0141}', "L"),
    ('\u{0142}', "l"),
    ('\u{0143}', "N"),
    ('\u{0144}', "n"),
    ('\u{0145}', "N"),
    ('\u{0146}', "n"),
    ('\u{0147}', "N"),
    ('\u{0148}', "n"),
    ('\u{0149}', "n"),
    ('\u{014A}', "N"),
    ('\u{014B}', "n"),
    ('\u{014C}', "O"),
    ('\u{014D}', "o"),
    ('\u{014E}', "O"),
    ('\u{014F}', "o"),
    ('\u{0150}', "O"),
    ('\u{0151}', "o"),
    ('\u{0152}', "OE"),
    ('\u{0153}', "oe"),
    ('\u{0154}', "R"),
    ('\u{0155}', "r"),
    ('\u{0156}', "R"),
    ('\u{0157}', "r"),
    ('\u{0158}', "R"),
    ('\u{0159}', "r"),
    ('\u{015A}', "S"),
    ('\u{015B}', "s"),
    ('\u{015C}', "S"),
    ('\u{015D}', "s"),
    ('\u{015E}', "S"),
    ('\u{015F}', "s"),
    ('\u{0160}', "S"),
    ('\u{0161}', "s"),
    ('\u{0162}', "T"),
    ('\u{0163}', "t"),
    ('\u{0164}', "T"),
    ('\u{0165}', "t"),
    ('\u{0166}', "T"),
    ('\u{0167}', "t"),
    ('\u{0168}', "U"),
    ('\u{0169}', "u"),
    ('\u{016A}', "U"),
    ('\u{016B}', "u"),
    ('\u{016C}', "U"),
    ('\u{016D}', "u"),
    ('\u{016E}', "U"),
    ('\u{016F}', "u"),
    ('\u{0170}', "U"),
    ('\u{0171}', "u"),
    ('\u{0172}', "U"),
    ('\u{0173}', "u"),
    ('\u{0174}', "W"),
    ('\u{0175}', "w"),
    ('\u{0176}', "Y"),
    ('\u{0177}', "y"),
    ('\u{0178}', "Y"),
    ('\u{0179}', "Z"),
    ('\u{017A}', "z"),
    ('\u{017B}', "Z"),
    ('\u{017C}', "z"),
    ('\u{017D}', "Z"),
    ('\u{017E}', "z"),
    ('\u{017F}', "s"),
    ('\u{0180}', "b"),
    ('\u{0181}', "B"),
    ('\u{0182}', "B"),
    ('\u{0183}', "b"),
    ('\u{0186}', "O"),
    ('\u{0187}', "C"),
    ('\u{0188}', "c"),
    ('\u{0189}', "D"),
    ('\u{018A}', "D"),
    ('\u{018B}', "D"),
    ('\u{018C}', "d"),
    ('\u{018E}', "E"),
    ('\u{018F}', "A"),
    ('\u{0190}', "E"),
    ('\u{0191}', "F"),
    ('\u{0192}', "f"),
    ('\u{0193}', "G"),
    ('\u{0195}', "hv"),
    ('\u{0196}', "I"),
    ('\u{0197}', "I"),
    ('\u{0198}', "K"),
    ('\u{0199}', "k"),
    ('\u{019A}', "l"),
    ('\u{019C}', "M"),
    ('\u{019D}', "N"),
    ('\u{019E}', "n"),
    ('\u{019F}', "O"),
    ('\u{01A0}', "O"),
    ('\u{01A1}', "o"),
    ('\u{01A4}', "P"),
    ('\u{01A5}', "p"),
    ('\u{01AB}', "t"),
    ('\u{01AC}', "T"),
    ('\u{01AD}', "t"),
    ('\u{01AE}', "T"),
    ('\u{01AF}', "U"),
    ('\u{01B0}', "u"),
    ('\u{01B2}', "V"),
    ('\u{01B3}', "Y"),
    ('\u{01B4}', "y"),
    ('\u{01B5}', "Z"),
    ('\u{01B6}', "z"),
    ('\u{01BF}', "w"),
    ('\u{01C4}', "DZ"),
    ('\u{01C5}', "Dz"),
    ('\u{01C6}', "dz"),
    ('\u{01C7}', "LJ"),
    ('\u{01C8}', "Lj"),
    ('\u{01C9}', "lj"),
    ('\u{01CA}', "NJ"),
    ('\u{01CB}', "Nj"),
    ('\u{01CC}', "nj"),
    ('\u{01CD}', "A"),
    ('\u{01CE}', "a"),
    ('\u{01CF}', "I"),
    ('\u{01D0}', "i"),
    ('\u{01D1}', "O"),
    ('\u{01D2}', "o"),
    ('\u{01D3}', "U"),
    ('\u{01D4}', "u"),
    ('\u{01D5}', "U"),
    ('\u{01D6}', "u"),
    ('\u{01D7}', "U"),
    ('\u{01D8}', "u"),
    ('\u{01D9}', "U"),
    ('\u{01DA}', "u"),
    ('\u{01DB}', "U"),
    ('\u{01DC}', "u"),
    ('\u{01DD}', "e"),
    ('\u{01DE}', "A"),
    ('\u{01DF}', "a"),
    ('\u{01E0}', "A"),
    ('\u{01E1}', "a"),
    ('\u{01E2}', "AE"),
    ('\u{01E3}', "ae"),
    ('\u{01E4}', "G"),
    ('\u{01E5}', "G"),
    ('\u{01E6}', "G"),
    ('\u{01E7}', "G"),
    ('\u{01E8}', "K"),
    ('\u{01E9}', "k"),
    ('\u{01EA}', "O"),
    ('\u{01EB}', "o"),
    ('\u{01EC}', "O"),
    ('\u{01ED}', "o"),
    ('\u{01F0}', "j"),
    ('\u{01F1}', "DZ"),
    ('\u{01F2}', "Dz"),
    ('\u{01F3}', "dz"),
    ('\u{01F4}', "G"),
    ('\u{01F5}', "g"),
    ('\u{01F6}', "HV"),
    ('\u{01F7}', "W"),
    ('\u{01F8}', "N"),
    ('\u{01F9}', "n"),
    ('\u{01FA}', "A"),
    ('\u{01FB}', "a"),
    ('\u{01FC}', "AE"),
    ('\u{01FD}', "ae"),
    ('\u{01FE}', "O"),
    ('\u{01FF}', "o"),
    ('\u{0200}', "A"),
    ('\u{0201}', "a"),
    ('\u{0202}', "A"),
    ('\u{0203}', "a"),
    ('\u{0204}', "E"),
    ('\u{0205}', "e"),
    ('\u{0206}', "E"),
    ('\u{0207}', "e"),
    ('\u{0208}', "I"),
    ('\u{0209}', "i"),
    ('\u{020A}', "I"),
    ('\u{020B}', "i"),
    ('\u{020C}', "O"),
    ('\u{020D}', "o"),
    ('\u{020E}', "O"),
    ('\u{020F}', "o"),
    ('\u{0210}', "R"),
    ('\u{0211}', "r"),
    ('\u{0212}', "R"),
    ('\u{0213}', "r"),
    ('\u{0214}', "U"),
    ('\u{0215}', "u"),
    ('\u{0216}', "U"),
    ('\u{0217}', "u"),
    ('\u{0218}', "S"),
    ('\u{0219}', "s"),
    ('\u{021A}', "T"),
    ('\u{021B}', "t"),
    ('\u{021C}', "Z"),
    ('\u{021D}', "z"),
    ('\u{021E}', "H"),
    ('\u{021F}', "h"),
    ('\u{0220}', "N"),
    ('\u{0221}', "d"),
    ('\u{0222}', "OU"),
    ('\u{0223}', "ou"),
    ('\u{0224}', "Z"),
    ('\u{0225}', "z"),
    ('\u{0226}', "A"),
    ('\u{0227}', "a"),
    ('\u{0228}', "E"),
    ('\u{0229}', "e"),
    ('\u{022A}', "O"),
    ('\u{022B}', "o"),
    ('\u{022C}', "O"),
    ('\u{022D}', "o"),
    ('\u{022E}', "O"),
    ('\u{022F}', "o"),
    ('\u{0230}', "O"),
    ('\u{0231}', "o"),
    ('\u{0232}', "Y"),
    ('\u{0233}', "y"),
    ('\u{0234}', "l"),
    ('\u{0235}', "n"),
    ('\u{0236}', "t"),
    ('\u{0237}', "j"),
    ('\u{0238}', "db"),
    ('\u{0239}', "qp"),
    ('\u{023A}', "A"),
    ('\u{023B}', "C"),
    ('\u{023C}', "c"),
    ('\u{023D}', "L"),
    ('\u{023E}', "T"),
    ('\u{023F}', "s"),
    ('\u{0240}', "z"),
    ('\u{0243}', "B"),
    ('\u{0244}', "U"),
    ('\u{0245}', "V"),
    ('\u{0246}', "E"),
    ('\u{0247}', "e"),
    ('\u{0248}', "J"),
    ('\u{0249}', "j"),
    ('\u{024A}', "Q"),
    ('\u{024B}', "q"),
    ('\u{024C}', "R"),
    ('\u{024D}', "r"),
    ('\u{024E}', "Y"),
    ('\u{024F}', "y"),
    ('\u{0250}', "a"),
    ('\u{0253}', "b"),
    ('\u{0254}', "o"),
    ('\u{0255}', "c"),
    ('\u{0256}', "d"),
    ('\u{0257}', "d"),
    ('\u{0258}', "e"),
    ('\u{0259}', "a"),
    ('\u{025A}', "a"),
    ('\u{025B}', "e"),
    ('\u{025C}', "e"),
    ('\u{025D}', "e"),
    ('\u{025E}', "e"),
    ('\u{025F}', "j"),
    ('\u{0260}', "g"),
    ('\u{0261}', "g"),
    ('\u{0262}', "G"),
    ('\u{0265}', "h"),
    ('\u{0266}', "h"),
    ('\u{0268}', "i"),
    ('\u{026A}', "I"),
    ('\u{026B}', "l"),
    ('\u{026C}', "l"),
    ('\u{026D}', "l"),
    ('\u{026F}', "m"),
    ('\u{0270}', "m"),
    ('\u{0271}', "m"),
    ('\u{0272}', "n"),
    ('\u{0273}', "n"),
    ('\u{0274}', "N"),
    ('\u{0275}', "o"),
    ('\u{0276}', "OE"),
    ('\u{027C}', "r"),
    ('\u{027D}', "r"),
    ('\u{027E}', "r"),
    ('\u{027F}', "r"),
    ('\u{0280}', "R"),
    ('\u{0281}', "R"),
    ('\u{0282}', "s"),
    ('\u{0284}', "j"),
    ('\u{0287}', "t"),
    ('\u{0288}', "t"),
    ('\u{0289}', "u"),
    ('\u{028B}', "v"),
    ('\u{028C}', "v"),
    ('\u{028D}', "w"),
    ('\u{028E}', "y"),
    ('\u{028F}', "Y"),
    ('\u{0290}', "z"),
    ('\u{0291}', "z"),
    ('\u{0297}', "C"),
    ('\u{0299}', "B"),
    ('\u{029A}', "e"),
    ('\u{029B}', "G"),
    ('\u{029C}', "H"),
    ('\u{029D}', "j"),
    ('\u{029E}', "k"),
    ('\u{029F}', "L"),
    ('\u{02A0}', "q"),
    ('\u{02A3}', "dz"),
    ('\u{02A5}', "dz"),
    ('\u{02A6}', "ts"),
    ('\u{02A8}', "tc"),
    ('\u{02AA}', "ls"),
    ('\u{02AB}', "lz"),
    ('\u{02AE}', "h"),
    ('\u{02AF}', "h"),
    ('\u{1D00}', "A"),
    ('\u{1D01}', "AE"),
    ('\u{1D02}', "ae"),
    ('\u{1D03}', "B"),
    ('\u{1D04}', "C"),
    ('\u{1D05}', "D"),
    ('\u{1D06}', "D"),
    ('\u{1D07}', "E"),
    ('\u{1D08}', "e"),
    ('\u{1D09}', "i"),
    ('\u{1D0A}', "J"),
    ('\u{1D0B}', "K"),
    ('\u{1D0C}', "L"),
    ('\u{1D0D}', "M"),
    ('\u{1D0E}', "N"),
    ('\u{1D0F}', "O"),
    ('\u{1D10}', "O"),
    ('\u{1D14}', "oe"),
    ('\u{1D15}', "OU"),
    ('\u{1D16}', "o"),
    ('\u{1D17}', "o"),
    ('\u{1D18}', "P"),
    ('\u{1D19}', "R"),
    ('\u{1D1A}', "R"),
    ('\u{1D1B}', "T"),
    ('\u{1D1C}', "U"),
    ('\u{1D20}', "V"),
    ('\u{1D21}', "W"),
    ('\u{1D22}', "Z"),
    ('\u{1D62}', "i"),
    ('\u{1D63}', "r"),
    ('\u{1D64}', "u"),
    ('\u{1D65}', "v"),
    ('\u{1D6B}', "ue"),
    ('\u{1D6C}', "b"),
    ('\u{1D6D}', "d"),
    ('\u{1D6E}', "f"),
    ('\u{1D6F}', "m"),
    ('\u{1D70}', "n"),
    ('\u{1D71}', "p"),
    ('\u{1D72}', "r"),
    ('\u{1D73}', "r"),
    ('\u{1D74}', "s"),
    ('\u{1D75}', "t"),
    ('\u{1D76}', "z"),
    ('\u{1D77}', "g"),
    ('\u{1D79}', "g"),
    ('\u{1D7A}', "th"),
    ('\u{1D7B}', "I"),
    ('\u{1D7C}', "i"),
    ('\u{1D7D}', "p"),
    ('\u{1D7E}', "U"),
    ('\u{1D80}', "b"),
    ('\u{1D81}', "d"),
    ('\u{1D82}', "f"),
    ('\u{1D83}', "g"),
    ('\u{1D84}', "k"),
    ('\u{1D85}', "l"),
    ('\u{1D86}', "m"),
    ('\u{1D87}', "n"),
    ('\u{1D88}', "p"),
    ('\u{1D89}', "r"),
    ('\u{1D8A}', "s"),
    ('\u{1D8C}', "v"),
    ('\u{1D8D}', "x"),
    ('\u{1D8E}', "z"),
    ('\u{1D8F}', "a"),
    ('\u{1D91}', "d"),
    ('\u{1D92}', "e"),
    ('\u{1D93}', "e"),
    ('\u{1D94}', "e"),
    ('\u{1D95}', "a"),
    ('\u{1D96}', "i"),
    ('\u{1D97}', "o"),
    ('\u{1D99}', "u"),
    ('\u{1E00}', "A"),
    ('\u{1E01}', "a"),
    ('\u{1E02}', "B"),
    ('\u{1E03}', "b"),
    ('\u{1E04}', "B"),
    ('\u{1E05}', "b"),
    ('\u{1E06}', "B"),
    ('\u{1E07}', "b"),
    ('\u{1E08}', "C"),
    ('\u{1E09}', "c"),
    ('\u{1E0A}', "D"),
    ('\u{1E0B}', "d"),
    ('\u{1E0C}', "D"),
    ('\u{1E0D}', "d"),
    ('\u{1E0E}', "D"),
    ('\u{1E0F}', "d"),
    ('\u{1E10}', "D"),
    ('\u{1E11}', "d"),
    ('\u{1E12}', "D"),
    ('\u{1E13}', "d"),
    ('\u{1E14}', "E"),
    ('\u{1E15}', "e"),
    ('\u{1E16}', "E"),
    ('\u{1E17}', "e"),
    ('\u{1E18}', "E"),
    ('\u{1E19}', "e"),
    ('\u{1E1A}', "E"),
    ('\u{1E1B}', "e"),
    ('\u{1E1C}', "E"),
    ('\u{1E1D}', "e"),
    ('\u{1E1E}', "F"),
    ('\u{1E1F}', "f"),
    ('\u{1E20}', "G"),
    ('\u{1E21}', "g"),
    ('\u{1E22}', "H"),
    ('\u{1E23}', "h"),
    ('\u{1E24}', "H"),
    ('\u{1E25}', "h"),
    ('\u{1E26}', "H"),
    ('\u{1E27}', "h"),
    ('\u{1E28}', "H"),
    ('\u{1E29}', "h"),
    ('\u{1E2A}', "H"),
    ('\u{1E2B}', "h"),
    ('\u{1E2C}', "I"),
    ('\u{1E2D}', "i"),
    ('\u{1E2E}', "I"),
    ('\u{1E2F}', "i"),
    ('\u{1E30}', "K"),
    ('\u{1E31}', "k"),
    ('\u{1E32}', "K"),
    ('\u{1E33}', "k"),
    ('\u{1E34}', "K"),
    ('\u{1E35}', "k"),
    ('\u{1E36}', "L"),
    ('\u{1E37}', "l"),
    ('\u{1E38}', "L"),
    ('\u{1E39}', "l"),
    ('\u{1E3A}', "L"),
    ('\u{1E3B}', "l"),
    ('\u{1E3C}', "L"),
    ('\u{1E3D}', "l"),
    ('\u{1E3E}', "M"),
    ('\u{1E3F}', "m"),
    ('\u{1E40}', "M"),
    ('\u{1E41}', "m"),
    ('\u{1E42}', "M"),
    ('\u{1E43}', "m"),
    ('\u{1E44}', "N"),
    ('\u{1E45}', "n"),
    ('\u{1E46}', "N"),
    ('\u{1E47}', "n"),
    ('\u{1E48}', "N"),
    ('\u{1E49}', "n"),
    ('\u{1E4A}', "N"),
    ('\u{1E4B}', "n"),
    ('\u{1E4C}', "O"),
    ('\u{1E4D}', "o"),
    ('\u{1E4E}', "O"),
    ('\u{1E4F}', "o"),
    ('\u{1E50}', "O"),
    ('\u{1E51}', "o"),
    ('\u{1E52}', "O"),
    ('\u{1E53}', "o"),
    ('\u{1E54}', "P"),
    ('\u{1E55}', "p"),
    ('\u{1E56}', "P"),
    ('\u{1E57}', "p"),
    ('\u{1E58}', "R"),
    ('\u{1E59}', "r"),
    ('\u{1E5A}', "R"),
    ('\u{1E5B}', "r"),
    ('\u{1E5C}', "R"),
    ('\u{1E5D}', "r"),
    ('\u{1E5E}', "R"),
    ('\u{1E5F}', "r"),
    ('\u{1E60}', "S"),
    ('\u{1E61}', "s"),
    ('\u{1E62}', "S"),
    ('\u{1E63}', "s"),
    ('\u{1E64}', "S"),
    ('\u{1E65}', "s"),
    ('\u{1E66}', "S"),
    ('\u{1E67}', "s"),
    ('\u{1E68}', "S"),
    ('\u{1E69}', "s"),
    ('\u{1E6A}', "T"),
    ('\u{1E6B}', "t"),
    ('\u{1E6C}', "T"),
    ('\u{1E6D}', "t"),
    ('\u{1E6E}', "T"),
    ('\u{1E6F}', "t"),
    ('\u{1E70}', "T"),
    ('\u{1E71}', "t"),
    ('\u{1E72}', "U"),
    ('\u{1E73}', "u"),
    ('\u{1E74}', "U"),
    ('\u{1E75}', "u"),
    ('\u{1E76}', "U"),
    ('\u{1E77}', "u"),
    ('\u{1E78}', "U"),
    ('\u{1E79}', "u"),
    ('\u{1E7A}', "U"),
    ('\u{1E7B}', "u"),
    ('\u{1E7C}', "V"),
    ('\u{1E7D}', "v"),
    ('\u{1E7E}', "V"),
    ('\u{1E7F}', "v"),
    ('\u{1E80}', "W"),
    ('\u{1E81}', "w"),
    ('\u{1E82}', "W"),
    ('\u{1E83}', "w"),
    ('\u{1E84}', "W"),
    ('\u{1E85}', "w"),
    ('\u{1E86}', "W"),
    ('\u{1E87}', "w"),
    ('\u{1E88}', "W"),
    ('\u{1E89}', "w"),
    ('\u{1E8A}', "X"),
    ('\u{1E8B}', "x"),
    ('\u{1E8C}', "X"),
    ('\u{1E8D}', "x"),
    ('\u{1E8E}', "Y"),
    ('\u{1E8F}', "y"),
    ('\u{1E90}', "Z"),
    ('\u{1E91}', "z"),
    ('\u{1E92}', "Z"),
    ('\u{1E93}', "z"),
    ('\u{1E94}', "Z"),
    ('\u{1E95}', "z"),
    ('\u{1E96}', "h"),
    ('\u{1E97}', "t"),
    ('\u{1E98}', "w"),
    ('\u{1E99}', "y"),
    ('\u{1E9A}', "a"),
    ('\u{1E9B}', "f"),
    ('\u{1E9C}', "s"),
    ('\u{1E9D}', "s"),
    ('\u{1E9E}', "SS"),
    ('\u{1EA0}', "A"),
    ('\u{1EA1}', "a"),
    ('\u{1EA2}', "A"),
    ('\u{1EA3}', "a"),
    ('\u{1EA4}', "A"),
    ('\u{1EA5}', "a"),
    ('\u{1EA6}', "A"),
    ('\u{1EA7}', "a"),
    ('\u{1EA8}', "A"),
    ('\u{1EA9}', "a"),
    ('\u{1EAA}', "A"),
    ('\u{1EAB}', "a"),
    ('\u{1EAC}', "A"),
    ('\u{1EAD}', "a"),
    ('\u{1EAE}', "A"),
    ('\u{1EAF}', "a"),
    ('\u{1EB0}', "A"),
    ('\u{1EB1}', "a"),
    ('\u{1EB2}', "A"),
    ('\u{1EB3}', "a"),
    ('\u{1EB4}', "A"),
    ('\u{1EB5}', "a"),
    ('\u{1EB6}', "A"),
    ('\u{1EB7}', "a"),
    ('\u{1EB8}', "E"),
    ('\u{1EB9}', "e"),
    ('\u{1EBA}', "E"),
    ('\u{1EBB}', "e"),
    ('\u{1EBC}', "E"),
    ('\u{1EBD}', "e"),
    ('\u{1EBE}', "E"),
    ('\u{1EBF}', "e"),
    ('\u{1EC0}', "E"),
    ('\u{1EC1}', "e"),
    ('\u{1EC2}', "E"),
    ('\u{1EC3}', "e"),
    ('\u{1EC4}', "E"),
    ('\u{1EC5}', "e"),
    ('\u{1EC6}', "E"),
    ('\u{1EC7}', "e"),
    ('\u{1EC8}', "I"),
    ('\u{1EC9}', "i"),
    ('\u{1ECA}', "I"),
    ('\u{1ECB}', "i"),
    ('\u{1ECC}', "O"),
    ('\u{1ECD}', "o"),
    ('\u{1ECE}', "O"),
    ('\u{1ECF}', "o"),
    ('\u{1ED0}', "O"),
    ('\u{1ED1}', "o"),
    ('\u{1ED2}', "O"),
    ('\u{1ED3}', "o"),
    ('\u{1ED4}', "O"),
    ('\u{1ED5}', "o"),
    ('\u{1ED6}', "O"),
    ('\u{1ED7}', "o"),
    ('\u{1ED8}', "O"),
    ('\u{1ED9}', "o"),
    ('\u{1EDA}', "O"),
    ('\u{1EDB}', "o"),
    ('\u{1EDC}', "O"),
    ('\u{1EDD}', "o"),
    ('\u{1EDE}', "O"),
    ('\u{1EDF}', "o"),
    ('\u{1EE0}', "O"),
    ('\u{1EE1}', "o"),
    ('\u{1EE2}', "O"),
    ('\u{1EE3}', "o"),
    ('\u{1EE4}', "U"),
    ('\u{1EE5}', "u"),
    ('\u{1EE6}', "U"),
    ('\u{1EE7}', "u"),
    ('\u{1EE8}', "U"),
    ('\u{1EE9}', "u"),
    ('\u{1EEA}', "U"),
    ('\u{1EEB}', "u"),
    ('\u{1EEC}', "U"),
    ('\u{1EED}', "u"),
    ('\u{1EEE}', "U"),
    ('\u{1EEF}', "u"),
    ('\u{1EF0}', "U"),
    ('\u{1EF1}', "u"),
    ('\u{1EF2}', "Y"),
    ('\u{1EF3}', "y"),
    ('\u{1EF4}', "Y"),
    ('\u{1EF5}', "y"),
    ('\u{1EF6}', "Y"),
    ('\u{1EF7}', "y"),
    ('\u{1EF8}', "Y"),
    ('\u{1EF9}', "y"),
    ('\u{1EFA}', "LL"),
    ('\u{1EFB}', "ll"),
    ('\u{1EFC}', "V"),
    ('\u{1EFE}', "Y"),
    ('\u{1EFF}', "y"),
    ('\u{2010}', "-"),
    ('\u{2011}', "-"),
    ('\u{2012}', "-"),
    ('\u{2013}', "-"),
    ('\u{2014}', "-"),
    ('\u{2018}', "'"),
    ('\u{2019}', "'"),
    ('\u{201A}', "'"),
    ('\u{201B}', "'"),
    ('\u{201C}', "\""),
    ('\u{201D}', "\""),
    ('\u{201E}', "\""),
    ('\u{2032}', "'"),
    ('\u{2033}', "\""),
    ('\u{2035}', "'"),
    ('\u{2036}', "\""),
    ('\u{2038}', "^"),
    ('\u{2039}', "'"),
    ('\u{203A}', "'"),
    ('\u{203C}', "!!"),
    ('\u{2044}', "/"),
    ('\u{2045}', "["),
    ('\u{2046}', "]"),
    ('\u{2047}', "??"),
    ('\u{2048}', "?!"),
    ('\u{2049}', "!?"),
    ('\u{204E}', "*"),
    ('\u{204F}', ";"),
    ('\u{2052}', "%"),
    ('\u{2053}', "~"),
    ('\u{2070}', "0"),
    ('\u{2071}', "i"),
    ('\u{2074}', "4"),
    ('\u{2075}', "5"),
    ('\u{2076}', "6"),
    ('\u{2077}', "7"),
    ('\u{2078}', "8"),
    ('\u{2079}', "9"),
    ('\u{207A}', "+"),
    ('\u{207B}', "-"),
    ('\u{207C}', "="),
    ('\u{207D}', "("),
    ('\u{207E}', ")"),
    ('\u{207F}', "n"),
    ('\u{2080}', "0"),
    ('\u{2081}', "1"),
    ('\u{2082}', "2"),
    ('\u{2083}', "3"),
    ('\u{2084}', "4"),
    ('\u{2085}', "5"),
    ('\u{2086}', "6"),
    ('\u{2087}', "7"),
    ('\u{2088}', "8"),
    ('\u{2089}', "9"),
    ('\u{208A}', "+"),
    ('\u{208B}', "-"),
    ('\u{208C}', "="),
    ('\u{208D}', "("),
    ('\u{208E}', ")"),
    ('\u{2090}', "a"),
    ('\u{2091}', "e"),
    ('\u{2092}', "o"),
    ('\u{2093}', "x"),
    ('\u{2094}', "a"),
    ('\u{2184}', "c"),
    ('\u{2460}', "1"),
    ('\u{2461}', "2"),
    ('\u{2462}', "3"),
    ('\u{2463}', "4"),
    ('\u{2464}', "5"),
    ('\u{2465}', "6"),
    ('\u{2466}', "7"),
    ('\u{2467}', "8"),
    ('\u{2468}', "9"),
    ('\u{2469}', "10"),
    ('\u{246A}', "11"),
    ('\u{246B}', "12"),
    ('\u{246C}', "13"),
    ('\u{246D}', "14"),
    ('\u{246E}', "15"),
    ('\u{246F}', "16"),
    ('\u{2470}', "17"),
    ('\u{2471}', "18"),
    ('\u{2472}', "19"),
    ('\u{2473}', "20"),
    ('\u{2474}', "(1)"),
    ('\u{2475}', "(2)"),
    ('\u{2476}', "(3)"),
    ('\u{2477}', "(4)"),
    ('\u{2478}', "(5)"),
    ('\u{2479}', "(6)"),
    ('\u{247A}', "(7)"),
    ('\u{247B}', "(8)"),
    ('\u{247C}', "(9)"),
    ('\u{247D}', "(10)"),
    ('\u{247E}', "(11)"),
    ('\u{247F}', "(12)"),
    ('\u{2480}', "(13)"),
    ('\u{2481}', "(14)"),
    ('\u{2482}', "(15)"),
    ('\u{2483}', "(16)"),
    ('\u{2484}', "(17)"),
    ('\u{2485}', "(18)"),
    ('\u{2486}', "(19)"),
    ('\u{2487}', "(20)"),
    ('\u{2488}', "1."),
    ('\u{2489}', "2."),
    ('\u{248A}', "3."),
    ('\u{248B}', "4."),
    ('\u{248C}', "5."),
    ('\u{248D}', "6."),
    ('\u{248E}', "7."),
    ('\u{248F}', "8."),
    ('\u{2490}', "9."),
    ('\u{2491}', "10."),
    ('\u{2492}', "11."),
    ('\u{2493}', "12."),
    ('\u{2494}', "13."),
    ('\u{2495}', "14."),
    ('\u{2496}', "15."),
    ('\u{2497}', "16."),
    ('\u{2498}', "17."),
    ('\u{2499}', "18."),
    ('\u{249A}', "19."),
    ('\u{249B}', "20."),
    ('\u{249C}', "(a)"),
    ('\u{249D}', "(b)"),
    ('\u{249E}', "(c)"),
    ('\u{249F}', "(d)"),
    ('\u{24A0}', "(e)"),
    ('\u{24A1}', "(f)"),
    ('\u{24A2}', "(g)"),
    ('\u{24A3}', "(h)"),
    ('\u{24A4}', "(i)"),
    ('\u{24A5}', "(j)"),
    ('\u{24A6}', "(k)"),
    ('\u{24A7}', "(l)"),
    ('\u{24A8}', "(m)"),
    ('\u{24A9}', "(n)"),
    ('\u{24AA}', "(o)"),
    ('\u{24AB}', "(p)"),
    ('\u{24AC}', "(q)"),
    ('\u{24AD}', "(r)"),
    ('\u{24AE}', "(s)"),
    ('\u{24AF}', "(t)"),
    ('\u{24B0}', "(u)"),
    ('\u{24B1}', "(v)"),
    ('\u{24B2}', "(w)"),
    ('\u{24B3}', "(x)"),
    ('\u{24B4}', "(y)"),
    ('\u{24B5}', "(z)"),
    ('\u{24B6}', "A"),
    ('\u{24B7}', "B"),
    ('\u{24B8}', "C"),
    ('\u{24B9}', "D"),
    ('\u{24BA}', "E"),
    ('\u{24BB}', "F"),
    ('\u{24BC}', "G"),
    ('\u{24BD}', "H"),
    ('\u{24BE}', "I"),
    ('\u{24BF}', "J"),
    ('\u{24C0}', "K"),
    ('\u{24C1}', "L"),
    ('\u{24C2}', "M"),
    ('\u{24C3}', "N"),
    ('\u{24C4}', "O"),
    ('\u{24C5}', "P"),
    ('\u{24C6}', "Q"),
    ('\u{24C7}', "R"),
    ('\u{24C8}', "S"),
    ('\u{24C9}', "T"),
    ('\u{24CA}', "U"),
    ('\u{24CB}', "V"),
    ('\u{24CC}', "W"),
    ('\u{24CD}', "X"),
    ('\u{24CE}', "Y"),
    ('\u{24CF}', "Z"),
    ('\u{24D0}', "a"),
    ('\u{24D1}', "b"),
    ('\u{24D2}', "c"),
    ('\u{24D3}', "d"),
    ('\u{24D4}', "e"),
    ('\u{24D5}', "f"),
    ('\u{24D6}', "g"),
    ('\u{24D7}', "h"),
    ('\u{24D8}', "i"),
    ('\u{24D9}', "j"),
    ('\u{24DA}', "k"),
    ('\u{24DB}', "l"),
    ('\u{24DC}', "m"),
    ('\u{24DD}', "n"),
    ('\u{24DE}', "o"),
    ('\u{24DF}', "p"),
    ('\u{24E0}', "q"),
    ('\u{24E1}', "r"),
    ('\u{24E2}', "s"),
    ('\u{24E3}', "t"),
    ('\u{24E4}', "u"),
    ('\u{24E5}', "v"),
    ('\u{24E6}', "w"),
    ('\u{24E7}', "x"),
    ('\u{24E8}', "y"),
    ('\u{24E9}', "z"),
    ('\u{24EA}', "0"),
    ('\u{24EB}', "11"),
    ('\u{24EC}', "12"),
    ('\u{24ED}', "13"),
    ('\u{24EE}', "14"),
    ('\u{24EF}', "15"),
    ('\u{24F0}', "16"),
    ('\u{24F1}', "17"),
    ('\u{24F2}', "18"),
    ('\u{24F3}', "19"),
    ('\u{24F4}', "20"),
    ('\u{24F5}', "1"),
    ('\u{24F6}', "2"),
    ('\u{24F7}', "3"),
    ('\u{24F8}', "4"),
    ('\u{24F9}', "5"),
    ('\u{24FA}', "6"),
    ('\u{24FB}', "7"),
    ('\u{24FC}', "8"),
    ('\u{24FD}', "9"),
    ('\u{24FE}', "10"),
    ('\u{24FF}', "0"),
    ('\u{275B}', "'"),
    ('\u{275C}', "'"),
    ('\u{275D}', "\""),
    ('\u{275E}', "\""),
    ('\u{2768}', "("),
    ('\u{2769}', ")"),
    ('\u{276A}', "("),
    ('\u{276B}', ")"),
    ('\u{276C}', "<"),
    ('\u{276D}', ">"),
    ('\u{276E}', "\""),
    ('\u{276F}', "\""),
    ('\u{2770}', "<"),
    ('\u{2771}', ">"),
    ('\u{2772}', "["),
    ('\u{2773}', "]"),
    ('\u{2774}', "{"),
    ('\u{2775}', "}"),
    ('\u{2776}', "1"),
    ('\u{2777}', "2"),
    ('\u{2778}', "3"),
    ('\u{2779}', "4"),
    ('\u{277A}', "5"),
    ('\u{277B}', "6"),
    ('\u{277C}', "7"),
    ('\u{277D}', "8"),
    ('\u{277E}', "9"),
    ('\u{277F}', "10"),
    ('\u{2780}', "1"),
    ('\u{2781}', "2"),
    ('\u{2782}', "3"),
    ('\u{2783}', "4"),
    ('\u{2784}', "5"),
    ('\u{2785}', "6"),
    ('\u{2786}', "7"),
    ('\u{2787}', "8"),
    ('\u{2788}', "9"),
    ('\u{2789}', "10"),
    ('\u{278A}', "1"),
    ('\u{278B}', "2"),
    ('\u{278C}', "3"),
    ('\u{278D}', "4"),
    ('\u{278E}', "5"),
    ('\u{278F}', "6"),
    ('\u{2790}', "7"),
    ('\u{2791}', "8"),
    ('\u{2792}', "9"),
    ('\u{2793}', "10"),
    ('\u{2C60}', "L"),
    ('\u{2C61}', "l"),
    ('\u{2C62}', "L"),
    ('\u{2C63}', "P"),
    ('\u{2C64}', "R"),
    ('\u{2C65}', "a"),
    ('\u{2C66}', "t"),
    ('\u{2C67}', "H"),
    ('\u{2C68}', "h"),
    ('\u{2C69}', "K"),
    ('\u{2C6A}', "k"),
    ('\u{2C6B}', "Z"),
    ('\u{2C6C}', "z"),
    ('\u{2C6E}', "M"),
    ('\u{2C6F}', "a"),
    ('\u{2C71}', "v"),
    ('\u{2C72}', "W"),
    ('\u{2C73}', "w"),
    ('\u{2C74}', "v"),
    ('\u{2C75}', "H"),
    ('\u{2C76}', "h"),
    ('\u{2C78}', "e"),
    ('\u{2C7A}', "o"),
    ('\u{2C7B}', "E"),
    ('\u{2C7C}', "j"),
    ('\u{2E28}', "(("),
    ('\u{2E29}', "))"),
    ('\u{A728}', "TZ"),
    ('\u{A729}', "tz"),
    ('\u{A730}', "F"),
    ('\u{A731}', "S"),
    ('\u{A732}', "AA"),
    ('\u{A733}', "aa"),
    ('\u{A734}', "AO"),
    ('\u{A735}', "ao"),
    ('\u{A736}', "AU"),
    ('\u{A737}', "au"),
    ('\u{A738}', "AV"),
    ('\u{A739}', "av"),
    ('\u{A73A}', "AV"),
    ('\u{A73B}', "av"),
    ('\u{A73C}', "AY"),
    ('\u{A73D}', "ay"),
    ('\u{A73E}', "c"),
    ('\u{A73F}', "c"),
    ('\u{A740}', "K"),
    ('\u{A741}', "k"),
    ('\u{A742}', "K"),
    ('\u{A743}', "k"),
    ('\u{A744}', "K"),
    ('\u{A745}', "k"),
    ('\u{A746}', "L"),
    ('\u{A747}', "l"),
    ('\u{A748}', "L"),
    ('\u{A749}', "l"),
    ('\u{A74A}', "O"),
    ('\u{A74B}', "o"),
    ('\u{A74C}', "O"),
    ('\u{A74D}', "o"),
    ('\u{A74E}', "OO"),
    ('\u{A74F}', "oo"),
    ('\u{A750}', "P"),
    ('\u{A751}', "p"),
    ('\u{A752}', "P"),
    ('\u{A753}', "p"),
    ('\u{A754}', "P"),
    ('\u{A755}', "p"),
    ('\u{A756}', "Q"),
    ('\u{A757}', "q"),
    ('\u{A758}', "Q"),
    ('\u{A759}', "q"),
    ('\u{A75A}', "R"),
    ('\u{A75B}', "r"),
    ('\u{A75E}', "V"),
    ('\u{A75F}', "v"),
    ('\u{A760}', "VY"),
    ('\u{A761}', "vy"),
    ('\u{A762}', "Z"),
    ('\u{A763}', "z"),
    ('\u{A766}', "TH"),
    ('\u{A767}', "th"),
    ('\u{A768}', "V"),
    ('\u{A779}', "D"),
    ('\u{A77A}', "d"),
    ('\u{A77B}', "F"),
    ('\u{A77C}', "f"),
    ('\u{A77D}', "G"),
    ('\u{A77E}', "G"),
    ('\u{A77F}', "g"),
    ('\u{A780}', "L"),
    ('\u{A781}', "l"),
    ('\u{A782}', "R"),
    ('\u{A783}', "r"),
    ('\u{A784}', "s"),
    ('\u{A785}', "S"),
    ('\u{A786}', "T"),
    ('\u{A7FB}', "F"),
    ('\u{A7FC}', "p"),
    ('\u{A7FD}', "M"),
    ('\u{A7FE}', "I"),
    ('\u{A7FF}', "M"),
    ('\u{FB00}', "ff"),
    ('\u{FB01}', "fi"),
    ('\u{FB02}', "fl"),
    ('\u{FB03}', "ffi"),
    ('\u{FB04}', "ffl"),
    ('\u{FB06}', "st"),
    ('\u{FF01}', "!"),
    ('\u{FF02}', "\""),
    ('\u{FF03}', "#"),
    ('\u{FF04}', "$"),
    ('\u{FF05}', "%"),
    ('\u{FF06}', "&"),
    ('\u{FF07}', "'"),
    ('\u{FF08}', "("),
    ('\u{FF09}', ")"),
    ('\u{FF0A}', "*"),
    ('\u{FF0B}', "+"),
    ('\u{FF0C}', ","),
    ('\u{FF0D}', "-"),
    ('\u{FF0E}', "."),
    ('\u{FF0F}', "/"),
    ('\u{FF10}', "0"),
    ('\u{FF11}', "1"),
    ('\u{FF12}', "2"),
    ('\u{FF13}', "3"),
    ('\u{FF14}', "4"),
    ('\u{FF15}', "5"),
    ('\u{FF16}', "6"),
    ('\u{FF17}', "7"),
    ('\u{FF18}', "8"),
    ('\u{FF19}', "9"),
    ('\u{FF1A}', ":"),
    ('\u{FF1B}', ";"),
    ('\u{FF1C}', "<"),
    ('\u{FF1D}', "="),
    ('\u{FF1E}', ">"),
    ('\u{FF1F}', "?"),
    ('\u{FF20}', "@"),
    ('\u{FF21}', "A"),
    ('\u{FF22}', "B"),
    ('\u{FF23}', "C"),
    ('\u{FF24}', "D"),
    ('\u{FF25}', "E"),
    ('\u{FF26}', "F"),
    ('\u{FF27}', "G"),
    ('\u{FF28}', "H"),
    ('\u{FF29}', "I"),
    ('\u{FF2A}', "J"),
    ('\u{FF2B}', "K"),
    ('\u{FF2C}', "L"),
    ('\u{FF2D}', "M"),
    ('\u{FF2E}', "N"),
    ('\u{FF2F}', "O"),
    ('\u{FF30}', "P"),
    ('\u{FF31}', "Q"),
    ('\u{FF32}', "R"),
    ('\u{FF33}', "S"),
    ('\u{FF34}', "T"),
    ('\u{FF35}', "U"),
    ('\u{FF36}', "V"),
    ('\u{FF37}', "W"),
    ('\u{FF38}', "X"),
    ('\u{FF39}', "Y"),
    ('\u{FF3A}', "Z"),
    ('\u{FF3B}', "["),
    ('\u{FF3C}', "\\"),
    ('\u{FF3D}', "]"),
    ('\u{FF3E}', "^"),
    ('\u{FF3F}', "_"),
    ('\u{FF41}', "a"),
    ('\u{FF42}', "b"),
    ('\u{FF43}', "c"),
    ('\u{FF44}', "d"),
    ('\u{FF45}', "e"),
    ('\u{FF46}', "f"),
    ('\u{FF47}', "g"),
    ('\u{FF48}', "h"),
    ('\u{FF49}', "i"),
    ('\u{FF4A}', "j"),
    ('\u{FF4B}', "k"),
    ('\u{FF4C}', "l"),
    ('\u{FF4D}', "m"),
    ('\u{FF4E}', "n"),
    ('\u{FF4F}', "o"),
    ('\u{FF50}', "p"),
    ('\u{FF51}', "q"),
    ('\u{FF52}', "r"),
    ('\u{FF53}', "s"),
    ('\u{FF54}', "t"),
    ('\u{FF55}', "u"),
    ('\u{FF56}', "v"),
    ('\u{FF57}', "w"),
    ('\u{FF58}', "x"),
    ('\u{FF59}', "y"),
    ('\u{FF5A}', "z"),
    ('\u{FF5B}', "{"),
    ('\u{FF5D}', "}"),
    ('\u{FF5E}', "~"),
];

impl AsciiFoldingFilter {
    /// Returns the ASCII fold for `c`, or `None` if `c` folds to itself.
    /// `FOLD_TABLE` is sorted by codepoint, so this is a binary search
    /// (~11 comparisons) rather than the 1242-arm `switch` Java relies on
    /// the JIT to turn into a jump table.
    fn fold_char(c: char) -> Option<&'static str> {
        FOLD_TABLE
            .binary_search_by_key(&c, |(k, _)| *k)
            .ok()
            .map(|i| FOLD_TABLE[i].1)
    }

    /// Folds each token's `term` codepoint by codepoint, leaving
    /// `start_offset`/`end_offset`/`position_increment` completely untouched
    /// even when folding changes the term's character length (e.g. a
    /// ligature growing to two ASCII characters).
    ///
    /// The pure-ASCII fast path is Lucene's own: `incrementToken` scans for a
    /// character `>= '\u0080'` and returns the token untouched when there is
    /// none.
    pub fn apply(tokens: Vec<Token>) -> Vec<Token> {
        Self::apply_with(tokens, false)
    }

    /// [`Self::apply`] with `ASCIIFoldingFilter`'s `preserveOriginal`
    /// constructor flag.
    ///
    /// When set, a token that folding actually changed is emitted **twice**:
    /// the folded form first, then the original with a position increment of
    /// `0`, so both index at the same position. Java captures the pre-fold
    /// `State` and replays it on the next `incrementToken` call
    /// (`posIncAttr.setPositionIncrement(0)`), which is exactly this shape.
    /// A token folding left alone is emitted once either way.
    /// The folded form of `term`, or `None` when folding leaves it as is
    /// (the streaming filter's per-token step).
    pub(crate) fn fold_term(term: &str) -> Option<String> {
        let mut folded = String::new();
        Self::fold_term_into(term, &mut folded).then_some(folded)
    }

    /// [`Self::fold_term`] into `folded` (cleared first): `true` when the
    /// term changed, `folded` then holding the folded term.
    pub(crate) fn fold_term_into(term: &str, folded: &mut String) -> bool {
        if term.is_ascii() {
            return false;
        }
        folded.clear();
        folded.reserve(term.len());
        let mut changed = false;
        for c in term.chars() {
            match Self::fold_char(c) {
                Some(replacement) => {
                    changed = true;
                    folded.push_str(replacement);
                }
                None => folded.push(c),
            }
        }
        changed
    }

    pub fn apply_with(tokens: Vec<Token>, preserve_original: bool) -> Vec<Token> {
        let mut out = Vec::with_capacity(tokens.len());
        for t in tokens {
            if t.term.is_ascii() {
                out.push(t);
                continue;
            }
            let mut folded = String::with_capacity(t.term.len());
            for c in t.term.chars() {
                match Self::fold_char(c) {
                    Some(replacement) => folded.push_str(replacement),
                    None => folded.push(c),
                }
            }
            if folded == t.term {
                out.push(t);
                continue;
            }
            let mut original = t.clone();
            let mut folded_token = t;
            folded_token.term = folded;
            out.push(folded_token);
            if preserve_original {
                original.position_increment = 0;
                out.push(original);
            }
        }
        out
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

/// The Porter2/"Snowball English" stemmer -- real Lucene's
/// `org.tartarus.snowball.ext.EnglishStemmer`, generated from Snowball's
/// `english.sbl` and exposed via `org.apache.lucene.analysis.snowball.
/// SnowballFilter` constructed with an `EnglishStemmer`. This is a
/// **different, separate** algorithm from [`PorterStemFilter`] (the classic
/// 1980 Porter algorithm, `org.tartarus.snowball.ext.PorterStemmer`, which is
/// what real Lucene's `EnglishAnalyzer` actually wires up by default) -- see
/// [`PorterStemFilter`] for why the two are not interchangeable. This filter
/// ports the Porter2/Snowball algorithm itself: step 0 (leading/trailing apostrophe
/// and possessive `'s` removal), the y/Y consonant-vowel bookkeeping, R1/R2
/// region computation (including the nine irregular-prefix words --
/// `arsenal`, `commune`, `emergency`, `generalization`, `interest`,
/// `lately`, `organization`, `pastime`, `university`, whose R1 is forced to
/// start right after the fixed prefix rather than the usual computed
/// position), steps 1a-1c, 2, 3, 4, 5, and the short-word/short-syllable
/// (`r_shortv`) and whole-word exception tables (`skis`->`ski`,
/// `skies`->`sky`, `idly`->`idl`, `gently`->`gentl`, `ugly`->`ugli`,
/// `early`->`earli`, `only`->`onli`, `singly`->`singl`, plus `andes`/
/// `atlas`/`bias`/`cosmos`/`howe`/`news`/`sky` left unchanged) -- a faithful,
/// complete port of the algorithm, not a subset.
///
/// **Domain of definition**: like [`PorterStemFilter`], operates only on
/// lowercase input; unlike it, this filter's own step 0 explicitly handles a
/// leading/trailing/possessive apostrophe (`"don't"`, `"cats'"`, `"'tis"`),
/// so the domain here is lowercase ASCII letters *plus* `'`. Anything
/// outside that (uppercase, digits, other punctuation, non-ASCII) passes
/// through **unchanged**, never a panic.
///
/// **Not wired into [`PorterStemFilter`] or `Analyzer::with_stemming`** --
/// this is an additive, separate filter (`Analyzer::with_snowball_stemming`
/// opts into it instead), matching real Lucene's own separation of
/// `EnglishAnalyzer` (classic Porter) from `SnowballFilter`+`EnglishStemmer`
/// (Porter2), which are two different, independently-selectable filters in
/// real Lucene, not a superset/subset relationship.
pub struct SnowballEnglishStemFilter;

impl SnowballEnglishStemFilter {
    pub fn apply(tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                t.term = snowball_english::stem(&t.term);
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

    /// Multi-word extension of [`SynonymFilter::apply`]/[`SynonymFilter::apply_bidirectional`]:
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
    /// **Emission** is `SynonymGraphFilter.bufferOutputTokens`', node for
    /// node. Each match spans a node range `startNode .. endNode`, one side
    /// path per `rule.outputs` entry runs alongside the original tokens'
    /// path, and every emitted token's attributes come out of the graph:
    /// `position_increment = startNode - lastNodeOut` and
    /// `position_length = endNode - startNode`
    /// (`SynonymGraphFilter.releaseBufferedToken`).
    ///
    /// The **order** is Lucene's too, and it is not the obvious one: the
    /// first token of every synonym path is emitted first, then the first
    /// original token, then the remainders of each synonym path, then the
    /// remaining originals. Java's comment on that ordering is
    /// *"We must do the original tokens last, else the offsets 'go
    /// backwards'"*.
    ///
    /// So `["wi", "fi"] -> ["wifi"]` over `wi fi` emits
    /// `wifi` (increment 1, length 2, offsets spanning the whole match),
    /// then `wi` (increment 0, length 1, its own offsets), then `fi`
    /// (increment 1, length 1, its own offsets).
    ///
    /// Two things were wrong here before, and both are visible in that one
    /// example:
    ///
    /// - The collapsed synonym was emitted **after** the originals with
    ///   `position_increment == 0`, which put it at the position of `fi`
    ///   rather than of `wi` -- one position too late, and spanning
    ///   `P+1 .. P+3` instead of `P .. P+2`.
    /// - Every emitted token, originals included, was given the whole
    ///   match's offsets, so the sequence ran `wi`(0-2), `fi`(3-5),
    ///   `wifi`(0-5): a **decreasing** `startOffset`, which real Lucene's
    ///   `IndexingChain` rejects outright with "startOffset must be
    ///   non-decreasing". Originals now keep their own offsets, and only
    ///   the synonym tokens get the match span, exactly as
    ///   `releaseBufferedToken` does (`restoreState` for an original,
    ///   `setOffset(matchStartOffset, matchEndOffset)` for a synonym).
    ///
    /// A single input token expanding to a multi-word output now also gives
    /// the **original** a `position_length` equal to the output path's
    /// length, since both paths must rejoin at the same `endNode`.
    ///
    /// `keepOrig` is always true here: this port has no equivalent of
    /// `SolrSynonymParser`'s per-rule `=>` (replace) form, so the matched
    /// input is never dropped.
    ///
    /// **Scope carve-out**: the output is still a flat `Vec<Token>` rather
    /// than a graph `TokenStream` -- the position/length attributes describe
    /// the lattice correctly, but nothing in this port consumes them the way
    /// `GraphTokenFilter`/`TokenStreamToAutomaton` do, so a phrase or span
    /// query cannot yet follow a side path (see `docs/parity.md`).
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
        // The graph bookkeeping `SynonymGraphFilter` keeps: `lastNodeOut` is
        // the node the previously emitted token departed from (-1 before
        // anything is emitted, so the first token's increment comes out as
        // 1), `nextNodeOut` the node the next match would start at.
        let mut last_node_out: i32 = -1;
        let mut next_node_out: i32 = 0;
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
                    let match_start = tokens[i].start_offset;
                    let match_end = tokens[i + len - 1].end_offset;
                    let synonym = |term: &String| Token {
                        term: term.clone(),
                        start_offset: match_start,
                        end_offset: match_end,
                        // Filled in from the graph below.
                        position_increment: 0,
                        position_length: 1,
                    };

                    // `totalPathNodes`: the intermediate nodes every path
                    // longer than one token needs, plus `matchInputLength - 1`
                    // for the original tokens' own path (`keepOrig`).
                    let mut total_path_nodes = len as i32 - 1;
                    for output in &rule.outputs {
                        total_path_nodes += output.len() as i32 - 1;
                    }
                    let start_node = next_node_out;
                    let end_node = start_node + total_path_nodes + 1;

                    // `(token, startNode, endNode)`, in Java's buffering order.
                    let mut buffered: Vec<(Token, i32, i32)> = Vec::new();
                    let mut new_node_count = 0;
                    for output in &rule.outputs {
                        let path_end = if output.len() == 1 {
                            end_node
                        } else {
                            let node = start_node + new_node_count + 1;
                            new_node_count += output.len() as i32 - 1;
                            node
                        };
                        buffered.push((synonym(&output[0]), start_node, path_end));
                    }
                    // "We must do the original tokens last, else the offsets
                    // go backwards."
                    let input_end_node = if len == 1 {
                        end_node
                    } else {
                        start_node + new_node_count + 1
                    };
                    let originals_index = buffered.len();
                    buffered.push((tokens[i].clone(), start_node, input_end_node));
                    next_node_out = end_node;

                    // Then the rest of each multi-token side path...
                    for (path_id, output) in rule.outputs.iter().enumerate() {
                        if output.len() > 1 {
                            let mut last = buffered[path_id].2;
                            for term in &output[1..output.len() - 1] {
                                buffered.push((synonym(term), last, last + 1));
                                last += 1;
                            }
                            let final_term = output.last().expect("output is non-empty");
                            buffered.push((synonym(final_term), last, end_node));
                        }
                    }
                    // ... and the rest of the original tokens.
                    if len > 1 {
                        let mut last = buffered[originals_index].2;
                        for k in 1..len - 1 {
                            buffered.push((tokens[i + k].clone(), last, last + 1));
                            last += 1;
                        }
                        buffered.push((tokens[i + len - 1].clone(), last, end_node));
                    }

                    for (mut t, node_start, node_end) in buffered {
                        t.position_increment = node_start - last_node_out;
                        t.position_length = node_end - node_start;
                        last_node_out = node_start;
                        out.push(t);
                    }
                    i += len;
                }
                None => {
                    // A pass-through token advances the graph by its own
                    // increment/length, so the next match starts at the right
                    // node.
                    let t = tokens[i].clone();
                    last_node_out += t.position_increment;
                    next_node_out = last_node_out + t.position_length;
                    out.push(t);
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
/// **Positions**: the first gram derived from a given input token gets that
/// token's own `position_increment`; every subsequent gram from the *same*
/// input token gets `position_increment == 0` (an alternative reading at the
/// same starting position).
///
/// Crucially, an input token that produces **no** grams still contributes its
/// increment: Java accumulates `curPosIncr += posIncrAtt.getPositionIncrement()`
/// per input token and only zeroes it once a gram is actually emitted, so a
/// skipped short token pushes the *next* token's grams one position further
/// along: with `min_gram = 3` the text `"a big cat"` puts `cat`'s grams two
/// positions after `a`, which keeps phrase and slop offsets downstream right.
///
/// **Offsets are the original token's, unchanged.** Java's `incrementToken`
/// calls `restoreState(state)` before emitting each gram, which restores the
/// captured `OffsetAttribute` wholesale; it never calls `setOffset`. The
/// filter's own javadoc says so outright and notes that this is why
/// highlighting does not work with it. A precise per-gram offset range would
/// look more useful but disagree with Lucene on every gram.
pub struct NGramTokenFilter;

/// Computes, for `term`, the ordered list of gram substrings whose codepoint
/// length falls in `min_gram..=max_gram`; if `edge_only` is true, only grams
/// starting at codepoint 0 are produced (the [`EdgeNGramTokenFilter`] case).
///
/// Grams never split a multi-byte UTF-8 character: Java measures with
/// `Character.codePointCount` and slices with `Character.offsetByCodePoints`,
/// which is what iterating `chars()` does here.
fn ngrams_for_term(term: &str, min_gram: i32, max_gram: i32, edge_only: bool) -> Vec<String> {
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
            grams.push(chars[start..end].iter().collect());
        }
    }
    grams
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
/// convention. Shared implementation for both [`NGramTokenFilter::apply`] and
/// [`EdgeNGramTokenFilter::apply`], and a direct transcription of Java's
/// `incrementToken` loop: `cur_pos_incr` is Java's `curPosIncr`, and the two
/// `preserve_original` branches are its "Token is shorter than minGram" and
/// "Token is longer than maxGram" arms.
///
/// `end()` publishes whatever `curPosIncr` is left over when the stream runs
/// dry, so a document whose *last* tokens were all shorter than `min_gram`
/// still advances the position counter. That is carried on
/// [`AnalyzedTokens::final_position_increment`] by
/// [`apply_ngram_filter_to_stream`]; this function is the token-vector-only
/// form both filters' `apply` entry points keep.
///
/// Note Java **overwrites** rather than adds:
/// `posIncrAtt.setPositionIncrement(curPosIncr)`, so an upstream filter's own
/// end-of-stream increment (a `StopFilter`'s `skippedPositions`, say) is
/// discarded when an n-gram filter sits downstream of it. Reproduced as-is.
fn apply_ngram_filter_to_stream(
    stream: AnalyzedTokens,
    min_gram: i32,
    max_gram: i32,
    edge_only: bool,
    preserve_original: bool,
) -> Result<AnalyzedTokens, String> {
    let AnalyzedTokens {
        tokens,
        final_position_increment: _,
        final_offset,
    } = stream;
    let (tokens, cur_pos_incr) =
        ngram_tokens(tokens, min_gram, max_gram, edge_only, preserve_original)?;
    Ok(AnalyzedTokens {
        tokens,
        final_position_increment: cur_pos_incr,
        final_offset,
    })
}

fn apply_ngram_filter(
    tokens: Vec<Token>,
    min_gram: i32,
    max_gram: i32,
    edge_only: bool,
    preserve_original: bool,
) -> Result<Vec<Token>, String> {
    Ok(ngram_tokens(tokens, min_gram, max_gram, edge_only, preserve_original)?.0)
}

/// The loop itself, returning the tokens and the `curPosIncr` left over at end
/// of stream.
fn ngram_tokens(
    tokens: Vec<Token>,
    min_gram: i32,
    max_gram: i32,
    edge_only: bool,
    preserve_original: bool,
) -> Result<(Vec<Token>, i32), String> {
    validate_gram_range(min_gram, max_gram)?;
    let mut out = Vec::new();
    // Java's `curPosIncr`: accumulated across input tokens that emit nothing,
    // spent on the first token this filter actually emits.
    let mut cur_pos_incr = 0;
    for t in tokens {
        let code_points = t.term.chars().count();
        cur_pos_incr += t.position_increment;

        if preserve_original && code_points < min_gram as usize {
            out.push(Token {
                position_increment: cur_pos_incr,
                ..t
            });
            cur_pos_incr = 0;
            continue;
        }

        for gram in ngrams_for_term(&t.term, min_gram, max_gram, edge_only) {
            out.push(Token {
                term: gram,
                // `restoreState(state)` puts back the input token's offsets
                // verbatim -- Lucene never narrows them to the gram.
                start_offset: t.start_offset,
                end_offset: t.end_offset,
                position_increment: cur_pos_incr,
                position_length: t.position_length,
            });
            cur_pos_incr = 0;
        }

        if preserve_original && code_points > max_gram as usize {
            out.push(Token {
                position_increment: 0,
                ..t
            });
        }
    }
    Ok((out, cur_pos_incr))
}

impl NGramTokenFilter {
    /// Grams every token in `tokens` per this filter's documented algorithm,
    /// with `preserveOriginal` off -- Java's
    /// `NGramTokenFilter.DEFAULT_PRESERVE_ORIGINAL`.
    ///
    /// Returns `Err` if `min_gram`/`max_gram` are not both positive or if
    /// `min_gram > max_gram` (see [`validate_gram_range`]); on success,
    /// tokens shorter than `min_gram` (in codepoints) contribute no output
    /// tokens at all.
    pub fn apply(tokens: Vec<Token>, min_gram: i32, max_gram: i32) -> Result<Vec<Token>, String> {
        apply_ngram_filter(tokens, min_gram, max_gram, false, false)
    }

    /// [`Self::apply`] with `NGramTokenFilter`'s `preserveOriginal`
    /// constructor flag.
    ///
    /// When set, a token too **short** to produce any gram is emitted
    /// verbatim (carrying the accumulated position increment) instead of
    /// being dropped, and a token **longer** than `max_gram` is emitted
    /// verbatim after its grams with a position increment of `0`. A token
    /// exactly `max_gram` long is not duplicated -- its grams already include
    /// the whole term.
    pub fn apply_preserving_original(
        tokens: Vec<Token>,
        min_gram: i32,
        max_gram: i32,
    ) -> Result<Vec<Token>, String> {
        apply_ngram_filter(tokens, min_gram, max_gram, false, true)
    }

    /// [`Self::apply`] with `NGramTokenFilter.end()` run: the leftover
    /// `curPosIncr` from trailing input tokens that produced no gram lands on
    /// [`AnalyzedTokens::final_position_increment`] instead of being dropped.
    pub fn apply_to_stream(
        stream: AnalyzedTokens,
        min_gram: i32,
        max_gram: i32,
        preserve_original: bool,
    ) -> Result<AnalyzedTokens, String> {
        apply_ngram_filter_to_stream(stream, min_gram, max_gram, false, preserve_original)
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
        apply_ngram_filter(tokens, min_gram, max_gram, true, false)
    }

    /// [`Self::apply`] with `EdgeNGramTokenFilter`'s `preserveOriginal`
    /// constructor flag -- see
    /// [`NGramTokenFilter::apply_preserving_original`].
    pub fn apply_preserving_original(
        tokens: Vec<Token>,
        min_gram: i32,
        max_gram: i32,
    ) -> Result<Vec<Token>, String> {
        apply_ngram_filter(tokens, min_gram, max_gram, true, true)
    }

    /// [`Self::apply`] with `EdgeNGramTokenFilter.end()` run -- see
    /// [`NGramTokenFilter::apply_to_stream`].
    pub fn apply_to_stream(
        stream: AnalyzedTokens,
        min_gram: i32,
        max_gram: i32,
        preserve_original: bool,
    ) -> Result<AnalyzedTokens, String> {
        apply_ngram_filter_to_stream(stream, min_gram, max_gram, true, preserve_original)
    }
}

/// Lucene's `org.apache.lucene.analysis.en.PorterStemmer` (Porter, 1980,
/// release 3 with Brian Goetz's buffer reuse), step for step: the word is
/// stemmed in a buffer of code units, `k` being its last index and `j` the
/// end of the stem a matched suffix leaves.
///
/// Java runs over the term's UTF-16 `char`s. Every test the algorithm makes
/// is an equality with an ASCII letter, so an ASCII term is stemmed over its
/// bytes, which are its units; any other term is stemmed over its UTF-16
/// units, as Java does, so a supplementary character counts as the two
/// consonants Java sees.
pub(crate) mod porter {
    /// A code unit: a byte of an ASCII term or a UTF-16 unit.
    pub(crate) trait Unit: Copy + Eq {
        fn of(b: u8) -> Self;
        fn is(self, b: u8) -> bool;
    }

    impl Unit for u8 {
        #[inline(always)]
        fn of(b: u8) -> Self {
            b
        }
        #[inline(always)]
        fn is(self, b: u8) -> bool {
            self == b
        }
    }

    impl Unit for u16 {
        #[inline(always)]
        fn of(b: u8) -> Self {
            u16::from(b)
        }
        #[inline(always)]
        fn is(self, b: u8) -> bool {
            self == u16::from(b)
        }
    }

    /// The stemmer's state over one word: `b[..=k]` is the word, `j`
    /// Java's `j`, which a matched `-ion` on a three-letter word leaves at
    /// `-1`. As in Java, the units past `k` stay in the buffer: step 6
    /// measures up to the `j` it set before dropping a final `e`.
    struct Stemmer<'a, T> {
        b: &'a mut Vec<T>,
        k: isize,
        j: isize,
        dirty: bool,
    }

    impl<T: Unit> Stemmer<'_, T> {
        #[inline(always)]
        fn k(&self) -> isize {
            self.k
        }

        #[inline(always)]
        fn at(&self, i: isize) -> T {
            self.b[i as usize]
        }

        /// `k = to`.
        #[inline(always)]
        fn set_k(&mut self, to: isize) {
            self.k = to;
        }

        /// `cons(i)`: `b[i]` is a consonant (`y` is one after a vowel or at
        /// the start).
        fn cons(&self, i: isize) -> bool {
            let c = self.at(i);
            if c.is(b'a') || c.is(b'e') || c.is(b'i') || c.is(b'o') || c.is(b'u') {
                false
            } else if c.is(b'y') {
                i == 0 || !self.cons(i - 1)
            } else {
                true
            }
        }

        /// `m()`: the number of consonant sequences between `0` and `j`.
        fn m(&self) -> u32 {
            let mut n = 0;
            let mut i = 0;
            loop {
                if i > self.j {
                    return n;
                }
                if !self.cons(i) {
                    break;
                }
                i += 1;
            }
            i += 1;
            loop {
                loop {
                    if i > self.j {
                        return n;
                    }
                    if self.cons(i) {
                        break;
                    }
                    i += 1;
                }
                i += 1;
                n += 1;
                loop {
                    if i > self.j {
                        return n;
                    }
                    if !self.cons(i) {
                        break;
                    }
                    i += 1;
                }
                i += 1;
            }
        }

        /// `vowelinstem()`: `0..=j` holds a vowel.
        fn vowel_in_stem(&self) -> bool {
            (0..=self.j).any(|i| !self.cons(i))
        }

        /// `doublec(j)`: `j-1, j` is a double consonant.
        fn doublec(&self, j: isize) -> bool {
            j >= 1 && self.at(j) == self.at(j - 1) && self.cons(j)
        }

        /// `cvc(i)`: `i-2, i-1, i` is consonant-vowel-consonant and the
        /// second consonant is not `w`, `x` or `y`.
        fn cvc(&self, i: isize) -> bool {
            if i < 2 || !self.cons(i) || self.cons(i - 1) || !self.cons(i - 2) {
                return false;
            }
            let c = self.at(i);
            !(c.is(b'w') || c.is(b'x') || c.is(b'y'))
        }

        /// `ends(s)`: `0..=k` ends with `s`; sets `j` to the stem's end.
        #[inline]
        fn ends(&mut self, s: &[u8]) -> bool {
            let l = s.len() as isize;
            let o = self.k() - l + 1;
            if o < 0 {
                return false;
            }
            let tail = &self.b[o as usize..=self.k as usize];
            if !tail.iter().zip(s).all(|(&c, &e)| c.is(e)) {
                return false;
            }
            self.j = self.k() - l;
            true
        }

        /// `setto(s)`: `j+1..` becomes `s`, `k` its end.
        fn setto(&mut self, s: &[u8]) {
            let o = (self.j + 1) as usize;
            if self.b.len() < o + s.len() {
                self.b.resize(o + s.len(), T::of(0));
            }
            for (slot, &c) in self.b[o..].iter_mut().zip(s) {
                *slot = T::of(c);
            }
            self.k = self.j + s.len() as isize;
            self.dirty = true;
        }

        /// `r(s)`: [`Self::setto`] when `m() > 0`.
        fn r(&mut self, s: &[u8]) {
            if self.m() > 0 {
                self.setto(s);
            }
        }

        /// `step1()`: plurals and `-ed`/`-ing`.
        fn step1(&mut self) {
            if self.at(self.k()).is(b's') {
                if self.ends(b"sses") {
                    self.set_k(self.k() - 2);
                } else if self.ends(b"ies") {
                    self.setto(b"i");
                } else if !self.at(self.k() - 1).is(b's') {
                    self.set_k(self.k() - 1);
                }
            }
            if self.ends(b"eed") {
                if self.m() > 0 {
                    self.set_k(self.k() - 1);
                }
            } else if (self.ends(b"ed") || self.ends(b"ing")) && self.vowel_in_stem() {
                self.set_k(self.j);
                if self.ends(b"at") {
                    self.setto(b"ate");
                } else if self.ends(b"bl") {
                    self.setto(b"ble");
                } else if self.ends(b"iz") {
                    self.setto(b"ize");
                } else if self.doublec(self.k()) {
                    let ch = self.at(self.k());
                    if !(ch.is(b'l') || ch.is(b's') || ch.is(b'z')) {
                        self.set_k(self.k() - 1);
                    }
                } else if self.m() == 1 && self.cvc(self.k()) {
                    self.setto(b"e");
                }
            }
        }

        /// `step2()`: a final `y` after a vowel in the stem becomes `i`.
        fn step2(&mut self) {
            if self.ends(b"y") && self.vowel_in_stem() {
                let k = self.k() as usize;
                self.b[k] = T::of(b'i');
                self.dirty = true;
            }
        }

        /// The first of `rules` whose suffix matches, replaced through
        /// [`Self::r`] (Java's `if (ends(..)) { r(..); break; }` chain).
        fn first(&mut self, rules: &[(&[u8], &[u8])]) {
            for (suffix, to) in rules {
                if self.ends(suffix) {
                    self.r(to);
                    return;
                }
            }
        }

        /// `step3()`: double suffixes to single ones, by `b[k-1]`.
        fn step3(&mut self) {
            if self.k() == 0 {
                return;
            }
            let c = self.at(self.k() - 1);
            let rules: &[(&[u8], &[u8])] = if c.is(b'a') {
                &[(b"ational", b"ate"), (b"tional", b"tion")]
            } else if c.is(b'c') {
                &[(b"enci", b"ence"), (b"anci", b"ance")]
            } else if c.is(b'e') {
                &[(b"izer", b"ize")]
            } else if c.is(b'l') {
                &[
                    (b"bli", b"ble"),
                    (b"alli", b"al"),
                    (b"entli", b"ent"),
                    (b"eli", b"e"),
                    (b"ousli", b"ous"),
                ]
            } else if c.is(b'o') {
                &[(b"ization", b"ize"), (b"ation", b"ate"), (b"ator", b"ate")]
            } else if c.is(b's') {
                &[
                    (b"alism", b"al"),
                    (b"iveness", b"ive"),
                    (b"fulness", b"ful"),
                    (b"ousness", b"ous"),
                ]
            } else if c.is(b't') {
                &[(b"aliti", b"al"), (b"iviti", b"ive"), (b"biliti", b"ble")]
            } else if c.is(b'g') {
                &[(b"logi", b"log")]
            } else {
                return;
            };
            self.first(rules);
        }

        /// `step4()`: `-ic-`, `-full`, `-ness` etc., by `b[k]`.
        fn step4(&mut self) {
            let c = self.at(self.k());
            let rules: &[(&[u8], &[u8])] = if c.is(b'e') {
                &[(b"icate", b"ic"), (b"ative", b""), (b"alize", b"al")]
            } else if c.is(b'i') {
                &[(b"iciti", b"ic")]
            } else if c.is(b'l') {
                &[(b"ical", b"ic"), (b"ful", b"")]
            } else if c.is(b's') {
                &[(b"ness", b"")]
            } else {
                return;
            };
            self.first(rules);
        }

        /// `step5()`: `-ant`, `-ence` etc. in context `<c>vcvc<v>`, by
        /// `b[k-1]`.
        fn step5(&mut self) {
            if self.k() == 0 {
                return;
            }
            let c = self.at(self.k() - 1);
            let matched = if c.is(b'o') {
                (self.ends(b"ion")
                    && self.j >= 0
                    && (self.at(self.j).is(b's') || self.at(self.j).is(b't')))
                    || self.ends(b"ou")
            } else {
                let suffixes: &[&[u8]] = if c.is(b'a') {
                    &[b"al"]
                } else if c.is(b'c') {
                    &[b"ance", b"ence"]
                } else if c.is(b'e') {
                    &[b"er"]
                } else if c.is(b'i') {
                    &[b"ic"]
                } else if c.is(b'l') {
                    &[b"able", b"ible"]
                } else if c.is(b'n') {
                    &[b"ant", b"ement", b"ment", b"ent"]
                } else if c.is(b's') {
                    &[b"ism"]
                } else if c.is(b't') {
                    &[b"ate", b"iti"]
                } else if c.is(b'u') {
                    &[b"ous"]
                } else if c.is(b'v') {
                    &[b"ive"]
                } else if c.is(b'z') {
                    &[b"ize"]
                } else {
                    return;
                };
                suffixes.iter().any(|s| self.ends(s))
            };
            if matched && self.m() > 1 {
                self.set_k(self.j);
            }
        }

        /// `step6()`: a final `-e`, and `-ll` to `-l`, when `m() > 1`.
        fn step6(&mut self) {
            self.j = self.k();
            if self.at(self.k()).is(b'e') {
                let a = self.m();
                if a > 1 || a == 1 && !self.cvc(self.k() - 1) {
                    self.set_k(self.k() - 1);
                }
            }
            if self.at(self.k()).is(b'l') && self.doublec(self.k()) && self.m() > 1 {
                self.set_k(self.k() - 1);
            }
        }
    }

    /// `PorterStemmer.stem(char[], int, int)` over `b` in place: `true`
    /// when the word changed (Java's `dirty`), `b` then holding the stem.
    pub(crate) fn stem_units<T: Unit>(b: &mut Vec<T>) -> bool {
        let len = b.len();
        // Java: `if (k > k0 + 1)` -- words of one or two units stay.
        if len < 3 {
            return false;
        }
        let mut s = Stemmer {
            b,
            k: len as isize - 1,
            j: 0,
            dirty: false,
        };
        s.step1();
        s.step2();
        s.step3();
        s.step4();
        s.step5();
        s.step6();
        let end = (s.k + 1) as usize;
        let dirty = s.dirty || end != len;
        b.truncate(end);
        dirty
    }

    /// Stems `term` in place, as `PorterStemFilter` copies the stemmer's
    /// buffer back when it is dirty; `units` is the UTF-16 buffer for a
    /// non-ASCII term.
    pub(crate) fn stem_term(term: &mut String, units: &mut Vec<u16>) {
        if term.len() < 3 {
            return;
        }
        if term.is_ascii() {
            let mut b = std::mem::take(term).into_bytes();
            stem_units(&mut b);
            // ASCII in, ASCII suffixes out.
            *term = String::from_utf8(b).unwrap_or_default();
            return;
        }
        units.clear();
        units.extend(term.encode_utf16());
        if stem_units(units) {
            term.clear();
            crate::util::push_str_utf16(term, units);
        }
    }

    /// Stems `term` (`PorterStemmer.stem(String)`).
    pub(crate) fn stem(term: &str) -> String {
        let mut t = term.to_string();
        stem_term(&mut t, &mut Vec::new());
        t
    }
}

/// The Porter2/"Snowball English" stemmer of [`SnowballEnglishStemFilter`]:
/// the generated `org.tartarus.snowball.ext.EnglishStemmer`
/// ([`snowball::algorithms::english`]).
mod snowball_english {
    /// `EnglishStemmer.stem()` over `word`.
    pub(crate) fn stem(word: &str) -> String {
        let mut s = crate::snowball::SnowballStemmer::for_name("English")
            .expect("English is a shipped stemmer");
        let mut out = word.to_string();
        s.stem_in_place(&mut out);
        out
    }
}

#[cfg(test)]
mod tests;
