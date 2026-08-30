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
pub struct TokenStream {
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
///
/// **Offsets are UTF-16 code units** -- Java `char` indices into `text`, the
/// unit `OffsetAttribute` reports (see [`Token`]). `unicode_word_indices`
/// hands back **byte** indices, so this walks the text once, converting as it
/// goes: the segmenter yields segments in ascending byte order, so the
/// conversion is a single running sum over the gaps between them, O(n) in the
/// text length overall rather than O(n) per token.
pub fn tokenize(text: &str) -> Vec<Token> {
    // One word-at-a-time scan decides the whole document: for ASCII text the
    // byte index the segmenter yields *is* the Java `char` index, so the
    // common case pays nothing per token.
    let ascii = text.is_ascii();
    let mut byte_pos = 0usize;
    let mut utf16_pos = 0usize;
    text.unicode_word_indices()
        .map(|(start, word)| {
            let (start_offset, end_offset) = if ascii {
                (start, start + word.len())
            } else {
                // The (non-token) gap since the previous segment, then the
                // token itself. Segments arrive in ascending byte order, so
                // this is one running sum over the text, not a rescan per
                // token.
                utf16_pos += utf16_len(&text[byte_pos..start]);
                let start_offset = utf16_pos;
                utf16_pos += utf16_len(word);
                byte_pos = start + word.len();
                (start_offset, utf16_pos)
            };
            Token {
                term: word.to_string(),
                start_offset: start_offset as i32,
                end_offset: end_offset as i32,
                position_increment: 1,
                position_length: 1,
            }
        })
        .collect()
}

/// [`tokenize`] as a whole `TokenStream`, i.e. with `Tokenizer.end()` run.
///
/// A tokenizer never swallows a position, so
/// [`TokenStream::final_position_increment`] is `0` (`TokenStream.end()`'s own
/// `posIncrAtt.setPositionIncrement(0)`); the value that matters here is
/// [`TokenStream::final_offset`], Java's
/// `Tokenizer.end()`'s `finalOffset = correctOffset(charCount)` -- the length
/// of the **whole input**, in UTF-16 code units, not the end of the last
/// token. `"fox   "` ends at offset 6, not 3, and a multi-valued field's next
/// value starts from there plus the analyzer's `getOffsetGap`.
pub fn tokenize_stream(text: &str) -> TokenStream {
    TokenStream {
        tokens: tokenize(text),
        final_position_increment: 0,
        final_offset: utf16_len(text) as i32,
    }
}

/// Java's `String.length()` for the same text: the number of UTF-16 code
/// units `s` encodes to, i.e. one per BMP scalar and two per
/// supplementary-plane scalar.
///
/// The ASCII fast path is what keeps [`tokenize`]'s new unit off the hot
/// path for the overwhelmingly common case: `str::is_ascii` is a word-at-a-
/// time scan over the bytes with no per-scalar decode, and for ASCII the
/// byte length *is* the code-unit length. Only genuinely non-ASCII text pays
/// the per-scalar `len_utf16` sum.
#[inline]
fn utf16_len(s: &str) -> usize {
    if s.is_ascii() {
        return s.len();
    }
    s.chars().map(char::len_utf16).sum()
}

/// Real Lucene's `LowerCaseFilter`: lowercases each token's term text,
/// leaving offsets and position increments untouched.
pub struct LowerCaseFilter;

/// Java's `CharacterUtils.toLowerCase` applied to one codepoint.
///
/// That method is
/// `Character.toChars(Character.toLowerCase(codePointAt(...)))` written
/// **back into the same buffer at the same index**, so it is the *simple*,
/// strictly 1:1 Unicode case mapping -- not the full mapping with its
/// expansions and context rules. Rust's `str::to_lowercase` is the full
/// mapping, and the two disagree on real text:
///
/// - `U+0130` LATIN CAPITAL LETTER I WITH DOT ABOVE (`İ`): Java's simple
///   mapping gives one character, `i`; Rust's full mapping gives two,
///   `i` + `U+0307` COMBINING DOT ABOVE.
/// - Greek final sigma: `"ΟΔΟΣ"` lowercases to `"οδοσ"` in Java, because
///   `Character.toLowerCase` has no notion of word position; Rust's
///   `str::to_lowercase` applies the final-sigma rule and produces
///   `"οδος"`.
///
/// Either disagreement means a term indexed under different bytes than
/// Lucene would use, which breaks exact term lookup outright.
///
/// `char::to_lowercase` in Rust is per-character (so no final-sigma context)
/// but still the full mapping, and the only unconditional full lowercase
/// mapping in Unicode that expands to more than one character is `U+0130`'s.
/// So: take the single-character result where there is one, special-case
/// `İ`, and otherwise leave the character alone -- which is what
/// `Character.toLowerCase` does for a codepoint with no simple mapping.
fn simple_to_lowercase(c: char) -> char {
    let mut it = c.to_lowercase();
    match (it.next(), it.next()) {
        (Some(lower), None) => lower,
        // `Character.toLowerCase('\u0130') == 'i'`.
        _ if c == '\u{0130}' => 'i',
        _ => c,
    }
}

impl LowerCaseFilter {
    pub fn apply(tokens: Vec<Token>) -> Vec<Token> {
        tokens
            .into_iter()
            .map(|mut t| {
                if !t.term.is_ascii() {
                    t.term = t.term.chars().map(simple_to_lowercase).collect();
                } else {
                    t.term.make_ascii_lowercase();
                }
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
pub const FRENCH_STOP_WORDS: &[&str] = &[
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
            TokenStream {
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
    /// [`TokenStream::final_position_increment`], which is what
    /// `IndexingChain` adds to the field's position counter.
    pub fn apply_to_stream(stream: TokenStream, stopwords: &HashSet<String>) -> TokenStream {
        let TokenStream {
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
        TokenStream {
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
/// This port previously carried 92 hand-picked entries (Latin-1 letters plus
/// 30 Latin Extended-A picks), 7% of the real table: Vietnamese, fullwidth
/// CJK-Latin, ligature and typographic-quote text folded to itself here and
/// to ASCII in Lucene, so the two engines indexed different terms.
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
const FOLD_TABLE: &[(char, &str)] = &[
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
/// task #175's parity-check note on [`PorterStemFilter`] for why the two are
/// not interchangeable. This filter is this crate's task #209 port of the
/// Porter2/Snowball algorithm itself: step 0 (leading/trailing apostrophe
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
/// along. This port used to drop that increment on the floor, so with
/// `min_gram = 3` the text `"a big cat"` put `cat`'s grams one position after
/// `a` where Lucene puts them two positions after -- a silent corruption of
/// every phrase and slop offset downstream.
///
/// **Offsets are the original token's, unchanged.** Java's `incrementToken`
/// calls `restoreState(state)` before emitting each gram, which restores the
/// captured `OffsetAttribute` wholesale; it never calls `setOffset`. The
/// filter's own javadoc says so outright and notes that this is why
/// highlighting does not work with it. This port used to compute a precise
/// per-gram offset range, which looks more useful and disagrees with Lucene
/// on every single gram.
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
/// [`TokenStream::final_position_increment`] by
/// [`apply_ngram_filter_to_stream`]; this function is the token-vector-only
/// form both filters' `apply` entry points keep.
///
/// Note Java **overwrites** rather than adds:
/// `posIncrAtt.setPositionIncrement(curPosIncr)`, so an upstream filter's own
/// end-of-stream increment (a `StopFilter`'s `skippedPositions`, say) is
/// discarded when an n-gram filter sits downstream of it. Reproduced as-is.
fn apply_ngram_filter_to_stream(
    stream: TokenStream,
    min_gram: i32,
    max_gram: i32,
    edge_only: bool,
    preserve_original: bool,
) -> Result<TokenStream, String> {
    let TokenStream {
        tokens,
        final_position_increment: _,
        final_offset,
    } = stream;
    let (tokens, cur_pos_incr) =
        ngram_tokens(tokens, min_gram, max_gram, edge_only, preserve_original)?;
    Ok(TokenStream {
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
    /// [`TokenStream::final_position_increment`] instead of being dropped.
    pub fn apply_to_stream(
        stream: TokenStream,
        min_gram: i32,
        max_gram: i32,
        preserve_original: bool,
    ) -> Result<TokenStream, String> {
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
        stream: TokenStream,
        min_gram: i32,
        max_gram: i32,
        preserve_original: bool,
    ) -> Result<TokenStream, String> {
        apply_ngram_filter_to_stream(stream, min_gram, max_gram, true, preserve_original)
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
///
/// A second, entirely distinct producer, [`Analyzer::keyword`], mirrors real
/// Lucene's `KeywordAnalyzer` instead: see that constructor's docs for its
/// no-tokenization, single-token semantics.
pub struct Analyzer {
    stopwords: Option<HashSet<String>>,
    ascii_folding: bool,
    stemming: bool,
    snowball_stemming: bool,
    synonyms: Option<HashMap<String, Vec<String>>>,
    synonyms_bidirectional: bool,
    /// When `true`, [`Analyzer::analyze`] short-circuits to
    /// [`Analyzer::keyword`]'s single-token behavior and every other field
    /// on this struct is inert (a keyword analyzer has no filter chain to
    /// configure -- see that constructor's docs).
    keyword: bool,
    /// `Analyzer.getPositionIncrementGap(String)` -- see
    /// [`Analyzer::with_position_increment_gap`]. Java's default: `0`.
    position_increment_gap: i32,
    /// `Analyzer.getOffsetGap(String)` -- see [`Analyzer::with_offset_gap`].
    /// Java's default: `1`.
    offset_gap: i32,
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
            snowball_stemming: false,
            synonyms: None,
            synonyms_bidirectional: false,
            keyword: false,
            position_increment_gap: 0,
            offset_gap: 1,
        }
    }

    /// Mirrors real Lucene's
    /// `org.apache.lucene.analysis.core.KeywordAnalyzer`: the entire input
    /// text becomes **exactly one token**, byte-for-byte as given -- no word
    /// segmentation, no lowercasing, no stopword removal, no stemming, no
    /// ASCII-folding, no synonym expansion. This is real Lucene's documented
    /// behavior for `KeywordAnalyzer` (which wires up a bare
    /// `KeywordTokenizer` with no filters at all), not a partial/deferred
    /// version of [`Analyzer::standard`] -- it is the intentional, complete
    /// scope of this producer, used for exact-match/sort fields (IDs, tags,
    /// status codes) where any tokenization at all would be wrong.
    ///
    /// The one edge case worth calling out explicitly: **empty input still
    /// produces exactly one token**, with an empty `term` and a zero-length
    /// `0..0` offset span -- matching real Lucene's `KeywordTokenizer`,
    /// whose `incrementToken()` unconditionally returns `true` (and reports
    /// `done = true` so the *next* call returns `false`) regardless of how
    /// many characters it read, including zero.
    ///
    /// Every other `Analyzer` builder method (`with_ascii_folding`,
    /// `with_stemming`, `with_synonyms`, `with_bidirectional_synonyms`) is
    /// meaningless on a keyword analyzer (there is no filter chain to
    /// configure) and calling one on the result of `Analyzer::keyword()` has
    /// no effect on [`Analyzer::analyze`]'s output.
    pub fn keyword() -> Self {
        Analyzer {
            stopwords: None,
            ascii_folding: false,
            stemming: false,
            snowball_stemming: false,
            synonyms: None,
            synonyms_bidirectional: false,
            keyword: true,
            position_increment_gap: 0,
            offset_gap: 1,
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

    /// Enables [`SnowballEnglishStemFilter`] (task #209's Porter2/Snowball
    /// English stemmer) instead of [`PorterStemFilter`] in this analyzer's
    /// chain. Filter order is otherwise identical to
    /// [`Analyzer::with_stemming`]: tokenize -> fold -> lowercase ->
    /// stopwords -> **stem**. Mutually exclusive with `with_stemming` in
    /// effect (not in flag storage) -- if both are enabled on the same
    /// `Analyzer`, this method's Snowball stemmer takes precedence and the
    /// classic Porter stemmer is skipped, since running both in sequence
    /// would double-stem and isn't a real Lucene configuration either
    /// filter is meant to model.
    pub fn with_snowball_stemming(mut self) -> Self {
        self.snowball_stemming = true;
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

    /// Sets this analyzer's `Analyzer.getPositionIncrementGap(String)`.
    ///
    /// The number of positions inserted **between two values of the same
    /// multi-valued field**. Java's base `Analyzer` returns `0` from it, which
    /// is this port's default too -- and 0 means a phrase query *can* match
    /// across a value boundary, which surprises people often enough that every
    /// consumer of Lucene (OpenSearch's `position_increment_gap`, default 100)
    /// exposes an override. Java overrides it by subclassing; this port has no
    /// per-field analyzer configuration, so the gap is per-`Analyzer` and the
    /// field name is not a parameter -- see this crate's scope notes.
    pub fn with_position_increment_gap(mut self, gap: i32) -> Self {
        self.position_increment_gap = gap;
        self
    }

    /// Sets this analyzer's `Analyzer.getOffsetGap(String)`: the number of
    /// character offsets inserted between two values of the same multi-valued
    /// field. Java's default is **`1`**, not `0` -- it exists so the last
    /// character of one value and the first of the next do not share an
    /// offset -- and that is this port's default too.
    pub fn with_offset_gap(mut self, gap: i32) -> Self {
        self.offset_gap = gap;
        self
    }

    /// `Analyzer.getPositionIncrementGap(fieldName)`.
    pub fn position_increment_gap(&self) -> i32 {
        self.position_increment_gap
    }

    /// `Analyzer.getOffsetGap(fieldName)`.
    pub fn offset_gap(&self) -> i32 {
        self.offset_gap
    }

    pub fn analyze(&self, text: &str) -> Vec<Token> {
        self.analyze_stream(text).tokens
    }

    /// [`Self::analyze`] as a whole `TokenStream`: the same tokens, plus the
    /// two end-of-stream attribute values `IndexingChain` reads after
    /// `stream.end()`. See [`TokenStream`] for why they matter.
    pub fn analyze_stream(&self, text: &str) -> TokenStream {
        if self.keyword {
            return TokenStream {
                tokens: vec![Token {
                    term: text.to_string(),
                    start_offset: 0,
                    // Java's `KeywordTokenizer` ends the one token at
                    // `finalOffset = correctOffset(charCount)`, a Java `char`
                    // count -- `utf16_len`, not `text.len()` (see [`Token`]).
                    end_offset: utf16_len(text) as i32,
                    position_increment: 1,
                    position_length: 1,
                }],
                final_position_increment: 0,
                final_offset: utf16_len(text) as i32,
            };
        }
        let TokenStream {
            tokens,
            final_position_increment,
            final_offset,
        } = tokenize_stream(text);
        let tokens = if self.ascii_folding {
            AsciiFoldingFilter::apply(tokens)
        } else {
            tokens
        };
        let tokens = LowerCaseFilter::apply(tokens);
        // `StopFilter` is the only filter in this chain that overrides `end()`;
        // every other one inherits `TokenFilter.end()`, which just forwards, so
        // they run on the token vector alone.
        let TokenStream {
            tokens,
            final_position_increment,
            final_offset,
        } = match &self.stopwords {
            Some(stopwords) => StopFilter::apply_to_stream(
                TokenStream {
                    tokens,
                    final_position_increment,
                    final_offset,
                },
                stopwords,
            ),
            None => TokenStream {
                tokens,
                final_position_increment,
                final_offset,
            },
        };
        let tokens = if self.snowball_stemming {
            SnowballEnglishStemFilter::apply(tokens)
        } else if self.stemming {
            PorterStemFilter::apply(tokens)
        } else {
            tokens
        };
        let tokens = match &self.synonyms {
            Some(synonyms) if self.synonyms_bidirectional => {
                SynonymFilter::apply_bidirectional(tokens, synonyms)
            }
            Some(synonyms) => SynonymFilter::apply(tokens, synonyms),
            None => tokens,
        };
        TokenStream {
            tokens,
            final_position_increment,
            final_offset,
        }
    }
}

/// The classic Porter stemming algorithm (Martin Porter, 1980), operating on
/// lowercase ASCII alphabetic words. See [`PorterStemFilter`] for the
/// documented per-step scope; this module is a direct, mechanical port of
/// the published algorithm's five steps.
mod porter {
    /// Stems `term`.
    ///
    /// Java's `PorterStemmer.stem(char[], int)` runs the six steps only when
    /// `k > k0 + 1`, i.e. when the word is **at least three characters**;
    /// one- and two-character words come back untouched. This port had no
    /// such guard and had an unrelated one of its own ("all lowercase ASCII"),
    /// which produced two families of wrong answers:
    ///
    /// - `"s"` stemmed to the **empty string** (step 1a deleted the only
    ///   character), and `"as"`/`"is"`/`"us"` lost their `s`; Java returns
    ///   all of them unchanged.
    /// - `"Cats"` and `"cafés"` came back unstemmed, because of the
    ///   ASCII-lowercase guard. Java has no such test -- `cons()` simply
    ///   treats every character that is not `a/e/i/o/u/y` as a consonant --
    ///   so it stems them to `"Cat"` and `"café"`.
    pub(super) fn stem(term: &str) -> String {
        let mut w: Vec<char> = term.chars().collect();
        if w.len() <= 2 {
            return term.to_string();
        }
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

    /// Java's `ends(s)`: does `w` end with `suffix`? Returns the length of
    /// the stem that would be left (Java's `j + 1`), or `None`.
    fn ends(w: &[char], suffix: &str) -> Option<usize> {
        let suf_len = suffix.chars().count();
        let stem_len = w.len().checked_sub(suf_len)?;
        w[stem_len..]
            .iter()
            .copied()
            .eq(suffix.chars())
            .then_some(stem_len)
    }

    /// Java's `switch` + `if (ends(..)) { r(..); break; }` chain, expressed
    /// once.
    ///
    /// The load-bearing detail, and the one this module used to get wrong:
    /// **the first suffix that matches wins, whether or not the measure test
    /// then lets the replacement through.** Java's `break` leaves the
    /// `switch` as soon as `ends()` succeeds, and only `r()` -- called before
    /// the `break` -- checks `m() > 0`. A rule list that instead kept
    /// searching after a measure failure produces different stems:
    ///
    /// - `"argument"` matches `-ment` in Java's step 5 with `m("argu") == 1`,
    ///   which is not `> 1`, so Java leaves the word alone. Falling through
    ///   to `-ent` instead gives `m("argum") == 2` and strips it, yielding
    ///   `"argum"`.
    /// - `"ization"` matches `-ization` in step 3 with `m("") == 0`, so Java
    ///   leaves it. Falling through to `-ation` gives `"izate"`.
    ///
    /// Returns `true` when a suffix matched at all.
    fn apply_first_match(w: &mut Vec<char>, rules: &[(&str, &str)], min_m: u32) -> bool {
        for (suffix, replacement) in rules {
            let Some(stem_len) = ends(w, suffix) else {
                continue;
            };
            if measure(&w[..stem_len]) >= min_m {
                w.truncate(stem_len);
                w.extend(replacement.chars());
            }
            return true;
        }
        false
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

    /// Java's `step3()` (the paper's step 2): maps double suffixes to single
    /// ones.
    ///
    /// Dispatch is a `switch` on the **second-to-last** character, and only
    /// the rules in that group are ever tried -- see [`apply_first_match`]
    /// for why "first match wins" matters. Two rules were also wrong here:
    /// the `l` group's first entry is `bli -> ble` (this port had
    /// `abli -> able`, so `"possibly"` stayed `"possibli"`), and the `g`
    /// group -- `logi -> log` -- was missing entirely, so `"technology"`,
    /// `"biology"` and `"apology"` all stopped at `-logi`.
    fn step2(w: &mut Vec<char>) {
        let n = w.len();
        // Java: `if (k == k0) return;` -- a one-character word has no
        // `b[k - 1]` to switch on.
        if n < 2 {
            return;
        }
        let rules: &[(&str, &str)] = match w[n - 2] {
            'a' => &[("ational", "ate"), ("tional", "tion")],
            'c' => &[("enci", "ence"), ("anci", "ance")],
            'e' => &[("izer", "ize")],
            'l' => &[
                ("bli", "ble"),
                ("alli", "al"),
                ("entli", "ent"),
                ("eli", "e"),
                ("ousli", "ous"),
            ],
            'o' => &[("ization", "ize"), ("ation", "ate"), ("ator", "ate")],
            's' => &[
                ("alism", "al"),
                ("iveness", "ive"),
                ("fulness", "ful"),
                ("ousness", "ous"),
            ],
            't' => &[("aliti", "al"), ("iviti", "ive"), ("biliti", "ble")],
            'g' => &[("logi", "log")],
            _ => return,
        };
        apply_first_match(w, rules, 1);
    }

    /// Java's `step4()` (the paper's step 3): `-ic-`, `-full`, `-ness` etc.
    /// Dispatch is on the **last** character here, not the second-to-last.
    fn step3(w: &mut Vec<char>) {
        let n = w.len();
        if n == 0 {
            return;
        }
        let rules: &[(&str, &str)] = match w[n - 1] {
            'e' => &[("icate", "ic"), ("ative", ""), ("alize", "al")],
            'i' => &[("iciti", "ic")],
            'l' => &[("ical", "ic"), ("ful", "")],
            's' => &[("ness", "")],
            _ => return,
        };
        apply_first_match(w, rules, 1);
    }

    /// Java's `step5()` (the paper's step 4): takes off `-ant`, `-ence` etc.
    /// in context `<c>vcvc<v>`, i.e. when `m(stem) > 1`.
    ///
    /// Dispatch is on the second-to-last character again, and the group falls
    /// straight through to `return` when no suffix in it matches -- there is
    /// no cross-group fallback. `-ion` additionally requires the stem to end
    /// in `s` or `t` (Java's `ends("ion") && j >= 0 && (b[j] == 's' ||
    /// b[j] == 't')`), and when that guard fails the `o` group tries `-ou`
    /// and then gives up.
    fn step4(w: &mut Vec<char>) {
        let n = w.len();
        if n < 2 {
            return;
        }
        let stem_len = if w[n - 2] == 'o' {
            match ends(w, "ion") {
                Some(stem_len) if stem_len >= 1 && matches!(w[stem_len - 1], 's' | 't') => {
                    Some(stem_len)
                }
                _ => ends(w, "ou"),
            }
        } else {
            let rules: &[&str] = match w[n - 2] {
                'a' => &["al"],
                'c' => &["ance", "ence"],
                'e' => &["er"],
                'i' => &["ic"],
                'l' => &["able", "ible"],
                // "element" etc. not stripped before the `m`.
                'n' => &["ant", "ement", "ment", "ent"],
                's' => &["ism"],
                't' => &["ate", "iti"],
                'u' => &["ous"],
                'v' => &["ive"],
                'z' => &["ize"],
                _ => return,
            };
            rules.iter().find_map(|suffix| ends(w, suffix))
        };
        let Some(stem_len) = stem_len else {
            return;
        };
        if measure(&w[..stem_len]) > 1 {
            w.truncate(stem_len);
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

/// The Porter2/"Snowball English" algorithm, operating on lowercase ASCII
/// words that may also contain a literal `'` (apostrophe), which step 0
/// explicitly strips/normalizes. See [`SnowballEnglishStemFilter`] for the
/// documented scope.
///
/// This is a mechanical port of the actual generated
/// `org.tartarus.snowball.ext.EnglishStemmer` bytecode shipped in real
/// Lucene 10.5.0's `lucene-analysis-common-10.5.0.jar` -- reconstructed by
/// decompiling that exact class (`javap`/CFR), **not** transcribed from the
/// `EnglishStemmer.java` source file found in a `lucene` git checkout at an
/// arbitrary revision, which turned out to implement a materially different
/// (newer/upstream) generation of `english.sbl` than what the pinned
/// 10.5.0 jar actually contains (extra `R1`-override prefixes, an
/// additional `r_exception2` whole-word protected-stem step, and a
/// different `-ing`-family special case) -- the two disagree on real words
/// like `"organization"` (`"organ"` vs. `"organiz"`) and `"emergency"`
/// (`"emerg"` vs. `"emergenc"`). The jar (what a real `EnglishAnalyzer`/
/// `SnowballFilter` user actually links against) is authoritative here.
mod snowball_english {
    /// Is `c` a "vowel" for the algorithm's own grouping purposes? Matches
    /// real Snowball's `g_v` grouping (`a`, `e`, `i`, `o`, `u`, `y`) --
    /// note this deliberately does **not** include the uppercase `'Y'`
    /// marker this module uses internally to mean "y acting as a
    /// consonant here" (see [`prelude`]).
    fn is_vowel(c: char) -> bool {
        matches!(c, 'a' | 'e' | 'i' | 'o' | 'u' | 'y')
    }

    /// Does `w` end with the literal characters of `suffix`?
    fn ends(w: &[char], suffix: &str) -> bool {
        let suf_len = suffix.chars().count();
        w.len() >= suf_len && w[w.len() - suf_len..].iter().copied().eq(suffix.chars())
    }

    /// Removes `tail_len` trailing characters from `w` and appends
    /// `replacement` in their place.
    fn replace_tail(w: &mut Vec<char>, tail_len: usize, replacement: &str) {
        let keep = w.len() - tail_len;
        w.truncate(keep);
        w.extend(replacement.chars());
    }

    /// Real Snowball's `r_prelude`: strips a leading apostrophe, marks a
    /// word-initial `y` as consonant (`'Y'`), and marks every `y`
    /// immediately following a vowel as consonant (`'Y'`) too -- so the rest
    /// of the algorithm can treat plain lowercase `'y'` as always a vowel
    /// and `'Y'` as always a consonant.
    fn prelude(w: &mut Vec<char>) {
        if w.first() == Some(&'\'') {
            w.remove(0);
        }
        if w.first() == Some(&'y') {
            w[0] = 'Y';
        }
        let mut i = 0;
        while i + 1 < w.len() {
            if is_vowel(w[i]) && w[i + 1] == 'y' {
                w[i + 1] = 'Y';
            }
            i += 1;
        }
    }

    /// Real Snowball's region search used for both R1 and R2: starting at
    /// `start`, finds the first vowel, then the first non-vowel following
    /// it, and returns the index right after that non-vowel (or `w.len()`
    /// if no such point exists).
    fn find_region(w: &[char], start: usize) -> usize {
        let n = w.len();
        let mut i = start;
        while i < n && !is_vowel(w[i]) {
            i += 1;
        }
        if i >= n {
            return n;
        }
        i += 1;
        while i < n && is_vowel(w[i]) {
            i += 1;
        }
        if i >= n {
            return n;
        }
        i + 1
    }

    /// The three irregular prefixes real Snowball's `a_0` table forces R1
    /// to start right after, instead of the normally-computed position (a
    /// fix-up for otherwise-too-short words like `"generalization"`).
    const R1_EXCEPTION_PREFIXES: &[&str] = &["arsen", "commun", "gener"];

    /// Computes `(R1, R2)` -- the region-start indices real Snowball calls
    /// `I_p1`/`I_p2` -- once per [`stem`] call, on the word as it stands
    /// right after [`prelude`] and before any suffix stripping. Both
    /// indices are absolute offsets into the original buffer and remain
    /// valid for every later step's boundary checks, since every later step
    /// only ever truncates the buffer's *tail* (suffix stripping), never
    /// touching or shifting characters before either region's start. R2 is
    /// always [`find_region`] starting from R1's own value (whether R1 came
    /// from the exceptional-prefix override or the standard computation).
    fn mark_regions(w: &[char]) -> (usize, usize) {
        let s: String = w.iter().collect();
        let p1 = R1_EXCEPTION_PREFIXES
            .iter()
            .find(|prefix| s.starts_with(**prefix))
            .map(|prefix| prefix.chars().count())
            .unwrap_or_else(|| find_region(w, 0));
        let p2 = find_region(w, p1);
        (p1, p2)
    }

    /// Real Snowball's `r_shortv`: does `w` end in a "short syllable" --
    /// either a vowel-consonant pair that is the *entire* word (a vowel at
    /// the very start followed immediately by a non-vowel), or a trailing
    /// consonant/vowel/consonant run whose final consonant is not `w`, `x`,
    /// or the internal `'Y'` marker?
    fn ends_short_syllable(w: &[char]) -> bool {
        let n = w.len();
        if n == 2 && is_vowel(w[0]) && !is_vowel(w[1]) {
            return true;
        }
        n >= 3
            && !is_vowel(w[n - 3])
            && is_vowel(w[n - 2])
            && !is_vowel(w[n - 1])
            && !matches!(w[n - 1], 'w' | 'x' | 'Y')
    }

    /// The whole-word exception table real Snowball checks *first*, before
    /// anything else in `stem()` (its own `r_exception1`, table `a_10`): if
    /// `word` matches one of these entries *exactly* (the entire word, not
    /// just a suffix), stemming stops here and this mapping (verbatim,
    /// including the no-op entries) is the final result.
    fn exception1(word: &str) -> Option<&'static str> {
        Some(match word {
            "skis" => "ski",
            "skies" => "sky",
            "dying" => "die",
            "lying" => "lie",
            "tying" => "tie",
            "idly" => "idl",
            "gently" => "gentl",
            "ugly" => "ugli",
            "early" => "earli",
            "only" => "onli",
            "singly" => "singl",
            "andes" => "andes",
            "atlas" => "atlas",
            "bias" => "bias",
            "cosmos" => "cosmos",
            "howe" => "howe",
            "news" => "news",
            "sky" => "sky",
            _ => return None,
        })
    }

    /// The second whole-word protected-stem table (real Snowball's
    /// `r_exception2`, table `a_9`), checked *after* Step 1a but *before*
    /// Step 1b: if the word (as it stands after Step 1a, which never
    /// changes any of these eight) is exactly one of these, every step from
    /// 1b through 5 is skipped entirely and the word is left as-is --
    /// `"succeed"`/`"proceed"`/`"exceed"` never lose their `-eed`, and
    /// `"canning"`/`"inning"`/`"earring"`/`"herring"`/`"outing"` are not
    /// treated as gerunds.
    fn is_exception2(w: &[char]) -> bool {
        let s: String = w.iter().collect();
        matches!(
            s.as_str(),
            "succeed"
                | "proceed"
                | "exceed"
                | "canning"
                | "inning"
                | "earring"
                | "herring"
                | "outing"
        )
    }

    /// Step 0 (apostrophe/possessive removal) + Step 1a (plural suffixes),
    /// exactly as real Snowball's generated code combines both into a
    /// single routine.
    fn step0_and_1a(w: &mut Vec<char>) {
        if ends(w, "'s'") {
            w.truncate(w.len() - 3);
        } else if ends(w, "'s") {
            w.truncate(w.len() - 2);
        } else if ends(w, "'") {
            w.truncate(w.len() - 1);
        }

        if ends(w, "sses") {
            replace_tail(w, 2, ""); // sses -> ss
        } else if ends(w, "ied") || ends(w, "ies") {
            let stem_len = w.len() - 3;
            if stem_len > 1 {
                replace_tail(w, 3, "i");
            } else {
                replace_tail(w, 3, "ie");
            }
        } else if ends(w, "ss") {
            // Protected: "ss" is left unchanged.
        } else if ends(w, "us") {
            // Protected: "us" is left unchanged.
        } else if ends(w, "s") {
            let n = w.len();
            if n >= 2 && w[..n - 2].iter().any(|&c| is_vowel(c)) {
                w.truncate(n - 1);
            }
        }
    }

    /// The shared post-suffix-deletion cleanup real Snowball's `a_3` table
    /// applies after Step 1b deletes `-ed`/`-edly`/`-ing`/`-ingly`: append
    /// `e` after `at`/`bl`/`iz`; drop one letter of a trailing doubled
    /// consonant from a fixed set, unless the word is *exactly* one of
    /// those consonants preceded by a single `a`/`e`/`o` (real Snowball's
    /// own narrow carve-out, e.g. a 3-letter word "aXX"); otherwise append
    /// `e` if the word is now both "short" (R1 is empty, i.e. `p1` is
    /// exactly the current length) and ends in a short syllable.
    fn a3_cleanup(w: &mut Vec<char>, p1: usize) {
        if ends(w, "at") || ends(w, "bl") || ends(w, "iz") {
            w.push('e');
            return;
        }
        let n = w.len();
        if n >= 2
            && w[n - 1] == w[n - 2]
            && matches!(
                w[n - 1],
                'b' | 'd' | 'f' | 'g' | 'm' | 'n' | 'p' | 'r' | 't'
            )
        {
            if n == 3 && matches!(w[0], 'a' | 'e' | 'o') {
                // Exception: leave the doubled consonant untouched.
            } else {
                w.pop();
            }
            return;
        }
        if w.len() == p1 && ends_short_syllable(w) {
            w.push('e');
        }
    }

    /// Deletes a `suf_len`-character suffix (already confirmed present by
    /// the caller) if the remaining stem contains a vowel, then runs
    /// [`a3_cleanup`] -- shared by `-ed`/`-edly`/`-ing`/`-ingly` (real
    /// Snowball's `among_var == 2` branch).
    fn delete_and_cleanup(w: &mut Vec<char>, p1: usize, suf_len: usize) {
        let stem_len = w.len() - suf_len;
        if !w[..stem_len].iter().any(|&c| is_vowel(c)) {
            return;
        }
        w.truncate(stem_len);
        a3_cleanup(w, p1);
    }

    /// Step 1b: the `-eed`/`-eedly` family (real Snowball's
    /// `among_var == 1`, R1-gated, always replaced with `"ee"` -- the
    /// `succ`/`proc`/`exc` protection this used to need is handled earlier,
    /// globally, by [`is_exception2`]) or the shared `-ed`/`-edly`/`-ing`/
    /// `-ingly` deletion+cleanup path (`among_var == 2`), tried
    /// longest-suffix-first (`"eedly"` before `"edly"` before `"eed"`
    /// before `"ed"`, matching real Snowball's `Among`-table longest-match
    /// semantics; `"ingly"`/`"ing"` don't overlap with the `-d` family).
    fn step1b(w: &mut Vec<char>, p1: usize) {
        if ends(w, "eedly") {
            let stem_len = w.len() - 5;
            if stem_len >= p1 {
                replace_tail(w, 5, "ee");
            }
        } else if ends(w, "edly") {
            delete_and_cleanup(w, p1, 4);
        } else if ends(w, "eed") {
            let stem_len = w.len() - 3;
            if stem_len >= p1 {
                replace_tail(w, 3, "ee");
            }
        } else if ends(w, "ed") {
            delete_and_cleanup(w, p1, 2);
        } else if ends(w, "ingly") {
            delete_and_cleanup(w, p1, 5);
        } else if ends(w, "ing") {
            delete_and_cleanup(w, p1, 3);
        }
    }

    /// Step 1c: a trailing `y`/`Y` becomes `i` if it's preceded by a
    /// consonant and something precedes *that* (i.e. the word is at least
    /// 3 characters -- a lone consonant+y is left alone).
    fn step1c(w: &mut [char]) {
        let n = w.len();
        if n >= 3 && matches!(w[n - 1], 'y' | 'Y') && !is_vowel(w[n - 2]) {
            w[n - 1] = 'i';
        }
    }

    /// Replaces `w`'s trailing `suf_len` characters with `replacement`,
    /// but only if the suffix boundary is at or past `p1` (R1) -- real
    /// Snowball's ubiquitous `r_R1()` guard. Leaves `w` untouched
    /// otherwise (no fallback to a shorter suffix -- matching real
    /// Snowball's `Among`-table semantics, where failing the R1 check on
    /// the longest matched suffix does not retry a shorter one).
    fn apply_if_r1(w: &mut Vec<char>, p1: usize, suf_len: usize, replacement: &str) {
        let stem_len = w.len() - suf_len;
        if stem_len >= p1 {
            replace_tail(w, suf_len, replacement);
        }
    }

    /// Same as [`apply_if_r1`] but gated on R2 (`p2`) and always a
    /// deletion (empty replacement) -- Step 4's shape.
    fn apply_if_r2(w: &mut Vec<char>, p2: usize, suf_len: usize) {
        let stem_len = w.len() - suf_len;
        if stem_len >= p2 {
            w.truncate(stem_len);
        }
    }

    /// Step 2: the long suffix-family table (real Snowball's `a_5`), tried
    /// longest-suffix-first. `"ogi"` additionally requires the character
    /// right before it to be `l` (so only `"logi"` qualifies); `"li"`
    /// additionally requires the character right before it to be one of
    /// the fixed `valid_LI` set (`c`/`d`/`e`/`g`/`h`/`k`/`m`/`n`/`r`/`t`).
    fn step2(w: &mut Vec<char>, p1: usize) {
        const RULES: &[(&str, &str)] = &[
            ("ational", "ate"),
            ("ization", "ize"),
            ("iveness", "ive"),
            ("fulness", "ful"),
            ("ousness", "ous"),
            ("lessli", "less"),
            ("biliti", "ble"),
            ("tional", "tion"),
            ("fulli", "ful"),
            ("ousli", "ous"),
            ("entli", "ent"),
            ("aliti", "al"),
            ("iviti", "ive"),
            ("ation", "ate"),
            ("alism", "al"),
            ("anci", "ance"),
            ("enci", "ence"),
            ("abli", "able"),
            ("alli", "al"),
            ("izer", "ize"),
            ("ator", "ate"),
        ];
        for (suf, rep) in RULES {
            if ends(w, suf) {
                apply_if_r1(w, p1, suf.chars().count(), rep);
                return;
            }
        }
        if ends(w, "ogi") {
            let n = w.len();
            if n >= 4 && w[n - 4] == 'l' {
                apply_if_r1(w, p1, 3, "og");
            }
        } else if ends(w, "bli") {
            apply_if_r1(w, p1, 3, "ble");
        } else if ends(w, "li") {
            let n = w.len();
            if n >= 3
                && matches!(
                    w[n - 3],
                    'c' | 'd' | 'e' | 'g' | 'h' | 'k' | 'm' | 'n' | 'r' | 't'
                )
            {
                apply_if_r1(w, p1, 2, "");
            }
        }
    }

    /// Step 3: the smaller suffix-family table (real Snowball's `a_6`),
    /// longest-suffix-first. `"ative"` is the one entry gated on R2 rather
    /// than R1.
    fn step3(w: &mut Vec<char>, p1: usize, p2: usize) {
        if ends(w, "ative") {
            apply_if_r2(w, p2, 5);
            return;
        }
        const RULES: &[(&str, &str)] = &[
            ("ational", "ate"),
            ("tional", "tion"),
            ("icate", "ic"),
            ("alize", "al"),
            ("iciti", "ic"),
            ("ical", "ic"),
            ("ness", ""),
            ("ful", ""),
        ];
        for (suf, rep) in RULES {
            if ends(w, suf) {
                apply_if_r1(w, p1, suf.chars().count(), rep);
                return;
            }
        }
    }

    /// Step 4: strips a suffix entirely, gated on R2 throughout (real
    /// Snowball's `a_7`); `"ion"` additionally requires the preceding
    /// character to be `s` or `t`.
    fn step4(w: &mut Vec<char>, p2: usize) {
        if ends(w, "ion") {
            let n = w.len();
            let stem_len = n - 3;
            if stem_len >= p2 && stem_len >= 1 && matches!(w[stem_len - 1], 's' | 't') {
                w.truncate(stem_len);
            }
            return;
        }
        const RULES: &[&str] = &[
            "ement", "ance", "ence", "able", "ible", "ment", "ate", "ive", "ize", "iti", "ism",
            "ous", "ant", "ent", "ic", "al", "er",
        ];
        for suf in RULES {
            if ends(w, suf) {
                apply_if_r2(w, p2, suf.chars().count());
                return;
            }
        }
    }

    /// Step 5: a trailing `e` is deleted if R2 holds at its boundary, or
    /// if R1 holds there and the remaining stem does *not* end in a short
    /// syllable; a trailing `ll` collapses to a single `l` if R2 holds.
    fn step5(w: &mut Vec<char>, p1: usize, p2: usize) {
        if ends(w, "e") {
            let stem_len = w.len() - 1;
            let ok = if stem_len >= p2 {
                true
            } else if stem_len >= p1 {
                !ends_short_syllable(&w[..stem_len])
            } else {
                false
            };
            if ok {
                w.truncate(stem_len);
            }
        } else if ends(w, "ll") {
            let stem_len = w.len() - 1;
            if stem_len >= p2 {
                w.truncate(stem_len);
            }
        }
    }

    /// Stems `term`, or returns it unchanged if it contains any character
    /// outside the algorithm's own domain of definition: lowercase ASCII
    /// letters plus a literal `'` (apostrophe), which step 0 explicitly
    /// handles (`"don't"`, `"cats'"`, `"'tis"`).
    pub(super) fn stem(term: &str) -> String {
        if term.is_empty() || !term.chars().all(|c| c.is_ascii_lowercase() || c == '\'') {
            return term.to_string();
        }
        if let Some(mapped) = exception1(term) {
            return mapped.to_string();
        }
        if term.chars().count() < 3 {
            return term.to_string();
        }
        let mut w: Vec<char> = term.chars().collect();
        prelude(&mut w);
        let (p1, p2) = mark_regions(&w);
        step0_and_1a(&mut w);
        if !is_exception2(&w) {
            step1b(&mut w, p1);
            step1c(&mut w);
            step2(&mut w, p1);
            step3(&mut w, p1, p2);
            step4(&mut w, p2);
            step5(&mut w, p1, p2);
        }
        for c in w.iter_mut() {
            if *c == 'Y' {
                *c = 'y';
            }
        }
        w.into_iter().collect()
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
        // Five Java `char`s ("cafe" + the combining mark), six UTF-8 bytes.
        assert_eq!(tokens[0].end_offset, 5);
        assert_eq!(tokens[1].term, "today");
        assert_eq!((tokens[1].start_offset, tokens[1].end_offset), (6, 11));
    }

    #[test]
    fn tokenize_cjk_ideographs_split_one_per_character() {
        // Each Han ideograph is its own token -- no word clustering across
        // CJK text, unlike Latin script.
        // Each ideograph is one UTF-16 code unit and three UTF-8 bytes.
        let tokens = tokenize("你好世界");
        assert_eq!(
            tokens,
            vec![
                tok("你", 0, 1, 1),
                tok("好", 1, 2, 1),
                tok("世", 2, 3, 1),
                tok("界", 3, 4, 1),
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
                tok("世", 6, 7, 1),
                tok("界", 7, 8, 1),
                tok("world", 9, 14, 1),
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
        // U+1F44D is one scalar, **two** UTF-16 code units and four UTF-8
        // bytes, so "emoji" starts at Java `char` 6, not scalar 5 or byte 8.
        let tokens = tokenize("test\u{1F44D}emoji");
        assert_eq!(tokens, vec![tok("test", 0, 4, 1), tok("emoji", 6, 11, 1)]);
    }

    #[test]
    fn lowercase_filter_changes_case_not_offsets_or_positions() {
        let tokens = vec![tok("THE", 0, 3, 1), tok("Quick", 4, 9, 2)];
        let out = LowerCaseFilter::apply(tokens);
        assert_eq!(out, vec![tok("the", 0, 3, 1), tok("quick", 4, 9, 2),]);
    }

    // ---- the TokenStream lifecycle (`end()`) --------------------------

    /// `Tokenizer.end()`'s `finalOffset = correctOffset(charCount)` is the
    /// length of the **whole input**, not the end of the last token -- which is
    /// what a multi-valued field's next value is offset by.
    #[test]
    fn tokenize_stream_ends_at_the_inputs_own_length_not_the_last_token() {
        let stream = tokenize_stream("fox   ");
        assert_eq!(stream.tokens.len(), 1);
        assert_eq!(stream.tokens[0].end_offset, 3);
        assert_eq!(stream.final_offset, 6);
        assert_eq!(stream.final_position_increment, 0);
        // UTF-16 code units, like every other offset in this crate.
        assert_eq!(tokenize_stream("\u{1D400} x").final_offset, 4);
        // Empty input: no tokens, and an end offset of 0.
        let empty = tokenize_stream("");
        assert!(empty.tokens.is_empty());
        assert_eq!(empty.final_offset, 0);
    }

    /// `FilteringTokenFilter.end()`: `skippedPositions` left over at end of
    /// stream is published on the position-increment attribute, so trailing
    /// stopwords still advance the field's position counter.
    #[test]
    fn stop_filter_end_publishes_the_trailing_skipped_positions() {
        let stopwords: HashSet<String> = ["the"].into_iter().map(String::from).collect();
        let stream = Analyzer::standard(Some(&stopwords)).analyze_stream("fox the the");
        assert_eq!(
            stream
                .tokens
                .iter()
                .map(|t| t.term.as_str())
                .collect::<Vec<_>>(),
            vec!["fox"]
        );
        assert_eq!(stream.final_position_increment, 2);
        assert_eq!(stream.final_offset, 11);

        // Nothing trailing to skip: the base `TokenStream.end()`'s 0.
        let stream = Analyzer::standard(Some(&stopwords)).analyze_stream("the fox");
        assert_eq!(stream.final_position_increment, 0);

        // Every token filtered out: all of them land on the final increment,
        // because no surviving token could carry them.
        let stream = Analyzer::standard(Some(&stopwords)).analyze_stream("the the the");
        assert!(stream.tokens.is_empty());
        assert_eq!(stream.final_position_increment, 3);
    }

    /// `StopFilter::apply` is `apply_to_stream` with the end-of-stream values
    /// dropped, which is what every caller that only wants the tokens gets.
    #[test]
    fn stop_filter_apply_and_apply_to_stream_agree_on_the_tokens() {
        let stopwords: HashSet<String> = ["the"].into_iter().map(String::from).collect();
        let tokens = tokenize("fox the the");
        let via_apply = StopFilter::apply(tokens.clone(), &stopwords);
        let via_stream = StopFilter::apply_to_stream(
            TokenStream {
                tokens,
                final_position_increment: 0,
                final_offset: 11,
            },
            &stopwords,
        );
        assert_eq!(via_apply, via_stream.tokens);
        assert_eq!(via_stream.final_offset, 11);
    }

    /// `NGramTokenFilter.end()` / `EdgeNGramTokenFilter.end()`:
    /// `posIncrAtt.setPositionIncrement(curPosIncr)` -- and note it **sets**
    /// rather than adds, so an upstream filter's own end-of-stream increment is
    /// discarded. Both halves are asserted, because the overwrite is the
    /// surprising one.
    #[test]
    fn ngram_end_publishes_the_leftover_increment_and_overwrites_the_upstream_one() {
        // "ab" is shorter than min_gram 3, so it emits nothing and its
        // increment is still owed at end of stream.
        let stream = TokenStream {
            tokens: tokenize("abcd ab"),
            final_position_increment: 7,
            final_offset: 7,
        };
        let out = NGramTokenFilter::apply_to_stream(stream, 3, 3, false).unwrap();
        assert_eq!(
            out.tokens
                .iter()
                .map(|t| t.term.as_str())
                .collect::<Vec<_>>(),
            vec!["abc", "bcd"]
        );
        assert_eq!(out.final_position_increment, 1, "the skipped \"ab\"");
        assert_eq!(out.final_offset, 7);

        let stream = TokenStream {
            tokens: tokenize("abcd ab"),
            final_position_increment: 7,
            final_offset: 7,
        };
        let out = EdgeNGramTokenFilter::apply_to_stream(stream, 3, 3, false).unwrap();
        assert_eq!(
            out.tokens
                .iter()
                .map(|t| t.term.as_str())
                .collect::<Vec<_>>(),
            vec!["abc"]
        );
        assert_eq!(out.final_position_increment, 1);

        // preserveOriginal emits the short token, so nothing is owed.
        let stream = TokenStream {
            tokens: tokenize("abcd ab"),
            final_position_increment: 0,
            final_offset: 7,
        };
        let out = NGramTokenFilter::apply_to_stream(stream, 3, 3, true).unwrap();
        assert_eq!(out.final_position_increment, 0);
        assert!(out.tokens.iter().any(|t| t.term == "ab"));
    }

    /// A keyword analyzer's stream: one token, no swallowed positions, and the
    /// `KeywordTokenizer.end()` final offset.
    #[test]
    fn keyword_analyzer_stream_ends_at_the_inputs_length() {
        let stream = Analyzer::keyword().analyze_stream("id-\u{1F600}");
        assert_eq!(stream.tokens.len(), 1);
        assert_eq!(stream.final_position_increment, 0);
        assert_eq!(stream.final_offset, 5);
    }

    /// The two gap accessors are Java's `Analyzer.getPositionIncrementGap` /
    /// `getOffsetGap`, defaulting to Java's own `0` and `1`.
    #[test]
    fn the_analyzer_gaps_default_to_javas_zero_and_one() {
        let a = Analyzer::standard(None);
        assert_eq!(a.position_increment_gap(), 0);
        assert_eq!(a.offset_gap(), 1);
        let a = a.with_position_increment_gap(100).with_offset_gap(3);
        assert_eq!(a.position_increment_gap(), 100);
        assert_eq!(a.offset_gap(), 3);
        assert_eq!(Analyzer::keyword().offset_gap(), 1);
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
    fn french_stop_words_has_exactly_154_entries_matching_real_lucene() {
        // Real Lucene's FrenchAnalyzer.getDefaultStopSet() loads
        // french_stop.txt (Snowball's French stop list) via
        // WordlistLoader.getSnowballWordSet, which strips each line's `|`
        // comment and splits the remainder on whitespace. Transcribing that
        // file (org/apache/lucene/analysis/snowball/french_stop.txt in a
        // real Lucene 10.5.0 checkout) the same way yields exactly 154
        // surviving words -- several classic French stopwords (e.g. "son",
        // "sommes", "été", "est", "avions", "auras", "aura", "as", "fût")
        // are deliberately commented out of that file as homonyms of
        // unrelated common words, so this is not simply "every French
        // function word".
        let stopwords = french_stop_words();
        assert_eq!(
            stopwords.len(),
            154,
            "FRENCH_STOP_WORDS must have exactly 154 entries, matching real \
             Lucene's french_stop.txt after comment-stripping"
        );
        assert_eq!(FRENCH_STOP_WORDS.len(), 154);
        // Spot-check a representative sample spanning every distinct source
        // section of french_stop.txt: core articles/pronouns, single-letter
        // elision remnants, and inflected forms of être/avoir.
        for word in [
            "le", "la", "les", "un", "une", "et", "de", "du", "des", "dans", "sur", "pour", "qu",
            "que", "qui", "c", "d", "j", "l", "à", "m", "n", "s", "t", "y", "suis", "es", "sont",
            "serai", "était", "avons", "avez", "ont", "aurai", "eusse", "ceci", "cela", "quel",
            "sans", "soi",
        ] {
            assert!(
                stopwords.contains(word),
                "canonical French stopword {word:?} is missing from FRENCH_STOP_WORDS"
            );
        }
        // Deliberately-omitted homonyms (real Lucene's own comments explain
        // why): must NOT be present.
        for word in [
            "son", "sommes", "été", "étés", "est", "avions", "auras", "aura", "as", "fût",
        ] {
            assert!(
                !stopwords.contains(word),
                "{word:?} is a homonym real Lucene deliberately omits from its default \
                 French stop set -- it must not be in FRENCH_STOP_WORDS either"
            );
        }
    }

    #[test]
    fn french_stop_words_case_is_already_lowercase() {
        for word in FRENCH_STOP_WORDS {
            assert_eq!(
                *word,
                word.to_lowercase(),
                "{word:?} must be stored lowercase"
            );
        }
    }

    #[test]
    fn french_stop_words_does_not_false_positive_on_content_words() {
        let stopwords = french_stop_words();
        for word in ["chat", "souris", "maison", "recherche", "document"] {
            assert!(!stopwords.contains(word), "{word:?} must NOT be a stopword");
        }
        let tokens = tokenize("Le chat et la souris sont dans la maison");
        let tokens = LowerCaseFilter::apply(tokens);
        let out = StopFilter::apply(tokens, &stopwords);
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["chat", "souris", "maison"]);
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
        // Offsets are the *source* text's Java `char` spans: folding grows
        // no term here, but each accented letter is 2 UTF-8 bytes and 1
        // UTF-16 code unit, so these are 0,4 / 5,10 / 11,16 -- what real
        // Lucene's `fold_then_lower` fixture case records.
        let out = analyzer.analyze("Café Naïve ÉCOLE");
        assert_eq!(
            out,
            vec![
                tok("cafe", 0, 4, 1),
                tok("naive", 5, 10, 1),
                tok("ecole", 11, 16, 1),
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

    /// Java's `PorterStemmer` has **no** "lowercase ASCII only" guard: its
    /// `cons()` treats every character that is not `a/e/i/o/u/y` as a
    /// consonant, so an uppercase or accented word is stemmed like any other.
    /// This port used to bail out on such words, which meant `"Running"`
    /// survived a `PorterStemFilter` that Lucene would have reduced to
    /// `"Run"` -- a real divergence for any chain that does not lowercase
    /// first.
    #[test]
    fn porter_stem_has_no_ascii_lowercase_guard() {
        // "Running": step 1 strips `-ing` and undoubles `nn`, exactly as it
        // would for "running".
        assert_eq!(
            PorterStemFilter::apply(vec![tok("Running", 0, 7, 1)])[0].term,
            "Run"
        );
        // Words with no matching suffix still come back untouched, and
        // nothing panics on non-ASCII or digits.
        for word in ["café", "123", "Cat"] {
            let out = PorterStemFilter::apply(vec![tok(word, 0, 3, 1)]);
            assert_eq!(out[0].term, word, "stemming {word:?}");
        }
        // ... and "Cats" loses its plural `s` the same way "cats" does.
        assert_eq!(
            PorterStemFilter::apply(vec![tok("Cats", 0, 4, 1)])[0].term,
            "Cat"
        );
    }

    /// Java runs the six steps only when `k > k0 + 1`, so a one- or
    /// two-character word is returned verbatim. Without that guard step 1a
    /// deleted the whole of `"s"`, producing a **zero-length term**.
    #[test]
    fn porter_stem_leaves_one_and_two_character_words_alone() {
        for word in ["s", "as", "is", "us", "es", "ay", "a", ""] {
            let out = PorterStemFilter::apply(vec![tok(word, 0, 1, 1)]);
            assert_eq!(out[0].term, word, "stemming {word:?}");
        }
    }

    /// The three step-3/step-5 rules this port had wrong: `bli -> ble` was
    /// written as `abli -> able`, `logi -> log` was missing outright, and a
    /// suffix whose measure test fails must stop the search rather than fall
    /// through to a shorter suffix.
    #[test]
    fn porter_stem_first_matching_suffix_wins_even_when_its_measure_test_fails() {
        let cases: &[(&str, &str)] = &[
            // `bli -> ble`, not `abli -> able`.
            ("possibly", "possibl"),
            // The `g` group, which was absent.
            ("technology", "technolog"),
            ("apology", "apolog"),
            // ... and `logi` still only fires when `m(stem) > 0`, so "bio"
            // (measure 0) blocks the replacement without falling through.
            ("biology", "biologi"),
            // `-ment` matches first with m("argu") == 1, which is not > 1, so
            // Java stops there instead of falling through to `-ent`.
            ("argument", "argument"),
            // `-ization` matches with m("") == 0, so no `-ation` fallback.
            ("ization", "izat"),
        ];
        for (input, expected) in cases {
            let out = PorterStemFilter::apply(vec![tok(input, 0, 1, 1)]);
            assert_eq!(out[0].term, *expected, "stemming {input:?}");
        }
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
            // Verified against real `SynonymGraphFilter`:
            // `wifi:1:2:0,5;wi:0:1:0,2;fi:1:1:3,5`.
            vec![
                tok_len("wifi", 0, 5, 1, 2),
                tok("wi", 0, 2, 0),
                tok("fi", 3, 5, 1),
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
            // Real `SynonymGraphFilter`:
            // `united:1:1:0,3;usa:0:4:0,3;states:1:1:0,3;of:1:1:0,3;america:1:1:0,3`.
            // Note the *original* gets position_length 4: both paths have to
            // rejoin at the same end node.
            vec![
                tok_len("united", 0, 3, 1, 1),
                tok_len("usa", 0, 3, 0, 4),
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
            // Real `SynonymGraphFilter`:
            // `big:1:1:0,8;new:0:2:0,3;apple:1:2:0,8;york:1:1:4,8`.
            vec![
                tok_len("big", 0, 8, 1, 1),
                tok_len("new", 0, 3, 0, 2),
                tok_len("apple", 0, 8, 1, 2),
                tok("york", 4, 8, 1),
            ]
        );
    }

    /// The graph's node counter has to keep advancing across tokens that
    /// match nothing, or a later match starts at the wrong node. Real
    /// `SynonymGraphFilter` over `"the wi fi router"` gives
    /// `the:1:1:0,3;wifi:1:2:4,9;wi:0:1:4,6;fi:1:1:7,9;router:1:1:10,16`.
    #[test]
    fn synonym_filter_multiword_match_surrounded_by_pass_through_tokens() {
        let tokens = vec![
            tok("the", 0, 3, 1),
            tok("wi", 4, 6, 1),
            tok("fi", 7, 9, 1),
            tok("router", 10, 16, 1),
        ];
        let rules = vec![SynonymRule {
            input: vec!["wi".to_string(), "fi".to_string()],
            outputs: vec![vec!["wifi".to_string()]],
        }];
        let out = SynonymFilter::apply_multiword(tokens, &rules);
        assert_eq!(
            out,
            vec![
                tok("the", 0, 3, 1),
                tok_len("wifi", 4, 9, 1, 2),
                tok("wi", 4, 6, 0),
                tok("fi", 7, 9, 1),
                tok("router", 10, 16, 1),
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
            // Real `SynonymGraphFilter`:
            // `nyc:1:2:0,8;new:0:1:0,3;york:1:1:4,8`.
            vec![
                tok_len("nyc", 0, 8, 1, 2),
                tok("new", 0, 3, 0),
                tok("york", 4, 8, 1),
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
            // Real `SynonymGraphFilter`:
            // `wifi:1:2:0,5;wireless:0:2:0,5;wi:0:1:0,2;fi:1:1:3,5`.
            vec![
                tok_len("wifi", 0, 5, 1, 2),
                tok_len("wireless", 0, 5, 0, 2),
                tok("wi", 0, 2, 0),
                tok("fi", 3, 5, 1),
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
        assert_eq!(out, vec![tok("café", 0, 4, 1)]);
    }

    // -- Analyzer::keyword (task #208) --

    #[test]
    fn keyword_analyzer_emits_whole_text_as_one_unmodified_token() {
        let analyzer = Analyzer::keyword();
        let out = analyzer.analyze("Status: ACTIVE!");
        assert_eq!(out, vec![tok("Status: ACTIVE!", 0, 15, 1)]);
    }

    #[test]
    fn keyword_analyzer_does_not_split_on_whitespace() {
        // Distinct from Analyzer::standard/tokenize -- whitespace-separated
        // words are NOT split into separate tokens.
        let analyzer = Analyzer::keyword();
        let out = analyzer.analyze("the quick fox");
        assert_eq!(out, vec![tok("the quick fox", 0, 13, 1)]);
    }

    #[test]
    fn keyword_analyzer_does_not_lowercase() {
        let analyzer = Analyzer::keyword();
        let out = analyzer.analyze("UPPER");
        assert_eq!(out, vec![tok("UPPER", 0, 5, 1)]);
    }

    #[test]
    fn keyword_analyzer_ignores_stopwords_even_if_term_matches() {
        // A term that would be a stopword under Analyzer::standard's
        // pipeline is passed through untouched -- keyword semantics have no
        // stopword filtering at all.
        let analyzer = Analyzer::keyword();
        let out = analyzer.analyze("the");
        assert_eq!(out, vec![tok("the", 0, 3, 1)]);
    }

    #[test]
    fn keyword_analyzer_empty_input_still_emits_one_empty_token() {
        // Matches real Lucene's KeywordTokenizer: incrementToken()
        // unconditionally succeeds once, even over zero characters.
        let analyzer = Analyzer::keyword();
        let out = analyzer.analyze("");
        assert_eq!(out, vec![tok("", 0, 0, 1)]);
    }

    #[test]
    fn keyword_analyzer_end_offset_is_the_java_char_count() {
        // Java's KeywordTokenizer ends its one token at
        // `correctOffset(charCount)`. "id-<U+1F600>-<U+00E9>" is 6 Unicode
        // scalars, **7** UTF-16 code units and 10 UTF-8 bytes, so all three
        // candidate units are visibly distinct here.
        let text = "id-\u{1F600}-\u{E9}";
        assert_eq!(text.chars().count(), 6);
        assert_eq!(text.len(), 10);
        let out = Analyzer::keyword().analyze(text);
        assert_eq!(out, vec![tok(text, 0, 7, 1)]);
    }

    #[test]
    fn tokenize_offsets_are_utf16_code_units_not_bytes_or_scalars() {
        // One text where all three units disagree at once: an accented letter
        // (1 char / 2 bytes), an ideograph (1 char / 3 bytes) and an astral
        // symbol (2 chars / 1 scalar / 4 bytes).
        let text = "alpha caf\u{E9} \u{4E16} \u{1D306} omega";
        let tokens = tokenize(text);
        assert_eq!(
            tokens,
            vec![
                tok("alpha", 0, 5, 1),
                tok("caf\u{E9}", 6, 10, 1),
                tok("\u{4E16}", 11, 12, 1),
                tok("omega", 16, 21, 1),
            ]
        );
        // Not the byte offsets (which would put "omega" at 21)...
        assert_eq!(text.len(), 26);
        // ...and not the scalar offsets either (which would put it at 15).
        assert_eq!(text.chars().count(), 20);
    }

    #[test]
    fn utf16_len_agrees_with_encode_utf16_on_both_branches() {
        // The ASCII fast path and the general path must not disagree.
        for text in [
            "",
            "plain ascii",
            "caf\u{E9}",
            "\u{4E16}\u{754C}",
            "a\u{1F600}b",
            "e\u{301}",
        ] {
            assert_eq!(
                utf16_len(text),
                text.encode_utf16().count(),
                "utf16_len disagreed for {text:?}"
            );
        }
    }

    #[test]
    fn keyword_analyzer_builder_methods_have_no_effect() {
        // Calling any of Analyzer's filter-chain builders on a keyword
        // analyzer doesn't change analyze()'s output -- keyword mode
        // short-circuits before any of those fields are consulted.
        let mut synonyms = HashMap::new();
        synonyms.insert("the".to_string(), vec!["a".to_string()]);

        let plain = Analyzer::keyword().analyze("the");
        let with_everything = Analyzer::keyword()
            .with_ascii_folding()
            .with_stemming()
            .with_synonyms(synonyms)
            .analyze("the");
        assert_eq!(plain, with_everything);
        assert_eq!(plain, vec![tok("the", 0, 3, 1)]);
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
        // Every gram carries the *input token's* offsets: Java restores the
        // captured state per gram and never calls `setOffset`.
        assert!(out.iter().all(|t| (t.start_offset, t.end_offset) == (0, 5)));
    }

    #[test]
    fn edge_ngram_filter_abcde_min2_max4_exact_prefix_gram_set() {
        let tokens = vec![tok("abcde", 0, 5, 1)];
        let out = EdgeNGramTokenFilter::apply(tokens, 2, 4).unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["ab", "abc", "abcd"]);
        let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
        assert_eq!(pos_incs, vec![1, 0, 0]);
        assert!(out.iter().all(|t| (t.start_offset, t.end_offset) == (0, 5)));
    }

    /// Java accumulates `curPosIncr` across input tokens that emit nothing,
    /// so a token skipped for being shorter than `min_gram` still pushes the
    /// next token's grams along. This port used to drop that increment.
    #[test]
    fn ngram_filter_carries_the_increment_of_a_skipped_short_token() {
        let tokens = vec![tok("big", 0, 3, 1), tok("a", 4, 5, 1), tok("cat", 6, 9, 1)];
        let out = NGramTokenFilter::apply(tokens, 3, 3).unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["big", "cat"]);
        // "cat" is two positions after "big", not one: "a" produced no gram
        // but still consumed a position.
        let pos_incs: Vec<i32> = out.iter().map(|t| t.position_increment).collect();
        assert_eq!(pos_incs, vec![1, 2]);
    }

    /// `NGramTokenFilter`'s `preserveOriginal` flag: too-short tokens are
    /// kept (carrying the accumulated increment) and too-long tokens are
    /// re-emitted after their grams at increment 0.
    #[test]
    fn ngram_filter_preserve_original_keeps_short_and_long_tokens() {
        let out = NGramTokenFilter::apply_preserving_original(
            vec![tok("a", 0, 1, 1), tok("abcd", 2, 6, 1)],
            2,
            3,
        )
        .unwrap();
        let observed: Vec<(&str, i32)> = out
            .iter()
            .map(|t| (t.term.as_str(), t.position_increment))
            .collect();
        assert_eq!(
            observed,
            vec![
                ("a", 1),
                ("ab", 1),
                ("abc", 0),
                ("bc", 0),
                ("bcd", 0),
                ("cd", 0),
                // "abcd" is longer than max_gram, so the original comes back.
                ("abcd", 0),
            ]
        );

        // A token exactly `max_gram` long is not duplicated.
        let out =
            NGramTokenFilter::apply_preserving_original(vec![tok("abc", 0, 3, 1)], 2, 3).unwrap();
        let terms: Vec<&str> = out.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["ab", "abc", "bc"]);
    }

    #[test]
    fn edge_ngram_filter_preserve_original_keeps_short_and_long_tokens() {
        let out = EdgeNGramTokenFilter::apply_preserving_original(
            vec![tok("a", 0, 1, 1), tok("abcd", 2, 6, 1)],
            2,
            3,
        )
        .unwrap();
        let observed: Vec<(&str, i32)> = out
            .iter()
            .map(|t| (t.term.as_str(), t.position_increment))
            .collect();
        assert_eq!(observed, vec![("a", 1), ("ab", 1), ("abc", 0), ("abcd", 0)]);
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
        // Offsets stay the input token's, as Lucene leaves them.
        assert!(out.iter().all(|t| (t.start_offset, t.end_offset) == (0, 5)));
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

    // -- SnowballEnglishStemFilter (task #209, Porter2/Snowball) --

    fn snowball(term: &str) -> String {
        SnowballEnglishStemFilter::apply(vec![tok(term, 0, 1, 1)])[0]
            .term
            .clone()
    }

    #[test]
    fn snowball_english_composed_via_analyzer_builder() {
        // Analyzer::with_snowball_stemming wires SnowballEnglishStemFilter
        // in as the analyzer's last stage, mirroring with_stemming's shape.
        let tokens = Analyzer::standard(None)
            .with_snowball_stemming()
            .analyze("running dogs");
        let terms: Vec<&str> = tokens.iter().map(|t| t.term.as_str()).collect();
        assert_eq!(terms, vec!["run", "dog"]);
    }

    #[test]
    fn snowball_english_snowball_stemming_takes_precedence_over_classic() {
        // If both builders are enabled, the Snowball stemmer wins (no
        // double-stemming) -- use a word where the two algorithms are known
        // to diverge: classic Porter's step 1b cleanup only re-appends a
        // trailing `e` when the 2-letter "CVC" check applies (which
        // requires 3+ characters, so never fires for a 2-letter stem),
        // giving "owed" -> "ow"; Snowball's own short-syllable definition
        // explicitly covers the 2-letter vowel+consonant case, giving
        // "owed" -> "owe" (see the dedicated
        // `snowball_english_short_word_append_e_fallback` test).
        let tokens = Analyzer::standard(None)
            .with_stemming()
            .with_snowball_stemming()
            .analyze("owed");
        assert_eq!(tokens[0].term, "owe");
    }

    #[test]
    fn snowball_english_leading_apostrophe_stripped_by_prelude() {
        // Real StandardTokenizer strips a bare leading apostrophe before
        // the stemmer ever sees it (confirmed by this crate's own
        // differential fixture, where "'tis" tokenizes to the term "tis"
        // directly) -- so this exercises SnowballEnglishStemFilter's own
        // prelude apostrophe-stripping directly, on a term a caller might
        // construct without going through tokenize() first.
        assert_eq!(snowball("'tis"), "tis");
    }

    #[test]
    fn snowball_english_step0_apostrophe_suffix_variants() {
        // Step 0's three apostrophe-suffix patterns, longest-match-first:
        // trailing "'s'" (rare, but part of the real Among table), "'s"
        // (possessive), and a bare trailing "'".
        assert_eq!(snowball("ab's'"), "ab");
        assert_eq!(snowball("dog's"), "dog");
        assert_eq!(snowball("boys'"), "boy"); // "'" stripped, then plain "s" stripped too.
    }

    #[test]
    fn snowball_english_step1a_suffix_family() {
        assert_eq!(snowball("caresses"), "caress"); // sses -> ss
        assert_eq!(snowball("ponies"), "poni"); // ies, stem len > 1 -> i
        assert_eq!(snowball("ties"), "tie"); // ies, stem len <= 1 -> ie
        assert_eq!(snowball("caress"), "caress"); // ss protected, no-op
        assert_eq!(snowball("virus"), "virus"); // us protected, no-op
        assert_eq!(snowball("cats"), "cat"); // plain s, vowel precedes
        assert_eq!(snowball("gas"), "gas"); // plain s, no vowel before the "as" -> unchanged
    }

    #[test]
    fn snowball_english_exception1_whole_word_table() {
        // The whole-word exception table checked before anything else --
        // some entries remap, some are explicitly left unchanged.
        let cases: &[(&str, &str)] = &[
            ("skis", "ski"),
            ("skies", "sky"),
            ("dying", "die"),
            ("lying", "lie"),
            ("tying", "tie"),
            ("idly", "idl"),
            ("gently", "gentl"),
            ("ugly", "ugli"),
            ("early", "earli"),
            ("only", "onli"),
            ("singly", "singl"),
            ("andes", "andes"),
            ("atlas", "atlas"),
            ("bias", "bias"),
            ("cosmos", "cosmos"),
            ("howe", "howe"),
            ("news", "news"),
            ("sky", "sky"),
        ];
        for (input, expected) in cases {
            assert_eq!(snowball(input), *expected, "stemming {input:?}");
        }
    }

    #[test]
    fn snowball_english_exception2_protected_stems() {
        // The second whole-word protected-stem table -- checked after
        // step 1a but before step 1b -- covers both the `-eed` family
        // (succeed/proceed/exceed) and the `-ing`-as-not-a-gerund family
        // (canning/inning/earring/herring/outing); all eight are left
        // completely unchanged.
        for word in [
            "succeed", "proceed", "exceed", "canning", "inning", "earring", "herring", "outing",
        ] {
            assert_eq!(snowball(word), word, "stemming {word:?}");
        }
    }

    #[test]
    fn snowball_english_double_consonant_aeo_exception() {
        // a3_cleanup's narrow carve-out: a doubled consonant immediately
        // preceded by a/e/o is left alone when that's the *entire* stem
        // (three letters total) -- "added" -> stem "add" (3 letters,
        // starts with "a") keeps both d's, unlike "hopping"/"tanned" (the
        // ordinary double-consonant-drop case, already covered by the
        // cross-engine fixture).
        assert_eq!(snowball("added"), "add");
        assert_eq!(snowball("hopping"), "hop");
        assert_eq!(snowball("tanned"), "tan");
    }

    #[test]
    fn snowball_english_short_word_append_e_fallback() {
        // a3_cleanup's "" fallback: append `e` back if the word is now
        // both short (R1 empty) and ends in a short syllable -- "hoped"
        // (stem "hop" after "-ed" deletion, itself CVC and exactly at R1's
        // start) restores the `e` to "hope"; "owed" (stem "ow", the
        // whole-word-is-vowel-then-consonant short-syllable case) restores
        // to "owe".
        assert_eq!(snowball("hoped"), "hope");
        assert_eq!(snowball("owed"), "owe");
    }

    #[test]
    fn snowball_english_eed_family_ee_replacement() {
        // "-eed"/"-eedly", R1-gated, unconditionally replaced with "ee"
        // (the succ/proc/exc protection this branch used to need in an
        // older generation of the algorithm is now handled globally by
        // the exception2 table, see the dedicated test above).
        assert_eq!(snowball("agreed"), "agre");
        assert_eq!(snowball("feed"), "feed"); // m(fe) == 0 at R1 -> R1 not yet reached, unchanged
    }

    #[test]
    fn snowball_english_ed_ing_edly_ingly_shared_cleanup() {
        assert_eq!(snowball("motoring"), "motor"); // plain -ing deletion + cleanup
        assert_eq!(snowball("plastered"), "plaster"); // plain -ed deletion + cleanup
        assert_eq!(snowball("sing"), "sing"); // no vowel before "-ing" -> unchanged
        assert_eq!(snowball("bled"), "bled"); // no vowel before "-ed" -> unchanged
    }

    #[test]
    fn snowball_english_step1c_trailing_y() {
        assert_eq!(snowball("cry"), "cri"); // y preceded by consonant -> i
        assert_eq!(snowball("toy"), "toy"); // y preceded by vowel -> unchanged
    }

    #[test]
    fn snowball_english_step2_suffix_families() {
        assert_eq!(snowball("carelessly"), "careless"); // lessli -> less
                                                        // fulli -> ful (step2), then step3's own "ful" entry deletes it
                                                        // entirely (same R1 boundary reached again) -> "care".
        assert_eq!(snowball("carefully"), "care");
        assert_eq!(snowball("analogy"), "analog"); // "logi" (preceded by l) -> "log"
                                                   // bli (not abli) -> ble (step2), then step5 deletes the trailing
                                                   // "e" it just added (R1 holds, R2 doesn't, and the resulting stem
                                                   // is not a short syllable) -> "trembl".
        assert_eq!(snowball("trembly"), "trembl");
        assert_eq!(snowball("national"), "nation"); // tional -> tion
    }

    #[test]
    fn snowball_english_step3_and_step4_and_step5() {
        assert_eq!(snowball("triplicate"), "triplic"); // step3 icate -> ic
        assert_eq!(snowball("hopefulness"), "hope"); // step2 fulness -> ful, then step3 ful -> deleted
        assert_eq!(snowball("controll"), "control"); // step5 ll -> l (R2)
        assert_eq!(snowball("roll"), "roll"); // step5 ll unchanged, R2 not reached
        assert_eq!(snowball("rate"), "rate"); // step5 e kept: R1 holds, but ends in short syllable
        assert_eq!(snowball("probate"), "probat"); // step5 e dropped: R2 holds
    }

    #[test]
    fn snowball_english_eedly_edly_ingly_variants() {
        // "eedly" (eed-family, replaced with "ee", then step5 deletes the
        // trailing "e" it left since R1 holds but R2 doesn't and the
        // result isn't a short syllable) -- "edly"/"ingly" (shared
        // delete+cleanup family, same as plain "ed"/"ing").
        assert_eq!(snowball("agreedly"), "agre");
        assert_eq!(snowball("reportedly"), "report");
        assert_eq!(snowball("lastingly"), "last");
    }

    #[test]
    fn snowball_english_leading_y_marked_as_consonant() {
        // prelude marks a word-initial "y" as the internal 'Y' consonant
        // marker; postlude reverts it back to lowercase "y" in the output
        // regardless of whether any step touched it.
        assert_eq!(snowball("yellow"), "yellow");
    }

    #[test]
    fn snowball_english_short_word_passes_through_unchanged() {
        // Below the 3-character minimum, per the algorithm's own domain.
        assert_eq!(snowball("at"), "at");
    }

    #[test]
    fn snowball_english_non_domain_terms_pass_through_unchanged() {
        // Uppercase, digits, and other non-ASCII-letter/apostrophe
        // characters are outside the algorithm's domain of definition --
        // passed through verbatim, never a panic.
        for term in ["Running", "3.14", "café", "", "42"] {
            assert_eq!(snowball(term), term, "term {term:?}");
        }
    }
}
