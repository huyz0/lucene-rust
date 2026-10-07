//! The per-language packages of `lucene-analysis-common` (`analysis/ar`,
//! `analysis/de`, ...): one module per Java package, each holding that
//! package's analyzer, stemmers, normalizers and stop set.
//!
//! The light and minimal stemmers and the normalizers are Java's
//! `int stem(char[] s, int len)` over the term's UTF-16 units, ported line
//! for line as [`CharStemmer`]s; their `TokenFilter`s are all the same three
//! lines of Java (stem unless `KeywordAttribute` is set; a normalizer
//! ignores it), which is [`StemFilter`] / [`NormalizeFilter`] here.
//!
//! Stop sets are Lucene's own resource files, vendored verbatim under
//! `lang/stopwords/` (their headers carry their licences, see
//! `docs/licences.md`) and parsed once.

// The stemmers' `match` arms list Java's `case` labels one by one (`0xE0 |
// 0xE1 | 0xE2`), for comparison with Lucene's source, rather than as ranges.
#![allow(clippy::manual_range_patterns)]

use std::sync::Arc;

use crate::java_character::to_lower_case;
use crate::miscellaneous::SetKeywordMarkerFilter;
use crate::snowball::{SnowballFilter, SnowballStemmer};
use crate::token_stream::{TokenFilter, TokenStream};
use crate::{AnalysisError, CharArraySet, LowerCaseFilter, StandardTokenizer, StopFilter};

/// A Java `stem(char[] s, int len)` / `normalize(char[] s, int len)`: edits
/// the first `len` units of `s` (whose length is `len`; it may grow it) and
/// returns the new length.
pub trait CharStemmer: Send + Sync {
    /// The Java method.
    fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize;
}

/// The `TokenFilter` over a [`CharStemmer`]; `KEYWORD` skips a keyword term
/// (the stem filters), else every term is rewritten (the normalization
/// filters).
pub struct CharStemFilter<I, S, const KEYWORD: bool> {
    input: I,
    stemmer: S,
    buf: Vec<u16>,
    /// The term's units before stemming, to skip an unneeded write-back.
    orig: Vec<u16>,
}

/// A stem filter: `if (!keywordAttr.isKeyword()) termAtt.setLength(stemmer.stem(...))`.
pub type StemFilter<I, S> = CharStemFilter<I, S, true>;
/// A normalization filter: `termAtt.setLength(normalizer.normalize(...))`.
pub type NormalizeFilter<I, S> = CharStemFilter<I, S, false>;

impl<I: TokenStream, S: CharStemmer + Default, const KEYWORD: bool> CharStemFilter<I, S, KEYWORD> {
    /// `new XxxFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        Self::with_stemmer(input, S::default())
    }
}

impl<I: TokenStream, S: CharStemmer, const KEYWORD: bool> CharStemFilter<I, S, KEYWORD> {
    /// The filter over a configured stemmer.
    pub fn with_stemmer(input: I, stemmer: S) -> Self {
        CharStemFilter {
            input,
            stemmer,
            buf: Vec::new(),
            orig: Vec::new(),
        }
    }
}

impl<I: TokenStream, S: CharStemmer, const KEYWORD: bool> TokenFilter
    for CharStemFilter<I, S, KEYWORD>
{
    crate::filter_input!();

    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if !(KEYWORD && a.is_keyword()) {
            let (buf, orig) = (&mut self.buf, &mut self.orig);
            buf.clear();
            let term = a.term();
            let ascii = term.is_ascii();
            if ascii {
                buf.extend(term.bytes().map(u16::from));
            } else {
                crate::util::push_utf16(term, buf);
                orig.clear();
                orig.extend_from_slice(buf);
            }
            let len = buf.len();
            let n = self.stemmer.stem(buf, len);
            // Java's `setLength(n)` over the edited buffer. Most terms come
            // back unchanged or only shortened, which the `String` can take
            // in place instead of being rebuilt from the units.
            let kept = n <= len
                && if ascii {
                    buf[..n]
                        .iter()
                        .zip(term.bytes())
                        .all(|(&u, b)| u == u16::from(b))
                } else {
                    buf[..n] == orig[..n]
                };
            if kept {
                if n < len {
                    if ascii {
                        a.term_mut().truncate(n);
                    } else {
                        truncate_utf16(a.term_mut(), n);
                    }
                }
            } else {
                a.set_term_utf16(&buf[..n]);
            }
        }
        Ok(true)
    }
}

/// Cuts `term` to its first `n` UTF-16 units, as `setLength(n)` does; a cut
/// inside a surrogate pair leaves the unpaired high surrogate, which the
/// attribute stores as U+FFFD (`set_term_utf16`).
fn truncate_utf16(term: &mut String, n: usize) {
    let mut seen = 0;
    for (at, c) in term.char_indices() {
        if seen == n {
            term.truncate(at);
            return;
        }
        seen += c.len_utf16();
        if seen > n {
            term.truncate(at);
            term.push(char::REPLACEMENT_CHARACTER);
            return;
        }
    }
}

/// Declares a language analyzer: a struct holding its stop set and stem
/// exclusion set (Java's `StopwordAnalyzerBase` subclasses with the
/// `(stopwords)` / `(stopwords, stemExclusionSet)` constructors), its
/// `Default` (the package's default stop set), and its `createComponents`
/// / `normalize` chains. `$s` binds the analyzer inside both bodies.
macro_rules! language_analyzer {
    (
        $(#[$meta:meta])*
        $name:ident, $default_stop:expr,
        components($s:ident) $components:block
        normalize($ns:ident, $input:ident) $normalize:block
        $(init_reader($reader:ident) $init:block)?
    ) => {
        $(#[$meta])*
        #[derive(Debug, Clone)]
        pub struct $name {
            stopwords: std::sync::Arc<crate::CharArraySet>,
            exclusion: std::sync::Arc<crate::CharArraySet>,
        }

        impl Default for $name {
            /// The no-argument constructor: the package's default stop set.
            fn default() -> Self {
                Self::new(&$default_stop)
            }
        }

        impl $name {
            /// `new X(CharArraySet stopwords)`.
            pub fn new(stopwords: &crate::CharArraySet) -> Self {
                Self::with_exclusions(stopwords, &crate::CharArraySet::empty())
            }

            /// `new X(stopwords, stemExclusionSet)`.
            pub fn with_exclusions(
                stopwords: &crate::CharArraySet,
                exclusions: &crate::CharArraySet,
            ) -> Self {
                $name {
                    stopwords: crate::lang::copy_set(stopwords),
                    exclusion: crate::lang::copy_set(exclusions),
                }
            }
        }

        impl crate::analyzer::AnalyzerDefinition for $name {
            fn create_components(
                &self,
                _field: &str,
            ) -> Result<crate::analyzer::TokenStreamComponents, crate::AnalysisError> {
                #[allow(unused_variables)]
                let $s = self;
                Ok(crate::analyzer::TokenStreamComponents::new($components))
            }

            fn normalize(
                &self,
                _field: &str,
                $input: Box<dyn crate::token_stream::TokenStream>,
            ) -> Box<dyn crate::token_stream::TokenStream> {
                #[allow(unused_variables)]
                let $ns = self;
                Box::new($normalize)
            }

            $(
                fn init_reader(
                    &self,
                    _field: &str,
                    $reader: Box<dyn crate::reader::CharReader>,
                ) -> Box<dyn crate::reader::CharReader> {
                    Box::new($init)
                }
            )?
        }
    };
}

pub mod ar;
pub mod bg;
pub mod bn;
pub mod br;
pub mod ca;
pub mod ckb;
pub mod cz;
pub mod da;
pub mod de;
pub mod el;
pub mod es;
pub mod et;
pub mod eu;
pub mod fa;
pub mod fi;
mod final_sigma;
pub mod fr;
pub mod ga;
pub mod gl;
pub mod hi;
pub mod hu;
pub mod hy;
pub mod id;
pub mod in_;
pub mod it;
pub mod lt;
pub mod lv;
pub mod ne;
pub mod nl;
pub mod no;
pub mod pt;
pub mod ro;
pub mod ru;
pub mod sr;
pub mod sv;
pub mod ta;
pub mod te;
pub mod tr;

/// `new StopFilter(new LowerCaseFilter(new StandardTokenizer()), stopwords)`.
pub(crate) fn std_lower_stop(
    stopwords: &Arc<CharArraySet>,
) -> StopFilter<LowerCaseFilter<StandardTokenizer>> {
    StopFilter::new(
        LowerCaseFilter::new(StandardTokenizer::new()),
        Arc::clone(stopwords),
    )
}

/// `if (!exclusion.isEmpty()) result = new SetKeywordMarkerFilter(result, exclusion)`.
pub(crate) fn mark_exclusions(
    input: impl TokenStream + 'static,
    exclusion: &Arc<CharArraySet>,
) -> Box<dyn TokenStream> {
    if exclusion.is_empty() {
        Box::new(input)
    } else {
        Box::new(SetKeywordMarkerFilter::new(input, Arc::clone(exclusion)))
    }
}

/// `new SnowballFilter(input, new <name>Stemmer())`.
pub(crate) fn snowball(
    input: impl TokenStream + 'static,
    name: &str,
) -> SnowballFilter<Box<dyn TokenStream>> {
    SnowballFilter::new(
        Box::new(input) as Box<dyn TokenStream>,
        SnowballStemmer::for_name(name).expect("a Snowball stemmer Lucene ships"),
    )
}

/// `WordlistLoader.getSnowballWordSet` of a vendored stop file.
pub(crate) fn snowball_set(text: &str) -> Arc<CharArraySet> {
    Arc::new(
        crate::wordlist_loader::get_snowball_word_set(text.as_bytes())
            .expect("a vendored stop file"),
    )
}

/// `WordlistLoader.getWordSet(reader)`: every trimmed non-empty line.
pub(crate) fn plain_set(text: &str) -> Arc<CharArraySet> {
    Arc::new(crate::wordlist_loader::get_word_set(text.as_bytes()).expect("a vendored stop file"))
}

/// `StopwordAnalyzerBase.loadStopwordSet(false, X.class, file, "#")`.
pub(crate) fn comment_set(text: &str) -> Arc<CharArraySet> {
    Arc::new(
        crate::wordlist_loader::get_word_set_with_comment(text.as_bytes(), "#")
            .expect("a vendored stop file"),
    )
}

/// `CharArraySet.copy` of a caller's set, as the analyzers' constructors do.
pub(crate) fn copy_set(set: &CharArraySet) -> Arc<CharArraySet> {
    let mut c = CharArraySet::with_capacity(set.len(), set.ignore_case());
    for w in set.iter() {
        c.add(w);
    }
    Arc::new(c)
}

/// `String.toLowerCase(Locale)` for a locale without tailorings (not `tr`,
/// `az`, `lt`): `Character.toLowerCase` per code point, plus Java's two
/// context-free/contextual special cases -- U+0130 becomes `i̇` (two units)
/// and a capital sigma ending a word becomes `ς`.
///
/// The final-sigma context is the JDK's ([`final_sigma`]: cased letters
/// within the `BreakIterator` word holding the sigma).
pub fn java_string_to_lower_case(units: &[u16]) -> Vec<u16> {
    let cps: Vec<u32> = char::decode_utf16(units.iter().copied())
        .map(|r| r.map_or_else(|e| u32::from(e.unpaired_surrogate()), u32::from))
        .collect();
    let mut sigma: Option<final_sigma::FinalSigma> = None;
    let mut out = Vec::with_capacity(units.len());
    for (i, &cp) in cps.iter().enumerate() {
        let lower = if cp == 0x130 {
            out.extend_from_slice(&[0x69, 0x307]);
            continue;
        } else if cp == 0x3A3 {
            let context = sigma.get_or_insert_with(|| final_sigma::FinalSigma::new(&cps));
            if context.is_final(i) {
                0x3C2
            } else {
                0x3C3
            }
        } else {
            to_lower_case(cp)
        };
        let mut b = [0u16; 2];
        match char::from_u32(lower) {
            Some(c) => out.extend_from_slice(c.encode_utf16(&mut b)),
            None => out.push(lower as u16),
        }
    }
    out
}

/// `String.toUpperCase(Locale)` for a locale without tailorings (not `tr`,
/// `az`, `lt`): `Character.toUpperCase` per code point, except the
/// characters whose full uppercase is several units
/// ([`crate::java_character::SPECIAL_UPPER`]: `ß` -> `SS`). An unpaired
/// surrogate is kept as is.
pub fn java_string_to_upper_case(units: &[u16]) -> Vec<u16> {
    let mut out = Vec::with_capacity(units.len());
    for r in char::decode_utf16(units.iter().copied()) {
        match r {
            Ok(c) => {
                let cp = u32::from(c);
                if let Some(m) = u16::try_from(cp)
                    .ok()
                    .and_then(crate::java_character::special_upper)
                {
                    out.extend_from_slice(m);
                    continue;
                }
                let up = crate::java_character::to_upper_case(cp);
                let mut b = [0u16; 2];
                match char::from_u32(up) {
                    Some(c) => out.extend_from_slice(c.encode_utf16(&mut b)),
                    None => out.push(up as u16),
                }
            }
            Err(e) => out.push(e.unpaired_surrogate()),
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::{render, Canned};

    #[test]
    fn string_to_upper_case_applies_special_casing() {
        let up = |s: &str| {
            String::from_utf16(&java_string_to_upper_case(
                &s.encode_utf16().collect::<Vec<_>>(),
            ))
            .unwrap()
        };
        assert_eq!(up("straße"), "STRASSE");
        assert_eq!(up("ŉx"), "ʼNX");
        assert_eq!(up("ǰ"), "J\u{30C}");
        assert_eq!(up("𐐨a"), "𐐀A");
        assert_eq!(java_string_to_upper_case(&[0xD800, 0x61]), [0xD800, 0x41]);
    }

    #[derive(Default)]
    struct DropLast;
    impl CharStemmer for DropLast {
        fn stem(&self, _s: &mut Vec<u16>, len: usize) -> usize {
            len.saturating_sub(1)
        }
    }

    #[test]
    fn stem_filters_respect_keywords_and_normalizers_do_not() {
        let mut c = Canned::parse("abc:0:3:1:1 xyz:4:7:1:1|7|0");
        c.set_keywords(&[true, false]);
        let mut f = StemFilter::<_, DropLast>::new(c);
        assert_eq!(render(&mut f), "abc:0:3:1:1 xy:4:7:1:1|7|0");
        let mut c = Canned::parse("abc:0:3:1:1 xyz:4:7:1:1|7|0");
        c.set_keywords(&[true, false]);
        let mut f = NormalizeFilter::with_stemmer(c, DropLast);
        assert_eq!(render(&mut f), "ab:0:3:1:1 xy:4:7:1:1|7|0");
    }

    #[derive(Default)]
    struct Edit;
    impl CharStemmer for Edit {
        fn stem(&self, s: &mut Vec<u16>, len: usize) -> usize {
            s[0] = u16::from(b'Z');
            len
        }
    }

    #[test]
    fn write_back_truncates_in_place_or_rebuilds() {
        let spec = "a\u{1F600}:0:3:1:1 é\u{1F600}x:4:8:1:1 αβ:9:11:1:1|11|0";
        let mut f = StemFilter::<_, DropLast>::new(Canned::parse(spec));
        assert_eq!(
            render(&mut f),
            "a\u{FFFD}:0:3:1:1 é\u{1F600}:4:8:1:1 α:9:11:1:1|11|0"
        );
        let mut f = NormalizeFilter::<_, Edit>::new(Canned::parse("ab:0:2:1:1 é:3:4:1:1|4|0"));
        assert_eq!(render(&mut f), "Zb:0:2:1:1 Z:3:4:1:1|4|0");
    }

    #[test]
    fn string_lowercase_special_cases() {
        let u = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        assert_eq!(java_string_to_lower_case(&u("AÄİ")), u("aäi\u{307}"));
        assert_eq!(java_string_to_lower_case(&u("ΟΔΟΣ Σ")), u("οδος σ"));
        assert_eq!(
            java_string_to_lower_case(&[0xD800, 0x41]),
            vec![0xD800, 0x61]
        );
        let s = comment_set("# c\nfoo\n bar \n");
        assert!(s.contains("foo") && s.contains("bar") && s.len() == 2);
        assert_eq!(copy_set(&s).len(), 2);
        assert!(snowball_set("a b | c\nd").contains("d"));
    }
}
