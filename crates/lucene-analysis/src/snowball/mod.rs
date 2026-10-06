//! `org.apache.lucene.analysis.snowball.SnowballFilter` and the Snowball
//! stemmers it runs (`org.tartarus.snowball`).
//!
//! The 30 stemmers of `org.tartarus.snowball.ext` are generated, not ported:
//! [`algorithms`] is the Snowball compiler's Rust backend run over the `.sbl`
//! sources at the Snowball commit Lucene 10.5.0's Java stemmers were generated
//! from (`tools/gen_snowball.sh`, which also checks that the same compiler's
//! Java backend reproduces Lucene's classes). [`program`] is the runtime they
//! call, with `SnowballProgram`'s semantics.
//!
//! Java's `SnowballStemmer` subclass per language becomes one
//! [`SnowballStemmer`] holding the language's generated `stem` function;
//! `SnowballFilter(input, name)`'s reflective `Class.forName` is a lookup in
//! [`algorithms::STEMMERS`].
//!
//! The program works on Java's UTF-16 units ([`program`]), so a term (a UTF-8
//! `String`) is encoded into the stemmer's buffer and the stem decoded back:
//! the stems are Java's unit for unit, an unpaired surrogate a stemmer leaves
//! becoming U+FFFD as everywhere in this crate.

#[rustfmt::skip]
pub mod algorithms;
pub mod program;

use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;
use program::SnowballEnv;

/// `org.tartarus.snowball.SnowballStemmer`: one language's generated
/// stemmer over its own [`SnowballEnv`].
#[derive(Debug, Clone)]
pub struct SnowballStemmer {
    name: &'static str,
    stem: fn(&mut SnowballEnv) -> bool,
    env: SnowballEnv,
}

impl SnowballStemmer {
    /// `new org.tartarus.snowball.ext.<name>Stemmer()`: `name` is the class
    /// name without `Stemmer` (`"English"`, `"Porter"`, ...), or `None` when
    /// Lucene ships no such class. Names are case-sensitive, as class names.
    pub fn for_name(name: &str) -> Option<Self> {
        algorithms::STEMMERS
            .iter()
            .find(|(n, _)| *n == name)
            .map(|&(name, stem)| SnowballStemmer {
                name,
                stem,
                env: SnowballEnv::default(),
            })
    }

    /// Every stemmer name Lucene 10.5.0 ships, in `languages.txt` order.
    pub fn names() -> impl Iterator<Item = &'static str> {
        algorithms::STEMMERS.iter().map(|(n, _)| *n)
    }

    /// The class name without `Stemmer`.
    pub fn name(&self) -> &'static str {
        self.name
    }

    /// Java: `SnowballProgram.setCurrent(String)` (a string beyond Java's
    /// `int` length is refused, `false`).
    pub fn set_current(&mut self, value: &str) -> bool {
        self.env.set_current(value)
    }

    /// Java: `SnowballStemmer.stem()`: stems the current string; whether the
    /// program's top routine succeeded (the result is the current string
    /// either way).
    pub fn stem(&mut self) -> bool {
        (self.stem)(&mut self.env)
    }

    /// Java: `SnowballProgram.getCurrent()`.
    pub fn current(&self) -> String {
        let mut s = String::new();
        self.env.get_current_into(&mut s);
        s
    }

    /// `setCurrent(termBuffer, length); stem(); getCurrentBuffer()` on a term:
    /// the term is replaced by its stem (left as it is when refused).
    pub fn stem_in_place(&mut self, term: &mut String) {
        if self.env.set_current(term) {
            (self.stem)(&mut self.env);
            if self.env.edited() {
                self.env.get_current_into(term);
            }
        }
    }
}

/// `org.apache.lucene.analysis.snowball.SnowballFilter`: stems every
/// non-keyword term with a [`SnowballStemmer`].
pub struct SnowballFilter<I> {
    input: I,
    stemmer: SnowballStemmer,
}

impl<I: TokenStream> SnowballFilter<I> {
    /// `new SnowballFilter(TokenStream, SnowballStemmer)`.
    pub fn new(input: I, stemmer: SnowballStemmer) -> Self {
        SnowballFilter { input, stemmer }
    }

    /// `new SnowballFilter(TokenStream, String)`: the stemmer named `name`
    /// (`"German2"` is `"German"`, as Java keeps it); an unknown name is
    /// Java's `IllegalArgumentException`.
    pub fn with_name(input: I, name: &str) -> Result<Self, AnalysisError> {
        let name = if name == "German2" { "German" } else { name };
        match SnowballStemmer::for_name(name) {
            Some(stemmer) => Ok(Self::new(input, stemmer)),
            None => Err(AnalysisError::IllegalArgument(format!(
                "Invalid stemmer class specified: {name}"
            ))),
        }
    }
}

impl<I: TokenStream> TokenFilter for SnowballFilter<I> {
    crate::filter_input!();
    // Java: SnowballFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if !a.is_keyword() {
            self.stemmer.stem_in_place(a.term_mut());
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::token_stream::Tokenizer;
    use crate::KeywordTokenizer;

    fn run(mut ts: impl TokenStream) -> Vec<String> {
        let mut out = Vec::new();
        ts.reset().unwrap();
        while ts.increment_token().unwrap() {
            out.push(ts.attributes().term().to_string());
        }
        ts.end().unwrap();
        out
    }

    fn keyword(text: &str) -> KeywordTokenizer {
        let mut t = KeywordTokenizer::new();
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        t
    }

    #[test]
    fn a_name_is_a_shipped_class_name_and_german2_is_german() {
        assert_eq!(SnowballStemmer::names().count(), 30);
        assert!(SnowballStemmer::for_name("english").is_none());
        let err = SnowballFilter::with_name(keyword("x"), "Klingon")
            .err()
            .unwrap();
        assert_eq!(
            err.to_string(),
            "illegal argument: Invalid stemmer class specified: Klingon"
        );
        let f = SnowballFilter::with_name(keyword("häuser"), "German2").unwrap();
        assert_eq!(f.stemmer.name(), "German");
        assert_eq!(run(f), ["haus"]);
    }

    #[test]
    fn a_keyword_term_is_left_alone() {
        let marked = crate::miscellaneous::KeywordRepeatFilter::new(keyword("running"));
        let f = SnowballFilter::new(marked, SnowballStemmer::for_name("English").unwrap());
        // KeywordRepeatFilter emits the keyword-marked original, then the copy.
        assert_eq!(run(f), ["running", "run"]);
    }

    #[test]
    fn the_direct_api_stems_the_current_string() {
        let mut s = SnowballStemmer::for_name("Porter").unwrap();
        assert!(s.set_current("generalizations"));
        assert!(s.stem());
        assert_eq!(s.current(), "gener");
        let mut term = String::from("caresses");
        s.stem_in_place(&mut term);
        assert_eq!(term, "caress");
    }
}
