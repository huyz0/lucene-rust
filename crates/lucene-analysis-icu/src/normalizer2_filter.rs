//! `org.apache.lucene.analysis.icu.ICUNormalizer2Filter`: normalizes each
//! term with a [`Normalizer2`] (NFKC_Casefold by default), leaving a term
//! the normalizer's quick check calls normalized untouched.

use lucene_analysis::{AnalysisError, TokenFilter, TokenStream};

use crate::icu4j::normalizer2::Normalizer2;

/// `ICUNormalizer2Filter`.
pub struct ICUNormalizer2Filter<I = Box<dyn TokenStream>> {
    input: I,
    normalizer: Normalizer2,
    units: Vec<u16>,
    buffer: Vec<u16>,
    scratch: String,
}

impl<I: TokenStream> ICUNormalizer2Filter<I> {
    /// `ICUNormalizer2Filter(TokenStream)`: NFKC_Casefold
    /// (`Normalizer2.getInstance(null, "nfkc_cf", COMPOSE)`).
    pub fn new(input: I) -> Self {
        Self::with_normalizer(input, Normalizer2::nfkc_casefold())
    }

    /// `ICUNormalizer2Filter(TokenStream, Normalizer2)`.
    pub fn with_normalizer(input: I, normalizer: Normalizer2) -> Self {
        ICUNormalizer2Filter {
            input,
            normalizer,
            units: Vec::new(),
            buffer: Vec::new(),
            scratch: String::new(),
        }
    }
}

impl<I: TokenStream> TokenFilter for ICUNormalizer2Filter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }

    // Java: ICUNormalizer2Filter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let atts = self.input.attributes_mut();
        // Two shortcuts the UTF-16 path would agree with (see
        // `Normalizer2::quick_yes_utf8` and `ascii_map`): a term the quick
        // check passes unit by unit, and a term of ASCII alone.
        if self.normalizer.quick_yes_utf8(atts.term()) {
            return Ok(true);
        }
        if atts.term().is_ascii() {
            if let Some(map) = self.normalizer.ascii_map() {
                let term = atts.term_mut();
                if term.bytes().any(|b| map[usize::from(b)] != b) {
                    self.scratch.clear();
                    self.scratch
                        .extend(term.bytes().map(|b| char::from(map[usize::from(b)])));
                    std::mem::swap(term, &mut self.scratch);
                }
                return Ok(true);
            }
        }
        self.units.clear();
        self.units.extend(atts.term().encode_utf16());
        // Java runs `quickCheck` and, unless it is `YES`, `normalize`. A
        // string the quick check passes normalizes to itself, so one
        // `normalize` (which spans the quick-check-yes prefix itself) and a
        // comparison give the same term in one pass instead of two.
        self.normalizer.normalize_to(&self.units, &mut self.buffer);
        if self.buffer != self.units {
            atts.set_term_utf16(&self.buffer);
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::{KeywordTokenizer, StrReader, Tokenizer};

    fn run(text: &str, f: impl Fn(KeywordTokenizer) -> Box<dyn TokenStream>) -> String {
        let mut t = KeywordTokenizer::new();
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        let mut s = f(t);
        s.reset().unwrap();
        assert!(s.increment_token().unwrap());
        let term = s.attributes().term().to_string();
        assert!(!s.increment_token().unwrap());
        s.end().unwrap();
        s.close().unwrap();
        term
    }

    #[test]
    fn normalizes() {
        assert_eq!(
            run("Ｈｅｌｌｏ ß", |t| Box::new(
                ICUNormalizer2Filter::new(t)
            )),
            "hello ss"
        );
        assert_eq!(
            run("plain", |t| Box::new(ICUNormalizer2Filter::new(t))),
            "plain"
        );
        let nfd = Normalizer2::get_instance("nfc", crate::Mode::Decompose).unwrap();
        assert_eq!(
            run("é", |t| Box::new(ICUNormalizer2Filter::with_normalizer(
                t,
                nfd.clone()
            ))),
            "e\u{301}"
        );
    }
}
