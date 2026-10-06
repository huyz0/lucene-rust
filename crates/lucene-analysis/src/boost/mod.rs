//! `org.apache.lucene.analysis.boost.DelimitedBoostTokenFilter`.

use crate::payloads::parse_java_float;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `org.apache.lucene.analysis.boost.DelimitedBoostTokenFilter`:
/// `term|boost` -> `term` with `BoostAttribute` set to `boost`.
pub struct DelimitedBoostTokenFilter<I> {
    input: I,
    delimiter: char,
}

impl<I: TokenStream> DelimitedBoostTokenFilter<I> {
    /// `new DelimitedBoostTokenFilter(TokenStream, char delimiter)`.
    pub fn new(input: I, delimiter: char) -> Self {
        DelimitedBoostTokenFilter { input, delimiter }
    }
}

impl<I: TokenStream> TokenFilter for DelimitedBoostTokenFilter<I> {
    crate::filter_input!();
    // Java: DelimitedBoostTokenFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if !self.input.increment_token()? {
            return Ok(false);
        }
        let a = self.input.attributes_mut();
        if let Some(i) = a.term().find(self.delimiter) {
            let boost = parse_java_float(&a.term()[i + self.delimiter.len_utf8()..])?;
            a.set_boost(boost);
            a.term_mut().truncate(i);
        }
        Ok(true)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::Canned;

    #[test]
    fn boosts() {
        let mut c = Canned::parse("a:0:1:1:1 b:0:1:1:1 c:0:1:1:1");
        c.set_terms(&["x|2.5", "y", "z|q"]);
        let mut f = DelimitedBoostTokenFilter::new(c, '|');
        f.reset().unwrap();
        let mut out = Vec::new();
        while let Ok(true) = f.increment_token() {
            out.push((f.attributes().term().to_string(), f.attributes().boost()));
        }
        assert_eq!(out, vec![("x".into(), 2.5), ("y".into(), 1.0)]);
    }
}
