//! `UpperCaseFilter`, `DecimalDigitFilter` and `TypeTokenFilter`.

use std::collections::HashSet;

use crate::attributes::AttributeSource;
use crate::java_character;
use crate::token_stream::{Accept, FilteringTokenFilter, TokenFilter, TokenStream};
use crate::AnalysisError;

/// Maps every code point of the term in place, rebuilding the string only
/// from the first code point that changes.
pub(crate) fn map_term_chars(term: &mut String, f: impl Fn(char) -> char) {
    let Some(first) = term
        .char_indices()
        .find(|&(_, c)| f(c) != c)
        .map(|(i, _)| i)
    else {
        return;
    };
    let mut out = String::with_capacity(term.len());
    out.push_str(&term[..first]);
    out.extend(term[first..].chars().map(f));
    *term = out;
}

/// `CharacterUtils.toUpperCase` on one code point: Java's simple mapping. No
/// simple case mapping crosses the BMP boundary, so Java's in-place
/// `Character.toChars` write never changes the term's length.
fn upper(c: char) -> char {
    char::from_u32(java_character::to_upper_case(c as u32)).unwrap_or(c)
}

/// `org.apache.lucene.analysis.core.UpperCaseFilter`.
pub struct UpperCaseFilter<I> {
    input: I,
}

impl<I: TokenStream> UpperCaseFilter<I> {
    /// `new UpperCaseFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        UpperCaseFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for UpperCaseFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    // Java: UpperCaseFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.input.increment_token()? {
            map_term_chars(self.input.attributes_mut().term_mut(), upper);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// `DecimalDigitFilter`'s rewrite of one code point: a non-ASCII decimal
/// digit becomes its ASCII digit. Java writes the digit over the first unit
/// and deletes the second of a supplementary digit, which is this per-code-
/// point mapping.
fn fold_digit(c: char) -> char {
    let cp = c as u32;
    if cp > 0x7F {
        if let Some(v) = java_character::decimal_digit_value(cp) {
            return char::from(b'0' + v as u8);
        }
    }
    c
}

/// `org.apache.lucene.analysis.core.DecimalDigitFilter`: folds every Unicode
/// decimal digit (`Nd`) to `0`-`9`.
pub struct DecimalDigitFilter<I> {
    input: I,
}

impl<I: TokenStream> DecimalDigitFilter<I> {
    /// `new DecimalDigitFilter(TokenStream)`.
    pub fn new(input: I) -> Self {
        DecimalDigitFilter { input }
    }
}

impl<I: TokenStream> TokenFilter for DecimalDigitFilter<I> {
    type Input = I;
    fn input(&self) -> &I {
        &self.input
    }
    fn input_mut(&mut self) -> &mut I {
        &mut self.input
    }
    // Java: DecimalDigitFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        if self.input.increment_token()? {
            map_term_chars(self.input.attributes_mut().term_mut(), fold_digit);
            Ok(true)
        } else {
            Ok(false)
        }
    }
}

/// `TypeTokenFilter.accept()`: `useWhiteList == stopTypes.contains(type)`.
pub struct TypeAccept {
    stop_types: HashSet<String>,
    use_white_list: bool,
}

impl Accept for TypeAccept {
    fn accept(&mut self, attributes: &AttributeSource) -> Result<bool, AnalysisError> {
        Ok(self.use_white_list == self.stop_types.contains(attributes.token_type()))
    }
}

/// `org.apache.lucene.analysis.core.TypeTokenFilter`: drops (or, with
/// `useWhiteList`, keeps only) the tokens of the given types.
pub struct TypeTokenFilter<I> {
    inner: FilteringTokenFilter<I, TypeAccept>,
}

impl<I: TokenStream> TypeTokenFilter<I> {
    /// `new TypeTokenFilter(TokenStream, Set<String>, boolean useWhiteList)`.
    pub fn new<S: Into<String>>(
        input: I,
        stop_types: impl IntoIterator<Item = S>,
        use_white_list: bool,
    ) -> Self {
        TypeTokenFilter {
            inner: FilteringTokenFilter::new(
                input,
                TypeAccept {
                    stop_types: stop_types.into_iter().map(Into::into).collect(),
                    use_white_list,
                },
            ),
        }
    }
}

impl<I: TokenStream> TokenFilter for TypeTokenFilter<I> {
    type Input = FilteringTokenFilter<I, TypeAccept>;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::reader::StrReader;
    use crate::token_stream::{consume, Tokenizer};
    use crate::util::WhitespaceTokenizer;
    use crate::StandardTokenizer;

    fn ws(text: &str) -> WhitespaceTokenizer {
        let mut t = WhitespaceTokenizer::new();
        t.set_reader(Box::new(StrReader::new(text))).unwrap();
        t
    }

    fn terms(ts: &mut dyn TokenStream) -> Vec<(String, i32)> {
        let mut out = Vec::new();
        consume(ts, |a| {
            out.push((a.term().to_string(), a.position_increment()))
        })
        .unwrap();
        out
    }

    #[test]
    fn upper_case_uses_the_simple_mapping() {
        let mut f = UpperCaseFilter::new(ws("straße ﬁx abc ΟΔΟς 中"));
        let t: Vec<String> = terms(&mut f).into_iter().map(|t| t.0).collect();
        assert_eq!(t, vec!["STRAßE", "ﬁX", "ABC", "ΟΔΟΣ", "中"]);
    }

    #[test]
    fn decimal_digits_fold_to_ascii() {
        let mut f = DecimalDigitFilter::new(ws("٠١٢ x𝟎𝟗y 42 ½"));
        let t: Vec<String> = terms(&mut f).into_iter().map(|t| t.0).collect();
        assert_eq!(t, vec!["012", "x09y", "42", "½"]);
    }

    #[test]
    fn type_filter_drops_or_keeps_and_carries_increments() {
        let mut t = StandardTokenizer::new();
        t.set_reader(Box::new(StrReader::new("a 1 b 2 3"))).unwrap();
        let mut f = TypeTokenFilter::new(t, ["<NUM>"], false);
        assert_eq!(terms(&mut f), vec![("a".into(), 1), ("b".into(), 2)]);
        let mut t = StandardTokenizer::new();
        t.set_reader(Box::new(StrReader::new("a 1 b 2"))).unwrap();
        let mut f = TypeTokenFilter::new(t, ["<NUM>".to_string()], true);
        assert_eq!(terms(&mut f), vec![("1".into(), 2), ("2".into(), 2)]);
    }
}
