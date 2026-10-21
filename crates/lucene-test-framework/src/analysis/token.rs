//! `org.apache.lucene.tests.analysis.Token`: one token's attributes as a
//! value -- `PackedTokenAttributeImpl` (term, offsets, type, position
//! increment and length, term frequency) plus flags and payload.
//!
//! Differs: Java's `Token` is also an `AttributeFactory`
//! (`TOKEN_ATTRIBUTE_FACTORY`) whose instances back a stream's attributes;
//! the port's attributes are a fixed struct, so a `Token` is copied into
//! and out of an [`AttributeSource`] ([`Token::copy_to`],
//! [`Token::from_attributes`]).

use std::borrow::Cow;

use lucene_analysis::attributes::{AttrValue, AttributeSource, DEFAULT_TYPE};
use lucene_analysis::AnalysisError;

/// `org.apache.lucene.tests.analysis.Token`.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Token {
    term: String,
    start_offset: i32,
    end_offset: i32,
    token_type: Cow<'static, str>,
    position_increment: i32,
    position_length: i32,
    term_frequency: i32,
    flags: i32,
    payload: Option<Vec<u8>>,
}

impl Default for Token {
    /// `Token()`: Java's cleared attributes.
    fn default() -> Self {
        Token {
            term: String::new(),
            start_offset: 0,
            end_offset: 0,
            token_type: Cow::Borrowed(DEFAULT_TYPE),
            position_increment: 1,
            position_length: 1,
            term_frequency: 1,
            flags: 0,
            payload: None,
        }
    }
}

impl std::fmt::Display for Token {
    /// `toString()`: the term.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.term)
    }
}

impl Token {
    /// `Token(text, start, end)`.
    pub fn new(text: &str, start: i32, end: i32) -> Self {
        Token {
            term: text.to_string(),
            start_offset: start,
            end_offset: end,
            ..Token::default()
        }
    }

    /// `Token(text, posInc, start, end)`.
    pub fn with_pos_inc(text: &str, pos_inc: i32, start: i32, end: i32) -> Self {
        Token {
            position_increment: pos_inc,
            ..Token::new(text, start, end)
        }
    }

    /// `Token(text, posInc, start, end, posLength)`.
    pub fn with_pos_len(text: &str, pos_inc: i32, start: i32, end: i32, pos_len: i32) -> Self {
        Token {
            position_length: pos_len,
            ..Token::with_pos_inc(text, pos_inc, start, end)
        }
    }

    /// The token held by `atts` (the attributes a `Token` carries).
    pub fn from_attributes(atts: &AttributeSource) -> Self {
        Token {
            term: atts.term().to_string(),
            start_offset: atts.start_offset(),
            end_offset: atts.end_offset(),
            token_type: atts.token_type_cow().clone(),
            position_increment: atts.position_increment(),
            position_length: atts.position_length(),
            term_frequency: atts.term_frequency(),
            flags: atts.flags(),
            payload: atts.payload().map(<[u8]>::to_vec),
        }
    }

    /// `copyTo(target)`: these attributes into `atts`.
    ///
    /// # Errors
    /// What the attribute setters refuse (a negative offset or increment, a
    /// zero position length or term frequency): Java's
    /// `IllegalArgumentException` from the target.
    pub fn copy_to(&self, atts: &mut AttributeSource) -> Result<(), AnalysisError> {
        atts.set_term(&self.term);
        atts.set_offset(self.start_offset, self.end_offset)?;
        atts.set_token_type(self.token_type.clone());
        atts.set_position_increment(self.position_increment)?;
        atts.set_position_length(self.position_length)?;
        atts.set_term_frequency(self.term_frequency)?;
        atts.set_flags(self.flags);
        atts.set_payload(self.payload.clone());
        Ok(())
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        *self = Token::default();
    }

    /// The term (`toString()`).
    pub fn term(&self) -> &str {
        &self.term
    }

    /// `setEmpty().append(text)`.
    pub fn set_term(&mut self, text: &str) {
        self.term.clear();
        self.term.push_str(text);
    }

    /// `startOffset()`.
    pub fn start_offset(&self) -> i32 {
        self.start_offset
    }

    /// `endOffset()`.
    pub fn end_offset(&self) -> i32 {
        self.end_offset
    }

    /// `setOffset(start, end)`.
    ///
    /// # Errors
    /// Java's `IllegalArgumentException` for a negative start or an end
    /// before the start.
    pub fn set_offset(&mut self, start: i32, end: i32) -> Result<(), AnalysisError> {
        if start < 0 || end < start {
            return Err(AnalysisError::IllegalArgument(format!(
                "startOffset must be non-negative, and endOffset must be >= startOffset; got startOffset={start},endOffset={end}"
            )));
        }
        self.start_offset = start;
        self.end_offset = end;
        Ok(())
    }

    /// `type()`.
    pub fn token_type(&self) -> &str {
        &self.token_type
    }

    /// `setType(type)`.
    pub fn set_token_type(&mut self, t: impl Into<Cow<'static, str>>) {
        self.token_type = t.into();
    }

    /// `getPositionIncrement()`.
    pub fn position_increment(&self) -> i32 {
        self.position_increment
    }

    /// `setPositionIncrement(inc)` (Java validates it on the attribute it is
    /// copied into, [`Self::copy_to`]).
    pub fn set_position_increment(&mut self, inc: i32) {
        self.position_increment = inc;
    }

    /// `getPositionLength()`.
    pub fn position_length(&self) -> i32 {
        self.position_length
    }

    /// `setPositionLength(len)`.
    pub fn set_position_length(&mut self, len: i32) {
        self.position_length = len;
    }

    /// `getTermFrequency()`.
    pub fn term_frequency(&self) -> i32 {
        self.term_frequency
    }

    /// `setTermFrequency(freq)`.
    pub fn set_term_frequency(&mut self, freq: i32) {
        self.term_frequency = freq;
    }

    /// `getFlags()`.
    pub fn flags(&self) -> i32 {
        self.flags
    }

    /// `setFlags(flags)`.
    pub fn set_flags(&mut self, flags: i32) {
        self.flags = flags;
    }

    /// `getPayload()`.
    pub fn payload(&self) -> Option<&[u8]> {
        self.payload.as_deref()
    }

    /// `setPayload(payload)`.
    pub fn set_payload(&mut self, payload: Option<Vec<u8>>) {
        self.payload = payload;
    }

    /// `reflectWith(reflector)`: `PackedTokenAttributeImpl`'s attributes,
    /// then flags and payload, in Java's order.
    pub fn reflect_with(
        &self,
        reflector: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>),
    ) {
        reflector(
            "org.apache.lucene.analysis.tokenattributes.CharTermAttribute",
            "term",
            AttrValue::Str(Some(Cow::Borrowed(&self.term))),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.TermToBytesRefAttribute",
            "bytes",
            AttrValue::Bytes(Some(self.term.as_bytes())),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.OffsetAttribute",
            "startOffset",
            AttrValue::Int(self.start_offset),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.OffsetAttribute",
            "endOffset",
            AttrValue::Int(self.end_offset),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.PositionIncrementAttribute",
            "positionIncrement",
            AttrValue::Int(self.position_increment),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.PositionLengthAttribute",
            "positionLength",
            AttrValue::Int(self.position_length),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.TypeAttribute",
            "type",
            AttrValue::Str(Some(Cow::Borrowed(&self.token_type))),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.TermFrequencyAttribute",
            "termFrequency",
            AttrValue::Int(self.term_frequency),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.FlagsAttribute",
            "flags",
            AttrValue::Int(self.flags),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.PayloadAttribute",
            "payload",
            AttrValue::Bytes(self.payload.as_deref()),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn setters_getters_and_clear() {
        let mut t = Token::with_pos_len("x", 2, 1, 3, 4);
        assert_eq!(
            (
                t.position_increment(),
                t.position_length(),
                t.start_offset(),
                t.end_offset()
            ),
            (2, 4, 1, 3)
        );
        t.set_term("yy");
        t.set_token_type("SYN");
        t.set_position_increment(0);
        t.set_position_length(2);
        t.set_term_frequency(7);
        t.set_flags(5);
        t.set_payload(Some(vec![9]));
        assert!(t.set_offset(-1, 0).is_err());
        assert!(t.set_offset(3, 2).is_err());
        t.set_offset(2, 4).unwrap();
        let mut atts = AttributeSource::new();
        t.copy_to(&mut atts).unwrap();
        assert_eq!(Token::from_attributes(&atts), t);
        assert_eq!(t.token_type(), "SYN");
        assert_eq!(t.term_frequency(), 7);
        assert_eq!(t.flags(), 5);
        assert_eq!(t.payload(), Some(&[9u8][..]));
        assert_eq!(t.term(), "yy");
        t.clear();
        assert_eq!(t, Token::default());
        // the target refuses what Java's attributes refuse
        let mut bad = Token::new("z", 0, 0);
        bad.set_position_length(0);
        assert!(bad.copy_to(&mut atts).is_err());
    }
}
