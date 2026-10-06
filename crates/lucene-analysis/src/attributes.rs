//! The token attribute model: Java's `org.apache.lucene.util.AttributeSource`
//! and every attribute of `org.apache.lucene.analysis.tokenattributes`.
//!
//! # Shape: one fixed struct, not a type-keyed registry
//!
//! Java's `AttributeSource` is a map from attribute *interface* to an
//! `AttributeImpl`, populated on demand by `addAttribute(Class)` through an
//! `AttributeFactory`; the default factory packs the six hottest attributes
//! into one `PackedTokenAttributeImpl` and creates the rest one object each.
//! Every stage of a chain shares that one source, so a filter "adds" an
//! attribute by asking the shared map for it.
//!
//! The port models the **core** attribute set -- the eleven attributes of
//! `tokenattributes` -- as the plain fields of [`AttributeSource`], always
//! present. That is a deliberate change, forced by nothing but chosen for
//! three reasons, each checked against the Java:
//!
//! 1. **Absence is never observable in core.** Every core attribute's
//!    `clear()` value is exactly what a consumer that did not find the
//!    attribute assumes: `IndexingChain` treats a missing `PayloadAttribute`
//!    as a null payload and a missing `TermFrequencyAttribute` as frequency 1;
//!    `GraphTokenFilter` treats a missing `PositionLengthAttribute` as length
//!    1; a missing `TypeAttribute` reads as `"word"`. So "present with its
//!    cleared value" and "absent" produce the same index and the same tokens.
//! 2. **The hot path is a field access**, not a hash lookup or a virtual call
//!    per attribute per token -- the job `PackedTokenAttributeImpl` exists to
//!    do in Java, done here for all eleven.
//! 3. **`captureState`/`restoreState`/`cloneAttributes`/`copyTo` become
//!    `Clone`/`clone_from`** of one value ([`State`]).
//!
//! What it gives up: a *custom* attribute type (a class outside
//! `tokenattributes`, e.g. analysis-common's `ScriptAttribute`) has no slot.
//! No core class defines one; a later module that needs one extends this
//! struct.
//!
//! `CharTermAttribute`'s `char[]` buffer is a Rust `String`. Java terms are
//! UTF-16 and may hold an unpaired surrogate; a Rust `String` cannot, so the
//! one place a term is built from UTF-16 code units
//! ([`AttributeSource::set_term_utf16`]) replaces an unpaired surrogate with
//! U+FFFD -- which is also what Java's `getBytesRef()` (the only way a term
//! reaches an index) turns it into, so the indexed bytes agree.
//! [`AttributeSource::term_utf16_len`] is Java's `CharTermAttribute.length()`.

use std::borrow::Cow;

use crate::AnalysisError;

/// `TypeAttribute.DEFAULT_TYPE`.
pub const DEFAULT_TYPE: &str = "word";

/// Java's `AttributeSource` for a token stream, with every core attribute as a
/// field (see the module docs for why a fixed struct).
///
/// Setters that Java validates (`setPositionIncrement`, `setPositionLength`,
/// `setOffset`, `setTermFrequency`) return `Err(IllegalArgument)` where Java
/// throws `IllegalArgumentException`, with Java's message.
#[derive(Debug, PartialEq, Eq)]
pub struct AttributeSource {
    /// `CharTermAttribute`.
    term: String,
    /// `BytesTermAttribute`: a binary term. When set, it is what
    /// [`Self::term_bytes`] (`TermToBytesRefAttribute.getBytesRef()`) reports
    /// instead of the UTF-8 of [`Self::term`].
    bytes_term: Option<Vec<u8>>,
    /// `OffsetAttribute`.
    start_offset: i32,
    end_offset: i32,
    /// `PositionIncrementAttribute`.
    position_increment: i32,
    /// `PositionLengthAttribute`.
    position_length: i32,
    /// `TypeAttribute`.
    token_type: Cow<'static, str>,
    /// `FlagsAttribute`.
    flags: i32,
    /// `KeywordAttribute`.
    keyword: bool,
    /// `PayloadAttribute`.
    payload: Option<Vec<u8>>,
    /// `TermFrequencyAttribute`.
    term_frequency: i32,
    /// `SentenceAttribute`.
    sentence_index: i32,
    /// `search.BoostAttribute`, as `f32` bits (so the struct stays `Eq`).
    boost_bits: u32,
}

/// `AttributeSource.State`: a captured copy of every attribute, restored with
/// [`AttributeSource::restore_state`].
pub type State = AttributeSource;

impl Clone for AttributeSource {
    fn clone(&self) -> Self {
        let mut a = AttributeSource::new();
        a.clone_from(self);
        a
    }

    /// Field by field, so [`AttributeSource::restore_state`] reuses this
    /// source's term and payload buffers (a derived `clone_from` would
    /// allocate a whole new value per call). The destructuring names every
    /// field, so a field added to the struct fails to compile here until it
    /// is copied too.
    fn clone_from(&mut self, source: &Self) {
        let AttributeSource {
            term,
            bytes_term,
            start_offset,
            end_offset,
            position_increment,
            position_length,
            token_type,
            flags,
            keyword,
            payload,
            term_frequency,
            sentence_index,
            boost_bits,
        } = source;
        self.term.clone_from(term);
        if self.bytes_term.is_some() || bytes_term.is_some() {
            self.bytes_term.clone_from(bytes_term);
        }
        self.start_offset = *start_offset;
        self.end_offset = *end_offset;
        self.position_increment = *position_increment;
        self.position_length = *position_length;
        self.token_type.clone_from(token_type);
        self.flags = *flags;
        self.keyword = *keyword;
        if self.payload.is_some() || payload.is_some() {
            self.payload.clone_from(payload);
        }
        self.term_frequency = *term_frequency;
        self.sentence_index = *sentence_index;
        self.boost_bits = *boost_bits;
    }
}

impl Default for AttributeSource {
    fn default() -> Self {
        Self::new()
    }
}

impl AttributeSource {
    /// A source with every attribute at its `clear()` value.
    pub fn new() -> Self {
        AttributeSource {
            term: String::new(),
            bytes_term: None,
            start_offset: 0,
            end_offset: 0,
            position_increment: 1,
            position_length: 1,
            token_type: Cow::Borrowed(DEFAULT_TYPE),
            flags: 0,
            keyword: false,
            payload: None,
            term_frequency: 1,
            sentence_index: 0,
            boost_bits: 1.0f32.to_bits(),
        }
    }

    // ------------------------------------------------------------ lifecycle

    /// `AttributeSource.clearAttributes()`: every attribute's `clear()`.
    /// Keeps the term's allocation, as Java keeps its `char[]`.
    pub fn clear_attributes(&mut self) {
        self.term.clear();
        self.bytes_term = None;
        self.start_offset = 0;
        self.end_offset = 0;
        self.position_increment = 1;
        self.position_length = 1;
        self.token_type = Cow::Borrowed(DEFAULT_TYPE);
        self.flags = 0;
        self.keyword = false;
        self.payload = None;
        self.term_frequency = 1;
        self.sentence_index = 0;
        self.boost_bits = 1.0f32.to_bits();
    }

    /// `AttributeSource.endAttributes()`: every attribute's `end()`, which is
    /// its `clear()` except `PositionIncrementAttributeImpl.end()` (increment
    /// 0) and `TermFrequencyAttributeImpl.end()` (frequency 1, as clear).
    pub fn end_attributes(&mut self) {
        self.clear_attributes();
        self.position_increment = 0;
    }

    /// `AttributeSource.captureState()`.
    pub fn capture_state(&self) -> State {
        self.clone()
    }

    /// `AttributeSource.restoreState(State)`; reuses this source's buffers.
    pub fn restore_state(&mut self, state: &State) {
        self.clone_from(state);
    }

    // ------------------------------------------------------ CharTermAttribute

    /// `CharTermAttribute.toString()`.
    pub fn term(&self) -> &str {
        &self.term
    }

    /// The term buffer, for a filter that rewrites it in place (Java's
    /// `buffer()` + `setLength()`).
    pub fn term_mut(&mut self) -> &mut String {
        &mut self.term
    }

    /// `CharTermAttribute.setEmpty().append(s)`.
    pub fn set_term(&mut self, s: &str) {
        self.term.clear();
        self.term.push_str(s);
    }

    /// `CharTermAttribute.copyBuffer(char[], off, len)` from UTF-16 code
    /// units; an unpaired surrogate becomes U+FFFD (see the module docs).
    pub fn set_term_utf16(&mut self, units: &[u16]) {
        self.term.clear();
        // The common case, an ASCII term: each unit is its own byte.
        if units.iter().all(|&u| u < 0x80) {
            self.term.reserve(units.len());
            self.term.extend(units.iter().map(|&u| char::from(u as u8)));
            return;
        }
        // At most 3 UTF-8 bytes per UTF-16 unit (a pair is 4 bytes for 2).
        self.term.reserve(units.len() * 3);
        let mut i = 0;
        while i < units.len() {
            let u = units[i];
            i += 1;
            let c = if u < 0x80 {
                u as u8 as char
            } else if !(0xD800..=0xDFFF).contains(&u) {
                char::from_u32(u32::from(u)).unwrap_or(char::REPLACEMENT_CHARACTER)
            } else if u <= 0xDBFF && i < units.len() && (0xDC00..=0xDFFF).contains(&units[i]) {
                let lo = units[i];
                i += 1;
                let cp = 0x10000 + ((u32::from(u) - 0xD800) << 10) + (u32::from(lo) - 0xDC00);
                char::from_u32(cp).unwrap_or(char::REPLACEMENT_CHARACTER)
            } else {
                char::REPLACEMENT_CHARACTER
            };
            self.term.push(c);
        }
    }

    /// `CharTermAttribute.length()`: the term's length in UTF-16 code units.
    pub fn term_utf16_len(&self) -> usize {
        crate::utf16_len(&self.term)
    }

    // ------------------------------------- TermToBytesRef / BytesTermAttribute

    /// `TermToBytesRefAttribute.getBytesRef()`: the binary term when one is
    /// set (`BytesTermAttributeImpl`), else the term's UTF-8
    /// (`CharTermAttributeImpl`).
    pub fn term_bytes(&self) -> &[u8] {
        match &self.bytes_term {
            Some(b) => b,
            None => self.term.as_bytes(),
        }
    }

    /// `BytesTermAttribute.setBytesRef`; `None` clears it.
    pub fn set_bytes_term(&mut self, bytes: Option<Vec<u8>>) {
        self.bytes_term = bytes;
    }

    /// `BytesTermAttributeImpl.getBytesRef()` itself.
    pub fn bytes_term(&self) -> Option<&[u8]> {
        self.bytes_term.as_deref()
    }

    // --------------------------------------------------------- OffsetAttribute

    /// `OffsetAttribute.startOffset()`, in UTF-16 code units.
    pub fn start_offset(&self) -> i32 {
        self.start_offset
    }

    /// `OffsetAttribute.endOffset()`, in UTF-16 code units.
    pub fn end_offset(&self) -> i32 {
        self.end_offset
    }

    /// `OffsetAttribute.setOffset`, with Java's validation.
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

    // ---------------------------------------------- PositionIncrementAttribute

    /// `PositionIncrementAttribute.getPositionIncrement()`.
    pub fn position_increment(&self) -> i32 {
        self.position_increment
    }

    /// `PositionIncrementAttribute.setPositionIncrement`, with Java's
    /// validation.
    pub fn set_position_increment(&mut self, inc: i32) -> Result<(), AnalysisError> {
        if inc < 0 {
            return Err(AnalysisError::IllegalArgument(format!(
                "Position increment must be zero or greater; got {inc}"
            )));
        }
        self.position_increment = inc;
        Ok(())
    }

    // ------------------------------------------------- PositionLengthAttribute

    /// `PositionLengthAttribute.getPositionLength()`.
    pub fn position_length(&self) -> i32 {
        self.position_length
    }

    /// `PositionLengthAttribute.setPositionLength`, with Java's validation.
    pub fn set_position_length(&mut self, len: i32) -> Result<(), AnalysisError> {
        if len < 1 {
            return Err(AnalysisError::IllegalArgument(format!(
                "Position length must be 1 or greater; got {len}"
            )));
        }
        self.position_length = len;
        Ok(())
    }

    // ----------------------------------------------------------- TypeAttribute

    /// `TypeAttribute.type()`.
    pub fn token_type(&self) -> &str {
        &self.token_type
    }

    /// `TypeAttribute.setType`.
    pub fn set_token_type(&mut self, t: impl Into<Cow<'static, str>>) {
        self.token_type = t.into();
    }

    // ---------------------------------------------------------- FlagsAttribute

    /// `FlagsAttribute.getFlags()`.
    pub fn flags(&self) -> i32 {
        self.flags
    }

    /// `FlagsAttribute.setFlags`.
    pub fn set_flags(&mut self, flags: i32) {
        self.flags = flags;
    }

    // -------------------------------------------------------- KeywordAttribute

    /// `KeywordAttribute.isKeyword()`.
    pub fn is_keyword(&self) -> bool {
        self.keyword
    }

    /// `KeywordAttribute.setKeyword`.
    pub fn set_keyword(&mut self, keyword: bool) {
        self.keyword = keyword;
    }

    // -------------------------------------------------------- PayloadAttribute

    /// `PayloadAttribute.getPayload()`.
    pub fn payload(&self) -> Option<&[u8]> {
        self.payload.as_deref()
    }

    /// `PayloadAttribute.setPayload`.
    pub fn set_payload(&mut self, payload: Option<Vec<u8>>) {
        self.payload = payload;
    }

    // -------------------------------------------------- TermFrequencyAttribute

    /// `TermFrequencyAttribute.getTermFrequency()`.
    pub fn term_frequency(&self) -> i32 {
        self.term_frequency
    }

    /// `TermFrequencyAttribute.setTermFrequency`, with Java's validation.
    pub fn set_term_frequency(&mut self, freq: i32) -> Result<(), AnalysisError> {
        if freq < 1 {
            return Err(AnalysisError::IllegalArgument(format!(
                "Term frequency must be 1 or greater; got {freq}"
            )));
        }
        self.term_frequency = freq;
        Ok(())
    }

    // ------------------------------------------------------- SentenceAttribute

    /// `SentenceAttribute.getSentenceIndex()`.
    pub fn sentence_index(&self) -> i32 {
        self.sentence_index
    }

    /// `SentenceAttribute.setSentenceIndex`.
    pub fn set_sentence_index(&mut self, index: i32) {
        self.sentence_index = index;
    }

    /// `BoostAttribute.getBoost()` (`org.apache.lucene.search`, which
    /// `DelimitedBoostTokenFilter` sets); `1.0` when cleared.
    pub fn boost(&self) -> f32 {
        f32::from_bits(self.boost_bits)
    }

    /// `BoostAttribute.setBoost(float)`.
    pub fn set_boost(&mut self, boost: f32) {
        self.boost_bits = boost.to_bits();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dirty() -> AttributeSource {
        let mut a = AttributeSource::new();
        a.set_term("Foo");
        a.set_bytes_term(Some(vec![1, 2]));
        a.set_offset(3, 7).unwrap();
        a.set_position_increment(4).unwrap();
        a.set_position_length(2).unwrap();
        a.set_token_type("<NUM>");
        a.set_flags(9);
        a.set_keyword(true);
        a.set_payload(Some(vec![5]));
        a.set_term_frequency(6).unwrap();
        a.set_sentence_index(8);
        a
    }

    #[test]
    fn clear_restores_every_java_clear_value() {
        let mut a = dirty();
        a.clear_attributes();
        assert_eq!(a, AttributeSource::new());
        assert_eq!(a.token_type(), "word");
        assert_eq!(a.position_increment(), 1);
        assert_eq!(a.term_frequency(), 1);
    }

    #[test]
    fn end_is_clear_with_a_zero_increment() {
        let mut a = dirty();
        a.end_attributes();
        let mut want = AttributeSource::new();
        want.position_increment = 0;
        assert_eq!(a, want);
    }

    #[test]
    fn capture_and_restore_round_trip() {
        let a = dirty();
        let s = a.capture_state();
        let mut b = AttributeSource::new();
        b.restore_state(&s);
        assert_eq!(a, b);
        assert_eq!(b.payload(), Some(&[5u8][..]));
        assert_eq!(b.bytes_term(), Some(&[1u8, 2][..]));
        assert_eq!(b.flags(), 9);
        assert!(b.is_keyword());
        assert_eq!(b.sentence_index(), 8);
        assert_eq!(b.position_length(), 2);
        assert_eq!((b.start_offset(), b.end_offset()), (3, 7));
    }

    #[test]
    fn term_bytes_prefers_the_binary_term() {
        let mut a = AttributeSource::new();
        a.set_term("é");
        assert_eq!(a.term_bytes(), "é".as_bytes());
        a.set_bytes_term(Some(vec![0xff]));
        assert_eq!(a.term_bytes(), &[0xff]);
        a.set_bytes_term(None);
        assert_eq!(a.term_bytes(), "é".as_bytes());
    }

    #[test]
    fn setters_validate_like_java() {
        let mut a = AttributeSource::new();
        assert!(a.set_offset(-1, 0).is_err());
        assert!(a.set_offset(5, 4).is_err());
        assert!(a.set_position_increment(-1).is_err());
        assert!(a.set_position_length(0).is_err());
        assert!(a.set_term_frequency(0).is_err());
        assert!(a.set_position_increment(0).is_ok());
        let msg = a.set_position_length(0).unwrap_err().to_string();
        assert!(
            msg.contains("Position length must be 1 or greater; got 0"),
            "{msg}"
        );
    }

    #[test]
    fn utf16_terms_decode_and_replace_lone_surrogates() {
        let mut a = AttributeSource::new();
        a.set_term_utf16(&[b'a' as u16, b'b' as u16]);
        assert_eq!(a.term(), "ab");
        a.set_term_utf16(&[0xD83D, 0xDE00, 0x00E9]);
        assert_eq!(a.term(), "😀é");
        assert_eq!(a.term_utf16_len(), 3);
        a.set_term_utf16(&[0xD800, b'x' as u16]);
        assert_eq!(a.term(), "\u{FFFD}x");
        a.term_mut().push('y');
        assert_eq!(a.term(), "\u{FFFD}xy");
    }
}
