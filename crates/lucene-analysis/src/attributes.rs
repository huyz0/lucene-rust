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
//! A *custom* attribute (a class outside `tokenattributes`: Morfologik's
//! `MorphosyntacticTagsAttribute`, Kuromoji's readings, ICU's
//! `ScriptAttribute`) is a [`CustomAttribute`] type, added on demand with
//! [`AttributeSource::add_custom`] as Java's `addAttribute` adds one, and
//! held in a list beside the fixed fields: cleared, ended, captured and
//! restored with them. A chain that adds none carries an empty list.
//!
//! `CharTermAttribute`'s `char[]` buffer is a Rust `String`. Java terms are
//! UTF-16 and may hold an unpaired surrogate; a Rust `String` cannot, so the
//! one place a term is built from UTF-16 code units
//! ([`AttributeSource::set_term_utf16`]) replaces an unpaired surrogate with
//! U+FFFD -- which is also what Java's `getBytesRef()` (the only way a term
//! reaches an index) turns it into, so the indexed bytes agree.
//! [`AttributeSource::term_utf16_len`] is Java's `CharTermAttribute.length()`.

use std::any::{Any, TypeId};
use std::borrow::Cow;
use std::fmt::Debug;

use crate::AnalysisError;

/// An attribute outside the core set: Java's `Attribute` interface and its
/// `AttributeImpl` in one type. `Default` is the value `addAttribute`
/// creates; [`Self::clear`] is `AttributeImpl.clear()`, [`Self::end`]
/// `AttributeImpl.end()` (`clear()` unless overridden); `Clone` is
/// `copyTo`/`captureState`.
pub trait CustomAttribute: Clone + Eq + Debug + Default + Send + Sync + 'static {
    /// `AttributeImpl.clear()`.
    fn clear(&mut self);

    /// `AttributeImpl.end()`.
    fn end(&mut self) {
        self.clear();
    }

    /// `AttributeImpl.reflectWith(AttributeReflector)`: each of the
    /// attribute's key/value pairs, with the fully qualified name of the
    /// attribute interface it belongs to, in Java's order.
    fn reflect(&self, reflector: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>));

    /// The implementation's class name, for `restoreState`'s error (Java's
    /// `getClass().getName()`).
    fn impl_class(&self) -> &'static str {
        std::any::type_name::<Self>()
    }
}

/// A reflected attribute value (`AttributeReflector.reflect`'s `Object`),
/// whose [`Display`](std::fmt::Display) is Java's `String.valueOf`.
#[derive(Debug, Clone, PartialEq)]
pub enum AttrValue<'a> {
    /// A `String`/`CharSequence` (`None`: `null`).
    Str(Option<Cow<'a, str>>),
    /// An `int`.
    Int(i32),
    /// A `boolean`.
    Bool(bool),
    /// A `float`.
    Float(f32),
    /// A `BytesRef` (`None`: `null`), shown as `BytesRef.toString()`.
    Bytes(Option<&'a [u8]>),
    /// A `List` of `CharSequence`s (`None`: `null`), shown as
    /// `AbstractCollection.toString()`.
    List(Option<Vec<Cow<'a, str>>>),
}

impl std::fmt::Display for AttrValue<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            AttrValue::Str(None) | AttrValue::Bytes(None) | AttrValue::List(None) => {
                f.write_str("null")
            }
            AttrValue::Str(Some(s)) => f.write_str(s),
            AttrValue::Int(i) => write!(f, "{i}"),
            AttrValue::Bool(b) => write!(f, "{b}"),
            // Java's Float.toString for the values a boost takes.
            AttrValue::Float(v) if v.is_finite() && v.fract() == 0.0 && v.abs() < 1e7 => {
                write!(f, "{v:.1}")
            }
            AttrValue::Float(v) => write!(f, "{v}"),
            AttrValue::Bytes(Some(b)) => {
                f.write_str("[")?;
                for (i, byte) in b.iter().enumerate() {
                    if i > 0 {
                        f.write_str(" ")?;
                    }
                    write!(f, "{byte:x}")?;
                }
                f.write_str("]")
            }
            AttrValue::List(Some(items)) => {
                f.write_str("[")?;
                for (i, s) in items.iter().enumerate() {
                    if i > 0 {
                        f.write_str(", ")?;
                    }
                    f.write_str(s)?;
                }
                f.write_str("]")
            }
        }
    }
}

/// [`CustomAttribute`] as an object, for the source's list.
trait DynAttribute: Debug + Send + Sync {
    fn clear(&mut self);
    fn end(&mut self);
    fn clone_box(&self) -> Box<dyn DynAttribute>;
    fn eq_dyn(&self, other: &dyn DynAttribute) -> bool;
    fn copy_from(&mut self, other: &dyn DynAttribute);
    fn reflect(&self, reflector: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>));
    fn impl_class(&self) -> &'static str;
    fn as_any(&self) -> &dyn Any;
    fn as_any_mut(&mut self) -> &mut dyn Any;
}

impl<T: CustomAttribute> DynAttribute for T {
    fn clear(&mut self) {
        CustomAttribute::clear(self);
    }
    fn end(&mut self) {
        CustomAttribute::end(self);
    }
    fn clone_box(&self) -> Box<dyn DynAttribute> {
        Box::new(self.clone())
    }
    fn eq_dyn(&self, other: &dyn DynAttribute) -> bool {
        other.as_any().downcast_ref::<T>() == Some(self)
    }
    fn copy_from(&mut self, other: &dyn DynAttribute) {
        if let Some(o) = other.as_any().downcast_ref::<T>() {
            self.clone_from(o);
        }
    }
    fn reflect(&self, reflector: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>)) {
        CustomAttribute::reflect(self, reflector);
    }
    fn impl_class(&self) -> &'static str {
        CustomAttribute::impl_class(self)
    }
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
}

impl PartialEq for Box<dyn DynAttribute> {
    fn eq(&self, other: &Self) -> bool {
        self.eq_dyn(other.as_ref())
    }
}

impl Eq for Box<dyn DynAttribute> {}

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
    /// The custom attributes, in the order they were added.
    custom: Vec<Box<dyn DynAttribute>>,
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

    /// An exact copy (`clone_from` is `*self = source.clone()`), reusing
    /// this source's buffers and its instances of the source's custom
    /// attributes; a custom attribute the source lacks is dropped.
    fn clone_from(&mut self, source: &Self) {
        self.copy_core_from(source);
        let same_types = self.custom.len() == source.custom.len()
            && self
                .custom
                .iter()
                .zip(&source.custom)
                .all(|(a, b)| a.as_any().type_id() == b.as_any().type_id());
        if same_types {
            for (mine, theirs) in self.custom.iter_mut().zip(&source.custom) {
                mine.copy_from(theirs.as_ref());
            }
        } else {
            self.custom = source.custom.iter().map(|a| a.clone_box()).collect();
        }
    }
}

impl AttributeSource {
    /// The core attributes of `source`, field by field, so a restore reuses
    /// this source's term and payload buffers (a derived `clone_from` would
    /// allocate a whole new value per call). The destructuring names every
    /// field, so a field added to the struct fails to compile here until it
    /// is copied too.
    fn copy_core_from(&mut self, source: &Self) {
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
            custom,
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
        let _ = custom; // copied by the callers, each its own way
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
            custom: Vec::new(),
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
        for attr in &mut self.custom {
            attr.clear();
        }
    }

    /// `AttributeSource.endAttributes()`: every attribute's `end()`, which is
    /// its `clear()` except `PositionIncrementAttributeImpl.end()` (increment
    /// 0) and `TermFrequencyAttributeImpl.end()` (frequency 1, as clear).
    pub fn end_attributes(&mut self) {
        self.clear_attributes();
        self.position_increment = 0;
        for attr in &mut self.custom {
            attr.end();
        }
    }

    // ------------------------------------------------------ custom attributes

    /// `addAttribute(Class)` for a [`CustomAttribute`]: the source's
    /// instance, created (at its `Default`) on first use.
    pub fn add_custom<T: CustomAttribute>(&mut self) -> &mut T {
        let id = TypeId::of::<T>();
        let i = match self.custom.iter().position(|a| a.as_any().type_id() == id) {
            Some(i) => i,
            None => {
                self.custom.push(Box::new(T::default()));
                self.custom.len() - 1
            }
        };
        self.custom[i]
            .as_any_mut()
            .downcast_mut::<T>()
            .expect("the attribute at this type's index is of this type")
    }

    /// `getAttribute(Class)` for a [`CustomAttribute`]; `None` when no
    /// stage added it (`hasAttribute` is `false`).
    pub fn custom<T: CustomAttribute>(&self) -> Option<&T> {
        self.custom
            .iter()
            .find_map(|a| a.as_any().downcast_ref::<T>())
    }

    /// `AttributeSource.captureState()`.
    pub fn capture_state(&self) -> State {
        self.clone()
    }

    /// `AttributeSource.restoreState(State)`; reuses this source's buffers.
    /// Each of the state's attributes is copied into this source's instance
    /// of its class; attributes the state does not hold keep their values.
    ///
    /// Differs: Java throws `IllegalArgumentException` for a state holding
    /// an attribute this source lacks; this adds it. Within one chain that
    /// cannot happen -- a state is captured from the chain's own source,
    /// and attributes are never removed -- so only a state from another
    /// chain reaches the difference; [`Self::try_restore_state`] refuses it
    /// as Java does.
    pub fn restore_state(&mut self, state: &State) {
        self.copy_core_from(state);
        for attr in &state.custom {
            let id = attr.as_any().type_id();
            match self.custom.iter_mut().find(|a| a.as_any().type_id() == id) {
                Some(mine) => mine.copy_from(attr.as_ref()),
                None => self.custom.push(attr.clone_box()),
            }
        }
    }

    /// `AttributeSource.restoreState(State)` with Java's check: a state
    /// holding an attribute this source lacks is an `IllegalArgument`
    /// error, and nothing is restored.
    pub fn try_restore_state(&mut self, state: &State) -> Result<(), AnalysisError> {
        if let Some(missing) = state.custom.iter().find(|s| {
            let id = s.as_any().type_id();
            !self.custom.iter().any(|a| a.as_any().type_id() == id)
        }) {
            return Err(AnalysisError::IllegalArgument(format!(
                "State contains AttributeImpl of type {} that is not in in this AttributeSource",
                missing.impl_class()
            )));
        }
        self.restore_state(state);
        Ok(())
    }

    /// `AttributeSource.reflectWith(AttributeReflector)`: every attribute's
    /// key/value pairs. The core attributes come first, in the order of the
    /// default factory's `PackedTokenAttributeImpl` (term, bytes, offsets,
    /// position increment and length, type, term frequency) and then flags,
    /// keyword, payload, sentence and boost; the custom attributes follow in
    /// the order they were added (Java's `addAttribute` order).
    ///
    /// Differs: Java reflects only the attributes a chain added, in the
    /// order it added them; the core attributes here are always present
    /// (see the module docs), so they are always reflected, in this fixed
    /// order. The custom attributes' pairs and their order are Java's.
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
            AttrValue::Bytes(Some(self.term_bytes())),
        );
        let offset = "org.apache.lucene.analysis.tokenattributes.OffsetAttribute";
        reflector(offset, "startOffset", AttrValue::Int(self.start_offset));
        reflector(offset, "endOffset", AttrValue::Int(self.end_offset));
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
            "org.apache.lucene.analysis.tokenattributes.KeywordAttribute",
            "keyword",
            AttrValue::Bool(self.keyword),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.PayloadAttribute",
            "payload",
            AttrValue::Bytes(self.payload.as_deref()),
        );
        reflector(
            "org.apache.lucene.analysis.tokenattributes.SentenceAttribute",
            "sentences",
            AttrValue::Int(self.sentence_index),
        );
        reflector(
            "org.apache.lucene.search.BoostAttribute",
            "boost",
            AttrValue::Float(self.boost()),
        );
        self.reflect_custom(reflector);
    }

    /// The custom attributes' part of [`Self::reflect_with`] alone, in the
    /// order they were added.
    pub fn reflect_custom(
        &self,
        reflector: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>),
    ) {
        for attr in &self.custom {
            attr.reflect(reflector);
        }
    }

    /// `AttributeSource.reflectAsString(prependAttClass)` over
    /// [`Self::reflect_with`].
    pub fn reflect_as_string(&self, prepend_att_class: bool) -> String {
        let mut out = String::new();
        self.reflect_with(&mut |class, key, value| {
            if !out.is_empty() {
                out.push(',');
            }
            if prepend_att_class {
                out.push_str(class);
                out.push('#');
            }
            out.push_str(key);
            out.push('=');
            out.push_str(&value.to_string());
        });
        out
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

    /// `TypeAttribute.type()` as stored: a clone of a type set from a
    /// `&'static str` copies no bytes.
    pub fn token_type_cow(&self) -> &Cow<'static, str> {
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

    #[derive(Debug, Clone, PartialEq, Eq, Default)]
    struct Tags(Option<Vec<String>>);

    impl CustomAttribute for Tags {
        fn clear(&mut self) {
            self.0 = None;
        }
        fn reflect(&self, r: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>)) {
            let tags = self
                .0
                .as_ref()
                .map(|t| t.iter().map(|s| Cow::Borrowed(s.as_str())).collect());
            r("x.Tags", "tags", AttrValue::List(tags));
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq, Default)]
    struct Sticky(u32);

    impl CustomAttribute for Sticky {
        fn clear(&mut self) {
            self.0 = 0;
        }
        fn reflect(&self, r: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>)) {
            r("x.Sticky", "n", AttrValue::Int(self.0 as i32));
            r("x.Sticky", "s", AttrValue::Str(None));
        }
        fn end(&mut self) {
            self.0 = 99;
        }
    }

    #[test]
    fn custom_attributes_follow_the_lifecycle() {
        let mut a = AttributeSource::new();
        assert!(a.custom::<Tags>().is_none());
        a.add_custom::<Tags>().0 = Some(vec!["x".into()]);
        assert_eq!(a.custom::<Tags>(), Some(&Tags(Some(vec!["x".into()]))));
        let state = a.capture_state();
        a.add_custom::<Tags>().0 = None;
        a.add_custom::<Sticky>().0 = 5;
        assert_ne!(a, state);
        a.restore_state(&state);
        assert_eq!(a.custom::<Tags>(), state.custom::<Tags>());
        assert_eq!(a.custom::<Sticky>(), Some(&Sticky(5)));
        // A state holding an attribute this source lacks adds it; Java's
        // check refuses it.
        let mut b = AttributeSource::new();
        let e = b.try_restore_state(&state).unwrap_err().to_string();
        assert!(
            e.contains("State contains AttributeImpl of type") && e.contains("Tags"),
            "{e}"
        );
        assert!(b.custom::<Tags>().is_none());
        b.restore_state(&state);
        assert_eq!(b.custom::<Tags>(), state.custom::<Tags>());
        b.try_restore_state(&state).unwrap();
        // clone_from is an exact copy: an attribute the source lacks goes.
        let mut c = a.clone();
        c.clone_from(&state);
        assert_eq!(c, state);
        assert!(c.custom::<Sticky>().is_none());
        c.clone_from(&state);
        assert_eq!(c, state);
        a.clear_attributes();
        assert_eq!(a.custom::<Tags>(), Some(&Tags(None)));
        a.end_attributes();
        assert_eq!(a.custom::<Sticky>(), Some(&Sticky(99)));
        assert_eq!(a.clone(), a);
        assert!(format!("{a:?}").contains("Sticky"));
    }

    #[test]
    fn reflection_lists_core_then_custom_attributes() {
        let mut a = dirty();
        a.add_custom::<Sticky>().0 = 3;
        a.add_custom::<Tags>().0 = Some(vec!["a".into(), "b".into()]);
        assert_eq!(
            a.reflect_as_string(false),
            "term=Foo,bytes=[1 2],startOffset=3,endOffset=7,positionIncrement=4,positionLength=2,\
             type=<NUM>,termFrequency=6,flags=9,keyword=true,payload=[5],sentences=8,boost=1.0,\
             n=3,s=null,tags=[a, b]"
        );
        let mut custom = Vec::new();
        a.reflect_custom(&mut |c, k, v| custom.push(format!("{c}#{k}={v}")));
        assert_eq!(
            custom,
            ["x.Sticky#n=3", "x.Sticky#s=null", "x.Tags#tags=[a, b]"]
        );
        let fresh = AttributeSource::new().reflect_as_string(true);
        assert!(fresh
            .starts_with("org.apache.lucene.analysis.tokenattributes.CharTermAttribute#term=,"));
        assert!(fresh.contains("PayloadAttribute#payload=null"), "{fresh}");
        a.set_boost(2.5);
        a.add_custom::<Tags>().0 = None;
        let s = a.reflect_as_string(false);
        assert!(s.contains("boost=2.5") && s.ends_with("tags=null"), "{s}");
        assert_eq!(AttrValue::Float(1e8).to_string(), "100000000");
        assert!(CustomAttribute::impl_class(&Sticky(0)).ends_with("Sticky"));
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
