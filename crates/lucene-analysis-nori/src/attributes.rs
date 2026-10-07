//! `org.apache.lucene.analysis.ko.tokenattributes`: the two attributes
//! `KoreanTokenizer` sets, each holding the current [`Token`] and asking it
//! lazily. Interface and implementation are one [`CustomAttribute`] type
//! each.

use std::borrow::Cow;
use std::sync::Arc;

use lucene_analysis::{AttrValue, CustomAttribute};

use crate::dict::Morpheme;
use crate::pos::{Tag, Type};
use crate::token::Token;

type Reflector<'r> = dyn FnMut(&'static str, &'static str, AttrValue<'_>) + 'r;

/// `PartOfSpeechAttribute` / `PartOfSpeechAttributeImpl`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PartOfSpeechAttribute {
    /// `setToken(token)`.
    pub token: Option<Arc<Token>>,
}

impl PartOfSpeechAttribute {
    /// `getPOSType()`.
    pub fn pos_type(&self) -> Option<Type> {
        Some(self.token.as_ref()?.pos_type())
    }
    /// `getLeftPOS()`.
    pub fn left_pos(&self) -> Option<Tag> {
        self.token.as_ref()?.left_pos()
    }
    /// `getRightPOS()`.
    pub fn right_pos(&self) -> Option<Tag> {
        self.token.as_ref()?.right_pos()
    }
    /// `getMorphemes()`.
    pub fn morphemes(&self) -> Option<Vec<Morpheme>> {
        self.token.as_ref()?.morphemes()
    }
}

fn tag_text(t: Tag) -> String {
    format!("{}({})", t.name(), t.description())
}

/// `PartOfSpeechAttributeImpl.displayMorphemes`.
fn display_morphemes(morphemes: Option<Vec<Morpheme>>) -> Option<String> {
    let morphemes = morphemes?;
    let mut b = String::new();
    for m in morphemes {
        if !b.is_empty() {
            b.push('+');
        }
        b.push_str(&format!("{}/{}", m.surface_form, tag_text(m.pos_tag)));
    }
    Some(b)
}

impl CustomAttribute for PartOfSpeechAttribute {
    fn impl_class(&self) -> &'static str {
        "org.apache.lucene.analysis.ko.tokenattributes.PartOfSpeechAttributeImpl"
    }
    fn clear(&mut self) {
        self.token = None;
    }
    fn reflect(&self, r: &mut Reflector<'_>) {
        const C: &str = "org.apache.lucene.analysis.ko.tokenattributes.PartOfSpeechAttribute";
        let s = |v: Option<String>| AttrValue::Str(v.map(Cow::Owned));
        r(
            C,
            "posType",
            s(self.pos_type().map(|t| t.name().to_string())),
        );
        r(C, "leftPOS", s(self.left_pos().map(tag_text)));
        r(C, "rightPOS", s(self.right_pos().map(tag_text)));
        r(C, "morphemes", s(display_morphemes(self.morphemes())));
    }
}

/// `ReadingAttribute` / `ReadingAttributeImpl`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct ReadingAttribute {
    /// `setToken(token)`.
    pub token: Option<Arc<Token>>,
}

impl ReadingAttribute {
    /// `getReading()`.
    pub fn reading(&self) -> Option<String> {
        self.token.as_ref()?.reading()
    }
}

impl CustomAttribute for ReadingAttribute {
    fn impl_class(&self) -> &'static str {
        "org.apache.lucene.analysis.ko.tokenattributes.ReadingAttributeImpl"
    }
    fn clear(&mut self) {
        self.token = None;
    }
    fn reflect(&self, r: &mut Reflector<'_>) {
        r(
            "org.apache.lucene.analysis.ko.tokenattributes.ReadingAttribute",
            "reading",
            AttrValue::Str(self.reading().map(Cow::Owned)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_analysis::AttributeSource;

    /// The implementation class `restoreState`'s error names.
    fn refused<T: CustomAttribute>() -> String {
        let mut a = AttributeSource::new();
        a.add_custom::<T>();
        AttributeSource::new()
            .try_restore_state(&a.capture_state())
            .unwrap_err()
            .to_string()
    }

    #[test]
    fn restore_state_names_java_impl_classes() {
        for (e, class) in [
            (
                refused::<PartOfSpeechAttribute>(),
                "org.apache.lucene.analysis.ko.tokenattributes.PartOfSpeechAttributeImpl",
            ),
            (
                refused::<ReadingAttribute>(),
                "org.apache.lucene.analysis.ko.tokenattributes.ReadingAttributeImpl",
            ),
        ] {
            assert!(e.contains(&format!("type {class} that")), "{e}");
        }
    }
}
