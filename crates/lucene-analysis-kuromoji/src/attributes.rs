//! `org.apache.lucene.analysis.ja.tokenattributes`: the four attributes
//! `JapaneseTokenizer` sets, each holding the current [`Token`] and asking
//! it lazily, as Java's `*AttributeImpl`s do. Interface and implementation
//! are one [`CustomAttribute`] type each; `setToken` is a field.

use std::borrow::Cow;
use std::sync::Arc;

use lucene_analysis::{AttrValue, CustomAttribute};

use crate::dict::to_string_util;
use crate::token::Token;

type Reflector<'r> = dyn FnMut(&'static str, &'static str, AttrValue<'_>) + 'r;

fn s(v: Option<String>) -> AttrValue<'static> {
    AttrValue::Str(v.map(Cow::Owned))
}

fn s_static(v: Option<&'static str>) -> AttrValue<'static> {
    AttrValue::Str(v.map(Cow::Borrowed))
}

/// `BaseFormAttribute` / `BaseFormAttributeImpl`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct BaseFormAttribute {
    /// `setToken(token)`.
    pub token: Option<Arc<Token>>,
}

impl BaseFormAttribute {
    /// `getBaseForm()`.
    pub fn base_form(&self) -> Option<String> {
        self.token.as_ref()?.base_form()
    }
}

impl CustomAttribute for BaseFormAttribute {
    fn impl_class(&self) -> &'static str {
        "org.apache.lucene.analysis.ja.tokenattributes.BaseFormAttributeImpl"
    }
    fn clear(&mut self) {
        self.token = None;
    }
    fn reflect(&self, r: &mut Reflector<'_>) {
        r(
            "org.apache.lucene.analysis.ja.tokenattributes.BaseFormAttribute",
            "baseForm",
            s(self.base_form()),
        );
    }
}

/// `InflectionAttribute` / `InflectionAttributeImpl`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct InflectionAttribute {
    /// `setToken(token)`.
    pub token: Option<Arc<Token>>,
}

impl InflectionAttribute {
    /// `getInflectionType()`.
    pub fn inflection_type(&self) -> Option<String> {
        self.token.as_ref()?.inflection_type()
    }
    /// `getInflectionForm()`.
    pub fn inflection_form(&self) -> Option<String> {
        self.token.as_ref()?.inflection_form()
    }
}

impl CustomAttribute for InflectionAttribute {
    fn impl_class(&self) -> &'static str {
        "org.apache.lucene.analysis.ja.tokenattributes.InflectionAttributeImpl"
    }
    fn clear(&mut self) {
        self.token = None;
    }
    fn reflect(&self, r: &mut Reflector<'_>) {
        const C: &str = "org.apache.lucene.analysis.ja.tokenattributes.InflectionAttribute";
        let t = self.inflection_type();
        let t_en = t
            .as_deref()
            .and_then(to_string_util::inflection_type_translation);
        r(C, "inflectionType", s(t));
        r(C, "inflectionType (en)", s_static(t_en));
        let f = self.inflection_form();
        let f_en = f
            .as_deref()
            .and_then(to_string_util::inflected_form_translation);
        r(C, "inflectionForm", s(f));
        r(C, "inflectionForm (en)", s_static(f_en));
    }
}

/// `PartOfSpeechAttribute` / `PartOfSpeechAttributeImpl`.
#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct PartOfSpeechAttribute {
    /// `setToken(token)`.
    pub token: Option<Arc<Token>>,
}

impl PartOfSpeechAttribute {
    /// `getPartOfSpeech()`.
    pub fn part_of_speech(&self) -> Option<&str> {
        self.token.as_ref()?.part_of_speech()
    }
}

impl CustomAttribute for PartOfSpeechAttribute {
    fn impl_class(&self) -> &'static str {
        "org.apache.lucene.analysis.ja.tokenattributes.PartOfSpeechAttributeImpl"
    }
    fn clear(&mut self) {
        self.token = None;
    }
    fn reflect(&self, r: &mut Reflector<'_>) {
        const C: &str = "org.apache.lucene.analysis.ja.tokenattributes.PartOfSpeechAttribute";
        let pos = self.part_of_speech();
        let pos_en = pos.and_then(to_string_util::pos_translation);
        r(C, "partOfSpeech", AttrValue::Str(pos.map(Cow::Borrowed)));
        r(C, "partOfSpeech (en)", s_static(pos_en));
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
    /// `getPronunciation()`.
    pub fn pronunciation(&self) -> Option<String> {
        self.token.as_ref()?.pronunciation()
    }
}

impl CustomAttribute for ReadingAttribute {
    fn impl_class(&self) -> &'static str {
        "org.apache.lucene.analysis.ja.tokenattributes.ReadingAttributeImpl"
    }
    fn clear(&mut self) {
        self.token = None;
    }
    fn reflect(&self, r: &mut Reflector<'_>) {
        const C: &str = "org.apache.lucene.analysis.ja.tokenattributes.ReadingAttribute";
        let reading = self.reading();
        let reading_en = reading.as_deref().map(to_string_util::romanization);
        let pron = self.pronunciation();
        let pron_en = pron.as_deref().map(to_string_util::romanization);
        r(C, "reading", s(reading));
        r(C, "reading (en)", s(reading_en));
        r(C, "pronunciation", s(pron));
        r(C, "pronunciation (en)", s(pron_en));
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
                refused::<BaseFormAttribute>(),
                "org.apache.lucene.analysis.ja.tokenattributes.BaseFormAttributeImpl",
            ),
            (
                refused::<InflectionAttribute>(),
                "org.apache.lucene.analysis.ja.tokenattributes.InflectionAttributeImpl",
            ),
            (
                refused::<PartOfSpeechAttribute>(),
                "org.apache.lucene.analysis.ja.tokenattributes.PartOfSpeechAttributeImpl",
            ),
            (
                refused::<ReadingAttribute>(),
                "org.apache.lucene.analysis.ja.tokenattributes.ReadingAttributeImpl",
            ),
        ] {
            assert!(e.contains(&format!("type {class} that")), "{e}");
        }
    }
}
