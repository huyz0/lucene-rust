//! `org.apache.lucene.analysis.icu.tokenattributes.ScriptAttribute` and
//! `ScriptAttributeImpl`: the ISO 15924 script code of a token, reflected
//! by its long name (`Chinese/Japanese` for the combined Japanese runs).

use std::borrow::Cow;

use lucene_analysis::{AttrValue, CustomAttribute};

use crate::icu4j::uprops::uprops;
use crate::segmentation::script_iterator::{COMMON, JAPANESE};

/// `ScriptAttribute` / `ScriptAttributeImpl`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ScriptAttribute {
    code: i32,
}

impl Default for ScriptAttribute {
    fn default() -> Self {
        ScriptAttribute { code: COMMON }
    }
}

impl ScriptAttribute {
    /// `getCode()`.
    pub fn code(&self) -> i32 {
        self.code
    }

    /// `setCode(code)`.
    pub fn set_code(&mut self, code: i32) {
        self.code = code;
    }

    /// `getName()`: `UScript.getName(code)`.
    pub fn name(&self) -> &'static str {
        uprops().script_name(self.code).map_or("", |(_, long)| long)
    }

    /// `getShortName()`: `UScript.getShortName(code)`.
    pub fn short_name(&self) -> &'static str {
        uprops()
            .script_name(self.code)
            .map_or("", |(short, _)| short)
    }
}

impl CustomAttribute for ScriptAttribute {
    fn impl_class(&self) -> &'static str {
        "org.apache.lucene.analysis.icu.tokenattributes.ScriptAttributeImpl"
    }

    fn clear(&mut self) {
        self.code = COMMON;
    }

    fn reflect(&self, r: &mut dyn FnMut(&'static str, &'static str, AttrValue<'_>)) {
        let name = if self.code == JAPANESE {
            "Chinese/Japanese"
        } else {
            self.name()
        };
        r(
            "org.apache.lucene.analysis.icu.tokenattributes.ScriptAttribute",
            "script",
            AttrValue::Str(Some(Cow::Borrowed(name))),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn names_and_reflection() {
        let mut a = ScriptAttribute::default();
        assert_eq!(a.code(), 0);
        assert_eq!(a.name(), "Common");
        a.set_code(25);
        assert_eq!((a.name(), a.short_name()), ("Latin", "Latn"));
        let mut seen = Vec::new();
        a.reflect(&mut |c, k, v| seen.push((c, k, v.to_string())));
        assert_eq!(seen[0].2, "Latin");
        a.set_code(JAPANESE);
        let mut seen = Vec::new();
        a.reflect(&mut |_, _, v| seen.push(v.to_string()));
        assert_eq!(seen, vec!["Chinese/Japanese"]);
        a.clear();
        assert_eq!(a.code(), COMMON);
        assert!(a.impl_class().ends_with("ScriptAttributeImpl"));
        a.set_code(-5);
        assert_eq!(a.name(), "");
    }
}
