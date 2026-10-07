//! `ja.dict.CharacterDefinition`: IPADIC's twelve character classes over
//! [`lucene_analysis::morph::CharacterDefinition`].

use std::sync::{Arc, LazyLock};

use lucene_analysis::morph::CharacterDefinition as MorphCharacterDefinition;
use lucene_analysis::AnalysisError;

use super::{inflate, CHARDEF_HEADER, VERSION};

/// `CharacterClass.NGRAM`.
pub const NGRAM: u8 = 0;
/// `CharacterClass.DEFAULT`.
pub const DEFAULT: u8 = 1;
/// `CharacterClass.SPACE`.
pub const SPACE: u8 = 2;
/// `CharacterClass.SYMBOL`.
pub const SYMBOL: u8 = 3;
/// `CharacterClass.NUMERIC`.
pub const NUMERIC: u8 = 4;
/// `CharacterClass.ALPHA`.
pub const ALPHA: u8 = 5;
/// `CharacterClass.CYRILLIC`.
pub const CYRILLIC: u8 = 6;
/// `CharacterClass.GREEK`.
pub const GREEK: u8 = 7;
/// `CharacterClass.HIRAGANA`.
pub const HIRAGANA: u8 = 8;
/// `CharacterClass.KATAKANA`.
pub const KATAKANA: u8 = 9;
/// `CharacterClass.KANJI`.
pub const KANJI: u8 = 10;
/// `CharacterClass.KANJINUMERIC`.
pub const KANJINUMERIC: u8 = 11;
/// `CharacterDefinition.CLASS_COUNT`.
pub const CLASS_COUNT: usize = 12;

const CLASS_NAMES: [&str; CLASS_COUNT] = [
    "NGRAM",
    "DEFAULT",
    "SPACE",
    "SYMBOL",
    "NUMERIC",
    "ALPHA",
    "CYRILLIC",
    "GREEK",
    "HIRAGANA",
    "KATAKANA",
    "KANJI",
    "KANJINUMERIC",
];

/// `ja.dict.CharacterDefinition`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CharacterDefinition {
    def: MorphCharacterDefinition,
}

impl std::ops::Deref for CharacterDefinition {
    type Target = MorphCharacterDefinition;
    fn deref(&self) -> &MorphCharacterDefinition {
        &self.def
    }
}

impl CharacterDefinition {
    /// The definition in a `CharacterDefinition.dat` file's bytes.
    pub fn read(bytes: &[u8]) -> Result<Self, AnalysisError> {
        Ok(CharacterDefinition {
            def: MorphCharacterDefinition::read(bytes, CHARDEF_HEADER, VERSION, CLASS_COUNT)?,
        })
    }

    /// `getInstance()`.
    pub fn instance() -> Arc<CharacterDefinition> {
        static INSTANCE: LazyLock<Arc<CharacterDefinition>> = LazyLock::new(|| {
            Arc::new(
                CharacterDefinition::read(&inflate(include_bytes!(
                    "../resources/character_definition.dat.z"
                )))
                .expect("the vendored character definition reads"),
            )
        });
        Arc::clone(&INSTANCE)
    }

    /// `isKanji(char)`.
    pub fn is_kanji(&self, c: u16) -> bool {
        let class = self.character_class(c);
        class == KANJI || class == KANJINUMERIC
    }

    /// `lookupCharacterClass(name)`: `CharacterClass.valueOf(name).ordinal()`
    /// (`IllegalArgumentException` for an unknown name).
    pub fn lookup_character_class(name: &str) -> Result<u8, AnalysisError> {
        CLASS_NAMES
            .iter()
            .position(|&n| n == name)
            .map(|i| i as u8)
            .ok_or_else(|| {
                AnalysisError::IllegalArgument(format!(
                    "No enum constant org.apache.lucene.analysis.ja.dict.CharacterDefinition.CharacterClass.{name}"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_of_the_ipadic_definition() {
        let d = CharacterDefinition::instance();
        assert_eq!(d.character_class(u16::from(b'A')), ALPHA);
        assert_eq!(d.character_class(0x3042), HIRAGANA);
        assert_eq!(d.character_class(0x30A2), KATAKANA);
        assert!(d.is_kanji(0x65E5));
        assert!(d.is_kanji(0x4E00)); // 一: KANJINUMERIC
        assert!(!d.is_kanji(0x3042));
        assert_eq!(
            CharacterDefinition::lookup_character_class("KANJI").unwrap(),
            KANJI
        );
        assert!(CharacterDefinition::lookup_character_class("X").is_err());
        assert!(CharacterDefinition::read(b"nope").is_err());
    }
}
