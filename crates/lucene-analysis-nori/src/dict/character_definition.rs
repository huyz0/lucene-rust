//! `ko.dict.CharacterDefinition`: mecab-ko-dic's fourteen character
//! classes over [`lucene_analysis::morph::CharacterDefinition`].

use std::sync::{Arc, LazyLock};

use lucene_analysis::morph::CharacterDefinition as MorphCharacterDefinition;
use lucene_analysis::AnalysisError;

use super::{inflate, CHARDEF_HEADER, VERSION};

/// `CharacterClass.NGRAM`.
pub const NGRAM: u8 = 0;
/// `CharacterClass.HANGUL`.
pub const HANGUL: u8 = 11;
/// `CharacterClass.HANJA`.
pub const HANJA: u8 = 12;
/// `CharacterClass.HANJANUMERIC`.
pub const HANJANUMERIC: u8 = 13;
/// `CharacterDefinition.CLASS_COUNT`.
pub const CLASS_COUNT: usize = 14;

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
    "HANGUL",
    "HANJA",
    "HANJANUMERIC",
];

/// `ko.dict.CharacterDefinition`.
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

    /// `isHanja(char)`.
    pub fn is_hanja(&self, c: u16) -> bool {
        let class = self.character_class(c);
        class == HANJA || class == HANJANUMERIC
    }

    /// `isHangul(char)`.
    pub fn is_hangul(&self, c: u16) -> bool {
        self.character_class(c) == HANGUL
    }

    /// `hasCoda(char)`: `((ch - 0xAC00) % 0x001C) != 0` in Java `int`s.
    pub fn has_coda(&self, ch: u16) -> bool {
        i32::from(ch).wrapping_sub(0xAC00).wrapping_rem(0x1C) != 0
    }

    /// `lookupCharacterClass(name)` (`IllegalArgumentException` for an
    /// unknown name).
    pub fn lookup_character_class(name: &str) -> Result<u8, AnalysisError> {
        CLASS_NAMES
            .iter()
            .position(|&n| n == name)
            .map(|i| i as u8)
            .ok_or_else(|| {
                AnalysisError::IllegalArgument(format!(
                    "No enum constant org.apache.lucene.analysis.ko.dict.CharacterDefinition.CharacterClass.{name}"
                ))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn classes_of_the_mecab_ko_definition() {
        let d = CharacterDefinition::instance();
        assert!(d.is_hangul(0xAC00));
        assert!(!d.has_coda(0xAC00)); // 가
        assert!(d.has_coda(0xAC01)); // 각
        assert!(d.has_coda(0x41));
        assert!(d.is_hanja(0x4E00));
        assert!(!d.is_hanja(0xAC00));
        assert_eq!(
            CharacterDefinition::lookup_character_class("HANGUL").unwrap(),
            HANGUL
        );
        assert!(CharacterDefinition::lookup_character_class("X").is_err());
        assert!(CharacterDefinition::read(b"nope").is_err());
        assert_eq!(d.character_class(0xAC00), HANGUL);
        let _ = NGRAM;
    }
}
