//! `ja.dict.UnknownDictionary` and `ja.dict.UnknownMorphData`: the words
//! proposed for a run of one character class, by class (the target map's
//! source id is the class).
//!
//! `UnknownMorphData` is `TokenInfoMorphData` with no reading and no
//! inflection; [`super::JaDict`] answers those as `null`.

use std::path::Path;
use std::sync::{Arc, LazyLock};

use lucene_analysis::morph::binary_dictionary::BinaryDictionary;
use lucene_analysis::AnalysisError;

use super::character_definition::CharacterDefinition;
use super::token_info_dictionary::TokenInfoMorphData;
use super::{inflate, read_file, DICT_HEADER, TARGETMAP_HEADER, VERSION};

/// `UnknownDictionary`.
#[derive(Debug)]
pub struct UnknownDictionary {
    character_definition: Arc<CharacterDefinition>,
    morph_atts: TokenInfoMorphData,
}

impl UnknownDictionary {
    /// The dictionary over a target map, part-of-speech table and entry
    /// buffer, each a file's bytes. As in Java, the character definition is
    /// always the default one.
    pub fn read(target_map: &[u8], pos_dict: &[u8], dict: &[u8]) -> Result<Self, AnalysisError> {
        let bin = BinaryDictionary::read(target_map, dict, TARGETMAP_HEADER, DICT_HEADER, VERSION)?;
        Ok(UnknownDictionary {
            character_definition: CharacterDefinition::instance(),
            morph_atts: TokenInfoMorphData::read(bin, pos_dict)?,
        })
    }

    /// `new UnknownDictionary(targetMapFile, posDictFile, dictFile)`.
    pub fn from_paths(
        target_map: &Path,
        pos_dict: &Path,
        dict: &Path,
    ) -> Result<Self, AnalysisError> {
        Self::read(
            &read_file(target_map)?,
            &read_file(pos_dict)?,
            &read_file(dict)?,
        )
    }

    /// `getInstance()`.
    pub fn instance() -> Arc<UnknownDictionary> {
        static INSTANCE: LazyLock<Arc<UnknownDictionary>> = LazyLock::new(|| {
            Arc::new(
                UnknownDictionary::read(
                    &inflate(include_bytes!("../resources/unknown_target_map.dat.z")),
                    &inflate(include_bytes!("../resources/unknown_pos_dict.dat.z")),
                    &inflate(include_bytes!("../resources/unknown_buffer.dat.z")),
                )
                .expect("the vendored unknown dictionary reads"),
            )
        });
        Arc::clone(&INSTANCE)
    }

    /// `getMorphAttributes()`.
    pub fn morph_attributes(&self) -> &TokenInfoMorphData {
        &self.morph_atts
    }

    /// `getCharacterDefinition()`.
    pub fn character_definition(&self) -> &Arc<CharacterDefinition> {
        &self.character_definition
    }

    /// `lookupWordIds(characterClass, ref)`.
    pub fn lookup_word_ids(&self, source_id: i32) -> &[i32] {
        self.morph_atts
            .binary_dictionary()
            .lookup_word_ids(source_id)
    }

    /// `lookup(text, offset, len)`: the length of the unknown word at
    /// `text[0]` -- 1, or the run of its class when the class groups.
    pub fn lookup(&self, text: &[u16]) -> usize {
        let Some(&first) = text.first() else {
            return 0;
        };
        if !self.character_definition.is_group(first) {
            return 1;
        }
        let class = self.character_definition.character_class(first);
        text[1..]
            .iter()
            .take_while(|&&c| self.character_definition.character_class(c) == class)
            .count()
            .saturating_add(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lookup_groups_a_run_of_one_class() {
        let d = UnknownDictionary::instance();
        let u = |s: &str| s.encode_utf16().collect::<Vec<u16>>();
        assert_eq!(d.lookup(&u("ABC漢")), 3);
        assert_eq!(d.lookup(&u("漢字")), 1); // KANJI does not group
        assert_eq!(d.lookup(&[]), 0);
        assert!(!d.lookup_word_ids(5).is_empty());
        assert!(d.morph_attributes().binary_dictionary().buffer().len() > 4);
    }
}
