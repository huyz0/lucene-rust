//! `org.apache.lucene.analysis.ja.dict`: the IPADIC system dictionary
//! ([`TokenInfoDictionary`]), the unknown-word dictionary
//! ([`UnknownDictionary`]), [`CharacterDefinition`], [`ConnectionCosts`],
//! user dictionaries ([`UserDictionary`]) and [`to_string_util`].
//!
//! The default instances (`getInstance()`) read the dictionary Lucene's jar
//! carries -- mecab-ipadic-2.7.0-20070801 compiled by Lucene's builder --
//! vendored zlib-compressed under `src/resources/` (licences in
//! `docs/licences.md`); every class also loads a caller's files
//! (`from_paths`) or bytes (`read`). The builders (`DictionaryBuilder`,
//! `TokenInfoDictionaryBuilder`, `UnknownDictionaryBuilder`,
//! `ConnectionCostsBuilder` and their writers) are not ported: M12 loads
//! dictionaries, it does not build them.

pub mod character_definition;
pub mod connection_costs;
pub mod to_string_util;
pub mod token_info_dictionary;
pub mod unknown_dictionary;
pub mod user_dictionary;

use std::sync::Arc;

pub use character_definition::CharacterDefinition;
pub use connection_costs::ConnectionCosts;
pub use token_info_dictionary::{TokenInfoDictionary, TokenInfoMorphData};
pub use unknown_dictionary::UnknownDictionary;
pub use user_dictionary::UserDictionary;

use lucene_analysis::morph::{MorphData, TokenType};
use lucene_analysis::AnalysisError;

/// `DictionaryConstants.DICT_HEADER`.
pub const DICT_HEADER: &str = "kuromoji_dict";
/// `DictionaryConstants.TARGETMAP_HEADER`.
pub const TARGETMAP_HEADER: &str = "kuromoji_dict_map";
/// `DictionaryConstants.POSDICT_HEADER`.
pub const POSDICT_HEADER: &str = "kuromoji_dict_pos";
/// `DictionaryConstants.CONN_COSTS_HEADER`.
pub const CONN_COSTS_HEADER: &str = "kuromoji_cc";
/// `DictionaryConstants.CHARDEF_HEADER`.
pub const CHARDEF_HEADER: &str = "kuromoji_cd";
/// `DictionaryConstants.VERSION`.
pub const VERSION: i32 = 1;

/// Inflates a vendored resource.
pub(crate) fn inflate(z: &[u8]) -> Vec<u8> {
    miniz_oxide::inflate::decompress_to_vec_zlib(z).expect("a vendored dictionary inflates")
}

/// Reads a caller's dictionary file (`Files.newInputStream(path)`).
pub(crate) fn read_file(path: &std::path::Path) -> Result<Vec<u8>, AnalysisError> {
    std::fs::read(path).map_err(|e| {
        let class = if e.kind() == std::io::ErrorKind::NotFound {
            "NoSuchFileException"
        } else {
            "IOException"
        };
        AnalysisError::Io(format!("{class}: {}: {e}", path.display()))
    })
}

/// The dictionary a token came from (`JaMorphData`): Java's
/// `dictionaryMap.get(type).getMorphAttributes()`.
#[derive(Debug, Clone)]
pub enum JaDict {
    /// The system dictionary (`TokenInfoMorphData`).
    Known(Arc<TokenInfoDictionary>),
    /// The unknown-word dictionary (`UnknownMorphData`).
    Unknown(Arc<UnknownDictionary>),
    /// A user dictionary (`UserMorphData`).
    User(Arc<UserDictionary>),
}

impl PartialEq for JaDict {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (JaDict::Known(a), JaDict::Known(b)) => Arc::ptr_eq(a, b),
            (JaDict::Unknown(a), JaDict::Unknown(b)) => Arc::ptr_eq(a, b),
            (JaDict::User(a), JaDict::User(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for JaDict {}

impl JaDict {
    /// The token type of this dictionary.
    pub fn token_type(&self) -> TokenType {
        match self {
            JaDict::Known(_) => TokenType::Known,
            JaDict::Unknown(_) => TokenType::Unknown,
            JaDict::User(_) => TokenType::User,
        }
    }

    /// The connection ids and costs.
    pub fn morph_data(&self) -> &dyn MorphData {
        match self {
            JaDict::Known(d) => d.morph_attributes(),
            JaDict::Unknown(d) => d.morph_attributes(),
            JaDict::User(d) => d.morph_attributes(),
        }
    }

    /// `getPartOfSpeech(morphId)`.
    pub fn part_of_speech(&self, morph_id: i32) -> Option<String> {
        match self {
            JaDict::Known(d) => d
                .morph_attributes()
                .part_of_speech(morph_id)
                .map(str::to_string),
            JaDict::Unknown(d) => d
                .morph_attributes()
                .part_of_speech(morph_id)
                .map(str::to_string),
            JaDict::User(d) => d.morph_attributes().part_of_speech(morph_id),
        }
    }

    /// `getReading(morphId, surface, off, len)`.
    pub fn reading(&self, morph_id: i32, surface: &[u16], off: i32, len: i32) -> Option<String> {
        match self {
            JaDict::Known(d) => Some(d.morph_attributes().reading(morph_id, surface, off, len)),
            JaDict::Unknown(_) => None,
            JaDict::User(d) => d.morph_attributes().reading(morph_id),
        }
    }

    /// `getBaseForm(morphId, surface, off, len)`.
    pub fn base_form(&self, morph_id: i32, surface: &[u16], off: i32, len: i32) -> Option<String> {
        match self {
            JaDict::Known(d) => d.morph_attributes().base_form(morph_id, surface, off, len),
            JaDict::Unknown(d) => d.morph_attributes().base_form(morph_id, surface, off, len),
            JaDict::User(_) => None,
        }
    }

    /// `getPronunciation(morphId, surface, off, len)`.
    pub fn pronunciation(
        &self,
        morph_id: i32,
        surface: &[u16],
        off: i32,
        len: i32,
    ) -> Option<String> {
        match self {
            JaDict::Known(d) => Some(
                d.morph_attributes()
                    .pronunciation(morph_id, surface, off, len, false),
            ),
            JaDict::Unknown(d) => {
                // UnknownMorphData: no reading, so the pronunciation falls
                // back to a null reading.
                let m = d.morph_attributes();
                m.has_pronunciation_data(morph_id)
                    .then(|| m.pronunciation(morph_id, surface, off, len, true))
            }
            JaDict::User(_) => None,
        }
    }

    /// `getInflectionType(morphId)`.
    pub fn inflection_type(&self, morph_id: i32) -> Option<String> {
        match self {
            JaDict::Known(d) => d
                .morph_attributes()
                .inflection_type(morph_id)
                .map(str::to_string),
            _ => None,
        }
    }

    /// `getInflectionForm(morphId)`.
    pub fn inflection_form(&self, morph_id: i32) -> Option<String> {
        match self {
            JaDict::Known(d) => d
                .morph_attributes()
                .inflection_form(morph_id)
                .map(str::to_string),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dictionaries_compare_by_identity() {
        let k = JaDict::Known(TokenInfoDictionary::instance());
        let u = JaDict::Unknown(UnknownDictionary::instance());
        let user = Arc::new(UserDictionary::open("a,a,a,n").unwrap().unwrap());
        let us = JaDict::User(Arc::clone(&user));
        assert_eq!(k, k.clone());
        assert_eq!(u, u.clone());
        assert_eq!(us, JaDict::User(user));
        assert_ne!(k, u);
        assert_eq!(us.token_type(), TokenType::User);
        let id = user_dictionary::CUSTOM_DICTIONARY_WORD_ID_OFFSET;
        assert_eq!(us.morph_data().word_cost(id), user_dictionary::WORD_COST);
        assert_eq!(us.part_of_speech(id).as_deref(), Some("n"));
        assert_eq!(us.reading(id, &[], 0, 0).as_deref(), Some("a"));
        assert_eq!(
            (us.base_form(id, &[], 0, 0), us.pronunciation(id, &[], 0, 0)),
            (None, None)
        );
        let uid = UnknownDictionary::instance().lookup_word_ids(1)[0];
        assert_eq!((us.inflection_type(id), u.inflection_form(uid)), (None, None));
        assert!(u.part_of_speech(uid).is_some());
        assert_eq!(u.base_form(uid, &[], 0, 0), None);
        assert!(read_file(std::path::Path::new("/"))
            .unwrap_err()
            .to_string()
            .contains("IOException"));
    }
}
