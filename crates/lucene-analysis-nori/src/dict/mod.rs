//! `org.apache.lucene.analysis.ko.dict`: the mecab-ko-dic system
//! dictionary ([`TokenInfoDictionary`]), the unknown-word dictionary
//! ([`UnknownDictionary`]), [`CharacterDefinition`], [`ConnectionCosts`]
//! and user dictionaries ([`UserDictionary`]).
//!
//! The default instances (`getInstance()`) read the dictionary Lucene's jar
//! carries -- mecab-ko-dic-2.1.1-20180720 compiled by Lucene's builder --
//! vendored zlib-compressed under `src/resources/` (Apache-2.0; see
//! `docs/licences.md`); every class also loads a caller's files
//! (`from_paths`) or bytes (`read`). The builders are not ported: M12 loads
//! dictionaries, it does not build them.

pub mod character_definition;
pub mod connection_costs;
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

use crate::pos::{Tag, Type};

/// `DictionaryConstants.DICT_HEADER`.
pub const DICT_HEADER: &str = "ko_dict";
/// `DictionaryConstants.TARGETMAP_HEADER`.
pub const TARGETMAP_HEADER: &str = "ko_dict_map";
/// `DictionaryConstants.POSDICT_HEADER`.
pub const POSDICT_HEADER: &str = "ko_dict_pos";
/// `DictionaryConstants.CONN_COSTS_HEADER`.
pub const CONN_COSTS_HEADER: &str = "ko_cc";
/// `DictionaryConstants.CHARDEF_HEADER`.
pub const CHARDEF_HEADER: &str = "ko_cd";
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

/// `KoMorphData.Morpheme`: a part of a compound or inflected entry.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Morpheme {
    /// `posTag()`.
    pub pos_tag: Tag,
    /// `surfaceForm()`.
    pub surface_form: String,
}

/// The dictionary a token came from (`KoMorphData`): Java's
/// `dictionaryMap.get(type).getMorphAttributes()`.
#[derive(Debug, Clone)]
pub enum KoDict {
    /// The system dictionary (`TokenInfoMorphData`).
    Known(Arc<TokenInfoDictionary>),
    /// The unknown-word dictionary (`UnknownMorphData`).
    Unknown(Arc<UnknownDictionary>),
    /// A user dictionary (`UserMorphData`).
    User(Arc<UserDictionary>),
}

impl PartialEq for KoDict {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (KoDict::Known(a), KoDict::Known(b)) => Arc::ptr_eq(a, b),
            (KoDict::Unknown(a), KoDict::Unknown(b)) => Arc::ptr_eq(a, b),
            (KoDict::User(a), KoDict::User(b)) => Arc::ptr_eq(a, b),
            _ => false,
        }
    }
}

impl Eq for KoDict {}

impl KoDict {
    /// The token type of this dictionary.
    pub fn token_type(&self) -> TokenType {
        match self {
            KoDict::Known(_) => TokenType::Known,
            KoDict::Unknown(_) => TokenType::Unknown,
            KoDict::User(_) => TokenType::User,
        }
    }

    /// The connection ids and costs.
    pub fn morph_data(&self) -> &dyn MorphData {
        match self {
            KoDict::Known(d) => d.morph_attributes(),
            KoDict::Unknown(d) => d.morph_attributes(),
            KoDict::User(d) => d.morph_attributes(),
        }
    }

    /// `getPOSType(morphId)`.
    pub fn pos_type(&self, id: i32) -> Type {
        match self {
            KoDict::Known(d) => d.morph_attributes().pos_type(id),
            KoDict::Unknown(d) => d.morph_attributes().pos_type(id),
            KoDict::User(d) => d.morph_attributes().pos_type(id),
        }
    }

    /// `getLeftPOS(morphId)`.
    pub fn left_pos(&self, id: i32) -> Option<Tag> {
        match self {
            KoDict::Known(d) => d.morph_attributes().left_pos(id),
            KoDict::Unknown(d) => d.morph_attributes().left_pos(id),
            KoDict::User(_) => Some(Tag::Nng),
        }
    }

    /// `getRightPOS(morphId)`.
    pub fn right_pos(&self, id: i32) -> Option<Tag> {
        match self {
            KoDict::Known(d) => d.morph_attributes().right_pos(id),
            KoDict::Unknown(d) => d.morph_attributes().right_pos(id),
            KoDict::User(_) => Some(Tag::Nng),
        }
    }

    /// `getReading(morphId)`.
    pub fn reading(&self, id: i32) -> Option<String> {
        match self {
            KoDict::Known(d) => d.morph_attributes().reading(id),
            _ => None,
        }
    }

    /// `getMorphemes(morphId, surfaceForm, off, len)`.
    pub fn morphemes(&self, id: i32, surface: &[u16], off: i32, len: i32) -> Option<Vec<Morpheme>> {
        match self {
            KoDict::Known(d) => d.morph_attributes().morphemes(id, surface, off, len),
            KoDict::Unknown(_) => None,
            KoDict::User(d) => d.morph_attributes().morphemes(id, surface, off),
        }
    }
}
