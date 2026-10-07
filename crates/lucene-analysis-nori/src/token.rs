//! `org.apache.lucene.analysis.ko.{Token, DictionaryToken, DecompoundToken}`:
//! a token from a dictionary entry, or a morpheme of a decompounded one.

use std::sync::Arc;

use lucene_analysis::morph::{self, MorphToken, TokenType};

use crate::dict::{KoDict, Morpheme};
use crate::pos::{Tag, Type};

/// What the token is (`DictionaryToken` or `DecompoundToken`).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// `DictionaryToken`: a word of a dictionary.
    Dictionary {
        /// `wordId`.
        word_id: i32,
        /// `morphAtts`.
        dict: KoDict,
    },
    /// `DecompoundToken`: a morpheme of a compound or inflected word.
    Decompound {
        /// `posTag`.
        pos_tag: Tag,
    },
}

/// `ko.Token`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    base: morph::Token,
    kind: Kind,
}

impl Token {
    /// `new DictionaryToken(type, morphAtts, wordId, surfaceForm, offset,
    /// length, startOffset, endOffset)`; the type is the dictionary's.
    pub fn dictionary(
        dict: KoDict,
        word_id: i32,
        surface_form: Arc<[u16]>,
        offset: i32,
        length: i32,
        start_offset: i32,
        end_offset: i32,
    ) -> Self {
        let t = dict.token_type();
        Token {
            base: morph::Token::new(surface_form, offset, length, start_offset, end_offset, t),
            kind: Kind::Dictionary { word_id, dict },
        }
    }

    /// `new DecompoundToken(posTag, surfaceForm, startOffset, endOffset,
    /// type)`.
    pub fn decompound(
        pos_tag: Tag,
        surface_form: &str,
        start_offset: i32,
        end_offset: i32,
        t: TokenType,
    ) -> Self {
        let units: Arc<[u16]> = surface_form.encode_utf16().collect();
        let length = i32::try_from(units.len()).unwrap_or(i32::MAX);
        Token {
            base: morph::Token::new(units, 0, length, start_offset, end_offset, t),
            kind: Kind::Decompound { pos_tag },
        }
    }

    /// The base token.
    pub fn base(&self) -> &morph::Token {
        &self.base
    }

    /// The kind.
    pub fn kind(&self) -> &Kind {
        &self.kind
    }

    /// `getPOSType()`.
    pub fn pos_type(&self) -> Type {
        match &self.kind {
            Kind::Dictionary { word_id, dict } => dict.pos_type(*word_id),
            Kind::Decompound { .. } => Type::Morpheme,
        }
    }

    /// `getLeftPOS()`.
    pub fn left_pos(&self) -> Option<Tag> {
        match &self.kind {
            Kind::Dictionary { word_id, dict } => dict.left_pos(*word_id),
            Kind::Decompound { pos_tag } => Some(*pos_tag),
        }
    }

    /// `getRightPOS()`.
    pub fn right_pos(&self) -> Option<Tag> {
        match &self.kind {
            Kind::Dictionary { word_id, dict } => dict.right_pos(*word_id),
            Kind::Decompound { pos_tag } => Some(*pos_tag),
        }
    }

    /// `getReading()`.
    pub fn reading(&self) -> Option<String> {
        match &self.kind {
            Kind::Dictionary { word_id, dict } => dict.reading(*word_id),
            Kind::Decompound { .. } => None,
        }
    }

    /// `getMorphemes()`.
    pub fn morphemes(&self) -> Option<Vec<Morpheme>> {
        match &self.kind {
            Kind::Dictionary { word_id, dict } => dict.morphemes(
                *word_id,
                &self.base.surface_form,
                self.base.offset,
                self.base.length,
            ),
            Kind::Decompound { .. } => None,
        }
    }
}

impl std::fmt::Display for Token {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let b = &self.base;
        match &self.kind {
            Kind::Dictionary { word_id, dict } => write!(
                f,
                "DictionaryToken(\"{}\" pos={} length={} posLen={} type={} wordId={} leftID={})",
                b.surface_form_string(),
                b.start_offset,
                b.length,
                b.pos_len,
                b.token_type.name(),
                word_id,
                dict.morph_data().left_id(*word_id)
            ),
            Kind::Decompound { .. } => write!(
                f,
                "DecompoundToken(\"{}\" pos={} length={} startOffset={} endOffset={})",
                b.surface_form_string(),
                b.start_offset,
                b.length,
                b.start_offset,
                b.end_offset
            ),
        }
    }
}

impl MorphToken for Token {
    fn base(&self) -> &morph::Token {
        &self.base
    }
    fn base_mut(&mut self) -> &mut morph::Token {
        &mut self.base
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dict::{TokenInfoDictionary, UnknownDictionary, UserDictionary};

    #[test]
    fn decompound_and_dictionary_tokens() {
        let d = Token::decompound(Tag::Nng, "가방", 3, 5, TokenType::Known);
        assert_eq!(d.kind(), &Kind::Decompound { pos_tag: Tag::Nng });
        assert_eq!(d.pos_type(), Type::Morpheme);
        assert_eq!(
            (d.left_pos(), d.right_pos()),
            (Some(Tag::Nng), Some(Tag::Nng))
        );
        assert_eq!((d.reading(), d.morphemes()), (None, None));
        assert_eq!(MorphToken::base(&d).length, 2);
        assert_eq!(
            d.to_string(),
            "DecompoundToken(\"가방\" pos=3 length=2 startOffset=3 endOffset=5)"
        );
        let mut m = d.clone();
        MorphToken::base_mut(&mut m).pos_len = 2;
        assert_ne!(m, d);

        let known = TokenInfoDictionary::instance();
        let unknown = UnknownDictionary::instance();
        let user = Arc::new(UserDictionary::open("가방\n").unwrap().unwrap());
        let k = KoDict::Known(Arc::clone(&known));
        assert_eq!(k, KoDict::Known(known));
        assert_ne!(k, KoDict::Unknown(Arc::clone(&unknown)));
        let u = KoDict::Unknown(Arc::clone(&unknown));
        assert_eq!(u, KoDict::Unknown(unknown));
        let us = KoDict::User(Arc::clone(&user));
        assert_eq!(us, KoDict::User(user));
        assert_eq!(
            [&k, &u, &us].map(KoDict::token_type),
            [TokenType::Known, TokenType::Unknown, TokenType::User]
        );
        let surface: Arc<[u16]> = "가방".encode_utf16().collect();
        for dict in [u, us] {
            let t = Token::dictionary(dict.clone(), 0, Arc::clone(&surface), 0, 2, 0, 2);
            assert_eq!(t.base().token_type, dict.token_type());
            let _ = (
                t.pos_type(),
                t.left_pos(),
                t.right_pos(),
                t.reading(),
                t.morphemes(),
            );
            assert!(t
                .to_string()
                .starts_with("DictionaryToken(\"가방\" pos=0 length=2"));
            let _ = dict.morph_data().word_cost(0);
        }
    }
}
