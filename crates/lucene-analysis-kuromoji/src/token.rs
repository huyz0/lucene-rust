//! `org.apache.lucene.analysis.ja.Token`: a morph token with its word id
//! and the dictionary that knows its reading, base form and part of speech.

use std::sync::Arc;

use lucene_analysis::morph::{self, MorphToken, TokenType};

use crate::dict::JaDict;

/// `ja.Token`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Token {
    base: morph::Token,
    morph_id: i32,
    dict: JaDict,
}

impl Token {
    /// `new Token(surfaceForm, offset, length, startOffset, endOffset,
    /// morphId, type, morphData)`; the type is the dictionary's.
    pub fn new(
        surface_form: Arc<[u16]>,
        offset: i32,
        length: i32,
        start_offset: i32,
        end_offset: i32,
        morph_id: i32,
        dict: JaDict,
    ) -> Self {
        let token_type = dict.token_type();
        Token {
            base: morph::Token::new(
                surface_form,
                offset,
                length,
                start_offset,
                end_offset,
                token_type,
            ),
            morph_id,
            dict,
        }
    }

    /// The base token (surface, offsets, position length).
    pub fn base(&self) -> &morph::Token {
        &self.base
    }

    /// The word id.
    pub fn morph_id(&self) -> i32 {
        self.morph_id
    }

    /// The dictionary.
    pub fn dict(&self) -> &JaDict {
        &self.dict
    }

    /// `getReading()`.
    pub fn reading(&self) -> Option<String> {
        self.dict.reading(
            self.morph_id,
            &self.base.surface_form,
            self.base.offset,
            self.base.length,
        )
    }

    /// `getPronunciation()`.
    pub fn pronunciation(&self) -> Option<String> {
        self.dict.pronunciation(
            self.morph_id,
            &self.base.surface_form,
            self.base.offset,
            self.base.length,
        )
    }

    /// `getPartOfSpeech()`.
    pub fn part_of_speech(&self) -> Option<&str> {
        self.dict.part_of_speech(self.morph_id)
    }

    /// `getInflectionType()`.
    pub fn inflection_type(&self) -> Option<String> {
        self.dict.inflection_type(self.morph_id)
    }

    /// `getInflectionForm()`.
    pub fn inflection_form(&self) -> Option<String> {
        self.dict.inflection_form(self.morph_id)
    }

    /// `getBaseForm()`.
    pub fn base_form(&self) -> Option<String> {
        self.dict.base_form(
            self.morph_id,
            &self.base.surface_form,
            self.base.offset,
            self.base.length,
        )
    }

    /// `isKnown()`.
    pub fn is_known(&self) -> bool {
        self.base.token_type == TokenType::Known
    }

    /// `isUnknown()`.
    pub fn is_unknown(&self) -> bool {
        self.base.token_type == TokenType::Unknown
    }

    /// `isUser()`.
    pub fn is_user(&self) -> bool {
        self.base.token_type == TokenType::User
    }
}

impl std::fmt::Display for Token {
    /// `toString()`.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Token(\"{}\" offset={} length={} posLen={} type={} morphId={} leftID={})",
            self.base.surface_form_string(),
            self.base.start_offset,
            self.base.length,
            self.base.pos_len,
            self.base.token_type.name(),
            self.morph_id,
            self.dict.morph_data().left_id(self.morph_id)
        )
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
    use crate::dict::{TokenInfoDictionary, UnknownDictionary};

    #[test]
    fn token_accessors_and_display() {
        let frag: Arc<[u16]> = "東京".encode_utf16().collect();
        let known = TokenInfoDictionary::instance();
        let id = known.lookup_word_ids({
            let fst = known.fst();
            let mut arc = fst.first_arc();
            let mut out = 0i64;
            for (i, u) in "東京".encode_utf16().enumerate() {
                arc = fst
                    .find_target_arc(i32::from(u), &arc, i == 0)
                    .unwrap()
                    .unwrap();
                out += arc.output();
            }
            (out + arc.next_final_output()) as i32
        })[0];
        let t = Token::new(frag.clone(), 0, 2, 5, 7, id, JaDict::Known(known));
        assert!(t.is_known() && !t.is_unknown() && !t.is_user());
        assert_eq!(t.morph_id(), id);
        assert_eq!(t.reading().as_deref(), Some("トウキョウ"));
        assert!(t.part_of_speech().unwrap().starts_with("名詞"));
        assert_eq!(t.base_form(), None);
        assert!(t
            .to_string()
            .starts_with("Token(\"東京\" offset=5 length=2 posLen=1 type=KNOWN"));
        assert_eq!(t.dict().token_type(), TokenType::Known);
        let u = Token::new(
            frag,
            0,
            1,
            0,
            1,
            3,
            JaDict::Unknown(UnknownDictionary::instance()),
        );
        assert!(u.is_unknown());
        assert_eq!(
            (u.reading(), u.inflection_type(), u.inflection_form()),
            (None, None, None)
        );
        assert_ne!(t, u);
        assert_eq!(MorphToken::base(&u).length, 1);
    }
}
