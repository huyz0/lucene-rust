//! `org.apache.lucene.analysis.ko.Viterbi`: Nori's half of the Viterbi
//! search -- unknown words grouped by character class and Unicode script,
//! the space penalty of particles and endings after a space, and the
//! backtrace into tokens (compounds decompounded per [`DecompoundMode`],
//! unknown words as unigrams, punctuation and spaces dropped or kept).

use std::sync::Arc;

use lucene_analysis::java_character::{self as jc, get_type, is_digit, NON_SPACING_MARK};
use lucene_analysis::java_unicode_script::{unicode_script_name, unicode_script_of};
use lucene_analysis::morph::viterbi::{add, read_char, MAX_UNKNOWN_WORD_LENGTH};
use lucene_analysis::morph::{
    GraphvizFormatter, MorphData, MorphToken, TokenType, Viterbi, ViterbiLang,
};
use lucene_analysis::reader::CharReader;
use lucene_analysis::AnalysisError;

use crate::dict::character_definition::NGRAM;
use crate::dict::{
    CharacterDefinition, KoDict, TokenInfoDictionary, UnknownDictionary, UserDictionary,
};
use crate::pos::{Tag, Type};
use crate::token::Token;

/// `KoreanTokenizer.DecompoundMode`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DecompoundMode {
    /// `NONE`: compounds are kept whole.
    None,
    /// `DISCARD`: compounds are replaced by their parts
    /// (`DEFAULT_DECOMPOUND`).
    #[default]
    Discard,
    /// `MIXED`: the parts and the compound.
    Mixed,
}

impl DecompoundMode {
    /// `DecompoundMode.valueOf(name)`.
    pub fn value_of(name: &str) -> Result<Self, AnalysisError> {
        match name {
            "NONE" => Ok(DecompoundMode::None),
            "DISCARD" => Ok(DecompoundMode::Discard),
            "MIXED" => Ok(DecompoundMode::Mixed),
            _ => Err(AnalysisError::IllegalArgument(format!(
                "No enum constant org.apache.lucene.analysis.ko.KoreanTokenizer.DecompoundMode.{name}"
            ))),
        }
    }
}

/// `Viterbi.isPunctuation(ch, cid)`: Hangul Letter Araea and the
/// punctuation, separator, control and symbol categories.
pub(crate) fn is_punctuation(ch: u16, cid: u8) -> bool {
    // special case for Hangul Letter Araea (interpunct)
    if ch == 0x318D {
        return true;
    }
    matches!(
        cid,
        jc::SPACE_SEPARATOR
            | jc::LINE_SEPARATOR
            | jc::PARAGRAPH_SEPARATOR
            | jc::CONTROL
            | jc::FORMAT
            | jc::DASH_PUNCTUATION
            | jc::START_PUNCTUATION
            | jc::END_PUNCTUATION
            | jc::CONNECTOR_PUNCTUATION
            | jc::OTHER_PUNCTUATION
            | jc::MATH_SYMBOL
            | jc::CURRENCY_SYMBOL
            | jc::MODIFIER_SYMBOL
            | jc::OTHER_SYMBOL
            | jc::INITIAL_QUOTE_PUNCTUATION
            | jc::FINAL_QUOTE_PUNCTUATION
    )
}

fn is_common_or_inherited(script: u8) -> bool {
    matches!(unicode_script_name(script), "INHERITED" | "COMMON")
}

/// `isSameScript(one, two)`.
fn is_same_script(one: u8, two: u8) -> bool {
    one == two || is_common_or_inherited(one) || is_common_or_inherited(two)
}

/// Nori's language state of the search.
pub struct KoViterbi {
    known: Arc<TokenInfoDictionary>,
    unknown: Arc<UnknownDictionary>,
    user: Option<Arc<UserDictionary>>,
    character_definition: Arc<CharacterDefinition>,
    discard_punctuation: bool,
    mode: DecompoundMode,
    output_unknown_unigrams: bool,
    pub(crate) dot_out: Option<GraphvizFormatter>,
}

fn idx(i: i32) -> usize {
    usize::try_from(i).unwrap_or(usize::MAX)
}

impl KoViterbi {
    /// `new Viterbi(...)`'s language half.
    pub(crate) fn new(
        known: Arc<TokenInfoDictionary>,
        unknown: Arc<UnknownDictionary>,
        user: Option<Arc<UserDictionary>>,
        discard_punctuation: bool,
        mode: DecompoundMode,
        output_unknown_unigrams: bool,
    ) -> Self {
        let character_definition = Arc::clone(unknown.character_definition());
        KoViterbi {
            known,
            unknown,
            user,
            character_definition,
            discard_punctuation,
            mode,
            output_unknown_unigrams,
            dot_out: None,
        }
    }

    /// `getDict(type)`.
    fn dict(&self, t: TokenType) -> KoDict {
        match (t, &self.user) {
            (TokenType::User, Some(u)) => KoDict::User(Arc::clone(u)),
            (TokenType::Unknown, _) => KoDict::Unknown(Arc::clone(&self.unknown)),
            _ => KoDict::Known(Arc::clone(&self.known)),
        }
    }

    /// `shouldFilterToken(token)`.
    fn should_filter_token(&self, token: &Token) -> bool {
        let first = token.base().surface().first().copied().unwrap_or(0);
        self.discard_punctuation && is_punctuation(first, get_type(u32::from(first)))
    }
}

impl ViterbiLang<Token> for KoViterbi {
    fn morph_data(&self, t: TokenType) -> &dyn MorphData {
        match (t, &self.user) {
            (TokenType::User, Some(u)) => u.morph_attributes(),
            (TokenType::Unknown, _) => self.unknown.morph_attributes(),
            _ => self.known.morph_attributes(),
        }
    }

    fn known_word_ids(&self, source_id: i32) -> &[i32] {
        self.known.lookup_word_ids(source_id)
    }

    /// `computeSpacePenalty(morphData, wordID, numSpaces)`: 3000 for an
    /// ending, particle, copula or suffix after a space.
    fn compute_space_penalty(&self, t: TokenType, word_id: i32, num_spaces: i32) -> i32 {
        if num_spaces <= 0 {
            return 0;
        }
        let left_pos = match (t, &self.user) {
            (TokenType::User, Some(_)) => Some(Tag::Nng),
            (TokenType::Unknown, _) => self.unknown.morph_attributes().left_pos(word_id),
            _ => self.known.morph_attributes().left_pos(word_id),
        };
        match left_pos {
            Some(
                Tag::Ep
                | Tag::Ef
                | Tag::Ec
                | Tag::Etn
                | Tag::Etm
                | Tag::Jks
                | Tag::Jkc
                | Tag::Jkg
                | Tag::Jko
                | Tag::Jkb
                | Tag::Jkv
                | Tag::Jkq
                | Tag::Jx
                | Tag::Jc
                | Tag::Vcp
                | Tag::Xsa
                | Tag::Xsn
                | Tag::Xsv,
            ) => 3000,
            _ => 0,
        }
    }

    fn process_unknown_word(
        &mut self,
        v: &mut Viterbi<Token>,
        reader: &mut dyn CharReader,
        any_matches: bool,
        pos_data: i32,
    ) -> Result<i32, AnalysisError> {
        let first = read_char(v, reader, v.pos)? as u16;
        let cd = Arc::clone(&self.character_definition);
        if any_matches && !cd.is_invoke(first) {
            return Ok(0);
        }
        // Find unknown match:
        let mut character_id = cd.character_class(first);
        // NOTE: copied from UnknownDictionary.lookup:
        let mut unknown_word_length: i32 = 1;
        if cd.is_group(first) {
            // Extract unknown word. Characters with the same script are
            // considered to be part of unknown word
            let mut script_code = unicode_script_of(first);
            let is_punct = is_punctuation(first, get_type(u32::from(first)));
            let first_is_digit = is_digit(u32::from(first));
            let mut pos_ahead = v.pos.wrapping_add(1);
            while unknown_word_length < MAX_UNKNOWN_WORD_LENGTH {
                let next = read_char(v, reader, pos_ahead)?;
                if next == -1 {
                    break;
                }
                let ch = next as u16;
                let ch_type = get_type(u32::from(ch));
                let sc = unicode_script_of(ch);
                // Non-spacing marks inherit the script of their base
                // character, following recommendations from UTR #24.
                let same_script = is_same_script(script_code, sc) || ch_type == NON_SPACING_MARK;
                if same_script
                    && is_punctuation(ch, ch_type) == is_punct
                    && is_digit(u32::from(ch)) == first_is_digit
                    && cd.is_group(ch)
                {
                    unknown_word_length = unknown_word_length.wrapping_add(1);
                } else {
                    break;
                }
                // Update the script code and character class if the
                // original script is Inherited or Common.
                if is_common_or_inherited(script_code) && !is_common_or_inherited(sc) {
                    script_code = sc;
                    character_id = cd.character_class(ch);
                }
                pos_ahead = pos_ahead.wrapping_add(1);
            }
        }
        // characters in input text are supposed to be the same
        let unknown = Arc::clone(&self.unknown);
        for &word_id in unknown.lookup_word_ids(i32::from(character_id)) {
            let word_pos = v.pos;
            add(
                v,
                &*self,
                TokenType::Unknown,
                pos_data,
                word_pos,
                word_pos.wrapping_add(unknown_word_length),
                word_id,
                false,
            );
        }
        // TODO (Java): should return meaningful value?
        Ok(0)
    }

    fn backtrace(
        &mut self,
        v: &mut Viterbi<Token>,
        end_pos: i32,
        from_idx: i32,
    ) -> Result<(), AnalysisError> {
        let last = v.last_back_trace_pos;
        if end_pos == last {
            return Ok(());
        }
        let fragment = v.fragment(last, end_pos.wrapping_sub(last));

        if let Some(mut dot) = self.dot_out.take() {
            let r = dot.on_backtrace(
                &*self,
                &v.positions,
                last,
                end_pos,
                from_idx,
                &fragment,
                v.end,
            );
            self.dot_out = Some(dot);
            r?;
        }

        let mut pos = end_pos;
        let mut best_idx = from_idx;
        while pos > last {
            let b = idx(best_idx);
            // The back pointer, copied out (not the whole position).
            let (back_pos, back_word_pos, back_type, back_id, next_best_idx) = {
                let p = v.positions.at(pos);
                (
                    p.back_pos(b)?,
                    p.back_word_pos(b)?,
                    p.back_type(b)?,
                    p.back_id(b)?,
                    p.back_index(b)?,
                )
            };
            // the length of the word without the whitespaces at the
            // beginning.
            let length = pos.wrapping_sub(back_word_pos);
            // the start of the word after the whitespace at the beginning.
            let fragment_offset = back_word_pos.wrapping_sub(last);

            if self.output_unknown_unigrams && back_type == TokenType::Unknown {
                // outputUnknownUnigrams converts unknown word into unigrams:
                let mut i = length.wrapping_sub(1);
                while i >= 0 {
                    let mut char_len: i32 = 1;
                    if i > 0
                        && fragment
                            .get(idx(fragment_offset.wrapping_add(i)))
                            .is_some_and(|&u| jc::is_low_surrogate(u))
                    {
                        i = i.wrapping_sub(1);
                        char_len = 2;
                    }
                    v.pending.push(Token::dictionary(
                        KoDict::Unknown(Arc::clone(&self.unknown)),
                        i32::from(NGRAM),
                        Arc::clone(&fragment),
                        fragment_offset.wrapping_add(i),
                        char_len,
                        back_word_pos.wrapping_add(i),
                        back_word_pos.wrapping_add(i).wrapping_add(char_len),
                    ));
                    i = i.wrapping_sub(1);
                }
            } else {
                let mut token = Token::dictionary(
                    self.dict(back_type),
                    back_id,
                    Arc::clone(&fragment),
                    fragment_offset,
                    length,
                    back_word_pos,
                    back_word_pos.wrapping_add(length),
                );
                if token.pos_type() == Type::Morpheme || self.mode == DecompoundMode::None {
                    if !self.should_filter_token(&token) {
                        v.pending.push(token);
                    }
                } else {
                    match token.morphemes() {
                        None => v.pending.push(token),
                        Some(morphemes) => {
                            let mut end_offset = back_word_pos.wrapping_add(length);
                            let mut pos_len: i32 = 0;
                            // decompose the compound
                            for (i, morpheme) in morphemes.iter().enumerate().rev() {
                                let m_len =
                                    i32::try_from(morpheme.surface_form.encode_utf16().count())
                                        .unwrap_or(i32::MAX);
                                let mut compound_token = if token.pos_type() == Type::Compound {
                                    Token::decompound(
                                        morpheme.pos_tag,
                                        &morpheme.surface_form,
                                        end_offset.wrapping_sub(m_len),
                                        end_offset,
                                        back_type,
                                    )
                                } else {
                                    Token::decompound(
                                        morpheme.pos_tag,
                                        &morpheme.surface_form,
                                        token.base().start_offset,
                                        token.base().end_offset,
                                        back_type,
                                    )
                                };
                                if i == 0 && self.mode == DecompoundMode::Mixed {
                                    MorphToken::base_mut(&mut compound_token).pos_incr = 0;
                                }
                                pos_len = pos_len.wrapping_add(1);
                                end_offset = end_offset.wrapping_sub(m_len);
                                v.pending.push(compound_token);
                            }
                            if self.mode == DecompoundMode::Mixed {
                                MorphToken::base_mut(&mut token).pos_len = pos_len.max(1);
                                v.pending.push(token);
                            }
                        }
                    }
                }
            }
            if !self.discard_punctuation && back_word_pos != back_pos {
                // Add a token for whitespaces between terms
                let offset = back_pos.wrapping_sub(last);
                let len = back_word_pos.wrapping_sub(back_pos);
                let space = self.character_definition.character_class(u16::from(b' '));
                let word_id = self
                    .unknown
                    .lookup_word_ids(i32::from(space))
                    .first()
                    .copied()
                    .ok_or_else(|| {
                        lucene_analysis::morph::resource::io_error(
                            "ArrayIndexOutOfBoundsException",
                            0,
                        )
                    })?;
                v.pending.push(Token::dictionary(
                    KoDict::Unknown(Arc::clone(&self.unknown)),
                    word_id,
                    Arc::clone(&fragment),
                    offset,
                    len,
                    back_pos,
                    back_pos.wrapping_add(len),
                ));
            }
            pos = back_pos;
            best_idx = next_best_idx;
        }

        v.last_back_trace_pos = end_pos;
        // Notify the circular buffers that we are done with these
        // positions:
        v.buffer.free_before(end_pos);
        v.positions.free_before(end_pos);
        Ok(())
    }
}
