//! `org.apache.lucene.analysis.ja.ViterbiNBest`: Kuromoji's half of the
//! Viterbi search -- unknown words by character class, the search-mode
//! penalty and 2nd-best decompounding, the backtrace into tokens (user
//! phrases expanded, extended-mode unigrams, punctuation dropped) and the
//! n-best nodes.

use std::sync::Arc;

use lucene_analysis::java_character::{self as jc, get_type};
use lucene_analysis::morph::viterbi::{add, read_char, MAX_UNKNOWN_WORD_LENGTH};
use lucene_analysis::morph::viterbi_nbest::{self, NBestLang, NBestState};
use lucene_analysis::morph::{
    GraphvizFormatter, Lattice, MorphData, MorphToken, TokenType, Viterbi, ViterbiLang,
};
use lucene_analysis::reader::CharReader;
use lucene_analysis::AnalysisError;

use crate::dict::character_definition::NGRAM;
use crate::dict::{
    CharacterDefinition, JaDict, TokenInfoDictionary, UnknownDictionary, UserDictionary,
};
use crate::token::Token;

const SEARCH_MODE_KANJI_LENGTH: i32 = 2;
const SEARCH_MODE_OTHER_LENGTH: i32 = 7; // Must be >= SEARCH_MODE_KANJI_LENGTH
const SEARCH_MODE_KANJI_PENALTY: i32 = 3000;
const SEARCH_MODE_OTHER_PENALTY: i32 = 1700;

/// `ViterbiNBest.isPunctuation(char)`.
pub(crate) fn is_punctuation(ch: u16) -> bool {
    matches!(
        get_type(u32::from(ch)),
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

/// Kuromoji's language state of the search.
pub struct JaViterbi {
    known: Arc<TokenInfoDictionary>,
    unknown: Arc<UnknownDictionary>,
    user: Option<Arc<UserDictionary>>,
    character_definition: Arc<CharacterDefinition>,
    discard_punctuation: bool,
    search_mode: bool,
    extended_mode: bool,
    output_compounds: bool,
    pub(crate) nbest: NBestState,
    pub(crate) dot_out: Option<GraphvizFormatter>,
    /// `pruneAndRescore`'s arcs, kept between calls.
    kept: Vec<(i32, i32, i32, TokenType)>,
    forwards: Vec<(i32, i32, TokenType)>,
}

fn idx(i: i32) -> usize {
    usize::try_from(i).unwrap_or(usize::MAX)
}

impl JaViterbi {
    /// `new ViterbiNBest(...)`'s language half.
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
        known: Arc<TokenInfoDictionary>,
        unknown: Arc<UnknownDictionary>,
        user: Option<Arc<UserDictionary>>,
        discard_punctuation: bool,
        search_mode: bool,
        extended_mode: bool,
        output_compounds: bool,
    ) -> Self {
        let character_definition = Arc::clone(unknown.character_definition());
        JaViterbi {
            known,
            unknown,
            user,
            character_definition,
            discard_punctuation,
            search_mode,
            extended_mode,
            output_compounds,
            nbest: NBestState::default(),
            dot_out: None,
            kept: Vec::new(),
            forwards: Vec::new(),
        }
    }

    /// `getDict(type)`.
    fn dict(&self, t: TokenType) -> JaDict {
        match (t, &self.user) {
            (TokenType::User, Some(u)) => JaDict::User(Arc::clone(u)),
            (TokenType::Unknown, _) => JaDict::Unknown(Arc::clone(&self.unknown)),
            _ => JaDict::Known(Arc::clone(&self.known)),
        }
    }

    /// `computeSecondBestThreshold(pos, length)`.
    fn compute_second_best_threshold(&self, v: &Viterbi<Token>, pos: i32, length: i32) -> i32 {
        self.compute_penalty(v, pos, length)
    }

    /// `pruneAndRescore(startPos, endPos, bestStartIDX)`: drops the arcs
    /// that are compound tokens or cross `start_pos`, and rescores the rest.
    fn prune_and_rescore(
        &mut self,
        v: &mut Viterbi<Token>,
        start_pos: i32,
        end_pos: i32,
        best_start_idx: i32,
    ) -> Result<(), AnalysisError> {
        let (mut kept, mut forwards) = (
            std::mem::take(&mut self.kept),
            std::mem::take(&mut self.forwards),
        );
        let r = self.prune_with(
            v,
            start_pos,
            end_pos,
            best_start_idx,
            &mut kept,
            &mut forwards,
        );
        (self.kept, self.forwards) = (kept, forwards);
        r
    }

    /// [`Self::prune_and_rescore`] with its scratch vectors.
    fn prune_with(
        &self,
        v: &mut Viterbi<Token>,
        start_pos: i32,
        end_pos: i32,
        best_start_idx: i32,
        kept: &mut Vec<(i32, i32, i32, TokenType)>,
        forwards: &mut Vec<(i32, i32, TokenType)>,
    ) -> Result<(), AnalysisError> {
        // First pass: walk backwards, building up the forward arcs and
        // pruning inadmissible arcs:
        // The arcs to keep, gathered before the positions they name are
        // touched (one scratch vector for the call, not a copy of every
        // position).
        let mut pos = end_pos;
        while pos > start_pos {
            kept.clear();
            let p = v.positions.get(pos);
            for arc_idx in 0..p.count() {
                let back_pos = p.back_pos(arc_idx)?;
                if back_pos >= start_pos {
                    // Keep this arc:
                    kept.push((
                        back_pos,
                        i32::try_from(arc_idx).unwrap_or(i32::MAX),
                        p.back_id(arc_idx)?,
                        p.back_type(arc_idx)?,
                    ));
                }
            }
            for &(back_pos, arc_idx, id, t) in kept.iter() {
                v.positions.get(back_pos).add_forward(pos, arc_idx, id, t);
            }
            v.positions.get(pos).set_count(0);
            pos = pos.wrapping_sub(1);
        }

        // Second pass: walk forward, re-scoring:
        let mut pos = start_pos;
        while pos < end_pos {
            let p = v.positions.get(pos);
            if p.count() == 0 {
                // No arcs arrive here...
                v.positions.get(pos).clear_forwards();
                pos = pos.wrapping_add(1);
                continue;
            }
            if pos == start_pos {
                // On the initial position, only consider the best path so
                // we "force congruence": the sub-segmentation is "in
                // context" of what the best path (compound token) had
                // matched:
                let right_id = if start_pos == 0 {
                    0
                } else {
                    let b = idx(best_start_idx);
                    self.morph_data(p.back_type(b)?).right_id(p.back_id(b)?)
                };
                let path_cost = p.cost(idx(best_start_idx))?;
                forwards.clear();
                for f in 0..p.forward_count() {
                    forwards.push(p.forward(f)?);
                }
                for &(to_pos, word_id, forward_type) in forwards.iter() {
                    let dict2 = self.morph_data(forward_type);
                    let new_cost = path_cost
                        .wrapping_add(dict2.word_cost(word_id))
                        .wrapping_add(v.costs.get(right_id, dict2.left_id(word_id)))
                        .wrapping_add(self.compute_penalty(v, pos, to_pos.wrapping_sub(pos)));
                    let right = dict2.right_id(word_id);
                    v.positions.get(to_pos).add(
                        new_cost,
                        right,
                        pos,
                        -1,
                        best_start_idx,
                        word_id,
                        forward_type,
                    );
                }
            } else {
                // On non-initial positions, we maximize score across all
                // arriving lastRightIDs:
                forwards.clear();
                for f in 0..p.forward_count() {
                    forwards.push(p.forward(f)?);
                }
                for &(to_pos, word_id, forward_type) in forwards.iter() {
                    add(v, self, forward_type, pos, pos, to_pos, word_id, true);
                }
            }
            v.positions.get(pos).clear_forwards();
            pos = pos.wrapping_add(1);
        }
        Ok(())
    }
}

impl ViterbiLang<Token> for JaViterbi {
    fn morph_data(&self, t: TokenType) -> &dyn MorphData {
        match (t, &self.user) {
            (TokenType::User, Some(u)) => u.morph_attributes(),
            (TokenType::Unknown, _) => self.unknown.morph_attributes(),
            _ => self.known.morph_attributes(),
        }
    }

    #[inline]
    fn connection(&self, t: TokenType, word_id: i32) -> (i32, i32, i32) {
        match (t, &self.user) {
            (TokenType::User, Some(u)) => u.morph_attributes().connection(word_id),
            (TokenType::Unknown, _) => self.unknown.morph_attributes().connection(word_id),
            _ => self.known.morph_attributes().connection(word_id),
        }
    }

    fn known_word_ids(&self, source_id: i32) -> &[i32] {
        self.known.lookup_word_ids(source_id)
    }

    fn should_skip_process_unknown_word(&self, unknown_word_end_index: i32, pos_data: i32) -> bool {
        !self.search_mode && unknown_word_end_index > pos_data
    }

    /// `computePenalty(pos, length)`.
    fn compute_penalty(&self, v: &Viterbi<Token>, pos: i32, length: i32) -> i32 {
        if length > SEARCH_MODE_KANJI_LENGTH {
            let end_pos = pos.wrapping_add(length);
            // check if node consists of only kanji
            let all_kanji =
                (pos..end_pos).all(|p| self.character_definition.is_kanji(v.char_at(p) as u16));
            if all_kanji {
                // Process only Kanji keywords
                return length
                    .wrapping_sub(SEARCH_MODE_KANJI_LENGTH)
                    .wrapping_mul(SEARCH_MODE_KANJI_PENALTY);
            } else if length > SEARCH_MODE_OTHER_LENGTH {
                return length
                    .wrapping_sub(SEARCH_MODE_OTHER_LENGTH)
                    .wrapping_mul(SEARCH_MODE_OTHER_PENALTY);
            }
        }
        0
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
        let character_id = cd.character_class(first);
        let is_punct = is_punctuation(first);

        // NOTE: copied from UnknownDictionary.lookup:
        let mut unknown_word_length: i32 = 1;
        if cd.is_group(first) {
            // Extract unknown word. Characters with the same character
            // class are considered to be part of unknown word
            let mut pos_ahead = v.pos.wrapping_add(1);
            while unknown_word_length < MAX_UNKNOWN_WORD_LENGTH {
                let ch = read_char(v, reader, pos_ahead)?;
                if ch == -1 {
                    break;
                }
                let ch = ch as u16;
                if character_id == cd.character_class(ch) && is_punctuation(ch) == is_punct {
                    unknown_word_length = unknown_word_length.wrapping_add(1);
                } else {
                    break;
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
                pos_data.wrapping_add(unknown_word_length),
                word_id,
                false,
            );
        }
        Ok(unknown_word_length)
    }

    fn backtrace(
        &mut self,
        v: &mut Viterbi<Token>,
        end_pos: i32,
        from_idx: i32,
    ) -> Result<(), AnalysisError> {
        // LUCENE-10059: If the endPos is the same as lastBackTracePos, we
        // don't want to backtrace.
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
        let mut alt_token: Option<Token> = None;
        // We trace backwards, so this will be the leftWordID of the token
        // after the one we are now on:
        let mut last_left_word_id: i32 = -1;
        let mut back_count: i32 = 0;

        while pos > last {
            // The one back pointer read, copied out (not the position:
            // cloning it allocated per token).
            let b = idx(best_idx);
            let (best_cost, mut back_pos, mut back_type, mut back_id, best_back_index) = {
                let p = v.positions.at(pos);
                (
                    p.cost(b)?,
                    p.back_pos(b)?,
                    p.back_type(b)?,
                    p.back_id(b)?,
                    p.back_index(b)?,
                )
            };
            let mut length = pos.wrapping_sub(back_pos);
            let mut next_best_idx = best_back_index;

            if self.search_mode && alt_token.is_none() && back_type != TokenType::User {
                // In searchMode, if best path had picked a too-long token,
                // we use the "penalty" to compute the allowed max cost of an
                // alternate back-trace. If we find an alternate back trace
                // with cost below that threshold, we pursue it instead (but
                // also output the long token).
                let penalty =
                    self.compute_second_best_threshold(v, back_pos, pos.wrapping_sub(back_pos));
                if penalty > 0 {
                    // Use the penalty to set maxCost on the 2nd best
                    // segmentation:
                    let mut max_cost = best_cost.wrapping_add(penalty);
                    if last_left_word_id != -1 {
                        max_cost = max_cost.wrapping_add(v.costs.get(
                            self.morph_data(back_type).right_id(back_id),
                            last_left_word_id,
                        ));
                    }
                    // Now, prune all too-long tokens from the graph:
                    self.prune_and_rescore(v, back_pos, pos, best_back_index)?;

                    // Finally, find 2nd best back-trace and resume
                    // backtrace there:
                    let p2 = v.positions.at(pos).clone();
                    let mut least_cost = i32::MAX;
                    let mut least_idx: i32 = -1;
                    for i in 0..p2.count() {
                        let mut cost = p2.cost(i)?;
                        if last_left_word_id != -1 {
                            cost = cost.wrapping_add(v.costs.get(
                                self.morph_data(p2.back_type(i)?).right_id(p2.back_id(i)?),
                                last_left_word_id,
                            ));
                        }
                        if cost < least_cost {
                            least_cost = cost;
                            least_idx = i32::try_from(i).unwrap_or(i32::MAX);
                        }
                    }
                    if least_idx != -1
                        && least_cost <= max_cost
                        && p2.back_pos(idx(least_idx))? != back_pos
                    {
                        // Save the current compound token, to output when
                        // this alternate path joins back:
                        alt_token = Some(Token::new(
                            Arc::clone(&fragment),
                            back_pos.wrapping_sub(last),
                            length,
                            back_pos,
                            back_pos.wrapping_add(length),
                            back_id,
                            self.dict(back_type),
                        ));
                        // Redirect our backtrace to 2nd best:
                        let l = idx(least_idx);
                        next_best_idx = p2.back_index(l)?;
                        back_pos = p2.back_pos(l)?;
                        length = pos.wrapping_sub(back_pos);
                        back_type = p2.back_type(l)?;
                        back_id = p2.back_id(l)?;
                        back_count = 0;
                    }
                    // else: in theory there may be no 2nd best path; then
                    // only the compound token is output.
                }
            }

            let offset = back_pos.wrapping_sub(last);

            if alt_token
                .as_ref()
                .is_some_and(|t| t.base().start_offset >= back_pos)
            {
                if self.output_compounds {
                    // We've backtraced to the position where the compound
                    // token starts; add it now:
                    if back_count > 0 {
                        back_count = back_count.wrapping_add(1);
                        if let Some(mut t) = alt_token.take() {
                            MorphToken::base_mut(&mut t).pos_len = back_count;
                            v.pending.push(t);
                        }
                    }
                    // else: the alt token was all punctuation tokens.
                }
                alt_token = None;
            }

            let dict = self.dict(back_type);

            if back_type == TokenType::User {
                // Expand the phraseID we recorded into the actual
                // segmentation:
                let user = self.user.clone();
                let segmentation = user
                    .as_deref()
                    .map_or(&[][..], |u| u.lookup_segmentation(back_id));
                if let Some((&word_id, lengths)) = segmentation.split_first() {
                    let mut current: i32 = 0;
                    let first_new = v.pending.len();
                    for (j, &len) in lengths.iter().enumerate() {
                        let start_offset = current.wrapping_add(back_pos);
                        v.pending.push(Token::new(
                            Arc::clone(&fragment),
                            current.wrapping_add(offset),
                            len,
                            start_offset,
                            start_offset.wrapping_add(len),
                            word_id.wrapping_add(i32::try_from(j).unwrap_or(i32::MAX)),
                            dict.clone(),
                        ));
                        current = current.wrapping_add(len);
                    }
                    // Reverse the tokens we just added, because when we
                    // serve them up from incrementToken we serve in reverse:
                    v.pending[first_new..].reverse();
                    back_count =
                        back_count.wrapping_add(i32::try_from(lengths.len()).unwrap_or(i32::MAX));
                }
            } else if self.extended_mode && back_type == TokenType::Unknown {
                // In EXTENDED mode we convert unknown word into unigrams:
                let mut unigram_token_count: i32 = 0;
                let mut i = length.wrapping_sub(1);
                while i >= 0 {
                    let mut char_len: i32 = 1;
                    if i > 0
                        && fragment
                            .get(idx(offset.wrapping_add(i)))
                            .is_some_and(|&u| jc::is_low_surrogate(u))
                    {
                        i = i.wrapping_sub(1);
                        char_len = 2;
                    }
                    let unit = fragment
                        .get(idx(offset.wrapping_add(i)))
                        .copied()
                        .unwrap_or(0);
                    if !self.discard_punctuation || !is_punctuation(unit) {
                        let start_offset = back_pos.wrapping_add(i);
                        v.pending.push(Token::new(
                            Arc::clone(&fragment),
                            offset.wrapping_add(i),
                            char_len,
                            start_offset,
                            start_offset.wrapping_add(char_len),
                            i32::from(NGRAM),
                            JaDict::Unknown(Arc::clone(&self.unknown)),
                        ));
                        unigram_token_count = unigram_token_count.wrapping_add(1);
                    }
                    i = i.wrapping_sub(1);
                }
                back_count = back_count.wrapping_add(unigram_token_count);
            } else if !self.discard_punctuation
                || length == 0
                || !is_punctuation(fragment.get(idx(offset)).copied().unwrap_or(0))
            {
                v.pending.push(Token::new(
                    Arc::clone(&fragment),
                    offset,
                    length,
                    back_pos,
                    back_pos.wrapping_add(length),
                    back_id,
                    dict.clone(),
                ));
                back_count = back_count.wrapping_add(1);
            }
            // else: skip punctuation token

            last_left_word_id = dict.morph_data().left_id(back_id);
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

    fn backtrace_nbest(
        &mut self,
        v: &mut Viterbi<Token>,
        end_pos: i32,
        use_eos: bool,
    ) -> Result<(), AnalysisError> {
        viterbi_nbest::backtrace_nbest(v, self, end_pos, use_eos)
    }

    fn fixup_pending_list(&mut self, v: &mut Viterbi<Token>) -> Result<(), AnalysisError> {
        viterbi_nbest::fixup_pending_list(&mut v.pending);
        Ok(())
    }
}

impl NBestLang<Token> for JaViterbi {
    fn nbest_state(&mut self) -> &mut NBestState {
        &mut self.nbest
    }

    /// `registerNode(node, fragment)`.
    fn register_node(
        &self,
        v: &mut Viterbi<Token>,
        lattice: &Lattice,
        node: usize,
        fragment: &Arc<[u16]>,
    ) -> Result<(), AnalysisError> {
        let left = lattice.node_left(node);
        let right = lattice.node_right(node);
        let t = lattice.node_dic_type(node);
        let root = lattice.root_base();
        let first = fragment.get(idx(left)).copied().unwrap_or(0);
        if self.discard_punctuation && is_punctuation(first) {
            return Ok(());
        }
        if let (TokenType::User, Some(user)) = (t, &self.user) {
            // Expand the phraseID we recorded into the actual segmentation:
            let segmentation = user.lookup_segmentation(lattice.node_word_id(node));
            let Some((&word_id, lengths)) = segmentation.split_first() else {
                return Ok(());
            };
            let dict = JaDict::User(Arc::clone(user));
            v.pending.push(Token::new(
                Arc::clone(fragment),
                left,
                right.wrapping_sub(left),
                root.wrapping_add(left),
                root.wrapping_add(right),
                word_id,
                dict.clone(),
            ));
            // Output compound
            let mut current: i32 = 0;
            for (j, &len) in lengths.iter().enumerate() {
                if len < right.wrapping_sub(left) {
                    let start_offset = root.wrapping_add(current).wrapping_add(left);
                    v.pending.push(Token::new(
                        Arc::clone(fragment),
                        current.wrapping_add(left),
                        len,
                        start_offset,
                        start_offset.wrapping_add(len),
                        word_id.wrapping_add(i32::try_from(j).unwrap_or(i32::MAX)),
                        dict.clone(),
                    ));
                }
                current = current.wrapping_add(len);
            }
        } else {
            v.pending.push(Token::new(
                Arc::clone(fragment),
                left,
                right.wrapping_sub(left),
                root.wrapping_add(left),
                root.wrapping_add(right),
                lattice.node_word_id(node),
                self.dict(t),
            ));
        }
        Ok(())
    }
}
