//! `org.apache.lucene.analysis.morph.Viterbi`: the forward pass over the
//! lattice of dictionary words (user, known, unknown) and the backtrace
//! hooks a language implements.
//!
//! Java's abstract class and its subclass are split here into the state
//! ([`Viterbi`]: the rolling buffer, the positions, the pending tokens) and
//! the language ([`ViterbiLang`]: `processUnknownWord`, `backtrace`, the
//! penalties). [`forward`] and [`add`] are Java's final methods, taking
//! both.
//!
//! Differs: a [`Position`] holds its back pointers as two vectors of
//! records rather than seven parallel arrays -- the path cost and right id
//! that [`add`]'s least-cost scan reads, and the rest (Java's
//! `ArrayUtil.grow` sizes are not observable); positions are named by their
//! absolute position rather than held by reference across a call that may
//! grow the array, and [`add`] reads the connection costs of the word's
//! left id as one row. Costs are Java `int`s: sums wrap (`wrapping_add`) as
//! Java's do.

use std::sync::Arc;

use super::connection_costs::ConnectionCosts;
use super::resource::io_error;
use super::token::{MorphData, TokenType};
use super::token_info_fst::TokenInfoFst;
use crate::charfilter::RollingCharBuffer;
use crate::java_character::{get_type, SPACE_SEPARATOR};
use crate::reader::CharReader;
use crate::AnalysisError;

/// `Viterbi.MAX_UNKNOWN_WORD_LENGTH`.
pub const MAX_UNKNOWN_WORD_LENGTH: i32 = 1024;
/// `Viterbi.MAX_BACKTRACE_GAP`.
const MAX_BACKTRACE_GAP: i32 = 1024;

/// What [`add`] reads of a [`Position`]'s back pointer, per arriving arc:
/// the path cost and the right id (Java's `costs` and `lastRightID`
/// arrays), apart from the rest so the least-cost scan walks 8 bytes an
/// arc.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Arrival {
    cost: i32,
    last_right_id: i32,
}

/// The rest of a back pointer (Java's other parallel arrays, one record).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Back {
    back_pos: i32,
    back_word_pos: i32,
    back_index: i32,
    back_id: i32,
    back_type: TokenType,
}

/// One forward pointer (`ViterbiNBest.PositionNBest`'s arrays).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct Forward {
    pos: i32,
    index: i32,
    id: i32,
    forward_type: TokenType,
}

/// `Viterbi.Position` (with `ViterbiNBest.PositionNBest`'s forward
/// pointers): every back pointer arriving at one position.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Position {
    pos: i32,
    /// `arrivals[i]` and `backs[i]` are back pointer `i`.
    arrivals: Vec<Arrival>,
    backs: Vec<Back>,
    forwards: Vec<Forward>,
}

fn bad_index(i: usize) -> AnalysisError {
    io_error("ArrayIndexOutOfBoundsException", i)
}

impl Position {
    /// `add(cost, lastRightID, backPos, backRPos, backIndex, backID,
    /// backType)`.
    #[allow(clippy::too_many_arguments)]
    #[inline]
    pub fn add(
        &mut self,
        cost: i32,
        last_right_id: i32,
        back_pos: i32,
        back_word_pos: i32,
        back_index: i32,
        back_id: i32,
        back_type: TokenType,
    ) {
        self.arrivals.push(Arrival {
            cost,
            last_right_id,
        });
        self.backs.push(Back {
            back_pos,
            back_word_pos,
            back_index,
            back_id,
            back_type,
        });
    }

    #[inline]
    fn back(&self, i: usize) -> Result<&Back, AnalysisError> {
        self.backs.get(i).ok_or_else(|| bad_index(i))
    }

    #[inline]
    fn arrival(&self, i: usize) -> Result<&Arrival, AnalysisError> {
        self.arrivals.get(i).ok_or_else(|| bad_index(i))
    }

    /// `reset()`.
    #[inline]
    pub fn reset(&mut self) {
        self.arrivals.clear();
        self.backs.clear();
    }
    /// `getPos()`.
    #[inline]
    pub fn pos(&self) -> i32 {
        self.pos
    }
    /// `getCount()`.
    #[inline]
    pub fn count(&self) -> usize {
        self.arrivals.len()
    }
    /// `setCount(count)`: keeps the first `count` back pointers.
    pub fn set_count(&mut self, count: usize) {
        self.arrivals.truncate(count);
        self.backs.truncate(count);
    }
    /// `getCost(index)`.
    #[inline]
    pub fn cost(&self, i: usize) -> Result<i32, AnalysisError> {
        Ok(self.arrival(i)?.cost)
    }
    /// `costs[index] = cost`.
    pub fn set_cost(&mut self, i: usize, cost: i32) {
        if let Some(a) = self.arrivals.get_mut(i) {
            a.cost = cost;
        }
    }
    /// `getBackPos(index)`.
    #[inline]
    pub fn back_pos(&self, i: usize) -> Result<i32, AnalysisError> {
        Ok(self.back(i)?.back_pos)
    }
    /// `getBackWordPos(index)`.
    #[inline]
    pub fn back_word_pos(&self, i: usize) -> Result<i32, AnalysisError> {
        Ok(self.back(i)?.back_word_pos)
    }
    /// `getBackID(index)`.
    #[inline]
    pub fn back_id(&self, i: usize) -> Result<i32, AnalysisError> {
        Ok(self.back(i)?.back_id)
    }
    /// `getBackIndex(index)`.
    #[inline]
    pub fn back_index(&self, i: usize) -> Result<i32, AnalysisError> {
        Ok(self.back(i)?.back_index)
    }
    /// `getBackType(index)`.
    #[inline]
    pub fn back_type(&self, i: usize) -> Result<TokenType, AnalysisError> {
        Ok(self.back(i)?.back_type)
    }
    /// `getLastRightID(index)`.
    #[inline]
    pub fn last_right_id(&self, i: usize) -> Result<i32, AnalysisError> {
        Ok(self.arrival(i)?.last_right_id)
    }

    /// `PositionNBest.addForward(forwardPos, forwardIndex, forwardID,
    /// forwardType)`.
    pub fn add_forward(&mut self, pos: i32, index: i32, id: i32, forward_type: TokenType) {
        self.forwards.push(Forward {
            pos,
            index,
            id,
            forward_type,
        });
    }
    /// `getForwardCount()`.
    pub fn forward_count(&self) -> usize {
        self.forwards.len()
    }
    /// `setForwardCount(0)`.
    pub fn clear_forwards(&mut self) {
        self.forwards.clear();
    }
    /// `getForwardPos`, `getForwardID`, `getForwardType` (and the index,
    /// which Java keeps but never reads) of forward pointer `i`.
    pub fn forward(&self, i: usize) -> Result<(i32, i32, TokenType), AnalysisError> {
        let f = self.forwards.get(i).ok_or_else(|| bad_index(i))?;
        Ok((f.pos, f.id, f.forward_type))
    }
}

/// `Viterbi.WrappedPositionArray`: the positions from the last freed one
/// on, in a ring that grows.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WrappedPositionArray {
    positions: Vec<Position>,
    next_write: usize,
    next_pos: i32,
    count: usize,
}

impl Default for WrappedPositionArray {
    fn default() -> Self {
        WrappedPositionArray {
            positions: vec![Position::default(); 8],
            next_write: 0,
            next_pos: 0,
            count: 0,
        }
    }
}

impl WrappedPositionArray {
    /// `reset()`.
    pub fn reset(&mut self) {
        // The live positions only (the freed ones were reset when freed;
        // forward pointers are cleared after every use).
        let len = self.positions.len();
        let mut index = self.next_write;
        for _ in 0..self.count {
            index = index.checked_sub(1).unwrap_or(len.saturating_sub(1));
            self.positions[index].reset();
            self.positions[index].forwards.clear();
        }
        self.next_write = 0;
        self.next_pos = 0;
        self.count = 0;
    }

    /// `get(pos)`: the position, created (with every one before it) if it
    /// is in the future.
    #[inline]
    pub fn get(&mut self, pos: i32) -> &mut Position {
        if pos >= self.next_pos {
            self.extend_to(pos);
        }
        let i = self.index(pos);
        &mut self.positions[i]
    }

    /// Creates the positions up to `pos`.
    fn extend_to(&mut self, pos: i32) {
        while pos >= self.next_pos {
            if self.count == self.positions.len() {
                // Grow, unrolling the ring so the oldest position is first.
                let len = self.positions.len();
                let mut grown = Vec::with_capacity(len.saturating_mul(2));
                grown.extend(self.positions.drain(self.next_write..));
                grown.append(&mut self.positions);
                grown.resize(
                    (len.saturating_mul(3) / 2).saturating_add(1),
                    Position::default(),
                );
                self.positions = grown;
                self.next_write = len;
            }
            if self.next_write == self.positions.len() {
                self.next_write = 0;
            }
            let next_pos = self.next_pos;
            let slot = &mut self.positions[self.next_write];
            slot.pos = next_pos;
            self.next_write = self.next_write.saturating_add(1);
            self.next_pos = self.next_pos.wrapping_add(1);
            self.count = self.count.saturating_add(1);
        }
    }

    /// The position `pos`, which must not be in the future
    /// (`pos < getNextPos()`).
    #[inline]
    pub fn at(&self, pos: i32) -> &Position {
        &self.positions[self.index(pos)]
    }

    /// `getNextPos()`.
    pub fn next_pos(&self) -> i32 {
        self.next_pos
    }

    /// `getIndex(pos)`.
    #[inline]
    fn index(&self, pos: i32) -> usize {
        // Java's `nextWrite - (nextPos - pos)`, plus the length when that is
        // negative. A live position is at most `count <= len` behind
        // `nextPos`, so one wrap suffices: no division on this hot path.
        let len = self.positions.len();
        let back = usize::try_from(self.next_pos.wrapping_sub(pos)).unwrap_or(0);
        match self.next_write.checked_sub(back) {
            Some(i) => i,
            None => len
                .wrapping_add(self.next_write)
                .wrapping_sub(back)
                .checked_rem(len)
                .unwrap_or(0),
        }
    }

    /// `freeBefore(pos)`.
    pub fn free_before(&mut self, pos: i32) {
        let ahead = usize::try_from(self.next_pos.wrapping_sub(pos)).unwrap_or(0);
        let to_free = self.count.saturating_sub(ahead);
        let len = self.positions.len();
        // nextWrite - count, wrapped into the ring.
        let mut index = match self.next_write.checked_sub(self.count) {
            Some(i) => i,
            None => self.next_write.wrapping_add(len).wrapping_sub(self.count),
        };
        for _ in 0..to_free {
            if index >= len {
                index = 0;
            }
            self.positions[index].reset();
            self.positions[index].forwards.clear();
            index = index.wrapping_add(1);
        }
        self.count = self.count.saturating_sub(to_free);
    }
}

/// The language half of Java's `Viterbi` subclasses.
pub trait ViterbiLang<T>: Send {
    /// The dictionary of a token type (`dictionaryMap.get(type)`).
    fn morph_data(&self, token_type: TokenType) -> &dyn MorphData;

    /// `(getLeftId, getRightId, getWordCost)` of a word of the dictionary
    /// of `token_type`: [`MorphData::connection`] of [`Self::morph_data`],
    /// which a language may answer without the `dyn` call.
    #[inline]
    fn connection(&self, token_type: TokenType, word_id: i32) -> (i32, i32, i32) {
        self.morph_data(token_type).connection(word_id)
    }

    /// `dictionary.lookupWordIds(sourceId, wordIdRef)` on the system
    /// dictionary.
    fn known_word_ids(&self, source_id: i32) -> &[i32];

    /// `processUnknownWord(anyMatches, posData)`: adds the unknown words
    /// starting at `v.pos`; `pos_data` names the position the arcs leave
    /// from. Returns the word length (`0` when none was added).
    fn process_unknown_word(
        &mut self,
        v: &mut Viterbi<T>,
        reader: &mut dyn CharReader,
        any_matches: bool,
        pos_data: i32,
    ) -> Result<i32, AnalysisError>;

    /// `backtrace(endPosData, fromIDX)`.
    fn backtrace(
        &mut self,
        v: &mut Viterbi<T>,
        end_pos: i32,
        from_idx: i32,
    ) -> Result<(), AnalysisError>;

    /// `backtraceNBest(endPosData, useEOS)`: Java's base throws
    /// `UnsupportedOperationException`; only an n-best language sets
    /// [`Viterbi::output_nbest`].
    fn backtrace_nbest(
        &mut self,
        _v: &mut Viterbi<T>,
        _end_pos: i32,
        _use_eos: bool,
    ) -> Result<(), AnalysisError> {
        Err(unsupported())
    }

    /// `fixupPendingList()`: as [`Self::backtrace_nbest`].
    fn fixup_pending_list(&mut self, _v: &mut Viterbi<T>) -> Result<(), AnalysisError> {
        Err(unsupported())
    }

    /// `shouldSkipProcessUnknownWord(unknownWordEndIndex, posData)`.
    fn should_skip_process_unknown_word(&self, unknown_word_end_index: i32, pos_data: i32) -> bool {
        unknown_word_end_index > pos_data
    }

    /// `computeSpacePenalty(morphData, wordID, numSpaces)`.
    fn compute_space_penalty(
        &self,
        _token_type: TokenType,
        _word_id: i32,
        _num_spaces: i32,
    ) -> i32 {
        0
    }

    /// `computePenalty(pos, length)`.
    fn compute_penalty(&self, _v: &Viterbi<T>, _pos: i32, _length: i32) -> i32 {
        0
    }
}

fn unsupported() -> AnalysisError {
    AnalysisError::IllegalState("UnsupportedOperationException".to_string())
}

/// `Viterbi`'s state.
pub struct Viterbi<T> {
    fst: Arc<TokenInfoFst>,
    user_fst: Option<Arc<TokenInfoFst>>,
    /// `costs`.
    pub costs: Arc<ConnectionCosts>,
    /// `buffer`.
    pub buffer: RollingCharBuffer,
    /// `positions`.
    pub positions: WrappedPositionArray,
    /// `end`: the input reader is exhausted.
    pub end: bool,
    /// `lastBackTracePos`.
    pub last_back_trace_pos: i32,
    /// `pos`: the next position to process.
    pub pos: i32,
    /// `pending`: parsed tokens, last first.
    pub pending: Vec<T>,
    /// `outputNBest`.
    pub output_nbest: bool,
    /// `enableSpacePenaltyFactor`.
    pub enable_space_penalty_factor: bool,
    /// `outputLongestUserEntryOnly`.
    pub output_longest_user_entry_only: bool,
}

impl<T> Viterbi<T> {
    /// `new Viterbi(fst, fstReader, dictionary, userFST, userFSTReader,
    /// userDictionary, costs, positionImpl)`.
    pub fn new(
        fst: Arc<TokenInfoFst>,
        user_fst: Option<Arc<TokenInfoFst>>,
        costs: Arc<ConnectionCosts>,
    ) -> Self {
        Viterbi {
            fst,
            user_fst,
            costs,
            buffer: RollingCharBuffer::default(),
            positions: WrappedPositionArray::default(),
            end: false,
            last_back_trace_pos: 0,
            pos: 0,
            pending: Vec::new(),
            output_nbest: false,
            enable_space_penalty_factor: false,
            output_longest_user_entry_only: false,
        }
    }

    /// `resetBuffer(reader)`: the buffer forgets its input.
    pub fn reset_buffer(&mut self) {
        self.buffer.reset();
    }

    /// `resetState()`: back to the beginning-of-sentence node.
    pub fn reset_state(&mut self) {
        self.positions.reset();
        self.pos = 0;
        self.end = false;
        self.last_back_trace_pos = 0;
        self.pending.clear();
        self.positions
            .get(0)
            .add(0, 0, -1, -1, -1, -1, TokenType::Known);
    }

    /// A buffered unit (`buffer.get(pos)` of a position already read);
    /// `-1` outside the buffer.
    pub fn char_at(&self, pos: i32) -> i32 {
        self.buffer.peek(pos).map_or(-1, i32::from)
    }

    /// `buffer.get(posStart, length)` as a shared fragment.
    pub fn fragment(&self, pos_start: i32, length: i32) -> Arc<[u16]> {
        Arc::from(self.buffer.slice(pos_start, length))
    }
}

/// `buffer.get(pos)`: the unit, `-1` past the end of the input.
fn char_at<T>(
    v: &mut Viterbi<T>,
    reader: &mut dyn CharReader,
    pos: i32,
) -> Result<i32, AnalysisError> {
    Ok(v.buffer.get(reader, pos)?.map_or(-1, i32::from))
}

/// `buffer.get(pos)` for a language's unknown-word scan, which may read
/// ahead of what [`forward`] has buffered.
pub fn read_char<T>(
    v: &mut Viterbi<T>,
    reader: &mut dyn CharReader,
    pos: i32,
) -> Result<i32, AnalysisError> {
    char_at(v, reader, pos)
}

/// `Viterbi.forward()`: runs the search forward until some tokens are
/// pending (or the input ends).
pub fn forward<T, L: ViterbiLang<T>>(
    v: &mut Viterbi<T>,
    lang: &mut L,
    reader: &mut dyn CharReader,
) -> Result<(), AnalysisError> {
    // Index of the last character of unknown word:
    let mut unknown_word_end_index: i32 = -1;
    // Maximum posAhead of user word in the entire input
    let mut user_word_max_pos_ahead: i32 = -1;
    // The automata, held once rather than per position.
    let fst = Arc::clone(&v.fst);
    let user_fst = v.user_fst.clone();

    while char_at(v, reader, v.pos)? != -1 {
        let pd = v.pos;
        let count = v.positions.get(pd).count();
        let is_frontier = v.positions.next_pos() == pd.wrapping_add(1);

        if count == 0 {
            // No arcs arrive here; move to next position:
            v.pos = v.pos.wrapping_add(1);
            continue;
        }

        if v.pos > v.last_back_trace_pos && count == 1 && is_frontier {
            // We are at a "frontier", and only one node is alive, so
            // whatever the eventual best path is must come through this
            // node. So we can safely commit to the prefix of the best path
            // at this point:
            if v.output_nbest {
                lang.backtrace_nbest(v, pd, false)?;
            }
            lang.backtrace(v, pd, 0)?;
            if v.output_nbest {
                lang.fixup_pending_list(v)?;
            }
            // Re-base cost so we don't risk int overflow:
            v.positions.get(pd).set_cost(0, 0);
            if !v.pending.is_empty() {
                return Ok(());
            }
            // The backtrace only produced punctuation tokens, so keep
            // parsing.
        }

        if v.pos.wrapping_sub(v.last_back_trace_pos) >= MAX_BACKTRACE_GAP {
            // Safety: if we've buffered too much, force a backtrace now. We
            // find the least-cost partial path, across all paths, backtrace
            // from it, and then prune all others.
            let mut least_idx: Option<usize> = None;
            let mut least_cost = i32::MAX;
            let mut least_pos = -1;
            for pos2 in v.pos..v.positions.next_pos() {
                let p = v.positions.at(pos2);
                for idx in 0..p.count() {
                    let cost = p.cost(idx)?;
                    if cost < least_cost {
                        least_cost = cost;
                        least_idx = Some(idx);
                        least_pos = pos2;
                    }
                }
            }
            // We will always have at least one live path:
            let least_idx = least_idx.ok_or_else(|| bad_index(0))?;

            if v.output_nbest {
                lang.backtrace_nbest(v, least_pos, false)?;
            }

            // Second pass: prune all but the best path:
            for pos2 in v.pos..v.positions.next_pos() {
                let p = v.positions.get(pos2);
                if pos2 != least_pos {
                    p.reset();
                } else {
                    if least_idx != 0 {
                        let (arrival, best) = (*p.arrival(least_idx)?, *p.back(least_idx)?);
                        p.arrivals[0] = arrival;
                        p.backs[0] = best;
                    }
                    p.set_count(1);
                }
            }

            lang.backtrace(v, least_pos, 0)?;
            if v.output_nbest {
                lang.fixup_pending_list(v)?;
            }

            // Re-base cost so we don't risk int overflow:
            for a in &mut v.positions.get(least_pos).arrivals {
                a.cost = 0;
            }

            if v.pos != least_pos {
                // We jumped into a future position:
                v.pos = least_pos;
            }
            if !v.pending.is_empty() {
                return Ok(());
            }
            continue;
        }

        if v.enable_space_penalty_factor
            && u32::try_from(char_at(v, reader, v.pos)?)
                .is_ok_and(|c| get_type(c) == SPACE_SEPARATOR)
        {
            // We add single space separator as prefixes of the terms that we
            // extract. This information is needed to compute the space
            // penalty factor of each term. These whitespace prefixes are
            // removed when the final tokens are generated, or added as
            // separated tokens when discardPunctuation is unset.
            v.pos = v.pos.wrapping_add(1);
            if char_at(v, reader, v.pos)? == -1 {
                v.pos = pd;
            }
        }

        let mut any_matches = false;

        // First try user dict:
        if let Some(user_fst) = user_fst.as_deref() {
            let mut arc = user_fst.first_arc();
            let mut output: i32 = 0;
            let mut max_pos_ahead: i32 = 0;
            let mut output_max_pos_ahead: i32 = 0;
            let mut arc_final_out_max_pos_ahead: i32 = 0;

            let mut pos_ahead = v.pos;
            loop {
                let ch = char_at(v, reader, pos_ahead)?;
                if ch == -1 {
                    break;
                }
                match user_fst.find_target_arc(ch, &arc, pos_ahead == v.pos)? {
                    Some(a) => arc = a,
                    None => break,
                }
                output = output.wrapping_add(arc.output() as i32);
                if arc.is_final() {
                    max_pos_ahead = pos_ahead;
                    output_max_pos_ahead = output;
                    arc_final_out_max_pos_ahead = arc.next_final_output() as i32;
                    any_matches = true;
                    if !v.output_longest_user_entry_only {
                        // add all matched user entries.
                        let word_pos = v.pos;
                        add(
                            v,
                            lang,
                            TokenType::User,
                            pd,
                            word_pos,
                            pos_ahead.wrapping_add(1),
                            output.wrapping_add(arc.next_final_output() as i32),
                            false,
                        );
                    }
                }
                pos_ahead = pos_ahead.wrapping_add(1);
            }

            // Longest matching for user word
            if any_matches && max_pos_ahead > user_word_max_pos_ahead {
                if v.output_longest_user_entry_only {
                    let word_pos = v.pos;
                    add(
                        v,
                        lang,
                        TokenType::User,
                        pd,
                        word_pos,
                        max_pos_ahead.wrapping_add(1),
                        output_max_pos_ahead.wrapping_add(arc_final_out_max_pos_ahead),
                        false,
                    );
                }
                user_word_max_pos_ahead = user_word_max_pos_ahead.max(max_pos_ahead);
            }
        }

        if !any_matches {
            // Next, try known dictionary matches
            let mut arc = fst.first_arc();
            let mut output: i32 = 0;
            let mut pos_ahead = v.pos;
            loop {
                let ch = char_at(v, reader, pos_ahead)?;
                if ch == -1 {
                    break;
                }
                match fst.find_target_arc(ch, &arc, pos_ahead == v.pos)? {
                    Some(a) => arc = a,
                    None => break,
                }
                output = output.wrapping_add(arc.output() as i32);
                if arc.is_final() {
                    let source = output.wrapping_add(arc.next_final_output() as i32);
                    let lang = &*lang;
                    for &word_id in lang.known_word_ids(source) {
                        let word_pos = v.pos;
                        add(
                            v,
                            lang,
                            TokenType::Known,
                            pd,
                            word_pos,
                            pos_ahead.wrapping_add(1),
                            word_id,
                            false,
                        );
                        any_matches = true;
                    }
                }
                pos_ahead = pos_ahead.wrapping_add(1);
            }
        }

        if !lang.should_skip_process_unknown_word(unknown_word_end_index, pd) {
            let unknown_word_length = lang.process_unknown_word(v, reader, any_matches, pd)?;
            unknown_word_end_index = pd.wrapping_add(unknown_word_length);
        }
        v.pos = v.pos.wrapping_add(1);
    }

    v.end = true;

    if v.pos > 0 {
        let end_pos = v.pos;
        let mut least_cost = i32::MAX;
        let mut least_idx: i32 = -1;
        let end_data = v.positions.get(end_pos);
        for (idx, b) in end_data.arrivals.iter().enumerate() {
            // Add EOS cost:
            let cost = b.cost.wrapping_add(v.costs.get(b.last_right_id, 0));
            if cost < least_cost {
                least_cost = cost;
                least_idx = i32::try_from(idx).unwrap_or(i32::MAX);
            }
        }

        if v.output_nbest {
            lang.backtrace_nbest(v, end_pos, true)?;
        }
        lang.backtrace(v, end_pos, least_idx)?;
        if v.output_nbest {
            lang.fixup_pending_list(v)?;
        }
    }
    // else: no characters in the input string; no tokens.
    Ok(())
}

/// `Viterbi.add(morphData, fromPosData, wordPos, endPos, wordID, type,
/// addPenalty)`: the least-cost arc from `from_pos` for the word, added to
/// `end_pos`.
#[allow(clippy::too_many_arguments)]
pub fn add<T, L: ViterbiLang<T> + ?Sized>(
    v: &mut Viterbi<T>,
    lang: &L,
    token_type: TokenType,
    from_pos: i32,
    word_pos: i32,
    end_pos: i32,
    word_id: i32,
    add_penalty: bool,
) {
    let (left_id, right_id, word_cost) = lang.connection(token_type, word_id);
    let mut least_cost = i32::MAX;
    let mut least_idx: i32 = -1;
    // Create the end position first: the array may grow.
    v.positions.get(end_pos);
    let from = v.positions.at(from_pos);
    // The number of spaces before the term
    let num_spaces = word_pos.wrapping_sub(from.pos);
    let space_penalty = lang.compute_space_penalty(token_type, word_id, num_spaces);
    // Every cost here has the word's left id: one row of the matrix.
    let row = v.costs.row(left_id);
    for (idx, b) in from.arrivals.iter().enumerate() {
        // Cost is path cost so far, plus word cost (added at end of loop),
        // plus bigram cost and space penalty cost.
        let bigram = match usize::try_from(b.last_right_id)
            .ok()
            .and_then(|r| row.get(r))
        {
            Some(&c) => i32::from(c),
            None => v.costs.get(b.last_right_id, left_id),
        };
        let cost = b.cost.wrapping_add(bigram).wrapping_add(space_penalty);
        if cost < least_cost {
            least_cost = cost;
            least_idx = i32::try_from(idx).unwrap_or(i32::MAX);
        }
    }
    let from_data_pos = from.pos;

    least_cost = least_cost.wrapping_add(word_cost);

    if add_penalty && token_type != TokenType::User {
        let penalty = lang.compute_penalty(v, from_data_pos, end_pos.wrapping_sub(from_data_pos));
        least_cost = least_cost.wrapping_add(penalty);
    }

    v.positions.get(end_pos).add(
        least_cost,
        right_id,
        from_data_pos,
        word_pos,
        least_idx,
        word_id,
        token_type,
    );
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]
    use super::*;
    use crate::morph::connection_costs::tests::costs_file;
    use crate::morph::token::Token;
    use crate::reader::StrReader;

    /// Every word costs its id; one connection class, cost 0.
    struct Toy;
    impl MorphData for Toy {
        fn left_id(&self, _: i32) -> i32 {
            0
        }
        fn right_id(&self, _: i32) -> i32 {
            0
        }
        fn word_cost(&self, id: i32) -> i32 {
            id * 10
        }
    }

    /// A language with no unknown words, whose backtrace emits the best
    /// path, and which keeps the trait's defaults (no n-best).
    struct ToyLang {
        ids: Vec<i32>,
    }
    impl ViterbiLang<Token> for ToyLang {
        fn morph_data(&self, _: TokenType) -> &dyn MorphData {
            &Toy
        }
        fn known_word_ids(&self, source_id: i32) -> &[i32] {
            let s = source_id as usize;
            &self.ids[s..s + 1]
        }
        fn process_unknown_word(
            &mut self,
            _v: &mut Viterbi<Token>,
            _r: &mut dyn CharReader,
            _any: bool,
            _pos: i32,
        ) -> Result<i32, AnalysisError> {
            Ok(0)
        }
        fn backtrace(
            &mut self,
            v: &mut Viterbi<Token>,
            end_pos: i32,
            from_idx: i32,
        ) -> Result<(), AnalysisError> {
            let last = v.last_back_trace_pos;
            let frag = v.fragment(last, end_pos - last);
            let (mut pos, mut idx) = (end_pos, from_idx.max(0) as usize);
            while pos > last {
                let p = v.positions.at(pos);
                let (back, t, next) = (p.back_pos(idx)?, p.back_type(idx)?, p.back_index(idx)?);
                v.pending.push(Token::new(
                    frag.clone(),
                    back - last,
                    pos - back,
                    back,
                    pos,
                    t,
                ));
                pos = back;
                idx = next.max(0) as usize;
            }
            v.last_back_trace_pos = end_pos;
            v.buffer.free_before(end_pos);
            v.positions.free_before(end_pos);
            Ok(())
        }
    }

    fn units(s: &str) -> Vec<u16> {
        s.encode_utf16().collect()
    }

    fn viterbi(user: Option<&[&str]>) -> Viterbi<Token> {
        let fst =
            TokenInfoFst::from_sorted(&[units("a"), units("ab"), units("b")], 0x7F, 0x20).unwrap();
        let user = user.map(|keys| {
            let keys: Vec<Vec<u16>> = keys.iter().map(|k| units(k)).collect();
            Arc::new(TokenInfoFst::from_sorted(&keys, 0x7F, 0x20).unwrap())
        });
        let costs = ConnectionCosts::read(&costs_file("cc", 1, &[0]), "cc", 1).unwrap();
        let mut v = Viterbi::new(Arc::new(fst), user, Arc::new(costs));
        v.reset_state();
        v
    }

    fn run(v: &mut Viterbi<Token>, text: &str) -> Result<Vec<String>, AnalysisError> {
        let mut lang = ToyLang { ids: vec![0, 1, 2] };
        let mut r = StrReader::new(text);
        let mut out = Vec::new();
        while !v.end || !v.pending.is_empty() {
            if v.pending.is_empty() {
                forward(v, &mut lang, &mut r)?;
            }
            if let Some(t) = v.pending.pop() {
                out.push(format!(
                    "{}:{}",
                    t.surface_form_string(),
                    t.token_type.name()
                ));
            }
        }
        Ok(out)
    }

    #[test]
    fn toy_language_over_the_trait_defaults() {
        // "ab" (id 1, cost 10) beats "a"+"b" (0 + 20).
        assert_eq!(
            run(&mut viterbi(None), "abab").unwrap(),
            ["ab:KNOWN", "ab:KNOWN"]
        );
        assert!(run(&mut viterbi(None), "").unwrap().is_empty());
        // User entries replace the system dictionary's at their start (every
        // one, or only the longest): "ab" (ordinal 1, cost 10) beats "a"
        // (0) + "b" (20).
        let mut v = viterbi(Some(&["a", "ab"]));
        assert_eq!(run(&mut v, "ab").unwrap(), ["ab:USER"]);
        let mut v = viterbi(Some(&["a", "ab"]));
        v.output_longest_user_entry_only = true;
        assert_eq!(run(&mut v, "ab").unwrap(), ["ab:USER"]);
        // A language without n-best refuses it, as Java's base class does.
        let mut v = viterbi(None);
        v.output_nbest = true;
        let e = run(&mut v, "ab").unwrap_err();
        assert!(
            e.to_string().contains("UnsupportedOperationException"),
            "{e}"
        );
        let lang = ToyLang { ids: vec![] };
        assert_eq!(lang.compute_space_penalty(TokenType::Known, 0, 3), 0);
        assert_eq!(lang.compute_penalty(&viterbi(None), 0, 3), 0);
        let mut lang = ToyLang { ids: vec![] };
        assert!(lang.fixup_pending_list(&mut viterbi(None)).is_err());
        // An index past the back pointers is Java's AIOOBE.
        let p = Position::default();
        assert!(p.cost(3).is_err());
        assert_eq!((p.pos(), p.count()), (0, 0));
    }
}
