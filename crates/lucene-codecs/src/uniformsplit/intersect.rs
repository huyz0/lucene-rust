//! Port of `IntersectBlockReader` and its `AutomatonNextTermCalculator`:
//! the terms an automaton accepts, scanning block lines while they match
//! and seeking past a run of rejected ones to the automaton's next
//! possible term.

use lucene_util::automaton::{ByteRunnable, CompiledAutomaton, Transition, TransitionAccessor};

use super::block_reader::BlockReader;
use super::{Error, Result, SeekStatus, TermBytes, UniformSplitTermState, UniformSplitTerms};
use crate::postings::{DocInput, Postings, PostingsFlags};

/// `NUM_CONSECUTIVELY_REJECTED_TERMS_THRESHOLD`: rejected lines in a row
/// before the reader seeks.
pub(crate) const NUM_CONSECUTIVELY_REJECTED_TERMS_THRESHOLD: i32 = 4;

/// `BlockIteration`: how [`IntersectBlockReader`] goes to the next block.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BlockIteration {
    /// The block after the current one.
    Next,
    /// The block of `seekTerm`.
    Seek,
    /// None: no term after can match.
    End,
}

/// `IntersectBlockReader`: a `TermsEnum` over the terms of one field an
/// automaton accepts.
pub struct IntersectBlockReader<'a> {
    pub(super) base: BlockReader<'a>,
    byte_runnable: &'a dyn ByteRunnable,
    /// `commonSuffix`.
    common_suffix: Option<&'a [u8]>,
    /// `minTermLength`.
    min_term_length: usize,
    /// `nextStringCalculator`.
    next_string_calculator: AutomatonNextTermCalculator<'a>,
    /// `seekTerm`.
    seek_term: Option<Vec<u8>>,
    /// `numMatchedBytes`.
    num_matched_bytes: usize,
    /// `states`: the automaton state after each matched byte of the line.
    states: Vec<i32>,
    /// `blockIteration`.
    block_iteration: BlockIteration,
    /// `numConsecutivelyRejectedTerms`.
    num_consecutively_rejected_terms: i32,
}

impl std::fmt::Debug for IntersectBlockReader<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IntersectBlockReader")
            .field("base", &self.base)
            .field("block_iteration", &self.block_iteration)
            .finish()
    }
}

impl<'a> IntersectBlockReader<'a> {
    /// `new IntersectBlockReader(compiled, startTerm, ...)`.
    ///
    /// # Errors
    /// [`Error::Unsupported`] for a `NORMAL` automaton without a runnable
    /// or transitions (never, from `CompiledAutomaton`).
    pub(crate) fn new(
        terms: &'a UniformSplitTerms,
        compiled: &'a CompiledAutomaton,
        start_term: Option<&[u8]>,
    ) -> Result<Self> {
        let (Some(byte_runnable), Some(transition_accessor)) = (
            compiled.get_byte_runnable(),
            compiled.get_transition_accessor(),
        ) else {
            return Err(Error::Unsupported(
                "please use CompiledAutomaton.getTermsEnum instead",
            ));
        };
        let finite = compiled.finite;
        let common_suffix = compiled.common_suffix_ref.as_deref();
        Ok(IntersectBlockReader {
            base: BlockReader::new(terms),
            byte_runnable,
            common_suffix,
            min_term_length: min_term_length(
                byte_runnable,
                transition_accessor,
                finite,
                common_suffix,
            ),
            next_string_calculator: AutomatonNextTermCalculator::new(
                byte_runnable,
                transition_accessor,
                finite,
            ),
            seek_term: start_term.map(<[u8]>::to_vec),
            num_matched_bytes: 0,
            states: Vec::new(),
            block_iteration: BlockIteration::Next,
            num_consecutively_rejected_terms: 0,
        })
    }

    /// `next()`.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or dictionary.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        if self.base.block_header.is_none() {
            if !self.seek_first_block()? {
                return Ok(None);
            }
            self.states = vec![0; 32];
            self.block_iteration = BlockIteration::Next;
        }
        self.base.term_state = None;
        loop {
            if self.next_term_in_block_matching()? {
                return Ok(self.base.term());
            }
            if !self.next_block()? {
                return Ok(None);
            }
        }
    }

    /// `seekFirstBlock()`.
    fn seek_first_block(&mut self) -> Result<bool> {
        self.seek_term = self
            .next_string_calculator
            .next_seek_term(self.seek_term.as_deref());
        let Some(seek_term) = self.seek_term.take() else {
            return Ok(false);
        };
        let mut block_start_fp = self
            .base
            .get_or_create_dictionary_browser()
            .seek_block(&seek_term)?;
        // SENTINEL-OK: `seek_block`'s `-1` is "before the first block".
        if block_start_fp == -1 {
            block_start_fp = self.base.terms.metadata.first_block_start_fp;
        } else if self.base.is_beyond_last_term(&seek_term, block_start_fp) {
            self.seek_term = Some(seek_term);
            return Ok(false);
        }
        let initialized = self
            .base
            .initialize_header(Some(&seek_term), block_start_fp);
        self.seek_term = Some(seek_term);
        initialized?;
        Ok(self.base.block_header.is_some())
    }

    /// `nextTermInBlockMatching()`: whether the reader is on a matching
    /// line of the current block.
    // ARITH: `index < term.len()` when incremented; `states` grows to more
    // than the term's length first.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_term_in_block_matching(&mut self) -> Result<bool> {
        match self.seek_term.take() {
            None => {
                if self.base.read_line_in_block()?.is_none() {
                    return Ok(false);
                }
            }
            Some(seek_term) => {
                if self.base.seek_in_block(&seek_term)? == SeekStatus::End {
                    return Ok(false);
                }
            }
        }
        loop {
            let Some((_, mdp_length)) = self.base.line else {
                // `blockLine` is set: the line was just read.
                return Ok(false);
            };
            let line_term = &self.base.line_reader.current_term;
            if self.states.len() <= line_term.len() {
                // Java grows to `ArrayUtil.oversize(length + 1)`; only the
                // capacity differs.
                self.states.resize(line_term.len() + 1, 0);
            }
            let mut index = TermBytes::suffix_offset(mdp_length).min(self.num_matched_bytes);
            if line_term.len() >= self.min_term_length
                && self
                    .common_suffix
                    .is_none_or(|suffix| line_term.ends_with(suffix))
            {
                let mut state = self.states[index];
                loop {
                    if index == line_term.len() {
                        if self.byte_runnable.is_accept(state) {
                            self.num_matched_bytes = index;
                            if self.num_consecutively_rejected_terms > 0 {
                                self.num_consecutively_rejected_terms = 0;
                            }
                            return Ok(true);
                        }
                        break;
                    }
                    state = self.byte_runnable.step(state, i32::from(line_term[index]));
                    if state == -1 {
                        break;
                    }
                    index += 1;
                    self.states[index] = state;
                }
            }
            self.num_matched_bytes = index;
            self.num_consecutively_rejected_terms =
                self.num_consecutively_rejected_terms.wrapping_add(1);
            let lines_count = self.base.block_header.map_or(0, |h| h.lines_count);
            if self.num_consecutively_rejected_terms >= NUM_CONSECUTIVELY_REJECTED_TERMS_THRESHOLD
                && self.base.line_index_in_block < lines_count.saturating_sub(1)
                && !self
                    .next_string_calculator
                    .is_linear_state(&self.base.line_reader.current_term)
            {
                let line_term = self.base.line_reader.current_term.clone();
                let Some(seek_term) = self.next_string_calculator.next_seek_term(Some(&line_term))
                else {
                    self.block_iteration = BlockIteration::End;
                    return Ok(false);
                };
                self.base.read_line_in_block()?;
                if self.base.line.is_some()
                    && seek_term.as_slice() > self.base.line_reader.current_term.as_slice()
                {
                    self.seek_term = Some(seek_term);
                    self.block_iteration = BlockIteration::Seek;
                    return Ok(false);
                }
                self.num_consecutively_rejected_terms = i32::MIN;
            } else if self.base.read_line_in_block()?.is_none() {
                return Ok(false);
            }
        }
    }

    /// `nextBlock()`.
    ///
    /// # Errors
    /// `Corrupted` where a seek would go back before the block just
    /// scanned -- the term the automaton skips to is after every line read,
    /// so only a corrupt dictionary or block leads there, and Java would
    /// scan the same blocks again forever.
    fn next_block(&mut self) -> Result<bool> {
        let block_start_fp = match self.block_iteration {
            BlockIteration::Next => self.base.block_input_fp as i64,
            BlockIteration::Seek => {
                let seek_term = self.seek_term.as_deref().unwrap_or_default();
                let fp = self
                    .base
                    .get_or_create_dictionary_browser()
                    .seek_block(seek_term)?;
                if self.base.is_beyond_last_term(seek_term, fp) {
                    return Ok(false);
                }
                if fp < self.base.block_start_fp {
                    return Err(self.base.corrupt_at("Illegal seek back to block", fp));
                }
                self.block_iteration = BlockIteration::Next;
                fp
            }
            BlockIteration::End => return Ok(false),
        };
        self.num_matched_bytes = 0;
        self.num_consecutively_rejected_terms = 0;
        let seek_term = self.seek_term.take();
        let initialized = self
            .base
            .initialize_header(seek_term.as_deref(), block_start_fp);
        self.seek_term = seek_term;
        initialized?;
        Ok(self.base.block_header.is_some())
    }

    /// `term()`.
    pub fn term(&self) -> Option<&[u8]> {
        self.base.term()
    }

    /// `docFreq()`.
    ///
    /// # Errors
    /// See [`BlockReader::term_state`].
    pub fn doc_freq(&mut self) -> Result<i32> {
        self.base.doc_freq()
    }

    /// `totalTermFreq()`.
    ///
    /// # Errors
    /// See [`BlockReader::term_state`].
    pub fn total_term_freq(&mut self) -> Result<i64> {
        self.base.total_term_freq()
    }

    /// `termState()`.
    ///
    /// # Errors
    /// See [`BlockReader::term_state`].
    pub fn term_state(&mut self) -> Result<UniformSplitTermState> {
        self.base.term_state()
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// See [`BlockReader::postings`].
    pub fn postings(&mut self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        self.base.postings(doc_in, flags)
    }

    /// `seekCeil`, `seekExact(term)`, `seekExact(ord)`, `seekExact(term,
    /// state)`.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn seek_ceil(&mut self, _term: &[u8]) -> Result<SeekStatus> {
        Err(Error::Unsupported("an intersect terms enum does not seek"))
    }
}

/// `getMinTermLength()`: the length of the automaton's single-byte-path
/// prefix (up to its first accept state) and its common suffix.
fn min_term_length(
    byte_runnable: &dyn ByteRunnable,
    transition_accessor: &dyn TransitionAccessor,
    finite: bool,
    common_suffix: Option<&[u8]>,
) -> usize {
    let common_suffix_length = common_suffix.map_or(0, <[u8]>::len);
    if !finite {
        return common_suffix_length;
    }
    let mut common_prefix_length = 0usize;
    let mut state = 0;
    let mut t = Transition::default();
    loop {
        if byte_runnable.is_accept(state) {
            return common_prefix_length.max(common_suffix_length);
        }
        if transition_accessor.get_num_transitions(state) == 1 {
            transition_accessor.get_transition(state, 0, &mut t);
            if t.min == t.max {
                state = t.dest;
                // A finite automaton has no cycle: at most its state count.
                common_prefix_length = common_prefix_length.saturating_add(1);
                continue;
            }
        }
        break;
    }
    common_prefix_length.saturating_add(common_suffix_length)
}

/// `AutomatonNextTermCalculator`: `AutomatonTermsEnum`'s `nextString`, the
/// smallest term after a rejected one the automaton could accept.
struct AutomatonNextTermCalculator<'a> {
    byte_runnable: &'a dyn ByteRunnable,
    transition_accessor: &'a dyn TransitionAccessor,
    finite: bool,
    /// `visited`: the generation each state was last seen in (infinite
    /// automata only).
    visited: Vec<i16>,
    /// `curGen`.
    cur_gen: i16,
    /// `seekBytesRef`.
    seek_bytes_ref: Vec<u8>,
    /// `linear`.
    linear: bool,
    /// `linearUpperBound`.
    linear_upper_bound: Vec<u8>,
    /// `transition`.
    transition: Transition,
    /// `savedStates`.
    saved_states: Vec<i32>,
}

/// A state as an index (states are non-negative once stepped to).
fn idx(state: i32) -> usize {
    usize::try_from(state).unwrap_or(usize::MAX)
}

// ARITH: positions are within `seek_bytes_ref` (a term), `saved_states`
// is grown to one more than its length; transition labels are bytes.
#[allow(clippy::arithmetic_side_effects)]
impl<'a> AutomatonNextTermCalculator<'a> {
    fn new(
        byte_runnable: &'a dyn ByteRunnable,
        transition_accessor: &'a dyn TransitionAccessor,
        finite: bool,
    ) -> Self {
        AutomatonNextTermCalculator {
            byte_runnable,
            transition_accessor,
            finite,
            visited: if finite {
                Vec::new()
            } else {
                vec![0; idx(byte_runnable.get_size())]
            },
            cur_gen: 0,
            seek_bytes_ref: Vec::new(),
            linear: false,
            linear_upper_bound: Vec::new(),
            transition: Transition::default(),
            saved_states: Vec::new(),
        }
    }

    /// `setVisited(state)`.
    fn set_visited(&mut self, state: i32) {
        if !self.finite {
            let s = idx(state);
            if s >= self.visited.len() {
                self.visited.resize(s + 1, 0);
            }
            self.visited[s] = self.cur_gen;
        }
    }

    /// `isVisited(state)`.
    fn is_visited(&self, state: i32) -> bool {
        !self.finite && self.visited.get(idx(state)) == Some(&self.cur_gen)
    }

    /// `isLinearState(term)`.
    fn is_linear_state(&self, term: &[u8]) -> bool {
        self.linear && term < self.linear_upper_bound.as_slice()
    }

    /// `nextSeekTerm(term)`.
    fn next_seek_term(&mut self, term: Option<&[u8]>) -> Option<Vec<u8>> {
        match term {
            None => {
                if self.byte_runnable.is_accept(0) {
                    return Some(self.seek_bytes_ref.clone());
                }
            }
            Some(term) => {
                self.seek_bytes_ref.clear();
                self.seek_bytes_ref.extend_from_slice(term);
            }
        }
        if self.next_string() {
            Some(self.seek_bytes_ref.clone())
        } else {
            None
        }
    }

    /// `setLinear(position)`.
    fn set_linear(&mut self, position: usize) {
        let mut state = 0;
        let mut max_interval = 0xff;
        for i in 0..position {
            state = self
                .byte_runnable
                .step(state, i32::from(self.seek_bytes_ref[i]));
        }
        let c = i32::from(self.seek_bytes_ref[position]);
        let num_transitions = self.transition_accessor.get_num_transitions(state);
        self.transition_accessor
            .init_transition(state, &mut self.transition);
        for _ in 0..num_transitions {
            self.transition_accessor
                .get_next_transition(&mut self.transition);
            if self.transition.min <= c && c <= self.transition.max {
                max_interval = self.transition.max;
                break;
            }
        }
        if max_interval != 0xff {
            max_interval += 1;
        }
        self.linear_upper_bound.clear();
        self.linear_upper_bound
            .extend_from_slice(&self.seek_bytes_ref[..position]);
        self.linear_upper_bound.push(max_interval as u8);
        self.linear = true;
    }

    /// `nextString()`.
    fn next_string(&mut self) -> bool {
        let mut pos = 0;
        self.saved_states.resize(self.seek_bytes_ref.len() + 1, 0);
        self.saved_states[0] = 0;
        loop {
            if !self.finite {
                self.cur_gen = self.cur_gen.wrapping_add(1);
                if self.cur_gen == 0 {
                    self.visited.fill(-1);
                }
            }
            self.linear = false;
            let mut state = self.saved_states[pos];
            while pos < self.seek_bytes_ref.len() {
                self.set_visited(state);
                let next_state = self
                    .byte_runnable
                    .step(state, i32::from(self.seek_bytes_ref[pos]));
                if next_state == -1 {
                    break;
                }
                self.saved_states[pos + 1] = next_state;
                if !self.linear && self.is_visited(next_state) {
                    self.set_linear(pos);
                }
                state = next_state;
                pos += 1;
            }
            if self.next_string_from(state, pos) {
                return true;
            }
            // No more solutions from this useful portion: backtrack.
            let Some(back) = self.backtrack(pos) else {
                return false;
            };
            pos = back;
            let new_state = self
                .byte_runnable
                .step(self.saved_states[pos], i32::from(self.seek_bytes_ref[pos]));
            if new_state >= 0 && self.byte_runnable.is_accept(new_state) {
                // The string is good to go as is.
                return true;
            }
            if !self.finite {
                pos = 0;
            }
        }
    }

    /// `nextString(state, position)`.
    fn next_string_from(&mut self, mut state: i32, position: usize) -> bool {
        // The next character must be greater than the existing one, if any.
        let mut c = 0;
        if position < self.seek_bytes_ref.len() {
            c = i32::from(self.seek_bytes_ref[position]);
            if c == 0xff {
                return false;
            }
            c += 1;
        }
        self.seek_bytes_ref.truncate(position);
        self.set_visited(state);
        let num_transitions = self.transition_accessor.get_num_transitions(state);
        self.transition_accessor
            .init_transition(state, &mut self.transition);
        for _ in 0..num_transitions {
            self.transition_accessor
                .get_next_transition(&mut self.transition);
            if self.transition.max >= c {
                let next_char = c.max(self.transition.min);
                self.seek_bytes_ref.push(next_char as u8);
                state = self.transition.dest;
                // Append the smallest labels until a loop or an accept
                // state.
                while !self.is_visited(state) && !self.byte_runnable.is_accept(state) {
                    self.set_visited(state);
                    self.transition_accessor
                        .init_transition(state, &mut self.transition);
                    self.transition_accessor
                        .get_next_transition(&mut self.transition);
                    state = self.transition.dest;
                    self.seek_bytes_ref.push(self.transition.min as u8);
                    if !self.linear && self.is_visited(state) {
                        self.set_linear(self.seek_bytes_ref.len() - 1);
                    }
                }
                return true;
            }
        }
        false
    }

    /// `backtrack(position)`: the position of the byte incremented.
    fn backtrack(&mut self, mut position: usize) -> Option<usize> {
        while position > 0 {
            position -= 1;
            let next_char = self.seek_bytes_ref[position];
            if next_char != 0xff {
                self.seek_bytes_ref[position] = next_char + 1;
                self.seek_bytes_ref.truncate(position + 1);
                return Some(position);
            }
        }
        None
    }
}
