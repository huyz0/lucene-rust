//! Port of `OrdsIntersectTermsEnum` and `OrdsIntersectTermsEnumFrame`: the
//! terms of one field an automaton accepts, walking only the blocks whose
//! prefixes the automaton can still accept, skipping floor blocks below the
//! first transition's label and a block's entries past its last.

use lucene_util::automaton::{ByteRunnableRef, Transition, TransitionAccessor};

use super::outputs::{FstOrdsOutputs, Output};
use super::segment_enum::{decode_block_meta, OrdsTermState, TermBuf};
use super::{
    corrupt, length, read_block, read_byte, read_vint, read_vlong, ushr32, ushr64, Error,
    OrdsFieldReader, Result, OUTPUT_FLAG_IS_FLOOR,
};
use crate::fst_compiler::FstOutputs;
use crate::fst_util::Arc;
use crate::postings::{DocInput, Postings, PostingsFlags, TermMetadata};

/// `OrdsIntersectTermsEnumFrame`. Frames are reused (`getFrame`), and so,
/// as in Java, is the floor data a frame last read.
#[derive(Debug, Clone)]
struct Frame {
    ord: usize,
    fp: i64,
    fp_orig: i64,
    fp_end: i64,
    last_sub_fp: i64,
    /// `state`: the automaton's state at the block's prefix.
    state: i32,
    meta_data_upto: i32,
    suffix_bytes: Vec<u8>,
    suffix_pos: usize,
    stat_bytes: Vec<u8>,
    stat_pos: usize,
    floor_data: Vec<u8>,
    floor_pos: usize,
    prefix: usize,
    ent_count: i32,
    next_ent: i32,
    is_last_in_floor: bool,
    is_leaf_block: bool,
    num_follow_floor_blocks: i32,
    next_floor_label: i32,
    transition: Transition,
    cur_transition_max: i32,
    transition_index: i32,
    transition_count: i32,
    arc: Arc<Output>,
    term_state: OrdsTermState,
    bytes: Vec<u8>,
    bytes_pos: usize,
    output_prefix: Output,
    start_byte_pos: usize,
    suffix: usize,
}

impl Frame {
    fn new(ord: usize, arc: Arc<Output>) -> Self {
        Frame {
            ord,
            fp: 0,
            fp_orig: 0,
            fp_end: 0,
            last_sub_fp: 0,
            state: 0,
            meta_data_upto: 0,
            suffix_bytes: Vec::new(),
            suffix_pos: 0,
            stat_bytes: Vec::new(),
            stat_pos: 0,
            floor_data: Vec::new(),
            floor_pos: 0,
            prefix: 0,
            ent_count: 0,
            next_ent: 0,
            is_last_in_floor: false,
            is_leaf_block: false,
            num_follow_floor_blocks: 0,
            next_floor_label: 0,
            transition: Transition::new(),
            cur_transition_max: 0,
            transition_index: 0,
            transition_count: 0,
            arc,
            term_state: OrdsTermState::default(),
            bytes: Vec::new(),
            bytes_pos: 0,
            output_prefix: Output::NO_OUTPUT,
            start_byte_pos: 0,
            suffix: 0,
        }
    }

    /// One step of the floor-follower walk: the next follower's fp, its
    /// lead label (256 past the last) and the ord delta Java adds to the
    /// unused `termOrd`.
    // ARITH: `num_follow_floor_blocks > 0` was checked by the caller.
    #[allow(clippy::arithmetic_side_effects)]
    fn skip_floor_block(&mut self) -> Result<()> {
        self.fp = self.fp_orig.wrapping_add(ushr64(
            read_vlong(&self.floor_data, &mut self.floor_pos)?,
            1,
        ));
        self.num_follow_floor_blocks -= 1;
        if self.num_follow_floor_blocks != 0 {
            // SENTINEL-OK: this module's `read_byte` reads an unsigned floor
            // label (0..=255) or errors; it has no `-1` sentinel.
            self.next_floor_label = read_byte(&self.floor_data, &mut self.floor_pos)?;
            read_vlong(&self.floor_data, &mut self.floor_pos)?;
        } else {
            self.next_floor_label = 256;
        }
        Ok(())
    }

    /// `loadNextFloorBlock()`.
    fn load_next_floor_block(&mut self, tio: &[u8]) -> Result<()> {
        loop {
            if self.num_follow_floor_blocks <= 0 {
                // Java asserts `numFollowFloorBlocks > 0`.
                return Err(corrupt("floor block without followers"));
            }
            self.skip_floor_block()?;
            if self.num_follow_floor_blocks == 0 || self.next_floor_label > self.transition.min {
                break;
            }
        }
        self.load(None, tio, None)
    }

    /// `setState(state)`.
    fn set_state(&mut self, state: i32, accessor: &dyn TransitionAccessor) {
        self.state = state;
        self.transition_index = 0;
        self.transition_count = accessor.get_num_transitions(state);
        if self.transition_count != 0 {
            accessor.init_transition(state, &mut self.transition);
            accessor.get_next_transition(&mut self.transition);
            self.cur_transition_max = self.transition.max;
        } else {
            self.cur_transition_max = -1;
        }
    }

    /// `load(output)`: the floor data of `output` (skipping the followers
    /// below the first transition unless the state accepts), then the block.
    fn load(
        &mut self,
        output: Option<&Output>,
        tio: &[u8],
        runnable: Option<ByteRunnableRef<'_>>,
    ) -> Result<()> {
        if let (Some(output), Some(runnable)) = (output, runnable) {
            if self.transition_count != 0 {
                self.floor_data.clear();
                self.floor_data.extend_from_slice(&output.bytes);
                self.floor_pos = 0;
                let code = read_vlong(&self.floor_data, &mut self.floor_pos)?;
                if code & OUTPUT_FLAG_IS_FLOOR != 0 {
                    self.num_follow_floor_blocks =
                        read_vint(&self.floor_data, &mut self.floor_pos)?;
                    // SENTINEL-OK: this module's `read_byte` reads an
                    // unsigned floor label (0..=255) or errors.
                    self.next_floor_label = read_byte(&self.floor_data, &mut self.floor_pos)?;
                    read_vlong(&self.floor_data, &mut self.floor_pos)?;
                    if !runnable.is_accept(self.state) {
                        while self.num_follow_floor_blocks > 0
                            && self.next_floor_label <= self.transition.min
                        {
                            self.skip_floor_block()?;
                        }
                    }
                }
            }
        }
        let block = read_block(
            tio,
            self.fp,
            &mut self.suffix_bytes,
            &mut self.stat_bytes,
            &mut self.bytes,
        )?;
        self.ent_count = block.ent_count;
        self.is_last_in_floor = block.is_last_in_floor;
        self.is_leaf_block = block.is_leaf_block;
        self.suffix_pos = 0;
        self.stat_pos = 0;
        self.bytes_pos = 0;
        self.meta_data_upto = 0;
        self.term_state.term_block_ord = 0;
        self.next_ent = 0;
        if !self.is_last_in_floor {
            self.fp_end = block.fp_end;
        }
        Ok(())
    }

    /// `next()`: `true` when the entry is a sub-block.
    // ARITH: `next_ent < ent_count`, an i32; a suffix checked against its
    // block's bytes.
    #[allow(clippy::arithmetic_side_effects)]
    fn next(&mut self) -> Result<bool> {
        if self.next_ent < 0 || self.next_ent >= self.ent_count {
            // Java asserts `nextEnt != -1 && nextEnt < entCount`.
            return Err(corrupt("read past the last entry of a block"));
        }
        self.next_ent += 1;
        let code = read_vint(&self.suffix_bytes, &mut self.suffix_pos)?;
        self.suffix = length(if self.is_leaf_block {
            code
        } else {
            ushr32(code, 1)
        })?;
        self.start_byte_pos = self.suffix_pos;
        if self.suffix > self.suffix_bytes.len() - self.suffix_pos {
            return Err(corrupt("suffix past the end of its block"));
        }
        self.suffix_pos += self.suffix;
        if self.is_leaf_block || code & 1 == 0 {
            if !self.is_leaf_block {
                self.term_state.term_block_ord += 1;
            }
            Ok(false)
        } else {
            self.last_sub_fp = self
                .fp
                .wrapping_sub(read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?);
            // The sub-block's term count, which this enum does not track.
            read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
            Ok(true)
        }
    }

    /// The current entry's suffix.
    // ARITH: `next` checked the span.
    #[allow(clippy::arithmetic_side_effects)]
    fn suffix_slice(&self) -> &[u8] {
        &self.suffix_bytes[self.start_byte_pos..self.start_byte_pos + self.suffix]
    }

    /// `decodeMetaData()`.
    fn decode_meta_data(&mut self, fr: &OrdsFieldReader) -> Result<()> {
        let limit = if self.is_leaf_block {
            self.next_ent
        } else {
            self.term_state.term_block_ord
        };
        decode_block_meta(
            fr,
            &self.stat_bytes,
            &mut self.stat_pos,
            &self.bytes,
            &mut self.bytes_pos,
            &mut self.meta_data_upto,
            limit,
            &mut self.term_state,
        )
    }
}

/// `OrdsIntersectTermsEnum`: the terms of a field a `NORMAL` compiled
/// automaton accepts, in order. It has no ords: `ord()` and the seeks are
/// Java's `UnsupportedOperationException`.
pub struct OrdsIntersectTermsEnum<'a> {
    fr: &'a OrdsFieldReader,
    runnable: ByteRunnableRef<'a>,
    accessor: &'a dyn TransitionAccessor,
    common_suffix: Option<&'a [u8]>,
    stack: Vec<Frame>,
    current: usize,
    term: TermBuf,
}

impl std::fmt::Debug for OrdsIntersectTermsEnum<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdsIntersectTermsEnum")
            .field("field", &self.fr.info.name)
            .field("term", &self.term.get())
            .field("current", &self.current)
            .finish()
    }
}

impl<'a> OrdsIntersectTermsEnum<'a> {
    /// `new OrdsIntersectTermsEnum(fr, compiled, startTerm)`, the automaton
    /// already taken apart by [`OrdsFieldReader::intersect`].
    pub(super) fn new(
        fr: &'a OrdsFieldReader,
        runnable: ByteRunnableRef<'a>,
        accessor: &'a dyn TransitionAccessor,
        common_suffix: Option<&'a [u8]>,
        start_term: Option<&[u8]>,
    ) -> Result<Self> {
        let arc = fr.index.first_arc();
        let output_prefix = arc.output().clone();
        let mut f = Frame::new(0, arc);
        f.fp = fr.root_block_fp;
        f.fp_orig = fr.root_block_fp;
        f.prefix = 0;
        f.set_state(0, accessor);
        f.output_prefix = output_prefix;
        f.load(Some(&fr.root_code), &fr.tio, Some(runnable))?;
        let mut this = OrdsIntersectTermsEnum {
            fr,
            runnable,
            accessor,
            common_suffix,
            stack: vec![f],
            current: 0,
            term: TermBuf::default(),
        };
        if let Some(start) = start_term {
            this.seek_to_start_term(start)?;
        }
        Ok(this)
    }

    fn cur(&self) -> &Frame {
        &self.stack[self.current]
    }

    fn cur_mut(&mut self) -> &mut Frame {
        &mut self.stack[self.current]
    }

    /// `pushFrame(state)`: the frame of the current sub-block entry, its
    /// arc and output found by walking the terms index from the current
    /// frame's prefix to the sub-block's.
    // ARITH: a frame ord is at most the longest term's length; prefix plus
    // suffix of one term.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_frame(&mut self, state: i32) -> Result<()> {
        let ord = self.current + 1;
        let (last_sub_fp, prefix, mut arc, mut output, mut idx) = {
            let c = self.cur();
            (
                c.last_sub_fp,
                c.prefix + c.suffix,
                c.arc.clone(),
                c.output_prefix.clone(),
                c.prefix,
            )
        };
        let mut reader = self.fr.index.bytes_reader();
        while idx < prefix {
            let label = i32::from(self.term.byte_at(idx));
            arc = self
                .fr
                .index
                .find_target_arc(label, &arc, &mut reader)?
                .ok_or_else(|| corrupt("sub-block prefix missing from the terms index"))?;
            FstOrdsOutputs::add_assign(&mut output, arc.output());
            idx += 1;
        }
        if !arc.is_final() {
            // Java asserts the sub-block's arc is final.
            return Err(corrupt("sub-block without a terms index entry"));
        }
        while self.stack.len() <= ord {
            let n = self.stack.len();
            self.stack.push(Frame::new(n, self.fr.index.first_arc()));
        }
        let data = FstOrdsOutputs::add(&output, arc.next_final_output());
        let f = &mut self.stack[ord];
        f.fp = last_sub_fp;
        f.fp_orig = last_sub_fp;
        f.prefix = prefix;
        f.set_state(state, self.accessor);
        f.arc = arc;
        f.output_prefix = output;
        f.load(Some(&data), &self.fr.tio, Some(self.runnable))?;
        self.current = ord;
        Ok(())
    }

    /// `term()`.
    pub fn term(&self) -> &[u8] {
        self.term.get()
    }

    /// `docFreq()`.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn doc_freq(&mut self) -> Result<i32> {
        Ok(self.term_state()?.doc_freq)
    }

    /// `totalTermFreq()`.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn total_term_freq(&mut self) -> Result<i64> {
        Ok(self.term_state()?.total_term_freq)
    }

    /// The current term's postings metadata.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn term_metadata(&mut self) -> Result<TermMetadata> {
        Ok(self.term_state()?.meta)
    }

    /// `termState()`.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn term_state(&mut self) -> Result<OrdsTermState> {
        let fr = self.fr;
        let f = self.cur_mut();
        f.decode_meta_data(fr)?;
        Ok(f.term_state)
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// What decoding the metadata or the `.doc` fails with.
    pub fn postings(&mut self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        let state = self.term_state()?;
        self.fr.postings(&state, doc_in, flags)
    }

    /// `ord()`: unsupported.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn ord(&self) -> Result<i64> {
        Err(Error::Unsupported("ord() on an intersect terms enum"))
    }

    /// `seekExact(ord)`, `seekExact(text)` and `seekCeil(text)`:
    /// unsupported.
    ///
    /// # Errors
    /// Always [`Error::Unsupported`].
    pub fn seek(&mut self) -> Result<()> {
        Err(Error::Unsupported("seeking an intersect terms enum"))
    }

    /// `getState()`: the automaton's state after the current suffix.
    fn suffix_state(&self) -> Result<i32> {
        let f = self.cur();
        let mut state = f.state;
        for &b in f.suffix_slice() {
            state = self.runnable.step(state, i32::from(b));
            if state == -1 {
                // Java asserts the state is live.
                return Err(corrupt("start term's prefix leaves the automaton"));
            }
        }
        Ok(state)
    }

    /// `copyTerm()`.
    fn copy_term(&mut self) {
        let f = &self.stack[self.current];
        self.term.set_suffix(f.prefix, f.suffix_slice());
    }

    /// `seekToStartTerm(target)`: positions just before the first term
    /// after `target` (or on `target`), descending into the sub-blocks
    /// whose prefixes `target` starts with.
    fn seek_to_start_term(&mut self, target: &[u8]) -> Result<()> {
        self.term.grow(target.len());
        for _ in 0..=target.len() {
            loop {
                let (save_pos, save_start, save_suffix, save_last_sub_fp, save_tbo) = {
                    let f = self.cur();
                    (
                        f.suffix_pos,
                        f.start_byte_pos,
                        f.suffix,
                        f.last_sub_fp,
                        f.term_state.term_block_ord,
                    )
                };
                let is_sub_block = self.cur_mut().next()?;
                self.copy_term();
                if is_sub_block && target.starts_with(self.term.get()) {
                    let state = self.suffix_state()?;
                    self.push_frame(state)?;
                    break;
                }
                match self.term.get().cmp(target) {
                    std::cmp::Ordering::Less => {
                        let f = self.cur();
                        if f.next_ent == f.ent_count {
                            if f.is_last_in_floor {
                                return Ok(());
                            }
                            let tio = &self.fr.tio;
                            self.stack[self.current].load_next_floor_block(tio)?;
                        }
                    }
                    std::cmp::Ordering::Equal => return Ok(()),
                    std::cmp::Ordering::Greater => {
                        let f = self.cur_mut();
                        f.next_ent = f.next_ent.saturating_sub(1);
                        f.last_sub_fp = save_last_sub_fp;
                        f.start_byte_pos = save_start;
                        f.suffix = save_suffix;
                        f.suffix_pos = save_pos;
                        f.term_state.term_block_ord = save_tbo;
                        self.copy_term();
                        return Ok(());
                    }
                }
            }
        }
        // Java asserts this is unreachable.
        Err(corrupt("start term seek ran past the term's length"))
    }

    /// Whether the current term entry (not a sub-block) ends with the
    /// automaton's common suffix.
    // ARITH: positions within the term, all checked against
    // `common.len() <= prefix + suffix`.
    #[allow(clippy::arithmetic_side_effects)]
    fn has_common_suffix(&self, common: &[u8]) -> bool {
        let f = self.cur();
        if f.prefix + f.suffix < common.len() {
            return false;
        }
        let suffix = f.suffix_slice();
        if common.len() > f.suffix {
            let in_prefix = common.len() - f.suffix;
            let start = f.prefix - in_prefix;
            (0..in_prefix).all(|i| self.term.byte_at(start + i) == common[i])
                && suffix == &common[in_prefix..]
        } else {
            suffix[f.suffix - common.len()..] == *common
        }
    }

    /// `next()`: the next accepted term, or `None` past the last.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or terms index.
    // ARITH: transition indices below `transition_count`; a frame ord above
    // 0, minus one.
    #[allow(clippy::arithmetic_side_effects, clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        'next_term: loop {
            while self.cur().next_ent == self.cur().ent_count {
                if !self.cur().is_last_in_floor {
                    let tio = &self.fr.tio;
                    self.stack[self.current].load_next_floor_block(tio)?;
                } else {
                    if self.cur().ord == 0 {
                        return Ok(None);
                    }
                    self.current -= 1;
                }
            }
            let is_sub_block = self.cur_mut().next()?;
            if self.cur().suffix != 0 {
                let label = i32::from(self.cur().suffix_slice()[0]);
                while label > self.cur().cur_transition_max {
                    let accessor = self.accessor;
                    let f = self.cur_mut();
                    if f.transition_index >= f.transition_count - 1 {
                        f.is_last_in_floor = true;
                        f.next_ent = f.ent_count;
                        continue 'next_term;
                    }
                    f.transition_index += 1;
                    accessor.get_next_transition(&mut f.transition);
                    f.cur_transition_max = f.transition.max;
                }
            }
            if let Some(common) = self.common_suffix {
                if !is_sub_block && !self.has_common_suffix(common) {
                    continue 'next_term;
                }
            }
            let mut state = self.cur().state;
            for &b in self.cur().suffix_slice() {
                state = self.runnable.step(state, i32::from(b));
                if state == -1 {
                    continue 'next_term;
                }
            }
            if is_sub_block {
                self.copy_term();
                self.push_frame(state)?;
            } else if self.runnable.is_accept(state) {
                self.copy_term();
                return Ok(Some(self.term.get()));
            }
        }
    }
}
