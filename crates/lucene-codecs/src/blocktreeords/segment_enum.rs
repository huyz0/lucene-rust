//! Port of `OrdsSegmentTermsEnum` and `OrdsSegmentTermsEnumFrame`: the
//! Lucene 4 block-tree terms enum, each frame also counting the ords of the
//! terms it has passed (`termOrd`), so `ord()` is the current frame's count
//! minus one and `seekExact(long ord)` finds a term by walking the terms
//! index along the ord ranges of its outputs.
//!
//! Java's `BytesRefBuilder` keeps the bytes past its length, and the seek
//! compares against them (a re-seek reads `term.byteAt(targetUpto)` beyond
//! the current term); [`TermBuf`] keeps them too.

use std::cmp::Ordering;

use lucene_store::data_input::SliceInput;

use super::outputs::{FstOrdsOutputs, Output};
use super::{
    corrupt, length, read_block, read_byte, read_vint, read_vlong, ushr32, ushr64, Error,
    OrdsFieldReader, Result, SeekStatus, OUTPUT_FLAGS_NUM_BITS, OUTPUT_FLAG_HAS_TERMS,
    OUTPUT_FLAG_IS_FLOOR,
};
use crate::field_infos::IndexOptions;
use crate::fst::{ARCS_FOR_BINARY_SEARCH, ARCS_FOR_DIRECT_ADDRESSING, BIT_ARC_HAS_OUTPUT};
use crate::fst_compiler::FstOutputs;
use crate::fst_util::{target_has_arcs, Arc};
use crate::postings::{self, DocInput, Postings, PostingsFlags, TermMetadata};

/// `BlockTermState` as the ords block tree fills it: the term's statistics,
/// its place in its block, its ord and its postings metadata.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct OrdsTermState {
    /// `docFreq`.
    pub doc_freq: i32,
    /// `totalTermFreq` (`docFreq` for a field without frequencies; -1
    /// before the first decode).
    pub total_term_freq: i64,
    /// `termBlockOrd`: the term's index among its block's terms, plus one.
    pub term_block_ord: i32,
    /// `ord` (`OrdTermState.ord`), set by `termState()`.
    pub ord: i64,
    /// `Lucene104PostingsReader.IntBlockTermState`'s file pointers.
    pub meta: TermMetadata,
}

impl Default for OrdsTermState {
    /// `postingsReader.newTermState()` with `totalTermFreq = -1`.
    fn default() -> Self {
        OrdsTermState {
            doc_freq: 0,
            total_term_freq: -1,
            term_block_ord: 0,
            ord: 0,
            meta: TermMetadata::EMPTY,
        }
    }
}

/// `BytesRefBuilder`: `len` bytes of term, the bytes past it kept.
#[derive(Debug, Default)]
pub(super) struct TermBuf {
    bytes: Vec<u8>,
    pub(super) len: usize,
}

impl TermBuf {
    /// `grow(n)`.
    pub(super) fn grow(&mut self, n: usize) {
        if self.bytes.len() < n {
            self.bytes.resize(n, 0);
        }
    }

    /// `get()`.
    pub(super) fn get(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// `byteAt(i)`: past the array (which Java's grow-before-seek rules
    /// out) reads 0.
    pub(super) fn byte_at(&self, i: usize) -> u8 {
        self.bytes.get(i).copied().unwrap_or(0)
    }

    /// `setByteAt(i, b)`.
    fn set_byte_at(&mut self, i: usize, b: u8) {
        self.grow(i.saturating_add(1));
        self.bytes[i] = b;
    }

    /// `setLength(n)` + `grow(n)`.
    fn set_length(&mut self, n: usize) {
        self.grow(n);
        self.len = n;
    }

    /// `copyBytes(src)`.
    pub(super) fn copy_bytes(&mut self, src: &[u8]) {
        self.grow(src.len());
        self.bytes[..src.len()].copy_from_slice(src);
        self.len = src.len();
    }

    /// `suffixesReader.readBytes(term.bytes(), prefix, suffix)` and
    /// `fillTerm`'s copy: `prefix + src.len()` bytes long afterwards.
    // ARITH: a prefix plus a suffix of one term, both from a block.
    #[allow(clippy::arithmetic_side_effects)]
    pub(super) fn set_suffix(&mut self, prefix: usize, src: &[u8]) {
        self.set_length(prefix + src.len());
        self.bytes[prefix..prefix + src.len()].copy_from_slice(src);
    }
}

/// `decodeMetaData()` of either enum's frame: the stats and
/// `postingsReader.decodeTerm` of every term of the block up to `limit`.
#[allow(clippy::too_many_arguments)]
pub(super) fn decode_block_meta(
    fr: &OrdsFieldReader,
    stat_bytes: &[u8],
    stat_pos: &mut usize,
    bytes: &[u8],
    bytes_pos: &mut usize,
    meta_data_upto: &mut i32,
    limit: i32,
    state: &mut OrdsTermState,
) -> Result<()> {
    let mut absolute = *meta_data_upto == 0;
    let index_options = fr.info.index_options;
    while *meta_data_upto < limit {
        state.doc_freq = read_vint(stat_bytes, stat_pos)?;
        state.total_term_freq = if index_options == IndexOptions::Docs {
            i64::from(state.doc_freq)
        } else {
            i64::from(state.doc_freq).wrapping_add(read_vlong(stat_bytes, stat_pos)?)
        };
        let mut input = SliceInput::new(bytes);
        input.seek(*bytes_pos)?;
        state.meta = postings::decode_term_metadata(
            &mut input,
            state.doc_freq,
            absolute,
            state.meta,
            index_options,
            fr.info.store_payloads,
            state.total_term_freq,
        )?;
        *bytes_pos = input.position();
        *meta_data_upto = meta_data_upto.saturating_add(1);
        absolute = false;
    }
    state.term_block_ord = *meta_data_upto;
    Ok(())
}

/// `OrdsSegmentTermsEnumFrame`.
#[derive(Debug, Clone, Default)]
struct Frame {
    ord: i32,
    has_terms: bool,
    has_terms_orig: bool,
    is_floor: bool,
    fp: i64,
    fp_orig: i64,
    fp_end: i64,
    /// `suffixBytes`, exactly the block's.
    suffix_bytes: Vec<u8>,
    /// `suffixesReader`'s position.
    suffix_pos: usize,
    /// `statBytes`, exactly the block's.
    stat_bytes: Vec<u8>,
    /// `statsReader`'s position.
    stat_pos: usize,
    /// `floorData`, exactly the frame's.
    floor_data: Vec<u8>,
    /// `floorDataReader`'s position.
    floor_pos: usize,
    prefix_length: usize,
    ent_count: i32,
    next_ent: i32,
    term_ord_orig: i64,
    term_ord: i64,
    is_last_in_floor: bool,
    is_leaf_block: bool,
    last_sub_fp: i64,
    next_floor_label: i32,
    next_floor_term_ord: i64,
    num_follow_floor_blocks: i32,
    meta_data_upto: i32,
    state: OrdsTermState,
    /// `bytes`: the block's term metadata.
    bytes: Vec<u8>,
    /// `bytesReader`'s position.
    bytes_pos: usize,
    start_byte_pos: usize,
    suffix_length: usize,
    sub_code: i64,
}

impl Frame {
    fn new(ord: i32) -> Self {
        Frame {
            ord,
            ..Frame::default()
        }
    }

    /// The floor data's header: followers, the next lead label and the ord
    /// it starts at.
    fn read_floor_header(&mut self) -> Result<()> {
        self.floor_pos = 0;
        self.num_follow_floor_blocks = read_vint(&self.floor_data, &mut self.floor_pos)?;
        // SENTINEL-OK: this module's `read_byte` reads an unsigned floor
        // label (0..=255) or errors; it has no `-1` sentinel.
        self.next_floor_label = read_byte(&self.floor_data, &mut self.floor_pos)?;
        self.next_floor_term_ord = self
            .term_ord_orig
            .wrapping_add(read_vlong(&self.floor_data, &mut self.floor_pos)?);
        Ok(())
    }

    /// `setFloorData(in, source)`: the bytes of `source` after the code.
    fn set_floor_data(&mut self, rest: &[u8]) -> Result<()> {
        self.floor_data.clear();
        self.floor_data.extend_from_slice(rest);
        self.read_floor_header()
    }

    /// `getTermBlockOrd()`.
    fn term_block_ord(&self) -> i32 {
        if self.is_leaf_block {
            self.next_ent
        } else {
            self.state.term_block_ord
        }
    }

    /// `loadBlock()`, after `ste.initIndexInput()`.
    fn load_block(&mut self, tio: &[u8]) -> Result<()> {
        if self.next_ent != -1 {
            return Ok(());
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
        self.meta_data_upto = 0;
        self.state.term_block_ord = 0;
        self.next_ent = 0;
        self.last_sub_fp = -1;
        self.bytes_pos = 0;
        self.fp_end = block.fp_end;
        Ok(())
    }

    /// `rewind()`.
    fn rewind(&mut self) -> Result<()> {
        self.fp = self.fp_orig;
        self.term_ord = self.term_ord_orig;
        self.next_ent = -1;
        self.has_terms = self.has_terms_orig;
        if self.is_floor {
            self.read_floor_header()?;
        }
        Ok(())
    }

    /// The next `n` suffix bytes, the reader moved past them.
    // ARITH: a position within `suffix_bytes` plus a length checked against it.
    #[allow(clippy::arithmetic_side_effects)]
    fn skip_suffix(&mut self, n: usize) -> Result<usize> {
        let start = self.suffix_pos;
        if n > self.suffix_bytes.len() - start {
            return Err(corrupt("suffix past the end of its block"));
        }
        self.suffix_pos = start + n;
        Ok(start)
    }

    /// `next()`: `true` when the entry is a sub-block.
    fn next(&mut self, term: &mut TermBuf, term_exists: &mut bool) -> Result<bool> {
        if self.next_ent < 0 || self.next_ent >= self.ent_count {
            // Java asserts `nextEnt != -1 && nextEnt < entCount`.
            return Err(corrupt("read past the last entry of a block"));
        }
        if self.is_leaf_block {
            self.next_leaf(term, term_exists)
        } else {
            self.next_non_leaf(term, term_exists)
        }
    }

    /// `nextLeaf()`.
    // ARITH: `next_ent < ent_count`, an i32.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_leaf(&mut self, term: &mut TermBuf, term_exists: &mut bool) -> Result<bool> {
        self.next_ent += 1;
        self.term_ord = self.term_ord.wrapping_add(1);
        self.suffix_length = length(read_vint(&self.suffix_bytes, &mut self.suffix_pos)?)?;
        self.start_byte_pos = self.skip_suffix(self.suffix_length)?;
        term.set_suffix(self.prefix_length, self.suffix());
        *term_exists = true;
        Ok(false)
    }

    /// `nextNonLeaf()`.
    // ARITH: `next_ent < ent_count`, an i32.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_non_leaf(&mut self, term: &mut TermBuf, term_exists: &mut bool) -> Result<bool> {
        self.next_ent += 1;
        let code = read_vint(&self.suffix_bytes, &mut self.suffix_pos)?;
        self.suffix_length = length(ushr32(code, 1))?;
        self.start_byte_pos = self.skip_suffix(self.suffix_length)?;
        term.set_suffix(self.prefix_length, self.suffix());
        if code & 1 == 0 {
            *term_exists = true;
            self.sub_code = 0;
            self.state.term_block_ord += 1;
            self.term_ord = self.term_ord.wrapping_add(1);
            Ok(false)
        } else {
            *term_exists = false;
            self.sub_code = read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
            let count = read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
            self.term_ord = self.term_ord.wrapping_add(count);
            self.last_sub_fp = self.fp.wrapping_sub(self.sub_code);
            Ok(true)
        }
    }

    /// The current entry's suffix.
    // ARITH: `skip_suffix` checked the span.
    #[allow(clippy::arithmetic_side_effects)]
    fn suffix(&self) -> &[u8] {
        &self.suffix_bytes[self.start_byte_pos..self.start_byte_pos + self.suffix_length]
    }

    /// The loop shared by both `scanToFloorFrame`s: moves along the floor
    /// followers while `past(nextFloorLabel, nextFloorTermOrd)`.
    // ARITH: `num_follow_floor_blocks` counts down from the vInt that bounds
    // the loop.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_floor(&mut self, past: impl Fn(i32, i64) -> bool) -> Result<()> {
        if self.num_follow_floor_blocks <= 0 {
            // Java asserts `numFollowFloorBlocks != 0`.
            return Err(corrupt("floor block without followers"));
        }
        let mut last_floor_term_ord;
        let mut new_fp;
        loop {
            let code = read_vlong(&self.floor_data, &mut self.floor_pos)?;
            new_fp = self.fp_orig.wrapping_add(ushr64(code, 1));
            self.has_terms = code & 1 != 0;
            self.is_last_in_floor = self.num_follow_floor_blocks == 1;
            self.num_follow_floor_blocks -= 1;
            last_floor_term_ord = self.next_floor_term_ord;
            if self.is_last_in_floor {
                self.next_floor_label = 256;
                self.next_floor_term_ord = i64::MAX;
                break;
            }
            // SENTINEL-OK: this module's `read_byte` reads an unsigned floor
            // label (0..=255) or errors; it has no `-1` sentinel.
            self.next_floor_label = read_byte(&self.floor_data, &mut self.floor_pos)?;
            self.next_floor_term_ord = self
                .next_floor_term_ord
                .wrapping_add(read_vlong(&self.floor_data, &mut self.floor_pos)?);
            if !past(self.next_floor_label, self.next_floor_term_ord) {
                break;
            }
        }
        if new_fp != self.fp {
            self.next_ent = -1;
            self.term_ord = last_floor_term_ord;
            self.fp = new_fp;
        }
        Ok(())
    }

    /// `scanToFloorFrame(BytesRef target)`.
    fn scan_to_floor_frame(&mut self, target: &[u8]) -> Result<()> {
        if !self.is_floor || target.len() <= self.prefix_length {
            return Ok(());
        }
        let target_label = i32::from(target[self.prefix_length]);
        if target_label < self.next_floor_label {
            return Ok(());
        }
        self.scan_floor(|label, _| target_label >= label)
    }

    /// `scanToFloorFrame(long targetOrd)`.
    fn scan_to_floor_frame_ord(&mut self, target_ord: i64) -> Result<()> {
        if !self.is_floor || target_ord < self.next_floor_term_ord {
            return Ok(());
        }
        self.scan_floor(|_, ord| target_ord >= ord)
    }

    /// `decodeMetaData()`.
    fn decode_meta_data(&mut self, fr: &OrdsFieldReader) -> Result<()> {
        let limit = self.term_block_ord();
        decode_block_meta(
            fr,
            &self.stat_bytes,
            &mut self.stat_pos,
            &self.bytes,
            &mut self.bytes_pos,
            &mut self.meta_data_upto,
            limit,
            &mut self.state,
        )
    }

    /// `scanToSubBlock(subFP)`.
    // ARITH: entry counts within a block.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_sub_block(&mut self, sub_fp: i64) -> Result<()> {
        if self.last_sub_fp == sub_fp {
            return Ok(());
        }
        let target_sub_code = self.fp.wrapping_sub(sub_fp);
        loop {
            if self.next_ent >= self.ent_count {
                // Java asserts `nextEnt < entCount`.
                return Err(corrupt("sub-block missing from its parent"));
            }
            self.next_ent += 1;
            let code = read_vint(&self.suffix_bytes, &mut self.suffix_pos)?;
            let skip = if self.is_leaf_block {
                code
            } else {
                ushr32(code, 1)
            };
            self.skip_suffix(length(skip)?)?;
            if code & 1 != 0 {
                let sub_code = read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
                let count = read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
                self.term_ord = self.term_ord.wrapping_add(count);
                if target_sub_code == sub_code {
                    self.last_sub_fp = sub_fp;
                    return Ok(());
                }
            } else {
                self.state.term_block_ord += 1;
                self.term_ord = self.term_ord.wrapping_add(1);
            }
        }
    }

    /// The current entry's suffix against `target` past the prefix.
    fn compare_suffix(&self, target: &[u8]) -> Ordering {
        self.suffix()
            .cmp(target.get(self.prefix_length..).unwrap_or(&[]))
    }

    /// `fillTerm()`.
    fn fill_term(&self, term: &mut TermBuf) {
        term.set_suffix(self.prefix_length, self.suffix());
    }

    /// `scanToTermLeaf(target, exactOnly)`.
    // ARITH: `next_ent < ent_count`, an i32.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_term_leaf(
        &mut self,
        target: &[u8],
        exact_only: bool,
        term: &mut TermBuf,
        term_exists: &mut bool,
    ) -> Result<SeekStatus> {
        *term_exists = true;
        self.sub_code = 0;
        if self.next_ent == self.ent_count {
            if exact_only {
                self.fill_term(term);
            }
            return Ok(SeekStatus::End);
        }
        loop {
            self.next_ent += 1;
            self.term_ord = self.term_ord.wrapping_add(1);
            self.suffix_length = length(read_vint(&self.suffix_bytes, &mut self.suffix_pos)?)?;
            self.start_byte_pos = self.skip_suffix(self.suffix_length)?;
            match self.compare_suffix(target) {
                Ordering::Less => {}
                Ordering::Greater => {
                    self.fill_term(term);
                    return Ok(SeekStatus::NotFound);
                }
                Ordering::Equal => {
                    self.fill_term(term);
                    return Ok(SeekStatus::Found);
                }
            }
            if self.next_ent >= self.ent_count {
                break;
            }
        }
        if exact_only {
            self.fill_term(term);
        }
        Ok(SeekStatus::End)
    }

    /// `scanToTermNonLeaf(target, exactOnly)`, but for its descent into the
    /// sub-block after a `NOT_FOUND` (which needs the enum's stack): the
    /// `Some(prevTermOrd)` asks the caller for it.
    // ARITH: `next_ent < ent_count`, an i32.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_term_non_leaf(
        &mut self,
        target: &[u8],
        exact_only: bool,
        term: &mut TermBuf,
        term_exists: &mut bool,
    ) -> Result<(SeekStatus, Option<i64>)> {
        if self.next_ent == self.ent_count {
            if exact_only {
                self.fill_term(term);
                *term_exists = self.sub_code == 0;
            }
            return Ok((SeekStatus::End, None));
        }
        while self.next_ent < self.ent_count {
            self.next_ent += 1;
            let code = read_vint(&self.suffix_bytes, &mut self.suffix_pos)?;
            self.suffix_length = length(ushr32(code, 1))?;
            *term_exists = code & 1 == 0;
            self.start_byte_pos = self.skip_suffix(self.suffix_length)?;
            let prev_term_ord = self.term_ord;
            if *term_exists {
                self.state.term_block_ord += 1;
                self.term_ord = self.term_ord.wrapping_add(1);
                self.sub_code = 0;
            } else {
                self.sub_code = read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
                let count = read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
                self.term_ord = self.term_ord.wrapping_add(count);
                self.last_sub_fp = self.fp.wrapping_sub(self.sub_code);
            }
            match self.compare_suffix(target) {
                Ordering::Less => {}
                Ordering::Greater => {
                    self.fill_term(term);
                    let descend = (!exact_only && !*term_exists).then_some(prev_term_ord);
                    return Ok((SeekStatus::NotFound, descend));
                }
                Ordering::Equal => {
                    self.fill_term(term);
                    return Ok((SeekStatus::Found, None));
                }
            }
        }
        if exact_only {
            self.fill_term(term);
        }
        Ok((SeekStatus::End, None))
    }
}

/// `OrdsSegmentTermsEnum`: one field's terms, in order, with their ords.
pub struct OrdsSegmentTermsEnum<'a> {
    fr: &'a OrdsFieldReader,
    /// `in != null`: some frame has loaded a block.
    in_initialized: bool,
    stack: Vec<Frame>,
    static_frame: Frame,
    /// `currentFrame`: `None` is `staticFrame`.
    current: Option<usize>,
    term_exists: bool,
    target_before_current_length: i32,
    valid_index_prefix: usize,
    term: TermBuf,
    arcs: Vec<Arc<Output>>,
    /// `positioned`: the frames describe the current term.
    positioned: bool,
    /// The output accumulated along the arcs walked so far (Java's
    /// `output`), grown in place.
    output: Output,
}

impl std::fmt::Debug for OrdsSegmentTermsEnum<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrdsSegmentTermsEnum")
            .field("field", &self.fr.info.name)
            .field("term", &self.term.get())
            .field("current", &self.current)
            .finish()
    }
}

impl<'a> OrdsSegmentTermsEnum<'a> {
    /// `new OrdsSegmentTermsEnum(fr)`.
    pub(super) fn new(fr: &'a OrdsFieldReader) -> Self {
        OrdsSegmentTermsEnum {
            fr,
            in_initialized: false,
            stack: Vec::new(),
            static_frame: Frame::new(-1),
            current: None,
            term_exists: false,
            target_before_current_length: 0,
            valid_index_prefix: 0,
            term: TermBuf::default(),
            arcs: vec![fr.index.first_arc()],
            positioned: false,
            output: Output::NO_OUTPUT,
        }
    }

    fn frame(&self, c: Option<usize>) -> &Frame {
        match c {
            Some(i) => &self.stack[i],
            None => &self.static_frame,
        }
    }

    fn frame_mut(&mut self, c: Option<usize>) -> &mut Frame {
        match c {
            Some(i) => &mut self.stack[i],
            None => &mut self.static_frame,
        }
    }

    fn cur(&self) -> &Frame {
        self.frame(self.current)
    }

    fn cur_mut(&mut self) -> &mut Frame {
        self.frame_mut(self.current)
    }

    /// `getFrame(ord)`'s index: makes slot `ord` exist.
    fn get_frame(&mut self, ord: usize) -> usize {
        while self.stack.len() <= ord {
            let n = i32::try_from(self.stack.len()).unwrap_or(i32::MAX);
            self.stack.push(Frame::new(n));
        }
        ord
    }

    /// `getFrame(1 + currentFrame.ord)`'s index.
    // ARITH: a frame ord is at most the longest term's length.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_frame_index(&mut self) -> usize {
        let ord = (self.cur().ord + 1) as usize;
        self.get_frame(ord)
    }

    /// `getArc(ord)`: makes slot `ord` exist.
    fn get_arc(&mut self, ord: usize) {
        while self.arcs.len() <= ord {
            self.arcs.push(self.fr.index.first_arc());
        }
    }

    /// `pushFrame(arc, frameData, length)`.
    fn push_frame_data(&mut self, data: &Output, length: usize) -> Result<usize> {
        let mut pos = 0;
        let code = read_vlong(&data.bytes, &mut pos)?;
        let fp_seek = ushr64(code, OUTPUT_FLAGS_NUM_BITS);
        let idx = self.next_frame_index();
        let f = &mut self.stack[idx];
        f.has_terms = code & OUTPUT_FLAG_HAS_TERMS != 0;
        f.has_terms_orig = f.has_terms;
        f.is_floor = code & OUTPUT_FLAG_IS_FLOOR != 0;
        if f.is_floor {
            f.term_ord_orig = data.start_ord;
            f.set_floor_data(data.bytes.get(pos..).unwrap_or_default())?;
        }
        self.push_frame_fp(fp_seek, length, data.start_ord)
    }

    /// `pushFrame(arc, fstOutputs.add(output, arc.nextFinalOutput()), length)`
    /// for final arc `arc_idx`.
    fn push_final_frame(&mut self, arc_idx: usize, length: usize) -> Result<usize> {
        let data = FstOrdsOutputs::add(&self.output, self.arcs[arc_idx].next_final_output());
        self.push_frame_data(&data, length)
    }

    /// `pushFrame(arc, fp, length, termOrd)`.
    fn push_frame_fp(&mut self, fp: i64, length: usize, term_ord: i64) -> Result<usize> {
        let idx = self.next_frame_index();
        let target_before = i64::from(self.target_before_current_length);
        let f = &mut self.stack[idx];
        if f.fp_orig == fp && f.next_ent != -1 {
            if f.prefix_length as i64 > target_before {
                f.rewind()?;
            }
        } else {
            f.next_ent = -1;
            f.prefix_length = length;
            f.state.term_block_ord = 0;
            f.term_ord_orig = term_ord;
            f.term_ord = term_ord;
            f.fp_orig = fp;
            f.fp = fp;
            f.last_sub_fp = -1;
        }
        Ok(idx)
    }

    /// `loadBlock()` of the current frame.
    fn load_block(&mut self) -> Result<()> {
        self.in_initialized = true;
        let tio = &self.fr.tio;
        let f = match self.current {
            Some(i) => &mut self.stack[i],
            None => &mut self.static_frame,
        };
        f.load_block(tio)
    }

    /// `currentFrame.next()`.
    fn frame_next(&mut self) -> Result<bool> {
        let c = self.current;
        let OrdsSegmentTermsEnum {
            stack,
            static_frame,
            term,
            term_exists,
            ..
        } = &mut *self;
        let f = match c {
            Some(i) => &mut stack[i],
            None => static_frame,
        };
        f.next(term, term_exists)
    }

    /// `currentFrame.scanToTerm(target, exactOnly)`, with the non-leaf
    /// descent into the next term's sub-block.
    fn scan_to_term(&mut self, target: &[u8], exact_only: bool) -> Result<SeekStatus> {
        let c = self.current;
        let OrdsSegmentTermsEnum {
            stack,
            static_frame,
            term,
            term_exists,
            ..
        } = &mut *self;
        let f = match c {
            Some(i) => &mut stack[i],
            None => static_frame,
        };
        if f.is_leaf_block {
            return f.scan_to_term_leaf(target, exact_only, term, term_exists);
        }
        let (status, descend) = f.scan_to_term_non_leaf(target, exact_only, term, term_exists)?;
        if let Some(prev_term_ord) = descend {
            // ARITH: a prefix plus a suffix of one term.
            #[allow(clippy::arithmetic_side_effects)]
            let len = f.prefix_length + f.suffix_length;
            let sub_fp = f.last_sub_fp;
            let idx = self.push_frame_fp(sub_fp, len, prev_term_ord)?;
            self.current = Some(idx);
            self.load_block()?;
            while self.frame_next()? {
                let sub_fp = self.cur().last_sub_fp;
                let idx = self.push_frame_fp(sub_fp, self.term.len, prev_term_ord)?;
                self.current = Some(idx);
                self.load_block()?;
            }
        }
        Ok(status)
    }

    /// `findTargetArc(label, arcs[from], getArc(to), fstReader)`, stored into
    /// `arcs[to]` when found.
    fn find_target_arc(&mut self, label: u8, from: usize, to: usize) -> Result<bool> {
        self.get_arc(to);
        let mut reader = self.fr.index.bytes_reader();
        let found =
            self.fr
                .index
                .find_target_arc(i32::from(label), &self.arcs[from], &mut reader)?;
        Ok(match found {
            Some(arc) => {
                self.arcs[to] = arc;
                true
            }
            None => false,
        })
    }

    /// The start shared by `seekExact`/`seekCeil`: reuses the frames of the
    /// previous seek for the prefix it shares with `target`, or starts from
    /// the root. `exact` picks `seekExact`'s `targetBeforeCurrentLength`
    /// for a target before the current term (`seekCeil` takes 0).
    // ARITH: `target_upto` counts bytes of `target`; frame indices follow
    // the arcs that were pushed.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_prefix(&mut self, target: &[u8], exact: bool) -> Result<SeekStart> {
        self.term.grow(1 + target.len());
        self.target_before_current_length = self.cur().ord;
        if self.positioned && self.current.is_some() {
            let mut arc_idx = 0;
            self.output.clone_from(self.arcs[0].output());
            let mut target_upto = 0;
            let mut last_frame = 0usize;
            let target_limit = target.len().min(self.valid_index_prefix);
            let mut cmp = Ordering::Equal;
            while target_upto < target_limit {
                cmp = self.term.byte_at(target_upto).cmp(&target[target_upto]);
                if cmp != Ordering::Equal {
                    break;
                }
                arc_idx = 1 + target_upto;
                FstOrdsOutputs::add_assign(&mut self.output, self.arcs[arc_idx].output());
                if self.arcs[arc_idx].is_final() {
                    last_frame += 1;
                }
                target_upto += 1;
            }
            if cmp == Ordering::Equal {
                let mine = self.term.get().get(target_upto..).unwrap_or(&[]);
                cmp = mine.cmp(&target[target_upto..]);
            }
            match cmp {
                Ordering::Less => self.current = Some(self.get_frame(last_frame)),
                Ordering::Greater => {
                    self.current = Some(self.get_frame(last_frame));
                    self.target_before_current_length = if exact { self.cur().ord } else { 0 };
                    self.cur_mut().rewind()?;
                }
                Ordering::Equal => {
                    if self.term_exists {
                        return Ok(SeekStart::SameTerm);
                    }
                }
            }
            self.positioned = true;
            Ok(SeekStart::Walk {
                arc_idx,
                target_upto,
            })
        } else {
            self.target_before_current_length = -1;
            self.arcs[0] = self.fr.index.first_arc();
            self.output.clone_from(self.arcs[0].output());
            self.current = None;
            self.current = Some(self.push_final_frame(0, 0)?);
            self.positioned = true;
            Ok(SeekStart::Walk {
                arc_idx: 0,
                target_upto: 0,
            })
        }
    }

    /// `seekExact(target)`: whether `target` is a term of the field.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or terms index.
    // ARITH: `target_upto` counts bytes of `target`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_exact(&mut self, target: &[u8]) -> Result<bool> {
        let (mut arc_idx, mut target_upto) = match self.seek_prefix(target, true)? {
            SeekStart::SameTerm => return Ok(true),
            SeekStart::Walk {
                arc_idx,
                target_upto,
            } => (arc_idx, target_upto),
        };
        while target_upto < target.len() {
            let target_label = target[target_upto];
            if !self.find_target_arc(target_label, arc_idx, 1 + target_upto)? {
                self.valid_index_prefix = self.cur().prefix_length;
                self.cur_mut().scan_to_floor_frame(target)?;
                if !self.cur().has_terms {
                    self.term_exists = false;
                    self.term.set_byte_at(target_upto, target_label);
                    self.term.set_length(1 + target_upto);
                    return Ok(false);
                }
                return self.finish_exact(target);
            }
            arc_idx = 1 + target_upto;
            self.term.set_byte_at(target_upto, target_label);
            FstOrdsOutputs::add_assign(&mut self.output, self.arcs[arc_idx].output());
            target_upto += 1;
            if self.arcs[arc_idx].is_final() {
                self.current = Some(self.push_final_frame(arc_idx, target_upto)?);
            }
        }
        self.valid_index_prefix = self.cur().prefix_length;
        self.cur_mut().scan_to_floor_frame(target)?;
        if !self.cur().has_terms {
            self.term_exists = false;
            self.term.set_length(target_upto);
            return Ok(false);
        }
        self.finish_exact(target)
    }

    /// The block scan that ends an exact seek.
    fn finish_exact(&mut self, target: &[u8]) -> Result<bool> {
        self.load_block()?;
        Ok(self.scan_to_term(target, true)? == SeekStatus::Found)
    }

    /// `seekCeil(target)`.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or terms index.
    // ARITH: `target_upto` counts bytes of `target`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        let (mut arc_idx, mut target_upto) = match self.seek_prefix(target, false)? {
            SeekStart::SameTerm => return Ok(SeekStatus::Found),
            SeekStart::Walk {
                arc_idx,
                target_upto,
            } => (arc_idx, target_upto),
        };
        while target_upto < target.len() {
            let target_label = target[target_upto];
            if !self.find_target_arc(target_label, arc_idx, 1 + target_upto)? {
                return self.finish_ceil(target);
            }
            self.term.set_byte_at(target_upto, target_label);
            arc_idx = 1 + target_upto;
            FstOrdsOutputs::add_assign(&mut self.output, self.arcs[arc_idx].output());
            target_upto += 1;
            if self.arcs[arc_idx].is_final() {
                self.current = Some(self.push_final_frame(arc_idx, target_upto)?);
            }
        }
        self.finish_ceil(target)
    }

    /// The block scan that ends a ceiling seek.
    fn finish_ceil(&mut self, target: &[u8]) -> Result<SeekStatus> {
        self.valid_index_prefix = self.cur().prefix_length;
        self.cur_mut().scan_to_floor_frame(target)?;
        self.load_block()?;
        let result = self.scan_to_term(target, false)?;
        if result == SeekStatus::End {
            self.term.copy_bytes(target);
            self.term_exists = false;
            if self.next()?.is_some() {
                Ok(SeekStatus::NotFound)
            } else {
                Ok(SeekStatus::End)
            }
        } else {
            Ok(result)
        }
    }

    /// `next()`: the next term, or `None` past the last.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or terms index.
    // ARITH: frame ords of the loaded blocks.
    #[allow(clippy::arithmetic_side_effects, clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>> {
        if !self.in_initialized {
            self.arcs[0] = self.fr.index.first_arc();
            let root = self.fr.root_code.clone();
            self.current = Some(self.push_frame_data(&root, 0)?);
            self.load_block()?;
            self.positioned = true;
        }
        self.target_before_current_length = self.cur().ord;
        if self.current.is_none() || !self.positioned {
            let target = self.term.get().to_vec();
            // Java asserts the term is found.
            self.seek_exact(&target)?;
        }
        while self.cur().next_ent == self.cur().ent_count {
            if !self.cur().is_last_in_floor {
                // `loadNextFloorBlock()`.
                let f = self.cur_mut();
                f.fp = f.fp_end;
                f.next_ent = -1;
                self.load_block()?;
            } else {
                if self.cur().ord <= 0 {
                    self.term.len = 0;
                    self.valid_index_prefix = 0;
                    self.cur_mut().rewind()?;
                    self.term_exists = false;
                    self.positioned = false;
                    return Ok(None);
                }
                let last_fp = self.cur().fp_orig;
                self.current = Some((self.cur().ord - 1) as usize);
                if self.cur().next_ent == -1 || self.cur().last_sub_fp != last_fp {
                    let term = self.term.get().to_vec();
                    self.cur_mut().scan_to_floor_frame(&term)?;
                    self.load_block()?;
                    self.cur_mut().scan_to_sub_block(last_fp)?;
                }
                self.valid_index_prefix = self.valid_index_prefix.min(self.cur().prefix_length);
            }
        }
        loop {
            let prev_term_ord = self.cur().term_ord;
            if self.frame_next()? {
                let sub_fp = self.cur().last_sub_fp;
                let idx = self.push_frame_fp(sub_fp, self.term.len, prev_term_ord)?;
                self.current = Some(idx);
                self.cur_mut().is_floor = false;
                self.load_block()?;
            } else {
                self.positioned = true;
                return Ok(Some(self.term.get()));
            }
        }
    }

    /// `term()`.
    pub fn term(&self) -> &[u8] {
        self.term.get()
    }

    /// `ord()`: the current term's ord (`-1` before the first term).
    pub fn ord(&self) -> i64 {
        self.cur().term_ord.wrapping_sub(1)
    }

    /// `docFreq()`.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn doc_freq(&mut self) -> Result<i32> {
        Ok(self.term_state_raw()?.doc_freq)
    }

    /// `totalTermFreq()`.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn total_term_freq(&mut self) -> Result<i64> {
        Ok(self.term_state_raw()?.total_term_freq)
    }

    /// The current term's postings metadata.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn term_metadata(&mut self) -> Result<TermMetadata> {
        Ok(self.term_state_raw()?.meta)
    }

    /// `postings(reuse, flags)`.
    ///
    /// # Errors
    /// What decoding the metadata or the `.doc` fails with.
    pub fn postings(&mut self, doc_in: &DocInput<'_>, flags: PostingsFlags) -> Result<Postings> {
        let state = self.term_state_raw()?;
        self.fr.postings(&state, doc_in, flags)
    }

    /// `currentFrame.decodeMetaData()` and its state.
    fn term_state_raw(&mut self) -> Result<OrdsTermState> {
        let fr = self.fr;
        let f = self.cur_mut();
        f.decode_meta_data(fr)?;
        Ok(f.state)
    }

    /// `termState()`: the decoded state, with the term's ord.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn term_state(&mut self) -> Result<OrdsTermState> {
        let mut state = self.term_state_raw()?;
        state.ord = self.ord();
        Ok(state)
    }

    /// `seekExact(target, otherState)`: positions on a term by a state taken
    /// from [`Self::term_state`], unless already on it.
    // ARITH: an ord below `numTerms`, plus one.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_exact_state(&mut self, target: &[u8], state: OrdsTermState) {
        if target != self.term.get() || !self.term_exists {
            self.current = None;
            self.static_frame.state = state;
            self.term.copy_bytes(target);
            self.static_frame.meta_data_upto = self.static_frame.term_block_ord();
            self.static_frame.term_ord = state.ord.wrapping_add(1);
            self.valid_index_prefix = 0;
        }
        self.positioned = true;
    }

    /// `seekExact(long targetOrd)`: positions on the term with that ord.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] outside `0..size()`, with Java's message;
    /// `Corrupted` for a malformed block or terms index.
    // ARITH: lengths of a prefix from the terms index.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_exact_ord(&mut self, target_ord: i64) -> Result<()> {
        if target_ord < 0 || target_ord >= self.fr.size() {
            return Err(Error::IllegalArgument(format!(
                "targetOrd out of bounds (got: {target_ord}, numTerms={})",
                self.fr.size()
            )));
        }
        let (input, output) = self.get_by_output(target_ord)?;
        self.term.copy_bytes(&input);
        let len = input.len();
        self.current = if len == 0 {
            None
        } else {
            Some(self.get_frame(len - 1))
        };
        self.get_arc(len);
        self.target_before_current_length = i32::MAX;
        self.current = Some(self.push_frame_data(&output, len)?);
        if self.cur().term_ord > target_ord {
            self.cur_mut().rewind()?;
        }
        self.cur_mut().scan_to_floor_frame_ord(target_ord)?;
        self.load_block()?;
        while self.cur().term_ord <= target_ord {
            self.frame_next()?;
        }
        if self.cur().term_ord != target_ord.wrapping_add(1) || !self.term_exists {
            // Java asserts both.
            return Err(corrupt(format!("ord {target_ord} not found in its block")));
        }
        self.valid_index_prefix = 0;
        self.positioned = false;
        Ok(())
    }

    /// `getByOutput(targetOrd)`: the longest prefix whose final output's
    /// ord range holds `target_ord`, and that output -- walking the terms
    /// index down the arcs whose ranges hold it.
    ///
    /// Java builds each candidate arc's summed `Output`; only the ords
    /// decide, so this sums them alone and appends an arc's bytes once it is
    /// taken (the outputs are the same).
    // ARITH: the binary search runs in `i64` over an `i32` arc count, so
    // `high - 1`, `low + high` and `mid + 1` cannot overflow, and `mid` lies
    // in `0..num_arcs`, so it fits back in `i32`.
    #[allow(clippy::arithmetic_side_effects)]
    fn get_by_output(&self, target_ord: i64) -> Result<(Vec<u8>, Output)> {
        let fst = &self.fr.index;
        let mut reader = fst.bytes_reader();
        let mut arc = fst.first_arc();
        let mut output = arc.output().clone();
        let mut result: Vec<u8> = Vec::new();
        let mut best_upto = 0usize;
        let mut best_output = None;
        // Whether `prefix + (start, end)` holds the target.
        let contains = |prefix: &Output, start: i64, end: i64| {
            let start = prefix.start_ord.wrapping_add(start);
            let end = prefix.end_ord.wrapping_add(end);
            target_ord >= start && target_ord <= i64::MAX.wrapping_sub(end)
        };
        // Nodes are written children first, so each step down lands on a
        // lower address; a target that does not is a cycle in a corrupt
        // `.tipo` (Java would loop forever).
        let mut last_target = i64::MAX;
        loop {
            if arc.is_final() {
                let fin = arc.next_final_output();
                if contains(&output, fin.start_ord, fin.end_ord) {
                    best_output = Some(FstOrdsOutputs::add(&output, fin));
                    best_upto = result.len();
                }
            }
            if !target_has_arcs(&arc) {
                break;
            }
            let target = arc.target();
            if target >= last_target {
                return Err(corrupt(format!(
                    "terms index arc target {target} is not below its node"
                )));
            }
            last_target = target;
            fst.read_first_real_target_arc(target, &mut arc, &mut reader)?;
            let fixed = arc.bytes_per_arc() != 0;
            let direct = fixed && arc.node_flags() == ARCS_FOR_DIRECT_ADDRESSING;
            if fixed && (direct || arc.node_flags() == ARCS_FOR_BINARY_SEARCH) {
                // A direct-addressing node's slots are as fixed as a
                // binary-search node's and its arcs' ranges as ordered, so
                // they are searched the same way (Java scans them).
                let num_arcs = if direct {
                    fst.presence_count(&arc, &mut reader)?
                } else {
                    arc.num_arcs()
                };
                let mut low = 0i64;
                let mut high = i64::from(num_arcs) - 1;
                let mut mid = 0i64;
                let mut found = false;
                while low <= high {
                    mid = (low + high) >> 1;
                    reader.set_position(arc.pos_arcs_start());
                    reader.skip_bytes(i64::from(arc.bytes_per_arc()) * mid);
                    let flags = reader.read_byte()?;
                    if !direct {
                        fst.read_label(&mut reader)?;
                    }
                    let (start, end) = if flags & BIT_ARC_HAS_OUTPUT != 0 {
                        // `FSTOrdsOutputs.read`, the bytes skipped.
                        let len = reader.read_vint()?;
                        reader.skip_bytes(i64::from(len));
                        (reader.read_vlong()?, reader.read_vlong()?)
                    } else {
                        (0, 0)
                    };
                    let min_start = output.start_ord.wrapping_add(start);
                    let max_ord = i64::MAX.wrapping_sub(output.end_ord.wrapping_add(end));
                    if target_ord > max_ord {
                        low = mid + 1;
                    } else if target_ord < min_start {
                        high = mid - 1;
                    } else {
                        found = true;
                        break;
                    }
                }
                if !found {
                    break;
                }
                if direct {
                    fst.read_arc_by_presence_index(&mut arc, &mut reader, mid as i32)?;
                } else {
                    fst.read_arc_by_index(&mut arc, &mut reader, mid as i32)?;
                }
                result.push(arc.label() as u8);
                FstOrdsOutputs::add_assign(&mut output, arc.output());
            } else {
                loop {
                    let o = arc.output();
                    if contains(&output, o.start_ord, o.end_ord) {
                        FstOrdsOutputs::add_assign(&mut output, o);
                        result.push(arc.label() as u8);
                        break;
                    }
                    let end_ord = i64::MAX.wrapping_sub(output.end_ord.wrapping_add(o.end_ord));
                    if target_ord < end_ord || arc.is_last() {
                        result.truncate(best_upto);
                        let best = best_output
                            .ok_or_else(|| corrupt(format!("no block holds ord {target_ord}")))?;
                        return Ok((result, best));
                    }
                    fst.read_next_real_arc(&mut arc, &mut reader)?;
                }
            }
        }
        result.truncate(best_upto);
        let best =
            best_output.ok_or_else(|| corrupt(format!("no block holds ord {target_ord}")))?;
        Ok((result, best))
    }
}

/// How the shared seek prefix ended.
enum SeekStart {
    /// `target` is the current term, which exists.
    SameTerm,
    /// Walk the terms index on from `target_upto`, the output accumulated
    /// in the enum's `output`.
    Walk { arc_idx: usize, target_upto: usize },
}
