//! Port of `IDVersionSegmentTermsEnum` and `IDVersionSegmentTermsEnumFrame`:
//! the Lucene 4 block-tree terms enum, plus the version-aware
//! [`IdVersionSegmentTermsEnum::seek_exact_with_version`] that gives up on a
//! block whose largest version (from the terms index) is below the one asked
//! for, without reading it.
//!
//! Java's `BytesRefBuilder` keeps the bytes past its length, and the seek
//! compares against them (a re-seek reads `term.byteAt(targetUpto)` beyond
//! the current term); [`TermBuf`] keeps them too.

use std::cmp::Ordering;

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::error::Error;

use super::{
    IdVersionPostings, IdVersionTermState, IndexOutput, SingleDocsEnum, SinglePostingsEnum,
    VersionFieldReader, OUTPUT_FLAGS_NUM_BITS, OUTPUT_FLAG_HAS_TERMS, OUTPUT_FLAG_IS_FLOOR,
};
use crate::blocktree::SeekStatus;
use crate::fst_util::Arc;

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Corrupted(msg.into())
}

/// `ByteArrayDataInput.readVInt` at `*pos`.
fn read_vint(buf: &[u8], pos: &mut usize) -> Result<i32, Error> {
    let mut input = SliceInput::new(buf);
    input.seek(*pos)?;
    let v = input.read_vint()?;
    *pos = input.position();
    Ok(v)
}

/// `ByteArrayDataInput.readVLong` at `*pos`.
fn read_vlong(buf: &[u8], pos: &mut usize) -> Result<i64, Error> {
    let mut input = SliceInput::new(buf);
    input.seek(*pos)?;
    let v = input.read_vlong()?;
    *pos = input.position();
    Ok(v)
}

/// `ByteArrayDataInput.readZLong` at `*pos`.
fn read_zlong(buf: &[u8], pos: &mut usize) -> Result<i64, Error> {
    let mut input = SliceInput::new(buf);
    input.seek(*pos)?;
    let v = input.read_zlong()?;
    *pos = input.position();
    Ok(v)
}

/// `ByteArrayDataInput.readByte() & 0xff` at `*pos`.
fn read_byte(buf: &[u8], pos: &mut usize) -> Result<i32, Error> {
    let b = *buf
        .get(*pos)
        .ok_or_else(|| corrupt("read past the end of a block"))?;
    *pos = pos.saturating_add(1);
    Ok(i32::from(b))
}

/// A length off disk as `usize`.
fn length(n: i32) -> Result<usize, Error> {
    usize::try_from(n).map_err(|_| corrupt(format!("negative length {n}")))
}

/// `code >>> k` of a Java `int`.
fn ushr32(code: i32, k: u32) -> i32 {
    ((code as u32) >> k) as i32
}

/// `code >>> k` of a Java `long`.
fn ushr64(code: i64, k: u32) -> i64 {
    ((code as u64) >> k) as i64
}

/// `BytesRefBuilder`: `len` bytes of term, the bytes past it kept.
#[derive(Debug, Default)]
struct TermBuf {
    bytes: Vec<u8>,
    len: usize,
}

impl TermBuf {
    /// `grow(n)`.
    fn grow(&mut self, n: usize) {
        if self.bytes.len() < n {
            self.bytes.resize(n, 0);
        }
    }

    /// `get()`.
    fn get(&self) -> &[u8] {
        &self.bytes[..self.len]
    }

    /// `byteAt(i)`: past the array (which Java's grow-before-seek rules
    /// out) reads 0.
    fn byte_at(&self, i: usize) -> u8 {
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
    fn copy_bytes(&mut self, src: &[u8]) {
        self.grow(src.len());
        self.bytes[..src.len()].copy_from_slice(src);
        self.len = src.len();
    }

    /// `suffixesReader.readBytes(term.bytes(), prefix, suffix)` and
    /// `fillTerm`'s copy: `prefix + src.len()` bytes long afterwards.
    // ARITH: a prefix plus a suffix of one term, both from a block.
    #[allow(clippy::arithmetic_side_effects)]
    fn set_suffix(&mut self, prefix: usize, src: &[u8]) {
        self.set_length(prefix + src.len());
        self.bytes[prefix..prefix + src.len()].copy_from_slice(src);
    }
}

/// `IDVersionSegmentTermsEnumFrame`.
#[derive(Debug, Clone, Default)]
struct Frame {
    ord: i32,
    has_terms: bool,
    has_terms_orig: bool,
    is_floor: bool,
    max_id_version: i64,
    fp: i64,
    fp_orig: i64,
    fp_end: i64,
    /// `suffixBytes`, exactly the block's.
    suffix_bytes: Vec<u8>,
    /// `suffixesReader`'s position.
    suffix_pos: usize,
    /// `floorData`, exactly the frame's.
    floor_data: Vec<u8>,
    /// `floorDataReader`'s position.
    floor_pos: usize,
    prefix_length: usize,
    ent_count: i32,
    next_ent: i32,
    is_last_in_floor: bool,
    is_leaf_block: bool,
    last_sub_fp: i64,
    next_floor_label: i32,
    num_follow_floor_blocks: i32,
    meta_data_upto: i32,
    state: IdVersionTermState,
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
            state: IdVersionTermState {
                total_term_freq: -1,
                ..IdVersionTermState::default()
            },
            ..Frame::default()
        }
    }

    /// `setFloorData(in, source)`: the bytes of `source` after the code.
    fn set_floor_data(&mut self, rest: &[u8]) -> Result<(), Error> {
        self.floor_data.clear();
        self.floor_data.extend_from_slice(rest);
        self.floor_pos = 0;
        self.num_follow_floor_blocks = read_vint(&self.floor_data, &mut self.floor_pos)?;
        // SENTINEL-OK: this file's `read_byte` reads an unsigned floor
        // label (0..=255) or errors; it has no `-1` sentinel.
        self.next_floor_label = read_byte(&self.floor_data, &mut self.floor_pos)?;
        Ok(())
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
    fn load_block(&mut self, tiv: &[u8]) -> Result<(), Error> {
        if self.next_ent != -1 {
            return Ok(());
        }
        let mut input = SliceInput::new(tiv);
        let fp = usize::try_from(self.fp).map_err(|_| corrupt(format!("block fp {}", self.fp)))?;
        input.seek(fp)?;
        let code = input.read_vint()?;
        self.ent_count = ushr32(code, 1);
        self.is_last_in_floor = code & 1 != 0;
        let code = input.read_vint()?;
        self.is_leaf_block = code & 1 != 0;
        let num_bytes = length(ushr32(code, 1))?;
        read_exact(&mut input, &mut self.suffix_bytes, num_bytes)?;
        self.suffix_pos = 0;
        self.meta_data_upto = 0;
        self.state.term_block_ord = 0;
        self.next_ent = 0;
        self.last_sub_fp = -1;
        let num_bytes = length(input.read_vint()?)?;
        read_exact(&mut input, &mut self.bytes, num_bytes)?;
        self.bytes_pos = 0;
        self.fp_end = input.position() as i64;
        Ok(())
    }

    /// `rewind()`.
    fn rewind(&mut self) -> Result<(), Error> {
        self.fp = self.fp_orig;
        self.next_ent = -1;
        self.has_terms = self.has_terms_orig;
        if self.is_floor {
            self.floor_pos = 0;
            self.num_follow_floor_blocks = read_vint(&self.floor_data, &mut self.floor_pos)?;
            // SENTINEL-OK: this file's `read_byte` reads an unsigned floor
            // label (0..=255) or errors; it has no `-1` sentinel.
            self.next_floor_label = read_byte(&self.floor_data, &mut self.floor_pos)?;
        }
        Ok(())
    }

    /// The next `n` suffix bytes, the reader moved past them.
    // ARITH: a position within `suffix_bytes` plus a length checked against it.
    #[allow(clippy::arithmetic_side_effects)]
    fn skip_suffix(&mut self, n: usize) -> Result<usize, Error> {
        let start = self.suffix_pos;
        if n > self.suffix_bytes.len() - start {
            return Err(corrupt("suffix past the end of its block"));
        }
        self.suffix_pos = start + n;
        Ok(start)
    }

    /// `next()`: `true` when the entry is a sub-block.
    fn next(&mut self, term: &mut TermBuf, term_exists: &mut bool) -> Result<bool, Error> {
        if self.is_leaf_block {
            self.next_leaf(term, term_exists)
        } else {
            self.next_non_leaf(term, term_exists)
        }
    }

    /// `nextLeaf()`.
    // ARITH: `next_ent < ent_count`, an i32.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_leaf(&mut self, term: &mut TermBuf, term_exists: &mut bool) -> Result<bool, Error> {
        self.next_ent += 1;
        self.suffix_length = length(read_vint(&self.suffix_bytes, &mut self.suffix_pos)?)?;
        self.start_byte_pos = self.skip_suffix(self.suffix_length)?;
        term.set_suffix(self.prefix_length, self.suffix());
        *term_exists = true;
        Ok(false)
    }

    /// `nextNonLeaf()`.
    // ARITH: `next_ent < ent_count`; a block's fp minus a smaller sub-block's.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_non_leaf(&mut self, term: &mut TermBuf, term_exists: &mut bool) -> Result<bool, Error> {
        self.next_ent += 1;
        let code = read_vint(&self.suffix_bytes, &mut self.suffix_pos)?;
        self.suffix_length = length(ushr32(code, 1))?;
        self.start_byte_pos = self.skip_suffix(self.suffix_length)?;
        term.set_suffix(self.prefix_length, self.suffix());
        if code & 1 == 0 {
            *term_exists = true;
            self.sub_code = 0;
            self.state.term_block_ord += 1;
            Ok(false)
        } else {
            *term_exists = false;
            self.sub_code = read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
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

    /// `scanToFloorFrame(target)`.
    // ARITH: `num_follow_floor_blocks` counts down from the vInt that bounds
    // the loop; fps are the floor data's sums.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_floor_frame(&mut self, target: &[u8]) -> Result<(), Error> {
        if !self.is_floor || target.len() <= self.prefix_length {
            return Ok(());
        }
        let target_label = i32::from(target[self.prefix_length]);
        if target_label < self.next_floor_label {
            return Ok(());
        }
        let mut new_fp;
        loop {
            let code = read_vlong(&self.floor_data, &mut self.floor_pos)?;
            new_fp = self.fp_orig.wrapping_add(ushr64(code, 1));
            self.has_terms = code & 1 != 0;
            self.is_last_in_floor = self.num_follow_floor_blocks == 1;
            self.num_follow_floor_blocks -= 1;
            if self.is_last_in_floor {
                self.next_floor_label = 256;
                break;
            }
            // SENTINEL-OK: this file's `read_byte` reads an unsigned floor
            // label (0..=255) or errors; it has no `-1` sentinel.
            self.next_floor_label = read_byte(&self.floor_data, &mut self.floor_pos)?;
            if target_label < self.next_floor_label {
                break;
            }
        }
        if new_fp != self.fp {
            self.next_ent = -1;
            self.fp = new_fp;
        }
        Ok(())
    }

    /// `decodeMetaData()`: `IDVersionPostingsReader.decodeTerm` up to the
    /// current term.
    // ARITH: `meta_data_upto < limit`, an i32.
    #[allow(clippy::arithmetic_side_effects)]
    fn decode_meta_data(&mut self) -> Result<(), Error> {
        let limit = self.term_block_ord();
        let mut absolute = self.meta_data_upto == 0;
        while self.meta_data_upto < limit {
            self.state.doc_freq = 1;
            self.state.total_term_freq = 1;
            self.state.doc_id = read_vint(&self.bytes, &mut self.bytes_pos)?;
            if absolute {
                self.state.id_version = read_vlong(&self.bytes, &mut self.bytes_pos)?;
            } else {
                let delta = read_zlong(&self.bytes, &mut self.bytes_pos)?;
                self.state.id_version = self.state.id_version.wrapping_add(delta);
            }
            self.meta_data_upto += 1;
            absolute = false;
        }
        self.state.term_block_ord = self.meta_data_upto;
        Ok(())
    }

    /// `scanToSubBlock(subFP)`.
    // ARITH: entry counts within a block.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_sub_block(&mut self, sub_fp: i64) -> Result<(), Error> {
        if self.last_sub_fp == sub_fp {
            return Ok(());
        }
        let target_sub_code = self.fp.wrapping_sub(sub_fp);
        loop {
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
                if target_sub_code == sub_code {
                    self.last_sub_fp = sub_fp;
                    return Ok(());
                }
            } else {
                self.state.term_block_ord += 1;
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
    ) -> Result<SeekStatus, Error> {
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
    /// `bool` asks the caller for it.
    // ARITH: `next_ent < ent_count`, an i32.
    #[allow(clippy::arithmetic_side_effects)]
    fn scan_to_term_non_leaf(
        &mut self,
        target: &[u8],
        exact_only: bool,
        term: &mut TermBuf,
        term_exists: &mut bool,
    ) -> Result<(SeekStatus, bool), Error> {
        if self.next_ent == self.ent_count {
            if exact_only {
                self.fill_term(term);
                *term_exists = self.sub_code == 0;
            }
            return Ok((SeekStatus::End, false));
        }
        while self.next_ent < self.ent_count {
            self.next_ent += 1;
            let code = read_vint(&self.suffix_bytes, &mut self.suffix_pos)?;
            self.suffix_length = length(ushr32(code, 1))?;
            *term_exists = code & 1 == 0;
            self.start_byte_pos = self.skip_suffix(self.suffix_length)?;
            if *term_exists {
                self.state.term_block_ord += 1;
                self.sub_code = 0;
            } else {
                self.sub_code = read_vlong(&self.suffix_bytes, &mut self.suffix_pos)?;
                self.last_sub_fp = self.fp.wrapping_sub(self.sub_code);
            }
            match self.compare_suffix(target) {
                Ordering::Less => {}
                Ordering::Greater => {
                    self.fill_term(term);
                    return Ok((SeekStatus::NotFound, !exact_only && !*term_exists));
                }
                Ordering::Equal => {
                    self.fill_term(term);
                    return Ok((SeekStatus::Found, false));
                }
            }
        }
        if exact_only {
            self.fill_term(term);
        }
        Ok((SeekStatus::End, false))
    }
}

/// `in.readBytes(dst, 0, n)` into a buffer reused across blocks.
fn read_exact(input: &mut SliceInput<'_>, dst: &mut Vec<u8>, n: usize) -> Result<(), Error> {
    if n > input.len().saturating_sub(input.position()) {
        return Err(corrupt(format!("{n} bytes past the end of the terms file")));
    }
    dst.clear();
    dst.resize(n, 0);
    input.read_bytes(dst)
}

/// `IDVersionSegmentTermsEnum`: one field's terms, with each term's single
/// document and version.
#[derive(Debug)]
pub struct IdVersionSegmentTermsEnum<'a> {
    tiv: &'a [u8],
    fr: &'a VersionFieldReader,
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
    arcs: Vec<Arc<IndexOutput>>,
    /// The first output accumulated along the arcs walked so far: Java's
    /// `output` (`ByteSequenceOutputs.add` concatenates), kept in one buffer
    /// instead of a new array per arc.
    out_first: Vec<u8>,
    /// Scratch for a final arc's frame data, `out_first` plus the arc's
    /// final output.
    frame_data: Vec<u8>,
}

impl<'a> IdVersionSegmentTermsEnum<'a> {
    /// `new IDVersionSegmentTermsEnum(fr)`.
    pub(super) fn new(tiv: &'a [u8], fr: &'a VersionFieldReader) -> Self {
        IdVersionSegmentTermsEnum {
            tiv,
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
            out_first: Vec::new(),
            frame_data: Vec::new(),
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

    /// `getFrame(1 + currentFrame.ord)`'s index.
    // ARITH: a frame ord is at most the longest term's length.
    #[allow(clippy::arithmetic_side_effects)]
    fn next_frame_index(&mut self) -> usize {
        let ord = (self.cur().ord + 1) as usize;
        while self.stack.len() <= ord {
            let n = self.stack.len() as i32;
            self.stack.push(Frame::new(n));
        }
        ord
    }

    /// `getArc(ord)`: makes slot `ord` exist.
    fn get_arc(&mut self, ord: usize) {
        while self.arcs.len() <= ord {
            self.arcs.push(self.fr.index.first_arc());
        }
    }

    /// `pushFrame(arc, frameData, length)`.
    fn push_frame_data(
        &mut self,
        first: &[u8],
        second: i64,
        length: usize,
    ) -> Result<usize, Error> {
        let mut pos = 0;
        let code = read_vlong(first, &mut pos)?;
        let fp_seek = ushr64(code, OUTPUT_FLAGS_NUM_BITS);
        let idx = self.next_frame_index();
        let f = &mut self.stack[idx];
        f.max_id_version = i64::MAX.wrapping_sub(second);
        f.has_terms = code & OUTPUT_FLAG_HAS_TERMS != 0;
        f.has_terms_orig = f.has_terms;
        f.is_floor = code & OUTPUT_FLAG_IS_FLOOR != 0;
        if f.is_floor {
            f.set_floor_data(first.get(pos..).unwrap_or(&[]))?;
        }
        self.push_frame_fp(fp_seek, length)
    }

    /// Adds arc `arc_idx`'s output to the accumulated one (`second` is the
    /// accumulated `PositiveIntOutputs` sum) and returns the new sum.
    fn add_arc_output(&mut self, arc_idx: usize, second: i64) -> i64 {
        let out = self.arcs[arc_idx].output();
        self.out_first.extend_from_slice(&out.first);
        second.wrapping_add(out.second)
    }

    /// `pushFrame(arc, fstOutputs.add(output, arc.nextFinalOutput()), length)`
    /// for final arc `arc_idx`.
    fn push_final_frame(
        &mut self,
        arc_idx: usize,
        second: i64,
        length: usize,
    ) -> Result<usize, Error> {
        let mut data = std::mem::take(&mut self.frame_data);
        data.clear();
        data.extend_from_slice(&self.out_first);
        let fin = self.arcs[arc_idx].next_final_output();
        data.extend_from_slice(&fin.first);
        let second = second.wrapping_add(fin.second);
        let pushed = self.push_frame_data(&data, second, length);
        self.frame_data = data;
        pushed
    }

    /// `pushFrame(arc, fp, length)`.
    fn push_frame_fp(&mut self, fp: i64, length: usize) -> Result<usize, Error> {
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
            f.fp_orig = fp;
            f.fp = fp;
            f.last_sub_fp = -1;
        }
        Ok(idx)
    }

    /// `loadBlock()` of frame `c`.
    fn load_block(&mut self, c: Option<usize>) -> Result<(), Error> {
        self.in_initialized = true;
        let tiv = self.tiv;
        self.frame_mut(c).load_block(tiv)
    }

    /// `currentFrame.next()`.
    fn frame_next(&mut self) -> Result<bool, Error> {
        let c = self.current;
        let IdVersionSegmentTermsEnum {
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
    fn scan_to_term(&mut self, target: &[u8], exact_only: bool) -> Result<SeekStatus, Error> {
        let c = self.current;
        let IdVersionSegmentTermsEnum {
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
        if descend {
            // ARITH: a prefix plus a suffix of one term.
            #[allow(clippy::arithmetic_side_effects)]
            let len = f.prefix_length + f.suffix_length;
            let sub_fp = f.last_sub_fp;
            self.descend(sub_fp, len)?;
        }
        Ok(status)
    }

    /// Pushes the sub-block at `fp` and walks down its first entries to
    /// the first term below it.
    fn descend(&mut self, fp: i64, length: usize) -> Result<(), Error> {
        let idx = self.push_frame_fp(fp, length)?;
        self.current = Some(idx);
        self.load_block(self.current)?;
        while self.frame_next()? {
            let sub_fp = self.cur().last_sub_fp;
            let idx = self.push_frame_fp(sub_fp, self.term.len)?;
            self.current = Some(idx);
            self.load_block(self.current)?;
        }
        Ok(())
    }

    /// `findTargetArc(label, arcs[from], getArc(to), fstReader)`, stored into
    /// `arcs[to]` when found.
    fn find_target_arc(&mut self, label: u8, from: usize, to: usize) -> Result<bool, Error> {
        self.get_arc(to);
        let mut reader = self.fr.index.bytes_reader();
        let found = self
            .fr
            .index
            .find_target_arc(i32::from(label), &self.arcs[from], &mut reader)
            .map_err(|e| corrupt(e.to_string()))?;
        Ok(match found {
            Some(arc) => {
                self.arcs[to] = arc;
                true
            }
            None => false,
        })
    }

    /// The shared start of `seekExact`/`seekCeil`: reuses the frames of the
    /// previous seek for the prefix it shares with `target`, or starts from
    /// the root. `Err(..)`-free `Some(..)` is the early return of a term
    /// equal to the current one.
    // ARITH: `target_upto` counts bytes of `target`; frame indices follow
    // the arcs that were pushed.
    #[allow(clippy::arithmetic_side_effects)]
    fn seek_prefix(&mut self, target: &[u8]) -> Result<SeekStart, Error> {
        if self.current.is_some() {
            let mut arc_idx = 0;
            self.out_first.clear();
            let mut output = self.add_arc_output(0, 0);
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
                output = self.add_arc_output(arc_idx, output);
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
                Ordering::Less => self.current = Some(last_frame),
                Ordering::Greater => {
                    self.target_before_current_length = 0;
                    self.current = Some(last_frame);
                    self.cur_mut().rewind()?;
                    return Ok(SeekStart::Walk {
                        arc_idx,
                        output,
                        target_upto,
                        changed: true,
                    });
                }
                Ordering::Equal => {
                    if self.term_exists {
                        return Ok(SeekStart::SameTerm);
                    }
                }
            }
            Ok(SeekStart::Walk {
                arc_idx,
                output,
                target_upto,
                changed: false,
            })
        } else {
            self.target_before_current_length = -1;
            self.arcs[0] = self.fr.index.first_arc();
            self.out_first.clear();
            let output = self.add_arc_output(0, 0);
            self.current = None;
            self.current = Some(self.push_final_frame(0, output, 0)?);
            Ok(SeekStart::Walk {
                arc_idx: 0,
                output,
                target_upto: 0,
                changed: false,
            })
        }
    }

    /// `seekExact(target)`: `seekExact(target, 0)`.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or terms index.
    pub fn try_seek_exact(&mut self, target: &[u8]) -> Result<bool, Error> {
        self.seek_exact_with_version(target, 0)
    }

    /// `getVersion()`: the current term's version (after a successful
    /// [`Self::seek_exact_with_version`] or [`Self::postings`]).
    pub fn version(&self) -> i64 {
        self.cur().state.id_version
    }

    /// `seekExact(target, minIDVersion)`: whether `target` exists with a
    /// version `>= min_id_version`. Refuses without reading the block when
    /// the terms index's largest version below the prefix is smaller.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or terms index.
    // ARITH: `target_upto` counts bytes of `target`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn seek_exact_with_version(
        &mut self,
        target: &[u8],
        min_id_version: i64,
    ) -> Result<bool, Error> {
        self.term.grow(1 + target.len());
        let start_frame_fp = self.cur().fp;
        self.target_before_current_length = self.cur().ord;
        let (mut arc_idx, mut output, mut target_upto, mut changed) =
            match self.seek_prefix(target)? {
                SeekStart::SameTerm => {
                    if self.cur().max_id_version < min_id_version {
                        return Ok(false);
                    }
                    self.cur_mut().decode_meta_data()?;
                    return Ok(self.cur().state.id_version >= min_id_version);
                }
                SeekStart::Walk {
                    arc_idx,
                    output,
                    target_upto,
                    changed,
                } => (arc_idx, output, target_upto, changed),
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
                if self.cur().max_id_version < min_id_version {
                    if self.cur().fp != start_frame_fp || changed {
                        self.term_exists = false;
                        self.term.set_byte_at(target_upto, target_label);
                        self.term.set_length(1 + target_upto);
                        self.valid_index_prefix = self.valid_index_prefix.min(self.term.len);
                    }
                    return Ok(false);
                }
                return self.finish_exact(target, min_id_version);
            }
            arc_idx = 1 + target_upto;
            if self.term.byte_at(target_upto) != target_label {
                changed = true;
                self.term.set_byte_at(target_upto, target_label);
                self.term_exists = false;
            }
            output = self.add_arc_output(arc_idx, output);
            target_upto += 1;
            if self.arcs[arc_idx].is_final() {
                self.current = Some(self.push_final_frame(arc_idx, output, target_upto)?);
            }
        }
        self.valid_index_prefix = self.cur().prefix_length;
        self.cur_mut().scan_to_floor_frame(target)?;
        if !self.cur().has_terms || self.cur().max_id_version < min_id_version {
            self.term_exists = false;
            self.term.set_length(target_upto);
            return Ok(false);
        }
        self.finish_exact(target, min_id_version)
    }

    /// The block scan that ends an exact seek.
    fn finish_exact(&mut self, target: &[u8], min_id_version: i64) -> Result<bool, Error> {
        self.load_block(self.current)?;
        if self.scan_to_term(target, true)? == SeekStatus::Found {
            self.cur_mut().decode_meta_data()?;
            Ok(self.cur().state.id_version >= min_id_version)
        } else {
            Ok(false)
        }
    }

    /// `seekCeil(target)`.
    ///
    /// # Errors
    /// `Corrupted` for a malformed block or terms index.
    // ARITH: `target_upto` counts bytes of `target`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn try_seek_ceil(&mut self, target: &[u8]) -> Result<SeekStatus, Error> {
        self.term.grow(1 + target.len());
        self.target_before_current_length = self.cur().ord;
        let (mut arc_idx, mut output, mut target_upto) = match self.seek_prefix(target)? {
            SeekStart::SameTerm => return Ok(SeekStatus::Found),
            SeekStart::Walk {
                arc_idx,
                output,
                target_upto,
                ..
            } => (arc_idx, output, target_upto),
        };
        while target_upto < target.len() {
            let target_label = target[target_upto];
            if !self.find_target_arc(target_label, arc_idx, 1 + target_upto)? {
                return self.finish_ceil(target);
            }
            self.term.set_byte_at(target_upto, target_label);
            arc_idx = 1 + target_upto;
            output = self.add_arc_output(arc_idx, output);
            target_upto += 1;
            if self.arcs[arc_idx].is_final() {
                self.current = Some(self.push_final_frame(arc_idx, output, target_upto)?);
            }
        }
        self.finish_ceil(target)
    }

    /// The block scan that ends a ceiling seek.
    fn finish_ceil(&mut self, target: &[u8]) -> Result<SeekStatus, Error> {
        self.valid_index_prefix = self.cur().prefix_length;
        self.cur_mut().scan_to_floor_frame(target)?;
        self.load_block(self.current)?;
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
    // ARITH: frame ords and floor positions of the loaded blocks.
    #[allow(clippy::arithmetic_side_effects, clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<&[u8]>, Error> {
        if !self.in_initialized {
            self.arcs[0] = self.fr.index.first_arc();
            let root = self.fr.root_code().clone();
            self.current = Some(self.push_frame_data(&root.first, root.second, 0)?);
            self.load_block(self.current)?;
        }
        self.target_before_current_length = self.cur().ord;
        if self.current.is_none() {
            let target = self.term.get().to_vec();
            self.try_seek_exact(&target)?;
        }
        while self.cur().next_ent == self.cur().ent_count {
            if !self.cur().is_last_in_floor {
                // `loadNextFloorBlock()`.
                let f = self.cur_mut();
                f.fp = f.fp_end;
                f.next_ent = -1;
                self.load_block(self.current)?;
            } else {
                if self.cur().ord == 0 {
                    self.term.len = 0;
                    self.valid_index_prefix = 0;
                    self.cur_mut().rewind()?;
                    self.term_exists = false;
                    return Ok(None);
                }
                let last_fp = self.cur().fp_orig;
                self.current = Some((self.cur().ord - 1) as usize);
                if self.cur().next_ent == -1 || self.cur().last_sub_fp != last_fp {
                    let term = self.term.get().to_vec();
                    self.cur_mut().scan_to_floor_frame(&term)?;
                    self.load_block(self.current)?;
                    self.cur_mut().scan_to_sub_block(last_fp)?;
                }
                self.valid_index_prefix = self.valid_index_prefix.min(self.cur().prefix_length);
            }
        }
        loop {
            if self.frame_next()? {
                let sub_fp = self.cur().last_sub_fp;
                let idx = self.push_frame_fp(sub_fp, self.term.len)?;
                self.current = Some(idx);
                self.cur_mut().is_floor = false;
                self.load_block(self.current)?;
            } else {
                return Ok(Some(self.term.get()));
            }
        }
    }

    /// `term()`.
    pub fn term(&self) -> &[u8] {
        self.term.get()
    }

    /// `docFreq()`: always 1.
    pub fn doc_freq(&self) -> i32 {
        1
    }

    /// `totalTermFreq()`: always 1.
    pub fn total_term_freq(&self) -> i64 {
        1
    }

    /// `postings(reuse, flags)`: `positions` is
    /// `PostingsEnum.featureRequested(flags, POSITIONS)`.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn postings(&mut self, positions: bool) -> Result<IdVersionPostings, Error> {
        self.cur_mut().decode_meta_data()?;
        let state = self.cur().state;
        Ok(if positions {
            IdVersionPostings::Positions(SinglePostingsEnum::new(state.doc_id, state.id_version))
        } else {
            IdVersionPostings::Docs(SingleDocsEnum::new(state.doc_id))
        })
    }

    /// `seekExact(target, otherState)`: positions on a term by a state taken
    /// from [`Self::term_state`], unless already on it.
    pub fn seek_exact_state(&mut self, target: &[u8], state: IdVersionTermState) {
        if target != self.term.get() || !self.term_exists {
            self.current = None;
            self.static_frame.state = state;
            self.term.copy_bytes(target);
            self.static_frame.meta_data_upto = self.static_frame.term_block_ord();
            self.valid_index_prefix = 0;
        }
    }

    /// `termState()`.
    ///
    /// # Errors
    /// `Corrupted` for malformed term metadata.
    pub fn term_state(&mut self) -> Result<IdVersionTermState, Error> {
        self.cur_mut().decode_meta_data()?;
        Ok(self.cur().state)
    }
}

/// How the shared seek prefix ended.
enum SeekStart {
    /// `target` is the current term, which exists.
    SameTerm,
    /// Walk the terms index on from `target_upto`.
    Walk {
        arc_idx: usize,
        /// The accumulated second output; the first is `out_first`.
        output: i64,
        target_upto: usize,
        changed: bool,
    },
}
