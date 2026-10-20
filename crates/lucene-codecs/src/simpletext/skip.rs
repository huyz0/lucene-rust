//! `SimpleTextSkipWriter`/`SimpleTextSkipReader`: the plain-text multi-level
//! skip list after a term's documents, over a port of
//! `MultiLevelSkipListWriter`/`MultiLevelSkipListReader` with
//! `skipInterval` 8, `skipMultiplier` 3 and at most 4 levels.
//!
//! Every 8th document of a term buffers a level-0 entry (every 24th a
//! level-1 one, every 72nd a level-2 one, ...); a term of at least 8
//! documents writes them after its last document:
//!
//! ```text
//!     skipList
//!       levelLength N        per level from the top down to 1: its length
//!       level L              before the level's first entry
//!         skipDoc D          the block's last document (absolute)
//!         skipDocFP F        where the block's first `doc` line starts
//!         impacts
//!           impact           the block's (level: the span's) competitive
//!             freq F         freq-norm pairs
//!             norm N
//!         impactsEnd
//!         childPointer C     levels above 0: the entry's end in the level below
//! ```
//!
//! The writer's level count comes from the segment's `maxDoc`, the reader's
//! from the term's `docFreq`; only levels with entries are written, so the
//! two agree.

use super::util::{self, parse_int, parse_long, tail};
use super::{corrupt, Result, NO_MORE_DOCS};
use crate::postings::Impact;
use crate::postings_writer::CompetitiveImpactAccumulator;

/// `SimpleTextSkipWriter.BLOCK_SIZE`: the level-0 skip interval.
pub const BLOCK_SIZE: i32 = 8;
/// `SimpleTextSkipWriter.skipMultiplier`.
pub const SKIP_MULTIPLIER: i32 = 3;
/// `SimpleTextSkipWriter.maxSkipLevels`.
pub const MAX_SKIP_LEVELS: usize = 4;

/// `SimpleTextSkipWriter.SKIP_LIST`.
pub const SKIP_LIST: &[u8] = b"    skipList ";
/// `SimpleTextSkipWriter.LEVEL_LENGTH`.
pub const LEVEL_LENGTH: &[u8] = b"      levelLength ";
/// `SimpleTextSkipWriter.LEVEL`.
pub const LEVEL: &[u8] = b"      level ";
/// `SimpleTextSkipWriter.SKIP_DOC`.
pub const SKIP_DOC: &[u8] = b"        skipDoc ";
/// `SimpleTextSkipWriter.SKIP_DOC_FP`.
pub const SKIP_DOC_FP: &[u8] = b"        skipDocFP ";
/// `SimpleTextSkipWriter.IMPACTS`.
pub const IMPACTS: &[u8] = b"        impacts ";
/// `SimpleTextSkipWriter.IMPACT`.
pub const IMPACT: &[u8] = b"          impact ";
/// `SimpleTextSkipWriter.FREQ`.
pub const FREQ: &[u8] = b"            freq ";
/// `SimpleTextSkipWriter.NORM`.
pub const NORM: &[u8] = b"            norm ";
/// `SimpleTextSkipWriter.IMPACTS_END`.
pub const IMPACTS_END: &[u8] = b"        impactsEnd ";
/// `SimpleTextSkipWriter.CHILD_POINTER`.
pub const CHILD_POINTER: &[u8] = b"        childPointer ";

/// `MathUtil.log(x, base)`: how many times `x` divides by `base`.
fn log(mut x: i32, base: i32) -> i32 {
    let mut ret = 0;
    // ARITH: `base >= 2` (the constant multiplier), so `x` shrinks and `ret`
    // counts at most 31 divisions.
    #[allow(clippy::arithmetic_side_effects)]
    while x >= base {
        x /= base;
        ret += 1;
    }
    ret
}

/// `1 + MathUtil.log(df / skipInterval, skipMultiplier)` capped at
/// `maxSkipLevels`, or 1 for `df <= skipInterval`: the constructor's and
/// `loadSkipLevels`' level count.
fn levels_for(df: i32) -> usize {
    if df > BLOCK_SIZE {
        // ARITH: `BLOCK_SIZE > 0`; `log` is at most 31, so `1 + log` fits.
        #[allow(clippy::arithmetic_side_effects)]
        let n = 1 + log(df / BLOCK_SIZE, SKIP_MULTIPLIER);
        usize::try_from(n).map_or(MAX_SKIP_LEVELS, |n| n.min(MAX_SKIP_LEVELS))
    } else {
        1
    }
}

fn line(out: &mut Vec<u8>, prefix: &[u8], value: impl std::fmt::Display) {
    util::write(out, prefix);
    util::write(out, value.to_string().as_bytes());
    util::write_newline(out);
}

fn bare_line(out: &mut Vec<u8>, prefix: &[u8]) {
    util::write(out, prefix);
    util::write_newline(out);
}

/// `SimpleTextSkipWriter`.
#[derive(Debug, Clone)]
pub struct SimpleTextSkipWriter {
    /// `numberOfSkipLevels`, from the segment's `maxDoc`.
    number_of_skip_levels: usize,
    /// `skipBuffer`, one per level.
    skip_buffer: Vec<Vec<u8>>,
    /// `wroteHeaderPerLevelMap`.
    wrote_header: [bool; MAX_SKIP_LEVELS],
    cur_doc: i32,
    cur_doc_file_pointer: i64,
    /// `curCompetitiveFreqNorms`.
    cur_competitive_freq_norms: Vec<CompetitiveImpactAccumulator>,
}

impl SimpleTextSkipWriter {
    /// `new SimpleTextSkipWriter(writeState)`: `super(BLOCK_SIZE,
    /// skipMultiplier, maxSkipLevels, maxDoc)`, then `resetSkip()`.
    pub fn new(max_doc: i32) -> Self {
        let number_of_skip_levels = levels_for(max_doc);
        let mut w = SimpleTextSkipWriter {
            number_of_skip_levels,
            skip_buffer: vec![Vec::new(); number_of_skip_levels],
            wrote_header: [false; MAX_SKIP_LEVELS],
            cur_doc: -1,
            cur_doc_file_pointer: -1,
            cur_competitive_freq_norms: vec![
                CompetitiveImpactAccumulator::default();
                MAX_SKIP_LEVELS
            ],
        };
        w.reset_skip();
        w
    }

    /// `resetSkip()`.
    pub fn reset_skip(&mut self) {
        for buffer in &mut self.skip_buffer {
            buffer.clear();
        }
        self.wrote_header = [false; MAX_SKIP_LEVELS];
        self.cur_doc = -1;
        self.cur_doc_file_pointer = -1;
        for acc in &mut self.cur_competitive_freq_norms {
            acc.clear();
        }
    }

    /// `bufferSkip(doc, docFilePointer, numDocs, accumulator)`: the block that
    /// ends at `doc` and starts at `doc_file_pointer`, its `num_docs`-th
    /// document.
    pub(crate) fn buffer_skip(
        &mut self,
        doc: i32,
        doc_file_pointer: i64,
        num_docs: i32,
        acc: &CompetitiveImpactAccumulator,
    ) {
        debug_assert!(doc > self.cur_doc);
        self.cur_doc = doc;
        self.cur_doc_file_pointer = doc_file_pointer;
        self.cur_competitive_freq_norms[0].add_all(acc);
        self.buffer_skip_levels(num_docs);
    }

    /// `MultiLevelSkipListWriter.bufferSkip(df)`.
    fn buffer_skip_levels(&mut self, mut df: i32) {
        debug_assert!(df % BLOCK_SIZE == 0);
        // ARITH: constants; `numLevels` stops at `number_of_skip_levels`.
        #[allow(clippy::arithmetic_side_effects)]
        let num_levels = {
            let window_length = BLOCK_SIZE * SKIP_MULTIPLIER;
            let mut num_levels = 1;
            if df % window_length == 0 {
                num_levels += 1;
                df /= window_length;
                while df % SKIP_MULTIPLIER == 0 && num_levels < self.number_of_skip_levels {
                    num_levels += 1;
                    df /= SKIP_MULTIPLIER;
                }
            }
            num_levels.min(self.number_of_skip_levels)
        };
        let mut child_pointer = 0usize;
        for level in 0..num_levels {
            self.write_skip_data(level);
            let new_child_pointer = self.skip_buffer[level].len();
            if level != 0 {
                // store child pointers for all levels except the lowest
                line(&mut self.skip_buffer[level], CHILD_POINTER, child_pointer);
            }
            child_pointer = new_child_pointer;
        }
    }

    /// `writeSkipData(level, skipBuffer)`.
    fn write_skip_data(&mut self, level: usize) {
        let buf = &mut self.skip_buffer[level];
        if !self.wrote_header[level] {
            line(buf, LEVEL, level);
            self.wrote_header[level] = true;
        }
        line(buf, SKIP_DOC, self.cur_doc);
        line(buf, SKIP_DOC_FP, self.cur_doc_file_pointer);
        let impacts = self.cur_competitive_freq_norms[level].get_competitive_freq_norm_pairs();
        debug_assert!(!impacts.is_empty());
        // ARITH: `level < number_of_skip_levels <= MAX_SKIP_LEVELS`.
        #[allow(clippy::arithmetic_side_effects)]
        let up = level + 1;
        if up < self.number_of_skip_levels {
            let (lower, upper) = self.cur_competitive_freq_norms.split_at_mut(up);
            upper[0].add_all(&lower[level]);
        }
        bare_line(buf, IMPACTS);
        for impact in impacts {
            bare_line(buf, IMPACT);
            line(buf, FREQ, impact.freq);
            line(buf, NORM, impact.norm);
        }
        bare_line(buf, IMPACTS_END);
        self.cur_competitive_freq_norms[level].clear();
    }

    /// `writeSkip(output)`: `skipList`, each upper level behind its length,
    /// then level 0; the offset it starts at.
    pub fn write_skip(&self, output: &mut Vec<u8>) -> usize {
        let skip_offset = output.len();
        bare_line(output, SKIP_LIST);
        for level in (1..self.number_of_skip_levels).rev() {
            let length = self.skip_buffer[level].len();
            if length > 0 {
                line(output, LEVEL_LENGTH, length);
                output.extend_from_slice(&self.skip_buffer[level]);
            }
        }
        if let Some(level0) = self.skip_buffer.first() {
            output.extend_from_slice(level0);
        }
        skip_offset
    }
}

/// The impacts a [`SimpleTextSkipReader`] last read: the anonymous
/// `Impacts` of `SimpleTextSkipReader`, or `SlowImpactsEnum`'s one level of
/// `(Integer.MAX_VALUE, 1)` up to `NO_MORE_DOCS`.
#[derive(Debug, Clone, Copy)]
pub enum Impacts<'s> {
    /// `SlowImpactsEnum.DUMMY_IMPACTS`.
    Slow,
    /// The skip reader's live view.
    Skip(&'s SimpleTextSkipReader<'s>),
}

/// `SlowImpactsEnum`'s and an unread level's `(Integer.MAX_VALUE, 1)`.
const DUMMY: &[Impact] = &[Impact {
    freq: i32::MAX,
    norm: 1,
}];

impl Impacts<'_> {
    /// `numLevels()`.
    pub fn num_levels(&self) -> usize {
        match self {
            Impacts::Slow => 1,
            Impacts::Skip(r) => r.num_levels,
        }
    }

    /// `getDocIdUpTo(level)`; `None` past `numLevels()` (Java asserts).
    pub fn doc_id_up_to(&self, level: usize) -> Option<i32> {
        if level >= self.num_levels() {
            return None;
        }
        match self {
            Impacts::Slow => Some(NO_MORE_DOCS),
            Impacts::Skip(r) => r.skip_doc.get(level).copied(),
        }
    }

    /// `getImpacts(level)`; `None` past `numLevels()` (Java asserts).
    pub fn impacts(&self, level: usize) -> Option<&[Impact]> {
        if level >= self.num_levels() {
            return None;
        }
        match self {
            Impacts::Slow => Some(DUMMY),
            Impacts::Skip(r) => r.per_level_impacts.get(level).map(Vec::as_slice),
        }
    }
}

/// `SimpleTextSkipReader` over the whole `.pst`.
#[derive(Debug, Clone)]
pub struct SimpleTextSkipReader<'a> {
    data: &'a [u8],
    /// `numberOfSkipLevels`: how many levels are still live.
    number_of_skip_levels: usize,
    doc_count: i32,
    /// `skipStream[i].getFilePointer()`.
    stream: [usize; MAX_SKIP_LEVELS],
    skip_pointer: [usize; MAX_SKIP_LEVELS],
    skip_interval: [i32; MAX_SKIP_LEVELS],
    num_skipped: [i32; MAX_SKIP_LEVELS],
    skip_doc: [i32; MAX_SKIP_LEVELS],
    last_doc: i32,
    child_pointer: [usize; MAX_SKIP_LEVELS],
    last_child_pointer: usize,
    per_level_impacts: Vec<Vec<Impact>>,
    next_skip_doc_fp: i64,
    num_levels: usize,
    has_skip_list: bool,
    scratch: Vec<u8>,
}

impl<'a> SimpleTextSkipReader<'a> {
    /// `new SimpleTextSkipReader(in.clone())`.
    pub fn new(data: &'a [u8]) -> Self {
        let mut skip_interval = [BLOCK_SIZE; MAX_SKIP_LEVELS];
        for i in 1..MAX_SKIP_LEVELS {
            // ARITH: 8 * 3^3 = 216 at most.
            #[allow(clippy::arithmetic_side_effects)]
            {
                skip_interval[i] = skip_interval[i - 1] * SKIP_MULTIPLIER;
            }
        }
        let mut r = SimpleTextSkipReader {
            data,
            number_of_skip_levels: 0,
            doc_count: 0,
            stream: [0; MAX_SKIP_LEVELS],
            skip_pointer: [0; MAX_SKIP_LEVELS],
            skip_interval,
            num_skipped: [0; MAX_SKIP_LEVELS],
            skip_doc: [0; MAX_SKIP_LEVELS],
            last_doc: 0,
            child_pointer: [0; MAX_SKIP_LEVELS],
            last_child_pointer: 0,
            per_level_impacts: Vec::new(),
            next_skip_doc_fp: -1,
            num_levels: 1,
            has_skip_list: false,
            scratch: Vec::new(),
        };
        r.init();
        r
    }

    /// `init()`: every level's impacts back to `(Integer.MAX_VALUE, 1)`.
    fn init(&mut self) {
        self.next_skip_doc_fp = -1;
        self.num_levels = 1;
        self.per_level_impacts = vec![DUMMY.to_vec(); MAX_SKIP_LEVELS];
        self.has_skip_list = false;
    }

    /// `reset(skipPointer, docFreq)`: a term's skip list, if it has one.
    ///
    /// # Errors
    /// `Corrupted` for a skip pointer or level length outside the file or a
    /// level length that does not parse.
    pub fn reset(&mut self, skip_pointer: i64, doc_freq: i32) -> Result<()> {
        self.init();
        if skip_pointer > 0 {
            self.base_init(skip_pointer, doc_freq)?;
            self.has_skip_list = true;
        }
        Ok(())
    }

    /// `MultiLevelSkipListReader.init(skipPointer, df)`.
    fn base_init(&mut self, skip_pointer: i64, df: i32) -> Result<()> {
        // Java asserts `skipPointer <= skipStream[0].length()`.
        let skip_pointer = self.file_pointer(skip_pointer, "skip pointer")?;
        self.skip_pointer[0] = skip_pointer;
        self.doc_count = df;
        self.skip_doc = [0; MAX_SKIP_LEVELS];
        self.num_skipped = [0; MAX_SKIP_LEVELS];
        self.child_pointer = [0; MAX_SKIP_LEVELS];
        self.load_skip_levels()
    }

    fn file_pointer(&self, fp: i64, what: &str) -> Result<usize> {
        usize::try_from(fp)
            .ok()
            .filter(|&p| p <= self.data.len())
            .ok_or_else(|| corrupt(format!("invalid {what}: {fp}")))
    }

    /// `loadSkipLevels()`.
    fn load_skip_levels(&mut self) -> Result<()> {
        self.number_of_skip_levels = levels_for(self.doc_count);
        self.stream[0] = self.skip_pointer[0];
        for i in (1..self.number_of_skip_levels).rev() {
            // the length of the current level
            let length = self.read_level_length()?;
            // the start pointer of the current level
            self.skip_pointer[i] = self.stream[0];
            self.stream[i] = self.stream[0];
            // move base stream beyond the current level
            let end = i64::try_from(self.stream[0])
                .ok()
                .and_then(|at| at.checked_add(length))
                .ok_or_else(|| corrupt(format!("invalid level length: {length}")))?;
            self.stream[0] = self.file_pointer(end, "level length")?;
        }
        // use base stream for the lowest level
        self.skip_pointer[0] = self.stream[0];
        Ok(())
    }

    /// `skipTo(target)`: `MultiLevelSkipListReader.skipTo`, then the impacts
    /// of the levels left (a dummy level once level 0 is exhausted).
    ///
    /// # Errors
    /// What reading an entry fails with.
    pub fn skip_to(&mut self, target: i32) -> Result<()> {
        if !self.has_skip_list {
            return Ok(());
        }
        self.base_skip_to(target)?;
        if self.number_of_skip_levels > 0 {
            self.num_levels = self.number_of_skip_levels;
        } else {
            // End of postings don't have skip data anymore, so we fill with
            // dummy data like SlowImpactsEnum.
            self.num_levels = 1;
            self.per_level_impacts.insert(0, DUMMY.to_vec());
        }
        Ok(())
    }

    /// `MultiLevelSkipListReader.skipTo(target)`.
    fn base_skip_to(&mut self, target: i32) -> Result<()> {
        // walk up the levels until highest level is found that has a skip
        // for this target
        let mut level = 0usize;
        // ARITH: `level + 1 < number_of_skip_levels <= MAX_SKIP_LEVELS`.
        #[allow(clippy::arithmetic_side_effects)]
        while level + 1 < self.number_of_skip_levels && target > self.skip_doc[level + 1] {
            level += 1;
        }
        loop {
            if target > self.skip_doc[level] {
                // `loadNextSkip` either reads an entry or exhausts the level
                // (`skip_doc[level] = MAX`), so this loop ends.
                self.load_next_skip(level)?;
                continue;
            }
            if level == 0 {
                return Ok(());
            }
            // ARITH: `level > 0` here.
            #[allow(clippy::arithmetic_side_effects)]
            let below = level - 1;
            // no more skips on this level, go down one level
            if self.last_child_pointer > self.stream[below] {
                self.seek_child(below)?;
            }
            level = below;
        }
    }

    /// `loadNextSkip(level)`.
    fn load_next_skip(&mut self, level: usize) -> Result<bool> {
        self.set_last_skip_data(level);
        // Java's `int` arithmetic; "numSkipped may overflow a signed int, so
        // compare as unsigned."
        self.num_skipped[level] = self.num_skipped[level].wrapping_add(self.skip_interval[level]);
        if (self.num_skipped[level] as u32) > (self.doc_count as u32) {
            // this skip list is exhausted
            self.skip_doc[level] = i32::MAX;
            if self.number_of_skip_levels > level {
                self.number_of_skip_levels = level;
            }
            return Ok(false);
        }
        // read next skip entry; Java's `int +=` (a missing `skipDoc` line
        // adds `NO_MORE_DOCS`).
        let delta = self.read_skip_data(level)?;
        self.skip_doc[level] = self.skip_doc[level].wrapping_add(delta);
        if level != 0 {
            // read the child pointer if we are not on the leaf level
            let child = self.read_child_pointer(level)?;
            // ARITH: `level != 0`.
            #[allow(clippy::arithmetic_side_effects)]
            let base = self.skip_pointer[level - 1];
            let at = i64::try_from(base)
                .ok()
                .and_then(|b| b.checked_add(child))
                .ok_or_else(|| corrupt(format!("invalid child pointer: {child}")))?;
            self.child_pointer[level] = self.file_pointer(at, "child pointer")?;
        }
        Ok(true)
    }

    /// `seekChild(level)`.
    fn seek_child(&mut self, level: usize) -> Result<()> {
        self.stream[level] = self.last_child_pointer;
        // ARITH: `level + 1 < MAX_SKIP_LEVELS` (the caller's `level - 1`);
        // Java's `int` subtraction.
        #[allow(clippy::arithmetic_side_effects)]
        let up = level + 1;
        self.num_skipped[level] = self.num_skipped[up].wrapping_sub(self.skip_interval[up]);
        self.skip_doc[level] = self.last_doc;
        if level > 0 {
            let child = self.read_child_pointer(level)?;
            // ARITH: `level > 0`.
            #[allow(clippy::arithmetic_side_effects)]
            let base = self.skip_pointer[level - 1];
            let at = i64::try_from(base)
                .ok()
                .and_then(|b| b.checked_add(child))
                .ok_or_else(|| corrupt(format!("invalid child pointer: {child}")))?;
            self.child_pointer[level] = self.file_pointer(at, "child pointer")?;
        }
        Ok(())
    }

    /// `setLastSkipData(level)`.
    fn set_last_skip_data(&mut self, level: usize) {
        self.last_doc = self.skip_doc[level];
        self.last_child_pointer = self.child_pointer[level];
    }

    /// `readSkipData(level, skipStream)`: one entry's lines up to
    /// `impactsEnd`; the delta to its `skipDoc`.
    fn read_skip_data(&mut self, level: usize) -> Result<i32> {
        self.per_level_impacts[level].clear();
        let mut skip_doc = NO_MORE_DOCS;
        // `new BufferedChecksumIndexInput(skipStream)`: checksums from here.
        let start = self.stream[level];
        let mut pos = start;
        let mut freq = 1;
        loop {
            util::read_line(self.data, &mut pos, &mut self.scratch)?;
            let line = self.scratch.as_slice();
            if line == super::fields_writer::END {
                util::check_footer(self.data, start, &mut pos)?;
                break;
            } else if line == IMPACTS_END
                || line == super::fields_writer::TERM
                || line == super::fields_writer::FIELD
            {
                break;
            } else if tail(line, SKIP_LIST).is_some() {
                // continue
            } else if let Some(v) = tail(line, SKIP_DOC) {
                // Because the MultiLevelSkipListReader stores doc id delta,
                // but simple text codec stores doc id
                skip_doc = parse_int(v)?.wrapping_sub(self.skip_doc[level]);
            } else if let Some(v) = tail(line, SKIP_DOC_FP) {
                self.next_skip_doc_fp = i64::from(parse_int(v)?);
            } else if tail(line, IMPACTS).is_some() || tail(line, IMPACT).is_some() {
                // continue
            } else if let Some(v) = tail(line, FREQ) {
                freq = parse_int(v)?;
            } else if let Some(v) = tail(line, NORM) {
                let norm = parse_long(v)?;
                self.per_level_impacts[level].push(Impact { freq, norm });
            }
        }
        self.stream[level] = pos;
        Ok(skip_doc)
    }

    /// The line at `stream[level]`, past `prefix`'s length (Java does not
    /// check the prefix itself).
    fn read_tail(&mut self, level: usize, prefix: &[u8]) -> Result<i64> {
        util::read_line(self.data, &mut self.stream[level], &mut self.scratch)?;
        let v = self
            .scratch
            .get(prefix.len()..)
            .ok_or_else(|| corrupt("skip list line shorter than its prefix"))?;
        parse_long(v)
    }

    /// `readLevelLength(skipStream)`.
    fn read_level_length(&mut self) -> Result<i64> {
        self.read_tail(0, LEVEL_LENGTH)
    }

    /// `readChildPointer(skipStream)`.
    fn read_child_pointer(&mut self, level: usize) -> Result<i64> {
        self.read_tail(level, CHILD_POINTER)
    }

    /// `getImpacts()`.
    pub fn impacts(&self) -> Impacts<'_> {
        Impacts::Skip(self)
    }

    /// `getNextSkipDocFP()`.
    pub fn next_skip_doc_fp(&self) -> i64 {
        self.next_skip_doc_fp
    }

    /// `getNextSkipDoc()`: `NO_MORE_DOCS` without a skip list.
    pub fn next_skip_doc(&self) -> i32 {
        if !self.has_skip_list {
            return NO_MORE_DOCS;
        }
        self.skip_doc[0]
    }

    /// `hasSkipList()`.
    pub fn has_skip_list(&self) -> bool {
        self.has_skip_list
    }
}
