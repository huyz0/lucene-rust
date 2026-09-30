//! The trailing multi-level skip list of the `Lucene90` and `Lucene99`
//! postings formats: ports of `codecs.MultiLevelSkipListReader`,
//! `backward_codecs.lucene90.Lucene90SkipReader` and
//! `Lucene90ScoreSkipReader` (`lucene99`'s two are the same classes renamed).
//!
//! Every term with more than one 128-document block of postings has its skip
//! data written after the postings, `skipOffset` bytes past `docStartFP`:
//! one level-0 entry per block (bar the last one when the term ends on a
//! block boundary), one level-1 entry per 8 level-0 entries, and so on, each
//! upper level's entries carrying a pointer to the matching entry below. An
//! entry at level 0 describes one block: its last document (a delta), where
//! the **next** block starts in `.doc`, the `.pos`/`.pay` pointers at that
//! point, and the block's competitive `(freq, norm)` impacts.
//!
//! `LazyDocsCursor` walks it the way `BlockImpactsDocsEnum.advanceShallow`
//! does: [`SkipList::skip_to`] positions on the block that can hold a target,
//! and the entry it stops on gives that block's extent and impacts without
//! touching the block itself.

use lucene_store::data_input::{DataInput, SliceInput};

use crate::postings::{Error, Result};

/// `Lucene90PostingsFormat.MAX_SKIP_LEVELS`.
pub(crate) const MAX_SKIP_LEVELS: usize = 10;
/// `ForUtil.BLOCK_SIZE`: the level-0 skip interval.
const SKIP_INTERVAL: i64 = 128;
/// The `skipMultiplier` `Lucene90SkipReader` passes (8).
const SKIP_MULTIPLIER: i64 = 8;

fn corrupted(msg: impl Into<String>) -> Error {
    Error::Store(lucene_store::Error::Corrupted(msg.into()))
}

/// What a term's field indexes, which decides the shape of every skip entry.
#[derive(Debug, Clone, Copy)]
pub(crate) struct SkipFields {
    pub has_pos: bool,
    pub has_offsets: bool,
    pub has_payloads: bool,
}

/// `Lucene90ScoreSkipReader` over one term.
#[derive(Debug, Clone)]
pub(crate) struct SkipList<'a> {
    buf: &'a [u8],
    fields: SkipFields,
    number_of_skip_levels: usize,
    /// `docCount`: the term's `docFreq`, trimmed by one when it is a multiple
    /// of the block size (`Lucene90SkipReader.trim`).
    doc_count: i64,
    /// Per level: where its stream reads next (`skipStream[i]`'s file pointer).
    stream: [usize; MAX_SKIP_LEVELS],
    /// `skipPointer`: where each level's entries start.
    skip_pointer: [usize; MAX_SKIP_LEVELS],
    skip_interval: [i64; MAX_SKIP_LEVELS],
    num_skipped: [i64; MAX_SKIP_LEVELS],
    /// `skipDoc`: the document of the entry each level last read.
    skip_doc: [i32; MAX_SKIP_LEVELS],
    last_doc: i32,
    child_pointer: [usize; MAX_SKIP_LEVELS],
    last_child_pointer: usize,
    doc_pointer: [i64; MAX_SKIP_LEVELS],
    pos_pointer: [i64; MAX_SKIP_LEVELS],
    pos_buffer_upto: [i32; MAX_SKIP_LEVELS],
    pay_pointer: [i64; MAX_SKIP_LEVELS],
    last_doc_pointer: i64,
    last_pos_pointer: i64,
    last_pos_buffer_upto: i32,
    last_pay_pointer: i64,
    /// `impactData[level]`: the serialized impacts of the entry each level
    /// last read, as a range of `buf` (decoded on demand, as Java does).
    impacts: [(usize, usize); MAX_SKIP_LEVELS],
    /// Whether [`Self::skip_to`] ever ran: `BlockImpactsDocsEnum.nextSkipDoc`
    /// starts at `-1`, not at the reader's initial `skipDoc[0] == 0`.
    positioned: bool,
}

impl<'a> SkipList<'a> {
    /// `Lucene90SkipReader.init(skipPointer, docBasePointer, posBasePointer,
    /// payBasePointer, df)`, with `MultiLevelSkipListReader.init` and
    /// `loadSkipLevels`.
    pub(crate) fn new(
        buf: &'a [u8],
        fields: SkipFields,
        skip_pointer: i64,
        doc_base: i64,
        pos_base: i64,
        pay_base: i64,
        doc_freq: i32,
    ) -> Result<Self> {
        let skip_pointer = usize::try_from(skip_pointer)
            .ok()
            .filter(|&p| p <= buf.len())
            .ok_or_else(|| corrupted(format!("skip pointer {skip_pointer} out of range")))?;
        // `trim`: a term that ends on a block boundary has no entry for its
        // last block.
        let df = i64::from(doc_freq);
        // ARITH: `df` is an `i32` widened to `i64`; one less cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        let doc_count = if df % SKIP_INTERVAL == 0 { df - 1 } else { df };
        let mut skip_interval = [0i64; MAX_SKIP_LEVELS];
        skip_interval[0] = SKIP_INTERVAL;
        for i in 1..MAX_SKIP_LEVELS {
            // ARITH: 128 * 8^9 = 2^37, far inside `i64`.
            #[allow(clippy::arithmetic_side_effects)]
            {
                skip_interval[i] = skip_interval[i - 1] * SKIP_MULTIPLIER;
            }
        }
        // `loadSkipLevels`: `1 + MathUtil.log(docCount / skipInterval, 8)`.
        let mut levels = if doc_count <= SKIP_INTERVAL {
            1
        } else {
            // ARITH: `doc_count > 128`, so the quotient is positive; the loop
            // divides it down and counts at most 21 steps.
            #[allow(clippy::arithmetic_side_effects)]
            {
                let mut x = doc_count / SKIP_INTERVAL;
                let mut log = 0usize;
                while x >= SKIP_MULTIPLIER {
                    x /= SKIP_MULTIPLIER;
                    log += 1;
                }
                1 + log
            }
        };
        levels = levels.min(MAX_SKIP_LEVELS);
        let mut list = SkipList {
            buf,
            fields,
            number_of_skip_levels: levels,
            doc_count,
            stream: [0; MAX_SKIP_LEVELS],
            skip_pointer: [0; MAX_SKIP_LEVELS],
            skip_interval,
            num_skipped: [0; MAX_SKIP_LEVELS],
            skip_doc: [0; MAX_SKIP_LEVELS],
            last_doc: 0,
            child_pointer: [0; MAX_SKIP_LEVELS],
            last_child_pointer: 0,
            doc_pointer: [doc_base; MAX_SKIP_LEVELS],
            pos_pointer: [pos_base; MAX_SKIP_LEVELS],
            pos_buffer_upto: [0; MAX_SKIP_LEVELS],
            pay_pointer: [pay_base; MAX_SKIP_LEVELS],
            last_doc_pointer: doc_base,
            last_pos_pointer: pos_base,
            last_pos_buffer_upto: 0,
            last_pay_pointer: pay_base,
            impacts: [(0, 0); MAX_SKIP_LEVELS],
            positioned: false,
        };
        let mut r = SliceInput::new(buf);
        r.seek(skip_pointer)?;
        for i in (1..levels).rev() {
            let length = r.read_vlong()?;
            let length = usize::try_from(length)
                .map_err(|_| corrupted(format!("skip level {i} length {length}")))?;
            let start = r.position();
            list.skip_pointer[i] = start;
            list.stream[i] = start;
            let end = start
                .checked_add(length)
                .ok_or_else(|| corrupted("skip level length overflows"))?;
            r.seek(end)?;
        }
        list.skip_pointer[0] = r.position();
        list.stream[0] = r.position();
        Ok(list)
    }

    /// `MultiLevelSkipListReader.skipTo`: positions every level on the last
    /// entry before `target`'s block, and returns how many documents precede
    /// that block, minus one (`numSkipped[0] - skipInterval[0] - 1`).
    pub(crate) fn skip_to(&mut self, target: i32) -> Result<i64> {
        self.positioned = true;
        let mut level = 0usize;
        // ARITH: `level + 1 < number_of_skip_levels <= MAX_SKIP_LEVELS`.
        #[allow(clippy::arithmetic_side_effects)]
        while level + 1 < self.number_of_skip_levels && target > self.skip_doc[level + 1] {
            level += 1;
        }
        loop {
            if target > self.skip_doc[level] {
                if !self.load_next_skip(level)? {
                    continue;
                }
            } else {
                // ARITH: guarded by `level > 0`.
                #[allow(clippy::arithmetic_side_effects)]
                if level > 0 && self.last_child_pointer > self.stream[level - 1] {
                    self.seek_child(level - 1)?;
                }
                if level == 0 {
                    break;
                }
                // ARITH: `level > 0`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    level -= 1;
                }
            }
        }
        Ok(self.num_skipped[0]
            .wrapping_sub(self.skip_interval[0])
            .wrapping_sub(1))
    }

    /// `loadNextSkip`: reads the next entry of `level`, after saving the
    /// current one as the last read; `false` once the level is exhausted
    /// (its `skipDoc` then `Integer.MAX_VALUE`).
    fn load_next_skip(&mut self, level: usize) -> Result<bool> {
        self.set_last_skip_data(level);
        self.num_skipped[level] = self.num_skipped[level].wrapping_add(self.skip_interval[level]);
        // `Integer.compareUnsigned(numSkipped[level], docCount) > 0`.
        if self.num_skipped[level] > self.doc_count {
            self.skip_doc[level] = i32::MAX;
            if self.number_of_skip_levels > level {
                self.number_of_skip_levels = level;
            }
            return Ok(false);
        }
        let mut r = SliceInput::new(self.buf);
        r.seek(self.stream[level])?;
        let delta = self.read_skip_data(level, &mut r)?;
        self.skip_doc[level] = self.skip_doc[level].wrapping_add(delta);
        if level != 0 {
            let child = r.read_vlong()?;
            // ARITH: guarded by `level != 0`.
            #[allow(clippy::arithmetic_side_effects)]
            let below = self.skip_pointer[level - 1];
            self.child_pointer[level] = usize::try_from(child)
                .ok()
                .and_then(|c| c.checked_add(below))
                .ok_or_else(|| corrupted(format!("skip child pointer {child}")))?;
        }
        self.stream[level] = r.position();
        Ok(true)
    }

    /// `seekChild` (with `Lucene90SkipReader`'s override): repositions
    /// `level` on the entry its parent's last read entry points at.
    fn seek_child(&mut self, level: usize) -> Result<()> {
        self.stream[level] = self.last_child_pointer;
        // ARITH: `level + 1 <` the parent's level count.
        #[allow(clippy::arithmetic_side_effects)]
        {
            self.num_skipped[level] =
                self.num_skipped[level + 1].wrapping_sub(self.skip_interval[level + 1]);
        }
        self.skip_doc[level] = self.last_doc;
        if level > 0 {
            let mut r = SliceInput::new(self.buf);
            r.seek(self.stream[level])?;
            let child = r.read_vlong()?;
            // ARITH: guarded by `level > 0`.
            #[allow(clippy::arithmetic_side_effects)]
            let below = self.skip_pointer[level - 1];
            self.child_pointer[level] = usize::try_from(child)
                .ok()
                .and_then(|c| c.checked_add(below))
                .ok_or_else(|| corrupted(format!("skip child pointer {child}")))?;
            self.stream[level] = r.position();
        }
        self.doc_pointer[level] = self.last_doc_pointer;
        if self.fields.has_pos {
            self.pos_pointer[level] = self.last_pos_pointer;
            self.pos_buffer_upto[level] = self.last_pos_buffer_upto;
            if self.fields.has_offsets || self.fields.has_payloads {
                self.pay_pointer[level] = self.last_pay_pointer;
            }
        }
        Ok(())
    }

    /// `setLastSkipData`.
    fn set_last_skip_data(&mut self, level: usize) {
        self.last_doc = self.skip_doc[level];
        self.last_child_pointer = self.child_pointer[level];
        self.last_doc_pointer = self.doc_pointer[level];
        if self.fields.has_pos {
            self.last_pos_pointer = self.pos_pointer[level];
            self.last_pos_buffer_upto = self.pos_buffer_upto[level];
            if self.fields.has_offsets || self.fields.has_payloads {
                self.last_pay_pointer = self.pay_pointer[level];
            }
        }
    }

    /// `Lucene90SkipReader.readSkipData` + `Lucene90ScoreSkipReader.
    /// readImpacts`: one entry, returning its document delta.
    fn read_skip_data(&mut self, level: usize, r: &mut SliceInput<'a>) -> Result<i32> {
        let delta = r.read_vint()?;
        self.doc_pointer[level] = self.doc_pointer[level].wrapping_add(r.read_vlong()?);
        if self.fields.has_pos {
            self.pos_pointer[level] = self.pos_pointer[level].wrapping_add(r.read_vlong()?);
            self.pos_buffer_upto[level] = r.read_vint()?;
            if self.fields.has_payloads {
                // `payloadByteUpto`: recomputed from the landing block's own
                // lengths by the positions reader, never read from here.
                r.read_vint()?;
            }
            if self.fields.has_offsets || self.fields.has_payloads {
                self.pay_pointer[level] = self.pay_pointer[level].wrapping_add(r.read_vlong()?);
            }
        }
        let len = r.read_length("skip entry impacts")?;
        let start = r.position();
        let end = start
            .checked_add(len)
            .ok_or_else(|| corrupted("skip entry impacts overflow"))?;
        r.seek(end)?;
        self.impacts[level] = (start, end);
        Ok(delta)
    }

    /// `getDoc()`: the last document before the block `skip_to` stopped on.
    pub(crate) fn last_doc(&self) -> i32 {
        self.last_doc
    }

    /// `getDocPointer()`: where that block starts in `.doc`.
    pub(crate) fn doc_pointer(&self) -> i64 {
        self.last_doc_pointer
    }

    /// `getNextSkipDoc()`: the last document of the block `skip_to` stopped
    /// on, or `i32::MAX` past the last entry.
    // SENTINEL: `-1` = "never positioned" (no `skip_to` yet), below every
    // target, as `BlockImpactsDocsEnum.nextSkipDoc` starts.
    pub(crate) fn next_skip_doc(&self) -> i32 {
        if self.positioned {
            self.skip_doc[0]
        } else {
            -1
        }
    }

    /// Where the block `skip_to` stopped on ends -- the next block's start,
    /// which that block's own entry records. Meaningful only while
    /// [`Self::next_skip_doc`] is a real document.
    pub(crate) fn next_doc_pointer(&self) -> i64 {
        self.doc_pointer[0]
    }

    /// The block's serialized level-0 impacts (`getImpacts().getImpacts(0)`
    /// before decoding), or empty past the last entry.
    pub(crate) fn level0_impacts(&self) -> &'a [u8] {
        if self.skip_doc[0] == i32::MAX {
            return &[];
        }
        let (a, b) = self.impacts[0];
        self.buf.get(a..b).unwrap_or(&[])
    }

    /// `getImpacts()` at level 1 after [`Self::skip_to`]: the last document
    /// its level-1 entry covers and that entry's serialized impacts, or
    /// `None` when the term has no second level (or has run past it).
    pub(crate) fn level1(&self) -> Option<(i32, &'a [u8])> {
        if self.number_of_skip_levels < 2 || self.skip_doc[1] == i32::MAX || !self.positioned {
            return None;
        }
        let (a, b) = self.impacts[1];
        Some((self.skip_doc[1], self.buf.get(a..b)?))
    }

    /// `getPosPointer()`/`getPosBufferUpto()`/`getPayPointer()`: the
    /// `.pos`/`.pay` state at the start of the block `skip_to` stopped on.
    pub(crate) fn pos_state(&self) -> (i64, i32, i64) {
        (
            self.last_pos_pointer,
            self.last_pos_buffer_upto,
            self.last_pay_pointer,
        )
    }
}

#[cfg(test)]
#[allow(clippy::arithmetic_side_effects)]
mod tests {
    use super::*;

    fn vint(out: &mut Vec<u8>, mut v: u64) {
        while v >= 0x80 {
            out.push((v as u8) | 0x80);
            v >>= 7;
        }
        out.push(v as u8);
    }

    /// Block `i`'s wire values: its entry's `.doc`/`.pos`/`.pay` pointers
    /// (the *next* block's start) and `posBufferUpto`.
    fn doc_ptr(i: usize) -> i64 {
        1000 * (i as i64 + 1)
    }
    fn pos_ptr(i: usize) -> i64 {
        10 * (i as i64 + 1)
    }
    fn pay_ptr(i: usize) -> i64 {
        7 * (i as i64 + 1)
    }
    fn pos_upto(i: usize) -> i32 {
        (i % 5) as i32
    }

    /// `Lucene90SkipWriter` + `MultiLevelSkipListWriter` for a term whose
    /// entry-carrying blocks end on `block_last` (the writer buffers a
    /// block's entry when the next block starts). Base pointers are 0, each
    /// entry's impacts are the one byte `i`. Returns the serialized skip
    /// data after `pad` filler bytes.
    fn write(block_last: &[i32], fields: SkipFields, levels: usize, pad: usize) -> Vec<u8> {
        let mut bufs: Vec<Vec<u8>> = vec![Vec::new(); levels];
        let mut last_doc = vec![0i32; levels];
        let mut last_doc_ptr = vec![0i64; levels];
        let mut last_pos_ptr = vec![0i64; levels];
        let mut last_pay_ptr = vec![0i64; levels];
        for (i, &doc) in block_last.iter().enumerate() {
            // `bufferSkip(df)`: how many levels this entry reaches.
            let mut blocks = i as i64 + 1;
            let mut n = 1;
            while blocks % SKIP_MULTIPLIER == 0 && n < levels {
                n += 1;
                blocks /= SKIP_MULTIPLIER;
            }
            let mut child = 0u64;
            for l in 0..n {
                let b = &mut bufs[l];
                vint(b, (doc - last_doc[l]) as u64);
                last_doc[l] = doc;
                vint(b, (doc_ptr(i) - last_doc_ptr[l]) as u64);
                last_doc_ptr[l] = doc_ptr(i);
                if fields.has_pos {
                    vint(b, (pos_ptr(i) - last_pos_ptr[l]) as u64);
                    last_pos_ptr[l] = pos_ptr(i);
                    vint(b, pos_upto(i) as u64);
                    if fields.has_payloads {
                        vint(b, 3);
                    }
                    if fields.has_offsets || fields.has_payloads {
                        vint(b, (pay_ptr(i) - last_pay_ptr[l]) as u64);
                        last_pay_ptr[l] = pay_ptr(i);
                    }
                }
                b.push(1);
                b.push(i as u8);
                let new_child = b.len() as u64;
                if l != 0 {
                    vint(b, child);
                }
                child = new_child;
            }
        }
        // `writeSkip`: levels top-down, each but level 0 length-prefixed.
        let mut out = vec![0xAAu8; pad];
        for l in (1..levels).rev() {
            vint(&mut out, bufs[l].len() as u64);
            out.extend_from_slice(&bufs[l]);
        }
        out.extend_from_slice(&bufs[0]);
        out
    }

    const DOCS_ONLY: SkipFields = SkipFields {
        has_pos: false,
        has_offsets: false,
        has_payloads: false,
    };
    const EVERYTHING: SkipFields = SkipFields {
        has_pos: true,
        has_offsets: true,
        has_payloads: true,
    };

    /// A term of `df` documents `0, 3, 6, ...`: the last documents of its
    /// entry-carrying blocks, the number of levels Lucene writes, and the
    /// documents.
    fn term(df: i32) -> (Vec<i32>, usize, Vec<i32>) {
        let docs: Vec<i32> = (0..df).map(|i| i * 3).collect();
        let full = df as usize / 128;
        // The last full block carries no entry when nothing follows it.
        let entries = if (df as usize).is_multiple_of(128) {
            full - 1
        } else {
            full
        };
        let block_last: Vec<i32> = (0..entries).map(|b| docs[128 * b + 127]).collect();
        let dc = i64::from(if df % 128 == 0 { df - 1 } else { df });
        let mut levels = 1;
        let mut x = dc / 128;
        while dc > 128 && x >= 8 {
            x /= 8;
            levels += 1;
        }
        (block_last, levels, docs)
    }

    fn check(df: i32, fields: SkipFields, targets: impl Iterator<Item = i32>, fresh: bool) {
        let (block_last, levels, docs) = term(df);
        let buf = write(&block_last, fields, levels, 5);
        let open = || SkipList::new(&buf, fields, 5, 0, 0, 0, df).unwrap();
        let mut s = open();
        assert_eq!(s.number_of_skip_levels, levels, "df {df}");
        assert_eq!(s.next_skip_doc(), -1);
        assert!(s.level1().is_none());
        let mut checked = 0;
        for t in targets {
            // `skipDoc[0]` starts at 0, so `skipTo(0)` reads no entry; the
            // cursor asks for at least 1, as `advanceShallow` does.
            let t = t.max(1);
            if fresh {
                s = open();
            }
            let at = docs.partition_point(|&d| d < t);
            if at == docs.len() {
                break;
            }
            let b = at / 128;
            s.skip_to(t).unwrap();
            checked += 1;
            let ctx = format!("df {df} target {t} block {b}");
            if b == 0 {
                assert_eq!((s.last_doc(), s.doc_pointer()), (0, 0), "{ctx}");
                assert_eq!(s.pos_state(), (0, 0, 0), "{ctx}");
            } else {
                assert_eq!(s.last_doc(), block_last[b - 1], "{ctx}");
                assert_eq!(s.doc_pointer(), doc_ptr(b - 1), "{ctx}");
                if fields.has_pos {
                    assert_eq!(
                        s.pos_state(),
                        (pos_ptr(b - 1), pos_upto(b - 1), pay_ptr(b - 1)),
                        "{ctx}"
                    );
                }
            }
            if b < block_last.len() {
                assert_eq!(s.next_skip_doc(), block_last[b], "{ctx}");
                assert_eq!(s.next_doc_pointer(), doc_ptr(b), "{ctx}");
                assert_eq!(s.level0_impacts(), &[b as u8], "{ctx}");
            } else {
                assert_eq!(s.next_skip_doc(), i32::MAX, "{ctx}");
                assert!(s.level0_impacts().is_empty(), "{ctx}");
            }
            let span_end = b / 8 * 8 + 7;
            if levels >= 2 && span_end < block_last.len() {
                assert_eq!(
                    s.level1(),
                    Some((block_last[span_end], &[span_end as u8][..])),
                    "{ctx}"
                );
            } else {
                assert!(s.level1().is_none(), "{ctx}");
            }
        }
        assert!(checked > 3, "df {df}: only {checked} targets");
    }

    #[test]
    fn skip_to_lands_on_the_block_holding_the_target_at_every_level() {
        // 3 levels (entries every 128, 1,024 and 8,192 docs) and a tail.
        let df = 128 * 70 + 5;
        check(df, DOCS_ONLY, (0..df * 3 + 2).step_by(7), false);
        check(df, EVERYTHING, (0..df * 3 + 2).step_by(389), false);
        // A cold list per target: the descent from the top level alone.
        check(df, EVERYTHING, (0..df * 3).step_by(1531), true);
        // Ends on a block boundary: `trim` drops the last block's entry.
        check(128 * 16, DOCS_ONLY, (0..128 * 48).step_by(5), false);
        // One entry, one level.
        check(129, EVERYTHING, 0..400, false);
    }

    #[test]
    fn skip_to_jumps_across_whole_upper_level_spans() {
        let df = 128 * 600 + 1;
        let (_, levels, _) = term(df);
        assert_eq!(levels, 4);
        check(
            df,
            DOCS_ONLY,
            [1, 50_000, 50_001, 190_000, 230_390].into_iter(),
            false,
        );
    }

    #[test]
    fn corrupt_skip_data_is_an_error_not_a_panic() {
        let (block_last, levels, _) = term(128 * 70 + 5);
        let buf = write(&block_last, DOCS_ONLY, levels, 0);
        for bad in [-1, buf.len() as i64 + 1] {
            assert!(SkipList::new(&buf, DOCS_ONLY, bad, 0, 0, 0, 8965).is_err());
        }
        // A level length past the end of the data.
        let mut long = buf.clone();
        long[0] = 0x7f;
        long.truncate(40);
        assert!(SkipList::new(&long, DOCS_ONLY, 0, 0, 0, 0, 8965).is_err());
        // A negative level length.
        let negative = [0xffu8, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0x01];
        assert!(SkipList::new(&negative, DOCS_ONLY, 0, 0, 0, 0, 8965).is_err());
        // Every truncation, and every byte flipped: an error or an answer,
        // never a panic.
        for cut in 0..buf.len() {
            if let Ok(mut s) = SkipList::new(&buf[..cut], DOCS_ONLY, 0, 0, 0, 0, 8965) {
                let _ = s.skip_to(26_000);
            }
        }
        for i in 0..buf.len() {
            let mut b = buf.clone();
            b[i] ^= 0xff;
            if let Ok(mut s) = SkipList::new(&b, DOCS_ONLY, 0, 0, 0, 0, 8965) {
                for t in [100, 5_000, 26_000] {
                    if s.skip_to(t).is_err() {
                        break;
                    }
                    let _ = (s.level0_impacts(), s.level1(), s.next_doc_pointer());
                }
            }
        }
    }

    #[test]
    fn a_child_pointer_that_does_not_fit_is_corrupt() {
        // Two levels by hand; the level-1 entry's child pointer is -1.
        let mut level1 = Vec::new();
        vint(&mut level1, 1023);
        vint(&mut level1, 8000);
        level1.push(0); // no impacts
        level1.extend_from_slice(&[0xff; 9]);
        level1.push(0x01);
        let mut buf = Vec::new();
        vint(&mut buf, level1.len() as u64);
        buf.extend_from_slice(&level1);
        buf.extend_from_slice(&[1, 1, 0]); // one level-0 entry
        let mut s = SkipList::new(&buf, DOCS_ONLY, 0, 0, 0, 0, 1100).unwrap();
        assert_eq!(s.number_of_skip_levels, 2);
        assert!(s.skip_to(1100).is_err());
    }
}
