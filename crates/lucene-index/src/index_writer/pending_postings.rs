//! The buffered segment's postings, inverted as each document arrives --
//! `FreqProxTermsWriter`'s per-field `BytesRefHash`, `ParallelPostingsArray`
//! and byte slices, filled by `IndexingChain.processDocument`.
//!
//! A term is hashed once, into its field's [`TermHash`], and each occurrence
//! is appended to the term's two byte streams as `FreqProxTermsWriterPerField`
//! writes them: the documents as vInt deltas, a frequency of one folded into
//! the delta's low bit, each written when the term's next document arrives
//! (the newest is held in the term's [`TermRun`]); and for a field that
//! indexes them, each occurrence's position delta, its payload and its
//! offsets. Every stream of a field lives in one [`BytePool`] of 32 KB blocks
//! as a chain of slices that grow by level, as Lucene's `ByteSlicePool` keeps
//! them, so a term costs no allocation of its own and the whole buffer is a
//! few large blocks. A document's positions count equals its frequency, so a
//! stream of positions needs no per-document boundaries.
//!
//! A document (or block) that fails part-way leaves nothing behind:
//! [`PendingPostings::rollback`] cuts every stream back to a [`Checkpoint`]
//! taken before it. Java instead keeps the partial postings and marks the
//! document deleted; this port never buffered a failed document.
//!
//! The flush decodes each field's streams into contiguous columns, terms in
//! byte order and documents renumbered into an index sort's order when there
//! is one ([`PendingPostings::flushed`]), and hands the postings writer a
//! [`TermView`] of each term, read in place.

use std::collections::BTreeMap;

use lucene_codecs::field_infos::IndexOptions;
use lucene_codecs::postings_writer::TermSource;

use super::explicit::{explicit_error, InvertedField};
use super::Result;
use crate::inverter::{hash_bytes, TermHash};

mod live;
pub(crate) use live::LiveFieldStats;

/// Bytes per full pool block: `ByteBlockPool.BYTE_BLOCK_SIZE`. A field's
/// first blocks are smaller (1, 2, 4, 8 and 16 KB), so a field with few
/// terms does not hold a whole block: Lucene shares one pool across a
/// buffer's fields, this port keeps one per field.
const BLOCK_SHIFT: u32 = 15;
const BLOCK_SIZE: usize = 1 << BLOCK_SHIFT;
const BLOCK_MASK: u32 = (1 << BLOCK_SHIFT) - 1;
const FIRST_SHIFT: u32 = 10;
const SMALL_BLOCKS: usize = (BLOCK_SHIFT - FIRST_SHIFT) as usize;
/// Bytes the small blocks span: `BLOCK_SIZE - 2^FIRST_SHIFT`.
const SMALL_SPAN: u32 = (1 << BLOCK_SHIFT) - (1 << FIRST_SHIFT);

/// Block `b`'s size.
fn block_size(b: usize) -> usize {
    if b < SMALL_BLOCKS {
        1usize << (FIRST_SHIFT as usize).saturating_add(b)
    } else {
        BLOCK_SIZE
    }
}

/// Block `b`'s first address.
fn block_start(b: usize) -> usize {
    if b < SMALL_BLOCKS {
        ((1usize << b).saturating_sub(1)) << FIRST_SHIFT
    } else {
        (SMALL_SPAN as usize).saturating_add(b.saturating_sub(SMALL_BLOCKS) << BLOCK_SHIFT)
    }
}

/// The block holding address `at`, and the offset in it.
#[inline]
fn locate(at: u32) -> (usize, usize) {
    if at < SMALL_SPAN {
        // block b spans [(2^b - 1) KB, (2^(b+1) - 1) KB)
        let q = (at >> FIRST_SHIFT).wrapping_add(1);
        let b = 31u32.wrapping_sub(q.leading_zeros());
        let start = ((1u32 << b).wrapping_sub(1)) << FIRST_SHIFT;
        (b as usize, at.wrapping_sub(start) as usize)
    } else {
        let rel = at.wrapping_sub(SMALL_SPAN);
        (
            SMALL_BLOCKS.wrapping_add((rel >> BLOCK_SHIFT) as usize),
            (rel & BLOCK_MASK) as usize,
        )
    }
}

/// Bytes per slice at each level: `ByteSlicePool.LEVEL_SIZE_ARRAY`. A
/// slice's last byte marks its end (`16 | level`) until the stream outgrows
/// it; then its last four bytes link the next slice.
const LEVEL_SIZES: [u32; 10] = [5, 14, 20, 30, 40, 40, 80, 80, 120, 200];
/// `ByteSlicePool.NEXT_LEVEL_ARRAY`.
const NEXT_LEVEL: [u8; 10] = [1, 2, 3, 4, 5, 6, 7, 8, 9, 9];

/// A stream with no slice yet.
const NONE: u32 = u32::MAX;

fn level_size(level: u8) -> u32 {
    LEVEL_SIZES[usize::from(level).min(LEVEL_SIZES.len().saturating_sub(1))]
}

fn next_level(level: u8) -> u8 {
    NEXT_LEVEL[usize::from(level).min(NEXT_LEVEL.len().saturating_sub(1))]
}

fn to_u32(n: usize) -> u32 {
    u32::try_from(n).unwrap_or(u32::MAX)
}

/// One field's streams: `ByteBlockPool` with `ByteSlicePool`'s slices.
#[derive(Debug, Default)]
struct BytePool {
    blocks: Vec<Box<[u8]>>,
    /// Where the next slice goes in the last block.
    used: usize,
}

/// Where a [`SliceReader`] stands: enough to cut its stream there.
#[derive(Debug, Clone, Copy)]
struct SlicePos {
    upto: u32,
    slice_start: u32,
    level: u8,
}

impl BytePool {
    #[inline]
    fn byte(&self, at: u32) -> u8 {
        let (b, o) = locate(at);
        self.blocks[b][o]
    }

    #[inline]
    fn set(&mut self, at: u32, v: u8) {
        let (b, o) = locate(at);
        self.blocks[b][o] = v;
    }

    fn ram_bytes(&self) -> usize {
        self.blocks
            .iter()
            .map(|b| b.len())
            .fold(0, usize::saturating_add)
    }

    /// Room for a slice of `size` bytes, zeroed, never across blocks. A
    /// buffer flushes long before a field's pool nears 4 GB
    /// (`HARD_MAX_BYTES_PER_SLOT`), so addresses fit a `u32`.
    fn alloc(&mut self, size: u32) -> u32 {
        let size = size as usize;
        let full = self
            .blocks
            .last()
            .is_none_or(|b| self.used.saturating_add(size) > b.len());
        if full {
            let len = block_size(self.blocks.len());
            self.blocks.push(vec![0u8; len].into_boxed_slice());
            self.used = 0;
        }
        let block = self.blocks.len().saturating_sub(1);
        let at = block_start(block).saturating_add(self.used);
        self.used = self.used.saturating_add(size);
        to_u32(at)
    }

    /// A new stream's first slice: `ByteSlicePool.newSlice`. Returns where
    /// it starts, which is where it writes first.
    fn new_slice(&mut self) -> u32 {
        let size = level_size(0);
        let at = self.alloc(size);
        self.set(at.wrapping_add(size).wrapping_sub(1), 16);
        at
    }

    /// Appends `b` at `upto`, moving to the next slice at an end marker.
    #[inline]
    fn write_byte(&mut self, upto: &mut u32, b: u8) {
        if self.byte(*upto) != 0 {
            *upto = self.next_slice(*upto);
        }
        self.set(*upto, b);
        *upto = upto.wrapping_add(1);
    }

    /// `ByteSlicePool.allocSlice`: links a slice one level larger after the
    /// one whose end marker is at `upto`, moving the three bytes before the
    /// marker into it so the link fits. Returns where the stream continues.
    #[cold]
    fn next_slice(&mut self, upto: u32) -> u32 {
        let level = next_level(self.byte(upto) & 15);
        let size = level_size(level);
        let at = self.alloc(size);
        let from = upto.wrapping_sub(3);
        for i in 0..3u32 {
            let b = self.byte(from.wrapping_add(i));
            self.set(at.wrapping_add(i), b);
        }
        for (i, &b) in at.to_le_bytes().iter().enumerate() {
            self.set(from.wrapping_add(i as u32), b);
        }
        self.set(at.wrapping_add(size).wrapping_sub(1), 16 | level);
        at.wrapping_add(3)
    }

    #[inline]
    fn write_vlong(&mut self, upto: &mut u32, mut v: u64) {
        while v >= 0x80 {
            self.write_byte(upto, (v as u8) | 0x80);
            v >>= 7;
        }
        self.write_byte(upto, v as u8);
    }

    /// The stream from `start` to `end` (its write position).
    fn reader(&self, start: u32, end: u32) -> SliceReader<'_> {
        let first = level_size(0);
        let (upto, limit) = if start == NONE {
            (end, end)
        } else if start.saturating_add(first) >= end {
            (start, end)
        } else {
            (start, start.wrapping_add(first).wrapping_sub(4))
        };
        SliceReader {
            pool: self,
            cur: self.span(upto, limit),
            limit,
            end,
            level: 0,
            slice_start: start,
        }
    }

    /// The bytes `[from, to)`, which never cross a block: one slice's.
    #[inline]
    fn span(&self, from: u32, to: u32) -> &[u8] {
        if from >= to {
            return &[];
        }
        let (b, o) = locate(from);
        let len = to.wrapping_sub(from) as usize;
        &self.blocks[b][o..o.saturating_add(len)]
    }

    /// Cuts a stream back to `pos`: the rest of its slice is cleared and
    /// marked as its end again, so the next write lands at `pos`. Returns
    /// the stream's new write position.
    fn cut(&mut self, pos: SlicePos) -> u32 {
        let marker = pos
            .slice_start
            .wrapping_add(level_size(pos.level))
            .wrapping_sub(1);
        let mut at = pos.upto;
        while at < marker {
            self.set(at, 0);
            at = at.wrapping_add(1);
        }
        self.set(marker, 16 | pos.level);
        pos.upto
    }

    /// Starts loading the block bytes at `at`.
    #[inline]
    fn prefetch(&self, at: u32) {
        let (b, o) = locate(at);
        if let Some(block) = self.blocks.get(b) {
            lucene_util::simd::prefetch(block, o);
        }
    }
}

/// `ByteSliceReader`: a stream's bytes in order, following its links.
struct SliceReader<'a> {
    pool: &'a BytePool,
    /// The current slice's bytes still to read, up to `limit`.
    cur: &'a [u8],
    /// Where this slice's bytes stop: the link, or the stream's end.
    limit: u32,
    end: u32,
    level: u8,
    slice_start: u32,
}

impl SliceReader<'_> {
    fn upto(&self) -> u32 {
        self.limit.wrapping_sub(self.cur.len() as u32)
    }

    fn eof(&self) -> bool {
        self.cur.is_empty() && self.limit == self.end
    }

    fn pos(&self) -> SlicePos {
        SlicePos {
            upto: self.upto(),
            slice_start: self.slice_start,
            level: self.level,
        }
    }

    #[inline]
    fn read_byte(&mut self) -> u8 {
        if self.cur.is_empty() {
            self.next_slice();
        }
        match self.cur.split_first() {
            Some((&b, rest)) => {
                self.cur = rest;
                b
            }
            None => 0,
        }
    }

    #[cold]
    fn next_slice(&mut self) {
        let mut link = [0u8; 4];
        link.copy_from_slice(
            self.pool
                .span(self.limit, self.limit.wrapping_add(4))
                .get(..4)
                .unwrap_or(&[0; 4]),
        );
        let next = u32::from_le_bytes(link);
        self.level = next_level(self.level);
        let size = level_size(self.level);
        self.slice_start = next;
        self.limit = if next.saturating_add(size) >= self.end {
            self.end
        } else {
            next.wrapping_add(size).wrapping_sub(4)
        };
        self.cur = self.pool.span(next, self.limit);
    }

    /// A vLong; read in place when the slice holds the longest one.
    #[inline]
    fn read_vlong(&mut self) -> u64 {
        if let Some(head) = self.cur.get(..10) {
            let mut v = 0u64;
            for (i, &b) in head.iter().enumerate() {
                v |= u64::from(b & 0x7f) << 7u32.wrapping_mul(i as u32);
                if b < 0x80 {
                    self.cur = self.cur.get(i.wrapping_add(1)..).unwrap_or_default();
                    return v;
                }
            }
        }
        let mut v = 0u64;
        let mut shift = 0u32;
        loop {
            let b = self.read_byte();
            v |= u64::from(b & 0x7f).checked_shl(shift).unwrap_or(0);
            if b < 0x80 {
                return v;
            }
            shift = shift.saturating_add(7);
        }
    }

    /// A vInt written from an `i32`'s bits.
    #[inline]
    fn read_i32(&mut self) -> i32 {
        self.read_vlong() as u32 as i32
    }
}

/// One term's newest document and where its streams continue:
/// `FreqProxPostingsArray`'s per-term ints, half a cache line, so adding a
/// term touches one line. The streams' starts are in
/// [`FieldRuns::starts`], read only at a flush or rollback.
#[derive(Debug, Clone, Copy)]
#[repr(C, align(32))]
struct TermRun {
    /// Where the documents stream and the occurrences stream write next.
    docs: u32,
    prox: u32,
    /// Documents, the newest included though not yet written.
    doc_count: u32,
    /// Positions written: the sum of the documents' frequencies.
    occurrences: u32,
    last_doc: i32,
    /// `last_doc` minus the document before it: what the documents stream
    /// gets for it once the term's next document arrives.
    last_doc_code: i32,
    /// `last_doc`'s frequency so far.
    freq: i32,
    /// The newest occurrence's position in `last_doc`.
    last_pos: i32,
}

impl TermRun {
    const NEW: TermRun = TermRun {
        docs: NONE,
        prox: NONE,
        doc_count: 0,
        occurrences: 0,
        last_doc: -1,
        last_doc_code: 0,
        freq: 0,
        last_pos: 0,
    };
}

/// What [`FieldRuns::occurrence`] found for a term in the current document.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TermSlot {
    id: u32,
    /// The term was new to the buffered segment: `newTerm`.
    pub(crate) new_to_segment: bool,
    /// The term's frequency in the document so far; `None` before its
    /// first occurrence there.
    pub(crate) freq: Option<i32>,
}

/// One field's dictionary under construction.
#[derive(Debug)]
pub(crate) struct FieldRuns {
    positions: bool,
    offsets: bool,
    terms: TermHash,
    /// By term id (`TermHash` ids are dense, in insertion order).
    runs: Vec<TermRun>,
    /// By term id: where the documents and occurrences streams start.
    starts: Vec<[u32; 2]>,
    /// By term id for a field with offsets, else empty: the newest
    /// occurrence's start offset in its document (`lastOffsets`).
    last_offsets: Vec<i32>,
    pool: BytePool,
    /// Some occurrence carried a payload: `sawPayloads`.
    payloads: bool,
    /// Documents with at least one term in the field: the postings
    /// `docCount`.
    doc_count: i32,
    last_doc: i32,
    /// The field's `sumDocFreq` and `sumTotalTermFreq` so far, kept as
    /// documents arrive for a search over the buffer.
    sum_doc_freq: i64,
    sum_total_term_freq: i64,
}

/// How many terms ahead of the one being added [`FieldRuns::prefetch_ahead`]
/// starts loading the likely term's bytes and run, and (half as far) the pool
/// bytes that run appends to.
const PREFETCH_TERMS: usize = 8;

/// How many terms ahead of the one being laid out [`FieldRuns::flushed`]
/// starts loading a term's run and starts, and (half as far) its bytes and
/// streams.
const LAYOUT_PREFETCH: usize = 16;

/// A streamed flush hands the writer a batch once it holds this many terms,
/// or [`STREAM_ENTRIES`] documents or positions: small enough to stay in
/// cache, large enough that a batch's fixed cost does not show.
const STREAM_TERMS: usize = 1024;
const STREAM_ENTRIES: usize = 1 << 15;

/// One occurrence's decoded parts.
#[derive(Debug, Clone, Copy, Default)]
struct Occurrence {
    position: i32,
    offsets: (i32, i32),
}

/// A term's columns decoded from its streams: what [`TermView`] reads.
#[derive(Debug, Default)]
struct Columns {
    docs: Vec<(i32, i32)>,
    /// Per document, where its positions end within its term's: parallel
    /// to `docs` for a field with positions, else empty.
    ends: Vec<u32>,
    positions: Vec<i32>,
    offsets: Vec<(i32, i32)>,
    payload_lengths: Vec<u32>,
    payload_bytes: Vec<u8>,
}

impl Columns {
    fn clear(&mut self) {
        self.docs.clear();
        self.ends.clear();
        self.positions.clear();
        self.offsets.clear();
        self.payload_lengths.clear();
        self.payload_bytes.clear();
    }
}

/// A term's fixed charge: its `ParallelPostingsArray` entries, its hash
/// slot's share, its start and hash.
const TERM_BYTES: usize = 24 + 4 + 4 + 8;

impl FieldRuns {
    fn new(index_options: IndexOptions) -> Self {
        FieldRuns {
            positions: index_options.subsumes_positions(),
            offsets: index_options.subsumes_offsets(),
            terms: TermHash::new(),
            runs: Vec::new(),
            starts: Vec::new(),
            last_offsets: Vec::new(),
            pool: BytePool::default(),
            payloads: false,
            doc_count: 0,
            last_doc: -1,
            sum_doc_freq: 0,
            sum_total_term_freq: 0,
        }
    }

    /// The field's share of the RAM buffer, as `bytesUsed` counts it: the
    /// pool blocks it took, and per term [`TERM_BYTES`] plus the term's
    /// bytes. The run and dictionary arrays' spare capacity is not charged,
    /// as Java charges its postings arrays by entry, so a buffer flushes
    /// after the documents Java's would.
    fn ram_bytes(&self) -> usize {
        self.runs
            .len()
            .saturating_mul(TERM_BYTES)
            .saturating_add(self.terms.term_bytes())
            .saturating_add(self.pool.ram_bytes())
    }

    /// Starts loading what adding the terms after `i` will read: for the
    /// term [`PREFETCH_TERMS`] ahead, the run and term start of the id its
    /// first table slot names (that slot was prefetched with every hash);
    /// for the term half as far ahead, whose run and start are in cache by
    /// now, its bytes and the pool bytes its occurrences go to next.
    #[inline]
    fn prefetch_ahead(&self, hashes: &[u64], i: usize) {
        if let Some(id) = hashes
            .get(i.saturating_add(PREFETCH_TERMS))
            .and_then(|&h| self.terms.likely_id(h))
        {
            self.terms.prefetch_start(id);
            lucene_util::simd::prefetch(&self.runs, id as usize);
        }
        if let Some(id) = hashes
            .get(i.saturating_add(PREFETCH_TERMS / 2))
            .and_then(|&h| self.terms.likely_id(h))
        {
            self.terms.prefetch_term(id);
            if let Some(run) = self.runs.get(id as usize) {
                self.pool.prefetch(run.docs);
                if self.positions {
                    self.pool.prefetch(run.prox);
                }
            }
        }
    }

    /// `TermsHashPerField.add`'s lookup: `term`'s id, added to the
    /// dictionary if new, and its frequency in `doc` so far.
    #[inline]
    pub(crate) fn occurrence(&mut self, doc: i32, term: &[u8]) -> TermSlot {
        self.occurrence_hashed(doc, term, hash_bytes(term))
    }

    /// [`Self::occurrence`] for a term whose [`hash_bytes`] is known.
    #[inline]
    fn occurrence_hashed(&mut self, doc: i32, term: &[u8], h: u64) -> TermSlot {
        let (id, new_to_segment) = self.terms.add_hashed(term, h);
        if new_to_segment {
            // `newTerm` starts both streams.
            let docs = self.pool.new_slice();
            let prox = if self.positions {
                self.pool.new_slice()
            } else {
                NONE
            };
            self.runs.push(TermRun {
                docs,
                prox,
                ..TermRun::NEW
            });
            self.starts.push([docs, prox]);
            if self.offsets {
                self.last_offsets.push(0);
            }
        }
        let run = &self.runs[id as usize];
        TermSlot {
            id,
            new_to_segment,
            freq: (run.last_doc == doc && run.doc_count > 0).then_some(run.freq),
        }
    }

    /// The term's first occurrence in `doc`, at `freq`: `newTerm` or
    /// `addTerm`'s new-document half. The term's previous document goes to
    /// its documents stream now that its frequency is final.
    #[inline]
    pub(crate) fn first_in_doc(&mut self, slot: TermSlot, doc: i32, freq: i32) {
        if self.last_doc != doc {
            self.last_doc = doc;
            self.doc_count = self.doc_count.saturating_add(1);
        }
        let id = slot.id as usize;
        let run = &mut self.runs[id];
        if run.doc_count > 0 {
            Self::write_doc(&mut self.pool, run);
            run.last_doc_code = doc.wrapping_sub(run.last_doc);
        } else {
            run.last_doc_code = doc;
        }
        run.last_doc = doc;
        run.freq = freq;
        run.last_pos = 0;
        run.doc_count = run.doc_count.saturating_add(1);
        self.sum_doc_freq = self.sum_doc_freq.saturating_add(1);
        self.sum_total_term_freq = self.sum_total_term_freq.saturating_add(i64::from(freq));
        if let Some(o) = self.last_offsets.get_mut(id) {
            *o = 0;
        }
    }

    /// Appends the run's newest document to its documents stream: the doc
    /// delta shifted left, its low bit set for a frequency of one, else the
    /// frequency after it.
    #[inline]
    fn write_doc(pool: &mut BytePool, run: &mut TermRun) {
        let code = u64::from(run.last_doc_code as u32) << 1;
        if run.freq == 1 {
            pool.write_vlong(&mut run.docs, code | 1);
        } else {
            pool.write_vlong(&mut run.docs, code);
            pool.write_vlong(&mut run.docs, u64::from(run.freq as u32));
        }
    }

    /// Sets the term's frequency in the current document, past its first
    /// occurrence.
    #[inline]
    pub(crate) fn set_freq(&mut self, slot: TermSlot, freq: i32) {
        let run = &mut self.runs[slot.id as usize];
        if run.doc_count > 0 {
            self.sum_total_term_freq = self
                .sum_total_term_freq
                .saturating_add(i64::from(freq))
                .saturating_sub(i64::from(run.freq));
            run.freq = freq;
        }
    }

    /// One occurrence's position, offsets (kept when the field indexes them)
    /// and payload (empty is none): `writeProx` and `writeOffsets`. Only for
    /// a field that indexes positions.
    #[inline]
    pub(crate) fn prox(
        &mut self,
        slot: TermSlot,
        position: i32,
        offsets: (i32, i32),
        payload: &[u8],
    ) {
        // `writeProx` runs only for a field with positions: only there can a
        // payload switch `storePayloads` on.
        if !self.positions {
            return;
        }
        let id = slot.id as usize;
        let run = &mut self.runs[id];
        let code = u64::from(position.wrapping_sub(run.last_pos) as u32) << 1;
        run.last_pos = position;
        run.occurrences = run.occurrences.saturating_add(1);
        if payload.is_empty() {
            self.pool.write_vlong(&mut run.prox, code);
        } else {
            self.payloads = true;
            self.pool.write_vlong(&mut run.prox, code | 1);
            self.pool.write_vlong(&mut run.prox, payload.len() as u64);
            for &b in payload {
                self.pool.write_byte(&mut run.prox, b);
            }
        }
        if let Some(last) = self.last_offsets.get_mut(id) {
            let (start, end) = offsets;
            self.pool
                .write_vlong(&mut run.prox, u64::from(start.wrapping_sub(*last) as u32));
            self.pool
                .write_vlong(&mut run.prox, u64::from(end.wrapping_sub(start) as u32));
            *last = start;
        }
    }

    /// Reads one occurrence, its payload appended to `payload` (when given)
    /// and its length returned.
    #[inline]
    fn read_occurrence(
        &self,
        r: &mut SliceReader<'_>,
        last: &mut Occurrence,
        payload: Option<&mut Vec<u8>>,
    ) -> u32 {
        let code = r.read_vlong();
        last.position = last.position.wrapping_add((code >> 1) as u32 as i32);
        let mut len = 0u32;
        if code & 1 != 0 {
            len = r.read_vlong() as u32;
            match payload {
                Some(out) => {
                    for _ in 0..len {
                        out.push(r.read_byte());
                    }
                }
                None => {
                    for _ in 0..len {
                        r.read_byte();
                    }
                }
            }
        }
        if self.offsets {
            let start = last.offsets.0.wrapping_add(r.read_i32());
            let end = start.wrapping_add(r.read_i32());
            last.offsets = (start, end);
        }
        len
    }

    /// Term `id`'s documents and occurrences decoded onto `out`'s columns,
    /// in buffered order.
    fn decode(&self, id: u32, out: &mut Columns) -> Result<()> {
        let run = &self.runs[id as usize];
        let [docs_start, prox_start] = self.starts[id as usize];
        let docs_from = out.docs.len();
        let mut r = self.pool.reader(docs_start, run.docs);
        let mut doc = 0i32;
        while !r.eof() {
            let code = r.read_vlong();
            doc = doc.wrapping_add((code >> 1) as u32 as i32);
            let freq = if code & 1 != 0 { 1 } else { r.read_i32() };
            out.docs.push((doc, freq));
        }
        if run.doc_count > 0 {
            out.docs.push((run.last_doc, run.freq));
        }
        if !self.positions {
            return Ok(());
        }
        let pos_from = out.positions.len();
        let mut r = self.pool.reader(prox_start, run.prox);
        let mut seen = 0u32;
        for i in docs_from..out.docs.len() {
            let freq = u32::try_from(out.docs[i].1).unwrap_or(0);
            let mut last = Occurrence::default();
            for _ in 0..freq {
                if r.eof() {
                    return Err(explicit_error(
                        "buffered positions disagree with the term frequencies",
                    ));
                }
                let len = self.read_occurrence(
                    &mut r,
                    &mut last,
                    self.payloads.then_some(&mut out.payload_bytes),
                );
                out.positions.push(last.position);
                if self.offsets {
                    out.offsets.push(last.offsets);
                }
                if self.payloads {
                    out.payload_lengths.push(len);
                }
            }
            seen = seen.saturating_add(freq);
            out.ends
                .push(to_u32(out.positions.len().saturating_sub(pos_from)));
        }
        // A document's positions are as many as its frequency.
        if seen != run.occurrences || !r.eof() {
            return Err(explicit_error(
                "buffered positions disagree with the term frequencies",
            ));
        }
        Ok(())
    }

    /// Term `id`'s documents, positions, offsets and payloads, appended to
    /// `out`'s columns with each document renumbered by `old_to_new` (and
    /// the documents put back in doc order) when given.
    fn lay_out(
        &self,
        id: u32,
        old_to_new: Option<&[usize]>,
        scratch: &mut Scratch,
        out: &mut FlushedField,
    ) -> Result<()> {
        let docs_from = out.cols.docs.len();
        let pos_from = out.cols.positions.len();
        let len_from = out.cols.payload_lengths.len();
        let bytes_from = out.cols.payload_bytes.len();
        match old_to_new {
            None => {
                // `ends` are relative to the term's first position.
                self.decode(id, &mut out.cols)?;
            }
            Some(map) => {
                let t = &mut scratch.cols;
                t.clear();
                scratch.order.clear();
                self.decode(id, t)?;
                for (i, &(doc, _)) in t.docs.iter().enumerate() {
                    let new = usize::try_from(doc)
                        .ok()
                        .and_then(|d| map.get(d))
                        .and_then(|&n| i32::try_from(n).ok())
                        .ok_or_else(|| {
                            explicit_error("a buffered document outside the sort map")
                        })?;
                    scratch.order.push((new, i));
                }
                scratch.order.sort_unstable();
                // Where each document's payload bytes start.
                scratch.payload_at.clear();
                if self.payloads {
                    let mut at = 0usize;
                    let mut from = 0usize;
                    for &end in &t.ends {
                        scratch.payload_at.push(at);
                        at = t.payload_lengths[from..end as usize]
                            .iter()
                            .fold(at, |a, &l| a.saturating_add(l as usize));
                        from = end as usize;
                    }
                }
                let mut written = 0u32;
                for &(new, i) in &scratch.order {
                    out.cols.docs.push((new, t.docs[i].1));
                    if !self.positions {
                        continue;
                    }
                    let from = i.checked_sub(1).map_or(0, |p| t.ends[p]);
                    let to = t.ends[i];
                    let span = from as usize..to as usize;
                    written = written.saturating_add(to.saturating_sub(from));
                    out.cols.ends.push(written);
                    out.cols
                        .positions
                        .extend_from_slice(t.positions.get(span.clone()).unwrap_or_default());
                    if self.offsets {
                        out.cols
                            .offsets
                            .extend_from_slice(t.offsets.get(span.clone()).unwrap_or_default());
                    }
                    if self.payloads {
                        let lengths = t.payload_lengths.get(span).unwrap_or_default();
                        let at = scratch.payload_at.get(i).copied().unwrap_or(0);
                        let end = lengths
                            .iter()
                            .fold(at, |a, &l| a.saturating_add(l as usize));
                        out.cols.payload_lengths.extend_from_slice(lengths);
                        out.cols
                            .payload_bytes
                            .extend_from_slice(t.payload_bytes.get(at..end).unwrap_or_default());
                    }
                }
            }
        }
        out.term_bytes.extend_from_slice(self.terms.term(id));
        out.terms.push(TermSpan {
            term_end: to_u32(out.term_bytes.len()),
            docs: (to_u32(docs_from), to_u32(out.cols.docs.len())),
            positions: (to_u32(pos_from), to_u32(out.cols.positions.len())),
            payload_lengths: (to_u32(len_from), to_u32(out.cols.payload_lengths.len())),
            payload_bytes: (to_u32(bytes_from), to_u32(out.cols.payload_bytes.len())),
        });
        Ok(())
    }

    /// The field laid out for the postings writer: [`PendingPostings::flushed`].
    fn flushed(&self, number: i32, old_to_new: Option<&[usize]>) -> Result<FlushedField> {
        let ids: Vec<u32> = self
            .terms
            .sorted_ids()
            .into_iter()
            .filter(|&id| self.runs[id as usize].doc_count > 0)
            .collect();
        let (docs, occurrences) = ids.iter().fold((0usize, 0usize), |(d, o), &id| {
            let r = &self.runs[id as usize];
            (
                d.saturating_add(r.doc_count as usize),
                o.saturating_add(r.occurrences as usize),
            )
        });
        let term_bytes = ids
            .iter()
            .map(|&id| self.terms.term(id).len())
            .fold(0usize, usize::saturating_add);
        let mut out = FlushedField {
            number,
            doc_count: self.doc_count,
            term_bytes: Vec::with_capacity(term_bytes),
            terms: Vec::with_capacity(ids.len()),
            cols: Columns {
                docs: Vec::with_capacity(docs),
                ends: Vec::with_capacity(if self.positions { docs } else { 0 }),
                positions: Vec::with_capacity(occurrences),
                offsets: Vec::with_capacity(if self.offsets { occurrences } else { 0 }),
                payload_lengths: Vec::with_capacity(if self.payloads { occurrences } else { 0 }),
                payload_bytes: Vec::new(),
            },
        };
        let mut scratch = Scratch::default();
        // Byte order visits the terms' runs and streams at random: start
        // loading a term's run and stream starts a few terms ahead, and its
        // bytes and streams' first bytes once those are in cache.
        for (i, &id) in ids.iter().enumerate() {
            self.prefetch_layout(&ids, i);
            self.lay_out(id, old_to_new, &mut scratch, &mut out)?;
        }
        Ok(out)
    }

    /// The field's terms in byte order, laid out a batch at a time into one
    /// reused [`FlushedField`] and handed to `write` as views: the flush
    /// never holds the whole field decoded.
    fn stream(
        &self,
        old_to_new: Option<&[usize]>,
        write: &mut dyn FnMut(&[TermView<'_>]) -> Result<()>,
    ) -> Result<()> {
        let ids: Vec<u32> = self
            .terms
            .sorted_ids()
            .into_iter()
            .filter(|&id| self.runs[id as usize].doc_count > 0)
            .collect();
        let mut out = FlushedField {
            number: 0,
            doc_count: self.doc_count,
            term_bytes: Vec::new(),
            terms: Vec::new(),
            cols: Columns::default(),
        };
        let mut scratch = Scratch::default();
        for (i, &id) in ids.iter().enumerate() {
            self.prefetch_layout(&ids, i);
            self.lay_out(id, old_to_new, &mut scratch, &mut out)?;
            if out.terms.len() >= STREAM_TERMS
                || out.cols.docs.len().max(out.cols.positions.len()) >= STREAM_ENTRIES
            {
                write(&out.views())?;
                out.clear();
            }
        }
        if !out.terms.is_empty() {
            write(&out.views())?;
        }
        Ok(())
    }

    /// Starts loading what laying out `ids[i]`'s successors reads: see
    /// [`Self::flushed`].
    #[inline]
    fn prefetch_layout(&self, ids: &[u32], i: usize) {
        if let Some(&ahead) = ids.get(i.saturating_add(LAYOUT_PREFETCH)) {
            lucene_util::simd::prefetch(&self.runs, ahead as usize);
            lucene_util::simd::prefetch(&self.starts, ahead as usize);
            self.terms.prefetch_start(ahead);
        }
        if let Some(&near) = ids.get(i.saturating_add(LAYOUT_PREFETCH / 2)) {
            self.terms.prefetch_term(near);
            if let Some(&[docs, prox]) = self.starts.get(near as usize) {
                self.pool.prefetch(docs);
                if self.positions {
                    self.pool.prefetch(prox);
                }
            }
        }
    }

    /// `document`'s occurrences of `id`, as [`InvertedField`] holds them:
    /// what tests read back after inverting a document.
    #[cfg(test)]
    fn doc_term(&self, id: u32, doc: i32) -> Option<super::explicit::InvertedTerm> {
        let mut cols = Columns::default();
        self.decode(id, &mut cols).ok()?;
        let i = cols.docs.iter().position(|&(d, _)| d == doc)?;
        let freq = cols.docs[i].1;
        let (from, to) = if self.positions {
            (
                i.checked_sub(1).map_or(0, |p| cols.ends[p] as usize),
                cols.ends[i] as usize,
            )
        } else {
            (0, 0)
        };
        let payloads = if self.payloads {
            let mut at: usize = cols.payload_lengths[..from]
                .iter()
                .map(|&l| l as usize)
                .sum();
            cols.payload_lengths[from..to]
                .iter()
                .map(|&l| {
                    let end = at.saturating_add(l as usize);
                    let b = cols.payload_bytes[at..end].to_vec();
                    at = end;
                    b
                })
                .collect()
        } else {
            Vec::new()
        };
        Some(super::explicit::InvertedTerm {
            term: self.terms.term(id).to_vec(),
            freq,
            positions: cols.positions[from..to].to_vec(),
            offsets: cols.offsets.get(from..to).unwrap_or_default().to_vec(),
            payloads,
        })
    }

    /// Cuts term `id`, whose pending document is at or past `cut_at`, back
    /// to its documents before `cut_at`: the newest of them becomes the
    /// run's pending document again, taken back off the documents stream.
    fn rollback_term(&mut self, id: usize, cut_at: i32) {
        let run = self.runs[id];
        let [docs_start, prox_start] = self.starts[id];
        // The written documents before `cut_at`, each with where its entry
        // starts and its delta.
        let mut kept: Vec<(SlicePos, i32, i32, i32)> = Vec::new();
        let mut r = self.pool.reader(docs_start, run.docs);
        let mut doc = 0i32;
        while !r.eof() {
            let pos = r.pos();
            let code = r.read_vlong();
            let delta = (code >> 1) as u32 as i32;
            doc = doc.wrapping_add(delta);
            let freq = if code & 1 != 0 { 1 } else { r.read_i32() };
            if doc >= cut_at {
                break;
            }
            kept.push((pos, doc, delta, freq));
        }
        let occurrences: u32 = kept
            .iter()
            .map(|k| u32::try_from(k.3).unwrap_or(0))
            .fold(0, u32::saturating_add);
        let mut new = TermRun {
            docs: docs_start,
            prox: prox_start,
            ..TermRun::NEW
        };
        let docs_cut = match kept.last() {
            Some(&(pos, d, delta, freq)) => {
                new.doc_count = to_u32(kept.len());
                new.last_doc = d;
                new.last_doc_code = delta;
                new.freq = freq;
                pos
            }
            None => self.pool.reader(docs_start, docs_start).pos(),
        };
        new.docs = self.pool.cut(docs_cut);
        if self.positions {
            let mut r = self.pool.reader(prox_start, run.prox);
            let mut last = Occurrence::default();
            for _ in 0..occurrences {
                if r.eof() {
                    break;
                }
                self.read_occurrence(&mut r, &mut last, None);
            }
            let pos = r.pos();
            new.prox = self.pool.cut(pos);
            new.occurrences = occurrences;
        }
        self.runs[id] = new;
    }
}

/// Reused per term by [`FieldRuns::lay_out`] under an index sort.
#[derive(Default)]
struct Scratch {
    cols: Columns,
    /// `(new doc, index into the term's documents)`, sorted into the new
    /// order.
    order: Vec<(i32, usize)>,
    /// Per document: where its payload bytes start.
    payload_at: Vec<usize>,
}

/// Where the buffered postings stood before a document or block: see
/// [`PendingPostings::rollback`].
#[derive(Debug)]
pub(crate) struct Checkpoint {
    doc: i32,
    fields: Vec<(i32, FieldMark)>,
}

/// One field's counts at a [`Checkpoint`].
#[derive(Debug, Clone, Copy)]
struct FieldMark {
    terms: usize,
    doc_count: i32,
    last_doc: i32,
    payloads: bool,
    /// `sumDocFreq` and `sumTotalTermFreq`.
    sums: [i64; 2],
}

/// Every field's buffered postings: see the module doc.
#[derive(Debug, Default)]
pub(crate) struct PendingPostings {
    fields: BTreeMap<i32, FieldRuns>,
}

impl PendingPostings {
    /// Field `number`'s dictionary, created with the registered field's
    /// `index_options` on its first term.
    pub(crate) fn field(&mut self, number: i32, index_options: IndexOptions) -> &mut FieldRuns {
        self.fields
            .entry(number)
            .or_insert_with(|| FieldRuns::new(index_options))
    }

    /// Inverts an explicit document's `field` for document `doc`, the
    /// buffer's newest. `index_options` is the registered field's.
    pub(crate) fn add_field(
        &mut self,
        doc: i32,
        field: &InvertedField,
        index_options: IndexOptions,
    ) -> Result<()> {
        if field.terms.is_empty() {
            return Ok(());
        }
        let f = self.field(field.field_number, index_options);
        // Every term's hash first, starting each one's table slot loading:
        // the table outgrows the caches, and lookups waiting on their slots
        // were most of the cost of buffering a document. Then, a few terms
        // ahead of the one being added, the likely term's bytes and run, and
        // a little later the pool bytes its run appends to.
        let hashes = field.terms.hashes();
        for &h in hashes {
            f.terms.prefetch_slot(h);
        }
        for (i, t) in field.terms.iter().enumerate() {
            f.prefetch_ahead(hashes, i);
            // every term once: `validate_explicit` refused a repeat
            let slot = f.occurrence_hashed(doc, t.term, t.hash);
            f.first_in_doc(slot, doc, t.freq);
            // the postings writer reads positions and offsets only for a
            // field that indexes positions
            if f.positions {
                for (i, &position) in t.positions.iter().enumerate() {
                    let offsets = t.offsets.get(i).copied().unwrap_or_default();
                    f.prox(slot, position, offsets, t.payloads.get(i));
                }
            }
        }
        Ok(())
    }

    /// Marks where the postings stand before document `doc` (a block's
    /// first) is inverted.
    pub(crate) fn checkpoint(&self, doc: i32) -> Checkpoint {
        Checkpoint {
            doc,
            fields: self
                .fields
                .iter()
                .map(|(&n, f)| {
                    let mark = FieldMark {
                        terms: f.runs.len(),
                        doc_count: f.doc_count,
                        last_doc: f.last_doc,
                        payloads: f.payloads,
                        sums: [f.sum_doc_freq, f.sum_total_term_freq],
                    };
                    (n, mark)
                })
                .collect(),
        }
    }

    /// Undoes everything since `checkpoint`: fields it did not know are
    /// dropped, terms it did not know leave the dictionary, and every run
    /// loses its documents from the checkpoint's on. It walks every buffered
    /// term, where Java marks the failed document deleted in O(1): a stream
    /// of refused documents against a large buffer pays per refusal. The
    /// pool keeps the slices the undone documents took.
    pub(crate) fn rollback(&mut self, checkpoint: Checkpoint) {
        let known: BTreeMap<i32, FieldMark> = checkpoint.fields.into_iter().collect();
        let cut_at = checkpoint.doc;
        self.fields.retain(|n, f| {
            let Some(&mark) = known.get(n) else {
                return false;
            };
            if f.runs.len() > mark.terms {
                f.runs.truncate(mark.terms);
                f.starts.truncate(mark.terms);
                f.last_offsets.truncate(mark.terms);
                f.terms.truncate(mark.terms);
            }
            for id in 0..f.runs.len() {
                if f.runs[id].last_doc >= cut_at {
                    f.rollback_term(id, cut_at);
                }
            }
            (f.doc_count, f.last_doc, f.payloads) = (mark.doc_count, mark.last_doc, mark.payloads);
            [f.sum_doc_freq, f.sum_total_term_freq] = mark.sums;
            true
        });
    }

    /// The fields that saw a non-empty payload: `storePayloads`.
    #[cfg(test)]
    fn payload_fields(&self) -> impl Iterator<Item = i32> + '_ {
        self.fields
            .iter()
            .filter(|(_, f)| f.payloads)
            .map(|(&n, _)| n)
    }

    /// The fields a flush writes postings for, field ascending.
    pub(crate) fn fields_to_flush(&self) -> Vec<FieldToFlush> {
        self.fields
            .iter()
            .filter(|(_, f)| f.doc_count > 0)
            .map(|(&number, f)| FieldToFlush {
                number,
                doc_count: f.doc_count,
                payloads: f.payloads,
            })
            .collect()
    }

    /// Field `number`'s postings handed to `write` in term order, a batch
    /// at a time, renumbered by `old_to_new` as [`Self::flushed`] does.
    pub(crate) fn stream_field(
        &self,
        number: i32,
        old_to_new: Option<&[usize]>,
        write: &mut dyn FnMut(&[TermView<'_>]) -> Result<()>,
    ) -> Result<()> {
        match self.fields.get(&number) {
            Some(f) if f.doc_count > 0 => f.stream(old_to_new, write),
            _ => Ok(()),
        }
    }

    /// Every field's postings laid out for the postings writer, documents
    /// renumbered by `old_to_new` (buffered doc id to sorted) under an index
    /// sort. The buffer is left as it was.
    pub(crate) fn flushed(&self, old_to_new: Option<&[usize]>) -> Result<FlushedPostings> {
        let fields = self
            .fields
            .iter()
            .filter(|(_, f)| f.doc_count > 0)
            .map(|(&n, f)| f.flushed(n, old_to_new))
            .collect::<Result<_>>()?;
        Ok(FlushedPostings { fields })
    }

    /// What the buffered postings count toward the RAM buffer: the heap
    /// they hold.
    pub(crate) fn bytes(&self) -> usize {
        self.fields
            .values()
            .map(FieldRuns::ram_bytes)
            .fold(0, usize::saturating_add)
    }

    pub(crate) fn clear(&mut self) {
        self.fields.clear();
    }

    /// Field `number`'s terms of document `doc`, in no particular order.
    #[cfg(test)]
    pub(crate) fn doc_terms(&self, number: i32, doc: i32) -> Vec<super::explicit::InvertedTerm> {
        self.fields.get(&number).map_or_else(Vec::new, |f| {
            (0..f.runs.len() as u32)
                .filter_map(|id| f.doc_term(id, doc))
                .collect()
        })
    }
}

/// A field with buffered postings: [`PendingPostings::fields_to_flush`].
#[derive(Debug, Clone, Copy)]
pub(crate) struct FieldToFlush {
    pub(crate) number: i32,
    /// `docsSeen`.
    pub(crate) doc_count: i32,
    /// `storePayloads`.
    pub(crate) payloads: bool,
}

/// One term's place in a [`FlushedField`]'s columns.
#[derive(Debug, Clone, Copy)]
struct TermSpan {
    term_end: u32,
    /// Its documents, and their `ends`.
    docs: (u32, u32),
    /// Its positions, and their offsets when the field has them.
    positions: (u32, u32),
    payload_lengths: (u32, u32),
    payload_bytes: (u32, u32),
}

/// One field's postings as the writer reads them: every term's documents,
/// positions, offsets and payloads in contiguous columns, terms in byte
/// order.
#[derive(Debug)]
struct FlushedField {
    number: i32,
    doc_count: i32,
    term_bytes: Vec<u8>,
    terms: Vec<TermSpan>,
    cols: Columns,
}

impl FlushedField {
    /// Every term's postings as the writer reads them, in place.
    fn views(&self) -> Vec<TermView<'_>> {
        let mut term_start = 0usize;
        self.terms
            .iter()
            .map(|s| {
                let range = |(a, b): (u32, u32)| a as usize..b as usize;
                let term = &self.term_bytes[term_start..s.term_end as usize];
                term_start = s.term_end as usize;
                let docs = range(s.docs);
                let positions = range(s.positions);
                let c = &self.cols;
                TermView {
                    term,
                    docs: &c.docs[docs.clone()],
                    ends: c.ends.get(docs).unwrap_or_default(),
                    positions: &c.positions[positions.clone()],
                    offsets: c.offsets.get(positions).unwrap_or_default(),
                    payload_lengths: &c.payload_lengths[range(s.payload_lengths)],
                    payload_bytes: &c.payload_bytes[range(s.payload_bytes)],
                }
            })
            .collect()
    }

    /// Empties the field, keeping its capacity.
    fn clear(&mut self) {
        self.term_bytes.clear();
        self.terms.clear();
        self.cols.clear();
    }
}

/// The buffered postings laid out for a flush: [`PendingPostings::flushed`].
#[derive(Debug)]
pub(crate) struct FlushedPostings {
    fields: Vec<FlushedField>,
}

impl FlushedPostings {
    /// Every field's postings as the writer takes them, `(field, docCount,
    /// terms in byte order)`, field ascending.
    pub(crate) fn term_lists(&self) -> Vec<(i32, i32, Vec<TermView<'_>>)> {
        self.fields
            .iter()
            .map(|f| (f.number, f.doc_count, f.views()))
            .collect()
    }
}

/// One term's flushed postings, read in place by the postings writer.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TermView<'a> {
    term: &'a [u8],
    docs: &'a [(i32, i32)],
    /// Parallel to `docs` for a field with positions, else empty.
    ends: &'a [u32],
    positions: &'a [i32],
    offsets: &'a [(i32, i32)],
    payload_lengths: &'a [u32],
    payload_bytes: &'a [u8],
}

impl TermView<'_> {
    /// Document `i`'s span `[start, end)` in the positions.
    fn span(&self, i: usize) -> std::ops::Range<usize> {
        let start = i
            .checked_sub(1)
            .and_then(|p| self.ends.get(p))
            .map_or(0, |&e| e as usize);
        start..self.ends.get(i).map_or(start, |&e| e as usize)
    }
}

impl TermSource for TermView<'_> {
    fn term(&self) -> &[u8] {
        self.term
    }
    fn docs(&self) -> &[(i32, i32)] {
        self.docs
    }
    fn position_lists(&self) -> usize {
        self.ends.len()
    }
    fn positions(&self, doc: usize) -> &[i32] {
        self.positions.get(self.span(doc)).unwrap_or_default()
    }
    fn offset_lists(&self) -> usize {
        if self.offsets.is_empty() {
            0
        } else {
            self.ends.len()
        }
    }
    fn offsets(&self, doc: usize) -> &[(i32, i32)] {
        self.offsets.get(self.span(doc)).unwrap_or_default()
    }
    fn payload_bytes(&self) -> &[u8] {
        self.payload_bytes
    }
    fn payload_lengths(&self) -> &[u32] {
        self.payload_lengths
    }
}

#[cfg(test)]
mod tests {
    use super::super::explicit::InvertedTerm;
    use super::*;

    const FULL: IndexOptions = IndexOptions::DocsAndFreqsAndPositionsAndOffsets;

    /// A term at `positions`, each at offsets `(2p, 2p + 1)`.
    fn term(t: &str, positions: &[i32], payloads: &[&[u8]]) -> InvertedTerm {
        InvertedTerm {
            term: t.as_bytes().to_vec(),
            freq: positions.len().max(1) as i32,
            positions: positions.to_vec(),
            offsets: positions
                .iter()
                .map(|&p| (p.saturating_mul(2), p.saturating_mul(2).saturating_add(1)))
                .collect(),
            payloads: payloads.iter().map(|p| p.to_vec()).collect(),
        }
    }

    fn add(
        p: &mut PendingPostings,
        field: i32,
        doc: i32,
        opts: IndexOptions,
        terms: Vec<InvertedTerm>,
    ) {
        let field = InvertedField {
            field_number: field,
            terms: terms.into(),
            norm: None,
        };
        p.add_field(doc, &field, opts).unwrap();
    }

    fn get(p: &PendingPostings, field: i32, doc: i32, t: &str) -> Option<InvertedTerm> {
        p.doc_terms(field, doc)
            .into_iter()
            .find(|x| x.term == t.as_bytes())
    }

    fn view<'a>(lists: &[(i32, i32, Vec<TermView<'a>>)], field: i32, t: &str) -> TermView<'a> {
        let (_, _, terms) = lists.iter().find(|(n, _, _)| *n == field).unwrap();
        *terms.iter().find(|v| v.term() == t.as_bytes()).unwrap()
    }

    /// Streams interleave in one pool across many slice levels and blocks,
    /// read back in order, and a cut at any length resumes appending from
    /// there.
    #[test]
    fn pool_streams_chain_slices_and_cut_anywhere() {
        let mut pool = BytePool::default();
        let (sa, sb) = (pool.new_slice(), pool.new_slice());
        let (mut a, mut b) = (sa, sb);
        let value = |i: u64| i.wrapping_mul(0x9e37_79b9) % (1 << (i % 40));
        for i in 0..20_000u64 {
            pool.write_vlong(&mut a, i);
            pool.write_vlong(&mut b, value(i));
        }
        assert!(pool.blocks.len() > 2);
        let mut r = pool.reader(sa, a);
        assert!((0..20_000u64).all(|i| r.read_vlong() == i));
        assert!(r.eof());
        let mut r = pool.reader(sb, b);
        assert!((0..20_000u64).all(|i| r.read_vlong() == value(i)));
        assert!(r.eof());
        // longest first: a cut overwrites what followed it
        for keep in [20_000u64, 7_777, 300, 129, 128, 127, 5, 4, 3, 1, 0] {
            let mut r = pool.reader(sa, a);
            for _ in 0..keep {
                r.read_vlong();
            }
            let pos = r.pos();
            let mut c = pool.cut(pos);
            for v in [9_999u64, 1, 300] {
                pool.write_vlong(&mut c, v);
            }
            let mut r = pool.reader(sa, c);
            assert!((0..keep).all(|i| r.read_vlong() == i), "{keep}");
            assert_eq!(
                [r.read_vlong(), r.read_vlong(), r.read_vlong()],
                [9_999, 1, 300]
            );
            assert!(r.eof(), "{keep}");
            a = c;
        }
        let empty = pool.new_slice();
        assert!(pool.reader(empty, empty).eof());
        assert!(pool.reader(NONE, NONE).eof());
    }

    /// Every address maps into the block that holds it, small blocks first.
    #[test]
    fn addresses_locate_their_blocks() {
        let mut at = 0usize;
        for b in 0..12 {
            assert_eq!(block_start(b), at);
            assert_eq!(locate(at as u32), (b, 0));
            let end = at + block_size(b) - 1;
            assert_eq!(locate(end as u32), (b, block_size(b) - 1));
            at += block_size(b);
        }
        let mut p = PendingPostings::default();
        add(&mut p, 0, 0, IndexOptions::Docs, vec![term("a", &[], &[])]);
        assert!(
            p.bytes() < 16 * 1024,
            "a one-term field holds {}",
            p.bytes()
        );
    }

    /// A term's documents and occurrences cost bytes, not structs: a frequent
    /// term's document is a byte or two, its positions a byte each.
    #[test]
    fn buffered_postings_cost_bytes_per_occurrence() {
        let mut p = PendingPostings::default();
        let opts = IndexOptions::DocsAndFreqsAndPositions;
        for doc in 0..200_000 {
            add(
                &mut p,
                0,
                doc,
                opts,
                vec![term("a", &[0, 2, 3], &[]), term("b", &[1], &[])],
            );
        }
        let per_doc = p.bytes() / 200_000;
        assert!(per_doc <= 8, "{per_doc} bytes per document");
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let a = view(&lists, 0, "a");
        assert_eq!(a.docs().len(), 200_000);
        assert_eq!(a.docs()[199_999], (199_999, 3));
        assert_eq!(a.positions(123_456), [0, 2, 3]);
        assert_eq!(view(&lists, 0, "b").positions(7), [1]);
    }

    /// A rollback across slice boundaries leaves every stream ready to
    /// take the next document.
    #[test]
    fn rollback_past_many_documents_keeps_appending() {
        let mut p = PendingPostings::default();
        for doc in 0..50 {
            add(&mut p, 0, doc, FULL, vec![term("a", &[doc, doc + 1], &[])]);
        }
        let cp = p.checkpoint(50);
        for doc in 50..80 {
            add(&mut p, 0, doc, FULL, vec![term("a", &[1, 2, 3], &[b"x"])]);
        }
        p.rollback(cp);
        add(&mut p, 0, 50, FULL, vec![term("a", &[7], &[])]);
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let a = view(&lists, 0, "a");
        assert_eq!(a.docs().len(), 51);
        assert_eq!(a.docs()[49], (49, 2));
        assert_eq!(a.docs()[50], (50, 1));
        assert_eq!(a.positions(49), [49, 50]);
        assert_eq!(a.positions(50), [7]);
        assert_eq!(a.offsets(50), [(14, 15)]);
        assert!(a.payload_lengths().is_empty());
    }

    /// A rollback takes back a document's new terms, its occurrences of old
    /// ones, and the payloads it was the first to bring -- the zero lengths
    /// they backfilled into earlier documents included.
    #[test]
    fn rollback_undoes_a_document_down_to_its_payloads() {
        let mut p = PendingPostings::default();
        add(
            &mut p,
            0,
            0,
            FULL,
            vec![term("a", &[0, 1], &[]), term("b", &[2], &[])],
        );
        let (cp, bytes) = (p.checkpoint(1), p.bytes());
        add(
            &mut p,
            0,
            1,
            FULL,
            vec![term("a", &[0], &[b"xy"]), term("c", &[1], &[])],
        );
        // the first payload backfills a zero length for every occurrence
        assert_eq!(get(&p, 0, 0, "a").unwrap().payloads, [b"", b""]);
        assert_eq!(get(&p, 0, 1, "a").unwrap().payloads, [b"xy"]);
        assert_eq!(p.payload_fields().collect::<Vec<_>>(), [0]);
        p.rollback(cp);
        assert!(p.doc_terms(0, 1).is_empty());
        let a = get(&p, 0, 0, "a").unwrap();
        assert_eq!((a.positions, a.payloads), (vec![0, 1], vec![]));
        assert_eq!(p.payload_fields().count(), 0);
        // the pool keeps what the undone document took
        assert!(p.bytes() >= bytes);
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        assert_eq!(lists.len(), 1);
        assert_eq!(lists[0].1, 1, "docCount");
        let terms: Vec<&[u8]> = lists[0].2.iter().map(TermSource::term).collect();
        assert_eq!(terms, [b"a", b"b"]);
        assert!(view(&lists, 0, "a").payload_bytes().is_empty());
    }

    /// Payloads the checkpoint already had survive, cut to its documents.
    #[test]
    fn rollback_keeps_the_payloads_before_the_checkpoint() {
        let mut p = PendingPostings::default();
        add(&mut p, 0, 0, FULL, vec![term("a", &[0], &[b"p"])]);
        let cp = p.checkpoint(1);
        add(&mut p, 0, 1, FULL, vec![term("a", &[0, 1], &[b"qq", b""])]);
        p.rollback(cp);
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let a = view(&lists, 0, "a");
        assert_eq!(
            (a.payload_bytes(), a.payload_lengths()),
            (&b"p"[..], &[1u32][..])
        );
        assert_eq!(a.docs(), [(0, 1)]);
        // a field the checkpoint never saw goes altogether
        let cp = p.checkpoint(1);
        add(&mut p, 5, 1, IndexOptions::Docs, vec![term("k", &[], &[])]);
        p.rollback(cp);
        assert!(p.doc_terms(5, 1).is_empty());
        assert_eq!(p.flushed(None).unwrap().term_lists().len(), 1);
    }

    /// An index sort's remap moves each document's positions, offsets and
    /// payloads with it.
    #[test]
    fn remap_carries_each_documents_occurrences_into_sort_order() {
        let mut p = PendingPostings::default();
        add(&mut p, 0, 0, FULL, vec![term("a", &[0], &[b"a0"])]);
        add(&mut p, 1, 0, IndexOptions::Docs, vec![term("k", &[], &[])]);
        add(&mut p, 0, 1, FULL, vec![term("a", &[0, 3], &[b"", b"b"])]);
        add(&mut p, 0, 2, FULL, vec![term("a", &[5], &[b"cc"])]);
        add(&mut p, 1, 2, IndexOptions::Docs, vec![term("k", &[], &[])]);
        let remapped = p.flushed(Some(&[2, 0, 1])).unwrap();
        let lists = remapped.term_lists();
        let a = view(&lists, 0, "a");
        assert_eq!(a.docs(), [(0, 2), (1, 1), (2, 1)]);
        assert_eq!(a.position_lists(), 3);
        assert_eq!(a.positions(0), [0, 3]);
        assert_eq!(a.positions(1), [5]);
        assert_eq!(a.positions(2), [0]);
        assert_eq!(a.offset_lists(), 3);
        assert_eq!(a.offsets(0), [(0, 1), (6, 7)]);
        assert_eq!(a.offsets(2), [(0, 1)]);
        assert_eq!(a.payload_lengths(), [0, 1, 2, 2]);
        assert_eq!(a.payload_bytes(), b"bcca0");
        // a docs-only field has no occurrences to move
        let k = view(&lists, 1, "k");
        assert_eq!(k.docs(), [(1, 1), (2, 1)]);
        assert_eq!((k.position_lists(), k.offset_lists()), (0, 0));
        assert!(k.positions(0).is_empty() && k.offsets(0).is_empty());
        assert!(k.payload_bytes().is_empty() && k.payload_lengths().is_empty());
        // the buffered order is untouched
        let flushed = p.flushed(None).unwrap();
        assert_eq!(view(&flushed.term_lists(), 0, "a").docs()[0], (0, 1));
        let e = p.flushed(Some(&[0])).unwrap_err().to_string();
        assert!(e.contains("outside the sort map"), "{e}");
    }

    /// The document API's side of a run: frequencies counted per document,
    /// occurrences kept only where the field indexes them.
    #[test]
    fn occurrences_are_kept_as_the_field_indexes_them() {
        let mut p = PendingPostings::default();
        add(&mut p, 0, 0, FULL, Vec::new());
        assert!(
            p.flushed(None).unwrap().term_lists().is_empty(),
            "an empty field adds nothing"
        );
        let f = p.field(2, IndexOptions::DocsAndFreqsAndPositions);
        let slot = f.occurrence(0, b"x");
        assert!(slot.new_to_segment && slot.freq.is_none());
        f.first_in_doc(slot, 0, 1);
        f.prox(slot, 4, (9, 9), b"");
        let slot = f.occurrence(0, b"x");
        assert_eq!((slot.new_to_segment, slot.freq), (false, Some(1)));
        f.set_freq(slot, 2);
        f.prox(slot, 6, (9, 9), b"");
        let docs_only = p.field(3, IndexOptions::Docs);
        let slot = docs_only.occurrence(0, b"y");
        docs_only.first_in_doc(slot, 0, 1);
        docs_only.prox(slot, 1, (0, 1), b"ignored");
        let flushed = p.flushed(None).unwrap();
        let lists = flushed.term_lists();
        let x = view(&lists, 2, "x");
        assert_eq!((x.docs(), x.positions(0)), (&[(0, 2)][..], &[4, 6][..]));
        assert_eq!(x.offset_lists(), 0, "no offsets indexed");
        assert!(x.offsets(0).is_empty());
        assert!(get(&p, 3, 0, "y").unwrap().positions.is_empty());
        assert_eq!(
            p.payload_fields().count(),
            0,
            "a docs-only field stores no payloads"
        );
        assert!(p.bytes() > 0);
        p.clear();
        assert_eq!(p.bytes(), 0);
    }
}
