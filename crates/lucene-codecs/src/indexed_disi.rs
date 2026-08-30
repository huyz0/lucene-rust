//! Port of `org.apache.lucene.codecs.lucene90.IndexedDISI` — the sparse
//! doc-id-set encoding shared by sparse norms and sparse doc values.
//!
//! Two read shapes, because two access patterns exist:
//!
//! - [`DisiCursor`] is the real port of Java's `IndexedDISI`: a forward-only
//!   `DocIdSetIterator` with per-block incremental state and the DENSE rank
//!   table's jump, holding **O(1) memory over the on-disk bytes** no matter how
//!   many documents the field covers. This is what every doc-values and norms
//!   lookup goes through.
//! - [`decode_doc_ids`] materialises the whole structure into a sorted
//!   `Vec<i32>`, which costs 4 bytes per present document. It survives for the
//!   callers that genuinely want an owned doc-id list (`lucene_search`'s
//!   soft-deletes reader builds a set from it) and for testing the cursor
//!   against a second implementation.
//!
//! The block **jump table** is still not read; see [`DisiCursor`] for why.
//!
//! Wire format (three block kinds, chosen per 65536-doc range by how many
//! docs in that range have a value; only non-empty ranges are written, and a
//! final synthetic block containing just the doc id `i32::MAX` — Lucene's
//! `NO_MORE_DOCS` sentinel — always terminates the structure):
//! ```text
//! per block:
//!   BlockIndex  --> u16          (which 65536-doc range this is)
//!   NumValues   --> 1 + u16      (how many docs in this block have a value)
//!   if NumValues <= 4095:                                    SPARSE
//!     DocLow16   --> u16 * NumValues   (low 16 bits of each doc id, ascending)
//!   elif NumValues == 65536:                                 ALL
//!     (no data: every doc in the range has a value)
//!   else:                                                    DENSE
//!     RankTable  --> u8 * rankBytes(denseRankPower)  (present iff denseRankPower != 0xFF)
//!     Bits       --> i64 * 1024        (a 65536-bit dense bitset, LE words)
//! ```
//! Trailing the last block: an optional jump table (int pairs) that this
//! decoder never reads, because sequential decoding naturally stops at the
//! `NO_MORE_DOCS` sentinel block, right before the jump table begins.

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::DataOutput;

const MAX_ARRAY_LENGTH: u32 = (1 << 12) - 1; // 4095
const BLOCK_SIZE: u32 = 65536;
const DENSE_BLOCK_LONGS: u32 = BLOCK_SIZE / 64; // 1024

/// Java's `denseRankPower == -1` as it is stored in `.dvm`/`.nvm` metadata:
/// "DENSE blocks in this region carry no rank table". A `byte` in Java, so
/// `-1` is `0xFF` here.
pub const NO_RANK: u8 = 0xFF;

/// Java's `IndexedDISI.DEFAULT_DENSE_RANK_POWER`: one rank entry per 512 doc
/// ids (8 words). What real Lucene writes for every DENSE block.
pub const DEFAULT_DENSE_RANK_POWER: u8 = 9;

/// `DocIdSetIterator.NO_MORE_DOCS` — the sentinel doc id the terminating block
/// carries, never a real document.
pub const NO_MORE_DOCS: i32 = i32::MAX;

pub type Result<T> = std::result::Result<T, lucene_store::Error>;

/// Little-endian `u16` at `pos`, bounds-checked. Used only for block headers:
/// inside a block the cursor indexes a payload slice it has already
/// range-checked, so the scans carry no error path at all.
// ARITH: `pos` is a block-header offset the caller took from `block_end`, or
// that plus 2. `block_end` is *not* always at or under `data.len()`:
// `read_block_header` assigns it before the `slice(..)?` that would reject it,
// so a cursor left in an error state keeps an out-of-range one. It is still
// bounded -- the assignment is `payload + num_values * 2` or
// `bitmap_start + 8192`, with `payload <= data.len() + 4` (this function
// having just accepted `pos + 2`) and `num_values <= 65536`, so
// `block_end <= data.len() + 4 + 131072`. A slice length is at most
// `isize::MAX`, which leaves the `+ 2` here nowhere near `usize::MAX`.
#[allow(clippy::arithmetic_side_effects)]
#[inline]
fn read_u16_at(data: &[u8], pos: usize) -> Result<u16> {
    data.get(pos..pos + 2)
        .map(|b| u16::from_le_bytes([b[0], b[1]]))
        .ok_or(lucene_store::Error::Eof { offset: pos })
}

/// Size in bytes of a DENSE block's rank table for `dense_rank_power`, or `0`
/// when no rank table is present.
///
/// Port of the `denseRankPower` validation both `IndexedDISI`'s constructor and
/// `IndexedDISI.writeBitSet` perform: the only legal values are `-1` (stored as
/// the byte `0xFF`, meaning "no rank table") and `7..=15`. Java throws
/// `IllegalArgumentException`; this returns a `Corrupted` error, because the
/// byte comes straight off disk (`.dvm`/`.nvm` metadata) and a decoder must not
/// trust it. Without the check, `dense_rank_power < 7` underflows the
/// `dense_rank_power - 7` shift below -- a debug-build panic, and a silently
/// wrong skip distance in release.
fn dense_rank_bytes(dense_rank_power: u8) -> Result<usize> {
    if dense_rank_power == NO_RANK {
        return Ok(0);
    }
    if !(7..=15).contains(&dense_rank_power) {
        return Err(lucene_store::Error::Corrupted(format!(
            "IndexedDISI denseRankPower must be 7-15 or 0xFF (none), got {dense_rank_power}"
        )));
    }
    // ARITH: the range check above establishes `7 <= dense_rank_power <= 15`,
    // so the subtraction cannot underflow and the shift is in `0..=8`.
    #[allow(clippy::arithmetic_side_effects)]
    let bytes = (DENSE_BLOCK_LONGS >> (dense_rank_power - 7)) as usize;
    Ok(bytes)
}

/// Decodes every doc id that has a value, in ascending order. `data` must
/// start exactly at the first block header (the same `offset` a `NormsEntry`
/// or doc-values entry records); `dense_rank_power` comes from that same
/// entry and only matters for correctly skipping DENSE blocks' rank bytes.
// ARITH: `read_u16` yields a `u16`, so `1u32 + it` is at most 65 536.
// `block` is a `u16` widened to `i64`, so `block << 16` is at most 2^32 and
// every `base + ...` below adds at most 65 535 to it -- all far inside `i64`.
// `word_idx` runs over `0..1024` and `bit` over `0..64`.
#[allow(clippy::arithmetic_side_effects)]
pub fn decode_doc_ids(data: &[u8], dense_rank_power: u8) -> Result<Vec<i32>> {
    let mut input = SliceInput::new(data);
    let mut docs = Vec::new();

    loop {
        let block = input.read_u16()? as i64;
        let num_values = 1u32 + input.read_u16()? as u32;

        if num_values <= MAX_ARRAY_LENGTH {
            // SPARSE: `num_values` explicit low-16-bit doc ids.
            let mut reached_sentinel = false;
            for _ in 0..num_values {
                let low = input.read_u16()? as i64;
                let doc = (block << 16) | low;
                if doc == i32::MAX as i64 {
                    reached_sentinel = true;
                    break;
                }
                docs.push(doc as i32);
            }
            if reached_sentinel {
                break;
            }
        } else if num_values == BLOCK_SIZE {
            // ALL: every doc in this 65536-range has a value; no bytes stored.
            let base = block << 16;
            docs.extend((0..BLOCK_SIZE as i64).map(|i| (base + i) as i32));
        } else {
            // DENSE: a 65536-bit array, optionally preceded by rank bytes we skip.
            input.skip(dense_rank_bytes(dense_rank_power)?)?;
            let base = block << 16;
            for word_idx in 0..DENSE_BLOCK_LONGS as i64 {
                let word = input.read_i64()? as u64;
                if word == 0 {
                    continue;
                }
                for bit in 0..64u32 {
                    if (word >> bit) & 1 != 0 {
                        docs.push((base + word_idx * 64 + bit as i64) as i32);
                    }
                }
            }
        }
    }

    Ok(docs)
}

/// A forward-only cursor over an `IndexedDISI` region -- a port of Lucene's
/// `IndexedDISI` itself, state for state.
///
/// `advance_exact(doc)` answers "does this document have a value, and at what
/// ordinal" in time proportional to the ground covered *since the last call*,
/// not to the size of the field, because every piece of per-block bookkeeping
/// Java carries is carried here too:
///
/// | Java field | here | what it saves |
/// |---|---|---|
/// | `block`, `blockEnd`, `method`, `index`, `nextBlockIndex` | same names | block headers are read once, in order |
/// | `exists`, `nextExistDocInBlock` + the slice pointer | `exists`, `next_exist_doc_in_block`, `sparse_pos` | a SPARSE block's 16-bit doc ids are read once across the whole block, and an overshoot is pushed back rather than rescanned |
/// | `word`, `wordIndex`, `numberOfOnes`, `denseOrigoIndex` | same names | a DENSE block's 1024 words are popcounted once for the whole block, not once per lookup |
/// | `denseRankTable` | `dense_rank_offset`/`dense_rank_len` (read in place, not copied) | `rank_skip` jumps straight to the nearest rank boundary, bounding a cold lookup at one rank read + at most `2^(power-6)` word reads |
/// | `gap` | `gap` | an ALL block is one subtraction |
///
/// The one Java structure still not used is the **block jump table**
/// (`createJumpTable`/`advanceBlock`'s two-blocks-ahead shortcut). Walking
/// block headers is `O(maxDoc / 65536)` -- 16 headers for a million documents,
/// each a 4-byte read -- and this port's own writers record
/// `jumpTableEntryCount = 0`, so there is usually no table to read. Recorded in
/// `docs/sweep/m2/c2-sparse-lookup.md`.
///
/// # Forward-only, and it says so loudly
///
/// `advance_exact` **must** be called with non-decreasing doc ids; a smaller
/// one **panics**. Java's `advanceExact` has the same precondition and enforces
/// it with a bare `assert` (off in production, undefined behaviour when
/// violated). Returning `None` instead -- what this cursor used to do -- is the
/// worst of the three options: "this document has no value" is a legitimate
/// answer, so a caller that violated the contract got a plausible wrong number
/// rather than a diagnosis. This file already panics on a violated writer
/// contract ([`write`]'s ascending check), so a violated reader contract panics
/// too; corrupt *data*, as always, is an `Err`.
///
/// A caller that genuinely needs random access calls [`reset`](Self::reset)
/// first, which rewinds to the start of the region -- Java's equivalent is
/// constructing a new `IndexedDISI`. That is what
/// [`crate::doc_values::NumericReader`] does, and it costs one block-header
/// walk, never an allocation.
#[derive(Debug)]
pub struct DisiCursor<'a> {
    data: &'a [u8],
    dense_rank_power: u8,

    /// Java `block`: the current block's first doc id (`blockIndex << 16`), or
    /// `-1` before the first block header has been read.
    block: i32,
    /// Java `blockEnd`: byte offset just past the current block's payload, and
    /// therefore of the next block's header.
    block_end: usize,
    /// Java `method`: which of the three encodings the current block uses.
    method: Method,
    /// Java `index`: the ordinal `advance_exact` last resolved. Set to
    /// "one before this block's first ordinal" by each block header, exactly
    /// as Java's `index = nextBlockIndex` does.
    index: i64,
    /// Java `nextBlockIndex`: the ordinal one past the current block's last.
    next_block_index: i64,
    /// Java `doc`: the last target passed to `advance_exact`, `-1` before the
    /// first call. Also the forward-only contract's witness.
    doc: i32,

    // --- SPARSE ---
    /// Java `exists`: whether `doc` was present.
    exists: bool,
    /// Java `nextExistDocInBlock`: the last 16-bit doc id read, so a target
    /// behind it is answered without touching the bytes again.
    next_exist_doc_in_block: i32,
    /// The current SPARSE block's payload: `num_values` little-endian 16-bit
    /// doc ids. Sliced (and therefore bounds-checked) once when the block
    /// header is read, so the scan below has no error path at all -- the
    /// eager `Error` construction one `ok_or` per doc id costs measurably more
    /// than the read itself (32% on `indexed_disi/cursor/sparse_forward`).
    sparse_docs: &'a [u8],
    /// Index into `sparse_docs` of the next doc id to read -- Java's slice file
    /// pointer, which it rewinds by two bytes on an overshoot.
    sparse_pos: usize,

    // --- DENSE ---
    /// Byte offset of the current block's rank table (`dense_rank_len == 0`
    /// when there is none). Java copies it into a `byte[]`; there is no reason
    /// to, so this borrows it in place.
    dense_rank: &'a [u8],
    /// The current DENSE block's 1024-word bitmap, sliced once at the header
    /// (Java's `denseBitmapOffset` plus a length it never checks).
    dense_bitmap: &'a [u8],
    /// Java `word`: the last 64-bit word read from the bitmap.
    word: u64,
    /// Java `wordIndex`: which word that was, `-1` at the start of a block.
    word_index: i32,
    /// Java `numberOfOnes`: set bits seen so far in this block, *including*
    /// `word`, offset by the block's first ordinal.
    number_of_ones: i64,
    /// Java `denseOrigoIndex`: this block's first ordinal, needed because rank
    /// values are absolute within the block.
    dense_origo_index: i64,

    // --- ALL ---
    /// Java `gap`: `block - index - 1`, so an ordinal is `doc - gap`.
    gap: i64,
}

/// Which of `IndexedDISI.Method`'s three encodings the current block uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Method {
    Sparse,
    Dense,
    All,
}

impl<'a> DisiCursor<'a> {
    /// `data` must start exactly at the first block header -- the same offset a
    /// `NormsEntry` or doc-values entry records. A trailing block jump table
    /// (which this port's writers never emit) is harmless: decoding stops at
    /// the `NO_MORE_DOCS` sentinel block, which precedes it.
    pub fn new(data: &'a [u8], dense_rank_power: u8) -> Self {
        Self {
            data,
            dense_rank_power,
            block: -1,
            block_end: 0,
            method: Method::Sparse,
            index: -1,
            next_block_index: -1,
            doc: -1,
            exists: false,
            next_exist_doc_in_block: -1,
            sparse_docs: &[],
            sparse_pos: 0,
            dense_rank: &[],
            dense_bitmap: &[],
            word: 0,
            word_index: -1,
            number_of_ones: 0,
            dense_origo_index: 0,
            gap: 0,
        }
    }

    /// Rewinds to the start of the region, so `advance_exact` may go backwards
    /// again. Java has no equivalent because it constructs a new `IndexedDISI`;
    /// this keeps the same borrow and allocates nothing.
    pub fn reset(&mut self) {
        *self = Self::new(self.data, self.dense_rank_power);
    }

    /// Java's `docID()`: the last doc id passed to [`advance_exact`], or `-1`
    /// before the first call. A caller that may go backwards compares against
    /// this and calls [`reset`](Self::reset).
    ///
    /// [`advance_exact`]: Self::advance_exact
    pub fn doc_id(&self) -> i32 {
        self.doc
    }

    /// The ordinal of `doc` among documents that have a value, or `None` when
    /// `doc` has none.
    ///
    /// Port of `IndexedDISI.advanceExact`, including its `block < targetBlock`
    /// / `block == targetBlock` structure.
    ///
    /// # Panics
    ///
    /// If `doc` is negative, or is less than the previous call's -- see the
    /// type's doc comment for why that is a panic and not a `None`.
    pub fn advance_exact(&mut self, doc: i32) -> Result<Option<usize>> {
        assert!(doc >= 0, "doc id must be non-negative, got {doc}");
        assert!(
            doc >= self.doc,
            "IndexedDISI cursors are forward-only (Lucene's `advanceExact` asserts the same): \
             advance_exact({doc}) after advance_exact({}); call reset() first",
            self.doc
        );
        // `NO_MORE_DOCS` is the sentinel block's payload, not a document. Java
        // never advances to it (its callers stop at `maxDoc`); answering
        // "present at ordinal N" here would be a decoded sentinel.
        if doc == NO_MORE_DOCS {
            self.doc = doc;
            return Ok(None);
        }
        let target_block = doc & !0xFFFF;
        if self.block < target_block {
            self.advance_block(target_block)?;
        }
        let found = self.block == target_block && self.advance_exact_within_block(doc);
        self.doc = doc;
        Ok(found.then_some(self.index as usize))
    }

    /// `IndexedDISI.advanceBlock`'s iteration fallback: step block by block
    /// until at or past `target_block`. The jump-table shortcut is not ported
    /// (see the type's doc comment).
    fn advance_block(&mut self, target_block: i32) -> Result<()> {
        loop {
            self.read_block_header()?;
            if self.block >= target_block {
                return Ok(());
            }
        }
    }

    /// Port of `IndexedDISI.readBlockHeader`, reading at `block_end` (Java
    /// seeks there first) and leaving `block_end` past the new block's payload.
    // ARITH: `read_u16_at` yields a `u16`, so `num_values` is at most 65 536.
    // `pos == self.block_end`, which every branch below leaves at or under
    // `self.data.len()` (each is either passed through `slice`, which
    // range-checks it, or is `pos + 4` after `read_u16_at` proved
    // `pos + 4 <= len`), so `pos + 2`, `pos + 4`, `payload + num_values * 2`
    // (at most `payload + 8190`) and `bitmap_start + 8192` all stay inside
    // `usize`. `self.index + 1` on the DENSE path is safe because the
    // `checked_add` below accepted `index + num_values` with `num_values >= 1`,
    // so `index + 1 <= next_block_index <= i64::MAX`. The `<< 16` is flagged
    // for its shift *amount*, which is the
    // constant 16; its value may wrap into `i32`'s sign bit for a block index
    // above 0x7FFF, and that is exactly what Java's `int` shift does too (no
    // real doc id lives in such a block; the sentinel is the only one near
    // there). Rust does not check shift-value overflow, so this is a plain
    // port, not a hazard.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_block_header(&mut self) -> Result<()> {
        let pos = self.block_end;
        let block_index = read_u16_at(self.data, pos)?;
        let num_values = 1u32 + read_u16_at(self.data, pos + 2)? as u32;
        let payload = pos + 4;

        self.block = (block_index as i32) << 16;
        self.index = self.next_block_index;
        // Not `+`: `next_block_index` accumulates 65 536 per block over a
        // block count bounded only by `data.len() / 4`. The bound is
        // astronomically large, but it is the input's, not ours, and this runs
        // once per 65 536-doc block rather than per doc -- so it is checked.
        self.next_block_index = self.index.checked_add(num_values as i64).ok_or_else(|| {
            lucene_store::Error::Corrupted("IndexedDISI block ordinals overflow i64".to_string())
        })?;

        // Each block's whole payload is bounds-checked once, here, and then
        // borrowed. Everything downstream indexes inside a slice whose length
        // it already knows, so no per-read error path survives into the scans.
        if num_values <= MAX_ARRAY_LENGTH {
            self.method = Method::Sparse;
            self.block_end = payload + num_values as usize * 2;
            self.sparse_docs = self.slice(payload, self.block_end)?;
            self.sparse_pos = 0;
            self.next_exist_doc_in_block = -1;
        } else if num_values == BLOCK_SIZE {
            self.method = Method::All;
            self.block_end = payload;
            self.gap = self.block as i64 - self.index - 1;
        } else {
            self.method = Method::Dense;
            let rank_len = dense_rank_bytes(self.dense_rank_power)?;
            let bitmap_start = payload + rank_len;
            self.block_end = bitmap_start + DENSE_BLOCK_LONGS as usize * 8;
            self.dense_rank = self.slice(payload, bitmap_start)?;
            self.dense_bitmap = self.slice(bitmap_start, self.block_end)?;
            self.word = 0;
            self.word_index = -1;
            self.number_of_ones = self.index + 1;
            self.dense_origo_index = self.number_of_ones;
        }
        Ok(())
    }

    /// `data[from..to]`, or a `Corrupted`-class EOF -- the one place a block's
    /// bytes are range-checked.
    fn slice(&self, from: usize, to: usize) -> Result<&'a [u8]> {
        self.data
            .get(from..to)
            .ok_or(lucene_store::Error::Eof { offset: from })
    }

    /// `IndexedDISI.Method.advanceExactWithinBlock`, dispatched on the block's
    /// encoding. Leaves `index` set to `doc`'s ordinal when it returns `true`.
    // ARITH: `gap` is `block - index - 1` for an ALL block, so
    // `target - gap` is `target - block + index + 1`; `target` is inside the
    // block by the caller's `target_block` check, so the result is in
    // `index..index + 65536`.
    #[allow(clippy::arithmetic_side_effects)]
    fn advance_exact_within_block(&mut self, target: i32) -> bool {
        match self.method {
            Method::All => {
                // `ALL.advanceExactWithinBlock`: every doc in the range is
                // present, so the ordinal is a subtraction.
                self.index = target as i64 - self.gap;
                true
            }
            Method::Sparse => self.sparse_advance_exact(target),
            Method::Dense => self.dense_advance_exact(target),
        }
    }

    /// `Method.SPARSE.advanceExactWithinBlock`.
    // ARITH: the loop guard is `sparse_pos + 2 <= sparse_docs.len()`, so
    // `sparse_pos + 1` indexes inside the slice and `sparse_pos += 2` lands at
    // or before its end; the `-= 2` and `-= 1` undo exactly one iteration that
    // just ran. `index` is bounded above by `next_block_index` (the same loop
    // guard) and below by the `index += 1` this pass performed.
    #[allow(clippy::arithmetic_side_effects)]
    fn sparse_advance_exact(&mut self, target: i32) -> bool {
        let target_in_block = target & 0xFFFF;
        if self.next_exist_doc_in_block > target_in_block {
            // Already read past `target` on an earlier call, and it was absent.
            debug_assert!(!self.exists);
            return false;
        }
        if target == self.doc {
            // Same document asked twice: Java caches the answer this way.
            return self.exists;
        }
        while self.index < self.next_block_index && self.sparse_pos + 2 <= self.sparse_docs.len() {
            let d = u16::from_le_bytes([
                self.sparse_docs[self.sparse_pos],
                self.sparse_docs[self.sparse_pos + 1],
            ]) as i32;
            self.sparse_pos += 2;
            self.index += 1;
            if d >= target_in_block {
                self.next_exist_doc_in_block = d;
                if d != target_in_block {
                    // Overshoot: push the doc id back so the next call re-reads
                    // it rather than rescanning the block from its start.
                    self.index -= 1;
                    self.sparse_pos -= 2;
                    break;
                }
                self.exists = true;
                return true;
            }
        }
        self.exists = false;
        false
    }

    /// `Method.DENSE.advanceExactWithinBlock`, rank table included.
    // ARITH: `target_in_block` is masked to `0..=0xFFFF`, so
    // `target_word_index` is in `0..=1023` and `word_index` in `-1..=1023`;
    // `from` and `to` are therefore at most 8192, which is `dense_bitmap`'s
    // exact length (the header sliced it that way). `dense_rank_power - 6` is
    // reached only when `dense_rank` is non-empty, which `dense_rank_bytes`
    // only returns for a power in `7..=15`. `ones` sums at most 1024 words of
    // 64 bits, and `number_of_ones` is bounded by the block's cardinality.
    #[allow(clippy::arithmetic_side_effects)]
    fn dense_advance_exact(&mut self, target: i32) -> bool {
        let target_in_block = target & 0xFFFF;
        let target_word_index = target_in_block >> 6;
        // Java: only worth a rank lookup when the jump is at least as far as
        // one rank entry covers, since the rank only lands on a boundary.
        if !self.dense_rank.is_empty()
            && target_word_index - self.word_index >= (1i32 << (self.dense_rank_power - 6))
        {
            self.rank_skip(target_in_block);
        }
        // Java reads these words one at a time off a positioned `IndexInput`.
        // Here the whole run is one bounds-checked slice and then a
        // branch-free popcount loop -- the range is known before the first
        // read, which Java's stateful input cannot express.
        // `target_in_block <= 0xFFFF` bounds `to` at 8192, which is exactly
        // `dense_bitmap`'s length, so this range is always inside the slice
        // the header already checked.
        let from = (self.word_index + 1) as usize * 8;
        let to = (target_word_index as usize + 1) * 8;
        if from < to {
            let mut ones = 0u32;
            let mut last = self.word;
            for chunk in self.dense_bitmap[from..to].chunks_exact(8) {
                last = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
                ones += last.count_ones();
            }
            self.word = last;
            self.number_of_ones += ones as i64;
        }
        self.word_index = target_word_index;

        // Java's `word >>> target` -- a long shift uses only the low 6 bits of
        // the shift amount, so the full doc id and its in-block low bits agree.
        let left_bits = self.word >> (target_in_block & 63);
        self.index = self.number_of_ones - left_bits.count_ones() as i64;
        left_bits & 1 != 0
    }

    /// Port of `IndexedDISI.rankSkip`: jump to the rank boundary at or before
    /// `target_in_block`, so at most `2^(denseRankPower - 6)` words are read
    /// and popcounted afterwards regardless of how far the cursor jumped.
    // ARITH: `target_in_block` is masked to `0..=0xFFFF` and
    // `dense_rank_power` is in `7..=15` (the caller only enters here when
    // `dense_rank` is non-empty, which `dense_rank_bytes` only returns for
    // that range), so `rank_index <= 0xFFFF >> 7 == 511`, `entry <= 1022` and
    // `entry + 1 <= 1023`; `dense_rank` is `1024 >> (power - 7)` bytes, which
    // is exactly `2 * (0xFFFF >> power) + 2`, so `entry + 1` is its last
    // index. `rank_aligned_word_index <= 0xFFFF >> 6 == 1023`, so
    // `at + 8 <= 8192`, `dense_bitmap`'s exact length.
    #[allow(clippy::arithmetic_side_effects)]
    fn rank_skip(&mut self, target_in_block: i32) {
        // `rank_index` is at most `0xFFFF >> power`, and the table is
        // `1024 >> (power - 7)` bytes, so `entry + 1` is always the last legal
        // pair -- both slices were length-checked when the header was read.
        let rank_index = (target_in_block >> self.dense_rank_power) as usize;
        let entry = rank_index << 1;
        let rank = ((self.dense_rank[entry] as i64) << 8) | self.dense_rank[entry + 1] as i64;

        let rank_aligned_word_index = (rank_index << self.dense_rank_power) >> 6;
        let at = rank_aligned_word_index * 8;
        let rank_word = u64::from_le_bytes(
            self.dense_bitmap[at..at + 8]
                .try_into()
                .expect("8 bytes inside the bitmap"),
        );
        self.word_index = rank_aligned_word_index as i32;
        self.word = rank_word;
        self.number_of_ones = self.dense_origo_index + rank + rank_word.count_ones() as i64;
    }
}

/// Convenience: whether `doc` has a value, and if so its ordinal (rank)
/// among docs that do — the position doc-values/norms sparse arrays index
/// by. `docs` must be the ascending list `decode_doc_ids` returns.
pub fn rank_of(docs: &[i32], doc: i32) -> Option<usize> {
    docs.binary_search(&doc).ok()
}

/// Builds the wire bytes [`decode_doc_ids`] parses back, given `doc_ids` in
/// strictly ascending order (the caller's job -- e.g. a sorted `BTreeMap`
/// key iteration). Chooses a block shape per 65536-doc range exactly the way
/// real `IndexedDISIBuilder` does (by cardinality within that range): `<=
/// 4095` docs -> SPARSE (explicit low-16-bit doc ids), `== 65536` -> ALL (no
/// data), otherwise -> DENSE (a 65536-bit bitset). Writes DENSE blocks
/// **without** a rank table, matching every sparse entry this port writes:
/// `dense_rank_power` is [`NO_RANK`]/`0xFF` in the meta, which is Java's own
/// "no rank table" encoding, so a reader on either side never looks for one.
/// [`write_with_dense_rank_power`] is the same writer with the table.
///
/// Panics if `doc_ids` isn't strictly ascending, or contains `i32::MAX`
/// (Lucene's `NO_MORE_DOCS` sentinel, never a real doc id) -- both are
/// caller bugs, not something a well-formed write path can hit.
pub fn write(doc_ids: &[i32]) -> Vec<u8> {
    write_with_dense_rank_power(doc_ids, NO_RANK)
}

/// [`write`], with control over the DENSE rank table -- the full port of
/// `IndexedDISI.writeBitSet(it, out, denseRankPower)`.
///
/// `dense_rank_power` must be [`NO_RANK`] (no table) or `7..=15`, the same
/// domain Java's `writeBitSet` validates; the caller must record it in the
/// matching `.dvm`/`.nvm` metadata byte or the region is unreadable.
/// [`DEFAULT_DENSE_RANK_POWER`] is what real Lucene uses.
///
/// The jump table `writeBitSet` appends after the last block is still not
/// emitted (this port's writers record `jumpTableEntryCount = 0`, Java's own
/// "no jump table" encoding); see [`DisiCursor`].
// ARITH: this is the encode side, driven by an in-memory `doc_ids` slice the
// assertions below prove strictly ascending and free of `i32::MAX`. `block` is
// a `u16`, so `block_base <= 0xFFFF0000` and `block_base + 65536` fits `i64`
// comfortably; `i` indexes `doc_ids`, whose length is at most `isize::MAX`;
// `count >= 1` because the inner scan always advances at least once (the doc
// that chose `block_end` is itself below it), so `count - 1` cannot underflow;
// and `d - block_base` is in `0..65536` for every doc the scan collected.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_with_dense_rank_power(doc_ids: &[i32], dense_rank_power: u8) -> Vec<u8> {
    assert!(
        dense_rank_power == NO_RANK || (7..=15).contains(&dense_rank_power),
        "denseRankPower must be 0xFF (none) or 7-15, got {dense_rank_power}"
    );
    let mut out = Vec::new();

    // Checked once over the whole slice rather than per block: the old
    // per-block-first-doc check let a descending or duplicated pair *inside* a
    // block through, which silently produced a SPARSE block with out-of-order
    // shorts (or a DENSE block whose header cardinality counted a duplicate
    // twice) -- exactly the corruption the documented contract says it rejects.
    assert!(
        doc_ids.windows(2).all(|w| w[0] < w[1]),
        "doc_ids must be strictly ascending"
    );
    assert!(
        doc_ids.last() != Some(&i32::MAX),
        "i32::MAX is not a valid doc id"
    );
    // Ascending order alone does not rule out a negative first doc id, and a
    // negative one puts `block_base` at 0xFFFF0000: the SPARSE path would then
    // write nonsense 16-bit ids and the DENSE path would index `words` with an
    // astronomic `rel`. This is the invariant the `d - block_base` half of the
    // proof below rests on.
    assert!(
        doc_ids.first().is_none_or(|&d| d >= 0),
        "doc ids must be non-negative"
    );

    let mut i = 0usize;
    while i < doc_ids.len() {
        let doc = doc_ids[i];
        let block = (doc as i64 >> 16) as u16;
        let block_base = (block as i64) << 16;
        let block_end = block_base + BLOCK_SIZE as i64;

        let start = i;
        while i < doc_ids.len() && (doc_ids[i] as i64) < block_end {
            i += 1;
        }
        let block_docs = &doc_ids[start..i];
        let count = block_docs.len() as u32;

        out.extend_from_slice(&block.to_le_bytes());
        out.extend_from_slice(&((count - 1) as u16).to_le_bytes());

        if count <= MAX_ARRAY_LENGTH {
            for &d in block_docs {
                out.write_i16(((d as i64 - block_base) as u16) as i16);
            }
        } else if count == BLOCK_SIZE {
            // ALL: nothing to write.
        } else {
            let mut words = vec![0u64; DENSE_BLOCK_LONGS as usize];
            for &d in block_docs {
                let rel = (d as i64 - block_base) as usize;
                words[rel / 64] |= 1u64 << (rel % 64);
            }
            if dense_rank_power != NO_RANK {
                out.extend_from_slice(&create_rank(&words, dense_rank_power));
            }
            for w in words {
                out.write_i64(w as i64);
            }
        }
    }

    // Terminating sentinel block: doc id `i32::MAX` (Lucene's `NO_MORE_DOCS`),
    // written as a 1-doc SPARSE block.
    let sentinel_block = (i32::MAX as i64 >> 16) as u16;
    out.extend_from_slice(&sentinel_block.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // numValues - 1 == 0
    out.write_i16(((i32::MAX as i64 & 0xFFFF) as u16) as i16);

    out
}

/// Port of `IndexedDISI.createRank`: one 2-byte big-endian entry per
/// `2^dense_rank_power` doc ids, each holding the number of set bits in the
/// block *before* that sub-block starts. `words` is the block's 1024-word
/// bitset. The entries are what [`DisiCursor::rank_skip`] jumps by.
// ARITH: the sole caller reaches this only past
// `write_with_dense_rank_power`'s assertion, so `dense_rank_power` is in
// `7..=15`: `- 6` and `- 7` cannot underflow, `1usize << (power - 6)` is in
// `2..=512` so `longs_per_rank - 1` is fine, and `word >> rank_index_shift` is
// at most `1023 >> 0 == 1023` with `rank` sized `1024 >> rank_index_shift`
// bytes -- the `+ 1` lands on its last index because `word` is a multiple of
// `longs_per_rank` whenever the write runs. `bit_count` sums at most 1024
// words of 64 bits, well inside `u32`.
#[allow(clippy::arithmetic_side_effects)]
fn create_rank(words: &[u64], dense_rank_power: u8) -> Vec<u8> {
    debug_assert_eq!(words.len(), DENSE_BLOCK_LONGS as usize);
    let longs_per_rank = 1usize << (dense_rank_power - 6);
    let rank_mark = longs_per_rank - 1;
    // 6 for the long (2^6) + 1 for the 2 bytes per entry, exactly as Java.
    let rank_index_shift = dense_rank_power - 7;
    let mut rank = vec![0u8; (DENSE_BLOCK_LONGS >> rank_index_shift) as usize];
    let mut bit_count: u32 = 0;
    for word in 0..DENSE_BLOCK_LONGS as usize {
        if word & rank_mark == 0 {
            rank[word >> rank_index_shift] = (bit_count >> 8) as u8;
            rank[(word >> rank_index_shift) + 1] = (bit_count & 0xFF) as u8;
        }
        bit_count += words[word].count_ones();
    }
    rank
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    /// The cursor and the decode-everything path must agree for every document
    /// in range, across all three block shapes and across block boundaries.
    ///
    /// Two implementations of one lookup is the shape that diverges silently,
    /// and this one has three encodings, a rank table and a sentinel to get
    /// wrong. Checked over the whole doc-id range rather than at sampled
    /// points, because the interesting answers are the `None`s.
    ///
    /// Run at every legal `denseRankPower` as well as with no table at all:
    /// the rank jump changes `word_index`/`number_of_ones` behind the answer,
    /// so an off-by-one there is only visible as a wrong *ordinal*, never as a
    /// wrong present/absent.
    fn assert_cursor_matches(doc_ids: &[i32], max_doc: i32) {
        for power in [NO_RANK, 7, 9, 12, 15] {
            let bytes = write_with_dense_rank_power(doc_ids, power);
            let decoded = decode_doc_ids(&bytes, power).unwrap();
            assert_eq!(
                decoded, doc_ids,
                "fixture does not round-trip at power {power}"
            );

            let mut cursor = DisiCursor::new(&bytes, power);
            for doc in 0..max_doc {
                let want = rank_of(&decoded, doc);
                let got = cursor.advance_exact(doc).unwrap();
                assert_eq!(got, want, "disagreement at doc {doc}, power {power}");
            }

            // The same answers again, but reached by jumping straight to each
            // present doc rather than walking every doc id. This is the path
            // `rank_skip` is on: a stride of 512+ doc ids inside a DENSE block
            // is exactly what triggers it.
            let mut cursor = DisiCursor::new(&bytes, power);
            for (ordinal, &doc) in decoded.iter().enumerate() {
                assert_eq!(
                    cursor.advance_exact(doc).unwrap(),
                    Some(ordinal),
                    "jumped lookup disagreed at doc {doc}, power {power}"
                );
            }
        }
    }

    #[test]
    fn cursor_matches_decode_for_a_sparse_block() {
        let docs: Vec<i32> = (0..50).map(|i| i * 7).collect();
        assert_cursor_matches(&docs, 400);
    }

    #[test]
    fn cursor_matches_decode_for_a_dense_block() {
        // Above MAX_ARRAY_LENGTH (4095) and below BLOCK_SIZE, so DENSE.
        let docs: Vec<i32> = (0..10_000).map(|i| i * 6).collect();
        assert_cursor_matches(&docs, 65_536);
    }

    #[test]
    fn cursor_matches_decode_for_an_all_block() {
        let docs: Vec<i32> = (0..65_536).collect();
        assert_cursor_matches(&docs, 65_536);
    }

    #[test]
    fn cursor_matches_decode_across_block_boundaries() {
        // Block 0 sparse, block 1 entirely absent, block 2 dense: the cursor has
        // to walk a block with no values at all, which is the case a
        // "step to the next block" loop gets wrong.
        let mut docs: Vec<i32> = (0..40).map(|i| i * 3).collect();
        docs.extend((0..6_000).map(|i| 2 * 65_536 + i * 9));
        docs.retain(|&d| d < 3 * 65_536);
        docs.sort_unstable();
        docs.dedup();
        assert_cursor_matches(&docs, 3 * 65_536);
    }

    #[test]
    fn cursor_matches_decode_for_an_all_block_followed_by_a_dense_one() {
        // An ALL block contributes 65,536 to the next block's ordinal base
        // while writing zero payload bytes -- the combination that breaks a
        // cursor which derives the base from bytes consumed instead of from
        // the header's `numValues`.
        let mut docs: Vec<i32> = (0..65_536).collect();
        docs.extend((0..5_000).map(|i| 65_536 + i * 13));
        assert_cursor_matches(&docs, 2 * 65_536);
    }

    /// Java's `advanceExact` asserts a non-decreasing target and is undefined
    /// otherwise. This port makes the assert unconditional, because the two
    /// alternatives are both worse: `None` is a legitimate answer ("no value
    /// here"), so a violated contract would come back as a plausible wrong
    /// number, and a silent wrong ordinal is worse still.
    #[test]
    #[should_panic(expected = "forward-only")]
    fn cursor_panics_on_a_backward_doc_rather_than_answering() {
        let docs: Vec<i32> = (0..50).map(|i| i * 7).collect();
        let bytes = write(&docs);
        let mut cursor = DisiCursor::new(&bytes, NO_RANK);
        assert_eq!(cursor.advance_exact(70).unwrap(), Some(10));
        cursor.advance_exact(7).unwrap();
    }

    #[test]
    #[should_panic(expected = "non-negative")]
    fn cursor_panics_on_a_negative_doc() {
        let bytes = write(&[1, 2, 3]);
        DisiCursor::new(&bytes, NO_RANK).advance_exact(-1).unwrap();
    }

    /// `reset` is what makes a random-access caller possible on a forward-only
    /// cursor -- `NumericReader` uses exactly this. Checked against the decoded
    /// list over a deliberately shuffled access order.
    #[test]
    fn reset_restores_random_access_and_agrees_with_decode() {
        let mut docs: Vec<i32> = (0..8_000).map(|i| i * 8).collect();
        docs.extend((0..60).map(|i| 65_536 + i * 11));
        let bytes = write(&docs);
        let decoded = decode_doc_ids(&bytes, NO_RANK).unwrap();

        let mut cursor = DisiCursor::new(&bytes, NO_RANK);
        // A pseudo-random order over the whole range, including absent docs.
        let mut probe = 12_345u32;
        for _ in 0..5_000 {
            probe = probe.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
            let doc = (probe % (2 * 65_536)) as i32;
            if doc < cursor.doc_id() {
                cursor.reset();
            }
            assert_eq!(
                cursor.advance_exact(doc).unwrap(),
                rank_of(&decoded, doc),
                "disagreement at doc {doc}"
            );
        }
    }

    /// Java short-circuits a repeated `advanceExact(target)` on `target ==
    /// doc`, returning the cached `exists`. Both answers have to survive it.
    #[test]
    fn repeating_the_same_doc_returns_the_same_answer() {
        let docs: Vec<i32> = (0..50).map(|i| i * 7).collect();
        let bytes = write(&docs);
        let mut cursor = DisiCursor::new(&bytes, NO_RANK);
        assert_eq!(cursor.advance_exact(70).unwrap(), Some(10));
        assert_eq!(cursor.advance_exact(70).unwrap(), Some(10));
        assert_eq!(cursor.advance_exact(71).unwrap(), None);
        assert_eq!(cursor.advance_exact(71).unwrap(), None);
        assert_eq!(cursor.advance_exact(77).unwrap(), Some(11));
    }

    /// `NO_MORE_DOCS` is the terminating block's payload, not a document. A
    /// cursor that just scanned the sentinel block would report it present at
    /// the field's cardinality.
    #[test]
    fn the_no_more_docs_sentinel_is_never_reported_as_present() {
        let bytes = write(&[1, 5, 100]);
        let mut cursor = DisiCursor::new(&bytes, NO_RANK);
        assert_eq!(cursor.advance_exact(NO_MORE_DOCS).unwrap(), None);
    }

    #[test]
    fn a_doc_past_every_block_has_no_value() {
        let bytes = write(&[1, 5, 100]);
        let mut cursor = DisiCursor::new(&bytes, NO_RANK);
        assert_eq!(cursor.advance_exact(1_000_000).unwrap(), None);
        assert_eq!(cursor.advance_exact(2_000_000).unwrap(), None);
    }

    /// A region that ends before its sentinel block is corrupt, and the cursor
    /// must say so rather than answering "no value" -- the region always ends
    /// with a block whose index outranks every legal target, so walking off the
    /// end can only mean truncation.
    #[test]
    fn a_truncated_region_is_an_error_not_a_missing_value() {
        let bytes = write(&[1, 5, 100]);
        let truncated = &bytes[..bytes.len() - 4];
        let mut cursor = DisiCursor::new(truncated, NO_RANK);
        // Inside the surviving first block the answer is still knowable.
        assert_eq!(cursor.advance_exact(5).unwrap(), Some(1));
        // Past it, the sentinel block's header is gone.
        assert!(cursor.advance_exact(70_000).is_err());
    }

    /// `createRank`'s exact byte layout: a big-endian 2-byte count of the set
    /// bits *before* each sub-block. Pinned rather than round-tripped, because
    /// a rank table that is self-consistent but not Java's would still read
    /// back correctly here and be silently wrong for a real Lucene reader.
    #[test]
    fn create_rank_matches_javas_byte_layout() {
        let mut words = vec![0u64; DENSE_BLOCK_LONGS as usize];
        // 64 set bits in each of the first 8 words: at power 9 (8 words per
        // entry) that is exactly 512 bits before the second sub-block.
        for w in words.iter_mut().take(8) {
            *w = u64::MAX;
        }
        // One more bit far along, in word 100 (sub-block 12).
        words[100] = 1;

        let rank = create_rank(&words, 9);
        assert_eq!(rank.len(), (DENSE_BLOCK_LONGS >> 2) as usize, "256 bytes");
        // Entry 0: nothing before the block starts.
        assert_eq!((rank[0], rank[1]), (0, 0));
        // Entry 1 covers words 8..16, and 512 bits precede it: 0x0200.
        assert_eq!((rank[2], rank[3]), (0x02, 0x00));
        // Entries 2..=12 all still see the same 512 bits.
        assert_eq!((rank[24], rank[25]), (0x02, 0x00));
        // Entry 13 covers words 104..112, past word 100's single bit: 513.
        assert_eq!((rank[26], rank[27]), (0x02, 0x01));
    }

    /// The rank table is only useful if `rank_skip` actually fires, and it only
    /// fires on a jump of at least `2^(power-6)` words. This walks a DENSE
    /// block in strides far larger than that at every legal power, so every
    /// answer comes through `rank_skip` rather than through the word loop.
    #[test]
    fn rank_skip_produces_the_same_ordinals_as_a_full_walk() {
        // Every 5th doc present: 13,108 in block 0, comfortably DENSE.
        let docs: Vec<i32> = (0..13_100).map(|i| i * 5).collect();
        let plain = decode_doc_ids(&write(&docs), NO_RANK).unwrap();
        for power in [7u8, 9, 12, 15] {
            let bytes = write_with_dense_rank_power(&docs, power);
            let mut cursor = DisiCursor::new(&bytes, power);
            // Stride of 32768 bits' worth of docs, so even power 15's
            // `2^9`-word threshold is crossed on every step.
            for doc in (0..65_536).step_by(32_771) {
                assert_eq!(
                    cursor.advance_exact(doc).unwrap(),
                    rank_of(&plain, doc),
                    "power {power}, doc {doc}"
                );
            }
        }
    }

    #[test]
    fn write_with_dense_rank_power_round_trips_through_decode() {
        let docs: Vec<i32> = (0..5_000).map(|i| i * 13).collect();
        for power in [NO_RANK, 7, 9, 15] {
            let bytes = write_with_dense_rank_power(&docs, power);
            assert_eq!(
                decode_doc_ids(&bytes, power).unwrap(),
                docs,
                "power {power}"
            );
        }
        // A rank table costs `1024 >> (power - 7)` bytes per DENSE block and
        // nothing at all elsewhere.
        assert_eq!(
            write_with_dense_rank_power(&docs, 9).len() - write(&docs).len(),
            256
        );
    }

    /// A negative doc id puts `block_base` at 0xFFFF0000, which the DENSE
    /// path turns into an astronomic `words[rel / 64]` index. Strict ascent
    /// alone does not rule it out.
    #[test]
    #[should_panic(expected = "doc ids must be non-negative")]
    fn write_rejects_a_negative_doc_id() {
        write_with_dense_rank_power(&[-5, 1, 2], DEFAULT_DENSE_RANK_POWER);
    }

    #[test]
    #[should_panic(expected = "denseRankPower")]
    fn write_rejects_an_out_of_range_dense_rank_power() {
        write_with_dense_rank_power(&[1, 2, 3], 6);
    }

    use super::*;

    fn write_block_header(out: &mut Vec<u8>, block: u16, num_values: u32) {
        out.extend_from_slice(&block.to_le_bytes());
        out.extend_from_slice(&((num_values - 1) as u16).to_le_bytes());
    }

    fn sentinel_block() -> Vec<u8> {
        let mut out = Vec::new();
        write_block_header(&mut out, (i32::MAX >> 16) as u16, 1);
        out.extend_from_slice(&((i32::MAX & 0xFFFF) as u16).to_le_bytes());
        out
    }

    #[test]
    fn sparse_block_then_sentinel() {
        let mut data = Vec::new();
        write_block_header(&mut data, 0, 3);
        for v in [1u16, 5, 100] {
            data.extend_from_slice(&v.to_le_bytes());
        }
        data.extend_from_slice(&sentinel_block());

        let docs = decode_doc_ids(&data, NO_RANK).unwrap();
        assert_eq!(docs, vec![1, 5, 100]);
    }

    #[test]
    fn all_block_then_sentinel() {
        let mut data = Vec::new();
        write_block_header(&mut data, 0, BLOCK_SIZE);
        data.extend_from_slice(&sentinel_block());

        let docs = decode_doc_ids(&data, NO_RANK).unwrap();
        assert_eq!(docs.len(), BLOCK_SIZE as usize);
        assert_eq!(docs[0], 0);
        assert_eq!(docs[BLOCK_SIZE as usize - 1], BLOCK_SIZE as i32 - 1);
    }

    #[test]
    fn dense_block_without_rank_then_sentinel() {
        let mut data = Vec::new();
        write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1); // smallest DENSE size
        let mut words = vec![0i64; DENSE_BLOCK_LONGS as usize];
        words[0] = 0b1011; // bits 0,1,3 set
        words[1] = 1 << 5; // doc 64+5 = 69
        for w in &words {
            data.extend_from_slice(&w.to_le_bytes());
        }
        data.extend_from_slice(&sentinel_block());

        let docs = decode_doc_ids(&data, NO_RANK).unwrap();
        assert_eq!(docs, vec![0, 1, 3, 69]);
    }

    #[test]
    fn dense_block_with_rank_table_is_skipped_correctly() {
        let dense_rank_power = 9u8; // default: rank every 512 docs (8 longs)
        let rank_bytes = (DENSE_BLOCK_LONGS >> (dense_rank_power - 7)) as usize;

        let mut data = Vec::new();
        write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1);
        data.extend(vec![0xAAu8; rank_bytes]); // rank table: content irrelevant, just skipped
        let mut words = vec![0i64; DENSE_BLOCK_LONGS as usize];
        words[0] = 1; // doc 0
        for w in &words {
            data.extend_from_slice(&w.to_le_bytes());
        }
        data.extend_from_slice(&sentinel_block());

        let docs = decode_doc_ids(&data, dense_rank_power).unwrap();
        assert_eq!(docs, vec![0]);
    }

    #[test]
    fn multiple_blocks_across_ranges() {
        let mut data = Vec::new();
        write_block_header(&mut data, 0, 1);
        data.extend_from_slice(&5u16.to_le_bytes()); // doc 5
        write_block_header(&mut data, 1, 1);
        data.extend_from_slice(&7u16.to_le_bytes()); // doc (1<<16)|7 = 65543
        data.extend_from_slice(&sentinel_block());

        let docs = decode_doc_ids(&data, NO_RANK).unwrap();
        assert_eq!(docs, vec![5, 65543]);
    }

    #[test]
    fn empty_structure_is_just_the_sentinel() {
        let data = sentinel_block();
        let docs = decode_doc_ids(&data, NO_RANK).unwrap();
        assert!(docs.is_empty());
    }

    #[test]
    fn dense_rank_power_outside_javas_legal_range_is_rejected() {
        // Java's IndexedDISI constructor throws IllegalArgumentException for
        // anything but -1 (0xFF) or 7..=15. The byte comes off disk, so an
        // out-of-range one must be an error, not a shift-underflow panic.
        let mut data = Vec::new();
        write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1); // forces DENSE
        data.extend(vec![0u8; DENSE_BLOCK_LONGS as usize * 8]);
        data.extend_from_slice(&sentinel_block());

        for bad in [0u8, 6, 16, 200] {
            assert!(
                decode_doc_ids(&data, bad).is_err(),
                "denseRankPower {bad} must be rejected"
            );
            let mut cursor = DisiCursor::new(&data, bad);
            assert!(cursor.advance_exact(0).is_err(), "denseRankPower {bad}");
        }
        // The legal extremes still parse.
        for good in [7u8, 9, 15, NO_RANK] {
            let mut data = Vec::new();
            write_block_header(&mut data, 0, MAX_ARRAY_LENGTH + 1);
            data.extend(vec![0u8; dense_rank_bytes(good).unwrap()]);
            let mut words = vec![0i64; DENSE_BLOCK_LONGS as usize];
            words[0] = 1; // doc 0
            for w in &words {
                data.extend_from_slice(&w.to_le_bytes());
            }
            data.extend_from_slice(&sentinel_block());
            assert_eq!(
                decode_doc_ids(&data, good).unwrap(),
                vec![0],
                "power {good}"
            );
            assert_eq!(
                DisiCursor::new(&data, good).advance_exact(0).unwrap(),
                Some(0),
                "power {good}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn write_rejects_a_descending_pair_inside_a_block() {
        // Both docs land in block 0, so only a whole-slice check catches this.
        write(&[10, 20, 15]);
    }

    #[test]
    #[should_panic(expected = "strictly ascending")]
    fn write_rejects_duplicate_doc_ids() {
        write(&[10, 10]);
    }

    #[test]
    fn rank_of_found_and_not_found() {
        let docs = vec![1, 5, 100];
        assert_eq!(rank_of(&docs, 5), Some(1));
        assert_eq!(rank_of(&docs, 2), None);
    }

    #[test]
    fn truncated_input_is_eof_error() {
        let data = vec![0u8; 2]; // half a block header
        assert!(decode_doc_ids(&data, NO_RANK).is_err());
    }

    // --- `write` (the writer added alongside the pre-existing reader above) ---

    #[test]
    fn write_empty_doc_ids_round_trips_to_empty() {
        let data = write(&[]);
        assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), Vec::<i32>::new());
    }

    #[test]
    fn write_single_doc_id_round_trips() {
        let data = write(&[42]);
        assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), vec![42]);
    }

    #[test]
    fn write_exactly_max_array_length_stays_sparse_shape() {
        // 4095 (MAX_ARRAY_LENGTH) present docs in one block: the boundary
        // value itself must still take the SPARSE-as-shorts shape (`<=`,
        // not `<`), not spill into the DENSE bitset shape one doc early.
        let doc_ids: Vec<i32> = (0..MAX_ARRAY_LENGTH as i32).collect();
        let data = write(&doc_ids);
        assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
    }

    #[test]
    fn write_one_more_than_max_array_length_switches_to_dense_shape() {
        // 4096 present docs: one past the SPARSE/DENSE boundary, must
        // decode identically via the DENSE bitset shape instead.
        let doc_ids: Vec<i32> = (0..(MAX_ARRAY_LENGTH as i32 + 1)).collect();
        let data = write(&doc_ids);
        assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
    }

    #[test]
    fn write_exactly_block_size_minus_one_stays_dense_shape() {
        // 65535 present docs (one short of a full 65536-doc block): must
        // NOT be mistaken for the ALL shape (which requires every doc in
        // the block, i.e. exactly BLOCK_SIZE).
        let doc_ids: Vec<i32> = (0..(BLOCK_SIZE as i32 - 1)).collect();
        let data = write(&doc_ids);
        assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
    }

    #[test]
    fn write_exactly_block_size_uses_all_shape() {
        // Every doc in the block present: the ALL shape (zero body bytes
        // for the block, per real Lucene's IndexedDISIBuilder).
        let doc_ids: Vec<i32> = (0..BLOCK_SIZE as i32).collect();
        let data = write(&doc_ids);
        assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
    }

    #[test]
    fn write_spans_a_block_boundary_correctly() {
        // One doc in the last slot of block 0 and one in the first slot of
        // block 1 -- proves block partitioning doesn't off-by-one at the
        // 65536 boundary itself.
        let doc_ids = vec![BLOCK_SIZE as i32 - 1, BLOCK_SIZE as i32];
        let data = write(&doc_ids);
        assert_eq!(decode_doc_ids(&data, NO_RANK).unwrap(), doc_ids);
    }
}
