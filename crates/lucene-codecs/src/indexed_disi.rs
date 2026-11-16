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
//! Both the DENSE **rank table** and the block **jump table** are written and
//! read; see [`DisiCursor::advance_block`] for the latter's two-blocks-ahead
//! rule.
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
//! Trailing the last block: an optional block jump table --
//! `jumpTableEntryCount` little-endian `(index, offset)` `i32` pairs, one per
//! logical 65536-doc block including the empty ones, where `index` is the
//! cardinality before that block and `offset` is its header's byte offset
//! within the region. The entry count is **not** in the region: it is
//! `writeBitSet`'s return value and lives in the `.dvm`/`.nvm`/`.vem` entry
//! beside the region's offset and length, and the length *includes* the table.
//!
//! [`decode_doc_ids`] ignores the table: sequential decoding stops at the
//! `NO_MORE_DOCS` sentinel block, right before the table begins.

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

/// `RandomAccessInput.readInt(pos)`: a little-endian `i32` at a byte offset,
/// which is how the block jump table's `(index, offset)` pairs are stored.
///
/// ARITH: `pos + 4` cannot overflow because `pos` comes from
/// `jump_table_entry_count * 8` with the count bounded by an `i16`.
#[allow(clippy::arithmetic_side_effects)]
fn read_i32_at(data: &[u8], pos: usize) -> Result<i32> {
    data.get(pos..pos + 4)
        .map(|b| i32::from_le_bytes([b[0], b[1], b[2], b[3]]))
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
///
/// [`decode_doc_ids_below`] with no segment to bound the documents by: the
/// order is still checked, but nothing caps the list's length below
/// `i32::MAX`. A reader that knows its segment's `maxDoc` passes it there.
pub fn decode_doc_ids(data: &[u8], dense_rank_power: u8) -> Result<Vec<i32>> {
    decode_doc_ids_below(data, dense_rank_power, i32::MAX)
}

/// [`decode_doc_ids`] for a segment of `max_doc` documents: a document at or
/// past `max_doc`, or one not strictly above the one before it, is
/// corruption, and it is refused as it is read -- so the list never holds
/// more than `max_doc` entries, whatever the bytes say. Without the order
/// check an ALL block repeated is 65,536 documents per four bytes of input.
// ARITH: `read_u16` yields a `u16`, so `1u32 + it` is at most 65 536.
// `block` is a `u16` widened to `i64`, so `block << 16` is at most 2^32 and
// every `base + ...` below adds at most 65 535 to it -- all far inside `i64`.
// `word_idx` runs over `0..1024`, `w - 1` runs only while `w != 0`, and a
// trailing-zero count is below 64. Every `as i32` is of
// a document already checked below `max_doc`.
#[allow(clippy::arithmetic_side_effects)]
pub fn decode_doc_ids_below(data: &[u8], dense_rank_power: u8, max_doc: i32) -> Result<Vec<i32>> {
    let mut input = SliceInput::new(data);
    let mut docs = Vec::new();
    let limit = i64::from(max_doc);
    // The last document taken, `-1` before the first.
    let mut last = -1i64;
    let check = |first: i64, end: i64, last: i64| {
        if first <= last || end >= limit {
            return Err(lucene_store::Error::Corrupted(format!(
                "IndexedDISI documents {first}..={end} after document {last} \
                 in a segment of {max_doc}: out of order or past the segment"
            )));
        }
        Ok(())
    };

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
                check(doc, doc, last)?;
                docs.push(doc as i32);
                last = doc;
            }
            if reached_sentinel {
                break;
            }
        } else if num_values == BLOCK_SIZE {
            // ALL: every doc in this 65536-range has a value; no bytes stored.
            let base = block << 16;
            let end = base + BLOCK_SIZE as i64 - 1;
            check(base, end, last)?;
            docs.extend((base..=end).map(|d| d as i32));
            last = end;
        } else {
            // DENSE: a 65536-bit array, optionally preceded by rank bytes we skip.
            input.skip(dense_rank_bytes(dense_rank_power)?)?;
            let base = block << 16;
            // `num_values` is below `BLOCK_SIZE` here, so this reserves at
            // most one block's worth whatever the header says.
            docs.reserve(num_values as usize);
            for word_idx in 0..DENSE_BLOCK_LONGS as i64 {
                let word = input.read_i64()? as u64;
                if word == 0 {
                    continue;
                }
                let at = base + word_idx * 64;
                let end = at + 63 - i64::from(word.leading_zeros());
                check(at + i64::from(word.trailing_zeros()), end, last)?;
                // One step per set bit: the lowest one, then clear it.
                let mut w = word;
                while w != 0 {
                    docs.push((at + i64::from(w.trailing_zeros())) as i32);
                    w &= w - 1;
                }
                last = end;
            }
        }
    }

    Ok(docs)
}

/// ORs every doc id that has a value into `words`, a bit set of
/// `words.len() * 64` documents: [`decode_doc_ids`] without the list, a DENSE
/// block's words copied whole and an ALL block's filled. A document past the
/// set is corruption.
// ARITH: `block` is a `u16`, so `block << 16 | low` is below 2^32 and
// `block << 10` plus a word index below 1024 is below 2^26 + 1024; the
// sentinel check compares against `i32::MAX`. `doc & 63` is below 64.
#[allow(clippy::arithmetic_side_effects)]
pub fn or_into_words(data: &[u8], dense_rank_power: u8, words: &mut [u64]) -> Result<()> {
    let mut input = SliceInput::new(data);
    let len = words.len();
    let past = move |doc: usize| {
        lucene_store::Error::Corrupted(format!(
            "IndexedDISI document {doc} past a {}-document set",
            len * 64
        ))
    };
    loop {
        let block = input.read_u16()? as usize;
        let num_values = 1u32 + input.read_u16()? as u32;
        let base_word = block << 10;
        if num_values <= MAX_ARRAY_LENGTH {
            for _ in 0..num_values {
                let low = input.read_u16()? as usize;
                let doc = (block << 16) | low;
                if doc == i32::MAX as usize {
                    return Ok(());
                }
                if doc >> 6 >= len {
                    return Err(past(doc));
                }
                words[doc >> 6] |= 1u64 << (doc & 63);
            }
        } else if num_values == BLOCK_SIZE {
            // Every one of the block's 65,536 documents exists, so all of
            // them must fit.
            let end = base_word + DENSE_BLOCK_LONGS as usize;
            if end > len {
                return Err(past((end << 6) - 1));
            }
            words[base_word..end].fill(u64::MAX);
        } else {
            input.skip(dense_rank_bytes(dense_rank_power)?)?;
            for i in 0..DENSE_BLOCK_LONGS as usize {
                let word = input.read_i64()? as u64;
                if word == 0 {
                    continue;
                }
                if base_word + i >= len {
                    return Err(past((base_word + i) << 6));
                }
                words[base_word + i] |= word;
            }
        }
    }
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
/// | `denseRankTable` | `dense_rank` (borrowed in place, not copied) | `rank_skip` jumps straight to the nearest rank boundary, bounding a cold lookup at one rank read + at most `2^(power-6)` word reads |
/// | `gap` | `gap` | an ALL block is one subtraction |
/// | `jumpTable`, `jumpTableEntryCount` | `jump_table`, `jump_table_entry_count` | a seek two or more blocks ahead is one random read instead of one header read per block skipped |
///
/// # Forward-only, and it says so loudly
///
/// `advance_exact` **must** be called with non-decreasing doc ids; a smaller
/// one **panics**. Java's `advanceExact` has the same precondition and enforces
/// it with a bare `assert` (off in production, undefined behaviour when
/// violated). Returning `None` instead would be the worst of the three
/// options: "this document has no value" is a legitimate answer, so a caller
/// that violated the contract would get a plausible wrong number rather than
/// a diagnosis. This file already panics on a violated writer
/// contract ([`write()`]'s ascending check), so a violated reader contract panics
/// too; corrupt *data*, as always, is an `Err`.
///
/// A caller that genuinely needs random access calls [`reset`](Self::reset)
/// first, which rewinds to the start of the region -- Java's equivalent is
/// constructing a new `IndexedDISI`. That is what
/// [`crate::doc_values::NumericReader`] does, and it costs one block-header
/// walk, never an allocation.
#[derive(Debug, Clone)]
pub struct DisiCursor<'a> {
    /// Java `slice`: the block payload, with the trailing jump table sliced
    /// off (`createBlockSlice`). Byte offsets inside it are exactly the
    /// offsets the jump table records, because both are relative to Java's
    /// `origo` -- the region's first byte.
    data: &'a [u8],
    /// Java `jumpTable`: `jumpTableEntryCount` little-endian `(index, offset)`
    /// `i32` pairs, trailing the blocks (`createJumpTable`). Empty when the
    /// region carries no table.
    jump_table: &'a [u8],
    /// Java `jumpTableEntryCount`: how many logical 65 536-doc blocks the
    /// table covers. `0`/`-1` mean "no table".
    jump_table_entry_count: i32,
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
    /// `data` is the **whole region** a `NormsEntry` or doc-values entry
    /// records -- `docsWithFieldOffset .. + docsWithFieldLength` -- which
    /// spans the block payloads *and* the trailing block jump table, because
    /// that is what `writeBitSet` appends and what `docsWithFieldLength`
    /// measures. Do not strip the table before calling: this constructor is
    /// `createBlockSlice` and `createJumpTable` together, and it needs the
    /// count to know where the split is.
    ///
    /// `jump_table_entry_count` is `writeBitSet`'s own return value as
    /// recorded in the matching `.dvm`/`.nvm`/`.vem` field; `0` and `-1` are
    /// Java's "no table". Passing a *correct* count over a region whose table
    /// has already been removed is the one misuse nothing here can catch --
    /// the count still fits, so the split silently takes `count * 8` bytes off
    /// the end of the last block instead.
    pub fn new(data: &'a [u8], dense_rank_power: u8, jump_table_entry_count: i16) -> Self {
        // `createBlockSlice` + `createJumpTable`: the table is the region's
        // last `jumpTableEntryCount * 2 * 4` bytes, and the blocks are
        // everything before it. A count of `0` or `-1` is Java's "no table".
        // A count that does not fit the region is treated as no table rather
        // than as an error, because the walk over the block headers is always
        // correct -- Java would slice past the end and fail later; degrading to
        // the (correct, slower) iteration cannot answer wrongly.
        let jump_bytes = (jump_table_entry_count.max(0) as usize).saturating_mul(8);
        let (blocks, jump_table) = match data.len().checked_sub(jump_bytes) {
            Some(split) if jump_bytes > 0 => data.split_at(split),
            _ => (data, &data[..0]),
        };
        Self::at_start(
            blocks,
            jump_table,
            i32::from(jump_table_entry_count),
            dense_rank_power,
        )
    }

    /// A cursor positioned before the first block header, over an
    /// already-split region. Java's equivalent is constructing an
    /// `IndexedDISI` from a `blockSlice`/`jumpTable` pair someone else made
    /// (its second public constructor, "in case it helps reuse").
    fn at_start(
        data: &'a [u8],
        jump_table: &'a [u8],
        jump_table_entry_count: i32,
        dense_rank_power: u8,
    ) -> Self {
        Self {
            data,
            jump_table,
            jump_table_entry_count,
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

    /// ORs every document of the set into `words` ([`or_into_words`] over
    /// this cursor's blocks): a bit set of `words.len() * 64` documents,
    /// independent of where the cursor stands.
    pub fn or_into_words(&self, words: &mut [u64]) -> Result<()> {
        or_into_words(self.data, self.dense_rank_power, words)
    }

    /// Rewinds to the start of the region, so `advance_exact` may go backwards
    /// again. Java has no equivalent because it constructs a new `IndexedDISI`;
    /// this keeps the same borrow and allocates nothing.
    pub fn reset(&mut self) {
        *self = Self::at_start(
            self.data,
            self.jump_table,
            self.jump_table_entry_count,
            self.dense_rank_power,
        );
    }

    /// Java's `docID()`: the last doc id passed to [`advance_exact`], or `-1`
    /// before the first call. A caller that may go backwards compares against
    /// this and calls [`reset`](Self::reset).
    ///
    /// [`advance_exact`]: Self::advance_exact
    pub fn doc_id(&self) -> i32 {
        self.doc
    }

    /// [`Self::advance_exact`] when `doc` is at or after the current
    /// document and inside the block already loaded -- the common case of a
    /// forward walk, with no header to read and so nothing that can fail.
    /// `None` means "ask [`Self::advance_exact`]".
    #[inline]
    pub fn advance_exact_in_block(&mut self, doc: i32) -> Option<Option<usize>> {
        if doc < self.doc || doc == NO_MORE_DOCS || (doc & !0xFFFF) != self.block {
            return None;
        }
        let found = self.advance_exact_within_block(doc);
        self.doc = doc;
        Some(found.then_some(self.index as usize))
    }

    /// The documents `start..start + 64 * words.len()` that have a value, as
    /// bits into `words`, and the ordinal the first of them has (the others
    /// follow one by one) -- or `None` when this cannot be read off a block in
    /// one piece: a window not aligned to 64 documents, one spanning two
    /// blocks, one in a SPARSE block, or one behind the cursor. The caller then
    /// asks [`Self::advance_exact`] document by document. Leaves the cursor
    /// where `advance_exact(start)` would.
    ///
    /// An ALL block's window is every document; a DENSE block's is its bitmap
    /// words, and its first ordinal what `advance_exact(start)` computes.
    // ARITH: `words.len()` is checked to fit with `start` below; the word
    // range `(start & 0xFFFF) >> 6 ..+ n` stays inside one block's 1,024
    // bitmap words because the window does not leave the block.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn window_bits(&mut self, start: i32, words: &mut [u64]) -> Result<Option<i64>> {
        let Ok(n) = i32::try_from(words.len()) else {
            return Ok(None);
        };
        let span = n.checked_mul(64).and_then(|s| start.checked_add(s));
        let Some(end) = span else {
            return Ok(None);
        };
        if n == 0 || start < 0 || start & 63 != 0 || start < self.doc {
            return Ok(None);
        }
        let block = start & !0xFFFF;
        if (end - 1) & !0xFFFF != block {
            return Ok(None);
        }
        self.advance_exact(start)?;
        if self.block != block {
            // No block for this range: none of its documents has a value.
            words.fill(0);
            return Ok(Some(self.index + 1));
        }
        match self.method {
            Method::All => {
                words.fill(u64::MAX);
                Ok(Some(self.index))
            }
            Method::Dense => {
                let first_word = ((start & 0xFFFF) >> 6) as usize;
                let bytes = &self.dense_bitmap[first_word * 8..(first_word + words.len()) * 8];
                for (w, chunk) in words.iter_mut().zip(bytes.chunks_exact(8)) {
                    *w = u64::from_le_bytes(chunk.try_into().expect("8 bytes"));
                }
                Ok(Some(self.index))
            }
            Method::Sparse => Ok(None),
        }
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

    /// Port of `IndexedDISI.advanceBlock`: one random read through the block
    /// jump table when the destination is two or more blocks ahead, and
    /// otherwise the block-by-block walk.
    ///
    /// The two-block threshold is Java's and is not arbitrary: one block ahead
    /// is a single sequential header read from bytes the previous block
    /// already warmed, which is cheaper than a random read into the table at
    /// the far end of the region.
    // ARITH: `target_block` is `doc & !0xFFFF` for a non-negative `doc`, so
    // `target_block >> 16` is in `0..=0x7FFF`; `self.block` is either `-1` or
    // a `u16` shifted left by 16, so `self.block >> 16` is `-1..=0xFFFF`. The
    // `+ 2` therefore cannot overflow. `in_range * 8` is bounded by
    // `jump_table_entry_count * 8`, which is the length of a slice that
    // exists.
    #[allow(clippy::arithmetic_side_effects)]
    fn advance_block(&mut self, target_block: i32) -> Result<()> {
        let block_index = target_block >> 16;
        if !self.jump_table.is_empty() && block_index >= (self.block >> 16) + 2 {
            // "If the jumpTableEntryCount is exceeded, there are no further
            // bits. Last entry is always NO_MORE_DOCS."
            let in_range = if block_index < self.jump_table_entry_count {
                block_index
            } else {
                self.jump_table_entry_count - 1
            } as usize;
            let entry = in_range * 8;
            let index = read_i32_at(self.jump_table, entry)?;
            let offset = read_i32_at(self.jump_table, entry + 4)?;
            // "-1 to compensate for the always-added 1 in readBlockHeader".
            self.next_block_index = i64::from(index) - 1;
            self.block_end = usize::try_from(offset).map_err(|_| {
                lucene_store::Error::Corrupted(format!(
                    "IndexedDISI jump-table offset is negative: {offset}"
                ))
            })?;
            self.read_block_header()?;
            return Ok(());
        }

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
    #[inline]
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
    #[inline]
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
pub fn write(doc_ids: &[i32]) -> (Vec<u8>, i16) {
    write_with_dense_rank_power(doc_ids, NO_RANK)
}

/// [`write()`], with control over the DENSE rank table -- the full port of
/// `IndexedDISI.writeBitSet(it, out, denseRankPower)`.
///
/// `dense_rank_power` must be [`NO_RANK`] (no table) or `7..=15`, the same
/// domain Java's `writeBitSet` validates; the caller must record it in the
/// matching `.dvm`/`.nvm` metadata byte or the region is unreadable.
/// [`DEFAULT_DENSE_RANK_POWER`] is what real Lucene uses.
///
/// Returns the region's bytes **and** `writeBitSet`'s own return value, the
/// `jumpTableEntryCount`, which the caller must record in the matching
/// `.dvm`/`.nvm`/`.vem` metadata field: it is how a reader knows how many of
/// the region's trailing bytes are the block jump table rather than block
/// payload. `0` means "no table" -- Java's own encoding, and what
/// `flushBlockJumps` returns for a single-block region, where a table would be
/// pure waste.
// ARITH: this is the encode side, driven by an in-memory `doc_ids` slice the
// assertions below prove strictly ascending and free of `i32::MAX`. `block` is
// a `u16`, so `block_base <= 0xFFFF0000` and `block_base + 65536` fits `i64`
// comfortably; `i` indexes `doc_ids`, whose length is at most `isize::MAX`;
// `count >= 1` because the inner scan always advances at least once (the doc
// that chose `block_end` is itself below it), so `count - 1` cannot underflow;
// and `d - block_base` is in `0..65536` for every doc the scan collected.
#[allow(clippy::arithmetic_side_effects)]
pub fn write_with_dense_rank_power(doc_ids: &[i32], dense_rank_power: u8) -> (Vec<u8>, i16) {
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

    // `writeBitSet`'s jump-table state: one `(index, offset)` pair per logical
    // 65 536-doc block, including the blocks that hold no document at all --
    // `addJumps` fills those with the *next* real block's pair, so a jump to an
    // empty block lands on the block that follows it.
    let mut jumps: Vec<(i32, i32)> = Vec::new();
    let mut last_block = 0usize;
    let mut total_cardinality = 0i32;

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

        add_jumps(
            &mut jumps,
            out.len(),
            total_cardinality,
            last_block,
            block as usize + 1,
        );
        last_block = block as usize + 1;
        total_cardinality += count as i32;

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
    //
    // Java: "To avoid creating 65K jump-table entries, only a single entry is
    // created pointing to the offset of the NO_MORE_DOCS block, with the
    // jumpBlockIndex set to the logical EMPTY block after all real blocks."
    add_jumps(
        &mut jumps,
        out.len(),
        total_cardinality,
        last_block,
        last_block + 1,
    );
    let sentinel_block = (i32::MAX as i64 >> 16) as u16;
    out.extend_from_slice(&sentinel_block.to_le_bytes());
    out.extend_from_slice(&0u16.to_le_bytes()); // numValues - 1 == 0
    out.write_i16(((i32::MAX as i64 & 0xFFFF) as u16) as i16);

    // `flushBlockJumps`: the table trails the blocks, and a two-entry table --
    // one real block plus the sentinel -- is "just wasted space so we ignore
    // that".
    let mut block_count = last_block + 1;
    if block_count == 2 {
        block_count = 0;
    }
    for &(index, offset) in jumps.iter().take(block_count) {
        out.write_i32(index);
        out.write_i32(offset);
    }
    // "As there are at most 32k blocks, the count is a short."
    (out, block_count as i16)
}

/// `IndexedDISI.addJumps`: record `(index, offset)` for every logical block in
/// `start_block..end_block`.
///
/// Java grows one `int[]` and indexes it by block; a `Vec` of pairs pushed in
/// order is the same table, because `writeBitSet` only ever appends and only
/// ever with `start_block == jumps.len()`.
///
/// ARITH: `end_block` is at most `0x10000` (a `u16` block index plus one) and
/// `offset` is a `Vec` length bounded by the region this writer just built,
/// which is at most `0x10000 * 8 KiB` -- both well inside `i32`, which is the
/// width Java stores them at. The `assert` is Java's own
/// (`offset < Integer.MAX_VALUE`).
#[allow(clippy::arithmetic_side_effects)]
fn add_jumps(
    jumps: &mut Vec<(i32, i32)>,
    offset: usize,
    index: i32,
    start_block: usize,
    end_block: usize,
) {
    debug_assert_eq!(jumps.len(), start_block);
    let offset = i32::try_from(offset)
        .expect("IndexedDISI region offset should not exceed 2^30 but was >= Integer.MAX_VALUE");
    for _ in start_block..end_block {
        jumps.push((index, offset));
    }
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
mod tests;
