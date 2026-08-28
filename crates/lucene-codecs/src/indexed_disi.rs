//! Port of `org.apache.lucene.codecs.lucene90.IndexedDISI` — the sparse
//! doc-id-set encoding shared by sparse norms and sparse doc values.
//!
//! **Design departure from Java on purpose** (see the `rust-performance`
//! skill): Java's `IndexedDISI` is a lazy, stateful `DocIdSetIterator` with a
//! jump table and DENSE rank cache, built for random-access seeking across
//! a long-lived reader. This port instead **decodes the whole structure once
//! into a sorted `Vec<i32>` of doc ids**, and callers binary-search it. That
//! trade is right for where this port currently sits (Phase 2: correctness
//! and read-side coverage, not the hot query path yet — see PLAN.md §7 for
//! where the dedicated performance pass belongs) and it means we never touch
//! the jump table or DENSE rank bytes at all: they exist purely to skip
//! ahead without a full scan, which a one-time decode doesn't need. We still
//! parse past them correctly (skipping the rank bytes at the right point) so
//! the cursor lands on the next block header.
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
//!     RankTable  --> u8 * rankBytes(denseRankPower)  (present iff denseRankPower != 0xFF; skipped, not used)
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
const NO_RANK: u8 = 0xFF; // Java's denseRankPower == -1, stored as a byte

pub type Result<T> = std::result::Result<T, lucene_store::Error>;

/// Decodes every doc id that has a value, in ascending order. `data` must
/// start exactly at the first block header (the same `offset` a `NormsEntry`
/// or doc-values entry records); `dense_rank_power` comes from that same
/// entry and only matters for correctly skipping DENSE blocks' rank bytes.
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
            if dense_rank_power != NO_RANK {
                let rank_bytes = (DENSE_BLOCK_LONGS >> (dense_rank_power - 7)) as usize;
                input.skip(rank_bytes)?;
            }
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

/// A forward-only cursor over an `IndexedDISI` region: `advance_exact(doc)`
/// answers "does this document have a value, and at what ordinal" in time
/// proportional to the ground covered since the last call, not to the size of
/// the field.
///
/// This is what Lucene's `IndexedDISI` is -- a `DocIdSetIterator`, with a jump
/// table -- and what [`decode_doc_ids`] is not. That function decodes the whole
/// region into a `Vec<i32>` on every call, so a caller looking documents up one
/// at a time is quadratic in the field's cardinality: measured at 874 ns for a
/// field present in 1,000 documents, 31.2 us at 10,000 and **324 us at
/// 100,000** (`indexed_disi/sparse_lookup` in `benches/hot_paths.rs`).
///
/// Forward-only, deliberately, matching Lucene: `advance_exact` must be called
/// with non-decreasing `doc`, and a smaller `doc` returns `None` rather than
/// rewinding. Sorting, faceting and range scans all walk documents in ascending
/// order, which is the access pattern this exists for. A caller that genuinely
/// needs random access should decode once and keep the result, as
/// `lucene_search::field_norms::FieldNorms` does.
///
/// The jump table and DENSE rank bytes this port writes as absent
/// (`dense_rank_power == 0xFF`) are still not used; block headers are walked
/// instead. That is O(blocks), not O(documents), and a block covers 65,536
/// documents.
#[derive(Debug)]
pub struct DisiCursor<'a> {
    data: &'a [u8],
    dense_rank_power: u8,
    /// Byte offset of the current block's header.
    block_start: usize,
    /// `doc >> 16` for the current block, or `None` before the first advance.
    block: Option<i64>,
    /// Ordinal of the current block's first present document.
    ordinal_base: usize,
    /// Present documents in the current block.
    num_values: u32,
    /// Byte offset of the current block's payload (past its header, and past a
    /// DENSE block's rank bytes).
    payload_start: usize,
    /// Set once the SPARSE sentinel (`doc == i32::MAX`) has been seen; the
    /// region has no further blocks.
    exhausted: bool,
    /// Highest doc ID passed to [`DisiCursor::advance_exact`] so far.
    ///
    /// Enforced rather than merely documented. Without it, going backwards
    /// happens to work *within* a block -- a SPARSE block rescans from its
    /// start, a DENSE one indexes by bit position -- and fails across one,
    /// which is the worst kind of contract: correct in testing, wrong on the
    /// data that spans two blocks.
    last_doc: i32,
}

impl<'a> DisiCursor<'a> {
    /// `data` must start exactly at the first block header -- the same offset a
    /// `NormsEntry` or doc-values entry records.
    pub fn new(data: &'a [u8], dense_rank_power: u8) -> Self {
        Self {
            data,
            dense_rank_power,
            block_start: 0,
            block: None,
            ordinal_base: 0,
            num_values: 0,
            payload_start: 0,
            exhausted: false,
            last_doc: -1,
        }
    }

    /// The ordinal of `doc` among documents that have a value, or `None` when
    /// `doc` has none (or is behind the cursor -- see the type's doc comment).
    pub fn advance_exact(&mut self, doc: i32) -> Result<Option<usize>> {
        if doc < 0 || self.exhausted || doc < self.last_doc {
            return Ok(None);
        }
        self.last_doc = doc;
        let want_block = (doc as i64) >> 16;
        while self.block.is_none_or(|cur| cur < want_block) {
            if !self.read_next_block_header()? {
                return Ok(None);
            }
        }
        if self.block != Some(want_block) {
            // Walked past it: this block carries no values at all.
            return Ok(None);
        }
        self.ordinal_within_block(doc)
    }

    /// Reads the next block's header and positions `payload_start`, skipping the
    /// previous block's payload. Returns `false` at the end of the region.
    fn read_next_block_header(&mut self) -> Result<bool> {
        // Step over the block we are currently on, if any.
        if self.block.is_some() {
            self.block_start = self.payload_start + self.payload_len();
            self.ordinal_base += self.num_values as usize;
        }
        if self.block_start + 4 > self.data.len() {
            self.exhausted = true;
            return Ok(false);
        }
        let mut input = SliceInput::new(self.data);
        input.seek(self.block_start)?;
        let block = input.read_u16()? as i64;
        let num_values = 1u32 + input.read_u16()? as u32;
        let mut payload_start = input.position();
        if num_values > MAX_ARRAY_LENGTH
            && num_values != BLOCK_SIZE
            && self.dense_rank_power != NO_RANK
        {
            payload_start += (DENSE_BLOCK_LONGS >> (self.dense_rank_power - 7)) as usize;
        }
        self.block = Some(block);
        self.num_values = num_values;
        self.payload_start = payload_start;
        Ok(true)
    }

    /// Byte length of the current block's payload.
    fn payload_len(&self) -> usize {
        if self.num_values <= MAX_ARRAY_LENGTH {
            self.num_values as usize * 2
        } else if self.num_values == BLOCK_SIZE {
            0
        } else {
            DENSE_BLOCK_LONGS as usize * 8
        }
    }

    /// `doc`'s ordinal within the block the cursor is positioned on.
    fn ordinal_within_block(&mut self, doc: i32) -> Result<Option<usize>> {
        let low = (doc as i64 & 0xFFFF) as u32;
        if self.num_values == BLOCK_SIZE {
            // ALL: every document in the range is present.
            return Ok(Some(self.ordinal_base + low as usize));
        }
        let mut input = SliceInput::new(self.data);
        input.seek(self.payload_start)?;
        if self.num_values <= MAX_ARRAY_LENGTH {
            // SPARSE: ascending 16-bit doc ids. Linear, bounded by 4095 and in
            // practice far shorter; the sentinel ends the whole region.
            for i in 0..self.num_values {
                let v = input.read_u16()? as i64;
                let full = ((*self.block.as_ref().expect("positioned")) << 16) | v;
                if full == i32::MAX as i64 {
                    self.exhausted = true;
                    return Ok(None);
                }
                match (v as u32).cmp(&low) {
                    std::cmp::Ordering::Equal => return Ok(Some(self.ordinal_base + i as usize)),
                    std::cmp::Ordering::Greater => return Ok(None),
                    std::cmp::Ordering::Less => {}
                }
            }
            return Ok(None);
        }
        // DENSE: count set bits before `low`, then test `low` itself.
        let word_idx = (low / 64) as usize;
        input.seek(self.payload_start + word_idx * 8)?;
        let word = input.read_i64()? as u64;
        let bit = low % 64;
        if (word >> bit) & 1 == 0 {
            return Ok(None);
        }
        let mut rank = (word & ((1u64 << bit) - 1)).count_ones() as usize;
        let mut before = SliceInput::new(self.data);
        before.seek(self.payload_start)?;
        for _ in 0..word_idx {
            rank += (before.read_i64()? as u64).count_ones() as usize;
        }
        Ok(Some(self.ordinal_base + rank))
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
/// data), otherwise -> DENSE (a 65536-bit bitset). Always writes DENSE blocks
/// **without** a rank table (matching every sparse entry this port writes:
/// `dense_rank_power` is always [`NO_RANK`]/`0xFF` in the meta, so a reader
/// never expects rank bytes here) -- real Lucene's rank table exists purely
/// to skip ahead without a full scan, irrelevant to a one-shot writer.
///
/// Panics if `doc_ids` isn't strictly ascending, or contains `i32::MAX`
/// (Lucene's `NO_MORE_DOCS` sentinel, never a real doc id) -- both are
/// caller bugs, not something a well-formed write path can hit.
pub fn write(doc_ids: &[i32]) -> Vec<u8> {
    let mut out = Vec::new();

    let mut i = 0usize;
    while i < doc_ids.len() {
        let doc = doc_ids[i];
        assert_ne!(doc, i32::MAX, "i32::MAX is not a valid doc id");
        if i > 0 {
            assert!(doc_ids[i - 1] < doc, "doc_ids must be strictly ascending");
        }
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

#[cfg(test)]
mod tests {

    /// The cursor and the decode-everything path must agree for every document
    /// in range, across all three block shapes and across block boundaries.
    ///
    /// Two implementations of one lookup is the shape that diverges silently,
    /// and this one has three encodings and a sentinel to get wrong. Checked
    /// over the whole doc-id range rather than at sampled points, because the
    /// interesting answers are the `None`s.
    fn assert_cursor_matches(doc_ids: &[i32], max_doc: i32) {
        let bytes = write(doc_ids);
        let decoded = decode_doc_ids(&bytes, NO_RANK).unwrap();
        assert_eq!(decoded, doc_ids, "fixture does not round-trip");

        let mut cursor = DisiCursor::new(&bytes, NO_RANK);
        for doc in 0..max_doc {
            let want = rank_of(&decoded, doc);
            let got = cursor.advance_exact(doc).unwrap();
            assert_eq!(got, want, "disagreement at doc {doc}");
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
    fn cursor_is_forward_only_and_says_so_by_returning_none() {
        let docs: Vec<i32> = (0..50).map(|i| i * 7).collect();
        let bytes = write(&docs);
        let mut cursor = DisiCursor::new(&bytes, NO_RANK);
        assert_eq!(cursor.advance_exact(70).unwrap(), Some(10));
        // Going backwards is not supported and must not silently answer wrongly.
        assert_eq!(cursor.advance_exact(7).unwrap(), None);
        // Forward still works afterwards.
        assert_eq!(cursor.advance_exact(77).unwrap(), Some(11));
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
