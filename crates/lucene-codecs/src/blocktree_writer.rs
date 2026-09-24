//! Write side of the block-tree term dictionary: `.tim` blocks and the `.tip`
//! trie over them, for one field at a time.
//!
//! A port of `Lucene103BlockTreeTermsWriter.TermsWriter` (`pushTerm`,
//! `writeBlocks`, `writeBlock`, `finish`) and `TrieBuilder` (`append`,
//! `saveNodes`, `freezeNode`, `ChildSaveStrategy`). The postings half --
//! `.doc`/`.pos`/`.pay` and the per-term metadata that points into them -- is
//! `crate::postings_writer`'s; it hands this module each term's bytes and
//! state (`BlockTermState`) plus a [`TermMetaEncoder`] that writes one
//! term's metadata (`PostingsWriterBase.encodeTerm`), one term at a time, as
//! Java's `TermsWriter.write` receives them; `finish` saves the trie and
//! writes the field's `.tmd` record.
//!
//! # What is ported, and what is deliberately different
//!
//! - **Block splitting, floor blocks and the multi-level trie are Java's
//!   algorithm, step for step**: the same pending stack, the same
//!   `prefixStarts` bookkeeping, the same `minItemsInBlock`/`maxItemsInBlock`
//!   defaults (25/48) and the same greedy floor segmentation. So for the same
//!   term list the blocks fall on the same boundaries as Java's, and the trie
//!   nodes pick the same child-label strategy.
//! - **Suffix compression is Java's decision procedure** (LZ4 when it saves
//!   more than a quarter, else lowercase-ASCII packing, only past a two-byte
//!   prefix and two suffix bytes per entry), with this port's own
//!   `HighCompressionHashTable`. Real Lucene's reader accepts any valid block,
//!   so byte-identity with Java's `.tim` is not a goal (`docs/milestones/
//!   m3-write-path-proven.md`, "Block-splitting thresholds change output
//!   bytes") -- readability by real Lucene is, and `VerifyTermDictionary`
//!   checks it.
//! - **`TrieBuilder` is ported with its in-memory form**: the separately
//!   held first key, the prefix-coded entry buffer `append` bulk-copies, and
//!   the two-phase frontier walk in `saveNodes`.
//!
//! Rust-forced differences only: `PendingEntry` is an enum rather than a
//! class hierarchy, sub-block tries are moved rather than referenced, and a
//! term is an index into the caller's term list rather than a copied
//! `byte[]`.

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::lz4::{self, HighCompressionHashTable};

/// `Lucene103BlockTreeTermsWriter.DEFAULT_MIN_BLOCK_SIZE`.
pub(crate) const DEFAULT_MIN_ITEMS_IN_BLOCK: usize = 25;
/// `Lucene103BlockTreeTermsWriter.DEFAULT_MAX_BLOCK_SIZE`.
pub(crate) const DEFAULT_MAX_ITEMS_IN_BLOCK: usize = 48;

const SIGN_NO_CHILDREN: u32 = 0x00;
const SIGN_SINGLE_CHILD_WITH_OUTPUT: u32 = 0x01;
const SIGN_SINGLE_CHILD_WITHOUT_OUTPUT: u32 = 0x02;
const SIGN_MULTI_CHILDREN: u32 = 0x03;
const LEAF_NODE_HAS_TERMS: u32 = 1 << 5;
const LEAF_NODE_HAS_FLOOR: u32 = 1 << 6;
const NON_LEAF_NODE_HAS_TERMS: u64 = 1 << 1;
const NON_LEAF_NODE_HAS_FLOOR: u64 = 1;

/// `CompressionAlgorithm` codes, as the low two bits of a block's token.
const COMPRESSION_NONE: u64 = 0;
const COMPRESSION_LOWERCASE_ASCII: u64 = 1;
const COMPRESSION_LZ4: u64 = 2;

/// [`TermsWriter::write`] was given a term that does not sort strictly
/// after the previous one.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TermOutOfOrder;

/// The length of the longest common prefix of `a` and `b` (`Arrays.mismatch`,
/// or the shorter length when one is a prefix of the other), eight bytes at
/// a time.
// ARITH: `i` advances by 8 only while `i + 8 <= n`, and `trailing_zeros / 8`
// of a non-zero XOR is below 8, so every sum stays within `n`.
#[allow(clippy::arithmetic_side_effects)]
fn common_prefix(a: &[u8], b: &[u8]) -> usize {
    let n = a.len().min(b.len());
    let mut i = 0;
    while i + 8 <= n {
        let x = u64::from_le_bytes(a[i..i + 8].try_into().unwrap());
        let y = u64::from_le_bytes(b[i..i + 8].try_into().unwrap());
        let diff = x ^ y;
        if diff != 0 {
            return i + (diff.trailing_zeros() / 8) as usize;
        }
        i += 8;
    }
    while i < n && a[i] == b[i] {
        i += 1;
    }
    i
}

/// `PostingsWriterBase`'s side of the term dictionary: the per-term state
/// its `writeTerm` returns (`BlockTermState`) and `encodeTerm`, which
/// appends one term's postings metadata. `absolute` is true for the first
/// term of every block, where the delta base resets -- exactly
/// `SegmentTermsEnumFrame`'s per-block reset on the read side.
pub(crate) trait TermMetaEncoder {
    type State: Copy;
    /// `BlockTermState.docFreq` and `totalTermFreq`.
    fn stats(state: &Self::State) -> (i32, i64);
    fn encode_term(&mut self, out: &mut Vec<u8>, state: &Self::State, absolute: bool);
}

/// Where one field's trie landed in `.tip` -- the three `.tmd` values
/// `TrieBuilder.save` writes (`indexStart`, `rootFP`, `indexEnd`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct TrieLocation {
    index_start: u64,
    root_fp: u64,
    index_end: u64,
}

enum PendingEntry<'t, S> {
    /// `PendingTerm`: the term's bytes (borrowed from the caller, where Java
    /// copies them) and its postings state.
    Term { term: &'t [u8], state: S },
    /// Boxed: a block is several times a term's size, and terms dominate
    /// the stack.
    Block(Box<PendingBlock>),
}

struct PendingBlock {
    prefix: Vec<u8>,
    fp: u64,
    has_terms: bool,
    is_floor: bool,
    /// The floor block's leading suffix byte, `-1` for the first block of a
    /// floor run (and for a block that is not a floor block at all).
    floor_lead_byte: i32,
    /// Set by `compile_index` on the first block of a run.
    index: Option<Trie>,
    /// The tries of the sub-blocks this (non-leaf) block points at, moved
    /// into the run's trie by `compile_index`.
    sub_indices: Vec<Trie>,
}

/// `Lucene103BlockTreeTermsWriter.TermsWriter`: one field's terms, fed in
/// sorted order through [`TermsWriter::write`], then [`TermsWriter::finish`].
pub(crate) struct TermsWriter<'a, 't, E: TermMetaEncoder> {
    has_freqs: bool,
    encoder: E,
    tim: &'a mut Vec<u8>,
    min_items_in_block: usize,
    max_items_in_block: usize,
    num_terms: i64,
    sum_doc_freq: i64,
    sum_total_term_freq: i64,
    first_term: Option<&'t [u8]>,
    last_pending_term: &'t [u8],
    last_term: Vec<u8>,
    /// `prefixStarts[i]`: the index into `pending` where the entries sharing
    /// `last_term[..=i]` begin.
    prefix_starts: Vec<usize>,
    pending: Vec<PendingEntry<'t, E::State>>,
    new_blocks: Vec<PendingBlock>,
    suffix_bytes: Vec<u8>,
    suffix_lengths: Vec<u8>,
    stats: Vec<u8>,
    meta: Vec<u8>,
    spare: Vec<u8>,
    lz4_table: Option<Box<HighCompressionHashTable>>,
}

impl<'a, 't, E: TermMetaEncoder> TermsWriter<'a, 't, E> {
    /// `new TermsWriter(fieldInfo)`. The block sizes must satisfy
    /// `Lucene103BlockTreeTermsWriter.validateSettings`, which the caller
    /// checks.
    pub(crate) fn new(
        tim: &'a mut Vec<u8>,
        has_freqs: bool,
        encoder: E,
        min_items_in_block: usize,
        max_items_in_block: usize,
    ) -> Self {
        debug_assert!(min_items_in_block >= 2);
        debug_assert!(min_items_in_block.saturating_sub(1).saturating_mul(2) <= max_items_in_block);
        Self {
            has_freqs,
            encoder,
            tim,
            min_items_in_block,
            max_items_in_block,
            num_terms: 0,
            sum_doc_freq: 0,
            sum_total_term_freq: 0,
            first_term: None,
            last_pending_term: &[],
            last_term: Vec::new(),
            prefix_starts: Vec::new(),
            pending: Vec::new(),
            new_blocks: Vec::new(),
            suffix_bytes: Vec::new(),
            suffix_lengths: Vec::new(),
            stats: Vec::new(),
            meta: Vec::new(),
            spare: Vec::new(),
            lz4_table: None,
        }
    }

    /// `TermsWriter.write`, after the postings writer has written the term:
    /// `pushTerm`, then the term joins the pending stack. Terms arrive sorted
    /// ascending without duplicates.
    // ARITH: the sums are of per-term document and occurrence counts over one
    // field, which Java keeps in `long` for the same reason: they are bounded
    // by the segment's in-memory postings.
    #[allow(clippy::arithmetic_side_effects)]
    pub(crate) fn write(&mut self, term: &'t [u8], state: E::State) -> Result<(), TermOutOfOrder> {
        // Java asserts `lastTerm < term`; the port checks it, off the same
        // common-prefix scan `pushTerm` needs anyway.
        let prefix_len = common_prefix(&self.last_term, term);
        if self.first_term.is_some() {
            let follows = prefix_len < term.len()
                && (prefix_len == self.last_term.len()
                    || term[prefix_len] > self.last_term[prefix_len]);
            if !follows {
                return Err(TermOutOfOrder);
            }
        }
        self.push_term_with_prefix(term, prefix_len);
        self.pending.push(PendingEntry::Term { term, state });
        let (doc_freq, total_term_freq) = E::stats(&state);
        self.sum_doc_freq += i64::from(doc_freq);
        self.sum_total_term_freq += total_term_freq;
        self.num_terms += 1;
        self.first_term.get_or_insert(term);
        self.last_pending_term = term;
        Ok(())
    }

    /// `TermsWriter.finish`: closes every open prefix, writes the root
    /// block, saves the trie to `tip` and appends the field's `.tmd` record
    /// to `meta`. A field with no terms writes nothing, as in Java.
    pub(crate) fn finish(
        mut self,
        tip: &mut Vec<u8>,
        meta: &mut Vec<u8>,
        field_number: i32,
        doc_count: i32,
    ) {
        let Some(first_term) = self.first_term else {
            return;
        };
        // Two empty terms: the first closes every open prefix, the second
        // is a no-op kept from Java (`pushTerm(new BytesRef())` twice).
        self.push_term(&[]);
        self.push_term(&[]);
        let count = self.pending.len();
        self.write_blocks(0, count);
        let root = match self.pending.pop() {
            Some(PendingEntry::Block(b)) if self.pending.is_empty() => b,
            _ => unreachable!("writeBlocks(0, all) leaves exactly one root block"),
        };
        debug_assert!(root.prefix.is_empty());

        meta.write_vint(field_number);
        meta.write_vlong(self.num_terms);
        if self.has_freqs {
            meta.write_vlong(self.sum_total_term_freq);
        }
        meta.write_vlong(self.sum_doc_freq);
        meta.write_vint(doc_count);
        meta.write_vint(first_term.len() as i32);
        meta.write_bytes(first_term);
        meta.write_vint(self.last_pending_term.len() as i32);
        meta.write_bytes(self.last_pending_term);
        let location = root
            .index
            .expect("compile_index ran on the root block it just wrote")
            .save(tip);
        meta.write_vlong(location.index_start as i64);
        meta.write_vlong(location.root_fp as i64);
        meta.write_vlong(location.index_end as i64);
    }

    /// `TermsWriter.pushTerm`: closes every prefix of the previous term that
    /// `text` abandons, writing a block for each that gathered at least
    /// `minItemsInBlock` entries.
    // ARITH: `i` runs over `prefix_len..last_term.len()`, so `i + 1` is at
    // most a term length. `prefix_starts[i]` is a `pending` index recorded
    // when `pending` was at least that long and `pending` only shrinks by
    // collapsing the entries above such an index, so `pending.len() -
    // prefix_starts[i]` cannot underflow; `prefix_top_size >= min >= 2` makes
    // `prefix_top_size - 1` non-negative and it is at most `prefix_starts[i]`'s
    // distance to the end, so the subtraction from it cannot underflow.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_term(&mut self, text: &'t [u8]) {
        let prefix_len = common_prefix(&self.last_term, text);
        self.push_term_with_prefix(text, prefix_len);
    }

    /// [`Self::push_term`] with the common prefix of `text` and the last
    /// term already known.
    // ARITH: as `push_term`.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_term_with_prefix(&mut self, text: &'t [u8], prefix_len: usize) {
        for i in (prefix_len..self.last_term.len()).rev() {
            let prefix_top_size = self.pending.len() - self.prefix_starts[i];
            if prefix_top_size >= self.min_items_in_block {
                self.write_blocks(i + 1, prefix_top_size);
                // Java's `prefixStarts[i] -= prefixTopSize - 1`, which can go
                // negative in its `int`. The value is dead either way: every
                // index at or above `prefix_len` is re-initialised below
                // before it is next read. Wrapping keeps Java's arithmetic
                // without a debug-build panic on a value nothing reads.
                self.prefix_starts[i] = self.prefix_starts[i].wrapping_sub(prefix_top_size - 1);
            }
        }
        if self.prefix_starts.len() < text.len() {
            self.prefix_starts.resize(text.len(), 0);
        }
        for i in prefix_len..text.len() {
            self.prefix_starts[i] = self.pending.len();
        }
        self.last_term.clear();
        self.last_term.extend_from_slice(text);
    }

    /// The lead byte of `ent`'s suffix past `prefix_len`, or `-1` for a term
    /// equal to the prefix.
    fn suffix_lead_label(ent: &PendingEntry<'t, E::State>, prefix_len: usize) -> i32 {
        match ent {
            PendingEntry::Term { term, .. } => term.get(prefix_len).map_or(-1, |&b| i32::from(b)),
            PendingEntry::Block(b) => i32::from(b.prefix[prefix_len]),
        }
    }

    /// `TermsWriter.writeBlocks`: writes the top `count` pending entries as
    /// one block, or as a run of floor blocks when they are more than
    /// `maxItemsInBlock`, then replaces them on the stack with the run's
    /// first block.
    // ARITH: `count <= pending.len()` (the callers pass either a distance to
    // `pending`'s end or its length), so `start` does not underflow; `i` and
    // `next_block_start` stay within `start..=end`, so every `-` between them
    // is non-negative.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_blocks(&mut self, prefix_len: usize, count: usize) {
        debug_assert!(count > 0);
        debug_assert!(prefix_len > 0 || count == self.pending.len());
        let end = self.pending.len();
        let start = end - count;
        let mut last_suffix_lead_label = -1i32;
        let mut has_terms = false;
        let mut has_sub_blocks = false;
        let mut next_block_start = start;
        let mut next_floor_lead_label = -1i32;

        for i in start..end {
            let ent = &self.pending[i];
            let suffix_lead_label = Self::suffix_lead_label(ent, prefix_len);
            let is_term = matches!(ent, PendingEntry::Term { .. });
            if suffix_lead_label != last_suffix_lead_label {
                let items_in_block = i - next_block_start;
                if items_in_block >= self.min_items_in_block
                    && end - next_block_start > self.max_items_in_block
                {
                    // Too many for one block: greedily cut a floor block as
                    // soon as it has `minItemsInBlock` entries.
                    let is_floor = items_in_block < count;
                    let block = self.write_block(
                        prefix_len,
                        is_floor,
                        next_floor_lead_label,
                        next_block_start,
                        i,
                        has_terms,
                        has_sub_blocks,
                    );
                    self.new_blocks.push(block);
                    has_terms = false;
                    has_sub_blocks = false;
                    next_floor_lead_label = suffix_lead_label;
                    next_block_start = i;
                }
                last_suffix_lead_label = suffix_lead_label;
            }
            if is_term {
                has_terms = true;
            } else {
                has_sub_blocks = true;
            }
        }

        if next_block_start < end {
            let items_in_block = end - next_block_start;
            let is_floor = items_in_block < count;
            let block = self.write_block(
                prefix_len,
                is_floor,
                next_floor_lead_label,
                next_block_start,
                end,
                has_terms,
                has_sub_blocks,
            );
            self.new_blocks.push(block);
        }

        let mut blocks = std::mem::take(&mut self.new_blocks);
        compile_index(&mut blocks);
        self.pending.truncate(start);
        let first = blocks.swap_remove(0);
        debug_assert!(blocks.iter().all(|b| b.sub_indices.is_empty()));
        self.pending.push(PendingEntry::Block(Box::new(first)));
        blocks.clear();
        self.new_blocks = blocks;
    }

    /// `TermsWriter.writeBlock`: writes `pending[start..end]` as one `.tim`
    /// block and returns it as a pending block.
    // ARITH: `end > start`, both within `pending`; `prefix_len` is at most
    // every entry's key length (every entry in the range shares the prefix),
    // so each `len - prefix_len` is non-negative; a sub-block was written
    // before this block, so `start_fp - block.fp` is positive. The `<< 1`/
    // `<< 3` shifts are on entry counts and byte lengths of one block, far
    // below the top bits.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    fn write_block(
        &mut self,
        prefix_len: usize,
        is_floor: bool,
        floor_lead_label: i32,
        start: usize,
        end: usize,
        has_terms: bool,
        has_sub_blocks: bool,
    ) -> PendingBlock {
        debug_assert!(end > start);
        let start_fp = self.tim.len() as u64;
        let has_floor_lead_label = is_floor && floor_lead_label != -1;
        let mut prefix = Vec::with_capacity(prefix_len + usize::from(has_floor_lead_label));
        prefix.extend_from_slice(&self.last_term[..prefix_len]);

        let num_entries = end - start;
        let mut code = (num_entries as i32) << 1;
        if end == self.pending.len() {
            code |= 1; // the last block of its floor run
        }
        self.tim.write_vint(code);

        let is_leaf_block = !has_sub_blocks;
        let mut sub_indices = Vec::new();
        let mut absolute = true;
        let mut stats = StatsWriter::new(self.has_freqs);
        for i in start..end {
            match &mut self.pending[i] {
                PendingEntry::Term { term, state } => {
                    let term: &[u8] = term;
                    debug_assert!(term.starts_with(&prefix));
                    let suffix = &term[prefix_len..];
                    if is_leaf_block {
                        self.suffix_lengths.write_vint(suffix.len() as i32);
                    } else {
                        // Non-leaf: bit 0 says term (0) or sub-block (1).
                        self.suffix_lengths.write_vint((suffix.len() as i32) << 1);
                    }
                    self.suffix_bytes.extend_from_slice(suffix);
                    let (doc_freq, total_term_freq) = E::stats(state);
                    stats.add(&mut self.stats, doc_freq, total_term_freq);
                    self.encoder.encode_term(&mut self.meta, state, absolute);
                    absolute = false;
                }
                PendingEntry::Block(block) => {
                    debug_assert!(block.prefix.starts_with(&prefix));
                    debug_assert!(block.fp < start_fp);
                    let suffix = &block.prefix[prefix_len..];
                    debug_assert!(!suffix.is_empty());
                    self.suffix_lengths
                        .write_vint(((suffix.len() as i32) << 1) | 1);
                    self.suffix_bytes.extend_from_slice(suffix);
                    self.suffix_lengths
                        .write_vlong((start_fp - block.fp) as i64);
                    sub_indices.push(
                        block
                            .index
                            .take()
                            .expect("a pending sub-block was compiled when it was written"),
                    );
                }
            }
        }
        stats.finish(&mut self.stats);

        // Suffix bytes, compressed when that pays (`writeBlock`'s own gates).
        let suffix_len = self.suffix_bytes.len();
        let mut compression = COMPRESSION_NONE;
        self.spare.clear();
        if suffix_len > 2 * num_entries && prefix_len > 2 {
            if suffix_len > 6 * num_entries {
                let table = self
                    .lz4_table
                    .get_or_insert_with(|| Box::new(HighCompressionHashTable::new()));
                lz4::compress_into(&self.suffix_bytes, &mut self.spare, table.as_mut());
                if self.spare.len() < suffix_len - (suffix_len >> 2) {
                    compression = COMPRESSION_LZ4;
                }
            }
            if compression == COMPRESSION_NONE {
                self.spare.clear();
                if compress_lowercase_ascii(&self.suffix_bytes, &mut self.spare) {
                    compression = COMPRESSION_LOWERCASE_ASCII;
                }
            }
        }
        let mut token = (suffix_len as u64) << 3;
        if is_leaf_block {
            token |= 0x04;
        }
        token |= compression;
        self.tim.write_vlong(token as i64);
        if compression == COMPRESSION_NONE {
            self.tim.write_bytes(&self.suffix_bytes);
        } else {
            self.tim.write_bytes(&self.spare);
        }
        self.suffix_bytes.clear();
        self.spare.clear();

        // Suffix lengths, collapsed to one byte when every one is the same.
        let n = self.suffix_lengths.len();
        let first = self.suffix_lengths[0];
        if self.suffix_lengths[1..].iter().all(|&b| b == first) {
            self.tim.write_vint(((n as i32) << 1) | 1);
            self.tim.write_byte(first);
        } else {
            self.tim.write_vint((n as i32) << 1);
            self.tim.write_bytes(&self.suffix_lengths);
        }
        self.suffix_lengths.clear();

        self.tim.write_vint(self.stats.len() as i32);
        self.tim.write_bytes(&self.stats);
        self.stats.clear();

        self.tim.write_vint(self.meta.len() as i32);
        self.tim.write_bytes(&self.meta);
        self.meta.clear();

        if has_floor_lead_label {
            prefix.push(floor_lead_label as u8);
        }
        PendingBlock {
            prefix,
            fp: start_fp,
            has_terms,
            is_floor,
            floor_lead_byte: floor_lead_label,
            index: None,
            sub_indices,
        }
    }
}

/// `PendingBlock.compileIndex`, on `blocks[0]` of a run: its trie is its own
/// prefix (with floor data naming every later block of the run), followed by
/// every sub-block trie any block of the run points at.
// ARITH: `blocks` is non-empty (a run writes at least one block) and later
// blocks of a run are written after the first, so `sub.fp - first.fp` is
// positive and small enough to shift left by one.
#[allow(clippy::arithmetic_side_effects)]
fn compile_index(blocks: &mut [PendingBlock]) {
    debug_assert!(
        (blocks[0].is_floor && blocks.len() > 1) || (!blocks[0].is_floor && blocks.len() == 1)
    );
    let fp = blocks[0].fp;
    let floor_data = if blocks[0].is_floor {
        let mut data = Vec::new();
        data.write_vint((blocks.len() - 1) as i32);
        for sub in &blocks[1..] {
            debug_assert!(sub.floor_lead_byte != -1);
            debug_assert!(sub.fp > fp);
            data.write_byte(sub.floor_lead_byte as u8);
            data.write_vlong((((sub.fp - fp) << 1) | u64::from(sub.has_terms)) as i64);
        }
        Some(data)
    } else {
        None
    };
    let mut trie = Trie::new(
        &blocks[0].prefix,
        TrieOutput {
            fp,
            has_terms: blocks[0].has_terms,
            floor_data,
        },
    );
    for block in blocks.iter_mut() {
        for sub in block.sub_indices.drain(..) {
            trie.append(sub);
        }
    }
    blocks[0].index = Some(trie);
}

/// `StatsWriter`: a term's `(docFreq, totalTermFreq)`, with runs of
/// singletons (`docFreq == 1`, and `totalTermFreq == 1` when freqs are
/// indexed) run-length encoded.
struct StatsWriter {
    has_freqs: bool,
    singleton_count: i32,
}

impl StatsWriter {
    fn new(has_freqs: bool) -> Self {
        Self {
            has_freqs,
            singleton_count: 0,
        }
    }

    // ARITH: `singleton_count` counts terms of one block, which holds far
    // fewer than `i32::MAX` entries; `total_term_freq >= doc_freq` when freqs
    // are indexed (every freq is at least 1), and `doc_freq << 1` is a
    // document count shifted once, which `docFreq`'s own `int` bound keeps
    // below the sign bit in Java too.
    #[allow(clippy::arithmetic_side_effects)]
    fn add(&mut self, out: &mut Vec<u8>, doc_freq: i32, total_term_freq: i64) {
        if doc_freq == 1 && (!self.has_freqs || total_term_freq == 1) {
            self.singleton_count += 1;
        } else {
            self.finish(out);
            out.write_vint(doc_freq << 1);
            if self.has_freqs {
                out.write_vlong(total_term_freq - i64::from(doc_freq));
            }
        }
    }

    // ARITH: `singleton_count > 0` on the branch that subtracts one.
    #[allow(clippy::arithmetic_side_effects)]
    fn finish(&mut self, out: &mut Vec<u8>) {
        if self.singleton_count > 0 {
            out.write_vint(((self.singleton_count - 1) << 1) | 1);
            self.singleton_count = 0;
        }
    }
}

/// `LowercaseAsciiCompression.isCompressible`.
// ARITH: `b` is a byte widened to `u32`, so `b + 1` is at most 256.
#[allow(clippy::arithmetic_side_effects)]
fn is_compressible(b: u8) -> bool {
    let high3 = (u32::from(b) + 1) & !0x1F;
    high3 == 0x20 || high3 == 0x60
}

/// `LowercaseAsciiCompression.compress`: packs four mostly-lowercase-ASCII
/// bytes into three, with an exception list for the bytes that are not.
/// Returns `false` (and leaves `out` in an unspecified state) when the input
/// is too short or has more than one exception per 32 bytes.
// ARITH: every index is below `len`; `compressed_len = len - len / 4` is at
// most `len`; `i - previous_exception_index` is non-negative because
// `previous_exception_index` only takes values of earlier `i` or steps of
// 0xFF that the loop condition keeps below `i`; `num_exceptions` is bounded
// by `len / 32`.
#[allow(clippy::arithmetic_side_effects)]
fn compress_lowercase_ascii(input: &[u8], out: &mut Vec<u8>) -> bool {
    let len = input.len();
    if len < 8 {
        return false;
    }
    let max_exceptions = len >> 5;
    let mut previous_exception_index = 0usize;
    let mut num_exceptions = 0usize;
    for (i, &b) in input.iter().enumerate() {
        if !is_compressible(b) {
            while i - previous_exception_index > 0xFF {
                num_exceptions += 1;
                previous_exception_index += 0xFF;
            }
            num_exceptions += 1;
            if num_exceptions > max_exceptions {
                return false;
            }
            previous_exception_index = i;
        }
    }

    let compressed_len = len - (len >> 2);
    let mut tmp: Vec<u8> = input
        .iter()
        .map(|&b| {
            let b = u32::from(b) + 1;
            ((b & 0x1F) | ((b & 0x40) >> 1)) as u8
        })
        .collect();
    let mut o = 0usize;
    for i in compressed_len..len {
        tmp[o] |= (tmp[i] & 0x30) << 2;
        o += 1;
    }
    for i in compressed_len..len {
        tmp[o] |= (tmp[i] & 0x0C) << 4;
        o += 1;
    }
    for i in compressed_len..len {
        tmp[o] |= (tmp[i] & 0x03) << 6;
        o += 1;
    }
    debug_assert!(o <= compressed_len);
    out.write_bytes(&tmp[..compressed_len]);

    out.write_vint(num_exceptions as i32);
    if num_exceptions > 0 {
        previous_exception_index = 0;
        for (i, &b) in input.iter().enumerate() {
            if !is_compressible(b) {
                while i - previous_exception_index > 0xFF {
                    // Deltas are single bytes, so a gap wider than 0xFF gets
                    // "artificial" exceptions that restore the byte already
                    // there.
                    out.write_byte(0xFF);
                    previous_exception_index += 0xFF;
                    out.write_byte(input[previous_exception_index]);
                }
                out.write_byte((i - previous_exception_index) as u8);
                previous_exception_index = i;
                out.write_byte(b);
            }
        }
    }
    true
}

/// `TrieBuilder.Output`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TrieOutput {
    fp: u64,
    has_terms: bool,
    floor_data: Option<Vec<u8>>,
}

impl TrieOutput {
    /// `TrieBuilder.encodeFP`.
    // ARITH: a `.tim` file pointer is far below `1 << 62`.
    #[allow(clippy::arithmetic_side_effects)]
    fn encoded_fp(&self) -> u64 {
        debug_assert!(self.fp < 1 << 62);
        (if self.floor_data.is_some() {
            NON_LEAF_NODE_HAS_FLOOR
        } else {
            0
        }) | (if self.has_terms {
            NON_LEAF_NODE_HAS_TERMS
        } else {
            0
        }) | (self.fp << 2)
    }
}

/// `TrieBuilder`: a trie of block prefixes under construction.
///
/// As in Java, the first non-empty key (`min_key`) is held apart and every
/// later entry lives in `buffer`, prefix-coded against its predecessor:
/// `[prefixLen: vInt] [suffixLen: vInt] [suffix] [fp: vLong] [hasTerms:
/// byte] [floorDataLen: vInt] [floorData]`. That is what lets `append`
/// re-encode only `other`'s first entry and bulk-copy the rest.
struct Trie {
    /// Output for the empty key (the root block's prefix), if any.
    empty_output: Option<TrieOutput>,
    min_key: Vec<u8>,
    /// Output for `min_key`; `None` when the trie has no non-empty key.
    min_output: Option<TrieOutput>,
    buffer: Vec<u8>,
    /// The last key appended, which is also the largest.
    last_key: Vec<u8>,
    max_key_depth: usize,
}

impl Trie {
    /// `TrieBuilder.bytesRefToTrie`.
    fn new(key: &[u8], output: TrieOutput) -> Self {
        let mut trie = Self {
            empty_output: None,
            min_key: key.to_vec(),
            min_output: None,
            buffer: Vec::new(),
            last_key: Vec::new(),
            max_key_depth: key.len(),
        };
        if key.is_empty() {
            trie.empty_output = Some(output);
        } else {
            trie.min_output = Some(output);
            trie.last_key.extend_from_slice(key);
        }
        trie
    }

    /// `TrieBuilder.append`: every key of `other` sorts after this trie's
    /// last key.
    // ARITH: `mismatch <= other.min_key.len()`, so the suffix length is
    // non-negative; key and floor-data lengths are those of in-memory buffers
    // built from terms, far below `i32::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
    fn append(&mut self, other: Trie) {
        debug_assert!(self.last_key < other.min_key);
        if other.empty_output.is_some() && self.empty_output.is_none() {
            self.empty_output = other.empty_output;
        }
        if let Some(min_output) = &other.min_output {
            let mismatch = self
                .last_key
                .iter()
                .zip(&other.min_key)
                .take_while(|(a, b)| a == b)
                .count();
            let suffix = &other.min_key[mismatch..];
            self.buffer.write_vint(mismatch as i32);
            self.buffer.write_vint(suffix.len() as i32);
            self.buffer.write_bytes(suffix);
            self.buffer.write_vlong(min_output.fp as i64);
            self.buffer.write_byte(u8::from(min_output.has_terms));
            match &min_output.floor_data {
                Some(floor) => {
                    self.buffer.write_vint(floor.len() as i32);
                    self.buffer.write_bytes(floor);
                }
                None => self.buffer.write_vint(0),
            }
            // `other`'s later entries are prefix-coded against
            // `other.min_key`, which is now our last entry: copy as-is.
            self.buffer.extend_from_slice(&other.buffer);
        }
        self.last_key.clear();
        self.last_key.extend_from_slice(&other.last_key);
        self.max_key_depth = self.max_key_depth.max(other.max_key_depth);
    }

    /// `TrieBuilder.save`: the nodes, then the eight bytes of over-read
    /// padding `TrieReader` relies on.
    // ARITH: `.tip` offsets are lengths of an in-memory buffer.
    #[allow(clippy::arithmetic_side_effects)]
    fn save(&self, tip: &mut Vec<u8>) -> TrieLocation {
        let index_start = tip.len() as u64;
        let root_fp = self.save_nodes(tip);
        tip.extend_from_slice(&0u64.to_le_bytes());
        TrieLocation {
            index_start,
            root_fp,
            index_end: tip.len() as u64,
        }
    }

    /// `TrieBuilder.saveNodes`: rebuilds the trie from the prefix-coded
    /// entries with a frontier of one open node per depth, writing each node
    /// once all its children are written, so every child pointer is a
    /// positive backwards delta.
    fn save_nodes(&self, tip: &mut Vec<u8>) -> u64 {
        let start = tip.len();
        let mut frontier: Vec<FrontierNode> = (0..=self.max_key_depth)
            .map(|_| FrontierNode::default())
            .collect();
        frontier[0].output = self.empty_output.clone();

        let mut iter = EntryIterator::new(self);
        while iter.has_next() {
            // Phase 1: the header only, while `iter.key` still holds the
            // previous key -- exactly what freezing its abandoned tail needs.
            let prev_key_len = iter.key_len;
            iter.read_header();
            freeze_from(
                &iter.key[..prev_key_len],
                iter.prefix_len,
                &mut frontier,
                start,
                tip,
            );
            // Phase 2: the suffix and output; `iter.key` is now this key.
            let output = iter.read_body();
            frontier[iter.key_len].output = Some(output);
        }
        freeze_from(&iter.key[..iter.key_len], 0, &mut frontier, start, tip);
        freeze_node(&frontier[0], start, tip)
    }
}

/// `TrieBuilder.EntryIterator`: walks `min_key` and then the prefix-coded
/// buffer, in two phases per entry (see [`Trie::save_nodes`]).
struct EntryIterator<'a> {
    trie: &'a Trie,
    input: SliceInput<'a>,
    min_key_consumed: bool,
    prefix_len: usize,
    suffix_len: usize,
    key: Vec<u8>,
    key_len: usize,
}

impl<'a> EntryIterator<'a> {
    fn new(trie: &'a Trie) -> Self {
        let mut key = vec![0u8; trie.max_key_depth.max(1)];
        key[..trie.min_key.len()].copy_from_slice(&trie.min_key);
        Self {
            trie,
            input: SliceInput::new(&trie.buffer),
            min_key_consumed: trie.min_output.is_none(),
            prefix_len: 0,
            suffix_len: 0,
            key,
            key_len: 0,
        }
    }

    fn has_next(&self) -> bool {
        !self.min_key_consumed || self.input.position() < self.trie.buffer.len()
    }

    /// Phase 1: `prefixLen` and `suffixLen` only.
    fn read_header(&mut self) {
        if !self.min_key_consumed {
            self.prefix_len = 0;
            self.suffix_len = self.trie.min_key.len();
            return;
        }
        self.prefix_len = read_len(&mut self.input);
        self.suffix_len = read_len(&mut self.input);
    }

    /// Phase 2: the suffix (over `key[prefix_len..]`) and the output.
    // ARITH: `prefix_len + suffix_len` is the length of a key this trie was
    // built from, at most `max_key_depth`.
    #[allow(clippy::arithmetic_side_effects)]
    fn read_body(&mut self) -> TrieOutput {
        if !self.min_key_consumed {
            self.key_len = self.trie.min_key.len();
            self.min_key_consumed = true;
            return self
                .trie
                .min_output
                .clone()
                .expect("min_key_consumed starts false only when min_output is set");
        }
        self.key_len = self.prefix_len + self.suffix_len;
        self.input
            .read_bytes(&mut self.key[self.prefix_len..self.key_len])
            .expect(BUFFER_WE_WROTE);
        let fp = self.input.read_vlong().expect(BUFFER_WE_WROTE) as u64;
        let has_terms = self.input.read_byte().expect(BUFFER_WE_WROTE) == 1;
        let floor_len = read_len(&mut self.input);
        let floor_data = (floor_len > 0).then(|| {
            let mut floor = vec![0u8; floor_len];
            self.input.read_bytes(&mut floor).expect(BUFFER_WE_WROTE);
            floor
        });
        TrieOutput {
            fp,
            has_terms,
            floor_data,
        }
    }
}

/// The trie buffer is written by [`Trie::append`] a few lines up and read
/// back only by [`EntryIterator`]; a failed read is a bug here, not input.
const BUFFER_WE_WROTE: &str = "reading back a trie buffer this writer encoded";

fn read_len(input: &mut SliceInput<'_>) -> usize {
    input.read_vint().expect(BUFFER_WE_WROTE) as usize
}

/// `TrieBuilder.FrontierNode`: the open node at one depth on the path to
/// the last key.
#[derive(Default)]
struct FrontierNode {
    output: Option<TrieOutput>,
    child_labels: Vec<u8>,
    child_fps: Vec<u64>,
}

impl FrontierNode {
    /// `FrontierNode.reset`, keeping the allocations.
    fn reset(&mut self) {
        self.output = None;
        self.child_labels.clear();
        self.child_fps.clear();
    }
}

/// `TrieBuilder.freezeFrom`: freezes the frontier nodes on `key`'s path
/// from depth `key.len()` down to (not including) `to_depth`, registering
/// each with its parent.
// ARITH: `d` runs over `to_depth + 1..=key.len()`, so `d - 1` is at least
// `to_depth`.
#[allow(clippy::arithmetic_side_effects)]
fn freeze_from(
    key: &[u8],
    to_depth: usize,
    frontier: &mut [FrontierNode],
    start: usize,
    tip: &mut Vec<u8>,
) {
    for d in (to_depth + 1..=key.len()).rev() {
        let fp = freeze_node(&frontier[d], start, tip);
        frontier[d - 1].child_labels.push(key[d - 1]);
        frontier[d - 1].child_fps.push(fp);
        frontier[d].reset();
    }
}

/// `TrieBuilder.bytesRequiredVLong`: bytes needed for `v` as a
/// little-endian integer, at least one.
// ARITH: `leading_zeros(v | 1) <= 63`, so the shifted value is at most 7.
#[allow(clippy::arithmetic_side_effects)]
fn bytes_required(v: u64) -> usize {
    8 - ((v | 1).leading_zeros() >> 3) as usize
}

/// `TrieBuilder.writeLongNBytes`: the low `n` bytes of `v`, little-endian.
fn write_n_bytes(tip: &mut Vec<u8>, v: u64, n: usize) {
    debug_assert!(n == 8 || v.checked_shr(n.saturating_mul(8) as u32) == Some(0));
    tip.extend_from_slice(&v.to_le_bytes()[..n]);
}

/// `TrieBuilder.freezeNode`: serializes one node and returns its fp
/// relative to the trie's start.
// ARITH: children are frozen before their parent, so `bottom_fp` exceeds
// every child fp; the header fields are byte counts of 1..=8 minus one, a
// strategy byte count of 1..=32 minus one, and a label below 256, each
// shifted into its own bit range of a 24-bit header.
#[allow(clippy::arithmetic_side_effects)]
fn freeze_node(node: &FrontierNode, start: usize, tip: &mut Vec<u8>) -> u64 {
    let bottom_fp = (tip.len() - start) as u64;
    let children = node.child_labels.len();
    match children {
        0 => {
            let output = node
                .output
                .as_ref()
                .expect("a trie leaf always carries an output");
            let fp_bytes = bytes_required(output.fp);
            let header = SIGN_NO_CHILDREN
                | (((fp_bytes - 1) as u32) << 2)
                | if output.has_terms {
                    LEAF_NODE_HAS_TERMS
                } else {
                    0
                }
                | if output.floor_data.is_some() {
                    LEAF_NODE_HAS_FLOOR
                } else {
                    0
                };
            tip.push(header as u8);
            write_n_bytes(tip, output.fp, fp_bytes);
            if let Some(floor) = &output.floor_data {
                tip.extend_from_slice(floor);
            }
        }
        1 => {
            let child_delta = bottom_fp - node.child_fps[0];
            debug_assert!(child_delta > 0);
            let child_bytes = bytes_required(child_delta);
            let output_bytes = node
                .output
                .as_ref()
                .map_or(0, |o| bytes_required(o.fp << 2));
            let sign = if node.output.is_some() {
                SIGN_SINGLE_CHILD_WITH_OUTPUT
            } else {
                SIGN_SINGLE_CHILD_WITHOUT_OUTPUT
            };
            // With no output, Java's `(0 - 1) << 5` sets bits 5..31 of an int
            // that is then truncated to a byte: bits 5-7 set. Reproduced
            // rather than tidied, so the header byte is Java's.
            let header =
                sign as i32 | (((child_bytes - 1) as i32) << 2) | ((output_bytes as i32 - 1) << 5);
            tip.push(header as u8);
            tip.push(node.child_labels[0]);
            write_n_bytes(tip, child_delta, child_bytes);
            if let Some(output) = &node.output {
                write_n_bytes(tip, output.encoded_fp(), output_bytes);
                if let Some(floor) = &output.floor_data {
                    tip.extend_from_slice(floor);
                }
            }
        }
        _ => {
            let min_label = u32::from(node.child_labels[0]);
            let max_label = u32::from(node.child_labels[children - 1]);
            debug_assert!(max_label > min_label);
            let strategy = ChildSaveStrategy::choose(min_label, max_label, children as u32);
            let strategy_bytes = strategy.need_bytes(min_label, max_label, children as u32);
            debug_assert!((1..=32).contains(&strategy_bytes));
            let max_child_delta = bottom_fp - node.child_fps[0];
            let children_fp_bytes = bytes_required(max_child_delta);
            let output_bytes = node
                .output
                .as_ref()
                .map_or(1, |o| bytes_required(o.fp << 2));
            let header = SIGN_MULTI_CHILDREN
                | (((children_fp_bytes - 1) as u32) << 2)
                | (u32::from(node.output.is_some()) << 5)
                | (((output_bytes - 1) as u32) << 6)
                | (strategy.code() << 9)
                | ((strategy_bytes - 1) << 11)
                | (min_label << 16);
            write_n_bytes(tip, u64::from(header), 3);
            if let Some(output) = &node.output {
                write_n_bytes(tip, output.encoded_fp(), output_bytes);
                if output.floor_data.is_some() {
                    tip.push((children - 1) as u8);
                }
            }
            let strategy_start = tip.len();
            strategy.save(&node.child_labels, tip);
            debug_assert_eq!(tip.len() - strategy_start, strategy_bytes as usize);
            for &child_fp in &node.child_fps {
                debug_assert!(bottom_fp > child_fp);
                write_n_bytes(tip, bottom_fp - child_fp, children_fp_bytes);
            }
            if let Some(floor) = node.output.as_ref().and_then(|o| o.floor_data.as_ref()) {
                tip.extend_from_slice(floor);
            }
        }
    }
    bottom_fp
}

/// `TrieBuilder.ChildSaveStrategy`: how a multi-child node stores its
/// children's labels.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum ChildSaveStrategy {
    /// A presence bitset over `min..=max`.
    Bits,
    /// The labels after the first, in order.
    Array,
    /// The max label, then the labels in `min..max` that are absent.
    ReverseArray,
}

impl ChildSaveStrategy {
    fn code(self) -> u32 {
        match self {
            Self::ReverseArray => 0,
            Self::Array => 1,
            Self::Bits => 2,
        }
    }

    // ARITH: labels are bytes with `max > min` and `count` distinct labels
    // in `min..=max`, so `count <= max - min + 1` and every result is in
    // 1..=32.
    #[allow(clippy::arithmetic_side_effects)]
    fn need_bytes(self, min: u32, max: u32, count: u32) -> u32 {
        let distance = max - min + 1;
        match self {
            Self::Bits => distance.div_ceil(8),
            Self::Array => count - 1,
            Self::ReverseArray => distance - count + 1,
        }
    }

    /// `ChildSaveStrategy.choose`: the cheapest, ties to the earlier of
    /// `BITS`, `ARRAY`, `REVERSE_ARRAY`.
    fn choose(min: u32, max: u32, count: u32) -> Self {
        let mut best = Self::Bits;
        let mut best_bytes = best.need_bytes(min, max, count);
        for s in [Self::Array, Self::ReverseArray] {
            let b = s.need_bytes(min, max, count);
            if b < best_bytes {
                best = s;
                best_bytes = b;
            }
        }
        best
    }

    // ARITH: labels ascend strictly, so `label - previous` is positive and
    // `presence_index` stays below 8 after each drain; `last + 1` stays below
    // the next label, itself at most 255.
    #[allow(clippy::arithmetic_side_effects)]
    fn save(self, labels: &[u8], tip: &mut Vec<u8>) {
        match self {
            Self::Bits => {
                let mut presence_bits: u8 = 1;
                let mut presence_index = 0u32;
                let mut previous = labels[0];
                for &label in &labels[1..] {
                    presence_index += u32::from(label - previous);
                    while presence_index >= 8 {
                        tip.push(presence_bits);
                        presence_bits = 0;
                        presence_index -= 8;
                    }
                    presence_bits |= 1 << presence_index;
                    previous = label;
                }
                tip.push(presence_bits);
            }
            Self::Array => tip.extend_from_slice(&labels[1..]),
            Self::ReverseArray => {
                tip.push(labels[labels.len() - 1]);
                let mut last = labels[0];
                for &label in &labels[1..] {
                    last += 1;
                    while last < label {
                        tip.push(last);
                        last += 1;
                    }
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    // A test's `i + 1` is not a length read off disk; see
    // `docs/arithmetic-gate.md`'s "Test code" section.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    #[test]
    fn child_strategy_matches_javas_choice_and_byte_counts() {
        // Dense run: BITS wins (one byte covers eight labels).
        let dense: Vec<u8> = (b'a'..=b'h').collect();
        assert_eq!(
            ChildSaveStrategy::choose(97, 104, dense.len() as u32),
            ChildSaveStrategy::Bits
        );
        // Two far-apart labels: ARRAY (one byte) beats BITS (32 bytes).
        assert_eq!(
            ChildSaveStrategy::choose(0, 255, 2),
            ChildSaveStrategy::Array
        );
        // Nearly full range with one hole: REVERSE_ARRAY (max + the hole).
        assert_eq!(
            ChildSaveStrategy::choose(0, 255, 255),
            ChildSaveStrategy::ReverseArray
        );
        for (labels, strategy) in [
            (vec![1u8, 3, 4, 200], ChildSaveStrategy::Array),
            (vec![10u8, 11, 12, 15, 17, 30], ChildSaveStrategy::Bits),
            (
                (0u8..=254).filter(|&b| b != 7).collect(),
                ChildSaveStrategy::ReverseArray,
            ),
        ] {
            let mut out = Vec::new();
            strategy.save(&labels, &mut out);
            let (min, max) = (u32::from(labels[0]), u32::from(*labels.last().unwrap()));
            assert_eq!(
                out.len() as u32,
                strategy.need_bytes(min, max, labels.len() as u32),
                "{strategy:?}"
            );
        }
        let mut rev = Vec::new();
        ChildSaveStrategy::ReverseArray.save(&[1, 2, 4, 6, 7, 8, 9, 10], &mut rev);
        assert_eq!(rev, vec![10, 3, 5], "Java's own doc example");
        let mut bits = Vec::new();
        ChildSaveStrategy::Bits.save(&[0, 1, 9], &mut bits);
        assert_eq!(bits, vec![0b0000_0011, 0b0000_0010]);
    }

    #[test]
    fn common_prefix_matches_a_bytewise_scan() {
        let words: [&[u8]; 9] = [
            b"",
            b"a",
            b"ab",
            b"abcdefgh",
            b"abcdefghi",
            b"abcdefgz",
            b"abcdefghijklmnopq",
            b"abcdefghijklmnopz",
            b"zzzzzzzzzzzz",
        ];
        for a in words {
            for b in words {
                let want = a.iter().zip(b).take_while(|(x, y)| x == y).count();
                assert_eq!(common_prefix(a, b), want, "{a:?} {b:?}");
            }
        }
    }

    #[test]
    fn bytes_required_is_at_least_one() {
        assert_eq!(bytes_required(0), 1);
        assert_eq!(bytes_required(0xFF), 1);
        assert_eq!(bytes_required(0x100), 2);
        assert_eq!(bytes_required(u64::MAX), 8);
    }

    #[test]
    fn lowercase_ascii_round_trips_through_the_reader() {
        let mut cases: Vec<Vec<u8>> = vec![
            b"abcdefghijklmnopqrstuvwxyz0123456789".to_vec(),
            b"lowercase-with.dots_and_digits_42".to_vec(),
        ];
        // One exception per 32 bytes is allowed, including a gap over 0xFF.
        let mut long = vec![b'q'; 600];
        long[3] = b'Q';
        long[500] = 0xC3;
        cases.push(long);
        for input in cases {
            let mut out = Vec::new();
            assert!(compress_lowercase_ascii(&input, &mut out), "{input:?}");
            assert!(out.len() < input.len() + 8);
            let mut decoded = vec![0u8; input.len()];
            let mut r = lucene_store::data_input::SliceInput::new(&out);
            crate::blocktree::decompress_lowercase_ascii(&mut r, &mut decoded).unwrap();
            assert_eq!(decoded, input);
        }
    }

    /// The same real-Lucene vector `blocktree`'s decoder is pinned by
    /// (`LowercaseAsciiCompression.compress` run from lucene-core-10.5.0 on
    /// this string): the port must emit Java's bytes exactly, exception list
    /// included.
    #[test]
    fn lowercase_ascii_is_byte_identical_to_real_lucene() {
        let original = b"the-quick_brown.fox.jumps_over-42.lazy_dogs.1234567890Z!abcdefghij";
        let expected_hex = "7569664ef236aaa4aca0a3b3b0b8af8fa7b0b90fab362e3174607077a6b38e95134fad62fbbaa0e53068b4cf125394d5161701365a";
        let mut out = Vec::new();
        assert!(compress_lowercase_ascii(original, &mut out));
        let hex: String = out.iter().map(|b| format!("{b:02x}")).collect();
        assert_eq!(hex, expected_hex);
    }

    #[test]
    fn lowercase_ascii_refuses_short_or_exception_heavy_input() {
        let mut out = Vec::new();
        assert!(!compress_lowercase_ascii(b"abc", &mut out));
        assert!(!compress_lowercase_ascii(b"ABCDEFGHIJKLMNOP", &mut out));
    }
}
