//! The block-splitting half of `Lucene103BlockTreeTermsWriter` and its
//! `.tip` index builder, `TrieBuilder` -- the part of the term-dictionary
//! write path that decides *which terms share a `.tim` block* and how the
//! trie over those blocks is laid out.
//!
//! [`crate::postings_writer`] owns everything per term (the `.doc`/`.pos`/
//! `.pay` postings and the term metadata that points into them) and the
//! per-field `.tmd` record. This module is handed a field's sorted terms and
//! a callback that encodes the metadata for any subset of them, and writes:
//!
//! - **`.tim` blocks**, split exactly as Java splits them: a term is pushed
//!   onto a pending stack; whenever the shared prefix shrinks, every prefix
//!   that has accumulated at least [`MIN_ITEMS_IN_BLOCK`] entries is written
//!   out as one or more blocks and replaced on the stack by a single
//!   *sub-block* entry. A prefix with more than [`MAX_ITEMS_IN_BLOCK`] entries
//!   is cut into **floor blocks** at changes of the next byte. A block holding
//!   sub-block entries is a non-leaf block (its suffix-length stream carries a
//!   flag bit per entry and a back-pointer per sub-block).
//! - **the `.tip` trie**: one node per block prefix, built bottom-up from the
//!   sub-block indexes each block collects, then serialised in one
//!   post-order pass with Java's four node encodings (`SIGN_NO_CHILDREN`,
//!   `SIGN_SINGLE_CHILD_WITH[OUT]_OUTPUT`, `SIGN_MULTI_CHILDREN`) and three
//!   child-label strategies (`BITS`, `ARRAY`, `REVERSE_ARRAY`).
//!
//! # What is and is not byte-identical to Java
//!
//! Block boundaries, block order, the trie's shape and every node's encoding
//! follow Java's algorithm step for step, and `VerifyIndex` checks the first
//! of those directly: for every segment of its 120 000-document index, real
//! Lucene's own writer, handed the same terms, cuts a dictionary whose
//! `Stats` (block count, floor runs, leaf/inner mix, blocks per prefix length)
//! are identical. The *bytes* differ in one deliberate way: suffixes are
//! always written `NO_COMPRESSION`, where Java tries `LZ4` and
//! `LOWERCASE_ASCII` on blocks with long enough suffixes and keeps whichever
//! saves space. Every reader accepts all three codes per block, so this changes
//! the dictionary's size, never its meaning -- but it moves block file
//! pointers, so the `.tip` bytes that encode them differ too. Recorded in
//! `docs/parity.md`.
//!
//! # The in-memory shape
//!
//! Java's `TrieBuilder` keeps its (key, output) entries prefix-coded in a
//! byte buffer, because a real segment can have millions of blocks. This port
//! keeps them as a plain `Vec` of owned keys: an entry exists per *block*, not
//! per term, so a field with a million terms has on the order of 30 000 of
//! them, and each entry moves up the pending stack at most once per trie level
//! it passes. The serialised output does not depend on the representation.

use lucene_store::data_output::DataOutput;

use crate::blocktree::{
    CHILD_STRATEGY_ARRAY, CHILD_STRATEGY_BITS, CHILD_STRATEGY_REVERSE_ARRAY, LEAF_NODE_HAS_FLOOR,
    LEAF_NODE_HAS_TERMS, NON_LEAF_NODE_HAS_FLOOR, NON_LEAF_NODE_HAS_TERMS, SIGN_MULTI_CHILDREN,
    SIGN_NO_CHILDREN, SIGN_SINGLE_CHILD_WITHOUT_OUTPUT, SIGN_SINGLE_CHILD_WITH_OUTPUT,
};

/// `Lucene103BlockTreeTermsWriter.DEFAULT_MIN_BLOCK_SIZE`.
pub const MIN_ITEMS_IN_BLOCK: usize = 25;
/// `Lucene103BlockTreeTermsWriter.DEFAULT_MAX_BLOCK_SIZE`.
pub const MAX_ITEMS_IN_BLOCK: usize = 48;

/// One term as the block writer needs it: its bytes and the two statistics
/// the block's stats stream carries. The term's postings metadata is not
/// here -- it is encoded by the caller's callback, by index.
pub(crate) struct BlockTerm<'a> {
    pub(crate) bytes: &'a [u8],
    pub(crate) doc_freq: i32,
    pub(crate) total_term_freq: i64,
}

/// Where a field's trie landed in `.tip`, as the `.tmd` record wants it:
/// `indexStart` and `indexEnd` are absolute offsets into `.tip` (`indexEnd`
/// after the 8-byte over-read pad), `root_fp` is relative to `indexStart`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TrieLocation {
    pub(crate) index_start: u64,
    pub(crate) root_fp: u64,
    pub(crate) index_end: u64,
}

/// `TrieBuilder.Output`: the block a trie node points to.
#[derive(Debug, Clone, PartialEq, Eq)]
struct TrieOutput {
    /// Absolute `.tim` offset of the block.
    fp: u64,
    /// `false` when the block holds only sub-block entries.
    has_terms: bool,
    /// Present when the prefix's entries were split into floor blocks:
    /// `vInt(count - 1)` then, per follow-on block, its lead byte and
    /// `vLong((fp - first fp) << 1 | hasTerms)`.
    floor_data: Option<Vec<u8>>,
}

/// `TrieBuilder`, as an ordered list of (non-empty key, output) entries
/// plus the empty key's output.
#[derive(Debug, Default)]
struct TrieBuilder {
    empty_output: Option<TrieOutput>,
    entries: Vec<(Vec<u8>, TrieOutput)>,
}

impl TrieBuilder {
    /// `TrieBuilder.bytesRefToTrie`.
    fn new(key: &[u8], output: TrieOutput) -> Self {
        if key.is_empty() {
            TrieBuilder {
                empty_output: Some(output),
                entries: Vec::new(),
            }
        } else {
            TrieBuilder {
                empty_output: None,
                entries: vec![(key.to_vec(), output)],
            }
        }
    }

    /// `TrieBuilder.append`: every key in `other` sorts after every key
    /// already here, which the block writer guarantees by appending sub-block
    /// indexes in the order their blocks appear.
    fn append(&mut self, other: TrieBuilder) {
        debug_assert!(
            match (self.entries.last(), other.entries.first()) {
                (Some((a, _)), Some((b, _))) => a < b,
                _ => true,
            },
            "trie entries must be appended in key order"
        );
        if self.empty_output.is_none() {
            self.empty_output = other.empty_output;
        }
        self.entries.extend(other.entries);
    }

    /// `TrieBuilder.save`: serialises the trie into `tip` and returns where
    /// it landed. `saveNodes` rebuilds the trie from the sorted entries with
    /// a frontier (one open node per depth along the last key) and writes
    /// each node as soon as no later key can add a child to it -- children
    /// always before their parent, which is what lets a parent store its
    /// children as backward deltas.
    fn save(&self, tip: &mut Vec<u8>) -> TrieLocation {
        let index_start = tip.len() as u64;
        let max_depth = self.entries.iter().map(|(k, _)| k.len()).max().unwrap_or(0);
        let mut frontier: Vec<FrontierNode<'_>> =
            (0..=max_depth).map(|_| FrontierNode::default()).collect();
        frontier[0].output = self.empty_output.as_ref();
        let mut prev: &[u8] = &[];
        for (key, output) in &self.entries {
            let common = common_prefix_len(prev, key);
            freeze_from(prev, common, &mut frontier, index_start, tip);
            frontier[key.len()].output = Some(output);
            prev = key;
        }
        freeze_from(prev, 0, &mut frontier, index_start, tip);
        let root_fp = freeze_node(&frontier[0], index_start, tip);
        // `index.writeLong(0L)`: the reader loads a node with fixed-width
        // reads that may run past its last byte.
        tip.write_i64(0);
        TrieLocation {
            index_start,
            root_fp,
            index_end: tip.len() as u64,
        }
    }
}

fn common_prefix_len(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).take_while(|(x, y)| x == y).count()
}

/// One open trie node on the path to the last key seen.
#[derive(Default)]
struct FrontierNode<'a> {
    output: Option<&'a TrieOutput>,
    /// (label, fp relative to `indexStart`), labels ascending.
    children: Vec<(u8, u64)>,
}

/// `TrieBuilder.freezeFrom`: writes the frontier nodes deeper than `to_depth`
/// on the path `key`, deepest first, registering each with its parent.
fn freeze_from(
    key: &[u8],
    to_depth: usize,
    frontier: &mut [FrontierNode<'_>],
    index_start: u64,
    tip: &mut Vec<u8>,
) {
    // `(to_depth, key.len()]`, deepest first.
    for depth in (to_depth..key.len()).map(|d| d.saturating_add(1)).rev() {
        let fp = freeze_node(&frontier[depth], index_start, tip);
        // ARITH: `depth` ranges over `to_depth + 1..`, so it is at least 1.
        #[allow(clippy::arithmetic_side_effects)]
        let (parent, label) = (depth - 1, key[depth - 1]);
        frontier[parent].children.push((label, fp));
        frontier[depth] = FrontierNode::default();
    }
}

/// `TrieBuilder.bytesRequiredVLong`: bytes needed for `v` little-endian,
/// at least one.
fn bytes_required(v: u64) -> usize {
    // ARITH: `leading_zeros` of a non-zero `u64` is at most 63, so the shift
    // yields at most 7 and the subtraction at least 1.
    #[allow(clippy::arithmetic_side_effects)]
    let n = 8 - ((v | 1).leading_zeros() >> 3) as usize;
    n
}

/// `TrieBuilder.writeLongNBytes`: the low `n` bytes of `v`, little-endian.
fn write_n_bytes(tip: &mut Vec<u8>, v: u64, n: usize) {
    debug_assert!(
        n == 8 || v.checked_shr(8u32.saturating_mul(n as u32)) == Some(0),
        "{v} does not fit {n} bytes"
    );
    tip.extend_from_slice(&v.to_le_bytes()[..n]);
}

/// `TrieBuilder.encodeFP`: an output's fp with its two flags, as a node with
/// children stores it.
fn encode_fp(output: &TrieOutput) -> u64 {
    debug_assert!(output.fp < 1 << 62);
    // ARITH: `fp` is a `.tim` offset, far below `1 << 62`.
    #[allow(clippy::arithmetic_side_effects)]
    let shifted = output.fp << 2;
    shifted
        | if output.floor_data.is_some() {
            NON_LEAF_NODE_HAS_FLOOR
        } else {
            0
        }
        | if output.has_terms {
            NON_LEAF_NODE_HAS_TERMS
        } else {
            0
        }
}

/// `TrieBuilder.freezeNode`: writes one node and returns its fp relative to
/// `index_start`.
// ARITH: every subtraction is `node position - child position`, and children
// are always written before their parent (the frontier freezes deepest
// first), so each is positive; the shifts assemble header fields whose widths
// are bounded by construction (byte counts 1..=8, a strategy code < 4, a
// strategy length 1..=32, a label < 256).
#[allow(clippy::arithmetic_side_effects)]
fn freeze_node(node: &FrontierNode<'_>, index_start: u64, tip: &mut Vec<u8>) -> u64 {
    let bottom_fp = tip.len() as u64 - index_start;
    match node.children.len() {
        0 => {
            let output = node
                .output
                .expect("a trie node with no children always has an output");
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
            let (label, child_fp) = node.children[0];
            let child_delta = bottom_fp - child_fp;
            let child_bytes = bytes_required(child_delta);
            let (sign, output_bytes) = match node.output {
                Some(o) => (SIGN_SINGLE_CHILD_WITH_OUTPUT, bytes_required(o.fp << 2)),
                // Java computes 0 here and writes `(0 - 1) << 5` into the
                // header byte; only the low eight bits survive the cast, so
                // the field reads back as 7 and is never consulted.
                None => (SIGN_SINGLE_CHILD_WITHOUT_OUTPUT, 0),
            };
            let header = (sign as i64)
                | (((child_bytes - 1) as i64) << 2)
                | ((output_bytes as i64 - 1) << 5);
            tip.push(header as u8);
            tip.push(label);
            write_n_bytes(tip, child_delta, child_bytes);
            if let Some(output) = node.output {
                write_n_bytes(tip, encode_fp(output), output_bytes);
                if let Some(floor) = &output.floor_data {
                    tip.extend_from_slice(floor);
                }
            }
        }
        n => {
            let min_label = u32::from(node.children[0].0);
            let max_label = u32::from(node.children[n - 1].0);
            debug_assert!(max_label > min_label);
            let (strategy, strategy_bytes) = choose_strategy(min_label, max_label, n as u32);
            // The first child is written first, so it is the furthest back.
            let children_fp_bytes = bytes_required(bottom_fp - node.children[0].1);
            let output_bytes = node.output.map_or(1, |o| bytes_required(o.fp << 2));
            let header = u64::from(SIGN_MULTI_CHILDREN)
                | (((children_fp_bytes - 1) as u64) << 2)
                | (u64::from(node.output.is_some()) << 5)
                | (((output_bytes - 1) as u64) << 6)
                | (u64::from(strategy) << 9)
                | (u64::from(strategy_bytes - 1) << 11)
                | (u64::from(min_label) << 16);
            write_n_bytes(tip, header, 3);
            if let Some(output) = node.output {
                write_n_bytes(tip, encode_fp(output), output_bytes);
                if output.floor_data.is_some() {
                    tip.push((n - 1) as u8);
                }
            }
            let strategy_start = tip.len();
            save_strategy(strategy, &node.children, tip);
            debug_assert_eq!(tip.len() - strategy_start, strategy_bytes as usize);
            for &(_, child_fp) in &node.children {
                write_n_bytes(tip, bottom_fp - child_fp, children_fp_bytes);
            }
            if let Some(floor) = node.output.and_then(|o| o.floor_data.as_ref()) {
                tip.extend_from_slice(floor);
            }
        }
    }
    bottom_fp
}

/// `ChildSaveStrategy.needBytes` for each strategy.
// ARITH: `max_label > min_label`, both < 256, and `count` is the number of
// distinct labels in `min_label..=max_label`, so it is at most their distance
// plus one: every difference below is non-negative and every sum tiny.
#[allow(clippy::arithmetic_side_effects)]
fn strategy_bytes(strategy: u32, min_label: u32, max_label: u32, count: u32) -> u32 {
    let distance = max_label - min_label + 1;
    match strategy {
        CHILD_STRATEGY_BITS => distance.div_ceil(8),
        CHILD_STRATEGY_ARRAY => count - 1,
        _ => distance - count + 1,
    }
}

/// `ChildSaveStrategy.choose`: the cheapest strategy, ties to the earlier
/// one in `BITS, ARRAY, REVERSE_ARRAY` order.
fn choose_strategy(min_label: u32, max_label: u32, count: u32) -> (u32, u32) {
    let mut best = (CHILD_STRATEGY_BITS, u32::MAX);
    for strategy in [
        CHILD_STRATEGY_BITS,
        CHILD_STRATEGY_ARRAY,
        CHILD_STRATEGY_REVERSE_ARRAY,
    ] {
        let cost = strategy_bytes(strategy, min_label, max_label, count);
        if cost < best.1 {
            best = (strategy, cost);
        }
    }
    best
}

/// `ChildSaveStrategy.save` for each strategy.
// ARITH: labels are strictly ascending bytes, so every difference is in
// `1..=255` and the presence index stays below 8 after each flush.
#[allow(clippy::arithmetic_side_effects)]
fn save_strategy(strategy: u32, children: &[(u8, u64)], tip: &mut Vec<u8>) {
    match strategy {
        CHILD_STRATEGY_BITS => {
            let mut bits: u8 = 1; // the first label is always present
            let mut index = 0u32;
            let mut previous = children[0].0;
            for &(label, _) in &children[1..] {
                index += u32::from(label - previous);
                while index >= 8 {
                    tip.push(bits);
                    bits = 0;
                    index -= 8;
                }
                bits |= 1 << index;
                previous = label;
            }
            tip.push(bits);
        }
        CHILD_STRATEGY_ARRAY => {
            tip.extend(children[1..].iter().map(|&(label, _)| label));
        }
        _ => {
            // REVERSE_ARRAY: the max label, then every absent label between.
            tip.push(children[children.len() - 1].0);
            let mut last = u32::from(children[0].0);
            for &(label, _) in &children[1..] {
                last += 1;
                while last < u32::from(label) {
                    tip.push(last as u8);
                    last += 1;
                }
            }
        }
    }
}

/// `PendingBlock`.
struct PendingBlock {
    prefix: Vec<u8>,
    fp: u64,
    has_terms: bool,
    is_floor: bool,
    /// `-1` for the first block of a floor run (and for a non-floor block).
    floor_lead_byte: i32,
    /// This block's own trie, once [`compile_index`] has run.
    index: Option<TrieBuilder>,
    /// The tries of the sub-blocks this block points to, until
    /// [`compile_index`] folds them into the first block of its run.
    sub_indices: Vec<TrieBuilder>,
}

enum PendingEntry {
    /// Index into the field's term list.
    Term(usize),
    Block(PendingBlock),
}

/// `StatsWriter`: `docFreq`/`totalTermFreq` per term, with runs of terms
/// that occur once (`docFreq == 1`, and `totalTermFreq == 1` when freqs are
/// indexed) collapsed into one run-length entry.
struct StatsWriter {
    out: Vec<u8>,
    has_freqs: bool,
    singletons: i32,
}

impl StatsWriter {
    // ARITH: `ttf >= df >= 1` for every indexed term (validated by the
    // postings writer before any block is written), and a block holds at
    // most a few dozen terms, so neither the difference nor the shifts can
    // overflow.
    #[allow(clippy::arithmetic_side_effects)]
    fn add(&mut self, df: i32, ttf: i64) {
        if df == 1 && (!self.has_freqs || ttf == 1) {
            self.singletons += 1;
        } else {
            self.finish();
            self.out.write_vint(df << 1);
            if self.has_freqs {
                self.out.write_vlong(ttf - i64::from(df));
            }
        }
    }

    // ARITH: `singletons >= 1` in the branch that subtracts.
    #[allow(clippy::arithmetic_side_effects)]
    fn finish(&mut self) {
        if self.singletons > 0 {
            self.out.write_vint(((self.singletons - 1) << 1) | 1);
            self.singletons = 0;
        }
    }
}

/// `TermsWriter`, for one field.
struct TermsWriter<'t, 'o, F> {
    terms: &'t [BlockTerm<'t>],
    has_freqs: bool,
    tim: &'o mut Vec<u8>,
    encode_meta: F,
    pending: Vec<PendingEntry>,
    prefix_starts: Vec<usize>,
    last_term: Vec<u8>,
}

/// Writes one field's terms into `.tim` blocks and its trie into `.tip`,
/// returning where the trie landed.
///
/// `terms` must be non-empty, strictly ascending and unique; `has_freqs` is
/// `indexOptions != DOCS`. `encode_meta(out, indices)` must append the
/// postings metadata of `terms[i]` for each `i` in `indices`, in order, as
/// `PostingsWriterBase.encodeTerm` does with `absolute = true` for the first
/// term and deltas after -- the block writer calls it once per block, with
/// that block's own terms.
pub(crate) fn write_field_terms<F>(
    tim: &mut Vec<u8>,
    tip: &mut Vec<u8>,
    terms: &[BlockTerm<'_>],
    has_freqs: bool,
    encode_meta: F,
) -> TrieLocation
where
    F: FnMut(&mut Vec<u8>, &[usize]),
{
    assert!(!terms.is_empty(), "a field with no terms writes no blocks");
    let mut w = TermsWriter {
        terms,
        has_freqs,
        tim,
        encode_meta,
        pending: Vec::new(),
        prefix_starts: Vec::new(),
        last_term: Vec::new(),
    };
    for (i, term) in terms.iter().enumerate() {
        w.push_term(term.bytes);
        w.pending.push(PendingEntry::Term(i));
    }
    // `finish()`: two empty terms flush every open prefix, then the root.
    w.push_term(&[]);
    w.push_term(&[]);
    let all = w.pending.len();
    w.write_blocks(0, all);
    debug_assert_eq!(w.pending.len(), 1);
    let Some(PendingEntry::Block(root)) = w.pending.pop() else {
        unreachable!("write_blocks(0, all) leaves exactly the root block");
    };
    debug_assert!(root.prefix.is_empty());
    root.index
        .expect("write_blocks compiles the index of the block it leaves")
        .save(tip)
}

impl<F> TermsWriter<'_, '_, F>
where
    F: FnMut(&mut Vec<u8>, &[usize]),
{
    /// `TermsWriter.pushTerm`: before `text` joins the stack, close every
    /// prefix of the previous term that `text` does not share and that has
    /// gathered at least `MIN_ITEMS_IN_BLOCK` entries.
    // ARITH: `prefix_starts[i]` records `pending.len()` at the time prefix
    // `i` opened, and entries are only removed from above it (a
    // `write_blocks` for a longer prefix replaces entries past its own start
    // with one), so `pending.len() - prefix_starts[i]` is non-negative.
    #[allow(clippy::arithmetic_side_effects)]
    fn push_term(&mut self, text: &[u8]) {
        let prefix_length = common_prefix_len(&self.last_term, text);
        for i in (prefix_length..self.last_term.len()).rev() {
            let top = self.pending.len() - self.prefix_starts[i];
            if top >= MIN_ITEMS_IN_BLOCK {
                self.write_blocks(i + 1, top);
                // Java follows this with `prefixStarts[i] -= prefixTopSize -
                // 1`, which can go negative and is never read: every slot
                // from `prefix_length` up is either reset below for `text`
                // or lies past `text`'s length, and is reset before a later
                // term can read it.
            }
        }
        if self.prefix_starts.len() < text.len() {
            self.prefix_starts.resize(text.len(), 0);
        }
        for start in &mut self.prefix_starts[prefix_length..text.len()] {
            *start = self.pending.len();
        }
        self.last_term.clear();
        self.last_term.extend_from_slice(text);
    }

    /// The byte after `prefix_length` of an entry, or `None` for the term
    /// equal to the prefix itself.
    fn suffix_lead_label(&self, entry: &PendingEntry, prefix_length: usize) -> Option<u8> {
        match entry {
            PendingEntry::Term(i) => self.terms[*i].bytes.get(prefix_length).copied(),
            PendingEntry::Block(b) => Some(b.prefix[prefix_length]),
        }
    }

    /// `TermsWriter.writeBlocks`: writes the top `count` pending entries,
    /// all sharing `last_term[..prefix_length]`, as one block or a run of
    /// floor blocks, and replaces them on the stack with the first block.
    // ARITH: `count <= pending.len()` (it is either the whole stack or a
    // prefix's span of it), and the index loop keeps `next_block_start <= i
    // <= end`.
    #[allow(clippy::arithmetic_side_effects)]
    fn write_blocks(&mut self, prefix_length: usize, count: usize) {
        debug_assert!(count > 0);
        let end = self.pending.len();
        let start = end - count;
        let mut last_lead: Option<Option<u8>> = None;
        let mut has_terms = false;
        let mut has_sub_blocks = false;
        let mut next_block_start = start;
        let mut next_floor_lead: i32 = -1;
        let mut new_blocks: Vec<PendingBlock> = Vec::new();
        for i in start..end {
            let lead = self.suffix_lead_label(&self.pending[i], prefix_length);
            if last_lead != Some(lead) {
                let items = i - next_block_start;
                if items >= MIN_ITEMS_IN_BLOCK && end - next_block_start > MAX_ITEMS_IN_BLOCK {
                    let is_floor = items < count;
                    new_blocks.push(self.write_block(
                        prefix_length,
                        is_floor,
                        next_floor_lead,
                        next_block_start,
                        i,
                        has_terms,
                        has_sub_blocks,
                    ));
                    has_terms = false;
                    has_sub_blocks = false;
                    next_floor_lead = lead.map_or(-1, i32::from);
                    next_block_start = i;
                }
                last_lead = Some(lead);
            }
            match self.pending[i] {
                PendingEntry::Term(_) => has_terms = true,
                PendingEntry::Block(_) => has_sub_blocks = true,
            }
        }
        if next_block_start < end {
            let is_floor = end - next_block_start < count;
            new_blocks.push(self.write_block(
                prefix_length,
                is_floor,
                next_floor_lead,
                next_block_start,
                end,
                has_terms,
                has_sub_blocks,
            ));
        }
        let first = compile_index(new_blocks);
        self.pending.truncate(start);
        self.pending.push(PendingEntry::Block(first));
    }

    /// `TermsWriter.writeBlock`: writes `pending[start..end]` as one `.tim`
    /// block and returns its pending entry. The sub-block entries' tries are
    /// moved into the returned block's `sub_indices`.
    // ARITH: `end > start`; a suffix is `len - prefix_length` of an entry that
    // extends the prefix, and a sub-block was written before this block, so
    // `start_fp - block.fp > 0`; lengths are in-memory buffer sizes.
    #[allow(clippy::arithmetic_side_effects, clippy::too_many_arguments)]
    fn write_block(
        &mut self,
        prefix_length: usize,
        is_floor: bool,
        floor_lead: i32,
        start: usize,
        end: usize,
        has_terms: bool,
        has_sub_blocks: bool,
    ) -> PendingBlock {
        let start_fp = self.tim.len() as u64;
        let has_floor_lead = is_floor && floor_lead != -1;
        let mut prefix = self.last_term[..prefix_length].to_vec();
        let num_entries = end - start;
        let is_last_in_floor = end == self.pending.len();
        self.tim
            .write_vint(((num_entries as i32) << 1) | i32::from(is_last_in_floor));

        let is_leaf = !has_sub_blocks;
        let mut suffixes = Vec::new();
        let mut suffix_lengths = Vec::new();
        let mut stats = StatsWriter {
            out: Vec::new(),
            has_freqs: self.has_freqs,
            singletons: 0,
        };
        let mut term_indices = Vec::with_capacity(num_entries);
        let mut sub_indices = Vec::new();
        for entry in &mut self.pending[start..end] {
            match entry {
                PendingEntry::Term(i) => {
                    let term = &self.terms[*i];
                    let suffix = &term.bytes[prefix_length..];
                    let len = suffix.len() as i32;
                    suffix_lengths.write_vint(if is_leaf { len } else { len << 1 });
                    suffixes.extend_from_slice(suffix);
                    stats.add(term.doc_freq, term.total_term_freq);
                    term_indices.push(*i);
                }
                PendingEntry::Block(block) => {
                    debug_assert!(!is_leaf);
                    let suffix = &block.prefix[prefix_length..];
                    debug_assert!(!suffix.is_empty());
                    suffix_lengths.write_vint(((suffix.len() as i32) << 1) | 1);
                    suffixes.extend_from_slice(suffix);
                    debug_assert!(block.fp < start_fp);
                    suffix_lengths.write_vlong((start_fp - block.fp) as i64);
                    sub_indices.push(
                        block
                            .index
                            .take()
                            .expect("a pending sub-block always has its index compiled"),
                    );
                }
            }
        }
        stats.finish();

        // Suffix bytes, always `NO_COMPRESSION` (code 0): see the module doc.
        let token = ((suffixes.len() as u64) << 3) | if is_leaf { 0x04 } else { 0 };
        self.tim.write_vlong(token as i64);
        self.tim.write_bytes(&suffixes);

        let n = suffix_lengths.len();
        if suffix_lengths[1..].iter().all(|&b| b == suffix_lengths[0]) {
            self.tim.write_vint(((n as i32) << 1) | 1);
            self.tim.push(suffix_lengths[0]);
        } else {
            self.tim.write_vint((n as i32) << 1);
            self.tim.write_bytes(&suffix_lengths);
        }

        self.tim.write_vint(stats.out.len() as i32);
        self.tim.write_bytes(&stats.out);

        let mut meta = Vec::new();
        (self.encode_meta)(&mut meta, &term_indices);
        self.tim.write_vint(meta.len() as i32);
        self.tim.write_bytes(&meta);

        if has_floor_lead {
            prefix.push(floor_lead as u8);
        }
        PendingBlock {
            prefix,
            fp: start_fp,
            has_terms,
            is_floor,
            floor_lead_byte: floor_lead,
            index: None,
            sub_indices,
        }
    }
}

/// `PendingBlock.compileIndex`: builds the trie for a run of blocks written
/// for one prefix (a single block, or a floor run) and returns the first
/// block carrying it.
// ARITH: follow-on floor blocks are written after the first, so their fp
// deltas are positive; the count is at most the handful of blocks one
// `write_blocks` call produces.
#[allow(clippy::arithmetic_side_effects)]
fn compile_index(blocks: Vec<PendingBlock>) -> PendingBlock {
    let mut blocks = blocks.into_iter();
    let mut first = blocks
        .next()
        .expect("write_blocks always writes at least one block");
    let rest: Vec<PendingBlock> = blocks.collect();
    debug_assert_eq!(first.is_floor, !rest.is_empty());
    let floor_data = first.is_floor.then(|| {
        let mut data = Vec::new();
        data.write_vint(rest.len() as i32);
        for sub in &rest {
            debug_assert!(sub.floor_lead_byte != -1 && sub.fp > first.fp);
            data.push(sub.floor_lead_byte as u8);
            data.write_vlong((((sub.fp - first.fp) << 1) | u64::from(sub.has_terms)) as i64);
        }
        data
    });
    let mut trie = TrieBuilder::new(
        &first.prefix,
        TrieOutput {
            fp: first.fp,
            has_terms: first.has_terms,
            floor_data,
        },
    );
    for sub in std::mem::take(&mut first.sub_indices) {
        trie.append(sub);
    }
    for block in rest {
        for sub in block.sub_indices {
            trie.append(sub);
        }
    }
    first.index = Some(trie);
    first
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn out(fp: u64) -> TrieOutput {
        TrieOutput {
            fp,
            has_terms: true,
            floor_data: None,
        }
    }

    #[test]
    fn bytes_required_matches_java() {
        assert_eq!(bytes_required(0), 1);
        assert_eq!(bytes_required(255), 1);
        assert_eq!(bytes_required(256), 2);
        assert_eq!(bytes_required(u64::MAX), 8);
    }

    #[test]
    fn strategy_choice_prefers_bits_then_array_then_reverse() {
        // Dense run a..h: BITS 1 byte, ARRAY 7, REVERSE 1 -> BITS (tie, earlier).
        assert_eq!(choose_strategy(97, 104, 8), (CHILD_STRATEGY_BITS, 1));
        // Two labels far apart: BITS 26, ARRAY 1, REVERSE 200 -> ARRAY.
        assert_eq!(choose_strategy(0, 200, 2), (CHILD_STRATEGY_ARRAY, 1));
        // 0..=100 with one gap: BITS 13, ARRAY 99, REVERSE 2 -> REVERSE_ARRAY.
        assert_eq!(
            choose_strategy(0, 100, 100),
            (CHILD_STRATEGY_REVERSE_ARRAY, 2)
        );
    }

    #[test]
    fn strategies_write_their_computed_length() {
        let cases: Vec<Vec<u8>> = vec![
            (b'a'..=b'h').collect(),
            vec![0, 200],
            (0..=100).filter(|&l| l != 50).collect(),
            vec![1, 9, 17, 64, 65, 255],
        ];
        for labels in cases {
            let children: Vec<(u8, u64)> = labels.iter().map(|&l| (l, 0)).collect();
            let (min, max) = (u32::from(labels[0]), u32::from(*labels.last().unwrap()));
            for strategy in [
                CHILD_STRATEGY_BITS,
                CHILD_STRATEGY_ARRAY,
                CHILD_STRATEGY_REVERSE_ARRAY,
            ] {
                let mut tip = Vec::new();
                save_strategy(strategy, &children, &mut tip);
                assert_eq!(
                    tip.len() as u32,
                    strategy_bytes(strategy, min, max, labels.len() as u32),
                    "strategy {strategy} labels {labels:?}"
                );
            }
        }
    }

    #[test]
    fn reverse_array_lists_max_then_the_gaps() {
        let children: Vec<(u8, u64)> = [1u8, 2, 4, 6, 7, 8, 9, 10]
            .iter()
            .map(|&l| (l, 0))
            .collect();
        let mut tip = Vec::new();
        save_strategy(CHILD_STRATEGY_REVERSE_ARRAY, &children, &mut tip);
        assert_eq!(tip, vec![10, 3, 5]);
    }

    #[test]
    fn bits_marks_each_present_label() {
        let children: Vec<(u8, u64)> = [3u8, 4, 12].iter().map(|&l| (l, 0)).collect();
        let mut tip = Vec::new();
        save_strategy(CHILD_STRATEGY_BITS, &children, &mut tip);
        // Offsets 0, 1 and 9 from label 3.
        assert_eq!(tip, vec![0b0000_0011, 0b0000_0010]);
    }

    #[test]
    fn single_leaf_trie_is_one_no_children_node_plus_pad() {
        let trie = TrieBuilder::new(b"", out(0x1234));
        let mut tip = vec![0xAA; 5]; // a header already in the file
        let loc = trie.save(&mut tip);
        assert_eq!(loc.index_start, 5);
        assert_eq!(loc.root_fp, 0);
        assert_eq!(
            &tip[5..],
            &[
                (SIGN_NO_CHILDREN | (1 << 2) | LEAF_NODE_HAS_TERMS) as u8,
                0x34,
                0x12,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0
            ]
        );
        assert_eq!(loc.index_end, tip.len() as u64);
    }

    #[test]
    fn append_keeps_the_first_empty_output() {
        let mut a = TrieBuilder::new(b"", out(1));
        a.append(TrieBuilder::new(b"", out(2)));
        a.append(TrieBuilder::new(b"b", out(3)));
        assert_eq!(a.empty_output, Some(out(1)));
        assert_eq!(a.entries.len(), 1);
        let mut b = TrieBuilder::new(b"a", out(1));
        b.append(TrieBuilder::new(b"", out(9)));
        assert_eq!(b.empty_output, Some(out(9)));
    }

    #[test]
    fn stats_writer_run_length_encodes_singletons() {
        let mut s = StatsWriter {
            out: Vec::new(),
            has_freqs: true,
            singletons: 0,
        };
        s.add(1, 1);
        s.add(1, 1);
        s.add(1, 3); // freq 3: not a singleton when freqs are indexed
        s.add(1, 1);
        s.finish();
        assert_eq!(s.out, vec![(1 << 1) | 1, 1 << 1, 2, 1]);
        let mut docs_only = StatsWriter {
            out: Vec::new(),
            has_freqs: false,
            singletons: 0,
        };
        docs_only.add(1, 7);
        docs_only.add(5, 5);
        docs_only.finish();
        assert_eq!(docs_only.out, vec![1, 10]);
    }
}
