//! Port of `org.apache.lucene.codecs.lucene103.blocktree.Lucene103BlockTreeTermsReader`
//! (`.tim` term dictionary + `.tip` term index + `.tmd` per-field metadata) —
//! read-only, scoped to **`seekExact` + `docFreq`/`totalTermFreq`** only.
//!
//! Note on naming: the pinned Lucene version (10.5.0) uses
//! `Lucene104PostingsFormat`, whose term dictionary is
//! `Lucene103BlockTreeTermsReader`/`Writer` (package
//! `o.a.l.codecs.lucene103.blocktree`) — *not* the `lucene90.blocktree`
//! classes, which live in `backward-codecs` and are out of scope for this
//! port (see PLAN.md's "pin one Lucene version" rule). The `.tip` term index
//! in this version is **not an FST** — Lucene 10.x replaced it with a
//! purpose-built binary trie (`TrieReader`/`TrieBuilder`), a flatter,
//! pointer-chasing encoding of the same "prefix trie whose leaves are term
//! blocks" idea `fst.rs`'s module doc describes for the *older* format.
//! `fst.rs` remains useful groundwork (arc-lookup style reasoning, shared
//! `codec_util` header handling) but is not used by this module.
//!
//! ## Wire format
//!
//! - `.tmd` (`TERMS_META_EXTENSION`): `IndexHeader(codec="BlockTreeTermsMeta")`,
//!   then the postings reader's own `init` header
//!   (`IndexHeader(codec="Lucene104PostingsWriterTerms")` + `indexBlockSize: vint`,
//!   which must equal `Lucene104PostingsFormat.BLOCK_SIZE` = 256 for this pinned
//!   version), then `numFields: vint`, then per field: `fieldNumber: vint`,
//!   `numTerms: vlong`, a `sumTotalTermFreq`/`sumDocFreq` pair (see
//!   [`read_freq_pair`] for the DOCS-only aliasing trick), `docCount: vint`,
//!   `minTerm`/`maxTerm` (vint-length-prefixed byte arrays), and finally
//!   `indexStart`/`rootFP`/`indexEnd` (three vlongs locating this field's root
//!   node in `.tip`). After the field loop: `indexLength: i64`, `termsLength: i64`,
//!   `Footer`.
//! - `.tip` (`TERMS_INDEX_EXTENSION`): `IndexHeader(codec="BlockTreeTermsIndex")`,
//!   then every field's trie nodes packed back to back (each field's node
//!   region spans `[indexStart, indexEnd)` from its `.tmd` record), `Footer`.
//!   A trie node's header byte packs a 2-bit `sign` selecting one of three
//!   encodings (`SIGN_NO_CHILDREN`/`SIGN_SINGLE_CHILD_*`/`SIGN_MULTI_CHILDREN`);
//!   see `TrieReader.java`/`TrieBuilder.java` for the full byte-packing scheme
//!   ([`load_node`] is a direct transliteration of `TrieReader.load`/
//!   `loadLeafNode`/`loadSingleChildNode`/`loadMultiChildrenNode`, and
//!   [`multi_children_labels_and_fps`] of `lookupChild`'s three `ChildSaveStrategy`
//!   decodings, generalized to enumerate *every* child rather than looking up
//!   one label at a time — see below for why).
//! - `.tim` (`TERMS_EXTENSION`): `IndexHeader(codec="BlockTreeTermsDict")`, then
//!   every field's blocks packed back to back (see [`decode_block`]), `Footer`.
//!
//! ## Scope of this slice
//!
//! Ported: opening a `.tim`/`.tip`/`.tmd` triple, per-field metadata, and
//! `seekExact`-equivalent term lookup with `docFreq`/`totalTermFreq`
//! readback, now covering **multi-child trie nodes and floor blocks** — i.e.
//! a field whose term dictionary spans more than one `.tim` block, whether
//! because a prefix's terms were split into floor sub-blocks (too many items
//! sharing one prefix, `LEAF_NODE_HAS_FLOOR`/`NON_LEAF_NODE_HAS_FLOOR`) or
//! because the trie root has children (`SIGN_SINGLE_CHILD_WITH_OUTPUT`/
//! `SIGN_SINGLE_CHILD_WITHOUT_OUTPUT`/`SIGN_MULTI_CHILDREN`, all three
//! `ChildSaveStrategy` encodings — `BITS`/`ARRAY`/`REVERSE_ARRAY`).
//!
//! **Design choice: eager whole-field materialization, not a single
//! root-to-leaf trie walk.** Real `SegmentTermsEnum.seekExact` walks the trie
//! one label at a time along the target term's own bytes, touching only the
//! one leaf block (and, within it, the one floor sub-block) that can contain
//! the term. This module instead recursively visits **every** reachable trie
//! node ([`collect_leaf_blocks`]), resolves every floor sub-block
//! ([`expand_floor`]) at every node that has output, decodes every resulting
//! `.tim` block, and merges all of a field's entries into one sorted `Vec` —
//! the same shape the prior (single-block) slice already used, just now fed
//! by a full trie traversal instead of one root-node read. This keeps
//! [`FieldTerms`] and its `seek_exact`/`postings`/`positions` API completely
//! unchanged (no caller-visible difference between a one-block and a
//! thousand-block field) and sidesteps a subtlety this port doesn't need to
//! solve yet: an internal trie node's *own* output block and its children's
//! blocks are **not** necessarily contiguous in sort order purely from
//! traversal order (a node's own block can hold terms interleaved in depth
//! with what its children cover), so collect-then-sort is simpler and
//! provably correct where a hand-rolled single-path merge would need much
//! more care to get right. The tradeoff is real: this eagerly decodes blocks
//! a real `seekExact` for one term would never touch. That's an acceptable
//! cost for this slice (no enumeration/streaming consumer exists yet to
//! notice; `rust-performance`'s "correctness first, profile before the next
//! phase" stance applies) and is flagged here rather than silently accepted.
//!
//! **Multi-level blocktree tries (`.tim` blocks that are themselves
//! non-leaf) are now decoded.** A `.tim` block can be `isLeafBlock == false`:
//! some of its entries are pointers to further-nested sub-blocks (an
//! in-block delta-fp, `SegmentTermsEnumFrame.nextNonLeaf`'s `code & 1`
//! "is this a sub-block" bit) rather than raw term suffixes — real Lucene's
//! own mechanism for a prefix so wide it isn't worth giving every one of its
//! sub-prefixes a separate `.tip` trie/index entry (distinct from *both* the
//! `.tip` trie's own multi-level node nesting -- root/single-child/multi-children,
//! already arbitrarily deep and covered by [`collect_leaf_blocks`] -- and
//! floor blocks, see [`expand_floor`]). [`decode_block`] now recurses into
//! every sub-block entry it finds (reattaching that entry's own key bytes as
//! a prefix before merging its sub-entries in), so a field whose dictionary
//! needed a genuinely deep block tree (root block -> internal block -> leaf
//! block, not just root -> leaf) round-trips correctly. The real
//! `Lucene103BlockTreeTermsWriter`-produced fixture (~8k terms, see
//! `crates/lucene-codecs/tests/blocktree_multilevel_fixture.rs`) does contain
//! a genuine non-leaf `.tim` block, proving that SHAPE is decoded without
//! error -- but its one sub-block pointer happens to also be independently
//! reachable via the `.tip` trie, so the dedup check (below) skips it there
//! and the *recursive re-prefixing* code path itself is only actually
//! exercised, not merely shape-checked, by the hand-built unit test
//! [`decode_block_recurses_into_sub_block`]. A future fixture engineered so a
//! sub-block pointer is the *only* path to some terms (not also trie-indexed)
//! would close that gap with a real differential proof; flagged here rather
//! than silently overclaimed.
//! **Ordered enumeration (`next()`) and
//! nearest-match seeking (`seekCeil()`) are now ported** — see
//! [`TermsEnum`]/[`FieldTerms::iter`] — as a thin cursor over the
//! already-sorted `entries` `Vec` rather than a reimplementation of
//! `SegmentTermsEnum`'s lazy block-walking machinery (see `TermsEnum`'s own
//! doc comment for the full rationale). **Suffix compression is now decoded**:
//! `CompressionAlgorithm::LZ4` (reusing `crate::lz4::decompress`) and
//! `LowercaseAscii` (a small standalone port of
//! `LowercaseAsciiCompression.decompress`, see `decompress_lowercase_ascii`)
//! are both handled read-side in [`decode_block`], alongside the original
//! `NO_COMPRESSION` path (unchanged). This port's own blocktree *writer*
//! (see the writer section below) still only ever emits `NO_COMPRESSION` —
//! this is purely a read-side feature for interoperating with real
//! Lucene-written segments whose blocks happened to compress. Only code `3`
//! (never assigned to a `CompressionAlgorithm` constant) is rejected, as
//! `Error::Store(Corrupted)`, matching `CompressionAlgorithm.byCode`'s own
//! `IllegalArgumentException`.
//!
//! Because this slice never decodes postings inline with block loading, the
//! per-term metadata bytes written by the postings writer (doc/pos/pay file
//! pointer deltas) are decoded per block via `crate::postings::decode_term_metadata`
//! (threaded across each individual block's own *term* entries -- sub-block
//! entries carry no metadata of their own, see [`decode_block`] -- `absolute`
//! true only for each block's first term entry — blocks never share metadata
//! state, matching `SegmentTermsEnumFrame`'s per-frame `metaDataUpto`/`absolute`
//! reset).

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

use crate::field_infos::{FieldInfos, IndexOptions};
use crate::fuzzy::FuzzyMatch;
use crate::postings::{self, DocInput, Postings, TermMetadata};
use crate::regexp::RegexpPattern;
use crate::wildcard::WildcardPattern;

pub(crate) const TERMS_CODEC_NAME: &str = "BlockTreeTermsDict";
pub(crate) const TERMS_INDEX_CODEC_NAME: &str = "BlockTreeTermsIndex";
pub(crate) const TERMS_META_CODEC_NAME: &str = "BlockTreeTermsMeta";
const VERSION_START: i32 = 0;
pub(crate) const VERSION_CURRENT: i32 = 0;

/// `Lucene104PostingsFormat.TERMS_CODEC` — the postings writer's own header,
/// embedded in the `.tmd` stream right after BlockTree's own index header.
pub(crate) const POSTINGS_TERMS_CODEC: &str = "Lucene104PostingsWriterTerms";
const POSTINGS_VERSION_START: i32 = 0;
pub(crate) const POSTINGS_VERSION_CURRENT: i32 = 0;
/// `Lucene104PostingsFormat.BLOCK_SIZE` (= `ForUtil.BLOCK_SIZE`), the postings
/// block size the `.tmd` stream's `indexBlockSize` field must match.
pub(crate) const POSTINGS_BLOCK_SIZE: i32 = 256;

/// `TrieBuilder.SIGN_NO_CHILDREN` — a leaf trie node (no children).
pub(crate) const SIGN_NO_CHILDREN: u32 = 0x00;
/// `TrieBuilder.SIGN_SINGLE_CHILD_WITH_OUTPUT`.
const SIGN_SINGLE_CHILD_WITH_OUTPUT: u32 = 0x01;
/// `TrieBuilder.SIGN_SINGLE_CHILD_WITHOUT_OUTPUT`.
const SIGN_SINGLE_CHILD_WITHOUT_OUTPUT: u32 = 0x02;
/// `TrieBuilder.SIGN_MULTI_CHILDREN`.
pub(crate) const SIGN_MULTI_CHILDREN: u32 = 0x03;
/// `TrieBuilder.LEAF_NODE_HAS_TERMS` (`1 << 5`).
pub(crate) const LEAF_NODE_HAS_TERMS: u32 = 1 << 5;
/// `TrieBuilder.LEAF_NODE_HAS_FLOOR` (`1 << 6`).
const LEAF_NODE_HAS_FLOOR: u32 = 1 << 6;
/// `TrieBuilder.NON_LEAF_NODE_HAS_TERMS` (`1L << 1`) — the equivalent flag
/// packed into a non-leaf node's *encoded output fp* (`encodeFP`), not its
/// header byte, since non-leaf nodes' header bits are all spoken for by
/// child-pointer bookkeeping.
const NON_LEAF_NODE_HAS_TERMS: u64 = 1 << 1;
/// `TrieBuilder.NON_LEAF_NODE_HAS_FLOOR` (`1L << 0`).
const NON_LEAF_NODE_HAS_FLOOR: u64 = 1;
/// `TrieBuilder.ChildSaveStrategy.REVERSE_ARRAY.code`.
const CHILD_STRATEGY_REVERSE_ARRAY: u32 = 0;
/// `TrieBuilder.ChildSaveStrategy.ARRAY.code`.
pub(crate) const CHILD_STRATEGY_ARRAY: u32 = 1;
/// `TrieBuilder.ChildSaveStrategy.BITS.code`.
const CHILD_STRATEGY_BITS: u32 = 2;

const BYTES_MINUS_1_MASK: [u64; 8] = [
    0xFF,
    0xFFFF,
    0xFF_FFFF,
    0xFFFF_FFFF,
    0xFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF,
    0xFF_FFFF_FFFF_FFFF,
    0xFFFF_FFFF_FFFF_FFFF,
];

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error(transparent)]
    FieldInfos(#[from] crate::field_infos::Error),
    #[error("invalid numFields: {0}")]
    InvalidNumFields(i32),
    #[error("invalid field number: {0}")]
    InvalidFieldNumber(i32),
    #[error("illegal numTerms for field number: {0}")]
    IllegalNumTerms(i32),
    #[error("invalid docCount: {doc_count} maxDoc: {max_doc}")]
    InvalidDocCount { doc_count: i32, max_doc: i32 },
    #[error("invalid sumDocFreq: {sum_doc_freq} docCount: {doc_count}")]
    InvalidSumDocFreq { sum_doc_freq: i64, doc_count: i32 },
    #[error("invalid sumTotalTermFreq: {sum_total_term_freq} sumDocFreq: {sum_doc_freq}")]
    InvalidSumTotalTermFreq {
        sum_total_term_freq: i64,
        sum_doc_freq: i64,
    },
    #[error("duplicate field: {0}")]
    DuplicateField(String),
    #[error(
        "index-time postings BLOCK_SIZE ({found}) != read-time BLOCK_SIZE ({POSTINGS_BLOCK_SIZE})"
    )]
    UnexpectedBlockSize { found: i32 },
    #[error(transparent)]
    Postings(#[from] postings::Error),
    #[error("unsupported: {0}")]
    Unsupported(&'static str),
}

pub type Result<T> = std::result::Result<T, Error>;

/// `docFreq`/`totalTermFreq` for one found term — the entirety of what this
/// slice can read back for a term (no postings/doc-ids).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TermStats {
    pub doc_freq: i32,
    pub total_term_freq: i64,
}

/// `TermsEnum.SeekStatus`-equivalent: the outcome of [`TermsEnum::seek_ceil`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SeekStatus {
    /// The target term itself was present.
    Found,
    /// The target term was absent; the enum is positioned on the smallest
    /// term greater than the target.
    NotFound,
    /// No term in the field is >= the target; the enum is positioned past
    /// the last term (a following [`TermsEnum::next`] returns `None`).
    End,
}

/// `TermsEnum`-equivalent: ordered iteration (`next()`) and nearest-match
/// seeking (`seekCeil()`) over one field's already-sorted term dictionary.
///
/// **Design choice**: this is a thin cursor over [`FieldTerms`]'s existing
/// sorted `entries` `Vec`, not a reimplementation of real Lucene's
/// `SegmentTermsEnum`/`SegmentTermsEnumFrame` stack-based lazy block walk.
/// The prior slice already made the eager-whole-field-materialization
/// tradeoff (see the module doc) — every term is already decoded into one
/// sorted `Vec` before any lookup happens, so `next()` is an index bump and
/// `seekCeil()` is a binary search, both O(log n) or O(1), with none of
/// Java's per-frame `FST`/trie-path push/pop state to reconstruct. Building
/// the Java-shaped stack machinery on top of an already-fully-materialized
/// Vec would only reintroduce complexity this port deliberately avoided
/// (see the `rust-performance` skill: "not a dumb port" — a redesign around
/// what's actually needed beats transliterating `SegmentTermsEnumFrame`'s
/// internals when the underlying representation no longer matches Java's).
#[derive(Debug, Clone)]
pub struct TermsEnum<'a> {
    entries: &'a [(Vec<u8>, TermStats, TermMetadata)],
    /// Index of the last term returned by `next()`/positioned by
    /// `seek_ceil()`, or `None` before the first `next()` call. Once
    /// exhausted this holds `Some(entries.len())` (or higher), so repeated
    /// `next()` calls after the end keep returning `None` without special
    /// casing.
    pos: Option<usize>,
}

impl<'a> TermsEnum<'a> {
    fn new(entries: &'a [(Vec<u8>, TermStats, TermMetadata)]) -> Self {
        Self { entries, pos: None }
    }

    /// `TermsEnum.next()`-equivalent: advance to (and return) the next term
    /// in sorted order, or `None` at end-of-terms (matching `BytesRef
    /// next()` returning `null`). Idempotent past the end.
    ///
    /// Named to mirror Java's `TermsEnum.next()` rather than `std::iter::Iterator::next`
    /// on purpose: a real `std::iter::Iterator` impl would need `Item` to
    /// borrow from `self` (a `(term, stats)` pair tied to `'a`, not to each
    /// `next()` call), which isn't expressible through that trait — this is
    /// deliberately its own cursor API, same shape as Java's, not an
    /// `Iterator`.
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Option<(&'a [u8], TermStats)> {
        let next_idx = self.pos.map_or(0, |p| p + 1);
        if next_idx >= self.entries.len() {
            self.pos = Some(self.entries.len());
            return None;
        }
        self.pos = Some(next_idx);
        let (term, stats, _) = &self.entries[next_idx];
        Some((term.as_slice(), *stats))
    }

    /// `TermsEnum.seekCeil(BytesRef)`-equivalent: binary-search for the
    /// smallest term >= `target`, position the cursor there (so a following
    /// `next()` continues from that point), and report whether it was an
    /// exact match, a ceiling match, or that no such term exists.
    pub fn seek_ceil(&mut self, target: &[u8]) -> SeekStatus {
        match self
            .entries
            .binary_search_by(|(t, _, _)| t.as_slice().cmp(target))
        {
            Ok(idx) => {
                self.pos = Some(idx);
                SeekStatus::Found
            }
            Err(idx) if idx >= self.entries.len() => {
                self.pos = Some(self.entries.len());
                SeekStatus::End
            }
            Err(idx) => {
                self.pos = Some(idx);
                SeekStatus::NotFound
            }
        }
    }

    /// The term/stats the cursor is currently positioned on (the last term
    /// returned by `next()`, or the term `seek_ceil()` landed on) — `None`
    /// before the first `next()`/`seek_ceil()` call or past the end.
    pub fn current(&self) -> Option<(&'a [u8], TermStats)> {
        let idx = self.pos?;
        self.entries.get(idx).map(|(t, s, _)| (t.as_slice(), *s))
    }
}

/// One field's decoded term dictionary: every (term, stats) pair in the
/// field's single `.tim` block, sorted (as the writer emits them), plus the
/// field-level aggregate stats from `.tmd`.
#[derive(Debug, Clone)]
pub struct FieldTerms {
    pub num_terms: i64,
    pub sum_total_term_freq: i64,
    pub sum_doc_freq: i64,
    pub doc_count: i32,
    pub min_term: Vec<u8>,
    pub max_term: Vec<u8>,
    index_options: IndexOptions,
    has_payloads: bool,
    entries: Vec<(Vec<u8>, TermStats, TermMetadata)>,
}

impl FieldTerms {
    /// `TermsEnum.seekExact(BytesRef)`-equivalent: exact lookup only, no
    /// enumeration/range-seeking. Terms are stored sorted, so this is a
    /// binary search over the materialized block.
    pub fn seek_exact(&self, term: &[u8]) -> Option<TermStats> {
        self.entries
            .binary_search_by(|(t, _, _)| t.as_slice().cmp(term))
            .ok()
            .map(|idx| self.entries[idx].1)
    }

    /// `Terms.iterator()`-equivalent: a cursor positioned before the first
    /// term, ready for `TermsEnum::next()`/`seek_ceil()`.
    pub fn iter(&self) -> TermsEnum<'_> {
        TermsEnum::new(&self.entries)
    }

    /// `Terms.intersect(CompiledAutomaton, BytesRef)`-equivalent: every term
    /// (in sorted order) that matches `pattern`, paired with its stats.
    ///
    /// **Design** (see `crate::wildcard`'s module doc for the full
    /// tradeoff): real Lucene's `IntersectTermsEnum` walks only the trie
    /// nodes/blocks a compiled automaton's reachable states can lead to,
    /// potentially touching a small fraction of a huge dictionary. This
    /// method instead narrows the already-sorted `entries` `Vec` to the
    /// contiguous range starting with the pattern's literal prefix (a plain
    /// binary search — free, since `entries` is already sorted for
    /// `seek_exact`) and then tests every term in that range against the
    /// pattern with a linear scan. For a pattern with no literal prefix
    /// (e.g. `*foo`) the "range" is the entire field, so this degrades to a
    /// full `O(n)` scan — correct, but not sub-linear the way a real
    /// automaton intersection would be. This is an honest, explicitly scoped
    /// first cut: no `CompiledAutomaton`/`ByteRunAutomaton`/`IntersectTermsEnum`
    /// block-skipping is implemented, matching this port's "correctness
    /// first, real optimization later, be honest about what's optimized"
    /// stance from the postings work (see `docs/parity.md`).
    pub fn intersect<'a>(
        &'a self,
        pattern: &'a WildcardPattern,
    ) -> impl Iterator<Item = (&'a [u8], TermStats)> + 'a {
        let prefix = pattern.literal_prefix();
        // The range of the sorted `Vec` whose *own* leading bytes could
        // possibly equal `prefix`: the first index at which `term >=
        // prefix` through the first index at which `term` no longer starts
        // with `prefix` (found by bumping `prefix`'s last byte, the
        // standard "prefix range via two binary searches" trick — matches
        // what real Lucene's own prefix-seek does one level below the
        // automaton, just expressed here as two `partition_point`s over
        // the materialized `Vec` instead of a trie walk).
        let start = self
            .entries
            .partition_point(|(t, _, _)| t.as_slice() < prefix.as_slice());
        let end = match prefix_upper_bound(&prefix) {
            Some(upper) => self
                .entries
                .partition_point(|(t, _, _)| t.as_slice() < upper.as_slice()),
            None => self.entries.len(),
        };
        self.entries[start..end]
            .iter()
            .filter(move |(t, _, _)| pattern.matches(t))
            .map(|(t, s, _)| (t.as_slice(), *s))
    }

    /// `FuzzyQuery`-equivalent term matching (task #42): every term (in
    /// sorted order) within `pattern`'s edit-distance budget, paired with its
    /// stats. Structurally identical to [`Self::intersect`]'s "narrow by
    /// literal prefix range via binary search, then linearly filter" design —
    /// here the literal prefix is `pattern`'s required `prefixLength`-byte
    /// exact prefix ([`crate::fuzzy::FuzzyMatch::literal_prefix`]) rather than
    /// a glob pattern's leading literal run, and the per-candidate filter is
    /// [`crate::fuzzy::FuzzyMatch::matches`]'s edit-distance test rather than
    /// glob matching. See `crate::fuzzy`'s module doc for the full
    /// automaton-vs-DP tradeoff writeup.
    pub fn fuzzy_intersect<'a>(
        &'a self,
        pattern: &'a FuzzyMatch<'a>,
    ) -> impl Iterator<Item = (&'a [u8], TermStats)> + 'a {
        let prefix = pattern.literal_prefix();
        let start = self
            .entries
            .partition_point(|(t, _, _)| t.as_slice() < prefix);
        let end = match prefix_upper_bound(prefix) {
            Some(upper) => self
                .entries
                .partition_point(|(t, _, _)| t.as_slice() < upper.as_slice()),
            None => self.entries.len(),
        };
        self.entries[start..end]
            .iter()
            .filter(move |(t, _, _)| pattern.matches(t))
            .map(|(t, s, _)| (t.as_slice(), *s))
    }

    /// `RegexpQuery`-equivalent term matching (task #43): every term (in
    /// sorted order) [`crate::regexp::RegexpPattern::matches`] accepts,
    /// paired with its stats. Structurally identical to [`Self::intersect`]/
    /// [`Self::fuzzy_intersect`]'s "narrow by literal prefix range via
    /// binary search, then linearly filter" design -- here the literal
    /// prefix is `pattern`'s guaranteed leading literal byte run
    /// ([`crate::regexp::RegexpPattern::literal_prefix`]) rather than a glob
    /// pattern's own, and the per-candidate filter is
    /// [`crate::regexp::RegexpPattern::matches`]'s whole-term backtracking
    /// match rather than glob matching. See `crate::regexp`'s module doc for
    /// the full syntax-subset and automaton-vs-backtracking tradeoff
    /// writeup.
    pub fn regexp_intersect<'a>(
        &'a self,
        pattern: &'a RegexpPattern,
    ) -> impl Iterator<Item = (&'a [u8], TermStats)> + 'a {
        let prefix = pattern.literal_prefix();
        let start = self
            .entries
            .partition_point(|(t, _, _)| t.as_slice() < prefix.as_slice());
        let end = match prefix_upper_bound(&prefix) {
            Some(upper) => self
                .entries
                .partition_point(|(t, _, _)| t.as_slice() < upper.as_slice()),
            None => self.entries.len(),
        };
        self.entries[start..end]
            .iter()
            .filter(move |(t, _, _)| pattern.matches(t))
            .map(|(t, s, _)| (t.as_slice(), *s))
    }

    /// `seekExact(term)` followed by `PostingsEnum` iteration
    /// (`postingsReader.postings(...)`, `DOCS_AND_FREQS` mode) — decodes the
    /// term's actual `(docID, freq)` pairs, scoped to a single postings block
    /// (see `crate::postings`'s module doc for exactly what that covers).
    /// `doc_in` is `None` for fields where a `.doc` file was never opened
    /// (e.g. no indexed field in the segment needs it) — passing `None` for a
    /// found term whose `docFreq > 1` is an error, since that path needs
    /// `.doc` file bytes.
    pub fn postings(&self, term: &[u8], doc_in: Option<&DocInput<'_>>) -> Result<Option<Postings>> {
        let Some(idx) = self
            .entries
            .binary_search_by(|(t, _, _)| t.as_slice().cmp(term))
            .ok()
        else {
            return Ok(None);
        };
        let (_, stats, meta) = &self.entries[idx];
        if stats.doc_freq == 1 {
            return Ok(Some(postings::singleton_postings(
                *meta,
                stats.total_term_freq,
            )?));
        }
        let doc_in = doc_in.ok_or(Error::Unsupported(
            "postings() needs an opened .doc file for docFreq > 1 terms",
        ))?;
        Ok(Some(doc_in.read_postings(
            *meta,
            stats.doc_freq,
            self.index_options,
            self.has_payloads,
        )?))
    }

    /// `seekExact(term)` followed by opening a [`postings::LazyDocsCursor`]:
    /// the decode-on-demand sibling of [`Self::postings`] (see that method
    /// and `crate::postings`'s module doc for the shared scope/validation —
    /// `docFreq <= 1`, `DocsAndCustomFreqs`, `docFreq >= LEVEL1_NUM_DOCS` are
    /// all rejected identically). Unlike `postings()`, this never decodes any
    /// `.doc` bytes until the cursor's `next_doc()`/`advance()` is actually
    /// called, and `advance()` can skip whole undecoded blocks — see
    /// `LazyDocsCursor`'s own doc comment.
    pub fn lazy_postings<'d>(
        &self,
        term: &[u8],
        doc_in: &DocInput<'d>,
    ) -> Result<Option<postings::LazyDocsCursor<'d>>> {
        let Some(idx) = self
            .entries
            .binary_search_by(|(t, _, _)| t.as_slice().cmp(term))
            .ok()
        else {
            return Ok(None);
        };
        let (_, stats, meta) = &self.entries[idx];
        Ok(Some(doc_in.lazy_cursor(
            *meta,
            stats.doc_freq,
            self.index_options,
            self.has_payloads,
        )?))
    }

    /// `postings(term, doc_in)` followed by `PostingsEnum.nextPosition()`/
    /// `startOffset()`/`endOffset()`/`getPayload()` for every occurrence in
    /// every doc — needs a field with `IndexOptions::DocsAndFreqsAndPositions`
    /// or higher (see `crate::postings::read_positions`'s doc comment for the
    /// exact scope). `pay_in` is only needed for a field with offsets or
    /// payloads whose `total_term_freq` spans at least one full 256-position
    /// block; `None` is otherwise fine even for such a field.
    pub fn positions(
        &self,
        term: &[u8],
        doc_in: Option<&DocInput<'_>>,
        pos_in: &postings::PosInput<'_>,
        pay_in: Option<&postings::PayInput<'_>>,
    ) -> Result<Option<Vec<Vec<postings::Position>>>> {
        let Some(doc_postings) = self.postings(term, doc_in)? else {
            return Ok(None);
        };
        let idx = self
            .entries
            .binary_search_by(|(t, _, _)| t.as_slice().cmp(term))
            .expect("found by self.postings() above, so seek_exact must succeed here too");
        let (_, stats, meta) = &self.entries[idx];
        Ok(Some(postings::read_positions(
            pos_in,
            pay_in,
            *meta,
            &doc_postings.freqs,
            stats.total_term_freq,
            self.index_options,
            self.has_payloads,
        )?))
    }
}

/// All fields' term dictionaries for one segment, keyed by field name.
#[derive(Debug, Clone, Default)]
pub struct BlockTreeFields {
    fields: Vec<(String, FieldTerms)>,
}

impl BlockTreeFields {
    pub fn field(&self, name: &str) -> Option<&FieldTerms> {
        self.fields.iter().find(|(n, _)| n == name).map(|(_, f)| f)
    }

    /// Every field's name paired with its term dictionary, in the order
    /// `.tmd` listed them -- used by callers (e.g. `CheckIndex`-equivalent's
    /// postings re-derivation) that need to walk *every* field's *every*
    /// term rather than looking one up by name via [`Self::field`].
    pub fn iter_fields(&self) -> impl Iterator<Item = (&str, &FieldTerms)> {
        self.fields.iter().map(|(n, f)| (n.as_str(), f))
    }

    /// A fields producer for a segment with no postings at all (no
    /// `.tim`/`.tip`/`.tmd` files) -- e.g. a stored-fields-only segment,
    /// where `FieldInfos.hasPostings()` is false for every field. Every
    /// lookup on this behaves exactly like a real segment whose term
    /// dictionary happens to be empty.
    pub fn empty() -> Self {
        BlockTreeFields { fields: Vec::new() }
    }
}

/// The exclusive upper bound of the sorted range whose bytes all start with
/// `prefix`: `prefix` with its last byte incremented (dropping any trailing
/// `0xFF` bytes first, since those can't be incremented in place — e.g.
/// `[0x61, 0xFF]` -> `[0x62]`). `None` when `prefix` is empty (no useful
/// bound — the whole `Vec` is the range) or entirely `0xFF` bytes (no finite
/// byte string is an upper bound; every real term is such a bound already).
fn prefix_upper_bound(prefix: &[u8]) -> Option<Vec<u8>> {
    let mut upper = prefix.to_vec();
    while let Some(&last) = upper.last() {
        if last == 0xFF {
            upper.pop();
        } else {
            *upper.last_mut().unwrap() += 1;
            return Some(upper);
        }
    }
    None
}

fn read_bytes_ref(input: &mut SliceInput) -> Result<Vec<u8>> {
    let len = input.read_vint()?;
    if len < 0 {
        return Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "invalid bytes length: {len}"
        ))));
    }
    let mut buf = vec![0u8; len as usize];
    input.read_bytes(&mut buf)?;
    Ok(buf)
}

/// Reads the `sumTotalTermFreq`/`sumDocFreq` pair, aliasing the single value
/// written when `IndexOptions::Docs` (frequencies aren't stored at all, so
/// `sumTotalTermFreq == sumDocFreq` and only one vlong is on the wire) —
/// mirrors `Lucene103BlockTreeTermsReader`'s constructor exactly.
fn read_freq_pair(input: &mut SliceInput, index_options: IndexOptions) -> Result<(i64, i64)> {
    let first = input.read_vlong()?;
    if index_options == IndexOptions::Docs {
        Ok((first, first))
    } else {
        let sum_doc_freq = input.read_vlong()?;
        Ok((first, sum_doc_freq))
    }
}

/// One decoded trie node (`TrieReader.Node`), covering all three shapes
/// (`SIGN_NO_CHILDREN`/`SIGN_SINGLE_CHILD_*`/`SIGN_MULTI_CHILDREN`) in a
/// single struct rather than Java's shape-specific fields left unset —
/// simpler than a Rust enum-per-shape here since [`load_node`] always fills
/// every field it needs for that shape and callers only ever read the
/// fields relevant to `node.sign`, mirroring how `TrieReader.Node` itself
/// mixes single-child/multi-child fields in one class.
#[derive(Debug, Clone, Copy)]
struct TrieNode {
    sign: u32,
    /// This node's own file pointer within the field's `.tip` index slice.
    fp: usize,
    /// `Node.outputFp`/`Node.hasOutput()` — `None` when this node has no
    /// terms/sub-block of its own (an internal node that exists purely to
    /// route to deeper children).
    output_fp: Option<u64>,
    has_terms: bool,
    /// `Node.floorDataFp`/`Node.isFloor()`.
    floor_data_fp: Option<usize>,
    /// Single-child only: `Node.childDeltaFp`/`Node.minChildrenLabel`.
    child_delta_fp: u64,
    min_children_label: u8,
    /// Multi-children only: `Node.strategyFp`/`childSaveStrategy`/
    /// `strategyBytes`/`childrenDeltaFpBytes` (`minChildrenLabel` above is
    /// shared with the single-child case; multi packs it in the same role).
    strategy_fp: usize,
    child_save_strategy: u32,
    strategy_bytes: usize,
    children_delta_fp_bytes: usize,
}

fn read_u64_at(slice: &[u8], fp: usize) -> Result<u64> {
    if fp + 8 > slice.len() {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "trie node read past end of index slice".into(),
        )));
    }
    Ok(u64::from_le_bytes(slice[fp..fp + 8].try_into().unwrap()))
}

fn read_u8_at(slice: &[u8], fp: usize) -> Result<u8> {
    slice.get(fp).copied().ok_or_else(|| {
        Error::Store(lucene_store::Error::Corrupted(
            "trie node read past end of index slice".into(),
        ))
    })
}

/// Reads `n_bytes` (1..=8) little-endian bytes starting at `fp` into a
/// `u64` — `TrieBuilder.writeLongNBytes`'s read-side inverse, used for the
/// multi-children children-fp array (`TrieReader.lookupChild`'s
/// `BYTES_MINUS_1_MASK`-free array-read, since here `n_bytes` is already a
/// byte count rather than a "minus 1" nibble).
fn read_u64_n_bytes(slice: &[u8], fp: usize, n_bytes: usize) -> Result<u64> {
    if fp + n_bytes > slice.len() {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "trie children-fp array read past end of index slice".into(),
        )));
    }
    let mut v = 0u64;
    for i in 0..n_bytes {
        v |= (slice[fp + i] as u64) << (8 * i);
    }
    Ok(v)
}

/// Reads one trie node at `fp` within `slice` (the field's `[indexStart,
/// indexEnd)` region of `.tip`) — `TrieReader.load`, dispatching on `sign`
/// to `loadLeafNode`/`loadSingleChildNode`/`loadMultiChildrenNode`.
fn load_node(slice: &[u8], fp: usize) -> Result<TrieNode> {
    let word = read_u64_at(slice, fp)?;
    let term = word as u32;
    let sign = term & 0x03;

    match sign {
        SIGN_NO_CHILDREN => {
            // loadLeafNode: [floor data][output fp][1x|floor|terms|3b fpBytes|2b sign]
            let fp_bytes_minus1 = (term >> 2) & 0x07;
            let output_fp = if fp_bytes_minus1 <= 6 {
                (word >> 8) & BYTES_MINUS_1_MASK[fp_bytes_minus1 as usize]
            } else {
                read_u64_at(slice, fp + 1)?
            };
            let has_terms = (term & LEAF_NODE_HAS_TERMS) != 0;
            let floor_data_fp = if (term & LEAF_NODE_HAS_FLOOR) != 0 {
                Some(fp + 2 + fp_bytes_minus1 as usize)
            } else {
                None
            };
            Ok(TrieNode {
                sign,
                fp,
                output_fp: Some(output_fp),
                has_terms,
                floor_data_fp,
                child_delta_fp: 0,
                min_children_label: 0,
                strategy_fp: 0,
                child_save_strategy: 0,
                strategy_bytes: 0,
                children_delta_fp_bytes: 0,
            })
        }
        SIGN_SINGLE_CHILD_WITH_OUTPUT | SIGN_SINGLE_CHILD_WITHOUT_OUTPUT => {
            // loadSingleChildNode: [floor][encoded output fp][child fp][label]
            // [3b encoded output fp bytes|3b child fp bytes|2b sign]
            let child_delta_bytes_minus1 = (term >> 2) & 0x07;
            let l = if child_delta_bytes_minus1 <= 5 {
                word >> 16
            } else {
                read_u64_at(slice, fp + 2)?
            };
            let child_delta_fp = l & BYTES_MINUS_1_MASK[child_delta_bytes_minus1 as usize];
            let min_children_label = ((term >> 8) & 0xFF) as u8;

            if sign == SIGN_SINGLE_CHILD_WITHOUT_OUTPUT {
                Ok(TrieNode {
                    sign,
                    fp,
                    output_fp: None,
                    has_terms: false,
                    floor_data_fp: None,
                    child_delta_fp,
                    min_children_label,
                    strategy_fp: 0,
                    child_save_strategy: 0,
                    strategy_bytes: 0,
                    children_delta_fp_bytes: 0,
                })
            } else {
                let encoded_bytes_minus1 = (term >> 5) & 0x07;
                let offset = fp + child_delta_bytes_minus1 as usize + 3;
                let encoded_fp =
                    read_u64_at(slice, offset)? & BYTES_MINUS_1_MASK[encoded_bytes_minus1 as usize];
                let output_fp = encoded_fp >> 2;
                let has_terms = (encoded_fp & NON_LEAF_NODE_HAS_TERMS) != 0;
                let floor_data_fp = if (encoded_fp & NON_LEAF_NODE_HAS_FLOOR) != 0 {
                    Some(offset + encoded_bytes_minus1 as usize + 1)
                } else {
                    None
                };
                Ok(TrieNode {
                    sign,
                    fp,
                    output_fp: Some(output_fp),
                    has_terms,
                    floor_data_fp,
                    child_delta_fp,
                    min_children_label,
                    strategy_fp: 0,
                    child_save_strategy: 0,
                    strategy_bytes: 0,
                    children_delta_fp_bytes: 0,
                })
            }
        }
        SIGN_MULTI_CHILDREN => {
            // loadMultiChildrenNode: [floor][children fps][strategy data]
            // [children count if floor][encoded output fp][label]
            // [5b strategy bytes|2b strategy|3b encoded fp bytes|1b has
            //  output|3b children fp bytes|2b sign]
            let children_delta_fp_bytes = (((term >> 2) & 0x07) + 1) as usize;
            let child_save_strategy = (term >> 9) & 0x03;
            let strategy_bytes = (((term >> 11) & 0x1F) + 1) as usize;
            let min_children_label = ((term >> 16) & 0xFF) as u8;

            if (term & 0x20) != 0 {
                let encoded_bytes_minus1 = (term >> 6) & 0x07;
                let l = if encoded_bytes_minus1 <= 4 {
                    word >> 24
                } else {
                    read_u64_at(slice, fp + 3)?
                };
                let encoded_fp = l & BYTES_MINUS_1_MASK[encoded_bytes_minus1 as usize];
                let output_fp = encoded_fp >> 2;
                let has_terms = (encoded_fp & NON_LEAF_NODE_HAS_TERMS) != 0;
                let (strategy_fp, floor_data_fp) = if (encoded_fp & NON_LEAF_NODE_HAS_FLOOR) != 0 {
                    let offset = fp + 4 + encoded_bytes_minus1 as usize;
                    let children_num = (read_u8_at(slice, offset)? as u64) + 1;
                    let sfp = offset + 1;
                    (
                        sfp,
                        Some(
                            sfp + strategy_bytes
                                + (children_num as usize) * children_delta_fp_bytes,
                        ),
                    )
                } else {
                    (fp + 4 + encoded_bytes_minus1 as usize, None)
                };
                Ok(TrieNode {
                    sign,
                    fp,
                    output_fp: Some(output_fp),
                    has_terms,
                    floor_data_fp,
                    child_delta_fp: 0,
                    min_children_label,
                    strategy_fp,
                    child_save_strategy,
                    strategy_bytes,
                    children_delta_fp_bytes,
                })
            } else {
                Ok(TrieNode {
                    sign,
                    fp,
                    output_fp: None,
                    has_terms: false,
                    floor_data_fp: None,
                    child_delta_fp: 0,
                    min_children_label,
                    strategy_fp: fp + 3,
                    child_save_strategy,
                    strategy_bytes,
                    children_delta_fp_bytes,
                })
            }
        }
        _ => unreachable!("sign is masked to 2 bits"),
    }
}

/// Enumerates a `SIGN_MULTI_CHILDREN` node's children's labels and file pointers —
/// generalizes `TrieReader.lookupChild`'s per-strategy label decoding
/// (`ChildSaveStrategy.BITS`/`ARRAY`/`REVERSE_ARRAY`) from "find one label"
/// to "list every label", since this module materializes a field's entire
/// term dictionary up front rather than walking toward one target term (see
/// the module doc). Order is irrelevant to callers ([`collect_leaf_blocks`]
/// sorts all decoded entries at the end), so labels are produced in
/// ascending order purely because that's the natural decode order for all
/// three strategies, not because it's required.
fn multi_children_labels_and_fps(slice: &[u8], node: &TrieNode) -> Result<Vec<(u8, usize)>> {
    let strategy_fp = node.strategy_fp;
    let strategy_bytes = node.strategy_bytes;
    let min_label = node.min_children_label;

    let mut labels: Vec<u8> = Vec::new();
    match node.child_save_strategy {
        CHILD_STRATEGY_REVERSE_ARRAY => {
            let max_label = read_u8_at(slice, strategy_fp)?;
            let mut missing = Vec::with_capacity(strategy_bytes.saturating_sub(1));
            for i in 0..strategy_bytes.saturating_sub(1) {
                missing.push(read_u8_at(slice, strategy_fp + 1 + i)?);
            }
            let mut mi = 0;
            let mut lbl = min_label;
            loop {
                if mi < missing.len() && missing[mi] == lbl {
                    mi += 1;
                } else {
                    labels.push(lbl);
                }
                if lbl == max_label {
                    break;
                }
                lbl = lbl.wrapping_add(1);
            }
        }
        CHILD_STRATEGY_ARRAY => {
            labels.push(min_label);
            for i in 0..strategy_bytes {
                labels.push(read_u8_at(slice, strategy_fp + i)?);
            }
        }
        CHILD_STRATEGY_BITS => {
            for i in 0..strategy_bytes {
                let byte = read_u8_at(slice, strategy_fp + i)?;
                for bit in 0..8u32 {
                    if byte & (1 << bit) != 0 {
                        let pos = (i as u32) * 8 + bit;
                        labels.push((min_label as u32 + pos) as u8);
                    }
                }
            }
        }
        other => {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "invalid child save strategy code: {other}"
            ))))
        }
    }

    let mut result = Vec::with_capacity(labels.len());
    for (i, label) in labels.iter().enumerate() {
        let off = strategy_fp + strategy_bytes + i * node.children_delta_fp_bytes;
        let delta = read_u64_n_bytes(slice, off, node.children_delta_fp_bytes)?;
        if (delta as usize) > node.fp {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "trie child delta fp exceeds parent fp".into(),
            )));
        }
        result.push((*label, node.fp - delta as usize));
    }
    Ok(result)
}

/// Resolves one trie node's own `(fp, hasTerms)` output into every physical
/// `.tim` block it addresses — just the one block if not a floor node, or
/// the base block plus every follow-on floor sub-block otherwise
/// (`SegmentTermsEnumFrame.setFloorData`/`scanToFloorFrame`'s byte layout:
/// `numFollowFloorBlocks: vint`, then that many `(floorLeadByte: byte,
/// code: vlong)` pairs where `code = (subFp - baseFp) << 1 | hasTerms`).
/// Labels are read past but not returned — picking a single floor sub-block
/// by label is what real Lucene's `scanToFloorFrame` does for one target
/// term; this module instead decodes every floor sub-block unconditionally
/// (see the module doc's eager-materialization tradeoff).
fn expand_floor(
    slice: &[u8],
    base_fp: u64,
    base_has_terms: bool,
    floor_data_fp: Option<usize>,
) -> Result<Vec<(u64, bool)>> {
    let mut blocks = vec![(base_fp, base_has_terms)];
    let Some(ffp) = floor_data_fp else {
        return Ok(blocks);
    };
    let mut r = SliceInput::new(slice);
    r.seek(ffp)?;
    let num_follow = r.read_vint()?;
    if num_follow < 0 {
        return Err(Error::Store(lucene_store::Error::Corrupted(format!(
            "invalid numFollowFloorBlocks: {num_follow}"
        ))));
    }
    for _ in 0..num_follow {
        let _label = r.read_byte()?;
        let code = r.read_vlong()? as u64;
        let fp = base_fp.wrapping_add(code >> 1);
        let has_terms = (code & 1) != 0;
        blocks.push((fp, has_terms));
    }
    Ok(blocks)
}

/// Recursively visits every trie node reachable from `node`, expanding
/// every node-with-output into its (possibly floor-split) physical block
/// list and appending them to `out` — the traversal side of the module
/// doc's "eager whole-field materialization" design. `depth` is a sanity
/// bound against a corrupted/cyclic trie, not a real limit (trie depth is
/// bounded by term length in any real index).
/// Every reachable physical `.tim` block, paired with the trie label path
/// (i.e. the block's own key prefix) that led to it -- needed because
/// `decode_block` only ever sees a block's *suffix* bytes (the writer never
/// repeats a shared prefix inside the block itself; see
/// `Lucene103BlockTreeTermsWriter`'s prefix-stripping), so a multi-level
/// trie's blocks must have that prefix re-applied by the caller to recover
/// full term bytes (the previous single-block-only slice never needed this,
/// since its one block always sat at the trie root with an empty prefix).
fn collect_leaf_blocks(
    slice: &[u8],
    node: &TrieNode,
    depth: u32,
    prefix: &mut Vec<u8>,
    out: &mut Vec<(u64, Vec<u8>)>,
) -> Result<()> {
    if depth > 10_000 {
        return Err(Error::Unsupported("trie nesting too deep (possible cycle)"));
    }
    if let Some(fp) = node.output_fp {
        for (block_fp, has_terms) in expand_floor(slice, fp, node.has_terms, node.floor_data_fp)? {
            // `hasTerms == false` means this specific physical `.tim` block
            // holds nothing but pointers to its own further-nested
            // sub-blocks (`Lucene103BlockTreeTermsWriter.writeBlocks`
            // recursing to a deeper prefix rather than floor-splitting).
            // Real Lucene can fall back to reading those pointers
            // in-block (`SegmentTermsEnumFrame.nextNonLeaf`/`subCode`), but
            // this port doesn't need to: `PendingBlock.compileIndex`
            // unconditionally merges every deeper recursion's own trie
            // (`subIndices`) into the very same trie this function is
            // already walking, so every term this pointer-only block would
            // have routed to is independently reachable as a *separate*,
            // deeper trie node/child -- this block itself is redundant for
            // indexed lookup and is simply skipped, not decoded.
            if has_terms {
                out.push((block_fp, prefix.clone()));
            }
        }
    }
    match node.sign {
        SIGN_NO_CHILDREN => {}
        SIGN_SINGLE_CHILD_WITH_OUTPUT | SIGN_SINGLE_CHILD_WITHOUT_OUTPUT => {
            if (node.child_delta_fp as usize) > node.fp {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "trie child delta fp exceeds parent fp".into(),
                )));
            }
            let child_fp = node.fp - node.child_delta_fp as usize;
            let child = load_node(slice, child_fp)?;
            prefix.push(node.min_children_label);
            let r = collect_leaf_blocks(slice, &child, depth + 1, prefix, out);
            prefix.pop();
            r?;
        }
        SIGN_MULTI_CHILDREN => {
            for (label, child_fp) in multi_children_labels_and_fps(slice, node)? {
                let child = load_node(slice, child_fp)?;
                prefix.push(label);
                let r = collect_leaf_blocks(slice, &child, depth + 1, prefix, out);
                prefix.pop();
                r?;
            }
        }
        _ => unreachable!("sign is masked to 2 bits"),
    }
    Ok(())
}

/// Port of `LowercaseAsciiCompression.decompress` (`o.a.l.util.compress`):
/// undoes the 4-into-3-byte 6-bit pack (bytes mostly in `[0x1F,0x3F)` /
/// `[0x5F,0x7F)`, i.e. ASCII digits/lowercase/`.`/`-`/`_`) plus a trailing
/// exception list for the rare non-compressible byte. `out.len()` is the
/// *original* (decompressed) length, matching Java's `len` parameter; the
/// compressed byte count (`compressedLen = len - len/4`) is derived from it,
/// not read from the stream.
fn decompress_lowercase_ascii(r: &mut SliceInput, out: &mut [u8]) -> Result<()> {
    let len = out.len();
    let saved = len >> 2;
    let compressed_len = len - saved;

    // 1. Copy the packed bytes.
    r.read_bytes(&mut out[..compressed_len])?;

    // 2. Restore the leading 2 bits of each packed byte into whole bytes.
    for i in 0..saved {
        out[compressed_len + i] = ((out[i] & 0xC0) >> 2)
            | ((out[saved + i] & 0xC0) >> 4)
            | ((out[(saved << 1) + i] & 0xC0) >> 6);
    }

    // 3. Move back to the original range.
    for b in out.iter_mut() {
        *b = ((*b & 0x1F) | 0x20 | ((*b & 0x20) << 1)).wrapping_sub(1);
    }

    // 4. Restore exceptions.
    let num_exceptions = r.read_vint()?;
    let mut i: usize = 0;
    for _ in 0..num_exceptions {
        i += r.read_byte()? as usize;
        if i >= out.len() {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "lowercase-ASCII exception index out of range".into(),
            )));
        }
        out[i] = r.read_byte()?;
    }

    Ok(())
}

/// Decodes a single physical `.tim` block at `fp`, materializing every
/// (term, stats, metadata) entry — `SegmentTermsEnumFrame.loadBlock` plus a
/// full `decodeMetaData` pass over every entry. Handles both **leaf** blocks
/// (`isLeafBlock`, every entry a term) and **non-leaf** blocks (some entries
/// are pointers to further-nested sub-blocks, `SegmentTermsEnumFrame.nextNonLeaf`'s
/// `code & 1` "is this a sub-block" bit) by recursing into [`decode_block`]
/// again at each sub-block's resolved `fp` — this is the genuine "multi-level
/// blocktree" case: a `.tim` block that is itself an *internal* node pointing
/// to child blocks rather than directly to postings, distinct from both the
/// `.tip` trie's own multi-level node structure (root/single-child/multi-children,
/// arbitrarily deep, already supported by [`collect_leaf_blocks`]) and from
/// floor sub-blocks (same trie node, multiple physical blocks, see
/// [`expand_floor`]). A sub-block entry's own key bytes are only that
/// sub-block's *own* shared prefix relative to *this* block's prefix (not a
/// full term) — the returned entries for a sub-block are prefixed with those
/// bytes before being merged into this block's own entries, so every entry
/// this function returns is relative to the same prefix depth regardless of
/// how many sub-block levels were recursed through (the caller, [`open`],
/// then prepends the `.tip`-trie-derived prefix on top, same as before).
/// Floor sub-blocks are just more calls to this same function at different
/// `fp`s — floor selection happens one level up, in
/// [`expand_floor`]/[`collect_leaf_blocks`], not here.
///
/// Test-only: [`open`] calls [`decode_block_at_depth`] directly (it always
/// has an `already_trie_indexed` set on hand to pass through), so this
/// no-cross-check convenience wrapper is only exercised by this module's own
/// unit tests, which decode one block in isolation.
#[cfg(test)]
fn decode_block(
    tim: &[u8],
    fp: usize,
    index_options: IndexOptions,
    has_payloads: bool,
) -> Result<Vec<(Vec<u8>, TermStats, TermMetadata)>> {
    decode_block_at_depth(tim, fp, index_options, has_payloads, 0, None)
}

/// `decode_block`'s actual implementation, `depth`-tracked so a
/// corrupted/cyclic sub-block chain (`subFP` pointing at or past its own
/// parent) fails with [`Error::Unsupported`] rather than recursing forever —
/// mirrors [`collect_leaf_blocks`]'s own `depth > 10_000` sanity bound for the
/// `.tip` trie's recursion.
///
/// `already_trie_indexed`, when given, is the full set of physical block fps
/// [`collect_leaf_blocks`] already collected directly from the `.tip` trie
/// for this field (i.e. every block reachable via its own dedicated trie/FST
/// arc). A sub-block pointer whose target fp is in that set is **not**
/// recursed into here: real Lucene's writer can and does give some sub-blocks
/// *both* an in-block pointer from their parent's non-leaf entries *and* a
/// separate, independently-indexed `.tip` trie arc reached by a different
/// path (confirmed empirically against a real ~8k-distinct-term fixture —
/// see `crates/lucene-codecs/tests/blocktree_multilevel_fixture.rs`'s module
/// doc). Real `SegmentTermsEnum` never notices this redundancy because it
/// only ever takes *one* of the two paths per lookup; this module's eager
/// whole-field materialization (see the module doc) visits every reachable
/// block, so without this check a doubly-addressed sub-block would be
/// decoded twice — once here, following its parent's pointer, and once more
/// as its own top-level entry in [`open`]'s block loop — producing duplicate
/// entries `entries.sort_by` can't detect (same term appears twice with
/// identical stats, silently violating an implicit "each term once"
/// invariant instead of throwing anything). Skipping it here is exactly
/// [`collect_leaf_blocks`]'s own existing "hasTerms == false is redundant,
/// don't decode it, it's independently reachable as a deeper trie node"
/// reasoning, generalized from *trie nodes* to *in-block sub-block pointers*.
/// `None` (used by direct/standalone [`decode_block`] callers, including
/// every unit test in this module) means "no known independent top-level set
/// to cross-check against", so every sub-block pointer is followed — correct
/// for decoding one block's own self-contained sub-tree in isolation, where
/// this cross-cutting duplication with *other* top-level blocks cannot arise.
fn decode_block_at_depth(
    tim: &[u8],
    fp: usize,
    index_options: IndexOptions,
    has_payloads: bool,
    depth: u32,
    already_trie_indexed: Option<&std::collections::HashSet<usize>>,
) -> Result<Vec<(Vec<u8>, TermStats, TermMetadata)>> {
    if depth > 10_000 {
        return Err(Error::Unsupported(
            "terms block sub-block nesting too deep (possible cycle)",
        ));
    }
    let mut r = SliceInput::new(tim);
    r.seek(fp)?;

    let code = r.read_vint()?;
    let ent_count = (code as u32) >> 1;
    if ent_count == 0 {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "empty terms block".into(),
        )));
    }
    // isLastInFloor (`code & 1`): whether this is the last physical block in
    // its floor set. Not needed here — [`expand_floor`]/[`collect_leaf_blocks`]
    // already resolved every floor sub-block's own fp up front, so this
    // decode doesn't need to chain to a "next" block by inspecting this bit;
    // it's read past purely to keep the cursor aligned with `entCount`.
    let _is_last_in_floor = (code & 1) != 0;

    let code_l = r.read_vlong()? as u64;
    let is_leaf_block = (code_l & 0x04) != 0;
    let num_suffix_bytes = (code_l >> 3) as usize;
    let compression_alg = code_l & 0x03;
    let mut suffix_bytes = vec![0u8; num_suffix_bytes];
    match compression_alg {
        0 => {
            // NO_COMPRESSION (`CompressionAlgorithm.NO_COMPRESSION.read`): the
            // suffix bytes sit raw in the stream.
            r.read_bytes(&mut suffix_bytes)?;
        }
        1 => {
            // LOWERCASE_ASCII (`CompressionAlgorithm.LOWERCASE_ASCII.read` ->
            // `LowercaseAsciiCompression.decompress`).
            decompress_lowercase_ascii(&mut r, &mut suffix_bytes)?;
        }
        2 => {
            // LZ4 (`CompressionAlgorithm.LZ4.read` -> `LZ4.decompress`),
            // reusing this port's own `lz4::decompress`.
            crate::lz4::decompress(&mut r, num_suffix_bytes, &mut suffix_bytes, 0)?;
        }
        _ => {
            // `code_l & 0x03` is masked to 2 bits, so `3` is the only
            // remaining value; real Lucene's `CompressionAlgorithm.byCode`
            // throws `IllegalArgumentException` for it too (only codes 0-2
            // are ever assigned to an enum constant).
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "illegal compression algorithm code (3) for a terms block".into(),
            )));
        }
    }

    let num_suffix_length_bytes_raw = r.read_vint()? as u32;
    let all_equal = (num_suffix_length_bytes_raw & 1) != 0;
    let num_suffix_length_bytes = (num_suffix_length_bytes_raw >> 1) as usize;
    let mut suffix_length_bytes = vec![0u8; num_suffix_length_bytes];
    if all_equal {
        let b = r.read_byte()?;
        suffix_length_bytes.fill(b);
    } else {
        r.read_bytes(&mut suffix_length_bytes)?;
    }

    let num_stat_bytes = r.read_vint()? as usize;
    let mut stat_bytes = vec![0u8; num_stat_bytes];
    r.read_bytes(&mut stat_bytes)?;

    // Per-term postings metadata (`Lucene104PostingsReader.decodeTerm`, see
    // `crate::postings`'s module doc): decoded below, threaded across entries
    // exactly like `SegmentTermsEnumFrame` threads `IntBlockTermState`
    // (`absolute` true only for this block's first term).
    let num_meta_bytes = r.read_vint()? as usize;
    let mut meta_bytes = vec![0u8; num_meta_bytes];
    r.read_bytes(&mut meta_bytes)?;

    let mut suffix_lengths_reader = SliceInput::new(&suffix_length_bytes);
    let mut suffixes_reader = SliceInput::new(&suffix_bytes);
    let mut stats_reader = SliceInput::new(&stat_bytes);
    let mut meta_reader = SliceInput::new(&meta_bytes);

    let mut singleton_run_length: u32 = 0;
    let mut prev_meta = TermMetadata::EMPTY;
    // Real-term ordinal within this block (`SegmentTermsEnumFrame.state.termBlockOrd`),
    // distinct from the raw entry loop counter once sub-block entries are
    // possible: stats/meta streams only ever hold one record per *term*
    // entry, never per sub-block entry, and `decode_term_metadata`'s
    // `absolute` flag is true only for this block's first *term* (ordinal 0),
    // not merely the first entry (which could be a sub-block).
    let mut term_ord: u32 = 0;
    let mut entries = Vec::with_capacity(ent_count as usize);
    for _ in 0..ent_count {
        // Leaf entries carry a plain suffix-length vint (every entry is a
        // term); non-leaf entries pack `suffixLength << 1 | isSubBlock` into
        // that same vint (`SegmentTermsEnumFrame.nextNonLeaf`'s `code`), with
        // a sub-block's own delta-fp (`subCode`, a vlong) following
        // immediately in the *same* suffix-lengths stream when the low bit is
        // set.
        let (suffix_len, is_sub_block) = if is_leaf_block {
            (suffix_lengths_reader.read_vint()? as usize, false)
        } else {
            let code = suffix_lengths_reader.read_vint()? as u32;
            ((code >> 1) as usize, (code & 1) != 0)
        };
        let mut suffix = vec![0u8; suffix_len];
        suffixes_reader.read_bytes(&mut suffix)?;

        if is_sub_block {
            let sub_code = suffix_lengths_reader.read_vlong()? as u64;
            if sub_code as usize > fp {
                return Err(Error::Store(lucene_store::Error::Corrupted(
                    "terms block sub-block delta fp exceeds parent fp".into(),
                )));
            }
            let sub_fp = fp - sub_code as usize;
            if already_trie_indexed.is_some_and(|s| s.contains(&sub_fp)) {
                // Independently reachable as its own top-level block via the
                // `.tip` trie (see this function's doc comment) -- decoding
                // it here too would duplicate every one of its terms.
                continue;
            }
            let sub_entries = decode_block_at_depth(
                tim,
                sub_fp,
                index_options,
                has_payloads,
                depth + 1,
                already_trie_indexed,
            )?;
            for (sub_suffix, stats, meta) in sub_entries {
                let mut full = suffix.clone();
                full.extend_from_slice(&sub_suffix);
                entries.push((full, stats, meta));
            }
            continue;
        }

        let (doc_freq, total_term_freq) = if singleton_run_length > 0 {
            singleton_run_length -= 1;
            (1, 1)
        } else {
            let token = stats_reader.read_vint()?;
            if token & 1 == 1 {
                singleton_run_length = (token as u32) >> 1;
                (1, 1)
            } else {
                let doc_freq = (token as u32) >> 1;
                let total_term_freq = if index_options == IndexOptions::Docs {
                    doc_freq as i64
                } else {
                    doc_freq as i64 + stats_reader.read_vlong()?
                };
                (doc_freq as i32, total_term_freq)
            }
        };

        let meta = postings::decode_term_metadata(
            &mut meta_reader,
            doc_freq,
            term_ord == 0,
            prev_meta,
            index_options,
            has_payloads,
            total_term_freq,
        )?;
        prev_meta = meta;
        term_ord += 1;

        entries.push((
            suffix,
            TermStats {
                doc_freq,
                total_term_freq,
            },
            meta,
        ));
    }

    Ok(entries)
}

/// Opens a `.tim`/`.tip`/`.tmd` triple already read whole into memory,
/// decoding every field's single-block term dictionary eagerly (see the
/// module doc for the size/shape scope this covers).
pub fn open(
    tim: &[u8],
    tip: &[u8],
    tmd: &[u8],
    field_infos: &FieldInfos,
    segment_id: &[u8; ID_LENGTH],
    segment_suffix: &str,
    max_doc: i32,
) -> Result<BlockTreeFields> {
    let mut tim_input = SliceInput::new(tim);
    let tim_header = codec_util::check_index_header(
        &mut tim_input,
        TERMS_CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;

    let mut tip_input = SliceInput::new(tip);
    codec_util::check_index_header(
        &mut tip_input,
        TERMS_INDEX_CODEC_NAME,
        tim_header.version,
        tim_header.version,
        segment_id,
        segment_suffix,
    )?;

    let mut tmd_input = SliceInput::new(tmd);
    codec_util::check_index_header(
        &mut tmd_input,
        TERMS_META_CODEC_NAME,
        tim_header.version,
        tim_header.version,
        segment_id,
        segment_suffix,
    )?;

    // PostingsReaderBase.init: the postings writer's own header, embedded in
    // the same .tmd stream right after BlockTree's index header.
    codec_util::check_index_header(
        &mut tmd_input,
        POSTINGS_TERMS_CODEC,
        POSTINGS_VERSION_START,
        POSTINGS_VERSION_CURRENT,
        segment_id,
        segment_suffix,
    )?;
    let index_block_size = tmd_input.read_vint()?;
    if index_block_size != POSTINGS_BLOCK_SIZE {
        return Err(Error::UnexpectedBlockSize {
            found: index_block_size,
        });
    }

    let num_fields = tmd_input.read_vint()?;
    if num_fields < 0 {
        return Err(Error::InvalidNumFields(num_fields));
    }

    let mut fields = Vec::with_capacity(num_fields as usize);
    for _ in 0..num_fields {
        let field_number = tmd_input.read_vint()?;
        let num_terms = tmd_input.read_vlong()?;
        if num_terms <= 0 {
            return Err(Error::IllegalNumTerms(field_number));
        }
        let field_info = field_infos
            .field_by_number(field_number)
            .ok_or(Error::InvalidFieldNumber(field_number))?;

        let (sum_total_term_freq, sum_doc_freq) =
            read_freq_pair(&mut tmd_input, field_info.index_options)?;
        let doc_count = tmd_input.read_vint()?;
        let min_term = read_bytes_ref(&mut tmd_input)?;
        let mut max_term = read_bytes_ref(&mut tmd_input)?;
        if num_terms == 1 {
            max_term = min_term.clone();
        }

        if !(0..=max_doc).contains(&doc_count) {
            return Err(Error::InvalidDocCount { doc_count, max_doc });
        }
        if sum_doc_freq < doc_count as i64 {
            return Err(Error::InvalidSumDocFreq {
                sum_doc_freq,
                doc_count,
            });
        }
        if sum_total_term_freq < sum_doc_freq {
            return Err(Error::InvalidSumTotalTermFreq {
                sum_total_term_freq,
                sum_doc_freq,
            });
        }

        let index_start = tmd_input.read_vlong()? as usize;
        let root_fp = tmd_input.read_vlong()? as usize;
        let index_end = tmd_input.read_vlong()? as usize;

        if index_end > tip.len() || index_start > index_end {
            return Err(Error::Store(lucene_store::Error::Corrupted(
                "field index region out of bounds".into(),
            )));
        }
        let index_slice = &tip[index_start..index_end];
        let root = load_node(index_slice, root_fp)?;
        let mut blocks = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(index_slice, &root, 0, &mut prefix, &mut blocks)?;
        if blocks.is_empty() {
            return Err(Error::Unsupported(
                "root block with no terms (all sub-blocks) not supported in this slice",
            ));
        }

        // Every block fp reached directly via the `.tip` trie -- passed down
        // so `decode_block_at_depth` can recognize (and skip re-decoding) a
        // sub-block that's *also* independently trie-indexed elsewhere (see
        // that function's doc comment for why real Lucene bytes can and do
        // address the same physical block both ways).
        let trie_block_fps: std::collections::HashSet<usize> =
            blocks.iter().map(|(fp, _)| *fp as usize).collect();

        let mut entries = Vec::with_capacity(num_terms as usize);
        for (block_fp, block_prefix) in blocks {
            for (suffix, stats, meta) in decode_block_at_depth(
                tim,
                block_fp as usize,
                field_info.index_options,
                field_info.store_payloads,
                0,
                Some(&trie_block_fps),
            )? {
                // `decode_block` only ever sees a block's suffix bytes (the
                // writer strips the shared trie-path prefix); re-attach it
                // here to recover the full term (see `collect_leaf_blocks`'s
                // doc comment).
                let mut term = block_prefix.clone();
                term.extend_from_slice(&suffix);
                entries.push((term, stats, meta));
            }
        }
        // Blocks are decoded in trie-traversal order, not necessarily sorted
        // term order (see the module doc) -- re-sort once, here, so
        // `FieldTerms::seek_exact`'s binary search stays correct regardless
        // of how many blocks a field spans.
        entries.sort_by(|a, b| a.0.cmp(&b.0));
        if entries.len() as i64 != num_terms {
            return Err(Error::Store(lucene_store::Error::Corrupted(format!(
                "decoded {} terms but field metadata says numTerms={num_terms}",
                entries.len()
            ))));
        }

        if fields
            .iter()
            .any(|(n, _): &(String, FieldTerms)| n == &field_info.name)
        {
            return Err(Error::DuplicateField(field_info.name.clone()));
        }
        fields.push((
            field_info.name.clone(),
            FieldTerms {
                num_terms,
                sum_total_term_freq,
                sum_doc_freq,
                doc_count,
                min_term,
                max_term,
                index_options: field_info.index_options,
                has_payloads: field_info.store_payloads,
                entries,
            },
        ));
    }

    let index_length = tmd_input.read_i64()?;
    let terms_length = tmd_input.read_i64()?;
    codec_util::check_footer(&mut tmd_input, tmd.len())?;

    if index_length as usize > tip.len() || terms_length as usize > tim.len() {
        return Err(Error::Store(lucene_store::Error::Corrupted(
            "recorded .tip/.tim length exceeds file size".into(),
        )));
    }
    codec_util::retrieve_checksum(tip)?;
    codec_util::retrieve_checksum(tim)?;

    Ok(BlockTreeFields { fields })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::FieldInfo;
    use lucene_store::data_output::DataOutput;

    fn field_info(number: i32, name: &str, index_options: IndexOptions) -> FieldInfo {
        FieldInfo {
            name: name.to_string(),
            number,
            store_term_vectors: false,
            omit_norms: false,
            store_payloads: false,
            soft_deletes_field: false,
            parent_field: false,
            index_options,
            doc_values_type: crate::field_infos::DocValuesType::None,
            doc_values_skip_index_type: crate::field_infos::DocValuesSkipIndexType::None,
            doc_values_gen: -1,
            attributes: Vec::new(),
            point_dimension_count: 0,
            point_index_dimension_count: 0,
            point_num_bytes: 0,
            vector_dimension: 0,
            vector_encoding: crate::field_infos::VectorEncoding::Float32,
            vector_similarity_function: crate::field_infos::VectorSimilarityFunction::Euclidean,
        }
    }

    /// Hand-builds a single-field, single-block `.tim`/`.tip`/`.tmd` triple
    /// (terms `["a", "ab", "b"]`, docFreq/totalTermFreq = 1/1, 2/3, 1/1) —
    /// this port's own encoder, test-only, to exercise error/boundary paths
    /// a real (small) fixture never reaches. Mirrors the pattern used by
    /// `codec_util.rs`/`segment_info.rs`'s own test-only encoders.
    struct Builder {
        id: [u8; ID_LENGTH],
        suffix: String,
    }

    impl Builder {
        fn new() -> Self {
            Builder {
                id: [7u8; ID_LENGTH],
                suffix: String::new(),
            }
        }

        fn build(
            &self,
            index_options: IndexOptions,
            terms: &[(&str, u32, u64)],
        ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
            // .tim
            let mut tim = Vec::new();
            codec_util::write_index_header(
                &mut tim,
                TERMS_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            let block_fp = tim.len();

            let ent_count = terms.len() as u32;
            let code = (ent_count << 1) | 1; // isLastInFloor
            tim.write_vint(code as i32);

            let mut suffix_bytes = Vec::new();
            let mut suffix_lengths = Vec::new();
            let mut stats = Vec::new();
            for (term, doc_freq, total_term_freq) in terms {
                suffix_bytes.extend_from_slice(term.as_bytes());
                suffix_lengths.write_vint(term.len() as i32);
                let token = (*doc_freq as i32) << 1; // never singleton-run-encoded, for test simplicity
                stats.write_vint(token);
                if index_options != IndexOptions::Docs {
                    stats.write_vlong((*total_term_freq as i64) - (*doc_freq as i64));
                }
            }

            let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // isLeafBlock, NO_COMPRESSION
            tim.write_vlong(code_l as i64);
            tim.write_bytes(&suffix_bytes);

            tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
            tim.write_bytes(&suffix_lengths);

            tim.write_vint(stats.len() as i32);
            tim.write_bytes(&stats);

            // Postings metadata: one entry per term via the bit=0
            // (docStartFP-delta) branch, legal regardless of `absolute` --
            // these seek_exact-only tests don't exercise postings decode, so
            // the fake docStartFP/singletonDocID values are never read back.
            let mut meta = Vec::new();
            for (_, doc_freq, _) in terms {
                meta.write_vlong(10 << 1);
                if *doc_freq == 1 {
                    meta.write_vint(0);
                }
            }
            tim.write_vint(meta.len() as i32);
            tim.write_bytes(&meta);

            codec_util::write_footer(&mut tim);

            // .tip: root node (SIGN_NO_CHILDREN), hasTerms, no floor.
            let mut tip = Vec::new();
            codec_util::write_index_header(
                &mut tip,
                TERMS_INDEX_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            let index_start = tip.len();
            let root_fp = 0usize;
            let output_fp_bytes = 8usize; // keep it simple: always 8 bytes
            let header = (SIGN_NO_CHILDREN as u8)
                | ((output_fp_bytes as u8 - 1) << 2)
                | (LEAF_NODE_HAS_TERMS as u8);
            tip.push(header);
            tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
            tip.extend_from_slice(&0u64.to_le_bytes()); // 8-byte over-read pad
            let index_end = tip.len();
            codec_util::write_footer(&mut tip);

            // .tmd
            let mut tmd = Vec::new();
            codec_util::write_index_header(
                &mut tmd,
                TERMS_META_CODEC_NAME,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            codec_util::write_index_header(
                &mut tmd,
                POSTINGS_TERMS_CODEC,
                VERSION_CURRENT,
                &self.id,
                &self.suffix,
            );
            tmd.write_vint(POSTINGS_BLOCK_SIZE);

            tmd.write_vint(1); // numFields
            tmd.write_vint(0); // field number
            let num_terms = terms.len() as i64;
            tmd.write_vlong(num_terms);
            let sum_doc_freq: i64 = terms.iter().map(|(_, d, _)| *d as i64).sum();
            let sum_total_term_freq: i64 = if index_options == IndexOptions::Docs {
                sum_doc_freq
            } else {
                terms.iter().map(|(_, _, t)| *t as i64).sum()
            };
            if index_options != IndexOptions::Docs {
                tmd.write_vlong(sum_total_term_freq);
            }
            tmd.write_vlong(sum_doc_freq);
            tmd.write_vint(1); // docCount
            let min_term = terms[0].0.as_bytes();
            let max_term = terms[terms.len() - 1].0.as_bytes();
            tmd.write_vint(min_term.len() as i32);
            tmd.write_bytes(min_term);
            tmd.write_vint(max_term.len() as i32);
            tmd.write_bytes(max_term);
            tmd.write_vlong(index_start as i64);
            tmd.write_vlong(root_fp as i64);
            tmd.write_vlong(index_end as i64);

            tmd.write_i64(index_end as i64); // indexLength
            tmd.write_i64((tim.len()) as i64); // termsLength
            codec_util::write_footer(&mut tmd);

            (tim, tip, tmd)
        }
    }

    #[test]
    fn seek_exact_found_and_not_found() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();
        assert_eq!(field.num_terms, 3);
        assert_eq!(field.sum_doc_freq, 4);
        assert_eq!(field.sum_total_term_freq, 5);
        assert_eq!(field.min_term, b"a");
        assert_eq!(field.max_term, b"b");

        assert_eq!(
            field.seek_exact(b"ab"),
            Some(TermStats {
                doc_freq: 2,
                total_term_freq: 3
            })
        );
        assert_eq!(
            field.seek_exact(b"a"),
            Some(TermStats {
                doc_freq: 1,
                total_term_freq: 1
            })
        );
        assert_eq!(field.seek_exact(b"missing"), None);
        assert_eq!(field.seek_exact(b""), None);
        assert!(fields.field("nope").is_none());
    }

    #[test]
    fn intersect_prefix_and_wildcard_over_materialized_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[
                ("apple", 1, 1),
                ("application", 1, 1),
                ("apply", 1, 1),
                ("banana", 1, 1),
                ("band", 1, 1),
            ],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        // Prefix: "app*" -> apple, application, apply (in sorted order).
        let pattern = WildcardPattern::new(b"app*");
        let got: Vec<&[u8]> = field.intersect(&pattern).map(|(t, _)| t).collect();
        assert_eq!(got, vec![b"apple".as_slice(), b"application", b"apply"]);

        // "?" wildcard: "ban?" matches nothing here ("band" is 4 bytes so
        // "ban?" matches "band" exactly -- exercise it precisely).
        let pattern = WildcardPattern::new(b"ban?");
        let got: Vec<&[u8]> = field.intersect(&pattern).map(|(t, _)| t).collect();
        assert_eq!(got, vec![b"band".as_slice()]);

        // No literal prefix ("*" in the middle only): "*ana*" -> banana.
        let pattern = WildcardPattern::new(b"*ana*");
        let got: Vec<&[u8]> = field.intersect(&pattern).map(|(t, _)| t).collect();
        assert_eq!(got, vec![b"banana".as_slice()]);

        // Matches everything.
        let pattern = WildcardPattern::new(b"*");
        assert_eq!(field.intersect(&pattern).count(), 5);

        // Matches nothing: valid prefix range, no candidate satisfies the
        // rest of the pattern.
        let pattern = WildcardPattern::new(b"app??????");
        assert_eq!(field.intersect(&pattern).count(), 0);

        // Matches nothing: prefix outside the field's term range entirely.
        let pattern = WildcardPattern::new(b"zzz*");
        assert_eq!(field.intersect(&pattern).count(), 0);

        // Exact-match pattern (no wildcard bytes at all) behaves like
        // seek_exact.
        let pattern = WildcardPattern::new(b"banana");
        let got: Vec<&[u8]> = field.intersect(&pattern).map(|(t, _)| t).collect();
        assert_eq!(got, vec![b"banana".as_slice()]);

        // PrefixQuery-shaped constructor.
        let pattern = WildcardPattern::prefix(b"ban");
        let got: Vec<&[u8]> = field.intersect(&pattern).map(|(t, _)| t).collect();
        assert_eq!(got, vec![b"banana".as_slice(), b"band"]);
    }

    #[test]
    fn regexp_intersect_over_materialized_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[
                ("apple", 1, 1),
                ("application", 1, 1),
                ("apply", 1, 1),
                ("banana", 1, 1),
                ("band", 1, 1),
            ],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        // Literal-prefix-narrowed range: "appl.*" -> apple, application,
        // apply (in sorted order).
        let pattern = RegexpPattern::new(b"appl.*").unwrap();
        let got: Vec<&[u8]> = field.regexp_intersect(&pattern).map(|(t, _)| t).collect();
        assert_eq!(got, vec![b"apple".as_slice(), b"application", b"apply"]);

        // Alternation has no useful literal prefix (falls back to a full
        // scan) but still matches correctly.
        let pattern = RegexpPattern::new(b"banana|band").unwrap();
        let got: Vec<&[u8]> = field.regexp_intersect(&pattern).map(|(t, _)| t).collect();
        assert_eq!(got, vec![b"banana".as_slice(), b"band"]);

        // Whole-term-match: "ban" alone matches neither "banana" nor "band".
        let pattern = RegexpPattern::new(b"ban").unwrap();
        assert_eq!(field.regexp_intersect(&pattern).count(), 0);

        // Matches nothing: prefix outside the field's term range entirely.
        let pattern = RegexpPattern::new(b"zzz.*").unwrap();
        assert_eq!(field.regexp_intersect(&pattern).count(), 0);
    }

    #[test]
    fn prefix_upper_bound_handles_ff_bytes_and_empty() {
        assert_eq!(prefix_upper_bound(b""), None);
        assert_eq!(prefix_upper_bound(&[0xFF]), None);
        assert_eq!(prefix_upper_bound(&[0xFF, 0xFF]), None);
        assert_eq!(prefix_upper_bound(b"a"), Some(b"b".to_vec()));
        assert_eq!(prefix_upper_bound(&[b'a', 0xFF]), Some(vec![b'b']));
        assert_eq!(prefix_upper_bound(b"app"), Some(b"apq".to_vec()));
    }

    #[test]
    fn single_term_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        assert_eq!(field.min_term, field.max_term);
        assert_eq!(
            field.seek_exact(b"only"),
            Some(TermStats {
                doc_freq: 1,
                total_term_freq: 1
            })
        );
        assert_eq!(field.seek_exact(b"other"), None);
    }

    #[test]
    fn terms_enum_next_walks_all_terms_in_order() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();
        let mut it = field.iter();
        assert_eq!(it.current(), None);
        assert_eq!(
            it.next(),
            Some((
                b"a".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );
        assert_eq!(
            it.current(),
            Some((
                b"a".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );
        assert_eq!(
            it.next(),
            Some((
                b"ab".as_slice(),
                TermStats {
                    doc_freq: 2,
                    total_term_freq: 3
                }
            ))
        );
        assert_eq!(
            it.next(),
            Some((
                b"b".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );
        assert_eq!(it.next(), None);
        // Idempotent past the end.
        assert_eq!(it.next(), None);
        assert_eq!(it.current(), None);
    }

    #[test]
    fn terms_enum_seek_ceil_found_notfound_end_and_continues() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(
            IndexOptions::DocsAndFreqs,
            &[("a", 1, 1), ("ab", 2, 3), ("b", 1, 1)],
        );
        let fis = FieldInfos {
            fields: vec![field_info(0, "text", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("text").unwrap();

        // Exact match.
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"ab"), SeekStatus::Found);
        assert_eq!(
            it.current(),
            Some((
                b"ab".as_slice(),
                TermStats {
                    doc_freq: 2,
                    total_term_freq: 3
                }
            ))
        );
        // next() after seekCeil continues past the found term.
        assert_eq!(
            it.next(),
            Some((
                b"b".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );

        // Ceiling match: falls strictly between "a" and "ab".
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"aa"), SeekStatus::NotFound);
        assert_eq!(
            it.current(),
            Some((
                b"ab".as_slice(),
                TermStats {
                    doc_freq: 2,
                    total_term_freq: 3
                }
            ))
        );

        // Before the first term: ceiling is the first term.
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b""), SeekStatus::NotFound);
        assert_eq!(
            it.current(),
            Some((
                b"a".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );

        // After the last term: no ceiling exists.
        let mut it = field.iter();
        assert_eq!(it.seek_ceil(b"z"), SeekStatus::End);
        assert_eq!(it.current(), None);
        assert_eq!(it.next(), None);
    }

    #[test]
    fn terms_enum_empty_field() {
        // A real writer never emits a zero-term field (`open()` itself
        // rejects `numTerms <= 0`), but `TermsEnum` over an empty `entries`
        // Vec is a valid state to reach in-memory (e.g. a field with no
        // terms in some hypothetical caller-constructed scenario) and its
        // cursor edge cases are worth covering directly.
        let field = FieldTerms {
            num_terms: 0,
            sum_total_term_freq: 0,
            sum_doc_freq: 0,
            doc_count: 0,
            min_term: Vec::new(),
            max_term: Vec::new(),
            index_options: IndexOptions::Docs,
            has_payloads: false,
            entries: Vec::new(),
        };
        let mut it = field.iter();
        assert_eq!(it.next(), None);
        assert_eq!(it.next(), None);
        assert_eq!(it.current(), None);
        assert_eq!(it.seek_ceil(b"anything"), SeekStatus::End);
        assert_eq!(it.current(), None);
    }

    #[test]
    fn terms_enum_single_term_field() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("only", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        let mut it = field.iter();
        assert_eq!(
            it.next(),
            Some((
                b"only".as_slice(),
                TermStats {
                    doc_freq: 1,
                    total_term_freq: 1
                }
            ))
        );
        assert_eq!(it.next(), None);

        let mut it2 = field.iter();
        assert_eq!(it2.seek_ceil(b"only"), SeekStatus::Found);
        assert_eq!(it2.next(), None);
    }

    #[test]
    fn docs_only_index_options_omits_total_term_freq_field() {
        // IndexOptions::Docs never writes a distinct sumTotalTermFreq, and
        // per-term stats never write the extra totalTermFreq vlong either.
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("x", 3, 3), ("y", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        assert_eq!(field.sum_total_term_freq, field.sum_doc_freq);
        assert_eq!(
            field.seek_exact(b"x"),
            Some(TermStats {
                doc_freq: 3,
                total_term_freq: 3
            })
        );
    }

    #[test]
    fn invalid_num_fields_rejected() {
        let mut tmd = Vec::new();
        let id = [1u8; ID_LENGTH];
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(-1); // invalid numFields
        codec_util::write_footer(&mut tmd);

        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tim);
        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tip);

        let fis = FieldInfos { fields: vec![] };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::InvalidNumFields(-1)));
    }

    #[test]
    fn unexpected_postings_block_size_rejected() {
        let mut tmd = Vec::new();
        let id = [1u8; ID_LENGTH];
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(128); // wrong block size
        codec_util::write_footer(&mut tmd);

        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tim);
        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_footer(&mut tip);

        let fis = FieldInfos { fields: vec![] };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::UnexpectedBlockSize { found: 128 }));
    }

    #[test]
    fn multi_children_node_with_invalid_strategy_code_rejected() {
        // childSaveStrategy code 3 doesn't exist (only 0/1/2 are defined) ->
        // a structural Corrupted error, not silently misdecoded.
        let mut slice = vec![0u8; 24];
        let term: u32 = SIGN_MULTI_CHILDREN | (3 << 9); // invalid strategy code
        slice[0..3].copy_from_slice(&term.to_le_bytes()[0..3]);
        let err = load_node(&slice, 0)
            .and_then(|node| multi_children_labels_and_fps(&slice, &node))
            .unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn single_child_trie_node_with_output_and_floor_round_trips() {
        // Build a leaf child at fp=0, then a SIGN_SINGLE_CHILD_WITH_OUTPUT
        // parent at fp=16 with its own output+floor data, pointing at the
        // child via a 1-byte delta -- exercises loadSingleChildNode's
        // "has output" branch end to end (TrieReader.loadSingleChildNode).
        let mut slice = vec![0u8; 40];
        // Child: SIGN_NO_CHILDREN, 1-byte output fp = 42, hasTerms.
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 42;

        let parent_fp = 16usize;
        let child_delta_fp: u8 = parent_fp as u8; // 16
        let label: u8 = b'x';
        // encodeFP: (floor?1:0) | (hasTerms?2:0) | (fp << 2); output fp = 20.
        let encoded_fp: u64 = NON_LEAF_NODE_HAS_FLOOR | NON_LEAF_NODE_HAS_TERMS | (20 << 2);
        assert!(encoded_fp <= 0xFF, "fits in 1 byte for this test");

        // childDeltaFpBytesMinus1 = 0 (1 byte), encodedOutputFpBytesMinus1 = 0 (1 byte)
        let term: u32 = SIGN_SINGLE_CHILD_WITH_OUTPUT;
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        slice[parent_fp + 1] = label;
        slice[parent_fp + 2] = child_delta_fp;
        slice[parent_fp + 3] = encoded_fp as u8;
        // Floor data right after the 1-byte encoded output fp: one follow
        // block, floorLeadByte='y', code = (5 << 1) | 1 (hasTerms).
        let floor_fp = parent_fp + 4;
        slice[floor_fp] = 1; // numFollowFloorBlocks vint
        slice[floor_fp + 1] = b'y';
        slice[floor_fp + 2] = (5 << 1) | 1; // code vlong

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.sign, SIGN_SINGLE_CHILD_WITH_OUTPUT);
        assert_eq!(node.output_fp, Some(20));
        assert!(node.has_terms);
        assert_eq!(node.floor_data_fp, Some(floor_fp));
        assert_eq!(node.min_children_label, label);

        let blocks = expand_floor(&slice, 20, true, node.floor_data_fp).unwrap();
        assert_eq!(blocks, vec![(20, true), (25, true)]);

        let child_fp = parent_fp - node.child_delta_fp as usize;
        assert_eq!(child_fp, 0);
        let child = load_node(&slice, child_fp).unwrap();
        assert_eq!(child.output_fp, Some(42));
        assert!(child.has_terms);
    }

    #[test]
    fn single_child_without_output_has_no_own_block() {
        let mut slice = vec![0u8; 24];
        let parent_fp = 8usize;
        // childDeltaFpBytesMinus1 = 0 (1 byte)
        let term: u32 = SIGN_SINGLE_CHILD_WITHOUT_OUTPUT;
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        slice[parent_fp + 1] = b'q';
        slice[parent_fp + 2] = 8; // child delta fp -> child at fp 0

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.output_fp, None);
        assert!(!node.has_terms);
        assert_eq!(node.min_children_label, b'q');
    }

    /// Builds a `SIGN_MULTI_CHILDREN` node (no output of its own) with two
    /// children under the given `strategy`, and asserts `multi_children_fps`
    /// recovers exactly the two child fps regardless of which
    /// `ChildSaveStrategy` encoded them -- `TrieReader.lookupChild`'s three
    /// strategies (`BITS`/`ARRAY`/`REVERSE_ARRAY`), generalized to "list all"
    /// (see [`multi_children_labels_and_fps`]'s doc comment for why).
    fn build_and_check_multi_children(strategy: u32, strategy_bytes_region: &[u8]) {
        let mut slice = vec![0u8; 32];
        // Child A: leaf, output fp = 10, hasTerms, at fp 0.
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 10;
        // Child B: leaf, output fp = 20, hasTerms, at fp 2.
        slice[2] = LEAF_NODE_HAS_TERMS as u8;
        slice[3] = 20;

        let parent_fp = 8usize;
        let min_label = b'a';
        let strategy_bytes = strategy_bytes_region.len();
        // childrenDeltaFpBytesMinus1 = 0 (1 byte), no output
        let term: u32 = SIGN_MULTI_CHILDREN
            | (strategy << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());

        let strategy_fp = parent_fp + 3;
        slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(strategy_bytes_region);
        let fps_fp = strategy_fp + strategy_bytes;
        slice[fps_fp] = parent_fp as u8; // delta to child A
        slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.output_fp, None);
        assert_eq!(node.child_save_strategy, strategy);
        assert_eq!(node.strategy_bytes, strategy_bytes);

        let mut child_fps: Vec<usize> = multi_children_labels_and_fps(&slice, &node)
            .unwrap()
            .into_iter()
            .map(|(_, fp)| fp)
            .collect();
        child_fps.sort_unstable();
        assert_eq!(child_fps, vec![0, 2]);

        let mut collected = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut collected).unwrap();
        let mut fps: Vec<u64> = collected.iter().map(|(fp, _)| *fp).collect();
        fps.sort_unstable();
        assert_eq!(fps, vec![10, 20]);
    }

    #[test]
    fn multi_children_array_strategy() {
        // ARRAY: labels[1..] stored explicitly ('b' = 0x62), minLabel='a'
        // implicit.
        build_and_check_multi_children(CHILD_STRATEGY_ARRAY, b"b");
    }

    #[test]
    fn multi_children_bits_strategy() {
        // BITS: byteDistance = 'b'-'a'+1 = 2 -> 1 byte; bit0 (label 'a')
        // and bit1 (label 'b') both set -> 0b011 = 3.
        build_and_check_multi_children(CHILD_STRATEGY_BITS, &[0b011]);
    }

    #[test]
    fn multi_children_reverse_array_strategy() {
        // REVERSE_ARRAY: byte0 = maxLabel ('b'), no missing labels between
        // 'a' and 'b' (they're consecutive) -> exactly 1 byte.
        build_and_check_multi_children(CHILD_STRATEGY_REVERSE_ARRAY, b"b");
    }

    #[test]
    fn multi_children_reverse_array_strategy_with_gap() {
        // Labels 'a' and 'd' (a gap of 'b','c' in between): byteDistance=4,
        // labelCnt=2 -> strategyBytes = 4-2+1 = 3: [maxLabel='d', 'b', 'c'].
        let mut slice = vec![0u8; 32];
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 10;
        slice[2] = LEAF_NODE_HAS_TERMS as u8;
        slice[3] = 20;

        let parent_fp = 8usize;
        let min_label = b'a';
        let strategy_bytes_region = [b'd', b'b', b'c'];
        let strategy_bytes = strategy_bytes_region.len();
        let term: u32 = SIGN_MULTI_CHILDREN
            | (CHILD_STRATEGY_REVERSE_ARRAY << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        let strategy_fp = parent_fp + 3;
        slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(&strategy_bytes_region);
        let fps_fp = strategy_fp + strategy_bytes;
        slice[fps_fp] = parent_fp as u8; // delta to child A (fp 0)
        slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B (fp 2)

        let node = load_node(&slice, parent_fp).unwrap();
        let mut child_fps: Vec<usize> = multi_children_labels_and_fps(&slice, &node)
            .unwrap()
            .into_iter()
            .map(|(_, fp)| fp)
            .collect();
        child_fps.sort_unstable();
        assert_eq!(child_fps, vec![0, 2]);
    }

    #[test]
    fn collect_leaf_blocks_skips_output_with_no_terms() {
        // A node whose own output has hasTerms=false (a pointer-only block,
        // e.g. a coarser prefix the writer recursed past rather than
        // floor-split) contributes no block of its own -- it's skipped, not
        // an error, since any real terms under that prefix are reachable
        // through this node's own children instead (see
        // `collect_leaf_blocks`'s doc comment). Here the node has
        // `SIGN_NO_CHILDREN`, so skipping it means zero blocks collected.
        let mut slice = vec![0u8; 16];
        slice[0] = 0; // sign=0, fpBytesMinus1=0, no LEAF_NODE_HAS_TERMS bit
        slice[1] = 5;
        let node = load_node(&slice, 0).unwrap();
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap();
        assert!(out.is_empty());
    }

    #[test]
    fn expand_floor_rejects_negative_num_follow() {
        let mut buf = Vec::new();
        buf.write_vint(-1);
        let err = expand_floor(&buf, 0, true, Some(0)).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn expand_floor_no_floor_data_returns_just_the_base_block() {
        let blocks = expand_floor(&[], 7, true, None).unwrap();
        assert_eq!(blocks, vec![(7, true)]);
    }

    /// End-to-end: a field whose terms span two floor sub-blocks under one
    /// leaf trie node (`LEAF_NODE_HAS_FLOOR`), exercising `open()`'s full
    /// multi-block merge-and-sort path (not just the trie-decode unit
    /// pieces above) with a hand-built `.tim`/`.tip`/`.tmd` triple no real
    /// (small) fixture reaches.
    #[test]
    fn open_floor_field_merges_two_blocks_in_sorted_order() {
        let id = [9u8; ID_LENGTH];
        let suffix = String::new();

        fn write_leaf_block(tim: &mut Vec<u8>, terms: &[(&str, u32, u64)]) -> usize {
            let block_fp = tim.len();
            let ent_count = terms.len() as u32;
            tim.write_vint(((ent_count << 1) | 1) as i32); // isLastInFloor (unused by decode now)

            let mut suffix_bytes = Vec::new();
            let mut suffix_lengths = Vec::new();
            let mut stats = Vec::new();
            for (term, doc_freq, total_term_freq) in terms {
                suffix_bytes.extend_from_slice(term.as_bytes());
                suffix_lengths.write_vint(term.len() as i32);
                stats.write_vint((*doc_freq as i32) << 1);
                stats.write_vlong((*total_term_freq as i64) - (*doc_freq as i64));
            }
            let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04;
            tim.write_vlong(code_l as i64);
            tim.write_bytes(&suffix_bytes);
            tim.write_vint((suffix_lengths.len() as i32) << 1);
            tim.write_bytes(&suffix_lengths);
            tim.write_vint(stats.len() as i32);
            tim.write_bytes(&stats);

            let mut meta = Vec::new();
            for (_, doc_freq, _) in terms {
                meta.write_vlong(10 << 1);
                if *doc_freq == 1 {
                    meta.write_vint(0);
                }
            }
            tim.write_vint(meta.len() as i32);
            tim.write_bytes(&meta);
            block_fp
        }

        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, &suffix);
        let block0_fp = write_leaf_block(&mut tim, &[("b", 1, 1), ("a", 1, 1)]);
        let block1_fp = write_leaf_block(&mut tim, &[("z", 2, 5), ("m", 1, 1)]);
        codec_util::write_footer(&mut tim);

        let mut tip = Vec::new();
        codec_util::write_index_header(
            &mut tip,
            TERMS_INDEX_CODEC_NAME,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        let index_start = tip.len();
        // Leaf root, floor: header | outputFp(block0_fp, 8 bytes to keep it
        // simple) | floor data.
        let header = LEAF_NODE_HAS_TERMS as u8 | LEAF_NODE_HAS_FLOOR as u8 | (7 << 2);
        tip.push(header);
        tip.extend_from_slice(&(block0_fp as u64).to_le_bytes());
        tip.write_vint(1); // numFollowFloorBlocks
        tip.write_byte(b'm'); // floorLeadByte for block1
        tip.write_vlong((((block1_fp - block0_fp) as i64) << 1) | 1); // code, hasTerms
        tip.extend_from_slice(&0u64.to_le_bytes()); // over-read pad
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        let mut tmd = Vec::new();
        codec_util::write_index_header(
            &mut tmd,
            TERMS_META_CODEC_NAME,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        codec_util::write_index_header(
            &mut tmd,
            POSTINGS_TERMS_CODEC,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1); // numFields
        tmd.write_vint(0); // field number
        tmd.write_vlong(4); // numTerms
        tmd.write_vlong(8); // sumTotalTermFreq = 1+1+1+5
        tmd.write_vlong(5); // sumDocFreq = 1+1+1+2
        tmd.write_vint(1); // docCount
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vint(1);
        tmd.write_bytes(b"z");
        tmd.write_vlong(index_start as i64);
        tmd.write_vlong(0); // root fp within index slice
        tmd.write_vlong(index_end as i64);
        tmd.write_i64(index_end as i64);
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::DocsAndFreqs)],
        };
        let fields = open(&tim, &tip, &tmd, &fis, &id, &suffix, 5).unwrap();
        let field = fields.field("f").unwrap();
        assert_eq!(field.num_terms, 4);

        // Entries must come back sorted even though block1 (containing "m"
        // and "z") is decoded after block0 (containing "b" and "a").
        for (term, expected) in [("a", (1, 1)), ("b", (1, 1)), ("m", (1, 1)), ("z", (2, 5))] {
            let stats = field.seek_exact(term.as_bytes()).unwrap();
            assert_eq!(stats.doc_freq, expected.0, "term={term}");
            assert_eq!(stats.total_term_freq, expected.1, "term={term}");
        }
        assert!(field.seek_exact(b"missing").is_none());
    }

    #[test]
    fn empty_terms_block_rejected() {
        let id = [2u8; ID_LENGTH];
        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        let block_fp = tim.len();
        tim.write_vint(1); // entCount=0, isLastInFloor=true -> code = 0<<1|1 = 1
        codec_util::write_footer(&mut tim);

        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        let index_start = tip.len();
        let header = LEAF_NODE_HAS_TERMS as u8; // SIGN_NO_CHILDREN, 1-byte fp
        tip.push(header);
        tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
        tip.extend_from_slice(&0u64.to_le_bytes());
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        let mut tmd = Vec::new();
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1);
        tmd.write_vint(0);
        tmd.write_vlong(1); // numTerms must be >0 to pass that check; block itself will be empty
        tmd.write_vlong(0); // sumDocFreq (Docs aliasing)
        tmd.write_vint(0); // docCount
        tmd.write_vint(0);
        tmd.write_bytes(&[]);
        tmd.write_vint(0);
        tmd.write_bytes(&[]);
        tmd.write_vlong(index_start as i64);
        tmd.write_vlong(0);
        tmd.write_vlong(index_end as i64);
        tmd.write_i64(index_end as i64);
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn read_bytes_ref_rejects_negative_length() {
        let mut buf = Vec::new();
        buf.write_vint(-1);
        let mut input = SliceInput::new(&buf);
        let err = read_bytes_ref(&mut input).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn load_node_leaf_eight_byte_output_fp() {
        // fpBytesMinus1 == 7 forces a fresh 8-byte read at fp+1.
        let mut slice = Vec::new();
        let header: u8 = LEAF_NODE_HAS_TERMS as u8 | (7 << 2); // sign=0, fpBytesMinus1=7
        slice.push(header);
        let big_fp: u64 = 0x0102_0304_0506_0708;
        slice.extend_from_slice(&big_fp.to_le_bytes()); // read fresh at fp+1
        slice.extend_from_slice(&0u64.to_le_bytes()); // over-read padding

        let node = load_node(&slice, 0).unwrap();
        assert_eq!(node.output_fp, Some(big_fp));
        assert!(node.has_terms);
    }

    #[test]
    fn load_node_rejects_truncated_slice() {
        let slice = [0u8; 4];
        let err = load_node(&slice, 0).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    /// Hand-builds a two-level `.tim` byte sequence: a leaf child block
    /// (term `"zz"`, docFreq/totalTermFreq 1/1) followed by a non-leaf parent
    /// block whose two entries are a real term (`"aa"`) and a sub-block
    /// pointer (key byte `b`) resolving back to the child block via
    /// `parent_fp - subCode` — the genuine "multi-level blocktree" case
    /// (`SegmentTermsEnumFrame.nextNonLeaf`'s `code & 1` sub-block bit),
    /// distinct from the `.tip` trie's own multi-level nesting (already
    /// covered by [`collect_leaf_blocks`]'s tests) and from floor blocks.
    /// Confirms `decode_block` recurses into the sub-block and reattaches its
    /// key byte as a prefix, producing all three terms in the same block's
    /// entry list.
    #[test]
    fn decode_block_recurses_into_sub_block() {
        let mut tim = Vec::new();

        // --- child (leaf) block: one term "zz", docFreq=1/totalTermFreq=1 ---
        let child_fp = tim.len();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
        let child_suffix = b"zz";
        let child_code_l = ((child_suffix.len() as u64) << 3) | 0x04; // leaf, no compression
        tim.write_vlong(child_code_l as i64);
        tim.write_bytes(child_suffix);
        tim.write_vint((1i32 << 1) | 1); // allEqual, logical len 1
        tim.write_byte(2); // suffix length 2
        let mut child_stats = Vec::new();
        child_stats.write_vint(1 << 1); // token&1==0, docFreq=1
        tim.write_vint(child_stats.len() as i32);
        tim.write_bytes(&child_stats);
        let mut child_meta = Vec::new();
        child_meta.write_vlong(10 << 1); // docStartFP delta=10, absolute
        child_meta.write_vint(0); // singleton_doc_id (docFreq==1)
        tim.write_vint(child_meta.len() as i32);
        tim.write_bytes(&child_meta);

        // --- parent (non-leaf) block: term "aa" + sub-block "b" -> child ---
        let parent_fp = tim.len();
        tim.write_vint((2 << 1) | 1); // entCount=2, isLastInFloor
        let parent_suffix_bytes = b"ab"; // "a" (term "aa"'s suffix) then "b" (sub-block key)
        let parent_code_l = (parent_suffix_bytes.len() as u64) << 3; // non-leaf, no compression
        tim.write_vlong(parent_code_l as i64);
        tim.write_bytes(parent_suffix_bytes);

        let mut suffix_lengths = Vec::new();
        suffix_lengths.write_vint(1 << 1); // entry 0: suffix len 1, not a sub-block
        suffix_lengths.write_vint((1 << 1) | 1); // entry 1: suffix len 1, IS a sub-block
        let sub_code = (parent_fp - child_fp) as i64;
        suffix_lengths.write_vlong(sub_code); // entry 1's subCode
        tim.write_vint((suffix_lengths.len() as i32) << 1); // not allEqual
        tim.write_bytes(&suffix_lengths);

        let mut parent_stats = Vec::new();
        parent_stats.write_vint(1 << 1); // entry 0 ("aa"): docFreq=1
        tim.write_vint(parent_stats.len() as i32);
        tim.write_bytes(&parent_stats);

        let mut parent_meta = Vec::new();
        parent_meta.write_vlong(5 << 1); // entry 0's docStartFP delta=5, absolute
        parent_meta.write_vint(0); // singleton_doc_id
        tim.write_vint(parent_meta.len() as i32);
        tim.write_bytes(&parent_meta);

        let entries = decode_block(&tim, parent_fp, IndexOptions::Docs, false).unwrap();
        assert_eq!(entries.len(), 2);
        assert_eq!(entries[0].0, b"a");
        assert_eq!(entries[0].1.doc_freq, 1);
        assert_eq!(entries[1].0, b"bzz");
        assert_eq!(entries[1].1.doc_freq, 1);
        assert_eq!(entries[1].1.total_term_freq, 1);
    }

    #[test]
    fn decode_block_rejects_sub_block_delta_fp_past_parent() {
        let mut tim = Vec::new();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
        let suffix_bytes = b"x";
        let code_l = (suffix_bytes.len() as u64) << 3; // non-leaf
        tim.write_vlong(code_l as i64);
        tim.write_bytes(suffix_bytes);
        let mut suffix_lengths = Vec::new();
        suffix_lengths.write_vint((1 << 1) | 1); // suffix len 1, is a sub-block
        suffix_lengths.write_vlong(1_000_000); // subCode far exceeding this block's own fp
        tim.write_vint((suffix_lengths.len() as i32) << 1);
        tim.write_bytes(&suffix_lengths);
        tim.write_vint(0); // no stat bytes
        tim.write_vint(0); // no meta bytes

        let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn decode_block_rejects_illegal_compression_code() {
        // `code_l & 0x03 == 3` never corresponds to a `CompressionAlgorithm`
        // enum constant (only 0/NO_COMPRESSION, 1/LOWERCASE_ASCII,
        // 2/LZ4 are assigned) -- real Lucene's `CompressionAlgorithm.byCode`
        // throws for it too.
        let mut tim = Vec::new();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor
        tim.write_vlong(0x04 | 0x03); // isLeafBlock, illegal compressionAlg=3
        let err = decode_block(&tim, 0, IndexOptions::Docs, false).unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn load_node_multi_children_with_output_and_floor() {
        // SIGN_MULTI_CHILDREN with its own output+floor data (the
        // `hasOutput`/`NON_LEAF_NODE_HAS_FLOOR` branch of
        // `loadMultiChildrenNode`, not yet exercised by the no-output
        // multi-children tests above).
        let mut slice = vec![0u8; 48];
        let parent_fp = 16usize;
        let min_label = b'a';
        let strategy_bytes_region = [b'b']; // ARRAY: one extra label 'b'.
        let strategy_bytes = strategy_bytes_region.len();
        let encoded_bytes_minus1 = 0u32; // 1-byte encoded output fp.
                                         // childrenDeltaFpBytesMinus1 = 0 (1 byte)
        let term: u32 = SIGN_MULTI_CHILDREN
            | (1 << 5) // has output
            | (encoded_bytes_minus1 << 6)
            | (CHILD_STRATEGY_ARRAY << 9)
            | (((strategy_bytes - 1) as u32) << 11)
            | ((min_label as u32) << 16);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());

        // encodeFP: (floor?1:0) | (hasTerms?2:0) | (fp << 2); output fp = 9.
        let encoded_fp: u64 = NON_LEAF_NODE_HAS_FLOOR | NON_LEAF_NODE_HAS_TERMS | (9 << 2);
        assert!(encoded_fp <= 0xFF);
        // The 3-byte header only fills the low 24 bits of `term`, so byte
        // offset +3 (the word's 4th byte) is already the start of the
        // encoded-output-fp region, not part of the header -- matches
        // `loadMultiChildrenNode`'s `termLong >>> 24` inline read.
        let encoded_fp_off = parent_fp + 3;
        slice[encoded_fp_off] = encoded_fp as u8;

        // "has floor" branch: one byte childrenNum-1, then strategy bytes,
        // then children fps, then floor data.
        let children_num_off = encoded_fp_off + 1;
        slice[children_num_off] = 1; // childrenNum - 1 = 1 -> 2 children
        let strategy_fp = children_num_off + 1;
        slice[strategy_fp..strategy_fp + strategy_bytes].copy_from_slice(&strategy_bytes_region);
        let fps_fp = strategy_fp + strategy_bytes;
        // Two children, both leaf nodes at fp 0 and fp 2.
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 30;
        slice[2] = LEAF_NODE_HAS_TERMS as u8;
        slice[3] = 40;
        slice[fps_fp] = parent_fp as u8; // delta to child A (fp 0)
        slice[fps_fp + 1] = (parent_fp - 2) as u8; // delta to child B (fp 2)
        let floor_fp = fps_fp + 2;
        slice[floor_fp] = 1; // numFollowFloorBlocks
        slice[floor_fp + 1] = b'z';
        slice[floor_fp + 2] = (3 << 1) | 1; // code

        let node = load_node(&slice, parent_fp).unwrap();
        assert_eq!(node.output_fp, Some(9));
        assert!(node.has_terms);
        assert_eq!(node.floor_data_fp, Some(floor_fp));
        assert_eq!(node.strategy_fp, strategy_fp);

        let mut out = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap();
        let mut fps: Vec<u64> = out.iter().map(|(fp, _)| *fp).collect();
        fps.sort_unstable();
        // Own output expands to blocks at fp 9 and fp 9+3=12, plus children
        // at fp 30 and fp 40.
        assert_eq!(fps, vec![9, 12, 30, 40]);
    }

    #[test]
    fn collect_leaf_blocks_rejects_trie_nesting_too_deep() {
        let mut slice = vec![0u8; 16];
        slice[0] = LEAF_NODE_HAS_TERMS as u8;
        slice[1] = 5;
        let node = load_node(&slice, 0).unwrap();
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        let err = collect_leaf_blocks(&slice, &node, 10_001, &mut prefix, &mut out).unwrap_err();
        assert!(matches!(err, Error::Unsupported(_)));
    }

    #[test]
    fn collect_leaf_blocks_rejects_single_child_delta_exceeding_parent_fp() {
        let mut slice = vec![0u8; 16];
        let term: u32 = SIGN_SINGLE_CHILD_WITHOUT_OUTPUT;
        slice[0..4].copy_from_slice(&term.to_le_bytes());
        slice[1] = b'x';
        slice[2] = 100; // child delta fp (100) > parent fp (0)
        let node = load_node(&slice, 0).unwrap();
        let mut out = Vec::new();
        let mut prefix = Vec::new();
        let err = collect_leaf_blocks(&slice, &node, 0, &mut prefix, &mut out).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn multi_children_fps_rejects_delta_exceeding_parent_fp() {
        let mut slice = vec![0u8; 24];
        let parent_fp = 8usize;
        let term: u32 = SIGN_MULTI_CHILDREN | (CHILD_STRATEGY_ARRAY << 9);
        slice[parent_fp..parent_fp + 4].copy_from_slice(&term.to_le_bytes());
        let strategy_fp = parent_fp + 3;
        slice[strategy_fp] = b'b'; // ARRAY strategy, one extra label
        slice[strategy_fp + 1] = 100; // delta (100) > parent fp (8)
        let node = load_node(&slice, parent_fp).unwrap();
        let err = multi_children_labels_and_fps(&slice, &node).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn decode_block_singleton_run_length_and_all_equal_suffixes() {
        // Hand-build a block with allEqual suffix lengths and a singleton
        // run (three consecutive docFreq=1/totalTermFreq=1 terms encoded via
        // the run-length token) to exercise both branches `Builder` (which
        // always emits per-entry non-run tokens and variable suffix
        // lengths) never reaches.
        let mut tim = Vec::new();
        let terms = ["aa", "bb", "cc"];
        let ent_count = terms.len() as u32;
        tim.write_vint(((ent_count << 1) | 1) as i32); // isLastInFloor

        let suffix_bytes: Vec<u8> = terms.iter().flat_map(|t| t.bytes()).collect();
        let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04; // leaf, no compression
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&suffix_bytes);

        // allEqual suffix lengths: all terms are 2 bytes. The logical array
        // size is still entCount (one vint-encoded length per entry) even
        // though only a single physical byte is written on disk.
        tim.write_vint(((ent_count as i32) << 1) | 1);
        tim.write_byte(2);

        // stats: one run-length token covering all three (docFreq=1 each).
        let mut stats = Vec::new();
        stats.write_vint((3 << 1) | 1); // token&1==1 -> singleton run of length 3
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        // Postings metadata: three singleton entries, each via the bit=0
        // (docStartFP-delta) branch of `decode_term_metadata` -- legal
        // whether or not `absolute` is set, unlike the zigzag-delta branch.
        let mut meta = Vec::new();
        for singleton_doc_id in [0i32, 1, 2] {
            meta.write_vlong(0); // docStartFP delta = 0
            meta.write_vint(singleton_doc_id);
        }
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);

        let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
        assert_eq!(entries.len(), 3);
        for (term, stats, _meta) in &entries {
            assert_eq!(term.len(), 2);
            assert_eq!(stats.doc_freq, 1);
            assert_eq!(stats.total_term_freq, 1);
        }
        assert_eq!(entries[0].0, b"aa");
        assert_eq!(entries[2].0, b"cc");
    }

    #[test]
    fn decode_block_lz4_compressed_suffixes() {
        // Hand-built block using `code_l & 0x03 == 2` (LZ4) with the suffix
        // bytes actually run through this port's own `crate::lz4::compress`
        // (a real, general-purpose LZ4 compressor, not a fake/no-op one --
        // see `lz4.rs`'s module doc), then decoded back via `decode_block`'s
        // new LZ4 dispatch arm. This is a hand-built *test vector* (compress
        // + decompress round-trip through this port's own LZ4, cross-checked
        // separately against real Lucene bytes by
        // `tests/blocktree_compressed_fixture.rs`, which decodes an actual
        // `Lucene103BlockTreeTermsWriter`-produced LZ4 block).
        let mut tim = Vec::new();
        let terms = ["aaaaaaaa", "aaaaaaab", "aaaaaaac", "aaaaaaad"];
        let ent_count = terms.len() as u32;
        tim.write_vint(((ent_count << 1) | 1) as i32);

        let suffix_bytes: Vec<u8> = terms.iter().flat_map(|t| t.bytes()).collect();
        let compressed = crate::lz4::compress(&suffix_bytes);
        // Sanity: this input is repetitive enough that LZ4 actually shrinks
        // it -- otherwise this test wouldn't be exercising anything real.
        assert!(compressed.len() < suffix_bytes.len());

        let code_l = ((suffix_bytes.len() as u64) << 3) | 0x04 | 0x02; // leaf, LZ4
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&compressed);

        tim.write_vint(((ent_count as i32) << 1) | 1); // allEqual suffix lengths
        tim.write_byte(8);

        let mut stats = Vec::new();
        stats.write_vint((ent_count << 1 | 1) as i32); // singleton run of length 4
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        let mut meta = Vec::new();
        for singleton_doc_id in 0..ent_count as i32 {
            meta.write_vlong(0);
            meta.write_vint(singleton_doc_id);
        }
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);

        let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
        assert_eq!(entries.len(), 4);
        for (i, (term, stats, _meta)) in entries.iter().enumerate() {
            assert_eq!(term, terms[i].as_bytes());
            assert_eq!(stats.doc_freq, 1);
            assert_eq!(stats.total_term_freq, 1);
        }
    }

    #[test]
    fn decompress_lowercase_ascii_matches_real_lucene_compress_output() {
        // Real Lucene bytes: generated by directly invoking
        // `org.apache.lucene.util.compress.LowercaseAsciiCompression.compress`
        // (from the pinned lucene-core-10.5.0.jar) on the ASCII string below,
        // which mixes lowercase letters, digits, `.`/`-`/`_` (all
        // compressible) with two exceptions (`Z`, `!`, both outside the
        // compressible ranges) to exercise the exception-list decode branch
        // too. Not embedded in an actual on-disk `.tim` block -- see
        // `tests/blocktree_compressed_fixture.rs`'s module doc for why
        // forcing a real `IndexWriter` to choose `LOWERCASE_ASCII` (as
        // opposed to `LZ4` or `NO_COMPRESSION`) for this port's own fixtures
        // wasn't achieved in reasonable effort, and why this vector is the
        // honest fallback for that one mode.
        let original = b"the-quick_brown.fox.jumps_over-42.lazy_dogs.1234567890Z!abcdefghij";
        let compressed_hex = "7569664ef236aaa4aca0a3b3b0b8af8fa7b0b90fab362e3174607077a6b38e95134fad62fbbaa0e53068b4cf125394d5161701365a";
        let compressed: Vec<u8> = (0..compressed_hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&compressed_hex[i..i + 2], 16).unwrap())
            .collect();

        let mut r = SliceInput::new(&compressed);
        let mut out = vec![0u8; original.len()];
        decompress_lowercase_ascii(&mut r, &mut out).unwrap();
        assert_eq!(out, original);
    }

    #[test]
    fn decompress_lowercase_ascii_rejects_out_of_range_exception_index() {
        // Hand-built, not real-Lucene-generated: 4-byte output (saved=1,
        // compressed_len=3), 3 arbitrary packed bytes, then a single
        // exception whose delta (10) pushes the cumulative index to 10,
        // past `out.len()` (4) -- must error before even reading the
        // exception's replacement value byte.
        let compressed: Vec<u8> = vec![0x61, 0x62, 0x63, 0x01, 0x0A];
        let mut r = SliceInput::new(&compressed);
        let mut out = vec![0u8; 4];
        let err = decompress_lowercase_ascii(&mut r, &mut out).unwrap_err();
        assert!(matches!(
            err,
            Error::Store(lucene_store::Error::Corrupted(_))
        ));
    }

    #[test]
    fn decode_block_lowercase_ascii_compressed_suffixes() {
        // Same real-Lucene-generated compressed bytes as the standalone
        // decompress test above, this time threaded through the full
        // `decode_block` dispatch (`code_l & 0x03 == 1`) with a single
        // whole-block suffix rather than per-term suffix lengths (the term
        // boundary doesn't line up with the compression -- LowercaseAscii
        // compresses the concatenated suffix blob as one unit, same as
        // LZ4 -- so this test uses one giant "term" spanning the whole
        // decompressed suffix, which is enough to prove the dispatch wires
        // the compression-alg byte, the decompressed length, and the
        // decoded bytes together correctly).
        let mut tim = Vec::new();
        tim.write_vint((1 << 1) | 1); // entCount=1, isLastInFloor

        let original = b"the-quick_brown.fox.jumps_over-42.lazy_dogs.1234567890Z!abcdefghij";
        let compressed_hex = "7569664ef236aaa4aca0a3b3b0b8af8fa7b0b90fab362e3174607077a6b38e95134fad62fbbaa0e53068b4cf125394d5161701365a";
        let compressed: Vec<u8> = (0..compressed_hex.len())
            .step_by(2)
            .map(|i| u8::from_str_radix(&compressed_hex[i..i + 2], 16).unwrap())
            .collect();

        let code_l = ((original.len() as u64) << 3) | 0x04 | 0x01; // leaf, LOWERCASE_ASCII
        tim.write_vlong(code_l as i64);
        tim.write_bytes(&compressed);

        tim.write_vint((1 << 1) | 1); // allEqual, single entry -> irrelevant, but still 1 length byte
        tim.write_byte(original.len() as u8);

        let mut stats = Vec::new();
        stats.write_vint(1 << 1 | 1); // singleton run of length 1
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);

        let mut meta = Vec::new();
        meta.write_vlong(0);
        meta.write_vint(0);
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);

        let entries = decode_block(&tim, 0, IndexOptions::DocsAndFreqs, false).unwrap();
        assert_eq!(entries.len(), 1);
        assert_eq!(entries[0].0, original);
        assert_eq!(entries[0].1.doc_freq, 1);
        assert_eq!(entries[0].1.total_term_freq, 1);
    }

    #[test]
    fn invalid_field_number_rejected() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        // FieldInfos has no field numbered 0.
        let fis = FieldInfos {
            fields: vec![field_info(9, "other", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 5).unwrap_err();
        assert!(matches!(err, Error::InvalidFieldNumber(0)));
    }

    #[test]
    fn invalid_doc_count_rejected() {
        let b = Builder::new();
        let (tim, tip, tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        // docCount (1, baked into Builder::build) exceeds maxDoc=0.
        let err = open(&tim, &tip, &tmd, &fis, &b.id, &b.suffix, 0).unwrap_err();
        assert!(matches!(err, Error::InvalidDocCount { .. }));
    }

    #[test]
    fn duplicate_field_rejected() {
        let id = [3u8; ID_LENGTH];
        let mut tmd = Vec::new();
        codec_util::write_index_header(&mut tmd, TERMS_META_CODEC_NAME, VERSION_CURRENT, &id, "");
        codec_util::write_index_header(&mut tmd, POSTINGS_TERMS_CODEC, VERSION_CURRENT, &id, "");
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(2); // numFields

        // Build a single shared .tim block (one term "a") and .tip root node
        // that both field records point at, so the same field *name* is
        // reachable twice (two field numbers mapping to fields named "f").
        let mut tim = Vec::new();
        codec_util::write_index_header(&mut tim, TERMS_CODEC_NAME, VERSION_CURRENT, &id, "");
        let block_fp = tim.len();
        tim.write_vint((1 << 1) | 1);
        tim.write_vlong(((1u64 << 3) | 0x04) as i64);
        tim.write_bytes(b"a");
        tim.write_vint(1 << 1);
        tim.write_bytes(&[1]);
        let mut stats = Vec::new();
        stats.write_vint(1 << 1); // docFreq=1, non-singleton token
        tim.write_vint(stats.len() as i32);
        tim.write_bytes(&stats);
        let mut meta = Vec::new();
        meta.write_vlong(0); // docStartFP delta = 0
        meta.write_vint(0); // singletonDocID (docFreq == 1)
        tim.write_vint(meta.len() as i32);
        tim.write_bytes(&meta);
        codec_util::write_footer(&mut tim);

        let mut tip = Vec::new();
        codec_util::write_index_header(&mut tip, TERMS_INDEX_CODEC_NAME, VERSION_CURRENT, &id, "");
        let index_start = tip.len();
        let header = LEAF_NODE_HAS_TERMS as u8;
        tip.push(header);
        tip.extend_from_slice(&(block_fp as u64).to_le_bytes());
        tip.extend_from_slice(&0u64.to_le_bytes());
        let index_end = tip.len();
        codec_util::write_footer(&mut tip);

        for field_number in [0i32, 1i32] {
            tmd.write_vint(field_number);
            tmd.write_vlong(1); // numTerms
            tmd.write_vlong(1); // sumDocFreq (Docs aliasing)
            tmd.write_vint(1); // docCount
            tmd.write_vint(1);
            tmd.write_bytes(b"a");
            tmd.write_vint(1);
            tmd.write_bytes(b"a");
            tmd.write_vlong(index_start as i64);
            tmd.write_vlong(0);
            tmd.write_vlong(index_end as i64);
        }
        tmd.write_i64(index_end as i64);
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![
                field_info(0, "f", IndexOptions::Docs),
                field_info(1, "f", IndexOptions::Docs),
            ],
        };
        let err = open(&tim, &tip, &tmd, &fis, &id, "", 5).unwrap_err();
        assert!(matches!(err, Error::DuplicateField(_)));
    }

    #[test]
    fn index_region_out_of_bounds_rejected() {
        let b = Builder::new();
        let (tim, tip, _tmd) = b.build(IndexOptions::Docs, &[("a", 1, 1)]);
        let id = b.id;
        let suffix = b.suffix.clone();

        // Hand-build a .tmd whose indexEnd points past the end of .tip.
        let mut tmd = Vec::new();
        codec_util::write_index_header(
            &mut tmd,
            TERMS_META_CODEC_NAME,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        codec_util::write_index_header(
            &mut tmd,
            POSTINGS_TERMS_CODEC,
            VERSION_CURRENT,
            &id,
            &suffix,
        );
        tmd.write_vint(POSTINGS_BLOCK_SIZE);
        tmd.write_vint(1);
        tmd.write_vint(0);
        tmd.write_vlong(1);
        tmd.write_vlong(1);
        tmd.write_vint(1);
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vint(1);
        tmd.write_bytes(b"a");
        tmd.write_vlong(0);
        tmd.write_vlong(0);
        tmd.write_vlong((tip.len() + 100) as i64); // out of bounds indexEnd
        tmd.write_i64(tip.len() as i64);
        tmd.write_i64(tim.len() as i64);
        codec_util::write_footer(&mut tmd);

        let fis = FieldInfos {
            fields: vec![field_info(0, "f", IndexOptions::Docs)],
        };
        let err = open(&tim, &tip, &tmd, &fis, &id, &suffix, 5).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    /// Structural proof (not just "lookups still work") that
    /// `fixtures/data/blocktree_multilevel_index/` -- 8000 pseudo-random
    /// terms, regenerated via `fixtures/src/GenBlockTreeMultilevel.java` --
    /// actually forces real Lucene to write a genuine **non-leaf** `.tim`
    /// block (some of its entries are in-block pointers to further-nested
    /// sub-blocks, not raw term suffixes) reachable from this field's `.tip`
    /// trie, i.e. the "root block -> internal block -> leaf block" case this
    /// module's `decode_block`/`decode_block_at_depth` now decode. Walks the
    /// same trie [`collect_leaf_blocks`] would, independently re-deriving
    /// which physical `.tim` blocks are leaf vs. non-leaf by peeking each
    /// one's own `isLeafBlock` bit -- this test would fail (assert
    /// `saw_non_leaf_block`) if a future regen of this fixture, or a change
    /// to real Lucene's own writer heuristics, stopped producing one, which
    /// is exactly the failure mode a purely behavioral "every term still
    /// findable" test could miss (it'd stay green even if this fixture
    /// degenerated to an all-leaf-blocks shape). The full differential
    /// (every term findable via the public API, matching real Lucene's own
    /// ground truth) lives in `crates/lucene-codecs/tests/blocktree_multilevel_fixture.rs`,
    /// same split as every other real-bytes fixture test in this crate:
    /// external test = public-API differential, in-crate test = structural
    /// invariant only reachable with this module's private internals.
    #[test]
    fn multilevel_fixture_reaches_a_genuine_non_leaf_block() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_multilevel_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTreeMultilevel)");
        let kv: std::collections::HashMap<String, String> = manifest
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let read_raw = |name: &str| {
            std::fs::read(format!("{dir}{name}.raw"))
                .unwrap_or_else(|_| panic!("missing {name}.raw"))
        };

        let tmd = read_raw(kv.get("tmd_file_name").unwrap());
        let tip = read_raw(kv.get("tip_file_name").unwrap());
        let tim = read_raw(kv.get("tim_file_name").unwrap());
        let fnm = read_raw(kv.get("fnm_file_name").unwrap());
        let id_hex = kv.get("id_hex").unwrap();
        let mut id = [0u8; ID_LENGTH];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = kv.get("segment_suffix").unwrap();
        let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

        // Re-derive "many"'s index_start/root_fp/index_end the same way
        // `open()` does, so this test doesn't need any new `pub` surface on
        // this module just to expose them.
        let mut tmd_input = SliceInput::new(&tmd);
        codec_util::check_index_header(
            &mut tmd_input,
            TERMS_META_CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        codec_util::check_index_header(
            &mut tmd_input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        let _index_block_size = tmd_input.read_vint().unwrap();
        let num_fields = tmd_input.read_vint().unwrap();
        let mut field_index = None;
        for _ in 0..num_fields {
            let field_number = tmd_input.read_vint().unwrap();
            let _num_terms = tmd_input.read_vlong().unwrap();
            let fi = field_infos.field_by_number(field_number).unwrap();
            read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
            let _doc_count = tmd_input.read_vint().unwrap();
            let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
            let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
            let index_start = tmd_input.read_vlong().unwrap() as usize;
            let root_fp = tmd_input.read_vlong().unwrap() as usize;
            let index_end = tmd_input.read_vlong().unwrap() as usize;
            if fi.name == "many" {
                field_index = Some((index_start, root_fp, index_end));
            }
        }
        let (index_start, root_fp, index_end) = field_index.expect("field \"many\" in .tmd");

        let index_slice = &tip[index_start..index_end];
        let root = load_node(index_slice, root_fp).unwrap();
        // This field's 8000 pseudo-random lowercase terms cover all 26
        // letters at depth 0, so the root's `(minLabel, maxLabel, labelCnt)`
        // is `('a', 'z', 26)` -- a fully dense range, for which
        // `TrieBuilder.ChildSaveStrategy.choose`'s cost formula picks
        // `REVERSE_ARRAY` (`needBytes` = `26 - 26 + 1` = 1, beating both
        // `ARRAY` = 25 and `BITS` = `ceil(26/8)` = 4). This is this fixture's
        // real-Lucene-forced multi-children strategy; `ARRAY` and `BITS` are
        // covered by the dedicated `blocktree_child_strategies_index`
        // fixture instead (see `child_strategies_fixture_forces_array_and_bits_strategies`).
        assert_eq!(root.child_save_strategy, CHILD_STRATEGY_REVERSE_ARRAY);
        assert_eq!(root.strategy_bytes, 1);
        assert_eq!(root.min_children_label, b'a');
        let mut blocks = Vec::new();
        let mut prefix = Vec::new();
        collect_leaf_blocks(index_slice, &root, 0, &mut prefix, &mut blocks).unwrap();
        assert!(
            blocks.len() > 1,
            "expected the trie to reach more than one physical block"
        );

        // Peek each reached block's own isLeafBlock bit directly (the same
        // two reads `decode_block_at_depth` starts with) without doing a
        // full decode -- purely structural.
        let mut saw_non_leaf_block = false;
        for (block_fp, _prefix) in &blocks {
            let mut r = SliceInput::new(&tim);
            r.seek(*block_fp as usize).unwrap();
            let _code = r.read_vint().unwrap();
            let code_l = r.read_vlong().unwrap() as u64;
            if (code_l & 0x04) == 0 {
                saw_non_leaf_block = true;
            }
        }
        assert!(
            saw_non_leaf_block,
            "expected at least one physical .tim block reachable from the \"many\" \
             field's trie to be non-leaf (isLeafBlock == false) -- this fixture is \
             supposed to force real Lucene into a genuine multi-level blocktree \
             (root block -> internal block -> leaf block), not just multiple \
             sibling leaf blocks/floor blocks under one trie node"
        );

        // And the full round trip through the *unmodified* public API must
        // still recover every term correctly despite that non-leaf block
        // (this is the behavioral half; the fuller differential -- matching
        // real Lucene's own sorted term list -- lives in
        // `tests/blocktree_multilevel_fixture.rs`).
        let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
        let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
        let field = fields.field("many").unwrap();
        let num_terms: i64 = kv.get("field.many.numTerms").unwrap().parse().unwrap();
        assert_eq!(field.num_terms, num_terms);
        for (term, stats, _meta) in &field.entries {
            assert_eq!(field.seek_exact(term).unwrap(), *stats);
        }
    }

    /// Real-Lucene-fixture differential test proving `ChildSaveStrategy::ARRAY`
    /// and `ChildSaveStrategy::BITS` (the two of the three real
    /// `TrieBuilder.ChildSaveStrategy` label-encodings that
    /// `multilevel_fixture_reaches_a_genuine_non_leaf_block`'s "many" field
    /// does *not* land on -- that root happens to pick `REVERSE_ARRAY`, see
    /// that test) are each forced onto a real `.tip` trie root node and
    /// decode correctly.
    ///
    /// `fixtures/src/GenBlockTreeChildStrategies.java` builds two fields
    /// whose terms' leading bytes were hand-picked so
    /// `TrieBuilder.ChildSaveStrategy.choose`'s own `needBytes` cost formula
    /// -- BITS: `ceil((maxLabel-minLabel+1)/8)`, ARRAY: `labelCnt-1`,
    /// REVERSE_ARRAY: `(maxLabel-minLabel+1)-labelCnt+1` -- picks a distinct
    /// winner for each field (see that file's module doc for the exact
    /// arithmetic): "arraystrat" (5 labels spanning printable-ASCII, distance
    /// 94: BITS=12, ARRAY=4, REVERSE_ARRAY=90 -> ARRAY wins) and "bitsstrat"
    /// (9 labels spaced 5 apart, distance 41: BITS=6, ARRAY=8,
    /// REVERSE_ARRAY=33 -> BITS wins). This test decodes each field's root
    /// trie node and asserts the exact `child_save_strategy` code real
    /// Lucene's writer chose, then round-trips every term through the
    /// public `open`/`seek_exact` API to prove the decode is not just
    /// structurally plausible but actually correct.
    #[test]
    fn child_strategies_fixture_forces_array_and_bits_strategies() {
        let dir = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/blocktree_child_strategies_index/"
        );
        let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
            .expect("run fixtures generator first (GenBlockTreeChildStrategies)");
        let kv: std::collections::HashMap<String, String> = manifest
            .lines()
            .filter_map(|l| l.split_once('='))
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let read_raw = |name: &str| {
            std::fs::read(format!("{dir}{name}.raw"))
                .unwrap_or_else(|_| panic!("missing {name}.raw"))
        };
        let tmd = read_raw(kv.get("tmd_file_name").unwrap());
        let tip = read_raw(kv.get("tip_file_name").unwrap());
        let tim = read_raw(kv.get("tim_file_name").unwrap());
        let fnm = read_raw(kv.get("fnm_file_name").unwrap());
        let id_hex = kv.get("id_hex").unwrap();
        let mut id = [0u8; ID_LENGTH];
        for (i, slot) in id.iter_mut().enumerate() {
            *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
        }
        let suffix = kv.get("segment_suffix").unwrap();
        let field_infos = crate::field_infos::parse(&fnm, &id, "").unwrap();

        let mut tmd_input = SliceInput::new(&tmd);
        codec_util::check_index_header(
            &mut tmd_input,
            TERMS_META_CODEC_NAME,
            VERSION_START,
            VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        codec_util::check_index_header(
            &mut tmd_input,
            POSTINGS_TERMS_CODEC,
            POSTINGS_VERSION_START,
            POSTINGS_VERSION_CURRENT,
            &id,
            suffix,
        )
        .unwrap();
        let _index_block_size = tmd_input.read_vint().unwrap();
        let num_fields = tmd_input.read_vint().unwrap();
        let mut field_index: std::collections::HashMap<String, (usize, usize, usize)> =
            std::collections::HashMap::new();
        for _ in 0..num_fields {
            let field_number = tmd_input.read_vint().unwrap();
            let _num_terms = tmd_input.read_vlong().unwrap();
            let fi = field_infos.field_by_number(field_number).unwrap();
            read_freq_pair(&mut tmd_input, fi.index_options).unwrap();
            let _doc_count = tmd_input.read_vint().unwrap();
            let _min_term = read_bytes_ref(&mut tmd_input).unwrap();
            let _max_term = read_bytes_ref(&mut tmd_input).unwrap();
            let index_start = tmd_input.read_vlong().unwrap() as usize;
            let root_fp = tmd_input.read_vlong().unwrap() as usize;
            let index_end = tmd_input.read_vlong().unwrap() as usize;
            field_index.insert(fi.name.clone(), (index_start, root_fp, index_end));
        }

        // "arraystrat": 5 labels, distance 94 (0x21..=0x7e) -> ARRAY (code 1).
        let (index_start, root_fp, index_end) = field_index["arraystrat"];
        let root = load_node(&tip[index_start..index_end], root_fp).unwrap();
        assert_eq!(
            root.child_save_strategy, CHILD_STRATEGY_ARRAY,
            "expected real Lucene to pick ChildSaveStrategy.ARRAY for \"arraystrat\"'s \
             root node (needBytes: BITS=12, ARRAY=4, REVERSE_ARRAY=90)"
        );
        assert_eq!(root.strategy_bytes, 4); // labelCnt - 1 = 5 - 1
        assert_eq!(root.min_children_label, 0x21);

        // "bitsstrat": 9 labels, distance 41 (0x21..=0x49) -> BITS (code 2).
        let (index_start, root_fp, index_end) = field_index["bitsstrat"];
        let root = load_node(&tip[index_start..index_end], root_fp).unwrap();
        assert_eq!(
            root.child_save_strategy, CHILD_STRATEGY_BITS,
            "expected real Lucene to pick ChildSaveStrategy.BITS for \"bitsstrat\"'s \
             root node (needBytes: BITS=6, ARRAY=8, REVERSE_ARRAY=33)"
        );
        assert_eq!(root.strategy_bytes, 6); // ceil(41 / 8)
        assert_eq!(root.min_children_label, 0x21);

        // Full round trip through the unmodified public API: every term in
        // both fields must be findable via seek_exact, proving the ARRAY and
        // BITS label decodes (not just the strategy *code*) are correct.
        let max_doc: i32 = kv.get("max_doc").unwrap().parse().unwrap();
        let fields = open(&tim, &tip, &tmd, &field_infos, &id, suffix, max_doc).unwrap();
        for name in ["arraystrat", "bitsstrat"] {
            let field = fields.field(name).unwrap();
            let expected_count: i64 = kv
                .get(&format!("field.{name}.count"))
                .unwrap()
                .parse()
                .unwrap();
            assert_eq!(field.num_terms, expected_count);
            assert_eq!(field.entries.len() as i64, expected_count);
            let terms_tsv = std::fs::read_to_string(format!("{dir}{name}.terms.tsv")).unwrap();
            let expected_terms: Vec<&str> = terms_tsv.lines().collect();
            assert_eq!(expected_terms.len() as i64, expected_count);
            for term in &expected_terms {
                let stats = field
                    .seek_exact(term.as_bytes())
                    .unwrap_or_else(|| panic!("term {term:?} not found in field {name}"));
                assert_eq!(stats.doc_freq, 1);
                assert_eq!(stats.total_term_freq, 1);
            }
            for (term, stats, _meta) in &field.entries {
                assert_eq!(field.seek_exact(term).unwrap(), *stats);
            }
        }
    }
}
