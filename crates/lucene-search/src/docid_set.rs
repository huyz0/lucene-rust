//! Minimal `DocIdSetIterator`-shaped merge combinators (`org.apache.lucene.search.
//! DocIdSetIterator`/`ConjunctionDISI`/`DisjunctionDISIApproximation`), pared down to
//! this slice's scope: given several already-materialized, ascending, duplicate-free
//! doc-ID sequences (one per clause), merge them into the AND (conjunction), OR
//! (disjunction), or AND-NOT (exclusion) result, still ascending and duplicate-free.
//!
//! **Why plain `Iterator<Item = i32>` instead of a bespoke `next_doc`/`advance` trait**:
//! Rust's `Iterator` already *is* a pull-based "give me the next doc ID or tell me
//! you're done" cursor — `Option<i32>` is `NO_MORE_DOCS` for free, `Peekable` gives the
//! one extra primitive (look-ahead without consuming) every merge algorithm below
//! needs, and every combinator composes with the rest of `std` (`.collect()`,
//! `for doc in iter`) with no adapter layer. Inventing a parallel `DocIdSet` trait with
//! its own `next_doc`/`advance` methods would just be re-deriving `Iterator` by hand —
//! transliterating Java's shape instead of using the idiomatic Rust one, which
//! `rust-performance` explicitly warns against.
//!
//! **Why `Box<dyn Iterator<Item = i32>>` instead of monomorphized generics**: `must`/
//! `should`/`must_not` are runtime-length `Vec<TermQuery>` (a `BooleanQuery` might have
//! two clauses or twenty), so the number and concrete type of per-clause iterators
//! being merged isn't known at compile time — there is no closed set of shapes to
//! monomorphize over the way a fixed 2-way merge would allow. `rust-performance`'s
//! "monomorphize per-doc loops" guidance is aimed at *scorers/DISIs* on the hot
//! single-query-type path; a boxed trait object at the clause-merge boundary (one
//! virtual call per doc per clause, not per byte) is the same tradeoff real Lucene's
//! own `Scorer`-hierarchy conjunctions make, and is the right place to pay it. This is
//! also explicitly a first cut ("correctness first, not final perf" per the task that
//! introduced it) — the merge algorithms below are the simple standard ones
//! (leapfrog conjunction, min-scan disjunction), not the skip-list-driven versions a
//! later performance pass would swap in.
//!
//! Every doc-ID sequence handed to these combinators is expected to already be
//! **ascending and duplicate-free** — [`crate::term_doc_ids`] (or any other producer)
//! is responsible for that invariant, same as real Lucene's `DocIdSetIterator` contract.
//!
//! ## Materialized sets: [`RoaringDocIdSet`] and [`CachedDocIdSet`]
//!
//! The combinators above are *streams*. The second half of this module is the
//! *storage* half — `org.apache.lucene.util.RoaringDocIdSet` and the
//! bitset-vs-Roaring choice `LRUQueryCache.cacheImpl` makes — for the one place
//! this port has to hold a whole query's match set in memory over time rather
//! than consume it once: [`crate::query_cache`]. Storing a `FixedBitSet` there
//! costs `maxDoc / 8` bytes per cached `(segment, query)` pair no matter how few
//! documents matched; Roaring's per-65536-doc-block encoding makes the cost track
//! the match count instead. See [`RoaringDocIdSet`] for the block encodings.

use lucene_util::fixed_bit_set::FixedBitSet;
use std::iter::Peekable;

/// A boxed, type-erased doc-ID sequence — see the module doc for why this shape
/// (dynamic clause count) beats a monomorphized alternative here.
pub type BoxDocIter<'a> = Box<dyn Iterator<Item = i32> + 'a>;

/// AND across every wrapped clause: a doc is emitted only when **all** clauses agree
/// on it. Standard leapfrog: track the current maximum among all clauses' peeked
/// heads, fast-forward every clause whose head is behind that maximum, and repeat
/// until either all clauses agree (emit) or one is exhausted (done) — the same
/// algorithm as `ConjunctionDISI.doNext` minus its two-phase-iterator special case
/// (no `TwoPhaseIterator` exists in this port yet).
pub struct Conjunction<'a> {
    iters: Vec<Peekable<BoxDocIter<'a>>>,
}

impl<'a> Conjunction<'a> {
    /// Builds the conjunction over `iters`. An empty `iters` list matches nothing
    /// (mirrors this port's `search_boolean_query`, which never constructs a
    /// `Conjunction` with zero `must` clauses in the first place — a `BooleanQuery`
    /// with no `must`/`should` clauses at all is rejected before reaching here, see
    /// that function's doc comment) — included as a defined, tested edge case rather
    /// than a panic.
    pub fn new(iters: Vec<BoxDocIter<'a>>) -> Self {
        Self {
            iters: iters.into_iter().map(Iterator::peekable).collect(),
        }
    }
}

impl Iterator for Conjunction<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if self.iters.is_empty() {
            return None;
        }
        loop {
            let mut max = i32::MIN;
            for it in &mut self.iters {
                match it.peek() {
                    Some(&v) => max = max.max(v),
                    None => return None,
                }
            }

            let mut all_match = true;
            for it in &mut self.iters {
                while it.peek().is_some_and(|&v| v < max) {
                    it.next();
                }
                match it.peek() {
                    Some(&v) if v == max => {}
                    Some(_) => all_match = false,
                    None => return None,
                }
            }

            if all_match {
                for it in &mut self.iters {
                    it.next();
                }
                return Some(max);
            }
        }
    }
}

/// OR across every wrapped clause: a doc is emitted once if **any** clause matches
/// it, even if several clauses share it (dedup happens by construction: every
/// clause currently peeking the emitted minimum is advanced past it in the same
/// step, so no clause can re-offer that doc). Simple min-scan over all clauses'
/// peeked heads per step — `DisjunctionDISIApproximation`'s min-heap does the same
/// thing in `O(log n)` per step instead of this port's `O(n)`, which is fine
/// for the clause counts queries have in practice.
pub struct Disjunction<'a> {
    iters: Vec<Peekable<BoxDocIter<'a>>>,
}

impl<'a> Disjunction<'a> {
    /// Builds the disjunction over `iters`. Like [`Conjunction::new`], an empty list
    /// is a defined "matches nothing" edge case, not a panic.
    pub fn new(iters: Vec<BoxDocIter<'a>>) -> Self {
        Self {
            iters: iters.into_iter().map(Iterator::peekable).collect(),
        }
    }
}

impl Iterator for Disjunction<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        let mut min: Option<i32> = None;
        for it in &mut self.iters {
            if let Some(&v) = it.peek() {
                if min.is_none_or(|m| v < m) {
                    min = Some(v);
                }
            }
        }
        let min = min?;
        for it in &mut self.iters {
            if it.peek() == Some(&min) {
                it.next();
            }
        }
        Some(min)
    }
}

/// Port of `BooleanScorer`'s window/bucket bulk OR: the disjunction mechanism
/// real Lucene uses instead of a document-at-a-time priority-queue merge.
///
/// ## What Java does
///
/// `BooleanScorer` (`SHIFT = 12`, so `SIZE = 4096` documents per window) walks
/// one window at a time. For each window it lets every clause pour its whole run
/// of doc ids into a `FixedBitSet` -- `scoreWindowIntoBitSetAndReplay` -- and,
/// when `minShouldMatch > 1` or scores are needed, into a parallel `Bucket[]`
/// carrying that document's clause count. It then replays the window by walking
/// the bitset's words with `Long.numberOfTrailingZeros`, emitting the documents
/// whose bucket cleared `minShouldMatch`. `BooleanScorerSupplier.booleanScorer`
/// picks it for a query with no required clauses and more than one optional
/// clause.
///
/// The win is memory access pattern, not complexity: the same total postings are
/// read either way, but each clause is walked in a long contiguous run instead of
/// being interleaved with every other clause, and the per-document "which clause
/// is next" decision disappears inside the window.
///
/// ## What this is
///
/// The same mechanism over this port's already-materialized per-clause doc-id
/// lists (see this module's doc comment for why clauses arrive materialized), and
/// pull-shaped rather than push-shaped so it composes with [`Excluding`] and the
/// rest of this module exactly as [`Disjunction`] does. It replaces two things at
/// once:
///
/// - [`Disjunction`]'s `O(clauses)` min-scan **per emitted document**, each step
///   of which is a `peek()`/`next()` through a `Box<dyn Iterator>`; and
/// - `crate::should_match_counts`' `HashMap<i32, usize>` tally over **the whole
///   segment**, which `minimum_should_match > 1` needed to know how many clauses
///   agreed on a document. Here that count is a `u16` at a window-relative index,
///   with no hashing and no per-document allocation.
///
/// ## Two deliberate divergences from `BooleanScorer`
///
/// 1. **Only non-empty windows are visited.** The next window's base is taken
///    from the smallest doc id any clause still has (`top.doc & ~MASK` in Java's
///    `scoreWindow`), so a run of empty windows costs nothing.
/// 2. **Only the touched word range is cleared and replayed.** Java clears all
///    64 words of `matching` and walks all 64 per window; this tracks the lowest
///    and highest word a clause actually set. That is a fixed per-window
///    overhead Java can afford because its priority queue only ever hands it a
///    window with a match in it; tracking the range keeps a very sparse
///    disjunction from paying 64 word writes per emitted document.
///
/// ## Why Java's density gate is not ported
///
/// `BooleanScorerSupplier.booleanScorer` refuses `BooleanScorer` when
/// `minShouldMatch > 1` and `cost() < maxDoc / 3`. **Java's stated reason applies
/// here verbatim** and the two divergences above do not address it: with a
/// minimum-should-match there is no way to know whether the clauses intersect
/// inside a window, so a window's postings can be poured in and yield nothing.
///
/// What does not transfer is the *choice* the gate expresses. Java is choosing
/// between `BooleanScorer`, which reads every posting of every clause, and
/// `MinShouldMatchSumScorer` (via `BS2`), which **leapfrogs** -- it can advance
/// past postings without reading them, so on a sparse query it does strictly
/// less I/O and the gate is picking the cheaper of two real options. This port's
/// alternative for `minimum_should_match > 1` is `crate::should_match_counts`,
/// which reads every posting of every clause *and* hashes each one into a
/// whole-segment `HashMap<i32, usize>` before a per-document min-scan. It is
/// worse on every axis at every density, so there is no trade-off left to gate
/// on and the windowed path is taken unconditionally.
///
/// **This becomes wrong the moment a leapfrogging min-should-match scorer
/// exists in this port.** Adding one means restoring the gate, with Java's
/// threshold and Java's reason.
pub struct WindowedDisjunction {
    /// One clause's remaining doc ids: the list and the position in it.
    clauses: Vec<(Vec<i32>, usize)>,
    /// Java's `minShouldMatch`. `0` and `1` both mean "any clause", which is
    /// what `BooleanQuery`'s `Math.max(1, minShouldMatch)` normalizes to.
    min_should_match: usize,
    /// Java's `matching`: one bit per document in the window, as a flat
    /// 64-word array rather than a [`FixedBitSet`], because the replay loop
    /// zeroes whole words as it consumes them and that is not something a
    /// bit-at-a-time `clear` should be asked to express.
    matching: [u64; WINDOW_WORDS],
    /// Java's `Bucket.freq`, empty when `min_should_match <= 1` (Java's
    /// `buckets == null` case, where a set bit is already the whole answer).
    ///
    /// `u16` caps the clause count at 65,535. `BooleanQuery`'s own
    /// `maxClauseCount` defaults to 1,024 and `matched_boolean_docs` is the only
    /// caller, so nothing in this port can reach it -- but
    /// [`WindowedDisjunction::new`] is `pub` and does not enforce it, which is
    /// why it is written down here rather than assumed.
    freqs: Vec<u16>,
    /// Documents of the window already replayed out of `matching`, in ascending
    /// order. Reused across windows; never reallocates after the first full one.
    ready: Vec<i32>,
    ready_pos: usize,
}

/// Java's `BooleanScorer.SHIFT`.
const WINDOW_SHIFT: u32 = 12;
/// Java's `BooleanScorer.SIZE`: 4,096 documents per window.
const WINDOW_SIZE: usize = 1 << WINDOW_SHIFT;
/// Java's `BooleanScorer.MASK`.
const WINDOW_MASK: i32 = (WINDOW_SIZE - 1) as i32;
/// 64-bit words in one window's `matching` bitmap.
const WINDOW_WORDS: usize = WINDOW_SIZE / 64;

impl WindowedDisjunction {
    /// `clause_docs`: one ascending, duplicate-free doc-id list per optional
    /// clause. `min_should_match`: how many of them must agree on a document
    /// (`0`/`1` both mean "any", as `BooleanQuery` normalizes them).
    ///
    /// An empty clause list is a defined "matches nothing", not a panic --
    /// same contract as [`Disjunction::new`].
    pub fn new(clause_docs: Vec<Vec<i32>>, min_should_match: usize) -> Self {
        let counting = min_should_match > 1;
        Self {
            clauses: clause_docs.into_iter().map(|d| (d, 0usize)).collect(),
            min_should_match,
            matching: [0u64; WINDOW_WORDS],
            freqs: if counting {
                vec![0u16; WINDOW_SIZE]
            } else {
                Vec::new()
            },
            ready: Vec::new(),
            ready_pos: 0,
        }
    }

    /// Whether this shape is the one `BooleanScorerSupplier.booleanScorer`
    /// would pick: no required clauses (the caller's precondition, not checked
    /// here) and **more than one** optional clause -- with a single clause there
    /// is nothing to OR and `BooleanScorer`'s own constructor rejects it
    /// (`"can only be used with two scorers or more"`), so the plain path stays.
    pub fn is_applicable(clause_count: usize) -> bool {
        clause_count > 1
    }

    /// Java's `scoreWindowIntoBitSetAndReplay`, plus `scoreWindow`'s choice of
    /// which window to visit next. Fills `ready` with the next non-empty
    /// window's qualifying documents; `false` once every clause is exhausted.
    fn fill_next_window(&mut self) -> bool {
        loop {
            // `scoreWindow`: the window is the one the next match belongs to,
            // so empty windows are never visited.
            let Some(min_doc) = self
                .clauses
                .iter()
                .filter_map(|(docs, pos)| docs.get(*pos).copied())
                .min()
            else {
                return false;
            };
            let base = min_doc & !WINDOW_MASK;
            // `i64`, deliberately. Java's `windowMax` is
            // `Math.min(max, windowBase + SIZE)`, bounded by the caller's `max`
            // and therefore by `maxDoc`, so `windowBase + SIZE` can never reach
            // `Integer.MAX_VALUE`. This is pull-shaped and has no `max`
            // argument, so it dropped that bound -- and an `i32`
            // `base + WINDOW_SIZE` for `min_doc == i32::MAX` is `0x8000_0000`,
            // which overflows. Saturating it to `i32::MAX` made `doc >= max`
            // true for the very document that chose the window, so no clause
            // advanced, `ready` stayed empty, and the loop re-derived the same
            // `min_doc` **forever**. Widening removes the cliff instead of
            // clamping at it.
            let max = base as i64 + WINDOW_SIZE as i64;

            let counting = self.min_should_match > 1;
            // Lowest/highest word of `matching` any clause touched, so the
            // replay walk and the clear are both bounded by what was written.
            let mut lo_word = usize::MAX;
            let mut hi_word = 0usize;

            for (docs, pos) in self.clauses.iter_mut() {
                // One long contiguous run per clause -- the whole point.
                while let Some(&doc) = docs.get(*pos) {
                    if doc as i64 >= max {
                        break;
                    }
                    debug_assert!(
                        doc >= base,
                        "clause doc-id lists must be ascending and duplicate-free \
                         (see this module's doc comment): doc {doc} is below the \
                         window base {base}, which means an earlier document was \
                         emitted after it"
                    );
                    let index = (doc - base) as usize;
                    self.matching[index >> 6] |= 1u64 << (index & 63);
                    if counting {
                        self.freqs[index] += 1;
                    }
                    let word = index >> 6;
                    lo_word = lo_word.min(word);
                    hi_word = hi_word.max(word);
                    *pos += 1;
                }
            }

            self.ready.clear();
            self.ready_pos = 0;
            if lo_word != usize::MAX {
                // Java's replay loop: walk the words, `numberOfTrailingZeros`
                // out each set bit, and zero the bucket as it goes.
                for word_index in lo_word..=hi_word {
                    let mut bits = self.matching[word_index];
                    // `matching.clear()`, done word by word as the word is
                    // consumed, so nothing outside the touched range is touched.
                    self.matching[word_index] = 0;
                    while bits != 0 {
                        let ntz = bits.trailing_zeros() as usize;
                        let index = (word_index << 6) | ntz;
                        if !counting || self.freqs[index] as usize >= self.min_should_match {
                            self.ready.push(base + index as i32);
                        }
                        if counting {
                            self.freqs[index] = 0;
                        }
                        bits ^= 1u64 << ntz;
                    }
                }
            }

            if !self.ready.is_empty() {
                return true;
            }
            // `minimum_should_match` rejected every document in this window;
            // Java simply collects nothing and moves on, as this does.
        }
    }
}

impl Iterator for WindowedDisjunction {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        if self.ready_pos == self.ready.len() && !self.fill_next_window() {
            return None;
        }
        let doc = self.ready[self.ready_pos];
        self.ready_pos += 1;
        Some(doc)
    }
}

/// AND-NOT: every doc from `base` that does **not** appear in `excluded` — the
/// `must_not` clause set's effect (`Occur.MUST_NOT`), applied as a final filter over
/// whatever `base` (a [`Conjunction`] or [`Disjunction`] of the query's `must`/
/// `should` clauses) already produced. Advances `excluded` in lockstep with `base`
/// rather than re-scanning from the start each time, since both sequences are
/// ascending.
pub struct Excluding<'a> {
    base: BoxDocIter<'a>,
    excluded: Peekable<BoxDocIter<'a>>,
}

impl<'a> Excluding<'a> {
    pub fn new(base: BoxDocIter<'a>, excluded: BoxDocIter<'a>) -> Self {
        Self {
            base,
            excluded: excluded.peekable(),
        }
    }
}

impl Iterator for Excluding<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        for doc in self.base.by_ref() {
            while self.excluded.peek().is_some_and(|&v| v < doc) {
                self.excluded.next();
            }
            if self.excluded.peek() != Some(&doc) {
                return Some(doc);
            }
        }
        None
    }
}

// ---------------------------------------------------------------------------
// Materialized doc-ID sets (`org.apache.lucene.search.DocIdSet` and its
// implementations), as distinct from the merge combinators above.
// ---------------------------------------------------------------------------

/// Number of documents one Roaring block covers (`RoaringDocIdSet.BLOCK_SIZE`).
const BLOCK_SIZE: usize = 1 << 16;

/// Beyond this many documents in a block, the short-array encoding stops paying
/// for itself and a bitset is used instead (`RoaringDocIdSet.MAX_ARRAY_LENGTH`).
const MAX_ARRAY_LENGTH: usize = 1 << 12;

/// The next set bit at or after `*next`, advancing `*next` past it -- a
/// word-at-a-time scan over [`FixedBitSet::words`], so walking a bitset costs
/// `O(maxDoc / 64)` plus one step per set bit rather than one bounds-checked
/// `get` per document. `FixedBitSet` exposes no such iterator itself
/// (`lucene-util` is another sweep batch's file), so it lives here, where both
/// consumers -- building a Roaring set from a bitset and iterating a cached
/// bitset -- are.
fn next_set_bit(bits: &FixedBitSet, next: &mut usize) -> Option<usize> {
    let words = bits.words();
    let mut w = *next / 64;
    let mut mask = if (*next).is_multiple_of(64) {
        u64::MAX
    } else {
        u64::MAX << (*next % 64)
    };
    while w < words.len() {
        let masked = words[w] & mask;
        if masked != 0 {
            let bit = w * 64 + masked.trailing_zeros() as usize;
            *next = bit + 1;
            return Some(bit);
        }
        w += 1;
        mask = u64::MAX;
    }
    *next = words.len() * 64;
    None
}

/// One 65536-document block's encoding, mirroring the five shapes
/// `RoaringDocIdSet.Builder.flush` picks between (plus the "block absent"
/// case Java represents with a `null` slot in its `DocIdSet[]`).
#[derive(Debug, Clone)]
enum Block {
    /// No document in this block matches (Java: a `null` slot).
    Empty,
    /// Every document in the block matches (Java: `AllDocIdSet`).
    All,
    /// A contiguous run `min..=max`, block-local (Java: `RangeDocIdSet`).
    Range { min: u16, max: u16 },
    /// Fewer than [`MAX_ARRAY_LENGTH`] docs, stored block-local and ascending
    /// (Java: `ShortArrayDocIdSet`).
    Sparse(Box<[u16]>),
    /// More than `BLOCK_SIZE - MAX_ARRAY_LENGTH` docs in a *full* block: the
    /// complement is what is stored (Java: `NotDocIdSet` over a
    /// `ShortArrayDocIdSet`).
    InverseSparse(Box<[u16]>),
    /// Neither sparse nor super-dense: a block-local bitset (Java:
    /// `BitDocIdSet`).
    Dense(FixedBitSet),
}

/// `FixedBitSet` has no `PartialEq`, so the dense arm compares its contents
/// directly. Two blocks are equal when they encode the same document set *in
/// the same encoding* -- which is what the builder-encoding tests need to
/// assert, and is why this isn't a set-equality relation across encodings.
impl PartialEq for Block {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Block::Empty, Block::Empty) | (Block::All, Block::All) => true,
            (
                Block::Range { min: a, max: b },
                Block::Range {
                    min: other_min,
                    max: other_max,
                },
            ) => a == other_min && b == other_max,
            (Block::Sparse(a), Block::Sparse(b))
            | (Block::InverseSparse(a), Block::InverseSparse(b)) => a == b,
            (Block::Dense(a), Block::Dense(b)) => a.len() == b.len() && a.words() == b.words(),
            _ => false,
        }
    }
}

impl Eq for Block {}

impl Block {
    /// Heap bytes this block's payload costs, for the cache's RAM accounting
    /// (`Accountable.ramBytesUsed`). Only the payload is counted; the enum
    /// discriminant itself is part of the `Vec<Block>` the owner already
    /// charges for.
    fn payload_bytes(&self) -> usize {
        match self {
            Block::Empty | Block::All | Block::Range { .. } => 0,
            Block::Sparse(docs) | Block::InverseSparse(docs) => docs.len() * 2,
            Block::Dense(bits) => bits.words().len() * 8,
        }
    }
}

/// A memory-compact materialized doc-ID set: a port of
/// `org.apache.lucene.util.RoaringDocIdSet`.
///
/// The doc-ID space is cut into 65536-document blocks and each block is encoded
/// on its own, so the cost of remembering a query's matches tracks the number of
/// matches rather than `maxDoc`. A [`FixedBitSet`] over a 10M-document segment
/// costs 1.25 MB whether 10 documents matched or 10 million; the same 10 matches
/// here cost one block header plus 20 bytes.
///
/// The block encodings and their thresholds are Java's, not re-derived: all-set
/// and contiguous-run blocks are free, up to [`MAX_ARRAY_LENGTH`] documents are
/// a `u16` array, a full block missing fewer than [`MAX_ARRAY_LENGTH`] documents
/// stores its complement instead, and anything else is a block-local bitset.
///
/// Documents must be added in strictly ascending order, exactly as
/// `RoaringDocIdSet.Builder.add` requires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RoaringDocIdSet {
    blocks: Vec<Block>,
    cardinality: usize,
}

impl RoaringDocIdSet {
    /// The in-line size of one block slot, which [`Self::ram_bytes_used`]
    /// charges per block whatever its encoding: what a test-side heap walk
    /// (`RamUsageTester`) needs to measure a set it cannot see inside.
    pub const BLOCK_SLOT_BYTES: usize = std::mem::size_of::<Block>();

    /// How many documents are in the set.
    pub fn cardinality(&self) -> usize {
        self.cardinality
    }

    pub fn is_empty(&self) -> bool {
        self.cardinality == 0
    }

    /// Approximate heap cost, in bytes -- the analogue of
    /// `RoaringDocIdSet.ramBytesUsed()`, used by [`crate::query_cache`] to
    /// bound itself by memory the way `LRUQueryCache`'s `maxRamBytesUsed`
    /// does.
    pub fn ram_bytes_used(&self) -> usize {
        std::mem::size_of::<Self>()
            + self.blocks.len() * std::mem::size_of::<Block>()
            + self.blocks.iter().map(Block::payload_bytes).sum::<usize>()
    }

    /// Whether `doc` is in the set. Not something Java's `RoaringDocIdSet`
    /// offers (its `DocIdSet` contract is iterate-only), but the natural
    /// random-access primitive a `live_docs`-style consumer wants, and cheap
    /// for every encoding here (`O(1)` except the two array shapes, which
    /// binary-search).
    pub fn contains(&self, doc: i32) -> bool {
        if doc < 0 {
            return false;
        }
        let doc = doc as usize;
        let block = doc >> 16;
        let local = (doc & 0xFFFF) as u16;
        match self.blocks.get(block) {
            None | Some(Block::Empty) => false,
            Some(Block::All) => true,
            Some(Block::Range { min, max }) => local >= *min && local <= *max,
            Some(Block::Sparse(docs)) => docs.binary_search(&local).is_ok(),
            Some(Block::InverseSparse(excluded)) => excluded.binary_search(&local).is_err(),
            // A trailing partial block's bitset is shorter than 65536, so the
            // length check is load-bearing: a doc past `max_doc` inside the
            // last block would otherwise reach `FixedBitSet::get`, which
            // panics on an out-of-range index. `local` is a `u16`-range offset, not a doc id,
            // so this is the hand-written form rather than `get_doc`.
            Some(Block::Dense(bits)) => (local as usize) < bits.len() && bits.get(local as usize),
        }
    }

    /// Every document in the set, ascending.
    pub fn iter(&self) -> RoaringIter<'_> {
        RoaringIter {
            set: self,
            block: 0,
            inner: BlockIter::Done,
            started: false,
        }
    }
}

/// Ascending iterator over a [`RoaringDocIdSet`] -- the `DocIdSetIterator`
/// `RoaringDocIdSet.iterator()` hands back, expressed as a plain Rust
/// `Iterator` for the same reason the merge combinators above are (see this
/// module's doc comment).
pub struct RoaringIter<'a> {
    set: &'a RoaringDocIdSet,
    block: usize,
    inner: BlockIter<'a>,
    started: bool,
}

enum BlockIter<'a> {
    Done,
    /// Block-local `next..end`, used for both `All` and `Range`.
    Range {
        next: u32,
        end: u32,
    },
    Sparse {
        docs: &'a [u16],
        i: usize,
    },
    InverseSparse {
        excluded: &'a [u16],
        i: usize,
        next: u32,
    },
    Dense {
        bits: &'a FixedBitSet,
        next: usize,
    },
}

impl<'a> RoaringIter<'a> {
    fn enter(&mut self, block: usize) {
        self.inner = match &self.set.blocks[block] {
            Block::Empty => BlockIter::Done,
            Block::All => BlockIter::Range {
                next: 0,
                end: BLOCK_SIZE as u32,
            },
            Block::Range { min, max } => BlockIter::Range {
                next: *min as u32,
                end: *max as u32 + 1,
            },
            Block::Sparse(docs) => BlockIter::Sparse { docs, i: 0 },
            Block::InverseSparse(excluded) => BlockIter::InverseSparse {
                excluded,
                i: 0,
                next: 0,
            },
            Block::Dense(bits) => BlockIter::Dense { bits, next: 0 },
        };
    }

    fn next_local(&mut self) -> Option<u32> {
        match &mut self.inner {
            BlockIter::Done => None,
            BlockIter::Range { next, end } => {
                if *next < *end {
                    let d = *next;
                    *next += 1;
                    Some(d)
                } else {
                    None
                }
            }
            BlockIter::Sparse { docs, i } => {
                let d = *docs.get(*i)?;
                *i += 1;
                Some(d as u32)
            }
            BlockIter::InverseSparse { excluded, i, next } => loop {
                if *next >= BLOCK_SIZE as u32 {
                    return None;
                }
                let d = *next;
                *next += 1;
                if excluded.get(*i).is_some_and(|&e| e as u32 == d) {
                    *i += 1;
                    continue;
                }
                return Some(d);
            },
            // Word-at-a-time, not bit-at-a-time: a dense block is up to 65536
            // bits, and `get` per bit would be 65536 bounds-checked loads to
            // yield at most 61440 documents.
            BlockIter::Dense { bits, next } => next_set_bit(bits, next).map(|bit| bit as u32),
        }
    }
}

impl Iterator for RoaringIter<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        loop {
            if !self.started {
                if self.block >= self.set.blocks.len() {
                    return None;
                }
                self.enter(self.block);
                self.started = true;
            }
            if let Some(local) = self.next_local() {
                return Some(((self.block << 16) as u32 + local) as i32);
            }
            self.block += 1;
            self.started = false;
            if self.block >= self.set.blocks.len() {
                return None;
            }
        }
    }
}

/// Builds a [`RoaringDocIdSet`] from strictly ascending doc IDs -- a port of
/// `RoaringDocIdSet.Builder`, including its buffer-then-upgrade-to-bitset
/// strategy (documents accumulate in a `MAX_ARRAY_LENGTH`-slot `u16` buffer
/// and only spill into a block-local bitset once that buffer overflows, so a
/// sparse block never allocates a bitset at all).
pub struct RoaringBuilder {
    max_doc: usize,
    blocks: Vec<Block>,
    cardinality: usize,
    first_doc_id: u32,
    last_doc_id: i64,
    current_block: i64,
    current_block_cardinality: usize,
    buffer: Vec<u16>,
    dense_buffer: Option<FixedBitSet>,
}

impl RoaringBuilder {
    pub fn new(max_doc: usize) -> Self {
        let num_blocks = max_doc.div_ceil(BLOCK_SIZE);
        Self {
            max_doc,
            blocks: vec![Block::Empty; num_blocks],
            cardinality: 0,
            first_doc_id: 0,
            last_doc_id: -1,
            current_block: -1,
            current_block_cardinality: 0,
            buffer: Vec::with_capacity(0),
            dense_buffer: None,
        }
    }

    /// Adds one document. Panics on a non-ascending doc ID **or one at or past
    /// the `max_doc` this builder was constructed with**, matching Java's
    /// `IllegalArgumentException` for the same misuse -- a programming error in
    /// the producer, not a data condition a caller recovers from.
    ///
    /// The `max_doc` bound is not decoration: a trailing partial block's dense
    /// bitset is sized `min(BLOCK_SIZE, max_doc - offset)`, and
    /// [`RoaringBuilder::append_in_current_block`]'s `// FBS:` proof
    /// rests on every accepted doc id being below `max_doc`, so the type
    /// enforces it.
    pub fn add(&mut self, doc_id: i32) {
        assert!(
            i64::from(doc_id) > self.last_doc_id,
            "doc ids must be added in order, got {doc_id} after {}",
            self.last_doc_id
        );
        assert!(
            doc_id >= 0 && (doc_id as usize) < self.max_doc,
            "doc id {doc_id} is outside this builder's 0..{} range",
            self.max_doc
        );
        let block = i64::from(doc_id) >> 16;
        if block != self.current_block {
            self.flush();
            self.current_block = block;
            self.first_doc_id = doc_id as u32;
        }
        self.append_in_current_block(doc_id as u32, block);
    }

    fn append_in_current_block(&mut self, doc_id: u32, block: i64) {
        if self.current_block_cardinality < MAX_ARRAY_LENGTH {
            self.buffer.push(doc_id as u16);
        } else {
            let offset = (block as usize) << 16;
            let dense = self.dense_buffer.get_or_insert_with(|| {
                let num_bits = BLOCK_SIZE.min(self.max_doc.saturating_sub(offset));
                let mut bits = FixedBitSet::new(num_bits);
                for &d in &self.buffer {
                    // FBS: `d` is a `u16` in-block offset pushed by this same
                    // method for a doc id this builder has already accepted,
                    // i.e. `doc_id < self.max_doc` with `doc_id >> 16 ==
                    // block`; so `d == doc_id - offset < max_doc - offset`,
                    // and `num_bits` is `min(BLOCK_SIZE, max_doc - offset)`
                    // -- the same bound, because `d` also fits 16 bits.
                    bits.set(d as usize);
                }
                bits
            });
            // FBS: same bound as the seeding loop above -- `doc_id - offset`
            // is this doc's in-block offset, below both `BLOCK_SIZE` and
            // `max_doc - offset`, which are what `num_bits` is the min of.
            dense.set(doc_id as usize - offset);
        }
        self.last_doc_id = i64::from(doc_id as i32);
        self.current_block_cardinality += 1;
    }

    /// Encodes whatever has accumulated for the current block and stores it.
    ///
    /// A block is only ever entered by adding a document to it, so a flushed
    /// block always has at least one -- blocks nobody wrote to keep the
    /// [`Block::Empty`] the constructor put there.
    fn flush(&mut self) {
        if self.current_block >= 0 && self.current_block_cardinality > 0 {
            let idx = self.current_block as usize;
            let card = self.current_block_cardinality;
            let block = if card == BLOCK_SIZE {
                Block::All
            } else if card as i64 == self.last_doc_id - i64::from(self.first_doc_id) + 1 {
                Block::Range {
                    min: self.first_doc_id as u16,
                    max: self.last_doc_id as u16,
                }
            } else if card <= MAX_ARRAY_LENGTH {
                Block::Sparse(std::mem::take(&mut self.buffer).into_boxed_slice())
            } else {
                let dense = self
                    .dense_buffer
                    .take()
                    .expect("a block past MAX_ARRAY_LENGTH always has a dense buffer");
                if dense.len() == BLOCK_SIZE && BLOCK_SIZE - card < MAX_ARRAY_LENGTH {
                    // Very dense: store the complement instead.
                    let mut excluded = Vec::with_capacity(BLOCK_SIZE - card);
                    for d in 0..BLOCK_SIZE {
                        if !dense.get(d) {
                            excluded.push(d as u16);
                        }
                    }
                    Block::InverseSparse(excluded.into_boxed_slice())
                } else {
                    Block::Dense(dense)
                }
            };
            self.blocks[idx] = block;
            self.cardinality += card;
        }
        self.buffer.clear();
        self.dense_buffer = None;
        self.current_block_cardinality = 0;
    }

    pub fn build(mut self) -> RoaringDocIdSet {
        self.flush();
        RoaringDocIdSet {
            blocks: self.blocks,
            cardinality: self.cardinality,
        }
    }
}

/// The representation a materialized match set is stored in -- the choice
/// `LRUQueryCache.cacheImpl` makes: a plain [`FixedBitSet`] (Java's
/// `BitDocIdSet`) when the set is dense enough that random access pays, a
/// [`RoaringDocIdSet`] when it is sparse.
///
/// Java's threshold, at Java's value: `cost * 100 >= maxDoc` picks the bitset,
/// i.e. a density of 1% or more. Below that the bitset is mostly zero words
/// and Roaring's per-block encoding is dramatically smaller.
///
/// One difference in *when* the decision is made: Java's `cost` is the scorer's
/// estimate, consulted before anything is materialized, so `cacheImpl` never
/// allocates a full bitset for a sparse set. This port has no `cost()` estimate
/// on its iterators, so the choice is made from the exact cardinality of an
/// already-built bitset. The stored representation is the same; the transient
/// allocation is not.
#[derive(Debug, Clone)]
pub enum CachedDocIdSet {
    Bits {
        bits: FixedBitSet,
        cardinality: usize,
    },
    Roaring(RoaringDocIdSet),
}

impl CachedDocIdSet {
    /// Picks the cheaper representation for an already-materialized bitset,
    /// applying `LRUQueryCache.cacheImpl`'s 1%-density rule.
    pub fn from_bitset(bits: FixedBitSet) -> Self {
        let cardinality = bits.cardinality();
        let max_doc = bits.len();
        if cardinality * 100 >= max_doc {
            CachedDocIdSet::Bits { bits, cardinality }
        } else {
            let mut builder = RoaringBuilder::new(max_doc);
            let mut next = 0usize;
            while let Some(doc) = next_set_bit(&bits, &mut next) {
                builder.add(doc as i32);
            }
            CachedDocIdSet::Roaring(builder.build())
        }
    }

    pub fn cardinality(&self) -> usize {
        match self {
            CachedDocIdSet::Bits { cardinality, .. } => *cardinality,
            CachedDocIdSet::Roaring(r) => r.cardinality(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.cardinality() == 0
    }

    /// Approximate heap cost in bytes -- what `LRUQueryCache` charges an entry
    /// against `maxRamBytesUsed`.
    pub fn ram_bytes_used(&self) -> usize {
        match self {
            CachedDocIdSet::Bits { bits, .. } => {
                std::mem::size_of::<Self>() + bits.words().len() * 8
            }
            CachedDocIdSet::Roaring(r) => r.ram_bytes_used(),
        }
    }

    pub fn contains(&self, doc: i32) -> bool {
        match self {
            CachedDocIdSet::Bits { bits, .. } => {
                doc >= 0 && (doc as usize) < bits.len() && bits.get(doc as usize)
            }
            CachedDocIdSet::Roaring(r) => r.contains(doc),
        }
    }

    /// Every document in the set, ascending. Costs `O(cardinality)` for the
    /// Roaring shape and `O(maxDoc / 64)` for the bitset shape -- never the
    /// `O(maxDoc)` bit-by-bit scan a naive `for doc in 0..len` would be.
    pub fn iter(&self) -> CachedDocIdSetIter<'_> {
        match self {
            CachedDocIdSet::Bits { bits, .. } => CachedDocIdSetIter::Bits { bits, next: 0 },
            CachedDocIdSet::Roaring(r) => CachedDocIdSetIter::Roaring(r.iter()),
        }
    }
}

pub enum CachedDocIdSetIter<'a> {
    Bits { bits: &'a FixedBitSet, next: usize },
    Roaring(RoaringIter<'a>),
}

impl Iterator for CachedDocIdSetIter<'_> {
    type Item = i32;

    fn next(&mut self) -> Option<i32> {
        match self {
            CachedDocIdSetIter::Bits { bits, next } => {
                next_set_bit(bits, next).map(|bit| bit as i32)
            }
            CachedDocIdSetIter::Roaring(it) => it.next(),
        }
    }
}

#[cfg(test)]
mod tests;
