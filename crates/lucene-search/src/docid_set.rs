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
/// thing in `O(log n)` per step instead of this port's `O(n)`; fine for a first cut
/// per the module doc, revisit if clause counts get large.
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
/// threshold and Java's reason. See `docs/sweep/m2/c6-search-followups.md`.
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
            // length check is load-bearing: `FixedBitSet::get` indexes its word
            // array behind a `debug_assert!`, and a doc past `max_doc` inside
            // the last block would reach it.
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

    /// Adds one document. Panics on a non-ascending doc ID, matching Java's
    /// `IllegalArgumentException` for the same misuse -- this is a programming
    /// error in the producer, not a data condition a caller recovers from.
    pub fn add(&mut self, doc_id: i32) {
        assert!(
            i64::from(doc_id) > self.last_doc_id,
            "doc ids must be added in order, got {doc_id} after {}",
            self.last_doc_id
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
                    bits.set(d as usize);
                }
                bits
            });
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
mod tests {
    use super::*;

    // -----------------------------------------------------------------
    // RoaringDocIdSet / CachedDocIdSet (`org.apache.lucene.util.
    // RoaringDocIdSet`, `LRUQueryCache.cacheImpl`)
    // -----------------------------------------------------------------

    fn roaring_of(docs: &[i32], max_doc: usize) -> RoaringDocIdSet {
        let mut b = RoaringBuilder::new(max_doc);
        for &d in docs {
            b.add(d);
        }
        b.build()
    }

    fn block_of(set: &RoaringDocIdSet, i: usize) -> &Block {
        &set.blocks[i]
    }

    /// The whole point of the encoding: every block shape must round-trip the
    /// exact doc set it was built from, and `contains` must agree with
    /// iteration for every doc in range -- including the docs *between* the
    /// matches, which is where an off-by-one in a block boundary would show.
    #[test]
    fn every_block_encoding_round_trips_and_contains_agrees_with_iteration() {
        let cases: Vec<(&str, Vec<i32>, usize)> = vec![
            ("empty", vec![], 200_000),
            ("single doc", vec![7], 200_000),
            ("sparse", vec![0, 5, 99, 65_535], 200_000),
            (
                "contiguous run (range encoding)",
                (100..350).collect(),
                200_000,
            ),
            (
                "second block only",
                vec![65_536, 65_537, 100_000, 131_071],
                200_000,
            ),
            (
                "straddles a block boundary",
                vec![65_534, 65_535, 65_536, 65_537],
                200_000,
            ),
            (
                "just over the array threshold (dense encoding)",
                (0..(1 << 12) + 1).map(|i| i * 3).collect(),
                200_000,
            ),
            (
                "whole first block set (all encoding)",
                (0..65_536).collect(),
                200_000,
            ),
            (
                "super dense: one hole (inverse encoding)",
                (0..65_536).filter(|d| *d != 4242).collect(),
                200_000,
            ),
        ];
        for (name, docs, max_doc) in cases {
            let set = roaring_of(&docs, max_doc);
            assert_eq!(set.cardinality(), docs.len(), "{name}: cardinality");
            assert_eq!(set.iter().collect::<Vec<_>>(), docs, "{name}: iteration");
            for doc in 0..max_doc as i32 {
                assert_eq!(
                    set.contains(doc),
                    docs.binary_search(&doc).is_ok(),
                    "{name}: contains({doc})"
                );
            }
        }
    }

    /// The encoding actually chosen per block is Java's, not merely "something
    /// that round-trips": `RoaringDocIdSet.Builder.flush`'s five cases, pinned.
    #[test]
    fn builder_picks_javas_block_encoding_for_each_density() {
        assert_eq!(block_of(&roaring_of(&[], 70_000), 0), &Block::Empty);
        assert_eq!(
            block_of(&roaring_of(&(0..65_536).collect::<Vec<_>>(), 200_000), 0),
            &Block::All
        );
        assert_eq!(
            block_of(&roaring_of(&[10, 11, 12], 70_000), 0),
            &Block::Range { min: 10, max: 12 }
        );
        assert!(matches!(
            block_of(&roaring_of(&[1, 5, 900], 70_000), 0),
            Block::Sparse(_)
        ));
        // MAX_ARRAY_LENGTH docs still fit the short array; one more spills.
        let exactly_max: Vec<i32> = (0..MAX_ARRAY_LENGTH as i32).map(|i| i * 3).collect();
        assert!(matches!(
            block_of(&roaring_of(&exactly_max, 200_000), 0),
            Block::Sparse(_)
        ));
        let one_more: Vec<i32> = (0..MAX_ARRAY_LENGTH as i32 + 1).map(|i| i * 3).collect();
        assert!(matches!(
            block_of(&roaring_of(&one_more, 200_000), 0),
            Block::Dense(_)
        ));
        // A full block missing fewer than MAX_ARRAY_LENGTH docs inverts.
        let almost_all: Vec<i32> = (0..65_536).filter(|d| d % 100 != 0).collect();
        assert!(matches!(
            block_of(&roaring_of(&almost_all, 200_000), 0),
            Block::InverseSparse(_)
        ));
    }

    /// A short final block can never be `All`/`InverseSparse` (both are
    /// defined over a whole 65536-doc block), and must still round-trip.
    #[test]
    fn trailing_partial_block_round_trips_without_the_full_block_encodings() {
        let max_doc = 65_536 + 10;
        let docs: Vec<i32> = (65_536..65_546).collect();
        let set = roaring_of(&docs, max_doc);
        assert_eq!(set.iter().collect::<Vec<_>>(), docs);
        assert_eq!(
            block_of(&set, 1),
            &Block::Range { min: 0, max: 9 },
            "10 contiguous docs in the trailing block is a range, not All"
        );
    }

    /// Doc IDs must be added in ascending order, same contract as Java's
    /// builder (which throws `IllegalArgumentException`).
    #[test]
    #[should_panic(expected = "must be added in order")]
    fn builder_rejects_out_of_order_docs() {
        let mut b = RoaringBuilder::new(1000);
        b.add(5);
        b.add(5);
    }

    /// The reason this encoding exists: a sparse match set must not cost
    /// `maxDoc / 8` bytes. 100 matches in a 10M-doc segment is 1.25 MB as a
    /// bitset; Roaring must be orders of magnitude smaller.
    #[test]
    fn sparse_set_costs_far_less_than_a_full_bitset() {
        let max_doc = 10_000_000;
        let docs: Vec<i32> = (0..100).map(|i| i * 97_777).collect();
        let mut bits = FixedBitSet::new(max_doc);
        for &d in &docs {
            bits.set(d as usize);
        }
        let bitset_bytes = bits.words().len() * 8;
        let roaring = roaring_of(&docs, max_doc);
        assert!(
            roaring.ram_bytes_used() * 100 < bitset_bytes,
            "roaring {} bytes vs bitset {bitset_bytes} bytes -- expected >100x smaller",
            roaring.ram_bytes_used()
        );
        assert_eq!(roaring.iter().collect::<Vec<_>>(), docs);
    }

    /// `LRUQueryCache.cacheImpl`'s exact rule: >= 1% density keeps the bitset,
    /// below that switches to Roaring. Boundary included, since that is where
    /// a `>` vs `>=` slip would hide.
    #[test]
    fn cached_doc_id_set_picks_javas_representation_at_the_one_percent_boundary() {
        let max_doc = 1000;
        // Exactly 1% (10 of 1000): Java keeps the bitset (`cost * 100 >= maxDoc`).
        let mut at = FixedBitSet::new(max_doc);
        for d in 0..10 {
            at.set(d * 7);
        }
        assert!(matches!(
            CachedDocIdSet::from_bitset(at),
            CachedDocIdSet::Bits { .. }
        ));
        // Just under 1% (9 of 1000): Roaring.
        let mut under = FixedBitSet::new(max_doc);
        for d in 0..9 {
            under.set(d * 7);
        }
        assert!(matches!(
            CachedDocIdSet::from_bitset(under),
            CachedDocIdSet::Roaring(_)
        ));
    }

    /// Both representations must be interchangeable at the API a consumer
    /// actually uses: same docs, same cardinality, same `contains`.
    #[test]
    fn both_cached_representations_agree_on_contents() {
        let max_doc = 5000;
        for &density in &[1usize, 5, 50, 500, 4999] {
            let docs: Vec<i32> = (0..density).map(|i| (i * 997 % max_doc) as i32).collect();
            let mut sorted = docs.clone();
            sorted.sort_unstable();
            sorted.dedup();
            let mut bits = FixedBitSet::new(max_doc);
            for &d in &sorted {
                bits.set(d as usize);
            }
            let cached = CachedDocIdSet::from_bitset(bits.clone());
            assert_eq!(cached.cardinality(), sorted.len());
            assert_eq!(cached.iter().collect::<Vec<_>>(), sorted);
            for doc in 0..max_doc as i32 {
                assert_eq!(cached.contains(doc), bits.get(doc as usize));
            }
        }
    }

    /// A bitset-shaped cached set iterates word-at-a-time; an empty one must
    /// still terminate, and a negative doc is never contained.
    #[test]
    fn cached_bitset_edge_cases() {
        let cached = CachedDocIdSet::Bits {
            bits: FixedBitSet::new(0),
            cardinality: 0,
        };
        assert!(cached.is_empty());
        assert_eq!(cached.iter().count(), 0);
        assert!(!cached.contains(-1));
        assert!(!roaring_of(&[], 100).contains(-1));
        assert!(!roaring_of(&[1], 100).contains(1_000_000));
        // A document past `max_doc` but *inside* the last block: the block
        // exists, so the encoding's own bound is what has to catch it. A dense
        // trailing block is the case that would otherwise index past the end
        // of a short bitset.
        let short_dense: Vec<i32> = (0..(1 << 12) + 1).map(|i| i * 2).collect();
        let set = roaring_of(&short_dense, 9000);
        assert!(matches!(block_of(&set, 0), Block::Dense(_)));
        assert!(!set.contains(9000));
        assert!(!set.contains(60_000));
        let mut bits = FixedBitSet::new(100);
        bits.set(3);
        let cached = CachedDocIdSet::Bits {
            bits,
            cardinality: 1,
        };
        assert!(cached.contains(3));
        assert!(!cached.contains(100));
        assert!(!cached.contains(1_000_000));
        assert!(roaring_of(&[], 100).is_empty());
        assert!(CachedDocIdSet::from_bitset(FixedBitSet::new(64)).ram_bytes_used() > 0);
    }

    /// `ram_bytes_used` must charge every block encoding, not just the sparse
    /// one -- the RAM bound `LRUQueryCache` enforces is only as good as the
    /// accounting behind it. A dense block is 8 KB of words, an inverse-sparse
    /// block is two bytes per *excluded* doc, and the free encodings (all /
    /// range / empty) really are free.
    #[test]
    fn ram_accounting_covers_every_block_encoding() {
        let free = roaring_of(&(0..65_536).collect::<Vec<_>>(), 200_000).ram_bytes_used();
        let range = roaring_of(&(0..100).collect::<Vec<_>>(), 200_000).ram_bytes_used();
        let empty = roaring_of(&[], 200_000).ram_bytes_used();
        assert_eq!(free, empty, "an all-set block carries no payload");
        assert_eq!(range, empty, "a contiguous run carries no payload");

        let dense: Vec<i32> = (0..(1 << 13)).map(|i| i * 4).collect();
        let dense_bytes = roaring_of(&dense, 200_000).ram_bytes_used();
        assert!(
            dense_bytes >= empty + 8192,
            "a dense block must charge its 65536-bit word array, got {dense_bytes}"
        );

        let inverse: Vec<i32> = (0..65_536).filter(|d| d % 100 != 0).collect();
        let inverse_bytes = roaring_of(&inverse, 200_000).ram_bytes_used();
        assert!(
            inverse_bytes < dense_bytes,
            "inverting a super-dense block must be cheaper than storing it, \
             got {inverse_bytes} vs {dense_bytes}"
        );
        assert!(inverse_bytes > empty);
    }

    /// Block equality is used by the encoding-choice tests, so it has to
    /// distinguish encodings rather than only compare within one. The dense arm
    /// in particular compares bitset contents by hand, since `FixedBitSet` has
    /// no `PartialEq`.
    #[test]
    fn block_equality_distinguishes_encodings_and_contents() {
        let all = roaring_of(&(0..65_536).collect::<Vec<_>>(), 200_000);
        let range = roaring_of(&(0..100).collect::<Vec<_>>(), 200_000);
        let sparse = roaring_of(&[1, 900], 200_000);
        let other_sparse = roaring_of(&[1, 901], 200_000);
        let dense_a = roaring_of(&(0..(1 << 13)).map(|i| i * 4).collect::<Vec<_>>(), 200_000);
        let dense_b = roaring_of(
            &(0..(1 << 13)).map(|i| i * 4 + 1).collect::<Vec<_>>(),
            200_000,
        );

        assert_ne!(block_of(&all, 0), block_of(&range, 0));
        assert_ne!(block_of(&sparse, 0), block_of(&other_sparse, 0));
        assert_ne!(block_of(&dense_a, 0), block_of(&dense_b, 0));
        assert_eq!(block_of(&dense_a, 0), block_of(&dense_a, 0));
        assert_ne!(block_of(&dense_a, 0), block_of(&sparse, 0));
        assert_eq!(
            block_of(&range, 0),
            &Block::Range { min: 0, max: 99 },
            "same encoding, same contents"
        );
        assert_ne!(block_of(&range, 0), &Block::Range { min: 0, max: 98 });
    }

    /// A `CachedDocIdSet` in the Roaring shape must answer the same shallow
    /// questions the bitset shape does.
    #[test]
    fn cached_roaring_reports_emptiness_and_cost() {
        let cached = CachedDocIdSet::Roaring(roaring_of(&[], 1_000_000));
        assert!(cached.is_empty());
        assert_eq!(cached.cardinality(), 0);
        assert!(cached.ram_bytes_used() > 0);
        assert!(!cached.contains(0));
    }

    /// A zero-document segment has no blocks at all: building and iterating one
    /// must terminate rather than index an empty block array.
    #[test]
    fn an_empty_doc_space_builds_an_empty_set() {
        let set = roaring_of(&[], 0);
        assert!(set.is_empty());
        assert_eq!(set.iter().count(), 0);
        assert!(!set.contains(0));
        assert_eq!(
            CachedDocIdSet::from_bitset(FixedBitSet::new(0)).cardinality(),
            0
        );
    }

    fn boxed(v: Vec<i32>) -> BoxDocIter<'static> {
        Box::new(v.into_iter())
    }

    fn collect_conjunction(inputs: Vec<Vec<i32>>) -> Vec<i32> {
        Conjunction::new(inputs.into_iter().map(boxed).collect()).collect()
    }

    fn collect_disjunction(inputs: Vec<Vec<i32>>) -> Vec<i32> {
        Disjunction::new(inputs.into_iter().map(boxed).collect()).collect()
    }

    fn collect_windowed(inputs: Vec<Vec<i32>>, min_should_match: usize) -> Vec<i32> {
        WindowedDisjunction::new(inputs, min_should_match).collect()
    }

    /// The reference answer, computed the obvious way rather than by calling
    /// either implementation: the union, restricted to documents at least
    /// `min_should_match` clauses agree on.
    fn brute_force_disjunction(inputs: &[Vec<i32>], min_should_match: usize) -> Vec<i32> {
        let mut counts: std::collections::BTreeMap<i32, usize> = Default::default();
        for docs in inputs {
            for &doc in docs {
                *counts.entry(doc).or_insert(0) += 1;
            }
        }
        counts
            .into_iter()
            .filter(|&(_, n)| n >= min_should_match.max(1))
            .map(|(doc, _)| doc)
            .collect()
    }

    /// The windowed OR must be indistinguishable from the min-scan disjunction
    /// it replaces -- including across window boundaries, which is where a
    /// window-at-a-time implementation goes wrong.
    ///
    /// The shapes cover: a document exactly on a window boundary (4095/4096),
    /// a clause that spans many windows against one confined to a single
    /// window, documents shared by several clauses (dedup), a clause that is
    /// empty, one that starts far past the others (so whole windows are
    /// skipped), and doc id 0.
    #[test]
    fn the_windowed_or_matches_the_min_scan_disjunction_on_every_shape() {
        let shapes: Vec<Vec<Vec<i32>>> = vec![
            vec![vec![], vec![]],
            vec![vec![0], vec![0]],
            vec![vec![0, 1, 2], vec![3, 4, 5]],
            vec![vec![4095, 4096], vec![4094, 4097]],
            // One clause spread over four windows, one confined to the third.
            vec![
                (0..16_384_i32).step_by(1_000).collect(),
                (8_192_i32..9_000).collect(),
            ],
            // A clause starting far past the others: whole windows are empty.
            vec![vec![1, 2, 3], vec![1_000_000, 1_000_001]],
            // Three clauses with heavy overlap.
            vec![
                (0..5_000_i32).collect(),
                (0..5_000_i32).step_by(2).collect(),
                (0..5_000_i32).step_by(3).collect(),
            ],
            vec![vec![], vec![7, 8, 9]],
            // The top of the doc-id space. `i32::MAX` is the last window's last
            // slot, and computing that window's exclusive end in `i32` overflows
            // -- which used to saturate to `i32::MAX`, make `doc >= max` true for
            // the very document that selected the window, and **hang**: no clause
            // advanced, nothing was emitted, and the next iteration re-derived
            // the same window forever. `Disjunction` terminates on this input, so
            // the replacement must too.
            vec![vec![i32::MAX], vec![i32::MAX - 1]],
            vec![vec![i32::MAX - 1, i32::MAX], vec![i32::MAX]],
            // ... and the same window reached from below, so the boundary is
            // crossed rather than only landed on.
            vec![vec![i32::MAX - 4096, i32::MAX], vec![i32::MAX - 1]],
        ];
        // Under a deadline: the failure this guards against is an infinite
        // loop, and a plain assertion cannot observe one -- it would wedge the
        // whole test binary instead of failing it.
        run_before(std::time::Duration::from_secs(30), move || {
            for shape in shapes {
                for msm in 0..=3usize {
                    let expected = brute_force_disjunction(&shape, msm);
                    assert_eq!(
                        collect_windowed(shape.clone(), msm),
                        expected,
                        "windowed OR disagrees for min_should_match={msm} on {shape:?}"
                    );
                    if msm <= 1 {
                        // ... and with the implementation it replaces, so a
                        // future change to either is caught by this file alone.
                        assert_eq!(collect_disjunction(shape.clone()), expected);
                    }
                }
            }
        });
    }

    /// Runs `body` on a worker thread and fails if it has not finished within
    /// `limit`. Used where the defect under test is non-termination, which an
    /// assertion cannot catch from inside the same thread.
    fn run_before(limit: std::time::Duration, body: impl FnOnce() + Send + 'static) {
        let (tx, rx) = std::sync::mpsc::channel();
        let worker = std::thread::spawn(move || {
            body();
            // A send failure means the receiver already gave up; the panic
            // below has been reported and this thread is just late.
            let _ = tx.send(());
        });
        match rx.recv_timeout(limit) {
            Ok(()) => worker.join().expect("worker panicked"),
            // Disconnected: the worker panicked, so surface that, not a timeout.
            Err(std::sync::mpsc::RecvTimeoutError::Disconnected) => {
                worker.join().expect("worker panicked")
            }
            Err(std::sync::mpsc::RecvTimeoutError::Timeout) => panic!(
                "did not finish within {limit:?} -- almost certainly a \
                 non-terminating window loop"
            ),
        }
    }

    /// `min_should_match` greater than the clause count matches nothing, and
    /// the per-window count must reset between windows -- a document in window
    /// 2 must not inherit window 1's tally.
    #[test]
    fn window_counts_reset_between_windows() {
        // doc 10 is in both clauses (count 2); doc 4106 (window 1) is in one.
        let shape = vec![vec![10, 4106], vec![10]];
        assert_eq!(collect_windowed(shape.clone(), 2), vec![10]);
        assert_eq!(collect_windowed(shape.clone(), 3), Vec::<i32>::new());
        // And the other way round, so a stale count cannot hide in either
        // direction.
        let shape = vec![vec![10, 4106], vec![4106]];
        assert_eq!(collect_windowed(shape, 2), vec![4106]);
    }

    /// A single clause is not this scorer's shape (`BooleanScorer`'s own
    /// constructor rejects fewer than two), so the caller must keep the plain
    /// path -- pinned here because the choice lives in `lib.rs`.
    #[test]
    fn a_single_clause_is_not_the_windowed_shape() {
        assert!(!WindowedDisjunction::is_applicable(0));
        assert!(!WindowedDisjunction::is_applicable(1));
        assert!(WindowedDisjunction::is_applicable(2));
    }

    #[test]
    fn conjunction_no_overlap_matches_nothing() {
        assert_eq!(
            collect_conjunction(vec![vec![1, 2], vec![3, 4]]),
            Vec::<i32>::new()
        );
    }

    #[test]
    fn conjunction_full_overlap_matches_every_doc() {
        assert_eq!(
            collect_conjunction(vec![vec![1, 2, 3], vec![1, 2, 3]]),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn conjunction_partial_overlap_matches_only_shared_docs() {
        assert_eq!(
            collect_conjunction(vec![vec![0, 2, 5, 7], vec![0, 1, 5, 9]]),
            vec![0, 5]
        );
    }

    #[test]
    fn conjunction_three_way_partial_overlap() {
        assert_eq!(
            collect_conjunction(vec![vec![1, 2, 3, 4], vec![2, 3, 4, 5], vec![3, 4, 5, 6]]),
            vec![3, 4]
        );
    }

    #[test]
    fn conjunction_one_iterator_empty_matches_nothing() {
        assert_eq!(
            collect_conjunction(vec![vec![1, 2, 3], vec![]]),
            Vec::<i32>::new()
        );
    }

    #[test]
    fn conjunction_single_clause_passes_through() {
        assert_eq!(collect_conjunction(vec![vec![4, 5, 6]]), vec![4, 5, 6]);
    }

    #[test]
    fn conjunction_no_clauses_matches_nothing() {
        assert_eq!(collect_conjunction(vec![]), Vec::<i32>::new());
    }

    #[test]
    fn disjunction_no_overlap_merges_both_sorted() {
        assert_eq!(
            collect_disjunction(vec![vec![1, 3], vec![2, 4]]),
            vec![1, 2, 3, 4]
        );
    }

    #[test]
    fn disjunction_full_overlap_dedups() {
        assert_eq!(
            collect_disjunction(vec![vec![1, 2, 3], vec![1, 2, 3]]),
            vec![1, 2, 3]
        );
    }

    #[test]
    fn disjunction_partial_overlap() {
        assert_eq!(
            collect_disjunction(vec![vec![0, 2, 4], vec![2, 3]]),
            vec![0, 2, 3, 4]
        );
    }

    #[test]
    fn disjunction_one_iterator_empty() {
        assert_eq!(collect_disjunction(vec![vec![1, 2], vec![]]), vec![1, 2]);
    }

    #[test]
    fn disjunction_no_clauses_matches_nothing() {
        assert_eq!(collect_disjunction(vec![]), Vec::<i32>::new());
    }

    #[test]
    fn excluding_removes_shared_docs() {
        let base = boxed(vec![0, 1, 2, 3, 4]);
        let excluded = boxed(vec![1, 3]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, vec![0, 2, 4]);
    }

    #[test]
    fn excluding_with_no_exclusions_passes_base_through() {
        let base = boxed(vec![0, 2, 4]);
        let excluded = boxed(vec![]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, vec![0, 2, 4]);
    }

    #[test]
    fn excluding_everything_matches_nothing() {
        let base = boxed(vec![0, 1, 2]);
        let excluded = boxed(vec![0, 1, 2]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, Vec::<i32>::new());
    }

    #[test]
    fn excluding_docs_not_in_base_have_no_effect() {
        let base = boxed(vec![0, 2, 4]);
        let excluded = boxed(vec![1, 3, 5, 6]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, vec![0, 2, 4]);
    }

    #[test]
    fn excluding_empty_base_matches_nothing() {
        let base = boxed(vec![]);
        let excluded = boxed(vec![1, 2]);
        let result: Vec<i32> = Excluding::new(base, excluded).collect();
        assert_eq!(result, Vec::<i32>::new());
    }
}
