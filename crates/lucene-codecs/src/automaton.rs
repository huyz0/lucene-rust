//! Byte-level finite automata for term-dictionary intersection: the part of
//! Lucene's `o.a.l.util.automaton` (`Automaton`, `Operations.determinize`,
//! `UTF32ToUTF8`, `ByteRunAutomaton`) that `CompiledAutomaton` hands to
//! `IntersectTermsEnum`.
//!
//! # What this is for
//!
//! A wildcard, regexp or fuzzy query has to find the terms it matches among
//! hundreds of thousands. Testing each term with a pattern matcher from
//! scratch is what this port did, and it is why those queries measured
//! 0.10x-0.40x of Lucene. A DFA changes two things:
//!
//! - **Terms share prefixes.** The walk is in sorted order, so the state after
//!   a term's common prefix with the previous term is already known; only the
//!   differing suffix is stepped. See [`DfaWalker`].
//! - **A dead state proves a whole range empty.** Once a prefix reaches a
//!   state from which no accepting state is reachable, every term sharing
//!   that prefix is a non-match and the walk seeks past all of them.
//!
//! # Superset automata
//!
//! Every automaton built here accepts **a superset** of its pattern's
//! language, and the caller confirms each accepted term with the pattern's
//! own exact matcher. That is a deliberate split: the DFA is the fast filter
//! and the source of skips, the existing matchers (which are pinned against
//! real Lucene by their own fixture tests) stay the authority on what
//! matches. Soundness needs only the superset property -- a prefix that is
//! dead for a superset automaton is dead for the pattern too. It is what lets
//! a construct with no cheap DFA (a regexp intersection `&`, a numeric
//! interval `<n-m>`, a malformed UTF-8 run) be approximated rather than
//! refused: the approximation only ever costs pruning, never an answer.
//!
//! Construction is bounded ([`MAX_NFA_STATES`], [`MAX_DFA_STATES`]); a pattern
//! that would exceed either gets `None`, and the caller keeps its
//! automaton-free scan -- the same fallback Lucene reaches with
//! `TooComplexToDeterminizeException`, minus the exception.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

/// The subset construction's state-set hasher: an FxHash-style multiply per
/// word. The keys are short sorted lists of small integers built by this
/// module, so the default SipHash buys nothing -- it was most of what
/// determinizing `t1[0-9]` cost, once per query.
#[derive(Default)]
struct SetHasher(u64);

impl Hasher for SetHasher {
    // `[u32]` hashes as one `write` of its bytes: eight at a time.
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.write_u64(u64::from_le_bytes(c.try_into().expect("eight bytes")));
        }
        for &b in chunks.remainder() {
            self.write_u64(u64::from(b));
        }
    }

    fn write_u32(&mut self, n: u32) {
        self.write_u64(u64::from(n));
    }

    fn write_u64(&mut self, n: u64) {
        self.0 = (self.0.rotate_left(5) ^ n).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
    }

    fn write_usize(&mut self, n: usize) {
        self.write_u64(n as u64);
    }

    fn finish(&self) -> u64 {
        self.0
    }
}

type SetMap = HashMap<Vec<u32>, u32, BuildHasherDefault<SetHasher>>;

/// The dead state: no accepting state is reachable from it.
pub const DEAD: u32 = u32::MAX;

/// Cap on NFA states a single pattern may build (Thompson construction plus
/// bounded-repetition expansion).
pub const MAX_NFA_STATES: usize = 20_000;

/// Cap on determinized states -- `Operations.DEFAULT_DETERMINIZE_WORK_LIMIT`'s
/// role. Each DFA state is a 256-entry row of `u32`, so this bounds a DFA at
/// 10 MiB.
pub const MAX_DFA_STATES: usize = 10_000;

/// The largest Unicode scalar value.
const MAX_CODE_POINT: u32 = 0x10_FFFF;

/// A deterministic automaton over bytes with a dense transition table.
#[derive(Debug, Clone)]
pub struct ByteDfa {
    /// `table[state * stride + classes[byte]]`, [`DEAD`] where no live state
    /// follows.
    table: Vec<u32>,
    /// Each byte's equivalence class: bytes no state tells apart share one
    /// column (the `regex-automata` crate's byte classes). A pattern over
    /// a few characters has a few classes, so its table is that much
    /// smaller than 256 columns -- and a walk's working set that much more
    /// likely to stay in cache.
    classes: Box<[u8; 256]>,
    /// Columns per state: the number of classes, `1..=256`.
    stride: usize,
    accept: Vec<bool>,
    start: u32,
    /// Each state's live bytes as maximal ascending `(lo, hi)` runs:
    /// state `s`'s are `live_ranges[live_offsets[s]..live_offsets[s + 1]]`.
    /// What [`DfaWalker::next_live_after`] searches instead of stepping
    /// every byte above the dead one -- a dictionary walk asks that once per
    /// dead branch, at every depth.
    live_ranges: Vec<(u8, u8)>,
    live_offsets: Vec<u32>,
}

impl ByteDfa {
    /// The start state, or [`DEAD`] when the automaton accepts nothing.
    #[inline]
    pub fn start(&self) -> u32 {
        self.start
    }

    /// The state after `byte` from `state`. `state` must not be [`DEAD`].
    // ARITH: `state < num_states` and a class `< stride`, so the index is
    // below `num_states * stride`, the table's length.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub fn step(&self, state: u32, byte: u8) -> u32 {
        // `state < accept.len()` for every state this automaton hands out,
        // and the table has exactly `stride` entries per state, one per class.
        self.table[(state as usize) * self.stride + usize::from(self.classes[usize::from(byte)])]
    }

    #[inline]
    pub fn is_accept(&self, state: u32) -> bool {
        state != DEAD && self.accept[state as usize]
    }

    /// Whether the whole of `bytes` is accepted.
    pub fn run(&self, bytes: &[u8]) -> bool {
        let mut s = self.start;
        for &b in bytes {
            if s == DEAD {
                return false;
            }
            s = self.step(s, b);
        }
        self.is_accept(s)
    }

    /// Number of states, for tests and diagnostics.
    pub fn num_states(&self) -> usize {
        self.accept.len()
    }
}

/// Walks a [`ByteDfa`] over terms in ascending order, reusing the states of
/// the prefix each term shares with the one before it.
#[derive(Debug, Clone)]
pub struct DfaWalker {
    /// `states[i]` is the state after the first `i` bytes of `prev`;
    /// `states.len() == prev.len() + 1` unless the walk died, in which case
    /// it stops at the dead byte.
    states: Vec<u32>,
    prev: Vec<u8>,
}

/// What [`DfaWalker::feed`] learned about one term.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Verdict {
    /// The automaton accepts the whole term.
    Accept,
    /// Not accepted, but some extension of the term might be.
    Reject,
    /// No term starting with the first `k` bytes can be accepted.
    DeadAt(usize),
}

impl DfaWalker {
    pub fn new(dfa: &ByteDfa) -> Self {
        DfaWalker {
            states: vec![dfa.start()],
            prev: Vec::new(),
        }
    }

    /// Runs `term` through `dfa`, starting from the state of the longest
    /// prefix it shares with the previously fed term.
    pub fn feed(&mut self, dfa: &ByteDfa, term: &[u8]) -> Verdict {
        let common = self
            .prev
            .iter()
            .zip(term)
            .take_while(|(a, b)| a == b)
            .count();
        // States are valid only as far as the previous walk got.
        let reuse = common.min(self.states.len().saturating_sub(1));
        self.states.truncate(reuse.saturating_add(1));
        self.prev.truncate(reuse);
        let mut s = self.states[reuse];
        if s == DEAD {
            return Verdict::DeadAt(reuse);
        }
        for (i, &b) in term.iter().enumerate().skip(reuse) {
            s = dfa.step(s, b);
            self.prev.push(b);
            self.states.push(s);
            if s == DEAD {
                return Verdict::DeadAt(i.saturating_add(1));
            }
        }
        if dfa.is_accept(s) {
            Verdict::Accept
        } else {
            Verdict::Reject
        }
    }

    /// `AutomatonTermsEnum.nextString`: after the term last fed died at byte
    /// `dead_at`, the smallest byte string that sorts after every term
    /// sharing its dead prefix and that the automaton can still extend to an
    /// accepted term -- `None` when there is no such string, i.e. no later
    /// term can match at all.
    ///
    /// Walks back from the dead byte: at each depth `d` it looks for the
    /// smallest byte greater than the term's own byte there that leads to a
    /// live state, and the answer is the term's first `d - 1` bytes plus that
    /// byte. `prefix_upper_bound(term[..dead_at])` would skip only the one
    /// dead prefix; this skips every dead sibling after it as well, which is
    /// the difference between one seek per matching term and one per
    /// dictionary branch.
    pub fn next_live_after(&self, dfa: &ByteDfa, term: &[u8], dead_at: usize) -> Option<Vec<u8>> {
        let deepest = dead_at
            .min(term.len())
            .min(self.states.len().saturating_sub(1));
        // `p` is the depth `d - 1` of the prose above: the state before, and
        // the byte at, position `p`.
        for p in (0..deepest).rev() {
            let from = self.states[p];
            if from == DEAD {
                continue;
            }
            let b = term[p];
            if let Some(next) = dfa.next_live_byte(from, b) {
                let mut target = term[..p].to_vec();
                target.push(next);
                return Some(target);
            }
        }
        None
    }
}

impl ByteDfa {
    /// The largest byte with a live transition out of `state`, or `None` --
    /// past it, every entry of a sorted term block is dead.
    #[inline]
    pub fn last_live_byte(&self, state: u32) -> Option<u8> {
        let s = state as usize;
        let hi = *self.live_offsets.get(s.checked_add(1)?)? as usize;
        let lo = self.live_offsets[s] as usize;
        self.live_ranges[lo..hi].last().map(|&(_, run_hi)| run_hi)
    }
}

impl ByteDfa {
    /// The smallest byte at or after `from` (`0..=256`) with a live transition
    /// out of `state` -- which floor blocks of a term-dictionary node can hold
    /// anything the automaton still wants.
    // ARITH: the `1..=256` arm has `from >= 1`, and `from - 1 <= 255`.
    #[allow(clippy::arithmetic_side_effects)]
    #[inline]
    pub fn first_live_from(&self, state: u32, from: u32) -> Option<u8> {
        match from {
            0 => {
                if self.step(state, 0) != DEAD {
                    Some(0)
                } else {
                    self.next_live_byte(state, 0)
                }
            }
            1..=256 => self.next_live_byte(state, (from - 1) as u8),
            _ => None,
        }
    }
}

impl ByteDfa {
    /// The smallest byte above `after` with a live transition out of
    /// `state`, from the state's precomputed live runs.
    #[inline]
    fn next_live_byte(&self, state: u32, after: u8) -> Option<u8> {
        let s = state as usize;
        let (lo, hi) = (
            self.live_offsets[s] as usize,
            *self.live_offsets.get(s.checked_add(1)?)? as usize,
        );
        self.live_ranges[lo..hi]
            .iter()
            .find(|&&(_, run_hi)| run_hi > after)
            .map(|&(run_lo, _)| run_lo.max(after.saturating_add(1)))
    }
}

impl ByteDfa {
    /// `state`'s transitions as maximal `(lo, hi, target)` byte runs to one
    /// target, ascending, dead ones left out.
    // ARITH: `b` counts `0..=256` and indexes a 256-entry row; `b - 1` runs
    // after `b` moved past `lo`, so it is at least `lo`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn transitions(&self, state: u32) -> Vec<(u8, u8, u32)> {
        let row: Vec<u32> = (0..=255u8).map(|b| self.step(state, b)).collect();
        let mut out = Vec::new();
        let mut b = 0usize;
        while b < 256 {
            let t = row[b];
            let lo = b;
            while b < 256 && row[b] == t {
                b += 1;
            }
            if t != DEAD {
                out.push((lo as u8, (b - 1) as u8, t));
            }
        }
        out
    }

    /// The product automaton: the terms both `self` and `other` accept
    /// (`Operations.intersection`), pruned. `None` past [`MAX_DFA_STATES`].
    // ARITH: product state ids are bounded by `MAX_DFA_STATES` (10 000),
    // checked before each is created, so they fit a `u32`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn intersect(&self, other: &ByteDfa) -> Option<ByteDfa> {
        let mut ids: HashMap<(u32, u32), u32, BuildHasherDefault<SetHasher>> = HashMap::default();
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut trans: Vec<(u32, u8, u8, u32)> = Vec::new();
        let mut accepting = Vec::new();
        let start = (self.start, other.start);
        if start.0 != DEAD && start.1 != DEAD {
            ids.insert(start, 0);
            pairs.push(start);
        } else {
            // The empty language: one dead start state.
            return Some(prune(&[], vec![false]));
        }
        let mut next = 0usize;
        while next < pairs.len() {
            let (a, b) = pairs[next];
            accepting.push(self.is_accept(a) && other.is_accept(b));
            let (ra, rb) = (self.transitions(a), other.transitions(b));
            let (mut i, mut j) = (0usize, 0usize);
            while i < ra.len() && j < rb.len() {
                let (alo, ahi, at) = ra[i];
                let (blo, bhi, bt) = rb[j];
                let (lo, hi) = (alo.max(blo), ahi.min(bhi));
                if lo <= hi {
                    let key = (at, bt);
                    let id = match ids.get(&key) {
                        Some(&id) => id,
                        None => {
                            if pairs.len() >= MAX_DFA_STATES {
                                return None;
                            }
                            let id = pairs.len() as u32;
                            ids.insert(key, id);
                            pairs.push(key);
                            id
                        }
                    };
                    trans.push((next as u32, lo, hi, id));
                }
                if ahi < bhi {
                    i += 1;
                } else {
                    j += 1;
                }
            }
            next += 1;
        }
        Some(prune(&trans, accepting))
    }
}

impl ByteDfa {
    /// The one-state automaton that accepts every byte string -- what
    /// `CompiledAutomaton`'s `AUTOMATON_TYPE.ALL` enumerates: the whole term
    /// dictionary, ill-formed UTF-8 included.
    pub fn total_bytes() -> ByteDfa {
        ByteDfa {
            table: vec![0],
            classes: Box::new([0; 256]),
            stride: 1,
            accept: vec![true],
            start: 0,
            live_ranges: vec![(0, 255)],
            live_offsets: vec![0, 1],
        }
    }

    /// Per state: whether the language from it is exactly "every well-formed
    /// UTF-8 string" -- `Operations.isTotal` on the code-point automaton this
    /// byte automaton was converted from. From such a state a term's
    /// remaining bytes match exactly when they are well-formed UTF-8, which a
    /// validator answers without stepping the table byte by byte.
    ///
    /// Decided in one exploration of the product with the canonical all-UTF-8
    /// automaton, started from every `(state, canonical start)` pair at once:
    /// a pair is bad when it disagrees on acceptance or on which bytes lead
    /// anywhere, badness propagates back along the product's edges, and a
    /// state is total when its starting pair stayed good -- linear in the
    /// product, as `Operations.isTotal` is in the automaton.
    // ARITH: pair ids count distinct `(state, canonical state)` pairs, at most
    // `num_states * canon.num_states()`, well inside `u32`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn utf8_total_states(&self) -> Vec<bool> {
        let canon = utf8_total_dfa();
        let n = self.num_states();
        let mut ids: HashMap<(u32, u32), u32, BuildHasherDefault<SetHasher>> = HashMap::default();
        let mut pairs: Vec<(u32, u32)> = Vec::new();
        let mut bad: Vec<bool> = Vec::new();
        let mut rev: Vec<Vec<u32>> = Vec::new();
        let mut intern = |p: (u32, u32),
                          pairs: &mut Vec<(u32, u32)>,
                          bad: &mut Vec<bool>,
                          rev: &mut Vec<Vec<u32>>| {
            *ids.entry(p).or_insert_with(|| {
                pairs.push(p);
                bad.push(false);
                rev.push(Vec::new());
                (pairs.len() - 1) as u32
            })
        };
        let roots: Vec<u32> = (0..n as u32)
            .map(|s| intern((s, canon.start), &mut pairs, &mut bad, &mut rev))
            .collect();
        // One exploration of the product for every starting state at once;
        // a pair is bad when it disagrees on acceptance or on which bytes
        // lead anywhere.
        let mut next = 0usize;
        while next < pairs.len() {
            let (a, b) = pairs[next];
            if self.is_accept(a) != canon.is_accept(b) {
                bad[next] = true;
            }
            for byte in 0..=255u8 {
                let (na, nb) = (self.step(a, byte), canon.step(b, byte));
                match (na == DEAD, nb == DEAD) {
                    (true, true) => {}
                    (false, false) => {
                        let t = intern((na, nb), &mut pairs, &mut bad, &mut rev);
                        rev[t as usize].push(next as u32);
                    }
                    _ => bad[next] = true,
                }
            }
            next += 1;
        }
        // Badness flows backwards: a pair that can reach a bad one is bad.
        let mut stack: Vec<u32> = (0..pairs.len() as u32)
            .filter(|&p| bad[p as usize])
            .collect();
        while let Some(p) = stack.pop() {
            // Each pair turns bad once, so its predecessor list is needed once.
            for q in std::mem::take(&mut rev[p as usize]) {
                if !bad[q as usize] {
                    bad[q as usize] = true;
                    stack.push(q);
                }
            }
        }
        roots.iter().map(|&r| !bad[r as usize]).collect()
    }
}

/// A [`ByteDfa`] ready for a term-dictionary walk -- `CompiledAutomaton`'s
/// role: the automaton to run (swapped for [`ByteDfa::total_bytes`] when it
/// accepts every code-point string, Lucene's `AUTOMATON_TYPE.ALL`) and which
/// of its states accept exactly the well-formed UTF-8 suffixes. Built once
/// per pattern and shared, read-only, by every segment's walk.
#[derive(Debug)]
pub struct CompiledDfa {
    pub dfa: ByteDfa,
    pub utf8_total: Vec<bool>,
}

impl CompiledDfa {
    pub fn new(dfa: ByteDfa) -> Self {
        let utf8_total = dfa.utf8_total_states();
        if dfa.start() != DEAD && utf8_total[dfa.start() as usize] {
            let dfa = ByteDfa::total_bytes();
            let utf8_total = vec![false];
            return CompiledDfa { dfa, utf8_total };
        }
        CompiledDfa { dfa, utf8_total }
    }
}

/// What a term-dictionary walk needs from an automaton, eager or lazy.
pub trait TermAutomaton {
    fn start(&self) -> u32;
    /// The state after `byte`, or `None` when a lazy automaton has reached
    /// its state budget and cannot answer -- the walk then fails the query,
    /// as Lucene's `TooComplexToDeterminizeException` does.
    fn step(&mut self, state: u32, byte: u8) -> Option<u32>;
    fn is_accept(&self, state: u32) -> bool;
    /// The largest byte leading anywhere from `state`.
    fn last_live_byte(&mut self, state: u32) -> Option<Option<u8>>;
    /// The smallest byte at or after `from` leading anywhere from `state`.
    fn first_live_from(&mut self, state: u32, from: u32) -> Option<Option<u8>>;
    /// Whether exactly the well-formed UTF-8 suffixes are accepted from
    /// `state` (see [`ByteDfa::utf8_total_states`]); `false` when unknown.
    fn utf8_total(&self, state: u32) -> bool;
}

impl TermAutomaton for std::sync::Arc<CompiledDfa> {
    #[inline]
    fn start(&self) -> u32 {
        self.dfa.start()
    }
    #[inline]
    fn step(&mut self, state: u32, byte: u8) -> Option<u32> {
        Some(self.dfa.step(state, byte))
    }
    #[inline]
    fn is_accept(&self, state: u32) -> bool {
        self.dfa.is_accept(state)
    }
    #[inline]
    fn last_live_byte(&mut self, state: u32) -> Option<Option<u8>> {
        Some(self.dfa.last_live_byte(state))
    }
    #[inline]
    fn first_live_from(&mut self, state: u32, from: u32) -> Option<Option<u8>> {
        Some(self.dfa.first_live_from(state, from))
    }
    #[inline]
    fn utf8_total(&self, state: u32) -> bool {
        self.utf8_total[state as usize]
    }
}

/// NFA-state entries, summed over every state set a [`LazyDfa`] builds, before
/// it gives up: its memory is those sets (each held once, shared by the
/// lookup map and the state list) plus a few byte ranges per state, so this
/// bounds one query to about 64 MB. A walk reaches it only for a pattern
/// whose automaton is exponential *and* whose dictionary exercises it.
pub const MAX_LAZY_SET_ENTRIES: usize = 16_000_000;

/// The subset construction run on demand: a state's transitions are worked
/// out the first time the walk steps out of it (`NFARunAutomaton`, and the
/// `regex-automata` crate's hybrid DFA). For a pattern whose full DFA is too
/// large to build ([`MAX_DFA_STATES`]) -- the ones Lucene rejects with
/// `TooComplexToDeterminizeException` by default -- only the states the
/// dictionary actually leads to are ever built, and the walk keeps its
/// block skipping instead of testing every term with a backtracker.
///
/// No pruning: a state is dead only when its set is empty, so a state that
/// can never reach acceptance still looks live and its sub-blocks are
/// visited. That costs work, never answers.
#[derive(Debug)]
pub struct LazyDfa {
    nfa: Nfa,
    accept_nfa: u32,
    x: Expander,
    sets: Vec<std::sync::Arc<[u32]>>,
    ids: HashMap<std::sync::Arc<[u32]>, u32, BuildHasherDefault<SetHasher>>,
    /// Set entries spent so far, against `limit`.
    entries: usize,
    /// [`MAX_LAZY_SET_ENTRIES`], or a test's smaller budget.
    limit: usize,
    /// `rows[s]`: `s`'s transitions as ascending `(lo, hi, target)` runs,
    /// once built.
    rows: Vec<Option<Vec<(u8, u8, u32)>>>,
    accept: Vec<bool>,
}

impl LazyDfa {
    /// [`Self::new`] with a smaller entry budget than
    /// [`MAX_LAZY_SET_ENTRIES`] -- for testing what happens when it runs out.
    pub fn with_limit(nfa: Nfa, start: u32, accept: u32, limit: usize) -> Self {
        let mut d = Self::new(nfa, start, accept);
        d.limit = limit;
        d
    }

    pub fn new(nfa: Nfa, start: u32, accept: u32) -> Self {
        let mut x = Expander::new(&nfa);
        let first: std::sync::Arc<[u32]> = x.start_set(&nfa, start).into();
        let mut ids = HashMap::default();
        ids.insert(first.clone(), 0);
        let accepting = first.binary_search(&accept).is_ok();
        LazyDfa {
            nfa,
            accept_nfa: accept,
            x,
            entries: first.len(),
            limit: MAX_LAZY_SET_ENTRIES,
            sets: vec![first],
            ids,
            rows: vec![None],
            accept: vec![accepting],
        }
    }

    // ARITH: every set has at least one entry, so `sets.len() <= entries <=
    // MAX_LAZY_SET_ENTRIES` (16 000 000), checked before each new state: ids
    // and the entry count fit a `u32` and a `usize`.
    #[allow(clippy::arithmetic_side_effects)]
    fn row(&mut self, state: u32) -> Option<&[(u8, u8, u32)]> {
        let s = state as usize;
        if self.rows[s].is_none() {
            let set = self.sets[s].clone();
            let mut row = Vec::new();
            let LazyDfa {
                nfa,
                accept_nfa,
                x,
                sets,
                ids,
                entries,
                limit,
                rows,
                accept,
            } = self;
            let ok = nfa.expand(&set, x, |a, b, target| {
                let id = match ids.get(target) {
                    Some(&id) => id,
                    None => {
                        if *entries + target.len() > *limit {
                            return None;
                        }
                        *entries += target.len();
                        let id = sets.len() as u32;
                        let shared: std::sync::Arc<[u32]> = target.into();
                        ids.insert(shared.clone(), id);
                        sets.push(shared);
                        rows.push(None);
                        accept.push(target.binary_search(accept_nfa).is_ok());
                        id
                    }
                };
                row.push((a, b, id));
                Some(())
            });
            ok?;
            self.rows[s] = Some(row);
        }
        self.rows[s].as_deref()
    }
}

impl TermAutomaton for LazyDfa {
    fn start(&self) -> u32 {
        0
    }

    fn step(&mut self, state: u32, byte: u8) -> Option<u32> {
        let row = self.row(state)?;
        Some(
            match row.binary_search_by(|&(lo, hi, _)| {
                if hi < byte {
                    std::cmp::Ordering::Less
                } else if lo > byte {
                    std::cmp::Ordering::Greater
                } else {
                    std::cmp::Ordering::Equal
                }
            }) {
                Ok(i) => row[i].2,
                Err(_) => DEAD,
            },
        )
    }

    fn is_accept(&self, state: u32) -> bool {
        state != DEAD && self.accept[state as usize]
    }

    fn last_live_byte(&mut self, state: u32) -> Option<Option<u8>> {
        Some(self.row(state)?.last().map(|&(_, hi, _)| hi))
    }

    fn first_live_from(&mut self, state: u32, from: u32) -> Option<Option<u8>> {
        Some(
            self.row(state)?
                .iter()
                .find(|&&(_, hi, _)| u32::from(hi) >= from)
                .map(|&(lo, _, _)| lo.max(from.min(255) as u8)),
        )
    }

    fn utf8_total(&self, _state: u32) -> bool {
        false
    }
}

/// `(any code point)*` as a byte automaton, built once.
fn utf8_total_dfa() -> &'static ByteDfa {
    static DFA: std::sync::OnceLock<ByteDfa> = std::sync::OnceLock::new();
    DFA.get_or_init(|| {
        let mut nfa = Nfa::new();
        let hub = nfa.state().expect("a handful of states");
        let back = nfa.state().expect("a handful of states");
        nfa.any_code_point(hub, back).expect("a handful of states");
        nfa.epsilon(back, hub);
        nfa.determinize(hub, hub).expect("a small automaton")
    })
}

impl Nfa {
    /// Copies `dfa` in as fresh states entered from `from`, returning the one
    /// state every accepting copy has an epsilon to -- how a sub-language
    /// that had to be determinized on its own (an intersection) rejoins the
    /// Thompson construction.
    pub fn embed(&mut self, from: u32, dfa: &ByteDfa) -> Option<u32> {
        let to = self.state()?;
        if dfa.start() == DEAD {
            return Some(to);
        }
        let mut map = Vec::with_capacity(dfa.num_states());
        for _ in 0..dfa.num_states() {
            map.push(self.state()?);
        }
        self.epsilon(from, map[dfa.start() as usize]);
        for s in 0..dfa.num_states() as u32 {
            for (lo, hi, t) in dfa.transitions(s) {
                self.range(map[s as usize], lo, hi, map[t as usize]);
            }
            if dfa.is_accept(s) {
                self.epsilon(map[s as usize], to);
            }
        }
        Some(to)
    }

    /// Paths `from -> to` over every string of exactly `lo.len()` ASCII
    /// digits between `lo` and `hi` inclusive (both that wide): the digit
    /// automaton that remembers only whether the prefix read so far still
    /// equals `lo`'s and `hi`'s -- at most four live states per position.
    // ARITH: `i + 1 <= lo.len()`; digits are `b'0'..=b'9'`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn fixed_width_digits(&mut self, from: u32, lo: &[u8], hi: &[u8], to: u32) -> Option<()> {
        debug_assert_eq!(lo.len(), hi.len());
        let w = lo.len();
        // `states[(tight_lo, tight_hi)]` at the current position.
        let mut cur: [[Option<u32>; 2]; 2] = [[None; 2]; 2];
        cur[1][1] = Some(from);
        for i in 0..w {
            let mut nxt: [[Option<u32>; 2]; 2] = [[None; 2]; 2];
            for (tl, row) in cur.iter().enumerate() {
                for (th, slot) in row.iter().enumerate() {
                    let Some(s) = *slot else { continue };
                    let min = if tl == 1 { lo[i] } else { b'0' };
                    let max = if th == 1 { hi[i] } else { b'9' };
                    if min > max {
                        continue;
                    }
                    // Split the digit range where either tightness changes.
                    let mut cuts = vec![min];
                    for c in [
                        lo[i],
                        lo[i].saturating_add(1),
                        hi[i],
                        hi[i].saturating_add(1),
                    ] {
                        if c > min && c <= max {
                            cuts.push(c);
                        }
                    }
                    cuts.push(max + 1);
                    cuts.sort_unstable();
                    cuts.dedup();
                    for win in cuts.windows(2) {
                        let (a, b) = (win[0], win[1] - 1);
                        let ntl = usize::from(tl == 1 && a == lo[i]);
                        let nth = usize::from(th == 1 && a == hi[i]);
                        let target = if i + 1 == w {
                            to
                        } else {
                            match nxt[ntl][nth] {
                                Some(t) => t,
                                None => {
                                    let t = self.state()?;
                                    nxt[ntl][nth] = Some(t);
                                    t
                                }
                            }
                        };
                        self.range(s, a, b, target);
                    }
                }
            }
            cur = nxt;
        }
        if w == 0 {
            self.epsilon(from, to);
        }
        Some(())
    }
}

// ---------------------------------------------------------------------------
// NFA construction.
// ---------------------------------------------------------------------------

/// A nondeterministic automaton over bytes: byte-range transitions and
/// epsilon moves, built by Thompson construction.
#[derive(Debug, Default)]
pub struct Nfa {
    ranges: Vec<Vec<(u8, u8, u32)>>,
    eps: Vec<Vec<u32>>,
}

impl Nfa {
    pub fn new() -> Self {
        Self::default()
    }

    /// A fresh state, or `None` past [`MAX_NFA_STATES`].
    //
    // ARITH: `len() - 1` follows a `push`, so `len() >= 1`; and `len() <=
    // MAX_NFA_STATES` (20 000) makes the `as u32` exact.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn state(&mut self) -> Option<u32> {
        if self.ranges.len() >= MAX_NFA_STATES {
            return None;
        }
        self.ranges.push(Vec::new());
        self.eps.push(Vec::new());
        Some((self.ranges.len() - 1) as u32)
    }

    pub fn range(&mut self, from: u32, lo: u8, hi: u8, to: u32) {
        self.ranges[from as usize].push((lo, hi, to));
    }

    pub fn epsilon(&mut self, from: u32, to: u32) {
        self.eps[from as usize].push(to);
    }

    /// A path `from -> ... -> end` spelling exactly `bytes`; returns `end`.
    pub fn bytes(&mut self, from: u32, bytes: &[u8]) -> Option<u32> {
        let mut s = from;
        for &b in bytes {
            let t = self.state()?;
            self.range(s, b, b, t);
            s = t;
        }
        Some(s)
    }

    /// Paths `from -> to` over the UTF-8 encoding of every code point in
    /// `lo..=hi` (surrogates excluded), as `UTF32ToUTF8` converts one
    /// code-point transition.
    pub fn code_points(&mut self, from: u32, lo: u32, hi: u32, to: u32) -> Option<()> {
        let mut seqs = Vec::new();
        utf8_sequences(lo, hi.min(MAX_CODE_POINT), &mut seqs);
        for seq in seqs {
            let mut s = from;
            let (last, head) = seq.split_last()?;
            for &(a, b) in head {
                let t = self.state()?;
                self.range(s, a, b, t);
                s = t;
            }
            self.range(s, last.0, last.1, to);
        }
        Some(())
    }

    /// Any one code point, `Automata.makeAnyChar`.
    pub fn any_code_point(&mut self, from: u32, to: u32) -> Option<()> {
        self.code_points(from, 0, MAX_CODE_POINT, to)
    }

    /// One "character" the way a byte-oriented matcher that does not validate
    /// UTF-8 sees it: a lead byte, then as many further bytes as the lead
    /// byte's high bits claim -- any bytes at all -- or fewer if the term ends
    /// first. A lead that is not a UTF-8 lead (a continuation byte, `0xF8..`)
    /// is one byte on its own.
    ///
    /// This is a superset both of wildcard `?` (which consumes the lead
    /// byte's nominal width, truncated at the end of the term) and of one
    /// `char` of a lossily-decoded term (whose invalid runs decode to one
    /// replacement character per maximal ill-formed subpart, never longer
    /// than the lead byte's nominal width). Because a shorter run is also
    /// allowed at every step, the automaton cannot be stricter than either.
    // ARITH: `i < extra <= 3`, so `i + 1 <= 3`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn lenient_char(&mut self, from: u32, to: u32) -> Option<()> {
        self.range(from, 0x00, 0x7F, to);
        self.range(from, 0x80, 0xBF, to);
        self.range(from, 0xF8, 0xFF, to);
        for (lo, hi, extra) in [(0xC0u8, 0xDFu8, 1usize), (0xE0, 0xEF, 2), (0xF0, 0xF7, 3)] {
            let mut s = self.state()?;
            self.range(from, lo, hi, s);
            self.epsilon(s, to);
            for i in 0..extra {
                let t = if i + 1 == extra { to } else { self.state()? };
                self.range(s, 0x00, 0xFF, t);
                if t != to {
                    self.epsilon(t, to);
                }
                s = t;
            }
        }
        Some(())
    }

    /// A path `from -> to` spelling exactly `bytes`.
    pub fn bytes_to(&mut self, from: u32, bytes: &[u8], to: u32) -> Option<()> {
        match bytes.split_last() {
            None => self.epsilon(from, to),
            Some((&last, head)) => {
                let s = self.bytes(from, head)?;
                self.range(s, last, last, to);
            }
        }
        Some(())
    }

    // ARITH: every state id is below `ranges.len() <= MAX_NFA_STATES`, and
    // `seen` has `ranges.len().div_ceil(64)` words, so `id >> 6` indexes it;
    // `1 << (id & 63)` shifts a `u64` by at most 63.
    #[allow(clippy::arithmetic_side_effects)]
    fn closure(&self, set: &mut Vec<u32>, stack: &mut Vec<u32>, seen: &mut [u64]) {
        stack.clear();
        for &s in set.iter() {
            seen[s as usize >> 6] |= 1 << (s & 63);
            stack.push(s);
        }
        while let Some(s) = stack.pop() {
            for &t in &self.eps[s as usize] {
                let (w, bit) = (t as usize >> 6, 1u64 << (t & 63));
                if seen[w] & bit == 0 {
                    seen[w] |= bit;
                    set.push(t);
                    stack.push(t);
                }
            }
        }
        for &s in set.iter() {
            seen[s as usize >> 6] &= !(1 << (s & 63));
        }
        set.sort_unstable();
        set.dedup();
    }

    /// Subset construction, then removal of every state that cannot reach
    /// `accept`. `None` past [`MAX_DFA_STATES`].
    // ARITH: byte bounds are `u8`s widened to `u16`, so `hi + 1 <= 256`;
    // `cuts` is sorted, deduplicated and holds 0 and 256, so every window has
    // `w[1] > w[0] >= 0` and `w[1] - 1` is in `0..=255`, so `a`/`b` fit a `u8`.
    // `next < sets.len() <= MAX_DFA_STATES` (10 000) fits a `u32`, and
    // `next += 1` stops at `sets.len()`.
    #[allow(clippy::arithmetic_side_effects)]
    pub fn determinize(&self, start: u32, accept: u32) -> Option<ByteDfa> {
        let mut x = Expander::new(self);
        let mut sets: Vec<Vec<u32>> = Vec::new();
        let mut ids = SetMap::default();
        // `(state, lo, hi, target)`, per state in ascending byte order: the
        // transitions as ranges, a handful per state, rather than a 256-entry
        // row that every later pass would have to scan.
        let mut trans: Vec<(u32, u8, u8, u32)> = Vec::new();

        let first = x.start_set(self, start);
        ids.insert(first.clone(), 0);
        sets.push(first);

        let mut next = 0usize;
        while next < sets.len() {
            let set = std::mem::take(&mut sets[next]);
            let ok = self.expand(&set, &mut x, |a, b, target| {
                let id = match ids.get(target) {
                    Some(&id) => id,
                    None => {
                        if sets.len() >= MAX_DFA_STATES {
                            return None;
                        }
                        let id = sets.len() as u32;
                        ids.insert(target.to_vec(), id);
                        sets.push(target.to_vec());
                        id
                    }
                };
                trans.push((next as u32, a, b, id));
                Some(())
            });
            sets[next] = set;
            ok?;
            next += 1;
        }

        let accepting: Vec<bool> = sets
            .iter()
            .map(|s| s.binary_search(&accept).is_ok())
            .collect();
        Some(prune(&trans, accepting))
    }

    /// One subset-construction step: calls `f(lo, hi, target_set)` for each
    /// maximal byte range out of the state set `set` that leads somewhere,
    /// ascending, with `target_set` epsilon-closed, sorted and deduplicated.
    /// Stops early, returning `None`, when `f` does.
    // ARITH: byte bounds are `u8`s widened to `u16`, so `hi + 1 <= 256`;
    // `cuts` is sorted, deduplicated and holds 0 and 256, so every window has
    // `w[1] > w[0] >= 0` and `w[1] - 1` is in `0..=255`, fitting a `u8`.
    #[allow(clippy::arithmetic_side_effects)]
    fn expand(
        &self,
        set: &[u32],
        x: &mut Expander,
        mut f: impl FnMut(u8, u8, &[u32]) -> Option<()>,
    ) -> Option<()> {
        x.edges.clear();
        for &s in set {
            x.edges.extend_from_slice(&self.ranges[s as usize]);
        }
        // The byte axis cut at every range boundary: within one piece,
        // every edge either covers all of it or none of it.
        x.cuts.clear();
        x.cuts.push(0);
        x.cuts.push(256);
        for &(lo, hi, _) in &x.edges {
            x.cuts.push(lo as u16);
            x.cuts.push(hi as u16 + 1);
        }
        x.cuts.sort_unstable();
        x.cuts.dedup();
        for i in 1..x.cuts.len() {
            let (a, b) = (x.cuts[i - 1], x.cuts[i] - 1);
            // One buffer for every piece, and a lookup by slice: a set is
            // copied out only when it turns out to be new.
            x.target.clear();
            let Expander { edges, target, .. } = &mut *x;
            target.extend(
                edges
                    .iter()
                    .filter(|&&(lo, hi, _)| lo as u16 <= a && b <= hi as u16)
                    .map(|&(_, _, t)| t),
            );
            if x.target.is_empty() {
                continue;
            }
            let Expander {
                target,
                stack,
                seen,
                ..
            } = &mut *x;
            self.closure(target, stack, seen);
            f(a as u8, b as u8, &x.target)?;
        }
        Some(())
    }
}

/// Scratch for [`Nfa::expand`], reused across every state of one
/// construction (eager or lazy).
#[derive(Debug)]
struct Expander {
    seen: Vec<u64>,
    stack: Vec<u32>,
    edges: Vec<(u8, u8, u32)>,
    cuts: Vec<u16>,
    target: Vec<u32>,
}

impl Expander {
    fn new(nfa: &Nfa) -> Self {
        Expander {
            seen: vec![0u64; nfa.ranges.len().div_ceil(64)],
            stack: Vec::new(),
            edges: Vec::new(),
            cuts: Vec::new(),
            target: Vec::new(),
        }
    }

    /// `start`'s epsilon closure: the construction's first state set.
    fn start_set(&mut self, nfa: &Nfa, start: u32) -> Vec<u32> {
        let mut first = vec![start];
        nfa.closure(&mut first, &mut self.stack, &mut self.seen);
        first
    }
}

/// Rewrites every transition into a state that cannot reach an accepting
/// state as [`DEAD`], and renumbers the survivors densely.
///
/// Works from the range transitions `determinize` produced (`(state, lo, hi,
/// target)`, each state's in ascending byte order) and only builds the dense
/// table at the end. Scanning a 256-entry row per state in each of three
/// passes was most of what compiling a small pattern such as `t1[0-9]` cost
/// -- once per query.
// ARITH: `n = accept.len() <= MAX_DFA_STATES` (10 000), so state numbers fit
// a `u32` and `ns << 8 | b` (`b < 256`) stays under 2^22, inside `out`, which
// holds 256 entries per live state; `count` counts live states, at most `n`,
// so `count += 1` and `count << 8` cannot overflow. `hi + 1` is on a `u8`
// widened to `usize`.
#[allow(clippy::arithmetic_side_effects)]
fn prune(trans: &[(u32, u8, u8, u32)], accept: Vec<bool>) -> ByteDfa {
    let n = accept.len();
    // Reverse edges, then a search back from the accepting states.
    let mut rev: Vec<Vec<u32>> = vec![Vec::new(); n];
    for &(s, _, _, t) in trans {
        if rev[t as usize].last() != Some(&s) {
            rev[t as usize].push(s);
        }
    }
    let mut live = accept.clone();
    let mut stack: Vec<u32> = (0..n as u32).filter(|&s| accept[s as usize]).collect();
    while let Some(t) = stack.pop() {
        for &s in &rev[t as usize] {
            if !live[s as usize] {
                live[s as usize] = true;
                stack.push(s);
            }
        }
    }
    let mut remap = vec![DEAD; n];
    let mut count = 0u32;
    for s in 0..n {
        if live[s] {
            remap[s] = count;
            count += 1;
        }
    }
    let mut out = vec![DEAD; (count as usize) << 8];
    let mut out_accept = vec![false; count as usize];
    for s in 0..n {
        if remap[s] != DEAD {
            out_accept[remap[s] as usize] = accept[s];
        }
    }
    // Each live state's transitions into live states, filled into its row
    // and merged into maximal runs of live bytes as they go (adjacent ranges
    // join whatever their targets, exactly as a scan of the row would).
    let mut live_ranges: Vec<(u8, u8)> = Vec::new();
    let mut live_offsets = vec![0u32; count as usize + 1];
    let mut current = DEAD;
    for &(s, lo, hi, t) in trans {
        let (ns, nt) = (remap[s as usize], remap[t as usize]);
        if ns == DEAD || nt == DEAD {
            continue;
        }
        let row = (ns as usize) << 8;
        out[row + lo as usize..=row + hi as usize].fill(nt);
        if ns != current {
            // Offsets are filled for every state up to this one: states
            // with no live transition get an empty run list.
            for o in &mut live_offsets[(current.wrapping_add(1)) as usize..=ns as usize] {
                *o = live_ranges.len() as u32;
            }
            current = ns;
            live_ranges.push((lo, hi));
        } else {
            match live_ranges.last_mut() {
                Some(last) if last.1 as usize + 1 == lo as usize => last.1 = hi,
                _ => live_ranges.push((lo, hi)),
            }
        }
    }
    for o in &mut live_offsets[(current.wrapping_add(1)) as usize..] {
        *o = live_ranges.len() as u32;
    }
    let (table, classes, stride) = byte_classes(&out, count as usize);
    ByteDfa {
        table,
        classes,
        stride,
        accept: out_accept,
        start: remap[0],
        live_ranges,
        live_offsets,
    }
}

/// Compresses a dense `states x 256` table into byte classes: two bytes share
/// a class when every state sends them to the same place. Returns the
/// compressed table, the byte-to-class map and the class count.
// ARITH: `states <= MAX_DFA_STATES` and `b < 256`, so `s << 8 | b` and
// `s * stride + c` stay inside tables of `states * 256` entries; at most 256
// distinct columns exist, so a class id fits a `u8`.
#[allow(clippy::arithmetic_side_effects)]
fn byte_classes(dense: &[u32], states: usize) -> (Vec<u32>, Box<[u8; 256]>, usize) {
    let mut classes = Box::new([0u8; 256]);
    let mut reps: Vec<usize> = Vec::new();
    let mut ids: HashMap<Vec<u32>, u8, BuildHasherDefault<SetHasher>> = HashMap::default();
    for b in 0..256usize {
        let column: Vec<u32> = (0..states).map(|s| dense[s << 8 | b]).collect();
        let next = reps.len();
        let id = *ids.entry(column).or_insert_with(|| {
            reps.push(b);
            next as u8
        });
        classes[b] = id;
    }
    let stride = reps.len().max(1);
    let mut table = Vec::with_capacity(states * stride);
    for s in 0..states {
        for &b in &reps {
            table.push(dense[s << 8 | b]);
        }
    }
    (table, classes, stride)
}

/// Splits `lo..=hi` into sequences of per-byte ranges whose UTF-8 encodings
/// cover exactly that code point range (surrogates excluded) -- the core of
/// `UTF32ToUTF8`. Each sequence is 1-4 `(lo, hi)` byte ranges, one per
/// encoded byte.
//
// ARITH: `hi` is clamped to `MAX_CODE_POINT` (0x10FFFF) first, and every
// range pushed lies inside `lo..=hi`, so `boundary + 1 <= 0x1_0000`, `m <
// 2^18` and `(s | m) + 1 < 2^21`. `(e & !m) - 1` runs only where `s & !m !=
// e & !m` with `s & m == 0`, i.e. `e & !m > s & !m == s >= 0`, so it is at
// least 0. `6 * i <= 18`.
#[allow(clippy::arithmetic_side_effects)]
pub fn utf8_sequences(lo: u32, hi: u32, out: &mut Vec<Vec<(u8, u8)>>) {
    let mut stack = vec![(lo, hi.min(MAX_CODE_POINT))];
    'next: while let Some((s, e)) = stack.pop() {
        if s > e {
            continue;
        }
        // Surrogates have no UTF-8 encoding.
        if s <= 0xDFFF && e >= 0xD800 {
            if s < 0xD800 {
                stack.push((s, 0xD7FF));
            }
            if e > 0xDFFF {
                stack.push((0xE000, e));
            }
            continue;
        }
        // Split where the encoded length changes.
        for boundary in [0x7F, 0x7FF, 0xFFFF] {
            if s <= boundary && boundary < e {
                stack.push((boundary + 1, e));
                stack.push((s, boundary));
                continue 'next;
            }
        }
        if e <= 0x7F {
            out.push(vec![(s as u8, e as u8)]);
            continue;
        }
        // Align on continuation-byte boundaries so each byte position spans
        // one contiguous range.
        for i in 1..4 {
            let m: u32 = (1 << (6 * i)) - 1;
            if s & !m != e & !m {
                if s & m != 0 {
                    stack.push(((s | m) + 1, e));
                    stack.push((s, s | m));
                    continue 'next;
                }
                if e & m != m {
                    stack.push((e & !m, e));
                    stack.push((s, (e & !m) - 1));
                    continue 'next;
                }
            }
        }
        let (a, b) = (encode_utf8(s), encode_utf8(e));
        debug_assert_eq!(a.len(), b.len());
        out.push(a.iter().zip(&b).map(|(&x, &y)| (x, y)).collect());
    }
}

fn encode_utf8(cp: u32) -> Vec<u8> {
    let c = char::from_u32(cp).expect("surrogates are split out before encoding");
    let mut buf = [0u8; 4];
    c.encode_utf8(&mut buf).as_bytes().to_vec()
}

/// The complement of a set of inclusive code point ranges within
/// `0..=MAX_CODE_POINT`.
pub fn complement_ranges(ranges: &[(u32, u32)]) -> Vec<(u32, u32)> {
    let mut sorted: Vec<(u32, u32)> = ranges.to_vec();
    sorted.sort_unstable();
    let mut out = Vec::new();
    let mut next = 0u32;
    for (lo, hi) in sorted {
        if lo > next {
            // ARITH: `lo > next >= 0`.
            #[allow(clippy::arithmetic_side_effects)]
            out.push((next, lo - 1));
        }
        next = next.max(hi.saturating_add(1));
    }
    if next <= MAX_CODE_POINT {
        out.push((next, MAX_CODE_POINT));
    }
    out
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn dfa_of(build: impl FnOnce(&mut Nfa, u32, u32)) -> ByteDfa {
        let mut nfa = Nfa::new();
        let s = nfa.state().unwrap();
        let a = nfa.state().unwrap();
        build(&mut nfa, s, a);
        nfa.determinize(s, a).expect("determinize")
    }

    #[test]
    fn utf8_sequences_cover_exactly_the_code_point_range() {
        for (lo, hi) in [
            (0u32, 0x7F),
            (0x41, 0x5A),
            (0x7E, 0x81),
            (0x7FF, 0x800),
            (0x100, 0x2000),
            (0xD7FF, 0xE001),
            (0xFFFF, 0x10000),
            (0, MAX_CODE_POINT),
            (0x10FFFE, MAX_CODE_POINT),
        ] {
            let dfa = dfa_of(|nfa, s, a| {
                nfa.code_points(s, lo, hi, a).unwrap();
            });
            // Sample the whole space densely at the edges and sparsely inside.
            let probes = (0..0x900u32)
                .chain((0xD700..0xE100).step_by(7))
                .chain((0xF000..0x10100).step_by(13))
                .chain((0x10F000..=MAX_CODE_POINT).step_by(3));
            for cp in probes {
                let Some(c) = char::from_u32(cp) else {
                    continue;
                };
                let mut buf = [0u8; 4];
                let bytes = c.encode_utf8(&mut buf).as_bytes();
                assert_eq!(
                    dfa.run(bytes),
                    (lo..=hi).contains(&cp),
                    "range {lo:#x}..={hi:#x}, code point {cp:#x}"
                );
            }
        }
    }

    #[test]
    fn a_dead_state_is_reported_at_the_first_dead_byte() {
        // "ab" then any byte, anything after that dead.
        let dfa = dfa_of(|nfa, s, a| {
            let t = nfa.bytes(s, b"ab").unwrap();
            nfa.range(t, 0, 255, a);
        });
        let mut w = DfaWalker::new(&dfa);
        assert_eq!(w.feed(&dfa, b"ab"), Verdict::Reject);
        assert_eq!(w.feed(&dfa, b"abc"), Verdict::Accept);
        assert_eq!(w.feed(&dfa, b"abcd"), Verdict::DeadAt(4));
        assert_eq!(w.feed(&dfa, b"ac"), Verdict::DeadAt(2));
        assert_eq!(w.feed(&dfa, b"acz"), Verdict::DeadAt(2));
        assert_eq!(w.feed(&dfa, b"abz"), Verdict::Accept);
        assert_eq!(w.feed(&dfa, b""), Verdict::Reject);
        assert_eq!(w.feed(&dfa, b"b"), Verdict::DeadAt(1));
    }

    #[test]
    fn an_empty_language_starts_dead() {
        let dfa = dfa_of(|_, _, _| {});
        assert_eq!(dfa.start(), DEAD);
        assert!(!dfa.run(b""));
        let mut w = DfaWalker::new(&dfa);
        assert_eq!(w.feed(&dfa, b"x"), Verdict::DeadAt(0));
    }

    #[test]
    fn complement_ranges_is_the_exact_complement() {
        assert_eq!(complement_ranges(&[]), vec![(0, MAX_CODE_POINT)]);
        assert_eq!(
            complement_ranges(&[(0x61, 0x7A), (0x30, 0x39)]),
            vec![(0, 0x2F), (0x3A, 0x60), (0x7B, MAX_CODE_POINT)]
        );
        assert_eq!(complement_ranges(&[(0, MAX_CODE_POINT)]), vec![]);
        assert_eq!(
            complement_ranges(&[(0, 5), (3, 9)]),
            vec![(10, MAX_CODE_POINT)]
        );
    }

    /// Random terms over an alphabet that exercises ASCII, multi-byte UTF-8
    /// and ill-formed bytes, so every superset path is reached.
    fn random_terms(seed: u64, n: usize) -> Vec<Vec<u8>> {
        let mut s = seed;
        let alphabet: &[&[u8]] = &[
            b"a",
            b"b",
            b"c",
            b"t",
            b"1",
            b"2",
            b"9",
            b"0",
            b"x",
            b"-",
            b".",
            b"_",
            "é".as_bytes(),
            "€".as_bytes(),
            "𝄞".as_bytes(),
            &[0xFF],
            &[0xC3],
            &[0x80],
        ];
        (0..n)
            .map(|_| {
                s ^= s << 13;
                s ^= s >> 7;
                s ^= s << 17;
                let len = (s % 7) as usize;
                let mut t = Vec::new();
                let mut x = s;
                for _ in 0..len {
                    x = x.rotate_left(9) ^ 0x9E37_79B9;
                    t.extend_from_slice(alphabet[(x % alphabet.len() as u64) as usize]);
                }
                t
            })
            .collect()
    }

    /// The contract every caller relies on: whatever the exact matcher
    /// accepts, the DFA accepts -- and a prefix the DFA calls dead has no
    /// accepted extension among the sample.
    fn assert_superset(
        dfa: &ByteDfa,
        terms: &[Vec<u8>],
        exact: impl Fn(&[u8]) -> bool,
        what: &str,
    ) {
        // The precomputed live runs (`prune` builds them from the range
        // transitions) against a scan of the table itself, for every state and
        // byte: the next live byte, and runs that are maximal and ordered.
        for s in 0..dfa.num_states() as u32 {
            for after in 0..=255u8 {
                let want = (after.saturating_add(1)..=255)
                    .filter(|_| after < 255)
                    .find(|&b| dfa.step(s, b) != DEAD);
                assert_eq!(
                    dfa.next_live_byte(s, after),
                    want,
                    "{what}: state {s} after {after}"
                );
            }
            let (lo, hi) = (
                dfa.live_offsets[s as usize] as usize,
                dfa.live_offsets[s as usize + 1] as usize,
            );
            for pair in dfa.live_ranges[lo..hi].windows(2) {
                assert!(
                    pair[0].1 as usize + 1 < pair[1].0 as usize,
                    "{what}: state {s} runs {pair:?} not maximal"
                );
            }
        }
        assert_eq!(dfa.live_offsets.len(), dfa.num_states() + 1, "{what}");

        let mut sorted = terms.to_vec();
        sorted.sort();
        sorted.dedup();
        let mut w = DfaWalker::new(dfa);
        for t in &sorted {
            let verdict = w.feed(dfa, t);
            if exact(t) {
                assert_eq!(verdict, Verdict::Accept, "{what}: exact accepts {t:?}");
            }
            if let Verdict::DeadAt(k) = verdict {
                for u in sorted.iter().filter(|u| u.starts_with(&t[..k])) {
                    assert!(
                        !exact(u),
                        "{what}: prefix {:?} dead but {u:?} matches",
                        &t[..k]
                    );
                }
            }
        }
    }

    #[test]
    fn regexp_automata_accept_a_superset_and_are_exact_for_plain_patterns() {
        let terms = random_terms(0xABCDEF, 4000);
        for (pattern, exact_expected) in [
            ("t1[0-9]", true),
            ("t.9", true),
            ("[a-c]+", true),
            ("(ab|t)*1", true),
            ("[^a]x?", true),
            ("a{2,3}b", true),
            ("é.", true),
            ("#", true),
            ("@", true),
            ("\"ab\"c", true),
            ("[a-c]+&.*b.*", true),
            ("t<1-99>", true),
            ("..€", true),
        ] {
            let p = crate::regexp::RegexpPattern::parse(pattern).expect("parse");
            let dfa = p.to_dfa().expect("dfa");
            assert_superset(&dfa, &terms, |t| p.matches(t), pattern);
            if exact_expected {
                for t in &terms {
                    assert_eq!(dfa.run(t), p.matches(t), "{pattern} exact on {t:?}");
                }
            }
        }
    }

    /// Intersection and numeric intervals compile to exact automata: every
    /// digit string up to four wide, with and without leading zeros and with
    /// a letter either side, gets the same answer from the DFA as from the
    /// exact matcher.
    #[test]
    fn regexp_dfas_are_exact_for_intersection_and_intervals() {
        let mut terms: Vec<Vec<u8>> = Vec::new();
        for n in 0..10_000u32 {
            for w in 1..=5usize {
                let s = format!("{n:0w$}");
                if s.len() == w {
                    terms.push(s.clone().into_bytes());
                    terms.push(format!("t{s}").into_bytes());
                    terms.push(format!("{s}x").into_bytes());
                }
            }
        }
        terms.extend(random_terms(0xFEED, 3000));
        for pattern in [
            "<0-0>",
            "<0-9>",
            "<5-123>",
            "<1-99>",
            "<007-120>",
            "<00-99>",
            "<10-9999>",
            "<0-4294967295>",
            "<3000000000-4294967295>",
            "<999-1000>",
            "t<1-99>",
            "<1-12>x",
            "<12-12>",
            "[a-c]+&.*b.*",
            "t.*&.*9",
            "(t1|t2).*&.*[0-4]",
            "<0-500>&<250-1000>",
            "#&.*",
            ".*&.*",
            "[0-9]{3}&<100-199>",
        ] {
            let p = crate::regexp::RegexpPattern::parse(pattern).expect("parse");
            let dfa = p.to_dfa().expect("dfa");
            for t in &terms {
                assert_eq!(
                    dfa.run(t),
                    p.matches(t),
                    "{pattern} on {:?}",
                    String::from_utf8_lossy(t)
                );
            }
        }
    }

    #[test]
    fn wildcard_automata_accept_a_superset_and_are_exact_on_ascii() {
        let terms = random_terms(0x1234, 4000);
        for pattern in [
            &b"t?9"[..],
            b"t*",
            b"*1",
            b"a?c*",
            b"?",
            b"\\*x",
            b"*",
            b"t??",
        ] {
            let p = crate::wildcard::WildcardPattern::new(pattern);
            let dfa = p.to_dfa().expect("dfa");
            assert_superset(
                &dfa,
                &terms,
                |t| p.matches(t),
                &String::from_utf8_lossy(pattern),
            );
            for t in terms.iter().filter(|t| t.is_ascii()) {
                assert_eq!(dfa.run(t), p.matches(t), "{pattern:?} exact on ascii {t:?}");
            }
        }
    }

    #[test]
    fn levenshtein_automata_accept_every_term_within_the_budget() {
        let terms = random_terms(0x77, 6000);
        for (target, edits, prefix, transpositions) in [
            (&b"t123"[..], 2u8, 0usize, true),
            (b"t12", 1, 0, true),
            (b"abc", 2, 1, false),
            ("té€".as_bytes(), 1, 0, true),
            (b"ba", 2, 0, true),
            (b"", 1, 0, true),
        ] {
            let m = crate::fuzzy::FuzzyMatch::new(target, edits, prefix, transpositions);
            let dfa = m.to_dfa(edits).expect("dfa");
            assert_superset(
                &dfa,
                &terms,
                |t| m.edits_within(t, edits).is_some(),
                &format!("{:?}~{edits}", String::from_utf8_lossy(target)),
            );
        }
    }

    #[test]
    fn construction_limits_refuse_rather_than_blow_up() {
        let mut nfa = Nfa::new();
        let mut last = None;
        for _ in 0..=MAX_NFA_STATES {
            last = nfa.state();
        }
        assert!(last.is_none());
    }
}
