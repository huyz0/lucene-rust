//! A standalone byte-sequence interning pool, analogous in spirit to
//! Lucene's `BytesRefHash` (used e.g. by indexing chains to deduplicate term
//! bytes before they hit a term dictionary). This is **not** a port of
//! `BytesRefHash.java` byte-for-byte (that class exposes sort/compact/rehash
//! machinery tied to Lucene's `ByteBlockPool` and int allocation strategy) --
//! it is a simpler, self-contained primitive with the same core value
//! proposition: given arbitrary byte strings that recur, hand back a stable,
//! cheap-to-copy integer handle instead of re-allocating or re-storing the
//! same bytes every time.
//!
//! **Not wired into any indexing or query path yet.** This module is a
//! tested building block only; integrating it into the postings/indexing
//! chain (where the real allocation savings would materialize) is future
//! work tracked in `PLAN.md`.

use std::collections::hash_map::RandomState;
use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher};

/// A stable, cheap-to-copy handle for a byte sequence previously interned by
/// a [`TermInterner`]. Two calls to [`TermInterner::intern`] with
/// byte-identical input always return the same `TermId`; different input
/// bytes always get different `TermId`s.
///
/// `TermId`s are only meaningful relative to the [`TermInterner`] that
/// produced them -- mixing IDs across two different interner instances is a
/// logic error (not memory-unsafe, since lookup just returns `None` for an
/// out-of-range ID, but the bytes returned would be for the wrong pool).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TermId(u32);

impl TermId {
    /// The raw index into the interner's storage. Exposed for callers that
    /// need to persist or compare IDs externally (e.g. as a dense array
    /// index into parallel per-term data).
    pub fn index(self) -> u32 {
        self.0
    }
}

/// A simple interning pool that deduplicates byte sequences (terms) and
/// hands back a small, `Copy` [`TermId`] in place of the bytes themselves.
///
/// Interning the same bytes twice is idempotent: the second call is a hash
/// lookup that returns the existing ID without storing a second copy. This
/// is the core win over passing owned `Vec<u8>`/`String` around every time
/// the same term recurs (e.g. across postings for a high-frequency term, or
/// repeated query terms).
///
/// Each distinct term's bytes are stored once, end to end in one buffer;
/// the lookup table holds only each term's (randomly keyed, so
/// input-controlled terms cannot force collisions) hash, hashed once per
/// call and never again on growth.
#[derive(Debug, Default, Clone)]
pub struct TermInterner {
    /// Every distinct term's bytes, in `TermId` order.
    bytes: Vec<u8>,
    /// `ends[i]`: where term `i` ends in `bytes` (it starts at `ends[i-1]`,
    /// or 0).
    ends: Vec<usize>,
    /// `next[i]`: the term interned before `i` with the same hash, or
    /// [`NO_TERM`].
    next: Vec<u32>,
    /// Term hash -> the latest term with that hash (the head of its chain).
    heads: HashMap<u64, u32, BuildHasherDefault<HashIsKey>>,
    hasher: RandomState,
}

/// The end of a [`TermInterner`] hash chain.
const NO_TERM: u32 = u32::MAX;

/// The hasher of a table keyed by a hash: it passes the key through.
#[derive(Debug, Default, Clone, Copy)]
struct HashIsKey(u64);

impl Hasher for HashIsKey {
    fn write(&mut self, bytes: &[u8]) {
        // Only `u64` keys reach this hasher; fold anything else in anyway.
        for &b in bytes {
            self.0 = self.0.rotate_left(8) ^ u64::from(b);
        }
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.0 = i;
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }
}

impl TermInterner {
    /// Creates an empty interner.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates an empty interner with storage pre-reserved for `capacity`
    /// distinct terms.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            bytes: Vec::new(),
            ends: Vec::with_capacity(capacity),
            next: Vec::with_capacity(capacity),
            heads: HashMap::with_capacity_and_hasher(capacity, Default::default()),
            hasher: RandomState::new(),
        }
    }

    fn term(&self, i: usize) -> &[u8] {
        let start = if i == 0 { 0 } else { self.ends[i - 1] };
        &self.bytes[start..self.ends[i]]
    }

    /// Interns `term`, returning the existing [`TermId`] if these exact bytes
    /// were interned before, or allocating and returning a new one otherwise.
    ///
    /// The empty byte string is a valid term and interns like any other.
    ///
    /// # Panics
    /// When more than `u32::MAX - 1` distinct terms are interned.
    pub fn intern(&mut self, term: &[u8]) -> TermId {
        let hash = self.hasher.hash_one(term);
        let head = self.heads.get(&hash).copied().unwrap_or(NO_TERM);
        let mut at = head;
        while at != NO_TERM {
            if self.term(at as usize) == term {
                return TermId(at);
            }
            at = self.next[at as usize];
        }
        // New distinct term: assign the next sequential ID.
        let id = u32::try_from(self.ends.len())
            .ok()
            .filter(|&id| id != NO_TERM)
            .expect("fewer than u32::MAX terms");
        self.bytes.extend_from_slice(term);
        self.ends.push(self.bytes.len());
        self.next.push(head);
        self.heads.insert(hash, id);
        TermId(id)
    }

    /// Convenience wrapper over [`Self::intern`] for UTF-8 input.
    pub fn intern_str(&mut self, term: &str) -> TermId {
        self.intern(term.as_bytes())
    }

    /// Looks up the original bytes for a previously-returned [`TermId`].
    /// Returns `None` if `id` was not produced by this interner (e.g. from a
    /// different `TermInterner` instance, or a stale/out-of-range value).
    pub fn get(&self, id: TermId) -> Option<&[u8]> {
        let i = id.0 as usize;
        (i < self.ends.len()).then(|| self.term(i))
    }

    /// The number of *distinct* terms interned so far -- the whole point of
    /// interning is that this stays far below the number of [`Self::intern`]
    /// calls when input has repeats.
    pub fn len(&self) -> usize {
        self.ends.len()
    }

    /// Whether no terms have been interned yet.
    pub fn is_empty(&self) -> bool {
        self.ends.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_bytes_intern_to_same_id() {
        let mut pool = TermInterner::new();
        let a = pool.intern(b"apple");
        let b = pool.intern(b"apple");
        assert_eq!(a, b);
        assert_eq!(pool.len(), 1);
    }

    #[test]
    fn different_bytes_get_different_ids() {
        let mut pool = TermInterner::new();
        let a = pool.intern(b"apple");
        let b = pool.intern(b"banana");
        assert_ne!(a, b);
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn round_trip_lookup_by_id() {
        let mut pool = TermInterner::new();
        let a = pool.intern(b"lucene");
        let b = pool.intern(b"rust");
        assert_eq!(pool.get(a), Some(&b"lucene"[..]));
        assert_eq!(pool.get(b), Some(&b"rust"[..]));
    }

    #[test]
    fn empty_term_is_valid_and_distinct() {
        let mut pool = TermInterner::new();
        let empty1 = pool.intern(b"");
        let empty2 = pool.intern(b"");
        let nonempty = pool.intern(b"x");
        assert_eq!(empty1, empty2);
        assert_ne!(empty1, nonempty);
        assert_eq!(pool.get(empty1), Some(&b""[..]));
        assert_eq!(pool.len(), 2);
    }

    #[test]
    fn lookup_of_unknown_id_returns_none() {
        let pool = TermInterner::new();
        assert_eq!(pool.get(TermId(0)), None);

        let mut pool2 = TermInterner::new();
        pool2.intern(b"only-term");
        assert_eq!(pool2.get(TermId(5)), None);
    }

    #[test]
    fn index_returns_sequential_assignment_order() {
        let mut pool = TermInterner::new();
        let a = pool.intern(b"first");
        let b = pool.intern(b"second");
        let c = pool.intern(b"third");
        assert_eq!(a.index(), 0);
        assert_eq!(b.index(), 1);
        assert_eq!(c.index(), 2);
    }

    #[test]
    fn intern_str_matches_intern_bytes() {
        let mut pool = TermInterner::new();
        let a = pool.intern_str("hello");
        let b = pool.intern(b"hello");
        assert_eq!(a, b);
    }

    #[test]
    fn is_empty_reflects_state() {
        let mut pool = TermInterner::new();
        assert!(pool.is_empty());
        pool.intern(b"x");
        assert!(!pool.is_empty());
    }

    #[test]
    fn stress_dedup_reduces_stored_entries() {
        // Simulate a realistic postings-like stream: a small vocabulary of
        // terms recurring across many "postings", the way the same term
        // bytes show up in document after document. Interning must collapse
        // this down to exactly the distinct-vocabulary count, not one entry
        // per call.
        let vocab: Vec<String> = (0..50).map(|i| format!("term-{i}")).collect();
        let mut pool = TermInterner::with_capacity(vocab.len());

        let total_calls = 20_000;
        let mut ids = Vec::with_capacity(total_calls);
        for i in 0..total_calls {
            let term = &vocab[i % vocab.len()];
            ids.push(pool.intern_str(term));
        }

        // Dedup actually happened: far fewer stored terms than calls.
        assert_eq!(pool.len(), vocab.len());
        assert!(pool.len() < total_calls);

        // Every occurrence of the same vocabulary word maps to the same ID,
        // and round-trips back to the right bytes.
        for i in 0..total_calls {
            let expected_word = &vocab[i % vocab.len()];
            assert_eq!(
                pool.get(ids[i]).unwrap(),
                expected_word.as_bytes(),
                "call {i}"
            );
            assert_eq!(ids[i], ids[i % vocab.len()], "call {i}");
        }
    }

    /// Terms sharing a hash chain: every one is still found, and only by
    /// its own bytes (the chain walk compares them).
    #[test]
    fn colliding_hashes_chain() {
        let mut pool = TermInterner::new();
        let a = pool.intern(b"a");
        let b = pool.intern(b"b");
        // Force a collision: hang `b` under `a`'s hash too.
        let ha = pool.hasher.hash_one(&b"a"[..]);
        pool.next[b.0 as usize] = a.0;
        pool.heads.insert(ha, b.0);
        assert_eq!(pool.intern(b"a"), a);
        assert_eq!(pool.intern(b"b"), b);
        let c = pool.intern(b"c");
        assert_eq!((pool.len(), c.index()), (3, 2));
        let mut h = HashIsKey::default();
        h.write(&[1, 2]);
        assert_eq!(h.finish(), 0x102);
    }

    #[test]
    fn many_distinct_terms_each_get_unique_ids() {
        let mut pool = TermInterner::new();
        let n = 5_000;
        let mut ids = Vec::with_capacity(n);
        for i in 0..n {
            ids.push(pool.intern_str(&format!("unique-{i}")));
        }
        assert_eq!(pool.len(), n);

        // All IDs distinct.
        let mut sorted = ids.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(sorted.len(), n);

        // Round trip every one.
        for (i, id) in ids.iter().enumerate() {
            assert_eq!(pool.get(*id).unwrap(), format!("unique-{i}").as_bytes());
        }
    }
}
