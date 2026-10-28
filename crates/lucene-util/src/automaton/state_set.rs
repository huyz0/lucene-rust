//! `IntSet`, `FrozenIntSet` and `StateSet`: the NFA-state-set keys of subset
//! construction.
//!
//! [`StateSet`] is a multiset of NFA states (a state stays in while any
//! transition into it covers the current interval); [`StateSet::freeze`]
//! snapshots its distinct members, sorted, as the [`FrozenIntSet`] key of a
//! DFA state. The hash is Lucene's (`size + sum(BitMixer.mix(state))`), kept
//! so the two types hash alike, as Java requires of equal `IntSet`s.

use std::hash::{Hash, Hasher};

/// hppc's `BitMixer.mix32`.
pub(crate) fn bit_mix(k: i32) -> i32 {
    let mut k = k as u32;
    k = (k ^ (k >> 16)).wrapping_mul(0x85eb_ca6b);
    k = (k ^ (k >> 13)).wrapping_mul(0xc2b2_ae35);
    (k ^ (k >> 16)) as i32
}

/// `IntSet`: a sorted set of ints with Lucene's `longHashCode`.
pub trait IntSet {
    /// `getArray()`: the members, ascending.
    fn get_array(&self) -> &[i32];
    /// `longHashCode()`.
    fn long_hash_code(&self) -> i64;
    /// `size()`.
    fn size(&self) -> usize {
        self.get_array().len()
    }
}

/// `FrozenIntSet`: an immutable [`IntSet`] tagged with the DFA state it
/// became.
#[derive(Clone, Debug)]
pub struct FrozenIntSet {
    /// Members, ascending.
    pub values: Vec<i32>,
    /// The DFA state this set was assigned.
    pub state: i32,
    hash_code: i64,
}

impl FrozenIntSet {
    /// `new FrozenIntSet(values, hashCode, state)`.
    pub fn new(values: Vec<i32>, hash_code: i64, state: i32) -> Self {
        FrozenIntSet {
            values,
            state,
            hash_code,
        }
    }

    /// The singleton `{s}` as determinize's initial set.
    #[cfg(test)]
    pub(crate) fn singleton(s: i32, state: i32) -> Self {
        FrozenIntSet::new(vec![s], i64::from(bit_mix(s)) + 1, state)
    }
}

impl IntSet for FrozenIntSet {
    fn get_array(&self) -> &[i32] {
        &self.values
    }
    fn long_hash_code(&self) -> i64 {
        self.hash_code
    }
}

impl PartialEq for FrozenIntSet {
    fn eq(&self, other: &Self) -> bool {
        self.hash_code == other.hash_code && self.values == other.values
    }
}
impl Eq for FrozenIntSet {}
impl Hash for FrozenIntSet {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.hash_code.hash(state);
    }
}

/// Lucene's `longHashCode` of a set given as its distinct members:
/// `size + sum(BitMixer.mix(member))`.
pub(crate) fn set_hash(members: &[i32]) -> i64 {
    members.iter().fold(members.len() as i64, |h, &k| {
        h.wrapping_add(i64::from(bit_mix(k)))
    })
}

fn state_index(state: i32) -> usize {
    usize::try_from(state).expect("negative NFA state")
}

/// `StateSet`: a counted multiset of states.
///
/// NFA states are dense `0..n`, so the counts live in a `Vec` indexed by
/// state (grown on demand) rather than a hash map, the distinct members in
/// an unordered `Vec` with swap-remove positions, and the hash is kept up to
/// date as members come and go (Java's `hashUpdated` cache). The sorted
/// member array is rebuilt only when asked for after a change.
#[derive(Clone, Debug, Default)]
pub struct StateSet {
    /// Occurrences per state.
    counts: Vec<u32>,
    /// Position of each present state in `members`.
    pos: Vec<u32>,
    /// The distinct members, unordered.
    members: Vec<i32>,
    /// `sum(bit_mix(member))` over `members`.
    mix_sum: i64,
    /// `members`, sorted, when `sorted_valid`.
    sorted: Vec<i32>,
    sorted_valid: bool,
}

impl StateSet {
    /// `new StateSet(capacity)`.
    pub fn new(capacity: usize) -> Self {
        StateSet {
            members: Vec::with_capacity(capacity),
            sorted_valid: true,
            ..StateSet::default()
        }
    }

    /// `incr(state)`: add one occurrence.
    ///
    /// # Panics
    /// If `state` is negative.
    #[inline]
    pub fn incr(&mut self, state: i32) {
        let s = state_index(state);
        if s >= self.counts.len() {
            self.counts.resize(s + 1, 0);
            self.pos.resize(s + 1, 0);
        }
        self.counts[s] += 1;
        if self.counts[s] == 1 {
            self.pos[s] = self.members.len() as u32;
            self.members.push(state);
            self.mix_sum = self.mix_sum.wrapping_add(i64::from(bit_mix(state)));
            self.sorted_valid = false;
        }
    }

    /// `decr(state)`: remove one occurrence.
    ///
    /// # Panics
    /// If `state` is not in the set (Java asserts this).
    #[inline]
    pub fn decr(&mut self, state: i32) {
        let s = state_index(state);
        let c = self
            .counts
            .get_mut(s)
            .filter(|c| **c > 0)
            .expect("decr of absent state");
        *c -= 1;
        if *c == 0 {
            let p = self.pos[s] as usize;
            self.members.swap_remove(p);
            if let Some(&moved) = self.members.get(p) {
                self.pos[state_index(moved)] = p as u32;
            }
            self.mix_sum = self.mix_sum.wrapping_sub(i64::from(bit_mix(state)));
            self.sorted_valid = false;
        }
    }

    /// `reset()`: empty the set.
    pub fn reset(&mut self) {
        for &m in &self.members {
            self.counts[state_index(m)] = 0;
        }
        self.members.clear();
        self.mix_sum = 0;
        self.sorted.clear();
        self.sorted_valid = true;
    }

    /// `freeze(state)`: the current distinct members as a key for `state`.
    pub fn freeze(&mut self, state: i32) -> FrozenIntSet {
        let hash = self.long_hash_code();
        FrozenIntSet::new(self.sorted().to_vec(), hash, state)
    }

    fn sorted(&mut self) -> &[i32] {
        if !self.sorted_valid {
            self.sorted.clear();
            self.sorted.extend_from_slice(&self.members);
            self.sorted.sort_unstable();
            self.sorted_valid = true;
        }
        &self.sorted
    }

    /// The distinct members, ascending (Java's `getArray`, which needs `&mut`
    /// here to refresh its cache).
    pub fn members(&mut self) -> &[i32] {
        self.sorted()
    }

    /// `size()`: distinct members.
    pub fn len(&self) -> usize {
        self.members.len()
    }

    /// `size() == 0`.
    pub fn is_empty(&self) -> bool {
        self.members.is_empty()
    }

    /// `longHashCode()`.
    pub fn long_hash_code(&self) -> i64 {
        (self.members.len() as i64).wrapping_add(self.mix_sum)
    }
}

/// Interns sorted state sets as dense ids `0, 1, 2, ...` in first-seen order:
/// determinize's `newstate` map, without a `Vec` key per DFA state. Keys
/// are stored back to back in one buffer; the open-addressed table holds
/// `id + 1` (0 = empty) and is probed with each key's precomputed
/// [`set_hash`], so a lookup hashes nothing and allocates nothing.
#[derive(Debug, Default)]
pub(crate) struct SetInterner {
    table: Vec<u32>,
    hashes: Vec<i64>,
    /// Key `id` is `data[offsets[id]..offsets[id + 1]]`.
    offsets: Vec<usize>,
    data: Vec<i32>,
}

impl SetInterner {
    pub(crate) fn new() -> Self {
        SetInterner {
            table: vec![0; 16],
            hashes: Vec::new(),
            offsets: vec![0],
            data: Vec::new(),
        }
    }

    /// Number of interned sets.
    pub(crate) fn len(&self) -> usize {
        self.hashes.len()
    }

    /// The members of set `id`.
    pub(crate) fn get(&self, id: usize) -> &[i32] {
        &self.data[self.offsets[id]..self.offsets[id + 1]]
    }

    #[inline]
    fn slot(hash: i64, mask: usize) -> usize {
        // Fibonacci hashing: the sum-of-mixes hash is good in its low bits
        // but small sets cluster; one multiply spreads it.
        ((hash as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as usize & mask
    }

    /// The id of `key` (whose [`set_hash`] is `hash`), interning it if new;
    /// the flag is true when it was new.
    pub(crate) fn intern(&mut self, key: &[i32], hash: i64) -> (usize, bool) {
        debug_assert_eq!(hash, set_hash(key));
        let mask = self.table.len() - 1;
        let mut i = Self::slot(hash, mask);
        loop {
            let e = self.table[i];
            if e == 0 {
                break;
            }
            let id = e as usize - 1;
            if self.hashes[id] == hash && self.get(id) == key {
                return (id, false);
            }
            i = (i + 1) & mask;
        }
        let id = self.hashes.len();
        self.hashes.push(hash);
        self.data.extend_from_slice(key);
        self.offsets.push(self.data.len());
        self.table[i] = id as u32 + 1;
        if 2 * self.hashes.len() > self.table.len() {
            self.grow();
        }
        (id, true)
    }

    fn grow(&mut self) {
        let size = self.table.len() * 2;
        let mask = size - 1;
        let mut table = vec![0u32; size];
        for (id, &h) in self.hashes.iter().enumerate() {
            let mut i = Self::slot(h, mask);
            while table[i] != 0 {
                i = (i + 1) & mask;
            }
            table[i] = id as u32 + 1;
        }
        self.table = table;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_and_freezes() {
        let mut s = StateSet::new(4);
        assert!(s.is_empty());
        s.incr(5);
        s.incr(2);
        s.incr(5);
        assert_eq!(s.len(), 2);
        assert_eq!(s.members(), &[2, 5]);
        s.decr(5);
        assert_eq!(s.members(), &[2, 5]);
        s.decr(5);
        assert_eq!(s.members(), &[2]);
        let f = s.freeze(9);
        assert_eq!(f.state, 9);
        assert_eq!(f.get_array(), &[2]);
        assert_eq!(f.size(), 1);
        assert_eq!(f, FrozenIntSet::singleton(2, 0));
        assert_eq!(f.long_hash_code(), s.long_hash_code());
        s.reset();
        assert!(s.members().is_empty());
        // BitMixer.mix32(1) from hppc.
        assert_eq!(bit_mix(0), 0);
        assert_eq!(bit_mix(1), 0x514E_28B7);
    }

    #[test]
    fn swap_remove_keeps_positions_and_hash() {
        let mut s = StateSet::new(0);
        for st in [7, 3, 9, 3, 0, 12] {
            s.incr(st);
        }
        assert_eq!(s.members(), &[0, 3, 7, 9, 12]);
        assert_eq!(s.long_hash_code(), set_hash(&[0, 3, 7, 9, 12]));
        s.decr(7);
        s.decr(3);
        assert_eq!(s.members(), &[0, 3, 9, 12]);
        s.decr(0);
        s.decr(3);
        assert_eq!(s.members(), &[9, 12]);
        assert_eq!(s.long_hash_code(), set_hash(&[9, 12]));
        s.incr(1);
        s.decr(12);
        assert_eq!(s.members(), &[1, 9]);
        s.reset();
        assert!(s.is_empty());
        s.incr(9);
        assert_eq!(s.members(), &[9]);
        assert_eq!(
            s.long_hash_code(),
            FrozenIntSet::singleton(9, 0).long_hash_code()
        );
    }

    #[test]
    #[should_panic(expected = "decr of absent state")]
    fn decr_absent_panics() {
        let mut s = StateSet::new(1);
        s.incr(1);
        s.decr(1);
        s.decr(1);
    }

    #[test]
    fn interner_assigns_dense_ids_and_grows() {
        let mut it = SetInterner::new();
        let mut keys = Vec::new();
        for i in 0..200 {
            let k: Vec<i32> = (i..i + (i % 5)).collect();
            keys.push(k);
        }
        let mut next = 0;
        for k in &keys {
            let (id, new) = it.intern(k, set_hash(k));
            if new {
                assert_eq!(id, next);
                next += 1;
            }
            assert_eq!(it.get(id), &k[..]);
        }
        assert_eq!(it.len(), next);
        for k in &keys {
            let (id, new) = it.intern(k, set_hash(k));
            assert!(!new);
            assert_eq!(it.get(id), &k[..]);
        }
    }
}
