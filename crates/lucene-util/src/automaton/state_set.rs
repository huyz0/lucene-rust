//! `IntSet`, `FrozenIntSet` and `StateSet`: the NFA-state-set keys of subset
//! construction.
//!
//! [`StateSet`] is a multiset of NFA states (a state stays in while any
//! transition into it covers the current interval); [`StateSet::freeze`]
//! snapshots its distinct members, sorted, as the [`FrozenIntSet`] key of a
//! DFA state. The hash is Lucene's (`size + sum(BitMixer.mix(state))`), kept
//! so the two types hash alike, as Java requires of equal `IntSet`s.

use std::collections::HashMap;
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

/// `StateSet`: a counted multiset of states.
#[derive(Clone, Debug, Default)]
pub struct StateSet {
    inner: HashMap<i32, i32>,
    array_cache: Vec<i32>,
    array_updated: bool,
}

impl StateSet {
    /// `new StateSet(capacity)`.
    pub fn new(capacity: usize) -> Self {
        StateSet {
            inner: HashMap::with_capacity(capacity),
            array_cache: Vec::new(),
            array_updated: true,
        }
    }

    /// `incr(state)`: add one occurrence.
    pub fn incr(&mut self, state: i32) {
        let c = self.inner.entry(state).or_insert(0);
        *c += 1;
        if *c == 1 {
            self.array_updated = false;
        }
    }

    /// `decr(state)`: remove one occurrence.
    ///
    /// # Panics
    /// If `state` is not in the set (Java asserts this).
    pub fn decr(&mut self, state: i32) {
        let c = self.inner.get_mut(&state).expect("decr of absent state");
        *c -= 1;
        if *c == 0 {
            self.inner.remove(&state);
            self.array_updated = false;
        }
    }

    /// `reset()`: empty the set.
    pub fn reset(&mut self) {
        self.inner.clear();
        self.array_updated = false;
    }

    /// `freeze(state)`: the current distinct members as a key for `state`.
    pub fn freeze(&mut self, state: i32) -> FrozenIntSet {
        let hash = self.long_hash_code();
        FrozenIntSet::new(self.sorted().to_vec(), hash, state)
    }

    fn sorted(&mut self) -> &[i32] {
        if !self.array_updated {
            self.array_cache = self.inner.keys().copied().collect();
            self.array_cache.sort_unstable();
            self.array_updated = true;
        }
        &self.array_cache
    }

    /// The distinct members, ascending (Java's `getArray`, which needs `&mut`
    /// here to refresh its cache).
    pub fn members(&mut self) -> &[i32] {
        self.sorted()
    }

    /// `size()`: distinct members.
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// `size() == 0`.
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// `longHashCode()`.
    pub fn long_hash_code(&self) -> i64 {
        self.inner.keys().fold(self.inner.len() as i64, |h, &k| {
            h.wrapping_add(i64::from(bit_mix(k)))
        })
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
}
