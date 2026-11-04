//! A fast, non-cryptographic hasher for in-memory tables keyed by data the
//! process itself built (FST suffix tables, ordinal maps): the `FxHasher`
//! rustc uses -- rotate, xor, multiply per word. Not for keys an attacker
//! chooses (no HashDoS resistance); std's `SipHash` stays the right default
//! for those.

use std::collections::{HashMap, HashSet};
use std::hash::{BuildHasherDefault, Hasher};

const SEED: u64 = 0x51_7c_c1_b7_27_22_0a_95;

/// rustc's `FxHasher`.
#[derive(Debug, Default, Clone, Copy)]
pub struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = (self.hash.rotate_left(5) ^ word).wrapping_mul(SEED);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            let mut w = [0u8; 8];
            w.copy_from_slice(c);
            self.add(u64::from_le_bytes(w));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut w = [0u8; 8];
            w[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(w) ^ ((rest.len() as u64) << 56));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u16(&mut self, i: u16) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash
    }
}

/// `BuildHasher` for [`FxHasher`].
pub type FxBuildHasher = BuildHasherDefault<FxHasher>;
/// A `HashMap` hashed with [`FxHasher`].
pub type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;
/// A `HashSet` hashed with [`FxHasher`].
pub type FxHashSet<K> = HashSet<K, FxBuildHasher>;

/// [`FxHasher`](super::FxHasher)'s word mixing with a final avalanche, for
/// maps keyed by values whose distinguishing bytes may sit anywhere in a word.
pub mod avalanche {
    use std::collections::HashMap;

    /// The multiply-rotate hash rustc uses (`FxHasher`): the group maps hash a
    /// group value per collected document, where SipHash's DoS resistance buys
    /// nothing .
    #[derive(Debug, Default, Clone, Copy)]
    pub struct FxHasher(u64);

    impl std::hash::Hasher for FxHasher {
        /// The state through a final avalanche (`fmix64`'s first half): a
        /// multiply carries entropy only upward, so the raw state's low bits --
        /// which `hashbrown` takes the bucket from -- depend only on the low
        /// bytes of the last word. Keys sharing those
        /// (`"group-value-000000001234"`: the last word's low bytes are `"00"`)
        /// fell in a handful of buckets, and every lookup walked a long probe
        /// of `memcmp`s (a grouping search with all groups was 17x slower).
        /// Folding the high half down and multiplying again makes every bit of
        /// the result depend on every bit of the state.
        fn finish(&self) -> u64 {
            let mut x = self.0;
            x ^= x >> 33;
            x = x.wrapping_mul(0xff51_afd7_ed55_8ccd);
            x ^ (x >> 33)
        }

        fn write(&mut self, bytes: &[u8]) {
            let mut chunks = bytes.chunks_exact(8);
            for c in &mut chunks {
                let mut w = [0u8; 8];
                w.copy_from_slice(c);
                self.add(u64::from_le_bytes(w));
            }
            for &b in chunks.remainder() {
                self.add(u64::from(b));
            }
        }

        fn write_u8(&mut self, i: u8) {
            self.add(u64::from(i));
        }

        fn write_u32(&mut self, i: u32) {
            self.add(u64::from(i));
        }

        fn write_u64(&mut self, i: u64) {
            self.add(i);
        }

        fn write_usize(&mut self, i: usize) {
            self.add(i as u64);
        }
    }

    impl FxHasher {
        fn add(&mut self, w: u64) {
            self.0 = (self.0.rotate_left(5) ^ w).wrapping_mul(0x51_7c_c1_b7_27_22_0a_95);
        }
    }

    /// A `HashMap` under [`FxHasher`].
    pub type FxHashMap<K, V> = HashMap<K, V, std::hash::BuildHasherDefault<FxHasher>>;

    /// A `HashSet` under [`FxHasher`].
    pub type FxHashSet<K> = std::collections::HashSet<K, std::hash::BuildHasherDefault<FxHasher>>;
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn h<T: Hash + ?Sized>(v: &T) -> u64 {
        FxBuildHasher::default().hash_one(v)
    }

    #[test]
    fn equal_keys_hash_equal_and_tails_differ() {
        assert_eq!(h(&[1i64, 2, 3][..]), h(&vec![1i64, 2, 3]));
        assert_ne!(h(&(1u8, 2u16, 3u32, 4usize)), h(&(1u8, 2u16, 3u32, 5usize)));
        // Byte strings that differ only in a short tail, or only in length.
        assert_ne!(h(b"abcdefghij".as_slice()), h(b"abcdefghik".as_slice()));
        assert_ne!(h(&b"abcdefgh"[..7]), h(&b"abcdefg\0"[..8]));
        let mut m: FxHashMap<Vec<u8>, i32> = FxHashMap::default();
        m.insert(b"k".to_vec(), 1);
        assert_eq!(m.get(b"k".as_slice()), Some(&1));
        let s: FxHashSet<i32> = (0..100).collect();
        assert!(s.contains(&99));
        assert_eq!(FxHasher::default().finish(), 0);
    }
}
