//! A tiny Fx-style hasher (the multiply-rotate hash rustc uses) for
//! crate-internal maps keyed by small integers or short integer vectors,
//! where `std`'s SipHash costs more than the lookup it serves. Not
//! DoS-resistant: use it only for keys this crate computes itself (automaton
//! states, labels), never for input-controlled strings.

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hasher};

const K: u64 = 0xf135_7aea_2e62_a9c5;

/// The hasher: `h = (h + word) * K` per word, rotated on finish so the low
/// bits a table indexes by depend on every input bit.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct FxHasher {
    hash: u64,
}

impl FxHasher {
    #[inline]
    fn add(&mut self, word: u64) {
        self.hash = self.hash.wrapping_add(word).wrapping_mul(K);
    }
}

impl Hasher for FxHasher {
    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let mut chunks = bytes.chunks_exact(8);
        for c in &mut chunks {
            self.add(u64::from_le_bytes(c.try_into().expect("8 bytes")));
        }
        let rest = chunks.remainder();
        if !rest.is_empty() {
            let mut buf = [0u8; 8];
            buf[..rest.len()].copy_from_slice(rest);
            self.add(u64::from_le_bytes(buf) ^ ((rest.len() as u64) << 56));
        }
    }
    #[inline]
    fn write_u8(&mut self, i: u8) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_u32(&mut self, i: u32) {
        self.add(u64::from(i));
    }
    #[inline]
    fn write_i32(&mut self, i: i32) {
        self.add(u64::from(i as u32));
    }
    #[inline]
    fn write_u64(&mut self, i: u64) {
        self.add(i);
    }
    #[inline]
    fn write_i64(&mut self, i: i64) {
        self.add(i as u64);
    }
    #[inline]
    fn write_usize(&mut self, i: usize) {
        self.add(i as u64);
    }
    #[inline]
    fn finish(&self) -> u64 {
        self.hash.rotate_left(26)
    }
}

/// `BuildHasher` for [`FxHasher`].
pub(crate) type FxBuildHasher = BuildHasherDefault<FxHasher>;

/// A `HashMap` hashed with [`FxHasher`].
pub(crate) type FxHashMap<K, V> = HashMap<K, V, FxBuildHasher>;

#[cfg(test)]
mod tests {
    use super::*;
    use std::hash::{BuildHasher, Hash};

    fn h<T: Hash>(v: T) -> u64 {
        FxBuildHasher::default().hash_one(v)
    }

    #[test]
    fn distinguishes_and_maps_work() {
        assert_ne!(h(1i32), h(2i32));
        assert_ne!(h(vec![1i32, 2]), h(vec![2i32, 1]));
        assert_ne!(h(&b"abcdefgh1"[..]), h(&b"abcdefgh2"[..]));
        assert_ne!(h(&b"ab"[..]), h(&b"ab\0"[..]));
        let mut fx = FxHasher::default();
        fx.write(b"abc");
        fx.write_u8(1);
        fx.write_u32(2);
        fx.write_i64(-3);
        fx.write_usize(4);
        assert_ne!(fx.finish(), 0);
        let mut m: FxHashMap<Vec<i32>, i32> = FxHashMap::default();
        for i in 0..1000 {
            m.insert(vec![i, i + 1], i);
        }
        assert_eq!(m.len(), 1000);
        assert_eq!(m.get(&vec![500, 501]), Some(&500));
    }
}
