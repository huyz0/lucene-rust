//! `NameIntCacheLRU` and `NameHashIntCacheLRU`: an LRU map from a category
//! (or its 64-bit hash) to its ordinal.
//!
//! Java's `NameHashIntCacheLRU` subclasses `NameIntCacheLRU` to override the
//! key; here the key is a type parameter ([`CacheKey`]), so both are one
//! generic type. A `maxCacheSize` of `i32::MAX` is Java's plain `HashMap` (no
//! LRU): its `get` does not reorder the entries, as Java's does not (nothing
//! is ever evicted from it, so the order is unobservable either way).

use std::hash::Hash;

use super::super::lru_hash_map::LinkedLru;
use super::super::FacetLabel;

/// How a [`NameIntCacheLru`] keys a category (`key(FacetLabel)`).
pub trait CacheKey: std::fmt::Debug + Send {
    type Key: Hash + Eq + Clone + std::fmt::Debug + Send;
    fn key(name: &FacetLabel) -> Self::Key;

    /// `cache.get(key(name))`; `access` is whether the map is access
    /// ordered (a `LinkedHashMap`, not a plain `HashMap`).
    fn lookup(
        cache: &mut LinkedLru<Self::Key, i32>,
        name: &FacetLabel,
        access: bool,
    ) -> Option<i32> {
        let key = Self::key(name);
        if access {
            cache.get(&key).copied()
        } else {
            cache.peek(&key).copied()
        }
    }
}

/// `NameIntCacheLRU.key`: the label itself.
#[derive(Debug, Clone, Copy)]
pub struct ByLabel;

impl CacheKey for ByLabel {
    type Key = FacetLabel;
    fn key(name: &FacetLabel) -> FacetLabel {
        name.clone()
    }

    /// Looks the label up as it is, without the clone `key` makes.
    fn lookup(
        cache: &mut LinkedLru<FacetLabel, i32>,
        name: &FacetLabel,
        access: bool,
    ) -> Option<i32> {
        if access {
            cache.get(name).copied()
        } else {
            cache.peek(name).copied()
        }
    }
}

/// `NameHashIntCacheLRU.key`: the label's `longHashCode()`. Two labels with
/// the same 64-bit hash share an entry, as in Java.
#[derive(Debug, Clone, Copy)]
pub struct ByLongHash;

impl CacheKey for ByLongHash {
    type Key = i64;
    fn key(name: &FacetLabel) -> i64 {
        name.long_hash_code()
    }
}

/// `NameIntCacheLRU`, generic over its key.
#[derive(Debug)]
pub struct NameIntCache<K: CacheKey> {
    cache: LinkedLru<K::Key, i32>,
    max_cache_size: i32,
    /// `nMisses` (for debugging).
    pub n_misses: i64,
    /// `nHits` (for debugging).
    pub n_hits: i64,
}

/// `NameIntCacheLRU`: keyed by the label.
pub type NameIntCacheLru = NameIntCache<ByLabel>;
/// `NameHashIntCacheLRU`: keyed by the label's `longHashCode()`.
pub type NameHashIntCacheLru = NameIntCache<ByLongHash>;

impl<K: CacheKey> NameIntCache<K> {
    /// `new NameIntCacheLRU(maxCacheSize)`.
    pub fn new(max_cache_size: i32) -> Self {
        NameIntCache {
            cache: LinkedLru::new(),
            max_cache_size,
            n_misses: 0,
            n_hits: 0,
        }
    }

    /// `getMaxSize()`.
    pub fn max_size(&self) -> i32 {
        self.max_cache_size
    }

    /// `getSize()`.
    pub fn size(&self) -> usize {
        self.cache.len()
    }

    /// `get(name)`.
    pub fn get(&mut self, name: &FacetLabel) -> Option<i32> {
        let res = K::lookup(&mut self.cache, name, self.max_cache_size < i32::MAX);
        if res.is_none() {
            self.n_misses = self.n_misses.saturating_add(1);
        } else {
            self.n_hits = self.n_hits.saturating_add(1);
        }
        res
    }

    /// `put(name, val)`: whether the cache is now over its limit.
    pub fn put(&mut self, name: &FacetLabel, val: i32) -> bool {
        self.cache.insert(K::key(name), val);
        self.is_cache_full()
    }

    /// `put(name, prefixLen, val)`: keyed by `name.subpath(prefixLen)`.
    pub fn put_prefix(&mut self, name: &FacetLabel, prefix_len: usize, val: i32) -> bool {
        self.cache.insert(K::key(&name.subpath(prefix_len)), val);
        self.is_cache_full()
    }

    fn is_cache_full(&self) -> bool {
        i64::try_from(self.cache.len()).unwrap_or(i64::MAX) > i64::from(self.max_cache_size)
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.cache.clear();
    }

    /// `stats()`.
    pub fn stats(&self) -> String {
        format!("#miss={} #hit={}", self.n_misses, self.n_hits)
    }

    /// `makeRoomLRU()`: when over the limit, drops the least recently used
    /// entries down to two thirds of it; whether anything was dropped.
    pub fn make_room_lru(&mut self) -> bool {
        if !self.is_cache_full() {
            return false;
        }
        let keep = 2i64.saturating_mul(i64::from(self.max_cache_size)) / 3;
        let n = i64::try_from(self.cache.len())
            .unwrap_or(i64::MAX)
            .saturating_sub(keep);
        if n <= 0 {
            return false;
        }
        let mut i = 0;
        while i < n && self.cache.remove_eldest().is_some() {
            i = i.saturating_add(1);
        }
        true
    }
}

#[cfg(test)]
mod tests;
