//! `org.apache.lucene.facet.taxonomy.LRUHashMap`, and the access-ordered
//! `LinkedHashMap` it (and the writer caches' `NameIntCacheLRU`) are built on.
//!
//! [`LinkedLru`] is `new LinkedHashMap<>(16, 0.75f, true)`'s behaviour: a hash
//! map whose iteration order is least recently accessed first, where a `get`
//! or a `put` of an existing key moves the entry to the end. It keeps its
//! entries in a slab with prev/next links, so every operation is O(1). The
//! index hashes with [`FxHasher`]: the taxonomy caches hash a label per
//! `addCategory`/`getOrdinal`, where SipHash's DoS resistance buys nothing
//! (stage 3: SipHash was most of a cached `addCategory`).

use std::collections::HashMap;
use std::hash::{BuildHasherDefault, Hash};

use crate::grouping::FxHasher;

const NIL: usize = usize::MAX;

#[derive(Debug, Clone)]
struct Slot<K, V> {
    entry: Option<(K, V)>,
    prev: usize,
    next: usize,
}

/// An access-ordered `LinkedHashMap`.
#[derive(Debug, Clone)]
pub struct LinkedLru<K, V> {
    index: HashMap<K, usize, BuildHasherDefault<FxHasher>>,
    slots: Vec<Slot<K, V>>,
    free: Vec<usize>,
    /// The least recently used entry.
    head: usize,
    /// The most recently used entry.
    tail: usize,
}

impl<K: Hash + Eq + Clone, V> Default for LinkedLru<K, V> {
    fn default() -> Self {
        Self::new()
    }
}

impl<K: Hash + Eq + Clone, V> LinkedLru<K, V> {
    pub fn new() -> Self {
        LinkedLru {
            index: HashMap::default(),
            slots: Vec::new(),
            free: Vec::new(),
            head: NIL,
            tail: NIL,
        }
    }

    /// `size()`.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    fn unlink(&mut self, at: usize) {
        let (prev, next) = (self.slots[at].prev, self.slots[at].next);
        if prev == NIL {
            self.head = next;
        } else {
            self.slots[prev].next = next;
        }
        if next == NIL {
            self.tail = prev;
        } else {
            self.slots[next].prev = prev;
        }
    }

    fn link_last(&mut self, at: usize) {
        self.slots[at].prev = self.tail;
        self.slots[at].next = NIL;
        if self.tail == NIL {
            self.head = at;
        } else {
            let tail = self.tail;
            self.slots[tail].next = at;
        }
        self.tail = at;
    }

    fn touch(&mut self, at: usize) {
        if self.tail != at {
            self.unlink(at);
            self.link_last(at);
        }
    }

    /// `get(key)`: the value, the entry becoming the most recently used.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        let at = *self.index.get(key)?;
        self.touch(at);
        self.slots[at].entry.as_ref().map(|(_, v)| v)
    }

    /// The value without an access: `HashMap.get`, for a map that is not
    /// access ordered.
    pub fn peek(&self, key: &K) -> Option<&V> {
        let at = *self.index.get(key)?;
        self.slots[at].entry.as_ref().map(|(_, v)| v)
    }

    /// `containsKey(key)`: no access.
    pub fn contains_key(&self, key: &K) -> bool {
        self.index.contains_key(key)
    }

    /// `put(key, value)`: the previous value; the entry becomes the most
    /// recently used either way.
    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        if let Some(&at) = self.index.get(&key) {
            self.touch(at);
            let slot = self.slots[at].entry.as_mut().expect("linked slot");
            return Some(std::mem::replace(&mut slot.1, value));
        }
        let slot = Slot {
            entry: Some((key.clone(), value)),
            prev: NIL,
            next: NIL,
        };
        let at = match self.free.pop() {
            Some(at) => {
                self.slots[at] = slot;
                at
            }
            None => {
                self.slots.push(slot);
                self.slots.len().saturating_sub(1)
            }
        };
        self.link_last(at);
        self.index.insert(key, at);
        None
    }

    /// Removes and returns the least recently used entry.
    pub fn remove_eldest(&mut self) -> Option<(K, V)> {
        if self.head == NIL {
            return None;
        }
        let at = self.head;
        self.unlink(at);
        let (k, v) = self.slots[at].entry.take().expect("linked slot");
        self.index.remove(&k);
        self.free.push(at);
        Some((k, v))
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.index.clear();
        self.slots.clear();
        self.free.clear();
        self.head = NIL;
        self.tail = NIL;
    }

    /// The keys, least recently used first (`keySet()`'s order).
    pub fn keys(&self) -> Vec<&K> {
        let mut out = Vec::with_capacity(self.len());
        let mut at = self.head;
        while at != NIL {
            if let Some((k, _)) = &self.slots[at].entry {
                out.push(k);
            }
            at = self.slots[at].next;
        }
        out
    }
}

/// `LRUHashMap<K, V>`: a [`LinkedLru`] that drops its least recently used
/// entry when a `put` takes it over `max_size` (`removeEldestEntry`: one
/// entry per `put`, so lowering the limit with [`LruHashMap::set_max_size`]
/// shrinks the map only as later entries are added).
#[derive(Debug, Clone)]
pub struct LruHashMap<K, V> {
    map: LinkedLru<K, V>,
    max_size: usize,
}

impl<K: Hash + Eq + Clone, V> LruHashMap<K, V> {
    /// `new LRUHashMap(maxSize)`.
    pub fn new(max_size: usize) -> Self {
        LruHashMap {
            map: LinkedLru::new(),
            max_size,
        }
    }

    /// `getMaxSize()`.
    pub fn max_size(&self) -> usize {
        self.max_size
    }

    /// `setMaxSize(maxSize)`.
    pub fn set_max_size(&mut self, max_size: usize) {
        self.max_size = max_size;
    }

    /// `get(key)`.
    pub fn get(&mut self, key: &K) -> Option<&V> {
        self.map.get(key)
    }

    /// `put(key, value)`.
    pub fn put(&mut self, key: K, value: V) -> Option<V> {
        let old = self.map.insert(key, value);
        // `LinkedHashMap.afterNodeInsertion`: only a new key can evict
        if old.is_none() && self.map.len() > self.max_size {
            self.map.remove_eldest();
        }
        old
    }

    /// `size()`.
    pub fn len(&self) -> usize {
        self.map.len()
    }

    /// `isEmpty()`.
    pub fn is_empty(&self) -> bool {
        self.map.is_empty()
    }

    /// `containsKey(key)`.
    pub fn contains_key(&self, key: &K) -> bool {
        self.map.contains_key(key)
    }

    /// `clear()`.
    pub fn clear(&mut self) {
        self.map.clear();
    }

    /// The keys, least recently used first.
    pub fn keys(&self) -> Vec<&K> {
        self.map.keys()
    }
}

#[cfg(test)]
mod tests;
