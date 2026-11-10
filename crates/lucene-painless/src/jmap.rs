//! A map with Java's iteration order: `HashMap`'s bucket order (the order
//! a script's `[:]` map prints and serialises its keys in), a
//! `LinkedHashMap`'s insertion order, or a `TreeMap`'s key order.
//!
//! `HashMap` iterates its table from bucket 0 up, each bucket's entries in
//! the order they were inserted; the table starts at 16 buckets and doubles
//! when the size passes three quarters of it, and a resize keeps each
//! bucket's relative order. So the order is the live entries stably sorted
//! by `spread(hash) & (capacity - 1)`, which is what [`JMap::iter`] computes
//! for [`Order::Hash`]. (A bucket of eight or more colliding keys turns into
//! a tree in Java, which changes its order; that is not modelled.)

use std::cell::RefCell;
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

use crate::value::Value;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Order {
    Hash,
    Insertion,
    Sorted,
}

#[derive(Debug, Clone)]
struct Entry {
    key: Value,
    value: Value,
    hash: i32,
    live: bool,
}

/// A key compared by Java's `equals` and hashed by `hashCode`.
#[derive(Debug, Clone)]
struct Key(Value, i32);

impl PartialEq for Key {
    fn eq(&self, other: &Self) -> bool {
        self.1 == other.1 && self.0.java_equals(&other.0)
    }
}
impl Eq for Key {}
impl Hash for Key {
    fn hash<H: Hasher>(&self, state: &mut H) {
        state.write_i32(self.1);
    }
}

#[derive(Debug, Clone)]
pub struct JMap {
    order: Order,
    entries: Vec<Entry>,
    index: HashMap<Key, usize>,
    /// Java's table length (0 before the first put), for [`Order::Hash`].
    capacity: usize,
    /// The capacity the first put allocates.
    initial: usize,
    live: usize,
    cached: RefCell<Option<Vec<usize>>>,
}

/// `HashMap.hash`: the high half folded into the low.
fn spread(h: i32) -> u32 {
    let h = h as u32;
    h ^ (h >> 16)
}

/// `HashMap.tableSizeFor`.
pub fn table_size_for(cap: usize) -> usize {
    cap.max(1).next_power_of_two().min(1 << 30)
}

impl JMap {
    pub fn new(order: Order) -> Self {
        Self::with_capacity(order, 16)
    }

    /// `new HashMap(initialCapacity)`.
    pub fn with_capacity(order: Order, initial: usize) -> Self {
        JMap {
            order,
            entries: Vec::new(),
            index: HashMap::new(),
            capacity: 0,
            initial: table_size_for(initial),
            live: 0,
            cached: RefCell::new(None),
        }
    }

    /// The table a `new HashMap(map)` of `size` entries starts with.
    pub fn for_copy(order: Order, size: usize) -> Self {
        // `putMapEntries`: `ft = s / loadFactor + 1.0F`, then `tableSizeFor(ft)`.
        let ft = (size as f32 / 0.75 + 1.0) as usize;
        let mut m = Self::with_capacity(order, ft);
        if size == 0 {
            m.initial = 16;
        }
        m
    }

    pub fn order(&self) -> Order {
        self.order
    }

    pub fn len(&self) -> usize {
        self.live
    }

    pub fn is_empty(&self) -> bool {
        self.live == 0
    }

    fn find(&self, key: &Value) -> Option<usize> {
        self.index.get(&Key(key.clone(), key.java_hash())).copied()
    }

    pub fn get(&self, key: &Value) -> Option<&Value> {
        self.find(key).map(|i| &self.entries[i].value)
    }

    pub fn contains_key(&self, key: &Value) -> bool {
        self.find(key).is_some()
    }

    /// `put`: the previous value, if any.
    pub fn insert(&mut self, key: Value, value: Value) -> Option<Value> {
        if let Some(i) = self.find(&key) {
            return Some(std::mem::replace(&mut self.entries[i].value, value));
        }
        let hash = key.java_hash();
        if self.capacity == 0 {
            self.capacity = self.initial;
        }
        self.index
            .insert(Key(key.clone(), hash), self.entries.len());
        self.entries.push(Entry {
            key,
            value,
            hash,
            live: true,
        });
        self.live += 1;
        // `++size > threshold` with `threshold = capacity * 0.75`.
        if self.live > self.capacity / 4 * 3 && self.capacity < (1 << 30) {
            self.capacity *= 2;
        }
        *self.cached.borrow_mut() = None;
        if self.entries.len() > 64 && self.entries.len() > 2 * self.live {
            self.compact();
        }
        None
    }

    pub fn remove(&mut self, key: &Value) -> Option<Value> {
        let i = self.index.remove(&Key(key.clone(), key.java_hash()))?;
        let e = &mut self.entries[i];
        e.live = false;
        self.live -= 1;
        *self.cached.borrow_mut() = None;
        Some(std::mem::take(&mut e.value))
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.index.clear();
        self.live = 0;
        *self.cached.borrow_mut() = None;
    }

    fn compact(&mut self) {
        self.entries.retain(|e| e.live);
        self.index.clear();
        for (i, e) in self.entries.iter().enumerate() {
            self.index.insert(Key(e.key.clone(), e.hash), i);
        }
    }

    fn order_indices(&self) -> Vec<usize> {
        if let Some(v) = self.cached.borrow().as_ref() {
            return v.clone();
        }
        let mut idx: Vec<usize> = (0..self.entries.len())
            .filter(|&i| self.entries[i].live)
            .collect();
        match self.order {
            Order::Insertion => {}
            Order::Hash => {
                let mask = (self.capacity.max(1) - 1) as u32;
                idx.sort_by_key(|&i| spread(self.entries[i].hash) & mask);
            }
            Order::Sorted => {
                idx.sort_by(|&a, &b| {
                    crate::compare::natural(&self.entries[a].key, &self.entries[b].key)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
        }
        *self.cached.borrow_mut() = Some(idx.clone());
        idx
    }

    /// The entries in the map's iteration order.
    pub fn iter(&self) -> impl Iterator<Item = (&Value, &Value)> {
        self.order_indices()
            .into_iter()
            .map(move |i| (&self.entries[i].key, &self.entries[i].value))
    }

    pub fn keys(&self) -> Vec<Value> {
        self.iter().map(|(k, _)| k.clone()).collect()
    }

    pub fn values(&self) -> Vec<Value> {
        self.iter().map(|(_, v)| v.clone()).collect()
    }

    /// A value to change in place.
    pub fn get_mut(&mut self, key: &Value) -> Option<&mut Value> {
        let i = self.find(key)?;
        Some(&mut self.entries[i].value)
    }
}
