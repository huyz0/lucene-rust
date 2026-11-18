//! `LongKeyedBucketOrds`' hash table, split out of the parent module.

/// `LongKeyedBucketOrds`' hash table: open addressing over `(owning, key)`
/// and its bucket ordinal (the caller's `keys` lists the keys in ordinal
/// order). A slot holds its key, so a probe reads one slot and nothing else.
#[derive(Default)]
pub(super) struct KeyOrds {
    /// A power of two long (at least [`KeyOrds::MIN_SLOTS`]), at most five
    /// eighths full: short probe runs, whose branches predict well. A free
    /// slot's ordinal is [`KeyOrds::EMPTY`].
    pub(super) slots: Vec<KeySlot>,
    /// Owner 0's bucket of each key in `0..dense.len()` ([`KeyOrds::EMPTY`]
    /// until it has one), when [`KeyOrds::densify`] chose it: a top-level
    /// keyword `terms` whose global ordinals the matches will mostly reach
    /// keeps 4 bytes an ordinal instead of a slot per bucket. Every other
    /// key lives in `slots`.
    pub(super) dense: Vec<u32>,
    /// The keys in `slots`.
    pub(super) hashed: usize,
}

#[derive(Clone, Copy)]
pub(super) struct KeySlot {
    key: i64,
    owning: u32,
    ord: u32,
}

impl KeyOrds {
    pub(super) const EMPTY: u32 = u32::MAX;
    pub(super) const MIN_SLOTS: usize = 256;
    pub(super) const FREE: KeySlot = KeySlot {
        key: 0,
        owning: 0,
        ord: Self::EMPTY,
    };

    /// Fibonacci hashing of the key and its owner, the slot from the high
    /// bits (well spread for strided keys such as rounded dates).
    #[inline]
    pub(super) fn slot_of(owning: u32, key: i64, shift: u32) -> usize {
        let h =
            ((key as u64) ^ u64::from(owning).rotate_left(32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
        (h >> shift) as usize
    }

    /// Tables from this many slots (1 MiB) no longer sit in the cache.
    pub(super) const PREFETCH_SLOTS: usize = 1 << 16;
    /// Documents ahead whose slot a lookup requests early.
    pub(super) const PREFETCH_AHEAD: usize = 12;

    /// Starts loading the slot `(owning, key)` hashes to.
    #[inline(always)]
    pub(super) fn prefetch(&self, owning: u32, key: i64) {
        if !self.slots.is_empty() {
            let at = Self::slot_of(owning, key, Self::shift(self.slots.len()));
            lucene_util::simd::prefetch(&self.slots, at);
        }
    }

    /// The shift [`KeyOrds::slot_of`] takes for a table of `len` slots.
    #[inline]
    pub(super) fn shift(len: usize) -> u32 {
        64 - len.trailing_zeros()
    }

    /// Whether owner 0's keys `0..keys` should be looked up in a dense array
    /// rather than the table: a lookup is 2-3x cheaper and holds 4 bytes a
    /// key rather than 16 a slot (at most five eighths full), which pays
    /// once the matches reach about one key in eight
    /// (`perf/memory-bounds/raw/followup`); past that the table, sized by
    /// the buckets made, holds less.
    pub(super) fn dense_pays(keys: usize, matches: u64) -> bool {
        keys > 0 && (keys as u64) <= matches.saturating_mul(8)
    }

    /// Looks up owner 0's keys `0..keys` in a dense array from now on, when
    /// the request's memory allows it; before any bucket is made.
    pub(super) fn densify(&mut self, keys: usize) {
        if self.hashed > 0 || !self.dense.is_empty() {
            return;
        }
        let bytes = keys.saturating_mul(std::mem::size_of::<u32>());
        if lucene_util::memory::reserve(bytes).is_ok() {
            self.dense = vec![Self::EMPTY; keys];
        }
    }

    /// The bucket of `(owning, key)`, appended to `keys` when new.
    #[inline]
    pub(super) fn ord(&mut self, keys: &mut Vec<(u32, i64)>, owning: u32, key: i64) -> u32 {
        if owning == 0 {
            if let Some(d) = usize::try_from(key)
                .ok()
                .and_then(|k| self.dense.get_mut(k))
            {
                if *d == Self::EMPTY {
                    *d = keys.len() as u32;
                    keys.push((0, key));
                }
                return *d;
            }
        }
        // ARITH: at most `u32::MAX - 1` buckets (an ordinal is a `u32`).
        #[allow(clippy::arithmetic_side_effects)]
        if (self.hashed + 1) * 8 > self.slots.len() * 5 {
            self.grow(keys);
        }
        let mask = self.slots.len() - 1;
        let mut i = Self::slot_of(owning, key, Self::shift(self.slots.len()));
        // FBS: `i <= mask`, below the table's power-of-two length, and a free
        // slot is always reached (the table is at most five eighths full).
        while let Some(slot) = self.slots.get_mut(i) {
            if slot.ord == Self::EMPTY {
                let ord = keys.len() as u32;
                *slot = KeySlot { key, owning, ord };
                keys.push((owning, key));
                // ARITH: below `keys.len()`.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    self.hashed += 1;
                }
                return ord;
            }
            if slot.key == key && slot.owning == owning {
                return slot.ord;
            }
            // ARITH: wraps within the table.
            #[allow(clippy::arithmetic_side_effects)]
            {
                i = (i + 1) & mask;
            }
        }
        // Not reached: a table without its slot is placed again, larger.
        self.grow(keys);
        self.ord(keys, owning, key)
    }

    /// Doubles the table, its slots placed again from `keys` (the old table
    /// freed first).
    #[cold]
    fn grow(&mut self, keys: &[(u32, i64)]) {
        let len = self.slots.len().saturating_mul(2).max(Self::MIN_SLOTS);
        self.slots = Vec::new();
        self.slots = vec![Self::FREE; len];
        let mask = len - 1;
        let shift = Self::shift(len);
        let dense = self.dense.len();
        for (ord, &(owning, key)) in keys.iter().enumerate() {
            if owning == 0 && usize::try_from(key).is_ok_and(|k| k < dense) {
                continue;
            }
            let mut i = Self::slot_of(owning, key, shift);
            // FBS: `i <= mask`; the table has room for every key.
            while self.slots.get(i).is_some_and(|s| s.ord != Self::EMPTY) {
                // ARITH: wraps within the table.
                #[allow(clippy::arithmetic_side_effects)]
                {
                    i = (i + 1) & mask;
                }
            }
            if let Some(slot) = self.slots.get_mut(i) {
                *slot = KeySlot {
                    key,
                    owning,
                    ord: ord as u32,
                };
            }
        }
    }
}
