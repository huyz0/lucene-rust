//! Bucket-ordinal tables for the keyed bucket aggregations: the date grid, the key cache and the key-to-ordinal map.

use super::*;

/// The bucket of each date of a rounding whose buckets all have one width
/// (a fixed interval, or a second, minute, hour, day or week in a fixed
/// zone), found without rounding it: the date's position on the grid is one
/// multiplication by the width's reciprocal, and a direct-mapped table of
/// recent positions gives the bucket. Dates in any order cost the same,
/// where [`DateRounding::bucket`] and the key map cost a division and a
/// probe for each date outside the last one's bucket.
pub(super) struct DateGrid {
    /// A bucket start well before the first date seen: positions count from here.
    pub(super) anchor: i64,
    pub(super) width: i64,
    /// `ceil(2^128 / width)`: `d / width` is the top 64 bits of the 192-bit
    /// product `d * recip`, for every `d`.
    pub(super) recip: u128,
    /// The position held by each slot (`u64::MAX`: none) and its bucket.
    pub(super) tags: Vec<u64>,
    pub(super) ords: Vec<u32>,
}

/// Grid positions below the first date's, so earlier dates still have one.
pub(super) const GRID_HEADROOM: i64 = 1 << 20;
pub(super) const GRID_SLOTS: usize = 4096;

impl DateGrid {
    /// The grid of `rounding`, through the bucket of `v`; `None` for a
    /// rounding whose buckets differ in width.
    pub(super) fn new(rounding: &DateRounding, v: i64) -> Option<Box<Self>> {
        match rounding.kind {
            RoundingKind::Interval(_)
            | RoundingKind::Unit(
                DateUnit::Week
                | DateUnit::Day
                | DateUnit::Hour
                | DateUnit::Minute
                | DateUnit::Second,
            ) => {}
            RoundingKind::Unit(_) => return None,
        }
        let (from, to) = rounding.bucket(v)?;
        let width = to.checked_sub(from)?;
        if width < 2 {
            return None;
        }
        let anchor = from.checked_sub(width.checked_mul(GRID_HEADROOM)?)?;
        let recip = u128::MAX / u128::from(width as u64) + 1;
        Some(Box::new(DateGrid {
            anchor,
            width,
            recip,
            tags: vec![u64::MAX; GRID_SLOTS],
            ords: vec![0; GRID_SLOTS],
        }))
    }

    /// The position of `v`, `None` for a date before the grid's anchor.
    #[inline(always)]
    pub(super) fn position(&self, v: i64) -> Option<u64> {
        if v < self.anchor {
            return None;
        }
        let d = u128::from(v.wrapping_sub(self.anchor) as u64);
        let (hi, lo) = ((self.recip >> 64), self.recip & u128::from(u64::MAX));
        Some(((d * hi + ((d * lo) >> 64)) >> 64) as u64)
    }

    /// The first value of position `p`'s bucket.
    #[inline(always)]
    pub(super) fn start(&self, p: u64) -> i64 {
        self.anchor
            .wrapping_add((p as i64).wrapping_mul(self.width))
    }

    #[inline(always)]
    pub(super) fn find(&self, p: u64) -> Option<u32> {
        let at = p as usize & (GRID_SLOTS - 1);
        (self.tags.get(at) == Some(&p)).then(|| self.ords.get(at).copied().unwrap_or(0))
    }

    #[inline(always)]
    pub(super) fn put(&mut self, p: u64, ord: u32) {
        let at = p as usize & (GRID_SLOTS - 1);
        if let (Some(t), Some(o)) = (self.tags.get_mut(at), self.ords.get_mut(at)) {
            *t = p;
            *o = ord;
        }
    }
}

/// A direct-mapped cache in front of [`keyed_ord`]'s map: the bucket of
/// each of the last `(owning, key)`s placed, one per slot, allocated at
/// first use. A bucket aggregation over few keys (a histogram's) then finds
/// its bucket without hashing into the map; one over many keys, which
/// misses more often than not, turns the cache off.
#[derive(Default)]
pub(super) struct KeyCache {
    pub(super) slots: Vec<(u32, i64, u32)>,
    pub(super) lookups: u32,
    pub(super) misses: u32,
    pub(super) off: bool,
    /// Owner 0's bucket of each small key, by its index below
    /// [`SMALL_KEYS`] (`u32::MAX` until placed): [`small_key_ord`]'s table,
    /// allocated at its first use (16 KB, as `slots`).
    pub(super) small: Vec<u32>,
}

/// Small keys [`small_key_ord`] places without hashing: a code, a count,
/// a status, a few thousand histogram buckets.
pub(super) const SMALL_KEYS: usize = 4096;
/// The bits of `SMALL_KEYS as f64`.
#[allow(clippy::cast_precision_loss)]
pub(super) const SMALL_KEYS_F64_BITS: u64 = (SMALL_KEYS as f64).to_bits();

/// The bucket of `key` under `owning`: a key of owner 0 whose `index` is
/// below [`SMALL_KEYS`] is found by that index in a table, with neither a
/// hash nor a compare; any other through [`keyed_ord_cached`] when `CACHED`,
/// else [`keyed_ord`]. `index` is a function of the key, the same for the
/// same key and different for different ones.
#[inline(always)]
pub(super) fn small_key_ord<const CACHED: bool>(
    cache: &mut KeyCache,
    ords: &mut KeyOrds,
    keys: &mut Vec<(u32, i64)>,
    owning: u32,
    index: u64,
    key: i64,
) -> u32 {
    if owning == 0 && index < SMALL_KEYS as u64 {
        if let Some(&s) = cache.small.get(index as usize) {
            if s != u32::MAX {
                return s;
            }
        }
    }
    small_key_miss::<CACHED>(cache, ords, keys, owning, index, key)
}

/// [`small_key_ord`] for a key not in its table yet: placed in the map, and
/// in the table when small (allocated at its first use).
#[cold]
#[inline(never)]
pub(super) fn small_key_miss<const CACHED: bool>(
    cache: &mut KeyCache,
    ords: &mut KeyOrds,
    keys: &mut Vec<(u32, i64)>,
    owning: u32,
    index: u64,
    key: i64,
) -> u32 {
    if owning == 0 && index < SMALL_KEYS as u64 {
        if cache.small.is_empty() {
            cache.small = vec![u32::MAX; SMALL_KEYS];
        }
        if let Some(s) = cache.small.get_mut(index as usize) {
            *s = keyed_ord(ords, keys, 0, key);
            return *s;
        }
    }
    if CACHED {
        keyed_ord_cached(cache, ords, keys, owning, key)
    } else {
        keyed_ord(ords, keys, owning, key)
    }
}

pub(super) const KEY_CACHE_SLOTS: usize = 1024;
/// Lookups before [`KeyCache`] judges its hit rate.
pub(super) const KEY_CACHE_TRIAL: u32 = 8192;

/// [`keyed_ord`] through `cache`: the same bucket, new or not.
#[inline]
pub(super) fn keyed_ord_cached(
    cache: &mut KeyCache,
    ords: &mut KeyOrds,
    keys: &mut Vec<(u32, i64)>,
    owning: u32,
    key: i64,
) -> u32 {
    if cache.off {
        return keyed_ord(ords, keys, owning, key);
    }
    if cache.slots.is_empty() {
        cache.slots = vec![(u32::MAX, 0, u32::MAX); KEY_CACHE_SLOTS];
    }
    let mixed = (key as u64 ^ (u64::from(owning) << 32)).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    let at = (mixed >> 54) as usize & (KEY_CACHE_SLOTS - 1);
    let Some(slot) = cache.slots.get_mut(at) else {
        return keyed_ord(ords, keys, owning, key);
    };
    cache.lookups += 1;
    let ord = if slot.0 == owning && slot.1 == key && slot.2 != u32::MAX {
        slot.2
    } else {
        let ord = keyed_ord(ords, keys, owning, key);
        *slot = (owning, key, ord);
        cache.misses += 1;
        ord
    };
    if cache.lookups >= KEY_CACHE_TRIAL {
        // More than one miss in eight: the map alone is cheaper.
        cache.off = cache.misses.saturating_mul(8) > cache.lookups;
        cache.lookups = 0;
        cache.misses = 0;
    }
    ord
}

/// `LongKeyedBucketOrds.add`: the bucket of `(owning, key)`, new or not.
#[inline]
pub(super) fn keyed_ord(
    ords: &mut KeyOrds,
    keys: &mut Vec<(u32, i64)>,
    owning: u32,
    key: i64,
) -> u32 {
    ords.ord(keys, owning, key)
}

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
    pub(super) key: i64,
    pub(super) owning: u32,
    pub(super) ord: u32,
}

impl KeyOrds {
    const EMPTY: u32 = u32::MAX;
    const MIN_SLOTS: usize = 256;
    const FREE: KeySlot = KeySlot {
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
    pub(super) fn grow(&mut self, keys: &[(u32, i64)]) {
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
