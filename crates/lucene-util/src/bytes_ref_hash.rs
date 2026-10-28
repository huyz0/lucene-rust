//! Port of `org.apache.lucene.util.BytesRefHash` and the
//! `BytesRefBlockPool`/`ByteBlockPool` storage under it: a hash set of byte
//! strings that hands out dense ids `0, 1, 2, ...` in insertion order.
//!
//! Terms are stored in 32 KiB blocks, each prefixed by a 1-byte (`< 128`) or
//! 2-byte big-endian (`| 0x8000`) length, never straddling a block; an id's
//! "byte start" is the global offset of that prefix. Everything observable
//! -- ids, `-(id + 1)` for a duplicate, `find`, `sort`'s order, byte starts,
//! the shrink-on-clear table sizes -- matches Lucene, except the order of
//! `compact`'s ids (see [`BytesRefHash::compact`]); the table
//! is open-addressed with Java's linear probing and `murmurhash3_x86_32`
//! under a per-instance seed (Java's `GOOD_FAST_HASH_SEED`), which decides
//! slots but nothing a caller can see.

use crate::string_helper::murmurhash3_x86_32;

/// `ByteBlockPool.BYTE_BLOCK_SHIFT`.
pub const BYTE_BLOCK_SHIFT: u32 = 15;
/// `ByteBlockPool.BYTE_BLOCK_SIZE`.
pub const BYTE_BLOCK_SIZE: usize = 1 << BYTE_BLOCK_SHIFT;
/// `ByteBlockPool.BYTE_BLOCK_MASK`.
pub const BYTE_BLOCK_MASK: usize = BYTE_BLOCK_SIZE - 1;
/// `BytesRefHash.DEFAULT_CAPACITY`.
pub const DEFAULT_CAPACITY: usize = 16;

/// `BytesRefBlockPool` over its own `ByteBlockPool`.
#[derive(Debug, Clone)]
pub struct BytesRefBlockPool {
    blocks: Vec<Vec<u8>>,
    /// `byteUpto`: write position in the head block.
    byte_upto: usize,
    /// `byteOffset`: the head block's global offset.
    byte_offset: i64,
}

impl Default for BytesRefBlockPool {
    fn default() -> Self {
        Self::new()
    }
}

impl BytesRefBlockPool {
    /// A pool with no blocks yet (`byteUpto = BYTE_BLOCK_SIZE`).
    pub fn new() -> Self {
        BytesRefBlockPool {
            blocks: Vec::new(),
            byte_upto: BYTE_BLOCK_SIZE,
            byte_offset: -(BYTE_BLOCK_SIZE as i64),
        }
    }

    /// `reset(false, false)`: drops every block.
    pub fn reset(&mut self) {
        *self = Self::new();
    }

    fn next_buffer(&mut self) {
        self.blocks.push(vec![0; BYTE_BLOCK_SIZE]);
        self.byte_upto = 0;
        self.byte_offset += BYTE_BLOCK_SIZE as i64;
    }

    /// `addBytesRef`: appends `bytes` with its length prefix, returning its
    /// start offset. `None` when longer than `BYTE_BLOCK_SIZE - 2`
    /// (`MaxBytesLengthExceededException`).
    pub fn add_bytes_ref(&mut self, bytes: &[u8]) -> Option<i32> {
        let length = bytes.len();
        let len2 = 2 + length;
        if len2 + self.byte_upto > BYTE_BLOCK_SIZE {
            if len2 > BYTE_BLOCK_SIZE {
                return None;
            }
            self.next_buffer();
        }
        let upto = self.byte_upto;
        let text_start = (upto as i64 + self.byte_offset) as i32;
        let buffer = self.blocks.last_mut()?;
        if length < 128 {
            buffer[upto] = length as u8;
            buffer[upto + 1..upto + 1 + length].copy_from_slice(bytes);
            self.byte_upto += length + 1;
        } else {
            buffer[upto..upto + 2].copy_from_slice(&((length as u16) | 0x8000).to_be_bytes());
            buffer[upto + 2..upto + 2 + length].copy_from_slice(bytes);
            self.byte_upto += length + 2;
        }
        Some(text_start)
    }

    /// `fillBytesRef(term, start)`: the bytes stored at `start`.
    pub fn get(&self, start: i32) -> &[u8] {
        let block = &self.blocks[(start as usize) >> BYTE_BLOCK_SHIFT];
        let pos = start as usize & BYTE_BLOCK_MASK;
        let (len, off) = if block[pos] & 0x80 == 0 {
            (block[pos] as usize, pos + 1)
        } else {
            (
                (u16::from_be_bytes([block[pos], block[pos + 1]]) & 0x7fff) as usize,
                pos + 2,
            )
        };
        &block[off..off + len]
    }
}

/// Error of [`BytesRefHash::add`].
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum BytesRefHashError {
    /// `MaxBytesLengthExceededException`.
    #[error("bytes can be at most {} in length; got {0}", BYTE_BLOCK_SIZE - 2)]
    MaxBytesLengthExceeded(usize),
    /// Used after `clear` without `reinit` (Java: a `NullPointerException`).
    #[error("bytesStart is null - not initialized")]
    NotInitialized,
    /// `capacity` is not a positive power of two.
    #[error("capacity must be a power of two, got {0}")]
    BadCapacity(usize),
}

/// `org.apache.lucene.util.BytesRefHash` with a `DirectBytesStartArray`.
#[derive(Debug, Clone)]
pub struct BytesRefHash {
    pool: BytesRefBlockPool,
    bytes_start: Option<Vec<i32>>,
    hash_size: usize,
    hash_half_size: usize,
    hash_mask: i32,
    high_mask: i32,
    count: usize,
    last_count: i64,
    ids: Option<Vec<i32>>,
    seed: i32,
}

/// The first eight bytes of `term`, big-endian, zero-padded: comparing two
/// of these orders terms as unsigned bytes do, up to a tie when one is a
/// prefix of the other or both share the first eight bytes.
#[inline]
fn be_prefix(term: &[u8]) -> u64 {
    let mut buf = [0u8; 8];
    let k = term.len().min(8);
    buf[..k].copy_from_slice(&term[..k]);
    u64::from_be_bytes(buf)
}

fn process_seed() -> i32 {
    use std::hash::{BuildHasher, Hasher};
    let mut h = std::collections::hash_map::RandomState::new().build_hasher();
    h.write_u8(0);
    h.finish() as i32
}

impl Default for BytesRefHash {
    fn default() -> Self {
        Self::new()
    }
}

impl BytesRefHash {
    /// `new BytesRefHash()`.
    pub fn new() -> Self {
        Self::with_capacity(DEFAULT_CAPACITY, process_seed()).expect("16 is a power of two")
    }

    /// `new BytesRefHash(pool, capacity, new DirectBytesStartArray(capacity))`
    /// with an explicit hash seed. Capacity 1 is refused: Java accepts it, but
    /// its `hashHalfSize` is then 0, the table never grows, and the second
    /// distinct `add` probes a full table forever.
    pub fn with_capacity(capacity: usize, seed: i32) -> Result<Self, BytesRefHashError> {
        if capacity < 2 || !capacity.is_power_of_two() || capacity > 1 << 30 {
            return Err(BytesRefHashError::BadCapacity(capacity));
        }
        Ok(BytesRefHash {
            pool: BytesRefBlockPool::new(),
            bytes_start: Some(Vec::new()),
            hash_size: capacity,
            hash_half_size: capacity >> 1,
            hash_mask: (capacity - 1) as i32,
            high_mask: !((capacity - 1) as i32),
            count: 0,
            last_count: -1,
            ids: Some(vec![-1; capacity]),
            seed,
        })
    }

    /// `size()`.
    pub fn size(&self) -> usize {
        self.count
    }

    /// The pool the terms live in.
    pub fn pool(&self) -> &BytesRefBlockPool {
        &self.pool
    }

    /// The current table size (`hashSize`).
    pub fn hash_size(&self) -> usize {
        self.hash_size
    }

    /// `get(bytesID, ref)`.
    pub fn get(&self, bytes_id: usize) -> Option<&[u8]> {
        let starts = self.bytes_start.as_ref()?;
        starts.get(bytes_id).map(|&s| self.pool.get(s))
    }

    /// `byteStart(bytesID)`.
    pub fn byte_start(&self, bytes_id: usize) -> Option<i32> {
        self.bytes_start.as_ref()?.get(bytes_id).copied()
    }

    fn do_hash(&self, bytes: &[u8]) -> i32 {
        murmurhash3_x86_32(bytes, self.seed)
    }

    /// `compact()`: the ids, then `-1`s, `hashSize` long.
    ///
    /// Java packs the ids in hash-slot order, which depends on the hash seed
    /// and so is unspecified to every caller; this returns them in id order
    /// (`0..count`). No caller in this port reads the order (every one that
    /// needs an order calls [`Self::sort`]), so the cheaper fill is kept.
    pub fn compact(&mut self) -> Vec<i32> {
        self.last_count = self.count as i64;
        let n = self.count as i32;
        let ids = self.ids.get_or_insert_with(Vec::new);
        ids.clear();
        ids.extend(0..n);
        ids.resize(self.hash_size, -1);
        ids.clone()
    }

    /// `sort()`: the ids ordered by their bytes (unsigned), then `-1`s. Like
    /// Java, this destroys the table (`compact` first); only `clear` makes
    /// the hash usable again.
    ///
    /// Terms are unique, so the order is fully determined whatever the
    /// algorithm (Java's `StringSorter` MSB radix sort gives the same).
    /// Each id is resolved once to its bytes and a big-endian 8-byte prefix:
    /// most comparisons are then a single integer compare, and only a tie on
    /// the prefix compares the term bytes.
    pub fn sort(&mut self) -> Vec<i32> {
        self.last_count = self.count as i64;
        let n = self.count;
        let mut out = Vec::with_capacity(self.hash_size);
        if let Some(starts) = &self.bytes_start {
            let pool = &self.pool;
            let mut keyed: Vec<(u64, &[u8], i32)> = starts[..n]
                .iter()
                .enumerate()
                .map(|(id, &start)| {
                    let term = pool.get(start);
                    (be_prefix(term), term, id as i32)
                })
                .collect();
            keyed.sort_unstable_by(|a, b| a.0.cmp(&b.0).then_with(|| a.1.cmp(b.1)));
            out.extend(keyed.iter().map(|k| k.2));
        } else {
            out.extend(0..n as i32);
        }
        out.resize(self.hash_size, -1);
        if let Some(ids) = &mut self.ids {
            ids.copy_from_slice(&out);
        }
        out
    }

    fn shrink(&mut self, target_size: usize) -> bool {
        let mut new_size = self.hash_size;
        while new_size >= 8 && new_size / 4 > target_size {
            new_size /= 2;
        }
        if new_size != self.hash_size {
            self.hash_size = new_size;
            self.ids = Some(vec![-1; new_size]);
            self.hash_half_size = new_size / 2;
            self.hash_mask = (new_size - 1) as i32;
            self.high_mask = !self.hash_mask;
            true
        } else {
            false
        }
    }

    /// `clear(resetPool)`: empties the hash (shrinking the table to at most
    /// four times the old count); `reinit` must follow before the next `add`,
    /// as in Java.
    pub fn clear(&mut self, reset_pool: bool) {
        self.last_count = self.count as i64;
        self.count = 0;
        if reset_pool {
            self.pool.reset();
        }
        self.bytes_start = None;
        if self.last_count != -1 && self.shrink(self.last_count as usize) {
            return;
        }
        if let Some(ids) = &mut self.ids {
            ids.fill(-1);
        }
    }

    /// `close()`: `clear(true)` and drop the table.
    pub fn close(&mut self) {
        self.clear(true);
        self.ids = None;
    }

    /// `reinit()`. A table dropped by `close` comes back empty (`-1`-filled):
    /// a zero-filled one would read every slot as occupied, and the next
    /// `add` would probe forever.
    pub fn reinit(&mut self) {
        if self.bytes_start.is_none() {
            self.bytes_start = Some(Vec::new());
        }
        if self.ids.is_none() {
            self.ids = Some(vec![-1; self.hash_size]);
        }
    }

    fn find_hash(&self, bytes: &[u8], hashcode: i32) -> Result<usize, BytesRefHashError> {
        let ids = self.ids.as_ref().ok_or(BytesRefHashError::NotInitialized)?;
        let starts = self
            .bytes_start
            .as_ref()
            .ok_or(BytesRefHashError::NotInitialized)?;
        let mut code = hashcode;
        let mut pos = (code & self.hash_mask) as usize;
        let mut e = ids[pos];
        let high_bits = hashcode & self.high_mask;
        while e != -1
            && ((e & self.high_mask) != high_bits
                || self.pool.get(starts[(e & self.hash_mask) as usize]) != bytes)
        {
            code = code.wrapping_add(1);
            pos = (code & self.hash_mask) as usize;
            e = ids[pos];
        }
        Ok(pos)
    }

    /// `add(bytes)`: the new id, or `-(id + 1)` when `bytes` is already in.
    pub fn add(&mut self, bytes: &[u8]) -> Result<i32, BytesRefHashError> {
        let hashcode = self.do_hash(bytes);
        let pos = self.find_hash(bytes, hashcode)?;
        let e = self.ids.as_ref().ok_or(BytesRefHashError::NotInitialized)?[pos];
        if e != -1 {
            return Ok(-((e & self.hash_mask) + 1));
        }
        let start = self
            .pool
            .add_bytes_ref(bytes)
            .ok_or(BytesRefHashError::MaxBytesLengthExceeded(bytes.len()))?;
        let id = self.count as i32;
        self.bytes_start
            .as_mut()
            .ok_or(BytesRefHashError::NotInitialized)?
            .push(start);
        self.count += 1;
        let high = hashcode & self.high_mask;
        if let Some(ids) = &mut self.ids {
            ids[pos] = id | high;
        }
        if self.count == self.hash_half_size {
            self.rehash(2 * self.hash_size, true);
        }
        Ok(id)
    }

    /// `find(bytes)`: the id, or -1.
    pub fn find(&self, bytes: &[u8]) -> Result<i32, BytesRefHashError> {
        let pos = self.find_hash(bytes, self.do_hash(bytes))?;
        let id = self.ids.as_ref().ok_or(BytesRefHashError::NotInitialized)?[pos];
        Ok(if id == -1 { -1 } else { id & self.hash_mask })
    }

    /// `addByPoolOffset(offset)`: registers a term already in the pool by its
    /// byte start (the hash is on the offset itself).
    pub fn add_by_pool_offset(&mut self, offset: i32) -> Result<i32, BytesRefHashError> {
        let ids = self.ids.as_ref().ok_or(BytesRefHashError::NotInitialized)?;
        let starts = self
            .bytes_start
            .as_ref()
            .ok_or(BytesRefHashError::NotInitialized)?;
        let mut code = offset;
        let mut pos = (offset & self.hash_mask) as usize;
        let mut e = ids[pos];
        while e != -1 && starts[e as usize] != offset {
            code = code.wrapping_add(1);
            pos = (code & self.hash_mask) as usize;
            e = ids[pos];
        }
        if e != -1 {
            return Ok(-(e + 1));
        }
        let id = self.count as i32;
        self.bytes_start
            .as_mut()
            .ok_or(BytesRefHashError::NotInitialized)?
            .push(offset);
        self.count += 1;
        if let Some(ids) = &mut self.ids {
            ids[pos] = id;
        }
        if self.count == self.hash_half_size {
            self.rehash(2 * self.hash_size, false);
        }
        Ok(id)
    }

    fn rehash(&mut self, new_size: usize, hash_on_data: bool) {
        let new_mask = (new_size - 1) as i32;
        let new_high_mask = !new_mask;
        let mut ids = vec![-1i32; new_size];
        let starts = self.bytes_start.as_deref().unwrap_or(&[]);
        for (id, &start) in starts.iter().enumerate().take(self.count) {
            let (hashcode, mut code) = if hash_on_data {
                let h = self.do_hash(self.pool.get(start));
                (h, h)
            } else {
                (0, start)
            };
            let mut pos = (code & new_mask) as usize;
            while ids[pos] != -1 {
                code = code.wrapping_add(1);
                pos = (code & new_mask) as usize;
            }
            ids[pos] = id as i32 | (hashcode & new_high_mask);
        }
        self.ids = Some(ids);
        self.hash_mask = new_mask;
        self.high_mask = new_high_mask;
        self.hash_size = new_size;
        self.hash_half_size = new_size / 2;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn add_find_get_sort() {
        let mut h = BytesRefHash::with_capacity(4, 7).unwrap();
        assert_eq!(h.add(b"banana").unwrap(), 0);
        assert_eq!(h.add(b"apple").unwrap(), 1);
        assert_eq!(h.add(b"banana").unwrap(), -1);
        for i in 0..100 {
            h.add(format!("t{i:03}").as_bytes()).unwrap();
        }
        assert_eq!(h.size(), 102);
        assert_eq!(h.find(b"apple").unwrap(), 1);
        assert_eq!(h.find(b"nope").unwrap(), -1);
        assert_eq!(h.get(0).unwrap(), b"banana");
        assert_eq!(h.byte_start(1), Some(7));
        assert_eq!(h.get(1000), None);
        let long = vec![b'x'; 300];
        let id = h.add(&long).unwrap();
        assert_eq!(h.get(id as usize).unwrap(), &long[..]);
        assert_eq!(h.pool().get(h.byte_start(id as usize).unwrap()), &long[..]);
        let sorted = h.sort();
        assert_eq!(sorted.len(), h.hash_size());
        assert_eq!(&sorted[..3], &[1, 0, 2]);
        assert_eq!(sorted[h.size()], -1);
        h.clear(true);
        assert_eq!(h.add(b"x"), Err(BytesRefHashError::NotInitialized));
        h.reinit();
        assert_eq!(h.add(b"x").unwrap(), 0);
        assert_eq!(h.byte_start(0), Some(0));
        assert!(h.hash_size() <= 512);
    }

    #[test]
    fn limits_pool_offsets_and_close() {
        let mut h = BytesRefHash::new();
        assert_eq!(
            h.add(&vec![0u8; BYTE_BLOCK_SIZE - 1]),
            Err(BytesRefHashError::MaxBytesLengthExceeded(
                BYTE_BLOCK_SIZE - 1
            ))
        );
        h.add(&vec![1u8; BYTE_BLOCK_SIZE - 2]).unwrap();
        // The next term does not fit in block 0.
        h.add(b"a").unwrap();
        assert_eq!(h.byte_start(1), Some(BYTE_BLOCK_SIZE as i32));
        let mut p = BytesRefHash::with_capacity(2, 0).unwrap();
        p.reinit();
        assert_eq!(p.add_by_pool_offset(40).unwrap(), 0);
        assert_eq!(p.add_by_pool_offset(40).unwrap(), -1);
        for o in 1..20 {
            p.add_by_pool_offset(o * 8).unwrap();
        }
        assert_eq!(p.size(), 19); // offset 40 again is a duplicate
        assert_eq!(p.compact()[..3], [0, 1, 2]);
        assert!(BytesRefHash::with_capacity(3, 0).is_err());
        assert!(BytesRefHash::with_capacity(0, 0).is_err());
        assert!(BytesRefHash::with_capacity(1, 0).is_err());
        let mut c = BytesRefHash::default();
        c.add(b"q").unwrap();
        c.close();
        c.reinit();
        assert_eq!(c.size(), 0);
        // Regression: `reinit` after `close` used to zero-fill the table,
        // which made this `add` probe forever.
        assert_eq!(c.add(b"q").unwrap(), 0);
        assert_eq!(c.add(b"r").unwrap(), 1);
        assert_eq!(c.add(b"q").unwrap(), -1);
        assert_eq!(c.find(b"r").unwrap(), 1);
        assert_eq!(c.find(b"s").unwrap(), -1);
        assert_eq!(BytesRefBlockPool::default().blocks.len(), 0);
    }

    #[test]
    fn sort_matches_plain_byte_order() {
        // Prefix ties (shared 8-byte prefixes, a term that is a prefix of
        // another, embedded zero bytes) all resolve as unsigned byte order.
        let mut words: Vec<Vec<u8>> = vec![
            b"".to_vec(),
            b"\0".to_vec(),
            b"abcdefgh".to_vec(),
            b"abcdefgh\0".to_vec(),
            b"abcdefghi".to_vec(),
            b"abcdefgg\xff".to_vec(),
            b"\xff\xfe".to_vec(),
            b"ab".to_vec(),
            b"ab\0".to_vec(),
        ];
        let mut s = 7u64;
        for _ in 0..500 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1);
            let len = (s >> 60) as usize;
            let w: Vec<u8> = (0..len).map(|i| (s >> (i * 4)) as u8 & 0x83).collect();
            words.push(w);
        }
        let mut h = BytesRefHash::with_capacity(4, 3).unwrap();
        for w in &words {
            h.add(w).unwrap();
        }
        let n = h.size();
        let sorted = h.sort();
        let got: Vec<&[u8]> = sorted[..n]
            .iter()
            .map(|&id| h.get(id as usize).unwrap())
            .collect();
        let mut want: Vec<&[u8]> = got.clone();
        want.sort();
        want.dedup();
        assert_eq!(got, want);
        assert!(sorted[n..].iter().all(|&v| v == -1));
        let compacted = h.compact();
        assert_eq!(&compacted[..n], &(0..n as i32).collect::<Vec<_>>()[..]);
    }
}
