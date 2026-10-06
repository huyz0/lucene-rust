//! `org.apache.lucene.analysis.minhash.MinHashFilter`.

use std::collections::BTreeSet;
use std::sync::LazyLock;

use crate::attributes::State;
use crate::token_stream::{TokenFilter, TokenStream};
use crate::AnalysisError;

/// `MinHashFilter.DEFAULT_HASH_COUNT`.
pub const DEFAULT_HASH_COUNT: i32 = 1;
/// `MinHashFilter.DEFAULT_HASH_SET_SIZE`.
pub const DEFAULT_HASH_SET_SIZE: i32 = 1;
/// `MinHashFilter.DEFAULT_BUCKET_COUNT`.
pub const DEFAULT_BUCKET_COUNT: i32 = 512;
/// `MinHashFilter.MIN_HASH_TYPE`.
pub const MIN_HASH_TYPE: &str = "MIN_HASH";
const HASH_CACHE_SIZE: usize = 512;

/// `MinHashFilter.LongPair`, ordered as Java's `compareTo`: `val2` then
/// `val1`, both unsigned.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
struct LongPair {
    val1: i64,
    val2: i64,
}

impl Ord for LongPair {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        (self.val2 as u64, self.val1 as u64).cmp(&(o.val2 as u64, o.val1 as u64))
    }
}

impl PartialOrd for LongPair {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}

fn fmix64(mut k: i64) -> i64 {
    k ^= ((k as u64) >> 33) as i64;
    k = k.wrapping_mul(0xff51afd7ed558ccdu64 as i64);
    k ^= ((k as u64) >> 33) as i64;
    k = k.wrapping_mul(0xc4ceb9fe1a85ec53u64 as i64);
    k ^= ((k as u64) >> 33) as i64;
    k
}

/// `MinHashFilter.murmurhash3_x64_128`.
fn murmurhash3_x64_128(key: &[u8], seed: i32) -> LongPair {
    let mut h1 = i64::from(seed as u32);
    let mut h2 = i64::from(seed as u32);
    let c1 = 0x87c37b91114253d5u64 as i64;
    let c2 = 0x4cf5ad432745937fu64 as i64;
    let len = key.len();
    let rounded_end = len & !15;
    let le = |i: usize| i64::from_le_bytes(key[i..i + 8].try_into().expect("8 bytes"));
    let mut i = 0;
    while i < rounded_end {
        let mut k1 = le(i);
        let mut k2 = le(i + 8);
        k1 = k1.wrapping_mul(c1).rotate_left(31).wrapping_mul(c2);
        h1 ^= k1;
        h1 = h1
            .rotate_left(27)
            .wrapping_add(h2)
            .wrapping_mul(5)
            .wrapping_add(0x52dce729);
        k2 = k2.wrapping_mul(c2).rotate_left(33).wrapping_mul(c1);
        h2 ^= k2;
        h2 = h2
            .rotate_left(31)
            .wrapping_add(h1)
            .wrapping_mul(5)
            .wrapping_add(0x38495ab5);
        i += 16;
    }
    let tail = &key[rounded_end..];
    let byte = |j: usize| i64::from(tail[j]);
    let rest = len & 15;
    if rest > 8 {
        let mut k2 = 0i64;
        for j in (8..rest).rev() {
            k2 |= byte(j) << ((j - 8) * 8);
        }
        k2 = k2.wrapping_mul(c2).rotate_left(33).wrapping_mul(c1);
        h2 ^= k2;
    }
    if rest > 0 {
        let mut k1 = 0i64;
        for j in (0..rest.min(8)).rev() {
            k1 |= byte(j) << (j * 8);
        }
        k1 = k1.wrapping_mul(c1).rotate_left(31).wrapping_mul(c2);
        h1 ^= k1;
    }
    h1 ^= len as i64;
    h2 ^= len as i64;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    LongPair { val1: h1, val2: h2 }
}

fn int_hash_uncached(i: i32) -> LongPair {
    murmurhash3_x64_128(&i.to_be_bytes(), 0)
}

static CACHED_INT_HASHES: LazyLock<Vec<LongPair>> =
    LazyLock::new(|| (0..HASH_CACHE_SIZE as i32).map(int_hash_uncached).collect());

/// `MinHashFilter.getIntHash`.
fn int_hash(i: i32) -> LongPair {
    match CACHED_INT_HASHES.get(i as usize) {
        Some(h) => *h,
        None => int_hash_uncached(i),
    }
}

/// `MinHashFilter.combineOrdered`.
fn combine_ordered(hashes: &[LongPair]) -> LongPair {
    let mut r = LongPair::default();
    for h in hashes {
        r.val1 = r.val1.wrapping_mul(37).wrapping_add(h.val1);
        r.val2 = r.val2.wrapping_mul(37).wrapping_add(h.val2);
    }
    r
}

/// `MinHashFilter.FixedSizeTreeSet.add`.
fn bounded_add(set: &mut BTreeSet<LongPair>, capacity: usize, to_add: LongPair) {
    if capacity <= set.len() {
        let last = *set.last().expect("capacity >= 1");
        if to_add >= last {
            return;
        }
        set.pop_last();
    }
    set.insert(to_add);
}

/// `org.apache.lucene.analysis.minhash.MinHashFilter`: the min-hash
/// signature of the whole stream as tokens of hash characters.
pub struct MinHashFilter<I> {
    input: I,
    min_hash_sets: Vec<Vec<BTreeSet<LongPair>>>,
    hash_set_size: usize,
    bucket_count: i32,
    hash_count: i32,
    requires_initialisation: bool,
    end_state: Option<State>,
    hash_position: i32,
    bucket_position: i32,
    bucket_size: i64,
    with_rotation: bool,
    end_offset: i32,
    exhausted: bool,
}

impl<I: TokenStream> MinHashFilter<I> {
    /// `new MinHashFilter(TokenStream, int hashCount, int bucketCount, int
    /// hashSetSize, boolean withRotation)`.
    pub fn new(
        input: I,
        hash_count: i32,
        bucket_count: i32,
        hash_set_size: i32,
        with_rotation: bool,
    ) -> Result<Self, AnalysisError> {
        for (v, name) in [
            (hash_count, "hashCount"),
            (bucket_count, "bucketCount"),
            (hash_set_size, "hashSetSize"),
        ] {
            if v <= 0 {
                return Err(AnalysisError::IllegalArgument(format!(
                    "{name} must be greater than zero"
                )));
            }
        }
        let mut bucket_size = (1i64 << 32) / i64::from(bucket_count);
        if (1i64 << 32) % i64::from(bucket_count) != 0 {
            bucket_size += 1;
        }
        Ok(MinHashFilter {
            input,
            min_hash_sets: vec![vec![BTreeSet::new(); bucket_count as usize]; hash_count as usize],
            hash_set_size: hash_set_size as usize,
            bucket_count,
            hash_count,
            requires_initialisation: true,
            end_state: None,
            hash_position: -1,
            bucket_position: -1,
            bucket_size,
            with_rotation,
            end_offset: 0,
            exhausted: false,
        })
    }

    // Java: MinHashFilter.doRest
    fn do_rest(&mut self) {
        for buckets in &mut self.min_hash_sets {
            for b in buckets {
                b.clear();
            }
        }
        self.end_state = None;
        self.hash_position = -1;
        self.bucket_position = -1;
        self.requires_initialisation = true;
        self.exhausted = false;
    }

    fn term_for(&self, hash: LongPair) -> Vec<u16> {
        let mut t = Vec::with_capacity(8);
        if self.hash_count > 1 {
            t.push((self.hash_position >> 16) as u16);
            t.push(self.hash_position as u16);
        }
        let high = hash.val2;
        t.extend([
            (high >> 48) as u16,
            (high >> 32) as u16,
            (high >> 16) as u16,
            high as u16,
        ]);
        let low = hash.val1;
        t.extend([(low >> 48) as u16, (low >> 32) as u16]);
        if self.hash_count == 1 {
            t.extend([(low >> 16) as u16, low as u16]);
        }
        t
    }
}

impl<I: TokenStream> TokenFilter for MinHashFilter<I> {
    crate::filter_input!();

    // Java: MinHashFilter.incrementToken
    fn increment(&mut self) -> Result<bool, AnalysisError> {
        let mut position_increment = 0;
        if self.requires_initialisation {
            self.requires_initialisation = false;
            let mut found = false;
            while self.input.increment_token()? {
                found = true;
                let a = self.input.attributes();
                let bytes: Vec<u8> = a.term().encode_utf16().flat_map(u16::to_le_bytes).collect();
                for i in 0..self.hash_count {
                    let hash = murmurhash3_x64_128(&bytes, 0);
                    let rehashed = combine_ordered(&[hash, int_hash(i)]);
                    let bucket = ((rehashed.val2 as u64 >> 32) as i64 / self.bucket_size) as usize;
                    bounded_add(
                        &mut self.min_hash_sets[i as usize][bucket],
                        self.hash_set_size,
                        rehashed,
                    );
                }
                self.end_offset = a.end_offset();
            }
            self.exhausted = true;
            self.input.end()?;
            self.end_state = Some(self.input.attributes().capture_state());
            if !found {
                return Ok(false);
            }
            position_increment = 1;
            if self.with_rotation && self.hash_set_size == 1 {
                let bc = self.bucket_count as usize;
                for buckets in &mut self.min_hash_sets {
                    for bucket_loop in 0..bc {
                        if buckets[bucket_loop].is_empty() {
                            for off in 1..bc {
                                let j = (bucket_loop + off) % bc;
                                if let Some(&first) = buckets[j].first() {
                                    bounded_add(&mut buckets[bucket_loop], 1, first);
                                    break;
                                }
                            }
                        }
                    }
                }
            }
        }
        self.input.attributes_mut().clear_attributes();
        while self.hash_position < self.hash_count {
            if self.hash_position == -1 {
                self.hash_position += 1;
                continue;
            }
            while self.bucket_position < self.bucket_count {
                if self.bucket_position == -1 {
                    self.bucket_position += 1;
                    continue;
                }
                let (h, b) = (self.hash_position as usize, self.bucket_position as usize);
                match self.min_hash_sets[h][b].pop_first() {
                    Some(hash) => {
                        let term = self.term_for(hash);
                        let a = self.input.attributes_mut();
                        a.set_term_utf16(&term);
                        a.set_position_increment(position_increment)?;
                        a.set_offset(0, self.end_offset)?;
                        a.set_token_type(MIN_HASH_TYPE);
                        a.set_position_length(1)?;
                        return Ok(true);
                    }
                    None => self.bucket_position += 1,
                }
            }
            self.bucket_position = -1;
            self.hash_position += 1;
        }
        Ok(false)
    }

    // Java: MinHashFilter.end
    fn end_filter(&mut self) -> Result<(), AnalysisError> {
        if !self.exhausted {
            self.input.end()?;
        }
        if let Some(s) = &self.end_state {
            self.input.attributes_mut().restore_state(s);
        }
        Ok(())
    }

    fn reset_filter(&mut self) -> Result<(), AnalysisError> {
        self.input.reset()?;
        self.do_rest();
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::util::canned::Canned;

    #[test]
    fn murmur_matches_the_reference_vectors() {
        // MurmurHash3_x64_128 of "" with seed 0 is all zeros; of "hello"
        // (seed 0) the reference implementation's h1, h2.
        assert_eq!(murmurhash3_x64_128(b"", 0), LongPair { val1: 0, val2: 0 });
        let h = murmurhash3_x64_128(b"hello", 0);
        assert_eq!(
            (h.val1 as u64, h.val2 as u64),
            (0xcbd8a7b341bd9b02, 0x5b1e906a48ae1d19)
        );
        let long = murmurhash3_x64_128(b"The quick brown fox jumps over the lazy dog", 0);
        assert_eq!(
            (long.val1 as u64, long.val2 as u64),
            (0xe34bbc7bbc071b6c, 0x7a433ca9c49a9347)
        );
        assert_eq!(int_hash(600), int_hash_uncached(600));
    }

    #[test]
    fn bounded_sets_and_errors() {
        let mut s = BTreeSet::new();
        for v in [5, 3, 9, 1] {
            bounded_add(&mut s, 2, LongPair { val1: 0, val2: v });
        }
        assert_eq!(s.iter().map(|p| p.val2).collect::<Vec<_>>(), vec![1, 3]);
        assert!(
            LongPair { val1: 0, val2: -1 } > LongPair { val1: 0, val2: 1 },
            "unsigned order"
        );
        assert!(MinHashFilter::new(Canned::parse(""), 0, 1, 1, false).is_err());
        assert!(MinHashFilter::new(Canned::parse(""), 1, 0, 1, false).is_err());
        assert!(MinHashFilter::new(Canned::parse(""), 1, 1, 0, false).is_err());
        let mut f = MinHashFilter::new(Canned::parse("|3|1"), 1, 3, 1, true).unwrap();
        f.reset().unwrap();
        assert!(!f.increment_token().unwrap());
        f.end().unwrap();
        assert_eq!(f.attributes().end_offset(), 3);
        let mut f = MinHashFilter::new(Canned::parse("a:0:1:1:1|3|1"), 1, 3, 1, true).unwrap();
        f.reset().unwrap();
        f.end().unwrap();
        assert_eq!(f.attributes().end_offset(), 3);
    }
}
