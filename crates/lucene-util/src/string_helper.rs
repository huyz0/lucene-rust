//! Port of `org.apache.lucene.util.StringHelper`: term-prefix helpers
//! (`bytesDifference`, `sortKeyLength`, `startsWith`, `endsWith`), the two
//! MurmurHash3 variants Lucene hashes with, and the segment/commit id stream
//! (`randomId`, `idToString`).
//!
//! `randomId` is Lucene's xorshift128-seeded 128-bit counter: ten xorshift
//! rounds over a 128-bit seed, then each id is the counter's big-endian bytes
//! and the counter increments (mod 2^128). [`IdGenerator`] exposes the
//! stream; [`IdGenerator::from_tests_seed`] reproduces what Lucene does under
//! `-Dtests.seed`, which is how the fixture test pins it against Java.

use std::sync::Mutex;

/// `StringHelper.ID_LENGTH`.
pub const ID_LENGTH: usize = 16;

/// `StringHelper.bytesDifference(prior, current)`: the index of the first
/// differing byte. `None` where Java throws "terms out of order" (the two are
/// equal).
pub fn bytes_difference(prior: &[u8], current: &[u8]) -> Option<usize> {
    let common = prior.len().min(current.len());
    match prior.iter().zip(current).position(|(a, b)| a != b) {
        Some(i) => Some(i),
        None if prior.len() != current.len() => Some(common),
        None => None,
    }
}

/// `StringHelper.sortKeyLength(prior, current)`: `bytesDifference + 1`.
pub fn sort_key_length(prior: &[u8], current: &[u8]) -> Option<usize> {
    bytes_difference(prior, current).map(|d| d + 1)
}

/// `StringHelper.startsWith(ref, prefix)`.
pub fn starts_with(r: &[u8], prefix: &[u8]) -> bool {
    r.starts_with(prefix)
}

/// `StringHelper.endsWith(ref, suffix)`.
pub fn ends_with(r: &[u8], suffix: &[u8]) -> bool {
    r.ends_with(suffix)
}

/// `StringHelper.murmurhash3_x86_32(data, offset, len, seed)` over `data`.
pub fn murmurhash3_x86_32(data: &[u8], seed: i32) -> i32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mut h1 = seed as u32;
    let mut blocks = data.chunks_exact(4);
    for block in &mut blocks {
        let mut k1 = u32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        k1 = k1.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let tail = blocks.remainder();
    if !tail.is_empty() {
        let mut k1 = 0u32;
        for (i, &b) in tail.iter().enumerate() {
            k1 |= (b as u32) << (8 * i);
        }
        k1 = k1.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h1 ^= k1;
    }
    h1 ^= data.len() as u32;
    h1 ^= h1 >> 16;
    h1 = h1.wrapping_mul(0x85eb_ca6b);
    h1 ^= h1 >> 13;
    h1 = h1.wrapping_mul(0xc2b2_ae35);
    h1 ^= h1 >> 16;
    h1 as i32
}

fn fmix64(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(0xff51_afd7_ed55_8ccd);
    h ^= h >> 33;
    h = h.wrapping_mul(0xc4ce_b9fe_1a85_ec53);
    h ^ (h >> 33)
}

/// `StringHelper.murmurhash3_x64_128(data, offset, length, seed)` over
/// `data`; the seed is zero-extended (`seed & 0xFFFFFFFFL`), as in Java.
pub fn murmurhash3_x64_128(data: &[u8], seed: i32) -> [i64; 2] {
    const C1: u64 = 0x87c3_7b91_1142_53d5;
    const C2: u64 = 0x4cf5_ad43_2745_937f;
    let mut h1 = seed as u32 as u64;
    let mut h2 = h1;
    let mut blocks = data.chunks_exact(16);
    for block in &mut blocks {
        let mut w = [0u8; 8];
        w.copy_from_slice(&block[..8]);
        let mut k1 = u64::from_le_bytes(w);
        w.copy_from_slice(&block[8..]);
        let mut k2 = u64::from_le_bytes(w);
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
        h1 = h1.rotate_left(27).wrapping_add(h2);
        h1 = h1.wrapping_mul(5).wrapping_add(0x52dc_e729);
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
        h2 = h2.rotate_left(31).wrapping_add(h1);
        h2 = h2.wrapping_mul(5).wrapping_add(0x3849_5ab5);
    }
    let tail = blocks.remainder();
    if tail.len() > 8 {
        let mut k2 = 0u64;
        for (i, &b) in tail[8..].iter().enumerate() {
            k2 ^= (b as u64) << (8 * i);
        }
        k2 = k2.wrapping_mul(C2).rotate_left(33).wrapping_mul(C1);
        h2 ^= k2;
    }
    if !tail.is_empty() {
        let mut k1 = 0u64;
        for (i, &b) in tail[..tail.len().min(8)].iter().enumerate() {
            k1 ^= (b as u64) << (8 * i);
        }
        k1 = k1.wrapping_mul(C1).rotate_left(31).wrapping_mul(C2);
        h1 ^= k1;
    }
    // Java xors in the `int` length, sign-extended to long.
    let len = data.len() as i32 as i64 as u64;
    h1 ^= len;
    h2 ^= len;
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    h1 = fmix64(h1);
    h2 = fmix64(h2);
    h1 = h1.wrapping_add(h2);
    h2 = h2.wrapping_add(h1);
    [h1 as i64, h2 as i64]
}

/// `StringHelper.murmurhash3_x64_128(BytesRef)`: seed 104729.
pub fn murmurhash3_x64_128_default(data: &[u8]) -> [i64; 2] {
    murmurhash3_x64_128(data, 104_729)
}

/// Lucene's `randomId` stream: a 128-bit counter seeded by ten xorshift128
/// rounds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IdGenerator {
    next: u128,
}

impl IdGenerator {
    /// Seeds from Lucene's `(x0, x1)` pair, running the ten mixing rounds.
    pub fn from_seed(mut x0: i64, mut x1: i64) -> Self {
        for _ in 0..10 {
            let mut s1 = x0;
            let s0 = x1;
            x0 = s0;
            s1 ^= s1 << 23;
            x1 = s1 ^ s0 ^ ((s1 as u64 >> 17) as i64) ^ ((s0 as u64 >> 26) as i64);
        }
        IdGenerator {
            next: ((x0 as u64 as u128) << 64) | x1 as u64 as u128,
        }
    }

    /// What Lucene seeds with under `-Dtests.seed=<prop>`: the last eight
    /// characters parsed as hex, for both halves. `None` where Java's
    /// `Long.parseLong` throws.
    pub fn from_tests_seed(prop: &str) -> Option<Self> {
        let chars: Vec<char> = prop.chars().collect();
        let tail: String = chars[chars.len().saturating_sub(8)..].iter().collect();
        let x0 = parse_java_hex_long(&tail)?;
        Some(Self::from_seed(x0, x0))
    }

    /// Seeds from the OS (`/dev/urandom`), falling back to the clock.
    pub fn from_entropy() -> Self {
        use std::io::Read;
        let mut buf = [0u8; 16];
        let from_os = std::fs::File::open("/dev/urandom")
            .and_then(|mut f| f.read_exact(&mut buf))
            .is_ok();
        if !from_os {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos())
                .unwrap_or(0);
            buf = (nanos ^ ((std::process::id() as u128) << 64)).to_be_bytes();
        }
        let mut a = [0u8; 8];
        a.copy_from_slice(&buf[..8]);
        let x0 = i64::from_be_bytes(a);
        a.copy_from_slice(&buf[8..]);
        Self::from_seed(x0, i64::from_be_bytes(a))
    }

    /// `StringHelper.randomId()`: the counter's 16 big-endian bytes, then
    /// increment mod 2^128.
    pub fn next_id(&mut self) -> [u8; ID_LENGTH] {
        let id = self.next.to_be_bytes();
        self.next = self.next.wrapping_add(1);
        id
    }
}

/// Java's `Long.parseLong(s, 16)` for the (at most eight-character) seed
/// tail: optional sign, then one or more hex digits.
fn parse_java_hex_long(s: &str) -> Option<i64> {
    let (neg, digits) = match s.as_bytes().first() {
        Some(b'-') => (true, &s[1..]),
        Some(b'+') => (false, &s[1..]),
        _ => (false, s),
    };
    if digits.is_empty() {
        return None;
    }
    let mut v: i64 = 0;
    for c in digits.chars() {
        v = v * 16 + c.to_digit(16)? as i64;
    }
    Some(if neg { -v } else { v })
}

static GLOBAL_IDS: Mutex<Option<IdGenerator>> = Mutex::new(None);

/// `StringHelper.randomId()` from the process-wide stream, seeded once from
/// [`IdGenerator::from_entropy`].
pub fn random_id() -> [u8; ID_LENGTH] {
    let mut guard = GLOBAL_IDS.lock().unwrap_or_else(|e| e.into_inner());
    guard
        .get_or_insert_with(IdGenerator::from_entropy)
        .next_id()
}

/// `StringHelper.idToString`: lowercase hex, with " (INVALID FORMAT)" when
/// the id is not [`ID_LENGTH`] bytes; `None` prints `(null)`.
pub fn id_to_string(id: Option<&[u8]>) -> String {
    let Some(id) = id else {
        return "(null)".to_string();
    };
    let mut s: String = id.iter().map(|b| format!("{b:02x}")).collect();
    if id.len() != ID_LENGTH {
        s.push_str(" (INVALID FORMAT)");
    }
    s
}

/// `StringHelper.intsRefToBytesRef`: each int must be a byte value; the
/// error is the offending position (Java's `IllegalArgumentException`).
pub fn ints_ref_to_bytes_ref(ints: &[i32]) -> Result<Vec<u8>, usize> {
    ints.iter()
        .enumerate()
        .map(|(i, &x)| u8::try_from(x).map_err(|_| i))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn prefix_helpers() {
        assert_eq!(bytes_difference(b"abc", b"abd"), Some(2));
        assert_eq!(bytes_difference(b"ab", b"abc"), Some(2));
        assert_eq!(bytes_difference(b"", b"a"), Some(0));
        assert_eq!(bytes_difference(b"abc", b"abc"), None);
        assert_eq!(sort_key_length(b"abc", b"abd"), Some(3));
        assert_eq!(sort_key_length(b"x", b"x"), None);
        assert!(starts_with(b"foobar", b"foo"));
        assert!(!starts_with(b"fo", b"foo"));
        assert!(ends_with(b"foobar", b"bar"));
        assert!(!ends_with(b"ar", b"bar"));
    }

    /// Reference values from the canonical MurmurHash3 test vectors
    /// (seed 0 / seed 1 for x86_32); the Java-fixture test covers the rest.
    #[test]
    fn murmur_known_vectors() {
        assert_eq!(murmurhash3_x86_32(b"", 0), 0);
        assert_eq!(murmurhash3_x86_32(b"", 1), 0x514e_28b7);
        assert_eq!(murmurhash3_x86_32(b"hello", 0), 0x248b_fa47_u32 as i32);
        assert_eq!(
            murmurhash3_x86_32(
                b"The quick brown fox jumps over the lazy dog",
                0x9747_b28c_u32 as i32
            ),
            0x2fa8_26cd
        );
        let [a, b] = murmurhash3_x64_128(b"", 0);
        assert_eq!((a, b), (0, 0));
        assert_ne!(
            murmurhash3_x64_128_default(b"abc"),
            murmurhash3_x64_128(b"abc", 0)
        );
        for len in 0..40 {
            let data: Vec<u8> = (0..len as u8).collect();
            let _ = murmurhash3_x64_128(&data, -1);
        }
    }

    #[test]
    fn id_stream() {
        let mut g = IdGenerator::from_seed(0, 0);
        assert_eq!(g.next_id(), [0; 16]);
        assert_eq!(g.next_id()[15], 1);
        let mut h = IdGenerator { next: u128::MAX };
        assert_eq!(h.next_id(), [0xff; 16]);
        assert_eq!(h.next_id(), [0; 16]);
        assert_eq!(IdGenerator::from_tests_seed("zz"), None);
        assert_eq!(IdGenerator::from_tests_seed(""), None);
        assert_eq!(IdGenerator::from_tests_seed("-"), None);
        assert_eq!(
            IdGenerator::from_tests_seed("123DEADBEEF"),
            IdGenerator::from_tests_seed("DEADBEEF")
        );
        assert_eq!(
            IdGenerator::from_tests_seed("+10"),
            Some(IdGenerator::from_seed(16, 16))
        );
        assert_eq!(
            IdGenerator::from_tests_seed("-10"),
            Some(IdGenerator::from_seed(-16, -16))
        );
        assert_ne!(random_id(), random_id());
        let _ = IdGenerator::from_entropy();
    }

    #[test]
    fn id_to_string_and_ints() {
        assert_eq!(id_to_string(None), "(null)");
        assert_eq!(id_to_string(Some(&[0xab; 16])), "ab".repeat(16));
        assert_eq!(id_to_string(Some(&[1, 2])), "0102 (INVALID FORMAT)");
        assert_eq!(ints_ref_to_bytes_ref(&[0, 255, 7]), Ok(vec![0, 255, 7]));
        assert_eq!(ints_ref_to_bytes_ref(&[0, 256]), Err(1));
        assert_eq!(ints_ref_to_bytes_ref(&[-1]), Err(0));
    }
}
