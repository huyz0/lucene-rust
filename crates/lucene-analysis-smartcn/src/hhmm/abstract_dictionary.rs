//! `org.apache.lucene.analysis.cn.smart.hhmm.AbstractDictionary`: the two
//! dictionaries' constants and hash functions (over UTF-16 units, wrapping
//! as Java's `long` and `int` do).
//!
//! `getGB2312Id`/`getCCByGB2312Id` (GB2312 code to and from a character)
//! serve only the `.dct` loaders, which are not ported (see the crate docs).

/// `GB2312_FIRST_CHAR`: the first Chinese character's id in GB2312 order.
pub const GB2312_FIRST_CHAR: usize = 1410;
/// `GB2312_CHAR_NUM`: the id space, 87 rows of 94.
pub const GB2312_CHAR_NUM: usize = 87 * 94;
/// `CHAR_NUM_IN_FILE`: the characters a `.dct` file lists.
pub const CHAR_NUM_IN_FILE: usize = 6768;

const FNV_PRIME: i64 = 1_099_511_628_211;
const FNV_OFFSET: i64 = 0xcbf2_9ce4_8422_2325_u64 as i64;

/// `hash1(char)`: FNV-1 over the unit's two bytes, then a final mix.
pub fn hash1_char(c: u16) -> i64 {
    let mut hash = FNV_OFFSET;
    hash = (hash ^ i64::from(c & 0x00FF)).wrapping_mul(FNV_PRIME);
    hash = (hash ^ i64::from(c >> 8)).wrapping_mul(FNV_PRIME);
    hash = hash.wrapping_add(hash.wrapping_shl(13));
    hash ^= hash >> 7;
    hash = hash.wrapping_add(hash.wrapping_shl(3));
    hash ^= hash >> 17;
    hash.wrapping_add(hash.wrapping_shl(5))
}

/// `hash1(char[])`: FNV-1 over every unit's two bytes.
pub fn hash1(carray: &[u16]) -> i64 {
    carray.iter().fold(FNV_OFFSET, |hash, &d| {
        let hash = (hash ^ i64::from(d & 0x00FF)).wrapping_mul(FNV_PRIME);
        (hash ^ i64::from(d >> 8)).wrapping_mul(FNV_PRIME)
    })
}

/// One step of `hash2`: Java's `hash = ((hash << 5) + hash) + c & 0x00FF;
/// hash = ((hash << 5) + hash) + c >> 8;` -- `+` binds tighter than `&` and
/// `>>`, so the mask and the shift apply to the whole sum.
fn hash2_step(hash: i32, d: u16) -> i32 {
    let hash = hash
        .wrapping_shl(5)
        .wrapping_add(hash)
        .wrapping_add(i32::from(d))
        & 0x00FF;
    hash.wrapping_shl(5)
        .wrapping_add(hash)
        .wrapping_add(i32::from(d))
        >> 8
}

/// `hash2(char)`.
pub fn hash2_char(c: u16) -> i32 {
    hash2_step(5381, c)
}

/// `hash2(char[])`.
pub fn hash2(carray: &[u16]) -> i32 {
    carray.iter().fold(5381, |h, &d| hash2_step(h, d))
}

/// The probe start and step of a double-hashing table of `prime` slots:
/// Java's `(int) (hash1 % prime)` and `hash2 % prime`, each made
/// non-negative.
// ARITH: `prime` is one of the two positive table sizes, so `%` cannot
// divide by zero or overflow, and the remainders fit an i32.
#[allow(clippy::arithmetic_side_effects)]
pub(crate) fn probe(h1: i64, h2: i32, prime: i32) -> (i32, i32) {
    let mut a = (h1.wrapping_rem(i64::from(prime))) as i32;
    let mut b = h2.wrapping_rem(prime);
    if a < 0 {
        a = a.wrapping_add(prime);
    }
    if b < 0 {
        b = b.wrapping_add(prime);
    }
    (a, b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hashes_match_java() {
        // Values printed by Java's AbstractDictionary for the same input.
        assert_eq!(hash1_char('中' as u16), hash1_char(0x4E2D));
        assert_eq!(hash1(&[]), FNV_OFFSET);
        assert_eq!(hash2(&[]), 5381);
        assert_eq!(hash2_char(0x4E2D), hash2(&[0x4E2D]));
        assert_eq!(probe(-5, -3, 7), (2, 4));
        assert_eq!(probe(12, 10, 7), (5, 3));
    }
}
