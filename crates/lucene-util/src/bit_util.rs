//! Port of `org.apache.lucene.util.BitUtil`'s bit tricks: the Morton
//! interleave used by `LatLonPoint`-style 2-D encodings, `flipFlop`,
//! `nextHighestPowerOfTwo` and `isZeroOrPowerOfTwo`. The zig-zag pair lives
//! in [`crate::zigzag`] (re-exported here), and Java's `VarHandle`
//! little/big-endian accessors are `from_le_bytes`/`from_be_bytes` at each
//! call site.

pub use crate::zigzag::{decode as zig_zag_decode, decode_i32 as zig_zag_decode_i32};
pub use crate::zigzag::{encode as zig_zag_encode, encode_i32 as zig_zag_encode_i32};

const MAGIC0: u64 = 0x5555555555555555;
const MAGIC1: u64 = 0x3333333333333333;
const MAGIC2: u64 = 0x0F0F0F0F0F0F0F0F;
const MAGIC3: u64 = 0x00FF00FF00FF00FF;
const MAGIC4: u64 = 0x0000FFFF0000FFFF;
const MAGIC5: u64 = 0x00000000FFFFFFFF;
const MAGIC6: u64 = 0xAAAAAAAAAAAAAAAA;

/// `BitUtil.nextHighestPowerOfTwo(int)`: the smallest power of two `>= v`
/// (Java's bit smear: 0 maps to 0, and past `2^30` it wraps to `i32::MIN`).
pub fn next_highest_power_of_two_i32(v: i32) -> i32 {
    let mut v = v.wrapping_sub(1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v.wrapping_add(1)
}

/// `BitUtil.nextHighestPowerOfTwo(long)`.
pub fn next_highest_power_of_two_i64(v: i64) -> i64 {
    let mut v = v.wrapping_sub(1);
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v |= v >> 32;
    v.wrapping_add(1)
}

/// `BitUtil.interleave(even, odd)`: the Morton code of two 32-bit values,
/// `even`'s bits in the even positions.
pub fn interleave(even: i32, odd: i32) -> i64 {
    let spread = |v: u64| {
        let v = (v | (v << 16)) & MAGIC4;
        let v = (v | (v << 8)) & MAGIC3;
        let v = (v | (v << 4)) & MAGIC2;
        let v = (v | (v << 2)) & MAGIC1;
        (v | (v << 1)) & MAGIC0
    };
    let v1 = spread(even as u32 as u64);
    let v2 = spread(odd as u32 as u64);
    ((v2 << 1) | v1) as i64
}

/// `BitUtil.deinterleave(b)`: the even bits of `b`, compacted (shift `b`
/// right by one first for the odd bits).
pub fn deinterleave(b: i64) -> i64 {
    let mut b = b as u64 & MAGIC0;
    b = (b ^ (b >> 1)) & MAGIC1;
    b = (b ^ (b >> 2)) & MAGIC2;
    b = (b ^ (b >> 4)) & MAGIC3;
    b = (b ^ (b >> 8)) & MAGIC4;
    b = (b ^ (b >> 16)) & MAGIC5;
    b as i64
}

/// `BitUtil.flipFlop(b)`: swaps each even bit with the odd bit above it.
pub fn flip_flop(b: i64) -> i64 {
    let b = b as u64;
    (((b & MAGIC6) >> 1) | ((b & MAGIC0) << 1)) as i64
}

/// `BitUtil.isZeroOrPowerOfTwo(x)`.
pub fn is_zero_or_power_of_two(x: i32) -> bool {
    x & x.wrapping_sub(1) == 0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn powers_of_two() {
        assert_eq!(next_highest_power_of_two_i32(0), 0);
        assert_eq!(next_highest_power_of_two_i32(1), 1);
        assert_eq!(next_highest_power_of_two_i32(3), 4);
        assert_eq!(next_highest_power_of_two_i32(1024), 1024);
        assert_eq!(next_highest_power_of_two_i32(1025), 2048);
        assert_eq!(next_highest_power_of_two_i32((1 << 30) + 1), i32::MIN);
        assert_eq!(next_highest_power_of_two_i64(5), 8);
        assert_eq!(next_highest_power_of_two_i64((1i64 << 40) + 1), 1i64 << 41);
        assert!(is_zero_or_power_of_two(0));
        assert!(is_zero_or_power_of_two(64));
        assert!(!is_zero_or_power_of_two(65));
        assert!(is_zero_or_power_of_two(i32::MIN));
    }

    #[test]
    fn interleave_round_trips_and_places_bits() {
        assert_eq!(interleave(1, 0), 1);
        assert_eq!(interleave(0, 1), 2);
        assert_eq!(interleave(-1, -1), -1);
        for (e, o) in [(0x1234_5678, -0x0765_4321), (i32::MIN, i32::MAX), (7, 9)] {
            let m = interleave(e, o);
            assert_eq!(deinterleave(m) as u32 as i32, e);
            assert_eq!(deinterleave(m >> 1) as u32 as i32, o);
            let f = flip_flop(m);
            assert_eq!(deinterleave(f) as u32 as i32, o);
        }
        assert_eq!(zig_zag_encode(-1), 1);
        assert_eq!(zig_zag_decode_i32(zig_zag_encode_i32(-5)), -5);
        assert_eq!(zig_zag_decode(zig_zag_encode(7)), 7);
    }
}
