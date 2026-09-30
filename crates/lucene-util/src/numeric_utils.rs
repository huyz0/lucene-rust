//! Port of `org.apache.lucene.util.NumericUtils`: the order-preserving
//! `f32` <-> `i32` and `f64` <-> `i64` mappings, the big-endian sortable-byte
//! encodings points use, and the per-dimension byte `add`/`subtract`.
//!
//! Java's `floatToIntBits`/`doubleToLongBits` canonicalise every NaN to one
//! bit pattern before the transform; Rust's `to_bits` does not, so the
//! `*_to_sortable_*` functions canonicalise explicitly.
//! (Some crates still carry private copies of the double/long half, recorded
//! in `docs/sweep/m2/LEDGER.md`.)

/// `NumericUtils.sortableFloatBits`.
pub fn sortable_float_bits(bits: i32) -> i32 {
    bits ^ ((bits >> 31) & 0x7fff_ffff)
}

/// `NumericUtils.floatToSortableInt`.
pub fn float_to_sortable_int(value: f32) -> i32 {
    sortable_float_bits(float_to_int_bits(value))
}

/// Java's `Float.floatToIntBits`: `to_bits` with every NaN collapsed to
/// `0x7fc00000`.
pub fn float_to_int_bits(value: f32) -> i32 {
    if value.is_nan() {
        0x7fc0_0000
    } else {
        value.to_bits() as i32
    }
}

/// Java's `Double.doubleToLongBits`: `to_bits` with every NaN collapsed to
/// `0x7ff8000000000000`.
pub fn double_to_long_bits(value: f64) -> i64 {
    if value.is_nan() {
        0x7ff8_0000_0000_0000
    } else {
        value.to_bits() as i64
    }
}

/// `NumericUtils.sortableDoubleBits`.
pub fn sortable_double_bits(bits: i64) -> i64 {
    bits ^ ((bits >> 63) & 0x7fff_ffff_ffff_ffff)
}

/// `NumericUtils.doubleToSortableLong`.
pub fn double_to_sortable_long(value: f64) -> i64 {
    sortable_double_bits(double_to_long_bits(value))
}

/// `NumericUtils.sortableLongToDouble`.
pub fn sortable_long_to_double(encoded: i64) -> f64 {
    f64::from_bits(sortable_double_bits(encoded) as u64)
}

/// `NumericUtils.intToSortableBytes`: 4 big-endian bytes, sign bit flipped.
/// Panics if `result[offset..offset + 4]` is out of range (Java's
/// `IndexOutOfBoundsException`).
pub fn int_to_sortable_bytes(value: i32, result: &mut [u8], offset: usize) {
    result[offset..offset + 4].copy_from_slice(&((value ^ i32::MIN).to_be_bytes()));
}

/// `NumericUtils.sortableBytesToInt`.
pub fn sortable_bytes_to_int(encoded: &[u8], offset: usize) -> i32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&encoded[offset..offset + 4]);
    i32::from_be_bytes(b) ^ i32::MIN
}

/// `NumericUtils.longToSortableBytes`: 8 big-endian bytes, sign bit flipped.
pub fn long_to_sortable_bytes(value: i64, result: &mut [u8], offset: usize) {
    result[offset..offset + 8].copy_from_slice(&((value ^ i64::MIN).to_be_bytes()));
}

/// `NumericUtils.sortableBytesToLong`.
pub fn sortable_bytes_to_long(encoded: &[u8], offset: usize) -> i64 {
    let mut b = [0u8; 8];
    b.copy_from_slice(&encoded[offset..offset + 8]);
    i64::from_be_bytes(b) ^ i64::MIN
}

/// Error of [`big_int_to_sortable_bytes`]: the value needs more than
/// `bigIntSize` bytes (Java's `IllegalArgumentException`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct BigIntTooLarge;

/// `NumericUtils.bigIntToSortableBytes`, taking the value as
/// `BigInteger.toByteArray()` gives it: minimal two's-complement, big-endian.
/// Sign-extends to `big_int_size` bytes and flips the top bit.
pub fn big_int_to_sortable_bytes(
    twos_complement: &[u8],
    big_int_size: usize,
    result: &mut [u8],
    offset: usize,
) -> Result<(), BigIntTooLarge> {
    let n = twos_complement.len();
    if n > big_int_size || n == 0 {
        return Err(BigIntTooLarge);
    }
    let out = &mut result[offset..offset + big_int_size];
    let pad = big_int_size - n;
    let fill = if twos_complement[0] & 0x80 != 0 {
        0xff
    } else {
        0
    };
    out[..pad].fill(fill);
    out[pad..].copy_from_slice(twos_complement);
    out[0] ^= 0x80;
    Ok(())
}

/// `NumericUtils.sortableBytesToBigInt`: the value's two's-complement bytes
/// (not minimised; `length` bytes, as `new BigInteger(byte[])` accepts them).
pub fn sortable_bytes_to_big_int(encoded: &[u8], offset: usize, length: usize) -> Vec<u8> {
    let mut v = encoded[offset..offset + length].to_vec();
    if let Some(first) = v.first_mut() {
        *first ^= 0x80;
    }
    v
}

/// Error of [`subtract`] (`a < b`) and [`add`] (overflow).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ByteArithmeticOverflow;

/// `NumericUtils.subtract`: `result = a - b` over dimension `dim`'s
/// `bytes_per_dim` unsigned big-endian bytes. Java's 4-byte stride is an
/// optimisation of the same borrow chain, so this is byte-wise.
pub fn subtract(
    bytes_per_dim: usize,
    dim: usize,
    a: &[u8],
    b: &[u8],
    result: &mut [u8],
) -> Result<(), ByteArithmeticOverflow> {
    let start = dim * bytes_per_dim;
    let mut borrow = 0i32;
    for i in (start..start + bytes_per_dim).rev() {
        let diff = a[i] as i32 - b[i] as i32 - borrow;
        borrow = i32::from(diff < 0);
        result[i - start] = diff as u8;
    }
    if borrow != 0 {
        return Err(ByteArithmeticOverflow);
    }
    Ok(())
}

/// `NumericUtils.add`: `result = a + b` over dimension `dim`'s
/// `bytes_per_dim` unsigned big-endian bytes.
pub fn add(
    bytes_per_dim: usize,
    dim: usize,
    a: &[u8],
    b: &[u8],
    result: &mut [u8],
) -> Result<(), ByteArithmeticOverflow> {
    let start = dim * bytes_per_dim;
    let mut carry = 0u32;
    for i in (start..start + bytes_per_dim).rev() {
        let sum = a[i] as u32 + b[i] as u32 + carry;
        carry = u32::from(sum >= 256);
        result[i - start] = sum as u8;
    }
    if carry != 0 {
        return Err(ByteArithmeticOverflow);
    }
    Ok(())
}

/// `NumericUtils.sortableIntToFloat`.
pub fn sortable_int_to_float(encoded: i32) -> f32 {
    f32::from_bits(sortable_float_bits(encoded) as u32)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The point of the mapping: the `i32` order is the `f32` order, across
    /// zero and across the sign boundary, and the transform is its own
    /// inverse. `NeighborQueue` packs a score through this so a plain integer
    /// heap orders by score.
    #[test]
    fn the_encoding_round_trips_and_preserves_order() {
        for v in [-1e30f32, -1.0, -0.0, 0.0, 1.0, 1e30] {
            assert_eq!(sortable_int_to_float(float_to_sortable_int(v)), v);
        }
        let ordered = [
            f32::NEG_INFINITY,
            -1e30,
            -1.0,
            -f32::MIN_POSITIVE,
            -0.0,
            0.0,
            f32::MIN_POSITIVE,
            1.0,
            1e30,
            f32::INFINITY,
        ];
        let encoded: Vec<i32> = ordered.iter().copied().map(float_to_sortable_int).collect();
        let mut sorted = encoded.clone();
        sorted.sort_unstable();
        assert_eq!(encoded, sorted, "the i32 order must be the f32 order");
        assert!(float_to_sortable_int(-1.0) < float_to_sortable_int(0.0));
        assert!(float_to_sortable_int(0.0) < float_to_sortable_int(1.0));
    }

    #[test]
    fn double_and_bytes_encodings() {
        let ordered = [
            f64::NEG_INFINITY,
            -1.5,
            -0.0,
            0.0,
            2.0,
            f64::INFINITY,
            f64::NAN,
        ];
        let enc: Vec<i64> = ordered
            .iter()
            .map(|&v| double_to_sortable_long(v))
            .collect();
        assert!(enc.windows(2).all(|w| w[0] < w[1]));
        assert_eq!(sortable_long_to_double(double_to_sortable_long(-1.5)), -1.5);
        let odd_nan = f64::from_bits(0x7ff8_0000_0000_0001);
        assert_eq!(
            double_to_sortable_long(odd_nan),
            double_to_sortable_long(f64::NAN)
        );
        let odd_nan32 = f32::from_bits(0x7fc0_0001);
        assert_eq!(
            float_to_sortable_int(odd_nan32),
            float_to_sortable_int(f32::NAN)
        );
        assert_eq!(double_to_long_bits(1.0), 1.0f64.to_bits() as i64);

        let mut buf = [0u8; 13];
        int_to_sortable_bytes(-2, &mut buf, 1);
        assert_eq!(&buf[1..5], &[0x7f, 0xff, 0xff, 0xfe]);
        assert_eq!(sortable_bytes_to_int(&buf, 1), -2);
        long_to_sortable_bytes(5, &mut buf, 5);
        assert_eq!(&buf[5..13], &[0x80, 0, 0, 0, 0, 0, 0, 5]);
        assert_eq!(sortable_bytes_to_long(&buf, 5), 5);

        let mut big = [0u8; 6];
        big_int_to_sortable_bytes(&[0xff, 0x38], 4, &mut big, 1).unwrap(); // -200
        assert_eq!(&big[1..5], &[0x7f, 0xff, 0xff, 0x38]);
        assert_eq!(
            sortable_bytes_to_big_int(&big, 1, 4),
            vec![0xff, 0xff, 0xff, 0x38]
        );
        big_int_to_sortable_bytes(&[0x01, 0x00], 3, &mut big, 0).unwrap();
        assert_eq!(&big[..3], &[0x80, 0x01, 0x00]);
        assert_eq!(
            big_int_to_sortable_bytes(&[1, 2, 3], 2, &mut big, 0),
            Err(BigIntTooLarge)
        );
        assert_eq!(
            big_int_to_sortable_bytes(&[], 2, &mut big, 0),
            Err(BigIntTooLarge)
        );
        assert!(sortable_bytes_to_big_int(&big, 0, 0).is_empty());
    }

    #[test]
    fn byte_add_and_subtract() {
        let a = [9, 9, 0x01, 0x00, 0x00, 0x00, 0x00];
        let b = [9, 9, 0x00, 0x00, 0x00, 0x00, 0x01];
        let mut r = [0u8; 7];
        subtract(7, 0, &a, &b, &mut r).unwrap();
        assert_eq!(r, [0, 0, 0x00, 0xff, 0xff, 0xff, 0xff]);
        let mut junk = [0u8; 7];
        assert_eq!(
            subtract(7, 0, &b, &a, &mut junk),
            Err(ByteArithmeticOverflow)
        );
        let mut s = [0u8; 7];
        add(7, 0, &r, &b, &mut s).unwrap();
        assert_eq!(s, a);
        assert_eq!(
            add(2, 0, &[0xff, 0xff], &[0, 1], &mut s),
            Err(ByteArithmeticOverflow)
        );
        let mut t = [0u8; 2];
        add(2, 1, &[0, 0, 1, 0xff], &[0, 0, 0, 1], &mut t).unwrap();
        assert_eq!(t, [2, 0]);
    }

    /// `sortableFloatBits` is an involution on the bit pattern.
    #[test]
    fn sortable_float_bits_is_its_own_inverse() {
        for bits in [0i32, 1, -1, i32::MIN, i32::MAX, 0x4048_f5c3] {
            assert_eq!(sortable_float_bits(sortable_float_bits(bits)), bits);
        }
    }
}
