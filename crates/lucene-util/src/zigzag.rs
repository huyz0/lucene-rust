//! ZigZag encoding, matching `org.apache.lucene.util.BitUtil.zigZag{Encode,Decode}`.

#[inline]
pub fn encode(v: i64) -> u64 {
    ((v >> 63) ^ (v << 1)) as u64
}

#[inline]
pub fn decode(v: u64) -> i64 {
    ((v >> 1) as i64) ^ -((v & 1) as i64)
}

/// `BitUtil.zigZagEncode(int)`: the 32-bit variant, used by
/// `DataOutput.writeZInt` (stored fields' numeric ints, term-vector and
/// doc-values deltas). Not the same function as [`encode`] narrowed -- the
/// sign bit lives at bit 31, so encoding an `i32` through the 64-bit variant
/// would produce a different (wider) result.
#[inline]
pub fn encode_i32(v: i32) -> u32 {
    ((v >> 31) ^ (v << 1)) as u32
}

/// `BitUtil.zigZagDecode(int)`: inverse of [`encode_i32`].
#[inline]
pub fn decode_i32(v: u32) -> i32 {
    ((v >> 1) as i32) ^ -((v & 1) as i32)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_edges() {
        for v in [0, 1, -1, i64::MAX, i64::MIN, 123456789, -987654321] {
            assert_eq!(decode(encode(v)), v);
        }
    }

    #[test]
    fn roundtrip_edges_i32() {
        for v in [0i32, 1, -1, i32::MAX, i32::MIN, 123_456_789, -987_654_321] {
            assert_eq!(decode_i32(encode_i32(v)), v, "v={v}");
        }
    }

    /// Hand-computed from `BitUtil.zigZagEncode(int)`'s definition -- the
    /// 32-bit variant must map small magnitudes to small unsigned values
    /// (that is the whole point: `writeZInt(-1)` is one byte, not five).
    #[test]
    fn encode_i32_known_values() {
        assert_eq!(encode_i32(0), 0);
        assert_eq!(encode_i32(-1), 1);
        assert_eq!(encode_i32(1), 2);
        assert_eq!(encode_i32(-2), 3);
        assert_eq!(encode_i32(i32::MAX), u32::MAX - 1);
        assert_eq!(encode_i32(i32::MIN), u32::MAX);
    }

    /// The 32-bit and 64-bit variants agree on every value an `i32` can
    /// hold once the 64-bit result is narrowed -- they must not, however, be
    /// used interchangeably on the wire (different byte counts).
    #[test]
    fn i32_and_i64_variants_agree_on_the_i32_range() {
        for v in [0i32, 1, -1, i32::MAX, i32::MIN, 77, -77] {
            assert_eq!(encode_i32(v) as u64, encode(v as i64), "v={v}");
        }
    }

    /// Cross-checked against Java BitUtil via fixtures/data/zigzag_pairs.expected.
    #[test]
    fn matches_java_reference() {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../fixtures/data/zigzag_pairs.expected"
        );
        let text = std::fs::read_to_string(path).expect("run fixtures/ generator first");
        for line in text.lines() {
            let mut it = line.split_whitespace();
            let v: i64 = it.next().unwrap().parse().unwrap();
            let enc: i64 = it.next().unwrap().parse().unwrap(); // Java prints as signed
            assert_eq!(encode(v), enc as u64, "encode({v})");
            assert_eq!(decode(enc as u64), v, "decode({enc})");
        }
    }
}
