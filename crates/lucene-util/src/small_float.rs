//! Port of `org.apache.lucene.util.SmallFloat`'s `int4`/`long4` family, scoped
//! to exactly the pair BM25 norms need: `intToByte4`/`byte4ToInt`. This is the
//! "4-bit-mantissa float-like" encoding `Similarity.computeNorm`'s default
//! implementation uses to squeeze a field's token-count length into a single
//! byte (`SmallFloat.intToByte4(state.getLength())`), and `BM25Similarity`'s
//! `LENGTH_TABLE`/`SimScorer.score` decode it back with `byte4ToInt` to
//! recover an *approximate* field length for the `b * fieldLength /
//! avgFieldLength` term. Not the general-purpose `byteToFloat`/`floatToByte`
//! (24-bit mantissa) pair used elsewhere in Lucene for float encodings — this
//! one only ever encodes non-negative integers.
//!
//! Verified byte-for-byte against `SmallFloat.java` (Lucene 10.5.0), not
//! guessed: the encoding is built on `longToInt4`/`int4ToLong`, a "keep the
//! top 4 significant bits plus a shift" scheme with 24 "free" (subnormal, i.e.
//! not re-encoded, exact) low values below `NUM_FREE_VALUES = 255 -
//! longToInt4(Integer.MAX_VALUE) = 24`.

/// `SmallFloat.longToInt4(long)`-equivalent (kept `pub(crate)` since only
/// [`int_to_byte4`] needs it — real Lucene doesn't expose either as a public
/// standalone API on their own, only through the `*4` byte pair).
fn long_to_int4(i: u64) -> u32 {
    let num_bits = 64 - i.leading_zeros();
    if num_bits < 4 {
        // Subnormal: exact for small values, no encoding needed.
        i as u32
    } else {
        let shift = num_bits - 4;
        // Keep the top 4 significant bits, mask off the implicit leading one,
        // then pack the shift (biased by +1: 0 is reserved for subnormals).
        let encoded = ((i >> shift) as u32) & 0x07;
        encoded | ((shift + 1) << 3)
    }
}

/// `SmallFloat.int4ToLong(int)`-equivalent.
fn int4_to_long(i: u32) -> u64 {
    let bits = (i & 0x07) as u64;
    let shift = i >> 3;
    if shift == 0 {
        // Subnormal.
        bits
    } else {
        (bits | 0x08) << (shift - 1)
    }
}

/// `NUM_FREE_VALUES = 255 - longToInt4(Integer.MAX_VALUE)`: byte values below
/// this decode/encode exactly (subnormal range); Java computes this as a
/// `static final` from `longToInt4`, so this port does too rather than
/// hardcoding the constant it evaluates to (24).
fn num_free_values() -> u32 {
    255 - long_to_int4(i32::MAX as u64)
}

/// `SmallFloat.intToByte4(int)`-equivalent: encodes a non-negative integer
/// (real Lucene: a field's token-count length) into a single byte. Lucene
/// throws `IllegalArgumentException` for negative input; this port's only
/// caller ([`crate`]-external norms-writing code, if any) never has a
/// negative length to encode, so this takes `u32` rather than mirroring
/// Java's runtime-checked `int` parameter.
pub fn int_to_byte4(i: u32) -> u8 {
    let free = num_free_values();
    if i < free {
        i as u8
    } else {
        (free + long_to_int4((i - free) as u64)) as u8
    }
}

/// `SmallFloat.byte4ToInt(byte)`-equivalent: decodes a norm byte (as produced
/// by [`int_to_byte4`] / real Lucene's `Similarity.computeNorm`) back to an
/// approximate integer — `BM25Similarity.LENGTH_TABLE[i] =
/// SmallFloat.byte4ToInt((byte) i)`, indexed by the *unsigned* interpretation
/// of the norm byte (`((byte) encodedNorm) & 0xff`) — this is why this port's
/// signature takes `u8`, not `i8`: callers must already have undone any sign
/// extension (e.g. `norms::norm_value`'s `i64 as u8`) before calling this.
pub fn byte4_to_int(b: u8) -> u32 {
    let i = b as u32;
    let free = num_free_values();
    if i < free {
        i
    } else {
        (free as u64 + int4_to_long(i - free)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn num_free_values_is_24() {
        // Real Lucene's static initializer evaluates `NUM_FREE_VALUES` to 24
        // for `Integer.MAX_VALUE` -- documented here as a concrete, checked
        // value rather than trusting the derivation alone.
        assert_eq!(num_free_values(), 24);
    }

    #[test]
    fn subnormal_range_decodes_exactly() {
        // Bytes below NUM_FREE_VALUES (24) are exact, unencoded lengths --
        // the "subnormal" branch of both longToInt4 and byte4ToInt.
        for b in 0..24u8 {
            assert_eq!(byte4_to_int(b), b as u32, "byte {b}");
        }
    }

    #[test]
    fn known_values_match_real_lucene() {
        // Hand-derived from SmallFloat.java's algorithm (not read off this
        // port's own output) -- see this module's doc comment for the
        // formula. Cross-checked with an independent Python re-implementation
        // of longToInt4/int4ToLong during development.
        let cases: &[(u8, u32)] = &[
            (0, 0),
            (1, 1),
            (23, 23),
            (24, 24),
            (100, 3096),
            (127, 30744),
            (128, 32792),
            (180, 3_145_752),
            (200, 16_777_240),
            (255, 2_013_265_944),
        ];
        for &(b, want) in cases {
            assert_eq!(byte4_to_int(b), want, "byte {b}");
        }
    }

    #[test]
    fn encode_decode_round_trips_for_exactly_representable_lengths() {
        // intToByte4/byte4ToInt aren't a perfect round trip for every integer
        // above NUM_FREE_VALUES (that's the whole point of a lossy 4-bit-
        // mantissa encoding), but every *encoder output*, decoded back, must
        // reproduce a value intToByte4 would re-encode to the same byte
        // (idempotent under re-encoding) -- and the subnormal range and the
        // first few normal-range values are exact both ways.
        for len in 0..=24u32 {
            let b = int_to_byte4(len);
            assert_eq!(byte4_to_int(b), len, "len {len}");
        }
        // A normal-range (lossy) value: encode then decode then re-encode
        // must be a fixed point.
        for len in [50u32, 1000, 100_000, 5_000_000] {
            let b = int_to_byte4(len);
            let decoded = byte4_to_int(b);
            assert_eq!(int_to_byte4(decoded), b, "len {len}");
        }
    }

    #[test]
    fn monotonic_nondecreasing() {
        // SmallFloat's whole purpose is preserving ordering -- a larger input
        // byte must never decode to a smaller length.
        let mut prev = 0u32;
        for b in 0..=255u8 {
            let v = byte4_to_int(b);
            assert!(v >= prev, "byte {b} decoded to {v} < prev {prev}");
            prev = v;
        }
    }
}
