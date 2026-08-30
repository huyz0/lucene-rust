//! Port of `org.apache.lucene.util.packed.BlockPackedReaderIterator` (decode
//! side of `BlockPackedWriter`) — read-only, **decode-once, not lazy**.
//!
//! A sequence of values is split into fixed-size blocks (64 values each,
//! for term vectors). Each block is self-describing: a one-byte token
//! (`bitsPerValue << 1 | minValueEquals0`), an optional zigzag-encoded
//! `minValue` (omitted when it's exactly 0), then `bitsPerValue`-wide
//! deltas-from-`minValue` for every value in the block, packed via
//! [`crate::packed_ints`]'s MSB-first bitstream (a `bitsPerValue` of 0 means
//! every value in the block equals `minValue`, with no packed data at all).
//!
//! Ported as a single `decode_all` that materializes the whole sequence,
//! not Java's seekable iterator (`next`/`skip`) -- this port doesn't need
//! partial reads, matching the decode-fully choice already made for
//! `IndexedDISI`, stored fields, and the terms dictionary.

use lucene_store::data_input::DataInput;
use lucene_store::Result;

use crate::packed_ints;

const BLOCK_SIZE: i64 = 64;
const MIN_VALUE_EQUALS_0: u32 = 1;

/// `BlockPackedReaderIterator.readVLong` / `AbstractBlockPackedWriter.writeVLong`
/// -- **not** the same varint as `DataInput.readVLong`.
///
/// Java's generic `readVLong` rejects a 10-byte encoding; this one is
/// deliberately negative-capable: at most 8 continuation groups (bits 0..55),
/// after which a ninth byte contributes all 8 of its bits at shift 56, high bit
/// and all. Reading a block's zigzag-encoded `minValue` with the generic varint
/// instead mis-parses exactly the values whose ninth byte has its top bit set
/// (`|minValue|` around 2^62 and up): the generic reader treats that bit as a
/// continuation marker and swallows a byte of the *next* block.
// ARITH: `shift` starts at 0 and only advances by 7 while `shift < 56`, so
// it never exceeds 56 and `1u64 << shift` stays in range.
#[allow(clippy::arithmetic_side_effects)]
fn read_min_value_vlong(input: &mut impl DataInput) -> Result<i64> {
    let mut l: u64 = 0;
    let mut shift = 0u32;
    while shift < 56 {
        let b = input.read_byte()?;
        l |= ((b & 0x7f) as u64) << shift;
        if b & 0x80 == 0 {
            return Ok(l as i64);
        }
        shift += 7;
    }
    let b = input.read_byte()?;
    Ok((l | ((b as u64) << 56)) as i64)
}

/// Encode side of [`read_min_value_vlong`].
// ARITH: `i` is a `u64` shifted right by a constant 7, and `k` is compared
// against 8 before every increment, so it never leaves `0..=8`.
#[allow(clippy::arithmetic_side_effects)]
fn write_min_value_vlong(out: &mut Vec<u8>, mut i: u64) {
    let mut k = 0;
    while i & !0x7f != 0 && k < 8 {
        out.push(((i & 0x7f) | 0x80) as u8);
        i >>= 7;
        k += 1;
    }
    out.push(i as u8);
}

/// `PackedInts.unsignedBitsRequired`: at least 1, never 0.
// ARITH: `u64::leading_zeros()` returns `0..=64`, so `64 - lz` is in `0..=64`.
#[allow(clippy::arithmetic_side_effects)]
fn unsigned_bits_required(v: i64) -> u32 {
    (64 - (v as u64).leading_zeros()).max(1)
}

/// `PackedInts.maxValue(bitsPerValue)` for `bits_per_value < 64`.
///
/// ARITH: the shift and the subtraction are safe because the sole call site
/// (in [`encode_all`]) reaches this only on the `bits_required != 64` branch,
/// and `bits_required` is itself `unsigned_bits_required`'s output, which is
/// `0..=64`. So `bits_per_value` is in `0..=63` here and `1u64 <<
/// bits_per_value` is at least 1. This is a write-path helper: no caller
/// passes it a width read off disk.
#[allow(clippy::arithmetic_side_effects)]
fn max_value(bits_per_value: u32) -> i64 {
    debug_assert!(bits_per_value < 64);
    ((1u64 << bits_per_value) - 1) as i64
}

/// Decodes `total_value_count` values written by `BlockPackedWriter`.
/// Reads nothing at all if `total_value_count == 0` (matches the writer,
/// which emits zero blocks for an empty stream).
pub(crate) fn decode_all(input: &mut impl DataInput, total_value_count: i64) -> Result<Vec<i64>> {
    // Every caller derives `total_value_count` from a chunk header in a `.tvd`
    // (`chunk_docs`, `total_terms`, `total_positions`, `total_offsets`,
    // `total_payloads`), so it is a value off disk. Reserving it up front lets
    // a corrupt header ask for a multi-terabyte allocation, which is an
    // *abort* -- `catch_unwind` cannot intercept an allocation failure, so it
    // takes the JVM down through the FFI.
    //
    // The input itself bounds how many values can really be there: the
    // cheapest possible block is a single token byte carrying 64 constant
    // values, so `remaining * BLOCK_SIZE` is a hard ceiling. Clamping to it
    // costs a well-formed stream nothing (its bytes are all there, so the
    // ceiling is above `total_value_count`) while a corrupt count can no
    // longer reserve more than the file could hold.
    let ceiling = input.remaining().saturating_mul(BLOCK_SIZE as usize);
    let reserve = (total_value_count.max(0) as usize).min(ceiling);
    let mut out = Vec::with_capacity(reserve);
    let mut produced: i64 = 0;
    while produced < total_value_count {
        let token = input.read_byte()? as u32;
        let min_value_equals_0 = token & MIN_VALUE_EQUALS_0 != 0;
        let bits_per_value = token >> 1;
        if bits_per_value > 64 {
            return Err(lucene_store::Error::Corrupted(format!(
                "block-packed bitsPerValue out of range: {bits_per_value}"
            )));
        }
        let min_value = if min_value_equals_0 {
            0i64
        } else {
            lucene_util::zigzag::decode(1u64.wrapping_add(read_min_value_vlong(input)? as u64))
        };

        // ARITH: `produced < total_value_count` is the loop condition and
        // `produced` starts at 0 and only ever grows, so the difference is in
        // `1..=total_value_count` and cannot overflow.
        #[allow(clippy::arithmetic_side_effects)]
        let block_value_count = (total_value_count - produced).min(BLOCK_SIZE) as usize;
        if bits_per_value == 0 {
            out.extend(std::iter::repeat_n(min_value, block_value_count));
        } else {
            let byte_len = packed_ints::byte_count(block_value_count as u64, bits_per_value);
            let mut block_bytes = vec![0u8; byte_len];
            input.read_bytes(&mut block_bytes)?;
            for i in 0..block_value_count {
                let raw = packed_ints::get(&block_bytes, bits_per_value, i as i64)?;
                out.push(raw.wrapping_add(min_value));
            }
        }
        // ARITH: `block_value_count <= total_value_count - produced` by its
        // definition above, so `produced` lands in `..=total_value_count`.
        #[allow(clippy::arithmetic_side_effects)]
        {
            produced += block_value_count as i64;
        }
    }
    Ok(out)
}

/// Encode side of [`decode_all`]: a full port of `BlockPackedWriter.flush`,
/// including the per-block minimum-value delta encoding.
///
/// Per 64-value block: `bitsPerValue` covers `max - min` rather than `max`,
/// `min` is then pulled down as far as that width still allows (Java's
/// `min = max(0, max - maxValue(bitsPerValue))`, which shortens the
/// `minValue` varint without widening anything), and `min == 0` is signalled
/// by the token's low bit instead of being written out.
///
/// This used to hardcode `minValue = 0` and size `bitsPerValue` from the block
/// max alone, which is not merely non-minimal: it is **wrong for any block
/// containing a negative value**. A block of all-negative values took
/// `max <= 0 -> bitsPerValue = 0`, encoding every value as the constant `0`;
/// a mixed block truncated each negative value to the low `bitsPerValue` bits.
/// That was latent when it was fixed and is **not latent any more** (c33):
/// the term-vectors writer stores a length as `span - prefixLength -
/// suffixLength`, where `span` is `endOffset - startOffset`: the
/// occurrence's span minus the term's length *in UTF-8 bytes* -- exactly
/// what Java's
/// `Lucene90CompressingTermVectorsWriter.flushOffsets` writes and what its
/// reader adds back. The span is now measured in UTF-16 code units
/// (`OffsetAttribute`'s unit; it was UTF-8 bytes here until c33), so **every
/// multi-byte term produces a negative length**: `caf\u{e9}` spans 4 `char`s
/// and occupies 5 bytes, giving `-1`. Start-offset deltas can be negative
/// too, since `charsPerTerm * position` may overshoot. `BlockPackedWriter` is
/// a general signed-`long` sequence writer and its decoder here already
/// handled negatives correctly on the read side.
///
/// Writes nothing for an empty slice, matching the decoder reading nothing for
/// `total_value_count == 0`.
pub(crate) fn encode_all(values: &[i64]) -> Vec<u8> {
    let mut out = Vec::new();
    for block in values.chunks(BLOCK_SIZE as usize) {
        let mut min = *block.iter().min().unwrap();
        let max = *block.iter().max().unwrap();

        // Java lets `max - min` overflow into a negative `delta` and relies on
        // `unsignedBitsRequired` reading it as unsigned -- which lands on 64,
        // the only width that can hold the pair anyway.
        let delta = max.wrapping_sub(min);
        let bits_required = if delta == 0 {
            0
        } else {
            unsigned_bits_required(delta)
        };
        if bits_required == 64 {
            // No need to delta-encode: every value fits raw.
            min = 0;
        } else if min > 0 {
            // Make min as small as possible so that its varint is shorter.
            min = 0.max(max.wrapping_sub(max_value(bits_required)));
        }

        let token = (bits_required << 1) | if min == 0 { MIN_VALUE_EQUALS_0 } else { 0 };
        out.push(token as u8);
        if min != 0 {
            write_min_value_vlong(&mut out, lucene_util::zigzag::encode(min).wrapping_sub(1));
        }
        if bits_required > 0 {
            let deltas: Vec<i64> = block.iter().map(|&v| v.wrapping_sub(min)).collect();
            out.extend_from_slice(&packed_ints::encode(&deltas, bits_required));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_store::data_input::SliceInput;

    fn write_vlong(out: &mut Vec<u8>, mut v: u64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v >>= 7;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    /// `total_value_count` is a chunk header field off a `.tvd`. Reserving it
    /// verbatim turned a corrupt header into a multi-gigabyte allocation --
    /// an abort, which `catch_unwind` cannot intercept -- before the decoder
    /// had read a single block. The loop itself is bounded by the input, so
    /// a truncated stream must come back as a decode error.
    #[test]
    fn absurd_total_value_count_errors_instead_of_reserving_it() {
        let bytes = [1u8]; // one bitsPerValue=0, min=0 block: 64 values
        let mut input = SliceInput::new(&bytes);
        let got = decode_all(&mut input, i64::MAX);
        assert!(got.is_err(), "{got:?}");
    }

    #[test]
    fn empty_stream_reads_nothing() {
        let mut input = SliceInput::new(&[]);
        assert_eq!(decode_all(&mut input, 0).unwrap(), Vec::<i64>::new());
    }

    #[test]
    fn single_block_all_same_value() {
        // bitsPerValue=0, minValueEquals0=1 (bit0 set -> no minValue byte,
        // implied value 0) -> token=1.
        let bytes = [1u8];
        let mut input = SliceInput::new(&bytes);
        assert_eq!(decode_all(&mut input, 3).unwrap(), vec![0, 0, 0]);
    }

    #[test]
    fn single_block_constant_nonzero_value() {
        // bitsPerValue=0, minValueEquals0=0 -> token=0, then zigzag(1+vlong)=minValue.
        // want minValue=42: 1+vlong == zigzag_encode(42) => vlong = zigzag_encode(42)-1
        let mut bytes = vec![0u8]; // token: bits=0, min_equals_0=false
        let target = lucene_util::zigzag::encode(42) - 1;
        write_vlong(&mut bytes, target);
        let mut input = SliceInput::new(&bytes);
        assert_eq!(decode_all(&mut input, 4).unwrap(), vec![42, 42, 42, 42]);
    }

    #[test]
    fn single_block_bit_packed_deltas_from_min() {
        // 5 values: [10, 12, 11, 10, 13] -> min=10, deltas=[0,2,1,0,3], bpv=2.
        let min_value = 10i64;
        let deltas = [0u8, 2, 1, 0, 3];
        let bits_per_value = 2u32;
        let mut bytes = Vec::new();
        let token = bits_per_value << 1; // min_equals_0=false (min=10 != 0)
        bytes.push(token as u8);
        let target = lucene_util::zigzag::encode(min_value) - 1;
        write_vlong(&mut bytes, target);
        // pack deltas MSB-first, 2 bits each: values 0,2,1,0,3 ->
        // byte0 = 00_10_01_00 (first 4 values), byte1 = 11_000000 (5th value)
        let mut packed_bits: u16 = 0;
        let mut nbits: usize = 0;
        for &d in &deltas {
            packed_bits = (packed_bits << 2) | d as u16;
            nbits += 2;
        }
        // left-align into full bytes
        let total_bytes = nbits.div_ceil(8);
        let packed_bits = packed_bits << (total_bytes * 8 - nbits);
        for i in (0..total_bytes).rev() {
            bytes.push(((packed_bits >> (i * 8)) & 0xFF) as u8);
        }
        let mut input = SliceInput::new(&bytes);
        assert_eq!(decode_all(&mut input, 5).unwrap(), vec![10, 12, 11, 10, 13]);
    }

    #[test]
    fn multiple_blocks_across_65_values() {
        // 65 values forces a second block (block size 64): first block all
        // zero (bpv=0, min=0), second block a single constant value 7.
        let mut bytes = vec![1u8]; // block 0: bpv=0, min_equals_0=true (min=0)
        bytes.push(0u8); // block 1: bpv=0, min_equals_0=false
        let target = lucene_util::zigzag::encode(7) - 1;
        write_vlong(&mut bytes, target);
        let mut input = SliceInput::new(&bytes);
        let values = decode_all(&mut input, 65).unwrap();
        assert_eq!(values.len(), 65);
        assert!(values[..64].iter().all(|&v| v == 0));
        assert_eq!(values[64], 7);
    }

    #[test]
    fn invalid_bits_per_value_rejected() {
        // token >> 1 must be <= 64; use a byte where bits_per_value = 127.
        let bytes = [0xFEu8]; // 0xFE >> 1 = 127
        let mut input = SliceInput::new(&bytes);
        assert!(decode_all(&mut input, 1).is_err());
    }

    #[test]
    fn encode_all_empty_writes_nothing() {
        assert_eq!(encode_all(&[]), Vec::<u8>::new());
    }

    #[test]
    fn encode_all_round_trips_through_decode_all_single_block() {
        let values = vec![0i64, 2, 1, 0, 3, 100];
        let encoded = encode_all(&values);
        let mut input = SliceInput::new(&encoded);
        assert_eq!(decode_all(&mut input, values.len() as i64).unwrap(), values);
    }

    #[test]
    fn encode_all_round_trips_across_multiple_blocks() {
        let values: Vec<i64> = (0..130).map(|i| (i * 37) % 1000).collect();
        let encoded = encode_all(&values);
        let mut input = SliceInput::new(&encoded);
        assert_eq!(decode_all(&mut input, values.len() as i64).unwrap(), values);
    }

    #[test]
    fn encode_all_round_trips_negative_and_mixed_sign_values() {
        // The case the old `minValue = 0` writer silently destroyed: a block
        // whose max is <= 0 used to pick bitsPerValue = 0 and decode as all
        // zeros, and a mixed-sign block truncated the negatives.
        for values in [
            vec![-5i64, -3, -100, -1],
            vec![-5i64, 10, -3, 0, 7],
            vec![i64::MIN, 0, i64::MAX],
            vec![i64::MIN; 3],
            (0..130).map(|i| i - 65).collect::<Vec<i64>>(),
        ] {
            let encoded = encode_all(&values);
            let mut input = SliceInput::new(&encoded);
            assert_eq!(
                decode_all(&mut input, values.len() as i64).unwrap(),
                values,
                "values={values:?}"
            );
        }
    }

    #[test]
    fn encode_all_matches_javas_flush_min_value_choice() {
        // Java: delta = 300 - 300 == 0 -> bitsPerValue 0, min stays 300, so the
        // token has minValueEquals0 clear and the minValue varint follows.
        let encoded = encode_all(&[300, 300, 300]);
        assert_eq!(encoded[0], 0, "bitsPerValue=0, minValueEquals0=false");
        let mut input = SliceInput::new(&encoded[1..]);
        let min = lucene_util::zigzag::decode(
            1u64.wrapping_add(read_min_value_vlong(&mut input).unwrap() as u64),
        );
        assert_eq!(min, 300);

        // delta = 260 - 256 = 4 -> bitsPerValue 3, maxValue(3) = 7, so Java
        // lowers min from 256 to max(0, 260 - 7) = 253 rather than keeping the
        // real block minimum.
        let encoded = encode_all(&[256, 258, 260]);
        assert_eq!(encoded[0], 3 << 1, "bitsPerValue=3, minValueEquals0=false");
        let mut input = SliceInput::new(&encoded[1..]);
        let min = lucene_util::zigzag::decode(
            1u64.wrapping_add(read_min_value_vlong(&mut input).unwrap() as u64),
        );
        assert_eq!(min, 253);
    }

    #[test]
    fn min_value_vlong_round_trips_the_nine_byte_negative_capable_form() {
        // The generic `DataInput::read_vlong` would treat the ninth byte's top
        // bit as a continuation marker and read into the next block.
        for v in [
            0u64,
            1,
            0x7f,
            0x80,
            (1 << 56) - 1,
            1 << 56,
            u64::MAX,
            u64::MAX - 1,
            0xFFFF_FFFF_FFFF_FFF0,
        ] {
            let mut bytes = Vec::new();
            write_min_value_vlong(&mut bytes, v);
            assert!(bytes.len() <= 9, "v={v:#x} took {} bytes", bytes.len());
            let mut input = SliceInput::new(&bytes);
            assert_eq!(
                read_min_value_vlong(&mut input).unwrap() as u64,
                v,
                "v={v:#x}"
            );
        }
    }

    #[test]
    fn encode_all_all_zero_block_uses_zero_bits() {
        let values = vec![0i64; 64];
        let encoded = encode_all(&values);
        assert_eq!(encoded, vec![1u8]); // token: bits=0, min_equals_0=1
    }
}
