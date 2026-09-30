//! The retired postings formats' bit-packing: ports of
//! `backward_codecs.lucene90.ForUtil`/`PForUtil`,
//! `backward_codecs.lucene99.ForUtil`/`ForDeltaUtil`/`PForUtil`,
//! `backward_codecs.lucene912.ForUtil`/`ForDeltaUtil`/`PForUtil` and
//! `backward_codecs.lucene101.ForUtil`/`ForDeltaUtil`/`PForUtil` (the last
//! shared with `lucene103`).
//!
//! Every one of them packs a block of 128 values the same way
//! [`crate::for_util`] packs 256 for `Lucene104`: the values are first
//! *collapsed* into `word / primitive` lanes of `primitive` bits each
//! (`collapse8`/`collapse16`/`collapse32`), then bit-packed with that
//! primitive size by `ForUtil.encode(.., primitiveSize, ..)`. What differs
//! between generations is only
//!
//! - the **word** the lanes live in and the file stores: a little-endian
//!   `long` for `lucene90`/`lucene99`/`lucene912` ([`Word::Long`]), a
//!   little-endian `int` for `lucene101`/`lucene103` ([`Word::Int`]);
//! - which **primitive size** a bit width is packed with: `ForUtil.encode`'s
//!   `<= 8` / `<= 16` / else, and the doc-delta encoders' own narrower
//!   thresholds (`lucene912.ForDeltaUtil.encodeDeltas`: `<= 4` / `<= 11`;
//!   `lucene101.ForDeltaUtil.encodeDeltas`: `<= 3` / `<= 10`) -- see
//!   [`Primitives`];
//! - `PForUtil`'s constant-block value: a `vlong` for the long-word
//!   generations, a `vint` for the int-word ones.
//!
//! Java decodes each bit width with an unrolled `decodeN` specialisation of
//! that one layout and falls back to `decodeSlow` for the rest. This port
//! decodes every width with `decodeSlow`'s algorithm ([`for_decode`]), the
//! exact inverse of `ForUtil.encode`, then undoes the collapse: the unrolled
//! variants are optimisations of the same bits, and these formats are read,
//! never written, so the port takes the one path that covers all of them.

use lucene_store::data_input::DataInput;
use lucene_store::Result;

/// `ForUtil.BLOCK_SIZE` of every retired generation.
pub const BLOCK_SIZE: usize = 128;

/// The word a generation's `ForUtil` packs into and the file stores.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Word {
    /// `long[]`, written with `writeLong` (`lucene90`, `lucene99`, `lucene912`).
    Long,
    /// `int[]`, written with `writeInt` (`lucene101`, `lucene103`).
    Int,
}

impl Word {
    fn bits(self) -> u32 {
        match self {
            Word::Long => 64,
            Word::Int => 32,
        }
    }
}

/// The largest bit widths packed with an 8- and a 16-bit primitive; wider
/// ones use 32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Primitives {
    pub max8: u32,
    pub max16: u32,
}

/// `ForUtil.encode`: `<= 8` -> 8, `<= 16` -> 16, else 32 -- every generation's
/// plain (non-delta) blocks: frequencies, positions, offsets, payload
/// lengths, and `lucene90`/`lucene99` doc deltas.
pub const FOR_UTIL: Primitives = Primitives { max8: 8, max16: 16 };
/// `lucene912.ForDeltaUtil.encodeDeltas`.
pub const LUCENE912_DELTAS: Primitives = Primitives { max8: 4, max16: 11 };
/// `lucene101.ForDeltaUtil.encodeDeltas` (and `lucene103`'s).
pub const LUCENE101_DELTAS: Primitives = Primitives { max8: 3, max16: 10 };

impl Primitives {
    fn for_bits(self, bits_per_value: u32) -> u32 {
        if bits_per_value <= self.max8 {
            8
        } else if bits_per_value <= self.max16 {
            16
        } else {
            32
        }
    }
}

/// `maskN(bits)` replicated into every `primitive`-bit lane of a `word`-bit
/// value (`expandMask8`/`expandMask16`/`expandMask32`).
// ARITH: `bits < 64` is checked before the shift, and `shift` steps by
// `primitive` (8, 16 or 32) only while below `word_bits <= 64`.
#[allow(clippy::arithmetic_side_effects)]
fn lane_mask(bits: u32, primitive: u32, word_bits: u32) -> u64 {
    let one = if bits >= 64 {
        u64::MAX
    } else {
        (1u64 << bits).wrapping_sub(1)
    };
    let mut m = 0u64;
    let mut shift = 0u32;
    while shift < word_bits {
        m |= one << shift;
        shift += primitive;
    }
    m
}

/// `ForUtil.numBytes(bitsPerValue)` for a 128-value block: `bitsPerValue *
/// 128 / 8`, the same for either word size.
pub fn num_bytes(bits_per_value: u32) -> usize {
    // ARITH: `bits_per_value` is a 5-bit token field (at most 31), so the
    // product is at most 496.
    #[allow(clippy::arithmetic_side_effects)]
    let n = bits_per_value as usize * BLOCK_SIZE / 8;
    n
}

fn corrupt(msg: String) -> lucene_store::Error {
    lucene_store::Error::Corrupted(msg)
}

/// `1` in the low bit of every `primitive`-bit lane of a `word_bits`-bit
/// value: multiplying a `bits <= primitive` mask by it replicates the mask
/// into every lane without carries -- [`lane_mask`] in one multiply.
fn lane_ones(primitive: u32, word_bits: u32) -> u64 {
    match (primitive, word_bits) {
        (8, 64) => 0x0101_0101_0101_0101,
        (16, 64) => 0x0001_0001_0001_0001,
        (32, 64) => 0x0000_0001_0000_0001,
        (8, 32) => 0x0101_0101,
        (16, 32) => 0x0001_0001,
        _ => 1,
    }
}

/// Decodes one block of 128 values packed with `bits_per_value` bits each,
/// in `word`-sized words, using the primitive size `primitives` assigns that
/// width: `ForUtil.decode` (`decodeSlow` plus the matching `expandN`).
///
/// The packed words are read in one call (`in.readLongs(tmp, 0, numLongs)`)
/// and every lane mask is one multiply of a precomputed replication constant
/// (`lane_ones`), where the first cut read word by word and rebuilt each
/// mask lane by lane inside the tail loop. Same bits either way; see
/// `for_decode_matches_encode_for_every_width` and the `bwc` fixtures.
pub fn for_decode<R: DataInput>(
    r: &mut R,
    bits_per_value: u32,
    word: Word,
    primitives: Primitives,
    out: &mut [u64; BLOCK_SIZE],
) -> Result<()> {
    if bits_per_value == 0 || bits_per_value > 32 {
        return Err(corrupt(format!(
            "ForUtil: bitsPerValue {bits_per_value} outside 1..=32"
        )));
    }
    let word_bits = word.bits();
    let primitive = primitives.for_bits(bits_per_value);
    // ARITH: `bits_per_value <= 32`, `word_bits` is 32 or 64 and `primitive`
    // 8, 16 or 32, so every product and quotient is a small positive number;
    // `num_words_per_shift = bits_per_value * 128 / word_bits` is at most 128.
    #[allow(clippy::arithmetic_side_effects)]
    let (num_words, num_words_per_shift) = (
        BLOCK_SIZE * primitive as usize / word_bits as usize,
        bits_per_value as usize * BLOCK_SIZE / word_bits as usize,
    );
    // `in.readLongs(tmp, 0, numWordsPerShift)` (or `readInts`): one read of
    // `num_bytes(bits_per_value)` bytes, at most 512.
    let mut bytes = [0u8; 4 * BLOCK_SIZE];
    let packed = &mut bytes[..num_bytes(bits_per_value)];
    r.read_bytes(packed)?;
    let mut tmp = [0u64; BLOCK_SIZE];
    match word {
        Word::Long => {
            for (t, b) in tmp.iter_mut().zip(packed.chunks_exact(8)) {
                *t = u64::from_le_bytes(b.try_into().expect("8 bytes"));
            }
        }
        Word::Int => {
            for (t, b) in tmp.iter_mut().zip(packed.chunks_exact(4)) {
                *t = u64::from(u32::from_le_bytes(b.try_into().expect("4 bytes")));
            }
        }
    }
    let tmp = &tmp[..num_words_per_shift];

    let ones = lane_ones(primitive, word_bits);
    // `maskN(bits)` in every lane, for `bits <= primitive`.
    // ARITH: `bits <= primitive <= 32`, so `1 << bits` fits a `u64`, and a
    // lane's mask times the lane-ones constant cannot carry across lanes.
    #[allow(clippy::arithmetic_side_effects)]
    let mask = |bits: u32| ((1u64 << bits) - 1) * ones;

    // `decodeSlow`, over `collapsed` words of `word_bits / primitive` lanes.
    let mut collapsed = [0u64; BLOCK_SIZE];
    let value_mask = mask(bits_per_value);
    let mut idx = 0usize;
    // ARITH: `shift` starts at `primitive - bits_per_value >= 0` and steps
    // down by `bits_per_value` while it stays non-negative; `idx` advances by
    // `num_words_per_shift` per step, at most `primitive / bits_per_value`
    // steps, which is exactly `num_words` values in all -- see `ForUtil.encode`.
    #[allow(clippy::arithmetic_side_effects)]
    let remaining_bits_per_word = {
        let mut shift = primitive as i32 - bits_per_value as i32;
        while shift >= 0 {
            for (c, t) in collapsed[idx..idx + num_words_per_shift]
                .iter_mut()
                .zip(tmp)
            {
                *c = (t >> shift) & value_mask;
            }
            idx += num_words_per_shift;
            shift -= bits_per_value as i32;
        }
        (shift + bits_per_value as i32) as u32
    };

    // The bits left in each word after the whole shifts: every value still
    // missing is spread over the low `remaining_bits_per_word` of successive
    // words, most significant part first.
    // ARITH: `ForUtil.encode`'s tail loop mirrored step for step: `rbv` is in
    // `1..=bits_per_value`, `remaining_bits_per_word < bits_per_value` whenever
    // this loop runs, so `rbw - rbv` is only taken when `rbv < rbw`;
    // `tmp_idx < num_words_per_shift` because the encoder filled exactly that
    // many words.
    #[allow(clippy::arithmetic_side_effects)]
    {
        if idx < num_words {
            let rbw = remaining_bits_per_word;
            let rbw_mask = mask(rbw);
            let mut tmp_idx = 0usize;
            let mut rbv = bits_per_value;
            while idx < num_words {
                let Some(&t) = tmp.get(tmp_idx) else {
                    return Err(corrupt("ForUtil: packed block ran out of words".into()));
                };
                if rbv >= rbw {
                    rbv -= rbw;
                    collapsed[idx] |= (t & rbw_mask) << rbv;
                    tmp_idx += 1;
                    if rbv == 0 {
                        idx += 1;
                        rbv = bits_per_value;
                    }
                } else {
                    collapsed[idx] |= (t >> (rbw - rbv)) & mask(rbv);
                    idx += 1;
                    let next = bits_per_value - rbw + rbv;
                    collapsed[idx] |= (t & mask(rbw - rbv)) << next;
                    tmp_idx += 1;
                    rbv = next;
                }
            }
        }
    }

    // `expand8`/`expand16`/`expand32`: lane `k` (most significant first) of
    // collapsed word `i` is value `k * num_words + i`.
    let prim_mask = lane_mask(primitive, word_bits, word_bits);
    // ARITH: `primitive` is 8, 16 or 32, never 0; `k < lanes = word_bits /
    // primitive`, so `primitive * (k + 1) <= word_bits` and the shift is in
    // `0..word_bits`; `k * num_words + i < lanes * num_words = 128`.
    #[allow(clippy::arithmetic_side_effects)]
    for (k, lane) in out.chunks_exact_mut(num_words).enumerate() {
        let shift = word_bits - primitive * (k as u32 + 1);
        for (o, c) in lane.iter_mut().zip(&collapsed[..num_words]) {
            *o = (c >> shift) & prim_mask;
        }
    }
    Ok(())
}

/// `PForUtil.decode`: a token byte (`bitsPerValue | numExceptions << 5`),
/// then either one constant (`bitsPerValue == 0`: a `vlong` for [`Word::Long`]
/// generations, a `vint` for [`Word::Int`]) or a [`for_decode`] block, then
/// `numExceptions` `(index, high bits)` byte pairs patched on top.
// ARITH: `token >> 5` of a byte is at most 7, and `bits_per_value` is its
// low five bits, so the exception's `<< bits_per_value` stays under 2^39.
#[allow(clippy::arithmetic_side_effects)]
pub fn pfor_decode<R: DataInput>(r: &mut R, word: Word, out: &mut [u64; BLOCK_SIZE]) -> Result<()> {
    let token = u32::from(r.read_byte()?);
    let bits_per_value = token & 0x1f;
    if bits_per_value == 0 {
        let v = match word {
            Word::Long => r.read_vlong()? as u64,
            Word::Int => u64::from(r.read_vint()? as u32),
        };
        out.fill(v);
    } else {
        for_decode(r, bits_per_value, word, FOR_UTIL, out)?;
    }
    let num_exceptions = token >> 5;
    for _ in 0..num_exceptions {
        let idx = r.read_byte()? as usize;
        let high = u64::from(r.read_byte()?);
        // `idx` is a byte, and `idx < 256`, but the block holds only 128:
        // Java throws `ArrayIndexOutOfBoundsException` on the same byte.
        let slot = out
            .get_mut(idx)
            .ok_or_else(|| corrupt(format!("PForUtil: exception index {idx} >= 128")))?;
        *slot |= high << bits_per_value;
    }
    Ok(())
}

/// `PForUtil.skip`: steps over one [`pfor_decode`] block without unpacking it.
pub fn pfor_skip<R: DataInput>(r: &mut R, word: Word) -> Result<()> {
    let token = u32::from(r.read_byte()?);
    let bits_per_value = token & 0x1f;
    // ARITH: `token >> 5` is at most 7.
    #[allow(clippy::arithmetic_side_effects)]
    let exception_bytes = (token >> 5) as usize * 2;
    if bits_per_value == 0 {
        match word {
            Word::Long => {
                r.read_vlong()?;
            }
            Word::Int => {
                r.read_vint()?;
            }
        }
        r.skip(exception_bytes)
    } else {
        r.skip(num_bytes(bits_per_value).saturating_add(exception_bytes))
    }
}

/// In-place prefix sum from `base`: `ForDeltaUtil.decodeAndPrefixSum`'s
/// second half (and `lucene90.PForUtil.decodeAndPrefixSum`'s). Wraps like
/// Java's `long`/`int` arithmetic on a corrupt block rather than trapping.
pub fn prefix_sum(values: &mut [u64; BLOCK_SIZE], base: i64) {
    let mut sum = base as u64;
    for v in values.iter_mut() {
        sum = sum.wrapping_add(*v);
        *v = sum;
    }
}

#[cfg(test)]
mod tests {
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;
    use lucene_store::data_input::SliceInput;

    /// `ForUtil.encode` for 128 values in `word`-sized words, written the
    /// Java way (collapse, shift-pack, spill the remainder), so the decoder
    /// can be checked against the encoder it inverts for every width.
    fn encode(values: &[u64; BLOCK_SIZE], bpv: u32, word: Word, prims: Primitives) -> Vec<u8> {
        let word_bits = word.bits();
        let primitive = prims.for_bits(bpv);
        let lanes = (word_bits / primitive) as usize;
        let num_words = BLOCK_SIZE / lanes;
        let mut collapsed = vec![0u64; num_words];
        for (i, c) in collapsed.iter_mut().enumerate() {
            for k in 0..lanes {
                let shift = word_bits - primitive * (k as u32 + 1);
                *c |= values[k * num_words + i] << shift;
            }
        }
        let nwps = bpv as usize * BLOCK_SIZE / word_bits as usize;
        let mut tmp = vec![0u64; nwps];
        let mut idx = 0;
        let mut shift = primitive as i32 - bpv as i32;
        for t in tmp.iter_mut() {
            *t = collapsed[idx] << shift;
            idx += 1;
        }
        shift -= bpv as i32;
        while shift >= 0 {
            for t in tmp.iter_mut() {
                *t |= collapsed[idx] << shift;
                idx += 1;
            }
            shift -= bpv as i32;
        }
        let rbw = (shift + bpv as i32) as u32;
        let mut tmp_idx = 0;
        let mut rbv = bpv;
        while idx < num_words {
            if rbv >= rbw {
                rbv -= rbw;
                tmp[tmp_idx] |= (collapsed[idx] >> rbv) & lane_mask(rbw, primitive, word_bits);
                tmp_idx += 1;
                if rbv == 0 {
                    idx += 1;
                    rbv = bpv;
                }
            } else {
                tmp[tmp_idx] |=
                    (collapsed[idx] & lane_mask(rbv, primitive, word_bits)) << (rbw - rbv);
                idx += 1;
                let old = rbv;
                rbv += bpv - rbw;
                tmp[tmp_idx] |=
                    (collapsed[idx] >> rbv) & lane_mask(rbw - old, primitive, word_bits);
                tmp_idx += 1;
            }
        }
        let mut out = Vec::new();
        for t in tmp {
            match word {
                Word::Long => out.extend_from_slice(&t.to_le_bytes()),
                Word::Int => out.extend_from_slice(&(t as u32).to_le_bytes()),
            }
        }
        out
    }

    fn values(bpv: u32, seed: u64) -> [u64; BLOCK_SIZE] {
        let mut v = [0u64; BLOCK_SIZE];
        let mut s = seed;
        let mask = if bpv == 64 {
            u64::MAX
        } else {
            (1u64 << bpv) - 1
        };
        for x in v.iter_mut() {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *x = (s >> 11) & mask;
        }
        v[7] = mask; // every width actually reaches its top bit
        v
    }

    #[test]
    fn decode_inverts_encode_for_every_width_word_and_primitive_choice() {
        for word in [Word::Long, Word::Int] {
            for prims in [FOR_UTIL, LUCENE912_DELTAS, LUCENE101_DELTAS] {
                for bpv in 1..=32 {
                    let v = values(bpv, u64::from(bpv) * 31 + 7);
                    let bytes = encode(&v, bpv, word, prims);
                    assert_eq!(bytes.len(), num_bytes(bpv), "bpv={bpv}");
                    let mut r = SliceInput::new(&bytes);
                    let mut out = [0u64; BLOCK_SIZE];
                    for_decode(&mut r, bpv, word, prims, &mut out).unwrap();
                    assert_eq!(out, v, "word={word:?} prims={prims:?} bpv={bpv}");
                    assert_eq!(r.position(), bytes.len());
                }
            }
        }
    }

    /// A token can only carry 1..=31 bits; anything else is not a block.
    #[test]
    fn bits_outside_range_are_rejected() {
        let mut out = [0u64; BLOCK_SIZE];
        let mut r = SliceInput::new(&[]);
        assert!(for_decode(&mut r, 0, Word::Long, FOR_UTIL, &mut out).is_err());
        assert!(for_decode(&mut r, 33, Word::Int, FOR_UTIL, &mut out).is_err());
    }

    #[test]
    fn pfor_constant_block_and_exceptions() {
        // bpv 0, 1 exception: vlong 5 then (idx 3, high 2) -> 5 | 2 << 0.
        let bytes = [1u8 << 5, 5, 3, 2];
        let mut r = SliceInput::new(&bytes);
        let mut out = [0u64; BLOCK_SIZE];
        pfor_decode(&mut r, Word::Long, &mut out).unwrap();
        assert_eq!(out[0], 5);
        assert_eq!(out[3], 7);
        let mut r = SliceInput::new(&bytes);
        pfor_skip(&mut r, Word::Long).unwrap();
        assert_eq!(r.position(), 4);
        let mut r = SliceInput::new(&bytes);
        pfor_skip(&mut r, Word::Int).unwrap();
        assert_eq!(r.position(), 4);
        let mut r = SliceInput::new(&bytes);
        pfor_decode(&mut r, Word::Int, &mut out).unwrap();
        assert_eq!(out[3], 7);

        // bpv 2 with an exception index past the block.
        let mut bytes = vec![2u8 | (1 << 5)];
        bytes.extend(std::iter::repeat_n(0, num_bytes(2)));
        bytes.extend([200, 1]);
        let mut r = SliceInput::new(&bytes);
        assert!(pfor_decode(&mut r, Word::Long, &mut out).is_err());
        let mut r = SliceInput::new(&bytes);
        pfor_skip(&mut r, Word::Long).unwrap();
        assert_eq!(r.position(), bytes.len());
    }

    #[test]
    fn prefix_sum_accumulates_from_base() {
        let mut v = [1u64; BLOCK_SIZE];
        prefix_sum(&mut v, 9);
        assert_eq!(v[0], 10);
        assert_eq!(v[127], 137);
    }
}
