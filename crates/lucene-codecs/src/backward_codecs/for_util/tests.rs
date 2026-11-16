#![cfg(test)]
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
            tmp[tmp_idx] |= (collapsed[idx] & lane_mask(rbv, primitive, word_bits)) << (rbw - rbv);
            idx += 1;
            let old = rbv;
            rbv += bpv - rbw;
            tmp[tmp_idx] |= (collapsed[idx] >> rbv) & lane_mask(rbw - old, primitive, word_bits);
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
