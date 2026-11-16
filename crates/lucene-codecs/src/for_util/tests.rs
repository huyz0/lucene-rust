#![cfg(test)]
// The gate is about values read off disk; a test's `i + 1` is not one.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_store::data_input::SliceInput;
use lucene_store::data_output::DataOutput;

#[test]
fn for_decode_roundtrips_bits_1_to_16() {
    for bits in 1u32..=16 {
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(2654435761) ^ (i as u32).rotate_left(3)) & mask32(bits);
        }
        let mut bytes = Vec::new();
        for_encode(&mut values.clone(), bits, &mut bytes);
        assert_eq!(bytes.len(), num_bytes(bits), "bits_per_value={bits}");

        let mut r = SliceInput::new(&bytes);
        let mut decoded = [0u32; BLOCK_SIZE];
        for_decode(bits, &mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values, "bits_per_value={bits}");
    }
}

#[test]
fn decode_slow_roundtrips_bits_17_to_32() {
    for bits in [17u32, 20, 24, 28, 31, 32] {
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            // Deterministic pseudo-random-ish pattern within [0, 2^bits).
            *v = ((i as u32).wrapping_mul(2654435761) ^ (i as u32)) & mask32(bits);
        }
        let mut bytes = Vec::new();
        for_encode(&mut values.clone(), bits, &mut bytes);
        assert_eq!(bytes.len(), num_bytes(bits));

        let mut r = SliceInput::new(&bytes);
        let mut decoded = [0u32; BLOCK_SIZE];
        for_decode(bits, &mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values, "bits_per_value={bits}");
    }
}

#[test]
fn mask32_boundary_values() {
    assert_eq!(mask32(0), 0);
    assert_eq!(mask32(1), 1);
    assert_eq!(mask32(31), (1u32 << 31) - 1);
    assert_eq!(mask32(32), u32::MAX);
}

#[test]
fn pfor_decode_all_equal_uses_vint_fast_path() {
    // token byte with bitsPerValue=0 and numExceptions=0, then a plain
    // vint carrying the single repeated value.
    let mut bytes = vec![0u8];
    bytes.write_vint(42);
    let mut r = SliceInput::new(&bytes);
    let mut ints = [0u32; BLOCK_SIZE];
    pfor_decode(&mut r, &mut ints).unwrap();
    assert!(ints.iter().all(|&v| v == 42));
}

#[test]
fn pfor_skip_consumes_exactly_what_pfor_decode_does() {
    // `PForUtil.skip` has to land on the same byte `PForUtil.decode`
    // would, for every shape of block: the all-equal fast path, a packed
    // body, and either with exceptions. A skip that is off by one byte
    // corrupts every block after it, silently.
    let shapes: Vec<Vec<u32>> = vec![
        // all-equal, no exceptions
        vec![7; BLOCK_SIZE],
        // all-equal with two exceptions
        {
            let mut v = vec![7u32; BLOCK_SIZE];
            v[10] = 7 | (3 << 8);
            v[200] = 7 | (1 << 8);
            v
        },
        // a packed body, no exceptions
        (0..BLOCK_SIZE).map(|i| (i % 13) as u32).collect(),
        // a packed body with exceptions (a handful of outliers)
        {
            let mut v: Vec<u32> = (0..BLOCK_SIZE).map(|i| (i % 13) as u32).collect();
            v[3] = 100_000;
            v[250] = 70_000;
            v
        },
        // maximum width `pfor_encode` supports (31 bits)
        (0..BLOCK_SIZE)
            .map(|i| (u32::MAX >> 1) - i as u32)
            .collect(),
    ];
    for (shape_idx, values) in shapes.into_iter().enumerate() {
        let mut bytes = Vec::new();
        let mut ints = [0u32; BLOCK_SIZE];
        ints.copy_from_slice(&values);
        pfor_encode(&mut ints, &mut bytes);
        // Trailing sentinel so a skip that overshoots is visible.
        bytes.extend_from_slice(&[0xAB, 0xCD]);

        let mut decode_in = SliceInput::new(&bytes);
        let mut out = [0u32; BLOCK_SIZE];
        ForUtil::new()
            .pfor_decode(&mut decode_in, &mut out)
            .unwrap();
        assert_eq!(out.to_vec(), values, "shape {shape_idx} round trip");

        let mut skip_in = SliceInput::new(&bytes);
        pfor_skip(&mut skip_in).unwrap();
        assert_eq!(
            skip_in.position(),
            decode_in.position(),
            "shape {shape_idx}: skip and decode must consume the same bytes"
        );
    }
}

#[test]
fn pfor_skip_reports_eof_rather_than_running_off_the_end() {
    let bytes = [1u8, 2, 3];
    let mut r = SliceInput::new(&bytes);
    assert!(pfor_skip(&mut r).is_err());
}

#[test]
fn pfor_decode_all_equal_with_exceptions() {
    // bitsPerValue=0 (all-equal base value), but 2 exceptions patch
    // specific slots to larger values via high bytes shifted by 0 bits.
    let num_exceptions = 2u8;
    let mut bytes = vec![num_exceptions << 5];
    bytes.write_vint(5);
    bytes.push(10); // exception at index 10
    bytes.push(3); // patch byte: ints[10] |= 3 << 0
    bytes.push(200); // exception at index 200
    bytes.push(1); // ints[200] |= 1 << 0
    let mut r = SliceInput::new(&bytes);
    let mut ints = [0u32; BLOCK_SIZE];
    pfor_decode(&mut r, &mut ints).unwrap();
    for (i, &v) in ints.iter().enumerate() {
        match i {
            10 => assert_eq!(v, 5 | 3),
            200 => assert_eq!(v, 5 | 1),
            _ => assert_eq!(v, 5),
        }
    }
}

#[test]
fn for_decode_bits_per_value_one_all_zero() {
    // 8 zero words -> 64 zero collapsed ints -> expand8 -> all 256 zero.
    let bytes = vec![0u8; num_bytes(1)];
    let mut r = SliceInput::new(&bytes);
    let mut ints = [0u32; BLOCK_SIZE];
    for_decode(1, &mut r, &mut ints).unwrap();
    assert!(ints.iter().all(|&v| v == 0));
}

#[test]
fn num_bytes_matches_bit_width() {
    assert_eq!(num_bytes(1), 32);
    assert_eq!(num_bytes(8), 256);
    assert_eq!(num_bytes(16), 512);
    assert_eq!(num_bytes(32), 1024);
}

// --- pfor_encode/for_encode round-trip tests -----------------------
//
// This repo's `BLOCK_SIZE` is 256 (real Lucene 10.5.0's
// `Lucene104PostingsFormat`/`ForUtil.BLOCK_SIZE`) -- these boundary tests exercise exactly
// one full block (256 entries) plus the module's own decode-side
// boundary cases (all-zero / full-32-bit) that a real-Lucene fixture
// term won't reliably hit. The 127/129-style "one packed block + a
// vint tail" boundary from the original task statement is a
// `postings_writer`-level concern (choosing how many full blocks vs. a
// tail to emit for a given docFreq); wiring `pfor_encode` into that
// writer is out of scope here -- see this module's doc comment and
// `docs/parity.md`.

fn bits_required_for_test(v: u32) -> u32 {
    bits_required(v)
}

#[test]
fn pfor_roundtrip_exactly_256_entries_no_exceptions() {
    // Every value fits in the same bit width -> patched_bits_required ==
    // max_bits_required, num_exceptions == 0.
    let mut values = [0u32; BLOCK_SIZE];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as u32) & 0xFF; // fits in 8 bits
    }
    let mut ints = values;
    let mut bytes = Vec::new();
    pfor_encode(&mut ints, &mut bytes);

    let mut r = SliceInput::new(&bytes);
    let mut decoded = [0u32; BLOCK_SIZE];
    pfor_decode(&mut r, &mut decoded).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn pfor_roundtrip_all_zero_needs_zero_bits() {
    let values = [0u32; BLOCK_SIZE];
    let mut ints = values;
    let mut bytes = Vec::new();
    pfor_encode(&mut ints, &mut bytes);
    // All-equal (0) with maxBitsRequired == 0 <= 8 takes the vint
    // fast path: 1 token byte + 1 vint byte for value 0.
    assert_eq!(bytes.len(), 2);

    let mut r = SliceInput::new(&bytes);
    let mut decoded = [1u32; BLOCK_SIZE];
    pfor_decode(&mut r, &mut decoded).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn pfor_roundtrip_full_31_bits_required() {
    // `pfor_encode`'s 1-byte token packs `numExceptions` into the top 3
    // bits and `bitsPerValue` into the bottom 5 (`token & 0x1f`), so
    // `bitsPerValue` can only ever range `0..=31` -- exactly matching the
    // real domain (Lucene doc deltas/freqs are non-negative Java `int`s,
    // so `bitsRequired` never exceeds 31 in practice; a `bitsPerValue`
    // of 32 would alias to the token's `0` == "all-equal" marker and is
    // out of scope here, just like it is for the real `PForUtil`).
    let mut values = [0u32; BLOCK_SIZE];
    for (i, v) in values.iter_mut().enumerate() {
        // A couple of entries pinned to the largest representable value
        // so max_bits_required == 31 deterministically, plus varied
        // filler so it isn't all-equal.
        *v = if i % 7 == 0 {
            0x7FFF_FFFF
        } else {
            (i as u32).wrapping_mul(2654435761) & 0x7FFF_FFFF
        };
    }
    assert_eq!(bits_required_for_test(0x7FFF_FFFF), 31);

    let mut ints = values;
    let mut bytes = Vec::new();
    pfor_encode(&mut ints, &mut bytes);

    let mut r = SliceInput::new(&bytes);
    let mut decoded = [0u32; BLOCK_SIZE];
    pfor_decode(&mut r, &mut decoded).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn pfor_roundtrip_few_outliers_use_patched_exceptions() {
    // Almost every value fits in 1 bit; a handful of outliers need ~21
    // bits. `pfor_encode` patches those outliers as exceptions rather
    // than paying 21 bits for all 256 entries -- but the patch byte can
    // only absorb up to 8 bits of reduction (`minBits = maxBits - 8`),
    // so the packed body still costs `patched_bits_required = 13` bits
    // per entry here, not the theoretical minimum of 1 bit. That
    // 8-bit-reduction ceiling is a real `PForUtil` constraint (the patch
    // is stored in a single byte), not a bug in this port.
    let mut values = [1u32; BLOCK_SIZE];
    let outlier_indices = [3usize, 17, 100, 255];
    for &i in &outlier_indices {
        values[i] = 0x000F_FFF0 + i as u32; // needs 20 or 21 bits
    }
    let mut ints = values;
    let mut bytes = Vec::new();
    pfor_encode(&mut ints, &mut bytes);
    // 1 token byte + a 13-bit-per-value packed body (num_bytes(13)) + 2
    // bytes (index, patch) per exception.
    let expected_len = 1 + num_bytes(13) + outlier_indices.len() * 2;
    assert_eq!(bytes.len(), expected_len);
    assert!(
        bytes.len() < 1 + num_bytes(21),
        "patched encoding ({} bytes) should still beat a plain 21-bit body ({} bytes)",
        bytes.len(),
        1 + num_bytes(21)
    );

    let mut r = SliceInput::new(&bytes);
    let mut decoded = [0u32; BLOCK_SIZE];
    pfor_decode(&mut r, &mut decoded).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn decode_rejects_bits_per_value_outside_the_supported_range() {
    // A corrupt `.doc` block header can carry any signed byte; every
    // out-of-range width must be an error rather than an index panic.
    let bytes = vec![0u8; 1024];
    for bits in [0u32, 33, 64, 127] {
        let mut r = SliceInput::new(&bytes);
        let mut ints = [0u32; BLOCK_SIZE];
        assert!(
            ForUtil::new().decode(bits, &mut r, &mut ints).is_err(),
            "bits_per_value={bits} must be rejected"
        );
    }
}

#[test]
fn bits_required_never_returns_zero_like_javas_packed_ints() {
    // `PackedInts.unsignedBitsRequired` is `Math.max(1, 32 - nlz)`.
    assert_eq!(bits_required(0), 1);
    assert_eq!(bits_required(1), 1);
    assert_eq!(bits_required(2), 2);
    assert_eq!(bits_required(0x7FFF_FFFF), 31);
    assert_eq!(bits_required(0xFFFF_FFFF), 32);
}

#[test]
fn pfor_encode_mostly_zero_block_picks_javas_bit_width() {
    // 254 zeros, a single 1 and a single 200: the shape where a
    // `bitsRequired(0) == 0` floor would let the width search reach
    // `b == 0` and take the all-equal/vint path, where real `PForUtil`
    // stops at `b == 1` and emits a 1-bit packed body plus one exception.
    let mut values = [0u32; BLOCK_SIZE];
    values[100] = 1;
    values[200] = 200;
    let mut ints = values;
    let mut bytes = Vec::new();
    pfor_encode(&mut ints, &mut bytes);

    // token = (numExceptions << 5) | patchedBitsRequired = (1 << 5) | 1
    assert_eq!(bytes[0], 33);
    assert_eq!(bytes.len(), 1 + num_bytes(1) + 2);
    // The exception is index 200 with high bits 200 >> 1 == 100.
    assert_eq!(&bytes[bytes.len() - 2..], &[200u8, 100u8]);

    let mut r = SliceInput::new(&bytes);
    let mut decoded = [0u32; BLOCK_SIZE];
    pfor_decode(&mut r, &mut decoded).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn pfor_encode_matches_for_decode_directly() {
    // pfor_encode with bitsPerValue > 0 and no exceptions must be
    // readable by the lower-level for_decode once the 1-byte token is
    // skipped, proving pfor_encode's packed body is byte-identical to
    // for_encode's.
    let mut values = [0u32; BLOCK_SIZE];
    for (i, v) in values.iter_mut().enumerate() {
        *v = (i as u32) & 0x1F; // 5 bits, no exceptions
    }
    let mut ints = values;
    let mut bytes = Vec::new();
    pfor_encode(&mut ints, &mut bytes);

    let mut r = SliceInput::new(&bytes);
    let token = r.read_byte().unwrap();
    assert_eq!(token, 5); // bitsPerValue=5, numExceptions=0
    let mut decoded = [0u32; BLOCK_SIZE];
    for_decode(5, &mut r, &mut decoded).unwrap();
    assert_eq!(decoded, values);
}

#[test]
#[should_panic(expected = "bitsPerValue out of range")]
fn encode_refuses_a_zero_bit_width_instead_of_spinning_forever() {
    // Java's `ForUtil.encode` loops `for (shift -= bitsPerValue; shift >=
    // 0; shift -= bitsPerValue)`: at `bitsPerValue == 0` the induction
    // variable never moves and the encoder hangs. A hang is the one
    // failure `catch_unwind` at the FFI boundary cannot convert back into
    // a Java exception, so this port refuses the width. Without the
    // `assert!` in `ForUtil::encode` this test does not fail — it never
    // returns at all.
    let mut values = [1u32; BLOCK_SIZE];
    let mut bytes = Vec::new();
    for_encode(&mut values, 0, &mut bytes);
}

#[test]
#[should_panic(expected = "bitsPerValue out of range")]
fn encode_refuses_a_width_above_32() {
    // Above 32 the very first `ints[idx] << shift` shifts by a negative
    // amount; Java walks off `MASKS32` instead. Either way the caller
    // deserves the width in the message, not a bare shift panic.
    let mut values = [1u32; BLOCK_SIZE];
    let mut bytes = Vec::new();
    for_encode(&mut values, 33, &mut bytes);
}

#[test]
fn every_token_byte_decodes_or_errors_but_never_panics() {
    // A corrupt `.doc`/`.pos` block header is a single arbitrary byte:
    // `bitsPerValue = token & 0x1f` steers the body and `numExceptions =
    // token >> 5` steers the patch loop, whose index byte then indexes
    // `ints` directly. Sweep all 256 tokens against a body long enough
    // for any width (1 KiB packed body + 7 patch pairs) and against a
    // truncated one, and require every outcome to be `Ok` or a typed
    // error.
    for token in 0u16..=255 {
        for body in [0x00u8, 0xFF, 0xA5] {
            let mut bytes = vec![token as u8];
            bytes.extend(std::iter::repeat_n(body, num_bytes(32) + 2 * 7));
            let full_len = bytes.len();
            for len in [1usize, 2, 33, full_len / 2, full_len] {
                let mut r = SliceInput::new(&bytes[..len]);
                let mut ints = [0u32; BLOCK_SIZE];
                let _ = ForUtil::new().pfor_decode(&mut r, &mut ints);
                let mut r = SliceInput::new(&bytes[..len]);
                let _ = pfor_skip(&mut r);
            }
        }
    }
}

#[test]
fn decode_sweeps_every_width_over_an_all_ones_body() {
    // `decode_slow`'s scratch bookkeeping (`tmp_idx`) and `split_ints`'
    // shift/offset bookkeeping are pinned by `debug_assert!`s; run every
    // legal width over a saturated body so a debug test run actually
    // exercises those claims rather than trusting the comment.
    let bytes = vec![0xFFu8; num_bytes(32)];
    for bits in 1u32..=32 {
        let mut r = SliceInput::new(&bytes);
        let mut ints = [0u32; BLOCK_SIZE];
        ForUtil::new().decode(bits, &mut r, &mut ints).unwrap();
        let limit = mask32(bits);
        assert!(
            ints.iter().all(|&v| v <= limit),
            "bits={bits} decoded a value wider than the declared width"
        );
    }
}

#[test]
fn all_widths_round_trip_including_the_encode_slow_boundary() {
    // Widths 1..=32 in one sweep, with a value pattern that fills the
    // full width. This is what pins the `chunks_exact` rewrite of
    // `decode3..decode15` and `expand8`/`expand16` against the previous
    // explicit-index form: a wrong stride or a wrong lane shows up as a
    // mismatched value here for the width that uses it.
    for bits in 1u32..=32 {
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(2654435761) ^ (i as u32).rotate_left(11)) & mask32(bits);
        }
        // Force both extremes of the width into the block.
        values[0] = 0;
        values[BLOCK_SIZE - 1] = mask32(bits);
        let mut bytes = Vec::new();
        for_encode(&mut values.clone(), bits, &mut bytes);
        assert_eq!(bytes.len(), num_bytes(bits), "bits={bits}");
        let mut r = SliceInput::new(&bytes);
        let mut decoded = [0u32; BLOCK_SIZE];
        for_decode(bits, &mut r, &mut decoded).unwrap();
        assert_eq!(decoded, values, "bits={bits}");
    }
}

#[test]
fn pfor_encode_all_equal_path_with_a_patch_shift() {
    // The one shape that exercises `patch << patched_bits_required` in
    // the all-equal branch: after masking, every value is equal, and the
    // patch's high byte has to be pre-shifted because the decoder's
    // exception loop will shift by `bitsPerValue == 0`. The shift is a
    // `u8` shift, so a `patched_bits_required` of 8 would panic — the
    // branch's `debug_assert!` states why it cannot be 8, and this test
    // is the one that runs it.
    let mut values = [3u32; BLOCK_SIZE];
    values[5] = 19;
    values[100] = 19;
    let mut ints = values;
    let mut bytes = Vec::new();
    pfor_encode(&mut ints, &mut bytes);
    // bitsPerValue == 0 (the all-equal marker) with two exceptions.
    assert_eq!(bytes[0], 2 << 5);

    let mut r = SliceInput::new(&bytes);
    let mut decoded = [0u32; BLOCK_SIZE];
    pfor_decode(&mut r, &mut decoded).unwrap();
    assert_eq!(decoded, values);
}

#[test]
fn default_is_a_fresh_decoder() {
    let mut fu = ForUtil::default();
    let bytes = vec![0u8; num_bytes(1)];
    let mut ints = [1u32; BLOCK_SIZE];
    fu.decode(1, &mut SliceInput::new(&bytes), &mut ints)
        .unwrap();
    assert!(ints.iter().all(|&v| v == 0));
}

// --- A/B microbenchmark -------------------------------------------
//
// Criterion is unusable on this machine (the same code measured
// 83/91/129 us across three runs), so this is a min-of-N harness: the
// minimum of N timed batches is the run least disturbed by the scheduler,
// and it is repeatable to within ~0.3% here where the mean is not. This
// is what the "iterators measured slower" claim in `decode3`..`decode15`
// rests on. `#[ignore]`d; run with
//   cargo test -p lucene-codecs --release --lib for_util::tests::bench \
//     -- --ignored --nocapture --test-threads=1
#[test]
#[ignore = "microbenchmark"]
fn bench_decode_and_encode_min_of_n() {
    use std::time::{Duration, Instant};
    fn min_of_n(n: u32, inner: u32, mut f: impl FnMut()) -> Duration {
        let mut best = Duration::from_secs(9999);
        for _ in 0..n {
            let t = Instant::now();
            for _ in 0..inner {
                f();
            }
            best = best.min(t.elapsed());
        }
        best
    }
    for bits in [1u32, 3, 5, 6, 7, 9, 11, 13, 15, 16, 21, 31] {
        let mut values = [0u32; BLOCK_SIZE];
        for (i, v) in values.iter_mut().enumerate() {
            *v = ((i as u32).wrapping_mul(2654435761) ^ (i as u32).rotate_left(7)) & mask32(bits);
        }
        let mut bytes = Vec::new();
        for_encode(&mut values.clone(), bits, &mut bytes);
        let (mut fu, mut ints) = (ForUtil::new(), [0u32; BLOCK_SIZE]);
        let d = min_of_n(60, 4000, || {
            let mut r = SliceInput::new(std::hint::black_box(&bytes));
            fu.decode(bits, &mut r, &mut ints).unwrap();
            std::hint::black_box(&ints);
        });
        let mut out = Vec::with_capacity(num_bytes(bits));
        let e = min_of_n(60, 4000, || {
            let mut scratch = *std::hint::black_box(&values);
            out.clear();
            fu.encode(&mut scratch, bits, &mut out);
        });
        println!("bits={bits:2} decode={d:?} encode={e:?}");
    }
}
