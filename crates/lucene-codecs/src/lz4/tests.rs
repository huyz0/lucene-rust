#![cfg(test)]
// The arithmetic gate is about values read off disk; a test's `i + 1` is
// not one. See docs/arithmetic-gate.md.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use lucene_store::data_input::SliceInput;

/// literal-only block: token=0x30 (literalLen=3, matchLen=0), then "abc".
/// Since matchLen would be read next but dOff reaches destEnd right after
/// the literals, no match bytes follow.
/// An LZ4 length extension is a run of `0xFF` bytes, each adding 255 to
/// the accumulator. A bare `usize +=` would let a long enough run wrap to a
/// small length and decode a silently truncated block (release) or panic
/// (debug). It has to come back as corruption instead, for both the
/// literal and the match halves of the token.
#[test]
fn length_extension_run_is_bounded() {
    // A short, legal extension still decodes: literalLen 15 + 3 = 18.
    let mut compressed = vec![0xF0u8, 3];
    compressed.extend_from_slice(&[b'z'; 18]);
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 18];
    assert_eq!(decompress(&mut input, 18, &mut dest, 0).unwrap(), 18);

    // An extension that runs off the end of the block is an EOF, not a
    // wrapped length.
    let compressed = vec![0xFFu8; 4096];
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 64];
    assert!(decompress(&mut input, 64, &mut dest, 0).is_err());

    // The accumulator itself must be checked: `read_length_extension` is
    // the only place it grows, and it must saturate into an error rather
    // than wrap.
    let huge = vec![0xFFu8; 64];
    let mut input = SliceInput::new(&huge);
    assert!(read_length_extension(&mut input, usize::MAX - 1).is_err());
}

/// Text-like input with runs, short and long repeats at every distance
/// class the fast path distinguishes (`< 8`, `>= 8`, long matches).
fn corpus(seed: u64, len: usize) -> Vec<u8> {
    let mut x = seed | 1;
    let mut next = || {
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        x
    };
    let words: [&[u8]; 8] = [
        b"the ",
        b"quick ",
        b"lucene ",
        b"a",
        b"zzzzzzzzzzzzzzzzzzzz",
        b"ab",
        b"\x00\x01",
        b"xyz ",
    ];
    let mut out = Vec::with_capacity(len);
    while out.len() < len {
        let r = next();
        if r % 5 == 0 {
            out.push((r >> 8) as u8);
        } else {
            out.extend_from_slice(words[(r >> 16) as usize % words.len()]);
        }
    }
    out.truncate(len);
    out
}

/// `decompress_slice` is an optimisation of `decompress`: on every block
/// the Java-compatible compressor emits -- with and without a preset
/// dictionary, at lengths either side of the fast path's 16/24-byte
/// margins -- it must produce the same bytes, the same return value and
/// consume exactly the same input.
#[test]
fn slice_decoder_matches_the_reference_decoder() {
    for seed in 1..40u64 {
        for len in [0usize, 1, 5, 15, 16, 17, 24, 31, 100, 1000, 20_000] {
            let dict_len = (seed as usize * 37) % 300;
            let whole = corpus(seed, dict_len + len);
            let mut compressed = Vec::new();
            compress_with_dictionary(
                &whole,
                0,
                dict_len,
                len,
                &mut compressed,
                &mut FastCompressionHashTable::new(),
            );
            // A trailing sentinel proves both stop at the block's end.
            compressed.push(0xAB);

            let mut want = vec![0u8; dict_len + len];
            want[..dict_len].copy_from_slice(&whole[..dict_len]);
            let mut got = want.clone();
            if len == 0 {
                continue; // nothing is written for an empty block
            }
            let mut a = SliceInput::new(&compressed);
            let mut b = SliceInput::new(&compressed);
            let ra = decompress(&mut a, len, &mut want, dict_len).unwrap();
            let rb = decompress_slice(&mut b, len, &mut got, dict_len).unwrap();
            assert_eq!(ra, rb, "seed {seed} len {len}");
            assert_eq!(want, got, "seed {seed} len {len}");
            assert_eq!(&got[..], &whole[..], "seed {seed} len {len}");
            assert_eq!(a.position(), b.position(), "seed {seed} len {len}");
        }
    }
}

/// Every truncation and every single-byte corruption of a real block must
/// come back as an error or as bytes -- never a panic -- and wherever the
/// reference decoder succeeds, the fast one agrees with it byte for byte.
#[test]
fn slice_decoder_survives_truncation_and_corruption() {
    let plain = corpus(7, 600);
    let compressed = compress(&plain);
    for cut in 0..compressed.len() {
        let mut d1 = vec![0u8; plain.len()];
        let mut d2 = d1.clone();
        let r1 = decompress(
            &mut SliceInput::new(&compressed[..cut]),
            plain.len(),
            &mut d1,
            0,
        );
        let r2 = decompress_slice(
            &mut SliceInput::new(&compressed[..cut]),
            plain.len(),
            &mut d2,
            0,
        );
        assert_eq!(r1.is_ok(), r2.is_ok(), "cut {cut}");
    }
    for at in 0..compressed.len() {
        for flip in [0x01u8, 0x0F, 0xF0, 0xFF] {
            let mut bad = compressed.clone();
            bad[at] ^= flip;
            let mut d1 = vec![0u8; plain.len()];
            let mut d2 = d1.clone();
            let r1 = decompress(&mut SliceInput::new(&bad), plain.len(), &mut d1, 0);
            let r2 = decompress_slice(&mut SliceInput::new(&bad), plain.len(), &mut d2, 0);
            assert_eq!(r1.is_ok(), r2.is_ok(), "at {at} flip {flip:#x}");
            if let (Ok(e1), Ok(e2)) = (r1, r2) {
                assert_eq!(e1, e2);
                // Only the produced range is defined; a wild copy may
                // have scribbled past it on a corrupt block's early stop.
                let n = e1.min(plain.len());
                assert_eq!(d1[..n], d2[..n], "at {at} flip {flip:#x}");
            }
        }
    }
    // A destination too small for the declared length is refused up front.
    let mut tiny = [0u8; 4];
    assert!(decompress_slice(&mut SliceInput::new(&compressed), 600, &mut tiny, 0).is_err());
    assert!(decompress_slice(&mut SliceInput::new(&compressed), usize::MAX, &mut tiny, 1).is_err());
    // An overflowing length extension is corruption, not a wrap.
    let mut s = 0;
    assert!(matches!(
        extension(&[0xFF; 8], &mut s, usize::MAX - 1),
        Err(Bad::Corrupt(_))
    ));
}

#[test]
fn short_matches_at_every_distance_decode() {
    // A block written by hand, so every (distance, length) pair is in it:
    // matches of 4..=18 bytes at distances 1..=40, each behind fresh
    // literals -- the fixed-size 8- and 16-byte moves, the exact copy
    // below distance 8, and their switch points.
    let mut rng = 0x9E37_79B9_7F4A_7C15u64;
    let mut next = || {
        rng ^= rng << 13;
        rng ^= rng >> 7;
        rng ^= rng << 17;
        rng
    };
    let (mut plain, mut block) = (Vec::new(), Vec::new());
    for dist in 1..=40usize {
        for len in 4..=18usize {
            let lit = dist.max(8);
            let literals: Vec<u8> = (0..lit).map(|_| (next() % 251) as u8).collect();
            let lit_nibble = lit.min(15);
            block.push(((lit_nibble << 4) | (len - MIN_MATCH)) as u8);
            if lit >= 15 {
                block.push((lit - 15) as u8);
            }
            block.extend_from_slice(&literals);
            plain.extend_from_slice(&literals);
            block.extend_from_slice(&(dist as u16).to_le_bytes());
            for _ in 0..len {
                plain.push(plain[plain.len() - dist]);
            }
        }
    }
    // The last sequence: literals only.
    block.push(0x50);
    block.extend_from_slice(b"tail!");
    plain.extend_from_slice(b"tail!");
    let mut out = vec![0u8; plain.len()];
    let end = decompress_slice(&mut SliceInput::new(&block), plain.len(), &mut out, 0).unwrap();
    assert_eq!(end, plain.len());
    assert_eq!(out, plain);
    // The exact, `DataInput` decoder agrees.
    let mut exact = vec![0u8; plain.len()];
    decompress(&mut SliceInput::new(&block), plain.len(), &mut exact, 0).unwrap();
    assert_eq!(exact, plain);
}

#[test]
fn literal_only_block() {
    let compressed = [0x30u8, b'a', b'b', b'c'];
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 3];
    let end = decompress(&mut input, 3, &mut dest, 0).unwrap();
    assert_eq!(end, 3);
    assert_eq!(&dest, b"abc");
}

/// "aaaa" then a match copying the first 'a' 4 more times: literals="a"
/// (len1), match_dec=1, match_len_field=0 -> matchLen=4 -> copies "aaaa".
#[test]
fn overlapping_match_copy() {
    // token: literalLen=1 (upper nibble), matchLen field=0 (lower nibble)
    let token = 1u8 << 4;
    let mut compressed = vec![token, b'a'];
    compressed.extend_from_slice(&1u16.to_le_bytes()); // matchDec = 1
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 5]; // 1 literal + 4 match bytes
    let end = decompress(&mut input, 5, &mut dest, 0).unwrap();
    assert_eq!(end, 5);
    assert_eq!(&dest, b"aaaaa");
}

#[test]
fn extended_literal_length_encoding() {
    // literalLen = 0x0F + 0xFF + 5 = 15 + 255 + 5 = 275... too big for a
    // small test; use a smaller extension: 0x0F then one length byte 3
    // -> literalLen = 15 + 3 = 18.
    let token = 0xF0u8; // literalLen nibble = 0x0F (extended), matchLen nibble = 0
    let mut compressed = vec![token, 3u8]; // extension byte: +3 -> 18 literal bytes
    let literal: Vec<u8> = (0..18).map(|i| i as u8).collect();
    compressed.extend_from_slice(&literal);
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 18];
    let end = decompress(&mut input, 18, &mut dest, 0).unwrap();
    assert_eq!(end, 18);
    assert_eq!(dest.as_slice(), literal.as_slice());
}

#[test]
fn preset_dictionary_reference_before_d_off() {
    // dest already has "hello" at [0..5] (the "dictionary"); decompress
    // a match-only block at d_off=5 that copies "hello" via a back-reference.
    let mut dest = *b"hello\0\0\0\0\0";
    // token: literalLen=0, matchLen field = 1 (since MIN_MATCH=4, +1=5 matches "hello"'s length)
    let token = 1u8;
    let mut compressed = vec![token];
    compressed.extend_from_slice(&5u16.to_le_bytes()); // matchDec=5, refers to dest[0]
    let mut input = SliceInput::new(&compressed);
    let end = decompress(&mut input, 5, &mut dest, 5).unwrap();
    assert_eq!(end, 10);
    assert_eq!(&dest, b"hellohello");
}

#[test]
fn zero_match_offset_is_error() {
    let token = 1u8; // matchLen field=1, literalLen=0
    let mut compressed = vec![token];
    compressed.extend_from_slice(&0u16.to_le_bytes());
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 5];
    assert!(decompress(&mut input, 5, &mut dest, 0).is_err());
}

#[test]
fn match_before_buffer_start_is_error() {
    let token = 1u8;
    let mut compressed = vec![token];
    compressed.extend_from_slice(&100u16.to_le_bytes()); // matchDec > dOff
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 5];
    assert!(decompress(&mut input, 5, &mut dest, 0).is_err());
}

#[test]
fn extended_match_length_encoding() {
    // 1 literal byte 'z', then an extended-length match (matchLen field
    // = 0x0F + one extension byte 3 -> matchLen = 15+3+MIN_MATCH(4) = 22)
    // referencing match_dec=1 (run-length-encodes 'z' for the rest).
    let token = (1u8 << 4) | 0x0F;
    let mut compressed = vec![token, b'z'];
    compressed.extend_from_slice(&1u16.to_le_bytes()); // matchDec = 1
    compressed.push(3); // extension byte -> +3
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 23]; // 1 literal + 22 match bytes
    let end = decompress(&mut input, 23, &mut dest, 0).unwrap();
    assert_eq!(end, 23);
    assert_eq!(&dest, &[b'z'; 23]);
}

#[test]
fn truncated_input_is_error() {
    let compressed = [0x30u8, b'a']; // claims 3 literal bytes, only 1 present
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 3];
    assert!(decompress(&mut input, 3, &mut dest, 0).is_err());
}

#[test]
fn dest_too_small_for_decompressed_len_is_error() {
    let compressed = [0x30u8, b'a', b'b', b'c'];
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 2];
    assert!(decompress(&mut input, 3, &mut dest, 0).is_err());
}

/// The doubling overlap copy against the byte-at-a-time definition of
/// an LZ4 match, for every period up to past the length.
#[test]
fn copy_match_matches_the_byte_loop_for_every_period() {
    for len in [1usize, 2, 3, 4, 7, 8, 15, 16, 17, 33, 100, 300] {
        for period in 1..=len + 2 {
            let mut base: Vec<u8> = (0..period + len + 4).map(|i| (i * 37 + 11) as u8).collect();
            let mut want = base.clone();
            let (from, to) = (2, 2 + period);
            for i in 0..len {
                want[to + i] = want[from + i];
            }
            copy_match(&mut base, from, to, len);
            assert_eq!(base, want, "len {len} period {period}");
        }
    }
}

fn assert_compress_round_trips(payload: &[u8]) {
    let compressed = compress(payload);
    let mut input = SliceInput::new(&compressed);
    let mut dest = vec![0u8; payload.len()];
    let end = decompress(&mut input, payload.len(), &mut dest, 0).unwrap();
    assert_eq!(end, payload.len());
    assert_eq!(
        dest,
        payload,
        "round trip mismatch for len {}",
        payload.len()
    );
}

#[test]
fn compress_empty_input_round_trips() {
    assert_compress_round_trips(&[]);
}

#[test]
fn compress_single_byte_round_trips() {
    assert_compress_round_trips(b"x");
}

#[test]
fn compress_short_input_below_match_threshold_round_trips() {
    // len <= LAST_LITERALS + MIN_MATCH, so the whole main-loop match
    // search is skipped entirely and this is pure last-literals.
    assert_compress_round_trips(b"abcdefghi");
}

#[test]
fn compress_input_one_byte_above_match_threshold_round_trips() {
    // len == LAST_LITERALS + MIN_MATCH + 1 == 10: the smallest input for
    // which the main match-search loop actually runs at least once
    // (`match_limit`/`off <= limit` at the narrowest possible active
    // window) -- distinct from the len<=9 case above, which skips the
    // loop entirely.
    assert_compress_round_trips(b"abcdefghij");
}

#[test]
fn compress_highly_repetitive_input_actually_finds_matches() {
    // A phrase repeated many times has abundant 4+ byte back-references
    // available; assert the compressor actually uses them (output much
    // smaller than input), not just that it round-trips.
    let payload = "the quick brown fox jumps over the lazy dog ".repeat(200);
    let payload = payload.as_bytes();
    let compressed = compress(payload);
    assert!(
        compressed.len() < payload.len() / 4,
        "expected real back-reference compression, got {} bytes from {} bytes of input",
        compressed.len(),
        payload.len()
    );
    assert_compress_round_trips(payload);
}

#[test]
fn compress_incompressible_input_round_trips() {
    // Pseudo-random bytes (a simple xorshift-like sequence, no external
    // RNG dependency): no meaningful matches expected, exercising the
    // literal-heavy path of a real compressor rather than the stub's
    // always-one-literal-run shape.
    let mut state: u32 = 0x1234_5678;
    let payload: Vec<u8> = (0..2000)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xFF) as u8
        })
        .collect();
    assert_compress_round_trips(&payload);
}

#[test]
fn compress_match_at_very_start_round_trips() {
    // First 8 bytes are a repeated 4-byte pattern (an immediate match),
    // followed by non-repeating tail bytes.
    let mut payload = b"abcdabcd".to_vec();
    payload.extend((0..50u8).map(|i| i.wrapping_mul(7).wrapping_add(3)));
    assert_compress_round_trips(&payload);
}

#[test]
fn compress_match_at_very_end_round_trips() {
    // Non-repeating head, then a repeated 4-byte pattern right at the
    // tail (within LAST_LITERALS of the end, exercising the boundary
    // where the main loop must stop and fall back to last-literals).
    let mut payload: Vec<u8> = (0..50u8)
        .map(|i| i.wrapping_mul(11).wrapping_add(5))
        .collect();
    payload.extend_from_slice(b"wxyzwxyz");
    assert_compress_round_trips(&payload);
}

#[test]
fn compress_long_input_forces_extended_length_encoding() {
    // `encode_len`'s `while l >= 0xFF { push 0xFF; l -= 0xFF }` loop
    // must run at least twice (i.e. emit >=2 continuation bytes, not
    // just one) for BOTH the literal-length and match-length nibbles --
    // an off-by-one in the loop's second iteration wouldn't be caught by
    // a length needing only one continuation byte. `encode_literals`
    // calls `encode_len(literal_len - 0x0F, ...)`, so `literal_len` must
    // be >= 0x0F + 2*0xFF = 525 to force two iterations there; match
    // length is encoded as `match_len_total - MIN_MATCH - 0x0F`, so the
    // match run must be >= 4 + 0x0F + 2*0xFF = 529 bytes.
    //
    // Head: a long pseudo-random run (same xorshift generator as
    // `compress_incompressible_input_round_trips`) long enough that no
    // accidental 4-byte repeat gives the compressor an early match,
    // unlike a short-period `i % 251` sequence which repeats within
    // ~250 bytes and would (as an earlier version of this test did)
    // trigger a match well before the literal run reached the length
    // needed to actually exercise the chaining loop.
    let mut state: u32 = 0x9E37_79B9;
    let mut payload: Vec<u8> = (0..600)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            (state & 0xFF) as u8
        })
        .collect();
    // Tail: one long repeated-byte run, giving a single match of
    // total length >= 600 (well past the 529-byte threshold derived
    // above).
    payload.extend(std::iter::repeat_n(0x7Au8, 600));
    assert_compress_round_trips(&payload);

    // Confirm the chaining loop was actually exercised (not just that
    // the round trip happened to still work): re-derive the token
    // stream's first sequence and check its literal length is large
    // enough to have required >=2 continuation bytes.
    let compressed = compress(&payload);
    let token = compressed[0];
    let literal_len_nibble = (token >> 4) & 0x0F;
    assert_eq!(
        literal_len_nibble, 0x0F,
        "expected the first token's literal-length nibble to be maxed out (extended encoding)"
    );
    // Walk the 0xFF continuation bytes right after the token byte and
    // confirm there are at least 2 of them (proving the loop ran more
    // than once), then confirm the terminating byte plus the two 0xFF
    // bytes reconstruct a literal_len >= 525.
    let mut i = 1usize;
    let mut continuation_bytes = 0usize;
    let mut extra = 0usize;
    while compressed[i] == 0xFF {
        continuation_bytes += 1;
        extra += 0xFF;
        i += 1;
    }
    extra += compressed[i] as usize;
    assert!(
            continuation_bytes >= 2,
            "expected >=2 0xFF continuation bytes in the literal-length encoding, got {continuation_bytes}"
        );
    assert!(0x0F + extra >= 525);
}

#[test]
fn overlapping_match_with_a_multi_byte_period_repeats_the_whole_run() {
    // matchDec=3 with matchLen=8: the copy has to be cut into 3-byte
    // runs (3+3+2), each reading bytes the previous run just wrote.
    // A single memmove of 8 bytes would produce "abcabcab" only by luck
    // of direction; a wrong split produces garbage. Literals "abc",
    // then matchLen field = 8-4 = 4.
    let token = (3u8 << 4) | 4;
    let mut compressed = vec![token, b'a', b'b', b'c'];
    compressed.extend_from_slice(&3u16.to_le_bytes());
    let mut input = SliceInput::new(&compressed);
    let mut dest = [0u8; 11];
    let end = decompress(&mut input, 11, &mut dest, 0).unwrap();
    assert_eq!(end, 11);
    assert_eq!(&dest, b"abcabcabcab");
}

#[test]
fn common_bytes_counts_across_the_word_loop_boundary() {
    // The 8-byte word loop plus byte tail must agree with a naive count
    // for prefixes on either side of, and exactly at, a word boundary.
    let mut b = vec![0u8; 64];
    for shared in [0usize, 1, 7, 8, 9, 16, 23] {
        for (i, slot) in b.iter_mut().enumerate() {
            *slot = (i % 251) as u8;
        }
        // b[0..] and b[32..] agree for `shared` bytes then differ.
        b.copy_within(0..shared, 32);
        b[32 + shared] = b[shared] ^ 0xFF;
        assert_eq!(common_bytes(&b, 0, 32, 64), shared, "shared={shared}");
    }
}

#[test]
fn compress_with_dictionary_round_trips_against_the_preset_dictionary() {
    // The dictionary is bytes the decoder already has sitting before
    // `d_off`; the compressed block must be able to back-reference into
    // it, and must NOT re-emit it.
    let dict = "the quick brown fox jumps over the lazy dog ".repeat(4);
    let block = format!("{dict}and then the quick brown fox went home");
    let mut buffer = dict.as_bytes().to_vec();
    buffer.extend_from_slice(block.as_bytes());

    let mut out = Vec::new();
    let mut ht = FastCompressionHashTable::new();
    compress_with_dictionary(&buffer, 0, dict.len(), block.len(), &mut out, &mut ht);

    // The dictionary makes this compress far better than the same block
    // compressed standalone would.
    let standalone = compress(block.as_bytes());
    assert!(
        out.len() < standalone.len(),
        "preset dictionary should help: {} vs {}",
        out.len(),
        standalone.len()
    );

    let mut dest = vec![0u8; dict.len() + block.len()];
    dest[..dict.len()].copy_from_slice(dict.as_bytes());
    let mut input = SliceInput::new(&out);
    let end = decompress(&mut input, block.len(), &mut dest, dict.len()).unwrap();
    assert_eq!(end, dict.len() + block.len());
    assert_eq!(&dest[dict.len()..], block.as_bytes());
}

#[test]
fn a_reused_fast_hash_table_stays_correct_across_many_inputs() {
    // Java never clears the table between `reset` calls -- stale entries
    // are rejected by `get`'s range + byte-for-byte re-check instead.
    // Compressing a series of *different* buffers through one table has
    // to keep round-tripping, or that re-check is wrong.
    let mut ht = FastCompressionHashTable::new();
    for i in 0..40u32 {
        let payload: Vec<u8> = (0..500u32)
            .map(|j| (j.wrapping_mul(i.wrapping_add(1)) % 97) as u8)
            .collect();
        let mut out = Vec::new();
        compress_into(&payload, &mut out, &mut ht);
        let mut dest = vec![0u8; payload.len()];
        let mut input = SliceInput::new(&out);
        decompress(&mut input, payload.len(), &mut dest, 0).unwrap();
        assert_eq!(dest, payload, "round {i}");
    }
}

#[test]
fn fast_hash_table_switches_hash_log_above_64kb_and_still_round_trips() {
    // `reset` picks hashLog 13 below 64kB and 12 at or above it (Java's
    // `MEMORY_USAGE + 3 - bitsPerOffsetLog`); exercise both, in that
    // order, through one reused table so the smaller-hashLog run also
    // reads a table sized for the larger one.
    let mut ht = FastCompressionHashTable::new();
    for len in [1_000usize, 200_000] {
        let payload: Vec<u8> = (0..len).map(|i| (i % 211) as u8).collect();
        let mut out = Vec::new();
        compress_into(&payload, &mut out, &mut ht);
        let mut dest = vec![0u8; len];
        let mut input = SliceInput::new(&out);
        decompress(&mut input, len, &mut dest, 0).unwrap();
        assert_eq!(dest, payload, "len {len}");
    }
}

#[test]
fn high_compression_hash_table_beats_the_fast_one_and_round_trips() {
    // A payload whose 4-byte hooks recur many times with only the *last*
    // occurrence extendable into a long match: the fast table keeps only
    // that last occurrence per hash, the HC table walks the chain and
    // finds the longer one.
    let mut payload = Vec::new();
    for i in 0..200u32 {
        payload.extend_from_slice(b"HOOK");
        payload.extend_from_slice(&i.to_le_bytes());
        payload.extend_from_slice(b"filler-that-does-not-repeat");
        payload.extend_from_slice(&(i * 7919).to_le_bytes());
    }
    payload.extend_from_slice(b"HOOKfiller-that-does-not-repeat");

    let fast = compress(&payload);
    let mut high = Vec::new();
    let mut hc = HighCompressionHashTable::new();
    compress_into(&payload, &mut high, &mut hc);

    assert!(
        high.len() <= fast.len(),
        "HighCompressionHashTable should never lose to the fast one: {} vs {}",
        high.len(),
        fast.len()
    );

    let mut dest = vec![0u8; payload.len()];
    let mut input = SliceInput::new(&high);
    decompress(&mut input, payload.len(), &mut dest, 0).unwrap();
    assert_eq!(dest, payload);
}

#[test]
fn high_compression_hash_table_is_reusable_across_short_and_long_inputs() {
    // `reset`'s two branches: the cheap partial chain-table wipe after a
    // sub-64kB input, and the full wipe after a larger one. Alternate
    // between them through one table, including a wrap-around case where
    // the touched chain-table window straddles the mask boundary.
    let mut hc = HighCompressionHashTable::new();
    for len in [300usize, 100_000, 300, 70_000, 50] {
        let payload: Vec<u8> = (0..len).map(|i| (i % 37) as u8).collect();
        let mut out = Vec::new();
        compress_into(&payload, &mut out, &mut hc);
        let mut dest = vec![0u8; len];
        let mut input = SliceInput::new(&out);
        decompress(&mut input, len, &mut dest, 0).unwrap();
        assert_eq!(dest, payload, "len {len}");
    }
}

#[test]
fn high_compression_with_dictionary_round_trips() {
    let dict: Vec<u8> = (0..2_000u32).map(|i| (i % 61) as u8).collect();
    let block: Vec<u8> = (0..3_000u32).map(|i| ((i + 13) % 61) as u8).collect();
    let mut buffer = dict.clone();
    buffer.extend_from_slice(&block);

    let mut out = Vec::new();
    let mut hc = HighCompressionHashTable::new();
    compress_with_dictionary(&buffer, 0, dict.len(), block.len(), &mut out, &mut hc);

    let mut dest = vec![0u8; dict.len() + block.len()];
    dest[..dict.len()].copy_from_slice(&dict);
    let mut input = SliceInput::new(&out);
    decompress(&mut input, block.len(), &mut dest, dict.len()).unwrap();
    assert_eq!(&dest[dict.len()..], &block[..]);
}

#[test]
fn zero_length_decompress_still_consumes_one_token_byte() {
    // Even an empty unit is encoded as a single zero token, matching
    // Java's do-while loop shape (it always reads at least one byte).
    let compressed = [0x00u8];
    let mut input = SliceInput::new(&compressed);
    let mut dest: [u8; 0] = [];
    let end = decompress(&mut input, 0, &mut dest, 0).unwrap();
    assert_eq!(end, 0);
}
