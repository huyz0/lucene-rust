//! Port of `org.apache.lucene.util.compress.LZ4`'s decode half, plus a real
//! (non-literal-only) block compressor.
//!
//! [`decompress`] handles the standard LZ4 block format (token byte,
//! optional extended literal/match lengths, 16-bit little-endian match
//! offset), self-terminating once `decompressed_len` bytes have been
//! produced — the caller never needs to know the *compressed* length
//! up front.
//!
//! `dest`/`d_off` mirror Java's signature exactly (rather than taking a
//! plain `&mut [u8]` starting at 0): [`crate::stored_fields`]'s preset-dictionary
//! scheme decompresses into a buffer that already has dictionary bytes sitting
//! before `d_off`, and match back-references are allowed to reach into that
//! region.
//!
//! [`compress`] is a real greedy back-reference compressor (see its own doc
//! comment), scoped to the zero-length/no-preset-dictionary case — the
//! `LZ4WithPresetDictCompressionMode` dictionary-based cross-block variant
//! is a separate, larger piece and out of scope here.

use lucene_store::data_input::DataInput;
use lucene_store::{Error, Result};

const MIN_MATCH: usize = 4;
const LAST_LITERALS: usize = 5;
const MAX_DISTANCE: usize = 1 << 16;
/// Fixed hash-table size for the compressor's match finder: `2^17` `i64`
/// slots (empty = `-1`), one last-occurrence position per hash bucket.
/// Real Lucene's `FastCompressionHashTable.reset` sizes this dynamically
/// from the input length (`hashLog = MEMORY_USAGE_FACTOR + 3 -
/// bitsPerOffsetLog`, working out to 13 for inputs under 64KB, 12 above --
/// never 17). This port deliberately does NOT replicate that sizing
/// formula: a larger fixed table can only ever *find* a match Lucene's
/// smaller table would have found too (every candidate is still verified
/// byte-for-byte before use, see `compress`'s match-verification step, so a
/// bigger table never risks a wrong match, only a differently-sized
/// compressed output than Lucene's own writer would produce for the same
/// input) — correctness and real compression are this slice's goal, not
/// byte-identical output to Lucene's own writer.
const HASH_LOG: u32 = 17;

/// Decompresses into `dest[d_off..d_off+decompressed_len]`, reading a
/// self-terminating LZ4 block from `input`. Back-references may reach
/// earlier into `dest` than `d_off` (a preset dictionary). Returns
/// `d_off + decompressed_len` (mirroring Java's return value, the new
/// write position) on success.
pub(crate) fn decompress(
    input: &mut impl DataInput,
    decompressed_len: usize,
    dest: &mut [u8],
    d_off: usize,
) -> Result<usize> {
    let dest_end = d_off
        .checked_add(decompressed_len)
        .ok_or(Error::Eof { offset: d_off })?;
    if dest_end > dest.len() {
        return Err(Error::Eof { offset: dest_end });
    }
    let mut d_off = d_off;

    loop {
        let token = input.read_byte()? as usize;
        let mut literal_len = token >> 4;
        if literal_len == 0x0F {
            loop {
                let len = input.read_byte()?;
                if len == 0xFF {
                    literal_len += 0xFF;
                } else {
                    literal_len += len as usize;
                    break;
                }
            }
        }
        if literal_len != 0 {
            let end = d_off
                .checked_add(literal_len)
                .ok_or(Error::Eof { offset: d_off })?;
            if end > dest.len() {
                return Err(Error::Eof { offset: end });
            }
            input.read_bytes(&mut dest[d_off..end])?;
            d_off = end;
        }

        if d_off >= dest_end {
            break;
        }

        let match_dec = input.read_u16()? as usize;
        if match_dec == 0 {
            return Err(Error::Corrupted("LZ4 match offset 0 is invalid".into()));
        }

        let mut match_len = token & 0x0F;
        if match_len == 0x0F {
            loop {
                let len = input.read_byte()?;
                if len == 0xFF {
                    match_len += 0xFF;
                } else {
                    match_len += len as usize;
                    break;
                }
            }
        }
        match_len += MIN_MATCH;

        if match_dec > d_off {
            return Err(Error::Corrupted(
                "LZ4 match references before the start of the buffer".into(),
            ));
        }
        let src_start = d_off - match_dec;
        let end = d_off
            .checked_add(match_len)
            .ok_or(Error::Eof { offset: d_off })?;
        if end > dest.len() {
            return Err(Error::Eof { offset: end });
        }
        // Byte-by-byte, not `copy_within`: when `match_dec < match_len` the
        // source and destination ranges overlap, and each output byte can
        // depend on one just written (this is how LZ4 encodes short runs).
        for i in 0..match_len {
            dest[d_off + i] = dest[src_start + i];
        }
        d_off = end;

        if d_off >= dest_end {
            break;
        }
    }

    Ok(d_off)
}

/// Reads 4 bytes at `buf[i..i+4]` as a native-endian `u32` for hashing/
/// comparison purposes only (mirrors Java's comment on `readInt`: LZ4's
/// algorithm doesn't care about endianness here since these bytes are never
/// written to the output -- only compared to each other and hashed).
#[inline]
fn read4(buf: &[u8], i: usize) -> u32 {
    let mut b = [0u8; 4];
    b.copy_from_slice(&buf[i..i + 4]);
    u32::from_ne_bytes(b)
}

/// Port of `LZ4.hash` (the multiplicative hash used by both hash-table
/// variants): `(i * -1640531535) >>> (32 - hashBits)` in Java's `int` math,
/// i.e. wrapping `u32` multiplication by the same constant reinterpreted as
/// unsigned (`0x9E3779B1`).
#[inline]
fn hash(v: u32, hash_bits: u32) -> u32 {
    v.wrapping_mul(0x9E3779B1) >> (32 - hash_bits)
}

/// Port of `Arrays.mismatch(b, o1, limit, b, o2, limit)`: the number of
/// leading bytes that agree between `b[o1..limit]` and `b[o2..limit]`. Java's
/// `commonBytes` asserts this is never -1 (i.e. never "fully equal" up to
/// `limit`) because the two regions always end up with differing lengths
/// available before `limit`; this port doesn't rely on that invariant, it
/// just stops at whichever bound is reached first.
#[inline]
fn common_bytes(b: &[u8], o1: usize, o2: usize, limit: usize) -> usize {
    let max = (limit - o1).min(limit - o2);
    let mut n = 0;
    while n < max && b[o1 + n] == b[o2 + n] {
        n += 1;
    }
    n
}

fn encode_len(mut l: usize, out: &mut Vec<u8>) {
    while l >= 0xFF {
        out.push(0xFF);
        l -= 0xFF;
    }
    out.push(l as u8);
}

fn encode_literals(bytes: &[u8], token: u8, anchor: usize, literal_len: usize, out: &mut Vec<u8>) {
    out.push(token);
    if literal_len >= 0x0F {
        encode_len(literal_len - 0x0F, out);
    }
    out.extend_from_slice(&bytes[anchor..anchor + literal_len]);
}

fn encode_last_literals(bytes: &[u8], anchor: usize, literal_len: usize, out: &mut Vec<u8>) {
    let token = (literal_len.min(0x0F) as u8) << 4;
    encode_literals(bytes, token, anchor, literal_len, out);
}

fn encode_sequence(
    bytes: &[u8],
    anchor: usize,
    match_ref: usize,
    match_off: usize,
    match_len: usize,
    out: &mut Vec<u8>,
) {
    let literal_len = match_off - anchor;
    debug_assert!(match_len >= MIN_MATCH);
    let token = ((literal_len.min(0x0F) as u8) << 4) | (match_len - MIN_MATCH).min(0x0F) as u8;
    encode_literals(bytes, token, anchor, literal_len, out);

    let match_dec = match_off - match_ref;
    debug_assert!(match_dec > 0 && match_dec < (1 << 16));
    out.extend_from_slice(&(match_dec as u16).to_le_bytes());

    if match_len >= MIN_MATCH + 0x0F {
        encode_len(match_len - 0x0F - MIN_MATCH, out);
    }
}

/// Real LZ4 block compressor -- a scoped-down port of
/// `LZ4.compressWithDictionary` with a zero-length dictionary (no preset
/// dictionary support; see [`crate::stored_fields::write_best_speed`]'s doc
/// comment for why that variant is out of scope here) and always using the
/// simple `FastCompressionHashTable` match-finding strategy (a single
/// last-occurrence-per-hash table, no hash-chain "try a better match" search
/// like `HighCompressionHashTable`/`compressHC` does). This trades a bit of
/// compression ratio for a much smaller port: every match this function
/// finds is real (a byte-for-byte-verified back-reference within
/// [`MAX_DISTANCE`]), just not necessarily the *longest possible* one at
/// each position the way Lucene's own `BEST_COMPRESSION` search would find.
///
/// Produces a single self-terminating LZ4 block (the same wire format
/// [`decompress`] reads): a sequence of token/literal/match-offset/
/// match-length groups, ending in a final literal-only run covering
/// whatever's left. Always decodable by both this port's own `decompress`
/// and real Lucene's `LZ4.decompress`.
pub(crate) fn compress(bytes: &[u8]) -> Vec<u8> {
    let mut out = Vec::new();
    let len = bytes.len();
    let mut anchor = 0usize;

    if len > LAST_LITERALS + MIN_MATCH {
        let limit = len - LAST_LITERALS;
        let match_limit = limit - MIN_MATCH;
        let mut table = vec![-1i64; 1usize << HASH_LOG];
        let mut off = 0usize;

        'main: while off <= limit {
            // find a match
            let match_ref;
            loop {
                if off >= match_limit {
                    break 'main;
                }
                let v = read4(bytes, off);
                let h = hash(v, HASH_LOG) as usize;
                let prev = table[h];
                table[h] = off as i64;
                if prev >= 0
                    && (prev as usize) < off
                    && off - (prev as usize) < MAX_DISTANCE
                    && read4(bytes, prev as usize) == v
                {
                    match_ref = prev as usize;
                    break;
                }
                off += 1;
            }

            let match_len =
                MIN_MATCH + common_bytes(bytes, match_ref + MIN_MATCH, off + MIN_MATCH, limit);

            encode_sequence(bytes, anchor, match_ref, off, match_len, &mut out);
            off += match_len;
            anchor = off;
        }
    }

    // last literals
    encode_last_literals(bytes, anchor, len - anchor, &mut out);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::data_input::SliceInput;

    /// literal-only block: token=0x30 (literalLen=3, matchLen=0), then "abc".
    /// Since matchLen would be read next but dOff reaches destEnd right after
    /// the literals, no match bytes follow.
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
    fn zero_length_decompress_still_consumes_one_token_byte() {
        // Even an empty unit is encoded as a single zero token, matching
        // Java's do-while loop shape (it always reads at least one byte).
        let compressed = [0x00u8];
        let mut input = SliceInput::new(&compressed);
        let mut dest: [u8; 0] = [];
        let end = decompress(&mut input, 0, &mut dest, 0).unwrap();
        assert_eq!(end, 0);
    }
}
