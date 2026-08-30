//! Port of the terms-dictionary half of `Lucene90DocValuesProducer.TermsDict`
//! (used by SORTED/SORTED_SET doc values to map an ordinal to its term
//! bytes) — read-only, **decode-once, not lazy**.
//!
//! On disk, terms are grouped into 64-term blocks: each block's first term
//! is stored uncompressed, and the remaining 63 are prefix-compressed
//! against their immediate predecessor and then LZ4-compressed together
//! using the first term as a preset dictionary. Two auxiliary
//! `DirectMonotonicReader` arrays exist purely to support random access
//! without a full scan: a block-address array (seek straight to any
//! 64-term block) and a coarser "reverse index" over rarer sample terms
//! (binary-search which block a target term might be in). Since this port
//! materializes the whole dictionary in one pass ([`decode_all_terms`])
//! rather than exposing a lazy seekable `TermsEnum`, neither array is
//! needed for lookups — they're still parsed structurally (to keep the
//! `.dvm` cursor aligned for whatever field comes next) but their values
//! are discarded. See the `rust-performance` skill: this is the same
//! decode-fully trade-off already made for `IndexedDISI` and stored fields.

use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::Result;

use crate::direct_monotonic;
use crate::lz4;

/// `Lucene90DocValuesFormat.TERMS_DICT_BLOCK_LZ4_SHIFT`.
const TERMS_DICT_BLOCK_LZ4_SHIFT: u32 = 6;

/// 64 terms per LZ4 block (`1 << TERMS_DICT_BLOCK_LZ4_SHIFT`).
const BLOCK_SIZE: i64 = 1 << TERMS_DICT_BLOCK_LZ4_SHIFT;

#[derive(Debug, Clone)]
pub struct TermsDictEntry {
    pub terms_dict_size: i64,
    pub max_term_length: i32,
    /// `TermsDictEntry.maxBlockLength`: the longest decompressed block body
    /// the writer produced. Java sizes `TermsDict`'s `blockBuffer` from it;
    /// this port uses it as the bound on the per-block length vint, which
    /// otherwise sizes an unbounded allocation.
    pub max_block_length: i32,
    pub terms_data_offset: i64,
    pub terms_data_length: i64,
}

/// Parses a `TermsDictEntry` from the `.dvm` metadata stream. Must be called
/// at the exact position `Lucene90DocValuesProducer.readTermDict` would
/// read from (right after a SORTED/SORTED_SET field's ords entry).
pub fn read_term_dict_entry(input: &mut SliceInput) -> Result<TermsDictEntry> {
    let terms_dict_size = input.read_vlong()?;
    let block_shift = input.read_i32()? as u32;
    // `terms_dict_size` is a vlong off the `.dvm` and so is unbounded in both
    // directions. Java's `(termsDictSize + (1L << SHIFT) - 1) >>> SHIFT` wraps
    // where this port's `+` panics, and a negative size then reaches
    // `load_meta` as a negative block count. Both are corruption.
    let Some(addresses_size) = terms_dict_size
        .checked_add(BLOCK_SIZE - 1)
        .filter(|_| terms_dict_size >= 0)
        .map(|n| n >> TERMS_DICT_BLOCK_LZ4_SHIFT)
    else {
        return Err(lucene_store::Error::Corrupted(format!(
            "terms dict size out of range: {terms_dict_size}"
        )));
    };
    let _terms_addresses_meta = direct_monotonic::load_meta(input, addresses_size, block_shift)?;
    let max_term_length = input.read_i32()?;
    let max_block_length = input.read_i32()?;
    if max_block_length < 0 {
        return Err(lucene_store::Error::Corrupted(format!(
            "terms dict maxBlockLength must be non-negative, got {max_block_length}"
        )));
    }
    let terms_data_offset = input.read_i64()?;
    let terms_data_length = input.read_i64()?;
    let _terms_addresses_offset = input.read_i64()?;
    let _terms_addresses_length = input.read_i64()?;
    let terms_dict_index_shift = input.read_i32()? as u32;
    // `termsDictIndexShift` is a raw `int` off the `.dvm`. Java's `1L <<
    // shift` masks the shift to its low six bits, so a corrupt value there is
    // merely a wrong answer; in Rust it is a panic in a debug build and a
    // masked shift in a release one. The writer only ever emits
    // `TERMS_DICT_REVERSE_INDEX_SHIFT`, so anything that cannot even be a
    // legal `i64` shift is corruption.
    if terms_dict_index_shift >= 63 {
        return Err(lucene_store::Error::Corrupted(format!(
            "terms dict reverse-index shift out of range: {terms_dict_index_shift}"
        )));
    }
    // ARITH: `terms_dict_index_shift < 63` so `1i64 << shift` is in range and
    // positive; `terms_dict_size >= 0` was established above, so the
    // `checked_add` is the only place this can go wrong.
    #[allow(clippy::arithmetic_side_effects)]
    let index_size = terms_dict_size.checked_add((1i64 << terms_dict_index_shift) - 1);
    let Some(num_index_values) = index_size
        .map(|n| n >> terms_dict_index_shift)
        .and_then(|n| n.checked_add(1))
    else {
        return Err(lucene_store::Error::Corrupted(format!(
            "terms dict reverse-index size out of range: termsDictSize={terms_dict_size} \
             shift={terms_dict_index_shift}"
        )));
    };
    let _terms_index_addresses_meta =
        direct_monotonic::load_meta(input, num_index_values, block_shift)?;
    let _terms_index_offset = input.read_i64()?;
    let _terms_index_length = input.read_i64()?;
    let _terms_index_addresses_offset = input.read_i64()?;
    let _terms_index_addresses_length = input.read_i64()?;

    Ok(TermsDictEntry {
        terms_dict_size,
        max_term_length,
        max_block_length,
        terms_data_offset,
        terms_data_length,
    })
}

/// Decodes every term in the dictionary, in ordinal order. `data` is the
/// whole `.dvd` file's bytes.
pub fn decode_all_terms(data: &[u8], entry: &TermsDictEntry) -> Result<Vec<Vec<u8>>> {
    // Both halves are `i64`s read straight off the `.dvm`, so their sum is as
    // untrusted as either one: `offset + length` overflows before `data.get`
    // ever sees a range, and a negative offset becomes a huge `usize` through
    // the `as` cast. Same shape, same fix as `norms::sparse_region`.
    let start = usize::try_from(entry.terms_data_offset)
        .map_err(|_| lucene_store::Error::Eof { offset: 0 })?;
    let end = entry
        .terms_data_offset
        .checked_add(entry.terms_data_length)
        .and_then(|e| usize::try_from(e).ok())
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let region = data
        .get(start..end)
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let mut input = SliceInput::new(region);

    // `termsDictSize` comes straight off the wire, so it cannot be trusted to
    // size an allocation on its own: every term costs at least one byte in the
    // terms region, so the region's own length is a hard upper bound on how
    // many there can be. Java never faces this -- it does not preallocate.
    let capacity = entry.terms_dict_size.min(entry.terms_data_length).max(0) as usize;
    let mut terms: Vec<Vec<u8>> = Vec::with_capacity(capacity);
    // Decompressed body of the current block (everything after its first,
    // uncompressed term), plus a manual read cursor into it -- not a
    // `SliceInput`, since that would borrow `block_body` across the loop
    // iteration that reassigns it.
    let mut block_body: Vec<u8> = Vec::new();
    let mut block_pos: usize = 0;

    let mut ord: i64 = 0;
    // ARITH: `entry.terms_dict_size` is non-negative (`read_term_dict_entry`
    // rejects a negative one), `ord` starts at 0 and advances by exactly 1 per
    // iteration while `ord < terms_dict_size`, so `ord` stays in
    // `0..=terms_dict_size` and `ord + 1` is at most `terms_dict_size <=
    // i64::MAX`. `BLOCK_SIZE` is a non-zero constant.
    #[allow(clippy::arithmetic_side_effects)]
    while ord < entry.terms_dict_size {
        if ord % BLOCK_SIZE == 0 {
            let first_len = input.read_vint()?;
            // Java allocates `term.bytes` at `maxTermLength` once and would
            // throw on an over-long length; `first_len as usize` on a
            // negative vint is ~2^64 here, and `vec![0u8; n]` aborts rather
            // than erroring, which the FFI boundary cannot catch.
            if first_len < 0 || first_len as usize > input.remaining() {
                return Err(lucene_store::Error::Corrupted(format!(
                    "invalid terms dict first-term length {first_len}"
                )));
            }
            let first_len = first_len as usize;
            let mut term = vec![0u8; first_len];
            input.read_bytes(&mut term)?;

            // Only decompress a block body if more terms remain after this
            // block's first term (mirrors Java's `decompressBlock`, which
            // skips this when the first term is the very last one written).
            if ord + 1 < entry.terms_dict_size {
                // `block_len` is the *decompressed* body length, a vint off
                // disk that sizes an allocation directly. A negative one
                // becomes ~2^64 through `as usize`, and `vec![0u8; n]`
                // *aborts* on a failed allocation rather than erroring --
                // `catch_unwind` cannot intercept that, so it takes the JVM
                // down through the FFI. It cannot be bounded by the bytes left
                // in the region (LZ4 expands), so bound it the way Java does:
                // `TermsDict` sizes its `blockBuffer` from `maxBlockLength`,
                // which the writer set to the longest body it emitted.
                let block_len = input.read_vint()?;
                if block_len < 0 || block_len > entry.max_block_length {
                    return Err(lucene_store::Error::Corrupted(format!(
                        "invalid terms dict block length {block_len} \
                         (maxBlockLength={})",
                        entry.max_block_length
                    )));
                }
                // ARITH: `term.len()` is bounded by the terms region (the
                // first-term length check above, itself at most `isize::MAX`)
                // and `block_len` by `max_block_length`, an `i32`, so the sum
                // cannot overflow `usize`.
                #[allow(clippy::arithmetic_side_effects)]
                let buffer_len = term.len() + block_len as usize;
                let block_len = block_len as usize;
                let mut buffer = vec![0u8; buffer_len];
                buffer[..term.len()].copy_from_slice(&term);
                lz4::decompress(&mut input, block_len, &mut buffer, term.len())?;
                buffer.drain(..term.len());
                block_body = buffer;
            } else {
                block_body.clear();
            }
            block_pos = 0;
            terms.push(term);
        } else {
            let token = read_u8(&block_body, &mut block_pos)? as usize;
            // ARITH: `token` is a single byte, so `token & 0x0F` is `0..=15`
            // and `1 + (token >> 4)` is `1..=16`.
            #[allow(clippy::arithmetic_side_effects)]
            let (mut prefix_len, mut suffix_len) = (token & 0x0F, 1 + (token >> 4));
            let previous_len = terms
                .last()
                .expect("a block always pushes its first term first")
                .len();
            // The two continuation vints are read off disk and size the term
            // this iteration builds. A negative vint becomes ~2^64 through
            // `as usize`, which overflows the `+=` (a debug-build panic) and
            // otherwise reaches `Vec::with_capacity` as an aborting
            // allocation. The prefix is copied out of the previous term and
            // the suffix out of the block body, so each has its own limit --
            // the first term of a block is stored outside the body and can be
            // longer than it, so bounding the prefix by the body would reject
            // files Lucene wrote.
            if prefix_len == 15 {
                prefix_len =
                    bounded_extension(prefix_len, previous_len, &block_body, &mut block_pos)?;
            }
            if suffix_len == 16 {
                suffix_len =
                    bounded_extension(suffix_len, block_body.len(), &block_body, &mut block_pos)?;
            }
            let previous = terms
                .last()
                .expect("a block always pushes its first term first");
            // Java copies into a `term.bytes` buffer sized `maxTermLength` and
            // lets the JVM throw on an over-long prefix; slicing here would
            // panic instead, and a panic cannot cross the FFI boundary.
            if prefix_len > previous.len() {
                return Err(lucene_store::Error::Corrupted(format!(
                    "terms dict prefix length {prefix_len} exceeds previous term length {}",
                    previous.len()
                )));
            }
            // ARITH: `prefix_len <= previous.len()` was just checked and
            // `suffix_len <= block_body.len()` comes out of
            // `bounded_extension`; both are slice lengths, each at most
            // `isize::MAX`, so their sum cannot overflow `usize`.
            // `block_pos <= block_body.len()` is
            // `read_u8`/`read_vint`'s invariant, so `block_pos + suffix_len`
            // cannot overflow either -- it is merely allowed to run past the
            // end, which `get` reports.
            #[allow(clippy::arithmetic_side_effects)]
            let (capacity, suffix_end) = (prefix_len + suffix_len, block_pos + suffix_len);
            let mut term = Vec::with_capacity(capacity);
            term.extend_from_slice(&previous[..prefix_len]);
            let suffix = block_body
                .get(block_pos..suffix_end)
                .ok_or(lucene_store::Error::Eof { offset: block_pos })?;
            term.extend_from_slice(suffix);
            block_pos = suffix_end;
            terms.push(term);
        }
        ord += 1;
    }

    Ok(terms)
}

/// Adds a term-length continuation vint (read from `buf` at `pos`) to `base`,
/// rejecting anything past `limit`. Shared by the prefix and suffix halves of
/// a prefix-compressed term, which have the same hazard -- the vint is off
/// disk, `as usize` on a negative one is ~2^64, and the sum sizes both an
/// allocation and a slice range -- but different limits: a prefix is copied
/// out of the previous term, a suffix out of the block body.
fn bounded_extension(base: usize, limit: usize, buf: &[u8], pos: &mut usize) -> Result<usize> {
    let extra = read_vint(buf, pos)?;
    let sum = usize::try_from(extra)
        .ok()
        .and_then(|extra| base.checked_add(extra))
        .filter(|&sum| sum <= limit);
    sum.ok_or_else(|| {
        lucene_store::Error::Corrupted(format!(
            "terms dict term-length extension {extra} is out of range (base {base}, limit {limit})"
        ))
    })
}

// ARITH: `get` returned `Some`, so `*pos < buf.len() <= isize::MAX`.
#[allow(clippy::arithmetic_side_effects)]
fn read_u8(buf: &[u8], pos: &mut usize) -> Result<u8> {
    let b = *buf
        .get(*pos)
        .ok_or(lucene_store::Error::Eof { offset: *pos })?;
    *pos += 1;
    Ok(b)
}

/// Plain vint decode over a `&[u8]` cursor (the block body isn't a
/// `DataInput` -- see the comment on `block_body`/`block_pos` above).
///
/// Bounded at five bytes, which is the most `DataOutput.writeVInt` emits
/// ("between one and five bytes" -- its loop is `i >>>= 7` on an `int`), so
/// the bound cannot reject a `vint` any writer produced. It is deliberately
/// stricter than 10.5.0's *reader*: `DataInput.readVInt` there is a bare
/// unchecked loop, and the "Invalid vInt detected (too many bits)" exception
/// this comment used to cite is a later addition that is **not** in the pinned
/// tree. Unbounded, the shift itself overflows on the sixth byte -- a panic in
/// debug builds, a silently wrong value in release ones.
// ARITH: `shift` starts at 7, is compared against 28 before every use, and
// advances by 7, so it never leaves `7..=35` and `<<` on an `i32` stays in
// range for the shifts that actually run (the loop returns before shifting by
// 35).
#[allow(clippy::arithmetic_side_effects)]
fn read_vint(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let mut b = read_u8(buf, pos)?;
    let mut v = (b & 0x7f) as i32;
    let mut shift = 7;
    while b & 0x80 != 0 {
        if shift > 28 {
            return Err(lucene_store::Error::Corrupted(
                "invalid vInt detected (too many bits)".into(),
            ));
        }
        b = read_u8(buf, pos)?;
        v |= ((b & 0x7f) as i32) << shift;
        shift += 7;
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
    // The arithmetic gate is about values read off disk; a test's `i + 1` is
    // not one. See docs/arithmetic-gate.md.
    #![allow(clippy::arithmetic_side_effects)]

    use super::*;

    fn write_vint(out: &mut Vec<u8>, mut v: i32) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u32) >> 7) as i32;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    /// A "stored, not compressed" literal-only LZ4 encoding of `bytes`
    /// (mirrors `stored_fields.rs`'s test helper of the same shape).
    fn encode_literal_lz4(bytes: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        let len = bytes.len();
        assert!(len < 0x0F, "test helper only supports short literals");
        out.push((len as u8) << 4);
        out.extend_from_slice(bytes);
        out
    }

    #[test]
    fn decode_all_terms_empty_dict() {
        let entry = TermsDictEntry {
            terms_dict_size: 0,
            max_term_length: 0,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: 0,
        };
        assert_eq!(
            decode_all_terms(&[], &entry).unwrap(),
            Vec::<Vec<u8>>::new()
        );
    }

    #[test]
    fn decode_all_terms_single_term_no_block_body() {
        // Only 1 term: no block body is ever written (ord+1 < size is
        // false immediately after the first, uncompressed term).
        let mut data = Vec::new();
        write_vint(&mut data, 5);
        data.extend_from_slice(b"apple");

        let entry = TermsDictEntry {
            terms_dict_size: 1,
            max_term_length: 5,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
        };
        assert_eq!(
            decode_all_terms(&data, &entry).unwrap(),
            vec![b"apple".to_vec()]
        );
    }

    #[test]
    fn decode_all_terms_prefix_compressed_block() {
        // 3 terms sharing prefixes: "apple", "application", "apply".
        // "application" vs "apple": common prefix "appl" (4), suffix "ication" (7).
        // "apply" vs "application": common prefix "appl" (4), suffix "y" (1).
        let mut block_body = Vec::new();
        block_body.push((6u8 << 4) | 4); // suffixLen-1=6, prefixLen=4
        block_body.extend_from_slice(b"ication");
        block_body.push(4); // suffixLen-1=0, prefixLen=4
        block_body.push(b'y');

        let compressed_body = encode_literal_lz4(&block_body);

        let mut data = Vec::new();
        write_vint(&mut data, 5);
        data.extend_from_slice(b"apple");
        write_vint(&mut data, block_body.len() as i32); // decompressed block length
        data.extend_from_slice(&compressed_body);

        let entry = TermsDictEntry {
            terms_dict_size: 3,
            max_term_length: 11,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
        };
        assert_eq!(
            decode_all_terms(&data, &entry).unwrap(),
            vec![
                b"apple".to_vec(),
                b"application".to_vec(),
                b"apply".to_vec(),
            ]
        );
    }

    #[test]
    fn decode_all_terms_extended_prefix_and_suffix_lengths() {
        // prefixLen=15 (needs a vint extension: +2 -> 17) and
        // suffixLen=16 (needs a vint extension: +3 -> 19).
        let previous = vec![b'a'; 20];
        let suffix = vec![b'b'; 19];
        let mut block_body = Vec::new();
        block_body.push((15u8 << 4) | 15); // suffixLen field=15 (extended), prefixLen field=15 (extended)
        write_vint(&mut block_body, 2); // prefixLen = 15 + 2 = 17
        write_vint(&mut block_body, 3); // suffixLen = 16 + 3 = 19
        block_body.append(&mut suffix.clone());

        let mut compressed_body = Vec::new();
        // literal length 25 (>= 0x0F) needs the extended-literal-length LZ4
        // encoding: token nibble 0x0F, then a length-extension byte.
        compressed_body.push(0xF0);
        compressed_body.push((block_body.len() - 0x0F) as u8);
        compressed_body.extend_from_slice(&block_body);

        let mut data = Vec::new();
        write_vint(&mut data, previous.len() as i32);
        data.extend_from_slice(&previous);
        write_vint(&mut data, block_body.len() as i32);
        data.extend_from_slice(&compressed_body);

        let entry = TermsDictEntry {
            terms_dict_size: 2,
            max_term_length: 39,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
        };
        let terms = decode_all_terms(&data, &entry).unwrap();
        assert_eq!(terms[0], previous);
        let mut expected_second = previous[..17].to_vec();
        expected_second.extend_from_slice(&suffix);
        assert_eq!(terms[1], expected_second);
    }

    /// A prefix length longer than the previous term is corrupt input. Java
    /// copies into a `maxTermLength`-sized array and lets the JVM throw;
    /// slicing `previous[..prefix_len]` here would panic, and a panic cannot
    /// cross the FFI boundary.
    #[test]
    fn decode_all_terms_rejects_prefix_longer_than_the_previous_term() {
        // suffixLen-1=0, prefixLen=9 -- but the previous term is 5 bytes.
        let block_body = vec![9u8, b'y'];
        let compressed_body = encode_literal_lz4(&block_body);

        let mut data = Vec::new();
        write_vint(&mut data, 5);
        data.extend_from_slice(b"apple");
        write_vint(&mut data, block_body.len() as i32);
        data.extend_from_slice(&compressed_body);

        let entry = TermsDictEntry {
            terms_dict_size: 2,
            max_term_length: 14,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
        };
        assert!(matches!(
            decode_all_terms(&data, &entry),
            Err(lucene_store::Error::Corrupted(_))
        ));
    }

    /// A first-term length past the end of the terms region must be an error,
    /// not a `vec![0u8; ~2^64]` that aborts the process.
    #[test]
    fn decode_all_terms_rejects_first_term_length_past_the_region() {
        for first_len in [i32::MAX, -1] {
            let mut data = Vec::new();
            write_vint(&mut data, first_len);
            data.extend_from_slice(b"apple");
            let entry = TermsDictEntry {
                terms_dict_size: 1,
                max_term_length: 5,
                max_block_length: 8192,
                terms_data_offset: 0,
                terms_data_length: data.len() as i64,
            };
            assert!(
                matches!(
                    decode_all_terms(&data, &entry),
                    Err(lucene_store::Error::Corrupted(_))
                ),
                "first_len={first_len}"
            );
        }
    }

    /// `DataInput.readVInt` throws past five bytes; unbounded, the shift here
    /// overflows on the sixth (a debug-build panic, a wrong value in release).
    #[test]
    fn block_body_vint_is_bounded_at_five_bytes_like_java() {
        let mut pos = 0usize;
        let too_long = [0x80u8, 0x80, 0x80, 0x80, 0x80, 0x80, 0x01];
        assert!(matches!(
            read_vint(&too_long, &mut pos),
            Err(lucene_store::Error::Corrupted(_))
        ));

        // Five bytes is still legal, and decodes the same value Java's
        // readVInt does.
        let mut pos = 0usize;
        let max = [0xFFu8, 0xFF, 0xFF, 0xFF, 0x07];
        assert_eq!(read_vint(&max, &mut pos).unwrap(), i32::MAX);
        assert_eq!(pos, 5);
    }

    #[test]
    fn read_term_dict_entry_round_trips_fixed_fields() {
        let mut out = Vec::new();
        write_vint_vlong(&mut out, 100); // termsDictSize
        out.extend_from_slice(&2i32.to_le_bytes()); // blockShift
        let addresses_size = (100 + BLOCK_SIZE - 1) >> 6;
        write_direct_monotonic_meta(&mut out, addresses_size);
        out.extend_from_slice(&50i32.to_le_bytes()); // maxTermLength
        out.extend_from_slice(&8192i32.to_le_bytes()); // maxBlockLength
        out.extend_from_slice(&1000i64.to_le_bytes()); // termsDataOffset
        out.extend_from_slice(&2000i64.to_le_bytes()); // termsDataLength
        out.extend_from_slice(&0i64.to_le_bytes()); // termsAddressesOffset
        out.extend_from_slice(&0i64.to_le_bytes()); // termsAddressesLength
        out.extend_from_slice(&4i32.to_le_bytes()); // termsDictIndexShift
        let index_size = (100 + (1i64 << 4) - 1) >> 4;
        write_direct_monotonic_meta(&mut out, 1 + index_size);
        out.extend_from_slice(&0i64.to_le_bytes());
        out.extend_from_slice(&0i64.to_le_bytes());
        out.extend_from_slice(&0i64.to_le_bytes());
        out.extend_from_slice(&0i64.to_le_bytes());

        let mut input = SliceInput::new(&out);
        let entry = read_term_dict_entry(&mut input).unwrap();
        assert_eq!(entry.terms_dict_size, 100);
        assert_eq!(entry.max_term_length, 50);
        assert_eq!(entry.max_block_length, 8192);
        assert_eq!(entry.terms_data_offset, 1000);
        assert_eq!(entry.terms_data_length, 2000);
        assert_eq!(input.remaining(), 0);
    }

    /// `termsDictIndexShift` is a raw `i32` off the `.dvm`. `1i64 << shift`
    /// panics for a shift of 64 or more in a debug build (Java's `1L <<`
    /// merely masks it), and the `termsDictSize + (1 << shift) - 1` above it
    /// overflows for a large size.
    #[test]
    fn corrupt_reverse_index_shift_is_a_decode_error_not_a_shift_panic() {
        for shift in [63i32, 64, 100, -1] {
            let mut out = Vec::new();
            write_vint_vlong(&mut out, 100);
            out.extend_from_slice(&2i32.to_le_bytes());
            write_direct_monotonic_meta(&mut out, (100 + BLOCK_SIZE - 1) >> 6);
            out.extend_from_slice(&50i32.to_le_bytes());
            out.extend_from_slice(&8192i32.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes());
            out.extend_from_slice(&0i64.to_le_bytes());
            out.extend_from_slice(&shift.to_le_bytes());
            out.extend_from_slice(&[0u8; 64]);

            let mut input = SliceInput::new(&out);
            assert!(read_term_dict_entry(&mut input).is_err(), "shift={shift}");
        }
    }

    /// A `termsDictSize` near `i64::MAX` overflowed the `+ (BLOCK_SIZE - 1)`
    /// rounding before it ever reached `DirectMonotonicReader.loadMeta`, and a
    /// negative one reached it as a negative block count.
    #[test]
    fn corrupt_terms_dict_size_is_a_decode_error() {
        for size in [i64::MAX, i64::MAX - 10, -1] {
            let mut out = Vec::new();
            write_vint_vlong(&mut out, size);
            out.extend_from_slice(&2i32.to_le_bytes());
            out.extend_from_slice(&[0u8; 64]);
            let mut input = SliceInput::new(&out);
            assert!(read_term_dict_entry(&mut input).is_err(), "size={size}");
        }
    }

    /// `termsDataOffset`/`termsDataLength` are two independent `i64`s off the
    /// `.dvm`, so their sum is as untrusted as either: it overflowed before
    /// `data.get` saw a range, and a negative offset became a huge `usize`.
    #[test]
    fn corrupt_terms_data_region_is_a_decode_error() {
        let data = [0u8; 64];
        for (offset, length) in [(i64::MAX, 1i64), (0, i64::MAX), (-1, 4), (4, -1)] {
            let entry = TermsDictEntry {
                terms_dict_size: 1,
                max_term_length: 4,
                max_block_length: 8192,
                terms_data_offset: offset,
                terms_data_length: length,
            };
            assert!(
                decode_all_terms(&data, &entry).is_err(),
                "offset={offset} length={length}"
            );
        }
    }

    /// The compressed block length is a vint off disk that sizes
    /// `vec![0u8; term.len() + block_len]`. A negative vint is ~2^64 through
    /// `as usize`, and a failed `vec!` allocation *aborts* -- `catch_unwind`
    /// cannot intercept it.
    #[test]
    fn corrupt_block_length_is_a_decode_error_not_an_allocation() {
        // A block length that is positive, plausible and still past the
        // `.dvm`'s own `maxBlockLength` -- the case neither the negative check
        // nor a later EOF would catch.
        {
            let mut data = Vec::new();
            write_vint(&mut data, 5);
            data.extend_from_slice(b"apple");
            write_vint(&mut data, 100);
            data.extend_from_slice(&[0u8; 64]);
            let entry = TermsDictEntry {
                terms_dict_size: 3,
                max_term_length: 11,
                max_block_length: 8,
                terms_data_offset: 0,
                terms_data_length: data.len() as i64,
            };
            let got = decode_all_terms(&data, &entry);
            assert!(
                matches!(&got, Err(lucene_store::Error::Corrupted(m)) if m.contains("block length")),
                "{got:?}"
            );
        }
        for block_len in [-1i32, i32::MIN, i32::MAX] {
            let mut data = Vec::new();
            write_vint(&mut data, 5);
            data.extend_from_slice(b"apple");
            write_vint(&mut data, block_len);
            data.extend_from_slice(&[0u8; 8]);

            let entry = TermsDictEntry {
                terms_dict_size: 3,
                max_term_length: 11,
                max_block_length: 8192,
                terms_data_offset: 0,
                terms_data_length: data.len() as i64,
            };
            let got = decode_all_terms(&data, &entry);
            assert!(got.is_err(), "block_len={block_len}: {got:?}");
        }
    }

    /// The prefix/suffix continuation vints size the term being built. A
    /// negative one overflowed the `+=` in a debug build and reached
    /// `Vec::with_capacity` as an aborting allocation otherwise.
    #[test]
    fn corrupt_term_length_extension_is_a_decode_error() {
        for extension in [-1i32, i32::MIN, i32::MAX] {
            for token in [0x0Fu8, 0xF0 | 0x04] {
                let mut block_body = Vec::new();
                block_body.push(token);
                write_vint(&mut block_body, extension);
                block_body.extend_from_slice(b"xyz");

                let mut data = Vec::new();
                write_vint(&mut data, 5);
                data.extend_from_slice(b"apple");
                write_vint(&mut data, block_body.len() as i32);
                data.extend_from_slice(&encode_literal_lz4(&block_body));

                let entry = TermsDictEntry {
                    terms_dict_size: 2,
                    max_term_length: 11,
                    max_block_length: 8192,
                    terms_data_offset: 0,
                    terms_data_length: data.len() as i64,
                };
                let got = decode_all_terms(&data, &entry);
                assert!(
                    got.is_err(),
                    "extension={extension} token={token:#x}: {got:?}"
                );
            }
        }
    }

    fn write_vint_vlong(out: &mut Vec<u8>, mut v: i64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u64) >> 7) as i64;
            if v != 0 {
                b |= 0x80;
                out.push(b);
            } else {
                out.push(b);
                break;
            }
        }
    }

    fn write_direct_monotonic_meta(out: &mut Vec<u8>, num_values: i64) {
        let block_shift = 2u32;
        let mut num_blocks = num_values >> block_shift;
        if (num_blocks << block_shift) < num_values {
            num_blocks += 1;
        }
        for _ in 0..num_blocks {
            out.extend_from_slice(&0i64.to_le_bytes()); // min
            out.extend_from_slice(&0i32.to_le_bytes()); // avg
            out.extend_from_slice(&0i64.to_le_bytes()); // offset
            out.push(0); // bpv
        }
    }
}
