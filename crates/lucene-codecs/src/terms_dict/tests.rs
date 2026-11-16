#![cfg(test)]
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
        mask: None,
        terms_dict_size: 0,
        max_term_length: 0,
        max_block_length: 8192,
        terms_data_offset: 0,
        terms_data_length: 0,
        index: None,
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
        mask: None,
        terms_dict_size: 1,
        max_term_length: 5,
        max_block_length: 8192,
        terms_data_offset: 0,
        terms_data_length: data.len() as i64,
        index: None,
    };
    assert_eq!(
        decode_all_terms(&data, &entry).unwrap(),
        vec![b"apple".to_vec()]
    );
}

/// 3 terms sharing prefixes: "apple", "application", "apply".
/// "application" vs "apple": common prefix "appl" (4), suffix "ication" (7).
/// "apply" vs "application": common prefix "appl" (4), suffix "y" (1).
fn prefix_compressed_block_fixture() -> (Vec<u8>, TermsDictEntry) {
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
        mask: None,
        terms_dict_size: 3,
        max_term_length: 11,
        max_block_length: 8192,
        terms_data_offset: 0,
        terms_data_length: data.len() as i64,
        index: None,
    };
    (data, entry)
}

#[test]
fn decode_all_terms_prefix_compressed_block() {
    let (data, entry) = prefix_compressed_block_fixture();
    assert_eq!(
        decode_all_terms(&data, &entry).unwrap(),
        vec![
            b"apple".to_vec(),
            b"application".to_vec(),
            b"apply".to_vec(),
        ]
    );
}

/// An exhausted cursor keeps saying so.
///
/// This is the one thing about the cursor `decode_all_terms`' own tests
/// cannot cover: `decode_all_terms` *is* the cursor collected
/// (`while let Some(term) = cursor.next_term()?`), so comparing the two
/// would compare a function to itself. What actually pins the two
/// decoders together is that there is only one, and what pins the cursor
/// against Java is `tests/sorted_doc_values_fixtures.rs`, which walks it
/// over a real `.dvd` and compares against the manifest real Lucene
/// wrote. A caller merging several cursors stops calling the exhausted
/// ones, but must not have to know that.
#[test]
fn an_exhausted_terms_cursor_keeps_returning_none() {
    let (data, entry) = prefix_compressed_block_fixture();
    let mut cursor = TermsCursor::open(&data, &entry).unwrap();
    let mut count = 0;
    while cursor.next_term().unwrap().is_some() {
        count += 1;
    }
    assert_eq!(count, entry.terms_dict_size, "the whole dictionary");
    assert!(cursor.next_term().unwrap().is_none());
    assert!(cursor.next_term().unwrap().is_none());
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
        mask: None,
        terms_dict_size: 2,
        max_term_length: 39,
        max_block_length: 8192,
        terms_data_offset: 0,
        terms_data_length: data.len() as i64,
        index: None,
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
        mask: None,
        terms_dict_size: 2,
        max_term_length: 14,
        max_block_length: 8192,
        terms_data_offset: 0,
        terms_data_length: data.len() as i64,
        index: None,
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
            mask: None,
            terms_dict_size: 1,
            max_term_length: 5,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
            index: None,
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
            mask: None,
            terms_dict_size: 1,
            max_term_length: 4,
            max_block_length: 8192,
            terms_data_offset: offset,
            terms_data_length: length,
            index: None,
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
            mask: None,
            terms_dict_size: 3,
            max_term_length: 11,
            max_block_length: 8,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
            index: None,
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
            mask: None,
            terms_dict_size: 3,
            max_term_length: 11,
            max_block_length: 8192,
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
            index: None,
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
                mask: None,
                terms_dict_size: 2,
                max_term_length: 11,
                max_block_length: 8192,
                terms_data_offset: 0,
                terms_data_length: data.len() as i64,
                index: None,
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
