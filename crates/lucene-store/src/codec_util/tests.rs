#![cfg(test)]
use super::*;

/// Test-only header/footer builder, independent of the Java fixtures under
/// `tests/codec_util_fixtures.rs`: those exercise real Java-written bytes;
/// this module exercises this decoder's own boundary/corruption handling
/// with hand-built buffers, so we don't need a JVM round-trip for every
/// error path (a truncated/tampered footer, an illegal CRC, etc).
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

fn write_string(out: &mut Vec<u8>, s: &str) {
    write_vint(out, s.len() as i32);
    out.extend_from_slice(s.as_bytes());
}

fn header_bytes(codec: &str, version: i32) -> Vec<u8> {
    let mut out = Vec::new();
    out.extend_from_slice(&CODEC_MAGIC.to_be_bytes());
    write_string(&mut out, codec);
    out.extend_from_slice(&version.to_be_bytes());
    out
}

#[test]
fn peek_codec_name_reads_only_the_name() {
    let buf = header_bytes("Lucene90FieldInfos", 7);
    assert_eq!(peek_codec_name(&buf).as_deref(), Some("Lucene90FieldInfos"));
    // A wrong magic, and a name cut short, are both "no name".
    let mut bad = buf.clone();
    bad[0] ^= 1;
    assert_eq!(peek_codec_name(&bad), None);
    assert_eq!(peek_codec_name(&buf[..6]), None);
    assert_eq!(peek_codec_name(&[]), None);
}

/// A complete, valid header + payload + footer, with a correct checksum.
fn valid_file(codec: &str, version: i32, payload: &[u8]) -> Vec<u8> {
    let mut out = header_bytes(codec, version);
    out.extend_from_slice(payload);
    out.extend_from_slice(&FOOTER_MAGIC.to_be_bytes());
    out.extend_from_slice(&0u32.to_be_bytes());
    let checksum = crc32fast::hash(&out) as u64;
    out.extend_from_slice(&checksum.to_be_bytes());
    out
}

#[test]
fn check_header_valid_roundtrip() {
    let buf = valid_file("Test", 3, b"payload");
    let mut input = SliceInput::new(&buf);
    let header = check_header(&mut input, "Test", 1, 3).unwrap();
    assert_eq!(header.version, 3);
}

#[test]
fn check_header_wrong_magic_rejected() {
    let mut buf = valid_file("Test", 1, b"x");
    buf[0] ^= 0xFF; // corrupt the magic itself, not just the codec name
    let mut input = SliceInput::new(&buf);
    assert!(matches!(
        check_header(&mut input, "Test", 1, 1),
        Err(Error::Corrupted(_))
    ));
}

/// The threaded CRC folds its pieces back to exactly the one-pass CRC,
/// at sizes that split unevenly and at the threshold itself.
#[test]
fn crc32_parallel_matches_the_single_pass_crc() {
    let mut x: u64 = 0x2545_F491_4F6C_DD1D;
    let big: Vec<u8> = (0..(PARALLEL_CRC_MIN * 3 + 12_345))
        .map(|_| {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x as u8
        })
        .collect();
    for len in [
        0,
        1,
        PARALLEL_CRC_MIN - 1,
        PARALLEL_CRC_MIN,
        PARALLEL_CRC_MIN * 2 + 7,
        big.len(),
    ] {
        assert_eq!(
            crc32(&big[..len]),
            crc32fast::hash(&big[..len]),
            "len {len}"
        );
    }
}

#[test]
fn check_footer_valid() {
    let buf = valid_file("Test", 1, b"hello");
    let mut input = SliceInput::new(&buf);
    check_header(&mut input, "Test", 1, 1).unwrap();
    input.seek(buf.len() - FOOTER_LENGTH).unwrap();
    let checksum = check_footer(&mut input, buf.len()).unwrap();
    assert_eq!(checksum, crc32fast::hash(&buf[..buf.len() - 8]) as u64);
}

#[test]
fn check_footer_file_too_small() {
    let buf = [0u8; 4]; // shorter than FOOTER_LENGTH
    let mut input = SliceInput::new(&buf);
    assert!(matches!(
        check_footer(&mut input, buf.len()),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn check_footer_wrong_position_rejected() {
    let buf = valid_file("Test", 1, b"hello");
    let mut input = SliceInput::new(&buf);
    // Positioned in the middle of the payload, not at the footer start.
    input.seek(5).unwrap();
    assert!(matches!(
        check_footer(&mut input, buf.len()),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn check_footer_wrong_magic_rejected() {
    let mut buf = valid_file("Test", 1, b"hello");
    let footer_start = buf.len() - FOOTER_LENGTH;
    buf[footer_start] ^= 0xFF; // corrupt footer magic
    let mut input = SliceInput::new(&buf);
    input.seek(footer_start).unwrap();
    assert!(matches!(
        check_footer(&mut input, buf.len()),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn check_footer_unknown_algorithm_id_rejected() {
    let mut buf = valid_file("Test", 1, b"hello");
    let footer_start = buf.len() - FOOTER_LENGTH;
    buf[footer_start + 7] = 1; // algorithmID's low byte -> 1 (only 0 is defined)
    let mut input = SliceInput::new(&buf);
    input.seek(footer_start).unwrap();
    assert!(matches!(
        check_footer(&mut input, buf.len()),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn check_footer_illegal_crc_high_bits_rejected() {
    let mut buf = valid_file("Test", 1, b"hello");
    let footer_start = buf.len() - FOOTER_LENGTH;
    // Set a high bit of the 64-bit checksum field, which a real CRC-32
    // (32 bits wide) could never produce.
    buf[footer_start + 8] = 0x01;
    let mut input = SliceInput::new(&buf);
    input.seek(footer_start).unwrap();
    assert!(matches!(
        check_footer(&mut input, buf.len()),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn check_footer_checksum_mismatch_rejected() {
    let mut buf = valid_file("Test", 1, b"hello");
    let last = buf.len() - 1;
    buf[last] ^= 0xFF; // flip a byte inside the checksum field itself
    let footer_start = buf.len() - FOOTER_LENGTH;
    let mut input = SliceInput::new(&buf);
    input.seek(footer_start).unwrap();
    assert!(matches!(
        check_footer(&mut input, buf.len()),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn check_whole_file_header_too_small() {
    let buf = [0u8; 4];
    assert!(matches!(
        check_whole_file_header(&buf, "Test", 1, 1),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn check_whole_file_header_and_footer_valid() {
    let buf = valid_file("Test", 2, b"body");
    let header = check_whole_file_header(&buf, "Test", 1, 2).unwrap();
    assert_eq!(header.version, 2);
    let payload_end = buf.len() - FOOTER_LENGTH;
    check_whole_file_footer(&buf, payload_end).unwrap();
}

#[test]
fn retrieve_checksum_valid() {
    let buf = valid_file("Test", 1, b"hello");
    let checksum = retrieve_checksum(&buf).unwrap();
    assert_eq!(checksum, crc32fast::hash(&buf[..buf.len() - 8]) as u64);
}

#[test]
fn retrieve_checksum_too_small() {
    let buf = [0u8; 4];
    assert!(matches!(retrieve_checksum(&buf), Err(Error::Corrupted(_))));
}

#[test]
fn retrieve_checksum_wrong_magic_rejected() {
    let mut buf = valid_file("Test", 1, b"hello");
    let footer_start = buf.len() - FOOTER_LENGTH;
    buf[footer_start] ^= 0xFF;
    assert!(matches!(retrieve_checksum(&buf), Err(Error::Corrupted(_))));
}

#[test]
fn retrieve_checksum_does_not_detect_payload_corruption() {
    // By design: retrieve_checksum only validates footer *shape*, not the
    // CRC against the payload — that's the whole point (cheap check for a
    // forward-only read pattern). Corrupting the payload without touching
    // the footer must NOT be caught here.
    let mut buf = valid_file("Test", 1, b"hello");
    let payload_byte = header_bytes("Test", 1).len();
    buf[payload_byte] ^= 0xFF;
    assert!(retrieve_checksum(&buf).is_ok());
}

#[test]
fn check_index_header_id_mismatch() {
    let mut buf = header_bytes("Test", 1);
    let id = [7u8; ID_LENGTH];
    buf.extend_from_slice(&id);
    buf.push(0); // empty suffix
    let mut input = SliceInput::new(&buf);
    check_header(&mut input, "Test", 1, 1).unwrap();
    let wrong_id = [8u8; ID_LENGTH];
    assert!(matches!(
        check_index_header_id(&mut input, &wrong_id),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn check_index_header_suffix_mismatch() {
    let mut buf = header_bytes("Test", 1);
    let id = [1u8; ID_LENGTH];
    buf.extend_from_slice(&id);
    buf.push(1);
    buf.push(b'a'); // suffix "a"
    let mut input = SliceInput::new(&buf);
    check_header(&mut input, "Test", 1, 1).unwrap();
    check_index_header_id(&mut input, &id).unwrap();
    assert!(matches!(
        check_index_header_suffix(&mut input, "b"),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn header_length_matches_the_bytes_write_header_emits() {
    for codec in ["Test", "Lucene90SegmentInfo", "X"] {
        let mut out = Vec::new();
        write_header(&mut out, codec, 7);
        assert_eq!(out.len(), header_length(codec), "codec {codec}");
    }
}

#[test]
fn index_header_length_matches_the_bytes_write_index_header_emits() {
    for (codec, suffix) in [("Test", ""), ("Lucene90Postings", "0"), ("C", "abcdef")] {
        let mut out = Vec::new();
        write_index_header(&mut out, codec, 7, &[0xAB; ID_LENGTH], suffix);
        assert_eq!(
            out.len(),
            index_header_length(codec, suffix),
            "codec {codec} suffix {suffix}"
        );
    }
}

#[test]
#[should_panic(expected = "codec must be simple ASCII")]
fn write_header_rejects_a_non_ascii_codec_name_in_debug() {
    // A multi-byte name would make header_length() -- and every codec that
    // seeks past the header by it -- wrong.
    let mut out = Vec::new();
    write_header(&mut out, "Lucene\u{e9}", 1);
}

#[test]
#[should_panic(expected = "suffix must be simple ASCII")]
fn write_index_header_rejects_an_over_long_suffix_in_debug() {
    // The suffix length is a single byte: 256 would be written as 0 and
    // silently produce a file no reader can parse.
    let mut out = Vec::new();
    write_index_header(&mut out, "Test", 1, &[0u8; ID_LENGTH], &"x".repeat(256));
}

#[test]
fn retrieve_checksum_with_expected_length_accepts_the_exact_length() {
    let buf = valid_file("Test", 1, b"payload");
    let expected = crc32fast::hash(&buf[..buf.len() - 8]) as u64;
    assert_eq!(
        retrieve_checksum_with_expected_length(&buf, buf.len()).unwrap(),
        expected
    );
}

#[test]
fn retrieve_checksum_with_expected_length_rejects_wrong_lengths() {
    let buf = valid_file("Test", 1, b"payload");
    // Truncated (file shorter than promised), extended (longer), and an
    // expectedLength that cannot hold a footer at all.
    assert!(matches!(
        retrieve_checksum_with_expected_length(&buf, buf.len() + 1),
        Err(Error::Corrupted(_))
    ));
    assert!(matches!(
        retrieve_checksum_with_expected_length(&buf, buf.len() - 1),
        Err(Error::Corrupted(_))
    ));
    assert!(matches!(
        retrieve_checksum_with_expected_length(&buf, FOOTER_LENGTH - 1),
        Err(Error::Corrupted(_))
    ));
}

#[test]
fn write_header_round_trips_through_check_header() {
    let mut buf = Vec::new();
    write_header(&mut buf, "Test", 3);
    let mut input = SliceInput::new(&buf);
    let header = check_header(&mut input, "Test", 1, 3).unwrap();
    assert_eq!(header.version, 3);
}

#[test]
fn write_index_header_round_trips_through_check_index_header() {
    let id = [5u8; ID_LENGTH];
    let mut buf = Vec::new();
    write_index_header(&mut buf, "Test", 2, &id, "seg1");
    let mut input = SliceInput::new(&buf);
    let header = check_index_header(&mut input, "Test", 1, 2, &id, "seg1").unwrap();
    assert_eq!(header.version, 2);
    assert_eq!(header.id, id);
    assert_eq!(header.suffix, "seg1");
}

#[test]
fn write_index_header_empty_suffix_round_trips() {
    let id = [9u8; ID_LENGTH];
    let mut buf = Vec::new();
    write_index_header(&mut buf, "Test", 1, &id, "");
    let mut input = SliceInput::new(&buf);
    check_index_header(&mut input, "Test", 1, 1, &id, "").unwrap();
}

#[test]
fn write_footer_round_trips_through_check_footer() {
    let id = [3u8; ID_LENGTH];
    let mut buf = Vec::new();
    write_index_header(&mut buf, "Test", 1, &id, "");
    buf.extend_from_slice(b"payload bytes");
    write_footer(&mut buf);

    let mut input = SliceInput::new(&buf);
    check_index_header(&mut input, "Test", 1, 1, &id, "").unwrap();
    input.skip(b"payload bytes".len()).unwrap();
    let checksum = check_footer(&mut input, buf.len()).unwrap();
    assert_eq!(checksum, crc32fast::hash(&buf[..buf.len() - 8]) as u64);
}

#[test]
fn write_footer_round_trips_through_retrieve_checksum() {
    let mut buf = Vec::new();
    write_header(&mut buf, "Test", 1);
    buf.extend_from_slice(b"x");
    write_footer(&mut buf);
    retrieve_checksum(&buf).unwrap();
}

#[test]
fn full_written_file_passes_check_whole_file_header_and_footer() {
    let id = [11u8; ID_LENGTH];
    let mut buf = Vec::new();
    write_index_header(&mut buf, "Test", 1, &id, "sfx");
    buf.extend_from_slice(b"body");
    write_footer(&mut buf);

    let header = check_whole_file_header(&buf, "Test", 1, 1).unwrap();
    assert_eq!(header.version, 1);
    let payload_end = buf.len() - FOOTER_LENGTH;
    check_whole_file_footer(&buf, payload_end).unwrap();
}
