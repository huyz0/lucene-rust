#![cfg(test)]
use super::*;
use lucene_store::data_input::SliceInput;

fn compress(block: &[u8]) -> Vec<u8> {
    let mut compressed = Vec::new();
    crate::deflater::Deflater::new().compress(block, &mut compressed);
    compressed
}

fn encode_unit(block: &[u8]) -> (Vec<u8>, usize) {
    let compressed = compress(block);
    let len = compressed.len();
    (compressed, len)
}

#[test]
fn zero_decompressed_length_reads_nothing() {
    let mut input = SliceInput::new(&[]);
    let mut dest = [0u8; 4];
    let new_off = decompress(&mut input, 0, 0, &mut dest, 0).unwrap();
    assert_eq!(new_off, 0);
}

#[test]
fn no_dictionary_round_trips_plain_bytes() {
    let payload = b"the quick brown fox jumps over the lazy dog";
    let (compressed, compressed_len) = encode_unit(payload);
    let mut input = SliceInput::new(&compressed);
    let mut dest = vec![0u8; payload.len()];
    let new_off = decompress(&mut input, compressed_len, payload.len(), &mut dest, 0).unwrap();
    assert_eq!(new_off, payload.len());
    assert_eq!(&dest, payload);
}

#[test]
fn preset_dictionary_back_references_resolve_into_earlier_buffer() {
    // miniz_oxide's high-level compressor doesn't expose a preset
    // dictionary API, so this proves the decode-side mechanism directly:
    // write dictionary bytes into `dest` first (as `decompress_unit`
    // does), then decompress a block whose own repeated content lets it
    // compress well on its own -- what's under test is that `decompress`
    // writes at `d_off` (not 0) while leaving the preceding dictionary
    // bytes in `dest` untouched and addressable.
    let dict = b"hello world ";
    let block = b"hello world hello world hello world";
    let mut dest = vec![0u8; dict.len() + block.len()];
    dest[..dict.len()].copy_from_slice(dict);

    let (compressed, compressed_len) = encode_unit(block);
    let mut input = SliceInput::new(&compressed);
    let new_off = decompress(
        &mut input,
        compressed_len,
        block.len(),
        &mut dest,
        dict.len(),
    )
    .unwrap();
    assert_eq!(new_off, dict.len() + block.len());
    assert_eq!(&dest[..dict.len()], dict);
    assert_eq!(&dest[dict.len()..], block);
}

#[test]
fn lucene_deflater_streams_with_a_preset_dictionary() {
    let words = [
        "stored ",
        "fields ",
        "{\"id\": ",
        "merge ",
        "1234 ",
        "segment ",
    ];
    let mut raw = Vec::new();
    let mut x = 7u32;
    while raw.len() < 120_000 {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        raw.extend_from_slice(words[x as usize % words.len()].as_bytes());
        raw.push((x >> 24) as u8);
    }
    let dict = &raw[..2000];
    for block in [&raw[2000..2100], &raw[2000..50_000], &raw[2000..]] {
        let mut d = crate::deflater::Deflater::new();
        d.set_dictionary(dict);
        let mut c = Vec::new();
        d.compress(block, &mut c);
        let mut dest = dict.to_vec();
        dest.resize(dict.len() + block.len(), 0);
        let mut input = SliceInput::new(&c);
        let end = decompress(&mut input, c.len(), block.len(), &mut dest, dict.len()).unwrap();
        assert_eq!(end, dest.len());
        assert_eq!(&dest[dict.len()..], block);
    }
}

#[test]
fn truncated_compressed_bytes_rejected() {
    let payload = b"abc";
    let (mut compressed, compressed_len) = encode_unit(payload);
    compressed.truncate(compressed.len() - 1); // drop the last byte
    let mut input = SliceInput::new(&compressed);
    let mut dest = vec![0u8; payload.len()];
    assert!(decompress(&mut input, compressed_len, payload.len(), &mut dest, 0).is_err());
}

#[test]
fn oversized_compressed_length_rejected_before_allocation_or_input_consumption() {
    // usize::MAX cannot back a Vec: returning a typed error here also
    // proves corrupt lengths are rejected before allocation is attempted.
    for decompressed_len in [0, 1] {
        let mut input = SliceInput::new(&[0x12, 0x34]);
        let mut destination = [0x56];
        assert!(matches!(
            decompress(
                &mut input,
                usize::MAX,
                decompressed_len,
                &mut destination,
                0
            ),
            Err(Error::Corrupted(_))
        ));
        assert_eq!(input.position(), 0);
        assert_eq!(destination, [0x56]);
    }
    // Destination validation still wins over the compressed-input check.
    assert!(matches!(
        decompress(&mut SliceInput::new(&[]), usize::MAX, 1, &mut [], 0),
        Err(Error::Eof { offset: 1 })
    ));
}
#[test]
fn zero_compressed_length_with_nonzero_decompressed_length_rejected() {
    let mut input = SliceInput::new(&[]);
    let mut dest = vec![0u8; 4];
    assert!(decompress(&mut input, 0, 4, &mut dest, 0).is_err());
}

#[test]
fn dest_too_small_rejected() {
    let payload = b"abc";
    let (compressed, compressed_len) = encode_unit(payload);
    let mut input = SliceInput::new(&compressed);
    let mut dest = vec![0u8; 2]; // too small for a 3-byte decompressed_len
    assert!(decompress(&mut input, compressed_len, payload.len(), &mut dest, 0).is_err());
}

#[test]
fn wrong_decompressed_length_rejected() {
    // Compressed data for a 3-byte payload, but caller claims 4 bytes
    // expected -- miniz_oxide reports success with fewer bytes written
    // than requested.
    let payload = b"abc";
    let (compressed, compressed_len) = encode_unit(payload);
    let mut input = SliceInput::new(&compressed);
    let mut dest = vec![0u8; 4];
    assert!(decompress(&mut input, compressed_len, 4, &mut dest, 0).is_err());
}

fn assert_compress_round_trips(payload: &[u8]) {
    let compressed = compress(payload);
    let mut input = SliceInput::new(&compressed);
    let mut dest = vec![0u8; payload.len()];
    let end = decompress(&mut input, compressed.len(), payload.len(), &mut dest, 0).unwrap();
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
fn compress_tiny_input_round_trips() {
    assert_compress_round_trips(b"hi");
}

#[test]
fn compress_highly_repetitive_input_actually_compresses() {
    let payload = "the quick brown fox jumps over the lazy dog ".repeat(200);
    let payload = payload.as_bytes();
    let compressed = compress(payload);
    assert!(
        compressed.len() < payload.len() / 4,
        "expected real compression, got {} bytes from {} bytes of input",
        compressed.len(),
        payload.len()
    );
    assert_compress_round_trips(payload);
}

#[test]
fn compress_incompressible_input_round_trips() {
    // Pseudo-random bytes (simple xorshift, no external RNG dependency):
    // exercises deflate's stored/literal-heavy path.
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
fn compress_at_dict_block_boundary_sizes_round_trip() {
    // Sizes chosen to exercise stored_fields.rs's dict/sub-block
    // arithmetic (len / 60 and ceil((len-dict)/10)) at and around a
    // boundary: 60 bytes gives a nonzero 1-byte dictionary, 600 bytes
    // gives a round-numbered dictionary/block split.
    for len in [59usize, 60, 61, 599, 600, 601] {
        let payload: Vec<u8> = (0..len).map(|i| (i % 251) as u8).collect();
        assert_compress_round_trips(&payload);
    }
}
