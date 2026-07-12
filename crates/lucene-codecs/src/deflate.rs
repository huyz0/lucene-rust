//! Port of the decode half of `org.apache.lucene.codecs.lucene90.
//! DeflateWithPresetDictCompressionMode` -- raw DEFLATE (no zlib header, no
//! Adler-32 trailer, matching Java's `Inflater(true)`), used by
//! [`crate::stored_fields`]'s `Mode.BEST_COMPRESSION`. Structurally the same
//! preset-dictionary scheme as [`crate::lz4`] (a dictionary unit, then
//! fixed-size sub-blocks that may back-reference into it).
//!
//! Unlike LZ4 (self-terminating from a plain output-length count alone),
//! DEFLATE needs to be told exactly how many *compressed* bytes to feed the
//! inflator, so [`decompress`] takes `compressed_len` explicitly rather than
//! reading it from `input` itself -- the caller ([`crate::stored_fields`]'s
//! `decompress_unit`) already had to read every unit's compressed length
//! upfront (Java's format groups all of a chunk's compressed-length vints
//! together before any of the actual compressed bytes; see that function's
//! doc comment).
//!
//! Built on `miniz_oxide::inflate::core`'s low-level `decompress` rather
//! than its `decompress_to_vec*` convenience wrappers: those don't support
//! a preset dictionary, whereas the low-level function decompresses into an
//! arbitrary `out[out_pos..]` slice and allows back-references to look
//! earlier in the same buffer -- exactly the "dictionary bytes already
//! sitting before `d_off`" trick [`crate::lz4::decompress`] uses.

use lucene_store::data_input::DataInput;
use lucene_store::{Error, Result};
use miniz_oxide::deflate::compress_to_vec;
use miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
use miniz_oxide::inflate::core::{decompress as tinfl_decompress, DecompressorOxide};
use miniz_oxide::inflate::TINFLStatus;

/// Compresses `input` into a raw DEFLATE stream (no zlib header/trailer),
/// decodable by [`decompress`]. Real Lucene's `Deflater` at
/// `BEST_COMPRESSION` uses zlib's own DEFLATE implementation; this port does
/// not need byte-identical compressed output (compression is an internal
/// implementation detail invisible to a reader that only sees the
/// decompressed result), only a *correctly decodable* real DEFLATE stream --
/// so this reuses `miniz_oxide::deflate::compress_to_vec`, which is already a
/// dependency of this module's decode side (see [`decompress`]'s doc
/// comment) and exposes exactly this: a real, already-vetted DEFLATE encoder
/// with no zlib wrapper to strip (the sibling `compress_to_vec_zlib` is the
/// wrapped variant; plain `compress_to_vec` omits it). Hand-writing a DEFLATE
/// encoder (Huffman coding + LZ77) from scratch, the way [`crate::lz4`] does
/// for LZ4's simpler byte-token scheme, would be substantially more work for
/// no wire-format benefit here.
///
/// Level 6 (`miniz_oxide`'s "default"/balanced level, matching zlib's own
/// default) is used unconditionally: real Lucene's `BEST_COMPRESSION` mode
/// passes `Deflater.BEST_COMPRESSION` (zlib level 9), but since only the
/// *decompressed* bytes need to match, the compression level is purely a
/// speed/ratio trade-off with no correctness implication.
pub(crate) fn compress(input: &[u8]) -> Vec<u8> {
    compress_to_vec(input, 6)
}

/// Decompresses `compressed_len` raw-DEFLATE bytes from `input` into
/// `dest[d_off..d_off+decompressed_len]`. Back-references may reach earlier
/// into `dest` than `d_off` (a preset dictionary).
pub(crate) fn decompress(
    input: &mut impl DataInput,
    compressed_len: usize,
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
    if compressed_len == 0 {
        if decompressed_len != 0 {
            return Err(Error::Corrupted(
                "deflate unit has zero compressed length but nonzero decompressed length"
                    .to_string(),
            ));
        }
        return Ok(d_off);
    }

    // Even when `decompressed_len == 0`, `compressed_len` bytes were still
    // written to `input` by the writer (a real DEFLATE stream always emits
    // at least a final-block marker, even for zero-length content -- see
    // `deflate::compress`'s doc comment) and MUST still be consumed here:
    // `input` is shared across every unit in a chunk ([`crate::stored_fields
    // ::decompress_unit`] decodes a dictionary then several sub-blocks off
    // the same reader), so skipping these bytes would desync every unit
    // that follows.
    let mut compressed = vec![0u8; compressed_len];
    input.read_bytes(&mut compressed)?;
    if decompressed_len == 0 {
        return Ok(d_off);
    }

    let mut decompressor = DecompressorOxide::new();
    let (status, _in_read, out_written) = tinfl_decompress(
        &mut decompressor,
        &compressed,
        dest,
        d_off,
        TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF,
    );
    if status != TINFLStatus::Done || out_written != decompressed_len {
        return Err(Error::Corrupted(format!(
            "deflate decompression failed: status={status:?}, wrote {out_written} of {decompressed_len} expected bytes"
        )));
    }
    Ok(dest_end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use lucene_store::data_input::SliceInput;

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
    fn truncated_compressed_bytes_rejected() {
        let payload = b"abc";
        let (mut compressed, compressed_len) = encode_unit(payload);
        compressed.truncate(compressed.len() - 1); // drop the last byte
        let mut input = SliceInput::new(&compressed);
        let mut dest = vec![0u8; payload.len()];
        assert!(decompress(&mut input, compressed_len, payload.len(), &mut dest, 0).is_err());
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
}
