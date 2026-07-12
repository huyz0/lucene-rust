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
use miniz_oxide::inflate::core::inflate_flags::TINFL_FLAG_USING_NON_WRAPPING_OUTPUT_BUF;
use miniz_oxide::inflate::core::{decompress as tinfl_decompress, DecompressorOxide};
use miniz_oxide::inflate::TINFLStatus;

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
    if decompressed_len == 0 {
        return Ok(d_off);
    }
    if compressed_len == 0 {
        return Err(Error::Corrupted(
            "deflate unit has zero compressed length but nonzero decompressed length".to_string(),
        ));
    }

    let mut compressed = vec![0u8; compressed_len];
    input.read_bytes(&mut compressed)?;

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
    use miniz_oxide::deflate::compress_to_vec;

    fn encode_unit(block: &[u8]) -> (Vec<u8>, usize) {
        let compressed = compress_to_vec(block, 6);
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
}
