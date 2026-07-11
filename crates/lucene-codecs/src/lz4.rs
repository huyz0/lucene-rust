//! Port of the decode half of `org.apache.lucene.util.compress.LZ4`.
//!
//! Only [`decompress`] is ported (no compressor) — this port only reads
//! already-written indexes. The format itself needs no explanation beyond
//! the inline comments below; it's the standard LZ4 block format (token
//! byte, optional extended literal/match lengths, 16-bit little-endian match
//! offset), self-terminating once `decompressed_len` bytes have been
//! produced — the caller never needs to know the *compressed* length
//! up front.
//!
//! `dest`/`d_off` mirror Java's signature exactly (rather than taking a
//! plain `&mut [u8]` starting at 0): [`crate::stored_fields`]'s preset-dictionary
//! scheme decompresses into a buffer that already has dictionary bytes sitting
//! before `d_off`, and match back-references are allowed to reach into that
//! region.

use lucene_store::data_input::DataInput;
use lucene_store::{Error, Result};

const MIN_MATCH: usize = 4;

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
