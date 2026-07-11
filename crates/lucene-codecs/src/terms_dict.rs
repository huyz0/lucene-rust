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

/// 64 terms per LZ4 block (`Lucene90DocValuesFormat.TERMS_DICT_BLOCK_LZ4_SHIFT`).
const BLOCK_SIZE: i64 = 1 << 6;

#[derive(Debug, Clone)]
pub struct TermsDictEntry {
    pub terms_dict_size: i64,
    pub max_term_length: i32,
    pub terms_data_offset: i64,
    pub terms_data_length: i64,
}

/// Parses a `TermsDictEntry` from the `.dvm` metadata stream. Must be called
/// at the exact position `Lucene90DocValuesProducer.readTermDict` would
/// read from (right after a SORTED/SORTED_SET field's ords entry).
pub fn read_term_dict_entry(input: &mut SliceInput) -> Result<TermsDictEntry> {
    let terms_dict_size = input.read_vlong()?;
    let block_shift = input.read_i32()? as u32;
    let addresses_size = (terms_dict_size + BLOCK_SIZE - 1) >> 6;
    let _terms_addresses_meta = direct_monotonic::load_meta(input, addresses_size, block_shift)?;
    let max_term_length = input.read_i32()?;
    let _max_block_length = input.read_i32()?;
    let terms_data_offset = input.read_i64()?;
    let terms_data_length = input.read_i64()?;
    let _terms_addresses_offset = input.read_i64()?;
    let _terms_addresses_length = input.read_i64()?;
    let terms_dict_index_shift = input.read_i32()? as u32;
    let index_size =
        (terms_dict_size + (1i64 << terms_dict_index_shift) - 1) >> terms_dict_index_shift;
    let _terms_index_addresses_meta =
        direct_monotonic::load_meta(input, 1 + index_size, block_shift)?;
    let _terms_index_offset = input.read_i64()?;
    let _terms_index_length = input.read_i64()?;
    let _terms_index_addresses_offset = input.read_i64()?;
    let _terms_index_addresses_length = input.read_i64()?;

    Ok(TermsDictEntry {
        terms_dict_size,
        max_term_length,
        terms_data_offset,
        terms_data_length,
    })
}

/// Decodes every term in the dictionary, in ordinal order. `data` is the
/// whole `.dvd` file's bytes.
pub fn decode_all_terms(data: &[u8], entry: &TermsDictEntry) -> Result<Vec<Vec<u8>>> {
    let region = data
        .get(
            entry.terms_data_offset as usize
                ..(entry.terms_data_offset + entry.terms_data_length) as usize,
        )
        .ok_or(lucene_store::Error::Eof { offset: 0 })?;
    let mut input = SliceInput::new(region);

    let mut terms: Vec<Vec<u8>> = Vec::with_capacity(entry.terms_dict_size as usize);
    let mut previous: Vec<u8> = Vec::new();
    // Decompressed body of the current block (everything after its first,
    // uncompressed term), plus a manual read cursor into it -- not a
    // `SliceInput`, since that would borrow `block_body` across the loop
    // iteration that reassigns it.
    let mut block_body: Vec<u8> = Vec::new();
    let mut block_pos: usize = 0;

    let mut ord: i64 = 0;
    while ord < entry.terms_dict_size {
        if ord % BLOCK_SIZE == 0 {
            let first_len = input.read_vint()? as usize;
            let mut term = vec![0u8; first_len];
            input.read_bytes(&mut term)?;

            // Only decompress a block body if more terms remain after this
            // block's first term (mirrors Java's `decompressBlock`, which
            // skips this when the first term is the very last one written).
            if ord + 1 < entry.terms_dict_size {
                let block_len = input.read_vint()? as usize;
                let mut buffer = vec![0u8; term.len() + block_len];
                buffer[..term.len()].copy_from_slice(&term);
                lz4::decompress(&mut input, block_len, &mut buffer, term.len())?;
                block_body = buffer[term.len()..].to_vec();
            } else {
                block_body.clear();
            }
            block_pos = 0;
            terms.push(term.clone());
            previous = term;
        } else {
            let token = read_u8(&block_body, &mut block_pos)? as usize;
            let mut prefix_len = token & 0x0F;
            let mut suffix_len = 1 + (token >> 4);
            if prefix_len == 15 {
                prefix_len += read_vint(&block_body, &mut block_pos)? as usize;
            }
            if suffix_len == 16 {
                suffix_len += read_vint(&block_body, &mut block_pos)? as usize;
            }
            let mut term = Vec::with_capacity(prefix_len + suffix_len);
            term.extend_from_slice(&previous[..prefix_len]);
            let suffix = block_body
                .get(block_pos..block_pos + suffix_len)
                .ok_or(lucene_store::Error::Eof { offset: block_pos })?;
            term.extend_from_slice(suffix);
            block_pos += suffix_len;
            terms.push(term.clone());
            previous = term;
        }
        ord += 1;
    }

    Ok(terms)
}

fn read_u8(buf: &[u8], pos: &mut usize) -> Result<u8> {
    let b = *buf
        .get(*pos)
        .ok_or(lucene_store::Error::Eof { offset: *pos })?;
    *pos += 1;
    Ok(b)
}

/// Plain vint decode over a `&[u8]` cursor (the block body isn't a
/// `DataInput` -- see the comment on `block_body`/`block_pos` above).
fn read_vint(buf: &[u8], pos: &mut usize) -> Result<i32> {
    let mut b = read_u8(buf, pos)?;
    let mut v = (b & 0x7f) as i32;
    let mut shift = 7;
    while b & 0x80 != 0 {
        b = read_u8(buf, pos)?;
        v |= ((b & 0x7f) as i32) << shift;
        shift += 7;
    }
    Ok(v)
}

#[cfg(test)]
mod tests {
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
            terms_data_offset: 0,
            terms_data_length: data.len() as i64,
        };
        let terms = decode_all_terms(&data, &entry).unwrap();
        assert_eq!(terms[0], previous);
        let mut expected_second = previous[..17].to_vec();
        expected_second.extend_from_slice(&suffix);
        assert_eq!(terms[1], expected_second);
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
        assert_eq!(entry.terms_data_offset, 1000);
        assert_eq!(entry.terms_data_length, 2000);
        assert_eq!(input.remaining(), 0);
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
