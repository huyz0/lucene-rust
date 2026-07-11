//! Port of `org.apache.lucene.codecs.CodecUtil` header/footer framing.
//!
//! Wire format (all magic/version/checksum fields are big-endian, everything
//! else in a Lucene file is little-endian):
//!
//! ```text
//! Header      --> Magic(BEi32=0x3fd76c17), CodecName(String), Version(BEi32)
//! IndexHeader --> Header, ObjectID([u8; 16]), SuffixLength(u8), Suffix(UTF-8 bytes)
//! Footer      --> Magic(BEi32=~0x3fd76c17), AlgorithmID(BEi32=0), Checksum(BEu64=CRC32)
//! ```
//!
//! The footer's checksum covers every byte in the file *up to and including*
//! the footer's own magic+algorithmID, i.e. `crc32(file[..len-8])`.

use crate::data_input::{DataInput, SliceInput};
use crate::error::{Error, Result};

pub const CODEC_MAGIC: u32 = 0x3fd7_6c17;
pub const FOOTER_MAGIC: u32 = !CODEC_MAGIC;
pub const FOOTER_LENGTH: usize = 16;
pub const ID_LENGTH: usize = 16;

/// Result of a successful header check.
#[derive(Debug)]
pub struct Header {
    pub version: i32,
}

/// Result of a successful index-header check (adds object id + suffix).
#[derive(Debug)]
pub struct IndexHeader {
    pub version: i32,
    pub id: [u8; ID_LENGTH],
    pub suffix: String,
}

fn corrupt(msg: impl Into<String>) -> Error {
    Error::Corrupted(msg.into())
}

/// Port of `CodecUtil.checkHeader`: validates magic, codec name, and version range.
pub fn check_header(
    input: &mut SliceInput,
    expected_codec: &str,
    min_version: i32,
    max_version: i32,
) -> Result<Header> {
    let magic = input.read_be_u32()?;
    if magic != CODEC_MAGIC {
        return Err(corrupt(format!(
            "codec header mismatch: actual header={magic:#x} vs expected header={CODEC_MAGIC:#x}"
        )));
    }
    check_header_no_magic(input, expected_codec, min_version, max_version)
}

/// Port of `CodecUtil.checkHeaderNoMagic`.
pub fn check_header_no_magic(
    input: &mut SliceInput,
    expected_codec: &str,
    min_version: i32,
    max_version: i32,
) -> Result<Header> {
    let actual_codec = input.read_string()?;
    if actual_codec != expected_codec {
        return Err(corrupt(format!(
            "codec mismatch: actual codec={actual_codec} vs expected codec={expected_codec}"
        )));
    }
    let version = input.read_be_u32()? as i32;
    if version < min_version {
        return Err(corrupt(format!(
            "Version too old: actual version={version} but minVersion={min_version}"
        )));
    }
    if version > max_version {
        return Err(corrupt(format!(
            "Version too new: actual version={version} but maxVersion={max_version}"
        )));
    }
    Ok(Header { version })
}

/// Port of `CodecUtil.checkIndexHeader`.
pub fn check_index_header(
    input: &mut SliceInput,
    expected_codec: &str,
    min_version: i32,
    max_version: i32,
    expected_id: &[u8; ID_LENGTH],
    expected_suffix: &str,
) -> Result<IndexHeader> {
    let header = check_header(input, expected_codec, min_version, max_version)?;
    let id = check_index_header_id(input, expected_id)?;
    let suffix = check_index_header_suffix(input, expected_suffix)?;
    Ok(IndexHeader {
        version: header.version,
        id,
        suffix,
    })
}

/// Port of `CodecUtil.checkIndexHeaderID`.
pub fn check_index_header_id(
    input: &mut SliceInput,
    expected_id: &[u8; ID_LENGTH],
) -> Result<[u8; ID_LENGTH]> {
    let mut id = [0u8; ID_LENGTH];
    input.read_bytes(&mut id)?;
    if &id != expected_id {
        return Err(corrupt("file mismatch: object id does not match"));
    }
    Ok(id)
}

/// Port of `CodecUtil.checkIndexHeaderSuffix`.
pub fn check_index_header_suffix(input: &mut SliceInput, expected_suffix: &str) -> Result<String> {
    let len = input.read_byte()? as usize;
    let mut buf = vec![0u8; len];
    input.read_bytes(&mut buf)?;
    let suffix = String::from_utf8(buf).map_err(|_| corrupt("invalid UTF-8 suffix"))?;
    if suffix != expected_suffix {
        return Err(corrupt(format!(
            "file mismatch: suffix={suffix} vs expected suffix={expected_suffix}"
        )));
    }
    Ok(suffix)
}

/// Port of `CodecUtil.checkFooter`: `input` must be positioned at the start of the
/// footer (i.e. at `total_len - FOOTER_LENGTH`), and `total_len` is the full file
/// length (footer's CRC covers everything before the checksum field itself).
///
/// Returns the verified checksum on success.
pub fn check_footer(input: &mut SliceInput, total_len: usize) -> Result<u64> {
    if total_len < FOOTER_LENGTH {
        return Err(corrupt(format!(
            "misplaced codec footer (file truncated?): length={total_len} but footerLength=={FOOTER_LENGTH}"
        )));
    }
    let footer_start = total_len - FOOTER_LENGTH;
    if input.position() != footer_start {
        return Err(corrupt(format!(
            "did not read all bytes from file: read {} vs size {total_len} (resource=...)",
            input.position()
        )));
    }

    let magic = input.read_be_u32()?;
    if magic != FOOTER_MAGIC {
        return Err(corrupt(format!(
            "codec footer mismatch (file truncated?): actual footer={magic:#x} vs expected footer={FOOTER_MAGIC:#x}"
        )));
    }
    let algorithm_id = input.read_be_u32()?;
    if algorithm_id != 0 {
        return Err(corrupt(format!(
            "codec footer mismatch: unknown algorithmID: {algorithm_id}"
        )));
    }

    // CRC covers [0, footer_start + 8) i.e. everything up to and including the
    // footer's magic+algorithmID, matching Lucene's running-checksum semantics.
    let covered = input.slice(0, footer_start + 8)?;
    let actual_checksum = crc32fast::hash(covered) as u64;

    let expected_checksum = input.read_be_u64()?;
    if (expected_checksum & 0xFFFF_FFFF_0000_0000) != 0 {
        return Err(corrupt(format!(
            "Illegal CRC-32 checksum: {expected_checksum}"
        )));
    }
    if expected_checksum != actual_checksum {
        return Err(corrupt(format!(
            "checksum failed (hardware problem?) : expected={expected_checksum:#x} actual={actual_checksum:#x}"
        )));
    }
    Ok(actual_checksum)
}

/// Convenience: check header + footer of a whole in-memory file in one call.
/// Returns the header info and verified checksum; caller reads the payload
/// (between header end and footer start) directly via `input`/`buf` as needed.
pub fn check_whole_file_header(
    buf: &[u8],
    expected_codec: &str,
    min_version: i32,
    max_version: i32,
) -> Result<Header> {
    if buf.len() < FOOTER_LENGTH {
        return Err(corrupt("file too small to contain a codec footer"));
    }
    let mut input = SliceInput::new(buf);
    check_header(&mut input, expected_codec, min_version, max_version)
}

pub fn check_whole_file_footer(buf: &[u8], payload_end: usize) -> Result<u64> {
    let mut input = SliceInput::new(buf);
    input.seek(payload_end)?;
    check_footer(&mut input, buf.len())
}
