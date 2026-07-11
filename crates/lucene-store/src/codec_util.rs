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

#[cfg(test)]
mod tests {
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
}
