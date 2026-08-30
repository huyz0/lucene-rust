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
use crate::data_output::DataOutput;
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

/// Port of `CodecUtil.headerLength`: `9 + codec.length()` — 4 magic bytes,
/// the codec name's 1-byte vint length plus its bytes, and 4 version bytes.
/// Only valid because [`write_header`] restricts the codec name to ASCII
/// shorter than 128 bytes, which is exactly why Java enforces that.
// ARITH: `str::len()` is bounded by `isize::MAX`, so `9 + len` cannot wrap
// for *any* `&str`. (Separately, and not what makes the `+` safe: this is
// only a correct byte *count* because `write_header` restricts the codec name
// to ASCII shorter than 128 bytes, which is what keeps its length prefix one
// byte.) Kept as a plain `+` because every codec's seek arithmetic folds
// through this.
#[allow(clippy::arithmetic_side_effects)]
pub fn header_length(codec: &str) -> usize {
    9 + codec.len()
}

/// Port of `CodecUtil.indexHeaderLength`: [`header_length`] plus the 16-byte
/// object id, the suffix's 1-byte length, and the suffix bytes.
// ARITH: as [`header_length`] -- every term is a `str::len()` or a small
// constant, all bounded by `isize::MAX`.
#[allow(clippy::arithmetic_side_effects)]
pub fn index_header_length(codec: &str, suffix: &str) -> usize {
    header_length(codec) + ID_LENGTH + 1 + suffix.len()
}

/// Port of `CodecUtil.writeHeader`: `Magic(BEi32), CodecName(String), Version(BEi32)`.
///
/// Java throws `IllegalArgumentException` unless the codec name is simple
/// ASCII shorter than 128 characters; that bound is what makes
/// [`header_length`] (and every codec that seeks by it) correct, since a
/// longer or non-ASCII name would take a 2-byte vint or more bytes than
/// characters. Codec names are compile-time constants, so this is a caller
/// bug rather than a corrupt-input case: debug-asserted here.
pub fn write_header(out: &mut impl DataOutput, codec: &str, version: i32) {
    debug_assert!(
        codec.is_ascii() && codec.len() < 128,
        "codec must be simple ASCII, less than 128 characters in length [got {codec}]"
    );
    out.write_be_u32(CODEC_MAGIC);
    out.write_string(codec);
    out.write_be_u32(version as u32);
}

/// Port of `CodecUtil.writeIndexHeader`: a [`write_header`] plus the object
/// id and segment suffix. `suffix` must be ASCII and at most 255 bytes
/// (mirrors Java's own `checkIndexHeaderSuffix`/`writeIndexHeader`
/// constraint, since the length is written as a single byte).
pub fn write_index_header(
    out: &mut impl DataOutput,
    codec: &str,
    version: i32,
    id: &[u8; ID_LENGTH],
    suffix: &str,
) {
    debug_assert!(
        suffix.is_ascii() && suffix.len() < 256,
        "suffix must be simple ASCII, less than 256 characters in length [got {suffix}]"
    );
    write_header(out, codec, version);
    out.write_bytes(id);
    out.write_byte(suffix.len() as u8);
    out.write_bytes(suffix.as_bytes());
}

/// Port of `CodecUtil.writeFooter`: `Magic(BEi32=~CODEC_MAGIC), AlgorithmID(BEi32=0),
/// Checksum(BEu64=CRC32)`. Operates directly on the accumulated output buffer
/// (rather than being generic over [`DataOutput`]) since the checksum must
/// cover every byte written so far, including the footer's own magic and
/// algorithm id -- there's no `DataOutput` method to read back what's
/// already been written, so this takes the buffer as `&mut Vec<u8>` and
/// hashes it directly, matching every hand-built test fixture elsewhere in
/// this port.
pub fn write_footer(buf: &mut Vec<u8>) {
    buf.write_be_u32(FOOTER_MAGIC);
    buf.write_be_u32(0); // algorithm id
    let checksum = crc32fast::hash(buf) as u64;
    buf.write_be_u64(checksum);
}

/// Port of `CodecUtil.checkFooter`: `input` must be positioned at the start of the
/// footer (i.e. at `total_len - FOOTER_LENGTH`), and `total_len` is the full file
/// length (footer's CRC covers everything before the checksum field itself).
///
/// Returns the verified checksum on success.
pub fn check_footer(input: &mut SliceInput, total_len: usize) -> Result<u64> {
    // The guard and the subtraction are the same operation: a `total_len`
    // shorter than a footer is exactly the case where `- FOOTER_LENGTH`
    // would wrap.
    let Some(footer_start) = total_len.checked_sub(FOOTER_LENGTH) else {
        return Err(corrupt(format!(
            "misplaced codec footer (file truncated?): length={total_len} but footerLength=={FOOTER_LENGTH}"
        )));
    };
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
    // ARITH: `footer_start == total_len - FOOTER_LENGTH` and
    // `FOOTER_LENGTH == 16`, so `+ 8` cannot reach `usize::MAX`.
    #[allow(clippy::arithmetic_side_effects)]
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

/// Port of `CodecUtil.retrieveChecksum(IndexInput)`: validates that the
/// footer is *structurally* well-formed (magic, algorithm id, checksum field
/// shape) and returns the stored checksum, without recomputing the CRC over
/// the whole file. Cheap; used where a full-file checksum is too costly for
/// a forward-only access pattern (e.g. norms data) but truncation/gross
/// corruption should still be caught on open.
pub fn retrieve_checksum(buf: &[u8]) -> Result<u64> {
    let Some(footer_start) = buf.len().checked_sub(FOOTER_LENGTH) else {
        return Err(corrupt(format!(
            "misplaced codec footer (file truncated?): length={} but footerLength=={FOOTER_LENGTH}",
            buf.len()
        )));
    };
    let mut input = SliceInput::new(buf);
    input.seek(footer_start)?;

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
    let checksum = input.read_be_u64()?;
    if (checksum & 0xFFFF_FFFF_0000_0000) != 0 {
        return Err(corrupt(format!("Illegal CRC-32 checksum: {checksum}")));
    }
    Ok(checksum)
}

/// Port of `CodecUtil.retrieveChecksum(IndexInput, long expectedLength)`: the
/// same structural footer check as [`retrieve_checksum`], plus the file-length
/// assertion Lucene makes when a caller already knows how long the file must
/// be (a truncated *or* over-long file is corrupt, and a truncated file whose
/// tail happens to look like a footer would otherwise slip through).
pub fn retrieve_checksum_with_expected_length(buf: &[u8], expected_length: usize) -> Result<u64> {
    if expected_length < FOOTER_LENGTH {
        return Err(corrupt(format!(
            "expectedLength cannot be less than the footer length: {expected_length}"
        )));
    }
    if buf.len() < expected_length {
        return Err(corrupt(format!(
            "truncated file: length={} but expectedLength=={expected_length}",
            buf.len()
        )));
    }
    if buf.len() > expected_length {
        return Err(corrupt(format!(
            "file too long: length={} but expectedLength=={expected_length}",
            buf.len()
        )));
    }
    retrieve_checksum(buf)
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
}
