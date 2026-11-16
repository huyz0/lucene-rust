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
    // Compared in place: a match is the common case, and needs no copy.
    let actual_codec = read_string_bytes(input)?;
    if actual_codec != expected_codec.as_bytes() {
        let actual_codec = std::str::from_utf8(actual_codec)
            .map_err(|_| Error::Corrupted("invalid UTF-8 string".into()))?;
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

/// `DataInput.readString`'s bytes, borrowed from the input rather than
/// copied: the same length checks, no UTF-8 check (the caller compares).
fn read_string_bytes<'a>(input: &mut SliceInput<'a>) -> Result<&'a [u8]> {
    let len = input.read_length("string")?;
    let start = input.position();
    let end = start.saturating_add(len);
    let bytes = input.slice(start, end)?;
    input.seek(end)?;
    Ok(bytes)
}

/// The codec name a header-carrying file starts with, without validating
/// anything past it -- `None` when the file does not start with
/// [`CODEC_MAGIC`] and a readable string.
///
/// Not a Java method: Lucene always knows which format to expect, because
/// the codec (named in `segments_N`) says so. A backward-codecs reader that
/// serves several generations of one component from one call site -- `.fnm`
/// written as `Lucene90FieldInfos` or `Lucene94FieldInfos`, postings files as
/// `Lucene90PostingsWriterDoc` .. `Lucene104PostingsWriterDoc` -- instead
/// dispatches on the name the file itself carries, and then runs the full
/// [`check_index_header`] for the generation it picked.
pub fn peek_codec_name(buf: &[u8]) -> Option<String> {
    let mut input = SliceInput::new(buf);
    if input.read_be_u32().ok()? != CODEC_MAGIC {
        return None;
    }
    input.read_string().ok()
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
    let start = input.position();
    let bytes = input.slice(start, start.saturating_add(len))?;
    input.seek(start.saturating_add(len))?;
    if bytes == expected_suffix.as_bytes() {
        return Ok(expected_suffix.to_string());
    }
    let suffix = std::str::from_utf8(bytes).map_err(|_| corrupt("invalid UTF-8 suffix"))?;
    Err(corrupt(format!(
        "file mismatch: suffix={suffix} vs expected suffix={expected_suffix}"
    )))
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
    // Assembled in one buffer and written once: a header usually opens an
    // empty buffer, which would otherwise regrow at every field of it. The
    // stack holds every header the `debug_assert`s allow.
    const MAX: usize = 4 + 5 + 127 + 4 + ID_LENGTH + 1 + 255;
    let mut stack = [0u8; MAX];
    let mut heap = Vec::new();
    let len = header_len(codec, suffix);
    let buf: &mut [u8] = if len <= MAX {
        &mut stack
    } else {
        heap.resize(len, 0);
        &mut heap
    };
    let mut cursor = HeaderCursor { buf, len: 0 };
    write_header(&mut cursor, codec, version);
    cursor.write_bytes(id);
    cursor.write_byte(suffix.len() as u8);
    cursor.write_bytes(suffix.as_bytes());
    let HeaderCursor { buf, len } = cursor;
    out.write_bytes(&buf[..len]);
}

/// The bytes [`write_index_header`] writes, the codec's length vint taken
/// at its widest.
fn header_len(codec: &str, suffix: &str) -> usize {
    [4, 5, codec.len(), 4, ID_LENGTH, 1, suffix.len()]
        .iter()
        .fold(0usize, |a, &b| a.saturating_add(b))
}

/// [`write_index_header`]'s buffer, as a [`DataOutput`]: sized by
/// [`header_len`] for the header it receives.
struct HeaderCursor<'a> {
    buf: &'a mut [u8],
    len: usize,
}

impl DataOutput for HeaderCursor<'_> {
    fn write_byte(&mut self, b: u8) {
        self.write_bytes(&[b]);
    }

    fn write_bytes(&mut self, b: &[u8]) {
        // `header_len` sized the buffer for every write the header makes
        let end = self.len.saturating_add(b.len());
        self.buf[self.len..end].copy_from_slice(b);
        self.len = end;
    }
}

/// Below this many bytes [`crc32`] hashes on the calling thread: a thread
/// spawn costs more than hashing a few MiB at crc32fast's PCLMULQDQ speed.
const PARALLEL_CRC_MIN: usize = 8 << 20;

/// CRC-32 of `buf`, bit-identical to `crc32fast::hash` (and to Java's
/// `java.util.zip.CRC32`), split across the available cores for a large
/// buffer: each thread hashes one contiguous piece and the partial CRCs are
/// folded together in order with [`crc32fast::Hasher::combine`]. A footer
/// check over a multi-GiB `.fdt`/`.vec` file (every `checkIntegrity` before a
/// merge, and every [`write_footer`]) otherwise runs on one core.
pub fn crc32(buf: &[u8]) -> u32 {
    // The small-buffer test comes first: nearly every footer is tiny, and
    // `available_parallelism` reads cgroup files on Linux on each call, so
    // its answer is cached.
    if buf.len() < PARALLEL_CRC_MIN {
        return crc32fast::hash(buf);
    }
    static THREADS: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    let threads =
        *THREADS.get_or_init(|| std::thread::available_parallelism().map_or(1, |n| n.get()));
    let pieces = threads.min(buf.len() / (PARALLEL_CRC_MIN / 2)).max(1);
    if pieces == 1 {
        return crc32fast::hash(buf);
    }
    let piece = buf.len().div_ceil(pieces);
    let partials: Vec<crc32fast::Hasher> = std::thread::scope(|scope| {
        let handles: Vec<_> = buf
            .chunks(piece)
            .map(|chunk| {
                scope.spawn(move || {
                    let mut h = crc32fast::Hasher::new();
                    h.update(chunk);
                    h
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("crc32 worker panicked"))
            .collect()
    });
    let mut iter = partials.into_iter();
    let mut acc = iter.next().unwrap_or_default();
    for h in iter {
        acc.combine(&h);
    }
    acc.finalize()
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
    let checksum = crc32(buf) as u64;
    buf.write_be_u64(checksum);
}

/// [`write_footer`] on a streaming output, its checksum the output's running
/// CRC over every byte written to it.
pub fn write_footer_to(out: &mut crate::index_output::FsIndexOutput) {
    use crate::index_output::IndexOutput;
    out.write_bytes(&FOOTER_MAGIC.to_be_bytes());
    out.write_bytes(&0u32.to_be_bytes());
    let checksum = out.checksum();
    out.write_bytes(&checksum.to_be_bytes());
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
    let actual_checksum = crc32(covered) as u64;

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
mod tests;
