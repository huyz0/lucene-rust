//! Port of `org.apache.lucene.codecs.lucene90.Lucene90CompoundFormat`
//! (`.cfs` data + `.cfe` entries) — read-only.
//!
//! A compound file packs every other file belonging to a segment (`.si`,
//! `.fnm`, `.dvd`, ...) into one blob (`.cfs`) plus a small entries table
//! (`.cfe`) mapping each sub-file's name to an `(offset, length)` slice —
//! purely a concatenation-with-directory, no compression. Simpler than most
//! formats in this port: no `IndexedDISI`, no bit-packing, just one flat
//! header/footer-framed table.
//!
//! `.cfe`:
//! ```text
//! IndexHeader(codec="Lucene90CompoundEntries", version, id, suffix="")
//! NumEntries    --> vint
//! per entry:
//!   Id          --> string   (sub-file name, without the segment-name prefix)
//!   Offset      --> i64      (into .cfs)
//!   Length      --> i64
//! Footer
//! ```
//!
//! `.cfs` is just `IndexHeader(codec="Lucene90CompoundData", ...), <concatenated
//! sub-file bytes>, Footer`. Java cross-checks the two files' versions must be
//! *exactly* equal (not just both in range) and that `.cfs`'s length matches
//! what the entries table implies, plus the footer — both of which this port
//! preserves since they're cheap, real corruption checks.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

const DATA_CODEC: &str = "Lucene90CompoundData";
const ENTRY_CODEC: &str = "Lucene90CompoundEntries";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = VERSION_START;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("duplicate cfs entry id: {0}")]
    DuplicateEntry(String),
    #[error("data file length should be {expected} bytes, but is {actual}")]
    WrongLength { expected: usize, actual: usize },
    #[error("no sub-file with id {0} found in compound file (files: {1:?})")]
    FileNotFound(String, Vec<String>),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy)]
pub struct FileEntry {
    pub offset: i64,
    pub length: i64,
}

#[derive(Debug, Clone)]
pub struct CompoundEntries {
    pub version: i32,
    pub entries: Vec<(String, FileEntry)>,
}

impl CompoundEntries {
    pub fn get(&self, id: &str) -> Option<&FileEntry> {
        self.entries
            .iter()
            .find(|(name, _)| name == id)
            .map(|(_, e)| e)
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.iter().map(|(name, _)| name.as_str())
    }
}

/// Parses a whole `.cfe` entries file already read into memory.
pub fn parse_entries(buf: &[u8], segment_id: &[u8; ID_LENGTH]) -> Result<CompoundEntries> {
    let mut input = SliceInput::new(buf);
    let header = codec_util::check_index_header(
        &mut input,
        ENTRY_CODEC,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        "",
    )?;

    let num_entries = input.read_vint()?;
    let mut entries = Vec::with_capacity(num_entries.max(0) as usize);
    for _ in 0..num_entries {
        let id = input.read_string()?;
        if entries
            .iter()
            .any(|(name, _): &(String, FileEntry)| *name == id)
        {
            return Err(Error::DuplicateEntry(id));
        }
        let offset = input.read_i64()?;
        let length = input.read_i64()?;
        entries.push((id, FileEntry { offset, length }));
    }

    codec_util::check_footer(&mut input, buf.len())?;

    Ok(CompoundEntries {
        version: header.version,
        entries,
    })
}

/// Validates a whole `.cfs` data file's header/footer/length against the
/// already-parsed entries table. Unlike most `.dvd`/`.nvd`-style data files
/// in this port, Java requires the `.cfs` version to *exactly* match the
/// `.cfe` version (not just fall in the supported range, enforced here by
/// passing `entries.version` as both the min and max) and cross-checks the
/// file's total length against what the entries table implies -- both
/// preserved here since they catch real truncation/corruption cheaply.
pub fn check_data_header_footer(
    buf: &[u8],
    segment_id: &[u8; ID_LENGTH],
    entries: &CompoundEntries,
) -> Result<()> {
    let mut input = SliceInput::new(buf);
    codec_util::check_index_header(
        &mut input,
        DATA_CODEC,
        entries.version,
        entries.version,
        segment_id,
        "",
    )?;
    codec_util::retrieve_checksum(buf)?;

    let max_extent = entries
        .entries
        .iter()
        .map(|(_, e)| (e.offset + e.length) as usize)
        .max()
        .unwrap_or_else(|| index_header_length(DATA_CODEC));
    let expected_length = max_extent + codec_util::FOOTER_LENGTH;
    if buf.len() != expected_length {
        return Err(Error::WrongLength {
            expected: expected_length,
            actual: buf.len(),
        });
    }

    Ok(())
}

/// Slices out a sub-file's bytes from the whole `.cfs` file's bytes. `id` is
/// the sub-file name without the segment-name prefix (Java's
/// `IndexFileNames.stripSegmentName`), e.g. `.fnm` rather than `_0.fnm`.
pub fn open_input<'d>(data: &'d [u8], entries: &CompoundEntries, id: &str) -> Result<&'d [u8]> {
    let entry = entries.get(id).ok_or_else(|| {
        Error::FileNotFound(
            id.to_string(),
            entries.names().map(str::to_string).collect(),
        )
    })?;
    data.get(entry.offset as usize..(entry.offset + entry.length) as usize)
        .ok_or_else(|| {
            lucene_store::Error::Eof {
                offset: entry.offset as usize,
            }
            .into()
        })
}

/// `header_length(codec) + ID_LENGTH + 1` (the vint-encoded empty-suffix
/// length byte) -- Java's `CodecUtil.indexHeaderLength(codec, "")`, used as
/// the minimum expected `.cfs` length when the entries table is empty.
fn index_header_length(codec: &str) -> usize {
    4 + vint_len(codec.len() as i32) + codec.len() + 4 + ID_LENGTH + 1
}

fn vint_len(mut v: i32) -> usize {
    let mut n = 1;
    while (v as u32) >= 0x80 {
        v = ((v as u32) >> 7) as i32;
        n += 1;
    }
    n
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

    fn write_string(out: &mut Vec<u8>, s: &str) {
        write_vint(out, s.len() as i32);
        out.extend_from_slice(s.as_bytes());
    }

    fn build_cfe(id: &[u8; ID_LENGTH], entries: &[(&str, i64, i64)]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, ENTRY_CODEC);
        out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
        out.extend_from_slice(id);
        out.push(0); // empty suffix
        write_vint(&mut out, entries.len() as i32);
        for (name, offset, length) in entries {
            write_string(&mut out, name);
            out.extend_from_slice(&offset.to_le_bytes());
            out.extend_from_slice(&length.to_le_bytes());
        }
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }

    fn build_cfs(id: &[u8; ID_LENGTH], version: i32, payload: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
        write_string(&mut out, DATA_CODEC);
        out.extend_from_slice(&(version as u32).to_be_bytes());
        out.extend_from_slice(id);
        out.push(0);
        out.extend_from_slice(payload);
        out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
        out.extend_from_slice(&0u32.to_be_bytes());
        let checksum = crc32fast::hash(&out) as u64;
        out.extend_from_slice(&checksum.to_be_bytes());
        out
    }

    #[test]
    fn empty_entries_parses() {
        let id = [1u8; ID_LENGTH];
        let buf = build_cfe(&id, &[]);
        let entries = parse_entries(&buf, &id).unwrap();
        assert_eq!(entries.version, VERSION_CURRENT);
        assert_eq!(entries.entries.len(), 0);
    }

    #[test]
    fn duplicate_entry_rejected() {
        let id = [1u8; ID_LENGTH];
        let buf = build_cfe(&id, &[(".fnm", 0, 10), (".fnm", 10, 5)]);
        assert!(matches!(
            parse_entries(&buf, &id),
            Err(Error::DuplicateEntry(name)) if name == ".fnm"
        ));
    }

    #[test]
    fn wrong_id_rejected() {
        let id = [1u8; ID_LENGTH];
        let buf = build_cfe(&id, &[]);
        let wrong_id = [2u8; ID_LENGTH];
        assert!(matches!(
            parse_entries(&buf, &wrong_id),
            Err(Error::Store(_))
        ));
    }

    #[test]
    fn open_input_slices_correct_bytes() {
        let id = [1u8; ID_LENGTH];
        let header_len = index_header_length(DATA_CODEC);
        let payload = b"hello world!";
        let cfe = build_cfe(
            &id,
            &[
                (".fnm", header_len as i64, 5),
                (".si", header_len as i64 + 5, 7),
            ],
        );
        let entries = parse_entries(&cfe, &id).unwrap();
        let cfs = build_cfs(&id, VERSION_CURRENT, payload);

        assert_eq!(open_input(&cfs, &entries, ".fnm").unwrap(), b"hello");
        assert_eq!(open_input(&cfs, &entries, ".si").unwrap(), b" world!");
    }

    #[test]
    fn open_input_unknown_id_rejected() {
        let id = [1u8; ID_LENGTH];
        let entries = parse_entries(&build_cfe(&id, &[]), &id).unwrap();
        let cfs = build_cfs(&id, VERSION_CURRENT, b"");
        assert!(matches!(
            open_input(&cfs, &entries, ".nope"),
            Err(Error::FileNotFound(name, _)) if name == ".nope"
        ));
    }

    #[test]
    fn check_data_header_footer_valid_with_entries() {
        let id = [1u8; ID_LENGTH];
        let header_len = index_header_length(DATA_CODEC);
        let payload = b"abcdefg";
        let cfe = build_cfe(&id, &[(".fnm", header_len as i64, payload.len() as i64)]);
        let entries = parse_entries(&cfe, &id).unwrap();
        let cfs = build_cfs(&id, VERSION_CURRENT, payload);
        check_data_header_footer(&cfs, &id, &entries).unwrap();
    }

    #[test]
    fn check_data_header_footer_valid_with_no_entries() {
        let id = [1u8; ID_LENGTH];
        let entries = parse_entries(&build_cfe(&id, &[]), &id).unwrap();
        let cfs = build_cfs(&id, VERSION_CURRENT, b"");
        check_data_header_footer(&cfs, &id, &entries).unwrap();
    }

    #[test]
    fn check_data_header_footer_wrong_length_rejected() {
        let id = [1u8; ID_LENGTH];
        let header_len = index_header_length(DATA_CODEC);
        let cfe = build_cfe(&id, &[(".fnm", header_len as i64, 100)]); // implies far more data than exists
        let entries = parse_entries(&cfe, &id).unwrap();
        let cfs = build_cfs(&id, VERSION_CURRENT, b"short");
        assert!(matches!(
            check_data_header_footer(&cfs, &id, &entries),
            Err(Error::WrongLength { .. })
        ));
    }

    #[test]
    fn check_data_header_footer_wrong_id_rejected() {
        let id = [1u8; ID_LENGTH];
        let entries = parse_entries(&build_cfe(&id, &[]), &id).unwrap();
        let cfs = build_cfs(&id, VERSION_CURRENT, b"");
        let wrong_id = [2u8; ID_LENGTH];
        assert!(matches!(
            check_data_header_footer(&cfs, &wrong_id, &entries),
            Err(Error::Store(_))
        ));
    }

    #[test]
    fn open_input_entry_offset_past_end_of_data_is_error() {
        let id = [1u8; ID_LENGTH];
        // Entry claims bytes far beyond what the (short) data buffer holds.
        let cfe = build_cfe(&id, &[(".fnm", 1000, 5)]);
        let entries = parse_entries(&cfe, &id).unwrap();
        let cfs = build_cfs(&id, VERSION_CURRENT, b"short");
        assert!(matches!(
            open_input(&cfs, &entries, ".fnm"),
            Err(Error::Store(_))
        ));
    }

    #[test]
    fn vint_len_multi_byte_codec_name() {
        // Widths for a few known vint boundaries: 127 fits in 1 byte, 128
        // needs 2, 16384 needs 3 -- exercises `vint_len`'s continuation loop,
        // which real codec names (all short ASCII constants) never trigger.
        assert_eq!(vint_len(0), 1);
        assert_eq!(vint_len(127), 1);
        assert_eq!(vint_len(128), 2);
        assert_eq!(vint_len(16_383), 2);
        assert_eq!(vint_len(16_384), 3);
    }
}
