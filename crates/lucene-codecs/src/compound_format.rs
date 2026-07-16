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
use lucene_store::data_output::DataOutput;

/// Sub-files are aligned to a 64-byte boundary within `.cfs` (the LCM of every
/// individual format's own alignment, so this guarantees each of those holds
/// too) -- purely a placement nicety for mmap access patterns, invisible to
/// [`open_input`], which slices by the recorded `(offset, length)` regardless
/// of any padding gaps.
const ALIGNMENT_BYTES: usize = 64;

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
    #[error("sub-file {0} is not a valid codec file (bad magic)")]
    BadSubFileMagic(String),
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

/// Opens a sub-file as its own independent-file-pointer [`SliceInput`],
/// addressed from 0 like a standalone file. `id` is the sub-file name without
/// the segment-name prefix (Java's `IndexFileNames.stripSegmentName`), e.g.
/// `.fnm` rather than `_0.fnm`.
///
/// This is `IndexInput.slice(sliceDescription, offset, length)` (see
/// `lucene_store::data_input::SliceInput::slice_input`) rather than
/// hand-rolled offset math: the real caller this generalizes for is segment
/// merging (task #15), which needs to read a sub-range of a shared `.cfs`
/// with its own cursor per source segment, independent of any other open
/// sub-file reader over the same bytes.
pub fn open_input<'d>(
    data: &'d [u8],
    entries: &CompoundEntries,
    id: &str,
) -> Result<SliceInput<'d>> {
    let entry = entries.get(id).ok_or_else(|| {
        Error::FileNotFound(
            id.to_string(),
            entries.names().map(str::to_string).collect(),
        )
    })?;
    SliceInput::new(data)
        .slice_input(id, entry.offset as u64, entry.length as u64)
        .map_err(Into::into)
}

/// Packs already-written sub-files (each a *complete* standalone codec file:
/// its own `IndexHeader` sharing `segment_id`, a body, and its own `Footer`)
/// into a `.cfs`/`.cfe` pair, returning `(cfs_bytes, cfe_bytes)`.
///
/// `sub_files` pairs each entry's stripped id (e.g. `.fnm`, the name recorded
/// in `.cfe` -- without the segment-name prefix, matching what
/// [`parse_entries`]/[`open_input`] expect) with that sub-file's complete
/// bytes. Mirrors Java's `Lucene90CompoundFormat.writeCompoundFile`:
///
/// - Sub-files are packed smallest-first (`Comparator.comparingLong(length)`)
///   so small files are more likely to land within one page.
/// - Each sub-file's start offset in `.cfs` is padded up to a 64-byte
///   boundary ([`ALIGNMENT_BYTES`]).
/// - Java "verifies and copies" each sub-file's header (checking its object
///   id matches `segment_id`) and re-derives its footer from the verified
///   checksum rather than copying the footer bytes directly -- but since that
///   checksum is exactly the value already stored in the sub-file's own
///   footer, the net effect is byte-identical to copying the whole sub-file
///   verbatim, which is what this port does after validating the header id
///   and footer checksum up front.
///
/// Returns [`Error::BadSubFileMagic`] or a wrapped [`lucene_store::Error`] if
/// a sub-file's header id doesn't match `segment_id` or its footer checksum
/// doesn't verify -- both real corruption checks Java performs before trusting
/// the bytes it's about to pack.
pub fn write(
    segment_id: &[u8; ID_LENGTH],
    sub_files: &[(String, Vec<u8>)],
) -> Result<(Vec<u8>, Vec<u8>)> {
    let mut ordered: Vec<&(String, Vec<u8>)> = sub_files.iter().collect();
    ordered.sort_by_key(|(_, bytes)| bytes.len());

    let mut cfs: Vec<u8> = Vec::new();
    codec_util::write_index_header(&mut cfs, DATA_CODEC, VERSION_CURRENT, segment_id, "");

    let mut cfe: Vec<u8> = Vec::new();
    codec_util::write_index_header(&mut cfe, ENTRY_CODEC, VERSION_CURRENT, segment_id, "");
    cfe.write_vint(ordered.len() as i32);

    for (name, bytes) in ordered {
        verify_sub_file(name, bytes, segment_id)?;

        while !cfs.len().is_multiple_of(ALIGNMENT_BYTES) {
            cfs.push(0);
        }
        let start_offset = cfs.len() as i64;
        cfs.extend_from_slice(bytes);
        let length = cfs.len() as i64 - start_offset;

        cfe.write_string(name);
        cfe.write_i64(start_offset);
        cfe.write_i64(length);
    }

    codec_util::write_footer(&mut cfs);
    codec_util::write_footer(&mut cfe);

    Ok((cfs, cfe))
}

/// Validates a sub-file's `IndexHeader` object id (must match `segment_id`,
/// though the codec name/version/suffix are whatever that sub-file's own
/// format uses and aren't constrained here) and its `Footer` checksum, port
/// of the checks `CodecUtil.verifyAndCopyIndexHeader`/`CodecUtil.checkFooter`
/// perform in Java's `writeCompoundFile` before trusting a sub-file's bytes.
fn verify_sub_file(name: &str, bytes: &[u8], segment_id: &[u8; ID_LENGTH]) -> Result<()> {
    let mut input = SliceInput::new(bytes);
    let magic = input.read_be_u32()?;
    if magic != codec_util::CODEC_MAGIC {
        return Err(Error::BadSubFileMagic(name.to_string()));
    }
    let _codec_name = input.read_string()?;
    let _version = input.read_be_u32()?;
    codec_util::check_index_header_id(&mut input, segment_id)?;
    let suffix_len = input.read_byte()? as usize;
    input.skip(suffix_len)?;

    let footer_start =
        bytes
            .len()
            .checked_sub(codec_util::FOOTER_LENGTH)
            .ok_or(lucene_store::Error::Eof {
                offset: bytes.len(),
            })?;
    input.seek(footer_start)?;
    codec_util::check_footer(&mut input, bytes.len())?;

    Ok(())
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

        assert_eq!(
            open_input(&cfs, &entries, ".fnm").unwrap().as_slice(),
            b"hello"
        );
        assert_eq!(
            open_input(&cfs, &entries, ".si").unwrap().as_slice(),
            b" world!"
        );
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

    /// Builds a standalone codec file (header + body + footer) as a real
    /// sub-file `write()` would consume -- the same shape produced by e.g.
    /// `field_infos::write`/`stored_fields::write_best_speed`.
    fn build_sub_file(codec: &str, version: i32, id: &[u8; ID_LENGTH], body: &[u8]) -> Vec<u8> {
        let mut out = Vec::new();
        codec_util::write_index_header(&mut out, codec, version, id, "");
        out.extend_from_slice(body);
        codec_util::write_footer(&mut out);
        out
    }

    #[test]
    fn write_round_trips_through_reader_with_multiple_sub_files() {
        let id = [42u8; ID_LENGTH];
        // Three distinct sub-files of different sizes/codecs so both the
        // ascending-size ordering and the alignment/offset math are actually
        // exercised (a single sub-file would leave that logic unchecked).
        let fnm = build_sub_file("FieldInfos", 1, &id, b"field infos body bytes");
        let fdt = build_sub_file("StoredFields", 3, &id, &[7u8; 200]);
        let dvd = build_sub_file("DocValues", 0, &id, b"dv");

        let sub_files = vec![
            (".fnm".to_string(), fnm.clone()),
            (".fdt".to_string(), fdt.clone()),
            (".dvd".to_string(), dvd.clone()),
        ];

        let (cfs, cfe) = write(&id, &sub_files).unwrap();

        let entries = parse_entries(&cfe, &id).unwrap();
        assert_eq!(entries.entries.len(), 3);
        check_data_header_footer(&cfs, &id, &entries).unwrap();

        // Every sub-file must come back byte-for-byte identical, including
        // its own header and footer -- a corrupted offset could otherwise
        // still leave the entries table "looking right" while shifting bytes.
        assert_eq!(
            open_input(&cfs, &entries, ".fnm").unwrap().as_slice(),
            fnm.as_slice()
        );
        assert_eq!(
            open_input(&cfs, &entries, ".fdt").unwrap().as_slice(),
            fdt.as_slice()
        );
        assert_eq!(
            open_input(&cfs, &entries, ".dvd").unwrap().as_slice(),
            dvd.as_slice()
        );

        // Ascending-size packing: .dvd (smallest) should be placed before
        // .fnm, which should be placed before .fdt (largest).
        let dvd_offset = entries.get(".dvd").unwrap().offset;
        let fnm_offset = entries.get(".fnm").unwrap().offset;
        let fdt_offset = entries.get(".fdt").unwrap().offset;
        assert!(dvd_offset < fnm_offset);
        assert!(fnm_offset < fdt_offset);

        // Every sub-file start offset is 64-byte aligned.
        for (_, entry) in &entries.entries {
            assert_eq!(entry.offset % ALIGNMENT_BYTES as i64, 0);
        }
    }

    #[test]
    fn write_empty_sub_files_round_trips() {
        let id = [9u8; ID_LENGTH];
        let (cfs, cfe) = write(&id, &[]).unwrap();
        let entries = parse_entries(&cfe, &id).unwrap();
        assert_eq!(entries.entries.len(), 0);
        check_data_header_footer(&cfs, &id, &entries).unwrap();
    }

    #[test]
    fn write_rejects_sub_file_with_wrong_segment_id() {
        let id = [1u8; ID_LENGTH];
        let wrong_id = [2u8; ID_LENGTH];
        let sub_file = build_sub_file("FieldInfos", 1, &wrong_id, b"body");
        let err = write(&id, &[(".fnm".to_string(), sub_file)]).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn write_rejects_sub_file_with_bad_magic() {
        let id = [1u8; ID_LENGTH];
        let mut sub_file = build_sub_file("FieldInfos", 1, &id, b"body");
        sub_file[0] ^= 0xFF; // corrupt the magic
        let err = write(&id, &[(".fnm".to_string(), sub_file)]).unwrap_err();
        assert!(matches!(err, Error::BadSubFileMagic(name) if name == ".fnm"));
    }

    #[test]
    fn write_rejects_sub_file_with_corrupt_footer_checksum() {
        let id = [1u8; ID_LENGTH];
        let mut sub_file = build_sub_file("FieldInfos", 1, &id, b"body");
        let last = sub_file.len() - 1;
        sub_file[last] ^= 0xFF; // corrupt checksum field
        let err = write(&id, &[(".fnm".to_string(), sub_file)]).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn write_rejects_sub_file_too_short_for_footer() {
        let id = [1u8; ID_LENGTH];
        // Valid header but no room left for a footer afterwards.
        let mut sub_file = Vec::new();
        codec_util::write_index_header(&mut sub_file, "FieldInfos", 1, &id, "");
        let err = write(&id, &[(".fnm".to_string(), sub_file)]).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn write_rejects_completely_empty_sub_file_bytes() {
        // Not even a full magic number's worth of bytes -- must fail cleanly
        // (EOF) rather than panic on the initial `read_be_u32`.
        let id = [1u8; ID_LENGTH];
        let err = write(&id, &[(".fnm".to_string(), Vec::new())]).unwrap_err();
        assert!(matches!(err, Error::Store(_)));
    }

    #[test]
    fn write_single_sub_file_round_trips() {
        // Exactly one sub-file: no ordering to exercise, but the header/footer
        // framing and offset math for a lone entry must still be correct.
        let id = [3u8; ID_LENGTH];
        let fnm = build_sub_file("FieldInfos", 1, &id, b"only file");
        let (cfs, cfe) = write(&id, &[(".fnm".to_string(), fnm.clone())]).unwrap();

        let entries = parse_entries(&cfe, &id).unwrap();
        assert_eq!(entries.entries.len(), 1);
        check_data_header_footer(&cfs, &id, &entries).unwrap();
        assert_eq!(
            open_input(&cfs, &entries, ".fnm").unwrap().as_slice(),
            fnm.as_slice()
        );
        assert_eq!(
            entries.get(".fnm").unwrap().offset % ALIGNMENT_BYTES as i64,
            0
        );
    }

    #[test]
    fn write_zero_length_body_sub_file_round_trips() {
        // A sub-file whose own body is empty (header+footer only, e.g. a
        // degenerate all-empty doc-values file) -- its recorded length is
        // just header+footer, and the *next* sub-file's offset must start
        // right after it with no gap or off-by-one, aligned as usual.
        let id = [4u8; ID_LENGTH];
        let empty_dvd = build_sub_file("DocValues", 0, &id, b"");
        let fnm = build_sub_file("FieldInfos", 1, &id, b"non-empty body");

        let sub_files = vec![
            (".dvd".to_string(), empty_dvd.clone()),
            (".fnm".to_string(), fnm.clone()),
        ];
        let (cfs, cfe) = write(&id, &sub_files).unwrap();
        let entries = parse_entries(&cfe, &id).unwrap();
        check_data_header_footer(&cfs, &id, &entries).unwrap();

        let dvd_entry = entries.get(".dvd").unwrap();
        assert_eq!(dvd_entry.length as usize, empty_dvd.len());
        assert_eq!(
            open_input(&cfs, &entries, ".dvd").unwrap().as_slice(),
            empty_dvd.as_slice()
        );
        assert_eq!(
            open_input(&cfs, &entries, ".fnm").unwrap().as_slice(),
            fnm.as_slice()
        );
    }

    #[test]
    fn write_many_sub_files_stresses_entry_count_vint_boundary() {
        // 200 sub-files pushes the `.cfe` entry-count vint from 1 byte to 2
        // bytes (boundary at 128) and stresses the alignment/offset math
        // across many small entries -- a scale no real segment's handful of
        // sub-files would ever reach.
        let id = [5u8; ID_LENGTH];
        let mut sub_files = Vec::new();
        for i in 0..200 {
            let name = format!(".x{i}");
            let body = vec![i as u8; (i % 7) as usize];
            sub_files.push((name, build_sub_file("Misc", 0, &id, &body)));
        }
        let (cfs, cfe) = write(&id, &sub_files).unwrap();
        let entries = parse_entries(&cfe, &id).unwrap();
        assert_eq!(entries.entries.len(), 200);
        check_data_header_footer(&cfs, &id, &entries).unwrap();

        for (name, original) in &sub_files {
            assert_eq!(
                open_input(&cfs, &entries, name).unwrap().as_slice(),
                original.as_slice(),
                "mismatch for {name}"
            );
        }
        for (_, entry) in &entries.entries {
            assert_eq!(entry.offset % ALIGNMENT_BYTES as i64, 0);
        }
    }

    #[test]
    fn write_unusual_sub_file_names_round_trip() {
        // Codec-suffix-shaped names (e.g. `_Lucene104_0.doc`) are what real
        // segments actually produce (see fixtures/data/compound_index's
        // manifest); also check an empty name and a non-ASCII one to be sure
        // the id is treated as an opaque string, not parsed/validated.
        let id = [6u8; ID_LENGTH];
        let names = ["_Lucene104_0.doc", "_Lucene90_0.dvd", "", "héllo.tïm"];
        let sub_files: Vec<(String, Vec<u8>)> = names
            .iter()
            .enumerate()
            .map(|(i, name)| {
                (
                    name.to_string(),
                    build_sub_file("Misc", 0, &id, format!("body {i}").as_bytes()),
                )
            })
            .collect();

        let (cfs, cfe) = write(&id, &sub_files).unwrap();
        let entries = parse_entries(&cfe, &id).unwrap();
        for (name, original) in &sub_files {
            assert_eq!(
                open_input(&cfs, &entries, name).unwrap().as_slice(),
                original.as_slice(),
                "mismatch for {name:?}"
            );
        }
    }

    #[test]
    fn write_duplicate_sub_file_names_detected_on_read() {
        // `write` itself doesn't dedupe by name (that's a caller-correctness
        // invariant, not something the packer can/should second-guess) but
        // the resulting `.cfe` must still be caught as corrupt/ambiguous by
        // `parse_entries` rather than silently letting the second entry
        // shadow the first.
        let id = [7u8; ID_LENGTH];
        let a = build_sub_file("Misc", 0, &id, b"aaa");
        let b = build_sub_file("Misc", 0, &id, b"bbbbb");
        let (_cfs, cfe) = write(&id, &[(".fnm".to_string(), a), (".fnm".to_string(), b)]).unwrap();

        assert!(matches!(
            parse_entries(&cfe, &id),
            Err(Error::DuplicateEntry(name)) if name == ".fnm"
        ));
    }
}
