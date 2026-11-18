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
    // An entry is at least 17 bytes (a 1-byte string length, two longs), so
    // a corrupt count cannot reserve more than the file could hold.
    let mut entries = Vec::with_capacity((num_entries.max(0) as usize).min(input.remaining() / 17));
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

    // Saturating, not plain `+`: `offset`/`length` come straight off a `.cfe`
    // whose contents may be corrupt, and `i64 + i64` would panic on overflow
    // in a debug build (Java's `long` addition silently wraps and then fails
    // the comparison below). A saturated extent can never equal a real file
    // length, so it lands in the same `WrongLength` error either way.
    let max_extent = entries
        .entries
        .iter()
        .map(|(_, e)| usize::try_from(e.offset.saturating_add(e.length)).unwrap_or(usize::MAX))
        .max()
        .unwrap_or_else(|| codec_util::index_header_length(DATA_CODEC, ""));
    let expected_length = max_extent.saturating_add(codec_util::FOOTER_LENGTH);
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
/// merging, which needs to read a sub-range of a shared `.cfs`
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
/// - Sub-files are packed smallest-first so small files are more likely to
///   land within one page. Lucene 10.5.0 does this by draining a
///   `PriorityQueue<SizedFile>` ordered by `length` (Lucene `main` later
///   replaced it with `List.sort(Comparator.comparingLong(length))`); this
///   port uses Rust's stable sort by length. The two agree whenever the
///   lengths are distinct; for equal-length sub-files 10.5.0's binary-heap
///   pop order is unspecified, so the resulting `.cfs` packing order can
///   differ. Nothing in the format depends on it -- `.cfe` maps names to
///   `(offset, length)` and the reader looks up by name.
/// - Each sub-file's start offset in `.cfs` is padded up to a 64-byte
///   boundary ([`ALIGNMENT_BYTES`]); flat vector data of a page or more
///   starts on a page instead ([`VECTOR_DATA_PAGE`]), and readers use the
///   recorded offset either way.
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

        let start = entry_start(cfs.len(), name, bytes);
        cfs.resize(start, 0);
        let start_offset = cfs.len() as i64;
        cfs.extend_from_slice(bytes);
        // ARITH: `start_offset` was `cfs.len()` immediately before the
        // `extend_from_slice` above, and a `Vec` only grows there, so
        // `cfs.len() >= start_offset`. Both fit `i64` because a `Vec`'s
        // length is bounded by `isize::MAX`.
        #[allow(clippy::arithmetic_side_effects)]
        let length = cfs.len() as i64 - start_offset;

        cfe.write_string(name);
        cfe.write_i64(start_offset);
        cfe.write_i64(length);
    }

    codec_util::write_footer(&mut cfs);
    codec_util::write_footer(&mut cfe);

    Ok((cfs, cfe))
}

/// Flat vector data starts on a page of its own within `.cfs`: the data
/// inside the file starts on a page ([`crate::vectors::VECTOR_DATA_ALIGNMENT`]),
/// and a member that began anywhere else in the archive would lose that.
const VECTOR_DATA_PAGE: usize = crate::vectors::VECTOR_DATA_ALIGNMENT;

/// Extensions of the other vector files whose members start on a page: IVF
/// rerank vectors (`.ivv`, each field's vectors start on a page) and disk
/// HNSW node records (`.dhn`, page-aligned record groups) lose the alignment
/// inside them if their member starts anywhere else; the HNSW graph (`.vex`),
/// quantized vectors (`.veq`) and IVF codes (`.ivp`) start on a page so a
/// paging directory can read each as a file of its own
/// (`Directory::open_member`).
const PAGE_ALIGNED_EXTENSIONS: [&str; 5] = ["ivv", "dhn", "vex", "veq", "ivp"];

/// Where a sub-file `name` appended at `pos` starts in `.cfs`: the next
/// [`ALIGNMENT_BYTES`] boundary, except that flat vector data and the files
/// of [`PAGE_ALIGNED_EXTENSIONS`] start on a
/// page ([`VECTOR_DATA_PAGE`]) and paged zstd stored fields, whose
/// chunks are packed into pages counted from the file's start
/// ([`crate::stored_fields::ZSTD_PAGED_PAGE_SIZE`]), never straddle a page
/// they need not. One that fits in what is left of the current page stays
/// there (a small segment's `.fdt` costs no padding); a longer one starts
/// on the next page. Readers slice by the recorded offset either way,
/// Java's included.
fn entry_start(pos: usize, name: &str, bytes: &[u8]) -> usize {
    let start = pos.next_multiple_of(ALIGNMENT_BYTES);
    let mut input = SliceInput::new(bytes);
    let codec = input
        .read_be_u32()
        .ok()
        .and_then(|_| input.read_string().ok());
    let ext = name.rsplit('.').next().unwrap_or(name);
    if bytes.len() >= VECTOR_DATA_PAGE
        && (matches!(
            codec.as_deref(),
            Some(crate::vectors::DATA_CODEC | crate::vectors::HALF_FLOAT_DATA_CODEC)
        ) || PAGE_ALIGNED_EXTENSIONS.contains(&ext))
    {
        return pos.next_multiple_of(VECTOR_DATA_PAGE);
    }
    if codec.as_deref() != Some(crate::stored_fields::DATA_CODEC_ZSTD_PAGED) {
        return start;
    }
    let page = crate::stored_fields::ZSTD_PAGED_PAGE_SIZE;
    let room = page.saturating_sub(start.checked_rem(page).unwrap_or(0));
    if bytes.len() <= room {
        start
    } else {
        pos.next_multiple_of(page)
    }
}

/// Validates a sub-file's `IndexHeader` object id (must match `segment_id`,
/// though the codec name/version/suffix are whatever that sub-file's own
/// format uses and aren't constrained here) and its `Footer` checksum, port
/// of the checks `CodecUtil.verifyAndCopyIndexHeader`/`CodecUtil.checkFooter`
/// perform in Java's `writeCompoundFile` before trusting a sub-file's bytes.
pub fn verify_sub_file(name: &str, bytes: &[u8], segment_id: &[u8; ID_LENGTH]) -> Result<()> {
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

#[cfg(test)]
mod tests;
