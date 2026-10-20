//! `SimpleTextCompoundFormat`: a segment's files packed into one `.scf`.
//!
//! ```text
//! cfs entry for: <file name>          per file, in name order
//! <the file's bytes, verbatim>
//! ...
//! table of contents, size: N
//!   filename: <file name>             per file, in name order
//!     start: <offset of its bytes>
//!     end: <offset past them>
//! table of contents begins at offset: <19-digit zero-padded offset>
//! ```
//!
//! The trailer has a fixed length, so a reader finds it from the end of the
//! file. There is no checksum line (Java's `checkIntegrity` is a no-op).
//!
//! What differs from Java: offsets are checked against the file when a
//! sub-file is opened (Java's `IndexInput.slice` does the same, lazily);
//! the trailer's offset parses from ASCII digits (Java's `DecimalFormat`
//! also takes a sign and grouping). Names are sorted, and looked up by
//! binary search, in byte order (Java's `Arrays.sort`/`binarySearch` in
//! UTF-16 order: the same for the ASCII names Lucene writes).

use super::util::{self, parse_integer, parse_long, read_line, read_string};
use super::{corrupt, Error, Result};

/// `SimpleTextCompoundFormat.DATA_EXTENSION`.
pub const DATA_EXTENSION: &str = "scf";
/// `HEADER`.
pub const HEADER: &[u8] = b"cfs entry for: ";
/// `TABLE`.
pub const TABLE: &[u8] = b"table of contents, size: ";
/// `TABLENAME`.
pub const TABLENAME: &[u8] = b"  filename: ";
/// `TABLESTART`.
pub const TABLESTART: &[u8] = b"    start: ";
/// `TABLEEND`.
pub const TABLEEND: &[u8] = b"    end: ";
/// `TABLEPOS`.
pub const TABLEPOS: &[u8] = b"table of contents begins at offset: ";
/// `OFFSETPATTERN.length()`: the digits of `Long.MAX_VALUE`.
pub const OFFSET_DIGITS: usize = 19;

/// `IndexFileNames.segmentFileName(segment, "", "scf")`.
pub fn compound_file_name(segment: &str) -> String {
    super::segment_file_name(segment, "", DATA_EXTENSION)
}

/// `IndexFileNames.stripSegmentName(filename)`: from the first `_` after the
/// first character, else from the first `.`, else the whole name.
pub fn strip_segment_name(filename: &str) -> &str {
    let idx = filename
        .get(1..)
        .and_then(|rest| rest.find('_'))
        // an index into `filename[1..]`, so `+ 1` is within `filename`.
        .map(|i| i.saturating_add(1))
        .or_else(|| filename.find('.'));
    match idx {
        Some(i) => filename.get(i..).unwrap_or(filename),
        None => filename,
    }
}

/// The anonymous `CompoundDirectory` of `getCompoundReader`: the table of
/// contents of one `.scf`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleTextCompoundReader {
    file_names: Vec<String>,
    start_offsets: Vec<i64>,
    end_offsets: Vec<i64>,
}

impl SimpleTextCompoundReader {
    /// `getCompoundReader(dir, si)` over the `.scf`'s bytes, for segment
    /// `segment` (each entry's name is re-prefixed with it).
    ///
    /// # Errors
    /// `Corrupted` for a trailer or table line without its prefix, an
    /// offset or count that does not parse, or a negative count; `Eof` for
    /// a short file.
    pub fn open(data: &[u8], segment: &str) -> Result<Self> {
        let mut scratch = Vec::new();
        // first get to TOC:
        let trailer = TABLEPOS
            .len()
            .checked_add(OFFSET_DIGITS)
            .and_then(|n| n.checked_add(1))
            .and_then(|n| data.len().checked_sub(n))
            .ok_or_else(|| corrupt("can't parse CFS trailer: file too short"))?;
        let mut pos = trailer;
        read_line(data, &mut pos, &mut scratch)?;
        let table_pos = scratch
            .strip_prefix(TABLEPOS)
            .and_then(|v| {
                let digits = v.iter().take_while(|b| b.is_ascii_digit()).count();
                std::str::from_utf8(v.get(..digits)?).ok()?.parse::<u64>().ok()
            })
            .and_then(|p| usize::try_from(p).ok())
            .ok_or_else(|| {
                corrupt(format!(
                    "can't parse CFS trailer, got: {}",
                    String::from_utf8_lossy(&scratch)
                ))
            })?;
        // seek to TOC and read it
        let mut pos = table_pos;
        let num_entries = parse_integer(util::read_value(data, &mut pos, &mut scratch, TABLE)?)?;
        let n = usize::try_from(num_entries)
            .map_err(|_| corrupt(format!("invalid CFS entry count: {num_entries}")))?;
        let cap = n.min(data.len());
        let mut file_names = Vec::with_capacity(cap);
        let mut start_offsets = Vec::with_capacity(cap);
        let mut end_offsets = Vec::with_capacity(cap);
        for _ in 0..n {
            let name = read_string(data, &mut pos, &mut scratch, TABLENAME)?;
            file_names.push(format!("{segment}{}", strip_segment_name(&name)));
            start_offsets.push(parse_long(util::read_value(
                data,
                &mut pos,
                &mut scratch,
                TABLESTART,
            )?)?);
            end_offsets.push(parse_long(util::read_value(
                data,
                &mut pos,
                &mut scratch,
                TABLEEND,
            )?)?);
        }
        Ok(SimpleTextCompoundReader {
            file_names,
            start_offsets,
            end_offsets,
        })
    }

    /// `getIndex(name)`.
    fn index(&self, name: &str) -> Result<usize> {
        self.file_names
            .binary_search_by(|n| n.as_str().cmp(name))
            .map_err(|_| {
                Error::IllegalArgument(format!(
                    "No sub-file found (fileName={name} files: [{}])",
                    self.file_names.join(", ")
                ))
            })
    }

    /// `listAll()`.
    pub fn list_all(&self) -> &[String] {
        &self.file_names
    }

    /// `fileLength(name)`.
    ///
    /// # Errors
    /// [`Error::IllegalArgument`] for a name not in the table (Java's
    /// `FileNotFoundException`); `Corrupted` when its end precedes its start.
    pub fn file_length(&self, name: &str) -> Result<i64> {
        let i = self.index(name)?;
        self.end_offsets[i]
            .checked_sub(self.start_offsets[i])
            .filter(|&l| l >= 0)
            .ok_or_else(|| corrupt(format!("invalid CFS entry {name}")))
    }

    /// `openInput(name)`: the sub-file's bytes within `data`.
    ///
    /// # Errors
    /// What [`Self::file_length`] fails with, and `Corrupted` for a range
    /// outside `data`.
    pub fn open_input<'d>(&self, data: &'d [u8], name: &str) -> Result<&'d [u8]> {
        let i = self.index(name)?;
        let range = usize::try_from(self.start_offsets[i])
            .ok()
            .zip(usize::try_from(self.end_offsets[i]).ok())
            .filter(|(s, e)| s <= e);
        range
            .and_then(|(s, e)| data.get(s..e))
            .ok_or_else(|| {
                corrupt(format!(
                    "slice {name} [{}, {}) outside the compound file ({} bytes)",
                    self.start_offsets[i],
                    self.end_offsets[i],
                    data.len()
                ))
            })
    }
}

/// `SimpleTextCompoundFormat.write`: the `.scf` packing `files` (each a
/// full file name and its bytes), sorted by name.
pub fn write(files: &[(String, Vec<u8>)]) -> Vec<u8> {
    let mut sorted: Vec<&(String, Vec<u8>)> = files.iter().collect();
    sorted.sort_by(|a, b| a.0.cmp(&b.0));
    let mut out = Vec::new();
    let mut offsets = Vec::with_capacity(sorted.len());
    for (name, bytes) in &sorted {
        // write header for file
        util::write_line(&mut out, HEADER, name.as_bytes());
        // write bytes for file
        let start = out.len();
        out.extend_from_slice(bytes);
        offsets.push((start, out.len()));
    }
    let toc_pos = out.len();
    // write CFS table
    util::write_line(&mut out, TABLE, sorted.len().to_string().as_bytes());
    for ((name, _), (start, end)) in sorted.iter().zip(&offsets) {
        util::write_line(&mut out, TABLENAME, name.as_bytes());
        util::write_line(&mut out, TABLESTART, start.to_string().as_bytes());
        util::write_line(&mut out, TABLEEND, end.to_string().as_bytes());
    }
    util::write_line(
        &mut out,
        TABLEPOS,
        format!("{toc_pos:0width$}", width = OFFSET_DIGITS).as_bytes(),
    );
    out
}
