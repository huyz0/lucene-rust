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
//! What differs from Java: the reader borrows the file's bytes; offsets are
//! checked against the file when a sub-file is opened (Java's
//! `IndexInput.slice` does the same, lazily) and a negative length is
//! `Corrupted`; the trailer's offset parses from ASCII digits (Java's
//! `DecimalFormat` also takes a sign). Java's `assert` that the names are
//! unique and sorted is `Corrupted`. Names sort, and are looked up, in
//! UTF-16 order, as Java's `Arrays.sort`/`binarySearch` on `String`s.

use super::util::{self, parse_integer, parse_long, read_line, read_string};
use super::{corrupt, Error, Result};

/// `String.compareTo`: UTF-16 code unit order.
fn utf16_cmp(a: &str, b: &str) -> std::cmp::Ordering {
    a.encode_utf16().cmp(b.encode_utf16())
}

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
    // `indexOf('_', 1)`: from the second character, whatever the first's
    // width (a surrogate pair's low half is never `_`).
    let first = filename.chars().next().map_or(0, char::len_utf8);
    let idx = filename
        .get(first..)
        .and_then(|rest| rest.find('_'))
        // an index into `filename[first..]`, so `+ first` is within `filename`.
        .map(|i| i.saturating_add(first))
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
        // Java: SimpleTextCompoundFormat.getCompoundReader
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
                std::str::from_utf8(v.get(..digits)?)
                    .ok()?
                    .parse::<u64>()
                    .ok()
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
        // Java's `new String[numEntries]` and the two offset arrays are sized
        // from the file: here they grow as entries parse, each entry taking
        // its own lines of input, so a hostile count runs out of bytes first.
        let mut file_names: Vec<String> = Vec::new();
        let mut start_offsets = Vec::new();
        let mut end_offsets = Vec::new();
        for _ in 0..n {
            let name = read_string(data, &mut pos, &mut scratch, TABLENAME)?;
            let name = format!("{segment}{}", strip_segment_name(&name));
            // files must be unique and in sorted order
            if let Some(prev) = file_names.last() {
                if utf16_cmp(&name, prev).is_le() {
                    return Err(corrupt(format!(
                        "CFS entries not unique and sorted: {name} after {prev}"
                    )));
                }
            }
            file_names.push(name);
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
            .binary_search_by(|n| utf16_cmp(n, name))
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
        range.and_then(|(s, e)| data.get(s..e)).ok_or_else(|| {
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
    // Java: SimpleTextCompoundFormat.write
    let mut sorted: Vec<&(String, Vec<u8>)> = files.iter().collect();
    sorted.sort_by(|a, b| utf16_cmp(&a.0, &b.0));
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

#[cfg(test)]
mod tests {
    use super::*;

    fn files() -> Vec<(String, Vec<u8>)> {
        vec![
            ("_7.si".to_string(), b"si bytes\n".to_vec()),
            ("_7.fld".to_string(), vec![0x0a, 0x5c, 0]),
            ("_7_1.inf".to_string(), Vec::new()),
        ]
    }

    #[test]
    fn round_trip_in_name_order() {
        let cfs = write(&files());
        let text = String::from_utf8_lossy(&cfs).into_owned();
        assert!(text.starts_with("cfs entry for: _7.fld\n"), "{text}");
        assert!(
            text.ends_with("table of contents begins at offset: 0000000000000000079\n"),
            "{text}"
        );
        let r = SimpleTextCompoundReader::open(&cfs, "_9").unwrap();
        assert_eq!(r.list_all(), ["_9.fld", "_9.si", "_9_1.inf"]);
        assert_eq!(r.open_input(&cfs, "_9.fld").unwrap(), &[0x0a, 0x5c, 0]);
        assert_eq!(r.open_input(&cfs, "_9.si").unwrap(), b"si bytes\n");
        assert_eq!(r.file_length("_9_1.inf").unwrap(), 0);
        let err = r.file_length("_9.pst").unwrap_err().to_string();
        assert!(
            err.starts_with("No sub-file found (fileName=_9.pst files: [_9.fld"),
            "{err}"
        );
        assert!(r.open_input(&cfs, "_9.pst").is_err());
        assert_eq!(compound_file_name("_7"), "_7.scf");
    }

    #[test]
    fn strip_segment_name_as_java() {
        assert_eq!(strip_segment_name("_7_1.inf"), "_1.inf");
        assert_eq!(strip_segment_name("_7.si"), ".si");
        assert_eq!(strip_segment_name("noext"), "noext");
        assert_eq!(strip_segment_name(""), "");
        // `indexOf('_', 1)` counts characters, not bytes.
        assert_eq!(strip_segment_name("\u{e9}_1.a"), "_1.a");
        assert_eq!(strip_segment_name("\u{1f600}_1.a"), "_1.a");
        assert_eq!(strip_segment_name("_\u{e9}.a"), ".a");
    }

    #[test]
    fn utf16_order_not_byte_order() {
        // U+FFFD sorts after U+10000 in UTF-8 bytes but before it in UTF-16.
        let cfs = write(&[
            ("_1_\u{10000}.a".to_string(), vec![1]),
            ("_1_\u{fffd}.a".to_string(), vec![2]),
        ]);
        let text = String::from_utf8(cfs.clone()).unwrap();
        assert!(text.starts_with("cfs entry for: _1_\u{10000}.a"), "{text}");
        let r = SimpleTextCompoundReader::open(&cfs, "_1").unwrap();
        assert_eq!(r.open_input(&cfs, "_1_\u{fffd}.a").unwrap(), &[2]);
        assert_eq!(r.open_input(&cfs, "_1_\u{10000}.a").unwrap(), &[1]);
    }

    fn reseal(table: &str) -> Vec<u8> {
        let mut out = table.as_bytes().to_vec();
        util::write_line(
            &mut out,
            TABLEPOS,
            format!("{:0width$}", 0, width = OFFSET_DIGITS).as_bytes(),
        );
        out
    }

    #[test]
    fn damaged_tables_are_refused() {
        for table in [
            "table of contents, size: -1\n",
            "table of contents, size: x\n",
            "table of contentz, size: 0\n",
            "table of contents, size: 1\n  filename: _1.a\n    start: 0\n    end: x\n",
            "table of contents, size: 1\n  filename: _1.a\n    start: x\n    end: 0\n",
            "table of contents, size: 1\n  filenome: _1.a\n",
            "table of contents, size: 2\n  filename: _1.b\n    start: 0\n    end: 0\n  \
             filename: _1.a\n    start: 0\n    end: 0\n",
            "table of contents, size: 2\n  filename: _1.a\n    start: 0\n    end: 0\n  \
             filename: _1.a\n    start: 0\n    end: 0\n",
            "table of contents, size: 2\n",
        ] {
            assert!(
                SimpleTextCompoundReader::open(&reseal(table), "_1").is_err(),
                "{table}"
            );
        }
        // Offsets outside the file, or end before start, fail on use.
        for (start, end) in [("0", "999"), ("5", "2"), ("-1", "0")] {
            let t = format!(
                "table of contents, size: 1\n  filename: _1.a\n    start: {start}\n    end: {end}\n"
            );
            let data = reseal(&t);
            let r = SimpleTextCompoundReader::open(&data, "_1").unwrap();
            assert!(r.open_input(&data, "_1.a").is_err(), "{start} {end}");
        }
        let data =
            reseal("table of contents, size: 1\n  filename: _1.a\n    start: 5\n    end: 2\n");
        let r = SimpleTextCompoundReader::open(&data, "_1").unwrap();
        assert!(r.file_length("_1.a").is_err());
        // A trailer that is not one, or a file too short for it.
        let mut bad = reseal("table of contents, size: 0\n");
        let n = bad.len();
        bad[n - 20] = b'x';
        assert!(SimpleTextCompoundReader::open(&bad, "_1").is_err());
        assert!(SimpleTextCompoundReader::open(b"short", "_1").is_err());
        let mut far = b"table of contents, size: 0\n".to_vec();
        util::write_line(&mut far, TABLEPOS, b"0000000000000009999");
        assert!(SimpleTextCompoundReader::open(&far, "_1").is_err());
    }
}
