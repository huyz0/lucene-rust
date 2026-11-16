//! `SimpleTextSegmentInfoFormat`: a segment's `SegmentInfo` as one `.si`
//! text file.
//!
//! ```text
//!     version <major.minor.bugfix>
//!     min version <version, or null>
//!     number of documents <maxDoc>
//!     uses compound file <bool>
//!     has blocks <bool>
//!     diagnostics <count>
//!       key <k>                     per diagnostic
//!       value <v>
//!     attributes <count>
//!       key <k>                     per attribute
//!       value <v>
//!     files <count>
//!       file <name>                 per file (the .si itself included)
//!     id <BytesRef.toString of the 16-byte id>
//!     sort <count>
//!       name <SortFieldProvider name>
//!       type <SortField.toString(), never read back>
//!       bytes <BytesRef.toString of SortFieldProvider.write's bytes>
//! checksum <20 digits>
//! ```
//!
//! Java fills a `HashMap`/`HashSet` and writes their iteration order; here
//! the diagnostics, attributes and files keep the order they are given in,
//! so re-writing what was read reproduces the file.
//!
//! The in-memory model is [`SimpleTextSegmentInfo`], not `lucene-index`'s
//! `SegmentInfo` (`lucene-codecs` sits below `lucene-index`). An index-sort
//! field is kept as its three lines ([`SortFieldEntry`]): the provider name
//! is checked against the providers Lucene ships (`SortFieldProvider.forName`),
//! but the serialized bytes are not decoded (`readSortField`) here. The
//! `lucene_index::segment_info::{parse_simpletext, write_simpletext}` adapters
//! use the same typed providers as binary segment infos.

use std::collections::HashSet;

use lucene_store::codec_util::ID_LENGTH;
use lucene_util::string_helper::id_to_string;
use lucene_util::version::Version;

use super::util::{
    self, boolean_str, bytes_ref_to_string, from_bytes_ref_string, parse_boolean, parse_integer,
    read_string, read_value, OrderedMap,
};
use super::{corrupt, Error, Result};

/// `SimpleTextSegmentInfoFormat.SI_EXTENSION`.
pub const SI_EXTENSION: &str = "si";
/// `SI_VERSION`.
pub const SI_VERSION: &[u8] = b"    version ";
/// `SI_MIN_VERSION`.
pub const SI_MIN_VERSION: &[u8] = b"    min version ";
/// `SI_DOCCOUNT`.
pub const SI_DOCCOUNT: &[u8] = b"    number of documents ";
/// `SI_USECOMPOUND`.
pub const SI_USECOMPOUND: &[u8] = b"    uses compound file ";
/// `SI_HAS_BLOCKS`.
pub const SI_HAS_BLOCKS: &[u8] = b"    has blocks ";
/// `SI_NUM_DIAG`.
pub const SI_NUM_DIAG: &[u8] = b"    diagnostics ";
/// `SI_DIAG_KEY`.
pub const SI_DIAG_KEY: &[u8] = b"      key ";
/// `SI_DIAG_VALUE`.
pub const SI_DIAG_VALUE: &[u8] = b"      value ";
/// `SI_NUM_ATT`.
pub const SI_NUM_ATT: &[u8] = b"    attributes ";
/// `SI_ATT_KEY`.
pub const SI_ATT_KEY: &[u8] = b"      key ";
/// `SI_ATT_VALUE`.
pub const SI_ATT_VALUE: &[u8] = b"      value ";
/// `SI_NUM_FILES`.
pub const SI_NUM_FILES: &[u8] = b"    files ";
/// `SI_FILE`.
pub const SI_FILE: &[u8] = b"      file ";
/// `SI_ID`.
pub const SI_ID: &[u8] = b"    id ";
/// `SI_SORT`.
pub const SI_SORT: &[u8] = b"    sort ";
/// `SI_SORT_TYPE`.
pub const SI_SORT_TYPE: &[u8] = b"      type ";
/// `SI_SORT_NAME`.
pub const SI_SORT_NAME: &[u8] = b"      name ";
/// `SI_SORT_BYTES`.
pub const SI_SORT_BYTES: &[u8] = b"      bytes ";

/// The `SortFieldProvider` names `lucene-core` registers
/// (`META-INF/services/org.apache.lucene.index.SortFieldProvider`).
pub const SORT_FIELD_PROVIDERS: &[&str] = &[
    "SortField",
    "SortedNumericSortField",
    "SortedSetSortField",
    "BinarySortField",
];

/// `IndexFileNames.segmentFileName(segment, "", "si")`.
pub fn segment_info_file_name(segment: &str) -> String {
    super::segment_file_name(segment, "", SI_EXTENSION)
}

/// One field of the index sort, as its three `.si` lines.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SortFieldEntry {
    /// `IndexSorter.getProviderName()`.
    pub provider: String,
    /// `SortField.toString()` (written, never read back by Java).
    pub description: String,
    /// `SortFieldProvider.write(sortField, out)`'s bytes.
    pub bytes: Vec<u8>,
}

/// What `SimpleTextSegmentInfoFormat` reads and writes of a `SegmentInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SimpleTextSegmentInfo<S = SortFieldEntry> {
    /// `SegmentInfo.name` (not in the file: the caller names the segment).
    pub name: String,
    /// `getVersion()`.
    pub version: Version,
    /// `getMinVersion()`; `None` is Java's `null`.
    pub min_version: Option<Version>,
    /// `maxDoc()`.
    pub doc_count: i32,
    /// `getUseCompoundFile()`.
    pub is_compound_file: bool,
    /// `getHasBlocks()`.
    pub has_blocks: bool,
    /// `getDiagnostics()`, in file order.
    pub diagnostics: Vec<(String, String)>,
    /// `getAttributes()`, in file order.
    pub attributes: Vec<(String, String)>,
    /// `files()`, in file order.
    pub files: Vec<String>,
    /// `getId()`.
    pub id: [u8; ID_LENGTH],
    /// `getIndexSort()`; empty is Java's `null`.
    pub index_sort: Vec<S>,
}

/// A count read off the file, for Java's `HashMap.newHashMap(n)` /
/// `new SortField[n]`, which refuse a negative one.
fn count(n: i32, what: &str) -> Result<usize> {
    usize::try_from(n).map_err(|_| corrupt(format!("invalid {what} count: {n}")))
}

/// `Version.parse(s)` as the reader calls it.
fn parse_version(s: &str) -> Result<Version> {
    Version::parse(s).map_err(|e| corrupt(format!("unable to parse version string: {e}")))
}

/// `IndexFileNames.CODEC_FILE_PATTERN` (`_[a-z0-9]+(_.*)?\..*`), as
/// `SegmentInfo.checkFileNames` matches it.
///
/// Java's `.` matches no line terminator (`\n`, `\r`, U+0085, U+2028,
/// U+2029), so a name holding one never matches.
pub fn matches_codec_file_pattern(name: &str) -> bool {
    if name.contains(['\n', '\r', '\u{85}', '\u{2028}', '\u{2029}']) {
        return false;
    }
    let Some(rest) = name.strip_prefix('_') else {
        return false;
    };
    let run = rest
        .bytes()
        .take_while(|b| b.is_ascii_lowercase() || b.is_ascii_digit())
        .count();
    match rest.get(run..) {
        _ if run == 0 => false,
        Some(tail) if tail.starts_with('.') => true,
        Some(tail) if tail.starts_with('_') => tail.contains('.'),
        _ => false,
    }
}

/// `SegmentInfo.setFiles(files)`'s `checkFileNames`.
///
/// # Errors
/// [`Error::IllegalArgument`] with Java's message for a name that does not
/// match [`matches_codec_file_pattern`].
pub fn check_file_names(files: &[String]) -> Result<()> {
    match files.iter().find(|f| !matches_codec_file_pattern(f)) {
        Some(bad) => Err(Error::IllegalArgument(format!(
            "invalid codec filename '{bad}', must match: _[a-z0-9]+(_.*)?\\..*"
        ))),
        None => Ok(()),
    }
}

/// `SimpleTextSegmentInfoFormat.read(directory, segmentName, segmentID,
/// context)` over the `.si`'s bytes.
///
/// # Errors
/// `Corrupted` for a line without its prefix, a number or version that does
/// not parse, a negative count, an id other than `segment_id` or a bad
/// checksum; `Eof` for a short file; [`Error::IllegalArgument`] for an id or
/// sort bytes not in `BytesRef.toString` form, an unknown sort provider, or
/// a file name `SegmentInfo.setFiles` refuses.
pub fn read(data: &[u8], segment_name: &str, segment_id: &[u8]) -> Result<SimpleTextSegmentInfo> {
    // Java: SimpleTextSegmentInfoFormat.read
    let mut pos = 0usize;
    let mut scratch = Vec::new();
    let version = parse_version(&read_string(data, &mut pos, &mut scratch, SI_VERSION)?)?;
    let min_version = match read_string(data, &mut pos, &mut scratch, SI_MIN_VERSION)?.as_str() {
        "null" => None,
        s => Some(parse_version(s)?),
    };
    let doc_count = parse_integer(read_value(data, &mut pos, &mut scratch, SI_DOCCOUNT)?)?;
    let is_compound_file = parse_boolean(read_value(data, &mut pos, &mut scratch, SI_USECOMPOUND)?);
    let has_blocks = parse_boolean(read_value(data, &mut pos, &mut scratch, SI_HAS_BLOCKS)?);

    let num_diag = parse_integer(read_value(data, &mut pos, &mut scratch, SI_NUM_DIAG)?)?;
    let mut diagnostics = OrderedMap::default();
    for _ in 0..count(num_diag, "diagnostics")? {
        let key = read_string(data, &mut pos, &mut scratch, SI_DIAG_KEY)?;
        let value = read_string(data, &mut pos, &mut scratch, SI_DIAG_VALUE)?;
        diagnostics.put(key, value);
    }

    let num_att = parse_integer(read_value(data, &mut pos, &mut scratch, SI_NUM_ATT)?)?;
    let mut attributes = OrderedMap::default();
    for _ in 0..count(num_att, "attributes")? {
        let key = read_string(data, &mut pos, &mut scratch, SI_ATT_KEY)?;
        let value = read_string(data, &mut pos, &mut scratch, SI_ATT_VALUE)?;
        attributes.put(key, value);
    }

    let num_files = parse_integer(read_value(data, &mut pos, &mut scratch, SI_NUM_FILES)?)?;
    let mut files: Vec<String> = Vec::new();
    let mut seen = HashSet::new();
    for _ in 0..count(num_files, "files")? {
        let file = read_string(data, &mut pos, &mut scratch, SI_FILE)?;
        // `files.add(fileName)` into a `HashSet`, kept in file order.
        if seen.insert(file.clone()) {
            files.push(file);
        }
    }

    let id = from_bytes_ref_string(&read_string(data, &mut pos, &mut scratch, SI_ID)?)?;
    if segment_id != id.as_slice() {
        return Err(corrupt(format!(
            "file mismatch, expected: {}, got: {}",
            id_to_string(Some(segment_id)),
            id_to_string(Some(&id))
        )));
    }

    let num_sort_fields = parse_integer(read_value(data, &mut pos, &mut scratch, SI_SORT)?)?;
    let mut index_sort = Vec::new();
    for _ in 0..count(num_sort_fields, "sort field")? {
        let provider = read_string(data, &mut pos, &mut scratch, SI_SORT_NAME)?;
        let description = read_string(data, &mut pos, &mut scratch, SI_SORT_TYPE)?;
        let bytes =
            from_bytes_ref_string(&read_string(data, &mut pos, &mut scratch, SI_SORT_BYTES)?)?;
        // `SortFieldProvider.forName(provider)`.
        if !SORT_FIELD_PROVIDERS.contains(&provider.as_str()) {
            return Err(Error::IllegalArgument(format!(
                "An SPI class of type org.apache.lucene.index.SortFieldProvider with name \
                 '{provider}' does not exist."
            )));
        }
        index_sort.push(SortFieldEntry {
            provider,
            description,
            bytes,
        });
    }
    util::check_footer(data, 0, &mut pos)?;
    // `new SegmentInfo(...)`'s `id.length != ID_LENGTH` check, after the
    // footer as in Java: "invalid id: " + `Arrays.toString(id)`.
    let id: [u8; ID_LENGTH] = id.as_slice().try_into().map_err(|_| {
        let bytes: Vec<String> = id.iter().map(|&b| (b as i8).to_string()).collect();
        Error::IllegalArgument(format!("invalid id: [{}]", bytes.join(", ")))
    })?;
    // `info.setFiles(files)`.
    check_file_names(&files)?;
    Ok(SimpleTextSegmentInfo {
        name: segment_name.to_string(),
        version,
        min_version,
        doc_count,
        is_compound_file,
        has_blocks,
        diagnostics: diagnostics.into_entries(),
        attributes: attributes.into_entries(),
        files,
        id,
        index_sort,
    })
}

/// `SimpleTextSegmentInfoFormat.write(dir, si, context)`: the `.si` for
/// `si`, after Java's `si.addFile(segFileName)` (appended to
/// [`SimpleTextSegmentInfo::files`] when absent).
///
/// # Errors
/// [`Error::IllegalArgument`] for a file name `SegmentInfo.addFile` refuses.
pub fn write(si: &mut SimpleTextSegmentInfo) -> Result<Vec<u8>> {
    // Java: SimpleTextSegmentInfoFormat.write
    let seg_file_name = segment_info_file_name(&si.name);
    check_file_names(std::slice::from_ref(&seg_file_name))?;
    check_file_names(&si.files)?;
    if !si.files.contains(&seg_file_name) {
        si.files.push(seg_file_name);
    }
    let mut out = Vec::new();
    util::write_line(&mut out, SI_VERSION, si.version.to_string().as_bytes());
    let min_version = si
        .min_version
        .map_or_else(|| "null".to_string(), |v| v.to_string());
    util::write_line(&mut out, SI_MIN_VERSION, min_version.as_bytes());
    util::write_line(&mut out, SI_DOCCOUNT, si.doc_count.to_string().as_bytes());
    util::write_line(&mut out, SI_USECOMPOUND, boolean_str(si.is_compound_file));
    util::write_line(&mut out, SI_HAS_BLOCKS, boolean_str(si.has_blocks));
    util::write_line(
        &mut out,
        SI_NUM_DIAG,
        si.diagnostics.len().to_string().as_bytes(),
    );
    for (key, value) in &si.diagnostics {
        util::write_line(&mut out, SI_DIAG_KEY, key.as_bytes());
        util::write_line(&mut out, SI_DIAG_VALUE, value.as_bytes());
    }
    util::write_line(
        &mut out,
        SI_NUM_ATT,
        si.attributes.len().to_string().as_bytes(),
    );
    for (key, value) in &si.attributes {
        util::write_line(&mut out, SI_ATT_KEY, key.as_bytes());
        util::write_line(&mut out, SI_ATT_VALUE, value.as_bytes());
    }
    util::write_line(
        &mut out,
        SI_NUM_FILES,
        si.files.len().to_string().as_bytes(),
    );
    for file in &si.files {
        util::write_line(&mut out, SI_FILE, file.as_bytes());
    }
    util::write_line(&mut out, SI_ID, bytes_ref_to_string(&si.id).as_bytes());
    util::write_line(
        &mut out,
        SI_SORT,
        si.index_sort.len().to_string().as_bytes(),
    );
    for sort in &si.index_sort {
        util::write_line(&mut out, SI_SORT_NAME, sort.provider.as_bytes());
        util::write_line(&mut out, SI_SORT_TYPE, sort.description.as_bytes());
        util::write_line(
            &mut out,
            SI_SORT_BYTES,
            bytes_ref_to_string(&sort.bytes).as_bytes(),
        );
    }
    util::write_checksum(&mut out);
    Ok(out)
}

#[cfg(test)]
mod tests;
