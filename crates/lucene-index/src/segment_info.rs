//! Port of `org.apache.lucene.codecs.lucene99.Lucene99SegmentInfoFormat` (`.si` files).
//!
//! Both directions ported: [`parse`] (read path, PLAN.md Phase 2) and
//! [`write`] (write path, PLAN.md Phase 5) are exact byte-level inverses of
//! each other.
//!
//! Wire format (all ints little-endian; header/footer per `codec_util`):
//! ```text
//! IndexHeader(codec="Lucene90SegmentInfo", version=0, id, suffix="")
//! SegVersion    --> i32 major, i32 minor, i32 bugfix
//! HasMinVersion --> u8 (0 or 1)
//! SegMinVersion --> [i32 major, i32 minor, i32 bugfix] iff HasMinVersion == 1
//! SegSize       --> i32 (maxDoc)
//! IsCompoundFile--> u8 (1 == YES, else NO)
//! HasBlocks     --> u8 (1 == YES, else NO)
//! Diagnostics   --> MapOfStrings
//! Files         --> SetOfStrings
//! Attributes    --> MapOfStrings
//! NumSortFields --> vint (0, or 1 for a single-field index sort; >1 rejected)
//! SortField     --> present iff NumSortFields == 1:
//!                    FieldName (string), SortType (u8, 0 == NUMERIC, only
//!                    value currently written/accepted), Reverse (u8, 0 ==
//!                    ascending, 1 == descending), MissingValue (u8, 0 ==
//!                    sort-first, 1 == sort-last)
//! Footer
//! ```
//!
//! **Index-sort byte-format disclaimer**: real Lucene's index-sort encoding
//! (`Lucene99SegmentInfoFormat`'s `numSortFields`/`SortFieldProvider.write`)
//! is written by pluggable `SortFieldProvider` Java classes looked up by name
//! (`SortFieldProvider.forName`), and its exact byte layout was not available
//! to verify against (no real-Lucene-written sorted-segment `.si` fixture
//! exists under `fixtures/`, see `docs/parity.md`). The `SortField` encoding
//! above is therefore **this port's own internal format**: round-trip-correct
//! against its own `write`/`parse`, but NOT confirmed byte-compatible with
//! real Lucene's `.si` files for an index-sorted segment. Only a single
//! NUMERIC-field sort is supported (`numSortFields` of 0 or 1); a value `> 1`
//! (a real multi-field `Sort`) is rejected as [`Error::UnsupportedIndexSort`]
//! rather than silently mis-parsed.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

const CODEC_NAME: &str = "Lucene90SegmentInfo";
const VERSION_START: i32 = 0;
const VERSION_CURRENT: i32 = 0;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("invalid docCount: {0}")]
    InvalidDocCount(i32),
    #[error("illegal boolean value for hasMinVersion: {0}")]
    IllegalHasMinVersion(u8),
    #[error("invalid index sort field count: {0}")]
    InvalidSortFieldCount(i32),
    #[error("multi-field index sorts are not yet supported (numSortFields={0})")]
    UnsupportedIndexSort(i32),
    #[error("unknown index sort field type byte: {0}")]
    UnknownSortFieldType(u8),
    #[error("illegal boolean value for index sort reverse flag: {0}")]
    IllegalSortReverse(u8),
    #[error("illegal index sort missing-value marker: {0}")]
    IllegalSortMissing(u8),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LuceneVersion {
    pub major: i32,
    pub minor: i32,
    pub bugfix: i32,
}

/// Where a document with no value for the index-sort field ends up, mirroring
/// real Lucene's `SortField.setMissingValue`/numeric-missing-value convention
/// (a missing value is treated as if it were `Long.MIN_VALUE`/`MAX_VALUE`,
/// which sorts it first or last regardless of ascending/descending).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortMissingValue {
    /// A doc with no value for the sort field sorts before every doc that has
    /// one.
    First,
    /// A doc with no value for the sort field sorts after every doc that has
    /// one.
    Last,
}

/// Single-field index sort descriptor (real Lucene's `SegmentInfo.indexSort`
/// is a `Sort` of one or more `SortField`s; this port supports exactly one
/// NUMERIC field, see this module's doc comment for the byte-format
/// disclaimer and the multi-field scope cut).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexSortField {
    pub field: String,
    /// `false` == ascending, `true` == descending (real Lucene's
    /// `SortField.reverse`).
    pub reverse: bool,
    pub missing: SortMissingValue,
}

#[derive(Debug, Clone)]
pub struct SegmentInfo {
    pub id: [u8; ID_LENGTH],
    pub version: LuceneVersion,
    pub min_version: Option<LuceneVersion>,
    pub doc_count: i32,
    pub is_compound_file: bool,
    pub has_blocks: bool,
    pub diagnostics: Vec<(String, String)>,
    pub files: Vec<String>,
    pub attributes: Vec<(String, String)>,
    /// `None` == unsorted segment (`numSortFields == 0`). See
    /// [`IndexSortField`] for the single-field-only scope.
    pub index_sort: Option<IndexSortField>,
}

/// Sort-field type byte. Only `Numeric` is currently produced by this port's
/// writer (`segment_writer`'s only doc-values shape today), but the byte is
/// reserved so a future STRING/other type doesn't have to renumber.
const SORT_TYPE_NUMERIC: u8 = 0;

/// Parses a whole `.si` file already read into memory, verifying header, footer,
/// and checksum. `segment_id` is the id Lucene stores alongside the segment in
/// `segments_N` and must match the id embedded in the `.si` file's index header.
pub fn parse(buf: &[u8], segment_id: &[u8; ID_LENGTH]) -> Result<SegmentInfo> {
    let mut input = SliceInput::new(buf);

    codec_util::check_index_header(
        &mut input,
        CODEC_NAME,
        VERSION_START,
        VERSION_CURRENT,
        segment_id,
        "",
    )?;

    let version = read_version(&mut input)?;

    let has_min_version = input.read_byte()?;
    let min_version = match has_min_version {
        0 => None,
        1 => Some(read_version(&mut input)?),
        other => return Err(Error::IllegalHasMinVersion(other)),
    };

    let doc_count = input.read_i32()?;
    if doc_count < 0 {
        return Err(Error::InvalidDocCount(doc_count));
    }

    let is_compound_file = input.read_byte()? == 1;
    let has_blocks = input.read_byte()? == 1;

    let diagnostics = input.read_map_of_strings()?;
    let files = input.read_set_of_strings()?;
    let attributes = input.read_map_of_strings()?;

    let num_sort_fields = input.read_vint()?;
    if num_sort_fields < 0 {
        return Err(Error::InvalidSortFieldCount(num_sort_fields));
    }
    if num_sort_fields > 1 {
        return Err(Error::UnsupportedIndexSort(num_sort_fields));
    }
    let index_sort = if num_sort_fields == 1 {
        Some(read_sort_field(&mut input)?)
    } else {
        None
    };

    let payload_end = input.position();
    codec_util::check_footer(&mut input, buf.len())?;
    debug_assert!(payload_end <= buf.len() - codec_util::FOOTER_LENGTH);

    Ok(SegmentInfo {
        id: *segment_id,
        version,
        min_version,
        doc_count,
        is_compound_file,
        has_blocks,
        diagnostics,
        files,
        attributes,
        index_sort,
    })
}

fn read_sort_field(input: &mut SliceInput) -> Result<IndexSortField> {
    let field = input.read_string()?;
    let sort_type = input.read_byte()?;
    if sort_type != SORT_TYPE_NUMERIC {
        return Err(Error::UnknownSortFieldType(sort_type));
    }
    let reverse = match input.read_byte()? {
        0 => false,
        1 => true,
        other => return Err(Error::IllegalSortReverse(other)),
    };
    let missing = match input.read_byte()? {
        0 => SortMissingValue::First,
        1 => SortMissingValue::Last,
        other => return Err(Error::IllegalSortMissing(other)),
    };
    Ok(IndexSortField {
        field,
        reverse,
        missing,
    })
}

fn write_sort_field(out: &mut Vec<u8>, sf: &IndexSortField) {
    out.write_string(&sf.field);
    out.write_byte(SORT_TYPE_NUMERIC);
    out.write_byte(if sf.reverse { 1 } else { 0 });
    out.write_byte(match sf.missing {
        SortMissingValue::First => 0,
        SortMissingValue::Last => 1,
    });
}

/// Port of `Lucene99SegmentInfoFormat.write`: the exact byte-level inverse of
/// [`parse`]. Emits `numSortFields = 0` unless `si.index_sort` is `Some`, in
/// which case it emits exactly one `SortField` (see this module's doc
/// comment for the single-field scope and the byte-format disclaimer).
/// Callers are responsible for `files` containing
/// only names prefixed by the segment's own name (the real writer enforces
/// this via `IndexFileNames.parseSegmentName`; this function does not
/// re-validate it, matching the parser's stance that a hand-built writer is
/// trusted).
pub fn write(si: &SegmentInfo, segment_suffix: &str) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();

    codec_util::write_index_header(
        &mut out,
        CODEC_NAME,
        VERSION_CURRENT,
        &si.id,
        segment_suffix,
    );

    write_version(&mut out, si.version);

    match si.min_version {
        Some(mv) => {
            out.write_byte(1);
            write_version(&mut out, mv);
        }
        None => out.write_byte(0),
    }

    out.write_i32(si.doc_count);
    out.write_byte(if si.is_compound_file { 1 } else { 0 });
    out.write_byte(if si.has_blocks { 1 } else { 0 });
    out.write_map_of_strings(&si.diagnostics);
    out.write_set_of_strings(&si.files);
    out.write_map_of_strings(&si.attributes);
    match &si.index_sort {
        Some(sf) => {
            out.write_vint(1);
            write_sort_field(&mut out, sf);
        }
        None => out.write_vint(0),
    }

    codec_util::write_footer(&mut out);
    out
}

fn write_version(out: &mut Vec<u8>, v: LuceneVersion) {
    out.write_i32(v.major);
    out.write_i32(v.minor);
    out.write_i32(v.bugfix);
}

fn read_version(input: &mut SliceInput) -> Result<LuceneVersion> {
    let major = input.read_i32()?;
    let minor = input.read_i32()?;
    let bugfix = input.read_i32()?;
    Ok(LuceneVersion {
        major,
        minor,
        bugfix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only `.si` byte builder: independent of the Java fixtures under
    /// `tests/segment_info_fixtures.rs` (which exercise real Lucene-written
    /// bytes) — this covers the parser's own corruption/error handling, which
    /// needs deliberately-invalid inputs a real Lucene codec would never write.
    struct SiBuilder {
        id: [u8; ID_LENGTH],
        has_min_version: u8,
        doc_count: i32,
        is_compound_file: u8,
        has_blocks: u8,
        num_sort_fields: i32,
        /// Raw bytes for the single sort field's payload, written verbatim
        /// after `num_sort_fields` when it's `1` -- lets tests hand-build
        /// both well-formed and deliberately-corrupt sort-field encodings.
        sort_field_bytes: Vec<u8>,
    }

    impl SiBuilder {
        fn valid() -> Self {
            Self {
                id: [1u8; ID_LENGTH],
                has_min_version: 0,
                doc_count: 5,
                is_compound_file: 1,
                has_blocks: 0,
                num_sort_fields: 0,
                sort_field_bytes: Vec::new(),
            }
        }

        fn valid_sort_field_bytes(field: &str, sort_type: u8, reverse: u8, missing: u8) -> Vec<u8> {
            let mut b = Vec::new();
            write_string(&mut b, field);
            b.push(sort_type);
            b.push(reverse);
            b.push(missing);
            b
        }

        fn build(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
            write_string(&mut out, CODEC_NAME);
            out.extend_from_slice(&(VERSION_CURRENT as u32).to_be_bytes());
            out.extend_from_slice(&self.id);
            out.push(0); // empty suffix

            out.extend_from_slice(&10i32.to_le_bytes()); // version major
            out.extend_from_slice(&0i32.to_le_bytes()); // minor
            out.extend_from_slice(&0i32.to_le_bytes()); // bugfix
            out.push(self.has_min_version);
            if self.has_min_version == 1 {
                out.extend_from_slice(&9i32.to_le_bytes());
                out.extend_from_slice(&0i32.to_le_bytes());
                out.extend_from_slice(&0i32.to_le_bytes());
            }
            out.extend_from_slice(&self.doc_count.to_le_bytes());
            out.push(self.is_compound_file);
            out.push(self.has_blocks);
            write_vint(&mut out, 0); // diagnostics: empty map
            write_vint(&mut out, 0); // files: empty set
            write_vint(&mut out, 0); // attributes: empty map
            write_vint(&mut out, self.num_sort_fields);
            out.extend_from_slice(&self.sort_field_bytes);

            out.extend_from_slice(&codec_util::FOOTER_MAGIC.to_be_bytes());
            out.extend_from_slice(&0u32.to_be_bytes());
            let checksum = crc32fast::hash(&out) as u64;
            out.extend_from_slice(&checksum.to_be_bytes());
            out
        }
    }

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

    #[test]
    fn valid_segment_info_parses() {
        let b = SiBuilder::valid();
        let si = parse(&b.build(), &b.id).unwrap();
        assert_eq!(si.doc_count, 5);
        assert!(si.is_compound_file);
        assert!(!si.has_blocks);
        assert!(si.min_version.is_none());
    }

    #[test]
    fn min_version_present_is_parsed() {
        let mut b = SiBuilder::valid();
        b.has_min_version = 1;
        let si = parse(&b.build(), &b.id).unwrap();
        let mv = si.min_version.unwrap();
        assert_eq!((mv.major, mv.minor, mv.bugfix), (9, 0, 0));
    }

    #[test]
    fn illegal_has_min_version_byte_rejected() {
        let b = SiBuilder::valid();
        let mut bytes = b.build();
        // has_min_version byte sits right after the 3 SegVersion i32s (12 bytes)
        // in the payload, following the fixed-size index header.
        let header_len =
            codec_util::CODEC_MAGIC.to_be_bytes().len() + 1 + CODEC_NAME.len() + 4 + ID_LENGTH + 1;
        let has_min_version_offset = header_len + 12;
        bytes[has_min_version_offset] = 7; // neither 0 nor 1
        assert!(matches!(
            parse(&bytes, &b.id),
            Err(Error::IllegalHasMinVersion(7))
        ));
    }

    #[test]
    fn negative_doc_count_rejected() {
        let mut b = SiBuilder::valid();
        b.doc_count = -1;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::InvalidDocCount(-1))
        ));
    }

    #[test]
    fn multi_field_sort_count_is_unsupported() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 2;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnsupportedIndexSort(2))
        ));
    }

    #[test]
    fn negative_sort_field_count_rejected() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = -1;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::InvalidSortFieldCount(-1))
        ));
    }

    #[test]
    fn single_numeric_sort_field_parses() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = SiBuilder::valid_sort_field_bytes("price", SORT_TYPE_NUMERIC, 0, 1);
        let si = parse(&b.build(), &b.id).unwrap();
        let sf = si.index_sort.unwrap();
        assert_eq!(sf.field, "price");
        assert!(!sf.reverse);
        assert_eq!(sf.missing, SortMissingValue::Last);
    }

    #[test]
    fn unknown_sort_field_type_byte_rejected() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = SiBuilder::valid_sort_field_bytes("price", 7, 0, 0);
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnknownSortFieldType(7))
        ));
    }

    #[test]
    fn illegal_sort_reverse_byte_rejected() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = SiBuilder::valid_sort_field_bytes("price", SORT_TYPE_NUMERIC, 5, 0);
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::IllegalSortReverse(5))
        ));
    }

    #[test]
    fn illegal_sort_missing_byte_rejected() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        b.sort_field_bytes = SiBuilder::valid_sort_field_bytes("price", SORT_TYPE_NUMERIC, 0, 9);
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::IllegalSortMissing(9))
        ));
    }

    #[test]
    fn wrong_id_rejected_with_store_error() {
        let b = SiBuilder::valid();
        let wrong_id = [9u8; ID_LENGTH];
        assert!(matches!(parse(&b.build(), &wrong_id), Err(Error::Store(_))));
    }

    // --- write() round-trips through parse() ---

    fn sample_si() -> SegmentInfo {
        SegmentInfo {
            id: [3u8; ID_LENGTH],
            version: LuceneVersion {
                major: 10,
                minor: 0,
                bugfix: 0,
            },
            min_version: None,
            doc_count: 42,
            is_compound_file: false,
            has_blocks: false,
            diagnostics: vec![],
            files: vec![],
            attributes: vec![],
            index_sort: None,
        }
    }

    #[test]
    fn write_minimal_round_trips() {
        let si = sample_si();
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert_eq!(parsed.doc_count, 42);
        assert!(!parsed.is_compound_file);
        assert!(!parsed.has_blocks);
        assert!(parsed.min_version.is_none());
        assert_eq!(parsed.version, si.version);
        assert!(parsed.diagnostics.is_empty());
        assert!(parsed.files.is_empty());
        assert!(parsed.attributes.is_empty());
    }

    #[test]
    fn write_compound_file_round_trips() {
        let mut si = sample_si();
        si.is_compound_file = true;
        si.has_blocks = true;
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert!(parsed.is_compound_file);
        assert!(parsed.has_blocks);
    }

    #[test]
    fn write_min_version_round_trips() {
        let mut si = sample_si();
        si.min_version = Some(LuceneVersion {
            major: 9,
            minor: 12,
            bugfix: 0,
        });
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        let mv = parsed.min_version.unwrap();
        assert_eq!((mv.major, mv.minor, mv.bugfix), (9, 12, 0));
    }

    #[test]
    fn write_diagnostics_files_attributes_round_trip() {
        let mut si = sample_si();
        si.diagnostics = vec![
            ("source".to_string(), "flush".to_string()),
            ("os".to_string(), "Linux".to_string()),
        ];
        si.files = vec!["_0.fdt".to_string(), "_0.fdx".to_string()];
        si.attributes = vec![(
            "Lucene90StoredFieldsFormat.mode".to_string(),
            "BEST_SPEED".to_string(),
        )];
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert_eq!(parsed.diagnostics, si.diagnostics);
        assert_eq!(parsed.files, si.files);
        assert_eq!(parsed.attributes, si.attributes);
    }

    #[test]
    fn write_with_segment_suffix_round_trips() {
        let si = sample_si();
        let bytes = write(&si, "suffix1");
        let parsed = parse(&bytes, &si.id);
        // parse() with the wrong (empty) suffix must fail; the exact suffix must match.
        assert!(parsed.is_err());
        let parsed_ok = {
            let mut input = SliceInput::new(&bytes);
            codec_util::check_index_header(
                &mut input,
                CODEC_NAME,
                VERSION_START,
                VERSION_CURRENT,
                &si.id,
                "suffix1",
            )
        };
        assert!(parsed_ok.is_ok());
    }

    #[test]
    fn write_zero_doc_count_round_trips() {
        let mut si = sample_si();
        si.doc_count = 0;
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert_eq!(parsed.doc_count, 0);
    }

    #[test]
    fn write_no_index_sort_round_trips_as_none() {
        let si = sample_si();
        assert!(si.index_sort.is_none());
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        assert!(parsed.index_sort.is_none());
    }

    #[test]
    fn write_ascending_index_sort_round_trips() {
        let mut si = sample_si();
        si.index_sort = Some(IndexSortField {
            field: "timestamp".to_string(),
            reverse: false,
            missing: SortMissingValue::First,
        });
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        let sf = parsed.index_sort.unwrap();
        assert_eq!(sf.field, "timestamp");
        assert!(!sf.reverse);
        assert_eq!(sf.missing, SortMissingValue::First);
    }

    #[test]
    fn write_descending_index_sort_with_missing_last_round_trips() {
        let mut si = sample_si();
        si.index_sort = Some(IndexSortField {
            field: "score".to_string(),
            reverse: true,
            missing: SortMissingValue::Last,
        });
        let bytes = write(&si, "");
        let parsed = parse(&bytes, &si.id).unwrap();
        let sf = parsed.index_sort.unwrap();
        assert_eq!(sf.field, "score");
        assert!(sf.reverse);
        assert_eq!(sf.missing, SortMissingValue::Last);
    }
}
