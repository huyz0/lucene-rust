//! Port of `org.apache.lucene.codecs.lucene99.Lucene99SegmentInfoFormat` (`.si` files).
//!
//! Read-only: this crate is on the read path (see PLAN.md Phase 2); `.si` writing is
//! deferred to the write-path phase.
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
//! NumSortFields --> vint (index sort — unsupported in v1, must be 0)
//! Footer
//! ```
//!
//! Index-sort (`numSortFields > 0`) is out of scope for v1: `SortFieldProvider`
//! implementations are pluggable Java classes, so faithfully decoding arbitrary
//! sort fields means either a matching plugin registry or a hard-coded allowlist.
//! We surface it as `Error::Unsupported` rather than silently mis-parsing.

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

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
    #[error("index-sorted segments are not yet supported (numSortFields={0})")]
    UnsupportedIndexSort(i32),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LuceneVersion {
    pub major: i32,
    pub minor: i32,
    pub bugfix: i32,
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
}

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
    if num_sort_fields > 0 {
        return Err(Error::UnsupportedIndexSort(num_sort_fields));
    }

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
    })
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
            }
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
    fn positive_sort_field_count_is_unsupported() {
        let mut b = SiBuilder::valid();
        b.num_sort_fields = 1;
        assert!(matches!(
            parse(&b.build(), &b.id),
            Err(Error::UnsupportedIndexSort(1))
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
    fn wrong_id_rejected_with_store_error() {
        let b = SiBuilder::valid();
        let wrong_id = [9u8; ID_LENGTH];
        assert!(matches!(parse(&b.build(), &wrong_id), Err(Error::Store(_))));
    }
}
