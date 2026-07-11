//! Port of `org.apache.lucene.index.SegmentInfos` (`segments_N` commit files).
//!
//! This is the top of the read path: `segments_N` is the file a `DirectoryReader`
//! opens first — it lists every segment in the commit (by name + id + codec) along
//! with per-segment delete/DV-update generations, but does *not* embed the segments'
//! own metadata (doc count, compound-file flag, ...). That lives in each segment's
//! `.si` file, parsed separately by [`crate::segment_info`]. Callers resolve
//! `SegmentCommitInfo::segment_name` to `<name>.si` themselves — this module has no
//! `Directory` dependency yet (Phase 1, still to come).
//!
//! Wire format (all ints little-endian unless noted "BE"; header/footer/BE
//! primitives per `lucene_store::codec_util`):
//! ```text
//! Header       --> IndexHeader(codec="segments", version in [VERSION_74, VERSION_CURRENT],
//!                   id, suffix=generation formatted base-36)
//! LuceneVersion --> vint major, vint minor, vint bugfix   (note: vint here, NOT the
//!                    fixed-i32 triple `.si` uses for its own SegVersion)
//! IndexCreatedVersionMajor --> vint
//! Version      --> BEi64             (commit's own monotonic version counter)
//! Counter      --> vlong             (next segment-name counter)
//! NumSegments  --> BEi32
//! MinSegmentLuceneVersion --> vint triple, present iff NumSegments > 0
//! per segment:
//!   SegName        --> String
//!   SegID          --> [u8; 16]
//!   CodecName      --> String
//!   DelGen         --> BEi64
//!   DelCount       --> BEi32
//!   FieldInfosGen  --> BEi64
//!   DocValuesGen   --> BEi64
//!   SoftDelCount   --> BEi32
//!   SciIdMarker    --> u8 (only if format > VERSION_74); 1 => SciId: [u8; 16] follows
//!   FieldInfosFiles --> SetOfStrings
//!   NumDVFields    --> BEi32
//!   per DV field: FieldNumber --> BEi32, Files --> SetOfStrings
//! UserData     --> MapOfStrings
//! Footer
//! ```

use lucene_store::codec_util::{self, ID_LENGTH};
use lucene_store::data_input::{DataInput, SliceInput};

const CODEC_NAME: &str = "segments";
pub const VERSION_74: i32 = 9;
pub const VERSION_86: i32 = 10;
const VERSION_CURRENT: i32 = VERSION_86;

#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error(transparent)]
    Store(#[from] lucene_store::Error),
    #[error("invalid segment count: {0}")]
    InvalidSegmentCount(i32),
    #[error("invalid deletion count: {0} vs maxDoc unknown at this layer (segment={1})")]
    InvalidDeletionCount(i32, String),
    #[error("invalid SegmentCommitInfo ID marker: {0}")]
    InvalidSciIdMarker(u8),
}

pub type Result<T> = std::result::Result<T, Error>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LuceneVersion {
    pub major: i32,
    pub minor: i32,
    pub bugfix: i32,
}

/// One segment's entry in a commit: everything `segments_N` records about it,
/// *excluding* what lives in the segment's own `.si` file.
#[derive(Debug, Clone)]
pub struct SegmentCommitInfo {
    pub segment_name: String,
    pub segment_id: [u8; ID_LENGTH],
    pub codec_name: String,
    pub del_gen: i64,
    pub del_count: i32,
    pub field_infos_gen: i64,
    pub doc_values_gen: i64,
    pub soft_del_count: i32,
    /// Present from format > VERSION_74 only.
    pub sci_id: Option<[u8; ID_LENGTH]>,
    pub field_infos_files: Vec<String>,
    /// field number -> doc-values update files for that field.
    pub dv_update_files: Vec<(i32, Vec<String>)>,
}

#[derive(Debug, Clone)]
pub struct SegmentInfos {
    pub id: [u8; ID_LENGTH],
    pub generation: i64,
    pub format_version: i32,
    pub lucene_version: LuceneVersion,
    pub index_created_version_major: i32,
    /// Commit's own monotonic version counter (`SegmentInfos.version`).
    pub version: i64,
    /// Next unused segment-name counter (`SegmentInfos.counter`).
    pub counter: i64,
    pub min_segment_lucene_version: Option<LuceneVersion>,
    pub segments: Vec<SegmentCommitInfo>,
    pub user_data: Vec<(String, String)>,
}

/// Parses a whole `segments_N` file already read into memory.
///
/// `generation` is the `N` from the filename (or the special generation for
/// `segments.gen`-less setups) — Lucene encodes it as a base-36 string in the
/// index header's suffix and we must match it exactly, just like the codec name
/// and id.
pub fn parse(buf: &[u8], generation: i64) -> Result<SegmentInfos> {
    let mut input = SliceInput::new(buf);

    let suffix = lucene_util::base36::to_base36(generation);
    // We don't yet know `id` (it's inside the file), so check the header without
    // the id/suffix-bound convenience wrapper and validate the suffix by hand —
    // mirrors Java's `checkHeaderNoMagic` + manual `checkIndexHeaderSuffix` split.
    let header = codec_util::check_header(&mut input, CODEC_NAME, VERSION_74, VERSION_CURRENT)?;
    let mut id = [0u8; ID_LENGTH];
    input.read_bytes(&mut id)?;
    codec_util::check_index_header_suffix(&mut input, &suffix)?;

    let lucene_version = read_vint_version(&mut input)?;
    let index_created_version_major = input.read_vint()?;

    let version = input.read_be_u64()? as i64;
    let counter = input.read_vlong()?;
    let num_segments = input.read_be_i32()?;
    if num_segments < 0 {
        return Err(Error::InvalidSegmentCount(num_segments));
    }

    let min_segment_lucene_version = if num_segments > 0 {
        Some(read_vint_version(&mut input)?)
    } else {
        None
    };

    let mut segments = Vec::with_capacity(num_segments as usize);
    for _ in 0..num_segments {
        let segment_name = input.read_string()?;
        let mut segment_id = [0u8; ID_LENGTH];
        input.read_bytes(&mut segment_id)?;
        let codec_name = input.read_string()?;

        let del_gen = input.read_be_u64()? as i64;
        let del_count = input.read_be_i32()?;
        if del_count < 0 {
            return Err(Error::InvalidDeletionCount(del_count, segment_name));
        }
        let field_infos_gen = input.read_be_u64()? as i64;
        let doc_values_gen = input.read_be_u64()? as i64;
        let soft_del_count = input.read_be_i32()?;
        if soft_del_count < 0 {
            return Err(Error::InvalidDeletionCount(soft_del_count, segment_name));
        }

        let sci_id = if header.version > VERSION_74 {
            match input.read_byte()? {
                0 => None,
                1 => {
                    let mut sci = [0u8; ID_LENGTH];
                    input.read_bytes(&mut sci)?;
                    Some(sci)
                }
                other => return Err(Error::InvalidSciIdMarker(other)),
            }
        } else {
            None
        };

        let field_infos_files = input.read_set_of_strings()?;
        let num_dv_fields = input.read_be_i32()?;
        let mut dv_update_files = Vec::with_capacity(num_dv_fields.max(0) as usize);
        for _ in 0..num_dv_fields {
            let field_number = input.read_be_i32()?;
            let files = input.read_set_of_strings()?;
            dv_update_files.push((field_number, files));
        }

        segments.push(SegmentCommitInfo {
            segment_name,
            segment_id,
            codec_name,
            del_gen,
            del_count,
            field_infos_gen,
            doc_values_gen,
            soft_del_count,
            sci_id,
            field_infos_files,
            dv_update_files,
        });
    }

    let user_data = input.read_map_of_strings()?;

    codec_util::check_footer(&mut input, buf.len())?;

    Ok(SegmentInfos {
        id,
        generation,
        format_version: header.version,
        lucene_version,
        index_created_version_major,
        version,
        counter,
        min_segment_lucene_version,
        segments,
        user_data,
    })
}

fn read_vint_version(input: &mut SliceInput) -> Result<LuceneVersion> {
    let major = input.read_vint()?;
    let minor = input.read_vint()?;
    let bugfix = input.read_vint()?;
    Ok(LuceneVersion {
        major,
        minor,
        bugfix,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Test-only `segments_N` byte builder, independent of the real
    /// `IndexWriter`-generated fixture under `tests/segment_infos_fixtures.rs`:
    /// that exercises real bytes end-to-end, this covers error paths (negative
    /// counts, bad markers, multiple segments/DV fields) that a real writer
    /// would never produce.
    struct SegBuilder {
        name: String,
        id: [u8; ID_LENGTH],
        codec: String,
        del_gen: i64,
        del_count: i32,
        field_infos_gen: i64,
        doc_values_gen: i64,
        soft_del_count: i32,
        sci_marker: Option<u8>, // None => omit entirely (format <= VERSION_74)
        dv_fields: Vec<(i32, Vec<String>)>,
    }

    impl SegBuilder {
        fn valid(name: &str) -> Self {
            Self {
                name: name.to_string(),
                id: [2u8; ID_LENGTH],
                codec: "Lucene104".to_string(),
                del_gen: -1,
                del_count: 0,
                field_infos_gen: -1,
                doc_values_gen: -1,
                soft_del_count: 0,
                sci_marker: Some(0),
                dv_fields: vec![],
            }
        }
    }

    struct SisBuilder {
        generation: i64,
        format_version: i32,
        id: [u8; ID_LENGTH],
        segments: Vec<SegBuilder>,
        num_segments_override: Option<i32>,
        user_data: Vec<(String, String)>,
    }

    impl SisBuilder {
        fn valid(generation: i64) -> Self {
            Self {
                generation,
                format_version: VERSION_86,
                id: [3u8; ID_LENGTH],
                segments: vec![],
                num_segments_override: None,
                user_data: vec![],
            }
        }

        fn build(&self) -> Vec<u8> {
            let mut out = Vec::new();
            out.extend_from_slice(&codec_util::CODEC_MAGIC.to_be_bytes());
            write_string(&mut out, CODEC_NAME);
            out.extend_from_slice(&(self.format_version as u32).to_be_bytes());
            out.extend_from_slice(&self.id);
            let suffix = lucene_util::base36::to_base36(self.generation);
            out.push(suffix.len() as u8);
            out.extend_from_slice(suffix.as_bytes());

            write_vint(&mut out, 10); // lucene_version major
            write_vint(&mut out, 0); // minor
            write_vint(&mut out, 0); // bugfix
            write_vint(&mut out, 10); // indexCreatedVersionMajor

            out.extend_from_slice(&1u64.to_be_bytes()); // commit version
            write_vlong(&mut out, 1); // counter

            let num_segments = self
                .num_segments_override
                .unwrap_or(self.segments.len() as i32);
            out.extend_from_slice(&(num_segments as u32).to_be_bytes());

            if num_segments > 0 {
                write_vint(&mut out, 10); // minSegmentLuceneVersion major
                write_vint(&mut out, 0);
                write_vint(&mut out, 0);
            }

            for seg in &self.segments {
                write_string(&mut out, &seg.name);
                out.extend_from_slice(&seg.id);
                write_string(&mut out, &seg.codec);
                out.extend_from_slice(&(seg.del_gen as u64).to_be_bytes());
                out.extend_from_slice(&(seg.del_count as u32).to_be_bytes());
                out.extend_from_slice(&(seg.field_infos_gen as u64).to_be_bytes());
                out.extend_from_slice(&(seg.doc_values_gen as u64).to_be_bytes());
                out.extend_from_slice(&(seg.soft_del_count as u32).to_be_bytes());
                if self.format_version > VERSION_74 {
                    if let Some(marker) = seg.sci_marker {
                        out.push(marker);
                        if marker == 1 {
                            out.extend_from_slice(&seg.id); // reuse id as a dummy sciId
                        }
                    }
                }
                write_vint(&mut out, 0); // fieldInfosFiles: empty set
                out.extend_from_slice(&(seg.dv_fields.len() as u32).to_be_bytes());
                for (field_number, files) in &seg.dv_fields {
                    out.extend_from_slice(&(*field_number as u32).to_be_bytes());
                    write_vint(&mut out, files.len() as i32);
                    for f in files {
                        write_string(&mut out, f);
                    }
                }
            }

            write_vint(&mut out, self.user_data.len() as i32);
            for (k, v) in &self.user_data {
                write_string(&mut out, k);
                write_string(&mut out, v);
            }

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

    fn write_vlong(out: &mut Vec<u8>, mut v: i64) {
        loop {
            let mut b = (v & 0x7f) as u8;
            v = ((v as u64) >> 7) as i64;
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
    fn empty_commit_no_segments() {
        let b = SisBuilder::valid(1);
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments.len(), 0);
        assert!(sis.min_segment_lucene_version.is_none());
    }

    #[test]
    fn single_segment_no_sci_id_and_no_dv_fields() {
        let mut b = SisBuilder::valid(2);
        b.segments.push(SegBuilder::valid("_0"));
        let sis = parse(&b.build(), 2).unwrap();
        assert_eq!(sis.segments.len(), 1);
        assert!(sis.segments[0].sci_id.is_none());
        assert!(sis.min_segment_lucene_version.is_some());
    }

    #[test]
    fn segment_with_sci_id_present() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.sci_marker = Some(1);
        b.segments.push(seg);
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments[0].sci_id, Some([2u8; ID_LENGTH]));
    }

    #[test]
    fn format_at_version_74_omits_sci_marker_entirely() {
        let mut b = SisBuilder::valid(1);
        b.format_version = VERSION_74;
        let mut seg = SegBuilder::valid("_0");
        seg.sci_marker = None; // omitted at this format version, per real Lucene
        b.segments.push(seg);
        let sis = parse(&b.build(), 1).unwrap();
        assert!(sis.segments[0].sci_id.is_none());
    }

    #[test]
    fn doc_values_update_fields_are_parsed() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.dv_fields = vec![
            (0, vec!["_0_1.dvd".to_string()]),
            (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
        ];
        b.segments.push(seg);
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments[0].dv_update_files, seg_dv_fields());
    }

    fn seg_dv_fields() -> Vec<(i32, Vec<String>)> {
        vec![
            (0, vec!["_0_1.dvd".to_string()]),
            (2, vec!["_0_2.dvd".to_string(), "_0_2b.dvd".to_string()]),
        ]
    }

    #[test]
    fn multiple_segments_and_user_data() {
        let mut b = SisBuilder::valid(1);
        b.segments.push(SegBuilder::valid("_0"));
        b.segments.push(SegBuilder::valid("_1"));
        b.user_data.push(("k".to_string(), "v".to_string()));
        let sis = parse(&b.build(), 1).unwrap();
        assert_eq!(sis.segments.len(), 2);
        assert_eq!(sis.user_data, vec![("k".to_string(), "v".to_string())]);
    }

    #[test]
    fn negative_num_segments_rejected() {
        let mut b = SisBuilder::valid(1);
        b.num_segments_override = Some(-1);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidSegmentCount(-1))
        ));
    }

    #[test]
    fn negative_del_count_rejected() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.del_count = -1;
        b.segments.push(seg);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidDeletionCount(-1, name)) if name == "_0"
        ));
    }

    #[test]
    fn negative_soft_del_count_rejected() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.soft_del_count = -1;
        b.segments.push(seg);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidDeletionCount(-1, name)) if name == "_0"
        ));
    }

    #[test]
    fn invalid_sci_marker_rejected() {
        let mut b = SisBuilder::valid(1);
        let mut seg = SegBuilder::valid("_0");
        seg.sci_marker = Some(7); // neither 0 nor 1
        b.segments.push(seg);
        assert!(matches!(
            parse(&b.build(), 1),
            Err(Error::InvalidSciIdMarker(7))
        ));
    }

    #[test]
    fn wrong_generation_suffix_rejected() {
        let b = SisBuilder::valid(1);
        assert!(matches!(parse(&b.build(), 2), Err(Error::Store(_))));
    }
}
