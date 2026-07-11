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

    let suffix = to_base36(generation);
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

/// Java's `Long.toString(generation, Character.MAX_RADIX)` (radix 36, lowercase digits).
fn to_base36(n: i64) -> String {
    if n == 0 {
        return "0".to_string();
    }
    const DIGITS: &[u8] = b"0123456789abcdefghijklmnopqrstuvwxyz";
    let negative = n < 0;
    let mut buf = Vec::new();
    // Work in i128 so i64::MIN negation doesn't overflow.
    let mut n = n as i128;
    if negative {
        n = -n;
    }
    while n > 0 {
        buf.push(DIGITS[(n % 36) as usize]);
        n /= 36;
    }
    if negative {
        buf.push(b'-');
    }
    buf.reverse();
    String::from_utf8(buf).unwrap()
}
