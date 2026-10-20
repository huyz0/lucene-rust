//! `SimpleTextPointsWriter` (and `SimpleTextPointsFormat`'s extensions): every
//! points field of a segment as one text BKD tree in `.dim`, and the file
//! pointer of each tree's index in `.dii`.
//!
//! ```text
//! .dim, per field (SimpleTextBKDWriter), leaf blocks first:
//!   block count <n>
//!     doc <docID>                     (n lines)
//!     block value <BytesRef.toString> (n lines, every data dim)
//! then its index, where `.dii` points:
//!   num data dims / num index dims / bytes per dim / max leaf points
//!   index count <numLeaves>
//!   min value / max value <index dims>
//!   point count / doc count
//!     block fp <leaf block offset>    (numLeaves lines)
//!   split count <numLeaves>
//!     split dim <d>
//!     split value <bytes>             (node i at entry i; entry 0 unused)
//! END
//! checksum <CRC32>
//!
//! .dii:
//!   field count <n>
//!     field fp name <name>
//!     field fp <index file pointer>
//! checksum <CRC32>
//! ```
//!
//! What differs from Java: the writer takes each field's points as a slice
//! of `(packedValue, docID)` in the order Java's `visitDocValues` hands them
//! over (insertion order on a flush, with doc ids remapped by an index
//! sort; segment by segment on a merge, through each doc map) plus the
//! `size()` Java sizes the tree writer with, and returns both files' bytes.
//! The `.dii` lists fields in Java's `HashMap<String, Long>` iteration order
//! ([`crate::completion::java_hash_map_order`]).

use lucene_store::directory::Directory;

use super::bkd_writer::{SimpleTextBkdWriter, DEFAULT_MAX_MB_SORT_IN_HEAP};
use super::util::{write, write_checksum, write_line, write_newline};
use super::{segment_file_name, Error, Result};
use crate::bkd_writer::{BkdConfig, DEFAULT_MAX_POINTS_IN_LEAF_NODE};
use crate::field_infos::FieldInfo;

/// `SimpleTextPointsFormat.POINT_EXTENSION`.
pub const POINT_EXTENSION: &str = "dim";
/// `SimpleTextPointsFormat.POINT_INDEX_EXTENSION`.
pub const POINT_INDEX_EXTENSION: &str = "dii";

/// `SimpleTextPointsWriter.NUM_DATA_DIMS`.
pub const NUM_DATA_DIMS: &[u8] = b"num data dims ";
/// `SimpleTextPointsWriter.NUM_INDEX_DIMS`.
pub const NUM_INDEX_DIMS: &[u8] = b"num index dims ";
/// `SimpleTextPointsWriter.BYTES_PER_DIM`.
pub const BYTES_PER_DIM: &[u8] = b"bytes per dim ";
/// `SimpleTextPointsWriter.MAX_LEAF_POINTS`.
pub const MAX_LEAF_POINTS: &[u8] = b"max leaf points ";
/// `SimpleTextPointsWriter.INDEX_COUNT`.
pub const INDEX_COUNT: &[u8] = b"index count ";
/// `SimpleTextPointsWriter.BLOCK_COUNT`.
pub const BLOCK_COUNT: &[u8] = b"block count ";
/// `SimpleTextPointsWriter.BLOCK_DOC_ID`.
pub const BLOCK_DOC_ID: &[u8] = b"  doc ";
/// `SimpleTextPointsWriter.BLOCK_FP`.
pub const BLOCK_FP: &[u8] = b"  block fp ";
/// `SimpleTextPointsWriter.BLOCK_VALUE`.
pub const BLOCK_VALUE: &[u8] = b"  block value ";
/// `SimpleTextPointsWriter.SPLIT_COUNT`.
pub const SPLIT_COUNT: &[u8] = b"split count ";
/// `SimpleTextPointsWriter.SPLIT_DIM`.
pub const SPLIT_DIM: &[u8] = b"  split dim ";
/// `SimpleTextPointsWriter.SPLIT_VALUE`.
pub const SPLIT_VALUE: &[u8] = b"  split value ";
/// `SimpleTextPointsWriter.FIELD_COUNT`.
pub const FIELD_COUNT: &[u8] = b"field count ";
/// `SimpleTextPointsWriter.FIELD_FP_NAME`.
pub const FIELD_FP_NAME: &[u8] = b"  field fp name ";
/// `SimpleTextPointsWriter.FIELD_FP`.
pub const FIELD_FP: &[u8] = b"  field fp ";
/// `SimpleTextPointsWriter.MIN_VALUE`.
pub const MIN_VALUE: &[u8] = b"min value ";
/// `SimpleTextPointsWriter.MAX_VALUE`.
pub const MAX_VALUE: &[u8] = b"max value ";
/// `SimpleTextPointsWriter.POINT_COUNT`.
pub const POINT_COUNT: &[u8] = b"point count ";
/// `SimpleTextPointsWriter.DOC_COUNT`.
pub const DOC_COUNT: &[u8] = b"doc count ";
/// `SimpleTextPointsWriter.END`.
pub const END: &[u8] = b"END";

/// The `.dim` file's name: `IndexFileNames.segmentFileName(segment, suffix,
/// "dim")`.
pub fn points_file_name(segment: &str, suffix: &str) -> String {
    segment_file_name(segment, suffix, POINT_EXTENSION)
}

/// The `.dii` file's name.
pub fn points_index_file_name(segment: &str, suffix: &str) -> String {
    segment_file_name(segment, suffix, POINT_INDEX_EXTENSION)
}

/// `org.apache.lucene.codecs.simpletext.SimpleTextPointsWriter`.
pub struct SimpleTextPointsWriter<'d> {
    segment: String,
    max_doc: usize,
    temp_dir: Option<&'d dyn Directory>,
    data_out: Vec<u8>,
    /// `indexFPs`, in first-`put` order (a repeated name keeps its slot and
    /// takes the last pointer).
    index_fps: Vec<(String, i64)>,
}

impl<'d> SimpleTextPointsWriter<'d> {
    /// `new SimpleTextPointsWriter(writeState)` for segment `segment` of
    /// `max_doc` documents; `temp_dir` (Java: the segment's directory) takes
    /// the tree writer's spill files, needed only for a field of more points
    /// than its 16 MB heap budget holds.
    pub fn new(segment: &str, max_doc: usize, temp_dir: Option<&'d dyn Directory>) -> Self {
        SimpleTextPointsWriter {
            segment: segment.to_string(),
            max_doc,
            temp_dir,
            data_out: Vec::new(),
            index_fps: Vec::new(),
        }
    }

    /// `SimpleTextPointsWriter.writeField`: `points` as `visitDocValues`
    /// visits them, `size` what the tree's `size()` reports (on a merge, the
    /// sources' sizes including deleted documents).
    ///
    /// # Errors
    /// A point configuration `BKDConfig` refuses, a value of the wrong
    /// length, more points than `size`, temp-file errors.
    pub fn write_field<'v>(
        &mut self,
        field_info: &FieldInfo,
        size: u64,
        points: impl IntoIterator<Item = (&'v [u8], i32)>,
    ) -> Result<()> {
        let dims = |v: i32| usize::try_from(v).unwrap_or(0);
        let config = BkdConfig::new(
            dims(field_info.point_dimension_count),
            dims(field_info.point_index_dimension_count),
            dims(field_info.point_num_bytes),
            DEFAULT_MAX_POINTS_IN_LEAF_NODE,
        )?;
        let mut writer = SimpleTextBkdWriter::new(
            self.max_doc,
            self.temp_dir,
            &self.segment,
            config,
            DEFAULT_MAX_MB_SORT_IN_HEAP,
            size,
        )?;
        for (packed_value, doc_id) in points {
            writer.add(packed_value, doc_id)?;
        }
        // We could have 0 points on merge since all docs with points may be
        // deleted.
        if writer.point_count() > 0 {
            let fp = writer.finish(&mut self.data_out)?;
            match self
                .index_fps
                .iter_mut()
                .find(|(name, _)| *name == field_info.name)
            {
                Some(entry) => entry.1 = fp,
                None => self.index_fps.push((field_info.name.clone(), fp)),
            }
        }
        Ok(())
    }

    /// `SimpleTextPointsWriter.finish` then `close`: the `.dim` and `.dii`
    /// files' bytes.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        write(&mut self.data_out, END);
        write_newline(&mut self.data_out);
        write_checksum(&mut self.data_out);
        let mut index = Vec::new();
        write_line(
            &mut index,
            FIELD_COUNT,
            self.index_fps.len().to_string().as_bytes(),
        );
        let names: Vec<&str> = self.index_fps.iter().map(|(n, _)| n.as_str()).collect();
        for i in crate::completion::java_hash_map_order(&names) {
            if let Some((name, fp)) = self.index_fps.get(i) {
                write_line(&mut index, FIELD_FP_NAME, name.as_bytes());
                write_line(&mut index, FIELD_FP, fp.to_string().as_bytes());
            }
        }
        write_checksum(&mut index);
        (self.data_out, index)
    }
}

impl From<crate::bkd_writer::BkdError> for Error {
    fn from(e: crate::bkd_writer::BkdError) -> Self {
        match e {
            crate::bkd_writer::BkdError::IllegalArgument(m) => Error::IllegalArgument(m),
            crate::bkd_writer::BkdError::Store(s) => Error::Store(s),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::simpletext::points_reader::SimpleTextPointsReader;

    #[test]
    fn repeated_and_empty_fields() {
        let fi = FieldInfo::new("a", 0).with_points(1, 1, 1);
        let mut w = SimpleTextPointsWriter::new("_0", 4, None);
        w.write_field(&fi, 1, [(&[1u8][..], 0)]).unwrap();
        // No points (a merge whose documents were all deleted): no tree.
        let empty = FieldInfo::new("b", 1).with_points(1, 1, 1);
        w.write_field(&empty, 0, std::iter::empty()).unwrap();
        // The same name again keeps its slot and takes the new tree.
        w.write_field(&fi, 1, [(&[7u8][..], 3)]).unwrap();
        let (dim, dii) = w.finish();
        assert!(dii.starts_with(b"field count 1\n"));
        let r = SimpleTextPointsReader::open(&dim, &dii).unwrap();
        r.check_integrity().unwrap();
        assert_eq!(r.field("a").unwrap().min_packed_value(), [7]);
        assert!(r.field("b").is_none());
    }

    #[test]
    fn refuses_what_bkd_config_refuses() {
        let fi = FieldInfo::new("a", 0).with_points(1, 1, 17);
        let mut w = SimpleTextPointsWriter::new("_0", 4, None);
        let e = w.write_field(&fi, 1, [(&[1u8; 17][..], 0)]).unwrap_err();
        assert!(matches!(e, Error::IllegalArgument(_)), "{e}");
        let e = Error::from(crate::bkd_writer::BkdError::Store(
            lucene_store::Error::Eof { offset: 0 },
        ));
        assert!(matches!(e, Error::Store(_)));
    }
}
