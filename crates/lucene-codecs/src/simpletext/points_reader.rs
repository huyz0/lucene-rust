//! `SimpleTextPointsReader`: the `.dii` field pointers, then each field's
//! tree index (`initReader`) from the `.dim` written by
//! [`super::points_writer`].
//!
//! What differs from Java: the reader borrows both files' bytes. Every
//! `assert` on a line's prefix or a value's length, every
//! `NumberFormatException`, `NegativeArraySizeException` (a negative index
//! or split count) and seek outside the `.dim` is `Corrupted` (or `Eof`); so
//! is a configuration `BKDConfig` refuses (Java's `IllegalArgumentException`)
//! and a split dim past the index dims (Java reads it as `& 0xff` and fails
//! on the array access when walking). Arrays grow with the lines read
//! rather than being sized from a count.

use super::bkd_reader::{bytes_per_index_entry, SimpleTextBkdReader};
use super::points_writer::{
    BLOCK_FP, BYTES_PER_DIM, DOC_COUNT, FIELD_COUNT, FIELD_FP, FIELD_FP_NAME, INDEX_COUNT,
    MAX_LEAF_POINTS, MAX_VALUE, MIN_VALUE, NUM_DATA_DIMS, NUM_INDEX_DIMS, POINT_COUNT, SPLIT_COUNT,
    SPLIT_DIM, SPLIT_VALUE,
};
use super::util::{
    self, from_bytes_ref_string, parse_integer, parse_long, read_string, read_value,
};
use super::{corrupt, file_pointer, Error, Result};
use crate::bkd_writer::BkdConfig;
use crate::field_infos::FieldInfos;

/// `org.apache.lucene.codecs.simpletext.SimpleTextPointsReader`.
#[derive(Debug, Clone)]
pub struct SimpleTextPointsReader<'a> {
    data: &'a [u8],
    /// `readers`, in the `.dii`'s order (a repeated name keeps its first
    /// slot and its last tree, as `HashMap.put`).
    readers: Vec<(String, SimpleTextBkdReader<'a>)>,
}

impl<'a> SimpleTextPointsReader<'a> {
    /// `new SimpleTextPointsReader(readState)` over the `.dim` and `.dii`
    /// bytes: the `.dii` is checksummed, the `.dim` only by
    /// [`Self::check_integrity`].
    ///
    /// # Errors
    /// `Corrupted` or `Eof` for a damaged file.
    pub fn open(dim: &'a [u8], dii: &[u8]) -> Result<Self> {
        let mut scratch = Vec::new();
        let mut pos = 0usize;
        let count = parse_integer(read_value(dii, &mut pos, &mut scratch, FIELD_COUNT)?)?;
        let mut field_to_file_offset: Vec<(String, i64)> = Vec::new();
        for _ in 0..count.max(0) {
            let name = read_string(dii, &mut pos, &mut scratch, FIELD_FP_NAME)?;
            let fp = parse_long(read_value(dii, &mut pos, &mut scratch, FIELD_FP)?)?;
            match field_to_file_offset.iter_mut().find(|(n, _)| *n == name) {
                Some(entry) => entry.1 = fp,
                None => field_to_file_offset.push((name, fp)),
            }
        }
        util::check_footer(dii, 0, &mut pos)?;
        let mut readers = Vec::new();
        for (name, fp) in field_to_file_offset {
            let reader = init_reader(dim, fp)?;
            readers.push((name, reader));
        }
        Ok(SimpleTextPointsReader { data: dim, readers })
    }

    /// `getValues(fieldName)`: the field's tree, `None` for a points field
    /// with no tree in this segment.
    ///
    /// # Errors
    /// `IllegalArgument` with Java's messages for a field `field_infos` does
    /// not know or one without points.
    pub fn values(
        &self,
        field_infos: &FieldInfos,
        field_name: &str,
    ) -> Result<Option<&SimpleTextBkdReader<'a>>> {
        let Some(field_info) = field_infos.field_by_name(field_name) else {
            return Err(Error::IllegalArgument(format!(
                "field=\"{field_name}\" is unrecognized"
            )));
        };
        if field_info.point_dimension_count == 0 {
            return Err(Error::IllegalArgument(format!(
                "field=\"{field_name}\" did not index points"
            )));
        }
        Ok(self.field(field_name))
    }

    /// The tree of `field_name`, if the `.dii` lists it.
    pub fn field(&self, field_name: &str) -> Option<&SimpleTextBkdReader<'a>> {
        self.readers
            .iter()
            .find(|(n, _)| n == field_name)
            .map(|(_, r)| r)
    }

    /// The fields the `.dii` lists, in its order.
    pub fn field_names(&self) -> impl Iterator<Item = &str> {
        self.readers.iter().map(|(n, _)| n.as_str())
    }

    /// `checkIntegrity()`: the `.dim`'s checksum.
    ///
    /// # Errors
    /// `Corrupted` for a footer elsewhere or a wrong checksum.
    pub fn check_integrity(&self) -> Result<()> {
        // Java: SimpleTextPointsReader.checkIntegrity
        util::check_integrity(self.data)
    }
}

/// `initReader(fp)`: one field's index lines.
fn init_reader(data: &[u8], fp: i64) -> Result<SimpleTextBkdReader<'_>> {
    let mut pos = file_pointer(data, fp, "SimpleText points: field fp")?;
    let mut scratch = Vec::new();
    let mut int = |pos: &mut usize, prefix: &[u8]| -> Result<i32> {
        parse_integer(read_value(data, pos, &mut scratch, prefix)?)
    };
    let num_data_dims = int(&mut pos, NUM_DATA_DIMS)?;
    let num_index_dims = int(&mut pos, NUM_INDEX_DIMS)?;
    let bytes_per_dim = int(&mut pos, BYTES_PER_DIM)?;
    let max_points_in_leaf_node = int(&mut pos, MAX_LEAF_POINTS)?;
    let count = int(&mut pos, INDEX_COUNT)?;
    let dim = |v: i32| usize::try_from(v).unwrap_or(0);
    let config = BkdConfig::new(
        dim(num_data_dims),
        dim(num_index_dims),
        dim(bytes_per_dim),
        dim(max_points_in_leaf_node),
    )
    .map_err(|e| corrupt(format!("SimpleText points: {e}")))?;
    let mut scratch = Vec::new();
    let mut bytes = |pos: &mut usize, prefix: &[u8]| -> Result<Vec<u8>> {
        let v = read_value(data, pos, &mut scratch, prefix)?;
        from_bytes_ref_string(&String::from_utf8_lossy(v))
    };
    let min_value = bytes(&mut pos, MIN_VALUE)?;
    let max_value = bytes(&mut pos, MAX_VALUE)?;
    let mut scratch = Vec::new();
    let point_count = parse_long(read_value(data, &mut pos, &mut scratch, POINT_COUNT)?)?;
    let doc_count = parse_integer(read_value(data, &mut pos, &mut scratch, DOC_COUNT)?)?;
    let count = non_negative(count, "index count")?;
    let mut leaf_block_fps = Vec::new();
    for _ in 0..count {
        leaf_block_fps.push(parse_long(read_value(
            data,
            &mut pos,
            &mut scratch,
            BLOCK_FP,
        )?)?);
    }
    let count = non_negative(
        parse_integer(read_value(data, &mut pos, &mut scratch, SPLIT_COUNT)?)?,
        "split count",
    )?;
    let entry = bytes_per_index_entry(&config);
    if i32::try_from(count.saturating_mul(entry)).is_err() {
        return Err(corrupt(format!("SimpleText points: split count {count}")));
    }
    let mut split_packed_values = Vec::new();
    for _ in 0..count {
        let split_dim = parse_integer(read_value(data, &mut pos, &mut scratch, SPLIT_DIM)?)?;
        if config.num_index_dims != 1 {
            // Java: `(byte) splitDim`, read back `& 0xff`.
            split_packed_values.push(split_dim.to_le_bytes()[0]);
        }
        let v = read_value(data, &mut pos, &mut scratch, SPLIT_VALUE)?;
        let value = from_bytes_ref_string(&String::from_utf8_lossy(v))?;
        if value.len() != config.bytes_per_dim {
            return Err(corrupt(format!(
                "SimpleText points: a split value of {} bytes, expected {}",
                value.len(),
                config.bytes_per_dim
            )));
        }
        split_packed_values.extend_from_slice(&value);
    }
    SimpleTextBkdReader::new(
        data,
        config,
        leaf_block_fps,
        split_packed_values,
        min_value,
        max_value,
        point_count,
        doc_count,
    )
}

/// A count Java sizes an array with: negative is its
/// `NegativeArraySizeException`.
fn non_negative(count: i32, what: &str) -> Result<usize> {
    usize::try_from(count)
        .map_err(|_| corrupt(format!("SimpleText points: negative {what} {count}")))
}
