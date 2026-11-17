//! The files of Lucene's sandbox `FaissKnnVectorsFormat` (Lucene `main`,
//! 11.0; no backward-compatibility guarantee upstream):
//!
//! * `.faissm` (`FaissKnnVectorsFormatMeta`, version 0): index header, then per
//!   field `fieldNumber(i32) offset(i64) length(i64)` and a closing `-1(i32)`,
//!   footer.
//! * `.faissd` (`FaissKnnVectorsFormatData`, version 0): index header, the
//!   fields' Faiss indexes ([`crate::format`]) back to back, footer.
//!
//! The format also writes the raw vectors with `Lucene99FlatVectorsFormat`,
//! which this crate leaves to the codec that owns it; the Faiss index holds
//! the vectors again, as Faiss's own storage.

use lucene_store::codec_util::{
    check_index_header, check_whole_file_footer, write_footer, write_index_header, ID_LENGTH,
};
use lucene_store::data_input::{DataInput, SliceInput};
use lucene_store::data_output::DataOutput;

use crate::error::{corrupt, Error, Result};
use crate::format::{read_index, write_index, Index};

/// `FaissKnnVectorsFormat.NAME`.
pub const NAME: &str = "FaissKnnVectorsFormat";
/// `META_CODEC_NAME`.
pub const META_CODEC_NAME: &str = "FaissKnnVectorsFormatMeta";
/// `DATA_CODEC_NAME`.
pub const DATA_CODEC_NAME: &str = "FaissKnnVectorsFormatData";
/// `META_EXTENSION`.
pub const META_EXTENSION: &str = "faissm";
/// `DATA_EXTENSION`.
pub const DATA_EXTENSION: &str = "faissd";
/// `VERSION_START` = `VERSION_CURRENT`.
pub const VERSION: i32 = 0;

/// Writes one segment's `.faissm` and `.faissd`.
pub struct FaissFilesWriter {
    id: [u8; ID_LENGTH],
    suffix: String,
    meta: Vec<u8>,
    data: Vec<u8>,
}

impl FaissFilesWriter {
    /// Starts both files for the segment with object id `id`.
    pub fn new(id: [u8; ID_LENGTH], suffix: &str) -> Self {
        let mut meta = Vec::new();
        write_index_header(&mut meta, META_CODEC_NAME, VERSION, &id, suffix);
        let mut data = Vec::new();
        write_index_header(&mut data, DATA_CODEC_NAME, VERSION, &id, suffix);
        FaissFilesWriter {
            id,
            suffix: suffix.to_string(),
            meta,
            data,
        }
    }

    /// Adds field `number`'s index.
    pub fn add_field(&mut self, number: i32, index: &Index) {
        let bytes = write_index(index);
        self.meta.write_i32(number);
        self.meta.write_i64(self.data.len() as i64);
        self.meta.write_i64(bytes.len() as i64);
        self.data.extend_from_slice(&bytes);
    }

    /// Closes both files: `(meta, data)`.
    pub fn finish(mut self) -> (Vec<u8>, Vec<u8>) {
        let _ = (&self.id, &self.suffix);
        self.meta.write_i32(-1);
        write_footer(&mut self.meta);
        write_footer(&mut self.data);
        (self.meta, self.data)
    }
}

/// The Faiss indexes of one segment, by field number.
#[derive(Debug)]
pub struct FaissSegment {
    /// `(field number, index)` in file order.
    pub fields: Vec<(i32, Index)>,
}

/// Reads one segment's `.faissm` and `.faissd`, checking headers (codec,
/// version, segment id, suffix), footers, and that the two files agree on
/// their version.
///
/// # Errors
/// [`Error::Store`] for a header or checksum failure, [`Error::Corrupt`] for
/// a field extent outside the data file or a bad Faiss index.
pub fn read_segment(
    meta: &[u8],
    data: &[u8],
    id: &[u8; ID_LENGTH],
    suffix: &str,
) -> Result<FaissSegment> {
    let mut m = SliceInput::new(meta);
    let meta_header = check_index_header(&mut m, META_CODEC_NAME, VERSION, VERSION, id, suffix)?;
    let mut extents = Vec::new();
    loop {
        let number = m.read_i32().map_err(Error::from)?;
        if number == -1 {
            break;
        }
        let offset = m.read_i64().map_err(Error::from)?;
        let length = m.read_i64().map_err(Error::from)?;
        extents.push((number, offset, length));
    }
    check_whole_file_footer(meta, m.position())?;
    let mut d = SliceInput::new(data);
    let data_header = check_index_header(&mut d, DATA_CODEC_NAME, VERSION, VERSION, id, suffix)?;
    if meta_header.version != data_header.version {
        return corrupt(format!(
            "format versions mismatch (meta={}, data={})",
            meta_header.version, data_header.version
        ));
    }
    let payload_start = d.position();
    let payload_end = data
        .len()
        .saturating_sub(lucene_store::codec_util::FOOTER_LENGTH);
    check_whole_file_footer(data, payload_end)?;
    let mut fields = Vec::with_capacity(extents.len());
    for (number, offset, length) in extents {
        if fields.iter().any(|(n, _)| *n == number) {
            return corrupt(format!("duplicate field number {number}"));
        }
        let (Ok(offset), Ok(length)) = (usize::try_from(offset), usize::try_from(length)) else {
            return corrupt(format!("field {number} extent"));
        };
        let end = offset.checked_add(length);
        let Some(end) = end.filter(|&e| offset >= payload_start && e <= payload_end) else {
            return corrupt(format!("field {number}'s index lies outside the data file"));
        };
        fields.push((number, read_index(&data[offset..end])?));
    }
    Ok(FaissSegment { fields })
}
