//! `SimpleTextNormsFormat`: norms are NUMERIC doc values in their own file,
//! `.len`, written and read by the doc values writer and reader
//! ([`super::doc_values_writer`], [`super::doc_values_reader`]).
//!
//! What differs from Java: the consumer and producer take and return bytes
//! (see the doc values modules); the producer's `getNorms` takes a field
//! name.

use super::doc_values_reader::{SimpleTextDocValuesReader, SimpleTextNumericDocValues};
use super::doc_values_writer::SimpleTextDocValuesWriter;
use super::Result;
use crate::field_infos::FieldInfo;

/// `NORMS_SEG_EXTENSION`.
pub const NORMS_SEG_EXTENSION: &str = "len";

/// `IndexFileNames.segmentFileName(segment, suffix, "len")`.
pub fn norms_file_name(segment: &str, suffix: &str) -> String {
    super::segment_file_name(segment, suffix, NORMS_SEG_EXTENSION)
}

/// `SimpleTextNormsProducer`.
#[derive(Debug, Clone)]
pub struct SimpleTextNormsProducer<'d> {
    reader: SimpleTextDocValuesReader<'d>,
}

impl<'d> SimpleTextNormsProducer<'d> {
    /// `new SimpleTextNormsProducer(state)` over the `.len`.
    ///
    /// # Errors
    /// What [`SimpleTextDocValuesReader::open`] fails with.
    pub fn open(data: &'d [u8], max_doc: i32) -> Result<Self> {
        Ok(SimpleTextNormsProducer {
            reader: SimpleTextDocValuesReader::open(data, max_doc)?,
        })
    }

    /// `getNorms(field)`.
    ///
    /// # Errors
    /// What [`SimpleTextDocValuesReader::numeric`] fails with.
    pub fn norms(&self, field: &str) -> Result<SimpleTextNumericDocValues<'_>> {
        self.reader.numeric(field)
    }

    /// `checkIntegrity()`.
    ///
    /// # Errors
    /// What [`SimpleTextDocValuesReader::check_integrity`] fails with.
    pub fn check_integrity(&self) -> Result<()> {
        self.reader.check_integrity()
    }
}

/// `SimpleTextNormsConsumer`.
#[derive(Debug)]
pub struct SimpleTextNormsConsumer {
    writer: SimpleTextDocValuesWriter,
}

impl SimpleTextNormsConsumer {
    /// `new SimpleTextNormsConsumer(state)` for `max_doc` documents.
    pub fn new(max_doc: i32) -> Self {
        SimpleTextNormsConsumer {
            writer: SimpleTextDocValuesWriter::new(max_doc),
        }
    }

    /// `addNormsField(field, normsProducer)`: `(doc, norm)` per document
    /// with a norm.
    ///
    /// # Errors
    /// What [`SimpleTextDocValuesWriter::add_numeric_field`] fails with.
    pub fn add_norms_field(&mut self, field: &FieldInfo, norms: &[(i32, i64)]) -> Result<()> {
        self.writer.add_numeric_field(field, norms)
    }

    /// `close()`: the `.len`.
    ///
    /// # Errors
    /// What [`SimpleTextDocValuesWriter::finish`] fails with.
    pub fn finish(self) -> Result<Vec<u8>> {
        self.writer.finish()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::field_infos::IndexOptions;

    #[test]
    fn norms_round_trip() {
        let mut f = FieldInfo::new("body", 0);
        f.index_options = IndexOptions::DocsAndFreqs;
        let mut c = SimpleTextNormsConsumer::new(2);
        c.add_norms_field(&f, &[(0, 7), (1, -1)]).unwrap();
        let data = c.finish().unwrap();
        let p = SimpleTextNormsProducer::open(&data, 2).unwrap();
        p.check_integrity().unwrap();
        let mut n = p.norms("body").unwrap();
        assert_eq!(n.next_doc().unwrap(), 0);
        assert_eq!(n.long_value().unwrap(), 7);
        assert!(p.norms("tag").is_err());
        assert_eq!(norms_file_name("_0", ""), "_0.len");
        assert!(SimpleTextNormsProducer::open(b"", 2).is_err());
    }
}
