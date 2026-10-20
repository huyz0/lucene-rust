//! `SimpleTextStoredFieldsWriter` (and `SimpleTextStoredFieldsFormat`'s
//! file name): a segment's stored fields as one `.fld` text file.
//!
//! ```text
//! doc <n>                      per document, 0..numDocs
//!   field <field number>       per stored value, in the order added
//!     name <field name>
//!     type <string|binary|int|long|float|double>
//!     value <the value>        strings and bytes escaped; numbers as Java's
//!                              Integer/Long/Float/Double.toString
//! END
//! checksum <20 digits>
//! ```
//!
//! There is no index file: the reader
//! ([`super::stored_fields_reader::SimpleTextStoredFieldsReader`]) walks the
//! whole file once at open and keeps where each document's lines start.

use lucene_util::geo::{java_double_string, java_float_string};

use super::util;
use super::{Error, Result};
use crate::field_infos::{FieldInfo, FieldInfos};
use crate::stored_fields::{Document, FieldValue};

/// `SimpleTextStoredFieldsWriter.FIELDS_EXTENSION`.
pub const FIELDS_EXTENSION: &str = "fld";
/// `TYPE_STRING`.
pub const TYPE_STRING: &[u8] = b"string";
/// `TYPE_BINARY`.
pub const TYPE_BINARY: &[u8] = b"binary";
/// `TYPE_INT`.
pub const TYPE_INT: &[u8] = b"int";
/// `TYPE_LONG`.
pub const TYPE_LONG: &[u8] = b"long";
/// `TYPE_FLOAT`.
pub const TYPE_FLOAT: &[u8] = b"float";
/// `TYPE_DOUBLE`.
pub const TYPE_DOUBLE: &[u8] = b"double";
/// `END`.
pub const END: &[u8] = b"END";
/// `DOC`.
pub const DOC: &[u8] = b"doc ";
/// `FIELD`.
pub const FIELD: &[u8] = b"  field ";
/// `NAME`.
pub const NAME: &[u8] = b"    name ";
/// `TYPE`.
pub const TYPE: &[u8] = b"    type ";
/// `VALUE`.
pub const VALUE: &[u8] = b"    value ";

/// `IndexFileNames.segmentFileName(segment, "", "fld")`.
pub fn stored_fields_file_name(segment: &str) -> String {
    super::segment_file_name(segment, "", FIELDS_EXTENSION)
}

/// `SimpleTextStoredFieldsWriter`: the `.fld` is built in memory and
/// returned by [`Self::finish`].
#[derive(Debug, Default)]
pub struct SimpleTextStoredFieldsWriter {
    out: Vec<u8>,
    num_docs_written: i32,
}

impl SimpleTextStoredFieldsWriter {
    /// `new SimpleTextStoredFieldsWriter(directory, segment, context)`.
    pub fn new() -> Self {
        Self::default()
    }

    /// `startDocument()`.
    pub fn start_document(&mut self) {
        // Java: SimpleTextStoredFieldsWriter.startDocument
        util::write_line(
            &mut self.out,
            DOC,
            self.num_docs_written.to_string().as_bytes(),
        );
        self.num_docs_written = self.num_docs_written.saturating_add(1);
    }

    /// `writeField(info, value)`, Java's six overloads in one: `field` and
    /// `name` (Java's private `writeField(info)`), then `type` and `value`.
    pub fn write_field(&mut self, info: &FieldInfo, value: &FieldValue) {
        // Java: SimpleTextStoredFieldsWriter.writeField(FieldInfo)
        util::write_line(&mut self.out, FIELD, info.number.to_string().as_bytes());
        util::write_line(&mut self.out, NAME, info.name.as_bytes());
        let (ty, text): (&[u8], std::borrow::Cow<'_, [u8]>) = match value {
            FieldValue::Int(v) => (TYPE_INT, v.to_string().into_bytes().into()),
            FieldValue::Long(v) => (TYPE_LONG, v.to_string().into_bytes().into()),
            FieldValue::Float(v) => (TYPE_FLOAT, java_float_string(*v).into_bytes().into()),
            FieldValue::Double(v) => (TYPE_DOUBLE, java_double_string(*v).into_bytes().into()),
            FieldValue::Binary(v) => (TYPE_BINARY, v.as_slice().into()),
            FieldValue::String(v) => (TYPE_STRING, v.as_bytes().into()),
        };
        util::write_line(&mut self.out, TYPE, ty);
        util::write_line(&mut self.out, VALUE, &text);
    }

    /// `finish(numDocs)` then `close()`: `END` and the checksum line; the
    /// whole file.
    ///
    /// # Errors
    /// [`Error::Runtime`] when `num_docs` is not the number of documents
    /// started (Java's `RuntimeException`, with its message).
    pub fn finish(mut self, num_docs: i32) -> Result<Vec<u8>> {
        // Java: SimpleTextStoredFieldsWriter.finish
        if self.num_docs_written != num_docs {
            return Err(Error::Runtime(format!(
                "mergeFields produced an invalid result: docCount is {num_docs} but only saw {} \
                 file=SimpleTextStoredFieldsWriter; now aborting this merge to prevent index \
                 corruption",
                self.num_docs_written
            )));
        }
        util::write(&mut self.out, END);
        util::write_newline(&mut self.out);
        util::write_checksum(&mut self.out);
        Ok(self.out)
    }
}

/// Every document of `docs` through a [`SimpleTextStoredFieldsWriter`] (as
/// `StoredFieldsConsumer` drives it at flush), each value's `name` line from
/// `infos`.
///
/// # Errors
/// [`Error::IllegalArgument`] for a field number `infos` lacks (Java's
/// `fieldInfos.fieldInfo(n)` is `null` there); what [`SimpleTextStoredFieldsWriter::finish`]
/// refuses for more than `i32::MAX` documents.
pub fn write(docs: &[Document], infos: &FieldInfos) -> Result<Vec<u8>> {
    let mut w = SimpleTextStoredFieldsWriter::new();
    for doc in docs {
        w.start_document();
        for f in &doc.fields {
            let info = infos.field_by_number(f.field_number).ok_or_else(|| {
                Error::IllegalArgument(format!("unknown field number {}", f.field_number))
            })?;
            w.write_field(info, &f.value);
        }
    }
    let num_docs = i32::try_from(docs.len())
        .map_err(|_| Error::IllegalArgument(format!("too many documents: {}", docs.len())))?;
    w.finish(num_docs)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stored_fields::StoredField;

    fn infos() -> FieldInfos {
        FieldInfos::new(vec![FieldInfo::new("a", 0), FieldInfo::new("b\nc", 1)]).unwrap()
    }

    #[test]
    fn every_value_type_and_its_text() {
        let docs = vec![
            Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("x\\y\nz".into()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Binary(vec![0x0a, 0x5c, 0xff]),
                    },
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Int(-7),
                    },
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Long(i64::MIN),
                    },
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Float(1.0e10),
                    },
                    StoredField {
                        field_number: 0,
                        value: FieldValue::Double(f64::MIN_POSITIVE / 2.0),
                    },
                ],
            },
            Document::default(),
        ];
        let out = write(&docs, &infos()).unwrap();
        let text = String::from_utf8_lossy(&out);
        let body = text.rsplit_once("checksum ").unwrap().0;
        assert_eq!(
            body,
            "doc 0\n  field 0\n    name a\n    type string\n    value x\\\\y\\\nz\n  field 1\n    \
             name b\\\nc\n    type binary\n    value \\\n\\\\\u{fffd}\n  field 0\n    name a\n    \
             type int\n    value -7\n  field 0\n    name a\n    type long\n    value \
             -9223372036854775808\n  field 0\n    name a\n    type float\n    value 1.0E10\n  \
             field 0\n    name a\n    type double\n    value 1.1125369292536007E-308\ndoc 1\nEND\n"
        );
        assert_eq!(stored_fields_file_name("_3"), "_3.fld");
    }

    #[test]
    fn unknown_field_number_and_doc_count_mismatch() {
        let docs = vec![Document {
            fields: vec![StoredField {
                field_number: 9,
                value: FieldValue::Int(1),
            }],
        }];
        assert!(matches!(
            write(&docs, &infos()),
            Err(Error::IllegalArgument(_))
        ));
        let mut w = SimpleTextStoredFieldsWriter::new();
        w.start_document();
        let err = w.finish(2).unwrap_err().to_string();
        assert!(err.contains("docCount is 2 but only saw 1"), "{err}");
    }
}
