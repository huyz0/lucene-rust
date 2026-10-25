//! Asserting stored-fields writers installed in the native codec's write path.
//!
//! This adapter runs the existing Java-style `AssertingStoredFieldsWriter`
//! lifecycle over the production Lucene90 encoder. It only replaces stored
//! fields writes. The native reader graph, other formats and `Lucene104`
//! segment codec name remain in use; this is not the whole `AssertingCodec`.
//! Shared configuration contains no RNG or operation-local writer state.

use lucene_codecs::codec::Lucene104Codec;
use lucene_codecs::stored_fields::{self, FieldValue, Mode};
use lucene_codecs::stored_fields_writer::{
    StoredFieldsWriteSink, StoredFieldsWriteState, StoredFieldsWriterFactory,
};
use lucene_store::data_input::DataInput;

use super::asserting_stored_fields::AssertingStoredFieldsWriter;
use super::cranky_stored_fields::{
    Lucene90StoredFieldsFormat, StoredFieldsFormat, StoredFieldsState, StoredFieldsWriter,
};

/// Native `Lucene104` routing with assertions on every stored-fields write.
#[derive(Debug, Default, Clone, Copy)]
pub struct AssertingStoredFieldsCodec;

impl Lucene104Codec for AssertingStoredFieldsCodec {
    fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
        Some(self)
    }
}

impl StoredFieldsWriterFactory for AssertingStoredFieldsCodec {
    fn fields_writer<'a>(
        &self,
        state: StoredFieldsWriteState<'a>,
    ) -> stored_fields::Result<Box<dyn StoredFieldsWriteSink + 'a>> {
        // This installation must remain readable by the unchanged native
        // Lucene104 graph. Wider format/mode selection is a separate seam.
        if state.mode != Mode::BestSpeed || !state.suffix.is_empty() {
            return Err(lucene_store::Error::Io(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "asserting stored-fields routing requires BestSpeed and an empty suffix",
            ))
            .into());
        }
        let writer = Lucene90StoredFieldsFormat.fields_writer(StoredFieldsState {
            directory: state.directory,
            segment: state.segment,
            id: state.id,
            suffix: state.suffix,
            mode: state.mode,
        })?;
        Ok(Box::new(NativeWriteSink(AssertingStoredFieldsWriter::new(
            writer,
        ))))
    }
}

// A framework-owned newtype keeps the lifecycle assertions in their existing
// implementation without making codecs depend on the test framework.
struct NativeWriteSink<W>(W);
impl<W: StoredFieldsWriter> StoredFieldsWriteSink for NativeWriteSink<W> {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.0.start_document()
    }
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        self.0.write_field(number, value)
    }
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        self.0.write_field_data_input(number, input, length)
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.0.finish_document()
    }
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
        self.0.finish(num_docs)
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.0.close()
    }
}

#[cfg(test)]
mod tests {
    use std::panic::{catch_unwind, AssertUnwindSafe};

    use lucene_codecs::stored_fields_writer::with_writer;
    use lucene_store::data_input::SliceInput;
    use lucene_store::{ByteBuffersDirectory, Directory};

    use super::*;

    const ID: [u8; 16] = [39; 16];

    fn state(directory: &dyn Directory) -> StoredFieldsWriteState<'_> {
        StoredFieldsWriteState {
            directory,
            segment: "_test",
            id: &ID,
            suffix: "",
            mode: Mode::BestSpeed,
        }
    }

    #[test]
    fn installed_factory_writes_all_overloads_and_empty_documents_as_native_bytes() {
        let directory = ByteBuffersDirectory::new();
        let codec: &dyn Lucene104Codec = &AssertingStoredFieldsCodec;
        assert_eq!(codec.name(), "Lucene104");
        let factory = codec.stored_fields_writer_factory().unwrap();
        let values = [
            FieldValue::String("stored café".into()),
            FieldValue::Binary(vec![0, 255, 1]),
            FieldValue::Int(i32::MIN),
            FieldValue::Long(i64::MAX),
            FieldValue::Float(-0.0),
            FieldValue::Double(3.25),
        ];
        let result: stored_fields::Result<()> = with_writer(factory, state(&directory), |writer| {
            writer.start_document()?;
            for (number, value) in (0_i32..).zip(&values) {
                writer.write_field(number, value)?;
            }
            let mut input = SliceInput::new(&[12, 13, 14, 15]);
            writer.write_field_data_input(6, &mut input, 3)?;
            assert_eq!(input.read_byte().unwrap(), 15);
            writer.finish_document()?;
            writer.start_document()?;
            writer.finish_document()?;
            writer.finish(2)
        });
        result.unwrap();
        let fdt = directory.open("_test.fdt").unwrap();
        let fdx = directory.open("_test.fdx").unwrap();
        let fdm = directory.open("_test.fdm").unwrap();
        let mut expected_fields: Vec<stored_fields::StoredField> = (0_i32..)
            .zip(values.iter().cloned())
            .map(|(field_number, value)| stored_fields::StoredField {
                field_number,
                value,
            })
            .collect();
        expected_fields.push(stored_fields::StoredField {
            field_number: 6,
            value: FieldValue::Binary(vec![12, 13, 14]),
        });
        let expected = stored_fields::write_best_speed(
            &[
                stored_fields::Document {
                    fields: expected_fields,
                },
                stored_fields::Document::default(),
            ],
            &ID,
            "",
        );
        assert_eq!(&*fdt, expected.0.as_slice());
        assert_eq!(&*fdx, expected.1.as_slice());
        assert_eq!(&*fdm, expected.2.as_slice());
        let reader = stored_fields::open(&fdt, &fdx, &fdm, &ID, "").unwrap();
        assert_eq!(reader.max_doc(), 2);
        reader.check_integrity().unwrap();
        let document = reader.document(0).unwrap();
        assert_eq!(document.fields.len(), 7);
        for ((number, expected), actual) in (0_i32..).zip(&values).zip(&document.fields) {
            assert_eq!(actual.field_number, number);
            assert_eq!(&actual.value, expected);
        }
        assert_eq!(document.fields[6].field_number, 6);
        assert_eq!(
            document.fields[6].value,
            FieldValue::Binary(vec![12, 13, 14])
        );
        let FieldValue::Float(value) = document.fields[4].value else {
            panic!("float field changed type")
        };
        assert_eq!(value.to_bits(), (-0.0_f32).to_bits());
        assert!(reader.document(1).unwrap().fields.is_empty());
    }

    #[test]
    fn zero_documents_finish_and_repeated_close_are_valid() {
        let directory = ByteBuffersDirectory::new();
        let mut writer = AssertingStoredFieldsCodec
            .fields_writer(state(&directory))
            .unwrap();
        writer.finish(0).unwrap();
        writer.close().unwrap();
        writer.close().unwrap();
        let fdt = directory.open("_test.fdt").unwrap();
        let fdx = directory.open("_test.fdx").unwrap();
        let fdm = directory.open("_test.fdm").unwrap();
        assert_eq!(
            stored_fields::open(&fdt, &fdx, &fdm, &ID, "")
                .unwrap()
                .max_doc(),
            0
        );
    }

    #[test]
    fn installed_wrapper_rejects_invalid_document_lifecycles() {
        for scenario in 0..4 {
            let directory = ByteBuffersDirectory::new();
            let mut writer = AssertingStoredFieldsCodec
                .fields_writer(state(&directory))
                .unwrap();
            assert!(catch_unwind(AssertUnwindSafe(|| {
                match scenario {
                    0 => writer.write_field(0, &FieldValue::Int(1)).unwrap(),
                    1 => writer.finish_document().unwrap(),
                    2 => {
                        writer.start_document().unwrap();
                        writer.start_document().unwrap();
                    }
                    _ => {
                        writer.start_document().unwrap();
                        writer.finish(1).unwrap();
                    }
                }
            }))
            .is_err());
            writer.close().unwrap();
        }
    }

    #[test]
    fn invalid_factory_state_cannot_change_native_format_selection() {
        let directory = ByteBuffersDirectory::new();
        for (mode, suffix) in [(Mode::BestCompression, ""), (Mode::BestSpeed, "other")] {
            let result = AssertingStoredFieldsCodec.fields_writer(StoredFieldsWriteState {
                mode,
                suffix,
                ..state(&directory)
            });
            let error = match result {
                Ok(_) => panic!("factory accepted a different native format"),
                Err(error) => error,
            };
            assert_eq!(
                error.to_string(),
                "asserting stored-fields routing requires BestSpeed and an empty suffix"
            );
        }
        assert!(directory.list_all().unwrap().is_empty());
    }

    #[test]
    fn input_and_finish_errors_escape_the_actual_wrapped_writer() {
        let directory = ByteBuffersDirectory::new();
        let result: stored_fields::Result<()> =
            with_writer(&AssertingStoredFieldsCodec, state(&directory), |writer| {
                writer.start_document()?;
                writer.write_field_data_input(0, &mut SliceInput::new(&[1]), 2)
            });
        assert!(matches!(result, Err(stored_fields::Error::Store(_))));
        assert!(directory.open("_test.fdx").is_err());

        let directory = ByteBuffersDirectory::new();
        let result: stored_fields::Result<()> =
            with_writer(&AssertingStoredFieldsCodec, state(&directory), |writer| {
                writer.start_document()?;
                writer.finish_document()?;
                writer.finish(2)
            });
        assert_eq!(
            result.unwrap_err().to_string(),
            "Wrote 1 docs, finish called with numDocs=2"
        );
        assert!(directory.open("_test.fdx").is_err());
    }
}
