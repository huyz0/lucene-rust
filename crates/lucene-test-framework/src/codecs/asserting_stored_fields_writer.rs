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
mod tests;
