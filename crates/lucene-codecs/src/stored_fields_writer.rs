//! Stored-fields writer lifecycle for same-wire `Lucene104` codec wrappers.
//!
//! Java's `StoredFieldsFormat.fieldsWriter` creates an operation-local writer.
//! The shared factory here is `Send + Sync`, while its writer may borrow the
//! directory and need not cross threads. Readers and on-disk format selection
//! remain the native codec's. A factory must produce the usual `.fdt`, `.fdx`
//! and `.fdm` files for the supplied state, rather than change the codec name.

use std::fmt;

use lucene_store::data_input::DataInput;
use lucene_store::Directory;

use crate::stored_fields::{self, FieldValue, Mode};

/// Directory and codec identity of one stored-fields write operation.
#[derive(Clone, Copy)]
pub struct StoredFieldsWriteState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
    pub mode: Mode,
}

/// Java's `StoredFieldsWriter` document/finish/close lifecycle.
///
/// The driver calls `finish` after all documents, then `close` even when a
/// lifecycle method returned an error. `write_field_data_input` corresponds
/// to Java's distinct `StoredFieldDataInput` overload.
pub trait StoredFieldsWriteSink {
    fn start_document(&mut self) -> stored_fields::Result<()>;
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()>;
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()>;
    fn finish_document(&mut self) -> stored_fields::Result<()>;
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()>;
    fn close(&mut self) -> stored_fields::Result<()>;
}

/// Shared factory creating a separate writer for each flush or merge.
pub trait StoredFieldsWriterFactory: Send + Sync + fmt::Debug {
    fn fields_writer<'a>(
        &self,
        state: StoredFieldsWriteState<'a>,
    ) -> stored_fields::Result<Box<dyn StoredFieldsWriteSink + 'a>>;
}

/// Runs a write and closes its writer on either returned success or error.
///
/// Like Java's `IOUtils.closeWhileHandlingException`, cleanup must not mask
/// the operation's original failure. A close failure is returned only after
/// an otherwise successful operation. The closure owns document traversal
/// and the final `finish` call. Panics are not caught by this helper.
pub fn with_writer<T, E>(
    factory: &dyn StoredFieldsWriterFactory,
    state: StoredFieldsWriteState<'_>,
    operation: impl FnOnce(&mut dyn StoredFieldsWriteSink) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<stored_fields::Error>,
{
    let mut writer = factory.fields_writer(state).map_err(E::from)?;
    let result = operation(writer.as_mut());
    let closed = writer.close().map_err(E::from);
    match result {
        Ok(value) => closed.map(|()| value),
        Err(error) => Err(error),
    }
}

#[cfg(test)]
mod tests;
