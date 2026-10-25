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
mod tests {
    use std::sync::{Arc, Mutex};

    use lucene_store::data_input::SliceInput;
    use lucene_store::ByteBuffersDirectory;

    use super::*;

    fn failure(message: &'static str) -> stored_fields::Error {
        lucene_store::Error::Io(std::io::Error::other(message)).into()
    }

    #[derive(Debug, Default)]
    struct Factory {
        events: Arc<Mutex<Vec<&'static str>>>,
        fail_open: bool,
        fail_close: bool,
    }
    struct Sink {
        events: Arc<Mutex<Vec<&'static str>>>,
        fail_close: bool,
    }
    impl StoredFieldsWriterFactory for Factory {
        fn fields_writer<'a>(
            &self,
            state: StoredFieldsWriteState<'a>,
        ) -> stored_fields::Result<Box<dyn StoredFieldsWriteSink + 'a>> {
            assert_eq!(state.segment, "_test");
            assert_eq!(state.id, &[7; 16]);
            assert_eq!(state.suffix, "");
            assert_eq!(state.mode, Mode::BestSpeed);
            self.events.lock().unwrap().push("open");
            if self.fail_open {
                return Err(failure("open failed"));
            }
            Ok(Box::new(Sink {
                events: self.events.clone(),
                fail_close: self.fail_close,
            }))
        }
    }
    impl StoredFieldsWriteSink for Sink {
        fn start_document(&mut self) -> stored_fields::Result<()> {
            self.events.lock().unwrap().push("start");
            Ok(())
        }
        fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
            assert_eq!(number, 3);
            assert_eq!(value, &FieldValue::Int(42));
            self.events.lock().unwrap().push("field");
            Ok(())
        }
        fn write_field_data_input(
            &mut self,
            number: i32,
            input: &mut dyn DataInput,
            length: usize,
        ) -> stored_fields::Result<()> {
            assert_eq!(number, 4);
            let mut bytes = vec![0; length];
            input.read_bytes(&mut bytes)?;
            assert_eq!(bytes, [1, 2, 3]);
            self.events.lock().unwrap().push("input");
            Ok(())
        }
        fn finish_document(&mut self) -> stored_fields::Result<()> {
            self.events.lock().unwrap().push("document");
            Ok(())
        }
        fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
            assert_eq!(num_docs, 1);
            self.events.lock().unwrap().push("finish");
            Ok(())
        }
        fn close(&mut self) -> stored_fields::Result<()> {
            self.events.lock().unwrap().push("close");
            if self.fail_close {
                Err(failure("close failed"))
            } else {
                Ok(())
            }
        }
    }
    fn state(directory: &dyn Directory) -> StoredFieldsWriteState<'_> {
        StoredFieldsWriteState {
            directory,
            segment: "_test",
            id: &[7; 16],
            suffix: "",
            mode: Mode::BestSpeed,
        }
    }

    #[test]
    fn successful_operation_drives_the_dyn_lifecycle_and_returns_its_value() {
        let factory = Factory::default();
        let directory = ByteBuffersDirectory::new();
        let result: stored_fields::Result<_> = with_writer(&factory, state(&directory), |writer| {
            writer.start_document()?;
            writer.write_field(3, &FieldValue::Int(42))?;
            writer.write_field_data_input(4, &mut SliceInput::new(&[1, 2, 3]), 3)?;
            writer.finish_document()?;
            writer.finish(1)?;
            Ok(19)
        });
        assert_eq!(result.unwrap(), 19);
        assert_eq!(
            *factory.events.lock().unwrap(),
            ["open", "start", "field", "input", "document", "finish", "close"]
        );
    }

    #[test]
    fn open_failure_never_runs_operation_or_closes_an_absent_writer() {
        let factory = Factory {
            fail_open: true,
            ..Factory::default()
        };
        let directory = ByteBuffersDirectory::new();
        let result: stored_fields::Result<()> = with_writer(&factory, state(&directory), |_| {
            panic!("operation must not run after failed open")
        });
        assert_eq!(result.unwrap_err().to_string(), "open failed");
        assert_eq!(*factory.events.lock().unwrap(), ["open"]);
    }

    #[test]
    fn close_failure_fails_an_otherwise_successful_operation() {
        let factory = Factory {
            fail_close: true,
            ..Factory::default()
        };
        let directory = ByteBuffersDirectory::new();
        let result: stored_fields::Result<()> =
            with_writer(&factory, state(&directory), |_| Ok(()));
        assert_eq!(result.unwrap_err().to_string(), "close failed");
        assert_eq!(*factory.events.lock().unwrap(), ["open", "close"]);
    }

    #[derive(Debug)]
    enum DriverError {
        Codec(stored_fields::Error),
        Traversal,
    }
    impl From<stored_fields::Error> for DriverError {
        fn from(error: stored_fields::Error) -> Self {
            Self::Codec(error)
        }
    }

    #[test]
    fn original_driver_error_wins_with_or_without_a_close_error() {
        let directory = ByteBuffersDirectory::new();
        for fail_close in [false, true] {
            let factory = Factory {
                fail_close,
                ..Factory::default()
            };
            let result: Result<(), DriverError> =
                with_writer(&factory, state(&directory), |_| Err(DriverError::Traversal));
            assert!(matches!(result, Err(DriverError::Traversal)));
            assert_eq!(*factory.events.lock().unwrap(), ["open", "close"]);
        }
        let factory = Factory {
            fail_open: true,
            ..Factory::default()
        };
        let result: Result<(), DriverError> = with_writer(&factory, state(&directory), |_| Ok(()));
        let Err(DriverError::Codec(error)) = result else {
            panic!("missing converted open error")
        };
        assert_eq!(error.to_string(), "open failed");
        let factory = Factory {
            fail_close: true,
            ..Factory::default()
        };
        let result: Result<(), DriverError> = with_writer(&factory, state(&directory), |_| Ok(()));
        let Err(DriverError::Codec(error)) = result else {
            panic!("missing converted close error")
        };
        assert_eq!(error.to_string(), "close failed");
    }
}
