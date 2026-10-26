//! Optional same-wire norms consumers installed by the native segment codec.
//!
//! The operation owns its consumer. The index retains publication, sync and
//! cleanup of created outputs. These callbacks return errors across crate
//! boundaries; panics unwind without an automatic close or cleanup guarantee.

use std::fmt;

use lucene_store::Directory;

use crate::field_infos::FieldInfo;
use crate::norms::{self, NormsField};

/// Native identity and document bounds for one norms output pair.
#[derive(Clone, Copy)]
pub struct NormsWriteState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
    pub max_doc: i32,
}

/// Java's `NormsConsumer` field and close callbacks over borrowed columns.
///
/// A field's column is in physical document order. Missing values are omitted
/// from sparse pairs; a present zero remains a value. Implementations retain
/// anything needed beyond the callback. Panics are not caught by this seam.
pub trait NormsWriteSink {
    fn add_norms_field(&mut self, field: &FieldInfo, values: NormsField<'_>) -> norms::Result<()>;
    fn close(&mut self) -> norms::Result<()>;
}

/// Shared codec configuration opening an operation-local norms consumer.
pub trait NormsWriterFactory: Send + Sync + fmt::Debug {
    fn norms_consumer<'a>(
        &self,
        state: NormsWriteState<'a>,
    ) -> norms::Result<Box<dyn NormsWriteSink + 'a>>;
}

/// Runs a norms operation and closes its consumer after success or returned error.
///
/// Java's `IOUtils.closeWhileHandlingException` preserves an operation's error
/// when close also fails. A close error fails an otherwise successful operation.
/// Panics unwind normally; this helper does not close during an unwind.
pub fn with_writer<T, E>(
    factory: &dyn NormsWriterFactory,
    state: NormsWriteState<'_>,
    operation: impl FnOnce(&mut dyn NormsWriteSink) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<norms::Error>,
{
    let mut writer = factory.norms_consumer(state).map_err(E::from)?;
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

    use lucene_store::ByteBuffersDirectory;

    use super::*;

    fn error(message: &'static str) -> norms::Error {
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
    impl NormsWriterFactory for Factory {
        fn norms_consumer<'a>(
            &self,
            state: NormsWriteState<'a>,
        ) -> norms::Result<Box<dyn NormsWriteSink + 'a>> {
            assert_eq!(state.segment, "_test");
            assert_eq!(state.id, &[19; 16]);
            assert_eq!(state.suffix, "");
            assert_eq!(state.max_doc, 1);
            assert!(state.directory.list_all().unwrap().is_empty());
            self.events.lock().unwrap().push("open");
            if self.fail_open {
                return Err(error("open failed"));
            }
            Ok(Box::new(Sink {
                events: self.events.clone(),
                fail_close: self.fail_close,
            }))
        }
    }
    impl NormsWriteSink for Sink {
        fn add_norms_field(
            &mut self,
            field: &FieldInfo,
            values: NormsField<'_>,
        ) -> norms::Result<()> {
            assert_eq!(field.number, 3);
            assert!(matches!(values, NormsField::Dense(3, [42])));
            self.events.lock().unwrap().push("field");
            Ok(())
        }
        fn close(&mut self) -> norms::Result<()> {
            self.events.lock().unwrap().push("close");
            if self.fail_close {
                Err(error("close failed"))
            } else {
                Ok(())
            }
        }
    }
    fn state(directory: &dyn Directory) -> NormsWriteState<'_> {
        NormsWriteState {
            directory,
            segment: "_test",
            id: &[19; 16],
            suffix: "",
            max_doc: 1,
        }
    }
    #[test]
    fn success_drives_fields_then_close_and_returns_value() {
        let factory = Factory::default();
        let directory = ByteBuffersDirectory::new();
        let result: norms::Result<_> = with_writer(&factory, state(&directory), |sink| {
            sink.add_norms_field(&FieldInfo::new("f", 3), NormsField::Dense(3, &[42]))?;
            Ok(17)
        });
        assert_eq!(result.unwrap(), 17);
        assert_eq!(*factory.events.lock().unwrap(), ["open", "field", "close"]);
    }
    #[test]
    fn failed_open_does_not_run_or_close_an_absent_consumer() {
        let factory = Factory {
            fail_open: true,
            ..Factory::default()
        };
        let result: norms::Result<()> =
            with_writer(&factory, state(&ByteBuffersDirectory::new()), |_| {
                panic!("must not run")
            });
        assert_eq!(result.unwrap_err().to_string(), "open failed");
        assert_eq!(*factory.events.lock().unwrap(), ["open"]);
    }
    #[test]
    fn close_error_fails_success_but_never_replaces_operation_error() {
        for fail_close in [false, true] {
            let factory = Factory {
                fail_close,
                ..Factory::default()
            };
            let directory = ByteBuffersDirectory::new();
            let result: norms::Result<()> =
                with_writer(&factory, state(&directory), |_| Err(error("primary")));
            assert_eq!(result.unwrap_err().to_string(), "primary");
            assert_eq!(*factory.events.lock().unwrap(), ["open", "close"]);
        }
        let factory = Factory {
            fail_close: true,
            ..Factory::default()
        };
        let result: norms::Result<()> =
            with_writer(&factory, state(&ByteBuffersDirectory::new()), |_| Ok(()));
        assert_eq!(result.unwrap_err().to_string(), "close failed");
    }
}
