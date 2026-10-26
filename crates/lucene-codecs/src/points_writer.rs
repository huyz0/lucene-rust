//! Optional same-wire points writers for native flush and merge operations.
//!
//! The operation owns its writer; the index layer owns output tracking, cleanup,
//! synchronization and publication. The native sink reserves the three outputs
//! in Java constructor order, buffers headers and flush fields until `finish`,
//! and encodes a merge in `merge` (which includes finish). Java streams bytes earlier;
//! this bounded sink reports encoder failures at finish and sticky output errors
//! at close. Native readers and BKD merge dispatch remain unchanged.

use std::{fmt, io};

use lucene_store::{DataOutput, Directory, FsIndexOutput};

use crate::field_infos::FieldInfo;
use crate::points::{self, MergePointsField, PackedPointsField};

/// Native identity and document bound for one points output triple.
#[derive(Clone, Copy)]
pub struct PointsWriteState<'a> {
    pub directory: &'a dyn Directory,
    pub segment: &'a str,
    pub id: &'a [u8; 16],
    pub suffix: &'a str,
    pub max_doc: i32,
}

/// Java's `PointsWriter` callbacks over owned native point buffers.
///
/// `merge` includes finish and retains source order, mapped document IDs and
/// the deleted-inclusive source budgets. `native_readers` records Java's
/// `instanceof Lucene90PointsReader` check, including point-free sources.
pub trait PointsWriteSink {
    fn write_field(&mut self, field: &FieldInfo, values: PackedPointsField) -> points::Result<()>;
    fn merge(&mut self, fields: Vec<MergePointsField>, native_readers: bool) -> points::Result<()>;
    fn finish(&mut self) -> points::Result<()>;
    fn close(&mut self) -> points::Result<()>;
}

/// Shared codec configuration opening an operation-local points writer.
pub trait PointsWriterFactory: Send + Sync + fmt::Debug {
    fn fields_writer<'a>(
        &self,
        state: PointsWriteState<'a>,
    ) -> points::Result<Box<dyn PointsWriteSink + 'a>>;
}

/// Closes after a returned success or error, preserving the operation's error.
/// A close error fails an otherwise successful operation. Panics unwind without
/// an automatic close or cleanup guarantee, as for the other writer seams.
pub fn with_writer<T, E>(
    factory: &dyn PointsWriterFactory,
    state: PointsWriteState<'_>,
    operation: impl FnOnce(&mut dyn PointsWriteSink) -> Result<T, E>,
) -> Result<T, E>
where
    E: From<points::Error>,
{
    let mut writer = factory.fields_writer(state).map_err(E::from)?;
    let result = operation(writer.as_mut());
    let closed = writer.close().map_err(E::from);
    match result {
        Ok(value) => closed.map(|()| value),
        Err(error) => Err(error),
    }
}

fn lifecycle_error(message: &str) -> points::Error {
    lucene_store::Error::Io(io::Error::new(io::ErrorKind::InvalidInput, message)).into()
}

/// Buffered production encoder retaining native field order and BKD settings.
pub struct NativePointsWriter<'a> {
    state: PointsWriteState<'a>,
    fields: Vec<PackedPointsField>,
    meta: Option<FsIndexOutput>,
    index: Option<FsIndexOutput>,
    data: Option<FsIndexOutput>,
    finished: bool,
    closed: bool,
}

impl<'a> NativePointsWriter<'a> {
    /// Java: `Lucene90PointsWriter` constructor, data then meta then index.
    /// A failed open closes every successfully created output, preserving the
    /// open error. The caller's tracked directory retains cleanup ownership.
    pub fn new(state: PointsWriteState<'a>) -> points::Result<Self> {
        let mut writer = Self {
            state,
            fields: Vec::new(),
            meta: None,
            index: None,
            data: None,
            finished: false,
            closed: false,
        };
        let stem = if state.suffix.is_empty() {
            state.segment.to_owned()
        } else {
            format!("{}_{}", state.segment, state.suffix)
        };
        let result = (|| -> lucene_store::Result<()> {
            writer.data = Some(state.directory.create_output(&format!("{stem}.kdd"))?);
            writer.meta = Some(state.directory.create_output(&format!("{stem}.kdm"))?);
            writer.index = Some(state.directory.create_output(&format!("{stem}.kdi"))?);
            Ok(())
        })();
        if let Err(error) = result {
            let _ = writer.close();
            return Err(error.into());
        }
        Ok(writer)
    }

    fn writable(&self) -> points::Result<()> {
        if self.closed {
            Err(lifecycle_error("closed points writer"))
        } else if self.finished {
            Err(lifecycle_error("already finished"))
        } else {
            Ok(())
        }
    }

    fn publish(&mut self, bytes: (Vec<u8>, Vec<u8>, Vec<u8>)) {
        // Outputs remain owned until close; publication here follows the
        // buffered encoders' tuple order, not the constructor's open order.
        self.meta
            .as_mut()
            .expect("open metadata")
            .write_bytes(&bytes.0);
        self.index
            .as_mut()
            .expect("open index")
            .write_bytes(&bytes.1);
        self.data.as_mut().expect("open data").write_bytes(&bytes.2);
    }
}

impl PointsWriteSink for NativePointsWriter<'_> {
    fn write_field(&mut self, field: &FieldInfo, values: PackedPointsField) -> points::Result<()> {
        self.writable()?;
        if field.number != values.field_number
            || field.point_dimension_count != values.num_dims
            || field.point_index_dimension_count != values.num_index_dims
            || field.point_num_bytes != values.bytes_per_dim
        {
            return Err(lifecycle_error(
                "point field identity/configuration mismatch",
            ));
        }
        self.fields.push(values);
        Ok(())
    }

    fn merge(&mut self, fields: Vec<MergePointsField>, native_readers: bool) -> points::Result<()> {
        self.writable()?;
        if !self.fields.is_empty() {
            return Err(lifecycle_error("cannot merge after writing flush fields"));
        }
        self.finished = true;
        let bytes = points::write_merged_with_mode(
            fields,
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            self.state.max_doc,
            self.state.directory,
            self.state.segment,
            self.state.id,
            self.state.suffix,
            native_readers,
        )?;
        self.publish(bytes);
        Ok(())
    }

    fn finish(&mut self) -> points::Result<()> {
        self.writable()?;
        self.finished = true;
        let bytes = points::write_packed(
            std::mem::take(&mut self.fields),
            points::DEFAULT_MAX_POINTS_IN_LEAF_NODE,
            self.state.max_doc,
            self.state.id,
            self.state.suffix,
        )?;
        self.publish(bytes);
        Ok(())
    }

    /// Java: `IOUtils.close(metaOut, indexOut, dataOut)`, including all closes
    /// after a failure. Taking each output makes repeated close harmless.
    fn close(&mut self) -> points::Result<()> {
        self.closed = true;
        self.fields.clear();
        let mut first_error = None;
        for output in [&mut self.meta, &mut self.index, &mut self.data] {
            if let Some(output) = output.take() {
                if let Err(error) = output.close() {
                    first_error.get_or_insert(error);
                }
            }
        }
        first_error.map_or(Ok(()), |error| Err(error.into()))
    }
}

#[cfg(test)]
mod tests {
    use std::sync::{Arc, Mutex};

    use lucene_store::ByteBuffersDirectory;

    use super::*;

    const ID: [u8; 16] = [51; 16];

    fn state(directory: &dyn Directory) -> PointsWriteState<'_> {
        PointsWriteState {
            directory,
            segment: "_p",
            id: &ID,
            suffix: "",
            max_doc: 2,
        }
    }
    fn field() -> FieldInfo {
        let mut field = FieldInfo::new("p", 3);
        field.point_dimension_count = 1;
        field.point_index_dimension_count = 1;
        field.point_num_bytes = 1;
        field
    }
    fn values() -> PackedPointsField {
        PackedPointsField {
            field_number: 3,
            num_dims: 1,
            num_index_dims: 1,
            bytes_per_dim: 1,
            docs: vec![1, 0, 1],
            values: vec![255, 0, 255],
        }
    }
    fn assert_bytes(directory: &dyn Directory, stem: &str, bytes: (Vec<u8>, Vec<u8>, Vec<u8>)) {
        for (extension, expected) in [("kdm", bytes.0), ("kdi", bytes.1), ("kdd", bytes.2)] {
            assert_eq!(
                &*directory.open(&format!("{stem}.{extension}")).unwrap(),
                expected
            );
        }
    }

    #[test]
    fn flush_buffers_fields_and_retains_native_bytes_suffix_and_document_order() {
        for suffix in ["", "s"] {
            let directory = ByteBuffersDirectory::new();
            let mut state = state(&directory);
            state.suffix = suffix;
            let mut writer = NativePointsWriter::new(state).unwrap();
            writer.write_field(&field(), values()).unwrap();
            assert!(!writer.finished);
            assert_eq!(writer.fields[0].docs, [1, 0, 1]);
            writer.finish().unwrap();
            assert_eq!(writer.finish().unwrap_err().to_string(), "already finished");
            assert_eq!(
                writer
                    .write_field(&field(), values())
                    .unwrap_err()
                    .to_string(),
                "already finished"
            );
            writer.close().unwrap();
            writer.close().unwrap();
            assert_eq!(
                writer.finish().unwrap_err().to_string(),
                "closed points writer"
            );
            let expected = points::write_packed(vec![values()], 512, 2, &ID, suffix).unwrap();
            assert_bytes(
                &directory,
                if suffix.is_empty() { "_p" } else { "_p_s" },
                expected,
            );
        }
    }

    #[test]
    fn field_identity_errors_do_not_buffer_and_failed_finish_is_terminal() {
        let directory = ByteBuffersDirectory::new();
        let mut writer = NativePointsWriter::new(state(&directory)).unwrap();
        for changed in 0..4 {
            let mut input = values();
            match changed {
                0 => input.field_number = 4,
                1 => input.num_dims = 2,
                2 => input.num_index_dims = 2,
                _ => input.bytes_per_dim = 2,
            }
            assert_eq!(
                writer.write_field(&field(), input).unwrap_err().to_string(),
                "point field identity/configuration mismatch"
            );
        }
        assert!(writer.fields.is_empty());
        let mut invalid = values();
        invalid.values.pop();
        writer.write_field(&field(), invalid).unwrap();
        assert!(matches!(
            writer.finish(),
            Err(points::Error::PackedValuesLength { .. })
        ));
        assert_eq!(writer.finish().unwrap_err().to_string(), "already finished");
        writer.close().unwrap();
    }

    #[test]
    fn merge_finishes_internally_retains_budgets_and_all_deleted_fields() {
        for native in [false, true] {
            let directory = ByteBuffersDirectory::new();
            let fields = vec![
                MergePointsField {
                    field_number: 3,
                    num_dims: 1,
                    num_index_dims: 1,
                    bytes_per_dim: 1,
                    sources: vec![points::PointsMergeSource {
                        docs: vec![0, 1],
                        values: vec![0, 255],
                    }],
                    max_point_count: 9,
                },
                MergePointsField {
                    field_number: 4,
                    num_dims: 1,
                    num_index_dims: 1,
                    bytes_per_dim: 1,
                    sources: vec![],
                    max_point_count: 7,
                },
            ];
            let mut writer = NativePointsWriter::new(state(&directory)).unwrap();
            writer.merge(fields.clone(), native).unwrap();
            assert_eq!(writer.finish().unwrap_err().to_string(), "already finished");
            writer.close().unwrap();
            let expected =
                points::write_merged_with_mode(fields, 512, 2, &directory, "_p", &ID, "", native)
                    .unwrap();
            assert_bytes(&directory, "_p", expected);
            let meta = directory.open("_p.kdm").unwrap();
            let index = directory.open("_p.kdi").unwrap();
            let data = directory.open("_p.kdd").unwrap();
            let reader = points::open(&meta, &index, &data, &ID, "").unwrap();
            assert_eq!(reader.decode_all_points(3).unwrap().len(), 2);
            assert!(reader.field(4).is_none());
        }
    }

    #[test]
    fn merge_cannot_discard_flush_fields_and_encoder_failure_still_closes() {
        let directory = ByteBuffersDirectory::new();
        let mut writer = NativePointsWriter::new(state(&directory)).unwrap();
        writer.write_field(&field(), values()).unwrap();
        assert_eq!(
            writer.merge(vec![], true).unwrap_err().to_string(),
            "cannot merge after writing flush fields"
        );
        writer.finish().unwrap();
        writer.close().unwrap();
        let directory = ByteBuffersDirectory::new();
        let mut writer = NativePointsWriter::new(state(&directory)).unwrap();
        assert!(writer
            .merge(vec![MergePointsField::default()], false)
            .is_err());
        writer.close().unwrap();
        assert_eq!(directory.list_all().unwrap().len(), 3);
    }

    #[test]
    fn failed_constructor_preserves_collision_and_closes_only_created_outputs() {
        for occupied in ["_p.kdd", "_p.kdm", "_p.kdi"] {
            let directory = ByteBuffersDirectory::new();
            let mut output = directory.create_output(occupied).unwrap();
            output.write_byte(19);
            output.close().unwrap();
            let error = NativePointsWriter::new(state(&directory)).err().unwrap();
            assert!(
                matches!(error, points::Error::Store(lucene_store::Error::Io(ref error)) if error.kind() == io::ErrorKind::AlreadyExists)
            );
            assert_eq!(&*directory.open(occupied).unwrap(), [19]);
            for name in directory.list_all().unwrap() {
                if name != occupied {
                    assert_eq!(directory.file_length(&name).unwrap(), 0);
                }
            }
        }
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
    impl PointsWriterFactory for Factory {
        fn fields_writer<'a>(
            &self,
            state: PointsWriteState<'a>,
        ) -> points::Result<Box<dyn PointsWriteSink + 'a>> {
            assert_eq!(state.segment, "_p");
            assert_eq!(state.id, &ID);
            assert_eq!(state.suffix, "");
            assert_eq!(state.max_doc, 2);
            assert!(state.directory.list_all().unwrap().is_empty());
            self.events.lock().unwrap().push("open");
            if self.fail_open {
                return Err(lifecycle_error("open failed"));
            }
            Ok(Box::new(Sink {
                events: self.events.clone(),
                fail_close: self.fail_close,
            }))
        }
    }
    impl PointsWriteSink for Sink {
        fn write_field(
            &mut self,
            field: &FieldInfo,
            input: PackedPointsField,
        ) -> points::Result<()> {
            assert_eq!(field.number, input.field_number);
            self.events.lock().unwrap().push("field");
            Ok(())
        }
        fn merge(&mut self, fields: Vec<MergePointsField>, native: bool) -> points::Result<()> {
            assert!(fields.is_empty());
            assert!(!native);
            self.events.lock().unwrap().push("merge");
            Ok(())
        }
        fn finish(&mut self) -> points::Result<()> {
            self.events.lock().unwrap().push("finish");
            Ok(())
        }
        fn close(&mut self) -> points::Result<()> {
            self.events.lock().unwrap().push("close");
            if self.fail_close {
                Err(lifecycle_error("close failed"))
            } else {
                Ok(())
            }
        }
    }
    #[test]
    fn operation_helper_drives_callbacks_and_preserves_primary_error() {
        let directory = ByteBuffersDirectory::new();
        let factory = Factory::default();
        let result: points::Result<_> = with_writer(&factory, state(&directory), |writer| {
            writer.write_field(&field(), values())?;
            writer.finish()?;
            Ok(17)
        });
        assert_eq!(result.unwrap(), 17);
        assert_eq!(
            *factory.events.lock().unwrap(),
            ["open", "field", "finish", "close"]
        );
        let factory = Factory::default();
        let result: points::Result<()> = with_writer(&factory, state(&directory), |writer| {
            writer.merge(vec![], false)
        });
        result.unwrap();
        assert_eq!(*factory.events.lock().unwrap(), ["open", "merge", "close"]);
        for fail_close in [false, true] {
            let factory = Factory {
                fail_close,
                ..Factory::default()
            };
            let result: points::Result<()> = with_writer(&factory, state(&directory), |_| {
                Err(lifecycle_error("primary"))
            });
            assert_eq!(result.unwrap_err().to_string(), "primary");
            assert_eq!(*factory.events.lock().unwrap(), ["open", "close"]);
        }
        let factory = Factory {
            fail_close: true,
            ..Factory::default()
        };
        let result: points::Result<()> = with_writer(&factory, state(&directory), |_| Ok(()));
        assert_eq!(result.unwrap_err().to_string(), "close failed");
        let factory = Factory {
            fail_open: true,
            ..Factory::default()
        };
        let result: points::Result<()> =
            with_writer(&factory, state(&directory), |_| panic!("must not run"));
        assert_eq!(result.unwrap_err().to_string(), "open failed");
        assert_eq!(*factory.events.lock().unwrap(), ["open"]);
    }
}
