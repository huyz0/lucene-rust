#![cfg(test)]
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
                field_name: String::new(),
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
                field_name: String::new(),
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

/// Splits on dimension 1 (of two) and counts its calls.
#[derive(Debug, Default)]
struct Policy {
    calls: Arc<Mutex<Vec<(String, usize)>>>,
}
impl AddFinishPolicy for Policy {
    fn max_points_in_leaf_node(&self) -> i32 {
        4
    }
    fn max_mb_sort_in_heap(&self) -> f64 {
        1.0
    }
    fn split(&self, field_name: &str) -> Option<crate::bkd_writer::SplitOverride<'static>> {
        let calls = self.calls.clone();
        let name = field_name.to_string();
        Some(Box::new(move |_, _, parent_splits: &[i32]| {
            calls
                .lock()
                .unwrap()
                .push((name.clone(), parent_splits.len()));
            1
        }))
    }
}

// ARITH: fixture values for at most 32 documents.
#[allow(clippy::arithmetic_side_effects)]
fn two_dims(field_name: &str, docs: usize) -> (FieldInfo, PackedPointsField) {
    let mut info = FieldInfo::new(field_name, 3);
    info.point_dimension_count = 2;
    info.point_index_dimension_count = 2;
    info.point_num_bytes = 1;
    let values = PackedPointsField {
        field_number: 3,
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 1,
        docs: (0..docs as i32).collect(),
        values: (0..docs)
            .flat_map(|d| [(d * 7 % 32) as u8, d as u8])
            .collect(),
    };
    (info, values)
}

// ARITH: fixture slices of 32 two-byte points.
#[allow(clippy::arithmetic_side_effects)]
#[test]
fn an_add_finish_policy_adds_flush_and_merge_points_with_its_split() {
    let policy = Arc::new(Policy::default());
    let directory = ByteBuffersDirectory::new();
    let mut state = state(&directory);
    state.max_doc = 32;
    let (info, values) = two_dims("q", 32);
    let mut writer = NativePointsWriter::with_add_finish(state, policy.clone()).unwrap();
    writer.write_field(&info, values.clone()).unwrap();
    writer.finish().unwrap();
    writer.close().unwrap();
    // Eight leaves of four: seven inner nodes, each split by the policy.
    assert_eq!(*policy.calls.lock().unwrap(), vec![("q".to_string(), 2); 7]);
    let read = |directory: &ByteBuffersDirectory| {
        let meta = directory.open("_p.kdm").unwrap();
        let index = directory.open("_p.kdi").unwrap();
        let data = directory.open("_p.kdd").unwrap();
        let reader = points::open(&meta, &index, &data, &ID, "").unwrap();
        let mut points: Vec<_> = reader
            .decode_all_points(3)
            .unwrap()
            .into_iter()
            .map(|p| (p.doc_id, p.packed_value))
            .collect();
        points.sort();
        (reader.decode_leaves(3).unwrap().len(), points)
    };
    let flushed = read(&directory);
    assert_eq!(flushed.0, 8);
    let expected: Vec<_> = (0..32)
        .map(|d| {
            (
                d,
                values.values[d as usize * 2..d as usize * 2 + 2].to_vec(),
            )
        })
        .collect();
    assert_eq!(flushed.1, expected);
    // Every split on dimension 1 differs from the widest-dimension tree.
    assert_ne!(
        &*directory.open("_p.kdi").unwrap(),
        points::write_packed(vec![values.clone()], 4, 32, &ID, "")
            .unwrap()
            .1
    );

    // A merge goes through the same writer, sources in order.
    policy.calls.lock().unwrap().clear();
    let directory = ByteBuffersDirectory::new();
    let mut state = super::tests::state(&directory);
    state.max_doc = 32;
    let half = |range: std::ops::Range<usize>| points::PointsMergeSource {
        docs: values.docs[range.clone()].to_vec(),
        values: values.values[range.start * 2..range.end * 2].to_vec(),
    };
    let merged = MergePointsField {
        field_number: 3,
        field_name: "q".to_string(),
        num_dims: 2,
        num_index_dims: 2,
        bytes_per_dim: 1,
        sources: vec![half(0..16), half(16..32)],
        max_point_count: 32,
    };
    let mut writer = NativePointsWriter::with_add_finish(state, policy.clone()).unwrap();
    writer.merge(vec![merged], true).unwrap();
    writer.close().unwrap();
    assert_eq!(policy.calls.lock().unwrap().len(), 7);
    assert_eq!(read(&directory), flushed);
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
    fn write_field(&mut self, field: &FieldInfo, input: PackedPointsField) -> points::Result<()> {
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
