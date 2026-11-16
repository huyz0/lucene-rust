#![cfg(test)]
use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};

const WRITE_ID: [u8; 16] = [53; 16];
fn write_state(directory: &dyn lucene_store::Directory) -> PointsWriteState<'_> {
    PointsWriteState {
        directory,
        segment: "_ap",
        id: &WRITE_ID,
        suffix: "",
        max_doc: 2,
    }
}
fn write_field_info() -> FieldInfo {
    let mut field = FieldInfo::new("p", 4);
    field.point_dimension_count = 1;
    field.point_index_dimension_count = 1;
    field.point_num_bytes = 1;
    field
}
fn write_values() -> lucene_codecs::points::PackedPointsField {
    lucene_codecs::points::PackedPointsField {
        field_number: 4,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 1,
        docs: vec![1, 0, 1],
        values: vec![255, 0, 255],
    }
}

#[test]
fn installed_points_codec_retains_native_identity_and_flush_bytes() {
    use lucene_store::{ByteBuffersDirectory, Directory};
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<AssertingPointsCodec>();
    let codec: &dyn Lucene104Codec = &AssertingPointsCodec;
    assert_eq!(codec.name(), "Lucene104");
    assert!(codec.norms_writer_factory().is_none());
    assert!(codec.live_docs_writer().is_none());
    assert!(codec.stored_fields_writer_factory().is_none());
    let directory = ByteBuffersDirectory::new();
    let result: lucene_codecs::points::Result<_> = lucene_codecs::points_writer::with_writer(
        codec.points_writer_factory().unwrap(),
        write_state(&directory),
        |writer| {
            writer.write_field(&write_field_info(), write_values())?;
            writer.finish()
        },
    );
    result.unwrap();
    let expected =
        lucene_codecs::points::write_packed(vec![write_values()], 512, 2, &WRITE_ID, "").unwrap();
    for (extension, bytes) in [
        ("kdm", expected.0),
        ("kdi", expected.1),
        ("kdd", expected.2),
    ] {
        assert_eq!(
            &*directory.open(&format!("_ap.{extension}")).unwrap(),
            bytes
        );
    }
    let meta = directory.open("_ap.kdm").unwrap();
    let index = directory.open("_ap.kdi").unwrap();
    let data = directory.open("_ap.kdd").unwrap();
    let reader = lucene_codecs::points::open(&meta, &index, &data, &WRITE_ID, "").unwrap();
    let decoded = reader.decode_all_points(4).unwrap();
    assert_eq!(
        decoded.iter().map(|point| point.doc_id).collect::<Vec<_>>(),
        [0, 1, 1]
    );
}

#[test]
fn installed_factory_rejects_suffix_before_creating_outputs_and_supports_empty_finish() {
    use lucene_store::{ByteBuffersDirectory, Directory};
    let directory = ByteBuffersDirectory::new();
    let mut state = write_state(&directory);
    state.suffix = "s";
    let error = AssertingPointsCodec.fields_writer(state).err().unwrap();
    assert_eq!(
        error.to_string(),
        "asserting points routing requires an empty suffix"
    );
    assert!(directory.list_all().unwrap().is_empty());
    let mut writer = AssertingPointsCodec
        .fields_writer(write_state(&directory))
        .unwrap();
    writer.finish().unwrap();
    writer.close().unwrap();
    writer.close().unwrap();
    assert_eq!(
        writer.finish().unwrap_err().to_string(),
        "closed points writer"
    );
    let meta = directory.open("_ap.kdm").unwrap();
    let index = directory.open("_ap.kdi").unwrap();
    let data = directory.open("_ap.kdd").unwrap();
    assert!(
        lucene_codecs::points::open(&meta, &index, &data, &WRITE_ID, "")
            .unwrap()
            .field_numbers()
            .next()
            .is_none()
    );
}

struct LowerProbe {
    calls: Vec<&'static str>,
    failure: &'static str,
    close_count: usize,
    fail_close_at: usize,
}
impl LowerProbe {
    fn call(&mut self, name: &'static str) -> lucene_codecs::points::Result<()> {
        self.calls.push(name);
        if self.failure == name {
            Err(lucene_store::Error::Io(std::io::Error::other(name)).into())
        } else {
            Ok(())
        }
    }
}
impl PointsWriteSink for LowerProbe {
    fn write_field(
        &mut self,
        field: &FieldInfo,
        values: lucene_codecs::points::PackedPointsField,
    ) -> lucene_codecs::points::Result<()> {
        assert_eq!(field.number, values.field_number);
        assert_eq!(values.docs, [1, 0, 1]);
        self.call("write")
    }
    fn merge(
        &mut self,
        fields: Vec<lucene_codecs::points::MergePointsField>,
        native: bool,
    ) -> lucene_codecs::points::Result<()> {
        assert!(!native);
        assert_eq!(fields[0].max_point_count, 17);
        assert_eq!(fields[0].sources[0].docs, [1]);
        self.call("merge")
    }
    fn finish(&mut self) -> lucene_codecs::points::Result<()> {
        self.call("finish")
    }
    fn close(&mut self) -> lucene_codecs::points::Result<()> {
        self.close_count += 1;
        self.calls.push("close");
        if self.close_count == self.fail_close_at {
            Err(lucene_store::Error::Io(std::io::Error::other("close")).into())
        } else {
            Ok(())
        }
    }
}
fn lower_probe(failure: &'static str, fail_close_at: usize) -> AssertingPointsWriter<LowerProbe> {
    AssertingPointsWriter::new(LowerProbe {
        calls: vec![],
        failure,
        close_count: 0,
        fail_close_at,
    })
}
fn lower_merge_fields() -> Vec<lucene_codecs::points::MergePointsField> {
    vec![lucene_codecs::points::MergePointsField {
        max_point_count: 17,
        sources: vec![lucene_codecs::points::PointsMergeSource {
            docs: vec![1],
            values: vec![255],
        }],
        ..lucene_codecs::points::MergePointsField::default()
    }]
}

#[test]
fn installed_asserting_sink_checks_dimensions_and_preserves_callback_errors_and_merge_identity() {
    let mut writer = lower_probe("", 0);
    let mut field = write_field_info();
    field.point_dimension_count = 0;
    let error = PointsWriteSink::write_field(&mut writer, &field, write_values()).unwrap_err();
    assert_eq!(
        error.to_string(),
        "writing field=\"p\" but pointDimensionalCount is 0"
    );
    assert!(writer.delegate().calls.is_empty());
    for failure in ["", "write", "merge", "finish"] {
        let mut writer = lower_probe(failure, 0);
        let write = PointsWriteSink::write_field(&mut writer, &write_field_info(), write_values());
        let merge = PointsWriteSink::merge(&mut writer, lower_merge_fields(), false);
        let finish = PointsWriteSink::finish(&mut writer);
        for (name, result) in [("write", write), ("merge", merge), ("finish", finish)] {
            if name == failure {
                assert_eq!(result.unwrap_err().to_string(), name);
            } else {
                result.unwrap();
            }
        }
        assert_eq!(writer.delegate().calls, ["write", "merge", "finish"]);
    }
}

#[test]
fn installed_asserting_double_close_stops_after_first_failure() {
    for fail_close_at in [0, 1, 2] {
        let mut writer = lower_probe("", fail_close_at);
        let result = PointsWriteSink::close(&mut writer);
        if fail_close_at == 0 {
            result.unwrap();
        } else {
            assert_eq!(result.unwrap_err().to_string(), "close");
        }
        assert_eq!(
            writer.delegate().close_count,
            if fail_close_at == 1 { 1 } else { 2 }
        );
    }
}

#[derive(Default)]
struct OutputDirectory {
    inner: lucene_store::ByteBuffersDirectory,
    events: Arc<std::sync::Mutex<Vec<String>>>,
    fail_open: &'static str,
    fail_close: bool,
    fail_write: bool,
}
struct OutputFailures {
    events: Arc<std::sync::Mutex<Vec<String>>>,
    fail_close: bool,
    fail_write: bool,
}
impl lucene_store::index_output::OutputFaults for OutputFailures {
    fn before_write(&mut self, name: &str, _: usize) -> lucene_store::index_output::WriteVerdict {
        if self.fail_write {
            lucene_store::index_output::WriteVerdict::Fail {
                keep: 1,
                error: std::io::Error::other(format!("write {name}")),
            }
        } else {
            lucene_store::index_output::WriteVerdict::Write
        }
    }
    fn on_close(&mut self, name: &str) -> lucene_store::index_output::CloseVerdict {
        self.events.lock().unwrap().push(format!("close:{name}"));
        if self.fail_close {
            lucene_store::index_output::CloseVerdict::FailAfterFlush(std::io::Error::other(
                format!("close {name}"),
            ))
        } else {
            lucene_store::index_output::CloseVerdict::Close
        }
    }
}
impl lucene_store::Directory for OutputDirectory {
    fn list_all(&self) -> lucene_store::Result<Vec<String>> {
        self.inner.list_all()
    }
    fn open(&self, name: &str) -> lucene_store::Result<lucene_store::Input> {
        self.inner.open(name)
    }
    fn create_output(&self, name: &str) -> lucene_store::Result<lucene_store::FsIndexOutput> {
        self.events.lock().unwrap().push(format!("open:{name}"));
        if name == self.fail_open {
            return Err(std::io::Error::other("open failure").into());
        }
        let mut output = self.inner.create_output(name)?;
        output.set_faults(Box::new(OutputFailures {
            events: self.events.clone(),
            fail_close: self.fail_close,
            fail_write: self.fail_write,
        }));
        Ok(output)
    }
    fn sync(&self, names: &[String]) -> lucene_store::Result<()> {
        self.inner.sync(names)
    }
    fn rename(&self, from: &str, to: &str) -> lucene_store::Result<()> {
        self.inner.rename(from, to)
    }
    fn delete_file(&self, name: &str) -> lucene_store::Result<()> {
        self.inner.delete_file(name)
    }
    fn sync_meta_data(&self) -> lucene_store::Result<()> {
        self.inner.sync_meta_data()
    }
    fn obtain_lock(&self, name: &str) -> lucene_store::Result<Box<dyn lucene_store::Lock>> {
        self.inner.obtain_lock(name)
    }
}

#[test]
fn native_sink_opens_and_closes_in_java_order_preserving_first_io_error() {
    for fail_write in [false, true] {
        for fail_close in [false, true] {
            let directory = OutputDirectory {
                fail_write,
                fail_close,
                ..OutputDirectory::default()
            };
            let mut writer = AssertingPointsCodec
                .fields_writer(write_state(&directory))
                .unwrap();
            writer
                .write_field(&write_field_info(), write_values())
                .unwrap();
            writer.finish().unwrap();
            let result = writer.close();
            if fail_write {
                assert_eq!(result.unwrap_err().to_string(), "write _ap.kdm");
            } else if fail_close {
                assert_eq!(result.unwrap_err().to_string(), "close _ap.kdm");
            } else {
                result.unwrap();
            }
            writer.close().unwrap();
            assert_eq!(
                *directory.events.lock().unwrap(),
                [
                    "open:_ap.kdd",
                    "open:_ap.kdm",
                    "open:_ap.kdi",
                    "close:_ap.kdm",
                    "close:_ap.kdi",
                    "close:_ap.kdd"
                ]
            );
        }
    }
}

#[test]
fn failed_factory_closes_every_opened_output_without_replacing_open_error() {
    for (fail_open, expected) in [
        ("_ap.kdd", vec!["open:_ap.kdd"]),
        (
            "_ap.kdm",
            vec!["open:_ap.kdd", "open:_ap.kdm", "close:_ap.kdd"],
        ),
        (
            "_ap.kdi",
            vec![
                "open:_ap.kdd",
                "open:_ap.kdm",
                "open:_ap.kdi",
                "close:_ap.kdm",
                "close:_ap.kdd",
            ],
        ),
    ] {
        let directory = OutputDirectory {
            fail_open,
            fail_close: true,
            ..OutputDirectory::default()
        };
        let error = AssertingPointsCodec
            .fields_writer(write_state(&directory))
            .err()
            .unwrap();
        assert_eq!(error.to_string(), "open failure");
        assert_eq!(*directory.events.lock().unwrap(), expected);
    }
}
#[derive(Default)]
struct Recorder {
    values: Vec<(i32, Vec<u8>)>,
    fail_grow: bool,
}
impl IntersectVisitor for Recorder {
    fn visit(&mut self, doc: i32) -> Result<()> {
        self.values.push((doc, Vec::new()));
        Ok(())
    }
    fn visit_value(&mut self, doc: i32, value: &[u8]) -> Result<()> {
        self.values.push((doc, value.to_vec()));
        Ok(())
    }
    fn grow(&mut self, _: usize) {
        assert!(!self.fail_grow, "grow failure");
    }
    fn compare(&mut self, _: &[u8], _: &[u8]) -> Relation {
        Relation::CellCrossesQuery
    }
}
#[test]
fn scalar_document_visit_consumes_budget_before_failing() {
    let mut recorder = Recorder::default();
    let mut visitor = AssertingIntersectVisitor::new(1, 1, 1, &mut recorder);
    visitor.grow(1);
    visitor.visit(7).unwrap();
    assert!(catch_unwind(AssertUnwindSafe(|| visitor.visit(8))).is_err());
    assert_eq!(visitor.budget, -1);
    assert_eq!(recorder.values, [(7, Vec::new())]);
}
#[test]
fn multidimensional_bounds_only_cover_index_dimensions_and_do_not_order_points() {
    let mut recorder = Recorder::default();
    let mut visitor = AssertingIntersectVisitor::new(2, 1, 1, &mut recorder);
    visitor.grow(2);
    assert_eq!(visitor.compare(&[10], &[20]), Relation::CellCrossesQuery);
    visitor.visit_value(8, &[15, 255]).unwrap();
    visitor.visit_value(0, &[0, 0]).unwrap();
    assert_eq!(recorder.values, [(8, vec![15, 255]), (0, vec![0, 0])]);
}
#[test]
fn failed_delegate_grow_preserves_previous_budget_and_failed_compare_preserves_previous_relation() {
    let mut recorder = Recorder {
        fail_grow: true,
        ..Recorder::default()
    };
    let mut visitor = AssertingIntersectVisitor::new(1, 1, 1, &mut recorder);
    visitor.budget = 2;
    assert!(catch_unwind(AssertUnwindSafe(|| visitor.grow(1))).is_err());
    assert_eq!(visitor.budget, 2);
    assert_eq!(visitor.compare(&[0], &[255]), Relation::CellCrossesQuery);
    assert!(catch_unwind(AssertUnwindSafe(|| visitor.compare(&[2], &[1]))).is_err());
    assert_eq!(visitor.last_compare, Some(Relation::CellCrossesQuery));
    visitor.visit_value(0, &[5]).unwrap();
    visitor.visit_value(1, &[6]).unwrap();
    assert_eq!(recorder.values, [(0, vec![5]), (1, vec![6])]);
}
