//! Installed points failures preserve committed bytes and retry buffered inputs.

use lucene_codecs::{
    codec::Lucene104Codec,
    field_infos::FieldInfo,
    points::{self, MergePointsField, PackedPointsField},
    points_writer::{PointsWriteSink, PointsWriteState, PointsWriterFactory},
};
use lucene_index::{
    document::{Document, IntPoint, Store, StringField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::LuceneVersion,
    segment_infos,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::{
    codecs::asserting_points::AssertingPointsCodec,
    store::mock_directory_wrapper::{FailOn, FailureOp, MockDirectoryWrapper},
};
use lucene_util::java_random::JavaRandom;
use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc, Mutex,
};

#[derive(Debug, Default)]
struct Probe {
    mode: AtomicU8,
    events: Mutex<Vec<&'static str>>,
    attempts: Mutex<Vec<String>>,
}
#[derive(Debug)]
struct Codec(Arc<Probe>);
impl Lucene104Codec for Codec {
    fn points_writer_factory(&self) -> Option<&dyn PointsWriterFactory> {
        Some(self)
    }
}
fn failure(message: &'static str) -> points::Error {
    lucene_store::Error::Io(std::io::Error::other(message)).into()
}
impl PointsWriterFactory for Codec {
    fn fields_writer<'a>(
        &self,
        state: PointsWriteState<'a>,
    ) -> points::Result<Box<dyn PointsWriteSink + 'a>> {
        self.0
            .attempts
            .lock()
            .unwrap()
            .push(state.segment.to_owned());
        self.0.events.lock().unwrap().push("open");
        let mode = self.0.mode.load(Ordering::SeqCst);
        if mode == 1 {
            return Err(failure("points open"));
        }
        Ok(Box::new(Sink {
            delegate: AssertingPointsCodec.fields_writer(state)?,
            probe: self.0.clone(),
            mode,
        }))
    }
}
struct Sink<'a> {
    delegate: Box<dyn PointsWriteSink + 'a>,
    probe: Arc<Probe>,
    mode: u8,
}
impl PointsWriteSink for Sink<'_> {
    fn write_field(&mut self, field: &FieldInfo, values: PackedPointsField) -> points::Result<()> {
        self.probe.events.lock().unwrap().push("field");
        if self.mode == 2 || self.mode == 6 {
            return Err(failure("points field"));
        }
        self.delegate.write_field(field, values)
    }
    fn merge(&mut self, fields: Vec<MergePointsField>, native_readers: bool) -> points::Result<()> {
        self.probe.events.lock().unwrap().push("merge");
        if self.mode == 7 {
            return Err(failure("points merge"));
        }
        self.delegate.merge(fields, native_readers)?;
        if self.mode == 8 {
            return Err(failure("points after merge"));
        }
        Ok(())
    }
    fn finish(&mut self) -> points::Result<()> {
        self.probe.events.lock().unwrap().push("finish");
        if self.mode == 3 {
            return Err(failure("points finish"));
        }
        self.delegate.finish()?;
        if self.mode == 4 {
            return Err(failure("points after finish"));
        }
        Ok(())
    }
    fn close(&mut self) -> points::Result<()> {
        self.probe.events.lock().unwrap().push("close");
        self.delegate.close()?;
        if self.mode == 5 || self.mode == 6 {
            return Err(failure("points secondary close"));
        }
        Ok(())
    }
}
fn writer<'a>(dir: &'a dyn Directory, probe: Arc<Probe>) -> IndexWriter<'a> {
    let mut writer = IndexWriter::open(
        dir,
        Vec::new(),
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer.enable_explicit_documents().unwrap();
    writer.set_codec(Some(Arc::new(Codec(probe)))).unwrap();
    writer.set_use_compound_file(false);
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    writer
}
fn add(writer: &mut IndexWriter<'_>, id: &str, value: i32) {
    let mut document = Document::new();
    document.add(StringField::new("id", id, Store::Yes));
    document.add(IntPoint::new("p", &[value]).unwrap());
    writer.add_fields_document(&document).unwrap();
}
fn visible(dir: &dyn Directory) -> Vec<i32> {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut result = Vec::new();
    for segment in reader.segment_readers() {
        segment.check_integrity().unwrap();
        let field = segment
            .field_infos()
            .fields
            .iter()
            .find(|field| field.name == "p")
            .unwrap();
        for point in segment
            .points_reader()
            .unwrap()
            .decode_all_points(field.number)
            .unwrap()
        {
            let bytes: [u8; 4] = point.packed_value.as_slice().try_into().unwrap();
            result.push(i32::from_be_bytes(bytes) ^ i32::MIN);
        }
    }
    result.sort_unstable();
    result
}
fn committed_bytes(dir: &dyn Directory) -> Vec<(String, Vec<u8>)> {
    let infos = segment_infos::read_latest(dir).unwrap();
    let mut names: Vec<_> = infos
        .segments
        .iter()
        .flat_map(|s| {
            let si = dir.open(&format!("{}.si", s.segment_name)).unwrap();
            lucene_index::segment_info::parse(&si, &s.segment_id)
                .unwrap()
                .files
        })
        .collect();
    names.sort();
    names.dedup();
    names
        .into_iter()
        .map(|name| {
            let bytes = dir.open(&name).unwrap().to_vec();
            (name, bytes)
        })
        .collect()
}
fn assert_no_outputs(dir: &dyn Directory, probe: &Probe) {
    let attempts = probe.attempts.lock().unwrap();
    assert_eq!(attempts.len(), 1);
    for ext in ["kdm", "kdi", "kdd"] {
        assert!(dir.open(&format!("{}.{ext}", attempts[0])).is_err());
    }
}
fn clear(probe: &Probe) {
    probe.events.lock().unwrap().clear();
    probe.attempts.lock().unwrap().clear();
}

#[test]
fn returned_flush_errors_close_acquired_writer_preserve_primary_and_retry_original_points() {
    for (mode, message) in [
        (1, "points open"),
        (2, "points field"),
        (3, "points finish"),
        (4, "points after finish"),
        (5, "points secondary close"),
        (6, "points field"),
    ] {
        let dir = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone());
        add(&mut writer, "a", -7);
        writer.commit().unwrap();
        let before = committed_bytes(&dir);
        clear(&probe);
        add(&mut writer, "b", 17);
        probe.mode.store(mode, Ordering::SeqCst);
        let error = writer.commit().unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(visible(&dir), [-7]);
        assert_eq!(committed_bytes(&dir), before);
        assert_no_outputs(&dir, &probe);
        assert_eq!(
            probe
                .events
                .lock()
                .unwrap()
                .iter()
                .filter(|&&event| event == "close")
                .count(),
            usize::from(mode != 1)
        );
        clear(&probe);
        probe.mode.store(0, Ordering::SeqCst);
        writer.commit().unwrap();
        assert_eq!(visible(&dir), [-7, 17]);
        assert_eq!(
            *probe.events.lock().unwrap(),
            ["open", "field", "finish", "close"]
        );
    }
}

#[test]
fn partial_constructor_and_output_close_failures_remove_only_attempt_outputs() {
    for operation in [FailureOp::CreateOutput, FailureOp::CloseOutput] {
        for extension in [".kdd", ".kdm", ".kdi"] {
            let dir =
                MockDirectoryWrapper::new(&mut JavaRandom::new(67), ByteBuffersDirectory::new());
            let probe = Arc::new(Probe::default());
            let mut writer = writer(&dir, probe.clone());
            add(&mut writer, "a", -7);
            writer.commit().unwrap();
            let before = committed_bytes(&dir);
            clear(&probe);
            add(&mut writer, "b", 17);
            let fault = FailOn::once(move |context| {
                context.op == operation && context.name.ends_with(extension)
            });
            dir.fail_on(fault.clone());
            assert!(writer.commit().is_err());
            assert_eq!(fault.fired(), 1);
            assert_eq!(committed_bytes(&dir), before);
            assert_eq!(visible(&dir), [-7]);
            assert_no_outputs(&dir, &probe);
            writer.commit().unwrap();
            assert_eq!(visible(&dir), [-7, 17]);
        }
    }
}

#[test]
fn merge_errors_preserve_source_files_and_close_without_outer_finish() {
    for (mode, message) in [
        (7, "points merge"),
        (8, "points after merge"),
        (5, "points secondary close"),
    ] {
        let dir = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone());
        add(&mut writer, "a", -7);
        writer.commit().unwrap();
        add(&mut writer, "b", 17);
        writer.commit().unwrap();
        let before = committed_bytes(&dir);
        clear(&probe);
        probe.mode.store(mode, Ordering::SeqCst);
        let error = writer.force_merge(1).unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert_eq!(committed_bytes(&dir), before);
        assert_eq!(visible(&dir), [-7, 17]);
        assert_no_outputs(&dir, &probe);
        assert_eq!(*probe.events.lock().unwrap(), ["open", "merge", "close"]);
        clear(&probe);
        probe.mode.store(0, Ordering::SeqCst);
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        assert_eq!(visible(&dir), [-7, 17]);
        assert_eq!(*probe.events.lock().unwrap(), ["open", "merge", "close"]);
    }
}

#[test]
fn merge_constructor_collisions_leave_unowned_candidate_bytes_unchanged() {
    for extension in ["kdd", "kdm", "kdi"] {
        let dir = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone());
        add(&mut writer, "a", -7);
        writer.commit().unwrap();
        add(&mut writer, "b", 17);
        writer.commit().unwrap();
        clear(&probe);
        let before = committed_bytes(&dir);
        let infos = segment_infos::read_latest(&dir).unwrap();
        let name = format!(
            "_{}.{extension}",
            lucene_util::base36::to_base36(infos.counter)
        );
        let mut out = dir.create_output(&name).unwrap();
        out.write_bytes(b"other owner");
        out.close().unwrap();
        assert!(writer.force_merge(1).is_err());
        assert_eq!(&*dir.open(&name).unwrap(), b"other owner");
        assert_eq!(committed_bytes(&dir), before);
        assert_eq!(visible(&dir), [-7, 17]);
        let attempts = probe.attempts.lock().unwrap();
        assert_eq!(attempts.len(), 1);
        for ext in ["kdd", "kdm", "kdi"] {
            let candidate = format!("{}.{ext}", attempts[0]);
            if candidate != name {
                assert!(dir.open(&candidate).is_err());
            }
        }
    }
}
