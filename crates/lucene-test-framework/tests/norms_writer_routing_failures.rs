//! Installed norms failures preserve committed bytes and clean only this attempt.

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc, Mutex,
};

use lucene_codecs::{
    codec::Lucene104Codec,
    field_infos::FieldInfo,
    norms::{self, NormsField},
    norms_writer::{NormsWriteSink, NormsWriteState, NormsWriterFactory},
    stored_fields::FieldValue,
};
use lucene_index::{
    document::{Document, Store, StringField, TextField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::LuceneVersion,
    segment_infos::{self, SegmentInfos},
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::{
    codecs::asserting_norms_writer::AssertingNormsCodec,
    store::mock_directory_wrapper::{FailOn, FailureOp, MockDirectoryWrapper},
};
use lucene_util::java_random::JavaRandom;

// Failures are armed only after the first committed segment exists.
const BEFORE_OPEN: u8 = 1;
const BEFORE_ADD: u8 = 2;
const AFTER_ADD: u8 = 3;
const AFTER_NATIVE_CLOSE: u8 = 4;
const ADD_AND_CLOSE: u8 = 5;

#[derive(Debug, Clone)]
struct Attempt {
    segment: String,
    id: [u8; 16],
    max_doc: i32,
}

#[derive(Debug, Default)]
struct Probe {
    mode: AtomicU8,
    attempts: Mutex<Vec<Attempt>>,
    events: Mutex<Vec<String>>,
}

impl Probe {
    fn take_attempts(&self) -> Vec<Attempt> {
        std::mem::take(&mut *self.attempts.lock().unwrap())
    }

    fn take_events(&self) -> Vec<String> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }

    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }
}

fn failure(message: &'static str) -> norms::Error {
    lucene_store::Error::Io(std::io::Error::other(message)).into()
}

impl NormsWriterFactory for ArcProbe {
    fn norms_consumer<'a>(
        &self,
        state: NormsWriteState<'a>,
    ) -> norms::Result<Box<dyn NormsWriteSink + 'a>> {
        self.0.attempts.lock().unwrap().push(Attempt {
            segment: state.segment.to_owned(),
            id: *state.id,
            max_doc: state.max_doc,
        });
        self.0.record("open");
        let mode = self.0.mode.load(Ordering::SeqCst);
        if mode == BEFORE_OPEN {
            return Err(failure("norms before open"));
        }
        Ok(Box::new(FailingSink {
            delegate: AssertingNormsCodec.norms_consumer(state)?,
            directory: state.directory,
            attempt: Attempt {
                segment: state.segment.to_owned(),
                id: *state.id,
                max_doc: state.max_doc,
            },
            probe: self.0.clone(),
            mode,
        }))
    }
}

#[derive(Debug)]
struct ArcProbe(Arc<Probe>);

impl Lucene104Codec for ArcProbe {
    fn norms_writer_factory(&self) -> Option<&dyn NormsWriterFactory> {
        Some(self)
    }
}

struct FailingSink<'a> {
    delegate: Box<dyn NormsWriteSink + 'a>,
    directory: &'a dyn Directory,
    attempt: Attempt,
    probe: Arc<Probe>,
    mode: u8,
}

impl NormsWriteSink for FailingSink<'_> {
    fn add_norms_field(&mut self, field: &FieldInfo, values: NormsField<'_>) -> norms::Result<()> {
        self.probe.record(format!("add:{}", field.name));
        if self.mode == BEFORE_ADD || self.mode == ADD_AND_CLOSE {
            return Err(failure("norms primary add failure"));
        }
        self.delegate.add_norms_field(field, values)?;
        if self.mode == AFTER_ADD {
            return Err(failure("norms after native add"));
        }
        Ok(())
    }

    fn close(&mut self) -> norms::Result<()> {
        self.probe.record("close");
        self.delegate.close()?;
        if self.mode == AFTER_NATIVE_CLOSE {
            // Opening validates actual output closure; these are the produced
            // native files, not a fabricated flag saying close was reached.
            let metadata = self
                .directory
                .open(&format!("{}.nvm", self.attempt.segment))
                .unwrap();
            let (_, parsed) = norms::parse_meta(&metadata, &self.attempt.id, "").unwrap();
            let data = self
                .directory
                .open(&format!("{}.nvd", self.attempt.segment))
                .unwrap();
            assert_eq!(parsed.entries.len(), 1);
            let entry = &parsed.entries[0];
            for doc in 0..self.attempt.max_doc {
                assert!(norms::norm_value(&data, entry, doc).unwrap().is_some());
            }
            return Err(failure("norms after native close"));
        }
        if self.mode == ADD_AND_CLOSE {
            return Err(failure("norms secondary close failure"));
        }
        Ok(())
    }
}

fn writer<'a>(directory: &'a dyn Directory, probe: Arc<Probe>) -> IndexWriter<'a> {
    let mut writer = IndexWriter::open(
        directory,
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
    writer.set_codec(Some(Arc::new(ArcProbe(probe)))).unwrap();
    writer.set_use_compound_file(false);
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    writer
}

fn add(writer: &mut IndexWriter<'_>, id: &str, text: &str) {
    let mut document = Document::new();
    document.add(StringField::new("id", id, Store::Yes));
    document.add(TextField::new("body", text, Store::No));
    writer.add_fields_document(&document).unwrap();
}

fn visible(directory: &dyn Directory) -> Vec<(String, Option<i64>)> {
    let reader = DirectoryReader::open(directory).unwrap();
    let mut result = Vec::new();
    for segment in reader.segment_readers() {
        segment.check_integrity().unwrap();
        let field = segment
            .field_infos()
            .fields
            .iter()
            .find(|field| field.name == "body")
            .unwrap();
        let entry = segment.norms_entry(field.number).unwrap();
        for doc in 0..segment.max_doc {
            if segment.live_docs().is_some_and(|bits| !bits.get_doc(doc)) {
                continue;
            }
            let stored = segment.stored_document(doc).unwrap().unwrap();
            let FieldValue::String(id) = &stored.fields[0].value else {
                panic!("stored ID changed type")
            };
            result.push((
                id.clone(),
                norms::norm_value(segment.norms_data().unwrap(), entry, doc).unwrap(),
            ));
        }
    }
    result
}

fn prepare(directory: &dyn Directory, probe: Arc<Probe>) -> IndexWriter<'_> {
    let mut writer = writer(directory, probe.clone());
    add(&mut writer, "a", "x");
    writer.commit().unwrap();
    assert_eq!(visible(directory), [("a".into(), Some(1))]);
    probe.take_attempts();
    probe.take_events();
    add(&mut writer, "b", "x x");
    add(&mut writer, "c", "");
    writer
}

fn assert_unchanged(directory: &dyn Directory, before: &SegmentInfos) {
    assert_eq!(
        segment_infos::read_latest(directory).unwrap().segments,
        before.segments
    );
    assert_eq!(visible(directory), [("a".into(), Some(1))]);
}

fn assert_norm_outputs_absent(directory: &dyn Directory, attempt: &Attempt) {
    for extension in ["nvm", "nvd"] {
        assert!(
            directory
                .open(&format!("{}.{extension}", attempt.segment))
                .is_err(),
            "failed norms output remains: {}.{extension}",
            attempt.segment
        );
    }
}

fn assert_retried(writer: &mut IndexWriter<'_>, directory: &dyn Directory, probe: &Probe) {
    probe.mode.store(0, Ordering::SeqCst);
    writer.commit().unwrap();
    assert_eq!(
        visible(directory),
        [
            ("a".into(), Some(1)),
            ("b".into(), Some(2)),
            ("c".into(), Some(0))
        ]
    );
    let calls = probe.take_attempts();
    assert_eq!(calls.len(), 1);
    assert_eq!(calls[0].max_doc, 2);
    let events = probe.take_events();
    assert_eq!(events, ["open", "add:body", "close"]);
}

#[test]
fn open_add_and_close_failures_do_not_publish_and_retry_the_original_documents() {
    for (mode, message) in [
        (BEFORE_OPEN, "norms before open"),
        (BEFORE_ADD, "norms primary add failure"),
        (AFTER_ADD, "norms after native add"),
        (AFTER_NATIVE_CLOSE, "norms after native close"),
        (ADD_AND_CLOSE, "norms primary add failure"),
    ] {
        let directory = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = prepare(&directory, probe.clone());
        let before = segment_infos::read_latest(&directory).unwrap();
        probe.mode.store(mode, Ordering::SeqCst);
        let error = writer.commit().unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert!(
            !error.to_string().contains("secondary"),
            "primary add failure must win"
        );
        assert_unchanged(&directory, &before);
        let attempts = probe.take_attempts();
        assert_eq!(attempts.len(), 1);
        assert_norm_outputs_absent(&directory, &attempts[0]);
        let events = probe.take_events();
        assert_eq!(
            events
                .iter()
                .filter(|event| event.as_str() == "close")
                .count(),
            usize::from(mode != BEFORE_OPEN),
            "driver closes each acquired sink once: {events:?}"
        );
        assert_retried(&mut writer, &directory, &probe);
    }
}

#[test]
fn native_constructor_partial_outputs_and_native_close_io_errors_are_cleaned() {
    for (operation, extension) in [
        (FailureOp::CreateOutput, ".nvd"),
        (FailureOp::CreateOutput, ".nvm"),
        (FailureOp::CloseOutput, ".nvd"),
        (FailureOp::CloseOutput, ".nvm"),
    ] {
        let directory =
            MockDirectoryWrapper::new(&mut JavaRandom::new(21), ByteBuffersDirectory::new());
        let probe = Arc::new(Probe::default());
        let mut writer = prepare(&directory, probe.clone());
        let before = segment_infos::read_latest(&directory).unwrap();
        let injected = FailOn::once(move |context| {
            context.op == operation && context.name.ends_with(extension)
        });
        directory.fail_on(injected.clone());
        let error = writer.commit().unwrap_err();
        assert!(
            error.to_string().contains(&format!("{operation:?}")),
            "{error}"
        );
        assert_eq!(injected.fired(), 1);
        assert_unchanged(&directory, &before);
        let attempts = probe.take_attempts();
        assert_eq!(attempts.len(), 1);
        assert_norm_outputs_absent(&directory, &attempts[0]);
        probe.take_events();
        assert_retried(&mut writer, &directory, &probe);
    }
}

#[test]
fn primary_error_survives_cleanup_failure_and_the_commit_stays_unchanged() {
    let directory =
        MockDirectoryWrapper::new(&mut JavaRandom::new(22), ByteBuffersDirectory::new());
    let probe = Arc::new(Probe::default());
    let mut writer = prepare(&directory, probe.clone());
    let before = segment_infos::read_latest(&directory).unwrap();
    probe.mode.store(AFTER_NATIVE_CLOSE, Ordering::SeqCst);
    let cleanup = FailOn::new(|context| {
        context.op == FailureOp::DeleteFile
            && (context.name.ends_with(".nvm") || context.name.ends_with(".nvd"))
    });
    directory.fail_on(cleanup.clone());
    let error = writer.commit().unwrap_err();
    assert!(
        error.to_string().contains("norms after native close"),
        "{error}"
    );
    assert!(cleanup.fired() > 0);
    assert_unchanged(&directory, &before);
    let attempts = probe.take_attempts();
    assert_eq!(attempts.len(), 1);
    // Cleanup may fail, but neither complete output may become a committed
    // segment merely because its writer managed to close before returning Err.
    assert!(directory
        .open(&format!("{}.nvm", attempts[0].segment))
        .is_ok());
    assert!(directory
        .open(&format!("{}.nvd", attempts[0].segment))
        .is_ok());
    cleanup.clear_do_fail();
    probe.take_events();
    assert_retried(&mut writer, &directory, &probe);
}

#[test]
fn unsuccessful_merge_creation_preserves_preexisting_native_candidate_files() {
    for extension in ["nvd", "nvm"] {
        let directory = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = prepare(&directory, probe.clone());
        writer.commit().unwrap();
        probe.take_attempts();
        probe.take_events();
        let before = segment_infos::read_latest(&directory).unwrap();
        let values = visible(&directory);
        assert_eq!(before.segments.len(), 2);
        let source_files: Vec<_> = directory
            .list_all()
            .unwrap()
            .into_iter()
            .map(|name| {
                let bytes = directory.open(&name).unwrap().to_vec();
                (name, bytes)
            })
            .collect();
        let candidate = format!("_{}", lucene_util::base36::to_base36(before.counter));
        let file = format!("{candidate}.{extension}");
        let mut output = directory.create_output(&file).unwrap();
        output.write_bytes(b"preexisting belongs to another attempt");
        output.close().unwrap();
        // A failed flush additionally calls the writer's deleter.refresh(),
        // which can reclaim every unreferenced orphan. Merge failure lets us
        // observe the norms helper's own narrower output-ownership cleanup.
        let error = writer.force_merge(1).unwrap_err();
        assert!(
            error.to_string().to_lowercase().contains("exist"),
            "{error}"
        );
        assert_eq!(
            &*directory.open(&file).unwrap(),
            b"preexisting belongs to another attempt"
        );
        assert_eq!(
            segment_infos::read_latest(&directory).unwrap().segments,
            before.segments
        );
        assert_eq!(visible(&directory), values);
        for (name, bytes) in source_files {
            assert_eq!(&*directory.open(&name).unwrap(), &*bytes, "{name}");
        }
        let attempts = probe.take_attempts();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].segment, candidate);
        assert_eq!(attempts[0].max_doc, 3);
        let other = if extension == "nvd" { "nvm" } else { "nvd" };
        assert!(
            directory.open(&format!("{candidate}.{other}")).is_err(),
            "only successfully created attempt outputs may be cleaned"
        );
        directory.delete_file(&file).unwrap();
        probe.take_events();
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        assert_eq!(visible(&directory), values);
        assert_eq!(
            DirectoryReader::open(&directory)
                .unwrap()
                .segment_readers()
                .len(),
            1
        );
        let attempts = probe.take_attempts();
        assert_eq!(attempts.len(), 1);
        assert_eq!(attempts[0].max_doc, 3);
        assert_eq!(probe.take_events(), ["open", "add:body", "close"]);
    }
}
#[test]
fn failed_merge_norms_preserve_sources_and_retry_with_the_same_values() {
    let directory = ByteBuffersDirectory::new();
    let probe = Arc::new(Probe::default());
    let mut writer = prepare(&directory, probe.clone());
    writer.commit().unwrap();
    probe.take_attempts();
    probe.take_events();
    let before = segment_infos::read_latest(&directory).unwrap();
    let values = visible(&directory);
    assert_eq!(before.segments.len(), 2);
    probe.mode.store(AFTER_NATIVE_CLOSE, Ordering::SeqCst);
    let error = writer.force_merge(1).unwrap_err();
    assert!(
        error.to_string().contains("norms after native close"),
        "{error}"
    );
    assert_eq!(
        segment_infos::read_latest(&directory).unwrap().segments,
        before.segments
    );
    assert_eq!(visible(&directory), values);
    let attempts = probe.take_attempts();
    assert_eq!(attempts.len(), 1);
    assert_eq!(attempts[0].max_doc, 3);
    assert_norm_outputs_absent(&directory, &attempts[0]);
    probe.mode.store(0, Ordering::SeqCst);
    probe.take_events();
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    assert_eq!(visible(&directory), values);
    assert_eq!(
        DirectoryReader::open(&directory)
            .unwrap()
            .segment_readers()
            .len(),
        1
    );
    let events = probe.take_events();
    assert_eq!(
        events
            .iter()
            .filter(|event| event.as_str() == "close")
            .count(),
        1
    );
}
