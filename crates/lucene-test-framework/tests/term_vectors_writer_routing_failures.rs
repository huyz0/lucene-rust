//! Installed binary term-vector failures preserve commits and retry inputs.

use std::sync::{
    atomic::{AtomicU8, Ordering},
    Arc, Mutex,
};

use lucene_codecs::{
    codec::Lucene104Codec,
    field_infos::{FieldInfo, IndexOptions},
    stored_fields::FieldValue,
    term_vectors,
    term_vectors_writer::{TermVectorsWriteSink, TermVectorsWriteState, TermVectorsWriterFactory},
};
use lucene_index::{
    document::{Document, Field, FieldType, Store, StringField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::LuceneVersion,
    segment_infos::{self, SegmentInfos},
};
use lucene_search::{directory_reader::DirectoryReader, reader::IndexReader};
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::{
    codecs::asserting_term_vectors_writer::AssertingTermVectorsCodec,
    store::mock_directory_wrapper::{FailOn, FailureOp, MockDirectoryWrapper},
};
use lucene_util::java_random::JavaRandom;

const BEFORE_OPEN: u8 = 1;
const BEFORE_POSITION: u8 = 2;
const AFTER_FINISH: u8 = 3;
const AFTER_CLOSE: u8 = 4;
const POSITION_AND_CLOSE: u8 = 5;
const EXTENSIONS: [&str; 3] = ["tvm", "tvd", "tvx"];

#[derive(Debug, Default)]
struct Probe {
    mode: AtomicU8,
    attempts: Mutex<Vec<String>>,
    events: Mutex<Vec<String>>,
}

impl Probe {
    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }
    fn clear(&self) {
        self.attempts.lock().unwrap().clear();
        self.events.lock().unwrap().clear();
    }
}

#[derive(Debug)]
struct Codec(Arc<Probe>);

impl Lucene104Codec for Codec {
    fn term_vectors_writer_factory(&self) -> Option<&dyn TermVectorsWriterFactory> {
        Some(self)
    }
}

fn failure(message: &'static str) -> term_vectors::Error {
    lucene_store::Error::Io(std::io::Error::other(message)).into()
}

impl TermVectorsWriterFactory for Codec {
    fn vectors_writer<'a>(
        &self,
        state: TermVectorsWriteState<'a>,
    ) -> term_vectors::Result<Box<dyn TermVectorsWriteSink + 'a>> {
        self.0
            .attempts
            .lock()
            .unwrap()
            .push(state.segment.to_owned());
        self.0.record("open");
        let mode = self.0.mode.load(Ordering::SeqCst);
        if mode == BEFORE_OPEN {
            return Err(failure("term vectors before open"));
        }
        Ok(Box::new(FailingSink {
            delegate: AssertingTermVectorsCodec.vectors_writer(state)?,
            directory: state.directory,
            segment: state.segment.to_owned(),
            id: *state.id,
            max_doc: state.max_doc,
            probe: self.0.clone(),
            mode,
        }))
    }
}

struct FailingSink<'a> {
    delegate: Box<dyn TermVectorsWriteSink + 'a>,
    directory: &'a dyn Directory,
    segment: String,
    id: [u8; 16],
    max_doc: i32,
    probe: Arc<Probe>,
    mode: u8,
}

impl TermVectorsWriteSink for FailingSink<'_> {
    fn start_document(&mut self, fields: i32) -> term_vectors::Result<()> {
        self.probe.record(format!("start_document:{fields}"));
        self.delegate.start_document(fields)
    }
    fn finish_document(&mut self) -> term_vectors::Result<()> {
        self.probe.record("finish_document");
        self.delegate.finish_document()
    }
    fn start_field(
        &mut self,
        field: &FieldInfo,
        terms: i32,
        p: bool,
        o: bool,
        y: bool,
    ) -> term_vectors::Result<()> {
        self.probe.record(format!("start_field:{}", field.name));
        self.delegate.start_field(field, terms, p, o, y)
    }
    fn finish_field(&mut self) -> term_vectors::Result<()> {
        self.probe.record("finish_field");
        self.delegate.finish_field()
    }
    fn start_term(&mut self, term: &[u8], frequency: i32) -> term_vectors::Result<()> {
        self.probe.record("start_term");
        self.delegate.start_term(term, frequency)
    }
    fn finish_term(&mut self) -> term_vectors::Result<()> {
        self.probe.record("finish_term");
        self.delegate.finish_term()
    }
    fn add_position(
        &mut self,
        p: i32,
        s: i32,
        e: i32,
        y: Option<&[u8]>,
    ) -> term_vectors::Result<()> {
        self.probe.record("add_position");
        if self.mode == BEFORE_POSITION || self.mode == POSITION_AND_CLOSE {
            return Err(failure("term vectors primary position failure"));
        }
        self.delegate.add_position(p, s, e, y)
    }
    fn finish(&mut self, documents: i32) -> term_vectors::Result<()> {
        self.probe.record(format!("finish:{documents}"));
        self.delegate.finish(documents)?;
        if self.mode == AFTER_FINISH {
            return Err(failure("term vectors after native finish"));
        }
        Ok(())
    }
    fn close(&mut self) -> term_vectors::Result<()> {
        self.probe.record("close");
        self.delegate.close()?;
        if self.mode == AFTER_CLOSE {
            // Verify that these are complete native binary outputs before
            // returning the failure that must prevent their publication.
            let data = self
                .directory
                .open(&format!("{}.tvd", self.segment))
                .unwrap();
            let index = self
                .directory
                .open(&format!("{}.tvx", self.segment))
                .unwrap();
            let meta = self
                .directory
                .open(&format!("{}.tvm", self.segment))
                .unwrap();
            let reader = term_vectors::open(&data, &index, &meta, &self.id, "").unwrap();
            reader.check_integrity().unwrap();
            for doc in 0..self.max_doc {
                reader.document(doc).unwrap();
            }
            return Err(failure("term vectors after native close"));
        }
        if self.mode == POSITION_AND_CLOSE {
            return Err(failure("term vectors secondary close failure"));
        }
        Ok(())
    }
    fn ram_bytes_used(&self) -> usize {
        self.delegate.ram_bytes_used()
    }
}

fn writer(directory: &dyn Directory, probe: Arc<Probe>) -> IndexWriter<'_> {
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
    writer.set_codec(Some(Arc::new(Codec(probe)))).unwrap();
    writer.set_use_compound_file(false);
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    writer
}

fn add(writer: &mut IndexWriter<'_>, id: &str, text: Option<&str>) {
    let mut document = Document::new();
    document.add(StringField::new("id", id, Store::Yes));
    if let Some(text) = text {
        let mut ty = FieldType::new();
        ty.set_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
            .unwrap();
        ty.set_tokenized(true).unwrap();
        ty.set_store_term_vectors(true).unwrap();
        ty.set_store_term_vector_positions(true).unwrap();
        ty.set_store_term_vector_offsets(true).unwrap();
        document.add(Field::from_string("body", text, ty).unwrap());
    }
    writer.add_fields_document(&document).unwrap();
}

type VisibleDocument = (String, Vec<(Vec<u8>, i32)>);

fn visible(directory: &dyn Directory) -> Vec<VisibleDocument> {
    let reader = DirectoryReader::open(directory).unwrap();
    let mut result = Vec::new();
    for segment in reader.segment_readers() {
        segment.check_integrity().unwrap();
        let vectors = segment.term_vectors_reader().unwrap();
        let id_field = segment
            .field_infos()
            .fields
            .iter()
            .find(|f| f.name == "id")
            .unwrap()
            .number;
        for doc in 0..segment.max_doc() {
            let stored = segment.stored_document(doc).unwrap().unwrap();
            let FieldValue::String(id) = &stored
                .fields
                .iter()
                .find(|f| f.field_number == id_field)
                .unwrap()
                .value
            else {
                panic!("stored ID changed type")
            };
            let terms = vectors
                .as_ref()
                .and_then(|r| r.document(doc).unwrap())
                .map(|d| {
                    d.fields
                        .into_iter()
                        .flat_map(|f| f.terms.into_iter().map(|t| (t.term, t.freq)))
                        .collect()
                })
                .unwrap_or_default();
            result.push((id.clone(), terms));
        }
    }
    result
}

fn prepare(directory: &dyn Directory, probe: Arc<Probe>) -> IndexWriter<'_> {
    let mut writer = writer(directory, probe.clone());
    add(&mut writer, "a", Some("x"));
    writer.commit().unwrap();
    assert_eq!(visible(directory), [("a".into(), vec![(b"x".to_vec(), 1)])]);
    probe.clear();
    add(&mut writer, "b", Some("x x"));
    add(&mut writer, "c", None);
    writer
}

fn unchanged(directory: &dyn Directory, before: &SegmentInfos) {
    assert_eq!(
        segment_infos::read_latest(directory).unwrap().segments,
        before.segments
    );
    assert_eq!(visible(directory), [("a".into(), vec![(b"x".to_vec(), 1)])]);
}

fn outputs_absent(directory: &dyn Directory, segment: &str) {
    for extension in EXTENSIONS {
        assert!(
            directory.open(&format!("{segment}.{extension}")).is_err(),
            "{segment}.{extension}"
        );
    }
}

fn retry(writer: &mut IndexWriter<'_>, directory: &dyn Directory, probe: &Probe) {
    probe.mode.store(0, Ordering::SeqCst);
    probe.clear();
    writer.commit().unwrap();
    assert_eq!(
        visible(directory),
        [
            ("a".into(), vec![(b"x".to_vec(), 1)]),
            ("b".into(), vec![(b"x".to_vec(), 2)]),
            ("c".into(), vec![]),
        ]
    );
    assert_eq!(probe.attempts.lock().unwrap().len(), 1);
    let events = probe.events.lock().unwrap();
    assert_eq!(
        events
            .iter()
            .filter(|e| e.starts_with("start_document:"))
            .count(),
        2
    );
    assert_eq!(events.iter().filter(|e| e.as_str() == "close").count(), 1);
    assert_eq!(
        &events[events.len().saturating_sub(2)..],
        ["finish:2", "close"]
    );
}

#[test]
fn acquisition_callback_finish_and_close_failures_preserve_commit_and_retry_inputs() {
    for (mode, message) in [
        (BEFORE_OPEN, "term vectors before open"),
        (BEFORE_POSITION, "term vectors primary position failure"),
        (AFTER_FINISH, "term vectors after native finish"),
        (AFTER_CLOSE, "term vectors after native close"),
        (POSITION_AND_CLOSE, "term vectors primary position failure"),
    ] {
        let directory = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = prepare(&directory, probe.clone());
        let before = segment_infos::read_latest(&directory).unwrap();
        probe.mode.store(mode, Ordering::SeqCst);
        let error = writer.commit().unwrap_err();
        assert!(error.to_string().contains(message), "{error}");
        assert!(!error.to_string().contains("secondary"));
        unchanged(&directory, &before);
        let attempts = probe.attempts.lock().unwrap().clone();
        assert_eq!(attempts.len(), 1);
        outputs_absent(&directory, &attempts[0]);
        assert_eq!(
            probe
                .events
                .lock()
                .unwrap()
                .iter()
                .filter(|e| e.as_str() == "close")
                .count(),
            usize::from(mode != BEFORE_OPEN)
        );
        retry(&mut writer, &directory, &probe);
    }
}

#[test]
fn native_partial_acquisition_write_and_close_failures_are_cleaned() {
    for operation in [
        FailureOp::CreateOutput,
        FailureOp::Write,
        FailureOp::CloseOutput,
    ] {
        for extension in EXTENSIONS {
            let directory =
                MockDirectoryWrapper::new(&mut JavaRandom::new(74), ByteBuffersDirectory::new());
            let probe = Arc::new(Probe::default());
            let mut writer = prepare(&directory, probe.clone());
            let before = segment_infos::read_latest(&directory).unwrap();
            let injected = FailOn::once(move |context| {
                context.op == operation && context.name.ends_with(&format!(".{extension}"))
            });
            directory.fail_on(injected.clone());
            let error = writer.commit().unwrap_err();
            assert!(
                error.to_string().contains(&format!("{operation:?}")),
                "{error}"
            );
            assert_eq!(injected.fired(), 1);
            unchanged(&directory, &before);
            let attempts = probe.attempts.lock().unwrap().clone();
            assert_eq!(attempts.len(), 1);
            outputs_absent(&directory, &attempts[0]);
            retry(&mut writer, &directory, &probe);
        }
    }
}

#[test]
fn failed_merge_acquisition_preserves_each_preexisting_candidate_file() {
    for extension in EXTENSIONS {
        let directory = ByteBuffersDirectory::new();
        let probe = Arc::new(Probe::default());
        let mut writer = prepare(&directory, probe.clone());
        writer.commit().unwrap();
        probe.clear();
        let before = segment_infos::read_latest(&directory).unwrap();
        let values = visible(&directory);
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
        let collision = format!("{candidate}.{extension}");
        let mut output = directory.create_output(&collision).unwrap();
        output.write_bytes(b"preexisting belongs to another attempt");
        output.close().unwrap();
        let error = writer.force_merge(1).unwrap_err();
        assert!(
            error.to_string().to_lowercase().contains("exist"),
            "{error}"
        );
        assert_eq!(
            &*directory.open(&collision).unwrap(),
            b"preexisting belongs to another attempt"
        );
        assert_eq!(
            segment_infos::read_latest(&directory).unwrap().segments,
            before.segments
        );
        assert_eq!(visible(&directory), values);
        for (name, bytes) in source_files {
            assert_eq!(&*directory.open(&name).unwrap(), bytes, "{name}");
        }
        for other in EXTENSIONS.into_iter().filter(|e| *e != extension) {
            assert!(directory.open(&format!("{candidate}.{other}")).is_err());
        }
        directory.delete_file(&collision).unwrap();
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        assert_eq!(visible(&directory), values);
        assert_eq!(
            segment_infos::read_latest(&directory)
                .unwrap()
                .segments
                .len(),
            1
        );
    }
}

#[test]
fn failed_configured_flush_preserves_preexisting_files_and_retries_buffered_documents() {
    #[derive(Debug)]
    struct NativeCodec;

    impl Lucene104Codec for NativeCodec {
        fn term_vectors_writer_factory(&self) -> Option<&dyn TermVectorsWriterFactory> {
            Some(&lucene_codecs::term_vectors_writer::NativeTermVectorsWriterFactory)
        }
    }

    fn document(id: &str, body: &str) -> lucene_codecs::stored_fields::Document {
        use lucene_codecs::stored_fields::StoredField;
        lucene_codecs::stored_fields::Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(id.into()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::String(body.into()),
                },
            ],
        }
    }

    for asserting in [false, true] {
        for extension in EXTENSIONS {
            let directory = ByteBuffersDirectory::new();
            let mut writer = IndexWriter::open(
                &directory,
                vec![
                    FieldInfo::new("id", 0),
                    FieldInfo::new("body", 1)
                        .with_index_options(IndexOptions::DocsAndFreqsAndPositionsAndOffsets)
                        .with_omit_norms(true)
                        .with_store_term_vectors(true),
                ],
                "Lucene104",
                LuceneVersion {
                    major: 10,
                    minor: 5,
                    bugfix: 0,
                },
            )
            .unwrap();
            let codec: Arc<dyn Lucene104Codec> = if asserting {
                Arc::new(AssertingTermVectorsCodec)
            } else {
                Arc::new(NativeCodec)
            };
            writer.set_codec(Some(codec)).unwrap();
            writer.set_use_compound_file(false);
            writer.set_max_full_flush_merge_wait_millis(0);
            writer.add_postings_field("body").unwrap();
            writer.add_term_vector_field("body").unwrap();
            writer.add_document(document("a", "x")).unwrap();
            writer.commit().unwrap();
            let before = segment_infos::read_latest(&directory).unwrap();
            writer.add_document(document("b", "x x")).unwrap();

            // Insert after IndexWriter opens: its initialization deliberately
            // reclaims old orphan index files, whereas a failed flush owns
            // only the names its own build successfully created.
            let candidate = format!("_{}", lucene_util::base36::to_base36(before.counter));
            let collision = format!("{candidate}.{extension}");
            for name in [collision.as_str(), "_z.fdt", "notes.txt"] {
                let mut output = directory.create_output(name).unwrap();
                output.write_bytes(b"preexisting belongs to another attempt");
                output.close().unwrap();
            }
            let mut snapshot: Vec<_> = directory
                .list_all()
                .unwrap()
                .into_iter()
                .map(|name| {
                    let bytes = directory.open(&name).unwrap().to_vec();
                    (name, bytes)
                })
                .collect();
            snapshot.sort_by(|a, b| a.0.cmp(&b.0));

            let error = writer.flush().unwrap_err();
            assert!(
                error.to_string().to_lowercase().contains("exist"),
                "asserting={asserting}, extension={extension}: {error}"
            );
            let mut names = directory.list_all().unwrap();
            names.sort();
            assert_eq!(
                names,
                snapshot
                    .iter()
                    .map(|(name, _)| name.clone())
                    .collect::<Vec<_>>(),
                "asserting={asserting}, extension={extension}"
            );
            for (name, bytes) in &snapshot {
                assert_eq!(&*directory.open(name).unwrap(), bytes, "{name}");
            }
            unchanged(&directory, &before);

            directory.delete_file(&collision).unwrap();
            writer.commit().unwrap();
            assert_eq!(
                visible(&directory),
                [
                    ("a".into(), vec![(b"x".to_vec(), 1)]),
                    ("b".into(), vec![(b"x".to_vec(), 2)]),
                ]
            );
            for name in ["_z.fdt", "notes.txt"] {
                assert_eq!(
                    &*directory.open(name).unwrap(),
                    b"preexisting belongs to another attempt"
                );
            }
        }
    }
}
