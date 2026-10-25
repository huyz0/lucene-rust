//! Real IndexWriter routing through the asserting stored-fields writer.
//! Java-produced live-document snapshots verify sorted/deleting/remapping merges;
//! the trace proves that document callbacks reached the installed wrapper.

use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, Mutex,
};

use lucene_codecs::{
    codec::Lucene104Codec,
    stored_fields::{self, FieldValue},
    stored_fields_writer::{
        StoredFieldsWriteSink, StoredFieldsWriteState, StoredFieldsWriterFactory,
    },
};
use lucene_index::{
    buffered_updates::Term,
    document::{Document, Field, NumericDocValuesField, Store, StringField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::{IndexSortField, LuceneVersion},
    segment_infos,
};
use lucene_search::directory_reader::DirectoryReader;
use lucene_store::{DataInput, Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_stored_fields_writer::AssertingStoredFieldsCodec;
use lucene_util::test_support::TempDir;

#[derive(Debug, Default)]
struct Probe {
    events: Mutex<Vec<String>>,
    fail_open: AtomicBool,
    fail_finish: AtomicBool,
    fail_close: AtomicBool,
}

impl Probe {
    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }

    fn take(&self) -> Vec<String> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }
}

#[derive(Debug)]
struct CountingCodec(Arc<Probe>);

impl Lucene104Codec for CountingCodec {
    fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
        Some(self)
    }
}

impl StoredFieldsWriterFactory for CountingCodec {
    fn fields_writer<'a>(
        &self,
        state: StoredFieldsWriteState<'a>,
    ) -> stored_fields::Result<Box<dyn StoredFieldsWriteSink + 'a>> {
        self.0.record("create");
        if self.0.fail_open.load(Ordering::SeqCst) {
            return Err(failure("routing open failure"));
        }
        Ok(Box::new(CountingSink {
            delegate: AssertingStoredFieldsCodec.fields_writer(state)?,
            probe: self.0.clone(),
        }))
    }
}

struct CountingSink<'a> {
    delegate: Box<dyn StoredFieldsWriteSink + 'a>,
    probe: Arc<Probe>,
}

fn failure(message: &'static str) -> stored_fields::Error {
    lucene_store::Error::Io(std::io::Error::other(message)).into()
}

impl StoredFieldsWriteSink for CountingSink<'_> {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.delegate.start_document()?;
        self.probe.record("start");
        Ok(())
    }

    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        self.delegate.write_field(number, value)?;
        self.probe
            .record(format!("field:{number}:{}", render_value(value)));
        Ok(())
    }

    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        self.delegate
            .write_field_data_input(number, input, length)?;
        self.probe.record(format!("input:{number}:{length}"));
        Ok(())
    }

    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.delegate.finish_document()?;
        self.probe.record("end");
        Ok(())
    }

    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
        self.probe.record(format!("finish:{num_docs}"));
        if self.probe.fail_finish.load(Ordering::SeqCst) {
            return Err(failure("routing finish failure"));
        }
        self.delegate.finish(num_docs)
    }

    fn close(&mut self) -> stored_fields::Result<()> {
        self.probe.record("close");
        self.delegate.close()?;
        if self.probe.fail_close.load(Ordering::SeqCst) {
            return Err(failure("routing close failure"));
        }
        Ok(())
    }
}

fn writer<'a>(dir: &'a dyn Directory, probe: Arc<Probe>, sorted: bool) -> IndexWriter<'a> {
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
    writer
        .set_codec(Some(Arc::new(CountingCodec(probe))))
        .unwrap();
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    writer.enable_explicit_documents().unwrap();
    if sorted {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .unwrap();
    }
    writer
}

fn simple_document(id: &str) -> Document {
    let mut document = Document::new();
    document.add(StringField::new("id", id, Store::No));
    document.add(Field::stored("value", FieldValue::String(id.into())));
    document
}

fn assert_trace(events: &[String], documents: usize, fields: usize) {
    assert_eq!(
        events.iter().filter(|e| e.as_str() == "start").count(),
        documents,
        "{events:?}"
    );
    assert_eq!(
        events.iter().filter(|e| e.as_str() == "end").count(),
        documents,
        "{events:?}"
    );
    assert_eq!(
        events
            .iter()
            .filter(|e| e.starts_with("field:") || e.starts_with("input:"))
            .count(),
        fields,
        "{events:?}"
    );
    assert_eq!(
        events.iter().filter(|e| e.as_str() == "create").count(),
        1,
        "{events:?}"
    );
    assert_eq!(
        events.iter().filter(|e| e.as_str() == "close").count(),
        1,
        "{events:?}"
    );
    assert!(
        events.iter().any(|e| e == &format!("finish:{documents}")),
        "{events:?}"
    );
}

#[test]
fn automatic_flush_explicit_flush_and_append_install_the_real_asserting_writer() {
    let temporary = TempDir::new("stored-routing-flush");
    let dir = FsDirectory::open(&temporary);
    let probe = Arc::new(Probe::default());
    {
        let mut writer = writer(&dir, probe.clone(), false);
        writer.set_max_buffered_docs(2).unwrap();
        writer.add_fields_document(&simple_document("a")).unwrap();
        assert!(probe.take().is_empty());
        writer.add_fields_document(&simple_document("b")).unwrap();
        assert_trace(&probe.take(), 2, 2);
        writer.add_fields_document(&simple_document("c")).unwrap();
        writer.flush().unwrap();
        assert_trace(&probe.take(), 1, 1);
        writer.commit().unwrap();
        assert!(probe.take().is_empty());
    }
    {
        let mut writer = writer(&dir, probe.clone(), false);
        writer.add_fields_document(&simple_document("d")).unwrap();
        writer.commit().unwrap();
        assert_trace(&probe.take(), 1, 1);
    }
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!(reader.num_docs(), 4);
    assert_eq!(reader.max_doc(), 4);
    let values: Vec<_> = reader
        .segment_readers()
        .iter()
        .flat_map(|segment| {
            (0..segment.max_doc).map(|doc| {
                render_value(&segment.stored_document(doc).unwrap().unwrap().fields[0].value)
            })
        })
        .collect();
    assert_eq!(values, ["s:61", "s:62", "s:63", "s:64"]);
}

#[test]
fn ordinary_fixed_schema_flush_routes_fields_and_closes_on_finish_failure() {
    for close_also_fails in [false, true] {
        let temporary = TempDir::new("stored-routing-ordinary");
        let dir = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        // This is the fixed-schema stored-document API. Deliberately do not
        // enable explicit documents: it exercises the ordinary flush branch.
        let mut writer = IndexWriter::open(
            &dir,
            vec![lucene_codecs::field_infos::FieldInfo::new("value", 0)],
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        writer
            .set_codec(Some(Arc::new(CountingCodec(probe.clone()))))
            .unwrap();
        writer.set_max_full_flush_merge_wait_millis(0);
        let document = |value: &str| stored_fields::Document {
            fields: vec![stored_fields::StoredField {
                field_number: 0,
                value: FieldValue::String(value.into()),
            }],
        };
        writer.add_document(document("a")).unwrap();
        writer.flush().unwrap();
        assert_trace(&probe.take(), 1, 1);
        writer.commit().unwrap();
        assert!(probe.take().is_empty());
        let original_memory = writer.segment_infos().segments.clone();
        let original_disk = segment_infos::read_latest(&dir).unwrap().segments;
        let original_view = snapshot(&dir, "ordinary", "before");
        probe.fail_finish.store(true, Ordering::SeqCst);
        probe.fail_close.store(close_also_fails, Ordering::SeqCst);
        writer.add_document(document("b")).unwrap();
        let error = writer.flush().unwrap_err();
        assert_eq!(error.to_string(), "routing finish failure");
        assert_trace(&probe.take(), 1, 1);
        assert_eq!(writer.segment_infos().segments, original_memory);
        assert_eq!(
            segment_infos::read_latest(&dir).unwrap().segments,
            original_disk
        );
        writer.try_rollback().unwrap();
        drop(writer);
        assert_eq!(snapshot(&dir, "ordinary", "before"), original_view);
    }
}

#[test]
fn finish_failure_closes_the_sink_and_preserves_the_original_commit_and_error() {
    for close_also_fails in [false, true] {
        let temporary = TempDir::new("stored-routing-failure");
        let dir = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone(), false);
        writer.add_fields_document(&simple_document("a")).unwrap();
        writer.commit().unwrap();
        assert_trace(&probe.take(), 1, 1);
        let committed = segment_infos::read_latest(&dir).unwrap();
        let original_segments = committed.segments.clone();
        let original_in_memory_segments = writer.segment_infos().segments.clone();
        let commit_name =
            lucene_store::directory::segments_file_name(committed.generation).unwrap();
        let original_commit = dir.open(&commit_name).unwrap().to_vec();
        probe.fail_finish.store(true, Ordering::SeqCst);
        probe.fail_close.store(close_also_fails, Ordering::SeqCst);
        writer.add_fields_document(&simple_document("b")).unwrap();
        let error = writer.flush().unwrap_err();
        assert_eq!(error.to_string(), "routing finish failure");
        assert_trace(&probe.take(), 1, 1);
        assert_eq!(writer.segment_infos().segments, original_in_memory_segments);
        assert_eq!(
            segment_infos::read_latest(&dir).unwrap().segments,
            original_segments
        );
        assert_eq!(
            &*dir.open(&commit_name).unwrap(),
            original_commit.as_slice()
        );
        writer.rollback();
        drop(writer);
        let reader = DirectoryReader::open(&dir).unwrap();
        assert_eq!((reader.max_doc(), reader.num_docs()), (1, 1));
        assert_eq!(
            render_value(
                &reader.segment_readers()[0]
                    .stored_document(0)
                    .unwrap()
                    .unwrap()
                    .fields[0]
                    .value
            ),
            "s:61"
        );
        let mut reopened = self::writer(&dir, Arc::new(Probe::default()), false);
        reopened.add_fields_document(&simple_document("c")).unwrap();
        reopened.commit().unwrap();
        assert_eq!(DirectoryReader::open(&dir).unwrap().num_docs(), 2);
    }
}

#[test]
fn merge_finish_failure_closes_the_sink_without_publishing_a_candidate() {
    for sorted in [false, true] {
        let temporary = TempDir::new("stored-routing-merge-failure");
        let dir = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone(), sorted);
        for batch in [["d", "b"], ["c", "a"]] {
            for id in batch {
                let mut document = simple_document(id);
                document.add(NumericDocValuesField::new(
                    "rank",
                    i64::from(id.as_bytes()[0]),
                ));
                writer.add_fields_document(&document).unwrap();
            }
            writer.commit().unwrap();
            assert_trace(&probe.take(), 2, 2);
        }
        let committed = segment_infos::read_latest(&dir).unwrap();
        assert_eq!(committed.segments.len(), 2);
        let original_in_memory_segments = writer.segment_infos().segments.clone();
        let commit_name =
            lucene_store::directory::segments_file_name(committed.generation).unwrap();
        let original_commit = dir.open(&commit_name).unwrap().to_vec();
        let original_view = snapshot(&dir, "failure", "before");
        // Install the failure only after both source segments are durable.
        probe.fail_finish.store(true, Ordering::SeqCst);
        probe.fail_close.store(true, Ordering::SeqCst);
        let error = writer.force_merge(1).unwrap_err();
        assert_eq!(error.to_string(), "routing finish failure");
        assert_trace(&probe.take(), 4, 4);
        assert_eq!(writer.segment_infos().segments, original_in_memory_segments);
        assert_eq!(
            segment_infos::read_latest(&dir).unwrap().segments,
            committed.segments
        );
        assert_eq!(
            &*dir.open(&commit_name).unwrap(),
            original_commit.as_slice()
        );
        assert_eq!(snapshot(&dir, "failure", "before"), original_view);
        // The failed target is unreferenced. Rollback asks the writer's file
        // deleter to refresh those outputs while preserving the source commit.
        writer.try_rollback().unwrap();
        drop(writer);
        assert_eq!(snapshot(&dir, "failure", "before"), original_view);
        let mut reopened = self::writer(&dir, Arc::new(Probe::default()), sorted);
        reopened.force_merge(1).unwrap();
        reopened.commit().unwrap();
        let reader = DirectoryReader::open(&dir).unwrap();
        assert_eq!((reader.max_doc(), reader.num_docs()), (4, 4));
        assert_eq!(reader.segment_readers().len(), 1);
    }
}

#[test]
fn merge_opens_the_factory_before_source_integrity_and_closes_on_corruption() {
    for open_also_fails in [false, true] {
        let temporary = TempDir::new("stored-routing-corrupt-source");
        let dir = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone(), false);
        writer.set_use_compound_file(false);
        for id in ["a", "b"] {
            writer.add_fields_document(&simple_document(id)).unwrap();
            writer.commit().unwrap();
            assert_trace(&probe.take(), 1, 1);
        }
        let committed = segment_infos::read_latest(&dir).unwrap();
        let original_in_memory_segments = writer.segment_infos().segments.clone();
        let original_view = snapshot(&dir, "corruption", "before");
        let data_file = format!("{}.fdt", committed.segments[0].segment_name);
        let original_bytes = dir.open(&data_file).unwrap().to_vec();
        let mut corrupted = original_bytes.clone();
        // Change a body byte immediately before the 16-byte footer. Header,
        // length and footer remain readable, so the failure is merge's real
        // source-integrity checksum check rather than reader construction.
        let corrupt_at = corrupted.len() - lucene_store::codec_util::FOOTER_LENGTH - 1;
        corrupted[corrupt_at] ^= 1;
        std::fs::write(temporary.path().join(&data_file), &corrupted).unwrap();
        probe.fail_open.store(open_also_fails, Ordering::SeqCst);
        probe.fail_close.store(true, Ordering::SeqCst);
        let error = writer.force_merge(1).unwrap_err();
        if open_also_fails {
            assert_eq!(error.to_string(), "routing open failure");
            assert_eq!(probe.take(), ["create"]);
        } else {
            assert!(error.to_string().contains("checksum failed"), "{error}");
            assert_eq!(probe.take(), ["create", "close"]);
        }
        assert_eq!(writer.segment_infos().segments, original_in_memory_segments);
        assert_eq!(
            segment_infos::read_latest(&dir).unwrap().segments,
            committed.segments
        );
        std::fs::write(temporary.path().join(&data_file), original_bytes).unwrap();
        writer.try_rollback().unwrap();
        drop(writer);
        assert_eq!(snapshot(&dir, "corruption", "before"), original_view);
    }
}

#[test]
fn add_indexes_merged_reenters_document_and_field_callbacks() {
    let source_temporary = TempDir::new("stored-routing-import-source");
    let source = FsDirectory::open(&source_temporary);
    {
        let mut source_writer = writer(&source, Arc::new(Probe::default()), false);
        for id in ["a", "b"] {
            source_writer
                .add_fields_document(&simple_document(id))
                .unwrap();
            source_writer.commit().unwrap();
        }
    }
    let temporary = TempDir::new("stored-routing-import");
    let dir = FsDirectory::open(&temporary);
    let probe = Arc::new(Probe::default());
    let mut writer = writer(&dir, probe.clone(), false);
    writer.add_indexes_merged(&[&source]).unwrap();
    assert_trace(&probe.take(), 2, 2);
    writer.commit().unwrap();
    assert!(probe.take().is_empty());
    let reader = DirectoryReader::open(&dir).unwrap();
    assert_eq!((reader.max_doc(), reader.num_docs()), (2, 2));
    assert_eq!(reader.segment_readers().len(), 1);
    for (doc, expected) in [(0, "s:61"), (1, "s:62")] {
        assert_eq!(
            render_value(
                &reader.segment_readers()[0]
                    .stored_document(doc)
                    .unwrap()
                    .unwrap()
                    .fields[0]
                    .value
            ),
            expected
        );
    }
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|b| format!("{b:02x}")).collect()
}

fn unhex(text: &str) -> Vec<u8> {
    (0..text.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&text[i..i + 2], 16).unwrap())
        .collect()
}

fn render_value(value: &FieldValue) -> String {
    match value {
        FieldValue::Binary(v) => format!("b:{}", hex(v)),
        FieldValue::String(v) => format!("s:{}", hex(v.as_bytes())),
        FieldValue::Int(v) => format!("i:{v}"),
        FieldValue::Long(v) => format!("l:{v}"),
        FieldValue::Float(v) => format!("f:{}", v.to_bits()),
        FieldValue::Double(v) => format!("d:{}", v.to_bits()),
    }
}

fn parse_value(kind: &str, value: &str) -> FieldValue {
    match kind {
        "b" => FieldValue::Binary(unhex(value)),
        "s" => FieldValue::String(String::from_utf8(unhex(value)).unwrap()),
        "i" => FieldValue::Int(value.parse().unwrap()),
        "l" => FieldValue::Long(value.parse().unwrap()),
        "f" => FieldValue::Float(f32::from_bits(value.parse().unwrap())),
        "d" => FieldValue::Double(f64::from_bits(value.parse().unwrap())),
        _ => panic!("unknown fixture field type {kind}"),
    }
}

fn input_document(row: &[&str]) -> Document {
    let mut document = Document::new();
    document.add(StringField::new("id", row[2], Store::No));
    document.add(NumericDocValuesField::new("rank", row[3].parse().unwrap()));
    if row[4] != "-" {
        for field in row[4].split(';') {
            let columns: Vec<_> = field.splitn(3, ':').collect();
            document.add(Field::stored(
                columns[0],
                parse_value(columns[1], columns[2]),
            ));
        }
    }
    document
}

fn snapshot(dir: &dyn Directory, case: &str, phase: &str) -> Vec<String> {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut rows = vec![format!(
        "CASE\t{case}\t{phase}\t{}\t{}\t{}",
        reader.max_doc(),
        reader.num_docs(),
        reader.segment_readers().len()
    )];
    let mut ordinal = 0;
    for segment in reader.segment_readers() {
        for doc in 0..segment.max_doc {
            if segment.live_docs().is_some_and(|live| !live.get_doc(doc)) {
                continue;
            }
            let document = segment.stored_document(doc).unwrap().unwrap_or_default();
            let fields: Vec<_> = document
                .fields
                .iter()
                .map(|field| {
                    let name = &segment
                        .field_infos()
                        .fields
                        .iter()
                        .find(|info| info.number == field.field_number)
                        .unwrap()
                        .name;
                    format!("{name}:{}", render_value(&field.value))
                })
                .collect();
            rows.push(format!(
                "DOC\t{case}\t{phase}\t{ordinal}\t{}\t{}",
                segment.doc_base + doc,
                if fields.is_empty() {
                    "-".into()
                } else {
                    fields.join(";")
                }
            ));
            ordinal += 1;
        }
    }
    rows
}

fn field_numbers(dir: &dyn Directory, case: &str, phase: &str) -> Vec<String> {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut rows = Vec::new();
    for (ordinal, segment) in reader.segment_readers().iter().enumerate() {
        for field in &segment.field_infos().fields {
            rows.push(format!(
                "FIELD\t{case}\t{phase}\t{ordinal}\t{}\t{}",
                field.name, field.number
            ));
        }
    }
    rows.sort();
    rows
}

#[test]
fn sorted_unsorted_deleting_and_field_number_remapping_merge_match_java() {
    let fixture = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/stored_fields_writer_routing.tsv"
    ))
    .unwrap();
    let inputs: Vec<Vec<&str>> = fixture
        .lines()
        .filter(|line| line.starts_with("INPUT\t"))
        .map(|line| line.split('\t').collect())
        .collect();
    assert_eq!(inputs.len(), 6);
    for case in [
        "direct_unsorted",
        "direct_sorted",
        "imported_unsorted",
        "imported_sorted",
    ] {
        let sorted = case.ends_with("_sorted");
        let imported = case.starts_with("imported");
        let temporary = TempDir::new("stored-routing-oracle");
        let dir = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&dir, probe.clone(), sorted);
        for batch in inputs.chunks(2) {
            if imported {
                let source_temp = TempDir::new("stored-routing-source");
                let source = FsDirectory::open(&source_temp);
                {
                    let mut source_writer =
                        self::writer(&source, Arc::new(Probe::default()), sorted);
                    for row in batch {
                        source_writer
                            .add_fields_document(&input_document(row))
                            .unwrap();
                    }
                    source_writer.commit().unwrap();
                }
                writer.add_indexes(&[&source]).unwrap();
            } else {
                for row in batch {
                    writer.add_fields_document(&input_document(row)).unwrap();
                }
            }
            writer.commit().unwrap();
        }
        let initial_events = probe.take();
        if imported {
            assert!(
                initial_events.is_empty(),
                "copying segments must not rewrite fields"
            );
        } else {
            assert_eq!(
                initial_events
                    .iter()
                    .filter(|event| event.as_str() == "start")
                    .count(),
                6
            );
            assert_eq!(
                initial_events
                    .iter()
                    .filter(|event| event.as_str() == "end")
                    .count(),
                6
            );
            assert_eq!(
                initial_events
                    .iter()
                    .filter(|event| event.starts_with("field:"))
                    .count(),
                24
            );
            for event in ["create", "close", "finish:2"] {
                assert_eq!(
                    initial_events
                        .iter()
                        .filter(|actual| actual.as_str() == event)
                        .count(),
                    3,
                    "{case}: {initial_events:?}"
                );
            }
        }
        for phase in ["initial", "before", "after"] {
            if phase == "before" {
                writer
                    .delete_documents_by_term(&[Term::new("id", "c")])
                    .unwrap();
                writer.commit().unwrap();
                assert!(probe.take().is_empty());
            } else if phase == "after" {
                writer.force_merge(1).unwrap();
                writer.commit().unwrap();
                assert_trace(&probe.take(), 5, 18);
            }
            let expected: Vec<String> = fixture
                .lines()
                .filter(|line| {
                    line.starts_with(&format!("CASE\t{case}\t{phase}\t"))
                        || line.starts_with(&format!("DOC\t{case}\t{phase}\t"))
                })
                .map(str::to_owned)
                .collect();
            assert_eq!(expected.len(), if phase == "initial" { 7 } else { 6 });
            assert_eq!(snapshot(&dir, case, phase), expected, "{case} {phase}");
            let mut expected_fields: Vec<String> = fixture
                .lines()
                .filter(|line| line.starts_with(&format!("FIELD\t{case}\t{phase}\t")))
                .map(str::to_owned)
                .collect();
            expected_fields.sort();
            assert!(!expected_fields.is_empty());
            assert_eq!(
                field_numbers(&dir, case, phase),
                expected_fields,
                "{case} {phase} field remapping"
            );
        }
    }
}
