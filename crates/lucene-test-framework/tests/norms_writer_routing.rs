//! Real Java writer snapshots for installed asserting norms consumers.
//! Producer iteration traces are Java provenance: the Rust hook receives the
//! already computed columns. Complete schema, presence, values and scores are
//! compared, including present-empty fields and deleted-only merged columns.

use std::sync::{Arc, Mutex};

use lucene_codecs::{
    codec::Lucene104Codec,
    field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo, IndexOptions},
    norms::{self, NormsField},
    norms_writer::{NormsWriteSink, NormsWriteState, NormsWriterFactory},
    per_field_doc_values::Lucene90DocValuesFormat,
    per_field_postings::Lucene104PostingsFormat,
    stored_fields::{self, FieldValue, StoredField},
    stored_fields_writer::{
        StoredFieldsWriteSink, StoredFieldsWriteState, StoredFieldsWriterFactory,
    },
};
use lucene_index::{
    buffered_updates::Term,
    document::{self, Field, NumericDocValuesField, Store, StringField, TextField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::{IndexSortField, LuceneVersion},
};
use lucene_search::{
    directory_reader::{DirectoryReader, SegmentReader},
    index_searcher::IndexSearcher,
    query::{BooleanQuery, Clause, TermQuery},
};
use lucene_store::{DataInput, Directory, FsDirectory};
use lucene_test_framework::codecs::{
    asserting_norms_writer::AssertingNormsCodec,
    asserting_stored_fields_writer::AssertingStoredFieldsCodec,
};
use lucene_util::test_support::TempDir;

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord)]
struct Column {
    max_doc: i32,
    field: String,
    number: i32,
    values: Vec<(i32, i64)>,
}

#[derive(Debug, Default)]
struct Probe {
    events: Mutex<Vec<String>>,
    columns: Mutex<Vec<Column>>,
}

impl Probe {
    fn record(&self, event: impl Into<String>) {
        self.events.lock().unwrap().push(event.into());
    }

    fn take_events(&self) -> Vec<String> {
        std::mem::take(&mut *self.events.lock().unwrap())
    }

    fn take_columns(&self) -> Vec<Column> {
        std::mem::take(&mut *self.columns.lock().unwrap())
    }
}

#[derive(Debug)]
struct CountingCodec(Arc<Probe>);

impl Lucene104Codec for CountingCodec {
    fn norms_writer_factory(&self) -> Option<&dyn NormsWriterFactory> {
        Some(self)
    }

    fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
        Some(self)
    }

    fn postings_format_for_field(&self, field: &str) -> Lucene104PostingsFormat {
        self.0.record(format!("postings:{field}"));
        Lucene104PostingsFormat::default()
    }

    fn doc_values_format_for_field(&self, field: &str) -> Lucene90DocValuesFormat {
        self.0.record(format!("doc-values:{field}"));
        Lucene90DocValuesFormat::default()
    }
}

impl NormsWriterFactory for CountingCodec {
    fn norms_consumer<'a>(
        &self,
        state: NormsWriteState<'a>,
    ) -> norms::Result<Box<dyn NormsWriteSink + 'a>> {
        self.0
            .record(format!("norms-open:{}:{}", state.segment, state.max_doc));
        Ok(Box::new(CountingNorms {
            max_doc: state.max_doc,
            segment: state.segment.to_owned(),
            delegate: AssertingNormsCodec.norms_consumer(state)?,
            probe: self.0.clone(),
        }))
    }
}

struct CountingNorms<'a> {
    max_doc: i32,
    segment: String,
    delegate: Box<dyn NormsWriteSink + 'a>,
    probe: Arc<Probe>,
}

impl NormsWriteSink for CountingNorms<'_> {
    fn add_norms_field(&mut self, field: &FieldInfo, values: NormsField<'_>) -> norms::Result<()> {
        let values_for_check = match values {
            NormsField::Dense(number, values) => {
                assert_eq!(number, field.number);
                assert_eq!(values.len(), usize::try_from(self.max_doc).unwrap());
                values
                    .iter()
                    .enumerate()
                    .map(|(doc, &value)| (i32::try_from(doc).unwrap(), value))
                    .collect()
            }
            NormsField::Sparse(number, values) => {
                assert_eq!(number, field.number);
                assert!(values.windows(2).all(|pair| pair[0].0 < pair[1].0));
                assert!(values
                    .iter()
                    .all(|&(doc, _)| (0..self.max_doc).contains(&doc)));
                values.to_vec()
            }
        };
        self.probe.columns.lock().unwrap().push(Column {
            max_doc: self.max_doc,
            field: field.name.clone(),
            number: field.number,
            values: values_for_check,
        });
        self.probe
            .record(format!("norms-add:{}:{}", self.segment, field.name));
        self.delegate.add_norms_field(field, values)
    }

    fn close(&mut self) -> norms::Result<()> {
        self.delegate.close()?;
        self.probe.record(format!("norms-close:{}", self.segment));
        Ok(())
    }
}

impl StoredFieldsWriterFactory for CountingCodec {
    fn fields_writer<'a>(
        &self,
        state: StoredFieldsWriteState<'a>,
    ) -> stored_fields::Result<Box<dyn StoredFieldsWriteSink + 'a>> {
        self.0.record(format!("stored-open:{}", state.segment));
        Ok(Box::new(CountingStored {
            segment: state.segment.to_owned(),
            delegate: AssertingStoredFieldsCodec.fields_writer(state)?,
            probe: self.0.clone(),
        }))
    }
}

struct CountingStored<'a> {
    segment: String,
    delegate: Box<dyn StoredFieldsWriteSink + 'a>,
    probe: Arc<Probe>,
}

impl StoredFieldsWriteSink for CountingStored<'_> {
    fn start_document(&mut self) -> stored_fields::Result<()> {
        self.delegate.start_document()
    }
    fn write_field(&mut self, number: i32, value: &FieldValue) -> stored_fields::Result<()> {
        self.delegate.write_field(number, value)
    }
    fn write_field_data_input(
        &mut self,
        number: i32,
        input: &mut dyn DataInput,
        length: usize,
    ) -> stored_fields::Result<()> {
        self.delegate.write_field_data_input(number, input, length)
    }
    fn finish_document(&mut self) -> stored_fields::Result<()> {
        self.delegate.finish_document()
    }
    fn finish(&mut self, num_docs: i32) -> stored_fields::Result<()> {
        self.delegate.finish(num_docs)
    }
    fn close(&mut self) -> stored_fields::Result<()> {
        self.delegate.close()?;
        self.probe.record(format!("stored-close:{}", self.segment));
        Ok(())
    }
}

#[derive(Debug)]
struct Input {
    id: String,
    rank: i64,
    fields: Vec<(String, Option<String>)>,
}

fn hex(text: &str) -> String {
    assert!(text.len().is_multiple_of(2));
    let bytes = text
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect();
    String::from_utf8(bytes).unwrap()
}

fn inputs(fixture: &str, case: &str) -> Vec<Input> {
    fixture
        .lines()
        .filter(|row| row.starts_with(&format!("INPUT\t{case}\t")))
        .enumerate()
        .map(|(ordinal, row)| {
            let cells: Vec<_> = row.split('\t').collect();
            assert_eq!(cells[2].parse::<usize>().unwrap(), ordinal);
            Input {
                id: cells[3].to_owned(),
                rank: cells[4].parse().unwrap(),
                fields: cells[5]
                    .split(';')
                    .map(|entry| {
                        let (field, value) = entry.split_once(':').unwrap();
                        (field.to_owned(), (value != "-").then(|| hex(value)))
                    })
                    .collect(),
            }
        })
        .collect()
}

fn schema(general: bool) -> Vec<FieldInfo> {
    let mut fields = vec![
        FieldInfo::new("id", 0)
            .with_index_options(IndexOptions::Docs)
            .with_omit_norms(true),
        FieldInfo::new("rank", 1).with_doc_values(
            DocValuesType::Numeric,
            DocValuesSkipIndexType::None,
            -1,
        ),
    ];
    for (name, number) in [
        ("constant", 2),
        ("varying", 3),
        ("omitted", 4),
        ("sparse", 5),
        ("vanish", 6),
    ] {
        fields.push(
            FieldInfo::new(name, number)
                .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
                .with_omit_norms(name == "omitted")
                .with_store_term_vectors(general && name == "constant"),
        );
    }
    fields
}

fn writer<'a>(directory: &'a dyn Directory, probe: Arc<Probe>, case: &str) -> IndexWriter<'a> {
    let explicit = case.starts_with("explicit");
    let general = case.starts_with("general");
    let mut writer = IndexWriter::open(
        directory,
        if explicit {
            Vec::new()
        } else {
            schema(general)
        },
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
    writer.set_use_compound_file(false);
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    if explicit {
        writer.enable_explicit_documents().unwrap();
    } else {
        for field in schema(general)
            .iter()
            .filter(|field| field.index_options != IndexOptions::None)
        {
            writer.add_postings_field(&field.name).unwrap();
        }
        writer.set_doc_values_field(Some("rank")).unwrap();
        if general {
            writer.set_term_vector_field(Some("constant")).unwrap();
        }
    }
    if case.ends_with("_sorted") && !case.ends_with("_unsorted") {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .unwrap();
    }
    writer
}

fn add(writer: &mut IndexWriter<'_>, input: &Input, explicit: bool) {
    if explicit {
        let mut document = document::Document::new();
        document.add(StringField::new("id", &input.id, Store::Yes));
        document.add(NumericDocValuesField::new("rank", input.rank));
        for (name, value) in &input.fields {
            if let Some(text) = value {
                if name == "omitted" {
                    let mut field_type = document::FieldType::new();
                    field_type
                        .set_index_options(IndexOptions::DocsAndFreqsAndPositions)
                        .unwrap();
                    field_type.set_tokenized(true).unwrap();
                    field_type.set_omit_norms(true).unwrap();
                    document.add(Field::from_string(name, text, field_type).unwrap());
                } else {
                    document.add(TextField::new(name, text, Store::No));
                }
            }
        }
        writer.add_fields_document(&document).unwrap();
    } else {
        let fields = schema(false);
        let mut document = stored_fields::Document {
            fields: vec![
                StoredField {
                    field_number: 0,
                    value: FieldValue::String(input.id.clone()),
                },
                StoredField {
                    field_number: 1,
                    value: FieldValue::Long(input.rank),
                },
            ],
        };
        for (name, value) in &input.fields {
            if let Some(text) = value {
                document.fields.push(StoredField {
                    field_number: fields
                        .iter()
                        .find(|field| field.name == *name)
                        .unwrap()
                        .number,
                    value: FieldValue::String(text.clone()),
                });
            }
        }
        writer.add_document(document).unwrap();
    }
}

fn id(segment: &SegmentReader, doc: i32) -> String {
    let document = segment.stored_document(doc).unwrap().unwrap();
    let field = document
        .fields
        .iter()
        .find(|field| field.field_number == 0)
        .unwrap();
    let FieldValue::String(id) = &field.value else {
        panic!("stored ID changed type")
    };
    id.clone()
}

fn snapshot(directory: &dyn Directory, case: &str, phase: &str) -> Vec<String> {
    let reader = DirectoryReader::open(directory).unwrap();
    let mut rows = vec![format!(
        "CASE\t{case}\t{phase}\t{}\t{}\t{}",
        reader.max_doc(),
        reader.num_docs(),
        reader.segment_readers().len()
    )];
    let mut all_ids = Vec::new();
    for (leaf, segment) in reader.segment_readers().iter().enumerate() {
        segment.check_integrity().unwrap();
        for field in &segment.field_infos().fields {
            let has_norms = field.index_options != IndexOptions::None && !field.omit_norms;
            rows.push(format!(
                "FIELD\t{case}\t{phase}\t{leaf}\t{}\t{}\t{}\t{}\t{}",
                field.name,
                field.number,
                document::index_options_name(field.index_options),
                field.omit_norms,
                has_norms
            ));
        }
        for doc in 0..segment.max_doc {
            let id = id(segment, doc);
            all_ids.push(id.clone());
            let live = i32::from(segment.live_docs().is_none_or(|bits| bits.get_doc(doc)));
            rows.push(format!("DOC\t{case}\t{phase}\t{leaf}\t{doc}\t{live}\t{id}"));
            for field in segment
                .field_infos()
                .fields
                .iter()
                .filter(|field| field.index_options != IndexOptions::None && !field.omit_norms)
            {
                let entry = segment
                    .norms_entry(field.number)
                    .expect("field claims norms");
                let value = norms::norm_value(segment.norms_data().unwrap(), entry, doc).unwrap();
                rows.push(format!(
                    "NORM\t{case}\t{phase}\t{leaf}\t{doc}\t{}\t{}\t{}",
                    field.name,
                    i32::from(value.is_some()),
                    value.map_or_else(|| "-".into(), |value| value.to_string())
                ));
            }
        }
    }
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let owned = reader
        .field_norms_by_field(&["varying".to_owned()])
        .unwrap();
    let norms: Vec<_> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let query = BooleanQuery {
        should: vec![Clause::Term(TermQuery::new("varying", "x"))],
        ..BooleanQuery::default()
    };
    for (ordinal, hit) in searcher
        .search(&query, reader.max_doc() as usize)
        .unwrap()
        .score_docs
        .iter()
        .enumerate()
    {
        rows.push(format!(
            "QUERY\t{case}\t{phase}\t{ordinal}\t{}\t{}\t{}",
            hit.doc,
            all_ids[usize::try_from(hit.doc).unwrap()],
            hit.score.to_bits()
        ));
    }
    rows
}

fn assert_snapshot(fixture: &str, directory: &dyn Directory, case: &str, phase: &str) {
    let mut expected: Vec<_> = fixture
        .lines()
        .filter(|line| {
            ["CASE", "FIELD", "DOC", "NORM", "QUERY"]
                .iter()
                .any(|kind| line.starts_with(&format!("{kind}\t{case}\t{phase}\t")))
        })
        .map(str::to_owned)
        .collect();
    assert!(!expected.is_empty(), "missing Java snapshot {case}/{phase}");
    let mut actual = snapshot(directory, case, phase);
    expected.sort();
    actual.sort();
    assert_eq!(actual, expected, "{case}/{phase}");
}

fn expected_columns(fixture: &str, case: &str, phase: &str, leaves: &[usize]) -> Vec<Column> {
    let rows: Vec<Vec<&str>> = fixture
        .lines()
        .map(|line| line.split('\t').collect())
        .filter(|cells: &Vec<&str>| cells.len() > 3 && cells[1] == case && cells[2] == phase)
        .collect();
    let mut result = Vec::new();
    for &leaf in leaves {
        let max_doc = i32::try_from(
            rows.iter()
                .filter(|row| row[0] == "DOC" && row[3].parse::<usize>().unwrap() == leaf)
                .count(),
        )
        .unwrap();
        for row in rows.iter().filter(|row| {
            row[0] == "FIELD" && row[3].parse::<usize>().unwrap() == leaf && row[8] == "true"
        }) {
            let name = row[4];
            let values = rows
                .iter()
                .filter(|row| {
                    row[0] == "NORM"
                        && row[3].parse::<usize>().unwrap() == leaf
                        && row[5] == name
                        && row[6] == "1"
                })
                .map(|row| (row[4].parse().unwrap(), row[7].parse().unwrap()))
                .collect();
            result.push(Column {
                max_doc,
                field: name.to_owned(),
                number: row[5].parse().unwrap(),
                values,
            });
        }
    }
    result.sort();
    result
}

fn assert_callbacks(probe: &Probe, mut expected: Vec<Column>, merging: bool) {
    let mut actual = probe.take_columns();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected);
    let events = probe.take_events();
    let opens: Vec<_> = events
        .iter()
        .enumerate()
        .filter(|(_, event)| event.starts_with("norms-open:"))
        .collect();
    assert!(!opens.is_empty());
    for &(start, event) in &opens {
        let segment = event.split(':').nth(1).unwrap();
        let close = format!("norms-close:{segment}");
        assert_eq!(
            events.iter().filter(|event| **event == close).count(),
            1,
            "{events:?}"
        );
        let end = events.iter().position(|event| *event == close).unwrap();
        assert!(end > start);
        assert!(
            events[start + 1..end]
                .iter()
                .any(|event| event.starts_with("norms-add:")),
            "{events:?}"
        );
        if merging {
            let stored = events
                .iter()
                .position(|event| *event == format!("stored-close:{segment}"))
                .expect("installed stored fields writer closed");
            assert!(
                stored < start,
                "stored fields must precede merge norms: {events:?}"
            );
        }
    }
    // Selectors execute during postings/doc-values construction, so they
    // independently expose a consumer accidentally deferred until publish.
    let mut closed = false;
    for event in &events {
        if event.starts_with("norms-open:") {
            closed = false;
        }
        if event.starts_with("norms-close:") {
            closed = true;
        }
        if event.starts_with("postings:") || event.starts_with("doc-values:") {
            assert!(
                closed,
                "norms must close before field selectors: {events:?}"
            );
        }
    }
}

#[test]
fn fast_general_explicit_flush_append_and_deleted_sorted_merges_match_java() {
    let fixture = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/norms_writer_routing.tsv"
    ))
    .unwrap();
    for case in [
        "fast_unsorted",
        "fast_sorted",
        "general_unsorted",
        "general_sorted",
        "explicit_unsorted",
        "explicit_sorted",
    ] {
        let temporary = TempDir::new("norms-writer-routing");
        let directory = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let inputs = inputs(&fixture, case);
        assert_eq!(inputs.len(), 8);
        let explicit = case.starts_with("explicit");
        {
            let mut writer = writer(&directory, probe.clone(), case);
            for batch in inputs[..6].chunks(3) {
                for input in batch {
                    add(&mut writer, input, explicit);
                }
                writer.commit().unwrap();
            }
            assert_snapshot(&fixture, &directory, case, "initial");
            assert_callbacks(
                &probe,
                expected_columns(&fixture, case, "initial", &[0, 1]),
                false,
            );
        }
        let mut writer = writer(&directory, probe.clone(), case);
        for input in &inputs[6..] {
            add(&mut writer, input, explicit);
        }
        writer.commit().unwrap();
        assert_snapshot(&fixture, &directory, case, "append");
        assert_callbacks(
            &probe,
            expected_columns(&fixture, case, "append", &[2]),
            false,
        );
        writer
            .delete_documents_by_term(&[
                Term::new("id", "b"),
                Term::new("id", "e"),
                Term::new("id", "g"),
            ])
            .unwrap();
        writer.commit().unwrap();
        assert_snapshot(&fixture, &directory, case, "deleted");
        assert!(
            probe.take_columns().is_empty(),
            "deleting must preserve existing norms"
        );
        assert!(
            probe.take_events().is_empty(),
            "deleting must not acquire a norms consumer"
        );
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        assert_snapshot(&fixture, &directory, case, "merged");
        assert_callbacks(
            &probe,
            expected_columns(&fixture, case, "merged", &[0]),
            true,
        );
    }
}

#[test]
fn resetting_the_codec_restores_native_flush_and_merge_without_callback_acquisition() {
    let fixture = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/norms_writer_routing.tsv"
    ))
    .unwrap();
    let inputs = inputs(&fixture, "explicit_sorted");
    let temporary = TempDir::new("norms-writer-reset");
    let directory = FsDirectory::open(&temporary);
    let probe = Arc::new(Probe::default());
    let mut writer = writer(&directory, probe.clone(), "explicit_sorted");
    writer.set_codec(None).unwrap();
    for batch in inputs[..6].chunks(3) {
        for input in batch {
            add(&mut writer, input, true);
        }
        writer.commit().unwrap();
    }
    assert_snapshot(&fixture, &directory, "explicit_sorted", "initial");
    assert!(probe.take_columns().is_empty());
    assert!(probe.take_events().is_empty());
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    assert!(probe.take_columns().is_empty());
    assert!(probe.take_events().is_empty());
    let reader = DirectoryReader::open(&directory).unwrap();
    assert_eq!(reader.num_docs(), 6);
    assert_eq!(reader.segment_readers().len(), 1);
    assert_eq!(
        reader.segment_readers()[0]
            .norms_entry(2)
            .unwrap()
            .num_docs_with_field,
        6
    );
}

#[test]
fn installed_norms_preserve_full_postings_block_impacts_and_bm25_scores() {
    let mut control = None;
    for installed in [false, true] {
        let temporary = TempDir::new("norms-writer-postings-blocks");
        let directory = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let mut writer = writer(&directory, probe.clone(), "explicit_unsorted");
        if !installed {
            writer.set_codec(None).unwrap();
        }
        // Each source has a complete 128-posting block plus a tail. Different
        // frequencies and lengths exercise nontrivial competitive frontiers.
        for doc in 0usize..270 {
            let input = Input {
                id: doc.to_string(),
                rank: i64::try_from(doc).unwrap(),
                fields: vec![
                    ("constant".into(), Some("x".into())),
                    ("varying".into(), Some(vec!["x"; 1 + doc % 17].join(" "))),
                    (
                        "sparse".into(),
                        (!doc.is_multiple_of(3)).then(|| {
                            if doc.is_multiple_of(5) {
                                String::new()
                            } else {
                                "s s".into()
                            }
                        }),
                    ),
                    ("vanish".into(), doc.is_multiple_of(19).then(|| "v".into())),
                    ("omitted".into(), Some("o".into())),
                ],
            };
            add(&mut writer, &input, true);
            if (doc + 1).is_multiple_of(135) {
                writer.commit().unwrap();
            }
        }
        let before = snapshot(&directory, "blocked", "initial");
        let terms: Vec<_> = (0usize..270)
            .filter(|doc| doc.is_multiple_of(13))
            .map(|doc| Term::new("id", doc.to_string()))
            .collect();
        writer.delete_documents_by_term(&terms).unwrap();
        writer.commit().unwrap();
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        let after = snapshot(&directory, "blocked", "merged");
        let results = lucene_index::check_index::check_directory(&directory).unwrap();
        assert!(
            results.iter().all(|result| result.all_passed()),
            "{results:?}"
        );
        assert!(results
            .iter()
            .flat_map(|result| &result.checks)
            .any(|check| check.name == "postings.impacts:varying" && check.passed()));
        if let Some((expected_before, expected_after)) = &control {
            assert_eq!(&before, expected_before);
            assert_eq!(&after, expected_after);
            let columns = probe.take_columns();
            assert!(!columns.is_empty());
            assert!(columns.iter().any(|column| column.max_doc == 249));
        } else {
            assert!(probe.take_columns().is_empty());
            assert!(probe.take_events().is_empty());
            control = Some((before, after));
        }
    }
}
