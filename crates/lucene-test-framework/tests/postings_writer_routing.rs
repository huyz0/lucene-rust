//! Java IndexWriter snapshots replayed through installed asserting postings groups.
//! Native readers verify physical deleted docs, complete terms/postings and norms.
use lucene_codecs::{
    codec::Lucene104Codec,
    field_infos::IndexOptions,
    per_field_postings::{Lucene104PostingsFormat, PER_FIELD_FORMAT_KEY, PER_FIELD_SUFFIX_KEY},
    postings_consumer::{PostingsWriteSink, PostingsWriteState, PostingsWriterFactory},
    postings_writer::{self, FieldNorms, FieldPostingsInput, TermSource},
    stored_fields::FieldValue,
};
use lucene_index::{
    buffered_updates::Term,
    document::{Document, Field, FieldToken, FieldTokens, FieldType, NumericDocValuesField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::{IndexSortField, LuceneVersion},
    segment_infos,
};
use lucene_search::{
    directory_reader::DirectoryReader,
    reader::{IndexReader, LeafReader, PostingsFlags, NO_MORE_DOCS},
};
use lucene_store::{Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_postings::AssertingPostingsCodec;
use lucene_util::test_support::TempDir;
use std::{
    collections::BTreeMap,
    sync::{Arc, Mutex},
};

#[derive(Clone, Debug)]
struct Definition {
    name: String,
    options: IndexOptions,
    omit_norms: bool,
    min: usize,
    max: usize,
}
#[derive(Clone, Debug, Default)]
struct Input {
    id: String,
    rank: i64,
    fields: BTreeMap<String, Vec<FieldToken>>,
}
#[derive(Clone, Debug)]
struct Consumer {
    phase: String,
    segment: String,
    suffix: String,
    max_doc: i32,
    operation: String,
    names: String,
    closed: bool,
}
#[derive(Debug, Default)]
struct Probe {
    phase: Mutex<String>,
    consumers: Mutex<Vec<Consumer>>,
}
#[derive(Debug)]
struct Codec {
    probe: Arc<Probe>,
    definitions: Vec<Definition>,
}
impl Lucene104Codec for Codec {
    fn postings_writer_factory(&self) -> Option<&dyn PostingsWriterFactory> {
        Some(self)
    }
    fn postings_format_for_field(&self, name: &str) -> Lucene104PostingsFormat {
        let field = self
            .definitions
            .iter()
            .find(|field| field.name == name)
            .unwrap();
        Lucene104PostingsFormat::new(field.min, field.max).unwrap()
    }
}
impl PostingsWriterFactory for Codec {
    fn fields_consumer<'a>(
        &self,
        state: PostingsWriteState<'a>,
    ) -> postings_writer::Result<Box<dyn PostingsWriteSink + 'a>> {
        let mut consumers = self.probe.consumers.lock().unwrap();
        assert!(consumers
            .iter()
            .all(|consumer| consumer.segment != state.segment || !consumer.closed));
        let index = consumers.len();
        consumers.push(Consumer {
            phase: self.probe.phase.lock().unwrap().clone(),
            segment: state.segment.into(),
            suffix: state.suffix.into(),
            max_doc: state.max_doc,
            operation: String::new(),
            names: String::new(),
            closed: false,
        });
        drop(consumers);
        Ok(Box::new(Sink {
            delegate: AssertingPostingsCodec.fields_consumer(state)?,
            probe: self.probe.clone(),
            index,
            names: state
                .field_infos
                .iter()
                .map(|info| (info.number, info.name.clone()))
                .collect(),
        }))
    }
}
struct Sink<'a> {
    delegate: Box<dyn PostingsWriteSink + 'a>,
    probe: Arc<Probe>,
    index: usize,
    names: BTreeMap<i32, String>,
}
impl Sink<'_> {
    fn record(&self, operation: &str, inputs: &[FieldPostingsInput<'_, &dyn TermSource>]) {
        let mut consumers = self.probe.consumers.lock().unwrap();
        let segment = consumers[self.index].segment.clone();
        assert!(consumers
            .iter()
            .all(|consumer| consumer.segment != segment || !consumer.closed));
        let record = &mut consumers[self.index];
        assert!(record.operation.is_empty());
        record.operation = operation.into();
        record.names = inputs
            .iter()
            .map(|input| self.names[&input.field_number].clone())
            .collect::<Vec<_>>()
            .join(",");
    }
}
impl PostingsWriteSink for Sink<'_> {
    fn write(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.record("write", inputs);
        self.delegate.write(inputs, norms)
    }
    fn merge(
        &mut self,
        inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
        norms: &[FieldNorms<'_>],
    ) -> postings_writer::Result<()> {
        self.record("merge", inputs);
        self.delegate.merge(inputs, norms)
    }
    fn close(&mut self) -> postings_writer::Result<()> {
        let mut consumers = self.probe.consumers.lock().unwrap();
        let record = &mut consumers[self.index];
        assert!(!record.closed);
        record.closed = true;
        drop(consumers);
        self.delegate.close()
    }
}
fn options(name: &str) -> IndexOptions {
    match name {
        "NONE" => IndexOptions::None,
        "DOCS" => IndexOptions::Docs,
        "DOCS_AND_FREQS" => IndexOptions::DocsAndFreqs,
        "DOCS_AND_FREQS_AND_POSITIONS" => IndexOptions::DocsAndFreqsAndPositions,
        "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS" => {
            IndexOptions::DocsAndFreqsAndPositionsAndOffsets
        }
        _ => panic!("unsupported fixture options {name}"),
    }
}
fn option_name(options: IndexOptions) -> &'static str {
    match options {
        IndexOptions::None => "NONE",
        IndexOptions::Docs => "DOCS",
        IndexOptions::DocsAndFreqs => "DOCS_AND_FREQS",
        IndexOptions::DocsAndFreqsAndPositions => "DOCS_AND_FREQS_AND_POSITIONS",
        IndexOptions::DocsAndFreqsAndPositionsAndOffsets => {
            "DOCS_AND_FREQS_AND_POSITIONS_AND_OFFSETS"
        }
        _ => panic!("unexpected options"),
    }
}
fn unhex(text: &str) -> Vec<u8> {
    if text == "-" {
        return vec![];
    }
    (0..text.len())
        .step_by(2)
        .map(|index| u8::from_str_radix(&text[index..index + 2], 16).unwrap())
        .collect()
}
fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "-".into()
    } else {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
fn writer<'a>(dir: &'a dyn Directory, codec: Arc<Codec>, sorted: bool) -> IndexWriter<'a> {
    let mut writer = IndexWriter::open(
        dir,
        vec![],
        "Lucene104",
        LuceneVersion {
            major: 10,
            minor: 5,
            bugfix: 0,
        },
    )
    .unwrap();
    writer.enable_explicit_documents().unwrap();
    writer.set_codec(Some(codec)).unwrap();
    writer.set_use_compound_file(false);
    writer.set_max_full_flush_merge_wait_millis(0);
    let mut policy = LogDocMergePolicy::new();
    policy.set_merge_factor(1000).unwrap();
    writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
    if sorted {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .unwrap();
    }
    writer
}
fn add(writer: &mut IndexWriter<'_>, input: &Input, definitions: &[Definition]) {
    let mut doc = Document::new();
    doc.add(Field::stored(
        "stored_id",
        FieldValue::String(input.id.clone()),
    ));
    doc.add(NumericDocValuesField::new("rank", input.rank));
    for definition in definitions {
        let Some(tokens) = input.fields.get(&definition.name) else {
            continue;
        };
        let mut ty = FieldType::new();
        ty.set_tokenized(true).unwrap();
        ty.set_index_options(definition.options).unwrap();
        ty.set_omit_norms(definition.omit_norms).unwrap();
        doc.add(
            Field::from_token_stream(
                &definition.name,
                FieldTokens {
                    tokens: tokens.clone(),
                    final_offset: tokens
                        .iter()
                        .map(|token| token.end_offset)
                        .max()
                        .unwrap_or(0),
                    ..FieldTokens::default()
                },
                ty,
            )
            .unwrap(),
        );
    }
    writer.add_fields_document(&doc).unwrap();
}
fn snapshots(
    dir: &dyn Directory,
    case: &str,
    phase: &str,
    definitions: &[Definition],
) -> Vec<String> {
    let reader = DirectoryReader::open(dir).unwrap();
    let mut rows = vec![format!(
        "CASE\t{case}\t{phase}\t{}\t{}\t{}",
        reader.max_doc(),
        reader.num_docs(),
        reader.segment_readers().len()
    )];
    for (leaf, segment) in reader.segment_readers().iter().enumerate() {
        segment.check_integrity().unwrap();
        let mut rank = segment.numeric_doc_values("rank").unwrap().unwrap();
        let id = segment
            .field_infos()
            .fields
            .iter()
            .find(|info| info.name == "stored_id")
            .unwrap()
            .number;
        for doc in 0..segment.max_doc() {
            let stored = segment.stored_document(doc).unwrap().unwrap();
            let FieldValue::String(identity) = &stored
                .fields
                .iter()
                .find(|field| field.field_number == id)
                .unwrap()
                .value
            else {
                panic!("stored id")
            };
            assert!(rank.advance_exact(doc).unwrap());
            rows.push(format!(
                "DOC\t{case}\t{phase}\t{leaf}\t{doc}\t{}\t{identity}\t{}",
                i32::from(segment.live_docs().is_none_or(|bits| bits.get_doc(doc))),
                rank.long_value()
            ));
        }
        for definition in definitions {
            let name = &definition.name;
            let info = segment
                .field_infos()
                .fields
                .iter()
                .find(|info| &info.name == name);
            let terms = segment.terms(name).unwrap();
            rows.push(format!("FIELD\t{case}\t{phase}\t{leaf}\t{name}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}",
                info.map_or(-1, |info| info.number), option_name(info.map_or(IndexOptions::None, |info| info.index_options)),
                i32::from(info.is_none_or(|info| info.omit_norms)), i32::from(info.is_some_and(|info| info.store_payloads)),
                info.and_then(|info| info.attributes.iter().find(|(key, _)| key == PER_FIELD_FORMAT_KEY).map(|(_, value)| value.as_str())).unwrap_or("-"),
                info.and_then(|info| info.attributes.iter().find(|(key, _)| key == PER_FIELD_SUFFIX_KEY).map(|(_, value)| value.as_str())).unwrap_or("-"),
                i32::from(terms.is_some()), terms.as_ref().map_or(0, |terms| terms.size()), terms.as_ref().map_or(0, |terms| terms.sum_doc_freq()),
                terms.as_ref().map_or(0, |terms| terms.sum_total_term_freq()), terms.as_ref().map_or(0, |terms| terms.doc_count())));
            let mut norms = segment.norm_values(name).unwrap();
            for doc in 0..segment.max_doc() {
                let value = norms
                    .as_mut()
                    .and_then(|norms| {
                        norms
                            .advance_exact(doc)
                            .unwrap()
                            .then(|| norms.long_value().to_string())
                    })
                    .unwrap_or_else(|| "-".into());
                rows.push(format!(
                    "NORM\t{case}\t{phase}\t{leaf}\t{name}\t{doc}\t{value}"
                ));
            }
            let Some(terms) = terms else {
                continue;
            };
            let mut iterator = terms.iterator().unwrap();
            while let Some(term) = iterator.next().unwrap().map(<[u8]>::to_vec) {
                let term = hex(&term);
                let frequency = iterator.doc_freq().unwrap();
                let total = iterator.total_term_freq().unwrap();
                rows.push(format!(
                    "TERM\t{case}\t{phase}\t{leaf}\t{name}\t{term}\t{frequency}\t{total}"
                ));
                let mut postings = iterator.postings(PostingsFlags::All).unwrap();
                loop {
                    let doc = postings.next_doc().unwrap();
                    if doc == NO_MORE_DOCS {
                        break;
                    }
                    let freq = postings.freq();
                    rows.push(format!(
                        "POSTING\t{case}\t{phase}\t{leaf}\t{name}\t{term}\t{doc}\t{freq}"
                    ));
                    if !definition.options.subsumes_positions() {
                        continue;
                    }
                    for occurrence in 0..freq {
                        let position = postings.next_position().unwrap();
                        rows.push(format!("POSITION\t{case}\t{phase}\t{leaf}\t{name}\t{term}\t{doc}\t{occurrence}\t{position}\t{}\t{}\t{}",
                            postings.start_offset(), postings.end_offset(), hex(postings.payload().unwrap_or_default())));
                    }
                }
            }
        }
        let prefix = format!("{}_", segment.segment_name);
        for file in dir.list_all().unwrap() {
            let Some(rest) = file.strip_prefix(&prefix) else {
                continue;
            };
            let Some((suffix, extension)) = rest.rsplit_once('.') else {
                continue;
            };
            if suffix.starts_with("Lucene104_")
                && ["psm", "doc", "pos", "pay", "tim", "tip", "tmd"].contains(&extension)
            {
                rows.push(format!(
                    "GROUP_FILE\t{case}\t{phase}\t{leaf}\t{suffix}\t{extension}"
                ));
            }
        }
    }
    let checks = lucene_index::check_index::check_directory(dir).unwrap();
    let failures: Vec<_> = checks
        .iter()
        .filter(|result| !result.all_passed())
        .map(|result| (&result.segment_name, result.failures()))
        .collect();
    assert!(
        failures.is_empty(),
        "{case}/{phase}: CheckIndex failed: {failures:?}"
    );
    let infos = segment_infos::read_latest(dir).unwrap();
    assert!(infos
        .segments
        .iter()
        .all(|segment| segment.codec_name == "Lucene104"));
    rows.sort();
    rows
}
fn check_callbacks(rows: &[Vec<&str>], probe: &Probe) {
    let mut expected = BTreeMap::<usize, Vec<&Vec<&str>>>::new();
    for row in rows.iter().filter(|row| row[0] == "EVENT") {
        expected
            .entry(row[4].parse().unwrap())
            .or_default()
            .push(row);
    }
    let actual = probe.consumers.lock().unwrap();
    assert_eq!(actual.len(), expected.len());
    let mut batches = BTreeMap::new();
    for consumer in actual.iter() {
        let next = batches.len();
        batches.entry(consumer.segment.clone()).or_insert(next);
    }
    let mut ordered: Vec<_> = actual.iter().collect();
    ordered.sort_by_key(|consumer| (batches[&consumer.segment], consumer.suffix.clone()));
    for (actual, expected) in ordered.into_iter().zip(expected.values()) {
        let open = expected.iter().find(|row| row[6] == "open").unwrap();
        assert_eq!(actual.phase, open[2]);
        assert_eq!(actual.suffix, open[5]);
        assert_eq!(format!("maxDoc={}", actual.max_doc), open[7]);
        let operation = expected
            .iter()
            .find(|row| row[6] == "write" || row[6] == "merge")
            .unwrap();
        assert_eq!(actual.operation, operation[6]);
        let mapped = expected
            .iter()
            .find(|row| row[6] == "native-write")
            .unwrap();
        assert_eq!(actual.names, mapped[7]);
        assert!(actual.closed);
        assert_eq!(expected.iter().filter(|row| row[6] == "close").count(), 1);
        assert_eq!(
            expected
                .iter()
                .filter(|row| row[6] == "native-close")
                .count(),
            2
        );
    }
}
fn replay(fixture: &str, case: &str) {
    let rows: Vec<Vec<&str>> = fixture
        .lines()
        .filter(|line| !line.starts_with('#'))
        .map(|line| line.split('\t').collect::<Vec<_>>())
        .filter(|row| row.get(1) == Some(&case))
        .collect();
    assert!(!rows.is_empty());
    let definitions: Vec<_> = rows
        .iter()
        .filter(|row| row[0] == "FIELD_DEF")
        .map(|row| Definition {
            name: row[2].into(),
            options: options(row[3]),
            omit_norms: row[4] == "1",
            min: row[6].parse().unwrap(),
            max: row[7].parse().unwrap(),
        })
        .collect();
    let mut inputs = BTreeMap::<usize, Input>::new();
    for row in &rows {
        match row[0] {
            "INPUT_DOC" => {
                inputs.insert(
                    row[2].parse().unwrap(),
                    Input {
                        id: row[3].into(),
                        rank: row[4].parse().unwrap(),
                        ..Input::default()
                    },
                );
            }
            "INPUT_EMPTY" => {
                inputs
                    .get_mut(&row[2].parse().unwrap())
                    .unwrap()
                    .fields
                    .entry(row[3].into())
                    .or_default();
            }
            "INPUT_TOKEN" => {
                inputs
                    .get_mut(&row[2].parse().unwrap())
                    .unwrap()
                    .fields
                    .entry(row[3].into())
                    .or_default()
                    .push(FieldToken {
                        term: unhex(row[4]),
                        position_increment: row[5].parse().unwrap(),
                        start_offset: row[6].parse().unwrap(),
                        end_offset: row[7].parse().unwrap(),
                        term_frequency: 1,
                        payload: (row[8] != "-").then(|| unhex(row[8])),
                    });
            }
            _ => {}
        }
    }
    let temp = TempDir::new(&format!("postings-routing-{case}"));
    let dir = FsDirectory::open(&temp);
    let probe = Arc::new(Probe::default());
    let codec = Arc::new(Codec {
        probe: probe.clone(),
        definitions: definitions.clone(),
    });
    let sorted = case.ends_with("_sorted");
    let mut active = Some(writer(&dir, codec.clone(), sorted));
    let mut next = 0;
    for row in &rows {
        match row[0] {
            "FLUSH" => {
                *probe.phase.lock().unwrap() = row[2].into();
                let writer = active.get_or_insert_with(|| writer(&dir, codec.clone(), sorted));
                let through: usize = row[3].parse().unwrap();
                while next <= through {
                    add(writer, &inputs[&next], &definitions);
                    next += 1;
                }
                writer.commit().unwrap();
            }
            "DELETE" => {
                *probe.phase.lock().unwrap() = row[2].into();
                active
                    .as_mut()
                    .unwrap()
                    .delete_documents_by_term(&[Term::new("id", row[3])])
                    .unwrap();
            }
            "CASE" => {
                let phase = row[2];
                *probe.phase.lock().unwrap() = phase.into();
                if phase == "merged" || phase == "remerged" {
                    active.as_mut().unwrap().force_merge(1).unwrap();
                    active.as_mut().unwrap().commit().unwrap();
                }
                if phase == "deleted" {
                    active.as_mut().unwrap().commit().unwrap();
                }
                let actual = snapshots(&dir, case, phase, &definitions);
                let kinds = [
                    "CASE",
                    "DOC",
                    "FIELD",
                    "NORM",
                    "TERM",
                    "POSTING",
                    "POSITION",
                    "GROUP_FILE",
                ];
                let mut expected: Vec<String> = rows
                    .iter()
                    .filter(|row| kinds.contains(&row[0]) && row[2] == phase)
                    .map(|row| {
                        if row[0] == "GROUP_FILE" {
                            row[..row.len() - 1].join("\t")
                        } else {
                            row.join("\t")
                        }
                    })
                    .collect();
                expected.sort();
                let mismatch = actual
                    .iter()
                    .zip(&expected)
                    .position(|(left, right)| left != right)
                    .unwrap_or(actual.len().min(expected.len()));
                assert!(actual == expected, "{case}/{phase}: {} actual rows / {} expected; first difference at {mismatch}: actual={:?} expected={:?}", actual.len(), expected.len(), actual.get(mismatch), expected.get(mismatch));
                let checkpoint = rows
                    .iter()
                    .find(|row| row[0] == "CHECKPOINT" && row[2] == phase)
                    .unwrap();
                assert_eq!(
                    probe.consumers.lock().unwrap().len(),
                    checkpoint[3].parse::<usize>().unwrap(),
                    "{case}/{phase}"
                );
                if phase == "initial" {
                    drop(active.take());
                }
            }
            _ => {}
        }
    }
    check_callbacks(&rows, &probe);
    assert!(probe
        .consumers
        .lock()
        .unwrap()
        .iter()
        .all(|consumer| consumer.closed));
}
#[test]
fn java_writer_snapshots_cover_native_asserting_groups_sorting_sparse_fields_deletes_and_postings_boundaries(
) {
    let fixture = std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/postings_writer_routing.tsv"
    ))
    .unwrap();
    for case in [
        "rich_unsorted",
        "rich_sorted",
        "empty_unsorted",
        "empty_sorted",
        "block_boundary",
    ] {
        replay(&fixture, case);
    }
}

mod configured {
    use super::*;
    use lucene_codecs::{
        field_infos::{DocValuesSkipIndexType, DocValuesType, FieldInfo},
        norms::NormsField,
        norms_writer::{NormsWriteSink, NormsWriteState, NormsWriterFactory},
        per_field_doc_values::Lucene90DocValuesFormat,
        stored_fields::{self, StoredField},
        stored_fields_writer::{
            StoredFieldsWriteSink, StoredFieldsWriteState, StoredFieldsWriterFactory,
        },
    };
    use lucene_test_framework::codecs::{
        asserting_norms_writer::AssertingNormsCodec,
        asserting_stored_fields_writer::AssertingStoredFieldsCodec,
    };
    #[derive(Debug, Default)]
    struct Order(Arc<Mutex<Vec<&'static str>>>);
    impl Order {
        fn record(&self, action: &'static str) {
            self.0.lock().unwrap().push(action);
        }
    }
    impl Lucene104Codec for Order {
        fn postings_writer_factory(&self) -> Option<&dyn PostingsWriterFactory> {
            Some(self)
        }
        fn norms_writer_factory(&self) -> Option<&dyn NormsWriterFactory> {
            Some(self)
        }
        fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
            Some(self)
        }
        fn doc_values_format_for_field(&self, _: &str) -> Lucene90DocValuesFormat {
            self.record("doc-values");
            Lucene90DocValuesFormat::default()
        }
    }
    impl NormsWriterFactory for Order {
        fn norms_consumer<'a>(
            &self,
            state: NormsWriteState<'a>,
        ) -> lucene_codecs::norms::Result<Box<dyn NormsWriteSink + 'a>> {
            self.record("norms-open");
            Ok(Box::new(Norms {
                delegate: AssertingNormsCodec.norms_consumer(state)?,
                order: self.0.clone(),
            }))
        }
    }
    struct Norms<'a> {
        delegate: Box<dyn NormsWriteSink + 'a>,
        order: Arc<Mutex<Vec<&'static str>>>,
    }
    impl NormsWriteSink for Norms<'_> {
        fn add_norms_field(
            &mut self,
            info: &FieldInfo,
            values: NormsField<'_>,
        ) -> lucene_codecs::norms::Result<()> {
            self.delegate.add_norms_field(info, values)
        }
        fn close(&mut self) -> lucene_codecs::norms::Result<()> {
            self.delegate.close()?;
            self.order.lock().unwrap().push("norms-close");
            Ok(())
        }
    }
    impl StoredFieldsWriterFactory for Order {
        fn fields_writer<'a>(
            &self,
            state: StoredFieldsWriteState<'a>,
        ) -> stored_fields::Result<Box<dyn StoredFieldsWriteSink + 'a>> {
            self.record("stored-open");
            AssertingStoredFieldsCodec.fields_writer(state)
        }
    }
    impl PostingsWriterFactory for Order {
        fn fields_consumer<'a>(
            &self,
            state: PostingsWriteState<'a>,
        ) -> postings_writer::Result<Box<dyn PostingsWriteSink + 'a>> {
            self.record("postings-open");
            Ok(Box::new(Postings {
                delegate: AssertingPostingsCodec.fields_consumer(state)?,
                order: self.0.clone(),
            }))
        }
    }
    struct Postings<'a> {
        delegate: Box<dyn PostingsWriteSink + 'a>,
        order: Arc<Mutex<Vec<&'static str>>>,
    }
    impl PostingsWriteSink for Postings<'_> {
        fn write(
            &mut self,
            inputs: &[FieldPostingsInput<'_, &dyn TermSource>],
            norms: &[FieldNorms<'_>],
        ) -> postings_writer::Result<()> {
            self.order.lock().unwrap().push("postings-write");
            let body = inputs.iter().find(|input| input.field_number == 1).unwrap();
            assert_eq!(
                body.terms
                    .iter()
                    .find(|term| term.term() == b"x")
                    .unwrap()
                    .docs(),
                [(0, 2), (1, 1), (2, 2)]
            );
            assert_eq!(
                norms
                    .iter()
                    .find(|norms| norms.field_number == 1)
                    .unwrap()
                    .values,
                [2, 1, 3]
            );
            self.delegate.write(inputs, norms)
        }
        fn close(&mut self) -> postings_writer::Result<()> {
            self.delegate.close()?;
            self.order.lock().unwrap().push("postings-close");
            Ok(())
        }
    }
    #[test]
    fn configured_fast_and_general_flushes_install_after_norms_and_docvalues() {
        for general in [false, true] {
            let temp = TempDir::new("configured-postings-routing");
            let dir = FsDirectory::open(&temp);
            let order = Arc::new(Order::default());
            let fields = vec![
                FieldInfo::new("id", 0)
                    .with_index_options(IndexOptions::Docs)
                    .with_omit_norms(true),
                FieldInfo::new("body", 1)
                    .with_index_options(IndexOptions::DocsAndFreqsAndPositions)
                    .with_store_term_vectors(general),
                FieldInfo::new("rank", 2).with_doc_values(
                    DocValuesType::Numeric,
                    DocValuesSkipIndexType::None,
                    -1,
                ),
            ];
            let mut writer = IndexWriter::open(
                &dir,
                fields,
                "Lucene104",
                LuceneVersion {
                    major: 10,
                    minor: 5,
                    bugfix: 0,
                },
            )
            .unwrap();
            writer.set_codec(Some(order.clone())).unwrap();
            writer.set_use_compound_file(false);
            writer.add_postings_field("id").unwrap();
            writer.add_postings_field("body").unwrap();
            writer.set_doc_values_field(Some("rank")).unwrap();
            if general {
                writer.set_term_vector_field(Some("body")).unwrap();
            }
            for (doc, (body, rank)) in [("x x", -4), ("x", 8), ("y x x", 2)]
                .into_iter()
                .enumerate()
            {
                writer
                    .add_document(stored_fields::Document {
                        fields: vec![
                            StoredField {
                                field_number: 0,
                                value: FieldValue::String(format!("d{doc}")),
                            },
                            StoredField {
                                field_number: 1,
                                value: FieldValue::String(body.into()),
                            },
                            StoredField {
                                field_number: 2,
                                value: FieldValue::Long(rank),
                            },
                        ],
                    })
                    .unwrap();
            }
            assert!(order.0.lock().unwrap().is_empty());
            writer.commit().unwrap();
            let events = order.0.lock().unwrap().clone();
            let index = |name| {
                events
                    .iter()
                    .position(|event| *event == name)
                    .unwrap_or_else(|| panic!("{name}: {events:?}"))
            };
            assert!(index("norms-close") < index("doc-values"), "{events:?}");
            assert!(index("doc-values") < index("stored-open"), "{events:?}");
            // Existing native flush order runs postings before stored fields;
            // whole Java sibling-order parity remains outside this seam.
            assert!(index("postings-close") < index("stored-open"), "{events:?}");
            assert!(
                index("postings-open") < index("postings-write")
                    && index("postings-write") < index("postings-close"),
                "{events:?}"
            );
            let reader = DirectoryReader::open(&dir).unwrap();
            let segment = &reader.segment_readers()[0];
            assert_eq!(reader.num_docs(), 3);
            segment.check_integrity().unwrap();
            assert_eq!(segment.term_vectors_reader().unwrap().is_some(), general);
            let terms = segment.terms("body").unwrap().unwrap();
            let mut iterator = terms.iterator().unwrap();
            assert!(iterator.try_seek_exact(b"x").unwrap());
            assert_eq!(iterator.doc_freq().unwrap(), 3);
            assert_eq!(iterator.total_term_freq().unwrap(), 5);
            let mut norms = segment.norm_values("body").unwrap().unwrap();
            let mut ranks = segment.numeric_doc_values("rank").unwrap().unwrap();
            for (doc, (rank, length)) in [(-4, 2), (8, 1), (2, 3)].into_iter().enumerate() {
                let doc = doc as i32;
                assert!(norms.advance_exact(doc).unwrap());
                assert_eq!(norms.long_value(), length);
                assert!(ranks.advance_exact(doc).unwrap());
                assert_eq!(ranks.long_value(), rank);
                assert_eq!(
                    segment.stored_document(doc).unwrap().unwrap().fields.len(),
                    3
                );
            }
            assert!(lucene_index::check_index::check_directory(&dir)
                .unwrap()
                .iter()
                .all(|result| result.all_passed()));
        }
    }
}
