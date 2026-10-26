//! Real Java IndexWriter snapshots for installed asserting points writers.
//! Point rows are multisets, retaining duplicate packed values and physical
//! deleted documents before merge. Callback field order is not a contract:
//! Java flushes its field hash table, while Rust visits field-number order.

use std::sync::{Arc, Mutex};

use lucene_codecs::{
    codec::Lucene104Codec,
    field_infos::FieldInfo,
    points::{self, MergePointsField, PackedPointsField},
    points_writer::{PointsWriteSink, PointsWriteState, PointsWriterFactory},
    stored_fields::FieldValue,
};
use lucene_index::{
    buffered_updates::Term,
    document::{Document, Field, FieldType, NumericDocValuesField, Store, StringField},
    index_writer::IndexWriter,
    merge_policy::log::LogDocMergePolicy,
    segment_info::{IndexSortField, LuceneVersion},
};
use lucene_search::{
    directory_reader::{DirectoryReader, SegmentReader},
    extended_query::PointRangeQuery,
    index_searcher::IndexSearcher,
    query::BooleanQuery,
};
use lucene_store::{Directory, FsDirectory};
use lucene_test_framework::codecs::asserting_points::AssertingPointsCodec;
use lucene_util::test_support::TempDir;

const FIELDS: [&str; 7] = ["one", "sparse", "multi", "two", "three", "vanish", "late"];

#[derive(Debug, Default)]
struct Probe {
    writers: Mutex<Vec<Vec<String>>>,
    budgets: Mutex<Vec<Vec<(i32, u64)>>>,
}

#[derive(Debug)]
struct Codec(Arc<Probe>);

impl Lucene104Codec for Codec {
    fn points_writer_factory(&self) -> Option<&dyn PointsWriterFactory> {
        Some(self)
    }
}

impl PointsWriterFactory for Codec {
    fn fields_writer<'a>(
        &self,
        state: PointsWriteState<'a>,
    ) -> points::Result<Box<dyn PointsWriteSink + 'a>> {
        assert!(state.suffix.is_empty());
        let id = {
            let mut writers = self.0.writers.lock().unwrap();
            let id = writers.len();
            writers.push(vec![format!("open:{}", state.max_doc)]);
            id
        };
        Ok(Box::new(Sink {
            delegate: AssertingPointsCodec.fields_writer(state)?,
            probe: self.0.clone(),
            id,
        }))
    }
}

struct Sink<'a> {
    delegate: Box<dyn PointsWriteSink + 'a>,
    probe: Arc<Probe>,
    id: usize,
}

impl Sink<'_> {
    fn record(&self, event: impl Into<String>) {
        self.probe.writers.lock().unwrap()[self.id].push(event.into());
    }
}

impl PointsWriteSink for Sink<'_> {
    fn write_field(&mut self, field: &FieldInfo, values: PackedPointsField) -> points::Result<()> {
        assert_eq!(field.number, values.field_number);
        assert_eq!(field.point_dimension_count, values.num_dims);
        assert_eq!(field.point_index_dimension_count, values.num_index_dims);
        assert_eq!(field.point_num_bytes, values.bytes_per_dim);
        self.record(format!(
            "write:{}:{}:{}:{}:{}",
            field.name, field.number, values.num_dims, values.num_index_dims, values.bytes_per_dim
        ));
        self.delegate.write_field(field, values)
    }

    fn merge(&mut self, fields: Vec<MergePointsField>, native_readers: bool) -> points::Result<()> {
        self.record(format!("merge:{native_readers}"));
        self.probe.budgets.lock().unwrap().push(
            fields
                .iter()
                .map(|field| (field.field_number, field.max_point_count))
                .collect(),
        );
        self.delegate.merge(fields, native_readers)
    }

    fn finish(&mut self) -> points::Result<()> {
        self.record("finish");
        self.delegate.finish()
    }

    fn close(&mut self) -> points::Result<()> {
        self.delegate.close()?;
        self.record("close");
        Ok(())
    }
}

#[derive(Debug)]
struct Input {
    id: String,
    rank: i64,
    points: Vec<(String, i32, i32, i32, Vec<u8>)>,
}

fn decode_hex(text: &str) -> Vec<u8> {
    assert!(text.len().is_multiple_of(2));
    text.as_bytes()
        .chunks_exact(2)
        .map(|pair| u8::from_str_radix(std::str::from_utf8(pair).unwrap(), 16).unwrap())
        .collect()
}

fn hex(bytes: &[u8]) -> String {
    bytes.iter().map(|byte| format!("{byte:02x}")).collect()
}

fn inputs(fixture: &str, case: &str) -> Vec<Input> {
    let mut inputs = Vec::new();
    for line in fixture.lines() {
        let cells: Vec<_> = line.split('\t').collect();
        if cells.len() < 3 || cells[1] != case {
            continue;
        }
        if cells[0] == "INPUT_DOC" {
            assert_eq!(cells[2].parse::<usize>().unwrap(), inputs.len());
            inputs.push(Input {
                id: cells[3].into(),
                rank: cells[4].parse().unwrap(),
                points: Vec::new(),
            });
        } else if cells[0] == "INPUT_POINT" {
            inputs[cells[2].parse::<usize>().unwrap()].points.push((
                cells[3].into(),
                cells[4].parse().unwrap(),
                cells[5].parse().unwrap(),
                cells[6].parse().unwrap(),
                decode_hex(cells[7]),
            ));
        }
    }
    assert!(!inputs.is_empty());
    inputs
}

fn writer<'a>(directory: &'a dyn Directory, probe: Arc<Probe>, sorted: bool) -> IndexWriter<'a> {
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
    if sorted {
        writer
            .set_index_sort(Some(&[IndexSortField::long("rank", false, None)]))
            .unwrap();
    }
    writer
}

fn add(writer: &mut IndexWriter<'_>, input: &Input) {
    let mut document = Document::new();
    document.add(StringField::new("id", &input.id, Store::Yes));
    document.add(NumericDocValuesField::new("rank", input.rank));
    for (name, dimensions, index_dimensions, bytes, value) in &input.points {
        let mut kind = FieldType::new();
        kind.set_dimensions_with_index(*dimensions, *index_dimensions, *bytes)
            .unwrap();
        document.add(Field::from_bytes(name, value.clone(), kind).unwrap());
    }
    writer.add_fields_document(&document).unwrap();
}

fn stored_id(segment: &SegmentReader, doc: i32) -> String {
    let document = segment.stored_document(doc).unwrap().unwrap();
    let field = document
        .fields
        .iter()
        .find(|field| field.field_number == 0)
        .unwrap();
    let FieldValue::String(id) = &field.value else {
        panic!("stored id changed type")
    };
    id.clone()
}

fn snapshot(directory: &dyn Directory, case: &str, phase: &str, large: bool) -> Vec<String> {
    let reader = DirectoryReader::open(directory).unwrap();
    let mut rows = vec![format!(
        "CASE\t{case}\t{phase}\t{}\t{}\t{}",
        reader.max_doc(),
        reader.num_docs(),
        reader.segment_readers().len()
    )];
    let mut identities = Vec::new();
    for (leaf, segment) in reader.segment_readers().iter().enumerate() {
        segment.check_integrity().unwrap();
        for doc in 0..segment.max_doc {
            let id = stored_id(segment, doc);
            identities.push(id.clone());
            let live = i32::from(segment.live_docs().is_none_or(|bits| bits.get_doc(doc)));
            rows.push(format!("DOC\t{case}\t{phase}\t{leaf}\t{doc}\t{live}\t{id}"));
        }
        // Opening this API parses the native Lucene90 .kdm/.kdi/.kdd reader,
        // independently of the installed writer callback object.
        let points = segment.points_reader().unwrap();
        for name in FIELDS {
            let info = segment
                .field_infos()
                .fields
                .iter()
                .find(|info| info.name == name);
            let field = info.and_then(|info| points.field(info.number));
            let number = info.map_or(-1, |info| info.number);
            let dimensions = info.map_or(0, |info| info.point_dimension_count);
            let index_dimensions = info.map_or(0, |info| info.point_index_dimension_count);
            let bytes = info.map_or(0, |info| info.point_num_bytes);
            let (size, documents, min, max) =
                field.map_or((0, 0, "-".into(), "-".into()), |field| {
                    assert_eq!(
                        (dimensions, index_dimensions, bytes),
                        (field.num_dims, field.num_index_dims, field.bytes_per_dim)
                    );
                    (
                        field.point_count,
                        field.doc_count,
                        hex(&field.min_packed_value),
                        hex(&field.max_packed_value),
                    )
                });
            rows.push(format!("FIELD\t{case}\t{phase}\t{leaf}\t{name}\t{number}\t{dimensions}\t{index_dimensions}\t{bytes}\t{size}\t{documents}\t{min}\t{max}"));
            if field.is_some() {
                let leaves = points.decode_leaves(number).unwrap();
                for point in leaves.iter().flat_map(|leaf| &leaf.points) {
                    rows.push(format!(
                        "POINT\t{case}\t{phase}\t{leaf}\t{name}\t{}:{}",
                        point.doc_id,
                        hex(&point.packed_value)
                    ));
                }
                if large {
                    if size > 1024 {
                        assert!(leaves.len() > 1);
                    }
                    rows.push(format!(
                        "TREE\t{case}\t{phase}\t{leaf}\t{name}\t{size}\t{}",
                        leaves.len()
                    ));
                }
            }
        }
    }
    if large {
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        for (ordinal, (min, max)) in [(i32::MIN, i32::MAX), (-20, -10), (7, 7), (600, 700)]
            .into_iter()
            .enumerate()
        {
            let mut query = BooleanQuery::new();
            query.filter.push(
                PointRangeQuery::int_range("multi", &[min], &[max])
                    .unwrap()
                    .into(),
            );
            let mut docs: Vec<_> = searcher
                .search(&query, usize::try_from(reader.max_doc()).unwrap())
                .unwrap()
                .score_docs
                .into_iter()
                .map(|hit| hit.doc)
                .collect();
            docs.sort();
            rows.push(format!(
                "RANGE\t{case}\t{phase}\t{ordinal}\tmulti\t{min}\t{max}\t{}",
                docs.len()
            ));
            for doc in docs {
                rows.push(format!(
                    "RANGE_HIT\t{case}\t{phase}\t{ordinal}\t{doc}\t{}",
                    identities[usize::try_from(doc).unwrap()]
                ));
            }
        }
    }
    let infos = lucene_index::segment_infos::read_latest(directory).unwrap();
    assert!(infos
        .segments
        .iter()
        .all(|segment| segment.codec_name == "Lucene104"));
    let checks = lucene_index::check_index::check_directory(directory).unwrap();
    assert!(
        checks.iter().all(|result| result.all_passed()),
        "{checks:?}"
    );
    rows
}

fn assert_snapshot(fixture: &str, directory: &dyn Directory, case: &str, phase: &str, large: bool) {
    let mut expected: Vec<_> = fixture
        .lines()
        .filter(|line| {
            [
                "CASE",
                "DOC",
                "FIELD",
                "POINT",
                "TREE",
                "RANGE",
                "RANGE_HIT",
            ]
            .iter()
            .any(|kind| line.starts_with(&format!("{kind}\t{case}\t{phase}\t")))
        })
        .map(str::to_owned)
        .collect();
    assert!(!expected.is_empty(), "missing Java snapshot {case}/{phase}");
    let mut actual = snapshot(directory, case, phase, large);
    expected.sort();
    actual.sort();
    assert_eq!(actual, expected, "{case}/{phase}");
}

fn canonical_events(events: &[String]) -> Vec<String> {
    assert!(events.first().unwrap().starts_with("open:"));
    assert_eq!(events.last().unwrap(), "close");
    assert_eq!(
        events
            .iter()
            .filter(|event| event.starts_with("open:"))
            .count(),
        1
    );
    assert_eq!(events.iter().filter(|event| *event == "close").count(), 1);
    let mut result = vec![events[0].clone()];
    if events.iter().any(|event| event.starts_with("merge:")) {
        assert_eq!(events.len(), 3);
        assert!(events[1].starts_with("merge:"));
        result.push(events[1].clone());
    } else {
        assert_eq!(events[events.len() - 2], "finish");
        let mut writes = events[1..events.len() - 2].to_vec();
        assert!(writes.iter().all(|event| event.starts_with("write:")));
        writes.sort();
        result.extend(writes);
        result.push("finish".into());
    }
    result.push("close".into());
    result
}

fn assert_callbacks(fixture: &str, case: &str, probe: &Probe) {
    let mut expected: Vec<Vec<String>> = Vec::new();
    let mut expected_budgets = Vec::new();
    let merged_fields: Vec<Vec<&str>> = fixture
        .lines()
        .map(|line| line.split('\t').collect())
        .filter(|row: &Vec<&str>| {
            row.len() > 5 && row[0] == "FIELD" && row[1] == case && row[2] == "merged"
        })
        .collect();
    let mut budgets = Vec::new();
    for line in fixture
        .lines()
        .filter(|line| line.starts_with(&format!("TRACE\t{case}\t")))
    {
        let trace = line.split('\t').nth(3).unwrap();
        let parts: Vec<_> = trace.split(':').collect();
        let id = parts[1].parse::<usize>().unwrap();
        while expected.len() <= id {
            expected.push(Vec::new());
        }
        let event = match parts[2] {
            "open" => Some(format!("open:{}", parts[4])),
            "write" => Some(parts[2..].join(":")),
            "merge" => Some("merge:true".into()),
            "source" => {
                assert!(matches!(parts[5], "none" | "native"));
                if parts[5] == "none" {
                    let dispatch = expected[id]
                        .iter_mut()
                        .find(|event| event.starts_with("merge:"))
                        .unwrap();
                    *dispatch = "merge:false".into();
                }
                None
            }
            "finish" => Some("finish".into()),
            "close" if expected[id].last().is_none_or(|event| event != "close") => {
                Some("close".into())
            }
            "max-point-count" => {
                let row = merged_fields.iter().find(|row| row[4] == parts[3]).unwrap();
                budgets.push((
                    row[5].parse::<i32>().unwrap(),
                    parts[4].parse::<u64>().unwrap(),
                ));
                None
            }
            "merged" => {
                budgets.sort();
                expected_budgets.push(std::mem::take(&mut budgets));
                None
            }
            _ => None,
        };
        if let Some(event) = event {
            expected[id].push(event);
        }
    }
    let mut actual: Vec<_> = probe
        .writers
        .lock()
        .unwrap()
        .iter()
        .map(|events| canonical_events(events))
        .collect();
    let mut expected: Vec<_> = expected
        .iter()
        .map(|events| canonical_events(events))
        .collect();
    actual.sort();
    expected.sort();
    assert_eq!(actual, expected, "{case} callback lifecycle/multiset");
    let mut actual_budgets = probe.budgets.lock().unwrap().clone();
    for budget in &mut actual_budgets {
        budget.sort();
    }
    actual_budgets.sort();
    expected_budgets.sort();
    assert_eq!(
        actual_budgets, expected_budgets,
        "{case} unfiltered merge sizing"
    );
}

fn fixture() -> String {
    std::fs::read_to_string(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/points_writer_routing.tsv"
    ))
    .unwrap()
}

#[test]
fn explicit_points_and_pointfree_flush_append_delete_sorted_merge_match_java() {
    let fixture = fixture();
    for case in [
        "points_unsorted",
        "points_sorted",
        "empty_unsorted",
        "empty_sorted",
    ] {
        let temporary = TempDir::new("points-writer-routing");
        let directory = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let inputs = inputs(&fixture, case);
        assert_eq!(inputs.len(), 9);
        let sorted = case.ends_with("_sorted");
        {
            let mut writer = writer(&directory, probe.clone(), sorted);
            for batch in inputs[..6].chunks(3) {
                for input in batch {
                    add(&mut writer, input);
                }
                writer.commit().unwrap();
            }
            assert_snapshot(&fixture, &directory, case, "initial", false);
        }
        let mut writer = writer(&directory, probe.clone(), sorted);
        for input in &inputs[6..] {
            add(&mut writer, input);
        }
        writer.commit().unwrap();
        assert_snapshot(&fixture, &directory, case, "append", false);
        let before_delete = probe.writers.lock().unwrap().clone();
        writer
            .delete_documents_by_term(&[
                Term::new("id", "b"),
                Term::new("id", "e"),
                Term::new("id", "g"),
            ])
            .unwrap();
        writer.commit().unwrap();
        assert_snapshot(&fixture, &directory, case, "deleted", false);
        assert_eq!(
            *probe.writers.lock().unwrap(),
            before_delete,
            "deletes must not acquire points writers"
        );
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        assert_snapshot(&fixture, &directory, case, "merged", false);
        assert_callbacks(&fixture, case, &probe);
    }
}

#[test]
fn native_multileaf_merge_preserves_duplicates_empty_fields_sizing_and_range_hits() {
    let fixture = fixture();
    let case = "multi_leaf_mixed_sources";
    let temporary = TempDir::new("points-writer-multileaf");
    let directory = FsDirectory::open(&temporary);
    let probe = Arc::new(Probe::default());
    let inputs = inputs(&fixture, case);
    assert_eq!(inputs.len(), 1043);
    let mut writer = writer(&directory, probe.clone(), false);
    for (ordinal, input) in inputs.iter().enumerate() {
        add(&mut writer, input);
        if ordinal == 519 || ordinal == 1039 {
            writer.commit().unwrap();
        }
    }
    writer.commit().unwrap();
    assert_snapshot(&fixture, &directory, case, "initial", true);
    writer
        .delete_documents_by_term(&[
            Term::new("id", "p1"),
            Term::new("id", "p2"),
            Term::new("id", "p521"),
        ])
        .unwrap();
    writer.commit().unwrap();
    assert_snapshot(&fixture, &directory, case, "deleted", true);
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    assert_snapshot(&fixture, &directory, case, "merged", true);
    assert_callbacks(&fixture, case, &probe);
    // Merge the result again with a point-free segment: an absent BKD entry
    // must not erase the retained, positive-dimensional vanished FieldInfo.
    add(
        &mut writer,
        &Input {
            id: "tail".into(),
            rank: 2000,
            points: Vec::new(),
        },
    );
    writer.commit().unwrap();
    writer.force_merge(1).unwrap();
    writer.commit().unwrap();
    let reader = DirectoryReader::open(&directory).unwrap();
    assert_eq!(reader.num_docs(), 1041);
    assert_eq!(reader.segment_readers().len(), 1);
    let segment = &reader.segment_readers()[0];
    let vanished = segment
        .field_infos()
        .fields
        .iter()
        .find(|field| field.name == "vanish")
        .unwrap();
    assert_eq!(
        (
            vanished.point_dimension_count,
            vanished.point_index_dimension_count,
            vanished.point_num_bytes
        ),
        (1, 1, 4)
    );
    assert!(segment
        .points_reader()
        .unwrap()
        .field(vanished.number)
        .is_none());
    let multi = segment
        .field_infos()
        .fields
        .iter()
        .find(|field| field.name == "multi")
        .unwrap();
    assert_eq!(
        segment
            .points_reader()
            .unwrap()
            .field(multi.number)
            .unwrap()
            .point_count,
        3111
    );
    assert!(lucene_index::check_index::check_directory(&directory)
        .unwrap()
        .iter()
        .all(|result| result.all_passed()));
}

mod configured {
    use super::*;
    use lucene_codecs::{
        field_infos::{
            DocValuesSkipIndexType, DocValuesType, IndexOptions, VectorEncoding,
            VectorSimilarityFunction,
        },
        norms::{self, NormsField},
        norms_writer::{NormsWriteSink, NormsWriteState, NormsWriterFactory},
        per_field_doc_values::Lucene90DocValuesFormat,
        per_field_knn_vectors::KnnVectorsFormat,
        per_field_postings::Lucene104PostingsFormat,
        stored_fields::{self, StoredField},
        stored_fields_writer::{
            StoredFieldsWriteSink, StoredFieldsWriteState, StoredFieldsWriterFactory,
        },
    };
    use lucene_index::index_writer::DocumentVector;
    use lucene_search::{extended_query::int_bytes, query::TermQuery};
    use lucene_test_framework::codecs::{
        asserting_norms_writer::AssertingNormsCodec,
        asserting_stored_fields_writer::AssertingStoredFieldsCodec,
    };

    #[derive(Debug, Default)]
    struct Order(Arc<Mutex<Vec<&'static str>>>);

    impl Order {
        fn record(&self, event: &'static str) {
            self.0.lock().unwrap().push(event);
        }
    }

    impl Lucene104Codec for Order {
        fn points_writer_factory(&self) -> Option<&dyn PointsWriterFactory> {
            Some(self)
        }
        fn norms_writer_factory(&self) -> Option<&dyn NormsWriterFactory> {
            Some(self)
        }
        fn stored_fields_writer_factory(&self) -> Option<&dyn StoredFieldsWriterFactory> {
            Some(self)
        }

        fn postings_format_for_field(&self, _: &str) -> Lucene104PostingsFormat {
            self.record("postings");
            Lucene104PostingsFormat::default()
        }

        fn doc_values_format_for_field(&self, _: &str) -> Lucene90DocValuesFormat {
            self.record("doc-values");
            Lucene90DocValuesFormat::default()
        }

        fn knn_vectors_format_for_field(&self, _: &str) -> KnnVectorsFormat {
            self.record("vectors");
            KnnVectorsFormat::default()
        }
    }

    impl PointsWriterFactory for Order {
        fn fields_writer<'a>(
            &self,
            state: PointsWriteState<'a>,
        ) -> points::Result<Box<dyn PointsWriteSink + 'a>> {
            self.record("points-open");
            assert_eq!(state.max_doc, 3);
            Ok(Box::new(OrderedPoints {
                delegate: AssertingPointsCodec.fields_writer(state)?,
                order: self.0.clone(),
            }))
        }
    }

    struct OrderedPoints<'a> {
        delegate: Box<dyn PointsWriteSink + 'a>,
        order: Arc<Mutex<Vec<&'static str>>>,
    }

    impl PointsWriteSink for OrderedPoints<'_> {
        fn write_field(
            &mut self,
            field: &FieldInfo,
            values: PackedPointsField,
        ) -> points::Result<()> {
            self.order.lock().unwrap().push("points-field");
            assert_eq!((field.name.as_str(), field.number), ("point", 3));
            assert_eq!(
                (values.num_dims, values.num_index_dims, values.bytes_per_dim),
                (1, 1, 4)
            );
            assert_eq!(values.docs, [0, 0, 2]);
            let expected: Vec<u8> = [-3, 7, i32::MAX].into_iter().flat_map(int_bytes).collect();
            assert_eq!(values.values, expected);
            self.delegate.write_field(field, values)
        }

        fn merge(
            &mut self,
            fields: Vec<MergePointsField>,
            native_readers: bool,
        ) -> points::Result<()> {
            self.delegate.merge(fields, native_readers)
        }

        fn finish(&mut self) -> points::Result<()> {
            self.order.lock().unwrap().push("points-finish");
            self.delegate.finish()
        }

        fn close(&mut self) -> points::Result<()> {
            self.delegate.close()?;
            self.order.lock().unwrap().push("points-close");
            Ok(())
        }
    }

    impl NormsWriterFactory for Order {
        fn norms_consumer<'a>(
            &self,
            state: NormsWriteState<'a>,
        ) -> norms::Result<Box<dyn NormsWriteSink + 'a>> {
            self.record("norms-open");
            Ok(Box::new(OrderedNorms {
                delegate: AssertingNormsCodec.norms_consumer(state)?,
                order: self.0.clone(),
            }))
        }
    }

    struct OrderedNorms<'a> {
        delegate: Box<dyn NormsWriteSink + 'a>,
        order: Arc<Mutex<Vec<&'static str>>>,
    }

    impl NormsWriteSink for OrderedNorms<'_> {
        fn add_norms_field(
            &mut self,
            field: &FieldInfo,
            values: NormsField<'_>,
        ) -> norms::Result<()> {
            self.order.lock().unwrap().push("norms-field");
            self.delegate.add_norms_field(field, values)
        }

        fn close(&mut self) -> norms::Result<()> {
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

    #[test]
    fn fast_and_general_configured_flushes_install_points_between_norms_docvalues_and_other_formats(
    ) {
        for general in [false, true] {
            let temporary = TempDir::new("configured-points-order");
            let directory = FsDirectory::open(&temporary);
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
                FieldInfo::new("point", 3).with_points(1, 1, 4),
                FieldInfo::new("vector", 4).with_vectors(
                    2,
                    VectorEncoding::Float32,
                    VectorSimilarityFunction::Euclidean,
                ),
            ];
            let mut writer = IndexWriter::open(
                &directory,
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
            writer.set_max_full_flush_merge_wait_millis(0);
            writer.add_postings_field("id").unwrap();
            writer.add_postings_field("body").unwrap();
            writer.set_doc_values_field(Some("rank")).unwrap();
            writer.add_points_field("point").unwrap();
            writer.set_vector_field(Some("vector")).unwrap();
            // Term vectors force the general inversion branch. Without them,
            // these same postings and norms satisfy fast_inversion_applies.
            if general {
                writer.set_term_vector_field(Some("body")).unwrap();
            }
            for (doc, (body, rank)) in [("x x", -4), ("x", 8), ("y x x", 2)]
                .into_iter()
                .enumerate()
            {
                let mut fields = vec![
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
                ];
                let point_values: &[i32] = match doc {
                    0 => &[-3, 7],
                    1 => &[],
                    _ => &[i32::MAX],
                };
                fields.extend(point_values.iter().map(|&value| StoredField {
                    field_number: 3,
                    value: FieldValue::Int(value),
                }));
                writer
                    .add_document_with_vectors(
                        stored_fields::Document { fields },
                        vec![DocumentVector::float32("vector", vec![doc as f32, 1.0])],
                    )
                    .unwrap();
            }
            assert!(
                order.0.lock().unwrap().is_empty(),
                "selectors must run during flush"
            );
            writer.commit().unwrap();
            let events = order.0.lock().unwrap().clone();
            let index = |name| {
                events
                    .iter()
                    .position(|event| *event == name)
                    .unwrap_or_else(|| panic!("missing {name}: {events:?}"))
            };
            assert!(index("norms-close") < index("doc-values"), "{events:?}");
            assert!(index("doc-values") < index("points-open"), "{events:?}");
            assert!(index("points-open") < index("points-field"), "{events:?}");
            assert!(index("points-field") < index("points-finish"), "{events:?}");
            assert!(index("points-finish") < index("points-close"), "{events:?}");
            for next in ["postings", "vectors", "stored-open"] {
                assert!(index("points-close") < index(next), "{general}: {events:?}");
            }
            for event in [
                "points-open",
                "points-field",
                "points-finish",
                "points-close",
            ] {
                assert_eq!(
                    events.iter().filter(|actual| **actual == event).count(),
                    1,
                    "{events:?}"
                );
            }
            let reader = DirectoryReader::open(&directory).unwrap();
            assert_eq!(reader.num_docs(), 3);
            assert_eq!(reader.segment_readers().len(), 1);
            let segment = &reader.segment_readers()[0];
            segment.check_integrity().unwrap();
            assert_eq!(segment.term_vectors_reader().unwrap().is_some(), general);
            let points = segment.points_reader().unwrap();
            let meta = points.field(3).unwrap();
            assert_eq!((meta.point_count, meta.doc_count), (3, 2));
            assert_eq!(meta.min_packed_value, int_bytes(-3));
            assert_eq!(meta.max_packed_value, int_bytes(i32::MAX));
            let mut actual: Vec<_> = points
                .decode_all_points(3)
                .unwrap()
                .into_iter()
                .map(|point| (point.doc_id, point.packed_value))
                .collect();
            actual.sort();
            let mut expected = vec![
                (0, int_bytes(-3).to_vec()),
                (0, int_bytes(7).to_vec()),
                (2, int_bytes(i32::MAX).to_vec()),
            ];
            expected.sort();
            assert_eq!(actual, expected);
            let vectors = segment.flat_vectors_reader_for(4).unwrap().unwrap();
            let values = vectors.float_vector_values(4).unwrap();
            assert_eq!(values.size(), 3);
            let (meta, data) = segment.doc_values_for_field(2).unwrap();
            for (doc, (rank, length)) in [(-4, 2), (8, 1), (2, 3)].into_iter().enumerate() {
                let doc = i32::try_from(doc).unwrap();
                assert_eq!(stored_id(segment, doc), format!("d{doc}"));
                assert_eq!(
                    lucene_codecs::doc_values::numeric_value(
                        data,
                        meta.numeric_entry(2).unwrap(),
                        doc
                    )
                    .unwrap(),
                    Some(rank)
                );
                assert_eq!(
                    norms::norm_value(
                        segment.norms_data().unwrap(),
                        segment.norms_entry(1).unwrap(),
                        doc
                    )
                    .unwrap(),
                    Some(length)
                );
                assert_eq!(values.ord_to_doc(doc).unwrap(), doc);
                assert_eq!(values.vector(doc).unwrap(), [doc as f32, 1.0]);
            }
            let mut opened = reader.open_segments().unwrap();
            opened.open_points().unwrap();
            let segments = opened.as_open_segments();
            let owned = reader.field_norms_by_field(&["body".into()]).unwrap();
            let norms: Vec<_> = owned.iter().map(Some).collect();
            let searcher = IndexSearcher::new(&segments, &norms).unwrap();
            let mut query = BooleanQuery::new();
            query.filter.push(
                PointRangeQuery::int_range("point", &[0], &[10])
                    .unwrap()
                    .into(),
            );
            let hits = searcher.search(&query, 3).unwrap();
            assert_eq!(hits.score_docs.len(), 1);
            assert_eq!(hits.score_docs[0].doc, 0);
            let mut text = BooleanQuery::new();
            text.should.push(TermQuery::new("body", "x").into());
            assert_eq!(searcher.search(&text, 3).unwrap().score_docs.len(), 3);
            assert!(lucene_index::check_index::check_directory(&directory)
                .unwrap()
                .iter()
                .all(|result| result.all_passed()));
        }
    }

    #[test]
    fn configured_point_shape_without_points_optin_preserves_stored_field_identity() {
        let temporary = TempDir::new("configured-point-shape-stored-only");
        let directory = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let fields = vec![
            FieldInfo::new("id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
            FieldInfo::new("p", 1).with_points(1, 1, 4),
        ];
        let mut writer = IndexWriter::open(
            &directory,
            fields,
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        writer
            .set_codec(Some(Arc::new(Codec(probe.clone()))))
            .unwrap();
        writer.set_use_compound_file(false);
        writer.add_postings_field("id").unwrap();
        // The configured facade stores every supplied field. Declaring point
        // dimensions does not opt this field into points indexing.
        writer
            .add_document(stored_fields::Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("stored".into()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(7),
                    },
                ],
            })
            .unwrap();
        writer.commit().unwrap();
        assert!(probe.writers.lock().unwrap().is_empty());
        let reader = DirectoryReader::open(&directory).unwrap();
        assert_eq!(reader.num_docs(), 1);
        let segment = &reader.segment_readers()[0];
        let info = segment
            .field_infos()
            .fields
            .iter()
            .find(|field| field.name == "p")
            .expect("stored values must retain their FieldInfo");
        assert_eq!(info.number, 1);
        assert_eq!(
            (
                info.point_dimension_count,
                info.point_index_dimension_count,
                info.point_num_bytes
            ),
            (0, 0, 0)
        );
        assert_eq!(segment.points_reader().unwrap().field_numbers().count(), 0);
        let document = segment.stored_document(0).unwrap().unwrap();
        assert!(document
            .fields
            .iter()
            .any(|field| field.field_number == info.number && field.value == FieldValue::Int(7)));
        assert_eq!(stored_id(segment, 0), "stored");
        segment.check_integrity().unwrap();
        assert!(lucene_index::check_index::check_directory(&directory)
            .unwrap()
            .iter()
            .all(|result| result.all_passed()));
    }

    #[test]
    fn configured_numeric_docvalues_and_points_merge_an_initially_absent_field() {
        let temporary = TempDir::new("configured-sparse-docvalues-points");
        let directory = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let fields = vec![
            FieldInfo::new("id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
            FieldInfo::new("p", 1).with_points(1, 1, 4).with_doc_values(
                DocValuesType::Numeric,
                DocValuesSkipIndexType::None,
                -1,
            ),
            FieldInfo::new("rank", 2).with_doc_values(
                DocValuesType::Numeric,
                DocValuesSkipIndexType::None,
                -1,
            ),
        ];
        let mut writer = IndexWriter::open(
            &directory,
            fields,
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        writer
            .set_codec(Some(Arc::new(Codec(probe.clone()))))
            .unwrap();
        writer.set_use_compound_file(false);
        writer.set_max_full_flush_merge_wait_millis(0);
        let mut policy = LogDocMergePolicy::new();
        policy.set_merge_factor(1000).unwrap();
        writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
        writer.add_postings_field("id").unwrap();
        writer.add_points_field("p").unwrap();
        // Multiple numeric fields use collect_dense_column, whose sparse
        // variant accepts an entirely empty column. The one-field numeric
        // facade intentionally rejects an all-missing flush.
        writer.set_doc_values_field(Some("rank")).unwrap();
        writer.add_doc_values_field("p").unwrap();
        writer
            .add_document(stored_fields::Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("missing".into()),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::Long(11),
                    },
                ],
            })
            .unwrap();
        writer.commit().unwrap();
        {
            let events = probe.writers.lock().unwrap();
            assert_eq!(events.len(), 1);
            assert_eq!(canonical_events(&events[0]), ["open:1", "finish", "close"]);
        }
        let first = DirectoryReader::open(&directory).unwrap();
        let source = &first.segment_readers()[0];
        // This configured field also has a doc-values role. Its retained
        // point shape requires an empty native triple, even without a BKD
        // field entry. A nonnull native reader still selects native merging.
        let info = source
            .field_infos()
            .fields
            .iter()
            .find(|field| field.name == "p")
            .unwrap();
        assert_eq!(
            (
                info.point_dimension_count,
                info.point_index_dimension_count,
                info.point_num_bytes
            ),
            (1, 1, 4)
        );
        assert_eq!(source.points_reader().unwrap().field_numbers().count(), 0);
        assert_eq!(
            directory
                .list_all()
                .unwrap()
                .iter()
                .filter(|name| {
                    name.ends_with(".kdm") || name.ends_with(".kdi") || name.ends_with(".kdd")
                })
                .count(),
            3
        );
        let (meta, data) = source.doc_values_for_field(1).unwrap();
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(data, meta.numeric_entry(1).unwrap(), 0)
                .unwrap(),
            None
        );
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(data, meta.numeric_entry(2).unwrap(), 0)
                .unwrap(),
            Some(11)
        );
        writer
            .add_document(stored_fields::Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("present".into()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(7),
                    },
                    StoredField {
                        field_number: 2,
                        value: FieldValue::Long(22),
                    },
                ],
            })
            .unwrap();
        writer.commit().unwrap();
        assert_eq!(
            DirectoryReader::open(&directory)
                .unwrap()
                .segment_readers()
                .len(),
            2
        );
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        let reader = DirectoryReader::open(&directory).unwrap();
        assert_eq!(reader.num_docs(), 2);
        assert_eq!(reader.segment_readers().len(), 1);
        let segment = &reader.segment_readers()[0];
        assert_eq!(stored_id(segment, 0), "missing");
        assert_eq!(stored_id(segment, 1), "present");
        let info = segment
            .field_infos()
            .fields
            .iter()
            .find(|field| field.name == "p")
            .unwrap();
        assert_eq!(info.number, 1);
        assert_eq!(info.doc_values_type, DocValuesType::Numeric);
        assert_eq!(
            (
                info.point_dimension_count,
                info.point_index_dimension_count,
                info.point_num_bytes
            ),
            (1, 1, 4)
        );
        let points = segment.points_reader().unwrap();
        let meta = points.field(1).unwrap();
        assert_eq!((meta.point_count, meta.doc_count), (1, 1));
        let actual = points.decode_all_points(1).unwrap();
        assert_eq!(actual.len(), 1);
        assert_eq!(actual[0].doc_id, 1);
        assert_eq!(actual[0].packed_value, int_bytes(7));
        let (meta, data) = segment.doc_values_for_field(1).unwrap();
        let entry = meta.numeric_entry(1).unwrap();
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(data, entry, 0).unwrap(),
            None
        );
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(data, entry, 1).unwrap(),
            Some(7)
        );
        let (meta, data) = segment.doc_values_for_field(2).unwrap();
        let entry = meta.numeric_entry(2).unwrap();
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(data, entry, 0).unwrap(),
            Some(11)
        );
        assert_eq!(
            lucene_codecs::doc_values::numeric_value(data, entry, 1).unwrap(),
            Some(22)
        );
        let events = probe.writers.lock().unwrap();
        assert_eq!(events.len(), 3);
        assert_eq!(canonical_events(&events[0]), ["open:1", "finish", "close"]);
        assert_eq!(
            canonical_events(&events[1]),
            ["open:1", "write:p:1:1:1:4", "finish", "close"]
        );
        assert_eq!(
            canonical_events(&events[2]),
            ["open:2", "merge:true", "close"]
        );
        assert_eq!(*probe.budgets.lock().unwrap(), [vec![(1, 1)]]);
        segment.check_integrity().unwrap();
        assert!(lucene_index::check_index::check_directory(&directory)
            .unwrap()
            .iter()
            .all(|result| result.all_passed()));
    }

    #[test]
    fn pointfree_configured_source_keeps_generic_merge_dispatch_and_sparse_points() {
        let temporary = TempDir::new("configured-points-sparse-source");
        let directory = FsDirectory::open(&temporary);
        let probe = Arc::new(Probe::default());
        let fields = vec![
            FieldInfo::new("id", 0)
                .with_index_options(IndexOptions::Docs)
                .with_omit_norms(true),
            FieldInfo::new("point", 1).with_points(1, 1, 4),
        ];
        let mut writer = IndexWriter::open(
            &directory,
            fields,
            "Lucene104",
            LuceneVersion {
                major: 10,
                minor: 5,
                bugfix: 0,
            },
        )
        .unwrap();
        writer
            .set_codec(Some(Arc::new(Codec(probe.clone()))))
            .unwrap();
        writer.set_use_compound_file(false);
        writer.set_max_full_flush_merge_wait_millis(0);
        let mut policy = LogDocMergePolicy::new();
        policy.set_merge_factor(1000).unwrap();
        writer.set_pluggable_merge_policy(Some(Arc::new(policy)));
        writer.add_postings_field("id").unwrap();
        writer.add_points_field("point").unwrap();
        writer
            .add_document(stored_fields::Document {
                fields: vec![StoredField {
                    field_number: 0,
                    value: FieldValue::String("empty".into()),
                }],
            })
            .unwrap();
        writer.commit().unwrap();
        assert!(
            probe.writers.lock().unwrap().is_empty(),
            "a configured but unused point field must not open a writer"
        );
        let source = DirectoryReader::open(&directory).unwrap();
        assert!(source.segment_readers()[0]
            .field_infos()
            .fields
            .iter()
            .all(|field| field.name != "point"));
        assert_eq!(
            source.segment_readers()[0]
                .points_reader()
                .unwrap()
                .field_numbers()
                .count(),
            0
        );
        writer
            .add_document(stored_fields::Document {
                fields: vec![
                    StoredField {
                        field_number: 0,
                        value: FieldValue::String("pointful".into()),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(7),
                    },
                    StoredField {
                        field_number: 1,
                        value: FieldValue::Int(7),
                    },
                ],
            })
            .unwrap();
        writer.commit().unwrap();
        let before = DirectoryReader::open(&directory).unwrap();
        assert_eq!(before.segment_readers().len(), 2);
        writer.force_merge(1).unwrap();
        writer.commit().unwrap();
        let events = probe.writers.lock().unwrap();
        assert_eq!(events.len(), 2);
        assert_eq!(
            canonical_events(&events[0]),
            ["open:1", "write:point:1:1:1:4", "finish", "close"]
        );
        assert_eq!(
            canonical_events(&events[1]),
            ["open:2", "merge:false", "close"]
        );
        drop(events);
        assert_eq!(*probe.budgets.lock().unwrap(), [vec![(1, 2)]]);
        let reader = DirectoryReader::open(&directory).unwrap();
        assert_eq!(reader.segment_readers().len(), 1);
        assert_eq!(reader.num_docs(), 2);
        let segment = &reader.segment_readers()[0];
        assert_eq!(stored_id(segment, 0), "empty");
        assert_eq!(stored_id(segment, 1), "pointful");
        let points = segment.points_reader().unwrap();
        let meta = points.field(1).unwrap();
        assert_eq!((meta.point_count, meta.doc_count), (2, 1));
        let actual = points.decode_all_points(1).unwrap();
        assert_eq!(actual.len(), 2);
        assert!(actual
            .iter()
            .all(|point| point.doc_id == 1 && point.packed_value == int_bytes(7)));
        let mut opened = reader.open_segments().unwrap();
        opened.open_points().unwrap();
        let segments = opened.as_open_segments();
        let norms = vec![None; segments.len()];
        let searcher = IndexSearcher::new(&segments, &norms).unwrap();
        let mut query = BooleanQuery::new();
        query.filter.push(
            PointRangeQuery::int_range("point", &[7], &[7])
                .unwrap()
                .into(),
        );
        let hits = searcher.search(&query, 2).unwrap();
        assert_eq!(hits.score_docs.len(), 1);
        assert_eq!(hits.score_docs[0].doc, 1);
        assert!(lucene_index::check_index::check_directory(&directory)
            .unwrap()
            .iter()
            .all(|result| result.all_passed()));
    }
}
