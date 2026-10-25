use std::borrow::Cow;
use std::cell::RefCell;

use lucene_codecs::doc_values::{
    self as dv, BinaryColumn, SortedNumericColumn, SortedOrdsColumn, SortedSetOrdsColumn,
};
use lucene_codecs::field_infos::{
    self, DocValuesSkipIndexType, DocValuesType, FieldInfo, FieldInfos,
};
use lucene_codecs::segment_state::{SegmentReadState, SegmentWriteState};
use lucene_test_framework::codecs::cranky_doc_values::*;
use lucene_util::java_random::JavaRandom;
use std::rc::Rc;
const ID: [u8; 16] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15];
fn data(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data")
            .join(name),
    )
    .unwrap()
}
fn files(prefix: &str) -> Files {
    ["dvm", "dvd", "dvs"]
        .into_iter()
        .map(|extension| (extension.to_owned(), data(&format!("{prefix}.{extension}"))))
        .collect()
}
fn write<'a>(infos: &'a FieldInfos, max_doc: i32) -> SegmentWriteState<'a> {
    SegmentWriteState {
        max_doc,
        segment_id: &ID,
        segment_suffix: "s",
        field_infos: infos,
    }
}
fn read<'a>(infos: &'a FieldInfos, max_doc: i32, files: &'a Files) -> ReadState<'a> {
    ReadState {
        files,
        segment: SegmentReadState {
            max_doc,
            segment_id: &ID,
            segment_suffix: "s",
            field_infos: infos,
        },
    }
}
fn add<C: DocValuesConsumer>(
    consumer: &mut C,
    field: &FieldInfo,
    producer: &dyn DocValuesProducer,
) -> Result<()> {
    match field.doc_values_type {
        DocValuesType::Numeric => consumer.add_numeric_field(field, producer),
        DocValuesType::Binary => consumer.add_binary_field(field, producer),
        DocValuesType::Sorted => consumer.add_sorted_field(field, producer),
        DocValuesType::SortedNumeric => consumer.add_sorted_numeric_field(field, producer),
        DocValuesType::SortedSet => consumer.add_sorted_set_field(field, producer),
        DocValuesType::None => panic!(),
    }
}
#[derive(Default)]
struct Events {
    fail: bool,
    events: Vec<String>,
}
fn event(state: &RefCell<Events>, value: impl Into<String>) -> Result<()> {
    let mut state = state.borrow_mut();
    state.events.push(value.into());
    if state.fail {
        Err(Error::Message("delegate failure".into()))
    } else {
        Ok(())
    }
}
struct RecordingProducer(Rc<RefCell<Events>>);
impl DocValuesProducer for RecordingProducer {
    fn get_numeric(&self, f: &FieldInfo) -> Result<Cow<'_, NumericColumn>> {
        event(&self.0, format!("getNumeric:{}", f.name))?;
        Ok(Cow::Owned(Vec::new()))
    }
    fn get_binary(&self, f: &FieldInfo) -> Result<Cow<'_, BinaryColumn>> {
        event(&self.0, format!("getBinary:{}", f.name))?;
        Ok(Cow::Owned(BinaryColumn::default()))
    }
    fn get_sorted(&self, f: &FieldInfo) -> Result<Cow<'_, SortedOrdsColumn>> {
        event(&self.0, format!("getSorted:{}", f.name))?;
        Ok(Cow::Owned(SortedOrdsColumn::default()))
    }
    fn get_sorted_numeric(&self, f: &FieldInfo) -> Result<Cow<'_, SortedNumericColumn>> {
        event(&self.0, format!("getSortedNumeric:{}", f.name))?;
        Ok(Cow::Owned(SortedNumericColumn::default()))
    }
    fn get_sorted_set(&self, f: &FieldInfo) -> Result<Cow<'_, SortedSetOrdsColumn>> {
        event(&self.0, format!("getSortedSet:{}", f.name))?;
        Ok(Cow::Owned(SortedSetOrdsColumn::default()))
    }
    fn check_integrity(&self) -> Result<()> {
        event(&self.0, "integrity")
    }
    fn close(&mut self) -> Result<()> {
        event(&self.0, "producerClose")
    }
}
struct ProducerHandle(Rc<RecordingProducer>);
impl DocValuesProducer for ProducerHandle {
    fn check_integrity(&self) -> Result<()> {
        self.0.check_integrity()
    }
    fn close(&mut self) -> Result<()> {
        event(&self.0 .0, "producerClose")
    }
}
struct RecordingConsumer {
    state: Rc<RefCell<Events>>,
    infos: Rc<FieldInfos>,
    producer: Rc<RecordingProducer>,
}
impl RecordingConsumer {
    fn record(
        &self,
        kind: &str,
        field: &FieldInfo,
        producer: &dyn DocValuesProducer,
    ) -> Result<()> {
        event(
            &self.state,
            format!(
                "{kind}:{}:{}",
                std::ptr::eq(field, self.infos.field_by_number(field.number).unwrap()),
                std::ptr::eq(producer, self.producer.as_ref() as &dyn DocValuesProducer)
            ),
        )
    }
}
impl DocValuesConsumer for RecordingConsumer {
    fn add_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.record("numeric", f, p)
    }
    fn add_binary_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.record("binary", f, p)
    }
    fn add_sorted_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.record("sorted", f, p)
    }
    fn add_sorted_numeric_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.record("sortedNumeric", f, p)
    }
    fn add_sorted_set_field(&mut self, f: &FieldInfo, p: &dyn DocValuesProducer) -> Result<()> {
        self.record("sortedSet", f, p)
    }
    fn close(&mut self) -> Result<()> {
        event(&self.state, "consumerClose")
    }
    fn merge(&mut self, _: &MergeState<'_>) -> Result<()> {
        panic!("cranky must inherit merge")
    }
}
struct RecordingFormat {
    state: Rc<RefCell<Events>>,
    infos: Rc<FieldInfos>,
    producer: Rc<RecordingProducer>,
}
impl DocValuesFormat for RecordingFormat {
    type Consumer<'a> = RecordingConsumer;
    type Producer = ProducerHandle;
    fn name(&self) -> &str {
        "RecordingDocValues"
    }
    fn fields_consumer<'a>(&mut self, _: SegmentWriteState<'a>) -> Result<Self::Consumer<'a>> {
        event(&self.state, "consumer")?;
        Ok(RecordingConsumer {
            state: self.state.clone(),
            infos: self.infos.clone(),
            producer: self.producer.clone(),
        })
    }
    fn fields_producer(&mut self, _: ReadState<'_>) -> Result<Self::Producer> {
        event(&self.state, "producer")?;
        Ok(ProducerHandle(self.producer.clone()))
    }
}
fn trace_infos() -> FieldInfos {
    FieldInfos::new(
        [
            ("n", DocValuesType::Numeric),
            ("b", DocValuesType::Binary),
            ("s", DocValuesType::Sorted),
            ("sn", DocValuesType::SortedNumeric),
            ("ss", DocValuesType::SortedSet),
        ]
        .into_iter()
        .enumerate()
        .map(|(i, (name, kind))| {
            FieldInfo::new(name, i32::try_from(i).unwrap()).with_doc_values(
                kind,
                DocValuesSkipIndexType::None,
                -1,
            )
        })
        .collect(),
    )
    .unwrap()
}
#[test]
fn actual_java_callback_error_rng_trace() {
    let fixture = String::from_utf8(data("cranky_doc_values.tsv")).unwrap();
    let mut lines = fixture.lines();
    for seed in [0, 1, 42, -1, i64::MIN, i64::MAX] {
        let infos = Rc::new(trace_infos());
        let state = Rc::new(RefCell::new(Events::default()));
        let producer = Rc::new(RecordingProducer(state.clone()));
        let random = Rc::new(RefCell::new(JavaRandom::new(seed)));
        let mut format = CrankyDocValuesFormat::new(
            RecordingFormat {
                state: state.clone(),
                infos: infos.clone(),
                producer: producer.clone(),
            },
            random.clone(),
        );
        let mut consumer = loop {
            if let Ok(consumer) = format.fields_consumer(write(&infos, 4)) {
                break consumer;
            }
        };
        let empty = FieldInfos::new(Vec::new()).unwrap();
        let sources = [
            MergeSource {
                producer: Some(producer.as_ref()),
                field_infos: &infos,
                max_doc: 2,
                doc_map: &[0, 1],
                live_docs: None,
            },
            MergeSource {
                producer: None,
                field_infos: &empty,
                max_doc: 0,
                doc_map: &[],
                live_docs: None,
            },
            MergeSource {
                producer: Some(producer.as_ref()),
                field_infos: &infos,
                max_doc: 2,
                doc_map: &[2, 3],
                live_docs: None,
            },
        ];
        let merge = MergeState {
            sources: &sources,
            field_infos: &infos,
            check_aborted: &|| Ok(()),
        };
        let empty_files = Files::new();
        for step in 0..2400 {
            state.borrow_mut().events.clear();
            state.borrow_mut().fail = step % 11 == 0;
            let op = step % 14;
            let mut status = "ok".to_owned();
            let result = match op {
                0 => {
                    status = format.name().to_owned();
                    Ok(())
                }
                1 => format.fields_consumer(write(&infos, 4)).map(|_| ()),
                2..=6 => add(&mut consumer, &infos.fields[op - 2], producer.as_ref()),
                7 => consumer.close(),
                8 => format
                    .fields_producer(read(&infos, 4, &empty_files))
                    .map(|actual| status = Rc::ptr_eq(&actual.0, &producer).to_string()),
                9 => producer.check_integrity(),
                10 => event(&state, "producerClose"),
                11 => consumer.merge(&merge),
                12 => consumer.merge_sorted_field(&infos.fields[2], &merge),
                13 => consumer.merge_sorted_set_field(&infos.fields[4], &merge),
                _ => unreachable!(),
            };
            if let Err(error) = result {
                status = error.to_string();
            }
            let actual = format!(
                "{seed}\t{step}\t{op}\t{status}\t{}\t{}",
                state.borrow().events.join(","),
                random.borrow_mut().next_int()
            );
            assert_eq!(actual, lines.next().unwrap(), "seed {seed} step {step}");
        }
    }
    assert!(lines.next().is_none());
}
fn assert_snapshot(
    producer: &OwnedDocValuesProducer,
    infos: &FieldInfos,
    max_doc: i32,
    prefix: &str,
) {
    let fixture = String::from_utf8(data(&format!("{prefix}.tsv"))).unwrap();
    let mut actual = format!("maxDoc\t{max_doc}\n");
    for field in &infos.fields {
        let kind = match field.doc_values_type {
            DocValuesType::Numeric => "NUMERIC",
            DocValuesType::Binary => "BINARY",
            DocValuesType::Sorted => "SORTED",
            DocValuesType::SortedNumeric => "SORTED_NUMERIC",
            DocValuesType::SortedSet => "SORTED_SET",
            DocValuesType::None => continue,
        };
        actual.push_str(&format!(
            "field\t{}\t{}\t{kind}\t{}\n",
            field.number,
            field.name,
            if field.doc_values_skip_index_type == DocValuesSkipIndexType::None {
                "NONE"
            } else {
                "RANGE"
            }
        ));
        let n = field.number;
        match producer.columns.get(&n).unwrap() {
            Column::Numeric(c) => {
                for (doc, value) in c {
                    actual.push_str(&format!("numeric\t{n}\t{doc}\t{value}\n"));
                }
            }
            Column::Binary(c) => {
                let mut offset = 0;
                for (&doc, &len) in c.docs.iter().zip(&c.lengths) {
                    let end = offset + usize::try_from(len).unwrap();
                    actual.push_str(&format!(
                        "binary\t{n}\t{doc}\t{}\n",
                        hex(&c.bytes[offset..end])
                    ));
                    offset = end;
                }
            }
            Column::Sorted(c) => {
                terms(&mut actual, n, &c.dict);
                for (&doc, &ord) in c.docs.iter().zip(&c.ords) {
                    actual.push_str(&format!("sorted\t{n}\t{doc}\t{ord}\n"));
                }
            }
            Column::SortedNumeric(c) => multi(
                &mut actual,
                "sortedNumeric",
                n,
                &c.docs,
                &c.counts,
                &c.values,
            ),
            Column::SortedSet(c) => {
                terms(&mut actual, n, &c.dict);
                multi(&mut actual, "sortedSet", n, &c.docs, &c.counts, &c.ords);
            }
        }
        if let Some(skip) = producer.get_skipper(field).unwrap() {
            actual.push_str(&format!(
                "skipGlobal\t{n}\t{}\t{}\t{}\t{}\n",
                skip.min_value, skip.max_value, skip.doc_count, skip.max_value_count
            ));
            for interval in &skip.intervals {
                for (level, entry) in interval.levels.iter().enumerate() {
                    actual.push_str(&format!(
                        "skip\t{n}\t{level}\t{}\t{}\t{}\t{}\t{}\n",
                        entry.min_doc_id,
                        entry.max_doc_id,
                        entry.min_value,
                        entry.max_value,
                        entry.doc_count
                    ));
                }
            }
        }
    }
    let expected: Vec<_> = fixture
        .lines()
        .filter(|line| !line.starts_with("rng\t"))
        .collect();
    assert_eq!(actual.lines().collect::<Vec<_>>(), expected, "{prefix}");
}
fn hex(bytes: &[u8]) -> String {
    if bytes.is_empty() {
        "-".into()
    } else {
        bytes.iter().map(|byte| format!("{byte:02x}")).collect()
    }
}
fn terms(out: &mut String, number: i32, dict: &[Vec<u8>]) {
    for (ord, term) in dict.iter().enumerate() {
        out.push_str(&format!("term\t{number}\t{ord}\t{}\n", hex(term)));
    }
}
fn multi(out: &mut String, kind: &str, n: i32, docs: &[i32], counts: &[u32], values: &[i64]) {
    let mut offset = 0;
    for (&doc, &count) in docs.iter().zip(counts) {
        out.push_str(&format!("{kind}\t{n}\t{doc}"));
        for value in &values[offset..offset + usize::try_from(count).unwrap()] {
            out.push_str(&format!("\t{value}"));
        }
        out.push('\n');
        offset += usize::try_from(count).unwrap();
    }
}
#[test]
fn native_and_cranky_reproduce_all_three_java_files_and_decoded_values() {
    let infos = field_infos::parse(&data("cranky_doc_values.fnm"), &ID, "").unwrap();
    for prefix in ["cranky_doc_values_left", "cranky_doc_values_right"] {
        let expected = files(prefix);
        let mut native = Lucene90DocValuesFormat(dv::Lucene90DocValuesFormat::new(4).unwrap());
        let producer = native.fields_producer(read(&infos, 12, &expected)).unwrap();
        producer.check_integrity().unwrap();
        assert_snapshot(&producer, &infos, 12, prefix);
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut cranky = CrankyDocValuesFormat::new(native, random.clone());
        let mut consumer = cranky.fields_consumer(write(&infos, 12)).unwrap();
        for field in &infos.fields {
            add(&mut consumer, field, &producer).unwrap();
        }
        consumer.close().unwrap();
        assert_eq!(consumer.delegate().output().unwrap(), &expected);
        assert_eq!(random.borrow_mut().next_int(), 1583910553);
    }
}
#[test]
fn inherited_sequential_and_sorted_merge_prune_deleted_dictionaries_and_preserve_empty_columns() {
    let infos = field_infos::parse(&data("cranky_doc_values.fnm"), &ID, "").unwrap();
    let mut native = Lucene90DocValuesFormat(dv::Lucene90DocValuesFormat::new(4).unwrap());
    let left = native
        .fields_producer(read(&infos, 12, &files("cranky_doc_values_left")))
        .unwrap();
    let right = native
        .fields_producer(read(&infos, 12, &files("cranky_doc_values_right")))
        .unwrap();
    let live: Vec<_> = (0..12).map(|doc| doc != 3 && doc != 10).collect();
    for sorted in [false, true] {
        let maps: Vec<Vec<i32>> = (0..2)
            .map(|side| {
                (0..12)
                    .map(|doc| {
                        if doc == 3 || doc == 10 {
                            -1
                        } else {
                            let at = doc - i32::from(doc > 3) - i32::from(doc > 10);
                            if sorted {
                                2 * at + side
                            } else {
                                10 * side + at
                            }
                        }
                    })
                    .collect()
            })
            .collect();
        let sources = [
            MergeSource {
                producer: Some(&left),
                field_infos: &infos,
                max_doc: 12,
                doc_map: &maps[0],
                live_docs: Some(&live),
            },
            MergeSource {
                producer: Some(&right),
                field_infos: &infos,
                max_doc: 12,
                doc_map: &maps[1],
                live_docs: Some(&live),
            },
        ];
        let state = MergeState {
            sources: &sources,
            field_infos: &infos,
            check_aborted: &|| Ok(()),
        };
        let random = Rc::new(RefCell::new(JavaRandom::new(42)));
        let mut cranky = CrankyDocValuesFormat::new(native, random.clone());
        let mut consumer = cranky.fields_consumer(write(&infos, 20)).unwrap();
        consumer.merge(&state).unwrap();
        consumer.close().unwrap();
        let prefix = format!(
            "cranky_doc_values_merge_{}",
            if sorted { "sorted" } else { "sequential" }
        );
        assert_eq!(consumer.delegate().output().unwrap(), &files(&prefix));
        assert_eq!(random.borrow_mut().next_int(), 1583910553);
        let producer = native
            .fields_producer(read(&infos, 20, consumer.delegate().output().unwrap()))
            .unwrap();
        assert_snapshot(&producer, &infos, 20, &prefix);
    }
}
#[test]
fn empty_native_close_matches_actual_java() {
    let infos = FieldInfos::new(Vec::new()).unwrap();
    let random = Rc::new(RefCell::new(JavaRandom::new(42)));
    let mut format = CrankyDocValuesFormat::new(Lucene90DocValuesFormat::default(), random.clone());
    let mut consumer = format.fields_consumer(write(&infos, 0)).unwrap();
    consumer.close().unwrap();
    assert_eq!(
        consumer.delegate().output().unwrap(),
        &files("cranky_doc_values_empty")
    );
    let producer = format
        .fields_producer(read(&infos, 0, consumer.delegate().output().unwrap()))
        .unwrap();
    producer.check_integrity().unwrap();
    assert_snapshot(&producer, &infos, 0, "cranky_doc_values_empty");
    let fixture = String::from_utf8(data("cranky_doc_values_empty.tsv")).unwrap();
    assert_eq!(
        fixture.lines().last().unwrap(),
        format!("rng\t{}", random.borrow_mut().next_int())
    );
}
