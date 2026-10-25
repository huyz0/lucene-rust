//! Callback ordering and production norms bytes from actual Java asserting wrappers.
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Mutex};

use lucene_codecs::field_infos::{FieldInfo, FieldInfos, IndexOptions};
use lucene_search::reader::{DocIdSetIterator, DocValuesIterator, NumericDocValues, NO_MORE_DOCS};
use lucene_search::Result;
use lucene_store::{ByteBuffersDirectory, DataOutput, Directory};
use lucene_test_framework::codecs::asserting_norms::{
    AssertingNormsConsumer, AssertingNormsFormat, AssertingNormsProducer, Lucene90NormsFormat,
    NormsConsumer, NormsFormat, NormsProducer,
};
use lucene_test_framework::codecs::cranky_norms::NormsState;

fn field(name: &str, number: i32, norms: bool) -> FieldInfo {
    let mut field = FieldInfo::new(name, number);
    field.index_options = IndexOptions::Docs;
    field.omit_norms = !norms;
    field
}
fn failure(message: &str) -> lucene_search::Error {
    lucene_store::Error::Io(std::io::Error::other(message.to_owned())).into()
}
#[derive(Default)]
struct State {
    events: Vec<String>,
    docs: Vec<i32>,
    values: Vec<i64>,
    initial_doc: i32,
    fail_get: bool,
    fail_next: bool,
    fail_integrity: bool,
    fail_add: bool,
    fail_open: bool,
    producer_closes: usize,
    consumer_closes: usize,
    fail_producer_close: usize,
    fail_consumer_close: usize,
}
#[derive(Clone)]
struct Producer(Arc<Mutex<State>>);
struct Values {
    state: Arc<Mutex<State>>,
    docs: Vec<i32>,
    values: Vec<i64>,
    position: Option<usize>,
    doc: i32,
    fail_next: bool,
}
impl DocIdSetIterator for Values {
    fn doc_id(&self) -> i32 {
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        if self.fail_next {
            self.state.lock().unwrap().events.push("next-error".into());
            return Err(failure("next failure"));
        }
        let position = self.position.map_or(0, |p| p + 1);
        self.position = Some(position);
        self.doc = self.docs.get(position).copied().unwrap_or(NO_MORE_DOCS);
        self.state
            .lock()
            .unwrap()
            .events
            .push(format!("next:{}", self.doc));
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        while self.next_doc()? < target {}
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        self.docs.len() as i64
    }
}
impl DocValuesIterator for Values {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        let mut next = self.position.map_or(0, |p| p + 1);
        while next < self.docs.len() && self.docs[next] < target {
            self.position = Some(next);
            next += 1;
        }
        let mut exists = self
            .position
            .is_some_and(|p| self.docs.get(p) == Some(&target));
        if !exists && self.docs.get(next) == Some(&target) {
            self.position = Some(next);
            exists = true;
        }
        self.doc = target;
        self.state
            .lock()
            .unwrap()
            .events
            .push(format!("exact:{target}:{exists}"));
        Ok(exists)
    }
}
impl NumericDocValues for Values {
    fn long_value(&self) -> i64 {
        let value = self.values[self.position.unwrap()];
        self.state
            .lock()
            .unwrap()
            .events
            .push(format!("value:{value}"));
        value
    }
}
impl NormsProducer for Producer {
    fn get_norms(&self, _: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        let mut state = self.0.lock().unwrap();
        state.events.push("get".into());
        if state.fail_get {
            return Err(failure("get failure"));
        }
        Ok(Box::new(Values {
            state: self.0.clone(),
            docs: state.docs.clone(),
            values: state.values.clone(),
            position: None,
            doc: state.initial_doc,
            fail_next: state.fail_next,
        }))
    }
    fn close(&mut self) -> Result<()> {
        let mut state = self.0.lock().unwrap();
        state.producer_closes += 1;
        let closes = state.producer_closes;
        state.events.push(format!("pclose:{closes}"));
        if closes == state.fail_producer_close {
            Err(failure("producer close failure"))
        } else {
            Ok(())
        }
    }
    fn check_integrity(&self) -> Result<()> {
        let mut state = self.0.lock().unwrap();
        state.events.push("integrity".into());
        if state.fail_integrity {
            Err(failure("integrity failure"))
        } else {
            Ok(())
        }
    }
    fn get_merge_instance(&self) -> Self {
        self.0.lock().unwrap().events.push("merge".into());
        self.clone()
    }
}
struct Consumer {
    state: Arc<Mutex<State>>,
    expected: usize,
}
impl NormsConsumer for Consumer {
    fn add_norms_field(&mut self, _: &FieldInfo, producer: &dyn NormsProducer) -> Result<()> {
        assert_eq!(
            producer as *const dyn NormsProducer as *const () as usize,
            self.expected
        );
        let mut state = self.state.lock().unwrap();
        state.events.push("delegate-add".into());
        if state.fail_add {
            Err(failure("add failure"))
        } else {
            Ok(())
        }
    }
    fn close(&mut self) -> Result<()> {
        let mut state = self.state.lock().unwrap();
        state.consumer_closes += 1;
        let closes = state.consumer_closes;
        state.events.push(format!("cclose:{closes}"));
        if closes == state.fail_consumer_close {
            Err(failure("consumer close failure"))
        } else {
            Ok(())
        }
    }
}
struct Format(Arc<Mutex<State>>);
impl NormsFormat for Format {
    type Consumer = Consumer;
    type Producer = Producer;
    fn norms_consumer(&mut self, _: &NormsState<'_>) -> Result<Consumer> {
        let mut state = self.0.lock().unwrap();
        state.events.push("open-consumer".into());
        if state.fail_open {
            Err(failure("open failure"))
        } else {
            Ok(Consumer {
                state: self.0.clone(),
                expected: 0,
            })
        }
    }
    fn norms_producer(&mut self, _: &NormsState<'_>, _: &FieldInfos) -> Result<Producer> {
        let mut state = self.0.lock().unwrap();
        state.events.push("open-producer".into());
        if state.fail_open {
            Err(failure("open failure"))
        } else {
            Ok(Producer(self.0.clone()))
        }
    }
}

fn replay(kind: &str) -> (String, String, String) {
    let state = Arc::new(Mutex::new(State {
        docs: vec![0, 2, 4],
        values: vec![10, 20, 30],
        initial_doc: -1,
        ..State::default()
    }));
    let producer = Producer(state.clone());
    let consumer = Consumer {
        state: state.clone(),
        expected: &producer as *const Producer as usize,
    };
    {
        let mut s = state.lock().unwrap();
        match kind {
            "consumer_duplicate" => s.docs = vec![0, 0, 4],
            "consumer_backwards" => s.docs = vec![2, 1, 4],
            "consumer_negative" => s.docs = vec![-1, 2, 4],
            "consumer_past_end" => s.docs = vec![0, 2, 5],
            "consumer_get_error" | "producer_get_error" => s.fail_get = true,
            "consumer_next_error" => s.fail_next = true,
            "consumer_delegate_error" => s.fail_add = true,
            "consumer_close_first" => s.fail_consumer_close = 1,
            "consumer_close_second" => s.fail_consumer_close = 2,
            "producer_close_first" => s.fail_producer_close = 1,
            "producer_close_second" => s.fail_producer_close = 2,
            "producer_positioned" => s.initial_doc = 0,
            "producer_integrity_error" => s.fail_integrity = true,
            _ => {}
        }
    }
    let f = field("norm", 0, true);
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<()> {
        if kind.starts_with("consumer_") {
            let mut wrapped = AssertingNormsConsumer::new(consumer, 5);
            if kind.contains("close") {
                wrapped.close()?;
            } else {
                wrapped.add_norms_field(&f, &producer)?;
            }
        } else if kind.starts_with("format_") {
            let directory = ByteBuffersDirectory::new();
            let segment = NormsState {
                directory: &directory,
                segment: "_an",
                id: &[0; 16],
                suffix: "",
                max_doc: 5,
            };
            let mut wrapped = AssertingNormsFormat::new(Format(state.clone()));
            wrapped.delegate_mut().0.lock().unwrap().fail_open = kind.ends_with("error");
            assert!(Arc::ptr_eq(&wrapped.delegate().0, &state));
            if kind.contains("consumer") {
                wrapped.norms_consumer(&segment)?;
            } else {
                wrapped.norms_producer(
                    &segment,
                    &FieldInfos {
                        fields: if kind == "format_no_norms" {
                            vec![]
                        } else {
                            vec![f.clone()]
                        },
                    },
                )?;
            }
        } else {
            let mut wrapped = AssertingNormsProducer::new(producer.clone(), 5);
            if kind.contains("close") {
                wrapped.close()?;
            } else if kind.contains("integrity") {
                wrapped.check_integrity()?;
            } else if kind == "producer_merge" {
                wrapped.get_merge_instance().get_norms(&f)?.next_doc()?;
            } else if kind == "producer_merge_thread" {
                let merging = wrapped.get_merge_instance();
                let rejected = std::thread::spawn(move || {
                    catch_unwind(AssertUnwindSafe(|| {
                        let _ = merging.get_norms(&f);
                    }))
                    .is_err()
                })
                .join()
                .unwrap();
                assert!(rejected);
                panic!("expected Java wrong-thread assertion");
            } else {
                let requested = if kind == "producer_no_norms" {
                    field("none", 1, false)
                } else {
                    f
                };
                let mut values = wrapped.get_norms(&requested)?;
                match kind {
                    "producer_before_value" => {
                        values.long_value();
                    }
                    "producer_exact_bounds" => {
                        values.advance_exact(5)?;
                    }
                    "producer_backwards" => {
                        values.next_doc()?;
                        values.advance(0)?;
                    }
                    "producer_missing_value" => {
                        values.advance_exact(1)?;
                        values.long_value();
                    }
                    _ => {
                        values.next_doc()?;
                        values.long_value();
                        values.advance(2)?;
                        values.long_value();
                        values.advance_exact(4)?;
                        values.long_value();
                        values.next_doc()?;
                    }
                }
            }
        }
        Ok(())
    }));
    let (status, message) = match result {
        Err(_) => ("assert".into(), "-".into()),
        Ok(Err(error)) => ("io".into(), error.to_string()),
        Ok(Ok(())) => ("ok".into(), "-".into()),
    };
    let events = state.lock().unwrap().events.join(",");
    (status, message, events)
}
#[test]
fn java_callback_trace_preserves_validation_and_failure_order() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/asserting_norms.tsv"
    ));
    let mut count = 0;
    for line in fixture.lines() {
        let row: Vec<_> = line.split('\t').collect();
        assert_eq!(row.len(), 5);
        let actual = replay(row[1]);
        assert_eq!(
            (actual.0.as_str(), actual.1.as_str(), actual.2.as_str()),
            (row[2], row[3], row[4]),
            "{}",
            row[1]
        );
        count += 1;
    }
    assert_eq!(count, 31);
}

const META: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/testframework/data/asserting_norms.nvm"
));
const DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/testframework/data/asserting_norms.nvd"
));
const EMPTY_META: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/testframework/data/asserting_norms_empty.nvm"
));
const EMPTY_DATA: &[u8] = include_bytes!(concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../fixtures/testframework/data/asserting_norms_empty.nvd"
));
fn publish(directory: &ByteBuffersDirectory, file: &str, bytes: &[u8]) {
    let mut output = directory.create_output(file).unwrap();
    output.write_bytes(bytes);
    output.close().unwrap();
}
struct RealValues;
impl NormsProducer for RealValues {
    fn get_norms(&self, field: &FieldInfo) -> Result<Box<dyn NumericDocValues + '_>> {
        let (docs, values) = match field.number {
            0 => (vec![0, 1, 2, 3, 4], vec![-129, 0, 127, 128, 32768]),
            1 => (vec![0, 2, 4], vec![7, 8, 9]),
            _ => (vec![], vec![]),
        };
        Ok(Box::new(Values {
            state: Arc::new(Mutex::new(State::default())),
            docs,
            values,
            position: None,
            doc: -1,
            fail_next: false,
        }))
    }
    fn close(&mut self) -> Result<()> {
        Ok(())
    }
    fn check_integrity(&self) -> Result<()> {
        Ok(())
    }
    fn get_merge_instance(&self) -> Self {
        Self
    }
}
#[test]
fn native_delegate_matches_java_dense_sparse_empty_bytes_and_values() {
    for empty in [false, true] {
        let directory = ByteBuffersDirectory::new();
        let id = std::array::from_fn(|i| i as u8);
        let state = NormsState {
            directory: &directory,
            segment: "_an",
            id: &id,
            suffix: "s",
            max_doc: 5,
        };
        let fields = FieldInfos {
            fields: if empty {
                vec![]
            } else {
                vec![
                    field("dense", 0, true),
                    field("sparse", 1, true),
                    field("empty", 2, true),
                ]
            },
        };
        let mut format = AssertingNormsFormat::new(Lucene90NormsFormat);
        let mut consumer = format.norms_consumer(&state).unwrap();
        for field in &fields.fields {
            consumer.add_norms_field(field, &RealValues).unwrap();
        }
        consumer.close().unwrap();
        consumer.close().unwrap();
        assert_eq!(
            &*directory.open("_an_s.nvm").unwrap(),
            if empty { EMPTY_META } else { META }
        );
        assert_eq!(
            &*directory.open("_an_s.nvd").unwrap(),
            if empty { EMPTY_DATA } else { DATA }
        );
        if empty {
            assert!(
                catch_unwind(AssertUnwindSafe(|| format.norms_producer(&state, &fields))).is_err()
            );
            continue;
        }
        let mut producer = format.norms_producer(&state, &fields).unwrap();
        producer.check_integrity().unwrap();
        assert_eq!(
            producer.to_string(),
            "AssertingNormsProducer(Lucene90NormsProducer(fields=3))"
        );
        for field in &fields.fields {
            let mut expected = RealValues.get_norms(field).unwrap();
            let mut actual = producer.get_norms(field).unwrap();
            assert_eq!(actual.cost(), expected.cost());
            assert_eq!(actual.doc_id(), -1);
            for doc in 0..5 {
                let exists = expected.advance_exact(doc).unwrap();
                assert_eq!(actual.advance_exact(doc).unwrap(), exists);
                assert_eq!(actual.doc_id(), doc);
                if exists {
                    assert_eq!(actual.long_value(), expected.long_value());
                }
            }
            assert_eq!(actual.next_doc().unwrap(), NO_MORE_DOCS);
        }
        let merging = producer.get_merge_instance();
        let mut values = merging.get_norms(&fields.fields[1]).unwrap();
        assert_eq!(values.next_doc().unwrap(), 0);
        assert_eq!(values.long_value(), 7);
        assert_eq!(values.advance(3).unwrap(), 4);
        assert_eq!(values.long_value(), 9);
        assert_eq!(values.next_doc().unwrap(), NO_MORE_DOCS);
        drop(values);
        drop(merging);
        producer.close().unwrap();
        assert!(producer.get_norms(&fields.fields[0]).is_err());
        assert!(producer.check_integrity().is_err());
    }
}
#[test]
fn corrupted_java_bytes_fail_open_or_integrity() {
    for corruption in [
        "missing-meta",
        "bad-meta",
        "missing-data",
        "truncated-data",
        "wrong-id",
        "checksum",
    ] {
        let directory = ByteBuffersDirectory::new();
        let id = std::array::from_fn(|i| i as u8);
        let mut meta = META.to_vec();
        let mut data = DATA.to_vec();
        if corruption == "bad-meta" {
            meta[0] ^= 1;
        }
        if corruption == "truncated-data" {
            data.truncate(8);
        }
        if corruption == "checksum" {
            let end = data.len() - 1;
            data[end] ^= 1;
        }
        if corruption != "missing-meta" {
            publish(&directory, "_an_s.nvm", &meta);
        }
        if corruption != "missing-data" {
            publish(&directory, "_an_s.nvd", &data);
        }
        let wrong_id = [99; 16];
        let state = NormsState {
            directory: &directory,
            segment: "_an",
            id: if corruption == "wrong-id" {
                &wrong_id
            } else {
                &id
            },
            suffix: "s",
            max_doc: 5,
        };
        let mut format = AssertingNormsFormat::new(Lucene90NormsFormat);
        let result = format.norms_producer(
            &state,
            &FieldInfos {
                fields: vec![
                    field("dense", 0, true),
                    field("sparse", 1, true),
                    field("empty", 2, true),
                ],
            },
        );
        if corruption == "checksum" {
            assert!(result.unwrap().check_integrity().is_err());
        } else {
            assert!(result.is_err(), "{corruption}");
        }
    }
}

#[test]
fn native_merge_handles_close_independently_like_java_byte_buffers_inputs() {
    let fixture = include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/testframework/data/asserting_norms_merge_close.tsv"
    ));
    let directory = ByteBuffersDirectory::new();
    publish(&directory, "_an_s.nvm", META);
    publish(&directory, "_an_s.nvd", DATA);
    let id = std::array::from_fn(|i| i as u8);
    let state = NormsState {
        directory: &directory,
        segment: "_an",
        id: &id,
        suffix: "s",
        max_doc: 5,
    };
    let fields = FieldInfos {
        fields: vec![
            field("dense", 0, true),
            field("sparse", 1, true),
            field("empty", 2, true),
        ],
    };
    let mut format = AssertingNormsFormat::new(Lucene90NormsFormat);
    let mut output = String::new();
    let mut original = format.norms_producer(&state, &fields).unwrap();
    let mut merging = original.get_merge_instance();
    original.close().unwrap();
    let mut values = merging.get_norms(&fields.fields[1]).unwrap();
    assert_eq!(values.next_doc().unwrap(), 0);
    assert_eq!(values.long_value(), 7);
    merging.check_integrity().unwrap();
    output.push_str(&format!("origin-close\t{}\n", values.long_value()));
    drop(values);
    merging.close().unwrap();
    let mut original = format.norms_producer(&state, &fields).unwrap();
    let mut merging = original.get_merge_instance();
    merging.close().unwrap();
    let mut values = original.get_norms(&fields.fields[0]).unwrap();
    assert_eq!(values.next_doc().unwrap(), 0);
    assert_eq!(values.long_value(), -129);
    original.check_integrity().unwrap();
    output.push_str(&format!("merge-close\t{}\n", values.long_value()));
    drop(values);
    original.close().unwrap();
    assert_eq!(output, fixture);
}
