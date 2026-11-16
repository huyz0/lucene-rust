#![cfg(test)]
use super::*;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Mutex;
const KINDS: [&str; 5] = ["numeric", "binary", "sorted", "sortedNumeric", "sortedSet"];
const TYPES: [DocValuesType; 5] = [
    DocValuesType::Numeric,
    DocValuesType::Binary,
    DocValuesType::Sorted,
    DocValuesType::SortedNumeric,
    DocValuesType::SortedSet,
];
fn fields() -> FieldInfos {
    FieldInfos::new(
        KINDS
            .into_iter()
            .zip(TYPES)
            .enumerate()
            .map(|(i, (name, kind))| {
                FieldInfo::new(name, i as i32).with_doc_values(
                    kind,
                    if i == 0 {
                        DocValuesSkipIndexType::Range
                    } else {
                        DocValuesSkipIndexType::None
                    },
                    -1,
                )
            })
            .collect(),
    )
    .unwrap()
}
#[derive(Default)]
struct State {
    scenario: String,
    calls: Vec<String>,
    closes: usize,
}
type Shared = Arc<Mutex<State>>;
fn shared(scenario: &str) -> Shared {
    Arc::new(Mutex::new(State {
        scenario: scenario.into(),
        ..State::default()
    }))
}
fn note(state: &Shared, call: impl Into<String>) {
    state.lock().unwrap().calls.push(call.into());
}
fn call(state: &Shared, call: impl Into<String>) -> Result<()> {
    let call = call.into();
    let mut state = state.lock().unwrap();
    state.calls.push(call.clone());
    if state.scenario == "fail-close2" && call == "close" {
        state.closes += 1;
    }
    if state.scenario == format!("fail-{call}")
        || state.scenario == "fail-close2" && state.closes == 2 && call == "close"
    {
        return Err(error(format!("{call} failure")));
    }
    Ok(())
}
struct Cursor {
    state: Shared,
    scenario: String,
    row: i32,
    doc: i32,
    upto: usize,
}
impl Cursor {
    fn new(state: &Shared) -> Self {
        let scenario = state.lock().unwrap().scenario.clone();
        let doc = if scenario == "positioned" { 1 } else { -1 };
        Self {
            state: state.clone(),
            scenario,
            row: -1,
            doc,
            upto: 0,
        }
    }
    fn docs(&self) -> Vec<i32> {
        match self.scenario.as_str() {
            "empty" => vec![],
            "bounds" => vec![0, 7],
            "reverse" => vec![2, 0],
            "duplicate" => vec![0, 0],
            "zero" => vec![0],
            _ => vec![0, 2],
        }
    }
    fn count(&self) -> i32 {
        note(&self.state, "count");
        if self.scenario == "zero" {
            0
        } else {
            2
        }
    }
    fn dictionary_size(&self) -> i32 {
        note(&self.state, "valueCount");
        match self.scenario.as_str() {
            "empty" | "zero" => 0,
            "orphan" => 3,
            _ => 2,
        }
    }
    fn lookup(&self, ord: i64) -> Result<Vec<u8>> {
        call(&self.state, format!("lookup:{ord}"))?;
        Ok(vec![if self.scenario == "dictionary" {
            b'z' - ord as u8
        } else {
            b'a' + ord as u8
        }])
    }
    fn ordinal(&self, set: bool, index: usize) -> i64 {
        if self.scenario == "badord" && self.row == 1 {
            5
        } else if self.scenario == "negative" && self.row == 1 {
            -1
        } else if self.scenario == "descending" && set && self.row == 0 {
            if index == 0 {
                1
            } else {
                0
            }
        } else if set {
            index as i64
        } else {
            i64::from(self.row)
        }
    }
}
impl DocIdSetIterator for Cursor {
    fn doc_id(&self) -> i32 {
        note(&self.state, "docID");
        self.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        call(&self.state, "next")?;
        self.row += 1;
        self.upto = 0;
        self.doc = self
            .docs()
            .get(self.row as usize)
            .copied()
            .unwrap_or(NO_MORE_DOCS);
        Ok(self.doc)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        while self.next_doc()? < target {}
        Ok(self.doc)
    }
    fn cost(&self) -> i64 {
        2
    }
}
impl DocValuesIterator for Cursor {
    fn advance_exact(&mut self, _: i32) -> Result<bool> {
        Err(error("unsupported exact"))
    }
}
impl NumericDocValues for Cursor {
    fn long_value(&self) -> i64 {
        note(&self.state, "long");
        i64::from(self.row + 1)
    }
}
impl BinaryDocValues for Cursor {
    fn binary_value(&self) -> &[u8] {
        note(&self.state, "binary");
        &[0]
    }
}
impl SortedDocValues for Cursor {
    fn ord_value(&self) -> i32 {
        note(&self.state, "ordValue");
        self.ordinal(false, 0) as i32
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        self.lookup(i64::from(ord))
    }
    fn value_count(&self) -> i32 {
        self.dictionary_size()
    }
}
impl SortedNumericDocValues for Cursor {
    fn doc_value_count(&self) -> i32 {
        self.count()
    }
    fn next_value(&mut self) -> Result<i64> {
        call(&self.state, "value")?;
        let value = if self.scenario == "descending" && self.row == 0 {
            if self.upto == 0 {
                3
            } else {
                -1
            }
        } else if self.row == 0 {
            if self.upto == 0 {
                -1
            } else {
                3
            }
        } else {
            4
        };
        self.upto += 1;
        Ok(value)
    }
}
impl SortedSetDocValues for Cursor {
    fn doc_value_count(&self) -> i32 {
        self.count()
    }
    fn next_ord(&mut self) -> Result<i64> {
        call(&self.state, "ord")?;
        let value = self.ordinal(true, self.upto);
        self.upto += 1;
        Ok(value)
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        self.lookup(ord)
    }
    fn value_count(&self) -> i64 {
        i64::from(self.dictionary_size())
    }
}
#[derive(Clone)]
struct Producer(Shared);
impl fmt::Display for Producer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        note(&self.0, "toString");
        f.write_str("ProbeProducer")
    }
}
macro_rules! getter {
    ($method:ident, $trait:ident, $kind:literal) => {
        fn $method(&self, _: &FieldInfo) -> Result<Box<dyn $trait + '_>> {
            call(&self.0, concat!("get:", $kind))?;
            Ok(Box::new(Cursor::new(&self.0)))
        }
    };
}
impl DocValuesProducer for Producer {
    getter!(get_numeric, NumericDocValues, "numeric");
    getter!(get_binary, BinaryDocValues, "binary");
    getter!(get_sorted, SortedDocValues, "sorted");
    getter!(get_sorted_numeric, SortedNumericDocValues, "sortedNumeric");
    getter!(get_sorted_set, SortedSetDocValues, "sortedSet");
    fn get_skipper(&self, _: &FieldInfo) -> Result<Option<Box<dyn DocValuesSkipper + '_>>> {
        call(&self.0, "get:skipper")?;
        if self.0.lock().unwrap().scenario == "null-skipper" {
            return Ok(None);
        }
        let positioned = self.0.lock().unwrap().scenario == "positioned";
        Ok(Some(Box::new(Skipper {
            state: self.0.clone(),
            doc: if positioned { 0 } else { -1 },
        })))
    }
    fn close(&mut self) -> Result<()> {
        call(&self.0, "close")
    }
    fn check_integrity(&self) -> Result<()> {
        call(&self.0, "integrity")
    }
    fn get_merge_instance(&self) -> Self {
        note(&self.0, "merge");
        self.clone()
    }
}
struct Consumer<'a> {
    state: Shared,
    producer: &'a Producer,
    fields: &'a FieldInfos,
}
impl Consumer<'_> {
    fn add(&self, field: &FieldInfo, producer: &dyn DocValuesProducer) -> Result<()> {
        assert!(std::ptr::eq(
            field,
            self.fields.field_by_name(&field.name).unwrap()
        ));
        assert!(std::ptr::addr_eq(
            producer,
            self.producer as &dyn DocValuesProducer
        ));
        call(&self.state, "delegate")
    }
}
macro_rules! add {
    ($method:ident) => {
        fn $method(&mut self, field: &FieldInfo, producer: &dyn DocValuesProducer) -> Result<()> {
            self.add(field, producer)
        }
    };
}
impl DocValuesConsumer for Consumer<'_> {
    add!(add_numeric_field);
    add!(add_binary_field);
    add!(add_sorted_field);
    add!(add_sorted_numeric_field);
    add!(add_sorted_set_field);
    fn close(&mut self) -> Result<()> {
        call(&self.state, "close")
    }
}
struct Skipper {
    state: Shared,
    doc: i32,
}
impl DocValuesSkipper for Skipper {
    fn advance(&mut self, target: i32) -> Result<()> {
        call(&self.state, "skipAdvance")?;
        self.doc = if target < 5 { target } else { NO_MORE_DOCS };
        Ok(())
    }
    fn num_levels(&self) -> usize {
        2
    }
    fn min_doc_id(&self, _: usize) -> i32 {
        self.doc
    }
    fn max_doc_id(&self, level: usize) -> i32 {
        if self.doc == -1 || self.doc == NO_MORE_DOCS {
            self.doc
        } else if level == 0 {
            self.doc + 1
        } else {
            self.doc + 2
        }
    }
    fn min_value(&self, _: usize) -> i64 {
        -2
    }
    fn max_value(&self, _: usize) -> i64 {
        9
    }
    fn doc_count(&self, _: usize) -> i32 {
        2
    }
    fn global_min_value(&self) -> i64 {
        -2
    }
    fn global_max_value(&self) -> i64 {
        9
    }
    fn global_doc_count(&self) -> i32 {
        4
    }
    fn max_value_count(&self) -> i32 {
        2
    }
}
pub(super) fn add_field(
    consumer: &mut impl DocValuesConsumer,
    field: &FieldInfo,
    producer: &dyn DocValuesProducer,
) -> Result<()> {
    match field.doc_values_type {
        DocValuesType::Numeric => consumer.add_numeric_field(field, producer),
        DocValuesType::Binary => consumer.add_binary_field(field, producer),
        DocValuesType::Sorted => consumer.add_sorted_field(field, producer),
        DocValuesType::SortedNumeric => consumer.add_sorted_numeric_field(field, producer),
        DocValuesType::SortedSet => consumer.add_sorted_set_field(field, producer),
        DocValuesType::None => Err(error("no doc values")),
    }
}
fn get(producer: &dyn DocValuesProducer, kind: usize, field: &FieldInfo) -> Result<()> {
    match kind {
        0 => {
            producer.get_numeric(field)?;
        }
        1 => {
            producer.get_binary(field)?;
        }
        2 => {
            producer.get_sorted(field)?;
        }
        3 => {
            producer.get_sorted_numeric(field)?;
        }
        4 => {
            producer.get_sorted_set(field)?;
        }
        _ => unreachable!(),
    }
    Ok(())
}
fn outcome(action: impl FnOnce() -> Result<()>) -> String {
    match catch_unwind(AssertUnwindSafe(action)) {
        Ok(Ok(())) => "ok".into(),
        Ok(Err(e)) => format!("io:{e}"),
        Err(payload) => {
            if payload.downcast_ref::<&str>() == Some(&"null field membership") {
                "null-field".into()
            } else {
                "assert".into()
            }
        }
    }
}
#[test]
fn actual_java_consumer_producer_constructor_and_skipper_traces() {
    let fixtures = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../fixtures/testframework/data/asserting_doc_values.tsv"),
    )
    .unwrap();
    let fields = fields();
    let mut checked = 0;
    for row in fixtures.lines() {
        let p: Vec<_> = row.split('\t').collect();
        let group = p[0];
        let name = p[1];
        let scenario = p[2];
        let state = shared(scenario);
        let producer = Producer(state.clone());
        let status = if group == "consumer" {
            let mut consumer = AssertingDocValuesConsumer::new(
                Consumer {
                    state: state.clone(),
                    producer: &producer,
                    fields: &fields,
                },
                5,
            );
            outcome(|| {
                if name == "close" {
                    consumer.close()
                } else {
                    let field = fields.field_by_name(name).unwrap();
                    add_field(&mut consumer, field, &producer)
                }
            })
        } else if group == "producer" && name == "lifecycle" {
            let mut producer = AssertingDocValuesProducer::new(producer, &fields, 5);
            outcome(|| {
                if scenario.contains("integrity") {
                    producer.check_integrity()
                } else {
                    producer.close()
                }
            })
        } else if group == "producer" {
            let kind = KINDS.iter().position(|kind| *kind == name).unwrap();
            let base = AssertingDocValuesProducer::new(producer, &fields, 5);
            let producer = if matches!(scenario, "merge" | "thread") {
                base.get_merge_instance()
            } else {
                base
            };
            let field = match scenario {
                "wrong-number" => FieldInfo::new(name, 17).with_doc_values(
                    TYPES[kind],
                    DocValuesSkipIndexType::None,
                    -1,
                ),
                "wrong-type" => FieldInfo::new(name, kind as i32).with_doc_values(
                    TYPES[(kind + 1) % 5],
                    DocValuesSkipIndexType::None,
                    -1,
                ),
                "missing" => FieldInfo::new("absent", kind as i32).with_doc_values(
                    TYPES[kind],
                    DocValuesSkipIndexType::None,
                    -1,
                ),
                _ => fields.fields[kind].clone(),
            };
            if scenario == "thread" {
                thread::scope(|scope| {
                    scope
                        .spawn(move || outcome(|| get(&producer, kind, &field)))
                        .join()
                        .unwrap()
                })
            } else {
                outcome(|| get(&producer, kind, &field))
            }
        } else if group == "skipper" && scenario == "thread" {
            let producer = AssertingDocValuesProducer::new(producer, &fields, 5);
            let mut skipper = producer.get_skipper(&fields.fields[0]).unwrap().unwrap();
            let mut statuses = Vec::new();
            for op in 0..11 {
                let status = thread::scope(|scope| {
                    let skipper = &mut skipper;
                    scope
                        .spawn(move || {
                            outcome(|| {
                                match op {
                                    0 => skipper.advance(0)?,
                                    1 => {
                                        skipper.num_levels();
                                    }
                                    2 => {
                                        skipper.min_doc_id(0);
                                    }
                                    3 => {
                                        skipper.max_doc_id(0);
                                    }
                                    4 => {
                                        skipper.min_value(0);
                                    }
                                    5 => {
                                        skipper.max_value(0);
                                    }
                                    6 => {
                                        skipper.doc_count(0);
                                    }
                                    7 => {
                                        skipper.global_min_value();
                                    }
                                    8 => {
                                        skipper.global_max_value();
                                    }
                                    9 => {
                                        skipper.global_doc_count();
                                    }
                                    10 => {
                                        skipper.max_value_count();
                                    }
                                    _ => unreachable!(),
                                }
                                Ok(())
                            })
                        })
                        .join()
                        .unwrap()
                });
                statuses.push(status);
            }
            statuses.join(";")
        } else if group == "skipper" {
            let producer = AssertingDocValuesProducer::new(producer, &fields, 5);
            outcome(|| {
                let field = match scenario {
                    "none" => FieldInfo::new("numeric", 0).with_doc_values(
                        DocValuesType::Numeric,
                        DocValuesSkipIndexType::None,
                        -1,
                    ),
                    "missing" => FieldInfo::new("absent", 0).with_doc_values(
                        DocValuesType::Numeric,
                        DocValuesSkipIndexType::Range,
                        -1,
                    ),
                    "wrong-number" => FieldInfo::new("numeric", 17).with_doc_values(
                        DocValuesType::Numeric,
                        DocValuesSkipIndexType::Range,
                        -1,
                    ),
                    _ => fields.fields[0].clone(),
                };
                let mut skipper = producer.get_skipper(&field)?.unwrap();
                match scenario {
                    "unpositioned" => {
                        skipper.min_value(0);
                    }
                    "level" => {
                        skipper.min_doc_id(2);
                    }
                    "advance" => {
                        skipper.advance(0)?;
                        skipper.advance(1)?;
                    }
                    _ => {
                        skipper.advance(0)?;
                        assert_eq!(skipper.num_levels(), 2);
                        assert_eq!(skipper.min_doc_id(1), 0);
                        assert_eq!(skipper.max_doc_id(1), 2);
                        assert_eq!(skipper.min_value(0), -2);
                        assert_eq!(skipper.max_value(0), 9);
                        assert_eq!(skipper.doc_count(0), 2);
                        assert_eq!(skipper.global_min_value(), -2);
                        assert_eq!(skipper.global_max_value(), 9);
                        assert_eq!(skipper.global_doc_count(), 4);
                        assert_eq!(skipper.max_value_count(), 2);
                    }
                }
                Ok(())
            })
        } else {
            panic!("unknown group {group}");
        };
        assert_eq!(
            (status, state.lock().unwrap().calls.join(",")),
            (p[3].to_string(), p.get(4).unwrap_or(&"").to_string()),
            "{row}"
        );
        checked += 1;
    }
    assert!(checked > 130);
}
