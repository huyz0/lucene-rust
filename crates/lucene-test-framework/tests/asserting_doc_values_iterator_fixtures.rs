//! Actual Java asserting-leaf iterator callback order and failure-state replay.
use lucene_search::reader::{
    BinaryDocValues, DocIdSetIterator, DocValuesIterator, NumericDocValues, SortedDocValues,
    SortedNumericDocValues, SortedSetDocValues,
};
use lucene_search::Result;
use lucene_test_framework::index::asserting_leaf_reader::{
    AssertingBinaryDocValues, AssertingNumericDocValues, AssertingSortedDocValues,
    AssertingSortedNumericDocValues, AssertingSortedSetDocValues,
};
use std::{
    cell::RefCell,
    panic::{catch_unwind, AssertUnwindSafe},
    rc::Rc,
};

struct State {
    doc: i32,
    next: i32,
    advanced: i32,
    exact_doc: i32,
    report: Option<i32>,
    exists: bool,
    count: i32,
    count2: i32,
    count_calls: usize,
    dictionary: i32,
    ord: i32,
    value: i32,
    lookup: i32,
    cost: i64,
    fail: String,
    events: Vec<String>,
}
impl Default for State {
    fn default() -> Self {
        Self {
            doc: -1,
            next: 0,
            advanced: 2,
            exact_doc: 2,
            report: None,
            exists: true,
            count: 2,
            count2: 2,
            count_calls: 0,
            dictionary: 4,
            ord: 1,
            value: 7,
            lookup: 1,
            cost: 3,
            fail: "none".into(),
            events: vec![],
        }
    }
}
#[derive(Clone, Default)]
struct Probe(Rc<RefCell<State>>);
impl Probe {
    fn event(&self, name: &str) -> Result<()> {
        let mut s = self.0.borrow_mut();
        s.events.push(name.into());
        let base = name.split(':').next().unwrap();
        let occurrence = s
            .events
            .iter()
            .filter(|event| event.split(':').next() == Some(base))
            .count();
        if s.fail == format!("{base}:runtime")
            || (s.fail == format!("{base}-second:runtime") && occurrence == 2)
            || (base == "count" && s.fail == "count-second:runtime" && s.count_calls == 1)
        {
            std::panic::panic_any("delegate-runtime");
        }
        if s.fail == format!("{base}:io") {
            return Err(lucene_store::Error::Io(std::io::Error::other("delegate-io")).into());
        }
        Ok(())
    }
    fn scalar(&self, name: &str) {
        self.event(name).unwrap();
    }
    fn count(&self) -> i32 {
        self.scalar("count");
        let mut s = self.0.borrow_mut();
        let count = if s.count_calls == 0 {
            s.count
        } else {
            s.count2
        };
        s.count_calls += 1;
        count
    }
    fn dictionary(&self) -> i32 {
        self.scalar("dictionary");
        self.0.borrow().dictionary
    }
    fn value(&self) -> Result<i64> {
        self.event("value")?;
        Ok(i64::from(self.0.borrow().value))
    }
    fn ord(&self) -> Result<i64> {
        self.event("ord")?;
        Ok(i64::from(self.0.borrow().ord))
    }
    fn lookup(&self, ord: i64) -> Result<Vec<u8>> {
        self.event(&format!("lookupOrd:{ord}"))?;
        Ok(format!("t{ord}").into_bytes())
    }
    fn term(&self, key: &[u8]) -> Result<i32> {
        self.event(&format!("lookupTerm:{}", std::str::from_utf8(key).unwrap()))?;
        Ok(self.0.borrow().lookup)
    }
    fn set(&self, command: &str) {
        let (key, value) = command.split_once('=').unwrap();
        let mut s = self.0.borrow_mut();
        match key {
            "doc" => s.doc = value.parse().unwrap(),
            "next" => s.next = value.parse().unwrap(),
            "advanced" => s.advanced = value.parse().unwrap(),
            "exactDoc" => s.exact_doc = value.parse().unwrap(),
            "report" => {
                s.report = if value == "none" {
                    None
                } else {
                    Some(value.parse().unwrap())
                }
            }
            "exists" => s.exists = value.parse().unwrap(),
            "count" => {
                s.count = value.parse().unwrap();
                s.count2 = s.count;
                s.count_calls = 0;
            }
            "count2" => s.count2 = value.parse().unwrap(),
            "dictionary" => s.dictionary = value.parse().unwrap(),
            "ord" => s.ord = value.parse().unwrap(),
            "value" => s.value = value.parse().unwrap(),
            "lookup" => s.lookup = value.parse().unwrap(),
            "cost" => s.cost = value.parse().unwrap(),
            "fail" => s.fail = value.into(),
            _ => panic!("unknown fixture setting {command}"),
        }
    }
}
impl DocIdSetIterator for Probe {
    fn doc_id(&self) -> i32 {
        self.scalar("docID");
        let s = self.0.borrow();
        s.report.unwrap_or(s.doc)
    }
    fn next_doc(&mut self) -> Result<i32> {
        self.event("next")?;
        let mut s = self.0.borrow_mut();
        s.doc = s.next;
        Ok(s.next)
    }
    fn advance(&mut self, target: i32) -> Result<i32> {
        self.event(&format!("advance:{target}"))?;
        let mut s = self.0.borrow_mut();
        s.doc = s.advanced;
        Ok(s.advanced)
    }
    fn cost(&self) -> i64 {
        self.scalar("cost");
        self.0.borrow().cost
    }
}
impl DocValuesIterator for Probe {
    fn advance_exact(&mut self, target: i32) -> Result<bool> {
        self.event(&format!("exact:{target}"))?;
        let mut s = self.0.borrow_mut();
        s.doc = s.exact_doc;
        Ok(s.exists)
    }
}
impl NumericDocValues for Probe {
    fn long_value(&self) -> i64 {
        self.value().unwrap()
    }
}
impl BinaryDocValues for Probe {
    fn binary_value(&self) -> &[u8] {
        self.scalar("binary");
        b"v"
    }
}
impl SortedDocValues for Probe {
    fn ord_value(&self) -> i32 {
        self.ord().unwrap() as i32
    }
    fn lookup_ord(&mut self, ord: i32) -> Result<Vec<u8>> {
        self.lookup(i64::from(ord))
    }
    fn value_count(&self) -> i32 {
        self.dictionary()
    }
    fn lookup_term(&mut self, key: &[u8]) -> Result<i32> {
        self.term(key)
    }
}
impl SortedNumericDocValues for Probe {
    fn doc_value_count(&self) -> i32 {
        self.count()
    }
    fn next_value(&mut self) -> Result<i64> {
        self.value()
    }
}
impl SortedSetDocValues for Probe {
    fn doc_value_count(&self) -> i32 {
        self.count()
    }
    fn next_ord(&mut self) -> Result<i64> {
        self.ord()
    }
    fn lookup_ord(&mut self, ord: i64) -> Result<Vec<u8>> {
        self.lookup(ord)
    }
    fn value_count(&self) -> i64 {
        i64::from(self.dictionary())
    }
    fn lookup_term(&mut self, key: &[u8]) -> Result<i64> {
        self.term(key).map(i64::from)
    }
}
enum Wrapped {
    Numeric(AssertingNumericDocValues<'static>),
    Binary(AssertingBinaryDocValues<'static>),
    Sorted(AssertingSortedDocValues<'static>),
    SortedNumeric(AssertingSortedNumericDocValues<'static>),
    SortedSet(AssertingSortedSetDocValues<'static>),
}
macro_rules! every {
    ($this:expr, $dv:ident, $body:expr) => {
        match $this {
            Wrapped::Numeric($dv) => $body,
            Wrapped::Binary($dv) => $body,
            Wrapped::Sorted($dv) => $body,
            Wrapped::SortedNumeric($dv) => $body,
            Wrapped::SortedSet($dv) => $body,
        }
    };
}
impl Wrapped {
    fn new(kind: &str, probe: Probe) -> Self {
        match kind {
            "numeric" => Self::Numeric(AssertingNumericDocValues::new(Box::new(probe), 8)),
            "binary" => Self::Binary(AssertingBinaryDocValues::new(Box::new(probe), 8)),
            "sorted" => Self::Sorted(AssertingSortedDocValues::new(Box::new(probe), 8)),
            "sortedNumeric" => {
                Self::SortedNumeric(AssertingSortedNumericDocValues::new(Box::new(probe), 8))
            }
            "sortedSet" => Self::SortedSet(AssertingSortedSetDocValues::new(Box::new(probe), 8)),
            _ => panic!("unknown fixture kind {kind}"),
        }
    }
    fn call(&mut self, command: &str) -> Result<String> {
        let (name, arg) = command.split_once(':').unwrap_or((command, ""));
        let number = || arg.parse::<i32>().unwrap();
        Ok(match name {
            "docID" => every!(self, dv, dv.doc_id().to_string()),
            "next" => every!(self, dv, dv.next_doc()?.to_string()),
            "advance" => every!(self, dv, dv.advance(number())?.to_string()),
            "exact" => every!(self, dv, dv.advance_exact(number())?.to_string()),
            "cost" => every!(self, dv, dv.cost().to_string()),
            "value" => match self {
                Self::Numeric(dv) => dv.long_value().to_string(),
                Self::Binary(dv) => std::str::from_utf8(dv.binary_value()).unwrap().to_owned(),
                Self::Sorted(dv) => dv.ord_value().to_string(),
                Self::SortedNumeric(dv) => dv.next_value()?.to_string(),
                Self::SortedSet(dv) => dv.next_ord()?.to_string(),
            },
            "count" => match self {
                Self::SortedNumeric(dv) => dv.doc_value_count().to_string(),
                Self::SortedSet(dv) => dv.doc_value_count().to_string(),
                _ => panic!("count on scalar fixture"),
            },
            "dictionary" => match self {
                Self::Sorted(dv) => dv.value_count().to_string(),
                Self::SortedSet(dv) => dv.value_count().to_string(),
                _ => panic!("dictionary on non-dictionary fixture"),
            },
            "lookup" => {
                let bytes = match self {
                    Self::Sorted(dv) => dv.lookup_ord(number())?,
                    Self::SortedSet(dv) => dv.lookup_ord(i64::from(number()))?,
                    _ => panic!("lookup on non-dictionary fixture"),
                };
                String::from_utf8(bytes).unwrap()
            }
            "term" => match self {
                Self::Sorted(dv) => dv.lookup_term(arg.as_bytes())?.to_string(),
                Self::SortedSet(dv) => dv.lookup_term(arg.as_bytes())?.to_string(),
                _ => panic!("term on non-dictionary fixture"),
            },
            _ => panic!("unknown fixture command {command}"),
        })
    }
}
fn replay(kind: &str, script: &str) -> String {
    let probe = Probe::default();
    let mut wrapped = None;
    let mut output = vec![];
    for command in std::iter::once("new").chain(script.split(';')) {
        if command.contains('=') {
            probe.set(command);
            continue;
        }
        probe.0.borrow_mut().events.clear();
        let result = catch_unwind(AssertUnwindSafe(|| {
            if command == "new" {
                wrapped = Some(Wrapped::new(kind, probe.clone()));
                Ok("ok".to_owned())
            } else {
                wrapped
                    .as_mut()
                    .unwrap()
                    .call(command)
                    .map(|value| format!("ok:{value}"))
            }
        }));
        let status = match result {
            Ok(Ok(value)) => value,
            Ok(Err(_)) => "io".into(),
            Err(payload) => {
                let message = payload
                    .downcast_ref::<&str>()
                    .copied()
                    .or_else(|| payload.downcast_ref::<String>().map(String::as_str));
                if message == Some("delegate-runtime") {
                    "runtime".into()
                } else {
                    "assert".into()
                }
            }
        };
        output.push(format!(
            "{command}~{status}~{}",
            probe.0.borrow().events.join(",")
        ));
    }
    output.join("|")
}
#[test]
fn actual_java_iterator_callbacks_and_failure_state_match_for_all_five_types() {
    let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/testframework/data/asserting_doc_values_iterators.tsv");
    let text = std::fs::read_to_string(path).unwrap();
    assert!(!text.is_empty(), "the Java fixture must contain cases");
    let mut kinds = std::collections::BTreeSet::new();
    for line in text.lines() {
        let columns: Vec<_> = line.split('\t').collect();
        assert_eq!(columns.len(), 4, "malformed Java fixture: {line}");
        kinds.insert(columns[0]);
        assert_eq!(
            replay(columns[0], columns[2]),
            columns[3],
            "{} / {}",
            columns[0],
            columns[1]
        );
    }
    assert_eq!(
        kinds.len(),
        5,
        "the actual Java fixture must exercise all five wrappers"
    );
}
