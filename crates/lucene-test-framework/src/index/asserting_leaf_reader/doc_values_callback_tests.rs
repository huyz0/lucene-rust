#![cfg(test)]
use super::*;
use std::cell::RefCell;
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::rc::Rc;

struct State {
    calls: Vec<&'static str>,
    doc: i32,
    returned_doc: i32,
    reported_doc: i32,
    exists: bool,
    counts: Vec<i32>,
    count_calls: usize,
    panic_count_at: Option<usize>,
    fail_value: bool,
}
impl Default for State {
    fn default() -> Self {
        Self {
            calls: Vec::new(),
            doc: -1,
            returned_doc: 0,
            reported_doc: 0,
            exists: true,
            counts: vec![1],
            count_calls: 0,
            panic_count_at: None,
            fail_value: false,
        }
    }
}
type Shared = Rc<RefCell<State>>;
struct Probe(Shared);
impl Probe {
    fn step(&self, call: &'static str) -> i32 {
        let mut state = self.0.borrow_mut();
        state.calls.push(call);
        state.doc = state.reported_doc;
        state.returned_doc
    }
    fn count(&self) -> i32 {
        let mut state = self.0.borrow_mut();
        state.calls.push("count");
        state.count_calls += 1;
        assert_ne!(
            Some(state.count_calls),
            state.panic_count_at,
            "count callback"
        );
        state.counts[(state.count_calls - 1).min(state.counts.len() - 1)]
    }
    fn value(&self, call: &'static str) -> Result<i64> {
        let mut state = self.0.borrow_mut();
        state.calls.push(call);
        if state.fail_value {
            return Err(lucene_store::Error::Io(std::io::Error::other("value failure")).into());
        }
        Ok(0)
    }
}
impl DocIdSetIterator for Probe {
    fn doc_id(&self) -> i32 {
        let mut state = self.0.borrow_mut();
        state.calls.push("docID");
        state.doc
    }
    fn next_doc(&mut self) -> Result<i32> {
        Ok(self.step("next"))
    }
    fn advance(&mut self, _: i32) -> Result<i32> {
        Ok(self.step("advance"))
    }
    fn cost(&self) -> i64 {
        1
    }
}
impl DocValuesIterator for Probe {
    fn advance_exact(&mut self, _: i32) -> Result<bool> {
        self.step("exact");
        Ok(self.0.borrow().exists)
    }
}
impl NumericDocValues for Probe {
    fn long_value(&self) -> i64 {
        7
    }
}
impl BinaryDocValues for Probe {
    fn binary_value(&self) -> &[u8] {
        b"v"
    }
}
impl SortedDocValues for Probe {
    fn ord_value(&self) -> i32 {
        0
    }
    fn lookup_ord(&mut self, _: i32) -> Result<Vec<u8>> {
        Ok(vec![0])
    }
    fn value_count(&self) -> i32 {
        2
    }
}
impl SortedNumericDocValues for Probe {
    fn doc_value_count(&self) -> i32 {
        self.count()
    }
    fn next_value(&mut self) -> Result<i64> {
        self.value("value")
    }
}
impl SortedSetDocValues for Probe {
    fn doc_value_count(&self) -> i32 {
        self.count()
    }
    fn next_ord(&mut self) -> Result<i64> {
        self.value("ord")
    }
    fn lookup_ord(&mut self, _: i64) -> Result<Vec<u8>> {
        Ok(vec![0])
    }
    fn value_count(&self) -> i64 {
        2
    }
}
fn iterator(kind: usize, state: &Shared) -> Box<dyn DocValuesIterator> {
    let probe = Box::new(Probe(state.clone()));
    let iterator: Box<dyn DocValuesIterator> = match kind {
        0 => Box::new(AssertingNumericDocValues::new(probe, 4)),
        1 => Box::new(AssertingBinaryDocValues::new(probe, 4)),
        2 => Box::new(AssertingSortedDocValues::new(probe, 4)),
        3 => Box::new(AssertingSortedNumericDocValues::new(probe, 4)),
        4 => Box::new(AssertingSortedSetDocValues::new(probe, 4)),
        _ => unreachable!(),
    };
    state.borrow_mut().calls.clear();
    iterator
}
fn panics(action: impl FnOnce()) {
    assert!(catch_unwind(AssertUnwindSafe(action)).is_err());
}

#[test]
fn movement_short_circuits_before_observing_delegate_position() {
    for kind in 0..5 {
        for returned_doc in [-1, 4] {
            let state = Shared::default();
            let mut values = iterator(kind, &state);
            state.borrow_mut().returned_doc = returned_doc;
            panics(|| {
                values.next_doc().unwrap();
            });
            assert_eq!(state.borrow().calls, ["next"], "kind={kind}");
        }
        for exact in [false, true] {
            let state = Shared::default();
            let mut values = iterator(kind, &state);
            panics(|| {
                if exact {
                    values.advance_exact(-1).unwrap();
                } else {
                    values.advance(-1).unwrap();
                }
            });
            assert!(state.borrow().calls.is_empty(), "kind={kind}");
        }
        let state = Shared::default();
        let mut values = iterator(kind, &state);
        state.borrow_mut().reported_doc = 1;
        if kind < 3 {
            // Java accepts the returned document without querying docID again.
            assert_eq!(values.advance(0).unwrap(), 0);
            assert_eq!(state.borrow().calls, ["docID", "advance"]);
        } else {
            panics(|| {
                values.advance(0).unwrap();
            });
            assert_eq!(state.borrow().calls, ["docID", "advance", "docID"]);
        }
    }
}

#[test]
fn advance_exact_commits_exists_before_failed_postcondition_only() {
    macro_rules! check {
        ($wrapper:ident) => {
            for exists in [false, true] {
                let state = Shared::default();
                let mut values = $wrapper::new(Box::new(Probe(state.clone())), 4);
                values.check.exists = !exists;
                state.borrow_mut().reported_doc = 1;
                state.borrow_mut().exists = exists;
                panics(|| {
                    values.advance_exact(0).unwrap();
                });
                assert_eq!(values.check.exists, exists);
                assert_eq!(values.check.last_doc_id, -1);
            }
        };
    }
    check!(AssertingNumericDocValues);
    check!(AssertingBinaryDocValues);
    check!(AssertingSortedDocValues);
    check!(AssertingSortedNumericDocValues);
    check!(AssertingSortedSetDocValues);

    let state = Shared::default();
    let mut numeric = AssertingSortedNumericDocValues::new(Box::new(Probe(state.clone())), 4);
    numeric.value_upto = 1;
    state.borrow_mut().reported_doc = 1;
    panics(|| {
        numeric.advance_exact(0).unwrap();
    });
    assert_eq!(numeric.value_upto, 1);
    assert!(numeric.advance_exact(1).unwrap());
    assert_eq!(numeric.value_upto, 0);

    let state = Shared::default();
    let mut set = AssertingSortedSetDocValues::new(Box::new(Probe(state.clone())), 4);
    set.ords_retrieved = 1;
    state.borrow_mut().reported_doc = 1;
    panics(|| {
        set.advance_exact(0).unwrap();
    });
    assert_eq!(set.ords_retrieved, 1);
    assert!(set.advance_exact(1).unwrap());
    assert_eq!(set.ords_retrieved, 0);
}

#[test]
fn per_kind_count_getters_preserve_duplicate_and_unchecked_calls() {
    let state = Shared::default();
    let mut numeric = AssertingSortedNumericDocValues::new(Box::new(Probe(state.clone())), 4);
    numeric.next_doc().unwrap();
    state.borrow_mut().counts = vec![1, 0];
    state.borrow_mut().calls.clear();
    assert_eq!(numeric.doc_value_count(), 0);
    assert_eq!(state.borrow().calls, ["count", "count"]);

    state.borrow_mut().count_calls = 0;
    state.borrow_mut().counts = vec![0];
    state.borrow_mut().calls.clear();
    panics(|| {
        numeric.doc_value_count();
    });
    assert_eq!(state.borrow().calls, ["count"]);

    let state = Shared::default();
    state.borrow_mut().counts = vec![0];
    let mut set = AssertingSortedSetDocValues::new(Box::new(Probe(state.clone())), 4);
    assert_eq!(set.doc_value_count(), 0);
    assert_eq!(state.borrow().calls, ["count"]);
    set.next_doc().unwrap();
    state.borrow_mut().calls.clear();
    assert_eq!(set.doc_value_count(), 0);
    assert_eq!(state.borrow().calls, ["count"]);
}

#[test]
fn failed_value_callbacks_keep_increment_and_assertion_message_reads() {
    let state = Shared::default();
    let mut numeric = AssertingSortedNumericDocValues::new(Box::new(Probe(state.clone())), 4);
    numeric.next_doc().unwrap();
    state.borrow_mut().fail_value = true;
    state.borrow_mut().calls.clear();
    assert!(numeric.next_value().is_err());
    assert_eq!(numeric.value_upto, 1);
    assert_eq!(state.borrow().calls, ["count", "value"]);
    state.borrow_mut().calls.clear();
    // The failing condition reads count, then its message reads count again.
    state.borrow_mut().panic_count_at = Some(3);
    let failure = catch_unwind(AssertUnwindSafe(|| numeric.next_value())).unwrap_err();
    assert!(failure
        .downcast_ref::<String>()
        .unwrap()
        .contains("count callback"));
    assert_eq!(state.borrow().calls, ["count", "count"]);
    assert_eq!(numeric.value_upto, 1);

    let state = Shared::default();
    let mut set = AssertingSortedSetDocValues::new(Box::new(Probe(state.clone())), 4);
    set.next_doc().unwrap();
    state.borrow_mut().fail_value = true;
    state.borrow_mut().calls.clear();
    assert!(set.next_ord().is_err());
    assert_eq!(set.ords_retrieved, 1);
    assert_eq!(state.borrow().calls, ["count", "ord"]);
    state.borrow_mut().calls.clear();
    panics(|| {
        set.next_ord().unwrap();
    });
    assert_eq!(set.ords_retrieved, 1);
    assert_eq!(state.borrow().calls, ["count"]);
}
