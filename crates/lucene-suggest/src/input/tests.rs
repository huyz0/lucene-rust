#![cfg(test)]
use super::*;

fn words(ws: &[&str]) -> InputIteratorWrapper<std::vec::IntoIter<Vec<u8>>> {
    InputIteratorWrapper::new(
        ws.iter()
            .map(|w| w.as_bytes().to_vec())
            .collect::<Vec<_>>()
            .into_iter(),
    )
}

fn drain(it: &mut dyn InputIterator) -> Vec<(String, i64)> {
    let mut out = Vec::new();
    while let Some(t) = it.next() {
        out.push((String::from_utf8(t).unwrap(), it.weight()));
    }
    out
}

#[test]
fn wrapper_has_weight_one_and_no_extras() {
    let mut w = words(&["a", "b"]);
    assert_eq!(w.weight(), 1);
    assert!(w.payload().is_none());
    assert!(!w.has_payloads());
    assert!(w.contexts().is_none());
    assert!(!w.has_contexts());
    assert_eq!(drain(&mut w), vec![("a".into(), 1), ("b".into(), 1)]);
}

#[test]
fn empty_input_yields_nothing() {
    let mut e = input_empty();
    assert!(e.next().is_none());
    assert_eq!(e.weight(), 1);
}

/// A source with payloads, contexts and weights, to exercise every
/// accessor the buffering and sorting paths read.
struct Rich {
    rows: std::vec::IntoIter<(&'static str, i64)>,
    cur: Option<(&'static str, i64)>,
    payloads: bool,
    contexts: bool,
}

impl InputIterator for Rich {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.cur = self.rows.next();
        self.cur.map(|(t, _)| t.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        self.cur.map_or(0, |(_, w)| w)
    }
    fn payload(&self) -> Option<&[u8]> {
        if self.payloads {
            Some(b"p")
        } else {
            None
        }
    }
    fn has_payloads(&self) -> bool {
        self.payloads
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        None
    }
    fn has_contexts(&self) -> bool {
        self.contexts
    }
}

fn rich(rows: Vec<(&'static str, i64)>, payloads: bool, contexts: bool) -> Rich {
    Rich {
        rows: rows.into_iter(),
        cur: None,
        payloads,
        contexts,
    }
}

#[test]
fn buffered_replays_in_input_order_with_weights() {
    let mut src = rich(vec![("b", 5), ("a", 2), ("b", 7)], true, false);
    let mut buf = BufferedInputIterator::new(&mut src);
    assert!(buf.has_payloads());
    assert!(!buf.has_contexts());
    // before the first next, nothing is current
    assert_eq!(buf.weight(), 0);
    assert!(buf.payload().is_none());
    assert_eq!(
        drain(&mut buf),
        vec![("b".into(), 5), ("a".into(), 2), ("b".into(), 7)]
    );
    // exhausted: no current entry
    assert!(buf.next().is_none());
    assert_eq!(buf.weight(), 0);
    assert!(buf.payload().is_none());
    assert!(buf.contexts().is_none());
}

#[test]
fn buffered_exposes_payloads_and_contexts_of_the_current_entry() {
    let mut src = rich(vec![("x", 1)], true, true);
    let mut buf = BufferedInputIterator::new(&mut src);
    assert!(buf.next().is_some());
    assert_eq!(buf.payload(), Some(&b"p"[..]));
    // the rich source's contexts are None, so the buffer has none either
    assert!(buf.contexts().is_none());
    assert!(buf.has_contexts());
}

#[test]
fn buffered_without_payloads_reports_none() {
    let mut src = rich(vec![("x", 1)], false, false);
    let mut buf = BufferedInputIterator::new(&mut src);
    assert!(buf.next().is_some());
    assert!(buf.payload().is_none());
}

#[test]
fn buffered_substitutes_an_empty_payload_for_a_null_one() {
    struct NullPayload(bool);
    impl InputIterator for NullPayload {
        fn next(&mut self) -> Option<Vec<u8>> {
            if self.0 {
                None
            } else {
                self.0 = true;
                Some(b"t".to_vec())
            }
        }
        fn weight(&self) -> i64 {
            3
        }
        fn payload(&self) -> Option<&[u8]> {
            None
        }
        fn has_payloads(&self) -> bool {
            true
        }
        fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
            None
        }
        fn has_contexts(&self) -> bool {
            false
        }
    }
    let mut buf = BufferedInputIterator::new(&mut NullPayload(false));
    assert!(buf.next().is_some());
    assert_eq!(buf.payload(), Some(&b""[..]));
}

#[test]
fn sorted_orders_by_term_then_weight() {
    let mut src = rich(
        vec![("b", 1), ("a", 9), ("b", 0), ("a", 2), ("ab", 4)],
        false,
        false,
    );
    let mut s = SortedInputIterator::new(&mut src);
    assert_eq!(
        drain(&mut s),
        vec![
            ("a".into(), 2),
            ("a".into(), 9),
            ("ab".into(), 4),
            ("b".into(), 0),
            ("b".into(), 1),
        ]
    );
}

#[test]
fn sorted_with_a_custom_comparator_reverses_terms() {
    let mut src = rich(vec![("a", 1), ("b", 2)], false, false);
    let mut s = SortedInputIterator::with_comparator(&mut src, |x, y| y.cmp(x));
    assert_eq!(drain(&mut s), vec![("b".into(), 2), ("a".into(), 1)]);
}

#[test]
fn sorted_exposes_payloads_and_contexts_for_the_current_entry() {
    let mut src = rich(vec![("q", 4)], true, true);
    let mut s = SortedInputIterator::new(&mut src);
    // before the first next there is no current entry
    assert_eq!(s.payload(), Some(&b""[..]));
    assert!(s.contexts().is_none());
    assert!(s.has_contexts());
    assert!(s.next().is_some());
    assert_eq!(s.payload(), Some(&b"p"[..]));
    assert_eq!(s.weight(), 4);
    assert!(s.contexts().is_none());
    assert!(s.next().is_none());
    assert_eq!(s.weight(), 0);
}

#[test]
fn sorted_without_payloads_reports_none() {
    let mut src = rich(vec![("q", 1)], false, false);
    let s = SortedInputIterator::new(&mut src);
    assert!(s.payload().is_none());
    assert!(!s.has_payloads());
    assert!(!s.has_contexts());
}

#[test]
fn index_guards_against_a_negative_cursor() {
    let buf = BufferedInputIterator {
        entries: vec![b"z".to_vec()],
        payloads: Vec::new(),
        context_sets: Vec::new(),
        freqs: vec![5],
        cur: -1,
        has_payloads: false,
        has_contexts: false,
    };
    assert_eq!(buf.index(), None);
    assert_eq!(buf.weight(), 0);
}
