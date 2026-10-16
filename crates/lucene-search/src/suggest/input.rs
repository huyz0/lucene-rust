//! `org.apache.lucene.search.suggest.InputIterator` and its wrappers (M13
//! T13.3a): the `(term, weight, payload, contexts)` stream a suggester is
//! built from, the buffering and sorting adapters, and the in-memory form of
//! `SortedInputIterator`.
//!
//! Java sorts on disk with `OfflineSorter` once the input outgrows memory. The
//! ordering is the same here, `(term, weight)` by unsigned bytes then weight
//! ascending, but the sort runs in memory: a suggester's input is a
//! `Vec` in this port, not a temp file. `Set<BytesRef> contexts` is a
//! `BTreeSet` so that iteration order is deterministic.

use std::collections::BTreeSet;

/// `InputIterator`: a stream of suggestion entries. Each call to
/// [`next`](Self::next) advances to the next entry; the accessors describe the
/// entry it last returned.
pub trait InputIterator {
    /// `next()`: the next term, or `None` at the end.
    fn next(&mut self) -> Option<Vec<u8>>;

    /// `weight()`: the entry's weight (its popularity).
    fn weight(&self) -> i64;

    /// `payload()`: the entry's payload, when [`has_payloads`](Self::has_payloads).
    fn payload(&self) -> Option<&[u8]>;

    /// `hasPayloads()`.
    fn has_payloads(&self) -> bool;

    /// `contexts()`: the entry's context set, when
    /// [`has_contexts`](Self::has_contexts).
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>>;

    /// `hasContexts()`.
    fn has_contexts(&self) -> bool;
}

/// `InputIterator.InputIteratorWrapper`: a plain term stream, weight 1, no
/// payloads, no contexts. `InputIterator.EMPTY` is `input_empty()`.
#[derive(Debug, Clone)]
pub struct InputIteratorWrapper<I> {
    wrapped: I,
}

impl<I> InputIteratorWrapper<I> {
    /// `new InputIteratorWrapper(BytesRefIterator)`.
    pub fn new(wrapped: I) -> Self {
        Self { wrapped }
    }
}

impl<I: Iterator<Item = Vec<u8>>> InputIterator for InputIteratorWrapper<I> {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.wrapped.next()
    }

    fn weight(&self) -> i64 {
        1
    }

    fn payload(&self) -> Option<&[u8]> {
        None
    }

    fn has_payloads(&self) -> bool {
        false
    }

    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        None
    }

    fn has_contexts(&self) -> bool {
        false
    }
}

/// `InputIterator.EMPTY`: a stream with no entries.
pub fn input_empty() -> InputIteratorWrapper<std::iter::Empty<Vec<u8>>> {
    InputIteratorWrapper::new(std::iter::empty())
}

/// `BufferedInputIterator`: drains its source at construction, then replays the
/// entries in order, so a suggester that needs two passes can read them twice.
#[derive(Debug, Clone)]
pub struct BufferedInputIterator {
    entries: Vec<Vec<u8>>,
    payloads: Vec<Vec<u8>>,
    context_sets: Vec<Option<BTreeSet<Vec<u8>>>>,
    freqs: Vec<i64>,
    /// `curPos`: the index of the current entry, `-1` before the first `next`.
    cur: isize,
    has_payloads: bool,
    has_contexts: bool,
}

impl BufferedInputIterator {
    /// `new BufferedInputIterator(source)`: reads every entry of `source`.
    pub fn new(source: &mut dyn InputIterator) -> Self {
        let has_payloads = source.has_payloads();
        let has_contexts = source.has_contexts();
        let mut out = Self {
            entries: Vec::new(),
            payloads: Vec::new(),
            context_sets: Vec::new(),
            freqs: Vec::new(),
            cur: -1,
            has_payloads,
            has_contexts,
        };
        while let Some(term) = source.next() {
            out.entries.push(term);
            if has_payloads {
                // Java appends the payload unchecked; a null here fails later
                // in Java, so the empty payload stands in for it.
                out.payloads
                    .push(source.payload().map(<[u8]>::to_vec).unwrap_or_default());
            }
            if has_contexts {
                out.context_sets.push(source.contexts().cloned());
            }
            out.freqs.push(source.weight());
        }
        out
    }
}

impl InputIterator for BufferedInputIterator {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.cur += 1;
        usize::try_from(self.cur)
            .ok()
            .and_then(|i| self.entries.get(i))
            .cloned()
    }

    fn weight(&self) -> i64 {
        // Java reads freqs[curPos] with no bounds check; before the first
        // `next` or after the end that is an exception there, and 0 here.
        self.index()
            .and_then(|i| self.freqs.get(i))
            .copied()
            .unwrap_or(0)
    }

    fn payload(&self) -> Option<&[u8]> {
        if !self.has_payloads {
            return None;
        }
        self.index()
            .and_then(|i| self.payloads.get(i))
            .map(Vec::as_slice)
    }

    fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        if !self.has_contexts {
            return None;
        }
        self.index()
            .and_then(|i| self.context_sets.get(i))
            .and_then(Option::as_ref)
    }

    fn has_contexts(&self) -> bool {
        self.has_contexts
    }
}

impl BufferedInputIterator {
    fn index(&self) -> Option<usize> {
        usize::try_from(self.cur).ok()
    }
}

/// One buffered record of a [`SortedInputIterator`].
#[derive(Debug, Clone)]
struct Entry {
    term: Vec<u8>,
    weight: i64,
    payload: Vec<u8>,
    contexts: Option<BTreeSet<Vec<u8>>>,
}

/// `SortedInputIterator`: drains its source, sorts by term (then by weight,
/// the tie-break Java's `tieBreakByCostComparator` uses), and replays in that
/// order.
#[derive(Debug, Clone)]
pub struct SortedInputIterator {
    entries: std::vec::IntoIter<Entry>,
    current: Option<Entry>,
    has_payloads: bool,
    has_contexts: bool,
}

impl SortedInputIterator {
    /// `new SortedInputIterator(tempDir, prefix, source)` with Java's natural
    /// order on terms: unsigned bytes, a proper prefix first.
    pub fn new(source: &mut dyn InputIterator) -> Self {
        Self::with_comparator(source, <[u8]>::cmp)
    }

    /// The `Comparator<BytesRef>` overload.
    pub fn with_comparator(
        source: &mut dyn InputIterator,
        cmp: impl Fn(&[u8], &[u8]) -> std::cmp::Ordering,
    ) -> Self {
        let has_payloads = source.has_payloads();
        let has_contexts = source.has_contexts();
        let mut entries = Vec::new();
        while let Some(term) = source.next() {
            entries.push(Entry {
                term,
                weight: source.weight(),
                payload: if has_payloads {
                    source.payload().map(<[u8]>::to_vec).unwrap_or_default()
                } else {
                    Vec::new()
                },
                contexts: if has_contexts {
                    source.contexts().cloned()
                } else {
                    None
                },
            });
        }
        // Stable, like the sort it stands in for; equal (term, weight) pairs
        // keep their input order.
        entries.sort_by(|a, b| cmp(&a.term, &b.term).then(a.weight.cmp(&b.weight)));
        Self {
            entries: entries.into_iter(),
            current: None,
            has_payloads,
            has_contexts,
        }
    }
}

impl InputIterator for SortedInputIterator {
    fn next(&mut self) -> Option<Vec<u8>> {
        self.current = self.entries.next();
        self.current.as_ref().map(|e| e.term.clone())
    }

    fn weight(&self) -> i64 {
        self.current.as_ref().map_or(0, |e| e.weight)
    }

    fn payload(&self) -> Option<&[u8]> {
        if !self.has_payloads {
            return None;
        }
        Some(
            self.current
                .as_ref()
                .map_or(&[][..], |e| e.payload.as_slice()),
        )
    }

    fn has_payloads(&self) -> bool {
        self.has_payloads
    }

    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        self.current.as_ref().and_then(|e| e.contexts.as_ref())
    }

    fn has_contexts(&self) -> bool {
        self.has_contexts
    }
}

#[cfg(test)]
mod tests {
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
}
