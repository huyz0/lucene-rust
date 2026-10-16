//! Boundary and error handling of the FST suggesters (M13 T13.3b) that the
//! Java-fixture differential tests do not reach: unbuilt lookups, builder
//! limits, unsupported inputs, and the sorters.

use std::collections::BTreeSet;

use lucene_search::suggest::fst_completion::{
    BytesRefSorter, FstCompletion, FstCompletionBuilder, FstCompletionLookup, InMemorySorter,
};
use lucene_search::suggest::input::{InputIterator, InputIteratorWrapper, UnsortedInputIterator};
use lucene_search::suggest::lookup::Lookup;
use lucene_search::suggest::wfst::WfstCompletionLookup;
use lucene_store::SliceInput;

/// An iterator that claims payloads or contexts.
struct Claims {
    payloads: bool,
    contexts: bool,
}

impl InputIterator for Claims {
    fn next(&mut self) -> Option<Vec<u8>> {
        None
    }
    fn weight(&self) -> i64 {
        0
    }
    fn payload(&self) -> Option<&[u8]> {
        None
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

fn message(e: lucene_store::Error) -> String {
    e.to_string()
}

#[test]
fn sorter_refuses_additions_after_iteration() {
    let mut s = InMemorySorter::natural();
    s.add(b"b").unwrap();
    s.add(b"a").unwrap();
    assert_eq!(s.iterator().unwrap(), vec![b"a".to_vec(), b"b".to_vec()]);
    assert!(message(s.add(b"c").unwrap_err()).contains("IllegalStateException"));
}

#[test]
fn builder_limits_and_empty_completion() {
    let mut b = FstCompletionBuilder::default();
    assert!(message(b.add(b"x", 10).unwrap_err()).contains("Bucket outside of the allowed range"));
    assert!(message(b.add(b"x", -1).unwrap_err()).contains("[0, 10)"));
    let empty = b.build().unwrap();
    assert_eq!(empty.bucket_count(), 0);
    assert_eq!(empty.bucket_of("x").unwrap(), -1);
    assert!(empty.lookup_all("x").unwrap().is_empty());
    assert!(empty.fst_handle().is_none());

    for bad in [0, 256] {
        let err = FstCompletionBuilder::with_sorter(bad, Box::new(InMemorySorter::natural()), 1.0)
            .err()
            .unwrap();
        assert!(message(err).contains("Buckets must be >= 1 and <= 255"));
    }
}

#[test]
fn completion_display_and_empty_key() {
    let mut b = FstCompletionBuilder::new();
    b.add(b"foo", 3).unwrap();
    b.add(b"foobar", 7).unwrap();
    let c: FstCompletion = b.build().unwrap();
    assert!(c.lookup_all("").unwrap().is_empty());
    let all = c.lookup_all("foo").unwrap();
    let shown: Vec<String> = all.iter().map(|c| c.to_display()).collect();
    assert!(shown.contains(&"foobar/7".to_string()), "{shown:?}");
    assert_eq!(c.bucket_of("foo").unwrap(), 3);

    // a lookup wrapped around an FST built elsewhere answers from it
    let wrapped = FstCompletionLookup::from_completion(&c, true).unwrap();
    assert_eq!(wrapped.get("foo").unwrap(), Some(3));
    assert_eq!(wrapped.get("fo").unwrap(), None);
    let hits = wrapped.lookup("foo", None, false, 5).unwrap();
    assert_eq!(hits.len(), 2);
}

#[test]
fn fst_completion_lookup_unbuilt_and_unsupported() {
    let l = FstCompletionLookup::new();
    assert!(message(l.get("a").unwrap_err()).contains("not built"));
    assert!(message(l.lookup("a", None, false, 1).unwrap_err()).contains("not built"));
    assert!(message(l.lookup("a", None, true, 1).unwrap_err()).contains("not built"));
    let mut out = lucene_store::VecDataOutput::new();
    // nothing built: the count is written, the FST is not
    assert!(!l.store(&mut out).unwrap());
    assert_eq!(l.get_count().unwrap(), 0);

    let mut l = FstCompletionLookup::with_options(4, false);
    for (claims, refused) in [
        (
            Claims {
                payloads: true,
                contexts: false,
            },
            "doesn't support payloads",
        ),
        (
            Claims {
                payloads: false,
                contexts: true,
            },
            "doesn't support contexts",
        ),
    ] {
        let mut claims = claims;
        assert!(message(l.build(&mut claims).unwrap_err()).contains(refused));
    }
    // contexts are refused before the unbuilt state is noticed
    let ctx = BTreeSet::new();
    let err = message(l.lookup("a", Some(&ctx), false, 1).unwrap_err());
    assert!(err.contains("doesn't support contexts"), "{err}");
}

#[test]
fn wfst_unsupported_inputs_and_unbuilt_state() {
    let mut l = WfstCompletionLookup::default();
    assert_eq!(l.get("a").unwrap(), None);
    assert!(l.lookup("a", None, false, 3).unwrap().is_empty());
    let mut out = lucene_store::VecDataOutput::new();
    assert!(!l.store(&mut out).unwrap());
    for (claims, refused) in [
        (
            Claims {
                payloads: true,
                contexts: false,
            },
            "doesn't support payloads",
        ),
        (
            Claims {
                payloads: false,
                contexts: true,
            },
            "doesn't support contexts",
        ),
    ] {
        let mut claims = claims;
        assert!(message(l.build(&mut claims).unwrap_err()).contains(refused));
    }
    let mut terms = InputIteratorWrapper::new(vec![b"abc".to_vec()].into_iter());
    l.build(&mut terms).unwrap();
    assert_eq!(l.get_count().unwrap(), 1);
    let err = message(l.lookup("a", None, true, 1).unwrap_err());
    assert!(err.contains("onlyMorePopular=false"), "{err}");
    let ctx = BTreeSet::new();
    let err = message(l.lookup("a", Some(&ctx), false, 1).unwrap_err());
    assert!(err.contains("doesn't support contexts"), "{err}");

    // a stored lookup of garbage is refused rather than trusted
    let junk = [1u8, 0xff, 0xff, 0xff, 0xff];
    assert!(l.load(&mut SliceInput::new(&junk)).is_err());
}

#[test]
fn java_random_refuses_a_bound_below_one() {
    let mut r = lucene_search::suggest::input::JavaRandom::new(42);
    for bound in [0, -1, i32::MIN] {
        assert!(message(r.next_int(bound).unwrap_err()).contains("bound must be positive"));
    }
    assert!((0..7).contains(&r.next_int(7).unwrap()));
}

#[test]
fn unsorted_iterator_seeded_from_the_clock_keeps_every_entry() {
    let mut source = InputIteratorWrapper::new(
        ["a", "b", "c", "d"]
            .iter()
            .map(|s| s.as_bytes().to_vec())
            .collect::<Vec<_>>()
            .into_iter(),
    );
    let mut shuffled = UnsortedInputIterator::new(&mut source);
    assert!(!shuffled.has_payloads());
    assert!(!shuffled.has_contexts());
    assert!(shuffled.contexts().is_none());
    let mut seen = Vec::new();
    while let Some(t) = shuffled.next() {
        seen.push(String::from_utf8(t).unwrap());
    }
    seen.sort();
    assert_eq!(seen, ["a", "b", "c", "d"]);
}
