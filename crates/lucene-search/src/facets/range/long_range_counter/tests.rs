//! Unit tests.

use super::*;

fn r(label: &str, min: i64, max: i64) -> LongRange {
    LongRange::new(label, min, true, max, true).unwrap()
}

#[test]
fn exclusive_ranges_count_single_and_multi_valued_documents() {
    let ranges = [r("a", 0, 9), r("b", 20, 29)];
    assert!(!has_overlapping_ranges(&ranges));
    let mut c = LongRangeCounter::create(&ranges);
    for v in [i64::MIN, 0, 9, 10, 25, i64::MAX] {
        c.add_single_valued(v);
    }
    // One document with values in both ranges, one with none.
    c.start_multi_valued_doc();
    for v in [5, 6, 21] {
        c.add_multi_valued(v);
    }
    assert!(c.end_multi_valued_doc());
    c.start_multi_valued_doc();
    c.add_multi_valued(15);
    assert!(!c.end_multi_valued_doc());
    // MIN, 10 and MAX fell in no range.
    assert_eq!(c.finish(), 3);
    assert_eq!(c.into_counts(), vec![3, 2]);
}

#[test]
fn overlapping_ranges_roll_up_each_document_once() {
    let ranges = [
        r("all", i64::MIN, i64::MAX),
        r("low", 0, 10),
        r("mid", 5, 15),
    ];
    assert!(has_overlapping_ranges(&ranges));
    let mut c = LongRangeCounter::create(&ranges);
    for v in [-1, 7, 12] {
        c.add_single_valued(v);
    }
    assert_eq!(c.finish(), 0);
    c.start_multi_valued_doc();
    c.add_multi_valued(1);
    c.add_multi_valued(2);
    c.add_multi_valued(14);
    assert!(c.end_multi_valued_doc());
    assert_eq!(c.into_counts(), vec![4, 2, 3]);

    let mut none = LongRangeCounter::create(&[r("a", 0, 5), r("b", 3, 9)]);
    // Nothing added: nothing to roll up.
    assert_eq!(none.finish(), 0);
    none.add_single_valued(100);
    assert_eq!(none.finish(), 1);
}

#[test]
fn no_ranges_count_every_value_missing() {
    let mut c = LongRangeCounter::create(&[]);
    c.add_single_valued(1);
    c.start_multi_valued_doc();
    c.add_multi_valued(1);
    assert!(!c.end_multi_valued_doc());
    assert_eq!(c.finish(), 1);
    assert!(c.into_counts().is_empty());
}

#[test]
fn elementary_intervals_split_at_every_end() {
    let iv = build_elementary_intervals(&[r("a", 0, 5), r("b", 3, 3)]);
    assert_eq!(
        iv,
        vec![(i64::MIN, -1), (0, 2), (3, 3), (4, 5), (6, i64::MAX)]
    );
}
