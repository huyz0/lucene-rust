//! Unit tests.

use lucene_index::document::{BinaryDocValuesField, LongRangeDocValuesField};

use super::*;
use crate::facets::test_index::{collect, id_doc, match_all, plain_index, term, FacetIndex};

/// Two segments: a [0, 5], b (none) | c [10, 20].
fn index() -> FacetIndex {
    let mut docs = Vec::new();
    for (id, r) in [("a", Some((0, 5))), ("b", None), ("c", Some((10, 20)))] {
        let mut d = id_doc(id);
        if let Some((lo, hi)) = r {
            d.add(LongRangeDocValuesField::new("r", &[lo], &[hi]).unwrap());
        }
        docs.push(d);
    }
    plain_index(docs, &[1])
}

fn ranges() -> Vec<LongRange> {
    vec![
        LongRange::new("low", 0, true, 3, true).unwrap(),
        LongRange::new("high", 15, true, 30, true).unwrap(),
        LongRange::new("never", 100, true, 200, true).unwrap(),
    ]
}

#[test]
fn counts_intersecting_ranges_and_refuses_what_java_does() {
    index().with(|searcher, reader, _| {
        let all = collect(searcher, &match_all(reader), false);
        let f = LongRangeOnRangeFacetCounts::new("r", &all, RangeQueryType::Intersects, &ranges())
            .unwrap();
        let r = f.get_all_children("r", &[]).unwrap().unwrap();
        // b has no range.
        assert_eq!(r.value, FacetValue::Int(2));
        let top = f.get_top_children(10, "r", &[]).unwrap().unwrap();
        assert_eq!(top.child_count, 2);
        assert!(matches!(
            f.get_specific_value("r", &["low"]),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(f.get_all_dims(1).unwrap()[0].label_values.len(), 1);
        assert!(f.get_all_dims(0).is_err());
        assert!(format!("{f:?}").contains("tot_count: 2"));

        // Only `c`: the first segment has none of the query's documents.
        let fast = LongRangeOnRangeFacetCounts::with_fast_match(
            "r",
            Some(searcher),
            &all,
            RangeQueryType::Intersects,
            Some(term("id", "c")),
            &ranges(),
        )
        .unwrap();
        assert_eq!(
            fast.get_all_children("r", &[]).unwrap().unwrap().value,
            FacetValue::Int(1)
        );
        assert!(matches!(
            LongRangeOnRangeFacetCounts::with_fast_match(
                "r",
                None,
                &all,
                RangeQueryType::Intersects,
                Some(term("id", "c")),
                &ranges()
            ),
            Err(Error::IllegalState(_))
        ));
        assert!(matches!(
            LongRangeOnRangeFacetCounts::new("r", &all, RangeQueryType::Within, &[]),
            Err(Error::IllegalArgument(m)) if m == "ranges cannot be empty"
        ));
        let mixed = [
            LongRange::new("a", 0, true, 1, true).unwrap(),
            LongRange::with_dims("b", vec![0, 0], vec![1, 1]).unwrap(),
        ];
        assert!(matches!(
            LongRangeOnRangeFacetCounts::new("r", &all, RangeQueryType::Within, &mixed),
            Err(Error::IllegalArgument(m)) if m == "all ranges must have 1 dimensions, found 2"
        ));
        let d = [DoubleRange::new("d", 0.0, true, 1.0, true).unwrap()];
        let f = DoubleRangeOnRangeFacetCounts::new("nofield", &all, RangeQueryType::Contains, &d)
            .unwrap();
        assert!(format!("{f:?}").starts_with("DoubleRangeOnRangeFacetCounts"));
        // Two-dimensional ranges over one-dimensional values.
        let two = [LongRange::with_dims("b", vec![0, 0], vec![1, 1]).unwrap()];
        assert!(matches!(
            LongRangeOnRangeFacetCounts::new("r", &all, RangeQueryType::Within, &two),
            Err(Error::IllegalState(m)) if m.contains("shorter than the ranges' 32")
        ));
    });
}

#[test]
fn a_binary_value_too_short_is_refused() {
    let mut d = id_doc("x");
    d.add(BinaryDocValuesField::new("r", vec![1u8, 2, 3]));
    plain_index(vec![d], &[]).with(|searcher, reader, _| {
        let all = collect(searcher, &match_all(reader), false);
        assert!(
            LongRangeOnRangeFacetCounts::new("r", &all, RangeQueryType::Intersects, &ranges())
                .is_err()
        );
    });
}
