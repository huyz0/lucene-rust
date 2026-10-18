//! Unit tests.

use lucene_index::document::{NumericDocValuesField, SortedNumericDocValuesField};

use super::*;
use crate::facets::multi_long_values_source::from_long_field;
use crate::facets::test_index::{collect, id_doc, match_all, plain_index, term, FacetIndex};
use crate::values_source::long_from_long_field;
use crate::Error;

/// Three segments: a(s=1, m={1,20}), b(s=5) | b(s=7, m={7}) | c(s=9, m={9}).
fn index() -> FacetIndex {
    let mut docs = Vec::new();
    for (id, s, m) in [
        ("a", 1, vec![1, 20]),
        ("b", 5, vec![]),
        ("b", 7, vec![7]),
        ("c", 9, vec![9]),
    ] {
        let mut d = id_doc(id);
        d.add(NumericDocValuesField::new("s", s));
        for v in m {
            d.add(SortedNumericDocValuesField::new("m", v));
        }
        docs.push(d);
    }
    plain_index(docs, &[1, 2])
}

fn ranges() -> Vec<LongRange> {
    vec![
        LongRange::new("low", 0, true, 5, true).unwrap(),
        LongRange::new("high", 5, false, 100, true).unwrap(),
    ]
}

fn counts(f: &LongRangeFacetCounts, dim: &str) -> (FacetValue, Vec<i64>) {
    let r = f.get_all_children(dim, &[]).unwrap().unwrap();
    (
        r.value,
        r.label_values
            .iter()
            .map(|l| i64::from(l.value.int_value()))
            .collect(),
    )
}

#[test]
fn a_fast_match_query_skips_the_segments_it_misses() {
    index().with(|searcher, reader, _| {
        let all = collect(searcher, &match_all(reader), false);
        // Single-valued: Java adds each segment's hits (both of the first
        // segment's), not the query's.
        let f = LongRangeFacetCounts::with_fast_match(
            "s",
            Some(searcher),
            &all,
            Some(term("id", "a")),
            &ranges(),
        )
        .unwrap();
        assert_eq!(counts(&f, "s"), (FacetValue::Int(2), vec![1, 0]));
        assert!(format!("{f:?}").starts_with("RangeFacetCounts totCount=2"));
        // Multi-valued.
        let f = LongRangeFacetCounts::with_fast_match(
            "m",
            Some(searcher),
            &all,
            Some(term("id", "a")),
            &ranges(),
        )
        .unwrap();
        assert_eq!(counts(&f, "m"), (FacetValue::Int(1), vec![1, 1]));
        let ctx = ValuesContext::new(searcher);
        let f = LongRangeFacetCounts::with_multi_values_source(
            "m",
            from_long_field("m").as_ref(),
            &ctx,
            &all,
            Some(term("id", "c")),
            &ranges(),
        )
        .unwrap();
        assert_eq!(counts(&f, "m"), (FacetValue::Int(1), vec![0, 1]));
        let f = LongRangeFacetCounts::with_values_source(
            "s",
            long_from_long_field("s").as_ref(),
            &ctx,
            &all,
            Some(term("id", "c")),
            &ranges(),
        )
        .unwrap();
        assert_eq!(counts(&f, "s"), (FacetValue::Int(1), vec![0, 1]));

        // A query needs the searcher.
        for field in ["s", "m"] {
            assert!(matches!(
                LongRangeFacetCounts::with_fast_match(
                    field,
                    None,
                    &all,
                    Some(term("id", "a")),
                    &ranges()
                ),
                Err(Error::IllegalState(_))
            ));
        }
    });
}

#[test]
fn segments_without_hits_are_skipped() {
    index().with(|searcher, _, _| {
        let c = collect(searcher, &term("id", "c"), false);
        let f = LongRangeFacetCounts::new("s", &c, &ranges()).unwrap();
        assert_eq!(counts(&f, "s"), (FacetValue::Int(1), vec![0, 1]));
        let f = LongRangeFacetCounts::new("m", &c, &ranges()).unwrap();
        assert_eq!(counts(&f, "m"), (FacetValue::Int(1), vec![0, 1]));
        let none = collect(searcher, &term("id", "zzz"), false);
        let f = LongRangeFacetCounts::new("s", &none, &ranges()).unwrap();
        assert_eq!(counts(&f, "s"), (FacetValue::Int(0), vec![0, 0]));
    });
}

#[test]
fn the_hits_alone_count_missing_values_out_of_the_total() {
    // `n` NUMERIC and `o` SORTED_NUMERIC with one value each, both missing
    // from the third document: the single-valued walk over the hits.
    let mut docs = Vec::new();
    for (id, v) in [("a", Some(3)), ("b", Some(50)), ("c", None)] {
        let mut d = id_doc(id);
        if let Some(v) = v {
            d.add(NumericDocValuesField::new("n", v));
            d.add(SortedNumericDocValuesField::new("o", v));
        }
        docs.push(d);
    }
    plain_index(docs, &[]).with(|searcher, reader, _| {
        let all = collect(searcher, &match_all(reader), false);
        for field in ["n", "o"] {
            let f = LongRangeFacetCounts::new(field, &all, &ranges()).unwrap();
            assert_eq!(
                counts(&f, field),
                (FacetValue::Int(2), vec![1, 1]),
                "{field}"
            );
        }
    });
}
