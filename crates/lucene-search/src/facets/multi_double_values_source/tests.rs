//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::document::{NumericDocValuesField, SortedNumericDocValuesField};

use super::*;
use crate::facets::test_index::{id_doc, plain_index, FacetIndex};
use crate::values_source::from_double_field as single_double;

/// Three documents: `m` {2.5, -0.5}, {}, {NaN}; `s` 1.5, none, -3.25.
fn index() -> FacetIndex {
    let mut docs = Vec::new();
    for (m, s) in [
        (vec![2.5, -0.5], Some(1.5)),
        (vec![], None),
        (vec![f64::NAN], Some(-3.25)),
    ] {
        let mut d = id_doc("x");
        for v in m {
            d.add(SortedNumericDocValuesField::new("m", v.to_bits() as i64));
        }
        if let Some(s) = s {
            d.add(NumericDocValuesField::new("s", f64::to_bits(s) as i64));
        }
        docs.push(d);
    }
    plain_index(docs, &[])
}

fn read_all(v: &mut dyn MultiDoubleValues) -> Vec<Option<Vec<u64>>> {
    (0..3)
        .map(|doc| {
            v.advance_exact(doc).unwrap().then(|| {
                (0..v.get_value_count())
                    .map(|_| v.next_value().unwrap().to_bits())
                    .collect()
            })
        })
        .collect()
}

fn bits(v: &[f64]) -> Option<Vec<u64>> {
    Some(v.iter().map(|d| d.to_bits()).collect())
}

#[test]
fn field_and_single_valued_sources_read_each_documents_values() {
    index().with(|searcher, _, _| {
        let ctx = ValuesContext::new(searcher);
        let m = from_double_field("m");
        assert_eq!(m.describe(), "multi-double(m)");
        assert_eq!(format!("{:?}", m.as_ref()), "multi-double(m)");
        assert!(m.is_cacheable(&ctx, 0));
        assert!(unwrap_singleton(m.as_ref()).is_none());
        let mut v = m.get_values(&ctx, 0).unwrap();
        assert_eq!(
            read_all(v.as_mut()),
            vec![bits(&[-0.5, 2.5]), None, bits(&[f64::NAN])]
        );
        assert_eq!(from_int_field("m").describe(), "multi-double(m)");
        assert_eq!(from_float_field("m").describe(), "multi-double(m)");

        let s = from_single_valued(single_double("s"));
        assert_eq!(s.describe(), "multi-double(double(s))");
        assert!(s.is_cacheable(&ctx, 0));
        assert_eq!(
            unwrap_singleton(s.as_ref()).unwrap().describe(),
            "double(s)"
        );
        let mut v = s.get_values(&ctx, 0).unwrap();
        assert_eq!(
            read_all(v.as_mut()),
            vec![bits(&[1.5]), None, bits(&[-3.25])]
        );
    });
}

#[test]
fn long_conversions_cast_or_sort_each_value() {
    index().with(|searcher, _, _| {
        let ctx = ValuesContext::new(searcher);
        let cast = to_multi_long_values_source(from_double_field("m"));
        assert_eq!(cast.describe(), "multi-double(multi-double(m))");
        assert!(cast.is_cacheable(&ctx, 0));
        let mut v = cast.get_values(&ctx, 0).unwrap();
        // -0.5 truncates to 0, NaN casts to 0.
        assert!(v.advance_exact(0).unwrap());
        assert_eq!((v.next_value().unwrap(), v.next_value().unwrap()), (0, 2));
        assert!(!v.advance_exact(1).unwrap());
        assert!(v.advance_exact(2).unwrap());
        assert_eq!((v.get_value_count(), v.next_value().unwrap()), (1, 0));

        let sortable = to_sortable_multi_long_values_source(from_double_field("m"));
        assert_eq!(sortable.describe(), "sortableMultiLong(multi-double(m))");
        assert!(sortable.is_cacheable(&ctx, 0));
        let mut v = sortable.get_values(&ctx, 0).unwrap();
        assert!(v.advance_exact(0).unwrap());
        assert_eq!(v.get_value_count(), 2);
        assert_eq!(v.next_value().unwrap(), double_to_sortable_long(-0.5));

        let mut v = sortable_longs(from_double_field("m").get_values(&ctx, 0).unwrap());
        assert!(v.advance_exact(2).unwrap());
        assert_eq!(v.next_value().unwrap(), double_to_sortable_long(f64::NAN));
        assert!(cast.get_values(&ctx, 9).is_err());
    });
}
