//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use lucene_index::document::{NumericDocValuesField, SortedNumericDocValuesField};

use super::*;
use crate::facets::test_index::{id_doc, plain_index, FacetIndex};
use crate::values_source::long_from_long_field;

/// Three documents: `m` {3, -1}, {}, {7}; `s` 5, none, -2.
fn index() -> FacetIndex {
    let mut docs = Vec::new();
    for (m, s) in [(vec![3, -1], Some(5)), (vec![], None), (vec![7], Some(-2))] {
        let mut d = id_doc("x");
        for v in m {
            d.add(SortedNumericDocValuesField::new("m", v));
        }
        if let Some(s) = s {
            d.add(NumericDocValuesField::new("s", s));
        }
        docs.push(d);
    }
    plain_index(docs, &[])
}

/// Each document's values, `None` for one without.
fn read_all(v: &mut dyn MultiLongValues, max_doc: i32) -> Vec<Option<Vec<i64>>> {
    (0..max_doc)
        .map(|doc| {
            v.advance_exact(doc).unwrap().then(|| {
                (0..v.get_value_count())
                    .map(|_| v.next_value().unwrap())
                    .collect()
            })
        })
        .collect()
}

#[test]
fn field_and_single_valued_sources_read_each_documents_values() {
    index().with(|searcher, _, _| {
        let ctx = ValuesContext::new(searcher);
        let m = from_long_field("m");
        assert_eq!(m.describe(), "multi-long(m)");
        assert_eq!(format!("{:?}", m.as_ref()), "multi-long(m)");
        assert!(m.is_cacheable(&ctx, 0));
        assert!(unwrap_singleton(m.as_ref()).is_none());
        let mut v = m.get_values(&ctx, 0).unwrap();
        assert_eq!(
            read_all(v.as_mut(), 3),
            vec![Some(vec![-1, 3]), None, Some(vec![7])]
        );
        assert_eq!(from_int_field("m").describe(), "multi-long(m)");

        let s = from_single_valued(long_from_long_field("s"));
        assert_eq!(s.describe(), "multi-long(long(s))");
        assert!(s.is_cacheable(&ctx, 0));
        assert_eq!(unwrap_singleton(s.as_ref()).unwrap().describe(), "long(s)");
        let mut v = s.get_values(&ctx, 0).unwrap();
        assert_eq!(
            read_all(v.as_mut(), 3),
            vec![Some(vec![5]), None, Some(vec![-2])]
        );
    });
}

#[test]
fn to_doubles_casts_each_value() {
    index().with(|searcher, _, _| {
        let ctx = ValuesContext::new(searcher);
        let d = to_multi_double_values_source(from_long_field("m"));
        assert_eq!(d.describe(), "multi-long(multi-long(m))");
        assert!(d.is_cacheable(&ctx, 0));
        let mut v = d.get_values(&ctx, 0).unwrap();
        assert!(v.advance_exact(0).unwrap());
        assert_eq!(v.get_value_count(), 2);
        assert_eq!(
            (v.next_value().unwrap(), v.next_value().unwrap()),
            (-1.0, 3.0)
        );
        assert!(!v.advance_exact(1).unwrap());
        // A leaf past the reader's.
        assert!(d.get_values(&ctx, 9).is_err());
    });
}
