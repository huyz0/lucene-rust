//! Unit tests.

use lucene_index::document::{NumericDocValuesField, SortedNumericDocValuesField};

use super::*;
use crate::facets::test_index::{collect, id_doc, plain_index, term};

#[test]
fn counts_single_and_multi_valued_fields() {
    let mut docs = Vec::new();
    for (id, s, m) in [
        ("a", 3, vec![1, 1, 2]),
        ("b", 3, vec![]),
        ("a", -7, vec![2]),
    ] {
        let mut d = id_doc(id);
        d.add(NumericDocValuesField::new("s", s));
        for v in m {
            d.add(SortedNumericDocValuesField::new("m", v));
        }
        docs.push(d);
    }
    plain_index(docs, &[0]).with(|searcher, reader, _| {
        let f = LongValueFacetCounts::count_all("s", reader).unwrap();
        assert!(f.get_specific_value("s", &["3"]).is_err());
        let all = f.get_all_children("s", &[]).unwrap().unwrap();
        assert_eq!(all.value, FacetValue::Int(3));
        assert_eq!(format!("{f:?}"), f.to_string());
        let sorted = f.get_all_children_sort_by_value();
        assert_eq!(sorted.label_values[0].label, "-7");

        let a = collect(searcher, &term("id", "a"), false);
        let m = LongValueFacetCounts::new("m", &a).unwrap();
        // The first document's 1 counts once.
        let top = m.get_top_children(1, "m", &[]).unwrap().unwrap();
        assert_eq!((top.value, top.child_count), (FacetValue::Int(2), 2));
    });
}
