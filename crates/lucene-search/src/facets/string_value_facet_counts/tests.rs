//! Unit tests.

use lucene_index::document::SortedSetDocValuesField;

use super::*;
use crate::facets::test_index::{id_doc, plain_index};

#[test]
fn an_empty_reader_counts_nothing() {
    plain_index(Vec::new(), &[]).with(|_, reader, _| {
        let state = StringDocValuesReaderState::new(reader, "f").unwrap();
        assert_eq!(state.field(), "f");
        assert_eq!(
            format!("{state:?}"),
            "StringDocValuesReaderState(field=f reader=0 leaves)"
        );
        let f = StringValueFacetCounts::new(&state).unwrap();
        assert_eq!(
            f.get_specific_value("f", &["x"]).unwrap(),
            FacetValue::Int(-1)
        );
        let top = f.get_top_children(3, "f", &[]).unwrap().unwrap();
        assert_eq!((top.child_count, top.label_values.len()), (0, 0));
        assert!(matches!(f.label(0), Err(Error::IllegalState(_))));
        assert!(
            format!("{f:?}").starts_with("StringValueFacetCounts { field: \"f\", cardinality: 0")
        );
    });
}

#[test]
fn counts_terms_across_segments() {
    let mut docs = Vec::new();
    for vals in [&["b", "a"][..], &["a"], &["c"]] {
        let mut d = id_doc("x");
        for v in vals {
            d.add(SortedSetDocValuesField::new("f", v.as_bytes().to_vec()));
        }
        docs.push(d);
    }
    plain_index(docs, &[1]).with(|_, reader, _| {
        let state = StringDocValuesReaderState::new(reader, "f").unwrap();
        assert!(format!("{state:?}").ends_with("reader=2 leaves)"));
        let f = StringValueFacetCounts::new(&state).unwrap();
        assert_eq!(
            f.get_specific_value("f", &["a"]).unwrap(),
            FacetValue::Int(2)
        );
        assert!(matches!(f.label(99), Err(Error::IllegalState(_))));
    });
}

#[test]
fn bad_ordinals_and_leaves_are_illegal_state() {
    assert!(
        matches!(bad_leaf(3), Error::IllegalState(m) if m == "leaf 3 is not in the ordinal map")
    );
    assert!(matches!(global(&[0, 2], 5), Err(Error::IllegalState(_))));
    let (mut dense, mut sparse) = (Some(vec![0; 2]), None);
    assert!(increment(&mut dense, &mut sparse, 7, 1).is_err());
    assert!(increment(&mut dense, &mut sparse, i64::MAX, 1).is_err());
    increment(&mut dense, &mut sparse, 1, 3).unwrap();
    assert_eq!(dense, Some(vec![0, 3]));
    let (mut none, mut sparse) = (None, Some(FxHashMap::default()));
    increment(&mut none, &mut sparse, 9, 2).unwrap();
    assert_eq!(sparse.unwrap().get(&9), Some(&2));
}
