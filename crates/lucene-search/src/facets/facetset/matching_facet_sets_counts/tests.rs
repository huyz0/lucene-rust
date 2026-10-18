//! Unit tests.

use lucene_index::document::{BinaryDocValuesField, Document};

use super::*;
use crate::facets::facetset::DimRange;
use crate::facets::facetset::{
    decode_longs, ExactFacetSetMatcher, FacetSetsField, LongFacetSet, RangeFacetSetMatcher,
};
use crate::facets::test_index::{collect, id_doc, match_all, plain_index, term, FacetIndex};

fn doc(id: &str, sets: &[&[i64]]) -> Document {
    let mut d = id_doc(id);
    if !sets.is_empty() {
        let sets: Vec<LongFacetSet> = sets
            .iter()
            .map(|s| LongFacetSet::new(s.to_vec()).unwrap())
            .collect();
        let refs: Vec<&dyn crate::facets::facetset::FacetSet> = sets
            .iter()
            .map(|s| s as &dyn crate::facets::facetset::FacetSet)
            .collect();
        d.add(FacetSetsField::create("fs", &refs).unwrap());
    }
    d
}

/// Two segments: {1,2}+{3,4}, {1,2}, (none) | {5,6}.
fn index() -> FacetIndex {
    plain_index(
        vec![
            doc("a", &[&[1, 2], &[3, 4]]),
            doc("b", &[&[1, 2]]),
            doc("c", &[]),
            doc("a", &[&[5, 6]]),
        ],
        &[2],
    )
}

fn raw_index(value: Vec<u8>) -> FacetIndex {
    let mut d = id_doc("x");
    d.add(BinaryDocValuesField::new("fs", value));
    plain_index(vec![d], &[])
}

#[test]
fn counts_each_matcher_and_the_documents_any_matched() {
    let e12 = ExactFacetSetMatcher::new("e12", &LongFacetSet::new(vec![1, 2]).unwrap()).unwrap();
    let big =
        RangeFacetSetMatcher::new("big", &[DimRange::new(3, 9), DimRange::new(0, 9)]).unwrap();
    let none = ExactFacetSetMatcher::new("none", &LongFacetSet::new(vec![0, 0]).unwrap()).unwrap();
    let ms: [&dyn FacetSetMatcher; 3] = [&e12, &big, &none];
    index().with(|searcher, reader, _| {
        let fc = collect(searcher, &match_all(reader), false);
        let f = MatchingFacetSetsCounts::new("fs", &fc, &decode_longs, &ms).unwrap();
        let all = f.get_all_children("fs", &[]).unwrap().unwrap();
        assert_eq!(all.value, FacetValue::Int(3));
        assert_eq!(all.child_count, 3);
        let top = f.get_top_children(1, "fs", &[]).unwrap().unwrap();
        assert_eq!(top.child_count, 2);
        assert_eq!(top.label_values[0].label, "big");
        assert!(f.get_all_children("other", &[]).is_err());
        assert!(matches!(
            f.get_specific_value("fs", &["x"]),
            Err(Error::Unsupported(_))
        ));
        assert_eq!(f.get_all_dims(5).unwrap().len(), 1);
        assert!(f.get_all_dims(0).is_err());
        assert!(format!("{f:?}").contains("tot_count: 3"));

        // Only the `a` documents, through the searcher.
        let fast = MatchingFacetSetsCounts::with_fast_match(
            "fs",
            Some(searcher),
            &fc,
            &decode_longs,
            Some(term("id", "a")),
            &ms,
        )
        .unwrap();
        let all = fast.get_all_children("fs", &[]).unwrap().unwrap();
        assert_eq!(all.value, FacetValue::Int(2));
        // A query matching nothing in a segment skips it.
        let none = MatchingFacetSetsCounts::with_fast_match(
            "fs",
            Some(searcher),
            &fc,
            &decode_longs,
            Some(term("id", "b")),
            &ms,
        )
        .unwrap();
        assert_eq!(
            none.get_all_children("fs", &[]).unwrap().unwrap().value,
            FacetValue::Int(1)
        );
        assert!(matches!(
            MatchingFacetSetsCounts::with_fast_match(
                "fs",
                None,
                &fc,
                &decode_longs,
                Some(term("id", "a")),
                &ms
            ),
            Err(Error::IllegalState(_))
        ));
    });
}

#[test]
fn a_value_without_a_dimension_count_is_refused() {
    let e = ExactFacetSetMatcher::new("e", &LongFacetSet::new(vec![1]).unwrap()).unwrap();
    for (value, want) in [
        (vec![0u8, 0], "has no dimension count"),
        (
            vec![0x7f, 0xff, 0xff, 0xff],
            "negative facet set dimensions",
        ),
        (
            vec![0xff, 0xff, 0xff, 0xff, 0, 0, 0, 0],
            "cannot hold 2147483647 dimensions",
        ),
        (
            vec![0x80, 0, 0, 0, 1, 2, 3, 4],
            "no progress decoding at byte 4",
        ),
    ] {
        raw_index(value).with(|searcher, reader, _| {
            let fc = collect(searcher, &match_all(reader), false);
            let r = MatchingFacetSetsCounts::new("fs", &fc, &decode_longs, &[&e]);
            assert!(matches!(r, Err(Error::IllegalState(m)) if m.contains(want)));
        });
    }
}
