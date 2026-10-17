//! Unit tests.
#![allow(clippy::arithmetic_side_effects)]

use super::*;
use crate::facets::test_index::{basic_index, collect, match_all, term};
use crate::facets::DocIdSet;
use crate::reader::NO_MORE_DOCS;

fn docs_of(it: &mut dyn DocIdSetIterator) -> Vec<i32> {
    let mut out = Vec::new();
    loop {
        let d = it.next_doc().unwrap();
        if d == NO_MORE_DOCS {
            return out;
        }
        out.push(d);
    }
}

#[test]
fn the_iterator_intersects_hits_iterators_and_the_fast_match_query() {
    let index = basic_index();
    index.with(|searcher, reader, _| {
        let fc = collect(searcher, &match_all(reader), false);
        // Segment 1 holds d2 and d3, as local documents 0 and 1.
        let hits = &fc.matching_docs()[1];

        let plain = FacetCountsWithFilterQuery::default();
        assert!(plain.fast_match_query().is_none());
        let mut only_hits = plain
            .create_iterator(searcher, hits, Vec::new())
            .unwrap()
            .unwrap();
        assert_eq!(docs_of(&mut *only_hits), vec![0, 1]);

        let mut with_it = plain
            .create_iterator(
                searcher,
                hits,
                vec![Some(Box::new(DocIdSet::Array(vec![1]).into_iterator()))],
            )
            .unwrap()
            .unwrap();
        assert_eq!(docs_of(&mut *with_it), vec![1]);
        // A missing iterator (a field the segment lacks): nothing.
        assert!(plain
            .create_iterator(searcher, hits, vec![None])
            .unwrap()
            .is_none());

        let fast = FacetCountsWithFilterQuery::new(Some(term("id", "d3")));
        assert!(fast.fast_match_query().is_some());
        let mut it = fast
            .create_iterator(searcher, hits, Vec::new())
            .unwrap()
            .unwrap();
        assert_eq!(docs_of(&mut *it), vec![1]);
        // The query matches nothing in this segment.
        let other = FacetCountsWithFilterQuery::new(Some(term("id", "d0")));
        assert!(other
            .create_iterator(searcher, hits, Vec::new())
            .unwrap()
            .is_none());
    });
}
