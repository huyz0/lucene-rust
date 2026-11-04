//! `ValueSourceGroupSelector` over the function fixture index: the states it
//! reports in the first pass, and in the second once its groups are set.

use std::sync::Arc;

use lucene_grouping::{GroupSelector, GroupState, SearchGroup, ValueSourceGroupSelector};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::function::valuesource::IntFieldSource;
use lucene_search::function::{FunctionContext, TopLevel, ValueSource};
use lucene_search::index_searcher::{IndexSearcher, SegmentNorms};

fn index() -> DirectoryReader {
    let dir =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/function/index");
    DirectoryReader::open(&lucene_store::FsDirectory::open(dir)).unwrap()
}

#[test]
fn value_source_group_selector_states() {
    let reader = index();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let source: Arc<dyn ValueSource> = Arc::new(IntFieldSource::new("i"));
    let norms: Vec<SegmentNorms<'_, '_>> = segments.iter().map(|_| None).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let top = TopLevel::of_searcher(&searcher, None);
    let max_doc = reader.segment_readers()[0].max_doc;
    let context = Arc::new(FunctionContext::create(source.as_ref(), &top).unwrap());
    let mut sel = ValueSourceGroupSelector::new(Arc::clone(&source), Arc::clone(&context));
    // Moved before a segment: Java's NullPointerException.
    assert!(matches!(
        sel.advance_to(0, 1.0),
        Err(lucene_search::Error::IllegalState(_))
    ));
    // Without the searcher's leaves the segment stands alone.
    sel.set_next_reader(0, &segments[0]).unwrap();
    let mut present = None;
    let mut missing = None;
    for doc in 0..max_doc {
        match sel.advance_to(doc, 1.0).unwrap() {
            GroupState::Accept => {
                present.get_or_insert((doc, sel.current_value().cloned().unwrap()));
            }
            GroupState::Skip => {
                assert!(sel.current_value().is_none());
                missing.get_or_insert(doc);
            }
        }
    }
    let (doc, value) = present.unwrap();
    let missing = missing.expect("the fixture has documents without i");
    // The second pass keeps only the chosen groups; the empty group is
    // accepted once it is one of them. Values are read forward, so each
    // pass enters the segment again.
    sel.set_groups(&[SearchGroup {
        group_value: Some(value.clone()),
        sort_values: Vec::new(),
    }]);
    sel.set_next_reader(0, &segments[0]).unwrap();
    let mut others = 0;
    for d in 0..max_doc {
        let state = sel.advance_to(d, 1.0).unwrap();
        if d == doc {
            assert_eq!(state, GroupState::Accept);
        } else if d == missing {
            assert_eq!(state, GroupState::Skip);
        } else if state == GroupState::Skip && sel.current_value().is_some() {
            others += 1;
        }
    }
    assert!(others > 0);
    sel.set_groups(&[SearchGroup {
        group_value: None,
        sort_values: Vec::new(),
    }]);
    sel.set_next_reader(0, &segments[0]).unwrap();
    for d in 0..max_doc {
        let state = sel.advance_to(d, 1.0).unwrap();
        if d == doc {
            assert_eq!(state, GroupState::Skip);
        } else if d == missing {
            assert_eq!(state, GroupState::Accept);
        }
    }
}
