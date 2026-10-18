//! Boundary and error handling of `FreeTextSuggester` (M13 T13.3c) that the
//! Java-fixture differential tests do not reach.

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_search::suggest::free_text::FreeTextSuggester;
use lucene_search::suggest::input::InputIterator;
use lucene_search::suggest::lookup::Lookup;
use lucene_store::{SliceInput, VecDataOutput};

struct One {
    text: String,
    taken: bool,
}

impl InputIterator for One {
    fn next(&mut self) -> Option<Vec<u8>> {
        if std::mem::replace(&mut self.taken, true) {
            return None;
        }
        Some(self.text.as_bytes().to_vec())
    }
    fn weight(&self) -> i64 {
        1
    }
    fn payload(&self) -> Option<&[u8]> {
        None
    }
    fn has_payloads(&self) -> bool {
        false
    }
    fn contexts(&self) -> Option<&BTreeSet<Vec<u8>>> {
        None
    }
    fn has_contexts(&self) -> bool {
        false
    }
}

fn one(text: &str) -> One {
    One {
        text: text.to_string(),
        taken: false,
    }
}

#[test]
fn a_term_over_the_index_writer_limit_is_refused() {
    let mut s = FreeTextSuggester::new(Arc::new(Analyzer::keyword()));
    let err = s.build(&mut one(&"x".repeat(32767))).unwrap_err();
    assert!(err.to_string().contains("immense term"), "{err}");
    s.build(&mut one(&"x".repeat(32766))).unwrap();
    assert_eq!(s.get_count().unwrap(), 1);
}

#[test]
fn a_saved_model_that_is_cut_short_is_refused() {
    let analyzer = Arc::new(Analyzer::standard(None));
    let mut s = FreeTextSuggester::new(analyzer.clone());
    s.build(&mut one("hello world hello there")).unwrap();
    let mut out = VecDataOutput::new();
    assert!(s.store(&mut out).unwrap());
    let bytes = out.into_inner();
    for cut in [0, 1, 4, 20, bytes.len() / 2, bytes.len() - 1] {
        let mut fresh = FreeTextSuggester::new(analyzer.clone());
        assert!(
            fresh.load(&mut SliceInput::new(&bytes[..cut])).is_err(),
            "cut at {cut}"
        );
    }
    let mut fresh = FreeTextSuggester::new(analyzer);
    assert!(fresh.load(&mut SliceInput::new(&bytes)).unwrap());
    let hits = fresh.lookup("hello ", None, false, 5).unwrap();
    assert!(hits[0].key.starts_with("hello\u{1e}"), "{hits:?}");
}

#[test]
fn a_rebuild_replaces_the_model() {
    let mut s = FreeTextSuggester::new(Arc::new(Analyzer::standard(None)));
    s.build(&mut one("first phrase here")).unwrap();
    s.build(&mut one("second phrase there")).unwrap();
    assert!(s.lookup("first ", None, false, 3).unwrap().is_empty());
    assert!(!s.lookup("second ", None, false, 3).unwrap().is_empty());
}
