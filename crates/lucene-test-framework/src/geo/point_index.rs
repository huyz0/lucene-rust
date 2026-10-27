//! The point bases' index and searches: Java's `indexPoints`,
//! `searchIndex` and `searcher.count`, over one segment written by a plain
//! `IndexWriter` (deletes in one last commit), so a doc id is the id it
//! was written with.

use std::collections::BTreeSet;
use std::sync::Arc;

use lucene_analysis::Analyzer;
use lucene_index::document::{Document, NumericDocValuesField, Store, StringField};
use lucene_search::collector::ScoreMode;
use lucene_search::document::{DocumentQuery, MatchNoDocs};
use lucene_search::extended_query::ExtendedQuery;
use lucene_search::query::Clause;
use lucene_util::java_random::JavaRandom;

use crate::search::base_explanation::top_level;
use crate::search::check_hits;
use crate::search::test_index::TestIndex;

/// One segment of `docs`, deleting by `(field, term)` in a last commit.
pub(crate) fn build(docs: Vec<Document>, deletes: &[(&str, &str)]) -> TestIndex {
    TestIndex::build(Arc::new(Analyzer::standard(None)), &[docs], deletes, &[])
}

/// `indexPoints`: `n` documents with the `id` field as a term and a doc
/// value, `add_point` adding id's point unless it is missing, deletes drawn
/// as Java draws them, then the `forceMerge` coin (ignored). Returns the
/// index and the deleted ids.
pub(crate) fn index_points(
    random: &mut JavaRandom,
    n: usize,
    mut add_point: impl FnMut(usize, &mut Document),
) -> (TestIndex, BTreeSet<usize>) {
    let mut deleted = BTreeSet::new();
    let mut written = Vec::with_capacity(n);
    for id in 0..n {
        let mut doc = Document::new();
        doc.add(StringField::new("id", id.to_string(), Store::No));
        doc.add(NumericDocValuesField::new(
            "id",
            i64::try_from(id).expect("small"),
        ));
        add_point(id, &mut doc);
        written.push(doc);
        if id > 0 && random.next_int_bounded(100) == 42 {
            let bound = i32::try_from(id).expect("an int bound");
            deleted.insert(usize::try_from(random.next_int_bounded(bound)).expect("non-negative"));
        }
    }
    let _force_merge = random.next_boolean();
    let ids: Vec<String> = deleted.iter().map(ToString::to_string).collect();
    let deletes: Vec<(&str, &str)> = ids.iter().map(|id| ("id", id.as_str())).collect();
    (build(written, &deletes), deleted)
}

/// The doc ids `q` matches.
pub(crate) fn hits(index: &TestIndex, q: Clause) -> BTreeSet<usize> {
    let q = top_level(q);
    index.with_searcher(|s| {
        check_hits::all_hits(s, &q, ScoreMode::CompleteNoScores)
            .expect("search")
            .into_iter()
            .map(|(doc, _)| usize::try_from(doc).expect("a doc id"))
            .collect()
    })
}

/// `searcher.count(q)` over one segment of `docs`.
pub(crate) fn count(docs: Vec<Document>, q: Clause) -> usize {
    hits(&build(docs, &[]), q).len()
}

/// Checks one query's hits over ids `0..n` against `expected(id)` for every
/// id neither deleted nor `missing`.
///
/// # Panics
/// Listing the wrong ids and `what`, as Java's "some hits were wrong".
pub(crate) fn check(
    index: &TestIndex,
    deleted: &BTreeSet<usize>,
    n: usize,
    q: Clause,
    what: &str,
    missing: impl Fn(usize) -> bool,
    expected: impl Fn(usize) -> bool,
) {
    let hits = hits(index, q);
    let wrong: Vec<usize> = (0..n)
        .filter(|&id| {
            let want = !deleted.contains(&id) && !missing(id) && expected(id);
            hits.contains(&id) != want
        })
        .collect();
    assert!(
        wrong.is_empty(),
        "some hits were wrong: ids {wrong:?} for {what}"
    );
}

/// Asserts `result` failed with a message containing every one of
/// `parts` (Java's `expectThrows(IllegalArgumentException)` and its
/// message checks).
///
/// # Panics
/// When `result` is `Ok`, or its message lacks a part.
pub(crate) fn assert_rejected<T>(result: Result<T, String>, parts: &[&str]) {
    let Err(message) = result else {
        panic!("expected an IllegalArgumentException containing {parts:?}");
    };
    for part in parts {
        assert!(message.contains(part), "{message:?} lacks {part:?}");
    }
}

/// `searchSmallSet`'s documents after the points: Java's loop draws its
/// bound again every iteration.
pub(crate) fn small_set_string_docs(random: &mut JavaRandom, docs: &mut Vec<Document>) {
    let mut i = 0;
    while i < random.next_int_bounded(10) {
        let mut doc = Document::new();
        doc.add(StringField::new("string", i.to_string(), Store::No));
        docs.push(doc);
        i += 1;
    }
}

/// The document-package query a clause wraps, if it wraps one.
fn document_query(c: &Clause) -> Option<&dyn DocumentQuery> {
    match c {
        Clause::Extended(e) => match e.as_ref() {
            ExtendedQuery::Document(d) => Some(&*d.query),
            _ => None,
        },
        _ => None,
    }
}

/// `Query.equals` over clauses: a document-package query by its own
/// [`DocumentQuery::equals`], anything else by `==`.
pub(crate) fn query_equals(a: &Clause, b: &Clause) -> bool {
    match (document_query(a), document_query(b)) {
        (Some(x), Some(y)) => x.equals(y),
        _ => a == b,
    }
}

/// `q instanceof MatchNoDocsQuery`.
pub(crate) fn is_match_no_docs(q: &Clause) -> bool {
    document_query(q).is_some_and(|d| d.as_any_query().is::<MatchNoDocs>())
}
