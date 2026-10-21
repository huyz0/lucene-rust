//! `MatchesTestBase` over `TestMatchesIterator`'s documents, with that
//! suite's expected positions and offsets: each check passes on the
//! matches Lucene reports, and fails on matches it does not.

use std::panic::{catch_unwind, AssertUnwindSafe};

use lucene_search::query::{BooleanQuery, Clause, PhraseQuery, SpanQuery, TermQuery};
use lucene_test_framework::search::matches_test_base::{
    MatchesTestBase, TermMatch, FIELD_DOCS_ONLY, FIELD_FREQS, FIELD_NO_OFFSETS, FIELD_WITH_OFFSETS,
};
use lucene_util::java_random::JavaRandom;

const DOCS: [&str; 6] = [
    "w1 w2 w3 w4 w5",
    "w1 w3 w2 w3 zz",
    "w1 xx w2 yy w4",
    "w1 w2 w1 w4 w2 w3",
    "a phrase sentence with many phrase sentence iterations of a phrase sentence",
    "nothing matches this document",
];

/// `"phrase sentence"`, in order, as spans: matches with term sub-matches.
fn span_phrase() -> Clause {
    Clause::Span(SpanQuery::span_near(
        [
            SpanQuery::span_term(FIELD_WITH_OFFSETS, "phrase"),
            SpanQuery::span_term(FIELD_WITH_OFFSETS, "sentence"),
        ],
        0,
        true,
    ))
}

fn base() -> MatchesTestBase {
    MatchesTestBase::new(&mut JavaRandom::new(42), &DOCS)
}

fn term(field: &str, t: &str) -> Clause {
    Clause::Term(TermQuery::new(field, t))
}

fn or(clauses: Vec<Clause>) -> Clause {
    BooleanQuery {
        should: clauses,
        ..BooleanQuery::default()
    }
    .into()
}

fn panics(f: impl FnOnce()) -> String {
    let err = catch_unwind(AssertUnwindSafe(f)).expect_err("the check should fail");
    err.downcast_ref::<String>()
        .cloned()
        .or_else(|| err.downcast_ref::<&str>().map(|s| (*s).to_string()))
        .unwrap_or_default()
}

/// `testTermQuery`.
#[test]
fn term_query() {
    let b = base();
    let q = term(FIELD_WITH_OFFSETS, "w1");
    b.check_matches(
        &q,
        FIELD_WITH_OFFSETS,
        &[
            &[0, 0, 0, 0, 2],
            &[1, 0, 0, 0, 2],
            &[2, 0, 0, 0, 2],
            &[3, 0, 0, 0, 2, 2, 2, 6, 8],
            &[4],
        ],
    );
    b.check_label_count(&q, FIELD_WITH_OFFSETS, &[1, 1, 1, 1, 0, 0]);
    b.assert_is_leaf_match(&q, FIELD_WITH_OFFSETS);
    b.check_sub_matches(
        &[("q".to_string(), q)],
        &[&["q"], &["q"], &["q"], &["q"], &[], &[]],
    );
}

/// `testTermQueryNoStoredOffsets`.
#[test]
fn term_query_no_stored_offsets() {
    base().check_matches(
        &term(FIELD_NO_OFFSETS, "w1"),
        FIELD_NO_OFFSETS,
        &[
            &[0, 0, 0, -1, -1],
            &[1, 0, 0, -1, -1],
            &[2, 0, 0, -1, -1],
            &[3, 0, 0, -1, -1, 2, 2, -1, -1],
            &[4],
        ],
    );
}

/// `testTermQueryNoPositions`.
#[test]
fn term_query_no_positions() {
    let b = base();
    for field in [FIELD_DOCS_ONLY, FIELD_FREQS] {
        b.check_no_positions_matches(&term(field, "w1"), field, &[true, true, true, true, false]);
    }
}

/// `testDisjunction`.
#[test]
fn disjunction() {
    let b = base();
    let (w1, w3) = (
        term(FIELD_WITH_OFFSETS, "w1"),
        term(FIELD_WITH_OFFSETS, "w3"),
    );
    let q = or(vec![w1.clone(), w3.clone()]);
    b.check_matches(
        &q,
        FIELD_WITH_OFFSETS,
        &[
            &[0, 0, 0, 0, 2, 2, 2, 6, 8],
            &[1, 0, 0, 0, 2, 1, 1, 3, 5, 3, 3, 9, 11],
            &[2, 0, 0, 0, 2],
            &[3, 0, 0, 0, 2, 2, 2, 6, 8, 5, 5, 15, 17],
            &[4],
        ],
    );
    b.check_label_count(&q, FIELD_WITH_OFFSETS, &[2, 2, 1, 2, 0, 0]);
    b.assert_is_leaf_match(&q, FIELD_WITH_OFFSETS);
    b.check_sub_matches(
        &[("w1".to_string(), w1), ("w3".to_string(), w3)],
        &[
            &["w1", "w3"],
            &["w1", "w3"],
            &["w1"],
            &["w1", "w3"],
            &[],
            &[],
        ],
    );
}

/// `testExactPhraseQuery`; its terms as sub-matches through the same
/// phrase as a `SpanNearQuery`, since a `PhraseQuery`'s matches have no
/// sub-matches here (see `lucene_search::matches`).
#[test]
fn exact_phrase_query() {
    let b = base();
    let pq: Clause = PhraseQuery::new(
        FIELD_WITH_OFFSETS,
        [b"phrase".to_vec(), b"sentence".to_vec()],
    )
    .into();
    b.check_matches(
        &pq,
        FIELD_WITH_OFFSETS,
        &[
            &[0],
            &[1],
            &[2],
            &[3],
            &[4, 1, 2, 2, 17, 5, 6, 28, 43, 10, 11, 60, 75],
        ],
    );
    let tm = TermMatch::new;
    b.check_term_matches(
        &span_phrase(),
        FIELD_WITH_OFFSETS,
        &[
            &[],
            &[],
            &[],
            &[],
            &[
                &[tm(1, 2, 8), tm(2, 9, 17)],
                &[tm(5, 28, 34), tm(6, 35, 43)],
                &[tm(10, 60, 66), tm(11, 67, 75)],
            ],
            &[],
        ],
    );
    assert_eq!(tm(1, 2, 8).to_string(), "1[2->8]");
}

/// Each check fails on matches the query does not report.
#[test]
fn checks_fail_on_wrong_expectations() {
    let b = base();
    let q = term(FIELD_WITH_OFFSETS, "w1");
    let msg = panics(|| b.check_matches(&q, FIELD_WITH_OFFSETS, &[&[0, 0, 0, 0, 3]]));
    assert!(msg.contains("Wrong end offset"), "{msg}");
    let msg = panics(|| b.check_matches(&q, FIELD_WITH_OFFSETS, &[&[0, 0, 0, 0, 2, 1, 1, 3, 5]]));
    assert!(msg.contains("left: 9"), "{msg}");
    panics(|| b.check_matches(&q, FIELD_WITH_OFFSETS, &[&[4, 0, 0, 0, 2]]));
    panics(|| b.check_matches(&q, FIELD_WITH_OFFSETS, &[&[0]]));
    let msg = panics(|| b.check_label_count(&q, FIELD_WITH_OFFSETS, &[2]));
    assert!(msg.contains("labels on doc 0"), "{msg}");
    let msg = panics(|| b.check_label_count(&q, FIELD_WITH_OFFSETS, &[1, 1, 1, 1, 1]));
    assert!(
        msg.contains("Expected to get matches on document 4"),
        "{msg}"
    );
    panics(|| b.check_label_count(&q, FIELD_NO_OFFSETS, &[1]));
    panics(|| b.check_no_positions_matches(&q, FIELD_WITH_OFFSETS, &[true]));
    panics(|| b.check_no_positions_matches(&q, FIELD_WITH_OFFSETS, &[false]));
    let named = [("q".to_string(), q.clone())];
    let msg = panics(|| b.check_sub_matches(&named, &[&[]]));
    assert!(msg.contains("Unexpected matching leaf queries"), "{msg}");
    let msg = panics(|| b.check_sub_matches(&named, &[&["q", "r"]]));
    assert!(msg.contains("Missing matching leaf queries"), "{msg}");
    panics(|| b.check_term_matches(&q, FIELD_WITH_OFFSETS, &[&[]]));
    panics(|| b.check_term_matches(&q, FIELD_WITH_OFFSETS, &[&[], &[], &[], &[], &[&[]]]));
    let pq = span_phrase();
    let tm = TermMatch::new;
    let msg = panics(|| {
        b.check_term_matches(
            &pq,
            FIELD_WITH_OFFSETS,
            &[&[], &[], &[], &[], &[&[tm(1, 2, 8)]]],
        );
    });
    assert!(msg.contains("Unexpected term match: 2[9->17]"), "{msg}");
    let msg = panics(|| {
        b.check_term_matches(
            &pq,
            FIELD_WITH_OFFSETS,
            &[
                &[],
                &[],
                &[],
                &[],
                &[&[tm(1, 2, 8), tm(2, 9, 17), tm(3, 1, 1)]],
            ],
        );
    });
    assert!(msg.contains("Missing term matches: 3[1->1]"), "{msg}");
}

/// `AssertingMatchesIterator`: reading before `next()` or after the end,
/// or advancing past the end, fails; everything else passes through.
#[test]
fn asserting_matches_iterator_fires() {
    use lucene_search::matches::{BoxMatchesIterator, Matches as _};
    use lucene_test_framework::search::asserting_matches::AssertingMatches;
    let b = base();
    let q = term(FIELD_WITH_OFFSETS, "w1");
    let m = b
        .with_searcher(|s| lucene_search::matches::matches(s, &q, 0))
        .expect("matches")
        .expect("doc 0 matches w1");
    let m = AssertingMatches::new(m);
    assert_eq!(m.fields(), vec![FIELD_WITH_OFFSETS.to_string()]);
    assert!(m.sub_matches().is_empty() && m.as_named().is_none() && !m.is_match_with_no_terms());
    assert!(m.get_matches("no such field").expect("matches").is_none());

    let fresh = || {
        m.get_matches(FIELD_WITH_OFFSETS)
            .expect("matches")
            .expect("w1")
    };
    let reads: [fn(&mut BoxMatchesIterator); 6] = [
        |it| {
            let _ = it.start_position();
        },
        |it| {
            let _ = it.end_position();
        },
        |it| {
            let _ = it.start_offset();
        },
        |it| {
            let _ = it.end_offset();
        },
        |it| {
            let _ = it.sub_matches();
        },
        |it| {
            let _ = it.query();
        },
    ];
    for read in reads {
        let msg = panics(|| read(&mut fresh()));
        assert!(msg.contains("not positioned"), "{msg}");
        let msg = panics(|| {
            let mut it = fresh();
            assert!(it.next().expect("next"));
            read(&mut it);
            assert!(!it.next().expect("next"));
            read(&mut it);
        });
        assert!(msg.contains("not positioned"), "{msg}");
    }
    let mut it = fresh();
    assert!(it.next().expect("next"));
    assert_eq!((it.start_position(), it.end_position()), (0, 0));
    assert_eq!((it.start_offset(), it.end_offset()), (0, 2));
    assert!(!it.next().expect("next"));
    let msg = panics(|| {
        let _ = it.next();
    });
    assert!(msg.contains("after the last match"), "{msg}");
}
