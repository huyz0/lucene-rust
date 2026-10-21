//! The unified highlighter's API edges, over the `uhighlight` fixture index:
//! what the differential test (`uhighlight_fixtures.rs`) does not configure.

use std::collections::{BTreeSet, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use lucene_analysis::{Analyzer, StandardAnalyzer};
use lucene_search::directory_reader::DirectoryReader;
use lucene_search::field_norms::FieldNorms;
use lucene_search::highlight::unified::{
    passage_by_start_offset, DefaultPassageFormatter, PassageScorer, UnifiedHighlighter,
};
use lucene_search::highlight::Error;
use lucene_search::index_searcher::IndexSearcher;
use lucene_search::query::Clause;
use lucene_search::queryparser::classic::QueryParser;

fn dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/data/uhighlight/index")
}

fn analyzer() -> Arc<Analyzer> {
    Arc::new(Analyzer::new(StandardAnalyzer::new()))
}

fn query(text: &str, max_doc: i32) -> Clause {
    QueryParser::new("offs", analyzer())
        .parse(text)
        .unwrap()
        .to_clause(max_doc)
}

/// Runs `body` with the index, a searcher over it and its documents.
fn with_index<R>(body: impl FnOnce(&DirectoryReader, &IndexSearcher<'_, '_>, Vec<i32>) -> R) -> R {
    let reader = DirectoryReader::open(&lucene_store::FsDirectory::open(dir())).unwrap();
    let opened = reader.open_segments().unwrap();
    let segments = opened.as_open_segments();
    let mut owned: Vec<HashMap<String, FieldNorms<'_>>> = Vec::new();
    for field in ["id", "offs", "pos", "docs", "tv", "both", "tvpos"] {
        for (i, n) in reader.field_norms(field).into_iter().enumerate() {
            if owned.len() <= i {
                owned.push(HashMap::new());
            }
            if let Some(n) = n {
                owned[i].insert(field.to_string(), n);
            }
        }
    }
    let norms: Vec<Option<&HashMap<String, FieldNorms<'_>>>> = owned.iter().map(Some).collect();
    let searcher = IndexSearcher::new(&segments, &norms).unwrap();
    let all: Vec<i32> = (0..reader.max_doc()).collect();
    body(&reader, &searcher, all)
}

#[test]
fn highlight_is_highlight_fields_for_one_field() {
    with_index(|reader, searcher, all| {
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()));
        let q = query("offs:fox", reader.max_doc());
        let one = uh.highlight(searcher, "offs", &q, &all, 2).unwrap();
        let many = uh
            .highlight_fields(searcher, &["offs"], &q, &all, &[2])
            .unwrap();
        assert_eq!(one, many["offs"]);
        assert!(one.iter().flatten().any(|s| s.contains("<b>fox</b>")));
        assert_eq!(uh.max_length(), 10_000);
        assert!(uh.flags().weight_matches);
    });
}

#[test]
fn bad_arguments_are_illegal_argument_errors() {
    with_index(|reader, searcher, all| {
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()));
        let q = query("offs:fox", reader.max_doc());
        assert!(matches!(
            uh.highlight_fields(searcher, &[], &q, &all, &[]),
            Err(Error::IllegalArgument(m)) if m.contains("must not be empty")
        ));
        assert!(matches!(
            uh.highlight_fields(searcher, &["offs"], &q, &all, &[1, 2]),
            Err(Error::IllegalArgument(m)) if m.contains("maxPassagesIn")
        ));
        let none = UnifiedHighlighter::new(None, Some(analyzer()));
        assert!(matches!(
            none.highlight_fields(searcher, &["offs"], &q, &all, &[1]),
            Err(Error::Search(lucene_search::Error::IllegalState(m)))
                if m.contains("highlightWithoutSearcher")
        ));
        assert!(matches!(
            uh.highlight_without_searcher("offs", &q, "a fox", 1),
            Err(Error::IllegalArgument(m)) if m.contains("without an IndexSearcher")
        ));
        // an out-of-range document is in no segment
        assert!(matches!(
            uh.highlight(searcher, "offs", &q, &[9999], 1),
            Err(Error::IllegalArgument(m)) if m.contains("no segment")
        ));
    });
}

#[test]
#[should_panic(expected = "maxLength must be < Integer.MAX_VALUE")]
fn a_max_length_of_int_max_is_refused() {
    let _ = UnifiedHighlighter::new(None, None).with_max_length(i32::MAX);
}

#[test]
fn without_a_searcher_the_text_is_analyzed() {
    let q = query("offs:fox", 1);
    let uh = UnifiedHighlighter::new(None, Some(analyzer()));
    let got = uh
        .highlight_without_searcher("offs", &q, "The quick brown fox. Another fox.", 1)
        .unwrap();
    assert_eq!(got.as_deref(), Some("Another <b>fox</b>."));
    // a text without a hit shows its start (by default), or nothing
    assert_eq!(
        uh.highlight_without_searcher("offs", &q, "no match here", 1)
            .unwrap()
            .as_deref(),
        Some("no match here")
    );
    let strict = UnifiedHighlighter::new(None, Some(analyzer())).with_max_no_highlight_passages(0);
    assert_eq!(
        strict
            .highlight_without_searcher("offs", &q, "no match here", 1)
            .unwrap(),
        None
    );
    assert_eq!(
        uh.highlight_without_searcher("offs", &q, "", 1).unwrap(),
        None
    );
    // no analyzer, no analysis
    let bare = UnifiedHighlighter::new(None, None);
    assert!(matches!(
        bare.highlight_without_searcher("offs", &q, "a fox", 1),
        Err(Error::IllegalArgument(m)) if m.contains("analyzer")
    ));
}

#[test]
fn a_phrase_without_weight_matches_marks_only_the_phrase() {
    let q = query("offs:\"quick brown\" offs:fox", 1);
    let uh = UnifiedHighlighter::new(None, Some(analyzer())).with_flags(false, true, true, true);
    let got = uh
        .highlight_without_searcher("offs", &q, "brown quick. quick brown fox", 2)
        .unwrap()
        .unwrap();
    assert!(
        got.contains("<b>quick</b> <b>brown</b> <b>fox</b>"),
        "{got}"
    );
    assert!(!got.contains("<b>brown</b> <b>quick</b>"), "{got}");
}

#[test]
fn masked_fields_are_highlighted_with_the_field() {
    with_index(|reader, searcher, all| {
        let q = query("pos:fox", reader.max_doc());
        // `offs` is highlighted by what is asked of `pos`
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()))
            .with_flags(true, true, true, false)
            .with_masked_fields(Arc::new(|f| {
                (f == "offs").then(|| BTreeSet::from(["pos".to_string()]))
            }));
        let got = uh.highlight(searcher, "offs", &q, &all, 1).unwrap();
        assert!(got.iter().flatten().any(|s| s.contains("<b>fox</b>")));
        // a field that masks nothing
        let none = UnifiedHighlighter::new(Some(reader), Some(analyzer()))
            .with_masked_fields(Arc::new(|_| Some(BTreeSet::new())));
        let q = query("offs:fox", reader.max_doc());
        assert!(none
            .highlight(searcher, "offs", &q, &all, 1)
            .unwrap()
            .iter()
            .flatten()
            .any(|s| s.contains("<b>fox</b>")));
    });
}

#[test]
fn scorer_formatter_and_comparator_are_used() {
    with_index(|reader, searcher, all| {
        let q = query("offs:fox", reader.max_doc());
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()))
            .with_scorer(PassageScorer::new(2.0, 0.5, 40.0))
            .with_formatter(Arc::new(DefaultPassageFormatter::new("<", ">", "|", false)))
            .with_passage_sort_comparator(passage_by_start_offset)
            .with_max_no_highlight_passages(1);
        let got = uh.highlight(searcher, "offs", &q, &all, 2).unwrap();
        assert!(got.iter().flatten().any(|s| s.contains("<fox>")));
    });
}

#[test]
fn a_field_without_offsets_is_reanalyzed_or_refused() {
    with_index(|reader, searcher, all| {
        // `docs` has no positions, so it is re-analyzed from the stored text
        let q = query("docs:fox", reader.max_doc());
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()));
        assert!(uh
            .highlight(searcher, "docs", &q, &all, 1)
            .unwrap()
            .iter()
            .flatten()
            .any(|s| s.contains("<b>fox</b>")));
        // and without an analyzer it cannot be
        let bare = UnifiedHighlighter::new(Some(reader), None);
        assert!(matches!(
            bare.highlight(searcher, "docs", &q, &all, 1),
            Err(Error::IllegalArgument(m)) if m.contains("analyzer")
        ));
        // an unindexed field is nothing to highlight
        let q = query("nope:fox", reader.max_doc());
        assert!(uh
            .highlight(searcher, "nope", &q, &all, 1)
            .unwrap()
            .iter()
            .all(Option::is_none));
    });
}

#[test]
fn an_analyzer_with_another_offset_gap_is_refused() {
    let q = query("offs:fox", 1);
    let gapped = Arc::new(Analyzer::new(StandardAnalyzer::new()).with_offset_gap(2));
    let uh = UnifiedHighlighter::new(None, Some(gapped));
    assert!(matches!(
        uh.highlight_without_searcher("offs", &q, "a fox", 1),
        Err(Error::IllegalArgument(m)) if m.contains("offset gap")
    ));
}

fn wrapped(q: Clause) -> Vec<Clause> {
    use lucene_search::query::{BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery};
    vec![
        Clause::Boost(Box::new(BoostQuery::new(q.clone(), 2.0))),
        Clause::ConstantScore(Box::new(ConstantScoreQuery::new(q.clone(), 1.0))),
        Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            [q.clone(), query("offs:zzz", 1)],
            0.0,
        ))),
        q,
    ]
}

#[test]
fn wrapped_queries_are_highlighted_like_the_bare_one() {
    let text = "The quick brown fox jumps over the lazy dog";
    for flags in [
        (true, true, true, true),
        (true, true, true, false),
        (true, false, false, false),
    ] {
        for bare in [
            query("offs:fox", 1),
            query("offs:\"quick brown\"", 1),
            query("offs:fo*", 1),
        ] {
            for q in wrapped(bare) {
                let uh = UnifiedHighlighter::new(None, Some(analyzer()))
                    .with_flags(flags.0, flags.1, flags.2, flags.3);
                let got = uh
                    .highlight_without_searcher("offs", &q, text, 1)
                    .unwrap()
                    .unwrap();
                assert!(got.contains("<b>"), "{flags:?} {got}");
            }
        }
    }
}

#[test]
fn span_queries_are_highlighted_from_their_spans() {
    use lucene_search::query::SpanQuery;
    with_index(|reader, searcher, all| {
        let near = Clause::Span(SpanQuery::span_near(
            [
                SpanQuery::span_term("offs", "quick"),
                SpanQuery::span_term("offs", "brown"),
            ],
            0,
            true,
        ));
        for flags in [
            (true, true, true, true),
            (true, true, true, false),
            (true, false, true, false),
        ] {
            let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()))
                .with_flags(flags.0, flags.1, flags.2, flags.3);
            let got = uh.highlight(searcher, "offs", &near, &all, 1).unwrap();
            assert!(
                got.iter().flatten().any(|s| s.contains("<b>quick")),
                "{flags:?} {got:?}"
            );
        }
    });
}

#[test]
fn a_field_matcher_reads_other_fields_as_the_highlighted_one() {
    with_index(|reader, searcher, all| {
        let q = query("pos:fox", reader.max_doc());
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()))
            .with_field_matcher(Arc::new(|_| true));
        let got = uh.highlight(searcher, "offs", &q, &all, 1).unwrap();
        assert!(got.iter().flatten().any(|s| s.contains("<b>fox</b>")));
    });
}

#[test]
fn clauses_on_other_fields_count_as_hits_only_when_required() {
    with_index(|reader, searcher, all| {
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()));
        for text in [
            "+docs:fox +offs:quick",
            "docs:fox offs:quick",
            "docs:fox -offs:cat",
            "docs:fox pos:quick",
        ] {
            let q = query(text, reader.max_doc());
            let got = uh.highlight(searcher, "docs", &q, &all, 1).unwrap();
            assert!(
                got.iter().flatten().any(|s| s.contains("<b>fox</b>")),
                "{text}: {got:?}"
            );
        }
    });
}

#[test]
fn values_joined_by_the_separator_are_analyzed_one_by_one() {
    let q = query("offs:fox", 1);
    let uh = UnifiedHighlighter::new(None, Some(analyzer()));
    let got = uh
        .highlight_without_searcher("offs", &q, "one fox\0two fox", 2)
        .unwrap()
        .unwrap();
    assert_eq!(got.matches("<b>fox</b>").count(), 2, "{got}");
}

#[test]
fn a_query_it_cannot_take_apart_is_highlighted_by_its_matches() {
    use lucene_search::query::SpanQuery;
    let q = Clause::Span(SpanQuery::span_or([
        SpanQuery::span_term("offs", "quick"),
        SpanQuery::span_term("offs", "lazy"),
    ]));
    let uh = UnifiedHighlighter::new(None, Some(analyzer()));
    let got = uh
        .highlight_without_searcher("offs", &q, "The quick fox and a lazy dog", 1)
        .unwrap()
        .unwrap();
    assert!(
        got.contains("<b>quick</b>") && got.contains("<b>lazy</b>"),
        "{got}"
    );
}

#[test]
fn a_match_needs_room_for_a_passage() {
    with_index(|reader, searcher, all| {
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()));
        let q = query("offs:fox", reader.max_doc());
        // Java: a `NullPointerException` (0) or an `IllegalArgumentException`
        // (below 0) from the passage queue, at the first match
        for max in [0, -1, -5] {
            assert!(
                matches!(
                    uh.highlight(searcher, "offs", &q, &all, max),
                    Err(Error::IllegalArgument(m)) if m.contains("maxPassages")
                ),
                "{max}"
            );
        }
        // a query that matches nothing never reaches the queue
        let none = query("offs:zzzzz", reader.max_doc());
        assert!(uh.highlight(searcher, "offs", &none, &all, 0).is_ok());
    });
}

#[test]
fn a_term_vector_without_offsets_is_refused_on_the_classic_path_too() {
    with_index(|reader, searcher, all| {
        let q = query("tvpos:fox", reader.max_doc());
        let classic = UnifiedHighlighter::new(Some(reader), Some(analyzer()))
            .with_flags(true, true, false, false);
        assert!(matches!(
            classic.highlight(searcher, "tvpos", &q, &all, 1),
            Err(Error::IllegalArgument(m)) if m.contains("without offsets")
        ));
    });
}

#[test]
fn the_overlay_asks_the_index_about_other_fields() {
    use lucene_search::query::{BooleanQuery, PhraseQuery, TermQuery};
    with_index(|reader, searcher, all| {
        let uh = UnifiedHighlighter::new(Some(reader), Some(analyzer()));
        // a phrase over a field without positions fails as it does in the
        // index, which the overlay asks
        let q: Clause = BooleanQuery::new()
            .with_should(vec![
                Clause::from(TermQuery::new("pos", "fox")),
                Clause::from(PhraseQuery::new("docs", ["quick", "brown"])),
            ])
            .into();
        assert!(matches!(
            uh.highlight(searcher, "pos", &q, &all, 1),
            Err(Error::Search(lucene_search::Error::IllegalState(m))) if m.contains("position")
        ));
        // a query part it cannot take apart (`lucene-join`'s term set, a
        // `MultiTermQuery` whose `visit` is a leaf): nothing is filtered out
        // of the one-document index, and its matches still mark the term
        let range = lucene_search::extended_query::MultiTermQuery::new(
            lucene_search::extended_query::MultiTermSource::TermSet(
                lucene_search::join::TermSetSource {
                    field: "pos".into(),
                    terms: vec![b"brown".to_vec()].into(),
                    from_field: "id".into(),
                    from_query: std::sync::Arc::new(Clause::from(TermQuery::new("id", "d0"))),
                },
            ),
            lucene_search::extended_query::RewriteMethod::default(),
        );
        let q: Clause = BooleanQuery::new()
            .with_should(vec![
                Clause::from(TermQuery::new("pos", "fox")),
                Clause::from(range),
            ])
            .into();
        let got = uh.highlight(searcher, "pos", &q, &all, 1).unwrap();
        assert!(
            got.iter().flatten().any(|s| s.contains("<b>fox</b>")),
            "{got:?}"
        );
    });
}

#[test]
fn a_span_term_on_a_rejected_field_is_not_marked() {
    use lucene_search::spans::SpanNode;
    with_index(|reader, searcher, all| {
        // the little span is `brown` of `pos` through a mask: the query's
        // field is `tv`, so the span is the helper's, but `brown`'s own field
        // is one the default matcher rejects -- it is not marked. (`tv` is
        // read from its term vectors, which nothing filters.)
        let big = SpanNode::near(
            vec![SpanNode::term("tv", "quick"), SpanNode::term("tv", "fox")],
            2,
            true,
        )
        .unwrap();
        let little = SpanNode::field_masking(SpanNode::term("pos", "brown"), "tv");
        let q: Clause = SpanNode::containing(big, little).unwrap().into();
        let classic = UnifiedHighlighter::new(Some(reader), Some(analyzer()))
            .with_flags(true, true, false, false);
        let got = classic.highlight(searcher, "tv", &q, &all, 1).unwrap();
        let snippets: Vec<&String> = got.iter().flatten().collect();
        assert!(
            snippets.iter().any(|s| s.contains("<b>quick</b>")),
            "{got:?}"
        );
        assert!(
            !snippets.iter().any(|s| s.contains("<b>brown</b>")),
            "{got:?}"
        );
    });
}
