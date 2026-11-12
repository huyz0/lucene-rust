//! Expected fragments are OpenSearch 3.9's `plain` highlighter's on the same
//! text, a `text` field (`standard` analyzer).

use super::*;
use lucene_analysis::StandardAnalyzer;
use lucene_search::extended_query::MultiPhrasePrefixQuery;
use lucene_search::query::{BooleanQuery, PhraseQuery, PrefixQuery, TermQuery, WildcardQuery};

const TEXT: &str = "The quick brown fox jumps over the lazy dog. A second sentence mentions \
the fox again & it ends here! Third <b>sentence</b> without the animal. The last one has quick \
foxes running quickly over the hills and far away from the lazy dogs of the village.";

fn options() -> PlainOptions {
    PlainOptions {
        field: "body".into(),
        require_field_match: true,
        fragmenter: PlainFragmenter::Span(100),
        number_of_fragments: 5,
        pre_tag: "<em>".into(),
        post_tag: "</em>".into(),
        html: false,
    }
}

fn run(options: PlainOptions, query: Clause) -> Vec<String> {
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    PlainHighlighter::new(options, query, analyzer)
        .fragments(TEXT)
        .unwrap()
        .into_iter()
        .map(|f| f.text)
        .collect()
}

fn term(field: &str, t: &str) -> Clause {
    Clause::Term(TermQuery::new(field, t.as_bytes().to_vec()))
}

#[test]
fn a_term_in_span_fragments() {
    assert_eq!(
        run(options(), term("body", "fox")),
        ["The quick brown <em>fox</em> jumps over the lazy dog. A second sentence mentions the <em>fox</em> again & it ends"]
    );
}

#[test]
fn a_phrase_marks_its_occurrence() {
    let q = Clause::Phrase(PhraseQuery::new(
        "body",
        [b"lazy".to_vec(), b"dog".to_vec()],
    ));
    assert_eq!(
        run(options(), q),
        ["The quick brown fox jumps over the <em>lazy</em> <em>dog</em>. A second sentence mentions the fox again & it ends"]
    );
}

#[test]
fn a_prefix_is_expanded_over_the_text() {
    let q = Clause::Prefix(PrefixQuery::new("body", b"qui".to_vec()));
    assert_eq!(
        run(options(), q),
        [
            " here! Third <b>sentence</b> without the animal. The last one has <em>quick</em> foxes running <em>quickly</em> over the hills and far away from the lazy dogs of the village.",
            "The <em>quick</em> brown fox jumps over the lazy dog. A second sentence mentions the fox again & it ends",
        ]
    );
    let q = Clause::Wildcard(WildcardQuery::new("body", "d*g*"));
    assert_eq!(
        run(options(), q),
        [
            "The quick brown fox jumps over the lazy <em>dog</em>. A second sentence mentions the fox again & it ends",
            " here! Third <b>sentence</b> without the animal. The last one has quick foxes running quickly over the hills and far away from the lazy <em>dogs</em> of the village.",
        ]
    );
}

#[test]
fn small_fragments_by_span_or_by_size() {
    let expected = [
        "The quick brown <em>fox</em>",
        " sentence mentions the <em>fox</em>",
    ];
    for fragmenter in [PlainFragmenter::Span(20), PlainFragmenter::Simple(20)] {
        let mut o = options();
        o.fragmenter = fragmenter;
        o.number_of_fragments = 3;
        assert_eq!(run(o, term("body", "fox")), expected);
    }
}

#[test]
fn no_fragments_is_the_whole_value() {
    let mut o = options();
    o.fragmenter = PlainFragmenter::Null;
    o.number_of_fragments = 0;
    assert_eq!(
        run(o, term("body", "fox")),
        ["The quick brown <em>fox</em> jumps over the lazy dog. A second sentence mentions the <em>fox</em> again & it ends here! Third <b>sentence</b> without the animal. The last one has quick foxes running quickly over the hills and far away from the lazy dogs of the village."]
    );
}

#[test]
fn the_html_encoder_and_custom_tags() {
    let mut o = options();
    o.html = true;
    o.pre_tag = "<x>".into();
    o.post_tag = "</x>".into();
    assert_eq!(
        run(o, term("body", "sentence")),
        [
            "The quick brown fox jumps over the lazy dog. A second <x>sentence</x> mentions the fox again &amp; it ends",
            " here! Third &lt;b&gt;<x>sentence</x>&lt;&#x2F;b&gt; without the animal. The last one has quick foxes running quickly over the hills and far away from the lazy dogs of the village.",
        ]
    );
}

#[test]
fn a_phrase_prefix_is_rewritten_over_the_text() {
    let q: Clause = MultiPhrasePrefixQuery::new(
        "body",
        vec![vec![b"quick".to_vec()], vec![b"fo".to_vec()]],
        vec![0, 1],
        0,
    )
    .unwrap()
    .with_max_expansions(50)
    .into();
    assert_eq!(
        run(options(), q),
        [" here! Third <b>sentence</b> without the animal. The last one has <em>quick</em> <em>foxes</em> running quickly over the hills and far away from the lazy dogs of the village."]
    );
}

#[test]
fn require_field_match_picks_the_terms_of_the_field() {
    let mut o = options();
    o.require_field_match = false;
    assert_eq!(
        run(o, term("title", "fox")),
        ["The quick brown <em>fox</em> jumps over the lazy dog. A second sentence mentions the <em>fox</em> again & it ends"]
    );
    let q = Clause::Boolean(Box::new(
        BooleanQuery::new().with_should([term("title", "fox"), term("body", "dog")]),
    ));
    assert_eq!(
        run(options(), q),
        ["The quick brown fox jumps over the lazy <em>dog</em>. A second sentence mentions the fox again & it ends"]
    );
}

#[test]
fn the_no_match_summary_ends_at_a_token() {
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let h = PlainHighlighter::new(options(), term("title", "fox"), analyzer);
    assert!(h.fragments(TEXT).unwrap().is_empty());
    let end = h.no_match_end(30, TEXT).unwrap();
    assert_eq!(&TEXT[..end as usize], "The quick brown fox jumps over");
    // a token ending exactly at the size, past every token, before any
    assert_eq!(h.no_match_end(9, TEXT).unwrap(), 9);
    assert_eq!(h.no_match_end(10_000, "a b").unwrap(), 3);
    assert_eq!(h.no_match_end(2, "abc").unwrap(), -1);
    assert_eq!(h.options().field, "body");
}

#[test]
fn a_span_read_to_the_end_keeps_the_next_text_from_cutting() {
    let q =
        Clause::Phrase(PhraseQuery::new("body", [b"quick".to_vec(), b"fox".to_vec()]).with_slop(3));
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let mut o = options();
    o.fragmenter = PlainFragmenter::Span(10);
    let h = PlainHighlighter::new(o, q, analyzer);
    // the span quick..fox ends at the last token: position 2 is waited for
    let mut wait = -1;
    let first = h.fragments_carrying("a quick fox", &mut wait).unwrap();
    assert_eq!(first.len(), 1);
    assert_eq!(wait, 3);
    // the next text does not cut before its position 3
    let texts = |f: Vec<PlainFragment>| f.into_iter().map(|f| f.text).collect::<Vec<_>>();
    let text = "one two three quick fox";
    assert_eq!(
        texts(h.fragments_carrying(text, &mut wait).unwrap()),
        ["one two three <em>quick</em> <em>fox</em>"]
    );
    assert_eq!(wait, 5);
    assert_eq!(
        texts(h.fragments(text).unwrap()),
        [" three <em>quick</em> <em>fox</em>"]
    );
}

#[test]
fn an_index_or_doc_values_query_and_a_term_set_give_their_terms() {
    use lucene_search::extended_query::IndexOrDocValuesQuery;
    use lucene_search::query::{MatchAllDocsQuery, TermInSetQuery};
    let iodv: Clause =
        IndexOrDocValuesQuery::new(term("body", "fox"), MatchAllDocsQuery { max_doc: 1 }).into();
    let q = Clause::Boolean(Box::new(BooleanQuery::new().with_should([
        iodv,
        Clause::TermInSet(TermInSetQuery::new(
            "body",
            [b"dog".to_vec(), b"zzz".to_vec()],
        )),
    ])));
    let mut o = options();
    o.number_of_fragments = 1;
    assert_eq!(
        run(o, q),
        ["The quick brown <em>fox</em> jumps over the lazy <em>dog</em>. A second sentence mentions the <em>fox</em> again & it ends"]
    );
}

#[test]
fn an_unknown_query_after_a_should_adds_no_terms() {
    use lucene_search::extended_query::{ExtendedQuery, VisitEvent, VisitRecord};
    use lucene_search::query_visitor::Term;
    // an intervals query's recorded visit: its terms count only when nothing
    // else gave any, and stock meets the shoulds before the filters
    let visited = Clause::Extended(Box::new(ExtendedQuery::Visited(VisitRecord {
        events: vec![VisitEvent::Terms {
            guard: None,
            terms: vec![Term::new("body", "dog")],
        }],
    })));
    let mut b = BooleanQuery::new();
    b.should.push(term("body", "fox"));
    b.filter.push(visited.clone());
    let both = run(options(), Clause::Boolean(Box::new(b))).concat();
    assert!(both.contains("<em>fox</em>") && !both.contains("<em>dog</em>"), "{both}");
    let mut alone = BooleanQuery::new();
    alone.filter.push(visited);
    let alone = run(options(), Clause::Boolean(Box::new(alone))).concat();
    assert!(alone.contains("<em>dog</em>"), "{alone}");
}

#[test]
fn a_prefix_past_the_clause_count_is_stock_s_error() {
    let words: Vec<String> = (0..1100).map(|i| format!("pre{i:04}")).collect();
    let text = words.join(" ");
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let q = Clause::Prefix(PrefixQuery::new("body", b"pre".to_vec()));
    let h = PlainHighlighter::new(options(), q, analyzer);
    assert!(matches!(
        h.fragments(&text),
        Err(crate::Error::TooManyClauses(_))
    ));
    assert!(h.fragments(&words[..20].join(" ")).is_ok());
}
