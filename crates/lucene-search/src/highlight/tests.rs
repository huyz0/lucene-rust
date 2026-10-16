//! Unit tests for what the Lucene-generated fixtures (`highlight_fixtures.rs`)
//! do not reach: argument and offset errors, the constructors that take
//! weighted terms, and span queries of every shape through the extractor.

use super::*;
use crate::extended_query::{MultiTermQuery, MultiTermSource, RewriteMethod};
use crate::query::{Clause, PrefixQuery, TermQuery, WildcardQuery};
use crate::spans::SpanNode;
use lucene_analysis::{Analyzer, StandardAnalyzer, TokenStream};
use std::sync::Arc;

fn stream<'a>(analyzer: &'a Analyzer, text: &str) -> DynStream<'a> {
    DynStream::new(analyzer.token_stream("f", text).unwrap())
}

fn term(t: &str) -> Clause {
    Clause::Term(TermQuery::new("f", t.as_bytes().to_vec()))
}

fn terms_of(query: &Clause, text: &str) -> HashMap<String, WeightedSpanTerm> {
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let mut qse = WeightedSpanTermExtractor::new(None);
    qse.set_expand_multi_term_query(true);
    qse.set_max_doc_chars_to_analyze(1000);
    let extraction = qse
        .get_weighted_span_terms(query, 1.0, stream(&analyzer, text), Some("f"))
        .unwrap();
    let terms = extraction.terms.clone();
    drop(extraction);
    terms
}

use std::collections::HashMap;

#[test]
fn a_token_past_the_end_of_the_text_is_an_error() {
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let mut h = Highlighter::new(QueryTermScorer::new(&term("hello")));
    let err = h
        .get_best_fragments(stream(&analyzer, "hello there world"), "hello", 1)
        .unwrap_err();
    assert!(matches!(err, Error::InvalidTokenOffsets(_)));
    assert_eq!(
        err.to_string(),
        "org.apache.lucene.search.highlight.InvalidTokenOffsetsException: \
         Token there exceeds length of provided text sized 5"
    );
}

#[test]
fn every_error_names_its_java_class() {
    let e = Error::IllegalArgument("x".into());
    assert_eq!(e.class(), "java.lang.IllegalArgumentException");
    assert_eq!(e.message(), "x");
    let e = Error::Search(crate::Error::IllegalArgument("y".into()));
    assert_eq!(e.class(), "java.io.IOException");
    assert!(e.message().contains('y'));
    let e: Error = lucene_analysis::AnalysisError::IllegalArgument("z".into()).into();
    assert_eq!(e.class(), "java.io.IOException");
    assert!(e.to_string().contains('z'));
}

#[test]
fn a_gradient_formatter_refuses_malformed_colours() {
    let bad = |a, b, c, d| GradientFormatter::new(1.0, a, b, c, d).err().unwrap();
    assert!(bad(Some("#FFF"), Some("#FFFFFF"), None, None)
        .message()
        .contains("7 bytes long"));
    assert!(bad(Some("#FFFFFF"), Some("#FFFFFFF"), None, None)
        .message()
        .contains("7 bytes long"));
    assert!(bad(None, None, Some("#GGGGGG"), Some("#000000"))
        .message()
        .contains("NumberFormatException"));
    assert!(bad(None, None, Some("#12345"), Some("#000000"))
        .message()
        .contains("minBackgroundColor"));
    // a formatter with neither colour writes a bare font tag, and an unscored
    // group is left alone
    let g = GradientFormatter::new(1.0, None, None, None, None).unwrap();
    let mut group = TokenGroup::new();
    assert_eq!(g.highlight_term("x", &group), "x");
    assert_eq!(group.num_tokens(), 0);
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let mut h = Highlighter::with_formatter(g, QueryTermScorer::new(&term("x")));
    assert_eq!(
        h.get_best_fragment(stream(&analyzer, "x"), "x")
            .unwrap()
            .as_deref(),
        Some("<font >x</font>")
    );
    group.clear();
}

#[test]
fn encoders_escape_what_html_needs() {
    assert_eq!(
        SimpleHtmlEncoder::html_encode("<a href=\"x\">'&'/</a>"),
        "&lt;a href=&quot;x&quot;&gt;&#x27;&amp;&#x27;&#x2F;&lt;&#x2F;a&gt;"
    );
    assert_eq!(DefaultEncoder.encode_text("<"), "<");
    assert_eq!(SimpleHtmlEncoder.encode_text("&"), "&amp;");
}

#[test]
fn scorers_built_from_weighted_terms_keep_the_heaviest_duplicate() {
    let s = QueryScorer::from_weighted_span_terms(vec![
        WeightedSpanTerm::new(1.0, "a"),
        WeightedSpanTerm::with_position_sensitive(3.0, "a", true),
        WeightedSpanTerm::new(2.0, "b"),
        WeightedSpanTerm::new(0.5, "b"),
    ]);
    assert_eq!(s.max_term_weight(), 3.0);
    let a = s.weighted_span_term("a").unwrap();
    assert!(a.is_position_sensitive());
    assert_eq!(a.weight(), 3.0);
    assert_eq!(s.weighted_span_term("b").unwrap().weight(), 2.0);
    assert!(s.weighted_span_term("c").is_none());

    let t = QueryTermScorer::from_weighted_terms(vec![
        WeightedTerm::new(1.0, "a"),
        WeightedTerm::new(4.0, "a"),
        WeightedTerm::new(2.0, "b"),
    ]);
    assert_eq!(t.max_term_weight(), 4.0);
    assert_eq!(WeightedTerm::new(2.0, "q").term(), "q");
    assert_eq!(WeightedTerm::new(2.0, "q").weight(), 2.0);

    // a scorer with no query extracts nothing and reads the stream as it is;
    // "a" is position sensitive with no span to be in, so it scores nothing
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let mut h = Highlighter::new(s);
    assert_eq!(
        h.get_best_fragment(stream(&analyzer, "b a"), "b a")
            .unwrap()
            .as_deref(),
        Some("<B>b</B> a")
    );
}

#[test]
fn a_position_sensitive_term_only_scores_inside_its_spans() {
    let mut w = WeightedSpanTerm::with_position_sensitive(1.0, "x", true);
    w.add_position_spans(&[PositionSpan::new(2, 3), PositionSpan::new(7, 7)]);
    assert!(w.check_position(2) && w.check_position(3) && w.check_position(7));
    assert!(!w.check_position(1) && !w.check_position(4) && !w.check_position(8));
    assert_eq!(w.position_spans().len(), 2);
    assert_eq!(w.term(), "x");
    // `QueryScorer.getTokenScore` skips the "x" outside them
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let mut h = Highlighter::new(QueryScorer::from_weighted_span_terms(vec![w]));
    let out = h
        .get_best_fragments(stream(&analyzer, "a b x c d e f x"), "a b x c d e f x", 1)
        .unwrap();
    assert_eq!(out, vec!["a b <B>x</B> c d e f <B>x</B>".to_string()]);
    let out = h
        .get_best_fragments(stream(&analyzer, "x a b c x"), "x a b c x", 1)
        .unwrap();
    assert!(out.is_empty());
}

#[test]
fn every_span_query_shape_reaches_the_extractor() {
    let text = "the quick brown fox jumps over the quick lazy dog";
    let s = |t: &str| SpanNode::term("f", t.as_bytes().to_vec());
    let spans = |q: SpanNode| terms_of(&Clause::from(q), text);

    // containing / within: both sides contribute their terms
    let big = SpanNode::near(vec![s("quick"), s("fox")], 3, true).unwrap();
    let little = s("brown");
    let got = spans(SpanNode::containing(big.clone(), little.clone()).unwrap());
    assert!(got.contains_key("quick") && got.contains_key("brown") && got.contains_key("fox"));
    let got = spans(SpanNode::within(big, little).unwrap());
    assert!(got.contains_key("brown"));

    // position range and first
    let got = spans(SpanNode::position_range(s("quick"), 5, 9));
    assert_eq!(got["quick"].position_spans(), &[PositionSpan::new(7, 7)]);
    let got = spans(SpanNode::first(s("quick"), 3));
    assert_eq!(got["quick"].position_spans(), &[PositionSpan::new(1, 1)]);

    // not: only the include side's terms are the query's
    let got = spans(SpanNode::not(s("quick"), s("brown"), 0, 1).unwrap());
    assert!(got.contains_key("quick") && !got.contains_key("brown"));

    // a field-masking query reads its inner query
    let got = spans(SpanNode::field_masking(s("fox"), "other"));
    assert!(got.contains_key("fox"));

    // a wrapped multi-term query is expanded against the document's terms
    let prefix = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("f", "qui")),
        RewriteMethod::ScoringBoolean,
    ));
    let got = spans(prefix.clone());
    assert_eq!(got["quick"].position_spans().len(), 2);
    let wild = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Wildcard(WildcardQuery::new("f", "l*y")),
        RewriteMethod::ScoringBoolean,
    ));
    assert!(spans(wild)["lazy"].is_position_sensitive());
    // near over a wrapper that matches nothing matches nothing; an or drops it
    let none = SpanNode::multi_term(MultiTermQuery::new(
        MultiTermSource::Prefix(PrefixQuery::new("f", "zzz")),
        RewriteMethod::ScoringBoolean,
    ));
    assert!(spans(SpanNode::near(vec![s("quick"), none.clone()], 2, true).unwrap()).is_empty());
    assert!(spans(none.clone()).is_empty());
    let got = spans(SpanNode::or(vec![s("fox"), none.clone()]).unwrap());
    assert_eq!(got.len(), 1);
    // a not query whose exclude side matches nothing is its include side
    let got = spans(SpanNode::not(s("fox"), none.clone(), 0, 0).unwrap());
    assert!(got.contains_key("fox"));
    assert!(spans(SpanNode::not(none.clone(), s("fox"), 0, 0).unwrap()).is_empty());
    assert!(spans(SpanNode::containing(none.clone(), s("fox")).unwrap()).is_empty());
    assert!(spans(SpanNode::within(s("fox"), none.clone()).unwrap()).is_empty());
    assert!(spans(SpanNode::first(none.clone(), 3)).is_empty());
    assert!(spans(SpanNode::position_range(none.clone(), 0, 3)).is_empty());
    assert!(spans(SpanNode::field_masking(none, "x")).is_empty());

    // without expansion a wrapper is left to the searcher, which finds nothing
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let qse = WeightedSpanTermExtractor::new(None);
    assert!(!qse.expand_multi_term_query());
    let e = qse
        .get_weighted_span_terms(
            &Clause::from(prefix),
            1.0,
            stream(&analyzer, text),
            Some("f"),
        )
        .unwrap();
    assert!(e.terms.is_empty());
}

#[test]
fn multi_term_clauses_are_read_through_the_documents_terms() {
    let text = "quick quack quark fuzzy fuzz";
    let f = |c: Clause| terms_of(&c, text);
    let mut keys: Vec<String> = f(Clause::Prefix(PrefixQuery::new("f", "qua")))
        .into_keys()
        .collect();
    keys.sort();
    assert_eq!(keys, ["quack", "quark"]);
    let got = f(Clause::Regexp(crate::query::RegexpQuery {
        field: "f".into(),
        pattern: "qu.ck".into(),
    }));
    assert_eq!(got.len(), 2);
    // a fuzzy term carries its similarity as its weight
    let fuzzy = f(Clause::Fuzzy(crate::query::FuzzyQuery::new(
        "f",
        b"fuzzy".to_vec(),
    )));
    assert_eq!(fuzzy["fuzzy"].weight(), 1.0);
    assert!(fuzzy["fuzz"].weight() < 1.0);
    // a regexp Lucene's syntax refuses is an error
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let mut qse = WeightedSpanTermExtractor::new(None);
    qse.set_expand_multi_term_query(true);
    qse.set_max_doc_chars_to_analyze(100);
    let bad = Clause::Regexp(crate::query::RegexpQuery {
        field: "f".into(),
        pattern: "(".into(),
    });
    assert!(qse
        .get_weighted_span_terms(&bad, 1.0, stream(&analyzer, text), Some("f"))
        .is_err());
    // and a query on another field is not read
    let other = Clause::Prefix(PrefixQuery::new("g", "qua"));
    assert!(f(other).is_empty());
}

#[test]
fn query_shapes_reach_the_extractor() {
    let text = "alpha beta gamma delta alpha beta";
    // a constant-score, dismax, boost and multi-phrase leaf are read through
    let boosted = Clause::Boost(Box::new(crate::query::BoostQuery::new(term("beta"), 2.5)));
    assert_eq!(terms_of(&boosted, text)["beta"].weight(), 2.5);
    let cs = Clause::ConstantScore(Box::new(crate::query::ConstantScoreQuery::new(
        term("gamma"),
        1.0,
    )));
    assert!(terms_of(&cs, text).contains_key("gamma"));
    let dm = Clause::DisjunctionMax(Box::new(crate::query::DisjunctionMaxQuery::new(
        vec![term("delta"), term("alpha")],
        0.0,
    )));
    assert_eq!(terms_of(&dm, text).len(), 2);
    let mp = Clause::MultiPhrase(crate::query::MultiPhraseQuery::new(
        "f",
        vec![
            vec![b"alpha".to_vec(), b"gamma".to_vec()],
            vec![b"beta".to_vec(), b"delta".to_vec()],
        ],
    ));
    let got = terms_of(&mp, text);
    assert!(got["alpha"].is_position_sensitive());
    assert_eq!(got["alpha"].position_spans().len(), 3);
    // a one-term phrase is that term's span
    let one = Clause::Phrase(crate::query::PhraseQuery::new("f", [b"beta".to_vec()]));
    assert_eq!(terms_of(&one, text)["beta"].position_spans().len(), 2);
    // a phrase with a gap between its positions
    let mut gap = crate::query::PhraseQuery::new("f", [b"alpha".to_vec(), b"gamma".to_vec()]);
    gap.positions = vec![0, 2];
    // alpha at 0 and gamma at 2: a phrase with a hole of one position is a
    // span query with one position of slop, ordered
    let got = terms_of(&Clause::Phrase(gap), text);
    assert_eq!(got["gamma"].position_spans(), &[PositionSpan::new(0, 2)]);
    // match-all and match-none name no terms
    assert!(terms_of(
        &Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::default()),
        text
    )
    .is_empty());
}

#[test]
fn the_idf_weights_scale_the_terms() {
    struct Idx;
    impl DocFreqSource for Idx {
        fn max_doc(&self) -> i32 {
            100
        }
        fn doc_freq(&self, _field: &str, term: &str) -> i32 {
            if term == "rare" {
                0
            } else {
                49
            }
        }
    }
    let q = Clause::Boolean(Box::new({
        let mut b = crate::query::BooleanQuery::new();
        b.should.push(term("rare"));
        b.should.push(term("common"));
        b
    }));
    let terms = QueryTermExtractor::get_idf_weighted_terms(&q, &Idx, "f");
    let weight = |t: &str| terms.iter().find(|w| w.term() == t).unwrap().weight();
    assert!(weight("rare") > weight("common"));
    assert!((weight("common") - ((100.0f64 / 50.0).ln() + 1.0) as f32).abs() < 1e-6);
    // a prohibited clause is only a term when asked for
    let mut b = crate::query::BooleanQuery::new();
    b.must.push(term("a"));
    b.must_not.push(term("b"));
    let q = Clause::Boolean(Box::new(b));
    assert_eq!(QueryTermExtractor::get_terms(&q).len(), 1);
    assert_eq!(QueryTermExtractor::get_terms_in(&q, true, None).len(), 2);
    assert_eq!(
        QueryTermExtractor::get_terms_in(&q, true, Some("g")).len(),
        0
    );
}

#[test]
fn fragmenters_and_scorers_expose_their_settings() {
    let mut f = SimpleFragmenter::default();
    assert_eq!(f.fragment_size(), 100);
    f.set_fragment_size(7);
    assert_eq!(f.fragment_size(), 7);
    let mut q = QueryScorer::new(&term("a"));
    assert!(q.is_expand_multi_term_query() && !q.is_use_payloads());
    q.set_expand_multi_term_query(false);
    q.set_use_payloads(true);
    assert!(!q.is_expand_multi_term_query() && q.is_use_payloads());
    // a highlighter's settings
    let mut h = Highlighter::new(QueryScorer::new(&term("a")));
    assert_eq!(h.max_doc_chars_to_analyze(), DEFAULT_MAX_CHARS_TO_ANALYZE);
    h.set_fragment_scorer(QueryTermScorer::new(&term("b")));
    h.set_encoder(SimpleHtmlEncoder);
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    assert_eq!(
        h.get_best_fragments_joined(stream(&analyzer, "a b <"), "a b <", 3, "|")
            .unwrap(),
        "a <B>b</B> &lt;"
    );
    // payloads are read when asked for
    let mut p = QueryScorer::new(&term("a"));
    p.set_use_payloads(true);
    let mut h = Highlighter::new(p);
    assert!(h
        .get_best_fragment(stream(&analyzer, "a"), "a")
        .unwrap()
        .is_some());
}

fn vector(
    positions: bool,
    offsets: bool,
    payloads: bool,
) -> lucene_codecs::term_vectors::TermVectorField {
    use lucene_codecs::term_vectors::{TermVectorField, TermVectorTerm};
    TermVectorField {
        field_number: 0,
        has_positions: positions,
        has_offsets: offsets,
        has_payloads: payloads,
        terms: vec![
            TermVectorTerm {
                term: b"cat".to_vec(),
                freq: 2,
                positions: positions.then(|| vec![0, 3]),
                start_offsets: offsets.then(|| vec![0, 12]),
                end_offsets: offsets.then(|| vec![3, 15]),
                payloads: payloads.then(|| vec![b"p".to_vec(), vec![]]),
            },
            TermVectorTerm {
                term: b"dog".to_vec(),
                freq: 1,
                positions: positions.then(|| vec![1]),
                start_offsets: offsets.then(|| vec![4]),
                end_offsets: offsets.then(|| vec![7]),
                payloads: payloads.then(|| vec![vec![]]),
            },
        ],
    }
}

fn tokens(mut ts: impl TokenStream) -> Vec<String> {
    ts.reset().unwrap();
    let mut out = Vec::new();
    while ts.increment_token().unwrap() {
        let a = ts.attributes();
        out.push(format!(
            "{}@{}-{}+{}",
            a.term(),
            a.start_offset(),
            a.end_offset(),
            a.position_increment()
        ));
    }
    out
}

#[test]
fn a_term_vector_is_a_token_stream() {
    // positions and offsets: by position, increments from the gaps
    let v = vector(true, true, false);
    let ts = TokenStreamFromTermVector::new(&v, -1).unwrap();
    assert_eq!(ts.term_vector().terms.len(), 2);
    assert_eq!(tokens(ts), ["cat@0-3+1", "dog@4-7+1", "cat@12-15+2"]);
    // a start-offset limit drops the later tokens
    assert_eq!(
        tokens(TokenStreamFromTermVector::new(&v, 4).unwrap()),
        ["cat@0-3+1", "dog@4-7+1"]
    );
    // offsets only: positions are guessed from the offsets (start / 8)
    let v = vector(false, true, false);
    assert_eq!(
        tokens(TokenStreamFromTermVector::new(&v, -1).unwrap()),
        ["cat@0-3+1", "dog@4-7+1", "cat@12-15+1"]
    );
    // payloads reach the attribute only when asked for
    let v = vector(true, true, true);
    let mut ts = TokenStreamFromTermVector::new(&v, -1)
        .unwrap()
        .with_payloads();
    ts.reset().unwrap();
    assert!(ts.increment_token().unwrap());
    assert_eq!(ts.attributes().payload(), Some(&b"p"[..]));
    assert!(ts.increment_token().unwrap());
    assert!(ts.attributes().payload().is_none_or(<[u8]>::is_empty));
    // positions without offsets give no offsets
    let v = vector(true, false, false);
    assert_eq!(
        tokens(TokenStreamFromTermVector::new(&v, -1).unwrap()),
        ["cat@0-0+1", "dog@0-0+1", "cat@0-0+2"]
    );
    // neither is not a stream
    let none = vector(false, false, false);
    assert!(TokenStreamFromTermVector::new(&none, -1).is_err());
}

#[test]
fn token_sources_choose_between_the_vector_and_the_text() {
    let analyzer = Analyzer::new(StandardAnalyzer::new());
    let with_offsets = vector(true, true, false);
    let without = vector(true, false, false);
    assert!(TokenSources::get_term_vector_token_stream_or_null(None, -1)
        .unwrap()
        .is_none());
    assert!(
        TokenSources::get_term_vector_token_stream_or_null(Some(&without), -1)
            .unwrap()
            .is_none()
    );
    assert!(TokenSources::get_token_stream_from_vector(&without).is_err());
    assert!(TokenSources::get_token_stream_with_offsets(Some(&without))
        .unwrap()
        .is_none());
    assert!(TokenSources::get_token_stream_with_offsets(None)
        .unwrap()
        .is_none());
    assert!(
        TokenSources::get_token_stream_with_offsets(Some(&with_offsets))
            .unwrap()
            .is_some()
    );
    // a vector wins; without one the text is analyzed, and cut at the limit
    let ts = TokenSources::get_token_stream("f", Some(&with_offsets), "x", &analyzer, -1).unwrap();
    assert_eq!(tokens(ts).len(), 3);
    let ts = TokenSources::get_token_stream("f", None, "aa bb cc dd", &analyzer, 4).unwrap();
    assert_eq!(tokens(ts), ["aa@0-2+1", "bb@3-5+1"]);
    let ts = TokenSources::get_token_stream("f", None, "aa bb", &analyzer, 100).unwrap();
    assert_eq!(tokens(ts).len(), 2);
    // `getAnyTokenStream`: the vector, the stored text, or an error
    let ts = TokenSources::get_any_token_stream(Some(&with_offsets), None, "f", &analyzer).unwrap();
    assert_eq!(tokens(ts).len(), 3);
    let ts = TokenSources::get_any_token_stream(None, Some("a b"), "f", &analyzer).unwrap();
    assert_eq!(tokens(ts).len(), 2);
    let err = TokenSources::get_any_token_stream(None, None, "f", &analyzer)
        .err()
        .unwrap();
    assert_eq!(
        err.message(),
        "Field f in document is not stored and cannot be analyzed"
    );
    let ts = TokenSources::get_token_stream_from_text("f", "x y z", &analyzer).unwrap();
    assert_eq!(tokens(ts).len(), 3);
}

// ---------------------------------------------------------------------------
// matchhighlight
// ---------------------------------------------------------------------------

fn utf16(s: &str) -> Vec<u16> {
    s.encode_utf16().collect()
}

#[test]
fn passages_are_windows_around_sorted_markers() {
    // markers given out of order are still each put in a window of their own
    let value = utf16(&"x".repeat(40));
    let mut selector = PassageSelector::default();
    let markers = [OffsetRange::new(30, 33), OffsetRange::new(4, 7)];
    let got = selector.pick_best_whole(&value, &markers, 10, 5);
    assert_eq!(
        got,
        vec![
            Passage::new(0, 10, vec![OffsetRange::new(4, 7)]),
            Passage::new(26, 36, vec![OffsetRange::new(30, 33)]),
        ]
    );
    assert_eq!(got[0].length(), 10);
    // the nearer the passages, the fewer: adjacent ones keep the better
    let near = [OffsetRange::new(10, 12), OffsetRange::new(14, 20)];
    let got = selector.pick_best_whole(&value, &near, 12, 5);
    assert_eq!(got.len(), 1);
    assert_eq!(got[0].markers, vec![OffsetRange::new(14, 20)]);
    // nothing to pick in an empty value or with no window
    assert!(selector.pick_best_whole(&[], &markers, 10, 5).is_empty());
    assert!(selector.pick_best_whole(&value, &markers, 0, 5).is_empty());
    // no markers: the start of each permitted range
    let permitted = [OffsetRange::new(0, 15), OffsetRange::new(20, 40)];
    let got = selector.pick_best(&value, &[], 8, 5, &permitted);
    assert_eq!(
        got,
        vec![Passage::new(0, 8, vec![]), Passage::new(20, 28, vec![])]
    );
    assert_eq!(selector.pick_best(&value, &[], 8, 1, &permitted).len(), 1);
    // a marker beyond the permitted range is cut to it, one longer than the
    // window to the window
    let cut = selector.pick_best(&value, &[OffsetRange::new(8, 30)], 6, 1, &permitted);
    assert_eq!(cut[0].markers, vec![OffsetRange::new(8, 14)]);
    let ctx = OffsetRange::new(2, 4);
    assert!(ctx.contains(&OffsetRange::new(2, 3)) && !ctx.contains(&OffsetRange::new(1, 3)));
    assert_eq!(ctx.to_string(), "[from=2, to=4]");
}

#[test]
fn passages_are_formatted_with_nested_markers() {
    let value = utf16("abcdefghij");
    let f =
        PassageFormatter::with_markers("..", |r| format!("({}-{}:", r.from, r.to), |_| ")".into());
    let mut out = String::new();
    // nested: the inner marker closes inside the outer one
    f.format_passage(
        &mut out,
        &value,
        &Passage::new(0, 10, vec![OffsetRange::new(0, 10), OffsetRange::new(2, 5)]),
    );
    assert_eq!(out, "(0-10:ab(2-5:cde)fghij)");
    // two starting together: the longer opens first
    let mut out = String::new();
    f.format_passage(
        &mut out,
        &value,
        &Passage::new(0, 10, vec![OffsetRange::new(0, 5), OffsetRange::new(0, 10)]),
    );
    assert_eq!(out, "(0-10:(0-5:abcde)fghij)");
    // overlapping, neither inside the other: the first closes, the second
    // is closed and reopened around it
    let mut out = String::new();
    f.format_passage(
        &mut out,
        &value,
        &Passage::new(0, 10, vec![OffsetRange::new(1, 6), OffsetRange::new(4, 8)]),
    );
    assert_eq!(out, "a(1-6:bcd(4-8:ef))(4-8:gh)ij");
    // no markers, one marker
    let mut out = String::new();
    f.format_passage(&mut out, &value, &Passage::new(2, 6, vec![]));
    assert_eq!(out, "cdef");
    let mut out = String::new();
    f.format_passage(
        &mut out,
        &value,
        &Passage::new(2, 6, vec![OffsetRange::new(3, 4)]),
    );
    assert_eq!(out, "c(3-4:d)ef");
    // ellipses where a passage stops short of its range; none for no ranges
    let ranges = [OffsetRange::new(0, 10)];
    let got = f.format(&value, &[Passage::new(2, 6, vec![])], &ranges);
    assert_eq!(got, ["..cdef.."]);
    assert!(f
        .format(&value, &[Passage::new(2, 6, vec![])], &[])
        .is_empty());
    // passages in a later range, and passages past the last range
    let two = [OffsetRange::new(0, 4), OffsetRange::new(6, 10)];
    let got = f.format(
        &value,
        &[Passage::new(0, 4, vec![]), Passage::new(6, 8, vec![])],
        &two,
    );
    assert_eq!(got, ["abcd", "gh.."]);
    let got = f.format(
        &value,
        &[Passage::new(0, 4, vec![]), Passage::new(8, 10, vec![])],
        &two[..1],
    );
    assert_eq!(got, ["abcd"]);
}

#[test]
fn the_shrinking_adjuster_keeps_passages_on_words() {
    let mut a = BreakIteratorShrinkingAdjuster::default();
    let value = utf16("foo  ");
    a.current_value(&value);
    // an edge that cannot move forward past the end stays where it was
    assert_eq!(
        a.adjust(&Passage::new(4, 5, vec![])),
        OffsetRange::new(4, 5)
    );
    let value = utf16("alpha beta gamma delta");
    a.current_value(&value);
    // both edges drawn in to word boundaries, past the white space
    assert_eq!(
        a.adjust(&Passage::new(3, 14, vec![])),
        OffsetRange::new(6, 10)
    );
    // a marker the new edges would cut is kept whole
    let marked = Passage::new(3, 14, vec![OffsetRange::new(4, 12)]);
    assert_eq!(a.adjust(&marked), OffsetRange::new(4, 12));
    // an edge already on a word stays
    assert_eq!(
        a.adjust(&Passage::new(0, 22, vec![])),
        OffsetRange::new(0, 22)
    );
}

struct NoMatches(Clause);

impl crate::matches::MatchesIterator for NoMatches {
    fn next(&mut self) -> crate::Result<bool> {
        Ok(false)
    }
    // SENTINEL: -1 is `MatchesIterator`'s "no positions/offsets" -- this stub has none
    fn start_position(&self) -> i32 {
        -1
    }
    // SENTINEL: -1 is `MatchesIterator`'s "no positions/offsets" -- this stub has none
    fn end_position(&self) -> i32 {
        -1
    }
    // SENTINEL: -1 is `MatchesIterator`'s "no positions/offsets" -- this stub has none
    fn start_offset(&self) -> i32 {
        -1
    }
    // SENTINEL: -1 is `MatchesIterator`'s "no positions/offsets" -- this stub has none
    fn end_offset(&self) -> i32 {
        -1
    }
    fn query(&self) -> &Clause {
        &self.0
    }
}

/// A match list the iterator replays: positions and optional offsets.
struct Listed {
    matches: Vec<(i32, i32, i32, i32)>,
    at: usize,
    query: Clause,
}

impl crate::matches::MatchesIterator for Listed {
    fn next(&mut self) -> crate::Result<bool> {
        self.at += 1;
        Ok(self.at <= self.matches.len())
    }
    fn start_position(&self) -> i32 {
        self.matches[self.at - 1].0
    }
    fn end_position(&self) -> i32 {
        self.matches[self.at - 1].1
    }
    fn start_offset(&self) -> i32 {
        self.matches[self.at - 1].2
    }
    fn end_offset(&self) -> i32 {
        self.matches[self.at - 1].3
    }
    fn query(&self) -> &Clause {
        &self.query
    }
}

fn values_of(field: &str, values: &[&str]) -> StoredValues {
    let mut s = StoredValues::default();
    s.push_for_test(field, values);
    s
}

#[test]
fn offsets_come_from_matches_positions_tokens_or_values() {
    let analyzer = Arc::new(Analyzer::new(StandardAnalyzer::new()));
    let doc = values_of("f", &["one two three", "four five"]);
    assert_eq!(doc.fields(), ["f"]);
    assert!(doc.get_values("g").is_empty());
    let none = || NoMatches(term("x"));
    let listed = |m: Vec<(i32, i32, i32, i32)>| Listed {
        matches: m,
        at: 0,
        query: term("two"),
    };

    // positions are looked up in the re-analyzed values, across values
    let strategy = OffsetsFromPositions::new("f", Arc::clone(&analyzer));
    let got = strategy
        .get(&mut listed(vec![(1, 2, -1, -1), (3, 3, -1, -1)]), &doc)
        .unwrap();
    assert_eq!(got, [OffsetRange::new(4, 13), OffsetRange::new(14, 18)]);
    assert!(strategy.get(&mut none(), &doc).unwrap().is_empty());
    let err = strategy
        .get(&mut listed(vec![(-1, 2, 0, 0)]), &doc)
        .unwrap_err();
    assert_eq!(
        err.to_string(),
        "java.io.IOException: Matches API returned negative positions for field: f"
    );
    // a position past every token has no offsets
    let err = strategy
        .get(&mut listed(vec![(40, 41, -1, -1)]), &doc)
        .unwrap_err();
    assert!(
        err.message().starts_with("One of the offsets missing"),
        "{err}"
    );

    // matches that carry offsets are used as they are; the first without
    // sends the lot to the positions
    let from_matches =
        OffsetsFromMatchIterator::new("f", OffsetsFromPositions::new("f", Arc::clone(&analyzer)));
    let got = from_matches
        .get(&mut listed(vec![(1, 1, 4, 7)]), &doc)
        .unwrap();
    assert_eq!(got, [OffsetRange::new(4, 7)]);
    let got = from_matches
        .get(&mut listed(vec![(1, 1, 4, 7), (3, 3, -1, -1)]), &doc)
        .unwrap();
    assert_eq!(got, [OffsetRange::new(4, 7), OffsetRange::new(14, 18)]);
    assert!(from_matches
        .get(&mut listed(vec![(-1, 1, 0, 1)]), &doc)
        .is_err());

    // tokens whose term the matched queries name
    let tokens = OffsetsFromTokens::new("f", Arc::clone(&analyzer));
    let got = tokens.get(&mut listed(vec![(1, 1, -1, -1)]), &doc).unwrap();
    assert_eq!(got, [OffsetRange::new(4, 7)]);
    let other = OffsetsFromTokens::new("g", Arc::clone(&analyzer));
    assert!(other
        .get(&mut listed(vec![(1, 1, -1, -1)]), &doc)
        .unwrap()
        .is_empty());

    // each whole value
    let whole = OffsetsFromValues::new("f", analyzer);
    let got = whole.get(&mut none(), &doc).unwrap();
    assert_eq!(got, [OffsetRange::new(0, 13), OffsetRange::new(14, 23)]);
}

#[test]
fn field_highlighters_combine_and_name_what_they_fetch() {
    let always = FieldValueHighlighters::verbatim_value(&["id", "name"]);
    let mut fetched = always.always_fetched_fields();
    fetched.sort();
    assert_eq!(fetched, ["id", "name"]);
    assert!(always.is_applicable("id", false) && !always.is_applicable("body", true));
    let skip = FieldValueHighlighters::skip_remaining();
    assert!(skip.is_applicable("anything", false));
    assert!(skip.always_fetched_fields().is_empty());
    let both = OrHighlighter::new(always, skip);
    assert!(both.is_applicable("body", false));
    assert_eq!(both.always_fetched_fields().len(), 2);
    let values = vec!["v".to_string()];
    assert_eq!(
        both.format("id", &values, &utf16("v"), &[OffsetRange::new(0, 1)], None),
        Some(values.clone())
    );
    assert_eq!(
        both.format(
            "body",
            &values,
            &utf16("v"),
            &[OffsetRange::new(0, 1)],
            Some(&[])
        ),
        None
    );
    // the leading characters of a field, whether or not it matched
    let lead = FieldValueHighlighters::max_leading_characters(
        6,
        "..",
        ["f".to_string()].into_iter().collect(),
    );
    assert_eq!(lead.always_fetched_fields(), ["f"]);
    let text = utf16("alpha beta gamma");
    let got = lead.format(
        "f",
        &["alpha beta gamma".into()],
        &text,
        &[OffsetRange::new(0, 16)],
        None,
    );
    assert_eq!(got, Some(vec!["alpha..".to_string()]));
}
