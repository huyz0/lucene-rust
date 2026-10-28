use super::*;
use crate::query::{MatchNoDocsQuery, SpanQuery};

fn term(field: &str, t: &str) -> TermQuery {
    TermQuery::new(field, t.as_bytes().to_vec())
}

#[test]
fn term_text_falls_back_to_hex_for_bytes_that_are_not_utf8() {
    assert_eq!(term_text(b"abc"), "abc");
    assert_eq!(term_text(&[0xff, 0x01]), "[ff 1]");
}

#[test]
fn from_clause_keeps_every_query_class() {
    let cases: Vec<(Clause, &str)> = vec![
        (Clause::Term(term("f", "a")), "TermQuery"),
        (
            Clause::Wildcard(WildcardQuery::new("f", b"a*".to_vec())),
            "WildcardQuery",
        ),
        (
            Clause::Prefix(PrefixQuery::new("f", b"a".to_vec())),
            "PrefixQuery",
        ),
        (
            Clause::Fuzzy(FuzzyQuery::new("f", b"a".to_vec())),
            "FuzzyQuery",
        ),
        (Clause::Regexp(RegexpQuery::new("f", "a.*")), "RegexpQuery"),
        (
            Clause::Extended(Box::new(ExtendedQuery::Synonym(
                SynonymQuery::new("f", vec![(b"a".to_vec(), 1.0), (b"b".to_vec(), 0.5)]).unwrap(),
            ))),
            "SynonymQuery",
        ),
        (
            Clause::Extended(Box::new(ExtendedQuery::MultiTerm(MultiTermQuery::new(
                MultiTermSource::TermRange(TermRangeQuery::new("f", None, None, true, false)),
                RewriteMethod::ConstantScore,
            )))),
            "TermRangeQuery",
        ),
        (
            Clause::Extended(Box::new(ExtendedQuery::MultiTerm(MultiTermQuery::new(
                MultiTermSource::Prefix(PrefixQuery::new("f", b"x".to_vec())),
                RewriteMethod::ConstantScore,
            )))),
            "PrefixQuery",
        ),
        (
            Clause::MatchNoDocs(MatchNoDocsQuery::new()),
            "MatchNoDocsQuery",
        ),
        (
            Clause::Span(SpanQuery::SpanTerm {
                field: "f".into(),
                term: b"a".to_vec(),
            }),
            "SpanTermQuery",
        ),
        (
            Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(7)),
            "MatchAllDocsQuery",
        ),
    ];
    for (clause, class) in cases {
        let parsed = ParsedQuery::from_clause(clause.clone());
        assert_eq!(parsed.class_name(), class, "{clause:?}");
        // back to the same query to run, its rewrite method kept
        assert_eq!(parsed.to_clause(7), clause);
    }
}

#[test]
fn a_rewrite_method_wraps_only_when_not_the_default() {
    let w = ParsedQuery::Wildcard(WildcardQuery::new("f", b"a*b".to_vec()));
    assert_eq!(
        ParsedQuery::with_rewrite(w.clone(), RewriteMethod::ConstantScoreBlended),
        w
    );
    let names = [
        (
            RewriteMethod::ConstantScoreBlended,
            "CONSTANT_SCORE_BLENDED_REWRITE",
        ),
        (RewriteMethod::ConstantScore, "CONSTANT_SCORE_REWRITE"),
        (RewriteMethod::ScoringBoolean, "SCORING_BOOLEAN_REWRITE"),
        (
            RewriteMethod::ConstantScoreBoolean,
            "CONSTANT_SCORE_BOOLEAN_REWRITE",
        ),
        (
            RewriteMethod::TopTermsScoringBoolean(3),
            "TopTermsScoringBooleanQueryRewrite(3)",
        ),
        (
            RewriteMethod::TopTermsBoostOnlyBoolean(4),
            "TopTermsBoostOnlyBooleanQueryRewrite(4)",
        ),
        (
            RewriteMethod::TopTermsBlendedFreqScoring(5),
            "TopTermsBlendedFreqScoringRewrite(5)",
        ),
        (RewriteMethod::DocValues, "DocValuesRewriteMethod"),
    ];
    for (m, name) in names {
        assert_eq!(rewrite_name(m), name);
    }
    let r = ParsedQuery::with_rewrite(w, RewriteMethod::DocValues);
    assert_eq!(r.class_name(), "WildcardQuery");
    assert_eq!(r.to_string("f"), "a*b");
    assert_eq!(r.describe("f"), "WildcardQuery(a*b)@DocValuesRewriteMethod");
    assert_eq!(ParsedQuery::from_clause(r.to_clause(0)), r);
    let re = ParsedQuery::with_rewrite(
        ParsedQuery::Regexp(RegexpQuery::new("f", "a.")),
        RewriteMethod::ScoringBoolean,
    );
    assert_eq!(ParsedQuery::from_clause(re.to_clause(0)), re);
    // Anything else keeps its own query.
    let t = ParsedQuery::Rewritten(
        Box::new(ParsedQuery::Term(term("f", "a"))),
        RewriteMethod::ScoringBoolean,
    );
    assert_eq!(t.to_clause(0), Clause::Term(term("f", "a")));
    let syn = ParsedQuery::Rewritten(
        Box::new(ParsedQuery::Synonym(
            SynonymQuery::new("f", vec![(b"a".to_vec(), 1.0)]).unwrap(),
        )),
        RewriteMethod::ScoringBoolean,
    );
    assert!(matches!(syn.to_clause(0), Clause::Extended(_)));
}

#[test]
fn the_parsers_build_with_their_rewrite_method() {
    use crate::queryparser::classic::QueryParser;
    use crate::queryparser::flexible::StandardQueryParser;
    let analyzer = std::sync::Arc::new(lucene_analysis::Analyzer::keyword());
    let mut p = QueryParser::new("f", std::sync::Arc::clone(&analyzer));
    assert_eq!(
        p.multi_term_rewrite_method(),
        RewriteMethod::ConstantScoreBlended
    );
    assert_eq!(p.determinize_work_limit(), 10_000);
    p.set_multi_term_rewrite_method(RewriteMethod::ConstantScore);
    p.set_determinize_work_limit(100);
    for q in ["a*", "a*b", "/a./", "[a TO b]"] {
        let parsed = p.parse(q).unwrap();
        assert!(
            matches!(
                parsed,
                ParsedQuery::Rewritten(_, RewriteMethod::ConstantScore)
            ),
            "{q}: {parsed:?}"
        );
    }
    // A complex phrase's words rewrite as booleans whatever is set; the
    // words outside a phrase take the parser's method.
    let mut c = QueryParser::complex_phrase("f", analyzer);
    c.set_multi_term_rewrite_method(RewriteMethod::ConstantScore);
    let parsed = c.parse("\"a* b\" c*").unwrap();
    let ParsedQuery::Boolean(b) = parsed else {
        panic!("{parsed:?}")
    };
    let ParsedQuery::ComplexPhrase(phrase) = &b.clauses[0].1 else {
        panic!("{b:?}")
    };
    let ParsedQuery::Boolean(words) = phrase.contents.as_ref().unwrap() else {
        panic!("{phrase:?}")
    };
    assert!(matches!(words.clauses[0].1, ParsedQuery::Prefix(_)));
    assert!(matches!(b.clauses[1].1, ParsedQuery::Rewritten(..)));

    let mut f = StandardQueryParser::new();
    assert_eq!(
        f.multi_term_rewrite_method(),
        RewriteMethod::ConstantScoreBlended
    );
    f.set_multi_term_rewrite_method(RewriteMethod::ScoringBoolean);
    assert_eq!(f.multi_term_rewrite_method(), RewriteMethod::ScoringBoolean);
}

#[test]
fn a_boolean_keeps_its_occurs_and_minimum_should_match() {
    let mut b = BooleanQuery::new();
    b.must.push(Clause::Term(term("f", "m")));
    b.filter.push(Clause::Term(term("f", "fi")));
    b.should.push(Clause::Term(term("f", "s")));
    b.must_not.push(Clause::Term(term("f", "n")));
    b.minimum_should_match = 1;
    let boosted = Clause::Boost(Box::new(BoostQuery::new(Clause::Boolean(Box::new(b)), 2.0)));
    let parsed = ParsedQuery::from_clause(boosted.clone());
    assert_eq!(parsed.to_string("f"), "((+m #fi s -n)~1)^2.0");
    assert_eq!(
        parsed.describe("f"),
        "Boost(Boolean(msm=1 +TermQuery(m) #TermQuery(fi) TermQuery(s) -TermQuery(n))^2.0)"
    );
    assert_eq!(parsed.to_clause(0), boosted);
    assert_eq!(parsed.class_name(), "BoostQuery");
}

#[test]
fn leaves_print_as_lucene_prints_them() {
    let fuzzy = ParsedQuery::Fuzzy(FuzzyQuery::new("g", b"ab".to_vec()).with_max_edits(1));
    assert_eq!(fuzzy.to_string("f"), "g:ab~1");
    assert_eq!(fuzzy.to_string("g"), "ab~1");
    let regexp = ParsedQuery::Regexp(RegexpQuery::new("g", "a.*"));
    assert_eq!(regexp.to_string("f"), "g:/a.*/");
    let prefix = ParsedQuery::Prefix(PrefixQuery::new("f", b"ab".to_vec()));
    assert_eq!(prefix.to_string("f"), "ab*");
    let wildcard = ParsedQuery::Wildcard(WildcardQuery::new("g", b"a?".to_vec()));
    assert_eq!(wildcard.to_string("f"), "g:a?");
    let range = |lo: Option<&[u8]>, hi: Option<&[u8]>, il: bool, iu: bool| {
        ParsedQuery::TermRange(TermRangeQuery::new(
            "g",
            lo.map(<[u8]>::to_vec),
            hi.map(<[u8]>::to_vec),
            il,
            iu,
        ))
    };
    assert_eq!(
        range(Some(b"a"), Some(b"z"), true, true).to_string("g"),
        "[a TO z]"
    );
    assert_eq!(range(None, None, false, false).to_string("f"), "g:{* TO *}");
    assert_eq!(
        range(Some(b"*"), Some(b"*"), true, false).to_string("g"),
        "[\\* TO \\*}"
    );
    assert_eq!(ParsedQuery::MatchAll.to_string("f"), "*:*");
    assert_eq!(
        ParsedQuery::MatchAll.describe("f"),
        "MatchAllDocsQuery(*:*)"
    );
    assert_eq!(
        ParsedQuery::MatchAll.to_clause(5),
        Clause::MatchAllDocs(MatchAllDocsQuery::new(5))
    );
    let synonym = ParsedQuery::Synonym(
        SynonymQuery::new("f", vec![(b"a".to_vec(), 1.0), (b"b".to_vec(), 0.5)]).unwrap(),
    );
    assert_eq!(synonym.to_string("f"), "Synonym(a b^0.5)");
    let multi = ParsedQuery::MultiPhrase(MultiPhraseQuery::new(
        "f",
        vec![vec![b"a".to_vec(), b"b".to_vec()], vec![b"c".to_vec()]],
    ));
    assert_eq!(multi.to_string("f"), "\"(a b) c\"");
    assert_eq!(multi.class_name(), "MultiPhraseQuery");
    let phrase = ParsedQuery::Phrase(PhraseQuery::new("f", vec![b"a".to_vec(), b"b".to_vec()]));
    assert_eq!(phrase.describe("f"), "PhraseQuery(\"a b\")");
    let range = ParsedQuery::TermRange(TermRangeQuery::new("f", None, None, true, true));
    assert!(matches!(range.to_clause(0), Clause::Extended(_)));
    let nested = ParsedQuery::Boolean(ParsedBoolean {
        clauses: vec![
            (Occur::Must, ParsedQuery::Boolean(ParsedBoolean::default())),
            (Occur::Should, ParsedQuery::Term(term("f", "x"))),
        ],
        minimum_should_match: 0,
    });
    assert_eq!(nested.to_string("f"), "+() x");
}

#[test]
fn spans_print_and_name_as_lucene_does() {
    let t = |f: &str, s: &str| SpanQuery::SpanTerm {
        field: f.into(),
        term: s.as_bytes().to_vec(),
    };
    let near = ParsedQuery::Span(SpanQuery::SpanNear {
        clauses: vec![t("f", "a"), t("g", "b")],
        slop: 1,
        in_order: true,
    });
    assert_eq!(near.to_string("f"), "spanNear([a, g:b], 1, true)");
    assert_eq!(near.class_name(), "SpanNearQuery");
    let or = ParsedQuery::Span(SpanQuery::SpanOr {
        clauses: vec![t("f", "a"), t("f", "b")],
    });
    assert_eq!(or.class_name(), "SpanOrQuery");
    assert_eq!(or.to_string("f"), "spanOr([a, b])");
    assert_eq!(ParsedQuery::Span(t("f", "a")).to_string("g"), "f:a");

    use crate::spans::SpanNode;
    let a = SpanNode::term("f", b"a".to_vec());
    let b = SpanNode::term("f", b"b".to_vec());
    let cases = [
        (a.clone(), "SpanTermQuery"),
        (
            SpanNode::near(vec![a.clone(), b.clone()], 0, true).unwrap(),
            "SpanNearQuery",
        ),
        (
            SpanNode::or(vec![a.clone(), b.clone()]).unwrap(),
            "SpanOrQuery",
        ),
        (
            SpanNode::not(a.clone(), b.clone(), 0, 0).unwrap(),
            "SpanNotQuery",
        ),
        (SpanNode::first(a.clone(), 3), "SpanFirstQuery"),
    ];
    for (node, class) in cases {
        let q = ParsedQuery::SpanNode(node.clone());
        assert_eq!(q.class_name(), class);
    }
    // a term is printed bare in its own field, qualified in another
    assert_eq!(ParsedQuery::SpanNode(a.clone()).to_string("f"), "a");
    assert_eq!(ParsedQuery::SpanNode(a.clone()).to_string("g"), "f:a");
    assert_eq!(
        ParsedQuery::SpanNode(SpanNode::first(a.clone(), 3)).to_string("f"),
        "spanFirst(a, 3)"
    );
    assert_eq!(
        ParsedQuery::SpanNode(SpanNode::PositionRange {
            inner: Box::new(a.clone()),
            start: 0,
            end: 2
        })
        .to_string("f"),
        "spanPosRange(a, 0, 2)"
    );
    assert_eq!(
        ParsedQuery::SpanNode(SpanNode::not(a.clone(), b.clone(), 1, 2).unwrap()).to_string("f"),
        "spanNot(a, b, 1, 2)"
    );
    assert_eq!(
        ParsedQuery::Other(Clause::ConstantScore(Box::new(
            crate::query::ConstantScoreQuery::new(Clause::Term(term("f", "a")), 1.0)
        )))
        .class_name(),
        "ConstantScoreQuery"
    );
    assert_eq!(ParsedQuery::MatchAll.class_name(), "MatchAllDocsQuery");
    let empty = ParsedQuery::Boolean(ParsedBoolean {
        clauses: Vec::new(),
        minimum_should_match: 0,
    });
    assert_eq!(empty.class_name(), "BooleanQuery");
}

#[test]
fn the_parsers_hold_their_date_settings() {
    use crate::queryparser::classic::QueryParser;
    use crate::queryparser::flexible::StandardQueryParser;
    use lucene_analysis::miscellaneous::DateLocale;
    use lucene_index::document::Resolution;
    let analyzer = std::sync::Arc::new(lucene_analysis::Analyzer::keyword());
    let de = DateLocale::for_language_tag("de-DE").unwrap();
    let thai = DateLocale::for_language_tag("th-TH").unwrap();

    let mut p = QueryParser::new("f", std::sync::Arc::clone(&analyzer));
    assert_eq!(p.locale().tag(), "en-US");
    assert_eq!((p.time_zone(), p.date_resolution("d")), (0, None));
    p.set_default_time_zone(3_600_000);
    assert_eq!(p.time_zone(), 3_600_000);
    p.set_time_zone(-7_200_000);
    assert_eq!(p.time_zone(), -7_200_000);
    p.set_locale(thai);
    assert_eq!(p.locale().tag(), "th-TH");
    // Refused once a bound is a date, not before.
    assert!(p.parse("d:[1/2/2563 TO 3/4/2564]").is_ok());
    p.set_date_resolution(Some(Resolution::Day));
    let err = p.parse("d:[1/2/2563 TO 3/4/2564]").unwrap_err();
    assert!(err.message.contains("th-TH"), "{err:?}");
    p.set_locale(de);
    assert_eq!(p.locale().tag(), "de-DE");
    p.set_date_resolution(Some(Resolution::Year));
    p.set_field_date_resolution("d", Resolution::Day);
    assert_eq!(p.date_resolution("d"), Some(Resolution::Day));
    assert_eq!(p.date_resolution("e"), Some(Resolution::Year));
    p.set_date_clock(None);
    let q = p.parse("d:[1.2.20 TO 3.4.21]").unwrap();
    assert_eq!(q.to_string("f"), "d:[20200131 TO 20210403]");

    let mut f = StandardQueryParser::with_analyzer(std::sync::Arc::clone(&analyzer));
    assert_eq!((f.locale().tag(), f.time_zone()), ("en-US", 0));
    f.set_default_time_zone(3_600_000);
    f.set_time_zone(0);
    f.set_locale(thai);
    assert!(f.parse("d:[1/2/2563 TO 3/4/2564]", Some("f")).is_ok());
    f.set_date_resolution(Some(Resolution::Day));
    let err = f.parse("d:[1/2/2563 TO 3/4/2564]", Some("f")).unwrap_err();
    assert!(err.message.contains("th-TH"), "{err:?}");
    f.set_locale(de);
    f.set_date_clock(None);
    f.set_date_resolution(Some(Resolution::Year));
    f.set_date_resolution_map(std::collections::HashMap::from([(
        "d".to_string(),
        Resolution::Day,
    )]));
    assert_eq!(f.date_resolution("d"), Some(Resolution::Day));
    assert_eq!(f.date_resolution("e"), Some(Resolution::Year));
    let q = f.parse("d:[1.2.20 TO 3.4.21]", Some("f")).unwrap();
    assert_eq!(q.to_string("f"), "d:[20200131 TO 20210402]");
}

/// Every recursive-descent parser refuses a hostile nesting depth with an
/// error rather than overflowing the stack (which aborts the process), and
/// still parses nesting up to its limit.
#[test]
fn deep_nesting_is_an_error_in_every_parser() {
    use lucene_analysis::{Analyzer, StandardAnalyzer};
    let nested = |open: &str, depth: usize| format!("{}a{}", open.repeat(depth), ")".repeat(depth));

    let flexible = flexible::StandardQueryParser::new();
    assert!(flexible.parse(&nested("(", 64), Some("f")).is_ok());
    assert!(flexible.parse(&nested("(", 65), Some("f")).is_err());
    assert!(flexible.parse(&nested("(", 200_000), Some("f")).is_err());

    let simple = simple::SimpleQueryParser::new(Analyzer::new(StandardAnalyzer::new()), "f");
    assert!(simple.parse(&nested("(", 128)).is_ok());
    assert!(simple.parse(&nested("(", 129)).is_err());
    assert!(simple.parse(&nested("(", 200_000)).is_err());

    assert!(surround::parser::parse(&nested("(", 127)).is_ok());
    assert!(surround::parser::parse(&nested("(", 200_000)).is_err());
    assert!(surround::parser::parse(&nested("OR(", 200_000)).is_err());
}
