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
            "Query",
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
    ];
    for (clause, class) in cases {
        let parsed = ParsedQuery::from_clause(clause.clone());
        assert_eq!(parsed.class_name(), class, "{clause:?}");
        // back to the same query to run, the term range at the parser's
        // rewrite
        if class != "TermRangeQuery" {
            assert_eq!(parsed.to_clause(7), clause);
        }
    }
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
