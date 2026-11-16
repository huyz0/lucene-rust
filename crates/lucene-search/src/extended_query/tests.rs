#![cfg(test)]
use super::*;
use crate::query::TermQuery;

fn t(w: &str) -> Clause {
    Clause::Term(TermQuery::new("body", w.as_bytes().to_vec()))
}

fn invalid<T: std::fmt::Debug>(r: Result<T>) -> String {
    match r {
        Err(Error::InvalidQuery(m)) => m,
        other => panic!("accepted: {other:?}"),
    }
}

/// The builders' argument checks, with Java's messages.
#[test]
fn builders_refuse_what_javas_refuse() {
    for boost in [0.0, -1.0, 1.5, f32::NAN] {
        let m = invalid(SynonymQuery::new("f", [(b"a".to_vec(), boost)]));
        assert!(
            m.starts_with("boost must be a positive float"),
            "{boost}: {m}"
        );
    }
    let many: Vec<(Vec<u8>, f32)> = (0..=MAX_CLAUSE_COUNT)
        .map(|i| (i.to_string().into_bytes(), 1.0))
        .collect();
    assert_eq!(invalid(SynonymQuery::new("f", many)), "too many clauses");

    assert_eq!(
        invalid(CombinedFieldQuery::new(b"t".to_vec(), [("a", 0.5)])),
        "weight must be greater or equal to 1"
    );
    let fields: Vec<(String, f32)> = (0..=MAX_CLAUSE_COUNT)
        .map(|i| (format!("f{i}"), 1.0))
        .collect();
    assert_eq!(
        invalid(CombinedFieldQuery::new(b"t".to_vec(), fields)),
        "too many clauses"
    );
    // A field added twice keeps its last weight.
    let c = CombinedFieldQuery::new(b"t".to_vec(), [("b", 2.0), ("a", 1.0), ("b", 3.0)]).unwrap();
    assert_eq!(c.fields, [("a".to_string(), 1.0), ("b".to_string(), 3.0)]);

    let terms: Vec<(String, Vec<u8>, f32)> = (0..=MAX_CLAUSE_COUNT)
        .map(|i| ("f".to_string(), i.to_string().into_bytes(), 1.0))
        .collect();
    assert_eq!(
        invalid(BlendedTermQuery::new(terms, BlendedRewrite::default())),
        "too many clauses"
    );
    assert_eq!(
        BlendedRewrite::default(),
        BlendedRewrite::DisjunctionMax(0.01)
    );

    let two = || [t("a"), t("b")];
    for alpha in [-0.1, 1.1, f32::NAN] {
        let m = invalid(LogOddsFusionQuery::new(two(), alpha, None, None));
        assert!(m.starts_with("alpha must be in [0, 1]"), "{m}");
    }
    assert_eq!(
        invalid(LogOddsFusionQuery::new(two(), 0.5, Some(vec![1.0]), None)),
        "weights length 1 must equal clauses size 2"
    );
    for w in [vec![-0.5, 1.5], vec![f32::INFINITY, 0.0]] {
        let m = invalid(LogOddsFusionQuery::new(two(), 0.5, Some(w), None));
        assert!(
            m.starts_with("weights must be non-negative and finite"),
            "{m}"
        );
    }
    assert_eq!(
        invalid(LogOddsFusionQuery::new(
            two(),
            0.5,
            Some(vec![0.5, 0.25]),
            None
        )),
        "weights must sum to 1.0, got 0.75"
    );
    assert_eq!(
        invalid(LogOddsFusionQuery::new(
            two(),
            0.5,
            None,
            Some((vec![0.0], vec![1.0, 1.0]))
        )),
        "logit bounds must have one entry per clause (2)"
    );
    assert!(LogOddsFusionQuery::new(two(), 0.5, Some(vec![0.5, 0.5]), None).is_ok());

    for alpha in [0.0, -1.0, f32::INFINITY, f32::NAN] {
        let m = invalid(BayesianScoreQuery::new(t("a"), alpha, 0.0, 0.5));
        assert!(
            m.starts_with("alpha must be a positive finite value"),
            "{m}"
        );
    }
    let m = invalid(BayesianScoreQuery::new(t("a"), 1.0, f32::NAN, 0.5));
    assert!(m.starts_with("beta must be a finite value"), "{m}");
    for base_rate in [-0.1, 1.0] {
        let m = invalid(BayesianScoreQuery::new(t("a"), 1.0, 0.0, base_rate));
        assert!(m.starts_with("baseRate must be in [0, 1)"), "{m}");
    }
    // `baseRate < 0 || baseRate >= 1`: a NaN passes Java's check too.
    assert!(BayesianScoreQuery::new(t("a"), 1.0, 0.0, f32::NAN).is_ok());
}

/// `getClass().getSimpleName()` of each query, as explanations and
/// errors name it.
#[test]
fn every_query_names_its_java_class() {
    let multi = |s: MultiTermSource| {
        ExtendedQuery::MultiTerm(MultiTermQuery::new(s, RewriteMethod::default()))
    };
    let cases: Vec<(ExtendedQuery, &str)> = vec![
        (
            ExtendedQuery::CombinedField(
                CombinedFieldQuery::new(b"t".to_vec(), [("a", 1.0)]).unwrap(),
            ),
            "CombinedFieldQuery",
        ),
        (
            ExtendedQuery::NGramPhrase(NGramPhraseQuery::new(
                2,
                PhraseQuery::new("f", [b"ab".to_vec()]),
            )),
            "NGramPhraseQuery",
        ),
        (
            multi(MultiTermSource::Prefix(PrefixQuery::new(
                "f",
                b"a".to_vec(),
            ))),
            "PrefixQuery",
        ),
        (
            multi(MultiTermSource::Wildcard(WildcardQuery::new(
                "f",
                b"a*".to_vec(),
            ))),
            "WildcardQuery",
        ),
        (
            multi(MultiTermSource::Regexp(RegexpQuery::new("f", "a.*"))),
            "RegexpQuery",
        ),
        (
            multi(MultiTermSource::TermRange(TermRangeQuery::new(
                "f", None, None, true, true,
            ))),
            "TermRangeQuery",
        ),
        (
            multi(MultiTermSource::Automaton(AutomatonQuery::new(
                "f",
                lucene_util::automaton::automata::make_string("ab"),
                false,
            ))),
            "AutomatonQuery",
        ),
        (
            ExtendedQuery::Blended(
                BlendedTermQuery::new([("f", b"a".to_vec(), 1.0)], BlendedRewrite::Boolean)
                    .unwrap(),
            ),
            "BlendedTermQuery",
        ),
        (
            ExtendedQuery::IndriAnd(IndriAndQuery::new([t("a")])),
            "IndriAndQuery",
        ),
        (
            ExtendedQuery::LogOddsFusion(
                LogOddsFusionQuery::new([t("a")], 0.5, None, None).unwrap(),
            ),
            "LogOddsFusionQuery",
        ),
        (
            ExtendedQuery::BayesianScore(BayesianScoreQuery::new(t("a"), 1.0, 0.0, 0.5).unwrap()),
            "BayesianScoreQuery",
        ),
        (
            ExtendedQuery::NumericDocValuesRange(NumericDocValuesRangeQuery::new("n", 1, 2)),
            "SortedNumericDocValuesRangeQuery",
        ),
        (
            ExtendedQuery::IndexSortRange(IndexSortSortedNumericDocValuesRangeQuery::new(
                "n",
                1,
                2,
                t("a"),
            )),
            "IndexSortSortedNumericDocValuesRangeQuery",
        ),
        (
            ExtendedQuery::PointRange(PointRangeQuery::int_range("p", &[1], &[2]).unwrap()),
            "PointRangeQuery",
        ),
        (
            ExtendedQuery::PointInSet(PointInSetQuery::int_set("p", &[1, 2]).unwrap()),
            "PointInSetQuery",
        ),
        (
            ExtendedQuery::IndexOrDocValues(IndexOrDocValuesQuery::new(t("a"), t("b"))),
            "IndexOrDocValuesQuery",
        ),
    ];
    for (q, name) in cases {
        assert_eq!(q.name(), name, "{q:?}");
    }
}

/// An automaton query is equal by field, binary flag and language.
#[test]
fn automaton_queries_compare_by_field_and_automaton() {
    use lucene_util::automaton::automata::make_string;
    let a = AutomatonQuery::new("f", make_string("ab"), false);
    assert_eq!(a, AutomatonQuery::new("f", make_string("ab"), false));
    assert_ne!(a, AutomatonQuery::new("g", make_string("ab"), false));
    assert_ne!(a, AutomatonQuery::new("f", make_string("ab"), true));
    assert_ne!(a, AutomatonQuery::new("f", make_string("ac"), false));
}

/// `InetAddressPoint.encode`: an IPv4 address as its IPv4-mapped IPv6
/// form, an IPv6 address as its 16 bytes.
#[test]
fn inet_bytes_encode_both_families_in_sixteen_bytes() {
    let v6: std::net::IpAddr = "2001:db8::1".parse().unwrap();
    let mut want = [0u8; 16];
    want[..4].copy_from_slice(&[0x20, 0x01, 0x0d, 0xb8]);
    want[15] = 1;
    assert_eq!(inet_bytes(v6), want);
    let v4: std::net::IpAddr = "10.1.2.3".parse().unwrap();
    let mut mapped = [0u8; 16];
    mapped[10] = 0xff;
    mapped[11] = 0xff;
    mapped[12..].copy_from_slice(&[10, 1, 2, 3]);
    assert_eq!(inet_bytes(v4), mapped);
}
