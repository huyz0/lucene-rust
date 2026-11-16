#![cfg(test)]
use super::*;
use crate::query::{BooleanQuery, TermQuery};

#[test]
fn a_visit_record_replays_behind_its_guards_and_sub_visitors() {
    use crate::extended_query::{ExtendedQuery, VisitEvent, VisitRecord};
    struct Seen(Vec<String>, bool);
    impl QueryVisitor for Seen {
        fn accept_field(&self, field: &str) -> bool {
            field == "f"
        }
        fn consume_terms(&mut self, _q: &Clause, terms: &[Term]) {
            self.0.extend(
                terms
                    .iter()
                    .map(|t| String::from_utf8_lossy(&t.bytes).into()),
            );
        }
        fn visit_leaf(&mut self, _q: &Clause) {
            self.0.push("leaf".into());
        }
        fn sub_visitor(&mut self, occur: Occur, _parent: &Clause) -> SubVisitor {
            if self.1 && occur == Occur::MustNot {
                SubVisitor::Empty
            } else {
                SubVisitor::Same
            }
        }
    }
    let terms = |guard: Option<&str>, t: &str| VisitEvent::Terms {
        guard: guard.map(str::to_owned),
        terms: vec![Term::new("f", t)],
    };
    let record = VisitRecord {
        events: vec![
            terms(Some("f"), "a"),
            terms(Some("g"), "b"),
            terms(None, "c"),
            VisitEvent::Leaf {
                guard: Some("g".into()),
            },
            VisitEvent::Leaf { guard: None },
            VisitEvent::Sub(
                Occur::MustNot,
                VisitRecord {
                    events: vec![terms(None, "d")],
                },
            ),
            VisitEvent::Node(Clause::Term(TermQuery::new("f", "e"))),
        ],
    };
    let q = Clause::Extended(Box::new(ExtendedQuery::Visited(record)));
    let seen = |skip_not| {
        let mut v = Seen(Vec::new(), skip_not);
        visit(&q, &mut v);
        v.0
    };
    assert_eq!(seen(false), ["a", "c", "leaf", "d", "e"]);
    assert_eq!(seen(true), ["a", "c", "leaf", "e"]);
    let Clause::Extended(e) = &q else {
        unreachable!()
    };
    assert_eq!(e.children().len(), 1);
}

#[test]
fn span_wrappers_visit_what_they_wrap() {
    use crate::extended_query::{MultiTermQuery, MultiTermSource, RewriteMethod};
    use crate::query::PrefixQuery;
    use crate::spans::payloads::{
        PayloadDecoder, PayloadFunction, PayloadScoreQuery, SpanPayloadCheckQuery,
    };
    use crate::spans::SpanNode;
    struct Seen(Vec<String>);
    impl QueryVisitor for Seen {
        fn accept_field(&self, field: &str) -> bool {
            field == "f"
        }
        fn consume_terms(&mut self, _q: &Clause, terms: &[Term]) {
            self.0.extend(
                terms
                    .iter()
                    .map(|t| String::from_utf8_lossy(&t.bytes).into()),
            );
        }
        fn consume_terms_matching(&mut self, _q: &Clause, _f: &str, m: &dyn TermMatcher) {
            self.0.push(format!("matcher:{}", m.matches(b"quick")));
        }
        fn visit_leaf(&mut self, _q: &Clause) {
            self.0.push("leaf".into());
        }
    }
    let seen = |node: SpanNode| {
        let mut v = Seen(Vec::new());
        visit(&Clause::from(node), &mut v);
        v.0
    };
    let prefix = |f: &str| {
        SpanNode::multi_term(MultiTermQuery::new(
            MultiTermSource::Prefix(PrefixQuery::new(f, "qu")),
            RewriteMethod::default(),
        ))
    };
    assert_eq!(seen(prefix("f")), ["matcher:true"]);
    assert!(seen(prefix("g")).is_empty());
    // `AutomatonQuery.visit` for a range and an automaton; the join's
    // term set is a `MultiTermQuery` leaf
    let wrap = |source| SpanNode::multi_term(MultiTermQuery::new(source, RewriteMethod::default()));
    let range = |lo: &str, hi: &str, incl_hi| {
        wrap(MultiTermSource::TermRange(
            crate::extended_query::TermRangeQuery::new(
                "f",
                (!lo.is_empty()).then(|| lo.as_bytes().to_vec()),
                (!hi.is_empty()).then(|| hi.as_bytes().to_vec()),
                false,
                incl_hi,
            ),
        ))
    };
    assert_eq!(seen(range("", "", false)), ["matcher:true"]);
    assert_eq!(seen(range("a", "b", false)), ["matcher:false"]);
    assert_eq!(seen(range("a", "r", true)), ["matcher:true"]);
    let automaton = wrap(MultiTermSource::Automaton(
        crate::extended_query::AutomatonQuery::new(
            "f",
            lucene_util::automaton::automata::make_any_binary(),
            true,
        ),
    ));
    assert_eq!(seen(automaton), ["matcher:true"]);
    let set = wrap(MultiTermSource::TermSet(crate::join::TermSetSource {
        field: "f".into(),
        terms: vec![b"quick".to_vec()].into(),
        from_field: "g".into(),
        from_query: std::sync::Arc::new(Clause::MatchAllDocs(
            crate::query::MatchAllDocsQuery::new(1),
        )),
    }));
    assert_eq!(seen(set), ["leaf"]);
    // a top-level multi-term query visits as its wrapper's inner one
    let mut v = Seen(Vec::new());
    visit(
        &Clause::Extended(Box::new(crate::extended_query::ExtendedQuery::MultiTerm(
            MultiTermQuery::new(
                MultiTermSource::Prefix(PrefixQuery::new("f", "z")),
                RewriteMethod::default(),
            ),
        ))),
        &mut v,
    );
    assert_eq!(v.0, ["matcher:false"]);
    let check = SpanNode::PayloadCheck(Box::new(SpanPayloadCheckQuery::new(
        SpanNode::term("f", "a"),
        vec![None],
    )));
    assert_eq!(seen(check), ["a"]);
    let elsewhere = SpanNode::PayloadCheck(Box::new(SpanPayloadCheckQuery::new(
        SpanNode::term("g", "a"),
        vec![None],
    )));
    assert!(seen(elsewhere).is_empty());
    let score = SpanNode::PayloadScore(Box::new(PayloadScoreQuery::new(
        SpanNode::term("f", "b"),
        PayloadFunction::Max,
        PayloadDecoder::Float,
        false,
    )));
    assert_eq!(seen(score), ["b"]);
}

#[test]
fn index_or_doc_values_visits_both_and_ranges_only_on_accepted_fields() {
    use crate::extended_query::{ExtendedQuery, IndexOrDocValuesQuery, NumericDocValuesRangeQuery};
    use crate::query::PointsRangeQuery;
    struct Leaves {
        field: &'static str,
        leaves: Vec<String>,
    }
    impl QueryVisitor for Leaves {
        fn accept_field(&self, field: &str) -> bool {
            field == self.field
        }
        fn visit_leaf(&mut self, query: &Clause) {
            self.leaves.push(match query {
                Clause::PointsRange(_) => "points".into(),
                Clause::Extended(e) => e.name().into(),
                _ => "other".into(),
            });
        }
    }
    let q: Clause = IndexOrDocValuesQuery::new(
        PointsRangeQuery::new("num", 1, 5),
        NumericDocValuesRangeQuery::new("num", 1, 5),
    )
    .into();
    let mut other = Leaves {
        field: "body",
        leaves: Vec::new(),
    };
    visit(&q, &mut other);
    assert!(other.leaves.is_empty());
    let mut num = Leaves {
        field: "num",
        leaves: Vec::new(),
    };
    visit(&q, &mut num);
    assert_eq!(num.leaves.len(), 2);
    assert_eq!(num.leaves[0], "points");
    let dv = Clause::Extended(Box::new(ExtendedQuery::NumericDocValuesRange(
        NumericDocValuesRangeQuery::new("num", 1, 5),
    )));
    let mut other = Leaves {
        field: "body",
        leaves: Vec::new(),
    };
    visit(&dv, &mut other);
    assert!(other.leaves.is_empty());
}

#[test]
fn a_phrase_prefix_visits_its_whole_terms_and_no_leaf() {
    use crate::extended_query::MultiPhrasePrefixQuery;
    let q: Clause = MultiPhrasePrefixQuery::new(
        "f",
        vec![
            vec![b"a".to_vec()],
            vec![b"b".to_vec(), b"c".to_vec()],
            vec![b"pre".to_vec()],
        ],
        vec![0, 1, 2],
        0,
    )
    .unwrap()
    .into();
    let terms: Vec<_> = extract_terms(&q).into_iter().map(|t| t.bytes).collect();
    assert_eq!(terms, [b"a".to_vec(), b"b".to_vec(), b"c".to_vec()]);
    struct Seen {
        field: &'static str,
        leaf: bool,
        terms: usize,
    }
    impl QueryVisitor for Seen {
        fn accept_field(&self, field: &str) -> bool {
            field == self.field
        }
        fn visit_leaf(&mut self, _query: &Clause) {
            self.leaf = true;
        }
        fn consume_terms(&mut self, _query: &Clause, terms: &[Term]) {
            self.terms += terms.len();
        }
    }
    let mut on = Seen {
        field: "f",
        leaf: false,
        terms: 0,
    };
    visit(&q, &mut on);
    assert!(!on.leaf);
    assert_eq!(on.terms, 3);
    let mut off = Seen {
        field: "g",
        leaf: false,
        terms: 0,
    };
    visit(&q, &mut off);
    assert_eq!((off.leaf, off.terms), (false, 0));
    assert!(MultiPhrasePrefixQuery::new("f", Vec::new(), Vec::new(), 0).is_err());
    assert!(MultiPhrasePrefixQuery::new("f", vec![Vec::new()], vec![0], 0).is_err());
}

#[test]
fn a_phrase_prefix_rewrites_over_the_terms_it_completes() {
    use crate::extended_query::MultiPhrasePrefixQuery;
    use crate::query::MultiPhraseQuery;
    let terms: Vec<Vec<u8>> = ["brown", "quack", "quick", "quiet", "zoo"]
        .iter()
        .map(|t| t.as_bytes().to_vec())
        .collect();
    let q = |arrays: Vec<Vec<&str>>, max: i32| {
        let n = arrays.len() as i32;
        MultiPhrasePrefixQuery::new(
            "f",
            arrays
                .into_iter()
                .map(|a| a.into_iter().map(|t| t.as_bytes().to_vec()).collect())
                .collect(),
            (0..n).collect(),
            1,
        )
        .unwrap()
        .with_max_expansions(max)
    };
    let expected = MultiPhraseQuery::with_positions(
        "f",
        [
            (vec![b"brown".to_vec()], 0),
            (vec![b"quick".to_vec(), b"quiet".to_vec()], 1),
        ],
    )
    .unwrap()
    .with_slop(1);
    assert_eq!(
        q(vec![vec!["brown"], vec!["qui"]], 50).rewrite_over(&terms),
        Clause::MultiPhrase(expected)
    );
    // one expansion per prefix until there are more than the maximum
    let Clause::MultiPhrase(m) = q(vec![vec!["q", "z"]], 1).rewrite_over(&terms) else {
        panic!("a multi-phrase")
    };
    assert_eq!(m.term_arrays, [vec![b"quack".to_vec(), b"zoo".to_vec()]]);
    let Clause::MultiPhrase(m) = q(vec![vec!["q", "z", "b"]], 1).rewrite_over(&terms) else {
        panic!("a multi-phrase")
    };
    assert_eq!(m.term_arrays.len(), 1);
    assert_eq!(m.term_arrays[0].len(), 2);
    assert!(matches!(
        q(vec![vec!["x"]], 50).rewrite_over(&terms),
        Clause::MatchNoDocs(_)
    ));
    let Clause::Boolean(b) = q(vec![vec!["brown"], vec!["x"]], 50).rewrite_over(&terms) else {
        panic!("a boolean")
    };
    assert!(matches!(b.must[0], Clause::MultiPhrase(_)));
    assert!(matches!(b.must[1], Clause::MatchNoDocs(_)));
}

#[test]
fn empty_visitor_accepts_nothing_and_must_not_is_skipped() {
    let mut b = BooleanQuery::new();
    b.must
        .push(Clause::Term(TermQuery::new("f", b"a".to_vec())));
    b.must_not
        .push(Clause::Term(TermQuery::new("f", b"b".to_vec())));
    let terms = extract_terms(&Clause::Boolean(Box::new(b)));
    assert_eq!(terms.len(), 1);
    assert!(terms.contains(&Term::new("f", b"a".to_vec())));
    let mut e = EmptyVisitor;
    assert!(!e.accept_field("f"));
    visit(&Clause::Term(TermQuery::new("f", b"a".to_vec())), &mut e);
    e.visit_leaf(&Clause::Term(TermQuery::new("f", b"a".to_vec())));
    assert!(matches!(
        e.sub_visitor(
            Occur::Must,
            &Clause::Term(TermQuery::new("f", b"a".to_vec()))
        ),
        SubVisitor::Same
    ));
}

#[test]
fn wildcard_escapes_and_prefix_automata() {
    let a = wildcard_automaton(b"a\\*b").unwrap();
    let c = CompiledAutomaton::with_options(&a, false, true, false).unwrap();
    assert_eq!(c.automaton_type, AutomatonType::SINGLE);
    assert_eq!(c.term.as_deref(), Some(&b"a*b"[..]));
    let trailing = wildcard_automaton(b"ab\\").unwrap();
    let c = CompiledAutomaton::with_options(&trailing, false, true, false).unwrap();
    assert_eq!(c.term.as_deref(), Some(&b"ab\\"[..]));
    let all = CompiledAutomaton::with_options(&prefix_automaton(b""), false, true, true).unwrap();
    assert_eq!(all.automaton_type, AutomatonType::ALL);
}

/// `CompiledAutomaton.visit`: a field the visitor rejects sees nothing;
/// `NONE` nothing; `SINGLE` its term; `NORMAL` -- determinized or not --
/// a matcher over the accepted terms.
#[test]
fn a_compiled_automaton_visits_by_its_type() {
    #[derive(Default)]
    struct Seen {
        terms: Vec<Term>,
        matched: Vec<bool>,
        reject: bool,
    }
    impl QueryVisitor for Seen {
        fn consume_terms(&mut self, _q: &Clause, terms: &[Term]) {
            self.terms.extend_from_slice(terms);
        }
        fn consume_terms_matching(&mut self, _q: &Clause, _f: &str, m: &dyn TermMatcher) {
            self.matched = vec![m.matches(b"ab"), m.matches(b"ac"), m.matches(b"b")];
        }
        fn accept_field(&self, _field: &str) -> bool {
            !self.reject
        }
    }
    let q = Clause::Term(TermQuery::new("f", b"x".to_vec()));
    // a(b|c), non-deterministic: two `a` transitions.
    let mut nfa = automaton::Automaton::new();
    let (s0, s1, s2, s3) = (
        nfa.create_state(),
        nfa.create_state(),
        nfa.create_state(),
        nfa.create_state(),
    );
    nfa.add_transition(s0, s1, i32::from(b'a'), i32::from(b'a'));
    nfa.add_transition(s0, s2, i32::from(b'a'), i32::from(b'a'));
    nfa.add_transition(s1, s3, i32::from(b'b'), i32::from(b'b'));
    nfa.add_transition(s2, s3, i32::from(b'c'), i32::from(b'c'));
    nfa.set_accept(s3, true);
    nfa.finish_state();
    let c = CompiledAutomaton::with_options(&nfa, false, false, true).unwrap();
    assert_eq!(c.automaton_type, AutomatonType::NORMAL);
    let mut seen = Seen::default();
    visit_compiled(&c, &mut seen, &q, "f");
    assert_eq!(seen.matched, vec![true, true, false]);
    let mut rejecting = Seen {
        reject: true,
        ..Seen::default()
    };
    visit_compiled(&c, &mut rejecting, &q, "f");
    assert!(rejecting.matched.is_empty());
    let none =
        CompiledAutomaton::with_options(&automaton::Automaton::new(), false, true, true).unwrap();
    assert_eq!(none.automaton_type, AutomatonType::NONE);
    let mut seen = Seen::default();
    visit_compiled(&none, &mut seen, &q, "f");
    assert!(seen.terms.is_empty() && seen.matched.is_empty());
}

/// Every matcher a visit hands over can be kept past it, as the same
/// test -- except a non-determinized automaton's.
#[test]
fn matchers_can_be_kept_past_the_visit() {
    use crate::extended_query::{AutomatonQuery, MultiTermQuery, MultiTermSource, RewriteMethod};
    use crate::query::{FuzzyQuery, PrefixQuery, RegexpQuery, TermInSetQuery};
    #[derive(Default)]
    struct Kept(Vec<Option<Box<dyn TermMatcher + Send + Sync>>>);
    impl QueryVisitor for Kept {
        fn consume_terms(&mut self, _q: &Clause, _terms: &[Term]) {}
        fn consume_terms_matching(&mut self, _q: &Clause, _f: &str, m: &dyn TermMatcher) {
            let owned = m.to_owned_matcher();
            if let Some(o) = &owned {
                for t in [&b"ab"[..], b"abc", b"abd", b"x", b""] {
                    assert_eq!(o.matches(t), m.matches(t), "{t:?}");
                    // and a copy of the copy
                    let again = o.to_owned_matcher().unwrap();
                    assert_eq!(again.matches(t), m.matches(t));
                }
            }
            self.0.push(owned);
        }
        fn visit_leaf(&mut self, _q: &Clause) {}
    }
    let kept = |q: Clause| {
        let mut v = Kept::default();
        visit(&q, &mut v);
        assert_eq!(v.0.len(), 1, "{q:?}");
        v.0.remove(0)
    };
    for q in [
        Clause::from(PrefixQuery::new("f", "ab")),
        Clause::from(PrefixQuery::new("f", "")),
        Clause::from(RegexpQuery::new("f", "ab[cd]")),
        Clause::from(FuzzyQuery::new("f", "abc")),
        Clause::from(TermInSetQuery::new("f", ["abc", "x"])),
    ] {
        assert!(kept(q.clone()).is_some(), "{q:?}");
    }
    // an automaton query keeps its automaton as given: here an NFA
    let nfa = lucene_util::automaton::RegExp::new("(a|ab)*c")
        .unwrap()
        .to_automaton()
        .unwrap();
    let q = MultiTermQuery::new(
        MultiTermSource::Automaton(AutomatonQuery::new("f", nfa, false)),
        RewriteMethod::default(),
    );
    assert!(kept(Clause::from(q)).is_none());
    // a plain closure keeps nothing
    assert!((|_: &[u8]| true).to_owned_matcher().is_none());
}
