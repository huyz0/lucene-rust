#![cfg(test)]
use super::*;
use crate::query::{
    BoostQuery, ConstantScoreQuery, DisjunctionMaxQuery, PrefixQuery, WildcardQuery,
};
use crate::{
    search_boolean_query_scored, search_disjunction_max_query_scored, search_term_query_scored,
    BooleanQuery, ScoringCollector, TermQuery,
};
use lucene_codecs::blocktree;

/// Test-only collector capturing every `(doc_id, score)` pair in
/// collection order -- this test module's ground truth to compare
/// `explain_clause`'s reported value against (see this module's own doc
/// comment on why bit-for-bit equality, not an epsilon comparison, is
/// the correctness bar here).
#[derive(Default)]
struct ScoreCapture {
    scores: Vec<(i32, f32)>,
}
impl ScoringCollector for ScoreCapture {
    fn collect(&mut self, doc_id: i32, score: f32) {
        self.scores.push((doc_id, score));
    }
}

fn open_fixture() -> (BlockTreeFields, Option<DocInputOwned>) {
    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties"))
        .expect("run fixtures generator first (GenBlockTree)");
    let get = |key: &str| -> String {
        manifest
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };
    let id_hex = get("id_hex");
    let mut id = [0u8; 16];
    for (i, slot) in id.iter_mut().enumerate() {
        *slot = u8::from_str_radix(&id_hex[i * 2..i * 2 + 2], 16).unwrap();
    }
    let suffix = get("segment_suffix");
    let max_doc: i32 = get("max_doc").parse().unwrap();

    let read_raw = |name: &str| -> Vec<u8> {
        std::fs::read(format!("{dir}{name}.raw")).unwrap_or_else(|_| panic!("missing {name}"))
    };
    let fnm = read_raw(&get("fnm_file_name"));
    let field_infos = lucene_codecs::field_infos::parse(&fnm, &id, "").expect("parse .fnm");
    let tim = read_raw(&get("tim_file_name"));
    let tip = read_raw(&get("tip_file_name"));
    let tmd = read_raw(&get("tmd_file_name"));
    let fields = blocktree::open(&tim, &tip, &tmd, &field_infos, &id, &suffix, max_doc)
        .expect("open blocktree");
    let doc = read_raw(&get("doc_file_name"));
    let pos = read_raw(&get("pos_file_name"));
    let pay = read_raw(&get("pay_file_name"));
    (
        fields,
        Some(DocInputOwned {
            doc,
            pos,
            pay,
            id,
            suffix,
        }),
    )
}

struct DocInputOwned {
    doc: Vec<u8>,
    pos: Vec<u8>,
    pay: Vec<u8>,
    id: [u8; 16],
    suffix: String,
}

impl DocInputOwned {
    fn open(&self) -> DocInput<'_> {
        DocInput::open(&self.doc, &self.id, &self.suffix).expect("open .doc")
    }
    fn open_pos(&self) -> PosInput<'_> {
        PosInput::open(&self.pos, &self.id, &self.suffix).expect("open .pos")
    }
    fn open_pay(&self) -> PayInput<'_> {
        PayInput::open(&self.pay, &self.id, &self.suffix).expect("open .pay")
    }
}

// --- Description-string fidelity ---
//
// Real Lucene's explanation *strings* are a public contract: downstream
// tooling (OpenSearch's `_explain` API, relevance-debugging UIs) parses
// them. These tests pin every description this module emits to the exact
// literal real Lucene 10.5.0 produces, quoted from
// `BM25Similarity.idfExplain`/`BM25Scorer.explain`/`explainTF`,
// `TermQuery.TermWeight.explain`, `PhraseWeight.explain`,
// `BooleanWeight.explain`, `DisjunctionMaxQuery.DisjunctionMaxWeight.
// explain` and `ConstantScoreWeight.explain`.

#[test]
fn term_explanation_descriptions_are_real_lucenes_verbatim() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let clause = Clause::Term(TermQuery::new("body", "cat"));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();

    assert_eq!(
        e.description,
        "weight(body:cat in 0) [BM25Similarity], result of:"
    );
    let score = &e.details[0];
    assert!(
        score
            .description
            .ends_with("), computed as boost * idf * tf from:"),
        "{}",
        score.description
    );
    assert!(
        score.description.starts_with("score(freq="),
        "{}",
        score.description
    );
    // Java renders the freq as a `Float`, i.e. always with a decimal point.
    assert!(
        score.description.contains(".0)"),
        "freq must render Java-style (\"2.0\", not \"2\"): {}",
        score.description
    );

    let idf = &score.details[0];
    assert_eq!(
        idf.description,
        "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:"
    );
    assert_eq!(
        idf.details[0].description,
        "n, number of documents containing term"
    );
    assert_eq!(
        idf.details[1].description,
        "N, total number of documents with field"
    );

    let tf = &score.details[1];
    assert_eq!(
        tf.description,
        "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:"
    );
    let descs: Vec<&str> = tf.details.iter().map(|d| d.description.as_str()).collect();
    assert_eq!(
        descs,
        vec![
            "freq, occurrences of term within document",
            "k1, term saturation parameter",
            "b, length normalization parameter",
            "dl, length of field",
            "avgdl, average length of field",
        ]
    );
}

#[test]
fn phrase_explanation_descriptions_are_real_lucenes_verbatim() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let owned = doc.as_ref().unwrap();
    let pos_in = owned.open_pos();
    let pay_in = owned.open_pay();
    // "alpha beta" matches real doc 8555 in the "pos" field at slop 0 --
    // the same known-good phrase this module's other phrase tests use.
    let query = crate::query::PhraseQuery::new("pos", ["alpha", "beta"]);
    let clause = Clause::Phrase(query);

    // Whichever doc actually contains the phrase; the no-match strings on
    // the way there are just as much a part of the contract.
    let mut checked = false;
    for doc_id in [0, 8555] {
        let e = explain_clause(
            &fields,
            doc_in.as_ref(),
            Some(&pos_in),
            Some(&pay_in),
            None,
            &clause,
            doc_id,
            None,
        )
        .unwrap();
        if !e.matched {
            assert!(
                e.description == "no matching terms" || e.description == "no matching phrase",
                "{}",
                e.description
            );
            continue;
        }
        checked = true;
        assert_eq!(
            e.description,
            format!("weight(pos:\"alpha beta\" in {doc_id}) [BM25Similarity], result of:")
        );
        let score = &e.details[0];
        assert!(
            score.description.starts_with("score(freq=")
                && score
                    .description
                    .ends_with("), computed as boost * idf * tf from:"),
            "{}",
            score.description
        );
        assert_eq!(score.details[0].description, "idf, sum of:");
        for per_term in &score.details[0].details {
            assert_eq!(
                per_term.description,
                "idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:"
            );
        }
        let tf = &score.details[1];
        assert_eq!(
            tf.description,
            "tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:"
        );
        assert!(
            tf.details[0].description.starts_with("phraseFreq="),
            "{}",
            tf.details[0].description
        );
    }
    assert!(checked, "fixture must contain the phrase \"alpha beta\"");
}

#[test]
fn filter_clause_explanation_is_real_lucenes_verbatim() {
    // `BooleanWeight.explain`'s FILTER arm:
    //
    //     subs.add(Explanation.match(0f, "match on required clause, product of:",
    //         Explanation.match(0f, Occur.FILTER + " clause"), e));
    //
    // `Occur.FILTER.toString()` is `"#"`, so the inner description is
    // exactly `"# clause"` and both values are `0f`.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    let clause = Clause::Boolean(Box::new(
        BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "dog")]),
    ));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
    assert!(e.matched);
    assert_eq!(e.description, "sum of:");
    assert_eq!(e.details.len(), 2, "one MUST sub, one FILTER sub");

    let filter_sub = &e.details[1];
    assert!(filter_sub.matched);
    assert_eq!(filter_sub.value, 0.0);
    assert_eq!(
        filter_sub.description,
        "match on required clause, product of:"
    );
    assert_eq!(filter_sub.details.len(), 2);
    assert_eq!(filter_sub.details[0].description, "# clause");
    assert_eq!(filter_sub.details[0].value, 0.0);
    assert!(
        filter_sub.details[1]
            .description
            .starts_with("weight(body:dog"),
        "the filtered clause's own explanation survives as a child: {}",
        filter_sub.details[1].description
    );

    // ... and the total is the MUST clause's value alone -- bit for bit.
    assert_eq!(e.value.to_bits(), e.details[0].value.to_bits());
}

#[test]
fn a_failing_filter_clause_explains_as_a_failing_required_clause() {
    // Java routes the non-matching case through `c.isRequired()`, which is
    // true for FILTER, so the description is the same
    // `"no match on required clause (...)"` a MUST clause gets.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let clause = Clause::Boolean(Box::new(
        BooleanQuery::new()
            .with_must([TermQuery::new("body", "cat")])
            .with_filter([TermQuery::new("body", "nonexistentterm")]),
    ));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
    assert!(!e.matched);
    assert_eq!(
        e.description,
        "Failure to meet condition(s) of required/prohibited clause(s)"
    );
    assert!(e
        .details
        .iter()
        .any(|d| d.description == "no match on required clause (body:nonexistentterm)"));
}

#[test]
fn a_filter_only_query_explains_as_a_match_of_zero() {
    // `matchCount` counts a matching FILTER clause (Java increments it for
    // every non-prohibited match), so this is a match, and the sum is 0.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let clause = Clause::Boolean(Box::new(
        BooleanQuery::new().with_filter([TermQuery::new("body", "cat")]),
    ));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
    assert!(e.matched);
    assert_eq!(e.description, "sum of:");
    assert_eq!(e.value, 0.0);
}

#[test]
fn a_filter_clause_does_not_count_toward_minimum_should_match_in_explain() {
    // Java increments `shouldMatchCount` only for `Occur.SHOULD`. Doc 2
    // matches the `cat` filter and neither optional clause, so the
    // explanation must be the minimum-not-reached failure with
    // `matched: 0` -- not a match.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let clause = Clause::Boolean(Box::new(
        BooleanQuery::new()
            .with_filter([TermQuery::new("body", "cat")])
            .with_should([TermQuery::new("body", "dog")])
            .with_minimum_should_match(1),
    ));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 2, None).unwrap();
    assert!(!e.matched);
    assert_eq!(
        e.description,
        "Failure to match minimum number of optional clauses: 1, matched: 0"
    );
}

#[test]
fn describe_clause_prints_a_filter_clause_with_lucenes_hash_prefix() {
    // `Occur.FILTER.toString()` is `"#"`, and `BooleanQuery.toString`
    // prefixes each clause with its occur.
    assert_eq!(
        describe_clause(&Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat")])
                .with_filter([TermQuery::new("body", "dog")])
                .with_should([TermQuery::new("body", "bird")])
                .with_must_not([TermQuery::new("body", "fish")])
        ))),
        "+body:cat #body:dog body:bird -body:fish"
    );
}

#[test]
fn boolean_explanation_descriptions_are_real_lucenes_verbatim() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    // A matching boolean: `BooleanWeight.explain` emits a bare "sum of:",
    // with no `"{value} = "` prefix (the value lives in the node itself,
    // and `Explanation.toString` re-renders it).
    let clause = Clause::Boolean(Box::new(BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("body", "cat"))],
        filter: Vec::new(),
        should: vec![Clause::Term(TermQuery::new("body", "nonexistentterm"))],
        must_not: Vec::new(),
        minimum_should_match: 0,
    }));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
    assert!(e.matched);
    assert_eq!(e.description, "sum of:");

    // A required clause that fails.
    let clause = Clause::Boolean(Box::new(BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("body", "nonexistentterm"))],
        filter: Vec::new(),
        should: Vec::new(),
        must_not: Vec::new(),
        minimum_should_match: 0,
    }));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
    assert!(!e.matched);
    assert_eq!(
        e.description,
        "Failure to meet condition(s) of required/prohibited clause(s)"
    );
    assert_eq!(
        e.details[0].description,
        "no match on required clause (body:nonexistentterm)"
    );

    // A prohibited clause that matches.
    let clause = Clause::Boolean(Box::new(BooleanQuery {
        must: vec![Clause::Term(TermQuery::new("body", "cat"))],
        filter: Vec::new(),
        should: Vec::new(),
        must_not: vec![Clause::Term(TermQuery::new("body", "cat"))],
        minimum_should_match: 0,
    }));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
    assert!(!e.matched);
    assert_eq!(
        e.description,
        "Failure to meet condition(s) of required/prohibited clause(s)"
    );
    assert!(e
        .details
        .iter()
        .any(|d| d.description == "match on prohibited clause (body:cat)"));

    // Nothing matched at all.
    let clause = Clause::Boolean(Box::new(BooleanQuery {
        must: Vec::new(),
        filter: Vec::new(),
        should: vec![Clause::Term(TermQuery::new("body", "nonexistentterm"))],
        must_not: Vec::new(),
        minimum_should_match: 0,
    }));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
    assert_eq!(e.description, "No matching clauses");
    assert_eq!(
        e.details[0].description,
        "no match on optional clause (body:nonexistentterm)"
    );

    // minimumShouldMatch not reached, but at least one clause matched.
    let clause = Clause::Boolean(Box::new(BooleanQuery {
        must: Vec::new(),
        filter: Vec::new(),
        should: vec![
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "nonexistentterm")),
        ],
        must_not: Vec::new(),
        minimum_should_match: 2,
    }));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap();
    assert_eq!(
        e.description,
        "Failure to match minimum number of optional clauses: 2, matched: 1"
    );
}

#[test]
fn dismax_explanation_descriptions_are_real_lucenes_verbatim() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    // `tieBreakerMultiplier == 0` -> "max of:"; otherwise
    // "max plus {tie} times others of:".
    let zero_tie = Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
        [Clause::Term(TermQuery::new("body", "cat"))],
        0.0,
    )));
    let e = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &zero_tie,
        0,
        None,
    )
    .unwrap();
    assert_eq!(e.description, "max of:");

    let tie = Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
        [Clause::Term(TermQuery::new("body", "cat"))],
        0.5,
    )));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &tie, 0, None).unwrap();
    assert_eq!(e.description, "max plus 0.5 times others of:");

    let none = Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
        [Clause::Term(TermQuery::new("body", "nonexistentterm"))],
        0.0,
    )));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &none, 0, None).unwrap();
    assert_eq!(e.description, "No matching clause");
}

#[test]
fn constant_score_and_flat_clause_descriptions_are_real_lucenes_verbatim() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    // Score exactly 1.0 -> no "^score" suffix (ConstantScoreWeight.explain).
    let one = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
        Clause::Term(TermQuery::new("body", "cat")),
        1.0,
    )));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &one, 0, None).unwrap();
    assert_eq!(e.description, "ConstantScore(body:cat)");

    let boosted = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
        Clause::Term(TermQuery::new("body", "cat")),
        2.5,
    )));
    let e = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &boosted,
        0,
        None,
    )
    .unwrap();
    assert_eq!(e.description, "ConstantScore(body:cat)^2.5");

    let miss = Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
        Clause::Term(TermQuery::new("body", "nonexistentterm")),
        1.0,
    )));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &miss, 3, None).unwrap();
    assert_eq!(
        e.description,
        "ConstantScore(body:nonexistentterm) doesn't match id 3"
    );

    // A flat (constant-scoring) multi-term clause renders as the query
    // itself on a match and `"<query> doesn't match id <doc>"` otherwise.
    let prefix = Clause::Prefix(PrefixQuery::new("body", "ca"));
    let e = explain_clause(&fields, doc_in.as_ref(), None, None, None, &prefix, 0, None).unwrap();
    assert_eq!(e.description, "body:ca*");
    let missing = Clause::Prefix(PrefixQuery::new("body", "zzzz"));
    let e = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &missing,
        0,
        None,
    )
    .unwrap();
    assert_eq!(e.description, "body:zzzz* doesn't match id 0");
}

#[test]
fn describe_clause_matches_lucene_query_to_string() {
    assert_eq!(
        describe_clause(&Clause::Term(TermQuery::new("f", "t"))),
        "f:t"
    );
    assert_eq!(
        describe_clause(&Clause::Phrase(crate::query::PhraseQuery::new(
            "f",
            ["a", "b"]
        ))),
        "f:\"a b\""
    );
    assert_eq!(
        describe_clause(&Clause::Phrase(
            crate::query::PhraseQuery::new("f", ["a", "b"]).with_slop(3)
        )),
        "f:\"a b\"~3"
    );
    assert_eq!(
        describe_clause(&Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("f", "a"))],
            filter: Vec::new(),
            should: vec![Clause::Term(TermQuery::new("f", "b"))],
            must_not: vec![Clause::Term(TermQuery::new("f", "c"))],
            minimum_should_match: 0,
        }))),
        "+f:a f:b -f:c"
    );
    // A sub-boolean in parentheses; the whole only under a minimum.
    assert_eq!(
        describe_clause(&Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Boolean(Box::new(BooleanQuery {
                should: vec![
                    Clause::Term(TermQuery::new("f", "a")),
                    Clause::Term(TermQuery::new("f", "b")),
                ],
                ..Default::default()
            }))],
            should: vec![
                Clause::Term(TermQuery::new("f", "c")),
                Clause::Term(TermQuery::new("f", "d")),
            ],
            minimum_should_match: 1,
            ..Default::default()
        }))),
        "(+(f:a f:b) f:c f:d)~1"
    );
    assert_eq!(
        describe_clause(&Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            [
                Clause::Term(TermQuery::new("f", "a")),
                Clause::Term(TermQuery::new("f", "b")),
            ],
            0.3,
        )))),
        "(f:a | f:b)~0.3"
    );
    assert_eq!(
        describe_clause(&Clause::Boost(Box::new(BoostQuery::new(
            Clause::Term(TermQuery::new("f", "a")),
            2.0,
        )))),
        "(f:a)^2.0"
    );
    assert_eq!(
        describe_clause(&Clause::Wildcard(WildcardQuery::new("f", "a*b"))),
        "f:a*b"
    );
    assert_eq!(
        describe_clause(&Clause::Fuzzy(crate::query::FuzzyQuery::new("f", "a"))),
        "f:a~2"
    );
    assert_eq!(
        describe_clause(&Clause::Regexp(crate::query::RegexpQuery::new("f", "a.*"))),
        "f:/a.*/"
    );
    assert_eq!(
        describe_clause(&Clause::MatchAllDocs(crate::query::MatchAllDocsQuery::new(
            7
        ))),
        "*:*"
    );
    assert_eq!(
        describe_clause(&Clause::Span(crate::query::SpanQuery::span_near(
            [
                crate::query::SpanQuery::span_term("f", "a"),
                crate::query::SpanQuery::span_term("f", "b"),
            ],
            2,
            true,
        ))),
        "spanNear([spanTerm(f:a), spanTerm(f:b)], 2, true)"
    );
}

/// The segment, boost and score mode an explanation runs under are
/// restored when it unwinds, so a panic caught at the FFI boundary leaves
/// a pooled thread's state as it found it.
#[test]
fn leaf_and_boost_are_restored_after_a_panic() {
    assert_eq!(leaf(), None);
    let r = std::panic::catch_unwind(|| {
        with_leaf(Some(9), 4, None, None, || {
            assert_eq!(leaf(), Some((9, 4)));
            with_boost(3.0, || {
                assert_eq!(weight_boost(), 3.0);
                panic!("explain failed");
            })
        })
    });
    assert!(r.is_err());
    assert_eq!(leaf(), None);
    assert_eq!(weight_boost(), 1.0);
    with_leaf_reader(|reader, sim| assert!(reader.is_none() && sim.is_none()));
    // The normal exit restores them too.
    assert_eq!(
        with_leaf(Some(2), 0, None, None, || with_boost(2.0, weight_boost)),
        2.0
    );
    assert_eq!((leaf(), weight_boost()), (None, 1.0));
    // The score mode.
    let r = std::panic::catch_unwind(|| {
        with_no_scores(|| {
            assert!(no_scores());
            panic!("explain failed");
        })
    });
    assert!(r.is_err());
    assert!(!no_scores());
    // A segment swap that panics half way -- the reader state still
    // borrowed by an enclosing `with_leaf_reader` -- puts back the half
    // it had already swapped.
    let r = std::panic::catch_unwind(|| {
        with_leaf_reader(|_, _| with_leaf(Some(5), 1, None, None, || ()))
    });
    assert!(r.is_err());
    assert_eq!(leaf(), None);
}

/// `DisjunctionMaxQuery.toString` wraps a boolean disjunct in
/// parentheses, as `BooleanQuery.toString` wraps a sub-boolean.
#[test]
fn describe_clause_wraps_a_dis_max_boolean_disjunct() {
    let t = |s: &str| Clause::Term(TermQuery::new("f", s));
    let ab = Clause::Boolean(Box::new(BooleanQuery {
        should: vec![t("a"), t("b")],
        ..Default::default()
    }));
    assert_eq!(
        describe_clause(&Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            [ab, t("c")],
            0.0,
        )))),
        "((f:a f:b) | f:c)"
    );
}

/// Pins `describe_clause` (`BooleanQuery.toString`): a top-level boolean
/// whose first clause is a dis-max and whose last is a sub-boolean keeps
/// both clauses' own parentheses -- Java adds no outer pair to strip, so
/// a "strip the outer parentheses" shortcut must not fire here.
#[test]
fn describe_clause_keeps_a_boolean_edge_clauses_parentheses() {
    let t = |s: &str| Clause::Term(TermQuery::new("f", s));
    let b = |x: &str, y: &str| {
        Clause::Boolean(Box::new(BooleanQuery {
            should: vec![t(x), t(y)],
            ..Default::default()
        }))
    };
    let dm = Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new([t("a"), t("b")], 0.0)));
    let q = Clause::Boolean(Box::new(BooleanQuery {
        should: vec![dm.clone(), b("c", "d")],
        ..Default::default()
    }));
    assert_eq!(describe_clause(&q), "(f:a | f:b) (f:c f:d)");
    let q = Clause::Boolean(Box::new(BooleanQuery {
        should: vec![b("a", "b"), b("c", "d")],
        ..Default::default()
    }));
    assert_eq!(describe_clause(&q), "(f:a f:b) (f:c f:d)");
    let q = Clause::Boolean(Box::new(BooleanQuery {
        should: vec![dm],
        ..Default::default()
    }));
    assert_eq!(describe_clause(&q), "(f:a | f:b)");
}

#[test]
fn display_renders_java_explanation_to_string_layout() {
    let e = Explanation::match_(2.0, "sum of:").with_details(vec![
        Explanation::match_(1.5, "weight(f:a in 0) [BM25Similarity], result of:")
            .with_details(vec![Explanation::match_(1.5, "score(freq=1.0), ...")]),
        Explanation::match_(0.5, "boost"),
    ]);
    assert_eq!(
        e.to_string(),
        "2.0 = sum of:\n  \
             1.5 = weight(f:a in 0) [BM25Similarity], result of:\n    \
             1.5 = score(freq=1.0), ...\n  \
             0.5 = boost\n"
    );
}

#[test]
fn java_float_always_keeps_a_decimal_point() {
    // Rust's own `Display` for f32 prints "2"; Java's `Float.toString`
    // prints "2.0", and every explanation string in this module has to
    // agree with Java's.
    assert_eq!(java_float(2.0), "2.0");
    assert_eq!(java_float(0.5), "0.5");
    assert_eq!(java_float(1.2), "1.2");
}

#[test]
fn java_float_switches_to_scientific_notation_where_java_does() {
    // `Float.toString` under JDK 21, one by one.
    for (v, java) in [
        (1.0E-4_f32, "1.0E-4"),
        (12_345_678.0, "1.2345678E7"),
        (0.001, "0.001"),
        (1e7, "1.0E7"),
        (9_999_999.0, "9999999.0"),
        (-3.5e-5, "-3.5E-5"),
        (0.000_999_99, "9.9999E-4"),
        (0.0, "0.0"),
        (-0.0, "-0.0"),
        (f32::NAN, "NaN"),
        (f32::INFINITY, "Infinity"),
        (f32::NEG_INFINITY, "-Infinity"),
    ] {
        assert_eq!(java_float(v), java, "{v:?}");
    }
}

#[test]
fn a_count_prints_as_a_java_long() {
    let e = Explanation::match_long(7991, "n, number of documents containing term");
    assert_eq!(e.value, 7991.0);
    assert_eq!(
        e.to_string(),
        "7991 = n, number of documents containing term\n"
    );
}

#[test]
fn dl_description_flips_exactly_where_the_encoded_norm_byte_does() {
    // `BM25Scorer.explainTF` picks "(approximate)" when the encoded norm
    // byte is > 39. `dl_description` uses the *decoded* length instead;
    // that is only equivalent because `LENGTH_TABLE[i] == i` for every
    // `i <= 39` and the table is monotonically non-decreasing. Assert both
    // properties against this port's own `decode_norm`, so a change to the
    // table would fail here rather than silently drift from Java.
    let mut previous = f32::NEG_INFINITY;
    for i in 0..256u32 {
        let decoded = similarity::decode_norm(i as i64);
        assert!(decoded >= previous, "LENGTH_TABLE must not decrease at {i}");
        previous = decoded;
        if i <= 39 {
            assert_eq!(decoded, i as f32, "LENGTH_TABLE[{i}] must be exactly {i}");
        }
        let java_says_approximate = i > 39;
        assert_eq!(
            dl_description(decoded) == "dl, length of field (approximate)",
            java_says_approximate,
            "encoded norm {i} (decoded {decoded})"
        );
    }
}

#[test]
fn points_range_clause_is_not_yet_explainable() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let clause = Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100));
    let err =
        explain_clause(&fields, doc_in.as_ref(), None, None, None, &clause, 0, None).unwrap_err();
    assert!(matches!(err, crate::Error::MissingPointsInput(field) if field == "body"));
}

#[test]
fn term_explain_matching_doc_equals_scored_search_output_exactly() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = TermQuery::new("body", "cat");

    let mut capture = ScoreCapture::default();
    search_term_query_scored(&fields, doc_in.as_ref(), None, &query, None, &mut capture).unwrap();
    assert!(
        !capture.scores.is_empty(),
        "fixture must have a doc matching body:cat"
    );
    let (target_doc, expected_score) = capture.scores[0];

    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Term(query),
        target_doc,
        None,
    )
    .unwrap();

    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);

    // The nested idf/tfNorm sub-values must multiply back to the top-level
    // value -- but only to within rounding, not exactly. Real Lucene says
    // so itself in `BM25Scorer.explain`: "not using 'product of' since the
    // rewrite that we do in score() introduces a small rounding error that
    // CheckHits complains about". The score is
    // `weight - weight / (1 + freq * normInverse)` and the `tf` node is
    // `1 - 1 / (1 + freq * normInverse)`; those are equal in exact
    // arithmetic and a few ULP apart in `f32`. The tolerance is
    // **relative**: `f32::EPSILON` on its own is the gap between 1.0 and
    // the next float, which is smaller than one ULP for any score >= 2 --
    // the multi-phrase fixture already scores 1.57 -- so an absolute
    // epsilon would make this flake rather than fail honestly.
    let score_node = &explanation.details[0];
    let idf_node = &score_node.details[0];
    let tf_norm_node = &score_node.details[1];
    assert!(
        (idf_node.value * tf_norm_node.value - expected_score).abs()
            <= 4.0 * f32::EPSILON * expected_score.abs().max(1.0),
        "idf {} * tfNorm {} = {}, score {expected_score}",
        idf_node.value,
        tf_norm_node.value,
        idf_node.value * tf_norm_node.value
    );
    assert_eq!(score_node.value, expected_score);
}

/// Confirms `explain_clause` has no interaction whatsoever with
/// `crate::search_term_query_scored_maxscore`'s MAXSCORE-skip path (task
/// #135): `explain_term` only ever calls `crate::term_doc_freqs` (the
/// eager, non-pruning path this module's top doc comment already
/// documents), so it can correctly explain a doc that a MAXSCORE search
/// over the *same term* would have safely skipped decoding entirely once
/// its `TopDocsCollector` filled up. This uses the same "big"/"everywhere"
/// fixture term (`docFreq == 300`, real impacts) as `lib.rs`'s
/// `maxscore_lazy_path_matches_eager_path_on_real_fixture_and_actually_skips_blocks`:
/// a `top_n = 1` MAXSCORE search only ever collects/returns its single
/// best-scoring doc, skipping real-Lucene-block decode for the rest of the
/// term's docs -- yet `explain_clause` asked to explain one of those
/// *not*-returned docs still produces the exact score
/// `search_term_query_scored` (the eager path) computes for it,
/// unaffected by the unrelated search having pruned blocks
/// (`explain_clause` never touches the maxscore path, by construction;
/// this makes the non-interaction explicit).
#[test]
fn explain_is_unaffected_by_an_unrelated_maxscore_search_pruning_the_same_term() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = TermQuery::new("big", b"everywhere".as_slice());

    // Eager, ground-truth per-doc scores for every doc matching this term.
    let mut eager = ScoreCapture::default();
    search_term_query_scored(&fields, doc_in.as_ref(), None, &query, None, &mut eager).unwrap();
    assert!(
        eager.scores.len() > 1,
        "fixture term must match more than one doc for this test to be meaningful"
    );

    // A top_n=1 MAXSCORE search over the identical term/doc_in/norms
    // returns only its single best doc.
    let mut maxscore = crate::TopDocsCollector::new(1);
    crate::search_term_query_scored_maxscore(
        &fields,
        doc_in.as_ref(),
        None,
        &query,
        None,
        &mut maxscore,
    )
    .unwrap();
    let kept_docs: std::collections::HashSet<i32> = maxscore
        .top_docs()
        .iter()
        .map(|score_doc| score_doc.doc_id)
        .collect();
    assert_eq!(kept_docs.len(), 1, "top_n=1 keeps exactly one doc");

    // Pick a doc the maxscore search did NOT keep (and thus, per its own
    // design, may not have even decoded) and confirm explain_clause still
    // reports the correct eager score for it.
    let (pruned_doc, expected_score) = *eager
        .scores
        .iter()
        .find(|(doc_id, _)| !kept_docs.contains(doc_id))
        .expect("at least one doc must have been pruned by top_n=1");

    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Term(query),
        pruned_doc,
        None,
    )
    .unwrap();

    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);
}

/// Builds a synthetic, dense [`FieldNorms`] covering every doc up to
/// `max_doc` with the same norm byte -- exercises the "real opened
/// norms" (`Some(fn_)`) branch in [`explain_term`]/[`explain_phrase`],
/// which the fixture's own real `.nvd`/`.nvm` files aren't wired up for
/// in this crate's existing tests (same synthetic-entry pattern
/// `field_norms.rs`'s own unit tests already use).
fn synthetic_norms(data: &[u8], max_doc: i32) -> FieldNorms<'_> {
    let entry = lucene_codecs::norms::NormsEntry {
        field_number: 0,
        docs_with_field_offset: -1,
        docs_with_field_length: 0,
        jump_table_entry_count: 0,
        // `0xFF` is Java's `denseRankPower == -1`, "no rank table". `0` is
        // not in `IndexedDISI`'s legal set (`-1`, or `7..=15`), so
        // `dense_rank_bytes(0)` rejects it and any DENSE block reached with
        // it fails to decode. Unreachable here -- this entry is dense, so
        // nothing looks at the field -- but it described metadata no writer
        // can produce, which is the kind of almost-right test input that
        // hides a real decode bug. Same fix `field_norms.rs`'s `NO_RANK`
        // constant records. (`field_norms.rs`'s own literal `0` is
        // deliberate and stays: it is the input to
        // `an_illegal_dense_rank_power_is_rejected_rather_than_guessed`,
        // which asserts the decode *refuses* it. Named rather than cited by
        // line number, which drifts.)
        dense_rank_power: 0xFF,
        num_docs_with_field: max_doc,
        bytes_per_norm: 1,
        norms_offset: 0,
    };
    FieldNorms::open(data, entry, max_doc, None).unwrap()
}

#[test]
fn term_explain_with_real_norms_matches_scored_search_output_exactly() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = TermQuery::new("body", "cat");

    let mut capture = ScoreCapture::default();
    search_term_query_scored(&fields, doc_in.as_ref(), None, &query, None, &mut capture).unwrap();
    let (target_doc, _) = capture.scores[0];

    // Norms must cover every doc the term matches, not just `target_doc`
    // -- `search_term_query_scored` computes every matched doc's field
    // length eagerly, in doc-ID order.
    let max_doc = capture.scores.iter().map(|&(d, _)| d).max().unwrap() + 1;
    let data = vec![10u8; max_doc as usize];
    let norms = synthetic_norms(&data, max_doc);

    let mut capture_normed = ScoreCapture::default();
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        &query,
        Some(&norms),
        &mut capture_normed,
    )
    .unwrap();
    let (_, expected_score) = capture_normed
        .scores
        .into_iter()
        .find(|&(d, _)| d == target_doc)
        .unwrap();

    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Term(query),
        target_doc,
        Some(&HashMap::from([("body".to_string(), norms)])),
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);
}

#[test]
fn term_explain_non_matching_doc_is_no_match_with_zero_value() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = TermQuery::new("body", "cat");

    // doc 999999 is far outside this small fixture's doc range.
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Term(query),
        999_999,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
    assert_eq!(explanation.value, 0.0);
}

#[test]
fn term_explain_missing_field_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Term(TermQuery::new("nonexistent", "x")),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
    assert_eq!(explanation.value, 0.0);
}

#[test]
fn term_explain_missing_term_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Term(TermQuery::new("body", "zzz-missing")),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn boolean_explain_matching_doc_equals_scored_search_output_exactly() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_should([TermQuery::new("body", "dog")]);

    let mut capture = ScoreCapture::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut capture,
    )
    .unwrap();
    assert!(!capture.scores.is_empty());
    let (target_doc, expected_score) = capture.scores[0];

    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Boolean(Box::new(query)),
        target_doc,
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);

    // Sub-clause explanation values must sum to the top-level value.
    let sum: f32 = explanation.details.iter().map(|d| d.value).sum();
    assert_eq!(sum, expected_score);
}

#[test]
fn boolean_explain_no_match_when_must_clause_fails() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = BooleanQuery::new().with_must([
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "zzz-missing"),
    ]);
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Boolean(Box::new(query)),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
    assert_eq!(explanation.value, 0.0);
}

#[test]
fn boolean_explain_no_match_for_empty_query() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Boolean(Box::new(BooleanQuery::new())),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn boolean_explain_no_match_for_pure_must_not_query() {
    // Task #60 edge case: a `BooleanQuery` with only `must_not` clauses
    // (no `must`/`should`) matches nothing -- `matched_boolean_docs` already
    // folds this into the same "no must/should clauses" `Ok(None)` case an
    // entirely empty query hits (see that function's doc comment), so
    // `explain_boolean` must report a no-match here too, not "everything
    // except the excluded set."
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
    // Doc 2 doesn't contain "dog" at all -- if pure must_not were buggily
    // treated as "match everything except the excluded set," this doc would
    // wrongly explain as a match.
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Boolean(Box::new(query)),
        2,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
    assert_eq!(explanation.value, 0.0);
}

#[test]
fn boolean_explain_no_match_when_minimum_should_match_exceeds_should_clause_count() {
    // Task #60 edge case: `minimum_should_match` greater than the number of
    // `should` clauses can never be satisfied -- must explain as no-match,
    // not panic or (worse) accidentally match.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_minimum_should_match(5);
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Boolean(Box::new(query)),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
    assert_eq!(explanation.value, 0.0);
}

#[test]
fn boolean_explain_duplicate_should_clause_sums_twice_matching_scored_search() {
    // Edge case: a duplicated `should` clause must contribute its
    // score twice in the explanation, exactly matching
    // `search_boolean_query_scored`'s own double-counting (real Lucene does
    // not dedupe clauses -- see the `lib.rs` test
    // `boolean_duplicate_should_clause_counts_and_scores_twice` for the
    // full rationale). Verified here by requiring bit-for-bit equality
    // against the scored search path, the same technique
    // `boolean_explain_matching_doc_equals_scored_search_output_exactly`
    // already uses.
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")]);

    let mut capture = ScoreCapture::default();
    search_boolean_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut capture,
    )
    .unwrap();
    assert!(!capture.scores.is_empty());
    let (target_doc, expected_score) = capture.scores[0];

    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Boolean(Box::new(query)),
        target_doc,
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);
    // Two should-clause detail entries, both for "cat", summing to the
    // duplicated total -- not deduplicated to one.
    assert_eq!(explanation.details.len(), 2);
}

#[test]
fn dismax_explain_matching_doc_equals_scored_search_output_exactly() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = DisjunctionMaxQuery::new(
        [
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ],
        0.5,
    );

    let mut capture = ScoreCapture::default();
    search_disjunction_max_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        None,
        &query,
        None,
        &mut capture,
    )
    .unwrap();
    assert!(!capture.scores.is_empty());
    let (target_doc, expected_score) = capture.scores[0];

    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::DisjunctionMax(Box::new(query)),
        target_doc,
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);
}

#[test]
fn dismax_explain_no_disjuncts_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = DisjunctionMaxQuery::new(Vec::<Clause>::new(), 0.0);
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::DisjunctionMax(Box::new(query)),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn dismax_explain_no_matching_disjunct_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query =
        DisjunctionMaxQuery::new([Clause::Term(TermQuery::new("body", "zzz-missing"))], 0.0);
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::DisjunctionMax(Box::new(query)),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn constant_score_explain_matching_doc_reports_the_constant() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = ConstantScoreQuery::new(TermQuery::new("body", "cat"), 2.5);

    let mut capture = ScoreCapture::default();
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "cat"),
        None,
        &mut capture,
    )
    .unwrap();
    let (target_doc, _) = capture.scores[0];

    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::ConstantScore(Box::new(query)),
        target_doc,
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, 2.5);
}

#[test]
fn constant_score_explain_non_matching_doc_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = ConstantScoreQuery::new(TermQuery::new("body", "zzz-missing"), 2.5);
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::ConstantScore(Box::new(query)),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
    assert_eq!(explanation.value, 0.0);
}

#[test]
fn boost_explain_matching_doc_multiplies_inner_score() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());

    let mut capture = ScoreCapture::default();
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "cat"),
        None,
        &mut capture,
    )
    .unwrap();
    let (target_doc, inner_score) = capture.scores[0];

    let query = BoostQuery::new(TermQuery::new("body", "cat"), 3.0);
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Boost(Box::new(query)),
        target_doc,
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, inner_score * 3.0);
}

/// `createWeight(..., boost)`: the boosts around a dis-max of terms
/// multiply into the term weight's `boost` node; around anything else
/// the boost stays a `product of:`; `MatchNoDocsQuery` explains as its
/// reason.
#[test]
fn boosts_fold_into_term_weights_and_wrap_the_rest() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let mut capture = ScoreCapture::default();
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "cat"),
        None,
        &mut capture,
    )
    .unwrap();
    let target_doc = capture.scores[0].0;
    let explain = |c: &Clause, doc: i32| {
        explain_clause(&fields, doc_in.as_ref(), None, None, None, c, doc, None).unwrap()
    };
    let dismax = DisjunctionMaxQuery::new([TermQuery::new("body", "cat")], 0.0);
    let q = Clause::Boost(Box::new(BoostQuery::new(BoostQuery::new(dismax, 2.0), 1.5)));
    let e = explain(&q, target_doc);
    assert!(e.matched);
    let text = e.to_string();
    assert!(text.contains("3.0 = boost"), "{text}");
    assert!(!text.contains("product of"), "{text}");
    assert_eq!(weight_boost(), 1.0, "the boost is restored afterwards");

    let wildcard = Clause::Boost(Box::new(BoostQuery::new(
        WildcardQuery::new("body", "ca*"),
        2.0,
    )));
    let e = explain(&wildcard, target_doc);
    assert_eq!(e.description, "product of:");
    assert!(!takes_boost(&Clause::Wildcard(WildcardQuery::new(
        "body", "c*"
    ))));

    let none = Clause::MatchNoDocs(crate::query::MatchNoDocsQuery::new().with_reason("why"));
    assert!(takes_boost(&none));
    let e = explain(&none, target_doc);
    assert!(!e.matched);
    assert_eq!(e.description, "why");
}

#[test]
fn boost_explain_non_matching_doc_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = BoostQuery::new(TermQuery::new("body", "zzz-missing"), 3.0);
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Boost(Box::new(query)),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn wildcard_explain_matching_doc_is_flat_constant_score() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let matched = crate::wildcard_doc_ids(
        &fields,
        doc_in.as_ref(),
        None,
        &WildcardQuery::new("body", "ca*"),
    )
    .unwrap();
    assert!(!matched.is_empty());
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Wildcard(WildcardQuery::new("body", "ca*")),
        matched[0],
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, 1.0);
}

#[test]
fn wildcard_explain_non_matching_doc_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Wildcard(WildcardQuery::new("body", "zzz-nomatch*")),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
    assert_eq!(explanation.value, 0.0);
}

#[test]
fn prefix_explain_matching_doc_is_flat_constant_score() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let matched = crate::prefix_doc_ids(
        &fields,
        doc_in.as_ref(),
        None,
        &PrefixQuery::new("body", "ca"),
    )
    .unwrap();
    assert!(!matched.is_empty());
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Prefix(PrefixQuery::new("body", "ca")),
        matched[0],
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, 1.0);
}

#[test]
fn fuzzy_explain_matching_and_non_matching_docs() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = crate::FuzzyQuery::new("body", "cat");
    let matched = crate::fuzzy_doc_ids(&fields, doc_in.as_ref(), None, &query).unwrap();
    assert!(!matched.is_empty());

    let hit = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Fuzzy(query.clone()),
        matched[0],
        None,
    )
    .unwrap();
    assert!(hit.matched);
    assert_eq!(hit.value, 1.0);

    let miss = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Fuzzy(crate::FuzzyQuery::new("body", "zzzzzzzzzz")),
        0,
        None,
    )
    .unwrap();
    assert!(!miss.matched);
    assert_eq!(miss.value, 0.0);
}

#[test]
fn regexp_explain_matching_and_non_matching_docs() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = crate::RegexpQuery::new("body", "ca.*");
    let matched = crate::regexp_doc_ids(&fields, doc_in.as_ref(), None, &query).unwrap();
    assert!(!matched.is_empty());

    let hit = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Regexp(query),
        matched[0],
        None,
    )
    .unwrap();
    assert!(hit.matched);
    assert_eq!(hit.value, 1.0);

    let miss = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Regexp(crate::RegexpQuery::new("body", "zzz-nomatch.*")),
        0,
        None,
    )
    .unwrap();
    assert!(!miss.matched);
}

#[test]
fn span_explain_matching_and_non_matching_docs() {
    let (fields, doc) = open_fixture();
    let doc = doc.expect("fixture has an opened .doc file");
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let query = crate::SpanQuery::span_term("pos", "alpha");
    let matched = crate::span_doc_ids(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &query,
    )
    .unwrap();
    assert!(!matched.is_empty());

    let hit = explain_clause(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &Clause::Span(query.clone()),
        matched[0],
        None,
    )
    .unwrap();
    assert!(hit.matched);
    // `SpanWeight.explain`'s value is the score the scorer gives the
    // document, not a flat constant.
    let scored = crate::exec::span::resolve_span(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        None,
        None,
        &query,
    )
    .unwrap();
    assert_eq!(scored[0], (matched[0], hit.value));
    assert_ne!(hit.value, 1.0);
    // Unscored, the same documents; nothing for a query with no leaf, no
    // leaf in the index, or leaves that never meet.
    let ctx = crate::exec::LeafContext {
        fields: &fields,
        doc_in: Some(&doc_in),
        pos_in: Some(&pos_in),
        pay_in: Some(&pay_in),
        live_docs: None,
        points: None,
        norms: None,
        global: None,
        max_doc: None,
        cache: None,
        reader: None,
        similarity: None,
    };
    let (docs, scores) = crate::exec::span::span_doc_scores(&ctx, &query, 1.0, false)
        .unwrap()
        .unwrap();
    assert_eq!(docs, matched);
    assert!(scores.is_empty());
    for nothing in [
        crate::SpanQuery::span_or(std::iter::empty()),
        crate::SpanQuery::span_term("pos", "zzz-missing"),
        crate::SpanQuery::span_near(
            [
                crate::SpanQuery::span_term("pos", "alpha"),
                crate::SpanQuery::span_term("pos", "zzz-missing"),
            ],
            0,
            true,
        ),
    ] {
        assert!(
            crate::exec::span::span_doc_scores(&ctx, &nothing, 1.0, true)
                .unwrap()
                .is_none()
        );
    }

    let miss_query = crate::SpanQuery::span_term("pos", "zzz-missing");
    let miss = explain_clause(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &Clause::Span(miss_query),
        0,
        None,
    )
    .unwrap();
    assert!(!miss.matched);
}

#[test]
fn phrase_explain_multi_term_matching_doc_equals_scored_search_output_exactly() {
    let (fields, doc) = open_fixture();
    let doc = doc.expect("fixture has an opened .doc file");
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();

    // "alpha beta" is known (see `lib.rs`'s own phrase tests) to match
    // real doc 8555 in the "pos" field exactly, at slop 0.
    let query = crate::PhraseQuery::new("pos", ["alpha", "beta"]);

    let mut capture = ScoreCapture::default();
    crate::search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &query,
        None,
        &mut capture,
    )
    .unwrap();
    assert!(
        !capture.scores.is_empty(),
        "fixture must have a doc matching \"alpha beta\""
    );
    let (target_doc, expected_score) = capture.scores[0];

    let explanation = explain_clause(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &Clause::Phrase(query),
        target_doc,
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);

    // Multiply back to the top-level value to within rounding -- see
    // `term_explain_matching_doc_equals_scored_search_output_exactly` for
    // why real Lucene's own `explain` refuses to call this a "product of".
    let score_node = &explanation.details[0];
    let idf_node = &score_node.details[0];
    let tf_norm_node = &score_node.details[1];
    assert!(
        (idf_node.value * tf_norm_node.value - expected_score).abs()
            <= 4.0 * f32::EPSILON * expected_score.abs().max(1.0),
        "idf {} * tfNorm {} vs score {expected_score}",
        idf_node.value,
        tf_norm_node.value
    );
}

/// Closes a coverage gap the field norms/sloppy-match branches of
/// `explain_phrase` weren't exercised by: `slop != 0` and the real-norms
/// path. Reuses `GenBlockTree.java`'s known sloppy-gap fixture doc
/// (`"alpha"@0`, `"beta"@3` -- a real 2-move gap) with a real
/// `FieldNorms` built the same way `term_explain_with_real_norms_...`
/// does for `Clause::Term`, at a slop large enough to match.
#[test]
fn phrase_explain_sloppy_match_with_real_norms_equals_scored_search_output_exactly() {
    let (fields, doc) = open_fixture();
    let doc = doc.expect("fixture has an opened .doc file");
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();

    let dir = concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../fixtures/data/blocktree_index/"
    );
    let manifest = std::fs::read_to_string(format!("{dir}manifest.properties")).unwrap();
    let get = |key: &str| -> String {
        manifest
            .lines()
            .find_map(|l| l.strip_prefix(&format!("{key}=")))
            .unwrap_or_else(|| panic!("manifest key {key} missing"))
            .to_string()
    };
    let gap_doc: i32 = get("field.pos.sloppyGapDoc").parse().unwrap();
    let moves_needed: u32 = get("field.pos.sloppyGap.movesNeeded").parse().unwrap();
    assert_eq!(
        moves_needed, 2,
        "fixture's known gap size changed underneath this test"
    );

    let query = crate::PhraseQuery::new("pos", ["alpha", "beta"]).with_slop(moves_needed);

    let max_doc = gap_doc + 1;
    let data = vec![10u8; max_doc as usize];
    let norms = synthetic_norms(&data, max_doc);

    let mut capture = ScoreCapture::default();
    crate::search_phrase_query_scored(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &query,
        Some(&norms),
        &mut capture,
    )
    .unwrap();
    let (target_doc, expected_score) = *capture
        .scores
        .iter()
        .find(|(d, _)| *d == gap_doc)
        .unwrap_or_else(|| panic!("expected doc {gap_doc} to match at slop={moves_needed}"));

    let norms_map: HashMap<String, FieldNorms> = HashMap::from([("pos".to_string(), norms)]);
    let explanation = explain_clause(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &Clause::Phrase(query),
        target_doc,
        Some(&norms_map),
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);
}

#[test]
fn phrase_explain_no_alignment_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc = doc.expect("fixture has an opened .doc file");
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();

    // Reversed order ("beta alpha") never aligns in this fixture.
    let query = crate::PhraseQuery::new("pos", ["beta", "alpha"]);
    let explanation = explain_clause(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &Clause::Phrase(query),
        8555,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn phrase_explain_missing_field_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc = doc.expect("fixture has an opened .doc file");
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let query = crate::PhraseQuery::new("nonexistent", ["a", "b"]);
    let explanation = explain_clause(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &Clause::Phrase(query),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn phrase_explain_missing_term_in_multi_term_phrase_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc = doc.expect("fixture has an opened .doc file");
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let query = crate::PhraseQuery::new("pos", ["alpha", "zzz-missing"]);
    let explanation = explain_clause(
        &fields,
        Some(&doc_in),
        Some(&pos_in),
        Some(&pay_in),
        None,
        &Clause::Phrase(query),
        8555,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn phrase_explain_empty_terms_is_no_match() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Phrase(crate::PhraseQuery::default()),
        0,
        None,
    )
    .unwrap();
    assert!(!explanation.matched);
}

#[test]
fn phrase_explain_single_term_delegates_to_term_explain() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let query = crate::PhraseQuery::new("body", ["cat"]);

    let mut capture = ScoreCapture::default();
    search_term_query_scored(
        &fields,
        doc_in.as_ref(),
        None,
        &TermQuery::new("body", "cat"),
        None,
        &mut capture,
    )
    .unwrap();
    let (target_doc, expected_score) = capture.scores[0];

    let explanation = explain_clause(
        &fields,
        doc_in.as_ref(),
        None,
        None,
        None,
        &Clause::Phrase(query),
        target_doc,
        None,
    )
    .unwrap();
    assert!(explanation.matched);
    assert_eq!(explanation.value, expected_score);
}

#[test]
fn phrase_explain_over_a_field_without_positions_is_javas_error() {
    // `body` is `DOCS_AND_FREQS`: `getPhraseMatcher`'s check fires first.
    let (fields, _doc) = open_fixture();
    let query = crate::PhraseQuery::new("body", ["quick", "fox"]);
    let err = explain_clause(
        &fields,
        None,
        None,
        None,
        None,
        &Clause::Phrase(query),
        0,
        None,
    )
    .unwrap_err();
    assert_eq!(
            err.to_string(),
            "field \"body\" was indexed without position data; cannot run PhraseQuery (phrase=body:\"quick fox\")"
        );
}

#[test]
fn phrase_explain_on_a_field_the_segment_lacks_is_no_match_without_pos_input() {
    let (fields, _doc) = open_fixture();
    let query = crate::PhraseQuery::new("nosuchfield", ["quick", "fox"]);
    let e = explain_clause(
        &fields,
        None,
        None,
        None,
        None,
        &Clause::Phrase(query),
        0,
        None,
    )
    .unwrap();
    assert!(!e.matched);
}

#[test]
fn explanation_match_and_no_match_constructors() {
    let m =
        Explanation::match_(1.5, "matched").with_details(vec![Explanation::match_(1.0, "child")]);
    assert!(m.matched);
    assert_eq!(m.value, 1.5);
    assert_eq!(m.details.len(), 1);

    let nm = Explanation::no_match("nope");
    assert!(!nm.matched);
    assert_eq!(nm.value, 0.0);
    assert!(nm.details.is_empty());
}

/// The positional leaves of a filter's or an exclusion's subtree explain
/// as Java's weights created without scores do: a phrase's (and a
/// multi-phrase's) dummy scorer scores `1` over its frequency, a
/// one-position multi-phrase and a synonym query are booleans of their
/// terms, a span query has no similarity -- and only the similarity's
/// name changes under another one.
#[test]
fn weights_without_scores_explain_through_their_dummy_scorers() {
    let (fields, doc) = open_fixture();
    let doc = doc.expect("fixture has an opened .doc file");
    let doc_in = doc.open();
    let pos_in = doc.open_pos();
    let pay_in = doc.open_pay();
    let explain = |clause: &Clause, d: i32| {
        with_no_scores(|| {
            explain_clause(
                &fields,
                Some(&doc_in),
                Some(&pos_in),
                Some(&pay_in),
                None,
                clause,
                d,
                None,
            )
        })
        .unwrap()
    };
    let phrase = crate::PhraseQuery::new("pos", ["alpha", "beta"]);
    let freqs = crate::phrase_doc_freqs(
        fields.field("pos").unwrap(),
        Some(&doc_in),
        &pos_in,
        Some(&pay_in),
        None,
        &phrase,
    )
    .unwrap();
    let matched: Vec<i32> = freqs.iter().map(|&(d, _)| d).collect();
    let d = freqs
        .iter()
        .find(|&&(_, f)| f == 1.0)
        .expect("a document with the phrase once")
        .0;
    let e = explain(&Clause::Phrase(phrase), d);
    assert_eq!(
        e.to_string(),
        format!(
            "1.0 = weight(pos:\"alpha beta\" in {d}) [BM25Similarity], result of:\n  \
                 1.0 = score(freq=1.0), with freq of:\n    1.0 = phraseFreq=1.0\n"
        )
    );
    // A multi-phrase of one term per position is the same phrase.
    let multi = crate::query::MultiPhraseQuery::new("pos", [["alpha"], ["beta"]]);
    let m = explain(&Clause::MultiPhrase(multi.clone()), d);
    assert_eq!((m.value, &m.details), (e.value, &e.details));
    assert!(m.description.starts_with("weight(pos:\"alpha beta\" in "));
    let no_doc = (0..)
        .find(|x| !matched.contains(x))
        .expect("a document without the phrase");
    assert_eq!(
        explain(&Clause::MultiPhrase(multi), no_doc).description,
        "no matching terms"
    );
    // One position: the boolean of its terms, and a synonym query too.
    let terms = Clause::from(BooleanQuery {
        should: vec![
            Clause::Term(TermQuery::new("pos", "alpha")),
            Clause::Term(TermQuery::new("pos", "beta")),
        ],
        ..BooleanQuery::default()
    });
    let want = explain(&terms, d);
    assert_eq!(want.description, "sum of:");
    assert_eq!(want.value, 0.0);
    let one_position = crate::query::MultiPhraseQuery::new("pos", [["alpha", "beta"]]);
    assert_eq!(explain(&Clause::MultiPhrase(one_position), d), want);
    let synonym = crate::extended_query::SynonymQuery::new(
        "pos",
        [(b"alpha".to_vec(), 1.0), (b"beta".to_vec(), 0.5)],
    )
    .unwrap();
    assert_eq!(explain(&Clause::from(synonym), d), want);
    // A span query: matched, without a score.
    let span = crate::SpanQuery::span_term("pos", "alpha");
    let s = explain(&Clause::Span(span), d);
    assert_eq!(
        (s.matched, s.value, s.description.as_str()),
        (
            true,
            0.0,
            format!("match spanTerm(pos:alpha) in {d} without score").as_str()
        )
    );
    // Another similarity: the same dummy scorers, under its name.
    let classic = crate::similarities::ClassicSimilarity::default();
    let term = Clause::Term(TermQuery::new("pos", "alpha"));
    let t = with_leaf(None, 0, None, Some(&classic), || explain(&term, d));
    assert_eq!(
        t.to_string(),
        format!(
            "0.0 = weight(pos:alpha in {d}) [ClassicSimilarity], result of:\n  \
                 0.0 = score(freq=1.0), with freq of:\n    \
                 1.0 = freq, occurrences of term within document\n"
        )
    );
}

#[test]
fn dismax_explains_its_value_in_doubles_as_java() {
    // `TestSimpleExplanations.testDMQ8`'s doc 3: Java's explain and
    // scorer both give 28344.451
    let v = crate::dismax_value([0.283_443_1f32, 28_344.309].into_iter(), 0.5);
    assert_eq!(v, 28_344.451f32);
    // the float arithmetic explain had (`max + tie * (sum - max)`) is
    // one ulp below
    let (lo, hi) = (0.283_443_1f32, 28_344.309f32);
    assert_eq!(hi + 0.5 * ((lo + hi) - hi), 28_344.45f32);
    // Java's `score >= max` from 0: a negative score is an "other"
    assert_eq!(crate::dismax_value([-2.0f32, -1.0].into_iter(), 1.0), -3.0);
}

/// The scored search's dis-max is `dismax_value`'s doubles, as
/// `DisjunctionMaxScorer.score`: over `testDMQ8`-shaped disjuncts (a
/// small score and a hugely boosted one, tie 0.5) every document both
/// match scores `dismax_value` of its two disjunct scores, and on some
/// of them the float formula would round differently.
#[test]
fn dismax_search_scores_in_doubles_as_java() {
    let (fields, doc) = open_fixture();
    let doc_in = doc.as_ref().map(|d| d.open());
    let scored = |q: &DisjunctionMaxQuery| {
        let mut capture = ScoreCapture::default();
        search_disjunction_max_query_scored(
            &fields,
            doc_in.as_ref(),
            None,
            None,
            None,
            None,
            q,
            None,
            &mut capture,
        )
        .unwrap();
        capture.scores
    };
    let mut differs = 0;
    let mut compared = 0;
    for boost in [100.0f32, 1_000.0, 100_000.0, 12_345.678, 3.0e6] {
        let small = Clause::Term(TermQuery::new("body", "cat"));
        let big = Clause::Boost(Box::new(crate::query::BoostQuery::new(
            Clause::Term(TermQuery::new("body", "dog")),
            boost,
        )));
        let alone = |c: &Clause| -> std::collections::HashMap<i32, f32> {
            scored(&DisjunctionMaxQuery::new([c.clone()], 0.5))
                .into_iter()
                .collect()
        };
        let (lo, hi) = (alone(&small), alone(&big));
        for (d, score) in scored(&DisjunctionMaxQuery::new([small, big], 0.5)) {
            let (Some(&a), Some(&b)) = (lo.get(&d), hi.get(&d)) else {
                continue;
            };
            compared += 1;
            assert_eq!(
                score,
                crate::dismax_value([a, b].into_iter(), 0.5),
                "doc {d}"
            );
            let max = a.max(b);
            if max + 0.5 * ((a + b) - max) != score {
                differs += 1;
            }
        }
    }
    assert!(compared > 0, "no document matches both disjuncts");
    assert!(differs > 0, "no boost separates doubles from floats");
}
