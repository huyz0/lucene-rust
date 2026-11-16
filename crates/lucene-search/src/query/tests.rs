#![cfg(test)]
use super::*;

#[test]
fn new_stores_field_and_term_bytes() {
    let q = TermQuery::new("body", "cat");
    assert_eq!(q.field, "body");
    assert_eq!(q.term, b"cat");
}

#[test]
fn equality_is_field_and_term_based() {
    assert_eq!(TermQuery::new("body", "cat"), TermQuery::new("body", "cat"));
    assert_ne!(TermQuery::new("body", "cat"), TermQuery::new("body", "dog"));
    assert_ne!(TermQuery::new("body", "cat"), TermQuery::new("id", "cat"));
}

#[test]
fn boolean_query_default_is_all_empty_clause_lists() {
    let q = BooleanQuery::new();
    assert!(q.must.is_empty());
    assert!(q.should.is_empty());
    assert!(q.must_not.is_empty());
    assert_eq!(q.minimum_should_match, 0);
}

#[test]
fn boolean_query_builder_methods_populate_each_clause_bucket() {
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_should([TermQuery::new("body", "dog")])
        .with_must_not([TermQuery::new("body", "bird")]);
    assert_eq!(q.must, vec![Clause::Term(TermQuery::new("body", "cat"))]);
    assert_eq!(q.should, vec![Clause::Term(TermQuery::new("body", "dog"))]);
    assert_eq!(
        q.must_not,
        vec![Clause::Term(TermQuery::new("body", "bird"))]
    );
    assert_eq!(q.minimum_should_match, 0);
}

#[test]
fn clause_from_term_query_wraps_in_term_variant() {
    let clause: Clause = TermQuery::new("body", "cat").into();
    assert_eq!(clause, Clause::Term(TermQuery::new("body", "cat")));
}

#[test]
fn clause_from_boolean_query_wraps_in_boxed_boolean_variant() {
    let nested = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
    let clause: Clause = nested.clone().into();
    assert_eq!(clause, Clause::Boolean(Box::new(nested)));
}

#[test]
fn with_must_accepts_a_nested_boolean_query_clause() {
    let nested = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
    let q = BooleanQuery::new().with_must([nested.clone()]);
    assert_eq!(q.must, vec![Clause::Boolean(Box::new(nested))]);
}

#[test]
fn nested_boolean_clauses_can_recurse_to_multiple_levels() {
    // A 3-level tree: top.must = [inner], inner.must = [innermost], innermost.must
    // = [TermQuery] -- confirms `Clause::Boolean` genuinely nests, not just one
    // extra level.
    let innermost = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
    let inner = BooleanQuery::new().with_must([innermost.clone()]);
    let top = BooleanQuery::new().with_must([inner.clone()]);

    let Clause::Boolean(top_inner) = &top.must[0] else {
        panic!("expected a nested Boolean clause");
    };
    assert_eq!(**top_inner, inner);
    let Clause::Boolean(inner_innermost) = &top_inner.must[0] else {
        panic!("expected a nested Boolean clause");
    };
    assert_eq!(**inner_innermost, innermost);
}

#[test]
fn boolean_query_with_minimum_should_match_sets_the_field() {
    let q = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_minimum_should_match(2);
    assert_eq!(q.minimum_should_match, 2);
}

#[test]
fn phrase_query_new_stores_field_and_terms_in_order() {
    let q = PhraseQuery::new("body", ["quick", "brown", "fox"]);
    assert_eq!(q.field, "body");
    assert_eq!(
        q.terms,
        vec![b"quick".to_vec(), b"brown".to_vec(), b"fox".to_vec()]
    );
    assert_eq!(q.slop, 0);
}

#[test]
fn phrase_query_default_is_empty() {
    let q = PhraseQuery::default();
    assert_eq!(q.field, "");
    assert!(q.terms.is_empty());
    assert_eq!(q.slop, 0);
}

#[test]
fn phrase_query_with_slop_sets_the_field() {
    let q = PhraseQuery::new("body", ["quick", "fox"]).with_slop(2);
    assert_eq!(q.slop, 2);
}

#[test]
fn clause_from_phrase_query_wraps_in_phrase_variant() {
    let clause: Clause = PhraseQuery::new("body", ["quick", "fox"]).into();
    assert_eq!(
        clause,
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
    );
}

#[test]
fn with_must_accepts_a_phrase_query_clause() {
    let q = BooleanQuery::new().with_must([PhraseQuery::new("body", ["quick", "fox"])]);
    assert_eq!(
        q.must,
        vec![Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))]
    );
}

#[test]
fn phrase_query_equality_is_field_and_terms_based() {
    assert_eq!(
        PhraseQuery::new("body", ["a", "b"]),
        PhraseQuery::new("body", ["a", "b"])
    );
    assert_ne!(
        PhraseQuery::new("body", ["a", "b"]),
        PhraseQuery::new("body", ["a", "c"])
    );
    assert_ne!(
        PhraseQuery::new("body", ["a", "b"]),
        PhraseQuery::new("id", ["a", "b"])
    );
}

#[test]
fn constant_score_query_new_stores_inner_and_score() {
    let q = ConstantScoreQuery::new(TermQuery::new("body", "cat"), 2.0);
    assert_eq!(*q.inner, Clause::Term(TermQuery::new("body", "cat")));
    assert_eq!(q.score, 2.0);
}

#[test]
fn clause_from_constant_score_query_wraps_in_boxed_variant() {
    let inner = ConstantScoreQuery::new(TermQuery::new("body", "cat"), 1.5);
    let clause: Clause = inner.clone().into();
    assert_eq!(clause, Clause::ConstantScore(Box::new(inner)));
}

#[test]
fn boost_query_new_stores_inner_and_boost() {
    let q = BoostQuery::new(TermQuery::new("body", "cat"), 3.0);
    assert_eq!(*q.inner, Clause::Term(TermQuery::new("body", "cat")));
    assert_eq!(q.boost, 3.0);
}

#[test]
fn clause_from_boost_query_wraps_in_boxed_variant() {
    let inner = BoostQuery::new(TermQuery::new("body", "cat"), 2.5);
    let clause: Clause = inner.clone().into();
    assert_eq!(clause, Clause::Boost(Box::new(inner)));
}

/// `BoostQuery.rewrite` bubbles a `MatchNoDocsQuery` up without its
/// boost.
#[test]
fn rewrite_drops_the_boost_of_match_no_docs() {
    let none = Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason("x"));
    let boosted = Clause::Boost(Box::new(BoostQuery::new(none.clone(), 3.0)));
    assert_eq!(boosted.rewrite(), none);
    // An empty boolean rewrites to one, then bubbles up too.
    let empty = Clause::Boost(Box::new(BoostQuery::new(BooleanQuery::new(), 2.0)));
    assert!(matches!(empty.rewrite(), Clause::MatchNoDocs(_)));
}

#[test]
fn match_all_docs_query_new_stores_max_doc() {
    let q = MatchAllDocsQuery::new(5);
    assert_eq!(q.max_doc, 5);
}

#[test]
fn clause_from_match_all_docs_query_wraps_in_variant() {
    let clause: Clause = MatchAllDocsQuery::new(5).into();
    assert_eq!(clause, Clause::MatchAllDocs(MatchAllDocsQuery::new(5)));
}

#[test]
fn match_no_docs_query_default_has_empty_reason() {
    let q = MatchNoDocsQuery::new();
    assert_eq!(q.reason, "");
}

#[test]
fn match_no_docs_query_with_reason_sets_the_field() {
    let q = MatchNoDocsQuery::new().with_reason("rewrite collapsed to nothing");
    assert_eq!(q.reason, "rewrite collapsed to nothing");
}

#[test]
fn clause_from_match_no_docs_query_wraps_in_variant() {
    let clause: Clause = MatchNoDocsQuery::new().into();
    assert_eq!(clause, Clause::MatchNoDocs(MatchNoDocsQuery::new()));
}

#[test]
fn prefix_query_new_stores_field_and_prefix_bytes() {
    let q = PrefixQuery::new("body", "ca");
    assert_eq!(q.field, "body");
    assert_eq!(q.prefix, b"ca");
}

#[test]
fn prefix_query_equality_is_field_and_prefix_based() {
    assert_eq!(
        PrefixQuery::new("body", "ca"),
        PrefixQuery::new("body", "ca")
    );
    assert_ne!(
        PrefixQuery::new("body", "ca"),
        PrefixQuery::new("body", "do")
    );
    assert_ne!(PrefixQuery::new("body", "ca"), PrefixQuery::new("id", "ca"));
}

#[test]
fn clause_from_prefix_query_wraps_in_prefix_variant() {
    let clause: Clause = PrefixQuery::new("body", "ca").into();
    assert_eq!(clause, Clause::Prefix(PrefixQuery::new("body", "ca")));
}

#[test]
fn with_must_accepts_a_prefix_query_clause() {
    let q = BooleanQuery::new().with_must([PrefixQuery::new("body", "ca")]);
    assert_eq!(q.must, vec![Clause::Prefix(PrefixQuery::new("body", "ca"))]);
}

#[test]
fn fuzzy_query_new_uses_real_fuzzy_querys_defaults() {
    let q = FuzzyQuery::new("body", "cat");
    assert_eq!(q.field, "body");
    assert_eq!(q.term, b"cat");
    assert_eq!(q.max_edits, 2);
    assert_eq!(q.prefix_length, 0);
    assert!(q.transpositions);
}

#[test]
fn fuzzy_query_builder_methods_set_each_field() {
    let q = FuzzyQuery::new("body", "cat")
        .with_max_edits(1)
        .with_prefix_length(2)
        .with_transpositions(false);
    assert_eq!(q.max_edits, 1);
    assert_eq!(q.prefix_length, 2);
    assert!(!q.transpositions);
}

#[test]
fn clause_from_fuzzy_query_wraps_in_fuzzy_variant() {
    let clause: Clause = FuzzyQuery::new("body", "cat").into();
    assert_eq!(clause, Clause::Fuzzy(FuzzyQuery::new("body", "cat")));
}

#[test]
fn with_must_accepts_a_fuzzy_query_clause() {
    let q = BooleanQuery::new().with_must([FuzzyQuery::new("body", "cat")]);
    assert_eq!(q.must, vec![Clause::Fuzzy(FuzzyQuery::new("body", "cat"))]);
}

#[test]
fn regexp_query_new_stores_field_and_pattern() {
    let q = RegexpQuery::new("body", "ca.*");
    assert_eq!(q.field, "body");
    assert_eq!(q.pattern, "ca.*");
}

#[test]
fn regexp_query_equality_is_field_and_pattern_based() {
    assert_eq!(
        RegexpQuery::new("body", "ca.*"),
        RegexpQuery::new("body", "ca.*")
    );
    assert_ne!(
        RegexpQuery::new("body", "ca.*"),
        RegexpQuery::new("body", "do.*")
    );
    assert_ne!(
        RegexpQuery::new("body", "ca.*"),
        RegexpQuery::new("id", "ca.*")
    );
}

#[test]
fn clause_from_regexp_query_wraps_in_regexp_variant() {
    let clause: Clause = RegexpQuery::new("body", "ca.*").into();
    assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", "ca.*")));
}

#[test]
fn with_must_accepts_a_regexp_query_clause() {
    let q = BooleanQuery::new().with_must([RegexpQuery::new("body", "ca.*")]);
    assert_eq!(
        q.must,
        vec![Clause::Regexp(RegexpQuery::new("body", "ca.*"))]
    );
}

#[test]
fn rewrite_collapses_single_must_clause_with_no_should_or_must_not() {
    let q = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
    assert_eq!(q.rewrite(), Clause::Term(TermQuery::new("body", "cat")));
}

#[test]
fn rewrite_collapses_single_should_clause_with_default_minimum_should_match() {
    let q = BooleanQuery::new().with_should([TermQuery::new("body", "cat")]);
    assert_eq!(q.rewrite(), Clause::Term(TermQuery::new("body", "cat")));
}

#[test]
fn rewrite_collapses_single_should_clause_with_minimum_should_match_one() {
    let q = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat")])
        .with_minimum_should_match(1);
    assert_eq!(q.rewrite(), Clause::Term(TermQuery::new("body", "cat")));
}

#[test]
fn rewrite_of_a_must_clause_with_an_unreachable_minimum_should_match_is_match_no_docs() {
    // must=[cat], should=[], minimum_should_match=1 matches nothing -- no
    // `should` clause can ever reach the threshold. Collapsing to a bare
    // `cat` clause would silently turn "matches nothing" into "matches
    // whatever cat matches"; real Lucene instead reports it, via the same
    // `shoulds.size() < minimumNumberShouldMatch` rule.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_minimum_should_match(1);
    let rewritten = q.rewrite();
    let Clause::MatchNoDocs(reason) = rewritten else {
        panic!("expected MatchNoDocs, got {rewritten:?}");
    };
    assert_eq!(
        reason.reason.as_str(),
        "SHOULD clause count less than minimumNumberShouldMatch"
    );
}

#[test]
fn rewrite_of_one_should_clause_with_minimum_should_match_two_is_match_no_docs() {
    // Real `BooleanQuery.rewrite()`: `shoulds.size() <
    // minimumNumberShouldMatch` -> `MatchNoDocsQuery("SHOULD clause count
    // less than minimumNumberShouldMatch")`. Leaving the query alone and
    // relying on `matched_boolean_docs` reaching the same "matches nothing"
    // outcome at execution time gives the same answer, but the rewrite
    // could not report it, and a caller inspecting the rewritten query
    // would see one that looks satisfiable.
    let q = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat")])
        .with_minimum_should_match(2);
    let rewritten = q.rewrite();
    let Clause::MatchNoDocs(reason) = rewritten else {
        panic!("expected MatchNoDocs, got {rewritten:?}");
    };
    assert_eq!(
        reason.reason.as_str(),
        "SHOULD clause count less than minimumNumberShouldMatch"
    );
}

#[test]
fn rewrite_does_not_collapse_single_must_clause_with_a_must_not_present() {
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([TermQuery::new("body", "dog")]);
    let rewritten = q.clone().rewrite();
    assert_eq!(
        rewritten,
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "cat"))],
            filter: Vec::new(),
            should: vec![],
            must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_of_a_pure_must_not_only_query_is_match_no_docs() {
    // `BooleanQuery.rewrite()`: `clauses.size() ==
    // clauseSets.get(Occur.MUST_NOT).size()` -> `new
    // MatchNoDocsQuery("pure negative BooleanQuery")`. It is *not* "the
    // must_not clause itself", and it is no longer left structurally
    // intact -- see rule 2's doc comment for why that changed.
    let q = BooleanQuery::new().with_must_not([TermQuery::new("body", "dog")]);
    let Clause::MatchNoDocs(reason) = q.rewrite() else {
        panic!("expected MatchNoDocs");
    };
    assert_eq!(reason.reason.as_str(), "pure negative BooleanQuery");
}

#[test]
fn rewrite_of_a_query_left_pure_negative_by_its_own_recursion_is_match_no_docs() {
    // The case Java only reaches on `IndexSearcher.rewrite`'s *second*
    // pass: the sole `should` clause is a nested query that itself
    // rewrites to `MatchNoDocsQuery`, rule 5 drops it, and what is left is
    // a pure negative query. This port's single bottom-up pass has to
    // catch it in one go, and this is the test that says so.
    let doomed = BooleanQuery::new().with_must_not([TermQuery::new("body", "cat")]);
    let q = BooleanQuery::new()
        .with_should([Clause::from(doomed)])
        .with_must_not([TermQuery::new("body", "dog")]);
    let Clause::MatchNoDocs(reason) = q.rewrite() else {
        panic!("expected MatchNoDocs");
    };
    assert_eq!(
        reason.reason.as_str(),
        "pure negative BooleanQuery",
        "it had a SHOULD clause originally, so it is not the *empty* case"
    );
}

#[test]
fn rewrite_removes_exact_duplicate_must_not_clauses() {
    // Rule 4: duplicate `must_not` clauses collapse to one, mirroring real
    // `BooleanQuery.rewrite()`'s "remove duplicate FILTER and MUST_NOT
    // clauses" step (`MUST_NOT` is stored in a `HashSet` there). Provably
    // safe: `must_not` only ever excludes docs (never scores), so
    // repeating a clause changes neither which docs match nor scores.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "dog"),
        ]);
    let rewritten = q.clone().rewrite();
    assert_eq!(
        rewritten,
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "cat"))],
            filter: Vec::new(),
            should: vec![],
            must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_keeps_distinct_must_not_clauses_and_preserves_order() {
    // Non-duplicate `must_not` clauses are untouched, and dedup keeps the
    // *first* occurrence's position rather than reordering.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "fox"),
            TermQuery::new("body", "dog"),
        ]);
    let rewritten = q.clone().rewrite();
    assert_eq!(
        rewritten,
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "cat"))],
            filter: Vec::new(),
            should: vec![],
            must_not: vec![
                Clause::Term(TermQuery::new("body", "dog")),
                Clause::Term(TermQuery::new("body", "fox")),
            ],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_deduplicates_must_not_clauses_inside_a_nested_boolean_clause() {
    // Rule 4 applies per-`BooleanQuery` level via the ordinary recursive
    // rewrite (rule 3), and the deduped nested query is then *inlined*
    // into its parent by rule 7 ("Inline required / prohibited clauses"),
    // because it is a required clause with no `should` list -- exactly
    // what real `BooleanQuery.rewrite()` does to feed every clause to one
    // conjunction scorer instead of nesting two.
    let inner = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([TermQuery::new("body", "dog"), TermQuery::new("body", "dog")]);
    let outer = BooleanQuery::new()
        .with_must([Clause::from(inner)])
        .with_should([TermQuery::new("body", "bird")]);
    assert_eq!(
        outer.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "cat"))],
            filter: Vec::new(),
            should: vec![Clause::Term(TermQuery::new("body", "bird"))],
            must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_deduplicates_must_not_inside_a_nested_clause_that_cannot_be_inlined() {
    // Same per-level dedup, but on a nested query rule 7 must decline to
    // inline (it has its own `should` list, whose "at least one" floor
    // would be lost if merged into the parent's) -- so the nesting
    // survives and the dedup is visible at the inner level.
    let inner = BooleanQuery::new()
        .with_should([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "bird"),
        ])
        .with_must_not([TermQuery::new("body", "dog"), TermQuery::new("body", "dog")]);
    let outer = BooleanQuery::new()
        .with_must([Clause::from(inner)])
        .with_should([TermQuery::new("body", "fish")]);
    assert_eq!(
        outer.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Boolean(Box::new(BooleanQuery {
                must: vec![],
                filter: Vec::new(),
                should: vec![
                    Clause::Term(TermQuery::new("body", "cat")),
                    Clause::Term(TermQuery::new("body", "bird")),
                ],
                must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
                minimum_should_match: 0,
            }))],
            filter: Vec::new(),
            should: vec![Clause::Term(TermQuery::new("body", "fish"))],
            must_not: vec![],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_of_a_match_no_docs_must_clause_is_the_whole_querys_result() {
    // Rule 5, the `MUST` half. `BooleanQuery.rewrite()`: when a clause
    // rewrites to `MatchNoDocsQuery`, `case MUST: case FILTER: return
    // rewritten;` -- the *inner* query is returned, reason string and all,
    // not a fresh one.
    let q = BooleanQuery::new()
        .with_must([
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason("nothing here")),
        ])
        .with_should([TermQuery::new("body", "bird")]);
    let Clause::MatchNoDocs(reason) = q.rewrite() else {
        panic!("expected MatchNoDocs");
    };
    assert_eq!(
        reason.reason.as_str(),
        "nothing here",
        "the offending clause's own reason must survive, not be replaced"
    );
}

#[test]
fn rewrite_drops_a_match_no_docs_clause_from_should_and_must_not() {
    // Rule 5, the other half: `case SHOULD: case MUST_NOT: // the clause
    // can be safely ignored; break;`. A `SHOULD` that can never match
    // contributes nothing to the union, and a `MUST_NOT` that can never
    // match excludes nothing.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_should([
            Clause::Term(TermQuery::new("body", "bird")),
            Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason("ignored")),
        ])
        .with_must_not([
            Clause::Term(TermQuery::new("body", "dog")),
            Clause::MatchNoDocs(MatchNoDocsQuery::new().with_reason("also ignored")),
        ]);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "cat"))],
            filter: Vec::new(),
            should: vec![Clause::Term(TermQuery::new("body", "bird"))],
            must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_flattens_a_nested_pure_disjunction_out_of_should() {
    // Rule 6. `BooleanQuery.rewrite()`: "Flatten nested disjunctions, this
    // is important for block-max WAND to perform well." Java's
    // `isPureDisjunction()` is "every clause is SHOULD and
    // minimumNumberShouldMatch <= 1", and the *outer* query must also have
    // `minimumNumberShouldMatch <= 1` or the count would change meaning.
    let inner = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let q = BooleanQuery::new()
        .with_should([
            Clause::Term(TermQuery::new("body", "bird")),
            Clause::from(inner),
        ])
        .with_must([TermQuery::new("body", "fish")]);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "fish"))],
            filter: Vec::new(),
            should: vec![
                Clause::Term(TermQuery::new("body", "bird")),
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            must_not: vec![],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_does_not_flatten_a_nested_disjunction_that_is_not_pure() {
    // The four ways `isPureDisjunction()` (`clauses.size() ==
    // getClauses(SHOULD).size() && minimumNumberShouldMatch <= 1`) fails,
    // each of which must leave the nesting alone: the inner query has a
    // `must`, a `must_not` or a `filter`, or carries a
    // `minimumNumberShouldMatch > 1`.
    let impure = [
        BooleanQuery::new()
            .with_should([TermQuery::new("body", "cat")])
            .with_must([TermQuery::new("body", "dog")]),
        BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "bird"),
            ])
            .with_must_not([TermQuery::new("body", "dog")]),
        BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "bird"),
            ])
            .with_minimum_should_match(2),
        // `[(body:cat #body:dog), body:fish]`: flattening this one would
        // silently drop the `#body:dog` restriction from `body:cat`.
        BooleanQuery::new()
            .with_should([
                TermQuery::new("body", "cat"),
                TermQuery::new("body", "bird"),
            ])
            .with_filter([TermQuery::new("body", "dog")]),
    ];
    for inner in impure {
        // The inner query is still *recursively rewritten* (rule 3) -- the
        // third case's `minimumNumberShouldMatch == 2` over two `SHOULD`
        // clauses becomes two `MUST` clauses via rule 9, for instance.
        // What must not happen is its clauses being lifted into the
        // parent, so the expectation is "still one nested clause, whatever
        // rewriting did to its insides".
        let expected_inner = Clause::from(inner.clone()).rewrite();
        let q = BooleanQuery::new().with_should([
            Clause::Term(TermQuery::new("body", "fish")),
            Clause::from(inner),
        ]);
        let Clause::Boolean(rewritten) = q.rewrite() else {
            panic!("expected a BooleanQuery");
        };
        assert_eq!(
            rewritten.should,
            vec![Clause::Term(TermQuery::new("body", "fish")), expected_inner],
            "a non-pure disjunction must not be flattened"
        );
    }
}

#[test]
fn rewrite_does_not_flatten_when_the_outer_minimum_should_match_exceeds_one() {
    // Flattening a 1-clause `SHOULD` into 2 changes what "at least 2 of
    // the should clauses" means, so Java guards the whole block on
    // `minimumNumberShouldMatch <= 1`.
    let inner = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    // Three `SHOULD` clauses against a minimum of 2, so rule 9's
    // "every should is required" case does not also fire and the only
    // thing under test is the flattening guard.
    let q = BooleanQuery::new()
        .with_should([
            Clause::Term(TermQuery::new("body", "bird")),
            Clause::Term(TermQuery::new("body", "fish")),
            Clause::from(inner.clone()),
        ])
        .with_minimum_should_match(2);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![],
            filter: Vec::new(),
            should: vec![
                Clause::Term(TermQuery::new("body", "bird")),
                Clause::Term(TermQuery::new("body", "fish")),
                Clause::from(inner),
            ],
            must_not: vec![],
            minimum_should_match: 2,
        }))
    );
}

#[test]
fn rewrite_of_a_clause_that_is_both_required_and_excluded_is_match_no_docs() {
    // Rule 8, first half. `BooleanQuery.rewrite()`: "Check whether some
    // clauses are both required and excluded" -> `new
    // MatchNoDocsQuery("FILTER or MUST clause also in MUST_NOT")`. Both
    // halves of Java's predicate are reachable
    // (`clauseSets.get(MUST)::contains` **or**
    // `clauseSets.get(FILTER)::contains`), so the reason string is Java's
    // verbatim.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_must_not([TermQuery::new("body", "cat")]);
    let Clause::MatchNoDocs(reason) = q.rewrite() else {
        panic!("expected MatchNoDocs");
    };
    assert_eq!(
        reason.reason.as_str(),
        "FILTER or MUST clause also in MUST_NOT"
    );

    // The `FILTER` half of the same predicate: a clause required only as a
    // filter is just as excluded by a matching `MUST_NOT`.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "dog")])
        .with_filter([TermQuery::new("body", "cat")])
        .with_must_not([TermQuery::new("body", "cat")]);
    let Clause::MatchNoDocs(reason) = q.rewrite() else {
        panic!("expected MatchNoDocs");
    };
    assert_eq!(
        reason.reason.as_str(),
        "FILTER or MUST clause also in MUST_NOT"
    );
}

// ---- `Occur.FILTER` rewrite rules -------------------------------------

#[test]
fn rewrite_of_a_lone_filter_clause_is_a_zero_boosted_constant_score_query() {
    // Java's single-clause optimization: `case FILTER: // no scoring
    // clauses, so return a score of 0 -- return new BoostQuery(new
    // ConstantScoreQuery(query), 0);`. Pinned bit-for-bit against real
    // Lucene in `tests/bm25_scoring_fixtures.rs`
    // (`scoring.boolean.filter.single` scores docs 0 and 2 at exactly 0).
    let q = BooleanQuery::new().with_filter([TermQuery::new("body", "cat")]);
    assert_eq!(
        q.rewrite(),
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                TermQuery::new("body", "cat"),
                1.0
            ))),
            0.0
        )))
    );
}

#[test]
fn rewrite_of_a_filter_only_query_is_not_a_pure_negative_query() {
    // `clauses.size() == clauseSets.get(MUST_NOT).size()` is false for a
    // FILTER clause, so the pure-negative collapse does not apply. Two
    // filters so the single-clause rule above does not fire either.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    assert_eq!(
        q.clone().rewrite(),
        Clause::Boolean(Box::new(q)),
        "a filter-only query rewrites to itself, not to MatchNoDocs"
    );
}

#[test]
fn rewrite_of_a_match_no_docs_filter_clause_collapses_the_whole_query() {
    // Java: `case MUST: case FILTER: return rewritten;` -- a required
    // clause that matches nothing makes the conjunction match nothing.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([Clause::MatchNoDocs(
            MatchNoDocsQuery::new().with_reason("filter reason"),
        )]);
    let Clause::MatchNoDocs(reason) = q.rewrite() else {
        panic!("expected MatchNoDocs");
    };
    assert_eq!(
        reason.reason.as_str(),
        "filter reason",
        "the offending clause's own reason survives, as in Java"
    );
}

#[test]
fn rewrite_removes_duplicate_filter_clauses() {
    // `clauseSets` holds FILTER (and MUST_NOT) in a `HashSet`, so
    // duplicates cannot survive. Unlike MUST/SHOULD dedup, this changes no
    // score: a filter contributes none.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ]);
    let Clause::Boolean(rewritten) = q.rewrite() else {
        panic!("expected a BooleanQuery");
    };
    assert_eq!(
        rewritten.filter,
        vec![
            Clause::Term(TermQuery::new("body", "dog")),
            Clause::Term(TermQuery::new("body", "bird")),
        ]
    );
}

#[test]
fn rewrite_drops_a_filter_clause_that_is_also_a_must_clause() {
    // `filters.removeAll(clauseSets.get(Occur.MUST))`. The MUST copy is the
    // one kept, because it is the scoring one. Pinned against real Lucene
    // as `scoring.boolean.filter.dupmust`, whose recorded scores are
    // bit-identical to `scoring.boolean.must`.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_filter([TermQuery::new("body", "dog")]);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery::new().with_must([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "dog")
        ])))
    );
}

#[test]
fn rewrite_drops_a_match_all_docs_filter_only_when_something_else_constrains() {
    // Dropped: there is a MUST clause, so the filter is pure overhead.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([Clause::MatchAllDocs(MatchAllDocsQuery::new(100))]);
    assert_eq!(
        q.rewrite(),
        Clause::Term(TermQuery::new("body", "cat")),
        "with the redundant filter gone, the single-clause unwrap fires"
    );

    // Dropped: more than one filter, so the others still constrain.
    let q = BooleanQuery::new().with_filter([
        Clause::MatchAllDocs(MatchAllDocsQuery::new(100)),
        Clause::Term(TermQuery::new("body", "cat")),
    ]);
    assert_eq!(
        q.rewrite(),
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                TermQuery::new("body", "cat"),
                1.0
            ))),
            0.0
        ))),
        "one filter left, so the lone-filter rule takes over"
    );

    // NOT dropped: it is the only filter and there is no MUST clause, so
    // removing it would turn "every document, scored 0" into "matches
    // nothing". This is the guard `filters.size() > 1 ||
    // clauseSets.get(MUST).isEmpty() == false` exists for.
    let q = BooleanQuery::new()
        .with_filter([Clause::MatchAllDocs(MatchAllDocsQuery::new(100))])
        .with_must_not([TermQuery::new("body", "dog")]);
    assert_eq!(
        q.clone().rewrite(),
        Clause::Boolean(Box::new(q)),
        "the sole MatchAllDocs filter must survive"
    );
}

#[test]
fn rewrite_promotes_a_clause_that_is_both_filter_and_should_to_must() {
    // Required *and* scored is what MUST means. The FILTER copy goes and
    // `minimumNumberShouldMatch` drops by one, since the promoted clause no
    // longer counts toward the threshold.
    //
    // Java returns `+cat dog~1` from this rule and reaches the final form
    // on `IndexSearcher.rewrite`'s next pass, where rule 9
    // (`shoulds.size() == minimumNumberShouldMatch`) turns the surviving
    // `dog` into a `MUST` too. This port's single bottom-up pass runs rule 9
    // after this one, so it lands on that fixpoint directly.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat")])
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_minimum_should_match(2);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery::new().with_must([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "dog")
        ])))
    );
}

#[test]
fn rewrite_floors_minimum_should_match_at_zero_when_promoting_filters() {
    // `builder.setMinimumNumberShouldMatch(Math.max(0, minShouldMatch))`.
    let q = BooleanQuery::new()
        .with_filter([TermQuery::new("body", "cat")])
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let Clause::Boolean(rewritten) = q.rewrite() else {
        panic!("expected a BooleanQuery");
    };
    assert_eq!(rewritten.minimum_should_match, 0);
    assert_eq!(
        rewritten.must,
        vec![Clause::Term(TermQuery::new("body", "cat"))]
    );
    assert_eq!(
        rewritten.should,
        vec![Clause::Term(TermQuery::new("body", "dog"))]
    );
    assert!(rewritten.filter.is_empty());
}

#[test]
fn rewrite_turns_a_single_match_all_docs_must_plus_filters_into_a_constant_score_query() {
    // "Rewrite queries whose single scoring clause is a MUST clause on a
    // MatchAllDocsQuery to a ConstantScoreQuery."
    let q = BooleanQuery::new()
        .with_must([Clause::MatchAllDocs(MatchAllDocsQuery::new(100))])
        .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    assert_eq!(
        q.rewrite(),
        Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
            BooleanQuery::new()
                .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]),
            1.0
        )))
    );
}

#[test]
fn rewrite_carries_the_match_all_docs_boost_onto_the_constant_score_query() {
    // `if (boost != 1f) rewritten = new BoostQuery(rewritten, boost);`, and
    // the SHOULD clauses are re-attached around the result.
    let q = BooleanQuery::new()
        .with_must([Clause::Boost(Box::new(BoostQuery::new(
            Clause::MatchAllDocs(MatchAllDocsQuery::new(100)),
            3.0,
        )))])
        .with_filter([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_should([TermQuery::new("body", "bird")]);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_must([Clause::Boost(Box::new(BoostQuery::new(
                    Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
                        BooleanQuery::new().with_filter([
                            TermQuery::new("body", "cat"),
                            TermQuery::new("body", "dog")
                        ]),
                        1.0
                    ))),
                    3.0
                )))])
                .with_should([TermQuery::new("body", "bird")])
        ))
    );
}

#[test]
fn rewrite_relabels_an_inner_must_as_a_filter_when_inlining_into_a_filter_clause() {
    // Java: `assert outerClause.occur() == Occur.FILTER && innerOccur ==
    // Occur.MUST; // In this case we need to change the occur of the inner
    // query from MUST to FILTER.` Getting this wrong would make the inner
    // clause score, which is exactly what the outer FILTER forbids.
    let inner = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([TermQuery::new("body", "dog")])
        .with_must_not([TermQuery::new("body", "bird")]);
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "fish")])
        .with_filter([Clause::from(inner)]);
    let Clause::Boolean(rewritten) = q.rewrite() else {
        panic!("expected a BooleanQuery");
    };
    assert_eq!(
        rewritten.must,
        vec![Clause::Term(TermQuery::new("body", "fish"))],
        "the inner MUST must NOT have landed in the parent's scoring bucket"
    );
    assert_eq!(
        rewritten.filter,
        vec![
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ]
    );
    assert_eq!(
        rewritten.must_not,
        vec![Clause::Term(TermQuery::new("body", "bird"))]
    );
}

#[test]
fn rewrite_inlines_an_inner_querys_filter_clauses_into_a_must_clause() {
    // The other direction: an outer MUST over an inner BooleanQuery lifts
    // the inner FILTER clauses as parent FILTER clauses (they stay
    // non-scoring), and the inner MUST clauses as parent MUST clauses.
    let inner = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([TermQuery::new("body", "dog")]);
    let q = BooleanQuery::new()
        .with_must([Clause::from(inner)])
        .with_should([TermQuery::new("body", "bird")]);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat")])
                .with_filter([TermQuery::new("body", "dog")])
                .with_should([TermQuery::new("body", "bird")])
        ))
    );
}

#[test]
fn rewrite_inlines_the_should_clauses_of_the_only_must_clause() {
    // Rule 13: `+(cat dog) -bird` is `cat dog -bird` with at least one of
    // the two required (`~1`); the inner query's own minimum is kept when
    // it is larger.
    let inner = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let q = BooleanQuery::new()
        .with_must([Clause::from(inner)])
        .with_must_not([TermQuery::new("body", "bird")]);
    let mut want = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_must_not([TermQuery::new("body", "bird")]);
    want.minimum_should_match = 1;
    assert_eq!(q.rewrite(), Clause::Boolean(Box::new(want)));

    let mut inner = BooleanQuery::new().with_should([
        TermQuery::new("body", "a"),
        TermQuery::new("body", "b"),
        TermQuery::new("body", "c"),
    ]);
    inner.minimum_should_match = 2;
    let q = BooleanQuery::new()
        .with_must([Clause::from(inner)])
        .with_filter([TermQuery::new("body", "f")]);
    let Clause::Boolean(b) = q.rewrite() else {
        panic!("expected a BooleanQuery");
    };
    assert_eq!(
        (b.minimum_should_match, b.should.len(), b.filter.len()),
        (2, 3, 1)
    );

    // with nothing beside the only MUST clause, the unwrap returns the
    // inner query as it is
    let inner = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let q = BooleanQuery::new().with_must([Clause::from(inner.clone())]);
    assert_eq!(q.rewrite(), Clause::Boolean(Box::new(inner)));
}

#[test]
fn rewrite_leaves_a_filter_alongside_a_must_alone() {
    // The negative case for every rule above: nothing about `+cat #dog` is
    // redundant, so `rewrite` must be a no-op on it.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_filter([TermQuery::new("body", "dog")]);
    assert_eq!(q.clone().rewrite(), Clause::Boolean(Box::new(q)));
}

#[test]
fn rewrite_of_a_must_not_match_all_docs_is_match_no_docs() {
    // Rule 8, second half: `if (mustNotClauses.contains(MatchAllDocsQuery
    // .INSTANCE)) return new MatchNoDocsQuery("MUST_NOT clause is
    // MatchAllDocsQuery");` -- excluding every document leaves none.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([Clause::MatchAllDocs(MatchAllDocsQuery::new(100))]);
    let Clause::MatchNoDocs(reason) = q.rewrite() else {
        panic!("expected MatchNoDocs");
    };
    assert_eq!(
        reason.reason.as_str(),
        "MUST_NOT clause is MatchAllDocsQuery"
    );
}

#[test]
fn rewrite_leaves_a_must_not_alone_when_it_only_resembles_a_must_clause() {
    // The negative case for rule 8's first half: the clauses must be
    // *equal*, not merely same-field or same-shape. Without this, a query
    // like `+body:cat -body:dog` would be destroyed.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat")])
        .with_must_not([TermQuery::new("body", "dog")]);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "cat"))],
            filter: Vec::new(),
            should: vec![],
            must_not: vec![Clause::Term(TermQuery::new("body", "dog"))],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_turns_every_should_into_must_when_the_minimum_equals_their_count() {
    // Rule 9's second half: `if (shoulds.size() > 0 && shoulds.size() ==
    // minimumNumberShouldMatch)` -> every `SHOULD` becomes `MUST`. All of
    // them are required, which is what `MUST` means; `clause_scores` sums
    // `must` and `should` identically, so scores are unchanged too.
    let q = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")])
        .with_minimum_should_match(2);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            filter: Vec::new(),
            should: vec![],
            must_not: vec![],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_collapses_duplicate_must_clauses_into_one_with_summed_boosts() {
    // 10.5.0's `Deduplicate MUST clauses by summing up their boosts`:
    // `+cat +cat` becomes one clause at boost 2 -- and then rule 1's
    // single-clause unwrap takes the `BooleanQuery` away entirely, which
    // is what Java's next `IndexSearcher.rewrite` pass would also do.
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")]);
    assert_eq!(
        q.rewrite(),
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            2.0
        )))
    );
}

#[test]
fn rewrite_sums_explicit_boosts_across_duplicate_should_clauses() {
    // `a^2 a^3` -> `a^5`. The sum is Java's `double` accumulation narrowed
    // once at the end, not three `f32` additions.
    let q = BooleanQuery::new().with_should([
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            2.0,
        ))),
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            3.0,
        ))),
    ]);
    assert_eq!(
        q.rewrite(),
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            5.0
        )))
    );
}

#[test]
fn a_summed_boost_of_exactly_one_leaves_no_boost_wrapper_behind() {
    // Java: `if (boost != 1f) query = new BoostQuery(query, boost);` -- so
    // `a^0.5 a^0.5` collapses to a *bare* `a`, not to `a^1`. This is also
    // the case that makes the re-entry after a rebuild load-bearing: the
    // unwrapped `a` is now equal to the FILTER clause, so rule 10 must get
    // another look at it.
    let q = BooleanQuery::new()
        .with_should([
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                0.5,
            ))),
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "cat")),
                0.5,
            ))),
        ])
        .with_must([TermQuery::new("body", "dog")]);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "dog")])
                .with_should([Clause::Term(TermQuery::new("body", "cat"))])
        ))
    );
}

#[test]
fn nested_boost_wrappers_multiply_before_the_boosts_are_summed() {
    // Java's `while (query instanceof BoostQuery) { boost *= ...; }`:
    // `(a^2)^3` contributes 6, not 3 and not 2.
    let nested = Clause::Boost(Box::new(BoostQuery::new(
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            2.0,
        ))),
        3.0,
    )));
    let q = BooleanQuery::new().with_should([nested, Clause::Term(TermQuery::new("body", "cat"))]);
    assert_eq!(
        q.rewrite(),
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            7.0
        )))
    );
}

#[test]
fn should_dedup_is_skipped_when_minimum_should_match_counts_more_than_one() {
    // Java gates the SHOULD half on `minimumNumberShouldMatch <= 1`: the
    // threshold counts *clauses*, so collapsing two into one would change
    // what the query means. `mSM == 2` with two identical clauses stays a
    // two-clause query -- and, since `should.len() == mSM`, rule 9 then
    // promotes both to MUST, which is where they end up.
    let q = BooleanQuery::new()
        .with_should([TermQuery::new("body", "cat"), TermQuery::new("body", "cat")])
        .with_minimum_should_match(2);
    assert_eq!(
        q.rewrite(),
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Term(TermQuery::new("body", "cat")),
            2.0
        )))
    );
}

#[test]
fn distinct_clauses_are_left_exactly_as_they_were() {
    // Java rebuilds only `if (map.size() != clauseSets.get(occur).size())`
    // -- with no duplicates there is no rebuild, so no clause acquires a
    // `BoostQuery` wrapper and the order is untouched.
    let q = BooleanQuery::new().with_should([
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "dog"),
        TermQuery::new("body", "bird"),
    ]);
    assert_eq!(
        q.rewrite(),
        Clause::Boolean(Box::new(BooleanQuery::new().with_should([
            TermQuery::new("body", "cat"),
            TermQuery::new("body", "dog"),
            TermQuery::new("body", "bird"),
        ])))
    );
}

#[test]
fn duplicate_clauses_collapse_in_first_seen_order() {
    // Java rebuilds from a `HashMap.entrySet()`, whose order is
    // unspecified; this port keeps first-seen order, deliberately -- a
    // rewrite whose output feeds `explain` should be deterministic.
    let q = BooleanQuery::new().with_should([
        TermQuery::new("body", "dog"),
        TermQuery::new("body", "cat"),
        TermQuery::new("body", "dog"),
    ]);
    let Clause::Boolean(rewritten) = q.rewrite() else {
        panic!("expected a BooleanQuery");
    };
    assert_eq!(
        rewritten.should,
        vec![
            Clause::Boost(Box::new(BoostQuery::new(
                Clause::Term(TermQuery::new("body", "dog")),
                2.0
            ))),
            Clause::Term(TermQuery::new("body", "cat")),
        ]
    );
}

#[test]
fn rewrite_of_an_empty_boolean_query_is_match_no_docs() {
    // `BooleanQuery.rewrite()`'s very first line: `if (clauses.size() == 0)
    // return new MatchNoDocsQuery("empty BooleanQuery");`. The reason
    // string is what distinguishes this from the pure-negative case, and
    // only the *original* clause lists can tell them apart.
    let Clause::MatchNoDocs(reason) = BooleanQuery::new().rewrite() else {
        panic!("expected MatchNoDocs");
    };
    assert_eq!(reason.reason.as_str(), "empty BooleanQuery");
}

#[test]
fn rewrite_does_not_collapse_when_more_than_one_clause_is_present() {
    let q = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let rewritten = q.clone().rewrite();
    assert_eq!(
        rewritten,
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![
                Clause::Term(TermQuery::new("body", "cat")),
                Clause::Term(TermQuery::new("body", "dog")),
            ],
            filter: Vec::new(),
            should: vec![],
            must_not: vec![],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_recurses_into_a_nested_boolean_must_clause_before_checking_the_parent() {
    // inner: must=[cat] alone -> collapses to Term(cat). Outer: must=[inner]
    // alone (after inner's own collapse) -> the *outer* BooleanQuery now also
    // has exactly one must clause and no should/must_not, so it collapses too,
    // all the way down to the bare leaf term.
    let inner = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
    let outer = BooleanQuery::new().with_must([inner]);
    assert_eq!(outer.rewrite(), Clause::Term(TermQuery::new("body", "cat")));
}

#[test]
fn rewrite_recurses_into_a_nested_boolean_clause_that_does_not_itself_collapse() {
    // inner has two must clauses, so it does NOT collapse -- but it must still
    // come back as a rewritten (structurally-normalized) nested Boolean clause,
    // proving the recursion actually reaches nested clauses rather than only
    // rewriting the top level.
    let inner = BooleanQuery::new()
        .with_must([TermQuery::new("body", "cat"), TermQuery::new("body", "dog")]);
    let outer = BooleanQuery::new()
        .with_should([inner.clone()])
        .with_must([TermQuery::new("body", "bird")]);
    let rewritten = outer.rewrite();
    assert_eq!(
        rewritten,
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![Clause::Term(TermQuery::new("body", "bird"))],
            filter: Vec::new(),
            should: vec![Clause::Boolean(Box::new(inner))],
            must_not: vec![],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn rewrite_recurses_into_disjunction_max_disjuncts() {
    let single_must = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
    let dismax = DisjunctionMaxQuery::new([Clause::from(single_must)], 0.5);
    let clause: Clause = dismax.into();
    assert_eq!(
        clause.rewrite(),
        Clause::DisjunctionMax(Box::new(DisjunctionMaxQuery::new(
            [Clause::Term(TermQuery::new("body", "cat"))],
            0.5
        )))
    );
}

#[test]
fn rewrite_recurses_into_constant_score_inner_clause() {
    let single_must = BooleanQuery::new().with_must([TermQuery::new("body", "cat")]);
    let csq = ConstantScoreQuery::new(single_must, 2.0);
    let clause: Clause = csq.into();
    assert_eq!(
        clause.rewrite(),
        Clause::ConstantScore(Box::new(ConstantScoreQuery::new(
            TermQuery::new("body", "cat"),
            2.0
        )))
    );
}

#[test]
fn rewrite_recurses_into_boost_inner_clause() {
    let single_should = BooleanQuery::new().with_should([TermQuery::new("body", "cat")]);
    let bq = BoostQuery::new(single_should, 3.0);
    let clause: Clause = bq.into();
    assert_eq!(
        clause.rewrite(),
        Clause::Boost(Box::new(BoostQuery::new(
            TermQuery::new("body", "cat"),
            3.0
        )))
    );
}

#[test]
fn rewrite_leaves_leaf_clauses_unchanged() {
    assert_eq!(
        Clause::Wildcard(WildcardQuery::new("body", "ca*")).rewrite(),
        Clause::Wildcard(WildcardQuery::new("body", "ca*"))
    );
    assert_eq!(
        Clause::Prefix(PrefixQuery::new("body", "ca")).rewrite(),
        Clause::Prefix(PrefixQuery::new("body", "ca"))
    );
    assert_eq!(
        Clause::Fuzzy(FuzzyQuery::new("body", "cat")).rewrite(),
        Clause::Fuzzy(FuzzyQuery::new("body", "cat"))
    );
    assert_eq!(
        Clause::Regexp(RegexpQuery::new("body", "ca.*")).rewrite(),
        Clause::Regexp(RegexpQuery::new("body", "ca.*"))
    );
    assert_eq!(
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"])).rewrite(),
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
    );
}

#[test]
fn with_must_accepts_constant_score_and_boost_query_clauses() {
    let q = BooleanQuery::new().with_must([
        Clause::from(ConstantScoreQuery::new(TermQuery::new("body", "cat"), 1.0)),
        Clause::from(BoostQuery::new(TermQuery::new("body", "dog"), 2.0)),
    ]);
    assert_eq!(q.must.len(), 2);
}
