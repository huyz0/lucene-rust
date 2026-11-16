#![cfg(test)]
use super::*;
use crate::query::{BooleanQuery, PrefixQuery, WildcardQuery};

#[test]
fn deeply_nested_groups_are_refused_instead_of_overflowing_the_stack() {
    // The limits fit an unoptimised build on a 2 MiB thread; a coverage-
    // instrumented build's frames are larger, so run on a big stack.
    std::thread::Builder::new()
        .stack_size(64 << 20)
        .spawn(|| {
            let nested = |depth: usize| format!("{}a{}", "(".repeat(depth), ")".repeat(depth));
            assert!(matches!(
                parse_query(&nested(MAX_GROUP_DEPTH), Some("f")),
                Ok(Clause::Term(_))
            ));
            assert_eq!(
                parse_query(&nested(MAX_GROUP_DEPTH + 1), Some("f")),
                Err(ParseError::TooDeep(MAX_GROUP_DEPTH))
            );
            assert_eq!(
                parse_query(&nested(200_000), Some("f")),
                Err(ParseError::TooDeep(MAX_GROUP_DEPTH))
            );
        })
        .expect("spawn")
        .join()
        .expect("no panic");
}

// --- `AND`/`OR`/`NOT` operators (`QueryParserBase.addClause`) ---

fn boolean(clause: &Clause) -> &BooleanQuery {
    match clause {
        Clause::Boolean(b) => b,
        other => panic!("expected a BooleanQuery, got {other:?}"),
    }
}

// --- `MultiFieldQueryParser` -------------------------------------------

fn term(field: &str, t: &str) -> Clause {
    Clause::Term(TermQuery::new(field, t))
}

fn should_over(clauses: Vec<Clause>) -> Clause {
    Clause::Boolean(Box::new(BooleanQuery::new().with_should(clauses)))
}

#[test]
fn a_bare_term_fans_out_across_every_field_as_should() {
    // `getMultiFieldQuery` adds each field's sub-query with Occur.SHOULD.
    assert_eq!(
        parse_multi_field_query("cat", &["title", "body"], None).unwrap(),
        should_over(vec![term("title", "cat"), term("body", "cat")])
    );
}

#[test]
fn an_explicitly_fielded_term_is_left_alone() {
    // Java's `field != null` branch: no fan-out, no boost lookup.
    let clause = parse_multi_field_query("title:cat dog", &["title", "body"], None).unwrap();
    let b = boolean(&clause);
    assert_eq!(
        b.should,
        vec![
            term("title", "cat"),
            should_over(vec![term("title", "dog"), term("body", "dog")]),
        ]
    );
}

/// The reason the disjunction has to sit at the *leaf*: with
/// `AND_OPERATOR`, Java documents `+(title:t1 body:t1) +(title:t2 body:t2)`
/// -- each term required in *some* field. Expanding at the top instead
/// (`+(title:t1 title:t2) +(body:t1 body:t2)`) would require both terms in
/// the same field, a different query.
#[test]
fn the_disjunction_sits_under_the_conjunction_not_over_it() {
    let clause = parse_multi_field_query_with_boosts(
        "cat dog",
        &[("title", 1.0), ("body", 1.0)],
        None,
        DefaultOperator::And,
    )
    .unwrap();
    let b = boolean(&clause);
    assert!(b.should.is_empty());
    assert_eq!(
        b.must,
        vec![
            should_over(vec![term("title", "cat"), term("body", "cat")]),
            should_over(vec![term("title", "dog"), term("body", "dog")]),
        ]
    );
}

#[test]
fn per_field_boosts_wrap_each_fields_own_clause() {
    // Java's documented example: `+(title:t^5.0 body:t^10.0)`.
    let clause = parse_multi_field_query_with_boosts(
        "cat",
        &[("title", 5.0), ("body", 10.0)],
        None,
        DefaultOperator::Or,
    )
    .unwrap();
    assert_eq!(
        clause,
        should_over(vec![
            Clause::Boost(Box::new(BoostQuery::new(term("title", "cat"), 5.0))),
            Clause::Boost(Box::new(BoostQuery::new(term("body", "cat"), 10.0))),
        ])
    );
    // A boost of 1.0 is Java's "no entry in the boosts map": unwrapped.
    assert_eq!(
        parse_multi_field_query_with_boosts(
            "cat",
            &[("title", 1.0), ("body", 2.0)],
            None,
            DefaultOperator::Or
        )
        .unwrap(),
        should_over(vec![
            term("title", "cat"),
            Clause::Boost(Box::new(BoostQuery::new(term("body", "cat"), 2.0))),
        ])
    );
}

#[test]
fn every_bare_atom_shape_fans_out_not_just_plain_terms() {
    // `getPrefixQuery`/`getWildcardQuery`/`getFuzzyQuery`/`getRegexpQuery`/
    // `getRangeQuery` all have the same `field == null` fan-out.
    let fields = ["title", "body"];
    assert_eq!(
        parse_multi_field_query("ca*", &fields, None).unwrap(),
        should_over(vec![
            Clause::Prefix(PrefixQuery::new("title", "ca")),
            Clause::Prefix(PrefixQuery::new("body", "ca")),
        ])
    );
    let wild = parse_multi_field_query("c?t", &fields, None).unwrap();
    assert_eq!(
        wild,
        should_over(vec![
            Clause::Wildcard(WildcardQuery::new("title", "c?t")),
            Clause::Wildcard(WildcardQuery::new("body", "c?t")),
        ])
    );
    // A quoted phrase, and a phrase with slop.
    let phrase = parse_multi_field_query("\"a b\"~2", &fields, None).unwrap();
    let b = boolean(&phrase);
    assert_eq!(b.should.len(), 2);
    // And a range, whose bare form uses the default field in Java too.
    let range = parse_multi_field_query("[1 TO 9]", &fields, None).unwrap();
    assert_eq!(boolean(&range).should.len(), 2);
}

#[test]
fn a_group_expands_its_members_once_each_not_the_whole_group_again() {
    // `(cat dog)` must be `((title:cat body:cat) (title:dog body:dog))`,
    // not a group duplicated per field.
    let clause = parse_multi_field_query("(cat dog)", &["title", "body"], None).unwrap();
    let b = boolean(&clause);
    assert_eq!(
        b.should,
        vec![
            should_over(vec![term("title", "cat"), term("body", "cat")]),
            should_over(vec![term("title", "dog"), term("body", "dog")]),
        ]
    );
}

#[test]
fn a_query_text_boost_over_a_fanned_out_atom_keeps_working() {
    // `cat^2` over two fields: the boost lands inside each expansion here
    // and outside the disjunction in Java. A SHOULD BooleanQuery sums its
    // clauses' scores, so `sum(b * s_i) == b * sum(s_i)` -- the same query.
    let clause = parse_multi_field_query("cat^2", &["title", "body"], None).unwrap();
    assert_eq!(
        clause,
        should_over(vec![
            Clause::Boost(Box::new(BoostQuery::new(term("title", "cat"), 2.0))),
            Clause::Boost(Box::new(BoostQuery::new(term("body", "cat"), 2.0))),
        ])
    );
}

#[test]
fn one_field_is_exactly_the_single_field_parser() {
    assert_eq!(
        parse_multi_field_query("cat AND dog", &["body"], None).unwrap(),
        parse_query("cat AND dog", Some("body")).unwrap()
    );
}

#[test]
fn no_fields_and_empty_input_are_clean_errors() {
    assert!(matches!(
        parse_multi_field_query("cat", &[], None),
        Err(ParseError::MissingField(0))
    ));
    assert!(matches!(
        parse_multi_field_query("   ", &["body"], None),
        Err(ParseError::EmptyQuery)
    ));
}

#[test]
fn a_parse_error_inside_a_fanned_out_atom_still_surfaces() {
    // The re-parse must propagate its error, not swallow it.
    assert!(parse_multi_field_query("cat^bad", &["title", "body"], None).is_err());
}

#[test]
fn multi_field_parsing_runs_the_analyzer_the_same_way_for_every_field() {
    let analyzer = lucene_analysis::Analyzer::standard(None);
    let clause = parse_multi_field_query("Cats", &["title", "body"], Some(&analyzer)).unwrap();
    let b = boolean(&clause);
    assert_eq!(b.should.len(), 2);
    // Whatever the analyzer produced, both fields got the same term.
    let terms: Vec<String> = b
        .should
        .iter()
        .map(|c| match c {
            Clause::Term(t) => String::from_utf8(t.term.clone()).unwrap(),
            other => panic!("expected a term, got {other:?}"),
        })
        .collect();
    assert_eq!(terms[0], terms[1]);
}

#[test]
fn and_makes_both_sides_required() {
    // Java's retroactive rewrite: the clause *before* an AND becomes MUST
    // too, so `a AND b` is `+a +b`, not `a +b`.
    let clause = parse_query("cat AND dog", Some("body")).unwrap();
    let b = boolean(&clause);
    assert_eq!(
        b.must,
        vec![
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ]
    );
    assert!(b.should.is_empty() && b.must_not.is_empty());
}

#[test]
fn ampersand_ampersand_is_an_alias_for_and() {
    assert_eq!(
        parse_query("cat && dog", Some("body")).unwrap(),
        parse_query("cat AND dog", Some("body")).unwrap()
    );
    assert_eq!(
        parse_query("cat || dog", Some("body")).unwrap(),
        parse_query("cat OR dog", Some("body")).unwrap()
    );
    assert_eq!(
        parse_query("cat AND !dog", Some("body")).unwrap(),
        parse_query("cat AND NOT dog", Some("body")).unwrap()
    );
}

#[test]
fn or_under_the_default_or_operator_leaves_both_sides_optional() {
    let clause = parse_query("cat OR dog", Some("body")).unwrap();
    let b = boolean(&clause);
    assert_eq!(
        b.should,
        vec![
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ]
    );
    assert!(b.must.is_empty());
    // Adjacent clauses with no conjunction mean the same thing under OR.
    assert_eq!(clause, parse_query("cat dog", Some("body")).unwrap());
}

#[test]
fn not_is_a_modifier_identical_to_minus() {
    let with_not = parse_query("cat AND NOT dog", Some("body")).unwrap();
    let with_minus = parse_query("cat AND -dog", Some("body")).unwrap();
    assert_eq!(with_not, with_minus);
    let b = boolean(&with_not);
    assert_eq!(b.must, vec![Clause::Term(TermQuery::new("body", "cat"))]);
    assert_eq!(
        b.must_not,
        vec![Clause::Term(TermQuery::new("body", "dog"))]
    );
}

#[test]
fn and_does_not_promote_a_prohibited_preceding_clause() {
    // "unless it's already prohibited": `-a AND b` keeps `a` as MUST_NOT.
    let clause = parse_query("-cat AND dog", Some("body")).unwrap();
    let b = boolean(&clause);
    assert_eq!(
        b.must_not,
        vec![Clause::Term(TermQuery::new("body", "cat"))]
    );
    assert_eq!(b.must, vec![Clause::Term(TermQuery::new("body", "dog"))]);
    assert!(b.should.is_empty());
}

#[test]
fn default_and_operator_makes_adjacent_clauses_required() {
    let clause =
        parse_query_with_operator("cat dog", Some("body"), None, DefaultOperator::And).unwrap();
    let b = boolean(&clause);
    assert_eq!(
        b.must,
        vec![
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ]
    );
    assert!(b.should.is_empty());
}

#[test]
fn default_and_operator_or_demotes_the_preceding_clause() {
    // The second retroactive rewrite: under AND_OPERATOR, `a OR b` must
    // come out `a b`, not `+a b`.
    let clause =
        parse_query_with_operator("cat OR dog", Some("body"), None, DefaultOperator::And).unwrap();
    let b = boolean(&clause);
    assert_eq!(
        b.should,
        vec![
            Clause::Term(TermQuery::new("body", "cat")),
            Clause::Term(TermQuery::new("body", "dog")),
        ]
    );
    assert!(b.must.is_empty());
}

#[test]
fn default_and_operator_keeps_an_explicit_plus_required() {
    let clause =
        parse_query_with_operator("+cat OR dog", Some("body"), None, DefaultOperator::And).unwrap();
    let b = boolean(&clause);
    // `+cat` is demoted by the following OR exactly as a bare `cat` would
    // be -- Java's rewrite only skips *prohibited* clauses.
    assert_eq!(b.should.len(), 2);
}

#[test]
fn and_or_not_are_only_operators_as_whole_uppercase_tokens() {
    // "ANDROID"/"NOTHING" are longer tokens, "and" is lowercase: all
    // ordinary terms, exactly as JavaCC's tokenizer decides.
    assert_eq!(
        parse_query("ANDROID", Some("body")).unwrap(),
        Clause::Term(TermQuery::new("body", "ANDROID"))
    );
    assert_eq!(
        parse_query("NOTHING", Some("body")).unwrap(),
        Clause::Term(TermQuery::new("body", "NOTHING"))
    );
    let lowercase = parse_query("cat and dog", Some("body")).unwrap();
    let b = boolean(&lowercase);
    assert_eq!(b.should.len(), 3, "lowercase \"and\" is a term");
    assert!(b.must.is_empty());
}

#[test]
fn operators_apply_inside_a_group_independently() {
    let clause = parse_query("(cat AND dog) OR bird", Some("body")).unwrap();
    let b = boolean(&clause);
    assert_eq!(b.should.len(), 2);
    let inner = boolean(&b.should[0]);
    assert_eq!(inner.must.len(), 2);
}

#[test]
fn a_trailing_conjunction_is_an_error_not_a_silent_drop() {
    assert_eq!(
        parse_query("cat AND", Some("body")).unwrap_err(),
        ParseError::UnexpectedEnd("a clause after a conjunction")
    );
}

#[test]
fn a_leading_conjunction_is_parsed_as_a_term_like_javas_grammar() {
    // Java's `Query()` production only allows a conjunction from the
    // second clause on, so a leading `AND` is not a conjunction at all;
    // here it falls through to `Modifiers()`/`Clause()` and lexes as the
    // bareword it is, giving two SHOULD clauses.
    let clause = parse_query("AND cat", Some("body")).unwrap();
    let b = boolean(&clause);
    assert_eq!(
        b.should,
        vec![
            Clause::Term(TermQuery::new("body", "AND")),
            Clause::Term(TermQuery::new("body", "cat")),
        ]
    );
}

// --- Phrase slop ---

#[test]
fn phrase_slop_sets_the_phrase_querys_slop() {
    let clause = parse_query(r#"body:"quick fox"~3"#, None).unwrap();
    assert_eq!(
        clause,
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]).with_slop(3))
    );
}

#[test]
fn bare_tilde_after_a_phrase_leaves_the_default_slop() {
    let clause = parse_query(r#"body:"quick fox"~"#, None).unwrap();
    assert_eq!(
        clause,
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
    );
}

#[test]
fn fractional_phrase_slop_truncates_like_javas_int_cast() {
    let clause = parse_query(r#"body:"quick fox"~1.9"#, None).unwrap();
    assert_eq!(
        clause,
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]).with_slop(1))
    );
}

#[test]
fn phrase_slop_and_boost_parse_in_either_order() {
    let slop_first = parse_query(r#"body:"quick fox"~2^3"#, None).unwrap();
    let boost_first = parse_query(r#"body:"quick fox"^3~2"#, None).unwrap();
    assert_eq!(slop_first, boost_first);
    assert_eq!(
        slop_first,
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]).with_slop(2)),
            3.0,
        )))
    );
}

#[test]
fn fuzzy_and_boost_parse_in_either_order() {
    let fuzzy_first = parse_query("body:cat~1^2", None).unwrap();
    let boost_first = parse_query("body:cat^2~1", None).unwrap();
    assert_eq!(fuzzy_first, boost_first);
    assert_eq!(
        fuzzy_first,
        Clause::Boost(Box::new(BoostQuery::new(
            Clause::Fuzzy(FuzzyQuery::new("body", "cat").with_max_edits(1)),
            2.0,
        )))
    );
}

#[test]
fn tilde_after_a_wildcard_or_prefix_is_ignored_like_javas_precedence() {
    // `handleBareTokenQuery` checks wildcard/prefix before fuzzy.
    assert_eq!(
        parse_query("body:ca*~2", None).unwrap(),
        Clause::Prefix(PrefixQuery::new("body", "ca"))
    );
    assert_eq!(
        parse_query("body:c*t~2", None).unwrap(),
        Clause::Wildcard(WildcardQuery::new("body", "c*t"))
    );
}

#[test]
fn tilde_after_a_group_is_unsupported_syntax_not_silently_dropped() {
    assert!(matches!(
        parse_query("(cat dog)~2", Some("body")).unwrap_err(),
        ParseError::UnsupportedSyntax(_, _)
    ));
}

#[test]
fn single_bare_term_uses_default_field() {
    let clause = parse_query("cat", Some("body")).unwrap();
    assert_eq!(clause, Clause::Term(TermQuery::new("body", "cat")));
}

#[test]
fn bare_term_with_no_default_field_is_an_error() {
    let err = parse_query("cat", None).unwrap_err();
    assert_eq!(err, ParseError::MissingField(0));
}

#[test]
fn field_prefixed_term() {
    let clause = parse_query("title:cat", Some("body")).unwrap();
    assert_eq!(clause, Clause::Term(TermQuery::new("title", "cat")));
}

#[test]
fn quoted_phrase() {
    let clause = parse_query(r#"body:"quick fox""#, None).unwrap();
    assert_eq!(
        clause,
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
    );
}

#[test]
fn quoted_phrase_uses_default_field() {
    let clause = parse_query(r#""quick fox""#, Some("body")).unwrap();
    assert_eq!(
        clause,
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
    );
}

#[test]
fn plus_minus_combination() {
    let clause = parse_query("+cat -dog bird", Some("body")).unwrap();
    assert_eq!(
        clause,
        Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat")])
                .with_should([TermQuery::new("body", "bird")])
                .with_must_not([TermQuery::new("body", "dog")])
        ))
    );
}

#[test]
fn wildcard_term() {
    let clause = parse_query("body:c?t", None).unwrap();
    assert_eq!(clause, Clause::Wildcard(WildcardQuery::new("body", "c?t")));
}

#[test]
fn wildcard_term_with_interior_star() {
    let clause = parse_query("body:c*t", None).unwrap();
    assert_eq!(clause, Clause::Wildcard(WildcardQuery::new("body", "c*t")));
}

#[test]
fn trailing_star_is_a_prefix_query() {
    let clause = parse_query("body:ca*", None).unwrap();
    assert_eq!(clause, Clause::Prefix(PrefixQuery::new("body", "ca")));
}

#[test]
fn fuzzy_default_edit_distance() {
    let clause = parse_query("body:cat~", None).unwrap();
    assert_eq!(clause, Clause::Fuzzy(FuzzyQuery::new("body", "cat")));
}

#[test]
fn fuzzy_explicit_edit_distance() {
    let clause = parse_query("body:cat~1", None).unwrap();
    assert_eq!(
        clause,
        Clause::Fuzzy(FuzzyQuery::new("body", "cat").with_max_edits(1))
    );
}

#[test]
fn fuzzy_edit_distance_over_two_is_an_error() {
    let err = parse_query("body:cat~3", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidFuzziness(_, _)));
}

#[test]
fn fuzzy_fractional_similarity_is_an_error() {
    let err = parse_query("body:cat~0.8", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidFuzziness(_, _)));
}

#[test]
fn regexp_term() {
    let clause = parse_query("body:/ca.*/", None).unwrap();
    assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", "ca.*")));
}

#[test]
fn regexp_with_escaped_slash() {
    let clause = parse_query(r"body:/a\/b/", None).unwrap();
    assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", "a/b")));
}

#[test]
fn parenthesized_group() {
    let clause = parse_query("(+cat -dog)", Some("body")).unwrap();
    assert_eq!(
        clause,
        Clause::Boolean(Box::new(
            BooleanQuery::new()
                .with_must([TermQuery::new("body", "cat")])
                .with_must_not([TermQuery::new("body", "dog")])
        ))
    );
}

#[test]
fn nested_parenthesized_groups() {
    let clause = parse_query("+(cat (dog bird))", Some("body")).unwrap();
    let Clause::Boolean(top) = &clause else {
        panic!("expected Boolean");
    };
    assert_eq!(top.must.len(), 1);
    let Clause::Boolean(inner) = &top.must[0] else {
        panic!("expected nested Boolean");
    };
    assert_eq!(inner.should.len(), 2);
    assert_eq!(inner.should[0], Clause::Term(TermQuery::new("body", "cat")));
    let Clause::Boolean(innermost) = &inner.should[1] else {
        panic!("expected nested Boolean for '(dog bird)'");
    };
    assert_eq!(innermost.should.len(), 2);
}

#[test]
fn boost_suffix_on_term() {
    let clause = parse_query("body:cat^2.5", None).unwrap();
    assert_eq!(
        clause,
        Clause::Boost(Box::new(BoostQuery::new(
            TermQuery::new("body", "cat"),
            2.5
        )))
    );
}

#[test]
fn boost_suffix_on_group() {
    let clause = parse_query("(body:cat body:dog)^3", None).unwrap();
    let Clause::Boost(boost) = &clause else {
        panic!("expected Boost");
    };
    assert_eq!(boost.boost, 3.0);
    assert!(matches!(*boost.inner, Clause::Boolean(_)));
}

#[test]
fn escaped_special_char_is_literal() {
    let clause = parse_query(r"body:ca\*t", None).unwrap();
    assert_eq!(clause, Clause::Term(TermQuery::new("body", "ca*t")));
}

/// A term mixing an escaped `*` (must stay literal) with two genuine
/// wildcard `*`s (so the term is a `Wildcard`, not a `Prefix` -- a
/// single genuine trailing star alone would take the `Prefix` branch
/// instead, which is correct there since `PrefixQuery` never
/// re-interprets escapes) must resolve with the escape still intact:
/// `WildcardPattern::new` (the resolve-time consumer) must still see the
/// escape and treat only the genuine `*`s as real operators -- not a
/// pattern where the escaped `*` and a genuine `*` look identical once
/// resolved (which would silently turn the intended literal `*` into an
/// extra live wildcard, over-matching).
#[test]
fn escaped_wildcard_char_mixed_with_genuine_ones_stays_literal_at_resolve_time() {
    let clause = parse_query(r"body:a\*b*c*", None).unwrap();
    let Clause::Wildcard(w) = &clause else {
        panic!("expected Wildcard, got {clause:?}");
    };
    // The parser must hand WildcardPattern::new a pattern that still
    // carries the escape (`\*`), not an already-unescaped "a*b*c*" where
    // all three stars are indistinguishable.
    let pattern = lucene_codecs::wildcard::WildcardPattern::new(&w.pattern);
    assert!(
        pattern.matches(b"a*bXYZcXYZ"),
        "the literal `a*b` prefix, anything, then literal `c`, then anything, must match"
    );
    assert!(
            !pattern.matches(b"aXYZbXYZcXYZ"),
            "the escaped `*` must NOT be treated as a live wildcard operator: {pattern:?} vs aXYZbXYZcXYZ"
        );
}

/// The same property with the escaped star in the middle instead of
/// right after the leading literal, closing the same gap from another
/// position.
#[test]
fn escaped_wildcard_char_in_the_middle_stays_literal_at_resolve_time() {
    let clause = parse_query(r"body:a*b\*c*", None).unwrap();
    let Clause::Wildcard(w) = &clause else {
        panic!("expected Wildcard, got {clause:?}");
    };
    let pattern = lucene_codecs::wildcard::WildcardPattern::new(&w.pattern);
    assert!(
        pattern.matches(b"aXYZb*cXYZ"),
        "literal `a`, anything, literal `b*c`, anything, must match"
    );
    assert!(
        !pattern.matches(b"aXYZbXYZcXYZ"),
        "the escaped `*` must NOT be treated as a live wildcard operator"
    );
}

#[test]
fn unclosed_quote_is_a_clean_error() {
    let err = parse_query(r#"body:"quick fox"#, None).unwrap_err();
    assert!(matches!(err, ParseError::UnclosedQuote(_)));
}

#[test]
fn unclosed_paren_is_a_clean_error() {
    let err = parse_query("(body:cat", None).unwrap_err();
    assert!(matches!(err, ParseError::UnclosedParen(_)));
}

#[test]
fn unclosed_regexp_is_a_clean_error() {
    let err = parse_query("body:/ca.*", None).unwrap_err();
    assert!(matches!(err, ParseError::UnclosedRegexp(_)));
}

#[test]
fn unmatched_close_paren_is_a_clean_error() {
    let err = parse_query("body:cat)", None).unwrap_err();
    assert!(matches!(err, ParseError::UnmatchedCloseParen(_)));
}

#[test]
fn inclusive_inclusive_range_query() {
    let clause = parse_query("body:[0 TO 100]", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100))
    );
}

#[test]
fn exclusive_exclusive_range_query() {
    // `{0 TO 100}` excludes both 0 and 100, i.e. equivalent to `[1 TO 99]`.
    let clause = parse_query("body:{0 TO 100}", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 1, 99))
    );
}

#[test]
fn inclusive_exclusive_range_query() {
    // `[0 TO 100}` includes 0, excludes 100.
    let clause = parse_query("body:[0 TO 100}", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 99))
    );
}

#[test]
fn exclusive_inclusive_range_query() {
    // `{0 TO 100]` excludes 0, includes 100.
    let clause = parse_query("body:{0 TO 100]", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 1, 100))
    );
}

#[test]
fn bare_exclusive_range_query_is_supported() {
    // The bare-atom `{` arm (no `field:` prefix) is a separate code path
    // from `parse_term`'s -- exercise it directly, not just the
    // field-prefixed siblings above.
    let clause = parse_query("{0 TO 100}", Some("body")).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 1, 99))
    );
}

#[test]
fn exclusive_range_with_open_bound_is_not_stepped() {
    // `*` denotes "no bound", so the exclusive side doesn't get stepped
    // even though it's written with `{`/`}`.
    let clause = parse_query("body:{* TO 100}", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", i64::MIN, 99))
    );
    let clause = parse_query("body:{0 TO *}", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 1, i64::MAX))
    );
}

#[test]
fn exclusive_range_bound_at_extreme_does_not_overflow() {
    // An exclusive literal bound already at i64::MAX/i64::MIN saturates
    // instead of overflowing when stepped.
    let clause = parse_query(&format!("body:{{{} TO {}}}", i64::MIN, i64::MAX), None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new(
            "body",
            i64::MIN + 1,
            i64::MAX - 1
        ))
    );
}

#[test]
fn mismatched_range_delimiter_is_a_clean_error() {
    let err = parse_query("body:[0 TO 100)", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn non_numeric_range_bound_is_a_clean_error() {
    let err = parse_query("body:[a TO b]", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn inclusive_numeric_range_query() {
    let clause = parse_query("body:[0 TO 100]", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100))
    );
}

#[test]
fn inclusive_numeric_range_query_with_negative_bounds() {
    let clause = parse_query("body:[-100 TO -1]", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", -100, -1))
    );
}

#[test]
fn range_query_with_star_on_low_end() {
    let clause = parse_query("body:[* TO 100]", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", i64::MIN, 100))
    );
}

#[test]
fn range_query_with_star_on_high_end() {
    let clause = parse_query("body:[0 TO *]", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, i64::MAX))
    );
}

#[test]
fn range_query_with_star_on_both_ends() {
    let clause = parse_query("body:[* TO *]", None).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new(
            "body",
            i64::MIN,
            i64::MAX
        ))
    );
}

#[test]
fn range_query_uses_default_field_when_bare() {
    let clause = parse_query("[0 TO 100]", Some("body")).unwrap();
    assert_eq!(
        clause,
        Clause::PointsRange(crate::query::PointsRangeQuery::new("body", 0, 100))
    );
}

#[test]
fn range_query_missing_to_keyword_is_a_clean_error() {
    let err = parse_query("body:[0 100]", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn range_query_lowercase_to_is_not_recognized() {
    let err = parse_query("body:[0 to 100]", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn range_query_missing_closing_bracket_is_a_clean_error() {
    let err = parse_query("body:[0 TO 100", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn range_query_missing_min_bound_is_a_clean_error() {
    let err = parse_query("body:[ TO 100]", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn range_query_missing_max_bound_is_a_clean_error() {
    let err = parse_query("body:[0 TO ]", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn range_query_non_numeric_max_bound_is_a_clean_error() {
    let err = parse_query("body:[0 TO a]", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn range_query_bound_overflowing_i64_is_a_clean_error() {
    let err = parse_query("body:[99999999999999999999 TO 100]", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn range_query_to_keyword_without_trailing_word_boundary_is_a_clean_error() {
    // "TOxyz" must not be accepted as the "TO" keyword just because it
    // starts with the right two characters.
    let err = parse_query("body:[0 TOxyz 100]", None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidRangeBound(_, _)));
}

#[test]
fn range_query_boosted() {
    let clause = parse_query("body:[0 TO 100]^2", None).unwrap();
    assert_eq!(
        clause,
        Clause::Boost(Box::new(BoostQuery::new(
            crate::query::PointsRangeQuery::new("body", 0, 100),
            2.0
        )))
    );
}

#[test]
fn empty_input_is_an_error() {
    let err = parse_query("", Some("body")).unwrap_err();
    assert_eq!(err, ParseError::EmptyQuery);
}

#[test]
fn whitespace_only_input_is_an_error() {
    let err = parse_query("   ", Some("body")).unwrap_err();
    assert_eq!(err, ParseError::EmptyQuery);
}

#[test]
fn invalid_boost_is_a_clean_error() {
    let err = parse_query("body:cat^abc", None).unwrap_err();
    // '^' followed by non-digit/non-'.' parses as an empty number,
    // which fails to parse as f32.
    assert!(matches!(err, ParseError::InvalidBoost(_, _)));
}

#[test]
fn empty_group_is_an_error() {
    let err = parse_query("()", Some("body")).unwrap_err();
    assert!(matches!(err, ParseError::UnexpectedChar(_, '(')));
}

#[test]
fn bare_colon_with_no_field_name_falls_back_to_wordterm() {
    // ':' isn't a valid leading field char, so `try_parse_field` finds
    // no identifier before it and `parse_wordterm` is reached instead --
    // but ':' is itself a term-stop char, so this yields an empty
    // bareword, which is a clean error rather than a panic.
    let err = parse_query(":cat", Some("body")).unwrap_err();
    assert!(matches!(err, ParseError::UnexpectedChar(_, ':')));
}

#[test]
fn trailing_backslash_at_end_of_input_is_an_error() {
    let err = parse_query(r"body:cat\", None).unwrap_err();
    assert!(matches!(err, ParseError::UnexpectedEnd(_)));
}

#[test]
fn unclosed_paren_with_no_atom_inside_is_a_clean_error() {
    // '(' followed only by whitespace/EOF: `parse_clause_list` sees
    // `None` immediately and reports `EmptyQuery` (bubbling up through
    // `parse_group`'s `?`) rather than `UnclosedParen` -- an honest, if
    // slightly imprecise, label for "nothing valid was ever found," and
    // still a clean `Err`, not a panic.
    let err = parse_query("(   ", Some("body")).unwrap_err();
    assert_eq!(err, ParseError::EmptyQuery);
}

#[test]
fn boost_value_too_large_for_f32_is_an_error() {
    // All-digit text that overflows `f32::MAX` parses successfully to
    // `f32::INFINITY` rather than failing `str::parse`, so this exercises
    // the separate `is_finite()` check.
    let query = format!("body:cat^{}", "9".repeat(50));
    let err = parse_query(&query, None).unwrap_err();
    assert!(matches!(err, ParseError::InvalidBoost(_, _)));
}

#[test]
fn trailing_modifier_with_nothing_after_is_an_error() {
    let err = parse_query("body:cat +", Some("body")).unwrap_err();
    assert!(matches!(err, ParseError::UnexpectedEnd(_)));
}

#[test]
fn bare_range_query_syntax_without_field_prefix_and_no_default_field_is_missing_field() {
    let err = parse_query("[0 TO 100]", None).unwrap_err();
    assert!(matches!(err, ParseError::MissingField(_)));
}

#[test]
fn field_colon_with_nothing_after_is_an_error() {
    let err = parse_query("body:", None).unwrap_err();
    assert!(matches!(err, ParseError::UnexpectedEnd(_)));
}

#[test]
fn unclosed_quote_after_trailing_escape_is_a_clean_error() {
    let err = parse_query("body:\"foo\\", None).unwrap_err();
    assert!(matches!(err, ParseError::UnclosedQuote(_)));
}

#[test]
fn unclosed_regexp_after_trailing_escape_is_a_clean_error() {
    let err = parse_query(r"body:/a\", None).unwrap_err();
    assert!(matches!(err, ParseError::UnclosedRegexp(_)));
}

#[test]
fn regexp_escaped_non_slash_char_keeps_the_backslash_literally() {
    // Only `\/` is special-cased to a literal `/`; any other escaped
    // byte is passed through as `\` + that byte unchanged, left for
    // `RegexpPattern::new`'s own escaping rules to interpret later.
    let clause = parse_query(r"body:/a\d/", None).unwrap();
    assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", r"a\d")));
}

// --- Analyzer wiring ---

use lucene_analysis::Analyzer;
use std::collections::HashSet;

#[test]
fn none_analyzer_behavior_is_unchanged() {
    // Every existing test above calls `parse_query`, which now delegates
    // to `parse_query_with_analyzer(.., None)` -- this test additionally
    // pins that calling the two spellings directly produces identical
    // results for a representative case.
    let a = parse_query("Quick", Some("body")).unwrap();
    let b = parse_query_with_analyzer("Quick", Some("body"), None).unwrap();
    assert_eq!(a, b);
    assert_eq!(a, Clause::Term(TermQuery::new("body", "Quick")));
}

#[test]
fn bareword_through_lowercase_only_analyzer_is_lowercased() {
    let analyzer = Analyzer::standard(None);
    let clause = parse_query_with_analyzer("Quick", Some("body"), Some(&analyzer)).unwrap();
    assert_eq!(clause, Clause::Term(TermQuery::new("body", "quick")));
}

#[test]
fn bareword_that_is_a_stopword_yields_no_match_not_panic() {
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let analyzer = Analyzer::standard(Some(&stopwords));
    let clause = parse_query_with_analyzer("the", Some("body"), Some(&analyzer)).unwrap();
    assert_eq!(
        clause,
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![],
            filter: Vec::new(),
            should: vec![],
            must_not: vec![],
            minimum_should_match: 0,
        }))
    );
}

#[test]
fn bareword_analyzer_splits_into_multiple_tokens_becomes_phrase() {
    // The analysis-crate tokenizer splits on non-alphanumeric boundaries,
    // so a hyphenated bareword like "state-of-the-art" naturally becomes
    // multiple tokens -- exercising the "analyzer produced >1 token from
    // one bareword" path without needing a custom analyzer.
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let analyzer = Analyzer::standard(Some(&stopwords));
    let clause =
        parse_query_with_analyzer("state-of-the-art", Some("body"), Some(&analyzer)).unwrap();
    assert_eq!(
        clause,
        Clause::Phrase(PhraseQuery::new("body", ["state", "of", "art"]))
    );
}

#[test]
fn wildcard_pattern_text_is_not_analyzed() {
    // Uppercase letters in a wildcard pattern must survive untouched --
    // the analyzer must never see wildcard/prefix/fuzzy/regexp pattern
    // text.
    let analyzer = Analyzer::standard(None);
    let clause = parse_query_with_analyzer("body:C?T", None, Some(&analyzer)).unwrap();
    assert_eq!(clause, Clause::Wildcard(WildcardQuery::new("body", "C?T")));
}

#[test]
fn prefix_pattern_text_is_not_analyzed() {
    let analyzer = Analyzer::standard(None);
    let clause = parse_query_with_analyzer("body:CA*", None, Some(&analyzer)).unwrap();
    assert_eq!(clause, Clause::Prefix(PrefixQuery::new("body", "CA")));
}

#[test]
fn fuzzy_pattern_text_is_not_analyzed() {
    let analyzer = Analyzer::standard(None);
    let clause = parse_query_with_analyzer("body:CAT~", None, Some(&analyzer)).unwrap();
    assert_eq!(clause, Clause::Fuzzy(FuzzyQuery::new("body", "CAT")));
}

#[test]
fn regexp_pattern_text_is_not_analyzed() {
    // A would-be-stopword-shaped substring ("the") inside the pattern
    // must survive verbatim, and case must be untouched.
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let analyzer = Analyzer::standard(Some(&stopwords));
    let clause = parse_query_with_analyzer("body:/THE.*/", None, Some(&analyzer)).unwrap();
    assert_eq!(clause, Clause::Regexp(RegexpQuery::new("body", "THE.*")));
}

#[test]
fn quoted_phrase_words_are_analyzed_per_word() {
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let analyzer = Analyzer::standard(Some(&stopwords));
    let clause =
        parse_query_with_analyzer(r#"body:"The Quick FOX""#, None, Some(&analyzer)).unwrap();
    // "The" is a stopword and drops out entirely; the rest lowercase.
    assert_eq!(
        clause,
        Clause::Phrase(PhraseQuery::new("body", ["quick", "fox"]))
    );
}

#[test]
fn quoted_phrase_entirely_stopwords_is_no_match() {
    let stopwords: HashSet<String> = ["the".to_string()].into_iter().collect();
    let analyzer = Analyzer::standard(Some(&stopwords));
    let clause = parse_query_with_analyzer(r#"body:"the the""#, None, Some(&analyzer)).unwrap();
    assert_eq!(
        clause,
        Clause::Boolean(Box::new(BooleanQuery {
            must: vec![],
            filter: Vec::new(),
            should: vec![],
            must_not: vec![],
            minimum_should_match: 0,
        }))
    );
}

/// Execution-level proof, not just a parsed-`Clause` shape check: an
/// exclusive bound produced by [`parse_query`] actually excludes the
/// boundary value when the resulting [`PointsRangeQuery`]'s `min`/`max`
/// are fed through the real BKD points search path
/// ([`crate::points_query::search_points_range`]). `Clause::PointsRange`
/// itself has no resolver yet (see that variant's doc comment), so this
/// drives `search_points_range` directly with the parser's output --
/// the same thing a future resolver would do -- rather than going
/// through `resolve_clause_docs`.
#[test]
fn exclusive_bound_excludes_boundary_value_end_to_end() {
    use crate::collector::VecCollector;
    use crate::points_query::search_points_range;
    use lucene_codecs::points::{self, WritePointsField};
    use lucene_store::codec_util::ID_LENGTH;

    fn long_bytes(v: i64) -> [u8; 8] {
        ((v as u64) ^ 0x8000_0000_0000_0000).to_be_bytes()
    }

    // Doc 0 -> 10, doc 1 -> 20, doc 2 -> 30, doc 3 -> 40, doc 4 -> 50.
    let segment_id = [7u8; ID_LENGTH];
    let points: Vec<(i32, Vec<u8>)> = vec![
        (0, long_bytes(10).to_vec()),
        (1, long_bytes(20).to_vec()),
        (2, long_bytes(30).to_vec()),
        (3, long_bytes(40).to_vec()),
        (4, long_bytes(50).to_vec()),
    ];
    let field = WritePointsField {
        field_number: 1,
        num_dims: 1,
        num_index_dims: 1,
        bytes_per_dim: 8,
        points,
    };
    let (kdm, kdi, kdd) = points::write(&[field], 512, &segment_id, "").unwrap();
    let reader = points::open(&kdm, &kdi, &kdd, &segment_id, "").unwrap();

    // `{10 TO 40}` must exclude both boundary docs (values 10 and 40)
    // and match only the strictly-between ones (20, 30).
    let clause = parse_query("body:{10 TO 40}", None).unwrap();
    let Clause::PointsRange(range) = clause else {
        panic!("expected PointsRange, got {clause:?}");
    };
    assert_eq!(range.min, 11);
    assert_eq!(range.max, 39);
    let min = long_bytes(range.min);
    let max = long_bytes(range.max);
    let mut collector = VecCollector::default();
    search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
    assert_eq!(
        collector.docs,
        vec![1, 2],
        "must match only values 20 and 30, excluding boundary values 10 and 40"
    );

    // Sanity check: the equivalent inclusive `[10 TO 40]` does include
    // both boundary docs, proving the exclusion above is really coming
    // from `{`/`}`, not from some unrelated bug.
    let clause = parse_query("body:[10 TO 40]", None).unwrap();
    let Clause::PointsRange(range) = clause else {
        panic!("expected PointsRange, got {clause:?}");
    };
    let min = long_bytes(range.min);
    let max = long_bytes(range.max);
    let mut collector = VecCollector::default();
    search_points_range(&reader, None, 1, &min, &max, &mut collector).unwrap();
    assert_eq!(collector.docs, vec![0, 1, 2, 3]);
}
