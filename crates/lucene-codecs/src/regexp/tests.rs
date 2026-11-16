#![cfg(test)]
use super::*;

/// **A stream of distinct patterns cannot grow the process-wide compiled
/// cache past its bound** -- the regexp counterpart of the wildcard
/// cache's test. Three times the bound, checked after each.
/// **Large automata cannot fill the compiled cache past its bytes**: each
/// pattern below determinizes to thousands of states.
#[test]
fn the_compiled_cache_stays_within_its_bytes() {
    let mut largest = 0;
    for i in 0..24 {
        let p = RegexpPattern::parse(&format!("big{i}(a|b)*a(a|b){{10}}")).unwrap();
        largest = largest.max(p.compiled().map_or(0, |d| d.ram_bytes()));
        let held: usize = compiled_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .iter()
            .map(entry_bytes)
            .sum();
        assert!(
            held <= COMPILED_CACHE_BYTES,
            "pattern {i}: {held} bytes cached"
        );
    }
    assert!(
        largest > COMPILED_CACHE_BYTES / 64,
        "{largest} bytes: not a large automaton"
    );
}

#[test]
fn the_process_wide_compiled_cache_stays_bounded_under_distinct_patterns() {
    for i in 0..3 * COMPILED_CACHE_ENTRIES {
        RegexpPattern::parse(&format!("bound{i}[a-c]+"))
            .unwrap()
            .compiled();
        let len = compiled_cache()
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .len();
        assert!(len <= COMPILED_CACHE_ENTRIES, "pattern {i}: {len} cached");
    }
}

fn m(pattern: &str, term: &str) -> bool {
    RegexpPattern::parse(pattern)
        .unwrap()
        .matches(term.as_bytes())
}

fn err(pattern: &str) -> RegexpError {
    RegexpPattern::parse(pattern).unwrap_err()
}

#[test]
fn literal_pattern_matches_only_itself() {
    assert!(m("cat", "cat"));
    assert!(!m("cat", "cats"));
    assert!(!m("cat", "ca"));
    assert!(!m("cat", "CAT"));
}

#[test]
fn whole_term_match_is_enforced_not_substring() {
    assert!(!m("ca", "cat"));
    assert!(m("ca", "ca"));
    assert!(!m("at", "cat"));
}

#[test]
fn dot_matches_any_single_codepoint() {
    assert!(m("c.t", "cat"));
    assert!(m("c.t", "cot"));
    assert!(!m("c.t", "ct"));
    assert!(!m("c.t", "caat"));
    // `Automata.makeAnyChar` is one *codepoint*, not one UTF-8 byte:
    // `€` is 3 bytes and still matches a single `.`.
    assert!(m(".", "€"));
    assert!(m("a.c", "a€c"));
    assert!(m(".", "\u{10348}"));
}

#[test]
fn ill_formed_utf8_matches_nothing() {
    // `UTF32ToUTF8` only ever emits well-formed UTF-8, so no compiled
    // `RegExp` automaton accepts an ill-formed term.
    let p = RegexpPattern::parse(".*").unwrap();
    assert!(p.matches(b"ok"));
    assert!(!p.matches(&[0xFF]));
    assert!(!p.matches(&[0x80]));
    // Truncated 3-byte sequence.
    assert!(!p.matches(&[0xE2, 0x82]));
    // Overlong encoding of '/'.
    assert!(!p.matches(&[0xC0, 0xAF]));
    // Surrogate half.
    assert!(!p.matches(&[0xED, 0xA0, 0x80]));
    assert!(!RegexpPattern::parse("[^a]").unwrap().matches(&[0xFF]));
}

#[test]
fn star_is_zero_or_more_of_preceding_atom() {
    assert!(m("ca*t", "ct"));
    assert!(m("ca*t", "cat"));
    assert!(m("ca*t", "caaaat"));
    assert!(!m("ca*t", "cbt"));
}

#[test]
fn plus_is_one_or_more_of_preceding_atom() {
    assert!(!m("ca+t", "ct"));
    assert!(m("ca+t", "cat"));
    assert!(m("ca+t", "caaaat"));
}

#[test]
fn question_mark_is_zero_or_one_of_preceding_atom() {
    assert!(m("ca?t", "ct"));
    assert!(m("ca?t", "cat"));
    assert!(!m("ca?t", "caat"));
}

#[test]
fn repeated_quantifiers_stack_like_lucene_rather_than_erroring() {
    // `parseRepeatExp` loops `while (peek("?*+{"))`, so `a**` is a repeat
    // of a repeat, not a parse error.
    assert!(m("a**", ""));
    assert!(m("a**", "aaa"));
    assert!(m("a?+", "aa"));
    assert!(m("a{2}?", ""));
    assert!(m("a{2}?", "aa"));
    assert!(!m("a{2}?", "a"));
}

#[test]
fn character_class_matches_any_listed_codepoint() {
    assert!(m("[cb]at", "cat"));
    assert!(m("[cb]at", "bat"));
    assert!(!m("[cb]at", "hat"));
}

#[test]
fn character_class_range_matches_any_codepoint_in_range() {
    assert!(m("[a-c]at", "aat"));
    assert!(m("[a-c]at", "bat"));
    assert!(m("[a-c]at", "cat"));
    assert!(!m("[a-c]at", "dat"));
}

#[test]
fn negated_character_class_is_one_codepoint_not_listed() {
    assert!(m("[^ab]at", "cat"));
    assert!(!m("[^ab]at", "aat"));
    assert!(!m("[^ab]at", "bat"));
    // Lucene builds `[^..]` as `anyChar & complement(class)`, so it spans
    // the whole codepoint space, not just ASCII.
    assert!(m("[^a]", "€"));
    assert!(m("[^a]", "\u{10348}"));
}

#[test]
fn escaped_member_inside_a_class_cannot_open_a_range() {
    // Lucene's escape branch in `parseCharClasses` adds the codepoint and
    // loops without checking for `-`, so `[\--z]` is the members `-`, `-`
    // and `z`, not the range `-`..`z`.
    assert!(m(r"[\--z]", "-"));
    assert!(m(r"[\--z]", "z"));
    assert!(!m(r"[\--z]", "a"));
}

#[test]
fn dash_before_the_closing_bracket_is_a_range_end_in_lucene() {
    // `[a-]` is *not* "a literal trailing dash": `parseCharExp` consumes
    // the `]` as the range's upper bound, and `makeCharRange('a', ']')`
    // throws before the unclosed class is noticed.
    assert_eq!(err("[a-]"), RegexpError::InvalidRange { from: 97, to: 93 });
    // With an upper bound above `a`, the unclosed class is the error.
    assert_eq!(
        err("[a-~"),
        RegexpError::ExpectedChar {
            expected: ']',
            pos: 4
        }
    );
}

#[test]
fn escaped_bracket_inside_class_is_a_literal_class_member() {
    assert!(m(r"[\]]", "]"));
    assert!(!m(r"[\]]", "x"));
}

#[test]
fn trailing_backslash_inside_class_is_an_end_of_string_error() {
    assert_eq!(err(r"[a\"), RegexpError::UnexpectedEndOfString);
}

#[test]
fn predefined_classes_expand_like_javas() {
    assert!(m(r"\d+", "1234"));
    assert!(!m(r"\d+", "12a4"));
    assert!(m(r"\D", "a"));
    assert!(!m(r"\D", "1"));
    assert!(m(r"\w+", "a_Z9"));
    assert!(!m(r"\w+", "a-b"));
    assert!(m(r"\W", "-"));
    assert!(m(r"\s", " "));
    assert!(m(r"\s", "\t"));
    assert!(m(r"\s", "\n"));
    assert!(m(r"\s", "\r"));
    assert!(!m(r"\s", "a"));
    assert!(m(r"\S", "a"));
    assert!(!m(r"\S", " "));
    // ... and inside a class, where they union with the other members.
    assert!(m(r"[\dx]+", "1x2"));
    assert!(!m(r"[\dx]+", "1y2"));
    assert!(m(r"[^\d]", "a"));
    assert!(!m(r"[^\d]", "7"));
    // `\\` is the escape for a literal backslash in both positions.
    assert!(m(r"\\", "\\"));
    assert!(m(r"[\\]", "\\"));
}

#[test]
fn an_alphabetic_escape_that_is_not_a_predefined_class_is_an_error() {
    assert_eq!(err(r"\A"), RegexpError::InvalidCharacterClass('A'));
    assert_eq!(err(r"\q"), RegexpError::InvalidCharacterClass('q'));
    assert_eq!(err(r"[\A]"), RegexpError::InvalidCharacterClass('A'));
}

#[test]
fn alternation_matches_either_side() {
    assert!(m("cat|dog", "cat"));
    assert!(m("cat|dog", "dog"));
    assert!(!m("cat|dog", "bird"));
}

#[test]
fn intersection_requires_both_sides_to_accept_the_same_string() {
    // `&` is enabled by `RegExp.ALL`, which is what `RegexpQuery(Term)`
    // uses.
    assert!(m("[a-z]+&...", "abc"));
    assert!(!m("[a-z]+&...", "ab"));
    assert!(!m("[a-z]+&...", "ab1"));
    assert!(m("(cat|dog)&(dog|bird)", "dog"));
    assert!(!m("(cat|dog)&(dog|bird)", "cat"));
    // Intersection binds tighter than union.
    assert!(m("ab&ab|cd", "cd"));
    assert!(m("ab&ab|cd", "ab"));
    // ... and looser than concatenation, so both sides are whole spans.
    assert!(m("a.&.b", "ab"));
    assert!(!m("a.&.b", "ac"));
}

#[test]
fn grouping_scopes_quantifiers_and_alternation() {
    assert!(m("(cat)+", "catcat"));
    assert!(!m("(cat)+", "ca"));
    assert!(m("(cat|dog)s", "cats"));
    assert!(m("(cat|dog)s", "dogs"));
    assert!(!m("(cat|dog)s", "birds"));
}

#[test]
fn empty_group_is_the_empty_string() {
    assert!(m("()", ""));
    assert!(!m("()", "a"));
    assert!(m("a()b", "ab"));
}

#[test]
fn quoted_string_is_a_literal_with_no_escapes_inside() {
    assert!(m(r#""a*b""#, "a*b"));
    assert!(!m(r#""a*b""#, "aab"));
    assert!(m(r#""""#, ""));
    assert!(m(r#"x"[](){}"y"#, "x[](){}y"));
    assert_eq!(
        err(r#""abc"#),
        RegexpError::ExpectedChar {
            expected: '"',
            pos: 4
        }
    );
}

#[test]
fn hash_is_the_empty_language() {
    // `#` (the `EMPTY` flag, on in `RegExp.ALL`) accepts nothing at all,
    // not the empty string.
    assert!(!m("#", ""));
    assert!(!m("#", "a"));
    assert!(!m("a#b", "ab"));
    // Union with the empty language is the other side.
    assert!(m("#|cat", "cat"));
}

#[test]
fn at_sign_is_any_string() {
    assert!(m("@", ""));
    assert!(m("@", "anything at all"));
    assert!(m("@", "€uro"));
    assert!(m("cat@", "catalog"));
    assert!(!m("cat@", "dog"));
}

#[test]
fn tilde_is_an_ordinary_literal_because_complement_is_not_in_regexp_all() {
    // `DEPRECATED_COMPLEMENT` (0x10000) is not part of `RegExp.ALL`, so
    // `RegexpQuery(Term)` treats `~` as a plain character. This port used
    // to reject the whole pattern.
    assert!(m("a~b", "a~b"));
    assert!(!m("a~b", "ab"));
    assert!(m("~", "~"));
}

#[test]
fn numeric_interval_with_equal_width_bounds_is_fixed_width() {
    // `<05-40>`: both bounds two characters, so `digits == 2` and only
    // zero-padded two-digit strings match.
    assert!(m("<05-40>", "05"));
    assert!(m("<05-40>", "40"));
    assert!(m("<05-40>", "23"));
    assert!(!m("<05-40>", "5"));
    assert!(!m("<05-40>", "41"));
    assert!(!m("<05-40>", "004"));
}

#[test]
fn numeric_interval_with_unequal_width_bounds_accepts_any_width() {
    // `<5-40>`: bounds of different widths, so `digits == 0` and Lucene
    // adds a `0*` prefix loop -- any number of leading zeros, any width.
    assert!(m("<5-40>", "5"));
    assert!(m("<5-40>", "05"));
    assert!(m("<5-40>", "0005"));
    assert!(m("<5-40>", "40"));
    assert!(!m("<5-40>", "4"));
    assert!(!m("<5-40>", "41"));
    assert!(!m("<5-40>", ""));
}

#[test]
fn numeric_interval_swaps_out_of_order_bounds_and_composes() {
    assert!(m("<40-5>", "23"));
    assert!(m("x<1-3>y", "x2y"));
    assert!(!m("x<1-3>y", "x4y"));
    // A value far beyond `u32` is out of range, not an overflow panic.
    assert!(!m("<1-30>", "999999999999999999999"));
    assert!(m("<1-30>", "0000000009"));
}

#[test]
fn malformed_interval_is_a_parse_error() {
    assert_eq!(err("<-3>"), RegexpError::IntervalSyntax { pos: 3 });
    assert_eq!(err("<1->"), RegexpError::IntervalSyntax { pos: 3 });
    assert_eq!(err("<1-2-3>"), RegexpError::IntervalSyntax { pos: 6 });
    assert_eq!(err("<a-b>"), RegexpError::IntervalSyntax { pos: 4 });
    assert_eq!(
        err("<1-2"),
        RegexpError::ExpectedChar {
            expected: '>',
            pos: 4
        }
    );
}

#[test]
fn named_automaton_is_not_found_because_there_is_no_provider() {
    // `RegexpQuery.DEFAULT_PROVIDER` returns null for every name, so
    // `RegExp.toAutomaton` throws `"'name' not found"`.
    assert_eq!(
        err("<name>"),
        RegexpError::NamedAutomatonNotFound("name".to_string())
    );
}

#[test]
fn nested_groups_and_alternation_compose() {
    assert!(m("(a(b|c)d)+", "abdacd"));
    assert!(!m("(a(b|c)d)+", "abdaed"));
}

#[test]
fn escaped_metacharacter_is_a_literal() {
    assert!(m(r"a\*b", "a*b"));
    assert!(!m(r"a\*b", "aab"));
    assert!(m(r"a\.b", "a.b"));
    assert!(!m(r"a\.b", "axb"));
    assert!(m(r"a\[b", "a[b"));
    assert!(m(r"a\&b", "a&b"));
    assert!(m(r"a\@b", "a@b"));
    assert!(m(r"a\#b", "a#b"));
}

#[test]
fn trailing_unescaped_backslash_is_an_end_of_string_error() {
    // Lucene's `parseCharExp` calls `next()` past the end and throws.
    assert_eq!(err(r"ab\"), RegexpError::UnexpectedEndOfString);
}

#[test]
fn empty_pattern_matches_only_empty_term() {
    assert!(m("", ""));
    assert!(!m("", "a"));
}

#[test]
fn a_leading_quantifier_is_an_ordinary_literal_in_lucene() {
    // Lucene's `parseSimpleExp` falls through to `makeChar`, so `*cat`
    // matches the literal term `*cat` -- it is not a "dangling
    // quantifier" error.
    assert!(m("*cat", "*cat"));
    assert!(m("{2,3}", "{2,3}"));
    assert!(m("+x", "+x"));
    assert!(m("?", "?"));
}

#[test]
fn a_leading_union_or_intersection_operator_is_a_literal() {
    // `iterativeParseExp` gathers before testing its stop condition, so
    // the operator at position 0 has no left-hand side and falls through
    // to `makeChar`.
    assert!(m("|a", "|a"));
    assert!(m("&a", "&a"));
    assert!(!m("|a", "a"));
}

#[test]
fn a_trailing_operator_runs_off_the_end_of_the_pattern() {
    assert_eq!(err("a|"), RegexpError::UnexpectedEndOfString);
    assert_eq!(err("a&"), RegexpError::UnexpectedEndOfString);
}

#[test]
fn unmatched_open_paren_is_a_parse_error() {
    assert_eq!(
        err("(cat"),
        RegexpError::ExpectedChar {
            expected: ')',
            pos: 4
        }
    );
}

#[test]
fn unmatched_close_paren_is_a_parse_error() {
    assert_eq!(err("cat)"), RegexpError::EndOfStringExpected { pos: 3 });
}

#[test]
fn unmatched_open_bracket_is_a_parse_error() {
    assert_eq!(
        err("[cat"),
        RegexpError::ExpectedChar {
            expected: ']',
            pos: 4
        }
    );
}

#[test]
fn empty_class_is_an_unclosed_class_error_not_an_empty_class_error() {
    // Lucene's do-while consumes the `]` as a member, then fails to find
    // the real closing bracket.
    assert_eq!(
        err("[]"),
        RegexpError::ExpectedChar {
            expected: ']',
            pos: 2
        }
    );
    assert_eq!(
        err("[^]"),
        RegexpError::ExpectedChar {
            expected: ']',
            pos: 3
        }
    );
}

#[test]
fn every_error_variant_has_a_non_empty_display_message() {
    for e in [
        RegexpError::UnexpectedEndOfString,
        RegexpError::ExpectedChar {
            expected: ')',
            pos: 1,
        },
        RegexpError::IntegerExpected { pos: 1 },
        RegexpError::InvalidRepetitionRange { min: 3, max: 2 },
        RegexpError::InvalidCharacterClass('A'),
        RegexpError::IntervalSyntax { pos: 1 },
        RegexpError::IllegalIdentifier { pos: 1 },
        RegexpError::NamedAutomatonNotFound("x".to_string()),
        RegexpError::EndOfStringExpected { pos: 1 },
    ] {
        assert!(!e.to_string().is_empty());
    }
    assert_eq!(
        RegexpError::InvalidRepetitionRange { min: 3, max: 2 }.to_string(),
        "invalid repetition range(out of order): 3..2"
    );
}

// -- {n,m} bounded repetition -------------------------------------

#[test]
fn exact_count_repeat_matches_only_that_many() {
    assert!(!m("a{3}", "aa"));
    assert!(m("a{3}", "aaa"));
    assert!(!m("a{3}", "aaaa"));
}

#[test]
fn zero_zero_repeat_matches_zero_occurrences_only() {
    assert!(m("a{0,0}b", "b"));
    assert!(!m("a{0,0}b", "ab"));
}

#[test]
fn min_only_repeat_is_unbounded_above() {
    assert!(!m("a{2,}", "a"));
    assert!(m("a{2,}", "aa"));
    assert!(m("a{2,}", "aaa"));
    assert!(m("a{2,}", "aaaaaaaa"));
}

#[test]
fn min_max_repeat_bounds_both_ends() {
    assert!(!m("a{2,4}", "a"));
    assert!(m("a{2,4}", "aa"));
    assert!(m("a{2,4}", "aaa"));
    assert!(m("a{2,4}", "aaaa"));
    assert!(!m("a{2,4}", "aaaaa"));
}

#[test]
fn repeat_zero_min_allows_absence() {
    assert!(m("a{0,2}b", "b"));
    assert!(m("a{0,2}b", "ab"));
    assert!(m("a{0,2}b", "aab"));
    assert!(!m("a{0,2}b", "aaab"));
}

#[test]
fn repeat_composes_with_other_operators() {
    assert!(m("a{2,3}b*", "aa"));
    assert!(m("a{2,3}b*", "aaab"));
    assert!(m("a{2,3}b*", "aaabbb"));
    assert!(!m("a{2,3}b*", "a"));
    assert!(!m("a{2,3}b*", "aaaab"));
}

#[test]
fn repeat_on_group_applies_to_whole_group() {
    assert!(m("(ab){2,3}", "abab"));
    assert!(m("(ab){2,3}", "ababab"));
    assert!(!m("(ab){2,3}", "ab"));
    assert!(!m("(ab){2,3}", "abababab"));
}

/// A nested bounded-repeat pattern whose two `{1,15}` counts combine
/// multiplicatively against a matching-but-ultimately-failing input
/// (an all-`a` term with no trailing `b`) -- without the step budget in
/// `RegexpPattern::matches`, this would backtrack combinatorially and
/// hang; with it, this must return promptly (the test itself times out
/// the whole suite otherwise) and correctly report no match.
#[test]
fn nested_bounded_repeat_does_not_hang_on_a_failing_match() {
    assert!(!m("(a{1,15}){1,15}b", &"a".repeat(40)));
}

#[test]
fn zero_width_repeat_inner_terminates() {
    // `(a?)*` can consume nothing per iteration; without the
    // no-progress guard in `repeat_match` this loops forever.
    assert!(m("(a?)*", "aaa"));
    assert!(m("(a?)*", ""));
    assert!(!m("(a?)*b", "aaa"));
}

#[test]
fn malformed_repeat_missing_close_brace_is_a_parse_error() {
    assert_eq!(
        err("a{2,3"),
        RegexpError::ExpectedChar {
            expected: '}',
            pos: 5
        }
    );
}

#[test]
fn malformed_repeat_non_numeric_bound_is_a_parse_error() {
    assert_eq!(err("a{x}"), RegexpError::IntegerExpected { pos: 2 });
    assert_eq!(err("a{}"), RegexpError::IntegerExpected { pos: 2 });
}

#[test]
fn malformed_repeat_max_less_than_min_is_a_parse_error() {
    assert_eq!(
        err("a{3,2}"),
        RegexpError::InvalidRepetitionRange { min: 3, max: 2 }
    );
}

#[test]
fn repeat_bound_too_large_for_u32_is_a_parse_error() {
    assert_eq!(
        err("a{99999999999}"),
        RegexpError::IntegerExpected { pos: 2 }
    );
}

#[test]
fn bounds_past_i32_max_are_errors_as_in_java() {
    // `Integer.parseInt` rejects 2^31 .. 2^32-1, which a `u32` accepts.
    assert_eq!(
        err("a{3000000000}"),
        RegexpError::IntegerExpected { pos: 2 }
    );
    assert!(matches!(
        err("<0-3000000000>"),
        RegexpError::IntervalSyntax { .. }
    ));
}

#[test]
fn inverted_class_range_is_an_error_as_in_java() {
    let e = err("[z-a]");
    assert_eq!(e, RegexpError::InvalidRange { from: 122, to: 97 });
    assert_eq!(
        e.to_string(),
        "invalid range: from (122) cannot be > to (97)"
    );
    assert_eq!(err("[^z-a]"), e);
    assert!(RegexpPattern::parse("[a-a]").is_ok());
}

// -- literal_prefix ------------------------------------------------

fn prefix(pattern: &str) -> Vec<u8> {
    RegexpPattern::parse(pattern).unwrap().literal_prefix()
}

#[test]
fn literal_prefix_of_pure_literal_pattern_is_the_whole_pattern() {
    assert_eq!(prefix("cat"), b"cat".to_vec());
    assert_eq!(prefix("c"), b"c".to_vec());
}

#[test]
fn literal_prefix_stops_at_first_non_literal_atom() {
    assert_eq!(prefix("ca.*"), b"ca".to_vec());
    assert_eq!(prefix("ca*t"), b"c".to_vec());
    assert_eq!(prefix("ca?t"), b"c".to_vec());
}

#[test]
fn literal_prefix_uses_a_mandatory_repeats_minimum_count() {
    // `a{2,3}` guarantees two leading `a`s; the old implementation gave
    // up at any quantifier and returned just `c`.
    assert_eq!(prefix("ca{2,3}t"), b"caa".to_vec());
    assert_eq!(prefix("ca+t"), b"ca".to_vec());
    assert_eq!(prefix("c(ab){2,}t"), b"cabab".to_vec());
    assert_eq!(prefix("ca{0,3}t"), b"c".to_vec());
}

#[test]
fn literal_prefix_of_alternation_is_empty() {
    assert_eq!(prefix("cat|dog"), Vec::<u8>::new());
}

#[test]
fn literal_prefix_of_leading_wildcard_atom_is_empty() {
    assert_eq!(prefix(".*cat"), Vec::<u8>::new());
    assert_eq!(prefix("[ab]cat"), Vec::<u8>::new());
    assert_eq!(prefix("@cat"), Vec::<u8>::new());
    assert_eq!(prefix("#"), Vec::<u8>::new());
    assert_eq!(prefix("<1-9>a"), Vec::<u8>::new());
}

#[test]
fn literal_prefix_of_quoted_string_and_intersection() {
    assert_eq!(prefix(r#""cat".*"#), b"cat".to_vec());
    assert_eq!(prefix("cat.*&catalog"), b"catalog".to_vec());
}

#[test]
fn literal_prefix_of_empty_pattern_is_empty() {
    assert_eq!(prefix(""), Vec::<u8>::new());
}

#[test]
fn literal_prefix_is_a_true_prefix_of_every_match() {
    // Property check: whatever `literal_prefix` claims must actually be a
    // prefix of every term the pattern accepts, or `regexp_intersect`'s
    // binary-search narrowing would silently drop matches.
    let cases: &[(&str, &[&str])] = &[
        ("ca{2,3}t", &["caat", "caaat"]),
        ("ca+t", &["cat", "caaat"]),
        ("cat.*", &["cat", "catalog"]),
        (r#""cat"[0-9]"#, &["cat7"]),
        ("cat.*&catalog", &["catalog"]),
        ("c(ab){2,}t", &["cababt", "cabababt"]),
    ];
    for (pattern, terms) in cases {
        let p = RegexpPattern::parse(pattern).unwrap();
        let pfx = p.literal_prefix();
        for term in *terms {
            assert!(p.matches(term.as_bytes()), "{pattern} should match {term}");
            assert!(
                term.as_bytes().starts_with(&pfx),
                "{pattern}: prefix {pfx:?} is not a prefix of {term}"
            );
        }
    }
}

// -- dead-prefix skipping (the `ByteRunAutomaton` dead-state signal) ---

/// The soundness property the whole skip rests on: if
/// `could_match_prefix(p)` is false, then **no** string starting with `p`
/// matches. Checked exhaustively over a small alphabet, which for these
/// patterns covers every structurally distinct case.
#[test]
fn a_dead_prefix_has_no_matching_extension() {
    const ALPHABET: &[u8] = b"abz0";
    const MAX_LEN: usize = 4;
    let patterns = [
        "cat", "ca*t", "c.t", "a*z", "[ab]+z", "a{2,3}b", "(ab|az)+", "a.&.b", r"\d+z", "z@",
        "<1-30>b", "\"ab\"z", "a|bz", "#", "[^a]b",
    ];
    // Every string over ALPHABET up to MAX_LEN, shortest first.
    let mut words: Vec<Vec<u8>> = vec![Vec::new()];
    let mut frontier: Vec<Vec<u8>> = vec![Vec::new()];
    for _ in 0..MAX_LEN {
        let mut next = Vec::new();
        for w in &frontier {
            for &c in ALPHABET {
                let mut e = w.clone();
                e.push(c);
                next.push(e);
            }
        }
        words.extend(next.iter().cloned());
        frontier = next;
    }

    for pattern in patterns {
        let p = RegexpPattern::parse(pattern).unwrap();
        for word in &words {
            if p.could_match_prefix(word) {
                continue;
            }
            for candidate in &words {
                assert!(
                    !(candidate.starts_with(word) && p.matches(candidate)),
                    "{pattern}: {:?} declared dead but {:?} matches",
                    String::from_utf8_lossy(word),
                    String::from_utf8_lossy(candidate)
                );
            }
        }
    }
}

#[test]
fn dead_prefix_len_finds_the_shortest_dead_prefix() {
    let p = RegexpPattern::parse("cat").unwrap();
    // "cat" itself is alive (it matches); "cats" dies at "cats" -- the
    // pattern is exhausted, so no longer term can match.
    assert_eq!(p.dead_prefix_len(b"cat"), None);
    assert_eq!(p.dead_prefix_len(b"cats"), Some(4));
    // A mismatch at the second byte kills the 2-byte prefix.
    assert_eq!(p.dead_prefix_len(b"cot"), Some(2));
    assert_eq!(p.dead_prefix_len(b"dog"), Some(1));

    // A trailing `.*` keeps every longer term alive.
    let p = RegexpPattern::parse("cat.*").unwrap();
    assert_eq!(p.dead_prefix_len(b"catalog"), None);
    assert_eq!(p.dead_prefix_len(b"cot"), Some(2));

    // `t1[0-9]` is the shape that makes this worth doing: exactly one
    // digit may follow, so a third byte kills the whole subtree.
    let p = RegexpPattern::parse("t1[0-9]").unwrap();
    assert_eq!(p.dead_prefix_len(b"t1"), None);
    assert_eq!(p.dead_prefix_len(b"t10"), None);
    assert_eq!(p.dead_prefix_len(b"t100"), Some(4));
    assert_eq!(p.dead_prefix_len(b"t1x"), Some(3));
}

#[test]
fn dead_prefix_len_never_splits_a_codepoint() {
    // `é` is two bytes; a dead prefix that cut between them would skip
    // terms sharing only half a character.
    let p = RegexpPattern::parse("caf.").unwrap();
    let term = "caféx".as_bytes();
    let k = p
        .dead_prefix_len(term)
        .expect("term is longer than the pattern");
    assert!(
        std::str::from_utf8(&term[..k]).is_ok(),
        "dead prefix {k} splits a codepoint"
    );
}

#[test]
fn an_all_accepting_pattern_has_no_dead_prefix() {
    let p = RegexpPattern::parse("@").unwrap();
    assert_eq!(p.dead_prefix_len(b"anything at all"), None);
    let p = RegexpPattern::parse(".*").unwrap();
    assert_eq!(p.dead_prefix_len(b"anything at all"), None);
}

#[test]
fn the_empty_language_kills_every_prefix() {
    let p = RegexpPattern::parse("#").unwrap();
    assert!(!p.could_match_prefix(b""));
    assert_eq!(p.dead_prefix_len(b"a"), Some(0));
}

#[test]
fn new_accepts_raw_bytes_and_decodes_them_leniently() {
    assert!(RegexpPattern::new(b"cat").unwrap().matches(b"cat"));
    // Ill-formed pattern bytes decode to U+FFFD rather than failing.
    assert!(RegexpPattern::new(&[0xFF]).is_ok());
}
