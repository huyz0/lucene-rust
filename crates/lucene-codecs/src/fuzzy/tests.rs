#![cfg(test)]
use super::*;

#[test]
fn identical_strings_have_zero_distance() {
    assert_eq!(edit_distance(b"cat", b"cat", true), 0);
    assert_eq!(edit_distance(b"", b"", true), 0);
}

#[test]
fn single_substitution_is_distance_one() {
    assert_eq!(edit_distance(b"cat", b"cot", true), 1);
    assert_eq!(edit_distance(b"cat", b"cot", false), 1);
}

#[test]
fn single_insertion_is_distance_one() {
    assert_eq!(edit_distance(b"cat", b"cats", true), 1);
    assert_eq!(edit_distance(b"cat", b"scat", true), 1);
}

#[test]
fn single_deletion_is_distance_one() {
    assert_eq!(edit_distance(b"cats", b"cat", true), 1);
    assert_eq!(edit_distance(b"scat", b"cat", true), 1);
}

/// The single most important test in this module: real `FuzzyQuery`'s
/// default `transpositions = true` treats an adjacent swap as one edit
/// (restricted Damerau-Levenshtein), while `transpositions = false`
/// treats the same swap as two edits (plain Levenshtein).
#[test]
fn transposition_is_one_edit_with_transpositions_two_without() {
    assert_eq!(edit_distance(b"cat", b"cta", true), 1);
    assert_eq!(edit_distance(b"cat", b"cta", false), 2);
}

#[test]
fn multiple_transpositions_still_count_correctly() {
    // "abcd" -> "badc": swap(a,b) + swap(c,d), 2 transpositions.
    assert_eq!(edit_distance(b"abcd", b"badc", true), 2);
}

#[test]
fn distance_grows_with_more_edits() {
    assert_eq!(edit_distance(b"kitten", b"sitting", true), 3);
}

/// `FuzzyAutomatonBuilder` decodes the term to `int[]` codepoints before
/// building the automaton, so one multi-byte character substituted for
/// another is **one** edit; counting UTF-8 bytes would report 2 or 3 for the
/// same pair and silently narrow every non-ASCII fuzzy query.
#[test]
fn distance_counts_codepoints_not_utf8_bytes() {
    // é (2 bytes) -> è (2 bytes): one codepoint substitution.
    assert_eq!(edit_distance("café".as_bytes(), "cafè".as_bytes(), true), 1);
    // € (3 bytes) -> a: still one substitution.
    assert_eq!(edit_distance("€".as_bytes(), "a".as_bytes(), true), 1);
    // Deleting a 4-byte codepoint is one edit.
    assert_eq!(
        edit_distance("a\u{10348}b".as_bytes(), "ab".as_bytes(), true),
        1
    );
    // ... and a transposition of two multi-byte codepoints is one edit.
    assert_eq!(edit_distance("éè".as_bytes(), "èé".as_bytes(), true), 1);
}

#[test]
fn ill_formed_utf8_is_replaced_rather_than_rejected() {
    assert_eq!(edit_distance(&[0xFF], &[0xFF], true), 0);
    assert_eq!(edit_distance(&[0xFF], b"a", true), 1);
}

/// The straightforward full-matrix Damerau-Levenshtein, kept as a
/// **reference** for the rolling-row implementation: three
/// live rows and an index-arithmetic band is exactly the shape where an
/// off-by-one hides, and the hand-picked cases above cannot reach the
/// combinations that would expose one (a transposition that must read two
/// rows back while the band's lower edge has already moved past it, say).
// ARITH: the reference implementation, deliberately written the obvious
// way -- `i` runs over `1..=n` and `j` over `1..=m` where `n`/`m` are
// slice lengths, so every `- 1` is in range and `- 2` runs only under the
// `i > 1 && j > 1` guard; the `+ 1`/`+ cost` operate on distances bounded
// by `n + m`, which is at most twice a slice length and so far below
// `usize::MAX`. It is test-only, and its whole value is being written
// differently from the code it checks.
#[allow(clippy::arithmetic_side_effects)]
fn reference_distance(a: &[char], b: &[char], transpositions: bool) -> usize {
    let (n, m) = (a.len(), b.len());
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate() {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate() {
        *cell = j;
    }
    for i in 1..=n {
        for j in 1..=m {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut best = (dp[i - 1][j] + 1)
                .min(dp[i][j - 1] + 1)
                .min(dp[i - 1][j - 1] + cost);
            if transpositions && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(dp[i - 2][j - 2] + 1);
            }
            dp[i][j] = best;
        }
    }
    dp[n][m]
}

/// Every string over a 3-letter alphabet up to length 5, against every
/// other, at every budget from 0 to 5, both with and without
/// transpositions -- checked against [`reference_distance`]. A 3-letter
/// alphabet is what makes transpositions and repeats common enough to
/// exercise the third row; longer alphabets mostly produce substitutions.
#[test]
fn the_banded_rolling_row_dp_agrees_with_the_full_matrix_everywhere() {
    let alphabet = ['a', 'b', 'c'];
    let mut words: Vec<Vec<char>> = vec![Vec::new()];
    let mut frontier: Vec<Vec<char>> = vec![Vec::new()];
    for _ in 0..5 {
        let mut next = Vec::new();
        for w in &frontier {
            for &c in &alphabet {
                let mut w2 = w.clone();
                w2.push(c);
                next.push(w2);
            }
        }
        words.extend(next.iter().cloned());
        frontier = next;
    }
    // 3^0 + ... + 3^5 = 364 words; the full cross product is 132 496
    // pairs, which runs in well under a second at this size.
    for transpositions in [true, false] {
        for a in &words {
            for b in &words {
                let exact = reference_distance(a, b, transpositions);
                assert_eq!(
                    distance_chars(a, b, transpositions, usize::MAX),
                    Some(exact),
                    "unbounded {a:?} vs {b:?}, transpositions={transpositions}"
                );
                for max in 0..=5usize {
                    let got = distance_chars(a, b, transpositions, max);
                    if exact <= max {
                        assert_eq!(
                            got,
                            Some(exact),
                            "{a:?} vs {b:?} at max {max}, transpositions={transpositions}"
                        );
                    } else {
                        assert_eq!(
                            got, None,
                            "{a:?} vs {b:?} at max {max}, transpositions={transpositions}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn bounded_distance_short_circuits_but_agrees_where_it_answers() {
    for (a, b) in [
        (&b"kitten"[..], &b"sitting"[..]),
        (b"cat", b"cta"),
        (b"", b"abc"),
        (b"abcdef", b"abcdef"),
        (b"abcdef", b"badcfe"),
    ] {
        let exact = edit_distance(a, b, true);
        for max in 0..8usize {
            let bounded = edit_distance_at_most(a, b, true, max);
            if exact <= max {
                assert_eq!(bounded, Some(exact), "{a:?} vs {b:?} at max {max}");
            } else {
                assert_eq!(bounded, None, "{a:?} vs {b:?} at max {max}");
            }
        }
    }
}

#[test]
fn length_difference_alone_rejects_without_running_the_dp() {
    assert_eq!(edit_distance_at_most(b"a", b"abcdefgh", true, 2), None);
    assert_eq!(edit_distance_at_most(b"abcdefgh", b"a", true, 2), None);
}

#[test]
fn fuzzy_match_respects_max_edits_boundary() {
    assert!(FuzzyMatch::new(b"cat", 1, 0, true).matches(b"cot"));
    assert!(!FuzzyMatch::new(b"cat", 0, 0, true).matches(b"cot"));
    assert!(FuzzyMatch::new(b"kitten", 3, 0, true).matches(b"sitting"));
    assert!(!FuzzyMatch::new(b"kitten", 2, 0, true).matches(b"sitting"));
}

#[test]
fn fuzzy_match_prefix_length_excludes_otherwise_in_range_candidate() {
    let m = FuzzyMatch::new(b"cat", 2, 1, true);
    assert!(m.matches(b"cot")); // starts with "c", distance 1
    assert!(!m.matches(b"bat")); // starts with "b", excluded regardless
}

#[test]
fn fuzzy_match_prefix_length_zero_imposes_no_prefix_requirement() {
    assert!(FuzzyMatch::new(b"cat", 2, 0, true).matches(b"bat"));
}

#[test]
fn fuzzy_match_rejects_candidate_shorter_than_prefix_length() {
    assert!(!FuzzyMatch::new(b"cat", 2, 2, true).matches(b"c"));
}

#[test]
fn fuzzy_match_exact_match_is_distance_zero() {
    assert!(FuzzyMatch::new(b"cat", 0, 0, true).matches(b"cat"));
    assert_eq!(FuzzyMatch::new(b"cat", 0, 0, true).edits(b"cat"), Some(0));
}

/// `FuzzyAutomatonBuilder` builds the automaton from the **suffix** past
/// `prefixLength`, so the edit budget applies to the suffix alone. Running
/// the DP over the whole terms would be a strictly weaker test: a shared
/// prefix can absorb part of an alignment.
#[test]
fn prefix_length_takes_the_prefix_out_of_the_edit_budget() {
    // "aab" vs "ab" is distance 1 over the whole terms, but with
    // prefixLength=1 the fixed "a" is removed from both and the suffixes
    // "ab" vs "b" are still distance 1 -- agreement here.
    let m = FuzzyMatch::new(b"aab", 1, 1, true);
    assert_eq!(m.edits(b"aab"), Some(0));
    assert_eq!(m.edits(b"ab"), Some(1));
    // A candidate whose only cheap alignment runs through the fixed
    // prefix is rejected: target "aaa" with prefixLength 2 fixes "aa",
    // leaving "a" vs "" (1 edit) for candidate "aa" -- fine -- but
    // candidate "aaaaa" leaves "a" vs "aaa" (2 edits), out of range at
    // maxEdits 1 even though the whole-term distance is 2 as well.
    let m = FuzzyMatch::new(b"aaa", 1, 2, true);
    assert_eq!(m.edits(b"aa"), Some(1));
    assert_eq!(m.edits(b"aaaa"), Some(1));
    assert_eq!(m.edits(b"aaaaa"), None);
}

#[test]
fn prefix_length_is_measured_in_codepoints() {
    // "é" is 2 UTF-8 bytes; prefixLength 1 must fix one *character*.
    let m = FuzzyMatch::new("état".as_bytes(), 1, 1, true);
    assert_eq!(m.literal_prefix(), "é".as_bytes());
    assert!(m.matches("étet".as_bytes()));
    assert!(!m.matches("etat".as_bytes()));
    // A prefix longer than the term clamps to the term.
    let m = FuzzyMatch::new("été".as_bytes(), 2, 10, true);
    assert_eq!(m.literal_prefix(), "été".as_bytes());
    assert_eq!(m.term_length(), 3);
}

#[test]
fn literal_prefix_returns_the_targets_own_prefix_bytes() {
    assert_eq!(FuzzyMatch::new(b"cat", 2, 2, true).literal_prefix(), b"ca");
    assert_eq!(FuzzyMatch::new(b"cat", 2, 0, true).literal_prefix(), b"");
    assert_eq!(
        FuzzyMatch::new(b"cat", 2, 10, true).literal_prefix(),
        b"cat"
    );
}

/// `FuzzyTermsEnum.next`'s `BoostAttribute` formula, term by term. This
/// is what makes `FuzzyQuery` scored rather than a flat 1.0.
#[test]
fn boost_matches_fuzzy_terms_enums_similarity_formula() {
    let m = FuzzyMatch::new(b"kitten", 2, 0, true);
    // Exact match short-circuits to 1.0 without dividing.
    assert_eq!(m.boost(b"kitten"), Some(1.0));
    // ed=1, min(codePointCount("kitte")=5, termLength=6) = 5.
    assert_eq!(m.boost(b"kitte"), Some(1.0 - 1.0 / 5.0));
    // ed=1, min(7, 6) = 6.
    assert_eq!(m.boost(b"kittens"), Some(1.0 - 1.0 / 6.0));
    // ed=2, min(6, 6) = 6.
    assert_eq!(m.boost(b"mitten"), Some(1.0 - 1.0 / 6.0));
    assert_eq!(m.boost(b"sitteb"), Some(1.0 - 2.0 / 6.0));
    // Out of range at all.
    assert_eq!(m.boost(b"zzzzzz"), None);
}

#[test]
fn boost_can_be_zero_or_negative_for_very_short_terms() {
    // Lucene's own javadoc calls this out: `FuzzyQuery` on "a" with
    // maxEdits 2 gives "abc" a similarity of 1 - 2/1 = -1, which
    // `TopTermsRewrite.build` then truncates to 0.
    let m = FuzzyMatch::new(b"a", 2, 0, true);
    assert_eq!(m.boost(b"abc"), Some(-1.0));
    let m = FuzzyMatch::new(b"ab", 2, 0, true);
    assert_eq!(m.boost(b"abcd"), Some(0.0));
}

#[test]
fn boost_uses_codepoint_counts_not_byte_counts() {
    // "café" is 5 bytes but 4 codepoints; ed("café","cafe") = 1, so the
    // similarity denominator must be 4, not 5.
    let m = FuzzyMatch::new("café".as_bytes(), 2, 0, true);
    assert_eq!(m.boost("cafe".as_bytes()), Some(1.0 - 1.0 / 4.0));
}

#[test]
fn constants_match_fuzzy_querys_defaults() {
    assert_eq!(MAXIMUM_SUPPORTED_DISTANCE, 2);
    assert_eq!(DEFAULT_MAX_EDITS, 2);
    assert_eq!(DEFAULT_PREFIX_LENGTH, 0);
    assert_eq!(DEFAULT_MAX_EXPANSIONS, 50);
    const { assert!(DEFAULT_TRANSPOSITIONS) };
}

/// The per-budget helpers agree with the whole-budget ones: matching
/// within `k` edits is having a distance within `k`; the boost of an
/// edit count is `boost`'s for the candidate's own distance -- invalid
/// UTF-8 counted as the decoder reads it, an empty side boosting 1; the
/// smaller automata are built once each, the full budget's is the
/// pattern's own, and a term the automaton cannot spell builds none.
#[test]
fn per_budget_helpers_agree_with_the_whole_budget() {
    let m = FuzzyMatch::new(b"lucene", 2, 1, true);
    // ill-formed UTF-8 is out of range, as in Lucene's UTF-8 automaton
    assert_eq!(m.edits(b"l\xffcene"), None);
    assert_eq!(m.edits(b"lucen\xff"), None);
    for c in [
        &b"lucene"[..],
        b"lucine",
        b"lcuene",
        b"lucenes",
        b"luc",
        b"xucene",
        b"l\xffcene",
    ] {
        for k in 0..=2u8 {
            assert_eq!(
                m.matches_within(c, k),
                m.edits_within(c, k).is_some(),
                "{c:?} {k}"
            );
        }
        if let Some(ed) = m.edits(c) {
            assert_eq!(Some(m.boost_from_edits(c, ed)), m.boost(c), "{c:?}");
        }
    }
    assert_eq!(m.boost_from_edits(b"lucene", 0), 1.0);
    let empty = FuzzyMatch::new(b"", 1, 0, true);
    assert_eq!(empty.boost(b"a"), Some(1.0));
    assert_eq!(empty.boost_from_edits(b"a", 1), 1.0);

    let full = m.cached_compiled_dfa().unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &m.cached_compiled_dfa_within(2).unwrap(),
        &full
    ));
    assert!(std::sync::Arc::ptr_eq(
        &m.cached_compiled_dfa_within(5).unwrap(),
        &full
    ));
    for k in 0..2u8 {
        let once = m.cached_compiled_dfa_within(k).unwrap();
        assert!(!std::sync::Arc::ptr_eq(&once, &full));
        assert!(std::sync::Arc::ptr_eq(
            &m.cached_compiled_dfa_within(k).unwrap(),
            &once
        ));
    }
    // An unpaired surrogate decodes to U+FFFD, which no automaton spells.
    let bad = FuzzyMatch::new(b"a\xed\xa0\x80b", 1, 0, true);
    assert!(bad.to_dfa(1).is_none());
    assert!(bad.cached_compiled_dfa().is_none());
    assert!(bad.cached_compiled_dfa_within(0).is_none());
}

#[test]
fn compiled_automata_are_shared_between_matchers_of_one_term() {
    let a = FuzzyMatch::new(b"sharedterm", 2, 1, true);
    let b = FuzzyMatch::new(b"sharedterm", 2, 1, true);
    let first = a.cached_compiled_dfa().unwrap();
    assert!(std::sync::Arc::ptr_eq(
        &b.cached_compiled_dfa().unwrap(),
        &first
    ));
    // Any difference in what the automaton is built from is a different one.
    for other in [
        FuzzyMatch::new(b"sharedterm", 1, 1, true),
        FuzzyMatch::new(b"sharedterm", 2, 0, true),
        FuzzyMatch::new(b"sharedterm", 2, 1, false),
        FuzzyMatch::new(b"sharedterms", 2, 1, true),
    ] {
        assert!(!std::sync::Arc::ptr_eq(
            &other.cached_compiled_dfa().unwrap(),
            &first
        ));
    }
    // A pattern without an automaton is remembered as having none.
    let bad = FuzzyMatch::new(b"sh\xed\xa0\x80red", 1, 0, true);
    assert!(bad.cached_compiled_dfa().is_none());
    let key = (bad.term.to_vec(), 1, 0, true);
    assert!(matches!(shared_dfas().get(&key), Some(None)));
}

#[test]
fn the_shared_automata_are_emptied_when_over_budget() {
    let cache = SharedDfas {
        map: std::sync::Mutex::new(SharedDfaMap::default()),
    };
    let m = FuzzyMatch::new(b"budget", 1, 0, true);
    let dfa = m.cached_compiled_dfa();
    let small = (b"small".to_vec(), 1u8, 0usize, true);
    cache.put(small.clone(), dfa.clone());
    cache.put(small.clone(), dfa.clone());
    let weight = cache.map.lock().unwrap().bytes;
    assert_eq!(
        weight,
        small.0.len() + dfa.as_ref().unwrap().dfa.heap_bytes()
    );
    // A key as big as the budget cannot sit beside anything else.
    cache.put((vec![0; SHARED_DFA_BUDGET_BYTES], 1, 0, true), None);
    assert!(cache.get(&small).is_none());
    assert_eq!(cache.map.lock().unwrap().dfas.len(), 1);
}

#[test]
fn an_owned_pattern_matches_as_the_borrowed_one() {
    let term = b"quick".to_vec();
    let borrowed = FuzzyMatch::new(&term, 1, 1, true);
    let expected: Vec<bool> = [&b"quick"[..], b"quack", b"qiuck", b"xuick", b""]
        .iter()
        .map(|t| borrowed.matches(t))
        .collect();
    let owned = borrowed.into_owned();
    drop(term);
    let got: Vec<bool> = [&b"quick"[..], b"quack", b"qiuck", b"xuick", b""]
        .iter()
        .map(|t| owned.matches(t))
        .collect();
    assert_eq!(got, expected);
    assert_eq!(expected, [true, true, true, false, false]);
}
