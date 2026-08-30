//! Edit-distance matching over already-decoded term bytes -- the term side of
//! what real Lucene's `org.apache.lucene.search.FuzzyQuery` does when it
//! compiles a target term into a `LevenshteinAutomata`/`CompiledAutomaton`
//! and drives `FuzzyTermsEnum`/`IntersectTermsEnum` to walk only the trie
//! blocks within edit-distance range.
//!
//! ## What this module reproduces
//!
//! Real Lucene never computes an edit distance per candidate term: it builds
//! a Levenshtein automaton (`LevenshteinAutomata.toAutomaton(n, prefix)`) and
//! runs it. This module computes the distance directly with a banded DP
//! instead. Same accept/reject decision, different machine -- see
//! `docs/sweep/m2/b8-automata-analysis.md` for the cost.
//!
//! The **semantics** are Lucene's exactly, and each of these is a rule this
//! port used to get wrong:
//!
//! - **Codepoints, not bytes.** `FuzzyAutomatonBuilder` decodes the term to
//!   `int[]` codepoints (`stringToUTF32`) and builds the automaton over
//!   `Character.MAX_CODE_POINT`, so substituting one 3-byte character for
//!   another costs **one** edit, not three. [`edit_distance`] decodes both
//!   sides the same way.
//! - **`prefixLength` is a fixed, non-fuzzy prefix measured in codepoints**,
//!   clamped to the term's length, and the edit budget applies to what is
//!   *left* -- `FuzzyAutomatonBuilder`'s constructor splits `codePoints` at
//!   `prefixLength` and hands only the `suffix` to `LevenshteinAutomata`.
//!   Comparing the whole terms instead is not the same test: a shared prefix
//!   can absorb part of an alignment, so whole-term distance is `<=` suffix
//!   distance and would accept terms Lucene rejects.
//! - **`maxEdits` is capped at
//!   [`MAXIMUM_SUPPORTED_DISTANCE`]** (`LevenshteinAutomata
//!   .MAXIMUM_SUPPORTED_DISTANCE == 2`): `FuzzyQuery`'s constructor throws
//!   `IllegalArgumentException` above it, because Lucene only ships
//!   parametric descriptions for distances 1 and 2. This module accepts a
//!   larger `max_edits` (its DP has no such ceiling) but callers that want
//!   `FuzzyQuery` parity must reject it -- see
//!   [`MAXIMUM_SUPPORTED_DISTANCE`].
//! - **Transpositions cost one edit** (`FuzzyQuery.defaultTranspositions ==
//!   true`), and the variant is the *restricted* one ("optimal string
//!   alignment"): a transposed pair may not be edited again afterwards. With
//!   `transpositions: false` an adjacent swap costs two edits, matching
//!   classic Levenshtein.
//! - **Scoring**: [`FuzzyMatch::boost`] is `FuzzyTermsEnum.next`'s
//!   `BoostAttribute` value, which is what makes `FuzzyQuery` a *scored*
//!   multi-term query rather than a constant-scoring one.
//!
//! An ill-formed UTF-8 byte in either term decodes to `U+FFFD`, so it
//! compares equal to any other ill-formed byte rather than making the term
//! unmatchable -- terms here are `Vec<u8>` with no guaranteed UTF-8 validity,
//! and edit distance is a *similarity* measure where being lenient is the
//! safer failure mode.

/// `LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE`. `FuzzyQuery`'s
/// constructor rejects a `maxEdits` above this outright ("maxEdits must be
/// between 0 and 2"), because Lucene only ships `Lev1`/`Lev2` parametric
/// descriptions. The DP in this module has no such ceiling, so this constant
/// exists for callers that need to reproduce Lucene's validation.
pub const MAXIMUM_SUPPORTED_DISTANCE: u8 = 2;

/// `FuzzyQuery.defaultMaxEdits` -- `MAXIMUM_SUPPORTED_DISTANCE`.
pub const DEFAULT_MAX_EDITS: u8 = MAXIMUM_SUPPORTED_DISTANCE;

/// `FuzzyQuery.defaultPrefixLength`.
pub const DEFAULT_PREFIX_LENGTH: usize = 0;

/// `FuzzyQuery.defaultMaxExpansions`.
pub const DEFAULT_MAX_EXPANSIONS: usize = 50;

/// `FuzzyQuery.defaultTranspositions`.
pub const DEFAULT_TRANSPOSITIONS: bool = true;

/// Decodes `bytes` to codepoints, mapping each ill-formed byte to `U+FFFD`.
fn to_chars(bytes: &[u8]) -> Vec<char> {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.chars().collect(),
        Err(_) => String::from_utf8_lossy(bytes).chars().collect(),
    }
}

/// Computes the edit distance between two byte strings, **in codepoints**.
///
/// With `transpositions: true` this is the "restricted"/optimal-string-
/// alignment Damerau-Levenshtein distance: substitution, insertion, deletion
/// and adjacent transposition each cost 1, and (unlike full, unrestricted
/// Damerau-Levenshtein) a transposed pair may not be edited again afterwards.
/// That is the variant `LevenshteinAutomata`'s `Lev1T`/`Lev2T` parametric
/// descriptions encode, so this matches real Lucene's actual behaviour rather
/// than "a" Damerau-Levenshtein.
///
/// With `transpositions: false` this is plain Levenshtein distance, matching
/// `FuzzyQuery(term, maxEdits, prefixLength, maxExpansions, false)`, where an
/// adjacent swap costs 2.
///
/// `O(n*m)` in codepoints. [`edit_distance_at_most`] is the banded form the
/// hot path uses.
pub fn edit_distance(a: &[u8], b: &[u8], transpositions: bool) -> usize {
    distance_chars(&to_chars(a), &to_chars(b), transpositions, usize::MAX)
        .expect("an unbounded budget always yields a distance")
}

/// [`edit_distance`] with an early exit: returns `None` as soon as the true
/// distance is known to exceed `max`.
///
/// This is the shape a term scan actually needs, and it turns the DP from
/// `O(n*m)` into `O(n*(2*max+1))` -- with `FuzzyQuery`'s `maxEdits <= 2` that
/// is 5 cells per row instead of `m`. It also short-circuits on the length
/// difference, which alone rejects most of a term dictionary.
pub fn edit_distance_at_most(
    a: &[u8],
    b: &[u8],
    transpositions: bool,
    max: usize,
) -> Option<usize> {
    distance_chars(&to_chars(a), &to_chars(b), transpositions, max)
}

/// The banded DP itself. `max == usize::MAX` means "unbounded", in which case
/// the band covers the whole table and the result is always `Some`.
// ARITH: `n` and `m` are `Vec<char>` lengths, so `n + 1` and `m + 1` cannot
// overflow (a `Vec`'s length is at most `isize::MAX`, and a `Vec<char>`'s at
// most a quarter of that). Inside the DP, `i` runs over `1..=n` and `j` over
// `lo..=hi` with `lo >= 1` and `hi <= m`, so `i - 1`, `j - 1` are in bounds;
// `i - 2` and `j - 2` only run under the `i > 1 && j > 1` guard. The band's
// upper edge is `i.saturating_add(max)` rather than `i + max`: `max` is a
// caller-supplied edit budget whose sentinel is `usize::MAX`, and the
// `.min(m)` that follows makes saturation exactly equal to the unsaturated
// result, not an approximation of it.
#[allow(clippy::arithmetic_side_effects)]
fn distance_chars(a: &[char], b: &[char], transpositions: bool, max: usize) -> Option<usize> {
    let n = a.len();
    let m = b.len();
    if max != usize::MAX && n.abs_diff(m) > max {
        return None;
    }
    // `dp[i][j]` = edit distance between `a[..i]` and `b[..j]`. Rows are kept
    // in full (three of them are live at once for the transposition rule);
    // the *band* is applied to the `j` loop, not the allocation, since a row
    // is at most a term's length and terms are short.
    let unreachable = usize::MAX / 4;
    let mut dp = vec![vec![unreachable; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate().take(n + 1) {
        if i <= max {
            row[0] = i;
        }
    }
    for (j, cell) in dp[0].iter_mut().enumerate().take(m + 1) {
        if j <= max {
            *cell = j;
        }
    }
    for i in 1..=n {
        // Only `j` within `max` of the diagonal can ever hold a value `<=
        // max`, so everything outside the band stays `unreachable`.
        let lo = if max == usize::MAX {
            1
        } else {
            i.saturating_sub(max).max(1)
        };
        let hi = if max == usize::MAX {
            m
        } else {
            i.saturating_add(max).min(m)
        };
        for j in lo..=hi {
            let cost = usize::from(a[i - 1] != b[j - 1]);
            let mut best = dp[i - 1][j]
                .min(dp[i][j - 1])
                .saturating_add(1)
                .min(dp[i - 1][j - 1].saturating_add(cost));
            if transpositions && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(dp[i - 2][j - 2].saturating_add(1));
            }
            dp[i][j] = best;
        }
    }
    let d = dp[n][m];
    if d > max {
        None
    } else {
        Some(d)
    }
}

/// A compiled `FuzzyQuery` match predicate: a target `term`, a maximum edit
/// distance `max_edits`, a required exact-match `prefix_length` (in
/// **codepoints**, as `FuzzyAutomatonBuilder`'s is), and whether
/// transpositions count as a single edit. Mirrors `wildcard.rs`'s
/// `WildcardPattern`: a small, cheap-to-build value that
/// [`crate::blocktree::FieldTerms`]'s scanning logic tests every candidate
/// term against.
#[derive(Debug, Clone)]
pub struct FuzzyMatch<'a> {
    term: &'a [u8],
    /// `term` decoded once, so a scan over thousands of candidates does not
    /// re-decode the target for each of them.
    term_chars: Vec<char>,
    /// `prefix_length` clamped to `term_chars.len()`, as
    /// `FuzzyAutomatonBuilder` clamps it.
    prefix_chars: usize,
    /// Byte length of the first `prefix_chars` codepoints of `term`.
    prefix_bytes: usize,
    max_edits: u8,
    transpositions: bool,
}

impl<'a> FuzzyMatch<'a> {
    pub fn new(term: &'a [u8], max_edits: u8, prefix_length: usize, transpositions: bool) -> Self {
        let term_chars = to_chars(term);
        let prefix_chars = prefix_length.min(term_chars.len());
        let prefix_bytes = term_chars[..prefix_chars]
            .iter()
            .map(|c| c.len_utf8())
            .sum();
        Self {
            term,
            term_chars,
            prefix_chars,
            prefix_bytes,
            max_edits,
            transpositions,
        }
    }

    /// The target term's fixed, non-fuzzy prefix as bytes -- the first
    /// `prefix_length` **codepoints**, clamped to the term's own length.
    /// Every matching candidate must start with exactly this run, since real
    /// `FuzzyQuery` holds those codepoints outside the automaton entirely
    /// rather than spending edits on them. Used by
    /// [`crate::blocktree::FieldTerms`] to narrow its scan to a contiguous
    /// sorted range via binary search first, the same literal-prefix-range
    /// trick `wildcard.rs`'s `literal_prefix` already uses.
    pub fn literal_prefix(&self) -> &'a [u8] {
        &self.term[..self.prefix_bytes]
    }

    /// The exact edit distance between the target and `candidate`, or `None`
    /// when `candidate` is out of range.
    ///
    /// This is `FuzzyTermsEnum.next`'s inner loop expressed directly: Lucene
    /// runs the `maxEdits` automaton to accept the term, then walks *down*
    /// through the `maxEdits-1`, `maxEdits-2`, ... automata to find the
    /// smallest distance that still accepts. The distance measured is the one
    /// between the two **suffixes** past `prefix_length`, because the prefix
    /// is fixed.
    pub fn edits(&self, candidate: &[u8]) -> Option<usize> {
        if candidate.len() < self.prefix_bytes
            || candidate[..self.prefix_bytes] != self.term[..self.prefix_bytes]
        {
            return None;
        }
        let candidate_chars = to_chars(candidate);
        if candidate_chars.len() < self.prefix_chars {
            return None;
        }
        distance_chars(
            &self.term_chars[self.prefix_chars..],
            &candidate_chars[self.prefix_chars..],
            self.transpositions,
            self.max_edits as usize,
        )
    }

    /// Tests whether `candidate` matches: it must start with this pattern's
    /// fixed prefix exactly, and the two suffixes' edit distance must be
    /// `<= max_edits`.
    pub fn matches(&self, candidate: &[u8]) -> bool {
        self.edits(candidate).is_some()
    }

    /// `FuzzyTermsEnum.next`'s `BoostAttribute` value for `candidate`, or
    /// `None` when `candidate` does not match at all.
    ///
    /// Java (`FuzzyTermsEnum.java`):
    ///
    /// ```java
    /// if (ed == 0) {                       // exact match
    ///   boostAtt.setBoost(1.0F);
    /// } else {
    ///   final int codePointCount = UnicodeUtil.codePointCount(term);
    ///   int minTermLength = Math.min(codePointCount, termLength);
    ///   float similarity = 1.0f - (float) ed / (float) minTermLength;
    ///   boostAtt.setBoost(similarity);
    /// }
    /// ```
    ///
    /// `termLength` there is `FuzzyAutomatonBuilder.getTermLength()`, i.e.
    /// the **whole** query term's codepoint count -- not the suffix's -- so a
    /// long shared prefix raises every candidate's boost. The result can be
    /// zero or negative for very short terms (`"a"` vs `"abc"` at 2 edits
    /// gives `-1.0`); `TopTermsRewrite.build` truncates that to `0` when it
    /// builds the query, so this method returns it unclamped and leaves the
    /// truncation to the scorer, exactly as Lucene splits the two steps.
    pub fn boost(&self, candidate: &[u8]) -> Option<f32> {
        let ed = self.edits(candidate)?;
        if ed == 0 {
            return Some(1.0);
        }
        let candidate_len = to_chars(candidate).len();
        let min_term_length = candidate_len.min(self.term_chars.len());
        if min_term_length == 0 {
            // Unreachable in practice (a zero-length candidate at distance
            // `ed > 0` from a zero-length term cannot exist), but division by
            // zero is not an acceptable way to find that out.
            return Some(1.0);
        }
        Some(1.0 - (ed as f32) / (min_term_length as f32))
    }

    /// The target term's codepoint count -- `FuzzyAutomatonBuilder
    /// .getTermLength()`, the denominator half of [`Self::boost`].
    pub fn term_length(&self) -> usize {
        self.term_chars.len()
    }
}

#[cfg(test)]
mod tests {
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
    /// another is **one** edit. This module used to count UTF-8 bytes and
    /// reported 2 or 3 for the same pair, which silently narrowed every
    /// non-ASCII fuzzy query.
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
    /// `prefixLength`, so the edit budget applies to the suffix alone. This
    /// port used to run the DP over the whole terms, which is a strictly
    /// weaker test: a shared prefix can absorb part of an alignment.
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
    /// is what makes `FuzzyQuery` scored; the port used to give every fuzzy
    /// hit a flat 1.0.
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
}
