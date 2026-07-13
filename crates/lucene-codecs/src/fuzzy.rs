//! Edit-distance matching over already-decoded term bytes — the term side of
//! what real Lucene's `org.apache.lucene.search.FuzzyQuery` does when it
//! compiles a target term into a `LevenshteinAutomata`/`CompiledAutomaton`
//! and drives `FuzzyTermsEnum`/`IntersectTermsEnum` to walk only the trie
//! blocks within edit-distance range.
//!
//! ## Scope of this slice (task #42)
//!
//! **What's here**: a plain Damerau-Levenshtein-with-transpositions (and,
//! with `transpositions: false`, plain Levenshtein) edit-distance function
//! plus a [`FuzzyMatch`] predicate combining it with `FuzzyQuery`'s
//! `prefixLength` exact-prefix requirement, tested by linear scan over
//! [`crate::blocktree::FieldTerms`]'s already-materialized `entries` (same
//! "narrow by literal prefix, then filter" pattern `wildcard.rs`/
//! `FieldTerms::intersect` already established for `WildcardQuery`/
//! `PrefixQuery` — see that module's doc comment).
//!
//! **What's deliberately NOT here** (matching `wildcard.rs`'s own documented
//! tradeoff, now closing that module's last remaining explicitly-deferred
//! gap): no `LevenshteinAutomata` table, no `CompiledAutomaton`/
//! `ByteRunAutomaton` DFA compilation, no `IntersectTermsEnum` block-skipping.
//! Real Lucene's `LevenshteinAutomata` only supports precomputed edit
//! distances 0/1/2 (`LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE = 2`)
//! specifically because building the automaton table is the expensive part;
//! this port instead computes the distance directly with an `O(n*m)` DP each
//! time a candidate term is tested, so there is no such ceiling here — a
//! `max_edits` above 2 works, it is just not what real `FuzzyQuery` itself
//! ever produces (its public API rejects `maxEdits > 2` outright).
//!
//! ## Byte-vs-codepoint scope decision — read before assuming this handles
//! full Unicode
//!
//! Real Lucene's `LevenshteinAutomata` operates on **UTF-32 codepoints**, not
//! UTF-16 `char`s and not raw UTF-8 bytes — `FuzzyQuery` explicitly decodes
//! the target term to `int[]` codepoints (`UnicodeUtil.codePointCount`/
//! `codePointAt`) before building the automaton, specifically so a multi-byte
//! character counts as *one* edit unit, not 2-4. This module's
//! [`edit_distance`] operates on raw **bytes** instead (this port's terms are
//! `Vec<u8>` with no guaranteed UTF-8 validity — see `TermQuery::term`'s own
//! doc comment, and `wildcard.rs`'s `?` token already had to special-case
//! codepoint width for the same reason). This is a **deliberate, stated
//! shortcut**, not an oversight: every term in this port's fixtures is ASCII,
//! where one byte and one codepoint coincide, so byte-level distance is
//! pragmatically equivalent to codepoint-level distance for everything this
//! port can currently test against. For a non-ASCII term containing a
//! multi-byte UTF-8 character, this module's byte-level distance would
//! over-count edits inside that character (e.g. substituting one 3-byte
//! codepoint for another 3-byte codepoint is 1 codepoint-edit in real Lucene
//! but up to 3 byte-edits here) — full codepoint decoding (mirroring
//! `wildcard.rs`'s `utf8_codepoint_len` approach, generalized to decode
//! rather than just measure) is deferred as a documented gap, not
//! implemented as a false-equivalence "shortcut that quietly also claims
//! Unicode correctness."

/// Computes the edit distance between two byte strings.
///
/// With `transpositions: true`, this is the common "restricted"/"optimal
/// string alignment" Damerau-Levenshtein distance: substitution, insertion,
/// deletion, and adjacent-transposition each cost 1 edit, and (unlike full,
/// unrestricted Damerau-Levenshtein) a transposed pair may not be edited
/// again afterward — real Lucene's own `LevenshteinAutomata`/
/// `FuzzyTermsEnum` (`FuzzyQuery`'s default `transpositions = true`)
/// documents itself as this same restricted variant, not full
/// Damerau-Levenshtein, so this matches real Lucene's actual behavior, not
/// just "a" Damerau-Levenshtein.
///
/// With `transpositions: false`, this is plain Levenshtein distance
/// (substitution/insertion/deletion only) — matching `FuzzyQuery`'s behavior
/// when constructed with `transpositions = false`, in which case an adjacent
/// swap costs 2 edits (a deletion plus an insertion, or two substitutions),
/// not 1.
///
/// Standard `O(n*m)` dynamic-programming table; `n`/`m` are the two inputs'
/// byte lengths, small for real term dictionary entries, so no early-exit
/// optimization (banding the DP to `max_edits`) is applied here — this
/// mirrors `wildcard.rs`'s own "correctness first" stance for this slice's
/// scope (see the module doc).
pub fn edit_distance(a: &[u8], b: &[u8], transpositions: bool) -> usize {
    let n = a.len();
    let m = b.len();
    // `dp[i][j]` = edit distance between `a[..i]` and `b[..j]`.
    let mut dp = vec![vec![0usize; m + 1]; n + 1];
    for (i, row) in dp.iter_mut().enumerate().take(n + 1) {
        row[0] = i;
    }
    for (j, cell) in dp[0].iter_mut().enumerate().take(m + 1) {
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

/// A compiled `FuzzyQuery` match predicate: a target `term`, a maximum edit
/// distance `max_edits`, a required exact-match `prefix_length` (in bytes —
/// see the module doc's byte-vs-codepoint scope note), and whether
/// transpositions count as a single edit ([`edit_distance`]'s
/// `transpositions` flag). Mirrors `wildcard.rs`'s `WildcardPattern`: a
/// small, cheap-to-build value that [`crate::blocktree::FieldTerms`]'s
/// scanning logic tests every candidate term against.
#[derive(Debug, Clone)]
pub struct FuzzyMatch<'a> {
    term: &'a [u8],
    max_edits: u8,
    prefix_length: usize,
    transpositions: bool,
}

impl<'a> FuzzyMatch<'a> {
    pub fn new(term: &'a [u8], max_edits: u8, prefix_length: usize, transpositions: bool) -> Self {
        Self {
            term,
            max_edits,
            prefix_length,
            transpositions,
        }
    }

    /// The target term's first `prefix_length` bytes — every matching
    /// candidate must start with exactly this run (real `FuzzyQuery`'s
    /// `prefixLength` characters are held fixed, outside the
    /// automaton/edit-distance budget entirely, not merely "free" edits
    /// within it). Used by [`crate::blocktree::FieldTerms`] to narrow its
    /// scan to a contiguous sorted range via binary search first, the same
    /// literal-prefix-range trick `wildcard.rs`'s `literal_prefix`/
    /// `FieldTerms::intersect` already use for `WildcardQuery`/`PrefixQuery`.
    pub fn literal_prefix(&self) -> &'a [u8] {
        &self.term[..self.prefix_length.min(self.term.len())]
    }

    /// Tests whether `candidate` matches: `candidate` must start with this
    /// pattern's `prefix_length`-byte literal prefix exactly (a shorter
    /// candidate that can't even hold that many bytes never matches), and
    /// the two terms' edit distance (over their **full** bytes — a matching
    /// literal prefix already costs 0 edits under any correct edit-distance
    /// alignment, so restricting the DP to just the suffixes would be an
    /// optimization, not a semantic difference) must be `<= max_edits`.
    pub fn matches(&self, candidate: &[u8]) -> bool {
        let prefix = self.literal_prefix();
        if candidate.len() < prefix.len() || &candidate[..prefix.len()] != prefix {
            return false;
        }
        edit_distance(self.term, candidate, self.transpositions) <= self.max_edits as usize
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
    /// (Damerau-Levenshtein), while `transpositions = false` treats the same
    /// swap as two edits (plain Levenshtein) -- getting this backwards is
    /// exactly the class of subtle bug the `differential-testing` skill
    /// exists to catch.
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

    #[test]
    fn fuzzy_match_respects_max_edits_boundary() {
        // "cat" vs "cot" is distance 1: matches at max_edits=1, not at 0.
        assert!(FuzzyMatch::new(b"cat", 1, 0, true).matches(b"cot"));
        assert!(!FuzzyMatch::new(b"cat", 0, 0, true).matches(b"cot"));
        // Exactly at the limit vs one over.
        assert!(FuzzyMatch::new(b"kitten", 3, 0, true).matches(b"sitting"));
        assert!(!FuzzyMatch::new(b"kitten", 2, 0, true).matches(b"sitting"));
    }

    #[test]
    fn fuzzy_match_prefix_length_excludes_otherwise_in_range_candidate() {
        // "cat" -> "bat" is distance 1, well within max_edits=2, but with
        // prefix_length=1 the candidate must start with "c" exactly.
        let m = FuzzyMatch::new(b"cat", 2, 1, true);
        assert!(m.matches(b"cot")); // starts with "c", distance 1
        assert!(!m.matches(b"bat")); // starts with "b", excluded regardless of distance
    }

    #[test]
    fn fuzzy_match_prefix_length_zero_imposes_no_prefix_requirement() {
        let m = FuzzyMatch::new(b"cat", 2, 0, true);
        assert!(m.matches(b"bat"));
    }

    #[test]
    fn fuzzy_match_rejects_candidate_shorter_than_prefix_length() {
        let m = FuzzyMatch::new(b"cat", 2, 2, true);
        assert!(!m.matches(b"c"));
    }

    #[test]
    fn fuzzy_match_exact_match_is_distance_zero() {
        assert!(FuzzyMatch::new(b"cat", 0, 0, true).matches(b"cat"));
    }

    #[test]
    fn literal_prefix_returns_the_targets_own_prefix_bytes() {
        assert_eq!(FuzzyMatch::new(b"cat", 2, 2, true).literal_prefix(), b"ca");
        assert_eq!(FuzzyMatch::new(b"cat", 2, 0, true).literal_prefix(), b"");
        // prefix_length longer than the term itself clamps to the term's length.
        assert_eq!(
            FuzzyMatch::new(b"cat", 2, 10, true).literal_prefix(),
            b"cat"
        );
    }
}
