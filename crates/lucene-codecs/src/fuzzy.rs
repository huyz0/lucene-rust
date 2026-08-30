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
///
/// **Three rolling rows, one allocation.** The table is `(n + 1) * (m + 1)`
/// but only rows `i`, `i - 1` and `i - 2` are ever read -- the third only
/// because of the transposition rule, which is `Lev1T`/`Lev2T`'s and not
/// plain Levenshtein's. This used to be a `Vec<Vec<usize>>`, i.e. **`n + 2`
/// heap allocations per candidate term**, and a fuzzy expansion tests every
/// term in the dictionary's prefix range: on the 1 M-term corpus that is over
/// eight million allocations for one query. Java allocates nothing at all here
/// (it runs a compiled DFA over the term's bytes), so the flat buffer is not
/// an optimisation past Java, it is the gap closing.
// ARITH: `n` and `m` are `Vec<char>` lengths, so `n + 1` and `m + 1` cannot
// overflow (a `Vec`'s length is at most `isize::MAX`, and a `Vec<char>`'s at
// most a quarter of that); `3 * width` likewise, since `width <= isize::MAX/4`.
// Inside the DP, `i` runs over `1..=n` and `j` over `lo..=hi` with `lo >= 1`
// and `hi <= m`, so `i - 1`, `j - 1` are in bounds; `i - 2` and `j - 2` only
// run under the `i > 1 && j > 1` guard. Every `row * width + j` is below
// `3 * width` because `row < 3` and `j <= m < width`. The band's upper edge is
// `i.saturating_add(max)` rather than `i + max`: `max` is a caller-supplied
// edit budget whose sentinel is `usize::MAX`, and the `.min(m)` that follows
// makes saturation exactly equal to the unsaturated result, not an
// approximation of it.
#[allow(clippy::arithmetic_side_effects)]
fn distance_chars(a: &[char], b: &[char], transpositions: bool, max: usize) -> Option<usize> {
    let n = a.len();
    let m = b.len();
    if max != usize::MAX && n.abs_diff(m) > max {
        return None;
    }
    // A cell outside the band, or past the budget, is "unreachable" rather
    // than a real distance. `usize::MAX / 4` leaves room for the
    // `saturating_add(1)`s below without wrapping into a plausible distance.
    let unreachable = usize::MAX / 4;
    let width = m + 1;
    let mut buf = vec![unreachable; 3 * width];

    // Row 0: `dp[0][j] = j`, but only inside the budget.
    for (j, cell) in buf.iter_mut().enumerate().take(width) {
        *cell = if j <= max { j } else { unreachable };
    }
    if n == 0 {
        let d = buf[m];
        return if d > max { None } else { Some(d) };
    }

    for i in 1..=n {
        // `i % 3`, `(i - 1) % 3`, `(i - 2) % 3`, written so they stay in
        // `usize` at `i == 1`.
        let cur = i % 3;
        let prev = (i + 2) % 3;
        let prev2 = (i + 1) % 3;
        let (cur0, prev0, prev20) = (cur * width, prev * width, prev2 * width);

        // The row being (re)used is two iterations old; every cell it still
        // holds is stale, so clear it before the band is written into it.
        buf[cur0..cur0 + width].fill(unreachable);
        // `dp[i][0] = i`, again only inside the budget.
        if i <= max {
            buf[cur0] = i;
        }

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
            let mut best = buf[prev0 + j]
                .min(buf[cur0 + j - 1])
                .saturating_add(1)
                .min(buf[prev0 + j - 1].saturating_add(cost));
            if transpositions && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1] {
                best = best.min(buf[prev20 + j - 2].saturating_add(1));
            }
            buf[cur0 + j] = best;
        }
    }

    let d = buf[(n % 3) * width + m];
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
        self.edits_within(candidate, self.max_edits)
    }

    /// [`Self::edits`] against a **tighter budget than this pattern's own**.
    ///
    /// This is `FuzzyTermsEnum`'s `automata[k]` for `k < maxEdits`: the same
    /// target term, a smaller edit distance, a smaller (cheaper) machine.
    /// Lucene builds the whole ladder up front (`buildAutomatonSet` returns
    /// `automata[0..=maxEdits]`) and swaps between them -- `getAutomatonEnum(k)`
    /// to prune the enumeration once the top-terms queue is full
    /// (`bottomChanged`), and `matches(term, ed - 1)` to walk *down* to the
    /// exact distance of a term it has already accepted. Here the ladder is one
    /// parameter, because the machine is a banded DP whose band is that
    /// parameter.
    ///
    /// A budget above `max_edits` is not clamped: nothing in this port asks for
    /// one, and clamping would silently answer a different question.
    pub fn edits_within(&self, candidate: &[u8], max_edits: u8) -> Option<usize> {
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
            max_edits as usize,
        )
    }

    /// This pattern's own edit budget -- `FuzzyQuery`'s `maxEdits`, the top of
    /// [`Self::edits_within`]'s ladder.
    pub fn max_edits(&self) -> u8 {
        self.max_edits
    }

    /// Tests whether `candidate` matches: it must start with this pattern's
    /// fixed prefix exactly, and the two suffixes' edit distance must be
    /// `<= max_edits`.
    pub fn matches(&self, candidate: &[u8]) -> bool {
        self.edits(candidate).is_some()
    }

    /// [`Self::matches`] against a tighter budget -- see [`Self::edits_within`].
    pub fn matches_within(&self, candidate: &[u8], max_edits: u8) -> bool {
        self.edits_within(candidate, max_edits).is_some()
    }

    /// The codepoint count of `bytes` -- `UnicodeUtil.codePointCount` -- without
    /// decoding it into a `Vec<char>`.
    ///
    /// For well-formed UTF-8 a codepoint is exactly one non-continuation byte,
    /// so this is a byte scan. Ill-formed input falls back to the same
    /// `U+FFFD`-substituting decode [`to_chars`] does, so the two agree
    /// everywhere.
    fn codepoint_count(bytes: &[u8]) -> usize {
        if std::str::from_utf8(bytes).is_ok() {
            bytes.iter().filter(|&&b| (b & 0xC0) != 0x80).count()
        } else {
            to_chars(bytes).len()
        }
    }

    /// [`Self::boost`] for a candidate whose edit distance is **already known**
    /// -- the value `FuzzyTermsEnum.next` has in hand from its own
    /// `while (ed > 0) { if (matches(term, ed - 1)) ed--; ... }` walk by the
    /// time it sets `boostAtt`. Recomputing the distance to score a term the
    /// matcher has just accepted is one banded DP and one `Vec<char>` per
    /// accepted term for an answer already computed.
    pub fn boost_from_edits(&self, candidate: &[u8], ed: usize) -> f32 {
        if ed == 0 {
            return 1.0;
        }
        let min_term_length = Self::codepoint_count(candidate).min(self.term_chars.len());
        if min_term_length == 0 {
            return 1.0;
        }
        1.0 - (ed as f32) / (min_term_length as f32)
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
        let candidate_len = Self::codepoint_count(candidate);
        let min_term_length = candidate_len.min(self.term_chars.len());
        if min_term_length == 0 {
            // `min(codePointCount(candidate), termLength)`, so this is a
            // zero-length *query* term -- an empty `FuzzyQuery`, which nothing
            // rejects -- against a candidate at distance `ed > 0`, i.e. any
            // non-empty term at all within the budget.
            //
            // **This is a deliberate divergence, not a can't-happen.** Java
            // computes `1.0f - (float) ed / 0.0f`, which is `-Infinity`, and
            // returns that as the boost; `TopTermsRewrite.build` then truncates
            // it to `0`. Returning `1.0` here instead makes every candidate of
            // an empty query term tie at the top rather than at the bottom,
            // which changes *which* `maxExpansions` terms are selected (the
            // lexicographically first, rather than an arbitrary set of ties at
            // zero) but not the fact that they all score zero once truncated.
            // `FuzzyTermsEnum.bottomChanged`'s own copy of this arithmetic
            // (`lucene-search`'s `fuzzy_expanded_terms_pruned`) does *not*
            // special-case it and yields Java's `-inf`, so the two disagree
            // about this one degenerate input; pruning stays result-preserving
            // there because all boosts tie and later terms are lexicographically
            // greater. Left as-is rather than "fixed" to `-inf`, because a
            // division by zero in a similarity is a worse thing to introduce
            // than a documented tie, and no caller of this port has an empty
            // fuzzy term.
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

    /// The straightforward full-matrix Damerau-Levenshtein this module used to
    /// compute, kept as a **reference** for the rolling-row rewrite: three
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
                if transpositions && i > 1 && j > 1 && a[i - 1] == b[j - 2] && a[i - 2] == b[j - 1]
                {
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
