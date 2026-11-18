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
//! instead. Same accept/reject decision, different machine.
//!
//! The **semantics** are Lucene's exactly, and each of these is a rule that is easy
//! to get wrong:
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
/// plain Levenshtein's. A `Vec<Vec<usize>>` would cost **`n + 2` heap
/// allocations per candidate term**, and a fuzzy expansion tests every term in
/// the dictionary's prefix range. Java allocates nothing here (it runs a
/// compiled DFA over the term's bytes), so one flat buffer is the closest
/// match.
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
    distance_chars_in(a, b, transpositions, max, &mut Vec::new())
}

/// [`distance_chars`] with its three DP rows in `buf`, reused across calls.
// ARITH: the DP itself; see [`distance_chars`] for every bound.
#[allow(clippy::arithmetic_side_effects)]
fn distance_chars_in(
    a: &[char],
    b: &[char],
    transpositions: bool,
    max: usize,
    buf: &mut Vec<usize>,
) -> Option<usize> {
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
    buf.clear();
    buf.resize(3 * width, unreachable);

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

type DfaKey = (Vec<u8>, u8, usize, bool);

/// Compiled Levenshtein automata of the terms recently fuzzy-matched, shared
/// by every query in the process. Lucene rebuilds the automaton for every
/// `FuzzyQuery`, and that build (the NFA, `UTF32ToUtf8`, the dense table) is
/// most of what a short fuzzy query costs; the terms a workload fuzzes
/// repeat, so a hit skips it. Bounded by the bytes its automata hold: a full
/// cache is emptied, which suits a working set that fits and costs one
/// rebuild per term when it does not.
struct SharedDfas {
    map: std::sync::Mutex<SharedDfaMap>,
}

/// A [`SharedDfas`]' automata, and the bytes they hold.
#[derive(Default)]
struct SharedDfaMap {
    dfas: std::collections::HashMap<DfaKey, SharedDfa>,
    bytes: usize,
}

/// A term's compiled automaton, `None` for a pattern that has none.
type SharedDfa = Option<std::sync::Arc<crate::automaton::CompiledDfa>>;

/// What [`SharedDfas`] may hold before it is emptied.
const SHARED_DFA_BUDGET_BYTES: usize = 16 << 20;

fn shared_dfas() -> &'static SharedDfas {
    static CACHE: std::sync::OnceLock<SharedDfas> = std::sync::OnceLock::new();
    CACHE.get_or_init(|| SharedDfas {
        map: std::sync::Mutex::new(SharedDfaMap::default()),
    })
}

impl SharedDfas {
    fn get(&self, key: &DfaKey) -> Option<SharedDfa> {
        let guard = self.map.lock().unwrap_or_else(|e| e.into_inner());
        guard.dfas.get(key).cloned()
    }

    fn put(&self, key: DfaKey, dfa: SharedDfa) {
        let weight = key
            .0
            .len()
            .saturating_add(dfa.as_ref().map_or(0, |d| d.dfa.heap_bytes()));
        let mut guard = self.map.lock().unwrap_or_else(|e| e.into_inner());
        if guard.bytes.saturating_add(weight) > SHARED_DFA_BUDGET_BYTES {
            guard.dfas.clear();
            guard.bytes = 0;
        }
        if guard.dfas.insert(key, dfa).is_none() {
            guard.bytes = guard.bytes.saturating_add(weight);
        }
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
    term: std::borrow::Cow<'a, [u8]>,
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
    /// [`Self::to_dfa`] at `max_edits`, built on first use and shared by every
    /// segment a query walks -- determinizing the Levenshtein NFA once per
    /// query rather than once per leaf, as Lucene builds its automata once in
    /// `FuzzyQuery.getAutomata`.
    dfa: std::sync::OnceLock<Option<std::sync::Arc<crate::automaton::ByteDfa>>>,
    /// The same DFA as a [`crate::automaton::CompiledDfa`], for the
    /// block-tree intersection ([`crate::blocktree::FieldTerms::fuzzy_intersect`]).
    compiled: std::sync::OnceLock<Option<std::sync::Arc<crate::automaton::CompiledDfa>>>,
    /// The ladder below `max_edits` (`FuzzyQuery.getAutomata`'s
    /// `automata[0..maxEdits]`), each built the first time a walk lowers its
    /// budget to it -- see [`Self::cached_compiled_dfa_within`].
    lower: [std::sync::OnceLock<Option<std::sync::Arc<crate::automaton::CompiledDfa>>>; 2],
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
            term: std::borrow::Cow::Borrowed(term),
            term_chars,
            prefix_chars,
            prefix_bytes,
            max_edits,
            transpositions,
            dfa: std::sync::OnceLock::new(),
            compiled: std::sync::OnceLock::new(),
            lower: [std::sync::OnceLock::new(), std::sync::OnceLock::new()],
        }
    }

    /// This pattern over its own copy of the term, for a holder that
    /// outlives the term's owner; what it has cached is kept.
    pub fn into_owned(self) -> FuzzyMatch<'static> {
        FuzzyMatch {
            term: std::borrow::Cow::Owned(self.term.into_owned()),
            term_chars: self.term_chars,
            prefix_chars: self.prefix_chars,
            prefix_bytes: self.prefix_bytes,
            max_edits: self.max_edits,
            transpositions: self.transpositions,
            dfa: self.dfa,
            compiled: self.compiled,
            lower: self.lower,
        }
    }

    /// [`Self::to_dfa`] at this pattern's own `max_edits`, cached -- see
    /// the `dfa` field.
    pub(crate) fn cached_dfa(&self) -> Option<std::sync::Arc<crate::automaton::ByteDfa>> {
        self.dfa
            .get_or_init(|| self.to_dfa(self.max_edits).map(std::sync::Arc::new))
            .clone()
    }

    /// [`Self::cached_dfa`] as the automaton `Terms.intersect` walks, cached
    /// alongside it.
    pub(crate) fn cached_compiled_dfa(
        &self,
    ) -> Option<std::sync::Arc<crate::automaton::CompiledDfa>> {
        self.compiled
            .get_or_init(|| {
                let key = (
                    self.term.to_vec(),
                    self.max_edits,
                    self.prefix_chars,
                    self.transpositions,
                );
                if let Some(hit) = shared_dfas().get(&key) {
                    return hit;
                }
                let built = self.cached_dfa().map(|dfa| {
                    std::sync::Arc::new(crate::automaton::CompiledDfa::finite((*dfa).clone()))
                });
                shared_dfas().put(key, built.clone());
                built
            })
            .clone()
    }

    /// [`Self::cached_compiled_dfa`] for a budget of `max_edits`: the
    /// pattern's own automaton at or above its `max_edits`, else the smaller
    /// one `FuzzyTermsEnum.bottomChanged` swaps in (`automata[maxEdits]`),
    /// built once. `None` when no automaton can be built for it.
    pub(crate) fn cached_compiled_dfa_within(
        &self,
        max_edits: u8,
    ) -> Option<std::sync::Arc<crate::automaton::CompiledDfa>> {
        if max_edits >= self.max_edits {
            return self.cached_compiled_dfa();
        }
        self.lower
            .get(usize::from(max_edits))?
            .get_or_init(|| {
                self.to_dfa(max_edits)
                    .map(|dfa| std::sync::Arc::new(crate::automaton::CompiledDfa::finite(dfa)))
            })
            .clone()
    }

    /// The target term's fixed, non-fuzzy prefix as bytes -- the first
    /// `prefix_length` **codepoints**, clamped to the term's own length.
    /// Every matching candidate must start with exactly this run, since real
    /// `FuzzyQuery` holds those codepoints outside the automaton entirely
    /// rather than spending edits on them. Used by
    /// [`crate::blocktree::FieldTerms`] to narrow its scan to a contiguous
    /// sorted range via binary search first, the same literal-prefix-range
    /// trick `wildcard.rs`'s `literal_prefix` already uses.
    pub fn literal_prefix(&self) -> &[u8] {
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
    ///
    /// A candidate that is not valid UTF-8 is out of range, whatever its
    /// lossy decoding is within reach of: Lucene matches through a UTF-8
    /// automaton, which never accepts one (and neither does [`Self::to_dfa`]).
    pub fn edits_within(&self, candidate: &[u8], max_edits: u8) -> Option<usize> {
        if candidate.len() < self.prefix_bytes
            || candidate[..self.prefix_bytes] != self.term[..self.prefix_bytes]
        {
            return None;
        }
        let Ok(candidate) = std::str::from_utf8(candidate) else {
            return None;
        };
        // Called once per term a fuzzy walk tests: the decoded candidate and
        // the DP rows live in per-thread scratch rather than fresh vectors.
        thread_local! {
            static SCRATCH: std::cell::RefCell<(Vec<char>, Vec<usize>)> =
                const { std::cell::RefCell::new((Vec::new(), Vec::new())) };
        }
        SCRATCH.with(|scratch| {
            let (chars, rows) = &mut *scratch.borrow_mut();
            chars.clear();
            chars.extend(candidate.chars());
            if chars.len() < self.prefix_chars {
                return None;
            }
            distance_chars_in(
                &self.term_chars[self.prefix_chars..],
                &chars[self.prefix_chars..],
                self.transpositions,
                max_edits as usize,
                rows,
            )
        })
    }

    /// This pattern's own edit budget -- `FuzzyQuery`'s `maxEdits`, the top of
    /// [`Self::edits_within`]'s ladder.
    pub fn max_edits(&self) -> u8 {
        self.max_edits
    }

    /// A byte DFA accepting the terms within `max_edits` of this pattern --
    /// `FuzzyAutomatonBuilder`'s automaton: `LevenshteinAutomata` over the
    /// code points past `prefix_length` (Lucene's parametric
    /// `Lev1`/`Lev1T`/`Lev2`/`Lev2T` descriptions, so construction is linear
    /// in the term), the exact prefix in front (`toAutomaton(n, prefix)`),
    /// then `UTF32ToUTF8` and `CompiledAutomaton`'s determinization, which
    /// has nothing to do unless a non-ASCII pattern's edges come to share a
    /// UTF-8 lead byte.
    ///
    /// The automaton reads UTF-8, so a term that is not valid UTF-8 is
    /// rejected, as Lucene's is, and as [`Self::edits_within`] rejects it. A
    /// pattern that is not valid UTF-8 itself (its lossy decoding holds
    /// U+FFFD) gets no automaton, and the walk falls back to
    /// [`Self::edits_within`] alone. `None` also when the automaton would be
    /// too large.
    pub(crate) fn to_dfa(&self, max_edits: u8) -> Option<crate::automaton::ByteDfa> {
        use lucene_util::automaton::{LevenshteinAutomata, Utf32ToUtf8};
        if self.term_chars.contains(&char::REPLACEMENT_CHARACTER) {
            return None;
        }
        let prefix: String = self.term_chars[..self.prefix_chars].iter().collect();
        let suffix: String = self.term_chars[self.prefix_chars..].iter().collect();
        let lev = LevenshteinAutomata::new(&suffix, self.transpositions);
        let automaton = lev.to_automaton_with_prefix(i32::from(max_edits), &prefix)?;
        crate::automaton::from_byte_automaton(&Utf32ToUtf8::new().convert(&automaton))
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
mod tests;
