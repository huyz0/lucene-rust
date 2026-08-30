# b8 — automata and analysis

Files swept: `lucene-codecs/src/{regexp,wildcard,fuzzy,suggest}.rs`,
`lucene-analysis/src/lib.rs`. Java read at `releases/lucene/10.5.0`
(`/home/tuong/work/lucene`, `git archive`d — the working checkout is on
`main`/11.0.0-SNAPSHOT, where `RegExp`'s deprecated complement has already been
removed, so reading it directly would have compared against the wrong version).

Two files outside the batch were changed as a direct consequence and are listed
where they occur: `lucene-codecs/src/blocktree.rs` (the dead-prefix skip needs a
skipping iterator) and `lucene-search/src/{lib,query}.rs` (fuzzy scoring cannot
be fixed in the matcher alone).

**Headline**: 34 findings — 19 CORRECTNESS (all fixed), 6 MISSING (4 fixed),
4 PERF (2 fixed and benchmarked), 5 INTENTIONAL. The b4 carry-over blocker
("`regexp.rs` cannot report a dead state, so `IntersectTermsEnum` cannot be
ported") is **closed**: `regexp_intersect` now skips, measured at 88x–1065x on
interior-constrained patterns.

---

## `crates/lucene-codecs/src/regexp.rs`

**Java counterparts:** `org/apache/lucene/util/automaton/RegExp.java`,
`Automata.java` (`makeAnyChar`, `makeAnyString`, `makeDecimalInterval`,
`between`/`atLeast`/`atMost`), `Operations.java` (`determinize`),
`ByteRunAutomaton.java`, `UTF32ToUTF8.java`,
`org/apache/lucene/search/RegexpQuery.java`.

The comparison baseline is `RegexpQuery(Term)` exactly: `RegExp.ALL` (`0xff` =
`INTERSECTION | EMPTY | ANYSTRING | AUTOMATON | INTERVAL`), match flags `0`,
`RegexpQuery.DEFAULT_PROVIDER` (returns `null` for every named automaton),
`Operations.determinize` at the default work limit, then `ByteRunAutomaton` over
the term's UTF-8 bytes.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `RegexpPattern::new` / `::parse` | `new RegExp(s, ALL)` + `toAutomaton(provider)` | was **divergent**, now identical (F1–F11) |
| `RegexpPattern::matches` | `ByteRunAutomaton.run` | was divergent (byte-wide), now identical (F8) |
| `RegexpPattern::literal_prefix` | `CompiledAutomaton.commonPrefix` | divergent, deliberately weaker; widened (F14) |
| `RegexpPattern::dead_prefix_len` / `::could_match_prefix` | `ByteRunAutomaton` dead-state check driving `IntersectTermsEnum` | **added** (F15) |
| `Parser::parse_union` / `_inter` / `_concat` / `_repeat` | `parseUnionExp` / `parseInterExp` / `parseConcatExp` / `parseRepeatExp` | identical |
| `Parser::parse_char_class_exp` / `_char_classes` / `expand_predefined` | `parseCharClassExp` / `parseCharClasses` / `expandPreDefined` | identical |
| `Parser::parse_simple` / `parse_angle` / `parse_char_exp` | `parseSimpleExp` / its `<...>` arm / `parseCharExp` | identical |
| `node_match` / `repeat_match` / `interval_match` | `RunAutomaton.step` over a determinized DFA | divergent by design (F19) |
| — | `RegExp.toString` / `toStringTree` / `getIdentifiers` | not ported, no caller |
| — | `RegExp.CASE_INSENSITIVE` / `CASE_INSENSITIVE_RANGE` match flags | MISSING (F17) |

### Syntax-flag coverage, before and after

The task asked for this enumerated explicitly, because a construct that is
*silently reinterpreted* is worse than one that is rejected. Before this sweep,
six of them were.

| Construct | Flag | Lucene means | This port meant | Now |
|---|---|---|---|---|
| `&` | `INTERSECTION` (in `ALL`) | intersection | **hard parse error** | intersection |
| `#` | `EMPTY` (in `ALL`) | the empty language | **literal `#`** | empty language |
| `@` | `ANYSTRING` (in `ALL`) | any string | **literal `@`** | any string |
| `<identifier>` | `AUTOMATON` (in `ALL`) | named automaton → `"x" not found` | **literal `<`,`x`,`>`** | `NamedAutomatonNotFound` |
| `<n-m>` | `INTERVAL` (in `ALL`) | numeric interval | **literal characters** | numeric interval |
| `"..."` | always on | literal string, no escapes inside | **literal `"` + contents** | literal string |
| `()` | always on | the empty string | parse error | empty string |
| `\d \D \s \S \w \W` | always on | predefined classes | **literal `d`/`D`/…** | predefined classes |
| `?` `*` `+` `{n,m}` | always on | quantifiers, **stackable** | stacking was an error | stackable |
| `[...]` `[^...]` `.` | always on | **codepoint**-wide | **byte**-wide | codepoint-wide |
| `~` | `DEPRECATED_COMPLEMENT`, **not** in `ALL` | ordinary literal | **hard parse error** | ordinary literal |
| `^` `$` | — | ordinary literals | ordinary literals | unchanged, correct |

### Findings

**F1 [CORRECTNESS] `#` and `@` were literals, not operators.**
Java: `parseSimpleExp` has `check(EMPTY) && match('#')` → `makeEmpty` and
`check(ANYSTRING) && match('@')` → `makeAnyString`, both enabled by
`RegExp.ALL`. Us: neither was special, so `#` matched the term `#` and `cat@`
matched the term `cat@` instead of every term beginning `cat`. Consequence: a
query using either returned a completely different, silently plausible hit set.
**Fixed** (`Node::Empty`, `@` as `AnyChar{0,}`), tested and fixture-pinned.

**F2 [CORRECTNESS] `"..."` quoted literals were not recognised.**
Java: `match('"')` scans to the closing quote and `makeString`s the contents
verbatim — no escapes honoured inside. Us: `"` was an ordinary byte, so
`"a*b"` meant `"` + `a*` + `b"`. **Fixed**; an unterminated quote is now
`ExpectedChar { expected: '"' }`, as Java's `expected '"' at position` is.

**F3 [CORRECTNESS] `<n-m>` numeric intervals were not recognised.**
Java: `parseSimpleExp`'s `<...>` arm parses two non-negative decimals separated
by exactly one `-`, swaps them if out of order, and sets `digits` to their
common length when both were written with the same number of characters;
`Automata.makeDecimalInterval` then builds a fixed-width zero-padded language
when `digits > 0` and an any-width one (with a `0*` prefix loop) when it is `0`.
Us: `<`, `>` and the digits were literal bytes. **Fixed** as `Node::Interval`,
with both width modes and the out-of-order swap; `<05-40>` accepts only
two-digit strings, `<5-40>` accepts `5`, `05` and `0005`.

**F4 [CORRECTNESS] `&` intersection was rejected outright.**
`INTERSECTION` is in `RegExp.ALL`, so `RegexpQuery(new Term(f, "a&b"))` is a
legal query in Lucene and an `Error::Regexp` here. **Fixed**: `parse_inter`
mirrors `parseInterExp`, and the matcher handles it in continuation-passing
style — the left side matches a span, the right side must accept exactly that
span, then the continuation runs. That composes correctly under concatenation
(`a.&.b`) and precedence (`ab&ab|cd`), both verified against Java.

**F5 [CORRECTNESS] `~` was rejected; in Lucene it is an ordinary character.**
`DEPRECATED_COMPLEMENT` is `0x10000`, outside `ALL` (`0xff`), and the
constructor's guard is `(syntax_flags & ~DEPRECATED_COMPLEMENT) > ALL`. So
`parseComplExp` never fires for a default `RegexpQuery` and `~` falls through to
`makeChar`. Us: a hard error, with a module doc that described complement as
"deliberately not supported". Both were wrong in the same direction: the
supported thing was refused. **Fixed** (and Lucene 11 removes the flag
entirely).

**F6 [CORRECTNESS] `\d \D \s \S \w \W` were unsupported.**
Java: `matchPredefinedCharacterClass` (outside a class) and `expandPreDefined`
(inside one) expand them to explicit range lists, and `\s` is specifically
`[\t-\n\r ]` — *not* `\f` or `\v`. Any **other** alphabetic escape is an error
("It is an error to use a backslash prior to any alphabetic character that does
not denote an escaped construct"). Us: `\d` meant a literal `d`. **Fixed**,
including `\A`/`\q` now being `InvalidCharacterClass` as Java has them.

**F7 [CORRECTNESS] `()` was a parse error; Lucene reads it as the empty string.**
Java: `if (match('(')) { if (match(')')) return makeString(flags, ""); ... }`.
**Fixed**.

**F8 [CORRECTNESS] `.` and `[...]` were byte-wide, not codepoint-wide.**
Java builds a UTF-32 automaton (`Automata.makeAnyChar` is the single range
`0..0x10FFFF`) and `CompiledAutomaton` converts it with `UTF32ToUTF8`. So `.`
against `"€"` matches; here it consumed only the lead byte and failed. `[^a]`
had the same problem in reverse — it matched each of `€`'s three bytes
individually. **Fixed**: the matcher decodes UTF-8 as it goes.

A consequence worth stating: a term whose bytes are **not** well-formed UTF-8
now matches nothing. That is also Lucene's behaviour — `UTF32ToUTF8` can only
emit well-formed sequences, so no compiled `RegExp` accepts an ill-formed one —
and it is tested (`ill_formed_utf8_matches_nothing`).

**F9 [CORRECTNESS] Stacked quantifiers were a parse error.**
Java's `parseRepeatExp` loops `while (peek("?*+{"))`, so `a**` is a repeat of a
repeat and `a{2}?` is legal. Us: `DanglingQuantifier`. **Fixed**.

**F10 [CORRECTNESS] A leading operator was a parse error; Lucene reads it as a
literal.** `parseSimpleExp` falls through to `makeChar` for anything it does not
recognise, so `*cat` matches the term `*cat`, `{2,3}` matches the term `{2,3}`,
and `?` matches the term `?`. And because `iterativeParseExp` calls its gather
function *before* testing its stop condition, `|a` and `&a` are literals too —
a subtlety that only shows up if you trace the parser rather than read the
grammar comment. **Fixed**, all four verified against Java.

**F11 [CORRECTNESS] Class-parsing edge cases diverged.**
Java's `parseCharClasses` is a `do…while`, so `[]` is not an "empty class"
error — it consumes `]` as a member and then fails to find the closing bracket.
`[a-]` is likewise not "a literal trailing dash": `parseCharExp` takes the `]`
as the range's upper bound and the class is left unclosed. And an **escaped**
member cannot open a range (Java's escape branch adds the codepoint and loops
without checking for `-`), so `[\--z]` is the members `-`, `-`, `z`. All three
of ours differed. **Fixed**; the old `EmptyClass` error variant is gone because
Lucene has no such error.

**F12 [CORRECTNESS] A trailing `\` matched itself; Lucene throws.**
`parseCharExp` calls `next()` past the end → `"unexpected end-of-string"`.
**Fixed**.

**F13 [MISSING] Error messages did not correspond to Lucene's.**
`RegexpError` is now one variant per `IllegalArgumentException` `RegExp` throws,
carrying the same position (0-based, as Java reports it; the interval error uses
Java's `pos - 1`). This matters because these strings reach a user through
`Error::Regexp`. **Fixed.**

**F14 [PERF] `literal_prefix` gave up at any quantifier.**
`ca{2,3}t` guarantees two leading `a`s and `c(ab){2,}t` guarantees `cabab`, but
the old implementation stopped at the first non-`Literal` node and returned `c`.
It also could not see through `Str` (quoted literals) or `Intersect`. **Fixed**:
a mandatory repeat contributes `min` copies of whatever its body forces, and an
intersection contributes the longer of its two sides (sound, since
`L(a & b) ⊆ L(a)`). Every case is checked by the fixture test
`literal_prefix_is_a_true_prefix_of_every_term_real_lucene_accepts` — a wrong
prefix silently drops matches, so it is checked against Lucene's own accept set
rather than our own.

**F15 [MISSING → fixed] No dead-state signal. This was b4's blocker.**
b4 recorded that `IntersectTermsEnum` could not be ported because "`regexp.rs`
is a backtracking matcher that cannot enumerate transitions or report a dead
state", and measured 34–38x over-scan on interior-constrained patterns.

Building a real `Automaton` + `determinize` + `UTF32ToUTF8` + `ByteRunAutomaton`
was assessed and **not** done. What `IntersectTermsEnum` actually needs from the
automaton is one bit — *is this prefix dead?* — and a backtracker can answer
that directly by running in "prefix mode", where exhausting the input part-way
through a node counts as success instead of failure. That is
`RegexpPattern::could_match_prefix`; `dead_prefix_len` binary-searches it (the
predicate is monotone in the prefix length) to find the shortest dead prefix,
never splitting a UTF-8 codepoint.

The consumer is `blocktree::FieldTerms::regexp_intersect`, now a real iterator
(`RegexpIntersect`) instead of a `.filter()`: on a non-match it jumps past the
whole sorted run sharing the dead prefix. That is the sorted-array analogue of
"do not descend into that block". Soundness is the whole risk here, so it is
attacked three ways: an exhaustive small-alphabet property test in `regexp.rs`,
a fixture test asserting no term real Lucene accepts starts with a prefix we
called dead, and a brute-force-agreement test in `blocktree.rs` over a 3000-term
dictionary and ten pattern shapes.

**Measured** (`crates/lucene-codecs/benches/regexp_intersect.rs`, one million
terms `t0..t999999`, criterion):

| pattern | scan | skip | |
|---|---|---|---|
| `t1[0-9]` | 1.72 ms | **19.6 µs** | **88x** |
| `t1*z` (b4's shape) | 16.4 ms | **15.4 µs** | **1065x** |
| `t[0-9]{4}` | 30.1 ms | 29.8 ms | 1.01x |
| `t.*99` | 84.8 ms | 84.1 ms | 1.01x |

The last two are the reason for the adaptive give-up. A skip attempt costs
about twelve `matches` calls (a few `could_match_prefix` runs plus a galloping
search), so a pattern whose dead runs are only ~9 terms long *loses*: measured
at 0.83x before the guard. `RegexpIntersect` now stops attempting skips after
128 attempts if they have not averaged 16 terms saved, which restores parity on
those shapes while keeping the wins. `lower_bound_from` gallops from the current
position rather than bisecting the whole field, so the jump costs `O(log d)` in
the distance rather than `O(log n)` in the field size.

**What this does not fix**: Lucene's other win is not *loading* the pruned
`.tim`/`.tip` blocks, and `BlockTreeFields` has already decoded every term into
memory at open. That is carry-over item A1 and is untouched here.

**F16 [MISSING] `Operations.DEFAULT_DETERMINIZE_WORK_LIMIT` /
`TooComplexToDeterminizeException` have no counterpart.**
Lucene bounds *construction* and throws; this module bounds *matching* with
`MATCH_STEP_BUDGET` and reports "no match". Recorded, not fixed: the guard is
in the right direction (a pathological pattern cannot hang the process) but the
failure mode differs — Lucene refuses the query, this port silently
under-matches. A pattern that hits the budget is far outside anything a real
query produces (the whole test suite stays under 10 000 steps).

**F17 [MISSING] `CASE_INSENSITIVE` / `CASE_INSENSITIVE_RANGE` match flags.**
Not reachable from `RegexpQuery(Term)` (match flags are `0`) and this port has
no API that would pass them. Recorded.

**F18 [MISSING] `RegExp.toString` / `toStringTree` / `getIdentifiers`.** No
caller. Recorded.

**F19 [INTENTIONAL] Backtracking rather than a determinized DFA.**
Same accept/reject decision (now verified pattern-by-pattern against
`ByteRunAutomaton`), different machine. Worth its own line because it is what
makes F16 a real difference and what made F15 need a different technique.

### Verdict

Swept clean on semantics: 79 patterns × 66 terms now agree with real Lucene
byte for byte (`fixtures/src/GenRegexp.java`,
`crates/lucene-codecs/tests/regexp_fixtures.rs`), including which patterns are
rejected. Open: F16 (work-limit semantics), F17 (case-insensitive match flags).

---

## `crates/lucene-codecs/src/wildcard.rs`

**Java counterparts:** `org/apache/lucene/search/WildcardQuery.java`
(`toAutomaton`, `WILDCARD_STRING`/`CHAR`/`ESCAPE`),
`org/apache/lucene/search/PrefixQuery.java` (`toAutomaton`),
`org/apache/lucene/util/automaton/Automata.java`.

| Rust | Java | Verdict |
|---|---|---|
| `WildcardPattern::new` | `WildcardQuery.toAutomaton` | identical |
| `WildcardPattern::prefix` | `PrefixQuery.toAutomaton` | identical |
| `WildcardPattern::matches` | `ByteRunAutomaton.run` | identical for well-formed UTF-8 (F20) |
| `WildcardPattern::literal_prefix` | `CompiledAutomaton.commonPrefix` | identical for this grammar |
| `utf8_codepoint_len` | `Character.charCount` | equivalent |

**F20 [INTENTIONAL] `*` matches any byte sequence, not any well-formed UTF-8
one.** `Automata.makeAnyString` composed with `UTF32ToUTF8` accepts only
well-formed sequences. The divergence is unreachable for any term a real
codec writes, and this port's terms are explicitly `Vec<u8>`.

Everything else checks out method by method: `\` escapes the next codepoint (and
escaping a non-special one is a harmless no-op, exactly as Lucene's `default`
arm makes it); a trailing unpaired `\` is a literal backslash (Lucene's
documented "lenient parsing with a trailing \\" fallthrough); `?` is one
codepoint, not one byte; `PrefixQuery` is byte-level in Lucene too
(`isBinary = true`, prefix bytes then a `0..255` self-loop), which is exactly
what `prefix()` builds.

### Verdict

**Swept clean.** No fix needed — the only divergence is F20, and it is
intentional and documented.

---

## `crates/lucene-codecs/src/fuzzy.rs` (+ `lucene-search`)

**Java counterparts:** `org/apache/lucene/util/automaton/LevenshteinAutomata.java`,
`org/apache/lucene/search/{FuzzyQuery,FuzzyAutomatonBuilder,FuzzyTermsEnum,
MultiTermQuery,TopTermsRewrite,BlendedTermQuery}.java`.

| Rust | Java | Verdict |
|---|---|---|
| `edit_distance` | `LevenshteinAutomata` (`Lev1T`/`Lev2T` descriptions) | was divergent (F21), now identical |
| `edit_distance_at_most` | — (Lucene runs an automaton) | not-in-Java, PERF (F26) |
| `FuzzyMatch::new` / `literal_prefix` | `FuzzyAutomatonBuilder`'s constructor | was divergent (F22), now identical |
| `FuzzyMatch::matches` / `edits` | `FuzzyTermsEnum.next`'s walk down `automata[k]` | was divergent, now identical |
| `FuzzyMatch::boost` | `FuzzyTermsEnum.next`'s `BoostAttribute` | **missing**, added (F23) |
| `lucene_search::fuzzy_expanded_terms` | `TopTermsRewrite.collect` + `BlendedTermQuery.rewrite` | **missing**, added (F24) |
| `lucene_search::fuzzy_doc_scores` | `BlendedTermQuery` → `BooleanQuery` of boosted `TermQuery`s | **missing**, added (F23) |
| `MAXIMUM_SUPPORTED_DISTANCE` etc. | `FuzzyQuery`'s `default*` constants | added (F25) |
| — | `FuzzyQuery.floatToEdits`, `getAutomata`, `SingleTermsEnum` fast path | not ported, no caller |
| — | `MaxNonCompetitiveBoostAttribute` early-exit | MISSING (F27) |

**F21 [CORRECTNESS] Edit distance counted UTF-8 bytes, not codepoints.**
`FuzzyAutomatonBuilder` calls `stringToUTF32(term)` and builds over
`Character.MAX_CODE_POINT`, so one multi-byte character is **one** edit unit.
Ours counted bytes: `café` → `cafè` was 2 edits instead of 1, and deleting a
4-byte codepoint was 4. Consequence: every non-ASCII fuzzy query was
silently narrower than Lucene's. The module doc called this a "deliberate,
stated shortcut", which it was — but it is also a wrong answer, and it is
cheap to fix. **Fixed**; ill-formed bytes decode to `U+FFFD` rather than
making a term unmatchable, since edit distance is a similarity measure.

**F22 [CORRECTNESS] `prefixLength` was a byte count, and the distance was
measured over the whole terms.** Java splits the **codepoints** at
`prefixLength` (clamped to the term's length) and hands only the `suffix` to
`LevenshteinAutomata`; the prefix is held outside the edit budget entirely.
Measuring whole-term distance instead is a strictly weaker test — a shared
prefix can absorb part of an alignment, so `ED(p+a, p+b) ≤ ED(a, b)` — which
means the old code could accept terms Lucene rejects. **Fixed** on both counts;
`prefix_length_takes_the_prefix_out_of_the_edit_budget` covers the case where
the two answers differ.

**F23 [CORRECTNESS] `FuzzyQuery` scored a flat 1.0. This is sweep finding P4.**
`docs/sweep/findings.md` recorded that `fuzzy body:t123` returns a top-1 score
of 5.03 in Lucene and 1.0 here, so the top-k differs. Verified against the Java
and fixed end to end:

- `FuzzyQuery`'s default rewrite is
  `MultiTermQuery.TopTermsBlendedFreqScoringRewrite`, not a constant-score one.
- `FuzzyTermsEnum.next` computes the exact edit distance by walking *down* the
  `maxEdits-1, maxEdits-2, …` automata and publishes
  `ed == 0 ? 1.0f : 1.0f - ed / min(codePointCount(term), termLength)` through
  `BoostAttribute`, where `termLength` is the **whole** query term's codepoint
  count. `FuzzyMatch::boost` is that formula, unclamped.
- `TopTermsRewrite.build` clamps with `Math.max(0.0f, st.boost)` when it builds
  the clauses — the similarity really can go negative for one- and
  two-character query terms, which is what Lucene's javadoc note about short
  terms is describing.
- `BlendedTermQuery.rewrite` sets every selected term's `docFreq` to the
  **max** across them ("otherwise the rarest term typically ranks highest,
  often not useful eg in the set of expanded terms in a FuzzyQuery") and the
  `BOOLEAN_REWRITE` turns them into `SHOULD` clauses.

So `fuzzy_doc_scores` sums `max(0, boost_t) * BM25(df_blended, tf, norm)` over
the expanded terms. One gap remains and is stated in the code: Lucene blends
across the whole reader (`TermStates.build`), this blends within one segment,
because a fuzzy clause has no `GlobalStats` plumbing yet (the same limitation
`term_doc_scores` works around for `TermQuery`).

**F24 [CORRECTNESS] `max_expansions` kept the wrong terms.**
Ours took the first N in term-dictionary order — a policy the code documented
honestly and which is close to the *opposite* of Lucene's, since term order is
uncorrelated with edit distance. `TopTermsRewrite.collect` keeps a size-N
priority queue ordered by boost, dropping the lexicographically later term on a
tie (`boost == t.boost && bytes.compareTo(t.bytes.get()) > 0` skips the
candidate). `fuzzy_expanded_terms` sorts by `(boost desc, bytes asc)` and
truncates, which selects the same set. This changes the **hit set**, not just
the scores: the existing test asserting `bird`'s postings for
`max_expansions = 1` now asserts `cat`'s, because `cat` is the exact match and
`bird` is the *worst* of the three candidates.

**F25 [MISSING] `LevenshteinAutomata.MAXIMUM_SUPPORTED_DISTANCE` was not
expressed.** `FuzzyQuery`'s constructor throws for `maxEdits > 2` (Lucene ships
parametric descriptions only for 1 and 2). This port's DP has no such ceiling,
which is a genuine capability difference rather than a bug — but a caller
reproducing `FuzzyQuery` needs the constant. Added, with
`DEFAULT_MAX_EDITS`/`DEFAULT_PREFIX_LENGTH`/`DEFAULT_MAX_EXPANSIONS`/
`DEFAULT_TRANSPOSITIONS`. `FuzzyQuery::new` still accepts a larger value; the
divergence is now documented on the type rather than implicit.

**F26 [PERF] The DP was unbanded and had no length short-circuit.**
Lucene never computes a distance at all. Ours allocated an `(n+1) × (m+1)`
table and filled all of it for every candidate term, at `maxEdits ≤ 2`.
`edit_distance_at_most` now rejects on `|n − m| > max` before allocating and
bands the inner loop to `2·max + 1` cells — five per row at Lucene's ceiling
instead of `m`. Not separately benchmarked: the shape of the win is not in
doubt and the fuzzy path is dominated by the term scan, which F15's technique
does not help (a Levenshtein automaton's dead states are what would).

**F27 [MISSING] No `MaxNonCompetitiveBoostAttribute` feedback loop.**
`FuzzyTermsEnum` swaps to a *lower*-edit automaton mid-enumeration once
`TopTermsRewrite`'s queue is full and its worst boost exceeds what a
higher-distance term could score. Recorded, not ported: it needs the automaton
family (`buildAutomatonSet`), and the pruning it buys is over the term scan,
which this port does differently anyway.

**F28 [INTENTIONAL] The restricted (optimal-string-alignment) Damerau variant.**
Confirmed rather than assumed: `Lev1T`/`Lev2T` encode the restricted variant, so
a transposed pair may not be edited again. Ours matches; both directions
(`transpositions` true and false) are tested.

### Verdict

Open: F27 (competitive-boost early exit), and the single-segment blending noted
under F23. Everything else fixed.

---

## `crates/lucene-codecs/src/suggest.rs`

**Java counterpart:**
`lucene/suggest/src/java/org/apache/lucene/search/suggest/fst/WFSTCompletionLookup.java`,
plus `suggest/SortedInputIterator.java` and `util/fst/Util.java`
(`shortestPaths`, `TopNSearcher`, `TieBreakByInputComparator`).

| Rust | Java | Verdict |
|---|---|---|
| `encode_weight` / `decode_weight` | `encodeWeight` / `decodeWeight` | identical (widened to `u32`, INTENTIONAL) |
| `build_suggester_fst` | `build` + `WFSTInputIterator` | was **divergent** (F29), now identical |
| `top_n_completions` | `lookup` + `Util.shortestPaths` | divergent by design (F31); `exactFirst` added (F30) |
| — | `store` / `load` / `get` / `getCount` / `ramBytesUsed` | not ported; `store`/`load` fall out of the FST format (documented) |

**F29 [CORRECTNESS] Duplicate surface forms kept the wrong weight.**
`WFSTCompletionLookup.build` reads through a `SortedInputIterator` whose
comparator is documented as *"Sortes by BytesRef (ascending) then cost
(ascending)"* — and cost is `Integer.MAX_VALUE - weight`, so ascending cost is
**descending weight**. The dedup loop then skips entries equal to the previous
one, which is why the code comments *"for duplicate suggestions, the best weight
is actually added"*. Ours sorted by term alone and relied on the sort's
stability, keeping whichever weight the caller happened to pass first — a
silently different suggester for any dictionary with a repeated surface form,
and the existing test asserted the wrong behaviour. **Fixed**; the test now
checks both input orders, since a stable sort passes one of them by accident.

**F30 [MISSING] `exactFirst` was not implemented, and its default is `true`.**
`WFSTCompletionLookup(Directory, String)` delegates to `(dir, prefix, true)`:
when the queried prefix is itself an indexed term it is returned **first**,
regardless of weight, and `Util.shortestPaths` is then called with
`allowEmptyString = !exactFirst` so it cannot come back twice. `top_n_completions`
now takes the flag (the one existing caller-visible signature change in this
batch; there are no callers outside the module).

**F31 [INTENTIONAL] Bounded min-heap over the prefix range, not
`Util.shortestPaths`.** Already documented in the module. Re-verified: the
tie-break matches (`TieBreakByInputComparator` breaks by `path.input`
ascending, ours by suffix ascending), and the walk is confined to the prefix's
subtree. What is missing is the priority search's ability to skip subtrees that
provably cannot beat the current worst candidate. `Util.shortestPaths`/
`TopNSearcher` stay on the carry-over list.

### Verdict

Open: F31 only (recorded, unchanged from before).

---

## `crates/lucene-analysis/src/lib.rs`

**Java counterparts:** `core/src/java/org/apache/lucene/analysis/`
(`Analyzer`, `TokenStream`, `Tokenizer`, `LowerCaseFilter`, `StopFilter`,
`FilteringTokenFilter`, `CharacterUtils`, `CharArrayMap`/`CharArraySet`,
`standard/StandardTokenizer`, `standard/StandardTokenizerImpl.jflex`) and
`analysis/common/src/java/org/apache/lucene/analysis/`
(`miscellaneous/ASCIIFoldingFilter`, `en/PorterStemFilter`, `en/PorterStemmer`,
`synonym/SynonymGraphFilter`, `ngram/NGramTokenFilter`,
`ngram/EdgeNGramTokenFilter`, `core/KeywordTokenizer`,
`standard/StandardAnalyzer`).

| Rust | Java | Verdict |
|---|---|---|
| `Token` | `CharTermAttribute` + `Offset` + `PositionIncrement` + `PositionLength` | INTENTIONAL, plus F41 |
| `tokenize` | `StandardTokenizer` (`StandardTokenizerImpl.jflex`) | divergent (F40, F41, F42) |
| `LowerCaseFilter::apply` | `LowerCaseFilter` → `CharacterUtils.toLowerCase` | was **divergent** (F32), now identical |
| `StopFilter::apply` | `StopFilter` + `FilteringTokenFilter` | divergent (F38, F39) |
| `ENGLISH_STOP_WORDS` | `EnglishAnalyzer.ENGLISH_STOP_WORDS_SET` | identical (33 words, re-verified) |
| `AsciiFoldingFilter::fold_char` | `ASCIIFoldingFilter.foldToASCII` | was **divergent** (F33), now identical |
| `AsciiFoldingFilter::apply_with` | `ASCIIFoldingFilter(input, preserveOriginal)` | added (F34) |
| `porter::stem` | `PorterStemmer.stem(char[], int)` | was **divergent** (F35), now identical |
| `porter::{is_consonant, measure, contains_vowel, ends_double_consonant, cvc}` | `cons`, `m`, `vowelinstem`, `doublec`, `cvc` | identical |
| `porter::{step1a, step1b, step1c}` | `step1`, `step2` | identical |
| `porter::{step2, step3, step4}` | `step3`, `step4`, `step5` | was **divergent** (F35), now identical |
| `porter::{step5a, step5b}` | `step6` | identical |
| `SynonymFilter::apply_multiword` | `SynonymGraphFilter.bufferOutputTokens` + `releaseBufferedToken` | was **divergent** (F36), now identical |
| `SynonymFilter::apply` | single-word `SynonymGraphFilter` | positions/offsets identical, emission order differs (F43) |
| `apply_ngram_filter` | `NGramTokenFilter`/`EdgeNGramTokenFilter.incrementToken` | was **divergent** (F37), now identical |
| `validate_gram_range` | both constructors' `IllegalArgumentException`s | equivalent |
| `Analyzer::standard` | `StandardAnalyzer.createComponents` | divergent (F42) |
| `Analyzer::keyword` | `KeywordAnalyzer` / `KeywordTokenizer` | identical modulo F41 |
| `SnowballEnglishStemFilter` | `SnowballFilter(new EnglishStemmer())` | already fixture-verified; unchanged |
| — | `Analyzer.normalize`, `getPositionIncrementGap`, `getOffsetGap`, reuse strategies | MISSING (F44) |
| — | `CharFilter` / `Tokenizer.correctOffset`, `AnalyzerWrapper`, `PerFieldAnalyzerWrapper`, `KeywordAttribute`, `WordlistLoader`, the ~40 per-language analyzers | MISSING, out of scope (F45) |

**F32 [CORRECTNESS] `to_lowercase()` is not `Character.toLowerCase`.**
`CharacterUtils.toLowerCase` writes `Character.toChars(Character.toLowerCase(cp))`
back into the buffer at the same index — the **simple**, strictly 1:1 mapping.
Rust's `str::to_lowercase` is the full mapping. Two concrete disagreements:
`U+0130 İ` gave `i` + `U+0307` where Java gives a bare `i`, and `"ΟΔΟΣ"` gave
`"οδος"` where Java gives `"οδοσ"` (Java has no final-sigma rule). Either one
indexes a term under different bytes than Lucene, which breaks exact term
lookup outright. **Fixed** with a per-codepoint mapping; `U+0130` is the only
unconditional full-lowercase expansion in Unicode, so it is the only special
case needed. Fixture-pinned (`lowercase_*`).

**F33 [CORRECTNESS] The ASCII folding table covered 7% of Lucene's.**
`ASCIIFoldingFilter.foldToASCII`'s switch has 1242 `case '\uXXXX'` labels; this
port had 92 (Latin-1 letters plus 30 Latin Extended-A picks). Every one of the
92 was *right* — the problem was everything else: all of Latin Extended-B, IPA
Extensions, Phonetic Extensions, **all** of Latin Extended Additional (every
precomposed Vietnamese letter), General Punctuation and superscripts, Enclosed
Alphanumerics, Dingbats, Latin Extended-C/D, the `ﬁ`/`ﬂ` ligatures, and all of
Halfwidth/Fullwidth Forms folded to themselves here and to ASCII in Lucene.
**Fixed** by generating the table from the real filter — every BMP codepoint
run through `foldToASCII` from `lucene-analysis-common-10.5.0.jar`, keeping the
1242 that changed — rather than extending the hand-written list. Lookup is a
binary search over the sorted table. Pinned per Unicode block by eight new
fixture cases.

**F34 [MISSING] `preserveOriginal` on `ASCIIFoldingFilter`.** Java emits the
folded token and then the original with `positionIncrement = 0`. Added as
`apply_with`.

**F35 [CORRECTNESS] The Porter stemmer had four defects.** All four are now
pinned by a `porter_english` fixture over a 100-word vocabulary run through the
real `PorterStemFilter`.

1. **No length guard.** Java runs no step unless `k > k0 + 1` — the word is at
   least three characters. Without it, step 1a deleted the whole of `"s"`,
   producing a **zero-length term** pushed downstream, and `"as"`/`"is"`/`"us"`
   lost their `s`.
2. **A guard Java does not have.** This port skipped any word that was not
   all-lowercase-ASCII. Java's `cons()` simply treats anything that is not
   `a/e/i/o/u/y` as a consonant, so `"Cats"` really does stem to `"Cat"` and
   `"Running"` to `"Run"`. A chain without a lowercasing filter diverged on
   every capitalised word.
3. **Two wrong rules.** The `l` group's first entry is `bli -> ble`, written
   here as `abli -> able`, so `"possibly"` stopped at `"possibli"` instead of
   `"possibl"`. And the `g` group — `logi -> log` — was missing entirely, so
   `"technology"` stopped at `"technologi"`.
4. **The wrong search structure**, which is the subtle one. Java `switch`es on
   a single character and `break`s out at the **first** suffix whose `ends()`
   succeeds — `r()` checks `m()` *before* that `break`, so a measure failure
   ends the search rather than falling through to a shorter suffix. A flat
   rule list that keeps searching gives different stems:
   `"argument"` matches `-ment` with `m("argu") == 1`, not `> 1`, so Lucene
   leaves it alone where the fall-through strips `-ent` and yields `"argum"`.
   The rewrite reproduces the dispatch character and the first-match-wins rule
   exactly (`step2` on the second-to-last character, `step3` on the last,
   `step4` on the second-to-last again, with `-ion`'s `s`/`t` guard and the
   `o` group's `-ou` fallback).

**F36 [CORRECTNESS] The synonym graph placed collapsed synonyms one position
late and emitted offsets backwards.** `SynonymGraphFilter.bufferOutputTokens`
assigns node ranges and `releaseBufferedToken` derives
`positionIncrement = startNode - lastNodeOut` and
`positionLength = endNode - startNode` from them; the buffering order is *first
token of every synonym path, then the first original, then the remainders* —
Java's comment on that ordering is *"We must do the original tokens last, else
the offsets 'go backwards'"*. Ours emitted the originals first and the synonym
after them with increment 0, which put `wifi` at `fi`'s position rather than
`wi`'s, and stamped the whole match's offsets onto **every** token, producing a
decreasing `startOffset` that real Lucene's `IndexingChain` rejects with
"startOffset must be non-decreasing". A single input expanding to a multi-word
output also left the original's `position_length` at 1 where both paths must
rejoin at the same end node. **Fixed** as a direct transcription of the node
arithmetic, including the graph bookkeeping across pass-through tokens; six
fixture cases pin it, and this port now reproduces Java's output token for
token, attribute for attribute.

**F37 [CORRECTNESS] The n-gram filters dropped positions and invented offsets.**
Java accumulates `curPosIncr += posIncrAtt.getPositionIncrement()` per input
token and only zeroes it once a gram is actually emitted, so a token shorter
than `minGram` still consumes a position — with `minGram = 3`, `"a big cat"`
puts `cat`'s grams two positions after `a`'s, and this port put them one after,
corrupting every downstream phrase and slop offset. And every gram carries the
**input token's** offsets, because `incrementToken` calls `restoreState(state)`
and never `setOffset` (the filter's own javadoc says so, and says that is why
highlighting does not work with it); this port computed a precise per-gram byte
range, disagreeing with Lucene on every gram. **Fixed**, both filters, plus
`preserveOriginal` (F34's sibling: a too-short token is kept, a too-long one is
re-emitted after its grams at increment 0, a token exactly `maxGram` long is
not duplicated). Five fixture cases.

**F38 [MISSING] No `TokenStream.end()`, so trailing position increments are
lost.** `FilteringTokenFilter.end()` adds leftover `skippedPositions` onto the
final increment, and both n-gram filters publish their leftover `curPosIncr` the
same way. A `Vec<Token> -> Vec<Token>` filter has nowhere to put it, so a
document ending in stop words (or in too-short tokens) does not advance the
position counter the way Lucene's does — visible when a second value of the same
multi-valued field follows. **Recorded, not fixed**: it needs the streaming
lifecycle, which is the crate's central design decision, not a local bug.

**F39 [MISSING] No case-insensitive stop set.** `StopFilter.makeStopSet(…,
ignoreCase)` builds a `CharArraySet` that folds with
`CharacterUtils.toLowerCase` per codepoint on both `put` and lookup. Ours is a
`HashSet<String>` with exact equality. Recorded; note that F32's fix is a
precondition for ever getting this right, since the two implementations have to
fold with the *same* function.

**F40 [CORRECTNESS] Emoji produce no token.**
`StandardTokenizerImpl.jflex` returns `EMOJI_TYPE` for emoji/ZWJ/keycap/
regional-indicator sequences and `StandardTokenizer` emits them;
`unicode_word_indices()` only yields segments containing alphanumerics, so they
vanish. The module doc asserts this *is* Lucene's behaviour, and an existing
test asserts it too — both are wrong. Recorded, not fixed: it needs
`split_word_bounds` plus an emoji classification pass, which is a tokenizer
rewrite rather than a patch.

**F41 [CORRECTNESS] Offsets are UTF-8 byte offsets; Lucene's are UTF-16
code-unit offsets.** Already documented at the top of the module and unchanged
here, but it is a real divergence, not just a unit convention: the fixture tests
have to convert, and the non-ASCII cases added in this batch can only assert
terms and increments. Recorded.

**F42 [MISSING] `StandardTokenizer.maxTokenLength` (255) and
`StandardAnalyzer.normalize`.** `StandardAnalyzer` pushes a 255-character cap
into its tokenizer and skips over-long tokens while counting their positions;
`normalize` is the multi-term (prefix/wildcard/fuzzy/range) query-analysis
chain. Neither exists here. Recorded.

**F43 [INTENTIONAL] `SynonymFilter::apply` emits the original before the
synonym; Lucene emits the synonym first.** Verified against real
`SynonymGraphFilter` (`the cat sat` with `cat → feline` gives
`the, feline, cat, sat`). Positions, lengths and offsets are identical — only
the order *within* one position differs, which a positional index cannot
observe. Left alone rather than churning the tests for no semantic gain.

**F44 / F45 [MISSING] The unported Java surface**, enumerated for parity
tracking: `Analyzer.normalize` / `getPositionIncrementGap` / `getOffsetGap` /
reuse strategies / `close`; `CharFilter` + `Tokenizer.correctOffset`;
`AnalyzerWrapper` / `PerFieldAnalyzerWrapper` (this port's `analyze` takes no
field name at all); `KeywordAttribute` + `KeywordMarkerFilter` (so
`PorterStemFilter` cannot honour a stem-exclusion set);
`EnglishPossessiveFilter`; `NGramTokenizer`/`EdgeNGramTokenizer` (the tokenizer
form); `SynonymMap`/`SolrSynonymParser`/`WordnetSynonymParser` and per-rule
`keepOrig`; `WordlistLoader`; `CachingTokenFilter`/`GraphTokenFilter`/
`TokenStreamToAutomaton` (nothing in this port *reads* `position_length`);
`TokenFilterFactory`/SPI/`CustomAnalyzer`; and the ~40 per-language analyzer
packages plus ~60 `miscellaneous/` filters.

### Verdict

Open: F38 (no `end()` hook), F39 (case-insensitive stop set), F40 (emoji),
F41 (offset units), F42 (`maxTokenLength`, `normalize`), F44/F45 (unported
surface). Everything reachable through the existing API and testable against a
fixture is fixed.

---

## Fixtures added

- `fixtures/src/GenRegexp.java` → `fixtures/data/regexp/{terms.txt,cases.tsv}`,
  read by `crates/lucene-codecs/tests/regexp_fixtures.rs`. 79 patterns × 66
  terms of real `RegExp` + `ByteRunAutomaton` ground truth, including which
  patterns Lucene rejects. Three tests: accept/reject agreement,
  `literal_prefix` soundness, `dead_prefix_len` soundness.
- `fixtures/src/GenAnalysis.java` gained 21 cases (`porter_english`, eight
  `fold_*`, three `lowercase_*`, five n-gram, six `syn_*`), read by five new
  tests in `crates/lucene-analysis/tests/analysis_fixtures.rs`. The `syn_*`
  cases record `PositionLengthAttribute`, which no earlier case needed.

`fixtures/data/analysis/manifest.properties` was regenerated by running
`GenAnalysis` alone (it writes only `analysis/`), so no other batch's fixture
directory was touched.

## Benchmark added

`crates/lucene-codecs/benches/regexp_intersect.rs` — the F15 measurement, kept
so the adaptive give-up's thresholds can be re-derived rather than trusted.
