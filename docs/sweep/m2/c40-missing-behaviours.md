# c40-missing-behaviours

The remaining **tier-B** items: Lucene behaviours a caller can reach that this
port does not implement. Four were named
(`LEDGER.md` items **8**, **10**, **9** and **7b**); all four are closed, one
of them by discovering that the ledger's own prescription for it was wrong.

Java read from **`/home/tuong/work/lucene-10.5.0`** throughout.

| | |
|---|---|
| Findings | **13** |
| `CORRECTNESS` | **4** (all fixed) |
| `MISSING` | **4** (all fixed) |
| `PERF` | **3** (2 fixed + measured, 1 recorded) |
| `INTENTIONAL` | **2** |

Files touched: `crates/lucene-codecs/src/{field_infos.rs,fuzzy.rs,blocktree.rs}`,
`crates/lucene-codecs/examples/write_field_infos_fixture.rs`,
`crates/lucene-analysis/src/lib.rs`,
`crates/lucene-index/src/{indexing_chain.rs,index_writer.rs}`,
`crates/lucene-index/tests/multi_valued_fields.rs` (new),
`crates/lucene-search/src/{near_spans.rs (new),highlighter.rs,lib.rs}`,
`crates/lucene-search/examples/fuzzy_pruning.rs` (new),
`crates/lucene-search/tests/{highlighter_offsets_fixtures.rs,span_query_fixtures.rs,multi_valued_phrase.rs (new)}`,
`crates/lucene-ffi/src/{writer.rs,registry.rs,handle.rs}`,
`fixtures/src/{GenAnalysis.java,AppendHighlightManifest.java (new),AppendSpanNearManifest.java (new)}`,
`scripts/gen-fixtures.sh`, `docs/parity.md`, `docs/sweep/m2/LEDGER.md`.

---

## `crates/lucene-codecs/src/field_infos.rs`

Java: `lucene/core/src/java/org/apache/lucene/index/FieldInfo.java`,
`FieldInfos.java`,
`lucene/core/src/java/org/apache/lucene/codecs/lucene94/Lucene94FieldInfosFormat.java`.

| Rust | Java | verdict |
|---|---|---|
| `FieldInfo` (struct literal) | `FieldInfo`'s 18-parameter constructor | **missing** -> fixed (`new`/`with_*`/`checked`) |
| `FieldInfo::checked` | the constructor body: non-indexed coercion, then `checkConsistency()` | added, identical |
| `FieldInfo::check_consistency` | `FieldInfo.checkConsistency` | identical; **now `pub`, as Java's is** |
| `FieldInfos::new` | `FieldInfos(FieldInfo[])` | **missing** -> added |
| `FieldInfos::check_consistency` | the cross-field half of that constructor | identical |
| `parse` | `Lucene94FieldInfosFormat.read` | **divergent** -> fixed (finding 1) |
| `write` | `Lucene94FieldInfosFormat.write` | identical (already coerced) |

### 1 `[CORRECTNESS]` `parse` rejected a `.fnm` real Lucene opens — fixed

`Lucene94FieldInfosFormat.read` builds each entry with
`new FieldInfo(...)` and **then** calls `checkConsistency()`. The constructor's
"for non-indexed fields, leave defaults" branch has already forced
`storeTermVector`/`storePayloads`/`omitNorms` to `false`, so
`checkConsistency`'s three "non-indexed field cannot …" throws are unreachable
from the reader. This port ran the check on the raw bits and **rejected the
file**.

Consequence: exactly the defect c23 met from the other side — an `IndexWriter`
(this port's or anyone's) that emitted those bits produced a segment its own
reader refused, which `check_index` reports as a segment with no readable
postings rather than as a coercion. Fixed by routing every decoded field
through `FieldInfo::checked()`.

Evidence, both directions: `write_field_infos_fixture` now emits a
`noindex_flags` field whose FieldBits carry all three flags with
`IndexOptions = NONE` — a combination no writer can produce, so the example
patches the IndexOptions byte after `write` and re-stamps the footer CRC.
Real Lucene reads it back with all three cleared (`verify-write-path.sh`,
`VerifyFieldInfos`, 23/23 pass), and
`non_indexed_field_has_its_indexed_only_bits_coerced_away_not_rejected`
requires the same of `parse` over all four bit combinations. The three tests
that asserted the *rejection* are gone; that is the behaviour change.

### 2 `[MISSING]` no validating constructor — fixed, without touching 213 literals

`FieldInfo::new(name, number)` seeds the one trivially-consistent shape (no
options, `dvGen = -1`, `FLOAT32`/`EUCLIDEAN` as `IndexingChain`'s own
non-vector fields carry); chained `with_*` setters take the constructor's
parameters, grouped where Java cross-checks them (`with_doc_values`,
`with_points`, `with_vectors`); `checked()` is the constructor **body**.
`FieldInfos::new(Vec<FieldInfo>)` is `FieldInfos(FieldInfo[])`.

The 213 existing struct literals are untouched. What closes the class is that
**both doors a caller can reach are now shut**: `field_infos::parse` (finding 1)
and `IndexWriter::open`, which puts its whole field list through
`FieldInfos::new` and so both coerces and validates before a document is added.
`field_infos::write` keeps coercing rather than erroring — Java's writer never
sees the combination at all, and `write` -> `parse` staying total is what makes
the round-trip tests meaningful.

Tests: five new in `field_infos.rs` (the seed's defaults, the coercion, the one
indexed-field violation the coercion does not rescue, all ten other
`checkConsistency` violations, every setter), two in `index_writer.rs`
(`open` rejects; `open` coerces).

### 3 `[INTENTIONAL]` `Error::PayloadsWithoutPositions` deleted

`resolve_postings_field` re-checked "indexed field cannot have payloads without
positions" so `set_postings_field` could name the caller's mistake. With `open`
validating, no input reaches it — a guard nothing can trip is a guard nothing
tests. Removed, along with the error variant; `lucene-ffi`'s
argument-error classification now lists `Error::FieldInfos(_)`, which carries
the same message. Its test became `open_rejects_payloads_on_a_field_without_positions`.

### Verdict

Swept-clean; ledger item 8 closed.

---

## `crates/lucene-analysis/src/lib.rs` + `crates/lucene-index/src/indexing_chain.rs`

Java: `lucene/core/src/java/org/apache/lucene/analysis/{TokenStream,Tokenizer,Analyzer,FilteringTokenFilter}.java`,
`lucene/analysis/common/src/java/org/apache/lucene/analysis/ngram/{NGramTokenFilter,EdgeNGramTokenFilter}.java`,
`lucene/core/src/java/org/apache/lucene/index/IndexingChain.java`
(`PerField.invertTokenStream`), `FieldInvertState.java`.

| Rust | Java | verdict |
|---|---|---|
| `TokenStream` (new struct) | the two attributes read after `stream.end()` | **missing** -> added |
| `tokenize_stream` | `Tokenizer.end()`'s `finalOffset = correctOffset(charCount)` | **missing** -> added |
| `StopFilter::apply_to_stream` | `FilteringTokenFilter.end()` | **missing** -> added |
| `{NGram,EdgeNGram}TokenFilter::apply_to_stream` | those filters' `end()` | **missing** -> added |
| `Analyzer::{position_increment_gap,offset_gap}` | `Analyzer.get{PositionIncrementGap,OffsetGap}` | **missing** -> added (Java's defaults 0 / 1) |
| `invert_documents_with_payloads`' per-value loop | `PerField.invert(docID, field, first)` + `FieldInvertState.reset()` | **divergent** -> fixed |
| `IndexWriter::build_postings_output`'s field lookup | `Document.getFields(name)` | **divergent** -> fixed |

### 4 `[CORRECTNESS]` a multi-valued field restarted at position 0 and offset 0 — fixed

`invert_documents*` took `docs: &[(doc_id, field, text)]` and reset
`position = -1`/`offset = 0` for **every tuple**. Java resets
`FieldInvertState` only for the *first* value of a field in a document
(`PerField.invert(docID, field, first)`), then, after each value:

```java
stream.end();
invertState.position += invertState.posIncrAttribute.getPositionIncrement();
invertState.offset   += invertState.offsetAttribute.endOffset();
if (analyzed) {
  invertState.position += analyzer.getPositionIncrementGap(fieldInfo.name);
  invertState.offset   += analyzer.getOffsetGap(fieldInfo.name);
}
```

Consequence: two values of one field both began at position 0 with identical
offsets — a phrase match across the boundary Lucene does not have, and
occurrences claiming the same character span. Fixed by grouping tuples by
`(doc_id, field)` into one `FieldInvertState` and carrying both counters, plus
the two gaps.

**By key, not by adjacency** — the Tier-2 review caught the first attempt
joining only *consecutive* runs, which re-creates the same defect for
`[(0,"f",..), (0,"g",..), (0,"f",..)]`, an input `invert_documents`' own
documented contract accepts. Java has no adjacency requirement either: `PerField`
owns its `FieldInvertState` for the whole document and `processField` resets it
on `pf.fieldGen != fieldGen`. Grouping is now a `HashMap` keyed lookup with a
`Vec` preserving first-appearance order (a linear scan over the groups would be
quadratic in a 50 000-document flush), and the contract now states the one
property of the input order that legitimately decides the answer: the order of
one field's own values, which is `Document.getFields(name)`' order in Java.
`a_fields_values_are_one_field_even_with_another_fields_value_between_them`
requires the interleaved and adjacent inputs to produce identical postings.

### 5 `[MISSING]` `end()`'s trailing position increment — fixed

`FilteringTokenFilter.end()` adds its leftover `skippedPositions` to the
position-increment attribute; the n-gram filters publish their leftover
`curPosIncr` (and **set** rather than add, discarding an upstream filter's own —
reproduced as-is). This crate's `Vec<Token> -> Vec<Token>` filters had nowhere
to put either. `TokenStream { tokens, final_position_increment, final_offset }`
is those two attribute reads; `Analyzer::analyze` is now
`analyze_stream(..).tokens`, so no caller changed.

### 6 `[MISSING]` `IndexWriter` indexed only the first value of a multi-valued field — fixed

`build_postings_output` used `doc.fields.iter().find(|f| f.field_number == n)`.
Every later value was **stored and never indexed**. Now every matching value
becomes a tuple, consecutively, which is what finding 4's grouping consumes.
`IndexWriter::set_position_increment_gap`/`set_offset_gap` are the subclass hook
Java's `Analyzer` exposes (this facade has no per-field analyzer configuration).

### Evidence

`GenAnalysis.java` gained four `mv_*` cases, each indexing the values as
repeated values of one field through a **real `IndexWriter`** into a
`ByteBuffersDirectory` and recording (a) every occurrence's
`(term, position, startOffset, endOffset)` read back off the postings and
(b) `IndexSearcher.count` for phrases straddling the value boundary. Reading
the *postings* rather than a `TokenStream` is the point: all of this happens
downstream of every attribute a token list can show.

| case | analyzer | Lucene's postings | Lucene's phrase answers |
|---|---|---|---|
| `mv_default_gap` | StandardTokenizer + lowercase, gaps 0/1 | `alpha:0:0,5;beta:1:6,10;gamma:2:11,16;delta:3:17,22` | `"beta gamma"~0` **matches** |
| `mv_gap_100` | + `getPositionIncrementGap = 100` | `gamma:102`, `delta:103` | `~0`/`~99` no, `~100` yes |
| `mv_trailing_stopwords` | + StopFilter{the}, gaps 0/1 | `fox:0:0,3;dog:3:12,15` | `"fox dog"~0`/`~1` no, `~2` yes |
| `mv_stopwords_and_gap` | + gaps 5/2, three values | `fox:0;dog:8;bird:14` | `"fox dog"~0` no, `~7` yes |

`mv_default_gap` is deliberately the *surprising* direction: Java's base
`Analyzer` returns **0**, so a phrase does match across a value boundary, and a
port that "fixed" this by always inserting a gap fails here.

`crates/lucene-index/tests/multi_valued_fields.rs` requires all four postings
strings from `invert_documents`; verified to fail on all four against the
unfixed code (`mv_trailing_stopwords` gave `dog:0:0,3;fox:0:0,3`).
`crates/lucene-search/tests/multi_valued_phrase.rs` requires Lucene's own hit
counts end to end through `IndexWriter` -> `DirectoryReader` ->
`search_phrase_query` for the two cases this writer's analyzer can reproduce
exactly (it analyses with `Analyzer::standard(None)`, so a stopword case would
index different *terms*; those two are covered one layer down). Seven unit tests
cover the lifecycle values themselves.

### Verdict

Ledger item 10's takeable half closed. Still open there and restated in the
ledger: no case-insensitive `CharArraySet`, no `maxTokenLength`.

---

## `crates/lucene-search/src/near_spans.rs` (new) + `highlighter.rs`

Java: `lucene/queries/src/java/org/apache/lucene/queries/spans/{NearSpansUnordered,NearSpansOrdered,ConjunctionSpans}.java`,
`lucene/highlighter/src/java/org/apache/lucene/search/uhighlight/PhraseHelper.java`,
`lucene/highlighter/src/java/org/apache/lucene/search/highlight/WeightedSpanTermExtractor.java`.

| Rust | Java | verdict |
|---|---|---|
| `near_spans::unordered_width` | `SpanTotalLengthEndPositionWindow.atMatch()`'s quantity | added |
| `near_spans::for_each_unordered_match` | `twoPhaseCurrentDocMatches` + `nextStartPosition` + `collect` | added |
| `phrase_match_offsets` (`slop == 0`) | `NearSpansOrdered.stretchToOrder` | identical (unchanged) |
| `phrase_match_offsets` (`slop > 0`) | `NearSpansUnordered` | **missing** -> fixed |
| `collect_span` | `PhraseHelper.OffsetSpanCollector.collectLeaf` + `SpanCollectedOffsetsEnum.add` | identical (extracted) |

### 7 `[CORRECTNESS]` the highlighter enumerated the wrong matcher — fixed, and the ledger's plan was wrong

Ledger item 7b said: make `phrase_match_offsets` enumerate through
`sloppy_phrase`'s walk, growing the matcher a `startPosition()` accessor.
**That would have been a second wrong answer.** `PhraseHelper` never touches
`SloppyPhraseMatcher`. `WeightedSpanTermExtractor.extract` rewrites a
`PhraseQuery` as

```java
boolean inorder = (phraseQuery.getSlop() == 0);
new SpanNearQuery(clauses, phraseQuery.getSlop() + positionGaps, inorder)
```

and `NearSpansUnordered.atMatch()` is
`maxEndPosition - top().startPosition() - totalSpanLength <= slop`, i.e.
`max(p) - min(p) + 1 - n` for `n` one-position term spans — where
`SloppyPhraseMatcher.matchLength` is `max(p_i - i) - min(p_i - i)`. For the
reordered pair `alpha@0 beta@1` queried as `"beta alpha"` those are **0** and
**2**.

So real Lucene's answers, which nothing but running it could have supplied:

| case | doc | phrase | slop | Lucene's `PhraseHelper` |
|---|---|---|---|---|
| `reordered_slop0` | 8555 | `beta alpha` | 0 | **nothing** (`inorder` is forced at slop 0) |
| `reordered_slop1` | 8555 | `beta alpha` | 1 | `alpha:0,5;beta:6,10` — *the scorer does not match here at all* |
| `gap_reordered_slop2` | 8557 | `beta alpha` | 2 | both — the scorer needs slop 4 |
| `repeat_single_occurrence` | 8555 | `alpha alpha` | 2 | `alpha:0,5` — **a document the query does not match** |

The last one is `SpanNearQuery` having no `rptGroups`: two clauses may settle
on one occurrence, giving a negative width. All of that is Lucene's own
inconsistency between highlighting and scoring; reproducing it is the port's
job.

Fixed: `near_spans.rs` is `NearSpansUnordered` over decoded spans, and
`phrase_match_offsets` dispatches on `slop == 0` exactly as
`WeightedSpanTermExtractor`'s `inorder` does. `positionGaps` is always 0 here
because `PhraseQuery` holds `terms` with no per-term positions.

**One deliberate divergence, on the function**: `positionsOrdered` is not a
total order (equal `(start, end)` compares "not less" both ways), so Java's
binary heap leaves `top()` unspecified among ties; the clause index breaks the
tie here, which makes the walk deterministic and can only choose between spans
`atMatch` cannot distinguish.

Evidence: `AppendHighlightManifest.java` runs **real Lucene's own
`PhraseHelper.createOffsetsEnumsForSpans`** (the `lucene-highlighter` jar, added
to `gen-fixtures.sh`'s module list) over the committed `blocktree_index` for
thirteen cases and records the `OffsetsEnum`s in `OffsetsEnum.compareTo` order.
No index was regenerated (`--append-only`).
`highlighter_offsets_fixtures.rs::phrase_helper_offsets_match_real_lucene`
requires all thirteen and additionally asserts that at least one *reordered*
case produced a highlight, so the test cannot pass vacuously; verified to fail
(`reordered_slop1`: `""` vs `alpha:0,5;beta:6,10`) with the in-order path
forced.

### 8 `[CORRECTNESS]` `SpanNearQuery(inOrder = false)` rejected overlapping sub-spans — fixed

Met on the way. `span_near_matches` required the arranged spans to be
non-overlapping on **both** arms. That is `NearSpansOrdered`'s rule
(`stretchToOrder` advances each sub-span to `>= prevSpans.endPosition()`), and
it is *not* `NearSpansUnordered`'s, which has no such rule at all.

Real Lucene, recorded: `SpanNearQuery([alpha, alpha], 0, false)` matches docs
**8555, 8556, 8557** — every document containing `alpha`, including the two with
a single occurrence. This port returned only 8556. Nine `spannear.*` cases in
`AppendSpanNearManifest.java` (new) cover the repeat/transposition/in-order
matrix plus a three-clause case; `span_query_fixtures.rs::span_near_hit_sets_match_real_lucene`
requires all nine and was verified to fail (`[8556]` vs `[8555, 8556, 8557]`).

Fixed by using `near_spans::unordered_width` for the unordered arm. For a
non-overlapping arrangement the two formulas are the same number —
`sum(next.start - prev.end)` telescopes to `maxEnd - minStart - sum(lengths)` —
so only the overlapping case moved.

### 9 `[MISSING]` `span_near_matches` reports extents Java's walk never visits — recorded (ledger 8c)

The cartesian product enumerates arrangements Java's priority-queue walk skips,
and Java's unordered `endPosition()` is its *running* `maxEndPosition`, which
can exceed the current arrangement's own maximum end. Neither changes a hit set
— `span_doc_ids` only asks whether the span list is non-empty — so this can
only matter for a nested `SpanNear`-of-`SpanNear`, where an outer clause
consumes the inner's extents. Recorded on the function and in the ledger rather
than half-fixed: closing it means `span_matches_in_doc` returning the walk's
*sequence*, i.e. generalising `near_spans` to the ordered arm too.

### Verdict

Ledger item 7b closed; 8b raised and closed; 8c raised.

---

## `crates/lucene-search/src/lib.rs` (`fuzzy_expanded_terms`) + `crates/lucene-codecs/src/{fuzzy.rs,blocktree.rs}`

Java: `lucene/core/src/java/org/apache/lucene/search/{FuzzyTermsEnum,TopTermsRewrite,MaxNonCompetitiveBoostAttribute,BoostAttribute}.java`.

| Rust | Java | verdict |
|---|---|---|
| `fuzzy_expanded_terms_pruned`'s bounded queue | `TopTermsRewrite.collectTerms`' `stQueue` | **divergent (sort-then-truncate)** -> now the queue |
| the `bottomChanged` loop | `FuzzyTermsEnum.bottomChanged` | **missing** -> added |
| `FuzzyIntersect::set_max_edits` | `getAutomatonEnum(maxEdits, lastTerm)` | **missing** -> added |
| `FuzzyMatch::edits_within` | `automata[k]` for `k < maxEdits` | **missing** -> added |
| `FuzzyMatch::boost` | `FuzzyTermsEnum.next`'s `BoostAttribute` | identical |
| `distance_chars` | (no counterpart: Java runs a DFA) | `PERF` -> fixed |

### 10 `[MISSING]` the `MaxNonCompetitiveBoostAttribute` feedback loop — fixed and measured

`TopTermsRewrite.collectTerms` publishes the worst boost still in its
size-`maxExpansions` queue plus that term's bytes; `FuzzyTermsEnum.next`
notices they changed and runs

```java
while (maxEdits > 0) {
  float maxBoost = 1.0f - ((float) maxEdits / (float) termLength);
  if (bottom < maxBoost || (bottom == maxBoost && termAfter == false)) break;
  maxEdits--;
}
```

then swaps in `automata[maxEdits]`. This port collected **every** matching term,
sorted and truncated — so the cap bounded the result and nothing else.

Ported: the queue is a sorted `Vec` of at most `maxExpansions` entries in the
final `(boost desc, bytes asc)` order (Java's `ScoreTerm.compareTo` makes
`peek()` exactly the last element here), with `collectTerms`' own
not-competitive rejection; `bottomChanged`'s loop runs against its bottom after
each term; and `blocktree::FuzzyIntersect::set_max_edits` tightens the walk's
predicate. **No re-seek is needed** where Java needs one: the walk is a forward
scan over one sorted range, so the position is already correct — that
difference is documented on the type.

`FuzzyIntersect::set_max_edits` only ever narrows: widening mid-scan would make
the walk yield terms it had already rejected behind the cursor.

*Result-preserving, and tested as such*: a term at distance `d > m` has boost
`<= 1 - d/termLength < 1 - m/termLength`, and the loop only reached `m` because
`bottom >= 1 - d/termLength` at every `d` it passed.
`pruning_never_changes_the_selected_expansion` runs both arms over the
fixture's 400-term `many` field and requires identical terms, boosts and
blended `docFreq`, *and* that the pruned arm's budget actually fell (2 -> 0) —
otherwise the test proves only that a no-op is a no-op. Two more cover the
directions that must not prune: a queue that never fills, and
`maxExpansions == 0`.

### 11 `[PERF]` measured — the min-of-40 alternating A/B

Criterion is unusable here (c24: 83/91/129 µs for identical code), so
`crates/lucene-search/examples/fuzzy_pruning.rs` is an alternating min-of-40 A/B:
one build, one process, both arms, only the `prune` argument differing, and the
two arms' *results* asserted equal before either is timed. Its entry point is
`#[doc(hidden)] pub` rather than feature-gated, so `cargo clippy --all-targets`
builds it — this project has twice had a measurement target rot behind a
feature or a stale binary, which `scripts/gate.sh` already says is worse than a
red build. Corpus:
`benchmarks/.corpus/terms1m` (1 000 000 terms, `t0`..`t999999`); the example
skips itself when it is absent.

| query | pruned | whole | speedup | maxEdits |
|---|---|---|---|---|
| `t100000` ed2 exp50 | **113.9 ms** | 142.6 ms | 1.25x | 2 -> 0 |
| `t100000` ed2 exp10 | **102.8 ms** | 142.4 ms | 1.38x | 2 -> 0 |
| `t100000` ed1 exp50 | **119.1 ms** | 127.9 ms | 1.07x | 1 -> 0 |
| `t12345` ed2 exp50 | **68.8 ms** | 140.0 ms | **2.03x** | 2 -> 0 |
| `t1234` ed2 exp50 | **63.8 ms** | 126.4 ms | **1.98x** | 2 -> 0 |
| `t100000` ed2 exp50 prefix 2 | **12.1 ms** | 13.4 ms | 1.12x | 2 -> 1 |

The saving tracks how early the queue fills and how far the budget then falls.
The shorter query terms gain most: `1 - maxEdits/termLength` is a *lower* bar
for a short term, so the queue's bottom clears it sooner. The `prefix 2` row is
the control — a literal prefix already narrows the scan by two orders of
magnitude, so there is little left for the budget to save, which is the honest
shape of the result rather than a uniform win.

**What it does not buy**: the residual ~114 ms is the scan itself. This port has
no automaton, so it cannot do what Java's `getAutomatonEnum` additionally does —
skip whole `.tim` blocks a dead prefix rules out. That is the same gap
`b8-automata-analysis.md` recorded and it is unchanged; the loop narrows the
band, it does not remove the walk.

### 12 `[PERF]` `distance_chars` allocated `n + 2` `Vec`s per candidate — fixed

Met while measuring finding 11. The banded DP allocated
`vec![vec![unreachable; m + 1]; n + 1]` — one `Vec` for the outer plus one per
row — **per candidate term**. A fuzzy expansion tests every term in the prefix
range, so one query on the 1 M-term corpus made over eight million allocations.
Java allocates nothing here (a compiled DFA over the term's bytes), so this is
not an optimisation past Java, it is the gap closing.

Only rows `i`, `i-1` and `i-2` are ever read (the third solely because of the
transposition rule), so the table is now three rolling rows in one flat buffer.
Measured on the same harness with finding 11's loop already in place, so the
two are separable: the *whole* arm went 203.1 -> 161.9 ms, 189.3 -> 158.0,
176.7 -> 143.9, 187.8 -> 155.9, 172.4 -> 139.4, 18.5 -> 14.4 — **~15-20% on
both arms**, orthogonal to finding 11's saving. (The table above is a later
run, after finding 11's `competitive` gate; the machine drifts a few per cent
between runs, which is exactly why every figure is a min of 40 alternating
repetitions rather than a mean.)

A rolling-row band with index arithmetic is exactly the shape an off-by-one
hides in, and the hand-picked unit tests cannot reach the combination that
would expose one, so
`the_banded_rolling_row_dp_agrees_with_the_full_matrix_everywhere` checks every
string over a 3-letter alphabet up to length 5 against every other, at every
budget 0..=5, with and without transpositions, against a full-matrix reference
kept in the test module — 132 496 pairs x 6 budgets x 2, and a 3-letter alphabet
is what makes transpositions and repeats common enough to exercise the third row.

### 12b `[PERF]` the boost recomputed the distance the matcher had just computed — fixed

`FuzzyTermsEnum.next` computes `ed` (walking *down* the automaton ladder) and
sets `boostAtt` from it in the same method. This port's expansion loop called
`pattern.boost(&term)`, which re-entered `edits` — a second banded DP and a
second `Vec<char>` per **accepted** term, for a number the matcher had just
produced. `TermMatcher::matches` now takes `&mut self`, `FuzzyMatcher` records
the accepted distance, `FuzzyIntersect::last_edits()` carries it out, and
`FuzzyMatch::boost_from_edits` scores from it. `codepoint_count` replaces the
`to_chars(candidate).len()` inside the boost with a byte scan (a codepoint is
one non-continuation byte in well-formed UTF-8; ill-formed input falls back to
the same `U+FFFD` decode, so the two agree everywhere).

Below the harness's resolution on the measured queries, and that is the honest
report of it: accepted terms are a small minority of candidates, and the
pruning loop makes them a smaller one still. It is recorded as fixed because a
DP run twice for one answer is a defect independent of how much of the profile
it happens to be. Raised by the Tier-2 review.

### Verdict

Ledger item 9's feedback-loop half closed. Still open there: `fuzzy_doc_scores`
blends `docFreq` within one segment where `BlendedTermQuery` blends across the
whole reader — the fuzzy clause has no `GlobalStats` plumbing.

---

## Two things met on the way that are not in any of the four items

### 13 `[CORRECTNESS]` `registry::a_handle_with_the_wrong_shard_bits_is_rejected` is order-dependent — fixed

It failed once in this batch's first full `lucene-ffi` run and passed on three
re-runs. It is not flaky noise: the test asserts that rewriting a handle's shard
field "never aliases a live entry in another shard", and **that is not a
property the design has**. The shard field is *routing* — the named shard's own
tag/generation check is what then applies — and every shard's slots start at
index 0 and generation 1, so any concurrently-running test holding a
first-insert results handle in another shard makes exactly that collision
(`concurrent_inserts_spread_across_shards` holds eight of them). Fixed by giving
the test its own `Sharded` registry, where the property under test is the one
being asserted; the over-claiming sentence in `registry.rs`' and `handle.rs`'
module docs is corrected in the same change.

### 14 `[INTENTIONAL]` `points_index`' manifest was missing `AppendPointEstimateManifest`'s keys

`--only GenAnalysis` runs every appender, which filled in 76
`point_estimate.multi.*` lines that were absent from the committed manifest —
c39's appender is present in the tree but its output had never been written
back. `crates/lucene-codecs/tests/points_fixtures.rs` reads those keys, so the
committed tree was one `--check` away from being called out. Left in.
`scripts/gen-fixtures.sh --check` now passes end to end: 48 deterministic files
byte-identical, 0 mismatches, 0 manifests with a wrong key set, 0 segment-id
disagreements.

---

## What the Tier-2 review caught

Two of the twelve items it raised were defects **this batch introduced**, which
is the fifth batch running where that has been true, so they are recorded here
rather than only fixed:

1. **Three doc blocks were stolen by newly-inserted items.** Adding a function
   immediately above an existing one attaches the existing one's `///` block to
   the new one. It happened three times (`phrase_match_offsets`,
   `set_postings_field`, `payload_field_names`), and two of the three victims
   are caller-facing API left with no documentation at all. Nothing in the gate
   catches it — `#![warn(missing_docs)]` on `lucene-search`/`lucene-index` would
   have, and is worth a follow-up batch's consideration.
2. **The first grouping joined only *consecutive* runs** — see finding 4.

The rest were doc/test-quality items, all fixed: the `near_spans` tie-break
justification claimed the choice "cannot matter" when it can (two *different*
terms at one position lead to different continuations; only Java's answer being
unspecified makes the deterministic tie-break defensible); an
`assert!(...all(...))` that an empty list satisfies vacuously;
`AppendHighlightManifest`'s javadoc contradicting the fixture it writes
(slop 0 vs slop 1); `registry.rs`' test doc retracted by its own next
paragraph; `pub mod near_spans` exporting nothing (its two functions are now
`pub`, which is what `docs/parity.md` describes); `FuzzyMatch::boost`'s
zero-length guard justified by the wrong case (a zero-length *query* term, not
candidate — a real, now-documented divergence from Java's `-Infinity`, which
this batch made load-bearing by putting the same arithmetic in
`bottomChanged`); a single-query equivalence test widened to nine shapes; and
finding 12b above.

## Gate

`scripts/docker-test.sh gate`: **ok**. 98.13% lines workspace-wide, no file
below 95% (lowest: `hnsw_vectors.rs` 95.28%, `explain.rs` 95.50%);
`near_spans.rs` at 100%. `scripts/verify-write-path.sh` 23/23.
`scripts/gen-fixtures.sh --check` exit 0.
