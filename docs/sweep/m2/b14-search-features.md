# b14 — search features (`lucene-search`: doc-values / points / facets / highlighting / term vectors / explain / query parser)

Java source of truth: `/home/tuong/work/lucene` (Lucene 10.5.0).

Gate: `cargo fmt --all` ✅ · `cargo clippy -p lucene-search --all-targets -- -D warnings` ✅ ·
`cargo test -p lucene-search` ✅ (725 lib tests + integration suites, all green) ·
`cargo test -p lucene-ffi` ✅ (421, the downstream consumer of `highlighter::Fragment`).

Findings: **9 CORRECTNESS**, **8 MISSING**, **2 PERF**, **5 INTENTIONAL**. All CORRECTNESS
and MISSING findings are fixed with tests; the PERF findings are reasoned and measured.

---

## `crates/lucene-search/src/points_query.rs`

Java counterparts:
- `org/apache/lucene/search/PointRangeQuery.java`
- `org/apache/lucene/search/PointInSetQuery.java` (`MergePointVisitor`, `SinglePointVisitor`)
- `org/apache/lucene/index/PointValues.java` (`intersect`, `Relation`, `IntersectVisitor`)

| Rust `fn` | Java method | Verdict |
|---|---|---|
| `PointsInput::field_number` | (glue; `LeafReader.getPointValues(field)`'s name→reader step) | not-in-Java |
| `pack_i64` | `NumericUtils.longToSortableBytes` | identical |
| `search_points_range` | `PointRangeQuery.createWeight`'s scorer + `MatchingPoints` visitor | identical (via `lucene_index::points_delete::resolve_points_range_doc_ids` → `PointsReader::range_query`) |
| `search_points_in_set` *(new)* | `PointInSetQuery` (`MergePointVisitor` / `SinglePointVisitor`) | **added this batch** |
| — | `PointValues.estimateDocCount` / `estimatePointCount` | missing (recorded, §1.4) |
| — | `PointRangeQuery.toString` | missing (recorded, §1.5) |

### 1.1 [PERF — already fixed upstream, docs were stale] `decode_all_points` claim

Java `PointRangeQuery` walks the BKD tree with `PointValues.intersect`, pruning whole subtrees
on `CELL_OUTSIDE_QUERY` and short-circuiting `CELL_INSIDE_QUERY` to a doc-id-block read.

The module doc claimed (three times, plus on `search_points_range` itself) that this port
"deliberately decodes every point via `decode_all_points` and filters in memory". That stopped
being true when batch b7 moved `lucene_index::points_delete::resolve_points_range_doc_ids` onto
`PointsReader::range_query`, which *is* the pruning walk — `search_points_range` delegates to it,
so it has been sublinear since b7 without anyone updating its docs.

**Measured** (the b7 benches, re-confirmed as the honest numbers to cite): on a 200k-point tree,
`points/range_query_selective_200k` vs `points/decode_all_then_filter_selective_200k` = **577x**.
No code change was needed; the stale doc comments are now corrected and cite the two bench names.

### 1.2 [MISSING → fixed] `PointInSetQuery`

Java ships a second BKD query: "value is exactly one of a set". Not ported at all.

Added `search_points_in_set`, faithful to both of Java's visitor strategies:
- **`numDims == 1`** — a real port of `MergePointVisitor`: the sorted, deduplicated query set is
  merge-walked against the tree's cells, so a cell entirely below the next unconsumed query point
  advances the cursor, a cell entirely above it returns `CELL_OUTSIDE_QUERY` (pruning the subtree
  without reading a `.kdd` byte), and a cell whose min *and* max both equal the query point
  short-circuits to `CELL_INSIDE_QUERY` (Java's "> 512 docs share this one value" case).
- **`numDims > 1`** — `SinglePointVisitor`: one `[point, point]` traversal per query point, results
  folded together (Java's `DocIdSetBuilder`, i.e. sort + dedup).

Java's `sortedPackedPoints` normalization (sort + dedup) is reproduced, `live_docs` filtering and
the "unknown field / empty set collects nothing" convention match `search_points_range`.
Fixed in `points_query.rs`; 7 new tests, including the exact-equality-vs-range distinction, an
unsorted/duplicated input, and a multi-dimension point that shares one dimension with two docs but
matches neither.

### 1.3 [handed over from b11 — fixed] `corrupt_kdd_leaf_data_surfaces_as_points_error` failed

The test scrambled the `.kdd` bytes between header and footer and asserted the query erred. Under
b7's pruning walk an all-encompassing `[i64::MIN, i64::MAX]` range hits `CELL_INSIDE_QUERY` at the
root, which — exactly like Java's `visitDocIDs` shortcut — reads only the leaf's doc-id block and
never decodes a packed value, so it returned garbage doc ids without erroring.

**`points.rs`'s validation was not weakened.** The test now uses a *narrow* range, making the root
cell `CELL_CROSSES_QUERY` and forcing the leaf-block decode that genuinely fails
(`Store(MalformedVarint)` → `Error::Points`). The comment records why the wide range doesn't error
and that detecting *that* corruption is the footer checksum's job in Java too.

### 1.4 [MISSING — recorded] `PointValues.estimateDocCount` / `estimatePointCount`

Java uses these for `ScorerSupplier.cost()`, which drives `IndexOrDocValuesQuery`'s plan choice.
This port has no `ScorerSupplier`/query planner to feed, so a cost estimate would have no consumer.
Recorded in the module doc's out-of-scope list.

### 1.5 [MISSING — recorded] `PointRangeQuery.toString`

Would need a `Query.toString()` port across `query.rs` (batch b12's file). `explain.rs` grew a
`describe_clause` that covers `Clause::PointsRange` in the shape Java prints; promoting it to a real
`Display` on the query types belongs to whoever owns `query.rs`.

### Verdict

Swept clean. Docs corrected, `PointInSetQuery` ported, the handed-over test failure resolved without
touching `points.rs`. Two `MISSING` items recorded with reasons.

---

## `crates/lucene-search/src/doc_value_query.rs`

Java counterparts:
- `org/apache/lucene/document/{SortedNumericDocValuesField,SortedSetDocValuesField,NumericDocValuesField}.java`
- `org/apache/lucene/search/{SortedNumericDocValuesRangeQuery,SortedSetDocValuesRangeQuery,DocValuesRangeIterator,IndexOrDocValuesQuery,FieldExistsQuery}.java`
- `org/apache/lucene/search/{SortedNumericSelector,SortedNumericSortField,TopFieldCollector}.java`

| Rust `fn` | Java method | Verdict |
|---|---|---|
| `search_numeric_range` | `NumericDocValuesField.newSlowRangeQuery` → per-doc `advanceExact` + range test | identical |
| `search_numeric_range_with_skip_index` | `DocValuesRangeIterator` / `DocValuesSkipper` block skipping | divergent, PERF only (§2.3) |
| `search_sorted_ord_range` | `SortedSetDocValuesRangeQuery`, single-valued (`SortedDocValues`) case | identical (ordinal resolution is the caller's, matching `sorted_ord`'s contract) |
| `search_sorted_numeric_range` *(new)* | `SortedNumericDocValuesRangeQuery.getTwoPhaseIterator` | **added this batch** |
| `search_multi_valued_range` | *no Java counterpart* — a filter shaped after `SortedNumericSelector` | **CORRECTNESS: was documented as `newSlowRangeQuery` (§2.1)** |
| `ValueSelector::reduce` | `SortedNumericSelector.Type.{MIN,MAX}` | identical (MIDDLE_* deferred) |
| `sort_by_numeric_doc_value` / `sort_top_n_by_numeric_doc_value` / `search_numeric_range_sorted_by_field` / `sort_by_multi_valued_doc_value` | `SortField.Type.LONG` + `TopFieldCollector` | identical |
| — | `FieldExistsQuery` | missing (recorded, §2.4) |
| — | `IndexOrDocValuesQuery` | missing (recorded, §2.5) |

### 2.1 [CORRECTNESS → fixed] `search_multi_valued_range` is not `newSlowRangeQuery`

Java's `SortedNumericDocValuesRangeQuery.getTwoPhaseIterator().matches()`:

```java
for (int i = 0, count = values.docValueCount(); i < count; ++i) {
  final long value = values.nextValue();
  if (value < lowerValue) continue;
  return value <= upperValue;   // values are sorted: first candidate decides
}
return false;
```

i.e. **a doc matches when any of its values is in range.** `SortedNumericSelector` is real Lucene's
*sort-field* reduction (`SortedNumericSortField`), never a range-query input. The module doc and
`search_multi_valued_range`'s own doc both claimed the selector-based filter was
`newSlowRangeQuery`'s equivalent; it isn't, and the two disagree on real data:

| doc values | query range | Java `newSlowRangeQuery` | `ValueSelector::Min` | `ValueSelector::Max` |
|---|---|---|---|---|
| `[1, 50]` | `[40, 60]` | **match** (50) | no match (1) | match (50) |
| `[5, 10]` | `[4, 6]` | **match** (5) | match (5) | no match (10) |
| `[1, 2, 3]` | `[2, 2]` | **match** (2) | no match | no match |

The third row is the sharpest: no selector can express "some middle value is in range".

**Fixed**: added `search_sorted_numeric_range`, a port of Java's loop including its early exit
(the first value at or above `min` decides the whole document, so a doc whose first candidate
already exceeds `max` costs one comparison, not `docValueCount`) and its `MatchNoDocsQuery`
treatment of `lower > upper`. `search_multi_valued_range` keeps its selector semantics and is
re-documented as the `SortedNumericSelector`-shaped filter it actually is, with the divergence
table above spelled out in its doc comment. 6 new tests against the real
`multi_valued_dv_index` fixture, one of which asserts the disagreement with both selectors
directly.

### 2.2 [INTENTIONAL] Ordinal resolution is the caller's job

`search_sorted_ord_range` compares already-known ordinals; Java's
`SortedSetDocValuesField.newSlowRangeQuery` takes `BytesRef` bounds and calls `lookupTerm`
internally. Deliberate and documented — `sorted_ord`'s contract is "just the ordinal", and the
terms-dictionary seek is already exposed separately.

### 2.3 [PERF — better than Java's fallback, worse than Java's best] skip-index range

`search_numeric_range_with_skip_index` consults only level 0 of the skip index; Java's
`DocValuesSkipper` descends coarser levels first, so a query that misses an entire top-level
interval skips it in one comparison instead of one per level-0 interval underneath it. Level 0
alone is *exact* (it decides skip-or-scan correctly for every interval), so this is purely a
constant factor on the skipping itself, never on the values decoded. Left as-is — it was already
recorded on the function, and the fix belongs with a full `DocValuesSkipper` port.

### 2.4 [MISSING — recorded] `FieldExistsQuery`

"Every doc with any value for this field" over doc values / norms / points / vectors. Reachable by
a caller and genuinely absent. Not added here: it spans four different readers (doc values, norms,
points, KNN vectors) and its natural home is a `Clause::FieldExists` variant in `query.rs`, which is
batch b12's file. Recorded for that batch rather than half-built here.

### 2.5 [MISSING — recorded] `IndexOrDocValuesQuery`

A query *planner* (choose the points index or the doc-values scan from `ScorerSupplier.cost()`).
Needs the cost estimates §1.4 records as absent and a `ScorerSupplier` abstraction this port has
no equivalent of. Recorded.

### Verdict

Swept clean on semantics. Two `MISSING` items (`FieldExistsQuery`, `IndexOrDocValuesQuery`)
recorded with reasons, both blocked on files owned by other batches.

---

## `crates/lucene-search/src/facets.rs`

Java counterparts (`lucene/facet/src/java/org/apache/lucene/facet/`):
- `Facets.java`, `FacetResult.java`, `LabelAndValue.java`, `FacetsCollector.java`, `TopOrdAndIntQueue.java`
- `sortedset/{AbstractSortedSetDocValueFacetCounts,SortedSetDocValuesFacetCounts,SortedSetDocValuesReaderState,DefaultSortedSetDocValuesReaderState}.java`
- `range/{RangeFacetCounts,LongRangeFacetCounts,DoubleRangeFacetCounts,LongRange,DoubleRange,Range}.java`
- `org/apache/lucene/util/NumericUtils.java`

| Rust `fn` | Java method | Verdict |
|---|---|---|
| `facet_counts` | `SortedSetDocValuesFacetCounts.countOneSegment` (no `OrdinalMap`) | identical for one segment |
| `resolve_labels` | `dv.lookupOrd` + `FacetsConfig.stringToPath` | divergent (no path splitting, §3.5) |
| `top_children` *(new)* | `AbstractSortedSetDocValueFacetCounts.getTopChildren` / `computeTopChildren` / `createFacetResult` | **added this batch** |
| `all_children` *(new)* | `AbstractSortedSetDocValueFacetCounts.getAllChildren` | **added this batch** |
| `top_n_facets` | *(kept as the raw ordering primitive; explicitly not `getTopChildren`)* | **CORRECTNESS (§3.1)** |
| `NumericRange::new_long` *(new)* | `LongRange(String, long, boolean, long, boolean)` | **added this batch** |
| `NumericRange::new_double` *(new)* | `DoubleRange(...)` + `DoubleRangeFacetCounts.getLongRanges`'s `toLongRange` | **added this batch** |
| `double_to_sortable_long` / `sortable_double_bits` *(new)* | `NumericUtils.doubleToSortableLong` / `sortableDoubleBits` | **added this batch** |
| `NumericRange::contains` | `LongRange.accept` | identical (after `new_long` normalization) |
| `range_facet_counts` | `RangeFacetCounts.count` (single-valued) + `getAllChildren` | identical for the single-valued path |
| `double_range_facet_counts` *(new)* | `DoubleRangeFacetCounts.count` (incl. `mapDocValue`) | **added this batch** |
| `top_range_children` *(new)* | `RangeFacetCounts.getTopChildren` | **added this batch** |
| — | `getSpecificValue`, `getAllDims`, `getTopDims` | missing (recorded, §3.6) |
| — | `OrdinalMap` / `SortedSetDocValuesReaderState` | missing (recorded, §3.7) |
| — | `RangeFacetCounts`'s multi-valued `startMultiValuedDoc`/`endMultiValuedDoc` | missing (recorded, §3.8) |

### 3.1 [CORRECTNESS → fixed] zero-count facets were reported

`computeTopChildren` only ever enqueues a child when `count > 0`, and `createFacetResult` returns a
`null` `FacetResult` when `childCount == 0`. `top_n_facets` sorted *every* ordinal and truncated, so
`top_n_facets(facets, 5)` over a dictionary with two used terms returned five entries — three of them
zeroes that Java would never emit. Callers rendering a facet sidebar would show empty buckets.

**Fixed**: `top_children` drops zero-count children, and returns `None` — not an empty list — when
no child had a non-zero count, which is how "a dim with no values in the matched set" is reported
(Java's `null`, filtered out by `getAllDims`). `top_n_facets` is kept and re-documented as the raw
ordering primitive it is, explicitly not `getTopChildren`.

### 3.2 [CORRECTNESS/verified] the tie-break was already right; `child_count`/`value` were absent

`TopOrdAndIntQueue.OrdAndInt.lessThan` is `value <`, then `ord >` — a min-heap whose *worst* entry is
the lowest count and, on a tie, the highest ordinal — so the best-first output is **count DESC,
ordinal ASC**. `top_n_facets` already sorted that way; verified and now pinned by a test that names
the Java rule.

What was missing is the rest of `FacetResult`: `value` (Java's `pathCount`, the sum over **every**
non-zero child, accumulated before the queue truncates) and `childCount` (how many children had a
non-zero count, which can exceed the returned list's length). Both are now returned by
`top_children`/`all_children`.

### 3.3 [MISSING → fixed] `validateTopN`

`Facets.validateTopN` throws `IllegalArgumentException("topN must be > 0 (got: 0)")`. `top_n_facets(_, 0)`
silently returned an empty list. `top_children`/`top_range_children` now panic with Java's exact
message — both are unchecked in Java too, and a `topN` of zero is a caller bug, not a runtime branch.

### 3.4 [MISSING → fixed] `LongRange` validation, and `DoubleRange` entirely

`NumericRange` had public fields and no constructor, so neither of Java's constructors existed:

- **`LongRange`** normalizes an exclusive bound to the equivalent inclusive one (`min++` / `max--`)
  and calls `failNoMatch()` — `IllegalArgumentException("range is empty: ...")` — for a bound already
  at `Long.MAX_VALUE`/`MIN_VALUE` and for any range whose normalized `min > max`. This port silently
  accepted `(42, 42]` as a bucket that always counts zero.
- **`DoubleRange`** was absent. The transform is not a cast: Java rejects `NaN`, nudges an exclusive
  bound with `Math.nextUp`/`Math.nextAfter(_, NEGATIVE_INFINITY)` **in double space, before** the
  integer transform (nudging afterwards is a different value around zero and in the subnormals),
  then applies `NumericUtils.doubleToSortableLong` to both ends. The counted doc values must go
  through `sortableDoubleBits` too (`DoubleRangeFacetCounts.mapDocValue`) or negative doubles compare
  as huge positives and land in the wrong buckets.

**Fixed**: `NumericRange::new_long`, `NumericRange::new_double`, the
`double_to_sortable_long`/`sortable_double_bits` helpers, and `double_range_facet_counts` (which
applies `mapDocValue`). Tested: the exclusive nudge lands strictly between the two representable
neighbours, the subnormal neighbourhood of zero comes out right, `sortable_double_bits` is an
involution *and* order-preserving across ±∞/±0/subnormals, and — the point of the whole thing —
a range pinned to the mapped form of a negative stored value counts through
`double_range_facet_counts` and counts *nothing* through `range_facet_counts`.

### 3.5 [MISSING → fixed] range `getTopChildren`, with its own tie-break

`RangeFacetCounts.getTopChildren` orders by count with the tie broken on **label** (its queue
comparator is `comparingInt(count).thenComparing(label, reverseOrder())`), drops zero-count ranges,
and — unlike the SORTED_SET case — returns a `FacetResult` even when every range is empty rather
than `null`. Only `getAllChildren`'s caller-order listing existed here.

**Fixed**: `top_range_children` + `RangeFacetResult`. `range_facet_counts`'s doc no longer claims
caller order is `getTopChildren`'s convention; it names it as `getAllChildren`'s, which it is.

### 3.6 [MISSING — recorded] `getSpecificValue` / `getAllDims` / `getTopDims`

All three need a notion of *dims* (`state.getDims()`, `FacetsConfig.getDimConfig`) that this port
doesn't have — it counts one field's ordinals, with no dim/path layer above it. `resolve_labels`
correspondingly returns the raw term string rather than
`FacetsConfig.stringToPath(term)[parts.length - 1]`. Recorded; a real fix is the `FacetsConfig`/
`SortedSetDocValuesReaderState` port, not a one-function addition.

### 3.7 [MISSING — recorded] `OrdinalMap` / cross-segment ordinals

Already documented at length in the module doc and unchanged by this batch: without a merged
ordinal map, summing raw ordinal counts across segments would conflate unrelated terms, so callers
must merge by resolved label. Correct and honest; the fix is a real `OrdinalMap` port.

### 3.8 [MISSING — recorded] multi-valued range counting

`RangeFacetCounts.count`'s multi-valued branch dedups consecutive equal values within a doc
(`if (j == 0 || val != previous)`) and counts `totCount` once per doc via
`endMultiValuedDoc()`. This port's range functions take a single-valued `NumericEntry` only.
Recorded.

### Verdict

Swept clean on everything reachable without a `FacetsConfig`/`OrdinalMap` port. Three `MISSING`
items recorded, all of them genuinely one layer up.

---

## `crates/lucene-search/src/highlighter.rs`

Java counterparts (`lucene/highlighter/src/java/org/apache/lucene/search/uhighlight/`):
`UnifiedHighlighter.java`, `FieldHighlighter.java`, `Passage.java`, `PassageScorer.java`,
`PassageFormatter.java`, `DefaultPassageFormatter.java`, `OffsetsEnum.java`,
`FieldOffsetStrategy.java`, `SplittingBreakIterator.java`.

| Rust `fn` | Java method | Verdict |
|---|---|---|
| `PassageScorer::{weight,tf,norm,score}` *(new)* | `PassageScorer.{weight,tf,norm,score}` | **added this batch, formulas verbatim** |
| `assemble_fragments` | `FieldHighlighter.highlightOffsetsEnums` | **CORRECTNESS on selection (§4.2)**; boundary chooser divergent by design (§4.5) |
| `render_cluster` | `DefaultPassageFormatter.format`'s per-passage body | **CORRECTNESS on overlapping matches (§4.3)** |
| `format_fragments` *(new)* | `DefaultPassageFormatter.format`'s outer loop (ellipsis) | **added this batch** |
| `append_escaped` *(new)* | `DefaultPassageFormatter.append`'s `escape` branch | **added this batch** |
| `sentence_start_offsets` + friends | `BreakIterator.getSentenceInstance` | divergent, INTENTIONAL (§4.5) |
| `char_offset_to_byte` / `byte_offset_to_char` | *(no Java counterpart — Java indexes `String` by UTF-16 unit)* | not-in-Java |
| — | `SplittingBreakIterator`, `PhraseHelper`, `FieldOffsetStrategy`, `getSummaryPassagesNoHighlight` | missing (recorded, §4.6) |

### 4.1 [MISSING → fixed] `PassageScorer` did not exist

Ported verbatim, including every default and the exact expression shapes:

- `k1 = 1.2`, `b = 0.75`, `pivot = 87`.
- `weight(contentLength, ttf) = (k1 + 1) * ln(1 + (numDocs + 0.5) / (ttf + 0.5))`,
  `numDocs = 1 + contentLength / pivot` — accumulated in `double` and cast once, as Java does.
- `tf(freq, passageLen) = freq / (freq + k1 * ((1 - b) + b * (passageLen / pivot)))` — the
  saturating term frequency, length-normalized against `pivot`, **not** an average field length.
- `norm(passageStart) = 1 + 1 / ln(pivot + passageStart)` — earlier passages score higher.
- `score = norm(startOffset) * Σ over each **distinct** matched term of
  tf(freqInThisPassage, passageLength) * weight(contentLength, freqInWholeDoc)`.

Java reads the whole-document frequency from `OffsetsEnum.freq()`; here it is derived by counting
the term's occurrences across *every* supplied span, which is the same number for spans produced by
`term_vectors_query::matched_term_offsets` (it emits every occurrence of every matched term).
Documented on `PassageScorer::score`.

Tested against the algebra directly (each formula re-derived in the test), plus the saturation and
length/position monotonicity properties the formulas are supposed to have.

### 4.2 [CORRECTNESS → fixed] `max_fragments` kept the *first* N, not the *best* N

`FieldHighlighter.highlightOffsetsEnums` scores every candidate passage, keeps the best
`maxPassages` in a priority queue (`maybeAddPassage`), and only then sorts the survivors back into
document order for rendering (`Arrays.sort(passages, passageSortComparator)`).

This port emitted clusters left-to-right and called `.take(max_fragments)` — so a document whose
densest, most relevant passage sits late silently lost it in favour of a sparse early one. That is a
wrong *result*, not a formatting difference: it is the snippet the user reads.

**Fixed**: every cluster is scored, selection is score-descending with Java's start-offset tie-break,
and the survivors are re-sorted by offset before rendering. `Fragment` now carries `score`,
`start_offset` and `end_offset` (`Passage.getScore/getStartOffset/getEndOffset`, in the same `char`
unit `TermOffsetSpan` uses). Tested with a document whose dense cluster is last: uncapped, it scores
higher; capped to one fragment, it is the one that survives.

### 4.3 [CORRECTNESS → fixed] overlapping matches produced nested markers

`DefaultPassageFormatter.format` coalesces any run of overlapping matches into a *single*
`pre`/`post`-wrapped span:

```java
while (i + 1 < passage.getNumMatches() && passage.getMatchStarts()[i + 1] < end) {
  end = Math.max(end, passage.getMatchEnds()[++i]);
}
```

`render_cluster` inserted markers back-to-front, one pair per match, with no overlap handling — so
two matches at the same start (`cat` and `cats` from a synonym or n-gram field) rendered as
interleaved, unbalanced markup. It also had no place to escape the content.

**Fixed**: `render_cluster` is now a forward walk, a port of Java's loop, with the coalescing rule
and the escaping hook. Tested: fully-nested (`cat` inside `cats`) renders `<b>cats</b> sleep`, and
partially-overlapping spans extend to the furthest end.

### 4.4 [MISSING → fixed] ellipsis and HTML escaping

`DefaultPassageFormatter`'s remaining two behaviours were absent:
- `ellipsis` (default `"... "`), emitted between two passages that aren't contiguous in the original
  text (`sb.isEmpty() == false && passage.getStartOffset() != pos`). Added as
  `FragmentConfig::ellipsis` + a new `format_fragments`, which is also the first thing in this port
  that produces `UnifiedHighlighter.highlight`'s single-string output.
- `escape`, escaping exactly `& < > " ' /` to `&amp; &lt; &gt; &quot; &#x27; &#x2F;` in the content
  while leaving the markers raw. Added as `FragmentConfig::escape`, applied inside `render_cluster`
  (it has to be, since the markers must survive it). Tested character-by-character.

### 4.5 [INTENTIONAL] the boundary chooser is still not `BreakIterator`

Unchanged and still documented as a simplification: fixed `window_chars` windows snapped to
whitespace, or the opt-in narrow sentence heuristic. This is the one place the module deliberately
isn't Java. The module doc now states precisely which parts *are* verbatim (scorer, selection,
formatter) so the remaining gap reads as the single scoped item it is rather than a blanket
"simplified highlighter".

### 4.6 [MISSING — recorded] the offset-strategy layer

`FieldOffsetStrategy`/`PhraseHelper`/`OffsetsEnum` (deciding *where* offsets come from: postings,
term vectors, or re-analysis; and restricting matches to actual phrase hits) have no counterpart —
this port's entry point takes already-computed spans. `SplittingBreakIterator` and
`getSummaryPassagesNoHighlight` (the "no match, show the first N sentences" fallback) are likewise
absent. All three sit above this module's input contract. Recorded.

### Verdict

Two `CORRECTNESS` fixes (passage selection, overlapping markers), two `MISSING` fixes (`PassageScorer`,
the rest of `DefaultPassageFormatter`). The `BreakIterator` gap stays, now precisely scoped.

---

## `crates/lucene-search/src/term_vectors_query.rs`

Java counterparts: `org/apache/lucene/index/IndexReader.getTermVector(int, String)`,
`org/apache/lucene/search/uhighlight/OffsetsEnum.java` (for the ordering contract).

| Rust `fn` | Java method | Verdict |
|---|---|---|
| `term_vector_for_doc` | `IndexReader.getTermVector(doc, field)` | identical (incl. all three `null` cases) |
| `matched_term_offsets` | *(no single Java method; the primitive `TermVectorOffsetStrategy` feeds `OffsetsEnum` with)* | **CORRECTNESS on ordering (§5.1)** |

### 5.1 [CORRECTNESS → fixed] span ordering was under-specified

`OffsetsEnum.compareTo` orders by `startOffset`, then `endOffset`, then the term itself.
`matched_term_offsets` sorted on `start_offset` alone, leaving the order of two terms starting at
the same offset dependent on terms-dictionary order — the synonym / overlapping-token case, which is
exactly where a highlighter must be deterministic (and, after §4.3, where the coalescing rule now
depends on the order). Now sorts on all three keys; tested with three terms sharing a start offset.

### Verdict

Swept clean. `matched_term_offsets` is a Rust-side primitive with no single Java counterpart; its
*ordering* contract now matches the Java type that consumes it.

---

## `crates/lucene-search/src/explain.rs`

Java counterparts: `org/apache/lucene/search/Explanation.java`, `IndexSearcher.explain`,
`Weight.explain`, `TermQuery.TermWeight.explain`, `PhraseWeight.explain`, `BooleanWeight.explain`,
`DisjunctionMaxQuery.DisjunctionMaxWeight.explain`, `ConstantScoreWeight.explain`, `BoostQuery.java`,
`similarities/BM25Similarity.java` (`idfExplain`, `BM25Scorer.explain`, `explainTF`).

*(Re-read `similarity.rs` after b12's edits before pinning these; `DEFAULT_K1`/`DEFAULT_B`/`idf`/
`norm_inverse`/`do_score`/`decode_norm` are unchanged and are what the fixed strings are built on.)*

| Rust `fn` | Java method | Verdict |
|---|---|---|
| `Explanation::{match_,no_match,with_details}` | `Explanation.{match,noMatch}` | identical |
| `Display for Explanation` *(new)* | `Explanation.toString()` | **added this batch** |
| `describe_clause` / `describe_span` *(new)* | each query class's `toString()` | **added this batch** |
| `explain_clause` | `IndexSearcher.explain` / `Weight.explain` dispatch | identical shape |
| `explain_term` | `TermWeight.explain` + `BM25Scorer.explain`/`explainTF`/`idfExplain` | **CORRECTNESS: 5 divergent strings (§6.1)** |
| `explain_phrase` | `PhraseWeight.explain` + the same BM25 pair | **CORRECTNESS: 4 divergent strings (§6.2)** |
| `explain_boolean` | `BooleanWeight.explain` | **CORRECTNESS: description + 3 missing no-match branches (§6.3)** |
| `explain_dismax` | `DisjunctionMaxWeight.explain` | **CORRECTNESS: description + tie-breaker switch (§6.4)** |
| `explain_constant_score` | `ConstantScoreWeight.explain` | **CORRECTNESS (§6.5)** |
| `explain_flat_match` | `ConstantScoreWeight.explain` (post-`MultiTermQuery` rewrite) | **CORRECTNESS (§6.5)** |
| `explain_boost` | *(none — `BoostQuery` has no `Weight` of its own)* | divergent, INTENTIONAL (§6.6) |
| `clause_matches` | `Weight.scorer(...).iterator().advance(doc) == doc` | identical |

### 6.1 [CORRECTNESS → fixed] every BM25 term-explanation string differed

Downstream tooling parses these, so each is a compatibility break. Java (10.5.0) → what this port
emitted:

| Java | was |
|---|---|
| `weight(body:cat in 0) [BM25Similarity], result of:` | `weight(body:cat), result of:` |
| `score(freq=2.0), computed as boost * idf * tf from:` | `score(freq=2), computed as idf * tfNorm from:` |
| `idf, computed as log(1 + (N - n + 0.5) / (n + 0.5)) from:` | `idf, computed as log(1 + (docCount - docFreq + 0.5) / (docFreq + 0.5)) from docFreq=2, docCount=8` |
| `n, number of documents containing term` | `docFreq, number of documents containing term` |
| `N, total number of documents with field` | `docCount, total number of documents with field` |
| `tf, computed as freq / (freq + k1 * (1 - b + b * dl / avgdl)) from:` | `tfNorm, computed as freq / (freq + k1 * (1 - b + b * fieldLength / avgFieldLength)) from:` |
| `dl, length of field` / `dl, length of field (approximate)` | `fieldLength` |
| `avgdl, average length of field` | `avgFieldLength` |
| `no matching term` (all three miss cases) | `no matching term, field 'x' not found` etc. |

All fixed. Three details worth naming:

- **`freq` renders as a Java `Float`.** Java's `freq` local is a `float`, so `freq.getValue()` prints
  `2.0`. Rust's `Display` for `f32` prints `2`. Added a `java_float` helper (`{:?}`, which keeps the
  decimal point and picks the same shortest round-trip representation) used for every number that
  goes into a description.
- **`dl, length of field (approximate)`** is chosen by Java from the *encoded* norm byte
  (`(norm & 0xFF) > 39`). This port only carries the decoded length, and `FieldNorms` (batch b13's
  file) exposes no raw-norm accessor — so `dl_description` uses `field_length > 39.0`, which is
  exactly equivalent because `BM25Similarity.LENGTH_TABLE[i] == i` for every `i <= 39` and the table
  is monotonically non-decreasing. A test asserts **both** properties against this port's own
  `decode_norm` over all 256 byte values *and* that the predicate flips at the same index Java's
  does, so a table change fails here instead of drifting silently.
- The lost diagnostic detail in the no-match strings is deliberate: Java is the spec, and
  `TermWeight.explain` returns a bare `Explanation.noMatch("no matching term")` for a missing field,
  a missing term and a non-matching doc alike.

### 6.2 [CORRECTNESS → fixed] phrase explanations

`PhraseWeight.explain` builds `Explanation.match(freq, "phraseFreq=" + freq)` and hands it to the
*same* `BM25Scorer.explain`/`explainTF`, so the phrase tree is the term tree with a different freq
leaf. Fixed: root is `weight(pos:"alpha beta" in 8555) [BM25Similarity], result of:`, the idf node is
`idf, sum of:` with one Java-format child per term (was `idf, sum of each phrase term's own idf,
from:` with `idf(alpha), docFreq=.., docCount=..` children), the tf leaf is `phraseFreq=1.0` (was a
sentence describing the matcher), and the no-match strings are Java's `no matching terms` /
`no matching phrase`.

### 6.3 [CORRECTNESS → fixed] boolean explanations

- Description was `"{value} = sum of:"`; Java's is a bare `"sum of:"` (the value lives in the node,
  and `Explanation.toString` re-renders it — emitting it twice broke the format).
- Only one of Java's **four** outcomes existed. Now all four, with Java's exact wording:
  `Failure to meet condition(s) of required/prohibited clause(s)` (a required clause missed or a
  prohibited one matched), `No matching clauses` (`matchCount == 0`),
  `Failure to match minimum number of optional clauses: {mm}, matched: {n}`, and the matching
  `sum of:`.
- Sub-explanations were absent on every no-match. Java attaches them, wrapped:
  `no match on required clause ({query})`, `no match on optional clause ({query})`,
  `match on prohibited clause ({query})` — the failing optionals appended only in the
  `No matching clauses` / minimum-should-match branches, as Java does.
- `must_not` clauses were never explained at all; they are now.

### 6.4 [CORRECTNESS → fixed] dismax explanations

Java switches on the tie-breaker: `"max of:"` when it is `0.0f`, else
`"max plus {tie} times others of:"`. This port always emitted
`"{value} = max of:, plus {tie} times others of:"` — a string matching neither. No-match is Java's
`No matching clause` (singular) with the non-matching disjuncts attached, and Java collects those
sub-explanations only while no disjunct has matched yet; both reproduced.

### 6.5 [CORRECTNESS → fixed] constant-score and flat multi-term clauses

`ConstantScoreWeight.explain` is
`Explanation.match(score, getQuery().toString() + (score == 1f ? "" : "^" + score))`, no-match
`getQuery().toString() + " doesn't match id " + doc`. This port emitted
`"{score} = ConstantScore, discarding the wrapped clause's own score"` / `"no matching clause"`, and
`explain_flat_match` (wildcard/prefix/fuzzy/regexp/span/match-all/term-in-set — all of which Java
rewrites to a constant-scoring weight) emitted `"1.0 = matches, unscored constant score"`.

Fixed, which required knowing what each query prints. Added `describe_clause`/`describe_span`,
mirroring each Java class's `toString()`: `field:term`, `field:"a b"~3`, `(+a b -c)~2`,
`(a | b)~0.3`, `ConstantScore(x)`, `(x)^2.0`, `field:pfx*`, `field:t~2`, `field:/re/`, `*:*`,
`MatchNoDocsQuery("...")`, `field:(a b)`, `field:[1 TO 9]`,
`spanNear([spanTerm(f:a), spanTerm(f:b)], 2, true)`.

### 6.6 [INTENTIONAL] `Clause::Boost` explains as its own node

Real `BoostQuery` has no `Weight`: `createWeight` multiplies the boost into the wrapped weight, so
for a term it surfaces as a `boost` child of `score(freq=..), computed as boost * idf * tf from:`.
This port's `clause_scores` multiplies *after* the fact, so an explanation shaped like Java's would
no longer equal this port's own score — and "the explanation equals the score" is the invariant
`explain.rs` is built on and tests directly. Kept as a `product of:` node (with the `"{value} = "`
prefix removed for consistency), and recorded here and in `docs/parity.md`.

### 6.7 [MISSING → fixed] `Explanation.toString()`

Java renders an explanation as one `"{value} = {description}"` line per node, two spaces of indent
per level, `\n`-terminated — the format tools parse. There was no `Display`. Added, with a test
pinning the exact layout including the trailing newline.

### Verdict

Swept clean; every description string is now Java's verbatim, pinned by six dedicated tests
(term, phrase, boolean, dismax, constant-score/flat, `describe_clause`) plus the `Display` and
`java_float`/`dl_description` tests. One `INTENTIONAL` divergence recorded (`Clause::Boost`).

---

## `crates/lucene-search/src/query_parser.rs`

Java counterparts (`lucene/queryparser/src/java/org/apache/lucene/queryparser/classic/`):
`QueryParser.jj` (the grammar), `QueryParserBase.java`, `MultiFieldQueryParser.java`; plus
`flexible/standard/StandardQueryParser.java`.

| Rust `fn` | Java production / method | Verdict |
|---|---|---|
| `parse_query` / `parse_query_with_analyzer` | `QueryParser.parse` / `TopLevelQuery` | identical |
| `parse_query_with_operator` *(new)* | `QueryParserBase.setDefaultOperator` | **added this batch** |
| `parse_clause_list` | `Query(field)` | **MISSING: no conjunctions (§7.1)** |
| `add_clause` *(new)* | `QueryParserBase.addClause` | **added this batch, truth table verbatim** |
| `parse_conjunction` *(new)* | `Conjunction()` (`<AND: ("AND"\|"&&")>`, `<OR: ("OR"\|"\|\|")>`) | **added this batch** |
| `parse_modifiers` *(new)* | `Modifiers()` (`<PLUS>`, `<MINUS>`, `<NOT: ("NOT"\|"!")>`) | **added this batch** (`NOT`/`!` were missing) |
| `parse_boosted_atom` | `Term()`'s `[ <CARAT> boost [ fuzzySlop ] \| fuzzySlop [ <CARAT> boost ] ]` | **MISSING: one order only (§7.3)** |
| `parse_tilde_suffix` *(new)* | `<FUZZY_SLOP>` + `handleBareFuzzy` + `handleQuotedTerm` | **added this batch** |
| `parse_phrase` | `<QUOTED>` + `handleQuotedTerm` | **MISSING: slop rejected (§7.2)** |
| `parse_wordterm` | `handleBareTokenQuery` | **CORRECTNESS: fuzzy-vs-wildcard precedence (§7.4)** |
| `parse_range` / `parse_range_bound` | `Term()`'s range production + `getRangeQuery` | divergent, INTENTIONAL (numeric only) |
| `parse_regexp` | `<REGEXPTERM>` + `getRegexpQuery` | identical (goes through b8's `RegexpQuery`/`RegexpPattern`, §7.6) |
| `parse_group` | `Clause()`'s `<LPAREN> Query <RPAREN>` | identical |
| `try_parse_field` | `Clause()`'s `LOOKAHEAD(2) (<TERM> <COLON>)` | identical |
| — | `MultiFieldQueryParser` | missing (recorded, §7.7) |
| — | `allowLeadingWildcard` guard | missing (recorded, §7.8) |
| — | `TermRangeQuery` string/date ranges | deferred (pre-existing, unchanged) |

### 7.1 [MISSING → fixed] `AND` / `OR` / `NOT` were not operators at all

The module doc previously declared them out of scope: "`AND`/`OR`/`NOT` tokens are treated as
ordinary bare terms". That is a real, reachable behaviour difference — `cat AND dog` parsed as
*three* optional terms, so it matched documents containing only `dog`, and silently searched for the
literal token `AND`.

**Fixed** as a port of `QueryParserBase.addClause`, including both of its retroactive rewrites of the
**preceding** clause, which are the non-obvious part:

- an `AND` promotes the previous clause to `MUST` unless it was prohibited — this is what makes
  `a AND b` come out `+a +b` rather than `a +b`;
- an `OR`, **under `AND_OPERATOR` only**, demotes the previous clause to `SHOULD`, so `a OR b` is
  `a b` rather than `+a b`.

`&&`/`||`/`!` aliases, `NOT` as a modifier identical to `-`, and Java's
`clauses.size() == 1 && firstQuery != null` unwrapping (a single unmodified clause is returned bare,
not as a one-clause `BooleanQuery`) are all reproduced.

Tokenization matches JavaCC's longest-match-then-first-declared-rule behaviour: the word forms are
operators only when the whole bareword token is exactly `AND`/`OR`/`NOT`, so `ANDROID`, `NOTHING`
and lowercase `and` stay terms. A trailing conjunction is a clean `UnexpectedEnd` error; a *leading*
one lexes as a term, because Java's `Query()` production only admits a conjunction from the second
clause onward.

### 7.2 [MISSING → fixed] `DefaultOperator`

`setDefaultOperator` had no counterpart. Added `DefaultOperator` + `parse_query_with_operator`;
`parse_query`/`parse_query_with_analyzer` keep Java's `OR` default, so no existing caller changes
behaviour.

### 7.3 [MISSING → fixed] phrase slop `"a b"~n` was a parse error

`handleQuotedTerm` reads the same `<FUZZY_SLOP>` token after a `<QUOTED>` and uses it as the phrase
slop: `(int) Float.parseFloat(image.substring(1))`, wrapped in a `catch` that leaves the default.
This port only ever treated `~` as a fuzzy marker on a bareword, so `"quick fox"~3` failed to parse
outright — one of the most commonly written classic-syntax queries.

**Fixed**: `PhraseQuery::slop` is set, a fractional slop truncates (`~1.9` → `1`, Java's `(int)`
cast), and a bare `~` leaves the default `0`.

### 7.4 [MISSING → fixed] boost and `~` in only one order

Java's `Term()` accepts `[ <CARAT> boost [ fuzzySlop ] | fuzzySlop [ <CARAT> boost ] ]`, i.e. either
order, at most one of each. `~` was consumed inside `parse_wordterm` and `^` afterwards, so
`term~1^2` worked but `term^2~1` (and both phrase forms) did not. **Fixed** by lifting the `~`
handling to the atom level as `parse_tilde_suffix`; tested that the two orders produce identical
clauses for both fuzzy and phrase-slop.

### 7.5 [CORRECTNESS → fixed] `~` after a wildcard/prefix was applied instead of ignored

`handleBareTokenQuery` tests `wildcard`, then `prefix`, then `regexp`, and only then `fuzzy`. So
`ca*~2` is a `PrefixQuery` in Java and the `~2` is discarded. This port checked fuzzy *first*, so
`ca*~2` became `FuzzyQuery("ca*")` — a search for the literal `*` character within edit distance 2,
not a prefix search. Fixed, with a test for both the prefix and the interior-wildcard shape.

Lifting `~` out of `parse_wordterm` also required keeping the **unanalyzed** term text around (a new
internal `BareToken`), because Java's `getFuzzyQuery` never runs the analyzer over a fuzzy term while
`getFieldQuery` does — the previous code got this right by accident of ordering and would have
regressed otherwise.

### 7.6 [verified] regexp goes through b8's grammar

`parse_regexp` builds a `Clause::Regexp`, resolved by `crate::regexp_doc_ids` →
`lucene_codecs::regexp::RegexpPattern` — batch b8's rewrite against Lucene's real grammar. The
parser's only regexp-specific lexing is `\/` (escaping a literal slash so it doesn't close the
pattern), and it deliberately leaves every other backslash escape intact for `RegexpPattern` to
interpret. Correct as-is; confirmed, not changed.

### 7.7 [MISSING — recorded] `MultiFieldQueryParser`

Fans a bare term out across several fields as a disjunction, with per-field boosts. One default
field only here. Recorded — it is a wrapper *around* this parser, not a change to it, and belongs in
its own slice.

### 7.8 [INTENTIONAL] no `allowLeadingWildcard` guard

Real `QueryParserBase` throws `"'*' or '?' not allowed as first character in WildcardQuery"` unless
the flag is set — a guard against `MultiTermQuery`'s full-dictionary scan. This port always allows
it; `lucene_codecs::wildcard::WildcardPattern` has no equivalent cost cliff being guarded against
here. Recorded in the module doc's deferred list.

### 7.9 [INTENTIONAL] numeric-only ranges

`[a TO b]` builds a `Clause::PointsRange` over `i64`, not Java's `TermRangeQuery` over `BytesRef`;
a non-numeric bound is a clean `InvalidRangeBound` error rather than a silent string comparison.
Pre-existing, already documented, unchanged. The exclusive-bound `±1` adjustment matches
`getRangeQuery`'s.

### Verdict

Swept clean on the grammar this module claims. Four `MISSING` and one `CORRECTNESS` finding fixed
(operators, default operator, phrase slop, suffix ordering, fuzzy-vs-wildcard precedence);
`MultiFieldQueryParser`, the leading-wildcard guard and string ranges recorded.

---

## Cross-file notes

- **Coordination.** `similarity.rs` (b12) was re-read before pinning `explain.rs`'s strings;
  `field_norms.rs` (b13) was read but **not** modified — §6.1 works around its missing raw-norm
  accessor with a table property proven by test rather than by editing another batch's file. No file
  outside this batch was changed except `docs/parity.md` (append-only, per invariant #7).
- **Public API changes** (all additive except one): `highlighter::Fragment` gained `score`,
  `start_offset`, `end_offset` and lost `Eq` (it now holds an `f32`); `lucene-ffi`, its only
  downstream consumer, reads `.text`/`.matched_terms` only and its 421 tests pass unchanged.
  `FragmentConfig` gained three fields — the one in-tree literal construction was switched to
  `..Default::default()`.
