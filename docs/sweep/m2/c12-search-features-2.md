# c12 — search features, round 2 (`lucene-search`: b14's remaining gaps + two carry-overs)

Java source of truth: `/home/tuong/work/lucene` (Lucene 10.5.0), plus the JDK
itself for `java.text.BreakIterator` (§3.1).

Follow-up batch closing the feature gaps `b14-search-features` recorded, the
dead-code item `c11-occur-filter` raised, `c1-lazy-blocktree`'s F-13, and one
item handed over mid-batch by `c8-tv-chunking`.

**Findings: 2 CORRECTNESS, 14 MISSING (11 fixed, 3 recorded with named
blockers), 4 PERF (3 fixed and measured, 1 recorded with a named blocker),
4 INTENTIONAL**, plus one re-assessment (§5.1). Every
CORRECTNESS finding and every fixable MISSING finding is fixed with tests;
every PERF claim is measured with the "before" re-run, not quoted.

Two new Java fixture generators: `fixtures/src/GenFacets.java` (a three-segment
faceted index + real `lucene-facet` answers) →
`crates/lucene-search/tests/facets_fixtures.rs` (18 tests), and
`fixtures/src/GenBreakIterator.java` (the JDK's own sentence boundaries) →
`highlighter::sentence_boundaries`' test. Plus
`crates/lucene-search/tests/highlighter_offsets_fixtures.rs` (3 tests) against
`GenBlockTree.java`'s existing per-occurrence offsets ground truth.

---

## 1. `crates/lucene-search/src/lib.rs` — the dead MAXSCORE body

Java: `BooleanScorerSupplier` (`TOP_SCORES`), `WANDScorer`.

### 1.1 [PERF → fixed by deletion] `search_boolean_query_scored_maxscore_with_stats`' body was unreachable

c11's F-7 claim, verified two ways before acting on it.

**By reading both predicates.** The function's first act is
`try_disjunction_lazy(...)?` and return-if-handled. The two gates are exactly
complementary:

| shape | `try_disjunction_lazy` | the body |
|---|---|---|
| `must`/`filter`/`must_not` non-empty, `minimumShouldMatch > 1`, `should` empty | declines | falls back |
| a non-`Clause::Term` clause | declines | falls back |
| `doc_in` absent | declines | falls back |
| a pulsed `docFreq <= 1` term | declines | falls back |
| **field absent / term absent / `lazy_postings` → `None`** | **accepts** (drops the clause from the union) | would have fallen back — but is never reached |
| `lazy_postings` → `Err(Unsupported)` | propagates with `?` | its fallback arm is never reached |

and, after its setup loop, `try_disjunction_lazy` never returns `Ok(false)`
again — it returns `Ok(true)` at exhaustion. So control reaching the body
implies the body immediately falls back.

**By coverage.** `cargo llvm-cov -p lucene-search` before this batch: all 78
executable lines of the body uncovered by the whole 899-test suite, including
the six `boolean_maxscore_falls_back_*` tests written specifically to drive its
own fallback arms. (Lines 3252–3424 of the pre-batch file; the only covered
line in the function was the `try_disjunction_lazy` call itself.)

**Deleted, not revived**, and the reason is in the deleted code's own doc
comment: it recorded itself as 4–5x *slower* than the lazy union on M1's
5M-document corpus (655 ms vs 163 ms on `t0 OR t1`) and said "prefer the plain
scored entry point". `try_disjunction_lazy` has since grown real block-max
pruning, so reviving the body would mean routing queries to the slower of two
pruning implementations.

The two public entry points keep their names, signatures and behaviour — the
behaviour was already "`try_disjunction_lazy`, else the exhaustive path", which
is precisely `search_boolean_query_scored_with_stats` (its
`try_conjunction_lazy` attempt declines every shape `try_disjunction_lazy`
accepts and vice versa: the gates are mutually exclusive on
`query.should.is_empty()`). `lucene-ffi`'s four exported functions and
`multi_segment`'s two fan-outs are unaffected.

Test: `boolean_maxscore_agrees_with_the_plain_scored_path_on_every_shape` — a
10-shape × 4-`top_n` matrix asserting byte-identical `TopDocsCollector` output
against the plain scored path, including the absent-field and absent-term
shapes that pinned the deadness.

### Verdict

~180 lines gone, coverage hole closed at the source rather than papered over.

---

## 2. `crates/lucene-search/src/facets.rs` + `src/ordinal_map.rs` — the missing layer

Java counterparts:
- `lucene/facet/src/java/org/apache/lucene/facet/FacetsConfig.java`,
  `Facets.java`, `FacetResult.java`, `LabelAndValue.java`
- `facet/sortedset/{SortedSetDocValuesReaderState,DefaultSortedSetDocValuesReaderState,AbstractSortedSetDocValueFacetCounts,SortedSetDocValuesFacetCounts}.java`
- `facet/range/{RangeFacetCounts,LongRangeFacetCounts,DoubleRangeFacetCounts,LongRangeCounter}.java`
- `lucene/core/src/java/org/apache/lucene/index/OrdinalMap.java`

| Rust item | Java | Verdict |
|---|---|---|
| `OrdinalMap::build` | `OrdinalMap.build(owner, TermsEnum[], weights, ratio)` | **added**, representation divergent (§2.6) |
| `OrdinalMap::{value_count, global_ord, segment_ords, first_segment, first_segment_ord}` | `getValueCount`/`getGlobalOrds`/`getFirstSegmentNumber`/`getFirstSegmentOrd` | **added** |
| `merge_segment_counts` | `SortedSetDocValuesFacetCounts.countOneSegment`'s remap step | **added** |
| `FacetsConfig`/`DimConfig` + setters | `FacetsConfig`/`DimConfig` + setters | **added** |
| `path_to_string`/`path_components_to_string`/`string_to_path` | `FacetsConfig.pathToString`(×3)/`stringToPath` | **added, verbatim** |
| `FacetsState::new` | `DefaultSortedSetDocValuesReaderState`'s constructor | **added** |
| `FacetsState::build_one_flat_dim` | `createOneFlatFacetDimState` | **added** |
| `FacetsState::build_one_hierarchical_dim` + `DimTree` | `createOneHierarchicalFacetDimState` + `DimTree` | **added** |
| `FacetsState::{dims,ord_range,dim_tree,size,label,lookup_term}` | `getDims`/`getOrdRange`/`getDimTree`/`getSize`/`lookupOrd`/`lookupTerm` | **added**; `dims` order divergent (§2.7) |
| `SortedSetFacetCounts::top_children` | `getTopChildren(topN, dim, path...)` | **added** |
| `SortedSetFacetCounts::all_children` | `getAllChildren(dim, path...)` | **added** |
| `SortedSetFacetCounts::specific_value` | `getSpecificValue(dim, path...)` | **added** |
| `SortedSetFacetCounts::all_dims` | `getAllDims(topN)` | **added** |
| `SortedSetFacetCounts::top_dims` | `Facets.getTopDims` (base contract) | **added**, §2.8 |
| `adjust_path_count` | `adjustPathCountIfNecessary` | **added, all three cases** |
| `multi_valued_range_facet_counts` / `_double_` | `RangeFacetCounts.count`'s multi-valued branch | **added** |
| `range_facet_counts_with_total` / `double_..._with_total` | `RangeFacetCounts.totCount` | **added** |
| `FacetResult.{dim,path}` and `value: i64` | `FacetResult` | **CORRECTNESS (§2.3)** |
| — | `FacetsConfig.build(Document)` | recorded (write path, §2.9) |
| — | `ramBytesUsed`/`getChildResources`/`cachedOrdMap` | recorded (§2.9) |

### 2.1 [MISSING → fixed] `OrdinalMap`: cross-segment faceting was *unavailable*, and the b14 doc's workaround was the only option

b14 §3.7 recorded this and left the question open. Establishing which: it was
**unavailable, not wrong** — `facets.rs` counted one segment, and its module
doc told callers to merge per-segment results *by resolved label*. Nothing in
the port summed raw ordinals across segments, so no caller could get a wrong
count from the code as it stood; they simply could not get a cross-segment
count from this module at all.

That is now closed. `OrdinalMap::build` merge-sorts every segment's term list
into one global, still-sorted dictionary (a `BinaryHeap` k-way merge, ties
breaking to the lowest segment index so `first_segment` is well-defined) and
records `segmentOrd -> globalOrd` per segment; `merge_segment_counts` is
`counts[(int) ordMap.get(ord)] += count`, `countOneSegment`'s own line.

The fixture makes the stakes concrete rather than rhetorical: over the
three-segment `facets_index`, segment 0's ordinal 6 is `Publish Year` and
global ordinal 11 is the same term — an elementwise sum of the per-segment
count arrays does not even have the right *length*
(`summing_raw_per_segment_counts_would_conflate_unrelated_terms`).

Pinned against Lucene's own `MultiDocValues.getSortedSetValues(...).mapping`:
the local→global table for all three segments, and the global dictionary
rebuilt through `first_segment`/`first_segment_ord`, both byte-for-byte.

### 2.2 [MISSING → fixed] `FacetsConfig` and the dim/path encoding

`DimConfig` (`hierarchical`, `multiValued`, `requireDimCount`,
`indexFieldName`), `getDimConfig`'s default fallback, `isDimConfigured`, the
four setters, `DEFAULT_INDEX_FIELD_NAME` (`"$facets"`), `DELIM_CHAR`
(`U+001F`), `ESCAPE_CHAR` (`U+001E`), and `pathToString`/`stringToPath`
verbatim — including `pathToString`'s rejection of an empty component
(`EmptyPathComponent`, Java's `IllegalArgumentException`), `stringToPath`'s
`length == 0` early return, and its total handling of a trailing escape.

Java's own note is why the type has to exist and is quoted on it: the config is
*not stored in the index*, and a search-time config that disagrees with the
indexing-time one produces wrong counts rather than an error.

### 2.3 [CORRECTNESS → fixed] `FacetResult` had no `dim`/`path`, and `value` could not be `-1`

`FacetResult` is `(dim, path, value, labelValues[], childCount)` and `value` is
a `Number` because `adjustPathCountIfNecessary` returns **`-1`** for a
multi-valued dim without `requireDimCount` — "no accurate count is obtainable",
because summing children double-counts a document carrying two values of the
dim. b14's `FacetResult` had `value: u64` and documented the caveat in prose,
leaving a caller to read a plausible sum where Lucene reports `-1`.

Fixed: `dim`/`path` added, `value` is `i64`, and `adjust_path_count` implements
all three of Java's cases (hierarchical → the path ordinal's own count;
multi-valued **with** `requireDimCount` → the dim ordinal's own count;
multi-valued without → `-1`; single-valued → the computed sum). The dim-less
primitives `top_children`/`all_children` keep working with an empty
`dim`/`path` and the plain sum, documented as such.

The fixture pins it from real Lucene: `Tag` is multi-valued without
`requireDimCount` and Lucene reports `value = -1` where its three children sum
to 9 — a number that would have looked entirely plausible.

### 2.4 [MISSING → fixed] the dim layer: `getAllDims`, `getSpecificValue`, hierarchical paths

`FacetsState` is `DefaultSortedSetDocValuesReaderState`'s constructor over one
global ordinal space: it parses every term into dim + path and records either
the dim's contiguous `OrdRange` (flat) or its `DimTree` of sibling/has-child
links (hierarchical). `createOneHierarchicalFacetDimState`'s
pop-the-stack-while-at-least-as-deep pass is ported structurally, because that
step *is* the algorithm and it only works because `FacetsConfig.build` indexes
every ancestor path.

On top of it, `SortedSetFacetCounts` implements `getTopChildren`,
`getAllChildren`, `getSpecificValue`, `getAllDims` and `getTopDims`, with
`prepareChildIteration`'s four branches intact — including the non-obvious one:
a multi-valued dim **with** `requireDimCount` has its own ordinal leading the
range, so the child iterator starts one past it.

`getAllDims`'s sort is Java's: `value` descending, ties broken by ascending dim
name. Verified against Lucene's own `getAllDims(10)` over four dims covering
all four config shapes.

One place this is *stricter* than Java, deliberately: Java's constructor does
`FacetsConfig.stringToPath(term)[0]` unguarded in three places and throws
`ArrayIndexOutOfBoundsException` on an empty term. `FacetsState::new` rejects
every empty label once, up front, as `FacetsStateError::EmptyLabel` — which
both gives a typed error instead of a panic on caller data and makes every
`[0]` in the two builders provably in-bounds. A dictionary written by
`FacetsConfig.build` never contains an empty term, but `new` takes a
caller-supplied list.

### 2.5 [MISSING → fixed] multi-valued range counting, and `totCount`

b14 §3.8. Two rules, both load-bearing and both now ported:

- **A document is counted at most once per range.** Java uses
  `startMultiValuedDoc`/`addMultiValued`/`endMultiValuedDoc` (a bit per
  elementary interval, folded in at the end of the doc). Counting per value
  instead reports a document with sizes `{1,2,3}` three times in a `[1,3]`
  bucket. The fixture has exactly that document; `upto3` is 5, not 7.
- **`totCount` counts the document once**, however many ranges it hit.

`totCount` was also missing from the *single*-valued path: `top_range_children`
took it as a caller-supplied parameter, and a caller cannot derive it from the
per-range counts once ranges overlap. `range_facet_counts_with_total` and
`double_range_facet_counts_with_total` now return it; the older signatures are
kept and delegate. The fixture's `price` ranges overlap deliberately: the
counts sum to 15 over 9 documents and `totCount` is 9.

### 2.6 [INTENTIONAL] `OrdinalMap`'s representation, and no segment reordering

Java packs `segmentToGlobalOrds` as `PackedLongValues` deltas against
`globalOrdDeltas`/`firstSegments`, tuned by an `acceptableOverheadRatio`,
because it holds this for a whole index in a JVM heap. This stores a plain
`Vec<i64>` per segment: the same mapping (pinned against Lucene's), 8 bytes per
*segment ordinal*, no bit-unpacking on the lookup that faceting performs once
per non-zero ordinal per segment.

Java also sorts segments by weight (descending unique-term count) so common
terms' deltas land in the cheapest-to-pack segment. That ordering is invisible
in `segmentToGlobalOrds` — global ordinals are assigned by *term* order — so
with a plain vector it optimizes nothing and is not ported. The one observable
consequence is that `first_segment` reports the lowest *input* segment index
rather than the lowest weight-sorted one; documented on the method.

### 2.7 [INTENTIONAL] `getDims()` is deterministic

Java yields `prefixToDimTree`'s keys then `prefixToOrdRange`'s, each in
`HashMap` iteration order — unspecified. This uses `BTreeMap`s, so it is
hierarchical-then-flat with each group in ascending name order. `getAllDims`
sorts its own output regardless, so the only observable difference is for a
caller walking `dims()` directly, where "unspecified" is not a contract worth
reproducing.

### 2.8 [INTENTIONAL] `getTopDims` is the base contract, not the SSDV override

`Facets.getTopDims`'s documented contract is "the same as calling getAllDims
and then only using the first topNDims", and the base class implements exactly
that. `AbstractSortedSetDocValueFacetCounts` overrides it to avoid computing
children for dims that will not make the cut, reading a dim's count directly
where the encoding allows. Here `all_dims` is already a single pass over the
dims with no per-dim I/O, so the override has nothing to buy; implementing the
contract keeps the two provably equal instead of nearly equal.

### 2.9 [MISSING — recorded] what is still not here

- **`FacetsConfig.build(Document)`** — the indexing half (turning `FacetField`s
  into `SortedSetDocValuesField`s and drill-down `StringField`s). A *write*-path
  concern belonging with `lucene-index`'s document builder, which does not
  exist. Recorded in `docs/parity.md`.
- **`ramBytesUsed`/`getChildResources`** (`Accountable`) and the `OrdinalMap`
  reader cache (`cachedOrdMap`, keyed on `IndexReader.CacheKey`) — neither
  mechanism exists in this port.
- **Taxonomy faceting** (`facet/taxonomy/**`) — a different index structure
  entirely, out of scope for this batch and never claimed.

### Verdict

Swept clean. Every b14 facet gap closed, with 18 differential tests against
real `SortedSetDocValuesFacetCounts`/`LongRangeFacetCounts`/
`DoubleRangeFacetCounts`/`OrdinalMap` output over a purpose-built
three-segment index.

---

## 3. `crates/lucene-search/src/highlighter.rs` — `BreakIterator` and `FieldOffsetStrategy`

Java: `search/uhighlight/{FieldOffsetStrategy,PostingsOffsetStrategy,TermVectorOffsetStrategy,AnalysisOffsetStrategy,TokenStreamOffsetStrategy,NoOpOffsetStrategy,SplittingBreakIterator}.java`,
`UnifiedHighlighter.{getOffsetSource,getOptimizedOffsetSource,getOffsetStrategy}`,
and `java.text.BreakIterator`.

| Rust fn | Java | Verdict |
|---|---|---|
| `sentence_boundaries` | `BreakIterator.getSentenceInstance(Locale.ROOT)` | **CORRECTNESS (§3.1)** |
| `split_sentence_boundaries` | `SplittingBreakIterator` | **added** |
| `OffsetSource` | `UnifiedHighlighter.OffsetSource` | **added** |
| `offset_source_for_field` | `UnifiedHighlighter.getOffsetSource(field)` | **added, verbatim** |
| `optimize_offset_source` | `getOptimizedOffsetSource(components)` | **added, verbatim** |
| `offsets_from_postings` | `PostingsOffsetStrategy.getOffsetsEnum` | **added** |
| `offsets_from_analysis` | `TokenStreamOffsetStrategy.getOffsetsEnum` | **added** |
| (`term_vectors_query::matched_term_offsets`) | `TermVectorOffsetStrategy` | pre-existing, now named as the strategy |
| — | `PhraseHelper`, `MemoryIndexOffsetStrategy`, `MultiFieldsOffsetStrategy`, `getSummaryPassagesNoHighlight` | recorded (§3.4) |

### 3.1 [CORRECTNESS → fixed] the abbreviation list was a divergence from Java, not a refinement of it

**The brief asked whether a faithful `BreakIterator` is available in this
project's dependency set. It is.** `unicode-segmentation` is already a
workspace dependency (it backs `lucene-analysis`'s UAX #29 *word* tokenizer),
and its sentence segmentation implements
[UAX #29](https://www.unicode.org/reports/tr29/)'s SB rules — the same
specification the JDK's `BreakIterator.getSentenceInstance` is a rule-based
implementation of. This is the same algorithm, not a regex and not an
approximation.

What it replaced was worse than "a simplification". b14 §4.5 recorded the old
hand-rolled scan (a `.`/`!`/`?` followed by whitespace and an uppercase letter)
plus a hardcoded English abbreviation list — "Mr", "Mrs", "Dr", "St", … —
suppressing the break, and a test celebrating that `"Mr. Smith"` no longer
split. **Run against the JDK, Lucene splits it.** Verbatim output of
`BreakIterator.getSentenceInstance(Locale.ROOT)` (and `Locale.ENGLISH`,
identical):

```
Mr. Smith went home. He slept well.  -> [Mr. ][Smith went home. ][He slept well.]
He finished 21st. She started next.  -> [He finished 21st. ][She started next.]
She said "stop." Then she left.      -> [She said "stop." ][Then she left.]
no terminator at all here            -> [no terminator at all here]
Dr. Who visited St. Paul. Then left. -> [Dr. ][Who visited St. ][Paul. ][Then left.]
One.\nTwo.\n\nThree.                 -> [One.\n][Two.\n\n][Three.]
```

So the abbreviation list made this port's passages differ from Lucene's, and
the "false positive" it was closing is Lucene's actual behaviour. It is gone.

**The JDK's answers are a fixture, not a transcription.**
`fixtures/src/GenBreakIterator.java` runs
`BreakIterator.getSentenceInstance` over eleven texts in both `Locale.ROOT`
*and* `Locale.ENGLISH` and writes the resulting sentences to
`fixtures/data/break_iterator/manifest.properties`;
`sentence_boundaries_match_the_jdks_break_iterator` reads them. Hand-copied
literals would have gone stale silently on a JDK or CLDR data bump — the exact
failure mode that produced the abbreviation list in the first place — whereas
this is regenerable through `scripts/gen-fixtures.sh` like every other
Java-derived expectation in the repo. The test compares the sliced
*substrings* rather than offsets, because `BreakIterator` counts UTF-16 units
and this port counts UTF-8 bytes; every fixture text is ASCII so the comparison
never has to bridge the two. It also asserts ROOT and ENGLISH agree, since an
untailored port cannot match both if they do not.

**One tailoring is applied, and it goes toward Java.** UAX #29's SB4
(`ParaSep ÷`) ends a sentence at *every* paragraph separator, so
`"Two.\n\nThree."` segments as `["Two.\n", "\n", "Three."]` — a "sentence"
consisting of one newline. The JDK folds that run into the preceding sentence
(the sixth line above). Dropping any boundary whose slice is entirely
whitespace reproduces the JDK exactly and cannot affect a boundary that starts
real text.

**A defect found by the new test**: `unicode-segmentation` 1.13.3's
`USentenceBounds::size_hint` computes `lower - 1` unguarded, so collecting the
bounds of an **empty string** panics with "attempt to subtract with overflow"
in a debug build. `sentence_boundaries` returns early for the empty case;
without that guard this port would have panicked on an empty field.

### 3.2 [MISSING → fixed] `SplittingBreakIterator`

`split_sentence_boundaries(text, slice_char)`: sentence boundaries within each
slice between occurrences of `slice_char`, with the separator itself and the
position after it always boundaries (Java's "if the slice is 0-length … that
character is reported as a boundary"), so the enclosed iterator never sees it.
This is what `UnifiedHighlighter` uses for a multi-valued field whose values
are joined with `MULTIVAL_SEP_CHAR` (`'\0'`) and where a passage must never
straddle two values.

### 3.3 [MISSING → fixed] the offset-strategy layer

`offset_source_for_field` is `getOffsetSource`'s `FieldInfo` cascade verbatim
(offsets in the postings → `POSTINGS`, or `POSTINGS_WITH_TERM_VECTORS` if the
field also has vectors; else term vectors → `TERM_VECTORS`; else `ANALYSIS`,
including for a field with no `FieldInfo` at all).

`optimize_offset_source` is `getOptimizedOffsetSource`'s three adjustments:
nothing to highlight → `NONE_NEEDED`; `POSTINGS` with a multi-term part →
downgraded to `ANALYSIS` (a wildcard would otherwise mean scanning the whole
term dictionary); `POSTINGS_WITH_TERM_VECTORS` without one → upgraded to
`POSTINGS`.

Two of the three offset producers are new:

- `offsets_from_postings` (`PostingsOffsetStrategy`) reads each term's
  per-occurrence `startOffset`/`endOffset` out of the postings for one
  document. Verified against real Lucene's own occurrence list
  (`GenBlockTree.java`'s `field.pos.term.*.occurrences`, which is
  `position,startOffset,endOffset,payload` off a real `PostingsEnum`) for three
  documents including one carrying two occurrences of one term.
- `offsets_from_analysis` (`TokenStreamOffsetStrategy`) re-analyzes the stored
  text and keeps tokens whose term is in the query's set, with the analyzer's
  own offsets. The test asserts the offsets actually address the original text
  (`&content[start..end] == term`), not merely that some numbers came back.

All three producers emit `TermOffsetSpan`s in `OffsetsEnum.compareTo` order
(start, then end, then term), so they are interchangeable inputs to
`assemble_fragments` — which is the point of the strategy layer.

### 3.4 [PERF — recorded, with a named blocker] `offsets_from_postings` costs the whole postings list

Raised by this batch's Tier-2 review, and correct. Java's
`PostingsOffsetStrategy` does `postingsEnum.advance(doc)` and then walks only
that document's positions. This decodes **every** document's positions and
offsets, because `blocktree::FieldTerms::positions` is the only accessor in
this port that returns offsets and it returns them for the whole term. So
highlighting one document for a term with a large `docFreq` costs a full
postings sweep — invariant #3, and the fixture (docFreq 2–3) cannot see it.

**Not fixable from this crate**, which is why it is recorded rather than
half-fixed. `FieldTerms::positions_for_docs` exists for exactly this shape and
is what the phrase paths use, but it returns positions *only*:
`read_positions_for_docs` decodes each block's offset pairs into a scratch
buffer and discards them, and an offset for a document it skipped past cannot
be reconstructed from that — the wire format deltas offsets cumulatively within
a document, so a skipping decoder would have to carry per-document start-offset
state it deliberately does not keep. Closing this needs an offsets-carrying
sibling of `read_positions_for_docs` in `lucene-codecs/src/postings.rs`, which
is c8's file and was being edited while this batch ran.

**What was fixed**: the doc-list lookup no longer decodes a set of frequencies
nobody uses (it asks for `PostingsFlags::DocsOnly`), removing the duplicate
decode on top of `positions`'s own. `offsets_from_postings`' doc states the
remaining cost and the exact shape of the fix. In `LEDGER.md`.

### 3.5 [MISSING — recorded] `PhraseHelper` and the memory-index strategies

`PhraseHelper` (restricting matches to actual phrase hits, and its
`willRewrite`/`hasPositionSensitivity` inputs to the source choice) has no
counterpart; `MemoryIndexOffsetStrategy` needs a `MemoryIndex`;
`MultiFieldsOffsetStrategy` needs the masked-fields feature;
`getSummaryPassagesNoHighlight` (the "no match, show the first N sentences"
fallback) sits above this module's input contract. Recorded in
`docs/parity.md`. `optimize_offset_source` takes `has_multi_term_part` as a
caller-supplied boolean precisely because `PhraseHelper` is what would compute
it.

### Verdict

The b14 §4.5/§4.6 gap is closed except for `PhraseHelper` and the memory-index
strategies. The `BreakIterator` item turned out to be a CORRECTNESS finding
rather than a scoping one.

---

## 4. `crates/lucene-search/src/query_parser.rs` — `MultiFieldQueryParser`

Java: `queryparser/classic/MultiFieldQueryParser.java`.

| Rust fn | Java | Verdict |
|---|---|---|
| `parse_multi_field_query` | `MultiFieldQueryParser(fields, analyzer)` + `parse` | **added** |
| `parse_multi_field_query_with_boosts` | `MultiFieldQueryParser(fields, analyzer, boosts)` + `setDefaultOperator` | **added** |
| `Parser::expand_across_fields` | `getMultiFieldQuery(queries)` | **added** |
| `apply_field_boost` | `applyBoost` | **added** |
| — | per-field analyzers | recorded (§4.3) |

### 4.1 [MISSING → fixed] the `Occur` is `SHOULD`, and the disjunction goes at the leaf

The brief asked specifically which `Occur` a bare term expands with:
`getMultiFieldQuery` builds a `BooleanQuery` adding each field's sub-query with
**`Occur.SHOULD`**. Reproduced.

The placement matters more than the occur value. Java expands inside
`getFieldQuery(null, ...)`, so the disjunction sits at the *leaf*, under the
outer query's conjunctions and modifiers: `cat dog` with `AND_OPERATOR` is
Java's documented `+(title:cat body:cat) +(title:dog body:dog)`, which requires
each term in *some* field. Expanding at the top instead
(`+(title:cat title:dog) +(body:cat body:dog)`) would require both terms in the
*same* field — a different query, and the obvious way to get this wrong.
`the_disjunction_sits_under_the_conjunction_not_over_it` pins it.

Every bare atom shape fans out, not just plain terms:
`getPrefixQuery`/`getWildcardQuery`/`getFuzzyQuery`/`getRegexpQuery`/
`getRangeQuery` all carry the same `field == null` branch. An atom that names
its own field is untouched.

**Implementation**: each bare atom's own character span is re-parsed once per
field. `used_default_field` (set exactly where the parser falls back to the
default field, i.e. Java's `field == null`) triggers it, and is cleared by the
expansion so an enclosing group expands its members once each rather than
duplicating the whole group; the re-parse runs with multi-field mode off for
the same reason.

### 4.2 [MISSING → fixed] per-field boosts

Each field's clause is wrapped in a `BoostQuery` when that field has a boost,
so Java's documented `+(title:t^5.0 body:t^10.0)` comes out. A boost of `1.0`
is Java's "no entry in the boosts map" and is left unwrapped.

### 4.3 [INTENTIONAL] one analyzer, so re-parsing is faithful

Java calls `super.getFieldQuery(fields[i], queryText, quoted)` per field
*precisely so a per-field analyzer can produce different terms*, then zips the
i-th sub-clause across fields via its `maxTerms` loop. This parser takes a
single `Analyzer` for all fields, so every field's parse of the same span
yields the same shape and the zip degenerates to "expand each leaf" — which is
what re-parsing does. Per-field analyzers are recorded as not ported.

One divergence, algebraically equivalent and documented: a `^n` written in the
query text lands *inside* each expansion here and *outside* the disjunction in
Java. A `SHOULD` `BooleanQuery` sums its clauses' scores and
`sum(b·sᵢ) == b·sum(sᵢ)`, so the scores are identical.

### Verdict

Swept clean; b14 §7.7 closed. 11 new tests.

---

## 5. `crates/lucene-search/src/doc_value_query.rs` — `FieldExistsQuery`, `IndexOrDocValuesQuery`

Java: `search/{FieldExistsQuery,IndexOrDocValuesQuery}.java`.

### 5.1 [re-assessment] `Occur::FILTER` does not change b14's assessment, and the blocker was named wrongly

b14 §2.4/§2.5 said these "need a `Clause` variant and a `ScorerSupplier`
planner", and the brief asked whether c11's `Occur::FILTER` changes that. It
does not, and the `Clause`-variant half of the diagnosis was not the real
blocker either:

- **`FILTER` is orthogonal.** It governs how a clause *participates* in a
  boolean query (matched but not scored), not which readers a clause can
  reach. A `Clause::FieldExists` would be exactly as unreachable as before.
- **The real blocker is the clause-resolution signature.** `resolve_clause_docs`
  takes `fields`, `doc_in`, `pos_in`, `pay_in`, `live_docs`, `points` — and no
  doc-values, norms or vector input. A `Clause::FieldExists` arm would have
  nothing to read from. Threading a doc-values input through every search
  entry point is a C-ABI change in `lucene-ffi` too; it is a milestone, not a
  batch.

So both are ported **where the readers already are**, as first-class functions
in `doc_value_query.rs` alongside `search_numeric_range` and friends — which is
the same shape every other doc-values query in this port has.

### 5.2 [MISSING → fixed] `FieldExistsQuery` over the doc-values source

`doc_values_field` is `FieldExistsQuery.getDocValuesDocIdSetIterator`'s
`switch` over all five doc-values kinds (Java exposes it as a public static for
exactly this purpose); `doc_has_value` is the per-document half; and
`search_field_exists` is `ConstantScoreScorerSupplier.fromIterator`'s "every
doc the iterator lands on", live-docs filtered. A field with no doc-values
entry is Java's `null` iterator: no matches, not an error.

Tested over real fixtures for every kind — sparse NUMERIC and BINARY, dense
NUMERIC and SORTED, sparse SORTED_NUMERIC and multi-valued SORTED_SET — with
the expected doc sets taken from the fixtures' own manifests rather than
re-derived, plus a `doc_has_value`-vs-iterator agreement test.

### 5.3 [MISSING — recorded] the norms, vector and points sources

`FieldExistsQuery` also accepts a field indexing norms, vectors, or points,
choosing on `FieldInfo` in that order. `field_norms::FieldNorms` exposes no
"does this doc have a norm" predicate (only the total `field_length`/
`norm_inverse`), and the KNN vector reader has no doc-iterator accessor — so
neither can be answered without editing another crate's file. The points source
*is* expressible as an all-encompassing `points_query::search_points_range`,
but that is a full BKD sweep: a materially different cost profile that a caller
should choose deliberately rather than have a field-type switch pick for it.
`rewrite`'s "every doc has this field → `MatchAllDocsQuery`" shortcut needs
`Terms.getDocCount`/`PointValues.getDocCount`/`DocValuesSkipper.docCount` per
leaf and is likewise recorded.

### 5.4 [MISSING → fixed] `IndexOrDocValuesQuery`'s planner, with the default named

The decision rule is small and ported verbatim:

```java
final long threshold = cost() >>> 3;
if (threshold <= leadCost) { index } else { doc values }
```

including Java's own justification for the 8x penalty ("at equal costs, doc
values tend to be worse than points since they still need to perform one
comparison per document").

**The cost model it consumes does not exist here, and the brief asked me to say
so and pick a defensible default rather than guess per call.** Java's `cost()`
for a point range query is `PointValues.estimateDocCount(visitor)` — a BKD tree
walk whose absence b14 §1.4 recorded, noting it had no consumer at the time;
this is the consumer. So `plan_index_or_doc_values` takes
`index_cost: Option<i64>`, and **`None` plans `Index`**.

That is not a coin-flip: it is Java's own answer whenever this query is the
*lead* iterator, since `leadCost == cost()` makes `cost >>> 3 <= leadCost` true
for every non-negative cost, and it is unconditionally what `bulkScorer()` does
("bulk scorers need to consume the entire set of docs, so using an index
structure should perform better"). Choosing doc values without an estimate
would be the guess — it is only right when some *other* clause leads with a
much smaller doc set, which is information the caller has and passes as
`lead_cost`. A negative cost (which Java asserts cannot happen) is clamped
rather than sign-extended into a huge threshold that would silently flip the
plan.

### Verdict

Both ported at the layer that can execute them, with the wiring gap named
precisely instead of restated.

---

## 6. `crates/lucene-search/src/directory_reader.rs` — `blocktree::open_shared` (c1 F-13)

### 6.1 [PERF → fixed, 4.8x] the segment open was two-thirds a `memcpy`

**Before** (`benches/directory_reader_open.rs`, criterion, `MmapDirectory` over
`benchmarks/.corpus/merged`: one force-merged real-Lucene segment, 4.7 MB
`.tim`, 89 KB `.tip`, 579k terms), re-run rather than quoted:

| case | before |
|---|---|
| `DirectoryReader::open` (whole reader) | **579 µs** |
| `blocktree::open` (bytes already in memory) | 198.8 µs |
| `blocktree::open_shared` | 0.496 µs |
| `Arc::<[u8]>::from(&tim[..])` alone | 201.9 µs |

So c1's diagnosis was right — the copy *is* essentially the whole of the
0.175 ms residual — but **`open_shared` as it stood could not remove it from
`directory_reader`**, and that is the part c1's "three-line change" estimate
missed. `open_shared` took an `Arc<[u8]>`, and `Arc<[u8]>` owns its allocation:
there is no zero-copy route to one from `lucene_store::Input`, whether that is
a `memmap2::Mmap` (which cannot be moved into an `Arc`'s allocation) or a
`Vec<u8>` (`Arc::<[u8]>::from(Vec)` copies too). Migrating the caller alone
would have moved the copy, not removed it.

**Fixed by erasing the owner instead.** `blocktree::SharedBytes` is
`Arc<dyn AsRef<[u8]> + Send + Sync>`; `open_shared` takes those, and
`lucene_store::Input` gained an `AsRef<[u8]>` impl so `open_segment_file`'s
existing `Arc<Input>` — which for a `MmapDirectory` *is* the mapping —
coerces straight into one. `blocktree::open` keeps its `&[u8]` signature and
copies, as its doc says.

**After**, same bench, same corpus:

| case | before | after |
|---|---|---|
| `DirectoryReader::open` | 579 µs | **120.7 µs** (4.8x) |
| `blocktree::open` | 198.8 µs | 208.3 µs (unchanged; still copies, by contract) |
| `blocktree::open_shared` (bytes resident) | 0.496 µs | 0.449 µs |
| `open_shared` from a fresh mapping (what the reader now does) | — | 30.0 µs |

The win is larger than the 199 µs copy on its own because the copy also *first-
touched* 4.7 MB of the mapping; the lazy dictionary now faults in only the
pages a lookup actually walks.

**Cost of the erasure, measured, not assumed.** `SegmentTermsEnum::tim`/`index`
become a virtual `as_ref` instead of a field read. Both are hoisted into a
local at the top of each lookup, so it is a handful of predicted indirect calls
per *seek*, not per byte. `cargo bench -p lucene-codecs --bench blocktree_open`,
before and after (this machine's run-to-run spread on these cases is ±25%, so
the honest reading is "no measurable regression", not "faster"):

| case | before (`Arc<[u8]>`) | after (`SharedBytes`), run 1 / run 2 |
|---|---|---|
| `blocktree/open` | 210.2 µs | 223.5 / 201.8 µs |
| `seek_exact/hit` | 1.631 ms | 1.134 / 1.396 ms |
| `seek_exact/miss` | 882 µs | 468 / 700 µs |
| `next/whole_field` | 6.95 ms | 6.91 / 7.31 ms |

**Files outside `lucene-search` touched**: `lucene-codecs/src/blocktree.rs`
(the `SharedBytes` alias, `FieldTerms`' two fields and their two accessors, a
hand-written `Debug` since the erased type has none, and `open`/`open_shared`'s
signatures) and `lucene-store/src/directory.rs` (a six-line `AsRef<[u8]>` impl).
Both files were quiet (last modified 9 and 13 hours before this batch);
`blocktree.rs` is c1's, and c1 is finished. `open_shared`'s only other caller is
its own unit test.

### Verdict

c1's F-13 closed, with the reason its "three-line change" estimate was wrong
recorded: the blocker was `Arc<[u8]>`'s ownership, not the call site.

---

## 7. `PostingsFlags::DocsOnly` on the unscored paths (handed over by c8)

Java: `TermsEnum.postings(reuse, PostingsEnum.NONE)`, `PForUtil.skip`.

### 7.1 [MISSING → fixed] every matching-only path was asking for frequencies it discards

c8 ported `PostingsFlags` + `for_util::pfor_skip` into `lucene-codecs` and
wired its own crate's caller, but could not reach `lucene-search`. Wired here
for the paths that provably never call `freq()`:

| path | Java |
|---|---|
| `term_doc_ids` | `TermQuery` used as a matching clause |
| `prefix_doc_ids`, `wildcard_doc_ids`, `fuzzy_doc_ids`, `regexp_doc_ids` | `MultiTermQuery`'s constant-score rewrite |
| `term_in_set_doc_ids` | `TermInSetQuery` |
| `multi_phrase_slot_docs` | `MultiPhraseQuery`'s per-slot candidate union |
| `stream_constant_score_clause` | the constant-score streaming union |
| every `Occur::FILTER` leg of `try_conjunction_lazy` | `BooleanScorerSupplier`'s `required` minus `requiredScoring` |

`must_not` clauses go through `resolve_clause_docs` → `term_doc_ids` and so are
covered by the first row.

**Enforced structurally, not by convention** — the brief's requirement, and the
real risk, since `DocsOnly` fills `Postings::freqs` with `1`s rather than
leaving it empty: a path that asked for `DocsOnly` and then read a frequency
would get a plausible, silently wrong number.

- `term_docs_only` returns a bare `Vec<i32>`. The `1`-filled `freqs` array is
  never in scope, so there is nothing to misread.
- `stream_constant_score_clause`'s cursors are wrapped in a local
  `DocsOnlyCursor` newtype exposing `next_doc` and nothing else.
- `try_conjunction_lazy`'s `Leg` lost its `scoring: bool` in favour of a
  `LegRole` enum whose `Scoring` variant *holds* the scoring inputs
  (`weight`, `norms`). The only way to reach the `freq()` call is to have
  destructured a `Scoring` leg — i.e. a leg whose cursor decoded frequencies.
  `Leg::bound` gained an explicit `LegRole::Filter => 0.0` arm for the same
  reason: sound rather than merely unreached.

### 7.2 [PERF — measured here, not quoted] the win is small, as c8 said

`benches/docs_only_postings.rs`, criterion, `benchmarks/.corpus/merged`. The
A/B is inside one build — the same term resolved with `Freqs` and with
`DocsOnly` — so it is a true before/after of the decode rather than two
builds compared.

| term (`body`) | docFreq | `Freqs` | `DocsOnly` | speedup |
|---|---|---|---|---|
| `t0` | 4,997,130 | 7.099 ms | **6.325 ms** | 1.12x |
| `t1` | 4,917,475 | 7.897 ms | **6.984 ms** | 1.13x |
| `t999` | 2,506 | 2.312 µs | **2.143 µs** | 1.08x |

1.08–1.13x, inside the 1.07–1.32x c8 reported and at the low end of it — which
is what a `body` field of mostly-`1` frequencies should give, since a low-value
frequency block is the cheapest kind to unpack.

The bench deliberately carries **no** whole-query case: there is no way to run
the *old* clause shape in this build, so a query-level number would measure
"after" against nothing. The per-term A/B above is the change itself.

c8's explanation holds on this crate's call shape too: a `.doc` block is
dominated by the doc-delta bit-packing, not the frequency bit-packing, so
skipping the frequency blocks removes a minority of the work. **This is
primarily a correctness-of-contract change — asking the decoder for what the
caller actually uses — with a modest speedup attached.** No scored path was
flagged as docs-only; `scoring.boolean.filter*`'s bit-identical differentials
(c11's) and the whole 983-test suite are unchanged.

### Verdict

Wired, structurally enforced, measured.

---

## Coverage

Every file in `lucene-search` is above the 95%-per-file bar, and the crate total
rose. Both readings (the "before" as well) were taken by this batch with
`cargo llvm-cov -p lucene-search --summary-only` against a scratch
`CARGO_TARGET_DIR` — the shared `target/` is being emptied by other batches
mid-run, which llvm-cov reports as coverage rather than as an error, the same
precaution c1 recorded.

| file | before (c11's close) | after |
|---|---|---|
| `lib.rs` | 95.22% | **96.70%** |
| `facets.rs` | 99.04% | **99.47%** |
| `highlighter.rs` | 99.19% | 98.13% |
| `query_parser.rs` | 98.49% | **98.64%** |
| `doc_value_query.rs` | 99.47% | 99.38% |
| `ordinal_map.rs` (new) | — | **98.11%** |
| `directory_reader.rs` | 99.00% | 97.55% *(see below — not this batch)* |
| crate total | 97.19% | **97.61%** |

`lib.rs` rose because §1.1 removed 78 uncovered lines and §7.1's rewrites are
covered. The files that dipped by a fraction did so by *growing*, not by losing
tests: `highlighter.rs` gained 358 regions, and its uncovered remainder is
almost entirely the manifest-unescaping helper in
`sentence_boundaries_match_the_jdks_break_iterator` — the `\\` and `\uXXXX`
escape arms, which no current fixture text needs but which have to exist for
the next one that does.

**`directory_reader.rs`'s drop is not this batch's.** It was 99.00% when this
batch's own change to it (a three-line switch to `open_shared`) landed. Another
batch added ~110 lines of doc-values-update generation handling
(`dv_generations`) to the file mid-run; it read 95.88% before that batch's own
tests landed and 97.55% after. Flagged so the next reading is not attributed to
c12.

`explain.rs` (95.49%), `query_cache.rs` (96.36%), `multi_segment.rs` (96.82%)
and `soft_deletes.rs` (96.91%) are untouched by this batch and unchanged.

---

## Gate

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-search --all-targets -- -D warnings` — **clean,
  exit 0**. Four other batches' in-flight edits broke the build under it at
  different points along the way (`lucene-index/src/index_writer.rs`'s unused
  vector imports; `lucene-codecs/src/hnsw.rs`'s `int_roundings`; `vectors.rs`'s
  missing `validate_docs`; a `lucene-util` helper move; a `dv_generations` field
  whose type was not yet imported and a `needless_borrows_for_generic_args` lint
  in its new test — the last two *inside* `crates/lucene-search/`, nominally
  this batch's own directory; and finally an `unnecessary_cast` in
  `index_writer.rs`). Every one was **waited out and retried**, never edited
  over. This batch's own hunks survived all of it, and at no point was a lint
  or failure traced to this batch's code.

  The tree stayed in motion to the end: `cargo fmt --all --check` and
  `cargo clippy` results below are point-in-time. The last verified-green run
  of the full gate on this batch's code is recorded here; a run taken minutes
  later fails on `lucene-index`'s lint, which is c7's file.
- `cargo test -p lucene-search` — **983 passed, 0 failed** (899 before this
  batch). A later run of the same command read 1006, because another batch
  added 23 tests to this crate after that reading; 983 is the number this
  batch's own work accounts for.
- `cargo test -p lucene-ffi` — 492 passed, 0 failed, unchanged by the two
  breaking API changes below (it builds `BooleanQuery` through the builders and
  reads `Fragment`'s `text`/`matched_terms` only). Taken during a window when
  that crate compiled; a later re-run was blocked by another batch mid-edit in
  `lucene-ffi/src/segment.rs` (`open_segment_with_pay`), not by anything here.
- `scripts/gen-fixtures.sh` — `lucene-facet` added to `LUCENE_MODULES`, which
  `GenFacets.java` needs; both new generators (`GenFacets`, `GenBreakIterator`)
  are auto-discovered by the `ls Gen*.java` scan. Every file in
  `fixtures/data/facets_index/` (204 KB, 50 files) is produced by it, so
  `--check` has nothing "extra" to complain about; the `write.lock` an
  `IndexWriter` leaves behind is deleted, as `generate_into` does.
- `docs/parity.md` updated in the same change: 10 new rows, **plus three
  existing rows corrected** whose "still out of scope" text this batch's work
  falsified (see the Tier-2 review, finding 2) — invariant #7 is "drift is a
  bug", not "append a row".
- `javac` over all of `fixtures/src/*.java` — clean, so `gen-fixtures.sh`'s
  single compile step still works with both new generators.

Not committed, per the brief.

---

## Tier-2 review

Run on the batch's own files (the `quality-reviewer` subagent, reading the Java
sources alongside). It **independently re-ran the JDK** on §3.1's six strings
plus the blank-line case and confirmed the port's output matches
`BreakIterator.getSentenceInstance` byte for byte, and traced
`build_one_hierarchical_dim`, `adjust_path_count`, `DimTreeChildren`,
`OrdinalMap::build`'s heap ordering and the `LegRole` refactor against Java
without finding a path that returns a wrong count, ordinal, boundary or query
shape.

Two gating findings and nine advisories. All eleven acted on:

1. **`offsets_from_postings` decodes the whole postings list** (gating) —
   §3.4. Recorded with the named blocker, and the duplicate doc-list decode on
   top of it removed. The reviewer's suggested fix
   (`FieldTerms::positions_for_docs`) does not actually work: that function
   returns positions without offsets.
2. **Three stale `docs/parity.md` rows and the `facets.rs` module doc** still
   told callers to use the label-merge workaround this batch deleted (gating).
   All four rewritten; the append-only parity convention that caused it is now
   an open item in `LEDGER.md` with a suggested mechanical check.
3. `term_docs_only` had `term_doc_ids`' old doc comment glued to its front (so
   it claimed to filter by `live_docs`, which it does not) and `term_doc_ids`
   had none. Split apart.
4. The free `all_children`'s doc claimed Java returns `null` on
   `childCount == 0`; only `getTopChildren` does. Re-worded as this port's own
   convention for the dim-less primitives, and the divergence from
   `SortedSetFacetCounts::all_children` (which follows Java) named.
5. `range_facet_counts`/`double_range_facet_counts` duplicated the containment
   loop rather than delegating as §2.5 claimed. They delegate now.
6. `offset_source_for_field`'s `None if has_term_vectors` arm contradicted its
   own doc *and* Java, which reads `hasTermVectors()` only inside the
   `fieldInfo != null` branch. Arm deleted, test updated.
7. A stale `579 us -> 385 us` in `directory_reader.rs`' inline comment against
   this report's measured 120.7 µs. Corrected.
8. `MEASUREMENT_TABLE_PLACEHOLDER` in §7.2 — already filled before the review
   landed; verified.
9. §3.1's JDK expectations were hand-transcribed Rust literals, so a JDK/CLDR
   bump could silently reintroduce the divergence with the test still green.
   Now `fixtures/src/GenBreakIterator.java` + a manifest, regenerable through
   `scripts/gen-fixtures.sh` like every other Java-derived expectation. **This
   was the batch's weakest point and the reviewer was right to name it**: the
   whole §3.1 finding is about a hand-written approximation of Java going
   stale, and pinning its fix by hand repeated the mistake one level up.
10. §4.3's "one analyzer, so the zip degenerates" was the wrong reason — Java
    zips per analyzed *token*, re-parsing groups per *field*, and they coincide
    only because `clause_from_analyzed_terms` never produces a multi-clause
    `BooleanQuery`. `expand_across_fields`' doc now states that as the
    precondition, with what breaks if it changes.
11. `OrdinalMap::build` materializes every segment's term list where Java
    streams `TermsEnum`s. Named in the module doc as a known cost, with the
    blocker (no cursor API over the doc-values terms dictionary); open item in
    `LEDGER.md`.

Two changes that landed mid-review — `FacetsStateError::EmptyLabel` and
`sentence_boundaries`' backward whitespace pass — were re-checked by the
reviewer and confirmed correct.

---

## Cross-file notes

- **Files outside `crates/lucene-search/` changed**: `fixtures/src/GenFacets.java`
  and `fixtures/src/GenBreakIterator.java` (new), `fixtures/data/facets_index/`
  and `fixtures/data/break_iterator/` (new, generated), `scripts/gen-fixtures.sh`
  (one line: `lucene-facet` in `LUCENE_MODULES`), `crates/lucene-codecs/src/blocktree.rs`
  and `crates/lucene-store/src/directory.rs` (§6.1, both quiet and both required
  by the carry-over), `docs/parity.md` (append-only).
- **Public API changes**, all additive except two: `facets::FacetResult` gained
  `dim`/`path` and its `value` became `i64` (§2.3 — the `-1` case cannot be
  expressed otherwise); `blocktree::open_shared`'s two buffer parameters became
  `blocktree::SharedBytes` (§6.1 — its only non-test caller is
  `directory_reader`). `lucene-ffi` builds and its suite passes unchanged.
- **New dependency**: `unicode-segmentation` on `lucene-search` (already a
  workspace dependency, used by `lucene-analysis`). No new external crate enters
  the workspace.
- **Open items for a later batch**:
  - `FacetsConfig.build(Document)` — the indexing half of faceting, blocked on
    `lucene-index` having a document builder (§2.9).
  - `FieldExistsQuery`'s norms / KNN-vector / points sources, and its
    `rewrite`-to-`MatchAllDocsQuery` shortcut (§5.3).
  - `PointValues.estimateDocCount` — the cost estimate
    `IndexOrDocValuesQuery`'s planner would consume (§5.4); b14 §1.4 recorded it
    as having no consumer, and now it has one.
  - A doc-values / norms / vector input in `resolve_clause_docs`, which is what
    a `Clause::FieldExists` or `Clause::IndexOrDocValues` variant actually needs
    (§5.1). A C-ABI change in `lucene-ffi` too — a milestone, not a batch.
  - `PhraseHelper` and the memory-index offset strategies (§3.4).
  - Per-field analyzers for `MultiFieldQueryParser` (§4.3).
