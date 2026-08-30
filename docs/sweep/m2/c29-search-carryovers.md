# c29-search-carryovers

Follow-up batch closing the recorded carry-overs in the search crate: b14/c12's
`FieldExistsQuery` sources and `PointValues.estimateDocCount`, c12's
`PhraseHelper`, `OrdinalMap` input materialization and `FacetsConfig.build`,
c14's A1 (`doc_values_for_field` has no production caller) and c23's F13
(byte-vs-UTF-16 offsets).

Java read from **`/home/tuong/work/lucene-10.5.0`** throughout.

Files: `crates/lucene-search/src/{doc_value_query,highlighter,facets,
field_norms,points_query,ordinal_map,multi_segment,term_vectors_query,lib}.rs`,
`crates/lucene-ffi/src/{segment,registry,sort,facets,range_sort,highlighter,
results_fragments,lib}.rs`.

**Two of the seven items were wrong-answer bugs reachable from ordinary use
(items 6 and 7). Both are fixed, each with the pre-fix behaviour asserted as a
negative control so the test cannot pass either way.**

---

## 1. `crates/lucene-ffi/src/{segment,registry,sort,facets,range_sort}.rs` — c14's A1

Java: `index/SegmentDocValuesProducer` (`dvProducersByField`),
`index/SegmentReader`'s `initDocValuesProducers`.

| Rust | Java | verdict |
|---|---|---|
| `SegmentHandle::doc_values_for_field` | `SegmentDocValuesProducer.dvProducersByField.get(field)` | **added** |
| `ffi_segment_add_doc_values_generation` | (no Java counterpart — the C-ABI attach point; Java reads `SegmentCommitInfo` itself) | **added, not-in-Java** |
| `sort::numeric_entry_for` / `sorted_numeric_entry_for` | `LeafReader.getNumericDocValues`/`getSortedNumericDocValues` | divergent → fixed |
| `facets::numeric_entry_for` / `sorted_set_multi_entry_for` | `LeafReader.getNumericDocValues`/`getSortedSetDocValues` | divergent → fixed |
| `doc_value_query::search_numeric_range_sorted_by_field` | `IndexSearcher.search(q, n, sort)` over two fields | divergent → fixed |
| `multi_segment::DocValueSegment` | per-leaf `(range, sort)` producers | divergent → fixed |

### 1.1 `[CORRECTNESS → fixed]` an updated field sorted, faceted and read back its **superseded** values

Java routes each field to its own `DocValuesProducer`: a field whose newest
`FieldInfo.docValuesGen` is not `-1` is served from the generation-suffixed
`.dvm`/`.dvd` that `updateNumericDocValue` wrote, and every other field from
the base pair. c14 ported that on `lucene_search::SegmentReader`
(`doc_values_for_field`) and recorded that **nothing called it**: `lucene-ffi`'s
`SegmentHandle` knew only about the base `.dvm`/`.dvd`, so after an
`updateNumericDocValue` a sort, a facet count or a per-doc lookup returned the
pre-update values. Nothing structural notices — the base column is a perfectly
valid file, it is simply the wrong one.

Measured on the `GenDocValuesUpdates` fixture, whose generator sets `val` to
`7000` on every even document and resets 40 others to *no value*: before the
fix the FFI read `val` back as `0,1,2,...,99` (the original per-document
values) for all 100 documents, where real Lucene reads the manifest's
`expected_val`. A sort by `val` returned 100 documents in doc-id order instead
of 60 ranked by the updated values; a range facet on `[7000,7000]` counted 0
instead of 30.

Fixed in three parts:

1. **`SegmentHandle::dv_generations` + `doc_values_for_field`** (`registry.rs`)
   — the per-field routing, returning the `(meta, data)` pair **together**.
2. **`ffi_segment_add_doc_values_generation`** (`segment.rs`) — an additive
   C-ABI call taking the generational `.fnm` (`SegmentCommitInfo
   .getFieldInfosGen()`, the only file recording `docValuesGen`) plus the
   generation's `.dvm`/`.dvd`/suffix. It follows `ffi_segment_set_live_docs`'s
   pattern for the same reason: a generation changes without the segment being
   rewritten, and `ffi_open_segment`'s 30-parameter signature stays stable. The
   generation's `.dvm` is parsed against a **one-field** `FieldInfos`, exactly
   as `SegmentDocValuesProducer` does, so a `.dvm` naming another field fails to
   parse rather than being mapped onto the wrong column. `docValuesGen == -1`
   is refused with a message naming the likeliest cause (the caller passed the
   base `.fnm`, where every field reads `-1` and the silent outcome would be no
   generation attached at all).
3. **Every read routed through it** — `sort.rs`, `facets.rs`, `range_sort.rs`.

### 1.2 `[CORRECTNESS → fixed]` the entry and the bytes were resolved separately

The bug's *shape*, not just its instance: `numeric_entry_for` looked the entry
up in `segment.dv_meta` and its caller then fetched bytes from
`segment.dv_data`, with a comment asserting the two always correspond. That
holds only while a segment has exactly one doc-values column. Both helpers now
return `(&Entry, &[u8])` from one `doc_values_for_field` call, so the mismatch
is unrepresentable rather than merely fixed.

The same reasoning forced two `lucene-search` signature changes, because a
segment's **range** field and **sort** field need not be in the same column:
`doc_value_query::search_numeric_range_sorted_by_field` took one
`doc_values_data` for both entries and now takes `range_data`/`sort_data`, and
`multi_segment::DocValueSegment`'s `doc_values_data` became
`range_data`/`sort_data`. Callers that share one column pass the same slice
twice.

### 1.3 `[MISSING → fixed]` `ffi_open_segment` could not open a generational `.fnm`

Found while writing the fix. `ffi_open_segment` hardcodes
`field_infos::parse(&fnm, &id, "")`, but a generational `.fnm`'s index header
carries the generation in base 36 as its codec suffix (`"4"` for `_0_4.fnm`),
so passing one to `ffi_open_segment` is a header mismatch. That is why
`ffi_segment_add_doc_values_generation` takes the generational `.fnm` **and its
suffix** itself rather than requiring the segment to have been opened with it:
the caller keeps opening the base `.fnm` exactly as before, and the generation's
own files (`.fnm`, `.dvm`, `.dvd`) arrive together, which is what they are.

### Tests

Ten new tests in `segment.rs`, all against the real `GenDocValuesUpdates`
fixture: the updated column read back per document; a never-updated field still
served from the base pair alongside it; a sort ranked by the updated values;
a range facet counting them; re-attaching the same field replacing rather than
shadowing; and five refusals (`docValuesGen == -1`, unknown field, unknown
handle, wrong codec suffix, another field's `.dvm`) plus "a failed attach leaves
the handle usable". Three of them assert the **base** column's values first, so
they fail if the fix silently stops applying.

### Verdict

Swept clean; c14's A1 closed. c14's A2 (`check_index` reads the base `.fnm`) is
`check_index.rs`'s and stays open.

---

## 2. `crates/lucene-search/src/highlighter.rs` — c23's F13, the offset unit

Java: `analysis/tokenattributes/OffsetAttribute`,
`search/uhighlight/{Passage,DefaultPassageFormatter,FieldHighlighter}`.

| Rust | Java | verdict |
|---|---|---|
| `utf16_offset_to_byte` / `utf16_len` / `byte_offset_to_utf16` | (implicit: Java indexes `String` directly) | **added**, replacing `char_offset_to_byte`/`byte_offset_to_char` |
| `assemble_fragments` | `FieldHighlighter.highlightOffsetsEnums` + `DefaultPassageFormatter.format` | divergent (unit) → fixed |
| `offsets_from_analysis` | `AnalysisOffsetStrategy` | divergent (unit) → fixed |
| `Fragment::{start_offset,end_offset}` | `Passage.get{Start,End}Offset` | divergent (unit) → fixed |

### 2.1 `[CORRECTNESS → fixed]` what we emit, established against real Lucene

Three units were in play and this port used all three:

| producer | unit |
|---|---|
| real Lucene's `OffsetAttribute`, and therefore every offset in a `.tvd`/`.pos`/`.pay` | **UTF-16 code units** (indices into the original `String`) |
| `lucene_analysis::Token` | **UTF-8 bytes** (built from `char_indices()`/`len_utf8()`) |
| `highlighter.rs` (before this batch) | **Unicode scalars** (`char_indices().nth(n)`) |

They coincide for ASCII, which is why every fixture in the repo was blind to
it. `fixtures/src/GenBreakIterator.java` now also records, for five texts that
separate all three, every token a real `StandardAnalyzer` produces as
`term:start,end:slice`, where `slice` is Java's own
`text.substring(start, end)` — the offsets used the way a highlighter uses
them. `offset_text.2` is `"alpha 😀 beta 𝐀 gamma"`: 22 UTF-16 code units, 20
Unicode scalars, 30 UTF-8 bytes.

Two consequences, both wrong answers:

- **Reading a real Lucene index**: offsets are UTF-16, we read them as Unicode
  scalars. Identical across the whole BMP, wrong by one per preceding
  supplementary-plane character. A document containing an emoji highlighted the
  wrong span for every subsequent term. `crates/lucene-search/tests/
  highlighter_utf16_offsets_fixtures.rs`'s negative control asserts the fixture
  contains at least one token a scalar-count reader gets wrong, so this is not
  a theoretical claim.
- **Reading this port's own analyzer output**: `offsets_from_analysis` fed
  `lucene_analysis`'s **byte** offsets straight into `assemble_fragments`'s
  char-offset arithmetic. Wrong for *any* non-ASCII text, not just astral —
  `"café naïve dog"` highlighted `"ve dog"` instead of `"dog"`.

Fixed by making the module's unit **Java's `char`, i.e. the UTF-16 code unit**,
throughout: spans, `Fragment` offsets, `FragmentConfig::window_chars` and the
`contentLength`/`passageLength` `PassageScorer` consumes (Java's are the same
`String` lengths). `offsets_from_analysis` converts at the boundary, documented
as a boundary conversion rather than a fix for the underlying divergence.
`utf16_offset_to_byte` rounds an offset landing inside a surrogate pair down to
that code point's start, so a slice is always on a valid UTF-8 boundary — it
cannot panic on any input, which the old `char_indices().nth()` also
guaranteed and which is worth keeping.

The FFI boundary is documented to match: `ffi_assemble_fragments`'s input spans
and `ffi_fragment_result_span`'s output are Java `char` indices, so a JVM
caller passes what `OffsetAttribute.startOffset()` gave it and slices with
`String.substring` without converting anything.

### 2.2 `[CORRECTNESS — HANDOFF, `lucene-analysis`]` the *write* side is still byte offsets

`lucene_analysis::Token`'s `start_offset`/`end_offset` remain UTF-8 byte
offsets. Its own doc comment still says "No live code path is broken by this
today", which c23 already showed to be stale: `indexing_chain` writes those
offsets into `.pos`/`.pay`, real Lucene reads them back as
`startOffset()`/`endOffset()`, and `CheckIndex` never compares an offset
against the text it indexes. **A Rust-written index of non-ASCII text carries
offsets Lucene will mis-slice**, and this batch does not change that — it makes
the *reader* correct, which is the half in this batch's files.

The fix is contained but not here:

- `lucene-analysis/src/lib.rs::tokenize` builds offsets from
  `char_indices()`/`len_utf8()`; emitting `len_utf16()` sums instead is a
  two-line change plus its doc comment.
- `crates/lucene-analysis/tests/analysis_fixtures.rs` already has a
  `char_offsets_to_byte_offsets` reconciliation helper applied to every
  Lucene-derived expectation; that helper is what disappears (and note it
  converts *scalars* to bytes, so it is itself wrong for astral text — the
  `GenAnalysis` fixture has no supplementary-plane case, which is worth adding
  at the same time).
- `crates/lucene-search/src/highlighter.rs::offsets_from_analysis`'s boundary
  conversion then becomes an identity and should be deleted, not left in.
- `lucene-index`'s `indexing_chain` passes offsets through opaquely, so it
  needs no change; its ASCII-only positions fixtures stay green either way,
  which is precisely why nothing will catch a partial fix. Add a non-ASCII
  document to `GenTermVectors`/the positions write-path verifier as part of it.

### Verdict

The read side is swept clean and pinned against real Lucene. The write-side
unit is handed off, with the exact call sites.

---

## 3. `crates/lucene-search/src/doc_value_query.rs` — `FieldExistsQuery`'s other sources

Java: `search/FieldExistsQuery.java`.

| Rust | Java | verdict |
|---|---|---|
| `field_exists_source` | `ConstantScoreWeight.scorerSupplier`'s three-way `FieldInfo` if/else | **added** |
| `search_field_exists_norms` | its `getNormValues(field)` branch | **added** |
| `search_field_exists_vectors` | its `getFloatVectorValues/getByteVectorValues(field).iterator()` branch | **added** |
| `search_field_exists` | its doc-values branch (already ported, c12) | identical |
| `field_exists_leaf_is_complete` | `rewrite`'s per-leaf test | **added** |
| `FieldNormsCursor::has_norm` (`field_norms.rs`) | `NumericDocValues.advanceExact` on the norms iterator | **added** |
| — | `count(LeafReaderContext)`'s live-doc arithmetic, `isCacheable`, `visit`/`equals`/`hashCode`/`toString` | not ported (no `Query` object graph here — this port's queries are functions) |

### 3.1 `[MISSING → fixed]` the norms and vector sources

c12 recorded these as blocked: `FieldNorms` had no "does this doc have a norm"
predicate and the vector reader no doc iterator. b13 and c10/c16 removed both
blockers, so this is now a small change:

- **Norms.** `FieldNormsCursor::norm_byte` already answers exactly the
  question — it returns `Option`, and `field_length`/`norm_inverse`
  *deliberately* substitute `UNNORMED_FIELD_LENGTH` for the `None`, which is
  what made it unanswerable through the public API. `has_norm` exposes the
  `Option` with the value dropped, so the two can never disagree.
  `search_field_exists_norms` sweeps documents ascending holding one cursor,
  which for a sparse field is one forward `IndexedDISI` walk rather than a
  block walk per document.
- **Vectors.** Iterating **ordinals** and mapping each to its document, not
  sweeping `0..max_doc`: a `KnnVectorValues` iterator visits exactly `size()`
  documents, and their doc ids come back ascending because ordinals are
  assigned in document order (the `ordinal -> doc` map is a
  `DirectMonotonicReader`). On the fixture's sparse field that is 1 334 steps
  instead of 4 000 `IndexedDISI` lookups. A doc id outside the segment is
  reported as corruption rather than skipped.

Verified against real fixtures with the sparse case exercised in both: the
`norms_index` fixture's `sparse_body` field (manifest records `NONE` for
documents 1 and 3) and `vectors_index`'s `sparse_f32` (1 334 of 4 000
documents, with the manifest's own `ord_to_doc` spot checks used as the
expectation).

### 3.2 `[CORRECTNESS in the record → corrected]` there is no points *source*

c12 §5.3 recorded "the norms, vector and points sources" and reasoned about
expressing the points one as an all-encompassing `search_points_range`.
Re-reading 10.5.0: **`FieldExistsQuery`'s `scorerSupplier` never touches
`PointValues`.** Points appear only in `rewrite` and `count`, as a *docCount*
proxy for a doc-values field. So there were three sources, not four, and the
expensive one c12 was reluctant to pick automatically does not exist. The
record is corrected in `docs/parity.md` and in the function's doc comment.

### 3.3 `[MISSING → fixed]` `field_exists_source`'s order, and its error case

Java checks `hasNorms()` **first**, so a field with both norms and doc values
is answered from its norms — worth pinning, because "pick whichever exists" is
the obvious wrong implementation and is indistinguishable on any field that has
only one. A field indexing none of the three is Java's
`IllegalStateException`; this returns `Error::FieldExistsUnsupported` carrying
Java's own message, rather than an empty match set. (A field *name* absent from
`FieldInfos` is a different case — Java's `null` scorer, no matches, no error —
and never reaches this function, which takes a `FieldInfo`.)

### 3.4 `[MISSING → partially fixed]` `rewrite`'s `MatchAllDocsQuery` shortcut

`field_exists_leaf_is_complete` is the per-leaf half; the whole-reader decision
(rewrite only when **every** leaf says yes) stays with the caller, which is
where the leaf list lives. Two details of Java's are reproduced rather than
tidied: the doc-values branch is an **OR over three optional counts** (terms,
points, skipper) so a leaf with none of them available is *not* rewritable,
while the norms and vector branches have a single count each; and Java's norms
branch reads `reader.getDocCount(field)`/`reader.maxDoc()` off the **top-level**
`IndexReader` while the other two read the *leaf*'s, inside the same per-leaf
loop. That asymmetry is documented on the function, since a caller reproducing
`rewrite` has to feed it the numbers Java feeds it.

### Verdict

Swept clean; b14 §2.4 and c12 §5.3 closed, and c12's "points source" entry
withdrawn as a misreading. 6 new tests.

---

## 4. `crates/lucene-search/src/points_query.rs` — `PointValues.estimateDocCount`

Java: `index/PointValues.java`
(`estimateDocCount`/`estimatePointCount`), `search/IndexOrDocValuesQuery.java`.

| Rust | Java | verdict |
|---|---|---|
| `estimate_doc_count` | `PointValues.estimateDocCount(visitor)`'s arithmetic | **added** |
| — | `PointValues.estimatePointCount(visitor, pointTree, upperBound)` | missing (recorded, §4.2) |
| `doc_value_query::plan_index_or_doc_values` | `IndexOrDocValuesQuery.ScorerSupplier.get` | identical (c12) |

### 4.1 `[MISSING → fixed]` the estimate's arithmetic

Java's three branches, ported verbatim: an estimate at or above the field's
whole point count matches every document; a single-valued field
(`size == docCount`) or a zero estimate passes the point estimate through
unchanged; otherwise the multi-valued urn approximation
`D * (1 - ((N - n)/N)^(N/D))`, floored at 1 so a non-empty match never reports
zero cost. The `(long)` truncation toward zero is `as i64`'s behaviour for a
finite double, and is what makes the hand-computed expectations in the tests
exact.

### 4.2 `[MISSING — HANDOFF, `lucene-codecs`]` `estimatePointCount`'s BKD walk

The walk needs each node's subtree `size()` **mid-traversal**: it adds
`pointTree.size()` for a cell entirely inside the query without descending,
adds nothing for one outside, and assumes `(size+1)/2` at a leaf it cannot
descend past. `lucene_codecs::points` exposes no such thing —
`PointsReader::intersect`'s `IntersectVisitor::compare` sees cell *bounds* but
no size, `intersect` visits every document of a fully-inside cell (so using it
would cost exactly what the estimate exists to avoid), and `inner_nodes`/the
node-id walk are private.

**Handoff to whoever owns `crates/lucene-codecs/src/points.rs`**: the contained
change is a `PointsReader::estimate_point_count(field_number, &mut V) -> Result<i64>`
alongside `intersect`, reusing `IntersectCtx`/`intersect_node`'s existing
`node_id` bookkeeping and adding `BKDReader.IndexTree.size()` — for a full
binary tree over `num_leaves` leaves, a node's subtree leaf count is derivable
from its node id (Java's `leftMostLeafNode`/`rightMostLeafNode` walk), and
`size()` is that count times `max_points_in_leaf_node`, clamped at the field's
`point_count`. `estimate_doc_count` then consumes it unchanged and nothing in
`lucene-search` needs to move.

### 4.3 The brief's question: does a real estimate change the planner's choice?

**Yes, but only when another clause leads with fewer than `cost/8` documents —
and never when this query itself leads.** Java's rule is
`cost >>> 3 <= leadCost ? index : doc values`, so with `leadCost == cost` the
test is true for every non-negative cost: the estimate cannot move the plan off
`Index`, which is what c12's `None` default already picks. The estimate is not
wasted there — it is what turns "the default happens to be right" into "the
default is right" — but it changes nothing.

Pinned on the real `points_index` fixture (1 333 single-valued points over
2 000 documents), taking the exact match count as the estimate (which is what
Java's walk computes exactly for these ranges, since every cell a
fully-covering range touches is wholly inside or wholly outside it):

| lead cost | with the estimate | with `None` | agree? |
|---|---|---|---|
| `cost` (this query leads) | `Index` | `Index` | yes |
| `cost/8` | `Index` | `Index` | yes |
| `cost/8 - 1` | `DocValues` | `Index` | **no** |

So the single case the no-estimate default gets wrong is a conjunction whose
other clause is more than 8x more selective. The test also pins that a narrower
range has a proportionally smaller threshold, i.e. the estimate is doing real
work rather than scaling out of the comparison.

### Verdict

The consumer-visible half is ported and the planner's behaviour with a real
estimate is pinned; the BKD walk is a precise `lucene-codecs` handoff.

---

## 5. `crates/lucene-search/src/highlighter.rs` — `PhraseHelper`

Java: `search/uhighlight/PhraseHelper.java`.

| Rust | Java | verdict |
|---|---|---|
| `phrase_match_offsets` | `createOffsetsEnumsForSpans` + `OffsetSpanCollector.collectLeaf` + `SpanCollectedOffsetsEnum.add` | **added**, scoped (§5.2) |
| `offsets_from_phrase` | the same over a real `LeafReader` | **added** |
| — | the `WeightedSpanTermExtractor` constructor half (walking a `Query` tree to find position-sensitive sub-queries) | not ported, by design (§5.2) |
| — | `SingleFieldWithOffsetsFilterLeafReader`, `getAllPositionInsensitiveTerms`, `willRewrite` | not-in-Java-shape (this port has no `LeafReader` wrapper or query rewriting layer) |

### 5.1 `[MISSING → fixed]` phrase-match offsets

Both existing offset sources are position-*insensitive*: they return every
occurrence of every query term. For a phrase query that highlights words that
are not part of any phrase match — `"quick brown"` over
`the quick brown fox, a quick red fox` marks the second `quick` too.
`phrase_match_offsets` enumerates the document's phrase alignments and collects
only the occurrences inside them, per term, sorted and **deduplicated** exactly
as `SpanCollectedOffsetsEnum.add` does (which is what stops a position shared by
two overlapping matches being marked twice). A repeated phrase term shares one
output collection, as Java's collector keys by term bytes.

`offsets_from_phrase` is the postings-source wrapper, one
`occurrences_for_doc` per distinct term — c15/c20's skip-driven
single-document read, the same per-term cost `offsets_from_postings` pays.
The phrase filtering itself reads nothing further.

Verified against the `blocktree_index` fixture's `pos` field, whose manifest
carries real Lucene's own answer for which slops match
(`field.pos.sloppyGap.realLuceneSlopResults=0:false,1:false,2:true,3:true,5:true`):
at slop 0 the position-insensitive source returns both terms and this returns
nothing; at each slop Lucene says matches, this returns exactly the
position-insensitive set; at each slop Lucene says does not, nothing.

### 5.2 `[INTENTIONAL]` scope: the alignment walk, and the query walk

Match enumeration is this crate's existing **in-order** greedy alignment (one
alignment per starting position of the first slot), not Java's general
`SpanNearQuery`, which additionally allows the terms to appear reordered within
the slop budget. That is the same scope, and the same reason, this crate's
sloppy phrase *scoring* already documents (`phrase_matches_in_doc_sloppy`); at
`slop == 0` — the case `PhraseHelper` is overwhelmingly used for — the two
coincide exactly.

`PhraseHelper`'s other half, walking a `Query` tree to *discover* which
sub-queries are position-sensitive, has no counterpart here because this port
has no `Query` object graph to walk: its queries are functions, and the caller
that knows it is running a phrase query passes the phrase. Ported as a
`WeightedSpanTermExtractor` it would be a walk over a tree nobody builds.

### Verdict

Swept clean for the offset half, which is what c12 recorded. 5 new tests.

---

## 6. `crates/lucene-search/src/ordinal_map.rs` — the materialized input

Java: `index/OrdinalMap.java`.

### 6.1 `[PERF — measured, recorded, HANDOFF]` the input is 5x the map

Measured before changing anything, with `examples/ordinal_map_memory.rs` (new;
Linux RSS from `/proc/self/statm`, with the exact allocation totals printed
beside each figure so the two can be checked against each other — they agree to
within the allocator's slack), 17-byte terms:

| shape | materialized input | the map itself | peak RSS |
|---|---|---|---|
| 5 segments x 1 M terms, 1.2 M global | **267 MB** | 52 MB | 319 MB |
| 10 x 200 k, 380 k global | 107 MB | 20 MB | 127 MB |
| 20 x 50 k, 161 k global | 53 MB | 10 MB | 63 MB |

So the divergence c12 recorded as an aside is the **dominant** one: the term
lists are ~5x the map and ~84% of the peak, where Java holds neither (it
streams `TermsEnum`s and keeps only a packed map). c12's other recorded
divergence — `Vec<i64>` rather than `PackedLongValues` — is the 52 MB column,
i.e. the smaller half of what is already the smaller half.

**Not changed here, and it is not this file's change to make.** Closing it needs
a `TermsEnum`-shaped cursor over a doc-values terms dictionary in
`lucene-codecs` (`terms_dict::decode_all_terms` returns the whole dictionary,
and it is the only accessor), plus `facets.rs` to stop calling
`decode_all_terms`, plus `OrdinalMap::build` to take iterators. **Handoff to
`lucene-codecs`**: a `terms_dict::TermsCursor` with `next() -> Option<&[u8]>`
over an already-open `TermsDictEntry`, walking the prefix-compressed blocks the
existing decoder already walks but yielding one term at a time instead of
appending to a `Vec`. The numbers above are the justification and the
before-measurement for whoever takes it.

### 6.2 `[PERF → changed, no measurable time difference]` an allocation per distinct term

While measuring: the merge loop kept the previous popped term as an owned
`Vec<u8>` purely to compare against the next one — one heap allocation and one
copy **per distinct term**, 1.2 M of them on the largest shape, for a value
that is only ever compared and dropped. Every cursor's term already borrows
from the caller's own dictionary, so an `Option<&[u8]>` does the same job.

Re-running the harness: **104-111 ms before, 98-107 ms after** on the
5 x 1 M shape. That is within noise — the allocator handles a
same-size-repeatedly pattern well — so this is recorded as "1.2 M transient
allocations removed, no measurable time change", not as a speedup. Kept because
it is strictly less work and simpler code, not because it was faster.

### Verdict

Measured and recorded as the brief asked, with the saving quantified and the
blocker named. One contained cleanup, honestly reported as a wash.

---

## 7. `crates/lucene-search/src/facets.rs` — `FacetsConfig.build`

Java: `facet/FacetsConfig.java`.

| Rust | Java | verdict |
|---|---|---|
| `FacetsConfig::build_sorted_set_facet_fields` | `build(Document)`'s `processSSDVFacetFields` + `indexDrillDownTerms` + `checkSeen` | **added** |
| `DrillDownTermsIndexing` | `FacetsConfig.DrillDownTermsIndexing` | **added** |
| `DimConfig::drill_down_terms_indexing` | `DimConfig.drillDownTermsIndexing` (default `ALL`) | **added** |
| `BuiltFacetField` | the `Document` `build` returns | divergent by design (§7.2) |
| `FacetBuildError` | `build`'s three `IllegalArgumentException`s | **added** |
| — | `processFacetFields` (taxonomy) | not ported, out of scope (§7.3) |
| — | `processAssocFacetFields` | not ported, out of scope (§7.3) |

### 7.1 `[MISSING → fixed]` and the assessment the brief asked for

**It does not need an `IndexWriter` change, and it is not a handoff.** The SSDV
half of `FacetsConfig.build` is a pure transformation: `(dim, path)` plus a
`FacetsConfig` in, a list of doc-values values and drill-down terms out. Java
happens to package the result as a `Document` because that is its indexing API;
returning the fields instead keeps `lucene-search` free of any write-path
dependency and leaves the caller to add them to whatever document builder
exists when one does.

The rules, all Java's: a **hierarchical** dim indexes every prefix of the path
(`Path`, `Path/a`, `Path/a/b`), because a hierarchical count needs an ordinal
per level — which is exactly what makes this crate's own `DimTree` work; a
**flat** dim must have exactly one component beside the dim; a flat dim that is
both `multiValued` and `requireDimCount` also indexes the **bare dimension**,
which is the ordinal `dim_count` reads; and `indexDrillDownTerms` adds the
`StringField` terms its `DrillDownTermsIndexing` selects. A non-`multiValued`
dim may appear once per document.

This closes a real gap in the record too: `DimConfig` had no
`drill_down_terms_indexing` at all, so Java's **default** behaviour (`ALL` —
dimension, every sub-path, and the full path) was not representable.

Verified against real Lucene: `fixtures/src/GenFacets.java` now records, for
**eleven** configurations, the exact `SortedSetDocValuesField` values and
drill-down `StringField` terms `FacetsConfig.build(Document)` emitted — flat
default, flat multi-valued with and without `requireDimCount`, hierarchical at
depth 2 and 3, all five `DrillDownTermsIndexing` modes, two index field names in
one document, and a path component containing a `/`. A second test asserts the
fixture actually separates those branches (0/1/2/3/4 drill-down terms across
the modes, 4 values for a depth-3 hierarchical path, 4 for two labels under
`requireDimCount` against 2 without), so the differential cannot pass on a
`build` that only ever emitted the full path.

That fixture is also what makes the **read**-side tests in
`facets_fixtures.rs` non-circular: they decode an index Lucene wrote, and this
proves the port would have written the same one.

### 7.2 `[INTENTIONAL]` grouped output, and a deterministic order

Java groups by index field name in a `HashMap` and iterates its entries, so the
relative order of two *different* index fields is unspecified there. This
returns them in first-appearance order and groups per field. Nothing downstream
can observe the difference — doc-values values are sorted into a dictionary and
drill-down terms are indexed terms — and the differential test compares per
index field for exactly that reason.

### 7.3 `[INTENTIONAL]` the taxonomy and association halves

`processFacetFields` needs a `TaxonomyWriter` to turn a path into an ordinal
and a taxonomy index to store it; `processAssocFacetFields` needs an
association-reading facet source. Neither exists in this port, whose faceting
is SSDV-only end to end (`facets.rs`'s own module doc says so). Scope, not
oversight, and named as such in `docs/parity.md`.

### 7.4 `[not a finding]` `pathToString`'s escape branch is unreachable from `build`

Worth recording because it looks like missing coverage: the fixture case with a
`/` in a component does *not* exercise escaping, because Lucene's delimiter is
`U+001F` and `FacetField.verifyLabel` forbids the whole C0 range inside a
label. `path_components_to_string`'s escape branch is reachable only from a
caller building a path by hand, where it is already tested.

### Verdict

Swept clean; c12 §2.9 closed, and the "needs an `IndexWriter`" assumption in it
turned out to be false. 4 new tests.

---

## Cross-file notes

- **Files outside `crates/lucene-{search,ffi}/src/` changed**:
  `fixtures/src/GenBreakIterator.java` (analyzer offsets over non-ASCII text),
  `fixtures/src/GenFacets.java` (`FacetsConfig.build` cases),
  `crates/lucene-search/tests/{highlighter_utf16_offsets_fixtures.rs (new),
  highlighter_offsets_fixtures.rs, facets_fixtures.rs}`,
  `crates/lucene-search/examples/ordinal_map_memory.rs` (new),
  `docs/parity.md`. No file owned by `c27-arith-codecs-2` or `c28-arith-index`
  was touched.
- **Fixture regeneration, and a warning.** Running the whole
  `scripts/gen-fixtures.sh` rewrites **every** index with a fresh random
  segment id (`StringHelper.randomId()`), which invalidates the hardcoded
  segment ids in `lucene-ffi`'s tests and drops the `Append*Manifest` keys from
  the committed manifests. That happened once here and was fully reverted
  (`git checkout` of the 366 modified tracked files, then re-running the six
  `Append*Manifest` programs, which is what restores `blocktree_index`'s
  `scoring.*`/`fuzzy.*`/etc. keys). Both suites are green against the restored
  tree. **Regenerate one generator at a time** — compile `fixtures/src/*.java`
  and run just the class you changed — until `gen-fixtures.sh` grows a filter.
- **Several of this batch's files are untracked**, as much of the M2 sweep's
  output already was before it: `fixtures/src/{GenBreakIterator,GenFacets}.java`,
  `crates/lucene-search/tests/{facets_fixtures,highlighter_offsets_fixtures,
  highlighter_utf16_offsets_fixtures}.rs`,
  `crates/lucene-search/examples/ordinal_map_memory.rs`, and the regenerated
  `fixtures/data/{break_iterator,facets_index}/`. The commit that lands this
  must `git add` them — without the fixture data in particular, eight tests
  fail with "re-run scripts/gen-fixtures.sh".
- **Public API changes**: additive except three. `lucene-search`:
  `doc_value_query::search_numeric_range_sorted_by_field` gained a `sort_data`
  parameter and `multi_segment::DocValueSegment`'s `doc_values_data` became
  `range_data`/`sort_data` (§1.2, both forced by correctness);
  `highlighter`'s offsets changed unit (§2.1 — same type, different meaning,
  which is why it is called out here rather than buried). `lucene-ffi`: one new
  exported symbol, `ffi_segment_add_doc_values_generation`; no existing C
  signature changed.
- **New `crate::Error` variant**: `FieldExistsUnsupported(String)`.
- **No new dependency.**

## Gates

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-search -p lucene-ffi --all-targets -- -D warnings` —
  **exits zero**. Also clean for
  `--target aarch64-unknown-linux-gnu` (the `ffi-safety` skill's
  `c_char`-signedness check, run because this batch adds an exported function).
- `cargo test -p lucene-search -p lucene-ffi` — **1 567 passed, 0 failed**,
  including this batch's 34 new tests (10 in `lucene-ffi/src/segment.rs`, 7 in
  `doc_value_query.rs`, 4 in `facets_fixtures.rs`, 4 in the new
  `highlighter_utf16_offsets_fixtures.rs`, 5 in `highlighter_offsets_fixtures.rs`,
  2 in `highlighter.rs`, 2 in `points_query.rs`). **One caveat worth
  recording rather than hiding**: a single `lucene-ffi` lib test failed once
  mid-batch, in a run that coincided with another batch rebuilding
  `lucene-index` and (apparently) touching `fixtures/data`; the failing test's
  name was not captured, and **22 consecutive full runs since have been
  clean**. If it recurs, suspect a fixture directory being rewritten under a
  reading test rather than this batch's code — the two batches share
  `fixtures/data`.
- `python3 scripts/check-parity.py` — ok. Five `docs/parity.md` rows updated
  (`FacetsConfig`, `OrdinalMap`, the uhighlight offset strategies,
  `FieldExistsQuery`, `IndexOrDocValuesQuery`) plus the doc-values-updates row's
  "two gaps left open" paragraph.
- `python3 scripts/check-arith-allows.py` — ok (8 modules still unaudited, all
  in other crates; neither `lucene-search` nor `lucene-ffi` is gated).
- `cargo llvm-cov -p lucene-search -p lucene-ffi --summary-only` — every file
  this batch touched is above the 95%-per-file line bar: `doc_value_query.rs`
  99.44%, `facets.rs` 99.49%, `highlighter.rs` 98.55%, `points_query.rs`
  98.33%, `ordinal_map.rs` 98.11%, `field_norms.rs` 98.35%,
  `term_vectors_query.rs` 99.57%, `multi_segment.rs` 96.85%;
  `ffi/segment.rs` 97.34%, `ffi/sort.rs` 98.75%, `ffi/facets.rs` 97.48%,
  `ffi/range_sort.rs` 99.09%, `ffi/registry.rs` 100%,
  `ffi/results_fragments.rs` 99.79%, `ffi/highlighter.rs` 99.71%.

## Carry-overs opened by this batch

- [ ] **`lucene_analysis::Token`'s offsets are UTF-8 bytes where Lucene's are
      UTF-16 code units** (§2.2). The read side is now correct and the boundary
      converts, but a Rust-written index of non-ASCII text still carries
      offsets Lucene mis-slices. Owner `lucene-analysis`; the four call sites
      and the fixture to extend are listed in §2.2.
- [ ] **`PointValues.estimatePointCount`'s BKD walk** (§4.2). Owner
      `lucene-codecs/src/points.rs`; the shape of the addition is spelled out
      there, and `estimate_doc_count` consumes it unchanged.
- [ ] **A streaming `TermsEnum` cursor over a doc-values terms dictionary**
      (§6.1), worth **267 MB of a 319 MB peak** on a 5-segment x 1 M-term
      field. Owner `lucene-codecs/src/terms_dict.rs`, then `facets.rs` and
      `OrdinalMap::build`.
- [ ] **`check_index` reads the base `.fnm`** — c14's A2, untouched here and
      still `check_index.rs`'s.
- [ ] **`FieldExistsQuery.count`'s live-doc arithmetic and `rewrite`'s
      whole-reader decision** (§3.4). Both need a leaf list and a `numDocs`,
      i.e. a reader-level query layer this port does not have; the per-leaf
      predicate is ported and waiting.
- [ ] **`scripts/gen-fixtures.sh` has no way to regenerate one generator.**
      Every run rewrites every index with fresh segment ids and drops the
      appended manifest keys, so touching one fixture means reverting hundreds
      of files by hand. A `--only <Gen*>` flag (still running the `Append*`
      programs afterwards) would remove a genuine footgun.
