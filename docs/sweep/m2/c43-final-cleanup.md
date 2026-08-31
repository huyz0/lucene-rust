# c43-final-cleanup

The six remaining takeable items on the M2 sweep's prioritised list: **8c**
(span extents), **9**'s remaining half (the fuzzy clause's reader-wide
statistics), **11b** (100%-deleted segments not dropped), **23b** (the BKD
walk's per-node allocation), **23c** (no Java-written `IndexedDISI` jump table
is read) and **25b** (the deleter's `.si` read). Everything else on the list is
a sized milestone or blocked on a third-party crate — see the
[closing assessment](#closing-assessment).

| class | count |
|---|---|
| `CORRECTNESS` | **4** (F-1 span extents, F-2 the nested hit sets they produce, F-4 the fuzzy clause's per-segment blend, F-5 the boost multiplied into the score rather than the weight) |
| `MISSING` | **2** (F-3 `NearSpansOrdered` as a walk, F-6 the 100%-deleted segment drop) |
| `PERF` | **2** (F-7 the BKD walk's per-node allocation, F-8 the deleter's `.si` read) |
| `INTENTIONAL` | **2** (F-9 `keep_fully_deleted_segments` as a flag, F-10 the BKD stack grown on demand rather than sized from `num_leaves`) |
| met on the way | **1** (F-11 a test that asserted a race outcome and failed one run in five) |
| verified, no defect | **1** (F-12: the jump-table read side over Java-written bytes) |

Everything was fixed. Every fix has a test **verified to fail** against the
unfixed code, except F-12, where there was nothing to fix and the point was to
find that out.

---

## `crates/lucene-search/src/near_spans.rs` + `crates/lucene-search/src/lib.rs` (`span_near_matches`)

Java: `lucene/queries/src/java/org/apache/lucene/queries/spans/{NearSpansOrdered,NearSpansUnordered,ConjunctionSpans,Spans}.java`.

| Rust | Java | verdict |
|---|---|---|
| `for_each_unordered_match` | `NearSpansUnordered.twoPhaseCurrentDocMatches` + `nextStartPosition` | identical (extended: now reports `startPosition()`/`endPosition()`) |
| `for_each_ordered_match` | `NearSpansOrdered.twoPhaseCurrentDocMatches` + `nextStartPosition` + `stretchToOrder` + `advancePosition` | **new**, was missing |
| `span_near_matches` | `SpanNearQuery`'s per-document result | was a cartesian product; now the walks |
| `combine_span_clauses` | — | **deleted** |
| `unordered_width` | `SpanTotalLengthEndPositionWindow.atMatch()` over one arrangement | **deleted** — see F-3 |
| `span_doc_extents` | `SpanWeight.getSpans(ctx, POSITIONS)` walked out | **new** |
| `span_doc_ids` | the same, extents dropped | identical |
| Java's `GapSpans.skipToPosition` | — | not ported: `SpanNearQuery.addGap` has no equivalent in this port's `SpanQuery` enum |

### F-1 `[CORRECTNESS]` the extents were a cartesian product's, not a walk's — fixed

`NearSpansOrdered` and `NearSpansUnordered` are `Spans` **iterators**, and
their sub-span cursors only ever move forward. The sequence of
`(startPosition(), endPosition())` pairs they emit is therefore a strict subset
of "every arrangement of one span per clause that fits the slop budget", which
is what this port enumerated:

- `stretchToOrder` advances each later clause with
  `while (spans.startPosition() < position) spans.nextStartPosition()` and
  never rewinds, so each position of clause 0 contributes exactly **one**
  arrangement — the minimum-slop one. Its class doc says so: *"The formed spans
  only contains minimum slop matches."* On this port's own fixture,
  `SpanNear([fox, dog], 8, true)` over segment-1 document 0 has 20 extents in
  Lucene and 32 in the product.
- `NearSpansUnordered.endPosition()` returns `spanWindow.maxEndPosition`, a
  **running** maximum never recomputed when the span that set it moves on, so a
  reported extent can end past every span the arrangement actually holds.

Both are now reproduced. `for_each_ordered_match` is `stretchToOrder` statement
for statement, including the detail that made the old code's shape wrong:
**one exhausted sub-span ends the document**, not just the current candidate
(`oneExhaustedInCurrentDoc` gates the enclosing `while`). That is an early exit
rather than a lost match — every cursor is monotone and `prevSpans.endPosition()`
only rises with clause 0's position — and it is recorded on the function.

### F-2 `[CORRECTNESS]` and it is a **hit set** difference, for a nested `SpanNear` — fixed

c40 recorded this as "no hit set is affected". That is true of a flat
`SpanNear` — `span_doc_ids` only asks whether the list is non-empty — and false
of a nested one, which is the case the entry itself named: an outer clause
*consumes* the inner clause's extents.

Finding a discriminating query by hand did not work (the fixture's regular
40-token documents make the obvious candidates agree), so both algorithms were
modelled over the fixture's own position lists and the space of
`n(slop, inOrder, n(slop, inOrder, t, t), t)` queries searched for a
disagreement; 660 were found, and one per `(inner inOrder, outer inOrder)`
combination that has one was then recorded **from real Lucene**. All three
return segment-1 document 0 under the product and do not under Lucene, e.g.

```
n(2,true,n(2,false,t(body,fox),t(body,cat)),t(body,cat))   Lucene [1, 2]   product [0, 1, 2]
```

The search found **no** case in the other direction (a document the walk
matches and the product does not), which is what the containment argument
predicts.

### F-3 `[MISSING → fixed]` `NearSpansOrdered` had no module, and `unordered_width` was a trap

The in-order walk existed twice, hand-written: once in
`highlighter::phrase_match_offsets`' `slop == 0` arm and once inside
`combine_span_clauses`. Both are gone; both consumers now call
`near_spans::for_each_ordered_match`, so they cannot drift.

`unordered_width` was **deleted rather than kept**. It computed `atMatch()`'s
formula over one arrangement, and the walk's `atMatch()` is not that number:
`maxEndPosition` is a running maximum and `totalSpanLength` is maintained
incrementally. A helper that returns a subtly different number from the thing
it is named after is exactly the defect this item was, and keeping it as a
"test-only convenience" would have preserved the trap. `docs/parity.md`'s two
rows moved with it (`check-parity`'s `::item` rule caught both on the first
run, which is that gate doing its job).

### Evidence

`fixtures/src/AppendSpanExtentManifest.java` (new) walks real
`SpanWeight.getSpans(ctx, Postings.POSITIONS)` to
`NO_MORE_DOCS`/`NO_MORE_POSITIONS` and records every pair, per document, per
leaf, for 23 query shapes over the committed `multi_segment_scoring_index`.

Two choices in it are load-bearing:

- **Which index.** `blocktree_index`'s `pos` field — the one every existing
  span fixture uses — has two occurrences per document, and the walk and the
  product agree on every query over it (checked). `GenMultiSegmentScoring`'s
  `longBody` writes 40 tokens drawn from a four-term vocabulary, so a term
  occurs up to twenty times in one document. It was appended to without
  regenerating the index (`--append-only`), so no segment id moved.
- **How the query is recorded.** Each case is a tiny S-expression
  (`t(field,term)`, `n(slop,inOrder,child,…)`, `o(child,…)`) that *both* the
  Java generator and the Rust test parse, so the recorded query and the tested
  query are the same text rather than two hand-kept-in-sync constructions.

`crates/lucene-search/tests/span_extent_fixtures.rs` asserts the extent
sequences (46 comparisons) and the nested hit sets separately, and additionally
asserts that Lucene's own sequences are strictly increasing — so this port's
sort-and-dedup cannot be hiding a reordering. **Both tests verified to fail**
against the cartesian product: the extents on `ordered_fox_dog_slop8`, the hit
set on `nested_hitset_unordered_inner` (`[0, 1, 2]` against Lucene's `[1, 2]`).

### Verdict

Swept clean. Ledger item 8c closed.

---

## `crates/lucene-search/src/lib.rs` (`fuzzy_expanded_terms`, `fuzzy_doc_scores`) + `crates/lucene-search/src/multi_segment.rs`

Java: `lucene/core/src/java/org/apache/lucene/search/{FuzzyQuery,FuzzyTermsEnum,TopTermsRewrite,TermCollectingRewrite,MultiTermQuery,BlendedTermQuery,BM25Similarity}.java`.

| Rust | Java | verdict |
|---|---|---|
| `FuzzyTopTerms::collect_segment` | `TermCollectingRewrite.collectTerms`' per-leaf loop + `TopTermsRewrite`'s `TermCollector.collect` | **new**; was one leaf only |
| `FuzzyTopTerms::tighten` | `FuzzyTermsEnum.bottomChanged` | identical, now also run per-leaf at construction |
| `FuzzyTopTerms::finish` | `BlendedTermQuery.rewrite`'s `df = max(df, ctx.docFreq())` | identical, now reader-wide |
| `multi_segment::global_fuzzy_stats` | `MultiTermQuery.rewrite(IndexSearcher)` | **new** |
| `GlobalStats` | `TermStatistics` + `CollectionStatistics` | widened to carry a fuzzy clause's whole expansion |
| `fuzzy_doc_scores` | `BooleanQuery` of `BoostQuery(TermQuery)` under `BM25Scorer` | divergent on the boost — see F-5 |
| `BlendedTermQuery.DISJUNCTION_MAX_REWRITE` | — | not reachable: `FuzzyQuery` always uses `BOOLEAN_REWRITE` |
| `TopTermsRewrite.getMaxSize` (`IndexSearcher.getMaxClauseCount`) | — | not ported: no clause-count ceiling exists in this port |

### F-4 `[CORRECTNESS]` the fuzzy expansion was per-segment — fixed

The entry recorded this as "`fuzzy_doc_scores` blends `docFreq` within one
segment". Reading Java showed it is **larger than a statistic**:
`TermCollectingRewrite.collectTerms` drives *one* `TopTermsRewrite` collector
across `topReaderContext.leaves()`, so

- the `maxExpansions` queue selects one term set for the whole reader, not one
  per leaf;
- a term already in the queue has its `docFreq` **accumulated** across leaves
  (`t.termState.register(...)`), and a term evicted and later re-encountered
  loses what earlier leaves contributed — Java's own behaviour, reproduced;
- the rejection test runs *before* the visited-term lookup, so a term already
  in the queue but no longer competitive is skipped without accumulating;
- the `MaxNonCompetitiveBoostAttribute` lives on the collector's shared
  `AttributeSource`, so the published bottom carries from one leaf's
  `FuzzyTermsEnum` to the next, and each new enum re-tightens from it in its
  constructor (`bottom = maxBoostAtt.getMaxNonCompetitiveBoost(); bottomChanged(null)`).

One detail changed in the single-segment path too, and it only matters
across leaves: the competitiveness test is Java's
`boost == bottom.boost && bytes.compareTo(bottom.bytes) > 0 → skip`, i.e.
`term <= bottom.term` is competitive. This port had `term < bottom.term`.
Within one leaf a term dictionary yields each term once so the equal case is
unreachable; across leaves it is the bottom term arriving again, which Java
routes into the accumulate branch.

`GlobalStats` became a struct with two maps rather than one. A fuzzy clause's
statistic is keyed by the *query*, not by a `(field, term)` pair — a
`FuzzyQuery` and a `TermQuery` on the same `(field, term)` can appear in one
boolean query and mean different things — and it carries the whole expansion,
not a `docFreq`/`docCount` pair.

### F-5 `[CORRECTNESS]` the boost was multiplied into the score, not the weight — fixed

Found by the fixture, not by reading: the first run was one ULP out on one
document. `BM25Similarity.scorer(boost, …)` folds the boost into the
**weight** (`this.weight = boost * idf.getValue().floatValue()`) and
`BM25Scorer.score` then evaluates `weight - weight / (1 + freq * normInverse)`.
This port computed `boost * score(...)`. The two are algebraically equal and
bitwise different — which is the rewrite `similarity::do_score`'s own doc
comment exists to warn about, and which every *other* scored path in this port
already got right. `fuzzy_doc_scores` now goes through `do_score` directly.

### Evidence

`fixtures/src/AppendMultiSegmentFuzzyManifest.java` (new) records, for 13
`FuzzyQuery` shapes over the two-segment `multi_segment_scoring_index`: the
rewritten query's own selected terms and boosts (walked out of the
`BooleanQuery` `BlendedTermQuery.BOOLEAN_REWRITE` produces — including the
single-clause case, where `searcher.rewrite` collapses it to a bare
`BoostQuery`), each selected term's reader-wide `docFreq`, and real
`IndexSearcher` `TopDocs` as `Float.floatToIntBits`.

The fixture separates the two answers: `dog`'s `docFreq` is 3 in *each* leaf
and **6** for the reader, so `max(df)` over the selected terms is 3 per leaf
and 6 reader-wide, and `fog~1` selects both `dog` and `fox` so the fold is a
real choice rather than the only number available. The test asserts
`saw_blend_beyond_any_leaf`, so a corpus that flattened out would fail rather
than pass vacuously.

`crates/lucene-search/tests/multi_segment_fuzzy_fixtures.rs` asserts the
selection and the scores separately — a score comparison alone would let a port
pick a different term set and land on the same numbers by luck. **Verified to
fail** against the per-segment blend: doc 4 scored 0.40036508 where Lucene
scores 0.36528432.

### Verdict

Swept clean. Ledger item 9 closed (both halves).

---

## `crates/lucene-index/src/index_writer.rs` (`apply_all_deletes_and_updates`) + `merge_policy.rs`

Java: `lucene/core/src/java/org/apache/lucene/index/{IndexWriter,PendingDeletes,PendingSoftDeletes,MergePolicy,SoftDeletesRetentionMergePolicy}.java`.

| Rust | Java | verdict |
|---|---|---|
| `IndexWriter::drop_fully_deleted_segments` | `IndexWriter.finishApply`'s `result.allDeleted()` loop | **new**, was missing |
| `apply_packets_to_segment`'s return | `closeSegmentStates`' `rld.isFullyDeleted()` | **new** |
| `MergePolicyConfig::keep_fully_deleted_segments` | `MergePolicy.keepFullyDeletedSegment(IOSupplier<CodecReader>)` | reduced to a flag — see F-9 |
| `dropDeletedSegment`'s `mergingSegments` guard | | not modelled: no concurrent merges here |
| `readerPool.drop`, `adjustPendingNumDocs` | | not modelled: neither exists in this port |

### F-6 `[MISSING → fixed]` a 100%-deleted segment stayed in the commit forever

`IndexWriter.finishApply` removes every segment `closeSegmentStates` found
fully deleted:
`rld.isFullyDeleted() && mergePolicy.keepFullyDeletedSegment(...) == false`,
where `PendingDeletes.isFullyDeleted` is `getDelCount() == info.info.maxDoc()`
— **hard deletes only**, which is why `PendingSoftDeletes` overrides the
*policy* hook rather than the count. This port kept such a segment forever: a
segment nothing can ever match, carried by every later open, merge and
`CheckIndex`.

Only segments this apply touched are candidates, exactly as in Java
(`openSegmentStates` filters to the segments the packets may reach), which
`apply_packets_to_segment` reproduces by returning `false` for a segment it
skipped. The drop is in-memory plus the deleter checkpoint the flush already
runs — Java's `checkpoint()` is `changed(); deleter.checkpoint(infos, false)`,
not a new `segments_N` — so a `rollback()` restores the segment from
`rollback_segments` unchanged.

### F-9 `[INTENTIONAL]` the hook is a flag

Java's `keepFullyDeletedSegment` takes an `IOSupplier<CodecReader>` and may run
a query over the segment. This port has no `MergePolicy` trait and no reader
supplier to hand it, and the hook's two in-tree implementations are "always
false" (the base class, which `TieredMergePolicy` does not override) and "true
iff a retention query matches" (`SoftDeletesRetentionMergePolicy`). A boolean
the caller sets is the honest reduction; it is documented as such on the field,
with the note that it becomes a method when a reader-aware policy arrives.
`keep_fully_deleted_segments_suppresses_the_drop` pins it.

### The three recorded blockers, all false or trivial

This is the sixth entry in this sweep whose stated reason for deferral turned
out to *be* the work, and the first where all three stated reasons were:

| c36 recorded | what it was |
|---|---|
| (a) "`MergePolicyConfig` has no `keepFullyDeletedSegment` hook, and it is not decoration" | one `bool` with a default of `false`, plus one test |
| (b) "no `adjustPendingNumDocs`/reader-pool bookkeeping to update" | nothing to do — the counter and the pool do not exist here; both are now named on the function as deliberately unmodelled |
| (c) "at least one existing test asserts on `segments[0]` after deleting that segment's only document" | **four** tests, each fixed by adding one document the delete does not match, which made three of them *sharper* |

The fourth, `a_delete_applies_to_the_segments_flushed_before_it_and_not_to_the_ones_after`,
now asserts the drop itself: `_0` vanishing is the proof that *both* its
documents were reached, and `_1` surviving is the proof the delete did not
reach it — a delete that wrongly reached both would leave zero segments.

### Evidence

`fixtures/src/GenFullyDeletedDrop.java` (new) runs four scripts through a real
`IndexWriter` in a `ByteBuffersDirectory` and records the committed segment
count, each segment's `(maxDoc, delCount)` and the visible ids. No index is
committed to `fixtures/data/`, deliberately: the bytes of an index with a
dropped segment are indistinguishable from those of one that never had it, so
the *behaviour* is what there is to record. The four are `drop`, `partial` (the
control — one of two documents deleted, so it must **not** drop), `all` (every
segment emptied, so the commit is empty — the case a fix that only looked at
the first segment would still pass `drop` on) and `block`
(`updateDocuments` replacing a whole block).

`a_fully_deleted_segment_is_dropped_exactly_where_real_lucene_drops_it`
replays all four through this port's writer. **Verified to fail** against the
unfixed code (2 segments where Lucene commits 1).

### Verdict

Swept clean. Ledger item 11b closed.

---

## `crates/lucene-codecs/src/points.rs` (the BKD walk)

Java: `lucene/core/src/java/org/apache/lucene/util/bkd/BKDReader.java` (`BKDPointTree`).

| Rust | Java | verdict |
|---|---|---|
| `IntersectCtx::stack` | `BKDPointTree`'s `splitValuesStack`/`splitDimValueStack`/`negativeDeltas` | **new**; was per-node allocation |
| `clamp_bound`/`restore_bound` | `BKDPointTree.pushLeft`/`pushRight`/`pop`'s cell-bound half | identical |
| `InnerNode` | `readNodeData`'s decoded state | now offsets only, `saved_split_tail`/`split_value` moved to the stack |
| `getTreeDepth` | | not ported — see F-10 |

### F-7 `[PERF → fixed]` four heap allocations per inner node visited

`read_inner_node` returned an `InnerNode` owning `saved_split_tail` and
`split_value`, and both callers additionally `to_vec()`'d the cell bound they
were about to overwrite. Java preallocates per-level stacks for exactly this
state. The two cell bounds share one buffer here, because the walk never holds
both: `intersect_node` restores `max` before it saves `min`.

**Measured** — `crates/lucene-codecs/examples/bkd_walk_scratch.rs`, min of 40
alternating repetitions, both arms in one process from one build (the arm is
`intersect_with_scratch`'s `reuse_scratch`, so the "before" is shipped code and
cannot be a stale binary — `c42`'s finding):

| tree | `intersect` | `estimate_point_count` |
|---|---|---|
| 1D, 200k points, leaf 512 | 92.8 → 96.3 µs (1.04x) | 1.5 → 1.9 µs (**1.27x**) |
| 2D, 200k points, leaf 512 | 640.6 → 645.5 µs (1.01x) | 6.1 → 9.9 µs (**1.63x**) |
| 2D, 200k points, leaf 64 | 317.0 → 330.2 µs (1.04x) | 18.3 → 30.7 µs (**1.68x**) |
| 4D, 100k points, leaf 64 | 817.6 → 834.5 µs (1.02x) | 53.9 → 73.9 µs (**1.37x**) |

(reuse → realloc; the ratio is how much the old shape cost.)

`estimate_point_count` is where it lands, which is the point c39's review made:
it reads no `.kdd` at all, so the allocation was most of what it did.
`intersect` is dominated by leaf decode, and 1.01x–1.04x is the honest report
of it. `the_two_scratch_arms_visit_exactly_the_same_documents` asserts the two
arms are one walk (same documents, same `compare` count, same estimate) over a
3-dimensional tree — the buffers are indexed by split dimension, and a 1D tree
would restore the one dimension it has even with the level indexing wrong.

### F-10 `[INTENTIONAL]` the stack grows on demand rather than being sized from `num_leaves`

The ledger's sizing was "depth is bounded at ~31 by `child_ids`' `checked_mul`,
so per-level scratch sizes trivially". It does — but sizing it up front means
`depth × packed_index_bytes_length`, and `bytesPerDim` comes off disk. A
crafted `.kdm` could turn a plausible allocation into a 32x one, which is the
row `docs/arithmetic-gate.md` singles out as the one that *aborts* rather than
merely being wrong. Growing one entry per level actually reached bounds the
memory by the real tree instead, costs `O(depth)` allocations per query rather
than `O(nodes)`, and is recorded on the field with this reasoning.

### Verdict

Swept clean. Ledger item 23b closed.

---

## `crates/lucene-codecs/src/indexed_disi.rs` (read side, over Java-written bytes)

Java: `lucene/core/src/java/org/apache/lucene/codecs/lucene90/IndexedDISI.java`.

### F-12 `[verified, no defect]` the jump table's read side, over bytes Lucene wrote

c39 ported the table in both directions and proved the *write* side against
real Lucene. The read side had never run over Java-written bytes, for a fixture
reason: `IndexedDISI.writeBitSet` emits `jumpTableEntryCount = 0` below two
logical 65 536-document blocks, and the largest Java-written index in the tree
was 36 000 documents. It is the only direction of that format not covered by
real bytes, and this sweep has twice found a writer and a reader agreeing on a
shared mistake — the FST framing bug and the invented `.si` sort encoding —
precisely where only one direction had been checked.

`fixtures/src/GenDisiJumpTable.java` (new) writes 200 000 documents:
`sparse` on every third (four DENSE blocks) and `very_sparse` on every
20 000th (SPARSE blocks, last logical block empty, which is `flushBlockJumps`'
empty-block fill). Real Lucene emits a four-entry table for each. No indexed
field at all — an `id` term would add a 900 KB term dictionary to a fixture
whose subject is the `.dvd`.

`crates/lucene-codecs/tests/disi_jump_table_fixtures.rs` (new) has four tests:

1. `real_lucene_emitted_a_block_jump_table_for_both_fields` — pins
   `jumpTableEntryCount > 2`, so a regenerated corpus that fell below two
   blocks fails loudly instead of testing nothing.
2. `cold_seeks_through_a_java_written_jump_table_match_lucenes_values` — a
   fresh cursor per probe, which is `Lucene90DocValuesProducer`'s own shape and
   the only pattern `advanceBlock` consults the table for; it also counts how
   many probes were ≥2 blocks in, so the jump-table branch cannot go untaken.
3. `a_full_scan_...` — the complement, which never touches the table, against
   Lucene's own cardinality and value checksum.
4. `corrupting_the_java_written_jump_table_changes_the_answer` — **the
   assertion that makes the other three non-vacuous.** Each half of a table
   entry (the index, i.e. the cardinality before the block; and the offset,
   i.e. where the block header starts) is perturbed independently and the
   answer must change. Without it, a reader that ignored the table and walked
   the block headers would pass everything above.

**No divergence found.** That is the outcome only running it can establish, and
it is why the item was worth doing.

### Verdict

Swept clean. Ledger item 23c closed.

---

## `crates/lucene-index/src/index_file_deleter.rs` + `segment_writer.rs`

Java: `lucene/core/src/java/org/apache/lucene/index/{IndexFileDeleter,SegmentInfos,SegmentCommitInfo}.java`.

| Rust | Java | verdict |
|---|---|---|
| `IndexFileDeleter::record_segment_files` | `SegmentCommitInfo` owning its `SegmentInfo` | **new** (structural in Java, explicit here) |
| `ensure_si_files` | the fallback for a segment this process did not write | identical |
| `commit_files` | `SegmentInfos.files(boolean)` | identical, now borrow instead of clone |
| `seal_flushed_segment` | `IndexWriter.sealFlushedSegment` | now returns the file list beside the commit |

### F-8 `[PERF → fixed]` the `.si` read in the flush path — and a stale premise

The entry said the deleter "re-reads and re-parses every segment's `.si` ...
once per segment per checkpoint". **That had stopped being true**: c36 gave the
deleter a per-`(segment_name, segment_id)` cache, so the parse was once per
segment *ever*. Verifying against the tree rather than the entry is what the
ledger's own preamble asks for, and it changed what the work was.

What was actually left, and is now gone:

1. **The first read.** `record_segment_files` takes a segment's file list
   straight off the in-memory `SegmentInfo` that `seal_flushed_segment` just
   encoded (which now returns it beside the `SegmentCommitInfo`). The recorded
   blocker — "a signature change across `index_file_deleter.rs` and every
   checkpoint call site" — did not materialise, because the writer *pushes* the
   list in at seal time instead of the deleter *pulling* it at checkpoint time:
   **no checkpoint call site changed.**
2. **A per-checkpoint clone.** `si_files_for` returned an owned `Vec<String>`
   because `&mut self` and a borrow of its result cannot coexist. A
   fill-then-borrow two-pass removes it: one `Vec<String>` per segment per
   checkpoint, on a path that runs twice per commit and reads nothing.

**Evidence, countable**: c36's own counting-`Directory` test now asserts
**zero** `.si` reads during a flush, down from the one it pinned — and it was
that test failing that reported the change, which is a gate doing exactly what
it was built for.

**Measured** — `crates/lucene-index/examples/deleter_checkpoint.rs`, min of 40
alternating repetitions, both arms shipped code in one process (arm B is
`forget_segment_files`, i.e. the pre-c43 cache state):

| segments | from memory | from `.si` | ratio | per segment |
|---|---|---|---|---|
| 8 | 10.8 µs | 25.0 µs | **2.30x** | 1.77 µs |
| 32 | 42.0 µs | 98.7 µs | **2.35x** | 1.77 µs |
| 128 | 170.1 µs | 393.1 µs | **2.31x** | 1.74 µs |

Flat per segment, as it must be: one `open` plus a full `segment_info::parse`,
index-header check and a CRC over the whole file included.

**Not fixed, and reported anyway**: a *merged* segment's `.si` is still parsed
once by the deleter. `execute_merge` writes it through a different path that
does not carry a `FlushedSegment`, and the flush path is what item 25b named;
one parse per merge output is a fraction of what a merge costs.

### Verdict

Swept clean. Ledger item 25b closed.

---

## One thing met on the way

### F-11 `[CORRECTNESS]` a test that asserted a race outcome — fixed

`multi_segment::tests::the_shared_max_score_fan_out_returns_what_the_plain_one_does`
failed the gate's coverage run, and then **one run in five** when re-run alone.
It is not flaky noise, and it is the same shape c40 found in
`registry::a_handle_with_the_wrong_shard_bits_is_rejected`: the test asserted
that leaf 1 *observed* leaf 0's published pruning bar, and **that is not a
property the design has**. The `MaxScoreAccumulator` is opportunistic; under
`rayon` leaf 1 may finish before leaf 0 publishes anything, and the recorded
failure shows exactly that (`(1, None), (1, Some(2.0)), (1, Some(2.0))`).

Split in two, both deterministic:

- the fan-out test keeps the **result** invariant (`shared == plain`, same
  hits), which holds however the leaves interleave, and says in a comment why
  it no longer asserts the observation;
- `a_leaf_sees_a_bar_another_leaf_published_through_the_accumulator` pins the
  mechanism at its seam — one accumulator, two `TopDocsCollector`s wired to it
  by hand, no threads — plus a negative control (the same leaf without the
  accumulator can only ever see its own queue's bottom).

Not attributable to this batch's changes: reproduced against the unchanged
code.

---

## Gate

`scripts/docker-test.sh gate` → **`gate: ok`**. Workspace **98.12% lines /
97.55% regions**, unchanged. Every file this batch touched is above invariant
#8's per-file bar:

| file | lines |
|---|---|
| `lucene-search/src/near_spans.rs` | **100.00%** |
| `lucene-search/src/lib.rs` | 96.88% |
| `lucene-search/src/multi_segment.rs` | 97.07% |
| `lucene-search/src/highlighter.rs` | 98.58% |
| `lucene-search/src/query.rs` | 98.37% |
| `lucene-search/src/similarity.rs` | 99.50% |
| `lucene-search/src/collector.rs` | 99.04% |
| `lucene-codecs/src/points.rs` | 99.07% |
| `lucene-codecs/src/indexed_disi.rs` | 98.64% |
| `lucene-index/src/index_writer.rs` | 98.30% |
| `lucene-index/src/index_file_deleter.rs` | 98.56% |
| `lucene-index/src/segment_writer.rs` | 99.49% |
| `lucene-index/src/merge_policy.rs` | 99.48% |
| `lucene-ffi/src/writer.rs` | 97.75% |

Also green: `scripts/verify-write-path.sh` **23/23**;
`scripts/gen-fixtures.sh --check` (49 deterministic files byte-identical, 0
mismatches, 0 missing, 0 extras, 0 manifest key-set differences, 0 segment-id
disagreements). The only fixture bytes that moved are the two new directories
and the two appended key blocks in `multi_segment_scoring_index/manifest.properties`;
`fixtures/segment-ids.txt` gained two lines and changed none, which is the
readable proof that no existing index was regenerated.

---

## Closing assessment

**The sweep's find-and-fix phase is done.** What remains on the prioritised
list is four entries, and every one of them is a sized milestone or a
dependency problem — not an unfinished finding:

| item | what it is | why it is not a batch |
|---|---|---|
| **5** | `TwoPhaseIterator`/`matchCost`/`ScorerSupplier.cost()`, and the `Similarity`/`SimScorer` hierarchy | needs every clause expressed as `(approximation, matches(), matchCost())`, i.e. turning the per-shape free functions into a scorer enum. Assessed as a milestone by c6 **and** c11 independently. The contained part (ordering conjunction clauses by `docFreq` ascending) is batch-sized on its own and is still available. |
| **10** | no case-insensitive `CharArraySet`, no `maxTokenLength` | `lucene-analysis` API changes; the half a caller can observe as a wrong position (the `TokenStream` lifecycle) was closed by c40. |
| **12** | `TopNSearcher`'s FST output-pushing | c39 established that porting `Util.shortestPaths` onto this builder's FSTs would be a **wrong answer**, not a speed-up: its pruning is admissible only over outputs `FSTCompiler` has pushed toward the root. Needs `Outputs::common`/`subtract`, output pushing in `build_fst` (which changes the bytes it emits) and an `Outputs`-aware `output_add` threaded through the reader. Sized in `c39-codecs-readpath.md`. |
| **15** | the `ByteBlockPool` family | ten Java classes, 2 800 lines, all three consumers of `InMemoryInvertedIndex` moved onto a `TermsEnum`-shaped read-back, and it only lands fully with a borrowed-token `Analyzer` API. c38 measured it and **declined it deliberately** rather than half-building a pool. |
| **25** | DEFLATE preset dictionary | `miniz_oxide` exposes no `deflateSetDictionary`. Blocked on a dependency; compression ratio only, and the decode side is correct and fixture-verified. |
| **27 (c), (e)** | two record-keeping rules | `docs/parity.md` prose going stale, and "a key a generator writes and no test reads is ground truth that does not exist". Both are habits, not code. |

I have not found anything still outstanding that is a *finding* rather than a
milestone. The four things worth saying about that claim:

1. **It is a claim about the prioritised list, which is the only list.**
   `check-port-invariants.py --only=ledger-single-list` makes that mechanical:
   the ledger cannot carry an unticked box outside it, and every archive twin
   of the six items closed here was updated in the same change.
2. **The batches were finding less and less, and this one found the least.**
   c39/c40/c41 each raised two to five new items while closing theirs; this one
   raised none. The two `CORRECTNESS` findings it *did* add (F-5's boost, F-11's
   racy test) were both surfaced by evidence built for something else, which is
   what a sweep looks like when the deliberate search has stopped paying.
3. **One direction of one format has now been checked in both directions**
   (F-12) and found clean. That was the last "we only ever checked this one
   way" item on the list. The general habit it came from — write a fixture for
   the direction nobody has run — is worth keeping, but there is no specific
   instance of it left recorded.
4. **The recorded-blocker pattern is the finding to carry forward, not a code
   defect.** Six of the last eight batches found the stated reason for
   deferring an item to be the whole of the work, and this batch found all
   three of one item's blockers false at once. The ledger already says "a
   recorded blocker is a claim with an expiry date"; the evidence is now strong
   enough that the right default is to *re-verify a blocker before planning
   around it*, every time, which costs an hour and has repeatedly saved a
   batch.

Not committed, per the batch instruction.
