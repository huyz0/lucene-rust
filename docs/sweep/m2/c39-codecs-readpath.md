# c39-codecs-readpath — the remaining `lucene-codecs` read-path items

Four items from `LEDGER.md`'s "Open work, prioritised", all rooted in
`lucene-codecs`: the blocktree `try_*` migration (item 20 / c1's F-11),
`IndexedDISI`'s block jump table (item 22 / c2's #7),
`PointValues.estimatePointCount`'s BKD walk (item 13 / c29's §4.2 handoff), and
`Util.shortestPaths`/`TopNSearcher`/`readCeilArc` (item 12 / b8's F31).

Three are closed. The fourth is **restated**: its recorded blocker names the
wrong thing, and porting the named class onto this port's FSTs would produce a
wrong answer rather than a speed-up. It is sized and declined, as
`c38-allocation-shape` did with item 15.

Files swept:

- `crates/lucene-codecs/src/blocktree.rs` — the four infallible lookups and the
  `Intersect` iterators
- `crates/lucene-codecs/src/indexed_disi.rs` — `writeBitSet`'s jump table,
  `createBlockSlice`/`createJumpTable`, `advanceBlock`
- `crates/lucene-codecs/src/points.rs` — `estimatePointCount`,
  `BKDPointTree.size()`
- `crates/lucene-codecs/src/suggest.rs` + `fst.rs` — assessed, not changed
- `crates/lucene-codecs/src/{doc_values,norms,vectors}.rs` — the three
  `IndexedDISI` write sites and the five read sites
- `crates/lucene-search/src/{lib,explain,weight_count,multi_segment,
  points_query,field_norms}.rs` — the migrated callers and the estimate's
  consumer
- `crates/lucene-index/src/check_index.rs`, `crates/lucene-ffi/src/query.rs` —
  ripple
- `fixtures/src/{AppendPointEstimateManifest.java,
  VerifySparseNumericDocValues.java}`, `scripts/gen-fixtures.sh`,
  `crates/lucene-codecs/examples/disi_jump_table.rs`

Java counterparts, all under `/home/tuong/work/lucene-10.5.0/lucene/`:
`core/src/java/org/apache/lucene/codecs/lucene90/IndexedDISI.java`,
`core/src/java/org/apache/lucene/index/PointValues.java`,
`core/src/java/org/apache/lucene/util/bkd/BKDReader.java`,
`core/src/java/org/apache/lucene/search/PointRangeQuery.java`,
`core/src/java/org/apache/lucene/util/fst/Util.java`,
`core/src/java/org/apache/lucene/codecs/lucene103/blocktree/
Lucene103BlockTreeTermsReader.java` (`SegmentTermsEnum`, `IntersectTermsEnum`),
`suggest/src/java/org/apache/lucene/search/suggest/fst/
WFSTCompletionLookup.java`.

---

## Counts

| class | count |
|---|---|
| `CORRECTNESS` | **1** (F-2: the intersect iterators had no error channel at all) |
| `MISSING` | **4** (F-1 the `try_*` migration, F-3 the jump table write side, F-4 the jump table read side, F-6 `estimatePointCount`) |
| `PERF` | **3** (F-5 the jump table's measured win, F-7 the estimate's cost model, F-8 the suggester's linear walk — sized and declined) |
| `INTENTIONAL` | **2** (F-9 the infallible spellings kept for tests, F-10 `readCeilArc` has no consumer in scope) |

Every `CORRECTNESS` and `MISSING` finding is fixed with a test verified to fail
against the unfixed code.

---

## `crates/lucene-codecs/src/blocktree.rs`

Java counterpart:
`codecs/lucene103/blocktree/Lucene103BlockTreeTermsReader.java`
(`SegmentTermsEnum.seekExact`/`next`/`seekCeil`, `IntersectTermsEnum.next`).

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `FieldTerms::try_seek_exact` | `SegmentTermsEnum.seekExact` + `docFreq`/`totalTermFreq` | identical; **now the only production spelling** |
| `FieldTerms::seek_exact` | *(none — Java throws)* | INTENTIONAL, test-only (F-9) |
| `TermsEnum::try_next` / `try_seek_ceil` / `try_current` | `TermsEnum.next`/`seekCeil`/`term`+`docFreq` | identical; now the only production spellings |
| `TermsEnum::next` / `seek_ceil` / `current` | *(none)* | INTENTIONAL, test-only (F-9) |
| `Intersect::next_result` | `IntersectTermsEnum.next()` | was **divergent** (F-2), now identical in its error contract |
| `read_inner_node` | *(none — a shared helper, see points.rs)* | not-in-Java |

### F-1 `[MISSING → fixed]` the infallible lookups were still what every caller used

**Java.** `SegmentTermsEnum.seekExact`/`next`/`seekCeil` throw
`IOException`, and `loadBlock` throws `CorruptIndexException` from inside them.
There is no infallible spelling to reach for.

**This port, before.** c1 added `try_seek_exact`/`try_next`/`try_seek_ceil`/
`try_current` and kept the infallible spellings so no caller had to change
while three other batches were editing those crates. The ledger recorded
**136 infallible call sites against 61 `try_*` uses**, and the cost as "a
corrupt index reads as an empty one".

**What the count was hiding.** Marking the four methods `#[deprecated]` and
building `--all-targets` — which is also how the migration stays checkable —
separates production from test callers exactly:

| | before | after |
|---|---|---|
| production call sites (`src/`, outside `#[cfg(test)]`) | **15** | **0** |
| test and bench call sites | 149 | 149 |

The fifteen were ten in `lucene-search/src/lib.rs` (the term, boolean, phrase
and multi-phrase executors), two in `explain.rs`, one in `weight_count.rs`, one
in `multi_segment.rs`, and one in `blocktree.rs`'s own `Intersect`. **Every one
was already inside a `Result`-returning function**, so fourteen of them were a
`try_` prefix and a `?`.

**Fixed.** Two signatures had to widen, and both are named here rather than in
aggregate because they are the only places the migration was not local:

- `weight_count::count_term_query_shortcut` returned `Option<i64>`, where
  `None` means "no shortcut applies" and `Some(0)` means "the term is absent".
  It is now `Result<Option<i64>>`. This is exactly the shape c2 argued about
  for `advance_exact`: "the term is absent" is a legitimate answer, so
  degrading a corrupt block to it hands the caller a plausible wrong count.
  `ffi_count_term_query` maps the new error through `map_search_error`.
- `multi_segment::global_term_stats` returned `Option<CollectionStats>`, and
  `global_boolean_stats` returned `GlobalStats`. Both are now `Result<_>`.
  Dropping one segment's `docFreq` here would change *every other segment's*
  idf, since these are the reader-wide statistics every leaf scores against.

**Not migrated, and why, per site.** There is no such site. The one caller that
genuinely could not propagate — `Intersect`, an `Iterator` — was fixed by
changing what it yields, which is F-2.

**Tests.** `a_corrupt_block_errors_from_the_lookup_not_from_open` (c1's, whose
`intersect` assertion is inverted below) plus the whole suite, which exercises
the migrated paths end to end. The deprecation scan above is the standing check
that no production caller has come back.

### F-2 `[CORRECTNESS → fixed]` the intersect iterators had no error channel at all

**Java.** `IntersectTermsEnum.next()` throws `IOException`; `MultiTermQuery`'s
rewrite propagates it, so a corrupt block fails the query.

**This port, before.** `FieldTerms::{intersect, fuzzy_intersect,
regexp_intersect}` return `impl Iterator<Item = (Vec<u8>, TermStats)>`. Inside,
`Intersect::next` already called the `try_*` forms — and threw the errors away:
`Ok(SeekStatus::End) | Err(_) => { self.done = true; return None; }`. c1 tested
this deliberately (`assert_eq!(field.intersect(&pattern).count(), 0)`) and
called it "ends rather than yielding a wrong term".

**Why that is a wrong answer, not a smaller one.** A wildcard, fuzzy or regexp
clause is the *union* of its expanded terms. Ending the expansion early does
not yield fewer terms with a caveat attached — it yields a **smaller hit set**,
silently, with no signal anywhere. `expanded_terms`, `prefix_doc_ids`,
`wildcard_doc_ids`, `fuzzy_expanded_terms` and `regexp_doc_ids` all collect the
iterator and hand the result to a scorer. This is the same shape as every
tier-A finding this sweep has fixed, and it was inside the one construct that
made it invisible: an `Iterator` whose `Item` cannot fail.

**Fixed.** `Item = Result<(Vec<u8>, TermStats)>`. The body moved to
`Intersect::next_result`, which is Java's `next()` with `?` everywhere;
`Iterator::next` wraps it and sets `done` on the first error, because the
cursor's frame stack is not recoverable and Java's enumeration is likewise dead
once `loadBlock` has thrown. `check_index::compare_intersect_with_scan` reports
a failed intersect walk the way it already reported a failed linear scan.

**Test.** `a_corrupt_block_errors_from_the_lookup_not_from_open`'s intersect
assertion is now the opposite of what it was: the walk yields exactly one item
and that item is `Err(Error::Store(_))`. It fails against the unfixed code
(which yields nothing).

### F-9 `[INTENTIONAL]` the infallible spellings stay, as test-only

149 sites remain, every one under `#[cfg(test)]`, `tests/` or `benches/`. A
test that has just built its own bytes gains nothing from an error channel, and
`try_next().unwrap()` at 149 sites would be churn without a reader. Each of the
four methods now says on itself that it is a test convenience and that no
production caller uses it, and the `#[deprecated]` scan makes that claim
re-checkable in one command rather than by grep. **Deleting them is the only
step left on this item**, and its size is exactly those 149 sites.

### Verdict

Swept-clean. F-1 and F-2 fixed; F-9 intentional and bounded.

---

## `crates/lucene-codecs/src/indexed_disi.rs`

Java counterpart: `codecs/lucene90/IndexedDISI.java` (`writeBitSet`,
`addJumps`, `flushBlockJumps`, `createBlockSlice`, `createJumpTable`,
`advanceBlock`).

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `write_with_dense_rank_power` | `writeBitSet(it, out, denseRankPower)` | was **incomplete** (F-3), now identical including the `short` return |
| `add_jumps` | `addJumps` | identical (a `Vec` of pairs; Java grows one `int[]` and indexes it by block, and only ever appends at `start_block == jumps.len()`) |
| *(inline in `write_with_dense_rank_power`)* | `flushBlockJumps` | identical, including the `blockCount == 2 -> 0` exemption |
| `DisiCursor::new` | ctor + `createBlockSlice` + `createJumpTable` | was **incomplete** (F-4), now identical |
| `DisiCursor::at_start` | the second, slice-taking ctor ("in case it helps reuse") | identical in role |
| `DisiCursor::advance_block` | `advanceBlock` | was the iteration fallback only (F-4), now both arms |
| `read_i32_at` | `RandomAccessInput.readInt(pos)` | identical (little-endian) |
| everything else | — | unchanged since c2 |
| — | `advance`, `nextDoc`, `intoBitSet`, `docIDRunEnd`, `index()`, `cost()`, `asDocIndexIterator` | **still no Rust counterpart**: the `DocIdSetIterator` half, over an iterator API this port's random-access doc-values surface does not expose. Unchanged from c2's #7 table. |

### F-3 `[MISSING → fixed]` this port's writers emitted no jump table

**Java.** `writeBitSet` records `(index, offset)` for **every logical
65 536-doc block**, including the ones that hold no document (`addJumps` fills
those with the *next* real block's pair, so a jump into an empty block lands on
the block that follows it), appends the pairs after the sentinel block, and
returns the count as a `short` for the caller to store in its metadata.
`flushBlockJumps` writes nothing when `blockCount == 2` — one real block plus
the sentinel is "just wasted space".

**This port, before.** `write_with_dense_rank_power` returned only bytes, and
all three write sites recorded `jumpTableEntryCount = -1`/`0`. c2 recorded this
as the reason its #7 was not worth taking: "for our own files there is nothing
to read".

**Fixed.** `write`/`write_with_dense_rank_power` return `(Vec<u8>, i16)` —
`writeBitSet`'s own two outputs — and the four write sites
(`doc_values::write_docs_with_field`, `write_sparse_numeric_entry_body`,
`norms::write_fields`' sparse branch, `vectors`' sparse `ordToDoc`) record the
count. The subtlety worth stating: **`docsWithFieldLength` now spans the jump
table**, because that is what `createBlockSlice` subtracts the table's bytes
from (`slice(offset, length - jumpTableBytes)`). Getting that wrong in the
other direction would put the table inside the block region and the sentinel
walk would run into it.

One case is easy to assume wrong and is pinned by a test: an **empty** region
gets a one-entry table. `blockCount` is `lastBlock + 1 == 1`, which is not 2,
so the exemption does not apply.

**Tests.** `jump_table_matches_javas_byte_layout` derives all four entries of a
three-block region by hand from Java's own arithmetic — including the empty
block 1, whose entry points at block 2 — rather than round-tripping;
`a_single_block_region_writes_no_jump_table` pins both the exemption and the
empty-region case.

### F-4 `[MISSING → fixed]` `advanceBlock`'s two-blocks-ahead shortcut

**Java.** `advanceBlock` uses the table when `blockIndex >= (block >> 16) + 2`,
clamping to the last entry when the destination is past the table ("if the
`jumpTableEntryCount` is exceeded, there are no further bits; last entry is
always `NO_MORE_DOCS`"), sets `nextBlockIndex = index - 1` ("to compensate for
the always-added 1 in `readBlockHeader`") and seeks. Otherwise it walks.

**This port, before.** The walk only.

**Fixed.** `DisiCursor::new` splits its region into a block slice and a jump
table exactly as `createBlockSlice`/`createJumpTable` do, and `advance_block`
has both arms. Two deliberate deviations, both stated on the code:

- a `jump_table_entry_count` that does not fit the region degrades to the walk
  rather than erroring. Java would slice past the end and fail later; the walk
  is *always* correct, so a wrong count can only cost time, never an answer.
- the two-block threshold is Java's and is kept: one block ahead is a
  sequential header read from bytes the previous block already warmed, which
  beats a random read into a table at the far end of the region.

**Tests.** `assert_cursor_matches` — the sweep that already runs every fixture
at five rank powers over every doc id — now runs **each arm twice**, once with
the writer's own table declared and once with it declared absent, so the two
`advanceBlock` paths are proved to agree over the same bytes. Alone that is not
enough: a table that is written and never consulted passes it. So
`a_corrupted_jump_table_changes_the_answer_which_proves_it_is_read` points block
2's entry at block 1's header and asserts the answer *changes*, then asserts
the same corrupt bytes read table-less are right again — which separates "the
table is read" from "the corruption broke something else".
`a_jump_beyond_the_tables_last_block_lands_on_the_sentinel` and
`an_over_large_jump_table_count_falls_back_to_walking_the_blocks` cover the two
clamps.

### Real Lucene reads what this port writes

This is the item that changes bytes on disk, and it is the shape b4 and b11
both fell into: a writer and a reader making mirror-image mistakes round-trip
perfectly. Ruling that out needs Java, in the direction only Java can settle.

`fixtures/src/VerifySparseNumericDocValues.java` gained a **block-jump pass**
over the Rust-written `_1` segment (200 000 documents, four real blocks):
strides of two and three 65 536-doc blocks from 66 starting offsets, one fresh
`NumericDocValues` per offset, because `advanceExact` is forward-only and each
iterator is there to make exactly one long jump. Neither of the existing passes
reads a single jump-table byte — the doc-by-doc pass never advances a block at
all, and the strided passes (701/4096/20011) never advance *two*.

Both halves of every entry were negative-controlled independently, each run
end to end through `scripts/verify-write-path.sh`:

| perturbation in `add_jumps` | result |
|---|---|
| `index + 1` (the ordinal half) | `MISMATCH (block jump 131072+27916) _1 doc 158988: expected=1112913 got=1112934` — and the doc-by-doc and strided passes stay **green**, which is the point |
| `offset + 2` (the byte-offset half) | `MISMATCH (block jump 131072+27916) _1 doc 158988: expected present (1112913), got absent` |

`scripts/verify-write-path.sh`: **23/23**, confirmed by running it.

**The read side against Java-written bytes** is not directly covered, and the
reason is worth recording rather than glossing: **no committed Java-written
fixture has more than 65 536 documents** (the largest is 36 000), and Lucene
itself emits `jumpTableEntryCount = 0` for a single-block region, so there is
no Java-written table in the tree to read. What stands in for it is a chain
rather than a gap: the byte layout is pinned to Java's `flushBlockJumps` by
hand (F-3's test, not a round trip); real Lucene reads a table written to that
layout (above); and the cursor's use of it is checked against the block walk
over those same bytes at every rank power (F-4's test). A mirror-image mistake
would have to survive all three.

### F-5 `[PERF]` measured

Criterion is unusable on this machine (c24: 83/91/129 µs for identical code),
so `crates/lucene-codecs/examples/disi_jump_table.rs` is an **alternating
min-of-40 A/B**. It is the cleanest shape the method has: both arms are the
same bytes, the same cursor and one build — only the
`jump_table_entry_count` argument differs — so there is no second binary and no
rebuild between arms.

2 000 cold seeks (fresh cursor, one `advance_exact`, targets spread over the
whole doc-id range — `Lucene90DocValuesProducer`'s single-lookup shape):

| region | blocks | with the table | without | |
|---|---|---|---|---|
| 1M docs, 1 in 10 | 16 | **51.0 µs** | 157.9 µs | **3.10x** |
| 4M docs, 1 in 10 | 62 | **37.6 µs** | 458.4 µs | **12.21x** |
| 16M docs, 1 in 40 | 245 | **843.7 µs** | 2 020.0 µs | **2.39x** |

The regression check is in the same run: a **forward scan** of every present
doc never advances two blocks, so it never consults the table and measures only
`advance_block`'s new guard — 382.6 vs 381.2 µs, 1 567.1 vs 1 560.5 µs,
1 175.7 vs 1 174.3 µs, i.e. **0.4% or less in every arm, in both directions**.
c2's cursor numbers are therefore intact.

Cost on disk: 8 bytes per logical block. For the 1M-doc region above that is
136 bytes on a 130 318-byte region, **0.1%**.

The middle row is the largest ratio because 62 blocks is where the walk is
long and the region is still small enough to be cache-resident; at 245 blocks
the random read into the table starts to miss. That is the same shape Java has
and is not a defect.

### Verdict

Swept-clean. c2's #7 is closed, and its "worth revisiting only alongside a
`nextDoc`-shaped iterator API" note was wrong about *where* the table pays: it
pays on `advanceExact`, which is the only API this port exposes.

---

## `crates/lucene-codecs/src/points.rs`

Java counterparts: `index/PointValues.java`
(`estimatePointCount`, `isEstimatedPointCountGreaterThanOrEqualTo`),
`util/bkd/BKDReader.java` (`BKDPointTree.size`, `isTreeBalanced`),
`search/PointRangeQuery.java` (`relate`).

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `PointsReader::estimate_point_count` | `PointValues.estimatePointCount(visitor)` | **added** (F-6) |
| `PointsReader::estimate_point_count_bounded` | the private `estimatePointCount(visitor, tree, upperBound)` / `isEstimatedPointCountGreaterThanOrEqualTo` | **added** |
| `PointsReader::estimate_range_point_count` | `PointRangeQuery.ScorerSupplier`'s `values.estimatePointCount(visitor)` | **added** |
| `estimate_node` | the private recursive `estimatePointCount` | identical, including the `cost < upperBound` sibling short-circuit |
| `PointsField::subtree_size` | `BKDPointTree.size()` | identical, unbalanced arm only (see below) |
| `read_inner_node` | `BKDReader.readNodeData` + `pushRight`'s `rightNodePositions` | not-in-Java as a unit; extracted so `intersect_node` and `estimate_node` share one decoder |
| `PointsReader::range_visitor` | `PointRangeQuery`'s ctor check + `relate`'s visitor | refactor of what `range_query` already did |
| — | `BKDPointTree.sizeFromBalancedTree`, `balanceTreeNodePosition` | **deliberately not ported**: `isTreeBalanced()` returns `false` outright for `version >= VERSION_META_FILE`, and this reader accepts BKD version 10 alone. A balanced tree is a pre-8.6 index `open` rejects. |
| — | `PointValues.estimateDocCount` | lives in `lucene-search` (`points_query::estimate_doc_count`, c29) |

### F-6 `[MISSING → fixed]` the BKD walk that produces the estimate

**Java.** `estimatePointCount` descends the same tree `intersect` does, but a
cell entirely inside the query contributes `pointTree.size()` **without
descending**, a cell entirely outside contributes nothing, and a leaf the walk
cannot descend past contributes `(size + 1) / 2` — "assume half the points
matched". No leaf block is ever decoded, which is what makes it "many times
faster than `intersect`".

**This port, before.** c29 ported `estimateDocCount`'s arithmetic and handed
this off: "`IntersectVisitor::compare` sees cell bounds but no subtree size,
and the node-id walk is private". Its callers stood an exact match count in for
the estimate.

**Fixed.** Exactly the handoff c29 wrote, which held up.
`PointsField::subtree_size(node_id)` is `BKDPointTree.size()` — the
`leftMostLeafNode`/`rightMostLeafNode` doubling walk, the same-level versus
one-level-deeper leaf count, and the `lastLeafNodePointCount` correction for
the subtree that contains the tree's rightmost leaf — derived from the node id
and the field's four counts alone, with **no `.kdi` or `.kdd` read**. That is
what makes the estimate cheap, and it is why the whole thing is a function of
`(num_leaves, max_points_in_leaf_node, point_count)`.

`intersect_node` and `estimate_node` now share `read_inner_node`, which returns
the node's split descriptor, its restored-on-exit state and its right child's
`.kdi` position. The two walks differ only in what they do *at* a node, which
is exactly how Java's two methods differ; keeping one decoder means they cannot
drift.

Two deviations from Java, both stated on the code:

- `subtree_size` computes in `i64` where Java uses `int`, and **saturates**
  rather than wrapping on the final multiplication. `numLeaves` and
  `maxPointsInLeafNode` are both `.kdm` vints bounded only by `i32::MAX`, so a
  corrupt pair can put the product a hair past `i64`; Java wraps and hands the
  query planner a *negative* cost.
- `estimate_node` folds the two children with `saturating_add` for the same
  reason.

**Tests.** `estimate_point_count_matches_lucene` checks all fourteen boxes
against real Lucene (below). Four unit tests pin `subtree_size` against Java's
formula by hand for the three-leaf shape the fixture has (`numLeaves = 3` gives
`treeDepth = 3` and `rightMostLeafNode = 3`, so node 3 is a leaf *and* the
tree's rightmost, holding `1333 % 512 == 309`), the exact-multiple promotion
(`lastLeafNodePointCount == 0 ? maxPointsInLeafNode`), the single-leaf tree and
the saturation. `a_crossing_estimate_halves_every_leaf_and_decodes_nothing`
drives an always-`CELL_CROSSES_QUERY` visitor whose `visit`/`visit_with_value`
**panic**, so "no leaf block is decoded" is asserted rather than assumed.

### The ground truth, and why it had to come from Lucene

`fixtures/src/AppendPointEstimateManifest.java` opens the committed
`points_index` read-only and appends `point_estimate.*` keys — no generator run,
no segment id perturbed. Fourteen boxes over the three fields (`val`: one
8-byte dimension over three leaves, so the tree is unbalanced and one subtree
is a level deeper than its sibling; `multi`: two indexed dimensions over four
leaves; `shape`: four dimensions of which two are indexed).

The boxes are chosen so the **estimate and the exact match count differ**:

| case | Lucene's estimate | exact matches |
|---|---|---|
| `val.all` | 1 333 | 1 333 |
| `val.lower_half` | **768** | 674 |
| `val.narrow` | **256** | 8 |
| `val.from_middle` | **565** | 659 |
| `multi.corner` | **744** | 251 |
| `shape.quadrant` | **1 232** | 1 000 |

That is the whole point of the fixture. A "wrong but plausible" port — one that
walks the same tree and gets the per-node size arithmetic wrong — produces
numbers in the right ballpark, and a test written against the exact answer
cannot tell it from a correct one. `val.from_middle` is the sharpest single
number: 565 is `256 + 309`, a halved leaf plus the last leaf's real
`lastLeafNodePointCount`, so it pins the one correction that is easy to skip.

**Negative controls**, both run:

| perturbation | result |
|---|---|
| drop the `rightMostLeafNode` / `lastLeafNodePointCount` arm | `val.all`: 1 536 where Lucene says 1 333 |
| drop the half-leaf assumption at a non-descendable leaf | `val.lower_half`: 1 024 where Lucene says 768 |

### F-7 `[PERF]` the estimate now has the consumer c29 built it for

`points_query`'s
`a_real_point_estimate_changes_the_index_or_doc_values_plan_only_when_another_clause_leads`
had been passing exact match counts to `estimate_doc_count` because the walk
did not exist. It now calls `estimate_range_point_count`, and the narrow arm is
the one where the two part company (256 estimated, 8 matched), so the test
would not pass against the stand-in. c29's answer to its own question is
unchanged and is now established over Java's numbers rather than a proxy for
them: the estimate changes `IndexOrDocValuesQuery`'s plan exactly when another
clause leads with fewer than `cost/8` documents.

### Verdict

Swept-clean. F-6 fixed with real-Lucene ground truth and two negative controls.

---

## `crates/lucene-codecs/src/suggest.rs` and `fst.rs` — assessed, not changed

Java counterparts: `util/fst/Util.java` (`shortestPaths`, `TopNSearcher`,
`FSTPath`, `TieBreakByInputComparator`, `readCeilArc`),
`suggest/.../WFSTCompletionLookup.java`.

The brief asked whether the suggester is reachable and correct without them,
and to port them or record precisely what a caller loses. All three answers
turned out to matter.

### Is it reachable? No.

`lucene_codecs::suggest` has **no caller anywhere in this workspace** — not
`lucene-search`, not `lucene-index`, not `lucene-ffi`, not the benchmarks. It
is a public module of `lucene-codecs` and nothing else. The only reachable path
is an embedder using the crate directly.

### Is it correct? Yes.

`top_n_completions` keeps the same top N as `Util.shortestPaths` would, under
the same order: `WFSTCompletionLookup`'s comparator is ascending *cost*
(`Integer.MAX_VALUE - weight`, so descending weight) with
`TieBreakByInputComparator` breaking ties by input ascending, which is what the
bounded min-heap's `(weight, suffix)` key reproduces. `WFSTCompletionLookup`
never overrides `acceptResult`, so `rejectCount` is 0 and
`TopResults.isComplete` is always `true` — there is no partial-result signal
being dropped. b8 verified the ordering; this batch verified the `isComplete`
half, which b8 did not name.

### F-8 `[PERF]` what a caller loses, measured

Cost, and only cost. Single-arm measurement (min of 20, release build, in the
container), `term%07d` dictionaries, top-10 with `exact_first`:

| dictionary | prefix `""` | prefix `"term0"` | prefix `"term00000"` |
|---|---|---|---|
| 10 000 entries | 1 413.7 µs | 1 406.7 µs | 15.6 µs |
| 100 000 entries | 14 904.9 µs | 15 150.4 µs | 15.8 µs |
| 500 000 entries | **75 797.9 µs** | 77 077.8 µs | 15.7 µs |

Linear in the prefix subtree, exactly as the shape predicts, against Java's
`O(topN × depth × out-degree)`, which is independent of dictionary size. A
per-keystroke suggester over half a million surface forms costs **76 ms** a
lookup here. (Reproduce by building a `build_suggester_fst` over those entries
and timing `top_n_completions`; no committed harness, since the API has no
consumer to regress.)

### The recorded blocker names the wrong thing

The ledger has said, since b8, that what is missing is
`Util.shortestPaths`/`TopNSearcher`. **Porting them onto this port's FSTs would
not fix the cost and would introduce a wrong answer.**

`TopNSearcher`'s pruning is admissible only because a partial path's
accumulated output is a **lower bound** on the total output of any completion
under it. That holds only for an FST whose outputs `FSTCompiler` has *pushed*
toward the root, via `Outputs.common`/`subtract`.

This port's builder does not push. `TrieNode` stores the whole output on the
accepting node (`final_output`), every other arc carries none, and
`output_add` is byte concatenation — which is correct for
`ByteSequenceOutputs`, the type blocktree's term index actually uses, and is
why `fst.rs`'s `Outputs` doc already states that a
`build_fst_typed::<PositiveIntOutputs>` FST "is a valid `FST<BytesRef>`, not a
valid `FST<Long>`".

So on these FSTs every partial path's accumulated output is `NO_OUTPUT`, and a
best-first search with `maxQueueDepth == topN` over an all-zero-cost graph
keeps the **alphabetically first** N completions and returns those — not the
top-weighted N. Correct results today; wrong results after the "fix". That is
worth naming loudly, because the item as written reads like a contained port.

### Sizing (milestone, not a batch)

What it actually needs, in order:

1. `Outputs::common`/`subtract` on the trait, and on all three impls
   (`PositiveIntOutputs`, `ByteSequenceOutputs`, `PairOutputs`). Small.
2. Output pushing in `build_fst` — `FSTCompiler.add`'s common-prefix/subtract
   loop over `TrieNode`. This **changes the bytes the builder emits** (still a
   valid FST that Lucene reads, since a reader accumulates along the path), so
   every FST write-path fixture and `VerifyFst` moves with it.
3. `output_add` becomes `Outputs`-aware, which means threading the output type
   through `Fst`, `Arc`, `FstEnum`, `Fst::get`, `find_target_arc` **and**
   blocktree's use of all of them as `ByteSequenceOutputs`. This is the "much
   larger change touching the reader too" `fst.rs` declines by name, and it is
   the bulk of the work.
4. Only then `TopNSearcher`/`shortestPaths` (~250 lines) — and it needs
   `FSTPath` with a `TreeSet`-shaped queue supporting both `pollFirst` and
   `pollLast`, which a `BinaryHeap` is not.

Steps 2 and 3 are what make this a milestone. Declined here, as
`c38-allocation-shape` declined item 15, rather than half-built.

### F-10 `[INTENTIONAL]` `readCeilArc` has no consumer in scope

Its only callers in the whole Lucene tree are
`FSTUtil.intersectPrefixPaths` (which serves `AnalyzingSuggester`, unported)
and `FSTTermsReader` (the `memory` codec; this port uses blocktree). Neither
`WFSTCompletionLookup` nor `TopNSearcher` uses it. Porting it would add an
untested public function with no reader — recorded, not written.

### Verdict

Open by design: item 12 restated with its real blocker and a sizing. Nothing in
these two files changed.

---

## `scripts/gen-fixtures.sh` — `--append-only`

Adding cross-engine ground truth to a **committed** fixture is a recurring need
(c37 did it three times, this batch once), and the script had no mode for it:
`--only <Generator>` regenerates that generator's index with a fresh segment
id, and the alternative was invoking `javac`/`java` by hand outside the script.
That is visible in the tree — `blocktree_index/manifest.properties` carries its
`count.*` block in a position a full run does not produce, because c37 appended
it by hand.

`--append-only` runs the `Append*Manifest` programs and no generator. It
composes with `--check` unchanged (which compares each manifest's *key set*, so
it already covers appended keys), and it refuses to combine with
`--only`/`--all`/`--check`. Documented in the script's own header.

---

## Gate

`cargo fmt --all --check`, `cargo clippy --workspace --all-targets` on x86_64
and aarch64 with `-D warnings`, `cargo check` on the out-of-workspace benchmark
crate, `scripts/check-arith-allows.py`, `scripts/check-parity.py`,
`scripts/check-java-refs.py`, and `cargo llvm-cov --workspace
--fail-under-lines 95`.

`scripts/verify-write-path.sh`: **23/23**, run and confirmed, including the new
block-jump pass.

`scripts/docker-test.sh gate` -> **`gate: ok`**; workspace **98.12% lines /
97.55% regions**, and every file this batch touched is above invariant #8's
per-file bar: `points.rs` 99.09%, `indexed_disi.rs` 98.64%, `norms.rs` 98.67%,
`doc_values.rs` 97.98%, `blocktree.rs` 96.68%, `vectors.rs` 96.86%,
`weight_count.rs` 99.22%, `check_index.rs` 98.28%, `points_query.rs` 98.37%,
`field_norms.rs` 98.36%, `multi_segment.rs` 97.22%, `lucene-search/src/lib.rs`
96.79%, `explain.rs` 95.50%, `lucene-ffi/src/query.rs` 97.81%.

## What the Tier-2 review changed

Three fixes and two new ledger items.

- **`DisiCursor::new`'s doc comment was left stale and understating the new
  contract** — it still said a trailing jump table "which this port's writers
  never emit" was harmless. Rewritten to state the actual contract: `data` is
  the whole `docsWithFieldOffset .. + docsWithFieldLength` region, table
  included, and the one misuse nothing can catch is passing a *correct* count
  over a region whose table has already been stripped (the count still fits,
  so the split silently takes `count * 8` bytes off the last block).
- **`docs/parity.md` claimed `estimateDocCount` was pinned by the new fixture,
  and no test read those keys.** `AppendPointEstimateManifest` writes
  `.docs`/`.size`/`.doc_count`; nothing read them. Now asserted — and the claim
  narrowed to what it actually pins, because all three fixture fields are
  single-valued, so every box takes branch 1 or 2 and `.docs == .points`
  throughout. **Branch 3**, the multi-valued urn approximation and the one place
  float order-of-operations could diverge from Java, has no cross-engine ground
  truth; that is now said in both the parity row and the test rather than
  implied away. This is a general shape and is now item 27(e): a key a
  generator writes and no test reads is ground truth that does not exist.
- `AppendPointEstimateManifest`'s class doc said `val` is "1 314 points", where
  the manifest it writes says 1 333 — and the paragraph exists precisely so a
  reader can re-derive `lastLeafNodePointCount = 1333 % 512 = 309` by hand.
- Raised as **23b** (the BKD walk's four heap allocations per inner node, where
  Java preallocates per-level stacks — pre-existing, but this batch promoted it
  to a shared path and hung the cheap-by-design estimate off it) and **23c**
  (no Java-written jump table exists in the fixture tree to read, because the
  largest Java-written index is 36 000 documents).
- Also raised as item 27(d): the `#[deprecated]` audit that proves no
  production caller uses the infallible spellings is documented on the methods
  but nothing runs it, so the migration can silently regress.

Not committed, per the batch instruction.
