# c21-hnsw-seeded

Follow-up batch closing the three items `c16-knn-query` left blocked on files
it did not own:

- its **finding 6** — the optimistic re-entry pass is not *seeded*
  (`SeededHnswGraphSearcher` lives in `crates/lucene-codecs/src/hnsw.rs`);
- its **finding 7** — `hnsw_vectors::search` has no `acceptOrds`/
  `filteredDocCount`, so `vector_query::hnsw_search` duplicated ~40 lines of
  it;
- its **carry-over** — filtered KNN is ported in `lucene-search` but is not
  reachable over the C ABI.

Files swept: `crates/lucene-codecs/src/{hnsw,hnsw_vectors}.rs`,
`crates/lucene-search/src/vector_query.rs`,
`crates/lucene-ffi/src/vectors.rs`. New:
`fixtures/src/{GenVectorsSeeded,GenVectorsFiltered}.java`,
`fixtures/data/{vectors_seeded_index,vectors_filter_index}/`. Also touched:
`crates/lucene-search/tests/vector_query_fixtures.rs`,
`crates/lucene-search/benches/knn_multi_segment.rs`,
`crates/lucene-codecs/tests/vectors_fixtures.rs`,
`crates/lucene-codecs/benches/hot_paths.rs`, `docs/parity.md`,
`fixtures/README.md`.

Java counterparts, all read from the **pinned 10.5.0 tag**
(`/home/tuong/work/lucene-10.5.0`):

- `util/hnsw/{SeededHnswGraphSearcher, AbstractHnswGraphSearcher,
  HnswGraphSearcher}.java`
- `search/knn/KnnSearchStrategy.java` (`Hnsw`, `Seeded`)
- `search/{AbstractKnnVectorQuery, SeededKnnVectorQuery}.java`
  (`ReentrantKnnCollectorManager`, `getLeafResults`, `MappedDISI`,
  `TopDocsDISI`)
- `codecs/lucene99/Lucene99HnswVectorsReader.java` (its `search` dispatch)

`lucene-ffi/src/vectors.rs` has **no** Java counterpart — it is the C-ABI
boundary; per the protocol's rule 1 none is invented for it.

Totals: **19 findings** — 2 CORRECTNESS (both fixed), 7 MISSING (all fixed),
1 PERF (measured), 9 INTENTIONAL. Plus a Tier-2 review at the end: three
gating findings and four advisories, all seven acted on.

---

## `crates/lucene-codecs/src/hnsw.rs`

Java counterparts: `util/hnsw/{SeededHnswGraphSearcher,
AbstractHnswGraphSearcher, HnswGraphSearcher}.java`,
`search/knn/KnnSearchStrategy.java`.

### Method correspondence (only what this batch changed)

| Rust | Java | Verdict |
|---|---|---|
| `HnswGraphSearcher::search` | `AbstractHnswGraphSearcher.search` | unchanged, identical |
| `HnswGraphSearcher::search_seeded` (**new**) | `SeededHnswGraphSearcher` entire — `fromEntryPoints`' validation, `findBestEntryPoint`'s constant override, and the inherited `search`/`searchLevel` | ported (finding 1) |
| — | `SeededHnswGraphSearcher.searchLevel` | not-in-Rust *by construction*: it delegates verbatim to the wrapped searcher's, which here is the same method (finding 10) |
| — | `KnnSearchStrategy.{Hnsw,Seeded,Patience}` | not ported as types (finding 11) |

### 1. [MISSING → fixed] `SeededHnswGraphSearcher` was not ported

**Java.** `HnswGraphSearcher.search(scorer, collector, graph, acceptOrds,
filteredDocCount)` builds the inner searcher, then — *after* it — checks
whether the collector's strategy is a `KnnSearchStrategy.Seeded` with
`numberOfEntryPoints() > 0` and, if so, wraps it in
`SeededHnswGraphSearcher.fromEntryPoints(inner, numEps, eps, graph.size())`.
That subclass overrides `findBestEntryPoint` to return the seed ordinals and
delegates `searchLevel` unchanged, so the whole class is "skip the upper-level
descent and start level 0's beam here".

**Us.** No seeded searcher at all; the re-entry pass started from the graph's
own entry node.

**Consequence.** Only *where* the walk starts, never the collector, the accept
set or the merge — so it can only change which approximate answer comes back,
and it changes the cost (finding 8). Note that it is not free of correctness
risk in the other direction either: the seeded walk is what Java's phase 2
*is*, so a port without it was running a different search from the one the
fixture's ground truth was recorded from — it simply happened to converge to
the same answer (finding 9 says exactly how far that "happened to" was
checked).

**Resolution — fixed.** `HnswGraphSearcher::search_seeded` is the port.
`fromEntryPoints`' two rejections come across as errors rather than as Java's
`IllegalArgumentException`/`assert` pair: an empty entry-point set ("The
number of entry points must be > 0"), and an ordinal outside `0..graph.size()`
— Java only `assert`s that one, i.e. does not check it in production, and here
it would index the `visited` `FixedBitSet` out of bounds and panic.

*Tests*: `seeded_entry_points_are_validated_before_they_reach_the_bitset`
(negative, `== size`, `i32::MAX`, and the empty list),
`a_seeded_search_starts_where_it_is_told` (seeding a walk with its own answer
is a fixpoint; a single seed under a visit cap of one returns exactly that
node, which is what says the seeds are the *starting set* and not a hint), and
`a_seeded_search_still_honours_the_accept_set` (Java's `scoreEntryPoints`,
shared by both searchers, gates collection of the seeds themselves).

### 10. [INTENTIONAL] A method, not a class

Java needs a subclass because `findBestEntryPoint` is a virtual it wants to
override with a constant; its `searchLevel` then delegates *verbatim* to the
wrapped searcher. In Rust that shape would be a second type holding a
`&mut HnswGraphSearcher` plus a slice, whose only method forwards — and it
would either duplicate or borrow-split the scratch state
(`prepare_scratch_state`'s bitset and bulk buffers) that the two searches
share. A second entry point on the one searcher is the same behaviour with
neither cost. Stated on the function, with the Java class named.

### 11. [INTENTIONAL] `KnnSearchStrategy` is not a type here

Java carries the strategy on the *collector* and reads it back inside
`HnswGraphSearcher.search` through two `instanceof` chains. This port passes
the two things those chains extract — the seed ordinals, and (for
`FilteredHnswGraphSearcher`) the filtered ratio — as arguments, so there is
nothing to downcast. `Hnsw.filteredSearchThreshold` is not represented at all
because 10.5.0's `DEFAULT_FILTERED_SEARCH_THRESHOLD` is **`0`** and
`useFilteredSearch(ratio)` is `ratio * 100 < 0`, i.e. false for every ratio:
no query reachable from `KnnFloatVectorQuery` selects `FilteredHnswGraphSearcher`.
(`main` raised the threshold to 60 — a post-10.5.0 change, per
`c18-version-audit`. The module doc now says which version it is describing.)
`Patience` belongs to `PatienceKnnVectorQuery`, which is not ported.

### 12. [INTENTIONAL, doc] The module's "not ported" list was stale

`hnsw.rs`'s header said `FilteredHnswGraphSearcher` and
`SeededHnswGraphSearcher` are "strategy variants selected by a
`KnnSearchStrategy` this port has no query surface for yet". There has been a
query surface since `c16-knn-query` — that is the whole point of
`vector_query.rs` — so the sentence justified a gap with a reason that had
stopped being true. Rewritten: `SeededHnswGraphSearcher` is now ported and
named, and `FilteredHnswGraphSearcher` is recorded as unreachable *for the
pinned version's own reason* (finding 11) rather than for a missing caller.

### Verdict

Swept clean for the seeded search. Open, unchanged from c5/c10:
`FilteredHnswGraphSearcher` (unreachable at 10.5.0's threshold),
`ConcurrentHnswMerger`, `SparseFixedBitSet`.

---

## `crates/lucene-codecs/src/hnsw_vectors.rs`

Java counterpart: `codecs/lucene99/Lucene99HnswVectorsReader.search(FieldEntry,
KnnCollector, AcceptDocs, IOSupplier<RandomVectorScorer>)`.

### Method correspondence (only what this batch changed)

| Rust | Java | Verdict |
|---|---|---|
| `SearchOptions` (**new**) | `search`'s `AcceptDocs` parameter, its derived `filteredDocCount`, and the `KnnSearchStrategy.Seeded` it reads off the collector | ported as data (finding 14) |
| `search` | the same method | now identical: `acceptOrds`, `filteredDocCount`, the seeded dispatch, and the early-terminated flag (findings 2, 3, 4) |

### 2. [MISSING → fixed] `acceptOrds` and `filteredDocCount`, and the duplicate they caused

**Java.** `Lucene99HnswVectorsReader.search` takes `AcceptDocs`, computes
`filteredDocCount = min(acceptDocs.cost(), graphSize)`, derives
`acceptedOrds = scorer.getAcceptOrds(accepted)`, and hands both to
`HnswGraphSearcher.search`. `filteredDocCount` is what decides the graph walk
against the exhaustive scan (`unfilteredVisit >= filteredDocCount`), and
`acceptedOrds` gates collection in *both* branches.

**Us.** `hnsw_vectors::search` had neither and hardcoded
`filteredDocCount = graphSize`. c16 needed both, so it wrote
`vector_query::hnsw_search` — a ~40-line copy of the same dispatch — and
recorded the duplication with the owner named.

**Consequence.** Two copies of one decode/dispatch walk, asserted equivalent
only on the half they shared. That is precisely the shape that bit `c15`: two
copies of the same walk drifted apart exactly on corrupt input, where no test
compared them.

**Resolution — fixed.** `hnsw_vectors::search` gained a `SearchOptions`
argument carrying `accept_ords`, `filtered_doc_count` and `seed_ords`, and
`vector_query::hnsw_search` **is deleted**; `leaf_results` calls the codec
directly. c16's `the_unfiltered_dispatch_matches_the_codec_entry_point` went
with it — it existed only to pin a copy against its original, and there is now
one function.

Two details of Java's that moved down with the code and are worth naming,
because both are easy to lose in a re-type:

- the exhaustive branch's early-termination check sits **inside** the accept
  test (`if (acceptedOrds == null || acceptedOrds.get(i)) { if
  (knnCollector.earlyTerminated()) break; ... }`), so a run of rejected
  ordinals can never end the scan;
- `filteredDocCount` is `min(maxDoc, graphSize)` even on a **deleted**
  segment. `BitsAcceptDocs` uses `bitSet.cardinality()` only `if (bits
  instanceof BitSet)`, and `Lucene90LiveDocsFormat.readLiveDocs` returns a
  `DenseLiveDocs`/`SparseLiveDocs`, which implement `LiveDocs extends Bits`
  and are not `BitSet`s — so deletions alone never drop a segment to the
  exhaustive scan. c16 established this against a Tier-2 misreading; the
  reasoning now lives on `SearchOptions::filtered_doc_count`, where the
  parameter is.

*Tests*: `a_selective_filtered_doc_count_drops_the_search_to_an_accept_aware_scan`
(both branches, with the accept set gating each) and the whole filtered
differential suite in `lucene-search`, which now runs through this function.

### 3. [MISSING → fixed] The early-terminated flag was not returned

`AbstractKnnVectorQuery.getLeafResults` reads
`results.totalHits.relation() == EQUAL_TO` to decide whether a filtered
approximate search must fall back to `exactSearch`. `hnsw_vectors::search`
returned only the hits, so the flag existed only in c16's duplicate. It is now
part of the return type (`(Vec<(i32, f32)>, bool)`), which is what let the
duplicate go. *Test*: `the_exhaustive_branch_honours_the_visit_limit` now
asserts the flag as well as the truncation.

### 4. [CORRECTNESS → fixed] A caller-supplied accept set could panic

Pushing `acceptOrds` down made a new input reachable from a **public codec
API**: a `FixedBitSet` shorter than the field's vector count. `FixedBitSet::get`
past the end is an out-of-bounds index — a panic, and in `lucene-ffi` a panic
`catch_unwind` would turn into a status but which is still not an answer.
Java has nothing to check here: its `Bits` is unbounded by construction
(`getAcceptOrds` wraps doc-space bits behind `ordToDoc`) and its
`filteredDocCount` bound is an `assert`, i.e. absent in production.

Both are now validated into `Error::InvalidGraphParameter` before anything is
allocated: an accept set shorter than `maxOrd`, and a negative
`filtered_doc_count` ("it is a cardinality"). The check is before the
`num_vectors == 0 || k == 0` early return, so a degenerate field does not
smuggle a bad argument past it. *Test*:
`a_short_accept_set_or_a_negative_filtered_doc_count_is_an_error_not_a_panic`,
which also pins that a set of *exactly* the field's size — what every caller in
this workspace builds — is accepted.

### 14. [INTENTIONAL] One options struct, not three more positional arguments

All three new parameters are optional, all three mean "narrow this walk", and
`SearchOptions::default()` is exactly Java's plain `KnnFloatVectorQuery`. Three
more positional `Option`s at a call site would read as three unrelated
booleans-with-payloads; `Copy`, `Default` and struct-update syntax make the
narrowed cases read as the narrowing they are. It also keeps
`filtered_doc_count` as `Option<i32>` rather than a magic number: `None` is
"every vector passes", which is the unfiltered case, and is provably the same
value Java computes there (`min(maxDoc, graphSize) == graphSize`, since
`graphSize <= maxDoc` always).

### 13. [INTENTIONAL, doc] The scope note was stale

`hnsw_vectors.rs`'s header still said "no sorted-index (`writeSortingField`)
or merge (`mergeOneField`) paths". `mergeOneField` has been ported since
`c10-vectors-wiring`. Corrected, and the note now also states which of Java's
`search` parameters this module carries.

### Verdict

Swept clean. `search` is now `Lucene99HnswVectorsReader.search` in full, and
is the only copy of that dispatch in the workspace.

---

## `crates/lucene-search/src/vector_query.rs`

Java counterparts: `search/AbstractKnnVectorQuery.java`
(`ReentrantKnnCollectorManager`, `getLeafResults`, `rewrite`'s re-entry loop),
`search/SeededKnnVectorQuery.java` (`MappedDISI`, `TopDocsDISI`).

### Method correspondence (only what this batch changed)

| Rust | Java | Verdict |
|---|---|---|
| `LeafHits` (**new**) | the `TopDocs` `ReentrantKnnCollectorManager` reads back, plus what `MappedDISI`/`TopDocsDISI` derive from it | ported, without the re-derivation (finding 6) |
| `leaf_results` | `getLeafResults` | now calls `hnsw_vectors::search`; also returns the leaf's seed ordinals |
| `seed_slice` (**new**) | `ReentrantKnnCollectorManager.newCollector`'s `seedTopDocs.totalHits.value() == 0` fall-back + `HnswGraphSearcher.search`'s `numberOfEntryPoints() > 0` gate | ported |
| `reentry_plan` | `ReentrantKnnCollectorManager`'s delegate + `getLeafResults`' own `perLeafTopK` | **was divergent** (finding 5) |
| ~~`hnsw_search`~~ | — | **deleted** (finding 2) |

### 5. [CORRECTNESS → fixed] Phase 2's cost thresholds were `k`, not the pro-rata `perLeafTopK`

**This is a defect this batch found in c16's own code, not a carry-over.**

**Java.** `getLeafResults` recomputes `perLeafTopK` from `ctx.parent` on
**every** call — phase 1 and phase 2 alike:

```java
if (ctx.parent != null) {
  float leafProportion = ctx.reader().maxDoc() / (float) ctx.parent.reader().maxDoc();
  perLeafTopK = perLeafTopKCalculation(k, leafProportion);
} else { perLeafTopK = k; }
```

The full `k` reaches phase 2 through the *collector manager*
(`ReentrantKnnCollectorManager` delegates to a fresh
`TopKnnCollectorManager(k, searcher)`), which `getLeafResults` never inspects.
So in phase 2 the **collector** is `k` and the **two cost thresholds** are
still pro-rata.

**Us.** `reentry_plan` set `per_leaf_top_k: phase1.k`, raising both thresholds
with the collector.

**Consequence**, on the filtered path only (`perLeafTopK` is used nowhere
else): Java's `cost <= perLeafTopK` short circuit and its
`scoreDocs.length >= perLeafTopK` fall-back both moved. Concretely, on the new
seeded fixture's clustered leaf (`perLeafTopK = 93`, `k = 100`) a filter
accepting 96 documents makes Java walk the graph and made this port take the
exact scan, and an approximate result of 95 hits made Java keep it and made
this port throw it away for an exact one. Both are *different searches*; the
answers coincide in the common case, which is exactly why it survived c16's
suite.

**Resolution — fixed**: `reentry_plan` now changes `collector_k` alone, with
Java's own reason stated on it. *Test*:
`reentry_restores_the_full_k_on_the_collector_and_nothing_else` pins that
`per_leaf_top_k` stays pro-rata across the pass boundary.

**Honest limit on the evidence.** The differential fixtures *execute* the
changed line — instrumenting `knn_multi_segment` shows the filtered
multi-segment tests re-entering leaf 3 with a threshold of 5 against `k = 10`
on all 15 filtered queries — but that leaf is graphless, so both sides of the
moved branch are exact scans and produce the same list. Making the two
branches *disagree* needs a re-entered leaf that has a graph, a filter cost in
`(perLeafTopK, k]`, and a walk that is not itself degenerate — three
conditions that Java's own design pushes apart, since a small filter cost
takes `unfilteredVisit >= filteredDocCount` to the exact scan anyway. So the
fix is pinned by the unit test and by the Java quotation, not by a divergent
fixture, and this paragraph exists so that is not overclaimed.

### 6. [MISSING → fixed] The re-entry pass is now seeded

`ReentrantKnnCollectorManager.newCollector` wraps phase 1's per-leaf `TopDocs`
in a `KnnSearchStrategy.Seeded` whose entry points are that leaf's hits mapped
into ordinal space: `TopDocsDISI` subtracts `ctx.docBase` and sorts the local
doc ids ascending, and `MappedDISI` runs each through the vector values'
`DocIndexIterator.advance(doc)`/`index()` to get its ordinal.

This port keeps the ordinals from the walk that already produced them
(`LeafHits::ords`, sorted ascending) instead of re-deriving them from doc ids.
The two are the same list: `ordToDoc` is monotonic, so "the hits' doc ids
ascending, mapped to ordinals" and "the hits' ordinals ascending" are equal by
construction, and every seed doc has a vector (it came out of a vector search),
so no `advance` can land on a different document. Java pays an `advance` per
seed over the whole `IndexedDISI` because its `TopDocs` carry only doc ids;
this costs one `i32` per hit, at most `perLeafTopK` of them.

`seed_slice` ports both of Java's "not seeded after all" gates — the
`seedTopDocs.totalHits.value() == 0` fall-back to the unseeded collector, and
`HnswGraphSearcher.search`'s `numberOfEntryPoints() > 0` — as `None`, which is
distinct from an *empty* entry-point set (which `search_seeded` rejects, as
Java's `fromEntryPoints` does). *Test*:
`an_empty_phase_one_hit_list_means_not_seeded`.

Seeding does not reach the exact-search branch, exactly as in Java: the
strategy is read inside `HnswGraphSearcher.search`, and `exactSearch` never
calls it. Stated at the branch.

### 7. [MISSING → fixed] The fixture could not see seeding at all

`fixtures/data/vectors_multi_index` reaches the re-entry pass — that is what
c16 built it for — but the **only** leaf it ever re-enters is the 40-document
one, which is below `shouldCreateGraph` and therefore takes the exhaustive
branch, where Java ignores the search strategy. Instrumenting `run_leaves`
confirms it: `reenter` is `[3]` on every query that re-enters at all — all 20
dense queries at `k = 100` (with `perLeafTopK` 130/94/92/16) and all 15
filtered ones at `k = 10` (threshold 5). So none of c16's 80 recorded queries
says anything about seeding, and none of them would have failed on a port that
skipped it — which the A/B in finding 9 confirms directly.

Two constraints have to hold at once for a *graph-bearing* leaf to be
re-entered, and 4000 documents cannot satisfy both:

1. **re-enterable**: `perLeaf.scoreDocs[len-1].score >= minTopKScore` needs a
   score tie unless `perLeafTopK < k`, i.e.
   `k*p + 16*sqrt(k*p*(1-p)) < k` — `p < 0.039` at `k = 10`, so under ~156
   documents of a 4000-document index;
2. **has a graph**: `shouldCreateGraph` needs `n > ln(n) * 100`, about 660
   vectors.

They are compatible at a larger `k`. **New fixture**
`fixtures/data/vectors_seeded_index` (`GenVectorsSeeded.java`):
1400/700/700/40 documents queried at `k = 100`, where the four `perLeafTopK`
values are 129, **93**, 93 and 20, and the second 700-document segment holds a
tight cluster every query target sits next to. Instrumentation confirms the
intended shape: all 20 queries re-enter leaf 1, with 93 seed ordinals.

*Tests*: `the_seeded_reentry_pass_reproduces_lucene_over_a_graph_bearing_leaf`
(20 queries at `k = 100`, doc for doc and score for score),
`the_seeded_fixture_also_reproduces_lucene_without_any_reentry` (the same
index at `k = 10`, the no-re-entry control, so a failure in one and not the
other localises the fault), and
`the_seeded_fixture_still_reaches_the_reentry_pass_on_a_leaf_with_a_graph`,
which asserts the three preconditions directly — the recorded `perLeafTopK`
values are reproduced by `per_leaf_top_k`, the clustered leaf's is below `k`,
that leaf carries a graph, and Lucene's own recorded answer takes **more**
than `perLeafTopK` hits from it, which one pass cannot produce. A fixture that
quietly stops reaching its branch proves nothing, so that third assertion is
what fails if it ever does.

### 9. [PERF, measured] What seeding is worth, and what it is not

**It does not change the answer on either fixture.** Disabling the seeding and
re-running the whole suite: all 21 differential tests still pass, including
the new `k = 100` one where seeding demonstrably runs on a graph-bearing leaf.
That is the honest headline — seeding is a cost optimisation whose approximate
answer coincides with the unseeded walk's here — and it is why the
discriminating evidence for finding 1 is structural
(`a_seeded_walk_restarts_from_its_entry_points_and_skips_the_descent`,
`a_seeded_search_starts_where_it_is_told`) rather than differential. It is
*not* a reason to skip the port: agreement with Lucene on two fixtures is not
agreement in general, and Java's phase 2 is the seeded walk.

**Vector comparisons** (instrumented scorer, counting `score`/`bulk_score`
calls, mean over 20 queries):

| | unseeded | seeded | saving |
|---|---|---|---|
| `vectors_index`, 4000 vectors, `k = 10`, seeds = the unseeded top 10 | 292 | **216** | 26.0% |
| `vectors_seeded_index` leaf 1, 700 vectors, phase 2 at `k = 100`, seeds = phase 1's 93 hits | 553 | **530** | 4.1% |

The gap between the two rows is the whole story: what seeding removes is
`findBestEntryPoint`'s hill climb over every level above 0, which is
`O(log n)` work against a level-0 beam of `O(collector_k)`. It is a quarter of
a narrow `k = 10` search and a rounding error next to a 100-wide one.

**Query level**, `cargo bench -p lucene-search --bench knn_multi_segment`
(20 queries per iteration, release), with the seeding switched off and on:

| | per query |
|---|---|
| `vectors_seeded_index`, four segments, `k = 100`, unseeded | 130.5 us |
| `vectors_seeded_index`, four segments, `k = 100`, **seeded** | 130.3 us |
| `vectors_seeded_index`, four segments, `k = 10` (no re-entry at all) | 26.9 us |

`change: [-2.23% +0.18% +2.39%] (p = 0.88 > 0.05)` — **no measurable
difference**. The re-entered leaf is one of four and phase 2 is one of two
passes, so a 4% saving inside it is well under this bench's noise.

**So seeding does not claw back c16's 7x fan-out gap**, and the note in c16's
finding 13 that it might should be read as answered: the gap is the pro-rata
collectors (83 slots against one leaf's 10) and Java pays it too. Seeding
saves the descent, which is the cheap part.

### 8. [INTENTIONAL] Phase 1 and phase 2 keep their own hit lists

`per_leaf_ords` is captured from phase 1 and *not* refreshed from phase 2's
results. Java is the same — `ReentrantKnnCollectorManager` reads
`perLeafResults`, the phase-1 map, and there is no phase 3 — but it is worth
saying, because `per_leaf` *is* overwritten with phase 2's hits two lines
away, and a reader could reasonably expect the two to move together.

### Verdict

Swept clean. `hnsw_search` is gone, the re-entry pass is seeded, and the
phase-2 threshold defect is fixed. Nothing open.

---

## `crates/lucene-ffi/src/vectors.rs`

No Java counterpart — the C-ABI boundary. The wrapped concepts are
`lucene_search::vector_query`'s filtered entry points and
`crate::query::read_boolean_query`'s clause-array format.

### Method correspondence

| Rust | Was | Now |
|---|---|---|
| `ffi_knn_float_vector_search` / `ffi_knn_byte_vector_search` | unfiltered only | unchanged, and still the way to ask for Java's *unfiltered* path |
| `ffi_knn_float_vector_search_filtered` / `ffi_knn_byte_vector_search_filtered` (**new**) | — | `new KnnFloatVectorQuery(field, target, k, filter)` / `KnnByteVectorQuery`'s |
| `resolve_filter` (**new**) | — | clause array → `BooleanQuery` → doc set → `vector_query::accept_bitset` |
| `vectors_max_doc` (**new**) | — | the handle's `maxDoc`, read and released before the segments registry is touched |
| `vectors_input` | `filter: None`, with the gap recorded | still `filter: None` — it returns the *unfiltered* input, and the filtered entry points set the field afterwards on the value they get back (the accept bitset has to outlive `vectors_input`'s frame). Only the stale "no filtered KNN over this ABI yet" comment is gone |

### 15. [MISSING → fixed] Filtered KNN is reachable over the C ABI

c16 ported filtered KNN in full and left it unexposed, because "the filter
query would need its own clause array at the boundary".

**No second encoding was invented.** The two new entry points take the same
eight `clause_*` arrays plus `clause_count` and `minimum_should_match` that
`crate::query::read_boolean_query` already decodes for every
`ffi_search_boolean_query*` entry point and `ffi_explain_boolean_query` — which
is exactly what `c13-ffi-surface` rebuilt that format for ("an `Occur` and a
clause kind are *values*, so the next one of either costs no ABI change").
This is the first clause-shaped addition since, and it cost none. The same
`check_clause_count` cap and the same `MAX_CLAUSE_DEPTH` nesting bound apply,
because it is the same decoder.

*Tests*: `filtered_knn_over_the_c_abi_reproduces_lucene` — 80 queries (two
encodings x 20 targets x both filters) through the exported symbols, doc for
doc and score for score against real Lucene;
`every_filtered_hit_comes_from_the_terms_own_postings`;
`an_empty_clause_list_is_a_filter_that_accepts_nothing`;
`the_unfiltered_entry_point_is_still_javas_unfiltered_path`.

### 16. [MISSING → fixed] Single-segment filtered KNN had no Lucene ground truth

The C ABI opens **one** segment's vector files per handle, so its ground truth
has to be `IndexSearcher.search(query, k)` over a *one-leaf* index — where
`leafProportion == 1` makes `perLeafTopK == k` and `perLeafResults.size() > 1`
is false, so there is no pro-rata sizing and no re-entry. Running the same
query against one leaf of `vectors_multi_index` is a **different search** (that
leaf's collector is pro-rata sized), so its recorded results cannot stand in;
and `vectors_index`, the existing single-segment fixture, has no term
dictionary at all, so no filter clause can be resolved against it.

**New fixture** `fixtures/data/vectors_filter_index` (`GenVectorsFiltered.java`):
one 1200-document segment with a FLOAT32 and a BYTE vector field *and*
`bucket`/`group` `StringField`s. `bucket:b0` accepts 6 documents against
`k = 10`, so every query takes Java's `cost <= perLeafTopK` short circuit into
`exactSearch`; `group:g0` accepts a quarter of the index, so the graph is
walked with `acceptOrds` and `visitedLimit = cost + 1`. Both branches, both
encodings. It also closes the same gap one level down: the new
`single_segment_filtered_knn_reproduces_lucene` and
`a_single_segment_filtered_hit_is_always_in_the_filter` are the first
Lucene-differential coverage `lucene-search`'s *single*-segment filtered path
has had.

### 17. [INTENTIONAL] A second handle, cross-checked on `maxDoc`

A vector field needs no term dictionary, which is why `ffi_open_vectors` opens
none and why a vectors-only segment is openable at all (c13's reason, and
`fixtures/data/vectors_index` is such a segment). A filter clause *is* a
`TermQuery` and needs `.tim`/`.tip`/`.tmd`/`.doc`. Widening the vectors handle
would undo the thing it exists for, so the filtered entry points take the same
segment's `SegmentHandle` as a second argument. Java has no equivalent choice —
its `LeafReader` is one object with both.

The two handles must describe the same segment, and `maxDoc` is the check:
a filter resolved against a different segment yields doc ids that mean
something else, and nothing downstream can detect it — the accept set is simply
wrong, silently. It is the same hazard `accept_bitset` and `KnnSegment::doc_base`
already document.

Worth recording as a fact discovered while testing it: **`ffi_open_segment`
already refuses a wrong `maxDoc`**, cross-checking it against the term
dictionary's own metadata, so a mis-described *segment* handle cannot be
constructed at all. That is why the mismatch test has to stage the error on the
vectors side, which has no such cross-check available — and why this check has
to exist there.

### 18. [INTENTIONAL] Two registries, never held together

`resolve_filter` needs the segments registry and the search needs the vectors
registry. Rather than hold both, `vectors_max_doc` takes the vectors guard,
reads one `i32` and drops it; `resolve_filter` then takes the segments guard
and drops it; only then is the vectors guard retaken for the search. Two
registries held together in one order here and the other order anywhere else
is a lock cycle, and this is the only entry point in the crate that needs two.
Stated on the function.

### 19. [INTENTIONAL] Every new numeric and pointer is validated

Per the `ffi-safety` skill, and asserted end to end rather than by inspection
(`the_filtered_entry_points_reject_every_caller_mistake_by_status`): both new
exports are wrapped in `guard` (`catch_unwind`); a null out pointer and a null
query pointer with a non-zero length are `NullPointer`; an unknown vectors
handle and an unknown segment handle are `InvalidHandle`; a mismatched pair is
`InvalidArgument`; `clause_count` is bounded by `check_clause_count` **before**
any array is dereferenced (`usize::MAX` is rejected, not indexed); an unknown
`Occur` tag and a negative `minimumNumberShouldMatch` are `InvalidArgument`
from the shared decoder; and a query-level mistake (`k = 0`, an unknown field)
stays `InvalidArgument` rather than becoming `Decode` — c13's finding 23,
which a new entry point could easily have regressed. Every one carries a
retrievable message, read back through the real `ffi_get_last_error_message`.

### Verdict

Swept clean. `vectors.rs` remains a thin wrapper: the two new exports decode
arguments, resolve a filter through the existing boolean decoder, and call
`lucene-search`.

---

## Evidence

**Against real Lucene, not against our own reader.** Four fixtures, all driven
by a real `IndexWriter` and queried through a real `IndexSearcher`:

- `fixtures/data/vectors_index` (c5's) and `fixtures/data/vectors_multi_index`
  (c16's) — **unchanged and not regenerated**; c16's 16 differential tests and
  the 25 pre-existing `lucene-ffi` vector tests (c13's, as amended by c16)
  all pass **unmodified**, which is what says the seeding, the codec-API
  change and the deleted duplicate preserved behaviour.
- `fixtures/data/vectors_seeded_index` (**new**) — 1400/700/700/40 documents,
  `k = 100`, the seeded re-entry pass on a graph-bearing leaf. 40 queries
  (20 at `k = 100`, 20 at the `k = 10` control), doc for doc and score for
  score, plus the fixture-shape assertions.
- `fixtures/data/vectors_filter_index` (**new**) — one 1200-document segment
  with postings. 120 queries at the `lucene-search` level (two encodings x 20
  targets x unfiltered/selective/permissive) and 80 through the exported C
  symbols, all doc for doc and score for score.

**No assertion is a recall threshold**, per c5's Tier-2 lesson. Every
differential test compares against Lucene's actual result list; the three that
are not differential comparisons (`a_seeded_walk_restarts_from_its_entry_points_and_skips_the_descent`,
`a_seeded_search_starts_where_it_is_told`, `a_seeded_search_still_honours_the_accept_set`)
are structural equalities and comparison counts, not metrics.

## Coverage

`cargo llvm-cov -p lucene-codecs -p lucene-search -p lucene-ffi --summary-only`,
lines:

| file | before | after |
|---|---|---|
| `lucene-codecs/src/hnsw.rs` | 96.75% | **96.96%** |
| `lucene-codecs/src/hnsw_vectors.rs` | 95.60% | **95.79%** |
| `lucene-search/src/vector_query.rs` | 97.65% | **97.63%** |
| `lucene-ffi/src/vectors.rs` | 98.51% | **98.11%** |

All above the 95%-per-file bar. What is uncovered in this batch's own new
code, exhaustively: the three-line `.doc` re-open error branch in
`resolve_filter` (unreachable — `ffi_open_segment` validates `.doc` at open
time; it is the thirteenth copy of a defensive block `query.rs` and
`explain.rs` already carry twelve of, and it is written the same way rather
than a `.expect()`), and multi-line expression fragments the region counter splits.
`vectors.rs`'s move is that block plus ~560 new lines against a 1080-line
starting point; the other three files rose.

## Gate

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-codecs -p lucene-search -p lucene-ffi --all-targets
  -- -D warnings` — clean.
- `cargo test -p lucene-codecs -p lucene-search` — **2300 tests, all green**,
  including all 21 of `vector_query_fixtures.rs` (c16's 16 unmodified) and the
  new codec unit tests.
- `cargo test -p lucene-ffi` — **512 tests, all green**. `vectors.rs`'s module
  goes from 25 tests to 30: the 25 pre-existing ones (c13's, as amended by
  c16 — including `knn_search_reproduces_lucene_knn_vector_query_results`)
  are **unmodified**, and the five new ones are this batch's filtered
  submodule.
- `cargo test -p lucene-codecs -p lucene-search -p lucene-ffi` together —
  **2817 tests, 0 failures**.

  Worth recording because it cost this batch about an hour: for most of the
  batch `lucene-ffi` did not *compile* in this working tree, for a reason
  entirely outside it. A concurrent batch added five variants to
  `lucene_index::index_writer::Error` (`DocValuesRead`, `NormsRead`,
  `MergeSortDisagreement`, `UnknownSortField`, `MergeSortColumnMissing`) and
  `crates/lucene-ffi/src/writer.rs`'s `map_writer_error` — which deliberately
  has **no** `_` arm, precisely so a new variant has to be *classified*
  rather than silently becoming `Io` — had not caught up. That is the guard
  working as designed, and this batch left the file alone rather than
  classifying another batch's errors for it; the interim gate results were
  obtained by adding the five arms locally, running the suite, and reverting.
  The owning batch has since landed them, and every number above is from the
  tree as it stands.
- `python3 scripts/check-parity.py` — ok.
- **`c19-coverage-hardening`'s arithmetic gate**: `hnsw.rs` and
  `hnsw_vectors.rs` are still under `#[allow(clippy::arithmetic_side_effects)]
  // TODO(arith-audit)` in `lucene-codecs/src/lib.rs`, and `lucene-search` and
  `lucene-ffi` have the lint off entirely, so nothing here is gated yet. This
  batch's new arithmetic is written to pass it regardless: the only addition
  on a disk-derived value is `i64::from(g.max_node_id()).saturating_add(1)`
  in `search`'s accept-set bound, widened before it is added to and saturating
  rather than wrapping, and every comparison around it is in `i64` with an
  explicit `.max(0)`. No new `Vec::with_capacity`, index or slice takes a
  length from the caller or the file unbounded.
- `scripts/verify-write-path.sh` — green, confirmed by running it rather than
  assumed: **20/20** when this batch started and **21/21** at the end, the
  extra case being `VerifySortedSegment <- write_sorted_merged_segment_fixture`,
  added by a concurrent batch while this one ran. Nothing here writes bytes,
  so this is a no-regression check rather than evidence about the change.
- `docs/parity.md` updated in the same change: `SeededHnswGraphSearcher` moved
  from "not ported" to ported with the measurement, the `vector_query.rs` row
  records the seeded re-entry and the deleted duplicate, and the
  `lucene-ffi/src/vectors.rs` row gains the two filtered exports.
- `fixtures/README.md` documents both new generators, including why each
  fixture's shape is forced rather than chosen.

## Tier-2 review

Run on this batch's files after the gate was green. Three gating findings and
four advisories; **all seven acted on**, and two of the three gating ones are
the same defect this batch had just written up twice as findings 12 and 13 —
a comment that justifies a gap the change had already closed.

**Gating 1 (real, fixed).** `vectors_input`'s `filter: None` still carried
"No filtered KNN over this ABI yet: the filter query would need its own clause
array here". Both new entry points set `filter` on the value it returns. The
comment now states the actual contract — `vectors_input` returns the
*unfiltered* input and the filtered entry points patch it, because the accept
bitset has to outlive that frame — and this report's correspondence table,
which said "`filter` is the caller's, when there is one", is corrected to say
the same.

**Gating 2 (real, fixed).** Inserting `seed_slice` between `run_leaves`'s doc
comment and `run_leaves` left a 22-line rustdoc block on a four-line accessor,
opening with three paragraphs about why the fan-out is not
`merge_multi_segment_scored_concurrent` — and `run_leaves` with no doc at all.
The paragraph is load-bearing; it is back where it belongs.

**Gating 3 (real, fixed).** A discarded draft ("`perLeafTopK` is 13/6/6/2… no:
…") survived into a test's doc comment. Removed, and the corrected numbers
(30, 24, 24, 6 at `k = 10`) are stated properly — see advisory 2, which the
same paragraph turned out to need.

**Advisories, all acted on.**

1. *The new accept-set bound used the wrong upper bound.* It checked
   `bits.len() >= scorer.max_ord()`, but the **graph** branch indexes the
   accept set with ordinals the arcs name, bounded by `maxNodeId()`, and
   nothing cross-checks the `.vem`'s node count against the `.vec`'s vector
   count. A corrupt meta or a graph handed in for the wrong field therefore
   still reached `FixedBitSet::get` out of range — the same hazard finding 4
   exists to close, and inconsistent with `search_seeded`, which already
   bounds against the graph. Now `max(maxOrd, maxNodeId + 1)`, with a test
   (`OnHeapHnswGraph::with_size(4, 400)` over a 300-vector scorer).
2. *The "no re-entry" control asserted nothing about re-entry.* It only
   compared against Lucene, and the 40-document leaf's `perLeafTopK` at
   `k = 10` is **6**, i.e. below `k` — so re-entry is excluded by score, not
   structurally, and the control would silently stop being one if the vectors
   ever shifted. It now asserts, from Lucene's recorded answer alone, that no
   leaf contributes more than its `perLeafTopK` hits.
3. *The recorded `perLeafTopK` values are re-derived, not ground truth.*
   `perLeafTopKCalculation` is private, so `GenVectorsSeeded` copies the
   formula — meaning that assertion compares this port's formula against a
   hand-copy of the same one and cannot catch a shared misreading. Said so in
   the generator, in the test, and in finding 7 above; assertion 3 (Lucene's
   answer taking more than `perLeafTopK` from the leaf) is the one with weight.
4. *"Strictly cheaper" overstated what seeding buys.* Seeding removes an
   `O(log n)` descent and adds a bulk score of `|seeds|` entry points, so it
   can lose once the seed set approaches the beam — and this batch's own
   measurement (93 seeds over 700 nodes, 4.1%) sits near that crossover. The
   module doc now states the trade and cites both numbers, matching finding 9.

The reviewer independently confirmed against the tag: `search_seeded`'s fold
of `fromEntryPoints` + `AbstractHnswGraphSearcher.search` (including that the
`eps[0] == UNK_EP` branch is subsumed by the `ord < 0` rejection, and that
validating against `graph.size()` is the stricter of the two available
bounds); `hnsw_vectors::search` statement for statement against
`Lucene99HnswVectorsReader.search`; **finding 5**, including that
`ReentrantKnnCollectorManager` delegates to a fresh `TopKnnCollectorManager`
and not to the optimistic wrapper, so phase 2's collector can legitimately be
*smaller* than phase 1's (24 → 10 in the unit test); that `AcceptDocs.cost()`
really is exactly this port's ordinal-space cardinality, because `rewrite`
ANDs `FieldExistsQuery(field)` into the filter before creating the weight; and
that the seeded fixture discriminates — all 20 `k = 100` queries take 100 of
100 hits from a leaf whose `perLeafTopK` is 93.

## Carry-over

- **A lazy accept set** (c16 finding 12's `O(size)` sparse build) — still open.
  It needs a `Bits`-equivalent on `HnswGraphSearcher::search`, i.e. a trait
  object or a generic in the innermost loop, which is a design question rather
  than a missing parameter now that the parameter exists.
- **`FilteredHnswGraphSearcher`** — unreachable at 10.5.0's
  `DEFAULT_FILTERED_SEARCH_THRESHOLD == 0`. It becomes reachable if this port
  ever exposes the threshold (`main` defaults it to 60), and only then.
- **A multi-segment KNN entry point over the C ABI.** `lucene-search` has the
  fan-out, the re-entry pass and the merge; `lucene-ffi` still exposes one
  segment per call, so a JVM caller must merge leaf results itself. That is a
  `directory_reader.rs`-shaped addition, not a `vectors.rs` one.
