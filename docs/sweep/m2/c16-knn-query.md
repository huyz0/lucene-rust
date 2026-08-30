# c16-knn-query

Follow-up batch closing `c13-ffi-surface`'s finding 22: Java's *query-level*
KNN policy sat in `lucene-ffi` because `lucene-search` had no vector module at
all, so no non-FFI consumer — including the multi-segment fan-out — could run
a vector query. That is upside down against the crate rule
`util <- store <- codecs <- index <- search <- core <- ffi`.

Files swept: new `crates/lucene-search/src/vector_query.rs`,
`crates/lucene-ffi/src/vectors.rs` (rewritten to delegate),
`crates/lucene-search/src/lib.rs` (module wiring + two `Error` variants). New:
`fixtures/src/GenVectorsMulti.java`, `fixtures/data/vectors_multi_index/`,
`crates/lucene-search/tests/vector_query_fixtures.rs`,
`crates/lucene-search/benches/knn_multi_segment.rs`. Also touched:
`docs/parity.md`, `fixtures/README.md`.

Java counterparts, all read from the **pinned 10.5.0 tag**
(`/home/tuong/work/lucene-10.5.0`, cross-checked byte-identical against
`git show releases/lucene/10.5.0:` in the working checkout — this matters, see
finding 1):

- `search/{AbstractKnnVectorQuery, KnnFloatVectorQuery, KnnByteVectorQuery,
  AcceptDocs, KnnCollector, AbstractKnnCollector, TopKnnCollector,
  VectorScorer}.java`
- `search/knn/{KnnCollectorManager, TopKnnCollectorManager,
  KnnSearchStrategy}.java`
- `codecs/lucene99/Lucene99HnswVectorsReader.java` (its `search` dispatch)
- `util/hnsw/HnswGraphSearcher.java` (the `acceptOrds`/`filteredDocCount`
  overload)

Plus one handed-off file, `crates/lucene-search/src/query.rs` — see
"`BooleanQuery` duplicate dedup (handoff from `c18-version-audit`)" at the end.

Totals: **15 findings** — 4 CORRECTNESS (all fixed), 7 MISSING (6 fixed, 1
recorded with the owner named), 3 PERF (all measured; 1 fixed), 1 INTENTIONAL.

---

## `crates/lucene-search/src/vector_query.rs` (new)

Java counterparts: the eight `search/*.java` files above plus
`search/knn/*.java` and the dispatch half of
`Lucene99HnswVectorsReader.search`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `KnnFloatVectorQuery`/`KnnByteVectorQuery` (structs) + `::new` | the two query classes' constructors, incl. `k < 1` | identical, plus three non-Java knobs (finding 11) |
| `search_knn_float_vector_query`/`..._byte_...` | `IndexSearcher.search(query, k)` over a single-leaf reader | ported |
| `search_knn_*_multi_segment` | `AbstractKnnVectorQuery.rewrite` + `runSearchTasks` + `mergeLeafResults` | ported; seeding of the re-entry pass is finding 6 |
| `search_knn_*_multi_segment_concurrent` | the same over `IndexSearcher`'s `TaskExecutor` | ported (finding 13 measures it) |
| `per_leaf_top_k` | `AbstractKnnVectorQuery.perLeafTopKCalculation` | identical, float/double split included (finding 1) |
| `reentry_leaves` | `rewrite`'s `perLeaf.scoreDocs[len-1].score >= minTopKScore` loop | identical |
| `reentry_plan` | `ReentrantKnnCollectorManager.newCollector`'s delegate | ported minus the seeding (finding 6) |
| `plan_leaves` | `OptimisticKnnCollectorManager.newCollector` + `getLeafResults`' `leafProportion` | identical |
| `merge_leaves` | `mergeLeafResults` -> `TopDocs.merge(k, ..)` | reuses `multi_segment::merge_multi_segment_scored` |
| `resolve_field` | `KnnFloat/ByteVectorQuery.approximateSearch`'s `checkField` + `AbstractKnnVectorQuery`'s dimension/`k` checks | ported (finding 3) |
| `AcceptOrds`/`accept_ords` | `AcceptDocs` + `KnnVectorValues.getAcceptOrds` | ported, in ordinal space (findings 2, 12) |
| `accept_bitset` | `AcceptDocs.fromIteratorSupplier`'s bitset build | ported; the caller supplies the doc ids (finding 5) |
| `leaf_results` | `AbstractKnnVectorQuery.getLeafResults` | ported, both branches and both fall-backs (finding 4) |
| `exact_search` | `AbstractKnnVectorQuery.exactSearch` | ported; `HitQueue` -> `KnnCollector` (finding 10) |
| `hnsw_search` | `Lucene99HnswVectorsReader.search`'s dispatch | ported; lives here only for the `acceptOrds` parameter (finding 7) |
| `flush_bulk` | the same reader's bulk-score batch | identical |
| `similarity_ordinal`/`similarity_from_ordinal` | `Lucene94FieldInfosFormat`'s pinned ordinal list | identical |

Java methods with **no** Rust counterpart, deliberately: `rewrite`'s
`MatchNoDocsQuery`/`DocAndScoreQuery` result-query construction (this port
returns a `Vec<ScoreDoc>`, it has no `Query` rewrite pipeline);
`ReentrantKnnCollectorManager`'s seeding (finding 6);
`TimeLimitingKnnCollectorManager` (`multi_segment` already has this port's
deadline shape, and no `QueryTimeout` reaches here);
`visit`/`equals`/`hashCode`/`toString`; `getTargetCopy`;
`PatienceKnnVectorQuery`, `SeededKnnVectorQuery`, `KnnFloat16VectorQuery` and
the `*VectorSimilarityQuery` family (different queries, not this one).

### 1. [CORRECTNESS] Per-leaf `k` is **pro-rata** in 10.5.0, not `k`

The brief for this batch said, and it is the intuitive reading, that "Java
searches each leaf for `k` and merges, it does not divide `k` across leaves."
That was true before Lucene 10.3 and is **false for the pinned 10.5.0**.

Java: `TopKnnCollectorManager.isOptimistic()` returns `true`, so
`AbstractKnnVectorQuery.rewrite` wraps it in an
`OptimisticKnnCollectorManager` whose `newCollector` sizes each leaf's
collector at

```
perLeafTopK = (int) max(1, k*p + 16*sqrt(k*p*(1-p))),  p = leafMaxDoc / indexMaxDoc
```

Us (had the brief been followed): `k` per leaf. For the four unequal segments
in the new fixture at `k = 10` the two disagree on every leaf — Java collects
30, 24, 24 and 5 where "k per leaf" collects 10, 10, 10 and 10. A wider
collector is a wider beam, so the *approximate* answer differs; three of the
four leaves would have been searched too narrowly and the fourth too widely.

Consequence: a multi-segment KNN query would have returned a different result
set from Lucene's on the same index — usually a *better* one for three leaves
and a worse one for the small leaf, which is precisely the shape of divergence
a recall metric hides (c5's Tier-2 lesson).

Resolution: **fixed** — `per_leaf_top_k` ports the formula bit-for-bit,
including Java's float/double split (`k * leafProportion` and the variance are
`float`, `Math.sqrt` widens to `double`, `(int)` truncates toward zero). A
rounding difference here moves the collector size by one and therefore moves
which documents come back, so it is a bit-level port, not a formula that looks
the same. Tests: `per_leaf_top_k_is_javas_pro_rata_formula` (five points,
including the `Math.max(1, ..)` floor and a NaN `leafProportion` from a
zero-document index), and — the one that actually discriminates —
`multi_segment_knn_reproduces_lucene`, doc-for-doc over 80 queries on a
four-segment index.

### 2. [CORRECTNESS] Deletions were filtered *after* the walk, not inside it

Java: even with no filter query, `getLeafResults` builds
`AcceptDocs.fromLiveDocs(liveDocs, maxDoc)` and the reader hands
`scorer.getAcceptOrds(accepted)` to `HnswGraphSearcher` as a lazy `Bits`, so a
deleted node never enters the collector.

Us (c13's shape, inherited): `hnsw_vectors::search` has no `acceptOrds`
parameter, so c13 widened the beam by the segment's deleted count and dropped
deleted documents after the ordinal->doc translation. c13 recorded the
divergence honestly and named the blocker as "adding `acceptOrds` is a
`lucene-codecs` change".

**The blocker was misdiagnosed, and that is the finding.**
`lucene_codecs::hnsw::HnswGraphSearcher::search` already takes
`accept_ords: Option<&FixedBitSet>` — c5 ported it, with a test
(`search_honours_accepted_ordinals_and_the_visit_limit`). Only the
`hnsw_vectors::search` *convenience wrapper* lacks the parameter. So the
faithful mechanism was reachable from the search layer all along, using public
codec primitives and touching nobody else's files.

Consequence of the old shape: a different (still correct, but not Java's)
approximate answer on any segment with deletions, plus a beam widened by the
deleted count — which on a heavily-deleted segment degrades into the
exhaustive scan.

Resolution: **fixed.** `accept_ords` builds Java's ordinal-space accept set
and `leaf_results` hands it to the walk. The beam widening is gone. Tests:
`deleted_documents_never_come_back` and
`deletions_on_a_sparse_field_are_translated_into_ordinal_space` (the sparse
case is the one where ordinal != doc id, so an off-by-one in the translation
returns documents the caller deleted). c13's own FFI-level deletion test still
passes **unmodified**, which is what says the exported behaviour is preserved.

### 3. [CORRECTNESS] `k` reached a heap allocation unvalidated (carried over, kept)

c13's finding 12 in new code: `KnnCollector::new(k, ..)` builds a
`TernaryLongHeap` of `k` entries, and a negative Java `int` widened to `usize`
would reach `Vec::with_capacity` and **abort**, which `catch_unwind` cannot
contain. The clamp moved down with the policy and is now in `leaf_results`
(`plan.per_leaf_k.min(size)`), documented there as a safety property rather
than an optimisation — it changes no result, since a queue larger than the
population can never fill. `check_k` keeps Java's `k < 1` rejection at the
query constructor *and* at every entry point, since the struct's fields are
public. Tests: `k_zero_is_rejected_like_javas_constructor`, and c13's
`an_absurd_k_is_clamped_to_the_field_size_not_allocated` (`usize::MAX` through
the exported C symbol) still passes unmodified.

### 4. [MISSING] Filtered KNN did not exist

Java: `AbstractKnnVectorQuery` takes an optional filter `Query`, rewrites it
together with an implicit `FieldExistsQuery(field)` under two `Occur.FILTER`
clauses, and then per leaf:

1. `cost <= perLeafTopK` -> `exactSearch` over the accepted documents;
2. otherwise an approximate search with `acceptOrds` and
   `visitedLimit = cost + 1`;
3. if that came back early-terminated **or** with fewer than `perLeafTopK`
   hits -> `exactSearch` after all.

Us: nothing. The brief asked to "assess whether it is contained; if the
fallback heuristic needs a cost model this port does not have, port the
unfiltered path plus `acceptDocs` and record the heuristic with the blocker
named — do not guess a threshold."

**It is contained, and no threshold had to be guessed**, because Java's `cost`
is not an estimate: `AcceptDocs.DocIdSetIteratorAcceptDocs.cost()` materialises
the accept bitset and returns `BitSet.cardinality()`. This port has the same
number exactly — and gets it *more* cheaply, because building the accept set
in **ordinal** space (finding 12) folds in the `FieldExistsQuery` conjunct for
free, and `|accept ords|` equals Java's doc-space cost by construction.

Resolution: **fixed** — all three steps ported, plus Java's separate
`filterWeight == null` path (which is not "a filter that accepts everything":
Java skips the cost heuristic *and* the `visitedLimit` cap entirely). Tests:
`a_selective_filter_matches_lucenes_exact_fallback` (20 accepted documents in
the whole index, so every leaf takes step 1 — doc-for-doc against Lucene),
`a_permissive_filter_matches_lucenes_filtered_graph_walk` (a quarter of the
index, so step 2 — doc-for-doc, and every returned document asserted to be in
the filter), and `an_empty_filter_returns_no_hits` (the degenerate cost-0
case, which a graph walk that quietly ignored the filter would fail).

### 5. [INTENTIONAL] The filter is a resolved doc set, not a `Query`

Java's query object holds a filter `Query` and resolves it per leaf through
`Weight`/`Scorer` inside `rewrite`. This port has no `IndexSearcher`/`Weight`
at all, so `VectorsInput::filter` takes the already-resolved per-segment
bitset — exactly the shape `live_docs` already has in every other query
function in this crate, and exactly the shape
`points_query::PointsInput` established for "an opened resource the caller
supplies". `accept_bitset` converts a `resolve_clause_docs` doc-id list (which
is what a `BooleanQuery` with `Occur::FILTER` clauses produces, since c11)
into it in one line.

The one thing this loses is Java's two rewrite short circuits (a
`MatchNoDocsQuery` filter returns `MatchNoDocsQuery`; a `MatchAllDocsQuery`
filter is dropped and the unfiltered path taken). The first is behaviourally
identical here — an all-zero bitset gives cost 0 and an empty result, which
`an_empty_filter_returns_no_hits` asserts. The second is a caller decision:
pass `filter: None` for a match-all filter and you are on Java's exact path.
Both are stated on `VectorsInput::filter`.

### 6. [MISSING, recorded — owner named] The re-entry pass is not *seeded*

Java: `ReentrantKnnCollectorManager` builds phase 2's collector with a
`KnnSearchStrategy.Seeded` wrapping phase 1's hits as entry points, and
`HnswGraphSearcher.search` honours it by delegating to
`SeededHnswGraphSearcher.fromEntryPoints`.

Us: phase 2 runs with the same full-`k` collector but starts from the graph's
own entry point.

Consequence: seeding changes only *where the walk starts*, never the collector
size, the accept set or the merge, so it can only change which approximate
answer comes back — and on this fixture it does not change it at all
(`multi_segment_knn_reproduces_lucene` matches Lucene doc-for-doc on all 80
queries, five of which take the re-entry path).

**Recorded, not fixed. Owner named**: `SeededHnswGraphSearcher` and
`KnnSearchStrategy` are `org.apache.lucene.util.hnsw` /
`org.apache.lucene.search.knn`, which this port places in
`crates/lucene-codecs/src/hnsw.rs` — `c10-vectors-wiring`'s file. Stated on
`search_knn_float_vector_query_multi_segment` and in `docs/parity.md`.

### 7. [MISSING, recorded — owner named] `hnsw_vectors::search` has no `acceptOrds`

Java's `Lucene99HnswVectorsReader.search` takes `AcceptDocs` and computes
`filteredDocCount = min(acceptDocs.cost(), graphSize)`; this port's
`hnsw_vectors::search` takes neither, and hardcodes
`filteredDocCount = graphSize`. Finding 2's fix therefore needed the dispatch
with both parameters, and it lives in `vector_query.rs::hnsw_search` because
`crates/lucene-codecs/src/hnsw_vectors.rs` is `c10-vectors-wiring`'s file.

That is a genuine altitude wart — the same class of problem this batch exists
to fix — so it is bounded two ways rather than merely noted:

- the duplication is ~40 lines and is stated as duplication on the function
  itself, with the owner and the exact missing parameters named;
- `the_unfiltered_dispatch_matches_the_codec_entry_point` runs **both**
  functions over the real fixture graph at `k = 1, 10, 200` and requires
  identical output. Stated precisely, because it is easy to overclaim: that
  pins the *shared* half only — with no accept set and
  `filteredDocCount == graphSize` the two functions are the same code — not
  the two parameters that motivated the copy. Those are pinned instead by the
  filtered and deletions-aware differential tests, against real Lucene rather
  than against this function's twin.

The fix, when `hnsw_vectors.rs` is free: add `accept_ords:
Option<&FixedBitSet>` and `filtered_doc_count: i32` to
`hnsw_vectors::search`, and delete `vector_query::hnsw_search`.

### 8. [MISSING] `TopKnnCollector`/`VectorScorer` — already ported, not re-ported

The brief listed `KnnCollector`, `TopKnnCollector` and `VectorScorer` as files
to port. They already are, and porting them again would have been the mistake:

- `TopKnnCollector`/`AbstractKnnCollector` are `lucene_codecs::hnsw::KnnCollector`
  (c5 collapsed them into one type, because Java's collector is a
  `NeighborQueue` plus a visit limit and the graph builder needs the same
  thing);
- `VectorScorer` is `hnsw::VectorScorer` +
  `vectors::{FloatVectorScorer, ByteVectorScorer}`;
- `KnnCollector` the interface has no separate existence in Rust.

Recorded here rather than silently skipped, and stated at the top of
`vector_query.rs` so the next reader does not re-derive it. The one Java
method on that path with no equivalent, `VectorScorer.bulk(DocIdSetIterator)`,
is subsumed by `VectorScorer::bulk_score(&[i32], &mut [f32])` — c5's port
already batches by ordinal, which is what `exact_search` uses.

### 9. [MISSING] `k`/`efSearch` and the four similarities across both encodings

Covered by construction rather than by a branch: `KnnQuery` is one trait with
two implementations, so `k`, `ef_search`, `visited_limit`, `similarity`, the
accept set, the fan-out, the re-entry pass and the merge are written once and
both encodings get all of them. Tests:
`knn_queries_reproduce_lucene_over_one_segment` runs every query the
single-segment fixture records — 80 of them, across a dense EUCLIDEAN field, a
sparse COSINE one, a sparse MAXIMUM_INNER_PRODUCT one and a BYTE DOT_PRODUCT
one — and `multi_segment_byte_knn_reproduces_lucene` plus
`the_concurrent_byte_fan_out_returns_exactly_the_sequential_answer` do the same
for BYTE across four segments.

### 10. [INTENTIONAL] `exactSearch`'s sentinel drain is not reproduced

Java's `exactSearch` prefills a `HitQueue` with `(Integer.MAX_VALUE,
-Infinity)` sentinels and drains them afterwards with
`while (queue.size() > 0 && queue.top().score < 0) queue.pop()`. That loop
removes exactly the unfilled slots, because every `VectorSimilarityFunction`
maps into a non-negative range — including the byte `DOT_PRODUCT` transform,
which bottoms out at exactly `0` (`0.5 + dot/(dim * 2^15)`, and
`dot >= -dim * 2^14`). A collector that simply holds fewer than `k` hits is
therefore the same answer, and `KnnCollector` is that collector. The tie-break
is also the same: `HitQueue.lessThan` prefers the lower doc id on an equal
score, `NeighborQueue` prefers the lower ordinal, and ordinals ascend with doc
ids by construction. Both equivalences are argued on `exact_search` rather
than assumed, and
`a_selective_filter_matches_lucenes_exact_fallback` checks the result against
Lucene's rather than against the argument.

### 11. [INTENTIONAL] Three knobs Java's query object does not have

`ef_search` (OpenSearch's `num_candidates`), `visited_limit` and `similarity`
are c13's, kept because the C ABI already exposes them and this batch must not
change exported behaviour. All three are defined so that the Lucene default
reproduces Lucene exactly: `ef_search == 0` means "use this leaf's own `k`",
`visited_limit == 0` is Java's `Integer.MAX_VALUE`, and `similarity == None`
is "the field's own". `similarity` is a **cross-check, not an override** — an
HNSW graph's arcs encode the build-time similarity's neighbourhood, so walking
it under another one silently degrades recall with no error at all. Tests:
`query_builders_set_exactly_what_they_name`,
`a_wider_ef_search_never_returns_worse_hits`,
`a_tight_visited_limit_stops_the_exhaustive_scan_early`, and (through the C
ABI, unmodified from c13) `a_matching_similarity_is_accepted_and_a_mismatching_one_is_not`.

### 12. [PERF] The accept set is built in ordinal space, and borrowed when it can be

Java allocates a `Bits` wrapper per leaf per query
(`KnnVectorValues.getAcceptOrds`) and pays a virtual call for every visited
node. This port builds a `FixedBitSet` over **ordinals** instead, which:

- makes the `FieldExistsQuery(field)` conjunct free rather than a second
  clause to evaluate, and makes `cost()` a `cardinality()` on a bitset the
  size of the *field* rather than of `maxDoc`;
- costs `O(size)` once per leaf per query where Java's is lazy — which is why
  the **dense** case (ordinal == doc id, which is every field every document
  has a value for) skips the build entirely and borrows the caller's bitset,
  `AcceptOrds::Borrowed`. That is the common case: it is what a deletions-only
  search on a dense field does, i.e. the one that runs on every shard.

The sparse build is `O(size)` `ordToDoc` lookups, against a graph walk's
~`O(log size)` visited nodes — a real cost, and the honest statement is that
it is bounded by the same `O(size)` an exhaustive scan would pay and is only
reached when the field is sparse *and* the query is filtered or the segment
has deletions. Removing it needs a lazy `Bits`-equivalent on
`HnswGraphSearcher::search`, i.e. `hnsw.rs` — finding 7's owner.

### 13. [PERF] The four-leaf fan-out costs 7x one leaf, and rayon costs more than it saves here

Measured with `cargo bench -p lucene-search --bench knn_multi_segment`
(20 queries per iteration, `k = 10` unless stated, 4000 documents either way,
release):

| | per query |
|---|---|
| one segment, 4000 documents | **5.4 us** |
| four unequal segments (2000/1000/960/40), sequential | **38 us** |
| four segments, rayon `par_iter` | **198 us** |
| four segments, sequential, `k = 100` | **128 us** |
| four segments, rayon, `k = 100` | **232 us** |
| four segments + selective filter (20 documents) | **7.8 us** |
| four segments + permissive filter (1000 documents) | **52 us** |

Two things worth saying plainly:

- **The 7x is the pro-rata policy, not fan-out overhead.** Four leaves at
  `k = 10` collect 30 + 24 + 24 + 5 = 83 slots against one leaf's 10, and a
  wider collector is a wider beam. Java pays exactly the same — this port
  reproduces its collector sizes, which is finding 1. Splitting an index into
  segments genuinely costs KNN queries more; that is a Lucene property, not a
  port artifact.
- **The rayon fan-out is a pessimization at this size**, by 5.2x at `k = 10`
  and 1.8x at `k = 100`. The gap narrows with the work per leaf, as a fixed
  dispatch cost should, but has not closed even at ten times the work: a leaf
  search here is tens of microseconds and rayon's per-task cost is comparable.
  Recorded on the function itself rather than removed, because
  `multi_segment.rs` established the sequential/concurrent pair and a real
  OpenSearch shard's leaves are three orders of magnitude larger than this
  fixture's. Not "fixed" — there is nothing to fix; the measurement is the
  finding.
- The **selective filter is cheaper than no filter at all** (7.8 vs 38 us),
  which is the whole point of Java's `cost <= perLeafTopK` short circuit: an
  exact scan over <= 10 accepted vectors beats a graph walk. The permissive
  filter costs 1.4x the unfiltered fan-out, which is the `acceptOrds` check
  per visited node plus the ordinal-space accept set of finding 12.

### Verdict

Swept clean for the read/search path. Open: finding 6 (seeding) and finding 7
(`acceptOrds` on `hnsw_vectors::search`), both blocked on
`c10-vectors-wiring`'s files and both bounded by a test.

---

## `crates/lucene-ffi/src/vectors.rs` (rewritten)

No Java counterpart — it *is* the C-ABI boundary. Per the protocol's rule 1 no
Java path is invented for it; the wrapped concepts are the `lucene-search`
functions above.

### Method correspondence

| Rust | Was | Now |
|---|---|---|
| `ffi_open_vectors` | opens `.fnm`/`.vemf`/`.vec`/`.vem`/`.vex`, validates | unchanged |
| `ffi_vectors_set_live_docs` | `segment::decode_live_docs` | unchanged |
| `ffi_close_vectors` | registry removal | unchanged |
| `ffi_knn_float_vector_search` / `ffi_knn_byte_vector_search` | owned `knn_params`/`check_similarity`/`ord_to_doc`/`to_scored_docs`/`knn_search` | decode arguments, build the query, call `lucene_search::vector_query`, insert a `ScoredResultsHandle` |
| `decode_similarity` (new) | part of `check_similarity` | the *decode* half only: `-1` -> `None`, `0..=3` -> the enum, anything else `InvalidArgument` |
| `vectors_input` (new) | inline in `knn_search` | reopens the readers as a `VectorsInput` |
| `map_knn_error` (new) | `map_vectors_error` for everything | `InvalidKnnQuery` -> `InvalidArgument`, `Vectors` -> `Decode`, else `Search` |
| ~~`knn_params`~~, ~~`check_similarity`~~, ~~`ord_to_doc`~~, ~~`to_scored_docs`~~, ~~`knn_search`~~, ~~`similarity_ordinal`~~ | — | **removed**; moved to `lucene-search` |

### 14. [MISSING -> fixed] c13's finding 22, closed

The exported symbols, their signatures and their behaviour are unchanged. What
proves it is c13's own test suite, kept **unmodified**: 23 tests including
`knn_search_reproduces_lucene_knn_vector_query_results`, which runs every one
of the 80 queries `fixtures/data/vectors_index/manifest.properties` records
*through the exported C symbols* and asserts real Lucene's
`KnnFloatVectorQuery`/`KnnByteVectorQuery` doc-for-doc and score-for-score.
Two c13 unit tests were replaced rather than kept, because the functions they
tested no longer exist here: `knn_params_clamps_and_defaults_as_documented`
(now `vector_query`'s own tests) and
`similarity_ordinals_are_the_pinned_file_format_order` (now
`the_similarity_argument_decodes_the_pinned_file_format_ordinals`, which tests
the half this module kept — the decode — over every valid ordinal and five
invalid ones).

The one *behavioural* change is finding 2: deletions now take Java's
`acceptOrds` path. c13's deletion test passes unmodified, including its
assertion that the surviving prefix of the undeleted top-10 is unchanged and
in order.

### 15. [CORRECTNESS] The error split had to move with the policy

If every `lucene_search::Error` had mapped to `FfiStatus::Search`, c13's
finding 23 would have regressed: a caller mistake (an unknown field, a
wrong-length query vector, `k = 0`) would stop being `InvalidArgument`. The
split is carried by a dedicated `Error::InvalidKnnQuery(String)` holding
Java's own message, next to `Error::Vectors` for a genuine decode failure —
which a JNI caller reads as "this index is corrupt" and may fail a shard over.
Test: `a_caller_error_is_an_invalid_argument_and_a_decode_error_is_a_decode`
asserts all three arms directly, and the five inherited argument-rejection
tests assert the statuses and Java's messages end to end.

### Verdict

Swept clean. `vectors.rs` is now a thin wrapper, the same shape as
`points_query.rs` and `sort.rs`.

---

## Evidence

**Against real Lucene, not against our own reader.** Two fixtures, both driven
by a real `IndexWriter` and queried through a real `IndexSearcher`:

- `fixtures/data/vectors_index/` (c5's): one 4000-document segment, five
  vector fields. 80 single-segment queries reproduced doc-for-doc and
  score-for-score, plus the graphless exact branch, the sparse ord->doc
  mapping, deletions on both a dense and a sparse field, and a
  `visited_limit`-truncated scan.
- `fixtures/data/vectors_multi_index/` (**new**, `GenVectorsMulti.java`): the
  same four field shapes over 4000 documents split across four deliberately
  *unequal* segments (2000/1000/960/40), plus two `StringField`s used as
  filters. 80 multi-segment queries (dense/sparse/MIP FLOAT32 and BYTE) and 40
  filtered ones, all doc-for-doc and score-for-score.

Three things about the multi fixture are deliberate rather than incidental,
because a fixture that does not reach a branch proves nothing about it:

1. **Unequal segments** put four different `perLeafTopK` values (30, 24, 23,
   5) into one query — finding 1's whole point.
2. **The 40-document segment's vectors are a tight cluster near the origin**,
   and every fifth dense query target is pulled toward it. That is what makes
   the optimistic re-entry pass fire: a leaf can only be re-entered when its
   `perLeafTopK` is *below* `k` and its hits are genuinely competitive. Five of
   the twenty dense queries then return more than five hits from that leaf,
   which no phase-1-only implementation can produce —
   `the_optimistic_reentry_pass_is_what_fills_k_from_a_small_leaf` asserts
   exactly that, so `multi_segment_knn_reproduces_lucene` cannot pass by
   accident on a port that skipped phase 2.
3. **The 40-document segment carries no HNSW graph at all** (Lucene's
   `shouldCreateGraph` threshold), so the fan-out has to merge one exact leaf
   with three approximate ones.

**No assertion here is a recall threshold**, per c5's Tier-2 lesson that recall
does not discriminate for this subsystem (mutating the diversity rule took
graph agreement to 1/4273 while recall *rose*). Every test compares against
Lucene's actual result list; the two that are not differential comparisons
(`the_optimistic_reentry_pass_is_what_fills_k_from_a_small_leaf`,
`the_unfiltered_dispatch_matches_the_codec_entry_point`) are structural
equalities, not metrics.

The filter tests take the **accepted local doc ids straight out of Lucene's
own postings**, recorded per leaf in the manifest, so what is under test is the
KNN policy and not this port's term-query resolution.

## Coverage

`cargo llvm-cov -p lucene-search -p lucene-ffi --summary-only`, lines:

| file | before | after |
|---|---|---|
| `lucene-search/src/vector_query.rs` | — (new) | **97.65%** |
| `lucene-ffi/src/vectors.rs` | 97.89% | **98.51%** |
| `lucene-search/src/query.rs` (handoff) | 98.27% | **98.37%** |

All above the 95%-per-file bar. The remaining uncovered lines in
`vector_query.rs` are the `num_vectors == 0` guard (a field with no vectors at
all, which no fixture segment has), the "a field in `.vemf` but not in `.vem`"
arm (unreachable with Lucene-written files — both metas list every vector
field), and multi-line expression fragments the region counter splits.

## Gate

- `cargo fmt --all` — clean.
- `cargo clippy -p lucene-search -p lucene-ffi --all-targets -- -D warnings` —
  clean.
- `cargo test -p lucene-search -p lucene-ffi` — 29 test binaries, **1520
  tests, all green**.
- `python3 scripts/check-parity.py` — ok (the missing
  `lucene-search/src/vector_query.rs` row it reported is added).
- `docs/parity.md` updated in the same change: a new row for
  `vector_query.rs`, the `lucene-ffi/src/vectors.rs` row rewritten to say it is
  a thin wrapper and that deletions take the `acceptOrds` path, and the
  `hnsw_vectors` row's "no query surface here yet" corrected (there is one now,
  and `FilteredHnswGraphSearcher` is unreachable from `KnnFloatVectorQuery`
  regardless, since `DEFAULT_FILTERED_SEARCH_THRESHOLD` is `0`).
- `fixtures/README.md` documents `GenVectorsMulti.java` and why each of its
  three deliberate choices exists.

---

## `crates/lucene-search/src/query.rs` — `BooleanQuery` duplicate dedup

Handed off by `c18-version-audit` (its finding 3) while this batch held
`crates/lucene-search`. Java counterpart:
`/home/tuong/work/lucene-10.5.0/lucene/core/src/java/org/apache/lucene/search/BooleanQuery.java`,
`rewrite`'s two `Deduplicate <occur> clauses by summing up their boosts` blocks
(lines 428-458 and 460-490 of the tag).

### 16. [CORRECTNESS] The recorded rationale cited a method that does not exist in 10.5.0

`Query::rewrite`'s doc block declined to implement MUST/SHOULD dedup and
justified it at length: Java "does so by replacing the repeated clause with one
copy carrying a recombined boost computed from
`IndexSearcher.getSimilarity().computeQueryTermWeight(count)`", which this port
has no `Similarity` in scope to reproduce.

**`computeQueryTermWeight` does not exist in Lucene 10.5.0's `Similarity`.** It
is a later addition on Lucene `main`, which several earlier batches were
reading by mistake (the source-of-truth correction `c18` made repo-wide). So
the port justified a gap by citing code the pinned version does not contain —
which is worse than an unjustified gap, because it reads as settled.

Resolution: **fixed, both halves.** The rationale is replaced with 10.5.0's
actual rule, and the rule is implemented.

### 17. [MISSING] 10.5.0's dedup, which needs no `Similarity` at all

Java (10.5.0): unwrap each clause's `BoostQuery` chain multiplying the boosts
(`double`), sum per distinct unwrapped query (`double`), and — only when that
collapsed the count (`if (map.size() != clauseSets.get(occur).size())`) —
rebuild the bucket with one clause per distinct query, wrapped in a
`BoostQuery` iff its summed boost is not `1`. `SHOULD` is gated on
`minimumNumberShouldMatch <= 1`; `MUST` is ungated. No `IndexSearcher`
anywhere.

Resolution: **fixed** — `dedup_by_boost_sum` + `unwrap_boost_chain`, placed at
Java's own position in `rewrite` (after rule 11, before rule 12). The boost
arithmetic is Java's `double` product and `double` sum narrowed to `f32` once
at the end, not an `f32` accumulation, which would round differently for three
or more duplicates.

Two divergences, both stated on the function rather than left implicit:

- **Clause order is first-seen and deterministic.** Java rebuilds from a
  `HashMap.entrySet()`, whose iteration order is unspecified. A stable order is
  a strict improvement for a rewrite whose output feeds `explain`.
- **A rebuild re-enters `rewrite`** where Java returns and lets
  `IndexSearcher.rewrite` loop to a fixpoint. That is load-bearing, not
  cosmetic: summing `a^0.5 a^0.5` back to a *bare* `a` (Java's
  `if (boost != 1f)`) can make a later rule newly apply — rule 10's "drop a
  FILTER that is also a MUST" is exactly such a case, and
  `a_summed_boost_of_exactly_one_leaves_no_boost_wrapper_behind` pins it.

**One more re-entry was needed, and it is a real bug this found.** Java's
`should.size() == minimumNumberShouldMatch` promotion (all `SHOULD` become
`MUST`, threshold to 0) runs *after* the dedup blocks. Clauses it promotes were
exempt from SHOULD dedup precisely because `minimumNumberShouldMatch > 1` — and
that is no longer true of them once they are `MUST`. Java reaches the right
answer because it returns and loops; this port fell through and left
`+cat +cat` un-collapsed, where Java's fixpoint is `cat^2`. The promotion now
re-enters too. Test:
`should_dedup_is_skipped_when_minimum_should_match_counts_more_than_one`.

### 18. [INTENTIONAL] The dedup is score-neutral, and that is pinned, not argued

BM25 is linear in the clause sum, so `a a` scores exactly what `a^2` does
(`x + x` and `2 * x` are the same IEEE-754 value, and BM25 folds the boost into
`weight = boost * idf * (k1+1)`). Rather than leave that as an argument, two
new `AppendScoringManifest` entries record what real Lucene scores for
`body:cat body:cat body:dog` and `+body:cat +body:cat` over the existing
`blocktree_index` fixture, and
`bm25_scoring_fixtures.rs::duplicate_clause_dedup_is_score_neutral_against_real_lucene`
asserts this port's executor — which sums the *un*-rewritten duplicates
separately — reproduces them **bit for bit** (`f32::to_bits`, no tolerance).
The same test also asserts the rewrite still collapses, so it fails in both
directions.

`AppendScoringManifest` is an *appender*: it opens the already-generated index
read-only and rewrites only the `scoring.*` keys, so no fixture index was
regenerated and no segment id moved.

Seven further unit tests cover the rule itself: summed explicit boosts
(`a^2 a^3` -> `a^5`), nested `BoostQuery` chains multiplying before summing
(`(a^2)^3` -> 6), the boost-of-exactly-1 unwrap, the `SHOULD` gate, "no
duplicates means no rebuild and no reordering", and first-seen order.

### Verdict

Swept clean. Java's rewrite is now 14 of 14 rules; the two still absent
(`ConstantScoreQuery.rewrite`'s non-scoring simplification and "inline SHOULD
clauses from the only MUST clause") were already recorded before this batch and
are unchanged.

## Tier-2 review

Run on this batch's own diff after the gate was green. Two gating findings and
seven advisories; **one gating finding was a false positive and is worth
recording as such**, the other was real. All seven advisories are acted on.

**Gating 1 (rejected, and the code now says why).** The reviewer read
`AcceptDocs.BitsAcceptDocs`' `if (bits instanceof BitSet) this.maxDoc =
bitSet.cardinality()` and concluded that on a deleted segment Java's unfiltered
`cost()` is the *live* count, so `filteredDocCount` would be
`min(liveCount, graphSize)` and a heavily-deleted segment would drop to the
exhaustive scan where this port still walks the graph. Checked against the tag:
`Lucene90LiveDocsFormat.readLiveDocs` returns a `DenseLiveDocs`/`SparseLiveDocs`,
which `implements LiveDocs` (`interface LiveDocs extends Bits`) and are **not**
`BitSet`s, so that branch is not taken and `cost()` is the plain `maxDoc` — as
the code had it. The comment now states the `LiveDocs`-is-not-a-`BitSet` reason
explicitly, since the misreading is an easy one and the next reader deserves the
answer rather than the assertion.

**Gating 2 (real, fixed).** `per_leaf_top_k`'s comment claimed Java's
`Math.max(1, NaN)` returns 1 as Rust's `f64::max` does. It does not — Java
propagates the NaN and `(int) NaN == 0`, which Lucene's own comment says
(*"if we divided by zero above, leafProportion can be NaN and then this would
be 0"*, above `assert perLeafTopK > 0`). The behaviour is unchanged and
unobservable (an index with no documents has no vectors, so a collector of 0
and one of 1 both come back empty), but the whole point of finding 1's
bit-level framing is that this function's Java facts are trustworthy. The
comment and the test's comment now state the divergence and why `1` is the
better of the two.

**Advisories, all acted on.**

1. *`ef_search` leaked into Java's cost heuristic.* `LeafPlan` carried one
   `per_leaf_k` used for three different Java quantities, so a widened beam
   also raised the `cost <= perLeafTopK` and `scoreDocs.length >= perLeafTopK`
   thresholds — taking the exact branch where Java walks the graph. Harmless in
   result terms but a different *kind* of answer from a knob documented as only
   ever buying recall. Split into `per_leaf_top_k` (Java's threshold) and
   `collector_k` (widened), with the reason on the fields. Test:
   `a_wider_beam_still_honours_the_filter_and_never_scores_worse`, and
   `reentry_plan_restores_the_full_k` now pins both fields.
2. *The quoted per-leaf collector sizes were wrong for one leaf.* 2000/1000/960/40
   of 4000 at `k = 10` gives 30, 24, **24**, 5 (sum 83), not 23 / 82. Corrected
   in this report, `docs/parity.md` and the bench's module doc.
3. *A broken intra-doc link* to the deleted `check_similarity`. Fixed. (Checked
   with `RUSTDOCFLAGS="-D warnings" cargo doc --no-deps`: no unresolved link
   remains in this batch's files. The tool also reports a large pre-existing
   crop of `private_intra_doc_links` warnings across both crates, which is a
   repo-wide gate question, not this batch's.)
4. *The score-neutrality claim was only half-pinned*: the test scored the
   *un*-rewritten query and only checked that the rewrite contained a
   `BoostQuery` at all. It now scores **both** forms against the same recorded
   bits and asserts the rewrite actually changed the query, so a wrong summed
   boost fails.
5. *A `&dyn Fn` in the `O(size)` loop the batch itself calls out.* The
   `ord_to_doc` callback is now `impl Fn`, consistent with everything else in
   the module being monomorphized.
6. *The duplication mitigation overclaimed.*
   `the_unfiltered_dispatch_matches_the_codec_entry_point` pins the shared half
   only; the two parameters that motivated the copy have no equality partner.
   Restated precisely in finding 7 and on the function.
7. *A field in `.vemf` but not in `.vem`* surfaced as a decode error ("your
   index is corrupt") where Java raises `IllegalArgumentException`. Unreachable
   with Lucene-written files, but it is the same distinction finding 15 exists
   for, so it is now an `InvalidKnnQuery`.

The reviewer independently confirmed against the tag: the float/double split,
both `getLeafResults` branches including the `visitedLimit = cost + 1` cap and
the `scoreDocs.length >= perLeafTopK` fallback, `exactSearch`'s
`queueSize = min(k, cost)` and its sentinel-drain/tie-break equivalence, the
`Lucene99HnswVectorsReader.search` dispatch, the re-entry loop's two subtleties
(phase 2 replaces phase 1's list even when shorter; the merge runs over all
leaves), both `BooleanQuery.rewrite` dedup blocks including that
`Multiset.size()` counts duplicates and that Java's `should.size() == mSM`
promotion does not carry the threshold forward, `DEFAULT_FILTERED_SEARCH_THRESHOLD
== 0` making `useFilteredSearch` unreachable, and `shouldCreateGraph` leaving
the 40-document segment graphless.


## Carry-over

- **Seeded re-entry** (finding 6) — `SeededHnswGraphSearcher` /
  `KnnSearchStrategy.Seeded` in `crates/lucene-codecs/src/hnsw.rs`
  (`c10-vectors-wiring`).
- **`acceptOrds`/`filteredDocCount` on `hnsw_vectors::search`** (finding 7) —
  `crates/lucene-codecs/src/hnsw_vectors.rs` (`c10-vectors-wiring`). Deleting
  `vector_query::hnsw_search` is the follow-up.
- **Filtered KNN over the C ABI.** `lucene-ffi` passes `filter: None`: a
  filter query would need its own clause array at the boundary, which c13's
  occur-tagged format can now carry but which is a boundary-design change of
  its own. Stated on `vectors_input`.
- **A lazy accept set** (finding 12's sparse `O(size)` build) — needs a
  `Bits`-equivalent on `HnswGraphSearcher::search`, same file as finding 7.
