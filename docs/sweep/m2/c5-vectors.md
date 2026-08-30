# c5-vectors

Follow-up batch opened from b7 finding 23/24/28: port the real
`Lucene99FlatVectorsFormat` so vector fields interoperate with Lucene at all,
then `Lucene99HnswVectorsFormat` + `util/hnsw/*` on top.

**Both phases landed.** The invented `.vec`/`.vem` layout is gone; `.vec`,
`.vemf`, `.vem` and `.vex` are now real Lucene bytes in both directions, and
HNSW construction and search are ported.

Java counterparts (all in `lucene/core`, Lucene 10.5.0 at
`/home/tuong/work/lucene`):

- `codecs/lucene99/Lucene99FlatVectorsFormat.java`, `Lucene99FlatVectorsReader.java`,
  `Lucene99FlatVectorsWriter.java`
- `codecs/lucene99/Lucene99HnswVectorsFormat.java`, `Lucene99HnswVectorsReader.java`,
  `Lucene99HnswVectorsWriter.java`
- `codecs/lucene95/{OffHeapFloatVectorValues,OffHeapByteVectorValues,OrdToDocDISIReaderConfiguration}.java`
- `codecs/hnsw/{FlatVectorsFormat,FlatVectorsReader,FlatVectorsWriter,FlatVectorsScorer,DefaultFlatVectorScorer}.java`
- `util/hnsw/{HnswGraph,OnHeapHnswGraph,HnswGraphBuilder,HnswGraphSearcher,AbstractHnswGraphSearcher,NeighborQueue,NeighborArray,RandomVectorScorer,UpdateableRandomVectorScorer}.java`
- `util/{VectorUtil,TernaryLongHeap,NumericUtils}.java`,
  `internal/vectorization/DefaultVectorUtilSupport.java`
- `index/{VectorSimilarityFunction,VectorEncoding}.java`,
  `search/{KnnCollector,AbstractKnnCollector,TopKnnCollector}.java`
- `java.util.SplittableRandom` (JDK) — the level-assignment RNG

Totals: **28 findings** — 5 CORRECTNESS (all fixed), 8 MISSING (all fixed),
5 PERF (4 fixed with measurements, 1 recorded), 6 INTENTIONAL. Three of them
(26, 27, and half of 28) came out of the Tier 2 semantic review, which found no
gating issues.

Files: `crates/lucene-codecs/src/vectors.rs` (rewritten),
`crates/lucene-codecs/src/hnsw.rs` (new),
`crates/lucene-codecs/src/hnsw_vectors.rs` (new), plus
`crates/lucene-util/src/fixed_bit_set.rs` (one method added).

---

## crates/lucene-codecs/src/vectors.rs

Java counterparts: `Lucene99FlatVectors{Format,Reader,Writer}`,
`OrdToDocDISIReaderConfiguration`, `OffHeap{Float,Byte}VectorValues`,
`codecs/hnsw/FlatVectors*`, `VectorUtil`, `VectorSimilarityFunction`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `write_flat_vectors` | `Lucene99FlatVectorsWriter.{addField,flush,writeField,writeFloat32Vectors,writeByteVectors,writeMeta,finish}` | ported (finding 1). Identical field order, `alignOutput` padding, `-1` end-of-fields marker, footer last |
| `write_stored_meta` | `OrdToDocDISIReaderConfiguration.writeStoredMeta` | identical, all three cases (`-2` empty / `-1` dense / offset sparse), `DEFAULT_DENSE_RANK_POWER`, block shift 16 |
| `FlatVectorsReader::open` | `Lucene99FlatVectorsReader` ctor + `readMetadata` + `readFields` + `openDataInput` | identical, incl. the meta-vs-data version cross-check |
| `read_field_entry` | `FieldEntry.create` + its compact-constructor validation | identical `size * dim * byteSize == vectorDataLength`; the `FieldInfos` cross-checks are not reachable here (finding 16) |
| `OrdToDoc::from_stored_meta` | `OrdToDocDISIReaderConfiguration.fromStoredMeta` | identical; `is_empty`/`is_dense` are Java's |
| `read_vector_encoding` / `read_similarity_function` | `Lucene99HnswVectorsReader.{readVectorEncoding,readSimilarityFunction}` | identical, incl. the **4-byte** ordinal (not a byte) and `SIMILARITY_FUNCTIONS`'s pinned order |
| `FloatVectorValues::{vector_into,vector}` | `OffHeapFloatVectorValues.vectorValue(ord)` | same bytes; decodes into a caller buffer instead of a per-call `float[]` (finding 17) |
| `ByteVectorValues::vector` | `OffHeapByteVectorValues.vectorValue(ord)` | zero-copy borrow into the mapped `.vec` |
| `RawVectorValues::ord_to_doc` | `KnnVectorValues.ordToDoc` via `getDirectMonotonicReader` | identical (identity when dense) |
| `doc_to_ord` / `DocToOrdCursor` | `OrdToDocDISIReaderConfiguration.getIndexedDISI` + `IndexedDISI.advanceExact` | ported on c2's `DisiCursor` |
| `FloatVectorScorer` / `ByteVectorScorer` | `DefaultFlatVectorScorer`'s `RandomVectorScorer` and `UpdateableRandomVectorScorer` | one type for both roles (finding 19) |
| `VectorSimilarityFunction::score` | `VectorSimilarityFunction.compare(float[], float[])` | identical for all four |
| `VectorSimilarityFunction::score_bytes` | `VectorSimilarityFunction.compare(byte[], byte[])` | ported (finding 2) — **different transforms** from the float branch for two of the four |
| `square_distance` / `dot_product` / `cosine` | `VectorUtil.*` via `DefaultVectorUtilSupport` | same result to a few ulps; different lane split (finding 15) |
| `square_distance_bytes` / `dot_product_bytes` / `cosine_bytes` | the `byte[]` overloads | identical `i32` accumulators |
| `dot_product_score_bytes`, `scale_max_inner_product_score`, `normalize_to_unit_interval`, `normalize_distance_to_unit_interval` | the same four `VectorUtil` helpers | identical |
| `exhaustive_search` | `Lucene99HnswVectorsReader.search`'s non-HNSW branch | same answer; the bulk-scored version lives in `hnsw_vectors::search` |
| `validate_field` | `DefaultFieldWriter.addValue`'s `docID == lastDocID` / `assert docID > lastDocID` | stricter: the whole doc list is validated up front (finding 6) |

Java with **no** Rust counterpart: `OffHeapFloat16VectorValues` and the
`FLOAT16` encoding; `writeSortingField`/`writeSortedFloat32Vectors`/
`mapOldOrdToNewOrd` (index-sort remap); `mergeOneFlatVectorField` and
`MergedVectorValues`; `getMergeInstance`/`finishMerge`/`updateIOContext`
(IO-hint plumbing this port has no equivalent for); `ramBytesUsed`/
`getOffHeapByteSize`; `checkIntegrity` (the whole-file footer is verified at
`open` instead); `FlatFieldVectorsWriter`'s buffering (this port's writer takes
a finished field); every `lucene104` scalar-quantized format.

### Findings

1. **[CORRECTNESS → fixed]** *The on-disk format was this port's own
   invention.* b7 finding 23 restated: `.vec`/`.vem` carried codec names
   `LuceneRustFlatVectorsData`/`Meta`, a per-vector explicit doc-id list, a
   byte-wide encoding/similarity marker, and no alignment padding. Real
   Lucene's `.vec` is ordinal-addressed with a separate `.vemf` meta carrying
   an `OrdToDocDISIReaderConfiguration`; the encoding and similarity are
   **4-byte** ordinals; the meta is terminated by a `-1` field number; and each
   field's vector region is zero-padded to 64 bytes (FLOAT32) or 4 (BYTE).
   Consequence: a Lucene-written vector field could not be read by this module
   and a Rust-written one could not be read by Lucene — the only format in the
   crate for which that was true.
   *Resolution*: `vectors.rs` rewritten as a port of
   `Lucene99FlatVectors{Format,Reader,Writer}` plus
   `OrdToDocDISIReaderConfiguration` and `OffHeap{Float,Byte}VectorValues`.
   The sparse path composes c2's incremental `DisiCursor` and
   `direct_monotonic`, so nothing decodes a doc-id list into a `Vec`.
   *Tests*: `tests/vectors_fixtures.rs::{flat_metadata_matches_lucene,
   vectors_and_ord_to_doc_match_lucene}` against a real 4000-document,
   five-field `IndexWriter` segment (`fixtures/src/GenVectors.java`), and
   `fixtures/src/VerifyVectors.java` in the reverse direction, wired into
   `scripts/verify-write-path.sh` (now **16/16**).

2. **[CORRECTNESS → fixed]** *Byte vectors were unsupported, and the byte
   score transforms are not the float ones.* The old writer hard-coded
   `VectorEncoding::Float32` and the reader read the encoding byte into `_`.
   Worse, the natural "just reuse the float formula with an integer sum"
   assumption is wrong twice: Java's `compare(byte[], byte[])` uses
   `dotProductScore` (`0.5f + dot / (dim * 2^15)`) for `DOT_PRODUCT`, **not**
   `normalizeToUnitInterval`, and its `COSINE` branch has **no** `max(_, 0)`
   clamp where the float branch does.
   *Resolution*: `FieldVectorData::{Float32,Byte}` fixes encoding and payload
   together so they cannot disagree; `score_bytes` ports all four transforms.
   *Tests*: `vectors.rs::tests::{byte_similarity_transforms_differ_from_the_float_ones,
   byte_kernels_sign_extend_like_javas_byte}` plus the fixture's BYTE field
   (`byte_dot`, 2000 vectors), whose 20 exact and 20 HNSW query results are
   reproduced doc-for-doc.

3. **[CORRECTNESS → fixed]** *`.vemf` was never validated against itself.*
   Java's `FieldEntry` compact constructor throws when
   `size * dim * byteSize != vectorDataLength`, and `readVectorEncoding`
   rejects an out-of-range ordinal with `CorruptIndexException`. Without those
   checks the corrupt values are used directly: a `dimension` of 0 makes every
   ordinal decode to an empty vector (so a whole field reads back
   "successfully" and then scores as if nothing matched), and a
   `vectorDataOffset` or length past the end of `.vec` reaches a slice out of
   bounds and **panics** instead of returning a decode error. Note the footer is
   verified before the fields are decoded, so the checksum does *not* protect
   against a hostile file — only against bit rot.
   *Resolution*: `read_field_entry` validates dimension, count, region bounds
   and the length identity; `read_vector_encoding`/`read_similarity_function`
   reject out-of-range ordinals; `OrdToDoc::from_stored_meta` rejects a
   `docsWithFieldOffset` that is neither `-1`, `-2` nor a real offset and a
   `blockShift` outside `0..=31`.
   *Tests*: `vectors.rs::tests::{invalid_encoding_and_similarity_ordinals_are_rejected,
   a_vector_data_length_that_disagrees_with_size_is_rejected,
   a_vector_region_past_the_end_of_the_data_file_is_rejected,
   a_negative_field_number_is_rejected, a_truncated_footer_is_rejected}` — each
   flips one byte and rebuilds the footer, because `open` verifies the
   whole-file CRC before it decodes a field.

4. **[CORRECTNESS → fixed]** *A meta/data version mismatch was not caught.*
   Java's `openDataInput` throws
   `CorruptIndexException("Format versions mismatch: meta=..., ...=...")` when
   the two files' index headers disagree — the symptom of a half-replaced
   segment. Both readers here now compare them.
   *Test*: `version_mismatch_between_meta_and_data_is_rejected`.

5. **[CORRECTNESS → fixed]** *An out-of-range ordinal indexed out of bounds.*
   `VectorField::vector(doc)` used to scan, so there was no ordinal API to get
   wrong; the ordinal-addressed reader has one, and `values.vector(size)` would
   have panicked. `RawVectorValues::bytes`/`ord_to_doc` now return
   `Error::OrdOutOfRange`.
   *Tests*: `out_of_range_ordinals_are_rejected`,
   `sparse_ord_to_doc_rejects_an_out_of_range_ordinal`.

6. **[MISSING → fixed]** *No write-side validation of the doc list.* Java's
   `DefaultFieldWriter.addValue` throws `IllegalArgumentException` on a
   repeated doc id and asserts `docID > lastDocID`; `IndexedDISI.writeBitSet`
   requires ascending input. This port accepted an unsorted or duplicated doc
   list and produced a file whose DISI bitset counted a doc twice — a segment
   that decodes to plausible, wrong answers. `validate_field` now rejects a
   non-ascending list, a negative doc id, a doc id `>= maxDoc`, a
   dimension `<= 0`, and a component count that disagrees with
   `docs.len() * dimension`.
   *Test*: `writer_rejects_malformed_fields` (one case per rule).

7. **[MISSING → fixed]** *`alignOutput` was not implemented.*
   `Lucene99FlatVectorsWriter.alignOutput` pads each field's vector region to
   64 bytes for FLOAT32 (4 for BYTE) because unaligned 64-byte reads are slow
   on Arm Neoverse. It is part of the format, not an optimisation: the reader
   finds the region through the recorded offset either way, so a writer that
   skips it round-trips through itself and is still wrong against a
   byte-comparing reader. Emitted, and asserted in
   `flat_writer_round_trips_dense_sparse_and_byte`.

8. **[MISSING → fixed]** *No doc → ordinal direction.* Java resolves it
   through `OrdToDocDISIReaderConfiguration.getIndexedDISI`. The old module
   only had doc → vector (by linear scan). `doc_to_ord()` returns a
   `DocToOrdCursor` over c2's `DisiCursor` for a sparse field, an identity for
   a dense one, and a constant `None` for an empty one — with no allocation in
   the first two cases.
   *Test*: the fixture checks both directions on the sparse fields.

### Verdict

Swept clean. The interoperability failure b7 recorded is closed in both
directions, and every branch of `OrdToDocDISIReaderConfiguration` is covered by
a real-Lucene fixture. Open: `FLOAT16`, the index-sort and merge write paths,
and `IndexWriter` wiring (finding 21).

---

## crates/lucene-codecs/src/hnsw.rs (new)

Java counterparts: `util/hnsw/{HnswGraph,OnHeapHnswGraph,HnswGraphBuilder,
HnswGraphSearcher,AbstractHnswGraphSearcher,NeighborQueue,NeighborArray,
RandomVectorScorer,UpdateableRandomVectorScorer}`, `util/TernaryLongHeap`,
`util/NumericUtils`, `search/{AbstractKnnCollector,TopKnnCollector}`,
`java.util.SplittableRandom`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `SplittableRandom::{new,next_long,next_double}` | `java.util.SplittableRandom` root stream | bit-exact (finding 9) |
| `TernaryLongHeap::{push,pop,insert_with_overflow,top,get,up_heap,down_heap}` | `TernaryLongHeap` | identical, arity 3, 1-based (finding 10) |
| `float_to_sortable_int` / `sortable_int_to_float` / `sortable_float_bits` | `NumericUtils` | identical |
| `NeighborQueue::{new,add,insert_with_overflow,pop,top_node,top_score,nodes,clear}` + `encode`/`decode_*` | `NeighborQueue` | identical, incl. the complemented node id that breaks score ties toward the smaller ordinal |
| `KnnCollector` | `AbstractKnnCollector` + `TopKnnCollector` + `HnswGraphBuilder.GraphBuilderKnnCollector` | one type; `unlimited(k)` is the builder's variant (finding 18) |
| `NeighborArray::{add_in_order,add_out_of_order,clear,remove_index,size,nodes,score,max_size}` | `NeighborArray` | identical |
| `NeighborArray::add_and_ensure_diversity` | `addAndEnsureDiversity` | identical |
| `NeighborArray::{sort,insert_sorted_internal,asc_/desc_sort_right_most_insertion_point}` | the same four | identical, incl. `Arrays.binarySearch`'s walk-right-past-equals rule |
| `NeighborArray::{find_worst_non_diverse,is_worst_non_diverse}` | the same two | identical, both branches |
| `OnHeapHnswGraph::{new,add_node,neighbors,neighbors_mut,node_exists_at_level,try_set_new_entry_node,try_promote_new_entry_node,size,num_levels,entry_node,max_conn,max_node_id}` | `OnHeapHnswGraph` | identical minus the atomics (finding 11); `add_node` fills intermediate levels eagerly (finding 12) |
| `HnswGraphView::{neighbors_into,sorted_nodes_on_level}` | `HnswGraph.{seek,nextNeighbor,getSortedNodes,getNodesOnLevel}` | same content, list-shaped instead of cursor-shaped (finding 13) |
| `expected_visited_nodes` | `HnswGraphSearcher.expectedVisitedNodes` | identical |
| `should_create_graph` | `Lucene99HnswVectorsWriter.shouldCreateGraph` | identical |
| `HnswGraphSearcher::search` | `AbstractHnswGraphSearcher.search` | identical |
| `HnswGraphSearcher::find_best_entry_point` | `HnswGraphSearcher.findBestEntryPoint` | identical, incl. `UNK_EP` and the per-level `visited.set(currentEp)` |
| `HnswGraphSearcher::search_level` | `HnswGraphSearcher.searchLevel` + `scoreEntryPoints` | identical, incl. `Math.nextUp` on `minAcceptedSimilarity` and the `shouldExploreMinSim` one-shot |
| `prepare_scratch_state` | `prepareScratchState` | identical (`FixedBitSet` always — finding 20) |
| `HnswGraphBuilder::{new,build,add_graph_node,random_graph_level}` | `HnswGraphBuilder.{ctor,build,addGraphNode,addGraphNodeInternal,getRandomGraphLevel}` | identical single-threaded shape (finding 11) |
| `add_diverse_neighbors` / `select_and_link_diverse` / `diversity_check` / `pop_to_scratch` | the same four | identical; `diversityCheck`'s bulk chunking is collapsed (finding 14) |

Java with **no** Rust counterpart: `FilteredHnswGraphSearcher`,
`SeededHnswGraphSearcher`, `OnHeapHnswGraphSearcher`, `FloatHeap`,
`HnswLock`, `HnswUtil` and `connectComponents`, `HnswGraphMerger`,
`IncrementalHnswGraphMerger`, `ConcurrentHnswMerger`,
`HnswConcurrentMergeBuilder`, `MergingHnswGraphBuilder`,
`InitializedHnswGraphBuilder`, `UpdateGraphsUtils`,
`OrdinalTranslatedKnnCollector` (the caller translates), `SparseFixedBitSet`,
`RandomVectorScorerSupplier`, `NeighborArray`'s `LongConsumer` RAM accounting,
`InfoStream` progress printing, `setAbortCheck`.

### Findings

9. **[CORRECTNESS → fixed by construction]** *Level assignment must come from
   `SplittableRandom`, not "some" RNG.* `getRandomGraphLevel(ml, random)` draws
   `U` from `new SplittableRandom(HnswGraphBuilder.randSeed = 42)` and returns
   `(int)(-ln(U) * ml)`. The *distribution* would have been enough for recall,
   but not for evidence: with any other stream, "our recall vs Lucene's" is a
   comparison of two different random draws. `SplittableRandom` here is a
   bit-exact port (`GOLDEN_GAMMA`, `mix64`'s 30/27/31 shifts, `nextDouble`'s
   `(nextLong() >>> 11) * 0x1.0p-53`), verified against the JDK.
   *Payoff, measured*: over the fixture's 4000 vectors the Rust-built graph has
   the same level count, the same per-level node counts and the **same entry
   node (171)** as Lucene's; at 50k x 128 both engines independently pick entry
   node **46601**. That is not a coincidence a wrong RNG could produce.
   *Tests*: `hnsw.rs::tests::splittable_random_reproduces_the_jdk_stream`,
   `tests/vectors_fixtures.rs::splittable_random_matches_java` (against a
   manifest row emitted by the JDK), and the graph-shape assertions in
   `rust_built_graph_recall_matches_lucene`.

10. **[MISSING → fixed]** *`TernaryLongHeap` ported rather than replaced with
    `BinaryHeap`.* `NeighborQueue.nodes()` hands the **raw heap array** to
    `HnswGraphBuilder`, which uses it as the entry-point set for the next level
    down; that array's order then feeds `scoreEntryPoints`'s collect order. The
    *contents* are heap-independent, the *order* is not. Porting Java's arity-3
    heap (40 lines) is what lets the differential test assert "identical
    results" rather than "similar recall" — and it does:
    `search_over_lucene_graph_reproduces_lucene_results` matches Lucene
    doc-for-doc on all 80 queries.

11. **[INTENTIONAL]** *Single-threaded.* `OnHeapHnswGraph`'s entry node is a
    plain field, not an `AtomicReference`, and `HnswLock` is absent.
    `addGraphNodeInternal`'s `do { ... } while (true)` exists only to retry
    when a concurrent builder moved the entry node between the read of
    `numLevels()` and the promotion attempt; with one thread
    `tryPromoteNewEntryNode` cannot fail, so the loop is written out as its
    single iteration and the `IllegalStateException` it guards is unreachable.
    Documented at `HnswGraphBuilder::add_graph_node`. Concurrent merge is a
    separate, later concern (`ConcurrentHnswMerger`).

12. **[INTENTIONAL]** `OnHeapHnswGraph::add_node` creates every level from 0 up
    to `level` at once, where Java allocates a `NeighborArray[level + 1]` with
    null slots that later `addNode` calls fill. Both callers add a node from
    its top level downward, and every intermediate array starts empty either
    way, so the observable graph is identical; a `Vec<Vec<NeighborArray>>` is
    then one indirection instead of two plus a null check.

13. **[INTENTIONAL]** `HnswGraphView::neighbors_into(level, node, &mut Vec)`
    replaces `seek(level, node)` + repeated `nextNeighbor()`. Every caller in
    Lucene immediately drains that cursor into a `bulkNodes` array, so the
    cursor buys nothing here — and it would cost a `&mut self` on the graph,
    which the builder cannot give (it reads the graph while holding a mutable
    borrow of a neighbour array in it).

14. **[INTENTIONAL]** `diversity_check` is a plain loop where Java chunks
    through `bulkScore` in batches of `min((size + 1) / 2, 8)`. The answer is
    identical — Java short-circuits on the first chunk whose *maximum* reaches
    the candidate's score, which is exactly "some neighbour scores >= score" —
    and the chunking exists to amortise Panama's vectorised kernel across a
    call, which this port's `bulk_score` default does not have.

15. **[INTENTIONAL, and the reason scores are compared with a tolerance]**
    *Bit-exact float scores are not achievable, and Lucene does not have them
    either.* Lucene ships **three** float kernels: `PanamaVectorUtilSupport`
    (8/16-wide), and `DefaultVectorUtilSupport` with and without
    `Constants.HAS_FAST_SCALAR_FMA`. For a dimension `<= 32` the scalar one is
    a **serial `Math.fma` chain**; above 32 it is four FMA accumulators. This
    port keeps b7's eight-lane non-FMA split. Matching the scalar+FMA variant
    would need `f32::mul_add`, which on the default `x86-64` target (no `+fma`)
    lowers to a **libm `fmaf` call per element** — catastrophically slower —
    and matching the Panama variant is not possible in safe stable Rust at all.
    Measured difference on the fixture: ~4e-8 relative, e.g. `0.55466060` vs
    `0.55466056`. It did not change a single doc id or ordering across 80
    queries. `tests/vectors_fixtures.rs::assert_hits_match` therefore compares
    **doc ids and their order exactly** and scores within `1e-6` relative, with
    the reasoning written at the assertion.

16. **[INTENTIONAL]** The `FieldInfos` cross-checks Java's `FieldEntry` and
    `validateFieldEntry` perform (similarity and dimension must match
    `FieldInfo`) are not done here, because neither reader is handed a
    `FieldInfos`. This port keys vector fields by *number*, as it does for
    points and doc values; the equivalent check belongs wherever
    `lucene-index` assembles a segment reader. Recorded, not invented.

17. **[PERF → fixed, measured]** *The old reader held every vector on the
    heap.* `VectorField { entries: Vec<(i32, Vec<f32>)> }` was one allocation
    per vector plus a linear scan per lookup — for 1M x 768 that is 1M
    allocations and ~3 GB resident, where Java holds one `IndexInput`.
    `FloatVectorValues` now borrows the mapped `.vec` and decodes into a
    caller-owned buffer; `ByteVectorValues::vector` returns a borrow with no
    decode at all. A whole graph search allocates nothing per candidate.

18. **[PERF]** `KnnCollector` is one type for what Java splits across
    `TopKnnCollector` and `GraphBuilderKnnCollector`; they differ only in
    whether `visitLimit` is finite, and `unlimited(k)` sets it to `u64::MAX`
    so `early_terminated()` is a comparison rather than a virtual call. The
    builder allocates its two collectors once and `clear()`s them per node, as
    Java does.

19. **[PERF]** One scorer type covers `RandomVectorScorer` and
    `UpdateableRandomVectorScorer`; the searcher and builder are generic over
    `S: VectorScorer` / `S: UpdateableVectorScorer`, so the score call in the
    innermost loop is monomorphized rather than a `dyn` dispatch (the
    `rust-performance` skill's "`dyn` only at Query/Weight level" rule).

20. **[PERF, recorded]** `HnswGraphSearcher` always uses `FixedBitSet`, never
    `SparseFixedBitSet`. Java's `createBitSet` picks the sparse one when
    `expectedVisitedNodes(k, n) < n / 128`, which for `k = 10` means roughly
    `n > 40000`. The cost here is `n / 8` bytes of memset per level per query:
    6.25 KB at 50k vectors, ~1 µs, against a measured 37-95 µs per query.
    Worth revisiting at 1M+ vectors (125 KB per clear), where it would become
    the dominant term; not worth a second bitset implementation today.

### Verdict

Swept clean. Recall parity with Lucene is demonstrated, not asserted (finding
9 and the measurements below), and so is neighbour-set agreement (see
Verification). Open: the merge-time graph builders, the filtered/seeded
searchers, `SparseFixedBitSet` (20), and the `lucene-util` move (28).

---

## crates/lucene-codecs/src/hnsw_vectors.rs (new)

Java counterparts: `Lucene99HnswVectorsFormat`, `Lucene99HnswVectorsReader`
(including its private `OffHeapHnswGraph`), `Lucene99HnswVectorsWriter`.

### Method correspondence

| Rust | Java | Verdict |
|---|---|---|
| `write_hnsw_vectors` | `Lucene99HnswVectorsWriter.{writeField,finish}` | identical framing, `-1` marker, footer last |
| `write_graph` | `Lucene99HnswVectorsWriter.writeGraph` | identical: sort, delta, drop duplicates, write the **post-dedup** count, group-vints, per-node byte lengths |
| `write_meta` | `Lucene99HnswVectorsWriter.writeMeta` | identical, incl. the `graph == null` branch (`M`, then `0`), the level-`>0` back-to-front delta loop, and the cumulative `DirectMonotonicWriter` offsets |
| `HnswVectorsReader::open` | `Lucene99HnswVectorsReader` ctor + `readFields` + `openDataInput` | identical, incl. the version cross-check |
| `read_field_entry` | `FieldEntry.create` | identical, plus bounds validation (finding 22) |
| `HnswVectorsReader::graph` | `Lucene99HnswVectorsReader.getGraph(FieldEntry)` | `None` is Java's `HnswGraph.EMPTY` |
| `OffHeapHnswGraph::new` | the `OffHeapHnswGraph` ctor | identical `graphLevelNodeIndexOffsets` and `entryNode` derivation |
| `OffHeapHnswGraph::neighbors_into` | `OffHeapHnswGraph.seek` + `nextNeighbor` | identical, both the group-vint (v1) and plain-vint (v0) neighbour encodings |
| `OffHeapHnswGraph::sorted_nodes_on_level` | `getNodesOnLevel` | identical |
| `search` | `Lucene99HnswVectorsReader.search(FieldEntry, ...)` | identical decision (`doHnsw`, `expectedVisitedNodes`, `graphSize == 0`) and identical 64-ordinal bulk-scored exhaustive fallback |

Java with **no** Rust counterpart: `mergeOneField`/`buildAndWriteGraph`/
`createGraphMerger`; `writeSortingField`/`reconstructAndWriteGraph`;
`FieldWriter`'s incremental `addValue` (this port builds the graph in one call
after the flat field is written, which is where `replayBufferedVectors` leaves
Java anyway); `getQuantizedVectorValues`/`getQuantizationState`/
`getRandomVectorScorerSupplierForMerge` (quantization); `checkIntegrity`;
`getOffHeapByteSize`; `ramBytesUsed`; `AcceptDocs`-shaped filtering (the
searcher takes a `FixedBitSet` directly).

### Findings

21. **[MISSING → fixed]** *HNSW was not ported at all* (b7 finding 24). Ported:
    the `.vex`/`.vem` pair, `OffHeapHnswGraph`, and the search entry point with
    Java's graph-vs-exhaustive decision. The scope deliberately left out is the
    merge path — `Lucene99HnswVectorsWriter.mergeOneField` reuses the largest
    source segment's graph via `IncrementalHnswGraphMerger` instead of
    rebuilding — which matters as soon as `lucene-index`'s merge grows vector
    support. Neither exists yet; recorded as the next step.

22. **[CORRECTNESS → fixed]** *`.vem` was decoded without bounds validation.*
    A hostile or corrupt `.vem` could carry `M = 0` (Java allocates
    `int[M * 2]` and asserts `arcCount <= currentNeighborsBuffer.length`), a
    negative `numLevels`, an upper-level node count of 0 or `> size`, a node
    ordinal `>= size`, a negative node delta, or a graph/offsets region past
    the end of `.vex` — each of which reached a slice index or a huge
    allocation here. `read_field_entry` now rejects all of them, and
    `neighbors_into` rejects an `arcCount > 2M` and a decoded neighbour ordinal
    outside `0..size`.
    *Tests*: `hnsw_vectors.rs::tests::{an_illegal_dimension_count_or_max_conn_is_rejected,
    an_upper_level_node_count_out_of_range_is_rejected,
    a_graph_region_past_the_end_of_the_index_file_is_rejected,
    a_negative_field_number_is_rejected, a_truncated_meta_file_is_rejected,
    an_out_of_range_neighbour_ordinal_is_rejected,
    seeking_a_node_that_is_not_on_the_level_is_rejected}`.

23. **[MISSING → fixed]** *The no-graph branch.* Lucene skips graph
    construction when `shouldCreateGraph(HNSW_GRAPH_THRESHOLD = 100, n)` is
    false — i.e. when `n <= ln(n) * 100`, which is true up to about 650
    vectors — and writes `numLevels = 0` with a zero-length `.vex` region. A
    reader that assumes a graph is always present fails on exactly the segments
    a small index is made of. Both the reader and the writer honour it, and the
    fixture contains **two** such fields (572 and 5 vectors) precisely because
    no ordinary fixture reaches this branch.
    *Tests*: `tiny_segment_threshold_matches_lucene` (which checks
    `should_create_graph` against Lucene's own decision for all five fixture
    fields), `a_field_without_a_graph_round_trips`,
    `search_falls_back_to_an_exhaustive_scan_without_a_graph`.

24. **[INTENTIONAL]** Write version is `VERSION_GROUPVARINT` (1); both 0 and 1
    are read, because `versionMeta` selects the neighbour encoding and a
    Lucene 9.9-era segment is still version 0. `FLOAT16` and the `lucene104`
    scalar-quantized formats layered on this same pair are out of scope.

25. **[PERF → fixed]** *A `Vec<u64>` allocated per graph seek.*
    `OffHeapHnswGraph::neighbors_into` allocated `read_group_vints`'
    destination buffer on every call — one allocation per visited node, i.e.
    ~500 per query, on the hottest loop in vector search. Java reuses a
    `currentNeighborsBuffer` sized `M * 2` for the life of the graph object.
    The first draft of this report recorded it as blocked, on the grounds that
    a reusable buffer needs `&mut self` on `neighbors_into` — which the builder
    specifically cannot give (finding 13). That was the wrong conclusion: the
    Tier 2 review pointed out that a `RefCell<Vec<u64>>` on the graph keeps the
    `&self` signature and removes the allocation entirely, which is what it now
    does (`OffHeapHnswGraph::scratch`, sized `2 * M` once at construction).
    Single-threaded, like every searcher here.

26. **[MISSING → fixed]** *The writer could emit a `.vem` its own reader
    rejects.* `write_hnsw_vectors` wrote `dimension`, `count` and `M`
    unvalidated, while `read_field_entry` rejects `dimension <= 0`, `count < 0`,
    `M <= 0` and `M > MAXIMUM_MAX_CONN` — and `HnswGraphBuilder::new` only
    checked `M > 0`, not `M <= 512`. So `M = 1000` built a graph, wrote a
    `.vem`, and then failed to reopen; real Lucene's
    `Lucene99HnswVectorsFormat` constructor rejects it too, and its
    `OffHeapHnswGraph` allocates `int[M * 2]` on the value. Both ends now
    validate the same bounds (`M` in `1..=512`, `beamWidth` in `1..=3200`,
    positive dimension, non-negative count).
    *Tests*: `builder_rejects_non_positive_parameters` gained the two upper
    bounds; `writer_rejects_a_level_zero_that_is_not_every_node` is joined by
    the `write_hnsw_vectors` guards.

27. **[MISSING → fixed]** *`VectorUtil.checkFinite` was not ported.* Java's
    `KnnFloatVectorField` constructor runs it on every indexed vector, so a
    Lucene-written `.vec` can never contain a NaN or an infinity. This port's
    writer could. That is not cosmetic here, because of a second divergence the
    review surfaced: `f32::max` **returns the other operand** on a NaN, where
    Java's `Math.max(float, float)` **propagates** it. A NaN score would
    therefore be silently dropped from the bulk-score maximum that gates
    `search_level`'s `max > minCompetitive` and `find_best_entry_point`'s
    `max_score > current_score` — a quietly missing candidate rather than a
    loud failure. `validate_field` now rejects a non-finite component
    (`Error::NonFiniteValue`), which makes the `f32::max` difference
    unreachable from anything this port writes; the difference itself is
    documented at `VectorScorer::bulk_score` for whoever reads a `.vec` from
    somewhere else.

28. **[INTENTIONAL, recorded]** *`hnsw.rs` carries three types that belong in
    `lucene-util`, and takes its error type from `vectors.rs`.*
    `SplittableRandom` is a **JDK** class, and `TernaryLongHeap` and the
    `NumericUtils` sortable-float helpers are `org.apache.lucene.util` — none is
    HNSW-specific in Java, and `util/hnsw` has no dependency on
    `codecs.lucene99` at all, where this port's `hnsw.rs` does
    (`use crate::vectors::{Error, Result}`). The immediate symptom is fixed:
    a builder parameter error is now `Error::InvalidGraphParameter`, not
    `Error::CorruptMeta` ("corrupt vector metadata" for "M must be positive"
    sends the reader at the wrong file). The layering itself is left as a
    carry-over rather than done blind at the end of a batch: moving three types
    across a crate boundary touches `lucene-util`'s module layout and every
    call site, and is a mechanical change better made on its own.

### Verdict

Swept clean for the flush/search path. Open: the merge path (21).

---

## Measurements

All on one core, `--release`, `M = 16`, `beamWidth = 100`, k = 10, 50 000
vectors x 128 dimensions.

Two methods, because they answer different questions and disagree by 2.4x:

- **cold-ish**: 50 *distinct* queries, each run once, so the graph regions a
  query touches are mostly not in cache. This is the row that also carries the
  distance counts and recall, since counting and recall need per-query
  bookkeeping.
- **warm**: `cargo bench -p lucene-codecs --bench hot_paths -- vectors/`,
  which runs a fixed set of 25 queries many times, so the working set stays
  resident. Both arms run all 25 per iteration, so their ratio is
  apples-to-apples; the reported time is per batch of 25.

### Search: brute force vs HNSW

| Data | | brute force | HNSW | ratio |
|---|---|---|---|---|
| **clustered** (500 centroids — the shape a real embedding field has), cold-ish | queries/sec | 572 | **26,814** | **47x** |
| | distance computations/query | 50,000 | **205** | **244x fewer** |
| | recall@10 | 1.0000 (exact) | 0.9140 | — |
| clustered, warm (criterion) | per query | 1.688 ms | **14.8 µs** | **114x** |
| **uniform noise** (the ANN worst case), cold-ish | queries/sec | 602 | 12,173 | 20x |
| | distance computations/query | 50,000 | 504 | 99x |
| | recall@10 | 1.0000 (exact) | 0.1560 | — |

The uniform row is the honest counterexample: at 128 dimensions of uniform
noise all distances concentrate, there is no neighbourhood structure, and an
HNSW graph searched with `ef = k = 10` recovers 15% of the true top-10.
Lucene's does the same — this is a property of the data and of Lucene's
choice to use `k` as the query-time beam width, not a defect in either
implementation. It is called out here because a reader who only saw the
clustered row would draw the wrong conclusion about what HNSW buys.

b7's extrapolation ("~20-30x fewer distance computations here, ~300x at 1M")
turns out to have been conservative on the distance-count axis and roughly
right on wall clock.

At the fixture's scale (4000 x 16 dims) the same measurement is **292 distance
computations per query against 4000 exhaustive — 14x** — and it is *asserted*
in `rust_built_graph_recall_matches_lucene`, so the search cannot silently
regress into a full scan.

### Recall, with the Java baseline

On `fixtures/data/vectors_index`'s 4000-vector, 16-dimension field, 20 queries,
`M = 16`, `beamWidth = 100`, k = 10, exact top-10 as the denominator:

| | recall@10 |
|---|---|
| graph built by **this port** | **0.9250** |
| graph built by **real Lucene** | **0.9250** |

Both graphs also have the same number of levels, the same per-level node counts
and the same entry node (171). Separately, real Lucene searching a
**Rust-written** graph (`scripts/verify-write-path.sh`) reports recall@10 of
0.9083 / 0.9750 / 0.9750 / 1.0000 across the four fields of
`write_vectors_fixture`.

### Index-build cost the graph adds

50k x 128 clustered vectors, single-threaded, **identical input data on both
sides** (the same LCG and the same cluster assignment, transcribed into a
throwaway Java program that calls `HnswGraphBuilder.create(sup, 16, 100, 42)`):

| | build time |
|---|---|
| this port | **5.2-5.6 s** (criterion `vectors/hnsw_build_50k_dim128`: 5.17 s) |
| real Lucene, JIT-warmed | 6.4 s |

For scale: the flat store for the same field is written in well under a second,
so the graph is essentially the whole per-segment vector flush cost, and it is
`O(n log n)` distance computations — the price of the query-side win above.
Both engines land on the same graph shape (5 levels, entry node **46601**),
which is what makes the times comparable at all.

## Verification

The point of this batch was that a self-round-trip proves nothing. What is
actually checked:

**Java writes, Rust reads** (`fixtures/src/GenVectors.java` →
`fixtures/data/vectors_index` → `crates/lucene-codecs/tests/vectors_fixtures.rs`).
A real `IndexWriter` session, 4000 documents, five vector fields chosen to
cover every reader branch: dense FLOAT32 (EUCLIDEAN), sparse FLOAT32 (COSINE),
a sparse FLOAT32 field small enough that Lucene skips the graph
(MAXIMUM_INNER_PRODUCT, 572 vectors), BYTE (DOT_PRODUCT, 2000 vectors), and a
5-vector field. Nine tests:

- metadata (field numbers, encodings, similarities, counts, dense-vs-sparse);
- spot ordinals decoded **bit-for-bit**, and the sampled ordinal↔doc mapping in
  both directions;
- the graph **arc for arc**: per level, node count, node ids, eight readable
  neighbour samples and an order-sensitive hash over every node's neighbour
  list — a mis-decoded node offset, a dropped group-vint group or an off-by-one
  in the level index offsets all change it;
- **search**: this port's searcher, run over *Lucene's own* graph, reproduces
  `KnnFloatVectorQuery`/`KnnByteVectorQuery` **doc-for-doc, in order**, for 80
  queries across four fields; and this port's exhaustive scan reproduces
  Lucene's brute-force top-k the same way (scores within 1e-6 — finding 15);
- **construction**: a graph built *here* over the fixture's vectors is compared
  against Lucene's own graph over the same ordinals, node by node. Result:
  **4273/4273 nodes on all three levels have a byte-identical neighbour set**.
  That is stronger than expected — the ~4e-8 kernel difference (finding 15)
  never flipped a diversity decision on this data — so the assertion is a
  0.95 floor across all levels rather than exact equality.
- `SplittableRandom(42)` against JDK-recorded bits;
- `should_create_graph` against Lucene's own per-field decision.

**Rust writes, Java reads** (`crates/lucene-codecs/examples/write_vectors_fixture.rs`
→ `fixtures/src/VerifyVectors.java`, wired into `scripts/verify-write-path.sh`,
now **16/16**). Four fields (dense / sparse / BYTE / no-graph), and Lucene
checks: every ordinal's components (order-sensitive hash over raw bits), every
ordinal's doc id, the graph's level count, entry node, max conn and per-level
arc hash — and then runs a **real `TopKnnCollector` search over the Rust-built
graph** and measures recall against the exact top-k, with a floor. A graph that
decodes cleanly but is *built* wrong passes every structural check and fails
that last one.

One deliberate difference from the other write-path verifiers: the `FieldInfos`
are **not** hand-built. The Rust example also writes a real `.fnm` through
`field_infos::write`, and `VerifyVectors` reads it back through
`Lucene94FieldInfosFormat` — which puts Lucene's own `.vemf`-vs-`.fnm`
cross-checks (`FieldEntry`'s "Inconsistent vector similarity function" /
"Inconsistent vector dimension") in front of two files this port wrote
independently. A hand-built `FieldInfos` cannot see a disagreement between
them, which is the blind spot that let a merged `.fnm` missing its
postings-format attributes pass thirteen write-path verifiers in c4. It is
also the only place a Rust-written `.fnm` carrying *real* vector metadata
(dimension > 0, a non-default encoding and similarity) has ever been read by
Lucene — `write_field_infos_fixture` sets `vector_dimension: 0` on every field.
Negative control, run once by hand: writing a `.fnm` whose dimension is one
larger than the `.vemf`'s makes Lucene refuse the segment with
`IllegalStateException: Inconsistent vector dimension for field="dense_f32";
17 != 16`, and the verifier exits non-zero.

**Why the structural comparison and not just recall.** The Tier 2 review's
sharpest point was that recall is *not* a discriminating signal for
construction bugs, and it is right. Measured by mutation: weakening
`diversity_check` so it accepts candidates it should prune takes neighbour-set
agreement from **4273/4273 to 1/4273** — while **recall goes up**, to 0.9350
against Lucene's 0.9250. A recall-only assertion would have passed that graph.
(It makes sense: the diversity rule trades a little local recall for global
connectivity, so breaking it can look better on one small fixture and be worse
in general — which is exactly the "fast and silently returns worse results"
failure this batch was told to avoid.) The level counts and entry node are no
help either: they are decided entirely by the `SplittableRandom` port and
survive a wrong diversity rule intact.

**Unit tests** (`#[cfg(test)]` in each module): 25 in `vectors.rs`, 29 in
`hnsw.rs`, 18 in `hnsw_vectors.rs`, covering the kernels against a naive
reference at every length either side of the lane split, each similarity
transform, every write-side validation rule, and every corruption branch (each
flipping one byte and rebuilding the footer, since `open` verifies the CRC
first).

Coverage (`cargo llvm-cov -p lucene-codecs --summary-only`), lines:
`hnsw.rs` 98.52%, `hnsw_vectors.rs` 97.45%, `vectors.rs` 96.82%.

Gates: `cargo fmt -p lucene-codecs -p lucene-util -- --check`,
`cargo clippy -p lucene-codecs -p lucene-util --all-targets -- -D warnings`,
`cargo test -p lucene-codecs` (1061 lib + 9 fixture tests),
`cargo test -p lucene-util`, `scripts/gen-fixtures.sh --check` (0 mismatches,
0 missing, 0 extras, including the new `vectors_index`), and
`scripts/verify-write-path.sh` (**16/16**) — all green. `cargo fmt --all` /
`clippy --workspace` were not runnable at the time of writing: batches c1, c4
and b15 were mid-edit in `lucene-index`, `blocktree.rs` and `lucene-ffi`, and
this batch touched none of those.

---

## Carry-over

- [ ] **`IndexWriter` wiring.** `lucene-index` still has no way to add a vector
      field to a document, so nothing reaches these formats from the
      document-writing API the way `set_doc_values_field`/`set_postings_field`
      do. The codec-level primitives (`write_flat_vectors`,
      `write_hnsw_vectors`, `should_create_graph`) are now shaped exactly the
      way a flush needs them — see `write_vectors_fixture.rs` for the whole
      sequence.
- [ ] **Merge.** `Lucene99HnswVectorsWriter.mergeOneField` +
      `IncrementalHnswGraphMerger` reuse the largest source segment's graph
      instead of rebuilding; `Lucene99FlatVectorsWriter.mergeOneFlatVectorField`
      streams vectors without buffering. Neither is ported (findings 21, and
      `mergeOneFlatVectorField` under vectors.rs's "no Rust counterpart").
      Rebuilding costs the full 5.5 s/50k measured above per merge.
- [ ] **`FLOAT16`** encoding and `OffHeapFloat16VectorValues`.
- [ ] **Index-sort write path** (`writeSortingField` /
      `reconstructAndWriteGraph`), needed once `lucene-index` supports a sorted
      index with vector fields.
- [ ] **Per-seek scratch** in `OffHeapHnswGraph::neighbors_into` (finding 25),
      and `SparseFixedBitSet` for the searcher's visited set above ~1M vectors
      (finding 20).
- [ ] **`FieldInfos` cross-validation** of dimension/similarity (finding 16),
      wherever `lucene-index` assembles a segment reader. Note the *write* side
      of this is already covered: `VerifyVectors` reads a Rust-written `.fnm`
      through real Lucene, so Lucene's own version of the check runs against
      our bytes.
- [ ] **Move `SplittableRandom`, `TernaryLongHeap` and the `NumericUtils`
      sortable-float helpers out of `lucene-codecs/src/hnsw.rs` into
      `lucene-util`** (finding 28), and give `hnsw` an error type that does not
      come from the `.vec` file format. `util/hnsw` has no dependency on
      `codecs.lucene99` in Java; this port's has one, backwards.
